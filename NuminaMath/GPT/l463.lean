import ComplexAnalysis.Complex.Bounds
import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.WithOne
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Expression
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.Special_Functions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Analysis.Trigonometry.Complex
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Binomial
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Module
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Logarithm
import Mathlib.Data.Set.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.NumberTheory.PrimeCounting
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Algebra.Field
import Real
import tactic

namespace blue_face_area_factor_l463_463798

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l463_463798


namespace boat_speed_in_still_water_l463_463428

-- Definitions for conditions
def speed_of_current : ℝ := 6
def distance_downstream : ℝ := 10.67
def time_downstream : ℝ := 20 / 60 -- Convert 20 minutes to hours

-- Theorem statement
theorem boat_speed_in_still_water :
  ∃ b : ℝ, distance_downstream = (b + speed_of_current) * time_downstream → b = 26.01 :=
by {
  -- This will provide the necessary structure
  sorry
}

end boat_speed_in_still_water_l463_463428


namespace geometric_sequence_common_ratio_l463_463694

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 - 3 * a 2 = 3 * a 2 - 5 * a 1)
  (h2 : ∀ n, a n > 0)
  (h3 : ∀ n, a n < a (n + 1))
  (h4 : ∀ n, a (n + 1) = a n * q) : 
  q = 5 :=
  sorry

end geometric_sequence_common_ratio_l463_463694


namespace probability_of_playing_exactly_one_is_0_12_l463_463063

def total_people : ℕ := 800
def fraction_play_at_least_one : ℚ := 1 / 5
def people_play_two_or_more : ℕ := 64

def number_play_at_least_one : ℕ := (fraction_play_at_least_one * total_people : ℚ).toNat
def number_play_exactly_one : ℕ := number_play_at_least_one - people_play_two_or_more
def probability_play_exactly_one : ℚ := number_play_exactly_one / total_people

theorem probability_of_playing_exactly_one_is_0_12 :
  probability_play_exactly_one = 0.12 :=
sorry

end probability_of_playing_exactly_one_is_0_12_l463_463063


namespace fraction_difference_l463_463769

theorem fraction_difference (a b : ℝ) : 
  (a / (a + 1)) - (b / (b + 1)) = (a - b) / ((a + 1) * (b + 1)) :=
sorry

end fraction_difference_l463_463769


namespace range_a_sufficient_not_necessary_l463_463606

theorem range_a_sufficient_not_necessary (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, (x ^ 2 - 4 * a * x + 3 * a ^ 2 < 0) → (|x - 3| > 1)) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2 / 3) :=
sorry

end range_a_sufficient_not_necessary_l463_463606


namespace total_rope_is_150_l463_463982

-- Define the lengths of the ropes for each post according to the given conditions
def rope_length (n : Nat) : Int :=
  match n with
  | 1 => 24
  | k + 1 => if (k + 1) % 2 == 0 then rope_length k - 2 else rope_length k + 4

-- Total rope length calculation for the 6 posts
def total_rope_length : Int :=
  List.sum (List.map rope_length [1, 2, 3, 4, 5, 6])

-- The theorem stating the problem
theorem total_rope_is_150 : total_rope_length = 150 := by
  sorry

end total_rope_is_150_l463_463982


namespace geometry_problem_l463_463249

-- Definitions of the conditions
def eccentricity (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (sqrt (a^2 - b^2) / a = 1 / 2)
def hyperbola_distance (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (2 * sqrt (4 * a^2 / b^2 + a^2) / 2 = 4 * sqrt 21 / 3)

-- Equations of the ellipse and hyperbola to be proven
def ellipse_eq (a b : ℝ) := (x y : ℝ) → x^2 / a^2 + y^2 / b^2 = 1
def hyperbola_eq (a b : ℝ) := (x y : ℝ) → y^2 / a^2 - x^2 / b^2 = 1

-- Function to calculate area of triangle F1PQ
def area_F1PQ (a b x1 x2 y1 y2 : ℝ) : ℝ := 
  let F1 := (-1 : ℝ, 0 : ℝ)
  let F2 := (1 : ℝ, 0 : ℝ)
  let P := (x1, y1)
  let Q := (x2, y2)
  let d_F1_PQ := dist(F1,P) * dist(P,Q)
  0.5 * dist(P,Q) * dist(F1, F2)

-- The statement that captures the mathematical equivalent of the proof
theorem geometry_problem (a b x1 x2 y1 y2 : ℝ) 
  (h_eccentricity : eccentricity a b)
  (h_hyperbola_distance : hyperbola_distance a b)
  (h_intersection_points : x1 = 1 ∧ x2 = 1 ∧ y1 = 4 * sqrt 3 / 3 ∧ y2 = -4 * sqrt 3 / 3) :
  ellipse_eq a b = (λ x y, x^2 / 4 + y^2 / 3 = 1) ∧
  hyperbola_eq a b = (λ x y, y^2 / 4 - x^2 / 3 = 1) ∧
  area_F1PQ a b x1 x2 y1 y2 = 8 * sqrt 3 / 3 := 
sorry

end geometry_problem_l463_463249


namespace distinct_monic_quadratic_polynomials_count_l463_463185

theorem distinct_monic_quadratic_polynomials_count :
  let c := 122^85 in
  let max_a_b := log c / log 5 in
  ∑ i in finset.range (max_a_b+1), finset.range (i+1) > 0 ->  # valid a and b
  ∑ i in finset.range (max_a_b+1), finset.range (i+1).card = 16511 :=
by
  sorry

end distinct_monic_quadratic_polynomials_count_l463_463185


namespace construct_isosceles_triangle_l463_463143

-- Define the isosceles triangle construction with the given conditions.
structure Triangle where
  A B C : Point
  isosceles_with_base : length A B = length A C
  base_length_difference_known : length A B - length B C = known_difference
  angle_at_vertex_known : ∠ B A C = known_angle

-- A theorem to construct triangle ABC with given conditions of isosceles triangle.
theorem construct_isosceles_triangle
  (known_difference : ℝ)
  (known_angle : ℝ)
  : ∃ (ABC : Triangle), 
    ABC.base_length_difference_known = known_difference ∧ 
    ABC.angle_at_vertex_known = known_angle
:= sorry

end construct_isosceles_triangle_l463_463143


namespace max_real_roots_of_polynomial_l463_463993

def polynomial (n : ℕ) : Polynomial ℝ :=
  ∑ i in Finset.range (n+2), Polynomial.C 1 * Polynomial.X ^ i

theorem max_real_roots_of_polynomial (n : ℕ) (h : 0 < n) :
  let P : Polynomial ℝ := polynomial n in
  if n % 2 = 1 then P.roots.count 1 = 0 ∧ P.roots.count (-1) = 1
  else P.roots.count 1 = 0 ∧ P.roots.count (-1) = 0 :=
sorry

end max_real_roots_of_polynomial_l463_463993


namespace largest_sum_of_digits_l463_463670

theorem largest_sum_of_digits :
  ∃ (a b c z : ℕ), (a ≤ 9) ∧ (b ≤ 9) ∧ (c ≤ 9) ∧ (0 < z ∧ z ≤ 12) ∧ (0.abc = 1 / (z : ℚ)) ∧ 
  (∀ a' b' c' z', (a' ≤ 9) ∧ (b' ≤ 9) ∧ (c' ≤ 9) ∧ (0 < z' ∧ z' ≤ 12) ∧ (0.abc = 1 / (z' : ℚ)) → (a + b + c ≥ a' + b' + c')) :=
by
  sorry

end largest_sum_of_digits_l463_463670


namespace cannot_have_triangular_cross_section_l463_463440

-- Definition of the geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | TriangularPrism
  | Cube

-- Theorem statement
theorem cannot_have_triangular_cross_section (s : GeometricSolid) :
  s = GeometricSolid.Cylinder → ¬(∃ c : ℝ^3 → Prop, is_triangular_cross_section s c) := 
by
  intros h
  apply h.rec 
  intro
  sorry

end cannot_have_triangular_cross_section_l463_463440


namespace coefficient_of_x_squared_term_l463_463726

noncomputable def integral_a : ℝ := ∫ x in -(Real.pi / 2)..(Real.pi / 2), Real.cos x

theorem coefficient_of_x_squared_term :
  let a := integral_a in
  (a = 2) →
  (let f := λ x : ℝ, (a * Real.sqrt x - 1 / Real.sqrt x)^6 in
  let coeff := (finset.sum (finset.range 7) (λ r, if 3 - r = 2 then ((nat.choose 6 r) * (-1)^r * 2^(6 - r)) else 0)) in
  coeff = -192) :=
by {
  sorry
}

end coefficient_of_x_squared_term_l463_463726


namespace power_division_correct_l463_463461

theorem power_division_correct :
  (∀ x : ℝ, x^4 / x = x^3) ∧ 
  ¬(∀ x : ℝ, 3 * x^2 * 4 * x^2 = 12 * x^2) ∧
  ¬(∀ x : ℝ, (x - 1) * (x - 1) = x^2 - 1) ∧
  ¬(∀ x : ℝ, (x^5)^2 = x^7) := 
by {
  -- Proof would go here
  sorry
}

end power_division_correct_l463_463461


namespace derivative_at_zero_l463_463605

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x*(-2) -- Assuming f'(1) = -2 from the solution

theorem derivative_at_zero : f' 0 = -4 := by
  sorry

end derivative_at_zero_l463_463605


namespace base_six_digits_unique_l463_463701

theorem base_six_digits_unique (b : ℕ) (h : (b-1)^2*(b-2) = 100) : b = 6 :=
by
  sorry

end base_six_digits_unique_l463_463701


namespace speed_of_current_l463_463504

variable (m c : ℝ)

theorem speed_of_current (h1 : m + c = 15) (h2 : m - c = 10) : c = 2.5 :=
sorry

end speed_of_current_l463_463504


namespace min_length_tangent_line_ellipse_l463_463619

theorem min_length_tangent_line_ellipse :
  (∀ (x y : ℝ), (x^2 / 25 + y^2 / 9 = 1) → 
  ∃ (AB : ℝ), (AB = 8) ∧
  ∀ (P : ℝ × ℝ), (P = (5 * Real.cos θ, 3 * Real.sin θ)) → 
  ∀ (θ : ℝ), ∀ tangent_at_P : ℝ × ℝ → ℝ,
  (tangent_at_P = λ Q, (Q.1 * (Real.cos θ / 5) + Q.2 * (Real.sin θ / 3))) → 
  let A := (5 / Real.cos θ, 0),
      B := (0, 3 / Real.sin θ) in
  A.1 = (5 / Real.cos θ) ∧ A.2 = 0 ∧ B.1 = 0 ∧ B.2 = (3 / Real.sin θ) ∧
  ∀ AB', AB' = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
  AB' ≥ 8 ) :=
sorry

end min_length_tangent_line_ellipse_l463_463619


namespace max_intersection_points_l463_463898

theorem max_intersection_points (c1 c2 c3 : Circle) (l : Line) : 
  ∃ n, n = 12 ∧ maximum_intersection_points c1 c2 c3 l = n :=
sorry

end max_intersection_points_l463_463898


namespace even_function_exists_l463_463860

noncomputable def example_even_function : ℝ → ℝ :=
  λ x, (8 / 21) * x^4 - (80 / 21) * x^2 + (24 / 7)

theorem even_function_exists :
  (example_even_function (-1) = 0) ∧
  (example_even_function (0.5) = 2.5) ∧
  (example_even_function 3 = 0) ∧
  (∀ x : ℝ, example_even_function x = example_even_function (-x)) :=
begin
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  {
    intro x,
    simp,
    sorry
  }
end

end even_function_exists_l463_463860


namespace total_cost_eq_1400_l463_463345

theorem total_cost_eq_1400 (stove_cost : ℝ) (wall_repair_fraction : ℝ) (wall_repair_cost : ℝ) (total_cost : ℝ)
  (h₁ : stove_cost = 1200)
  (h₂ : wall_repair_fraction = 1/6)
  (h₃ : wall_repair_cost = stove_cost * wall_repair_fraction)
  (h₄ : total_cost = stove_cost + wall_repair_cost) :
  total_cost = 1400 :=
sorry

end total_cost_eq_1400_l463_463345


namespace given_fraction_l463_463723

variable (initial_cards : ℕ)
variable (cards_given_to_friend : ℕ)
variable (fraction_given_to_brother : ℚ)

noncomputable def fraction_given (initial_cards cards_given_to_friend : ℕ) (fraction_given_to_brother : ℚ) : Prop :=
  let cards_left := initial_cards / 2
  initial_cards - cards_left - cards_given_to_friend = fraction_given_to_brother * initial_cards

theorem given_fraction
  (h_initial : initial_cards = 16)
  (h_given_to_friend : cards_given_to_friend = 2)
  (h_fraction : fraction_given_to_brother = 3 / 8) :
  fraction_given initial_cards cards_given_to_friend fraction_given_to_brother :=
by
  sorry

end given_fraction_l463_463723


namespace blue_red_area_ratio_l463_463839

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l463_463839


namespace bathtub_fill_time_l463_463922

-- Defining the conditions as given in the problem
def fill_rate (t_fill : ℕ) : ℚ := 1 / t_fill
def drain_rate (t_drain : ℕ) : ℚ := 1 / t_drain

-- Given specific values for the problem
def t_fill : ℕ := 10
def t_drain : ℕ := 12

-- Net fill rate calculation
def net_fill_rate (t_fill t_drain : ℕ) : ℚ :=
  fill_rate t_fill - drain_rate t_drain

-- Time to fill the bathtub given net fill rate
def time_to_fill (net_rate : ℚ) : ℚ :=
  1 / net_rate

-- The proof statement:
theorem bathtub_fill_time : time_to_fill (net_fill_rate t_fill t_drain) = 60 := by
  sorry

end bathtub_fill_time_l463_463922


namespace prank_helpers_combinations_l463_463017

theorem prank_helpers_combinations :
  let Monday := 1
  let Tuesday := 2
  let Wednesday := 3
  let Thursday := 4
  let Friday := 1
  (Monday * Tuesday * Wednesday * Thursday * Friday = 24) :=
by
  intros
  sorry

end prank_helpers_combinations_l463_463017


namespace find_x_l463_463596

theorem find_x (x M : ℤ) (hx1 : x > 0) (hx2 : is_square x) (h : 2520 * x = M^3) : x = 11025 :=
sorry

end find_x_l463_463596


namespace function_intersects_line_at_most_once_l463_463000

variable {α β : Type} [Nonempty α]

def function_intersects_at_most_once (f : α → β) (a : α) : Prop :=
  ∀ (b b' : β), f a = b → f a = b' → b = b'

theorem function_intersects_line_at_most_once {α β : Type} [Nonempty α] (f : α → β) (a : α) :
  function_intersects_at_most_once f a :=
by
  sorry

end function_intersects_line_at_most_once_l463_463000


namespace vanessa_days_missed_l463_463067

theorem vanessa_days_missed (V M S : ℕ) 
  (h1 : V + M + S = 17) 
  (h2 : V + M = 14) 
  (h3 : M + S = 12) : 
  V = 5 :=
begin
  sorry
end

end vanessa_days_missed_l463_463067


namespace least_root_of_polynomial_l463_463592

theorem least_root_of_polynomial : 
  ∃ x, 10 * x ^ 4 - 8 * x ^ 2 + 3 = 0 ∧ ∀ y, (10 * y ^ 4 - 8 * y ^ 2 + 3 = 0 → x ≤ y) ↔ x = -real.sqrt (3 / 5) := 
by
  sorry

end least_root_of_polynomial_l463_463592


namespace A_card_l463_463480

-- Define the cards held by players A, B, and C
def cards := { 1, 2, 3 }
def card_A := {1, 2}
def card_B := {1, 3}
def card_C := {2, 3}

variables (a b c : cards → Prop) -- Variables representing the cards held by A, B, and C

-- A's statement: "The number that is the same on B's and my card is not 2."
axiom A_statement : ∀ (b : cards), (b ∩ card_A ≠ {2})

-- B's statement: "The number that is the same on C's and my card is not 1."
axiom B_statement : ∀ (c : cards), (c ∩ card_B ≠ {1})

-- C's statement: "The sum of the numbers on my card is not 5."
axiom C_statement : ∑ (c : card_C) ≠ 5

-- The proof statement
theorem A_card : card_A = {1, 3} :=
sorry

end A_card_l463_463480


namespace max_intersections_l463_463895

theorem max_intersections (C₁ C₂ C₃ : Circle) (L : Line) : 
  greatest_points_of_intersection 3 1 = 12 :=
sorry

end max_intersections_l463_463895


namespace initial_pencils_correct_l463_463881

variable (pencils_taken remaining_pencils initial_pencils : ℕ)

def initial_number_of_pencils (pencils_taken remaining_pencils : ℕ) : ℕ :=
  pencils_taken + remaining_pencils

theorem initial_pencils_correct (h₁ : pencils_taken = 22) (h₂ : remaining_pencils = 12) :
  initial_number_of_pencils pencils_taken remaining_pencils = 34 := by
  rw [h₁, h₂]
  rfl

end initial_pencils_correct_l463_463881


namespace interval_of_decrease_range_of_k_l463_463647

-- Define the function f(x) = -1/2 * a * x^2 + (1 + a) * x - ln x.
def f (a : ℝ) (x : ℝ) : ℝ := - (1/2) * a * x^2 + (1 + a) * x - Real.log x

-- The intervals of monotonic decrease for f(x) given different conditions on a.
theorem interval_of_decrease (a : ℝ) : 
  (a = 1 → ∀ x : ℝ, (0 < x) → f' a x ≤ 0) ∧
  (a > 1 → ∀ x : ℝ, (1 < x ∨ 0 < x ∧ x < 1/a) → f' a x ≤ 0) ∧
  (0 < a ∧ a < 1 → ∀ x : ℝ, (1/a < x ∨ 0 < x ∧ x < 1) → f' a x ≤ 0) :=
by
  sorry

-- Define the function g(x) = x * f(x), where a = 0.
def g (x : ℝ) : ℝ := x^2 - x * Real.log x

-- The range of the real number k.
theorem range_of_k (m n : ℝ) (k : ℝ) :
  (∃ (m n : ℝ) (k : ℝ), (0 < m < n) ∧ (m, n) ⊆ (m, n) ∧ k ∈ (1, (9 + 2 * Real.log 2)/10]) :=
by
  sorry

end interval_of_decrease_range_of_k_l463_463647


namespace number_of_valid_quadruples_is_15_l463_463593

def valid_quadruples_count (a b c d : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ a^2 + b^2 + c^2 + d^2 = 4 ∧ (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 16

theorem number_of_valid_quadruples_is_15 :
  (set.count {x : ℝ × ℝ × ℝ × ℝ | valid_quadruples_count x.1 x.2.1 x.2.2.1 x.2.2.2} = 15) :=
by
  sorry

end number_of_valid_quadruples_is_15_l463_463593


namespace train_length_correct_l463_463926

noncomputable def train_speed_kmph : ℝ := 60
noncomputable def train_time_seconds : ℝ := 15

noncomputable def length_of_train : ℝ :=
  let speed_mps := train_speed_kmph * 1000 / 3600
  speed_mps * train_time_seconds

theorem train_length_correct :
  length_of_train = 250.05 :=
by
  -- Proof goes here
  sorry

end train_length_correct_l463_463926


namespace graham_crackers_left_over_l463_463751

-- Defining all given conditions
def boxesOfGrahamCrackers := 14
def packetsOfOreos := 15
def grahamCrackersPerCheesecake := 2
def oreosPerCheesecake := 3

-- The goal is to prove Lionel will have 4 boxes of Graham crackers left over
theorem graham_crackers_left_over :
  let cheesecakes_with_graham := boxesOfGrahamCrackers / grahamCrackersPerCheesecake,
      cheesecakes_with_oreos := packetsOfOreos / oreosPerCheesecake,
      max_cheesecakes := min cheesecakes_with_graham cheesecakes_with_oreos,
      graham_used := max_cheesecakes * grahamCrackersPerCheesecake,
      graham_left := boxesOfGrahamCrackers - graham_used
  in graham_left = 4 := by
  sorry

end graham_crackers_left_over_l463_463751


namespace determine_asymptotes_l463_463613

noncomputable def hyperbola_eccentricity_asymptote_relation (a b : ℝ) (e : ℝ) (k : ℝ) :=
  a > 0 ∧ b > 0 ∧ (e = Real.sqrt 2 * |k|) ∧ (k = b / a)

theorem determine_asymptotes (a b : ℝ) (h : hyperbola_eccentricity_asymptote_relation a b (Real.sqrt (a^2 + b^2) / a) (b / a)) :
  true := sorry

end determine_asymptotes_l463_463613


namespace solve_chris_age_l463_463403

/-- 
The average of Amy's, Ben's, and Chris's ages is 12. Six years ago, Chris was the same age as Amy is now. In 3 years, Ben's age will be 3/4 of Amy's age at that time. 
How old is Chris now? 
-/
def chris_age : Prop := 
  ∃ (a b c : ℤ), 
    (a + b + c = 36) ∧
    (c - 6 = a) ∧ 
    (b + 3 = 3 * (a + 3) / 4) ∧
    (c = 17)

theorem solve_chris_age : chris_age := 
  by
    sorry

end solve_chris_age_l463_463403


namespace price_equivalence_l463_463372

-- Definitions
def price_A := 100
def price_B := 75
def n_A : ℕ := 2 * n_B
def n_B : ℕ := 750 / price_B 

-- Theorem statement
theorem price_equivalence :
  (price_A = price_B + 25) ∧ 
  (n_A * price_A = 2000) ∧ 
  (n_A = 2 * n_B) ∧ 
  (n_B * price_B = 750) → 
  (price_A = 100) ∧ (price_B = 75) :=
by {
  -- Proof is omitted
  sorry
}

end price_equivalence_l463_463372


namespace rectangle_area_increase_l463_463863

theorem rectangle_area_increase (b : ℕ) (h1 : 2 * b = 40) (h2 : b = 20) : 
  let l := 2 * b
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 5
  let A_new := l_new * b_new
  A_new - A_original = 75 := 
by
  sorry

end rectangle_area_increase_l463_463863


namespace cos_value_l463_463242

-- Definitions and conditions from the problem
def alpha (a : ℝ) : Prop := 0 < a ∧ a < π / 2
def sin_value (a : ℝ) : Prop := Real.sin (a + π / 6) = 3 / 5

-- Proof goal in Lean 4
theorem cos_value (a : ℝ) (h1 : alpha a) (h2 : sin_value a) : 
  Real.cos (2 * a + π / 12) = (31 * Real.sqrt 2) / 50 :=
by
  sorry  -- Proof omitted

end cos_value_l463_463242


namespace basketball_opponents_score_l463_463940

noncomputable def total_opponents_score (team_scores : List ℕ) (lost_games : List ℕ) (won_games : List ℕ) : ℕ :=
  let lost_opponent_scores := List.map (λ s, s + 1) lost_games
  let won_opponent_scores := List.map (λ s, s / 3) won_games
  List.sum lost_opponent_scores + List.sum won_opponent_scores

theorem basketball_opponents_score :
  let team_scores := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] in
  let lost_games := [1, 3, 5, 7, 9, 11, 13] in
  let won_games := [6, 12, 15] in
  total_opponents_score team_scores lost_games won_games = 67 :=
by
  sorry

end basketball_opponents_score_l463_463940


namespace largest_sum_of_digits_l463_463672

noncomputable theory

-- Definition of the problem
def digits (n : ℕ) : Prop := n < 10

-- Main theorem statement
theorem largest_sum_of_digits {a b c : ℕ} (h_a : digits a) (h_b : digits b) (h_c : digits c) (z : ℕ) (h_z : 0 < z ∧ z ≤ 12) :
  (0.abc = 1 / z) → a + b + c ≤ 8 := 
sorry

end largest_sum_of_digits_l463_463672


namespace cylinder_lateral_surface_area_unchanged_and_volume_doubled_l463_463411

theorem cylinder_lateral_surface_area_unchanged_and_volume_doubled
  (r h : ℝ) :
  let r' := 2 * r,
      h' := h / 2,
      A_lateral := 2 * Real.pi * r * h,
      A_lateral' := 2 * Real.pi * r' * h',
      V := Real.pi * r^2 * h,
      V' := Real.pi * r'^2 * h'
  in A_lateral' = A_lateral ∧ V' = 2 * V :=
by
  let r' := 2 * r
  let h' := h / 2
  let A_lateral := 2 * Real.pi * r * h
  let A_lateral' := 2 * Real.pi * r' * h'
  let V := Real.pi * r^2 * h
  let V' := Real.pi * r'^2 * h'
  have h1 : A_lateral' = A_lateral := by 
    calc 
      A_lateral'
        = 2 * Real.pi * r' * h' : by rfl
    ... = 2 * Real.pi * (2 * r) * (h / 2) : by rfl
    ... = 2 * Real.pi * r * h : by ring
  have h2 : V' = 2 * V := by 
    calc 
      V'
        = Real.pi * r'^2 * h' : by rfl
    ... = Real.pi * (2 * r)^2 * (h / 2) : by rfl
    ... = 2 * Real.pi * r^2 * h : by ring_nf
  exact ⟨h1, h2⟩

end cylinder_lateral_surface_area_unchanged_and_volume_doubled_l463_463411


namespace find_omega_l463_463311

variable (ω : ℝ)

def function_period (ω : ℝ) := ∀ x, (cos(2 * ω * x)) = cos(2 * ω * (x + π))

theorem find_omega (hω : ω > 0) (h_period : function_period ω) : ω = 1 := 
by 
  sorry

end find_omega_l463_463311


namespace avocados_in_dozens_l463_463373

/--
A fruit vendor sold 2.5 dozen lemons and some dozens of avocados, with a total of 90 fruits sold.
Prove that the number of dozens of avocados sold is 5, given that a dozen contains 12 fruits.
-/
theorem avocados_in_dozens (total_fruits lemons_per_dozen avocados_per_dozen lemons_dozens: ℕ) :
  total_fruits = 90 →
  lemons_per_dozen = 12 →
  avocados_per_dozen = 12 →
  lemons_dozens = 2.5 →
  (total_fruits - lemons_dozens * lemons_per_dozen) / avocados_per_dozen = 5 :=
by
  intros
  sorry

end avocados_in_dozens_l463_463373


namespace initial_stock_before_shipment_l463_463541

-- Define the conditions for the problem
def initial_stock (total_shelves new_shipment_bears bears_per_shelf: ℕ) : ℕ :=
  let total_bears_on_shelves := total_shelves * bears_per_shelf
  total_bears_on_shelves - new_shipment_bears

-- State the theorem with the conditions
theorem initial_stock_before_shipment : initial_stock 2 10 7 = 4 := by
  -- Mathematically, the calculation details will be handled here
  sorry

end initial_stock_before_shipment_l463_463541


namespace swimming_pool_width_l463_463880

theorem swimming_pool_width 
  (V_G : ℝ) (G_CF : ℝ) (height_inch : ℝ) (L : ℝ) (V_CF : ℝ) (height_ft : ℝ) (A : ℝ) (W : ℝ) :
  V_G = 3750 → G_CF = 7.48052 → height_inch = 6 → L = 40 →
  V_CF = V_G / G_CF → height_ft = height_inch / 12 →
  A = L * W → V_CF = A * height_ft →
  W = 25.067 :=
by
  intros hV hG hH hL hVC hHF hA hVF
  sorry

end swimming_pool_width_l463_463880


namespace prob_at_least_one_solves_l463_463376

theorem prob_at_least_one_solves (p1 p2 : ℝ) (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (1 : ℝ) - (1 - p1) * (1 - p2) = 1 - ((1 - p1) * (1 - p2)) :=
by sorry

end prob_at_least_one_solves_l463_463376


namespace find_tangent_perpendicular_t_l463_463309

noncomputable def y (x : ℝ) : ℝ := x * Real.log x

theorem find_tangent_perpendicular_t (t : ℝ) (ht : 0 < t) (h_perpendicular : (1 : ℝ) * (1 + Real.log t) = -1) :
  t = Real.exp (-2) :=
by
  sorry

end find_tangent_perpendicular_t_l463_463309


namespace sum_of_valid_c_l463_463188

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_valid_c :
  (∑ c in Finset.filter (λ c, ∃ k, k * k = 49 - 12 * c) (Finset.range 5), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463188


namespace animal_market_problem_l463_463051

theorem animal_market_problem:
  ∃ (s c : ℕ), 0 < s ∧ 0 < c ∧ 28 * s + 27 * c = 1200 ∧ c > s :=
by
  sorry

end animal_market_problem_l463_463051


namespace radius_of_circular_film_l463_463368

noncomputable def volume_of_box (l w h : ℝ) : ℝ := l * w * h
noncomputable def volume_of_film (r t : ℝ) : ℝ := π * r^2 * t

theorem radius_of_circular_film :
  ∀ (l w h t : ℝ), l = 8 → w = 4 → h = 10 → t = 0.2 → 
  let V := volume_of_box l w h in
  V = volume_of_film ((sqrt ((320 : ℝ) / t)) / sqrt π) t →
  sqrt ((1600 : ℝ) / π) = (sqrt ((320 : ℝ) / t)) / sqrt π :=
begin
  intros,
  sorry
end

end radius_of_circular_film_l463_463368


namespace units_digit_T_l463_463291

theorem units_digit_T : 
  let T := (Finset.range 49).sum (λ n, (n + 1)!)
  in T % 10 = 3 := 
by
  -- Define the sum of factorials T
  let T := (Finset.range 49).sum (λ n, (n + 1)!)
  
  -- sorry here indicates a placeholder for the proof
  have units_digit_T : T % 10 = 3 := sorry

  exact units_digit_T

end units_digit_T_l463_463291


namespace polynomial_mod_3_l463_463476

theorem polynomial_mod_3 
  (f : ℤ → ℤ)
  (hf : ∀ x : ℤ, f x = a_n * x^n + a_(n-1) * x^(n-1) + ... + a_0) -- polynomial with integer coefficients
  (k : ℤ)
  (hk : f k % 3 = 0)
  (hk1 : f (k + 1) % 3 = 0)
  (hk2 : f (k + 2) % 3 = 0) : 
  ∀ m : ℤ, f m % 3 = 0 := 
by 
  sorry

end polynomial_mod_3_l463_463476


namespace concurrency_of_lines_l463_463729

open EuclideanGeometry

theorem concurrency_of_lines
  (A B C P D E : Point)
  (hP_inside : IsInTriangle P A B C)
  (h_angle_condition : ∠APB - ∠ACB = ∠APC - ∠ABC)
  (hD_incenter : IsIncenter D A P B)
  (hE_incenter : IsIncenter E A P C) :
  AreConcurrent (AP : Line) (BD : Line) (CE : Line) :=
sorry

end concurrency_of_lines_l463_463729


namespace smallest_7_digit_number_divisible_by_all_l463_463029

def smallest_7_digit_number : ℕ := 7207200

theorem smallest_7_digit_number_divisible_by_all :
  smallest_7_digit_number >= 1000000 ∧ smallest_7_digit_number < 10000000 ∧
  smallest_7_digit_number % 35 = 0 ∧ 
  smallest_7_digit_number % 112 = 0 ∧ 
  smallest_7_digit_number % 175 = 0 ∧ 
  smallest_7_digit_number % 288 = 0 ∧ 
  smallest_7_digit_number % 429 = 0 ∧ 
  smallest_7_digit_number % 528 = 0 :=
by
  sorry

end smallest_7_digit_number_divisible_by_all_l463_463029


namespace correct_statement_D_l463_463048

def competition_probability : ℚ := 3 / 5
def cure_rate : ℚ := 10 / 100
def precipitation_probability : ℚ := 90 / 100

theorem correct_statement_D :
  (∃ (A B C D : Prop),
    (A → competition_probability = 3 / 5 → (¬ (person_A_wins_exactly_three_out_of_five_matches competition_probability))) ∧
    (B → cure_rate = 10 / 100 → (¬ (tenth_patient_cure_guarantee cure_rate 9))) ∧
    (C → ¬ (frequency_equals_probability (random_experiment))) ∧
    (D → precipitation_probability = 90 / 100 → (tomorrow_precipitation_chance precipitation_probability)) ∧
    D) 
  → D :=
by
  sorry

end correct_statement_D_l463_463048


namespace tic_tac_toe_probability_l463_463539

theorem tic_tac_toe_probability :
  let total_positions := Nat.choose 9 3,
      winning_positions := 8 in
  (winning_positions / total_positions : ℚ) = 2 / 21 :=
by
  sorry

end tic_tac_toe_probability_l463_463539


namespace rationalize_denominator_l463_463382

theorem rationalize_denominator :
  ∃ (A B C D : ℕ),
  (4 / (cbrt 5 - cbrt 2)) = ((cbrt A + cbrt B + cbrt C) / D) ∧
  (A + B + C + D = 159) := by
sorry

end rationalize_denominator_l463_463382


namespace convert_polar_to_rect_and_back_l463_463146

theorem convert_polar_to_rect_and_back (r θ : ℝ) (hr : r = 7) (hθ : θ = Real.pi) :
  let x := r * Real.cos θ,
      y := r * Real.sin θ in
  (x = -7 ∧ y = 0) ∧ (Real.sqrt (x^2 + y^2) = 7 ∧ Real.atan2 y x = Real.pi) :=
by
  sorry

end convert_polar_to_rect_and_back_l463_463146


namespace solve_problem_l463_463273

noncomputable def problem_statement (x : ℝ) (k n : ℕ) : Prop :=
  (x = 1/3) → (k = 1007) → (n = 2014) → ((binom x k * 3^n) / (binom n k) = 1/2013)

theorem solve_problem : problem_statement (1/3) 1007 2014 :=
by
  sorry

end solve_problem_l463_463273


namespace solution_set_xf_pos_l463_463630

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem solution_set_xf_pos (f : ℝ → ℝ) (h₁ : odd_function f) (h₂ : f 2 = 0)
  (h₃ : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x > 0) :
  { x | x * f x > 0 } = { x | x < -2 } ∪ { x | x > 2 } := by
  sorry

end solution_set_xf_pos_l463_463630


namespace problem_l463_463629

def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
def g (x : ℝ) : ℝ := 2 * x + 4
theorem problem : f (g 5) - g (f 5) = 123 := 
by 
  sorry

end problem_l463_463629


namespace speed_of_stream_l463_463953

theorem speed_of_stream (b s : ℕ) 
  (h1 : b + s = 42) 
  (h2 : b - s = 24) :
  s = 9 := by sorry

end speed_of_stream_l463_463953


namespace quadrilateral_is_parallelogram_l463_463859

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 2 * a * b + 2 * c * d) : a = b ∧ c = d :=
by
  sorry

end quadrilateral_is_parallelogram_l463_463859


namespace the_proof_problem_l463_463791

def conditions (a b : ℝ) (gamma : ℝ) (A₁ B₁ C₁ : Point) : Prop :=
  AC = 5 ∧ BC = 2 ∧ gamma = real.arccos (13 / 20) ∧ menelaus_theorem A₁ B₁ C₁

def proof_problem : Prop :=
  ∀ (A₁ B₁ C₁ : Point) 
    (a b c : ℝ)
    (gamma : ℝ),
  conditions a b gamma A₁ B₁ C₁ → 
  (∠ A₁ C₁ B₁ = 180°) ∧ (length A₁ B₁ = sqrt 190)
  
theorem the_proof_problem : proof_problem :=
by noSorries

end the_proof_problem_l463_463791


namespace sum_positive_integer_values_c_l463_463199

theorem sum_positive_integer_values_c :
  (∑ c in (finset.filter (λ c : ℤ, is_integer c ∧ c > 0)
    (finset.image (λ k : ℤ, (49 - k^2) / 12)
      (finset.range 7))), c) = 6 :=
sorry

end sum_positive_integer_values_c_l463_463199


namespace square_side_length_in_right_triangle_l463_463513

theorem square_side_length_in_right_triangle (PQ QR PR : ℕ) (h1 : PQ = 5) (h2 : QR = 12) (h3 : PR = 13) 
  (h4 : nat.coprime 5 12) : ∃ t : ℚ, t = 156 / 25 :=
by
  use 156 / 25
  sorry

end square_side_length_in_right_triangle_l463_463513


namespace deposit_is_500_l463_463344

-- Definitions corresponding to the conditions
def janet_saved : ℕ := 2225
def rent_per_month : ℕ := 1250
def advance_months : ℕ := 2
def extra_needed : ℕ := 775

-- Definition that encapsulates the deposit calculation
def deposit_required (saved rent_monthly months_advance extra : ℕ) : ℕ :=
  let total_rent := months_advance * rent_monthly
  let total_needed := saved + extra
  total_needed - total_rent

-- Theorem statement for the proof problem
theorem deposit_is_500 : deposit_required janet_saved rent_per_month advance_months extra_needed = 500 :=
by
  sorry

end deposit_is_500_l463_463344


namespace local_maximum_at_e_l463_463216

noncomputable def f : ℝ → ℝ := λ x, (Real.log x) / x
noncomputable def f' : ℝ → ℝ := λ x, (1 - Real.log x) / (x^2)

theorem local_maximum_at_e : ∀ x > 0, (0 < x ∧ x < Real.exp 1 → f' x > 0) →
                               (x > Real.exp 1 → f' x < 0) →
                               for (some x), x = Real.exp 1 ∧ is_local_maximum_at f x := by 
  sorry

end local_maximum_at_e_l463_463216


namespace ratio_pat_mark_l463_463375

-- Conditions (as definitions)
variables (K P M : ℕ)
variables (h1 : P = 2 * K)  -- Pat charged twice as much time as Kate
variables (h2 : M = K + 80) -- Mark charged 80 more hours than Kate
variables (h3 : K + P + M = 144) -- Total hours charged is 144

theorem ratio_pat_mark (h1 : P = 2 * K) (h2 : M = K + 80) (h3 : K + P + M = 144) : 
  P / M = 1 / 3 :=
by
  sorry -- to be proved

end ratio_pat_mark_l463_463375


namespace eval_f_at_5_l463_463563

def f (x : ℝ) : ℝ := 2 * x^7 - 9 * x^6 + 5 * x^5 - 49 * x^4 - 5 * x^3 + 2 * x^2 + x + 1

theorem eval_f_at_5 : f 5 = 56 := 
 by 
   sorry

end eval_f_at_5_l463_463563


namespace find_a_in_triangle_l463_463690

noncomputable def a_value (b c cos_half_A : ℝ) : ℝ :=
  let cos_A := 2 * cos_half_A ^ 2 - 1
  in Real.sqrt (b ^ 2 + c ^ 2 - 2 * b * c * cos_A)

theorem find_a_in_triangle :
  ∀ (b c : ℝ) (cos_half_A : ℝ), b = 1 ∧ c = 3 ∧ cos_half_A = Real.sqrt 3 / 3 → a_value b c cos_half_A = 2 * Real.sqrt 3 :=
by
  intros b c cos_half_A h
  cases h with hb h1
  cases h1 with hc hcos_half_A
  rw [hb, hc, hcos_half_A]
  sorry

end find_a_in_triangle_l463_463690


namespace queenie_overtime_hours_l463_463771

theorem queenie_overtime_hours:
  ∀ (daily_pay overtime_pay total_pay days hours:int),
    daily_pay = 150 -> 
    overtime_pay = 5 -> 
    total_pay = 770 -> 
    days = 5 -> 
    (total_pay - (daily_pay * days)) / overtime_pay = 4 :=
by 
  intros daily_pay overtime_pay total_pay days hours
  assume h1: daily_pay = 150
  assume h2: overtime_pay = 5
  assume h3: total_pay = 770
  assume h4: days = 5
  calc 
    (total_pay - (daily_pay * days)) / overtime_pay 
        = (770 - (150 * 5)) / 5 : by rw [h1, h2, h3, h4]
    ... = 4                        : by norm_num

end queenie_overtime_hours_l463_463771


namespace expected_adjacent_pairs_3_out_of_9_is_2_div_3_l463_463118

-- Definitions of the problem
def is_adjacent_pair (a b : ℕ) : Prop := abs (a - b) = 1
def all_pairs_adjacent (s : Finset ℕ) : ℕ := 
  (s.toList).pairwise (λ a b, is_adjacent_pair a b) -- Counts number of pairs of adjacent elements in a set

noncomputable def expected_adjacent_pairs (n : ℕ) (chosen : ℕ) : ℚ :=
  let p_0 : ℚ := 5 / 12
  let p_1 : ℚ := 1 / 2
  let p_2 : ℚ := 1 / 12
  p_0 * 0 + p_1 * 1 + p_2 * 2

-- The main proof statement
theorem expected_adjacent_pairs_3_out_of_9_is_2_div_3 :
  expected_adjacent_pairs 9 3 = 2 / 3 :=
sorry

end expected_adjacent_pairs_3_out_of_9_is_2_div_3_l463_463118


namespace wet_surface_area_l463_463057

noncomputable theory

open Real

def length : Real := 12
def width : Real := 4
def depth : Real := 1.25
def Area_bottom : Real := length * width
def Area_long_side : Real := length * depth
def Area_short_side : Real := width * depth
def Total_area_long_sides : Real := 2 * Area_long_side
def Total_area_short_sides : Real := 2 * Area_short_side

theorem wet_surface_area : Area_bottom + Total_area_long_sides + Total_area_short_sides = 88 :=
by
  unfold Area_bottom Total_area_long_sides Total_area_short_sides Area_long_side Area_short_side
  have Area_bottom_val : Area_bottom = 48 := by norm_num
  have Total_area_long_sides_val : Total_area_long_sides = 30 := by norm_num
  have Total_area_short_sides_val : Total_area_short_sides = 10 := by norm_num
  rw [Area_bottom_val, Total_area_long_sides_val, Total_area_short_sides_val]
  norm_num

end wet_surface_area_l463_463057


namespace min_additional_stickers_l463_463754

theorem min_additional_stickers (total_stickers : ℕ) (remaining_stickers : ℕ) : 
  29 = total_stickers →
  (total_stickers mod 4 = 0) →
  ((total_stickers + remaining_stickers) mod 5 = 0) →
  remaining_stickers = 11 :=
begin
  sorry
end

end min_additional_stickers_l463_463754


namespace find_k_value_l463_463020

variables {V : Type} [AddCommGroup V] [VectorSpace ℝ V]

variables (e₁ e₂ : V) (non_collinear : ¬ (∃ c : ℝ, e₁ = c • e₂))
variables (k : ℝ)

variables (A B C D : V)
variables (AB : V) (CB : V) (CD : V) (BD : V)
variables (collinear : Filter ℝ)

-- Definitions of the given conditions as Lean variables
def AB_def : AB = e₁ - k • e₂ := sorry

def CB_def : CB = 2 • e₁ + e₂ := sorry

def CD_def : CD = 3 • e₁ - e₂ := sorry

-- Collinearity condition for points A, B, and D
def collinear_condition : ∃ λ : ℝ, AB = λ • BD := sorry

-- The main theorem to prove
theorem find_k_value : k = 2 :=
sorry

end find_k_value_l463_463020


namespace geometric_sequence_tenth_term_l463_463140

theorem geometric_sequence_tenth_term :
  let a := 5
      r := (3 : ℚ) / 4
  in ∃ a₁₀ : ℚ, a₁₀ = a * r^9 ∧ a₁₀ = 98415 / 262144 := by
  sorry

end geometric_sequence_tenth_term_l463_463140


namespace joined_after_8_months_l463_463056

theorem joined_after_8_months
  (investment_A investment_B : ℕ)
  (time_A time_B : ℕ)
  (profit_ratio : ℕ × ℕ)
  (h_A : investment_A = 36000)
  (h_B : investment_B = 54000)
  (h_ratio : profit_ratio = (2, 1))
  (h_time_A : time_A = 12)
  (h_eq : (investment_A * time_A) / (investment_B * time_B) = (profit_ratio.1 / profit_ratio.2)) :
  time_B = 4 := by
  sorry

end joined_after_8_months_l463_463056


namespace initial_num_balls_eq_18_l463_463498

variable (N B : ℕ)
variable (B_eq : B = 6)
variable (prob_cond : ((B - 3) : ℚ) / ((N - 3) : ℚ) = 1 / 5)

-- Prove that N = 18 given the conditions
theorem initial_num_balls_eq_18 (N B : ℕ) (B_eq : B = 6) (prob_cond : ((B - 3) : ℚ) / ((N - 3) : ℚ) = 1 / 5) : N = 18 := 
  sorry

end initial_num_balls_eq_18_l463_463498


namespace greatest_percentage_increase_l463_463708

noncomputable def percentage_increase (pop1990 pop2000 : ℕ) : ℚ :=
  (((pop2000 - pop1990) : ℚ) / (pop1990 : ℚ)) * 100

theorem greatest_percentage_increase :
  let city_F_1990 := 120_000
  let city_F_2000 := 150_000
  let city_G_1990 := 150_000
  let city_G_2000 := 195_000
  let city_H_1990 := 80_000
  let city_H_2000 := 100_000
  let city_I_1990 := 200_000
  let city_I_2000 := 260_000
  let city_J_1990 := 200_000
  let city_J_2000 := city_J_1990 + (city_J_1990 / 2)
  percentage_increase city_F_1990 city_F_2000 < 50 ∧
  percentage_increase city_G_1990 city_G_2000 < 50 ∧
  percentage_increase city_H_1990 city_H_2000 < 50 ∧
  percentage_increase city_I_1990 city_I_2000 < 50 → 
  percentage_increase city_J_1990 city_J_2000 = 50 :=
by
  sorry

end greatest_percentage_increase_l463_463708


namespace prevent_exploit_l463_463933

structure Purchase :=
  (laptop_price : ℕ)
  (headphones_price : ℕ)
  (total_price : ℕ)
  (delivery_free : Prop)
  (headphones_delivered_first : Prop)

def mechanism (p : Purchase) :=
  p.total_price = p.laptop_price + p.headphones_price ∧
  p.laptop_price = 115000 ∧
  p.headphones_price = 15000 ∧
  p.delivery_free ∧
  p.headphones_delivered_first

theorem prevent_exploit (p : Purchase) (m : mechanism p) :
  (m → ∃ payment_policy: ∀ (headphones_returned : Prop) (laptop_returned : Prop), 
    (headphones_returned → ¬ laptop_returned)) :=
sorry

end prevent_exploit_l463_463933


namespace max_intersections_circles_line_l463_463903

axiom disjoint_union_prod (X Y Z : Type) (n m k : ℕ) (hX : Finset.card X = n) (hY : Finset.card Y = m) (hZ : Finset.card Z = k) :
    (∃ (U : Type),  (Finset.card U = n * k) ∧ (Finset.card U = m * k)) → (∃ (W : Type), (Finset.card W = (n + m) * k))

theorem max_intersections_circles_line (n m : ℕ) (h1 : n = 3) (h2 : m = 1) : 
  let max_intersections := 2 * n + 2 * (n * (n - 1) / 2)
  in max_intersections = 12 := 
by {
  sorry 
}

end max_intersections_circles_line_l463_463903


namespace range_of_u_l463_463245

def satisfies_condition (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

def u (x y : ℝ) : ℝ := |2 * x + y - 4| + |3 - x - 2 * y|

theorem range_of_u {x y : ℝ} (h : satisfies_condition x y) : ∀ u, 1 ≤ u ∧ u ≤ 13 :=
sorry

end range_of_u_l463_463245


namespace largest_sum_of_digits_l463_463674

noncomputable theory

-- Definition of the problem
def digits (n : ℕ) : Prop := n < 10

-- Main theorem statement
theorem largest_sum_of_digits {a b c : ℕ} (h_a : digits a) (h_b : digits b) (h_c : digits c) (z : ℕ) (h_z : 0 < z ∧ z ≤ 12) :
  (0.abc = 1 / z) → a + b + c ≤ 8 := 
sorry

end largest_sum_of_digits_l463_463674


namespace solved_work_problem_l463_463483

noncomputable def work_problem : Prop :=
  ∃ (m w x : ℝ), 
  (3 * m + 8 * w = 6 * m + x * w) ∧ 
  (4 * m + 5 * w = 0.9285714285714286 * (3 * m + 8 * w)) ∧
  (x = 14)

theorem solved_work_problem : work_problem := sorry

end solved_work_problem_l463_463483


namespace log_relationship_l463_463604

theorem log_relationship 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h₁ : a = Real.log 0.3 / Real.log 2) 
  (h₂ : b = Real.log 0.4 / Real.log (1/2)) 
  (h₃ : c = 0.4 ^ 0.3) :
  a < c ∧ c < b :=
by {
  sorry
}

end log_relationship_l463_463604


namespace interest_rate_compound_interest_l463_463003

theorem interest_rate_compound_interest :
  ∀ (P A : ℝ) (t n : ℕ), 
  P = 156.25 → A = 169 → t = 2 → n = 1 → 
  (∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r * 100 = 4) :=
by
  intros P A t n hP hA ht hn
  use 0.04
  rw [hP, hA, ht, hn]
  sorry

end interest_rate_compound_interest_l463_463003


namespace clock_strikes_12_times_in_22_seconds_l463_463408

theorem clock_strikes_12_times_in_22_seconds :
  ∀ (n: ℕ) (t: ℕ), n > 1 ∧ t > 0 ∧ (5 - 1 = 4) ∧ (n - 1 = 11) ∧ (8 / 4 = 2)  → 
  (t = 8) → (n = 12) → ((8 / 4) * 11 = 22) :=
by
  intros n t hn ht h_interval_5 h_interval_12 h_time_interval ht_8 hn_12
  have h_time_one_interval : 8 / 4 = 2 := by sorry
  have h_intervals_12 : n - 1 = 11 := by sorry
  show (8 / 4) * 11 = 22 := by
    rw h_time_one_interval
    rw h_intervals_12
    exact eq.refl 22


end clock_strikes_12_times_in_22_seconds_l463_463408


namespace binomial_multiplication_subtract_240_l463_463570

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_multiplication_subtract_240 :
  binom 10 3 * binom 8 3 - 240 = 6480 :=
by
  sorry

end binomial_multiplication_subtract_240_l463_463570


namespace yellow_beads_needed_l463_463566

variable (Total green yellow : ℕ)

theorem yellow_beads_needed (h_green : green = 4) (h_yellow : yellow = 0) (h_fraction : (4 / 5 : ℚ) = 4 / (green + yellow + 16)) :
    4 + 16 + green = Total := by
  sorry

end yellow_beads_needed_l463_463566


namespace positive_factors_perfect_cubes_180_l463_463664

noncomputable theory

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

def prime_factors_exponents := {
  p2 : ℕ := 2,
  p3 : ℕ := 2,
  p5 : ℕ := 1
}

def count_perfect_cube_factors (n : ℕ) : ℕ :=
  if h : n = 180 
  then 
    let p2 := 0 in
    let p3 := 0 in
    let p5 := 0 in
    1
  else
    0

theorem positive_factors_perfect_cubes_180 : count_perfect_cube_factors 180 = 1 :=
by {
  rw count_perfect_cube_factors,
  simp,
  sorry,
}

end positive_factors_perfect_cubes_180_l463_463664


namespace difference_in_amount_paid_l463_463058

variable (P Q : ℝ)

theorem difference_in_amount_paid (hP : P > 0) (hQ : Q > 0) :
  (1.10 * P * 0.80 * Q - P * Q) = -0.12 * (P * Q) := 
by 
  sorry

end difference_in_amount_paid_l463_463058


namespace lucille_expense_l463_463753

theorem lucille_expense
  (R : ℕ) (revenue : ℝ)
  (ratio : Fin 4 → ℕ)
  (h_ratio : ratio = ![3, 5, 2, 7])
  (h_revenue : revenue = 10_000) :
  (let part := revenue / (ratio 0 + ratio 1 + ratio 2 + ratio 3) in
   (3 * part) + (2 * part) + (7 * part)) = 7_058.88 :=
by
  have ratio_sum := ratio 0 + ratio 1 + ratio 2 + ratio 3
  have part := revenue / ratio_sum
  have employee_salary := 3 * part
  have rent := 2 * part
  have marketing_costs := 7 * part
  have total_amount := employee_salary + rent + marketing_costs
  exact sorry

end lucille_expense_l463_463753


namespace find_m_solution_l463_463182

noncomputable def solution (m : ℤ) : Prop :=
  (-180 ≤ m ∧ m ≤ 180) ∧ (Float.sin (m * (Float.pi / 180)) = Float.sin (750 * (Float.pi / 180)))

theorem find_m_solution (m : ℤ) : solution m ↔ m = 30 ∨ m = 150 :=
by sorry

end find_m_solution_l463_463182


namespace fraction_addition_l463_463027

theorem fraction_addition :
  (1 / 3 * 2 / 5) + 1 / 4 = 23 / 60 := 
  sorry

end fraction_addition_l463_463027


namespace value_of_expression_l463_463507

def a : ℕ := 7
def b : ℕ := 5

theorem value_of_expression : (a^2 - b^2)^4 = 331776 := by
  sorry

end value_of_expression_l463_463507


namespace fraction_girls_trip_l463_463125

-- Define the conditions of the problem
variables (b g : ℕ) -- Number of boys and girls
variables (trip_girls trip_boys : ℕ) -- Number of girls and boys who went on the trip
variables (fraction_girls fraction_boys : ℚ) -- Fraction of girls and boys who went on the trip

-- Conditions
def condition1 : Prop := g = b
def condition2 : Prop := fraction_girls = 1 / 2
def condition3 : Prop := fraction_boys = 3 / 4
def trip_girls_def : Prop := trip_girls = fraction_girls * g
def trip_boys_def : Prop := trip_boys = fraction_boys * b

-- Proof problem: fraction of trip attendees who were girls is 2/5
theorem fraction_girls_trip 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3) 
  (h4 : trip_girls_def) 
  (h5 : trip_boys_def) 
  : (trip_girls : ℚ) / (trip_girls + trip_boys) = 2 / 5 := 
  sorry

end fraction_girls_trip_l463_463125


namespace coefficient_x75_l463_463178

-- Define the polynomial directly from the conditions
noncomputable def polynomial : Polynomial ℤ :=
  (∏ i in Finset.range (12 + 1), Polynomial.X^i - i)

-- State the theorem in Lean 4
theorem coefficient_x75 : polynomial.coeff 75 = -1 :=
  sorry

end coefficient_x75_l463_463178


namespace solve_equation_l463_463174

theorem solve_equation (x : ℝ) : 
  (sqrt x + 3 * sqrt (x^2 + 8 * x) + sqrt (x + 8) = 40 - 3 * x) → 
  x = 49 / 9 :=
by
  intro h,
  sorry

end solve_equation_l463_463174


namespace equation_of_ellipse_max_area_triangle_l463_463618

namespace EllipseProblem

noncomputable def e := (Real.sqrt 6) / 3
noncomputable def a := Real.sqrt 3
noncomputable def c := Real.sqrt 2
noncomputable def b := 1

-- Theorem 1: Prove that the given conditions determine the equation of the ellipse
theorem equation_of_ellipse :
  a > b ∧ b > 0 ∧ e = c / a ∧ ∀ (x y : ℝ),
  (Real.sqrt 6 / 3 = Real.sqrt 2 / Real.sqrt 3) ∧ 
  ((x^2) / (a^2) + (y^2) / (b^2) = 1) → 
  (x^2) / 3 + (y^2) = 1 :=
sorry

-- Theorem 2: Prove that the maximum area of triangle PAB is 9/4
theorem max_area_triangle :
  ∀ (P : ℝ × ℝ) (θ : ℝ),
  (P = (Real.sqrt 3 * Real.cos θ, Real.sin θ)) → 
  (y = x + 1) intersects (x^2 / 3 + y^2 = 1) at A (0, 1) and B (-3/2, -1/2) → 
  (* compute area logic here using described distances and points *)
  area_max := 9 / 4 :=
sorry

end EllipseProblem

end equation_of_ellipse_max_area_triangle_l463_463618


namespace students_not_together_l463_463011

def student : Type := {A, B, C, D, E, F}

theorem students_not_together (n : ℕ) (h : n = 6) (hA : student) (hB : student) (hCDEF : Finset student) :
  hA ≠ hB → hCDEF = {C, D, E, F} → 
  ((∀ a b ∈ hCDEF, a ≠ A ∧ b ≠ B) →
  ∃ arrangements : Finset (List student), arrangements.card = 480) :=
by
  sorry

end students_not_together_l463_463011


namespace move_piece_l463_463764

def adjacent_squares (i j : ℕ) (n : ℕ) : Prop :=
  (abs (i - j) = 1 ∧ 0 ≤ i ∧ i < n ∧ 0 ≤ j ∧ j < n)

def non_adjacent_placement (board : ℕ × ℕ → option ℕ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, (i < n ∧ j < n ∧ i ≠ j) → (board (i, j) = none ∨ ¬adjacent_squares i j n)

theorem move_piece (n : ℕ) (board : ℕ × ℕ → option ℕ) :
  n > 1 → (∃ k < n, option.is_none (board (k, k))) →
  (Σ i j, (i < n ∧ j < n ∧ i ≠ j ∧ option.is_some (board (i, j)))) →
  non_adjacent_placement board n →
  ∃ board', non_adjacent_placement board' n ∧
             ∃ i j : ℕ, (adjacent_squares i j n ∧ option.is_some (board' (i, j))). :=
by
  sorry

end move_piece_l463_463764


namespace trajectory_description_l463_463231

def trajectory_of_A (x y : ℝ) (m : ℝ) : Prop :=
  m * x^2 - y^2 = m ∧ y ≠ 0
  
theorem trajectory_description (x y m : ℝ) (h : m ≠ 0) :
  trajectory_of_A x y m →
    (m < -1 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m = -1 → (x^2 + y^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0))) ∧
    (-1 < m ∧ m < 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m > 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) :=
by
  intro h_trajectory
  sorry

end trajectory_description_l463_463231


namespace max_intersection_points_l463_463900

theorem max_intersection_points (c1 c2 c3 : Circle) (l : Line) : 
  ∃ n, n = 12 ∧ maximum_intersection_points c1 c2 c3 l = n :=
sorry

end max_intersection_points_l463_463900


namespace zeros_before_first_nonzero_decimal_of_fraction_l463_463165

theorem zeros_before_first_nonzero_decimal_of_fraction :
  let fraction := (1 : ℚ) / (2^4 * 5^7 : ℚ) in
  (∃ a b : ℚ, fraction = a / 10^b) →
  ∃ n : ℕ, n = 6 ∧ fraction = 8 / 10^(n+1) :=
by
  sorry

end zeros_before_first_nonzero_decimal_of_fraction_l463_463165


namespace balls_into_boxes_l463_463287

theorem balls_into_boxes :
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1) 
  combination = 15 :=
by
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1)
  show combination = 15
  sorry

end balls_into_boxes_l463_463287


namespace patriots_wins_l463_463276

theorem patriots_wins (Tigers Eagles Patriots Cubs Mounties Falcons: ℕ)
  (h1: Tigers > Eagles)
  (h2: Cubs > 15)
  (h3: Patriots > Cubs ∧ Patriots < Mounties)
  (h4: Falcons > Eagles ∧ Falcons < Mounties)
  (h5: {Tigers, Eagles, Patriots, Cubs, Mounties, Falcons} = {10, 18, 22, 27, 33}) :
  Patriots = 27 := sorry

end patriots_wins_l463_463276


namespace radius_of_circle_l463_463945

-- Circle with area x and circumference y
def circle_area (r : ℝ) : ℝ := π * r^2
def circle_circumference (r : ℝ) : ℝ := 2 * π * r
def circle_equation (r : ℝ) : ℝ := circle_area r + circle_circumference r

-- The given condition
theorem radius_of_circle (r : ℝ) (h : circle_equation r = 100 * π) : r = 10 :=
sorry

end radius_of_circle_l463_463945


namespace base8_addition_l463_463966

def base8_to_base10 (n : ℕ) : ℕ :=
  n.digits 8.reverse.foldl (fun acc d => acc * 8 + d) 0

def base10_to_base8 (n : ℕ) : ℕ :=
  let digits := n.digits 8
  digits.reverse.foldl (fun acc d => acc * 10 + d) 0

theorem base8_addition (a b : ℕ) (h1 : base8_to_base10 53 = 43) (h2 : base8_to_base10 27 = 23) : 
  base10_to_base8 ((base8_to_base10 53) + (base8_to_base10 27)) = 102 :=
by 
  sorry

end base8_addition_l463_463966


namespace max_possible_numbers_sum_2019_l463_463385

theorem max_possible_numbers_sum_2019 (n : ℕ) (a : fin n → ℕ)
  (h_sum : (∑ i, a i) = 2019)
  (h_cond : ∀ (i j : fin n), i ≤ j → a i ≠ 40 ∧ (∑ k in finset.Icc i j, a k) ≠ 40) :
  n ≤ 1019 :=
sorry

end max_possible_numbers_sum_2019_l463_463385


namespace range_correct_mean_correct_l463_463230

def data : List ℕ := [1, 2, 4, 3, 1, 2, 1]

def range (l: List ℕ) : ℕ := l.maximum - l.minimum

def mean (l: List ℕ) : ℚ := l.sum / l.length

theorem range_correct : range data = 3 := by
  -- data: [1, 2, 4, 3, 1, 2, 1]
  sorry

theorem mean_correct : mean data = 2 := by
  -- data: [1, 2, 4, 3, 1, 2, 1]
  sorry

end range_correct_mean_correct_l463_463230


namespace closest_to_100kg_total_excess_weight_total_weight_10_bags_l463_463544

-- Conditions
def measurements : List ℤ := [+3, +1, 0, +2, +6, -1, +2, +1, -4, +1]

-- Proof problems
theorem closest_to_100kg : (measurements.nth 2).getD 0 = 0 := 
sorry

theorem total_excess_weight : measurements.sum = 11 := 
sorry

theorem total_weight_10_bags : (measurements.length = 10) → 
  (1000 + measurements.sum = 1011) := 
sorry

end closest_to_100kg_total_excess_weight_total_weight_10_bags_l463_463544


namespace max_area_of_triangle_ABC_is_27_over_8_l463_463352

noncomputable def maxAreaOfTriangleABC : ℝ :=
  let A := (1 : ℝ, 0 : ℝ)
  let B := (4 : ℝ, 3 : ℝ)
  let C := (p, -p^2 + 6*p - 5)
  let area := (p : ℝ) -> (1 / 2) * ((1*3) + (4*(-p^2 + 6*p - 5)) + (p*0) 
                                    - (0*4) - (3*p) - ((-p^2 + 6*p - 5)*1)) 
  (3 / 2) * abs ((p - 1) * (p - 4))
  
theorem max_area_of_triangle_ABC_is_27_over_8 :
  (∃ p, 1 ≤ p ∧ p ≤ 4 ∧ (area p = (27 / 8))) :=
sorry

end max_area_of_triangle_ABC_is_27_over_8_l463_463352


namespace blue_area_factor_12_l463_463820

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l463_463820


namespace find_hansol_weight_l463_463879

variable (H : ℕ)

theorem find_hansol_weight (h : H + (H + 4) = 88) : H = 42 :=
by
  sorry

end find_hansol_weight_l463_463879


namespace prob_one_unqualified_prob_at_least_one_unqualified_penalty_standard_reasonable_l463_463944

-- We define the number total masks and unqualified masks in the sample
def total_masks : ℕ := 20
def unqualified_masks : ℕ := 2
def qualified_masks : ℕ := total_masks - unqualified_masks

-- Part (Ⅰ): Prove the probability of selecting 1 unqualified mask
theorem prob_one_unqualified : 
  (unqualified_masks : ℚ) / total_masks = 1 / 10 :=
sorry

-- Part (Ⅱ): Prove the probability of selecting at least one unqualified mask when choosing 3 masks
theorem prob_at_least_one_unqualified : 
  ((18.choose 2 * 2.choose 1) + (18.choose 1 * 2.choose 2)) / 20.choose 3 = 27 / 95 :=
sorry

-- Part (Ⅲ): Prove that the penalty standard is unreasonable
def cost_per_test : ℚ := 0.05
def boxes_per_day : ℕ := 100
def masks_per_box : ℕ := 300
def fine_per_unqualified_mask : ℚ := 500
def tests_per_box : ℕ := 3

-- Calculate testing cost
def daily_testing_cost := cost_per_test * (boxes_per_day * masks_per_box : ℚ)

-- Calculate expected daily fine amount assuming no testing
def prob_unqualified : ℚ := 1 / 10
def expected_fine_amount := fine_per_unqualified_mask * prob_unqualified

-- Check reasonableness of penalty standard
theorem penalty_standard_reasonable : daily_testing_cost ≠ expected_fine_amount * (boxes_per_day * tests_per_box : ℚ) :=
sorry

end prob_one_unqualified_prob_at_least_one_unqualified_penalty_standard_reasonable_l463_463944


namespace product_of_three_numbers_l463_463878

theorem product_of_three_numbers (
  a b c m : ℚ
) (h1 : a + b + c = 210) 
  (h2 : m = 8 * a) 
  (h3 : b = m + 11) 
  (h4 : c = m - 11) : a * b * c = 4173.75 :=
by
  have ha : a = 35 / 4,
  have hb : b = 81,
  have hc : c = 59,
  sorry

end product_of_three_numbers_l463_463878


namespace determine_a_values_l463_463990

theorem determine_a_values (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) ↔ a = 2 ∨ a = 8 :=
by
  sorry

end determine_a_values_l463_463990


namespace problem_statement_l463_463709

open Geometry

variables {A B C P E F G H : Point} {O₁ O₂ : Circle}

def TangentAt (O : Circle) (p : Point) : Prop :=
  is_tangent O p

def LinePerpendicular (l₁ l₂ : Line) : Prop :=
  angle l₁ l₂ = 90

theorem problem_statement
  (h1 : TangentAt O₁ E)
  (h2 : TangentAt O₂ F)
  (h3 : TangentAt O₁ G)
  (h4 : TangentAt O₂ H)
  (h5 : ∃ P, line_through P E ∧ line_through P G ∧ line_through P F ∧ line_through P H)
  (h6 : LinePerpendicular (line_through A P) (line_through B C)) :
  LinePerpendicular (line_through A P) (line_through B C) :=
begin
  sorry
end

end problem_statement_l463_463709


namespace shaded_area_l463_463333

-- Given conditions
def radius : ℝ := 2
def semicircle_area (r : ℝ) : ℝ := (1 / 2) * real.pi * r^2
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

def area_of_two_semicircles : ℝ := 2 * semicircle_area radius
def area_of_triangle_BCD : ℝ := triangle_area radius radius

-- Proof problem statement
theorem shaded_area :
  area_of_two_semicircles - area_of_triangle_BCD = 4 * real.pi - 2 := by
  sorry

end shaded_area_l463_463333


namespace blue_red_face_area_ratio_l463_463814

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l463_463814


namespace max_intersection_points_l463_463899

theorem max_intersection_points (c1 c2 c3 : Circle) (l : Line) : 
  ∃ n, n = 12 ∧ maximum_intersection_points c1 c2 c3 l = n :=
sorry

end max_intersection_points_l463_463899


namespace relationship_between_y1_y2_l463_463241

theorem relationship_between_y1_y2 (b y1 y2 : ℝ) 
  (h₁ : y1 = -(-1) + b) 
  (h₂ : y2 = -(2) + b) : 
  y1 > y2 := 
by 
  sorry

end relationship_between_y1_y2_l463_463241


namespace directrix_of_parabola_l463_463414

theorem directrix_of_parabola (y x : ℝ) (h_eq : y^2 = 8 * x) :
  x = -2 :=
sorry

end directrix_of_parabola_l463_463414


namespace shaded_area_proof_l463_463073

noncomputable theory

def abcd_area (l w : ℕ) : ℕ := l * w

def bp_qc (w : ℕ) : ℕ := w / 6

def shaded_area (l w : ℕ) : ℕ := 6

theorem shaded_area_proof (l w : ℕ) (h1 : abcd_area l w = 24) 
  (h2 : bp_qc w = w / 6) : 
  shaded_area l w = 6 := 
sorry

end shaded_area_proof_l463_463073


namespace max_points_of_intersection_l463_463890

theorem max_points_of_intersection
  (A B C : Circle) (L : Line)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ B ≠ C) :
  (∃ pA pB pC : Point, on_line L pA ∧ on_circle A pA ∧
                    on_line L pB ∧ on_circle B pB ∧
                    on_line L pC ∧ on_circle C pC) ∧
  max_intersection_points A B C L = 12 :=
sorry

end max_points_of_intersection_l463_463890


namespace probability_at_least_one_red_bean_paste_probability_distribution_X_expectation_X_l463_463581

-- Definition of the conditions
def total_zongzi : ℕ := 6
def egg_yolk_zongzi : ℕ := 4
def red_bean_paste_zongzi : ℕ := 2
def total_selected_zongzi : ℕ := 3

-- Definitions for probability calculations
noncomputable def combination (n r : ℕ) : ℚ := (nat.factorial n) / ((nat.factorial r) * (nat.factorial (n-r)))

-- Statement 1: Prove the probability of at least one red bean paste zongzi
theorem probability_at_least_one_red_bean_paste :
  (combination egg_yolk_zongzi 2 * combination red_bean_paste_zongzi 1 + combination egg_yolk_zongzi 1 * combination red_bean_paste_zongzi 2) / combination total_zongzi total_selected_zongzi = 4 / 5 :=
by sorry

-- Definitions and theorems for probability distribution and expectation of X
def P_X_0 : ℚ := combination egg_yolk_zongzi 3 / combination total_zongzi total_selected_zongzi
def P_X_1 : ℚ := combination egg_yolk_zongzi 2 * combination red_bean_paste_zongzi 1 / combination total_zongzi total_selected_zongzi
def P_X_2 : ℚ := combination egg_yolk_zongzi 1 * combination red_bean_paste_zongzi 2 / combination total_zongzi total_selected_zongzi

theorem probability_distribution_X :
  (P_X_0 = 1 / 5) ∧ (P_X_1 = 3 / 5) ∧ (P_X_2 = 1 / 5) :=
by sorry

theorem expectation_X :
  (0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2) = 1 :=
by sorry

end probability_at_least_one_red_bean_paste_probability_distribution_X_expectation_X_l463_463581


namespace solve_equation_l463_463782

theorem solve_equation (x : ℝ) : 3 * 4^x - 2^x - 2 = 0 ↔ x = 0 := by
  sorry

end solve_equation_l463_463782


namespace task_probabilities_l463_463464

theorem task_probabilities (P1_on_time : ℚ) (P2_on_time : ℚ) 
  (h1 : P1_on_time = 2/3) (h2 : P2_on_time = 3/5) : 
  P1_on_time * (1 - P2_on_time) = 4/15 := 
by
  -- proof is omitted
  sorry

end task_probabilities_l463_463464


namespace red_blue_area_ratio_is_12_l463_463806

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l463_463806


namespace complex_modulus_conjugate_l463_463640

theorem complex_modulus_conjugate (z : ℂ) (h : (1 + 2 * complex.I) * z = 2 - complex.I) : 
  complex.abs (complex.conj z) = 1 := 
sorry

end complex_modulus_conjugate_l463_463640


namespace sum_positive_integer_values_c_l463_463200

theorem sum_positive_integer_values_c :
  (∑ c in (finset.filter (λ c : ℤ, is_integer c ∧ c > 0)
    (finset.image (λ k : ℤ, (49 - k^2) / 12)
      (finset.range 7))), c) = 6 :=
sorry

end sum_positive_integer_values_c_l463_463200


namespace flea_distance_after_n_days_flea_distance_seventh_day_flea_distance_tenth_day_flea_less_than_0_001_day_l463_463445

theorem flea_distance_after_n_days (n : ℕ) :=
  let total_distance := 10
  let remaining_distance := total_distance * (1 / 2) ^ n
  let distance_covered := total_distance - remaining_distance
  distance_covered

theorem flea_distance_seventh_day : flea_distance_after_n_days 7 ≈ 9.9 :=
  sorry

theorem flea_distance_tenth_day : flea_distance_after_n_days 10 ≈ 9.99 :=
  sorry

theorem flea_less_than_0_001_day : ∃ n : ℕ, n ≥ 14 ∧ total_distance * (1 / 2) ^ n < 0.001 :=
  sorry

end flea_distance_after_n_days_flea_distance_seventh_day_flea_distance_tenth_day_flea_less_than_0_001_day_l463_463445


namespace surface_area_of_sphere_l463_463252

def A : Type := sorry
def B : Type := sorry
def C : Type := sorry
def sphere : Type := sorry

axiom AB_eq : ∀ (A B : sphere), dist A B = 3
axiom AC_perp_BC : ∀ (A C B : sphere), ⟂ (A - C) (B - C)
axiom center_to_plane : ∀ (O : sphere) (A B C : sphere), dist (orthogonal_projection (span {A, B, C}) O) O = O.radius / 2
axiom radius_squared_eq : ∀ (r : ℝ), r^2 = (r / 2)^2 + (3 / 2)^2

theorem surface_area_of_sphere : ∀ (r : ℝ), 4 * π * r^2 = 12 * π :=
by
  sorry

end surface_area_of_sphere_l463_463252


namespace coffee_preference_l463_463436

theorem coffee_preference (fraction: ℚ) (sample_size: ℕ) (expected: ℕ) : 
  fraction = 3 / 7 → 
  sample_size = 350 → 
  expected = 150 → 
  expected = fraction * sample_size :=
by
  intros 
  rw [← h, ← h_1]
  norm_num -- Simplifies the computation
  exact h_2
  sorry

end coffee_preference_l463_463436


namespace nicky_cristina_race_l463_463471

theorem nicky_cristina_race :
  ∀ (head_start t : ℕ), ∀ (cristina_speed nicky_speed time_nicky_run : ℝ),
  head_start = 12 →
  cristina_speed = 5 →
  nicky_speed = 3 →
  ((cristina_speed * t) = (nicky_speed * t + nicky_speed * head_start)) →
  time_nicky_run = head_start + t →
  time_nicky_run = 30 :=
by
  intros
  sorry

end nicky_cristina_race_l463_463471


namespace k_expression_series_inequality_l463_463649

noncomputable def k (x : ℝ) : ℝ := (1/4)*x^2 + (1/2)*x + (1/4)

theorem k_expression (a b c : ℝ) (a_ne_zero : a ≠ 0) (h1 : k (-1) = 0)
  (h2 : ∀ x : ℝ, x ∈ set.Icc (-real.sqrt 2) (real.sqrt 2) → k x = f x - (1/2)*x) 
  (h3 : ∀ x : ℝ, k x ≤ (1/2)*x^2 + (1/2)) : 
  k = λ x, (1/4)*x^2 + (1/2)*x + (1/4) :=
sorry

theorem series_inequality (n : ℕ) (hn : 0 < n) :
  (∑ i in finset.range n, 1 / (k (i + 1))) > (2 * n) / (n + 2) :=
sorry

end k_expression_series_inequality_l463_463649


namespace value_of_m_minus_n_l463_463678

variables {a b : ℕ}
variables {m n : ℤ}

def are_like_terms (m n : ℤ) : Prop :=
  (m - 2 = 4) ∧ (n + 7 = 4)

theorem value_of_m_minus_n (h : are_like_terms m n) : m - n = 9 :=
by
  sorry

end value_of_m_minus_n_l463_463678


namespace blue_face_area_greater_than_red_face_area_l463_463834

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l463_463834


namespace sum_inequality_l463_463360

theorem sum_inequality
  (n : ℕ) (a : ℕ → ℝ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ a i ∧ a i ≤ 1)
  (h2 : ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → a i ≤ a j) :
  (∑ i in finset.range n, (2 * i + 1) * a (i + 1)) < 
    n * (∑ i in finset.range n, (a (i + 1))^2) + (n^2) / 3 :=
sorry

end sum_inequality_l463_463360


namespace solve_inequality_l463_463783

theorem solve_inequality (x : ℝ) : 2 * (5 * x + 3) ≤ x - 3 * (1 - 2 * x) → x ≤ -3 :=
by
  sorry

end solve_inequality_l463_463783


namespace probability_normal_within_range_l463_463228

noncomputable def normalDistribution (μ σ : ℝ) : MeasureTheory.ProbMeasure ℝ :=
  MeasureTheory.ProbMeasure.toMeasure (MeasureTheory.gaussian μ σ)

theorem probability_normal_within_range (σ : ℝ) (hσ : σ > 0) :
  let ξ := MeasureTheory.ProbMeasure.sample (normalDistribution 0 σ),
  (MeasureTheory.ProbMeasure.prob (λ x, x > 2) ξ = 0.023) →
  MeasureTheory.ProbMeasure.prob (λ x, -2 ≤ x ∧ x ≤ 2) ξ = 0.954 := by
  sorry

end probability_normal_within_range_l463_463228


namespace find_n_l463_463591

theorem find_n : ∃ (n : ℤ), -150 < n ∧ n < 150 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1600 * Real.pi / 180) :=
sorry

end find_n_l463_463591


namespace sequence_values_l463_463371

theorem sequence_values (x y z : ℕ) 
    (h1 : x = 14 * 3) 
    (h2 : y = x - 1) 
    (h3 : z = y * 3) : 
    x = 42 ∧ y = 41 ∧ z = 123 := by 
    sorry

end sequence_values_l463_463371


namespace distance_is_eight_l463_463490

/-- Define the problem conditions -/
def chord_length : ℝ := 16
def arc_angle : ℝ := 90

/-- Define the radius of the circle -/
noncomputable def radius (chord_length : ℝ) (arc_angle : ℝ) : ℝ :=
  (chord_length / 2) / real.sin (real.pi * arc_angle / 360)

/-- Define the distance from the center to the chord. -/
noncomputable def distance_from_center (chord_length : ℝ) (arc_angle : ℝ) : ℝ :=
  radius chord_length arc_angle * real.cos (real.pi * arc_angle / 360)

/-- The theorem stating the distance from the center to the chord is 8. -/
theorem distance_is_eight (h_chord : chord_length = 16) (h_angle : arc_angle = 90) :
  distance_from_center chord_length arc_angle = 8 :=
by {
  sorry,
}

end distance_is_eight_l463_463490


namespace blue_area_factor_12_l463_463821

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l463_463821


namespace updated_mean_correct_l463_463755

noncomputable def original_mean : ℝ := 210
noncomputable def original_observations : ℕ := 60
noncomputable def corrected_observations : ℕ := 58

noncomputable def sum_original : ℝ := original_mean * original_observations

-- Adjust sum for the two observations that were not recorded
noncomputable def sum_after_missing : ℝ := sum_original - (original_mean * 2)

-- Adjust sum for the three incorrect increments
noncomputable def sum_after_increments: ℝ := sum_after_missing - (3 * 10)

-- Adjust sum for the six incorrect decrements
noncomputable def sum_after_decrements: ℝ := sum_after_increments + (6 * 6)

-- Calculate the updated mean
noncomputable def updated_mean : ℝ := sum_after_decrements / corrected_observations

theorem updated_mean_correct :
  updated_mean ≈ 210.10 := 
sorry

end updated_mean_correct_l463_463755


namespace largest_sum_of_digits_l463_463669

theorem largest_sum_of_digits :
  ∃ (a b c z : ℕ), (a ≤ 9) ∧ (b ≤ 9) ∧ (c ≤ 9) ∧ (0 < z ∧ z ≤ 12) ∧ (0.abc = 1 / (z : ℚ)) ∧ 
  (∀ a' b' c' z', (a' ≤ 9) ∧ (b' ≤ 9) ∧ (c' ≤ 9) ∧ (0 < z' ∧ z' ≤ 12) ∧ (0.abc = 1 / (z' : ℚ)) → (a + b + c ≥ a' + b' + c')) :=
by
  sorry

end largest_sum_of_digits_l463_463669


namespace perpendicular_AZ_BC_l463_463712

-- Definitions of points based on midpoints and semicircles
def midpoint (A B : Point) : Point := sorry -- Placeholder for midpoint formula
def semicircle (A B : Point) : Set Point := sorry -- Placeholder for semicircle points
def tangent_at (semicircle : Set Point) (P : Point) : Set Point := sorry -- Placeholder for tangent lines

-- Given triangle ABC
variables {A B C M N K X Y Z : Point}

-- Given conditions
axiom BC_midpoint : M = midpoint B C
axiom AC_midpoint : N = midpoint A C
axiom AB_midpoint : K = midpoint A B

axiom semicircle_AB : ∀ P, P ∈ semicircle A B → sorry
axiom semicircle_AC : ∀ P, P ∈ semicircle A C → sorry

axiom MK_intersects_semi_AB : X ∈ (semicircle A B) ∧ X ∈ line_through M K
axiom MN_intersects_semi_AC : Y ∈ (semicircle A C) ∧ Y ∈ line_through M N

axiom tangents_intersect : Z ∈ tangent_at (semicircle A B) X ∧ Z ∈ tangent_at (semicircle A C) Y

-- Statement to prove 
theorem perpendicular_AZ_BC : line_perpendicular (line_through A Z) (line_through B C) :=
sorry

end perpendicular_AZ_BC_l463_463712


namespace three_f_x_eq_l463_463295

theorem three_f_x_eq (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 2 / (3 + x)) (x : ℝ) (hx : x > 0) : 
  3 * f x = 18 / (9 + x) := sorry

end three_f_x_eq_l463_463295


namespace unique_elements_of_A_n_l463_463599

def A_n (n : ℕ) : Set ℝ := {x | ∃ k : ℕ, k ≤ n ∧ x = Real.sin (k * Real.pi / n)}

theorem unique_elements_of_A_n (n : ℕ) (hpos : 0 < n) :
  (A_n n).card = 4 → (n = 6 ∨ n = 7) :=
by
  sorry

end unique_elements_of_A_n_l463_463599


namespace evaluate_P7_eq_zero_l463_463365

noncomputable def P (a b c d e f : ℝ) : Polynomial ℝ :=
  (2 * Polynomial.X ^ 4 - 26 * Polynomial.X ^ 3 + a * Polynomial.X ^ 2 + b * Polynomial.X + c) *
  (5 * Polynomial.X ^ 4 - 80 * Polynomial.X ^ 3 + d * Polynomial.X ^ 2 + e * Polynomial.X + f)

theorem evaluate_P7_eq_zero (a b c d e f : ℝ)
  (hroots : ∀ x : ℂ, x ∈ [{1, 2, 3, 4, (1 : ℝ) / 2, (3 : ℝ) / 2}] → Polynomial.eval complex.of_real x P a b c d e f = 0) :
  Polynomial.eval 7 (P a b c d e f) = 0 :=
by {
  sorry
}

end evaluate_P7_eq_zero_l463_463365


namespace solve_log_eq_l463_463392

theorem solve_log_eq {x : ℝ} 
  (h : log 2 ((4 * x + 12) / (6 * x - 4)) + log 2 ((6 * x - 4) / (2 * x - 3)) = 3) :
  x = 3 := 
by
  sorry

end solve_log_eq_l463_463392


namespace integral_absolute_difference_zero_l463_463725

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)
variable (a : ℝ)
variable (I : ℝ → ℝ)

-- Define the function f(x) = x^2 + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the function g(x) = a*x + 3, aligning with the line equation condition
def g (a : ℝ) (x : ℝ) : ℝ := a * x + 3

-- Define the integral I(a) = 3 * ∫_{-1}^1 |f(x) - g(a, x)| dx
noncomputable def I (a : ℝ) : ℝ := 3 * ∫ x in -1..1, |f x - g a x| 

-- State the theorem to prove that I(a) = 0 for all a ∈ ℝ
theorem integral_absolute_difference_zero : ∀ a : ℝ, I a = 0 := by
  sorry

end integral_absolute_difference_zero_l463_463725


namespace incorrect_statement_d_l463_463918

variable (α : ℝ) (k : ℤ)

/-- The set of angles with terminal sides on the x-axis -/
def angles_on_x_axis : set ℝ := {α | ∃ k : ℤ, α = k * real.pi}

/-- The set of angles with terminal sides on the y-axis -/
def angles_on_y_axis : set ℝ := {α | ∃ k : ℤ, α = real.pi / 2 + k * real.pi}

/-- The set of angles with terminal sides on the coordinate axes -/
def angles_on_coordinate_axes : set ℝ := {α | ∃ k : ℤ, α = k * (real.pi / 2)}

/-- The set of angles with terminal sides on the line y = x -/
def angles_on_y_eq_x_line : set ℝ := {α | ∃ k : ℤ, α = real.pi / 4 + 2 * k * real.pi}

/-- The set of angles with terminal sides on the line y = -x -/
def angles_on_y_eq_minus_x_line : set ℝ := {α | ∃ k : ℤ, α = 3 * real.pi / 4 + k * real.pi}

theorem incorrect_statement_d :
  ¬angles_on_y_eq_x_line = angles_on_y_eq_minus_x_line := by
  sorry

end incorrect_statement_d_l463_463918


namespace blue_area_factor_12_l463_463824

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l463_463824


namespace problem_statement_l463_463489

variables {totalBuyers : ℕ}
variables {C M K CM CK MK CMK : ℕ}

-- Given conditions
def conditions (totalBuyers : ℕ) (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : Prop :=
  totalBuyers = 150 ∧
  C = 70 ∧
  M = 60 ∧
  K = 50 ∧
  CM = 25 ∧
  CK = 15 ∧
  MK = 10 ∧
  CMK = 5

-- Number of buyers who purchase at least one mixture
def buyersAtLeastOne (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : ℕ :=
  C + M + K - CM - CK - MK + CMK

-- Number of buyers who purchase none
def buyersNone (totalBuyers : ℕ) (buyersAtLeastOne : ℕ) : ℕ :=
  totalBuyers - buyersAtLeastOne

-- Probability computation
def probabilityNone (totalBuyers : ℕ) (buyersNone : ℕ) : ℚ :=
  buyersNone / totalBuyers

-- Theorem statement
theorem problem_statement : conditions totalBuyers C M K CM CK MK CMK →
  probabilityNone totalBuyers (buyersNone totalBuyers (buyersAtLeastOne C M K CM CK MK CMK)) = 0.1 :=
by
  intros h
  -- Assumptions from the problem
  have h_total : totalBuyers = 150 := h.left
  have hC : C = 70 := h.right.left
  have hM : M = 60 := h.right.right.left
  have hK : K = 50 := h.right.right.right.left
  have hCM : CM = 25 := h.right.right.right.right.left
  have hCK : CK = 15 := h.right.right.right.right.right.left
  have hMK : MK = 10 := h.right.right.right.right.right.right.left
  have hCMK : CMK = 5 := h.right.right.right.right.right.right.right
  sorry

end problem_statement_l463_463489


namespace cider_apples_production_l463_463494

def apples_total : Real := 8.0
def baking_fraction : Real := 0.30
def cider_fraction : Real := 0.60

def apples_remaining : Real := apples_total * (1 - baking_fraction)
def apples_for_cider : Real := apples_remaining * cider_fraction

theorem cider_apples_production : 
    apples_for_cider = 3.4 := 
by
  sorry

end cider_apples_production_l463_463494


namespace line_circle_relationship_l463_463577

noncomputable def point_on_line (k : ℝ) : Prop :=
  (0 : ℝ, 1 : ℝ) ∈ {p | p.2 = k * p.1 + 1}

noncomputable def point_in_circle : Prop :=
  (0 : ℝ)^2 + (1 : ℝ)^2 < 2

noncomputable def center_of_circle_in_line (k : ℝ) : Prop :=
  (0 : ℝ, 0 : ℝ) ∈ {p | p.2 = k * p.1 + 1}

noncomputable def line_intersects_circle (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 = 2 ∧ y = k * x + 1

theorem line_circle_relationship (k : ℝ) :
  point_on_line k ∧ point_in_circle ∧ ¬center_of_circle_in_line k → line_intersects_circle k :=
by
  sorry

end line_circle_relationship_l463_463577


namespace solve_congruence_l463_463781

theorem solve_congruence (y : ℤ) : 10 * y + 3 ≡ 7 [MOD 18] → y ≡ 4 [MOD 9] := by
  sorry

end solve_congruence_l463_463781


namespace smallest_munificence_monic_quadratic_l463_463600

-- Definition: Munificence of a polynomial
def munificence (p : ℝ → ℝ) : ℝ :=
  Sup (set.image (abs ∘ p) (set.Icc (-2 : ℝ) 2))

-- Problem statement: Prove the smallest possible munificence of a monic quadratic polynomial
theorem smallest_munificence_monic_quadratic :
  ∃ (p : ℝ → ℝ), (∀ x, p x = x^2 + (b : ℝ) * x + (c : ℝ)) ∧
  ∀ q, (∀ x, q x = x^2 + (d : ℝ) * x + (e : ℝ)) → munificence p ≤ munificence q ∧ munificence p = 2 :=
by
  sorry

end smallest_munificence_monic_quadratic_l463_463600


namespace incorrect_conclusion_l463_463105

theorem incorrect_conclusion (b x : ℂ) (h : x^2 - b * x + 1 = 0) : x = 1 ∨ x = -1
  ↔ (b = 2 ∨ b = -2) :=
by sorry

end incorrect_conclusion_l463_463105


namespace sum_of_divisors_of_8_eq_8_l463_463740

-- Definition of a positive divisor
def is_divisor (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

-- Sum of all positive divisors of a given number
def sum_divisors (a : ℕ) : ℕ :=
  (Finset.range (a + 1)).filter (is_divisor a) |>.sum id

-- The proof statement, without the proof itself
theorem sum_of_divisors_of_8_eq_8 : sum_divisors 8 = 8 :=
by
  -- We do not provide the proof here, just the statement
  sorry

end sum_of_divisors_of_8_eq_8_l463_463740


namespace red_blue_area_ratio_is_12_l463_463804

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l463_463804


namespace smaller_angle_at_3_30_pm_l463_463025

noncomputable def angle_between_clock_hands (h : ℕ) (m : ℕ) : ℝ :=
let minute_angle : ℝ := (m * 6)
    hour_angle : ℝ := (h % 12 * 30 + m * 0.5)
    angle : ℝ := abs (minute_angle - hour_angle) in
min angle (360 - angle)

theorem smaller_angle_at_3_30_pm :
  angle_between_clock_hands 3 30 = 75 :=
by
  sorry

end smaller_angle_at_3_30_pm_l463_463025


namespace max_intersections_l463_463894

theorem max_intersections (C₁ C₂ C₃ : Circle) (L : Line) : 
  greatest_points_of_intersection 3 1 = 12 :=
sorry

end max_intersections_l463_463894


namespace jason_total_expenditure_l463_463347

theorem jason_total_expenditure :
  let stove_cost := 1200
  let wall_repair_ratio := 1 / 6
  let wall_repair_cost := stove_cost * wall_repair_ratio
  let total_cost := stove_cost + wall_repair_cost
  total_cost = 1400 := by
  {
    sorry
  }

end jason_total_expenditure_l463_463347


namespace min_value_of_expression_l463_463184

noncomputable def minExpression (x : ℝ) : ℝ := (15 - x) * (14 - x) * (15 + x) * (14 + x)

theorem min_value_of_expression : ∀ x : ℝ, ∃ m : ℝ, (m ≤ minExpression x) ∧ (m = -142.25) :=
by
  sorry

end min_value_of_expression_l463_463184


namespace smallest_five_digit_int_equiv_4_mod_5_l463_463908

theorem smallest_five_digit_int_equiv_4_mod_5 : 
  ∃ n : ℕ, n ≡ 4 [MOD 5] ∧ n ≥ 10000 ∧ ∀ m : ℕ, (m ≡ 4 [MOD 5] ∧ m ≥ 10000) → n ≤ m :=
begin
  sorry
end

end smallest_five_digit_int_equiv_4_mod_5_l463_463908


namespace intersection_point_l463_463023

variable (x y : ℝ)

-- Definitions given by the conditions
def line1 (x y : ℝ) := 3 * y = -2 * x + 6
def line2 (x y : ℝ) := -2 * y = 6 * x + 4

-- The theorem we want to prove
theorem intersection_point : ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ x = -12/7 ∧ y = 22/7 := 
sorry

end intersection_point_l463_463023


namespace polar_equation_of_curve_C_area_of_triangle_AOB_l463_463652

noncomputable def curve_polar_equation (α θ : ℝ) : ℝ :=
  let x := 2 + sqrt 5 * cos α
  let y := 1 + sqrt 5 * sin α
  (x - 2) ^ 2 + (y - 1) ^ 2 - 5

theorem polar_equation_of_curve_C (θ : ℝ) (h : true) :
  ∃ ρ : ℝ, ρ = 4 * cos θ + 2 * sin θ :=
sorry

theorem area_of_triangle_AOB (A B : ℝ × ℝ) (hA : A = (2 * sqrt 3 + 1, 0))
  (hB : B = (2 + sqrt 3, 0)) (hθ : ∠ (0, 0) A B = π / 6) :
  let OA := dist (0, 0) A
  let OB := dist (0, 0) B
  1 / 2 * OA * OB * sin (π / 6) = (8 + 5 * sqrt 3) / 4 :=
sorry

end polar_equation_of_curve_C_area_of_triangle_AOB_l463_463652


namespace logarithmic_solution_l463_463779

theorem logarithmic_solution (x : ℝ) (h : log 2 x + (log 2 x / log 2 4) + (log 2 x / log 2 8) = 9) : 
  x = 2^(54/11) :=
by
  sorry 

end logarithmic_solution_l463_463779


namespace blue_to_red_face_area_ratio_l463_463858

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l463_463858


namespace positive_product_probability_l463_463448

noncomputable def probability_positive_product : ℝ :=
  let interval := Icc (-15 : ℝ) (20 : ℝ) in
  let interval_length := 20 - (-15) in
  let neg_interval := Icc (-15) 0 in
  let pos_interval := Ioc 0 20 in
  let neg_length := 0 - (-15) in
  let pos_length := 20 - 0 in
  let neg_prob := neg_length / interval_length in
  let pos_prob := pos_length / interval_length in
  (neg_prob * neg_prob) + (pos_prob * pos_prob)

theorem positive_product_probability :
  probability_positive_product = 25 / 49 :=
  sorry

end positive_product_probability_l463_463448


namespace sequence_contains_even_sequence_contains_infinitely_many_evens_l463_463351

-- Define the sequence of positive integers with the given properties
def sequence (x : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then x else sorry -- Placeholder definition

-- Prove that the sequence contains at least one even term
theorem sequence_contains_even (x : ℕ) (hx : x > 0) :
  ∃ n : ℕ, n ≥ 1 ∧ 2 ∣ sequence x n := 
sorry

-- Prove that the sequence contains infinitely many even terms
theorem sequence_contains_infinitely_many_evens (x : ℕ) (hx : x > 0) :
  ∃ᶠ n in (Filter.atTop : Filter ℕ), 2 ∣ sequence x n := 
sorry

end sequence_contains_even_sequence_contains_infinitely_many_evens_l463_463351


namespace polynomial_coeff_is_0_or_1_l463_463350

noncomputable def a_seq (r : ℕ) : ℕ → ℝ
| i := if h : i < r then a_coeff h else 0

noncomputable def b_seq (s : ℕ) : ℕ → ℝ
| j := if h : j < s then b_coeff h else 0

theorem polynomial_coeff_is_0_or_1 {r s : ℕ} (a_coeff b_coeff : ℕ → ℝ) 
  (h_r : r ≥ 1) (h_s : s ≥ 1)
  (h_a_nonneg : ∀ i : ℕ, a_coeff i ≥ 0) 
  (h_b_nonneg : ∀ j : ℕ, b_coeff j ≥ 0)
  (h_poly_eq : (∑ i in finset.range r, a_coeff i * polynomial.X ^ i + polynomial.X ^ r) *
                (∑ j in finset.range s, b_coeff j * polynomial.X ^ j + polynomial.X ^ s) =
                polynomial.C (1 : ℝ) + polynomial.X + ∑ k in finset.range (r + s - 1), polynomial.X ^ k + polynomial.X ^ (r + s)) :
                (∀i, a_seq r a_coeff i ∈ {0, 1} ∧ ∀j, b_seq s b_coeff j ∈ {0, 1}) :=
by
  sorry

end polynomial_coeff_is_0_or_1_l463_463350


namespace cannot_have_triangular_cross_section_l463_463441

-- Definition of the geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | TriangularPrism
  | Cube

-- Theorem statement
theorem cannot_have_triangular_cross_section (s : GeometricSolid) :
  s = GeometricSolid.Cylinder → ¬(∃ c : ℝ^3 → Prop, is_triangular_cross_section s c) := 
by
  intros h
  apply h.rec 
  intro
  sorry

end cannot_have_triangular_cross_section_l463_463441


namespace blue_face_area_factor_l463_463800

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l463_463800


namespace find_a_l463_463262

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2

theorem find_a (a : ℝ) (h_deriv : deriv (f a) 2 = -3/2) : a = 1/2 :=
by
  -- Define the function f(x)
  let f := λ x, Real.log x - a * x^2
  -- Differentiate f(x)
  have h_deriv_f : deriv f = λ x, 1/x - 2 * a * x,
    from (by
      sorry)
  -- Use the given condition
  rw [h_deriv_f] at h_deriv
  -- Plug in x=2
  have h : 1/2 - 4 * a = -3/2, from h_deriv
  -- Solve for a
  linarith

end find_a_l463_463262


namespace evaluate_expression_l463_463160

theorem evaluate_expression : 
  (3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001) :=
by
  sorry

end evaluate_expression_l463_463160


namespace total_profit_l463_463109

theorem total_profit (investment_A investment_B investment_C share_A : ℕ)
  (h_investment_A : investment_A = 6300)
  (h_investment_B : investment_B = 4200)
  (h_investment_C : investment_C = 10500)
  (h_share_A : share_A = 3780) :
  let total_profit := 12600 in total_profit = (investment_A + investment_B + investment_C) * share_A / investment_A := 
by
  sorry

end total_profit_l463_463109


namespace cryptarithm_no_solution_l463_463284

theorem cryptarithm_no_solution :
  ∀ (B O C b M U K J A : ℕ),
  B ≠ O ∧ B ≠ C ∧ B ≠ b ∧ B ≠ M ∧ B ≠ U ∧ B ≠ K ∧ B ≠ J ∧ B ≠ A ∧
  O ≠ C ∧ O ≠ b ∧ O ≠ M ∧ O ≠ U ∧ O ≠ K ∧ O ≠ J ∧ O ≠ A ∧
  C ≠ b ∧ C ≠ M ∧ C ≠ U ∧ C ≠ K ∧ C ≠ J ∧ C ≠ A ∧
  b ≠ M ∧ b ≠ U ∧ b ≠ K ∧ b ≠ J ∧ b ≠ A ∧
  M ≠ U ∧ M ≠ K ∧ M ≠ J ∧ M ≠ A ∧
  U ≠ K ∧ U ≠ J ∧ U ≠ A ∧
  K ≠ J ∧ K ≠ A ∧
  J ≠ A ∧
  B ≥ 1 ∧ B ≤ 9 ∧ O ≥ 1 ∧ O ≤ 9 ∧ C ≥ 1 ∧ C ≤ 9 ∧ b ≥ 1 ∧ b ≤ 9 ∧
  M ≥ 1 ∧ M ≤ 9 ∧ U ≥ 1 ∧ U ≤ 9 ∧ K ≥ 1 ∧ K ≤ 9 ∧ J ≥ 1 ∧ J ≤ 9 ∧
  A ≥ 1 ∧ A ≤ 9 ∧
  ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, i = 45 ∧ -- This is just an inclusion for completeness.
  K + J + A + 2 * C = 29 ∧
  B + O + C + b + M + O + U = 22
  → false := sorry

end cryptarithm_no_solution_l463_463284


namespace distance_borya_vasya_l463_463123

-- Definitions of the houses and distances on the road
def distance_andrey_gena : ℕ := 2450
def race_length : ℕ := 1000

-- Variables to represent the distances
variables (y b : ℕ)

-- Conditions
def start_position := y
def finish_position := b / 2 + 1225

axiom distance_eq : distance_andrey_gena = 2 * y
axiom race_distance_eq : finish_position - start_position = race_length

-- Proving the distance between Borya's and Vasya's houses
theorem distance_borya_vasya :
  ∃ (d : ℕ), d = 450 :=
by
  sorry

end distance_borya_vasya_l463_463123


namespace area_of_red_region_on_larger_sphere_l463_463961

/-- 
A smooth ball with a radius of 1 cm was dipped in red paint and placed between two 
absolutely smooth concentric spheres with radii of 4 cm and 6 cm, respectively
(the ball is outside the smaller sphere but inside the larger sphere).
As the ball moves and touches both spheres, it leaves a red mark. 
After traveling a closed path, a region outlined in red with an area of 37 square centimeters is formed on the smaller sphere. 
Find the area of the region outlined in red on the larger sphere. 
The answer should be 55.5 square centimeters.
-/
theorem area_of_red_region_on_larger_sphere
  (r1 r2 r3 : ℝ)
  (A_small : ℝ)
  (h_red_small_sphere : 37 = 2 * π * r2 * (A_small / (2 * π * r2)))
  (h_red_large_sphere : 55.5 = 2 * π * r3 * (A_small / (2 * π * r2))) :
  ∃ A_large : ℝ, A_large = 55.5 :=
by
  -- Definitions and conditions
  let r1 := 1  -- radius of small ball (1 cm)
  let r2 := 4  -- radius of smaller sphere (4 cm)
  let r3 := 6  -- radius of larger sphere (6 cm)

  -- Given: A small red area is 37 cm^2 on the smaller sphere.
  let A_small := 37

  -- Proof of the relationship of the spherical caps
  sorry

end area_of_red_region_on_larger_sphere_l463_463961


namespace value_of_x_l463_463481

theorem value_of_x (x : ℚ) : 1 / 3 = (5 / 3) / x → x = 5 := 
by {
  intros h,
  sorry
}

end value_of_x_l463_463481


namespace binary_remainder_div_4_is_1_l463_463130

def binary_to_base_10_last_two_digits (b1 b0 : Nat) : Nat :=
  2 * b1 + b0

noncomputable def remainder_of_binary_by_4 (n : Nat) : Nat :=
  match n with
  | 111010110101 => binary_to_base_10_last_two_digits 0 1
  | _ => 0

theorem binary_remainder_div_4_is_1 :
  remainder_of_binary_by_4 111010110101 = 1 := by
  sorry

end binary_remainder_div_4_is_1_l463_463130


namespace triangle_max_area_l463_463446

theorem triangle_max_area (PQ PR QR : ℝ) (hratio : PR / QR = 25 / 24) (hPQ : PQ = 12) : 
  ∃ x : ℝ, x > 0 ∧ PR = 25 * x ∧ QR = 24 * x ∧ 
  ∀ s : ℝ, s = (PQ + PR + QR) / 2 → 
  let area := sqrt (s * (s - PQ) * (s - PR) * (s - QR)) in 
  area ≤ 2592 := 
sorry

end triangle_max_area_l463_463446


namespace find_speed_of_current_l463_463506

variable {m c : ℝ}

theorem find_speed_of_current
  (h1 : m + c = 15)
  (h2 : m - c = 10) :
  c = 2.5 :=
sorry

end find_speed_of_current_l463_463506


namespace tic_tac_toe_winning_probability_l463_463527

theorem tic_tac_toe_winning_probability :
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  probability = (2 : ℚ) / 21 :=
by
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  have probability_correct : probability = (2 : ℚ) / 21 := sorry
  exact probability_correct

end tic_tac_toe_winning_probability_l463_463527


namespace evaluate_expression_l463_463164

open Real

def a := 2999
def b := 3000
def delta := b - a

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 :=
by
  let a := 2999
  let b := 3000
  have h1 : b - a = 1 := by sorry
  calc
    3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = a^3 + b^3 - ab^2 - a^2b := by sorry
                                            ... = (b - a)^2 * (b + a)       := by sorry
                                            ... = (1)^2 * (b + a)           := by
                                                                           rw [h1]
                                                                           exact sorry
                                            ... = 3000 + 2999               := by
                                                                           exact sorry
                                            ... = 5999                     := rfl

end evaluate_expression_l463_463164


namespace exists_sequence_n_not_2_exists_sequence_n_3_exists_sequence_n_gt_3_l463_463217

-- Definitions for the conditions
def sum_fractions (n : ℕ) (a : Fin n → ℕ) : ℚ :=
  (Finset.univ.sum (λ i : Fin n, (a i : ℚ) / (a (i + 1) % n)))

theorem exists_sequence_n_not_2 :
  ¬(∃ (a : Fin 2 → ℕ), (∀ i j, i ≠ j → a i ≠ a j) ∧ (sum_fractions 2 a).denom = 1) :=
sorry

theorem exists_sequence_n_3 :
  ∃ (a : Fin 3 → ℕ), (∀ i j, i ≠ j → a i ≠ a j) ∧ (sum_fractions 3 a).denom = 1 :=
sorry

theorem exists_sequence_n_gt_3 (n : ℕ) (h : n > 3) :
  ∃ (a : Fin n → ℕ), (∀ i j, i ≠ j → a i ≠ a j) ∧ (sum_fractions n a).denom = 1 :=
sorry

end exists_sequence_n_not_2_exists_sequence_n_3_exists_sequence_n_gt_3_l463_463217


namespace probability_exactly_four_of_eight_show_2_l463_463154

noncomputable def dice_probability : ℝ :=
  let n : ℕ := 8
  let k : ℕ := 4
  let p_show_2 : ℝ := 1 / 8
  let p_not_show_2 : ℝ := 7 / 8
  let ways := Nat.choose n k
  (ways * (p_show_2 ^ k) * (p_not_show_2 ^ (n - k)))

theorem probability_exactly_four_of_eight_show_2 : 
  (Real.toRoundedString (dice_probability) 3) = "0.010" :=
by
  sorry

end probability_exactly_four_of_eight_show_2_l463_463154


namespace train_passing_time_l463_463339

theorem train_passing_time (L : ℕ) (v_kmph : ℕ) (v_mps : ℕ) (time : ℕ)
  (h1 : L = 90)
  (h2 : v_kmph = 36)
  (h3 : v_mps = v_kmph * (1000 / 3600))
  (h4 : v_mps = 10)
  (h5 : time = L / v_mps) :
  time = 9 := by
  sorry

end train_passing_time_l463_463339


namespace probability_of_winning_position_l463_463523

theorem probability_of_winning_position : 
    (let total_positions := Nat.choose 9 3,
         winning_positions := 8 in
       (winning_positions : ℚ) / total_positions = 2 / 21) := 
by
  sorry

end probability_of_winning_position_l463_463523


namespace negate_statement_l463_463759

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_statement :
  (∃! x : ℕ, x ∈ {a, b, c} ∧ is_even x) ↔ (at_least_two_even_or_all_odd a b c) :=
sorry

end negate_statement_l463_463759


namespace exists_polyhedron_l463_463932

-- Define vertices as an enum (or type)
inductive Vertex
| A | B | C | D | E | F | G | H

open Vertex

-- Define edges as a list of pairs of vertices
def edges : List (Vertex × Vertex) := [
  (A, B), (A, C), (B, C), (B, D),
  (C, D), (D, E), (E, F), (E, G),
  (F, G), (F, H), (G, H), (A, H)
]

-- Problem statement: Does there exist a polyhedron with the given edges?
theorem exists_polyhedron : ∃ p : polyhedron, p.edges = edges := sorry

end exists_polyhedron_l463_463932


namespace ice_cream_permutations_l463_463117

def flavors : List String := ["vanilla", "chocolate", "strawberry", "cherry", "mint"]

theorem ice_cream_permutations : List.length flavors = 5 → (Nat.factorial 5) = 120 :=
by
  intro h
  rw [Nat.factorial, List.length] at h
  exact h.symm

end ice_cream_permutations_l463_463117


namespace blue_red_area_ratio_l463_463837

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l463_463837


namespace minimize_area_condition_l463_463744

-- Defining the geometric setup and conditions
def triangle (α : Type) [inhabited α] := α × α × α

variables {α : Type} [inhabited α]

def is_acute_triangle (T : triangle α) : Prop := sorry -- define the property of acute triangle

def is_circumcenter (O : α) (T : triangle α) : Prop := sorry -- define the property of circumcenter

def perpendicular (X Y Z : α) : Prop := sorry -- define perpendicularity condition

def inscribed_circle_center (T : triangle α) : α := sorry -- define center of the circumscribed circle

def minor_arc (T : triangle α) (X : α) : Prop := sorry -- define what it means for X to be on the minor arc AB

-- Definition of the problem
def problem_statement (T : triangle α) (O : α) (O₁ O₂ : α) (X D : α) : Prop :=
  is_acute_triangle T ∧
  is_circumcenter O₁ (T.1, D, X) ∧
  is_circumcenter O₂ (T.2, D, X) ∧
  perpendicular O O₁ O₂

-- The lean theorem to prove
theorem minimize_area_condition :
  ∀ (T : triangle α) (O : α) (O₁ O₂ : α) (X D : α),
    is_acute_triangle T →
    minor_arc T X →
    (segment T.3 X D) ∧
    is_circumcenter O₁ (T.1, D, X) →
    is_circumcenter O₂ (T.2, D, X) →
    (area (O, O₁, O₂) is_minimal_if_perpendicular_to_segment (T.1, T.2, O₁, O₂)) ↔ 
    perpendicular T.3 T.1 T.2 sorry

end minimize_area_condition_l463_463744


namespace average_weight_of_abc_l463_463792

theorem average_weight_of_abc 
  (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 46)
  (h3 : B = 37) :
  (A + B + C) / 3 = 45 := 
by
  sorry

end average_weight_of_abc_l463_463792


namespace difference_of_values_l463_463022

noncomputable def sqrt_of_90_percent_of_40 : ℚ := real.sqrt (0.9 * 40)
def four_fifths_of_two_thirds_of_25 : ℚ := (4 / 5) * ((2 / 3) * 25)

theorem difference_of_values : 
  sqrt_of_90_percent_of_40 - four_fifths_of_two_thirds_of_25 = -22 / 3 :=
by
  sorry

end difference_of_values_l463_463022


namespace phase_shift_to_cosine_l463_463442

-- Define the two functions involved
def f1 (x : ℝ) : ℝ := 2 * Real.sin (2 * x)
def f2 (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 6)

-- Define the shift transformation
def shifted (g : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := g (x + a)

-- Proof statement: to obtain f2 from f1, shift f1 by π/6 units to the left
theorem phase_shift_to_cosine : f2 = shifted f1 (-Real.pi / 6) := 
sorry

end phase_shift_to_cosine_l463_463442


namespace tic_tac_toe_probability_l463_463536

theorem tic_tac_toe_probability :
  let total_positions := Nat.choose 9 3,
      winning_positions := 8 in
  (winning_positions / total_positions : ℚ) = 2 / 21 :=
by
  sorry

end tic_tac_toe_probability_l463_463536


namespace least_of_consecutive_odd_integers_l463_463307

-- Definitions and conditions
def consecutive_odd_integers (a : ℤ) (n : ℕ) : list ℤ :=
  list.map (λ k, a + k * 2) (list.range n)

def average (lst : list ℤ) : ℤ :=
  lst.sum / lst.length

-- Problem statement
theorem least_of_consecutive_odd_integers 
  (a : ℤ)
  (n : ℕ)
  (h_n : n = 102)
  (h_avg : average (consecutive_odd_integers a n) = 414) :
  (consecutive_odd_integers a n).head = 313 :=
sorry

end least_of_consecutive_odd_integers_l463_463307


namespace ratio_of_larger_to_smaller_l463_463429

theorem ratio_of_larger_to_smaller (x y : ℝ) (h_pos : 0 < y) (h_ineq : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by 
  sorry

end ratio_of_larger_to_smaller_l463_463429


namespace James_total_passengers_l463_463719

def trucks := 12
def buses := 2
def taxis := 2 * buses
def cars := 30
def motorbikes := 52 - trucks - buses - taxis - cars

def people_in_truck := 2
def people_in_bus := 15
def people_in_taxi := 2
def people_in_motorbike := 1
def people_in_car := 3

def total_passengers := 
  trucks * people_in_truck + 
  buses * people_in_bus + 
  taxis * people_in_taxi + 
  motorbikes * people_in_motorbike + 
  cars * people_in_car

theorem James_total_passengers : total_passengers = 156 := 
by 
  -- Placeholder proof, needs to be completed
  sorry

end James_total_passengers_l463_463719


namespace min_supreme_supervisors_l463_463010

-- Definitions
def num_employees : ℕ := 50000
def supervisors (e : ℕ) : ℕ := 7 - e

-- Theorem statement
theorem min_supreme_supervisors (k : ℕ) (num_employees_le_reached : ∀ n : ℕ, 50000 ≤ n) : 
  k ≥ 28 := 
sorry

end min_supreme_supervisors_l463_463010


namespace eval_expression_l463_463301

theorem eval_expression (x y z : ℝ) (h1 : y > z) (h2 : z > 0) (h3 : x = y + z) : 
  ( (y+z+y)^z + (y+z+z)^y ) / (y^z + z^y) = 2^y + 2^z :=
by
  sorry

end eval_expression_l463_463301


namespace calculate_expression_l463_463028

theorem calculate_expression : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end calculate_expression_l463_463028


namespace fib_mod_150_eq_8_l463_463789

def fib_mod_9 (n : ℕ) : ℕ :=
  (Nat.fib n) % 9

theorem fib_mod_150_eq_8 : fib_mod_9 150 = 8 :=
  sorry

end fib_mod_150_eq_8_l463_463789


namespace volume_first_cube_l463_463432

theorem volume_first_cube (a b : ℝ) (h_ratio : a = 3 * b) (h_volume : b^3 = 8) : a^3 = 216 :=
by
  sorry

end volume_first_cube_l463_463432


namespace triangle_BDE_area_l463_463379

theorem triangle_BDE_area (A B C D E M : Point) (hD_on_AC : D ∈ line A C)
  (hE_on_AC : E ∈ line A C) (hBD_tri_sect : is_trisect BD (med AM))
  (hBE_tri_sect : is_trisect BE (med AM)) (hABC_area : area ABC = 1) :
  area BDE = 0.3 :=
sorry

end triangle_BDE_area_l463_463379


namespace angle_between_AP_and_PE_is_90_l463_463872

/-
The right triangular prism \(ABC-A_1B_1C_1\) has a base \( \triangle ABC \) which is an equilateral triangle.
Points \( P \) and \( E \) are movable points on \( BB_1 \) and \( CC_1 \) respectively.
\( D \) is the midpoint of side \( BC \), and \( PD \perp PE \).
Prove that the angle between lines \( AP \) and \( PE \) is \( 90^\circ \).
-/

variables {A A1 B B1 C C1 P E D : Type*}
variables [triangle_prism ABC ABC_A1B1C1 A A1 B B1 C C1]
variables [equilateral_triangle ABC]
variables [point_on BB1 P] [point_on CC1 E]
variables [midpoint D BC]
variables [perpendicular PD PE]

theorem angle_between_AP_and_PE_is_90 :
  angle_between AP PE = 90 :=
sorry

end angle_between_AP_and_PE_is_90_l463_463872


namespace not_possible_primes_after_1996_iters_l463_463452

noncomputable def transform (a b c d : ℤ) : ℤ × ℤ × ℤ × ℤ :=
  (a - b, b - c, c - d, d - a)

theorem not_possible_primes_after_1996_iters (a b c d : ℤ) :
  let final_vals := (fin.iterate 1996 (λ x : ℤ × ℤ × ℤ × ℤ, transform x.1 x.2 x.3 x.4) (a, b, c, d))
  in
  let A := final_vals.1
  let B := final_vals.2
  let C := final_vals.3
  let D := final_vals.4
  in ¬(nat.prime (nat_abs (B * C - A * D)) ∧ nat.prime (nat_abs (A * C - B * D)) ∧ nat.prime (nat_abs (A * B - C * D))) :=
sorry

end not_possible_primes_after_1996_iters_l463_463452


namespace remainder_when_abc_divided_by_7_l463_463293

theorem remainder_when_abc_divided_by_7 (a b c : ℕ) (h0 : a < 7) (h1 : b < 7) (h2 : c < 7)
  (h3 : (a + 2 * b + 3 * c) % 7 = 0)
  (h4 : (2 * a + 3 * b + c) % 7 = 4)
  (h5 : (3 * a + b + 2 * c) % 7 = 4) :
  (a * b * c) % 7 = 6 := 
sorry

end remainder_when_abc_divided_by_7_l463_463293


namespace sin1993_cos1993_leq_zero_l463_463633

theorem sin1993_cos1993_leq_zero (x : ℝ) (h : Real.sin x + Real.cos x ≤ 0) : 
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := 
by 
  sorry

end sin1993_cos1993_leq_zero_l463_463633


namespace transform_cos_to_sin_shift_l463_463443

theorem transform_cos_to_sin_shift (x : ℝ) :
  ∃ (shift : ℝ), (∀ x, cos (2 * x + π / 3) = sin (2 * (x + shift))) ∧ shift = 5 * π / 12 :=
by
  sorry

end transform_cos_to_sin_shift_l463_463443


namespace blue_faces_ratio_l463_463849

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l463_463849


namespace eccentricity_of_ellipse_equation_of_ellipse_l463_463256

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b ∧ b > 0) (M : ℝ × ℝ) 
(hM: M.1 = 2 * a / 3 ∧ M.2 = b / 3) (hm_slope: M.2 / M.1 = 1 / 4) : ℝ :=
  real.sqrt (1 - (b / a) ^ 2)

theorem eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) (M : ℝ × ℝ)
(hM: M.1 = 2 * a / 3 ∧ M.2 = b / 3) (hm_slope: M.2 / M.1 = 1 / 4) 
: ellipse_eccentricity a b h M hM hm_slope = real.sqrt 3 / 2 := by {
  sorry
}

def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) (E : ℝ × ℝ → Prop) : Prop :=
  ∀ P Q : ℝ × ℝ, (E P ∧ E Q) → 
  (∃ C : ℝ × ℝ, (C.1 + 2)^2 + (C.2 - 1)^2 = 15 / 2 ∧ 
   dist P Q = real.sqrt 30)

theorem equation_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) (E : ℝ × ℝ → Prop)
(hE : E = λ x, x.1^2 / 20 + x.2^2 / 5 = 1) : 
ellipse_equation a b h E := by {
  sorry
}

end eccentricity_of_ellipse_equation_of_ellipse_l463_463256


namespace cubic_function_inverse_sum_range_l463_463641

theorem cubic_function_inverse_sum_range (m n p a b c : ℝ) 
  (h_zeroes : ∀ x, x = a ∨ x = b ∨ x = c → f x = 0)
  (h_function : f = λ x, x^3 + m * x^2 + n * x + p)
  (h_f_neg : f (-1) < 0 ∧ f 2 < 0)
  (h_f_pos : f 1 > 0 ∧ f 4 > 0) :
  ∃ (a b c : ℝ), (f = λ x, (x - a) * (x - b) * (x - c)) ∧ ((2 : ℝ) < p ∧ p < (7 : ℝ)) ∧ 
  let s := (1/a + 1/b + 1/c) in 
  s ∈ (-3/4 : ℝ) .. -3/14 :=
sorry

end cubic_function_inverse_sum_range_l463_463641


namespace tangent_line_at_1_e_l463_463180

-- Define the function y = x * exp (2 * x - 1)
def f (x : ℝ) : ℝ := x * Real.exp (2 * x - 1)

-- Define the tangent line equation
def tangent_line (m x₁ y₁ x y : ℝ) : Prop := y - y₁ = m * (x - x₁)

-- Prove the equation of the tangent line at point (1, e)
theorem tangent_line_at_1_e : ∀ (x y : ℝ), tangent_line (3 * Real.exp 1) 1 (Real.exp 1) x y ↔ 3 * Real.exp (1 : ℝ) * x - y - 2 * Real.exp 1 = 0 :=
by
  intro x y
  sorry

end tangent_line_at_1_e_l463_463180


namespace product_of_solutions_l463_463907

theorem product_of_solutions :
  let a := 3
  let b := 18
  let c := -12
  ∃ x₁ x₂ : ℝ, (3 * x₁^2 + 18 * x₁ - 12 = 0) ∧ (3 * x₂^2 + 18 * x₂ - 12 = 0) ∧ (x₁ * x₂ = -4) := sorry

end product_of_solutions_l463_463907


namespace intersection_A_B_l463_463239

def A : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x^2) }
def B : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 1 } := by
  sorry

end intersection_A_B_l463_463239


namespace james_passenger_count_l463_463716

theorem james_passenger_count :
  ∀ (total_vehicles trucks buses taxis motorbikes cars trucks_population buses_population taxis_population motorbikes_population cars_population : ℕ),
  total_vehicles = 52 →
  trucks = 12 →
  buses = 2 →
  taxis = 2 * buses →
  motorbikes = total_vehicles - (trucks + buses + taxis + cars) →
  cars = 30 →
  trucks_population = 2 →
  buses_population = 15 →
  taxis_population = 2 →
  motorbikes_population = 1 →
  cars_population = 3 →
  (trucks * trucks_population + buses * buses_population + taxis * taxis_population +
   motorbikes * motorbikes_population + cars * cars_population) = 156 := 
by
  -- Placeholder for the proof
  sorry

end james_passenger_count_l463_463716


namespace solve_price_reduction_l463_463080

-- Definitions based on conditions
def daily_sales_volume (x : ℝ) : ℝ := 30 + 2 * x
def profit_per_item (x : ℝ) : ℝ := 50 - x
def daily_profit (x : ℝ) : ℝ := (50 - x) * (30 + 2 * x)

-- Statement
theorem solve_price_reduction :
  ∃ x : ℝ, daily_profit x = 2100 ∧ x ∈ {15, 20} :=
begin
  -- solution here
  sorry,
end

end solve_price_reduction_l463_463080


namespace max_intersection_points_l463_463901

theorem max_intersection_points (c1 c2 c3 : Circle) (l : Line) : 
  ∃ n, n = 12 ∧ maximum_intersection_points c1 c2 c3 l = n :=
sorry

end max_intersection_points_l463_463901


namespace rhombus_properties_l463_463512

-- Definition of conditions
def d1 : ℕ := 18
def d2 : ℕ := 14

-- Definition of area and side length using the given conditions
def areaOfRhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2
def sideLengthOfRhombus (d1 d2 : ℕ) : Real := 
  Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)

-- The proof statement
theorem rhombus_properties : 
  areaOfRhombus d1 d2 = 126 ∧ sideLengthOfRhombus d1 d2 = Real.sqrt 130 :=
by
  sorry

end rhombus_properties_l463_463512


namespace sum_positive_integer_values_c_l463_463198

theorem sum_positive_integer_values_c :
  (∑ c in (finset.filter (λ c : ℤ, is_integer c ∧ c > 0)
    (finset.image (λ k : ℤ, (49 - k^2) / 12)
      (finset.range 7))), c) = 6 :=
sorry

end sum_positive_integer_values_c_l463_463198


namespace quadrilateral_cyclic_l463_463743

variables {A B C D L M N P : Point}
variables (k₁ k₂ k₃ k₄ : Circle)

-- Define that ABCD is a square
axiom square_ABCD : is_square A B C D

-- Define circles passing through the square's vertices
axiom k₁_passes_through : passes_through k₁ A ∧ passes_through k₁ B
axiom k₂_passes_through : passes_through k₂ B ∧ passes_through k₂ C
axiom k₃_passes_through : passes_through k₃ C ∧ passes_through k₃ D
axiom k₄_passes_through : passes_through k₄ D ∧ passes_through k₄ A

-- Define centers of circles are outside the square
axiom k₁_center_outside : outside_square k₁.center A B C D
axiom k₂_center_outside : outside_square k₂.center A B C D
axiom k₃_center_outside : outside_square k₃.center A B C D
axiom k₄_center_outside : outside_square k₄.center A B C D

-- Define intersections of the circles within the square
axiom k₄_k₁_intersect : intersection_points_inside_square k₄ k₁ A B C D = {L}
axiom k₁_k₂_intersect : intersection_points_inside_square k₁ k₂ A B C D = {M}
axiom k₂_k₃_intersect : intersection_points_inside_square k₂ k₃ A B C D = {N}
axiom k₃_k₄_intersect : intersection_points_inside_square k₃ k₄ A B C D = {P}

-- The theorem to be proved
theorem quadrilateral_cyclic : is_cyclic_quadrilateral L M N P :=
by
  sorry

end quadrilateral_cyclic_l463_463743


namespace problem_statement_l463_463746

variable {a b c d m n : ℝ}

def P (a b c d : ℝ) : ℝ := Real.sqrt (a * b) + Real.sqrt (c * d)
def Q (a b c d m n : ℝ) : ℝ := 
  Real.sqrt (m * a + n * c) * Real.sqrt ((b / m) + (d / n))

theorem problem_statement (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
                          (hm : 0 < m) (hn : 0 < n) :
  P a b c d ≤ Q a b c d := by
  sorry

end problem_statement_l463_463746


namespace row_col_average_ratio_l463_463571

theorem row_col_average_ratio :
  ∀ (a : ℕ → ℕ → ℝ), 
    (∀ i j, 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 100) →
    (let C := (∑ i in (finRange 50), ∑ j in (finRange 100), a i j) / 50 in
     let D := (∑ j in (finRange 100), ∑ i in (finRange 50), a i j) / 100 in
     C / D = 2) :=
begin
  intros a h,
  let C := (∑ i in (finRange 50), ∑ j in (finRange 100), a i j) / 50,
  let D := (∑ j in (finRange 100), ∑ i in (finRange 50), a i j) / 100,
  sorry
end

end row_col_average_ratio_l463_463571


namespace total_miles_traveled_l463_463750

-- Define the conditions
def travel_time_per_mile (n : ℕ) : ℕ :=
  match n with
  | 0 => 10
  | _ => 10 + 6 * n

def daily_miles (n : ℕ) : ℕ :=
  60 / travel_time_per_mile n

-- Statement of the problem
theorem total_miles_traveled : (daily_miles 0 + daily_miles 1 + daily_miles 2 + daily_miles 3 + daily_miles 4) = 20 := by
  sorry

end total_miles_traveled_l463_463750


namespace probability_at_least_one_bean_distribution_of_X_expectation_of_X_l463_463582

noncomputable def total_ways := Nat.choose 6 3
noncomputable def ways_select_2_egg_1_bean := (Nat.choose 4 2) * (Nat.choose 2 1)
noncomputable def ways_select_1_egg_2_bean := (Nat.choose 4 1) * (Nat.choose 2 2)
noncomputable def at_least_one_bean_probability := (ways_select_2_egg_1_bean + ways_select_1_egg_2_bean) / total_ways

theorem probability_at_least_one_bean : at_least_one_bean_probability = 4 / 5 :=
by sorry

noncomputable def p_X_eq_0 := (Nat.choose 4 3) / total_ways
noncomputable def p_X_eq_1 := ways_select_2_egg_1_bean / total_ways
noncomputable def p_X_eq_2 := ways_select_1_egg_2_bean / total_ways

theorem distribution_of_X : p_X_eq_0 = 1 / 5 ∧ p_X_eq_1 = 3 / 5 ∧ p_X_eq_2 = 1 / 5 :=
by sorry

noncomputable def E_X := (0 * p_X_eq_0) + (1 * p_X_eq_1) + (2 * p_X_eq_2)

theorem expectation_of_X : E_X = 1 :=
by sorry

end probability_at_least_one_bean_distribution_of_X_expectation_of_X_l463_463582


namespace number_of_roots_l463_463621

-- Define the function f with given properties
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ [-1,1] then
    x^2
  else if h : x ∈ [-2,-1] then
    (x + 2)^2
  else if h : x ∈ [1,2] then
    (x - 2)^2
  else
    -- Use periodicity
    f (x - 2 * real.floor (x / 2))

-- Define the given conditions
def even_function (x : ℝ) : Prop := f(x) = f(-x)
def periodic_function (x : ℝ) : Prop := f(x - 1) = f(x + 1)

-- Define the root finding equation
def root_eq (x : ℝ) : Prop := f(x) = (1 / 10)^(|x|)

-- The theorem to prove
theorem number_of_roots : ∃ n : ℕ, n = 4 ∧
  (∃ l : ℝ -> bool, ∀ x ∈ [-2, 3], root_eq x ↔ l x = tt) ∧
  (∀ l : ℝ -> bool, (∀ x ∈ [-2, 3], root_eq x ↔ l x = tt) → (Σ' x, l x = tt) = n) :=
begin
  sorry
end

end number_of_roots_l463_463621


namespace intersection_of_log_functions_l463_463150

theorem intersection_of_log_functions : 
  ∃ x : ℝ, (3 * Real.log x = Real.log (3 * x)) ∧ x = Real.sqrt 3 := 
by 
  sorry

end intersection_of_log_functions_l463_463150


namespace number_of_zeros_of_g_l463_463288

open Real

noncomputable def g (x : ℝ) : ℝ := cos (π * log x + x)

theorem number_of_zeros_of_g : ¬ ∃ (x : ℝ), 1 < x ∧ x < exp 2 ∧ g x = 0 :=
sorry

end number_of_zeros_of_g_l463_463288


namespace max_score_combinations_l463_463721

noncomputable def table_tennis_combinations : ℕ := sorry

axiom win_conditions (α β : ℕ) : Prop :=
(α = 11 ∧ β < 10 ∨ β = 11 ∧ α < 10) ∨ (α ≥ 10 ∧ β ≥ 10 ∧ (α - β = 2 ∨ β - α = 2))

axiom total_points : 3 * 11 ≥ 30

theorem max_score_combinations : table_tennis_combinations = 16 := sorry

end max_score_combinations_l463_463721


namespace polar_to_rectangular_l463_463573

theorem polar_to_rectangular (ρ θ : ℝ) :
  ρ = sqrt 2 * cos (θ - π / 4) →
  (∃ x y, x = ρ * cos θ ∧ y = ρ * sin θ ∧ (x - 1/2)^2 + (y - 1/2)^2 = 1/2) :=
by
  intro h
  use [ρ * cos θ, ρ * sin θ]
  split
  any_goals { sorry }

end polar_to_rectangular_l463_463573


namespace find_x_l463_463458

theorem find_x (x : ℝ) (h : x + 5 * 12 / (180 / 3) = 41) : x = 40 :=
sorry

end find_x_l463_463458


namespace probability_of_winning_position_l463_463533

-- Given Conditions
def tic_tac_toe_board : Type := Fin 3 × Fin 3
def is_nought (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := b p
def is_cross (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := ¬ b p

-- Winning positions in Tic-Tac-Toe are vertical, horizontal, or diagonal lines
def is_winning_position (b : tic_tac_toe_board → bool) : Prop :=
  (∃ i : Fin 3, ∀ j : Fin 3, is_nought b (i, j)) ∨ -- Row
  (∃ j : Fin 3, ∀ i : Fin 3, is_nought b (i, j)) ∨ -- Column
  (∀ i : Fin 3, is_nought b (i, i)) ∨ -- Main diagonal
  (∀ i : Fin 3, is_nought b (i, Fin.mk (2 - i.1) (by simp [i.1]))) -- Anti-diagonal

-- Problem Statement
theorem probability_of_winning_position :
  let total_positions := Nat.choose 9 3
  let winning_positions := 8
  winning_positions / total_positions = (2:ℚ) / 21 :=
by sorry

end probability_of_winning_position_l463_463533


namespace red_blue_area_ratio_is_12_l463_463805

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l463_463805


namespace blue_area_factor_12_l463_463825

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l463_463825


namespace percentage_of_B_students_is_20_l463_463692

-- Define the set of scores
def scores : List ℕ := [86, 73, 55, 98, 76, 93, 88, 72, 77, 62, 81, 79, 68, 82, 91]

-- Define the range for grade B
def isB (score : ℕ) : Prop := 87 ≤ score ∧ score ≤ 93

-- Count the number of students who received a B
def countB : ℕ := (scores.filter isB).length

-- Total number of students
def totalStudents : ℕ := scores.length

-- Calculate the percentage of students who received a B
noncomputable def percentageB : ℚ := (countB.to_rat / totalStudents.to_rat) * 100

-- Theorem to prove that the percentage of students who received a B is 20%
theorem percentage_of_B_students_is_20 : percentageB = 20 := by
  sorry

end percentage_of_B_students_is_20_l463_463692


namespace angle_PRS_is_90_degrees_l463_463137

noncomputable def Circle (center : Point) (radius : Real) : Set Point := sorry

theorem angle_PRS_is_90_degrees (P Q R S : Point) (PR : Real) (hPR : PR = 15) 
  (hCircle : Circle P 15) (hPQ_circle : Q ∈ hCircle) (hPR_circle : R ∈ hCircle) 
  (hAnglePQ_circle : S ∈ hCircle) (hScalene : ¬(IsIsosceles P Q R))
  (hRightAngleQ : angle P Q R = 90)
  (hOnExtendedPQ : collinear P Q S ∧ PQ < PS) :
  angle P R S = 90 := 
by sorry

end angle_PRS_is_90_degrees_l463_463137


namespace probability_of_winning_position_l463_463525

theorem probability_of_winning_position : 
    (let total_positions := Nat.choose 9 3,
         winning_positions := 8 in
       (winning_positions : ℚ) / total_positions = 2 / 21) := 
by
  sorry

end probability_of_winning_position_l463_463525


namespace greatest_divisor_of_976543_and_897623_l463_463065

theorem greatest_divisor_of_976543_and_897623 :
  Nat.gcd (976543 - 7) (897623 - 11) = 4 := by
  sorry

end greatest_divisor_of_976543_and_897623_l463_463065


namespace perpendicular_lines_b_value_l463_463935

theorem perpendicular_lines_b_value :
  (∀ b : ℝ, (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 → bx - 3 * y + 1 = 0 → (2/3) * (b/3) = -1) → b = -9/2) :=
begin
  intro b,
  rw eq_comm,
  have h1 : ∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 → 3 * y = 2 * x + 6 → y = (2/3) * x + 2 := sorry,
  have h2 : ∀ x y : ℝ, bx - 3 * y + 1 = 0 → 3 * y = b * x + 1 → y = (b/3) * x + 1/3 := sorry,
  sorry
end

end perpendicular_lines_b_value_l463_463935


namespace surface_area_of_eighth_block_l463_463965

-- Define the surface areas of the known blocks
def surface_area_blocks : List ℕ := [148, 46, 72, 28, 88, 126, 58]

-- Total surface area of all blocks
def total_surface_area : ℕ := List.sum surface_area_blocks

-- The problem statement
theorem surface_area_of_eighth_block :
  ∃ (eighth_block_area : ℕ), eighth_block_area = 22 ∧
    (total_surface_area + eighth_block_area) = 8 * (total_surface_area + eighth_block_area) / 8 :=
begin
  sorry
end

end surface_area_of_eighth_block_l463_463965


namespace max_sum_digits_l463_463675

theorem max_sum_digits (a b c : ℕ) (z : ℕ)
  (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) (h_digits : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum : float_eq_decimal_frac a b c z)
  (h_z : z ∈ {1, 2, 4, 5, 8, 10}) :
  ∃ a b c, a + b + c = 8 :=
sorry

def float_eq_decimal_frac (a b c z : ℕ) : Prop := 
  (100 * a + 10 * b + c) = 1000 / z

end max_sum_digits_l463_463675


namespace new_average_weight_l463_463013

-- noncomputable theory can be enabled if necessary for real number calculations.
-- noncomputable theory

def original_players : Nat := 7
def original_avg_weight : Real := 103
def new_players : Nat := 2
def weight_first_new_player : Real := 110
def weight_second_new_player : Real := 60

theorem new_average_weight :
  let original_total_weight : Real := original_players * original_avg_weight
  let total_weight : Real := original_total_weight + weight_first_new_player + weight_second_new_player
  let total_players : Nat := original_players + new_players
  total_weight / total_players = 99 := by
  sorry

end new_average_weight_l463_463013


namespace sqrt_x_plus_one_defined_l463_463034

theorem sqrt_x_plus_one_defined (x : ℝ) : (∃ y : ℝ, y = √(x + 1)) → x ≥ -1 :=
by
  intro h
  sorry

end sqrt_x_plus_one_defined_l463_463034


namespace triangle_area_relation_l463_463354

noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_relation : area 20 21 29 = (5 / 2) * area 13 14 15 :=
by
  sorry

end triangle_area_relation_l463_463354


namespace pairwise_product_sum_leq_quarter_l463_463219

theorem pairwise_product_sum_leq_quarter {n : ℕ} (h : n ≥ 4) (a : Fin n → ℝ) 
  (nonneg : ∀ i, 0 ≤ a i) (sum_eq_one : (Finset.univ.sum a) = 1) :
  (Finset.univ.sum (λ i, a i * a ((i + 1) % n))) ≤ 1 / 4 := 
by 
  sorry

end pairwise_product_sum_leq_quarter_l463_463219


namespace problem_solution_l463_463171

theorem problem_solution {n : ℕ} :
  (∀ x y z : ℤ, x + y + z = 0 → ∃ k : ℤ, (x^n + y^n + z^n) / 2 = k^2) ↔ n = 1 ∨ n = 4 :=
by
  sorry

end problem_solution_l463_463171


namespace find_a_l463_463361

theorem find_a (a : ℝ) (h₁ : a ≠ 0) (h₂ : ∀ x y : ℝ, x^2 + a*y^2 + a^2 = 0) (h₃ : 4 = 4) :
  a = (1 - Real.sqrt 17) / 2 := sorry

end find_a_l463_463361


namespace distance_between_foci_of_ellipse_l463_463179

-- Define the parameters a^2 and b^2 according to the problem
def a_sq : ℝ := 25
def b_sq : ℝ := 16

-- State the problem
theorem distance_between_foci_of_ellipse : 
  (2 * Real.sqrt (a_sq - b_sq)) = 6 := by
  -- Proof content is skipped 
  sorry

end distance_between_foci_of_ellipse_l463_463179


namespace circumscribed_circle_radius_iso_trap_l463_463876

theorem circumscribed_circle_radius_iso_trap (a b c d : ℝ) 
  (h1 : a = 1) (h2 : b = 1) (h3 : c = 1) (h4 : d = sqrt 2) : 
  ∃ r, r = sqrt ((sqrt 2 + 3) / 7) :=
by
  sorry

end circumscribed_circle_radius_iso_trap_l463_463876


namespace problem_solution_l463_463975

noncomputable def equilateral_triangle_area_to_perimeter_square_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * Real.sqrt 3 / 2
  let area := 1 / 2 * s * altitude
  let perimeter := 3 * s
  let perimeter_squared := perimeter^2
  area / perimeter_squared

theorem problem_solution :
  equilateral_triangle_area_to_perimeter_square_ratio 10 rfl = Real.sqrt 3 / 36 :=
sorry

end problem_solution_l463_463975


namespace distinct_colorings_of_cube_l463_463888

def g (m : ℕ) : ℚ :=
  (m^6 + 3 * m^4 + 12 * m^3 + 8 * m^2) / 24

theorem distinct_colorings_of_cube (m : ℕ) : g(m) = (m^6 + 3 * m^4 + 12 * m^3 + 8 * m^2) / 24 :=
by sorry

end distinct_colorings_of_cube_l463_463888


namespace geometric_sequence_general_term_sum_of_first_n_terms_l463_463227

noncomputable def a_n (n : ℕ) : ℕ := 2^(2*n - 3)

def b_n (n : ℕ) : ℤ := Int.log2 (a_n n)

def S_n (n : ℕ) : ℤ := (Range n).Sum (λ i => b_n (i + 1))

theorem geometric_sequence_general_term :
  (∀ n, a_n n = 2 ^ (2 * n - 3)) :=
sorry

theorem sum_of_first_n_terms (n : ℕ) (h : S_n n = 360) : n = 20 :=
sorry

end geometric_sequence_general_term_sum_of_first_n_terms_l463_463227


namespace problem_solution_l463_463930

open Finset

def S : Finset ℝ := {-10, -6, -5, -4, -2.5, -1, 0, 2.5, 4, 6, 7, 10}

def is_solution (x : ℝ) : Prop := (x - 5 = 0) ∨ (x + 10 = 0) ∨ (2 * x - 5 = 0)

def probability_solution_in_set (S : Finset ℝ) : ℚ :=
  (S.filter is_solution).card / S.card

theorem problem_solution :
  probability_solution_in_set S = 1 / 6 :=
by
  sorry

end problem_solution_l463_463930


namespace inclination_angle_70_l463_463861

noncomputable def inclination_angle (x y : ℝ) (t : ℝ) (sin20 cos20: ℝ) : ℝ :=
let x_eq := 3 + t * sin20 in
let y_eq := -1 + t * cos20 in
if x_eq = x ∧ y_eq = y then 70 else 0

theorem inclination_angle_70 (t sin20 cos20 : ℝ) (ht : cos20 = real.cos (real.pi / 9)) (hs : sin20 = real.sin (real.pi / 9)) :
  inclination_angle (3 + t * sin20) (-1 + t * cos20) t sin20 cos20 = 70 :=
by
  sorry

end inclination_angle_70_l463_463861


namespace diff_in_areas_diff_in_perimeters_l463_463696

-- Define the side length of the square
def side_length : ℝ := 300

-- Define π (pi)
def pi_approx : ℝ := 3.14

-- Define the areas
def area_square : ℝ := side_length ^ 2
def area_quarter_circle : ℝ := (1 / 4) * pi_approx * side_length ^ 2
def area_semicircle : ℝ := (1 / 2) * pi_approx * (side_length / 2) ^ 2

-- Define the total quarter circles' area
def total_area_quarters : ℝ := 2 * area_quarter_circle

-- Define the area difference
def area_difference : ℝ := total_area_quarters - area_square - area_semicircle

-- Define the perimeters
def perimeter_quarter_circle : ℝ := (1 / 2) * pi_approx * side_length
def perimeter_semicircle : ℝ := pi_approx * (side_length / 2)

-- Define the perimeter difference
def perimeter_difference : ℝ := (2 * perimeter_quarter_circle) - perimeter_semicircle

-- The lean 4 statement proofs for the given equivalence
theorem diff_in_areas : area_difference = 15975 := by
  sorry

theorem diff_in_perimeters : perimeter_difference = 485 := by
  sorry

end diff_in_areas_diff_in_perimeters_l463_463696


namespace max_value_f_l463_463644

def f (x : ℝ) : ℝ := sin x + (sqrt 3) * cos x 

theorem max_value_f : ∀ x : ℝ, f x ≤ 2 ∧ (∃ x : ℝ, f x = 2) :=
  by
  sorry

end max_value_f_l463_463644


namespace perimeter_of_WXY_is_correct_l463_463103

noncomputable def perimeter_of_triangle_WXY : ℝ :=
  let h := 20
  let side_length := 15
  let mid_segment := side_length / 2
  let WT := Real.sqrt (mid_segment ^ 2 + h ^ 2)
  let XT := WT
  let WX := mid_segment
  WX + WT + XT

theorem perimeter_of_WXY_is_correct (P Q R S T U W X Y : ℝ) 
  (prism_height : ℝ) (base_side_length : ℝ) (W_midpoint : W = P/2)
  (X_midpoint : X = Q/2) (Y_midpoint : Y = R/2) 
  (equi_triangle_base : (P = Q) ∧ (Q = R) ∧ (R = P)) 
  (height_condition : prism_height = 20) 
  (side_length_condition : base_side_length = 15) : 
  perimeter_of_triangle_WXY = 50.25 := by
s

end perimeter_of_WXY_is_correct_l463_463103


namespace units_digit_17_pow_28_l463_463912

theorem units_digit_17_pow_28 : (17 ^ 28) % 10 = 1 :=
by
  sorry

end units_digit_17_pow_28_l463_463912


namespace blue_to_red_face_area_ratio_l463_463857

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l463_463857


namespace max_xy_l463_463298

theorem max_xy (x y : ℕ) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 :=
sorry

end max_xy_l463_463298


namespace remainder_of_2519_div_8_l463_463089

theorem remainder_of_2519_div_8 : 2519 % 8 = 7 := 
by 
  sorry

end remainder_of_2519_div_8_l463_463089


namespace count_valid_numbers_l463_463283

def is_valid_number (n : ℕ) : Prop := n % 5 = 0 ∧ n % 2 = 0 ∧ n < 3000 ∧ (∃ k : ℕ, n = k * k)

theorem count_valid_numbers : {n : ℕ | is_valid_number n}.toFinset.card = 5 :=
by
  sorry

end count_valid_numbers_l463_463283


namespace eval_expression_l463_463156

theorem eval_expression : 
  let a := 2999 in
  let b := 3000 in
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 := 
by 
  sorry

end eval_expression_l463_463156


namespace blue_red_face_area_ratio_l463_463816

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l463_463816


namespace problem_l463_463578

theorem problem : 
  (∑ k in finset.range 2020, (2021 - (k + 1)) / (k + 1)) / (∑ k in finset.range 2020, 1 / (k + 2)) = 2021 :=
by
  sorry

end problem_l463_463578


namespace lattice_points_in_T_l463_463362

-- Definitions based on the conditions
def T (n : ℝ) : set (ℝ × ℝ) :=
  { p | 0 < p.1 ∧ 0 < p.2 ∧ p.1 * p.2 ≤ n }

noncomputable def count_lattice_points_T (n : ℕ) : ℕ :=
  finset.card { p : finset (ℕ × ℕ) | T n p }

-- Defining the sum and the floor function
open_locale big_operators
open finset

noncomputable def sum_floor (n : ℕ) : ℕ :=
  ∑ x in range (nat.floor (real.sqrt n) + 1), nat.floor (n / x)

-- The statement to be proved
theorem lattice_points_in_T (n : ℕ) (h : n > 0) :
  count_lattice_points_T n = 2 * sum_floor n - (nat.floor (real.sqrt n)) ^ 2 := sorry

end lattice_points_in_T_l463_463362


namespace total_trees_after_planting_l463_463695

-- Define the initial counts of the trees
def initial_maple_trees : ℕ := 2
def initial_poplar_trees : ℕ := 5
def initial_oak_trees : ℕ := 4

-- Define the planting rules
def maple_trees_planted (initial_maple : ℕ) : ℕ := 3 * initial_maple
def poplar_trees_planted (initial_poplar : ℕ) : ℕ := 3 * initial_poplar

-- Calculate the total number of each type of tree after planting
def total_maple_trees (initial_maple : ℕ) : ℕ :=
  initial_maple + maple_trees_planted initial_maple

def total_poplar_trees (initial_poplar : ℕ) : ℕ :=
  initial_poplar + poplar_trees_planted initial_poplar

def total_oak_trees (initial_oak : ℕ) : ℕ := initial_oak

-- Calculate the total number of trees in the park
def total_trees (initial_maple initial_poplar initial_oak : ℕ) : ℕ :=
  total_maple_trees initial_maple + total_poplar_trees initial_poplar + total_oak_trees initial_oak

-- The proof statement
theorem total_trees_after_planting :
  total_trees initial_maple_trees initial_poplar_trees initial_oak_trees = 32 := 
by
  -- Proof placeholder
  sorry

end total_trees_after_planting_l463_463695


namespace num_terminating_decimals_l463_463215

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 518) :
  (∃ k, (1 ≤ k ∧ k ≤ 518) ∧ n = k * 21) ↔ n = 24 :=
sorry

end num_terminating_decimals_l463_463215


namespace arithmetic_sequence_product_l463_463357

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_a4a5 : a 3 * a 4 = 24) :
  a 2 * a 5 = 16 :=
sorry

end arithmetic_sequence_product_l463_463357


namespace find_speed_of_current_l463_463505

variable {m c : ℝ}

theorem find_speed_of_current
  (h1 : m + c = 15)
  (h2 : m - c = 10) :
  c = 2.5 :=
sorry

end find_speed_of_current_l463_463505


namespace swimmers_meetings_in_15_minutes_l463_463887

noncomputable def swimmers_pass_each_other_count 
    (pool_length : ℕ) (rate_swimmer1 : ℕ) (rate_swimmer2 : ℕ) (time_minutes : ℕ) : ℕ :=
sorry -- Definition of the function to count passing times

theorem swimmers_meetings_in_15_minutes :
  swimmers_pass_each_other_count 120 4 3 15 = 23 :=
sorry -- The proof is not required as per instruction.

end swimmers_meetings_in_15_minutes_l463_463887


namespace evaluate_expression_l463_463161

theorem evaluate_expression : 
  (3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001) :=
by
  sorry

end evaluate_expression_l463_463161


namespace largest_divisor_39_l463_463928

theorem largest_divisor_39 (m : ℕ) (hm : 0 < m) (h : 39 ∣ m ^ 2) : 39 ∣ m :=
by sorry

end largest_divisor_39_l463_463928


namespace vertex_y_coordinate_l463_463142

theorem vertex_y_coordinate : 
  ∀ (y : ℝ → ℝ), (y = λ x, -2 * x^2 + 16 * x + 72) → (∃ n : ℝ, n = 104) :=
by
  intros y h_eq
  use 104
  sorry

end vertex_y_coordinate_l463_463142


namespace probability_exactly_four_of_eight_show_2_l463_463155

noncomputable def dice_probability : ℝ :=
  let n : ℕ := 8
  let k : ℕ := 4
  let p_show_2 : ℝ := 1 / 8
  let p_not_show_2 : ℝ := 7 / 8
  let ways := Nat.choose n k
  (ways * (p_show_2 ^ k) * (p_not_show_2 ^ (n - k)))

theorem probability_exactly_four_of_eight_show_2 : 
  (Real.toRoundedString (dice_probability) 3) = "0.010" :=
by
  sorry

end probability_exactly_four_of_eight_show_2_l463_463155


namespace total_oranges_albert_l463_463967

theorem total_oranges_albert : 
  ∀ (boxes : ℕ) (oranges_per_box : ℕ), boxes = 7 → oranges_per_box = 5 → (boxes * oranges_per_box) = 35 :=
by
  intros boxes oranges_per_box h1 h2
  rw [h1, h2]
  norm_num

end total_oranges_albert_l463_463967


namespace price_reduction_2100_yuan_l463_463082

-- Definitions based on conditions
def initial_sales : ℕ := 30
def initial_profit_per_item : ℕ := 50
def additional_sales_per_yuan (x : ℕ) : ℕ := 2 * x
def new_profit_per_item (x : ℕ) : ℕ := 50 - x
def target_profit : ℕ := 2100

-- Final proof statement, showing the price reduction needed
theorem price_reduction_2100_yuan (x : ℕ) 
  (h : (50 - x) * (30 + 2 * x) = 2100) : 
  x = 20 := 
by 
  sorry

end price_reduction_2100_yuan_l463_463082


namespace num_hens_in_caravan_l463_463317

variable (H G C K : ℕ)  -- number of hens, goats, camels, keepers
variable (total_heads total_feet : ℕ)

-- Defining the conditions
def num_goats := 35
def num_camels := 6
def num_keepers := 10
def heads := H + G + C + K
def feet := 2 * H + 4 * G + 4 * C + 2 * K
def relation := feet = heads + 193

theorem num_hens_in_caravan :
  G = num_goats → C = num_camels → K = num_keepers → relation → 
  H = 60 :=
by 
  intros _ _ _ _
  sorry

end num_hens_in_caravan_l463_463317


namespace find_x_l463_463033

variables (x : ℝ)
axiom h1 : (180 / x) + (5 * 12 / x) + 80 = 81

theorem find_x : x = 240 :=
by {
  sorry
}

end find_x_l463_463033


namespace probability_of_winning_noughts_l463_463519

def ticTacToeProbability : ℚ :=
  let totalWays := Nat.choose 9 3
  let winningWays := 8
  winningWays / totalWays

theorem probability_of_winning_noughts :
  ticTacToeProbability = 2 / 21 := by
  sorry

end probability_of_winning_noughts_l463_463519


namespace largest_angle_of_convex_hexagon_l463_463086

noncomputable def hexagon_largest_angle (x : ℚ) : ℚ :=
  max (6 * x - 3) (max (5 * x + 1) (max (4 * x - 4) (max (3 * x) (max (2 * x + 2) x))))

theorem largest_angle_of_convex_hexagon (x : ℚ) (h : x + (2*x+2) + 3*x + (4*x-4) + (5*x+1) + (6*x-3) = 720) : 
  hexagon_largest_angle x = 4281 / 21 := 
sorry

end largest_angle_of_convex_hexagon_l463_463086


namespace max_sum_digits_l463_463676

theorem max_sum_digits (a b c : ℕ) (z : ℕ)
  (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) (h_digits : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum : float_eq_decimal_frac a b c z)
  (h_z : z ∈ {1, 2, 4, 5, 8, 10}) :
  ∃ a b c, a + b + c = 8 :=
sorry

def float_eq_decimal_frac (a b c z : ℕ) : Prop := 
  (100 * a + 10 * b + c) = 1000 / z

end max_sum_digits_l463_463676


namespace speed_of_current_l463_463503

variable (m c : ℝ)

theorem speed_of_current (h1 : m + c = 15) (h2 : m - c = 10) : c = 2.5 :=
sorry

end speed_of_current_l463_463503


namespace intersection_is_empty_l463_463460

-- Define sets M and N
def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {0, 3, 4}

-- Define isolated elements for a set
def is_isolated (A : Set ℕ) (x : ℕ) : Prop :=
  x ∈ A ∧ (x - 1 ∉ A) ∧ (x + 1 ∉ A)

-- Define isolated sets
def isolated_set (A : Set ℕ) : Set ℕ :=
  {x | is_isolated A x}

-- Define isolated sets for M and N
def M' := isolated_set M
def N' := isolated_set N

-- The intersection of the isolated sets
theorem intersection_is_empty : M' ∩ N' = ∅ := 
  sorry

end intersection_is_empty_l463_463460


namespace rectangle_ratio_l463_463325

theorem rectangle_ratio (s E F : ℝ) (h1 : s = 4) (h2 : E = s / 3) (h3 : F = 2 * s / 3) (G BF : ℝ)
  (hAG_right_angle: ∀ (A : ℝ), ∃ G, AG * BF = s^2) :
  let XY := BF in
  let YZ := s^2 / XY in
  XY / YZ = 13 / 9 :=
by
  sorry

end rectangle_ratio_l463_463325


namespace units_digit_17_pow_28_l463_463911

theorem units_digit_17_pow_28 : (17 ^ 28) % 10 = 1 :=
by
  sorry

end units_digit_17_pow_28_l463_463911


namespace weed_spread_limitation_l463_463515

-- Defining the conditions of the problem
variables {Field : Type} [fintype Field] (plots : Field → bool)

-- Condition 1: 100 identical square plots
def square_field (n : ℕ) := n = 100

-- Condition 2: Initially, exactly 9 plots overgrown with weeds
def initial_weeded_num (plots : Field → bool) : ℕ := (finset.filter plots finset.univ).card
def initial_condition (plots : Field → bool) : Prop := initial_weeded_num plots = 9

-- Condition 3: Weeds can only spread to a plot if it has at least two neighboring plots overgrown with weeds
def neighbors (field : Field) : set Field := 
  -- define the adjacency or neighborhood relationship
  sorry 

def spread_rule (plots : Field → bool) (p : Field) : Prop := 
  (finset.filter (λ n, plots n) (neighbors p)).card ≥ 2

-- Proposition: The entire field will never be completely overgrown with weeds given the conditions
theorem weed_spread_limitation 
  (plots : Field → bool) 
  (h1 : square_field 100) 
  (h2 : initial_condition plots)
  (h3 : ∀ p : Field, spread_rule plots p) :
  ∃ p : Field, ¬plots p := 
sorry

end weed_spread_limitation_l463_463515


namespace average_weight_a_b_l463_463406

variables (A B C : ℝ)

theorem average_weight_a_b (h1 : (A + B + C) / 3 = 43)
                          (h2 : (B + C) / 2 = 43)
                          (h3 : B = 37) :
                          (A + B) / 2 = 40 :=
by
  sorry

end average_weight_a_b_l463_463406


namespace conic_not_parabola_l463_463410

def conic_equation (m x y : ℝ) : Prop :=
  m * x^2 + (m + 1) * y^2 = m * (m + 1)

theorem conic_not_parabola (m : ℝ) :
  ¬ (∃ (x y : ℝ), conic_equation m x y ∧ ∃ (a b c d e f : ℝ), m * x^2 + (m + 1) * y^2 = a * x^2 + b * xy + c * y^2 + d * x + e * y + f ∧ (a = 0 ∨ c = 0) ∧ (b ≠ 0 ∨ a ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0)) :=  
sorry

end conic_not_parabola_l463_463410


namespace sum_positive_integer_values_c_l463_463201

theorem sum_positive_integer_values_c :
  (∑ c in (finset.filter (λ c : ℤ, is_integer c ∧ c > 0)
    (finset.image (λ k : ℤ, (49 - k^2) / 12)
      (finset.range 7))), c) = 6 :=
sorry

end sum_positive_integer_values_c_l463_463201


namespace blue_red_area_ratio_l463_463840

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l463_463840


namespace max_intersections_l463_463897

theorem max_intersections (C₁ C₂ C₃ : Circle) (L : Line) : 
  greatest_points_of_intersection 3 1 = 12 :=
sorry

end max_intersections_l463_463897


namespace total_height_of_siblings_l463_463585

theorem total_height_of_siblings 
    (h1 : ℕ) (h2 : ℕ) (sibling1 : ℕ) (sibling2 : ℕ) (eliza : ℕ) (last_sibling : ℕ)
    (c1 : h1 = 66) (c2 : h2 = 66) 
    (c3 : sibling1 = 60) 
    (c4 : eliza = 68)
    (c5 : last_sibling = eliza + 2) :
  h1 + h2 + sibling1 + eliza + last_sibling = 330 :=
by
  have h_sum : h1 + h2 = 2 * 66, from sorry,
  have last_sibling_val : last_sibling = 70, from sorry,
  calc
    h1 + h2 + sibling1 + eliza + last_sibling 
      = 132 + 60 + 68 + 70 : by sorry
    ... = 330 : by sorry

end total_height_of_siblings_l463_463585


namespace projection_formula_l463_463623

variable {V : Type _} [InnerProductSpace ℝ V]

def projection (a b : V) : V := (inner a b / inner b b) • b

theorem projection_formula (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  projection a b = (inner a b / ‖b‖) • (b / ‖b‖) :=
by
  sorry

end projection_formula_l463_463623


namespace tenth_term_geometric_sequence_l463_463138

def a := 5
def r := Rat.ofInt 3 / 4
def n := 10

theorem tenth_term_geometric_sequence :
  a * r^(n-1) = Rat.ofInt 98415 / Rat.ofInt 262144 := sorry

end tenth_term_geometric_sequence_l463_463138


namespace total_cost_difference_l463_463400

def cost_of_rooster_stamps (mr_vance_qty mr_vance_price mrs_vance_qty mrs_vance_price john_qty john_price : ℝ) : ℝ :=
  (mr_vance_qty * mr_vance_price) + (mrs_vance_qty * mrs_vance_price) + (john_qty * john_price)

def cost_of_daffodil_stamps (mr_vance_qty mr_vance_price mrs_vance_qty mrs_vance_price john_qty john_price : ℝ) : ℝ :=
  (mr_vance_qty * mr_vance_price) + (mrs_vance_qty * mrs_vance_price) + (john_qty * john_price)

theorem total_cost_difference :
  let rooster_total := cost_of_rooster_stamps 3 1.50 2 1.25 4 1.40 in
  let daffodil_total := cost_of_daffodil_stamps 5 0.75 7 0.80 3 0.70 in
  rooster_total - daffodil_total = 1.15 :=
by
  sorry

end total_cost_difference_l463_463400


namespace more_likely_even_number_l463_463700

-- Definition statements for conditions
def balls : Finset ℕ := Finset.range 11 \ {0}
def evenNumbers : Finset ℕ := {2, 4, 6, 8, 10}
def numbersNotLessThan7 : Finset ℕ := {7, 8, 9, 10}

-- Proof statement for solution
theorem more_likely_even_number :
  (evenNumbers.card / balls.card.to_rat) > 
  (numbersNotLessThan7.card / balls.card.to_rat) :=
by
  sorry

end more_likely_even_number_l463_463700


namespace sum_of_valid_c_l463_463204

theorem sum_of_valid_c : 
  let discriminant (c : ℕ) := 49 - 12 * c in
  (∀ (c : ℕ), (3 * x^2 + 7 * x + c = 0) → (∃ k : ℕ, discriminant c = k^2)) →
  (∑ c in (finset.filter (λ c, (∃ k : ℕ, discriminant c = k^2) ∧ c > 0 ∧ c < 5) (finset.range 5)), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463204


namespace souvenir_purchasing_plans_l463_463397

-- Define the conditions
def types := 4
def total_pieces := 25
def pieces_per_type := 10
def at_least_one_of_each := 1

-- The main statement
theorem souvenir_purchasing_plans : 
  ∃ n : ℕ, n = 592 ∧ 
  ∑ i in finset.range(types), 1 ≤ total_pieces ∧ 
  total_pieces ≤ types * pieces_per_type :=
sorry

end souvenir_purchasing_plans_l463_463397


namespace sufficient_but_not_necessary_condition_l463_463617

variable (a₁ d : ℝ)

def S₄ := 4 * a₁ + 6 * d
def S₅ := 5 * a₁ + 10 * d
def S₆ := 6 * a₁ + 15 * d

theorem sufficient_but_not_necessary_condition (h : d > 1) :
  S₄ a₁ d + S₆ a₁ d > 2 * S₅ a₁ d :=
by
  -- proof omitted
  sorry

end sufficient_but_not_necessary_condition_l463_463617


namespace simplify_fraction_l463_463478

theorem simplify_fraction (x y : ℕ) : (x + y)^3 / (x + y) = (x + y)^2 := by
  sorry

end simplify_fraction_l463_463478


namespace red_blue_area_ratio_is_12_l463_463810

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l463_463810


namespace candidate_a_valid_votes_l463_463470

/-- In an election, candidate A got 80% of the total valid votes.
If 15% of the total votes were declared invalid and the total number of votes is 560,000,
find the number of valid votes polled in favor of candidate A. -/
theorem candidate_a_valid_votes :
  let total_votes := 560000
  let invalid_percentage := 0.15
  let valid_percentage := 0.85
  let candidate_a_percentage := 0.80
  let valid_votes := (valid_percentage * total_votes : ℝ)
  let candidate_a_votes := (candidate_a_percentage * valid_votes : ℝ)
  candidate_a_votes = 380800 :=
by
  sorry

end candidate_a_valid_votes_l463_463470


namespace trains_crossing_time_l463_463066

theorem trains_crossing_time (length : ℕ) (time1 time2 : ℕ) (h1 : length = 120) (h2 : time1 = 10) (h3 : time2 = 20) :
  (2 * length : ℚ) / (length / time1 + length / time2 : ℚ) = 13.33 :=
by
  sorry

end trains_crossing_time_l463_463066


namespace zahar_sorting_terminates_finitely_l463_463433

theorem zahar_sorting_terminates_finitely (n : ℕ) (h : n ≥ 1) :
  ∃ m, fin_seq_length n ≤ m :=
begin
  sorry
end

end zahar_sorting_terminates_finitely_l463_463433


namespace Little_Twelve_Basketball_Conference_games_l463_463398

theorem Little_Twelve_Basketball_Conference_games :
  ∀ (div1 div2 : Finset ℕ),
    div1.card = 6 →
    div2.card = 6 →
    (∀ t ∈ div1, ∀ t' ∈ div1, t ≠ t' → 2 * (Finset.card div1 - 1))
    + (∀ t ∈ div1, ∀ t' ∈ div2, t ≠ t' → 2 * Finset.card div2)
    + (∀ t ∈ div2, ∀ t' ∈ div1, t ≠ t' → 2 * Finset.card div1)
    + (∀ t ∈ div2, ∀ t' ∈ div2, t ≠ t' → 2 * (Finset.card div2 - 1)) = 132 :=
by
  sorry

end Little_Twelve_Basketball_Conference_games_l463_463398


namespace find_f_l463_463359

noncomputable def f (a : ℚ⁺) : ℤ := sorry -- We assume this function definition

namespace Solution

variables (f : ℚ⁺ → ℤ)

-- Given conditions
axiom f_condition1 {a b : ℚ⁺} : f (a * b) = f a + f b
axiom f_condition2 : f 1999 = 1
axiom f_condition3 {a b : ℚ⁺} : f (a + b) ≥ min (f a) (f b)

-- Goal to prove
theorem find_f : ∀ a : ℚ⁺, f a = int.log 1999 a := sorry

end Solution

end find_f_l463_463359


namespace reflect_over_x_axis_reflect_over_y_axis_l463_463615

-- Mathematical Definitions
def Point := (ℝ × ℝ)

-- Reflect a point over the x-axis
def reflectOverX (M : Point) : Point :=
  (M.1, -M.2)

-- Reflect a point over the y-axis
def reflectOverY (M : Point) : Point :=
  (-M.1, M.2)

-- Theorem statements
theorem reflect_over_x_axis (M : Point) : reflectOverX M = (M.1, -M.2) :=
by
  sorry

theorem reflect_over_y_axis (M : Point) : reflectOverY M = (-M.1, M.2) :=
by
  sorry

end reflect_over_x_axis_reflect_over_y_axis_l463_463615


namespace inequality_cannot_hold_l463_463668

theorem inequality_cannot_hold (a b : ℝ) (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) :=
by {
  sorry
}

end inequality_cannot_hold_l463_463668


namespace count_sets_sum_to_18_including_6_l463_463121

-- Definitions
def number_set : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Problem Statement
theorem count_sets_sum_to_18_including_6 : 
  (∃ s : set ℕ, s ⊆ number_set ∧ s.card = 3 ∧ 6 ∈ s ∧ s.sum = 18) 
  ↔ 3 :=
sorry

end count_sets_sum_to_18_including_6_l463_463121


namespace red_blue_area_ratio_is_12_l463_463809

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l463_463809


namespace sqrt_2_is_same_type_quadratic_surd_as_sqrt_8_l463_463667

theorem sqrt_2_is_same_type_quadratic_surd_as_sqrt_8 :
  ∀ (n : ℕ), n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 → (n = 8 ↔ (∃ a : ℤ, sqrt n = a * sqrt 2)) :=
by {
  intro n,
  intro h,
  split,
  {
    intro hn,
    rw hn,
    use 2,
    norm_num,
  },
  {
    intro hsq,
    cases hsq with a ha,
    have h2 : (n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12),
    exact h,
    cases h2,
      {exfalso,
      simp at ha, linarith,},
      {exfalso,
      simp at ha, linarith,},
      {exact h2,},
      {exfalso,
      simp at ha, linarith,},
  },
  sorry,
}

end sqrt_2_is_same_type_quadratic_surd_as_sqrt_8_l463_463667


namespace slope_probability_correct_l463_463037

noncomputable def slope_probability : ℚ := 
  let total_cases := 36
  let favorable_cases := 9
  favorable_cases / total_cases

theorem slope_probability_correct (a b : ℕ) (ha : a ∈ {1, 2, 3, 4, 5, 6}) (hb : b ∈ {1, 2, 3, 4, 5, 6}) : 
  slope_probability = 1 / 4 := by 
  sorry

end slope_probability_correct_l463_463037


namespace water_percentage_in_honey_l463_463916

noncomputable def nectar_mass : ℝ := 1.5
def honey_mass : ℝ := 1
def water_content_in_nectar : ℝ := 0.50

theorem water_percentage_in_honey :
  let water_in_nectar := water_content_in_nectar * nectar_mass in
  let solid_in_nectar := nectar_mass - water_in_nectar in
  let water_in_honey := honey_mass - solid_in_nectar in
  (water_in_honey / honey_mass) * 100 = 25 :=
by
  sorry

end water_percentage_in_honey_l463_463916


namespace area_triangle_XYZ_l463_463336

-- Definitions of points and their properties:
variables {X Y Z D E R S G : Type}

-- Conditions about the triangle and points
variables (triangle_XYZ : true) -- Symbolic representation of triangle XYZ
variables (midpt_D : true) -- D is the midpoint of YZ
variables (midpt_E : true) -- E is the midpoint of XZ
variables (midpt_R : true) -- R is the midpoint of XY
variables (centroid_G : true) -- G is the centroid of XYZ
variables (intersection_S : true) -- S is the intersection of DR and YE
variables (area_GDS : ℝ) -- The area of triangle GDS is k

-- Statement to prove
theorem area_triangle_XYZ (k : ℝ) (h_GDS : area_GDS = k) : 
  area (triangle_XYZ) = 24 * k := sorry

end area_triangle_XYZ_l463_463336


namespace probability_of_winning_noughts_l463_463520

def ticTacToeProbability : ℚ :=
  let totalWays := Nat.choose 9 3
  let winningWays := 8
  winningWays / totalWays

theorem probability_of_winning_noughts :
  ticTacToeProbability = 2 / 21 := by
  sorry

end probability_of_winning_noughts_l463_463520


namespace average_percent_increase_per_year_l463_463002

def initial_population : ℕ := 175000
def final_population : ℕ := 262500
def years : ℕ := 10

theorem average_percent_increase_per_year :
  ( ( ( ( final_population - initial_population ) / years : ℝ ) / initial_population ) * 100 ) = 5 := by
  sorry

end average_percent_increase_per_year_l463_463002


namespace identify_false_proposition_l463_463120

def propA (a b : ℝ) : Prop := (b = 0) → (a = 0)
def propB (a b c : Prop) : Prop := (a → b) → (b → c) → (a → c)
def propC : Prop := ∀ (l1 l2 : Prop), (l1 → l2) → ¬(l1 ∧ l2)
def propD (α β : Type) [LinearOrder α] [LinearOrder β] : Prop := ∀ (a b : α), (a = b) → ¬ (a ≠ b)

theorem identify_false_proposition : ¬ propD :=
by
  sorry

end identify_false_proposition_l463_463120


namespace tic_tac_toe_winning_probability_l463_463526

theorem tic_tac_toe_winning_probability :
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  probability = (2 : ℚ) / 21 :=
by
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  have probability_correct : probability = (2 : ℚ) / 21 := sorry
  exact probability_correct

end tic_tac_toe_winning_probability_l463_463526


namespace train_passes_jogger_in_given_time_l463_463952

-- Conditions modeled as definitions
def jogger_speed_kmh : ℝ := 10
def train_speed_kmh : ℝ := 46
def initial_distance_m : ℝ := 340
def train_length_m : ℝ := 120

-- Convert from km/hr to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := (speed_kmh * 1000) / 3600

-- Define speeds in m/s
def jogger_speed_ms : ℝ := kmh_to_ms jogger_speed_kmh
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

-- Relative speed of train with respect to jogger
def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms

-- Total distance to be covered by the train to pass the jogger
def total_distance_m : ℝ := initial_distance_m + train_length_m

-- Time taken for the train to pass the jogger
def time_to_pass_jogger : ℝ := total_distance_m / relative_speed_ms

-- Lean statement for the proof problem
theorem train_passes_jogger_in_given_time : time_to_pass_jogger = 46 := by
  sorry

end train_passes_jogger_in_given_time_l463_463952


namespace number_of_valid_menus_l463_463100

def desserts := { "cake", "pie", "ice cream", "pudding", "jelly" }

def no_repeat_consecutive (menu : List String) : Prop := 
  ∀ i, i < menu.length - 1 → menu[i] ≠ menu[i + 1]

def pie_on_wednesday (menu : List String) : Prop := 
  2 < menu.length → menu[2] = "pie"

def ice_cream_on_monday (menu : List String) : Prop := 
  0 < menu.length → menu.head = "ice cream"

def valid_menu (menu : List String) : Prop := 
  no_repeat_consecutive menu ∧ pie_on_wednesday menu ∧ ice_cream_on_monday menu

theorem number_of_valid_menus : 
  (∃ menu : List String, length menu = 7 ∧ valid_menu menu) = 768 := 
sorry

end number_of_valid_menus_l463_463100


namespace product_digits_sum_l463_463869

theorem product_digits_sum (n : ℕ) : 
  (9 * (10^n - 1) / 9).digits.sum = 1080 ↔ n = 120 :=
by sorry

end product_digits_sum_l463_463869


namespace ln_binom_le_sum_floor_log_exists_c_for_pi_l463_463475

open BigOperators

-- Theorem 1: Show that
-- \ln \left(\binom{2 n}{n}\right) \leq 
-- \sum_{\substack{p \text{ prime} \\ p \leq 2 n}}\left\lfloor\frac{\ln (2 n)}{\ln p}\right\rfloor \ln p
theorem ln_binom_le_sum_floor_log (n : ℕ) :
  Real.log (Nat.choose (2 * n) n) ≤ ∑ p in Finset.filter Nat.prime (Finset.range (2 * n + 1)), 
    (Real.log (2 * n) / Real.log p).floor * Real.log p :=
by
  sorry

-- Theorem 2: Using the first theorem,
-- show that there exists a constant c > 0 such that for any real x,
-- \pi(x) \geq c \frac{x}{\ln x}
theorem exists_c_for_pi (c : ℝ) (hc : 0 < c) : ∃ c > 0, ∀ (x : ℝ), 
  0 < x → Nat.PrimeCounting.pi x ≥ c * x / Real.log x :=
by
  use c
  split
  · sorry
  · intro x hx
    rw [Nat.PrimeCounting.pi_le_iff]
    apply le_of_eq
    sorry

end ln_binom_le_sum_floor_log_exists_c_for_pi_l463_463475


namespace concurrency_of_perpendiculars_l463_463689

universe u

-- Define the necessary concepts
variables {A B C D I I_a A' B' C' : Type u} [triangle ABC A B C]

-- Assume the required conditions
structure triangle_configuration (ABC I I_a A' B' C' : Type u) :=
(D : Type u) -- D is the intersection of the external bisector of angle A and line BC
(I : Type u) -- I is the incenter of triangle ABC
(I_a : Type u) -- I_a is the excenter opposite angle A
(A' : Type u) -- A' is the intersection of the perpendicular from I to DI_a and the circumcircle
(B' : Type u) -- Similarly defined point B'
(C' : Type u) -- Similarly defined point C'

theorem concurrency_of_perpendiculars 
  (config : triangle_configuration ABC I I_a A' B' C') :
  concurrent {A A'} ( {B B'}) ( {C C'} := sorry

end concurrency_of_perpendiculars_l463_463689


namespace K_is_equidistant_from_A_and_C_l463_463324

open EuclideanGeometry

-- Define the conditions of the problem
variable (A B C H P K : Point)
variable (triangle_ABC : Triangle A B C)
variable (isAcute_triangle_ABC : isAcuteTriangle A B C)
variable (A_prime C_prime : Point)
variable (altitude_BH : Line)
variable (midpoint_A_prime : isMidpoint A_prime B C)
variable (midpoint_C_prime : isMidpoint C_prime A B)
variable (point_P_on_BH : liesOn P altitude_BH)
variable (perpendicular_from_A_prime : Perpendicular_from_line A_prime (line_through C P))
variable (perpendicular_from_C_prime : Perpendicular_from_line C_prime (line_through A P))
variable (intersection_K : intersection perpendicular_from_A_prime perpendicular_from_C_prime = K)

-- Define the statement to prove
theorem K_is_equidistant_from_A_and_C :
  equidistant K A C :=
sorry

end K_is_equidistant_from_A_and_C_l463_463324


namespace construct_triangle_from_excenters_l463_463144

theorem construct_triangle_from_excenters:
  ∀ (A1 B1 C1 A B C : Point),
    -- Condition 1: Given points A1, B1, and C1 are excenters
    -- We assume A1, B1, C1 are points representing the excenters of a triangle ABC
    
    -- Condition 2: Constructs based on altitudes
    (is_foot_of_altitude A A1 B1 C1) →
    (is_foot_of_altitude B B1 A1 C1) →
    (is_foot_of_altitude C C1 A1 B1) →
    -- Conclude that correctly constructed A, B, C form triangle ABC
    forms_triangle A B C :=
begin
  sorry
end

end construct_triangle_from_excenters_l463_463144


namespace solution_set_of_inequality_l463_463426

theorem solution_set_of_inequality (x : ℝ) : (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by sorry

end solution_set_of_inequality_l463_463426


namespace correct_measure_of_dispersion_l463_463041

theorem correct_measure_of_dispersion (mean variance median mode : Type) :
  ∃ d : Type, (d = variance) :=
by
  use variance
  sorry

end correct_measure_of_dispersion_l463_463041


namespace sqrt_floor_square_18_l463_463999

-- Condition: the sqrt function and floor function
def sqrt (x : ℝ) : ℝ := Real.sqrt x
def floor (x : ℝ) : ℤ := Int.floor x

-- Mathematically equivalent proof problem
theorem sqrt_floor_square_18 : floor (sqrt 18) ^ 2 = 16 := 
by
  sorry

end sqrt_floor_square_18_l463_463999


namespace add_digits_base9_l463_463114

theorem add_digits_base9 : 
  ∀ n1 n2 n3 : ℕ, 
    (n1 = 2 * 9^2 + 5 * 9^1 + 4 * 9^0) →
    (n2 = 3 * 9^2 + 6 * 9^1 + 7 * 9^0) →
    (n3 = 1 * 9^2 + 4 * 9^1 + 2 * 9^0) →
    ((n1 + n2 + n3) = 7 * 9^2 + 7 * 9^1 + 4 * 9^0) := 
by
  intros n1 n2 n3 h1 h2 h3
  sorry

end add_digits_base9_l463_463114


namespace ten_percent_eq_l463_463302

variable (s t : ℝ)

def ten_percent_of (x : ℝ) : ℝ := 0.1 * x

theorem ten_percent_eq (h : ten_percent_of s = t) : s = 10 * t :=
by sorry

end ten_percent_eq_l463_463302


namespace monkey_ladder_min_rungs_l463_463508

/-- 
  Proof that the minimum number of rungs n that allows the monkey to climb 
  to the top of the ladder and return to the ground, given that the monkey 
  ascends 16 rungs or descends 9 rungs at a time, is 24. 
-/
theorem monkey_ladder_min_rungs (n : ℕ) (ascend descend : ℕ) 
  (h1 : ascend = 16) (h2 : descend = 9) 
  (h3 : (∃ x y : ℤ, 16 * x - 9 * y = n) ∧ 
        (∃ x' y' : ℤ, 16 * x' - 9 * y' = 0)) : 
  n = 24 :=
sorry

end monkey_ladder_min_rungs_l463_463508


namespace statement_min_distance_BC_l463_463486

-- Conditions
def speed_boat_still_water : ℝ := 42
def current_downstream : ℝ := 5
def current_upstream : ℝ := 7
def time_downstream : ℝ := 1 + 10/60
def time_upstream : ℝ := 2 + 30/60

/--
Theorem statement:
Given the speed of the boat in still water is 42 km/hr, the rate of the current while sailing downstream is 5 km/hr with a journey time of 1.1667 hours, and the rate of the current while sailing upstream is 7 km/hr with a journey time of 2.5 hours, 
prove that the minimum distance between point B and point C is 87.5 km.
-/
theorem min_distance_BC : 
  let effective_speed_downstream := speed_boat_still_water + current_downstream,
      distance_AB := effective_speed_downstream * time_downstream,
      effective_speed_upstream := speed_boat_still_water - current_upstream,
      distance_BC := effective_speed_upstream * time_upstream
  in distance_BC = 87.5 :=
by
  let effective_speed_downstream := speed_boat_still_water + current_downstream
  let distance_AB := effective_speed_downstream * time_downstream
  let effective_speed_upstream := speed_boat_still_water - current_upstream
  let distance_BC := effective_speed_upstream * time_upstream
  sorry -- Proof omitted

end statement_min_distance_BC_l463_463486


namespace least_fraction_to_unity_l463_463032

theorem least_fraction_to_unity :
  let S := (Finset.range 20).sum (λ n, 1 / (↑(n + 2) * ↑(n + 3))) in
  S + 0.5227272727272727 = 1 := 
by
  sorry

end least_fraction_to_unity_l463_463032


namespace vote_difference_l463_463084

-- Definitions of initial votes for and against the policy
def vote_initial_for (x y : ℕ) : Prop := x + y = 450
def initial_margin (x y m : ℕ) : Prop := y > x ∧ y - x = m

-- Definitions of votes for and against in the second vote
def vote_second_for (x' y' : ℕ) : Prop := x' + y' = 450
def second_margin (x' y' m : ℕ) : Prop := x' - y' = 3 * m
def second_vote_ratio (x' y : ℕ) : Prop := x' = 10 * y / 9

-- Theorem to prove the increase in votes
theorem vote_difference (x y x' y' m : ℕ)
  (hi : vote_initial_for x y)
  (hm : initial_margin x y m)
  (hs : vote_second_for x' y')
  (hsm : second_margin x' y' m)
  (hr : second_vote_ratio x' y) : 
  x' - x = 52 :=
sorry

end vote_difference_l463_463084


namespace unique_pair_of_angles_l463_463882

theorem unique_pair_of_angles :
  ∃! (x k : ℝ), (k > 1 ∧ k ∈ ℤ) ∧
    (∀ n1 n2 : ℕ, n1 ≠ n2 → 180 - 360 / n1 = x → 180 - 360 / n2 = k * x) → 
    (x = 60 ∧ k = 2) :=
by sorry

end unique_pair_of_angles_l463_463882


namespace net_increase_in_wealth_l463_463487

-- Definitions for yearly changes and fees
def firstYearChange (initialAmt : ℝ) : ℝ := initialAmt * 1.75 - 0.02 * initialAmt * 1.75
def secondYearChange (amt : ℝ) : ℝ := amt * 0.7 - 0.02 * amt * 0.7
def thirdYearChange (amt : ℝ) : ℝ := amt * 1.45 - 0.02 * amt * 1.45
def fourthYearChange (amt : ℝ) : ℝ := amt * 0.85 - 0.02 * amt * 0.85

-- Total Value after 4th year accounting all changes and fees
def totalAfterFourYears (initialAmt : ℝ) : ℝ :=
  let afterFirstYear := firstYearChange initialAmt
  let afterSecondYear := secondYearChange afterFirstYear
  let afterThirdYear := thirdYearChange afterSecondYear
  fourthYearChange afterThirdYear

-- Capital gains tax calculation
def capitalGainsTax (initialAmt finalAmt : ℝ) : ℝ :=
  0.20 * (finalAmt - initialAmt)

-- Net value after taxes
def netValueAfterTaxes (initialAmt : ℝ) : ℝ :=
  let total := totalAfterFourYears initialAmt
  total - capitalGainsTax initialAmt total

-- Main theorem statement
theorem net_increase_in_wealth :
  ∀ (initialAmt : ℝ), netValueAfterTaxes initialAmt = initialAmt * 1.31408238206 := sorry

end net_increase_in_wealth_l463_463487


namespace drinkable_amount_l463_463343

variable {LiquidBeforeTest : ℕ}
variable {Threshold : ℕ}

def can_drink_more (LiquidBeforeTest : ℕ) (Threshold : ℕ): ℕ :=
  Threshold - LiquidBeforeTest

theorem drinkable_amount :
  LiquidBeforeTest = 24 ∧ Threshold = 32 →
  can_drink_more LiquidBeforeTest Threshold = 8 := by
  sorry

end drinkable_amount_l463_463343


namespace calculation1_calculation2_calculation3_calculation4_l463_463133

-- Define the problem and conditions
theorem calculation1 : 9.5 * 101 = 959.5 := 
by 
  sorry

theorem calculation2 : 12.5 * 8.8 = 110 := 
by 
  sorry

theorem calculation3 : 38.4 * 187 - 15.4 * 384 + 3.3 * 16 = 1320 := 
by 
  sorry

theorem calculation4 : 5.29 * 73 + 52.9 * 2.7 = 529 := 
by 
  sorry

end calculation1_calculation2_calculation3_calculation4_l463_463133


namespace blue_faces_ratio_l463_463845

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l463_463845


namespace floor_T_value_l463_463364

noncomputable def floor_T : ℝ := 
  let p := (0 : ℝ)
  let q := (0 : ℝ)
  let r := (0 : ℝ)
  let s := (0 : ℝ)
  p + q + r + s

theorem floor_T_value (p q r s : ℝ) (hpq: p^2 + q^2 = 2500) (hrs: r^2 + s^2 = 2500) (hpr: p * r = 1200) (hqs: q * s = 1200) (hpos: p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 140 := 
  by
  sorry

end floor_T_value_l463_463364


namespace blue_face_area_factor_l463_463802

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l463_463802


namespace triangle_ratio_l463_463337

variables {k p q : ℝ} -- k for area of triangle ABC, p and q for ratio in the problem

def is_equal_ratio := p / q = 7 / 3

theorem triangle_ratio (BR_eq_RC : True) (CS_eq_3SA : True)
    (AT_TB_ratio : True) (area_RST_eq_2TBR : True) :
    is_equal_ratio :=
by sorry

end triangle_ratio_l463_463337


namespace red_blue_area_ratio_is_12_l463_463808

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l463_463808


namespace blue_red_face_area_ratio_l463_463812

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l463_463812


namespace solve_for_x_l463_463393

theorem solve_for_x : 
  let x := (Real.sqrt (8 ^ 2 + 15 ^ 2)) / (Real.sqrt (36 + 64))
  in x = 17 / 10 :=
by
  let x := (Real.sqrt (8 ^ 2 + 15 ^ 2)) / (Real.sqrt (36 + 64))
  show x = 17 / 10
  sorry

end solve_for_x_l463_463393


namespace sum_of_valid_c_l463_463203

theorem sum_of_valid_c : 
  let discriminant (c : ℕ) := 49 - 12 * c in
  (∀ (c : ℕ), (3 * x^2 + 7 * x + c = 0) → (∃ k : ℕ, discriminant c = k^2)) →
  (∑ c in (finset.filter (λ c, (∃ k : ℕ, discriminant c = k^2) ∧ c > 0 ∧ c < 5) (finset.range 5)), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463203


namespace expr1_value_expr2_value_l463_463977

open Real

noncomputable def expr1 : ℝ :=
  (9 / 4)^(1/2) - (-1.2)^0 - (27 / 8)^(-2/3) + (3 / 2)^(-2)

noncomputable def expr2 : ℝ :=
  logBase 3 (27^(1/4)) + log 25 + log 4 - (7^(logBase 7 2))

theorem expr1_value : expr1 = 1 / 2 := by
  sorry

theorem expr2_value : expr2 = 3 / 4 := by
  sorry

end expr1_value_expr2_value_l463_463977


namespace find_cost_prices_l463_463101

noncomputable def cost_price_per_meter
  (selling_price_per_meter : ℕ) (loss_per_meter : ℕ) : ℕ :=
  selling_price_per_meter + loss_per_meter

theorem find_cost_prices
  (selling_A : ℕ) (meters_A : ℕ) (loss_A : ℕ)
  (selling_B : ℕ) (meters_B : ℕ) (loss_B : ℕ)
  (selling_C : ℕ) (meters_C : ℕ) (loss_C : ℕ)
  (H_A : selling_A = 9000) (H_meters_A : meters_A = 300) (H_loss_A : loss_A = 6)
  (H_B : selling_B = 7000) (H_meters_B : meters_B = 250) (H_loss_B : loss_B = 4)
  (H_C : selling_C = 12000) (H_meters_C : meters_C = 400) (H_loss_C : loss_C = 8) :
  cost_price_per_meter (selling_A / meters_A) loss_A = 36 ∧
  cost_price_per_meter (selling_B / meters_B) loss_B = 32 ∧
  cost_price_per_meter (selling_C / meters_C) loss_C = 38 :=
by {
  sorry
}

end find_cost_prices_l463_463101


namespace lines_symmetric_about_y_axis_l463_463277

theorem lines_symmetric_about_y_axis (m n p : ℝ) :
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ x + n * y + p = 0)
  ↔ (m = -n ∧ p = -5) :=
sorry

end lines_symmetric_about_y_axis_l463_463277


namespace max_ratio_l463_463225

theorem max_ratio {a b c d : ℝ} 
  (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d > 0) 
  (h2 : a^2 + b^2 + c^2 + d^2 = (a + b + c + d)^2 / 3) : 
  ∃ x, x = (7 + 2 * Real.sqrt 6) / 5 ∧ x = (a + c) / (b + d) :=
by
  sorry

end max_ratio_l463_463225


namespace carlos_salary_in_july_l463_463980

theorem carlos_salary_in_july 
  (initial_salary : ℝ) 
  (june_raise_percent july_cut_percent : ℝ) 
  (initial_salary_eq : initial_salary = 3000) 
  (june_raise_percent_eq : june_raise_percent = 0.15) 
  (july_cut_percent_eq : july_cut_percent = 0.10) 
  : 
  (initial_salary * (1 + june_raise_percent) * (1 - july_cut_percent) = 3105) :=
by 
  rw [initial_salary_eq, june_raise_percent_eq, july_cut_percent_eq]
  simp
  norm_num
  sorry

end carlos_salary_in_july_l463_463980


namespace otimes_nested_l463_463213

-- Define the operation ⊗
def otimes (a b c : ℝ) (h : b ≠ c) : ℝ := a / (b - c)

-- Define the specific proof problem
theorem otimes_nested :
  otimes (otimes 2 4 6 (by norm_num))
         (otimes 3 6 2 (by norm_num))
         (otimes 4 2 5 (by norm_num))
         (by {
           norm_num, 
           have h1 : (3 / 4) ≠ - (4 / 3), by norm_num, 
           exact h1
         }) 
  = - (12 / 25) := 
sorry

end otimes_nested_l463_463213


namespace gcd_f_50_51_l463_463358

def f (x : ℤ) : ℤ :=
  x ^ 2 - 2 * x + 2023

theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 11 := by
  sorry

end gcd_f_50_51_l463_463358


namespace sum_of_valid_c_l463_463206

theorem sum_of_valid_c : 
  let discriminant (c : ℕ) := 49 - 12 * c in
  (∀ (c : ℕ), (3 * x^2 + 7 * x + c = 0) → (∃ k : ℕ, discriminant c = k^2)) →
  (∑ c in (finset.filter (λ c, (∃ k : ℕ, discriminant c = k^2) ∧ c > 0 ∧ c < 5) (finset.range 5)), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463206


namespace blue_face_area_factor_l463_463796

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l463_463796


namespace function_conditions_l463_463611

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := (b - 2^(-x)) / (2^(-x + 1) + 2)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, x1 < x2 → f (x1) < f (x2)

axiom exists_inequality (k : ℝ) (t : ℝ) :
  (f (t^2 - 2 * t) 1 + f (2 * t^2 - k) 1 < 0) → (-1 / 3 < k)

theorem function_conditions :
  (is_odd_function (f x b)) →
  (b = 1) ∧
  (increasing (λ x, f x 1)) ∧
  (∀ t : ℝ, exists_inequality k t)
:=
sorry

end function_conditions_l463_463611


namespace blue_red_area_ratio_l463_463842

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l463_463842


namespace smallest_integer_mod_inverse_l463_463909

theorem smallest_integer_mod_inverse (n : ℕ) (h1 : n > 1) (h2 : gcd n 1001 = 1) : n = 2 :=
sorry

end smallest_integer_mod_inverse_l463_463909


namespace minimum_cubes_needed_l463_463451

def min_number_of_cubes (n : ℕ) : Prop :=
  ∀ (digits : Fin 10), ∃ (cubes : Fin n → Fin 6 → Fin 10),
    ∃ (comb : Finset (Fin 10 × Fin 10 × Fin 10)), 
    comb = {⟨d1, d2, d3⟩ | ∃ c1 c2 c3 : Fin n,
             c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧ 
             cubes c1 ⟨0%6⟩ = d1 ∧ 
             cubes c2 ⟨0%6⟩ = d2 ∧ 
             cubes c3 ⟨0%6⟩ = d3}

theorem minimum_cubes_needed : min_number_of_cubes 5 :=
by
  sorry

end minimum_cubes_needed_l463_463451


namespace sum_prime_factors_221_l463_463459

theorem sum_prime_factors_221 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 221 = p * q ∧ p + q = 30 :=
by {
  existsi 13,
  existsi 17,
  split,
  { exact nat.prime_of_nat_abs_prime 13 (by norm_num) },
  split,
  { exact nat.prime_of_nat_abs_prime 17 (by norm_num) },
  split,
  { exact nat.mul_eq_of_eq_div (by norm_num) (by norm_num) },
  { exact rfl }
}

end sum_prime_factors_221_l463_463459


namespace sequence_general_formula_l463_463335

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n + 2 * n - 1) :
  ∀ n : ℕ, a n = (2 / 3) * 3^n - n :=
by
  sorry

end sequence_general_formula_l463_463335


namespace part1_part2_l463_463263

noncomputable def f (x : ℝ) := sin (2 * x - π / 6) + cos (x) ^ 2

theorem part1 (θ : ℝ) (h : f θ = 1) : sin θ * cos θ = sqrt 3 / 6 := 
sorry

theorem part2 : ∀ k : ℤ, ∀ x ∈ set.Icc (-π/4 + k * π) (π/4 + k * π), 
  (deriv f x > 0) := 
sorry

end part1_part2_l463_463263


namespace blue_red_area_ratio_l463_463836

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l463_463836


namespace tic_tac_toe_winning_probability_l463_463528

theorem tic_tac_toe_winning_probability :
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  probability = (2 : ℚ) / 21 :=
by
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  have probability_correct : probability = (2 : ℚ) / 21 := sorry
  exact probability_correct

end tic_tac_toe_winning_probability_l463_463528


namespace units_digit_of_17_pow_28_l463_463914

theorem units_digit_of_17_pow_28 : ((17:ℕ) ^ 28) % 10 = 1 := by
  sorry

end units_digit_of_17_pow_28_l463_463914


namespace kidsMealCost_l463_463973

noncomputable def BurgerCost : ℝ := 5
noncomputable def FriesCost : ℝ := 3
noncomputable def DrinkCost : ℝ := 3
noncomputable def BurgerMealCost : ℝ := 9.50
noncomputable def KidsBurgerCost : ℝ := 3
noncomputable def KidsFriesCost : ℝ := 2
noncomputable def KidsJuiceCost : ℝ := 2
noncomputable def Savings : ℝ := 10

def IndividualAdultCost : ℝ := 2 * BurgerCost + 2 * FriesCost + 2 * DrinkCost
def MealAdultCost : ℝ := 2 * BurgerMealCost
def IndividualKidsCost : ℝ := 2 * KidsBurgerCost + 2 * KidsFriesCost + 2 * KidsJuiceCost
def TotalIndividualCost : ℝ := IndividualAdultCost + IndividualKidsCost

-- Let K be the cost of one kids' meal
variable {K : ℝ}

def MealKidsCost : ℝ := 2 * K
def TotalMealCost : ℝ := MealAdultCost + MealKidsCost

-- Savings condition
def SavingsCondition : Prop := TotalIndividualCost - Savings = TotalMealCost

-- We want to prove K = 3.50 under the given conditions
theorem kidsMealCost (h : SavingsCondition) : K = 3.50 := by
  sorry

end kidsMealCost_l463_463973


namespace lunch_ratio_l463_463434

def total_school_days : ℕ := 180
def becky_lunch_days : ℕ := 45

def aliyah_lunch_days (becky_days : ℕ) : ℕ := becky_days * 2

theorem lunch_ratio (total_days : ℕ) (becky_days : ℕ) (aliyah_days : ℕ) :
  (total_days = 180) → 
  (becky_days = 45) → 
  (aliyah_days = aliyah_lunch_days becky_days) → 
  (aliyah_days : total_days) = (1 : 2) :=
by
  intros h_total_days h_becky_days h_aliyah_days
  rw [h_total_days, h_becky_days, h_aliyah_days]
  sorry

end lunch_ratio_l463_463434


namespace min_over_max_ratio_l463_463250

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := sqrt (1 - x) + sqrt (x + 3)

-- The theorem to prove
theorem min_over_max_ratio (M m : ℝ) (h1 : ∀ x ∈ Ici (-3) ∩ Iic 1, f x ≤ M) (h2 : ∀ x ∈ Ici (-3) ∩ Iic 1, m ≤ f x) (h_max : M = 2 * sqrt 2) (h_min : m = 2) : m / M = sqrt 2 / 2 :=
by sorry

end min_over_max_ratio_l463_463250


namespace abigail_initial_money_l463_463110

variable (X : ℝ) -- Let X be the initial amount of money

def spent_on_food (X : ℝ) := 0.60 * X
def remaining_after_food (X : ℝ) := X - spent_on_food X
def spent_on_phone (X : ℝ) := 0.25 * remaining_after_food X
def remaining_after_phone (X : ℝ) := remaining_after_food X - spent_on_phone X
def final_amount (X : ℝ) := remaining_after_phone X - 20

theorem abigail_initial_money
    (food_spent : spent_on_food X = 0.60 * X)
    (phone_spent : spent_on_phone X = 0.10 * X)
    (remaining_after_entertainment : final_amount X = 40) :
    X = 200 :=
by
    sorry

end abigail_initial_money_l463_463110


namespace percent_equivalence_l463_463938

def percent (p : ℝ) (x : ℝ) := (p / 100) * x

theorem percent_equivalence :
  percent 25 2004 = percent 50 1002 := 
by 
  -- The proof steps go here, but for now we use sorry to skip the proof.
  sorry

end percent_equivalence_l463_463938


namespace initial_percentage_reduction_l463_463868

theorem initial_percentage_reduction
  (x: ℕ)
  (h1: ∀ P: ℝ, P * (1 - x / 100) * 0.85 * 1.5686274509803921 = P) :
  x = 25 :=
by
  sorry

end initial_percentage_reduction_l463_463868


namespace area_of_quadrilateral_ZEBK_l463_463886

-- Define the key components of the problem in Lean.
noncomputable def triangle_ABC : Prop :=
  ∃ (A B C : ℝ × ℝ),
  A = (0, 0) ∧ B = (2, 0) ∧ C = (0, 4) ∧
  ∃ (ω1 ω2 : ℝ × ℝ → ℝ),
  ω1 = λ p, (p.1)^2 + (p.2 - 4)^2 - 16 ∧
  ω2 = λ p, (p.1 - 2)^2 + (p.2)^2 - 4 ∧
  
  ∃ (E : ℝ × ℝ),
  ω1 E = 0 ∧ ω2 E = 0 ∧ E ≠ (0, 0) ∧ E.1 > 0 ∧ E.2 > 0 ∧

  ∃ (M : ℝ × ℝ),
  ω2 M = 0 ∧ M.1 > 0 ∧ M.2 > 0 ∧
  let EC_slope := (E.2 - 4) / E.1 in
  let BM_slope := if (M.1 - 2) = 0 then 0 else (M.2 / (M.1 - 2)) in
  EC_slope = BM_slope ∧

  ∃ (K Z : ℝ × ℝ),
  let EM_slope := (M.2 - E.2) / (M.1 - E.1) in
  (∃ t : ℝ, K = (E.1 + t * (M.1 - E.1), E.2 + t * (M.2 - E.2))) ∧
  let AM_slope := M.2 / M.1 in
  (∃ t : ℝ, Z = (t * M.1, t * M.2)) ∧
  
  ∃ (area : ℝ),
  area = 1 / 2 * abs (
    K.1 * E.2 + E.1 * 0 + 2 * K.2 + (-12 / 5) * Z.2 - (
    Z.1 * E.2 + E.1 * 2 + 0 + K.2 * Z.1)) ∧
  area = 20
  
-- The final theorem to be proven.
theorem area_of_quadrilateral_ZEBK : 
  triangle_ABC := sorry

end area_of_quadrilateral_ZEBK_l463_463886


namespace binomial_expansion_negate_l463_463569

theorem binomial_expansion_negate :
  (1 - 3 * Nat.choose 10 1 + 9 * Nat.choose 10 2 - 27 * Nat.choose 10 3 +
   ∑ k in Finset.Ico 4 10, if k % 2 = 0 then (3^k * Nat.choose 10 k) else -(3^k * Nat.choose 10 k)) + 3^10 = -1024 :=
by
  sorry

end binomial_expansion_negate_l463_463569


namespace g_equals_cos_x_l463_463787

variable (f g: ℝ → ℝ)

def is_even (h : ℝ → ℝ) : Prop := ∀ x : ℝ, h(-x) = h(x)

axiom a_dot_b : ∀ x : ℝ, (1:ℝ) * f(x) + x * (-x) = g(x)
axiom f_even : is_even f

theorem g_equals_cos_x (x : ℝ) (h : g(x) = cos x) : is_even g :=
by 
  intro x
  have h_f : f(x) = x^2 + g(x), from sorry
  have h_f_even : ∀ x, f(-x) = f(x), from sorry
  show g(-x) = g(x), from sorry

end g_equals_cos_x_l463_463787


namespace find_dividend_l463_463496

theorem find_dividend (q : ℕ) (d : ℕ) (r : ℕ) (D : ℕ) 
  (h_q : q = 15000)
  (h_d : d = 82675)
  (h_r : r = 57801)
  (h_D : D = 1240182801) :
  D = d * q + r := by 
  sorry

end find_dividend_l463_463496


namespace sum_of_valid_c_values_l463_463195

theorem sum_of_valid_c_values:
  let quadratic_rational_roots := λ (c : ℕ), (∃ d : ℕ, 49 - 12 * c = d * d) ∧ (c > 0)
  in ∑ c in finset.filter quadratic_rational_roots (finset.range 5), c = 6 :=
by
  -- The proof will go here
  sorry

end sum_of_valid_c_values_l463_463195


namespace polar_coordinates_of_M_l463_463773

theorem polar_coordinates_of_M :
  ∀ (Ox OP M : ℝ) (θ : ℝ),
  Ox > 0 ∧ OP = Ox - π/6 ∧ M = OP ∧ ¬θ = 0 ∧ θ ∈ Set.Ico 0 (2*π) → 
  θ = 2 * π - π / 6 → 
  (4, 11 * π / 6) := 
by
  intro Ox OP M θ h1 h2
  sorry

end polar_coordinates_of_M_l463_463773


namespace max_angle_between_tangents_l463_463634

open Real

theorem max_angle_between_tangents (x y : ℝ) (hP : y^2 = 4 * x) :
    let d := sqrt ((x - 3)^2 + y^2),
        r := sqrt 2,
        θ := 60 * (π / 180)
    in d ≥ 2 * sqrt 2 ∧ θ = 60 * (π / 180) :=
by
  sorry

end max_angle_between_tangents_l463_463634


namespace parabola_vertex_focus_l463_463090

open Real

noncomputable def parabola_equation (p : ℝ) : ℝ → ℝ → Prop :=
λ x y, y^2 = 2 * p * x

theorem parabola_vertex_focus (h : parabola_equation 1 2 2) : ∀ x y, parabola_equation 1 x y ↔ parabola_equation 1 x y :=
by 
  -- The parabola passes through the point (2, 2), from which we can derive p = 1
  have hp : 4 = 2 * 2 * 1 := by norm_num,
  exact sorry

end parabola_vertex_focus_l463_463090


namespace probability_of_winning_position_l463_463534

-- Given Conditions
def tic_tac_toe_board : Type := Fin 3 × Fin 3
def is_nought (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := b p
def is_cross (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := ¬ b p

-- Winning positions in Tic-Tac-Toe are vertical, horizontal, or diagonal lines
def is_winning_position (b : tic_tac_toe_board → bool) : Prop :=
  (∃ i : Fin 3, ∀ j : Fin 3, is_nought b (i, j)) ∨ -- Row
  (∃ j : Fin 3, ∀ i : Fin 3, is_nought b (i, j)) ∨ -- Column
  (∀ i : Fin 3, is_nought b (i, i)) ∨ -- Main diagonal
  (∀ i : Fin 3, is_nought b (i, Fin.mk (2 - i.1) (by simp [i.1]))) -- Anti-diagonal

-- Problem Statement
theorem probability_of_winning_position :
  let total_positions := Nat.choose 9 3
  let winning_positions := 8
  winning_positions / total_positions = (2:ℚ) / 21 :=
by sorry

end probability_of_winning_position_l463_463534


namespace find_a_3_l463_463612

variable {a : ℕ → ℝ} -- Define the geometric sequence
variable {S : ℕ → ℝ} -- Define the sum of the first n terms

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = a n * 2
axiom first_term : a 1 = 1
axiom sum_seq (n : ℕ) : S n = ∑ i in Finset.range n, a i
axiom arith_seq (n : ℕ) : 2 * S (n + 1) = S n + S (n + 2)

-- Proof to show a_3 = 1/4 given the conditions
theorem find_a_3 : a 3 = 1 / 4 := sorry

end find_a_3_l463_463612


namespace B_squared_B_sixth_l463_463732

noncomputable def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![0, 3], ![2, -1]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  1

theorem B_squared :
  B * B = 3 * B - I := by
  sorry

theorem B_sixth :
  B^6 = 84 * B - 44 * I := by
  sorry

end B_squared_B_sixth_l463_463732


namespace james_passenger_count_l463_463717

theorem james_passenger_count :
  ∀ (total_vehicles trucks buses taxis motorbikes cars trucks_population buses_population taxis_population motorbikes_population cars_population : ℕ),
  total_vehicles = 52 →
  trucks = 12 →
  buses = 2 →
  taxis = 2 * buses →
  motorbikes = total_vehicles - (trucks + buses + taxis + cars) →
  cars = 30 →
  trucks_population = 2 →
  buses_population = 15 →
  taxis_population = 2 →
  motorbikes_population = 1 →
  cars_population = 3 →
  (trucks * trucks_population + buses * buses_population + taxis * taxis_population +
   motorbikes * motorbikes_population + cars * cars_population) = 156 := 
by
  -- Placeholder for the proof
  sorry

end james_passenger_count_l463_463717


namespace nell_initial_ace_cards_l463_463760

def initial_ace_cards (initial_baseball_cards final_ace_cards final_baseball_cards given_difference : ℕ) : ℕ :=
  final_ace_cards + (initial_baseball_cards - final_baseball_cards)

theorem nell_initial_ace_cards : 
  initial_ace_cards 239 376 111 265 = 504 :=
by
  /- This is to show that the initial count of Ace cards Nell had is 504 given the conditions -/
  sorry

end nell_initial_ace_cards_l463_463760


namespace find_s_l463_463734

theorem find_s (n r s c d : ℚ) 
  (h1 : Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 3 = 0) 
  (h2 : c * d = 3)
  (h3 : Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s = 
        Polynomial.C (c + d⁻¹) * Polynomial.C (d + c⁻¹)) : 
  s = 16 / 3 := 
by
  sorry

end find_s_l463_463734


namespace min_mnp_l463_463016

theorem min_mnp (m n p : Nat) (P_m : Nat.prime m) (P_n : Nat.prime n) (P_p : Nat.prime p) (H : m + n = p) (H_distinct : m ≠ n ∧ m ≠ p ∧ n ≠ p) :
  m * n * p = 30 ∨ m * n * p > 30 := sorry

end min_mnp_l463_463016


namespace quadrilateral_with_distinct_sides_is_trapezoid_l463_463304

-- Defining the distinct elements condition
def four_distinct_elements (a b c d : ℝ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Statement: Given that a, b, c, and d are distinct, the quadrilateral with these side lengths is a trapezoid.
theorem quadrilateral_with_distinct_sides_is_trapezoid (a b c d : ℝ) 
  (h : four_distinct_elements a b c d) : 
  (is_trapezoid a b c d) :=
sorry

end quadrilateral_with_distinct_sides_is_trapezoid_l463_463304


namespace sequence_general_term_l463_463229

open Nat

axiom sequence (a : ℕ → ℝ)
axiom partial_sum (S : ℕ → ℝ)
axiom sequence_relation : ∀ n : ℕ, S n = (3 - ((n + 3) / (n + 1)) * a n)

theorem sequence_general_term (n : ℕ) (hn : 0 < n) : a n = (n + 1) / (2 ^ n) :=
by
  sorry

end sequence_general_term_l463_463229


namespace radius_of_centroid_sphere_distance_from_O_to_center_of_S_l463_463152

variables {V : Type} [inner_product_space ℝ V]

-- Definitions of vectors representing the positions of the vertices
variables {a b c d : V}

-- Conditions: tetrahedron T inscribed in a unit sphere, so all vectors have norm 1
axiom ha : ∥a∥ = 1
axiom hb : ∥b∥ = 1
axiom hc : ∥c∥ = 1
axiom hd : ∥d∥ = 1

-- Prove that the radius of the sphere passing through centroids of each face is 1/3
theorem radius_of_centroid_sphere : 
  let G_bcd := (b + c + d) / 3 in
  let G_acd := (a + c + d) / 3 in
  let G_abd := (a + b + d) / 3 in
  let G_abc := (a + b + c) / 3 in
  let P := (a + b + c + d) / 4 in
  ∥G_bcd - P∥ = 1 / 3 := sorry

-- Prove the distance from O to the center of S as a function of the edges of T
theorem distance_from_O_to_center_of_S (AB AC AD BC BD CD : ℝ) :
  let G := (a + b + c + d) / 4 in 
  let edge_len_edge (u v: V): ℝ := real.sqrt(∥u - v∥^2) in
  edge_len_edge a b = AB ∧
  edge_len_edge a c = AC ∧
  edge_len_edge a d = AD ∧
  edge_len_edge b c = BC ∧
  edge_len_edge b d = BD ∧
  edge_len_edge c d = CD
  → ∥G∥ = sqrt ((4 + 2 * (6 - 1/2 * (AB^2 + AC^2 + AD^2 + BC^2 + BD^2 + CD^2))) / 16) :=
sorry

end radius_of_centroid_sphere_distance_from_O_to_center_of_S_l463_463152


namespace floor_T_value_l463_463363

noncomputable def floor_T : ℝ := 
  let p := (0 : ℝ)
  let q := (0 : ℝ)
  let r := (0 : ℝ)
  let s := (0 : ℝ)
  p + q + r + s

theorem floor_T_value (p q r s : ℝ) (hpq: p^2 + q^2 = 2500) (hrs: r^2 + s^2 = 2500) (hpr: p * r = 1200) (hqs: q * s = 1200) (hpos: p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 140 := 
  by
  sorry

end floor_T_value_l463_463363


namespace find_HCF_l463_463306

-- Given conditions
def LCM : ℕ := 750
def product_of_two_numbers : ℕ := 18750

-- Proof statement
theorem find_HCF (h : ℕ) (hpos : h > 0) :
  (LCM * h = product_of_two_numbers) → h = 25 :=
by
  sorry

end find_HCF_l463_463306


namespace find_p_probability_additional_training_l463_463097

/- Define the random variable X and associated parameters -/
def X_success_shots (n : ℕ) (p : ℝ) : ℕ → Prop :=
  λ (k : ℕ), binomial_dist n p k

/- Define the conditions -/
def expected_value_X (n : ℕ) (p : ℝ) : ℝ :=
  let X := X_success_shots n p in
  X.mean_val

/- Proof statements -/
theorem find_p (n : ℕ) :
  expected_value_X n p = 4 → p = 2/3 :=
sorry

theorem probability_additional_training (n : ℕ) (p : ℝ) :
  p = 2/3 →
  P (λ k, k = 4 ∨ k = 5 ∨ k = 6) (binomial_dist n (1-p)) = 64/729 :=
sorry

/- Check successful build -/
#check find_p
#check probability_additional_training

end find_p_probability_additional_training_l463_463097


namespace manuscript_age_in_decimal_l463_463122

-- Given conditions
def octal_number : ℕ := 12345

-- Translate the problem statement into Lean:
theorem manuscript_age_in_decimal : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 :=
by
  sorry

end manuscript_age_in_decimal_l463_463122


namespace correct_proposition_is_D_l463_463119

-- Define the propositions
def propositionA : Prop :=
  (∀ x : ℝ, x^2 = 4 → x = 2 ∨ x = -2) → (∀ x : ℝ, (x ≠ 2 ∨ x ≠ -2) → x^2 ≠ 4)

def propositionB (p : Prop) : Prop :=
  (p → (∀ x : ℝ, x^2 - 2*x + 3 > 0)) → (¬p → (∃ x : ℝ, x^2 - 2*x + 3 < 0))

def propositionC : Prop :=
  ∀ (a b : ℝ) (n : ℕ), a > b → n > 0 → a^n > b^n

def p : Prop := ∀ x : ℝ, x^3 ≥ 0
def q : Prop := ∀ e : ℝ, e > 0 → e < 1
def propositionD := p ∧ q

-- The proof problem
theorem correct_proposition_is_D : propositionD :=
  sorry

end correct_proposition_is_D_l463_463119


namespace necessary_and_sufficient_condition_sufficient_not_necessary_condition_l463_463275

-- Defining the sets M and P
def M (x : ℝ) : Prop := (x < -3) ∨ (x > 5)
def P (x a : ℝ) : Prop := (x - a) * (x - 8) ≤ 0

-- The necessary and sufficient condition (1)
theorem necessary_and_sufficient_condition (a : ℝ) :
  (∃ x : ℝ, M x ∧ P x a ∧ 5 < x ∧ x ≤ 8) ↔ (a ∈ Icc (-3 : ℝ) 5) := sorry

-- Sufficient but not necessary condition (2)
theorem sufficient_not_necessary_condition (a : ℝ) :
  (a = 0 → (∃ x : ℝ, M x ∧ P x a ∧ 5 < x ∧ x ≤ 8)) ∧
  (M ∩ P = { x : ℝ | 5 < x ∧ x ≤ 8 } → (∃ a' : ℝ, a' ≠ 0 ∧ (∃ x : ℝ, M x ∧ P x a' ∧ 5 < x ∧ x ≤ 8))) := sorry

end necessary_and_sufficient_condition_sufficient_not_necessary_condition_l463_463275


namespace min_n_for_constant_term_l463_463224

theorem min_n_for_constant_term (n : ℕ) (h : 0 < n) : 
  (∃ (r : ℕ), 0 = n - 4 * r / 3) → n = 4 :=
by
  sorry

end min_n_for_constant_term_l463_463224


namespace determine_hyperbola_equation_l463_463269

-- Define the conditions of the hyberbola
def hyperbola_equation (a b : ℝ) : Prop := ∀ x y : ℝ, (x / a)^2 - (y / b)^2 = 1
def eccentricity (c a : ℝ) : ℝ := c / a

-- Define the specific conditions
def given_conditions : Prop :=
  hyperbola_equation a b /\ 
  eccentricity 5 a = 5 / 3

-- Define the target equation
def target_equation : Prop := hyperbola_equation 3 4

-- Theorem statement
theorem determine_hyperbola_equation (a b : ℝ) (h : given_conditions) : target_equation :=
by
  sorry

end determine_hyperbola_equation_l463_463269


namespace angle_in_third_quadrant_l463_463556

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2023) :
    ∃ k : ℤ, (2023 - k * 360) = 223 ∧ 180 ≤ 223 ∧ 223 < 270 := by
sorry

end angle_in_third_quadrant_l463_463556


namespace degree_of_polynomial_P_l463_463453

noncomputable def P : Polynomial ℝ :=
  Polynomial.C 7 * Polynomial.X ^ 5 +
  Polynomial.C 4 * Polynomial.X ^ 3 +
  Polynomial.C 8 * Polynomial.pi * Polynomial.X ^ 6 +
  Polynomial.C (3 * Real.sqrt 2) * Polynomial.X ^ 2 +
  Polynomial.C 15 +
  Polynomial.C 2

theorem degree_of_polynomial_P : P.degree = 6 :=
by
  -- proof goes here
  sorry

end degree_of_polynomial_P_l463_463453


namespace common_chord_length_l463_463657

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - sqrt 3)^2 = 9

-- Define the theorem stating that the length of the common chord is sqrt(65)/2
theorem common_chord_length :
  ∃ l : ℝ, l = (sqrt 65) / 2 ∧
    (
      ∃ (x y : ℝ), circle1 x y ∧ circle2 x y
    ) →
    (
      let line_eq (x y : ℝ) : Prop := 2 * x - 2 * (sqrt 3) * y - 3 = 0 in
      let center_O1 := (1 : ℝ, 0 : ℝ) in
      let d := abs ((2 * 1 - 2 * (sqrt 3) * 0 - 3) / (sqrt (2^2 + (2 * (sqrt 3))^2))) in
      d = 1 / 4 →
      l = 2 * sqrt (4 - d^2)
    ) :=
begin
  sorry
end

end common_chord_length_l463_463657


namespace principal_amount_l463_463469

variable (P : ℝ)

def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem principal_amount :
  (simple_interest P 4 5) = P - 2000 → P = 2500 :=
by
  intro h
  sorry

end principal_amount_l463_463469


namespace number_of_subsets_of_A_l463_463007

-- Definitions for the problem
def A : Set ℕ := {1, 2}
def num_elements (s : Set ℕ) : Nat := s.toFinset.card

-- The number of subsets theorem
theorem number_of_subsets_of_A : num_elements A = 2 → 2 ^ (num_elements A) = 4 := by
  intros h
  rw [h]
  norm_num
  -- Since this doesn't require an actual proof, we indicate it's skipped
  sorry

end number_of_subsets_of_A_l463_463007


namespace blue_face_area_greater_than_red_face_area_l463_463832

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l463_463832


namespace problem_statement_l463_463049

def pair_product_not_minus_48 : Prop :=
  ∃ a b : ℤ, (a = 3 ∧ b = 16 ∧ a * b ≠ -48)

theorem problem_statement : pair_product_not_minus_48 :=
by
succeeding sorry := skipProof 0
succeeding intro a
succeeding intro b
succeeding handler intros h₁
succeeding handler simp at *
succeeding specialize h₁ h₂
succeeding substVars h₂
succeeding cc
sorry

end problem_statement_l463_463049


namespace volume_cylinders_equal_volume_of_first_cylinder_eq_49_pi_h1_l463_463449

noncomputable def volume_first_cylinder (h₁ : ℝ) (r₁ : ℝ := 7) : ℝ :=
  π * r₁^2 * h₁

theorem volume_cylinders_equal (h₁ : ℝ) :
  let r₁ := 7
  let r₂ := 1.2 * r₁
  let h₂ := 0.85 * h₁
  π * r₁^2 * h₁ = π * r₂^2 * h₂ :=
by
  let r₁ := 7
  let r₂ := 1.2 * r₁
  let h₂ := 0.85 * h₁
  calc
    π * r₁^2 * h₁ = π * (7 * 7) * h₁ : by rw r₁
               ... = π * 49 * h₁ : by norm_num
               ... = π * (1.2 * 7)^2 * 0.85 * h₁ : by sorry
               -- require more steps for full algebraic simplification
               ... : sorry  -- Concluding the proof with the statement

theorem volume_of_first_cylinder_eq_49_pi_h1 (h₁ : ℝ) : 
  volume_first_cylinder h₁ = 49 * π * h₁ := 
by
  unfold volume_first_cylinder
  calc
    π * (7)^2 * h₁ = π * 49 * h₁ : by norm_num

end volume_cylinders_equal_volume_of_first_cylinder_eq_49_pi_h1_l463_463449


namespace possible_length_of_third_side_l463_463686

theorem possible_length_of_third_side (a b c : ℤ) (h1 : a - b = 7) (h2 : (a + b + c) % 2 = 1) : c = 8 :=
sorry

end possible_length_of_third_side_l463_463686


namespace prove_problem_statement_l463_463312

noncomputable def problem_statement : Prop :=
  ∀ λ : ℝ, ¬(∃ x : ℝ, x ∈ set.Icc (1/2) 2 ∧ 2 * x^2 - λ * x + 1 < 0) ↔ λ ∈ set.Iic (2 * Real.sqrt 2)

theorem prove_problem_statement : problem_statement := sorry

end prove_problem_statement_l463_463312


namespace find_y_l463_463169

theorem find_y (y : ℝ) (h : 9^(Real.log y / Real.log 8) = 81) : y = 64 :=
by
  sorry

end find_y_l463_463169


namespace find_cost_price_l463_463959

-- Definitions, conditions and the equivalent proof problem
theorem find_cost_price (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) 
  (h1 : SP = 1800)
  (h2 : profit_percent = 0.20)
  (h3 : SP = CP * (1 + profit_percent)) :
  CP = 1500 :=
begin
  sorry -- Proof goes here
end

end find_cost_price_l463_463959


namespace blue_to_red_face_area_ratio_l463_463856

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l463_463856


namespace sum_of_valid_c_values_l463_463193

theorem sum_of_valid_c_values:
  let quadratic_rational_roots := λ (c : ℕ), (∃ d : ℕ, 49 - 12 * c = d * d) ∧ (c > 0)
  in ∑ c in finset.filter quadratic_rational_roots (finset.range 5), c = 6 :=
by
  -- The proof will go here
  sorry

end sum_of_valid_c_values_l463_463193


namespace diplomats_speak_french_l463_463126

-- Definitions
variable (T : ℕ) (T_eq : T = 70)
variable (neither_percentage : ℕ) (neither_eq : neither_percentage = 20)
variable (both_percentage : ℕ) (both_eq : both_percentage = 10)
variable (not_speak_Russian : ℕ) (not_speak_Russian_eq : not_speak_Russian = 32)
variable (diplomats_total : ℕ) (diplomats_total_eq : diplomats_total = 70)

-- Prove number of diplomats who speak French
theorem diplomats_speak_french :
  (0.2 * T + (T - not_speak_Russian) - (0.1 * T) = 70 - 14) → F = 25 := by
  sorry

end diplomats_speak_french_l463_463126


namespace remaining_movie_duration_l463_463546

/--
Given:
1. The laptop was fully charged at 3:20 pm.
2. Hannah started watching a 3-hour series.
3. The laptop turned off at 5:44 pm (fully discharged).

Prove:
The remaining duration of the movie Hannah needs to watch is 36 minutes.
-/
theorem remaining_movie_duration
    (start_full_charge : ℕ := 200)  -- representing 3:20 pm as 200 (20 minutes past 3:00)
    (end_discharge : ℕ := 344)  -- representing 5:44 pm as 344 (44 minutes past 5:00)
    (total_duration_minutes : ℕ := 180)  -- 3 hours in minutes
    (start_time_minutes : ℕ := 200)  -- convert 3:20 pm to minutes past noon
    (end_time_minutes : ℕ := 344)  -- convert 5:44 pm to minutes past noon
    : (total_duration_minutes - (end_time_minutes - start_time_minutes)) = 36 :=
by
  sorry

end remaining_movie_duration_l463_463546


namespace blue_face_area_greater_than_red_face_area_l463_463828

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l463_463828


namespace sum_of_exponents_l463_463038

theorem sum_of_exponents (a b c : ℝ) : 
  let radicand := (48 * a^5 * b^8 * c^14 : ℝ)
  let radical := (radicand : ℝ)^(1/4)
  let simplified_outside := (2 * b^2 * c^3 : ℝ)
  in radical = simplified_outside * ((3 * a^5 * c^2 : ℝ)^(1/4)) → 
  (2 + 3 = 5) := 
by
  sorry

end sum_of_exponents_l463_463038


namespace probability_of_winning_position_l463_463535

-- Given Conditions
def tic_tac_toe_board : Type := Fin 3 × Fin 3
def is_nought (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := b p
def is_cross (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := ¬ b p

-- Winning positions in Tic-Tac-Toe are vertical, horizontal, or diagonal lines
def is_winning_position (b : tic_tac_toe_board → bool) : Prop :=
  (∃ i : Fin 3, ∀ j : Fin 3, is_nought b (i, j)) ∨ -- Row
  (∃ j : Fin 3, ∀ i : Fin 3, is_nought b (i, j)) ∨ -- Column
  (∀ i : Fin 3, is_nought b (i, i)) ∨ -- Main diagonal
  (∀ i : Fin 3, is_nought b (i, Fin.mk (2 - i.1) (by simp [i.1]))) -- Anti-diagonal

-- Problem Statement
theorem probability_of_winning_position :
  let total_positions := Nat.choose 9 3
  let winning_positions := 8
  winning_positions / total_positions = (2:ℚ) / 21 :=
by sorry

end probability_of_winning_position_l463_463535


namespace probability_a_n_gets_A_n_l463_463246

/- 
  Define the conditions and required probability.
  a (assume n ≥ 2)
  a_i corresponds to A_i (assuming proper assignment)
  Picking process: a_1 picks first, and so on.
  We want to prove that P_n = 1 / 2 for the probability that a_n gets A_n.
-/

axiom people : Type
axiom cards : Type
axiom corresponds (a : people) (A : cards) : Prop
axiom picking_process (n : ℕ) (h : n ≥ 2) : 
    (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ a : people, ∃ A : cards, corresponds a A))

noncomputable def probability (n : ℕ) : ℚ := 1 / 2

theorem probability_a_n_gets_A_n (n : ℕ) (h : n ≥ 2) :
  let P_n := probability n in P_n = 1 / 2 :=
by sorry

end probability_a_n_gets_A_n_l463_463246


namespace angle_FCG_67_l463_463763

noncomputable def AE := diameter (circle C)
def angle_ABF : ℝ := 81
def angle_EDG : ℝ := 76

theorem angle_FCG_67 :
  ∀ A B C D E F G : Point,
  AE,
  angle_ABF = 81,
  angle_EDG = 76
  → angle FCG = 67 :=
by {
  sorry
}

end angle_FCG_67_l463_463763


namespace evaluate_expression_l463_463163

open Real

def a := 2999
def b := 3000
def delta := b - a

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 :=
by
  let a := 2999
  let b := 3000
  have h1 : b - a = 1 := by sorry
  calc
    3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = a^3 + b^3 - ab^2 - a^2b := by sorry
                                            ... = (b - a)^2 * (b + a)       := by sorry
                                            ... = (1)^2 * (b + a)           := by
                                                                           rw [h1]
                                                                           exact sorry
                                            ... = 3000 + 2999               := by
                                                                           exact sorry
                                            ... = 5999                     := rfl

end evaluate_expression_l463_463163


namespace sufficient_but_not_necessary_l463_463074

theorem sufficient_but_not_necessary (a : ℝ) : a = π / 6 → tan (π - a) = - real.sqrt 3 / 3 :=
by
  sorry

end sufficient_but_not_necessary_l463_463074


namespace max_value_f_min_value_f_in_interval_range_a_l463_463260

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5 * x + 5) / Real.exp x

theorem max_value_f : ∃ x : ℝ, f x = 5 :=
by
  sorry

theorem min_value_f_in_interval : ∃ x ∈ Iic (0 : ℝ), f x = -Real.exp 3 :=
by
  sorry

theorem range_a (a : ℝ) : (∀ x : ℝ, x^2 + 5 * x + 5 - a * Real.exp x ≥ 0) → a ≤ -Real.exp 3 :=
by
  sorry

end max_value_f_min_value_f_in_interval_range_a_l463_463260


namespace r_value_when_n_is_3_l463_463737

namespace Proof

def r (s : ℤ) : ℤ := 3^s + s
def m : ℤ := 3
def s (n : ℤ) : ℤ := 2^n - m

theorem r_value_when_n_is_3 : r (s 3) = 248 := by
  sorry

end Proof

end r_value_when_n_is_3_l463_463737


namespace intercepts_sum_l463_463031

theorem intercepts_sum :
  (let x_intercept := (λ l : ℝ, ∃ y : ℝ, (20 * l + 16 * y - 40 = 0) ∧ (y = 0)) in
   let y_intercept := (λ l : ℝ, ∃ x : ℝ, (20 * x + 16 * l - 64 = 0) ∧ (x = 0)) in
     (∃ x, x_intercept x ∧ x = 2) ∧ ∃ y, y_intercept y ∧ y = 4 → 
     ∃ sum, sum = (2 + 4)) :=
by
  sorry

end intercepts_sum_l463_463031


namespace second_discount_percentage_l463_463873

theorem second_discount_percentage (listed_price : ℝ) (first_discount : ℝ) (final_sale_price : ℝ) :
  listed_price = 350 → first_discount = 0.20 → final_sale_price = 266 →
  let first_discount_amount := first_discount * listed_price in
  let price_after_first_discount := listed_price - first_discount_amount in
  let second_discount_amount := price_after_first_discount - final_sale_price in
  let second_discount_percentage := (second_discount_amount / price_after_first_discount) * 100 in
  second_discount_percentage = 5 :=
by
  intros h_listed h_first h_final;
  sorry

end second_discount_percentage_l463_463873


namespace sufficient_but_not_necessary_condition_l463_463477

noncomputable def perpendicular_condition (a : ℝ) : Prop :=
  let line1 := λ x y : ℝ, a * x + 2 * y + 3 * a = 0
  let line2 := λ x y : ℝ, (a + 1) * x - 3 * y + 4 = 0
  ∃ (m1 m2 : ℝ), (∀ x y : ℝ, line1 x y → y = m1 * x) ∧ (∀ x y : ℝ, line2 x y → y = m2 * x) ∧ (m1 * m2 = -1)

theorem sufficient_but_not_necessary_condition : perpendicular_condition 2 ∧ ∃ a : ℝ, a ≠ 2 ∧ perpendicular_condition a :=
by sorry

end sufficient_but_not_necessary_condition_l463_463477


namespace solution_set_of_inequality_l463_463427

theorem solution_set_of_inequality :
  {x : ℝ | abs (x^2 - 5 * x + 6) < x^2 - 4} = { x : ℝ | x > 2 } :=
sorry

end solution_set_of_inequality_l463_463427


namespace part_a_l463_463724

theorem part_a (G : Type) [Group G] (hG : Fintype G) (n : ℕ) (hGn : Fintype.card G = n) 
  (p : ℕ) (hp : Nat.Prime p) (hpn : p ∣ n) (hlp : ∀ q, Nat.Prime q → q ∣ n → q ≤ p) :
  (∃ f : G →* G, True) → (Fintype.card (G →* G) ≤ Nat.root p (n ^ n)) :=
sorry

end part_a_l463_463724


namespace probability_of_winning_position_l463_463521

theorem probability_of_winning_position : 
    (let total_positions := Nat.choose 9 3,
         winning_positions := 8 in
       (winning_positions : ℚ) / total_positions = 2 / 21) := 
by
  sorry

end probability_of_winning_position_l463_463521


namespace find_a_in_set_A_l463_463221

theorem find_a_in_set_A :
  ∃ (a : ℚ), -3 ∈ ({a - 2, 2 * a ^ 2 + 5 * a, 12} : set ℚ) ∧ ({a | -3 ∈ ({a - 2, 2 * a ^ 2 + 5 * a, 12} : set ℚ)} = {-3 / 2}) := 
sorry

end find_a_in_set_A_l463_463221


namespace go_out_to_sea_is_optimal_l463_463087

def profit_if_good_weather := 6000
def loss_if_bad_weather := 8000
def loss_if_stay := 1000
def p_good_weather := 0.6
def p_bad_weather := 0.4

def expected_profit_go_out_to_sea := profit_if_good_weather * p_good_weather - loss_if_bad_weather * p_bad_weather
def expected_profit_stay := -loss_if_stay

theorem go_out_to_sea_is_optimal : expected_profit_go_out_to_sea > expected_profit_stay -> decision == "go out to sea" := 
by 
  sorry

end go_out_to_sea_is_optimal_l463_463087


namespace locus_of_P_l463_463616

variables {A B C D E F P M N : Type*}

-- Necessary background geometry definitions and assumptions
-- An acute triangle with vertex A being an acute angle
axiom acute_triangle_ABC : ∀ (A B C : Type*), ∡A < 90

-- D is a moving point on side BC
axiom D_on_BC (B C : Type*) : ∀ (D : Type*), (D ∈ line_segment B C)

-- Perpendiculars from D to AB and AC, intersecting at E and F respectively
axiom perp_from_D_to_AB_AC (D A B C E F : Type*):
  (perpendicular_to D A B ∧ perpendicular_to D A C) → 
  (intersect_at E ∧ intersect_at F)

-- P is the orthocenter of triangle AEF
axiom orthocenter_AEF (A E F P : Type*): is_orthocenter (triangle A E F) P

-- The statement to prove: The locus of P as D moves is the segment MN
theorem locus_of_P (A B C D E F P M N : Type*) : 
  ∀ D (H1: A ∡A (acute_triangle_ABC A B C)) 
      (H2: (D_on_BC B C D)) 
      (H3: perp_from_D_to_AB_AC D A B C E F) 
      (H4: orthocenter_AEF A E F P), 
    (locus P) = (line_segment M N) := 
sorry

end locus_of_P_l463_463616


namespace line_intersects_y_axis_at_point_l463_463499

def line_intersects_y_axis (x1 y1 x2 y2 : ℚ) : Prop :=
  ∃ c : ℚ, ∀ x : ℚ, y1 + (y2 - y1) / (x2 - x1) * (x - x1) = (y2 - y1) / (x2 - x1) * x + c

theorem line_intersects_y_axis_at_point :
  line_intersects_y_axis 3 21 (-9) (-6) :=
  sorry

end line_intersects_y_axis_at_point_l463_463499


namespace min_product_xyz_l463_463620

-- Definition of the problem statement in Lean
theorem min_product_xyz 
  (A B C D E F P R Q S : Type)
  (hABC : EquilateralTriangle ABC 4)
  (hD : OnSegment D B C)
  (hE : OnSegment E C A)
  (hF : OnSegment F A B)
  (hAE : |AE| = 1)
  (hBF : |BF| = 1)
  (hCD : |CD| = 1)
  (hRQS : Connect AD BE CF R QS)
  (hP : MovesInsideOrOn P RQS)
  (x y z : ℝ)
  (hx : x = Distance P SideA)
  (hy : y = Distance P SideB)
  (hz : z = Distance P SideC) :
  (∃ u ∈ {R, Q, S}, xyz = min_xyz) 
    ∧ (min_xyz = 648 / 2197 * sqrt 3)
  := by
  sorry

end min_product_xyz_l463_463620


namespace correctness_of_statements_l463_463866

theorem correctness_of_statements :
  let S := {1, 3/2, 1.5, -0.5, 0.5} in
  (|S| = 4) ∧
  (¬ ∀ (f : ℝ → ℝ), (f(0) = 0 → odd f)) ∧
  (∀ (f : ℝ → ℝ), (f(1) > f(2) → ¬ increasing ℝ f)) ∧
  (¬ ∀ (f : ℝ → ℝ) (a b : ℝ), (a < b ∧ f(a) * f(b) < 0 → ∃ c ∈ (a, b), f(c) = 0)) ->
  nat.find (λ n, (|S| = 4) ∧
    (¬ ∀ (f : ℝ → ℝ), (f(0) = 0 → odd f)) ∧
    (∀ (f : ℝ → ℝ), (f(1) > f(2) → ¬ increasing ℝ f)) ∧
    (¬ ∀ (f : ℝ → ℝ) (a b : ℝ), (a < b ∧ f(a) * f(b) < 0 → ∃ c ∈ (a, b), f(c) = 0)))
  = 1 := by
  sorry

end correctness_of_statements_l463_463866


namespace card_area_l463_463383

theorem card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_after_shortening : (length - 1) * width = 24 ∨ length * (width - 1) = 24) :
  length * (width - 1) = 18 :=
by
  sorry

end card_area_l463_463383


namespace tic_tac_toe_probability_l463_463537

theorem tic_tac_toe_probability :
  let total_positions := Nat.choose 9 3,
      winning_positions := 8 in
  (winning_positions / total_positions : ℚ) = 2 / 21 :=
by
  sorry

end tic_tac_toe_probability_l463_463537


namespace sum_of_possible_values_of_c_l463_463211

theorem sum_of_possible_values_of_c : 
  (∑ c in {c | c ∈ (Set.range (λ n : ℕ, if (∃ k : ℕ, 49 - 12 * c = k^2) then n else 0)) ∧ c ≠ 0}) = 6 :=
by
  sorry

end sum_of_possible_values_of_c_l463_463211


namespace sqrt_expression_simplified_l463_463978

theorem sqrt_expression_simplified : sqrt 12 - sqrt 2 * (sqrt 8 - 3 * sqrt (1 / 2)) = 2 * sqrt 3 - 1 := 
by
  -- We use sorry here to skip the proof.
  sorry

end sqrt_expression_simplified_l463_463978


namespace solve_system_of_equations_l463_463054

-- Define the conditions and the final proof goal
noncomputable def system_of_equations (x y a : ℝ) :=
  log (x^2 + y^2) / log (real.sqrt 10) = 2 * real.log a + 2 * log (x^2 - y^2) / log 100 ∧
  x * y = a^2

-- Define the expected outcome
noncomputable def expected_solutions (x y a : ℝ) :=
  (x = a * real.sqrt (real.sqrt 2 + 1) ∨ x = -a * real.sqrt (real.sqrt 2 + 1)) ∧
  (y = a * real.sqrt (real.sqrt 2 - 1) ∨ y = -a * real.sqrt (real.sqrt 2 - 1))

-- Prove that given the conditions, the solution is as expected
theorem solve_system_of_equations (x y a : ℝ) (h : system_of_equations x y a) :
  expected_solutions x y a :=
  sorry

end solve_system_of_equations_l463_463054


namespace problem_l463_463259

def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 2 else 3^x

theorem problem : f (f (1/4)) = 1/9 :=
by
  sorry

end problem_l463_463259


namespace complex_point_in_second_quadrant_l463_463705

def point_quadrant : String := 
  let z := (2 : ℂ) / (1 - complex.I) - 2
  if z.re < 0 ∧ z.im > 0 then "second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "third quadrant"
  else if z.re > 0 ∧ z.im > 0 then "first quadrant"
  else "fourth quadrant"

theorem complex_point_in_second_quadrant : point_quadrant = "second quadrant" := by
  sorry

end complex_point_in_second_quadrant_l463_463705


namespace replace_last_e_is_o_l463_463495

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def character_shift (char : Char) (occurrence : ℕ) : Char :=
  let shift := triangular_number occurrence
  let alphabet_position := char.val - 'a'.val
  let new_position := (alphabet_position + shift) % 26
  Char.of_nat (new_position + 'a'.val)

theorem replace_last_e_is_o :
  let message := "Hello, can everyone check the event, please?"
  let occurrences := message.toList.filter (· = 'e').length
  character_shift 'e' occurrences = 'o' := by
sorry

end replace_last_e_is_o_l463_463495


namespace sum_of_roots_l463_463741

theorem sum_of_roots {
  f : ℝ → ℝ,
  f_inv : ℝ → ℝ,
  h_f : ∀ x, f x = 3 * x - 2,
  h_f_inv : ∀ x, f_inv x = (x + 2) / 3,
  h_eqn : ∀ x, f_inv x = f (x⁻²)
} : (∃ s : finset ℝ, (∑ x in s, x) = -8 ∧ ∀ x ∈ s, x^3 + 8 * x^2 - 9 = 0) :=
by {
  sorry
}

end sum_of_roots_l463_463741


namespace vector_perpendicular_of_equal_norms_l463_463917

variables (a b : EuclideanSpace ℝ (Fin 2))
open_locale real_inner_product_space

theorem vector_perpendicular_of_equal_norms (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ‖a - b‖ = ‖a + b‖) : inner a b = 0 :=
by {
  have h_eq := calc
    ‖a - b‖ ^ 2 = ‖a + b‖ ^ 2 : by rw [h]
    ... = inner (a + b) (a + b) : by rw [norm_sq_eq_inner]
    ... = inner a a + 2 * inner a b + inner b b : by rw [inner_add_add_self]
    ... = ‖a‖ ^ 2 + 2 * inner a b + ‖b‖ ^ 2 : by simp [norm_sq_eq_inner],
  have h_sq := calc
    ‖a - b‖ ^ 2 = inner (a - b) (a - b) : by rw [norm_sq_eq_inner]
    ... = inner a a - 2 * inner a b + inner b b : by rw [inner_sub_sub_self]
    ... = ‖a‖ ^ 2 - 2 * inner a b + ‖b‖ ^ 2 : by simp [norm_sq_eq_inner],
  linarith,
}


end vector_perpendicular_of_equal_norms_l463_463917


namespace solution_set_of_inequality_l463_463008

theorem solution_set_of_inequality (x m : ℝ) : 
  (x^2 - (2 * m + 1) * x + m^2 + m < 0) ↔ m < x ∧ x < m + 1 := 
by
  sorry

end solution_set_of_inequality_l463_463008


namespace find_angle_QRS_l463_463706

theorem find_angle_QRS
  (angle_PQS : ℝ)
  (angle_QRT : ℝ)
  (angle_QRS : ℝ)
  (external_angle_theorem : angle_PQS = angle_QRS + angle_QRT) :
  angle_QRS = 48 :=
by
  have h : angle_PQS = 124 := by sorry
  have h' : angle_QRT = 76 := by sorry
  have h'' : angle_QRS = 124 - 76 := by sorry
  exact h''

end find_angle_QRS_l463_463706


namespace sum_of_possible_values_of_c_l463_463210

theorem sum_of_possible_values_of_c : 
  (∑ c in {c | c ∈ (Set.range (λ n : ℕ, if (∃ k : ℕ, 49 - 12 * c = k^2) then n else 0)) ∧ c ≠ 0}) = 6 :=
by
  sorry

end sum_of_possible_values_of_c_l463_463210


namespace true_discount_proof_l463_463941

-- Define the necessary variables and conditions
variables (PV R T : ℝ)
-- The given condition with the true discount formula for double the time.
axiom condition_TD2 : (PV * R * (2 * T)) / 100 = 18.333333333333332

-- Define the true discount at the end of the initial time.
def true_discount_initial_time : ℝ := (PV * R * T) / 100

-- State the theorem we need to prove
theorem true_discount_proof : true_discount_initial_time PV R T = 9.166666666666666 :=
by
  -- Here we would provide the proof but skip it with sorry.
  sorry

end true_discount_proof_l463_463941


namespace james_passenger_count_l463_463715

theorem james_passenger_count :
  ∀ (total_vehicles trucks buses taxis motorbikes cars trucks_population buses_population taxis_population motorbikes_population cars_population : ℕ),
  total_vehicles = 52 →
  trucks = 12 →
  buses = 2 →
  taxis = 2 * buses →
  motorbikes = total_vehicles - (trucks + buses + taxis + cars) →
  cars = 30 →
  trucks_population = 2 →
  buses_population = 15 →
  taxis_population = 2 →
  motorbikes_population = 1 →
  cars_population = 3 →
  (trucks * trucks_population + buses * buses_population + taxis * taxis_population +
   motorbikes * motorbikes_population + cars * cars_population) = 156 := 
by
  -- Placeholder for the proof
  sorry

end james_passenger_count_l463_463715


namespace min_distance_to_line_l463_463687

theorem min_distance_to_line (m n : ℝ) (h : 4 * m + 3 * n = 10)
  : m^2 + n^2 ≥ 4 :=
sorry

end min_distance_to_line_l463_463687


namespace arun_deepak_age_ratio_l463_463972

-- Define the current age of Arun based on the condition that after 6 years he will be 26 years old
def Arun_current_age : ℕ := 26 - 6

-- Define Deepak's current age based on the given condition
def Deepak_current_age : ℕ := 15

-- The present ratio between Arun's age and Deepak's age
theorem arun_deepak_age_ratio : Arun_current_age / Nat.gcd Arun_current_age Deepak_current_age = (4 : ℕ) ∧ Deepak_current_age / Nat.gcd Arun_current_age Deepak_current_age = (3 : ℕ) := 
by
  -- Proof omitted
  sorry

end arun_deepak_age_ratio_l463_463972


namespace composite_ratio_l463_463985

def composite_numbers : list ℕ := [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24]

theorem composite_ratio :
  (∏ x in composite_numbers.take 7, x) / (∏ x in composite_numbers.drop 7, x) = 1 / 110 :=
by
  sorry

end composite_ratio_l463_463985


namespace gcd_1729_867_l463_463454

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by 
  sorry

end gcd_1729_867_l463_463454


namespace find_b_l463_463862

theorem find_b (a b : ℤ) (h1 : 0 ≤ a) (h2 : a < 2^2008) (h3 : 0 ≤ b) (h4 : b < 8) (h5 : 7 * (a + 2^2008 * b) % 2^2011 = 1) :
  b = 3 :=
sorry

end find_b_l463_463862


namespace sqrt_81_div_3_eq_3_l463_463473

theorem sqrt_81_div_3_eq_3 : real.sqrt 81 / 3 = 3 := by
  sorry

end sqrt_81_div_3_eq_3_l463_463473


namespace incorrect_statement_A_l463_463702

-- Definitions based on conditions
variable {α : Plane}
variable {m n : Line}

-- Option A conditions
def m_parallel_alpha (m : Line) (α : Plane) : Prop := m ∥ α
def m_not_parallel_n (m n : Line) : Prop := ¬ (m ∥ n)

-- Option A conclusion to disprove
def statement_A (m n : Line) (α : Plane) [m_parallel_alpha m α] [m_not_parallel_n m n] : Prop :=
  ¬(n ∥ α)

theorem incorrect_statement_A (α : Plane) (m n : Line) (h1 : m_parallel_alpha m α) (h2 : m_not_parallel_n m n) : ¬(statement_A m n α) := by
  sorry

end incorrect_statement_A_l463_463702


namespace sum_alternating_binomial_coeffs_l463_463564

open Complex

theorem sum_alternating_binomial_coeffs :
  (∑ k in Finset.range 50, (-1) ^ k * Nat.choose 100 (2 * k + 1)) = 0 := 
sorry

end sum_alternating_binomial_coeffs_l463_463564


namespace Tim_doctors_visit_cost_l463_463885

variable {cost_doc_visit : ℤ} -- Cost of Tim's doctor's visit

-- Conditions
constant cost_cat_visit : ℤ := 120                     -- Cat's visit cost
constant coverage_cat_insurance : ℤ := 60               -- Pet insurance coverage for cat's visit
constant total_payment : ℤ := 135                       -- Tim's total payment
constant coverage_doc_insurance : ℚ := 0.75             -- Insurance coverage for doctor's visit

-- Definition based on the conditions
def paid_cat_visit : ℤ := cost_cat_visit - coverage_cat_insurance -- Amount Tim paid for cat's visit
def paid_doc_visit (X : ℚ) : ℚ := (1 - coverage_doc_insurance) * X -- Amount Tim paid for doctor's visit out of pocket

-- Problem statement
theorem Tim_doctors_visit_cost : 
  ∃ (X : ℤ), (paid_doc_visit X + paid_cat_visit) = total_payment ∧ X = 300 :=
sorry

end Tim_doctors_visit_cost_l463_463885


namespace sum_of_possible_values_of_c_l463_463207

theorem sum_of_possible_values_of_c : 
  (∑ c in {c | c ∈ (Set.range (λ n : ℕ, if (∃ k : ℕ, 49 - 12 * c = k^2) then n else 0)) ∧ c ≠ 0}) = 6 :=
by
  sorry

end sum_of_possible_values_of_c_l463_463207


namespace green_balloons_correct_l463_463550

-- Defining the quantities
def total_balloons : ℕ := 67
def red_balloons : ℕ := 29
def blue_balloons : ℕ := 21

-- Calculating the green balloons
def green_balloons : ℕ := total_balloons - red_balloons - blue_balloons

-- The theorem we want to prove
theorem green_balloons_correct : green_balloons = 17 :=
by
  -- proof goes here
  sorry

end green_balloons_correct_l463_463550


namespace ratio_of_Lev_to_Akeno_l463_463547

theorem ratio_of_Lev_to_Akeno (L : ℤ) (A : ℤ) (Ambrocio : ℤ) :
  A = 2985 ∧ Ambrocio = L - 177 ∧ A = L + Ambrocio + 1172 → L / A = 1 / 3 :=
by
  intro h
  sorry

end ratio_of_Lev_to_Akeno_l463_463547


namespace bottom_level_legos_l463_463129

theorem bottom_level_legos
  (x : ℕ)
  (h : x^2 + (x - 1)^2 + (x - 2)^2 = 110) :
  x = 7 :=
by {
  sorry
}

end bottom_level_legos_l463_463129


namespace volume_of_region_l463_463565

noncomputable def g (x y z : ℝ) := 2 * |2 * x + y + z| + 2 * |x + 2 * y - z| + |x - y + 2 * z| + |y - 2 * x + z|

theorem volume_of_region :
  let region := {p : ℝ × ℝ × ℝ | g p.1 p.2 p.3 ≤ 10} in
  volume (region) = 125 / 6 :=
sorry

end volume_of_region_l463_463565


namespace iron_heating_time_l463_463009

-- Define the conditions as constants
def ironHeatingRate : ℝ := 9 -- degrees Celsius per 20 seconds
def ironCoolingRate : ℝ := 15 -- degrees Celsius per 30 seconds
def coolingTime : ℝ := 180 -- seconds

-- Define the theorem to prove the heating back time
theorem iron_heating_time :
  (coolingTime / 30) * ironCoolingRate = 90 →
  (90 / ironHeatingRate) * 20 = 200 :=
by
  sorry

end iron_heating_time_l463_463009


namespace calculate_percentage_gain_l463_463093

-- Define the parameters
def cost_price (num_bowls : ℕ) (price_per_bowl : ℕ) : ℕ := num_bowls * price_per_bowl

def selling_price (num_bowls : ℕ) (price_per_bowl : ℕ) : ℕ := num_bowls * price_per_bowl

def total_selling_price (sp1 sp2 : ℕ) : ℕ := sp1 + sp2

def gain (sp cp : ℕ) : ℕ := sp - cp

def percentage_gain (gain cp : ℕ) : ℚ :=
  (gain.to_rat / cp.to_rat) * 100

-- The main theorem statement
theorem calculate_percentage_gain :
  let cp := cost_price 300 20 in
  let sp1 := selling_price 200 25 in
  let sp2 := selling_price 80 30 in
  let total_sp := total_selling_price sp1 sp2 in
  let g := gain total_sp cp in
  percentage_gain g cp = 23.33 :=
by
  sorry

end calculate_percentage_gain_l463_463093


namespace area_EFGC_is_20_l463_463711

theorem area_EFGC_is_20 (AB CD AD BC : ℝ) (E F G : ℝ × ℝ)
  (hAB_parallel_CD : AB ∥ CD)
  (hAB_eq_2CD : AB = 2 * CD)
  (hE_mid_AD : E = midpoint AD)
  (hF_mid_BC : F = midpoint BC)
  (hG_mid_AB : G = midpoint AB)
  (hArea_ABCD : (1/2) * (AB + CD) * height = 48) :
  let EF := (AB + CD) / 2 in
  let height_half : ℝ := height / 2 in
  let area_EFGC := (1/2) * (EF + CD) * height_half in
  area_EFGC = 20 := sorry

end area_EFGC_is_20_l463_463711


namespace low_card_value_is_one_l463_463318

-- Definitions and setting up the conditions
def num_high_cards : ℕ := 26
def num_low_cards : ℕ := 26
def high_card_points : ℕ := 2
def draw_scenarios : ℕ := 4

-- The point value of a low card L
noncomputable def low_card_points : ℕ :=
  if num_high_cards = 26 ∧ num_low_cards = 26 ∧ high_card_points = 2
     ∧ draw_scenarios = 4
  then 1 else 0 

theorem low_card_value_is_one :
  low_card_points = 1 :=
by
  sorry

end low_card_value_is_one_l463_463318


namespace pure_alcohol_addition_l463_463468

theorem pure_alcohol_addition (x : ℝ) : 
  20 + x = 0.3 * (100 + x) → x ≈ 14.29 :=
by
  sorry

end pure_alcohol_addition_l463_463468


namespace cakes_remain_l463_463127

def initial_cakes := 110
def sold_cakes := 75
def new_cakes := 76

theorem cakes_remain : (initial_cakes - sold_cakes) + new_cakes = 111 :=
by
  sorry

end cakes_remain_l463_463127


namespace distance_center_of_sphere_to_plane_of_triangle_l463_463430

/-
  Prove that the distance from the center of a sphere to the plane of a right-angled triangle ABC,
  where the right-angle sides are 6 and 8, and the sphere has radius 13, is 12.
-/
theorem distance_center_of_sphere_to_plane_of_triangle 
  (A B C : Point) -- Points corresponding to vertices of triangle ABC
  (r : ℝ) (h₁ h₂ h₃ : ℝ) (R : ℝ)
  (h1 : right_angled_triangle A B C)
  (h2 : distance A B = 6)
  (h3 : distance B C = 8)
  (h4 : distance A C = 10) -- this is derived from the Pythagorean theorem, distance A C = sqrt(6^2 + 8^2)
  (h5 : is_surface_of_sphere A B C R)
  (hR : R = 13) :
  distance_from_center_to_plane r = 12 :=
begin
  sorry
end

end distance_center_of_sphere_to_plane_of_triangle_l463_463430


namespace comb_sum_identity_l463_463986

open Nat

theorem comb_sum_identity (n : ℕ) (m : ℕ) (k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ m) (h3 : m ≤ n) :
  (∑ i in finset.range (k+1), (binom k i) * (binom n (m - i))) = binom (n + k) m :=
by
  sorry

end comb_sum_identity_l463_463986


namespace clarence_oranges_after_giving_l463_463134

def initial_oranges : ℝ := 5.0
def oranges_given : ℝ := 3.0

theorem clarence_oranges_after_giving : (initial_oranges - oranges_given) = 2.0 :=
by
  sorry

end clarence_oranges_after_giving_l463_463134


namespace apples_total_l463_463681

theorem apples_total (apples_per_person : ℝ) (number_of_people : ℝ) (h_apples : apples_per_person = 15.0) (h_people : number_of_people = 3.0) : 
  apples_per_person * number_of_people = 45.0 := by
  sorry

end apples_total_l463_463681


namespace sum_possible_quantities_l463_463036

theorem sum_possible_quantities : 
  let C := λ n, n ∈ (List.range 100).filter (λ n, n % 6 = 2 ∧ n % 8 = 5) in
  (C.summation id) = 176 :=
by
  sorry

end sum_possible_quantities_l463_463036


namespace arithmetic_sequence_max_n_l463_463233

theorem arithmetic_sequence_max_n
  (a : ℕ → ℝ) -- arithmetic sequence
  (d : ℝ)     -- common difference of the arithmetic sequence
  (h0 : ∀ n, a (n+1) - a n = d) -- definition of arithmetic sequence
  (h1 : a 11 > 0) 
  (h2 : a 12 / a 11 < -1) 
  (h3 : ∃ n, ∀ m < n, (finset.range m).sum a < 0 ∧ (finset.range n).sum a > 0)  -- maximum value property
  : ∃ n, n = 21 ∧ (finset.range n).sum a > 0 :=
sorry

end arithmetic_sequence_max_n_l463_463233


namespace blue_to_red_face_area_ratio_l463_463851

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l463_463851


namespace bridge_length_correct_l463_463964

/-
  A train that is 485 meters long is running at a speed of 45 km/hour.
  In what time will it pass a bridge of a certain length if it takes 50 seconds to pass the bridge?
  What is the length of the bridge?
-/

-- Define the conditions
def train_length : ℝ := 485
def train_speed_kmh : ℝ := 45
def passing_time_seconds : ℝ := 50

-- Convert speed from km/h to m/s
def train_speed_mps : ℝ := train_speed_kmh * 1000 / 3600

-- Define the distance covered in the given passing time
def total_distance_covered : ℝ := train_speed_mps * passing_time_seconds

def bridge_length : ℝ := total_distance_covered - train_length

-- The proof problem
theorem bridge_length_correct : bridge_length = 140 := by
  sorry

end bridge_length_correct_l463_463964


namespace exists_tangent_circles_l463_463145

variable {A B C M L : Point}
variable {Δ : Triangle}
variable {r1 r2 m n : ℝ}

def tangent_circles_of_triangle (Δ : Triangle) (r1 r2 : ℝ) (m n : ℝ) : Prop :=
  let AC := Δ.side A C
  let BC := Δ.side B C
  ∃ (M L : Point), 
    M ∈ (AC ⊥ A) ∧ L ∈ (BC ⊥ B) ∧
    let C1 := circle M r1
    let C2 := circle L r2
    (C1.radius / C2.radius = m / n) ∧
    (tangent C1 C2)

theorem exists_tangent_circles (Δ : Triangle) (r1 r2 : ℝ) (m n : ℝ) (h : r1 / r2 = m / n)
  (h1 : ∃ M, ∀ (x : Point), x ∈ (Δ.side AC) → (⊥ M x))
  (h2 : ∃ L, ∀ (x : Point), x ∈ (Δ.side BC) → (⊥ L x)) : 
  tangent_circles_of_triangle Δ r1 r2 m n :=
sorry

end exists_tangent_circles_l463_463145


namespace largest_possible_number_l463_463387

theorem largest_possible_number (ns : List ℕ) (h_sum : ns.sum = 2019)
  (h_no40 : ∀ n ∈ ns, n ≠ 40)
  (h_no_consec_sum_40 : ∀ (sublists : List (List ℕ)), sublists.sum ≠ 40) :
  ns.length ≤ 1019 :=
sorry

end largest_possible_number_l463_463387


namespace trapezoid_area_is_correct_l463_463552

noncomputable def isosceles_trapezoid_area : ℝ :=
  let a : ℝ := 12
  let b : ℝ := 24 - 12 * Real.sqrt 2
  let h : ℝ := 6 * Real.sqrt 2
  (24 + b) / 2 * h

theorem trapezoid_area_is_correct :
  let a := 12
  let b := 24 - 12 * Real.sqrt 2
  let h := 6 * Real.sqrt 2
  (24 + b) / 2 * h = 144 * Real.sqrt 2 - 72 :=
by
  sorry

end trapezoid_area_is_correct_l463_463552


namespace fill_tanker_time_l463_463924

/-- Given that pipe A can fill the tanker in 60 minutes and pipe B can fill the tanker in 40 minutes,
    prove that the time T to fill the tanker if pipe B is used for half the time and both pipes 
    A and B are used together for the other half is equal to 30 minutes. -/
theorem fill_tanker_time (T : ℝ) (hA : ∀ (a : ℝ), a = 1/60) (hB : ∀ (b : ℝ), b = 1/40) :
  (T / 2) * (1 / 40) + (T / 2) * (1 / 24) = 1 → T = 30 :=
by
  sorry

end fill_tanker_time_l463_463924


namespace eccentricity_range_l463_463353

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) (P : ℝ × ℝ) (hP : (P.1 / a)^2 + (P.2 / b)^2 = 1) : ℝ :=
  real.sqrt (a^2 - b^2) / a

theorem eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h_cond : ∀ (P : ℝ × ℝ), (P.1 / a)^2 + (P.2 / b)^2 = 1 → dist P (0, b) ≤ 2 * b) :
  0 < eccentricity_of_ellipse a b h1 h2 ∧ eccentricity_of_ellipse a b h1 h2 ≤ real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l463_463353


namespace cube_surface_area_with_same_volume_as_prism_l463_463099

theorem cube_surface_area_with_same_volume_as_prism :
  let prism_volume := 10 * 3 * 30 in
  let s := (prism_volume:ℝ)^(1/3) in
  let surface_area := 6 * s^2 in
  surface_area = 6 * (900:ℝ)^(2/3) :=
by
  sorry

end cube_surface_area_with_same_volume_as_prism_l463_463099


namespace sum_of_valid_c_values_l463_463192

theorem sum_of_valid_c_values:
  let quadratic_rational_roots := λ (c : ℕ), (∃ d : ℕ, 49 - 12 * c = d * d) ∧ (c > 0)
  in ∑ c in finset.filter quadratic_rational_roots (finset.range 5), c = 6 :=
by
  -- The proof will go here
  sorry

end sum_of_valid_c_values_l463_463192


namespace find_certain_number_l463_463683

theorem find_certain_number (a : ℤ) (certain_number : ℤ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * certain_number) : certain_number = 49 := 
sorry

end find_certain_number_l463_463683


namespace fraction_expression_l463_463976

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 :=
by
  sorry

end fraction_expression_l463_463976


namespace donation_possible_values_l463_463997

-- Define the conditions
def donations (donations_list : List ℕ) : Prop :=
  let n := donations_list.length in
  n = 5 ∧
  (donations_list.sum / n = 560) ∧
  (∀ d ∈ donations_list, d % 100 = 0) ∧
  (List.minimum? donations_list = some 200) ∧
  (List.maximum? donations_list = some 800) ∧
  (donations_list.count 800 = 1) ∧
  (donations_list.count 600 = 1) ∧
  (List.nth_le (List.sort (≤) donations_list) 2 (by simp [n]) = 600)

-- Define the proof statement
theorem donation_possible_values :
  ∃ a b c d e: ℕ,
    donations [a, b, c, d, e] ∧
    ([a, b, c, d, e] = List.cons 200 (List.cons 800 (List.cons 600 [500, 700])) ∨ 
     [a, b, c, d, e] = List.cons 200 (List.cons 800 (List.cons 600 [600, 600]))) :=
sorry

end donation_possible_values_l463_463997


namespace possible_sums_of_products_neg11_l463_463420

theorem possible_sums_of_products_neg11 (a b c : ℤ) (h : a * b * c = -11) :
  a + b + c = -9 ∨ a + b + c = 11 ∨ a + b + c = 13 :=
sorry

end possible_sums_of_products_neg11_l463_463420


namespace expense_and_income_calculations_l463_463071

def alexander_salary : ℕ := 125000
def natalia_salary : ℕ := 61000
def utilities_transport_household : ℕ := 17000
def loan_repayment : ℕ := 15000
def theater_cost : ℕ := 5000
def cinema_cost_per_person : ℕ := 1000
def savings_crimea : ℕ := 20000
def dining_weekday_cost : ℕ := 1500
def dining_weekend_cost : ℕ := 3000
def weekdays : ℕ := 20
def weekends : ℕ := 10
def phone_A_cost : ℕ := 57000
def phone_B_cost : ℕ := 37000

def total_expenses : ℕ :=
  utilities_transport_household +
  loan_repayment +
  theater_cost + 2 * cinema_cost_per_person +
  savings_crimea +
  weekdays * dining_weekday_cost +
  weekends * dining_weekend_cost

def net_income : ℕ :=
  alexander_salary + natalia_salary

def can_buy_phones : Prop :=
  net_income - total_expenses < phone_A_cost + phone_B_cost

theorem expense_and_income_calculations :
  total_expenses = 119000 ∧
  net_income = 186000 ∧
  can_buy_phones :=
by
  sorry

end expense_and_income_calculations_l463_463071


namespace b_arithmetic_sum_first_n_terms_range_of_m_l463_463274

-- Condition 1: Definition of sequence a_n
def a (n : ℕ) : ℝ := (1 / 4) ^ n

-- Condition 2: Definition of sequence b_n based on a_n
def b (n : ℕ) : ℝ := 3 * Real.log (a n) / Real.log (1/4) - 2   -- rewritten using log properties

-- Condition 3: Definition of sequence c_n that is the product of a_n and b_n
def c (n : ℕ) : ℝ := a n * b n

-- Proof statement 1: Sequence b_n is arithmetic with first term 1 and common difference 3
theorem b_arithmetic : arithmetic_sequence (b : ℕ → ℝ) :=
sorry

-- Proof statement 2: Sum of the first n terms of c_n
def S (n : ℕ) : ℝ := (finset.range n).sum (fun i => c (i + 1))
theorem sum_first_n_terms (n : ℕ) : S n = (2 / 3) - (3 * (n : ℝ) + 2) / 3 * (1 / 4) ^ n :=
sorry

-- Proof statement 3: Finding range of m
theorem range_of_m (m : ℝ) : (∀ n : ℕ, c n ≤ (1 / 4) * m^2 + m - 1) ↔ (m ≥ 1 ∨ m ≤ -5) :=
sorry

end b_arithmetic_sum_first_n_terms_range_of_m_l463_463274


namespace determine_m_l463_463735

-- Define f and g according to the given conditions
def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

-- Define the value of x
def x := 5

-- State the main theorem we need to prove
theorem determine_m 
  (h : 3 * f x m = 2 * g x m) : m = 10 / 7 :=
by
  -- Proof is omitted
  sorry

end determine_m_l463_463735


namespace treadmill_time_saved_l463_463758

theorem treadmill_time_saved:
  let monday_speed := 6
  let tuesday_speed := 4
  let wednesday_speed := 5
  let thursday_speed := 6
  let friday_speed := 3
  let distance := 3 
  let daily_times : List ℚ := 
    [distance/monday_speed, distance/tuesday_speed, distance/wednesday_speed, distance/thursday_speed, distance/friday_speed]
  let total_time := (daily_times.map (λ t => t)).sum
  let total_distance := 5 * distance 
  let uniform_speed := 5 
  let uniform_time := total_distance / uniform_speed 
  let time_difference := total_time - uniform_time 
  let time_in_minutes := time_difference * 60 
  time_in_minutes = 21 := 
by 
  sorry

end treadmill_time_saved_l463_463758


namespace polygon_diagonals_l463_463098

theorem polygon_diagonals (n : ℕ) (h : 20 = n) : (n * (n - 3)) / 2 = 170 :=
by
  sorry

end polygon_diagonals_l463_463098


namespace smallest_n_for_terminating_decimal_l463_463910

theorem smallest_n_for_terminating_decimal : 
  ∃ n : ℕ, (0 < n) ∧ (∃ k m : ℕ, (n + 70 = 2 ^ k * 5 ^ m) ∧ k = 0 ∨ k = 1) ∧ n = 55 :=
by sorry

end smallest_n_for_terminating_decimal_l463_463910


namespace problem1_condition1_problem2_condition_l463_463315

noncomputable theory

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions for problem 1
theorem problem1_condition1 (h1 : c = real.sqrt 7) (h2 : C = real.pi / 3) (h3 : a^2 + b^2 - a * b = 7) (h4 : 2 * a = 3 * b) :
  a = 3 ∧ b = 2 :=
sorry

-- Conditions for problem 2
theorem problem2_condition (h5 : cos B = 3 * real.sqrt 10 / 10) :
  sin (2 * A) = (3 - 4 * real.sqrt 3) / 10 :=
sorry

end problem1_condition1_problem2_condition_l463_463315


namespace coefficient_x8_expansion_l463_463166

noncomputable def polynomial1 := (2 : ℤ) * (X ^ 6) - 3 * (X ^ 5) + 4 * (X ^ 4) - 7 * (X ^ 3) + 2 * X - 5
noncomputable def polynomial2 := (3 : ℤ) * (X ^ 5) - 3 * (X ^ 3) + 2 * (X ^ 2) + 3 * X - 8

theorem coefficient_x8_expansion :
  (coeff (polynomial1 * polynomial2) 8) = -8 :=
by
  sorry

end coefficient_x8_expansion_l463_463166


namespace common_difference_arithmetic_sequence_l463_463232

theorem common_difference_arithmetic_sequence (d : ℝ) :
  (∀ (n : ℝ) (a_1 : ℝ), a_1 = 9 ∧
  (∃ a₄ a₈ : ℝ, a₄ = a_1 + 3 * d ∧ a₈ = a_1 + 7 * d ∧ a₄ = (a_1 * a₈)^(1/2)) →
  d = 1) :=
sorry

end common_difference_arithmetic_sequence_l463_463232


namespace gcd_of_three_l463_463181

theorem gcd_of_three (a b c : ℕ) (h₁ : a = 9242) (h₂ : b = 13863) (h₃ : c = 34657) :
  Nat.gcd (Nat.gcd a b) c = 1 :=
by
  sorry

end gcd_of_three_l463_463181


namespace _l463_463222

noncomputable def solveTrigProblem (α : ℝ) : Prop :=
  sin (α + π / 4) = sqrt 2 / 4 ∧ α ∈ Ioo (π / 2) π → sin α = (sqrt 7 + 1) / 4

-- Example theorem using the defined problem
example (α : ℝ) : solveTrigProblem α := by
  sorry

end _l463_463222


namespace neg_prop_p_equiv_l463_463865

open Classical

variable (x : ℝ)
def prop_p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 0

theorem neg_prop_p_equiv : ¬ prop_p ↔ ∃ x : ℝ, x^2 + 1 < 0 := by
  sorry

end neg_prop_p_equiv_l463_463865


namespace sum_of_valid_c_l463_463187

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_valid_c :
  (∑ c in Finset.filter (λ c, ∃ k, k * k = 49 - 12 * c) (Finset.range 5), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463187


namespace blue_face_area_factor_l463_463795

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l463_463795


namespace total_wire_length_l463_463501

theorem total_wire_length
  (A B C D E : ℕ)
  (hA : A = 16)
  (h_ratio : 4 * A = 5 * B ∧ 4 * A = 7 * C ∧ 4 * A = 3 * D ∧ 4 * A = 2 * E)
  (hC : C = B + 8) :
  (A + B + C + D + E) = 84 := 
sorry

end total_wire_length_l463_463501


namespace arithmetic_sequence_sum_property_l463_463329

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)  -- sequence terms are real numbers
  (d : ℝ)      -- common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum_condition : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 :=
sorry

end arithmetic_sequence_sum_property_l463_463329


namespace slopes_product_constant_line_PQ_fixed_point_l463_463509

open Set

-- Define the parabola and the line as sets of points
def parabola : Set (ℝ × ℝ) := {p | p.snd = p.fst ^ 2}
def line_y_eq_neg1 : Set (ℝ × ℝ) := {q | q.snd = -1}

-- Define the points A, P, and Q
def point_A (a : ℝ) : ℝ × ℝ := (a, -1)

-- Define the problem statement for the slopes of tangents
theorem slopes_product_constant (a : ℝ) (k_1 k_2 : ℝ) :
  (∀ (P Q : ℝ × ℝ), 
    P ∈ parabola ∧ Q ∈ parabola →
    line_y_eq_neg1 (point_A a) ∧ 
    tangent_at (point_A a) P k_1 ∧
    tangent_at (point_A a) Q k_2)
  → k_1 * k_2 = constant :=
sorry

-- Define the problem statement for the line PQ passing through a fixed point
theorem line_PQ_fixed_point (a : ℝ) :
  (∀ (P Q : ℝ × ℝ), 
    P ∈ parabola ∧ Q ∈ parabola →
    line_y_eq_neg1 (point_A a) ∧ 
    tangent_at (point_A a) P k_1 ∧
    tangent_at (point_A a) Q k_2
    → line_passing_through P Q (0, 1) :=
sorry


end slopes_product_constant_line_PQ_fixed_point_l463_463509


namespace math_proof_problem_l463_463736

-- Definitions for Unimodal Function and Peak Point
def unimodal (f : ℝ → ℝ) (a b : ℝ) := 
  ∃ (hat_x : ℝ), a < hat_x ∧ hat_x < b ∧ 
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ hat_x → f x ≤ f y) ∧ 
  (∀ x y, hat_x ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y)

-- Problem translation for function f₁
def f1 : ℝ → ℝ := λ x, x - 2 * x^2
def f₁_unimodal : Prop := unimodal f1 0 1 ∧ ∃ hat_x, hat_x = 1/4

-- Problem translation for function f₂
def f2 : ℝ → ℝ := λ x, abs (Real.log (x + 0.5) / Real.log 2)
def f₂_not_unimodal : Prop := ¬ unimodal f2 0 1

-- Problem translation for finding range of a for a unimodal function
def cubic_function_unimodal (a : ℝ) : Prop := 
  a < 0 ∧ 
  unimodal (λ x, a*x^3 + x) 1 2 → 
  -1/3 < a ∧ a < -1/12

-- Problem translation for proving peak containing interval
def peak_containing_interval (f : ℝ → ℝ) (a b : ℝ) (m n : ℝ) :=
  unimodal f a b ∧ a < m ∧ m < n ∧ n < b ∧ f m ≥ f n → 
  ∃ hat_x, (a,n) = (a, hat_x)

-- Main theorem that combines all the problem statements
theorem math_proof_problem : f₁_unimodal ∧ f₂_not_unimodal ∧ ∃ a, cubic_function_unimodal a ∧ 
  ∀ f a b m n, peak_containing_interval f a b m n := sorry

end math_proof_problem_l463_463736


namespace find_13th_result_l463_463064

theorem find_13th_result (avg25 : ℕ) (avg12_first : ℕ) (avg12_last : ℕ)
  (h_avg25 : avg25 = 18) (h_avg12_first : avg12_first = 10) (h_avg12_last : avg12_last = 20) :
  ∃ r13 : ℕ, r13 = 90 := by
  sorry

end find_13th_result_l463_463064


namespace unique_cylinder_identical_l463_463970

noncomputable def unique_cylinder (V S : ℝ) : Prop :=
  ∀ r₁ r₂ h₁ h₂ : ℝ,
  (π * r₁^2 * h₁ = V ∧ 2 * π * r₁^2 + 2 * π * r₁ * h₁ = S) ∧
  (π * r₂^2 * h₂ = V ∧ 2 * π * r₂^2 + 2 * π * r₂ * h₂ = S) →
  (r₁ = r₂ ∧ h₁ = h₂)

-- This will state the main theorem
theorem unique_cylinder_identical (V S : ℝ) : unique_cylinder V S := 
  by
    sorry -- Proof goes here; it shows that cylinders with given V and S are identical.

end unique_cylinder_identical_l463_463970


namespace problem_statement_l463_463983

theorem problem_statement : (complex.exp (real.pi * 225 / 180 * complex.I)) ^ 18 = complex.I :=
by 
  -- Conversion helper
  have h1 : real.pi * 225 / 180 = ↑(real.pi / (180 / 225)) := sorry,

  -- DeMoivre's theorem applied
  have h2 : (complex.cos (225 * real.pi / 180) + complex.sin (225 * real.pi / 180) * complex.I) ^ 18 
           = complex.cos (18 * (225 * real.pi / 180)) +
             complex.sin (18 * (225 * real.pi / 180)) * complex.I := sorry,

  -- Compute the angle 18 * 225 degrees reduced by full rotations 360 degrees
  have h3 : 18 * (225 * real.pi / 180) = (4050 * real.pi / 180) = (90 * real.pi / 180) :=
    by norm_num,

  -- Simplify as \(\cos 90^\circ + i \sin 90^\circ = 0 + i = i\)
  have h4 : (complex.cos (90 * real.pi / 180) + complex.sin (90 * real.pi / 180) * complex.I) = complex.I := sorry,

  exact h4

end problem_statement_l463_463983


namespace worker_total_pay_l463_463108

def regular_rate : ℕ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def non_cellphone_surveys := total_surveys - cellphone_surveys
def higher_rate := regular_rate + (30 * regular_rate / 100)

def pay_non_cellphone_surveys := non_cellphone_surveys * regular_rate
def pay_cellphone_surveys := cellphone_surveys * higher_rate

def total_pay := pay_non_cellphone_surveys + pay_cellphone_surveys

theorem worker_total_pay : total_pay = 605 := by
  sorry

end worker_total_pay_l463_463108


namespace projection_of_b_in_direction_of_a_l463_463279

variables (a b : ℝ^3) -- assuming vectors in three-dimensional space

-- Conditions
def norm_a : ℝ := 2
def dot_a_b_minus_a : ℝ := -3

-- Projection function
def projection (a b : ℝ^3) : ℝ :=
  (a.dot b) / (Real.sqrt (a.dot a))

theorem projection_of_b_in_direction_of_a :
  Real.sqrt (a.dot a) = norm_a →
  a.dot (b - a) = dot_a_b_minus_a →
  projection a b = 1 / 2 :=
sorry

end projection_of_b_in_direction_of_a_l463_463279


namespace circle_sum_value_l463_463707

-- Define the problem
theorem circle_sum_value (a b x : ℕ) (h1 : a = 35) (h2 : b = 47) : x = a + b :=
by
  -- Given conditions
  have ha : a = 35 := h1
  have hb : b = 47 := h2
  -- Prove that the value of x is the sum of a and b
  have h_sum : x = a + b := sorry
  -- Assert the value of x is 82 based on given a and b
  exact h_sum

end circle_sum_value_l463_463707


namespace original_numerator_l463_463104

theorem original_numerator (n : ℕ) (hn : (n + 3) / (9 + 3) = 2 / 3) : n = 5 :=
by
  sorry

end original_numerator_l463_463104


namespace cubes_difference_l463_463628

theorem cubes_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 65) (h3 : a + b = 6) : a^3 - b^3 = 432.25 :=
by
  sorry

end cubes_difference_l463_463628


namespace probability_ratio_l463_463391

theorem probability_ratio (a b : ℕ) (h1 : a ≠ b) : 
  let total_ways := Nat.choose 60 5,
      p := 15 / total_ways,
      q := (Nat.choose 15 2) * (Nat.choose 4 3) * (Nat.choose 4 2) / total_ways
  in q / p = 168 :=
by
  let total_ways := Nat.choose 60 5,
  have p := 15 / total_ways,
  have q := (Nat.choose 15 2) * (Nat.choose 4 3) * (Nat.choose 4 2) / total_ways,
  have ratio := q / p,
  sorry

end probability_ratio_l463_463391


namespace minimum_distance_parabola_line_l463_463955

theorem minimum_distance_parabola_line : 
  ∀ (M : ℝ × ℝ), M.snd = M.fst^2 → ∃ d : ℝ, d = 3 * sqrt 2 / 8 ∧ ∀ y: ℝ, y = x - y - 1 = 0 → dist M (x - y - 1 = 0) = d :=
sorry

end minimum_distance_parabola_line_l463_463955


namespace red_blue_area_ratio_is_12_l463_463803

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l463_463803


namespace length_of_train_l463_463106

-- Definitions of the conditions
def speed_km_hr := 72
def speed_m_s := speed_km_hr * (1000 / 3600)  -- Convert speed from km/hr to m/s
def time_seconds := 12.399008079353651
def length_bridge := 138
def total_distance := speed_m_s * time_seconds -- Calculate the total distance covered by the train

-- Definition we are trying to prove
def length_train := total_distance - length_bridge

theorem length_of_train :
  length_train = 109.98016158707302 :=
by
  sorry

end length_of_train_l463_463106


namespace cuboid_edge_length_l463_463794

theorem cuboid_edge_length
  (x : ℝ)
  (h_surface_area : 2 * (4 * x + 24 + 6 * x) = 148) :
  x = 5 :=
by
  sorry

end cuboid_edge_length_l463_463794


namespace mark_money_left_l463_463757

theorem mark_money_left (initial_money : ℕ) (cost_book1 cost_book2 cost_book3 : ℕ) (n_book1 n_book2 n_book3 : ℕ) 
  (total_cost : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 85)
  (h2 : cost_book1 = 7)
  (h3 : n_book1 = 3)
  (h4 : cost_book2 = 5)
  (h5 : n_book2 = 4)
  (h6 : cost_book3 = 9)
  (h7 : n_book3 = 2)
  (h8 : total_cost = 21 + 20 + 18)
  (h9 : money_left = initial_money - total_cost):
  money_left = 26 := by
  sorry

end mark_money_left_l463_463757


namespace probability_one_head_in_three_flips_l463_463061

theorem probability_one_head_in_three_flips : 
  let outcomes := {outcome | outcome = ['H', 'H', 'H'] ∨ outcome = ['H', 'H', 'T'] ∨ outcome = ['H', 'T', 'H'] ∨ 
                              outcome = ['H', 'T', 'T'] ∨ outcome = ['T', 'H', 'H'] ∨ outcome = ['T', 'H', 'T'] ∨ 
                              outcome = ['T', 'T', 'H'] ∨ outcome = ['T', 'T', 'T']} in
  let favorable := {outcome | outcome = ['H', 'T', 'T'] ∨ outcome = ['T', 'H', 'T'] ∨ outcome = ['T', 'T', 'H']} in
  ∑ (x ∈ favorable), 1 / ∑ (x ∈ outcomes), 1 = 3 / 8 :=
sorry

end probability_one_head_in_three_flips_l463_463061


namespace hexadecagon_area_correct_circle_to_hexadecagon_ratio_correct_l463_463511

noncomputable def hexadecagon_area (r : ℝ) : ℝ :=
  4 * r^2 * real.sqrt (2 - real.sqrt 2)

noncomputable def circle_to_hexadecagon_ratio (r : ℝ) : ℝ :=
  real.pi / (4 * r^2 * real.sqrt (2 - real.sqrt 2))

theorem hexadecagon_area_correct (r : ℝ) :
  hexadecagon_area r = 4 * r^2 * real.sqrt (2 - real.sqrt 2) :=
sorry

theorem circle_to_hexadecagon_ratio_correct (r : ℝ) :
  circle_to_hexadecagon_ratio r = real.pi / (4 * r^2 * real.sqrt (2 - real.sqrt 2)) :=
sorry

end hexadecagon_area_correct_circle_to_hexadecagon_ratio_correct_l463_463511


namespace solution_unique_l463_463172

-- Definitions and Constraints
def isSolution (x y n k : ℕ) : Prop :=
  coprime x y ∧ 3^n = x^k + y^k

-- The main theorem: proving the unique solution
theorem solution_unique : ∀ x y n k : ℕ, isSolution x y n k → (x, y, k) = (2, 1, 3) :=
by sorry

end solution_unique_l463_463172


namespace intersection_A_B_l463_463238

noncomputable def A : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def B : Set ℝ := {y | ∃ x, y = Real.log (x^2 + 1) ∧ y ≥ 0}

theorem intersection_A_B : A ∩ {x | ∃ y, y = Real.log (x^2 + 1) ∧ y ≥ 0} = {x | 0 < x ∧ x < 2} :=
  sorry

end intersection_A_B_l463_463238


namespace value_of_M_l463_463289

theorem value_of_M :
  let M := (√(3 + √8) + √(3 - √8)) / √(2√2 + 1) - √(4 - 2√3)
  in M = 3 - √3 :=
by
  let M := (√(3 + √8) + √(3 - √8)) / √(2√2 + 1) - √(4 - 2√3)
  have : M = 3 - √3 := sorry
  exact this

end value_of_M_l463_463289


namespace dispersion_is_variance_l463_463039

def Mean := "Mean"
def Variance := "Variance"
def Median := "Median"
def Mode := "Mode"

def dispersion_measure := Variance

theorem dispersion_is_variance (A B C D : String) (hA : A = Mean) (hB : B = Variance) (hC : C = Median) (hD : D = Mode) : 
  dispersion_measure = B :=
by
  rw [hB]
  exact sorry

end dispersion_is_variance_l463_463039


namespace sin_beta_l463_463631

theorem sin_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos α = 3 / 5) (h4 : Real.cos (α + β) = -5 / 13)
  : Real.sin β = 56 / 65 :=
by
  sorry

end sin_beta_l463_463631


namespace five_digit_even_numbers_less_than_50000_l463_463574

theorem five_digit_even_numbers_less_than_50000 :
  let digits := {1, 2, 3, 4, 5}
  let choices_for_first_digit := {1, 2, 4}
  let choices_for_last_digit := {2, 4}
  let count_options_first_digit := 3
  let count_options_last_digit := 2
  let count_remaining_permutations := Nat.factorial 3
  (count_options_first_digit * count_options_last_digit * count_remaining_permutations) = 36 :=
by
  let digits := {1, 2, 3, 4, 5}
  let choices_for_first_digit := {1, 2, 4}
  let choices_for_last_digit := {2, 4}
  let count_options_first_digit := 3
  let count_options_last_digit := 2
  let count_remaining_permutations := Nat.factorial 3
  simp [count_options_first_digit, count_options_last_digit, count_remaining_permutations, Nat.factorial]
  sorry

end five_digit_even_numbers_less_than_50000_l463_463574


namespace hyperbola_eccentricity_l463_463614

noncomputable def hyperbola : Type := sorry
noncomputable def parabola : Type := sorry

variables (x y a b c p : ℝ)

/- Define the hyperbola equation -/
def hyperbola_eq : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

/- Define the parabola equation -/
def parabola_eq : Prop := y^2 = 2 * p * x

/- Define the right focus of the hyperbola -/
def right_focus := (c, 0)

/- Define the intersection condition and focus alignment -/
def intersection_points := let p1 := (p / 2, p) in 
  hyperbola_eq p1.1 p1.2 ∧ parabola_eq p1.1 p1.2 ∧ right_focus = p1

/- Lean statement for proving the eccentricity -/
theorem hyperbola_eccentricity :
  hyperbola_eq x y a b →
  parabola_eq x y p →
  right_focus = (c, 0) →
  intersection_points →
  sqrt (a^2 + b^2) / a = sqrt(2) + 1 :=
sorry

end hyperbola_eccentricity_l463_463614


namespace cube_paint_sum_l463_463950

theorem cube_paint_sum :
  let painted_squares := [3, 7, 11, 15, 20, 24] in
  ∑ i in painted_squares, i = 80 :=
by
  let painted_squares := [3, 7, 11, 15, 20, 24]
  have sum_painted_squares : ∑ i in painted_squares, i = 80 := rfl
  exact sum_painted_squares

end cube_paint_sum_l463_463950


namespace sum_f_g_periodic_l463_463310

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem sum_f_g_periodic :
  (∀ x : ℝ, g (x + 2) ≠ 0 ∧ f (x - 2) ≠ 0 ∧ g x ≠ 0 → f x / g x = g (x + 2) / f (x - 2)) →
  (g 2024 ≠ 0 ∧ f 2022 ≠ 0 → f 2022 / g 2024 = 2) →
  ∑ k in Finset.range 24, f (2 * k) / g (2 * k + 2) = 30 :=
begin
  intros h1 h2,
  sorry
end

end sum_f_g_periodic_l463_463310


namespace sum_smallest_x_values_l463_463212

theorem sum_smallest_x_values (x : ℝ) (hx : x > 2017) :
  (∃ x1 x2 > 2017, x1 ≠ x2 ∧
  (cos (9 * x1) ^ 5 + cos (x1) ^ 5 = 32 * cos (5 * x1) ^ 5 * cos (4 * x1) ^ 5 + 5 * cos (9 * x1) ^ 2 * cos (x1) ^ 2 * (cos (9 * x1) + cos (x1))) ∧ 
  (cos (9 * x2) ^ 5 + cos (x2) ^ 5 = 32 * cos (5 * x2) ^ 5 * cos (4 * x2) ^ 5 + 5 * cos (9 * x2) ^ 2 * cos (x2) ^ 2 * (cos (9 * x2) + cos (x2))) ∧ 
  x1 + x2 = 4064) :=
sorry

end sum_smallest_x_values_l463_463212


namespace dan_initial_limes_l463_463575

variables (g l p : ℕ)

theorem dan_initial_limes (h1 : g = 4) (h2 : l = 5) : p = g + l := 
by 
  unfold g 
  unfold l 
  sorry

end dan_initial_limes_l463_463575


namespace row_length_in_feet_l463_463128

theorem row_length_in_feet (seeds_per_row : ℕ) (space_per_seed : ℕ) (inches_per_foot : ℕ) (H1 : seeds_per_row = 80) (H2 : space_per_seed = 18) (H3 : inches_per_foot = 12) : 
  seeds_per_row * space_per_seed / inches_per_foot = 120 :=
by
  sorry

end row_length_in_feet_l463_463128


namespace max_surface_area_l463_463510

theorem max_surface_area (l w h : ℕ) (h_conditions : l + w + h = 88) : 
  2 * (l * w + l * h + w * h) ≤ 224 :=
sorry

end max_surface_area_l463_463510


namespace range_of_a_l463_463988

def new_operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, new_operation x (x - a) > 1) ↔ (a < -3 ∨ 1 < a) := 
by
  sorry

end range_of_a_l463_463988


namespace total_amount_l463_463502

theorem total_amount (N50 N: ℕ) (h1: N = 90) (h2: N50 = 77) : 
  (N50 * 50 + (N - N50) * 500) = 10350 :=
by
  sorry

end total_amount_l463_463502


namespace max_PA_PB_PC_l463_463699

theorem max_PA_PB_PC
  (A B C P : Type)
  [metric_space P]
  (isosceles_right_triangle : P → P → P → Prop)
  (CA_CB_one : isosceles_right_triangle A B C ∧ metric_space.dist C A = 1 ∧ metric_space.dist C B = 1)
  (P_on_boundary : P) :
  (∃ (PA PB PC : ℝ), PA * PB * PC ≤ (real.sqrt 2) / 4) :=
sorry

end max_PA_PB_PC_l463_463699


namespace find_a_l463_463168

-- Definition of the binomial square form
def binomial_square (r s x : ℝ) : ℝ := (r * x + s)^2

-- Definition of the polynomial form
def polynomial (a x : ℝ) : ℝ := a * x^2 + 28 * x + 9

-- Statement
theorem find_a : ∃ (a : ℝ), ∀ x : ℝ, polynomial a x = binomial_square (14 / 3) 3 x :=
begin
  use 196 / 9,
  intro x,
  sorry
end

end find_a_l463_463168


namespace quadrilateral_parallelogram_l463_463462

theorem quadrilateral_parallelogram (A B C D : Point) 
  (AB CD AD BC : ℝ) 
  (h1 : distance A B = distance C D) 
  (h2 : distance A D = distance B C) :
  is_parallelogram A B C D :=
by
  sorry

end quadrilateral_parallelogram_l463_463462


namespace blue_to_red_face_area_ratio_l463_463853

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l463_463853


namespace max_intersections_circles_line_l463_463902

axiom disjoint_union_prod (X Y Z : Type) (n m k : ℕ) (hX : Finset.card X = n) (hY : Finset.card Y = m) (hZ : Finset.card Z = k) :
    (∃ (U : Type),  (Finset.card U = n * k) ∧ (Finset.card U = m * k)) → (∃ (W : Type), (Finset.card W = (n + m) * k))

theorem max_intersections_circles_line (n m : ℕ) (h1 : n = 3) (h2 : m = 1) : 
  let max_intersections := 2 * n + 2 * (n * (n - 1) / 2)
  in max_intersections = 12 := 
by {
  sorry 
}

end max_intersections_circles_line_l463_463902


namespace parallel_planes_l463_463659

variables {Point Line Plane : Type}
variables (a : Line) (α β : Plane)

-- Conditions
def line_perpendicular_plane (l: Line) (p: Plane) : Prop := sorry
def planes_parallel (p₁ p₂: Plane) : Prop := sorry

-- Problem statement
theorem parallel_planes (h1: line_perpendicular_plane a α) 
                        (h2: line_perpendicular_plane a β) : 
                        planes_parallel α β :=
sorry

end parallel_planes_l463_463659


namespace length_of_PS_l463_463713

noncomputable def triangle_PQR := {
  P : Point,
  Q : Point,
  R : Point,
  S : Point
}

axiom PQ_eq_13 : (distance triangle_PQR.P triangle_PQR.Q) = 13
axiom QR_eq_14 : (distance triangle_PQR.Q triangle_PQR.R) = 14
axiom PR_eq_15 : (distance triangle_PQR.P triangle_PQR.R) = 15
axiom PS_angle_bisector : is_angle_bisector triangle_PQR.P triangle_PQR.Q triangle_PQR.R triangle_PQR.S

theorem length_of_PS : (distance triangle_PQR.P triangle_PQR.S) = 12.1 := by
  sorry

end length_of_PS_l463_463713


namespace compute_difference_cubed_l463_463356

-- Definition for the number of positive multiples of 6 less than 60
def numMultiplesOf6 (n : ℕ) : ℕ :=
  Finset.card (Finset.filter (λ x, x % 6 = 0) (Finset.range n))

-- Definition for the number of positive integers less than n that are multiples of 3 and 2
def numMultiplesOf3And2 (n : ℕ) : ℕ :=
  Finset.card (Finset.filter (λ x, x % 6 = 0) (Finset.range n))

theorem compute_difference_cubed : (let a := numMultiplesOf6 60 in
                                    let b := numMultiplesOf3And2 60 in
                                    (a - b) ^ 3 = 0) := by
sorry

end compute_difference_cubed_l463_463356


namespace max_min_ab_eq_max_min_a_div_b_l463_463733

theorem max_min_ab_eq_max_min_a_div_b
  (a b : ℝ)
  (p q : ℕ)
  (h₁ : b + 2 * a = p)
  (h₂ : 2 * b + a = q)
  (h₃ : p + q = 6)
  (h₄ : 0 < p)
  (h₅ : 0 < q):
  let ab := a * b,
      a_div_b := a / b in
  (∀ (max_ab min_ab max_a_div_b min_a_div_b : ℝ),
   max_ab = 1 ∧ min_ab = -3 ∧
   max_a_div_b = 1 ∧ min_a_div_b = -3
   → (max_ab = max_a_div_b ∧ min_ab = min_a_div_b)) :=
by
  intros ab a_div_b h;
  sorry

end max_min_ab_eq_max_min_a_div_b_l463_463733


namespace abigail_initial_money_l463_463112

variables (A : ℝ) -- Initial amount of money Abigail had.

-- Conditions
variables (food_rate : ℝ := 0.60) -- 60% spent on food
variables (phone_rate : ℝ := 0.25) -- 25% of the remainder spent on phone bill
variables (entertainment_spent : ℝ := 20) -- $20 spent on entertainment
variables (final_amount : ℝ := 40) -- $40 left after all expenditures

theorem abigail_initial_money :
  (A - (A * food_rate)) * (1 - phone_rate) - entertainment_spent = final_amount → A = 200 :=
by
  intro h
  sorry

end abigail_initial_money_l463_463112


namespace complex_number_quadrant_l463_463992

theorem complex_number_quadrant (a b : ℝ) (h1 : (2 + a * (0+1*I)) / (1 + 1*I) = b + 1*I) (h2: a = 4) (h3: b = 3) : 
  0 < a ∧ 0 < b :=
by
  sorry

end complex_number_quadrant_l463_463992


namespace proof_problem_l463_463366

-- Defining domain A
def A (x : ℝ) : Prop := x^2 - x - 2 > 0

-- Defining domain B
def B (x : ℝ) : Prop := 3 - |x| ≥ 0

-- Defining domain compliments of A in real numbers
def complement_R_A (x : ℝ) : Prop := ¬ A x

-- Defining set C given a parameter m
def C (x m : ℝ) : Prop := (x - m + 1 ) * (x - 2 * m - 1) < 0

-- Proving problem statements
theorem proof_problem : 
  (∀ (x : ℝ), A x ↔ x ∈ Set.Ioo (-∞:ℝ) (-1) ∪ Set.Ioo (2) (∞)) ∧ 
  (∀ (x : ℝ), B x ↔ x ∈ Set.Icc (-3:ℝ) (3)) ∧ 
  (Set.inter A B = Set.Ico (-3) (-1) ∪ Set.Ioc (2) 3) ∧ 
  (Set.union complement_R_A  B = Set.Icc (-3:ℝ) 3) ∧ 
  (∀ (m : ℝ), (∀ x, C x m → B x) → m ∈ Set.Icc (-2) (1)) :=
by
  sorry

end proof_problem_l463_463366


namespace blue_red_area_ratio_l463_463835

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l463_463835


namespace min_value_fracs_l463_463251

-- Define the problem and its conditions in Lean.
theorem min_value_fracs (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  (2 / a + 3 / b) ≥ 8 + 4 * Real.sqrt 3 :=
  sorry

end min_value_fracs_l463_463251


namespace find_real_numbers_a_b_l463_463265

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (Real.sin x * Real.cos x) - (Real.sqrt 3) * a * (Real.cos x) ^ 2 + Real.sqrt 3 / 2 * a + b

theorem find_real_numbers_a_b (a b : ℝ) (h1 : 0 < a)
    (h2 : ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), -2 ≤ f a b x ∧ f a b x ≤ Real.sqrt 3)
    : a = 2 ∧ b = -2 + Real.sqrt 3 :=
sorry

end find_real_numbers_a_b_l463_463265


namespace domino_horizontal_arrangement_possible_l463_463931

theorem domino_horizontal_arrangement_possible (chessboard : list (list (option ℕ))) (extra_cell_added : bool)
  (initial_condition : ∀ (i j : ℕ), i < 8 → j < 8 → 
    ∃ (d : ℕ), chessboard[i][j] = some d ∧ chessboard[i+1][j] = some d ∨ chessboard[i][j+1] = some d) :
  ∃ (new_chessboard : list (list (option ℕ))), (∀ (i j : ℕ), i < 8 → j < 8 → 
    ∃ (d : ℕ), new_chessboard[i][j] = some d ∧ new_chessboard[i+1][j] = some d ∨ new_chessboard[i][j+1] = some d) :=
sorry

end domino_horizontal_arrangement_possible_l463_463931


namespace complement_intersection_l463_463655

open Set Real

def M : Set ℝ := { y | ∃ x : ℝ, y = exp x - 1 }

def N : Set ℝ := { x | 2⁻¹ < 2^(x + 1) ∧ 2^(x + 1) < 2^2 }

theorem complement_intersection :
  (compl M) ∩ N = Ioc (-2) (-1) :=
by
  sorry

end complement_intersection_l463_463655


namespace inverse_of_16_mod_97_l463_463625

theorem inverse_of_16_mod_97:
  (8 : ℤ)⁻¹ ≡ (85 : ℤ) [MOD 97] → (16 : ℤ)⁻¹ ≡ (47 : ℤ) [MOD 97] :=
by
  sorry

end inverse_of_16_mod_97_l463_463625


namespace James_total_passengers_l463_463718

def trucks := 12
def buses := 2
def taxis := 2 * buses
def cars := 30
def motorbikes := 52 - trucks - buses - taxis - cars

def people_in_truck := 2
def people_in_bus := 15
def people_in_taxi := 2
def people_in_motorbike := 1
def people_in_car := 3

def total_passengers := 
  trucks * people_in_truck + 
  buses * people_in_bus + 
  taxis * people_in_taxi + 
  motorbikes * people_in_motorbike + 
  cars * people_in_car

theorem James_total_passengers : total_passengers = 156 := 
by 
  -- Placeholder proof, needs to be completed
  sorry

end James_total_passengers_l463_463718


namespace percentage_gain_l463_463094

theorem percentage_gain : 
  let CP := 300 * 20 in
  let SP1 := 200 * 25 in
  let SP2 := 80 * 30 in
  let Total_SP := SP1 + SP2 in
  let Profit := Total_SP - CP in
  let Percentage_Gain := (Profit / CP) * 100 in
  Percentage_Gain = 23.33
:= by
  sorry

end percentage_gain_l463_463094


namespace max_points_of_intersection_l463_463893

theorem max_points_of_intersection
  (A B C : Circle) (L : Line)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ B ≠ C) :
  (∃ pA pB pC : Point, on_line L pA ∧ on_circle A pA ∧
                    on_line L pB ∧ on_circle B pB ∧
                    on_line L pC ∧ on_circle C pC) ∧
  max_intersection_points A B C L = 12 :=
sorry

end max_points_of_intersection_l463_463893


namespace initial_black_pieces_is_118_l463_463015

open Nat

-- Define the initial conditions and variables
variables (b w n : ℕ)

-- Hypotheses based on the conditions
axiom h1 : b = 2 * w
axiom h2 : w - 2 * n = 1
axiom h3 : b - 3 * n = 31

-- Goal to prove the initial number of black pieces were 118
theorem initial_black_pieces_is_118 : b = 118 :=
by 
  -- We only state the theorem, proof will be added as sorry
  sorry

end initial_black_pieces_is_118_l463_463015


namespace tic_tac_toe_probability_l463_463538

theorem tic_tac_toe_probability :
  let total_positions := Nat.choose 9 3,
      winning_positions := 8 in
  (winning_positions / total_positions : ℚ) = 2 / 21 :=
by
  sorry

end tic_tac_toe_probability_l463_463538


namespace value_of_goods_purchased_l463_463963

variable (V : ℝ)

theorem value_of_goods_purchased (tax_paid : ℝ) (no_tax_limit : ℝ) (tax_rate : ℝ)
    (tax_paid_eq : tax_paid = 112)
    (no_tax_limit_eq : no_tax_limit = 600)
    (tax_rate_eq : tax_rate = 0.10)
    (tax_eq : tax_paid = tax_rate * (V - no_tax_limit)) :
    V = 1720 := 
by 
  have H1 : tax_paid = 112 := tax_paid_eq;
  have H2 : no_tax_limit = 600 := no_tax_limit_eq;
  have H3 : tax_rate = 0.10 := tax_rate_eq;
  rw [H1, H2, H3] at tax_eq;
  have H4 : 112 = 0.10 * (V - 600) := tax_eq;
  have H5 : 112 / 0.10 = V - 600 := by sorry
  have H6 : 1120 = V - 600 := by sorry
  have H7 : V = 1120 + 600 := by sorry
  rw H7;
  norm_num;
  sorry

end value_of_goods_purchased_l463_463963


namespace blue_faces_ratio_l463_463847

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l463_463847


namespace blue_area_factor_12_l463_463826

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l463_463826


namespace max_intersections_circles_line_l463_463904

axiom disjoint_union_prod (X Y Z : Type) (n m k : ℕ) (hX : Finset.card X = n) (hY : Finset.card Y = m) (hZ : Finset.card Z = k) :
    (∃ (U : Type),  (Finset.card U = n * k) ∧ (Finset.card U = m * k)) → (∃ (W : Type), (Finset.card W = (n + m) * k))

theorem max_intersections_circles_line (n m : ℕ) (h1 : n = 3) (h2 : m = 1) : 
  let max_intersections := 2 * n + 2 * (n * (n - 1) / 2)
  in max_intersections = 12 := 
by {
  sorry 
}

end max_intersections_circles_line_l463_463904


namespace shortest_piece_length_l463_463314

theorem shortest_piece_length (initial_length : ℝ) (h : initial_length = 12) : 
    let first_cut := initial_length / 2,
        second_cut := first_cut / 2 in
    second_cut = 3 :=
by {
  intros,
  rw [h],
  simp [first_cut, second_cut],
  sorry
}

end shortest_piece_length_l463_463314


namespace x_intercept_of_l1_is_2_l463_463417

theorem x_intercept_of_l1_is_2 (a : ℝ) (l1_perpendicular_l2 : ∀ (x y : ℝ), 
  ((a+3)*x + y - 4 = 0) -> (x + (a-1)*y + 4 = 0) -> False) : 
  ∃ b : ℝ, (2*b + 0 - 4 = 0) ∧ b = 2 := 
by
  sorry

end x_intercept_of_l1_is_2_l463_463417


namespace permutation_equals_power_l463_463091

-- Definition of permutation with repetition
def permutation_with_repetition (n k : ℕ) : ℕ := n ^ k

-- Theorem to prove
theorem permutation_equals_power (n k : ℕ) : permutation_with_repetition n k = n ^ k :=
by
  sorry

end permutation_equals_power_l463_463091


namespace no_odd_n_with_totient_2_pow_32_l463_463342

theorem no_odd_n_with_totient_2_pow_32 : 
  ¬ ∃ n : ℕ, odd n ∧ nat.totient n = 2^32 := 
sorry

end no_odd_n_with_totient_2_pow_32_l463_463342


namespace problem1_solution_problem2_solution_l463_463979

noncomputable def problem1_expr : ℝ :=
  (-1/2)^0 + |1 - real.tan (real.pi / 3)| + real.sqrt 12 * (-2)^(-1)

theorem problem1_solution : problem1_expr = 0 :=
by sorry

noncomputable def problem2_expr (x : ℝ) : ℝ :=
  ((x^2 - 3*x + 5)/(x-1) + 3 - x) / ((x^2 + 4*x + 4)/(1-x))

theorem problem2_solution (h : x = real.sqrt 3 - 2) : problem2_expr x = -real.sqrt 3 / 3 :=
by sorry

end problem1_solution_problem2_solution_l463_463979


namespace sum_of_center_coordinates_l463_463419

theorem sum_of_center_coordinates (A B : ℝ × ℝ) (hA : A = (9, -5)) (hB : B = (-3, -1)) :
  let mid := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  (mid.1 + mid.2) = 0 :=
by
  sorry

end sum_of_center_coordinates_l463_463419


namespace blue_face_area_factor_l463_463801

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l463_463801


namespace train_crossing_time_l463_463542

theorem train_crossing_time
  (l1 l2 l_train : ℕ)
  (t2 : ℕ)
  (h_l1 : l1 = 170)
  (h_l2 : l2 = 250)
  (h_l_train : l_train = 70)
  (h_t2 : t2 = 20) :
  (l1 + l_train) / ((l2 + l_train) / t2) = 15 :=
by
  rw [h_l1, h_l2, h_l_train, h_t2]
  norm_num
  sorry

end train_crossing_time_l463_463542


namespace razorback_shop_tshirt_revenue_l463_463399

theorem razorback_shop_tshirt_revenue :
  (tshirts_sold : ℕ) (revenue_per_tshirt : ℕ) (total_revenue : ℕ)
  (h1 : tshirts_sold = 20)
  (h2 : revenue_per_tshirt = 215)
  (h3 : total_revenue = tshirts_sold * revenue_per_tshirt) :
  total_revenue = 4300 :=
by
  sorry

end razorback_shop_tshirt_revenue_l463_463399


namespace michael_twice_jacob_l463_463714

variable {J M Y : ℕ}

theorem michael_twice_jacob :
  (J + 4 = 13) → (M = J + 12) → (M + Y = 2 * (J + Y)) → (Y = 3) := by
  sorry

end michael_twice_jacob_l463_463714


namespace blue_area_factor_12_l463_463823

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l463_463823


namespace alcohol_to_water_ratio_l463_463927

theorem alcohol_to_water_ratio (alcohol water : ℚ) (h_alcohol : alcohol = 2/7) (h_water : water = 3/7) : alcohol / water = 2 / 3 := by
  sorry

end alcohol_to_water_ratio_l463_463927


namespace fraction_decomposition_l463_463415

theorem fraction_decomposition (P Q : ℚ) :
  (∀ x : ℚ, 4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24 = (2 * x ^ 2 - 5 * x + 3) * (2 * x - 3))
  → P / (2 * x ^ 2 - 5 * x + 3) + Q / (2 * x - 3) = (8 * x ^ 2 - 9 * x + 20) / (4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24)
  → P = 4 / 9 ∧ Q = 68 / 9 := by 
  sorry

end fraction_decomposition_l463_463415


namespace blue_face_area_greater_than_red_face_area_l463_463830

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l463_463830


namespace problem_accuracy_hundredths_place_l463_463044

def accurate_to_hundredths_place (n : ℝ) : Prop :=
  let s := to_string n
  (s.reverse.takeWhile (λ c, c ≠ '.')).length = 2

theorem problem_accuracy_hundredths_place:
  accurate_to_hundredths_place 25.00 :=
by
  sorry

end problem_accuracy_hundredths_place_l463_463044


namespace max_possible_numbers_sum_2019_l463_463386

theorem max_possible_numbers_sum_2019 (n : ℕ) (a : fin n → ℕ)
  (h_sum : (∑ i, a i) = 2019)
  (h_cond : ∀ (i j : fin n), i ≤ j → a i ≠ 40 ∧ (∑ k in finset.Icc i j, a k) ≠ 40) :
  n ≤ 1019 :=
sorry

end max_possible_numbers_sum_2019_l463_463386


namespace sum_of_valid_c_values_l463_463196

theorem sum_of_valid_c_values:
  let quadratic_rational_roots := λ (c : ℕ), (∃ d : ℕ, 49 - 12 * c = d * d) ∧ (c > 0)
  in ∑ c in finset.filter quadratic_rational_roots (finset.range 5), c = 6 :=
by
  -- The proof will go here
  sorry

end sum_of_valid_c_values_l463_463196


namespace count_ordered_pairs_l463_463728

-- Definitions based on the given conditions
def is_partition (C D : Finset ℕ) : Prop :=
  C ∪ D = Finset.range 1 9 ∧ C ∩ D = ∅ ∧ C ≠ ∅ ∧ D ≠ ∅

def elements_not_member_own_set (C D : Finset ℕ) : Prop :=
  C.card ∉ C ∧ D.card ∉ D

-- Main statement
theorem count_ordered_pairs : 
  let M := (Finset.powersetLen 1 7 (Finset.range 1 8)).filter (λ C, 
                    (8 - C.card) ∉ C ∧ 
                    C.card ∉ (Finset.range 1 9 \ C))
  in M.card = 44 := 
by
  sorry

end count_ordered_pairs_l463_463728


namespace detergent_cost_effectiveness_l463_463085

def cost_effectiveness (cost: ℝ) (quantity: ℝ) : ℝ := cost / quantity

variables {c_XS c_S c_M q_XS q_S q_L q_M : ℝ}

-- Conditions
axiom S_cost : c_S = 1.8 * c_XS
axiom S_quantity : q_S = 1.5 * q_XS
axiom M_cost : c_M = 1.2 * c_S
axiom M_quantity : q_M = 0.75 * q_L
axiom L_quantity : q_L = 1.4 * q_S

-- Expected result
def best_buy_order : list string := ["XS", "L", "S", "M"]

theorem detergent_cost_effectiveness : 
  [("XS", cost_effectiveness c_XS q_XS), 
   ("S", cost_effectiveness c_S q_S), 
   ("M", cost_effectiveness c_M q_M), 
   ("L", cost_effectiveness c_S q_L)]
  = [("XS", 1), ("L", 1.2), ("S", 1.2), ("M", 1.372)] 
  → ["XS", "L", "S", "M"] = best_buy_order :=
sorry

end detergent_cost_effectiveness_l463_463085


namespace blue_face_area_greater_than_red_face_area_l463_463831

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l463_463831


namespace point_T_coordinates_l463_463167

-- Definition of a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of a square with specific points O, P, Q, R
structure Square where
  O : Point
  P : Point
  Q : Point
  R : Point

-- Condition: O is the origin
def O : Point := {x := 0, y := 0}

-- Condition: Q is at (3, 3)
def Q : Point := {x := 3, y := 3}

-- Assuming the function area_triang for calculating the area of a triangle given three points
def area_triang (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x))

-- Assuming the function area_square for calculating the area of a square given the length of the side
def area_square (s : ℝ) : ℝ := s * s

-- Coordinates of point P and R since it's a square with sides parallel to axis
def P : Point := {x := 3, y := 0}
def R : Point := {x := 0, y := 3}

-- Definition of the square OPQR
def OPQR : Square := {O := O, P := P, Q := Q, R := R}

-- Length of the side of square OPQR
def side_length : ℝ := 3

-- Area of the square OPQR
def square_area : ℝ := area_square side_length

-- Twice the area of the square OPQR
def required_area : ℝ := 2 * square_area

-- Point T that needs to be proven
def T : Point := {x := 3, y := 12}

-- The main theorem to prove
theorem point_T_coordinates (T : Point) : area_triang P Q T = required_area → T = {x := 3, y := 12} :=
by
  sorry

end point_T_coordinates_l463_463167


namespace sum_positive_integer_values_c_l463_463197

theorem sum_positive_integer_values_c :
  (∑ c in (finset.filter (λ c : ℤ, is_integer c ∧ c > 0)
    (finset.image (λ k : ℤ, (49 - k^2) / 12)
      (finset.range 7))), c) = 6 :=
sorry

end sum_positive_integer_values_c_l463_463197


namespace perpendicular_lines_slope_l463_463635

theorem perpendicular_lines_slope (m : ℝ) : 
  (∃ (m : ℝ), (x y : ℝ), x - m * y + 2 * m = 0 ∧ x + 2 * y - m = 0 ∧ (1 / m) * (-1 / 2) = -1) → m = 1 / 2 :=
by
  sorry

end perpendicular_lines_slope_l463_463635


namespace system_of_equations_solution_l463_463572

theorem system_of_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 10) ∧ (12 * x - 8 * y = 8) ∧ (x = 14 / 9) ∧ (y = 4 / 3) :=
by
  sorry

end system_of_equations_solution_l463_463572


namespace planet_xavier_distance_midway_l463_463867

theorem planet_xavier_distance_midway (perigee apogee : ℝ) (AF BF : ℝ)
  (h1 : perigee = 2)
  (h2 : apogee = 12)
  (h3 : AF = 2)
  (h4 : BF = 12) :
  let AB := AF + BF in
  let MF := AB / 2 in
  MF = 7 := sorry

end planet_xavier_distance_midway_l463_463867


namespace final_pressure_of_helium_l463_463942

theorem final_pressure_of_helium
  (p v v' : ℝ) (k : ℝ)
  (h1 : p = 4)
  (h2 : v = 3)
  (h3 : v' = 6)
  (h4 : p * v = k)
  (h5 : ∀ p' : ℝ, p' * v' = k → p' = 2) :
  p' = 2 := by
  sorry

end final_pressure_of_helium_l463_463942


namespace store_profit_l463_463925

noncomputable def cost_price (C : ℝ) := C
noncomputable def initial_selling_price (C : ℝ) := 1.20 * C
noncomputable def second_selling_price (C : ℝ) := 1.50 * C
noncomputable def final_selling_price (C : ℝ) := 0.93 * (1.50 * C)
noncomputable def profit (C : ℝ) := final_selling_price C - cost_price C

theorem store_profit (C : ℝ) : profit C = 0.395 * C :=
by
  simp [profit, final_selling_price, cost_price]
  linarith

end store_profit_l463_463925


namespace pentagon_angle_sum_l463_463151

theorem pentagon_angle_sum
  (a b c d : ℝ) (Q : ℝ)
  (sum_angles : 180 * (5 - 2) = 540)
  (given_angles : a = 130 ∧ b = 80 ∧ c = 105 ∧ d = 110) :
  Q = 540 - (a + b + c + d) := by
  sorry

end pentagon_angle_sum_l463_463151


namespace distribution_not_possible_l463_463883

open Finset

-- Definitions of regions
inductive Region
| A | B | C | D | E | F | G

-- Establishing distinct values for these regions
open Region
def regions := {A, B, C, D, E, F, G}

-- Assume there exist three lines which divide the regions
noncomputable def lines : Finset (Finset Region) :=
  {{A, B, C}, {D, E, F, G}, -- Line 1
   {A, D, E}, {B, C, F, G}, -- Line 2
   {A, B, F}, {C, D, E, G}} -- Line 3

-- Sum conditions for each line's partition regions 
def sum {α : Type*} [AddCommMonoid α] (s : Finset α) (f : α → ℕ) : ℕ := s.sum f

-- Distributing numbers 1 to 7 in such a way that sums on either side of lines are equal
theorem distribution_not_possible :
  (∀ (a b c d e f g : ℕ), a ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         b ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         c ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         d ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         e ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         f ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         g ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         sum {A, B, C} (λ x, if x = A then a else if x = B then b else if x = C then c else 0) = 
                         sum {D, E, F, G} (λ x, if x = D then d else if x = E then e else if x = F then f else if x = G then g else 0) ∧
                         sum {A, D, E} (λ x, if x = A then a else if x = D then d else if x = E then e else 0) = 
                         sum {B, C, F, G} (λ x, if x = B then b else if x = C then c else if x = F then f else if x = G then g else 0) ∧
                         sum {A, B, F} (λ x, if x = A then a else if x = B then b else if x = F then f else 0) = 
                         sum {C, D, E, G} (λ x, if x = C then c else if x = D then d else if x = E then e else if x = G then g else 0)) → false :=
sorry

end distribution_not_possible_l463_463883


namespace find_length_AC_l463_463710

theorem find_length_AC {A B C E H : Type} 
  (AE CE x : ℝ) (AC : A → C → EuclideanSpace ℝ)
  (circle_passes_through : ∀ {A B C : Type}, collinear A B C)
  (trapezoid_ABCE : trapezoid A B C E) 
  (base_AE : AE = 16) 
  (side_CE : CE = 8 * Real.sqrt 3)
  (angle_AHB : ∠ A H B = 60) 
  : length AC = 8 :=
sorry

end find_length_AC_l463_463710


namespace hyperbola_foci_l463_463409

/-- The coordinates of the foci of the hyperbola y^2 / 3 - x^2 = 1 are (0, ±2). -/
theorem hyperbola_foci (x y : ℝ) :
  x^2 - (y^2 / 3) = -1 → (0 = x ∧ (y = 2 ∨ y = -2)) :=
sorry

end hyperbola_foci_l463_463409


namespace probability_of_winning_noughts_l463_463517

def ticTacToeProbability : ℚ :=
  let totalWays := Nat.choose 9 3
  let winningWays := 8
  winningWays / totalWays

theorem probability_of_winning_noughts :
  ticTacToeProbability = 2 / 21 := by
  sorry

end probability_of_winning_noughts_l463_463517


namespace balance_equation_l463_463762

variable (G Y W B : ℝ)
variable (balance1 : 4 * G = 8 * B)
variable (balance2 : 3 * Y = 7.5 * B)
variable (balance3 : 8 * B = 6 * W)

theorem balance_equation : 5 * G + 3 * Y + 4 * W = 23.5 * B := by
  sorry

end balance_equation_l463_463762


namespace solve_quadratic_l463_463060

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 → x = 1 :=
by
  intros x h
  sorry

end solve_quadratic_l463_463060


namespace h_prime_at_6_tangent_exists_and_points_of_tangency_l463_463075

section part1

variables (f g : ℝ → ℝ)
variables (h : ℝ → ℝ)

-- Given conditions
def f_6 : Prop := f 6 = 5
def g_6 : Prop := g 6 = 4
def f'_6 : Prop := deriv f 6 = 3
def g'_6 : Prop := deriv g 6 = 1
def h_def : h = λ x, f x * g x - 1

-- Proof statement
theorem h_prime_at_6 : f_6 f → g_6 g → f'_6 f → g'_6 g → h_def f g h → deriv h 6 = 17 := sorry

end part1

section part2

-- Given function definition
def sin_function (x : ℝ) : ℝ := Real.sin x

-- Proof of point of tangency
theorem tangent_exists_and_points_of_tangency (k : ℤ) :
  (∃ m, Real.cos m = 1 / 2) ∧ ((∃ m, ∃ n, (m = 2 * k * Real.pi + Real.pi / 3 ∨ m = 2 * k * Real.pi - Real.pi / 3) ∧ n = Real.sqrt 3 / 2)) := sorry

end part2

end h_prime_at_6_tangent_exists_and_points_of_tangency_l463_463075


namespace blue_red_face_area_ratio_l463_463811

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l463_463811


namespace blue_face_area_greater_than_red_face_area_l463_463833

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l463_463833


namespace fourth_term_of_expansion_l463_463747

def a : ℝ := ∫ x in 1..2, (3 * x ^ 2 - 2 * x)

theorem fourth_term_of_expansion : (ax^2 - 1/x)^6.choose(3) * (4x^2)^(6-3) * (-1/x)^3 = -1280 * x^3 :=
by
  sorry

end fourth_term_of_expansion_l463_463747


namespace blue_red_face_area_ratio_l463_463813

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l463_463813


namespace blue_area_factor_12_l463_463819

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l463_463819


namespace square_sides_product_l463_463864

theorem square_sides_product (a : ℝ) : 
  (∃ s : ℝ, s = 5 ∧ (a = -3 + s ∨ a = -3 - s)) → (a = 2 ∨ a = -8) → -8 * 2 = -16 :=
by
  intro _ _
  exact rfl

end square_sides_product_l463_463864


namespace volume_of_right_square_pyramid_l463_463598

theorem volume_of_right_square_pyramid (a α : ℝ) : 
  let S := a^2,
      H := (a / 2) * (Real.tan α) in
  (1 / 3) * S * H = (a^3 / 6) * (Real.tan α) := by
  sorry

end volume_of_right_square_pyramid_l463_463598


namespace proof_l463_463731

noncomputable def problem : Prop :=
∃ (a b : ℝ), 
(∀ n : ℕ, ∃ d : ℝ, a_{n+1} = a_n + d ∧ b_{1 + n} = (a_{1 + n})^2) ∧
(a < a + d) ∧
(lim b_{n+1} = 1 + sqrt 2) ∧
(a = -sqrt 2) ∧
(d = 2*sqrt 2 - 2)

theorem proof (exists_seq : problem) : 
∃ a d : ℝ, 
a = -sqrt 2 ∧ 
d = 2*sqrt 2 - 2 :=
sorry

end proof_l463_463731


namespace geometric_seq_log_sum_eq_seven_l463_463321

theorem geometric_seq_log_sum_eq_seven (b : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, b n > 0) 
  (h_geom : ∀ n, b (n + 1) = b n * r) (h_prod : b 6 * b 7 = 3) : 
  (∑ i in finset.range 14, real.log (b i) / real.log 3) = 7 := 
sorry

end geometric_seq_log_sum_eq_seven_l463_463321


namespace sum_of_eight_occurrences_l463_463131

theorem sum_of_eight_occurrences (a : ℝ) (n : ℕ) (m : ℝ) 
  (h1 : a = 5) 
  (h2 : n = 8) 
  (h3 : m ≈ 1.29248125) : 
  (finset.sum (finset.range n) (λ _ => a ^ 5)) = a ^ (5 + m) := 
sorry

end sum_of_eight_occurrences_l463_463131


namespace tan_x0_eq_neg_sqrt3_l463_463258

def f (x : ℝ) : ℝ := (1 / 2) * x - (1 / 4) * Real.sin x - (Real.sqrt 3 / 4) * Real.cos x

theorem tan_x0_eq_neg_sqrt3 (x0 : ℝ) (h : deriv f x0 = 1) : Real.tan x0 = -Real.sqrt 3 := 
sorry

end tan_x0_eq_neg_sqrt3_l463_463258


namespace derivative_of_f_at_alpha_l463_463679

-- Given condition
def f (x : ℝ) : ℝ := 1 - Real.cos x

-- Proof statement
theorem derivative_of_f_at_alpha (α : ℝ) : deriv f α = Real.sin α :=
by 
  -- Provided the proof steps are skipped
  sorry

end derivative_of_f_at_alpha_l463_463679


namespace miles_mike_l463_463370

def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie (A : ℕ) : ℝ := 2.50 + 5.00 + 0.25 * A

theorem miles_mike {M A : ℕ} (annie_ride_miles : A = 16) (same_cost : cost_mike M = cost_annie A) : M = 36 :=
by
  rw [cost_annie, annie_ride_miles] at same_cost
  simp [cost_mike] at same_cost
  sorry

end miles_mike_l463_463370


namespace simplify_expression_l463_463390

theorem simplify_expression (a b c : ℝ) : a - (a - b + c) = b - c :=
by sorry

end simplify_expression_l463_463390


namespace trapezoid_divisibility_by_3_l463_463305

/--
Let \( p_1 \) and \( p_2 \) be prime numbers, and let the trapezoid be isosceles with one acute angle of \( 45^\circ \).
We need to prove that either the midsegment \( \frac{p_1 + p_2}{2} \) or the height \( m \) of the trapezoid is divisible by 3.
-/
theorem trapezoid_divisibility_by_3 (p1 p2 : ℕ) (m : ℕ) (h_iso : is_isosceles_trapezoid) 
  (h_angle : acute_angle = 45) (h_primes : prime p1 ∧ prime p2) :
  ∃ k : ℕ, (k = (p1 + p2) / 2 ∨ k = m) ∧ k % 3 = 0 :=
sorry

end trapezoid_divisibility_by_3_l463_463305


namespace probability_of_winning_position_l463_463531

-- Given Conditions
def tic_tac_toe_board : Type := Fin 3 × Fin 3
def is_nought (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := b p
def is_cross (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := ¬ b p

-- Winning positions in Tic-Tac-Toe are vertical, horizontal, or diagonal lines
def is_winning_position (b : tic_tac_toe_board → bool) : Prop :=
  (∃ i : Fin 3, ∀ j : Fin 3, is_nought b (i, j)) ∨ -- Row
  (∃ j : Fin 3, ∀ i : Fin 3, is_nought b (i, j)) ∨ -- Column
  (∀ i : Fin 3, is_nought b (i, i)) ∨ -- Main diagonal
  (∀ i : Fin 3, is_nought b (i, Fin.mk (2 - i.1) (by simp [i.1]))) -- Anti-diagonal

-- Problem Statement
theorem probability_of_winning_position :
  let total_positions := Nat.choose 9 3
  let winning_positions := 8
  winning_positions / total_positions = (2:ℚ) / 21 :=
by sorry

end probability_of_winning_position_l463_463531


namespace relationship_among_a_b_c_l463_463244

theorem relationship_among_a_b_c
  (f : ℝ → ℝ)
  (h_deriv : ∀ x > 0, deriv f x ≠ 0)
  (h_eq : ∀ x > 0, f (f x - log 2016 x) = 2017) :
  let a := f (2 ^ 0.5),
      b := f (Real.log 3 / Real.log π),
      c := f (Real.log 3 / Real.log 4)
  in a > b ∧ b > c :=
by
  sorry

end relationship_among_a_b_c_l463_463244


namespace restaurant_total_cost_l463_463971

theorem restaurant_total_cost :
  let vegetarian_cost := 5
  let chicken_cost := 7
  let steak_cost := 10
  let kids_cost := 3
  let tax_rate := 0.10
  let tip_rate := 0.15
  let num_vegetarians := 3
  let num_chicken_lovers := 4
  let num_steak_enthusiasts := 2
  let num_kids_hot_dog := 3
  let subtotal := (num_vegetarians * vegetarian_cost) + (num_chicken_lovers * chicken_cost) + (num_steak_enthusiasts * steak_cost) + (num_kids_hot_dog * kids_cost)
  let tax := subtotal * tax_rate
  let tip := subtotal * tip_rate
  let total_cost := subtotal + tax + tip
  total_cost = 90 :=
by sorry

end restaurant_total_cost_l463_463971


namespace James_total_passengers_l463_463720

def trucks := 12
def buses := 2
def taxis := 2 * buses
def cars := 30
def motorbikes := 52 - trucks - buses - taxis - cars

def people_in_truck := 2
def people_in_bus := 15
def people_in_taxi := 2
def people_in_motorbike := 1
def people_in_car := 3

def total_passengers := 
  trucks * people_in_truck + 
  buses * people_in_bus + 
  taxis * people_in_taxi + 
  motorbikes * people_in_motorbike + 
  cars * people_in_car

theorem James_total_passengers : total_passengers = 156 := 
by 
  -- Placeholder proof, needs to be completed
  sorry

end James_total_passengers_l463_463720


namespace solution_set_inequality_l463_463425

theorem solution_set_inequality (x : ℝ) :
  ((x + (1 / 2)) * ((3 / 2) - x) ≥ 0) ↔ (- (1 / 2) ≤ x ∧ x ≤ (3 / 2)) :=
by sorry

end solution_set_inequality_l463_463425


namespace selling_price_is_correct_l463_463551

noncomputable def cost_price : ℝ := 192
def profit_percentage : ℝ := 0.25
def profit (cp : ℝ) (pp : ℝ) : ℝ := pp * cp
def selling_price (cp : ℝ) (pft : ℝ) : ℝ := cp + pft

theorem selling_price_is_correct : selling_price cost_price (profit cost_price profit_percentage) = 240 :=
sorry

end selling_price_is_correct_l463_463551


namespace solve_price_reduction_l463_463081

-- Definitions based on conditions
def daily_sales_volume (x : ℝ) : ℝ := 30 + 2 * x
def profit_per_item (x : ℝ) : ℝ := 50 - x
def daily_profit (x : ℝ) : ℝ := (50 - x) * (30 + 2 * x)

-- Statement
theorem solve_price_reduction :
  ∃ x : ℝ, daily_profit x = 2100 ∧ x ∈ {15, 20} :=
begin
  -- solution here
  sorry,
end

end solve_price_reduction_l463_463081


namespace solution_set_inequality_l463_463226

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (x1 x2 : ℝ) : f (x1 + x2) = f x1 + f x2 - 1
axiom f_pos (x : ℝ) (h : x > 0) : f x > 1
axiom f_at_4 : f 4 = 5

theorem solution_set_inequality : { m : ℝ | f (3 * m - 2) < 3 } = set.Iio (4 / 3) :=
by sorry

end solution_set_inequality_l463_463226


namespace sum_ratio_l463_463730

def arithmetic_sequence (a_1 d : ℚ) (n : ℕ) : ℚ :=
  a_1 + (n - 1) * d

def S_n (a_1 d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a_1 + (n - 1) * d) / 2 -- sum of first n terms of arithmetic sequence

theorem sum_ratio (a_1 d : ℚ) (h : 13 * (a_1 + 6 * d) = 7 * (a_1 + 3 * d)) :
  S_n a_1 d 13 / S_n a_1 d 7 = 1 :=
by
  -- Proof omitted
  sorry

end sum_ratio_l463_463730


namespace blue_red_face_area_ratio_l463_463818

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l463_463818


namespace prevent_exploit_l463_463934

structure Purchase :=
  (laptop_price : ℕ)
  (headphones_price : ℕ)
  (total_price : ℕ)
  (delivery_free : Prop)
  (headphones_delivered_first : Prop)

def mechanism (p : Purchase) :=
  p.total_price = p.laptop_price + p.headphones_price ∧
  p.laptop_price = 115000 ∧
  p.headphones_price = 15000 ∧
  p.delivery_free ∧
  p.headphones_delivered_first

theorem prevent_exploit (p : Purchase) (m : mechanism p) :
  (m → ∃ payment_policy: ∀ (headphones_returned : Prop) (laptop_returned : Prop), 
    (headphones_returned → ¬ laptop_returned)) :=
sorry

end prevent_exploit_l463_463934


namespace first_day_of_century_never_sun_wed_fri_l463_463052

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0))

theorem first_day_of_century_never_sun_wed_fri :
  ∀ c : ℕ, 
  let first_day_of_century := (1 + (∑ k in range (100*c + 1), if is_leap_year k then 2 else 1)) % 7 in
  ¬ (first_day_of_century = 0 ∨ first_day_of_century = 3 ∨ first_day_of_century = 5) := 
by
  sorry

end first_day_of_century_never_sun_wed_fri_l463_463052


namespace largest_sum_of_digits_l463_463673

noncomputable theory

-- Definition of the problem
def digits (n : ℕ) : Prop := n < 10

-- Main theorem statement
theorem largest_sum_of_digits {a b c : ℕ} (h_a : digits a) (h_b : digits b) (h_c : digits c) (z : ℕ) (h_z : 0 < z ∧ z ≤ 12) :
  (0.abc = 1 / z) → a + b + c ≤ 8 := 
sorry

end largest_sum_of_digits_l463_463673


namespace angle_in_third_quadrant_l463_463554

theorem angle_in_third_quadrant (α : ℝ) (h : α = 2023) : 180 < α % 360 ∧ α % 360 < 270 := by
  sorry

end angle_in_third_quadrant_l463_463554


namespace sum_of_valid_c_values_l463_463194

theorem sum_of_valid_c_values:
  let quadratic_rational_roots := λ (c : ℕ), (∃ d : ℕ, 49 - 12 * c = d * d) ∧ (c > 0)
  in ∑ c in finset.filter quadratic_rational_roots (finset.range 5), c = 6 :=
by
  -- The proof will go here
  sorry

end sum_of_valid_c_values_l463_463194


namespace candy_distribution_l463_463584

-- Definitions based on conditions
def distribution_problem (r b g w : Nat) : Prop :=
  (1 ≤ r) ∧ (1 ≤ b) ∧ (1 ≤ g) ∧ (r + b + g + w = 8)

-- Main theorem statement
theorem candy_distribution : 
  ∑ r in (Finset.range 7 \ {0}), ∑ b in (Finset.range (7-r) \ {0}), ∑ g in (Finset.range (7-r-b) \ {0}),
    Nat.choose 8 r * Nat.choose (8 - r) b * Nat.choose (8 - r - b) g * 2^(8 - (r + b + g)) = 1600 :=
by
  sorry

end candy_distribution_l463_463584


namespace find_ending_number_l463_463176

theorem find_ending_number (N : ℕ) :
  (∃ k : ℕ, N = 3 * k) ∧ (∀ x,  40 < x ∧ x ≤ N → x % 3 = 0) ∧ (∃ avg, avg = (N + 42) / 2 ∧ avg = 60) → N = 78 :=
by
  sorry

end find_ending_number_l463_463176


namespace expense_and_income_calculations_l463_463072

def alexander_salary : ℕ := 125000
def natalia_salary : ℕ := 61000
def utilities_transport_household : ℕ := 17000
def loan_repayment : ℕ := 15000
def theater_cost : ℕ := 5000
def cinema_cost_per_person : ℕ := 1000
def savings_crimea : ℕ := 20000
def dining_weekday_cost : ℕ := 1500
def dining_weekend_cost : ℕ := 3000
def weekdays : ℕ := 20
def weekends : ℕ := 10
def phone_A_cost : ℕ := 57000
def phone_B_cost : ℕ := 37000

def total_expenses : ℕ :=
  utilities_transport_household +
  loan_repayment +
  theater_cost + 2 * cinema_cost_per_person +
  savings_crimea +
  weekdays * dining_weekday_cost +
  weekends * dining_weekend_cost

def net_income : ℕ :=
  alexander_salary + natalia_salary

def can_buy_phones : Prop :=
  net_income - total_expenses < phone_A_cost + phone_B_cost

theorem expense_and_income_calculations :
  total_expenses = 119000 ∧
  net_income = 186000 ∧
  can_buy_phones :=
by
  sorry

end expense_and_income_calculations_l463_463072


namespace integral_f_eq_l463_463223

-- Define the piecewise function f
def f : ℝ → ℝ :=
  λ x, if 1 ≤ x ∧ x ≤ 2 then 1/x else if 0 ≤ x ∧ x ≤ 1 then Real.exp (-x) else 0

-- Statement to prove the definite integral
theorem integral_f_eq : ∫ x in 0..2, f x = 1 - Real.exp (-1) + Real.log 2 :=
by
  sorry

end integral_f_eq_l463_463223


namespace abigail_initial_money_l463_463111

variable (X : ℝ) -- Let X be the initial amount of money

def spent_on_food (X : ℝ) := 0.60 * X
def remaining_after_food (X : ℝ) := X - spent_on_food X
def spent_on_phone (X : ℝ) := 0.25 * remaining_after_food X
def remaining_after_phone (X : ℝ) := remaining_after_food X - spent_on_phone X
def final_amount (X : ℝ) := remaining_after_phone X - 20

theorem abigail_initial_money
    (food_spent : spent_on_food X = 0.60 * X)
    (phone_spent : spent_on_phone X = 0.10 * X)
    (remaining_after_entertainment : final_amount X = 40) :
    X = 200 :=
by
    sorry

end abigail_initial_money_l463_463111


namespace greatest_distance_l463_463704

noncomputable def setA : Set ℂ := {z | z^4 = 16}
noncomputable def setB : Set ℂ := {z | z^3 - 12 * z^2 + 36 * z - 64 = 0}

-- Define a distance function
def distance (z1 z2 : ℂ) : ℝ := Complex.abs (z1 - z2)

-- Find the greatest distance between any point in set A and any point in set B
theorem greatest_distance : (sup (setA.Product setB).image (λ p : ℂ × ℂ, distance p.1 p.2)) = 10 := sorry

end greatest_distance_l463_463704


namespace probability_at_least_one_bean_distribution_of_X_expectation_of_X_l463_463583

noncomputable def total_ways := Nat.choose 6 3
noncomputable def ways_select_2_egg_1_bean := (Nat.choose 4 2) * (Nat.choose 2 1)
noncomputable def ways_select_1_egg_2_bean := (Nat.choose 4 1) * (Nat.choose 2 2)
noncomputable def at_least_one_bean_probability := (ways_select_2_egg_1_bean + ways_select_1_egg_2_bean) / total_ways

theorem probability_at_least_one_bean : at_least_one_bean_probability = 4 / 5 :=
by sorry

noncomputable def p_X_eq_0 := (Nat.choose 4 3) / total_ways
noncomputable def p_X_eq_1 := ways_select_2_egg_1_bean / total_ways
noncomputable def p_X_eq_2 := ways_select_1_egg_2_bean / total_ways

theorem distribution_of_X : p_X_eq_0 = 1 / 5 ∧ p_X_eq_1 = 3 / 5 ∧ p_X_eq_2 = 1 / 5 :=
by sorry

noncomputable def E_X := (0 * p_X_eq_0) + (1 * p_X_eq_1) + (2 * p_X_eq_2)

theorem expectation_of_X : E_X = 1 :=
by sorry

end probability_at_least_one_bean_distribution_of_X_expectation_of_X_l463_463583


namespace price_reduction_2100_yuan_l463_463083

-- Definitions based on conditions
def initial_sales : ℕ := 30
def initial_profit_per_item : ℕ := 50
def additional_sales_per_yuan (x : ℕ) : ℕ := 2 * x
def new_profit_per_item (x : ℕ) : ℕ := 50 - x
def target_profit : ℕ := 2100

-- Final proof statement, showing the price reduction needed
theorem price_reduction_2100_yuan (x : ℕ) 
  (h : (50 - x) * (30 + 2 * x) = 2100) : 
  x = 20 := 
by 
  sorry

end price_reduction_2100_yuan_l463_463083


namespace faster_train_passes_slower_train_in_36_seconds_l463_463450

theorem faster_train_passes_slower_train_in_36_seconds 
  (v_f v_s : ℝ) (l : ℝ) (v_f_kmph v_s_kmph : v_f = 46 ∧ v_s = 36)
  (l_m : l = 50) :
  let v_r := (v_f - v_s) * (5 / 18)
  let d := 2 * l
  let t := d / v_r
  v_r = (10 * (5 / 18)) → -- ensures the relative speed calculation is correct
  d = 100 → -- ensures the total distance calculation is correct
  t ≈ 36 :=   -- proves that the time is approximately 36 seconds
by
  sorry

end faster_train_passes_slower_train_in_36_seconds_l463_463450


namespace total_points_on_surface_l463_463012

-- Define the conditions
def cubes : ℕ := 7
def faces_per_cube : ℕ := 6
def points_on_faces := [1, 2, 3, 4, 5, 6]
def sum_opposite_faces (n : ℕ) : Prop := ∀ i j, (i+j=7) → (i ≠ j)

-- Main problem statement
theorem total_points_on_surface 
  (cubes : ℕ)
  (faces_per_cube : ℕ)
  (points_on_faces : List ℕ)
  (sum_opposite_faces : ∀ i j, (i + j = 7) → (i ≠ j))
  (glued_pairs : ℕ) :
  let initial_points_per_cube := points_on_faces.sum
  let total_initial_points := cubes * initial_points_per_cube
  let points_lost_on_glued_faces := 54
  let remaining_visible_faces := 21
  9 * (6 / 2) →
  glued_pairs == 9 →
  total_initial_points - points_lost_on_glued_faces + remaining_visible_faces = 75 :=
by
  intros
  sorry

end total_points_on_surface_l463_463012


namespace red_blue_area_ratio_is_12_l463_463807

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l463_463807


namespace probability_at_least_two_consecutive_heads_l463_463497

open Classical

-- Define the number of possible outcomes when a fair coin is tossed 4 times.
def total_outcomes : ℕ := 2^4

-- Define the event of no consecutive heads in 4 coin tosses.
def no_consecutive_heads (s : list bool) : Prop :=
  ∀ i, (i + 1 < s.length) → ¬(s[i] = tt ∧ s[i + 1] = tt)

-- List all outcomes and count those without consecutive heads.
def count_no_consecutive_heads : ℕ :=
  let outcomes := [[],
                   [ff, ff, ff, ff], [ff, ff, ff, tt], [ff, ff, tt, ff], [ff, ff, tt, tt], 
                   [ff, tt, ff, ff], [ff, tt, ff, tt], [ff, tt, tt, ff], [ff, tt, tt, tt], 
                   [tt, ff, ff, ff], [tt, ff, ff, tt], [tt, ff, tt, ff], [tt, ff, tt, tt], 
                   [tt, tt, ff, ff], [tt, tt, ff, tt], [tt, tt, tt, ff], [tt, tt, tt, tt]] 
  in outcomes.filter no_consecutive_heads

-- Calculate the probability of having at least two consecutive heads.
def probability_two_consecutive_heads : ℚ :=
  1 - (count_no_consecutive_heads / total_outcomes)

theorem probability_at_least_two_consecutive_heads :
  probability_two_consecutive_heads = 5 / 8 :=
sorry

end probability_at_least_two_consecutive_heads_l463_463497


namespace find_y_l463_463170

theorem find_y (y : ℝ) (h : 9^(Real.log y / Real.log 8) = 81) : y = 64 :=
by
  sorry

end find_y_l463_463170


namespace probability_of_winning_noughts_l463_463518

def ticTacToeProbability : ℚ :=
  let totalWays := Nat.choose 9 3
  let winningWays := 8
  winningWays / totalWays

theorem probability_of_winning_noughts :
  ticTacToeProbability = 2 / 21 := by
  sorry

end probability_of_winning_noughts_l463_463518


namespace ratio_of_areas_l463_463491

variable (d : ℝ) (π : ℝ)

-- Define the radii
def r := d / 2
def R := 3 * d / 2

-- Define the areas
def A_original := π * r^2
def A_enlarged := π * R^2

-- Prove the ratio
theorem ratio_of_areas : A_enlarged / A_original = 9 :=
by
  sorry

end ratio_of_areas_l463_463491


namespace graph_abs_symmetric_yaxis_l463_463328

theorem graph_abs_symmetric_yaxis : 
  ∀ x : ℝ, |x| = |(-x)| :=
by
  intro x
  sorry

end graph_abs_symmetric_yaxis_l463_463328


namespace ratio_of_increase_to_current_l463_463560

-- Define the constants for the problem
def current_deductible : ℝ := 3000
def increase_deductible : ℝ := 2000

-- State the theorem that needs to be proven
theorem ratio_of_increase_to_current : 
  (increase_deductible / current_deductible) = (2 / 3) :=
by sorry

end ratio_of_increase_to_current_l463_463560


namespace asphalt_length_l463_463437

/-- Definition of man-hours calculation for a group of men working certain hours a day over some days -/
def man_hours (men : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  men * days * hours_per_day

/-- Definition of the problem conditions -/
def conditions := man_hours 30 12 8 = 2880 ∧ man_hours 20 (floor 28.8) 10 = 5760

/-- Proving the length of the road equivalent to the given conditions -/
theorem asphalt_length : conditions → ∃ (length : ℕ), length = 2 :=
by
  intro h
  sorry

end asphalt_length_l463_463437


namespace mark_wait_time_l463_463369

theorem mark_wait_time (t1 t2 T : ℕ) (h1 : t1 = 4) (h2 : t2 = 20) (hT : T = 38) : 
  T - (t1 + t2) = 14 :=
by sorry

end mark_wait_time_l463_463369


namespace function_increasing_in_interval_A_l463_463043

noncomputable def f (x : ℝ) : ℝ := x * Math.sin x + Math.cos x

def interval_A : Set ℝ := { x | (3 * Real.pi / 2) < x ∧ x < (5 * Real.pi / 2) }

theorem function_increasing_in_interval_A :
  ∀ x ∈ interval_A, 0 < (x * Math.cos x):=
by
  sorry

end function_increasing_in_interval_A_l463_463043


namespace blue_red_face_area_ratio_l463_463815

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l463_463815


namespace percentage_gain_l463_463095

theorem percentage_gain : 
  let CP := 300 * 20 in
  let SP1 := 200 * 25 in
  let SP2 := 80 * 30 in
  let Total_SP := SP1 + SP2 in
  let Profit := Total_SP - CP in
  let Percentage_Gain := (Profit / CP) * 100 in
  Percentage_Gain = 23.33
:= by
  sorry

end percentage_gain_l463_463095


namespace van_should_maintain_new_speed_l463_463543

-- Define the original distance and time
def original_distance : ℝ := 600
def original_time : ℝ := 5

-- Define the fraction of the original time
def time_fraction : ℝ := 3 / 2

-- Define the new time based on the fraction of the original time
def new_time : ℝ := time_fraction * original_time

-- Define the new speed which we are claiming to be 80 km/h
def new_speed : ℝ := original_distance / new_time

-- Now we state the theorem saying what the new speed should be
theorem van_should_maintain_new_speed :
  new_speed = 80 := by
  sorry

end van_should_maintain_new_speed_l463_463543


namespace students_play_both_l463_463766

def students := 450
def play_football := 325
def play_cricket := 175
def play_neither := 50

theorem students_play_both :
  (students - play_neither) = (play_football + play_cricket - 100) :=
by
  have total_students := 450 - 50
  have play_either := 325 + 175
  have students_play_both := play_either - 100
  show total_students = students_play_both
  sorry

end students_play_both_l463_463766


namespace arithmetic_sequence_fraction_l463_463214

variable {α : Type*} [LinearOrderedField α]
variable (a b : ℕ → α)
variable (S T : ℕ → α)

-- Define the arithmetic sequence property
def is_arithmetic_sequence (seq : ℕ → α) (d : α) :=
  ∀ n m, seq (n + m) = seq n + m * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (seq : ℕ → α) (d : α) (S : ℕ → α) :=
  S 0 = 0 ∧ ∀ n, S n = S (n - 1) + seq n

-- Given condition
axiom arithmetic_sum_ratio (hS : is_arithmetic_sequence a d) (hT : is_arithmetic_sequence b d') (hS_sum : sum_arithmetic_sequence a d S) (hT_sum : sum_arithmetic_sequence b d' T) :
  ∀ n, (S n) / (T n) = (2 * n + 1 : α) / (3 * n + 2 : α)

noncomputable def desired_fraction : α :=
  (a 2 + a 5 + a 17 + a 22) / (b 8 + b 10 + b 12 + b 16)

-- The main theorem to be proved
theorem arithmetic_sequence_fraction (hS : is_arithmetic_sequence a d) (hT : is_arithmetic_sequence b d') (hS_sum : sum_arithmetic_sequence a d S) (hT_sum : sum_arithmetic_sequence b d' T) :
  desired_fraction a b = (45 : α) / (68 : α) := by
  sorry

end arithmetic_sequence_fraction_l463_463214


namespace sum_of_valid_c_l463_463205

theorem sum_of_valid_c : 
  let discriminant (c : ℕ) := 49 - 12 * c in
  (∀ (c : ℕ), (3 * x^2 + 7 * x + c = 0) → (∃ k : ℕ, discriminant c = k^2)) →
  (∑ c in (finset.filter (λ c, (∃ k : ℕ, discriminant c = k^2) ∧ c > 0 ∧ c < 5) (finset.range 5)), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463205


namespace calculate_number_of_girls_l463_463691

-- Definitions based on the conditions provided
def ratio_girls_to_boys : ℕ := 3
def ratio_boys_to_girls : ℕ := 4
def total_students : ℕ := 35

-- The proof statement
theorem calculate_number_of_girls (k : ℕ) (hk : ratio_girls_to_boys * k + ratio_boys_to_girls * k = total_students) :
  ratio_girls_to_boys * k = 15 :=
by sorry

end calculate_number_of_girls_l463_463691


namespace max_roads_to_close_l463_463693

theorem max_roads_to_close (n : ℕ) (h₁ : n = 30) (h₂ : ∀ i j, i ≠ j → connected i j) : ∃ E_closed, E_closed = 406 :=
by 
  sorry

end max_roads_to_close_l463_463693


namespace probability_of_winning_noughts_l463_463516

def ticTacToeProbability : ℚ :=
  let totalWays := Nat.choose 9 3
  let winningWays := 8
  winningWays / totalWays

theorem probability_of_winning_noughts :
  ticTacToeProbability = 2 / 21 := by
  sorry

end probability_of_winning_noughts_l463_463516


namespace wizard_elixir_combinations_l463_463107

theorem wizard_elixir_combinations :
  (let roots := 4 in
   let minerals := 6 in
   let incompatible_combinations := 3 in
   roots * minerals - incompatible_combinations = 21) :=
by
  let roots := 4
  let minerals := 6
  let incompatible_combinations := 3
  show roots * minerals - incompatible_combinations = 21
  sorry

end wizard_elixir_combinations_l463_463107


namespace ratio_one_six_to_five_eighths_l463_463871

theorem ratio_one_six_to_five_eighths : (1 / 6) / (5 / 8) = 4 / 15 := by
  sorry

end ratio_one_six_to_five_eighths_l463_463871


namespace sum_of_numbers_l463_463308

theorem sum_of_numbers (avg : ℝ) (count : ℕ) (h_avg : avg = 5.7) (h_count : count = 8) : (avg * count = 45.6) :=
by
  sorry

end sum_of_numbers_l463_463308


namespace blue_to_red_face_area_ratio_l463_463852

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l463_463852


namespace sandy_age_l463_463472

-- Define the ages of Sandy and Molly
variables (S M : ℕ)

-- Define conditions from the problem
def condition1 : Prop := M = S + 20
def condition2 : Prop := S * 9 = 7 * M

-- Theorem statement: Sandy's age is 70 if conditions are met
theorem sandy_age (h1 : condition1) (h2 : condition2) : S = 70 :=
by sorry

end sandy_age_l463_463472


namespace initial_group_size_l463_463405

theorem initial_group_size (n : ℕ) (h : 34 = 8.5 * n) : n = 4 := 
by sorry

end initial_group_size_l463_463405


namespace diagonals_same_ratio_l463_463610

variable {Point : Type} [AffineSpace Point ℝ]

-- Define quadrilateral
variables (A B C D : Point)
-- Define orthocenters
variables (A' B' C' D' : Point)

-- Hypothesis: A', B', C', D' are orthocentres of respective triangles
def orthocenter (O : Point) (X Y Z : Point) : Prop :=
  -- Placeholder definition as we focus on the statement
  sorry

variables 
  (h_A' : orthocenter A' B C D)
  (h_B' : orthocenter B' C D A)
  (h_C' : orthocenter C' D A B)
  (h_D' : orthocenter D' A B C)

-- Theorem statement: Diagonals of ABCD and A'B'C'D' are divided in the same ratio
theorem diagonals_same_ratio :
  ∃ P Q : Point, ∃ a b : ℝ,
    (P ∈ line_through A C) ∧ (P ∈ line_through B D) ∧
    (Q ∈ line_through A' C') ∧ (Q ∈ line_through B' D') ∧
    P ≠ Q ∧
    (divides P a b A C ∧ divides Q a b A' C') ∧
    (divides P a b B D ∧ divides Q a b B' D') :=
sorry

end diagonals_same_ratio_l463_463610


namespace max_area_CDFE_l463_463727

-- Define the square and the points on the sides
variables {A B C D E F: Type} [hA: HasCoe ℝ A] [hB: HasCoe ℝ B] [hC: HasCoe ℝ C] [hD: HasCoe ℝ D]
[hE: HasCoe ℝ E] [hF: HasCoe ℝ F]

-- Condition: ABCD is a square with side length 3.
def square_side_length : ℝ := 3

-- Condition: Points E and F on sides AB and AD, respectively, such that AE = 2AF
def AE (AF: ℝ) := 2 * AF

-- Conclusion: Maximum area of quadrilateral CDFE is 9/2
theorem max_area_CDFE : (∀ AF, 0 ≤ AF ∧ AF ≤ square_side_length → 
    ∃ AE, AE = 2 * AF → (let area := (1 / 2) * (square_side_length - AF) * (square_side_length - AF)
    + (1 / 2) * AF * (square_side_length - 2 * AF) in
        area ≤ 9 / 2)) :=
begin
  sorry
end

end max_area_CDFE_l463_463727


namespace expression_evaluation_l463_463332

theorem expression_evaluation : 
  let y := (1 / 2 : ℝ)
  in ∀ x : ℝ, x = (x + 1) / (x - 1) → x = y → 
     ( ((x + 1)/(x - 1) + 1) / ((x + 1)/(x - 1) - 1) ) = -3 :=
by
  sorry

end expression_evaluation_l463_463332


namespace intersection_of_lines_l463_463183

theorem intersection_of_lines :
  ∃ x y : ℝ, 8 * x - 5 * y = 10 ∧ 6 * x + 2 * y = 22 ∧ x = 65 / 23 ∧ y = -137 / 23 :=
by { use [65 / 23, -137 / 23], split, { norm_num }, split, { norm_num }, split, { refl }, { refl } }
  sorry

end intersection_of_lines_l463_463183


namespace cos_4theta_value_l463_463303

theorem cos_4theta_value (theta : ℝ) 
  (h : ∑' n : ℕ, (Real.cos theta)^(2 * n) = 8) : 
  Real.cos (4 * theta) = 1 / 8 := 
sorry

end cos_4theta_value_l463_463303


namespace no_positive_integer_solution_l463_463381

def is_solution (x y z t : ℕ) : Prop :=
  x^2 + 5 * y^2 = z^2 ∧ 5 * x^2 + y^2 = t^2

theorem no_positive_integer_solution :
  ¬ ∃ (x y z t : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ is_solution x y z t :=
by
  sorry

end no_positive_integer_solution_l463_463381


namespace ellipse_intersection_range_a_l463_463969

theorem ellipse_intersection_range_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 →
       (∃ x1 x2 y1 y2 : ℝ, x1 - y1 = 1 ∧ x2 - y2 = 1 ∧
        0 = x1 * x2 + (x1 - 1) * (x2 - 1) ∧
        x1 < x ∧ x < x2 ∧ y1 < y ∧ y < y2))) :
  ( ∀ ε : ℝ, ε = (∃ y1, a^2 + b^2 = 2 * a^2 * b^2  ∧ 
        ( x1 + x2  =  2 * a^2 / (a^2 + b^2)) ∧ 
        ( x1 * x2  =  (a^2 - a^2 * b^2) / (a^2 + b^2) ) ∧ 
          x1 * x2 + (x1 - 1)*(x2- 1) ∧ (a + b > 0))  )   :
  ( ∃ a1,   sqr (5) / 2 ≤ a ∧ a ≤ sqr (6) / 2 )) =>
sorry

end ellipse_intersection_range_a_l463_463969


namespace minimum_decimal_digits_30_l463_463026

noncomputable def minimum_decimal_digits : ℕ :=
  (987654321 : ℚ) / (2^30 * 5^6)

theorem minimum_decimal_digits_30 :
  minimum_decimal_digits.digits_after_decimal = 30 :=
sorry

end minimum_decimal_digits_30_l463_463026


namespace old_lamp_height_is_one_l463_463341

def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := 1.3333333333333333
def old_lamp_height : ℝ := new_lamp_height - height_difference

theorem old_lamp_height_is_one :
  old_lamp_height = 1 :=
by
  sorry

end old_lamp_height_is_one_l463_463341


namespace number_of_lines_at_distance_two_l463_463334

theorem number_of_lines_at_distance_two (l : ℝ → ℝ) :
  ∃ l1 l2 : ℝ → ℝ, 
    ∀ x : ℝ, 
      abs ((l1 x - l x) / (1 + l x * l1 x)) = 2 ∧
      abs ((l2 x - l x) / (1 + l x * l2 x)) = 2 :=
begin
  sorry
end

end number_of_lines_at_distance_two_l463_463334


namespace sum_of_squares_cos_sin_l463_463135

theorem sum_of_squares_cos_sin (deg : ℕ) (h : 30 ≤ deg ∧ deg ≤ 60) :
  ∑ θ in (finset.range (61 - 30)).map (finset.range 30).add, (cos (θ * π / 180))^2 + (sin (θ * π / 180))^2 = 62 := 
by
  sorry

end sum_of_squares_cos_sin_l463_463135


namespace bangles_per_box_l463_463465

-- Define the total number of pairs of bangles
def totalPairs : Nat := 240

-- Define the number of boxes
def numberOfBoxes : Nat := 20

-- Define the proof that each box can hold 24 bangles
theorem bangles_per_box : (totalPairs * 2) / numberOfBoxes = 24 :=
by
  -- Here we're required to do the proof but we'll use 'sorry' to skip it
  sorry

end bangles_per_box_l463_463465


namespace gear_angular_speed_proportion_l463_463218

theorem gear_angular_speed_proportion :
  ∀ (ω_A ω_B ω_C ω_D k: ℝ),
    30 * ω_A = k →
    45 * ω_B = k →
    50 * ω_C = k →
    60 * ω_D = k →
    ω_A / ω_B = 1 ∧
    ω_B / ω_C = 45 / 50 ∧
    ω_C / ω_D = 50 / 60 ∧
    ω_A / ω_D = 10 / 7.5 :=
  by
    -- proof goes here
    sorry

end gear_angular_speed_proportion_l463_463218


namespace problem_l463_463666

variables {a b : ℝ}

theorem problem (h₁ : -1 < a) (h₂ : a < b) (h₃ : b < 0) : 
  (1/a > 1/b) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a + (1/a) > b + (1/b)) :=
by
  sorry

end problem_l463_463666


namespace C2A_hex_is_300222_base4_l463_463987

-- Define base conversion functions
def hex_to_dec : char → ℕ
| '0' := 0
| '1' := 1
| '2' := 2
| '3' := 3
| '4' := 4
| '5' := 5
| '6' := 6
| '7' := 7
| '8' := 8
| '9' := 9
| 'A' := 10
| 'B' := 11
| 'C' := 12
| 'D' := 13
| 'E' := 14
| 'F' := 15
| _ := 0  -- This case handles invalid input, it's not necessary for correct problems

def dec_to_base4 : ℕ → list ℕ
| 0 := []
| n := dec_to_base4 (n / 4) ++ [n % 4]

-- Convert hex digit to base 4 digits
def hex_to_base4 (c : char) : list ℕ :=
  dec_to_base4 (hex_to_dec c)

-- Convert a list of hex digits to a list of base 4 digits
def hex_list_to_base4 (l : list char) : list ℕ :=
  l.bind hex_to_base4

-- Convert hex "C2A" to base 4
def C2A_base4 : list ℕ :=
  hex_list_to_base4 ['C', '2', 'A']

-- Representation of 300222 in base 4 as a list
def base4_300222 : list ℕ := [3, 0, 0, 2, 2, 2]

-- Theorem stating the equivalence
theorem C2A_hex_is_300222_base4 : C2A_base4 = base4_300222 :=
by
  -- The proof would involve showing each step but is omitted here
  sorry

end C2A_hex_is_300222_base4_l463_463987


namespace angle_in_third_quadrant_l463_463557

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2023) :
    ∃ k : ℤ, (2023 - k * 360) = 223 ∧ 180 ≤ 223 ∧ 223 < 270 := by
sorry

end angle_in_third_quadrant_l463_463557


namespace geometric_sequence_common_ratio_l463_463248

noncomputable def common_ratio_q (a1 a5 a : ℕ) (q : ℕ) : Prop :=
  a1 * a5 = 16 ∧ a1 > 0 ∧ a5 > 0 ∧ a = 2 ∧ q = 2

theorem geometric_sequence_common_ratio : ∀ (a1 a5 a q : ℕ), 
  common_ratio_q a1 a5 a q → q = 2 :=
by
  intros a1 a5 a q h
  have h1 : a1 * a5 = 16 := h.1
  have h2 : a1 > 0 := h.2.1
  have h3 : a5 > 0 := h.2.2.1
  have h4 : a = 2 := h.2.2.2.1
  have h5 : q = 2 := h.2.2.2.2
  exact h5

end geometric_sequence_common_ratio_l463_463248


namespace arithmetic_seq_first_term_l463_463412

def sum_of_squares (a d : ℤ) (n : ℕ) : ℤ :=
  n * a^2 + 2 * a * d * (n * (n - 1) / 2) + d^2 * (n * (n - 1) * (2 * n - 1) / 6)

theorem arithmetic_seq_first_term (a : ℤ) :
  let d := 3 in
  sum_of_squares a d 1001 = sum_of_squares a d 1000 →
  a = -3000 ∨ a = 6003000 :=
by
  intros d h
  sorry

end arithmetic_seq_first_term_l463_463412


namespace num_terms_arith_seq_l463_463663

theorem num_terms_arith_seq {a d t : ℕ} (h_a : a = 5) (h_d : d = 3) (h_t : t = 140) :
  ∃ n : ℕ, t = a + (n-1) * d ∧ n = 46 :=
by
  sorry

end num_terms_arith_seq_l463_463663


namespace removed_tetrahedra_volume_correct_l463_463147

def edge_length : ℝ := 2
def removed_tetrahedra_volume : ℝ := (4 * real.sqrt 3) / 27

theorem removed_tetrahedra_volume_correct :
  ∀ (original_edge_length : ℝ),
  original_edge_length = edge_length →
  total_removed_volume original_edge_length = removed_tetrahedra_volume :=
by
  intros original_edge_length h_edge_length_eq
  -- Proof omitted
  sorry

end removed_tetrahedra_volume_correct_l463_463147


namespace common_tangent_slope_eq_l463_463684

theorem common_tangent_slope_eq
  (f g : ℝ → ℝ)
  (l : ℝ → ℝ)
  (t : ℝ)
  (h_f : ∀ x, f x = t * Real.log x)
  (h_g : ∀ x, g x = x^2 - 1)
  (h_l : ∀ x, l x = 2 * (x - 1))
  (common_point : (f 1 = 0 ∧ g 1 = 0))
  (common_tangent : (forall x, (f x = g x) → (f' x = g' x))) 
  : t = 2 :=
sorry

end common_tangent_slope_eq_l463_463684


namespace euclidean_algorithm_division_count_l463_463285

/--
Using the Euclidean algorithm to find the greatest common divisor of 294 and 84
requires performing exactly 2 divisions.
-/
theorem euclidean_algorithm_division_count : 
  let a := 294
  let b := 84
  euclidean_div_count a b = 2 :=
by
  sorry

end euclidean_algorithm_division_count_l463_463285


namespace sum_of_valid_c_l463_463202

theorem sum_of_valid_c : 
  let discriminant (c : ℕ) := 49 - 12 * c in
  (∀ (c : ℕ), (3 * x^2 + 7 * x + c = 0) → (∃ k : ℕ, discriminant c = k^2)) →
  (∑ c in (finset.filter (λ c, (∃ k : ℕ, discriminant c = k^2) ∧ c > 0 ∧ c < 5) (finset.range 5)), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463202


namespace find_a2009_l463_463874

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, 0 < n → 2 * (Finset.sum (Finset.range (n + 1)) a) = a n + (a n) ^ 2

theorem find_a2009 (a : ℕ → ℕ) (h1 : sequence a) : a 2009 = 2009 :=
sorry

end find_a2009_l463_463874


namespace blue_faces_ratio_l463_463850

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l463_463850


namespace inclination_angle_range_l463_463424

theorem inclination_angle_range (x : ℝ) (k : ℝ) (α : ℝ) 
  (h1 : k = x^2 + 1) 
  (h2 : k = Real.tan α) 
  : ∃ α, α ∈ Icc (Real.pi / 4) (Real.pi / 2) :=
by
  sorry

end inclination_angle_range_l463_463424


namespace find_f_one_third_l463_463268

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : f 1 = 1
axiom cond2 : ∀ (x : ℝ), 0 < x ∧ x < 1 → f(x) > 0
axiom cond3 : ∀ (x y : ℝ), f(x + y) - f(x - y) = 2 * f(1 - x) * f(y)

theorem find_f_one_third : f (1/3) = 1/2 :=
by
  sorry

end find_f_one_third_l463_463268


namespace actual_time_when_watch_reads_8_pm_l463_463981

-- Define the initial conditions and the problem statement
theorem actual_time_when_watch_reads_8_pm :
  ∀ (C : Type) [Inhabited C] (noon_time : C)
    (actual_time_at_2pm : C)
    (watch_time_at_actual_2pm : C)
    (watch_time_at_8pm : C)
    (read_time_at_2pm : (nat × nat × nat))
    (time_difference : C → C → C)
    (constant_rate_loss : (nat × nat × nat) → (nat × nat × nat) → Prop),
    noon_time = noon_time
    ∧ actual_time_at_2pm = time_difference noon_time (2 * 60)
    ∧ watch_time_at_actual_2pm = time_difference noon_time (1 * 60 + 56 * 1 + 12 * (1 / 60))
    ∧ constant_rate_loss (2 * 60, 1 * 60 + 56, 12)
    ∧ watch_time_at_8pm = time_difference noon_time (8 * 60) ->
    actual_time_when_watch_reads_8_pm = noon_time + time_difference noon_time (8 * 60 + 18) := 
by sorry

end actual_time_when_watch_reads_8_pm_l463_463981


namespace jason_total_expenditure_l463_463348

theorem jason_total_expenditure :
  let stove_cost := 1200
  let wall_repair_ratio := 1 / 6
  let wall_repair_cost := stove_cost * wall_repair_ratio
  let total_cost := stove_cost + wall_repair_cost
  total_cost = 1400 := by
  {
    sorry
  }

end jason_total_expenditure_l463_463348


namespace magnitude_of_vector_difference_l463_463660

variables (a b : ℝ^3)
variables (h1 : dot_product a b = 0)
variables (h2 : ‖a‖ = 2)
variables (h3 : ‖b‖ = 3)

theorem magnitude_of_vector_difference : ‖3 • a - 2 • b‖ = 6 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_vector_difference_l463_463660


namespace peregrine_falcon_dive_time_l463_463407

theorem peregrine_falcon_dive_time
  (bald_eagle_speed : ℕ)
  (peregrine_factor : ℕ)
  (bald_eagle_time_sec : ℕ)
  (peregrine_speed := peregrine_factor * bald_eagle_speed)
  (hours_to_seconds := 3600)
  (bald_eagle_time_hour := bald_eagle_time_sec.to_real / hours_to_seconds)
  (dive_distance := bald_eagle_speed.to_real * bald_eagle_time_hour) :
  (dive_distance / peregrine_speed) * hours_to_seconds = 15 := 
by
  sorry

end peregrine_falcon_dive_time_l463_463407


namespace minimum_value_of_g_l463_463745

noncomputable def g (a b x : ℝ) : ℝ :=
  max (|x + a|) (|x + b|)

theorem minimum_value_of_g (a b : ℝ) (h : a < b) :
  ∃ x : ℝ, g a b x = (b - a) / 2 :=
by
  use - (a + b) / 2
  sorry

end minimum_value_of_g_l463_463745


namespace quadrilateral_proofs_l463_463463

def is_parallelogram (ABCD : Quadrilateral) : Prop :=
  -- Assuming the definition of a parallelogram in Lean

def statement_1 (ABCD : Quadrilateral) : Prop :=
  ABCD.has_one_pair_opposite_sides_parallel ∧ ABCD.has_one_pair_opposite_sides_equal

def statement_2 (ABCD : Quadrilateral) : Prop :=
  ABCD.has_one_pair_opposite_sides_parallel ∧ ABCD.has_one_pair_opposite_angles_equal

def statement_3 (ABCD : Quadrilateral) : Prop :=
  ABCD.has_one_pair_opposite_sides_equal ∧ ABCD.has_one_pair_opposite_angles_equal

def statement_4 (ABCD : Quadrilateral) : Prop :=
  ABCD.has_both_pairs_opposite_angles_equal

theorem quadrilateral_proofs (ABCD : Quadrilateral) :
  (¬ statement_1 ABCD → ¬ is_parallelogram ABCD) ∧
  (statement_2 ABCD → is_parallelogram ABCD) ∧
  (¬ statement_3 ABCD → ¬ is_parallelogram ABCD) ∧
  (statement_4 ABCD → is_parallelogram ABCD) :=
by
  sorry

end quadrilateral_proofs_l463_463463


namespace blue_red_face_area_ratio_l463_463817

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l463_463817


namespace ott_fraction_is_3_over_13_l463_463752

-- Defining the types and quantities involved
noncomputable def moes_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def lokis_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def nicks_original_money (amount_given: ℚ) := amount_given * 3

-- Total original money of the group (excluding Ott)
noncomputable def total_original_money (amount_given: ℚ) :=
  moes_original_money amount_given + lokis_original_money amount_given + nicks_original_money amount_given

-- Total money received by Ott
noncomputable def otts_received_money (amount_given: ℚ) := 3 * amount_given

-- Fraction of the group's total money Ott now has
noncomputable def otts_fraction_of_total_money (amount_given: ℚ) : ℚ :=
  otts_received_money amount_given / total_original_money amount_given

-- The theorem to be proved
theorem ott_fraction_is_3_over_13 :
  otts_fraction_of_total_money 1 = 3 / 13 :=
by
  -- The body of the proof is skipped with sorry
  sorry

end ott_fraction_is_3_over_13_l463_463752


namespace tic_tac_toe_winning_probability_l463_463530

theorem tic_tac_toe_winning_probability :
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  probability = (2 : ℚ) / 21 :=
by
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  have probability_correct : probability = (2 : ℚ) / 21 := sorry
  exact probability_correct

end tic_tac_toe_winning_probability_l463_463530


namespace sqrt_floor_square_18_l463_463998

-- Condition: the sqrt function and floor function
def sqrt (x : ℝ) : ℝ := Real.sqrt x
def floor (x : ℝ) : ℤ := Int.floor x

-- Mathematically equivalent proof problem
theorem sqrt_floor_square_18 : floor (sqrt 18) ^ 2 = 16 := 
by
  sorry

end sqrt_floor_square_18_l463_463998


namespace max_S_n_l463_463254

theorem max_S_n (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 3 + a 5 = 105) (h2 : a 2 + a 4 + a 6 = 99) :
  let S_n := λ n, (n / 2) * (2 * a 1 + (n - 1) * d) in
  ∃ n, S_n n = 400 ∧ (∀ m, S_n m ≤ 400) :=
by
  sorry

end max_S_n_l463_463254


namespace max_xy_l463_463299

theorem max_xy (x y : ℕ) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 :=
sorry

end max_xy_l463_463299


namespace solve_for_diamond_l463_463784

theorem solve_for_diamond (d : ℤ) (h : d * 9 + 5 = d * 10 + 2) : d = 3 :=
by
  sorry

end solve_for_diamond_l463_463784


namespace range_of_a_l463_463920

def set_A (a : ℝ) : set ℝ := {x | |x - a| ≤ 1}
def set_B : set ℝ := {x | x ^ 2 - 5 * x + 4 ≥ 0}

theorem range_of_a (a : ℝ) (h : set_A a ∩ set_B = ∅) : 2 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l463_463920


namespace ranking_of_anna_bella_carol_l463_463559

-- Define three people and their scores
variables (Anna Bella Carol : ℕ)

-- Define conditions based on problem statements
axiom Anna_not_highest : ∃ x : ℕ, x > Anna
axiom Bella_not_lowest : ∃ x : ℕ, x < Bella
axiom Bella_higher_than_Carol : Bella > Carol

-- The theorem to be proven
theorem ranking_of_anna_bella_carol (h : Anna < Bella ∧ Carol < Anna) :
  (Bella > Anna ∧ Anna > Carol) :=
by sorry

end ranking_of_anna_bella_carol_l463_463559


namespace area_of_triangle_AEB_l463_463327

-- Definitions based on the given conditions
def rectangle (A B C D : Type) := A = B ∧ B = C ∧ C = D ∧ D = A
def length_AB : ℝ := 6
def length_BC : ℝ := 4
def length_CD : ℝ := 6
def length_DA : ℝ := 4
def length_DF : ℝ := 2
def length_GC : ℝ := 1.5
def F (D C : Type) := D = C -- F is on CD
def G (C D : Type) := C = D -- G is on CD
def E (A B D C F G : Type) := intersect_lines A F B G -- intersection of AF and BG

-- The target theorem to state and prove
theorem area_of_triangle_AEB (A B C D F G E : Type) (length_AB length_BC length_CD length_DA length_DF length_GC : ℝ) : 
  let area_AEB := (1 / 2) * ((20 / 7) + 4) * 6 in
  area_AEB = 144 / 7 :=
  sorry

end area_of_triangle_AEB_l463_463327


namespace complement_A_when_a_5_union_A_B_when_a_2_l463_463624

noncomputable def U : Set ℝ := Set.univ

def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 1}

def B : Set ℝ := {x | x < 0 ∨ x > 5}

-- Proof Problem 1: Complement of A when a = 5
theorem complement_A_when_a_5 : A 5ᶜ = {x | x < 4 ∨ x > 11} := sorry

-- Proof Problem 2: Union of A and B when a = 2
theorem union_A_B_when_a_2 : A 2 ∪ B = {x | x < 0 ∨ x ≥ 1} := sorry

end complement_A_when_a_5_union_A_B_when_a_2_l463_463624


namespace prob_complex_abs_l463_463607

-- Definitions based on the conditions
variables (a b : ℝ)
def z : ℂ := a + b * Complex.i
def given_eq : ℂ := (1 + Complex.i) * (1 - a * Complex.i)

-- The theorem to prove
theorem prob_complex_abs : given_eq = b + 2 * Complex.i → |z| = 1 :=
by
  sorry

end prob_complex_abs_l463_463607


namespace num_words_with_at_least_two_vowels_l463_463662

def letters := {A, B, C, D, E, F}
def vowels := {A, E}
def consonants := {B, C, D, F}
def len := 5

theorem num_words_with_at_least_two_vowels : 
  (∀ word : vector letters len, (cardinal.mk {w ∈ word | w ∈ vowels} ≥ 2)) ↔ 4192 :=
by
  sorry

end num_words_with_at_least_two_vowels_l463_463662


namespace find_b_l463_463790

def fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| n+2 := fibonacci (n+1) + fibonacci n

theorem find_b (b : ℕ) (F : ℕ → ℕ) 
  (hF1 : F 1 = 1) 
  (hF2 : F 2 = 1) 
  (hFn : ∀ n ≥ 3, F n = F (n-1) + F (n-2)) 
  (hPattern : ∀ a b c, (a, b, c) = (b-3, b, b+3)) 
  (hSum : ∀ a b c, a + b + c = 2253) :
  b = 751 := 
sorry

end find_b_l463_463790


namespace num_arrangements_correct_l463_463282

def num_arrangements : ℕ :=
  fact 8 / (fact 2 * fact 1 * fact 3 * fact 2)

theorem num_arrangements_correct :
  num_arrangements = 1680 := by
  sorry

end num_arrangements_correct_l463_463282


namespace largest_sum_of_digits_l463_463671

theorem largest_sum_of_digits :
  ∃ (a b c z : ℕ), (a ≤ 9) ∧ (b ≤ 9) ∧ (c ≤ 9) ∧ (0 < z ∧ z ≤ 12) ∧ (0.abc = 1 / (z : ℚ)) ∧ 
  (∀ a' b' c' z', (a' ≤ 9) ∧ (b' ≤ 9) ∧ (c' ≤ 9) ∧ (0 < z' ∧ z' ≤ 12) ∧ (0.abc = 1 / (z' : ℚ)) → (a + b + c ≥ a' + b' + c')) :=
by
  sorry

end largest_sum_of_digits_l463_463671


namespace gcd_three_digit_palindromes_l463_463889

theorem gcd_three_digit_palindromes : 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → 
  ∃ d : ℕ, d = 1 ∧ ∀ n m : ℕ, (n = 101 * a + 10 * b) → (m = 101 * a + 10 * b) → gcd n m = d := 
by sorry

end gcd_three_digit_palindromes_l463_463889


namespace surface_area_increase_l463_463958

theorem surface_area_increase :
  let l := 4
  let w := 3
  let h := 2
  let side_cube := 1
  let original_surface := 2 * (l * w + l * h + w * h)
  let additional_surface := 6 * side_cube * side_cube
  let new_surface := original_surface + additional_surface
  new_surface = original_surface + 6 :=
by
  sorry

end surface_area_increase_l463_463958


namespace curve_is_line_l463_463590

theorem curve_is_line (r θ : ℝ) : 
  (∃ (x y : ℝ), r = sqrt(x^2 + y^2) 
    ∧ θ = arctan(y / x)
    ∧ r = 1 / (1 + sin(θ))) 
  → (∃ (a b c : ℝ), a * x + b * y = c) :=
by
  sorry

end curve_is_line_l463_463590


namespace abigail_initial_money_l463_463113

variables (A : ℝ) -- Initial amount of money Abigail had.

-- Conditions
variables (food_rate : ℝ := 0.60) -- 60% spent on food
variables (phone_rate : ℝ := 0.25) -- 25% of the remainder spent on phone bill
variables (entertainment_spent : ℝ := 20) -- $20 spent on entertainment
variables (final_amount : ℝ := 40) -- $40 left after all expenditures

theorem abigail_initial_money :
  (A - (A * food_rate)) * (1 - phone_rate) - entertainment_spent = final_amount → A = 200 :=
by
  intro h
  sorry

end abigail_initial_money_l463_463113


namespace eval_expression_l463_463158

theorem eval_expression : 
  let a := 2999 in
  let b := 3000 in
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 := 
by 
  sorry

end eval_expression_l463_463158


namespace domain_of_sqrt_l463_463149

theorem domain_of_sqrt (x : ℝ) : (x - 1 ≥ 0) → (x ≥ 1) :=
by
  sorry

end domain_of_sqrt_l463_463149


namespace product_of_two_numbers_l463_463870

theorem product_of_two_numbers (A B : ℕ) (hcf : ℕ := 22) (lcm : ℕ := 2828) :
  (nat.gcd A B = hcf) → (nat.lcm A B = lcm) → A * B = hcf * lcm :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  exact rfl

end product_of_two_numbers_l463_463870


namespace max_distinct_terms_degree_6_l463_463024

-- Step 1: Define the variables and conditions
def polynomial_max_num_terms (deg : ℕ) (vars : ℕ) : ℕ :=
  Nat.choose (deg + vars - 1) (vars - 1)

-- Step 2: State the specific problem
theorem max_distinct_terms_degree_6 :
  polynomial_max_num_terms 6 5 = 210 :=
by
  sorry

end max_distinct_terms_degree_6_l463_463024


namespace algebraic_expression_value_l463_463035

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a + 2 * b = 2) : 
  (-a * (-2) ^ 2 + b * (-2) + 1) = -1 :=
by
  sorry

end algebraic_expression_value_l463_463035


namespace angle_in_third_quadrant_l463_463555

theorem angle_in_third_quadrant (α : ℝ) (h : α = 2023) : 180 < α % 360 ∧ α % 360 < 270 := by
  sorry

end angle_in_third_quadrant_l463_463555


namespace sum_of_digits_l463_463921

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 3 + 984 = 1 * 1000 + 3 * 100 + b * 10 + 7)
  (h2 : (1 + b) - (3 + 7) % 11 = 0) : a + b = 10 := 
by
  sorry

end sum_of_digits_l463_463921


namespace mutually_exclusive_union_independent_intersection_independent_complement_l463_463237

variables (A B : Event) (PA PB : ℝ)
variable (mutually_exclusive : Prop)
variable (independent : Prop)

-- Given conditions
axiom PA_eq : PA = 0.4
axiom PB_eq : PB = 0.1

-- Definitions for the Lean 4 problem

-- Mutually exclusive events definition
def mutually_exclusive_def (A B : Event) : Prop :=
  P(A ∩ B) = 0

-- Independent events definition
def independent_def (A B : Event) : Prop :=
  P(A ∩ B) = P(A) * P(B)

-- Theorems to be proved
theorem mutually_exclusive_union (h : mutually_exclusive_def A B) : P(A ∪ B) = PA + PB :=
by sorry

theorem independent_intersection (h : independent_def A B) : P(A ∩ B) = PA * PB :=
by sorry

theorem independent_complement (h : independent_def A B) : P(-(A ∩ B)) = 1 - PA * PB :=
by sorry

end mutually_exclusive_union_independent_intersection_independent_complement_l463_463237


namespace incorrect_conclusion_D_l463_463703

-- Define lines and planes
variables (l m n : Type) -- lines
variables (α β γ : Type) -- planes

-- Define the conditions
def intersection_planes (p1 p2 : Type) : Type := sorry
def perpendicular (a b : Type) : Prop := sorry

-- Given conditions for option D
axiom h1 : intersection_planes α β = m
axiom h2 : intersection_planes β γ = l
axiom h3 : intersection_planes γ α = n
axiom h4 : perpendicular l m
axiom h5 : perpendicular l n

-- Theorem stating that the conclusion of option D is incorrect
theorem incorrect_conclusion_D : ¬ perpendicular m n :=
by sorry

end incorrect_conclusion_D_l463_463703


namespace find_a_l463_463396

def f (x : ℝ) : ℝ := (2 * x) / 3 + 4
def g (x : ℝ) : ℝ := 5 - 2 * x
def a := 13 / 4

theorem find_a : f (g a) = 3 := by
  sorry

end find_a_l463_463396


namespace absolute_value_equality_l463_463785

variables {a b c d : ℝ}

theorem absolute_value_equality (h1 : |a - b| + |c - d| = 99) (h2 : |a - c| + |b - d| = 1) : |a - d| + |b - c| = 99 :=
sorry

end absolute_value_equality_l463_463785


namespace coefficient_of_x3_is_26_l463_463177

-- Define the given expression
def expression : ℤ[X] := 
  3 * (X^2 - X^3 + X) + 3 * (X + 2 * X^3 - 3 * X^2 + 3 * X^5 + X^3) - 5 * (1 + X - 4 * X^3 - X^2)

-- Statement of the proof problem
theorem coefficient_of_x3_is_26 : (expression.coeff 3) = 26 :=
by
  sorry

end coefficient_of_x3_is_26_l463_463177


namespace find_diagonal_length_l463_463068

noncomputable def parallelepiped_diagonal_length 
  (s : ℝ) -- Side length of square face
  (h : ℝ) -- Length of vertical edge
  (θ : ℝ) -- Angle between vertical edge and square face edges
  (hsq : s = 5) -- Length of side of the square face ABCD
  (hedge : h = 5) -- Length of vertical edge AA1
  (θdeg : θ = 60) -- Angle in degrees
  : ℝ :=
5 * Real.sqrt 3

-- The main theorem to be proved
theorem find_diagonal_length
  (s : ℝ)
  (h : ℝ)
  (θ : ℝ)
  (hsq : s = 5)
  (hedge : h = 5)
  (θdeg : θ = 60)
  : parallelepiped_diagonal_length s h θ hsq hedge θdeg = 5 * Real.sqrt 3 := 
sorry

end find_diagonal_length_l463_463068


namespace example_special_set_no_special_set_ap_length_four_l463_463021

-- Definition of a special set
def is_special_set (E : set ℕ) : Prop :=
  ∀ a b ∈ E, a ≠ b → ((a - b) ^ 2) ∣ (a * b)

-- Example special set of three elements
theorem example_special_set : 
  is_special_set ({2, 3, 4} : set ℕ) := 
sorry

-- Proving non-existence of four-element special set in arithmetic progression
theorem no_special_set_ap_length_four :
  ¬ ∃ (x y : ℕ), y ≠ 0 ∧ is_special_set ({x, x + y, x + 2 * y, x + 3 * y} : set ℕ) :=
sorry

end example_special_set_no_special_set_ap_length_four_l463_463021


namespace digitlength_converges_to_four_l463_463989

-- Define the digit length function
def digitlength (n : ℕ) : ℕ :=
  let digits : List ℕ := nat.digits 10 n 
  digits.map (fun d => match d with
    | 0 => 4 -- "zero"
    | 1 => 3 -- "one"
    | 2 => 3 -- "two"
    | 3 => 5 -- "three"
    | 4 => 4 -- "four"
    | 5 => 4 -- "five"
    | 6 => 3 -- "six"
    | 7 => 5 -- "seven"
    | 8 => 5 -- "eight"
    | 9 => 4 -- "nine"
    | _ => 0 -- should never happen as d ∈ [0, 9]
  end).sum

-- Define the theorem to be proved
theorem digitlength_converges_to_four (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, (iter digitlength k n) = 4 :=
sorry

end digitlength_converges_to_four_l463_463989


namespace max_product_xy_l463_463296

theorem max_product_xy (x y : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : 7 * x + 4 * y = 150) : xy = 200 :=
by
  sorry

end max_product_xy_l463_463296


namespace sum_of_possible_values_of_c_l463_463209

theorem sum_of_possible_values_of_c : 
  (∑ c in {c | c ∈ (Set.range (λ n : ℕ, if (∃ k : ℕ, 49 - 12 * c = k^2) then n else 0)) ∧ c ≠ 0}) = 6 :=
by
  sorry

end sum_of_possible_values_of_c_l463_463209


namespace solve_x_squared_minus_floor_x_eq_1_l463_463589

open Real

theorem solve_x_squared_minus_floor_x_eq_1 :
  ∃ x : ℝ, (x^2 - ⌊x⌋ = 1) ∧ (∀ y, y^2 - ⌊y⌋ = 1 → y = x) := 
sorry

end solve_x_squared_minus_floor_x_eq_1_l463_463589


namespace number_of_ways_to_select_co_leaders_l463_463962

theorem number_of_ways_to_select_co_leaders (n k : ℕ) (hn : n = 20) (hk : k = 2) :
  (nat.choose n k) = 190 :=
by {
  rw [hn, hk],
  exact nat.choose_eq 20 2,
  sorry
}

end number_of_ways_to_select_co_leaders_l463_463962


namespace square_diag_proof_l463_463377

theorem square_diag_proof
    (W X Y Z Q O₃ O₄ : Type)
    [has_diagonal : has_segment AC (segment W X Y Z)]
    [has_circumcenter_WZQ : it_is_circumcenter WZQ O₃]
    [has_circumcenter_XYZ : it_is_circumcenter XYZ O₄]
    (WZ_length : segment_length WZ = 10)
    (angle_right : angle Q O₃ O₄ = 90) :
    ∃ (m n : ℕ), QZ = sqrt m - sqrt n ∧ m + n = 75 :=
sorry

end square_diag_proof_l463_463377


namespace gcd_1729_867_l463_463456

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by
  sorry

end gcd_1729_867_l463_463456


namespace different_ways_to_eat_spaghetti_l463_463939

-- Define the conditions
def red_spaghetti := 5
def blue_spaghetti := 5
def total_spaghetti := 6

-- This is the proof statement
theorem different_ways_to_eat_spaghetti : 
  ∃ (ways : ℕ), ways = 62 ∧ 
  (∃ r b : ℕ, r ≤ red_spaghetti ∧ b ≤ blue_spaghetti ∧ r + b = total_spaghetti) := 
sorry

end different_ways_to_eat_spaghetti_l463_463939


namespace probability_of_winning_position_l463_463532

-- Given Conditions
def tic_tac_toe_board : Type := Fin 3 × Fin 3
def is_nought (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := b p
def is_cross (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := ¬ b p

-- Winning positions in Tic-Tac-Toe are vertical, horizontal, or diagonal lines
def is_winning_position (b : tic_tac_toe_board → bool) : Prop :=
  (∃ i : Fin 3, ∀ j : Fin 3, is_nought b (i, j)) ∨ -- Row
  (∃ j : Fin 3, ∀ i : Fin 3, is_nought b (i, j)) ∨ -- Column
  (∀ i : Fin 3, is_nought b (i, i)) ∨ -- Main diagonal
  (∀ i : Fin 3, is_nought b (i, Fin.mk (2 - i.1) (by simp [i.1]))) -- Anti-diagonal

-- Problem Statement
theorem probability_of_winning_position :
  let total_positions := Nat.choose 9 3
  let winning_positions := 8
  winning_positions / total_positions = (2:ℚ) / 21 :=
by sorry

end probability_of_winning_position_l463_463532


namespace blue_red_area_ratio_l463_463838

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l463_463838


namespace x_range_l463_463005

theorem x_range (x : ℝ) : (x + 2) > 0 → (3 - x) ≥ 0 → (-2 < x ∧ x ≤ 3) :=
by
  intro h1 h2
  constructor
  { linarith }
  { linarith }

end x_range_l463_463005


namespace germination_probability_l463_463788

theorem germination_probability :
  (∀ (X : Type) (experiment_group : ℕ) (plots : ℕ) (avg_non_germinate : ℚ), 
    experiment_group = 3 ∧ plots = 1 ∧ avg_non_germinate = 1 / 3 →
    let p := 8 / 9 in 
    p = 8 / 9) :=
begin
  intro X,
  intros experiment_group plots avg_non_germinate h,
  rcases h with ⟨heq1, heq2, heq3⟩,
  let p := (8 : ℚ) / 9,
  exact eq.refl p
end

end germination_probability_l463_463788


namespace max_value_f_l463_463418

-- Define the function f
def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + (Real.sqrt 3) * (Real.sin x) * (Real.cos x)

-- State the theorem that the maximum value of the function f is 3/2
theorem max_value_f : ∃ x : ℝ, f x = 3 / 2 :=
by
  sorry

end max_value_f_l463_463418


namespace cos_alpha_eq_neg_three_fifths_l463_463638

variable (α : ℝ)
hypothesis h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi
hypothesis h2 : Real.sin (Real.pi - α) = 4 / 5

theorem cos_alpha_eq_neg_three_fifths
  (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : Real.sin (Real.pi - α) = 4 / 5) : Real.cos α = -3 / 5 := 
by
  sorry

end cos_alpha_eq_neg_three_fifths_l463_463638


namespace sum_even_coefficients_l463_463588

axiom expansion (x : ℤ) (n : ℕ) :
  ((x^2 - x - 1)^n).expand = (λ x : ℤ, a_2n*x^(2n) + a_(2n-1)*x^(2n-1) + ... + a_2*x^2 + a_1*x + a_0 : List Int)

theorem sum_even_coefficients (n : ℕ) : (a_0 + a_2 + a_4 + ... + a_(2n)) = 1/2 * (1 + (-1)^n) :=
by
  sorry

end sum_even_coefficients_l463_463588


namespace circle_center_product_l463_463594

theorem circle_center_product (h k : ℝ)
  (h_eq : h = 3)
  (k_eq : k = 5)
  (circle_eq : ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 20 ↔ x^2 + y^2 = 6x + 10y - 14) : 
  h * k = 15 :=
by
  -- Using given values of h and k
  rw [h_eq, k_eq]
  -- Simply compute the product 3 * 5
  exact (3 * 5)

end circle_center_product_l463_463594


namespace A_inter_complement_B_eq_l463_463240

-- Define set A
def set_A : Set ℝ := {x | -3 < x ∧ x < 6}

-- Define set B
def set_B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of set B in the real numbers
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- Define the intersection of set A with the complement of set B
def A_inter_complement_B : Set ℝ := set_A ∩ complement_B

-- Stating the theorem to prove
theorem A_inter_complement_B_eq : A_inter_complement_B = {x | -3 < x ∧ x ≤ 2} :=
by
  -- Proof goes here
  sorry

end A_inter_complement_B_eq_l463_463240


namespace total_money_l463_463545

variable (A B C : ℕ)

theorem total_money
  (h1 : A + C = 250)
  (h2 : B + C = 450)
  (h3 : C = 100) :
  A + B + C = 600 := by
  sorry

end total_money_l463_463545


namespace percent_of_day_is_hours_l463_463915

theorem percent_of_day_is_hours (h : ℝ) (day_hours : ℝ) (percent : ℝ) 
  (day_hours_def : day_hours = 24)
  (percent_def : percent = 29.166666666666668) :
  h = 7 :=
by
  sorry

end percent_of_day_is_hours_l463_463915


namespace g_at_10_l463_463148

noncomputable def g : ℕ → ℝ := sorry

axiom g_zero : g 0 = 2
axiom g_one : g 1 = 1
axiom g_func_eq (m n : ℕ) (h : m ≥ n) : 
  g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2 + 2

theorem g_at_10 : g 10 = 102 := sorry

end g_at_10_l463_463148


namespace hyperbola_standard_eq_line_through_Q_intersects_hyperbola_once_l463_463626

noncomputable def hyperbola_eq (x y : ℝ) := x^2 / 8 - y^2 / 2 = 1

def point_P (x y : ℝ) := (x = 4) ∧ (y = real.sqrt 2)

def point_Q (x y : ℝ) := (x = 2) ∧ (y = 2)

def line_1 (x y : ℝ) := x - 2 * y + 2 = 0

def line_2 (x y : ℝ) := x + 2 * y - 6 = 0

def line_3 (x y : ℝ) := y - 2 = (real.sqrt 10 - 1) / 2 * (x - 2)

def line_4 (x y : ℝ) := y - 2 = (-real.sqrt 10 - 1) / 2 * (x - 2)

theorem hyperbola_standard_eq (x y : ℝ) : hyperbola_eq x y :=
sorry

theorem line_through_Q_intersects_hyperbola_once (x y : ℝ) :
  point_Q x y →
  ( ∃ x y, line_1 x y ∧ hyperbola_eq x y ) ∨
  ( ∃ x y, line_2 x y ∧ hyperbola_eq x y ) ∨
  ( ∃ x y, line_3 x y ∧ hyperbola_eq x y ) ∨
  ( ∃ x y, line_4 x y ∧ hyperbola_eq x y ) :=
sorry

end hyperbola_standard_eq_line_through_Q_intersects_hyperbola_once_l463_463626


namespace F_is_decreasing_range_of_m_l463_463651

-- Given Functions
def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := (1 / 2) * x * (Real.abs x)
def F (x : ℝ) : ℝ := x * f x - g x

-- Problem Ⅰ: Monotonic Interval
theorem F_is_decreasing : ∀ x > 0, ∀ y > x, F x > F y := by
  sorry

-- Problem Ⅱ: Range of m
theorem range_of_m (m : ℝ) : (∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 < x2 → m * (g x2 - g x1) > x2 * f x2 - x1 * f x1) ↔ m ≥ 1 := by
  sorry

end F_is_decreasing_range_of_m_l463_463651


namespace correct_measure_of_dispersion_l463_463042

theorem correct_measure_of_dispersion (mean variance median mode : Type) :
  ∃ d : Type, (d = variance) :=
by
  use variance
  sorry

end correct_measure_of_dispersion_l463_463042


namespace distance_center_to_origin_l463_463236

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

theorem distance_center_to_origin :
  (∃ (x y : ℝ), circle_eq x y ∧ x = 1 ∧ y = 0) →
  ∃ d : ℝ, d = 1 ∧ ∀ x y, (x = 1 ∧ y = 0) → d = real.sqrt ((0 - x) ^ 2 + (0 - y) ^ 2) :=
by
  intros h
  obtain ⟨cx, cy, h_eq, hc_x, hc_y⟩ := h
  use 1
  split
  · refl
  intros x y hx 
  rw [hc_x, hc_y] at hx
  exact hx.symm ▸ rfl
  sorry

end distance_center_to_origin_l463_463236


namespace fastest_path_l463_463006

-- Definitions based on conditions
def distance_AC := 9 -- Distance from A to C in km
def distance_AE := 3 -- Distance from A to E in km
def distance_EC := Real.sqrt 37 -- Distance from E to C in km

variable (v : ℝ) -- Speed on sand
def speed_sand := v -- Speed on sand
def speed_peat := 2 * v -- Speed on peat

-- Time calculations for each path based on the provided speeds
def time_AC := distance_AC / speed_sand
def time_AE := distance_AE / speed_sand
def time_EC := distance_EC / speed_peat

-- Total time for path A -> E -> C
def time_AEC := time_AE + time_EC

-- The main theorem to prove that A -> E -> C is faster than A -> C
theorem fastest_path (v : ℝ) (hv : 0 < v) : time_AEC v < time_AC v := by
  sorry

end fastest_path_l463_463006


namespace sum_of_valid_c_l463_463189

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_valid_c :
  (∑ c in Finset.filter (λ c, ∃ k, k * k = 49 - 12 * c) (Finset.range 5), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463189


namespace triangle_BC_length_l463_463338

theorem triangle_BC_length (A B C D : Type)
  [MetricSpace A B C D]
  (h_pos : A ≠ B)
  (h_isos_ABD : IsoscelesTriangle A B D)
  (h_isos_ACD : IsoscelesTriangle A C D)
  (h_AB_gt_AC : dist A B > dist A C)
  (h2 : dist A C = 4)
  (h3 : dist A D = 3)
  (h_on_BC : ∃ x y : Real, dist B C = x + y) :
  dist B C = 7 ∨ dist B C = 8 :=
by
  sorry

end triangle_BC_length_l463_463338


namespace find_second_prime_number_l463_463004

open Nat

-- Define primes in the given range
def prime_in_range : Nat → Prop := λ p => prime p ∧ 2 < p ∧ p < 6

-- Define the second prime number based on the product condition
def second_prime_condition (z a b : Nat) : Prop := prime b ∧ z = a * b

-- Main theorem statement to be proved
theorem find_second_prime_number (a b z : Nat) (h1 : 15 < z) (h2 : z < 36) (h3 : prime_in_range a) (h4 : z = 33) : b = 11 := by
  sorry

end find_second_prime_number_l463_463004


namespace greatest_integer_l463_463738

noncomputable def x := (∑ n in Finset.range 45, Real.cos (n.succ * Real.pi / 180)) / 
                       (∑ n in Finset.range 45, Real.sin (n.succ * Real.pi / 180))

theorem greatest_integer (x := (∑ n in Finset.range 45, Real.cos ((n + 1) * Real.pi / 180)) / 
                              (∑ n in Finset.range 45, Real.sin ((n + 1) * Real.pi / 180))) : 
  ⌊100 * x⌋ = 341 :=
sorry

end greatest_integer_l463_463738


namespace parallel_planes_theorem_l463_463688

-- Define the structure of a tetrahedron
structure Tetrahedron (P : Type) :=
  (S A B C : P)

-- Define the midpoint property
def is_midpoint {P : Type} [AddCommGroup P] [Module ℝ P] (M : P) (S A : P) :=
  M = (S + A) / 2

-- Define the parallel planes property
def parallel_planes {P : Type} [AffineSpace ℝ P] (plane1 plane2 : AffineSubspace ℝ P) :=
  ∃ L1 L2 L3 L4 : AffineSubspace ℝ P, L1 ∈ plane1 ∧ L2 ∈ plane1 ∧ L1 ⟂ L2 ∧
  L3 ∈ plane2 ∧ L4 ∈ plane2 ∧ L3 ⟂ L4 ∧ L1 ∋ L2 ∧ L3 ∋ L4

-- Noncomputable definition as it involves a theoretical proof
noncomputable def Problem (P : Type) [AddCommGroup P] [Module ℝ P] [AffineSpace ℝ P] :=
  let T := Tetrahedron P in
  let plane_ABC := AffineSpan ℝ ({T.A, T.B, T.C} : Set P) in
  let M := (T.S + T.A) / 2 in
  let N := (T.S + T.B) / 2 in
  let P := (T.S + T.C) / 2 in
  let plane_MNP := AffineSpan ℝ ({M, N, P} : Set P) in
  parallel_planes plane_MNP plane_ABC

-- The actual statement that needs to be proven
theorem parallel_planes_theorem {P : Type} [AddCommGroup P] [Module ℝ P] [AffineSpace ℝ P] :
  Problem P :=
sorry  -- Proof is omitted

end parallel_planes_theorem_l463_463688


namespace lucky_n_iff_power_of_two_l463_463622

/--
A configuration of colored cubes is defined to be good if the final remaining cube's color after
a series of operations does not depend on the robot's initial starting point.
N is defined to be lucky if every arrangement of N cubes is good.

This theorem states that N is lucky if and only if N is a power of 2.
-/
theorem lucky_n_iff_power_of_two (N : ℕ) : 
  (∀ cubes : Vector ℕ N, ∀ start : Fin N, good_configuration cubes start) ↔ ∃ k : ℕ, N = 2^k :=
sorry

end lucky_n_iff_power_of_two_l463_463622


namespace tic_tac_toe_probability_l463_463540

theorem tic_tac_toe_probability :
  let total_positions := Nat.choose 9 3,
      winning_positions := 8 in
  (winning_positions / total_positions : ℚ) = 2 / 21 :=
by
  sorry

end tic_tac_toe_probability_l463_463540


namespace basketball_team_players_l463_463484

theorem basketball_team_players (score_total : ℕ) (min_points : ℕ) (max_individual : ℕ) (player_points_sum : ∀ n : ℕ, n * min_points ≤ score_total) 
  (players_count_max_bound : ∀ n : ℕ, n ≤ score_total / min_points)
  (max_single_player : ℕ) (remaining_points : ∀ m: ℕ, m = score_total - max_single_player)
  (players_for_remaining_points : ∀ k : ℕ, k * min_points = remaining_points) : 
  score_total = 100 ∧ min_points = 7 ∧ max_individual = 23 → ∃ n : ℕ, n = 12 := 
sorry

end basketball_team_players_l463_463484


namespace determine_abc_l463_463323

theorem determine_abc (N : ℕ) (a b c : ℕ) (h1 : N = 4422) (h2 : a + b + c ≥ 18) :
  ∃ a b c, 100a + 10b + c = 785 ∧ N = (100a + 10b + c) + (100a + c + 10b) + (100b + 10c + a) + (100b + c + 10a) + (100c + 10a + b) + (100c + 10b + a) := by
  sorry

end determine_abc_l463_463323


namespace probability_of_winning_position_l463_463522

theorem probability_of_winning_position : 
    (let total_positions := Nat.choose 9 3,
         winning_positions := 8 in
       (winning_positions : ℚ) / total_positions = 2 / 21) := 
by
  sorry

end probability_of_winning_position_l463_463522


namespace probability_of_winning_position_l463_463524

theorem probability_of_winning_position : 
    (let total_positions := Nat.choose 9 3,
         winning_positions := 8 in
       (winning_positions : ℚ) / total_positions = 2 / 21) := 
by
  sorry

end probability_of_winning_position_l463_463524


namespace value_of_k_l463_463326

   noncomputable def k (a b : ℝ) : ℝ := 3 / 4

   theorem value_of_k (a b k : ℝ) 
     (h1: b = 4 * k + 1) 
     (h2: 5 = a * k + 1) 
     (h3: b + 1 = a * k + 1) : 
     k = 3 / 4 := 
   by 
     -- Proof goes here 
     sorry
   
end value_of_k_l463_463326


namespace bird_needs_more_twigs_l463_463485

variable (base_twigs : ℕ := 12)
variable (additional_twigs_per_base : ℕ := 6)
variable (fraction_dropped : ℚ := 1/3)

theorem bird_needs_more_twigs (tree_dropped : ℕ) : 
  tree_dropped = (additional_twigs_per_base * base_twigs) * 1/3 →
  (base_twigs * additional_twigs_per_base - tree_dropped) = 48 :=
by
  sorry

end bird_needs_more_twigs_l463_463485


namespace double_grandfather_pension_l463_463319

-- Define the total family income and individual contributions
def total_income (masha mother father grandfather : ℝ) : ℝ :=
  masha + mother + father + grandfather

-- Define the conditions provided in the problem
variables
  (masha mother father grandfather : ℝ)
  (cond1 : 2 * masha = total_income masha mother father grandfather * 1.05)
  (cond2 : 2 * mother = total_income masha mother father grandfather * 1.15)
  (cond3 : 2 * father = total_income masha mother father grandfather * 1.25)

-- Define the statement to be proved
theorem double_grandfather_pension :
  2 * grandfather = total_income masha mother father grandfather * 1.55 :=
by
  -- Proof placeholder
  sorry

end double_grandfather_pension_l463_463319


namespace triangle_area_from_squares_l463_463639

theorem triangle_area_from_squares (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁ = 36) (h₂ : a₂ = 64) (h₃ : a₃ = 100)
  (h_side1 : sqrt a₁ = 6) (h_side2 : sqrt a₂ = 8) (h_hypotenuse : sqrt a₃ = 10) :
  1 / 2 * sqrt a₁ * sqrt a₂ = 24 :=
by
  sorry

end triangle_area_from_squares_l463_463639


namespace real_roots_of_polynomial_l463_463595

theorem real_roots_of_polynomial :
  {x : ℝ | (x^4 - 4*x^3 + 5*x^2 - 2*x + 2) = 0} = {1, -1} :=
sorry

end real_roots_of_polynomial_l463_463595


namespace trig_sign_product_l463_463153

theorem trig_sign_product :
  (0 < 1 ∧ 1 < real.pi / 2) ∧
  (real.pi / 2 < 2 ∧ 2 < real.pi) ∧
  (real.pi / 2 < 3 ∧ 3 < real.pi) →
  real.sin 1 * real.cos 2 * real.tan 3 > 0 :=
by {
  sorry
}

end trig_sign_product_l463_463153


namespace problem_set_M_is_interval_l463_463367

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 5 * x + 6

noncomputable def h (x : ℝ) : ℝ := 1 + Real.logb 2 x

def P : Set ℝ := { x | h x > 2 }

def Q : Set ℝ := { x | f (h x) ≥ 0 ∧ x > 0 }

def M : Set ℝ := { x | x ∈ P ∧ x ∉ Q }

theorem problem_set_M_is_interval : M = set.Ioo 2 4 :=
sorry

end problem_set_M_is_interval_l463_463367


namespace blue_faces_ratio_l463_463843

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l463_463843


namespace log_ratio_problem_l463_463562

theorem log_ratio_problem : (Real.log 2 / Real.log 3) / (Real.log 8 / Real.log 9) = (2 : ℝ / 3 : ℝ) := 
sorry

end log_ratio_problem_l463_463562


namespace quadratic_inequality_solution_l463_463636

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : 0 > a) 
(h2 : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (0 < ax^2 + bx + c)) : 
(∀ x : ℝ, (x < 1/2 ∨ 1 < x) ↔ (0 < 2*a*x^2 - 3*a*x + a)) :=
sorry

end quadratic_inequality_solution_l463_463636


namespace sum_of_coefficients_of_y_terms_l463_463561

theorem sum_of_coefficients_of_y_terms: 
  let p := (5 * x + 3 * y + 2) * (2 * x + 5 * y + 3)
  ∃ (a b c: ℝ), p = (10 * x^2 + a * x * y + 19 * x + b * y^2 + c * y + 6) ∧ a + b + c = 65 :=
by
  sorry

end sum_of_coefficients_of_y_terms_l463_463561


namespace probability_at_least_one_red_bean_paste_probability_distribution_X_expectation_X_l463_463580

-- Definition of the conditions
def total_zongzi : ℕ := 6
def egg_yolk_zongzi : ℕ := 4
def red_bean_paste_zongzi : ℕ := 2
def total_selected_zongzi : ℕ := 3

-- Definitions for probability calculations
noncomputable def combination (n r : ℕ) : ℚ := (nat.factorial n) / ((nat.factorial r) * (nat.factorial (n-r)))

-- Statement 1: Prove the probability of at least one red bean paste zongzi
theorem probability_at_least_one_red_bean_paste :
  (combination egg_yolk_zongzi 2 * combination red_bean_paste_zongzi 1 + combination egg_yolk_zongzi 1 * combination red_bean_paste_zongzi 2) / combination total_zongzi total_selected_zongzi = 4 / 5 :=
by sorry

-- Definitions and theorems for probability distribution and expectation of X
def P_X_0 : ℚ := combination egg_yolk_zongzi 3 / combination total_zongzi total_selected_zongzi
def P_X_1 : ℚ := combination egg_yolk_zongzi 2 * combination red_bean_paste_zongzi 1 / combination total_zongzi total_selected_zongzi
def P_X_2 : ℚ := combination egg_yolk_zongzi 1 * combination red_bean_paste_zongzi 2 / combination total_zongzi total_selected_zongzi

theorem probability_distribution_X :
  (P_X_0 = 1 / 5) ∧ (P_X_1 = 3 / 5) ∧ (P_X_2 = 1 / 5) :=
by sorry

theorem expectation_X :
  (0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2) = 1 :=
by sorry

end probability_at_least_one_red_bean_paste_probability_distribution_X_expectation_X_l463_463580


namespace length_of_QR_of_triangle_l463_463875

def length_of_QR (PQ PR PM : ℝ) : ℝ := sorry

theorem length_of_QR_of_triangle (PQ PR : ℝ) (PM : ℝ) (hPQ : PQ = 4) (hPR : PR = 7) (hPM : PM = 7 / 2) : length_of_QR PQ PR PM = 9 := by
  sorry

end length_of_QR_of_triangle_l463_463875


namespace gcd_1729_867_l463_463457

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by
  sorry

end gcd_1729_867_l463_463457


namespace blue_faces_ratio_l463_463844

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l463_463844


namespace range_of_x_l463_463994

theorem range_of_x
  (a : Fin 25 → ℝ)
  (h : ∀ i, a i = 0 ∨ a i = 3) :
  let x := ∑ i in Finset.range 25, a i / 3^(i+1)
  in (0 ≤ x ∧ x < 1/2) ∨ (1 ≤ x ∧ x < 3/2) :=
by
  sorry

end range_of_x_l463_463994


namespace max_points_of_intersection_l463_463891

theorem max_points_of_intersection
  (A B C : Circle) (L : Line)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ B ≠ C) :
  (∃ pA pB pC : Point, on_line L pA ∧ on_circle A pA ∧
                    on_line L pB ∧ on_circle B pB ∧
                    on_line L pC ∧ on_circle C pC) ∧
  max_intersection_points A B C L = 12 :=
sorry

end max_points_of_intersection_l463_463891


namespace max_points_of_intersection_l463_463892

theorem max_points_of_intersection
  (A B C : Circle) (L : Line)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ B ≠ C) :
  (∃ pA pB pC : Point, on_line L pA ∧ on_circle A pA ∧
                    on_line L pB ∧ on_circle B pB ∧
                    on_line L pC ∧ on_circle C pC) ∧
  max_intersection_points A B C L = 12 :=
sorry

end max_points_of_intersection_l463_463892


namespace problem_solution_l463_463247

theorem problem_solution (a b c : ℝ) (h1 : a^2 - b^2 = 5) (h2 : a * b = 2) (h3 : a^2 + b^2 + c^2 = 8) : 
  a^4 + b^4 + c^4 = 38 :=
sorry

end problem_solution_l463_463247


namespace five_letter_word_count_l463_463991

theorem five_letter_word_count :
  let vowels := {'A', 'E', 'I', 'O', 'U'}
  let num_letters := 26
  ∃ (num_words : ℕ), num_words = num_letters * num_letters * num_letters * 5 :=
  sorry

end five_letter_word_count_l463_463991


namespace sin1993_cos1993_leq_zero_l463_463632

theorem sin1993_cos1993_leq_zero (x : ℝ) (h : Real.sin x + Real.cos x ≤ 0) : 
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := 
by 
  sorry

end sin1993_cos1993_leq_zero_l463_463632


namespace monotonically_decreasing_log_less_than_x_l463_463261

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log (x + 1) - x

-- State the problem
theorem monotonically_decreasing (x : ℝ) (h : x > 0) : 
  (∃ I : Set ℝ, I = set.Ioi 0 ∧ ∀ y ∈ I, differentiable_at ℝ f y) ∧ (∀ y, y ∈ set.Ioi 0 → deriv f y < 0) := sorry

theorem log_less_than_x (x : ℝ) (h : x > -1) : Real.log (x + 1) ≤ x := sorry

end monotonically_decreasing_log_less_than_x_l463_463261


namespace sum_fractional_parts_of_zeta_l463_463601

noncomputable def zeta (x : ℝ) : ℝ :=
∑' (n : ℕ), (1 : ℝ) / (n ^ x)

def fractional_part (x : ℝ) : ℝ :=
x - x.floor

theorem sum_fractional_parts_of_zeta :
  ∑ k in Finset.range 8, fractional_part (zeta (6 + 2 * k)) = 1 / 4 := 
sorry

end sum_fractional_parts_of_zeta_l463_463601


namespace bus_stops_per_hour_l463_463059

-- Define the speeds as constants
def speed_excluding_stoppages : ℝ := 60
def speed_including_stoppages : ℝ := 50

-- Formulate the main theorem
theorem bus_stops_per_hour :
  (1 - speed_including_stoppages / speed_excluding_stoppages) * 60 = 10 := 
by
  sorry

end bus_stops_per_hour_l463_463059


namespace inradius_length_l463_463355

noncomputable theory

variables {A B C I D: Type}
variables (BC : ℝ) (AB AC : ℝ) (IC : ℝ)
variables [IsoscelesTriangle A B C]

def midpoint (B C : Type) : Type := sorry -- assume we have a definition for the midpoint of BC
def incenter (A B C : Type) : Type := I -- assume we have a definition of incenter I

def is_inradius (ID : ℝ) : Prop :=
  let DC := 20 in
  ID^2 + DC^2 = IC^2

theorem inradius_length :
  AB = AC → 
  BC = 40 →
  IC = 24 →
  ∃ ID : ℝ, is_inradius ID ∧ ID = 4 * sqrt 11 :=
by
  intros h1 h2 h3
  use 4 * sqrt 11
  unfold is_inradius
  sorry   -- proof is omitted

end inradius_length_l463_463355


namespace units_digit_of_17_pow_28_l463_463913

theorem units_digit_of_17_pow_28 : ((17:ℕ) ^ 28) % 10 = 1 := by
  sorry

end units_digit_of_17_pow_28_l463_463913


namespace dispersion_is_variance_l463_463040

def Mean := "Mean"
def Variance := "Variance"
def Median := "Median"
def Mode := "Mode"

def dispersion_measure := Variance

theorem dispersion_is_variance (A B C D : String) (hA : A = Mean) (hB : B = Variance) (hC : C = Median) (hD : D = Mode) : 
  dispersion_measure = B :=
by
  rw [hB]
  exact sorry

end dispersion_is_variance_l463_463040


namespace park_illuminated_min_lamps_park_reliably_illuminated_min_lamps_l463_463001

-- Definitions for the problem conditions
def park : Type := matrix (fin 10) (fin 10) bool

-- Part (a): Illuminated park
theorem park_illuminated_min_lamps (p : park) :
  (∀ r c : fin 10, ∃ i j : fin 3, p (r + i) (c + j)) → ∃ l, l ≤ 4 := sorry

-- Part (b): Reliably illuminated park
theorem park_reliably_illuminated_min_lamps (p : park) :
  (∀ r c : fin 10, ∃ k : fin 10, ∃ i j : fin 3, p (r + i) (c + j))
  → ∃ l, l ≤ 10 := sorry

end park_illuminated_min_lamps_park_reliably_illuminated_min_lamps_l463_463001


namespace partner_p_investment_time_l463_463421

theorem partner_p_investment_time
  (investment_ratio_pq : ℕ → ℕ → Prop)
  (profit_ratio_pq : ℕ → ℕ → Prop)
  (investment_time_q : ℕ)
  (investment_time_p : ℕ) :
  investment_ratio_pq 7 5 →
  profit_ratio_pq 7 12 →
  investment_time_q = 12 →
  investment_time_p = 5 :=
begin
  sorry
end

end partner_p_investment_time_l463_463421


namespace resistance_construction_l463_463493

theorem resistance_construction (R0 : ℝ) (R0_val : R0 = 1) : 
  let R := 0.94 in 
  (resistance : ℝ) = R :=
by 
  -- Definitions based directly on the problem conditions.
  let homogeneous_wire_constant_cross_section := true,
  let points_midpoints := true,
  let AB_resistance_is_R0 := R0,
  admit -- Use 'sorry' to skip the proof as specified in the guidelines.

end resistance_construction_l463_463493


namespace blue_to_red_face_area_ratio_l463_463855

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l463_463855


namespace sum_of_coefficients_l463_463132

noncomputable def poly : ℤ[X] :=
  -3 * (X^8 - 2 * X^5 + X^3 - 6) + 5 * (2 * X^4 - 3 * X + 1) - 2 * (X^6 - 5)

theorem sum_of_coefficients : (eval 1 poly) = 26 :=
by sorry

end sum_of_coefficients_l463_463132


namespace telepathic_connection_correct_l463_463079

def telepathic_connection_probability : ℚ := sorry

theorem telepathic_connection_correct :
  telepathic_connection_probability = 7 / 25 := sorry

end telepathic_connection_correct_l463_463079


namespace isosceles_triangle_perimeter_l463_463235

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), 
  (a = 3 ∧ b = 6 ∧ (c = 6 ∨ c = 3)) ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
  (a + b + c = 15) :=
sorry

end isosceles_triangle_perimeter_l463_463235


namespace total_time_correct_l463_463548

def pictures := 
  { trees := 300, flowers := 400, grass := 250, shrubs := 150, ferns := 100 }

def rates :=
  { trees := 35, flowers := 20, grass := 50, shrubs := 15, ferns := 25 }

def processing_time (pictures : ℕ) (rate : ℕ) : ℕ := pictures / rate

def total_processing_time :=
  processing_time pictures.trees rates.trees +
  processing_time pictures.flowers rates.flowers +
  processing_time pictures.grass rates.grass +
  processing_time pictures.shrubs rates.shrubs +
  processing_time pictures.ferns rates.ferns

theorem total_time_correct :
  total_processing_time = 47.57 := by
  sorry

end total_time_correct_l463_463548


namespace work_completion_l463_463923

theorem work_completion : 
  ∀ (A B : ℝ), (A = 2 * B) → (A = 1 / 27) → 1 / (A + B) = 18 :=
by 
  -- Assume definitions of A and B
  intros A B 
  assume h1 : A = 2 * B
  assume h2 : A = 1 / 27
  -- Variables should be used in the proof
  have h3 : B = (A / 2),
    sorry,
  have h4 : A + B = A + A/2,
    sorry,
  have h5 : A + A/2 = 3A / 2,
    sorry,
  have h6 : 1 / (3A / 2) = 18,
    sorry,
  show 1 / (A + B) = 18 from 
  sorry,

end work_completion_l463_463923


namespace find_B_l463_463656

variable (A B : Set ℤ)
variable (U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6})

theorem find_B (hU : U = {x | 0 ≤ x ∧ x ≤ 6})
               (hA_complement_B : A ∩ (U \ B) = {1, 3, 5}) :
  B = {0, 2, 4, 6} :=
sorry

end find_B_l463_463656


namespace intersection_a_eq_1_parallel_lines_value_of_a_l463_463270

-- Define lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y - a + 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 2 * a * x + (a + 3) * y + a - 5 = 0

-- Part 1: Prove intersection point for a = 1
theorem intersection_a_eq_1 :
  line1 1 (-4) 3 ∧ line2 1 (-4) 3 :=
by sorry

-- Part 2: Prove value of a for which lines are parallel
theorem parallel_lines_value_of_a :
  ∃ a : ℝ, ∀ x y : ℝ, line1 a x y ∧ line2 a x y →
  (2 * a^2 - a - 3 = 0 ∧ a ≠ -1 ∧ a = 3/2) :=
by sorry

end intersection_a_eq_1_parallel_lines_value_of_a_l463_463270


namespace john_trip_duration_l463_463722

theorem john_trip_duration : 
  let t1 := 2 in
  let t2 := 2 * t1 in
  let t3 := 2 * t1 in
  t1 + t2 + t3 = 10 :=
by
  sorry

end john_trip_duration_l463_463722


namespace shaded_area_ratio_l463_463330

-- Defining the problem
theorem shaded_area_ratio (x : ℝ) (h1 : x > 0) :
  let AC := x,
      CB := 2 * x,
      AB := 3 * x,
      r_large := 1.5 * x,
      r_small1 := 0.5 * x,
      r_small2 := x,
      area_large := (1 / 2) * Real.pi * (r_large ^ 2),
      area_small1 := (1 / 2) * Real.pi * (r_small1 ^ 2),
      area_small2 := (1 / 2) * Real.pi * (r_small2 ^ 2),
      shaded_area := area_large - area_small1 - area_small2,
      r_CD := r_large,
      area_circle_CD := Real.pi * (r_CD ^ 2) in
  shaded_area / area_circle_CD = 11 / 9 :=
by
  sorry

end shaded_area_ratio_l463_463330


namespace equal_partitions_l463_463014

def weights : List ℕ := List.range (81 + 1) |>.map (λ n => n * n)

theorem equal_partitions (h : weights.sum = 178605) :
  ∃ P1 P2 P3 : List ℕ, P1.sum = 59535 ∧ P2.sum = 59535 ∧ P3.sum = 59535 ∧ P1 ++ P2 ++ P3 = weights := sorry

end equal_partitions_l463_463014


namespace part1_part2_l463_463294

-- Definition of the custom operator "⊗" in terms of the relationship a^c = b
def custom_op (a b c : ℕ) : Prop := a^c = b

-- Part 1
theorem part1 (x : ℕ) : custom_op 3 81 x → x = 4 := by
  intro h1
  sorry

-- Part 2
theorem part2 (a b c : ℕ) : custom_op 3 5 a → custom_op 3 6 b → custom_op 3 10 c → a < b < c := by
  intros h1 h2 h3
  sorry

end part1_part2_l463_463294


namespace radius_of_circle_l463_463946

-- Circle with area x and circumference y
def circle_area (r : ℝ) : ℝ := π * r^2
def circle_circumference (r : ℝ) : ℝ := 2 * π * r
def circle_equation (r : ℝ) : ℝ := circle_area r + circle_circumference r

-- The given condition
theorem radius_of_circle (r : ℝ) (h : circle_equation r = 100 * π) : r = 10 :=
sorry

end radius_of_circle_l463_463946


namespace smallest_positive_period_of_sin_2x_l463_463597

noncomputable def period_of_sine (B : ℝ) : ℝ := (2 * Real.pi) / B

theorem smallest_positive_period_of_sin_2x :
  period_of_sine 2 = Real.pi := sorry

end smallest_positive_period_of_sin_2x_l463_463597


namespace tournament_cycle_exists_l463_463937

theorem tournament_cycle_exists :
  ∃ (A B C : Fin 12), 
  (∃ M : Fin 12 → Fin 12 → Bool, 
    (∀ p : Fin 12, ∃ q : Fin 12, q ≠ p ∧ M p q) ∧
    M A B = true ∧ M B C = true ∧ M C A = true) :=
sorry

end tournament_cycle_exists_l463_463937


namespace minimum_value_of_m_l463_463643

-- Define the necessary parameters and assumptions based on the problem conditions
variables (k k1 : ℤ)
def f (x : ℝ) : ℝ := Real.cos (2 * x + (k * Real.pi - (5 * Real.pi / 6)))
def g (x : ℝ) (m : ℝ) : ℝ := Real.cos (2 * x - 2 * m + k * Real.pi - (5 * Real.pi / 6))

-- Define the condition for the graph of f(x) being symmetric about the point (2π/3, 0)
axiom symmetry_about_point : 
  2 * (2 * Real.pi / 3) + (k * Real.pi - 5 * Real.pi / 6) = k * Real.pi + Real.pi / 2

-- Define the condition for g(x) being an even function after translating f(x) to the right by m units
axiom even_function_after_translation : 
  ∀ m > 0, g 0 m = g 0 (-m)

-- Prove that the minimum value of m which satisfies all conditions
theorem minimum_value_of_m : 
  ∃ m > 0, m = Real.pi / 12 :=
sorry

end minimum_value_of_m_l463_463643


namespace tic_tac_toe_winning_probability_l463_463529

theorem tic_tac_toe_winning_probability :
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  probability = (2 : ℚ) / 21 :=
by
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  have probability_correct : probability = (2 : ℚ) / 21 := sorry
  exact probability_correct

end tic_tac_toe_winning_probability_l463_463529


namespace blue_face_area_factor_l463_463797

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l463_463797


namespace quadratic_distinct_roots_l463_463046

theorem quadratic_distinct_roots :
  ∀ a b c : ℝ, a ≠ 0 ∧ b = 4 ∧ c = -4 → (b ^ 2 - 4 * a * c > 0) :=
by
  intros a b c ha hb hc
  rw [hb, hc]
  sorry

end quadratic_distinct_roots_l463_463046


namespace problem_statement_l463_463380

open Complex

noncomputable def unit_circle (n : ℕ) (ω : Fin n → ℂ) : Prop :=
  ∀ j, abs (ω j) = 1

theorem problem_statement (n : ℕ) (ω : Fin n → ℂ)
  (h : unit_circle n ω) :
  (abs (∑ j in (Finset.finRange n), ω j) = n - 1)
  ∧ (abs (∑ j in (Finset.finRange n), (ω j)^2) = n - 1)
  ↔ n = 2 := by sorry

end problem_statement_l463_463380


namespace blue_face_area_greater_than_red_face_area_l463_463829

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l463_463829


namespace yoojung_initial_candies_l463_463053

theorem yoojung_initial_candies (candies_given_older_sister : ℕ) (candies_given_younger_sister : ℕ) (candies_left : ℕ) :
  candies_given_older_sister = 7 →
  candies_given_younger_sister = 6 →
  candies_left = 15 →
  ∃ C : ℕ, C = 28 ∧ C = candies_given_older_sister + candies_given_younger_sister + candies_left :=
by
  intros h1 h2 h3
  use (candies_given_older_sister + candies_given_younger_sister + candies_left)
  rw [h1, h2, h3]
  simp
  sorry

end yoojung_initial_candies_l463_463053


namespace find_k_l463_463500

-- Definition of vectors a and b
variable (a b : Type) [AddCommGroup a] [Module ℝ a]

-- Definition of k and a specific vector
variable (k : ℝ)

-- A line passing through a and b can be described by a + t(b - a)
def line_through := λ (t : ℝ), a + t • (b - a)

-- Proof statement: Given that k a + 5/8 b lies on the line through a and b, show k = 3/8
theorem find_k (h : ∃ t : ℝ, k • a + (5 / 8) • b = a + t • (b - a)) : k = 3 / 8 :=
  sorry

end find_k_l463_463500


namespace common_chord_length_l463_463423

theorem common_chord_length (O₁ O₂ A B K : Point)
  (r : ℝ)
  (h1 : O₁ = CenterCircle1)
  (h2 : O₂ = CenterCircle2)
  (h3 : CircleRadiusO₁ = r)
  (h4 : CircleRadiusO₂ = 2 * r)
  (h5 : SegmentLength(K, O₁) = 2)
  (h6 : SegmentLength(K, O₂) = 5)
  (h7 : Perpendicular(O₁O₂, AB))
  (h8 : IntersectAt(O₁O₂, AB, K))
  : SegmentLength(A, B) = 2 * Real.sqrt(3) := by
  sorry

end common_chord_length_l463_463423


namespace range_of_a_l463_463645

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ Real.exp 1) := by
  sorry

end range_of_a_l463_463645


namespace sine_shift_equiv_l463_463444

theorem sine_shift_equiv {x : ℝ} :
  (sin (2 * x - π / 6)) = (sin (2 * (x - π / 4) + π / 3)) :=
by sorry

end sine_shift_equiv_l463_463444


namespace Pentagon_PA_PD_equal_l463_463349

-- Definitions of the pentagon and properties

structure Pentagon (A B C D E P : Type*) :=
  (equal_sides : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ E) ∧ (E ≠ A) ∧ (A = B = C = D = E))
  (right_angle_C : angle_eq C D A (π / 2))
  (right_angle_D : angle_eq C D B (π / 2))
  (intersection_point : ∃ P, intersects (line AC) (line BD))

-- Proof statement for the problem
theorem Pentagon_PA_PD_equal {A B C D E P : Type*} 
  (p : Pentagon A B C D E P) : 
  (dist P A) = (dist P D) :=
sorry

end Pentagon_PA_PD_equal_l463_463349


namespace find_omega_find_zeros_of_g_l463_463648

noncomputable def f (ω x : ℝ) : ℝ := 2 * sin (ω * x - π / 6) * cos (ω * x) + 1 / 2

theorem find_omega 
  (ω : ℝ)
  (h_pos : ω > 0) 
  (h_period : ∀ T > 0, (∀ x, f ω (x + T) = f ω x) → T = π) :
  ω = 1 := 
sorry

noncomputable def shifted_f (x : ℝ) : ℝ := f 1 (x + π / 6)
noncomputable def g (x : ℝ) : ℝ := shifted_f (x / 2)

theorem find_zeros_of_g :
  ∀ x ∈ Icc (-π) π, g x = 0 ↔ x = -π / 6 ∨ x = 5 * π / 6 :=
sorry

end find_omega_find_zeros_of_g_l463_463648


namespace sum_even_factors_720_l463_463996

theorem sum_even_factors_720 : 
  let n := 720
  have h1 : n = 2^4 * 3^2 * 5^1,
  let sum_powers_2 := 2 + 4 + 8 + 16,
  let sum_powers_3 := 1 + 3 + 9,
  let sum_powers_5 := 1 + 5,
  30 * 13 * 6 = 2340 := -- derived from sums
by
  have h_n := 720_eq_2_4_3_2_5_1
  sorry

end sum_even_factors_720_l463_463996


namespace profit_with_discount_l463_463960

theorem profit_with_discount (CP SP_with_discount SP_no_discount : ℝ) (discount profit_no_discount : ℝ) (H1 : discount = 0.1) (H2 : profit_no_discount = 0.3889) (H3 : SP_no_discount = CP * (1 + profit_no_discount)) (H4 : SP_with_discount = SP_no_discount * (1 - discount)) : (SP_with_discount - CP) / CP * 100 = 25 :=
by
  -- The proof will be filled here
  sorry

end profit_with_discount_l463_463960


namespace proof_problem_l463_463253

variables {a : ℕ → ℕ} -- sequence a_n is positive integers
variables {b : ℕ → ℕ} -- sequence b_n is integers
variables {q : ℕ} -- ratio for geometric sequence
variables {d : ℕ} -- difference for arithmetic sequence
variables {a1 b1 : ℕ} -- initial terms for the sequences

-- Additional conditions as per the problem statement
def geometric_seq (a : ℕ → ℕ) (a1 q : ℕ) : Prop :=
∀ n : ℕ, a n = a1 * q^(n-1)

def arithmetic_seq (b : ℕ → ℕ) (b1 d : ℕ) : Prop :=
∀ n : ℕ, b n = b1 + (n-1) * d

-- Given conditions
variable (geometric : geometric_seq a a1 q)
variable (arithmetic : arithmetic_seq b b1 d)
variable (equal_term : a 6 = b 7)

-- The proof task
theorem proof_problem : a 3 + a 9 ≥ b 4 + b 10 :=
by sorry

end proof_problem_l463_463253


namespace scout_weekend_earnings_l463_463384

-- Definitions for conditions
def base_pay_per_hour : ℝ := 10.00
def tip_saturday : ℝ := 5.00
def tip_sunday_low : ℝ := 3.00
def tip_sunday_high : ℝ := 7.00
def transportation_cost_per_delivery : ℝ := 1.00
def hours_worked_saturday : ℝ := 6
def deliveries_saturday : ℝ := 5
def hours_worked_sunday : ℝ := 8
def deliveries_sunday : ℝ := 10
def deliveries_sunday_low_tip : ℝ := 5
def deliveries_sunday_high_tip : ℝ := 5
def holiday_multiplier : ℝ := 2

-- Calculation of total earnings for the weekend after transportation costs
theorem scout_weekend_earnings : 
  let base_pay_saturday := hours_worked_saturday * base_pay_per_hour
  let tips_saturday := deliveries_saturday * tip_saturday
  let transportation_costs_saturday := deliveries_saturday * transportation_cost_per_delivery
  let total_earnings_saturday := base_pay_saturday + tips_saturday - transportation_costs_saturday

  let base_pay_sunday := hours_worked_sunday * base_pay_per_hour * holiday_multiplier
  let tips_sunday := deliveries_sunday_low_tip * tip_sunday_low + deliveries_sunday_high_tip * tip_sunday_high
  let transportation_costs_sunday := deliveries_sunday * transportation_cost_per_delivery
  let total_earnings_sunday := base_pay_sunday + tips_sunday - transportation_costs_sunday

  let total_earnings_weekend := total_earnings_saturday + total_earnings_sunday

  total_earnings_weekend = 280.00 :=
by
  -- Add detailed proof here
  sorry

end scout_weekend_earnings_l463_463384


namespace algebraic_expression_value_l463_463281

-- Definitions based on the conditions
variables (a b c d x : ℝ)
hypothesis h1 : a = -b
hypothesis h2 : c * d = 1
hypothesis h3 : x = 2

-- The statement we want to prove
theorem algebraic_expression_value : (c * d) ^ 2015 * x ^ 2 + (a + b) ^ 2015 = 4 :=
by
  sorry

end algebraic_expression_value_l463_463281


namespace max_sum_digits_l463_463677

theorem max_sum_digits (a b c : ℕ) (z : ℕ)
  (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) (h_digits : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum : float_eq_decimal_frac a b c z)
  (h_z : z ∈ {1, 2, 4, 5, 8, 10}) :
  ∃ a b c, a + b + c = 8 :=
sorry

def float_eq_decimal_frac (a b c z : ℕ) : Prop := 
  (100 * a + 10 * b + c) = 1000 / z

end max_sum_digits_l463_463677


namespace exist_X_equal_chords_l463_463378

-- Define points A, B on a diameter of a circle with center O
variables (O A B : Point)
variables (circle : Circle O)
variables (is_diameter : O.is_center_of_circle A B) -- indicates A and B lie on the diameter with O being center
variables (P Q X : Point)

-- Define the existence of point X
theorem exist_X_equal_chords : 
  ∃ (X : Point), (∃ (P : Point), (circle.contains P ∧ A.on_diameter_of_circle circle P X)) ∧ 
                 (∃ (Q : Point), (circle.contains Q ∧ B.on_diameter_of_circle circle Q X)) ∧ (dist X P = dist X Q) :=
sorry

end exist_X_equal_chords_l463_463378


namespace jacob_peter_age_ratio_l463_463579

theorem jacob_peter_age_ratio
  (Drew Maya Peter John Jacob : ℕ)
  (h1: Drew = Maya + 5)
  (h2: Peter = Drew + 4)
  (h3: John = 2 * Maya)
  (h4: John = 30)
  (h5: Jacob = 11) :
  Jacob + 2 = 1 / 2 * (Peter + 2) := by
  sorry

end jacob_peter_age_ratio_l463_463579


namespace cost_of_four_stamps_l463_463765

theorem cost_of_four_stamps (cost_one_stamp : ℝ) (h : cost_one_stamp = 0.34) : 4 * cost_one_stamp = 1.36 := 
by
  rw [h]
  norm_num

end cost_of_four_stamps_l463_463765


namespace iso_triangle_congruent_side_length_l463_463603

noncomputable def side_length_of_isosceles_triangle : ℝ :=
  let side_length_eq_triangle := 2 in
  let area_eq_triangle := (Float.sqrt 3) in
  let sum_areas_isosceles := area_eq_triangle / 2 in
  side_length_eq_triangle

theorem iso_triangle_congruent_side_length : side_length_of_isosceles_triangle =
  Float.sqrt (13) / 4 :=
by
  sorry

end iso_triangle_congruent_side_length_l463_463603


namespace Mo_and_Bo_Neg_Pos_l463_463943

/-- Assume Mo and Bo are people in a city where people can either be 'positives' or 'negatives'. 
    Positives ask questions for which the correct answer is always "yes", and negatives ask questions
    for which the correct answer is always "no". Mo asked Jo the question: "Are Bo and I both negative?" 
    Based on this information, prove that Mo is negative and Bo is positive. -/
theorem Mo_and_Bo_Neg_Pos (positives negatives : Type) (Mo Bo Jo : positives ⊕ negatives) :
  (∀ (p : positives), ∃ (q : negatives), Jo = q ∨ Bo = p) →
  (∀ (n : negatives), ∃ (q : positives), Mo = n ∧ Bo = q) →
  sorry

end Mo_and_Bo_Neg_Pos_l463_463943


namespace area_enclosed_by_curves_l463_463642

theorem area_enclosed_by_curves :
  let f1 := λ x: ℝ, Real.sqrt x
  let f2 := λ x: ℝ, 2 - x
  let f3 := λ x: ℝ, - (1/3:ℝ) * x
  let S := (∫ x in (0:ℝ)..(1:ℝ), f1 x + (1/3:ℝ) * x) + (1/2) * (4/3) * 2
  S = 13/6 := 
by 
  sorry

end area_enclosed_by_curves_l463_463642


namespace cylinder_has_no_triangular_cross_section_l463_463438

inductive GeometricSolid
  | cylinder
  | cone
  | triangularPrism
  | cube

open GeometricSolid

-- Define the cross section properties
def can_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cone ∨ s = triangularPrism ∨ s = cube

-- Define the property where a solid cannot have a triangular cross-section
def cannot_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cylinder

theorem cylinder_has_no_triangular_cross_section :
  cannot_have_triangular_cross_section cylinder ∧
  ¬ can_have_triangular_cross_section cylinder :=
by
  -- This is where we state the proof goal
  sorry

end cylinder_has_no_triangular_cross_section_l463_463438


namespace blue_red_area_ratio_l463_463841

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l463_463841


namespace trajectory_of_P_l463_463749

-- Define the points F1 and F2
def F1 := (2, 0)
def F2 := (-2, 0)

-- Define the condition for the moving point P
def satisfying_condition (P : ℝ × ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ((dist P F1 + dist P F2) = (4 * a + 1 / a))

-- The main theorem stating the trajectory of point P
theorem trajectory_of_P (P : ℝ × ℝ) (a : ℝ) :
  satisfying_condition P a → is_ellipse P F1 F2 ∨ is_line_segment P F1 F2 := 
sorry

end trajectory_of_P_l463_463749


namespace problem1_problem2_l463_463267

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^3 - (a * x^2) - (b * x) + c

theorem problem1 (a b c : ℝ) 
  (h1 : deriv (f x a b c) 1 = 4) 
  (h2 : deriv (f x a b c) (-1) = 0) 
  (h3 : f (-1) a b c = 2) :
  f x (-1) 1 1 = x^3 + x^2 - x + 1 := 
  sorry

theorem problem2 :
  let f (x : ℝ) := x^3 + x^2 - x + 1 in
  ∃ x_min x_max : ℝ, -2 ≤ x_min ∧ x_min ≤ 1 ∧ -2 ≤ x_max ∧ x_max ≤ 1 ∧
    (∀ x ∈ Icc (-2 : ℝ) 1, f x_min ≤ f x ∧ f x ≤ f x_max) ∧ 
    f x_min = -1 ∧ f x_max = 2 :=
  sorry

end problem1_problem2_l463_463267


namespace find_absolute_difference_l463_463096

def condition_avg_sum (m n : ℝ) : Prop :=
  m + n + 5 + 6 + 4 = 25

def condition_variance (m n : ℝ) : Prop :=
  (m - 5) ^ 2 + (n - 5) ^ 2 = 8

theorem find_absolute_difference (m n : ℝ) (h1 : condition_avg_sum m n) (h2 : condition_variance m n) : |m - n| = 4 :=
sorry

end find_absolute_difference_l463_463096


namespace min_marked_cells_l463_463906

theorem min_marked_cells (marking : Fin 15 → Fin 15 → Prop) :
  (∀ i : Fin 15, ∃ j : Fin 15, ∀ k : Fin 10, marking i (j + k % 15)) ∧
  (∀ j : Fin 15, ∃ i : Fin 15, ∀ k : Fin 10, marking (i + k % 15) j) →
  ∃s : Finset (Fin 15 × Fin 15), s.card = 20 ∧ ∀ i : Fin 15, (∃ j, (i, j) ∈ s ∨ (j, i) ∈ s) :=
sorry

end min_marked_cells_l463_463906


namespace imaginary_part_of_z_l463_463748

noncomputable def omega1 : ℂ := -1/2 + (real.sqrt 3)/2 * complex.I
noncomputable def omega2 : ℂ := complex.of_real (real.cos (real.pi / 12)) + (complex.I * (complex.of_real (real.sin (real.pi / 12))))

noncomputable def z : ℂ := omega1 * omega2

theorem imaginary_part_of_z : z.im = real.sqrt 2 / 2 :=
sorry

end imaginary_part_of_z_l463_463748


namespace christine_sales_value_l463_463567

variable {X : ℝ}

def commission_rate : ℝ := 0.12
def personal_needs_percent : ℝ := 0.60
def savings_amount : ℝ := 1152
def savings_percent : ℝ := 0.40

theorem christine_sales_value:
  (savings_percent * (commission_rate * X) = savings_amount) → 
  (X = 24000) := 
by
  intro h
  sorry

end christine_sales_value_l463_463567


namespace tower_height_l463_463374

noncomputable theory
open Real

theorem tower_height (d1 d2 d3 : ℝ) (alpha beta gamma : ℝ) (h1 : d1 = 100) (h2 : d2 = 200) 
    (h3 : d3 = 300) (h_sum : alpha + beta + gamma = π / 2) :
    let x := 100 in
    tan alpha = x / d1 ∧ tan beta = x / d2 ∧ tan gamma = x / d3 :=
begin
  sorry
end

end tower_height_l463_463374


namespace marbles_problem_l463_463316

theorem marbles_problem (h_total: ℕ) (h_each: ℕ) (h_total_eq: h_total = 35) (h_each_eq: h_each = 7) :
    h_total / h_each = 5 := by
  sorry

end marbles_problem_l463_463316


namespace volume_of_enclosing_cube_l463_463568

theorem volume_of_enclosing_cube (h s : ℕ) (h_eq : h = 15) (s_eq : s = 8) :
  h * h * h = 3375 := by
  rw [h_eq]
  exact sorry

end volume_of_enclosing_cube_l463_463568


namespace cylinder_has_no_triangular_cross_section_l463_463439

inductive GeometricSolid
  | cylinder
  | cone
  | triangularPrism
  | cube

open GeometricSolid

-- Define the cross section properties
def can_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cone ∨ s = triangularPrism ∨ s = cube

-- Define the property where a solid cannot have a triangular cross-section
def cannot_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cylinder

theorem cylinder_has_no_triangular_cross_section :
  cannot_have_triangular_cross_section cylinder ∧
  ¬ can_have_triangular_cross_section cylinder :=
by
  -- This is where we state the proof goal
  sorry

end cylinder_has_no_triangular_cross_section_l463_463439


namespace possible_values_of_a_l463_463116

def sequence (a : ℤ) (n : ℕ) : ℤ :=
  if n = 0 then a
  else if n % 2 = 1 then 
    2 ^ (n / 2) * a - (2 ^ (n / 2) - 1) * 45
  else
    2 ^ (n / 2) * a - (2 ^ (n / 2) - 2) * 45

theorem possible_values_of_a : ∀ (a : ℤ), -100 ≤ a ∧ a ≤ 100 → 
  (∃ (n : ℕ), n > 0 ∧ sequence a n = a) → 
  a = 0 ∨ a = 30 ∨ a = 42 ∨ a = 45 :=
by sorry

end possible_values_of_a_l463_463116


namespace collinearity_F_G_O_l463_463331

-- Definitions of the geometric objects
def is_equilateral_triangle (A B C : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

def is_midpoint (M : Point) (X Y : Point) : Prop := 
  2 * dist M X = dist X Y ∧ 2 * dist M Y = dist X Y

def is_on_circumcircle (P O : Point) (ABC : Triangle) : Prop :=
  dist P O = circumradius ABC

noncomputable def collinear (O F G : Point) : Prop :=
  (∃ k, O.coord + k * (G.coord - O.coord) = F.coord)

-- Main theorem to prove collinearity given the conditions
theorem collinearity_F_G_O
  (A B C O D E P F G : Point)
  (h1 : is_equilateral_triangle A B C)
  (h2 : is_midpoint D A B)
  (h3 : is_midpoint E A C)
  (h4 : is_on_circumcircle P O (mkTriangle A B C))
  (h5 : intersection (line_through P D) (line_through A B) = F)
  (h6 : intersection (line_through P E) (line_through A C) = G) :
  collinear O F G :=
sorry

end collinearity_F_G_O_l463_463331


namespace oldest_child_age_l463_463402

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 :=
by
  sorry

end oldest_child_age_l463_463402


namespace alex_has_highest_final_result_l463_463756

def maria_final (start: ℕ) : ℕ :=
  let step1 := start - 2
  let step2 := step1 * 3
  step2 + 4

def alex_final (start: ℕ) : ℕ :=
  let step1 := start * 3
  let step2 := step1 - 3
  step2 + 4

def lee_final (start: ℕ) : ℕ :=
  let step1 := start - 2
  let step2 := step1 + 4
  step2 * 3

theorem alex_has_highest_final_result:
  let maria := maria_final 12
  let alex := alex_final 15
  let lee := lee_final 13
  alex > maria ∧ alex > lee := by {
    let maria := maria_final 12
    let alex := alex_final 15
    let lee := lee_final 13
    have h_maria: maria = 34 := by
      unfold maria_final
      rfl
    have h_alex: alex = 46 := by
      unfold alex_final
      rfl
    have h_lee: lee = 45 := by
      unfold lee_final
      rfl
    have h_alex_maria: alex > maria := by
      rw [h_alex, h_maria]
      exact Nat.lt_trans (by norm_num) (by norm_num)
    have h_alex_lee: alex > lee := by
      rw [h_alex, h_lee]
      exact Nat.lt_trans (by norm_num) (by norm_num)
    exact ⟨h_alex_maria, h_alex_lee⟩
} sorry

end alex_has_highest_final_result_l463_463756


namespace find_number_l463_463076

theorem find_number : ∃ x : ℝ, 0.20 * x - 4 = 6 ∧ x = 50 :=
by
  use 50
  split
  -- Showing the left-hand side of the conjunction
  show 0.20 * 50 - 4 = 6
  sorry
  -- Showing the right-hand side of the conjunction
  show 50 = 50
  rfl

end find_number_l463_463076


namespace fixed_distance_l463_463278

variables {V : Type} [inner_product_space ℝ V]
variables (a c p : V)

noncomputable def s := 9 / 8
noncomputable def v := 1 / 8

def condition (p a c : V) := ‖p - c‖ = 3 * ‖p - a‖

theorem fixed_distance (p a c : V) (h : condition p a c) :
  ∃ k : ℝ, ‖p - (s • a + v • c)‖ = k :=
begin
  -- Proof skipped
  sorry
end

end fixed_distance_l463_463278


namespace collinear_vectors_solution_l463_463280

theorem collinear_vectors_solution (x : ℝ) :
  let a : ℝ × ℝ := (-3, 3)
  ∧ let b : ℝ × ℝ := (3, x)
  ∧ ∃ (λ : ℝ), b = (λ • a) in
  x = -3 :=
by
  sorry

end collinear_vectors_solution_l463_463280


namespace sphere_radius_correct_l463_463413

-- Define the variable representing the edge length of the regular tetrahedron
def tetrahedron_edge_length : ℝ := 4 * real.sqrt 6

-- Define the function to calculate the radius of the sphere touching the faces of the tetrahedron.
def sphere_radius_touches_faces (edge_length : ℝ) : ℝ :=
  -- Since this is a non-computable expected result based on geometric properties, we'll add noncomputable.
  noncomputable 3

-- Prove that for the given edge length, the calculated radius of the sphere touching the faces is 3.
theorem sphere_radius_correct : (sphere_radius_touches_faces tetrahedron_edge_length) = 3 :=
by
  sorry

end sphere_radius_correct_l463_463413


namespace pq_value_l463_463136

def Q (x : ℝ) (p q : ℝ) : ℝ := x^2 + p*x + q

theorem pq_value :
  let Q := Q x p q,
  let r1 := Real.sin (π / 6),
  let r2 := Real.sin (5 * π / 6),
  r1 = 1 / 2 ∧ r2 = 1 / 2 →
  p = -(r1 + r2) ∧ q = r1 * r2 →
  p * q = -1 / 4 :=
by
  intros Q r1 r2 h_roots h_vieta
  sorry

end pq_value_l463_463136


namespace polynomial_has_roots_l463_463173

theorem polynomial_has_roots :
  ∃ x : ℝ, x ∈ [-4, -3, -1, 2] ∧ (x^4 + 6 * x^3 + 7 * x^2 - 14 * x - 12 = 0) :=
by
  sorry

end polynomial_has_roots_l463_463173


namespace tenth_term_geometric_sequence_l463_463139

def a := 5
def r := Rat.ofInt 3 / 4
def n := 10

theorem tenth_term_geometric_sequence :
  a * r^(n-1) = Rat.ofInt 98415 / Rat.ofInt 262144 := sorry

end tenth_term_geometric_sequence_l463_463139


namespace middle_digit_is_3_l463_463088

theorem middle_digit_is_3 (d e f : ℕ) (hd : 0 ≤ d ∧ d ≤ 7) (he : 0 ≤ e ∧ e ≤ 7) (hf : 0 ≤ f ∧ f ≤ 7)
    (h_eq : 64 * d + 8 * e + f = 100 * f + 10 * e + d) : e = 3 :=
sorry

end middle_digit_is_3_l463_463088


namespace sum_of_possible_values_of_c_l463_463208

theorem sum_of_possible_values_of_c : 
  (∑ c in {c | c ∈ (Set.range (λ n : ℕ, if (∃ k : ℕ, 49 - 12 * c = k^2) then n else 0)) ∧ c ≠ 0}) = 6 :=
by
  sorry

end sum_of_possible_values_of_c_l463_463208


namespace blue_to_red_face_area_ratio_l463_463854

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l463_463854


namespace Patrick_fish_count_l463_463558

variable (Angus Patrick Ollie : ℕ)

-- Conditions
axiom h1 : Ollie + 7 = Angus
axiom h2 : Angus = Patrick + 4
axiom h3 : Ollie = 5

-- Theorem statement
theorem Patrick_fish_count : Patrick = 8 := 
by
  sorry

end Patrick_fish_count_l463_463558


namespace problem1_problem2_l463_463479
-- Problem 1


/--
Theorem: If \( 4x^2 = 1 \), then \( x = \pm \frac{1}{2} \).
-/
theorem problem1 (x : ℝ) (h : 4 * x^2 = 1) : x = 1 / 2 ∨ x = -1 / 2 :=
sorry

-- Problem 2


/--
Theorem: \( \sqrt[3]{8} + |\sqrt{3} - 1| - (\pi - 2023)^0 = \sqrt{3} \).
-/
theorem problem2 : (real.cbrt 8) + (abs (real.sqrt 3 - 1)) - (real.pi - 2023)^0 = real.sqrt 3 :=
sorry

end problem1_problem2_l463_463479


namespace surface_area_calculation_l463_463395

-- Conditions:
-- Original rectangular sheet dimensions
def length : ℕ := 25
def width : ℕ := 35
-- Dimensions of the square corners
def corner_side : ℕ := 7

-- Surface area of the interior calculation
noncomputable def surface_area_interior : ℕ :=
  let original_area := length * width
  let corner_area := corner_side * corner_side
  let total_corner_area := 4 * corner_area
  original_area - total_corner_area

-- Theorem: The surface area of the interior of the resulting box
theorem surface_area_calculation : surface_area_interior = 679 := by
  -- You can fill in the details to compute the answer
  sorry

end surface_area_calculation_l463_463395


namespace num_days_correct_l463_463300

variable (x : ℝ)

-- Definitions based on conditions
def daily_production_per_cow := (x + 3) / ((x + 2) * (x + 4))
def total_daily_production := (x + 4) * daily_production_per_cow
def days_needed := (x + 7) / total_daily_production

-- Proving the equivalence
theorem num_days_correct : days_needed x = (x * (x + 2) * (x + 7)) / ((x + 3) * (x + 4)) :=
by 
  unfold days_needed total_daily_production daily_production_per_cow
  sorry

end num_days_correct_l463_463300


namespace part_one_part_two_l463_463650

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + a * x + 6

theorem part_one (x : ℝ) : ∀ a, a = 5 → f x a < 0 ↔ -3 < x ∧ x < -2 :=
by
  sorry

theorem part_two : ∀ a, (∀ x, f x a > 0) ↔ - 2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 :=
by
  sorry

end part_one_part_two_l463_463650


namespace total_cost_eq_1400_l463_463346

theorem total_cost_eq_1400 (stove_cost : ℝ) (wall_repair_fraction : ℝ) (wall_repair_cost : ℝ) (total_cost : ℝ)
  (h₁ : stove_cost = 1200)
  (h₂ : wall_repair_fraction = 1/6)
  (h₃ : wall_repair_cost = stove_cost * wall_repair_fraction)
  (h₄ : total_cost = stove_cost + wall_repair_cost) :
  total_cost = 1400 :=
sorry

end total_cost_eq_1400_l463_463346


namespace amount_of_tin_approx_l463_463078

-- Define the variables and constants from conditions
def weight_of_bar : Real := 50
def weight_loss_in_water : Real := 5
def loss_per_kg_tin : Real := 1.375
def loss_per_kg_silver : Real := 0.075
def ratio_tin_to_silver : Real := 2 / 3

noncomputable def amount_of_silver (T : Real) : Real := 
  weight_of_bar - T

noncomputable def weight_loss_calculation (T : Real) : Real :=
  loss_per_kg_tin * T + loss_per_kg_silver * (weight_of_bar - T)

theorem amount_of_tin_approx : 
  ∃ T : Real, 
    T + amount_of_silver(T) = weight_of_bar ∧ 
    T / amount_of_silver(T) = ratio_tin_to_silver ∧ 
    weight_loss_calculation(T) = weight_loss_in_water 
    ∧ T ≈ 3.361 :=
sorry

end amount_of_tin_approx_l463_463078


namespace suitable_graph_for_air_composition_is_pie_chart_l463_463115

/-- The most suitable type of graph to visually represent the percentage 
of each component in the air is a pie chart, based on the given conditions. -/
theorem suitable_graph_for_air_composition_is_pie_chart 
  (bar_graph : Prop)
  (line_graph : Prop)
  (pie_chart : Prop)
  (histogram : Prop)
  (H1 : bar_graph → comparing_quantities)
  (H2 : line_graph → display_data_over_time)
  (H3 : pie_chart → show_proportions_of_whole)
  (H4 : histogram → show_distribution_of_dataset) 
  : suitable_graph_to_represent_percentage = pie_chart :=
sorry

end suitable_graph_for_air_composition_is_pie_chart_l463_463115


namespace angles_of_triangle_tangent_identity_l463_463768

namespace TriangleTangentProblem

theorem angles_of_triangle_tangent_identity
  (α β γ : ℝ)
  (h₁ : α + β + γ = 180) :
  tan (α / 2) * tan (β / 2) + tan (β / 2) * tan (γ / 2) + tan (γ / 2) * tan (α / 2) = 1 :=
by sorry

end TriangleTangentProblem

end angles_of_triangle_tangent_identity_l463_463768


namespace log_equation_solution_l463_463778

theorem log_equation_solution : log 2 (4 / 7) + log 2 7 = 2 := by
  -- First, use the property of logarithms to split the fraction:
  have h1 : log 2 (4 / 7) = log 2 4 - log 2 7 := by sorry
  -- Then, use the second property to combine the terms:
  have h2 : log 2 4 - log 2 7 + log 2 7 = log 2 4 := by sorry
  -- Simplify log 2 4 using the power rule:
  have h3 : log 2 4 = log 2 (2^2) := by sorry
  have h4 : log 2 (2^2) = 2 := by sorry
  -- Conclude the original equation:
  show log 2 (4 / 7) + log 2 7 = 2 from by
    rw [h1, h2, h3, h4]
  sorry

end log_equation_solution_l463_463778


namespace sin_alpha_l463_463313

-- Define the point P and the coordinates
def P : ℝ × ℝ := (-1, 3)
def x : ℝ := P.1
def y : ℝ := P.2

-- Define the distance from the origin
def OP : ℝ := Real.sqrt (x^2 + y^2)

-- State the theorem
theorem sin_alpha (α : ℝ) (hx : x = -1) (hy : y = 3) (hop : OP = Real.sqrt 10) :
  Real.sin α = 3 * Real.sqrt 10 / 10 :=
sorry

end sin_alpha_l463_463313


namespace cone_height_is_correct_l463_463949

noncomputable def volume : Real := 8000 * Real.pi

-- The statement of the problem according to the given conditions and required proof
theorem cone_height_is_correct : 
  ∃ (h : Real), let r := Real.sqrt (3 * volume / Real.pi) in h = r ≈ 36.3 := by
  sorry

end cone_height_is_correct_l463_463949


namespace incorrect_major_premise_error_l463_463257

theorem incorrect_major_premise_error :
  (∀ z : ℂ, z^2 ≥ 0) → (i^2 = -1) → (-1 > 0) → false :=
by
  intro h1
  intro h2
  intro h3
  -- Skipping the proof, as it is not required.
  sorry

end incorrect_major_premise_error_l463_463257


namespace triangle_inequality_of_sums_l463_463742

theorem triangle_inequality_of_sums (n : ℕ) (hn : n ≥ 3) (t : Fin n → ℝ) (ht : ∀ i, 0 < t i)
  (H : n^2 + 1 > (∑ i, t i) * (∑ i, 1 / t i)) :
  ∀ i j k : Fin n, i < j → j < k → i < k → t i + t j > t k ∧ t i + t k > t j ∧ t j + t k > t i :=
sorry

end triangle_inequality_of_sums_l463_463742


namespace car_average_speed_is_correct_l463_463488

noncomputable def average_speed_of_car : ℝ :=
  let d1 := 30
  let s1 := 30
  let d2 := 35
  let s2 := 55
  let t3 := 0.5
  let s3 := 70
  let t4 := 40 / 60 -- 40 minutes converted to hours
  let s4 := 36
  let t1 := d1 / s1
  let t2 := d2 / s2
  let d3 := s3 * t3
  let d4 := s4 * t4
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  total_distance / total_time

theorem car_average_speed_is_correct :
  average_speed_of_car = 44.238 := 
sorry

end car_average_speed_is_correct_l463_463488


namespace part_a_part_b_part_c_l463_463951

def is_valid_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def is_valid_leading_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def balanced_number_count (sum : ℕ) (valid_leading: ℕ → Prop) (valid_digit: ℕ → Prop) : ℕ :=
  (finset.range (sum + 1)).filter valid_leading.product (finset.range (sum + 1)).filter valid_digit |>.card

theorem part_a : balanced_number_count 8 is_valid_leading_digit is_valid_digit = 72 := sorry

theorem part_b : balanced_number_count 16 is_valid_leading_digit is_valid_digit = 9 := sorry

noncomputable def total_balanced_count : ℕ :=
  (finset.range 10).sum (λ s, balanced_number_count s is_valid_leading_digit is_valid_digit) +
  (finset.range (19 - 10 + 1)).sum (λ s, balanced_number_count (18 - s + 10) is_valid_leading_digit is_valid_digit)

theorem part_c : total_balanced_count = 615 := sorry

end part_a_part_b_part_c_l463_463951


namespace largest_possible_number_l463_463388

theorem largest_possible_number (ns : List ℕ) (h_sum : ns.sum = 2019)
  (h_no40 : ∀ n ∈ ns, n ≠ 40)
  (h_no_consec_sum_40 : ∀ (sublists : List (List ℕ)), sublists.sum ≠ 40) :
  ns.length ≤ 1019 :=
sorry

end largest_possible_number_l463_463388


namespace triangle_angles_l463_463447

theorem triangle_angles (A B C D M : Point) (h1 : Triangle MAD ≅ MAB)
  (h2 : ∠ BAM = 45) (h3 : ∠ MAD = 45) (h4 : ∠ MAB = 120)
  (h5 : ∠ MAC + ∠ MAD + ∠ MBC = 360) :
  ∠ BAM = 45 ∧ ∠ MAD = 45 ∧ ∠ AMB = 120 :=
sorry

end triangle_angles_l463_463447


namespace number_of_sets_S_l463_463653

def A := {-1, 1, 2}
def B := {y | ∃ x ∈ A, y = x^2}

theorem number_of_sets_S (S : Set ℤ) :
  (A ∩ B) ⊆ S ∧ S ⊆ (A ∪ B) → ∃ n, n = 8 :=
by {
  sorry
}

end number_of_sets_S_l463_463653


namespace units_digits_greater_than_tens_count_l463_463968

/--
Among all two-digit numbers, the number of those with a units digit greater than the tens digit is 36.
-/
theorem units_digits_greater_than_tens_count : 
  (finset.filter (λ n : ℕ, n % 10 > n / 10) (finset.range 100).filter (λ n, 10 ≤ n)).card = 36 :=
by
  sorry

end units_digits_greater_than_tens_count_l463_463968


namespace math_problem_l463_463264

noncomputable def fx (omega: ℝ) (varphi: ℝ) := λ x:ℝ, Real.sin (omega * x + varphi)

theorem math_problem
  (omega: ℝ) (varphi: ℝ)
  (h1: omega > 0)
  (h2: ∀ x y: ℝ, ∀ (h3: x < y), x ∈ Ioo (7 * Real.pi / 12) (5 * Real.pi / 6) → y ∈ Ioo (7 * Real.pi / 12) (5 * Real.pi / 6) → (fx omega varphi x <= fx omega varphi y ∨ fx omega varphi x >= fx omega varphi y))
  (h4: fx omega varphi (7 * Real.pi / 12) = -fx omega varphi (3 * Real.pi / 4)):
  (fx omega varphi (2 * Real.pi / 3) = 0) ∧
  ((∀ x, fx omega varphi (5 * Real.pi / 6 - x) = fx omega varphi x) → ∀ T, (T = Real.pi / ∃ T > 0, ∀ x, fx omega varphi (x + T) = fx omega varphi x)) ∧
  (¬ ∃ (x1 x2 x3 x4: ℝ), (0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < 2 * Real.pi) ∧ (fx omega varphi x1 = 1 ∧ fx omega varphi x2 = 1 ∧ fx omega varphi x3 = 1 ∧ fx omega varphi x4 = 1)) ∧
  ( (∃ (zero_count: ℕ), (zero_count = 5 ∧ ∀ x ∈ Ico (2 * Real.pi / 3) (13 * Real.pi / 6), fx omega varphi x = 0) → 8 / 3 < omega ∧ omega ≤ 3) ) :=
sorry

end math_problem_l463_463264


namespace find_a_l463_463786

def f (x : ℝ) : ℝ := (x / 4) + 2
def g (x : ℝ) : ℝ := 5 - x

theorem find_a (a : ℝ) (h : f(g(a)) = 4) : a = -3 :=
by 
  have h1 : f(g(a)) = (5 - a) / 4 + 2 := by 
    unfold f g
    simp
  have h2 : (5 - a) / 4 + 2 = 4 := h
  sorry

end find_a_l463_463786


namespace blue_faces_ratio_l463_463846

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l463_463846


namespace pointA_on_ellipse_no_value_a_exists_l463_463271

open Real

noncomputable def pointA (a : ℝ) : ℝ × ℝ := (a + 4, 0)
noncomputable def pointF (a : ℝ) : ℝ × ℝ := (a, 0)
noncomputable def circle_radius (a : ℝ) : ℝ := abs (a + 4 - a)
noncomputable def parabola (a : ℝ) (x : ℝ) : ℝ := sqrt (4 * a * x)
noncomputable def is_intersection_M (a : ℝ) (M : ℝ × ℝ) : Prop :=
  let (xM, yM) := M in
  yM^2 = 4 * a * xM ∧ (xM - (a + 4))^2 + yM^2 = 16
noncomputable def is_intersection_N (a : ℝ) (N : ℝ × ℝ) : Prop :=
  let (xN, yN) := N in
  yN^2 = 4 * a * xN ∧ (xN - (a + 4))^2 + yN^2 = 16

theorem pointA_on_ellipse (a : ℝ) (M N : ℝ × ℝ) (hM : is_intersection_M a M) (hN : is_intersection_N a N) (F := pointF a) (A := pointA a) : 
  ∃ e : Set (ℝ × ℝ), e = { p : ℝ × ℝ | dist p M + dist p N = dist F M + dist F N } ∧ A ∈ e := sorry

theorem no_value_a_exists (a : ℝ) (M N : ℝ × ℝ) (hM : is_intersection_M a M) (hN : is_intersection_N a N) (F := pointF a) (P := (fst M + fst N) / 2, (parabola a (fst M) + parabola a (fst N)) / 2) :
  ¬(dist F P = ((dist F M + dist F N) / 2)) := sorry

end pointA_on_ellipse_no_value_a_exists_l463_463271


namespace emma_missing_coins_l463_463586

theorem emma_missing_coins :
  let n := 100
  let lost := (n : ℚ) / 3
  let found := 3 / 4 * lost
  let total := n - lost + found
  let missing_fraction := (n - total) / n
  in missing_fraction = 5 / 12 :=
by
  -- Definitions
  let n := (100 : ℚ)
  let lost := n / 3
  let found := 3 / 4 * lost
  let total := n - lost + found
  let missing_fraction := (n - total) / n

  -- Show the calculation
  have h1 : n = 100 := by rfl
  have h2 : lost = 100 / 3 := by simp [lost, h1]
  have h3 : found = 3 / 4 * (100 / 3) := by simp [found, h2]
  have h4 : total = 100 - 100 / 3 + 3 / 4 * (100 / 3) := by simp [total, h3]
  have h5 : missing_fraction = (100 - (100 - 100 / 3 + 3 / 4 * (100 / 3))) / 100 := by simp [missing_fraction, h4]

  -- Conclude with the final equality
  have h_final : missing_fraction = 5 / 12 := by
    have : (100 - (100 - 100 / 3 + 3 / 4 * (100 / 3))) = (100 / 3 - 3 / 4 * (100 / 3)) := by ring
    simp [h5, this]
    ring_nf
    norm_num
    

  exact h_final

end emma_missing_coins_l463_463586


namespace number_of_valid_pairs_equiangular_polygons_l463_463658

theorem number_of_valid_pairs_equiangular_polygons 
    (P1 P2 : Type) 
    (n1 n2 : ℕ) 
    (x : ℝ) 
    (k : ℕ) 
    (hk : k > 1) 
    (hn1_neq_n2 : n1 ≠ n2) 
    (hn1_multiple_of_3 : ∃ m : ℕ, n1 = 3 * m) 
    (hx1 : x = 180 - 360 / n1.to_real) 
    (hx2 : k * x < 180) : 
    (number_of_valid_pairs P1 P2 n1 n2 x k = 2) := 
sorry

end number_of_valid_pairs_equiangular_polygons_l463_463658


namespace eval_expression_l463_463157

theorem eval_expression : 
  let a := 2999 in
  let b := 3000 in
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 := 
by 
  sorry

end eval_expression_l463_463157


namespace evaluate_expression_l463_463159

theorem evaluate_expression : 
  (3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001) :=
by
  sorry

end evaluate_expression_l463_463159


namespace area_formed_by_line_and_parabola_is_18_l463_463401

-- Condition 1: The line equation
def line (x : ℝ) : ℝ := x - 4

-- Condition 2: The parabola equation
def parabola (y : ℝ) : ℝ := y^2

-- Statement: Prove the area of the figure formed by the given line and parabola
theorem area_formed_by_line_and_parabola_is_18 : 
  ∃ (S : ℝ), S = ∫ y in -2..4, (y + 4 - (1/2)*y^2) ∧ S = 18 :=
by 
  sorry

end area_formed_by_line_and_parabola_is_18_l463_463401


namespace sum_of_valid_c_l463_463191

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_valid_c :
  (∑ c in Finset.filter (λ c, ∃ k, k * k = 49 - 12 * c) (Finset.range 5), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463191


namespace math_proof_problem_l463_463019

-- Define the given curve and line
def curve_C (ρ θ : ℝ) : Prop := ρ = 2 * real.sin θ
def line_l (t : ℝ) (x y : ℝ) : Prop := (x = 1 + t) ∧ (y = 2 - t)

-- Standard equation of the curve C
def standard_eq_curve_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Standard equation of the line l
def standard_eq_line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Equation of line l' that is parallel to l and tangent to the circle x^2 + (y-1)^2 = 1
def equation_line_l' (x y m : ℝ) : Prop := x + y + m = 0

-- Final proof statement
theorem math_proof_problem :
  (∀ (x y: ℝ), (∃ (ρ θ : ℝ), curve_C ρ θ ∧ x = ρ * real.cos θ ∧ y = ρ * real.sin θ) ↔ standard_eq_curve_C x y)
  ∧ (∀ (x y : ℝ), (∃ t, line_l t x y) ↔ standard_eq_line_l x y)
  ∧ (∀ m, m = -1 + real.sqrt 2 ∨ m = -1 - real.sqrt 2 → ∃ (x y : ℝ), x + y + m = 0 
       ∧ ∃ k, standard_eq_curve_C (x + k) (y + k) ∧ equation_line_l' x y m) :=
by sorry

end math_proof_problem_l463_463019


namespace blue_face_area_factor_l463_463799

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l463_463799


namespace calculate_percentage_gain_l463_463092

-- Define the parameters
def cost_price (num_bowls : ℕ) (price_per_bowl : ℕ) : ℕ := num_bowls * price_per_bowl

def selling_price (num_bowls : ℕ) (price_per_bowl : ℕ) : ℕ := num_bowls * price_per_bowl

def total_selling_price (sp1 sp2 : ℕ) : ℕ := sp1 + sp2

def gain (sp cp : ℕ) : ℕ := sp - cp

def percentage_gain (gain cp : ℕ) : ℚ :=
  (gain.to_rat / cp.to_rat) * 100

-- The main theorem statement
theorem calculate_percentage_gain :
  let cp := cost_price 300 20 in
  let sp1 := selling_price 200 25 in
  let sp2 := selling_price 80 30 in
  let total_sp := total_selling_price sp1 sp2 in
  let g := gain total_sp cp in
  percentage_gain g cp = 23.33 :=
by
  sorry

end calculate_percentage_gain_l463_463092


namespace chessboard_coloring_l463_463435

theorem chessboard_coloring (m n : ℕ) (h_m : m ≥ 2) :
  (∃ (a : ℕ), a = (n^2 - 3n + 3)*(nat.factorial (n-2)) ↔ 
  (∃ (colors : ℕ → ℕ), ∀ i < n, colors i < m ∧ (i = 0 ∨ colors i ≠ colors (i-1)))) :=
sorry

end chessboard_coloring_l463_463435


namespace hyperbola_eccentricity_l463_463627

variable {a b c e : ℝ}
variable {P F₁ F₂ : Point}

-- Given conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : ∃ P : Point, (|PF₁| - |PF₂|)^2 = b^2 - 3ab

-- Proving the eccentricity of the hyperbola is √17
theorem hyperbola_eccentricity : e = sqrt 17 := by
  sorry

end hyperbola_eccentricity_l463_463627


namespace find_a_minus_b_l463_463637

theorem find_a_minus_b (a b : ℝ) (h1: ∀ x : ℝ, (ax^2 + bx - 2 = 0 → x = -2 ∨ x = -1/4)) : (a - b = 5) :=
sorry

end find_a_minus_b_l463_463637


namespace expected_value_of_X_l463_463956

def is_optimal_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  (n >= 1000 ∧ n < 10000) ∧
  (digits.count 8) % 2 = 0

def optimal_numbers : set ℕ := { n | is_optimal_number n }

noncomputable def E_X : ℚ :=
  let total_numbers := 9000
  let optimal_count := 460
  let p := optimal_count / total_numbers
  10 * p

theorem expected_value_of_X : E_X = 23 / 45 :=
  by
  sorry

end expected_value_of_X_l463_463956


namespace can_not_buy_both_phones_l463_463069

noncomputable def alexander_salary : ℕ := 125000
noncomputable def natalia_salary : ℕ := 61000
noncomputable def utilities_transport_household_expenses : ℕ := 17000
noncomputable def loan_expenses : ℕ := 15000
noncomputable def cultural_theater : ℕ := 5000
noncomputable def cultural_cinema : ℕ := 2000
noncomputable def crimea_savings : ℕ := 20000
noncomputable def dining_out_weekdays_cost : ℕ := 1500 * 20
noncomputable def dining_out_weekends_cost : ℕ := 3000 * 10
noncomputable def phone_A_cost : ℕ := 57000
noncomputable def phone_B_cost : ℕ := 37000

theorem can_not_buy_both_phones :
  let total_expenses := utilities_transport_household_expenses + loan_expenses + cultural_theater + cultural_cinema + crimea_savings + dining_out_weekdays_cost + dining_out_weekends_cost in
  let total_income := alexander_salary + natalia_salary in
  let net_income := total_income - total_expenses in
    total_expenses = 119000 ∧ net_income = 67000 ∧ 67000 < (phone_A_cost + phone_B_cost) :=
by 
  intros;
  sorry

end can_not_buy_both_phones_l463_463069


namespace minute_hand_moves_180_degrees_l463_463220

noncomputable def minute_hand_angle_6_15_to_6_45 : ℝ :=
  let degrees_per_hour := 360
  let hours_period := 0.5
  degrees_per_hour * hours_period

theorem minute_hand_moves_180_degrees :
  minute_hand_angle_6_15_to_6_45 = 180 :=
by
  sorry

end minute_hand_moves_180_degrees_l463_463220


namespace direct_proportion_function_l463_463685

theorem direct_proportion_function (m : ℝ) (h : ∀ x : ℝ, -2*x + m = k*x → m = 0) : m = 0 :=
sorry

end direct_proportion_function_l463_463685


namespace chi_square_correlation_l463_463045

theorem chi_square_correlation:
  ∀ (K2 : ℝ) (a b c d n : ℝ),
  (
    (∀ (indep_test : Prop), (K2 = \frac {n(ad - bc)^2}{(a + b)(c + d)(a + c)(b + d)})
     → ¬indep_test) ∧
    (¬ (K2 → infer_relation))
  ) →
  (K2 → (a, b, c, d, n) →
  greater_correlation)
:= 
sorry

end chi_square_correlation_l463_463045


namespace evaluate_expression_l463_463162

open Real

def a := 2999
def b := 3000
def delta := b - a

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 :=
by
  let a := 2999
  let b := 3000
  have h1 : b - a = 1 := by sorry
  calc
    3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = a^3 + b^3 - ab^2 - a^2b := by sorry
                                            ... = (b - a)^2 * (b + a)       := by sorry
                                            ... = (1)^2 * (b + a)           := by
                                                                           rw [h1]
                                                                           exact sorry
                                            ... = 3000 + 2999               := by
                                                                           exact sorry
                                            ... = 5999                     := rfl

end evaluate_expression_l463_463162


namespace boys_camp_total_l463_463062

theorem boys_camp_total (T : ℕ) 
  (h1 : 0.20 * T = (0.20 : ℝ) * T) 
  (h2 : (0.30 : ℝ) * (0.20 * T) = (0.30 : ℝ) * (0.20 * T)) 
  (h3 : (0.70 : ℝ) * (0.20 * T) = 63) :
  T = 450 :=
by
  sorry

end boys_camp_total_l463_463062


namespace tan_2a_value_beta_value_l463_463243

variable {a β : ℝ}

-- Conditions
axiom h_cos_a : cos a = 1 / 7
axiom h_cos_a_sub_beta : cos (a - β) = 13 / 14
axiom h_bounds : ∀ x : ℝ, 0 < β ∧ β < a ∧ a < π / 2

-- Problem (1): Find the value of tan 2a
theorem tan_2a_value : tan (2 * a) = - (8 * real.sqrt 3) / 47 :=
by
  -- Proof will go here
  sorry

-- Problem (2): Find the value of β
theorem beta_value : β = π / 3 :=
by
  -- Proof will go here
  sorry

end tan_2a_value_beta_value_l463_463243


namespace maximum_marks_l463_463467

theorem maximum_marks (M : ℝ) (h1 : 0.45 * M = 180) : M = 400 := 
by sorry

end maximum_marks_l463_463467


namespace domain_ln_x_squared_minus_2_l463_463576

theorem domain_ln_x_squared_minus_2 (x : ℝ) : 
  x^2 - 2 > 0 ↔ (x < -Real.sqrt 2 ∨ x > Real.sqrt 2) := 
by 
  sorry

end domain_ln_x_squared_minus_2_l463_463576


namespace range_of_f_l463_463186

noncomputable def f (x : ℝ) : ℝ :=
  let g := λ (y : ℝ), (y^(1/5) + (y^(1/5))⁻¹)
  1 / g (64 * g (16 * g (Real.log2 x)) / 5)

theorem range_of_f :
  set.range f = {t : ℝ | -2/5 ≤ t ∧ t < 0 ∨ 0 < t ∧ t ≤ 2/5} :=
sorry

end range_of_f_l463_463186


namespace circle_radius_l463_463948

theorem circle_radius (r : ℝ) (x y : ℝ) :
  x = π * r^2 ∧ y = 2 * π * r ∧ x + y = 100 * π → r = 10 := 
  by
  sorry

end circle_radius_l463_463948


namespace non_working_games_l463_463761

def total_games : ℕ := 30
def working_games : ℕ := 17

theorem non_working_games :
  total_games - working_games = 13 := 
by 
  sorry

end non_working_games_l463_463761


namespace blue_area_factor_12_l463_463822

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l463_463822


namespace align_circles_with_nonoverlapping_arcs_l463_463018

theorem align_circles_with_nonoverlapping_arcs (k : ℕ) 
  (φ : fin k → ℝ) 
  (hφ : ∀ i, φ i < 180 / (k^2 - k + 1)) : 
  ∃ θ, ∀ i, ¬ ∃ j, (φ i + θ) % 360 = φ j := 
sorry

end align_circles_with_nonoverlapping_arcs_l463_463018


namespace hershel_betta_fish_l463_463661

variable (B : ℕ) (G : ℕ := 15) (BexBetta : ℕ := (2 * B) / 5) (BexGold : ℕ := G / 3)

theorem hershel_betta_fish : 
  let T := (7 * B) / 5 + G + 5
  in (1 / 2) * T = 17 → B = 10 :=
by
  intro h1
  -- Start from the condition given after gifting the fish
  have h2 : T = 34 := by
    rw [h1]
    ring
  -- Substitute T and solve B
  have h3 : ((7 * B) / 5 + 20 = 34) := by sorry
  sorry

end hershel_betta_fish_l463_463661


namespace broken_line_equals_perimeter_l463_463697

theorem broken_line_equals_perimeter (A B C A1 B1 C1 A2 B2 C2 : Point) (triangle_ABC : Triangle ABC)
    (hA1 : foot_of_altitude A A1 triangle_ABC)
    (hB1 : foot_of_altitude B B1 triangle_ABC)
    (hC1 : foot_of_altitude C C1 triangle_ABC)
    (hA2 : midpoint A2 B C)
    (hB2 : midpoint B2 A C)
    (hC2 : midpoint C2 A B):
    length (Segment (A1, B2)) + length (Segment (B2, C1)) + length (Segment (C1, A2)) +
    length (Segment (A2, B1)) + length (Segment (B1, C2)) + length (Segment (C2, A1)) =
    perimeter triangle_ABC := 
sorry

end broken_line_equals_perimeter_l463_463697


namespace find_u_minus_v_l463_463290

theorem find_u_minus_v (u v : ℚ) (h1 : 5 * u - 6 * v = 31) (h2 : 3 * u + 5 * v = 4) : u - v = 5.3 := by
  sorry

end find_u_minus_v_l463_463290


namespace quadratic_distinct_roots_l463_463047

theorem quadratic_distinct_roots :
  ∀ a b c : ℝ, a ≠ 0 ∧ b = 4 ∧ c = -4 → (b ^ 2 - 4 * a * c > 0) :=
by
  intros a b c ha hb hc
  rw [hb, hc]
  sorry

end quadratic_distinct_roots_l463_463047


namespace eggs_per_basket_is_15_l463_463775

-- Definitions from the conditions in a)
def yellow_eggs : ℕ := 30
def pink_eggs : ℕ := 45
def min_eggs_per_basket : ℕ := 5
def is_divisor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

-- Lean 4 statement equivalent to the mathematical problem
theorem eggs_per_basket_is_15
  (h1 : ∃ k : ℕ, yellow_eggs = k * min_eggs_per_basket ∧ pink_eggs = k * min_eggs_per_basket)
  (h2 : min_eggs_per_basket ≤ yellow_eggs ∧ min_eggs_per_basket ≤ pink_eggs)
  (h3 : ∀ d, is_divisor d yellow_eggs ∧ is_divisor d pink_eggs → d ≥ min_eggs_per_basket → d ∈ {5, 15}) :
  ∃ b : ℕ, b = 15 ∧ is_divisor b yellow_eggs ∧ is_divisor b pink_eggs :=
by
  sorry

end eggs_per_basket_is_15_l463_463775


namespace rise_in_water_level_after_submerging_cone_l463_463492

def volume_cone (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h

def base_area_vessel (length width : ℝ) : ℝ := length * width

def rise_in_water_level (V A : ℝ) : ℝ := V / A

theorem rise_in_water_level_after_submerging_cone :
  let r := 5 -- radius of cone's base in cm
  let h := 15 -- height of the cone in cm
  let length := 20 -- length of the rectangular vessel's base in cm
  let width := 15 -- width of the rectangular vessel's base in cm
  let V := volume_cone r h
  let A := base_area_vessel length width
  rise_in_water_level V A = (5 / 12) * real.pi :=
by
  sorry

end rise_in_water_level_after_submerging_cone_l463_463492


namespace arccos_one_half_eq_pi_third_l463_463984

theorem arccos_one_half_eq_pi_third : Real.arccos (1 / 2) = Real.pi / 3 := by
  -- Using the condition that cos(pi / 3) = 1 / 2
  have h : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  exact Real.arccos_cos h

end arccos_one_half_eq_pi_third_l463_463984


namespace small_triangle_count_l463_463608

theorem small_triangle_count (n : ℕ) (h : n = 2009) : (2 * n + 1) = 4019 := 
by {
    sorry
}

end small_triangle_count_l463_463608


namespace xiao_ming_arrival_time_l463_463919

def left_home (departure_time : String) : Prop :=
  departure_time = "6:55"

def time_spent (duration : Nat) : Prop :=
  duration = 30

def arrival_time (arrival : String) : Prop :=
  arrival = "7:25"

theorem xiao_ming_arrival_time :
  left_home "6:55" → time_spent 30 → arrival_time "7:25" :=
by sorry

end xiao_ming_arrival_time_l463_463919


namespace sin_cos_sum_l463_463682

theorem sin_cos_sum (θ : ℝ) (b : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : cos (2 * θ) = b) :
  sin θ + cos θ = Real.sqrt (2 - b) :=
sorry

end sin_cos_sum_l463_463682


namespace tens_digit_of_sum_l463_463416

theorem tens_digit_of_sum (a b c : ℕ) (h : a = c + 3) (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) :
    ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ (202 * c + 20 * b + 303) % 100 = t ∧ t / 10 = 1 :=
by
  use (20 * b + 3)
  sorry

end tens_digit_of_sum_l463_463416


namespace blue_face_area_greater_than_red_face_area_l463_463827

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l463_463827


namespace typing_time_l463_463974

theorem typing_time (barbara_max_speed : ℕ)
  (jim_speed : ℕ)
  (jim_time : ℕ)
  (monica_speed : ℕ)
  (monica_time : ℕ)
  (mandatory_break_thresh : ℕ)
  (mandatory_break_time : ℕ)
  (document_length : ℕ) :
  barbara_max_speed = 212 → jim_speed = 100 → jim_time = 20 →
  monica_speed = 150 → monica_time = 10 →
  mandatory_break_thresh = 25 → mandatory_break_time = 5 →
  document_length = 3440 →
  jim_speed * jim_time + monica_speed * monica_time >= document_length →
  jim_time + monica_time > mandatory_break_thresh →
  (jim_time + monica_time + mandatory_break_time) = 35 :=
by {
  intros hb hj1 htj hm1 htm hth hbr hdl hwords hbreak,
  sorry
}

end typing_time_l463_463974


namespace square_circle_ratio_l463_463394

theorem square_circle_ratio (a b : ℝ) 
  (h1 : ∃ r1, (a = 2 * r1 * √2) ∧ (r1 = b * √2 / 2))
  (h2 : ∃ r2, (r2 = r1 * √2) ∧ (a = 2 * r2 * √2)) :
  a / b = 2 * √2 :=
sorry

end square_circle_ratio_l463_463394


namespace students_behind_Yoongi_l463_463482

theorem students_behind_Yoongi 
  (total_students : ℕ) 
  (position_Jungkook : ℕ) 
  (students_between : ℕ) 
  (position_Yoongi : ℕ) : 
  total_students = 20 → 
  position_Jungkook = 1 → 
  students_between = 5 → 
  position_Yoongi = position_Jungkook + students_between + 1 → 
  (total_students - position_Yoongi) = 13 :=
by
  sorry

end students_behind_Yoongi_l463_463482


namespace ak_divisibility_l463_463767

theorem ak_divisibility {a k m n : ℕ} (h : a ^ k % (m ^ n) = 0) : a ^ (k * m) % (m ^ (n + 1)) = 0 :=
sorry

end ak_divisibility_l463_463767


namespace sphere_radius_l463_463474

open Point3D

/-- Define the cube and midpoints -/
structure Cube (a : ℝ) :=
  (A B C D A1 B1 C1 D1 : Point3D)
  (edge_length : ∀ {P Q : Point3D}, P ≠ Q → dist P Q = a)
  (M : Point3D) (K : Point3D)
  (M_midpoint : midpoint A B M)
  (K_midpoint : midpoint C D K)

/-- Define the condition for the sphere radius -/
def sphere_radius_condition (c : Cube a) : ℝ :=
  let O := Point3D.midpoint c.A1 c.C1 in
  dist O c.M = dist O c.K ∧ dist O c.A1 = dist O c.C1

/-- Prove the radius of the sphere passing through points M, K, A1, and C1 is a√41/8 -/
theorem sphere_radius (a : ℝ) (c : Cube a) (cond : sphere_radius_condition c) : 
  dist (Point3D.midpoint c.A1 c.C1) c.M = a * Real.sqrt 41 / 8 :=
sorry

end sphere_radius_l463_463474


namespace solve_for_x_l463_463292

theorem solve_for_x (x : ℝ) (h : sqrt (2 / x + 3) = 5 / 3) : x = -9 := 
by 
  sorry

end solve_for_x_l463_463292


namespace anthony_success_rate_increase_l463_463124

theorem anthony_success_rate_increase :
  let initial_makes := 3
  let initial_attempts := 10
  let next_attempts := 28
  let next_makes := 3 * next_attempts / 4
  let total_makes := initial_makes + next_makes
  let total_attempts := initial_attempts + next_attempts
  let new_success_rate := total_makes / total_attempts * 100
  let initial_success_rate := initial_makes / initial_attempts * 100
  let success_rate_increase := new_success_rate - initial_success_rate
  in Int.round success_rate_increase = 33 :=
by
  sorry

end anthony_success_rate_increase_l463_463124


namespace car_distance_first_hour_l463_463877

theorem car_distance_first_hour (x : ℕ) (h1 : ∑ i in range 12, (x + 2 * i) = 672) : x = 45 :=
sorry

end car_distance_first_hour_l463_463877


namespace final_proof_l463_463255

-- Given definitions
def S₃ (a₁ d : ℕ) : ℕ := 3 * a₁ + 3 * d  -- Sum of first 3 terms of arithmetic sequence
def geom_seq (a₁ a₄ a₁₃ : ℕ) : Prop := (a₁ + 3 * d)^2 = a₁ * (a₁ + 12 * d)

-- Main problem partitions
def general_term_proved (a₁ d : ℕ) (a_n : ℕ → ℕ) : Prop :=
  S₃ a₁ d = 15 ∧ d ≠ 0 ∧
  geom_seq a₁ (a₁ + 3 * d) (a₁ + 12 * d) →
  (∀ n, a_n n = 2 * n + 1)

def Tₙ_proved (a_n : ℕ → ℕ) (Tₙ : ℕ → ℕ) : Prop :=
  (∀ n, b_n n = a_n (2^n)) →
  (∀ n, Tₙ n = (∑ i in range n, b_n i) →
  Tₙ n = 2^(n + 2) + n - 4)

theorem final_proof (a₁ d : ℕ) (a_n : ℕ → ℕ) (b_n Tₙ : ℕ → ℕ) : 
  general_term_proved a₁ d a_n ∧ Tₙ_proved a_n Tₙ := 
sorry

end final_proof_l463_463255


namespace solve_for_q_l463_463777

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 3 * p + 5 * q = 8) : q = 19 / 16 :=
by
  sorry

end solve_for_q_l463_463777


namespace k_value_l463_463175

noncomputable def find_k : ℚ := 49 / 15

theorem k_value :
  ∀ (a b : ℚ), (3 * a^2 + 7 * a + find_k = 0) ∧ (3 * b^2 + 7 * b + find_k = 0) →
                (a^2 + b^2 = 3 * a * b) →
                find_k = 49 / 15 :=
by
  intros a b h_eq_root h_rel
  sorry

end k_value_l463_463175


namespace meaningful_expression_range_l463_463602

theorem meaningful_expression_range (x : ℝ) (h1 : 3 * x + 2 ≥ 0) (h2 : x ≠ 0) : 
  x ∈ Set.Ico (-2 / 3) 0 ∪ Set.Ioi 0 := 
  sorry

end meaningful_expression_range_l463_463602


namespace william_ends_with_18_tickets_l463_463050

-- Define the initial number of tickets
def initialTickets : ℕ := 15

-- Define the tickets bought
def ticketsBought : ℕ := 3

-- Prove the total number of tickets William ends with
theorem william_ends_with_18_tickets : initialTickets + ticketsBought = 18 := by
  sorry

end william_ends_with_18_tickets_l463_463050


namespace max_product_xy_l463_463297

theorem max_product_xy (x y : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : 7 * x + 4 * y = 150) : xy = 200 :=
by
  sorry

end max_product_xy_l463_463297


namespace blue_faces_ratio_l463_463848

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l463_463848


namespace primes_between_60_and_80_l463_463665

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_between_60_and_80 :
  set.count (set_of (λ n, 60 < n ∧ n < 80 ∧ is_prime n)) = 5 :=
by sorry

end primes_between_60_and_80_l463_463665


namespace ounces_per_bowl_l463_463587

theorem ounces_per_bowl (oz_per_gallon : ℕ) (gallons : ℕ) (bowls_per_minute : ℕ) (minutes : ℕ) (total_ounces : ℕ) (total_bowls : ℕ) (oz_per_bowl : ℕ) : 
  oz_per_gallon = 128 → 
  gallons = 6 →
  bowls_per_minute = 5 →
  minutes = 15 →
  total_ounces = oz_per_gallon * gallons →
  total_bowls = bowls_per_minute * minutes →
  oz_per_bowl = total_ounces / total_bowls →
  round (oz_per_bowl : ℚ) = 10 :=
by
  sorry

end ounces_per_bowl_l463_463587


namespace count_outliers_l463_463514

def data_set : list ℝ := [55, 68, 72, 72, 78, 81, 81, 83, 95, 100]
def Q1 : ℝ := 72
def Q3 : ℝ := 83
def IQR : ℝ := Q3 - Q1
def lower_threshold : ℝ := Q1 - 1.5 * IQR
def upper_threshold : ℝ := Q3 + 1.5 * IQR
def is_outlier (x : ℝ) : Prop := x < lower_threshold ∨ x > upper_threshold

theorem count_outliers : (list.countp is_outlier data_set) = 2 := by sorry

end count_outliers_l463_463514


namespace can_not_buy_both_phones_l463_463070

noncomputable def alexander_salary : ℕ := 125000
noncomputable def natalia_salary : ℕ := 61000
noncomputable def utilities_transport_household_expenses : ℕ := 17000
noncomputable def loan_expenses : ℕ := 15000
noncomputable def cultural_theater : ℕ := 5000
noncomputable def cultural_cinema : ℕ := 2000
noncomputable def crimea_savings : ℕ := 20000
noncomputable def dining_out_weekdays_cost : ℕ := 1500 * 20
noncomputable def dining_out_weekends_cost : ℕ := 3000 * 10
noncomputable def phone_A_cost : ℕ := 57000
noncomputable def phone_B_cost : ℕ := 37000

theorem can_not_buy_both_phones :
  let total_expenses := utilities_transport_household_expenses + loan_expenses + cultural_theater + cultural_cinema + crimea_savings + dining_out_weekdays_cost + dining_out_weekends_cost in
  let total_income := alexander_salary + natalia_salary in
  let net_income := total_income - total_expenses in
    total_expenses = 119000 ∧ net_income = 67000 ∧ 67000 < (phone_A_cost + phone_B_cost) :=
by 
  intros;
  sorry

end can_not_buy_both_phones_l463_463070


namespace inequality_range_of_a_l463_463995

theorem inequality_range_of_a (a : ℝ) (h : a ∈ set.Icc (real.sqrt 2 / 2) 1) :
  ∀ x : ℝ, x ∈ set.Ioo 0 (1 / 2) → 4^x < real.log a x :=
by sorry

end inequality_range_of_a_l463_463995


namespace triangles_with_positive_area_count_l463_463286

/--
Given 16 points on the integer coordinate grid {1, 2, 3, 4} × {1, 2, 3, 4}, 
prove that the number of triangles with positive area that can be formed 
is 520.
-/
theorem triangles_with_positive_area_count : 
  let points := {(i, j) | i j : ℤ, 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4} in
  let count := ∑ S in (Finset.powersetLen 3 points), 
                if collinear S then 0 else 1 in
  count = 520 := 
sorry

end triangles_with_positive_area_count_l463_463286


namespace acute_angles_sum_bound_l463_463770

theorem acute_angles_sum_bound (x y z : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) (hz : 0 < z ∧ z < π / 2)
  (hcos : cos x ^ 2 + cos y ^ 2 + cos z ^ 2 = 1) : 
  (3 * π / 4) < x + y + z ∧ x + y + z < π :=
by
  sorry

end acute_angles_sum_bound_l463_463770


namespace max_intersections_circles_line_l463_463905

axiom disjoint_union_prod (X Y Z : Type) (n m k : ℕ) (hX : Finset.card X = n) (hY : Finset.card Y = m) (hZ : Finset.card Z = k) :
    (∃ (U : Type),  (Finset.card U = n * k) ∧ (Finset.card U = m * k)) → (∃ (W : Type), (Finset.card W = (n + m) * k))

theorem max_intersections_circles_line (n m : ℕ) (h1 : n = 3) (h2 : m = 1) : 
  let max_intersections := 2 * n + 2 * (n * (n - 1) / 2)
  in max_intersections = 12 := 
by {
  sorry 
}

end max_intersections_circles_line_l463_463905


namespace part1_part2_l463_463609

namespace LeanProof

variable (a : ℝ) (x : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * Real.log x - a * x ^ 2 + a / x ^ 2

-- Part (1): Prove that f(a / 2) > 0 for 0 < a < 1
theorem part1 : 0 < a → a < 1 → f (a / 2) > 0 :=
by
  intro ha1 ha2
  sorry

-- Part (2): Prove that f(x) has three zeros if and only if 0 < a < 1/2
theorem part2 : (∃ x y z, x ≠ y ∧ y ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) ↔ (0 < a ∧ a < 1/2) :=
by
  intro
  sorry

end LeanProof

end part1_part2_l463_463609


namespace train_crosses_man_in_6_seconds_l463_463077

/-- A train of length 240 meters, traveling at a speed of 144 km/h, will take 6 seconds to cross a man standing on the platform. -/
theorem train_crosses_man_in_6_seconds
  (length_of_train : ℕ)
  (speed_of_train : ℕ)
  (conversion_factor : ℕ)
  (speed_in_m_per_s : ℕ)
  (time_to_cross : ℕ)
  (h1 : length_of_train = 240)
  (h2 : speed_of_train = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_in_m_per_s = speed_of_train * conversion_factor)
  (h5 : speed_in_m_per_s = 40)
  (h6 : time_to_cross = length_of_train / speed_in_m_per_s) :
  time_to_cross = 6 := by
  sorry

end train_crosses_man_in_6_seconds_l463_463077


namespace inscribe_equal_circles_in_triangle_l463_463340

-- Given conditions
variable (Δ : Triangle) -- a triangle 
variable (C1 C2 : Circle) -- two equal circles

-- Define tangency properties
variable (tangent_to_two_sides : ∀ (c : Circle), tangent c Δ.side1 ∧ tangent c Δ.side2)
variable (tangent_to_each_other : tangent C1 C2)

-- Target statement: Prove the existence of such circles
theorem inscribe_equal_circles_in_triangle (Δ : Triangle) (C1 C2 : Circle)
  (eq_radius : C1.radius = C2.radius)
  (tangent_to_two_sides_C1 : tangent C1 Δ.side1 ∧ tangent C1 Δ.side2)
  (tangent_to_two_sides_C2 : tangent C2 Δ.side1 ∧ tangent C2 Δ.side2)
  (tangent_to_each_other : tangent C1 C2) :
  ∃ (C1 C2 : Circle), C1.radius = C2.radius ∧
                       (tangent C1 Δ.side1 ∧ tangent C1 Δ.side2) ∧
                       (tangent C2 Δ.side1 ∧ tangent C2 Δ.side2) ∧
                       tangent C1 C2 := sorry

end inscribe_equal_circles_in_triangle_l463_463340


namespace correct_options_l463_463234

open Real

variables (x y k : ℝ)
variables (F1 F2 M N B : ℝ×ℝ)

def ellipse := {p : ℝ × ℝ | (p.1^2)/4 + (p.2^2)/3 = 1}
def line_l := {p : ℝ × ℝ | p.2 = k * p.1 ∧ k ≠ 0}

def points_M_N_on_l : M ∈ line_l ∧ N ∈ line_l :=
{ 
  -- assumption that M and N are on line l
  left := sorry, 
  right := sorry 
}

def points_M_N_on_ellipse : M ∈ ellipse ∧ N ∈ ellipse :=
{ 
  -- assumption that M and N are on ellipse
  left := sorry, 
  right := sorry 
}

def angle_bisector := sorry
def intersects_x_y_axis : (∃ E G : ℝ×ℝ, E.2 = 0 ∧ G.1 = 0 ∧ G.2 = m) :=
sorry

theorem correct_options (H1 : M ∈ line_l) (H2 : N ∈ line_l) (H3 : M ∈ ellipse) 
  (H4 : N ∈ ellipse) (H5 : ∃ E G : ℝ×ℝ, E.2 = 0 ∧ G.1 = 0 ∧ G.2 = m) :
  (perimeter_quad MF1NF2 = 8) ∧ 
    (slope_BM * slope_BN = -3/4) :=
begin
  -- proof goes here
  sorry
end

end correct_options_l463_463234


namespace selection_of_teachers_l463_463549

theorem selection_of_teachers (male_teachers female_teachers : ℕ) (total_selection : ℕ) (total_ways only_female_ways : ℕ) :
  male_teachers = 3 → female_teachers = 6 → total_selection = 5 →
  total_ways = nat.choose 9 5 → only_female_ways = nat.choose 6 5 →
  (total_ways - only_female_ways = 120) :=
begin
  intros h_male h_female h_total h_ways h_only_female,
  rw [h_male, h_female, h_total, h_ways, h_only_female],
  norm_num,
end

end selection_of_teachers_l463_463549


namespace number_of_elements_is_six_l463_463404

-- Defining the conditions as described
def average_of_numbers (S : Set ℝ) (N : ℝ) : Prop :=
  (∑ x in S, x) / S.card = N

theorem number_of_elements_is_six
 (S : Set ℝ)
 (h_avg : average_of_numbers S 3.95)
 (S1 S2 S3 : Set ℝ)
 (hS1 : S1.card = 2 ∧ (∑ x in S1, x) = 8.4)
 (hS2 : S2.card = 2 ∧ (∑ x in S2, x) = 7.7)
 (hS3 : S3.card = 2 ∧ (∑ x in S3, x) = 7.6)
 (h_disjoint : Disjoint S1 S2 ∧ Disjoint S1 S3 ∧ Disjoint S2 S3 ∧ S1 ∪ S2 ∪ S3 = S) :
 S.card = 6 :=
begin
  sorry
end

end number_of_elements_is_six_l463_463404


namespace base_extension_1_kilometer_l463_463102

-- Definition of the original triangle with hypotenuse length and inclination angle
def original_triangle (hypotenuse : ℝ) (angle : ℝ) : Prop :=
  hypotenuse = 1 ∧ angle = 20

-- Definition of the extension required for the new inclination angle
def extension_required (new_angle : ℝ) (extension : ℝ) : Prop :=
  new_angle = 10 ∧ extension = 1

-- The proof problem statement
theorem base_extension_1_kilometer :
  ∀ (hypotenuse : ℝ) (original_angle : ℝ) (new_angle : ℝ),
    original_triangle hypotenuse original_angle →
    new_angle = 10 →
    ∃ extension : ℝ, extension_required new_angle extension :=
by
  -- Sorry is a placeholder for the actual proof
  sorry

end base_extension_1_kilometer_l463_463102


namespace extreme_value_f_max_b_a_plus_1_l463_463266

noncomputable def f (x : ℝ) := Real.exp x - x + (1/2)*x^2

noncomputable def g (x : ℝ) (a b : ℝ) := (1/2)*x^2 + a*x + b

theorem extreme_value_f :
  ∃ x, deriv f x = 0 ∧ f x = 3 / 2 :=
sorry

theorem max_b_a_plus_1 (a : ℝ) (b : ℝ) :
  (∀ x, f x ≥ g x a b) → b * (a+1) ≤ (a+1)^2 - (a+1)^2 * Real.log (a+1) :=
sorry

end extreme_value_f_max_b_a_plus_1_l463_463266


namespace complement_of_M_in_U_l463_463654

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

def M : Set ℕ := {x : ℕ | x^2 - 6 * x + 5 ≤ 0 ∧ x ∈ ℕ}

theorem complement_of_M_in_U : (U \ M) = {6, 7} :=
by
  sorry

end complement_of_M_in_U_l463_463654


namespace part_a_part_b_part_c_l463_463884

-- Define the operations of the three machines.
def machine1 (a b : ℕ) : ℕ × ℕ := (a + 1, b + 1)

def machine2 : ℕ × ℕ → Option (ℕ × ℕ)
| (a, b) := if a % 2 = 0 ∧ b % 2 = 0 then some (a / 2, b / 2) else none

def machine3 (a b c : ℕ) : ℕ × ℕ := (a, c)

-- Part (a) - Can we obtain card (1, 50) from (5, 19)?
theorem part_a : ∃ s : List (ℕ × ℕ), List.prod (map (λ f, f.2) s) = (1, 50) :=
sorry

-- Part (b) - Can we obtain card (1, 100) from (5, 19)?
theorem part_b : ¬∃ s : List (ℕ × ℕ), List.prod (map (λ f, f.2) s) = (1, 100) :=
sorry

-- Part (c) - Given (a, b) with a < b, for which n can we obtain (1, n)?
theorem part_c (a b n : ℕ) (h : a < b) : 
  (∃ k : ℕ, n = 1 + (Nat.gcd (b - a) (b - a)) * k) ↔ 
  (∃ s : List (ℕ × ℕ), List.prod (map (λ f, f.2) s) = (1, n)) :=
sorry

end part_a_part_b_part_c_l463_463884


namespace logarithmic_solution_l463_463780

theorem logarithmic_solution (x : ℝ) (h : log 2 x + (log 2 x / log 2 4) + (log 2 x / log 2 8) = 9) : 
  x = 2^(54/11) :=
by
  sorry 

end logarithmic_solution_l463_463780


namespace sqrt_difference_l463_463055

theorem sqrt_difference (a b : ℝ) (ha : a = 7 + 4 * Real.sqrt 3) (hb : b = 7 - 4 * Real.sqrt 3) :
  Real.sqrt a - Real.sqrt b = 2 * Real.sqrt 3 :=
sorry

end sqrt_difference_l463_463055


namespace perimeter_of_smaller_polygon_l463_463422

/-- The ratio of the areas of two similar polygons is 1:16, and the difference in their perimeters is 9.
Find the perimeter of the smaller polygon. -/
theorem perimeter_of_smaller_polygon (a b : ℝ) (h1 : a / b = 1 / 16) (h2 : b - a = 9) : a = 3 :=
by
  sorry

end perimeter_of_smaller_polygon_l463_463422


namespace circle_radius_l463_463947

theorem circle_radius (r : ℝ) (x y : ℝ) :
  x = π * r^2 ∧ y = 2 * π * r ∧ x + y = 100 * π → r = 10 := 
  by
  sorry

end circle_radius_l463_463947


namespace sphere_surface_area_l463_463431

theorem sphere_surface_area (V : ℝ) (h : V = 72 * Real.pi) : ∃ A, A = 36 * Real.pi * (2 ^ (2 / 3)) := 
by
  sorry

end sphere_surface_area_l463_463431


namespace game_positions_l463_463320

def spots := ["top-left", "top-right", "bottom-right", "bottom-left"]
def segments := ["top-left", "top-middle-left", "top-middle-right", "top-right", "right-top", "right-middle-top", "right-middle-bottom", "right-bottom", "bottom-right", "bottom-middle-right", "bottom-middle-left", "bottom-left", "left-top", "left-middle-top", "left-middle-bottom", "left-bottom"]

def cat_position_after_moves (n : Nat) : String :=
  spots.get! (n % 4)

def mouse_position_after_moves (n : Nat) : String :=
  segments.get! ((12 - (n % 12)) % 12)

theorem game_positions :
  cat_position_after_moves 359 = "bottom-right" ∧ 
  mouse_position_after_moves 359 = "left-middle-bottom" :=
by
  sorry

end game_positions_l463_463320


namespace divisor_is_four_l463_463929

theorem divisor_is_four (n d k l : ℤ) (hn : n % d = 3) (h2n : (2 * n) % d = 2) (hd : d > 3) : d = 4 :=
by
  sorry

end divisor_is_four_l463_463929


namespace find_B_find_b_l463_463698

variables {A B C : ℝ} (a b c : ℝ)

-- Condition 1: Acute triangle
axiom acute_triangle (h₁ : 0 < A) (h₂ : A < π / 2) (h₃ : 0 < B) (h₄ : B < π / 2) (h₅ : 0 < C) (h₆ : C < π / 2) : A + B + C = π

-- Condition 2: Sides opposite to angles A, B, C are a, b, c respectively
-- (This is implicit in the variable declarations)

-- Condition 3: a = 2b * sin A
def condition_3 (h₃ : a = 2 * b * Real.sin A) : Prop := true

-- Condition 4: a = 3 * √3
def condition_4 (h₄ : a = 3 * Real.sqrt 3) : Prop := true

-- Condition 5: c = 5
def condition_5 (h₅ : c = 5) : Prop := true

-- Question 1: Find the measure of angle B
theorem find_B (h₃ : a = 2 * b * Real.sin A) (acute : A < π / 2) (h₆ : 0 < B) (acute2 : B < π / 2) : B = π / 6 := 
  sorry

-- Question 2: Find b given a = 3√3, c = 5, and B = π / 6
theorem find_b (h₄ : a = 3 * Real.sqrt 3) (h₅ : c = 5) (B_angle : B = π / 6) : b = Real.sqrt 7 := 
  sorry

end find_B_find_b_l463_463698


namespace range_of_m_l463_463646

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m * x^2 + (m + 6) * x + 1

def discriminant_f' (m : ℝ) : ℝ := 4 * m^2 - 12 * (m + 6)

theorem range_of_m (m : ℝ) :
  (∃ (x : ℝ), has_max_and_min (f m x)) ↔ m < -3 ∨ m > 6 :=
sorry

end range_of_m_l463_463646


namespace cube_sum_identity_l463_463680

theorem cube_sum_identity (x : ℝ) (hx : x + x⁻¹ = 3) : x^3 + x⁻³ = 18 :=
by
  sorry

end cube_sum_identity_l463_463680


namespace gcd_1729_867_l463_463455

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by 
  sorry

end gcd_1729_867_l463_463455


namespace volume_of_soil_extracted_l463_463936

-- Define the dimensions of the pond
def length : ℝ := 20
def width : ℝ := 10
def height : ℝ := 5

-- State the theorem to prove the volume of the soil extracted
theorem volume_of_soil_extracted : length * width * height = 1000 := 
begin
  sorry
end

end volume_of_soil_extracted_l463_463936


namespace shelby_foggy_driving_time_l463_463776

def speed_sunny : ℝ := 2 / 3
def speed_foggy : ℝ := 5 / 12
def total_distance : ℝ := 30
def total_time : ℝ := 60

theorem shelby_foggy_driving_time (x : ℝ) :
  (speed_sunny * (total_time - x) + speed_foggy * x = total_distance) → x = 40 := by
  sorry

end shelby_foggy_driving_time_l463_463776


namespace seventh_fifth_tiles_difference_l463_463322

def side_length (n : ℕ) : ℕ := 2 * n - 1
def number_of_tiles (n : ℕ) : ℕ := (side_length n) ^ 2
def tiles_difference (n m : ℕ) : ℕ := number_of_tiles n - number_of_tiles m

theorem seventh_fifth_tiles_difference : tiles_difference 7 5 = 88 := by
  sorry

end seventh_fifth_tiles_difference_l463_463322


namespace geometric_series_sum_l463_463030

theorem geometric_series_sum :
  ∑ k in Finset.range 13, (3 * 2 ^ k) = 24573 :=
by {
  sorry
}

end geometric_series_sum_l463_463030


namespace decrease_in_area_of_triangle_flowerbed_l463_463957

theorem decrease_in_area_of_triangle_flowerbed (A : ℝ) (s : ℝ) (s_new : ℝ) (A_new : ℝ) :
  let original_area := 36 * Real.sqrt 3,
      original_side := Real.sqrt ((4 * original_area) / Real.sqrt 3),
      new_side := original_side - 3,
      new_area := (Real.sqrt 3 / 4) * new_side^2,
      decrease_in_area := original_area - new_area in 
  decrease_in_area = 15.75 * Real.sqrt 3 :=
by
  sorry

end decrease_in_area_of_triangle_flowerbed_l463_463957


namespace sum_of_valid_c_l463_463190

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_valid_c :
  (∑ c in Finset.filter (λ c, ∃ k, k * k = 49 - 12 * c) (Finset.range 5), c) = 6 :=
by
  sorry

end sum_of_valid_c_l463_463190


namespace problem_statement_l463_463272

theorem problem_statement (a b : ℝ) (h : a + b = 1) : 
  ((∀ (a b : ℝ), a + b = 1 → ab ≤ 1/4) ∧ 
   (∀ (a b : ℝ), ¬(ab ≤ 1/4) → ¬(a + b = 1)) ∧ 
   ¬(∀ (a b : ℝ), ab ≤ 1/4 → a + b = 1) ∧ 
   ¬(∀ (a b : ℝ), ¬(a + b = 1) → ¬(ab ≤ 1/4))) := 
sorry

end problem_statement_l463_463272


namespace simplify_trig_expression_l463_463389

open Real

theorem simplify_trig_expression (A : ℝ) (h1 : cos A ≠ 0) (h2 : sin A ≠ 0) :
  (1 - (cos A) / (sin A) + 1 / (sin A)) * (1 + (sin A) / (cos A) - 1 / (cos A)) = -2 * (cos (2 * A) / sin (2 * A)) :=
by
  sorry

end simplify_trig_expression_l463_463389


namespace max_intersections_l463_463896

theorem max_intersections (C₁ C₂ C₃ : Circle) (L : Line) : 
  greatest_points_of_intersection 3 1 = 12 :=
sorry

end max_intersections_l463_463896


namespace find_cost_price_l463_463954

-- Conditions
def initial_cost_price (C : ℝ) : Prop :=
  let SP := 1.07 * C
  let NCP := 0.92 * C
  let NSP := SP - 3
  NSP = 1.0304 * C

-- The problem is to prove the initial cost price C given the conditions
theorem find_cost_price (C : ℝ) (h : initial_cost_price C) : C = 75.7575 := 
  sorry

end find_cost_price_l463_463954


namespace z_in_fourth_quadrant_l463_463793

def z : ℂ := 1 - 2 * complex.i

theorem z_in_fourth_quadrant (z : ℂ) (h : z = 1 - 2 * complex.i) : 
  (z.re > 0) ∧ (z.im < 0) :=
by {
  intro h,
  rw h,
  split,
  calc (1 : ℝ) > 0 : by norm_num,
  calc (-2 : ℝ) < 0 : by norm_num,
}

end z_in_fourth_quadrant_l463_463793


namespace andrew_total_hours_l463_463553

theorem andrew_total_hours (days_worked : ℕ) (hours_per_day : ℝ)
    (h1 : days_worked = 3) (h2 : hours_per_day = 2.5) : 
    days_worked * hours_per_day = 7.5 := by
  sorry

end andrew_total_hours_l463_463553


namespace rob_travel_time_to_park_l463_463772

theorem rob_travel_time_to_park : 
  ∃ R : ℝ, 
    (∀ Tm : ℝ, Tm = 3 * R) ∧ -- Mark's travel time is three times Rob's travel time
    (∀ Tr : ℝ, Tm - 2 = R) → -- Considering Mark's head start of 2 hours
    R = 1 :=
sorry

end rob_travel_time_to_park_l463_463772


namespace geometric_sequence_tenth_term_l463_463141

theorem geometric_sequence_tenth_term :
  let a := 5
      r := (3 : ℚ) / 4
  in ∃ a₁₀ : ℚ, a₁₀ = a * r^9 ∧ a₁₀ = 98415 / 262144 := by
  sorry

end geometric_sequence_tenth_term_l463_463141


namespace weight_of_mixture_l463_463466

variable (A B : ℝ)
variable (ratio_A_B : A / B = 9 / 11)
variable (consumed_A : A = 26.1)

theorem weight_of_mixture (A B : ℝ) (ratio_A_B : A / B = 9 / 11) (consumed_A : A = 26.1) : 
  A + B = 58 :=
sorry

end weight_of_mixture_l463_463466


namespace ruiz_original_salary_l463_463774

theorem ruiz_original_salary (S : ℝ) (h : 1.06 * S = 530) : S = 500 :=
by {
  -- Proof goes here
  sorry
}

end ruiz_original_salary_l463_463774


namespace average_apples_per_hour_l463_463739

theorem average_apples_per_hour (A H : ℝ) (hA : A = 12) (hH : H = 5) : A / H = 2.4 := by
  -- sorry skips the proof
  sorry

end average_apples_per_hour_l463_463739
