import Algebra.Field.Basic
import Analysis.SpecialFunctions.Trigonometric
import Geometry.Euclidean.Triangle
import Mathlib
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Multiset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Init
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.EuclideanSpace
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Tactic.Suggest
import Mathlib.Topology.Instances.Real
import Real

namespace closest_integer_to_cube_root_of_250_l45_45494

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45494


namespace floor_ceiling_product_l45_45654

theorem floor_ceiling_product :
  (Int.floor (-6 - 0.5) * Int.ceil (6 + 0.5) *
   Int.floor (-5 - 0.5) * Int.ceil (5 + 0.5) *
   Int.floor (-4 - 0.5) * Int.ceil (4 + 0.5) *
   Int.floor (-3 - 0.5) * Int.ceil (3 + 0.5) *
   Int.floor (-2 - 0.5) * Int.ceil (2 + 0.5) *
   Int.floor (-1 - 0.5) * Int.ceil (1 + 0.5)) = -25401600 := 
sorry

end floor_ceiling_product_l45_45654


namespace new_percentage_girls_proof_l45_45591

-- Given initial conditions
variables (initial_students : ℕ) (percentage_girls : ℝ) (new_boys : ℕ)
variables (initial_students = 20) (percentage_girls = 0.40) (new_boys = 5)

-- Define the number of girls initially
def initial_girls : ℕ := (percentage_girls * initial_students).toNat

-- Define the total number of students after new boys join
def total_students : ℕ := initial_students + new_boys

-- The number of girls remains the same
def girls := initial_girls

-- Calculate the new percentage of girls
def new_percentage_girls : ℝ := (girls.toRat / total_students) * 100

theorem new_percentage_girls_proof : new_percentage_girls = 32 := 
by sorry

end new_percentage_girls_proof_l45_45591


namespace problem_eqn_l45_45311

theorem problem_eqn (a b c : ℝ) :
  (∃ r₁ r₂ : ℝ, r₁^2 + 3 * r₁ - 1 = 0 ∧ r₂^2 + 3 * r₂ - 1 = 0) ∧
  (∀ x : ℝ, (x^2 + 3 * x - 1 = 0) → (x^4 + a * x^2 + b * x + c = 0)) →
  a + b + 4 * c = -7 :=
by
  sorry

end problem_eqn_l45_45311


namespace volleyball_lineup_count_l45_45018

def volleyball_team := {p : ℕ // p = 15}
def triplets := {t : ℕ // t = 4}
def team_captain := {c : bool // c = true}
def other_players := {o : ℕ // o = (15 - 4 - 1)}

noncomputable def num_lineups (v : volleyball_team) (t : triplets) (c : team_captain) (o : other_players) : ℕ :=
  let case0 := Nat.choose 10 5
  let case1 := 4 * Nat.choose 10 4
  let case2 := 6 * Nat.choose 10 3
  case0 + case1 + case2

theorem volleyball_lineup_count :
  num_lineups ⟨15, rfl⟩ ⟨4, rfl⟩ ⟨true, rfl⟩ ⟨10, rfl⟩ = 1812 :=
  by
    sorry

end volleyball_lineup_count_l45_45018


namespace circle_equation_focus_parabola_origin_l45_45222

noncomputable def parabola_focus (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 4 * p * x

def passes_through_origin (x y : ℝ) : Prop :=
  (0 - x)^2 + (0 - y)^2 = x^2 + y^2

theorem circle_equation_focus_parabola_origin :
  (∃ x y : ℝ, parabola_focus 1 x y ∧ passes_through_origin x y)
    → ∃ k : ℝ, (x^2 - 2 * x + y^2 = k) :=
sorry

end circle_equation_focus_parabola_origin_l45_45222


namespace scrooge_share_l45_45167

def leftover_pie : ℚ := 8 / 9

def share_each (x : ℚ) : Prop :=
  2 * x + 3 * x = leftover_pie

theorem scrooge_share (x : ℚ):
  share_each x → (2 * x = 16 / 45) := by
  sorry

end scrooge_share_l45_45167


namespace sequence_third_and_fourth_terms_l45_45719

theorem sequence_third_and_fourth_terms (n : ℕ) (a : ℕ → ℤ) :
  (∀ n, a n = - (n : ℤ) ^ 2 + 7 * n + 9) →
  a 3 = 21 ∧ a 4 = 21 :=
by {
  intro h,
  split;
  { rw h,
    norm_num }
}

end sequence_third_and_fourth_terms_l45_45719


namespace modulus_of_z_l45_45251

noncomputable def z : ℂ := (Complex.I / (1 + 2 * Complex.I))

theorem modulus_of_z : Complex.abs z = (Real.sqrt 5) / 5 := by
  sorry

end modulus_of_z_l45_45251


namespace percent_decrease_area_pentagon_l45_45333

open Real

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * sqrt 3 / 2) * s ^ 2

noncomputable def area_pentagon (s : ℝ) : ℝ :=
  (sqrt (5 * (5 + 2 * sqrt 5)) / 4) * s ^ 2

noncomputable def diagonal_pentagon (s : ℝ) : ℝ :=
  (1 + sqrt 5) / 2 * s

theorem percent_decrease_area_pentagon :
  let s_p := sqrt (400 / sqrt (5 * (5 + 2 * sqrt 5)))
  let d := diagonal_pentagon s_p
  let new_d := 0.9 * d
  let new_s := new_d / ((1 + sqrt 5) / 2)
  let new_area := area_pentagon new_s
  (100 - new_area) / 100 * 100 = 20 :=
by
  sorry

end percent_decrease_area_pentagon_l45_45333


namespace area_triangle_ZGH_l45_45058

theorem area_triangle_ZGH :
  ∀ (length width : ℝ) (G H : ℝ × ℝ) (Z : ℝ × ℝ) (XW : ℝ) (GH : ℝ),
  length = 8 ∧ width = 6 ∧
  XW = Real.sqrt (length^2 + width^2) ∧
  GH = XW / 3 ∧
  Z = (0, 6) ∧
  (X, W) = (0, 0), (8, 6) ∧  -- corrected typo here, ensuring (X,W) placement
  G and H are points dividing XW into three equal segments (geometrically defined) →
  -- Therefore Statement
  (area GHZ = 8).

noncomputable definition area (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

Sorry

end area_triangle_ZGH_l45_45058


namespace marbles_left_in_the_box_l45_45439

-- Define the main problem parameters.
def total_marbles : ℕ := 50
def white_marbles : ℕ := 20
def blue_marbles : ℕ := (total_marbles - white_marbles) / 2
def red_marbles : ℕ := blue_marbles
def removed_marbles : ℕ := 2 * (white_marbles - blue_marbles)
def remaining_marbles : ℕ := total_marbles - removed_marbles

-- The theorem to prove the number of marbles left in the box.
theorem marbles_left_in_the_box : remaining_marbles = 40 := by
  unfold total_marbles white_marbles blue_marbles red_marbles removed_marbles remaining_marbles
  -- Here goes the calculus step simplification
  sorry

end marbles_left_in_the_box_l45_45439


namespace arithmetic_sequence_n_positive_l45_45699

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}
variable {d a1 : ℤ}

noncomputable def a (n : ℕ) := a1 + (n - 1) * d
noncomputable def S (n : ℕ) := n * (a1 + a1 + (n - 1) * d) / 2

theorem arithmetic_sequence_n_positive
  (h1 : a 11 - a 8 = 3)
  (h2 : S 11 - S 8 = 3) :
  (∃ n : ℕ, n > 0 ∧ a n > 0) :=
by {
  -- Note: This is the statement part. The proof is not required.
  sorry
}

end arithmetic_sequence_n_positive_l45_45699


namespace inequality_holds_l45_45817

-- Define the sequence with nested square roots.
def nested_sqrt (n : ℕ) : ℝ :=
  if n = 0 then 0 else (2 + nested_sqrt (n - 1))^.half

-- Define the theorem to prove the inequality with the given conditions
theorem inequality_holds (n : ℕ) (hn : n > 0) : 
  ((2 - nested_sqrt n) / (2 - nested_sqrt (n - 1))) > (1 / 4) :=
by
  sorry

end inequality_holds_l45_45817


namespace find_p_plus_q_l45_45805

variables (x y z : ℝ)

theorem find_p_plus_q (hx : x^2 + y^2 = 49)
                     (hy : y^2 + y * z + z^2 = 36)
                     (hz : x^2 + Real.sqrt 3 * x * z + z^2 = 25) :
  ∃ (p q : ℤ), p * Real.sqrt q = 2 * x * y + Real.sqrt 3 * y * z + x * z ∧ Nat.gcd q (q / (Int.natAbs q)) = 1 ∧ p + q = 30 :=
begin
  -- Reminder: Here you provide the proof.
  sorry
end

end find_p_plus_q_l45_45805


namespace probability_of_falling_bean_l45_45304

noncomputable def region_area : ℝ :=
  ∫ x in 0..1, exp x + 1

def total_area : ℝ := (exp 1 + 1)

def probability : ℝ := region_area / total_area

theorem probability_of_falling_bean :
  probability = 1 / (exp 1 + 1) :=
by
  sorry

end probability_of_falling_bean_l45_45304


namespace warehouse_bins_total_l45_45609

theorem warehouse_bins_total (x : ℕ) (h1 : 12 * 20 + x * 15 = 510) : 12 + x = 30 :=
by
  sorry

end warehouse_bins_total_l45_45609


namespace count_lucky_numbers_l45_45453

-- Definitions
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digits_sum_to_six (n : ℕ) : Prop := (n / 100 + (n / 10) % 10 + n % 10 = 6)

-- Proposition to prove
theorem count_lucky_numbers : {n : ℕ | is_three_digit_number n ∧ digits_sum_to_six n}.to_finset.card = 21 := 
by 
  sorry

end count_lucky_numbers_l45_45453


namespace salary_net_decrease_l45_45069

theorem salary_net_decrease (S : ℝ) : 
  let increased_salary := 1.4 * S,
      final_salary := increased_salary * 0.6 
  in final_salary = 0.84 * S → (final_salary - S) = -0.16 * S :=
by
  intros
  sorry

end salary_net_decrease_l45_45069


namespace exists_triangle_with_area_le_one_fourth_l45_45019

-- Definitions of the problem conditions:
variable (P Q R S A B C : Point)
variable (convex_PQRS : convex_quad P Q R S)
variable (inside_triangle_ABC : ∀ p ∈ {P, Q, R, S}, p ∈ △ A B C)
variable (area_ABC : area_triangle A B C = 1)

-- Theorem statement: There exists a triangle formed by three vertices of PQRS whose area is 1/4 or less.
theorem exists_triangle_with_area_le_one_fourth :
  ∃ X Y Z ∈ {P, Q, R, S}, {X, Y, Z}.size = 3 ∧ area_triangle X Y Z ≤ 1 / 4 :=
  sorry

end exists_triangle_with_area_le_one_fourth_l45_45019


namespace stickers_per_student_l45_45010

theorem stickers_per_student 
  (gold_stickers : ℕ) 
  (silver_stickers : ℕ) 
  (bronze_stickers : ℕ) 
  (students : ℕ)
  (h1 : gold_stickers = 50)
  (h2 : silver_stickers = 2 * gold_stickers)
  (h3 : bronze_stickers = silver_stickers - 20)
  (h4 : students = 5) : 
  (gold_stickers + silver_stickers + bronze_stickers) / students = 46 :=
by
  sorry

end stickers_per_student_l45_45010


namespace simplify_expression_l45_45852

theorem simplify_expression (y : ℝ) : 
  2 * y * (4 * y^2 - 3 * y + 1) - 6 * (y^2 - 3 * y + 4) = 8 * y^3 - 12 * y^2 + 20 * y - 24 := 
by
  sorry

end simplify_expression_l45_45852


namespace non_overlapping_carpet_cover_l45_45410

-- Room dimensions
def room_area := 6 * 3

-- Prove that it is possible to remove some carpets so that the remaining carpets do not overlap and cover more than \(2 \mu^2\).
theorem non_overlapping_carpet_cover (K : Set (Set ℝ)) (room : Set ℝ) 
  (h1 : room = set.univ ∨ (∀ x ∈ K, x ⊆ room) ∧ set.pairwise_disjoint K ∧ (set.Union K = room))
  : ∃ (C : (Set ℝ)), (C ⊆ K ∧ set.pairwise_disjoint C ∧ (set.measure (set.Union C) > 2)) :=
by
  sorry

end non_overlapping_carpet_cover_l45_45410


namespace find_angle_B_find_sin_A_l45_45638

-- Part (1): Proving the measure of angle B is π / 3
theorem find_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1 : c * Real.cos A - 2 * b * Real.cos B + a * Real.cos C = 0) 
  (h2 : A + B + C = Real.pi) : 
  B = Real.pi / 3 :=
begin
  sorry
end

-- Part (2): Proving the value of sin A given the conditions
theorem find_sin_A (a b c : ℝ) (A B : ℝ)
  (h1 : c * Real.cos B - a * Real.cos A = 20)
  (h2 : c > a)
  (h3 : a + b + c = 20 ∨ a + c = 13 ∨ b = 7) : 
  Real.sin A = 5 * Real.sqrt 3 / 14 :=
begin
  sorry
end

end find_angle_B_find_sin_A_l45_45638


namespace seating_round_table_l45_45323

theorem seating_round_table : 
  ∃ (n : ℕ), (n = 6) → 
  ∃ (people : Fin n), (people = ({A, B, C, D, E, F} : Fin 6)) → 
  (round_table : List (Set (Fin n))) → 
  (adjacency_condition : ∀ (sitting : List (Fin n)), 
    sitting.membership (A, B)) → 
  (num_ways_to_seat_A_B : ℕ), 
  num_ways_to_seat_A_B = 48 :=
begin
  sorry
end

end seating_round_table_l45_45323


namespace intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l45_45367

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + x^2 - 2 * a * x + a^2

-- Question Ⅰ
theorem intervals_of_monotonicity_when_a_eq_2 :
  (∀ x : ℝ, 0 < x ∧ x < (2 - Real.sqrt 2) / 2 → f x 2 > 0) ∧
  (∀ x : ℝ, (2 - Real.sqrt 2) / 2 < x ∧ x < (2 + Real.sqrt 2) / 2 → f x 2 < 0) ∧
  (∀ x : ℝ, (2 + Real.sqrt 2) / 2 < x → f x 2 > 0) := sorry

-- Question Ⅱ
theorem no_increasing_intervals_on_1_3_implies_a_ge_19_over_6 (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 0) → a ≥ (19 / 6) := sorry

end intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l45_45367


namespace smallest_n_is_4_l45_45117

noncomputable def smallest_n: ℕ :=
  let f : ℤ → ℕ := sorry in -- f is some function to be defined that satisfies the conditions
  classical.some (exists (n : ℕ) (f : ℤ → fin n), 
                   ∀ (A B : ℤ), A ≠ B ∧ |A - B| ∈ {5, 7, 12} → f A ≠ f B) -- conditions ensuring f satisfies constraints

theorem smallest_n_is_4 : smallest_n = 4 := 
by 
  -- Proof sketch:
  -- 1. Show that there cannot exist an f with the conditions for n = 3.
  -- 2. Construct such an f for n = 4.
  sorry

end smallest_n_is_4_l45_45117


namespace power_of_point_l45_45093

namespace ChordsIntersect

variables (A B C D P : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]

def AP := 4
def CP := 9

theorem power_of_point (BP DP : ℕ) :
  AP * BP = CP * DP -> (BP / DP) = 9 / 4 :=
by
  sorry

end ChordsIntersect

end power_of_point_l45_45093


namespace triangle_QST_area_l45_45380

noncomputable def dist3d (A B : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

theorem triangle_QST_area
  (P Q R S T : ℝ × ℝ × ℝ)
  (hPQ : dist3d P Q = 3)
  (hQR : dist3d Q R = 3)
  (hRS : dist3d R S = 3)
  (hST : dist3d S T = 3)
  (hTP : dist3d T P = 3)
  (hPQR : ∀ A B C : ℝ × ℝ × ℝ, ∠ A B C = 90)
  (hplanePQR_parallel_ST : ∀ x : ℝ × ℝ × ℝ, Plane P Q R x → Parallel x (Line S T)) :
  area Q S T = 4.5 := sorry

end triangle_QST_area_l45_45380


namespace S_even_cardinality_S_at_least_2_pow_k_l45_45946

variables (n k : ℕ)
variable S : Finset (EuclideanSpace (Fin n) ℤ)
variable friend : ℕ → ℕ → Prop -- friend relation
variable h_friend : ∀ x ∈ S, (Finset.filter (friend x) S).card = k -- each element has exactly k friends

-- Part 1: Prove S has an even number of elements
theorem S_even_cardinality : (S.card % 2 = 0) := sorry

-- Part 2: Prove S contains at least 2^k codes
theorem S_at_least_2_pow_k : (S.card ≥ 2^k) := sorry

end S_even_cardinality_S_at_least_2_pow_k_l45_45946


namespace distance_to_circumcenter_formula_l45_45022

theorem distance_to_circumcenter_formula
  (A B C P O : Point)
  (R a b c : Real)
  (area_ABC area_CAM area_AMB area_ABM area_BCM : Real) :
  (dist P O) ^ 2 = R ^ 2 -
    (a ^ 2 * area_CAM * area_AMB + b ^ 2 * area_ABM * area_BCM + c ^ 2 * area_BCM * area_CAM) / (area_ABC ^ 2) :=
  sorry

end distance_to_circumcenter_formula_l45_45022


namespace islanders_statements_l45_45582

-- Define the basic setup: roles of individuals
inductive Role
| knight
| liar
open Role

-- Define each individual's statement
def A_statement := λ (distance : ℕ), distance = 1
def B_statement := λ (distance : ℕ), distance = 2

-- Prove that given the conditions, the possible distances mentioned by the third and fourth islanders can be as specified
theorem islanders_statements :
  ∃ (C_statement D_statement : ℕ → Prop),
  (∀ distance, C_statement distance ↔ distance ∈ {1, 3, 4}) ∧ (∀ distance, D_statement distance ↔ distance = 2) :=
by
  sorry

end islanders_statements_l45_45582


namespace maximize_product_of_digits_l45_45450

theorem maximize_product_of_digits :
  ∃ (a b c d e : ℕ), a ∈ {1, 3, 5, 8, 9} ∧ b ∈ {1, 3, 5, 8, 9} ∧ c ∈ {1, 3, 5, 8, 9} ∧ d ∈ {1, 3, 5, 8, 9} ∧ e ∈ {1, 3, 5, 8, 9} ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  100 * a + 10 * b + c = 895 ∧
  100 * a + 10 * b + c * (10 * d + e) = 88143 :=
sorry

end maximize_product_of_digits_l45_45450


namespace delivery_charge_is_four_l45_45796

/-- Definitions based on the conditions --/
def pizza_cost : ℕ := 12
def num_pizzas_park : ℕ := 3
def num_pizzas_building : ℕ := 2
def distance_park : ℕ := 100 -- in meters
def distance_building : ℕ := 2000 -- in meters
def total_payment : ℕ := 64

/-- The theorem to prove the delivery charge for areas farther than 1 km from the pizzeria is $4 --/
theorem delivery_charge_is_four :
  let total_cost_pizzas := (num_pizzas_park * pizza_cost) + (num_pizzas_building * pizza_cost) in
  let delivery_charge := total_payment - total_cost_pizzas in
  delivery_charge = 4 := 
by
  sorry

end delivery_charge_is_four_l45_45796


namespace parabola_problem_l45_45732

-- defining the geometric entities and conditions
variables {x y k x1 y1 x2 y2 : ℝ}

-- the definition for the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- the definition for point M
def point_M (x y : ℝ) : Prop := (x = 0) ∧ (y = 2)

-- the definition for line passing through focus with slope k intersecting the parabola at A and B
def line_through_focus_and_k (x1 y1 x2 y2 k : ℝ) : Prop :=
  (y1 = k * (x1 - 1)) ∧ (y2 = k * (x2 - 1))

-- the definition for vectors MA and MB having dot product zero
def orthogonal_vectors (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 * x2 + y1 * y2 - 2 * (y1 + y2) + 4 = 0)

-- the main statement to be proved
theorem parabola_problem
  (h_parabola_A : parabola x1 y1)
  (h_parabola_B : parabola x2 y2)
  (h_point_M : point_M 0 2)
  (h_line_through_focus_and_k : line_through_focus_and_k x1 y1 x2 y2 k)
  (h_orthogonal_vectors : orthogonal_vectors x1 y1 x2 y2) :
  k = 1 :=
sorry

end parabola_problem_l45_45732


namespace closest_integer_to_cube_root_of_250_l45_45493

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45493


namespace number_of_people_joining_group_l45_45612

theorem number_of_people_joining_group (x : ℕ) (h1 : 180 / 18 = 10) 
  (h2 : 180 / (18 + x) = 9) : x = 2 :=
by
  sorry

end number_of_people_joining_group_l45_45612


namespace closest_cube_root_of_250_l45_45468

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45468


namespace closest_integer_to_cube_root_of_250_l45_45491

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45491


namespace islander_distances_l45_45581

theorem islander_distances (A B C D : ℕ) (k1 : A = 1 ∨ A = 2)
  (k2 : B = 2)
  (C_liar : C = 1) (is_knight : C ≠ 1) :
  C = 1 ∨ C = 3 ∨ C = 4 ∧ D = 2 :=
by {
  sorry
}

end islander_distances_l45_45581


namespace angle_between_skew_lines_l45_45404

noncomputable def angle_between_MN_BD' : ℝ :=
  let B  := (1, 1, 0)
  let D' := (0, 0, 1)
  let M  := (0.5, 0.5, 0)
  let N  := (0.5, 0, 0.5)
  let BD' := (λ x, (0:ℝ) - 1) ⟨1, 1, 0⟩ 
  let MN  := (λ x, (0:ℝ) - 0.5) ⟨0.5, 0, 0.5⟩ 
  let dot_product := 1
  let magnitude_BD' := Real.sqrt(3)
  let magnitude_MN := Real.sqrt(0.5)
  let cos_theta := dot_product / (magnitude_BD' * magnitude_MN)
  Real.arccos (Real.sqrt(6) / 3)

theorem angle_between_skew_lines :
  angle_between_MN_BD' = Real.arccos (Real.sqrt(6) / 3) :=
sorry

end angle_between_skew_lines_l45_45404


namespace closest_integer_to_cbrt_250_l45_45565

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45565


namespace total_pumpkin_weight_l45_45373

-- Conditions
def weight_first_pumpkin : ℝ := 4
def weight_second_pumpkin : ℝ := 8.7

-- Statement
theorem total_pumpkin_weight :
  weight_first_pumpkin + weight_second_pumpkin = 12.7 :=
by
  -- Proof can be done manually or via some automation here
  sorry

end total_pumpkin_weight_l45_45373


namespace greatest_4digit_base9_divisible_by_7_l45_45909

theorem greatest_4digit_base9_divisible_by_7 :
  ∃ (n : ℕ), n < 9^4 ∧ (∃ (k : ℕ), n = k * 7) ∧ ∀ (m : ℕ), m < 9^4 ∧ (∃ (j : ℕ), m = j * 7) → n ≥ m ∧ to_digits 9 n = [9, 0, 0, 0] :=
begin
  sorry
end

end greatest_4digit_base9_divisible_by_7_l45_45909


namespace remainder_div_13_l45_45599

theorem remainder_div_13 {N : ℕ} (k : ℕ) (h : N = 39 * k + 18) : N % 13 = 5 := sorry

end remainder_div_13_l45_45599


namespace correct_option_is_A_l45_45106

def optionA (x : ℕ) : String := "4 * x"
def optionB : String := "INPUT"
def optionC : String := "INPUTB=3"
def optionD (x : ℕ) : String := "y=2*x+1"

theorem correct_option_is_A (x : ℕ) :
  optionA x = "4 * x" ∧ optionB = "INPUT" ∧ optionC = "INPUTB=3" ∧ optionD x = "y=2*x+1" → "The correct option is A" :=
sorry

end correct_option_is_A_l45_45106


namespace binomial_12_3_eq_220_l45_45200

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l45_45200


namespace average_capacity_is_3_65_l45_45376

/-- Define the capacities of the jars as a list--/
def jarCapacities : List ℚ := [2, 1/4, 8, 1.5, 0.75, 3, 10]

/-- Calculate the average jar capacity --/
def averageCapacity (capacities : List ℚ) : ℚ :=
  (capacities.sum) / (capacities.length)

/-- The average jar capacity for the given list of jar capacities is 3.65 liters. --/
theorem average_capacity_is_3_65 :
  averageCapacity jarCapacities = 3.65 := 
by
  unfold averageCapacity
  dsimp [jarCapacities]
  norm_num
  sorry

end average_capacity_is_3_65_l45_45376


namespace trajectory_is_a_ray_l45_45706

-- Define points M and N
def M := (-2, 0)
def N := (2, 0)

-- Define the condition
def distance_condition (P : ℝ × ℝ) : Prop :=
  let PM_dist := real.sqrt ((P.1 + 2)^2 + P.2^2)
  let PN_dist := real.sqrt ((P.1 - 2)^2 + P.2^2)
  PM_dist - PN_dist = 4

-- Define the trajectory condition to be proved: P describes a ray starting from N towards infinity on the positive x-axis
def is_ray_from_N (P : ℝ × ℝ) : Prop :=
  P.1 >= 2 ∧ P.2 = 0

-- The theorem to be proved
theorem trajectory_is_a_ray (P : ℝ × ℝ) :
  distance_condition P → is_ray_from_N P :=
sorry

end trajectory_is_a_ray_l45_45706


namespace closest_cube_root_l45_45498

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45498


namespace average_fruits_per_basket_is_correct_l45_45040

noncomputable def average_fruits_per_basket : ℕ :=
  let basket_A := 15
  let basket_B := 30
  let basket_C := 20
  let basket_D := 25
  let basket_E := 35
  let total_fruits := basket_A + basket_B + basket_C + basket_D + basket_E
  let number_of_baskets := 5
  total_fruits / number_of_baskets

theorem average_fruits_per_basket_is_correct : average_fruits_per_basket = 25 := by
  unfold average_fruits_per_basket
  rfl

end average_fruits_per_basket_is_correct_l45_45040


namespace slope_tangent_at_zero_l45_45889

def f (x : ℝ) : ℝ := Real.exp x

theorem slope_tangent_at_zero : HasDerivAt f 1 0 :=
by {
  -- proof not included according to instruction
  sorry
}

end slope_tangent_at_zero_l45_45889


namespace speed_of_second_train_is_30_l45_45903

def speed_second_train (length_train1 length_train2 : ℕ) 
(speed_train1 : ℝ) 
(clear_time : ℝ) : ℝ := 
  let distance : ℝ := (length_train1 + length_train2) / 1000  -- in km
  let time : ℝ := clear_time / 3600   -- in hours
  let relative_speed : ℝ := distance / time  -- in km/h
  relative_speed - speed_train1  -- speed of second train in km/h

theorem speed_of_second_train_is_30 
  (length_train1 length_train2 : ℕ) 
  (speed_train1 : ℝ) 
  (clear_time : ℝ) 
  (h_conditions : length_train1 = 100 ∧ length_train2 = 200 ∧ speed_train1 = 42 ∧ clear_time = 14.998800095992321) : 
  speed_second_train length_train1 length_train2 speed_train1 clear_time = 30 :=
by {
  sorry
}

end speed_of_second_train_is_30_l45_45903


namespace simplify_nested_fraction_root_l45_45847

theorem simplify_nested_fraction_root :
  Real.sqrt (Real.cbrt (Real.sqrt (1 / 65536))) = 1 / 2 := by
sorry

end simplify_nested_fraction_root_l45_45847


namespace sqrt_meaningful_iff_l45_45756

theorem sqrt_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / real.sqrt (5 - x)) ↔ x < 5 :=
by sorry

end sqrt_meaningful_iff_l45_45756


namespace necessary_but_not_sufficient_condition_l45_45406

noncomputable def condition_sufficiency (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + m*x + 1 > 0

theorem necessary_but_not_sufficient_condition (m : ℝ) : m < 2 → (¬ condition_sufficiency m ∨ condition_sufficiency m) :=
by
  sorry

end necessary_but_not_sufficient_condition_l45_45406


namespace binomial_12_3_eq_220_l45_45199

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l45_45199


namespace closest_integer_to_cube_root_of_250_l45_45513

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45513


namespace trailing_zeroes_12_fac_base_11_l45_45745

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_trailing_zeroes_base (n b : ℕ) : ℕ :=
  let fac := factorial n
  let rec count_powers (m : ℕ) (p : ℕ) : ℕ :=
    if (m % p) ≠ 0 then 0 else 1 + count_powers (m / p) p
  count_powers fac b

theorem trailing_zeroes_12_fac_base_11 : num_trailing_zeroes_base 12 11 = 1 :=
  sorry

end trailing_zeroes_12_fac_base_11_l45_45745


namespace closest_integer_to_cube_root_of_250_l45_45479

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45479


namespace nat_pow_eq_sub_two_case_l45_45220

theorem nat_pow_eq_sub_two_case (n : ℕ) : (∃ a k : ℕ, k ≥ 2 ∧ 2^n - 1 = a^k) ↔ (n = 0 ∨ n = 1) :=
by
  sorry

end nat_pow_eq_sub_two_case_l45_45220


namespace double_inequality_l45_45381

variable (a b c : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem double_inequality (h : triangle_sides a b c) : 
  3 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 ∧ (a + b + c) ^ 2 < 4 * (a * b + b * c + c * a) :=
by
  sorry

end double_inequality_l45_45381


namespace binomial_12_3_equals_220_l45_45195

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l45_45195


namespace closest_integer_to_cube_root_of_250_l45_45545

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45545


namespace prod_fraction_calculation_l45_45172

theorem prod_fraction_calculation : 
  ∏ (n : ℕ) in finset.range 15, (n + 1) * (n + 4) / (n + 6)^2 = 1 / 23514624000 :=
by
  sorry

end prod_fraction_calculation_l45_45172


namespace closest_cube_root_l45_45503

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45503


namespace discard_unknown_number_l45_45861

theorem discard_unknown_number (S : ℕ) (X : ℕ) (average_50 : ℕ) (discarded_known : ℕ) (new_average_48 : ℚ) :
  average_50 = 20 →
  discarded_known = 55 →
  new_average_48 = 18.75 →
  S / 50 = 20 →
  (S - discarded_known - X) / 48 = 18.75 →
  X = 45 :=
by
  intros h_avg50 h_discarded_known h_new_avg48 h_sum50 h_sum48
  sorry

end discard_unknown_number_l45_45861


namespace smallest_k_l45_45887

noncomputable def a_seq : ℕ → ℝ
| 0 => 2
| 1 => real.root 13 3
| n + 2 => a_seq (n + 1) * (a_seq n) ^ 2

def prod_a (k : ℕ) : ℝ := list.prod (list.map a_seq (list.range k))

theorem smallest_k (k : ℕ) :
  (∃ k, ∃ (n : ℕ), (n > 0) ∧ (k = n) ∧ (prod_a k ∈ ℤ)) → k = 13 := 
sorry

end smallest_k_l45_45887


namespace max_product_three_distinct_nats_sum_48_l45_45940

open Nat

theorem max_product_three_distinct_nats_sum_48
  (a b c : ℕ) (h_distinct: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_sum: a + b + c = 48) :
  a * b * c ≤ 4080 :=
sorry

end max_product_three_distinct_nats_sum_48_l45_45940


namespace ellipse_equation_line_through_fixed_point_l45_45279

-- Statement for proving the equation of the ellipse
theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a * a = b * b + c * c) (h4 : b = 1) (h5 : c / a = 1 / Real.sqrt 2) :
  ∀ x y : ℝ, (x^2) / 2 + y^2 = 1 :=
begin
  sorry
end

-- Statement for proving the line passes through fixed point
theorem line_through_fixed_point (k1 k2 : ℝ) (h1 : k1 + k2 = 4) :
  ∀ M A B : (ℝ × ℝ), 
    M = (0, 1) → 
    (quadratic_eq : (1 + 2*(k1*k1)) * M.fst^2 + 4 * k1 * (M.snd - 1) * M.fst + ( 2*(k1 * (M.snd - 1) - 1) )^2 - 2 = 0) → 
    A = (M.fst, M.snd) ∨ A = (M.fst, -M.snd) →
    B = (M.fst, M.snd) ∨ B = (M.fst, -M.snd) →
    ∃ k m : ℝ, 
      B.2 = k * B.1 + m ∧
      (-1 / 2, -1) ∈ ℝ × ℝ :=
begin
  sorry
end

end ellipse_equation_line_through_fixed_point_l45_45279


namespace closest_integer_to_cube_root_250_l45_45528

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45528


namespace smallest_coprime_is_prime_l45_45023

theorem smallest_coprime_is_prime (n : ℕ) :
  let N := (Nat.find (λ N, ∀ (k : ℕ), k ∈ (Finset.range (n+1)).toSet → Nat.gcd N k = 1)) in Nat.Prime N :=
by
  sorry

end smallest_coprime_is_prime_l45_45023


namespace point_coordinates_and_distance_property_l45_45782

noncomputable def C1 := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 4}

noncomputable def A := (2, 0 : ℝ)

noncomputable def B := (-1, Real.sqrt 3 : ℝ)

noncomputable def C := (-1, -Real.sqrt 3 : ℝ)

noncomputable def C2 := {p : ℝ × ℝ | (p.1 ^ 2) + (p.2 + Real.sqrt 3) ^ 2 = 1}

noncomputable def P (α : ℝ) : ℝ × ℝ := (Real.cos α, -Real.sqrt 3 + Real.sin α)

noncomputable def dist2 (p q : ℝ × ℝ) : ℝ := (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2

theorem point_coordinates_and_distance_property :
  B ∈ C1 ∧ C ∈ C1 ∧ 
  (∀ α ∈ Icc (0 : ℝ) (2 * Real.pi), P α ∈ C2) ∧ 
  (∀ α ∈ Icc (0 : ℝ) (2 * Real.pi), 
    let PB2 := dist2 (P α) B
    let PC2 := dist2 (P α) C
    (PB2 + PC2) ∈ Icc 8 24) := by 
  sorry

end point_coordinates_and_distance_property_l45_45782


namespace find_interest_rate_l45_45578

-- Definitions for principal amount and time
def principal : ℝ := 10000
def time : ℕ := 2

-- Definitions for Simple Interest and Compound Interest formulas
def simple_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ := P * R * T / 100

def compound_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ := P * (1 + R / 100)^T - P

-- Condition from the problem
def interest_difference_condition (P : ℝ) (R : ℝ) (T : ℕ) : Prop :=
  compound_interest P R T - simple_interest P R T = 49

-- Proof statement for the problem
theorem find_interest_rate (R : ℝ) :
  interest_difference_condition principal R time → R = 70 :=
begin
  sorry
end

end find_interest_rate_l45_45578


namespace number_of_shapes_after_4_folds_sum_of_areas_after_n_folds_l45_45102

-- Conditions based on the problem
def rectangular_paper : ℕ := 20 * 12  -- Initial area
def S1 : ℕ := 240  -- Area after first fold
def S2 : ℕ := 180  -- Area after second fold

-- Number of shapes after 4 folds
theorem number_of_shapes_after_4_folds : (number_of_shapes 4 = 5) :=
by
  -- Mathematically, paper folding logic gives 5 shapes after 4 folds
  sorry

-- Sum of areas after n folds
theorem sum_of_areas_after_n_folds (n : ℕ) : ∑ k in range(n), S k = 240 * (3 - (n + 3) / (2^n)) := 
by
  -- Sum calculation based on folding pattern
  sorry

end number_of_shapes_after_4_folds_sum_of_areas_after_n_folds_l45_45102


namespace angle_parallelogram_l45_45776

variable (A B C D P : Type)
variable [AffineSpace ℝ (A → B)]
variable [Field ℝ]

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : Type) [AffineSpace ℝ (A → B)] : Prop :=
  ∀ (vec : A → B), vec A B = vec D C ∧ vec A D = vec B C

-- Define the angle equality condition
def angle_equality (A D P C : Type) [AffineSpace ℝ (A → B)] : Prop :=
  ∀ (angle : A → D → P → ℝ) (angle2 : P → C → D → ℝ), angle A D P = angle2 P C D

theorem angle_parallelogram 
  (h_parallelogram : is_parallelogram A B C D)
  (h_angle : angle_equality A D P C) : 
  ∀ (angle1 angle2 : P → B → C → ℝ) (angle3 : P → D → C → ℝ), angle1 P B C = angle3 P D C :=
by sorry

end angle_parallelogram_l45_45776


namespace prove_a6_l45_45780

variable (a_n : ℕ → ℝ)
variable (a_1 a_8 a_10 a_3 a_5 : ℝ)

-- Given condition in the problem
def condition : Prop :=
  2 * (a_1 + a_3 + a_5) + 3 * (a_8 + a_10) = 36

-- We need to prove that given the above condition, a_6 equals 3
theorem prove_a6 (h : condition a_n a_1 a_8 a_10 a_3 a_5) : a_n 6 = 3 := 
sorry

end prove_a6_l45_45780


namespace polygon_sides_l45_45433

theorem polygon_sides (n : ℕ) 
  (h₁ : (n - 2) * 180 + 360 = 1800) : 
  n = 10 :=
begin
  sorry
end

end polygon_sides_l45_45433


namespace prop_A_l45_45108

theorem prop_A (x : ℝ) (h : x > 1) : (x + (1 / (x - 1)) >= 3) :=
sorry

end prop_A_l45_45108


namespace proof_of_circle_and_lines_l45_45341

-- Noncomputable as we deal with real numbers and algebraic proofs
noncomputable def equation_of_circle (D E : ℝ) : Prop :=
  (x^2 + y^2 + D * x + E * y + 3 = 0) ∧
  (D > 0 ∧ E < 0) ∧
  (D + E = -2) ∧
  ((D^2 + E^2 - 12) / 4 = 2)

noncomputable def equation_of_lines (a : ℝ) : Prop :=
  (x + y + 1 = 0 ∨ x + y - 3 = 0) ∧
  ((|-1 + 2 - a| / sqrt 2 = sqrt 2) ∧ a ≠ 0)

theorem proof_of_circle_and_lines : 
  ∃ (D E : ℝ), equation_of_circle D E ∧ ∃ (a : ℝ), equation_of_lines a :=
by
  sorry

end proof_of_circle_and_lines_l45_45341


namespace closest_integer_to_cube_root_of_250_l45_45543

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45543


namespace closest_cube_root_l45_45499

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45499


namespace fraction_sequence_product_l45_45177

theorem fraction_sequence_product :
  (∏ i in (finset.range 50).map (nat.succ ∘ (* 1) : ℕ → ℕ) \ (\lam i, 
    ((i / 4) \mod 2 = 1)) (finset.range 5 ): ℝ)) =
  2 / 632170 :=
sorry

end fraction_sequence_product_l45_45177


namespace first_term_of_infinite_geo_series_l45_45986

theorem first_term_of_infinite_geo_series (S r : ℝ) (hS : S = 80) (hr : r = 1/4) :
  let a := (S * (1 - r)) in a = 60 :=
by
  sorry

end first_term_of_infinite_geo_series_l45_45986


namespace relationship_among_abc_l45_45289

noncomputable def a (a : ℝ) : Prop := (1 / 3) ^ a = 2
noncomputable def b (b : ℝ) : Prop := log 3 b = 1 / 2
noncomputable def c (c : ℝ) : Prop := c ^ (-3) = 2

theorem relationship_among_abc (a b c : ℝ) (ha : a (a))
    (hb : b (b)) (hc : c (c)) : a < c ∧ c < b := by
  sorry

end relationship_among_abc_l45_45289


namespace tetrahedron_properties_l45_45934

-- Define the points A, B, C, and D
def A := (2, -1, 2) : ℝ × ℝ × ℝ
def B := (5, 5, 5) : ℝ × ℝ × ℝ
def C := (3, 2, 0) : ℝ × ℝ × ℝ
def D := (4, 1, 4) : ℝ × ℝ × ℝ

-- Define the vectors AB, AC, and AD
def vector_AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3) : ℝ × ℝ × ℝ
def vector_AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3) : ℝ × ℝ × ℝ
def vector_AD := (D.1 - A.1, D.2 - A.2, D.3 - A.3) : ℝ × ℝ × ℝ

-- Define the cross product of AB and AC
def cross_AB_AC := (vector_AB.2 * vector_AC.3 - vector_AB.3 * vector_AC.2,
                    vector_AB.3 * vector_AC.1 - vector_AB.1 * vector_AC.3,
                    vector_AB.1 * vector_AC.2 - vector_AB.2 * vector_AC.1)

-- Define the dot product of the cross product and AD
def dot_cross_AD := cross_AB_AC.1 * vector_AD.1 + cross_AB_AC.2 * vector_AD.2 + cross_AB_AC.3 * vector_AD.3

-- Calculate the volume of the tetrahedron
def volume := (1 / 6) * |dot_cross_AD|

-- Define the magnitude of the cross product
def magnitude_cross := Real.sqrt (cross_AB_AC.1^2 + cross_AB_AC.2^2 + cross_AB_AC.3^2)

-- Define the area of triangle ABC
def area_ABC := (1 / 2) * magnitude_cross

-- Calculate the height from vertex D to base ABC
def height_D := (3 * volume) / magnitude_cross

-- Define the magnitude of vector AD
def magnitude_AD := Real.sqrt (vector_AD.1^2 + vector_AD.2^2 + vector_AD.3^2)

-- Define the cosine of the angle psi
def cos_psi := |dot_cross_AD| / (magnitude_AD * magnitude_cross)

-- Define the angle between AD and ABC
def angle_AD_ABC := Real.arcsin cos_psi

-- The statement proving the mentioned quantities
theorem tetrahedron_properties :
  volume = 3 ∧
  height_D = 6 / Real.sqrt 59 ∧
  angle_AD_ABC = Real.arcsin (3 / Real.sqrt 177) :=
by {
  -- the proof is omitted
  sorry
}

end tetrahedron_properties_l45_45934


namespace original_selling_price_l45_45428

theorem original_selling_price (P : ℝ) (h : 0.7 * P = 560) : P = 800 :=
by
  sorry

end original_selling_price_l45_45428


namespace closest_integer_to_cube_root_of_250_l45_45542

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45542


namespace bamboo_height_on_may_8th_l45_45595

theorem bamboo_height_on_may_8th :
  let growth_rate := 105 -- in cm/day
  let initial_height := 2 -- in meters
  let days := 7 -- days from May 1st to May 8th
  (initial_height + (growth_rate * days) / 100) = 9.35 :=
by
  let growth_rate := 105
  let initial_height := 2
  let days := 7
  have growth_in_meters : (growth_rate * days) / 100 = 7.35 := sorry
  calc
    initial_height + (growth_rate * days) / 100 = initial_height + 7.35 : by rw [growth_in_meters]
    ... = 9.35 : by norm_num
  sorry

end bamboo_height_on_may_8th_l45_45595


namespace closest_integer_to_cube_root_of_250_l45_45544

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45544


namespace number_of_three_digit_odd_numbers_l45_45633

noncomputable def digits : Finset ℕ := {1, 2, 3, 4, 5}
noncomputable def odd_digits : Finset ℕ := {1, 3, 5}

theorem number_of_three_digit_odd_numbers : 
  let units_choices := odd_digits.card,
      tens_choices := (digits \ odd_digits).card,
      hundreds_choices := (digits \ ({1, 2, 3, 4, 5} \ odd_digits)).card
  in units_choices * tens_choices * hundreds_choices = 36 :=
by
  have units_choices_eq : units_choices = 3 := rfl
  have tens_choices_eq : tens_choices = 4 := rfl
  have hundreds_choices_eq : hundreds_choices = 3 := rfl
  rw [units_choices_eq, tens_choices_eq, hundreds_choices_eq]
  exact rfl

end number_of_three_digit_odd_numbers_l45_45633


namespace min_triangle_area_l45_45833

theorem min_triangle_area (a b : ℝ) (α : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < α) : 
  ∃ S : ℝ, S = a * b * Real.cot (α / 2) :=
by
  sorry

end min_triangle_area_l45_45833


namespace binom_12_3_eq_220_l45_45189

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l45_45189


namespace probability_B_wins_l45_45945

theorem probability_B_wins :
  let turns := List (ℕ × ℕ) in
  let sum_last_two (turns: List ℕ) := turns.take_right 2 |> List.sum in
  let sum_last_three (turns: List ℕ) := turns.take_right 3 |> List.sum in
  let game_continues (turns: List ℕ) :=
    (sum_last_two turns % 3 ≠ 0 ∧ sum_last_three turns % 2 ≠ 0) ∨
    (sum_last_three turns % 2 = 0 ∧ sum_last_two turns % 3 = 0) in
  let a_wins := λ (turns: List ℕ), (sum_last_three turns % 2 = 0) ∧ (sum_last_two turns % 3 ≠ 0) in
  let b_wins := λ (turns: List ℕ), (sum_last_two turns % 3 = 0) ∧ (sum_last_three turns % 2 ≠ 0) in
  let q_2 := 1 / 3 in
  let P := 4 / 9 in
  let Q := 5 / 9 in
  (∃ turns, (a_wins turns ∨ b_wins turns) ∧ q_2 = 1 / 3 ∧ a_wins turns ∨ b_wins turns) →
  (Q = 5 / 9) :=
by sorry

end probability_B_wins_l45_45945


namespace tangent_line_l45_45950

variable (a b x₀ y₀ x y : ℝ)
variable (h_ab : a > b)
variable (h_b0 : b > 0)

def ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem tangent_line (h_el : ellipse a b x₀ y₀) : 
  (x₀ * x / a^2) + (y₀ * y / b^2) = 1 :=
sorry

end tangent_line_l45_45950


namespace hyperbola_asymptote_l45_45409

theorem hyperbola_asymptote :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (y = (1/2) * x) ∨ (y = -(1/2) * x) :=
by
  intros x y h
  sorry

end hyperbola_asymptote_l45_45409


namespace smallest_b_for_factorization_l45_45676

theorem smallest_b_for_factorization : ∃ (p q : ℕ), p * q = 2007 ∧ p + q = 232 :=
by
  sorry

end smallest_b_for_factorization_l45_45676


namespace smaller_cube_edge_length_l45_45133

theorem smaller_cube_edge_length (x : ℝ) 
    (original_edge_length : ℝ := 7)
    (increase_percentage : ℝ := 600) 
    (original_surface_area_formula : ℝ := 6 * original_edge_length^2)
    (new_surface_area_formula : ℝ := (1 + increase_percentage / 100) * original_surface_area_formula) :
  ∃ x : ℝ, 6 * x^2 * (original_edge_length ^ 3 / x ^ 3) = new_surface_area_formula → x = 1 := by
  sorry

end smaller_cube_edge_length_l45_45133


namespace binomial_12_3_equals_220_l45_45194

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l45_45194


namespace find_m_if_purely_imaginary_l45_45863

theorem find_m_if_purely_imaginary : ∀ m : ℝ, (m^2 - 5*m + 6 = 0) → (m = 2) :=
by 
  intro m
  intro h
  sorry

end find_m_if_purely_imaginary_l45_45863


namespace twelve_point_five_minutes_in_seconds_l45_45300

-- Definitions
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- Theorem: Prove that 12.5 minutes is 750 seconds
theorem twelve_point_five_minutes_in_seconds : minutes_to_seconds 12.5 = 750 :=
by 
  sorry

end twelve_point_five_minutes_in_seconds_l45_45300


namespace carrots_problem_l45_45584

def total_carrots (faye_picked : Nat) (mother_picked : Nat) : Nat :=
  faye_picked + mother_picked

def bad_carrots (total_carrots : Nat) (good_carrots : Nat) : Nat :=
  total_carrots - good_carrots

theorem carrots_problem (faye_picked : Nat) (mother_picked : Nat) (good_carrots : Nat) (bad_carrots : Nat) 
  (h1 : faye_picked = 23) 
  (h2 : mother_picked = 5)
  (h3 : good_carrots = 12) :
  bad_carrots = 16 := sorry

end carrots_problem_l45_45584


namespace new_average_doubled_l45_45401

theorem new_average_doubled (n : ℕ) (avg : ℝ) (h1 : n = 12) (h2 : avg = 50) :
  2 * avg = 100 := by
sorry

end new_average_doubled_l45_45401


namespace shell_placements_correct_l45_45797

def num_shell_placements : ℕ :=
  (factorial 12) / 6

theorem shell_placements_correct :
  num_shell_placements = 79833600 :=
  sorry

end shell_placements_correct_l45_45797


namespace bee_total_correct_l45_45076

def initial_bees : Nat := 16
def incoming_bees : Nat := 10
def total_bees : Nat := initial_bees + incoming_bees

theorem bee_total_correct : total_bees = 26 := by
  sorry

end bee_total_correct_l45_45076


namespace smallest_fraction_x_eq_9_l45_45680

theorem smallest_fraction_x_eq_9 : 
  let x := 9 in 
  let A := 8 / x in 
  let B := 8 / (x + 2) in 
  let C := 8 / (x - 2) in 
  let D := x / 8 in 
  let E := (x^2 + 1) / 8 in 
  B < A ∧ B < C ∧ B < D ∧ B < E := by
  sorry

end smallest_fraction_x_eq_9_l45_45680


namespace tina_earnings_l45_45089

theorem tina_earnings (days : ℕ) (postcards_per_day : ℕ) (earnings_per_postcard : ℕ) (H_days : days = 6) (H_postcards_per_day : postcards_per_day = 30) (H_earnings_per_postcard : earnings_per_postcard = 5) :
  (days * postcards_per_day * earnings_per_postcard) = 900 :=
by
  rw [H_days, H_postcards_per_day, H_earnings_per_postcard]
  norm_num
  done

end tina_earnings_l45_45089


namespace flag_design_combinations_l45_45886

-- Definitions
def colors : Nat := 3  -- Number of colors: purple, gold, and silver
def stripes : Nat := 3  -- Number of horizontal stripes in the flag

-- The Lean statement
theorem flag_design_combinations :
  (colors ^ stripes) = 27 :=
by
  sorry

end flag_design_combinations_l45_45886


namespace total_amount_received_l45_45838

def sandwich_price : ℝ := 5
def side_dish_price : ℝ := 3
def drink_price : ℝ := 1.50
def delivery_fee : ℝ := 20
def sandwiches_ordered : ℕ := 18
def side_dishes_ordered : ℕ := 10
def drinks_ordered : ℕ := 15
def tip_rate : ℝ := 0.10
def discount_rate : ℝ := 0.15

noncomputable def total_received : ℝ :=
  let food_cost_before_discount := (sandwiches_ordered * sandwich_price) + (side_dishes_ordered * side_dish_price) + (drinks_ordered * drink_price),
      discount := discount_rate * ((sandwiches_ordered * sandwich_price) + (side_dishes_ordered * side_dish_price)),
      food_cost_after_discount := food_cost_before_discount - discount
  in (food_cost_after_discount + delivery_fee) * (1 + tip_rate)

theorem total_amount_received : total_received = 158.95 :=
  by
  sorry

end total_amount_received_l45_45838


namespace lucky_numbers_count_l45_45456

def isLuckyNumber (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3 = 6) && (100 ≤ n) && (n < 1000)

def countLuckyNumbers : ℕ :=
  (List.range' 100 900).filter isLuckyNumber |>.length

theorem lucky_numbers_count : countLuckyNumbers = 21 := 
  sorry

end lucky_numbers_count_l45_45456


namespace factorization_problem1_factorization_problem2_l45_45232

variables {a b x y : ℝ}

theorem factorization_problem1 (a b x y : ℝ) : a * (x + y) - 2 * b * (x + y) = (x + y) * (a - 2 * b) :=
by sorry

theorem factorization_problem2 (a b : ℝ) : a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 :=
by sorry

end factorization_problem1_factorization_problem2_l45_45232


namespace cube_edge_length_l45_45960

theorem cube_edge_length (a : ℝ) (h : 0 ≤ a ∧ a < 1) : 
  ∃ x : ℝ, x = (sqrt (6 - 2 * a^2) - 2 * a) / 3 :=
by
  use (sqrt (6 - 2 * a^2) - 2 * a) / 3
  sorry

end cube_edge_length_l45_45960


namespace prod_fraction_calculation_l45_45173

theorem prod_fraction_calculation : 
  ∏ (n : ℕ) in finset.range 15, (n + 1) * (n + 4) / (n + 6)^2 = 1 / 23514624000 :=
by
  sorry

end prod_fraction_calculation_l45_45173


namespace parabola_equation_l45_45050

-- Defining the point F and the line
def F : ℝ × ℝ := (0, 4)

def line_eq (y : ℝ) : Prop := y = -5

-- Defining the condition that point M is closer to F(0, 4) than to the line y = -5 by less than 1
def condition (M : ℝ × ℝ) : Prop :=
  let dist_to_F := (M.1 - F.1)^2 + (M.2 - F.2)^2
  let dist_to_line := abs (M.2 - (-5))
  abs (dist_to_F - dist_to_line) < 1

-- The equation we need to prove under the given condition
theorem parabola_equation (M : ℝ × ℝ) (h : condition M) : M.1^2 = 16 * M.2 := 
sorry

end parabola_equation_l45_45050


namespace min_value_quadratic_form_l45_45241

theorem min_value_quadratic_form : ∀ x y : ℝ, ∃ m ∈ set.Iio 1, (m = x^2 - x * y + y^2) :=
by
  intros x y
  use 0
  sorry

end min_value_quadratic_form_l45_45241


namespace value_of_c_area_of_ABC_l45_45335

noncomputable theory

-- Definitions for conditions
def b : ℝ := Real.sqrt 6
def A : ℝ := 2 * Real.pi / 3
def B : ℝ := Real.pi / 4
def a1 : ℝ := 3

-- Verifying the value of 'c' for valid conditions
theorem value_of_c : c1 = (3 * Real.sqrt 2 - Real.sqrt 6) / 2 := sorry

-- Area calculation
theorem area_of_ABC : S_ABC = (9 - 3 * Real.sqrt 3) / 4 := sorry

end value_of_c_area_of_ABC_l45_45335


namespace value_of_b_l45_45145

theorem value_of_b :
    ∃ b : ℝ, b < 0 ∧ (∀ x : ℝ, (x^2 + b*x + 1/4) = (x + -1/(2 * Real.sqrt 3))^2 + 1/6) :=
by
  use -1/(Real.sqrt 3)
  split
  · norm_num -- shows that -1/(Real.sqrt 3) < 0
  · sorry

end value_of_b_l45_45145


namespace exponential_grows_faster_l45_45979

noncomputable def f : ℝ → ℝ := λ x, 2 * x
noncomputable def g : ℝ → ℝ := λ x, 3 * x
noncomputable def h : ℝ → ℝ := λ x, 4 * x
noncomputable def k : ℝ → ℝ := λ x, Real.exp x

theorem exponential_grows_faster :
  ∀ x > 0, k x > f x ∧ k x > g x ∧ k x > h x :=
by
  sorry

end exponential_grows_faster_l45_45979


namespace no_unbounded_phine_sequence_l45_45149

theorem no_unbounded_phine_sequence :
  ¬ ∃ (a : ℕ → ℝ), (∀ n ≥ 0, a n > 0) ∧
  (∀ n ≥ 2, a (n+2) = (a (n+1) + a (n-1)) / a n) ∧
  ∀ r, ∃ n, a n > r :=
begin
  sorry
end

end no_unbounded_phine_sequence_l45_45149


namespace xiao_ya_chinese_score_l45_45570

noncomputable def min_chinese_score (average : ℝ) (math_chinese_diff : ℝ) (max_score : ℝ) : ℝ :=
let total_score := average * 3 in
let max_english := max_score in
let remaining := total_score - max_english in
let chinese_score := (remaining - math_chinese_diff) / 2 in
chinese_score

theorem xiao_ya_chinese_score :
  min_chinese_score 92 4 100 = 86 :=
by {
  unfold min_chinese_score,
  norm_num
}

end xiao_ya_chinese_score_l45_45570


namespace closest_integer_to_cube_root_of_250_l45_45515

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45515


namespace part_I_part_II_l45_45721

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - (2 * a + 1) * x

theorem part_I (a : ℝ) (ha : a = -2) : 
  (∃ x : ℝ, f a x = 1) ∧ ∀ x : ℝ, f a x ≤ 1 :=
by sorry

theorem part_II (a : ℝ) (ha : a < 1/2) :
  (∃ x : ℝ, 0 < x ∧ x < exp 1 ∧ f a x < 0) → a < (exp 1 - 1) / (exp 1 * (exp 1 - 2)) :=
by sorry

end part_I_part_II_l45_45721


namespace abs_y_lt_inequality_sum_l45_45126

-- Problem (1)
theorem abs_y_lt {
  x y : ℝ
} (h1 : |x - y| < 1) (h2 : |2 * x + y| < 1) :
  |y| < 1 := by
  sorry

-- Problem (2)
theorem inequality_sum {
  a b c d : ℝ
} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b) + 1 / (b - c) + 1 / (c - d)) ≥ 9 / (a - d) := by
  sorry

end abs_y_lt_inequality_sum_l45_45126


namespace melanie_total_plums_l45_45827

-- Define the initial conditions
def melaniePlums : Float := 7.0
def samGavePlums : Float := 3.0

-- State the theorem to prove
theorem melanie_total_plums : melaniePlums + samGavePlums = 10.0 := 
by
  sorry

end melanie_total_plums_l45_45827


namespace closest_integer_to_cube_root_of_250_l45_45523

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45523


namespace simplify_sqrt_cube_sqrt_l45_45849

theorem simplify_sqrt_cube_sqrt (h : 65536 = 2 ^ 16) : 
  Real.sqrt (Real.cbrt (Real.sqrt (1 / 65536))) = 1 / 2 := by
  sorry

end simplify_sqrt_cube_sqrt_l45_45849


namespace necessary_but_not_sufficient_condition_l45_45368

def f (x : ℝ) : ℝ := log x / log 2

theorem necessary_but_not_sufficient_condition (a b : ℝ) (h : a > b) :
  a > b ↔ f a > f b := sorry

end necessary_but_not_sufficient_condition_l45_45368


namespace rate_per_sq_meter_l45_45879

theorem rate_per_sq_meter
  (length : Float := 9)
  (width : Float := 4.75)
  (total_cost : Float := 38475)
  : (total_cost / (length * width)) = 900 := 
by
  sorry

end rate_per_sq_meter_l45_45879


namespace closest_cube_root_of_250_l45_45469

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45469


namespace cubic_with_root_p_sq_l45_45416

theorem cubic_with_root_p_sq (p : ℝ) (hp : p^3 + p - 3 = 0) : (p^2 : ℝ) ^ 3 + 2 * (p^2) ^ 2 + p^2 - 9 = 0 :=
sorry

end cubic_with_root_p_sq_l45_45416


namespace convert_28nm_to_scientific_notation_in_meters_l45_45975

-- Assuming 1nm = 10^-9 m
def nanometer_to_meter (n : ℝ) : ℝ := n * 10^(-9)

-- Proof problem statement
theorem convert_28nm_to_scientific_notation_in_meters :
  nanometer_to_meter 28 = 2.8 * 10^(-8) := 
sorry

end convert_28nm_to_scientific_notation_in_meters_l45_45975


namespace group1_more_polygons_l45_45999

variables {A : Type} (n : ℕ) (A_points : fin n → A) [fintype A]
def is_polygon (vertices : set A) : Prop := 
  vertices ⊆ {A_points i | i < n} ∧ 3 ≤ vertices.card

def group1 (s : finset A) : Prop :=
  is_polygon n s ∧ ↑A_points 0 ∈ s

def group2 (s : finset A) : Prop :=
  is_polygon n s ∧ ↑A_points 0 ∉ s

theorem group1_more_polygons 
  (h_group1 : ∀ s, group1 s → is_polygon n s)
  (h_group2 : ∀ s, group2 s → is_polygon n s) :
  ∃ f : finset A → finset A, ∀ s, group2 s → group1 (f s) ∧ ∀ s, ¬(group1 s → ∃ s', group2 s') :=
sorry

end group1_more_polygons_l45_45999


namespace exists_k_for_binary_operation_l45_45247

noncomputable def binary_operation (a b : ℤ) : ℤ := sorry

theorem exists_k_for_binary_operation :
  (∀ (a b c : ℤ), binary_operation a (b + c) = 
      binary_operation b a + binary_operation c a) →
  ∃ (k : ℤ), ∀ (a b : ℤ), binary_operation a b = k * a * b :=
by
  sorry

end exists_k_for_binary_operation_l45_45247


namespace closest_integer_to_cube_root_of_250_l45_45514

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45514


namespace water_depth_when_upright_l45_45147
-- Import the entire Mathlib library

-- Define the conditions and question as a theorem
theorem water_depth_when_upright (height : ℝ) (diameter : ℝ) (horizontal_depth : ℝ) :
  height = 20 → diameter = 6 → horizontal_depth = 4 → water_depth = 5.3 :=
by
  intro h1 h2 h3
  -- The proof would go here, but we insert sorry to skip it
  sorry

end water_depth_when_upright_l45_45147


namespace quadratic_eq_solutions_l45_45890

theorem quadratic_eq_solutions : ∃ x1 x2 : ℝ, (x^2 = x) ∨ (x = 0 ∧ x = 1) := by
  sorry

end quadratic_eq_solutions_l45_45890


namespace count_eight_digit_odd_numbers_l45_45170

theorem count_eight_digit_odd_numbers : 
    (∃ s : Finset (List ℕ), 
        s = {l | l.permutations ∧ l ≠ [0] ∧ l.take 1.all(≠ 0) ∧ (List.last l).isSome ∧ ((Option.get l.last).odd) ∧ l.sort = [0, 0, 1, 1, 2, 3, 5, 8]} 
        ∧ s.card = 3600) := 
begin
  sorry
end

end count_eight_digit_odd_numbers_l45_45170


namespace product_of_roots_l45_45672

theorem product_of_roots :
  let poly := (3 * X^3 + 2 * X^2 - 6 * X + 15) * (4 * X^3 - 20 * X^2 + 24) in
  (constant_coeff poly) / (leading_coeff poly) = 30 :=
by
  let poly := (3 * X^3 + 2 * X^2 - 6 * X + 15) * (4 * X^3 - 20 * X^2 + 24)
  sorry

end product_of_roots_l45_45672


namespace closest_integer_to_cube_root_of_250_l45_45478

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45478


namespace binomial_12_3_eq_220_l45_45201

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l45_45201


namespace polygon_sides_l45_45432

theorem polygon_sides (n : ℕ) 
  (h₁ : (n - 2) * 180 + 360 = 1800) : 
  n = 10 :=
begin
  sorry
end

end polygon_sides_l45_45432


namespace g_correct_l45_45646

-- Define the polynomials involved
def p1 (x : ℝ) : ℝ := 2 * x^5 + 4 * x^3 - 3 * x
def p2 (x : ℝ) : ℝ := 7 * x^3 + 5 * x - 2

-- Define g(x) as the polynomial we need to find
def g (x : ℝ) : ℝ := -2 * x^5 + 3 * x^3 + 8 * x - 2

-- Now, state the condition
def condition (x : ℝ) : Prop := p1 x + g x = p2 x

-- Prove the condition holds with the defined polynomials
theorem g_correct (x : ℝ) : condition x :=
by
  change p1 x + g x = p2 x
  sorry

end g_correct_l45_45646


namespace percentage_error_calculation_l45_45162

variable (A B : ℝ)

-- Given conditions on measurements
def measured_A := 1.06 * A
def measured_B := 1.08 * B

-- True area and measured area
def true_area := A * B
def measured_area := measured_A A * measured_B B

-- Percentage error formula
def percentage_error := ((measured_area A B - true_area A B) / true_area A B) * 100

-- The goal is to prove that the percentage error is approximately 14.48%
theorem percentage_error_calculation (A B : ℝ) : percentage_error A B = 14.48 := by
  -- we will use "sorry" as a placeholder for the proof
  sorry

end percentage_error_calculation_l45_45162


namespace daughter_age_l45_45898

-- Define the conditions and the question as a theorem
theorem daughter_age (D F : ℕ) (h1 : F = 3 * D) (h2 : F + 12 = 2 * (D + 12)) : D = 12 :=
by
  -- We need to provide a proof or placeholder for now
  sorry

end daughter_age_l45_45898


namespace discontinuity_points_l45_45218

noncomputable def y (x : ℝ) : ℝ :=
  if h : ∃ k : ℤ, x = k * (π / 2) then 0 else Real.tan x

theorem discontinuity_points : 
  (∀ x, (∃ k : ℤ, x = (2 * k + 1) * (π / 2)) → ¬ DifferentiableAt ℝ y x) := 
by 
  sorry

end discontinuity_points_l45_45218


namespace extended_ohara_triple_example_l45_45088

theorem extended_ohara_triple_example : 
  (2 * Real.sqrt 49 + Real.sqrt 64 = 22) :=
by
  -- We are stating the conditions and required proof here.
  sorry

end extended_ohara_triple_example_l45_45088


namespace collinear_k_perpendicular_k_l45_45736

def vector := ℝ × ℝ

def a : vector := (1, 3)
def b : vector := (3, -4)

def collinear (u v : vector) : Prop :=
  u.1 * v.2 = u.2 * v.1

def perpendicular (u v : vector) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def k_vector_a_minus_b (k : ℝ) (a b : vector) : vector :=
  (k * a.1 - b.1, k * a.2 - b.2)

def a_plus_b (a b : vector) : vector :=
  (a.1 + b.1, a.2 + b.2)

theorem collinear_k (k : ℝ) : collinear (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = -1 :=
sorry

theorem perpendicular_k (k : ℝ) : perpendicular (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = 16 :=
sorry

end collinear_k_perpendicular_k_l45_45736


namespace union_of_sets_l45_45306

def setA : Set ℝ := { x | -5 ≤ x ∧ x < 1 }
def setB : Set ℝ := { x | x ≤ 2 }

theorem union_of_sets : setA ∪ setB = { x | x ≤ 2 } :=
by sorry

end union_of_sets_l45_45306


namespace inequality_solution_l45_45395

-- Let f(x) = (3x - 2) / (x + 1)
def f (x : ℝ) : ℝ := (3 * x - 2) / (x + 1)

-- Prove that solution of |f(x)| > 3 is x in (-∞, -1) ∪ (-1, -1/6)
theorem inequality_solution (x : ℝ) : (|f x| > 3) ↔ (x < -1 ∨ (-1 < x ∧ x < -1 / 6)) :=
by
  sorry

end inequality_solution_l45_45395


namespace manuscript_typing_total_cost_is_1400_l45_45067

-- Defining the variables and constants based on given conditions
def cost_first_time_per_page := 10
def cost_revision_per_page := 5
def total_pages := 100
def pages_revised_once := 20
def pages_revised_twice := 30
def pages_no_revision := total_pages - pages_revised_once - pages_revised_twice

-- Calculations based on the given conditions
def cost_first_time :=
  total_pages * cost_first_time_per_page

def cost_revised_once :=
  pages_revised_once * cost_revision_per_page

def cost_revised_twice :=
  pages_revised_twice * cost_revision_per_page * 2

def total_cost :=
  cost_first_time + cost_revised_once + cost_revised_twice

-- Prove that the total cost equals the calculated value
theorem manuscript_typing_total_cost_is_1400 :
  total_cost = 1400 := by
  sorry

end manuscript_typing_total_cost_is_1400_l45_45067


namespace decreasing_cubic_function_l45_45408

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^3 - x

-- Define the condition that f is decreasing on (-∞, ∞)
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

-- The main theorem that needs to be proven
theorem decreasing_cubic_function (m : ℝ) : is_decreasing (f m) → m < 0 := 
by
  sorry

end decreasing_cubic_function_l45_45408


namespace prop_A_l45_45107

theorem prop_A (x : ℝ) (h : x > 1) : (x + (1 / (x - 1)) >= 3) :=
sorry

end prop_A_l45_45107


namespace closest_cube_root_l45_45504

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45504


namespace closest_integer_to_cbrt_250_l45_45562

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45562


namespace not_possible_five_ints_sum_three_prime_l45_45791

/-- It is not possible to choose five different positive integers such that the sum of any three of them is a prime number. -/
theorem not_possible_five_ints_sum_three_prime 
  (a b c d e : ℕ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : 0 < c) 
  (h4 : 0 < d)
  (h5 : 0 < e)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_prime_sum : ∀ x y z, x ≠ y ∧ x ≠ z ∧ y ≠ z → is_prime (x + y + z)) :
  false :=
sorry

end not_possible_five_ints_sum_three_prime_l45_45791


namespace find_nonnegative_integer_solutions_l45_45658

theorem find_nonnegative_integer_solutions :
  ∀ (a b c : ℕ), (ℕ.sqrt a + ℕ.sqrt b + ℕ.sqrt c = ℕ.sqrt 2014) →
  (a = 0 ∧ b = 0 ∧ c = 2014) ∨ 
  (a = 0 ∧ b = 2014 ∧ c = 0) ∨ 
  (a = 2014 ∧ b = 0 ∧ c = 0) := by
  sorry

end find_nonnegative_integer_solutions_l45_45658


namespace sufficient_not_necessary_condition_l45_45421

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≤ 0)) → (a ≥ 5) :=
sorry

end sufficient_not_necessary_condition_l45_45421


namespace increasing_interval_f_l45_45873

-- Definitions of the functions and conditions
def f (x : ℝ) : ℝ := sin (2 * x + π / 6)
def g (x : ℝ) : ℝ := 2 * cos (x - π / 6) ^ 2 + 1

-- Statement of the theorem
theorem increasing_interval_f :
  (ω > 0) →
  (0 < φ) → (φ < π) →
  (∀ x, sin (ω * x + φ) = f x) →
  (∀ x, g x = 2 * cos (x - π / 6) ^ 2 + 1) →
  (∃ a b : ℝ, -π / 3 = a ∧ π / 6 = b ∧ ∀ x, a ≤ x ∧ x ≤ b → ∃ t, f t < f (t + x)) :=
by
  sorry

end increasing_interval_f_l45_45873


namespace jungkook_mother_age_four_times_jungkook_age_l45_45897

-- Definitions of conditions
def jungkoo_age : ℕ := 16
def mother_age : ℕ := 46

-- Theorem statement for the problem
theorem jungkook_mother_age_four_times_jungkook_age :
  ∃ (x : ℕ), (mother_age - x = 4 * (jungkoo_age - x)) ∧ x = 6 :=
by
  sorry

end jungkook_mother_age_four_times_jungkook_age_l45_45897


namespace closest_integer_to_cube_root_of_250_l45_45547

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45547


namespace speed_conversion_l45_45230

-- Define the conversion factor
def conversion_factor := 3.6

-- Define the given speed in meters per second
def speed_mps := 16.668

-- Define the expected speed in kilometers per hour
def expected_speed_kmph := 60.0048

-- The theorem to prove that the given speed in m/s converts to the expected speed in km/h
theorem speed_conversion : speed_mps * conversion_factor = expected_speed_kmph := 
  by
    sorry

end speed_conversion_l45_45230


namespace integral_x_squared_minus_x_integral_abs_x_minus_2_integral_sqrt_1_minus_x_squared_l45_45451

open Real

-- Problem (1)
theorem integral_x_squared_minus_x :
  (∫ x in 0..1, x^2 - x) = -1/6 :=
by sorry

-- Problem (2)
theorem integral_abs_x_minus_2 :
  (∫ x in 1..3, |x-2|) = 2 :=
by sorry

-- Problem (3)
theorem integral_sqrt_1_minus_x_squared :
  (∫ x in 0..1, sqrt (1 - x^2)) = π/4 :=
by sorry

end integral_x_squared_minus_x_integral_abs_x_minus_2_integral_sqrt_1_minus_x_squared_l45_45451


namespace length_AC_length_BD_find_t_l45_45778

-- Definitions
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def A : Point := ⟨2, 3⟩
def B : Point := ⟨-1, -2⟩
def C : Point := ⟨-2, -1⟩

-- Vector operations
def vector_sub (p1 p2 : Point) : Point :=
  ⟨p1.x - p2.x, p1.y - p2.y⟩

def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

def length (v : Point) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

-- Theorems to prove
theorem length_AC : length (vector_sub C A) = 4 * Real.sqrt 2 := 
  sorry

theorem length_BD : ∃ D : Point, (vector_sub D B = vector_sub A C) ∧
                     length (vector_sub D B) = 2 * Real.sqrt 10 := 
  sorry

theorem find_t (t : ℝ) (D : Point) (hD : vector_sub D B = vector_sub A C) : 
  (dot_product (vector_sub B A + t • vector_sub O C) (vector_sub O C) = 0) → t = -11/5 := 
  sorry

end length_AC_length_BD_find_t_l45_45778


namespace initial_best_method_after_adding_grades_best_method_changes_l45_45933

def VasilyInitialGrades : List ℕ := [4, 1, 2, 5, 2]
def VasilyAddedGrades : List ℕ := VasilyInitialGrades ++ [5, 5]

-- Arithmetic Mean (rounded)
def arithmeticMeanRounded (l : List ℕ) : ℕ :=
  (l.sum : ℕ + l.length / 2) / l.length

-- Median
def median (l : List ℕ) : ℕ :=
  let sorted := l.sort
  if sorted.length % 2 = 1 then
    sorted[(sorted.length / 2).toNat]
  else
    ((sorted[(sorted.length / 2).toNat - 1] + sorted[(sorted.length / 2).toNat]) / 2)

theorem initial_best_method : arithmeticMeanRounded VasilyInitialGrades > median VasilyInitialGrades :=
sorry

theorem after_adding_grades_best_method_changes : 
  arithmeticMeanRounded VasilyAddedGrades < median VasilyAddedGrades :=
sorry

end initial_best_method_after_adding_grades_best_method_changes_l45_45933


namespace ratio_area_proof_angle_CBL_proof_l45_45017

noncomputable def ratio_area_ABP_LPMC (S : ℝ) (BM MC AL AC AP AM : ℝ) 
  (h1 : BM / MC = 3 / 7) (h2 : AL / AC = 3 / 10) (h3 : AP / AM = 1 / 2) (h4 : S = (7 / 10) * S)
  (h5 : S ≠ 0) : ℝ :=
  (3 / 20) * S / ((161 / 260) * S)

theorem ratio_area_proof (S : ℝ) (BM MC AL AC AP AM : ℝ) 
  (h1 : BM / MC = 3 / 7) (h2 : AL / AC = 3 / 10) (h3 : AP / AM = 1 / 2) (h4 : S = (7 / 10) * S)
  (h5 : S ≠ 0) : ratio_area_ABP_LPMC S BM MC AL AC AP AM = 39 / 161 :=
by sorry

noncomputable def angle_CBL : ℝ := Real.arccos (Real.sqrt (13 / 15))

theorem angle_CBL_proof (MT TC : ℝ) 
  (h1 : MT / TC = 1 / 6) : angle_CBL = Real.arccos (Real.sqrt (13 / 15)) :=
by sorry

end ratio_area_proof_angle_CBL_proof_l45_45017


namespace function_range_l45_45243

noncomputable def function_f (x y z : ℝ) : ℝ :=
  (x * Real.sqrt y + y * Real.sqrt z + z * Real.sqrt x) /
  Real.sqrt ((x + y) * (y + z) * (z + x))

theorem function_range :
  {f : ℝ | ∃ x y z : ℝ, (0 < x ∧ 0 < y ∧ 0 < z) ∧ f = function_f x y z} = Icc 0 (3 / (2 * Real.sqrt 2)) :=
sorry

end function_range_l45_45243


namespace a_range_of_complex_modulus_l45_45245

theorem a_range_of_complex_modulus (a : ℝ) (h : ∀ θ : ℝ, (a + Real.cos θ) ^ 2 + (2 * a - Real.sin θ) ^ 2 ≤ 4) :
  -real.sqrt 5 / 5 ≤ a ∧ a ≤ real.sqrt 5 / 5 :=
sorry

end a_range_of_complex_modulus_l45_45245


namespace imag_part_z_l45_45718

theorem imag_part_z {z : ℂ} (h : i * (z - 3) = -1 + 3 * i) : z.im = 1 :=
sorry

end imag_part_z_l45_45718


namespace min_value_of_sum_l45_45116

theorem min_value_of_sum (x y z : ℝ) (h : x^2 + 2 * y^2 + 5 * z^2 = 22) :
  xy - yz - zx ≥ (xy - yz - zx) :=
sorry

end min_value_of_sum_l45_45116


namespace jane_played_rounds_l45_45314

-- Define the conditions
def points_per_round := 10
def points_ended_with := 60
def points_lost := 20

-- Define the proof problem
theorem jane_played_rounds : (points_ended_with + points_lost) / points_per_round = 8 :=
by
  sorry

end jane_played_rounds_l45_45314


namespace simplify_expression_l45_45844

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (5 * x) * (x^4) = 27 * x^5 :=
by
  sorry

end simplify_expression_l45_45844


namespace roses_in_vase_now_l45_45442

-- Definitions of initial conditions and variables
def initial_roses : ℕ := 12
def initial_orchids : ℕ := 2
def orchids_cut : ℕ := 19
def orchids_now : ℕ := 21

-- The proof problem to show that the number of roses now is still the same as initially.
theorem roses_in_vase_now : initial_roses = 12 :=
by
  -- The proof itself is left as an exercise (add proof here)
  sorry

end roses_in_vase_now_l45_45442


namespace sum_of_distances_to_y_axis_l45_45832

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def focus := (1, 0)
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)
noncomputable def distance_to_y_axis (x : ℝ) : ℝ := Real.abs x

theorem sum_of_distances_to_y_axis
  (x1 y1 x2 y2 : ℝ)
  (hA : parabola x1 y1)
  (hB : parabola x2 y2)
  (h_sum_distances_to_focus : (distance x1 y1 1 0) + (distance x2 y2 1 0) = 7) :
  (distance_to_y_axis x1) + (distance_to_y_axis x2) = 5 :=
by sorry

end sum_of_distances_to_y_axis_l45_45832


namespace degree_g_is_3_l45_45397

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := f (g x) - g x

axiom deg_h : polynomial.degree (polynomial.C (h 0) * polynomial.X ^ 8) = 8
axiom deg_f : polynomial.degree (polynomial.C (f 0) * polynomial.X ^ 3) = 3

theorem degree_g_is_3 : polynomial.degree (g 0 * polynomial.X ^ 3) = 3 := 
sorry

end degree_g_is_3_l45_45397


namespace three_digit_numbers_19_sum_digits_l45_45396

theorem three_digit_numbers_19_sum_digits :
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ n = 19 * (a + b + c)} in
  three_digit_numbers = {114, 133, 152, 171, 190, 209, 228, 247, 266, 285, 399} :=
by
  sorry

end three_digit_numbers_19_sum_digits_l45_45396


namespace solve_equation1_solve_equation2_pos_solve_equation2_neg_l45_45679

theorem solve_equation1 (x : ℝ) (h : 2 * x^3 = 16) : x = 2 :=
sorry

theorem solve_equation2_pos (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 :=
sorry

theorem solve_equation2_neg (x : ℝ) (h : (x - 1)^2 = 4) : x = -1 :=
sorry

end solve_equation1_solve_equation2_pos_solve_equation2_neg_l45_45679


namespace mass_scientific_notation_l45_45880

def mass := 37e-6

theorem mass_scientific_notation : mass = 3.7 * 10^(-5) :=
by
  sorry

end mass_scientific_notation_l45_45880


namespace solve_quadratic_equation_solve_linear_equation_l45_45028

-- Equation (1)
theorem solve_quadratic_equation :
  ∀ x : ℝ, x^2 - 8 * x + 1 = 0 → (x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) :=
by
  sorry

-- Equation (2)
theorem solve_linear_equation :
  ∀ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x → (x = 1 ∨ x = -2/3) :=
by
  sorry

end solve_quadratic_equation_solve_linear_equation_l45_45028


namespace minimum_number_of_at_bats_l45_45322

theorem minimum_number_of_at_bats :
  ∃ y : ℕ, y = 138 ∧ ∃ x : ℕ, 0.3985 < (x : ℝ) / (y : ℝ) ∧ (x : ℝ) / (y : ℝ) < 0.3995 :=
begin
  sorry
end

end minimum_number_of_at_bats_l45_45322


namespace inheritance_amount_l45_45052

theorem inheritance_amount (x : ℝ) :
  (let child1_amount := 100 + (1/10) * (x - 100) in
   let child2_amount := 200 + (1/10) * (x - (100 + (1/10) * (x - 100)) - 200) in
   child1_amount = child2_amount) →
  x = 8100 := by
  sorry

end inheritance_amount_l45_45052


namespace perpendicular_DE_IO_l45_45818

noncomputable section

variables {A B C D E I O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace I] [MetricSpace O]
variables (triangle : A × B × C)
variables (incenter : I) (circumcenter : O)
variables (condition1 : dist B C < dist A B)
variables (condition2 : dist A B < dist A C)
variables (D_inside_AB : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ dist (A, D) = (1 - x) * dist (A, B))
variables (E_inside_AC : ∃ y : ℝ, 0 < y ∧ y < 1 ∧ dist (A, E) = (1 - y) * dist (A, C))
variables (DB_eq_BC : dist D B = dist B C)
variables (CE_eq_BC : dist C E = dist B C)

theorem perpendicular_DE_IO :
  ∃ (D : A) (E : C), ∃ (I : I) (O : O), 
  dist D B = dist B C ∧ dist C E = dist B C ∧ 
  (D_inside_AB) ∧ (E_inside_AC) ∧ 
  (line (D, E) ⊥ line (I, O)) := 
sorry

end perpendicular_DE_IO_l45_45818


namespace closest_integer_to_cubert_of_250_is_6_l45_45553

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45553


namespace base_7_digits_956_l45_45742

theorem base_7_digits_956 : ∃ n : ℕ, ∀ k : ℕ, 956 < 7^k → n = k ∧ 956 ≥ 7^(k-1) := sorry

end base_7_digits_956_l45_45742


namespace pencils_added_by_Nancy_l45_45083

def original_pencils : ℕ := 27
def total_pencils : ℕ := 72

theorem pencils_added_by_Nancy : ∃ x : ℕ, x = total_pencils - original_pencils := by
  sorry

end pencils_added_by_Nancy_l45_45083


namespace clock_tick_intervals_l45_45629

theorem clock_tick_intervals (intervals_6: ℕ) (intervals_12: ℕ) (total_time_12: ℕ) (interval_time: ℕ):
  intervals_6 = 5 →
  intervals_12 = 11 →
  total_time_12 = 88 →
  interval_time = total_time_12 / intervals_12 →
  intervals_6 * interval_time = 40 :=
by
  intros h1 h2 h3 h4
  -- will continue proof here
  sorry

end clock_tick_intervals_l45_45629


namespace height_prediction_l45_45613

theorem height_prediction (x : ℕ) (h₁ : x = 10) :
  let y := 7.19 * x + 73.93 in 
  y ≈ 145.83 :=
by
  -- Assume we have some approximation function similar to y ≈ 145.83 even though y == 145.83 will also work.
  sorry

end height_prediction_l45_45613


namespace cost_of_reebok_shoes_reebok_shoe_cost_correct_l45_45157

theorem cost_of_reebok_shoes
  (goal : ℝ) (adidas_cost : ℝ) (nike_cost : ℝ)
  (nike_sold : ℕ) (adidas_sold : ℕ) (reebok_sold : ℕ)
  (excess : ℝ) (total_sales : ℝ)
  (total_nike_sales : ℝ) (total_adidas_sales : ℝ)
  (reebok_cost : ℝ) : Prop :=
goal = 1000 ∧
adidas_cost = 45 ∧
nike_cost = 60 ∧
nike_sold = 8 ∧
adidas_sold = 6 ∧
reebok_sold = 9 ∧
excess = 65 ∧ 
total_sales = goal + excess ∧
total_nike_sales = nike_sold * nike_cost ∧
total_adidas_sales = adidas_sold * adidas_cost ∧
(total_sales - (total_nike_sales + total_adidas_sales)) = reebok_sold * reebok_cost ∧
reebok_cost = 35

theorem reebok_shoe_cost_correct 
  (goal : ℝ := 1000) 
  (adidas_cost : ℝ := 45) 
  (nike_cost : ℝ := 60)
  (nike_sold : ℕ := 8)
  (adidas_sold : ℕ := 6)
  (reebok_sold : ℕ := 9)
  (excess : ℝ := 65) : cost_of_reebok_shoes goal adidas_cost nike_cost nike_sold adidas_sold reebok_sold excess 35 (570) (270) := 
by 
  sorry

end cost_of_reebok_shoes_reebok_shoe_cost_correct_l45_45157


namespace reduced_expression_none_of_these_l45_45025

theorem reduced_expression_none_of_these (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : b ≠ a^2) (h4 : ab ≠ a^3) :
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 1 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (b^2 + b) / (b - a^2) ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 0 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (a^2 + b) / (a^2 - b) :=
by
  sorry

end reduced_expression_none_of_these_l45_45025


namespace polygon_sides_l45_45435

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 + 360 = 1800) : n = 10 := by
  sorry

end polygon_sides_l45_45435


namespace decreasing_log_function_l45_45309

theorem decreasing_log_function (a : ℝ) (h_decreasing : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → (log a (3 - a * y)) < (log a (3 - a * x))) :
  1 < a ∧ a < 3 :=
sorry

end decreasing_log_function_l45_45309


namespace sum_of_first_13_terms_l45_45700

variable {α : Type*} [LinearOrderedField α] (a d : α) (a_seq : ℕ → α)

def arithmetic_sequence (a d : α) : ℕ → α
| n := a + n * d

def sum_first_n_terms (a d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_13_terms (a d : α) (a_seq : ℕ → α)
  (h_seq : ∀ n, a_seq n = arithmetic_sequence a d n)
  (h_cond : a_seq 5 + a_seq 9 = 2) :
  sum_first_n_terms a d 13 = 13 :=
by
-- Proof goes here.
sorry

end sum_of_first_13_terms_l45_45700


namespace fraction_expression_l45_45231

theorem fraction_expression :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end fraction_expression_l45_45231


namespace parabola_analysis_l45_45269

theorem parabola_analysis (a : ℝ) (h : a ≠ 0) :
  let Parabola := λ x : ℝ, a * x^2 + 4 * a * x + 3,
      Conclusion1 := ∀ x, Parabola x = a * x^2 + 4 * a * x + 3 → (-4 * a / (2 * a)) = -2,
      Conclusion2 := Parabola 0 = 3,
      Conclusion3 := ∀ x1 x2 : ℝ, x1 > x2 > -2 → (a > 0 → Parabola x1 > Parabola x2) ∧ (a < 0 → Parabola x1 < Parabola x2),
      Conclusion4 := ∀ y1 y2 : ℝ, (∃ x1 x2 : ℝ, Parabola x1 = y1 ∧ Parabola x2 = y2) → x1 + x2 = -2,
  (Conclusion1 ∧ Conclusion2 ∧ ¬Conclusion3 ∧ ¬Conclusion4) := by
  sorry

end parabola_analysis_l45_45269


namespace closest_integer_to_cube_root_of_250_l45_45521

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45521


namespace find_B_coordinates_l45_45292

open Real

def vector := { x : ℝ // true }

@[ext] structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

def a : vector := ⟨8, true⟩ :: ⟨9, true⟩ :: ⟨-12, true⟩ :: []

def A : Point := { x := 2, y := -1, z := 7 }
def distance_AB : ℝ := 34

theorem find_B_coordinates :
  ∃ B : Point, (B.x = 18) ∧ (B.y = 17) ∧ (B.z = -17) :=
  sorry

end find_B_coordinates_l45_45292


namespace closest_integer_to_cube_root_of_250_l45_45524

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45524


namespace parallel_lines_find_m_l45_45703

theorem parallel_lines_find_m (m : ℝ) :
  (((3 + m) / 2 = 4 / (5 + m)) ∧ ((3 + m) / 2 ≠ (5 - 3 * m) / 8)) → m = -7 :=
sorry

end parallel_lines_find_m_l45_45703


namespace area_of_border_l45_45615

theorem area_of_border
  (photo_height : ℤ) (photo_width : ℤ) (border_width : ℤ)
  (photo_height_eq : photo_height = 8)
  (photo_width_eq : photo_width = 12)
  (border_width_eq : border_width = 3) :
  let frame_height := photo_height + 2 * border_width in
  let frame_width := photo_width + 2 * border_width in
  let photo_area := photo_height * photo_width in
  let frame_area := frame_height * frame_width in
  frame_area - photo_area = 156 :=
by
  sorry

end area_of_border_l45_45615


namespace problem1_problem2_problem3_l45_45248

-- Define fixed points and stable points
def fixed_points (f : ℝ → ℝ) : set ℝ := {x | f x = x}
def stable_points (f : ℝ → ℝ) : set ℝ := {x | f (f x) = x}

-- Problem 1: Find sets A and B for f(x) = 3x + 4
def f1 (x : ℝ) := 3 * x + 4
theorem problem1 : fixed_points f1 = {-2} ∧ stable_points f1 = {-2} := 
sorry

-- Problem 2: Prove that A ⊆ B
theorem problem2 (f : ℝ → ℝ) : fixed_points f ⊆ stable_points f := 
sorry

-- Problem 3: Prove that if f(x) = ax^2 + bx + c and A = ∅, then B = ∅
def f2 (a b c : ℝ) (x : ℝ) := a * x ^ 2 + b * x + c
theorem problem3 {a b c : ℝ} (h : a ≠ 0) : fixed_points (f2 a b c) = ∅ → stable_points (f2 a b c) = ∅ := 
sorry

end problem1_problem2_problem3_l45_45248


namespace solution_set_f_le_1_l45_45272

variable {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem solution_set_f_le_1 :
  is_even_function f →
  monotone_on_nonneg f →
  f (-2) = 1 →
  {x : ℝ | f x ≤ 1} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} :=
by
  intros h_even h_mono h_f_neg_2
  sorry

end solution_set_f_le_1_l45_45272


namespace calc_sum_eq_neg_half_l45_45171

theorem calc_sum_eq_neg_half :
  (1 / 2^2010) * ∑ n in Finset.range 1006, (-3)^n * Nat.choose 2010 (2 * n) = -1 / 2 :=
by
  sorry

end calc_sum_eq_neg_half_l45_45171


namespace circle_area_pqr_l45_45164

noncomputable def area_of_circumcircle (PQ PR: ℝ) (radius: ℝ) : ℝ :=
  let circum_radius := PQ / Math.sqrt 3
  in π * (circum_radius ^ 2)

theorem circle_area_pqr :
  ∀ (P Q R X Y : Type)
    (PQ PR : ℝ)
    (rPQ : PQR) 
    (rPR : PQR) 
    (tangent_circle : circle)
    (radius3 : tangent_circle.radius = 3)
    (same_rad : PQR.is_eq_side PQ PR)
    (radius_bisector : same_rad.and_eq.radius bisect (PQ PR 4)) 
    (tangent PQ = X & tangent PR = Y), 
    area_of_circumcircle 5 5 3 = (25 / 3) * π := 
begin
  sorry
end

end circle_area_pqr_l45_45164


namespace find_a2_geometric_sequence_l45_45694

theorem find_a2_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) 
  (h_a1 : a 1 = 1 / 4) (h_eq : a 3 * a 5 = 4 * (a 4 - 1)) : a 2 = 1 / 8 :=
by
  sorry

end find_a2_geometric_sequence_l45_45694


namespace min_time_calculation_l45_45835

noncomputable def min_time_to_receive_keys (diameter cyclist_speed_road cyclist_speed_alley pedestrian_speed : ℝ) : ℝ :=
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let distance_pedestrian := pedestrian_speed * 1
  let min_time := (2 * Real.pi * radius - 2 * distance_pedestrian) / (cyclist_speed_road + cyclist_speed_alley)
  min_time

theorem min_time_calculation :
  min_time_to_receive_keys 4 15 20 6 = (2 * Real.pi - 2) / 21 :=
by
  sorry

end min_time_calculation_l45_45835


namespace ellipse_equation_max_area_triangle_l45_45325

-- Problem 1
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (ecc : √(2/3)) (max_dist : ∀ x y, y^2 + 3 * y + 3b^2 = 3b^2 + 3):
  ∃ (b: ℝ), (b = 1) ∧ ∀ x y : ℝ, ∃ (C : ℝ), x^2 + 3*y^2 = 3*b^2 := sorry

-- Problem 2
theorem max_area_triangle (m n : ℝ) (h3 : m^2 / 3 + n^2 = 1) (h4 : m^2 + n^2 > 1) :
  ∃ (S: ℝ), (S = 1/2)
  ∧ ( (m = √6 / 2 ∧ n = √2 / 2) ∨ (m = -√6 / 2 ∧ n = √2 / 2) ∨ (m = √6 / 2 ∧ n = -√2 / 2) ∨ (m = -√6 / 2 ∧ n = -√2 / 2) ) := sorry

end ellipse_equation_max_area_triangle_l45_45325


namespace tetrahedron_volume_max_OB_l45_45892

-- Definition of the conditions
variable {P A B O H C : Type} [Inhabited P] [Inhabited A] [Inhabited B] [Inhabited O] [Inhabited H] [Inhabited C]

def is_isosceles_right_triangle (P O A : Type) : Prop := sorry
def on_circumference (A : Type) (base_circle : Type) : Prop := sorry
def inside_base_circle (B : Type) (base_circle : Type) : Prop := sorry
def center_of_base_circle (O : Type) (base_circle : Type) : Prop := sorry
def perpendicular (X Y : Type) : Prop := sorry
def midpoint (C : Type) (X Y : Type) : Prop := sorry
def distance (X Y : Type) : ℝ := sorry
def PA := distance P A
def OB := distance O B

-- Hypotheses
axiom h1 : is_isosceles_right_triangle P O A
axiom h2 : on_circumference A base_circle
axiom h3 : inside_base_circle B base_circle
axiom h4 : center_of_base_circle O base_circle
axiom h5 : perpendicular A B
axiom h6 : perpendicular O H
axiom h7 : midpoint C P A
axiom h8 : PA = 4

-- Proof statement
theorem tetrahedron_volume_max_OB : OB = (2 * real.sqrt 6) / 3 :=
sorry

end tetrahedron_volume_max_OB_l45_45892


namespace diana_bike_home_time_l45_45225

theorem diana_bike_home_time : 
  ∀ (dist total_dist speed1 speed2 time1 time2 : ℝ), 
  total_dist = 10 ∧ 
  speed1 = 3 ∧ time1 = 2 ∧ speed2 = 1 ∧
  dist = speed1 * time1 ∧
  (total_dist - dist) = speed2 * time2 → 
  time1 + time2 = 6 :=
by
  intros dist total_dist speed1 speed2 time1 time2 h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  cases h8 with h9 h10
  rw [h3, h4, h5, h6] at h10
  rw [h7, h8, h9, h10] at h10
  sorry

end diana_bike_home_time_l45_45225


namespace sin_cos_parallel_f_A_value_l45_45685

variable {x A : ℝ}
def m := (Real.sin x, 3/4)
def n := (Real.cos x, -1)
def f (x : ℝ) := 2 * ((Real.sin x + Real.cos x, -1/4) • (Real.cos x, -1))

theorem sin_cos_parallel (h : m.1 / m.2 = n.1 / n.2) : Real.sin x ^ 2 + Real.sin (2 * x) = -3 / 5 :=
  sorry

theorem f_A_value (hA : Real.sin A + Real.cos A = Real.sqrt 2) : f A = 5 / 2 :=
  sorry

end sin_cos_parallel_f_A_value_l45_45685


namespace factors_and_divisors_l45_45111

theorem factors_and_divisors :
  (∃ n : ℕ, 25 = 5 * n) ∧
  (¬(∃ n : ℕ, 209 = 19 * n ∧ ¬ (∃ m : ℕ, 57 = 19 * m))) ∧
  (¬(¬(∃ n : ℕ, 90 = 30 * n) ∧ ¬(∃ m : ℕ, 75 = 30 * m))) ∧
  (¬(∃ n : ℕ, 51 = 17 * n ∧ ¬ (∃ m : ℕ, 68 = 17 * m))) ∧
  (∃ n : ℕ, 171 = 9 * n) :=
by {
  sorry
}

end factors_and_divisors_l45_45111


namespace inverse_of_exponential_function_l45_45055

theorem inverse_of_exponential_function :
  ∀ (x : ℝ), 0 < x → (∃ y : ℝ, 2^(y + 1) = x) ↔ (∃ z : ℝ, z = -1 + log x) :=
by
  sorry

end inverse_of_exponential_function_l45_45055


namespace root_frac_value_l45_45747

theorem root_frac_value (m n : ℝ) (hm : (Polynomial.X^2 - 2 * Polynomial.X + 1).isRoot m)
  (hn : (Polynomial.X^2 - 2 * Polynomial.X + 1).isRoot n) :
  (m + n) / (m^2 - 2 * m) = -2 :=
by sorry

end root_frac_value_l45_45747


namespace quad_area_l45_45080

def is_int (n : ℤ) : Prop := ∃ m : ℤ, n = m

variable {z : ℂ}

def int_parts (z : ℂ) : Prop := is_int (z.re) ∧ is_int (z.im)

theorem quad_area (z₁ z₂ z₃ z₄ : ℂ) (h₁ : z₁ * (z₁.conj ^ 3) + (z₁.conj) * (z₁ ^ 3) = 98)
  (h₂ : int_parts z₁) (h₃ : int_parts z₂) (h₄ : int_parts z₃) (h₅ : int_parts z₄) :
  (area_of_quadrilateral z₁ z₂ z₃ z₄ = 392) :=
sorry

end quad_area_l45_45080


namespace correct_propositions_l45_45053

-- Proposition 1
def Proposition1 (f : ℝ → ℝ) :=
  f 2 < f 3 → ∃ (x1 x2 : ℝ), x1 < x2 ∧ f x1 < f x2

-- Proposition 2
def A : Set ℝ := {1, 4}
def B : Set ℝ := {1, -1, 2, -2}
def f2 (x : ℝ) : ℝ := real.root x 7

def Proposition2 := ∀ x ∈ A, f2 x ∈ B

-- Proposition 3
def f3 (x : ℝ) : ℝ := (2:ℝ) ^ (-x)
def g3 (x : ℝ) : ℝ := (2:ℝ) ^ (-(x-2)) - 1

def Proposition3 := ∀ x, f3 (x - 2) - 1 = g3 x

-- Proposition 4
def Proposition4 (a : ℝ) := a > 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ abs ((2:ℝ) ^ x1 - 1) = a ∧ abs ((2:ℝ) ^ x2 - 1) = a

-- Main Proposition
theorem correct_propositions : ∀ {f}
  (P1 : Proposition1 f)
  (P2 : Proposition2)
  (P3 : Proposition3)
  (P4 : ∀ a, Proposition4 a),
  ({1, 2} : Set ℤ) = {1, 2} :=
by
  sorry

end correct_propositions_l45_45053


namespace sum_of_zeros_of_transformed_parabola_l45_45413

def parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

def rotated_parabola (x : ℝ) : ℝ := -(x - 3)^2 + 4

def shifted_left_parabola (x : ℝ) : ℝ := -((x - 8)^2) + 4

def shifted_down_parabola (x : ℝ) : ℝ := -((x - 8)^2)

theorem sum_of_zeros_of_transformed_parabola : 
  (let p := -8 in let q := 8 in p + q) = 16 :=
by
  sorry

end sum_of_zeros_of_transformed_parabola_l45_45413


namespace part1_part2_l45_45724

-- Definitions for conditions
def f (x a : ℝ) : ℝ := real.log x + a / x

-- Statement for Part 1
theorem part1 (a : ℝ) (h : ∃ x : ℝ, 0 < x ∧ f x a = 0) : 
  0 < a ∧ a ≤ 1 / real.exp 1 :=
sorry

-- Statement for Part 2
theorem part2 (a b : ℝ) (h1 : a ≥ 2 / real.exp 1) (h2 : b > 1) :
  f (real.log b) a > 1 / b :=
sorry

end part1_part2_l45_45724


namespace divide_base_in_ratio_l45_45143

variables {A B C F K M : Type}
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]

def is_isosceles_triangle (A B C : Type) [LinearOrderedField A] : Prop :=
  A = C

def fold_to_midpoint (A F A B C : Type) [LinearOrderedField A] : Prop :=
  is_midpoint F B C

def divides_ratio (A K C : Type) [LinearOrderedField A] (p q : A) : Prop :=
  divides K A C (p, q)

def divides_base (A M B : Type) [LinearOrderedField A] (r : A) : Prop :=
  divides M A B (r, 2*r)

theorem divide_base_in_ratio
  (A B C F K M : Type)
  [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
  (p q : A)
  (h_iso : is_isosceles_triangle A B C)
  (h_fold : fold_to_midpoint A F A B C)
  (h_div_ac : divides_ratio A K C p q) :
  divides_base A M B (p / (2 * p - q)) :=
sorry

end divide_base_in_ratio_l45_45143


namespace range_of_a_l45_45307

noncomputable def isNotPurelyImaginary (a : ℝ) : Prop :=
  let re := a^2 - a - 2
  re ≠ 0

theorem range_of_a (a : ℝ) (h : isNotPurelyImaginary a) : a ≠ -1 :=
  sorry

end range_of_a_l45_45307


namespace three_digit_number_conditions_l45_45744

-- Define that a number is a three-digit number
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

-- Define the predicate that a number contains digit d
def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  let hundreds := n / 100 % 10
  let tens := n / 10 % 10
  let ones := n % 10
  (hundreds = d) ∨ (tens = d) ∨ (ones = d)

-- Define the original question in Lean
theorem three_digit_number_conditions :
  { n : ℕ // is_three_digit n ∧ contains_digit n 2 ∧ contains_digit n 3 ∧ contains_digit n 5 }.card = 6 :=
sorry

end three_digit_number_conditions_l45_45744


namespace journey_time_l45_45319

def river_speed : ℝ := 2
def upstream_distance : ℝ := 64
def boat_still_water_speed : ℝ := 6

theorem journey_time :
  let upstream_speed := boat_still_water_speed - river_speed,
      downstream_speed := boat_still_water_speed + river_speed,
      time_upstream := upstream_distance / upstream_speed,
      time_downstream := upstream_distance / downstream_speed,
      total_time := time_upstream + time_downstream
  in total_time = 24 := by sorry

end journey_time_l45_45319


namespace first_term_of_infinite_geo_series_l45_45985

theorem first_term_of_infinite_geo_series (S r : ℝ) (hS : S = 80) (hr : r = 1/4) :
  let a := (S * (1 - r)) in a = 60 :=
by
  sorry

end first_term_of_infinite_geo_series_l45_45985


namespace no_real_value_x_l45_45423

theorem no_real_value_x (R H : ℝ) (π : ℝ := Real.pi) :
  R = 10 → H = 5 →
  ¬∃ x : ℝ,  π * (R + x)^2 * H = π * R^2 * (H + x) ∧ x ≠ 0 :=
by
  intros hR hH; sorry

end no_real_value_x_l45_45423


namespace count_five_digit_numbers_ending_in_6_divisible_by_3_l45_45297

theorem count_five_digit_numbers_ending_in_6_divisible_by_3 : 
  (∃ (n : ℕ), n = 3000 ∧
  ∀ (x : ℕ), (x ≥ 10000 ∧ x ≤ 99999) ∧ (x % 10 = 6) ∧ (x % 3 = 0) ↔ 
  (∃ (k : ℕ), x = 10026 + k * 30 ∧ k < 3000)) :=
by
  -- Proof is omitted
  sorry

end count_five_digit_numbers_ending_in_6_divisible_by_3_l45_45297


namespace find_line_equation_l45_45441

noncomputable def equation_of_line (m b : ℝ) : Prop :=
∃ k : ℝ, 
  (b ≠ 0) ∧
  (2 * m + b = 5) ∧
  ((k^2 + (4 - m) * k - b = 0) ∨ (k^2 + (4 - m) * k + 6 - b = 0)) ∧
  abs(((k^2 + 4 * k + 3) - (m * k + b)) = 3)

theorem find_line_equation : 
  equation_of_line (9 / 2) (-4) :=
by sorry

end find_line_equation_l45_45441


namespace min_value_x_plus_one_over_x_minus_one_max_value_sqrt_x_times_10_minus_x_l45_45109

-- Statement for problem A
theorem min_value_x_plus_one_over_x_minus_one (x : ℝ) (h : 1 < x) : 
  ∃ y, y = x + (1 / (x - 1)) ∧ y ≥ 3 := by
  sorry

-- Statement for problem C
theorem max_value_sqrt_x_times_10_minus_x (x : ℝ) (h1 : 0 < x) (h2 : x < 10) :
  ∃ y, y = sqrt (x * (10 - x)) ∧ y ≤ 5 := by
  sorry

end min_value_x_plus_one_over_x_minus_one_max_value_sqrt_x_times_10_minus_x_l45_45109


namespace items_per_friend_l45_45390

theorem items_per_friend (pencils : ℕ) (erasers : ℕ) (friends : ℕ) 
    (pencils_eq : pencils = 35) 
    (erasers_eq : erasers = 5) 
    (friends_eq : friends = 5) : 
    (pencils + erasers) / friends = 8 := 
by
  sorry

end items_per_friend_l45_45390


namespace sum_of_binom_values_l45_45244

def binom : ℕ → ℕ → ℕ
| n, 0       := 1
| 0, k+1     := 0
| n+1, k+1   := binom n k + binom n (k+1)

theorem sum_of_binom_values :
  (∑ k in { k : ℕ | binom 27 5 + binom 27 6 = binom 28 k }, k) = 28 :=
by
  sorry

end sum_of_binom_values_l45_45244


namespace minimum_S_n_at_six_l45_45820

-- Define the initial term and conditions
def a1 := -11
def a4_plus_a6 := -6

-- Define the common difference
def d (a1 : ℤ) (a4_plus_a6 : ℤ) : ℤ := (a4_plus_a6 - 2 * a1) / 8

-- Define the sequence
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a1 d : ℤ) : ℕ → ℤ
| 0     := 0
| (n+1) := sum_arithmetic_sequence a1 d n + arithmetic_sequence a1 d (n+1)

-- Define the problem statement
theorem minimum_S_n_at_six : ∀ (S_n : ℕ → ℤ), 
  (a1 = -11) → 
  (a4_plus_a6 = -6) → 
  (S_n = sum_arithmetic_sequence a1 (d a1 a4_plus_a6)) → 
  ∃ (n : ℕ), n = 6 ∧ ∀ (m : ℕ), S_n(n) ≤ S_n(m) :=
by
  sorry

end minimum_S_n_at_six_l45_45820


namespace xy_value_l45_45032

theorem xy_value (x y : ℝ) 
(h1 : 8^x / 4^(x + y) = 32)
(h2 : 16^(x + y) / 4^(3y) = 256) : x * y = -2 := 
by
  sorry

end xy_value_l45_45032


namespace average_of_last_six_l45_45402

theorem average_of_last_six (avg_13 : ℕ → ℝ) (avg_first_6 : ℕ → ℝ) (middle_number : ℕ → ℝ) :
  (∀ n, avg_13 n = 9) →
  (∀ n, n ≤ 6 → avg_first_6 n = 5) →
  (middle_number 7 = 45) →
  ∃ (A : ℝ), (∀ n, n > 6 → n < 13 → avg_13 n = A) ∧ A = 7 :=
by
  sorry

end average_of_last_six_l45_45402


namespace complex_solution_l45_45000

theorem complex_solution (z : ℂ) (hi : z * (1 - complex.I) = 3 - complex.I) : z = 2 + complex.I :=
by
  sorry

end complex_solution_l45_45000


namespace eggs_in_basket_empty_l45_45417

theorem eggs_in_basket_empty (a : ℕ) : 
  let remaining_after_first := a - (a / 2 + 1 / 2)
  let remaining_after_second := remaining_after_first - (remaining_after_first / 2 + 1 / 2)
  let remaining_after_third := remaining_after_second - (remaining_after_second / 2 + 1 / 2)
  (remaining_after_first = a / 2 - 1 / 2) → 
  (remaining_after_second = remaining_after_first / 2 - 1 / 2) → 
  (remaining_after_third = remaining_after_second / 2 -1 / 2) → 
  (remaining_after_third = 0) → 
  (a = 7) := sorry

end eggs_in_basket_empty_l45_45417


namespace tom_paid_correct_amount_l45_45932

def quantity_of_apples : ℕ := 8
def rate_per_kg_apples : ℕ := 70
def quantity_of_mangoes : ℕ := 9
def rate_per_kg_mangoes : ℕ := 45

def cost_of_apples : ℕ := quantity_of_apples * rate_per_kg_apples
def cost_of_mangoes : ℕ := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid : ℕ := cost_of_apples + cost_of_mangoes

theorem tom_paid_correct_amount :
  total_amount_paid = 965 :=
sorry

end tom_paid_correct_amount_l45_45932


namespace greatest_difference_l45_45123

theorem greatest_difference (q r : ℕ) (a b : ℕ) (hq : q = 10 * a + b) (hr : r = 10 * b + a) 
  (cond : |q - r| < 20) (h_digit_a : 1 ≤ a ∧ a ≤ 9) (h_digit_b : 0 ≤ b ∧ b ≤ 9) (ha_gt_b : a > b) :
  q - r = 18 :=
by
  sorry

end greatest_difference_l45_45123


namespace diana_bike_home_time_l45_45226

theorem diana_bike_home_time : 
  ∀ (dist total_dist speed1 speed2 time1 time2 : ℝ), 
  total_dist = 10 ∧ 
  speed1 = 3 ∧ time1 = 2 ∧ speed2 = 1 ∧
  dist = speed1 * time1 ∧
  (total_dist - dist) = speed2 * time2 → 
  time1 + time2 = 6 :=
by
  intros dist total_dist speed1 speed2 time1 time2 h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  cases h8 with h9 h10
  rw [h3, h4, h5, h6] at h10
  rw [h7, h8, h9, h10] at h10
  sorry

end diana_bike_home_time_l45_45226


namespace buffy_less_brittany_by_40_seconds_l45_45345

/-
The following statement proves that Buffy's breath-holding time was 40 seconds less than Brittany's, 
given the initial conditions about their breath-holding times.
-/
theorem buffy_less_brittany_by_40_seconds 
  (kelly_time : ℕ) 
  (brittany_time : ℕ) 
  (buffy_time : ℕ) 
  (h_kelly : kelly_time = 180) 
  (h_brittany : brittany_time = kelly_time - 20) 
  (h_buffy : buffy_time = 120)
  :
  brittany_time - buffy_time = 40 :=
sorry

end buffy_less_brittany_by_40_seconds_l45_45345


namespace least_5_digit_divisible_l45_45662

theorem least_5_digit_divisible (n : ℕ) (h1 : n ≥ 10000) (h2 : n < 100000)
  (h3 : 15 ∣ n) (h4 : 12 ∣ n) (h5 : 18 ∣ n) : n = 10080 :=
by
  sorry

end least_5_digit_divisible_l45_45662


namespace total_selling_price_l45_45968

theorem total_selling_price (cost_price1 cost_price2 cost_price3 : ℝ) (profit_percent1 profit_percent2 profit_percent3 : ℝ)
  (hp1 : cost_price1 = 192) (hp2 : profit_percent1 = 0.25)
  (hp3 : cost_price2 = 350) (hp4 : profit_percent2 = 0.15)
  (hp5 : cost_price3 = 500) (hp6 : profit_percent3 = 0.30) :
  let selling_price1 := cost_price1 + profit_percent1 * cost_price1 in
  let selling_price2 := cost_price2 + profit_percent2 * cost_price2 in
  let selling_price3 := cost_price3 + profit_percent3 * cost_price3 in
  selling_price1 + selling_price2 + selling_price3 = 1292.50 := by
  sorry

end total_selling_price_l45_45968


namespace hyperbola_equation_ellipse_equation_l45_45941
-- Importing the necessary Mathlib library

-- Problem 1: Proving the equation of the hyperbola
theorem hyperbola_equation (λ : ℝ) 
  (ellipse_eq : ∀ x y : ℝ, (y^2 / 49 + x^2 / 24 = 1) → ...)
  (asymptotes : ∀ x y : ℝ, (y = 4 / 3 * x ∨ y = - 4 / 3 * x) → ...) :
  ∀ x y : ℝ, (y^2 / (16 * λ) - x^2 / (9 * λ) = 1) → 
  (λ = 1) → 
  (y^2 / 16 - x^2 / 9 = 1) :=
by
  sorry

-- Problem 2: Proving the standard equation of the ellipse
theorem ellipse_equation (x1 y1 x2 y2 : ℝ)
  (A : (x1, y1) = (0, 5 / 3))
  (B : (x2, y2) = (1, 1)) :
    ∃ m n : ℝ, (n = 25 / 9) ∧ (m = 25 / 16) ∧
    ∀ x y : ℝ, ((x^2 / m + y^2 / n = 1) ↔ (x^2 / (25 / 16) + y^2 / (25 / 9) = 1)) :=
by
  sorry

end hyperbola_equation_ellipse_equation_l45_45941


namespace tournament_probability_l45_45631

theorem tournament_probability :
  let a := 11097 in
  let b := 167040 in
  gcd a b = 1 → 100 * a + b = 1276740 :=
by
  sorry

end tournament_probability_l45_45631


namespace closest_integer_to_cubert_of_250_is_6_l45_45552

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45552


namespace binomial_12_3_eq_220_l45_45204

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l45_45204


namespace closest_cube_root_l45_45507

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45507


namespace avg_first_11_numbers_is_48_l45_45041

theorem avg_first_11_numbers_is_48
  (A : ℝ)
  (sum_first_11 : A * 11)
  (avg_21 : (sum_first_11 + 451) / 21 = 44)
  (avg_last_11 : (451 : ℝ) / 11 = 41)
  (eleventh_num : 55) :
  A = 48 :=
by
  sorry

end avg_first_11_numbers_is_48_l45_45041


namespace parallel_line_l45_45047

theorem parallel_line (x y : ℝ) (h1 : 2 * x - y - 1 = 0) (h2 : x = -1) (h3 : y = 1) : 
  ∃ c : ℝ, 2 * -1 - 1 + c = 0 ∧ (2 * x - y + c = 0) := 
by
  use 3
  split
  · calc
      2 * (-1) - 1 + 3 = -2 - 1 + 3 : by ring
                       ... = 0    : by ring
  · sorry

end parallel_line_l45_45047


namespace max_time_to_open_briefcase_l45_45926

/-
  Theorem: Given a briefcase with a number-lock system containing a combination of 3 digits, 
  where each digit can be a number from 0 to 8, and each trial takes 3 seconds, 
  the maximum time required to open the briefcase is 36.45 minutes.
-/
theorem max_time_to_open_briefcase : 
  let num_digits := 3 in
  let choices_per_digit := 9 in
  let trials_per_sec := 3 in
  let total_combinations := choices_per_digit ^ num_digits in
  let total_time_seconds := total_combinations * trials_per_sec in
  total_time_seconds / 60 = 36.45 :=
  sorry

end max_time_to_open_briefcase_l45_45926


namespace find_original_numbers_l45_45072

theorem find_original_numbers (a b c : ℕ) (h₁ : a + b = 39) (h₂ : a + b + c = 96) : 
  a = 21 ∧ b = 18 := 
begin
  sorry
end

end find_original_numbers_l45_45072


namespace probability_of_odd_function_l45_45667

def f1 (x : ℝ) := x
def f2 (x : ℝ) := Real.sin x
def f3 (x : ℝ) := x^2

def is_odd(f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

def is_even(f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem probability_of_odd_function :
  is_odd f1 ∧ is_odd f2 ∧ is_even f3 ∧ 
  (∃ n m : ℕ, n = 3 ∧ m = 2 ∧ m / n = 2 / 3) :=
by
  sorry

end probability_of_odd_function_l45_45667


namespace product_fractions_eq_l45_45180

theorem product_fractions_eq :
  (finset.range 50).prod (λ n, (n + 1 : ℚ) / (n + 5 : ℚ)) = 1 / 35 := 
sorry

end product_fractions_eq_l45_45180


namespace line_parallel_y_axis_l45_45048

theorem line_parallel_y_axis (x y : ℝ) (h₁ : (x, y) = (-1, 3)) (h₂ : ∀ y1 : ℝ, (x, y1) = x) : 
  ∀ y : ℝ, x = -1 :=
by
  sorry

end line_parallel_y_axis_l45_45048


namespace exists_n_not_perfect_square_l45_45347

theorem exists_n_not_perfect_square (a b : ℤ) (h1 : a > 1) (h2 : b > 1) (h3 : a ≠ b) : 
  ∃ (n : ℕ), (n > 0) ∧ ¬∃ (k : ℤ), (a^n - 1) * (b^n - 1) = k^2 :=
by sorry

end exists_n_not_perfect_square_l45_45347


namespace probability_defective_approx_l45_45616

def total_smartphones : ℕ := 240
def defective_smartphones : ℕ := 84

def probability_both_defective : ℚ :=
  (defective_smartphones / total_smartphones) * ((defective_smartphones - 1) / (total_smartphones - 1))

theorem probability_defective_approx :
  probability_both_defective ≈ 0.1216 := sorry

end probability_defective_approx_l45_45616


namespace heartsuit_ratio_l45_45810

def k : ℝ := 3

def heartsuit (n m : ℕ) : ℝ := k * n^3 * m^2

theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 := 
by
  sorry

end heartsuit_ratio_l45_45810


namespace closest_integer_to_cube_root_250_l45_45536

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45536


namespace estimated_probability_l45_45627

def is_hit (n : ℕ) : Prop := n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 0
def is_miss (n : ℕ) : Prop := n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

def count_hits (triplet : (ℕ × ℕ × ℕ)) : ℕ :=
  (if is_hit triplet.1 then 1 else 0) +
  (if is_hit triplet.2 then 1 else 0) +
  (if is_hit triplet.3 then 1 else 0)

def valid_triplet (triplet : (ℕ × ℕ × ℕ)) : Prop :=
  is_hit triplet.1 ∨ is_miss triplet.1 ∧
  is_hit triplet.2 ∨ is_miss triplet.2 ∧
  is_hit triplet.3 ∨ is_miss triplet.3

def two_hits (triplet : (ℕ × ℕ × ℕ)) : Prop := count_hits triplet = 2

def random_triplets : List (ℕ × ℕ × ℕ) :=
  [(9, 0, 7), (9, 6, 6), (1, 9, 1), (9, 2, 5), 
   (2, 7, 1), (9, 3, 2), (8, 1, 2), (4, 5, 8), 
   (5, 6, 9), (6, 8, 3)]

def valid_two_hits_triplets := random_triplets.filter two_hits

theorem estimated_probability : (valid_two_hits_triplets.length : ℝ) / random_triplets.length = 0.30 := 
  by
  -- Placeholder for the proof
  sorry

end estimated_probability_l45_45627


namespace count_divisible_k_squared_minus_2k_l45_45665

theorem count_divisible_k_squared_minus_2k (n : ℕ) (h : n ≤ 333300) : 
  {k : ℕ | k ≤ n ∧ 303 ∣ (k^2 - 2*k)}.card = 4400 :=
sorry

end count_divisible_k_squared_minus_2k_l45_45665


namespace greatest_4digit_base9_divisible_by_7_l45_45908

theorem greatest_4digit_base9_divisible_by_7 :
  ∃ (n : ℕ), n < 9^4 ∧ (∃ (k : ℕ), n = k * 7) ∧ ∀ (m : ℕ), m < 9^4 ∧ (∃ (j : ℕ), m = j * 7) → n ≥ m ∧ to_digits 9 n = [9, 0, 0, 0] :=
begin
  sorry
end

end greatest_4digit_base9_divisible_by_7_l45_45908


namespace student_failed_by_40_marks_l45_45619

-- Defining the conditions
def max_marks : ℕ := 500
def passing_percentage : ℚ := 33 / 100
def marks_obtained : ℕ := 125

-- Lean 4 statement to prove the problem
theorem student_failed_by_40_marks :
  (passing_percentage * max_marks - marks_obtained : ℚ) = 40 := 
by {
  simp [passing_percentage, max_marks, marks_obtained],
  norm_num,
  sorry
}

end student_failed_by_40_marks_l45_45619


namespace closest_integer_to_cube_root_of_250_l45_45527

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45527


namespace molecular_weight_one_mole_of_iron_oxide_l45_45912

def molecular_weight (m : ℕ) (total_weight : ℝ) : ℝ := total_weight / m

theorem molecular_weight_one_mole_of_iron_oxide :
  (molecular_weight 10 1600) = 160 :=
  by 
    -- This is where the proof would go, but we're focusing on the statement, not the proof itself
    sorry

end molecular_weight_one_mole_of_iron_oxide_l45_45912


namespace remaining_wax_l45_45793

-- Define the conditions
def ounces_for_car : ℕ := 3
def ounces_for_suv : ℕ := 4
def initial_wax : ℕ := 11
def spilled_wax : ℕ := 2

-- Define the proof problem: Show remaining wax after detailing car and SUV
theorem remaining_wax {ounces_for_car ounces_for_suv initial_wax spilled_wax : ℕ} :
  initial_wax - spilled_wax - (ounces_for_car + ounces_for_suv) = 2 :=
by
  -- Defining the variables according to the conditions
  have h1 : ounces_for_car = 3 := rfl
  have h2 : ounces_for_suv = 4 := rfl
  have h3 : initial_wax = 11 := rfl
  have h4 : spilled_wax = 2 := rfl
  -- Using the conditions to calculate the remaining wax
  calc
    initial_wax - spilled_wax - (ounces_for_car + ounces_for_suv)
        = 11 - 2 - (3 + 4) : by rw [h1, h2, h3, h4]
    ... = 11 - 2 - 7 : rfl
    ... = 9 - 7 : rfl
    ... = 2 : rfl

end remaining_wax_l45_45793


namespace fraction_spent_on_meat_products_l45_45990

-- Define the total money John had
variable (M : ℝ)

-- Define the conditions from the problem statement
def spent_on_fruits_and_vegetables : ℝ := (1/5) * M
def spent_on_bakery_products : ℝ := (1/10) * M
def spent_on_candy : ℝ := 11
def total_spent : ℝ := 30

-- Given the total spending condition
axiom spending_eq (h : spent_on_fruits_and_vegetables + spent_on_bakery_products + spent_on_candy + F * M = total_spent)

-- Define the fraction F
def F : ℝ := 8 / 15

-- Problem statement: Prove that the fraction of money spent on meat products is 8/15
theorem fraction_spent_on_meat_products (h : spent_on_fruits_and_vegetables + spent_on_bakery_products + spent_on_candy + F * M = total_spent) : 
  F = 8 / 15 :=
sorry

end fraction_spent_on_meat_products_l45_45990


namespace sum_of_abc_is_40_l45_45301

theorem sum_of_abc_is_40 (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * b + c = 55) (h2 : b * c + a = 55) (h3 : c * a + b = 55) :
    a + b + c = 40 :=
by
  sorry

end sum_of_abc_is_40_l45_45301


namespace geometric_sequence_terms_l45_45277

theorem geometric_sequence_terms
  (a_3 : ℝ) (a_4 : ℝ)
  (h1 : a_3 = 12)
  (h2 : a_4 = 18) :
  ∃ (a_1 a_2 : ℝ) (q: ℝ), 
    a_1 = 16 / 3 ∧ a_2 = 8 ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 := 
by
  sorry

end geometric_sequence_terms_l45_45277


namespace sum_of_first_n_terms_l45_45427

-- Define the sequence a_n as per the provided conditions
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 8
  else a_n (n - 1) + 2^n

-- Define the sum of the sequence up to the nth term
def sum_a_n (n : ℕ) : ℕ :=
  (Finset.range(n)).sum (λ i, a_n (i + 1))

-- The Lean statement of the proof problem
theorem sum_of_first_n_terms (n : ℕ) : sum_a_n n = 2^(n + 2) + 4 * n - 4 :=
by
  sorry

end sum_of_first_n_terms_l45_45427


namespace simplify_and_evaluate_expression_l45_45851

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -3) : (1 + 1/(x+1)) / ((x^2 + 4*x + 4) / (x+1)) = -1 :=
by
  sorry

end simplify_and_evaluate_expression_l45_45851


namespace lower_bound_third_inequality_l45_45757

theorem lower_bound_third_inequality (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : 8 > x ∧ x > 0)
  (h4 : x + 1 < 9) :
  x = 7 → ∃ l < 7, ∀ y, l < y ∧ y < 9 → y = x := 
sorry

end lower_bound_third_inequality_l45_45757


namespace closest_integer_to_cube_root_of_250_l45_45482

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45482


namespace binomial_12_3_eq_220_l45_45198

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l45_45198


namespace decreasing_f_range_l45_45282

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log x - 2 * k * x - 1

theorem decreasing_f_range (k : ℝ) (x₁ x₂ : ℝ) (h₁ : 2 ≤ x₁) (h₂ : x₁ < x₂) (h₃ : x₂ ≤ 4) :
  k ≥ 1 / 4 → (x₁ - x₂) * (f x₁ k - f x₂ k) < 0 :=
sorry

end decreasing_f_range_l45_45282


namespace cost_price_of_article_l45_45883

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 := 
by 
  sorry

end cost_price_of_article_l45_45883


namespace correct_operation_l45_45568

theorem correct_operation :
  ∀ (a x y m : ℝ), (m^2 * (-m)^4 = m^6) ∧ ¬ (a^3 - a^2 = a) ∧ ¬ (ax + ay = axy) ∧ ¬ ((m^2)^3 = m^5) :=
by
  intros a x y m
  split
  -- Prove the correct operation
  {
    calc
      m^2 * (-m)^4 = m^2 * m^4   : by rw pow_mul, simp
      ...        = m^(2+4)      : by rw pow_add
      ...        = m^6 : rfl,
  }
  -- Prove the incorrect operations
  {
    split
    {
      intros h,
      -- use contradiction or any contradiction reasoning here
      sorry,
    }
    {
      split
      {
        intros h,
        -- use contradiction or any contradiction reasoning here
        sorry,
      }
      {
        intros h,
        -- use contradiction or any contradiction reasoning here
        sorry,
      }
    }
  }

end correct_operation_l45_45568


namespace triple_three_number_l45_45952

-- Define the four digits of the number M
variables {a b c d : ℕ}

-- Define the properties M must satisfy
def M_props (M : ℕ) : Prop :=
  (1000 * a + 100 * b + 10 * c + d = M) ∧
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (d > 1) ∧
  (b + c = 3 * (a + d)) ∧
  (a + b + c + d = 16) ∧
  (∃ n : ℕ, (M - (1000 * a + 100 * c + 10 * b + d) = 270 * (n * (6 - b) / (4 - 2 * d))) ∧
  (n = (1 / 3) * ((6 - b) / (4 - 2 * d))))

theorem triple_three_number (M : ℕ) : 
  M_props M → M = 1933 ∨ M = 1393 :=
by
  sorry

end triple_three_number_l45_45952


namespace intersection_M_N_l45_45735

def M (x : ℝ) : Prop := (1 / 2) ^ x ≥ 1

def N (x : ℝ) : Prop := ∃ y : ℝ, y = log x + 2

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | -2 < x ∧ x ≤ 0} :=
by
  sorry

end intersection_M_N_l45_45735


namespace perfect_square_value_of_b_l45_45871

theorem perfect_square_value_of_b :
  (∃ b : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + b * b) = (11.98 + b)^2) →
  (∃ b : ℝ, b = 0.02) :=
sorry

end perfect_square_value_of_b_l45_45871


namespace closest_integer_to_cbrt_250_l45_45559

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45559


namespace min_convex_polygons_partition_l45_45027

noncomputable def min_convex_polygons (n : ℕ) : ℕ :=
  n + 1

theorem min_convex_polygons_partition (n : ℕ) (M : Polygon) (M' : Polygon)
  (hM : M.is_regular_n_gon n)
  (hM' : M' = M.rotate_center (π / n)) :
    min_convex_polygons n = n + 1 := by
  sorry

end min_convex_polygons_partition_l45_45027


namespace percentage_increase_in_expenses_l45_45958

theorem percentage_increase_in_expenses :
  let S := 5000 in
  let initial_savings := 0.20 * S in
  let final_savings := 200 in
  let initial_expenses := S - initial_savings in
  let new_expenses := S - final_savings in
  let increase_in_expenses := new_expenses - initial_expenses in
  (increase_in_expenses / initial_expenses) * 100 = 20 :=
by
  sorry

end percentage_increase_in_expenses_l45_45958


namespace solve_for_a_l45_45731

def matrix_A : Matrix (Fin 2) (Fin 3) ℤ := ![![1, 2, -1], ![2, 2, -3]]
def matrix_B (a : ℤ) : Matrix (Fin 3) (Fin 1) ℤ := ![![a], ![-2 * a], ![3 * a]]
def matrix_AB (a : ℤ) : Matrix (Fin 2) (Fin 1) ℤ := matrix_A.mul (matrix_B a)

theorem solve_for_a (a : ℤ) (h : matrix_AB a = ![![12], ![22]]) : a = -2 :=
by
  sorry

end solve_for_a_l45_45731


namespace new_percentage_girls_proof_l45_45592

-- Given initial conditions
variables (initial_students : ℕ) (percentage_girls : ℝ) (new_boys : ℕ)
variables (initial_students = 20) (percentage_girls = 0.40) (new_boys = 5)

-- Define the number of girls initially
def initial_girls : ℕ := (percentage_girls * initial_students).toNat

-- Define the total number of students after new boys join
def total_students : ℕ := initial_students + new_boys

-- The number of girls remains the same
def girls := initial_girls

-- Calculate the new percentage of girls
def new_percentage_girls : ℝ := (girls.toRat / total_students) * 100

theorem new_percentage_girls_proof : new_percentage_girls = 32 := 
by sorry

end new_percentage_girls_proof_l45_45592


namespace question_one_question_two_l45_45281

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - 2 * (cos x) ^ 2

theorem question_one : f (π / 6) = 0 :=
sorry

theorem question_two :
  ∀ k : ℤ, ∀ x : ℝ, -π / 6 + k * π ≤ x ∧ x ≤ π / 3 + k * π → (deriv f x) > 0 :=
sorry

end question_one_question_two_l45_45281


namespace max_value_of_a_b_c_d_l45_45357

open Matrix

noncomputable def B (a b c d : ℤ) : Matrix (Fin 3) (Fin 3) ℤ :=
  (1 : ℤ) / 7 • ![
    [-a, a, b],
    [0, c, d],
    [b, 0, c]
  ]

theorem max_value_of_a_b_c_d (a b c d : ℤ) (hB2 : (B a b c d).mul (B a b c d) = 1) : 
  ∃ a b c d, a + b + c + d ≤ 7 :=
begin
  sorry
end

end max_value_of_a_b_c_d_l45_45357


namespace larry_wins_probability_l45_45767

theorem larry_wins_probability :
  let larry_success := (1:ℚ)/2
  let julius_failure := (2:ℚ)/3
  let series := (finset.range ∞).sum (λ n, if n % 2 = 0 then 0 else larry_success * julius_failure^(n-1))
  series = 9/16 :=
by
  -- Definitions as given in the conditions
  let larry_success := (1:ℚ)/2
  let julius_failure := (2:ℚ)/3
  
  -- Construct the geometric series
  let series := (finset.range ∞).sum (λ n, if n % 2 = 0 then 0 else larry_success * julius_failure^(n-1))
  
  -- State the probability as given
  series = 9/16
  sorry

end larry_wins_probability_l45_45767


namespace hexagon_monochromatic_triangle_probability_l45_45653

noncomputable theory

def hexagon_edges : ℕ := 15
def triangles_in_hexagon : ℕ := 20
def non_monochromatic_triangle_probability : ℝ := (3 / 4) ^ triangles_in_hexagon
def monochromatic_triangle_probability : ℝ := 1 - non_monochromatic_triangle_probability

theorem hexagon_monochromatic_triangle_probability :
  monochromatic_triangle_probability = 999 / 1000 :=
  by 
    -- Simplifying the terms, calculation goes here (if pursued)
    -- ideally will form the above pre-determined solution
    sorry

end hexagon_monochromatic_triangle_probability_l45_45653


namespace distance_to_lateral_face_l45_45875

theorem distance_to_lateral_face 
  (height : ℝ) 
  (angle : ℝ) 
  (h_height : height = 6 * Real.sqrt 6)
  (h_angle : angle = Real.pi / 4) : 
  ∃ (distance : ℝ), distance = 6 * Real.sqrt 30 / 5 :=
by
  sorry

end distance_to_lateral_face_l45_45875


namespace hotel_rooms_count_l45_45160

theorem hotel_rooms_count
  (TotalLamps : ℕ) (TotalChairs : ℕ) (TotalBedSheets : ℕ)
  (LampsPerRoom : ℕ) (ChairsPerRoom : ℕ) (BedSheetsPerRoom : ℕ) :
  TotalLamps = 147 → 
  TotalChairs = 84 → 
  TotalBedSheets = 210 → 
  LampsPerRoom = 7 → 
  ChairsPerRoom = 4 → 
  BedSheetsPerRoom = 10 →
  (TotalLamps / LampsPerRoom = 21) ∧ 
  (TotalChairs / ChairsPerRoom = 21) ∧ 
  (TotalBedSheets / BedSheetsPerRoom = 21) :=
by
  intros
  sorry

end hotel_rooms_count_l45_45160


namespace integral_cos_zero_l45_45655

noncomputable def integral_cos_from_0_to_pi : ℝ :=
  ∫ x in 0..π, Real.cos x

theorem integral_cos_zero : integral_cos_from_0_to_pi = 0 := by
  sorry

end integral_cos_zero_l45_45655


namespace proof_arithmetic_sequence_geometric_sequence_l45_45695

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}
variable {b_n : ℕ → ℝ}
variable {T_n : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n+1) - a_n n = d

def sum_of_first_n_terms (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = ∑ i in range n, a_n i

def geometric_subsequence (a_n : ℕ → ℝ) : Prop :=
  ∃ a₁ a₂ a₄ : ℝ,
  a_n 1 = a₁ ∧
  a_n 2 = a₁ + d ∧
  a_n 4 = a₁ + 3 * d ∧
  ∃ q : ℝ, q ≠ 0 ∧ (a_n 2)^2 = a_n 1 * a_n 4

theorem proof_arithmetic_sequence_geometric_sequence
  (d : ℝ) (h₀ : d ≠ 0)
  (h₁ : arithmetic_sequence a_n d)
  (h₂ : sum_of_first_n_terms S_n a_n)
  (h₃ : S_n 10 = 110)
  (h₄ : geometric_subsequence a_n) :
  (∀ n, a_n n = 2 * n) ∧ (∀ n, let b_n := 1 / ((a_n n) - 1) * ((a_n n) + 1) in T_n n = n / (2 * n + 1)) :=
sorry

end proof_arithmetic_sequence_geometric_sequence_l45_45695


namespace time_to_travel_downstream_l45_45118

-- Definitions based on the conditions.
def speed_boat_still_water := 40 -- Speed of the boat in still water (km/hr)
def speed_stream := 5 -- Speed of the stream (km/hr)
def distance_downstream := 45 -- Distance to be traveled downstream (km)

-- The proof statement
theorem time_to_travel_downstream : (distance_downstream / (speed_boat_still_water + speed_stream)) = 1 :=
by
  -- This would be the place to include the proven steps, but it's omitted as per instructions.
  sorry

end time_to_travel_downstream_l45_45118


namespace first_dog_walks_two_miles_per_day_l45_45169

variable (x : ℝ)

theorem first_dog_walks_two_miles_per_day  
  (h1 : 7 * x + 56 = 70) : 
  x = 2 := 
by 
  sorry

end first_dog_walks_two_miles_per_day_l45_45169


namespace y_exceeds_x_by_25_percent_l45_45575

-- Define the variables and condition
variables {x y : ℝ}
hypothesis (h : x = 0.80 * y)

-- State the theorem to prove
theorem y_exceeds_x_by_25_percent (h : x = 0.80 * y) : ((y - x) / x) * 100 = 25 := 
by
  -- Provide the proof steps in text if you want to verify manually
  -- But end the statement with sorry as the proof is not required
  sorry

end y_exceeds_x_by_25_percent_l45_45575


namespace cube_surface_area_and_volume_l45_45071

theorem cube_surface_area_and_volume
  (r : ℝ) (a : ℝ)
  (h_radius : r = 2 * sqrt 3)
  (h_side : a = 4) :
  ∃ S V : ℝ,
    S = 6 * a^2 ∧
    V = a^3 ∧
    S = 96 ∧
    V = 64 := 
by
  use [6 * a^2, a^3]
  split
  · exact h_side ▸ rfl
  split
  · exact h_side ▸ rfl
  split
  · exact h_side ▸ rfl
  · exact h_side ▸ rfl

end cube_surface_area_and_volume_l45_45071


namespace graph_passes_through_quadrants_l45_45923

-- Definitions based on the conditions
def linear_function (x : ℝ) : ℝ := -2 * x + 1

-- The property to be proven
theorem graph_passes_through_quadrants :
  (∃ x > 0, linear_function x > 0) ∧  -- Quadrant I
  (∃ x < 0, linear_function x > 0) ∧  -- Quadrant II
  (∃ x > 0, linear_function x < 0) := -- Quadrant IV
sorry

end graph_passes_through_quadrants_l45_45923


namespace triangles_containing_center_l45_45696

theorem triangles_containing_center (n : ℕ) : 
  let k : ℕ := 2 * n + 1 in
  let total_triangles : ℕ := (k * (k - 1) * (k - 2)) / 6 in
  let triangles_not_containing_center : ℕ := k * (n * (n - 1)) / 2 in
  total_triangles - triangles_not_containing_center = (n * (n + 1) * (2 * n + 1)) / 6 :=
by 
  sorry

end triangles_containing_center_l45_45696


namespace water_evaporation_l45_45134

theorem water_evaporation :
  ∀ (initial_amount : ℝ) (days : ℕ) (percentage_evaporated : ℝ),
    initial_amount = 10 →
    days = 50 →
    percentage_evaporated = 0.04 →
    let total_evaporated := (percentage_evaporated / 100) * initial_amount in
    let daily_evaporation := total_evaporated / days in
    daily_evaporation = 0.0008 :=
by
  intros initial_amount days percentage_evaporated h1 h2 h3
  let total_evaporated := (percentage_evaporated / 100) * initial_amount
  let daily_evaporation := total_evaporated / days
  have h4 : total_evaporated = (0.04 / 100) * 10 := by rw [h1, h3]
  have h5 : total_evaporated = 0.04 := by norm_num at h4; exact h4
  have h6 : daily_evaporation = 0.04 / 50 := by rw [h5, h2]
  norm_num at h6
  exact h6

end water_evaporation_l45_45134


namespace trigonometric_sum_evaluation_l45_45938

theorem trigonometric_sum_evaluation :
  5 * sin (90 * Real.pi / 180) + 2 * cos (0 * Real.pi / 180) - 3 * sin (270 * Real.pi / 180) + 10 * cos (180 * Real.pi / 180) = -2 := 
by
  sorry

end trigonometric_sum_evaluation_l45_45938


namespace eccentricity_is_3_over_5_l45_45308

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  let c := 2*b - a in
  let e := c/a in
  e

theorem eccentricity_is_3_over_5 (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : 2*b = a + (2*b - a)) (h4 : a^2 = b^2 + (2*b - a)^2) :
  eccentricity_of_ellipse a b h1 h2 = 3/5 := by
  sorry

end eccentricity_is_3_over_5_l45_45308


namespace number_of_pages_in_bible_l45_45798

-- Definitions based on conditions
def hours_per_day := 2
def pages_per_hour := 50
def weeks := 4
def days_per_week := 7

-- Hypotheses transformed into mathematical facts
def total_days := weeks * days_per_week
def total_hours := total_days * hours_per_day
def total_pages := total_hours * pages_per_hour

-- Theorem to prove the Bible length based on conditions
theorem number_of_pages_in_bible : total_pages = 2800 := 
by
  sorry

end number_of_pages_in_bible_l45_45798


namespace circumcircle_triangle_congruence_l45_45813

noncomputable def orthocenter (A B C H : Point) : Prop :=
  IsOrthocenter A B C H

noncomputable def circumcenter (X Y Z O : Point) : Prop :=
  IsCircumcenter X Y Z O

theorem circumcircle_triangle_congruence
  (A B C H O1 O2 O3 : Point)
  (h : orthocenter A B C H)
  (hc1 : circumcenter B H C O1)
  (hc2 : circumcenter A H C O2)
  (hc3 : circumcenter A H B O3) :
  congruent (Triangle.mk O1 O2 O3) (Triangle.mk A B C) :=
sorry

end circumcircle_triangle_congruence_l45_45813


namespace Yan_distance_ratio_l45_45924

theorem Yan_distance_ratio (d x : ℝ) (v : ℝ) (h1 : d > 0) (h2 : x > 0) (h3 : x < d)
  (h4 : 7 * (d - x) = x + d) : 
  x / (d - x) = 3 / 4 :=
by
  sorry

end Yan_distance_ratio_l45_45924


namespace closest_integer_to_cube_root_of_250_l45_45488

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45488


namespace binom_12_3_eq_220_l45_45187

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l45_45187


namespace cyclic_quadrilateral_diagonals_perpendicular_implies_opposite_side_equal_l45_45046

variables {A B C D O P K : Type}
variables [In_circle A B C D O] -- Definition needed to represent 'ABCD' inscribed in the circle with center O
variables [Intersect_at P (Diagonal AC BD)] -- Definition needed to represent 'AC and BD intersect at P'
variables [Perpendicular OP BC] -- Definition needed to represent that OP is perpendicular to BC

theorem cyclic_quadrilateral_diagonals_perpendicular_implies_opposite_side_equal :
  AB = CD :=
sorry

end cyclic_quadrilateral_diagonals_perpendicular_implies_opposite_side_equal_l45_45046


namespace ratio_of_drinking_speeds_l45_45370

def drinking_ratio(mala_portion usha_portion : ℚ) (same_time: Bool) (usha_fraction: ℚ) : ℚ :=
if same_time then mala_portion / usha_portion else 0

theorem ratio_of_drinking_speeds
  (mala_portion : ℚ)
  (usha_portion : ℚ)
  (same_time : Bool)
  (usha_fraction : ℚ)
  (usha_drank : usha_fraction = 2 / 10)
  (mala_drank : mala_portion = 1 - usha_fraction)
  (equal_time : same_time = tt)
  (ratio : drinking_ratio mala_portion usha_portion same_time usha_fraction = 4) :
  mala_portion / usha_portion = 4 :=
by
  sorry

end ratio_of_drinking_speeds_l45_45370


namespace closest_integer_to_cube_root_250_l45_45533

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45533


namespace probability_convex_quadrilateral_l45_45391

theorem probability_convex_quadrilateral (n : ℕ) (h : n = 7) :
  (∑ x in (finset.powerset_len 4 (finset.univ : finset (finset.univ : finset (finset.range (∑ y in finset.univ : n ports on a circle).card 4 (finset.range 7))).card, by { rw [finset.card_univ], exact finset.card_eq.mp (finset.card_power.single_list card_eq_7), rw ← h, sorry, sorry)

end probability_convex_quadrilateral_l45_45391


namespace problem_solution_l45_45728

noncomputable def f : ℕ → ℝ → ℝ
| 0       := sin
| (n + 1) := deriv (f n)

theorem problem_solution :
  f 1 (15 * (real.pi / 180)) + f 2 (15 * (real.pi / 180)) + f 3 (15 * (real.pi / 180)) + ⋯ + f 2017 (15 * (real.pi / 180)) = (real.sqrt 6 + real.sqrt 2) / 4 :=
sorry

end problem_solution_l45_45728


namespace product_formula_l45_45174

theorem product_formula :
  (∏ n in Finset.range 15 + 1, (n * (n + 3)) / (n + 5)^2) = 75 / 1550400 := 
by
  -- Proof will go here
sorry

end product_formula_l45_45174


namespace area_of_tangential_quadrilateral_l45_45935

theorem area_of_tangential_quadrilateral (a r R : ℝ) (h1 : a > 0) (h2 : r > 0) (h3 : a + a * r^3 = a * r + a * r^2) : 
    (r = 1) → 
    (calculate_area a r R = a^2)
    :=
by sorry

noncomputable def calculate_area (a r R : ℝ) : ℝ :=
    let s1 := a
    let s2 := a * r
    let s3 := a * r^2
    let s4 := a * r^3
    let side_lengths_sum := s1 + s2 + s3 + s4
    let area := a^2
    area

end area_of_tangential_quadrilateral_l45_45935


namespace lattice_points_in_region_l45_45141

theorem lattice_points_in_region : 
  (∃ (n : ℤ), ∀ (x y : ℤ), y = |x| ∨ y = -x^2 + 9 → (x, y) ∈ ℤ × ℤ) → 
  (set.count (λ p : ℤ × ℤ, p.y ≥ |p.x| ∧ p.y ≤ -p.x^2 + 9) = 34) :=
by
  sorry

end lattice_points_in_region_l45_45141


namespace lilliputian_matchboxes_fit_l45_45098

variable (L W H : ℝ)

def lilliputian_box_volume := (L / 12) * (W / 12) * (H / 12)

theorem lilliputian_matchboxes_fit (h1 : L > 0) (h2 : W > 0) (h3 : H > 0) :
  (L * W * H) / ((L / 12) * (W / 12) * (H / 12)) = 1728 := by
  sorry

end lilliputian_matchboxes_fit_l45_45098


namespace find_ab_max_value_g_l45_45727

open Real

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x + b * cos x

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := a * cos x - b * sin x

-- Conditions on the tangent at π/3
def tangent_cond_1 (a b : ℝ) : Prop := (f a b (π / 3) = 0)
def tangent_cond_2 (a b : ℝ) : Prop := (f' a b (π / 3) = 1)

-- Define the function g
def g (k a b : ℝ) (x : ℝ) : ℝ := k * x - f a b (x + π / 3)

-- Maximum value of g on the interval [0, π/2] given k
def max_g (k a b : ℝ) : ℝ :=
  if k ≤ 2 / π then g k a b 0
  else g k a b (π / 2)

theorem find_ab : ∃ (a b : ℝ), tangent_cond_1 a b ∧ tangent_cond_2 a b ∧ a = 1 / 2 ∧ b = -sqrt 3 / 2 :=
sorry

theorem max_value_g (k : ℝ) : 
  ∀ (a b : ℝ), tangent_cond_1 a b ∧ tangent_cond_2 a b →
  max_g k a b = if k ≤ 2 / π then 0 else k * π / 2 - 1 :=
sorry

end find_ab_max_value_g_l45_45727


namespace area_is_rational_l45_45698

-- Definitions of the vertices of the triangle
def point1 : (ℤ × ℤ) := (2, 3)
def point2 : (ℤ × ℤ) := (5, 7)
def point3 : (ℤ × ℤ) := (3, 4)

-- Define a function to calculate the area of a triangle given vertices with integer coordinates
def triangle_area (A B C: (ℤ × ℤ)) : ℚ :=
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

-- Define the area of our specific triangle
noncomputable def area_of_triangle_with_given_vertices := triangle_area point1 point2 point3

-- Proof statement
theorem area_is_rational : ∃ (Q : ℚ), Q = area_of_triangle_with_given_vertices := 
sorry

end area_is_rational_l45_45698


namespace binomial_12_3_eq_220_l45_45209

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l45_45209


namespace arithmetic_sequence_problem_l45_45716

-- Definitions of the terms in the problem
def arithmetic_sequence (a1 d: ℝ) (n: ℕ) : ℝ := a1 + (n-1) * d
def sum_first_n_terms (a1 d: ℝ) (n: ℕ) : ℝ := n/2 * (2*a1 + (n-1) * d)

-- The problem statement
theorem arithmetic_sequence_problem (a1 d : ℝ) (h: d ≠ 0) (h1: arithmetic_sequence a1 d 10 = sum_first_n_terms a1 d 4) :
  (sum_first_n_terms a1 d 8) / (arithmetic_sequence a1 d 9) = 4 :=
by 
  sorry

end arithmetic_sequence_problem_l45_45716


namespace shaded_region_area_l45_45762

noncomputable def area_shaded_region (radius : ℝ) : ℝ :=
  let area_triangle := 2 * (1 / 2 * radius * radius)
  let area_sector := 2 * (1 / 4 * Math.pi * radius * radius)
  area_triangle + area_sector

theorem shaded_region_area
  (radius : ℝ) (h_radius : radius = 6):
  area_shaded_region radius = 36 + 18 * Real.pi := by
  sorry

end shaded_region_area_l45_45762


namespace tangents_circumcircle_intersect_on_AC_l45_45766

-- Let ABC be a non-isosceles triangle.
variables {A B C O I B' : Type*}
variables [non-isosceles_triangle A B C]
variables [is_circumcenter O A B C]
variables [is_incenter I A B C]
variables [is_reflection B' B (OI : line)]
variables [lies_in_angle B' B (I : point A B)]

-- The tangents to the circumcircle of triangle BB'I at B' and I intersect on AC.
theorem tangents_circumcircle_intersect_on_AC : 
  tangents_circumcircle_intersect_on (circle_circumscribed_triangle (B B' I)) B' I A C :=
by 
  sorry

end tangents_circumcircle_intersect_on_AC_l45_45766


namespace minimum_value_f_1_minimum_value_f_e_minimum_value_f_m_minus_1_range_of_m_l45_45287

noncomputable def f (x m : ℝ) : ℝ := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem minimum_value_f_1 (m : ℝ) : (m ≤ 2) → f 1 m = 2 - m := sorry

theorem minimum_value_f_e (m : ℝ) : (m ≥ Real.exp 1 + 1) → f (Real.exp 1) m = Real.exp 1 - m - (m - 1) / Real.exp 1 := sorry

theorem minimum_value_f_m_minus_1 (m : ℝ) : (2 < m ∧ m < Real.exp 1 + 1) → 
  f (m - 1) m = m - 2 - m * Real.log (m - 1) := sorry

theorem range_of_m (m : ℝ) : 
  (m ≤ 2) → 
  (∃ x1 ∈ Set.Icc (Real.exp 1) (Real.exp 1 ^ 2), ∀ x2 ∈ Set.Icc (-2 : ℝ) 0, f x1 m ≤ g x2) → 
  Real.exp 1 - m - (m - 1) / Real.exp 1 ≤ 1 → 
  (m ≥ (Real.exp 1 ^ 2 - Real.exp 1 + 1) / (Real.exp 1 + 1) ∧ m ≤ 2) := sorry

end minimum_value_f_1_minimum_value_f_e_minimum_value_f_m_minus_1_range_of_m_l45_45287


namespace median_correct_mean_correct_mode_correct_l45_45237

def data := [158, 149, 155, 157, 156, 162, 155, 168]

def median := 155.5
def mean   := 157.5
def mode   := 155

theorem median_correct : 
  (List.median (List.sort data) = median) :=
sorry

theorem mean_correct : 
  (List.sum data / (data.length:ℝ) = mean) :=
sorry

theorem mode_correct : 
  (List.mode data = mode) :=
sorry

end median_correct_mean_correct_mode_correct_l45_45237


namespace regular_pentagon_of_equal_altitudes_and_medians_l45_45625

theorem regular_pentagon_of_equal_altitudes_and_medians
  (P : Type)
  [metric_space P]
  [has_dist P]
  (ABCDE : fin 5 → P)
  (altitude : fin 5 → P → ℝ)
  (median : fin 5 → P → ℝ)
  (h : ∀ i, altitude i (ABCDE i) = altitude 0 (ABCDE 0))
  (m : ∀ i, median i (ABCDE i) = median 0 (ABCDE 0)) :
  (∀ i j, dist (ABCDE i) (ABCDE ((i + 1) % 5)) = dist (ABCDE j) (ABCDE ((j + 1) % 5))) ∧
  (∀ i j, ∠(ABCDE (i % 5), ABCDE ((i + 2) % 5), ABCDE ((i + 4) % 5)) = ∠(ABCDE (j % 5), ABCDE ((j + 2) % 5), ABCDE ((j + 4) % 5))) :=
by sorry

end regular_pentagon_of_equal_altitudes_and_medians_l45_45625


namespace closest_integer_to_cube_root_of_250_l45_45481

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45481


namespace response_rate_l45_45137

theorem response_rate (responses_needed : ℕ) (questionnaires_mailed : ℕ) (response_rate_percent : ℝ) :
  responses_needed = 300 → questionnaires_mailed = 375 → response_rate_percent = (300 / 375 * 100) →
  response_rate_percent = 80 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end response_rate_l45_45137


namespace broken_pieces_correct_l45_45035

variable (pieces_transported : ℕ)
variable (shipping_cost_per_piece : ℝ)
variable (compensation_per_broken_piece : ℝ)
variable (total_profit : ℝ)
variable (broken_pieces : ℕ)

def logistics_profit (pieces_transported : ℕ) (shipping_cost_per_piece : ℝ) 
                     (compensation_per_broken_piece : ℝ) (broken_pieces : ℕ) : ℝ :=
  shipping_cost_per_piece * (pieces_transported - broken_pieces) - compensation_per_broken_piece * broken_pieces

theorem broken_pieces_correct :
  pieces_transported = 2000 →
  shipping_cost_per_piece = 0.2 →
  compensation_per_broken_piece = 2.3 →
  total_profit = 390 →
  logistics_profit pieces_transported shipping_cost_per_piece compensation_per_broken_piece broken_pieces = total_profit →
  broken_pieces = 4 :=
by
  intros
  sorry

end broken_pieces_correct_l45_45035


namespace solve_for_y_l45_45854

theorem solve_for_y (y : ℝ) (h : 7 - y = 12) : y = -5 := sorry

end solve_for_y_l45_45854


namespace fraction_calculation_l45_45636

theorem fraction_calculation : 
  (1 / 4 + 1 / 6 - 1 / 2) / (-1 / 24) = 2 := 
by 
  sorry

end fraction_calculation_l45_45636


namespace closest_integer_to_cube_root_of_250_l45_45519

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45519


namespace circle_properties_l45_45717

noncomputable def circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2*x - 2*y + 1 = 0
noncomputable def line_eq : ℝ → ℝ → Prop := λ x y, x - y = 2
def center_of_circle := (1 : ℝ, 1 : ℝ)
def radius_of_circle := (1 : ℝ)

theorem circle_properties :
  ∀ (x y : ℝ), circle_eq x y ↔ dist (1, 1) = 1 ∧
    dist_to_line (1, 1) (x - y = 2) = sqrt 2 ∧
    min_dist_of_points_on_circle_to_line (circle_eq x y) (line_eq x y) = sqrt 2 - 1 :=
by sorry

end circle_properties_l45_45717


namespace closest_integer_to_cbrt_250_l45_45564

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45564


namespace age_difference_l45_45467

variable Hiram_age : ℕ := 40
variable Allyson_age : ℕ := 28

theorem age_difference :
  let sum_Hiram := Hiram_age + 12
  let twice_Allyson := 2 * Allyson_age
  (twice_Allyson - sum_Hiram = 4) := by
  let sum_Hiram := Hiram_age + 12
  let twice_Allyson := 2 * Allyson_age
  sorry

end age_difference_l45_45467


namespace closest_integer_to_cube_root_of_250_l45_45508

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45508


namespace choose_30_5_eq_142506_l45_45601

theorem choose_30_5_eq_142506 : nat.choose 30 5 = 142506 := 
by sorry

end choose_30_5_eq_142506_l45_45601


namespace closest_integer_to_cubert_of_250_is_6_l45_45554

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45554


namespace math_problem_l45_45834

noncomputable def problem_statement : Prop :=
  let A : ℝ × ℝ := (5, 6)
  let B : ℝ × ℝ := (8, 3)
  let slope : ℝ := (B.snd - A.snd) / (B.fst - A.fst)
  let y_intercept : ℝ := A.snd - slope * A.fst
  slope + y_intercept = 10

theorem math_problem : problem_statement := sorry

end math_problem_l45_45834


namespace number_of_shapes_correct_sum_of_areas_correct_l45_45101

def initial_paper_dimensions : ℕ × ℕ := (20, 12)

def area_after_folds (k : ℕ) : ℕ :=
  240 * (k + 1) / 2^k

def number_of_shapes_after_4_folds : ℕ := 5

def sum_of_areas_after_n_folds (n : ℕ) : ℝ :=
  240 * (3 - (n + 3) / (2^n))

theorem number_of_shapes_correct : number_of_shapes_after_4_folds = 5 := sorry

theorem sum_of_areas_correct (n : ℕ) : 
  (∑ k in Finset.range n, area_after_folds (k + 1)) = sum_of_areas_after_n_folds n := sorry

end number_of_shapes_correct_sum_of_areas_correct_l45_45101


namespace shortest_path_is_sqrt_5_l45_45044

-- Define the setup with given conditions
structure Cone :=
  (base_radius : ℝ)
  (slant_height : ℝ)
  (vertex : String)
  (midpoint_SB : String)
  (cross_section_SAB : String)

-- Given the base radius and slant height of the cone
def cone : Cone := {
  base_radius := 1,
  slant_height := 2,
  vertex := "S",
  midpoint_SB := "C",
  cross_section_SAB := "triangle_SAB"
}

-- Define the shortest path function (to be proven)
noncomputable def shortest_path_from_A_to_C (A B C : String) (l : ℝ) : ℝ :=
  sqrt(5)

-- Statement: Given the conditions, the shortest path from A to C
theorem shortest_path_is_sqrt_5 : 
  cone.base_radius = 1 ∧ 
  cone.slant_height = 2 →
  shortest_path_from_A_to_C "A" "B" "C" (2 * ℝ.pi) = sqrt(5) :=
begin
  intros,
  sorry
end

end shortest_path_is_sqrt_5_l45_45044


namespace loser_of_12th_match_l45_45980

-- Definitions based on the conditions in the problem
def played_matches (A B C : Type) : Prop :=
  ∃ (play : Type → ℕ), play A = 12 ∧ play B = 21 ∧ play C = 8 

-- Given condition: The loser of a match serves as the referee in the next match
def loser_serves_referee : Prop :=
  true -- this is a trivial condition in this context, just to reflect the given problem

-- Definition representing the question and conditions
def competition : Prop :=
  ∀ (n : ℕ), (n = 12) → ∃ (A B C : Type), played_matches A B C → loser_serves_referee → loser_of_nth_match n = A

-- Statement to be proved
theorem loser_of_12th_match (A B C : Type) (play : Type → ℕ) (match_results : list (Type × Type)) :
  played_matches A B C → loser_serves_referee → loser_of_nth_match 12 = A :=
by
  sorry

end loser_of_12th_match_l45_45980


namespace arithmetic_sequence_x_value_l45_45215

theorem arithmetic_sequence_x_value
  (x : ℝ)
  (h₁ : 2 * x - (1 / 3) = (x + 4) - 2 * x) :
  x = 13 / 3 := by
  sorry

end arithmetic_sequence_x_value_l45_45215


namespace polygon_sides_l45_45146

theorem polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 140 * n) : n = 9 :=
sorry

end polygon_sides_l45_45146


namespace number_of_D_divisible_8D4_by_4_l45_45644

def count_divisible_by_4 : ℕ := 
  (Finset.filter (λ D: ℕ, (10 * D + 4) % 4 = 0) (Finset.range 10)).card

theorem number_of_D_divisible_8D4_by_4 : count_divisible_by_4 = 5 := 
by 
  sorry

end number_of_D_divisible_8D4_by_4_l45_45644


namespace num_cuboids_diagonal_l45_45842

theorem num_cuboids_diagonal (L a b c : ℕ) (hL : L = 2002) (ha : a = 2) (hb : b = 7) (hc : c = 13) :
  let ab := a * b
      ac := a * c
      bc := b * c
      abc := a * b * c in
  let num_diagonal_cuboids := L / a + L / b + L / c - L / ab - L / ac - L / bc + L / abc in
  num_diagonal_cuboids * (L / (a * b * c)) = 1210 :=
by
  sorry

end num_cuboids_diagonal_l45_45842


namespace closest_integer_to_cbrt_250_l45_45560

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45560


namespace binomial_12_3_eq_220_l45_45207

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l45_45207


namespace polynomial_inequality_l45_45816

theorem polynomial_inequality (n : ℕ) (a : Fin (n + 1) → ℝ) :
  ∃ k ∈ Finset.range (n + 1), ∀ x ∈ Set.Icc (0 : ℝ) 1, 
  ∑ i in Finset.range (n + 1), a i * x^i ≤ ∑ i in Finset.range (k + 1), a i :=
by
  sorry

end polynomial_inequality_l45_45816


namespace number_of_correct_statements_l45_45366

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def is_decreasing_function (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y
def has_real_root (f : ℝ → ℝ) : Prop := ∃ x, f x = x

theorem number_of_correct_statements : 
  let conditions := [
    ∀ f, is_odd_function f → is_odd_function (f ∘ f),
    ∀ f T, is_periodic_function f T → is_periodic_function (f ∘ f) T,
    ∀ f, is_decreasing_function f → is_decreasing_function (f ∘ f),
    ∀ f, (∃ x, (f ∘ f) x = x) → (∃ x, f x = x)
  ] in
  (card (filter id conditions) = 3) :=
by
  -- Proof would go here
  sorry

end number_of_correct_statements_l45_45366


namespace area_union_correct_l45_45967

-- Define the side length of the square and the radius of the circle
def side_length_square : ℝ := 8
def radius_circle : ℝ := 8

-- Area of the square
def area_square (s : ℝ) : ℝ := s^2

-- Area of the circle
def area_circle (r : ℝ) : ℝ := π * r^2

-- Area of the quarter-circle
def area_quarter_circle (r : ℝ) : ℝ := (1 / 4) * area_circle r

-- Area of the union of the square and the circle
noncomputable def area_union (s r : ℝ) : ℝ :=
  area_square s + (area_circle r - area_quarter_circle r)

-- Theorem statement
theorem area_union_correct :
  area_union side_length_square radius_circle = 64 + 48 * π :=
by
  sorry

end area_union_correct_l45_45967


namespace find_shirt_numbers_calculate_profit_l45_45598

def total_shirts_condition (x y : ℕ) : Prop := x + y = 200
def total_cost_condition (x y : ℕ) : Prop := 25 * x + 15 * y = 3500
def profit_calculation (x y : ℕ) : ℕ := (50 - 25) * x + (35 - 15) * y

theorem find_shirt_numbers (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  x = 50 ∧ y = 150 :=
sorry

theorem calculate_profit (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  profit_calculation x y = 4250 :=
sorry

end find_shirt_numbers_calculate_profit_l45_45598


namespace logical_contraposition_l45_45375

variable {Student : Type}
variables (P Q : Student → Prop)

theorem logical_contraposition :
  (∀ s : Student, P s → Q s) ↔ (∀ s : Student, ¬ Q s → ¬ P s) :=
begin
  split,
  { intros h s hq hp,
    exact hq (h s hp) },
  { intros h s hp,
    by_contradiction nq,
    exact h s nq hp }
end

end logical_contraposition_l45_45375


namespace length_of_XD_l45_45803

variables {A B C D X ω : Type} [Incircle ω A B C] [PointOnSegment D B C] [TangentPoint D ω] 
          [Intersection X (LineThrough A D) ω] [Radius ω = 1] 

theorem length_of_XD : AX / XD = 1 / 3 ∧ AX / BC = 1 / 10 → length XD = 3 * (sqrt 10) / 5 :=
by
  sorry

end length_of_XD_l45_45803


namespace solve_fractional_equation_l45_45853

theorem solve_fractional_equation (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ -6) :
    (x + 11) / (x - 4) = (x - 3) / (x + 6) ↔ x = -9 / 4 :=
by
  sorry

end solve_fractional_equation_l45_45853


namespace smallest_portion_is_two_l45_45034

theorem smallest_portion_is_two (a1 a2 a3 a4 a5 : ℕ) (d : ℕ) (h1 : a1 = a3 - 2 * d) (h2 : a2 = a3 - d) (h3 : a4 = a3 + d) (h4 : a5 = a3 + 2 * d) (h5 : a1 + a2 + a3 + a4 + a5 = 120) (h6 : a3 + a4 + a5 = 7 * (a1 + a2)) : a1 = 2 :=
by sorry

end smallest_portion_is_two_l45_45034


namespace candy_count_l45_45585

theorem candy_count :
  ∀ (initial : ℕ) (eaten : ℕ) (received : ℕ), 
    initial = 33 → eaten = 17 → received = 19 → (initial - eaten + received = 35) :=
by
  intros initial eaten received
  intros h_initial h_eaten h_received
  rw [h_initial, h_eaten, h_received]
  exact sorry

end candy_count_l45_45585


namespace find_constants_l45_45723

noncomputable def f (x m n : ℝ) := (m * x + 1) / (x + n)

theorem find_constants (m n : ℝ) (h_symm : ∀ x y, f x m n = y → f (4 - x) m n = 8 - y) : 
  m = 4 ∧ n = -2 := 
by
  sorry

end find_constants_l45_45723


namespace complement_U_A_l45_45821

def U := {1, 2, 3, 4, 5, 6, 7}
def A := {1, 2, 3, 4, 5, 6}

theorem complement_U_A : U \ A = {7} :=
by
  sorry

end complement_U_A_l45_45821


namespace perpendicular_vectors_vector_sum_norm_min_value_f_l45_45737

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3*x/2), Real.sin (3*x/2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x/2), -Real.sin (x/2))

noncomputable def f (x m : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - 2 * m * Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem perpendicular_vectors (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0 ↔ x = Real.pi / 4 := sorry

theorem vector_sum_norm (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 ≥ 1 ↔ 0 ≤ x ∧ x ≤ Real.pi / 3 := sorry

theorem min_value_f (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x m ≥ -2) ↔ m = Real.sqrt 2 / 2 := sorry

end perpendicular_vectors_vector_sum_norm_min_value_f_l45_45737


namespace closest_integer_to_cube_root_of_250_l45_45497

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45497


namespace correct_options_l45_45730

noncomputable def line_l (a : ℝ) : ℝ → ℝ → Prop := 
λ x y, a * x + (2 * a - 3) * y - 3 = 0

noncomputable def line_n (a : ℝ) : ℝ → ℝ → Prop := 
λ x y, (a + 2) * x + a * y - 6 = 0

def perpendicular (a b : ℝ) : Prop := a * b = -1

def parallel (a b : ℝ) : Prop := a = b

-- Prove the options
theorem correct_options (a : ℝ) :
  (a = 1/3 → perpendicular (a / (2 * a - 3)) ((a + 2) / a)) ∧
  (parallel (a / (2 * a - 3)) ((a + 2) / a) → distances a = (√10) / 2) ∧
  (max_distance_origin_to_l = √5) :=
begin
  split,
  { intro ha, sorry },
  split,
  { intro hparallel, sorry },
  { sorry }
end

end correct_options_l45_45730


namespace parabola_directrix_equation_l45_45235

theorem parabola_directrix_equation :
  ∀ (x y : ℝ),
  y = -4 * x^2 - 16 * x + 1 →
  ∃ d : ℝ, d = 273 / 16 ∧ y = d :=
by
  sorry

end parabola_directrix_equation_l45_45235


namespace number_of_dolls_of_jane_l45_45090

-- Given conditions
def total_dolls (J D : ℕ) := J + D = 32
def jill_has_more (J D : ℕ) := D = J + 6

-- Statement to prove
theorem number_of_dolls_of_jane (J D : ℕ) (h1 : total_dolls J D) (h2 : jill_has_more J D) : J = 13 :=
by
  sorry

end number_of_dolls_of_jane_l45_45090


namespace equilateral_triangle_in_ellipse_l45_45982

theorem equilateral_triangle_in_ellipse :
  (∃ (a b : ℝ) (AC F1F2 s : ℝ),
       (a ≠ 0 ∧ b ≠ 0) ∧
       (∀ x y : ℝ, (x/a)^2 + (y/b)^2 = 1) ∧
       (s > 0) ∧ 
       (F1F2 = 2) ∧
       (b = √3) ∧
       (vertex_B = (0, √3)) ∧ 
       (vertex_C = (s/2, y_M)) ∧
       (vertex_A = (-s/2, y_M)) ∧
       (2 * c = F1F2) ∧ 
       (c = 1) ∧
       let mid_AC = (0, y_M) in y_M = -√3/2 * (s-2)
       → 
       AC / F1F2 = 8/5) :=
begin
  sorry
end

end equilateral_triangle_in_ellipse_l45_45982


namespace closest_cube_root_of_250_l45_45471

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45471


namespace exists_xi_greater_than_1988_l45_45666

theorem exists_xi_greater_than_1988 :
  ∃ (x : Fin 29 → ℕ), 
    (∑ i, (x i)^2 = 29 * ∏ i, (x i)) ∧ 
    (∃ i, x i > 1988) :=
sorry

end exists_xi_greater_than_1988_l45_45666


namespace twenty_four_is_eighty_percent_of_what_number_l45_45092

theorem twenty_four_is_eighty_percent_of_what_number (x : ℝ) (hx : 24 = 0.8 * x) : x = 30 :=
  sorry

end twenty_four_is_eighty_percent_of_what_number_l45_45092


namespace part_a_part_b_l45_45801

-- Define the set H of numbers where the number of divisors of n divides n
def H : Set ℕ := {n | Nat.tau n ∣ n}

-- Part (a): Prove that for sufficiently large n, n! ∈ H
theorem part_a (n : ℕ) : ∃ N, ∀ n ≥ N, Nat.tau n ! ∣ n ! :=
  sorry

-- Part (b): Prove that H has density 0
theorem part_b : AsymptoticallyDensity0 H :=
  sorry

end part_a_part_b_l45_45801


namespace total_operations_in_one_hour_l45_45951

theorem total_operations_in_one_hour :
  let additions_per_second := 12000
  let multiplications_per_second := 8000
  (additions_per_second + multiplications_per_second) * 3600 = 72000000 :=
by
  sorry

end total_operations_in_one_hour_l45_45951


namespace min_S_l45_45386

variable {x y : ℝ}
def condition (x y : ℝ) : Prop := (4 * x^2 + 5 * x * y + 4 * y^2 = 5)
def S (x y : ℝ) : ℝ := x^2 + y^2
theorem min_S (hx : condition x y) : S x y = (10 / 13) :=
sorry

end min_S_l45_45386


namespace ordered_2011_tuples_mod_1000_l45_45802

theorem ordered_2011_tuples_mod_1000 :
  let N := {a : Fin 2011 → Fin (2011^2) // ∃ f : ℤ[X], degree f = 4019 
            ∧ (∀ n : ℤ, is_int_eval f n) 
            ∧ (∀ i : Fin 2011, 2011^2 ∣ (eval (f i) - a i))
            ∧ (∀ n : ℤ, 2011^2 ∣ (eval (f (n + 2011)) - eval (f n))) } in
  (N.card % 1000 = 281) :=
sorry

end ordered_2011_tuples_mod_1000_l45_45802


namespace closest_cube_root_of_250_l45_45476

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45476


namespace hyperbola_equation_l45_45276

-- Conditions
def a : ℝ := 1
def b : ℝ := sqrt(3)
def c : ℝ := 2
def hyperbola_eq (x y : ℝ) := x^2 - (y^2 / 3) = 1
def is_focus (x : ℝ) := x = 2
def asymptotes (x y : ℝ) := y = sqrt(3) * x ∨ y = -sqrt(3) * x

-- Correct answer
theorem hyperbola_equation (x y : ℝ) : hyperbola_eq x y = (x^2 - (y^2 / 3) = 1) :=
by
  sorry

end hyperbola_equation_l45_45276


namespace scaling_transformation_of_sine_curve_l45_45223

theorem scaling_transformation_of_sine_curve (x y x' y' : ℝ)
  (h_scaling : x' = 2 * x ∧ y' = 3 * y) :
  (y = sin (2 * x)) → (y' = 3 * sin x') :=
by
  sorry

end scaling_transformation_of_sine_curve_l45_45223


namespace distinct_nat_sum_one_l45_45104

theorem distinct_nat_sum_one (n : ℕ) :
  (∃ (x : fin n → ℕ), (function.injective x)
  ∧ ((finset.univ.sum (λ (i : fin n), (1 / (x i : ℚ))) = 1)) ) ↔ (3 ≤ n) :=
sorry

end distinct_nat_sum_one_l45_45104


namespace stickers_per_student_l45_45011

theorem stickers_per_student 
  (gold_stickers : ℕ) 
  (silver_stickers : ℕ) 
  (bronze_stickers : ℕ) 
  (students : ℕ)
  (h1 : gold_stickers = 50)
  (h2 : silver_stickers = 2 * gold_stickers)
  (h3 : bronze_stickers = silver_stickers - 20)
  (h4 : students = 5) : 
  (gold_stickers + silver_stickers + bronze_stickers) / students = 46 :=
by
  sorry

end stickers_per_student_l45_45011


namespace circle_square_area_difference_l45_45868

theorem circle_square_area_difference :
  let d := 10
  let r := 6
  let s := d / Real.sqrt 2
  let A_square := s^2
  let A_circle := Real.pi * r^2
  A_circle - A_square = 36 * Real.pi - 50 :=
by
  let d := 10
  let r := 6
  let s := d / Real.sqrt 2
  let A_square := s ^ 2
  let A_circle := Real.pi * r^2
  have h1 : s = 10 / Real.sqrt 2 := rfl
  have h2 : s ^ 2 = 50 := by 
    rw [h1, pow_two]
    norm_num
    rw [div_pow, Real.sqrt_mul_self, div_self]
    norm_num
  have h3 : A_circle = 36 * Real.pi := rfl
  show A_circle - A_square = 36 * Real.pi - 50
  rw [<-h2, <-h3]
  sorry

end circle_square_area_difference_l45_45868


namespace question_l45_45420

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 1

-- Given three real numbers a, b, c such that they are the roots of P(x) and a > b > c
variables (a b c : ℝ)
hypothesis h1 : P a = 0
hypothesis h2 : P b = 0
hypothesis h3 : P c = 0
hypothesis h4 : a > b
hypothesis h5 : b > c

-- Define the expression we are interested in
def A : ℝ := a^2 * b + b^2 * c + c^2 * a

-- Prove that A = 4
theorem question : A = 4 := sorry

end question_l45_45420


namespace compare_f_ln_l45_45001

variable {f : ℝ → ℝ}

theorem compare_f_ln (h : ∀ x : ℝ, deriv f x > f x) : 3 * f (Real.log 2) < 2 * f (Real.log 3) :=
by
  sorry

end compare_f_ln_l45_45001


namespace triangle_BED_area_l45_45775

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_BED_area 
  {A B C D E M : ℝ × ℝ} 
  (h_ABC_obtuse_iso : triangle_area A B C = 30) 
  (h_angle_C_gt_90 : ∃ θ : ℝ, θ > (real.pi / 2) ∧ triangle.angle C A B = θ)
  (h_M_mid_AC : M.1 = (A.1 + C.1) / 2 ∧ M.2 = (A.2 + C.2) / 2)
  (h_MD_perp_BC : D.1 = B.1 ∧ D.2 = C.2)
  (h_EC_perp_BC : E.1 = A.1 ∧ E.2 = (A.2 + B.2) / 2)
  : triangle_area B E D = 15 := 
sorry

end triangle_BED_area_l45_45775


namespace monotonically_increasing_e_l45_45858

theorem monotonically_increasing_e {a : ℝ} :
  (∀ x ≥ 0, (λ x : ℝ, Real.exp x - 1 - x - a * x^2).deriv x ≥ 0) ↔ (a ≤ 1 / 2) :=
begin
  sorry
end

end monotonically_increasing_e_l45_45858


namespace part1_magical_number_pair_part2_find_t_part3_minimum_value_l45_45385

-- Part 1: Prove that (4/3, 4) is a magical number pair
theorem part1_magical_number_pair:
  (1 / (4 / 3) + 1 / 4 = 1) :=
by
  calc (1 / (4 / 3) + 1 / 4)
    = (3 / 4 + 1 / 4) : by sorry
    ... = (1) : by sorry

-- Part 2: Prove that t = ±√15 makes (5-t, 5+t) a magical number pair
theorem part2_find_t (t : ℝ) :
  (1 / (5 - t) + 1 / (5 + t) = 1) ↔ (t = √15 ∨ t = -√15) :=
by
  calc (1 / (5 - t) + 1 / (5 + t))
    = (5 + t + 5 - t) / ((5 - t) * (5 + t)) : by sorry
    ... = 1 : by sorry
    ... ↔ (t = √15 ∨ t = -√15) : by sorry

-- Part 3: Prove the minimum value of the given algebraic expression is -36
theorem part3_minimum_value (a b c m n : ℝ)
  (h1 : 1 / m + 1 / n = 1)
  (h2 : a = b + m)
  (h3 : b = c + n ) :
  (a - c) ^ 2 - 12 * (a - b) * (b - c) = -36 :=
by
  have h_m : m = a - b, from sorry,
  have h_n : n = b - c, from sorry,
  calc (a - c) ^ 2 - 12 * (a - b) * (b - c)
    = ((a - b) + (b - c)) ^ 2 - 12 * (a - b) * (b - c) : by sorry
    ... = (a - c) ^ 2 - 12 * (a - (b + m)) * (b - (c + n)) : by sorry
    ... = ((a - c - 6) ^ 2 - 36) : by sorry
    ... = -36 : by sorry

end part1_magical_number_pair_part2_find_t_part3_minimum_value_l45_45385


namespace find_a5_l45_45328

variable {a : ℕ → ℝ}

-- Condition 1: {a_n} is an arithmetic sequence
def arithmetic_sequence (a: ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition 2: a1 + a9 = 10
axiom a1_a9_sum : a 1 + a 9 = 10

theorem find_a5 (h_arith : arithmetic_sequence a) : a 5 = 5 :=
by {
  sorry
}

end find_a5_l45_45328


namespace division_problem_solution_l45_45462

theorem division_problem_solution (x : ℝ) (h : (2.25 / x) * 12 = 9) : x = 3 :=
sorry

end division_problem_solution_l45_45462


namespace possible_values_of_ab_plus_ac_plus_bc_l45_45355

theorem possible_values_of_ab_plus_ac_plus_bc (a b c : ℝ) (h : a + b + c = 1) : 
  ∃ t : set ℝ, t = {x | x ≤ 1/3} ∧ (ab + ac + bc ∈ t) :=
by
  sorry

end possible_values_of_ab_plus_ac_plus_bc_l45_45355


namespace sqrt_product_l45_45640

theorem sqrt_product (a b c : ℝ) (h1 : a = 128) (h2 : b = 50) (h3 : c = 27) (sqrt128 : Real.sqrt a = 8 * Real.sqrt 2)
  (sqrt50 : Real.sqrt b = 5 * Real.sqrt 2) (cbrt27 : Real.cbrt c = 3) : 
  Real.sqrt a * Real.sqrt b * Real.cbrt c = 240 :=
by
  sorry

end sqrt_product_l45_45640


namespace final_position_west_400m_calories_consumed_l45_45652

def movement_sequence := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

/-- Prove that Xiaozhao ends up 400 meters to the west of the bus stop --/
theorem final_position_west_400m : 
  (movement_sequence.sum) = -400 := 
by sorry

/-- Prove that Xiaozhao consumes 44.8 thousand calories in the morning --/
theorem calories_consumed : 
  let total_distance_walked := (movement_sequence.map Int.natAbs).sum in
  let distance_in_km := total_distance_walked / 1000 in
  (distance_in_km * 7000) = 44800 := 
by sorry

end final_position_west_400m_calories_consumed_l45_45652


namespace binomial_12_3_eq_220_l45_45210

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l45_45210


namespace find_t_l45_45657

-- Define the logarithm base 3 function
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Given Condition
def condition (t : ℝ) : Prop := 4 * log_base_3 t = log_base_3 (4 * t) + 2

-- Theorem stating if the given condition holds, then t must be 6
theorem find_t (t : ℝ) (ht : condition t) : t = 6 := 
by
  sorry

end find_t_l45_45657


namespace sum_sequence_eq_980_l45_45229

theorem sum_sequence_eq_980 :
  let sequence := λ (n : ℕ), if n % 2 = 0 then 1940 - 20 * n else -(1940 - 20 * n),
      n_terms := 98,
      sum_sequence := ∑ i in finset.range n_terms, sequence i
  in sum_sequence = 980 := by sorry

end sum_sequence_eq_980_l45_45229


namespace blue_bird_high_school_team_arrangement_l45_45859

theorem blue_bird_high_school_team_arrangement : 
  let girls := 2
  let boys := 3
  let girls_permutations := Nat.factorial girls
  let boys_permutations := Nat.factorial boys
  girls_permutations * boys_permutations = 12 := by
  sorry

end blue_bird_high_school_team_arrangement_l45_45859


namespace product_of_roots_of_cubic_eq_l45_45212

theorem product_of_roots_of_cubic_eq :
  let a := 1
  let b := -9
  let c := 27
  let d := -64
  (h : a ≠ 0) → 
  let poly := λ (x : ℂ), a * x^3 + b * x^2 + c * x + d
  ∀ roots, (∀ r ∈ roots, poly r = 0) → Multiset.prod roots = 64 :=
by
  have h1 : 64 = -(d / a) := by 
    sorry -- Details of the derivation using Vieta's formulas
  sorry -- Proof of the theorem

end product_of_roots_of_cubic_eq_l45_45212


namespace arithmetic_sequence_fifth_term_l45_45711

theorem arithmetic_sequence_fifth_term (x : ℝ) (a₂ : ℝ := x) (a₃ : ℝ := 3) 
    (a₁ : ℝ := -1) (h₁ : a₂ = a₁ + (1*(x + 1))) (h₂ : a₃ = a₁ + 2*(x + 1)) : 
    a₁ + 4*(a₃ - a₂ + 1) = 7 :=
by
  sorry

end arithmetic_sequence_fifth_term_l45_45711


namespace roof_friendly_min_pairs_roof_friendly_max_pairs_l45_45349

def roof_friendly (n : ℕ) (buildings : list ℕ) (i j : ℕ) : Prop :=
  1 ≤ i ∧ i < j ∧ j ≤ n ∧
  buildings.nth (i - 1) < buildings.nth (j - 1) ∧
  ∀ k, i < k ∧ k < j → (buildings.nth (k - 1) < buildings.nth (i - 1) ∧ buildings.nth (k - 1) < buildings.nth (j - 1))

theorem roof_friendly_min_pairs (n : ℕ) (h : n ≥ 2) : 
  ∃ b : list ℕ, (∀ i, i < n → roof_friendly n b i (i + 1)) ∧ (∀ i j, roof_friendly n b i j → j = i + 1 ∨ i + 1 = 1) → 
  List.length (list.filter (λ p, roof_friendly n b (p.1) (p.2)) (list.zip (list.range (n)) (list.range 1 n))) = n - 1 :=
sorry

theorem roof_friendly_max_pairs (n : ℕ) (h : n ≥ 2) : 
  ∃ b : list ℕ, ((list.zip (list.range n) (list.tail (list.range (n+1)))).length + 
  List.length (list.filter (λ p, ∃ k, roof_friendly n b k (p.2)) (list.zip (list.repeat n (n - 2)) (list.range 1 (n-1))))) = 2n - 3 :=
sorry

end roof_friendly_min_pairs_roof_friendly_max_pairs_l45_45349


namespace tangent_identity_l45_45273

theorem tangent_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2)
  = ((3 * x - x^3) / (1 - 3 * x^2)) * ((3 * y - y^3) / (1 - 3 * y^2)) * ((3 * z - z^3) / (1 - 3 * z^2)) :=
sorry

end tangent_identity_l45_45273


namespace polynomial_area_inequality_l45_45809

variable {R : Type} [Real]

/-- Given a polynomial of degree higher than 2 with all real roots, where 
  f(x) > 0 for all -1 < x < 1, and f(-1) = f(1) = 0, prove that 
  A = ∫_{-1}^{1} f(x) dx ≥ 2/3 * T 
  where T = 2 * f'(1) * f'(-1) / (f'(1) - f'(-1)) -/
theorem polynomial_area_inequality (f : R → R) [f_polynomial : is_polynomial f]
  (deg_f : degree f > 2)
  (roots_real : ∀ x, is_root f x → real.is_real x)
  (f_positive_in_range : ∀ x, -1 < x ∧ x < 1 → f x > 0)
  (f_neg_one_zero : f (-1) = 0)
  (f_one_zero : f 1 = 0) :
  let A := ∫ x in -1..1, f x d x,
      T := 2 * f'(-1) * f'(1) / (f'(1) - f'(-1)) in
  A ≥ 2/3 * T :=
by { sorry }

end polynomial_area_inequality_l45_45809


namespace geom_sequence_ratio_and_fifth_term_l45_45954

theorem geom_sequence_ratio_and_fifth_term 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 10) 
  (h₂ : a₂ = -15) 
  (h₃ : a₃ = 22.5) 
  (h₄ : a₄ = -33.75) : 
  ∃ r a₅, r = -1.5 ∧ a₅ = 50.625 ∧ (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ (a₄ = r * a₃) ∧ (a₅ = r * a₄) := 
by
  sorry

end geom_sequence_ratio_and_fifth_term_l45_45954


namespace sum_pascal_triangle_2018_row_l45_45020

def a : ℕ → ℕ → ℕ 
| 0, 0 := 1
| n, 0 := 1
| n, k := if k = n then 1 else if k = 0 then 1 else a (n - 1) k - a (n - 1) (k - 1)

def sum_row (n : ℕ) : ℕ := (List.range (n + 1)).sum (λ k => a n k)

theorem sum_pascal_triangle_2018_row : sum_row 2018 = 2 := 
by
  -- proof to be completed
  sorry

end sum_pascal_triangle_2018_row_l45_45020


namespace arithmetic_mean_eq_2_l45_45641

theorem arithmetic_mean_eq_2 (a x : ℝ) (hx: x ≠ 0) :
  (1/2) * (((2 * x + a) / x) + ((2 * x - a) / x)) = 2 :=
by
  sorry

end arithmetic_mean_eq_2_l45_45641


namespace minimum_n_days_correct_l45_45620

noncomputable def minimum_n_days
  (rain_count : ℕ)
  (afternoon_sunny_count : ℕ)
  (morning_sunny_count : ℕ)
  (afternoon_not_raining_implies_morning_sunny : Prop) : ℕ :=
  let n := 9 in
  n

theorem minimum_n_days_correct :
  ∃ n : ℕ,
    (rain_count = 7) ∧
    (afternoon_sunny_count = 5) ∧
    (morning_sunny_count = 6) ∧
    (∀ d : ℕ, d ≤ n → (afternoon_not_raining_implies_morning_sunny)) ∧
    (n = 9) :=
begin
  use 9,
  split,
  { exact rfl, }, -- rain_count = 7
  split,
  { exact rfl, }, -- afternoon_sunny_count = 5
  split,
  { exact rfl, }, -- morning_sunny_count = 6
  split,
  { intros d hd,
    sorry, -- afternoon_not_raining_implies_morning_sunny
  },
  { exact rfl, }, -- n = 9
end

end minimum_n_days_correct_l45_45620


namespace range_of_a_l45_45709

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a * x > 0) → a < 1 :=
by
  sorry

end range_of_a_l45_45709


namespace new_percentage_girls_is_32_l45_45593

-- Define the initial conditions
def initial_total_students : ℕ := 20
def initial_percentage_girls : ℝ := 0.4
def new_boys : ℕ := 5

-- Define the quantities derived from initial conditions
def initial_number_girls : ℕ := (initial_total_students : ℝ) * initial_percentage_girls
def initial_number_boys : ℕ := initial_total_students - initial_number_girls

-- Define the new state of the classroom after 5 boys join
def new_total_students : ℕ := initial_total_students + new_boys
def new_number_boys : ℕ := initial_number_boys + new_boys

-- Prove that the new percentage of girls is 32% or 0.32
theorem new_percentage_girls_is_32 : (initial_number_girls : ℝ) / (new_total_students : ℝ) = 0.32 := by
  sorry

end new_percentage_girls_is_32_l45_45593


namespace closest_integer_to_cube_root_of_250_l45_45484

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45484


namespace james_added_8_fish_l45_45896

theorem james_added_8_fish
  (initial_fish : ℕ := 60)
  (fish_eaten_per_day : ℕ := 2)
  (total_days_with_worm : ℕ := 21)
  (fish_remaining_when_discovered : ℕ := 26) :
  ∃ (additional_fish : ℕ), additional_fish = 8 :=
by
  let total_fish_eaten := total_days_with_worm * fish_eaten_per_day
  let fish_remaining_without_addition := initial_fish - total_fish_eaten
  let additional_fish := fish_remaining_when_discovered - fish_remaining_without_addition
  exact ⟨additional_fish, sorry⟩

end james_added_8_fish_l45_45896


namespace students_remaining_l45_45463

theorem students_remaining (weight_left : ℝ) (weight_increase : ℝ) 
                           (new_avg_weight : ℝ) (n : ℝ) : 
  weight_left = 45 ∧ weight_increase = 0.2 ∧ new_avg_weight = 57 → n = 59 :=
by
  -- Given conditions
  intro h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [h4, h3, h1],
  -- We would proceed with the formal proof here
  -- For now, we skip solving using 'sorry'
  sorry

end students_remaining_l45_45463


namespace star_pentagon_area_l45_45632

noncomputable def area_of_star_pentagon (p : ℝ) : ℝ :=
  2 * real.sqrt (85 - 38 * real.sqrt 5) * p^2

theorem star_pentagon_area (p : ℝ) : area_of_star_pentagon(p) = 2 * real.sqrt (85 - 38 * real.sqrt 5) * p^2 :=
by
  sorry

end star_pentagon_area_l45_45632


namespace percentage_of_juniors_l45_45768

variable (F S J Sr : ℕ)

theorem percentage_of_juniors 
  (h_total : F + S + J + Sr = 800)
  (h_seniors : Sr = 160)
  (h_sophomores : S = 0.25 * 800)
  (h_freshmen : F = S + 32)
  : (J / 800) * 100 = 26 :=
by
  sorry

end percentage_of_juniors_l45_45768


namespace meiosis_fertilization_stability_l45_45062

def maintains_chromosome_stability (x : String) : Prop :=
  x = "Meiosis and Fertilization"

theorem meiosis_fertilization_stability :
  maintains_chromosome_stability "Meiosis and Fertilization" :=
by
  sorry

end meiosis_fertilization_stability_l45_45062


namespace solve_for_x_l45_45074

theorem solve_for_x (x : ℝ) (h : 3 / (x + 2) = 2 / (x - 1)) : x = 7 :=
sorry

end solve_for_x_l45_45074


namespace proof_of_min_marked_cells_l45_45350

variable (n : ℕ) (a b c : ℕ) (board : Fin (2 * n + 1) → Fin (2 * n + 1) → Bool)

def white (i j : Fin (2 * n + 1)) : Prop := board i j = false
def black (i j : Fin (2 * n + 1)) : Prop := board i j = true

def count_color (color : Bool) : ℕ :=
  Fin.sum (Fin (2 * n + 1)) (fun i => Fin.sum (Fin (2 * n + 1)) (fun j => if board i j = color then 1 else 0))

theorem proof_of_min_marked_cells :
  a = count_color true →
  b = count_color false →
  c = Fin.sum (Fin (2 * n + 1)) (fun i =>
    Fin.sum (Fin (2 * n + 1)) (fun j =>
      if ((Fin.sum (Fin (2 * n + 1)) (fun k => if white i k then 1 else 0)) < (Fin.sum (Fin (2 * n + 1)) (fun k => if black i k then 1 else 0)) ∧
          Fin.sum (Fin (2 * n + 1)) (fun k => if white k j then 1 else 0) < (Fin.sum (Fin (2 * n + 1)) (fun k => if black k j then 1 else 0))) then 1 else 0))
  → c ≥ (1/2) * min a b :=
sorry

end proof_of_min_marked_cells_l45_45350


namespace prob_red_ball_is_three_fifths_l45_45774

def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3
def red_balls : ℕ := total_balls - (yellow_balls + green_balls)
def total_probability : ℚ := 1
def probability_of_red_ball : ℚ := red_balls / total_balls

theorem prob_red_ball_is_three_fifths :
  probability_of_red_ball = 3 / 5 :=
begin
  sorry
end

end prob_red_ball_is_three_fifths_l45_45774


namespace simplify_expression_l45_45845

theorem simplify_expression (x : ℤ) : (x + 15) + (150x + 20) = 151x + 35 := 
by
  sorry

end simplify_expression_l45_45845


namespace nine_cubed_expansion_l45_45097

theorem nine_cubed_expansion : 9^3 + 3 * 9^2 + 3 * 9 + 1 = 1000 := 
by 
  sorry

end nine_cubed_expansion_l45_45097


namespace find_absolut_slope_l45_45087

variable (slope : ℝ)

def circle(center : ℝ × ℝ, radius : ℝ) := 
  ∀ (x y : ℝ), (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2

def line_through (x₀ : ℝ) (y₀ : ℝ) (slope : ℝ) :=
  ∀ (x y : ℝ), y = slope * (x - x₀) + y₀

def bisects_area (c : (ℝ × ℝ) × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (p q : ℝ), c.2 = p ∨ c.2 = q ∨ l c.1.1 c.1.2

theorem find_absolut_slope :
  bisects_area ((0, 0), 4) (line_through 4 0 slope) ∧
  bisects_area ((4, 0), 3) (line_through 4 0 slope) ∧
  bisects_area ((2, 3), 2) (line_through 4 0 slope) →
  |slope| = 1.5 :=
sorry

end find_absolut_slope_l45_45087


namespace Rockham_Soccer_League_members_l45_45327

theorem Rockham_Soccer_League_members (sock_cost tshirt_cost cap_cost total_cost members : ℕ) (h1 : sock_cost = 6) (h2 : tshirt_cost = sock_cost + 10) (h3 : cap_cost = 3) (h4 : total_cost = 4620) (h5 : total_cost = 50 * members) : members = 92 :=
by
  sorry

end Rockham_Soccer_League_members_l45_45327


namespace closest_cube_root_of_250_l45_45475

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45475


namespace largest_divisor_of_n_l45_45574

theorem largest_divisor_of_n (n : ℕ) (hn : 0 < n) (h : 50 ∣ n^2) : 5 ∣ n :=
sorry

end largest_divisor_of_n_l45_45574


namespace acute_triangle_l45_45321

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ area, 
    (area = (1 / 2) * a * b * Real.sin C) ∧
    (a / Real.sin A = 2 * c / Real.sqrt 3) ∧
    (c = Real.sqrt 7) ∧
    (area = (3 * Real.sqrt 3) / 2)

theorem acute_triangle (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) :
  C = 60 ∧ a^2 + b^2 = 13 :=
by
  obtain ⟨_, h_area, h_sine, h_c, h_area_eq⟩ := h
  sorry

end acute_triangle_l45_45321


namespace find_an_l45_45714

-- Definitions based on conditions
def geometric_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) - a n = q^(n)
def a1 := 1
def a2 := 2
def a3 := 5

-- Lean statement to prove the expression for a_n
theorem find_an (a : ℕ → ℝ) 
  (h1 : a 1 = a1)
  (h2 : a 2 = a2)
  (h3 : a 3 = a3)
  (h_geom : geometric_seq (λ n, a (n + 1) - a n)) : 
  ∀ n : ℕ, a n = (3^(n-1)) / 2 + 1 / 2 :=
by sorry

end find_an_l45_45714


namespace slope_angle_45_degrees_l45_45675

noncomputable def slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
(y₂ - y₁) / (x₂ - x₁)

noncomputable def slope_angle (m : ℝ) : ℝ :=
Real.arctan m

theorem slope_angle_45_degrees :
  slope_angle (slope (-2) 1 1 4) = Real.pi / 4 :=
by
  sorry

end slope_angle_45_degrees_l45_45675


namespace sum_of_spheres_l45_45082

-- Definition of triangular number
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem statement
theorem sum_of_spheres (n : ℕ) : (finset.range (n + 1)).sum (λ i, 3^i) = (3^(n+1) - 1) / 2 := 
sorry

end sum_of_spheres_l45_45082


namespace geometric_sequence_conditions_l45_45998

-- Define the geometric sequence
def geom_seq (a : ℝ) (n : ℕ) : ℝ :=
  a * (a - 1) ^ n

-- Define the sum of the first n terms of the geometric series
def S_n (a : ℝ) (n : ℕ) : ℝ :=
  if a = 2 then
    2 * n
  else if a ≠ 0 ∧ a ≠ 1 ∧ a ≠ 2 then
    a * ((a - 1) ^ n - 1) / (a - 2)
  else
    0  -- invalid cases

-- Define the condition to check if S_1, S_2, S_3 form an arithmetic sequence
def arithmetic_sequence (a : ℝ) : Prop :=
  S_n a 1 + S_n a 2 = 2 * S_n a 3

-- The main theorem to prove
theorem geometric_sequence_conditions (a : ℝ) (n : ℕ) :
  (S_n a n = if a = 2 then 2 * n else if a ≠ 0 ∧ a ≠ 1 ∧ a ≠ 2 then a * ((a - 1) ^ n - 1) / (a - 2) else 0) ∧
  (arithmetic_sequence a ↔ a = 1/2) :=
by 
  sorry

end geometric_sequence_conditions_l45_45998


namespace find_smallest_number_l45_45105

theorem find_smallest_number :
  let A := 111111₂
  let B := 150₆
  let C := 1000₄
  let D := 101₈
  A < B ∧ A < C ∧ A < D :=
by
  -- Define the numbers in decimal
  let A := 63
  let B := 66
  let C := 64
  let D := 65

  -- Prove that A is the smallest
  split
  · exact Nat.lt_trans (by norm_num) (by norm_num)
  · split
    · exact Nat.lt_trans (by norm_num) (by norm_num)
    · exact Nat.lt_trans (by norm_num) (by norm_num)

end find_smallest_number_l45_45105


namespace area_ratio_l45_45346

variables (A B C D M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M]
variables (O1 O2 O3 O4 : Type) [MetricSpace O1] [MetricSpace O2] [MetricSpace O3] [MetricSpace O4]

-- Assume M is the intersection of diagonals of the convex quadrilateral ABCD
axiom M_is_intersection : Intersection (Diagonals (ConvexQuadrilateral A B C D)) = M

-- Assume that the angle ∠AMB = 60 degrees
axiom angle_AMB_60 : MeasureAngle A M B = 60

-- Assume O1, O2, O3, O4 are the circumcenters of triangles ABM, BCM, CDM, and DAM respectively
axiom O1_is_circumcenter : Circumcenter (Triangle A B M) = O1
axiom O2_is_circumcenter : Circumcenter (Triangle B C M) = O2
axiom O3_is_circumcenter : Circumcenter (Triangle C D M) = O3
axiom O4_is_circumcenter : Circumcenter (Triangle D A M) = O4

-- The goal is to prove the ratio of the areas
theorem area_ratio : 
  (Area (ConvexQuadrilateral A B C D)) / (Area (Quadrilateral O1 O2 O3 O4)) = 3 / 2 :=
sorry

end area_ratio_l45_45346


namespace product_base5_l45_45634

theorem product_base5 : 
  let a := 2 * 5^3 + 3 * 5^2 + 1 * 5^1 + 4 * 5^0
  let b := 2 * 5^1 + 3 * 5^0
  let product_base10 := a * b
  let base10_to_base5 := [6, 8, 3, 3, 2]  -- digits in base 5 from the conversion
  product_base10 = 4342 ∧ 4342_base5 == 68332_5 := 
by
  sorry

end product_base5_l45_45634


namespace ratio_books_donated_l45_45007

theorem ratio_books_donated (initial_books: ℕ) (books_given_nephew: ℕ) (books_after_nephew: ℕ) 
  (books_final: ℕ) (books_purchased: ℕ) (books_donated_library: ℕ) (ratio: ℕ):
    initial_books = 40 → 
    books_given_nephew = initial_books / 4 → 
    books_after_nephew = initial_books - books_given_nephew →
    books_final = 23 →
    books_purchased = 3 →
    books_donated_library = books_after_nephew - (books_final - books_purchased) →
    ratio = books_donated_library / books_after_nephew →
    ratio = 1 / 3 := sorry

end ratio_books_donated_l45_45007


namespace shaded_region_area_l45_45214

-- Define points and dimensions for the rectangles and their relative positions.
structure Point where
  x : ℝ
  y : ℝ

structure Rectangle where
  bottom_left : Point
  width : ℝ
  height : ℝ

def ABCD := Rectangle.mk (Point.mk 0 0) 4 4
def BCEF := Rectangle.mk (Point.mk 4 0) 12 12
def FGHI := Rectangle.mk (Point.mk 16 0) 16 16

-- Define points based on the problem statement.
def A := ABCD.bottom_left
def B := Point.mk 4 0
def C := Point.mk 16 0
def F := Point.mk 16 12
def I := Point.mk 16 16

-- Prove the area of the shaded region (triangle CFI) is 16 cm^2.
theorem shaded_region_area : 
  let C := Point.mk 16 0
  let F := Point.mk 16 12
  let I := Point.mk 16 16
  (½ * (C.x - F.x) * (I.y - F.y) = 16) :=
by {
  let C := Point.mk 16 0;
  let F := Point.mk 16 12;
  let I := Point.mk 16 16;
  sorry
}

end shaded_region_area_l45_45214


namespace F_maps_squares_to_squares_l45_45872

noncomputable def F : ℝ² → ℝ² := sorry -- the specific definition of the function F is irrelevant for the statement

-- Define a rectangle in 2D space
structure Rectangle :=
(A B C D : ℝ²)
(is_rectangle : (A.1 = C.1 ∧ B.1 = D.1 ∧ A.2 = B.2 ∧ C.2 = D.2) ∨ 
                (A.1 = B.1 ∧ C.1 = D.1 ∧ A.2 = D.2 ∧ B.2 = C.2))

-- Assume F maps rectangles to rectangles
axiom F_maps_rectangles (R : Rectangle) : ∃ S : Rectangle, ∀ p ∈ {R.A, R.B, R.C, R.D}, F p ∈ {S.A, S.B, S.C, S.D}

-- Define a square in 2D space
structure Square :=
(U V W X : ℝ²)
(is_square : U.1 = W.1 ∧ V.1 = X.1 ∧ U.2 = V.2 ∧ W.2 = X.2 ∧ (V.1 - U.1) = (U.2 - V.2))

-- The theorem statement
theorem F_maps_squares_to_squares (S : Square) : ∃ T : Square, ∀ p ∈ {S.U, S.V, S.W, S.X}, F p ∈ {T.U, T.V, T.W, T.X} :=
sorry

end F_maps_squares_to_squares_l45_45872


namespace unique_elements_set_l45_45784

theorem unique_elements_set (x : ℝ) : x ≠ 3 ∧ x ≠ -1 ∧ x ≠ 0 ↔ 3 ≠ x ∧ x ≠ (x ^ 2 - 2 * x) ∧ (x ^ 2 - 2 * x) ≠ 3 := by
  sorry

end unique_elements_set_l45_45784


namespace series_diverges_l45_45339

noncomputable def series (n : ℕ) := ∑ k in (Finset.range n), (Real.sin (k * Real.pi / 3))

theorem series_diverges : ¬(∃ l : ℝ, Filter.Tendsto (λ n, series n) Filter.atTop (Filter.pure l)) :=
by
  sorry

end series_diverges_l45_45339


namespace can_inscribe_circle_in_ABCD_l45_45448

noncomputable def quad_inscribed_circle_radius (R r : ℝ) (h : r < R) : ℝ :=
  (2 * r * R) / (r + R)

theorem can_inscribe_circle_in_ABCD {R r : ℝ} (h : r < R)
  (is_tangential: ∀ {A B C D}, circle_touches_externally A B R C D r) :
  ∃ (radius : ℝ), radius = quad_inscribed_circle_radius R r h :=
by
  -- Proof needs to be filled in here
  sorry

end can_inscribe_circle_in_ABCD_l45_45448


namespace max_expression_value_l45_45881

-- Define the condition for the numbers being a permutation of 1, 5, 2, 4
def is_permutation (l : List ℕ) : Prop := 
  List.perm l [1, 5, 2, 4]

-- Define the expression to be maximized
def expression (x y z w : ℕ) : ℕ :=
  x * y - y * z + z * w - w * x

-- The main theorem we want to prove
theorem max_expression_value :
  ∃ (x y z w : ℕ), is_permutation [x, y, z, w] ∧ expression x y z w = 9 :=
by
  sorry

end max_expression_value_l45_45881


namespace sum_divisors_two_three_l45_45891

theorem sum_divisors_two_three (i j : ℕ) (n := 2^i * 3^j)
  (h_sum : ∑ d in divisors n, d = 960) : i + j = 5 :=
by
  sorry

end sum_divisors_two_three_l45_45891


namespace at_least_one_hit_l45_45392

noncomputable def hitting_event (i : ℕ) : Prop :=
  i = 1 ∨ i = 2 ∨ i = 3

noncomputable def certain_event : Prop :=
  ∃ (n : ℕ), hitting_event(n) ∨ n = 0

axiom mutually_exclusive_hitting_events :
  ∀ (i j : ℕ), i ≠ j → (hitting_event(i) → ¬hitting_event(j))

axiom certain_event_defined :
  certain_event = true

theorem at_least_one_hit (A A_0 A_1 A_2 A_3 : Prop) :
  (A = A_1 ∨ A_2 ∨ A_3) →
  (A_0 ∨ A_1 ∨ A_2 ∨ A_3 = certain_event) →
  (mutually_exclusive_hitting_events 1 2 ∧
  mutually_exclusive_hitting_events 1 3 ∧
  mutually_exclusive_hitting_events 2 3) →
  (A = hitting_event(1) ∨ hitting_event(2) ∨ hitting_event(3)) :=
by
  sorry

end at_least_one_hit_l45_45392


namespace best_approximation_for_avg_square_feet_per_person_l45_45063

-- Define conditions
def population : ℕ := 226504825
def area_square_miles : ℕ := 3615122
def square_feet_per_square_mile : ℕ := (5280 ^ 2)
def options : list ℕ := [5000, 10000, 50000, 100000, 500000]

-- Define the correct approximation for average square feet per person
def avg_square_feet_per_person : ℕ :=
  area_square_miles * square_feet_per_square_mile / population

-- State the problem
theorem best_approximation_for_avg_square_feet_per_person :
  ∃ x ∈ options, abs (avg_square_feet_per_person - x) = min (list.map (λ y, abs (avg_square_feet_per_person - y)) options) :=
sorry

end best_approximation_for_avg_square_feet_per_person_l45_45063


namespace percentage_of_cars_with_no_features_l45_45437

theorem percentage_of_cars_with_no_features (N S W R SW SR WR SWR : ℕ)
  (hN : N = 120)
  (hS : S = 70)
  (hW : W = 40)
  (hR : R = 30)
  (hSW : SW = 20)
  (hSR : SR = 15)
  (hWR : WR = 10)
  (hSWR : SWR = 5) :
  (120 - (S + W + R - SW - SR - WR + SWR)) / (N : ℝ) * 100 = 16.67 :=
by
  sorry

end percentage_of_cars_with_no_features_l45_45437


namespace closest_integer_to_cube_root_of_250_l45_45509

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45509


namespace find_unsuitable_activity_l45_45159

-- Definitions based on the conditions
def suitable_for_questionnaire (activity : String) : Prop :=
  activity = "D: The radiation produced by various mobile phones during use"

-- Question transformed into a statement to prove in Lean
theorem find_unsuitable_activity :
  suitable_for_questionnaire "D: The radiation produced by various mobile phones during use" :=
by
  sorry

end find_unsuitable_activity_l45_45159


namespace binom_12_3_eq_220_l45_45188

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l45_45188


namespace math_club_partition_l45_45078

def is_played (team : Finset ℕ) (A B C : ℕ) : Bool :=
(A ∈ team ∧ B ∉ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∈ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∉ team ∧ C ∈ team) ∨ 
(A ∈ team ∧ B ∈ team ∧ C ∈ team)

theorem math_club_partition 
  (students : Finset ℕ) (A B C : ℕ) 
  (h_size : students.card = 24)
  (teams : List (Finset ℕ))
  (h_teams : teams.length = 4)
  (h_team_size : ∀ t ∈ teams, t.card = 6)
  (h_partition : ∀ t ∈ teams, t ⊆ students) :
  ∃ (teams_played : List (Finset ℕ)), teams_played.length = 1 ∨ teams_played.length = 3 :=
sorry

end math_club_partition_l45_45078


namespace log_sum_real_coefficients_expansion_l45_45365

theorem log_sum_real_coefficients_expansion :
  let T := (1 + Complex(0, sqrt 3) * 1) ^ 2023 +
           (1 - Complex(0, sqrt 3) * 1) ^ 2023 / 2 in
  real.log_base 3 T = 2023 * real.log_base 3 2 :=
by {
  let T := ((1 + Complex(0, sqrt 3) * 1) ^ 2023 +
           (1 - Complex(0, sqrt 3) * 1) ^ 2023) / 2,
  have H : T = 2 ^ 2023, 
  sorry,
  rw [H],
  exact congr_arg (λ x, real.log_base 3 x) (pow_log_base 2023 (2:ℝ)),
  field_simp,
  apply pow_nonneg,
  apply nat.cast_nonneg
}

end log_sum_real_coefficients_expansion_l45_45365


namespace initial_quantity_of_milk_in_container_A_l45_45121

variables {CA MB MC : ℝ}

theorem initial_quantity_of_milk_in_container_A (h1 : MB = 0.375 * CA)
    (h2 : MC = 0.625 * CA)
    (h_eq : MB + 156 = MC - 156) :
    CA = 1248 :=
by
  sorry

end initial_quantity_of_milk_in_container_A_l45_45121


namespace closest_integer_to_cube_root_of_250_l45_45512

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45512


namespace cost_of_notebook_in_dollars_l45_45014

def value_of_nickel : ℕ := 5
def number_of_nickels_used : ℕ := 26
def value_in_cents (nickels : ℕ) (value_per_nickel : ℕ) : ℕ := nickels * value_per_nickel

theorem cost_of_notebook_in_dollars :
  (value_in_cents number_of_nickels_used value_of_nickel).toRational / 100 = 1.30 :=
by
  sorry

end cost_of_notebook_in_dollars_l45_45014


namespace remainder_when_divided_by_7_l45_45099

-- Definitions based on conditions
def k_condition (k : ℕ) : Prop :=
(k % 5 = 2) ∧ (k % 6 = 5) ∧ (k < 38)

-- Theorem based on the question and correct answer
theorem remainder_when_divided_by_7 {k : ℕ} (h : k_condition k) : k % 7 = 3 :=
sorry

end remainder_when_divided_by_7_l45_45099


namespace closest_integer_to_cubert_of_250_is_6_l45_45555

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45555


namespace triangle_angle_B_l45_45759

theorem triangle_angle_B (a b : ℝ) (A B : ℝ) (ha : a = 1) (hb : b = sqrt 2) (hA : A = 30) :
  B = 45 ∨ B = 135 :=
sorry

end triangle_angle_B_l45_45759


namespace closest_integer_to_cube_root_of_250_l45_45522

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45522


namespace investment_banker_count_l45_45955

variable (total_bill : ℝ) (num_clients : ℕ) (avg_meal_cost_per_person : ℝ) (gratuity_rate : ℝ)

theorem investment_banker_count (B : ℕ) 
  (h_total_bill : total_bill = 756)
  (h_num_clients : num_clients = 5)
  (h_avg_meal_cost : avg_meal_cost_per_person = 70)
  (h_gratuity_rate : gratuity_rate = 0.20) :
  70 * ↑(B) + 70 * 5 + 0.20 * 70 * (↑B + 5) = 756 → B = 4 :=
by
  sorry

end investment_banker_count_l45_45955


namespace area_of_30_60_90_triangle_l45_45418

theorem area_of_30_60_90_triangle (p : ℝ) : 
  ∃ area : ℝ, 
    (∃ x : ℝ, 
      3 * p = x + x * real.sqrt 3 + 2 * x) ∧ 
    (area = p^2 * (6 * real.sqrt 3 - 9)) :=
sorry

end area_of_30_60_90_triangle_l45_45418


namespace binomial_12_3_eq_220_l45_45202

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l45_45202


namespace binomial_12_3_equals_220_l45_45192

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l45_45192


namespace landlord_needs_packages_l45_45877

-- Define the function to count the number of packages needed
def countPackages (upper1 upper2 : ℕ) : ℕ :=
  let digitCount : Array ℕ := Array.mkArray 10 0
  let countDigitsInRange (start end_ : ℕ) (arr : Array ℕ) : Array ℕ :=
    List.foldl
      (λ (arr : Array ℕ) (n : ℕ) => 
        let digits := 
          (List.range 100).map (λ _ => [n / 100, (n / 10) % 10, n % 10])
        digits.foldl (λ (arr : Array ℕ) (d : ℕ) => arr.modify d ((· + 1) ∘ (·))) arr
      )
      arr 
      (List.range (end_ - start + 1)).map (λ x => start + x)
  let digitCount := countDigitsInRange 150 upper1 digitCount
  let digitCount := countDigitsInRange 250 upper2 digitCount
  digitCount.maxD 0

-- The theorem stating the resulting number of packages needed.
theorem landlord_needs_packages : countPackages 180 280 = 41 :=
by sorry

end landlord_needs_packages_l45_45877


namespace det_scaled_matrix_l45_45689

theorem det_scaled_matrix (a b c d : ℝ) (h : a * d - b * c = 5) : 
  (3 * a) * (3 * d) - (3 * b) * (3 * c) = 45 :=
by 
  sorry

end det_scaled_matrix_l45_45689


namespace closest_integer_to_cube_root_of_250_l45_45520

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45520


namespace total_income_l45_45959

variables (I : ℝ) (correct_income : ℝ := 280701.75)

-- Provided conditions
def distributed_to_children (I : ℝ) : ℝ := 0.6 * I
def deposited_to_wife (I : ℝ) : ℝ := 0.25 * I
def remaining_after_distribution (I : ℝ) : ℝ := I - (distributed_to_children I) - (deposited_to_wife I)
def donated_to_orphan (I : ℝ) : ℝ := 0.05 * (remaining_after_distribution I)
def final_remaining (I : ℝ) : ℝ := (remaining_after_distribution I) - (donated_to_orphan I)

-- Given final remaining amount is $40,000
constant final_amount : ℝ := 40000

theorem total_income :
  I = correct_income :=
by
  sorry

end total_income_l45_45959


namespace moon_speed_conversion_l45_45060

theorem moon_speed_conversion (speed_kph : ℝ) (conversion_factor : ℝ) (expected_speed_kps : ℝ) :
  speed_kph = 3780 → conversion_factor = 3600 → expected_speed_kps = 1.05 →
  (speed_kph / conversion_factor = expected_speed_kps) :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  rfl

end moon_speed_conversion_l45_45060


namespace closest_cube_root_l45_45502

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45502


namespace closest_cube_root_l45_45506

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45506


namespace closest_integer_to_cube_root_250_l45_45529

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45529


namespace trial_point_value_l45_45783

theorem trial_point_value (x1 x2 x3 : ℝ) (h_interval : 2 ≤ x1 ∧ x1 ≤ 4 ∧ 2 ≤ x2 ∧ x2 ≤ 4)
  (h_better : x1 < x2)
  (h_618_method : (x1 = 2 + 0.618 * (4 - 2) ∧ x2 = 2 + (4 - x1)) ∨ 
                  (x1 = 2 + (4 - (2 + 0.618 * (4 - 2))) ∧ x2 = 2 + 0.618 * (4 - 2 /)))
  : x3 = 3.528 ∨ x3 = 2.472 := sorry

end trial_point_value_l45_45783


namespace green_disks_count_l45_45135

-- Definitions of the conditions given in the problem
def total_disks : ℕ := 14
def red_disks (g : ℕ) : ℕ := 2 * g
def blue_disks (g : ℕ) : ℕ := g / 2

-- The theorem statement to prove
theorem green_disks_count (g : ℕ) (h : 2 * g + g + g / 2 = total_disks) : g = 4 :=
sorry

end green_disks_count_l45_45135


namespace min_value_n_l45_45079

theorem min_value_n (students : ℕ) (h_students : students = 53) : 
  ∃ (n : ℕ), n = 9 ∧ 
  ∀ (s : set (set ℕ)), 
    (∀ x ∈ s, x.card ≥ 1 ∧ x.card ≤ 2) ∧ 
    (⋃₀ s).card = students → 
  (∀ t ∈ s, (t.card = 1 ∨ t.card = 2)) :=
begin
  use 9,
  split,
  { refl },
  { sorry }
end

end min_value_n_l45_45079


namespace largest_sum_of_products_l45_45288

theorem largest_sum_of_products : 
  ∃ (a b c d : ℕ), 
    (a ∈ {1, 3, 4, 5}) ∧ 
    (b ∈ {1, 3, 4, 5}) ∧ 
    (c ∈ {1, 3, 4, 5}) ∧ 
    (d ∈ {1, 3, 4, 5}) ∧ 
    (a ≠ b) ∧ 
    (a ≠ c) ∧ 
    (a ≠ d) ∧
    (b ≠ c) ∧
    (b ≠ d) ∧
    (c ≠ d) ∧
    (a + b + c + d = 13) ∧
    (a^2 + b^2 + c^2 + d^2 = 51) ∧
    ab + bc + cd + da = 42 :=
begin
  sorry

end largest_sum_of_products_l45_45288


namespace pills_needed_for_week_l45_45841

def pill_mg : ℕ := 50 -- Each pill has 50 mg of Vitamin A.
def recommended_daily_mg : ℕ := 200 -- The recommended daily serving of Vitamin A is 200 mg.
def days_in_week : ℕ := 7 -- There are 7 days in a week.

theorem pills_needed_for_week : (recommended_daily_mg / pill_mg) * days_in_week = 28 := 
by 
  sorry

end pills_needed_for_week_l45_45841


namespace domain_equivalence_l45_45710

theorem domain_equivalence (f : ℝ → ℝ) : 
  (∀ x, 0 ≤ x ∧ x ≤ 3 → ∃ y, f(y) = f(x^2 - 1)) →
  (∀ x, 0 ≤ x ∧ x ≤ (9 : ℝ) / 2 → ∃ y, f(y) = f(2*x - 1)) :=
by 
  sorry

end domain_equivalence_l45_45710


namespace percentage_defective_is_10_l45_45163

-- Definitions based on the conditions
def examined_meters : ℕ := 200
def rejected_meters : ℕ := 20

-- The calculation of the percentage
def percentage_defective : ℝ := (rejected_meters / examined_meters) * 100

-- Statement of the proof problem
theorem percentage_defective_is_10 :
  percentage_defective = 10 :=
by
  sorry

end percentage_defective_is_10_l45_45163


namespace hannahs_adblock_not_block_l45_45295

theorem hannahs_adblock_not_block (x : ℝ) (h1 : 0.8 * x = 0.16) : x = 0.2 :=
by {
  sorry
}

end hannahs_adblock_not_block_l45_45295


namespace roshini_spent_on_sweets_l45_45929

theorem roshini_spent_on_sweets
  (initial_amount : Real)
  (amount_given_per_friend : Real)
  (num_friends : Nat)
  (total_amount_given : Real)
  (amount_spent_on_sweets : Real) :
  initial_amount = 10.50 →
  amount_given_per_friend = 3.40 →
  num_friends = 2 →
  total_amount_given = amount_given_per_friend * num_friends →
  amount_spent_on_sweets = initial_amount - total_amount_given →
  amount_spent_on_sweets = 3.70 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end roshini_spent_on_sweets_l45_45929


namespace remainder_5_pow_2048_mod_17_l45_45674

theorem remainder_5_pow_2048_mod_17 : (5 ^ 2048) % 17 = 0 :=
by
  sorry

end remainder_5_pow_2048_mod_17_l45_45674


namespace solution_exists_unique_l45_45815

variable (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)

theorem solution_exists_unique (x y z : ℝ)
  (hx : x = (b + c) / 2)
  (hy : y = (c + a) / 2)
  (hz : z = (a + b) / 2)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by
  sorry

end solution_exists_unique_l45_45815


namespace closest_integer_to_cube_root_of_250_l45_45526

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45526


namespace islander_distances_l45_45580

theorem islander_distances (A B C D : ℕ) (k1 : A = 1 ∨ A = 2)
  (k2 : B = 2)
  (C_liar : C = 1) (is_knight : C ≠ 1) :
  C = 1 ∨ C = 3 ∨ C = 4 ∧ D = 2 :=
by {
  sorry
}

end islander_distances_l45_45580


namespace smallest_positive_period_center_of_symmetry_find_a_b_l45_45726

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^4 - (Real.cos x)^4 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

theorem smallest_positive_period (x : ℝ) :
  f x = 2 * Real.sin(2 * x - Real.pi / 6) - 1 ∧
  (∀ x, f (x + Real.pi) = f x) :=
sorry

theorem center_of_symmetry (k : ℤ) :
  ∃ c, c = (k : ℝ) * (Real.pi / 2) + Real.pi / 12 ∧ ∀ x, f (2 * c - x) = -f x :=
sorry

variable (a b : ℝ)
noncomputable def g (x : ℝ) : ℝ := a * f x + b

theorem find_a_b :
  (∀ x ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), 3 ≤ g x ∧ g x ≤ 11) →
  (a = 2 ∧ b = 9) ∨ (a = -2 ∧ b = 5) :=
sorry

end smallest_positive_period_center_of_symmetry_find_a_b_l45_45726


namespace no_circle_arrangement_0_to_9_l45_45942

theorem no_circle_arrangement_0_to_9 :
  ¬ ∃ (f : Fin 10 → Fin 10), (∀ i, abs ((f i).val - (f (i + 1) % 10).val) = 3 ∨ abs ((f i).val - (f (i + 1) % 10).val) = 4 ∨ abs ((f i).val - (f (i + 1) % 10).val) = 5) := sorry

end no_circle_arrangement_0_to_9_l45_45942


namespace solve_eq1_solve_eq2_l45_45394

theorem solve_eq1 (x : ℝ) : (12 * (x - 1) ^ 2 = 3) ↔ (x = 3/2 ∨ x = 1/2) := 
by sorry

theorem solve_eq2 (x : ℝ) : ((x + 1) ^ 3 = 0.125) ↔ (x = -0.5) := 
by sorry

end solve_eq1_solve_eq2_l45_45394


namespace max_beautiful_configuration_l45_45590

/-- A configuration is called beautiful if for any triangle formed by points within the configuration that has an angle ≥ 120 degrees, exactly two of its vertices share the same color. -/
def beautiful_configuration (points : Finset Point) : Prop :=
  ∀ (a b c : Point), a ∈ points → b ∈ points → c ∈ points →
  ∃ (angle_a angle_b angle_c : ℝ), angle_a + angle_b + angle_c = 180 ∧
  (angle_a ≥ 120 ∨ angle_b ≥ 120 ∨ angle_c ≥ 120) →
  ((a.color = b.color ∧ b.color ≠ c.color) ∨ 
   (a.color = c.color ∧ a.color ≠ b.color) ∨ 
   (b.color = c.color ∧ a.color ≠ b.color))

/-- The maximum number of points that can form a beautiful configuration -/
theorem max_beautiful_configuration (n : ℕ) (points : Finset Point) (h : beautiful_configuration points) : n ≤ 25 :=
sorry

end max_beautiful_configuration_l45_45590


namespace sum_alternating_binomial_l45_45914

theorem sum_alternating_binomial:
  (∑ k in Finset.range 51, (-1)^k * (Nat.choose 101 (2*k + 1))) = 2^50 :=
by
  sorry

end sum_alternating_binomial_l45_45914


namespace combination_30_choose_5_l45_45603

theorem combination_30_choose_5 :
  (nat.choose 30 5) = 142506 :=
by
  sorry

end combination_30_choose_5_l45_45603


namespace closest_integer_to_cube_root_of_250_l45_45541

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45541


namespace part_a_part_b_l45_45362

-- Given distinct primes p and q
variables (p q : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] (h : p ≠ q)

-- Prove p^q + q^p ≡ p + q (mod pq)
theorem part_a (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) :
  (p^q + q^p) % (p * q) = (p + q) % (p * q) := by
  sorry

-- Given distinct primes p and q, and neither are 2
theorem part_b (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) (hp2 : p ≠ 2) (hq2 : q ≠ 2) :
  Even (Nat.floor ((p^q + q^p) / (p * q))) := by
  sorry

end part_a_part_b_l45_45362


namespace closest_integer_to_cube_root_250_l45_45534

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45534


namespace trigonometric_identity_l45_45571

theorem trigonometric_identity
  (sin : ℝ → ℝ)
  (cos : ℝ → ℝ)
  (h_sin_cos_180: ∀ x, cos (180 - x) = - cos x)
  (h_cos_90: ∀ x, cos (90 + x) = - sin x)
  : (sin 20 * cos 10 + cos 160 * cos 100) / (sin 21 * cos 9 + cos 159 * cos 99) = 1 :=
by
  sorry

end trigonometric_identity_l45_45571


namespace marbles_left_in_the_box_l45_45440

-- Define the main problem parameters.
def total_marbles : ℕ := 50
def white_marbles : ℕ := 20
def blue_marbles : ℕ := (total_marbles - white_marbles) / 2
def red_marbles : ℕ := blue_marbles
def removed_marbles : ℕ := 2 * (white_marbles - blue_marbles)
def remaining_marbles : ℕ := total_marbles - removed_marbles

-- The theorem to prove the number of marbles left in the box.
theorem marbles_left_in_the_box : remaining_marbles = 40 := by
  unfold total_marbles white_marbles blue_marbles red_marbles removed_marbles remaining_marbles
  -- Here goes the calculus step simplification
  sorry

end marbles_left_in_the_box_l45_45440


namespace total_children_count_l45_45894

theorem total_children_count 
  (happy_children : ℕ)
  (sad_children : ℕ)
  (neither_happy_nor_sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (neither_happy_nor_sad_boys : ℕ)
  : happy_children = 30 → sad_children = 10 → neither_happy_nor_sad_children = 20 →
    boys = 22 → girls = 38 → happy_boys = 6 → sad_girls = 4 → neither_happy_nor_sad_boys = 10 →
    happy_children + sad_children + neither_happy_nor_sad_children = boys + girls :=
by {
  intros,
  sorry
}

end total_children_count_l45_45894


namespace female_employees_l45_45125

theorem female_employees (E M F : ℕ) (h1 : 300 = 300) (h2 : (2/5 : ℚ) * E = (2/5 : ℚ) * M + 300) (h3 : E = M + F) : F = 750 := 
by
  sorry

end female_employees_l45_45125


namespace sum_diff_l45_45057

-- Define the lengths of the ropes
def shortest_rope_length := 80
def ratio_shortest := 4
def ratio_middle := 5
def ratio_longest := 6

-- Use the given ratio to find the common multiple x.
def x := shortest_rope_length / ratio_shortest

-- Find the lengths of the other ropes
def middle_rope_length := ratio_middle * x
def longest_rope_length := ratio_longest * x

-- Define the sum of the longest and shortest ropes
def sum_of_longest_and_shortest := longest_rope_length + shortest_rope_length

-- Define the difference between the sum of the longest and shortest rope and the middle rope
def difference := sum_of_longest_and_shortest - middle_rope_length

-- Theorem statement
theorem sum_diff : difference = 100 := by
  sorry

end sum_diff_l45_45057


namespace incorrect_conclusions_count_l45_45623

theorem incorrect_conclusions_count :
  (1 ≠ ∃ n, prime n ∧ composite n) ∧ 
  (∃! n, even n ∧ ¬composite n) ∧ 
  (∃! n, last_digit_is_5 n ∧ ¬composite n) ∧ 
  (¬(∀ n, digit_sum_multiple_of_3 n → composite n)) →
  (\lambda cond1 cond2 cond3 cond4, cond1 + cond2 + cond3 + cond4 = 1) :=
by
  sorry

end incorrect_conclusions_count_l45_45623


namespace find_length_AD_l45_45789

noncomputable def length_AD 
  (A B C D : Type*) [metric_space A] [has_dist A] 
  (h_triangle : is_triangle A B C) 
  (h_bisector : is_angle_bisector A D C)
  (h_segment_D_B : dist B D = 40) 
  (h_segment_B_C : dist B C = 45) 
  (h_segment_A_C : dist A C = 27) : real :=
  24

theorem find_length_AD
  (A B C D : Type*) 
  [metric_space A] [has_dist A] 
  (h_triangle : is_triangle A B C) 
  (h_bisector : is_angle_bisector A D C)
  (h_segment_D_B : dist B D = 40) 
  (h_segment_B_C : dist B C = 45) 
  (h_segment_A_C : dist A C = 27) : 
  dist A D = 24 :=
  by
  sorry

end find_length_AD_l45_45789


namespace right_triangle_has_three_altitudes_l45_45921

theorem right_triangle_has_three_altitudes :
  ∀ (Δ : Type) [IsTriangle Δ] (right : IsRightTriangle Δ),
  ∃ (a1 a2 a3 : Altitude Δ), a1 ≠ a2 ∧ a2 ≠ a3 ∧ a3 ≠ a1 :=
sorry

end right_triangle_has_three_altitudes_l45_45921


namespace sqrt_defined_iff_ge_one_l45_45051

theorem sqrt_defined_iff_ge_one (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 := by
  sorry

end sqrt_defined_iff_ge_one_l45_45051


namespace base_change_2017_l45_45377

theorem base_change_2017 :
  let a := 4
  let b := 6
  let c := 12
  133201₄ = 2017 ∧ 13201₆ = 2017 ∧ 1201₁₂ = 2017 →
  a + b + c = 22 :=
by
  intros
  sorry

end base_change_2017_l45_45377


namespace binomial_12_3_eq_220_l45_45206

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l45_45206


namespace find_length_KT_l45_45763

-- Here we define lengths and angles in cm and degrees
variables {θ : ℝ}
variables {MK TL MT KL KT : ℝ}

-- Given conditions
def conditions : Prop := 
  MK = 5 ∧ TL = 9 ∧ MT = 6 ∧ KL = 8 ∧ ∠ MKT = ∠ MTL ∧
  MT^2 = MK^2 + KL^2 - 2 * MK * KL * real.cos θ ∧
  θ = real.acos(53/80) ∧
  KT^2 = MT^2 + TL^2 - 2 * MT * TL * real.cos θ

-- Main theorem to prove
theorem find_length_KT (h : conditions) : KT = 10.25 :=
by sorry

end find_length_KT_l45_45763


namespace evaluate_64_pow_7_over_6_l45_45227

theorem evaluate_64_pow_7_over_6 : (64 : ℝ)^(7 / 6) = 128 := by
  have h : (64 : ℝ) = 2^6 := by norm_num
  rw [h]
  norm_num
  sorry

end evaluate_64_pow_7_over_6_l45_45227


namespace Kamal_math_marks_l45_45799

theorem Kamal_math_marks :
  ∀ (marks_in_english marks_in_physics marks_in_chemistry marks_in_biology : ℕ)
    (average_marks total_subjects : ℕ),
  marks_in_english = 66 → 
  marks_in_physics = 77 → 
  marks_in_chemistry = 62 → 
  marks_in_biology = 75 → 
  average_marks = 69 → 
  total_subjects = 5 → 
  let total_marks := average_marks * total_subjects in
  let sum_known_marks := marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology in
  total_marks - sum_known_marks = 65 :=
by
  intros marks_in_english marks_in_physics marks_in_chemistry marks_in_biology average_marks total_subjects
  intros h_eng h_phy h_chem h_bio h_avg h_total
  have total_marks := average_marks * total_subjects
  have sum_known_marks := marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology
  rw [h_eng, h_phy, h_chem, h_bio, h_avg, h_total]
  dsimp
  norm_num
  sorry

end Kamal_math_marks_l45_45799


namespace circle_tangent_cosine_l45_45693

theorem circle_tangent_cosine
  (O A M N L P Q : Type)
  (r : ℝ)
  (h1 : is_circle_with_radius O r)
  (h2 : is_tangent_with_point A M O)
  (h3 : is_tangent_with_point A N O)
  (h4 : is_point_on_minor_arc L M N O)
  (h5 : is_parallel_line_through_point A M N)
  (h6 : is_parallel_line_intersect P Q A M N L) :
  r^2 = (O.P * O.Q * ℝ.cos (angle O P O Q)) :=
by
  sorry

end circle_tangent_cosine_l45_45693


namespace closest_cube_root_l45_45501

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45501


namespace solution_set_for_inequality_l45_45430

theorem solution_set_for_inequality :
  {x : ℝ | (1 / (x - 1) ≥ -1)} = {x : ℝ | x ≤ 0 ∨ x > 1} :=
by
  sorry

end solution_set_for_inequality_l45_45430


namespace rhombus_area_from_equilateral_triangles_l45_45152

theorem rhombus_area_from_equilateral_triangles 
  (s : ℝ)
  (hs : s = 4)
  (h_triangles : ∀ t : Triangle, t.isEquilateral ∧ t.base ∈ {side1, side3} (square s)) :
  area (rhombus_from_triangles t1 t2) = 8 * sqrt 3 - 8 := by
sorry

structure Square where
  side : ℝ

structure Triangle where
  side : ℝ
  isEquilateral : Prop

def rhombus : Type := ...

def area (r : rhombus) : ℝ := ...

def square (s : ℝ) : Square :=
  { side := s }

def rhombus_from_triangles (t1 t2 : Triangle) : rhombus :=
  ...

set_option maxHeartbeats 1000000

end rhombus_area_from_equilateral_triangles_l45_45152


namespace marcus_paintings_total_l45_45371

theorem marcus_paintings_total :
  ∃ (num_paintings_on_day : ℕ → ℕ), 
  num_paintings_on_day 1 = 2 ∧ 
  num_paintings_on_day 2 = 4 ∧ 
  num_paintings_on_day 3 = 7 ∧ 
  num_paintings_on_day 4 = 8 ∧ 
  num_paintings_on_day 5 = 8 ∧ 
  (∑ i in finset.range 5, num_paintings_on_day (i + 1)) = 29 :=
begin
  -- We'll fill this proof in later
  sorry
end

end marcus_paintings_total_l45_45371


namespace closest_cube_root_of_250_l45_45473

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45473


namespace min_value_x_plus_one_over_x_minus_one_max_value_sqrt_x_times_10_minus_x_l45_45110

-- Statement for problem A
theorem min_value_x_plus_one_over_x_minus_one (x : ℝ) (h : 1 < x) : 
  ∃ y, y = x + (1 / (x - 1)) ∧ y ≥ 3 := by
  sorry

-- Statement for problem C
theorem max_value_sqrt_x_times_10_minus_x (x : ℝ) (h1 : 0 < x) (h2 : x < 10) :
  ∃ y, y = sqrt (x * (10 - x)) ∧ y ≤ 5 := by
  sorry

end min_value_x_plus_one_over_x_minus_one_max_value_sqrt_x_times_10_minus_x_l45_45110


namespace blue_paint_cans_needed_l45_45379

theorem blue_paint_cans_needed (ratio_bg : ℤ × ℤ) (total_cans : ℤ) (r : ratio_bg = (4, 3)) (t : total_cans = 42) :
  let ratio_bw : ℚ := 4 / (4 + 3) 
  let blue_cans : ℚ := ratio_bw * total_cans 
  blue_cans = 24 :=
by
  sorry

end blue_paint_cans_needed_l45_45379


namespace part1_solution_set_part2_range_a_l45_45412

noncomputable def f (x a : ℝ) := 5 - abs (x + a) - abs (x - 2)

-- Part 1
theorem part1_solution_set (x : ℝ) (a : ℝ) (h : a = 1) :
  (f x a ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 3) := sorry

-- Part 2
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2) := sorry

end part1_solution_set_part2_range_a_l45_45412


namespace possible_integer_values_of_m_l45_45870

theorem possible_integer_values_of_m : 
  {m : ℕ // ∃ x : ℕ, (m + 2) * x = 12}.card = 4 := 
sorry

end possible_integer_values_of_m_l45_45870


namespace islanders_statements_l45_45583

-- Define the basic setup: roles of individuals
inductive Role
| knight
| liar
open Role

-- Define each individual's statement
def A_statement := λ (distance : ℕ), distance = 1
def B_statement := λ (distance : ℕ), distance = 2

-- Prove that given the conditions, the possible distances mentioned by the third and fourth islanders can be as specified
theorem islanders_statements :
  ∃ (C_statement D_statement : ℕ → Prop),
  (∀ distance, C_statement distance ↔ distance ∈ {1, 3, 4}) ∧ (∀ distance, D_statement distance ↔ distance = 2) :=
by
  sorry

end islanders_statements_l45_45583


namespace compote_relative_decrease_l45_45976

theorem compote_relative_decrease (V : ℝ) (hV : V > 0) :
  (Natasha_ate_one_third_and_level_dropped_by_one_fourth : ∃ V : ℝ, (V > 0) ∧ ∀ (natasha_ate : ℝ), natasha_ate = V / 3 → level_drop = V / 4 ∧ compote_level_afterwards = 3 / 4 * V) → 
  relative_decrease_after_all_peaches : (remaining_peaches_eaten : ℝ) → (volume_drop : remaining_peaches_eaten = V / 6 → ∀ (initial_volume_after_first_decrease : ℝ), initial_volume_after_first_decrease = 3 / 4 * V → 
  relative_decrease_after_peaches = volume_drop / initial_volume_after_first_decrease) :=
begin
  sorry
end

end compote_relative_decrease_l45_45976


namespace least_common_multiple_of_fractions_l45_45457

variable (x : ℤ) (h : x ≠ 0)

theorem least_common_multiple_of_fractions :
  LCM (1/x) (LCM (1/(3*x)) (1/(4*x))) = 1/(12*x) := by
  sorry

end least_common_multiple_of_fractions_l45_45457


namespace sum_difference_l45_45906

theorem sum_difference (n : ℕ) (h : n = 1000) :
  let sum_odd := (n / 2) * (1 + (2 * n - 1))
  let sum_even := (n / 2) * (0 + (2 * n - 2))
  sum_even - sum_odd = -1000 :=
by
  sorry

end sum_difference_l45_45906


namespace reduced_price_is_correct_l45_45964

noncomputable def P : ℝ := 0.398

noncomputable def reduced_price_per_banana : ℝ := 0.60 * P

noncomputable def reduced_price_per_dozen : ℝ := reduced_price_per_banana * 12

theorem reduced_price_is_correct :
  reduced_price_per_dozen ≈ 2.87 :=
begin
  -- The required calculations based on given conditions are processed here.
  sorry
end

end reduced_price_is_correct_l45_45964


namespace moles_NaOH_to_form_H2O_2_moles_l45_45663

-- Define the reaction and moles involved
def reaction : String := "NH4NO3 + NaOH -> NaNO3 + NH3 + H2O"
def moles_H2O_produced : Nat := 2
def moles_NaOH_required (moles_H2O : Nat) : Nat := moles_H2O

-- Theorem stating the required moles of NaOH to produce 2 moles of H2O
theorem moles_NaOH_to_form_H2O_2_moles : moles_NaOH_required moles_H2O_produced = 2 := 
by
  sorry

end moles_NaOH_to_form_H2O_2_moles_l45_45663


namespace two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l45_45234

theorem two_pow_add_three_perfect_square (n : ℕ) :
  ∃ k, 2^n + 3 = k^2 ↔ n = 0 :=
by {
  sorry
}

theorem two_pow_add_one_perfect_square (n : ℕ) :
  ∃ k, 2^n + 1 = k^2 ↔ n = 3 :=
by {
  sorry
}

end two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l45_45234


namespace general_term_of_sequence_l45_45219

noncomputable def harmonic_mean {n : ℕ} (p : Fin n → ℝ) : ℝ :=
  n / (Finset.univ.sum (fun i => p i))

theorem general_term_of_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, harmonic_mean (fun i : Fin n => a (i + 1)) = 1 / (2 * n - 1))
    (h₂ : ∀ n : ℕ, (Finset.range n).sum a = 2 * n^2 - n) :
  ∀ n : ℕ, a n = 4 * n - 3 := by
  sorry

end general_term_of_sequence_l45_45219


namespace red_ball_probability_l45_45771

-- Define the conditions
def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3
def red_balls : ℕ := total_balls - yellow_balls - green_balls

-- Define the probability function
def probability_of_red_ball (total red : ℕ) : ℚ := red / total

-- The main theorem statement to prove
theorem red_ball_probability :
  probability_of_red_ball total_balls red_balls = 3 / 5 :=
by
  sorry

end red_ball_probability_l45_45771


namespace solve_trig_eq_l45_45855

noncomputable def solve_eq (x : Real) : Prop :=
  (∃ k : ℤ, x = (π * k) / 1009 ∧ k % 1009 ≠ 0) ∨ 
  (∃ k : ℤ, x = (π + 4 * π * k) / 4036)

theorem solve_trig_eq (x : Real) (H : ∑ k in Finset.range 1009 | k.odd, Real.sin (k * x) = ∑ k in Finset.range 1009 | k.odd, Real.cos (k * x)) : solve_eq x :=
sorry

end solve_trig_eq_l45_45855


namespace transylvanian_sanity_l45_45398

theorem transylvanian_sanity (sane : Prop) (belief : Prop) (h1 : sane) (h2 : sane → belief) : belief :=
by
  sorry

end transylvanian_sanity_l45_45398


namespace dessert_menus_count_l45_45949

theorem dessert_menus_count:
  ∃ f : Fin 9 → Fin 5, (f 0 = 0) ∧ (f 4 = 1) ∧ (∀ i : Fin 8, f i ≠ f (i + 1)) ∧
  ( ∑ (i : Fin 9), 1 = 9) ∧
  (∃ n : Nat, n = 16384) := by
  sorry

end dessert_menus_count_l45_45949


namespace min_variance_l45_45697

/--
Given a sample x, 1, y, 5 with an average of 2,
prove that the minimum value of the variance of this sample is 3.
-/
theorem min_variance (x y : ℝ) 
  (h_avg : (x + 1 + y + 5) / 4 = 2) :
  3 ≤ (1 / 4) * ((x - 2) ^ 2 + (y - 2) ^ 2 + (1 - 2) ^ 2 + (5 - 2) ^ 2) :=
sorry

end min_variance_l45_45697


namespace num_50_not_30_ray_partitional_points_l45_45807

structure Rectangle :=
  (length : ℝ)
  (width : ℝ)

def is_n_ray_partitional (R : Rectangle) (Y : ℝ × ℝ) (n : ℕ) : Prop :=
  ∃ (rays : list (ℝ × ℝ)), rays.length = n ∧ -- This is a simplified representation
  all_areas_equal : Prop := true -- Assume we have a method to verify equal area conditions.

theorem num_50_not_30_ray_partitional_points :
  let R := Rectangle 2 1 in
  ∃ (count : ℕ),
  count = 20 ↔
  (∀ Y, is_n_ray_partitional R Y 50 → ¬ is_n_ray_partitional R Y 30) :=
by sorry

end num_50_not_30_ray_partitional_points_l45_45807


namespace sum_base8_of_256_and_64_l45_45414

theorem sum_base8_of_256_and_64 :
  let n1 := 256
  let n2 := 64
  let base := 8
  let b256 := "400" -- base 8 representation of 256
  let b64 := "100" -- base 8 representation of 64
  let result := "500" -- base 8 sum: 400 + 100
  
  convert n1 8 = b256 ∧ convert n2 8 = b64 ∧ b256 + b64 = result
sorry

end sum_base8_of_256_and_64_l45_45414


namespace decreasing_direct_proportion_l45_45278

theorem decreasing_direct_proportion (k : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → k * x1 > k * x2) : k < 0 :=
by
  sorry

end decreasing_direct_proportion_l45_45278


namespace product_of_common_divisors_l45_45669

theorem product_of_common_divisors (d : Set ℤ) 
  (h1 : d = {n | n ∣ 150}) 
  (h2 : Set.Subset d {n | n ∣ 30}) : 
  (∏ x in d, abs x) = 16443022500 := 
by 
  sorry

end product_of_common_divisors_l45_45669


namespace BC_length_l45_45042

-- Define trapezoid and points on it
variables (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] 
  [metric_space D] [metric_space E]
  
-- Define base length and equal perimeters
variables (AD_length : ℝ) (equal_perimeters : (segment A B E) + (segment B C E) + (segment C D E))

-- Condition that base AD of trapezoid ABCD measures 30 cm
noncomputable def AD : ℝ := 30

-- Condition that triangles have equal perimeters
noncomputable def triangles_equal_perimeters : 
  (segment A B E) = (segment B C E) ∧ (segment B C E) = (segment C D E) := sorry

-- Proposition statement: determining the length of BC
theorem BC_length : BC = 15 :=
by
  -- to be proved
  sorry

end BC_length_l45_45042


namespace cyclic_quadrilateral_circle_intersects_l45_45259

variable {α : Type*} [MetricSpace α]

structure Quadrilateral (α : Type*) [MetricSpace α] :=
  (A B C D : α)
  (Convex : ConvexSpace α)

def is_cyclic (quad : Quadrilateral α) : Prop := sorry

def circle_intersects (a b : α) : Prop := sorry

theorem cyclic_quadrilateral_circle_intersects (quad : Quadrilateral α) :
  circle_intersects quad.A quad.C ∧ circle_intersects quad.B quad.D ↔ is_cyclic quad :=
sorry

end cyclic_quadrilateral_circle_intersects_l45_45259


namespace correct_statement_l45_45922

-- Definitions based on the conditions.
def is_right_rectangular_prism_hex {P : Type} := P → Prop
def right_prism_lateral_edges_length {P : Type} := P → Prop
def right_pentagonal_prism_has_5_faces {P : Type} := P → Prop
def cube_is_right_rectangular_prism {P : Type} := P → Prop
def cuboid_is_right_rectangular_prism {P : Type} := P → Prop

-- Condition definitions
axiom hex_def (P : Type) : is_right_rectangular_prism_hex P → Prop
axiom edges_len_def (P : Type) : right_prism_lateral_edges_length P → Prop
axiom pent_faces_def (P : Type) : right_pentagonal_prism_has_5_faces P → Prop
axiom cube_def (P : Type) : cube_is_right_rectangular_prism P → Prop
axiom cuboid_def (P : Type) : cuboid_is_right_rectangular_prism P → Prop

-- Correct answer corresponding to option C
theorem correct_statement (P : Type) : right_pentagonal_prism_has_5_faces P := 
by
  apply pent_faces_def
  sorry

end correct_statement_l45_45922


namespace august_problem_solution_l45_45016

theorem august_problem_solution :
  let a1 := 600
  let a2 := 2 * a1
  let a3 := (a1 + a2) - 400
  let a4 := (a1 + a2 + a3) / 3
  in a1 + a2 + a3 + a4 = 4266.67 := by
  let a1 := 600
  let a2 := 2 * a1
  let a3 := (a1 + a2) - 400
  let a4 := (a1 + a2 + a3) / 3
  have : a1 + a2 + a3 + a4 = 4266.67 := sorry
  exact this

end august_problem_solution_l45_45016


namespace zinc_sulfate_production_l45_45664

-- Conditions:
def mol_H2SO4 := 2  -- moles of Sulfuric acid (H2SO4)
def mol_Zn := 2     -- moles of Zinc (Zn)

-- Statement:
theorem zinc_sulfate_production :
  (mol_H2SO4 = 2) → (mol_Zn = 2) → mol_H2SO4 = mol_Zn → (∃ (mol_ZnSO4 : ℕ), mol_ZnSO4 = 2) :=
begin
  sorry
end

end zinc_sulfate_production_l45_45664


namespace AB_value_l45_45836

-- Definitions of points and distances
variables (A B C D E : Point) (x : ℝ)
variable (h1 : collinear [A, B, C, D])
variable (h2 : dist A B = x)
variable (h3 : dist C D = x)
variable (h4 : dist B C = 15)
variable (h5 : ¬collinear [A, B, E])
variable (h6 : dist B E = 13)
variable (h7 : dist C E = 13)
variable (h8 : perimeter (triangle A E D) = 1.5 * perimeter (triangle B E C))

-- The theorem we need to prove
theorem AB_value : dist A B = 32 / 3 :=
sorry

end AB_value_l45_45836


namespace closest_integer_to_cube_root_of_250_l45_45490

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45490


namespace magic_square_sum_l45_45329

theorem magic_square_sum (p q r s t : ℕ) (S : ℕ)
  (h1 : 27 + s + 19 = S)
  (h2 : 27 + 17 + p = S)
  (h3 : 19 + t + q = S)
  (h4 : s + r + t = S)
  (p_eq : s - p = -2)
  (t_eq : t = s + 8)
  (r_sum_eq : r + s = 38) :
  s + t = 46 := by
  sorry

end magic_square_sum_l45_45329


namespace closest_integer_to_cube_root_250_l45_45532

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45532


namespace closest_integer_to_cubert_of_250_is_6_l45_45549

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45549


namespace shifted_function_correct_l45_45038

variable (x : ℝ)

/-- The original function -/
def original_function : ℝ := 3 * x - 4

/-- The function after shifting up by 2 units -/
def shifted_function : ℝ := original_function x + 2

theorem shifted_function_correct :
  shifted_function x = 3 * x - 2 :=
by
  sorry

end shifted_function_correct_l45_45038


namespace quadrilateral_perimeter_l45_45913

theorem quadrilateral_perimeter
  (AB BC DC : ℝ)
  (h1 : AB = 15)
  (h2 : BC = 18)
  (h3 : DC = 7)
  (h_height : 8 = 8) -- height from D to line AB
  (h_perp1 : ∀ A B C D : Type, ∀ a b c d : A, B ∘ C = D ∘ A ∘ B)
  (h_perp2 : ∀ A B C D : Type, ∀ a b c d : A, B ∘ C = D ∘ A ∘ B)
: AB + BC + DC + real.sqrt ((AB - DC) ^ 2 + BC ^ 2) = 40 + real.sqrt 388 := by
  sorry

end quadrilateral_perimeter_l45_45913


namespace sufficient_not_necessary_condition_l45_45738

theorem sufficient_not_necessary_condition (x k : ℝ) (p : x ≥ k) (q : (2 - x) / (x + 1) < 0) :
  (∀ x, x ≥ k → ((2 - x) / (x + 1) < 0)) ∧ (∃ x, (2 - x) / (x + 1) < 0 ∧ x < k) → k > 2 := by
  sorry

end sufficient_not_necessary_condition_l45_45738


namespace time_for_B_and_C_l45_45611

variables (a b c : ℝ)

-- Conditions
axiom cond1 : a = (1 / 2) * b
axiom cond2 : b = 2 * c
axiom cond3 : a + b + c = 1 / 26
axiom cond4 : a + b = 1 / 13
axiom cond5 : a + c = 1 / 39

-- Statement to prove
theorem time_for_B_and_C (a b c : ℝ) (cond1 : a = (1 / 2) * b)
                                      (cond2 : b = 2 * c)
                                      (cond3 : a + b + c = 1 / 26)
                                      (cond4 : a + b = 1 / 13)
                                      (cond5 : a + c = 1 / 39) :
  (1 / (b + c)) = 104 / 3 :=
sorry

end time_for_B_and_C_l45_45611


namespace length_of_train_l45_45154

def speed_kmh : ℝ := 162
def time_seconds : ℝ := 2.222044458665529
def speed_ms : ℝ := 45  -- from conversion: 162 * (1000 / 3600)

theorem length_of_train :
  (speed_kmh * (1000 / 3600)) * time_seconds = 100 := by
  -- Proof is left out
  sorry 

end length_of_train_l45_45154


namespace prove_equations_of_lines_l45_45236

noncomputable def equation_of_line_1 : Prop :=
  let α : ℝ := real.atan 3 in
  let m := - (2 * real.tan α) / (1 - (real.tan α) ^ 2) in
  ∃ (l : ℝ → ℝ → Prop),
    (∀ x y : ℝ, l x y ↔ y + 3 = m * (x + 1)) ∧
    (l (-1) (-3))

noncomputable def equation_of_line_2 : Prop :=
  let k_ab := (1 + 1) / (-1 - 2) in
  let slope := - 1 / k_ab in
  ∃ (l : ℝ → ℝ → Prop),
    (∀ x y : ℝ, l x y ↔ y - 1 = slope * (x + 1)) ∧
    (l (-1) 1)

theorem prove_equations_of_lines :
  equation_of_line_1 ∧ equation_of_line_2 :=
sorry

end prove_equations_of_lines_l45_45236


namespace largest_expression_l45_45217

def P : ℕ := 3 * 2024 ^ 2025
def Q : ℕ := 2024 ^ 2025
def R : ℕ := 2023 * 2024 ^ 2024
def S : ℕ := 3 * 2024 ^ 2024
def T : ℕ := 2024 ^ 2024
def U : ℕ := 2024 ^ 2023

theorem largest_expression : 
  (P - Q) > (Q - R) ∧ 
  (P - Q) > (R - S) ∧ 
  (P - Q) > (S - T) ∧ 
  (P - Q) > (T - U) :=
by sorry

end largest_expression_l45_45217


namespace jim_distance_24_steps_l45_45996

-- Setting up conditions
def distance_carly_3_steps : ℝ := 3 * 0.5
def distance_jim_4_steps : ℝ := distance_carly_3_steps
def distance_jim_1_step : ℝ := distance_jim_4_steps / 4
def distance_jim_24_steps : ℝ := 24 * distance_jim_1_step

-- Statement to prove
theorem jim_distance_24_steps : distance_jim_24_steps = 9 := by
  sorry

end jim_distance_24_steps_l45_45996


namespace parabola_properties_and_line_equation_l45_45260

-- Define the problem conditions
def parabola_eq (p : ℝ) := ∀ x y : ℝ, y^2 = 2 * p * x
def point_P (x y : ℝ) := (x, y) = (1, -1)
def distance_between_focus_and_directrix (p : ℝ) := 4
def midpoint_condition (A B : ℝ × ℝ) (P : ℝ × ℝ) := P = ((A.1 + B.1)/2, (A.2 + B.2)/2)

theorem parabola_properties_and_line_equation :
  (∃ p : ℝ, p > 0 ∧ distance_between_focus_and_directrix p ∧ ∀ x y, parabola_eq p x y) →
  (point_P 1 (-1)) →
  (∀ A B : ℝ × ℝ, (A ∈ {x | parabola_eq 4 x.1 x.2} ∧ B ∈ {x | parabola_eq 4 x.1 x.2}) →
    midpoint_condition A B (1, -1)) →
  ∃ k : ℝ, k = 4 ∧ ∀ x y : ℝ, y = -4 * (x - 1) -1 → 4 * x + y = 3 :=
by
  sorry

end parabola_properties_and_line_equation_l45_45260


namespace B_minus_A_gt_sqrt_A_l45_45812

theorem B_minus_A_gt_sqrt_A (A B : ℕ) (hA_gt_1 : A > 1) 
  (hB_div : B ∣ A^2 + 1) (hB_minus_A_pos : B - A > 0) : 
  B - A > nat.sqrt A :=
sorry

end B_minus_A_gt_sqrt_A_l45_45812


namespace count_even_functions_l45_45624

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x + Real.tan x - x
noncomputable def f2 (x : ℝ) : ℝ := Real.sin x * Real.sin x + Real.cos x
noncomputable def f3 (x : ℝ) : ℝ := Real.sin (abs x)
noncomputable def f4 (x : ℝ) : ℝ := 3 * Real.sin (2 * (x + (Real.pi / 4)))

theorem count_even_functions :
  (Set.count (λ f, ∀ x : ℝ, f (-x) = f x) {f1, f2, f3, f4}) = 3 :=
sorry

end count_even_functions_l45_45624


namespace n_minus_m_eq_zero_l45_45748

-- Definitions based on the conditions
def m : ℝ := sorry
def n : ℝ := sorry
def i := Complex.I
def condition : Prop := m + i = (1 + 2 * i) - n * i

-- The theorem stating the equivalence proof problem
theorem n_minus_m_eq_zero (h : condition) : n - m = 0 :=
sorry

end n_minus_m_eq_zero_l45_45748


namespace convex_quadrilateral_area_inequality_l45_45608

-- Statement of the problem
theorem convex_quadrilateral_area_inequality 
  (ABCD : Type) [convex_quadrilateral ABCD]
  (angle_ABD angle_ACD angle_BAC angle_BDC : ℝ)
  (S_ABD S_ACD S_BAC S_BDC : ℝ)
  (h_angle_inequality : angle_ABD + angle_ACD > angle_BAC + angle_BDC) :
  S_ABD + S_ACD > S_BAC + S_BDC := 
sorry

end convex_quadrilateral_area_inequality_l45_45608


namespace stickers_per_student_l45_45013

theorem stickers_per_student : 
  ∀ (gold silver bronze total : ℕ), 
    gold = 50 →
    silver = 2 * gold →
    bronze = silver - 20 →
    total = gold + silver + bronze →
    total / 5 = 46 :=
by
  intros
  sorry

end stickers_per_student_l45_45013


namespace shortest_among_tallest_leq_tallest_among_shortest_l45_45589

/-
  Define the height of students in a 2D matrix form where rows represent transverse rows and columns represent longitudinal rows.
  Given conditions:
  - 10 students in each cross row.
  - 20 students in each longitudinal row.
  - Let A be the shortest among the tallest students chosen from each row (transversal row).
  - Let B be the tallest among the shortest students chosen from each column (longitudinal row).

  We need to prove that A ≤ B.
-/

def height_matrix : Matrix (Fin 10) (Fin 20) ℝ := sorry

def tallest_in_row (i : Fin 10) : ℝ :=
  max (height_matrix i) sorry -- Assumes an appropriate 'max' function

def shortest_among_tallest_in_rows : ℝ :=
  min (λ i, tallest_in_row i) sorry -- Assumes an appropriate 'min' function

def shortest_in_column (j : Fin 20) : ℝ :=
  min (height_matrix j) sorry -- Assumes an appropriate 'min' function

def tallest_among_shortest_in_columns : ℝ :=
  max (λ j, shortest_in_column j) sorry -- Assumes an appropriate 'max' function

theorem shortest_among_tallest_leq_tallest_among_shortest :
  shortest_among_tallest_in_rows ≤ tallest_among_shortest_in_columns :=
sorry

end shortest_among_tallest_leq_tallest_among_shortest_l45_45589


namespace country_x_income_l45_45122

variable (income : ℝ)
variable (tax_paid : ℝ)
variable (income_first_40000_tax : ℝ := 40000 * 0.1)
variable (income_above_40000_tax_rate : ℝ := 0.2)
variable (total_tax_paid : ℝ := 8000)
variable (income_above_40000 : ℝ := (total_tax_paid - income_first_40000_tax) / income_above_40000_tax_rate)

theorem country_x_income : 
  income = 40000 + income_above_40000 → 
  total_tax_paid = tax_paid → 
  tax_paid = income_first_40000_tax + (income_above_40000 * income_above_40000_tax_rate) →
  income = 60000 :=
by sorry

end country_x_income_l45_45122


namespace minimize_sum_of_digits_l45_45650

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the expression in the problem
def expression (p : ℕ) : ℕ :=
  p^4 - 5 * p^2 + 13

-- Proposition stating the conditions and the expected result
theorem minimize_sum_of_digits (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∀ q : ℕ, Nat.Prime q → q % 2 = 1 → sum_of_digits (expression q) ≥ sum_of_digits (expression 5)) →
  p = 5 :=
by
  sorry

end minimize_sum_of_digits_l45_45650


namespace fully_charge_tablet_time_l45_45962

def time_to_fully_charge_smartphone := 26 -- 26 minutes to fully charge a smartphone
def total_charge_time := 66 -- 66 minutes to charge tablet fully and phone halfway
def halfway_charge_time := time_to_fully_charge_smartphone / 2 -- 13 minutes to charge phone halfway

theorem fully_charge_tablet_time : 
  ∃ T : ℕ, T + halfway_charge_time = total_charge_time ∧ T = 53 := 
by
  sorry

end fully_charge_tablet_time_l45_45962


namespace closest_integer_to_cube_root_of_250_l45_45517

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45517


namespace angleCPA_eq_angleDPT_l45_45165

open_locale classical

variables {O A B C D T P : Type} [metric_space O]

-- Assumptions
variables (CA AB BD : ℝ)
variables (AB_diameter : ∀ {O A B},  is_diameter O A B)
variables (CA_eq_AB : ∀ {C A}, CA = AB)
variables (AB_eq_BD : ∀ {A B D}, AB = BD)
variables (CT_tangent : ∀ {C T P : O}, tangent_to_circle T P)

-- Prove
theorem angleCPA_eq_angleDPT : 
  ∀ {A B C D T P : Type} [metric_space O] 
    (CA : ℝ) (AB : ℝ) (BD : ℝ)
    (AB_diameter : ∀ {O A B},  is_diameter O A B)
    (CA_eq_AB : ∀ {C A}, CA = AB)
    (AB_eq_BD : ∀ {A B D}, AB = BD)
    (CT_tangent : ∀ {C T P : O}, tangent_to_circle T P),
  ∠CPA = ∠DPT :=
sorry

end angleCPA_eq_angleDPT_l45_45165


namespace multiple_of_two_power_with_digits_l45_45843

theorem multiple_of_two_power_with_digits (n : ℕ) (hn : n ≥ 1) : 
  ∃ k : ℕ, (k < 10^n) ∧ (k > 10^(n-1)) ∧ (∀ d ∈ k.digits, d = 1 ∨ d = 2) ∧ (2^n ∣ k) :=
sorry

end multiple_of_two_power_with_digits_l45_45843


namespace product_diff_is_even_l45_45360

theorem product_diff_is_even :
  ∀ (a b : Fin 7 → ℤ), (∃ σ : Fin 7 → Fin 7, Function.Bijective σ ∧ ∀ i, b i = a (σ i)) →
  Even (∏ i, a i - b i) :=
by
  intro a b h
  sorry

end product_diff_is_even_l45_45360


namespace max_ab_value_l45_45596

variable (a b c : ℝ)

-- Conditions
axiom h1 : 0 < a ∧ a < 1
axiom h2 : 0 < b ∧ b < 1
axiom h3 : 0 < c ∧ c < 1
axiom h4 : 3 * a + 2 * b = 1

-- Goal
theorem max_ab_value : ab = 1 / 24 :=
by
  sorry

end max_ab_value_l45_45596


namespace range_of_a_l45_45129

def A : set ℝ := {x | x^2 + 4 * x = 0}
def B (a : ℝ) : set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 1) = 0}

theorem range_of_a (a : ℝ) :
  (A ∩ B a) = B a ↔ (a ≤ -1 ∨ a = 1) := by
  sorry

end range_of_a_l45_45129


namespace gaussian_guardians_total_points_l45_45073

theorem gaussian_guardians_total_points :
  let daniel := 7
  let curtis := 8
  let sid := 2
  let emily := 11
  let kalyn := 6
  let hyojeong := 12
  let ty := 1
  let winston := 7
  daniel + curtis + sid + emily + kalyn + hyojeong + ty + winston = 54 := by
  sorry

end gaussian_guardians_total_points_l45_45073


namespace closest_cube_root_of_250_l45_45474

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45474


namespace total_earnings_to_afford_car_l45_45120

-- Define the earnings per month
def monthlyEarnings : ℕ := 4000

-- Define the savings per month
def monthlySavings : ℕ := 500

-- Define the total amount needed to buy the car
def totalNeeded : ℕ := 45000

-- Define the number of months needed to save enough money
def monthsToSave : ℕ := totalNeeded / monthlySavings

-- Theorem stating the total money earned before he saves enough to buy the car
theorem total_earnings_to_afford_car : monthsToSave * monthlyEarnings = 360000 := by
  sorry

end total_earnings_to_afford_car_l45_45120


namespace exists_circle_with_exactly_n_integer_points_l45_45021

noncomputable def circle_with_n_integer_points (n : ℕ) : Prop :=
  ∃ r : ℤ, ∃ (xs ys : List ℤ), 
    xs.length = n ∧ ys.length = n ∧
    ∀ (x y : ℤ), x ∈ xs → y ∈ ys → x^2 + y^2 = r^2

theorem exists_circle_with_exactly_n_integer_points (n : ℕ) : 
  circle_with_n_integer_points n := 
sorry

end exists_circle_with_exactly_n_integer_points_l45_45021


namespace closest_integer_to_cubert_of_250_is_6_l45_45550

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45550


namespace Patrick_can_play_l45_45369

def friends_prob := 1/2
def participants_needed := 5

noncomputable def binomial (n k: ℕ) := nat.choose n k

theorem Patrick_can_play :
  ∑ k in (finset.range (10 + 1)).filter (λ n, n ≥ participants_needed),
    binomial 10 k * (friends_prob ^ k) * ((1 - friends_prob) ^ (10 - k)) = 319/512 :=
by sorry

end Patrick_can_play_l45_45369


namespace max_value_x_plus_inv_x_theorem_l45_45075

noncomputable def max_value_x_plus_inv_x (x : ℝ) (y : ℕ → ℝ) : ℝ :=
  if h1 : (∀ i, y i > 0) ∧ (∑ i in finset.range 1000, y i = 1002 - x) 
     ∧ (∑ i in finset.range 1000, (y i)⁻¹ = 1002 - (x⁻¹)) then 
    (x + x⁻¹)
  else 
    0

theorem max_value_x_plus_inv_x_theorem (x : ℝ) (y : ℕ → ℝ) (h1 : x > 0) 
  (h2 : (∀ i, y i > 0)) 
  (h3 : x + ∑ i in finset.range 1000, y i = 1002) 
  (h4 : x⁻¹ + ∑ i in finset.range 1000, (y i)⁻¹ = 1002) : 
  max_value_x_plus_inv_x x y = 4012 / 501 := 
sorry

end max_value_x_plus_inv_x_theorem_l45_45075


namespace zero_of_C_in_interval_l45_45400

-- Definition of the function
def C (x : ℝ) : ℝ := x^2

-- Statement of the proof problem
theorem zero_of_C_in_interval : (∃ x : ℝ, C x = 0) → 0 ∈ Ioo (-1 : ℝ) 1 := 
by 
  sorry

end zero_of_C_in_interval_l45_45400


namespace binomial_12_3_eq_220_l45_45197

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l45_45197


namespace complete_squares_l45_45746

def valid_solutions (x y z : ℝ) : Prop :=
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = -2 ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = 6) ∨
  (x = 0 ∧ y = -2 ∧ z = 6) ∨
  (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 4 ∧ y = -2 ∧ z = 0) ∨
  (x = 4 ∧ y = 0 ∧ z = 6) ∨
  (x = 4 ∧ y = -2 ∧ z = 6)

theorem complete_squares (x y z : ℝ) : 
  (x - 2)^2 + (y + 1)^2 = 5 →
  (x - 2)^2 + (z - 3)^2 = 13 →
  (y + 1)^2 + (z - 3)^2 = 10 →
  valid_solutions x y z :=
by
  intros h1 h2 h3
  sorry

end complete_squares_l45_45746


namespace circumcenter_incenter_coincide_l45_45378

theorem circumcenter_incenter_coincide 
  {A B C P Q : Point} (h_abc : Triangle A B C) 
  (h_largest : AB = max (AB, BC, CA)) 
  (h_aq_eq_ac : AQ = AC) 
  (h_bp_eq_bc : BP = BC) 
  (h_p_on_ab : OnLineSegment P A B)
  (h_q_on_ab : OnLineSegment Q A B) 
  : Circumcenter (Triangle P Q C) = Incenter (Triangle A B C) := 
sorry

end circumcenter_incenter_coincide_l45_45378


namespace radius_circumscribed_sphere_l45_45043

noncomputable def pyramid_radius_circumscribed_sphere (a : ℝ) (h : ℝ) : ℝ :=
  if h = (a * Real.sqrt 3 / 2) then (a * Real.sqrt 21 / 6) else 0

theorem radius_circumscribed_sphere (a : ℝ) (h : ℝ)
  (base_square : ℝ := a)
  (height_pyramid : ℝ := a * Real.sqrt 3 / 2) :
  h = height_pyramid →
  pyramid_radius_circumscribed_sphere a height_pyramid = a * Real.sqrt 21 / 6 :=
by
  intro h_eq
  rw [pyramid_radius_circumscribed_sphere, if_pos h_eq]
  rfl

end radius_circumscribed_sphere_l45_45043


namespace find_non_working_games_l45_45831

section NedVideoGames

variable (total_games working_games_price total_money_earned : ℕ)

def non_working_games : ℕ :=
  total_games - (total_money_earned / working_games_price)

theorem find_non_working_games :
  total_games = 15 →
  working_games_price = 7 →
  total_money_earned = 63 →
  non_working_games total_games working_games_price total_money_earned = 6 :=
by
  intros h1 h2 h3
  simp [non_working_games, h1, h2, h3]
  norm_num
  sorry

end NedVideoGames

end find_non_working_games_l45_45831


namespace complex_alpha_condition_l45_45647

noncomputable def f (α z : ℂ) : ℂ := (z + α) ^ 2 + α * complex.conj z

theorem complex_alpha_condition {α z1 z2 : ℂ} (h₁ : |z1| < 1) (h₂ : |z2| < 1) (h₃ : z1 ≠ z2) :
  |α| >= 2 → f α z1 ≠ f α z2 :=
by
  sorry

end complex_alpha_condition_l45_45647


namespace road_width_l45_45008

theorem road_width
  (road_length : ℝ) 
  (truckload_area : ℝ) 
  (truckload_cost : ℝ) 
  (sales_tax : ℝ) 
  (total_cost : ℝ) :
  road_length = 2000 ∧
  truckload_area = 800 ∧
  truckload_cost = 75 ∧
  sales_tax = 0.20 ∧
  total_cost = 4500 →
  ∃ width : ℝ, width = 20 :=
by
  sorry

end road_width_l45_45008


namespace existence_of_c_l45_45804

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry
noncomputable def f'' (x : ℝ) : ℝ := sorry
noncomputable def f''' (x : ℝ) : ℝ := sorry
-- and similarly for higher derivatives up to f^(n)

theorem existence_of_c (a b : ℝ) (h : a < b) :
  (∀ (f : ℝ → ℝ) (a b : ℝ), 
    ln ((f b + f' b + f'' b + ... + f^(n) b) / 
        (f a + f' a + f'' a + ... + f^(n) a)) = b - a) → 
  ∃ c ∈ set.Ioo a b, f^(n+1) c = f c :=
begin
  intros hf,
  -- the proof goes here
  sorry
end

end existence_of_c_l45_45804


namespace closest_integer_to_cube_root_of_250_l45_45492

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45492


namespace triangle_APQ_equilateral_or_coincide_l45_45806

noncomputable def centroid (A B C M : Point) : Prop :=
  vector (M, A) + vector (M, B) + vector (M, C) = 0

theorem triangle_APQ_equilateral_or_coincide 
  (A B C P Q M : Point)
  (h_centroid : centroid A B C M)
  (h_rot_B_to_P : rotate 120 M B = P)
  (h_rot_C_to_Q : rotate 240 M C = Q) :
  (equilateral (Triangle A P Q)) ∨ (A = P ∧ P = Q) :=
sorry

end triangle_APQ_equilateral_or_coincide_l45_45806


namespace pages_in_first_chapter_l45_45597

theorem pages_in_first_chapter
  (total_pages : ℕ)
  (second_chapter_pages : ℕ)
  (first_chapter_pages : ℕ)
  (h1 : total_pages = 81)
  (h2 : second_chapter_pages = 68) :
  first_chapter_pages = 81 - 68 :=
sorry

end pages_in_first_chapter_l45_45597


namespace closest_integer_to_cube_root_250_l45_45535

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45535


namespace horatio_sonnets_l45_45296

theorem horatio_sonnets (num_lines_per_sonnet : ℕ) (heard_sonnets : ℕ) (unheard_lines : ℕ) (h1 : num_lines_per_sonnet = 16) (h2 : heard_sonnets = 9) (h3 : unheard_lines = 126) :
  ∃ total_sonnets : ℕ, total_sonnets = 16 :=
by
  -- Note: The proof is not required, hence 'sorry' is included to skip it.
  sorry

end horatio_sonnets_l45_45296


namespace sum_first_10_terms_l45_45261

def a : ℕ → ℕ
| 0       => 0 -- This term is not used or defined in the problem
| 1       => 1
| (n + 2) => if n % 2 == 0 then a (n + 1) + 1 else a (n + 1) + 2

def S₁₀ : ℕ := (List.range 10).sum (λ n => a (n + 1))

theorem sum_first_10_terms : S₁₀ = 75 := by
  sorry

end sum_first_10_terms_l45_45261


namespace min_steps_to_empty_l45_45904

-- Definitions of sets and their sizes.
def A (i : ℕ) : Set ℕ :=
  { x | x < i }

-- Function to represent the removal process
def remove_elements (n : ℕ) (A_sets : ℕ → Set ℕ) (step : ℕ) : ℕ → Set ℕ :=
  λ i, if (i % 2^step) == 0 then A_sets i \ {x : ℕ | x < 2^step} else A_sets i

-- Prove that after 8 steps all sets are empty.
theorem min_steps_to_empty : ∀ (A_sets : ℕ → Set ℕ),
  (∀ i, i ∈ (Finset.range 1 160) → (A_sets i).card = i) →
  ∃ n, n = 8 ∧ (∀ i, i ∈ (Finset.range 1 160) → (remove_elements n A_sets 8) i = ∅) :=
sorry

end min_steps_to_empty_l45_45904


namespace number_took_preparatory_not_passed_l45_45086

variable (T : ℕ) (percentage_passed : ℝ) (not_taken_preparatory : ℕ)

theorem number_took_preparatory_not_passed (T_eq : T = 50) (percentage_passed_eq : percentage_passed = 0.3) (not_taken_preparatory_eq : not_taken_preparatory = 30) : 
  let passed := T * percentage_passed in 
  let not_passed := T - passed in 
  let took_preparatory_not_passed := not_passed - not_taken_preparatory in
  took_preparatory_not_passed = 5 := by sorry

end number_took_preparatory_not_passed_l45_45086


namespace solve_for_x_l45_45856

-- Defining the given conditions
def y : ℕ := 6
def lhs (x : ℕ) : ℕ := Nat.pow x y
def rhs : ℕ := Nat.pow 3 12

-- Theorem statement to prove
theorem solve_for_x (x : ℕ) (hypothesis : lhs x = rhs) : x = 9 :=
by sorry

end solve_for_x_l45_45856


namespace perpendicular_lines_find_a_l45_45310

theorem perpendicular_lines_find_a (a : ℝ) :
  let L1 := λ (x y : ℝ), ax + 2y + 6 = 0,
      L2 := λ (x y : ℝ), x + a*(a + 1)*y + (a^2 - 1) = 0 in
  (∀ x y, L1 x y ↔ L2 (x+y) (y-x)) →
  a = 0 ∨ a = -3 / 2 := by
  sorry

end perpendicular_lines_find_a_l45_45310


namespace closest_integer_to_cbrt_250_l45_45567

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45567


namespace triangle_third_side_max_length_l45_45857

open Real

theorem triangle_third_side_max_length 
    (A B C : ℝ)    -- angles in radians
    (a b c : ℝ)    -- lengths of triangle sides
    (h_cos_sum : cos (2 * A) + cos (2 * B) + cos (2 * C) = 0) 
    (a_eq : a = 8) 
    (b_eq : b = 15) 
    (angle_C : C = π / 4)  -- π / 4 is 45 degrees
    : c = sqrt (289 - 120 * sqrt 2) :=
sorry

end triangle_third_side_max_length_l45_45857


namespace part_1_part_2_part_3_l45_45262

-- Helper function to generate the new element based on the given rule.
def next_element (a : ℝ) : ℝ :=
  (1 + a) / (1 - a)

-- Definition of the set A satisfying the condition.
def set_A (A: Set ℝ) : Prop :=
  ∀ a ∈ A, next_element a ∈ A

-- Given elements in part (1)
def set_A1 : Set ℝ := 
  {2, -3, -1/2, 1/3}

-- Given elements in part (2)
def set_A2 : Set ℝ := 
  {3, -2, -1/3, 1/2}

-- Lean 4 Statements
theorem part_1 (A : Set ℝ) (hA : set_A A) (h2 : 2 ∈ A) : 
  A = set_A1 :=
sorry 

theorem part_2 (A : Set ℝ) (hA : set_A A) :
  0 ∉ A :=
sorry 

theorem part_3 (A : Set ℝ) (hA : set_A A) (h3 : 3 ∈ A) : 
  A = set_A2 :=
sorry 

end part_1_part_2_part_3_l45_45262


namespace fruit_prices_l45_45991

theorem fruit_prices (x y : ℝ) 
  (h₁ : 3 * x + 2 * y = 40) 
  (h₂ : 2 * x + 3 * y = 35) : 
  x = 10 ∧ y = 5 :=
by
  sorry

end fruit_prices_l45_45991


namespace negation_proposition_l45_45061

theorem negation_proposition (x : ℝ) :
  (¬ (∀ x > 0, sin x ≥ -1)) ↔ (∃ x > 0, sin x < -1) := 
by
  sorry

end negation_proposition_l45_45061


namespace orthogonal_projection_magnitude_correct_l45_45294

-- Define the vectors a and b
def a : ℝ × ℝ := (real.sqrt 3, 1)
def b : ℝ × ℝ := (1, 0)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the orthogonal projection magnitude of a onto b
def orthogonal_projection_magnitude (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

-- The proof statement
theorem orthogonal_projection_magnitude_correct :
  orthogonal_projection_magnitude a b = real.sqrt 3 :=
by
  sorry

end orthogonal_projection_magnitude_correct_l45_45294


namespace betsy_to_cindy_ratio_l45_45182

-- Definitions based on the conditions
def cindy_time : ℕ := 12
def tina_time : ℕ := cindy_time + 6
def betsy_time : ℕ := tina_time / 3

-- Theorem statement to prove
theorem betsy_to_cindy_ratio :
  (betsy_time : ℚ) / cindy_time = 1 / 2 :=
by sorry

end betsy_to_cindy_ratio_l45_45182


namespace remainder_r4_eq_10_l45_45708

noncomputable def r3 : ℝ := (1 : ℝ)^10
noncomputable def q3 (x : ℝ) : ℝ := x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1
def r4 := q3 1

theorem remainder_r4_eq_10 : r4 = 10 := by
  unfold r4
  unfold q3
  norm_num
  sorry

end remainder_r4_eq_10_l45_45708


namespace trisomic_rice_formation_variation_phenotypic_ratio_if_B_on_7_phenotypic_ratio_if_B_not_on_7_l45_45388

def rice := Type
def diploid (r : rice) := (chromosome_composition : ℕ) (composition_two_n : chromosome_composition = 24)

def trisomic_rice (r : rice) := 
  (fails_to_separate_in_first_meiosis : Prop) 
  (fails_to_separate_in_second_meiosis : Prop) 
  (gametes_n_plus_1 : Prop) 
  (gametes_n : Prop)
  (fertile_egg : Prop) 
  (sterile_pollen : Prop)

def chromosome_7_alleles (B b : rice) := Prop

theorem trisomic_rice_formation_variation (r : rice) 
  (dip : diploid r) 
  (trisomic : trisomic_rice r) : true := 
begin
  -- The formation of trisomic rice belongs to chromosomal number variation.
  sorry
end

theorem phenotypic_ratio_if_B_on_7 (r : rice) 
  (B b : rice) 
  (dip : diploid r) 
  (trisomic : trisomic_rice r) 
  (alleles_on_7 : chromosome_7_alleles B b) : 
  phenotypic_ratio_F2 (1, 2) := 
begin
  -- If the alleles (B, b) are located on chromosome 7,
  -- the phenotypic ratio in F2 is non-fragrant:fragrant=1:2.
  sorry
end

theorem phenotypic_ratio_if_B_not_on_7 (r : rice) 
  (B b : rice) 
  (dip : diploid r) 
  (trisomic : trisomic_rice r) 
  (alleles_not_on_7 : ¬chromosome_7_alleles B b) : 
  phenotypic_ratio_F2 (1, 1) := 
begin
  -- If the alleles (B, b) are not located on chromosome 7,
  -- the phenotypic ratio in F2 is non-fragrant:fragrant=1:1.
  sorry
end

noncomputable def phenotypic_ratio_F2 (ratio : ℕ × ℕ) : Prop := 
  true -- This is a placeholder definition to make the code compile. It should be replaced by the actual definition.


end trisomic_rice_formation_variation_phenotypic_ratio_if_B_on_7_phenotypic_ratio_if_B_not_on_7_l45_45388


namespace second_polygon_sides_l45_45902

theorem second_polygon_sides 
  (s : ℝ) -- side length of the second polygon
  (n1 n2 : ℕ) -- n1 = number of sides of the first polygon, n2 = number of sides of the second polygon
  (h1 : n1 = 40) -- first polygon has 40 sides
  (h2 : ∀ s1 s2 : ℝ, s1 = 3 * s2 → n1 * s1 = n2 * s2 → n2 = 120)
  : n2 = 120 := 
by
  sorry

end second_polygon_sides_l45_45902


namespace rational_roots_of_second_equation_l45_45033

theorem rational_roots_of_second_equation (p q c : ℚ) (hpq : p + q = 1) (hc : p * q = c) : 
  ∀ (x : ℚ), is_root (x^2 + p * x - q) x :=
by sorry

end rational_roots_of_second_equation_l45_45033


namespace reeya_weighted_average_l45_45026

theorem reeya_weighted_average :
  let scores := [50, 60, 70, 80, 80]
  let weights := [3, 2, 4, 1, 3]
  let weighted_sum := (50 * 3) + (60 * 2) + (70 * 4) + (80 * 1) + (80 * 3)
  let total_weight := 3 + 2 + 4 + 1 + 3
  let weighted_average := weighted_sum / total_weight in
  weighted_average = 66.92 :=
by
  sorry

end reeya_weighted_average_l45_45026


namespace fraction_sequence_product_l45_45178

theorem fraction_sequence_product :
  (∏ i in (finset.range 50).map (nat.succ ∘ (* 1) : ℕ → ℕ) \ (\lam i, 
    ((i / 4) \mod 2 = 1)) (finset.range 5 ): ℝ)) =
  2 / 632170 :=
sorry

end fraction_sequence_product_l45_45178


namespace problem_part_1_problem_part_2_l45_45312

variables {A B C : ℝ} {a b c : ℝ}

theorem problem_part_1 (h : (a - 2 * c) * real.cos B + b * real.cos A = 0): real.cos B = 1 / 3 :=
sorry

theorem problem_part_2 (h1 : real.cos B = 1 / 3) (h2 : real.sin A = 3 * real.sqrt 10 / 10): b / c = real.sqrt 7 :=
sorry

end problem_part_1_problem_part_2_l45_45312


namespace hotdog_cost_proof_l45_45343

-- Define the conditions
def winnings : ℝ := 114
def donation : ℝ := winnings / 2
def remaining_after_donation : ℝ := winnings - donation
def remaining_after_hotdog : ℝ := 55

-- The proof problem in Lean 4
theorem hotdog_cost_proof : 
  (remaining_after_donation - remaining_after_hotdog) = 2 := by
  -- Here we would provide the proof, but we leave it as a placeholder
  sorry

end hotdog_cost_proof_l45_45343


namespace percentage_vehicles_updated_2003_l45_45445

theorem percentage_vehicles_updated_2003 (a : ℝ) (h1 : 1.1^4 = 1.46) (h2 : 1.1^5 = 1.61) :
  (a * 1 / (a * 1.61) * 100 = 16.4) :=
  by sorry

end percentage_vehicles_updated_2003_l45_45445


namespace m_1_2_eq_m_2_1_mod_3_l45_45363

variables {n k : ℕ} (x : ℕ → ℤ)

-- Given conditions
def cyclic_sum (start idx len : ℕ) : ℤ :=
  (List.range len).sum (λ i, x ((start + idx + i) % n))

def S (i : ℕ) : ℤ := cyclic_sum x i k
def T (i : ℕ) : ℤ := cyclic_sum x (i + k) (n - k)

-- Counting function m(a, b)
def m (a b : ℤ) : ℕ :=
  (List.range n).count (λ i, (S x i % 3 = a) ∧ (T x i % 3 = b))

-- Theorem to prove
theorem m_1_2_eq_m_2_1_mod_3 : 
  m x 1 2 % 3 = m x 2 1 % 3 := 
sorry

end m_1_2_eq_m_2_1_mod_3_l45_45363


namespace time_addition_in_9876_sec_l45_45340

theorem time_addition_in_9876_sec :
  let current_hours := 8
  let current_minutes := 45
  let current_seconds := 0
  let interval_seconds := 9876
  let new_time := (11, 29, 36)
  add_seconds_to_time current_hours current_minutes current_seconds interval_seconds = new_time :=
by
  sorry

def add_seconds_to_time (hours : Nat) (minutes : Nat) (seconds : Nat) (interval : Nat) : Nat × Nat × Nat :=
  let total_seconds := hours * 3600 + minutes * 60 + seconds + interval
  let new_hours := total_seconds / 3600
  let rem_seconds := total_seconds % 3600
  let new_minutes := rem_seconds / 60
  let new_seconds := rem_seconds % 60
  (new_hours % 24, new_minutes, new_seconds)

end time_addition_in_9876_sec_l45_45340


namespace clocks_sync_again_in_lcm_days_l45_45988

-- Defining the given conditions based on the problem statement.

-- Arthur's clock gains 15 minutes per day, taking 48 days to gain 12 hours (720 minutes).
def arthur_days : ℕ := 48

-- Oleg's clock gains 12 minutes per day, taking 60 days to gain 12 hours (720 minutes).
def oleg_days : ℕ := 60

-- The problem asks to prove that the situation repeats after 240 days, which is the LCM of 48 and 60.
theorem clocks_sync_again_in_lcm_days : Nat.lcm arthur_days oleg_days = 240 := 
by 
  sorry

end clocks_sync_again_in_lcm_days_l45_45988


namespace verify_condition_C_l45_45919

variable (x y z : ℤ)

-- Given conditions
def condition_C : Prop := x = y ∧ y = z + 1

-- The theorem/proof problem
theorem verify_condition_C (h : condition_C x y z) : (x - y)^2 + (y - z)^2 + (z - x)^2 = 2 := 
by 
  sorry

end verify_condition_C_l45_45919


namespace find_g_at_half_l45_45285
noncomputable def f (a x : ℝ) : ℝ := log a (3 - x) + 1/4
noncomputable def g (α x : ℝ) : ℝ := x^α

theorem find_g_at_half (a : ℝ) (α : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : g α 2 = f a 2) (h4 : f a 2 = 1/4) :
  g α (1/2) = 4 := by
  sorry

end find_g_at_half_l45_45285


namespace bottles_sold_tuesday_l45_45216

def initial_inventory : ℕ := 4500
def sold_monday : ℕ := 2445
def sold_days_wed_to_sun : ℕ := 50 * 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

theorem bottles_sold_tuesday : 
  initial_inventory + bottles_delivered_saturday - sold_monday - sold_days_wed_to_sun - final_inventory = 900 := 
by
  sorry

end bottles_sold_tuesday_l45_45216


namespace find_A_l45_45785

-- Define the conditions as Lean definitions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides of the triangle

-- Given conditions
def b_eq_c : Prop := b = c
def a_squared_eq : Prop := a^2 = 2 * b^2 * (1 - Real.sin A)

-- The theorem statement
theorem find_A (h1 : b_eq_c) (h2 : a_squared_eq) : A = π / 4 :=
sorry

end find_A_l45_45785


namespace regular_tetrahedron_inequality_l45_45792

variable (A B C D P : Point)

-- Assuming some basic definitions for points, distances, and tetrahedrons.

-- Definition of a regular tetrahedron
def is_regular_tetrahedron (A B C D : Point) : Prop :=
  dist A B = dist A C ∧ dist A B = dist A D ∧ dist A B = dist B C ∧ dist A B = dist B D ∧ dist A B = dist C D

-- Definition of an internal point
def is_internal_point (P A B C D : Point) : Prop :=
  dist A P + dist B P < dist A B + dist C P - dist D P -- Simplified definition for demonstration.

-- The theorem to prove that the stated inequality holds.
theorem regular_tetrahedron_inequality (h : is_regular_tetrahedron A B C D) (hP: is_internal_point P A B C D) :
  dist P A + dist P B + dist P C < dist D A + dist D B + dist D C :=
sorry

end regular_tetrahedron_inequality_l45_45792


namespace forty_percent_of_number_l45_45930

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) : 0.40 * N = 120 :=
sorry

end forty_percent_of_number_l45_45930


namespace count_lucky_numbers_l45_45454

-- Definitions
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digits_sum_to_six (n : ℕ) : Prop := (n / 100 + (n / 10) % 10 + n % 10 = 6)

-- Proposition to prove
theorem count_lucky_numbers : {n : ℕ | is_three_digit_number n ∧ digits_sum_to_six n}.to_finset.card = 21 := 
by 
  sorry

end count_lucky_numbers_l45_45454


namespace simplify_expression_l45_45387

variables (a b c : ℝ)

theorem simplify_expression (hab : a ≠ 0) (hbb : b ≠ 0) (hc : ab - c² ≠ 0) :
  ((a^2 - b^2) / (a * b) - (ab - b^2) / (a * b - c^2)) = (a / b + 1) :=
by
  sorry

end simplify_expression_l45_45387


namespace BM_eq_DM_l45_45267

-- Definitions for the given problem
variables (A B C D E M : Point)
variables (circle : Circle)
variables (inscribed_quadrilateral : (A ∈ circle) ∧ (B ∈ circle) ∧ (C ∈ circle) ∧ (D ∈ circle))
variables (diameter_AC : diameter circle A C)
variables (E_on_BC : E ∈ Segment B C)
variables (angle_equality : ∠DAC = ∠EAB)
variables (midpoint_M : midpoint M E C)

-- The theorem to be proved
theorem BM_eq_DM : distance B M = distance D M := by
  sorry

end BM_eq_DM_l45_45267


namespace S_n_formula_l45_45181

-- Define S_n as described in the problem
noncomputable def S (n : ℕ) : ℕ :=
∑ i in Finset.range (n - 1),
    ∑ j in Finset.range i.succ.succ \ Finset.range i.succ,
      i * j

-- Proof statement asserting the equality
theorem S_n_formula (n : ℕ) (hn : 0 < n) :
  S n = (n * (n + 1) * (n - 1) * (3 * n + 2)) / 24 :=
sorry

end S_n_formula_l45_45181


namespace AliClosetCapacity_l45_45015

-- Definitions based on the conditions
def NoahClosetsCapacity : ℕ := 100
def FactorNoahToAli : ℕ := 4

-- Inferred definitions based on conditions
def OneNoahClosetCapacity := NoahClosetsCapacity / 2

-- The theorem to be proved
theorem AliClosetCapacity : ℕ :=
  OneNoahClosetCapacity * FactorNoahToAli = 200 :=
by 
  sorry

end AliClosetCapacity_l45_45015


namespace prob_red_ball_is_three_fifths_l45_45773

def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3
def red_balls : ℕ := total_balls - (yellow_balls + green_balls)
def total_probability : ℚ := 1
def probability_of_red_ball : ℚ := red_balls / total_balls

theorem prob_red_ball_is_three_fifths :
  probability_of_red_ball = 3 / 5 :=
begin
  sorry
end

end prob_red_ball_is_three_fifths_l45_45773


namespace Milly_study_ratio_l45_45828

theorem Milly_study_ratio (G : ℤ) (S : ℤ) :
  (60 + G + S = 135) ∧ (S = (60 + G) / 2) →
  (G / 60 = 1 / 2) :=
by {
  intro h,
  cases h with h1 h2,
  sorry  -- Proof to be provided
}

end Milly_study_ratio_l45_45828


namespace ob_perp_df_oc_perp_de_oh_perp_mn_l45_45332

variables {A B C D E F M N H O : Point}
variables (triangle_ABC : is_triangle A B C)
variables (circumcenter_O : is_circumcenter ABC O)
variables (altitude_AD : is_altitude A D B C)
variables (altitude_BE : is_altitude B E A C)
variables (altitude_CF : is_altitude C F A B)
variables (intersection_H : intersection_altitudes H A D B E C F)
variables (ED_intersects_AB : intersection E D A B M)
variables (FD_intersects_AC : intersection F D A C N)

-- Proof of \( OB \perp DF \)
theorem ob_perp_df (h1 : OB ⊥ DF) : OB ⊥ DF :=
sorry

-- Proof of \( OC \perp DE \)
theorem oc_perp_de (h2 : OC ⊥ DE) : OC ⊥ DE :=
sorry

-- Proof of \( OH \perp MN \)
theorem oh_perp_mn (h3 : OH ⊥ MN) : OH ⊥ MN :=
sorry

end ob_perp_df_oc_perp_de_oh_perp_mn_l45_45332


namespace problem_inequality_l45_45920

/-- Given the properties of logarithms, exponentiation, and their values, show the inequality. -/
theorem problem_inequality : 
  log 4 0.3 < 0.4 ^ 3 ∧ 0.4 ^ 3 < 3 ^ 0.4 :=
by
  -- We need to show that log_4(0.3) < 0.4^3 and that 0.4^3 < 3^0.4.
  sorry

end problem_inequality_l45_45920


namespace no_entangled_two_digit_numbers_l45_45973

theorem no_entangled_two_digit_numbers :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 → 10 * a + b ≠ 2 * (a + b ^ 3) :=
by
  intros a b h
  rcases h with ⟨ha1, ha9, hb9⟩
  sorry

end no_entangled_two_digit_numbers_l45_45973


namespace solution_set_equivalence_l45_45715

variable (a b c : ℝ)

theorem solution_set_equivalence 
  (h1 : ∀ x : ℝ, -2 < x ∧ x < 1 ↔ ax^2 + bx + c > 0)
  (h2 : ax^2 + bx + c = 0 → x = -2 ∨ x = 1)
  (h3 : a < 0) 
  (b_eq_a : b = a) 
  (c_eq_neg2a : c = -2a) 
: (∀ x : ℝ, -1 < x ∧ x < (1 / 2) ↔ cx^2 - bx + a < 0) :=
sorry

end solution_set_equivalence_l45_45715


namespace twenty_four_is_eighty_percent_of_what_number_l45_45091

theorem twenty_four_is_eighty_percent_of_what_number (x : ℝ) (hx : 24 = 0.8 * x) : x = 30 :=
  sorry

end twenty_four_is_eighty_percent_of_what_number_l45_45091


namespace problem_equiv_proof_l45_45128

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Define the set A based on the given condition
def A : Set ℝ := { x | x^2 + x - 2 ≤ 0 }

-- Define the set B based on the given condition
def B : Set ℝ := { y | ∃ x : ℝ, x ∈ A ∧ y = Real.log (x + 3) / Real.log 2 }

-- Define the complement of B in the universal set U
def complement_B : Set ℝ := { y | y < 0 ∨ y ≥ 2 }

-- Define the set C that is the intersection of A and complement of B
def C : Set ℝ := A ∩ complement_B

-- State the theorem we need to prove
theorem problem_equiv_proof : C = { x | -2 ≤ x ∧ x < 0 } :=
sorry

end problem_equiv_proof_l45_45128


namespace silver_status_families_l45_45070

theorem silver_status_families 
  (goal : ℕ) 
  (remaining : ℕ) 
  (bronze_families : ℕ) 
  (bronze_donation : ℕ) 
  (gold_families : ℕ) 
  (gold_donation : ℕ) 
  (silver_donation : ℕ) 
  (total_raised_so_far : goal - remaining = 700)
  (amount_raised_by_bronze : bronze_families * bronze_donation = 250)
  (amount_raised_by_gold : gold_families * gold_donation = 100)
  (amount_raised_by_silver : 700 - 250 - 100 = 350) :
  ∃ (s : ℕ), s * silver_donation = 350 ∧ s = 7 :=
by
  sorry

end silver_status_families_l45_45070


namespace min_value_y_l45_45458

theorem min_value_y : ∀ (x : ℝ), x^2 + 14 * x + 10 ≥ -39 :=
begin
  sorry
end

end min_value_y_l45_45458


namespace closest_integer_to_cbrt_250_l45_45558

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45558


namespace side_length_rhombus_l45_45425

-- Define the geometry of the problem
variable (ABCD : Type) [rhombus ABCD]
variable (A B C D S M N K L F P R Q : ABCD)
variable (O : midpoint A C) (h : ℝ) (d1 d2 : ℝ) (rB : ℝ)

-- Conditions
axiom pyramid_with_apex_S : apex S ABCD
axiom lateral_faces_inclined_60_deg : ∀ face, lateral_face face ABCD S → inclined_at angle 60 face ABCD
axiom midpoint_def : ∀ {X Y Z}, midpoint X Y Z → midpoint_property Z X Y
axiom parallelepiped_base_MNKL : rectangular_parallelepiped_base M N K L
axiom top_face_edges_PRQF_intersect_pyramid : ∀ X Y, top_face_edge X Y P Q R F → intersects_lateral_edges_pyramid X Y ABCD S
axiom polyhedron_volume_12_sqrt_3 : polyhedron_volume M N K L F P R Q = 12 * sqrt 3
axiom radius_inscribed_circle_2_4 : ∀ inscribed_circle, radius inscribed_circle = 2.4

-- Prove the side length of rhombus ABCD
theorem side_length_rhombus :
  find_side_length A B C D = 5 :=
begin
  -- Proof omitted
  sorry
end

end side_length_rhombus_l45_45425


namespace min_max_difference_l45_45965

-- Define the board size and movement properties of the rook
def valid_traversal (n : ℕ) (traverse : ℕ → ℕ × ℕ) :=
  (∀ m, 1 ≤ m ∧ m ≤ n^2 → ∃ (i j : ℕ), traverse m = (i, j) ∧ 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n) ∧
  (∀ m, 1 ≤ m < n^2 → ∥ (traverse (m + 1)).fst - (traverse m).fst ∥ + ∥ (traverse (m + 1)).snd - (traverse m).snd ∥ = 1)

-- Define the maximum difference M between numbers of adjacent squares
def max_difference (n : ℕ) (traverse : ℕ → ℕ × ℕ) : ℕ :=
  nat.max (∥ (traverse (id+1)).fst - (traverse id).fst ∥ + ∥ (traverse (id+1)).snd - (traverse id).snd ∥)

-- Problem to find the minimum possible value of M
theorem min_max_difference (n : ℕ) (traverse : ℕ → ℕ × ℕ) (h : valid_traversal n traverse) :
  max_difference n traverse = 2 * n - 1 :=
sorry

end min_max_difference_l45_45965


namespace log10_two_bound_l45_45452

theorem log10_two_bound (h1 : 10^3 = 1000) (h2 : 10^4 = 10000)
  (h3 : 2^9 = 512) (h4 : 2^11 = 2048) : log10 2 < 4 / 11 :=
sorry

end log10_two_bound_l45_45452


namespace closest_integer_to_cbrt_250_l45_45563

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45563


namespace winning_team_points_difference_l45_45630

variables (P1 Ptotal X : ℕ)

def losing_team_points_first_quarter : ℕ := 10

def winning_team_points_first_quarter : ℕ := 2 * losing_team_points_first_quarter

def winning_team_points_second_quarter : ℕ := winning_team_points_first_quarter + X

def winning_team_points_third_quarter : ℕ := winning_team_points_second_quarter + 20

def winning_team_total_points : ℕ := winning_team_points_first_quarter + X + 20 + winning_team_points_second_quarter

theorem winning_team_points_difference :
  winning_team_points_first_quarter = 20 →
  winning_team_total_points = 80 →
  winning_team_points_second_quarter = 10 :=
by {
  intros h1 h2,
  sorry
}

end winning_team_points_difference_l45_45630


namespace daniel_utility_equation_solution_l45_45645

theorem daniel_utility_equation_solution (t : ℚ) :
  t * (10 - t) = (4 - t) * (t + 4) → t = 8 / 5 := by
  sorry

end daniel_utility_equation_solution_l45_45645


namespace contributions_before_john_l45_45749

theorem contributions_before_john (n : ℕ) (A : ℚ) 
  (h1 : 1.5 * A = 75)
  (h2 : (n * A + 225) / (n + 1) = 75) : n = 6 :=
by {
  sorry
}

end contributions_before_john_l45_45749


namespace sequence_formula_l45_45888

open Nat

def a : ℕ → ℤ
| 0     => 0  -- Defining a(0) though not used
| 1     => 1
| (n+2) => 3 * a (n+1) + 2^(n+2)

theorem sequence_formula (n : ℕ) (hn : n ≥ 1) :
  a n = 5 * 3^(n-1) - 2^(n+1) :=
by
  sorry

end sequence_formula_l45_45888


namespace problem_to_prove_l45_45356

theorem problem_to_prove
  (a b c : ℝ)
  (h1 : a + b + c = -3)
  (h2 : a * b + b * c + c * a = -10)
  (h3 : a * b * c = -5) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = 70 :=
by
  sorry

end problem_to_prove_l45_45356


namespace closest_integer_to_cube_root_of_250_l45_45483

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45483


namespace limit_of_difference_quotient_l45_45754

theorem limit_of_difference_quotient (f : ℝ → ℝ) (h : Differentiable ℝ f) : 
  (lim (λ Δx : ℝ, (f (1 + 3 * Δx) - f 1) / (3 * Δx)) (𝓝 0)) = deriv f 1 :=
sorry

end limit_of_difference_quotient_l45_45754


namespace find_C_monthly_income_l45_45059

noncomputable def C_monthly_income (A_monthly_income B_monthly_income C_monthly_ratio annual_income_B monthly_income_B : ℝ) : ℝ := 
  let C := monthly_income_B / C_monthly_ratio 
  C

theorem find_C_monthly_income :
  let A_monthly_income := 436800.0000000001 / 12 in
  let B_monthly_income := 16960 in
  let C_monthly_ratio := 1 + 12 / 100 in
  C_monthly_income A_monthly_income B_monthly_income C_monthly_ratio 436800.0000000001 16960 = 15142.857142857143 :=
by
  sorry

end find_C_monthly_income_l45_45059


namespace cost_of_six_burritos_and_seven_sandwiches_l45_45639

variable (b s : ℝ)
variable (h1 : 4 * b + 2 * s = 5.00)
variable (h2 : 3 * b + 5 * s = 6.50)

theorem cost_of_six_burritos_and_seven_sandwiches : 6 * b + 7 * s = 11.50 :=
  sorry

end cost_of_six_burritos_and_seven_sandwiches_l45_45639


namespace binomial_12_3_equals_220_l45_45190

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l45_45190


namespace number_of_pairs_l45_45702

-- Define the special operation ⊕ based on the given conditions
def op (m n : ℕ) : ℕ :=
  if m % 2 = n % 2 then m + n else m * n

-- Prove that the number of pairs (a, b) such that a ⊕ b = 12 is 15.
theorem number_of_pairs : {p : ℕ × ℕ | op p.1 p.2 = 12 ∧ p.1 > 0 ∧ p.2 > 0}.toFinset.card = 15 := sorry

end number_of_pairs_l45_45702


namespace train_speed_is_126_kmh_l45_45971

noncomputable def train_speed_proof : Prop :=
  let length_meters := 560 / 1000           -- Convert length to kilometers
  let time_hours := 16 / 3600               -- Convert time to hours
  let speed := length_meters / time_hours   -- Calculate the speed
  speed = 126                               -- The speed should be 126 km/h

theorem train_speed_is_126_kmh : train_speed_proof := by 
  sorry

end train_speed_is_126_kmh_l45_45971


namespace increase_amplitude_increase_period_l45_45918

-- Defining the period of a simple pendulum
def period (L : ℝ) (g : ℝ) : ℝ := 2 * Real.pi * Real.sqrt (L / g)

-- Stating the theorem: An increase in amplitude leads to an increase in period
theorem increase_amplitude_increase_period (L g : ℝ) (hL0 : 0 < L) (hg0 : 0 < g) :
  ∃ (amplitude_increases : Prop), amplitude_increases → period (L + 1) g > period L g := 
sorry

end increase_amplitude_increase_period_l45_45918


namespace min_value_x_plus_9_div_x_l45_45302

theorem min_value_x_plus_9_div_x (x : ℝ) (hx : x > 0) : x + 9 / x ≥ 6 := by
  -- sorry indicates that the proof is omitted.
  sorry

end min_value_x_plus_9_div_x_l45_45302


namespace solve_z_solutions_l45_45677

noncomputable def z_solutions (z : ℂ) : Prop :=
  z ^ 6 = -16

theorem solve_z_solutions :
  {z : ℂ | z_solutions z} = {2 * Complex.I, -2 * Complex.I} :=
by {
  sorry
}

end solve_z_solutions_l45_45677


namespace train_speed_in_kmph_l45_45972

def length_of_train : ℝ := 60
def time_taken : ℝ := 1.4998800095992322
def conversion_factor : ℝ := (1000 / 3600)

theorem train_speed_in_kmph : (length_of_train / time_taken) * conversion_factor = 11.112 :=
by
  -- Proof goes here
  sorry

end train_speed_in_kmph_l45_45972


namespace shaded_area_equilateral_l45_45161

theorem shaded_area_equilateral (s : ℝ) (h : s = 10) : 
  ∃ a b c, a * Real.pi - b * Real.sqrt c = 25 * Real.pi ∧ a + b + c = 25 := 
by
  use 25, 0, 0
  simp
  sorry

end shaded_area_equilateral_l45_45161


namespace closest_integer_to_cbrt_250_l45_45561

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45561


namespace tree_has_n_minus_1_edges_l45_45258

theorem tree_has_n_minus_1_edges (G : Graph) (n : ℕ) (hn : n ≥ 1) 
  (h1 : G.numVertices = n) 
  (h2 : ∀ {v : G.Vertex}, ¬G.Adj v v) 
  (h3 : ∀ {u v : G.Vertex}, u ≠ v → ∃! p : G.Path u v, true) : 
  G.numEdges = n - 1 := 
sorry

end tree_has_n_minus_1_edges_l45_45258


namespace mode_of_student_scores_l45_45320

def student_scores : List ℕ := [61, 62, 71, 78, 85, 85, 92, 96]

theorem mode_of_student_scores : (student_scores.toMultiset.mode) = 85 :=
by
  sorry

end mode_of_student_scores_l45_45320


namespace find_vector_BC_l45_45704

structure Point2D where
  x : ℝ
  y : ℝ

def A : Point2D := ⟨0, 1⟩
def B : Point2D := ⟨3, 2⟩
def AC : Point2D := ⟨-4, -3⟩

def vector_add (p1 p2 : Point2D) : Point2D := ⟨p1.x + p2.x, p1.y + p2.y⟩
def vector_sub (p1 p2 : Point2D) : Point2D := ⟨p1.x - p2.x, p1.y - p2.y⟩

def C : Point2D := vector_add A AC
def BC : Point2D := vector_sub C B

theorem find_vector_BC : BC = ⟨-7, -4⟩ := by
  sorry

end find_vector_BC_l45_45704


namespace frustum_lateral_surface_area_l45_45607

/--
A frustum of a right circular cone is formed by cutting a small cone off of the top of a larger cone.
The frustum has a lower base radius of 6 inches, an upper base radius of 3 inches, and a height of 4 inches.
Prove that the lateral surface area of the frustum is 45 * π square inches.
-/
theorem frustum_lateral_surface_area (r_2 r_1 h : ℝ) 
  (h_r2 : r_2 = 6) 
  (h_r1 : r_1 = 3) 
  (h : h = 4) : 
  (lateral_surface_area_of_frustum r_2 r_1 h) = 45 * Real.pi := 
sorry

end frustum_lateral_surface_area_l45_45607


namespace coefficient_of_x_squared_in_expansion_l45_45661

theorem coefficient_of_x_squared_in_expansion :
    let expr := (1 + (1 : ℚ) / x + (1 : ℚ) / x^2) * (1 + x^2)^5 in
    expr.coeff 2 = 15 :=
by sorry

end coefficient_of_x_squared_in_expansion_l45_45661


namespace find_angle_BAC_l45_45399

-- Define the scalene acute triangle with vertices A, B, and C.
structure Triangle :=
(A B C : Point)
(sc_def : ¬(A = B ∨ B = C ∨ A = C)) -- Scalene condition
(acute_def : (angle A B C < π / 2) ∧ (angle B C A < π / 2) ∧ (angle C A B < π / 2))

-- Define the orthocenter H, intersecting altitudes of triangle ABC.
def orthocenter (T : Triangle) : Point :=
sorry -- Definition of the orthocenter for triangle T

-- Define the circumcenter O of triangle BHC.
def circumcenter_BHC (B C H : Point) : Point :=
sorry -- Definition of the circumcenter for triangle BHC

-- Define the incenter I of triangle ABC, lying on segment OA.
def incenter_on_OA (T : Triangle) (O : Point) : Point :=
sorry -- Definition of I on segment OA

-- Define the conditions that we need.
variables {T : Triangle}
(H := orthocenter T) -- The orthocenter of triangle T (H)
(O := circumcenter_BHC T.B T.C H) -- Circumcenter of triangle BHC
(I := incenter_on_OA T O) -- Incenter I of triangle T at segment OA

-- The theorem to find angle BAC.
theorem find_angle_BAC (T : Triangle) (H : Point) (O : Point) (I : Point) :
  I ∈ segment O T.A ∧ O = circumcenter_BHC T.B T.C H  → angle T.A T.B T.C  = π / 3 :=
by
  intro h
  sorry

end find_angle_BAC_l45_45399


namespace closest_cube_root_of_250_l45_45472

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45472


namespace minimum_value_of_expression_l45_45238

theorem minimum_value_of_expression :
  ∀ x y : ℝ, x^2 - x * y + y^2 ≥ 0 :=
by
  sorry

end minimum_value_of_expression_l45_45238


namespace avery_shirts_count_l45_45168

theorem avery_shirts_count {S : ℕ} (h_total : S + 2 * S + S = 16) : S = 4 :=
by
  sorry

end avery_shirts_count_l45_45168


namespace functional_relationship_maximum_profit_value_l45_45970

-- Define the problem parameters
def cost_per_item := 15
def original_selling_price := 20
def original_sales_volume := 100

-- Define the function y given x
noncomputable def profit_function (x : ℝ) (0 < x < 1) := 
  500 * (1 + 4*x - x^2 - 4*x^3)

-- Define the conditions 
def conditions (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Statement of the problems
theorem functional_relationship (x : ℝ) (h : conditions x) :
  profit_function x (h.left ∧ h.right) = 500 * (1 + 4*x - x^2 - 4*x^3) :=
sorry

theorem maximum_profit_value :
  ∃ x : ℝ, conditions x ∧ x = 1 / 2 ∧ profit_function x (by split; linarith) = 1125 :=
sorry

end functional_relationship_maximum_profit_value_l45_45970


namespace decreasing_condition_l45_45755

-- Definitions
def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x
def f_prime (x : ℝ) (a : ℝ) : ℝ := (1 / x) - a

-- Statement we want to prove
theorem decreasing_condition (a : ℝ) : (∀ x : ℝ, 1 < x → f_prime x a ≤ 0) ↔ (1 ≤ a) :=
by
  -- the proof goes here
  sorry

end decreasing_condition_l45_45755


namespace incorrect_statement_C_l45_45268

theorem incorrect_statement_C
  (x : ℝ)
  (y : ℤ)
  (h : y = (x - ⌊x⌋) / x) :
  ¬ ∀ (x : ℝ), x < 0 → ∃ (y : ℤ), y = (x - ⌊x⌋) / x ∧ y ∈ ℤ :=
sorry

end incorrect_statement_C_l45_45268


namespace log_2_a10_l45_45765

-- Given conditions
variable {a : ℕ → ℝ} 
variable {r a1 : ℝ} 
variable (h_pos : ∀ n, 0 < a n)
variable (h_geom : ∀ n, a (n + 1) = r * a n)
variable (h2_18 : a 2 * a 18 = 16)

noncomputable def a_n (n : ℕ) : ℝ := a1 * r^n

-- The theorem we need to prove
theorem log_2_a10 : log 2 (a 10) = 2 :=
by
  sorry

end log_2_a10_l45_45765


namespace proof_problem_l45_45808

variable (f : ℝ → ℝ)
variable (f'' : ℝ → ℝ) (A B : ℝ) (f''_def : ∀ x, derivative 2 f x = f'' x)

-- Given conditions
axiom cond1 : ∀ x, x * f'' x - 2 * f x > 0
axiom obtuse_C : ∃ C : ℝ, C > π / 2 ∧ C < π ∧ (cos A > sin B ∧ sin B > 0)

-- Proof goal
theorem proof_problem : f (cos A) * (sin B)^2 > f (sin B) * (cos A)^2 :=
by
  sorry

end proof_problem_l45_45808


namespace reflect_across_x_axis_l45_45779

-- Definitions for the problem conditions
def initial_point : ℝ × ℝ := (-2, 1)
def reflected_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- The statement to be proved
theorem reflect_across_x_axis :
  reflected_point initial_point = (-2, -1) :=
  sorry

end reflect_across_x_axis_l45_45779


namespace collinear_points_l45_45605

open Real

noncomputable def convex_quadrilateral := {A B C D : Type} 
  [convex A B C D] 
  (AB_x : A → B → ℝ) (BC_x : B → C → ℝ) (CD_x : C → D → ℝ) (DA_x : D → A → ℝ)
  (area : ∀ (s: ℝ), s = 2 * t) 
  (no_parallel_sides : ¬(is_parallel A B C D))

noncomputable def point_on_segment := {P : Type} 
  (line_segment : Type) [segment P line_segment]

axiom point_on_same_side {P : Type} 
  [on_segment : point_on_segment P ℝ]
  (line : Type) [side P line]: 
  ∀ (C : Type) (t : ℝ), P.c_str = t

noncomputable def triangle_area (a b c : Type) [area_eq : triangle_area a b c] :=
  ∀ (s : ℝ), s = t

theorem collinear_points {A B C D P1 P2 P3 P4 : Type} 
  [convex_quadrilateral A B C D] 
  [point_on_segment P1 CD]
  [point_on_segment P2 AD]
  [point_on_segment P3 AB]
  [point_on_segment P4 BC]
  [triangle_area A B P1 t]
  [triangle_area B C P2 t]
  [triangle_area C D P3 t]
  [triangle_area D A P4 t]
  (area_t : ℝ) 
  : collinear P1 P2 P3 P4 :=
sorry

end collinear_points_l45_45605


namespace binomial_12_3_equals_220_l45_45191

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l45_45191


namespace smallest_palindrome_after_8765_l45_45461

-- Define the concept of a palindrome
def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

-- Main theorem statement
theorem smallest_palindrome_after_8765 : ∃ x : ℕ, 0 < x ∧ x + 8765 = 8778 ∧ is_palindrome (x + 8765) :=
by
  -- statement only, no proof
  sorry

end smallest_palindrome_after_8765_l45_45461


namespace number_of_towers_equal_4200_l45_45138

/- Given conditions -/
def red_cubes := 3
def blue_cubes := 3
def green_cubes := 4

def cubes_total := red_cubes + blue_cubes + green_cubes
def cubes_to_use := cubes_total - 1 -- One cube is left out
def height_of_tower := 9

/- Hypothesis -/
-- Ensure we have total 10 cubes and we need to form a tower of exactly 9 cubes
example : cubes_total = 10 := by simp [cubes_total]

-- Theorem to prove: Total towers that can be built if exactly one cube is left out == 4200
theorem number_of_towers_equal_4200 : 
  ∑ (r in finset.range 1 red_cubes+1) (b in finset.range 1 blue_cubes+1),
      (if r == 3 && b == 3 then nat.factorial height_of_tower / (nat.factorial r * nat.factorial b * nat.factorial (cubes_to_use - r - b))
      else sorry) = 
  4200 := 
sorry

end number_of_towers_equal_4200_l45_45138


namespace increase_probability_sum_13_l45_45606

theorem increase_probability_sum_13 (T : Finset ℕ) (m : ℕ) (H : T = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}) :
  (Finset.Prob (fun p => (p.1 + p.2 = 13)) (T.erase 12).powerset) >
  (Finset.Prob (fun p => (p.1 + p.2 = 13)) T.powerset) :=
sorry

end increase_probability_sum_13_l45_45606


namespace find_coefficient_sum_l45_45686

theorem find_coefficient_sum (a : Fin 2016 -> ℝ) :
  let p := (x : ℝ) ↦ (x - 2) ^ 2015
  (Polynomial.of_fn a).eval x = p x →
  (Finset.range 2016).sum (λ i, (i + 1) * a i) = 2015 :=
by
  sorry

end find_coefficient_sum_l45_45686


namespace angle_CED_is_135_l45_45899

noncomputable def degree_measure_angle_CED (A B D C E : Point) (r : ℝ) : ℝ :=
  if h1 : centered_at_circle A r
     ∧ centered_at_circle B (r / 2)
     ∧ (intersects (circle A r) (circle B (r / 2)) 2)
     ∧ (B ∈ diameter_extended_from A)
     ∧ centered_at_circle A (3 * r / 2)
     ∧ (intersects_line A B D)
     ∧ linear_alignment A B D
     ∧ (D beyond B)
     ∧ (intersects_points_circle C E (circle_with_center A (3 * r / 2)))
     ∧ (E beyond D)
     ∧ (angle_measure C E D = 135) then 135 else 0

theorem angle_CED_is_135 (A B D C E : Point) (r : ℝ) 
  (h1 : centered_at_circle A r)
  (h2 : centered_at_circle B (r / 2))
  (h3 : intersects (circle A r) (circle B (r / 2)) 2)
  (h4 : B ∈ diameter_extended_from A)
  (h5 : centered_at_circle A (3 * r / 2))
  (h6 : intersects_line A B D)
  (h7 : linear_alignment A B D)
  (h8 : D beyond B)
  (h9 : intersects_points_circle C E (circle_with_center A (3 * r / 2)))
  (h10 : E beyond D) : 
  angle_measure C E D = 135 := 
sorry

end angle_CED_is_135_l45_45899


namespace simplify_neg_fractional_exponent_l45_45916

theorem simplify_neg_fractional_exponent : (- (1 : ℝ) / 125) ^ (-2 / 3) = 25 := by
  sorry

end simplify_neg_fractional_exponent_l45_45916


namespace no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l45_45003

-- Define the context for real numbers and the main property P.
def property_P (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + f (x + 2) ≤ 2 * f (x + 1)

-- For part (1)
theorem no_exp_function_satisfies_P (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, f x = a^x) ∧ property_P f :=
sorry

-- Define the context for natural numbers, d(x), and main properties related to P.
def d (f : ℕ → ℕ) (x : ℕ) : ℕ := f (x + 1) - f x

-- For part (2)(i)
theorem d_decreasing_nonnegative (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∀ x : ℕ, d f (x + 1) ≤ d f x ∧ d f x ≥ 0 :=
sorry

-- For part (2)(ii)
theorem exists_c_infinitely_many (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∃ c : ℕ, 0 ≤ c ∧ c ≤ d f 1 ∧ ∀ N : ℕ, ∃ n : ℕ, n > N ∧ d f n = c :=
sorry

end no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l45_45003


namespace pyramid_side_length_difference_l45_45372

theorem pyramid_side_length_difference (x : ℕ) (h1 : 1 + x^2 + (x + 1)^2 + (x + 2)^2 = 30) : x = 2 :=
by
  sorry

end pyramid_side_length_difference_l45_45372


namespace cuboid_volume_l45_45039

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 2) (h2 : b * c = 6) (h3 : a * c = 9) : a * b * c = 6 := by
  sorry

end cuboid_volume_l45_45039


namespace prism_edges_l45_45751

-- Define the number of faces of the prism
def num_faces : ℕ := 7

-- Define the total number of edges for a prism with given faces
theorem prism_edges (n : ℕ) (hn : n = num_faces) : 
  if n = 7 then 15 else 0 :=
by {
  sorry
}

end prism_edges_l45_45751


namespace binomial_12_3_eq_220_l45_45208

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l45_45208


namespace f_properties_l45_45004

noncomputable def f : ℚ × ℚ → ℚ := sorry

theorem f_properties :
  (∀ (x y z : ℚ), f (x*y, z) = f (x, z) * f (y, z)) →
  (∀ (x y z : ℚ), f (z, x*y) = f (z, x) * f (z, y)) →
  (∀ (x : ℚ), f (x, 1 - x) = 1) →
  (∀ (x : ℚ), f (x, x) = 1) ∧
  (∀ (x : ℚ), f (x, -x) = 1) ∧
  (∀ (x y : ℚ), f (x, y) * f (y, x) = 1) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end f_properties_l45_45004


namespace longest_chord_line_through_point_l45_45127

theorem longest_chord_line_through_point
  (P : ℝ × ℝ)
  (hP : P = (2, 1))
  (C : ℝ × ℝ)
  (hC : C = (1, -2))
  (k : ℝ)
  (hk : k = 3)
  (line_eq : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop)
  (hline_eq : line_eq P C k) : 
  ∃ line_eq, line_eq = (y - 1 = 3 * (x - 2)) :=
begin
  sorry
end

end longest_chord_line_through_point_l45_45127


namespace min_value_m_n_l45_45140

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem min_value_m_n 
  (a : ℝ) (m n : ℝ)
  (h_a_pos : a > 0) (h_a_ne1 : a ≠ 1)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_line_eq : 2 * m + n = 1) :
  m + n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_m_n_l45_45140


namespace symmetric_points_on_ellipse_l45_45280

theorem symmetric_points_on_ellipse (m : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), (x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
                        (x1^2 / 4 + y1^2 / 3 = 1) ∧ 
                        (x2^2 / 4 + y2^2 / 3 = 1) ∧ 
                        (∃ (M : ℝ × ℝ), M = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
                                        (snd M = 4 * (fst M) + m))) ↔ 
  (-2 * Real.sqrt 3 / 13 < m ∧ m < 2 * Real.sqrt 3 / 13) := sorry

end symmetric_points_on_ellipse_l45_45280


namespace width_rectangular_box_5_cm_l45_45974

theorem width_rectangular_box_5_cm 
  (W : ℕ)
  (h_dim_wooden_box : (8 * 10 * 6 * 100 ^ 3) = 480000000) -- dimensions of the wooden box in cm³
  (h_dim_rectangular_box : (4 * W * 6) = (24 * W)) -- dimensions of the rectangular box in cm³
  (h_max_boxes : 4000000 * (24 * W) = 480000000) -- max number of boxes that fit in the wooden box
: 
  W = 5 := 
by
  sorry

end width_rectangular_box_5_cm_l45_45974


namespace average_visitors_per_day_l45_45119

theorem average_visitors_per_day (avg_sunday_visitors : ℕ) (avg_otherday_visitors : ℕ) (days_in_month : ℕ)
  (starts_with_sunday : Bool) (num_sundays : ℕ) (num_otherdays : ℕ)
  (h1 : avg_sunday_visitors = 510)
  (h2 : avg_otherday_visitors = 240)
  (h3 : days_in_month = 30)
  (h4 : starts_with_sunday = true)
  (h5 : num_sundays = 5)
  (h6 : num_otherdays = 25) :
  (num_sundays * avg_sunday_visitors + num_otherdays * avg_otherday_visitors) / days_in_month = 285 :=
by 
  sorry

end average_visitors_per_day_l45_45119


namespace four_numbers_divisible_by_2310_in_4_digit_range_l45_45978

/--
There exist exactly 4 numbers within the range of 1000 to 9999 that are divisible by 2310.
-/
theorem four_numbers_divisible_by_2310_in_4_digit_range :
  ∃ n₁ n₂ n₃ n₄,
    1000 ≤ n₁ ∧ n₁ ≤ 9999 ∧ n₁ % 2310 = 0 ∧
    1000 ≤ n₂ ∧ n₂ ≤ 9999 ∧ n₂ % 2310 = 0 ∧ n₁ < n₂ ∧
    1000 ≤ n₃ ∧ n₃ ≤ 9999 ∧ n₃ % 2310 = 0 ∧ n₂ < n₃ ∧
    1000 ≤ n₄ ∧ n₄ ≤ 9999 ∧ n₄ % 2310 = 0 ∧ n₃ < n₄ ∧
    ∀ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 2310 = 0 → (n = n₁ ∨ n = n₂ ∨ n = n₃ ∨ n = n₄) :=
by
  sorry

end four_numbers_divisible_by_2310_in_4_digit_range_l45_45978


namespace length_of_AE_l45_45324

/-- Defining the geometric setup of the pentagon -/
structure Pentagon (A B C D E : Type) :=
  (BC_len : B → C → ℝ)
  (CD_len : C → D → ℝ)
  (DE_len : D → E → ℝ)
  (angle_BCD : B → C → D → ℝ)
  (angle_B : A → B → C → D → E → ℝ)
  (angle_C : A → B → C → D → E → ℝ)
  (angle_D : A → B → C → D → E → ℝ)

/-- Specifying the properties given in the problem -/
def givenPentagon : Pentagon :=
{ BC_len := λ B C, 2,
  CD_len := λ C D, 2,
  DE_len := λ D E, 2,
  angle_BCD := λ B C D, 90,
  angle_B := λ A B C D E, 120,
  angle_C := λ A B C D E, 120,
  angle_D := λ A B C D E, 120
}

/--  Proving the length of segment AE -/
theorem length_of_AE : (A B C D E : Type) [pent : Pentagon A B C D E] → 
  pent.BC_len  B C = 2 →
  pent.CD_len  C D = 2 →
  pent.DE_len  D E = 2 →
  pent.angle_BCD B C D = 90 →
  pent.angle_B  A B C D E = 120 →
  pent.angle_C  A B C D E = 120 →
  pent.angle_D  A B C D E = 120 →
  AE = 4 :=
by
  sorry

end length_of_AE_l45_45324


namespace eight_painters_finish_in_required_days_l45_45795

/- Conditions setup -/
def initial_painters : ℕ := 6
def initial_days : ℕ := 2
def job_constant := initial_painters * initial_days

def new_painters : ℕ := 8
def required_days := 3 / 2

/- Theorem statement -/
theorem eight_painters_finish_in_required_days : new_painters * required_days = job_constant :=
sorry

end eight_painters_finish_in_required_days_l45_45795


namespace cos_double_angle_of_parallel_vectors_l45_45293

theorem cos_double_angle_of_parallel_vectors
  (α : ℝ)
  (a : ℝ × ℝ := (1/3, Real.tan α))
  (b : ℝ × ℝ := (Real.cos α, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  Real.cos (2 * α) = 1 / 9 := 
sorry

end cos_double_angle_of_parallel_vectors_l45_45293


namespace circles_intersect_l45_45882

-- Definitions of the two circles
def circle1 : set (ℝ × ℝ) := { p | (p.1 - 1)^2 + p.2^2 = 1 }
def circle2 : set (ℝ × ℝ) := { p | (p.1 + 1)^2 + (p.2 + 1)^2 = 2 }

-- Theorem that states the positional relationship
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 := 
sorry

end circles_intersect_l45_45882


namespace inequality_of_angles_l45_45642

variable (P Q R : Type) [Triangle P Q R]
variable (C I O : Type) [Circumcenter C] [Incenter I] [Orthocenter O]
variable (α β γ : ℝ) [Angle α] [Angle β] [Angle γ]

-- α, β, γ are angles defined as follows:
-- α = angle QCR
-- β = angle QIR
-- γ = angle QOR

axiom acute_angled_triangle : isAcute (triangle P Q R)
axiom angles_defined : α = ∠ QCR ∧ β = ∠ QIR ∧ γ = ∠ QOR

theorem inequality_of_angles (acute_angled_triangle : isAcute (triangle P Q R)) 
  (angles_defined : α = ∠ QCR ∧ β = ∠ QIR ∧ γ = ∠ QOR) :
  (1/α + 1/β + 1/γ) > (1/45) :=
  by 
    -- the proof steps would be included here, but we skip them with 'sorry'
    sorry

end inequality_of_angles_l45_45642


namespace sum_lt_two_l45_45331

noncomputable def a : ℕ → ℕ
| 0 := 0  -- we're noting that the sequence is 1-indexed
| 1 := 1
| (n+2) := (n+2) * a (n+1) / (n+1) + 2 * (n+2) * 3^(n)

-- Definition of bₙ as derived from aₙ
noncomputable def b (n : ℕ) := a (n + 1) / (n + 1)

-- Definition of dₙ (the term to be summed)
noncomputable def d (n : ℕ) := (2 * b n) / ((b n - 1) ^ 2)

-- Summation of terms of the sequence we are interested in
noncomputable def S (n : ℕ) := ∑ i in finset.range n, (d i)

-- The main proof statement
theorem sum_lt_two (n : ℕ) : S n < 2 :=
sorry

end sum_lt_two_l45_45331


namespace closest_integer_to_cube_root_of_250_l45_45495

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45495


namespace amc12a_2006_p24_l45_45359

theorem amc12a_2006_p24 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
by
  sorry

end amc12a_2006_p24_l45_45359


namespace stickers_per_student_l45_45012

theorem stickers_per_student : 
  ∀ (gold silver bronze total : ℕ), 
    gold = 50 →
    silver = 2 * gold →
    bronze = silver - 20 →
    total = gold + silver + bronze →
    total / 5 = 46 :=
by
  intros
  sorry

end stickers_per_student_l45_45012


namespace find_a_in_terms_of_x_l45_45643

theorem find_a_in_terms_of_x (a b x : ℤ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 2 * x) :
a = (7 * x + 5 * sqrt 6 * x) / 6 ∨ a = (7 * x - 5 * sqrt 6 * x) / 6 :=
by sorry

end find_a_in_terms_of_x_l45_45643


namespace max_abs_f_lower_bound_l45_45691

theorem max_abs_f_lower_bound (a b M : ℝ) (hM : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → abs (x^2 + a*x + b) ≤ M) : 
  M ≥ 1/2 :=
sorry

end max_abs_f_lower_bound_l45_45691


namespace strong_2013_l45_45963

theorem strong_2013 :
  ∃ x : ℕ, x > 0 ∧ (x ^ (2013 * x) + 1) % (2 ^ 2013) = 0 :=
sorry

end strong_2013_l45_45963


namespace train_crossing_time_l45_45944

theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_cross_platform : ℝ)
  (train_speed : ℝ := (train_length + platform_length) / time_to_cross_platform)
  (time_to_cross_signal_pole : ℝ := train_length / train_speed) :
  train_length = 300 ∧ platform_length = 1000 ∧ time_to_cross_platform = 39 → time_to_cross_signal_pole = 9 := by
  intro h
  cases h
  sorry

end train_crossing_time_l45_45944


namespace problem_statement_l45_45811

noncomputable def f : ℝ → ℝ :=
sorry  -- We assume the definition of f, as specified in the conditions, will be detailed later.

theorem problem_statement :
  (∀ x : ℝ, f (-x) = -f(x)) ∧ (∀ x : ℝ, f (x + 2) = -f(x)) ∧ (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f(x) = x + 1) →
  f 7.5 = -1.5 :=
by sorry

end problem_statement_l45_45811


namespace area_of_rectangle_l45_45622

theorem area_of_rectangle (w l : ℕ) (hw : w = 10) (hl : l = 2) : (w * l) = 20 :=
by
  sorry

end area_of_rectangle_l45_45622


namespace product_fractions_eq_l45_45179

theorem product_fractions_eq :
  (finset.range 50).prod (λ n, (n + 1 : ℚ) / (n + 5 : ℚ)) = 1 / 35 := 
sorry

end product_fractions_eq_l45_45179


namespace probability_in_A_l45_45733

noncomputable def area_Omega : ℝ := 4
noncomputable def area_A : ℝ := ∫ x in -π/3..π/3, real.cos x

theorem probability_in_A (Ω : set (ℝ × ℝ)) (A : set (ℝ × ℝ)) :
  Ω = {p : ℝ × ℝ | |p.1| ≤ sqrt 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ sqrt 2} →
  A = {p : ℝ × ℝ | -π/3 ≤ p.1 ∧ p.1 ≤ π/3 ∧ 0 ≤ p.2 ∧ p.2 ≤ real.cos p.1} →
  (area_A / area_Omega) = sqrt 3 / 4 :=
by
  intros hΩ hA
  rw [hΩ, hA]
  sorry

end probability_in_A_l45_45733


namespace log_sum_eq_one_l45_45211

theorem log_sum_eq_one (log_2 log_5 : ℝ) (h_log_2 : log_2 = Real.log 2) (h_log_5 : log_5 = Real.log 5) :
  let log_10 := log_2 + log_5 in
  (∑ k in Finset.range 21, Nat.choose 20 k * log_2 ^ (20 - k) * log_5 ^ k) = 1 :=
by 
  have h : log_10 = Real.log 10 := 
    calc log_10 = log_2 + log_5 : rfl 
           ... = Real.log 2 + Real.log 5 : by rw [h_log_2, h_log_5]
           ... = Real.log (2 * 5) : Real.log_mul 2 5
           ... = Real.log 10 : by norm_num
  sorry

end log_sum_eq_one_l45_45211


namespace possible_number_of_students_l45_45610

theorem possible_number_of_students (n : ℕ) 
  (h1 : n ≥ 1) 
  (h2 : ∃ k : ℕ, 120 = 2 * n + 2 * k) :
  n = 58 ∨ n = 60 :=
sorry

end possible_number_of_students_l45_45610


namespace solve_inequality_l45_45255

noncomputable def solution_set (a : ℝ) (x : ℝ) : set ℝ := {x | log a (|x-1| - 3) < 0}

theorem solve_inequality (a : ℝ) (x : ℝ) (h : 0 < a ∧ a < 1) : 
  solution_set a x = {x | x < -3 ∨ x > 5} := 
by
  sorry

end solve_inequality_l45_45255


namespace quotient_poly_div_l45_45242

def poly_div (p q : Polynomial ℤ) : Polynomial ℤ × Polynomial ℤ :=
  Polynomial.divMod p q

theorem quotient_poly_div (x : ℤ) :
  let p := 10 * Polynomial.X ^ 4 - 7 * Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 5 * Polynomial.X - 1
  let q := Polynomial.X - 1
  let (quotient, remainder) := poly_div p q
  quotient = 10 * Polynomial.X ^ 3 - 3 * Polynomial.X ^ 2 + 5 * Polynomial.X + 10 ∧ remainder = 9 := 
by
  sorry

end quotient_poly_div_l45_45242


namespace closest_integer_to_cube_root_of_250_l45_45489

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45489


namespace area_under_curve_l45_45992

def f (x : ℝ) : ℝ :=
if x ∈ Icc 0 1 then x^2
else if x ∈ Ioc 1 2 then 2 - x
else 0

theorem area_under_curve : ∫ x in 0..1, x^2 + ∫ x in 1..2, 2 - x = 5 / 6 :=
by
  sorry

end area_under_curve_l45_45992


namespace find_pairs_satisfying_eq_l45_45659

/-- All pairs (m, n) of positive integers satisfying the equation 
  (2^n - 1) * (2^n - 2) * (2^n - 4) * ... * (2^n - 2^(n-1)) = m! are 
  (1, 1) and (3, 2) -/
theorem find_pairs_satisfying_eq (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∏ i in (finset.range n), 2^n - 2^i) = nat.factorial m ↔ (m, n) ∈ {(1, 1), (3, 2)} :=
by
  sorry

end find_pairs_satisfying_eq_l45_45659


namespace new_length_of_rectangle_l45_45878

-- Given conditions
def original_length : ℝ := 18
def original_breadth : ℝ := 10
def new_breadth : ℝ := 7.2
def original_area : ℝ := original_length * original_breadth

-- Proof statement
theorem new_length_of_rectangle : 
  ∃ new_length : ℝ, original_area = new_length * new_breadth ∧ new_length = 25 :=
by 
  use 25
  split
  . exact original_area
  . exact 25

end new_length_of_rectangle_l45_45878


namespace number_of_shapes_after_4_folds_sum_of_areas_after_n_folds_l45_45103

-- Conditions based on the problem
def rectangular_paper : ℕ := 20 * 12  -- Initial area
def S1 : ℕ := 240  -- Area after first fold
def S2 : ℕ := 180  -- Area after second fold

-- Number of shapes after 4 folds
theorem number_of_shapes_after_4_folds : (number_of_shapes 4 = 5) :=
by
  -- Mathematically, paper folding logic gives 5 shapes after 4 folds
  sorry

-- Sum of areas after n folds
theorem sum_of_areas_after_n_folds (n : ℕ) : ∑ k in range(n), S k = 240 * (3 - (n + 3) / (2^n)) := 
by
  -- Sum calculation based on folding pattern
  sorry

end number_of_shapes_after_4_folds_sum_of_areas_after_n_folds_l45_45103


namespace ordered_triples_count_l45_45353

-- Defining the set T
def T : Set ℕ := { n | 1 ≤ n ∧ n ≤ 25 }

-- Defining the relation 'succ'
def succ (a b : ℕ) : Prop := (0 < a - b ∧ a - b ≤ 12) ∨ (b - a > 12)

-- Main theorem statement
theorem ordered_triples_count :
  let count := { (x, y, z) ∈ T × T × T | succ x y ∧ succ y z ∧ succ z x |.card
  count = 1950 :=
by
  sorry

end ordered_triples_count_l45_45353


namespace find_a_l45_45351

def polynomial_expansion (a : ℝ) (a_0 a_1 a_2 ... a_2018: ℝ) : Prop :=
  (1 - a) ^ 2018 = a_0 + a_1 + (a_2 * 1^2) + ... + (a_2018 * 1^2018)

def condition (a a_1 a_2 a_3 ... a_2018: ℝ) : Prop :=
  a_1 + 2 * a_2 + 3 * a_3 + ... + 2018 * a_2018 = 2018 * a

theorem find_a
(a a_0 a_1 a_2 ... a_2018: ℝ)
(h_poly : polynomial_expansion a a_0 a_1 a_2 ... a_2018)
(h_cond : condition a a_1 a_2 a_3 ... a_2018)
(h_a_ne_zero : a ≠ 0) :
a = 2 :=
by
  sorry

end find_a_l45_45351


namespace friends_gift_l45_45344

-- Define the original number of balloons and the final number of balloons
def original_balloons := 8
def final_balloons := 10

-- The main theorem: Joan's friend gave her 2 orange balloons.
theorem friends_gift : (final_balloons - original_balloons) = 2 := by
  sorry

end friends_gift_l45_45344


namespace cost_prices_l45_45617

variable {C1 C2 : ℝ}

theorem cost_prices (h1 : 0.30 * C1 - 0.15 * C1 = 120) (h2 : 0.25 * C2 - 0.10 * C2 = 150) :
  C1 = 800 ∧ C2 = 1000 := 
by
  sorry

end cost_prices_l45_45617


namespace intervals_of_monotonicity_and_min_value_l45_45722

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

theorem intervals_of_monotonicity_and_min_value : 
  (∀ x, (x < -1 → f x < f (x + 0.0001)) ∧ (x > -1 ∧ x < 3 → f x > f (x + 0.0001)) ∧ (x > 3 → f x < f (x + 0.0001))) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≥ f 2) :=
by
  sorry

end intervals_of_monotonicity_and_min_value_l45_45722


namespace largest_cube_edge_length_l45_45156

theorem largest_cube_edge_length (a : ℕ) : 
  (6 * a ^ 2 ≤ 1500) ∧
  (a * 15 ≤ 60) ∧
  (a * 15 ≤ 25) →
  a ≤ 15 :=
by
  sorry

end largest_cube_edge_length_l45_45156


namespace sam_possible_lunches_without_violation_l45_45989

def main_dishes := ["Burger", "Fish and Chips", "Pasta", "Vegetable Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Apple Pie", "Chocolate Cake"]

def valid_combinations := 
  (main_dishes.length * beverages.length * snacks.length) - 
  ((1 * if "Fish and Chips" ∈ main_dishes then 1 else 0) * if "Soda" ∈ beverages then 1 else 0 * snacks.length)

theorem sam_possible_lunches_without_violation : valid_combinations = 14 := by
  sorry

end sam_possible_lunches_without_violation_l45_45989


namespace contrapositive_of_implication_l45_45114

theorem contrapositive_of_implication (a : ℝ) (h : a > 0 → a > 1) : a ≤ 1 → a ≤ 0 :=
by
  sorry

end contrapositive_of_implication_l45_45114


namespace jane_played_8_rounds_l45_45317

variable (points_per_round : ℕ)
variable (end_points : ℕ)
variable (lost_points : ℕ)
variable (total_points : ℕ)
variable (rounds_played : ℕ)

theorem jane_played_8_rounds 
  (h1 : points_per_round = 10) 
  (h2 : end_points = 60) 
  (h3 : lost_points = 20)
  (h4 : total_points = end_points + lost_points)
  (h5 : total_points = points_per_round * rounds_played) : 
  rounds_played = 8 := 
by 
  sorry

end jane_played_8_rounds_l45_45317


namespace knights_probability_sum_l45_45085

theorem knights_probability_sum 
  (knights : Finset ℕ)
  (n : ℕ)
  (h_n : n = 30)
  (selected_knights : Finset ℕ)
  (h_selected_knights : selected_knights.card = 4) :
  let P := 1 - ((26 * 23 * 20 * 17) / 27405) in
  let P_simplified := 943 / 1023 in
  (P = P_simplified) ∧ ((P_simplified.num : ℕ) + (P_simplified.denom : ℕ) = 1966) :=
by
  sorry

end knights_probability_sum_l45_45085


namespace choose_30_5_eq_142506_l45_45600

theorem choose_30_5_eq_142506 : nat.choose 30 5 = 142506 := 
by sorry

end choose_30_5_eq_142506_l45_45600


namespace gcd_437_323_eq_19_l45_45994

theorem gcd_437_323_eq_19 : Int.gcd 437 323 = 19 := 
by 
  sorry

end gcd_437_323_eq_19_l45_45994


namespace polynomial_third_and_fourth_equal_l45_45419

theorem polynomial_third_and_fourth_equal (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p + q = 1)
  (h_eq : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = (8 : ℝ) / 11 :=
by
  sorry

end polynomial_third_and_fourth_equal_l45_45419


namespace good_divisors_n_l45_45822

def is_good_divisor (n d : ℕ) : Prop := d ∣ n ∧ d + 1 ∣ n

theorem good_divisors_n (n : ℕ) (h : 1 < n) :
  (∃ S : Finset ℕ, S.card > 0 ∧ (∀ d ∈ S, is_good_divisor n d)) →
  n = 2 ∨ n = 6 ∨ n = 12 := by
  sorry

end good_divisors_n_l45_45822


namespace number_of_girls_l45_45884

-- Given conditions
def ratio_girls_boys_teachers (girls boys teachers : ℕ) : Prop :=
  3 * (girls + boys + teachers) = 3 * girls + 2 * boys + 1 * teachers

def total_people (total girls boys teachers : ℕ) : Prop :=
  total = girls + boys + teachers

-- Define the main theorem
theorem number_of_girls 
  (k total : ℕ)
  (h1 : ratio_girls_boys_teachers (3 * k) (2 * k) k)
  (h2 : total_people total (3 * k) (2 * k) k)
  (h_total : total = 60) : 
  3 * k = 30 :=
  sorry

end number_of_girls_l45_45884


namespace closest_cube_root_l45_45505

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45505


namespace right_triangle_AC_l45_45777

theorem right_triangle_AC (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (h1 : ∠ C = 90)
  (h2 : dist (A,B) = 10)
  (h3 : dist (B,C) = 6)
  (h4 : ∠ C = 90)
  :
  dist(A,C) = 8 :=
sorry

end right_triangle_AC_l45_45777


namespace cosine_generator_plane_l45_45443

variables {P : Type*} {𝒮 : Type*} [plane P] [sphere 𝒮] (O1 O2 O3 : P) (O : P)
           (r R : ℝ) (ϕ : ℝ) (cone : Type*) [right_circular_cone cone]

-- Given conditions
axiom h1 : O1 ≠ O2 ∧ O1 ≠ O3 ∧ O2 ≠ O3  -- Distinct points of tangency on the plane
axiom h2 : touching (O1,O2) ∧ touching (O1,O3) ∧ touching (O2,O3)  -- Pairwise touching
axiom h3 : touching_cone_surface (O1,cone) ∧ touching_cone_surface (O2,cone) ∧ touching_cone_surface (O3,cone) -- Touching cone surface
axiom h4 : identical_spheres (O2,O3) -- Two identical spheres
axiom h5 : angle O2 O1 O3 = 150 -- Angle between points of tangency

-- The statement to prove
theorem cosine_generator_plane : cos ϕ = 1 / 7 := sorry

end cosine_generator_plane_l45_45443


namespace perfect_square_factors_count_l45_45299

theorem perfect_square_factors_count :
  let P := (2 ^ 12) * (3 ^ 14) * (5 ^ 16) * (7 ^ 9)
  in (count_positive_perfect_square_factors P) = 2520 := sorry

noncomputable def count_positive_perfect_square_factors (n : ℕ) : ℕ := sorry

end perfect_square_factors_count_l45_45299


namespace closest_integer_to_cube_root_of_250_l45_45480

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45480


namespace digital_root_distribution_l45_45250

theorem digital_root_distribution (n : ℕ) (h : n = 1000000000) :
  ∀ (d : ℕ), d ∈ {1, 2} → (finset.filter (λ x, x = d) (finset.image (λ x, 1 + (x - 1) % 9) (finset.range n))).card = n / 9 := 
by
  sorry

end digital_root_distribution_l45_45250


namespace age_of_student_who_left_l45_45403

variables
  (avg_age_students : ℝ)
  (num_students_before : ℕ)
  (num_students_after : ℕ)
  (age_teacher : ℝ)
  (new_avg_age_class : ℝ)

theorem age_of_student_who_left
  (h1 : avg_age_students = 14)
  (h2 : num_students_before = 45)
  (h3 : num_students_after = 44)
  (h4 : age_teacher = 45)
  (h5 : new_avg_age_class = 14.66)
: ∃ (age_student_left : ℝ), abs (age_student_left - 15.3) < 0.1 :=
sorry

end age_of_student_who_left_l45_45403


namespace train_crossing_time_l45_45095

noncomputable def time_to_cross (length_train1 length_train2 speed_train1_kmph speed_train2_kmph : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * (5 / 18)
  let speed_train2_mps := speed_train2_kmph * (5 / 18)
  let relative_speed_mps := speed_train1_mps + speed_train2_mps
  let combined_length_m := length_train1 + length_train2
  combined_length_m / relative_speed_mps

theorem train_crossing_time :
  time_to_cross 100 60 60 40 = 2.88 :=
by
  have speed_train1 := 60 * (5 / 18) -- m/s
  have speed_train2 := 40 * (5 / 18) -- m/s
  have relative_speed := speed_train1 + speed_train2
  have combined_length := 100 + 60
  have time := combined_length / relative_speed
  norm_num at time
  exact time
  
  sorry

end train_crossing_time_l45_45095


namespace lucky_numbers_count_l45_45455

def isLuckyNumber (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3 = 6) && (100 ≤ n) && (n < 1000)

def countLuckyNumbers : ℕ :=
  (List.range' 100 900).filter isLuckyNumber |>.length

theorem lucky_numbers_count : countLuckyNumbers = 21 := 
  sorry

end lucky_numbers_count_l45_45455


namespace fixed_point_of_exponential_function_l45_45648

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ (P : ℝ × ℝ), P = (1, 6) ∧ (∀ (x : ℝ), f x = 4 + 2 * a^(x - 1) → f 1 = 6) :=
by
  let f := λ x : ℝ, 4 + 2 * a^(x - 1)
  use (1, 6)
  split
  · refl
  · intros x hx
    sorry

end fixed_point_of_exponential_function_l45_45648


namespace math_problem_l45_45264

noncomputable def ellipse_c_eq : Prop :=
  (∀ (x y a b : ℝ), a > b ∧ b > 0 ∧ y^2 / a^2 + x^2 / b^2 = 1 ∧ 
    (sqrt(2)/2) * a = 1 → (y^2 / 2 + x^2 = 1))

noncomputable def range_of_m : Prop :=
  (∀ (k : ℝ), k ≠ 0 → ∃ m : ℝ, 0 < m ∧ m < 1/2)

noncomputable def max_area_triangle : Prop :=
  (∀ (m : ℝ), 0 < m ∧ m < 1/2 → 
    sqrt(2) * sqrt(m * (1 - m)^3) ≤ 3 * sqrt(6) / 16)

theorem math_problem : ellipse_c_eq ∧ range_of_m ∧ max_area_triangle := sorry

end math_problem_l45_45264


namespace suff_but_not_necess_condition_l45_45271

theorem suff_but_not_necess_condition (a b : ℝ) (h1 : a < 0) (h2 : -1 < b ∧ b < 0) : a + a * b < 0 :=
  sorry

end suff_but_not_necess_condition_l45_45271


namespace closest_cube_root_l45_45500

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l45_45500


namespace binomial_12_3_eq_220_l45_45203

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l45_45203


namespace tree_age_count_l45_45155

theorem tree_age_count :
  let digits := [1, 1, 1, 3, 7, 9] in
  let valid_first_digits := [7, 9] in
  ∑ first_digit in valid_first_digits, 
    (5! / 3!) = 40 := by sorry

end tree_age_count_l45_45155


namespace selling_price_l45_45651

theorem selling_price (cost_price initial_price profit : ℝ) (sell_increase_per_decrease sales_initial price_reduction_unit : ℝ)
  (h1 : cost_price = 22)
  (h2 : initial_price = 38)
  (h3 : profit = 3640)
  (h4 : sales_initial = 160)
  (h5 : sell_increase_per_decrease = 40)
  (h6 : price_reduction_unit = 1) :
  let x := 9 in
  let selling_price := initial_price - price_reduction_unit * x in
  selling_price = 29 :=
by {
  sorry
}

end selling_price_l45_45651


namespace find_stiffnesses_l45_45081

def stiffnesses (m g x1 x2 k1 k2 : ℝ) : Prop :=
  (m = 3) ∧ (g = 10) ∧ (x1 = 0.4) ∧ (x2 = 0.075) ∧
  (k1 * k2 / (k1 + k2) * x1 = m * g) ∧
  ((k1 + k2) * x2 = m * g)

theorem find_stiffnesses (k1 k2 : ℝ) :
  stiffnesses 3 10 0.4 0.075 k1 k2 → 
  k1 = 300 ∧ k2 = 100 := 
sorry

end find_stiffnesses_l45_45081


namespace closest_integer_to_cube_root_of_250_l45_45525

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45525


namespace arithmetic_mean_of_sequence_l45_45860

theorem arithmetic_mean_of_sequence : 
    let a : ℕ → ℚ := λ n, 2 + (n - 1)
    let S_n : ℕ → ℚ := λ n, n / 2 * (a 1 + a n)
    (S_n 52) / 52 = 27.5 :=
by
    -- Definitions from conditions
    let a : ℕ → ℚ := λ n, 2 + (n - 1)
    let S_n : ℕ → ℚ := λ n, n / 2 * (a 1 + a n)
    
    -- Begin proof
    sorry

end arithmetic_mean_of_sequence_l45_45860


namespace closest_integer_to_cube_root_of_250_l45_45486

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45486


namespace product_common_divisors_150_30_l45_45671

theorem product_common_divisors_150_30 :
  let divisors_150 := [1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150]
  let divisors_30 := [1, 2, 3, 5, 6, 10, 15, 30]
  let common_divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  (common_divisors.map (id)).prod * (common_divisors.map (λ x, -x)).prod = 182250000 :=
by
  sorry

end product_common_divisors_150_30_l45_45671


namespace lcm_852_1491_l45_45927

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end lcm_852_1491_l45_45927


namespace smallest_n_cond_l45_45977

theorem smallest_n_cond (n : ℕ) (h1 : n >= 100 ∧ n < 1000) (h2 : n ≡ 3 [MOD 9]) (h3 : n ≡ 3 [MOD 4]) : n = 111 := 
sorry

end smallest_n_cond_l45_45977


namespace ryan_time_learning_l45_45656

variable (t : ℕ) (c : ℕ)

/-- Ryan spends a total of 3 hours on both languages every day. Assume further that he spends 1 hour on learning Chinese every day, and you need to find how many hours he spends on learning English. --/
theorem ryan_time_learning (h_total : t = 3) (h_chinese : c = 1) : (t - c) = 2 := 
by
  -- Proof goes here
  sorry

end ryan_time_learning_l45_45656


namespace volume_of_intersection_of_cube_and_tetrahedron_l45_45628

theorem volume_of_intersection_of_cube_and_tetrahedron
  (a b : ℝ) 
  (h1 : b = 2 * a)
  (h2 : ∀ (P Q R S T : ℝ) (cube tetra : set ℝ),
    midpoint P Q ∈ cube ∧ midpoint R S ∈ tetra ∧ lies_on_same_line P Q R S ∧ coincides_midpoints (midpoint P Q) (midpoint R S)) :
  ∃ V : ℝ, V = (a^3 * real.sqrt 2 / 12) * (16 * real.sqrt 2 - 17) :=
by
  sorry

end volume_of_intersection_of_cube_and_tetrahedron_l45_45628


namespace closest_cube_root_of_250_l45_45477

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45477


namespace largest_systematic_sample_number_l45_45024

theorem largest_systematic_sample_number (num_products : ℕ) (num_selections : ℕ) (first_sample : ℕ) (interval : ℕ) (largest_sample : ℕ) : num_products = 120 → num_selections = 10 → first_sample = 7 → interval = num_products / num_selections → largest_sample = first_sample + (num_selections - 1) * interval → largest_sample = 115 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  simp at h4
  rw h4 at h5
  rw h5
  sorry

end largest_systematic_sample_number_l45_45024


namespace original_savings_eq_920_l45_45823

variable (S : ℝ) -- Define S as a real number representing Linda's savings
variable (h1 : S * (1 / 4) = 230) -- Given condition

theorem original_savings_eq_920 :
  S = 920 :=
by
  sorry

end original_savings_eq_920_l45_45823


namespace task_assignment_schemes_l45_45389

theorem task_assignment_schemes :
  let n := (choose 2 1 * choose 3 1 * fact 3) + (choose 3 2 * fact 2 * choose 3 2 * fact 2)
  n = 72 :=
by
  sorry

end task_assignment_schemes_l45_45389


namespace inequality_solution_maximum_value_solution_l45_45284

-- Define the function f with variable x and parameter a
def f (a x : ℝ) : ℝ := a * x^2 + x - a

-- Problem 1: Prove the inequality solution for a = 2
theorem inequality_solution :
  ∀ x : ℝ, f 2 x > 1 ↔ (x < -3 / 2 ∨ x > 1) :=
by
  intros x
  dsimp [f]
  rw [mul_comm 2 x, ← add_sub_assoc, ← sub_add_eq_add_sub, sub_lt_zero]
  sorry

-- Problem 2: Prove that the maximum value condition implies certain values of a
theorem maximum_value_solution :
  ∀ a : ℝ, (∃ x : ℝ, f a x = 17 / 8) → (a = -2 ∨ a = -1 / 8) :=
by
  intros a h
  obtain ⟨x, hx⟩ := h
  -- The following is derived from the problem's given max value equation
  have h_eq : (-4 * a^2 - 1) / (4 * a) = 17 / 8 := sorry
  sorry

end inequality_solution_maximum_value_solution_l45_45284


namespace largest_product_in_set_l45_45056

-- Definition of the set of numbers
def num_set : Set ℤ := {-8, -3, 0, 2, 4}

-- Function to calculate the largest product of any two distinct elements in a set
def largest_product (s : Set ℤ) : ℤ :=
  let products := {x * y | x ∈ s, y ∈ s, x ≠ y}
  Set.max' products sorry

-- Problem statement in Lean
theorem largest_product_in_set :
  largest_product num_set = 24 :=
sorry

end largest_product_in_set_l45_45056


namespace closest_integer_to_cube_root_of_250_l45_45518

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l45_45518


namespace intersection_correct_l45_45291

def M : Set ℝ := {x | |x - 1| > 1}
def N : Set ℝ := {x | x^2 - 3x ≤ 0}
def intersection := {x | 2 < x ∧ x ≤ 3}

theorem intersection_correct : M ∩ N = intersection := 
sorry

end intersection_correct_l45_45291


namespace sum_of_first_2022_terms_l45_45712

-- Given definitions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def shifted_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1/2) = -f (-(x + 1/2))

-- The main theorem
theorem sum_of_first_2022_terms (f g : ℝ → ℝ) (a : ℕ → ℝ)
  (h_shifted_odd : shifted_odd_function f)
  (h_g_def : ∀ x, g x = f x + 1)
  (h_a_def : ∀ n, a n = g (n / 2023)) :
  (finset.range 2022).sum (λ n, a (n+1)) = 2022 :=
sorry

end sum_of_first_2022_terms_l45_45712


namespace percentage_died_by_bombardment_l45_45770

variable (P : ℕ) (final_population : ℕ) (percentage_left_due_to_fear : ℚ)

theorem percentage_died_by_bombardment 
  (h₁ : P = 4399)
  (h₂ : final_population = 3168)
  (h₃ : percentage_left_due_to_fear = 20/100) :
  ∃ x : ℚ, (x ≈ 9.98 / 100) ∧
    P - (x * P) - (percentage_left_due_to_fear * (P - (x * P))) = final_population := 
by
  sorry

end percentage_died_by_bombardment_l45_45770


namespace construct_triangle_a_construct_triangle_b_l45_45572

-- Definitions for basic geometric entities and properties.

variables {point : Type*} [metric_space point]

-- Given three points A', B', C'
variables (A' B' C' : point)

-- Assume A'B'C' forms an acute-angled triangle.
def acute_angled_triangle (A' B' C' : point) : Prop := sorry -- Assume this definition exists.

-- Conditions specific to part (a)
def angle_bisectors_intersect_circumcircle (A B C A' B' C' : point) : Prop := sorry -- Assume exists.
  
-- Conditions specific to part (b)
def altitudes_intersect_circumcircle (A B C A' B' C' : point) : Prop := sorry -- Assume exists.
  
-- Constructing triangle ABC under given conditions
theorem construct_triangle_a (A' B' C' : point) (h1 : acute_angled_triangle A' B' C') :
  ∃ A B C : point, angle_bisectors_intersect_circumcircle A B C A' B' C' := by
  sorry

theorem construct_triangle_b (A' B' C' : point) (h1 : acute_angled_triangle A' B' C') :
  ∃ A B C : point, altitudes_intersect_circumcircle A B C A' B' C' := by
  sorry

end construct_triangle_a_construct_triangle_b_l45_45572


namespace incenter_of_triangle_ABC_l45_45382

variables 
  (A B C M : Point)
  (O_1 : Circle B M C)
  (O_2 : Circle C M A)
  (O_3 : Circle A M B)

theorem incenter_of_triangle_ABC
  (h1 : A ∈ O_1.circumcenter)
  (h2 : B ∈ O_2.circumcenter)
  (h3 : C ∈ O_3.circumcenter)
  (h_AM : line_through A (O_1.circumcenter))
  (h_BM : line_through B (O_2.circumcenter))
  (h_CM : line_through C (O_3.circumcenter)) :
  is_incenter_of M A B C :=
sorry

end incenter_of_triangle_ABC_l45_45382


namespace closest_integer_to_cbrt_250_l45_45566

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l45_45566


namespace unique_function_l45_45233

theorem unique_function (f : ℕ → ℕ) (h : ∀ n, f(n) + f(f(n)) + f(f(f(n))) = 3 * n) : ∀ n, f(n) = n :=
by
  sorry

end unique_function_l45_45233


namespace closest_cube_root_of_250_l45_45470

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l45_45470


namespace inequality_solution_l45_45431

theorem inequality_solution (x : ℝ) (h : x - 3 > 4x) : x < -1 :=
by {
  sorry
}

end inequality_solution_l45_45431


namespace num_sets_satisfying_union_l45_45459

theorem num_sets_satisfying_union : 
  {M : Set ℕ | M ∪ {1, 2} = {1, 2, 3}}.card = 4 := 
by 
  sorry

end num_sets_satisfying_union_l45_45459


namespace number_of_numbers_l45_45862

theorem number_of_numbers (N : ℝ) :
  let sum1 := 2 * 5.2,
      sum2 := 2 * 5.8,
      sum3 := 2 * 5.200000000000003,
      total_sum := sum1 + sum2 + sum3,
      average := 5.40 in
  total_sum / average = 6 :=
by
  -- Here, we state the definitions and prove the theorem afterwards
  sorry

end number_of_numbers_l45_45862


namespace closest_integer_to_cube_root_250_l45_45531

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45531


namespace red_ball_probability_l45_45772

-- Define the conditions
def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3
def red_balls : ℕ := total_balls - yellow_balls - green_balls

-- Define the probability function
def probability_of_red_ball (total red : ℕ) : ℚ := red / total

-- The main theorem statement to prove
theorem red_ball_probability :
  probability_of_red_ball total_balls red_balls = 3 / 5 :=
by
  sorry

end red_ball_probability_l45_45772


namespace infinite_tetrahedra_from_rectangular_sheet_l45_45839

theorem infinite_tetrahedra_from_rectangular_sheet :
  ∃ (f : ℝ → Prop) (interval : set ℝ), (∀ θ ∈ interval, f θ) ∧ (interval.nonempty) ∧
  (∀ θ1 θ2 ∈ interval, θ1 ≠ θ2 → f θ1 ≠ f θ2) :=
begin
  sorry
end

end infinite_tetrahedra_from_rectangular_sheet_l45_45839


namespace complex_number_in_fourth_quadrant_l45_45405

variable {a b : ℝ}

theorem complex_number_in_fourth_quadrant (a b : ℝ): 
  (a^2 + 1 > 0) ∧ (-b^2 - 1 < 0) → 
  ((a^2 + 1, -b^2 - 1).fst > 0 ∧ (a^2 + 1, -b^2 - 1).snd < 0) :=
by
  intro h
  exact h

#check complex_number_in_fourth_quadrant

end complex_number_in_fourth_quadrant_l45_45405


namespace triangle_QS_and_altitude_l45_45786

noncomputable def triangleData := 
{ PQR : ℝ × ℝ × ℝ // PQR.1 = 8 ∧ PQR.2 = 15 ∧ PQR.3 = 17 }

theorem triangle_QS_and_altitude
  (PQ QR PR : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 15)
  (hPR : PR = 17) :
  ∃ (QS alt : ℝ),
    QS = 4.8 ∧ alt = 25 :=
by
  sorry

end triangle_QS_and_altitude_l45_45786


namespace closest_integer_to_cube_root_of_250_l45_45510

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45510


namespace select_three_people_with_combined_age_at_least_142_l45_45318

theorem select_three_people_with_combined_age_at_least_142
  (ages : Fin 7 → ℕ)
  (total_age : (∑ i, ages i) = 332) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (ages i + ages j + ages k) ≥ 142 :=
by
  sorry

end select_three_people_with_combined_age_at_least_142_l45_45318


namespace greatest_4_digit_base_9_div_by_7_is_8050_l45_45910

def is_4_digit_base_9 (n : ℕ) : Prop :=
  let dec_value := ∑ i in [0,1,2,3], 8 * 9^i
  ∧ n = 6560
  ∧ ∀ m, m ∈ [1000_9 .. 8888_9] ∧ m % 7 = 0

theorem greatest_4_digit_base_9_div_by_7_is_8050 : 
  ∃ n, is_4_digit_base_9 n ∧ toBase n 9 = 8050 :=
sorry

end greatest_4_digit_base_9_div_by_7_is_8050_l45_45910


namespace total_people_l45_45956

theorem total_people (N B : ℕ) (h1 : N = 4 * B + 10) (h2 : N = 5 * B + 1) : N = 46 := by
  -- The proof will follow from the conditions, but it is not required in this script.
  sorry

end total_people_l45_45956


namespace pie_remaining_portion_l45_45995

theorem pie_remaining_portion (Carlos_share Maria_share remaining: ℝ)
  (hCarlos : Carlos_share = 0.65)
  (hRemainingAfterCarlos : remaining = 1 - Carlos_share)
  (hMaria : Maria_share = remaining / 2) :
  remaining - Maria_share = 0.175 :=
by
  sorry

end pie_remaining_portion_l45_45995


namespace closest_integer_to_cube_root_of_250_l45_45485

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45485


namespace range_of_f_l45_45424

noncomputable def f (x : ℝ) : ℝ :=
  x + Real.sqrt (x - 2)

theorem range_of_f : Set.range f = {y : ℝ | 2 ≤ y} :=
by
  sorry

end range_of_f_l45_45424


namespace parallel_lines_transitivity_l45_45837

theorem parallel_lines_transitivity
  {A B C A₁ B₁ C₁ : Type}
  [Linear A] [Linear B] [Linear C] [Linear A₁] [Linear B₁] [Linear C₁] 
  (h_col_ABC : collinear A B C) 
  (h_col_A₁B₁C₁ : collinear A₁ B₁ C₁) 
  (h1 : parallel (line_through A B₁) (line_through B A₁)) 
  (h2 : parallel (line_through A C₁) (line_through C A₁)) : 
  parallel (line_through B C₁) (line_through C B₁) := 
by
  sorry

end parallel_lines_transitivity_l45_45837


namespace unit_prices_cost_effective_buses_l45_45444

variables (x m : ℕ)
def unit_price_A := 36
def unit_price_B := 40
def total_buses := 30
def min_buses_A := 10
def max_buses_A := 20

axiom price_relation : ∀ x: ℕ, 720 / x = 800 / (x + 4)
axiom quantity_relation : ∀ m : ℕ, m ≥ 10 ∧ m ≤ 20
axiom total_cost : ∀ m: ℕ, 36 * m + 40 * (30 - m) = 1200 - 4 * m

theorem unit_prices :
  (∃ x : ℕ, 720 / x = 800 / (x + 4) ∧ 36 = x ∧ 40 = x + 4) :=
begin
  use 36,
  split,
  { exact price_relation 36 },
  { split; refl }
end

theorem cost_effective_buses :
  (∀ m : ℕ, m ≥ 10 ∧ m ≤ 20 → 36 * m + 40 * (30 - m) = 1200 - 4 * m → m = 20) ∧
  (36 * 20 + 40 * 10 = 1120) :=
begin
  split,
  { intros m h_range h_cost,
    cases h_range with h_min h_max,
    linarith },
  { refl }
end

end unit_prices_cost_effective_buses_l45_45444


namespace simplify_nested_fraction_root_l45_45848

theorem simplify_nested_fraction_root :
  Real.sqrt (Real.cbrt (Real.sqrt (1 / 65536))) = 1 / 2 := by
sorry

end simplify_nested_fraction_root_l45_45848


namespace simplify_sqrt_cube_sqrt_l45_45850

theorem simplify_sqrt_cube_sqrt (h : 65536 = 2 ^ 16) : 
  Real.sqrt (Real.cbrt (Real.sqrt (1 / 65536))) = 1 / 2 := by
  sorry

end simplify_sqrt_cube_sqrt_l45_45850


namespace length_of_BC_l45_45158

theorem length_of_BC (b : ℝ) (h : b ^ 4 = 125) : 2 * b = 10 :=
sorry

end length_of_BC_l45_45158


namespace triangle_XYZ_XY_value_l45_45787

theorem triangle_XYZ_XY_value :
  ∀ (XYZ : Type) (X Y Z : XYZ), (angle X Y Z = 90) → (dist Y Z = 18) →
  (tan Z = 3 * sin Y) → (dist X Y = 15.248) := by 
  sorry

end triangle_XYZ_XY_value_l45_45787


namespace triangle_side_sum_l45_45788

theorem triangle_side_sum (a b c : ℝ) 
  (triangle_ABC : Triangle a b c) 
  (cos_ratio : cos B / cos C = b / (2 * a - c))
  (area : S_triangle_ABC = 3 * Real.sqrt 3 / 4)
  (b_val : b = 3) : 
  a + c = 3 * Real.sqrt 2 :=
by 
  sorry

end triangle_side_sum_l45_45788


namespace rain_probability_l45_45576

theorem rain_probability :
  let PM : ℝ := 0.62
  let PT : ℝ := 0.54
  let PMcTc : ℝ := 0.28
  let PMT : ℝ := PM + PT - (1 - PMcTc)
  PMT = 0.44 :=
by
  sorry

end rain_probability_l45_45576


namespace vowel_initial_probability_l45_45313

def num_students : Nat := 30
def alphabet_size : Nat := 26
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y', 'W'}

def num_vowel_initials : Nat := vowels.card

theorem vowel_initial_probability :
  \frac{num_vowel_initials}{alphabet_size} = \frac{7}{26} := 
by
  sorry

end vowel_initial_probability_l45_45313


namespace range_of_m_l45_45819

noncomputable def f (x : ℝ) : ℝ := Real.log2 (2^x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.log2 (2^x - 1)
noncomputable def F (x m : ℝ) : ℝ := g x - f x - m

theorem range_of_m 
  (h : ∃ x ∈ Icc 1 2, F x m = 0) 
  : log2 (1 / 3) ≤ m ∧ m ≤ log2 (3 / 5) :=
sorry

end range_of_m_l45_45819


namespace arithmetic_seq_infinitely_many_squares_l45_45626

theorem arithmetic_seq_infinitely_many_squares 
  (a d : ℕ) 
  (h : ∃ (n y : ℕ), a + n * d = y^2) : 
  ∃ (m : ℕ), ∀ k : ℕ, ∃ n' y' : ℕ, a + n' * d = y'^2 :=
by sorry

end arithmetic_seq_infinitely_many_squares_l45_45626


namespace closest_integer_to_cube_root_of_250_l45_45511

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45511


namespace intersection_A_B_l45_45734

def A := {x : ℝ | 2 < x ∧ x < 4}
def B := {x : ℝ | (x-1) * (x-3) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end intersection_A_B_l45_45734


namespace birds_initial_count_l45_45084

theorem birds_initial_count (x : ℝ) (h1 : 14.0 > 0) (h2 : x - 14.0 = 7) : x = 21.0 :=
by
  have h3 : x = 7 + 14.0 := by linarith
  rw [h3]
  linarith

end birds_initial_count_l45_45084


namespace smallest_n_l45_45384

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n % 8 = 5) (h4 : n > 20) : n = 136 := by
  sorry

end smallest_n_l45_45384


namespace price_per_rose_is_2_l45_45739

-- Definitions from conditions
def has_amount (total_dollars : ℕ) : Prop := total_dollars = 300
def total_roses (R : ℕ) : Prop := ∃ (j : ℕ) (i : ℕ), R / 3 = j ∧ R / 2 = i ∧ j + i = 125

-- Theorem stating the price per rose
theorem price_per_rose_is_2 (R : ℕ) : 
  has_amount 300 → total_roses R → 300 / R = 2 :=
sorry

end price_per_rose_is_2_l45_45739


namespace binom_12_3_eq_220_l45_45186

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l45_45186


namespace third_term_arithmetic_sequence_l45_45436

variable (a d : ℤ)
variable (h1 : a + 20 * d = 12)
variable (h2 : a + 21 * d = 15)

theorem third_term_arithmetic_sequence : a + 2 * d = -42 := by
  sorry

end third_term_arithmetic_sequence_l45_45436


namespace zero_of_f_in_interval_l45_45893

noncomputable def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem zero_of_f_in_interval : (∃ c : ℝ, (1 < c ∧ c < 3/2) ∧ f c = 0) :=
by {
  -- Declaring the conditions
  have h_continuous : continuous f := sorry,
  have h_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry,
  have h_f_one : f 1 < 0 := sorry,
  have h_f_three_halves : f (3/2) > 0 := sorry,
  -- Proof that the zero is in the interval (1, 3/2)
  exact sorry
}

end zero_of_f_in_interval_l45_45893


namespace oblique_asymptote_l45_45907

def f (x : ℝ) : ℝ := (3 * x^2 + 4 * x + 5) / (2 * x + 3)

theorem oblique_asymptote :
  ∃ (a b : ℝ), (∀ (x : ℝ), f(x) - (a * x + b) → 0 as x → ∞) ∧ a = 3 / 2 ∧ b = 1 / 4 :=
by
  sorry

end oblique_asymptote_l45_45907


namespace coin_stack_height_l45_45761

theorem coin_stack_height :
  ∃ (a b c d : ℕ), 2.1 * a + 1.8 * b + 1.2 * c + 2.0 * d = 18 ∧ d = 9 :=
by
  existsi [0, 0, 0, 9]
  simp
  sorry

end coin_stack_height_l45_45761


namespace inequality_transitive_l45_45112

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by
  sorry

end inequality_transitive_l45_45112


namespace tickets_per_ride_l45_45068

theorem tickets_per_ride (tickets : ℕ) (cost_per_ride : ℕ) (rides : ℕ) 
  (h1 : cost_per_ride = 5) 
  (h2 : tickets = 10) 
  : tickets / cost_per_ride = rides :=
by {
  have h : rides = 2, sorry,
  exact h
}

end tickets_per_ride_l45_45068


namespace binary_to_base5_l45_45132

theorem binary_to_base5 : ∃ b5, (binary_to_decimal 1101 = 13 ∧ decimal_to_base5 13 = b5) ∧ b5 = 23 :=
by
  sorry

end binary_to_base5_l45_45132


namespace sequence_values_l45_45274

variable {a1 a2 b2 : ℝ}

theorem sequence_values
  (arithmetic : 2 * a1 = 1 + a2 ∧ 2 * a2 = a1 + 4)
  (geometric : b2 ^ 2 = 1 * 4) :
  (a1 + a2) / b2 = 5 / 2 :=
by
  sorry

end sequence_values_l45_45274


namespace number_of_chocolates_bought_l45_45753

theorem number_of_chocolates_bought (C S : ℝ) 
  (h1 : ∃ n : ℕ, n * C = 21 * S) 
  (h2 : (S - C) / C * 100 = 66.67) : 
  ∃ n : ℕ, n = 35 := 
by
  sorry

end number_of_chocolates_bought_l45_45753


namespace steven_more_peaches_l45_45342

variable (Jake Steven Jill : ℕ)

-- Conditions
axiom h1 : Jake + 6 = Steven
axiom h2 : Jill = 5
axiom h3 : Jake = 17

-- Goal
theorem steven_more_peaches : Steven - Jill = 18 := by
  sorry

end steven_more_peaches_l45_45342


namespace find_m_when_z_is_real_l45_45466

theorem find_m_when_z_is_real (m : ℝ) (h : (m ^ 2 + 2 * m - 15 = 0)) : m = 3 :=
sorry

end find_m_when_z_is_real_l45_45466


namespace characters_with_initial_d_l45_45826

-- Definitions based on conditions
def total_characters : ℕ := 60
def initial_a : ℕ := total_characters / 2  -- Half of the characters have the initial A
def initial_c : ℕ := initial_a / 2  -- Half of the initial A characters have the initial C
def remaining_characters : ℕ := total_characters - (initial_a + initial_c)  -- Remaining characters are for D and E

-- Conditions
def initial_d (E : ℕ) : ℕ := 2 * E  -- Twice as many characters with initial D as with initial E

-- Proving the given question
theorem characters_with_initial_d : { D : ℕ // D = initial_d 5 } :=
by
  have E : ℕ := remaining_characters / 3
  have D : ℕ := initial_d E
  exact ⟨D, rfl⟩

end characters_with_initial_d_l45_45826


namespace total_amount_l45_45136

-- Declare the variables
variables (A B C : ℕ)

-- Introduce the conditions as hypotheses
theorem total_amount (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B = 290) : 
  A + B + C = 980 := 
by {
  sorry
}

end total_amount_l45_45136


namespace universal_quantifiers_true_l45_45569

noncomputable def check_propositions : Prop :=
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧            -- Proposition A
  (∀ P : Type, is_rhombus P → diagonals_perpendicular P) ∧  -- Proposition C
  (∀ Q : Type, is_square Q → symmetrical Q) -- Proposition D

theorem universal_quantifiers_true :
  check_propositions :=
by
  have : (∀ x : ℝ, x^2 - x + 1 ≥ 0), from sorry,
  have : (∀ P : Type, is_rhombus P → diagonals_perpendicular P), from sorry,
  have : (∀ Q : Type, is_square Q → symmetrical Q), from sorry,
  exact ⟨this, this, this⟩

end universal_quantifiers_true_l45_45569


namespace determine_coefficients_l45_45221

noncomputable def poly (a b c : ℚ) (x : ℚ) : ℚ := a * x^2 + b * x + c

theorem determine_coefficients :
  ∃ a b c : ℚ, 
    poly a b c 1 = 1 ∧ 
    poly a b c 2 = 1/2 ∧ 
    poly a b c 3 = 1/3 ∧
    a = 1/6 ∧ b = -1 ∧ c = 11/6 :=
begin
  let a := 1 / 6,
  let b := -1,
  let c := 11 / 6,
  use [a, b, c],
  split,
  { simp [poly, a, b, c], norm_num },
  split,
  { simp [poly, a, b, c], norm_num },
  split,
  { simp [poly, a, b, c], norm_num },
  split,
  { refl },
  split,
  { refl },
  { refl }
end

end determine_coefficients_l45_45221


namespace functional_equation_l45_45925

noncomputable def S : Set ℝ := {x : ℝ | x ≠ 2 / 3}

theorem functional_equation (f : ℝ → ℝ) (h : ∀ x ∈ S, 2 * f x + f (2 * x / (3 * x - 2)) = 996 * x) :
  f = λ x, 1992 * x * (x - 1) / (3 * x - 2) :=
by
  ext x
  sorry

end functional_equation_l45_45925


namespace g_function_expression_a_b_relationship_l45_45286

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ -2 ∨ x ≥ 1 then 0 else -x^2 - x + 2

theorem g_function_expression :
  ∀ x : ℝ, g(x) = (|f(x)| - f(x)) / 2 := sorry

theorem a_b_relationship (a b : ℝ) (ha : a > 0) :
  ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧
    (g x1 = a * x1 + b) ∧ (g x2 = a * x2 + b) ∧ (g x3 = a * x3 + b) ↔
    0 < a ∧ a < 3 ∧ 2 * a < b ∧ b < (1/4) * (a + 1)^2 + 2 := sorry

end g_function_expression_a_b_relationship_l45_45286


namespace evaluate_expression_l45_45228

theorem evaluate_expression :
  let term_1 := 2^3 - 2^2 in
  let term_2 := 3^3 - 3^2 in
  let term_3 := 4^3 - 4^2 in
  let term_4 := 5^3 - 5^2 in
  (term_1 - term_2 + term_3 - term_4) = -66 :=
by
  sorry

end evaluate_expression_l45_45228


namespace inconsistent_conditions_l45_45094

-- Define distance at specific times
def distance_at (t : ℕ) : ℝ :=
  if t = 12 then 5
  else if t = 14 then 7
  else if t = 15 then 2
  else 0

-- The proof problem
theorem inconsistent_conditions 
  (h1 : ∀ t1 t2, t1 ≠ t2 ∧ t1 ∈ {12, 14, 15} ∧ t2 ∈ {12, 14, 15} → distance_at t1 ≠ distance_at t2)
  (h2 : ∀ t : ℕ, t ∈ {12, 14, 15} → distance_at t ≥ 0)
  : false := 
sorry

end inconsistent_conditions_l45_45094


namespace transformed_function_is_cosine_l45_45037

def f (x : ℝ) : ℝ := sin (2 * x + (Real.pi / 6))

def shift_left (x : ℝ) : ℝ := x + (Real.pi / 6)

def shift_function (f : ℝ → ℝ) (shift : ℝ → ℝ) : ℝ → ℝ := λ x, f (shift x)

def transform_x (f : ℝ → ℝ) (transformation : ℝ → ℝ) : ℝ → ℝ := λ x, f (transformation x)

def g (x : ℝ) : ℝ := cos x

theorem transformed_function_is_cosine : 
  (λ x, transform_x (shift_function f shift_left) (λ x, x / 2)) = g :=
sorry

end transformed_function_is_cosine_l45_45037


namespace triangle_is_isosceles_at_A_l45_45249

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

open_locale big_operators

noncomputable def is_isosceles_at_A (A B C : Type)
  (d : MetricSpace.Distance A B = MetricSpace.Distance A C) : Prop := 
  MetricSpace.Distance A B = MetricSpace.Distance A C

noncomputable def midpoint (x y : Type) [AddCommGroup x] [Module ℝ x] : Type := sorry
noncomputable def foot_of_perpendicular (x y z : Type) [AddCommGroup x] [Module ℝ x] : Type := sorry

theorem triangle_is_isosceles_at_A (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (A'_mp : midpoint B C) (D_fp : foot_of_perpendicular A B C)
  (length_equality : MetricSpace.Distance A A'_mp = MetricSpace.Distance A D_fp) :
  is_isosceles_at_A A B C :=
sorry

end triangle_is_isosceles_at_A_l45_45249


namespace total_concrete_weight_l45_45604

theorem total_concrete_weight (w1 w2 : ℝ) (c1 c2 : ℝ) (total_weight : ℝ)
  (h1 : w1 = 1125)
  (h2 : w2 = 1125)
  (h3 : c1 = 0.093)
  (h4 : c2 = 0.113)
  (h5 : (w1 * c1 + w2 * c2) / (w1 + w2) = 0.108) :
  total_weight = w1 + w2 :=
by
  sorry

end total_concrete_weight_l45_45604


namespace park_bench_problem_l45_45151

/-- A single bench section at a park can hold either 8 adults or 12 children.
When N bench sections are connected end to end, an equal number of adults and 
children seated together will occupy all the bench space.
This theorem states that the smallest positive integer N such that this condition 
is satisfied is 3. -/
theorem park_bench_problem : ∃ N : ℕ, N > 0 ∧ (8 * N = 12 * N) ∧ N = 3 :=
by
  sorry

end park_bench_problem_l45_45151


namespace solve_simultaneous_equations_l45_45030

theorem solve_simultaneous_equations :
  (∃ x y : ℝ, x^2 + 3 * y = 10 ∧ 3 + y = 10 / x) ↔ 
  (x = 3 ∧ y = 1 / 3) ∨ 
  (x = 2 ∧ y = 2) ∨ 
  (x = -5 ∧ y = -5) := by sorry

end solve_simultaneous_equations_l45_45030


namespace ratio_of_savings_to_earnings_l45_45825

-- Definitions based on the given conditions
def earnings_washing_cars : ℤ := 20
def earnings_walking_dogs : ℤ := 40
def total_savings : ℤ := 150
def months : ℤ := 5

-- Statement to prove the ratio of savings per month to total earnings per month
theorem ratio_of_savings_to_earnings :
  (total_savings / months) = (earnings_washing_cars + earnings_walking_dogs) / 2 := by
  sorry

end ratio_of_savings_to_earnings_l45_45825


namespace closest_integer_to_cube_root_of_250_l45_45516

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45516


namespace cauchy_inequality_minimum_value_inequality_l45_45587

-- Part 1: Prove Cauchy Inequality
theorem cauchy_inequality (a b x y : ℝ) : 
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

-- Part 2: Find the minimum value under the given conditions
theorem minimum_value_inequality (x y : ℝ) (h₁ : x^2 + y^2 = 2) (h₂ : x ≠ y ∨ x ≠ -y) : 
  ∃ m, m = (1 / (9 * x^2) + 9 / y^2) ∧ m = 50 / 9 :=
by
  sorry

end cauchy_inequality_minimum_value_inequality_l45_45587


namespace range_of_m_l45_45705

variable (a b c m y1 y2 y3 : Real)

-- Given points and the parabola equation
def on_parabola (x y a b c : Real) : Prop := y = a * x^2 + b * x + c

-- Conditions
variable (hP : on_parabola (-2) y1 a b c)
variable (hQ : on_parabola 4 y2 a b c)
variable (hM : on_parabola m y3 a b c)
variable (h_vertex : 2 * a * m + b = 0)
variable (h_y_order : y3 ≥ y2 ∧ y2 > y1)

-- Theorem to prove m > 1
theorem range_of_m : m > 1 :=
sorry

end range_of_m_l45_45705


namespace ellipse_problem_l45_45275

-- Define the conditions
def is_on_ellipse (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (c a : ℝ) : Prop :=
  c / a = (Real.sqrt 3 / 2)

def point_O_origin : Prop := 
  (0, 0)

def passes_through (x y : ℝ) (p1 p2 : ℝ) : Prop :=
  y = p1 * x + p2

def intersects_ellipse (l : Line) (C : Ellipse) : Prop :=
  ∃ A B : Point, A ≠ B ∧ is_on_ellipse A.x A.y C.a C.b ∧ is_on_ellipse B.x B.y C.a C.b

-- Define the problem and proof statement
theorem ellipse_problem :
  ∃ (a b : ℝ), 
    a > b ∧ b > 0 ∧
    eccentricity (Real.sqrt (a^2 - b^2)) a ∧
    is_on_ellipse (Real.sqrt 3) (1/2) a b ∧
    ∃ (l : Line), 
      passes_through l.x l.y 1 1 ∧
      intersects_ellipse l (Ellipse.mk a b) ∧
      (passes_through l.x l.y 1 (1 : ℝ)) → 
      is_on_ellipse (8*(1-1)/(4*1^2+1)) (2*(1-1)/(4*(1^2)+1)) 2 1 ∧
      ((l.x = sorry) ∨ (l.x = 1)).

end ellipse_problem_l45_45275


namespace number_of_incorrect_interpretations_is_zero_l45_45681

def neg_eight := -8
def opposite_of_neg_eight := 8
def product_of_neg_one_and_neg_eight := -1 * neg_eight
def abs_value_of_neg_eight := |neg_eight|
def result_of_double_negation := -neg_eight

theorem number_of_incorrect_interpretations_is_zero :
  (opposite_of_neg_eight = result_of_double_negation) ∧
  (product_of_neg_one_and_neg_eight = result_of_double_negation) ∧
  (abs_value_of_neg_eight = result_of_double_negation) ∧
  (result_of_double_negation = 8) → 
  0 = 0 :=
by {
  intro h,
  exact rfl
} 

end number_of_incorrect_interpretations_is_zero_l45_45681


namespace one_eighth_of_2_pow_44_eq_2_pow_x_l45_45305

theorem one_eighth_of_2_pow_44_eq_2_pow_x (x : ℕ) :
  (2^44 / 8 = 2^x) → x = 41 :=
by
  sorry

end one_eighth_of_2_pow_44_eq_2_pow_x_l45_45305


namespace triangle_problem_l45_45760

theorem triangle_problem
  (a b c : ℝ)
  (h : ((a + b)^2 - c^2) / (3 * a * b) = 1) :
  (∠ C : ℝ) (C = 60° ∧
  c = sqrt(3) → b = sqrt(2) → (B = 45° ∧ S = (3 + sqrt(3)) / 4)) := 
sorry

end triangle_problem_l45_45760


namespace geniuses_eventually_know_either_number_l45_45937

theorem geniuses_eventually_know_either_number (m : ℕ) (h : m + 1 > m) :
  ∃ n : ℕ, (n = m ∨ n = m + 1) ∧ (some_step_of_elimination n) :=
by
  sorry -- To be proven

end geniuses_eventually_know_either_number_l45_45937


namespace jane_played_rounds_l45_45315

-- Define the conditions
def points_per_round := 10
def points_ended_with := 60
def points_lost := 20

-- Define the proof problem
theorem jane_played_rounds : (points_ended_with + points_lost) / points_per_round = 8 :=
by
  sorry

end jane_played_rounds_l45_45315


namespace new_percentage_girls_is_32_l45_45594

-- Define the initial conditions
def initial_total_students : ℕ := 20
def initial_percentage_girls : ℝ := 0.4
def new_boys : ℕ := 5

-- Define the quantities derived from initial conditions
def initial_number_girls : ℕ := (initial_total_students : ℝ) * initial_percentage_girls
def initial_number_boys : ℕ := initial_total_students - initial_number_girls

-- Define the new state of the classroom after 5 boys join
def new_total_students : ℕ := initial_total_students + new_boys
def new_number_boys : ℕ := initial_number_boys + new_boys

-- Prove that the new percentage of girls is 32% or 0.32
theorem new_percentage_girls_is_32 : (initial_number_girls : ℝ) / (new_total_students : ℝ) = 0.32 := by
  sorry

end new_percentage_girls_is_32_l45_45594


namespace crossing_time_approx_18_seconds_l45_45740

-- Definitions for the problem conditions
def lorry_length : ℝ := 200
def lorry_speed_kmph : ℝ := 80
def bridge_length : ℝ := 200

-- Conversion factor from kmph to mps
def kmph_to_mps_conversion : ℝ := 1000 / 3600

-- Statements of the problem and the solution
theorem crossing_time_approx_18_seconds :
  let total_distance := lorry_length + bridge_length
  let lorry_speed_mps := lorry_speed_kmph * kmph_to_mps_conversion
  let time_taken := total_distance / lorry_speed_mps
  abs (time_taken - 18) < 1 :=
by
  sorry

end crossing_time_approx_18_seconds_l45_45740


namespace divisible_by_five_solution_exists_l45_45573

theorem divisible_by_five_solution_exists
  (a b c d : ℤ)
  (h₀ : ∃ k : ℤ, d = 5 * k + d % 5 ∧ d % 5 ≠ 0)
  (h₁ : ∃ n : ℤ, (a * n^3 + b * n^2 + c * n + d) % 5 = 0) :
  ∃ m : ℤ, (a + b * m + c * m^2 + d * m^3) % 5 = 0 := 
sorry

end divisible_by_five_solution_exists_l45_45573


namespace closest_integer_to_cubert_of_250_is_6_l45_45551

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45551


namespace coefficients_relation_l45_45687

theorem coefficients_relation 
  (a : ℕ → ℤ)
  (h₀ : a 0 = 0)
  (h₁ : ∑ i in Finset.range 10, a i = 2)
  (h₂ : ∑ i in Finset.range 10, if i % 2 = 0 then a i else -a i = -2)
  (h₃ : a 2 + a 4  + a 6  + a 8  = 2)
  (h₄ : a 1 + a 3  + a 5  + a 7  + a 9  = 0) :
  (a 1 + a 3 + a 5 + a 7 + a 9 + 2) * (a 2 + a 4 + a 6 + a 8) = 4 :=
sorry

end coefficients_relation_l45_45687


namespace sequence_general_formula_triangle_area_l45_45253

-- Proof Problem 1: Given S_n = 2a_n - 2, prove that a_n = 2^n
theorem sequence_general_formula (S : ℕ → ℤ) (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 2) :
  ∀ n : ℕ, n > 0 → a n = 2^(n : ℤ) :=
sorry

-- Proof Problem 2: Given a triangle ABC with sides 3, a1, and a2, prove that the area is 3√15 / 4
theorem triangle_area (a1 a2 : ℕ) (h1 : a1 = 2) (h2 : a2 = 4) :
  let Area := (3 * a2 * Real.sqrt 15) / 8 in
  Area = (3 * Real.sqrt 15) / 4 :=
sorry

end sequence_general_formula_triangle_area_l45_45253


namespace geometric_first_term_l45_45983

-- Define the conditions
def is_geometric_series (first_term : ℝ) (r : ℝ) (sum : ℝ) : Prop :=
  sum = first_term / (1 - r)

-- Define the main theorem
theorem geometric_first_term (r : ℝ) (sum : ℝ) (first_term : ℝ) 
  (h_r : r = 1/4) (h_S : sum = 80) (h_sum_formula : is_geometric_series first_term r sum) : 
  first_term = 60 :=
by
  sorry

end geometric_first_term_l45_45983


namespace det_scaled_matrix_l45_45688

theorem det_scaled_matrix (a b c d : ℝ) (h : a * d - b * c = 5) : 
  (3 * a) * (3 * d) - (3 * b) * (3 * c) = 45 :=
by 
  sorry

end det_scaled_matrix_l45_45688


namespace circle_radius_l45_45066

theorem circle_radius (x y : ℝ) : (x^2 + y^2 + 2 * x + 6 * y = 0) → ∃ r : ℝ, r = sqrt 10 := 
by {
  intros h,
  use sqrt 10,
  sorry
}

end circle_radius_l45_45066


namespace problem_l45_45684

theorem problem
  (a : ℕ → ℝ)
  (h : (2 * x + Real.sqrt 3) ^ 21 = ∑ i in Finset.range 22, a i * x ^ i) :
  (∑ i in Finset.range 22, if even i then a i else 0)^2 -
  (∑ i in Finset.range 22, if odd i then a i else 0)^2 = -1 :=
by
  sorry

end problem_l45_45684


namespace original_price_of_sarees_l45_45885

theorem original_price_of_sarees (P : ℝ) (h : 0.72 * P = 108) : P = 150 := 
by 
  sorry

end original_price_of_sarees_l45_45885


namespace total_pages_in_book_l45_45840

-- Define the conditions
def pagesDay1To5 : Nat := 5 * 25
def pagesDay6To9 : Nat := 4 * 40
def pagesLastDay : Nat := 30

-- Total calculation
def totalPages (p1 p2 pLast : Nat) : Nat := p1 + p2 + pLast

-- The proof problem statement
theorem total_pages_in_book :
  totalPages pagesDay1To5 pagesDay6To9 pagesLastDay = 315 :=
  by
    sorry

end total_pages_in_book_l45_45840


namespace xiao_ming_calculation_is_incorrect_l45_45915

theorem xiao_ming_calculation_is_incorrect : 
  (let original_expression := (-36) / (-1 / 2 + 1 / 6 - 1 / 3) in 
   original_expression = 54) ∧ ¬ (original_expression = (-36) / (-1 / 2) + (-36) / (1 / 6) + (-36) / (-1 / 3) ∧ 
   72 - 216 + 108 = -36) := 
by
  sorry

end xiao_ming_calculation_is_incorrect_l45_45915


namespace extremum_at_one_range_of_a_l45_45720

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := log (a * x + 1) + (1 - x) / (1 + x)

-- Prove that f has an extremum at x = 1 when a = 1.
theorem extremum_at_one (h : ∀ x ≥ 0, a > 0) : (∀ x ≥ 0, (deriv (f 1) x = 0) → x = 1) ↔ a = 1 :=
sorry

-- Prove that f(x) ≥ ln 2 implies a ≥ 1.
theorem range_of_a (h : ∀ x ≥ 0, a > 0) : (∀ x ≥ 0, f a x ≥ log 2) ↔ a ≥ 1 :=
sorry

end extremum_at_one_range_of_a_l45_45720


namespace solve_crew_l45_45579

-- Define individual types
inductive Person
| Smirnov
| Zhukov
| Romanov

-- Define city types
inductive City
| Lviv
| Omsk
| Chita

-- Define occupation types
inductive Occupation
| Engineer
| Stoker
| Conductor

-- Define predicates
def lives_in : Person → City → Prop :=
| Person.Romanov, City.Lviv => true
| Person.Smirnov, City.Chita => true
| Person.Zhukov, City.Omsk => true
| _, _ => false

def occupation : Person → Occupation → Prop :=
| Person.Smirnov, Occupation.Engineer => true
| Person.Romanov, Occupation.Stoker => true
| Person.Zhukov, Occupation.Conductor => true
| _, _ => false

theorem solve_crew :
  (occupation Person.Smirnov Occupation.Engineer) ∧
  (occupation Person.Romanov Occupation.Stoker) ∧
  (occupation Person.Zhukov Occupation.Conductor) ∧
  (lives_in Person.Romanov City.Lviv) ∧
  (lives_in Person.Zhukov City.Chita) ∧
  (lives_in Person.Smirnov City.Omsk) :=
by sorry

end solve_crew_l45_45579


namespace store_makes_profit_of_8_yuan_l45_45618

noncomputable theory

def price := 64
def selling_price_c1 := price
def selling_price_c2 := price
def C1 := 64 / 1.6
def C2 := 64 / 0.8
def total_cost := C1 + C2
def total_revenue := selling_price_c1 + selling_price_c2
def profit := total_revenue - total_cost

theorem store_makes_profit_of_8_yuan : profit = 8 := 
by
  unfold profit total_revenue total_cost selling_price_c1 selling_price_c2 C1 C2 price
  norm_num
  sorry

end store_makes_profit_of_8_yuan_l45_45618


namespace man_swims_distance_back_l45_45142

def swimming_speed_still_water : ℝ := 8
def speed_of_water : ℝ := 4
def time_taken_against_current : ℝ := 2
def distance_swum : ℝ := 8

theorem man_swims_distance_back :
  (distance_swum = (swimming_speed_still_water - speed_of_water) * time_taken_against_current) :=
by
  -- The proof will be filled in later.
  sorry

end man_swims_distance_back_l45_45142


namespace sin_angle_BAG_of_cube_l45_45939

-- Definitions and conditions
structure Cube :=
  (a b g : ℝ)
  (edge_len : ℝ)
  (cube_structure : true)  -- Placeholder to indicate it is a cube. Additional structure may be needed in real scenarios.

-- Problem statement
theorem sin_angle_BAG_of_cube (c : Cube) (h_len : c.edge_len = 2) : 
  sin (c.a) = √3 / 3 := 
sorry

end sin_angle_BAG_of_cube_l45_45939


namespace monotonic_interval_of_f_axis_of_symmetry_max_min_values_of_f_l45_45256

-- Hypothesis: the monotonicity interval of the function f.
theorem monotonic_interval_of_f (k : ℤ) :
  ∀ x, ( -π/6 + k * π ≤ x ∧ x ≤ π/3 + k * π) ↔ 
       (2 * x - π/6 ∈ Set.Icc (-π/2 + 2 * k * π) (π/2 + 2 * k * π)) := sorry

-- Hypothesis: the axis of symmetry of the function f.
theorem axis_of_symmetry (k : ℤ) :
  2 * x - π/6 = π/2 + k * π ↔
  x = π/3 + k * π / 2 := sorry

-- Hypothesis: the maximum and minimum values of the function f on the interval [0, π/2].
theorem max_min_values_of_f :
  (∀ x, (0 ≤ x ∧ x ≤ π / 2) →
   -1 ≤ 2 * sin (2 * x - π / 6) ∧
   2 * sin (2 * x - π / 6) ≤ 2) ∧
   (∃ x, x = 0 ∧ 2 * sin (2 * 0 - π / 6) = -1) ∧
   (∃ x, x = π / 3 ∧ 2 * sin (2 * (π / 3) - π / 6) = 2) := sorry

end monotonic_interval_of_f_axis_of_symmetry_max_min_values_of_f_l45_45256


namespace closest_integer_to_cube_root_250_l45_45537

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45537


namespace mean_equality_l45_45415

theorem mean_equality (y z : ℝ)
  (h : (14 + y + z) / 3 = (8 + 15 + 21) / 3)
  (hyz : y = z) :
  y = 15 ∧ z = 15 :=
by sorry

end mean_equality_l45_45415


namespace closest_integer_to_cube_root_250_l45_45530

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l45_45530


namespace bisection_min_calculations_l45_45917

theorem bisection_min_calculations 
  (a b : ℝ)
  (h_interval : a = 1.4 ∧ b = 1.5)
  (delta : ℝ)
  (h_delta : delta = 0.001) :
  ∃ n : ℕ, 0.1 / (2 ^ n) ≤ delta ∧ n = 7 :=
sorry

end bisection_min_calculations_l45_45917


namespace graph_paper_problem_l45_45054

theorem graph_paper_problem :
  let line_eq := ∀ x y : ℝ, 7 * x + 268 * y = 1876
  ∃ (n : ℕ), 
  (∀ x y : ℕ, 0 < x ∧ x ≤ 268 ∧ 0 < y ∧ y ≤ 7 ∧ (7 * (x:ℝ) + 268 * (y:ℝ)) < 1876) →
  n = 801 :=
by
  sorry

end graph_paper_problem_l45_45054


namespace wave_number_count_l45_45614

def is_wave_number (n : List ℕ) : Prop :=
  (n.length == 5) ∧ 
  (n.nth 3 > n.nth 2 ∧ n.nth 3 > n.nth 4 ∧ n.nth 1 > n.nth 0 ∧ n.nth 1 > n.nth 2) ∧
  (n.nodup) ∧ 
  (∀ d, d ∈ n → d ∈ [1, 2, 3, 4, 5])

theorem wave_number_count : 
  (finset.univ.filter is_wave_number).card = 16 :=
sorry

end wave_number_count_l45_45614


namespace triangle_side_length_l45_45769

theorem triangle_side_length
  (a b : ℝ) (theta : ℝ) (ha : a = 10) (hb : b = 12) (hθ : theta = Real.pi / 3) :
  sqrt (a^2 + b^2 - 2 * a * b * Real.cos theta) = sqrt 124 :=
by
  sorry

end triangle_side_length_l45_45769


namespace smallest_sampled_number_l45_45148

theorem smallest_sampled_number (total_students : ℕ) (sample_size : ℕ) (sampled_student : ℕ) (sampling_interval : ℕ)
    (h1 : total_students = 1260) (h2 : sample_size = 60) (h3 : sampled_student = 355) (h4 : sampling_interval = total_students / sample_size) : 
    ∃ smallest_number, smallest_number = 19 :=
by
  have h4 : sampling_interval = 21 := by norm_num [h1, h2, h4]
  use 19
  sorry

end smallest_sampled_number_l45_45148


namespace greatest_4_digit_base_9_div_by_7_is_8050_l45_45911

def is_4_digit_base_9 (n : ℕ) : Prop :=
  let dec_value := ∑ i in [0,1,2,3], 8 * 9^i
  ∧ n = 6560
  ∧ ∀ m, m ∈ [1000_9 .. 8888_9] ∧ m % 7 = 0

theorem greatest_4_digit_base_9_div_by_7_is_8050 : 
  ∃ n, is_4_digit_base_9 n ∧ toBase n 9 = 8050 :=
sorry

end greatest_4_digit_base_9_div_by_7_is_8050_l45_45911


namespace paint_leftover_l45_45009

theorem paint_leftover (containers total_walls tiles_wall paint_ceiling : ℕ) 
  (h_containers : containers = 16) 
  (h_total_walls : total_walls = 4) 
  (h_tiles_wall : tiles_wall = 1) 
  (h_paint_ceiling : paint_ceiling = 1) : 
  containers - ((total_walls - tiles_wall) * (containers / total_walls)) - paint_ceiling = 3 :=
by 
  sorry

end paint_leftover_l45_45009


namespace calculate_platform_length_l45_45895

theorem calculate_platform_length (speed_cm_s : ℝ) (subway_length : ℝ) (time_s : ℝ) (speed_m_s : ℝ) (distance_traveled : ℝ) : 
  speed_cm_s / 100 = speed_m_s ∧
  speed_m_s * time_s = distance_traveled ∧
  distance_traveled - subway_length = 52 → 
  True := by
  sorry

def subway_speed_cm_s : ℝ := 288
def subway_length_m : ℝ := 20
def time_s : ℝ := 25
def subway_speed_m_s : ℝ := subway_speed_cm_s / 100
def distance_traveled_m : ℝ := subway_speed_m_s * time_s

example : distance_traveled_m - subway_length_m = 52 := by
  have h1 : subway_speed_cm_s / 100 = subway_speed_m_s := by
    rfl
  have h2 : subway_speed_m_s * time_s = distance_traveled_m := by
    rfl
  have h3 : distance_traveled_m - subway_length_m = 52 := by
    calc distance_traveled_m - subway_length_m = 72 - 20 : by
      rw h2
      rw [← h1, Real.div_eq_add_inv_mul]
    ...                           = 52 : by linarith
  exact h3

end calculate_platform_length_l45_45895


namespace marcus_milk_bottles_l45_45824

theorem marcus_milk_bottles (T J M : ℕ) (h1 : T = 45) (h2 : J = 20) : M = 25 :=
by
  -- Using the conditions to show Marcus's milk bottles
  have : M = T - J := by sorry
  -- Applying our given conditions
  rw [h1, h2] at this
  -- Simplifying to reach the conclusion
  exact this.symm

end marcus_milk_bottles_l45_45824


namespace closest_integer_to_cube_root_of_250_l45_45538

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45538


namespace sequence_a_n_sequence_b_n_sum_T_n_l45_45290

noncomputable def a_n (n : ℕ) := 3 * n - 1
noncomputable def b_n (n : ℕ) := 2^n
noncomputable def c_n (n : ℕ) := (a_n n + 1) * b_n n
noncomputable def T_n (n : ℕ) := (Finset.range n).sum (λ i, c_n (i + 1))

theorem sequence_a_n (n : ℕ) : a_n n = 3 * n - 1 := by sorry

theorem sequence_b_n (n : ℕ) : b_n n = 2^n := by sorry

theorem sum_T_n (n : ℕ) : T_n n = 6 + 3 * (n - 1) * 2^(n + 1) := by sorry

end sequence_a_n_sequence_b_n_sum_T_n_l45_45290


namespace jane_played_8_rounds_l45_45316

variable (points_per_round : ℕ)
variable (end_points : ℕ)
variable (lost_points : ℕ)
variable (total_points : ℕ)
variable (rounds_played : ℕ)

theorem jane_played_8_rounds 
  (h1 : points_per_round = 10) 
  (h2 : end_points = 60) 
  (h3 : lost_points = 20)
  (h4 : total_points = end_points + lost_points)
  (h5 : total_points = points_per_round * rounds_played) : 
  rounds_played = 8 := 
by 
  sorry

end jane_played_8_rounds_l45_45316


namespace card_union_of_sets_l45_45707

theorem card_union_of_sets : 
  let A := {1, 2, 3}
  let B := {2, 4, 5}
  #|A ∪ B| = 5 :=
by 
  sorry

end card_union_of_sets_l45_45707


namespace least_n_for_odd_product_div_by_45_l45_45649

theorem least_n_for_odd_product_div_by_45 :
  ∃ n : ℕ, (∀ (k : ℕ), (∏ i in finset.range n, (2 * k + 1 + 2 * i)) % 45 = 0) ∧
  (∀ m < n, ∃ k, (∏ i in finset.range m, (2 * k + 1 + 2 * i)) % 45 ≠ 0) :=
begin
  use 6,
  split,
  { intros k,
    sorry, -- Proof here
  },
  { intros m h,
    use m,
    sorry, -- Proof here
  }
end

end least_n_for_odd_product_div_by_45_l45_45649


namespace circle_area_through_square_vertices_l45_45139

theorem circle_area_through_square_vertices (a : ℝ) :
    let R := Real.sqrt(65 * a^2 / 4) in
    let area := Real.pi * R^2 in
    (∃ (C : ℝ), C = 3 * a) → 
    (∃ (A D : ℝ), (C, D) passes through two adjacent vertices of a square with side length a) →
    area = 65 * Real.pi * a^2 / 4 :=
by
  intros
  let R := Real.sqrt(65 * a^2 / 4)
  let area := Real.pi * R^2
  existsi C
  existsi A
  existsi D
  sorry

end circle_area_through_square_vertices_l45_45139


namespace constant_polynomial_solution_l45_45660

noncomputable theory
open_locale classical

-- Definition of the condition that P(2x) = P(x) for all x in ℝ
def polynomial_condition (P : ℝ[X]) : Prop :=
∀ x : ℝ, P.eval (2 * x) = P.eval x

-- The statement of the problem
theorem constant_polynomial_solution (P : ℝ[X]) :
  (polynomial_condition P) → ∃ c : ℝ, P = polynomial.C c :=
begin
  sorry
end

end constant_polynomial_solution_l45_45660


namespace closest_integer_to_cube_root_of_250_l45_45546

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45546


namespace arrangement_male_student_at_ends_arrangement_females_not_next_to_each_other_arrangement_females_constraints_l45_45438

-- 1. Male student A must stand at one of the ends

theorem arrangement_male_student_at_ends : 
  let total_students := [A, male_1, male_2, male_3, female_1, female_2, female_3]
  let permutations := (A :: tail total_students).permutations
  (∀ perm ∈ permutations, perm.head = A ∨ perm.last == A) -> 
  permutations.length = 1440 := 
sorry

-- 2. Female students B and C cannot stand next to each other

theorem arrangement_females_not_next_to_each_other:
  let students := [male_1, male_2, male_3, male_4, female_1, B, C]
  let not_next_to_each_other (l : List char) := ∀ i, l[i] = B ∨ l[i+1] = C -> false
  (∀ perm ∈ students.permutations, not_next_to_each_other perm) -> 
  students.permutations.length = 3600 := 
sorry

-- 3. Female student B not at ends and C not in the middle

theorem arrangement_females_constraints:
  let students := [male_1, male_2, male_3, male_4, female_2, B, C]
  let b_not_at_ends (l : List char) := head l ≠ B ∧ last l ≠ B
  let c_not_in_middle (l : List char) := (length l / 2 == C) -> false
  (∀ perm ∈ students.permutations, b_not_at_ends perm ∧ c_not_in_middle perm) ->
  students.permutations.length = 3120 := 
sorry

end arrangement_male_student_at_ends_arrangement_females_not_next_to_each_other_arrangement_females_constraints_l45_45438


namespace diplomats_not_speaking_english_l45_45166

theorem diplomats_not_speaking_english :
  ∀ (D F E neither both : ℕ),
  D = 120 →
  F = 20 →
  (neither : ℕ) = 24 →
  both = 12 →
  D * 0.20 = neither →
  D * 0.10 = both →
  F - both + neither = 32 := by
  intros D F E neither both D_eq F_eq neither_eq both_eq cmp_neither cmp_both
  sorry

end diplomats_not_speaking_english_l45_45166


namespace ellipse_equation_l45_45701

-- Define the given conditions
def foci_on_y_axis : Prop := ∀ (x : ℝ), ∃ (y : ℝ), is_focus y x
def major_axis_length : ℝ := 20
def eccentricity : ℝ := 2/5

-- Define the standard equation of the ellipse as a target statement
def standard_equation_of_ellipse : Prop :=
  ∀ (x y : ℝ), (y^2 / 100) + (x^2 / 84) = 1

theorem ellipse_equation (h_foci : foci_on_y_axis) (h_major_axis : major_axis_length = 20) (h_ecc : eccentricity = 2/5) : 
  standard_equation_of_ellipse :=
sorry

end ellipse_equation_l45_45701


namespace remaining_wax_l45_45794

-- Define the conditions
def ounces_for_car : ℕ := 3
def ounces_for_suv : ℕ := 4
def initial_wax : ℕ := 11
def spilled_wax : ℕ := 2

-- Define the proof problem: Show remaining wax after detailing car and SUV
theorem remaining_wax {ounces_for_car ounces_for_suv initial_wax spilled_wax : ℕ} :
  initial_wax - spilled_wax - (ounces_for_car + ounces_for_suv) = 2 :=
by
  -- Defining the variables according to the conditions
  have h1 : ounces_for_car = 3 := rfl
  have h2 : ounces_for_suv = 4 := rfl
  have h3 : initial_wax = 11 := rfl
  have h4 : spilled_wax = 2 := rfl
  -- Using the conditions to calculate the remaining wax
  calc
    initial_wax - spilled_wax - (ounces_for_car + ounces_for_suv)
        = 11 - 2 - (3 + 4) : by rw [h1, h2, h3, h4]
    ... = 11 - 2 - 7 : rfl
    ... = 9 - 7 : rfl
    ... = 2 : rfl

end remaining_wax_l45_45794


namespace x0_equals_pm1_l45_45358

-- Define the function f and its second derivative
def f (x : ℝ) : ℝ := x^3
def f'' (x : ℝ) : ℝ := 6 * x

-- Prove that if f''(x₀) = 6 then x₀ = ±1
theorem x0_equals_pm1 (x0 : ℝ) (h : f'' x0 = 6) : x0 = 1 ∨ x0 = -1 :=
by
  sorry

end x0_equals_pm1_l45_45358


namespace find_remainder_in_division_l45_45764

theorem find_remainder_in_division
  (D : ℕ)
  (r : ℕ) -- the remainder when using the incorrect divisor
  (R : ℕ) -- the remainder when using the correct divisor
  (h1 : D = 12 * 63 + r)
  (h2 : D = 21 * 36 + R)
  : R = 0 :=
by
  sorry

end find_remainder_in_division_l45_45764


namespace common_difference_l45_45263

theorem common_difference (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ)
    (h₁ : a 5 + a 6 = -10)
    (h₂ : S 14 = -14)
    (h₃ : ∀ n, S n = n * (a 1 + a n) / 2)
    (h₄ : ∀ n, a (n + 1) = a n + d) :
  d = 2 :=
sorry

end common_difference_l45_45263


namespace coordinates_of_B_l45_45265

-- Definitions of the points and vectors are given as conditions.
def A : ℝ × ℝ := (-1, -1)
def a : ℝ × ℝ := (2, 3)

-- Statement of the problem translated to Lean
theorem coordinates_of_B (B : ℝ × ℝ) (h : B = (5, 8)) :
  (B.1 + 1, B.2 + 1) = (3 * a.1, 3 * a.2) :=
sorry

end coordinates_of_B_l45_45265


namespace socks_pair_count_l45_45758

theorem socks_pair_count : 
  let W := 5; 
  let B := 4; 
  let Bl := 3 in 
  (Bl * W + Bl * B) = 27 :=
by 
  sorry

end socks_pair_count_l45_45758


namespace remainder_when_divided_by_17_l45_45131

theorem remainder_when_divided_by_17
  (N k : ℤ)
  (h : N = 357 * k + 36) :
  N % 17 = 2 :=
by
  sorry

end remainder_when_divided_by_17_l45_45131


namespace no_three_consecutive_zeros_l45_45006

theorem no_three_consecutive_zeros (a : ℕ → ℝ) :
  (∀ x, ∑' n, (x ^ n * (x - 1) ^ (2 * n) / n!) = ∑' n, a n * x ^ n) →
  ¬ (∃ n, a n = 0 ∧ a (n + 1) = 0 ∧ a (n + 2) = 0) :=
begin
  intro h,
  by_contradiction H,
  rcases H with ⟨n, an0, an1_0, an2_0⟩,
  -- Further proof steps would go here
  sorry
end

end no_three_consecutive_zeros_l45_45006


namespace number_of_shapes_correct_sum_of_areas_correct_l45_45100

def initial_paper_dimensions : ℕ × ℕ := (20, 12)

def area_after_folds (k : ℕ) : ℕ :=
  240 * (k + 1) / 2^k

def number_of_shapes_after_4_folds : ℕ := 5

def sum_of_areas_after_n_folds (n : ℕ) : ℝ :=
  240 * (3 - (n + 3) / (2^n))

theorem number_of_shapes_correct : number_of_shapes_after_4_folds = 5 := sorry

theorem sum_of_areas_correct (n : ℕ) : 
  (∑ k in Finset.range n, area_after_folds (k + 1)) = sum_of_areas_after_n_folds n := sorry

end number_of_shapes_correct_sum_of_areas_correct_l45_45100


namespace BD_is_symmedian_l45_45936

/-- Geometry problem setup -/
variables {A B C K D : Point}
variable (S : Circle A B C)

-- Conditions
axiom B_on_S : touches S B
axiom tangent_BK : tangent_to_line S B K (Line A C)
axiom tangent_KD : tangent_to_point S K D

-- Desired proof
theorem BD_is_symmedian (h1 : touches S B)
                        (h2 : tangent_to_line S B K (Line A C))
                        (h3 : tangent_to_point S K D) :
  symmedian (Line B D) (△ A B C) :=
sorry

end BD_is_symmedian_l45_45936


namespace min_beta_delta_l45_45213

noncomputable def g (z : ℂ) (β δ : ℂ) : ℂ :=
  (3 + 2 * Complex.i) * z^2 + β * z + δ

theorem min_beta_delta
  (β δ : ℂ)
  (h1 : ∃ x y u v : ℝ, β = x + y * Complex.i ∧ δ = u + v * Complex.i)
  (h2 : (g 1 β δ).im = 0)
  (h3 : (g (-Complex.i) β δ).im = 0) :
  ∃ β δ : ℂ, ∀ β δ : ℂ, (λ β δ, Complex.abs β + Complex.abs δ) = √40 :=
sorry

end min_beta_delta_l45_45213


namespace combination_30_choose_5_l45_45602

theorem combination_30_choose_5 :
  (nat.choose 30 5) = 142506 :=
by
  sorry

end combination_30_choose_5_l45_45602


namespace arithmetic_sequence_sum_l45_45713

noncomputable def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℕ :=
  n * 2^n

def S_n (n : ℕ) : ℕ :=
  (n - 1) * 2^(n + 1) + 2

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (h1 : a_n 1 + a_n 2 + a_n 3 = 6)
  (h2 : a_n 5 = 5)
  (h3 : ∀ n, b_n n = a_n n * 2^(a_n n)) :
  (∀ n, a_n n = n) ∧ (∀ n, S_n n = (n - 1) * 2^(n + 1) + 2) :=
by
  sorry

end arithmetic_sequence_sum_l45_45713


namespace range_of_k_l45_45252

theorem range_of_k (k : ℝ) (P : set ℕ) (h1 : P = {x ∈ ℕ | 2 < x ∧ x < k}) (h2 : P.card = 3) : 5 < k ∧ k ≤ 6 :=
sorry

end range_of_k_l45_45252


namespace batch_rejection_probability_l45_45947

-- Definitions following the conditions
def batchSize : ℕ := 100
def defectiveChance : ℝ := 0.03
def nonDefectiveChance : ℝ := 0.97
def samplesTested : ℕ := 4

-- Non-defective probabilities after removal
def P_non_def_1 := (97 : ℝ) / batchSize
def P_non_def_2 := (96 : ℝ) / (batchSize - 1)
def P_non_def_3 := (95 : ℝ) / (batchSize - 2)
def P_non_def_4 := (94 : ℝ) / (batchSize - 3)

-- Compute overall non-defective probability
def P_all_non_def := P_non_def_1 * P_non_def_2 * P_non_def_3 * P_non_def_4

-- Final probability that at least one item is defective
def P_batch_rejected := 1 - P_all_non_def

theorem batch_rejection_probability:
  P_batch_rejected ≈ 0.1164 := sorry

end batch_rejection_probability_l45_45947


namespace tom_payment_l45_45447

theorem tom_payment :
  let q_apples := 8
  let r_apples := 70
  let q_mangoes := 9
  let r_mangoes := 70
  let cost_apples := q_apples * r_apples
  let cost_mangoes := q_mangoes * r_mangoes
  let total_amount := cost_apples + cost_mangoes
  total_amount = 1190 :=
by
  let q_apples := 8
  let r_apples := 70
  let q_mangoes := 9
  let r_mangoes := 70
  let cost_apples := q_apples * r_apples
  let cost_mangoes := q_mangoes * r_mangoes
  let total_amount := cost_apples + cost_mangoes
  sorry

end tom_payment_l45_45447


namespace gcd_exponential_l45_45814

theorem gcd_exponential (a b n : ℤ) : 
  Int.gcd (n^a - 1) (n^b - 1) = n^(Int.gcd a b) - 1 := 
by 
  sorry

end gcd_exponential_l45_45814


namespace minimum_area_triangle_OAB_l45_45005

noncomputable def f (x : ℝ) : ℝ :=
  -x^2 + 1

noncomputable def tangent_slope (x : ℝ) : ℝ :=
  -2 * x

noncomputable def equation_of_tangent_line (m : ℝ) : ℝ → ℝ :=
  λ x, -2 * m * (x - m) - m^2 + 1

noncomputable def intersection_x_axis (m : ℝ) : ℝ :=
  (m^2 + 1) / (2 * m)

noncomputable def intersection_y_axis (m : ℝ) : ℝ :=
  m^2 + 1

noncomputable def area_triangle (m : ℝ) : ℝ :=
  (1 / 2) * (intersection_x_axis m) * (intersection_y_axis m)

noncomputable def f_area (m : ℝ) : ℝ :=
  m^3 + 2 * m + (1 / m)

theorem minimum_area_triangle_OAB : ∃ (m : ℝ) (h : m > 0), area_triangle m = (4 / 9) * Real.sqrt 3 :=
by
  sorry

end minimum_area_triangle_OAB_l45_45005


namespace prize_amount_l45_45144

theorem prize_amount (P : ℕ) :
  (∀ n, 1 ≤ n ∧ n ≤ 20 → n ≥ 1 → n ≤ 20 → true) →
  (∀ n, n = 20 → ∀ i, 1 ≤ i ∧ i ≤ n → 20 ≤ 20) →
  (∀ k, k = 12 → ∀ t, t = 4080 → (2/5 : ℝ) * (P : ℝ) = t) →
  (∀ m, m = 340 ∧ m * 12 = 4080) →
  (P : ℝ) = 10200 :=
begin
  intros,
  sorry
end

end prize_amount_l45_45144


namespace painted_cubes_l45_45953

def four_inch_cube_cut := 4^3 = 64
def painted_faces := 3
def painted_cubes_with_at_least_two_faces := 16

theorem painted_cubes (cuts : nat) (faces : nat) :
  cuts = 64 ∧ faces = 3 → painted_cubes_with_at_least_two_faces = 16 :=
by {
  sorry
}

end painted_cubes_l45_45953


namespace closest_integer_to_cube_root_of_250_l45_45539

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45539


namespace vertex_of_given_parabola_l45_45866

-- Define the standard form of a parabola
def standard_form (a : ℝ) (h : ℝ) (k : ℝ) (x : ℝ) : ℝ := a * (x - h) ^ 2 + k

-- Define the given equation of the parabola
def given_parabola (x : ℝ) : ℝ := 3 * (x - 2) ^ 2 - 5

-- State the theorem we are proving
theorem vertex_of_given_parabola : ∃ h k, ∀ x, given_parabola(x) = standard_form 3 h k x ∧ h = 2 ∧ k = -5 :=
by
  exists 2, -5
  intro x
  dsimp [given_parabola, standard_form]
  split
  · rfl
  · rfl
  · rfl
  sorry

end vertex_of_given_parabola_l45_45866


namespace broken_line_perimeter_eq_triangle_face_l45_45338

-- Definitions for conditions in the problem
def isAltitude (A1 B1 C1 : Point) (A B C: Triangle) : Prop :=
  -- Define what it means for A1, B1, C1 to be altitudes in triangle ABC
  sorry

def isMedian (A0 B0 C0 : Point) (A B C: Triangle) : Prop :=
  -- Define what it means for A0, B0, C0 to be medians in triangle ABC
  sorry

-- The main theorem to state the equivalent proof problem
theorem broken_line_perimeter_eq_triangle_face 
  (A B C A1 B1 C1 A0 B0 C0 : Point)
  (h1: isAltitude A1 B1 C1 A B C)
  (h2: isMedian A0 B0 C0 A B C) :
  length_of_broken_line A0 B1 C0 A1 B0 C1 A0 = perimeter_of_triangle A B C :=
sorry

end broken_line_perimeter_eq_triangle_face_l45_45338


namespace wilfred_carrots_on_tuesday_l45_45113

theorem wilfred_carrots_on_tuesday :
  ∀ (carrots_eaten_Wednesday carrots_eaten_Thursday total_carrots desired_total: ℕ),
    carrots_eaten_Wednesday = 6 →
    carrots_eaten_Thursday = 5 →
    desired_total = 15 →
    desired_total - (carrots_eaten_Wednesday + carrots_eaten_Thursday) = 4 :=
by
  intros
  sorry

end wilfred_carrots_on_tuesday_l45_45113


namespace product_of_common_divisors_l45_45668

theorem product_of_common_divisors (d : Set ℤ) 
  (h1 : d = {n | n ∣ 150}) 
  (h2 : Set.Subset d {n | n ∣ 30}) : 
  (∏ x in d, abs x) = 16443022500 := 
by 
  sorry

end product_of_common_divisors_l45_45668


namespace unique_fixed_points_l45_45348

open Set

variable (f : ℝ → ℝ)
variable (f_diff : ∀ x ∈ Icc (0 : ℝ) (1 : ℝ), DifferentiableAt ℝ f x)
variable (f_range : ∀ x ∈ Icc (0 : ℝ) (1 : ℝ), f x ∈ Icc (0 : ℝ) (1 : ℝ))
variable (f_deriv : ∀ x ∈ Icc (0 : ℝ) (1 : ℝ), abs (deriv f x) ≠ 1)

theorem unique_fixed_points : 
  ∃! α ∈ Icc (0 : ℝ) (1 : ℝ), f α = α ∧ 
  ∃! β ∈ Icc (0 : ℝ) (1 : ℝ), f β = 1 - β := 
sorry

end unique_fixed_points_l45_45348


namespace marble_arrangement_zero_l45_45115

def marble_arrangement_count (A B S T C : Type) :=
  let total_arrangements := 5.fact
  let adjacent_ST := 2 * 4.fact
  let not_adjacent_ST := total_arrangements - adjacent_ST
  let adjacent_AB := 2 * 4.fact
  let not_adjacent_ST_AB := not_adjacent_ST - adjacent_AB
  let overlap_ST_AB := 2 * 2 * 3.fact
  let valid_arrangements := not_adjacent_ST_AB - overlap_ST_AB
  valid_arrangements

theorem marble_arrangement_zero :
  ∀ (A B S T C : Type), marble_arrangement_count A B S T C = 0 :=
begin
  intros,
  unfold marble_arrangement_count,
  sorry
end

end marble_arrangement_zero_l45_45115


namespace team_b_fraction_of_total_calls_l45_45948

/-- Each team has members that process calls at specific rates compared to Team B, 
    and specific ratios of agents compared to Team B.
    We aim to prove the fraction of total calls processed by Team B. -/
theorem team_b_fraction_of_total_calls :
  ∀ (B N : ℚ),
  let total_a_calls := (7 / 5) * B * (5 / 8) * N in
  let total_b_calls := B * N in
  let total_c_calls := (5 / 3) * B * (3 / 4) * N in
  let total_calls := total_a_calls + total_b_calls + total_c_calls in
  total_b_calls / total_calls = 8 / 25 :=
by
  intros
  sorry

end team_b_fraction_of_total_calls_l45_45948


namespace min_value_PF1_PF2_l45_45729

-- Definitions and assumptions
variable {a b : ℝ} (ha : a > 0) (hb : b > 0)
def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Foci of the ellipse
def F1 := (-1, 0)
def F2 := (1, 0)

-- The condition that the eccentricities are reciprocal
def eccentricity_cond (e_hyperbola e_ellipse : ℝ) (c : ℝ) : Prop := e_hyperbola = 2 ∧ e_ellipse = 1 / 2 ∧ e_hyperbola = c / a

-- Points on the right branch of the hyperbola
variable (P : ℝ × ℝ)
def right_branch (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2 ∧ P.1 > 0

-- Distance calculations
def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)
def PF1 (P : ℝ × ℝ) := distance P F1
def PF2 (P : ℝ × ℝ) := distance P F2

-- Main statement to prove
theorem min_value_PF1_PF2 
    (e_hyperbola e_ellipse c : ℝ) 
    (ecc_cond : eccentricity_cond e_hyperbola e_ellipse c)
    (r_branch_P : right_branch P)
    (t : ℝ) (ht : t = PF2 P) : 
    ∃ t, (t + 1 / t + 2) ≥ 4 ∧ ∀ x, (x + 1 / x + 2) ≥ 4 → x = 1 := 
by sorry

end min_value_PF1_PF2_l45_45729


namespace length_major_axis_ellipse_l45_45981

theorem length_major_axis_ellipse (F1 F2 : ℝ × ℝ) (x_axis_tangent : Bool) : 
  F1 = (5, 10) ∧ F2 = (35, 30) ∧ x_axis_tangent = true → 
  let d := dist F1 F2
  2 * sqrt (d^2 - (F1.1 - F2.1)^2) = 10 * sqrt 13 := 
by
  intro h
  cases h with F1_coords h
  cases h with F2_coords tangent_cond
  simp_all at tangent_cond
  sorry

end length_major_axis_ellipse_l45_45981


namespace mixture_problem_l45_45124

theorem mixture_problem :
  (∀ (y : ℝ), (0.14 * (200 + y) = 20 + 0.3 * y) → y = 50) :=
begin
  sorry
end

end mixture_problem_l45_45124


namespace points_collinear_b_value_l45_45673

theorem points_collinear_b_value :
  ∀ b : ℝ, (¬ collinear (3, -5) (1, -1) (-2, b) → b = 5) :=
by sorry

end points_collinear_b_value_l45_45673


namespace derivative_at_zero_l45_45867

noncomputable def y : ℝ → ℝ := λ x, (2 * x + 1) ^ 3

theorem derivative_at_zero :
  (deriv y 0) = 6 :=
by
  sorry

end derivative_at_zero_l45_45867


namespace polynomial_fact_l45_45586

theorem polynomial_fact (a b : ℝ) (n : ℕ) : 
  (a - b) * (∑ i in Finset.range (n+1), a^(n-i) * b^i) = a^(n+1) - b^(n+1) :=
by sorry

end polynomial_fact_l45_45586


namespace quadratic_solution_l45_45752

theorem quadratic_solution (a b : ℝ) (h : a + b + 1 = 0) : 2022 - a - b = 2023 :=
by
  have h1 : a + b = -1 := by linarith
  show 2022 - a - b = 2023
  calc
    2022 - a - b = 2022 - (-1) : by rw [← h1]
              ... = 2022 + 1    : by linarith
              ... = 2023        : by linarith

-- Note that the above theorem uses real numbers for simplicity; ℝ can be adjusted to ℚ or ℤ if needed.

end quadratic_solution_l45_45752


namespace closest_integer_to_cubert_of_250_is_6_l45_45548

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45548


namespace system_of_equations_solutions_l45_45588

theorem system_of_equations_solutions (x y : ℝ) (h1 : x ^ 5 + y ^ 5 = 1) (h2 : x ^ 6 + y ^ 6 = 1) :
    (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end system_of_equations_solutions_l45_45588


namespace not_all_x_ne_1_imp_x2_ne_0_l45_45422

theorem not_all_x_ne_1_imp_x2_ne_0 : ¬ (∀ x : ℝ, x ≠ 1 → x^2 - 1 ≠ 0) :=
sorry

end not_all_x_ne_1_imp_x2_ne_0_l45_45422


namespace max_students_equal_division_l45_45931

theorem max_students_equal_division (pens pencils : ℕ) (h_pens : pens = 640) (h_pencils : pencils = 520) : 
  Nat.gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  have : Nat.gcd 640 520 = 40 := by norm_num
  exact this

end max_students_equal_division_l45_45931


namespace tangent_line_at_one_l45_45049

section
variables (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the function f and its conditions
def fx := λ x : ℝ, x^2 - (f'(-1)) * x + 1

-- The equation of the tangent line at x = 1 is y = 3x
theorem tangent_line_at_one :
  let f := fx in 
  let x := 1 in 
  let fx1 := f x in 
  let f_prime_1 := (f' 1) in 
  ∀ x y : ℝ, y = f_prime_1 * x → f x = y → y = 3 * x :=
sorry
end

end tangent_line_at_one_l45_45049


namespace perfect_squares_less_than_500_with_digit_1_5_6_l45_45298

theorem perfect_squares_less_than_500_with_digit_1_5_6 :
  ∃ (count : ℕ), count = (Finset.card $ 
  (Finset.filter (λ n, let d := n^2 % 10 in d = 1 ∨ d = 5 ∨ d = 6) 
    (Finset.filter (λ n, n^2 < 500) 
      (Finset.range 23)))) ∧ count = 7 :=
by {
  sorry
}

end perfect_squares_less_than_500_with_digit_1_5_6_l45_45298


namespace part_one_part_two_l45_45266

-- Definitions of propositions
def p (k : ℝ) : Prop := ∀ x : ℝ, x ∈ set.Icc (-1) 1 → x^2 + 2 * x - k ≤ 0
def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * k * x + 3 * k + 4 = 0

-- Part (1): prove range of k for which p is true (i.e., \neg (\neg p))
theorem part_one (k : ℝ) : (¬ (¬ p k)) → (k ≥ 3) := sorry

-- Part (2): find the range of k such that exactly one of p and q is false
theorem part_two (k : ℝ) : (¬ p k ∧ q k) ∨ (p k ∧ ¬q k) → (k ∈ set.Icc (-∞) (-1) ∪ set.Ico 3 4) := sorry

end part_one_part_two_l45_45266


namespace range_of_g_l45_45361

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by
  sorry

end range_of_g_l45_45361


namespace irrational_constructed_number_l45_45682

noncomputable def constructed_number : ℝ :=
"0.010100100101001" -- note: in practice, defining such sequences explicitly requires more advanced definitions.

def has_only_0_and_1_digits (x : ℝ) : Prop :=
-- Definition to ensure only digits 0 and 1 in decimal representation.

def no_two_adjacent_1 (x : ℝ) : Prop :=
-- Definition to ensure no two adjacent digits are both 1.

def no_more_than_two_adjacent_0 (x : ℝ) : Prop :=
-- Definition to ensure no more than two 0s are adjacent.

theorem irrational_constructed_number :
  has_only_0_and_1_digits constructed_number ∧
  no_two_adjacent_1 constructed_number ∧
  no_more_than_two_adjacent_0 constructed_number →
  ¬ ∃ r : ℚ, (constructed_number:ℝ) = r :=
by sorry

end irrational_constructed_number_l45_45682


namespace mowing_time_approx_l45_45829

noncomputable def lawn_length := 100
noncomputable def lawn_width := 120
noncomputable def swath_width := 30 / 12
noncomputable def overlap := 4 / 12
noncomputable def effective_swath_width := swath_width - overlap
noncomputable def mowing_rate := 5000
noncomputable def number_of_strips := Nat.ceil (lawn_width / effective_swath_width)
noncomputable def total_distance_mown := number_of_strips * lawn_length
noncomputable def mowing_time := total_distance_mown / mowing_rate

theorem mowing_time_approx : abs (mowing_time - 1.2) < 0.1 := sorry

end mowing_time_approx_l45_45829


namespace line_intersects_hyperbola_once_l45_45045

-- Definitions of the curves
def C1 (x y : ℝ) : Prop := x^2 - y^2 = 1
def C2 (x y : ℝ) : Prop := (1 / x^2) - (1 / y^2) = 1

-- Given point P on C2
def P_on_C2 (m n : ℝ) : Prop := (1 / m^2) - (1 / n^2) = 1

-- Equation of the line MN derived from P
def line_MN (m n x y : ℝ) : Prop := y = -(n / m) * x + n

-- The proof statement that needs to be proved
theorem line_intersects_hyperbola_once (m n : ℝ) (hmn : P_on_C2 m n) :
  ∀ x y, line_MN m n x y → C1 x y → (x, y) = (some x, some y) := 
sorry

end line_intersects_hyperbola_once_l45_45045


namespace num_unique_patterns_l45_45741

def three_shaded_patterns := 
  { patterns : set (fin 9) | patterns.card = 3 ∧ 
  ∀ (p q : set (fin 9)), p ∈ patterns ∧ q ∈ patterns → 
  (p ≠ q) → ¬(∃ θ : zmod 4, is_rotation_by θ p q)}

theorem num_unique_patterns : (set finite (three_shaded_patterns)).card = 17 :=
sorry

end num_unique_patterns_l45_45741


namespace triangle_inequality_l45_45337

variables {a b c : ℝ} -- Sides of the triangle
variables {A B C : ℝ} -- Angles of the triangle
variables {Δ s r : ℝ} -- Area, semiperimeter and incircle radius

def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

def incircle_radius (Δ s : ℝ) : ℝ := Δ / s

theorem triangle_inequality (a b c : ℝ) (Δ : ℝ) (h1 : Δ > 0)
    (h2: s = semi_perimeter a b c) 
    (h3: r = incircle_radius Δ s)
    (h4: A + B + C = π) :
    a^2 * (s - a) / r + b^2 * (s - b) / r + c^2 * (s - c) / r ≥ 12 * Δ := 
sorry

end triangle_inequality_l45_45337


namespace remainder_sum_l45_45460

-- Define the conditions given in the problem.
def remainder_13_mod_5 : ℕ := 3
def remainder_12_mod_5 : ℕ := 2
def remainder_11_mod_5 : ℕ := 1

theorem remainder_sum :
  ((13 ^ 6 + 12 ^ 7 + 11 ^ 8) % 5) = 3 := by
  sorry

end remainder_sum_l45_45460


namespace closest_integer_to_cube_root_of_250_l45_45496

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l45_45496


namespace number_of_intersected_unit_cubes_l45_45957

-- Define the large cube formed by stacking 27 unit cubes
def large_cube := (3*3*3 : ℕ)

-- Define a condition where a plane is perpendicular to one of the internal diagonals of the large cube
-- and bisects that diagonal
def plane_bisects_diagonal : Prop :=
  ∀ x y z : ℤ, (x, y, z) ∈ { 
    (0, 0, 0), (0, 0, 1), (0, 0, 2), 
    (0, 1, 0), (0, 1, 1), (0, 1, 2), 
    (0, 2, 0), (0, 2, 1), (0, 2, 2), 
    (1, 0, 0), (1, 0, 1), (1, 0, 2), 
    (1, 1, 0), (1, 1, 1), (1, 1, 2), 
    (1, 2, 0), (1, 2, 1), (1, 2, 2), 
    (2, 0, 0), (2, 0, 1), (2, 0, 2), 
    (2, 1, 0), (2, 1, 1), (2, 1, 2), 
    (2, 2, 0), (2, 2, 1), (2, 2, 2)
  } → (x + y + z = 3)

-- Define the theorem that states the problem in a mathematically equivalent way
theorem number_of_intersected_unit_cubes : ∀ plane_bisects_diagonal, (number_of_intersected_unit_cubes == 19) := by
  sorry

end number_of_intersected_unit_cubes_l45_45957


namespace sufficient_but_not_necessary_l45_45864

theorem sufficient_but_not_necessary (a : ℝ) : (a ∈ set.Ici (2 : ℝ)) → (∃ x : ℝ, x^2 - a * x + 1 = 0) ∧ ¬( ∀ a ∈ set_of (λ b, ∃ x : ℝ, x^2 - b * x + 1 = 0), a ∈ set.Ici 2) :=
by
  sorry

end sufficient_but_not_necessary_l45_45864


namespace least_beads_divisible_l45_45150

theorem least_beads_divisible :
  let n := 840 in
  ∃ (m : ℕ), m = n ∧ (∀ k ∈ {2, 3, 5, 7, 8}, k ∣ m) ∧ (m ≤ n) :=
by
  let m := 840
  use m
  split
  rfl
  split
  · intro k hk
    fin_cases hk <;> {
      norm_num
    }
  · exact le_refl m
  sorry

end least_beads_divisible_l45_45150


namespace tickets_equation_l45_45901

theorem tickets_equation 
    (A C : ℕ)
    (h1 : 5.50 * A + 3.50 * C = 83.50)
    (h2 : A + C = 21) : 
    A = 5 := 
by
  -- Proof goes here
sorry

end tickets_equation_l45_45901


namespace seq_100th_term_l45_45966

-- Definition of the sequence {a_n}
def seq_a (n : ℕ) : ℕ → Real -> Real
| 1, _ => 5
| n+1, a => (2*(n+1) + 3) * a - (2*(n+1) + 5) / (2*(n+1) + 3) / (2*(n+1) + 5) * log(1 + 1/↑n)

-- Definition of the transformed sequence {b_n}
def seq_b (n : ℕ) : ℝ :=
(a : ℝ) / (2 * n + 3)

-- Main theorem statement
theorem seq_100th_term : seq_b 100 = 3 :=
by
  sorry

end seq_100th_term_l45_45966


namespace intersection_H_independent_of_D_l45_45336

-- Given definitions and conditions in Lean
variables {A B C G D E F H : Type}
variables [has_angle A B C]
variables [line AB : set (A × B)]
variables [line AD : set (A × D)]
variables [line BC : set (B × C)]
variables [line AC : set (A × C)]
variables [line EF : set (E × F)]
variables [line CG : set (C × G)]
variables [point_on_line_A_B : G ∈ AB]
variables [point_on_line_C_G : D ∈ CG]
variables [intersection_E : E = intersection_point AD BC]
variables [intersection_F : F = intersection_point BD AC]
variables [intersection_H : H = intersection_point EF AB]

theorem intersection_H_independent_of_D (A B C G D E F H : Type)
    (h_angle : has_angle A B C)
    (h_AB : line AB : set (A × B))
    (h_AD : line AD : set (A × D))
    (h_BC : line BC : set (B × C))
    (h_AC : line AC : set (A × C))
    (h_EF : line EF : set (E × F))
    (h_CG : line CG : set (C × G))
    (h_point_on_line_A_B : G ∈ AB)
    (h_point_on_line_C_G : D ∈ CG)
    (h_intersection_E : E = intersection_point AD BC)
    (h_intersection_F : F = intersection_point BD AC)
    (h_intersection_H : H = intersection_point EF AB) :
  ∃ h : G ∈ AB, ∀ D ∈ CG, H ∈ AB :=
sorry

end intersection_H_independent_of_D_l45_45336


namespace least_common_denominator_sum_l45_45464

open Nat

theorem least_common_denominator_sum :
  lcm 5 (lcm 6 (lcm 8 (lcm 9 (lcm 10 11)))) = 3960 := by
  sorry

end least_common_denominator_sum_l45_45464


namespace dodecagon_diagonals_l45_45743

theorem dodecagon_diagonals :
  let n : ℕ := 12
  in (n * (n - 3)) / 2 = 54 := 
by
  let n := 12
  have h1 : n = 12 := rfl
  have h2 : n - 3 = 9 := rfl
  have h3 : n * 9 = 108 := by norm_num
  show (n * (n - 3)) / 2 = 54 sorry

end dodecagon_diagonals_l45_45743


namespace ce_length_l45_45364

noncomputable def CE_in_parallelogram (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) : ℝ :=
  280

theorem ce_length (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) :
  CE_in_parallelogram AB AD BD AB_eq AD_eq BD_eq = 280 :=
by
  sorry

end ce_length_l45_45364


namespace num_ordered_pairs_31_l45_45246

theorem num_ordered_pairs_31 :
  let pairs := { (a, b) : ℕ × ℕ | a < 1000 ∧ b < 1000 ∧ a * b = b^2 / a } in
  pairs.to_finset.card = 31 :=
sorry

end num_ordered_pairs_31_l45_45246


namespace binom_12_3_eq_220_l45_45183

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l45_45183


namespace binomial_12_3_equals_220_l45_45193

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l45_45193


namespace polygon_sides_l45_45434

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 + 360 = 1800) : n = 10 := by
  sorry

end polygon_sides_l45_45434


namespace slope_angle_vertical_line_l45_45429

theorem slope_angle_vertical_line : ∀ (x : ℝ),
  (2 * x - 3 = 1) → 
  ∃ θ : ℝ, θ = 90 := 
begin
  intros x hx,
  use 90,
  sorry
end

end slope_angle_vertical_line_l45_45429


namespace product_formula_l45_45175

theorem product_formula :
  (∏ n in Finset.range 15 + 1, (n * (n + 3)) / (n + 5)^2) = 75 / 1550400 := 
by
  -- Proof will go here
sorry

end product_formula_l45_45175


namespace vector_ratio_correct_l45_45254

noncomputable def vector_ratio (t : ℝ) : ℝ :=
if t = 1/2 then 1 else if t = 2/3 then 1/2 else 0

theorem vector_ratio_correct (O A B P : Type) [has_vector_space ℝ P] (t : ℝ)
  (cond : ∀ (O A B P : P), (vector_space.op (has_scalar.op (t : ℝ) B) (has_scalar.op (2 * t : ℝ) A (P : P))))
  : vector_ratio t = 1 ∨ vector_ratio t = 1/2 :=
by {
  have h1 : t = 1/2 ∨ t = 2/3, sorry,
  cases h1,
  { left, simp [vector_ratio, h1] },
  { right, simp [vector_ratio, h1] }
}

end vector_ratio_correct_l45_45254


namespace closest_integer_to_cube_root_of_250_l45_45540

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l45_45540


namespace binom_12_3_eq_220_l45_45185

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l45_45185


namespace quadratic_roots_real_coeffs_l45_45031

theorem quadratic_roots_real_coeffs
: (1 + complex.sqrt 2 * complex.i) is a root of (λ x : ℂ, x^2 + (b : ℝ) * x + (c : ℝ)) 
  → b = -2 ∧ c = 3 := sorry

end quadratic_roots_real_coeffs_l45_45031


namespace geometric_first_term_l45_45984

-- Define the conditions
def is_geometric_series (first_term : ℝ) (r : ℝ) (sum : ℝ) : Prop :=
  sum = first_term / (1 - r)

-- Define the main theorem
theorem geometric_first_term (r : ℝ) (sum : ℝ) (first_term : ℝ) 
  (h_r : r = 1/4) (h_S : sum = 80) (h_sum_formula : is_geometric_series first_term r sum) : 
  first_term = 60 :=
by
  sorry

end geometric_first_term_l45_45984


namespace find_five_numbers_l45_45065

theorem find_five_numbers :
  ∃ (a1 a2 a3 a4 a5 : ℝ), 
    a1 * a2 * a3 * a4 * a5 ≠ 0 ∧
    (a1 - 1) * (a2 - 1) * (a3 - 1) * (a4 - 1) * (a5 - 1) = a1 * a2 * a3 * a4 * a5 :=
by
  -- Example 1
  use [5, 6, 7, 8, -1]
  split
  case _ | intro h1
    calc (5: ℝ) * 6 * 7 * 8 * -1 = -1680 : sorry
  case _ | intro h2
    calc (5-1: ℝ) * (6-1) * (7-1) * (8-1) * (-1-1) = -1680 : sorry
    calc 5 * 6 * 7 * 8 * -1 = -1680 : sorry

  -- Example 2
  use [2, 2, 2, 2, -1/15]
  split
  case _ | intro h3
    calc (2: ℝ) * 2 * 2 * 2 * (-1/15) = -16/15 : sorry
  case _ | intro h4
    calc (2-1: ℝ) * (2-1) * (2-1) * (2-1) * ((-1/15)-1) = -16/15 : sorry
    calc 2 * 2 * 2 * 2 * (-1/15) = -16/15 : sorry

end find_five_numbers_l45_45065


namespace find_angle_B_find_triangle_area_l45_45334

-- Definitions to set up the problem
variables (A B C a b c : ℝ)
-- Conditions
axiom angleB_eq : B = 2 * Real.pi / 3
axiom side_b_eq : b = Real.sqrt 3
axiom angleA_eq : A = Real.pi / 4
axiom given_eq : 2 * b * Real.sin B = (2 * a + c) * Real.sin A + (2 * c + a) * Real.sin C

-- Proof for part (I)
theorem find_angle_B (A B C a b c : ℝ) : 2 * b * Real.sin B = (2 * a + c) * Real.sin A + (2 * c + a) * Real.sin C → B = 2 * Real.pi / 3 := 
sorry

-- Proof for part (II)
theorem find_triangle_area (A B C a b c : ℝ) : b = Real.sqrt 3 → A = Real.pi / 4 → B = 2 * Real.pi / 3 → 
area := (1 / 2) * b * c * Real.sin A  where c = (b * (Real.sin (Real.pi / 3) - Real.pi / 4 ))) / Real.sin (2 * Real.pi / 3)) → 
area = (3 - Real.sqrt 3) / 4 :=
sorry

end find_angle_B_find_triangle_area_l45_45334


namespace binomial_12_3_eq_220_l45_45205

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l45_45205


namespace non_parallel_non_concurrent_lines_l45_45257
-- Import the full math library

-- Define the problem context and the proof that n is not a power of two
theorem non_parallel_non_concurrent_lines (n : ℕ) 
  (h1 : n > 2)
  (h2 : ∀ i j : ℕ, i ≠ j → ∃ O : ℝ × ℝ, ∃ α : ℝ, (0 < α ∧ α < 180 ∧ 
    rotate_around O α (line i) = line j)) : 
  ¬ (∃ k : ℕ, n = 2 ^ k ∧ k ≥ 2) :=
sorry

end non_parallel_non_concurrent_lines_l45_45257


namespace solution_l45_45002

variable (x y z : ℝ)

noncomputable def problem := 
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x^2 + x * y + y^2 = 48 →
  y^2 + y * z + z^2 = 25 →
  z^2 + z * x + x^2 = 73 →
  x * y + y * z + z * x = 40

theorem solution : problem := by
  intros
  sorry

end solution_l45_45002


namespace remainder_t50_l45_45876

def sequence (T : ℕ → ℕ) : Prop :=
  T 1 = 3 ∧ ∀ n > 1, T n = 3 ^ T (n - 1)

theorem remainder_t50 (T : ℕ → ℕ) (h : sequence T) : T 50 % 4 = 3 :=
by
  -- Definitions based on the condition for the sequence
  have h1 : T 1 = 3 := h.1
  have h2 : ∀ n > 1, T n = 3 ^ T (n - 1) := h.2
  sorry

end remainder_t50_l45_45876


namespace radian_measure_of_acute_angle_l45_45900

theorem radian_measure_of_acute_angle 
  (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) (h3 : r3 = 2)
  (θ : ℝ) (S U : ℝ) 
  (hS : S = U * 9 / 14) (h_total_area : (π * r1^2) + (π * r2^2) + (π * r3^2) = S + U) :
  θ = 1827 * π / 3220 :=
by
  -- proof goes here
  sorry

end radian_measure_of_acute_angle_l45_45900


namespace part1_part2_l45_45283

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 1

theorem part1 (x : ℝ) (hx : 0 < x) : f 1 x ≤ 0 := by
  sorry

theorem part2 (x : ℝ) (hx : 0 < x) (hxpi : x < Real.pi / 2) : 
  Real.exp x * Real.sin x - x > f 1 x := by
  sorry

end part1_part2_l45_45283


namespace problem_l45_45683

theorem problem (a : ℕ → ℝ) : 
    ((1 - 2*x)^10 = ∑ i in Finset.range 11, a i * x^i) →
    a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 + 5 * a 5 + 6 * a 6 + 7 * a 7 + 8 * a 8 + 9 * a 9 + 10 * a 10 = 20 :=
by
  sorry

end problem_l45_45683


namespace three_rays_with_common_point_l45_45354

theorem three_rays_with_common_point (x y : ℝ) :
  (∃ (common : ℝ), ((5 = x - 1 ∧ y + 3 ≤ 5) ∨ 
                     (5 = y + 3 ∧ x - 1 ≤ 5) ∨ 
                     (x - 1 = y + 3 ∧ 5 ≤ x - 1 ∧ 5 ≤ y + 3)) 
  ↔ ((x = 6 ∧ y ≤ 2) ∨ (y = 2 ∧ x ≤ 6) ∨ (y = x - 4 ∧ x ≥ 6))) :=
sorry

end three_rays_with_common_point_l45_45354


namespace find_f_5_l45_45690

noncomputable def f : ℕ → ℕ
| x := if x ≥ 10 then x - 2 else f(f(x + 6))

theorem find_f_5 : f 5 = 11 := by sorry

end find_f_5_l45_45690


namespace average_mileage_is_correct_l45_45153

noncomputable def total_distance : ℝ := 150 + 200
noncomputable def sedan_efficiency : ℝ := 25
noncomputable def truck_efficiency : ℝ := 15
noncomputable def sedan_miles : ℝ := 150
noncomputable def truck_miles : ℝ := 200

noncomputable def total_gas_used : ℝ := (sedan_miles / sedan_efficiency) + (truck_miles / truck_efficiency)
noncomputable def average_gas_mileage : ℝ := total_distance / total_gas_used

theorem average_mileage_is_correct :
  average_gas_mileage = 18.1 := 
by
  sorry

end average_mileage_is_correct_l45_45153


namespace colorful_grid_coloring_l45_45621

-- Define the concept of a colorful unit square
structure ColorfulUnitSquare (Color : Type) :=
  (left right top bottom : Color)
  (isColorful : 
    (left ≠ right) ∧ 
    (left ≠ top) ∧ 
    (left ≠ bottom) ∧ 
    (right ≠ top) ∧ 
    (right ≠ bottom) ∧ 
    (top ≠ bottom))

-- Define the concept of the grid and the coloring problem
def Number_of_Colorings (Color : Type) [DecidableEq Color] [Fintype Color] :=
  {segments : Fin 10 → Color // 
    let g := Matrix (Fin 3) (Fin 2) Color,
        g :=
          λ i j, segments ((i * 2) + j).val,
          0 ≤ i ∧ i < 3 ∧
          0 ≤ j ∧ j < 2 → 
      ((ColorfulUnitSquare.mk 
        (g 0 0) (g 1 0) (g 0 1) (g 1 1)).isColorful ∧
        (ColorfulUnitSquare.mk 
        (g 1 0) (g 2 0) (g 1 1) (g 2 1)).isColorful ∧
        (ColorfulUnitSquare.mk 
        (g 2 0) (g 0 0) (g 2 1) (g 0 1)).isColorful) }

-- The main theorem to state the proof problem
theorem colorful_grid_coloring {Color : Type} [DecidableEq Color] [Fintype Color] (h : Fintype.card Color = 3) :
  Fintype.card (Number_of_Colorings Color) = 5184 :=
sorry

end colorful_grid_coloring_l45_45621


namespace parabola_and_x4_value_l45_45326

theorem parabola_and_x4_value :
  (∀ P, dist P (0, 1/2) = dist P (x, -1/2) → ∃ y, P = (x, y) ∧ x^2 = 2 * y) ∧
  (∀ (x1 x2 : ℝ), x1 = 6 → x2 = 2 → ∃ x4, 1/x4 = 1/((3/2) : ℝ) + 1/x2 ∧ x4 = 6/7) :=
by
  sorry

end parabola_and_x4_value_l45_45326


namespace find_north_speed_l45_45449

-- Define the variables and conditions
variables (v : ℝ)  -- the speed of the cyclist going towards the north
def south_speed : ℝ := 25  -- the speed of the cyclist going towards the south is 25 km/h
def time_taken : ℝ := 1.4285714285714286  -- time taken to be 50 km apart
def distance_apart : ℝ := 50  -- distance apart after given time

-- Define the hypothesis based on the conditions
def relative_speed (v : ℝ) : ℝ := v + south_speed
def distance_formula (v : ℝ) : Prop :=
  distance_apart = relative_speed v * time_taken

-- The statement to prove
theorem find_north_speed : distance_formula v → v = 10 :=
  sorry

end find_north_speed_l45_45449


namespace geometric_sequence_properties_l45_45411

open Nat Real

noncomputable def aₙ (n : ℕ) : ℝ := (32 / 3) * (1 / 2) ^ (n - 1)

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def Sn (n : ℕ) (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 1 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_properties (a : ℕ → ℝ) (q : ℝ)
  (h₁ : a 1 + a 6 = 11)
  (h₂ : a 3 * a 4 = 32 / 9)
  (h₃ : 0 < q ∧ q < 1)
  (h₄ : is_geometric_sequence a q) :
  a = aₙ ∧ Sn 6 a q = 21 :=
by
  sorry

end geometric_sequence_properties_l45_45411


namespace max_k_value_l45_45750

noncomputable def find_max_k (x y k : ℝ) : ℝ :=
  if (x > 0) ∧ (y > 0) ∧ (k > 0) ∧ (5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x))
  then (k = (-4 + Real.sqrt 29) / 13)
  else 0

theorem max_k_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (k : ℝ)
  (h3 : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  (k = (-4 + Real.sqrt 29) / 13) :=
begin
  sorry
end

end max_k_value_l45_45750


namespace decreasing_function_condition_l45_45064

theorem decreasing_function_condition (m : ℝ) :
  (∀ x : ℝ, 0 < x → deriv (λ x : ℝ, (m^2 - m - 1) * x^(-5 * m - 3)) x < 0) ↔ m > 1 :=
by
  sorry

end decreasing_function_condition_l45_45064


namespace mrs_hilt_initial_marbles_l45_45374

theorem mrs_hilt_initial_marbles (lost_marble : ℕ) (remaining_marble : ℕ) (h1 : lost_marble = 15) (h2 : remaining_marble = 23) : 
    (remaining_marble + lost_marble) = 38 :=
by
  sorry

end mrs_hilt_initial_marbles_l45_45374


namespace tangent_to_x_axis_at_1_zeros_of_f_on_1_e_l45_45725

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a + a / x

theorem tangent_to_x_axis_at_1 (a : ℝ) (h : 0 < a) : 
    deriv (fun x => f x a) 1 = 0 ∧ f 1 a = 0 ↔ a = 1 := 
by
  sorry

theorem zeros_of_f_on_1_e (a : ℝ) (h : 0 < a) : 
    (∀ x, x ∈ Ioo 1 Real.exp → (f x a = 0 ↔ 1 < a ∧ a < Real.exp / (Real.exp - 1))) ∨
    (∀ x, x ∈ Ioo 1 Real.exp → (f x a ≠ 0)) :=
by
  sorry

end tangent_to_x_axis_at_1_zeros_of_f_on_1_e_l45_45725


namespace product_common_divisors_150_30_l45_45670

theorem product_common_divisors_150_30 :
  let divisors_150 := [1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150]
  let divisors_30 := [1, 2, 3, 5, 6, 10, 15, 30]
  let common_divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  (common_divisors.map (id)).prod * (common_divisors.map (λ x, -x)).prod = 182250000 :=
by
  sorry

end product_common_divisors_150_30_l45_45670


namespace limit_of_Sn_over_n_an_l45_45330

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) > a n) ∧
  (∀ n : ℕ, (a (n + 1))^2 + (a n)^2 + 1 = 2 * (a (n + 1) + a n + 2 * (a (n + 1)) * (a n)))

noncomputable def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range n, a (k + 1)

theorem limit_of_Sn_over_n_an (a : ℕ → ℝ) (h : sequence a) :
  tendsto (λ n : ℕ, (S_n a n) / (n * a n)) at_top (𝓝 (1 / 3)) :=
sorry

end limit_of_Sn_over_n_an_l45_45330


namespace area_to_perimeter_ratio_l45_45096

-- Define the side length of the equilateral triangle
def side_length : ℝ := 6

-- Define the area of an equilateral triangle given its side length
def equilateral_triangle_area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the perimeter of an equilateral triangle given its side length
def equilateral_triangle_perimeter (s : ℝ) : ℝ := 3 * s

-- Prove that the ratio of the area to the perimeter for side length 6 is sqrt(3)/2
theorem area_to_perimeter_ratio : 
  equilateral_triangle_area side_length / equilateral_triangle_perimeter side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_to_perimeter_ratio_l45_45096


namespace min_value_quadratic_form_l45_45240

theorem min_value_quadratic_form : ∀ x y : ℝ, ∃ m ∈ set.Iio 1, (m = x^2 - x * y + y^2) :=
by
  intros x y
  use 0
  sorry

end min_value_quadratic_form_l45_45240


namespace radius_correct_l45_45969

open Real

noncomputable def radius_of_circle
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop) : ℝ := sorry

theorem radius_correct
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop)
  (h1 : tangent_length = 12) 
  (h2 : secant_internal_segment = 10) 
  (h3 : tangent_secant_perpendicular) : radius_of_circle tangent_length secant_internal_segment tangent_secant_perpendicular = 13 := 
sorry

end radius_correct_l45_45969


namespace calculate_expression_l45_45176

theorem calculate_expression :
  (-1)^(49 : ℤ) + (3 : ℤ)^(Int.ofNat ((2 ^ 3 + 5 ^ 2 - 4 ^ 2)!)) = -1 + (3 : ℤ)^(355687428096000 : ℤ) :=
by 
  sorry

end calculate_expression_l45_45176


namespace gum_given_by_steve_l45_45446

theorem gum_given_by_steve :
  ∀ (t_0 t_1 g : ℕ), t_0 = 38 → t_1 = 54 → (t_1 = t_0 + g) → g = 16 :=
by
  intros t_0 t_1 g h0 h1 h2
  rw [h1, h0] at h2
  exact Nat.add_left_cancel h2

end gum_given_by_steve_l45_45446


namespace sin_double_angle_given_tan_l45_45270

theorem sin_double_angle_given_tan :
  ∀ (α : ℝ), tan α = 3 → (sin (2 * α) = 3 / 5) :=
by
  intros α h
  sorry

end sin_double_angle_given_tan_l45_45270


namespace velocity_zero_times_l45_45961

noncomputable def s (t : ℝ) : ℝ := (1 / 4) * t^4 - (5 / 3) * t^3 + 2 * t^2

theorem velocity_zero_times :
  {t : ℝ | deriv s t = 0} = {0, 1, 4} :=
by 
  sorry

end velocity_zero_times_l45_45961


namespace Beth_can_win_l45_45987

def nim_value (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 4
  | 6 => 3
  | 7 => 5
  | _ => sorry -- Placeholder for more complex computations if needed.

def nim_sum (a b c : Nat) : Nat :=
  a xor b xor c

theorem Beth_can_win (w1 w2 w3 : Nat) (h1 : w1 = 6) (h2 : w2 = 2) (h3 : w3 = 1) : nim_sum (nim_value w1) (nim_value w2) (nim_value w3) = 0 :=
by
  rw [h1, h2, h3]
  simp [nim_value, nim_sum]
  exact Eq.refl 0

end Beth_can_win_l45_45987


namespace root_interval_l45_45426

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem root_interval : ∃ x ∈ Ioo (2.5 : ℝ) 2.75, f x = 0 :=
sorry

end root_interval_l45_45426


namespace ratio_of_ticket_prices_l45_45943

-- Given conditions
def num_adults := 400
def num_children := 200
def adult_ticket_price : ℕ := 32
def total_amount : ℕ := 16000
def child_ticket_price (C : ℕ) : Prop := num_adults * adult_ticket_price + num_children * C = total_amount

theorem ratio_of_ticket_prices (C : ℕ) (hC : child_ticket_price C) :
  adult_ticket_price / C = 2 :=
by
  sorry

end ratio_of_ticket_prices_l45_45943


namespace white_queen_letter_problem_l45_45036

theorem white_queen_letter_problem
  (letters : Fin 4) 
  (conditions : ∀ (correct : ℕ), 
                 correct ∈ {0, 1, 2, 3} ∧
                 ((correct ≠ 3) ∧ (correct ≠ 1)) ∧ 
                 correct ∈ {2}) :
  ∃ (correct_letters : ℕ), correct_letters = 2 :=
begin
  sorry
end

end white_queen_letter_problem_l45_45036


namespace differential_equation_solutions_l45_45790

-- Defining the equation and conditions
def equation (x y y' : ℝ) (a : ℝ) : Prop := y = x * y' + a / (2 * y')

-- The family of straight line solutions in the form y = Cx + a/(2C)
def family_of_straight_lines (C x a : ℝ) : ℝ := C * x + a / (2 * C)

-- The envelope solution y^2 = 2ax
def envelope_solution (x y a : ℝ) : Prop := y^2 = 2 * a * x

theorem differential_equation_solutions (a : ℝ) : 
  (∀ (C x : ℝ), equation x (family_of_straight_lines C x a) C a) ∧ 
  (∀ (x y : ℝ), envelope_solution x y a → equation x y (√(a / (2 * x))) a ∨ equation x y (-√(a / (2 * x))) a) := 
by 
  sorry

end differential_equation_solutions_l45_45790


namespace inscribe_circle_possible_if_sums_equal_l45_45383

-- Define basic terms and conditions
variable (A B C D P Q R S : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
variable [metric_space P] [metric_space Q] [metric_space R] [metric_space S]

-- Define convex quadrilateral and sum of opposite sides
def is_convex_quadrilateral (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] : Prop :=
convex [A, B, C, D] 

def sums_of_opposite_sides_equal (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] : Prop :=
∃ (AB CD AD BC : ℝ), AB + CD = AD + BC

-- The mathematical problem
theorem inscribe_circle_possible_if_sums_equal :
is_convex_quadrilateral A B C D →
sums_of_opposite_sides_equal A B C D →
∃ (circle : circumscribable C D), 
(∃ (tangent_points : tangent_points P Q R S), 
(tangent_points_on_quadrilateral P Q R S A B C D circle)) :=
sorry

end inscribe_circle_possible_if_sums_equal_l45_45383


namespace route_B_quicker_by_16_seconds_l45_45830

noncomputable def travel_time_route_A : ℝ :=
  (8 / 40) * 60  -- time in minutes

noncomputable def travel_time_route_B : ℝ :=
  ((5.5 / 45) * 60) + ((1 / 25) * 60) + ((0.5 / 15) * 60)  -- time in minutes

theorem route_B_quicker_by_16_seconds :
  (travel_time_route_A - travel_time_route_B) * 60 ≈ 16 :=
by
  sorry

end route_B_quicker_by_16_seconds_l45_45830


namespace average_method_eq1_average_method_eq2_l45_45465

-- Define the average method for solving the quadratic equation
theorem average_method_eq1 :
  (x + 5)^2 - 3^2 = 40 → (x + 5)^2 = 49 → x = -5 + 7 ∨ x = -5 - 7 :=
sorry

noncomputable def part1_values :=
  let a := 5
  let b := 3
  let c := 2
  let d := -12
  (a, b, c, d)

theorem average_method_eq2 :
  (x + 2)^2 - 4^2 = 4 → (x + 2)^2 = 20 → x = -2 + 2 * sqrt 5 ∨ x = -2 - 2 * sqrt 5 :=
sorry

noncomputable def part2_solutions :=
  let x1 := -2 + 2 * (Real.sqrt 5)
  let x2 := -2 - 2 * (Real.sqrt 5)
  (x1, x2)

end average_method_eq1_average_method_eq2_l45_45465


namespace minimum_value_of_expression_l45_45239

theorem minimum_value_of_expression :
  ∀ x y : ℝ, x^2 - x * y + y^2 ≥ 0 :=
by
  sorry

end minimum_value_of_expression_l45_45239


namespace p_complete_work_alone_l45_45577

-- Let's define the conditions and the result.
def work_time_p_alone (p_rate q_rate w total_days: ℕ) := 
  ∀ (p_alone q_help_days : ℕ), 
    p_rate * p_alone + (p_rate + q_rate) * q_help_days = w * total_days → 
      p_alone = 40

theorem p_complete_work_alone :
  ∀ (W : ℕ), 
    ∀ (p_rate q_rate : ℕ), 
    p_rate * 40 + q_rate * 24 = W → 
    p_rate + q_rate * 9 = W - q_rate * 24 →
    p_rate = 40 :=
begin
  intros W p_rate q_rate hp hq, 
  have p_complete_time := (25 - 16) * p_rate + 16 * (p_rate + q_rate),
  sorry
end

end p_complete_work_alone_l45_45577


namespace f_2017_eq_l45_45874

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f (noncomputable for general functions)

-- Stating the conditions as hypotheses
axiom f_symm : ∀ x : ℝ, f(x) = f(-x)
axiom f_periodic_neg : ∀ x : ℝ, f(x + 3) = -f(x)
axiom f_interval : ∀ x : ℝ, x ∈ set.Ioo (3/2) (5/2) → f(x) = (1/2)^x

-- Main theorem to prove
theorem f_2017_eq :
  f(2017) = -1/4 :=
sorry

end f_2017_eq_l45_45874


namespace net_effect_sale_value_reduction_increase_l45_45928

-- Define the given conditions
variables (P Q : ℝ) -- Define the original price and quantity

-- Theorem to prove the net effect on sale value
theorem net_effect_sale_value_reduction_increase (P Q : ℝ) :
  let new_price := 0.82 * P
  let new_quantity := 1.72 * Q
  let original_sale_value := P * Q
  let new_sale_value := new_price * new_quantity
  let net_effect := (new_sale_value / original_sale_value - 1) * 100
  in net_effect = 41.04 :=
by
  -- The proof is omitted, use sorry
  sorry

end net_effect_sale_value_reduction_increase_l45_45928


namespace binom_12_3_eq_220_l45_45184

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l45_45184


namespace log_inequality_solution_l45_45029

theorem log_inequality_solution (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) :
  (log x (2 * x) ≤ real.sqrt (log x (2 * x^3))) ↔ (0 < x ∧ x ≤ real.cbrt 1 / real.cbrt 2) ∨ (x ≥ 2) :=
sorry

end log_inequality_solution_l45_45029


namespace closest_integer_to_cubert_of_250_is_6_l45_45556

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45556


namespace conjugate_of_z_l45_45865

-- Define the imaginary unit
def i := Complex.I

-- Define the complex number z
def z := i * (i + 1)

-- State the theorem
theorem conjugate_of_z : Complex.conj z = -1 - i := 
by
  -- the proof would go here
  sorry

end conjugate_of_z_l45_45865


namespace keiko_speed_l45_45800

theorem keiko_speed (a b s : ℝ) 
  (width : ℝ := 8) 
  (radius_inner := b) 
  (radius_outer := b + width)
  (time_difference := 48) 
  (L_inner := 2 * a + 2 * Real.pi * radius_inner)
  (L_outer := 2 * a + 2 * Real.pi * radius_outer) :
  (L_outer / s = L_inner / s + time_difference) → 
  s = Real.pi / 3 :=
by 
  sorry

end keiko_speed_l45_45800


namespace integral_sqrt_ln_square_eq_l45_45993

noncomputable def definite_integral_sqrt_ln_square : ℝ :=
  ∫ x in 1..Real.exp 1, Real.sqrt x * (Real.log x) ^ 2

theorem integral_sqrt_ln_square_eq :
  definite_integral_sqrt_ln_square = (10 * Real.sqrt (Real.exp 1)^3 - 16) / 27 :=
by
  sorry

end integral_sqrt_ln_square_eq_l45_45993


namespace nesting_rectangles_exists_l45_45692

theorem nesting_rectangles_exists :
  ∀ (rectangles : List (ℕ × ℕ)), rectangles.length = 101
    ∧ (∀ r ∈ rectangles, r.fst ≤ 100 ∧ r.snd ≤ 100) 
    → ∃ (A B C : ℕ × ℕ), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles 
    ∧ (A.fst < B.fst ∧ A.snd < B.snd) 
    ∧ (B.fst < C.fst ∧ B.snd < C.snd) := 
by sorry

end nesting_rectangles_exists_l45_45692


namespace sqrt_of_expression_l45_45997

theorem sqrt_of_expression :
  Real.sqrt (4^4 * 9^2) = 144 :=
sorry

end sqrt_of_expression_l45_45997


namespace evaluate_expression_l45_45635

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 :=
by
  sorry

end evaluate_expression_l45_45635


namespace boys_and_girls_arrangement_l45_45077

theorem boys_and_girls_arrangement : 
  ∃ (arrangements : ℕ), arrangements = 48 :=
  sorry

end boys_and_girls_arrangement_l45_45077


namespace solve_for_n_l45_45393

theorem solve_for_n (n : ℕ) : (8 ^ n) * (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 4 → n = 2 :=
by 
  intro h
  sorry

end solve_for_n_l45_45393


namespace cara_pairs_l45_45637

theorem cara_pairs (friends_males_females : Σ n, Σ m, Σ f, (friends : Finset (Fin 6)) → m + f = 6 ∧ m = 3 ∧ f = 3) :
  ∃ valid_pairs, 12 = valid_pairs :=
by
  -- Assuming n represents the total number of friends, m represents the number of males, and f represents the number of females.
  let ⟨n, ⟨m, ⟨f, ⟨friends, ⟨h_sum, ⟨h_male, h_female⟩⟩⟩⟩⟩⟩ := friends_males_females
  -- Use the conditions to define valid pairs
  sorry

end cara_pairs_l45_45637


namespace closest_integer_to_cubert_of_250_is_6_l45_45557

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l45_45557


namespace cloth_amount_in_30_days_l45_45781

/-- A woman weaves cloth in a decreasing arithmetic sequence: 
she weaves 5 feet on the first day and 1 foot on the last day. 
Prove that the total amount of cloth she weaves in 30 days is 90 feet. -/
theorem cloth_amount_in_30_days : 
  ∀ (a₁ aₙ n : ℕ), a₁ = 5 → aₙ = 1 → n = 30 → 
  (∑ i in finset.range n, (a₁ - (a₁ - aₙ) * i / (n - 1)) = 90) :=
by
  intros a₁ aₙ n h₀ h₁ h₂
  sorry

end cloth_amount_in_30_days_l45_45781


namespace binomial_12_3_equals_220_l45_45196

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l45_45196


namespace correct_sampling_methods_l45_45905

-- Defining the types and categories
def Boxes : Type := Fin 10 -- 10 boxes of black pen refills
def Rows := Fin 32 -- 32 rows in the cinema
def SeatsPerRow := Fin 60 -- Each row has 60 seats
def CinemaAudience := Fin 32 -- 32 audience members needed for discussion
def TotalStaff := 160
def GeneralTeachers := 120
def AdminStaff := 16
def LogisticsStaff := 24
def SampleSizeStaff := 20

-- Assertions based on the problem conditions
axiom draw_boxes_from_boxes (n : Fin 10) : Prop -- Drawing 2 from 10 boxes
axiom cinema_audience_feedback (n : Fin 32) : Prop -- Selecting 32 audience members
axiom staff_survey (n : Fin 20) : Prop -- Survey of 20 staff members

-- Define a proposition to assert the correct sampling methods
theorem correct_sampling_methods :
  (draw_boxes_from_boxes Boxes -> SimpleRandomSampling) ∧
  (cinema_audience_feedback CinemaAudience -> SystematicSampling) ∧
  (staff_survey SampleSizeStaff -> StratifiedSampling) :=
sorry

end correct_sampling_methods_l45_45905


namespace sum_of_primes_l45_45869

theorem sum_of_primes (A B C : ℕ) (h₁ : A.prime) (h₂ : B.prime) (h₃ : C.prime)
  (h₄ : 291 * A = C * (291 - A + B)) : 
  A + B + C = 221 :=
  sorry

end sum_of_primes_l45_45869


namespace simplify_expression_l45_45846

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 18 = 45 * w + 18 := by
  sorry

end simplify_expression_l45_45846


namespace p_x_range_l45_45303

variable (x : ℝ)

def inequality_condition := x^2 - 5*x + 6 < 0
def polynomial_function := x^2 + 5*x + 6

theorem p_x_range (x_ineq : inequality_condition x) : 
  20 < polynomial_function x ∧ polynomial_function x < 30 :=
sorry

end p_x_range_l45_45303


namespace conjugate_of_complex_number_l45_45407

open Complex

theorem conjugate_of_complex_number : 
  let i := Complex.I in 
  let z := i * (i + 1) in
  conj z = -1 - i := 
by 
  sorry

end conjugate_of_complex_number_l45_45407


namespace sum_common_divisors_72_24_l45_45678

theorem sum_common_divisors_72_24 :
  let divisors_72 := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}
  let divisors_24 := {1, 2, 3, 4, 6, 8, 12, 24}
  let common_divisors := divisors_72 ∩ divisors_24
  ∑ d in common_divisors, d = 60 :=
by
  let divisors_72 := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}
  let divisors_24 := {1, 2, 3, 4, 6, 8, 12, 24}
  let common_divisors := divisors_72 ∩ divisors_24
  have h1 : common_divisors = {1, 2, 3, 4, 6, 8, 12, 24} :=
    by simp [divisors_72, divisors_24, common_divisors]
  have h2 : ∑ d in common_divisors, d = 60 :=
    by norm_num [common_divisors]
  sorry

end sum_common_divisors_72_24_l45_45678


namespace closest_integer_to_cube_root_of_250_l45_45487

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l45_45487


namespace num_revolutions_l45_45224

theorem num_revolutions (diameter : ℝ) (distance_mile : ℝ) (distance_ft_in_mile : ℝ) : diameter = 8 → distance_mile = 1 → distance_ft_in_mile = 5280 → 
  (distance_ft_in_mile / ( π * diameter )) = (660 / π) :=
by
  intros hdiameter hdistMile hdistFtInMile
  rw [hdiameter, hdistMile, hdistFtInMile]
  sorry

end num_revolutions_l45_45224


namespace points_concyclic_l45_45130

noncomputable theory
open_locale classical

variables {A B C P Q C1 B1 : Type}
  [point : A → P → B → C → Type] [line : A → B → C → Type]
  [cyclic : A → B → P → C1 → Type] [parallel : Q → C1 → C → A → Type]

def acute_triangle (A B C : Type) [triangle : A → B → C → Type] := sorry
def on_side (P Q : Type) (BC : Type) := sorry
def cyclic_quad (APBC1 : Type) := sorry
def parallel_lines (QC1 CA : Type) := sorry
def opposite_sides (C1 Q AB : Type) := sorry
def concyclic (B1 C1 P Q : Type) := sorry

theorem points_concyclic
  (h1 : acute_triangle A B C)
  (h2 : on_side P Q BC)
  (h3 : cyclic_quad APBC1)
  (h4 : parallel_lines QC1 CA)
  (h5 : opposite_sides C1 Q AB)
  (h6 : cyclic_quad APCB1)
  (h7 : parallel_lines QB1 BA)
  (h8 : opposite_sides B1 Q AC) :
  concyclic B1 C1 P Q :=
sorry

end points_concyclic_l45_45130


namespace maximize_area_CDFE_l45_45352

-- Given the side lengths of the rectangle
def AB : ℝ := 2
def AD : ℝ := 1

-- Definitions for points E and F
def AE (x : ℝ) : ℝ := x
def AF (x : ℝ) : ℝ := x

-- The formula for the area of quadrilateral CDFE
def area_CDFE (x : ℝ) : ℝ := 
  0.5 * x * (3 - 2 * x)

theorem maximize_area_CDFE : 
  ∃ x : ℝ, x = 3 / 4 ∧ area_CDFE x = 9 / 16 :=
by 
  sorry

end maximize_area_CDFE_l45_45352
