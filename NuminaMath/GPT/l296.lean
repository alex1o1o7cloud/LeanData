import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Order.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry.Euclidean.Space
import Mathlib.Analysis.SpecialFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.GCD
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.NumberTheory.PrimeDivisors
import Mathlib.Probability.ProbabilityMassFunc
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import data.real.basic
import function.basic

namespace integral_evaluation_l296_296859

theorem integral_evaluation :
  (∫ x in 0..(Real.pi / 2), (Real.cos x) / (1 + Real.cos x + Real.sin x)^2) = Real.log 2 - 1/2 :=
by
  sorry

end integral_evaluation_l296_296859


namespace probability_at_least_8_heads_eq_l296_296006

-- Definitions for the given conditions
def total_outcomes : ℕ := 1024

def successful_outcomes : ℕ :=
  Nat.choose 10 8 + Nat.choose 10 9 + Nat.choose 10 10

def probability_of_success : ℚ :=
  successful_outcomes / total_outcomes

-- Theorem stating the final proof problem
theorem probability_at_least_8_heads_eq :
  probability_of_success = 7 / 128 := by
  sorry

end probability_at_least_8_heads_eq_l296_296006


namespace concyclic_points_l296_296213

open EuclideanGeometry

-- Define the existence of points A, B, C forming a triangle
variables {A B C : Point}

-- Define point D as the intersection of the internal angle bisector from A to side BC
def is_angle_bisector (A B C D : Point) : Prop :=
  dist A D / dist D B = dist A D / dist D C

-- Define the perpendicular bisector of AD, intersecting angle bisectors from B and C at M and N respectively
def is_perp_bisector (A D M N B C : Point) : Prop :=
  is_bisector A D M ∧ is_bisector A D N

-- Our objective proof statement
theorem concyclic_points 
  (H_triangle : is_triangle A B C)
  (H_D : is_angle_bisector A B C D)
  (H_perp_bisector : is_perp_bisector A D M N B C) :
  concyclic {A, I, M, N} :=
sorry -- Proof goes here


end concyclic_points_l296_296213


namespace expression_for_F_range_of_k_l296_296925

-- Definitions
def f (x : ℝ) (a b : ℝ) := a * x^2 + b * x + 1

def F (x : ℝ) (a b : ℝ) : ℝ :=
  if x > 0 then f x a b else -f x a b

def g (x k : ℝ) (a b : ℝ) := f x a b - k * x

-- Theorem Statements
theorem expression_for_F 
  (a b : ℝ) (h_a_gt_0 : a > 0) 
  (h_disc : b^2 - 4 * a = 0) 
  (h_f_neg1 : f (-1) a b = 0) :
  ∀ x : ℝ, 
   (F x a b = ite (x > 0) (f x a b) (-f x a b)) :=
sorry

theorem range_of_k 
  (a b : ℝ) (h_a_gt_0 : a > 0) 
  (h_disc : b^2 - 4 * a = 0) 
  (h_f_neg1 : f (-1) a b = 0) 
  (k : ℝ) :
  (x ∈ Icc (-2 : ℝ) 2 → monotone_on (g x k a b) (Icc (-2 : ℝ) 2)) ↔ 
  k ≥ 6 ∨ k ≤ -2 :=
sorry

end expression_for_F_range_of_k_l296_296925


namespace box_weight_l296_296352

noncomputable def volume (h w l : ℕ) : ℕ := h * w * l

noncomputable def density (weight volume : ℕ) : ℚ := weight / volume

noncomputable def weight (density : ℚ) (volume : ℕ) : ℚ := density * volume

theorem box_weight :
  let height1 := 4
  let width1 := 5
  let length1 := 10
  let weight1 := 150
  let height2 := 8
  let width2 := 5
  let length2 := 15 in
  weight (density weight1 (volume height1 width1 length1)) (volume height2 width2 length2) = 450 :=
by
  sorry

end box_weight_l296_296352


namespace coeff_x5_expansion_l296_296642

theorem coeff_x5_expansion (x : ℂ) : 
  (polynomial.coeff ((1 + polynomial.X - polynomial.X^2)^6) 5 = 6) :=
sorry

end coeff_x5_expansion_l296_296642


namespace circle_eq_given_condition_tangent_line_eq_given_conditions_l296_296115

-- Problem 1
theorem circle_eq_given_condition (A: ℝ × ℝ) (O: ℝ × ℝ) (l: ℝ → ℝ) 
  (radius: ℝ) (center_on_line: ℝ × ℝ → Prop) (CO_eq_CA: Prop) : 
  (center_on_line (2, 0)) → (x - 2)^2 + y^2 = 1 := 
by sorry

-- Problem 2
theorem tangent_line_eq_given_conditions (A: ℝ × ℝ) (C: ℝ × ℝ) 
  (line1: ℝ → ℝ) (line2: ℝ → ℝ) (tangent_through_A: ℝ → ℝ → Prop) :
  (center_on_line (3, 2)) → (tangent_through_A (3*x + 4*y - 12 = 0) (x = 4)) := 
by sorry

end circle_eq_given_condition_tangent_line_eq_given_conditions_l296_296115


namespace kevin_speed_first_half_l296_296561

-- Let's define the conditions as variables and constants
variable (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ)
variable (time_20mph : ℝ) (time_8mph : ℝ) (distance_first_half : ℕ)
variable (speed_first_half : ℝ)

-- Conditions from the problem
def conditions (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ) : Prop :=
  total_distance = 17 ∧ 
  distance_20mph = 20 * 1 / 2 ∧
  distance_8mph = 8 * 1 / 4

-- Proof objective based on conditions and correct answer
theorem kevin_speed_first_half (
  h : conditions total_distance distance_20mph distance_8mph
) : speed_first_half = 10 := by
  sorry

end kevin_speed_first_half_l296_296561


namespace function_satisfies_conditions_l296_296968

theorem function_satisfies_conditions (f : ℝ → ℝ) (h1 : ∀ x, f(x + π) = f(-x)) (h2 : ∀ x, f(-x) = f(x)) :
  f = λ x, abs (sin x) :=
by
  -- Proof will go here
  sorry

end function_satisfies_conditions_l296_296968


namespace value_of_x_l296_296653

theorem value_of_x :
  ∀ (A : ℕ) (x : ℕ), 10 ^ 4 ≤ A ∧ A ≤ 10 ^ 5 ∧ A = x * 10 ^ 4 + 1 → x = 9 :=
by
  intros A x h
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end value_of_x_l296_296653


namespace minimum_pieces_chessboard_l296_296229

theorem minimum_pieces_chessboard (n : ℕ) (pieces : ℕ) :
  let board_size : ℕ := 20,
      min_pieces : ℕ := 121 in
  (pieces < min_pieces ∧ (∀ i j : fin board_size, pieces / board_size < board_size / 2)) →
  false :=
by -- Proof goes here
  sorry

end minimum_pieces_chessboard_l296_296229


namespace roots_solution_l296_296573

theorem roots_solution (p q : ℝ) (h1 : (∀ x : ℝ, (x - 3) * (3 * x + 8) = x^2 - 5 * x + 6 → (x = p ∨ x = q)))
  (h2 : p + q = 0) (h3 : p * q = -9) : (p + 4) * (q + 4) = 7 :=
by
  sorry

end roots_solution_l296_296573


namespace minimum_degree_of_g_l296_296253

variables {R : Type*} [CommRing R]
variables {f g h : R[X]}

theorem minimum_degree_of_g (hf : f.degree = 10) (hh : h.degree = 11) (h_eq : 5 * f + 3 * g = h) :
  g.degree ≥ 11 :=
sorry

end minimum_degree_of_g_l296_296253


namespace toll_for_18_wheel_truck_l296_296293

-- Define the number of axles given the conditions
def num_axles (total_wheels rear_axle_wheels front_axle_wheels : ℕ) : ℕ :=
  let rear_axles := (total_wheels - front_axle_wheels) / rear_axle_wheels
  rear_axles + 1

-- Define the toll calculation given the number of axles
def toll (axles : ℕ) : ℝ :=
  1.50 + 0.50 * (axles - 2)

-- Constants specific to the problem
def total_wheels : ℕ := 18
def rear_axle_wheels : ℕ := 4
def front_axle_wheels : ℕ := 2

-- Calculate the number of axles for the given truck
def truck_axles : ℕ := num_axles total_wheels rear_axle_wheels front_axle_wheels

-- The actual statement to prove
theorem toll_for_18_wheel_truck : toll truck_axles = 3.00 :=
  by
    -- proof will go here
    sorry

end toll_for_18_wheel_truck_l296_296293


namespace tangent_line_eq_l296_296274

noncomputable def f (x : ℝ) := 2 * Real.sqrt(x) - 1 / x

theorem tangent_line_eq (x y : ℝ) (h₀ : x = 1) (h₁ : f x = y)
  (h₂ : Deriv f x = 2) : 2 * x - y - 1 = 0 := sorry

end tangent_line_eq_l296_296274


namespace find_a_parallel_find_a_perpendicular_l296_296140

variables {a : ℝ} {x y : ℝ}

def line_l1 (a : ℝ) : ℝ × ℝ × ℝ := (2 * a + 1, a + 2, 3)
def line_l2 (a : ℝ) : ℝ × ℝ × ℝ := (a - 1, -2, 2)

def slope (A B : ℝ) := -A / B

def are_parallel (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  slope l1.1 l1.2 = slope l2.1 l2.2

def are_perpendicular (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  slope l1.1 l1.2 * slope l2.1 l2.2 = -1

theorem find_a_parallel : are_parallel (line_l1 a) (line_l2 a) ↔ a = 0 := by
  sorry

theorem find_a_perpendicular : are_perpendicular (line_l1 a) (line_l2 a) ↔ a = -1 ∨ a = 5 / 2 := by
  sorry

end find_a_parallel_find_a_perpendicular_l296_296140


namespace quadrilateral_area_is_6_l296_296762

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨3, 1⟩
def D : Point := ⟨5, 5⟩

def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

def quadrilateral_area (A B C D : Point) : ℝ :=
  area_triangle A B C + area_triangle A C D

theorem quadrilateral_area_is_6 : quadrilateral_area A B C D = 6 :=
  sorry

end quadrilateral_area_is_6_l296_296762


namespace value_of_I_l296_296000

variables (T H I S : ℤ)

theorem value_of_I :
  H = 10 →
  T + H + I + S = 50 →
  H + I + T = 35 →
  S + I + T = 40 →
  I = 15 :=
  by
  sorry

end value_of_I_l296_296000


namespace boris_neighbors_l296_296786

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l296_296786


namespace intersection_is_three_l296_296931

open Set

theorem intersection_is_three :
  let A := {-1, 0, 1, 2, 3} 
  let B := {x : ℤ | x > 2}
  A ∩ B = {3} :=
by
  let A := {-1, 0, 1, 2, 3}
  let B := {x : ℤ | x > 2}
  -- Proof is omitted
  sorry

end intersection_is_three_l296_296931


namespace max_number_of_small_boxes_l296_296722

def volume_of_large_box (length width height : ℕ) : ℕ :=
  length * width * height

def volume_of_small_box (length width height : ℕ) : ℕ :=
  length * width * height

def number_of_small_boxes (large_volume small_volume : ℕ) : ℕ :=
  large_volume / small_volume

theorem max_number_of_small_boxes :
  let large_box_length := 4 * 100  -- in cm
  let large_box_width := 2 * 100  -- in cm
  let large_box_height := 4 * 100  -- in cm
  let small_box_length := 4  -- in cm
  let small_box_width := 2  -- in cm
  let small_box_height := 2  -- in cm
  let large_volume := volume_of_large_box large_box_length large_box_width large_box_height
  let small_volume := volume_of_small_box small_box_length small_box_width small_box_height
  number_of_small_boxes large_volume small_volume = 2000000 := by
  -- Prove the statement
  sorry

end max_number_of_small_boxes_l296_296722


namespace continuous_compound_interest_solution_l296_296317

noncomputable def continuous_compound_interest_rate 
  (A P: ℝ) (t: ℝ) (h_A_value: A = 760) (h_P_value: P = 600) (h_t_value: t = 4) : ℝ :=
  (Real.log (A / P)) / t

theorem continuous_compound_interest_solution :
  continuous_compound_interest_rate 760 600 4 (by norm_num) (by norm_num) (by norm_num) ≈ 0.05909725 :=
by
  unfold continuous_compound_interest_rate
  norm_num
  rw [← Real.log_div]
  sorry

end continuous_compound_interest_solution_l296_296317


namespace probability_equation_l296_296494

def Q (x : ℝ) : ℝ := x^2 - 5 * x - 15

def probability (lower upper : ℝ) :=
  ∀ x : ℝ, lower ≤ x ∧ x ≤ upper → 
  ∃ f g h i j : ℝ, 
    ( ∑ k in finset.Icc lower upper, (Q k) = ∑ n in finset.range (upper - lower + 1), n^2 ) ∧
    ( (f + g + h - i) / j = 11 / 10)
  
theorem probability_equation (x : ℝ) :
  (10 ≤ x ∧ x ≤ 20) →
  probability 10 20 :=
begin
  sorry
end

end probability_equation_l296_296494


namespace person_next_to_Boris_arkady_galya_l296_296801

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l296_296801


namespace complex_sum_argument_l296_296491

theorem complex_sum_argument :
  ∃ r : ℝ, θ ∈ [0, 2 * Real.pi) ∧ (θ = 11 * Real.pi / 20) ∧
            (r * Complex.exp (0 + θ * Complex.I) = Complex.exp (11 * Real.pi * Complex.I / 40) +
                                                  Complex.exp (21 * Real.pi * Complex.I / 40) +
                                                  Complex.exp (31 * Real.pi * Complex.I / 40) +
                                                  Complex.exp (41 * Real.pi * Complex.I / 40) +
                                                  Complex.exp (51 * Real.pi * Complex.I / 40)) := 
by 
  -- Proof would go here
  sorry

end complex_sum_argument_l296_296491


namespace num_valid_digits_l296_296630

/-- 
    Given a digit d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    the number of values of d such that 20.d05 > 20.05 is 9.
-/
theorem num_valid_digits (d : ℕ) (h : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ({d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} | 20 + d * 0.1 + 0.005 > 20.05}.to_finset.card = 9) :=
sorry

end num_valid_digits_l296_296630


namespace predicted_temperature_l296_296066

-- Define the observation data points
def data_points : List (ℕ × ℝ) :=
  [(20, 25), (30, 27.5), (40, 29), (50, 32.5), (60, 36)]

-- Define the linear regression equation with constant k
def regression (x : ℕ) (k : ℝ) : ℝ :=
  0.25 * x + k

-- Proof statement
theorem predicted_temperature (k : ℝ) (h : regression 40 k = 30) : regression 80 k = 40 :=
by
  sorry

end predicted_temperature_l296_296066


namespace isosceles_triangle_l296_296606

variables {A B C M N : ℝ}

-- Define the lengths of the sides of the triangle and segments
variables (a b c x y : ℝ)

-- Conditions based on the problem statement
axiom triangle_ABC (ABC_triangle: triangle):
  (point_on_side_A_B M)
  (point_on_side_B_C N)
  (perimeter_eq_1: x + distance M C + b = y + (a - y) + b)
  (perimeter_eq_2: x + y + c = (a - x) + (c - x) + a)

-- Prove that triangle ABC is isosceles
theorem isosceles_triangle (ABC_triangle): isosceles_triangle ABC_triangle := sorry

end isosceles_triangle_l296_296606


namespace minimum_value_expression_l296_296883

noncomputable def minimum_value (a b : ℝ) := (1 / (2 * |a|)) + (|a| / b)

theorem minimum_value_expression
  (a : ℝ) (b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  ∃ (min_val : ℝ), min_val = 3 / 4 ∧ ∀ (a b : ℝ), a + b = 2 → b > 0 → minimum_value a b ≥ min_val :=
sorry

end minimum_value_expression_l296_296883


namespace complement_M_l296_296498

open Set

-- Definitions and conditions
def U : Set ℝ := univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Theorem stating the complement of M with respect to the universal set U
theorem complement_M : compl M = {x | x < -2 ∨ x > 2} :=
by
  sorry

end complement_M_l296_296498


namespace perfect_square_trinomial_m6_l296_296524

theorem perfect_square_trinomial_m6 (m : ℚ) (h₁ : 0 < m) (h₂ : ∃ a : ℚ, x^2 - 2 * m * x + 36 = (x - a)^2) : m = 6 :=
sorry

end perfect_square_trinomial_m6_l296_296524


namespace complementary_event_l296_296017

open Classical

-- Definition of events
def missing_both_shots := ¬ (hit first ∨ hit second)
def at_least_one_hit := hit first ∨ hit second

-- Theorem to prove 
theorem complementary_event :
  (missing_both_shots ↔ ¬ at_least_one_hit) :=
begin
  sorry
end

end complementary_event_l296_296017


namespace floor_neg_seven_thirds_l296_296078

theorem floor_neg_seven_thirds : ⌊-7 / 3⌋ = -3 :=
sorry

end floor_neg_seven_thirds_l296_296078


namespace angle_2016_216_in_same_quadrant_l296_296378

noncomputable def angle_in_same_quadrant (a b : ℝ) : Prop :=
  let normalized (x : ℝ) := x % 360
  normalized a = normalized b

theorem angle_2016_216_in_same_quadrant : angle_in_same_quadrant 2016 216 := by
  sorry

end angle_2016_216_in_same_quadrant_l296_296378


namespace correct_statement_D_l296_296714

theorem correct_statement_D : (- 3 / 5 : ℚ) < (- 4 / 7 : ℚ) :=
  by
  -- The proof step is omitted as per the instruction
  sorry

end correct_statement_D_l296_296714


namespace bob_initial_pennies_l296_296971

-- Definitions of conditions
variables (a b : ℕ)
def condition1 : Prop := b + 2 = 4 * (a - 2)
def condition2 : Prop := b - 2 = 3 * (a + 2)

-- Goal: Proving that b = 62
theorem bob_initial_pennies (h1 : condition1 a b) (h2 : condition2 a b) : b = 62 :=
by {
  sorry
}

end bob_initial_pennies_l296_296971


namespace probability_four_odd_rolls_l296_296677

theorem probability_four_odd_rolls :
  let p := 1 / 2 : ℚ,
      n := 5,
      k := 4,
      total_outcomes := (2 : ℕ) ^ n,
      favorable_outcomes := Nat.choose n k
  in  (favorable_outcomes : ℚ) / total_outcomes = 5 / 32 :=
by sorry

end probability_four_odd_rolls_l296_296677


namespace slower_train_speed_is_36_l296_296675

def speed_of_slower_train (v : ℕ) : Prop :=
  let length_of_each_train := 100
  let distance_covered := length_of_each_train * 2
  let time_taken := 72
  let faster_train_speed := 46
  let relative_speed := (faster_train_speed - v) * (1000 / 3600)
  distance_covered = relative_speed * time_taken

theorem slower_train_speed_is_36 : ∃ v, speed_of_slower_train v ∧ v = 36 :=
by
  use 36
  unfold speed_of_slower_train
  -- Prove that the equation holds when v = 36
  sorry

end slower_train_speed_is_36_l296_296675


namespace remainder_19008_div_31_l296_296152

theorem remainder_19008_div_31 :
  ∀ (n : ℕ), (n = 432 * 44) → n % 31 = 5 :=
by
  intro n h
  sorry

end remainder_19008_div_31_l296_296152


namespace new_trash_cans_in_veterans_park_l296_296054

def initial_trash_cans_in_veterans_park : ℕ := 24

def trash_cans_in_central_park (V : ℕ) : ℕ := (V / 2) + 8

def moved_trash_cans (C : ℕ) : ℕ := C / 2

theorem new_trash_cans_in_veterans_park :
  let V := initial_trash_cans_in_veterans_park in
  let C := trash_cans_in_central_park V in
  let moved_cans := moved_trash_cans C in
  V + moved_cans = 34 :=
by
  let V := initial_trash_cans_in_veterans_park
  let C := trash_cans_in_central_park V
  let moved_cans := moved_trash_cans C
  have h1 : V = 24 := rfl
  have h2 : C = 20 := by
    calc
      C = (V / 2) + 8 : rfl
      _ = (24 / 2) + 8 : by rw [h1]
      _ = 12 + 8 : by norm_num
      _ = 20 : rfl
  have h3 : moved_cans = 10 := by
    calc
      moved_cans = C / 2 : rfl
      _ = 20 / 2 : by rw [h2]
      _ = 10 : by norm_num
  show V + moved_cans = 34 from
    calc
      V + moved_cans = 24 + 10 : by rw [h1, h3]
      _ = 34 : by norm_num

end new_trash_cans_in_veterans_park_l296_296054


namespace integer_solutions_count_l296_296961

theorem integer_solutions_count :
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 :=
by sorry

end integer_solutions_count_l296_296961


namespace math_proof_l296_296125

-- Let f be an odd function on ℝ
axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x

-- Given f(1) = 1
axiom f_one (f : ℝ → ℝ) : f 1 = 1

-- Given f(1/(x-1)) = x f(x) for any x < 0
axiom functional_eq (f : ℝ → ℝ) : ∀ x, x < 0 → f (x / (x - 1)) = x * f x

-- Prove the theorem
theorem math_proof :
  ∃ (f : ℝ → ℝ),
  odd_function f ∧ f_one f ∧ 
  (∀ x, x < 0 → functional_eq f x) ∧ 
  f 1 * f (1 / 100) + f (1 / 2) * f (1 / 99) +
  f (1 / 3) * f (1 / 98) + ⋯ + f (1 / 50) * f (1 / 51) = 2^98 / 99! :=
by
  sorry

end math_proof_l296_296125


namespace area_triangle_ABC_l296_296725

variables (A B C E F D : Type) [tri : has_triangle A B C] [rho : has_rhombus B D E F]
  [vertices_on_sides A B C E F D] 
  (AE: ℝ) (CE: ℝ) (in_circle_radius: ℝ)
  (angle_at_E_obtuse : is_obtuse (angle B E F))

-- Given values
variables (hAE : AE = 3) (hCE : CE = 7) (hradius : in_circle_radius = 1)

-- Prove the area of the triangle ABC
theorem area_triangle_ABC : area (triangle A B C) = 5 * real.sqrt 5 :=
sorry

end area_triangle_ABC_l296_296725


namespace some_number_value_l296_296759

theorem some_number_value (some_number : ℝ): 
  (∀ n : ℝ, (n / some_number) * (n / 80) = 1 → n = 40) → some_number = 80 :=
by
  sorry

end some_number_value_l296_296759


namespace min_balls_to_ensure_20_of_one_color_l296_296530

theorem min_balls_to_ensure_20_of_one_color :
  ∀ (red green yellow blue white black : ℕ),
  red = 34 → green = 25 → yellow = 23 → blue = 18 → white = 14 → black = 10 →
  ( ∀ n, n < 100 →
    ( (modNat numBalls red 20) < 20 ∧
      (modNat numBalls green 20) < 20 ∧
      (modNat numBalls yellow 20) < 20 ∧
      (modNat numBalls blue 20) < 20 ∧
      (modNat numBalls white 20) < 20 ∧
      (modNat numBalls black 20) < 20 )) →
  (∀ n, n ≥ 100 →
    ( (modNat numBalls red 20) = 20 ∨
      (modNat numBalls green 20) = 20 ∨
      (modNat numBalls yellow 20) = 20 ∨
      (modNat numBalls blue 20) = 20 ∨
      (modNat numBalls white 20) = 20 ∨
      (modNat numBalls black 20) = 20 )) :=
sorry -- proof goes here

end min_balls_to_ensure_20_of_one_color_l296_296530


namespace imaginary_part_of_z_l296_296910

variable (z : ℂ)
variable (i : ℂ) [complex.imaging_unit : i = complex.I]

def z_def : z = (1 - i) / (2 * i) := sorry

theorem imaginary_part_of_z : complex.im z = -1/2 :=
by
  rw [z_def]
  sorry

end imaginary_part_of_z_l296_296910


namespace integer_count_in_range_l296_296958

theorem integer_count_in_range (x : Int) : 
  (Set.count (Set.range (λ x, ( -6 ≤ 3*x + 2 ∧ 3*x + 2 ≤ 9))) 5) := 
by 
  sorry

end integer_count_in_range_l296_296958


namespace lattice_point_sum_l296_296993

-- Definitions based on conditions
def is_lattice_point (p : ℕ × ℕ) : Prop :=
  ∃ n : ℕ, p = (n, n + 3)

-- Defining function f based on the gcd analysis
def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

-- Specifying the main theorem/problem to prove
theorem lattice_point_sum : 
  ∑ n in Finset.range 1990, f (n + 1) = 1326 :=
sorry

end lattice_point_sum_l296_296993


namespace robert_bike_time_l296_296009

noncomputable def semicircle_travel_time (width length travel_distance speed : ℝ) (radius_factor : ℝ) : ℝ :=
  let large_radius := width / 2
  let small_radius := large_radius / radius_factor
  let num_large_semicircles := travel_distance / (2 * large_radius)
  let large_travel := num_large_semicircles * (large_radius * π)
  let remaining_distance := length * 5280 - travel_distance
  let num_small_semicircles := remaining_distance / (2 * small_radius)
  let small_travel := num_small_semicircles * (small_radius * π)
  let total_travel := large_travel + small_travel
  total_travel / (5280 * speed)

theorem robert_bike_time : semicircle_travel_time 40 1 528 5 2 = π / 5 := 
  sorry

end robert_bike_time_l296_296009


namespace probability_gary_paula_l296_296881

theorem probability_gary_paula :
  let G := 4
  let P := 5
  let total := G + P
  ∃ (p : ℚ), 
  (∃ (draw1 draw2 : {x // x ∈ finset.range total} → finset ({x // x ∈ finset.range total})), 
   draw1 ≠ draw2 ∧
   ((draw1 ≤ G ∧ draw2 > G) ∨ (draw2 ≤ G ∧ draw1 > G)) ∧
   p = (G / total) * (P / (total - 1)) + (P / total) * (G / (total - 1))) 
   ∧ p = 5 / 9 := by
  let G := 4
  let P := 5
  let total := G + P
  let prob1 := (↑G / ↑total) * (↑P / (↑total - 1))
  let prob2 := (↑P / ↑total) * (↑G / (↑total - 1))
  have h : prob1 = 5 / 18 := by
    sorry
  have h1 : prob2 = 5 / 18 := by
    sorry
  use (prob1 + prob2)
  split
  sorry
  rw [h, h1]
  norm_num
  sorry

end probability_gary_paula_l296_296881


namespace meaningful_expression_condition_l296_296326

theorem meaningful_expression_condition (x : ℝ) : (x > 1) ↔ (∃ y : ℝ, y = 2 / Real.sqrt (x - 1)) :=
by
  sorry

end meaningful_expression_condition_l296_296326


namespace factorize_polynomial_l296_296336

theorem factorize_polynomial :
  (∀ x : ℤ, x^{15} + x^{10} + 1 = (x^2 + x + 1) * (x^{13} - x^{12} + x^{10} - x^9 + x^7 - x^6 + x^4 - x^3 + x) + 1) :=
by sorry

end factorize_polynomial_l296_296336


namespace area_pentagon_ABCDE_l296_296992

noncomputable def point := ℝ × ℝ

def A : point := (1, 9)
def C : point := (5, 8)
def D : point := (8, 2)
def E : point := (2, 2)

def line (p1 p2 : point) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let slope := (y2 - y1) / (x2 - x1)
  let intercept := y1 - slope * x1
  (slope, intercept)
  
def line_EC := line E C
def line_AD := line A D

def B : point :=
  let (m1, c1) := line_EC
  let (m2, c2) := line_AD
  let x := (c2 - c1) / (m1 - m2)
  let y := m1 * x + c1
  (x, y)

def polygon_area (points : list point) : ℝ :=
  0.5 * abs (list.foldl (λ acc ⟨xn, yn⟩ idx,
    let ⟨xn1, yn1⟩ := if idx == points.length - 1 then points.head else points[idx + 1]
    acc + xn * yn1 - yn * xn1
  ) 0 points)

def pentagon_ABCDE : list point := [A, B, C, D, E]

theorem area_pentagon_ABCDE : polygon_area pentagon_ABCDE = 27 :=
sorry

end area_pentagon_ABCDE_l296_296992


namespace standing_next_to_boris_l296_296829

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l296_296829


namespace sum_roots_eq_one_l296_296091

theorem sum_roots_eq_one :
  (let p1 := (3 : ℝ) * X^3 - 9 * X^2 - 6 * X + 54,
       p2 := (4 : ℝ) * X^3 + 8 * X^2 - 16 * X + 32 in
    (sum_roots p1) + (sum_roots p2) = 1) :=
sorry

end sum_roots_eq_one_l296_296091


namespace petya_cannot_prevent_vasya_l296_296591

noncomputable def petya_vasya_game : Prop :=
  ∀ (paint : ℕ → option bool) (next_move : ℕ → ℕ),
    (∀ n : ℕ, n < 99 → paint n = none) →
    (∀ n : ℕ, paint (next_move n) = some true ∨ paint (next_move n) = some false) →
    (∀ n : ℕ, n < 99 → ∃ k, k < 33 ∧ paint (k + next_move n) = paint (k + next_move n + 33) ∧ paint (k + next_move n + 66) = paint (k + next_move n + 33)) →
    true

theorem petya_cannot_prevent_vasya :
  petya_vasya_game :=
by
  sorry

end petya_cannot_prevent_vasya_l296_296591


namespace trajectory_of_point_M_l296_296439

theorem trajectory_of_point_M :
  let F1 := (ℝ × ℝ) (-3, 0)
  let F2 := (ℝ × ℝ) (3, 0)
  ∃ (M : ℝ × ℝ) (d : ℝ), |dist M F1 - dist M F2| = 6 ↔ M ∈ {f : ℝ × ℝ | true} :=
sorry

end trajectory_of_point_M_l296_296439


namespace measure_xi_le_eta_l296_296568

variable {Ω : Type} [MeasurableSpace Ω] (μ : Measure Ω)
variable (ξ η : Ω → ℝ)  [Integrable ξ μ] [Integrable η μ]
variable (𝒜 : Set (Set Ω)) [IsMeasurable 𝒜]

theorem measure_xi_le_eta {A : Set Ω} (hA : A ∈ 𝒜) : IntegralOn ξ A μ ≤ IntegralOn η A μ → Measure.mkOf (ξ > η) = 0 := by
  intros h
  sorry

end measure_xi_le_eta_l296_296568


namespace large_pots_delivered_l296_296344

theorem large_pots_delivered :
  ∀ (n : ℕ), 
    let total_small_pots := 36 * 32 in
    let total_pots_in_8_boxes := (36 + n) * 8 in
    (32 * n = total_pots_in_8_boxes) → (32 * n = 384) :=
by
  intros n total_small_pots total_pots_in_8_boxes h
  sorry

end large_pots_delivered_l296_296344


namespace local_minimum_at_1_has_two_extreme_points_local_minimum_value_maximum_value_on_interval_l296_296927

noncomputable def f (x : ℝ) : ℝ := x^3 + (1/2) * x^2 - 4 * x

theorem local_minimum_at_1 : ∃ x, f x = 1 ∧ (∀ y, f y > f 1 ∧ y ≠ 1) := sorry

theorem has_two_extreme_points : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
  (∀ y, f y ≥ f x1 ∨ f y ≥ f x2 ∨ f y ≤ f x1 ∨ f y ≤ f x2) := sorry

theorem local_minimum_value : f 1 = -(5/2) := sorry

theorem maximum_value_on_interval : ∃ x ∈ set.Icc 0 2, ∀ y ∈ set.Icc 0 2, f y ≤ f x ∧ f x = 2 := sorry

end local_minimum_at_1_has_two_extreme_points_local_minimum_value_maximum_value_on_interval_l296_296927


namespace graph_depicts_one_line_l296_296037

theorem graph_depicts_one_line {x y : ℝ} :
  (x - 1) ^ 2 * (x + y - 2) = (y - 1) ^ 2 * (x + y - 2) →
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b :=
by
  intros h
  sorry

end graph_depicts_one_line_l296_296037


namespace one_python_can_eat_per_week_l296_296740

-- Definitions based on the given conditions
def burmese_pythons := 5
def alligators_eaten := 15
def weeks := 3

-- Theorem statement to prove the number of alligators one python can eat per week
theorem one_python_can_eat_per_week : (alligators_eaten / burmese_pythons) / weeks = 1 := 
by 
-- sorry is used to skip the actual proof
sorry

end one_python_can_eat_per_week_l296_296740


namespace multiple_of_old_edition_l296_296014

theorem multiple_of_old_edition 
  (new_pages: ℕ) 
  (old_pages: ℕ) 
  (difference: ℕ) 
  (m: ℕ) 
  (h1: new_pages = 450) 
  (h2: old_pages = 340) 
  (h3: 450 = 340 * m - 230) : 
  m = 2 :=
sorry

end multiple_of_old_edition_l296_296014


namespace product_remainder_l296_296673

theorem product_remainder (a b : ℕ) (m n : ℤ) (ha : a = 3 * m + 2) (hb : b = 3 * n + 2) : 
  (a * b) % 3 = 1 := 
by 
  sorry

end product_remainder_l296_296673


namespace heart_ratio_correct_l296_296970

def heart (n m : ℕ) : ℕ := n^3 + m^2

theorem heart_ratio_correct : (heart 3 5 : ℚ) / (heart 5 3) = 26 / 67 :=
by
  sorry

end heart_ratio_correct_l296_296970


namespace girth_log_inequality_l296_296975

theorem girth_log_inequality (G : Type) [Graph G] 
  (min_degree : ∀ v : G, degree v ≥ 3) : girth G < 2 * log (card G) := 
sorry

end girth_log_inequality_l296_296975


namespace function_property_l296_296127

noncomputable def f : ℝ → ℝ := sorry

theorem function_property (h : ∀ x y : ℝ, f(x + y) + f(x - y) = 2 * f(x) * f(y)) :
  (f(1) = 1 → f(2) = 1) ∧
  (∀ x : ℝ, f(-x) = f(x)) ∧ 
  (f(1) = 0 → ∑ i in finset.range 2024, f (i + 1) = 0) :=
by
  sorry

end function_property_l296_296127


namespace factorization_correct_l296_296342

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by 
  sorry

end factorization_correct_l296_296342


namespace octal_to_decimal_l296_296004

theorem octal_to_decimal :
  let num_octal := 253;
  3 * 8^0 + 5 * 8^1 + 2 * 8^2 = 171 :=
begin
  sorry -- This placeholder means the proof is not provided.
end

end octal_to_decimal_l296_296004


namespace max_projection_length_l296_296888

theorem max_projection_length (A : ℝ × ℝ) (P : ℝ × ℝ) (λ : ℝ) 
  (h1 : A.1^2 / 25 + A.2^2 / 9 = 1)
  (h2 : P = (λ - 1) • A)
  (h3 : A.1 * P.1 + A.2 * P.2 = 72) : 
  ∃ x : ℝ, abs x = 15 :=
sorry

end max_projection_length_l296_296888


namespace linear_function_expression_and_value_l296_296011

-- Define the conditions
def linear_function (x : ℝ) (k b : ℝ) : ℝ := k * x + b

-- Given points
def point1 := (3 : ℝ, 1 : ℝ)
def point2 := (2 : ℝ, 0 : ℝ)

-- The corresponding Lean statement
theorem linear_function_expression_and_value
  (k b : ℝ)
  (h1 : linear_function 3 k b = 1)
  (h2 : linear_function 2 k b = 0) :
  (k = 1 ∧ b = -2 ∧ linear_function 6 1 (-2) = 4) := by
  -- Proof will be filled here
  sorry

end linear_function_expression_and_value_l296_296011


namespace diameter_of_well_l296_296743

theorem diameter_of_well (h : 14) 
  (V : 43.982297150257104) : 
  diameter = 2 :=
by
  sorry

end diameter_of_well_l296_296743


namespace boris_neighbors_l296_296792

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l296_296792


namespace triangle_isosceles_l296_296609

theorem triangle_isosceles
  {A B C M N : Point}
  (h_M_on_AB : ∃ t ∈ Set.Icc (0 : ℝ) 1, M = t • A + (1 - t) • B)
  (h_N_on_BC : ∃ t ∈ Set.Icc (0 : ℝ) 1, N = t • B + (1 - t) • C)
  (h_perimeter_AMC_CNA : dist A M + dist M C + dist C A = dist C N + dist N A + dist A C)
  (h_perimeter_ANB_CMB : dist A N + dist N B + dist B A = dist C M + dist M B + dist B C)
  : isosceles_triangle A B C := 
sorry

end triangle_isosceles_l296_296609


namespace evaluate_expression_l296_296418

theorem evaluate_expression (a b c : ℕ) (h1 : a = 47) (h2 : b = 28) (h3 : c = 100) :
  (a^2 - b^2) + c = 1525 :=
by
  rw [h1, h2, h3]
  calc (47^2 - 28^2) + 100 = (47 + 28) * (47 - 28) + 100 : by rw [sq_sub_sq]
  ... = 75 * 19 + 100 : by norm_num
  ... = 1525 : by norm_num

end evaluate_expression_l296_296418


namespace avg_people_per_hour_rounding_l296_296176

theorem avg_people_per_hour_rounding :
  let people := 3500
  let days := 5
  let hours := days * 24
  (people / hours : ℚ).round = 29 := 
by
  sorry

end avg_people_per_hour_rounding_l296_296176


namespace length_of_BE_l296_296355

theorem length_of_BE (ABC : Triangle) (r : ℝ) (BC : Segment) (D E : Point) (radius_2 : r = 2) : 
  is_equilateral ABC ∧ is_inscribed_circle ABC r ∧ intersects_altitude_circle D E ∧ 
  altitude_intersects_circle_not_on_base D BC ∧ right_intersects_circle_distinct D E →
  length_of_segment (segment BE) = 2 * sqrt 3 :=
by 
  sorry

end length_of_BE_l296_296355


namespace largest_k_statement_l296_296986

noncomputable def largest_k (n : ℕ) : ℕ :=
  n - 2

theorem largest_k_statement (S : Finset ℕ) (A : Finset (Finset ℕ)) (h1 : ∀ (A_i : Finset ℕ), A_i ∈ A → 2 ≤ A_i.card ∧ A_i.card < S.card) : 
  largest_k S.card = S.card - 2 :=
by
  sorry

end largest_k_statement_l296_296986


namespace proof_problem_l296_296892

def arithmetic_seq (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ∃ (a₁ : ℤ), d ≠ 0 ∧ a₁ = 25 ∧ (∀ n, a_n n = a₁ + d * (n - 1))

def geometric_seq (aₙ : ℕ → ℤ) : Prop :=
  aₙ 1 * aₙ 13 = (aₙ 11) ^ 2

def general_term_formula (a_n : ℕ → ℤ) : Prop :=
  ∀ n, a_n n = -2 * n + 27

def sum_arithmetic_sequence (a_n : ℕ → ℤ) (n : ℕ) : ℤ :=
  let S_n := (1 to n).toList.map (fun k => a_n (3 * k - 2)) in S_n.sum

def sum_arithmetic_seq_eq (a_n : ℕ → ℤ) : Prop :=
  ∀ n, sum_arithmetic_sequence a_n n = -3 * n ^ 2 + 28 * n

theorem proof_problem (a_n : ℕ → ℤ) (d : ℤ) :
  (arithmetic_seq a_n d) ∧ (geometric_seq a_n) →
  (general_term_formula a_n) ∧ (sum_arithmetic_seq_eq a_n) := by
  sorry

end proof_problem_l296_296892


namespace eq_exponents_problem_1_problem_2_problem_3_l296_296510

-- Definition: if a^m = a^n for a > 0 and a ≠ 1, then m = n.
theorem eq_exponents {a : ℝ} {m n : ℝ} (h : a > 0) (h1 : a ≠ 1) (h2 : a^m = a^n) : m = n :=
begin
  sorry
end

-- Problem 1: Given 2 * 8^x * 16^x = 2^22, prove x = 3.
theorem problem_1 (x : ℝ) (h : 2 * 8^x * 16^x = 2^22) : x = 3 :=
begin
  sorry
end

-- Problem 2: Given (27^{-x})^2 = (1/9)^{-6}, prove x = -2.
theorem problem_2 (x : ℝ) (h : (27^(-x))^2 = (1 / 9)^(-6)) : x = -2 :=
begin
  sorry
end

-- Problem 3: Given p = 5^7 and q = 7^5, express 35^35 as p^5 q^7.
theorem problem_3 (p q : ℝ) (h1 : p = 5^7) (h2 : q = 7^5) : 35^35 = p^5 * q^7 :=
begin
  sorry
end

end eq_exponents_problem_1_problem_2_problem_3_l296_296510


namespace volume_at_20_deg_l296_296093

theorem volume_at_20_deg
  (ΔV_per_ΔT : ∀ ΔT : ℕ, ΔT = 5 → ∀ V : ℕ, V = 5)
  (initial_condition : ∀ V : ℕ, V = 40 ∧ ∀ T : ℕ, T = 40) :
  ∃ V : ℕ, V = 20 :=
by
  sorry

end volume_at_20_deg_l296_296093


namespace frog_escape_probability_l296_296532

def P : ℕ → ℚ
noncomputable def P 0 := 0
noncomputable def P 12 := 1
noncomputable def P (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 12 then 1
  else if 0 < n ∧ n < 12 then 
    (n.to_nat / 12) * P (n - 1) + (1 - (n.to_nat / 12)) * P (n + 1)
  else 0

theorem frog_escape_probability : P 2 = 109 / 221 := sorry

end frog_escape_probability_l296_296532


namespace min_value_of_theta_l296_296669

theorem min_value_of_theta (θ : ℝ) (hθ : θ > 0) :
  ∀ f g : ℝ → ℝ,
  (∀ x, f x = sin x ^ 2 - cos x ^ 2) →
  (∀ x, g x = f (x - θ)) →
  (∀ x, g (-x) = -g x) →
  θ = π :=
sorry

end min_value_of_theta_l296_296669


namespace find_abs_ab_l296_296649

def ellipse_foci_distance := 5
def hyperbola_foci_distance := 7

def ellipse_condition (a b : ℝ) := b^2 - a^2 = ellipse_foci_distance^2
def hyperbola_condition (a b : ℝ) := a^2 + b^2 = hyperbola_foci_distance^2

theorem find_abs_ab (a b : ℝ) (h_ellipse : ellipse_condition a b) (h_hyperbola : hyperbola_condition a b) :
  |a * b| = 2 * Real.sqrt 111 :=
by
  sorry

end find_abs_ab_l296_296649


namespace largest_number_divisible_by_12_l296_296239

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

def max_number_formed_from_digits (digits : list ℕ) : ℕ :=
  -- Implementation that assumes the correct function to form the number
  -- This will be written correctly in formal proof, but for now, we use
  -- the known outcome.
  8654232

theorem largest_number_divisible_by_12 :
  ∃ (n : ℕ), (∀ (digits : list ℕ), digits = [2, 2, 3, 4, 5, 6, 8] → 
    max_number_formed_from_digits digits = n ∧ is_divisible_by_12 n) ∧ 
    n = 8654232 := 
by {
  existsi (8654232 : ℕ),
  split,
  { intros digits h,
    split,
    { rw h, 
      refl, },
    { dsimp [is_divisible_by_12], norm_num, }, },
  refl,
}

end largest_number_divisible_by_12_l296_296239


namespace EF_value_l296_296995

theorem EF_value
  (AB EF CD BC : ℝ) 
  (h1 : AB = 20)
  (h2 : CD = 80)
  (h3 : BC = 100)
  (h4 : AB ∥ EF ∥ CD) :
  EF = 16 := 
-- Proof omitted
sorry

end EF_value_l296_296995


namespace sequence_odd_numbers_l296_296289

theorem sequence_odd_numbers (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 7)
  (h3 : ∀ n ≥ 2, -1/2 < a (n + 1) - (a n)^2 / a (n - 1) ∧ a (n + 1) - (a n)^2 / a (n - 1) ≤ 1/2) :
  ∀ n, n > 1 → ∃ k : ℕ, odd k ∧ a n = k := 
sorry

end sequence_odd_numbers_l296_296289


namespace range_of_a_l296_296923

theorem range_of_a (a : ℝ) (h_a: a > 0) :
  (∃ x1 x2 ∈ set.Icc (0 : ℝ) 1, (2 * x1^2 / (x1 + 1)) = (a * real.sin ((π / 6) * x2) - 2 * a + 2)) ↔ (1 / 2 ≤ a ∧ a ≤ 4 / 3) :=
by sorry

end range_of_a_l296_296923


namespace number_of_students_at_end_of_year_l296_296538

def students_at_start_of_year : ℕ := 35
def students_left_during_year : ℕ := 10
def students_joined_during_year : ℕ := 10

theorem number_of_students_at_end_of_year : students_at_start_of_year - students_left_during_year + students_joined_during_year = 35 :=
by
  sorry -- Proof goes here

end number_of_students_at_end_of_year_l296_296538


namespace factor_correct_l296_296421

noncomputable def factor_expr (x : ℝ) : ℝ :=
  75 * x^3 - 225 * x^10
  
noncomputable def factored_form (x : ℝ) : ℝ :=
  75 * x^3 * (1 - 3 * x^7)

theorem factor_correct (x : ℝ): 
  factor_expr x = factored_form x :=
by
  -- Proof omitted
  sorry

end factor_correct_l296_296421


namespace standing_next_to_boris_l296_296830

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l296_296830


namespace nancy_paid_more_l296_296391

def total_cost := 15
def cost_per_slice := total_cost / 12
def extra_cheese_slices := 4
def plain_slices := 8
def total_slices := extra_cheese_slices + plain_slices

def nancy_slices := 7 -- All extra cheese slices + 3 plain slices
def carol_slices := 5 -- Remaining plain slices

def nancy_payment := extra_cheese_slices * cost_per_slice + 3 * cost_per_slice
def carol_payment := carol_slices * cost_per_slice

def payment_difference := nancy_payment - carol_payment
theorem nancy_paid_more : payment_difference = 2.5 := sorry

end nancy_paid_more_l296_296391


namespace tetrahedron_dihedral_face_areas_l296_296215

variables {S₁ S₂ a b : ℝ} {α φ : ℝ}

theorem tetrahedron_dihedral_face_areas :
  S₁^2 + S₂^2 - 2 * S₁ * S₂ * Real.cos α = (a * b * Real.sin φ / 4)^2 :=
sorry

end tetrahedron_dihedral_face_areas_l296_296215


namespace no_factors_l296_296073

-- Define the polynomials.
noncomputable def P : ℚ[X] := X^4 - 4*X^2 + 16
noncomputable def D1 : ℚ[X] := X^2 + 4
noncomputable def D2 : ℚ[X] := X^2 - 1
noncomputable def D3 : ℚ[X] := X^2 + 1
noncomputable def D4 : ℚ[X] := X^2 + 3*X + 2

-- State the condition that P is not divisible by any of the given divisors.
theorem no_factors : ¬ (D1 ∣ P) ∧ ¬ (D2 ∣ P) ∧ ¬ (D3 ∣ P) ∧ ¬ (D4 ∣ P) := by
  sorry

end no_factors_l296_296073


namespace solve_BSNK_l296_296566

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem solve_BSNK (B S N K : ℝ) (hB : 0 < B) (hS : 0 < S) (hN : 0 < N) (hK : 0 < K)
    (h1 : log_base_10 (B * K) + log_base_10 (B * N) = 3)
    (h2 : log_base_10 (N * K) + log_base_10 (N * S) = 4)
    (h3 : log_base_10 (S * B) + log_base_10 (S * K) = 5) : B * S * N * K = 10000 :=
begin
    sorry
end

end solve_BSNK_l296_296566


namespace probability_of_rolling_3_or_5_is_1_over_4_l296_296708

def fair_8_sided_die := {outcome : Fin 8 // true}

theorem probability_of_rolling_3_or_5_is_1_over_4 :
  (1 / 4 : ℚ) = 2 / 8 :=
by sorry

end probability_of_rolling_3_or_5_is_1_over_4_l296_296708


namespace train_length_correct_l296_296372

noncomputable def length_of_train
  (train_speed_kmhr : ℝ)
  (man_speed_kmhr : ℝ)
  (time_sec : ℝ) : ℝ :=
  let relative_speed_kmhr := train_speed_kmhr + man_speed_kmhr in
  let relative_speed_ms := relative_speed_kmhr * (5 / 18) in
  relative_speed_ms * time_sec

theorem train_length_correct :
  length_of_train 24 6 13.198944084473244 = 110.32453403727703 :=
by
  simp [length_of_train]
  sorry

end train_length_correct_l296_296372


namespace convert_rectangular_to_polar_l296_296849

-- Define the conversion from rectangular to polar coordinates
def toPolar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2)
  let θ := if x < 0 then real.pi - real.arctan (y / abs x) else real.arctan (y / x)
  (r, θ)

theorem convert_rectangular_to_polar :
  toPolar (-2) 1 = (real.sqrt 5, real.pi - real.arctan (1 / 2)) := 
by
  sorry

end convert_rectangular_to_polar_l296_296849


namespace median_in_interval_70_79_l296_296401

theorem median_in_interval_70_79 :
  let scores := [18, 20, 19, 17, 26] -- representing the intervals [90-100, 80-89, 70-79, 60-69, 50-59]
  ∑ i in (finset.range 5), scores.nth i = 100 -- 100 students
  → let median_interval := if scores.get 2 + 38 ≥ 50 then 70 else 0
  median_interval = 70 := by
  sorry

end median_in_interval_70_79_l296_296401


namespace statement_3_statement_4_l296_296460

variable {a b : ℝ}

-- Statement 3: If a and b are both positive and a^2 - b^2 = 1, then a - b <= 1
theorem statement_3 (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a^2 - b^2 = 1) : a - b ≤ 1 := 
  sorry

-- Statement 4: If a and b are both positive and sqrt(a) - sqrt(b) = 1, then a - b >= 1
theorem statement_4 (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : sqrt(a) - sqrt(b) = 1) : 1 ≤ a - b :=
  sorry

end statement_3_statement_4_l296_296460


namespace math_problem_l296_296683

theorem math_problem : 2 - (-3)^2 - 4 - (-5) - 6^2 - (-7) = -35 := 
by
  sorry

end math_problem_l296_296683


namespace find_courtyard_width_l296_296748

def brick_length_cm := 20
def brick_width_cm := 10
def bricks_required := 20000
def courtyard_length_m := 25

def brick_length_m : ℝ := brick_length_cm / 100.0
def brick_width_m : ℝ := brick_width_cm / 100.0
def area_per_brick_m2 : ℝ := brick_length_m * brick_width_m
def total_area_m2 : ℝ := bricks_required * area_per_brick_m2

theorem find_courtyard_width (length_m : ℝ) (area_m2 : ℝ) : 
  ∃ (W : ℝ), W = 16 :=
by
  have courtyard_width := area_m2 / length_m
  use courtyard_width
  have : courtyard_width = 16 := 
    calc
      courtyard_width = 400 / 25 : by sorry
      ... = 16 : by sorry
  exact this

end find_courtyard_width_l296_296748


namespace z_is_233_percent_greater_than_w_l296_296525

theorem z_is_233_percent_greater_than_w
  (w e x y z : ℝ)
  (h1 : w = 0.5 * e)
  (h2 : e = 0.4 * x)
  (h3 : x = 0.3 * y)
  (h4 : z = 0.2 * y) :
  z = 2.3333 * w :=
by
  sorry

end z_is_233_percent_greater_than_w_l296_296525


namespace customer_paid_approx_rs_57_33_l296_296281

def list_price : ℝ := 65.0
def first_discount : ℝ := 0.10
def second_discount : ℝ := 0.020000000000000027

def final_price (P : ℝ) (D₁ : ℝ) (D₂ : ℝ) : ℝ :=
  P * (1 - D₁) * (1 - D₂)

theorem customer_paid_approx_rs_57_33 :
  final_price list_price first_discount second_discount ≈ 57.33 := by
  sorry

end customer_paid_approx_rs_57_33_l296_296281


namespace other_x_intercept_l296_296453

theorem other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, y = a * x ^ 2 + b * x + c → (x, y) = (4, -3)) (h_x_intercept : ∀ y, y = a * 1 ^ 2 + b * 1 + c → (1, y) = (1, 0)) : 
  ∃ x, x = 7 := by
sorry

end other_x_intercept_l296_296453


namespace rem_value_l296_296094

def rem (x y : ℝ) : ℝ :=
  x - y * Real.floor (x / y)

theorem rem_value :
  rem (2 / 7) (-3 / 4) = 29 / 28 := by
  sorry

end rem_value_l296_296094


namespace num_integers_satisfying_ineq_count_l296_296940

theorem num_integers_satisfying_ineq_count :
  {x : ℤ | -6 ≤ 3 * (x : ℤ) + 2 ∧ 3 * (x : ℤ) + 2 ≤ 9}.finite.to_finset.card = 5 :=
by
  sorry

end num_integers_satisfying_ineq_count_l296_296940


namespace original_cost_price_l296_296013

theorem original_cost_price (C : ℝ) : 
  (0.89 * C * 1.20 = 54000) → C = 50561.80 :=
by
  sorry

end original_cost_price_l296_296013


namespace part_I_part_II_part_III_l296_296135

-- Define the function f(x) = e^x - a * x - 1
def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x - 1

-- Given the conditions, we prove the related properties.

-- Part I: Find the value of a and the intervals where f(x) is monotonic
theorem part_I (x : ℝ) (a : ℝ) (h : f x a = exp x - a * x - 1) : 
  (a = 2) ∧ (∀ x, x < real.log 2 -> f' x 2 < 0) ∧ (∀ x, x > real.log 2 -> f' x 2 > 0) := 
sorry

-- Part II: Prove e^x > x^2 + 1 for x > 0
theorem part_II (x : ℝ) (h : x > 0) : exp x > x^2 + 1 :=
sorry

-- Part III: Prove sum of 1/k > ln((n+1)^3 / (3e)^n) for positive n
theorem part_III (n : ℕ) (h : 0 < n) : 
  ∑ k in finset.range n, (1 : ℝ) / (k + 1) > real.log ((n+1)^3 / (3*exp 1)^n) :=
sorry

end part_I_part_II_part_III_l296_296135


namespace triangles_relation_l296_296189

variables {A B C A' B' C': Type*}
variables (angle_A angle_A' angle_C angle_C': ℝ)
variables (AB A'B' BC B'C' AC A'C': ℝ)

-- Conditions: \angle A = \angle A' and \angle C, \angle C' are supplementary.
def angles_equal (angle_A angle_A': ℝ) : Prop :=
  angle_A = angle_A'

def angles_supplementary (angle_C angle_C': ℝ) : Prop :=
  angle_C + angle_C' = π

-- Assertion to prove.
theorem triangles_relation
  (h1: angles_equal angle_A angle_A')
  (h2: angles_supplementary angle_C angle_C') :
  AB * A'B' = BC * B'C' + AC * A'C' :=
sorry

end triangles_relation_l296_296189


namespace calculation_correct_l296_296050

noncomputable def calc_expression : Float :=
  20.17 * 69 + 201.7 * 1.3 - 8.2 * 1.7

theorem calculation_correct : calc_expression = 1640 := 
  by 
    sorry

end calculation_correct_l296_296050


namespace train_passing_time_l296_296674

theorem train_passing_time
  (length_A : ℝ) (length_B : ℝ) (time_A : ℝ) (speed_B : ℝ) 
  (Dir_opposite : true) 
  (passenger_on_A_time : time_A = 10)
  (length_of_A : length_A = 150)
  (length_of_B : length_B = 200)
  (relative_speed : speed_B = length_B / time_A) :
  ∃ x : ℝ, length_A / x = length_B / time_A ∧ x = 7.5 :=
by
  -- conditions stated
  sorry

end train_passing_time_l296_296674


namespace jelly_ratio_l296_296362

theorem jelly_ratio (G S R P : ℕ) 
  (h1 : G = 2 * S)
  (h2 : R = 2 * P) 
  (h3 : P = 6) 
  (h4 : S = 18) : 
  R / G = 1 / 3 := by
  sorry

end jelly_ratio_l296_296362


namespace trig_function_properties_l296_296276

theorem trig_function_properties :
  ∀ x : ℝ, 
    (1 - 2 * (Real.sin (x - π / 4))^2) = Real.sin (2 * x) ∧ 
    (∀ x : ℝ, Real.sin (2 * (-x)) = -Real.sin (2 * x)) ∧ 
    2 * π / 2 = π :=
by
  sorry

end trig_function_properties_l296_296276


namespace who_next_to_boris_l296_296797

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l296_296797


namespace sum_log_floor_2048_l296_296081

open Int

noncomputable def sum_log_floor : ℤ :=
  ∑ N in finset.range 2048, floor (log N / log 4)

theorem sum_log_floor_2048 : sum_log_floor = 6372 := by
  sorry

end sum_log_floor_2048_l296_296081


namespace skyscraper_anniversary_l296_296521

theorem skyscraper_anniversary (current_year_event future_happens_year target_anniversary_year : ℕ) :
  current_year_event + future_happens_year = target_anniversary_year - 5 →
  target_anniversary_year > current_year_event →
  future_happens_year = 95 := 
by
  sorry

-- Definitions for conditions:
def current_year_event := 100
def future_happens_year := 95
def target_anniversary_year := 200

end skyscraper_anniversary_l296_296521


namespace possible_values_l296_296171

noncomputable def possible_values_approximation (α : ℝ) (α_positive : 0 < α) : Set ℝ :=
  let α_4 : ℝ := ⌊α * 10000⌋ / 10000 in
  { m / 10000 | m : ℕ, 0 ≤ m ∧ m ≤ 10000 }

theorem possible_values (α : ℝ) (α_positive : 0 < α) :
  possible_values_approximation α α_positive =
  { n / 10000 | n : ℕ, 0 ≤ n ∧ n ≤ 10000 } := 
sorry

end possible_values_l296_296171


namespace range_of_f_prime_l296_296915

noncomputable def f (θ x : ℝ) : ℝ := (sin θ / 3) * x^3 + (sqrt 3 * cos θ / 2) * x^2 + tan θ

theorem range_of_f_prime (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 5 * π / 12) :
  let f_prime := (λ x, sin θ * x^2 + sqrt 3 * cos θ * x)
  in (f_prime 1) ∈ set.interval sqrt 2 2 :=
sorry

end range_of_f_prime_l296_296915


namespace triathlete_average_speed_l296_296029

def harmonic_mean (a b c : ℝ) : ℝ :=
  3 / (1 / a + 1 / b + 1 / c)

theorem triathlete_average_speed :
  let swimming_speed := 2
  let cycling_speed := 25
  let running_speed := 8
  abs (harmonic_mean swimming_speed cycling_speed running_speed - 5) < 0.5 :=
by
  sorry

end triathlete_average_speed_l296_296029


namespace symmetric_point_wrt_x_axis_l296_296730

theorem symmetric_point_wrt_x_axis :
  ∀ (x y z : ℝ), (x, y, z) = (3, -2, 1) → (x, -y, -z) = (3, 2, -1) :=
by
  intros x y z h
  rw h
  sorry

end symmetric_point_wrt_x_axis_l296_296730


namespace problem_statement_l296_296425

noncomputable theory

def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Defining properties of binomial coefficients forming an arithmetic sequence
def is_arithmetic_seq (s : list ℕ) : Prop :=
∀ i, 0 < i → i < s.length - 1 → s[i + 1] - s[i] = s[i] - s[i - 1]

-- Statement of the theorem
theorem problem_statement :
  ∃ k : ℕ, (k = 4 ∧
    (∀ n : ℕ, ¬∃ j : ℕ, (0 ≤ j ∧ j ≤ n - k + 1) ∧
      is_arithmetic_seq (list.range k |>.map (λ i, binomial n (j + i)))) ∧
    ∃ n : ℕ, (∃ j : ℕ, (0 ≤ j ∧ j ≤ n - k + 2) ∧
      is_arithmetic_seq (list.range (k - 1) |>.map (λ i, binomial n (j + i))) ∧
      ∃ m : ℕ, n = m^2 - 2 ∧ 3 ≤ m))) :=
sorry

end problem_statement_l296_296425


namespace ratio_distance_office_to_concert_hall_l296_296717

-- Definitions based on conditions
def walking_speed (w : ℝ) : Prop := w > 0
def distance_to_office (x : ℝ) : Prop := x > 0
def distance_to_concert_hall (y : ℝ) : Prop := y > 0
def riding_speed (w : ℝ) : ℝ := 5 * w
def walking_time_to_concert_hall (y w : ℝ) : ℝ := y / w
def walking_time_to_office (x w : ℝ) : ℝ := x / w
def riding_time_to_concert_hall (x y w : ℝ) : ℝ := (x + y) / (riding_speed w)

theorem ratio_distance_office_to_concert_hall
  (w x y : ℝ)
  (hw : walking_speed w)
  (hx : distance_to_office x)
  (hy : distance_to_concert_hall y)
  (equal_time : walking_time_to_concert_hall y w = walking_time_to_office x w + riding_time_to_concert_hall x y w) :
  x / y = 2 / 3 :=
begin
  sorry
end

end ratio_distance_office_to_concert_hall_l296_296717


namespace range_sin_cos_l296_296479

theorem range_sin_cos (x : ℝ) (hx : 0 < x ∧ x ≤ π / 3) : 
  ∃ y, y = sin x + cos x ∧ 1 < y ∧ y ≤ real.sqrt 2 :=
sorry

end range_sin_cos_l296_296479


namespace sum_other_y_coordinates_l296_296117

-- Given points
structure Point where
  x : ℝ
  y : ℝ

def opposite_vertices (p1 p2 : Point) : Prop :=
  -- conditions defining opposite vertices of a rectangle
  (p1.x ≠ p2.x) ∧ (p1.y ≠ p2.y)

-- Function to sum y-coordinates of two points
def sum_y_coords (p1 p2 : Point) : ℝ :=
  p1.y + p2.y

-- Main theorem to prove
theorem sum_other_y_coordinates (p1 p2 : Point) (h : opposite_vertices p1 p2) :
  sum_y_coords p1 p2 = 11 ↔ 
  (p1 = {x := 1, y := 19} ∨ p1 = {x := 7, y := -8}) ∧ 
  (p2 = {x := 1, y := 19} ∨ p2 = {x := 7, y := -8}) :=
by {
  sorry
}

end sum_other_y_coordinates_l296_296117


namespace range_of_a_l296_296879

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 1| + |x - 2| ≤ a^2 + a + 1)) → -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l296_296879


namespace projection_correct_l296_296090

open Real EuclideanSpace

def projection_of_vector (v d : EuclideanSpace) : EuclideanSpace :=
  let scalar_projection := (dot_product v d) / (dot_product d d)
  scalar_projection • d

theorem projection_correct :
  let v := (5 : ℝ) • ![1, 2/5, -3/5]
  let d := (3 : ℝ) • ![1, 2/3, -2/3]
  let p := (75/17 : ℝ) • ![1, 2/3, -2/3]
  projection_of_vector v d = p :=
sorry

end projection_correct_l296_296090


namespace find_x_value_l296_296871

theorem find_x_value (x : ℝ) (h : sqrt (2 * x + 15) = 12) : x = 64.5 :=
by
  sorry

end find_x_value_l296_296871


namespace sector_angle_is_two_l296_296154

noncomputable def central_angle (area perimeter : ℝ) : ℝ :=
let α := 2 in -- the central angle we aim to prove
    if h : (∃ r : ℝ, (1/2) * α * r^2 = area ∧ 2 * r + α * r = perimeter) then α else 0

theorem sector_angle_is_two
  (area perimeter : ℝ)
  (h_area : area = 1)
  (h_perimeter : perimeter = 4) :
  central_angle area perimeter = 2 :=
by
  -- proof goes here
  sorry

end sector_angle_is_two_l296_296154


namespace pens_more_than_notebooks_l296_296297

theorem pens_more_than_notebooks
  (N P : ℕ) 
  (h₁ : N = 30) 
  (h₂ : N + P = 110) :
  P - N = 50 := 
by
  sorry

end pens_more_than_notebooks_l296_296297


namespace cory_initial_money_l296_296402

variable (cost_per_pack : ℝ) (packs : ℕ) (additional_needed : ℝ) (total_cost : ℝ) (initial_money : ℝ)

-- Conditions
def cost_per_pack_def : Prop := cost_per_pack = 49
def packs_def : Prop := packs = 2
def additional_needed_def : Prop := additional_needed = 78
def total_cost_def : Prop := total_cost = packs * cost_per_pack
def initial_money_def : Prop := initial_money = total_cost - additional_needed

-- Theorem
theorem cory_initial_money : cost_per_pack = 49 ∧ packs = 2 ∧ additional_needed = 78 → initial_money = 20 := by
  intro h
  have h1 : cost_per_pack = 49 := h.1
  have h2 : packs = 2 := h.2.1
  have h3 : additional_needed = 78 := h.2.2
  -- sorry
  sorry

end cory_initial_money_l296_296402


namespace ellipse_polar_coordinates_problem_l296_296550

-- Definitions from conditions
def is_point_on_ellipse (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 3 * Real.sin α)

def polar_form_ellipse (θ ρ : ℝ) : Prop :=
  (ρ * Real.cos θ)^2 / 4 + (ρ * Real.sin θ)^2 / 3 = 1

-- The main problem statement to prove
theorem ellipse_polar_coordinates_problem :
  ∀ θ1 θ2 ρ1 ρ2,
  polar_form_ellipse θ1 ρ1 ∧ polar_form_ellipse θ2 ρ2 ∧ (θ2 = θ1 + Real.pi / 2 % (2 * Real.pi)) →
  (1 / ρ1^2) + (1 / ρ2^2) = 7 / 12 :=
by
  intro θ1 θ2 ρ1 ρ2
  intro h
  have h1 := h.1
  have h2 := h.2.1
  have perp := h.2.2
  sorry -- Proof omitted

end ellipse_polar_coordinates_problem_l296_296550


namespace polynomial_factorization_l296_296333

noncomputable def poly_1 : Polynomial ℤ := (Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.C 1)
noncomputable def poly_2 : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X ^ 12 - Polynomial.C 1 * Polynomial.X ^ 11 +
  Polynomial.C 1 * Polynomial.X ^ 9 - Polynomial.C 1 * Polynomial.X ^ 8 +
  Polynomial.C 1 * Polynomial.X ^ 6 - Polynomial.C 1 * Polynomial.X ^ 4 +
  Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 1 * Polynomial.X + Polynomial.C 1
noncomputable def polynomial_expression : Polynomial ℤ := Polynomial.X ^ 15 + Polynomial.X ^ 10 + Polynomial.C 1

theorem polynomial_factorization : polynomial_expression = poly_1 * poly_2 :=
  by { sorry }

end polynomial_factorization_l296_296333


namespace approximate_number_of_fish_l296_296338

/-
  In a pond, 50 fish were tagged and returned. 
  Later, in another catch of 50 fish, 2 were tagged. 
  Assuming the proportion of tagged fish in the second catch approximates that of the pond,
  prove that the total number of fish in the pond is approximately 1250.
-/

theorem approximate_number_of_fish (N : ℕ) 
  (tagged_in_pond : ℕ := 50) 
  (total_in_second_catch : ℕ := 50) 
  (tagged_in_second_catch : ℕ := 2) 
  (proportion_approx : tagged_in_second_catch / total_in_second_catch = tagged_in_pond / N) :
  N = 1250 :=
by
  sorry

end approximate_number_of_fish_l296_296338


namespace persons_next_to_Boris_l296_296825

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l296_296825


namespace polynomial_roots_a_value_l296_296452

theorem polynomial_roots_a_value {r s t a b : ℝ} 
  (h_roots : ∀ x : ℝ, (x > 0) → 8 * x^3 + 3 * a * x^2 + 6 * b * x + a = 0) 
  (h_distinct : r ≠ s ∧ r ≠ t ∧ s ≠ t) 
  (h_log_sum : Real.log 3 r + Real.log 3 s + Real.log 3 t = 3) : 
  a = -216 := 
sorry

end polynomial_roots_a_value_l296_296452


namespace integer_solutions_count_l296_296963

theorem integer_solutions_count :
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 :=
by sorry

end integer_solutions_count_l296_296963


namespace min_value_of_t_tangents_slopes_reciprocal_l296_296490

-- Definition and conditions
def f (x : ℝ) (a : ℝ) := Real.log x - a * (x - 1)
def g (x : ℝ) := Real.exp x
def t (x : ℝ) := (1 / x) * g(x)

noncomputable def t_prime (x : ℝ) := (Real.exp x * (x - 1)) / (x^2)

-- Part I
theorem min_value_of_t (m : ℝ) (h : m > 0) :
  (m ≥ 1 → t m = Real.exp m / m) ∧ (m < 1 → t 1 = Real.exp 1) :=
sorry

-- Part II
theorem tangents_slopes_reciprocal (a x1: ℝ) (h1 : x1 > 0) (h2 : a = (1 / x1) - (1 / Real.exp 1)) :
  a = 0 ∨ (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 :=
sorry

end min_value_of_t_tangents_slopes_reciprocal_l296_296490


namespace ice_volume_after_two_hours_l296_296032

-- Define the original volume of the ice
def original_volume : ℝ := 3.2

-- Define the volume retained after the first hour (1/4 of the original volume)
def volume_after_first_hour : ℝ := (1/4) * original_volume

-- Define the volume retained after the second hour (1/4 of the remaining volume)
def volume_after_second_hour : ℝ := (1/4) * volume_after_first_hour

-- Define the main theorem to prove that the volume after two hours is 0.2 cubic inches
theorem ice_volume_after_two_hours : volume_after_second_hour = 0.2 := by
  sorry

end ice_volume_after_two_hours_l296_296032


namespace area_of_enclosed_shape_l296_296638

/-- The area of the enclosed shape formed by the curve given in parametric form 
y(t) = t and x(t) = t^2 and the line y = x - 2 is 9/2. -/
theorem area_of_enclosed_shape : 
  let curve (t : ℝ) := (t, t^2)
  let line (y : ℝ) := (y, y + 2)
  ∫ y in -1..2, (y + 2 - y^2) = 9 / 2 := 
by
  let x := (y : ℝ) => y^2
  let line := (y : ℝ) => y - 2
  have integral_value : ∫ y in -1..2, (y + 2 - y^2) = (1/2 * 2^2 + 2 * 2 - 1/3 * 2^3) - (1/2 * (-1)^2 + 2 * (-1) - 1/3 * (-1)^3) := sorry
  rw integral_value
  norm_num
  exact sorry

end area_of_enclosed_shape_l296_296638


namespace no_solution_ordered_triples_l296_296434

theorem no_solution_ordered_triples (x y z : ℝ) (h1 : x + y = 3) (h2 : xy + 2 * z ^ 2 = 5) : false :=
begin
  sorry
end

end no_solution_ordered_triples_l296_296434


namespace bob_paid_14_more_than_anne_l296_296785

/-- Math problem translation to Lean 4 statement:
    Anne and Bob shared a $12$-slice pizza with given costs and consumption.
    Prove that Bob paid $14$ dollars more than Anne.
 -/
theorem bob_paid_14_more_than_anne :
  let total_cost := 20 in
  let num_slices := 12 in
  let cost_per_slice := total_cost / num_slices in
  let anne_slices := 3 in
  let anne_payment := anne_slices in
  let bob_slices := 8 in
  let bob_payment := total_cost - anne_payment in
  let payment_difference := bob_payment - anne_payment in
  payment_difference = 14 :=
by
  let total_cost := 20
  let num_slices := 12
  let cost_per_slice := total_cost / num_slices
  let anne_slices := 3
  let anne_payment := anne_slices
  let bob_slices := 8
  let bob_payment := total_cost - anne_payment
  let payment_difference := bob_payment - anne_payment
  show payment_difference = 14
  sorry

end bob_paid_14_more_than_anne_l296_296785


namespace sum_of_cubes_mod_5_divisible_l296_296870

def sum_cubes_mod_5 : ℕ :=
  (∑ k in Finset.range 50.succ, k^3) % 5

theorem sum_of_cubes_mod_5_divisible : sum_cubes_mod_5 = 0 := by
  sorry

end sum_of_cubes_mod_5_divisible_l296_296870


namespace find_x_n_l296_296546

theorem find_x_n (x n : ℕ→ ℕ) (n a: ℕ): 
 (T₄ x n : ℕ→ ℕ)
 (∃ x n :  ℕ 
(sum_eqn :  2^n = 1024):
(fourth_term_eqn : (∑ (k : Finₓ(∑ (n : ℕ)), T₄ x n)) = 0.96):
x = 0.2
n = 10)
sorry

end find_x_n_l296_296546


namespace solve_BC_squared_given_conditions_l296_296985

variables (A B C I D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited I] [Inhabited D]
variables [has_incenter A B C I] [has_intersection AI BC D] (AI ID : ℝ)

def given_conditions (A B C I : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited I]
  [has_incenter A B C I] (AI ID : ℝ) (hAI : AI = 3) (hID : ID = 2) : Prop :=
  let BI := dist B I in
  let CI := dist C I in
  BI^2 + CI^2 = 64

theorem solve_BC_squared_given_conditions (A B C I D : Type) [Inhabited A] [Inhabited B] [Inhabited C] 
    [Inhabited I] [Inhabited D] [has_incenter A B C I] [has_intersection AI BC D] (AI ID : ℝ)
    (hAI : AI = 3) (hID : ID = 2) (cond : given_conditions A B C I AI ID hAI hID) : 
    BC^2 = (272/3 : ℝ) :=
sorry

end solve_BC_squared_given_conditions_l296_296985


namespace magnitude_of_C_max_value_l296_296528

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Given condition
def condition1 : Prop := c * sin A = a * cos C

-- Statement (1): Prove the magnitude of C is π/4 given the condition
theorem magnitude_of_C (h1 : condition1) : C = π / 4 := sorry

-- Additional condition for statement (2)
def condition2 : Prop := C = π / 4

-- Statement (2): Prove the maximum value of sqrt(3) * sin A - cos (B + π / 4) is 2,
-- and it is achieved when A = π / 3 and B = 5 * π / 12
theorem max_value (h1 : condition1) (h2 : condition2) :
  ∃ (A B : ℝ), 
  (sqrt 3 * sin A - cos (B + π / 4) = 2) ∧ 
  (A = π / 3) ∧ 
  (B = 5 * π / 12) := sorry

end magnitude_of_C_max_value_l296_296528


namespace planting_schemes_l296_296989

/--
There are 6 plots of land. Each plot can be planted with either type A or type B vegetables.
The constraint is that no two adjacent plots can both have type A vegetables.
Prove that the total number of possible planting schemes is 21.
-/
theorem planting_schemes (P : Fin 6 → Bool) (h : ∀ i, i < 5 → (P i = true → P (i + 1) = false)) :
  (∃0 ≤ t ≤ 3, (Nat.choose (6 - t) t) = 21) :=
begin
  sorry
end

end planting_schemes_l296_296989


namespace sample_variance_eq_two_l296_296141

variable {α : Type*} [field α]

-- Definitions for the problem
def sample := [7, 5, x, 3, 4 : α]
def mean_of_sample (s : list α) (n : α) := (s.sum / s.length) = n

-- Assertion of the mean condition
def mean_condition := mean_of_sample sample 5

-- Definition for the variance
def variance_of_sample (s : list α) : α :=
  (1 / s.length) * (s.map (λ x, (x - mean_of_sample s 5) ^ 2)).sum

-- The proof problem statement
theorem sample_variance_eq_two (x : α) (h : mean_condition) : variance_of_sample sample = 2 :=
sorry

end sample_variance_eq_two_l296_296141


namespace abs_pi_pi_4_eq_4_pi_sq_l296_296395

theorem abs_pi_pi_4_eq_4_pi_sq : abs (π * abs (π * 4)) = 4 * π^2 :=
by
  sorry

end abs_pi_pi_4_eq_4_pi_sq_l296_296395


namespace rectangle_circumference_15pi_l296_296763

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ := 
  Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def circumference_of_circle (d : ℝ) : ℝ := 
  Real.pi * d
  
theorem rectangle_circumference_15pi :
  let a := 9
  let b := 12
  let diagonal := rectangle_diagonal a b
  circumference_of_circle diagonal = 15 * Real.pi :=
by 
  sorry

end rectangle_circumference_15pi_l296_296763


namespace triangle_height_equality_l296_296195

theorem triangle_height_equality
  (A B C D E Z I T K : Type*)
  (triangle : Triangle A B C)
  (altitude_AD : Altitude A D B C)
  (angle_bisectors : AngleBisectors A E B Z intersecting_at I)
  (perpendicular_IT : Perpendicular I T A C)
  (perpendicular_line : PerpendicularLine e A C)
  (intersection_ET_e : Extension E T intersects e at K)
  : Distance A K = Distance A D :=
sorry

end triangle_height_equality_l296_296195


namespace car_average_speed_l296_296720

noncomputable def uphill_speed : ℝ := 30
noncomputable def downhill_speed : ℝ := 60
noncomputable def uphill_distance : ℝ := 100
noncomputable def downhill_distance : ℝ := 50

def average_speed (up_speed down_speed up_distance down_distance : ℝ) : ℝ :=
  (up_distance + down_distance) / ((up_distance / up_speed) + (down_distance / down_speed))

theorem car_average_speed :
  average_speed uphill_speed downhill_speed uphill_distance downhill_distance = 36.06 :=
by
  sorry

end car_average_speed_l296_296720


namespace calculate_diagonal_length_squared_calculate_mnp_sum_l296_296198

noncomputable theory

-- Define the rectangle with given area and width/height ratio
def Rectangle (w h : ℝ) := w * h = 24 ∧ w = 3 * h

-- Define the length of the diagonal given width and height
def diagonal_length_squared (w h : ℝ) := (w^2 + h^2)

theorem calculate_diagonal_length_squared :
  ∃ w h : ℝ, Rectangle w h ∧ diagonal_length_squared w h = 40 := by
  sorry

-- Define the parameters m, n, and p
def m := 40
def n := 0
def p := 1

theorem calculate_mnp_sum : (m + n + p) = 41 := by
  sorry

end calculate_diagonal_length_squared_calculate_mnp_sum_l296_296198


namespace who_next_to_boris_l296_296793

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l296_296793


namespace num_integers_satisfying_ineq_count_l296_296943

theorem num_integers_satisfying_ineq_count :
  {x : ℤ | -6 ≤ 3 * (x : ℤ) + 2 ∧ 3 * (x : ℤ) + 2 ≤ 9}.finite.to_finset.card = 5 :=
by
  sorry

end num_integers_satisfying_ineq_count_l296_296943


namespace simplify_trigonometric_expression_l296_296902

theorem simplify_trigonometric_expression (α : ℝ) (hα : π / 2 < α ∧ α < π) :
  (sqrt ((1 + sin α) / (1 - sin α)) - sqrt ((1 - sin α) / (1 + sin α))) = -2 * tan α :=
by
  sorry

end simplify_trigonometric_expression_l296_296902


namespace number_of_cyclic_symmetric_expressions_l296_296003

def cyclically_symmetric_1 (σ : ℕ → ℕ → ℕ → ℕ) : Prop := 
  ∀ a b c, σ a b c = σ b c a ∧ σ a b c = σ c a b 

def cyclically_symmetric_2 (σ : ℕ → ℕ → ℕ → ℕ) : Prop := 
  ∀ a b c, σ a b c = σ b c a ∧ σ a b c = σ c a b

def cyclically_symmetric_3 (σ : ℝ → ℝ → ℝ → ℝ) : Prop := 
  ∀ A B C, σ A B C = σ B C A ∧ σ A B C = σ C A B

theorem number_of_cyclic_symmetric_expressions : ℕ :=
  let σ1 := λ (a b c : ℕ), a * b * c in
  let σ2 := λ (a b c : ℕ), a^2 - b^2 + c^2 in
  let σ3 := λ (A B C : ℝ), Real.cos C * Real.cos (A - B) - Real.cos C^2 in
  (if cyclically_symmetric_1 σ1 then 1 else 0) +
  (if cyclically_symmetric_2 σ2 then 1 else 0) +
  (if cyclically_symmetric_3 σ3 then 1 else 0) = 1

end number_of_cyclic_symmetric_expressions_l296_296003


namespace carly_butterfly_days_l296_296840

-- Define the conditions
variable (x : ℕ) -- number of days Carly practices her butterfly stroke
def butterfly_hours_per_day := 3  -- hours per day for butterfly stroke
def backstroke_hours_per_day := 2  -- hours per day for backstroke stroke
def backstroke_days_per_week := 6  -- days per week for backstroke stroke
def total_hours_per_month := 96  -- total hours practicing swimming in a month
def weeks_in_month := 4  -- number of weeks in a month

-- The proof problem
theorem carly_butterfly_days :
  (butterfly_hours_per_day * x + backstroke_hours_per_day * backstroke_days_per_week) * weeks_in_month = total_hours_per_month
  → x = 4 := 
by
  sorry

end carly_butterfly_days_l296_296840


namespace arithmetic_sequence_sum_l296_296700

theorem arithmetic_sequence_sum (d : ℕ) (y : ℕ) (x : ℕ) (h_y : y = 39) (h_d : d = 6) 
  (h_x : x = y - d) : 
  x + y = 72 := by 
  sorry

end arithmetic_sequence_sum_l296_296700


namespace turban_price_l296_296935

theorem turban_price (initial_salary: ℝ) (raise_percent: ℝ) (total_months: ℕ) (received_salary: ℝ) (price_of_turban: ℝ) : 
  initial_salary = 90 → 
  raise_percent = 0.10 → 
  total_months = 9 → 
  received_salary = 65 → 
  price_of_turban = 4.75 :=
begin
  intros,
  let half_year_salary := initial_salary / 2,
  let initial_monthly_salary := initial_salary / 12,
  let raised_salary := initial_monthly_salary + (initial_monthly_salary * raise_percent),
  let first_6_months := initial_monthly_salary * 6,
  let next_3_months := raised_salary * 3,
  let total_cash_salary := first_6_months + next_3_months,
  
  have h1 : total_cash_salary = 69.75, {
    calc
    total_cash_salary = 6 * (initial_salary / 12) + 3 * raised_salary          : by sorry
                    ... = 45 + 24.75 : by sorry
                    ... = 69.75 : by sorry
  },
  have h2 : price_of_turban = total_cash_salary - received_salary, 
    from sorry,

  rw [h1] at h2,
  calc
  price_of_turban = 69.75 - received_salary : by sorry
                  ... = 4.75                 : by sorry
end

end turban_price_l296_296935


namespace num_unit_cubes_intersected_by_plane_l296_296363

theorem num_unit_cubes_intersected_by_plane :
  let n := 4
  let unit_cube_count := n ^ 3
  let diagonal_length := n * Real.sqrt 3
  ∀ (plane : ℝ → ℝ → ℝ → Prop), 
    (∀ x y z, plane x y z → are_perpendicular (diagonal (n, n, n)) plane ∧ bisects (diagonal midpoint (n, n, n)) plane) →
    intersected_unit_cubes plane unit_cube_count = 40 :=
by
  sorry

end num_unit_cubes_intersected_by_plane_l296_296363


namespace sum_of_digits_base10_representation_l296_296998

def digit_sum (n : ℕ) : ℕ := sorry  -- Define a function to calculate the sum of digits

noncomputable def a : ℕ := 7 * (10 ^ 1234 - 1) / 9
noncomputable def b : ℕ := 2 * (10 ^ 1234 - 1) / 9
noncomputable def product : ℕ := 7 * a * b

theorem sum_of_digits_base10_representation : digit_sum product = 11100 := 
by sorry

end sum_of_digits_base10_representation_l296_296998


namespace marco_scores_l296_296287

theorem marco_scores :
  ∃ s1 s2 s3 s4 s5 : ℤ,
    s1 = 86 ∧
    s2 = 75 ∧
    s3 = 71 ∧
    (s1 + s2 + s3 + s4 + s5) = 5 * 82 ∧
    s1 < 90 ∧ s2 < 90 ∧ s3 < 90 ∧ s4 < 90 ∧ s5 < 90 ∧
    s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s2 ≠ s3 ∧
    s2 ≠ s4 ∧ s2 ≠ s5 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s4 ≠ s5 ∧
    [s4, s5].permutations.any (λ l, l = [89, 87] ∨ l = [87, 89]) ∧
    ([s1, s2, s3, s4, s5].sort (≥) = [89, 87, 86, 75, 71]) :=
by
  sorry

end marco_scores_l296_296287


namespace average_percentage_l296_296514

theorem average_percentage (s1 s2 : ℕ) (a1 a2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : a1 = 70) (h3 : s2 = 10) (h4 : a2 = 90) (h5 : n = 25)
  : ((s1 * a1 + s2 * a2) / n : ℕ) = 78 :=
by
  -- We include sorry to skip the proof part.
  sorry

end average_percentage_l296_296514


namespace sum_of_roots_eq_neg3_l296_296523

theorem sum_of_roots_eq_neg3
  (a b c : ℝ)
  (h_eq : 2 * x^2 + 6 * x - 1 = 0)
  (h_a : a = 2)
  (h_b : b = 6) :
  (x1 x2 : ℝ) → x1 + x2 = -b / a :=
by
  sorry

end sum_of_roots_eq_neg3_l296_296523


namespace isosceles_triangle_l296_296603

variables {A B C M N : ℝ}

-- Define the lengths of the sides of the triangle and segments
variables (a b c x y : ℝ)

-- Conditions based on the problem statement
axiom triangle_ABC (ABC_triangle: triangle):
  (point_on_side_A_B M)
  (point_on_side_B_C N)
  (perimeter_eq_1: x + distance M C + b = y + (a - y) + b)
  (perimeter_eq_2: x + y + c = (a - x) + (c - x) + a)

-- Prove that triangle ABC is isosceles
theorem isosceles_triangle (ABC_triangle): isosceles_triangle ABC_triangle := sorry

end isosceles_triangle_l296_296603


namespace boris_neighbors_l296_296791

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l296_296791


namespace inequality_to_prove_l296_296488

variables {a b c A B C : ℝ}

-- Given conditions
def condition1 : ∀ x, f x = Real.cos x := sorry
def condition2 : 3 * a^2 + 3 * b^2 - c^2 = 4 * a * b := sorry

-- Definitions of the function and solutions
def f (x : ℝ) : ℝ := Real.cos x

theorem inequality_to_prove (h1: 3 * a^2 + 3 * b^2 - c^2 = 4 * a * b)
  (h2: A + B ≤ Real.pi / 2) (h3: 0 < B)
  (h4: B ≤ Real.pi / 2 - A) : f (Real.cos A) ≤ f (Real.sin B) :=
sorry

end inequality_to_prove_l296_296488


namespace intersection_of_sets_l296_296475

variable (x : ℝ)
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets 
  (hA : ∀ x, x ∈ A ↔ -2 < x ∧ x ≤ 1)
  (hB : ∀ x, x ∈ B ↔ 0 < x ∧ x ≤ 1) :
  ∀ x, (x ∈ A ∩ B) ↔ (0 < x ∧ x ≤ 1) := 
by
  sorry

end intersection_of_sets_l296_296475


namespace cricketer_average_increase_l296_296147

theorem cricketer_average_increase (A : ℝ) (H1 : 18 * A + 98 = 19 * 26) :
  26 - A = 4 :=
by
  sorry

end cricketer_average_increase_l296_296147


namespace avg_move_to_california_l296_296178

noncomputable def avg_people_per_hour (total_people : ℕ) (total_days : ℕ) : ℕ :=
  let total_hours := total_days * 24
  let avg_per_hour := total_people / total_hours
  let remainder := total_people % total_hours
  if remainder * 2 < total_hours then avg_per_hour else avg_per_hour + 1

theorem avg_move_to_california : avg_people_per_hour 3500 5 = 29 := by
  sorry

end avg_move_to_california_l296_296178


namespace simple_interest_correct_l296_296370

-- Define the given conditions
def Principal : ℝ := 9005
def Rate : ℝ := 0.09
def Time : ℝ := 5

-- Define the simple interest function
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- State the theorem to prove the total interest earned
theorem simple_interest_correct : simple_interest Principal Rate Time = 4052.25 := sorry

end simple_interest_correct_l296_296370


namespace num_of_valid_pairs_l296_296853

theorem num_of_valid_pairs : ∃ (n : ℕ), n = 6 ∧ ∀ (a b : ℕ), a + b = 13 → 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  ( (a = 4 ∧ b = 9) ∨ (a = 5 ∧ b = 8) ∨ (a = 6 ∧ b = 7) ∨
    (a = 7 ∧ b = 6) ∨ (a = 8 ∧ b = 5) ∨ (a = 9 ∧ b = 4) ) :=
begin
  -- We just provide the statement of the theorem, not its proof.
  sorry
end

end num_of_valid_pairs_l296_296853


namespace hyperbola_eccentricity_is_sqrt5_l296_296895

noncomputable def hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P Q : ℝ × ℝ)
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : (P.1^2) / a^2 - (P.2^2) / b^2 = 1)
  (h₄ : dist P F₁ = 2 * dist Q F₁)
  (h₅ : dist F₁ F₂ = 2 * sqrt (a^2 + b^2)) -- distance between foci
  (h₆ : (dist P Q = 2 * dist Q F₁)) : 
  ecc : ℝ :=
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e

theorem hyperbola_eccentricity_is_sqrt5 (a b : ℝ) (F₁ F₂ P Q : ℝ × ℝ) 
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : (P.1^2) / a^2 - (P.2^2) / b^2 = 1)
  (h₄ : dist P F₁ = 2 * dist Q F₁)
  (h₅ : dist F₁ F₂ = 2 * sqrt (a^2 + b^2)) -- distance between foci
  (h₆ : dist Q P = 2 * dist Q F₁) :
  hyperbola_eccentricity a b F₁ F₂ P Q h₁ h₂ h₃ h₄ h₅ h₆ = sqrt 5 := sorry

end hyperbola_eccentricity_is_sqrt5_l296_296895


namespace sum_first_11_terms_is_neg66_l296_296106

-- Sequence definition
def a_n (n : ℕ) : ℤ := 1 - 2 * n 

-- Sum of the first n terms of the sequence
def S_n (n : ℕ) : ℤ := ∑ i in range n, a_n i 

-- Sum of the first 11 terms of the transformed sequence
def sum_first_11_terms_sn_div_n : ℤ := ∑ k in range 11, S_n (k + 1) / (k + 1)

-- Theorem statement
theorem sum_first_11_terms_is_neg66 : sum_first_11_terms_sn_div_n = -66 := by 
  sorry

end sum_first_11_terms_is_neg66_l296_296106


namespace two_digit_number_satisfying_conditions_l296_296406

theorem two_digit_number_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 8 ∧
  ∀ p ∈ s, ∃ (a b : ℕ), p = (a, b) ∧
    (10 * a + b < 100) ∧
    (a ≥ 2) ∧
    (10 * a + b + 10 * b + a = 110) :=
by
  sorry

end two_digit_number_satisfying_conditions_l296_296406


namespace Player_A_Wins_For_All_l296_296761

def player_a_winning_strategy (n : ℕ) : Prop := 
  -- Base case where A wins if 3 is reached
  if n = 1 ∨ n = 6 ∨ n = 7 then true
  else if (n = 2 ∨ n = 4 ∨ n = 5 ∨ n = 9) then false
  -- Inductive step considering the conditions described
  else if (n > 7 ∨ (n < 9 ∧ n > 3)) then
    if (even n ∧ player_a_winning_strategy (n / 2)) then true
    else if (n % 4 = 0 ∧ player_a_winning_strategy (n / 4)) then true
    else if (odd n ∧ n > 1 ∧ (player_a_winning_strategy (3 * n + 1) ∨ player_a_winning_strategy (3 * n - 1))) then 
      true
  else false

theorem Player_A_Wins_For_All (n : ℕ) : player_a_winning_strategy n := by
  sorry

end Player_A_Wins_For_All_l296_296761


namespace equation_holds_except_two_values_l296_296341

noncomputable def check_equation (a y : ℝ) (h : a ≠ 0) : Prop :=
  (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 ↔ y ≠ a ∧ y ≠ -a

theorem equation_holds_except_two_values (a y: ℝ) (h: a ≠ 0): check_equation a y h := sorry

end equation_holds_except_two_values_l296_296341


namespace Sue_chewing_gums_count_l296_296651

theorem Sue_chewing_gums_count (S : ℕ) 
  (hMary : 5 = 5) 
  (hSam : 10 = 10) 
  (hTotal : 5 + 10 + S = 30) : S = 15 := 
by {
  sorry
}

end Sue_chewing_gums_count_l296_296651


namespace num_integers_satisfying_inequality_l296_296954

theorem num_integers_satisfying_inequality : 
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 := 
by 
  sorry

end num_integers_satisfying_inequality_l296_296954


namespace fourth_quadrant_angle_div_three_not_first_l296_296120

theorem fourth_quadrant_angle_div_three_not_first (k : ℤ) (α : ℝ) 
  (hα : α ∈ set.Ioo (2 * k * real.pi + 3 * real.pi / 2) (2 * k * real.pi + 2 * real.pi)) : 
  ¬(∃ n : ℤ, (α / 3) ∈ set.Ioo (2 * n * real.pi) (2 * n * real.pi + real.pi / 2)) :=
sorry

end fourth_quadrant_angle_div_three_not_first_l296_296120


namespace ben_kitchen_allocation_l296_296044

theorem ben_kitchen_allocation 
  (bonus : ℝ)
  (holidays_fraction : ℝ)
  (gifts_fraction : ℝ)
  (left_over : ℝ)
  (kitchen_fraction : ℝ)
  (total_bonus : ℝ)
  (h1 : bonus = 1496)
  (h2 : holidays_fraction = 1/4)
  (h3 : gifts_fraction = 1/8)
  (h4 : left_over = 867)
  (h5 : kitchen_fraction * bonus + holidays_fraction * bonus + gifts_fraction * bonus + left_over = bonus) : 
  kitchen_fraction = 221 / 748 := 
by
  rw [h1] at h5
  field_simp at h5
  norm_num at h5
  sorry

end ben_kitchen_allocation_l296_296044


namespace exists_midpoint_l296_296676

noncomputable def exists_parallel_through_point (l : Line) (P : Point) : ∃ m : Line, m ∥ l ∧ P ∈ m :=
by
  sorry

theorem exists_midpoint (A B : Point) : ∃ M : Point, dist A M = dist M B :=
by
  sorry

end exists_midpoint_l296_296676


namespace find_D_and_zeros_l296_296478

noncomputable def A := {x : ℝ | x > 0}
noncomputable def B (a : ℝ) := {x : ℝ | 2 * x ^ 2 - 3 * (1 + a) * x + 6 * a > 0}
noncomputable def D (a : ℝ) := A ∩ B a
noncomputable def f (a x : ℝ) := x^2 - (1 + a) * x + a
noncomputable def f_zeros_in_D (a : ℝ) (S : set ℝ) := {x ∈ S | f a x = 0}

theorem find_D_and_zeros (a : ℝ) (h : a < 1) : 
  (if 1 > a ∧ a > 1/3 then D a = {x : ℝ | x > 0 } 
  else if a = 1/3 then D a = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x > 1}
  else if 0 < a ∧ a < 1/3 then ∃ x1 x2 : ℝ, x1 < x2 ∧ 
    D a = ({x : ℝ | 0 < x ∧ x < x1} ∪ {x : ℝ | x > x2 })
  else ∃ x2 : ℝ, D a = {x : ℝ | x > x2 })
  ∧ 
  (if 1 > a ∧ a > 1/3 then f_zeros_in_D a (D a) = {1, a}
  else if a = 1/3 then f_zeros_in_D a (D a) = {1/3}
  else if 0 < a ∧ a < 1/3 then f_zeros_in_D a (D a) = {a}
  else f_zeros_in_D a (D a) = ∅) :=
sorry

end find_D_and_zeros_l296_296478


namespace who_next_to_boris_l296_296794

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l296_296794


namespace maximum_value_PM_PN_l296_296181

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sin θ)

noncomputable def line_l (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, √2 + t * Real.sin α)

theorem maximum_value_PM_PN {α : ℝ} (h : ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ curve_C t1 = line_l t1 α ∧ curve_C t2 = line_l t2 α) :
  ∃ M N : ℝ × ℝ, 
  (M = curve_C (classical.some h) ∧ N = curve_C (classical.some_spec h).some) ∧
  abs (dist (0, √2) M) + abs (dist (0, √2) N) ≤ (4 * √6) / 3 := 
sorry

end maximum_value_PM_PN_l296_296181


namespace symmetry_about_y_axis_l296_296997

def f (x : ℝ) : ℝ := 2^(x + 1)
def g (x : ℝ) : ℝ := 2^(1 - x)

theorem symmetry_about_y_axis :
  ∀ x : ℝ, f (-x) = g x :=
by
  intro x
  unfold f g
  sorry

end symmetry_about_y_axis_l296_296997


namespace noncongruent_triangles_count_l296_296965

/-- Prove the number of noncongruent integer-sided triangles with positive area,
    perimeter less than 20, that are neither equilateral, isosceles, nor right triangles
    is 17 -/
theorem noncongruent_triangles_count:
  ∃ (s : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ s → a + b + c < 20 ∧ a + b > c ∧ a < b ∧ b < c ∧ 
         ¬(a = b ∨ b = c ∨ a = c) ∧ ¬(a * a + b * b = c * c)) ∧ 
    s.card = 17 := 
sorry

end noncongruent_triangles_count_l296_296965


namespace limit_comparison_l296_296116

variable {a b : ℤ → ℝ}

-- Condition: For all n in positive integers, a(n) > b(n)
def seq_condition (n : ℤ) : Prop := n > 0 → a n > b n

-- Condition: The limits of the sequences {a_n} and {b_n} as n → +∞ are A and B respectively.
def limit_a (A : ℝ) : Prop := filter.tendsto a filter.at_top (nhds A)
def limit_b (B : ℝ) : Prop := filter.tendsto b filter.at_top (nhds B)

theorem limit_comparison (A B : ℝ) (h_seq : ∀ n, seq_condition n) (h_lim_a : limit_a A) (h_lim_b : limit_b B) : 
  A ≥ B :=
sorry

end limit_comparison_l296_296116


namespace tangent_slope_at_1_0_l296_296659

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_slope_at_1_0 : (deriv f 1) = 3 := by
  sorry

end tangent_slope_at_1_0_l296_296659


namespace smallest_real_number_among_l296_296778

theorem smallest_real_number_among (-2 : ℝ) (2 : ℝ) (-4 : ℝ) (-1 : ℝ) :
  ∃ smallest : ℝ, (smallest = -4) ∧ (smallest < -2) ∧ (smallest < -1) ∧ (smallest < 2) :=
by
  sorry

end smallest_real_number_among_l296_296778


namespace continuous_function_equality_l296_296068

-- Lean code
theorem continuous_function_equality
    (f : ℝ → ℝ)
    (hf_cont : ∀ x > 0, ContinuousAt f x)
    (hf_1 : f 1 = 2)
    (hf_add : ∀ x y > 0, f (x + y) = f x + f y) :
    f 4 = 8 :=
sorry

end continuous_function_equality_l296_296068


namespace sum_of_sequence_is_124_l296_296208

noncomputable def a : ℕ+ → ℝ
| 1 := 2
| (n+1) := 2 / (a n + 1)

noncomputable def b : ℕ+ → ℝ
| n := abs ((a n + 2) / (a n - 1))

noncomputable def S (n : ℕ+) : ℝ :=
finset.sum (finset.range n) (λ k, b (k + 1))

theorem sum_of_sequence_is_124 : S 5 = 124 := sorry

end sum_of_sequence_is_124_l296_296208


namespace sum_of_distances_to_asymptotes_l296_296980

variable (x y : ℝ)

def hyperbola (x y : ℝ) : Prop :=
  x^2 / 3 - y^2 = 1

def foci (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (-2, 0) ∧ F2 = (2, 0)

def condition (P F1 F2 : ℝ × ℝ) : Prop :=
  let PF1 := (fst P - fst F1, snd P - snd F1)
  let PF2 := (fst P - fst F2, snd P - snd F2)
  (PF1.1 * PF2.1) + (PF1.2 * PF2.2) = 1

theorem sum_of_distances_to_asymptotes (P F1 F2 : ℝ × ℝ) (hx : hyperbola (fst P) (snd P)) (hf : foci F1 F2) (hc : condition P F1 F2) :
  let d1 := abs ((fst P) + sqrt(3) * (snd P)) / sqrt(1 + 3)
  let d2 := abs ((fst P) - sqrt(3) * (snd P)) / sqrt(1 + 3)
  d1 + d2 = 3 * sqrt(2) / 2 := 
  sorry

end sum_of_distances_to_asymptotes_l296_296980


namespace kathryn_gave_56_pencils_l296_296383

-- Define the initial and total number of pencils
def initial_pencils : ℕ := 9
def total_pencils : ℕ := 65

-- Define the number of pencils Kathryn gave to Anthony
def pencils_given : ℕ := total_pencils - initial_pencils

-- Prove that Kathryn gave Anthony 56 pencils
theorem kathryn_gave_56_pencils : pencils_given = 56 :=
by
  -- Proof is omitted as per the requirement
  sorry

end kathryn_gave_56_pencils_l296_296383


namespace total_painting_cost_l296_296707

theorem total_painting_cost :
  let living_room_area := 600
  let bedroom_area := 450
  let kitchen_area := 300
  let bathroom_area := 100
  let living_room_cost_per_sqft := 30
  let bedroom_cost_per_sqft := 25
  let kitchen_cost_per_sqft := 20
  let bathroom_cost_per_sqft := 15
  let living_room_cost := living_room_area * living_room_cost_per_sqft
  let total_bedrooms_cost := 2 * bedroom_area * bedroom_cost_per_sqft
  let kitchen_cost := kitchen_area * kitchen_cost_per_sqft
  let total_bathrooms_cost := 2 * bathroom_area * bathroom_cost_per_sqft
  let total_cost := living_room_cost + total_bedrooms_cost + kitchen_cost + total_bathrooms_cost
  in total_cost = 49500 := by
  {
    let living_room_area := 600
    let bedroom_area := 450
    let kitchen_area := 300
    let bathroom_area := 100
    let living_room_cost_per_sqft := 30
    let bedroom_cost_per_sqft := 25
    let kitchen_cost_per_sqft := 20
    let bathroom_cost_per_sqft := 15
    let living_room_cost := living_room_area * living_room_cost_per_sqft
    let total_bedrooms_cost := 2 * bedroom_area * bedroom_cost_per_sqft
    let kitchen_cost := kitchen_area * kitchen_cost_per_sqft
    let total_bathrooms_cost := 2 * bathroom_area * bathroom_cost_per_sqft
    let total_cost := living_room_cost + total_bedrooms_cost + kitchen_cost + total_bathrooms_cost
    show total_cost = 49500
    sorry
  }

end total_painting_cost_l296_296707


namespace log2_lt_prob_l296_296904

open Set

noncomputable def prob_event (x : ℝ) := x ∈ Ioc 0 2 ∧ x ∈ Icc 0 3

theorem log2_lt_prob : 
  let space := Icc (0 : ℝ) 3 in
  let event := {x | log 2 x < 1} ∩ space in
  (event.measure (volume.restrict space)) / (space.measure volume) = 2 / 3 := sorry

end log2_lt_prob_l296_296904


namespace sum_of_elements_in_star_l296_296851

-- Definitions given in the problem
def A : Set ℕ := {1, 2}
def B : Set ℕ := {0, 2}
def star (A B : Set ℕ) : Set ℕ := {z | ∃ (x ∈ A) (y ∈ B), z = x * y}

-- The statement to prove the sum of elements in A*B is 6
theorem sum_of_elements_in_star (A B : Set ℕ) (hA : A = {1, 2}) (hB : B = {0, 2}) :
  (∑ x in (star A B).toFinset, x) = 6 := 
sorry

end sum_of_elements_in_star_l296_296851


namespace fraction_meaningful_l296_296667

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ∃ y, y = 1 / (x - 1) :=
by
  sorry

end fraction_meaningful_l296_296667


namespace problem_1_problem_2_l296_296912

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem problem_1 (x1 x2 : ℝ) (h1 : 1 ≤ x1) (h2 : 1 ≤ x2) (h3 : x1 < x2) :
  f x1 > f x2 :=
by
  -- Proof will go here
  sorry

theorem problem_2 :
  (∀ x ∈ set.Icc 2 5, f x ≤ f 2) ∧ (∀ x ∈ set.Icc 2 5, f x ≥ f 5) :=
by
  -- Proof will go here
  sorry

end problem_1_problem_2_l296_296912


namespace person_next_to_Boris_arkady_galya_l296_296805

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l296_296805


namespace triangle_is_isosceles_l296_296593

variables {A B C M N : Type} [pseudo_metric_space A] [pseudo_metric_space B] [pseudo_metric_space C]
[pseudo_metric_space M] [pseudo_metric_space N] 

variables {dist : ∀ {X : Type} [pseudo_metric_space X], X → X → ℝ}
variables {a b c x y : ℝ} -- edge lengths

-- Define the points and their distances
variables {MA MB NA NC : ℝ} (tABC : triangle A B C) 

-- Conditions from the problem
def condition1 : Prop :=
  dist A M + dist M C + dist C A = dist C N + dist N A + dist A C

def condition2 : Prop :=
  dist A N + dist N B + dist B A = dist C M + dist M B + dist B C

-- Proving the triangle is isosceles
theorem triangle_is_isosceles (tABC : triangle A B C) 
    (h1 : condition1)
    (h2 : condition2) : isosceles tABC :=
sorry

end triangle_is_isosceles_l296_296593


namespace vector_solution_l296_296502

theorem vector_solution
  (x y : ℝ)
  (h1 : (2*x - y = 0))
  (h2 : (x^2 + y^2 = 20)) :
  (x = 2 ∧ y = 4) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end vector_solution_l296_296502


namespace barb_saves_100_l296_296043

theorem barb_saves_100 (C : ℝ) (hC : C = 180) : 
  let paid := (C / 2) - 10 in
  C - paid = 100 := 
by
  sorry

end barb_saves_100_l296_296043


namespace sale_in_fourth_month_l296_296007

theorem sale_in_fourth_month (
  sale_month1 : ℕ := 6535,
  sale_month2 : ℕ := 6927,
  sale_month3 : ℕ := 6855,
  sale_month5 : ℕ := 6562,
  sale_month6 : ℕ := 4891,
  total_sales : ℕ := 39000
) : 
  ∃ sale_month4 : ℕ, 
    sale_month4 = total_sales - (sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6) ∧
    sale_month4 = 7230 :=
by {
  existsi (total_sales - (sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6)),
  split,
  { refl },
  { simp, sorry }  
}

end sale_in_fourth_month_l296_296007


namespace max_L_for_sweet_2023_tuple_l296_296315

def is_sweet_2023_tuple (a : Fin 2023 → Nat) : Prop :=
  (∑ i, a i = 2023) ∧ (∑ i, a i / 2^(i.1 + 1) ≤ 1)

theorem max_L_for_sweet_2023_tuple :
  ∀ a : Fin 2023 → Nat, is_sweet_2023_tuple a → (∑ i, (i.1 + 1) * a i) ≥ 20230 :=
by
  intros a h
  sorry

end max_L_for_sweet_2023_tuple_l296_296315


namespace find_100_digit_number_l296_296348

-- Main statement to be proved in Lean 4
theorem find_100_digit_number (b : ℕ) (hb : b ∈ {1, 2, 3}) :
    ∃ (N : ℕ), N = 325 * b * 10^97 ∧ (∃ (M : ℕ), M = N / 13 ∧ N - M*13 = 0) :=
by
    sorry -- Proof to be completed

end find_100_digit_number_l296_296348


namespace geometric_sequence_increasing_condition_l296_296207

variable {α : Type*} [OrderedField α]

-- Definitions from the problem conditions
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = r * a n

def increasing_sequence (a : ℕ → α) : Prop :=
  ∀ m n : ℕ, m < n → a m < a n

-- Problem statement to be proven
theorem geometric_sequence_increasing_condition :
  ∀ (a : ℕ → α),
  is_geometric_sequence a → 
  a 1 < a 2 → a 2 < a 4 → 
  (¬ (∀ n m : ℕ, n < m → a n < a m) → (∃ n m : ℕ, n < m ∧ a n ≥ a m)) :=
begin
  intros,
  sorry
end

end geometric_sequence_increasing_condition_l296_296207


namespace train_speed_l296_296771

noncomputable def distance : ℝ := 45  -- 45 km
noncomputable def time_minutes : ℝ := 30  -- 30 minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert minutes to hours

theorem train_speed (d : ℝ) (t_m : ℝ) : d = 45 → t_m = 30 → d / (t_m / 60) = 90 :=
by
  intros h₁ h₂
  sorry

end train_speed_l296_296771


namespace perimeter_of_region_l296_296366

theorem perimeter_of_region : 
  let side := 1
  let diameter := side
  let radius := diameter / 2
  let full_circumference := 2 * Real.pi * radius
  let arc_length := (3 / 4) * full_circumference
  let total_arcs := 4
  let perimeter := total_arcs * arc_length
  perimeter = 3 * Real.pi :=
by 
  sorry

end perimeter_of_region_l296_296366


namespace y1_minus_y2_positive_l296_296894

-- Define the inverse proportion function.
def inverse_proportion_function (x : ℝ) : ℝ := 2 / x

-- Define conditions.
variables (x1 x2 : ℝ)
variable (y1 : ℝ) (y2 : ℝ)
hypothesis h1 : x1 < x2
hypothesis h2 : x2 < 0
hypothesis h3 : y1 = inverse_proportion_function x1
hypothesis h4 : y2 = inverse_proportion_function x2

-- Prove that y1 - y2 is positive.
theorem y1_minus_y2_positive : (y1 - y2) > 0 := 
by 
  -- apply the proof according to the problem
  sorry

end y1_minus_y2_positive_l296_296894


namespace spring_outing_arrangements_l296_296304

theorem spring_outing_arrangements : ∃ n : ℕ, n = (A 5 2) * (6 ^ 4) := sorry

def A (n k : ℕ) : ℕ := nat.factorial n / nat.factorial (n - k)

def spring_outing_arrangements_proof : nat :=
  A 5 2 * 6 ^ 4

example : spring_outing_arrangements_proof = (A 5 2) * (6 ^ 4) :=
by simp [spring_outing_arrangements_proof]

end spring_outing_arrangements_l296_296304


namespace parabola_focus_coordinates_l296_296643

theorem parabola_focus_coordinates (x y : ℝ) (h : y = -2 * x^2) : (0, -1 / 8) = (0, (-1 / 2) * (y: ℝ)) :=
sorry

end parabola_focus_coordinates_l296_296643


namespace triangle_is_isosceles_l296_296596

variables {A B C M N : Type} [pseudo_metric_space A] [pseudo_metric_space B] [pseudo_metric_space C]
[pseudo_metric_space M] [pseudo_metric_space N] 

variables {dist : ∀ {X : Type} [pseudo_metric_space X], X → X → ℝ}
variables {a b c x y : ℝ} -- edge lengths

-- Define the points and their distances
variables {MA MB NA NC : ℝ} (tABC : triangle A B C) 

-- Conditions from the problem
def condition1 : Prop :=
  dist A M + dist M C + dist C A = dist C N + dist N A + dist A C

def condition2 : Prop :=
  dist A N + dist N B + dist B A = dist C M + dist M B + dist B C

-- Proving the triangle is isosceles
theorem triangle_is_isosceles (tABC : triangle A B C) 
    (h1 : condition1)
    (h2 : condition2) : isosceles tABC :=
sorry

end triangle_is_isosceles_l296_296596


namespace calculate_expression_l296_296046

theorem calculate_expression : (1100 * 1100) / ((260 * 260) - (240 * 240)) = 121 := by
  sorry

end calculate_expression_l296_296046


namespace part_a_part_b_l296_296731

theorem part_a (a b c d : ℤ) (h : a * d ≠ b * c) : 
  ∃ (r s : ℚ), (1 : ℚ) / ((a * x + b) * (c * x + d)) = r / (a * x + b) + s / (c * x + d) := 
  sorry

noncomputable def part_b_sum : ℚ := 
  ∑ n in (FiniteSet.range (665)), 1 / ((3 * n - 2) * (3 * n + 1))

theorem part_b : part_b_sum = 1995 / (3 * 1996) := 
  sorry

end part_a_part_b_l296_296731


namespace polynomial_factorization_l296_296334

noncomputable def poly_1 : Polynomial ℤ := (Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.C 1)
noncomputable def poly_2 : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X ^ 12 - Polynomial.C 1 * Polynomial.X ^ 11 +
  Polynomial.C 1 * Polynomial.X ^ 9 - Polynomial.C 1 * Polynomial.X ^ 8 +
  Polynomial.C 1 * Polynomial.X ^ 6 - Polynomial.C 1 * Polynomial.X ^ 4 +
  Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 1 * Polynomial.X + Polynomial.C 1
noncomputable def polynomial_expression : Polynomial ℤ := Polynomial.X ^ 15 + Polynomial.X ^ 10 + Polynomial.C 1

theorem polynomial_factorization : polynomial_expression = poly_1 * poly_2 :=
  by { sorry }

end polynomial_factorization_l296_296334


namespace difference_in_money_in_cents_l296_296226

theorem difference_in_money_in_cents (p : ℤ) (h₁ : ℤ) (h₂ : ℤ) 
  (h₁ : Linda_nickels = 7 * p - 2) (h₂ : Carol_nickels = 3 * p + 4) :
  5 * (Linda_nickels - Carol_nickels) = 20 * p - 30 := 
by sorry

end difference_in_money_in_cents_l296_296226


namespace num_valid_sets_l296_296664

-- Define the finite set of numbers
def finite_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

-- Define the condition for the set containing a 3 and summing to 18
def valid_set (s : Finset ℕ) : Prop := 
  s.card = 3 ∧ 3 ∈ s ∧ s.sum id = 18

-- Define the proof problem that exactly two sets (containing the element 3) have the specified property
theorem num_valid_sets : (finite_set.powerset.filter valid_set).card = 2 :=
  sorry

end num_valid_sets_l296_296664


namespace train_passing_time_l296_296148

theorem train_passing_time (L S: ℝ) (hL: L = 170) (hS: S = 68) : 
  (L / (S * 1000 / 3600) ≈ 8.99) :=
  sorry

end train_passing_time_l296_296148


namespace finite_graph_edge_contraction_l296_296359

open Classical

variable {V : Type} [Fintype V] (G G₀ Gₙ X : SimpleGraph V)
          (Gᵢ : Fin (n+1) → SimpleGraph V) (eᵢ : Fin n → Edge V) 
          (i : Fin n)

def edge_contraction (H : SimpleGraph V) (e : Edge V) : SimpleGraph V :=
  sorry  -- implement edge contraction definition

theorem finite_graph_edge_contraction :
  (¬ G.is_empty ∧ (∀ i < n, Gᵢ i / eᵢ i = Gᵢ (i + 1)) ∧ G₀ = G ∧ Gₙ ≃ X) ↔ (∃ G₀, G₀ = G ∧ ∃ Gₙ, Gₙ ≃ X ∧ ∃ Gᵢ eᵢ, ∀ i < n, Gᵢ (i + 1) = Gᵢ i / eᵢ i) := 
  by
    sorry

end finite_graph_edge_contraction_l296_296359


namespace num_integers_satisfy_inequality_l296_296949

theorem num_integers_satisfy_inequality : 
  ∃ n : ℕ, n = 5 ∧ {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.finite.card = n :=
sorry

end num_integers_satisfy_inequality_l296_296949


namespace conic_section_eccentricity_l296_296969

theorem conic_section_eccentricity (m : ℝ) 
  (h : m = Real.geomMean 2 8 ∨ m = - Real.geomMean 2 8) 
  (conic : ∀ (x y : ℝ), x^2 + (y^2 / m) = 1 ∨ x^2 - (y^2 / m) = 1) :
  ∃ e : ℝ, e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5 := 
sorry

end conic_section_eccentricity_l296_296969


namespace farthest_point_from_origin_l296_296713

def distance (p : ℝ × ℝ) : ℝ :=
  real.sqrt (p.1 ^ 2 + p.2 ^ 2)

theorem farthest_point_from_origin :
  let points := [(0, 7), (2, 1), (4, -3), (7, 0), (-2, -3), (5, 5)] in
  (5, 5) ∈ points →
  ∀ p ∈ points, distance (5, 5) ≥ distance p :=
by
  intro points h (p : ℝ × ℝ) hp
  sorry

end farthest_point_from_origin_l296_296713


namespace who_is_next_to_Boris_l296_296820

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l296_296820


namespace isosceles_triangle_perimeter_l296_296294

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6) (h2 : b = 13) 
  (triangle_inequality : b + b > a) : 
  (2 * b + a) = 32 := by
  sorry

end isosceles_triangle_perimeter_l296_296294


namespace solution_existence_l296_296325

def problem_statement : Prop :=
  ∃ x : ℝ, (0.38 * 80) - (0.12 * x) = 11.2 ∧ x = 160

theorem solution_existence : problem_statement :=
  sorry

end solution_existence_l296_296325


namespace max_product_eq_l296_296149

theorem max_product_eq (n k : ℕ) (n_1 n_2 ... n_k : ℕ) (h_sum : n_1 + n_2 + ... + n_k = n)
    (t : ℕ := n / k) (r : ℕ := n % k) (h_r_range : 0 ≤ r ∧ r < k) :
    (n_1 * n_2 * ... * n_k) ≤ (t + 1)^r * t^(k - r) := 
sorry

end max_product_eq_l296_296149


namespace who_next_to_boris_l296_296798

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l296_296798


namespace cross_section_properties_l296_296040

-- Step 1: Definitions of cube and intersections
structure Cube :=
(a : ℝ) -- side length of the cube

structure Plane (α : Type) :=
(perpendicular_to : α → α → Prop) -- A predicate indicating perpendicularity

-- Step 2: Given conditions in the problem
constant cube : Cube
constant plane : Plane (cube.a)

-- Definitions related to the plane intersection
def cross_section_perimeter (c : Cube) (p : Plane (c.a)) : ℝ := 3 * real.sqrt 2 * c.a
def cross_section_area_varies (c : Cube) (p : Plane (c.a)) : Prop := ∃ E F G, S ≠ (3 * real.sqrt 3 / 4 * c.a)

-- Final theorem statement
theorem cross_section_properties (c : Cube) (p : Plane (c.a)) :
  cross_section_perimeter c p = 3 * real.sqrt 2 * c.a ∧
  (∃ E F G, cross_section_area_varies c p) :=
sorry

end cross_section_properties_l296_296040


namespace water_in_bowl_is_14_l296_296351

variables (C : ℝ) -- Total capacity of the bowl in cups

-- Conditions
def initial_water : ℝ := C / 2 -- Initially half full
def added_water : ℝ := 4 -- 4 cups added
def final_water : ℝ := 0.7 * C -- Now 70% full

-- Statement to prove
theorem water_in_bowl_is_14 (h : initial_water C + added_water = final_water C) : final_water C = 14 := by
  sorry

end water_in_bowl_is_14_l296_296351


namespace find_removed_number_l296_296324

def list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def target_average : ℝ := 8.2

theorem find_removed_number (n : ℕ) (h : n ∈ list) :
  (list.sum - n) / (list.length - 1) = target_average -> n = 5 := by
  sorry

end find_removed_number_l296_296324


namespace problem_1_problem_2_l296_296926

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := (2 - Real.exp 1) * x

def F (x m : ℝ) (a : ℝ) : ℝ :=
  if x ≤ m then f x a else g x

theorem problem_1 (m : ℝ) (a : ℝ) (h : a = Real.exp 1) (hf : Set.range (F x m a) = Set.univ) :
  0 ≤ m ∧ m ≤ 1 / (Real.exp 1 - 2) := sorry

theorem problem_2 (a : ℝ) (x1 x2 : ℝ)
  (hx1_range : 0 ≤ x1 ∧ x1 ≤ 2)
  (hx2_range : 0 ≤ x2 ∧ x2 ≤ 2)
  (h_fx_eq : f x1 a = f x2 a)
  (h_diff : |x1 - x2| ≥ 1) :
  Real.exp 1 - 1 ≤ a ∧ a ≤ Real.exp 2 - Real.exp 1 := sorry

end problem_1_problem_2_l296_296926


namespace sum_largest_smallest_two_digit_224689_l296_296377

theorem sum_largest_smallest_two_digit_224689 : 
  let digits := [2, 4, 6, 8, 9] in
  ∃ (a b c d: ℕ), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ c ∈ digits ∧ d ∈ digits ∧ c ≠ d ∧ 
  (10 * a + b = 98) ∧ (10 * c + d = 24) ∧ (98 + 24 = 122) :=
by
  sorry

end sum_largest_smallest_two_digit_224689_l296_296377


namespace fixed_circle_isogonal_conjugate_l296_296186

variables {P Q S : Type} [IncidenceGeometry P Q S]

theorem fixed_circle_isogonal_conjugate 
  (A B C D E F P S Q : P)
  (hD : lies_on D (line_through B C))
  (hE : lies_on E (segment A C) ∧ distance B F = distance B D ∧ distance C D = distance C E)
  (hF : lies_on F (segment A B))
  (hS : cyclic (insert S (triangle_circumcircle A B C) ∧ cyclic (insert S (circle A E F)))
  (hP : lie_on P (line_through E F ∩ line_through B C))
  (hQ : cyclic (insert Q (circle P D S)) ∧ cyclic (insert Q (circle A E F))) :
  ∃ (Θ : P), isogonal_conjugate Q (triangle A B C) ∧ cyličal Θ Q A B C :=
begin
  sorry,
end

end fixed_circle_isogonal_conjugate_l296_296186


namespace segment_AX_length_2R_l296_296564

open EuclideanGeometry

variables {A B C G H X : Point}

// Defining circle ω with center A and radius R
def circle (A : Point) (R : ℝ) :=
  λ P : Point, dist A P = R

-- Given conditions
-- Points B, C, G, H are on the circumference of circle ω
-- G is on the extended B-median of triangle ABC
-- H is on the extension of the altitude from B of triangle ABC
-- Intersection of AC and GH is X

variables {ω : Circle} (R : ℝ) 
constant is_cyclic : ω A R → ω B R → ω C R → ω G R → ω H R → Prop 
constant B_median_extended : ∀ {D}, is_median A B C D → lies_on G (extension B D) 
constant B_altitude_extended : ∀ {I}, alt_line A B C I → lies_on H (extension B I)
constant intersection : ∀ {l1 l2}, line_through l1 l2 → line_through l3 l4 → Point

-- To Prove
theorem segment_AX_length_2R
  (circle ω A R : ω)
  (cyclic : is_cyclic A B C G H)
  (median : ∃ D, is_median A B C D ∧ lies_on G (extension B D))
  (altitude : ∃ I, alt_line A B C I ∧ lies_on H (extension B I))
  (intersect_AC_GH : intersection (line_through A C) (line_through G H) = X) :
  dist A X = 2 * R :=
sorry

end segment_AX_length_2R_l296_296564


namespace Sn_integers_for_all_l296_296565

noncomputable def S (a : ℝ) (n : ℤ) := a^n + a^(-n)

theorem Sn_integers_for_all (a : ℝ) (k : ℤ) (h₁ : a ≠ 0) (h₂ : S a k ∈ ℤ) (h₃ : S a (k + 1) ∈ ℤ) :
  ∀ n : ℤ, S a n ∈ ℤ :=
by
  sorry

end Sn_integers_for_all_l296_296565


namespace reduced_number_by_13_l296_296347

-- Definitions directly from the conditions
def original_number (m a n k : ℕ) := m + 10^k * a + 10^(k+1) * n
def replaced_number (m n k : ℕ) := m + 10^(k+1) * n

theorem reduced_number_by_13 (N m a n k : ℕ)
  (h1 : N = original_number m a n k)
  (h2 : N = 13 * replaced_number m n k)
  : ∃ (b : ℕ), b ∈ {1, 2, 3} ∧ N = 325 * b * 10^97 :=
sorry

end reduced_number_by_13_l296_296347


namespace floor_neg_seven_thirds_l296_296077

theorem floor_neg_seven_thirds : ⌊-7 / 3⌋ = -3 :=
sorry

end floor_neg_seven_thirds_l296_296077


namespace triangle_angle_A_triangle_bc_range_l296_296155

theorem triangle_angle_A (a b c A B C : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (ha : a = b * Real.sin C + c * Real.sin B)
  (hb : b = c * Real.sin A + a * Real.sin C)
  (hc : c = a * Real.sin B + b * Real.sin A)
  (h_eq : (Real.sqrt 3) * a * Real.sin C + a * Real.cos C = c + b)
  (h_angles_sum : A + B + C = π) :
    A = π/3 := -- π/3 radians equals 60 degrees
sorry

theorem triangle_bc_range (a b c : ℝ) (h : a = Real.sqrt 3) :
  Real.sqrt 3 < b + c ∧ b + c ≤ 2 * Real.sqrt 3 := 
sorry

end triangle_angle_A_triangle_bc_range_l296_296155


namespace shaded_region_area_l296_296545

theorem shaded_region_area 
    (A B C O E : Type) 
    (AC : LineSegment A C)
    (radius_O : ∀ P : Type, dist O P = 1) 
    (OB_perpendicular_AC : perpendicular OB AC) 
    (semi_circle_ABC : semi_circle A B C O) 
    (semi_circle_AEB : semi_circle A E B) :
    let area_shaded := area semi_circle_AEB - (area semi_circle_ABC) in
    area_shaded = 1 / 2 :=
by
  sorry

end shaded_region_area_l296_296545


namespace inequality_solution_l296_296716

theorem inequality_solution (x : ℚ) (hx : x = 3 ∨ x = 2 ∨ x = 1 ∨ x = 0) : 
  (1 / 3) - (x / 3) < -(1 / 2) → x = 3 :=
by
  sorry

end inequality_solution_l296_296716


namespace dividend_expression_l296_296400

theorem dividend_expression 
  (D d q r P : ℕ)
  (hq_square : ∃ k, q = k^2)
  (hd_expr1 : d = 3 * r + 2)
  (hd_expr2 : d = 5 * q)
  (hr_val : r = 6)
  (hD_expr : D = d * q + r)
  (hP_prime : Prime P)
  (hP_div_D : P ∣ D)
  (hP_factor : P = 2 ∨ P = 43) :
  D = 86 := 
sorry

end dividend_expression_l296_296400


namespace binomial_expansion_constant_and_x5_l296_296905

theorem binomial_expansion_constant_and_x5 (a : ℝ) : 
  (∃ T : ℝ, (ax^2 + (1 / sqrt x)) ^ 5 = T) ∧ 
  (∃ c : ℝ, c = 10) := 
by
  sorry

end binomial_expansion_constant_and_x5_l296_296905


namespace angle_B_value_area_triangle_l296_296165

-- Define the given conditions
variables (A B C a b c : ℝ)
variable (triangle_acute : ∀ x : ℝ, x = A ∨ x = B ∨ x = C → 0 < x ∧ x < π / 2)
variable (side_length_condition1 : sqrt 3 * a - 2 * b * sin A = 0)
variable (side_length_condition2 : a + c = 5)
variable (side_length_condition3 : a > c)
variable (side_length_condition4 : b = sqrt 7)

-- Prove the measure of angle B
theorem angle_B_value : B = π / 3 :=
by
  have h1 : 2 * b * sin A = sqrt 3 * a, from Eq.symm side_length_condition1
  have h2 : 2 * sin B * sin A = sqrt 3 * sin A, from side_length_condition1
  have h3 : sin B = sqrt 3 / 2, from ...
  sorry 

-- Prove the area of triangle ABC
theorem area_triangle : (1 / 2) * a * c * sin B = 3 * sqrt 3 / 2 :=
by
  have B_val : B = π / 3, from angle_B_value
  have h4 : b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * cos B_val, from ...
  have h5 : 7 = 25 - 2 * a * c - a * c, from ...
  have h6 : a * c = 6, from ...
  have h7 : (1 / 2) * a * c * sin (π / 3) = 3 * sqrt 3 / 2, from ...
  sorry

end angle_B_value_area_triangle_l296_296165


namespace right_isosceles_triangles_areas_l296_296247

variable (W X Y : ℝ)

theorem right_isosceles_triangles_areas :
  let W := (1 / 2) * 6 * 6,
      X := (1 / 2) * 8 * 8,
      Y := (1 / 2) * 10 * 10
  in W + X = Y :=
by
  sorry

end right_isosceles_triangles_areas_l296_296247


namespace cos_C_value_area_of_triangle_l296_296527

-- Defining the conditions
variables {A B C : ℝ} (cosA sinB BC : ℝ)
-- Assuming the given conditions
def condition1 : Prop := cosA = -5 / 13
def condition2 : Prop := sinB = 4 / 5
def condition3 : Prop := BC = 15

-- Statement to prove: the cosine of angle C
noncomputable def cosC := cos (A + B)
-- Statement to prove: the area of the triangle
noncomputable def area_ABC := 1/2 * BC * AC * sin C 

theorem cos_C_value : condition1 ∧ condition2 → cos C = 63 / 65 :=
sorry

theorem area_of_triangle : condition1 ∧ condition2 ∧ condition3 → area_ABC = 24 :=
sorry

end cos_C_value_area_of_triangle_l296_296527


namespace lines_parallel_or_coincide_l296_296220

-- Define a cyclic quadrilateral ABCD
variables (A B C D P Q R S X Y U V : Type*)
variables (h_cyclic : cyclic_quadrilateral A B C D) 
variables (h_not_parallel : ¬parallel (A, C) (B, D))
variables (h_P : lies_on_segment P A B)
variables (h_Q : lies_on_segment Q B C)
variables (h_R : lies_on_segment R C D)
variables (h_S : lies_on_segment S D A)

-- Define the specific angle conditions
variables (h_angle_PDA_PCB : angle_eq (angle PDA) (angle PCB))
variables (h_angle_QAB_QDC : angle_eq (angle QAB) (angle QDC))
variables (h_angle_RBC_RAD : angle_eq (angle RBC) (angle RAD))
variables (h_angle_SCD_SBA : angle_eq (angle SCD) (angle SBA))

-- Define intersections X and Y
variables (h_intersection_X : intersection_point X (line_through Q A) (line_through S B))
variables (h_intersection_Y : intersection_point Y (line_through Q D) (line_through S C))

-- Prop to prove: lines PR and XY are either parallel or coincide
theorem lines_parallel_or_coincide :
    lines_are_parallel_or_coincide (line_through P R) (line_through X Y) := 
sorry

end lines_parallel_or_coincide_l296_296220


namespace num_integers_satisfying_ineq_count_l296_296941

theorem num_integers_satisfying_ineq_count :
  {x : ℤ | -6 ≤ 3 * (x : ℤ) + 2 ∧ 3 * (x : ℤ) + 2 ≤ 9}.finite.to_finset.card = 5 :=
by
  sorry

end num_integers_satisfying_ineq_count_l296_296941


namespace total_compensation_women_l296_296002

theorem total_compensation_women (total_employees : ℕ) (total_women : ℕ) (women_agreed : total_women = 150) 
(total_comp : ℝ) (comp_per_woman : ℝ) (correct_calculation : comp_per_woman = 8.15)
(total_paid_to_women : ℝ) : 
total_employees = 350 → 
total_comp = 1222.50 :=
by 
  intro h1 
  have h2 : total_women = 350 - 200 := by sorry
  have h3 : total_comp = total_women * comp_per_woman := by sorry
  have h4 : total_comp = 150 * 8.15 := by sorry
  exact h4

end total_compensation_women_l296_296002


namespace integer_solutions_count_l296_296964

theorem integer_solutions_count :
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 :=
by sorry

end integer_solutions_count_l296_296964


namespace range_of_eccentricity_l296_296203

variables (a b : ℝ)
def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)

-- For the upper vertex B
def upper_vertex : ℝ × ℝ := (0, b)

-- Distance condition
def distance_condition (P : ℝ × ℝ) : Prop := 
  let (x, y) := P in 
  (x - 0)^2 + (y - b)^2 ≤ (2 * b)^2

-- Eccentricity
def eccentricity : ℝ := real.sqrt (a^2 - b^2) / a

theorem range_of_eccentricity 
  (h_ellipse : ∀ (x y : ℝ), ellipse_eq a b x y → distance_condition a b (x, y))
  : 0 < eccentricity a b ∧ eccentricity a b ≤ real.sqrt 2 / 2
:= sorry

end range_of_eccentricity_l296_296203


namespace pyramid_inscribed_sphere_radius_l296_296639

noncomputable def max_inscribed_sphere_radius_in_pyramid
  (AB CE EH_ : ℝ) (BC_eq_CE : BC = CE) (AH_plus_EH : AH + EH = Real.sqrt 2)
  (volume : volume_pyramid ABCDEH = 1 / 6) :
  ℝ :=
  (3 - Real.sqrt 5) / 4

theorem pyramid_inscribed_sphere_radius 
  (AB BC CE EH AH : ℝ)
  (h1 : AB = 1)
  (h2 : BC = CE)
  (h3 : AH + EH = Real.sqrt 2)
  (h4 : volume_pyramid ABCDEH = 1 / 6) :
  max_inscribed_sphere_radius_in_pyramid AB CE EH BC_eq_CE AH_plus_EH volume = (3 - Real.sqrt 5) / 4 :=
sorry

end pyramid_inscribed_sphere_radius_l296_296639


namespace num_real_solutions_of_exponential_eq_l296_296071

theorem num_real_solutions_of_exponential_eq :
  (∃! x : ℝ, 2 ^ (x^2 - 6 * x + 8) = 8) ∧ (∃! y : ℝ, 2 ^ (y^2 - 6 * y + 8) = 8) ∧ ∀ z : ℝ, 2 ^ (z^2 - 6 * z + 8) = 8 ↔ (z = 5 ∨ z = 1) :=
by sorry

end num_real_solutions_of_exponential_eq_l296_296071


namespace convex_polygon_sides_l296_296126

theorem convex_polygon_sides (n : ℕ) (h : ∀ angle, angle = 45 → angle * n = 360) : n = 8 :=
  sorry

end convex_polygon_sides_l296_296126


namespace weekend_visitors_l296_296858

theorem weekend_visitors (visitors_saturday : ℕ) (morning_increase afternoon_increase evening_increase : ℕ) :
  visitors_saturday = 200 →
  morning_increase = 20 →
  afternoon_increase = 30 →
  evening_increase = 50 →
  let visitors_morning := visitors_saturday / 3 * 120 / 100,
  let visitors_afternoon := visitors_saturday / 3 * 130 / 100,
  let visitors_evening := visitors_saturday / 3 * 150 / 100,
  let visitors_sunday := visitors_morning + visitors_afternoon + visitors_evening,
  visitors_saturday + visitors_sunday = 467 :=
by
  sorry

end weekend_visitors_l296_296858


namespace Kolya_Homework_Problem_l296_296562

-- Given conditions as definitions
def squaresToDigits (x : ℕ) (a b : ℕ) : Prop := x^2 = 10 * a + b
def doubledToDigits (x : ℕ) (a b : ℕ) : Prop := 2 * x = 10 * b + a

-- The main theorem statement
theorem Kolya_Homework_Problem :
  ∃ (x a b : ℕ), squaresToDigits x a b ∧ doubledToDigits x a b ∧ x = 9 ∧ x^2 = 81 :=
by
  -- proof skipped
  sorry

end Kolya_Homework_Problem_l296_296562


namespace find_integer_pairs_l296_296083

theorem find_integer_pairs :
  ∃ (x y : ℤ),
    (x, y) = (-7, -99) ∨ (x, y) = (-1, -9) ∨ (x, y) = (1, 5) ∨ (x, y) = (7, -97) ∧
    2 * x^3 + x * y - 7 = 0 :=
by
  sorry

end find_integer_pairs_l296_296083


namespace magnitude_eq_one_l296_296575

open Complex

theorem magnitude_eq_one (r : ℝ) (z : ℂ) (h1 : |r| < 2) (h2 : z + 1 / z = r) : |z| = 1 :=
sorry

end magnitude_eq_one_l296_296575


namespace computation_l296_296843

theorem computation :
  let x := 3 in 
  (x^3 + 4)^2 / (x^3 + 4) = 31 := by 
  let x := 3
  rw [pow_two, pow_succ, pow_zero]
  norm_num
  sorry

end computation_l296_296843


namespace time_to_school_building_l296_296587

theorem time_to_school_building 
  (total_time : ℕ := 30) 
  (time_to_gate : ℕ := 15) 
  (time_to_room : ℕ := 9)
  (remaining_time := total_time - time_to_gate - time_to_room) : 
  remaining_time = 6 :=
by
  sorry

end time_to_school_building_l296_296587


namespace map_float_time_l296_296075

theorem map_float_time
  (t₀ t₁ : Nat) -- times representing 12:00 PM and 12:21 PM in minutes since midnight
  (v_w v_b : ℝ) -- constant speed of water current and boat in still water
  (h₀ : t₀ = 12 * 60) -- t₀ is 12:00 PM
  (h₁ : t₁ = 12 * 60 + 21) -- t₁ is 12:21 PM
  : t₁ - t₀ = 21 := 
  sorry

end map_float_time_l296_296075


namespace num_integers_satisfy_l296_296939

theorem num_integers_satisfy : 
  ∃ n : ℕ, (n = 7 ∧ ∀ k : ℤ, (k > -5 ∧ k < 3) → (k = -4 ∨ k = -3 ∨ k = -2 ∨ k = -1 ∨ k = 0 ∨ k = 1 ∨ k = 2)) := 
sorry

end num_integers_satisfy_l296_296939


namespace who_next_to_boris_l296_296796

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l296_296796


namespace num_integers_satisfy_inequality_l296_296947

theorem num_integers_satisfy_inequality : 
  ∃ n : ℕ, n = 5 ∧ {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.finite.card = n :=
sorry

end num_integers_satisfy_inequality_l296_296947


namespace root_quadratic_eq_l296_296648

theorem root_quadratic_eq (a b c : ℝ) (h : a ≠ 0) (x₁ x₂ : ℝ)
  (h1 : a * x₁^2 + b * x₁ + c = 0) 
  (h2 : a * x₂^2 + b * x₂ + c = 0) 
  (h3 : x₁ < x₂) : 
  x₁ = - (b / (2 * a)) - sqrt ((b^2 - 4 * a * c) / (2 * |a|)) := sorry

end root_quadratic_eq_l296_296648


namespace parallel_lines_k_eq_one_l296_296906

def l1 (k : ℝ) : set (ℝ × ℝ) := {p | p.1 + (1 + k) * p.2 = 2 - k}
def l2 (k : ℝ) : set (ℝ × ℝ) := {p | k * p.1 + 2 * p.2 + 8 = 0}

theorem parallel_lines_k_eq_one (k : ℝ) :
  (∀ p₁ p₂, (p₁ ∈ l1 k ∧ p₂ ∈ l1 k) → (p₁ ∈ l2 k ∧ p₂ ∈ l2 k)) → k = 1 :=
sorry

end parallel_lines_k_eq_one_l296_296906


namespace votes_difference_l296_296536

theorem votes_difference (T : ℕ) (hT : T = 340) (hA : 40 * T = 136 * 100) : 
  let votes_against := 40 * T / 100,
      votes_in_favor := T - votes_against in
  votes_in_favor - votes_against = 68 :=
by
  rw [hT]
  have h_votes_against : votes_against = 136 :=
    by
      have : (40 * 340) / 100 = (40 * 340) / (10 * 10) := by rw [mul_comm 10, nat.mul_comm]
      exact (nat.div_eq_of_lt (nat.mul_lt_mul_right 340 (by norm_num : 10 > 1))) this
  have h_votes_in_favor : votes_in_favor = 204 := by simp [votes_in_favor, h_votes_against]
  rw [h_votes_in_favor, h_votes_against]
  norm_num
  done

end votes_difference_l296_296536


namespace coordinates_of_A_and_B_trajectory_of_Q_l296_296136

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x + 2

def f_prime (x : ℝ) : ℝ := -3 * x^2 + 3

def A : ℝ × ℝ := (-1, f (-1))
def B : ℝ × ℝ := (1, f (1))

theorem coordinates_of_A_and_B :
  A = (-1, 0) ∧ B = (1, 4) := by
  -- The proof for these coordinates is omitted
  sorry

lemma PA_dot_PB_eq_4 (m n : ℝ) :
  let PA := (m + 1, n)
  let PB := (m - 1, n - 4)
  PA.1 * PB.1 + PA.2 * PB.2 = 4 := by
  sorry

theorem trajectory_of_Q (m n x y : ℝ) :
  PA_dot_PB_eq_4 m n →
  let Q := (x, y)
  (x - 8)^2 + (y + 2)^2 = 9 := by
  -- The proof for this equation is omitted
  sorry

end coordinates_of_A_and_B_trajectory_of_Q_l296_296136


namespace who_is_next_to_boris_l296_296809

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l296_296809


namespace new_trash_cans_in_veterans_park_l296_296055

def initial_trash_cans_in_veterans_park : ℕ := 24

def trash_cans_in_central_park (V : ℕ) : ℕ := (V / 2) + 8

def moved_trash_cans (C : ℕ) : ℕ := C / 2

theorem new_trash_cans_in_veterans_park :
  let V := initial_trash_cans_in_veterans_park in
  let C := trash_cans_in_central_park V in
  let moved_cans := moved_trash_cans C in
  V + moved_cans = 34 :=
by
  let V := initial_trash_cans_in_veterans_park
  let C := trash_cans_in_central_park V
  let moved_cans := moved_trash_cans C
  have h1 : V = 24 := rfl
  have h2 : C = 20 := by
    calc
      C = (V / 2) + 8 : rfl
      _ = (24 / 2) + 8 : by rw [h1]
      _ = 12 + 8 : by norm_num
      _ = 20 : rfl
  have h3 : moved_cans = 10 := by
    calc
      moved_cans = C / 2 : rfl
      _ = 20 / 2 : by rw [h2]
      _ = 10 : by norm_num
  show V + moved_cans = 34 from
    calc
      V + moved_cans = 24 + 10 : by rw [h1, h3]
      _ = 34 : by norm_num

end new_trash_cans_in_veterans_park_l296_296055


namespace area_of_ABD_l296_296537

open Real
open EuclideanGeometry

noncomputable def vector_projection (v w : Vector ℝ) := (v ⬝ w) / (w ⬝ w) • w

theorem area_of_ABD (A B C D : Point ℝ) (hABC : IsEquilateralTriangle A B C)
  (hProj : vector_projection (B - A) (C - B) = -1)
  (hAD : 2 • (C - D) = A - D) :
  area (Triangle A B D) = (2 * sqrt 3) / 3 :=
begin
  sorry
end

end area_of_ABD_l296_296537


namespace correct_option_l296_296712

/- Define each of the four equations as conditions -/
def optionA : Prop := Real.sqrt ((-7)^2) = -7
def optionB : Prop := Real.sqrt 81 = 9 ∨ Real.sqrt 81 = -9
def optionC : Prop := -Real.cbrt (- (8 / 27)) = 2 / 3
def optionD : Prop := Real.sqrt (-9) = -3

/- State that option C is the correct one and prove the others are false -/
theorem correct_option :
  ¬optionA ∧ ¬optionB ∧ optionC ∧ ¬optionD :=
by sorry

end correct_option_l296_296712


namespace part1_part2_l296_296114

-- Definitions of the angles and sides of the triangle
variables {A B C a b c : ℝ}
variables {area : ℝ}

-- Given conditions for part (Ⅰ)
axiom cond1 : c = 2
axiom cond2 : C = real.pi / 3

-- Define the condition of the area for part (Ⅰ)
axiom cond_area_eq_sqrt3 : area = real.sqrt 3

-- First problem (Ⅰ): Prove that a = 2 and b = 2
theorem part1 (h₁ : cond1)
              (h₂ : cond2)
              (h₃ : cond_area_eq_sqrt3)
              (area_eq : area = 0.5 * a * b * real.sin C)
              : a = 2 ∧ b = 2 :=
sorry

-- Given conditions for part (Ⅱ)
axiom cond3 : real.sin C + real.sin(B - A) = 2 * real.sin (2 * A)

-- Second problem (Ⅱ): Prove that the area of the triangle is (2 * real.sqrt 3) / 3
theorem part2 (h₁ : cond1)
              (h₂ : cond2)
              (h₃ : cond3)
              : area = (2 * real.sqrt 3) / 3 :=
sorry

end part1_part2_l296_296114


namespace intercepted_segment_length_l296_296996

-- Definitions for the line and the circle equations
def line_eq (ρ θ : ℝ) : Prop := ρ * sin θ - ρ * cos θ = 1
def curve_eq (ρ : ℝ) : Prop := ρ = 1

theorem intercepted_segment_length :
  ∃ (ρ θ : ℝ), line_eq ρ θ ∧ curve_eq ρ ∧ segment_length = real.sqrt 2 :=
sorry

end intercepted_segment_length_l296_296996


namespace least_number_to_subtract_l296_296340

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (h1: n = 509) (h2 : d = 9): ∃ k : ℕ, k = 5 ∧ ∃ m : ℕ, n - k = d * m :=
by
  sorry

end least_number_to_subtract_l296_296340


namespace find_value_of_x2_plus_y2_l296_296219

theorem find_value_of_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + y^2 - 4 * x * y + 24 ≤ 10 * x - 1) : x^2 + y^2 = 125 := 
sorry

end find_value_of_x2_plus_y2_l296_296219


namespace johns_original_number_is_11_l296_296560

theorem johns_original_number_is_11 (x : ℕ) (hx1 : 10 ≤ x ∧ x < 100) 
  (hx2 : ∃ a b : ℕ, 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 5 * x - 7 = 10 * a + b ∧ 84 ≤ 10 * b + a ∧ 10 * b + a ≤ 90) :
  x = 11 :=
begin
  sorry
end

end johns_original_number_is_11_l296_296560


namespace length_of_conjugate_axis_l296_296463

theorem length_of_conjugate_axis (a b c : ℝ) 
    (h1 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
    (h2 : ∀ m n, (abs (m * b - n * a) / √(a^2 + b^2)) * (abs (m * b + n * a) / √(a^2 + b^2)) = 3 / 4)
    (h3 : c^2 / a^2 = 4) :
  b = sqrt 3 → 2 * b = 2 * sqrt 3 :=
by
  sorry

end length_of_conjugate_axis_l296_296463


namespace lena_found_numbers_l296_296034

def is_digit (d : ℕ) : Prop := d ∈ {1, 3, 7, 9}

def is_two_digit_prime (n: ℕ) : Prop :=
  n > 9 ∧ n < 100 ∧ Nat.Prime n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem lena_found_numbers :
  ∃ (a b : ℕ), 
  is_digit a ∧ is_digit b ∧
  let n1 := 10 * a + b in 
  let n2 := 10 * b + a in
  is_two_digit_prime n1 ∧
  is_two_digit_prime n2 ∧
  is_perfect_square (n1 - n2) ∧
  n1 - n2 = 36 :=
by
  sorry

end lena_found_numbers_l296_296034


namespace find_a_l296_296476

theorem find_a (a : ℝ) (α β : ℝ)
  (h1 : tan α = log10 (10 * a))
  (h2 : tan β = log10 a)
  (h3 : α - β = π / 4) :
  a = 1 ∨ a = 1 / 10 := 
sorry

end find_a_l296_296476


namespace centroid_triangle_inequality_l296_296214

variable {Point : Type} [MetricSpace Point] 
variable (A B C O : Point)

-- Defining the centroid and the condition on distances
def is_centroid (O : Point) (A B C : Point) : Prop := sorry

def s1 (O A B C : Point) : ℝ := dist O A + dist O B + dist O C
def s2 (A B C : Point) : ℝ := dist A B + dist B C + dist C A

theorem centroid_triangle_inequality 
  (h_centroid : is_centroid O A B C) :
  s1 O A B C < s2 A B C := 
begin
  sorry
end

end centroid_triangle_inequality_l296_296214


namespace man_age_twice_son_age_in_2_years_l296_296012

/-- Define the variables representing the ages of the man and his son. --/
def present_age_of_son : ℕ := 28
def age_difference : ℕ := 30 
def present_age_of_man : ℕ := present_age_of_son + age_difference

/-- Define the function that calculates the age of the man in X years. --/
def age_of_man_in_x_years (X : ℕ) : ℕ := present_age_of_man + X

/-- Define the function that calculates the age of the son in X years. --/
def age_of_son_in_x_years (X : ℕ) : ℕ := present_age_of_son + X

/-- The proof statement for the problem. --/
theorem man_age_twice_son_age_in_2_years :
  ∃ X : ℕ, age_of_man_in_x_years X = 2 * age_of_son_in_x_years X ∧ X = 2 :=
begin
  use 2,
  unfold age_of_man_in_x_years age_of_son_in_x_years present_age_of_man present_age_of_son age_difference,
  split,
  { rw add_comm, -- shows 58 + 2 = 2 * (28 + 2)
    norm_num,
  },
  { refl }
end

end man_age_twice_son_age_in_2_years_l296_296012


namespace car_can_leave_grid_and_crash_into_house_l296_296112

theorem car_can_leave_grid_and_crash_into_house
(grid : matrix (fin 102) (fin 102) bool)
(car_starts_central : car_starts_central (grid : matrix (fin 102) (fin 102) bool))
(move_only_north_or_west : ∀ x y, grid x y → (grid (x - 1) y ∨ grid x (y - 1))) : 
    ∃ (x : fin 102), (x = 0 ∨ ∃ (y : fin 102), (y = 0 ∨ ∃ (y : fin 102), grid x y = false)) :=
  sorry

end car_can_leave_grid_and_crash_into_house_l296_296112


namespace triangle_is_isosceles_l296_296595

variables {A B C M N : Type} [pseudo_metric_space A] [pseudo_metric_space B] [pseudo_metric_space C]
[pseudo_metric_space M] [pseudo_metric_space N] 

variables {dist : ∀ {X : Type} [pseudo_metric_space X], X → X → ℝ}
variables {a b c x y : ℝ} -- edge lengths

-- Define the points and their distances
variables {MA MB NA NC : ℝ} (tABC : triangle A B C) 

-- Conditions from the problem
def condition1 : Prop :=
  dist A M + dist M C + dist C A = dist C N + dist N A + dist A C

def condition2 : Prop :=
  dist A N + dist N B + dist B A = dist C M + dist M B + dist B C

-- Proving the triangle is isosceles
theorem triangle_is_isosceles (tABC : triangle A B C) 
    (h1 : condition1)
    (h2 : condition2) : isosceles tABC :=
sorry

end triangle_is_isosceles_l296_296595


namespace number_of_values_not_satisfied_l296_296450

-- Define the inequality condition for values of x
def inequality_not_satisfied (x : ℤ) : Prop :=
  3 * x^2 + 8 * x + 5 ≤ 10

-- Define a set of integers in the interval
def integers_in_interval (a b : ℤ) : set ℤ :=
  {x | a ≤ x ∧ x ≤ b}

-- Define the set of values of x where the inequality is not satisfied in the given interval
def values_not_satisfied : finset ℤ :=
  (finset.filter inequality_not_satisfied ⟨{-5, -4, -3, -2, -1, 0, 1}, sorry⟩).to_finset

-- The main theorem stating the number of integer solutions where the inequality is not satisfied
theorem number_of_values_not_satisfied : values_not_satisfied.card = 6 :=
by
  sorry

end number_of_values_not_satisfied_l296_296450


namespace shifted_parabola_sum_constants_l296_296709

theorem shifted_parabola_sum_constants :
  let a := 2
  let b := -17
  let c := 43
  a + b + c = 28 := sorry

end shifted_parabola_sum_constants_l296_296709


namespace integer_count_in_range_l296_296959

theorem integer_count_in_range (x : Int) : 
  (Set.count (Set.range (λ x, ( -6 ≤ 3*x + 2 ∧ 3*x + 2 ≤ 9))) 5) := 
by 
  sorry

end integer_count_in_range_l296_296959


namespace number_of_integers_satisfying_inequality_l296_296447

theorem number_of_integers_satisfying_inequality :
  { x : ℤ // 3 * x^2 + 8 * x + 5 ≤ 10 }.card = 5 :=
by
  sorry

end number_of_integers_satisfying_inequality_l296_296447


namespace costume_total_cost_l296_296008

variable (friends : ℕ) (cost_per_costume : ℕ) 

theorem costume_total_cost (h1 : friends = 8) (h2 : cost_per_costume = 5) : friends * cost_per_costume = 40 :=
by {
  sorry -- We omit the proof, as instructed.
}

end costume_total_cost_l296_296008


namespace hotel_assignment_ways_l296_296022

theorem hotel_assignment_ways :
  let rooms : ℕ := 4
  let friends : ℕ := 6
  let max_per_room : ℕ := 3
  (∀ room_assignments : list (list ℕ), room_assignments.length = rooms →
    (∀ room, room ∈ room_assignments → room.length ≤ max_per_room) →
    (∑ room in room_assignments, room.length = friends)) →
  ∃ num_ways : ℕ, num_ways = 1560 :=
by
  sorry

end hotel_assignment_ways_l296_296022


namespace james_used_5_containers_l296_296190

-- Conditions
def initial_balls : ℕ := 100
def balls_given_away : ℕ := initial_balls / 2
def remaining_balls : ℕ := initial_balls - balls_given_away
def balls_per_container : ℕ := 10

-- Question (statement of the theorem to prove)
theorem james_used_5_containers : (remaining_balls / balls_per_container) = 5 := by
  sorry

end james_used_5_containers_l296_296190


namespace rational_expression_simplify_l296_296394

theorem rational_expression_simplify (N : ℕ) : 
  (↑((N-1)! * N^2) / (↑(N+2)!)) = (N / (N+1)) := 
by 
  sorry

end rational_expression_simplify_l296_296394


namespace who_is_next_to_boris_l296_296811

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l296_296811


namespace deriv_at_3pi_over_4_l296_296917

def f (x : ℝ) : ℝ := sin x - 2 * cos x + 1

def f_deriv (x : ℝ) : ℝ := cos x + 2 * sin x

theorem deriv_at_3pi_over_4 : f_deriv (3 * π / 4) = sqrt 2 / 2 := by
  sorry

end deriv_at_3pi_over_4_l296_296917


namespace arrangement_count_l296_296035

open Finset

variable (A B C D E F : Type)

def arrangements : ℕ := 6.factorial

def valid_positions_of_ABC : Finset (Finset (A × B × C)) :=
  finset.univ.filter (λ s, (A, B, C) ∈ s ∧ (B, C) ∈ s)

def counted_arrangements : ℕ :=
  2 * arrangements / 3

theorem arrangement_count :
  counted_arrangements = 480 :=
by
  have total_arrangements: arrangements = 720 := by rw [Nat.factorial_six], sorry

  have valid_count : counted_arrangements = 480 := by
    rw [arrangements, valid_positions_of_ABC], sorry

  exact valid_count

end arrangement_count_l296_296035


namespace proposition_not_true_3_l296_296364

theorem proposition_not_true_3 (P : ℕ → Prop) (h1 : ∀ n, P n → P (n + 1)) (h2 : ¬ P 4) : ¬ P 3 :=
by
  sorry

end proposition_not_true_3_l296_296364


namespace optimal_play_final_payment_l296_296345

-- Define A and B's actions in terms of the removals and initial set.
def initialSet := {n | n ∈ (Finset.range 1025)}
def A_removal_step (step : ℕ) (s : Finset ℕ) : Finset ℕ :=
  s.filter (λ n, n % (2^step) = 0)
def B_removal_step (step : ℕ) (s : Finset ℕ) : Finset ℕ :=
  s.erase ((s.image (λ x, -x)).max' sorry)  -- B's strategy involves removing the k largest or smallest elements

-- Define the sequence of steps in the game.
def game_sequence (s : Finset ℕ) : Finset ℕ :=
  (List.range 10).foldl (λ acc step, if step % 2 = 0 then A_removal_step step acc else B_removal_step step acc)
  s

-- Final difference calculation
def final_difference (s : Finset ℕ) : ℕ :=
  (game_sequence s).max' sorry - (game_sequence s).min' sorry

-- The theorem statement
theorem optimal_play_final_payment : final_difference initialSet = 32 := 
by
-- Here would be the proof involving details of number removals and strategic plays
sorry

end optimal_play_final_payment_l296_296345


namespace standing_next_to_boris_l296_296832

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l296_296832


namespace persons_next_to_Boris_l296_296826

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l296_296826


namespace unique_root_l296_296407

def f (x : ℝ) : ℝ := 2 * x - sin x

theorem unique_root : ∃! x : ℝ, f x = 0 := sorry

end unique_root_l296_296407


namespace false_about_squares_l296_296715

theorem false_about_squares (squares_congruence_false : ¬ ∀ (sq1 sq2 : square), is_congruent sq1 sq2)
    (squares_all_convex : ∀ (sq : square), is_convex sq)
    (squares_equal_sides_angles : ∀ (sq : square), (∀ (a b : side sq), a = b) ∧ (∀ (a b: angle sq), a = b))
    (squares_regular_polygons : ∀ (sq : square), is_regular_polygon sq)
    (squares_similar : ∀ (sq1 sq2 : square), is_similar sq1 sq2) : 
  ¬ ∀ (sq1 sq2 : square), is_congruent sq1 sq2 := sorry

end false_about_squares_l296_296715


namespace find_constants_l296_296429

theorem find_constants
  (k m n : ℝ)
  (h : -x^3 + (k + 7) * x^2 + m * x - 8 = -(x - 2) * (x - 4) * (x - n)) :
  k = 7 ∧ m = 2 ∧ n = 1 :=
sorry

end find_constants_l296_296429


namespace vertex_angle_first_two_cones_l296_296666

theorem vertex_angle_first_two_cones (alpha beta : ℝ) 
    (h_beta : beta = arcsin (1 / 4))
    (h_half : 2 * alpha = (π / 3) + beta) :
    2 * alpha = π / 6 + beta :=
by
  sorry

end vertex_angle_first_two_cones_l296_296666


namespace isosceles_triangle_l296_296600

variables {A B C M N : Type*}

def is_triangle (A B C : Type*) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (X Y Z : Type*) : ℝ := -- Dummy function to represent perimeter

theorem isosceles_triangle
  {A B C M N : Type*}
  (hABC : is_triangle A B C)
  (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (hM_on_AB : M ∈ [A, B])
  (hN_on_BC : N ∈ [B, C])
  (h_perim_AMC_CNA : perimeter A M C = perimeter C N A)
  (h_perim_ANB_CMB : perimeter A N B = perimeter C M B) :
  (A = B) ∨ (B = C) ∨ (C = A) :=
by sorry

end isosceles_triangle_l296_296600


namespace trapezoid_area_l296_296994

variable (AD BC areaABE areaAED : ℝ)
variable (is_right_trapezoid : Prop)
variable (E_on_DC : Prop)

-- Conditions
def conditions : Prop := 
  AD = 2 ∧ BC = 6 ∧ areaABE = 15.6 ∧ areaAED = 4.8 ∧ is_right_trapezoid ∧ E_on_DC

-- Question to prove
def area_trapezoid (AD BC h : ℝ) := (AD + BC) * h / 2

theorem trapezoid_area (h : ℝ) : 
  conditions AD BC areaABE areaAED is_right_trapezoid E_on_DC → 
  h = 6 → 
  area_trapezoid AD BC h = 24 := 
by 
  intros _ h_eq
  rw [h_eq]
  unfold area_trapezoid
  sorry

end trapezoid_area_l296_296994


namespace find_s_l296_296976

axiom p q r s : Nat
axiom p_mul_q_r : 10 * p + q * r = 10 * s + p
axiom p_add_q_r_p : 10 * p + q + 10 * r + p = 10 * s + q

theorem find_s : s = 1 := 
by
  sorry

end find_s_l296_296976


namespace total_items_in_jar_l296_296299

/--
A jar contains 3409.0 pieces of candy and 145.0 secret eggs with a prize.
We aim to prove that the total number of items in the jar is 3554.0.
-/
theorem total_items_in_jar :
  let number_of_pieces_of_candy := 3409.0
  let number_of_secret_eggs := 145.0
  number_of_pieces_of_candy + number_of_secret_eggs = 3554.0 :=
by
  sorry

end total_items_in_jar_l296_296299


namespace number_of_correct_propositions_is_three_l296_296499

noncomputable def two_non_coincident_lines_and_planes
  (m n : Line) (alpha beta : Plane) : Prop :=
  ¬ (m = n) ∧ ¬ (alpha = beta)

def proposition_1 (m n : Line) (alpha : Plane) : Prop :=
  m ⟂ n ∧ m ⟂ alpha → n ∥ alpha

def proposition_2 (m n : Line) (alpha beta : Plane) : Prop :=
  m ⟂ alpha ∧ n ⟂ beta ∧ m ∥ n → alpha ∥ beta

def proposition_3 (m n : Line) (alpha beta : Plane) : Prop :=
  m.skew_with n ∧ m ⊆ alpha ∧ n ⊆ beta ∧ m ∥ beta ∧ n ∥ alpha → alpha ∥ beta

def proposition_4 (m n : Line) (alpha beta : Plane) : Prop :=
  alpha ⟂ beta ∧ (alpha ∩ beta = m) ∧ (n ⊆ beta) ∧ (n ⟂ m) → n ⟂ alpha

theorem number_of_correct_propositions_is_three
  (m n : Line) (alpha beta : Plane)
  (h : two_non_coincident_lines_and_planes m n alpha beta) :
  card ({proposition_1 m n alpha, proposition_2 m n alpha beta, proposition_3 m n alpha beta, proposition_4 m n alpha beta} . {Prop}) = 3 :=
sorry

end number_of_correct_propositions_is_three_l296_296499


namespace correct_statements_l296_296780

-- Definitions as conditions
def is_synthesis_cause_and_effect (s : Prop) : Prop := s
def is_synthesis_forward_reasoning (s : Prop) : Prop := s
def is_analysis_seeking_cause_from_effect (a : Prop) : Prop := a
def is_analysis_indirect_proof (a : Prop) : Prop := a
def is_contradiction_backward_reasoning (c : Prop) : Prop := c

-- Propositions as given in the problem
def proposition_1 : Prop := is_synthesis_cause_and_effect true
def proposition_2 : Prop := is_synthesis_forward_reasoning true
def proposition_3 : Prop := is_analysis_seeking_cause_from_effect true
def proposition_4 : Prop := is_analysis_indirect_proof false
def proposition_5 : Prop := is_contradiction_backward_reasoning false

-- The theorem to be proved
theorem correct_statements :
  proposition_1 ∧ proposition_2 ∧ proposition_3 ∧ ¬proposition_4 ∧ ¬proposition_5 :=
by
  split; sorry -- Proof here should demonstrate ①②③ are true while ④⑤ are false.

end correct_statements_l296_296780


namespace inequality_solution_sum_of_m_and_2n_l296_296222

-- Define the function f(x) = |x - a|
def f (x a : ℝ) : ℝ := abs (x - a)

-- Part (1): The inequality problem for a = 2
theorem inequality_solution (x : ℝ) :
  f x 2 ≥ 4 - abs (x - 1) → x ≤ 2 / 3 := sorry

-- Part (2): Given conditions with solution set [0, 2] and condition on m and n
theorem sum_of_m_and_2n (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : ∀ x, f x 1 ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) (h₄ : 1 / m + 1 / (2 * n) = 1) :
  m + 2 * n ≥ 4 := sorry

end inequality_solution_sum_of_m_and_2n_l296_296222


namespace developer_lots_l296_296753

theorem developer_lots (acres : ℕ) (cost_per_acre : ℕ) (lot_price : ℕ) 
  (h1 : acres = 4) 
  (h2 : cost_per_acre = 1863) 
  (h3 : lot_price = 828) : 
  ((acres * cost_per_acre) / lot_price) = 9 := 
  by
    sorry

end developer_lots_l296_296753


namespace sqrt_sum_of_cubes_l296_296686

theorem sqrt_sum_of_cubes :
  √(4^3 + 4^3 + 4^3 + 4^3) = 16 :=
by
  sorry

end sqrt_sum_of_cubes_l296_296686


namespace angle_between_AD_and_BC_l296_296185

theorem angle_between_AD_and_BC (A B C D : Point) 
  (h1 : dist A B = dist A C)
  (h2 : ∠ D A B = ∠ D A C) : 
  ∠ (line_through A D) (line_through B C) = 90 :=
by
  sorry

end angle_between_AD_and_BC_l296_296185


namespace abc_sum_seven_l296_296633

theorem abc_sum_seven (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 7 :=
sorry

end abc_sum_seven_l296_296633


namespace cassini_ovals_conclusions_correct_l296_296033

theorem cassini_ovals_conclusions_correct :
  let C := { P : ℝ × ℝ | let d1 := (P.1 + 1)^2 + P.2^2, d2 := (P.1 - 1)^2 + P.2^2 in d1 * d2 = 4 } in
  let conclusions := 
    [(∀ P ∈ C, ∃ Q ∈ C, P.1 = -Q.1 ∧ P.2 = Q.2),   -- axisymmetric
     (∀ P ∈ C, ∃ Q ∈ C, P.1 = -Q.1 ∧ P.2 = -Q.2),  -- centrally symmetric
     (∀ P ∈ C, -(3 : ℝ).sqrt ≤ P.1 ∧ P.1 ≤ (3 : ℝ).sqrt), -- range of abscissa
     (∀ P ∈ C, 1 ≤ (P.1^2 + P.2^2).sqrt ≤ (3 : ℝ).sqrt), -- range of |OP|
     ∃ P ∈ C, ((1/2) * abs((P.1 + 1) * (P.1 - 1) * P.2)) = 1] in  {- maximum area of tr.}
  conclusions.count(true) = 4 :=
by
  sorry

end cassini_ovals_conclusions_correct_l296_296033


namespace sufficient_but_not_necessary_l296_296900

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1 → x^2 > 1) ∧ (¬ (x^2 > 1 → x > 1)) :=
by {
  split,
  { -- Proof of sufficient condition
    intro h,
    exact lt_of_lt_of_le h (show 1 ≤ x^2, from pow_two_ge_one (one_le_abs_of_not_lt_zero h)),
    sorry
  },
  { -- Proof of not necessary condition
    intro h,
    have : ¬((-1 < x) → (x^2 < 1)) := sorry,
    sorry
  }
}

end sufficient_but_not_necessary_l296_296900


namespace correct_overestimation_l296_296410

theorem correct_overestimation (y : ℕ) : 
  25 * y + 4 * y = 29 * y := 
by 
  sorry

end correct_overestimation_l296_296410


namespace perpendicular_lines_a_l296_296308

theorem perpendicular_lines_a (a : ℝ) :
  let l1 := λ (x y : ℝ), ax + (1 + a) * y = 3,
      l2 := λ (x y : ℝ), (a + 1) * x + (3 - 2 * a) * y = 2,
      slope1 := -a / (1 + a),
      slope2 := -(a + 1) / (3 - 2 * a) in
  slope1 * slope2 = -1 ↔ a = -1 ∨ a = 3 := 
sorry

end perpendicular_lines_a_l296_296308


namespace area_of_triangles_l296_296172

-- Definitions for the problem conditions
variables (BD DC : ℝ) 

def ratio_BD_DC (BD DC : ℝ) : Prop := BD/DC = 2/5
def area_triangle_ABD : ℝ := 28
def area_triangle_ADC (area_triangle_ABD : ℝ) : ℝ := (5/2) * area_triangle_ABD
def total_area_triangle_ABC (area_triangle_ABD area_triangle_ADC : ℝ) : ℝ := area_triangle_ABD + area_triangle_ADC

-- The Lean 4 statement for the proof problem
theorem area_of_triangles (BD DC : ℝ) (h_ratio : ratio_BD_DC BD DC) :
  area_triangle_ADC area_triangle_ABD = 70 ∧
  total_area_triangle_ABC area_triangle_ABD (area_triangle_ADC area_triangle_ABD) = 98 :=
by 
  -- Proof omitted
  sorry

end area_of_triangles_l296_296172


namespace sample_size_n_l296_296030

theorem sample_size_n (n : ℕ)
  (h_population : 36 = 6 + 12 + 18)
  (h_systematic_sampling : (36 / n) ∈ ℕ)
  (h_stratified_sampling : 
    (6 * n / 36) ∈ ℕ ∧
    (12 * n / 36) ∈ ℕ ∧
    (18 * n / 36) ∈ ℕ)
  (h_increase_sample_exclude : (35 / (n + 1)) ∈ ℕ) :
  n = 6 := 
sorry

end sample_size_n_l296_296030


namespace last_integer_in_sequence_l296_296290

-- Let the sequence be defined as follows:
def seq (a : ℕ) : ℕ → ℕ
| 0 := a
| (n + 1) := seq n / 3

theorem last_integer_in_sequence (a : ℕ) (h : a = 1234567) :
  ∃ n : ℕ, seq a n = 137174 ∧ ∀ m : ℕ, n < m → ¬(is_integer (seq a m / 3)) :=
sorry

end last_integer_in_sequence_l296_296290


namespace angle_C_measure_l296_296979

-- We define angles and the specific conditions given in the problem.
def measure_angle_A : ℝ := 80
def external_angle_C : ℝ := 100

theorem angle_C_measure :
  ∃ (C : ℝ) (A B : ℝ), (A + B = measure_angle_A) ∧
                       (C + external_angle_C = 180) ∧
                       (external_angle_C = measure_angle_A) →
                       C = 100 :=
by {
  -- skipping proof
  sorry
}

end angle_C_measure_l296_296979


namespace maximum_and_minimum_values_of_f_exists_b_for_minimum_value_3_l296_296134

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - log x

theorem maximum_and_minimum_values_of_f :
  (∀ x ∈ Icc (1 / 2) 2, f (-1) 3 x ≤ f (-1) 3 1) ∧
  (∀ x ∈ Icc (1 / 2) 2, f (-1) 3 x ≥ f (-1) 3 (1 / 2)) :=
begin 
  sorry
end

theorem exists_b_for_minimum_value_3 :
  ∃ (b : ℝ), (0 < b) ∧
  (∀ x ∈ Icc 0 e, f 0 b x ≥ f 0 b (1 / b)) ∧
  (f 0 b (1 / b) = 3) ∧ (b = exp 2) :=
begin
  sorry
end

end maximum_and_minimum_values_of_f_exists_b_for_minimum_value_3_l296_296134


namespace cosine_dihedral_angle_result_l296_296310

noncomputable def cosine_dihedral_angle (R : ℝ) : ℝ :=
  let r := R / 2
  let distance_between_centers := 3 * R / 2
  let θ := 30 * Real.pi / 180
  let α := acos ((R / distance_between_centers)) / θ -- this is from interpreting in 3D
  1 / 9  -- result from derived conditions

theorem cosine_dihedral_angle_result (R : ℝ) (α : ℝ) (h1 : r = R / 2) (h2 : distance_between_centers = 3 * R / 2) (h3: θ = 30 * Real.pi / 180) : 
  cosine_dihedral_angle R = 1 / 9 :=
sorry

end cosine_dihedral_angle_result_l296_296310


namespace difference_in_amount_paid_l296_296752

variable (P Q : ℝ)

def original_price := P
def intended_quantity := Q

def new_price := P * 1.10
def new_quantity := Q * 0.80

theorem difference_in_amount_paid :
  ((new_price P * new_quantity Q) - (original_price P * intended_quantity Q)) = -0.12 * (original_price P * intended_quantity Q) :=
by
  sorry

end difference_in_amount_paid_l296_296752


namespace find_angle_C_find_side_c_l296_296098

variables {A B C : ℝ} {a b c : ℝ} {S : ℝ}

def triangle_ABC : Prop := 
triangle ∧ side_opposite_angle A = a ∧ side_opposite_angle B = b ∧ 
side_opposite_angle C = c ∧ a + b = sqrt 3 * c ∧ 
2 * (sin C) ^ 2 = 3 * sin A * sin B

theorem find_angle_C (h : triangle_ABC) : C = π / 3 :=
sorry

theorem find_side_c (h : triangle_ABC) (h_area : (1 / 2) * a * b * sin C = sqrt 3) : c = sqrt 6 :=
sorry

end find_angle_C_find_side_c_l296_296098


namespace new_weighted_average_l296_296296

def original_weighted_average (marks weights : List ℝ) : ℝ :=
  (List.sum (List.map₂ (· * ·) marks weights)) / (List.sum weights)

theorem new_weighted_average
  (marks : List ℝ)
  (weights : List ℝ)
  (h_len : marks.length = 20)
  (h_len_weights : weights.length = 20)
  (h_weight_values : weights = [1.2, 1.6, 1.8, 2.3, 2.6, 1.4, 1.7, 1.5, 1.9, 2.1, 1.1, 2.8, 2.4, 1.3, 1.8, 2.9, 2.5, 1.6, 1.9, 2.2])
  (h_weighted_avg : original_weighted_average marks weights = 36) :
  original_weighted_average (List.map (2 * ·) marks) weights = 72 :=
by
  sorry

end new_weighted_average_l296_296296


namespace new_bottles_from_recycling_l296_296632

theorem new_bottles_from_recycling (initial_bottles : ℕ) (recycle_ratio : ℕ) (final_count : ℕ) 
  (h1 : initial_bottles = 625)
  (h2 : recycle_ratio = 5)
  (h3 : final_count = 195) : 
  (∑ i in (range 4), (recycle_ratio ^ (3-i))) = final_count := 
by
  sorry

end new_bottles_from_recycling_l296_296632


namespace min_time_required_for_flashes_l296_296668

theorem min_time_required_for_flashes 
  (num_lights : Nat) 
  (colors : Fin num_lights → Color) 
  (flash_duration : Nat) 
  (interval_duration : Nat) 
  (total_flashes : Nat)
  (unique_flash_sequences : Fin num_lights!)
  (min_time : Nat)
  : min_time = 1195 :=
by
  have h1 : num_lights = 5 := by rfl
  have h2 : flash_duration = 5 := by rfl
  have h3 : interval_duration = 5 := by rfl
  have h4 : total_flashes = 120 := by rfl
  have h5 : unique_flash_sequences = 120 := by rfl
  sorry

end min_time_required_for_flashes_l296_296668


namespace evaluate_f_at_3_l296_296922

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -2 * x

theorem evaluate_f_at_3 : f 3 = -6 := by
  -- since we are only writing the statement, the proof is omitted
  sorry

end evaluate_f_at_3_l296_296922


namespace diameter_of_well_l296_296744

theorem diameter_of_well (h : 14) 
  (V : 43.982297150257104) : 
  diameter = 2 :=
by
  sorry

end diameter_of_well_l296_296744


namespace friends_contribution_l296_296873

theorem friends_contribution :
  ∃ (a b c d e : ℝ), 
    a = 0.5 * (b + c + d + e) ∧
    b = 1/3 * (a + c + d + e) ∧
    c = 1/4 * (a + b + d + e) ∧
    d = 1/5 * (a + b + c + e) ∧
    a + b + c + d + e = 120 ∧
    e ≈ 52.55 := sorry

end friends_contribution_l296_296873


namespace imaginary_part_of_z_l296_296909

theorem imaginary_part_of_z (z : ℂ) (h : z * complex.I = 1 + 2 * complex.I) : complex.im z = -1 :=
sorry

end imaginary_part_of_z_l296_296909


namespace congruence_solutions_count_l296_296123

theorem congruence_solutions_count : 
  ∃ n, number_of_solutions (λ x : ℕ, x + 17 ≡ 63 [MOD 29]) (λ x, x < 100) = n ∧ n = 3 :=
by
  sorry

end congruence_solutions_count_l296_296123


namespace range_of_a_l296_296271

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x : ℝ, 4 ≤ x → (if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a) ≥ 4) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l296_296271


namespace count_f100_eq_0_values_l296_296210

noncomputable def f₀ (x : ℝ) : ℝ :=
  x + abs (x - 100) - abs (x + 100)

noncomputable def f : ℕ → ℝ → ℝ
| 0, x := f₀ x
| (n + 1), x := abs (f n x) - 1

theorem count_f100_eq_0_values : 
  ∃ s : Finset ℝ, s.card = 301 ∧ ∀ x ∈ s, f 100 x = 0 :=
sorry

end count_f100_eq_0_values_l296_296210


namespace boris_neighbors_l296_296789

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l296_296789


namespace angles_AQD_BQC_equal_l296_296670

variables {P Q A B C D : Type*} [geometry P Q A B C D]

-- Given the conditions
axiom circles_intersect_at_PQ (circle1 circle2 : set P) (P Q : P) :
  P ∈ circle1 ∧ P ∈ circle2 ∧ Q ∈ circle1 ∧ Q ∈ circle2

axiom third_circle_centered_at_P_intersects_1_and_2
  (center : P) (circle3 : set P) (centre_P : center = P)
  (points1_2 : list P) :
  (A ∈ points1_2 ∧ B ∈ points1_2 ∧ C ∈ points1_2 ∧ D ∈ points1_2 ∧
   A ∈ circle3 ∧ B ∈ circle3 ∧ C ∈ circle3 ∧ D ∈ circle3)

-- The proof problem statement
theorem angles_AQD_BQC_equal (circle1 circle2 circle3 : set P) :
  ∀ (P Q A B C D : P),
  circles_intersect_at_PQ circle1 circle2 P Q →
  third_circle_centered_at_P_intersects_1_and_2 P circle3 A B C D →
  ∀ PQ1 PQ2 PQ3 PQ4,
  PQ1 ∈ circle1 → PQ2 ∈ circle1 → PQ1 = P → PQ2 = Q →
  PQ3 ∈ circle2 → PQ4 ∈ circle2 → PQ3 = P → PQ4 = Q →
  (AQD = BQC) :=
by
  sorry

end angles_AQD_BQC_equal_l296_296670


namespace cheese_cost_l296_296557

def bread_cost : ℝ := 0.15
def ham_cost : ℝ := 0.25
def sandwich_cost : ℝ := 0.90

theorem cheese_cost : ∃ c : ℝ, c = 0.50 ∧ sandwich_cost = bread_cost + ham_cost + c :=
by
  use 0.50
  split
  . exact rfl
  . sorry

end cheese_cost_l296_296557


namespace number_of_selection_plans_l296_296374

-- Definitions based on conditions
def male_students : Nat := 5
def female_students : Nat := 4
def total_volunteers : Nat := 3

def choose (n k : Nat) : Nat :=
  Nat.choose n k

def arrangement_count : Nat :=
  Nat.factorial total_volunteers

-- Theorem that states the total number of selection plans
theorem number_of_selection_plans :
  (choose male_students 2 * choose female_students 1 + choose male_students 1 * choose female_students 2) * arrangement_count = 420 :=
by
  sorry

end number_of_selection_plans_l296_296374


namespace gcd_n_cube_plus_25_n_plus_3_l296_296875

theorem gcd_n_cube_plus_25_n_plus_3 (n : ℕ) (h : n > 3^2) : 
  Int.gcd (n^3 + 25) (n + 3) = if n % 2 = 1 then 2 else 1 :=
by
  sorry

end gcd_n_cube_plus_25_n_plus_3_l296_296875


namespace eccentricity_range_l296_296201

noncomputable def ellipse_eccentricity {a b : ℝ} (ha : a > b) (hb : b > 0) 
(C : x² / a² + y² / b² = 1) : set (ℝ) :=
{e : ℝ | 0 < e ∧ e ≤ Real.sqrt 2 / 2}

theorem eccentricity_range (a b : ℝ) (ha : a > b) (hb : b > 0)
(C : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
(hPB : ∀ x y, x^2 + (y - b)^2 ≤ (2 * b)^2) :
  ∃ e, e ∈ ellipse_eccentricity ha hb C :=
sorry

end eccentricity_range_l296_296201


namespace num_integers_satisfying_inequality_l296_296953

theorem num_integers_satisfying_inequality : 
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 := 
by 
  sorry

end num_integers_satisfying_inequality_l296_296953


namespace min_value_sum_l296_296901

def positive_real (x : ℝ) : Prop := x > 0

theorem min_value_sum (x y : ℝ) (hx : positive_real x) (hy : positive_real y)
  (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : x + y ≥ 20 :=
sorry

end min_value_sum_l296_296901


namespace nicholas_paid_more_than_kenneth_l296_296588

def price_per_yard : ℝ := 40
def kenneth_yards : ℝ := 700
def nicholas_multiplier : ℝ := 6
def discount_rate : ℝ := 0.15

def kenneth_total_cost : ℝ := price_per_yard * kenneth_yards
def nicholas_yards : ℝ := nicholas_multiplier * kenneth_yards
def nicholas_original_cost : ℝ := price_per_yard * nicholas_yards
def discount_amount : ℝ := discount_rate * nicholas_original_cost
def nicholas_discounted_cost : ℝ := nicholas_original_cost - discount_amount
def difference_in_cost : ℝ := nicholas_discounted_cost - kenneth_total_cost

theorem nicholas_paid_more_than_kenneth :
  difference_in_cost = 114800 := by
  sorry

end nicholas_paid_more_than_kenneth_l296_296588


namespace sqrt_sum_of_cubes_l296_296688

theorem sqrt_sum_of_cubes :
  √(4^3 + 4^3 + 4^3 + 4^3) = 16 :=
by
  sorry

end sqrt_sum_of_cubes_l296_296688


namespace boy_swims_downstream_distance_l296_296353

noncomputable def downstream_distance (v : ℝ) (t : ℝ) := (8 + v) * t

theorem boy_swims_downstream_distance :
  let v : ℝ := 5 in
  let t : ℝ := 7 in
  let upstream_distance : ℝ := 21 in
  upstream_distance = (8 - v) * t → downstream_distance v t = 91 :=
begin
  sorry
end

end boy_swims_downstream_distance_l296_296353


namespace geom_seq_problem_l296_296462

-- Define a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a(n+1) = r * a(n)

-- Definitions from the problem
variables {a : ℕ → ℝ}  -- a is the geometric sequence
variable h_geom : geom_seq a

-- Conditions
variable h1 : a 3 + a 5 = 8
variable h2 : a 1 * a 5 = 4

-- Problem to prove
theorem geom_seq_problem : (a 13 / a 9) = 9 :=
by
  sorry

end geom_seq_problem_l296_296462


namespace coefficient_x5_in_expansion_of_3x_plus_2_power_7_l296_296318

theorem coefficient_x5_in_expansion_of_3x_plus_2_power_7 :
  (∑ k in Fin₇.finset, (7.choose k) * (3 * x) ^ (7 - k) * 2 ^ k) = 20412 * x^5 :=
by sorry

end coefficient_x5_in_expansion_of_3x_plus_2_power_7_l296_296318


namespace percentage_seats_not_taken_l296_296381

theorem percentage_seats_not_taken
  (rows : ℕ) (seats_per_row : ℕ) 
  (ticket_price : ℕ)
  (earnings : ℕ)
  (H_rows : rows = 150)
  (H_seats_per_row : seats_per_row = 10) 
  (H_ticket_price : ticket_price = 10)
  (H_earnings : earnings = 12000) :
  (1500 - (12000 / 10)) / 1500 * 100 = 20 := 
by
  sorry

end percentage_seats_not_taken_l296_296381


namespace no_positive_rational_solutions_l296_296579

theorem no_positive_rational_solutions (n : ℕ) (h_pos_n : 0 < n) : 
  ¬ ∃ (x y : ℚ) (h_x_pos : 0 < x) (h_y_pos : 0 < y), x + y + (1/x) + (1/y) = 3 * n :=
by
  sorry

end no_positive_rational_solutions_l296_296579


namespace angle_C_in_rectangle_is_90_l296_296541

-- Define rectangle and the problem conditions
variables {A B C D : Type} [rect : Rectangle A B C D]

-- Define the Lean version of the problem
theorem angle_C_in_rectangle_is_90 (h : is_rectangle A B C D) : measure (angle C) = 90 :=
by
  -- Normally, the proof would go here, but we are using sorry for now
  sorry

end angle_C_in_rectangle_is_90_l296_296541


namespace saturn_moon_approximation_l296_296592

theorem saturn_moon_approximation : (1.2 * 10^5) * 10 = 1.2 * 10^6 := 
by sorry

end saturn_moon_approximation_l296_296592


namespace rectangle_length_from_square_thread_l296_296769

theorem rectangle_length_from_square_thread (side_of_square width_of_rectangle : ℝ) (same_thread : Bool) 
  (h1 : side_of_square = 20) (h2 : width_of_rectangle = 14) (h3 : same_thread) : 
  ∃ length_of_rectangle : ℝ, length_of_rectangle = 26 := 
by
  sorry

end rectangle_length_from_square_thread_l296_296769


namespace sum_of_g12_l296_296403

def g (n : ℕ) : ℕ := sorry

axiom g_condition_1 : g 2 = 4
axiom g_condition_2 (m n : ℕ) (h : m ≥ n) : 
  g (m + n) + g (m - n) = (g (3 * m) + g (3 * n)) / 3

theorem sum_of_g12 :
  ∑ k in finset.range 1, g 12 = 288 := sorry

end sum_of_g12_l296_296403


namespace ice_formation_l296_296301

-- Definitions for problem setup
variable {T_in T_out T_critical : ℝ}
variable T_inside_surface : ℝ -- The temperature on the inside surface of the trolleybus windows

-- Given conditions
def is_humid (T_in : ℝ) : Prop := true -- 100% humidity condition (always true for this problem)
def condensation_point (T : ℝ) : Prop := T < 0 -- Temperature below which condensation occurs

-- Condition that the inner surface temperature is determined by both inside and outside temperatures
def inner_surface_temp (T_in T_out : ℝ) : ℝ := (T_in + T_out) / 2  -- Simplified model

theorem ice_formation 
  (h_humid : is_humid T_in)
  (h_T_in : T_in < T_critical)
  : condensation_point (inner_surface_temp T_in T_out) :=
sorry

end ice_formation_l296_296301


namespace avg_people_per_hour_rounding_l296_296177

theorem avg_people_per_hour_rounding :
  let people := 3500
  let days := 5
  let hours := days * 24
  (people / hours : ℚ).round = 29 := 
by
  sorry

end avg_people_per_hour_rounding_l296_296177


namespace distance_XY_l296_296640

variable (A B C D E X Y : Point)
variable (AB CD AD BC : Line)
variable [geometry : Geometry E X Y]

def trapezoid_condition (AB CD : Line) :=
  parallel AB CD

def base_lengths (AB CD : Line) :=
  length AB = 2 ∧ length CD = 3

def diagonal_intersection (E : Point) :=
  intersection (line_through A C) (line_through B D) = E

def line_through_E (X Y : Point) :=
  parallel (line_through X Y) AB ∧ parallel (line_through X Y) CD

theorem distance_XY (hexagon : trapezoid_condition AB CD)
  (hlen : base_lengths AB CD)
  (hintersection : diagonal_intersection E)
  (hline : line_through_E X Y) :
  distance X Y = 2.6 := 
sorry

end distance_XY_l296_296640


namespace star_calculation_l296_296446

def operation_star (a c : ℝ) (h : a ≠ c) : ℝ := (a + c) / (a - c)

theorem star_calculation : operation_star (operation_star 3 5 (by norm_num)) 6 (by norm_num) = -1 / 5 := 
by
  sorry

end star_calculation_l296_296446


namespace person_next_to_Boris_arkady_galya_l296_296800

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l296_296800


namespace AC_length_is_12_l296_296241

noncomputable def length_of_AC (A B C D E : Point) (segment_length : ℝ) (AC BC : segment_type) (AB BE : ℝ) 
  (AD EC : segment_type) (BD ED : segment_type) 
  (angle_BDC_DEB : angle_type) : ℝ :=
if (AD = EC) ∧ (BD = ED) ∧ (angle_BDC = angle_DEB) ∧ (AB = 7) ∧ (BE = 2) 
then 12 else sorry

theorem AC_length_is_12 (A B C D E : Point) (segment_length : ℝ) 
  (AC BC : segment_type) (AB : BE : ℝ) 
  (AD EC : segment_type) (BD ED : segment_type) 
  (angle_BDC angle_DEB : angle_type) 
  (h1 : AD = EC) 
  (h2 : BD = ED) 
  (h3 : angle_BDC = angle_DEB) 
  (h4 : AB = 7) 
  (h5 : BE = 2) :
  length_of_AC A B C D E segment_length AC BC AB AD EC BD ED angle_BDC angle_DEB = 12 := 
by
  sorry

end AC_length_is_12_l296_296241


namespace det_scaled_matrix_l296_296882

variable (a b c d : ℝ)
variable (h : Matrix.det ![![a, b], ![c, d]] = 5)

theorem det_scaled_matrix : Matrix.det ![![3 * a, 3 * b], ![4 * c, 4 * d]] = 60 := by
  sorry

end det_scaled_matrix_l296_296882


namespace closest_point_l296_296435

noncomputable def point_on_line_closest_to (P : ℝ × ℝ × ℝ) (A : ℝ × ℝ × ℝ) (D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let t := (D.1 * (A.1 - P.1) + D.2 * (A.2 - P.2) + D.3 * (A.3 - P.3)) / (D.1^2 + D.2^2 + D.3^2)
  (A.1 + t * D.1, A.2 + t * D.2, A.3 + t * D.3)

theorem closest_point : point_on_line_closest_to (1, 2, -1) (3, 1, -2) (-3, 7, -4) = (111/74, 173/74, -170/37) :=
by
  -- Proof omitted
  sorry

end closest_point_l296_296435


namespace problem_1_problem_2_l296_296471

noncomputable theory
open_locale real

-- Define the ellipse
def ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop := (a > 0 ∧ b > 0 ∧ a > b) ∧ (p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1)

-- Given conditions
def A : ℝ × ℝ := (0, 1)
def F1 (a b : ℝ) : ℝ × ℝ := (-sqrt (a ^ 2 - b ^ 2), 0)
def F2 (a b : ℝ) : ℝ × ℝ := (sqrt (a ^ 2 - b ^ 2), 0)
def is_arithmetic_sequence (c1 c2 c3 : ℝ) : Prop := 2 * c2 = c1 + c3

-- Define the problem
theorem problem_1 (a b : ℝ) (h_ellipse : ellipse a b A) (h_cond : is_arithmetic_sequence (sqrt 3 * dist (B, F1 a b)) (dist (F1 a b, F2 a b)) (sqrt 3 * dist (B, F2 a b)))
  : a = 2 ∧ b = 1 :=
sorry

theorem problem_2 (k : ℝ) (a b : ℝ) (h_ellipse : ellipse a b A) (h_slope : ∀ P Q : ℝ × ℝ, ∃ (x1 x2 y1 y2 : ℝ), P = (x1, y1) ∧ Q = (x2, y2) ∧ 
  (y1 = k * (x1 + 2) ∧ x1 ^ 2 / a ^ 2 + y1 ^ 2 / b ^ 2 = 1) ∧ 
  (A.1 - x1)^2 + (A.2 - y1)^2 > (P.1 - Q.1)^2 + (P.2 - Q.2)^2) :
  k < -3 / 10 ∨ k > 1 / 2 :=
sorry

end problem_1_problem_2_l296_296471


namespace intersection_M_P_l296_296224

def M : Set ℕ := { x | -1 < x ∧ x < 4 }
def P : Set ℝ := { x | Real.log x / Real.log 2 < 1 }

theorem intersection_M_P :
  { x : ℝ | x ∈ M ∧ x ∈ P } = {1} :=
sorry

end intersection_M_P_l296_296224


namespace find_AD_correct_l296_296196

noncomputable def find_AD (AB : ℝ) (DC : ℝ) (M : ℝ) (x : ℝ) : ℝ :=
  if (M = DC / 2) ∧ (AB = 9) ∧ (DC = 12) ∧ (AC ^ 2 = BM ^ 2 + AM ^ 2) then x else 0

theorem find_AD_correct (AB DC : ℕ) (M x : ℝ) :
  let A := 90;
  let D := 90;
  let M := DC / 2;
  let AB := 9;
  let DC := 12 
  in AC ⊥ BM →
  BM ^ 2 + (AD ^ 2 + 36) = AD ^ 2 + 144 →
  BM ^ 2 + 36 = 144 →
  BM ^ 2 = 108 →
  2 * AD ^ 2 = 72 →
  AD ^ 2 = 36 →
  find_AD 9 12 (12 / 2) 6 = 6
:= by
  assume A D (M : DC / 2)
  assume AB 9
  assume DC 12
  assume AC BM
  have BM_eq: BM = 6, from sorry,
  have AD_eq: AD = 6, from sorry,
  exact find_AD 9 12 6 6

end find_AD_correct_l296_296196


namespace angle_bisectors_of_triangle_l296_296497

variables (a b c : ℝ)
variables (ABC : Triangle a b c)

theorem angle_bisectors_of_triangle
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a + b > c)
  (h5 : b + c > a)
  (h6 : c + a > b) :
  (angle_bisector_length a b) = a * c / (a + b) ∧
  (angle_bisector_length b c) = b * c / (a + b) ∧
  (bisector_square_length a b c) = a * b * (1 - c ^ 2 / (a + b) ^ 2) :=
sorry

end angle_bisectors_of_triangle_l296_296497


namespace isosceles_triangle_l296_296607

variables {A B C M N : ℝ}

-- Define the lengths of the sides of the triangle and segments
variables (a b c x y : ℝ)

-- Conditions based on the problem statement
axiom triangle_ABC (ABC_triangle: triangle):
  (point_on_side_A_B M)
  (point_on_side_B_C N)
  (perimeter_eq_1: x + distance M C + b = y + (a - y) + b)
  (perimeter_eq_2: x + y + c = (a - x) + (c - x) + a)

-- Prove that triangle ABC is isosceles
theorem isosceles_triangle (ABC_triangle): isosceles_triangle ABC_triangle := sorry

end isosceles_triangle_l296_296607


namespace exercise_books_count_l296_296164

variable (num_pencils : ℕ)
variable (ratio_pencils ratio_exercise_books : ℕ)
variable (num_exercise_books : ℕ)

-- The ratio of pencils to exercise books is 14:3
def ratio_cond := ratio_pencils = 14 ∧ ratio_exercise_books = 3

-- There are 140 pencils
def pencils_cond := num_pencils = 140

-- Calculate number of exercise books
def exercise_books_cond := num_exercise_books = (num_pencils / ratio_pencils) * ratio_exercise_books

theorem exercise_books_count
  (h1 : ratio_cond) 
  (h2 : pencils_cond) 
  : exercise_books_cond :=
  by
    -- proof goes here
    sorry

end exercise_books_count_l296_296164


namespace triangle_area_given_inradius_l296_296467

theorem triangle_area_given_inradius
  (A B C : ℝ) (p r : ℝ)
  (hA : A = 40) 
  (hB : B = 60)
  (hP : p = 40) 
  (hR : r = 2.5) 
  (hSumABC : A + B + C = 180) : 
  let s := p / 2 in 
  let T := r * s in 
  T = 50 :=
by
  -- sorry, proof is omitted.
  sorry

end triangle_area_given_inradius_l296_296467


namespace y_increase_12_units_l296_296590

theorem y_increase_12_units (x y : ℝ) (h : ∀ Δx, Δx = 4 → y + Δy = y + 6) : 
  x + 12 → y + 18 :=
by sorry

end y_increase_12_units_l296_296590


namespace projection_zero_l296_296097

variables (a b c : ℝ × ℝ)

/-- The vectors a, b, and c, where c = 4a + b -/
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (2, -2)
def vec_c : ℝ × ℝ := (4 * vec_a + vec_b)

theorem projection_zero :
  let dot_product := (vec_b.1 * vec_c.1 + vec_b.2 * vec_c.2) in
  (dot_product * vec_a = (0, 0)) :=
by
  sorry

end projection_zero_l296_296097


namespace sum_primitive_roots_mod_11_l296_296682

def is_primitive_root_mod (a n : ℕ) : Prop :=
  let residues := (List.range n).map (λ k, (a ^ (k + 1)) % n)
  residues.sortWith (≤) = List.range n

def sum_of_primitive_roots_mod_n (s : List ℕ) (n : ℕ) : ℕ :=
  s.filter (λ a, is_primitive_root_mod a n).sum

theorem sum_primitive_roots_mod_11 :
  sum_of_primitive_roots_mod_n [1, 2, 3, 4, 5, 6, 7, 8, 9] 11 = 8 :=
sorry

end sum_primitive_roots_mod_11_l296_296682


namespace xy_condition_l296_296576

theorem xy_condition (x y : ℝ) (h : x * y + x / y + y / x = -3) : (x - 2) * (y - 2) = 3 :=
sorry

end xy_condition_l296_296576


namespace find_m_l296_296110

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    let C : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ x^2 / 3 + y^2 = 1 }
    let F1 : ℝ × ℝ := (-√2, 0)
    let F2 : ℝ × ℝ := (√2, 0)
    let line : Set (ℝ × ℝ) := { p | ∃ x, p = (x, x + m) } in
    A ∈ C ∧ B ∈ C ∧ A ∈ line ∧ B ∈ line ∧
    ∃ (area : ℝ), 
    (λ area_F1_AB, (1 / 2) * abs ((F1.1) * (A.2 - B.2) + (A.1) * (B.2 - F1.2) + (B.1) * (F1.2 - A.2)) = area_F1_AB) area ∧
    (λ area_F2_AB, (1 / 2) * abs ((F2.1) * (A.2 - B.2) + (A.1) * (B.2 - F2.2) + (B.1) * (F2.2 - A.2)) = 2 * area) area)
    → m = -√2 / 3 := sorry

end find_m_l296_296110


namespace who_is_next_to_Boris_l296_296817

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l296_296817


namespace isosceles_triangle_l296_296599

variables {A B C M N : Type*}

def is_triangle (A B C : Type*) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (X Y Z : Type*) : ℝ := -- Dummy function to represent perimeter

theorem isosceles_triangle
  {A B C M N : Type*}
  (hABC : is_triangle A B C)
  (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (hM_on_AB : M ∈ [A, B])
  (hN_on_BC : N ∈ [B, C])
  (h_perim_AMC_CNA : perimeter A M C = perimeter C N A)
  (h_perim_ANB_CMB : perimeter A N B = perimeter C M B) :
  (A = B) ∨ (B = C) ∨ (C = A) :=
by sorry

end isosceles_triangle_l296_296599


namespace tangent_line_eq_l296_296432

section
variables {x y : ℝ}
def parabola_eq : ℝ := (1 / 5) * x^2

def point_A := (2 : ℝ, 4 / 5 : ℝ)

theorem tangent_line_eq :
  ∃ m b, ∀ x y : ℝ, (point_A.2 = parabola_eq) → 
  (∀ (x : ℝ) (y : ℝ), y = m * x + b) ∧ 
  (4 : ℝ) * x - (5 : ℝ) * y - (4 : ℝ) = 0 :=
sorry
end

end tangent_line_eq_l296_296432


namespace find_f_x_range_of_f_on_interval_range_of_m_l296_296284

namespace QuadraticFunction
noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

lemma f_x_plus_1_minus_f_x (x : ℝ) : f (x + 1) - f x = 2 * x :=
by
  -- Express the given condition
  calc f (x + 1) - f x 
  = ((x + 1) ^ 2 - (x + 1) + 1) - (x^2 - x + 1) : by sorry
  -- Simplify it to obtain 2 * x
  = by sorry

lemma f_zero_eq_one : f 0 = 1 :=
by
  -- Given condition f(0) = 1
  calc f 0 = 0^2 - 0 + 1 : by sorry
  ... = 1 : by sorry

theorem find_f_x : f = λ x, x^2 - x + 1 :=
by
  apply funext
  intro x
  -- From the above lemmas and calculations, we can state the function
  exact sorry

theorem range_of_f_on_interval : set.range (λ x, f x) (set.Icc (-1 : ℝ) (1 : ℝ)) = set.Icc (3/4) 3 :=
by
  -- Derive the range of the function on the given interval
  have min_value : ∃ c ∈ set.Icc (-1 : ℝ) (1 : ℝ), ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), c ≤ f x := by sorry
  have max_value : ∃ c ∈ set.Icc (-1 : ℝ) (1 : ℝ), ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x ≤ c := by sorry
  exact sorry

theorem range_of_m (x : ℝ) (h : x ∈ set.Icc (-1 : ℝ) (1 : ℝ)) : f x > 2 * x + m :=
by
  -- Derive the range for m such that f(x) is always above 2x + m on [-1, 1].
  have inequality_condition : m < -1 := by sorry
  exact sorry
end QuadraticFunction

end find_f_x_range_of_f_on_interval_range_of_m_l296_296284


namespace flour_more_than_sugar_l296_296585

/-
  Mary is baking a cake. The recipe calls for 6 cups of sugar and 9 cups of flour. 
  She already put in 2 cups of flour. 
  Prove that the number of additional cups of flour Mary needs is 1 more than the number of additional cups of sugar she needs.
-/

theorem flour_more_than_sugar (s f a : ℕ) (h_s : s = 6) (h_f : f = 9) (h_a : a = 2) :
  (f - a) - s = 1 :=
by
  sorry

end flour_more_than_sugar_l296_296585


namespace standing_next_to_boris_l296_296833

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l296_296833


namespace point_in_fourth_quadrant_coords_l296_296542

theorem point_in_fourth_quadrant_coords 
  (P : ℝ × ℝ)
  (h1 : P.2 < 0)
  (h2 : abs P.2 = 2)
  (h3 : P.1 > 0)
  (h4 : abs P.1 = 5) :
  P = (5, -2) :=
sorry

end point_in_fourth_quadrant_coords_l296_296542


namespace num_integers_satisfying_ineq_count_l296_296942

theorem num_integers_satisfying_ineq_count :
  {x : ℤ | -6 ≤ 3 * (x : ℤ) + 2 ∧ 3 * (x : ℤ) + 2 ≤ 9}.finite.to_finset.card = 5 :=
by
  sorry

end num_integers_satisfying_ineq_count_l296_296942


namespace number_divided_by_five_is_same_as_three_added_l296_296705

theorem number_divided_by_five_is_same_as_three_added :
  ∃ x : ℚ, x / 5 = x + 3 ∧ x = -15 / 4 :=
by
  sorry

end number_divided_by_five_is_same_as_three_added_l296_296705


namespace part_a_part_b_l296_296132

-- Part (a)
theorem part_a {x y n : ℕ} (h : x^3 + 2^n * y = y^3 + 2^n * x) : x = y :=
sorry

-- Part (b)
theorem part_b {x y : ℤ} {n : ℕ} (h : x ≠ 0 ∧ y ≠ 0 ∧ x^3 + 2^n * y = y^3 + 2^n * x) : |x| = |y| :=
sorry

end part_a_part_b_l296_296132


namespace smallest_prime_factor_in_setB_l296_296249

def setB : Set ℕ := {55, 57, 58, 59, 61}

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 2 then 2 else (Nat.minFac (Nat.pred n)).succ

theorem smallest_prime_factor_in_setB :
  ∃ n ∈ setB, smallest_prime_factor n = 2 := by
  sorry

end smallest_prime_factor_in_setB_l296_296249


namespace max_value_g_l296_296059

-- Defining the conditions and goal as functions and properties
def condition_1 (f : ℕ → ℕ) : Prop :=
  (Finset.range 43).sum f ≤ 2022

def condition_2 (f g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a >= b → g (a + b) ≤ f a + f b

-- Defining the main theorem to establish the maximum value
theorem max_value_g (f g : ℕ → ℕ) (h1 : condition_1 f) (h2 : condition_2 f g) :
  (Finset.range 85).sum g ≤ 7615 :=
sorry


end max_value_g_l296_296059


namespace students_both_l296_296983

noncomputable def total_students := 60
noncomputable def students_french := 41
noncomputable def students_german := 22
noncomputable def students_neither := 6

theorem students_both (total_students students_french students_german students_neither : ℕ) :
  total_students = 60 → 
  students_french = 41 → 
  students_german = 22 → 
  students_neither = 6 →
  (total_students - students_neither) - (students_french + students_german - (total_students - students_neither)) = 9 :=
by
  intros h_total h_french h_german h_neither
  have h1 : total_students - students_neither = 54 := by rw [h_total, h_neither]
  have h2 : students_french + students_german = 63 := by rw [h_french, h_german]
  have h_both : 63 - 54 = 9 := by norm_num
  rw [←h1, ←h2]
  exact h_both

end students_both_l296_296983


namespace waste_percentage_of_rectangle_is_48_percent_l296_296458

theorem waste_percentage_of_rectangle_is_48_percent :
  (let 
    w := 11;
    h := 7;
    a := real.sqrt 10;
    area_rect := w * h;
    area_squares := 4 * (a * a);
    waste_area := area_rect - area_squares;
    percentage_waste := (waste_area / area_rect) * 100
   in percentage_waste ≈ 48) :=
sorry

end waste_percentage_of_rectangle_is_48_percent_l296_296458


namespace find_z_value_l296_296103

noncomputable def is_solution (z : ℂ) : Prop :=
(z - 1) * complex.I = complex.abs (complex.I + 1)

theorem find_z_value (z : ℂ) (h : is_solution z) : z = 1 - real.sqrt(2) * complex.I :=
by sorry

end find_z_value_l296_296103


namespace find_x0_symmetric_curve_l296_296911

theorem find_x0_symmetric_curve :
  ∃ x0 ∈ Icc (0 : ℝ) (Real.pi / 2),
    ∀ x, (sin (2 * x) + sqrt 3 * cos (2 * x)) = (sin (2 * (2 * x0 - x)) + sqrt 3 * cos (2 * (2 * x0 - x))) → x0 = Real.pi / 3 :=
by
  sorry

end find_x0_symmetric_curve_l296_296911


namespace avg_move_to_california_l296_296179

noncomputable def avg_people_per_hour (total_people : ℕ) (total_days : ℕ) : ℕ :=
  let total_hours := total_days * 24
  let avg_per_hour := total_people / total_hours
  let remainder := total_people % total_hours
  if remainder * 2 < total_hours then avg_per_hour else avg_per_hour + 1

theorem avg_move_to_california : avg_people_per_hour 3500 5 = 29 := by
  sorry

end avg_move_to_california_l296_296179


namespace isosceles_triangle_l296_296604

variables {A B C M N : ℝ}

-- Define the lengths of the sides of the triangle and segments
variables (a b c x y : ℝ)

-- Conditions based on the problem statement
axiom triangle_ABC (ABC_triangle: triangle):
  (point_on_side_A_B M)
  (point_on_side_B_C N)
  (perimeter_eq_1: x + distance M C + b = y + (a - y) + b)
  (perimeter_eq_2: x + y + c = (a - x) + (c - x) + a)

-- Prove that triangle ABC is isosceles
theorem isosceles_triangle (ABC_triangle): isosceles_triangle ABC_triangle := sorry

end isosceles_triangle_l296_296604


namespace shortest_chord_length_l296_296620

/-- The shortest chord passing through point D given the conditions provided. -/
theorem shortest_chord_length
  (O : Point) (D : Point) (r : ℝ) (OD : ℝ)
  (h_or : r = 5) (h_od : OD = 3) :
  ∃ (AB : ℝ), AB = 8 := 
  sorry

end shortest_chord_length_l296_296620


namespace proof_problem_l296_296331

theorem proof_problem :
  (var_x20 = 4 ∧ (X : ℝ → ℝ → ℝ) ∧ P_X_gt_1 = 0.68 ∧ 
   corr_coeff_r ∈ [-1, 1] ∧ ∀ A B : Prop, P_A_gt_0 ∧ P_B_gt_0 ∧ P_cond_B_A = P_B) →
  ( ∀ y1 y2 : ℝ, (var (λ x1 x2, 2 * x1 + 1) y1 y2 ≠ 9) ∧
    ( ∀ x : ℝ, (2 ≤ x ∧ x < 3) → P (X 2 σ²) x = 0.18) ∧
    ( ∀ r_val : ℝ, (|r_val| = 1 → (corr_coeff_r = r_val) ∧ stronger_linear_correlation r_val)) ∧
    ( ∀ (A B : Prop), (P (A | B) = P (A)) →
      (P (A ∧ B) = P (A) * P (B ∧ P (B) = P (B))) → 
      P (A | B) = P (A)))
:= 
by
  sorry

end proof_problem_l296_296331


namespace who_is_next_to_Boris_l296_296815

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l296_296815


namespace option_a_correct_option_b_incorrect_option_c_incorrect_option_d_correct_l296_296121

variable {a b c A B C : ℝ}
variable {O : EuclideanSpace ℝ (Fin 2)}
variable {OB OC OA : EuclideanSpace ℝ (Fin 2)}

-- Condition 1
def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop :=
  A + B + C = π

-- Condition 2
axiom sine_greater (A B : ℝ) (h : sin A > sin B) : A > B

-- Condition 3
noncomputable def equilateral_triangle_side_length_one (a : ℝ) : Prop :=
  a = 1

-- Condition 4
def given_b_c (B : ℝ) (b c : ℝ) : Prop :=
  B = π / 6 ∧ b = real.sqrt 2 ∧ c = 2

-- Condition 5
def vector_equality (OB OC OA : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (OB - OC).norm = (OB + OC - 2 * OA).norm

-- Proof problem statements
theorem option_a_correct (A B : ℝ) (h1 : sin A > sin B) : A > B := sorry

theorem option_b_incorrect (a : ℝ) (h1 : equilateral_triangle_side_length_one a) : 
  let AB := ((1 : ℝ) - (0 : ℝ), (real.sqrt 3 / 2 : ℝ) - (0 : ℝ));
      BC := ((1 : ℝ) - (1 : ℝ), (real.sqrt 3 / 2 : ℝ) - (-real.sqrt 3 / 2 : ℝ)) in
  ((AB.1 * BC.1) + (AB.2 * BC.2)) ≠ (real.sqrt 3 / 2) := sorry

theorem option_c_incorrect (B : ℝ) (b c : ℝ) (h : given_b_c B b c) : 
  ¬ (exists A : ℝ, triangle_sides b c (2 * (sqrt (1 - (b^2 / (4 * c^2)))) *= sin B ∧ A + B + (acos (b / c)) = π) := sorry

theorem option_d_correct (OB OC OA : EuclideanSpace ℝ (Fin 2)) (h1 : vector_equality OB OC OA) : 
  let vecA := (OC - OA) in 
  let vecB := (OB - OA) in 
  inner_product vecA vecB = 0 := sorry

end option_a_correct_option_b_incorrect_option_c_incorrect_option_d_correct_l296_296121


namespace choose_two_grades_l296_296369

theorem choose_two_grades (grades museums : ℕ) (choose_two : ℕ → ℕ → ℕ) (pow : ℕ → ℕ → ℕ) :
  grades = 6 →
  museums = 6 →
  choose_two grades 2 * pow (museums - 1) 4 = 
  (choose_two 6 2 * pow 5 4) := 
by 
  intros h1 h2
  simp [h1, h2]
  sorry

end choose_two_grades_l296_296369


namespace problem_highschool_aff_nanjing_l296_296563

open Set

noncomputable def size (s : Finset ℕ) : ℕ := s.card

theorem problem_highschool_aff_nanjing (A : Finset ℕ) (hA : ∀ a ∈ A, 0 < a) :
  size (Finset.image (λ p : ℕ × ℕ × ℕ × ℕ, (p.1.1 + p.1.2) / (p.2.1 + p.2.2))
    ((A.product A).product (A.product A))) ≥ 2 * size A * size A - 1 :=
sorry

end problem_highschool_aff_nanjing_l296_296563


namespace evaluate_zeroes_in_decimal_expansion_l296_296314

def is_valid_step (x y : ℕ) : Prop :=
  ∨ (x', y'), x' = x + 1 ∧ y' = y + 1 ∨ x' = x + 1 ∧ y' = y - 1

def never_below_x_axis (path : list (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ) (p ∈ path), y ≥ 0

def valid_path (path : list (ℕ × ℕ)) : Prop :=
  ∀ (idx : ℕ) (p p' ∈ path), idx < path.length ∧ is_valid_step (fst p) (snd p) ∧
  never_below_x_axis path ∧
    ∃ (x_fb y_fb : ℕ), nth path 0 = (0, 0) ∧ nth path (length path - 1) = (2 * n, 0)

def phi (T : list (ℕ × ℕ)) : ℕ :=
  (prod (map_with_index (λ i step, match step with
                                  | (x, y) => if y > snd T[i-1] then 1 else y
                                  end) T))

def M (n : ℕ) : finset (list (ℕ × ℕ)) :=
  finset.filter valid_path (finset.powerset (list.product (finset.range (2*n+1)) (finset.range (2*n+1))))

def f (n : ℕ) : ℕ :=
  finset.sum (M n) phi

theorem evaluate_zeroes_in_decimal_expansion (n : ℕ) :
  zeroes_in_decimal_expansion (f 2021) = 0 :=
sorry

end evaluate_zeroes_in_decimal_expansion_l296_296314


namespace solve_quadratic_l296_296251

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) := by
  intro x
  sorry

end solve_quadratic_l296_296251


namespace sqrt_of_sum_of_powers_l296_296697

theorem sqrt_of_sum_of_powers :
  sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 :=
sorry

end sqrt_of_sum_of_powers_l296_296697


namespace cheese_reduction_l296_296191

theorem cheese_reduction :
  ∀ (salt_teaspoons : ℕ) 
    (cheese_oz : ℕ) 
    (salt_sodium_per_teaspoon : ℕ) 
    (cheese_sodium_per_oz : ℕ), 
    salt_teaspoons = 2 →
    cheese_oz = 8 →
    salt_sodium_per_teaspoon = 50 →
    cheese_sodium_per_oz = 25 →
    (let total_sodium := salt_teaspoons * salt_sodium_per_teaspoon + cheese_oz * cheese_sodium_per_oz in
     let sodium_reduction := total_sodium / 3 in
     let sodium_per_less_cheese := cheese_sodium_per_oz in
     sodium_reduction / sodium_per_less_cheese = 4) :=
by 
  intros; 
  let total_sodium := argument1 * argument2 + argument3 * argument4 in
  let sodium_reduction := total_sodium / 3 in
  let sodium_per_less_cheese := argument4 in
  sorry

end cheese_reduction_l296_296191


namespace runner_time_second_half_l296_296765

variable (v : ℝ) (t1 : ℝ) (t2 : ℝ)

noncomputable def distance_first_half : ℝ := 20
noncomputable def distance_second_half : ℝ := 20

noncomputable def speed_after_injury : ℝ := v / 2

noncomputable def time_first_half : ℝ := distance_first_half / v

noncomputable def time_second_half : ℝ := distance_second_half / speed_after_injury

noncomputable def equation_relation : Prop := time_second_half = time_first_half + 11

theorem runner_time_second_half : 
  ∃ (t2 : ℝ), equation_relation t2 ∧ t2 = 22 := 
sorry

end runner_time_second_half_l296_296765


namespace alice_bob_not_both_l296_296880

-- Define the group of 8 students
def total_students : ℕ := 8

-- Define the committee size
def committee_size : ℕ := 5

-- Calculate the total number of unrestricted committees
def total_committees : ℕ := Nat.choose total_students committee_size

-- Calculate the number of committees where both Alice and Bob are included
def alice_bob_committees : ℕ := Nat.choose (total_students - 2) (committee_size - 2)

-- Calculate the number of committees where Alice and Bob are not both included
def not_both_alice_bob : ℕ := total_committees - alice_bob_committees

-- Now state the theorem we want to prove
theorem alice_bob_not_both : not_both_alice_bob = 36 :=
by
  sorry

end alice_bob_not_both_l296_296880


namespace who_is_next_to_Boris_l296_296818

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l296_296818


namespace constant_term_expansion_is_60_l296_296100

noncomputable def n : ℝ := (6 / Real.pi) * ∫ x in -1..1, (Real.sqrt (1 - x^2) - 2 * x)

-- Statement to prove: the constant term in the expansion of x * (1 - 2 / Real.sqrt x)^n is 60
theorem constant_term_expansion_is_60 :
  ∫ x in -1..1, (Real.sqrt (1 - x^2) - 2 * x) = Real.pi →
  let n : ℝ := 6 in
  let expr := fun x => x * (1 - 2 / Real.sqrt x)^n in
  expr(1) = 60 := sorry

end constant_term_expansion_is_60_l296_296100


namespace range_of_g_l296_296320

noncomputable def g (x : ℝ) : ℝ := 1 / x ^ 2 + 3

theorem range_of_g : set.image g {x : ℝ | x ≠ 0} = set.Ici 3 := by
  sorry

end range_of_g_l296_296320


namespace arithmetic_sequence_sum_l296_296699

theorem arithmetic_sequence_sum (d : ℕ) (y : ℕ) (x : ℕ) (h_y : y = 39) (h_d : d = 6) 
  (h_x : x = y - d) : 
  x + y = 72 := by 
  sorry

end arithmetic_sequence_sum_l296_296699


namespace courtyard_width_is_16_l296_296749

-- Define the conditions
def courtyard_length : ℝ := 25
def brick_length : ℝ := 0.2
def brick_width : ℝ := 0.1
def number_of_bricks : ℕ := 20000

-- Calculate the total area covered by the bricks
def total_area_covered := number_of_bricks * (brick_length * brick_width)

-- The proof statement stating that width equals 16
theorem courtyard_width_is_16 (L : ℝ) (B : ℝ) (W : ℝ) (N : ℕ) : 
  L = courtyard_length → B = brick_length → W = brick_width → N = number_of_bricks → 
  (total_area_covered L B W N / L = 16) :=
  by
    sorry

end courtyard_width_is_16_l296_296749


namespace circumscribed_quadrilateral_radius_range_l296_296018

variable {a b c r : ℝ}

theorem circumscribed_quadrilateral_radius_range (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  √(a * b * c / (a + b + c)) < r ∧ r < √(a * b + b * c + c * a) :=
by 
  sorry

end circumscribed_quadrilateral_radius_range_l296_296018


namespace plant_age_count_l296_296357

theorem plant_age_count : 
  let digits := [3, 3, 3, 4, 8, 9],
      even_start := 4 -- The only available even start digit
  in (∃ a: ℕ, ∃ l: list ℕ, 
      l ~ [3, 3, 3, 8, 9] ∧ list to_nat (even_start :: l) == a ∧ -- The age starts with an even digit (4)
      ∃ b: ℕ, b = 20) :=

sorry

end plant_age_count_l296_296357


namespace num_integers_satisfy_inequality_l296_296948

theorem num_integers_satisfy_inequality : 
  ∃ n : ℕ, n = 5 ∧ {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.finite.card = n :=
sorry

end num_integers_satisfy_inequality_l296_296948


namespace equal_probability_sampling_l296_296777

theorem equal_probability_sampling :
  let components := 100 in
  let first_grade := 20 in
  let second_grade := 30 in
  let third_grade := 50 in
  (∀ (method : String), method ∈ ["random", "systematic", "stratified"] →
  (method = "random" → (20 : ℝ) / components = 1 / 5) ∧
  (method = "systematic" → (systematic_prob : ℝ → Bool) → (∀ g in (1..20), systematic_prob g = 1 / 5 / ∏ 1 i) → systematic_prob = 1 / 5) ∧
  (method = "stratified" → (4 / first_grade = 1 / 5) ∧ (6 / second_grade = 1 / 5) ∧ (10 / third_grade = 1 / 5))
)
| "random" := by sorry
| "systematic" := by sorry
| "stratified" := by sorry.

end equal_probability_sampling_l296_296777


namespace prob_two_winning_same_person_prob_two_winning_different_people_prob_neither_A_nor_B_wins_prob_one_winning_A_one_winning_B_l296_296663

namespace Lottery

noncomputable def lottery_probabilities (total_tickets : Nat) (winning_tickets : Nat) (people : Fin n → Type) :=
  let num_people := 4
  let prob_same_person : Real := 2 / 11
  let prob_different_people : Real := 9 / 11
  let prob_neither_A_nor_B : Real := 5 / 22
  let prob_one_A_one_B : Real := 3 / 22

  -- Probability of giving 2 winning tickets to the same person
  theorem prob_two_winning_same_person :
    prob_same_person = 2 / 11 := sorry

  -- Probability of giving 2 winning tickets to different people
  theorem prob_two_winning_different_people :
    prob_different_people = 9 / 11 := sorry

  -- Probability that neither A nor B receives any of the 2 winning tickets
  theorem prob_neither_A_nor_B_wins :
    prob_neither_A_nor_B = 5 / 22 := sorry

  -- Probability of giving 1 winning ticket to A and 1 winning ticket to B
  theorem prob_one_winning_A_one_winning_B :
    prob_one_A_one_B = 3 / 22 := sorry

end Lottery

end prob_two_winning_same_person_prob_two_winning_different_people_prob_neither_A_nor_B_wins_prob_one_winning_A_one_winning_B_l296_296663


namespace girls_ran_more_laps_l296_296552

theorem girls_ran_more_laps
  (boys_laps : ℕ := 27)
  (distance_per_lap : ℝ := 3 / 4)
  (girls_miles : ℝ := 27) :
  let boys_total_miles := boys_laps * distance_per_lap,
      girls_laps := girls_miles / distance_per_lap
  in (girls_laps - boys_laps : ℝ) = 9 :=
by
  sorry

end girls_ran_more_laps_l296_296552


namespace max_rectangle_area_l296_296655

-- Lean statement for the proof problem

theorem max_rectangle_area (x : ℝ) (y : ℝ) (h1 : 2 * x + 2 * y = 24) : ∃ A : ℝ, A = 36 :=
by
  -- Definitions for perimeter and area
  let P := 2 * x + 2 * y
  let A := x * y

  -- Conditions
  have h1 : P = 24 := h1

  -- Setting maximum area and completing the proof
  sorry

end max_rectangle_area_l296_296655


namespace find_monotone_increasing_function_l296_296379

open Real

def is_monotone_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

theorem find_monotone_increasing_function :
  ∀ f, (f = (λ x, |log x|) ∨ f = (λ x, -log x) ∨ f = (λ x, 2^(-x)) ∨ f = (λ x, 2^(|x|))) →
       is_monotone_increasing_on f (set.Icc 0 1) ↔ f = (λ x, 2^(|x|)) :=
begin
  sorry
end

end find_monotone_increasing_function_l296_296379


namespace cube_root_64_is_4_l296_296268

theorem cube_root_64_is_4 : (∃ x : ℕ, x * x * x = 64) → ∃ x : ℕ, x = 4 :=
by
  intro h
  use 4
  sorry

end cube_root_64_is_4_l296_296268


namespace fraction_girls_on_trip_l296_296384

theorem fraction_girls_on_trip (b g : ℕ) (hb : g = 2 * b) 
  (f_g_on_trip : ℚ := 5/6) (f_b_on_trip : ℚ := 1/2) :
  (f_g_on_trip * g) / ((f_g_on_trip * g) + (f_b_on_trip * b)) = 10/13 :=
by
  sorry

end fraction_girls_on_trip_l296_296384


namespace find_courtyard_width_l296_296747

def brick_length_cm := 20
def brick_width_cm := 10
def bricks_required := 20000
def courtyard_length_m := 25

def brick_length_m : ℝ := brick_length_cm / 100.0
def brick_width_m : ℝ := brick_width_cm / 100.0
def area_per_brick_m2 : ℝ := brick_length_m * brick_width_m
def total_area_m2 : ℝ := bricks_required * area_per_brick_m2

theorem find_courtyard_width (length_m : ℝ) (area_m2 : ℝ) : 
  ∃ (W : ℝ), W = 16 :=
by
  have courtyard_width := area_m2 / length_m
  use courtyard_width
  have : courtyard_width = 16 := 
    calc
      courtyard_width = 400 / 25 : by sorry
      ... = 16 : by sorry
  exact this

end find_courtyard_width_l296_296747


namespace part1_1_part1_2_part2_l296_296930

def A (m : ℝ) := { x : ℝ | m - 1 ≤ x ∧ x ≤ 2 * m + 3 }
def B (x : ℝ) := -x^2 + 2 * x + 8 > 0
def complement (S : set ℝ) := { x : ℝ | x ∉ S }

-- Problem 1.1: When m = 2, find A ∪ B
theorem part1_1 : (A 2) ∪ { x : ℝ | B x } = { x : ℝ | -2 < x ∧ x ≤ 7 } := sorry

-- Problem 1.2: When m = 2, find (complement A) ∩ B 
theorem part1_2 : complement (A 2) ∩ { x : ℝ | B x } = { x : ℝ | -2 < x ∧ x < 1 } := sorry

-- Problem 2: If A ∩ B = A, find the range of m
theorem part2 : { m : ℝ | A m ∩ { x : ℝ | B x } = A m } = { m : ℝ | m < -4 ∨ (-1 < m ∧ m < 0.5) } := sorry

end part1_1_part1_2_part2_l296_296930


namespace find_x_l296_296444

def is_mean_twice_mode (l : List ℕ) (mean eq_mode : ℕ) : Prop :=
  l.sum / l.length = eq_mode * 2

theorem find_x (x : ℕ) (h1 : x > 0) (h2 : x ≤ 100)
  (h3 : is_mean_twice_mode [20, x, x, x, x] x (x * 2)) : x = 10 :=
sorry

end find_x_l296_296444


namespace number_of_rational_numbers_l296_296781

def is_rational (x : ℝ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ x = p / q

example : is_rational (7 / 8) := sorry
example : is_rational (6) := sorry  -- Because sqrt(36) = 6 
example : is_rational (1.41414141) := sorry
example : ¬ is_rational (-3 * Real.pi) := sorry
example : ¬ is_rational (Real.sqrt 7) := sorry

theorem number_of_rational_numbers : 
  {a | a ∈ [7 / 8, Real.sqrt 36, -3 * Real.pi, Real.sqrt 7, 1.41414141] ∧ is_rational a}.card = 3 :=
sorry

end number_of_rational_numbers_l296_296781


namespace area_of_rectangle_l296_296291

theorem area_of_rectangle
  (x : ℝ)
  (cost_per_meter : ℝ)
  (total_cost : ℝ)
  (side_ratio_3 : ℝ)
  (side_ratio_4 : ℝ)
  (cost_per_meter_val : cost_per_meter = 0.25)
  (total_cost_val : total_cost = 91)
  (ratio_val : side_ratio_3 / side_ratio_4 = 3 / 4) :
  let side1 := 3 * x,
      side2 := 4 * x,
      perimeter := 2 * (side1 + side2),
      cost := perimeter * cost_per_meter,
      area := side1 * side2
  in cost = total_cost → area = 8112 :=
by
  sorry

end area_of_rectangle_l296_296291


namespace women_per_table_l296_296375

theorem women_per_table (women_per_table : ℕ) (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) (total_men : ℕ) (total_women : ℕ) : 
  num_tables = 9 ∧ men_per_table = 3 ∧ total_customers = 90 → total_men = num_tables * men_per_table ∧ total_women = total_customers - total_men ∧ women_per_table = total_women / num_tables → women_per_table = 7 :=
by
  intro h1 h2
  cases h1 with ht hcus
  cases hcus with hmt hcus
  rw [hmt, ht] at hcus
  have htm := hcus.1
  rw [hcus.2.1, MulComm.mul_comm] at htm
  simp only [Nat.sub_mul] at hcus
  exact hcus.2.2


end women_per_table_l296_296375


namespace isosceles_triangle_l296_296598

variables {A B C M N : Type*}

def is_triangle (A B C : Type*) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (X Y Z : Type*) : ℝ := -- Dummy function to represent perimeter

theorem isosceles_triangle
  {A B C M N : Type*}
  (hABC : is_triangle A B C)
  (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (hM_on_AB : M ∈ [A, B])
  (hN_on_BC : N ∈ [B, C])
  (h_perim_AMC_CNA : perimeter A M C = perimeter C N A)
  (h_perim_ANB_CMB : perimeter A N B = perimeter C M B) :
  (A = B) ∨ (B = C) ∨ (C = A) :=
by sorry

end isosceles_triangle_l296_296598


namespace sum_of_first_three_cards_is_7_l296_296584

def red_cards : List ℕ := [1, 2, 3, 4, 5, 6]
def blue_cards : List ℕ := [4, 5, 6, 7, 8]

def is_valid_arrangement (arrangement : List (ℕ × Char)) : Prop :=
  -- The arrangement must alternate colors
  (∀ i, i % 2 = 0 → arrangement.get? i = some (r, 'r') → arrangement.get? (i + 1) = some (b, 'b')) ∧
  (∀ i, i % 2 = 1 → arrangement.get? i = some (b, 'b') → arrangement.get? (i + 1) = some (r, 'r')) ∧
  -- Red card must divide neighboring blue card
  (∀ i, arrangement.get? i = some (r, 'r') → arrangement.get? (i + 1) = some (b, 'b') → b % r = 0) ∧
  (∀ i, arrangement.get? i = some (b, 'b') → arrangement.get? (i + 1) = some (r, 'r') → r % b = 0)

def sum_of_first_three_cards (arrangement : List (ℕ × Char)) : ℕ :=
  match arrangement with
  | (a₁, _) :: (a₂, _) :: (a₃, _) :: _ => a₁ + a₂ + a₃
  | _ => 0

theorem sum_of_first_three_cards_is_7 :
  ∃ arrangement : List (ℕ × Char), is_valid_arrangement arrangement ∧ sum_of_first_three_cards arrangement = 7 :=
sorry

end sum_of_first_three_cards_is_7_l296_296584


namespace smallest_x_l296_296360

noncomputable def g : ℝ → ℝ
| x := 2 - |x - 3|

lemma g_property (x : ℝ) : g (4 * x) = 4 * g x := 
sorry

lemma g_definition (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : g x = 2 - |x - 3| := 
by simp [g]

theorem smallest_x (x : ℝ) (h : x ≠ 2022.1184 ∧ g x = g 2023) : x > 2022.1184 := 
sorry

end smallest_x_l296_296360


namespace factorial_inequality_factorial_equality_iff_l296_296101

theorem factorial_inequality (x : ℕ → ℕ) (n : ℕ) (h : ∀ i, x i ≥ 0) :
  (∏ i in finset.range n, (x i)!) ≥ (nat.floor (↑((finset.sum (finset.range n) (λ i, x i)) / n) : ℝ) !) ^ n :=
sorry

theorem factorial_equality_iff (x : ℕ → ℕ) (n : ℕ) (h : ∀ i, x i ≥ 0) :
  (∏ i in finset.range n, (x i)!) = (nat.floor ((↑(finset.sum (finset.range n) (λ i, x i))) / n) !) ^ n ↔ ∀ i j, x i = x j :=
sorry

end factorial_inequality_factorial_equality_iff_l296_296101


namespace find_smaller_than_neg3_l296_296036

theorem find_smaller_than_neg3 :
  ∀ (a b c d : ℤ), a = -2 → b = 0 → c = 1 → d = -4 → (∃ x ∈ {a, b, c, d}, x < -3 ∧ ∀ y ∈ {a, b, c, d}, y < -3 → y = d) :=
by
  intros a b c d ha hb hc hd
  use d
  simp [ha, hb, hc, hd]
  split
  · exact hd.symm ▸ (show -4 < -3 from by norm_num)
  · intros y hy
    cases hy
    all_goals { simp [ha, hb, hc, hd] }
  sorry

end find_smaller_than_neg3_l296_296036


namespace semicircle_triangle_area_equal_l296_296408

section Geometry

variables {A B O C : Point} {k : Circle} (hAB : diameter k A B) (hC_on_k : point_on_circle C k)
variables (AC : LineSegment) (hAC : AC.diameter A C)
variables (BC_semi : Semicircle) (h_BC_semi : BC_semi.diameter B C)
variables (triangle_OBC : Triangle O B C)

theorem semicircle_triangle_area_equal 
    (h_max_area : ∀ C' (hC'_on_k : point_on_circle C' k), 
        maximal_area_region_not_covered k (circle_with_diam AC C') = maximal_area_region_not_covered k (circle_with_diam AC C)) :
    semicircle_area BC_semi = triangle_area triangle_OBC :=
sorry
end Geometry

end semicircle_triangle_area_equal_l296_296408


namespace maria_trip_distance_l296_296411

theorem maria_trip_distance
  (D : ℝ)
  (h1 : D/2 = D/8 + 210) :
  D = 560 :=
sorry

end maria_trip_distance_l296_296411


namespace find_a2_l296_296104

noncomputable def geometric_sequence (n : ℕ) : ℝ := sorry -- Define the geometric sequence

variable (a1 : ℝ) (a3a5_eq : ℝ) -- Variables for given conditions

-- Main theorem statement
theorem find_a2 (h_geo : ∀ n, geometric_sequence n = a1 * (2 : ℝ) ^ (n - 1))
  (h_a1 : a1 = 1 / 4)
  (h_a3a5 : (geometric_sequence 3) * (geometric_sequence 5) = 4 * (geometric_sequence 4 - 1)) :
  geometric_sequence 2 = 1 / 2 :=
sorry  -- Proof is omitted

end find_a2_l296_296104


namespace mass_percentage_of_Ca_in_CaO_is_correct_l296_296433

noncomputable def molarMass_Ca : ℝ := 40.08
noncomputable def molarMass_O : ℝ := 16.00
noncomputable def molarMass_CaO : ℝ := molarMass_Ca + molarMass_O
noncomputable def massPercentageCaInCaO : ℝ := (molarMass_Ca / molarMass_CaO) * 100

theorem mass_percentage_of_Ca_in_CaO_is_correct :
  massPercentageCaInCaO = 71.47 :=
by
  -- This is where the proof would go
  sorry

end mass_percentage_of_Ca_in_CaO_is_correct_l296_296433


namespace points_lie_on_hyperbola_l296_296451

noncomputable
def point_on_hyperbola (t : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ 
    (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) }

theorem points_lie_on_hyperbola : 
  ∀ t : ℝ, ∀ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) → (x^2 / 16) - (y^2 / 1) = 1 :=
by 
  intro t x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end points_lie_on_hyperbola_l296_296451


namespace boris_neighbors_l296_296787

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l296_296787


namespace tan_alpha_half_l296_296477

theorem tan_alpha_half (α: ℝ) (h: Real.tan α = 1/2) :
  (1 + 2 * Real.sin (Real.pi - α) * Real.cos (-2 * Real.pi - α)) / (Real.sin (-α)^2 - Real.sin (5 * Real.pi / 2 - α)^2) = -3 := 
by
  sorry

end tan_alpha_half_l296_296477


namespace project_contribution_l296_296631

theorem project_contribution (total_cost : ℝ) (num_participants : ℝ) (expected_contribution : ℝ) 
  (h1 : total_cost = 25 * 10^9) 
  (h2 : num_participants = 300 * 10^6) 
  (h3 : expected_contribution = 83) : 
  total_cost / num_participants = expected_contribution := 
by 
  sorry

end project_contribution_l296_296631


namespace triangle_is_isosceles_l296_296616

variables {A B C M N : Type} [EuclideanGeometry A B C M N]

theorem triangle_is_isosceles 
  (hABC : triangle A B C) 
  (hM : OnSide M A B) 
  (hN : OnSide N B C) 
  (h1 : Perimeter (triangle A M C) = Perimeter (triangle C N A))
  (h2 : Perimeter (triangle A N B) = Perimeter (triangle C M B)) :
  IsIsosceles (triangle A B C) := 
sorry

end triangle_is_isosceles_l296_296616


namespace total_population_l296_296160

variables (b g t : ℕ)

-- Conditions
def cond1 := b = 4 * g
def cond2 := g = 2 * t

-- Theorem statement
theorem total_population (h1 : cond1 b g) (h2 : cond2 g t) : b + g + t = 11 * b / 8 :=
by sorry

end total_population_l296_296160


namespace number_of_tiles_l296_296319

def length : ℝ := 15.47
def breadth : ℝ := 9.67
def area_of_tile : ℝ := 1

theorem number_of_tiles : Real.ceil ((length * breadth) / area_of_tile) = 150 :=
  by
  sorry

end number_of_tiles_l296_296319


namespace max_dot_product_and_range_l296_296505

-- Given definitions and assumptions
def m : ℝ × ℝ := (2, -1)
def n (A B C : ℝ) : ℝ × ℝ := (Real.sin (A / 2), Real.cos (B + C))
variables (A B C a b c : ℝ)

-- Conditions for internal angles and sides of the triangle
axiom angles_sum : A + B + C = Real.pi
axiom law_of_sines : a / Real.sin A = b / Real.sin B = c / Real.sin C
axiom side_a : a = Real.sqrt 3

-- Problem to be proved
theorem max_dot_product_and_range:
  (A = Real.pi / 3 ∧ (b * b + c * c) ∈ Set.Ioc 3 6) := 
by 
  sorry

end max_dot_product_and_range_l296_296505


namespace complex_modulus_equation_l296_296130

theorem complex_modulus_equation (z : ℂ) (i : ℂ) (h : i = complex.I) (h_eq : z * i = 2 - i) : complex.abs z = real.sqrt 5 :=
by
  sorry

end complex_modulus_equation_l296_296130


namespace evaluate_expression_l296_296416

theorem evaluate_expression : (2^3002 * 3^3004) / (6^3003) = (3 / 2) := by
  sorry

end evaluate_expression_l296_296416


namespace sum_of_digits_of_N_l296_296327

theorem sum_of_digits_of_N :
  ∃ N : ℕ, 
    10 ≤ N ∧ N < 100 ∧
    5655 % N = 11 ∧ 
    5879 % N = 14 ∧ 
    ((N / 10) + (N % 10)) = 8 := 
sorry

end sum_of_digits_of_N_l296_296327


namespace finite_powers_of_2_under_condition_main_problem_l296_296440

-- Definition of digit sum
def digitSum (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

-- Lemma needed for the main theorem
lemma non_zero_digit_in_4k_interval (k : ℕ) (m : ℕ) (c : ℕ → ℕ) :
  4 * k ≤ m → (∀ i, k + 1 ≤ i ∧ i ≤ 4 * k → c i = 0) → (digitSum c k < 10^k) → (2^k < 2^n) → (∃ j, k + 1 ≤ j ∧ j ≤ 4 * k ∧ c j ≠ 0) :=
begin
  sorry -- Proof not provided as per instructions
end

-- Main theorem
theorem finite_powers_of_2_under_condition : ∀ (l : ℕ), (∃ N, ∀ n, n ≥ N → digitSum (2^n) > l) :=
begin
  intros l,
  use 4^(l - 1), -- Specific N given in the problem
  intros n hn,
  -- Using lemma and other required arguments to prove the condition
  sorry -- Proof not provided as per instructions
end

-- Specific instance for the given problem
theorem main_problem : ∃ N, ∀ n, n ≥ N → digitSum (2^n) > 2019^2019 :=
begin
  apply finite_powers_of_2_under_condition,
end

end finite_powers_of_2_under_condition_main_problem_l296_296440


namespace smallest_n_binom_n_4_div_10000_l296_296844

theorem smallest_n_binom_n_4_div_10000 :
  ∃ n : ℕ, n ≥ 4 ∧ (nat.choose n 4) % 10000 = 0 ∧
  (∀ m : ℕ, m ≥ 4 → (nat.choose m 4) % 10000 = 0 → n ≤ m) ∧
  n = 8128 :=
by
  sorry

end smallest_n_binom_n_4_div_10000_l296_296844


namespace vector_dot_product_problem_l296_296893

theorem vector_dot_product_problem :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (-1, 3)
  let C : ℝ × ℝ := (2, 1)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  let dot_prod := AB.1 * (2 * AC.1 + BC.1) + AB.2 * (2 * AC.2 + BC.2)
  dot_prod = -14 :=
by
  sorry

end vector_dot_product_problem_l296_296893


namespace determine_distance_l296_296534

noncomputable def distance_formula (d a b c : ℝ) : Prop :=
  (d / a = (d - 30) / b) ∧
  (d / b = (d - 15) / c) ∧
  (d / a = (d - 40) / c)

theorem determine_distance (d a b c : ℝ) (h : distance_formula d a b c) : d = 90 :=
by {
  sorry
}

end determine_distance_l296_296534


namespace slope_of_line_l296_296436

theorem slope_of_line (x y : ℝ) : (∃ (x y : ℝ), x / 4 + y / 3 = 2) → False :=
by
  let m := -3 / 4
  have h : y = -m * x + 6
  sorry


end slope_of_line_l296_296436


namespace no_real_solution_ffx_eq_x_l296_296578

theorem no_real_solution_ffx_eq_x (a b c : ℝ) :
  (b - 1)^2 - 4 * a * c < 0 → ∀ x : ℝ, (let f := λ x, a * x^2 + b * x + c in f (f x) ≠ x) :=
by
  -- outline the proof, but actual steps are omitted.
  intro h x
  let f := λ x, a * x^2 + b * x + c
  sorry

end no_real_solution_ffx_eq_x_l296_296578


namespace problem_equivalent_proof_l296_296779

-- Define the relevant functions
def fA (x : ℝ) : ℝ := x^(-2/3)
def fB (x : ℝ) : ℝ := sqrt (Real.log (abs x) - 3)
def fC (x : ℝ) : ℝ := abs (2^x - 1)
def fD (x : ℝ) : ℝ := 3^x + 3^(-x)

-- Define properties for even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the range property
def has_range_0_to_inf (f : ℝ → ℝ) : Prop := ∀ y : ℝ, y ≥ 0 ↔ ∃ x : ℝ, f x = y

-- State the theorem
theorem problem_equivalent_proof :
  (is_even fB ∧ has_range_0_to_inf fB) ∧
  (¬ is_even fA ∨ ¬ has_range_0_to_inf fA) ∧
  (¬ is_even fC ∨ ¬ has_range_0_to_inf fC) ∧
  (¬ is_even fD ∨ ¬ has_range_0_to_inf fD) :=
by sorry

end problem_equivalent_proof_l296_296779


namespace grandma_gave_each_l296_296245

-- Define the conditions
def gasoline: ℝ := 8
def lunch: ℝ := 15.65
def gifts: ℝ := 5 * 2  -- $5 each for two persons
def total_spent: ℝ := gasoline + lunch + gifts
def initial_amount: ℝ := 50
def amount_left: ℝ := 36.35

-- Define the proof problem
theorem grandma_gave_each :
  (amount_left - (initial_amount - total_spent)) / 2 = 10 :=
by
  sorry

end grandma_gave_each_l296_296245


namespace arithmetic_sequence_sum_l296_296701

theorem arithmetic_sequence_sum :
  ∃ x y d : ℕ,
    d = 6
    ∧ x = 3 + d * (3 - 1)
    ∧ y = x + d
    ∧ y + d = 39
    ∧ x + y = 60 :=
by
  sorry

end arithmetic_sequence_sum_l296_296701


namespace line_CE_perpendicular_BD_l296_296544

variables {A B C D A1 B1 C1 D1 E : ℝ^3} -- Assuming points are in 3D space

-- Definition of midpoint
def is_midpoint (E A1 C1 : ℝ^3) : Prop := E = (A1 + C1) / 2

-- Definition of perpendicularity in 3D space
def is_perpendicular (u v : ℝ^3) : Prop := u.dot v = 0 

-- Given the conditions in the problem
variables (hE : is_midpoint E A1 C1)
variables (CE := C - E) -- Vector CE
variables (BD := B - D) -- Vector BD

-- Prove that CE is perpendicular to BD
theorem line_CE_perpendicular_BD : is_perpendicular CE BD :=
sorry

end line_CE_perpendicular_BD_l296_296544


namespace num_integers_satisfy_l296_296938

theorem num_integers_satisfy : 
  ∃ n : ℕ, (n = 7 ∧ ∀ k : ℤ, (k > -5 ∧ k < 3) → (k = -4 ∨ k = -3 ∨ k = -2 ∨ k = -1 ∨ k = 0 ∨ k = 1 ∨ k = 2)) := 
sorry

end num_integers_satisfy_l296_296938


namespace stone_reaches_bottom_l296_296770

structure StoneInWater where
  σ : ℝ   -- Density of stone in g/cm³
  d : ℝ   -- Depth of lake in cm
  g : ℝ   -- Acceleration due to gravity in cm/sec²
  σ₁ : ℝ  -- Density of water in g/cm³

noncomputable def time_and_velocity (siw : StoneInWater) : ℝ × ℝ :=
  let g₁ := ((siw.σ - siw.σ₁) / siw.σ) * siw.g
  let t := Real.sqrt ((2 * siw.d) / g₁)
  let v := g₁ * t
  (t, v)

theorem stone_reaches_bottom (siw : StoneInWater)
  (hσ : siw.σ = 2.1)
  (hd : siw.d = 850)
  (hg : siw.g = 980.8)
  (hσ₁ : siw.σ₁ = 1.0) :
  time_and_velocity siw = (1.82, 935) :=
by
  sorry

end stone_reaches_bottom_l296_296770


namespace max_king_route_length_l296_296758

theorem max_king_route_length :
  ∃ (x y : ℕ), x + y = 80 ∧
               (16 : ℝ) + (64 : ℝ) * real.sqrt 2 =
               x * (real.sqrt 2) + y :=
by
  sorry

end max_king_route_length_l296_296758


namespace roots_eq_squares_l296_296582

theorem roots_eq_squares (p q : ℝ) (h1 : p^2 - 5 * p + 6 = 0) (h2 : q^2 - 5 * q + 6 = 0) :
  p^2 + q^2 = 13 :=
sorry

end roots_eq_squares_l296_296582


namespace average_speed_to_retreat_l296_296757

theorem average_speed_to_retreat (d : ℕ) (t : ℕ) (v_return : ℕ) (v_to : ℕ) : 
  d = 300 → t = 10 → v_return = 75 → (2 * d / (d / v_to + d / v_return)) = v_to :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_speed_to_retreat_l296_296757


namespace sanmao_current_age_is_15_l296_296724

variable {x : ℕ}  -- Sanmao's age when their dad was 36
variable {dad_current_age : ℕ}
variable {total_current_age : ℕ}

def damao_age_when_dad_36 := 4 * x
def ermao_age_when_dad_36 := 3 * x
def sanmao_age_when_dad_36 := x

def dad_current_age_condition : Prop := dad_current_age = 2 * (x + 3 * x + 4 * x)
def total_current_age_condition : Prop := total_current_age = 108

def damao_current_age := damao_age_when_dad_36 + dad_current_age - 36
def ermao_current_age := ermao_age_when_dad_36 + dad_current_age - 36
def sanmao_current_age := sanmao_age_when_dad_36 + dad_current_age - 36

def total_age_condition : Prop := dad_current_age + damao_current_age + ermao_current_age + sanmao_current_age = total_current_age

theorem sanmao_current_age_is_15 
  (h1 : dad_current_age_condition)
  (h2 : total_current_age_condition)
  (h3 : total_age_condition) :
  sanmao_current_age = 15 := 
sorry

end sanmao_current_age_is_15_l296_296724


namespace subtraction_example_l296_296082

axiom twenty_five_dot_019 : ℝ := 25.019
axiom three_dot_2663 : ℝ := 3.2663

theorem subtraction_example : twenty_five_dot_019 - three_dot_2663 = 21.7527 := 
by sorry

end subtraction_example_l296_296082


namespace find_tan_A_l296_296555

variable (a b c : ℝ)
variable (R : ℝ) (A B C : Real.Angle)
variable [hA : a ≠ 0] [hB : b ≠ 0] [hC : c ≠ 0] [hR : R ≠ 0]

axiom sin_law_1 : Real.sin A = a / (2 * R)
axiom sin_law_2 : Real.sin B = b / (2 * R)
axiom sin_law_3 : Real.sin C = c / (2 * R)
axiom given_cond : 2 * (Real.sin B)^2 + 3 * (Real.sin C)^2 = 2 * Real.sin A * Real.sin B * Real.sin C + (Real.sin A)^2

theorem find_tan_A : Real.tan A = -1 :=
sorry

end find_tan_A_l296_296555


namespace inequality_solution_set_l296_296138

noncomputable def f (x : ℝ) : ℝ := x + 1 / (2 * x) + 2

lemma f_increasing {x₁ x₂ : ℝ} (hx₁ : 1 ≤ x₁) (hx₂ : 1 ≤ x₂) (h : x₁ < x₂) : f x₁ < f x₂ := sorry

lemma solve_inequality (x : ℝ) (hx : 1 ≤ x) : (2 * x - 1 / 2 < x + 1007) → (f (2 * x - 1 / 2) < f (x + 1007)) := sorry

theorem inequality_solution_set {x : ℝ} : (1 ≤ x) → (2 * x - 1 / 2 < x + 1007) ↔ (3 / 4 ≤ x ∧ x < 2015 / 2) := sorry

end inequality_solution_set_l296_296138


namespace cakes_donated_l296_296857
-- Import necessary libraries for arithmetic operations and proofs

-- Define the conditions and required proof in Lean
theorem cakes_donated (c : ℕ) (h : 8 * c + 4 * c + 2 * c = 140) : c = 10 :=
by
  sorry

end cakes_donated_l296_296857


namespace net_profit_reaches_maximum_at_10_l296_296745

/-- Statement:
A company invests 720,000 yuan to build an environmentally friendly building materials factory. The 
operating costs in the first year are 120,000 yuan, and the operating costs increase by 40,000 yuan each year 
thereafter. The annual revenue from selling environmentally friendly building materials is 500,000 yuan. 
Prove that the net profit reaches its maximum in the 10th year. -/

noncomputable def net_profit_maximization_year : ℕ :=
  let initial_investment := 720000
  let first_year_cost := 120000
  let yearly_increase := 40000
  let annual_revenue := 500000

  have net_profit : ℕ → ℕ := λ n, 
    (annual_revenue * n) - initial_investment - (first_year_cost + yearly_increase * (n - 1) * n / 2)
  
  -- We need to prove the year when the net profit reaches its maximum
  10

theorem net_profit_reaches_maximum_at_10 :
  net_profit_maximization_year = 10 := sorry

end net_profit_reaches_maximum_at_10_l296_296745


namespace no_integer_solutions_exist_l296_296096

theorem no_integer_solutions_exist (n m : ℤ) : 
  (n ^ 2 - m ^ 2 = 250) → false := 
sorry 

end no_integer_solutions_exist_l296_296096


namespace trash_cans_veterans_park_l296_296052

theorem trash_cans_veterans_park : 
  ∀ (t_veteran : ℕ) (t_central_half : ℕ) (t_central : ℕ) (t_shift : ℕ), 
    t_veteran = 24 -> 
    t_central_half = t_veteran / 2 -> 
    t_central = t_central_half + 8 -> 
    t_shift = t_central / 2 -> 
    t_veteran + t_shift = 34 := 
by
  intros t_veteran t_central_half t_central t_shift 
  assume h1 : t_veteran = 24
  assume h2 : t_central_half = t_veteran / 2
  assume h3 : t_central = t_central_half + 8
  assume h4 : t_shift = t_central / 2
  sorry

end trash_cans_veterans_park_l296_296052


namespace circle_center_l296_296085

theorem circle_center :
    ∃ (h k : ℝ), (x^2 - 10 * x + y^2 - 4 * y = -4) →
                 (x - h)^2 + (y - k)^2 = 25 ∧ h = 5 ∧ k = 2 :=
sorry

end circle_center_l296_296085


namespace minimal_n_sets_l296_296866

theorem minimal_n_sets (n : ℕ) : 
  (∃ S : Finset (Finset ℕ), S.card = n ∧ 
    (∃ A B C ∈ S, ¬(A ⊆ B ∨ B ⊆ C ∨ C ⊆ A)) ∨
    ∃ A B C ∈ S, (A ⊆ B ∧ B ⊇ C ∧ C ⊇ A)) ↔ n = 5 := 
sorry 

end minimal_n_sets_l296_296866


namespace triangle_side_identity_l296_296852

theorem triangle_side_identity
  (a b c : ℝ)
  (alpha beta gamma : ℝ)
  (h1 : alpha = 60)
  (h2 : a^2 = b^2 + c^2 - b * c) :
  a^2 = (a^3 + b^3 + c^3) / (a + b + c) := 
by
  sorry

end triangle_side_identity_l296_296852


namespace mo_lock_code_count_l296_296723

theorem mo_lock_code_count (n : ℕ) : 
  (∃ (labels : Fin n → Bool) (colors : Fin n → Bool), 
    (∀ i : Fin n, labels (i + 1) = labels i ∨ colors (i + 1) = colors i)) → 
  (if n % 2 = 0 then mo_lock_code_count = 3^n + 3 else mo_lock_code_count = 3^n + 1) 
:= sorry

end mo_lock_code_count_l296_296723


namespace range_of_f_l296_296919

def f (x : ℤ) := x + 1

theorem range_of_f : 
  (∀ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x ∈ ({0, 1, 2, 3} : Set ℤ)) ∧ 
  (∀ y ∈ ({0, 1, 2, 3} : Set ℤ), ∃ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x = y) := 
by 
  sorry

end range_of_f_l296_296919


namespace isosceles_triangle_l296_296605

variables {A B C M N : ℝ}

-- Define the lengths of the sides of the triangle and segments
variables (a b c x y : ℝ)

-- Conditions based on the problem statement
axiom triangle_ABC (ABC_triangle: triangle):
  (point_on_side_A_B M)
  (point_on_side_B_C N)
  (perimeter_eq_1: x + distance M C + b = y + (a - y) + b)
  (perimeter_eq_2: x + y + c = (a - x) + (c - x) + a)

-- Prove that triangle ABC is isosceles
theorem isosceles_triangle (ABC_triangle): isosceles_triangle ABC_triangle := sorry

end isosceles_triangle_l296_296605


namespace triangle_isosceles_l296_296611

theorem triangle_isosceles
  {A B C M N : Point}
  (h_M_on_AB : ∃ t ∈ Set.Icc (0 : ℝ) 1, M = t • A + (1 - t) • B)
  (h_N_on_BC : ∃ t ∈ Set.Icc (0 : ℝ) 1, N = t • B + (1 - t) • C)
  (h_perimeter_AMC_CNA : dist A M + dist M C + dist C A = dist C N + dist N A + dist A C)
  (h_perimeter_ANB_CMB : dist A N + dist N B + dist B A = dist C M + dist M B + dist B C)
  : isosceles_triangle A B C := 
sorry

end triangle_isosceles_l296_296611


namespace ellipse_equation_range_m_l296_296111

open Real

variables {a b : ℝ} (P F1 F2: Point) (m : ℝ) (n : ℝ)

def ellipse (a b : ℝ) (x y : ℝ) := (x^2) / a^2 + (y^2) / b^2 = 1
def condition1 := a > 0 ∧ b > 0 ∧ a > b
def condition2 := dist F1 F2 = 2 * sqrt 3
def condition3 := IsOrthogonal (P - F1) (P - F2)
def condition4 := area_triangle P F1 F2 = 1

theorem ellipse_equation (h1 : condition1)
  (h2 : condition2) (h3 : condition3) (h4 : condition4) :
  ellipse 2 (sqrt 3) :=
by
  sorry

def symmetric_points (A B: Point) := A ∈ ellipse 2 (sqrt 3) ∧ B ∈ ellipse 2 (sqrt 3) ∧ symmetric_about_line y x (A, B) y x + m

theorem range_m (A B : Point) (h1 : condition1)
  (h2 : condition2) (h3 : condition3) (h4 : condition4)
  (h5 : symmetric_points A B) :
  -3 * (sqrt 5) / 5 < m ∧ m < 3 * (sqrt 5) / 5 :=
by
  sorry

end ellipse_equation_range_m_l296_296111


namespace integer_count_in_range_l296_296956

theorem integer_count_in_range (x : Int) : 
  (Set.count (Set.range (λ x, ( -6 ≤ 3*x + 2 ∧ 3*x + 2 ≤ 9))) 5) := 
by 
  sorry

end integer_count_in_range_l296_296956


namespace person_next_to_Boris_arkady_galya_l296_296804

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l296_296804


namespace system_of_equations_solution_l296_296629

theorem system_of_equations_solution :
  ∃ x y : ℚ, (2 * x - 3 * y = 5) ∧ (4 * x - 6 * y = 10) ∧ (x + 2 * y = 7) ∧ (x = 31/7) ∧ (y = 9/7) :=
begin
  sorry
end

end system_of_equations_solution_l296_296629


namespace find_grade_C_boxes_l296_296254

theorem find_grade_C_boxes (m n t : ℕ) (h : 2 * t = m + n) (total_boxes : ℕ) (h_total : total_boxes = 420) : t = 140 :=
by
  sorry

end find_grade_C_boxes_l296_296254


namespace integer_count_in_range_l296_296957

theorem integer_count_in_range (x : Int) : 
  (Set.count (Set.range (λ x, ( -6 ≤ 3*x + 2 ∧ 3*x + 2 ≤ 9))) 5) := 
by 
  sorry

end integer_count_in_range_l296_296957


namespace remainder_polynomial_l296_296321

theorem remainder_polynomial (x : ℤ) : (1 + x) ^ 2010 % (1 + x + x^2) = 1 := 
  sorry

end remainder_polynomial_l296_296321


namespace arithmetic_sequence_common_difference_l296_296898

theorem arithmetic_sequence_common_difference
  (a_n : ℕ → ℤ) (h_arithmetic : ∀ n, (a_n (n + 1) = a_n n + d)) 
  (h_sum1 : a_n 1 + a_n 3 + a_n 5 = 105)
  (h_sum2 : a_n 2 + a_n 4 + a_n 6 = 99) : 
  d = -2 :=
by
  sorry

end arithmetic_sequence_common_difference_l296_296898


namespace smallest_k_for_multiple_of_180_l296_296438

theorem smallest_k_for_multiple_of_180 : ∃ k : ℕ, k > 0 ∧ (∑ i in finset.range (k+1), i^2) % 180 = 0 ∧ k = 1080 := by
  sorry

end smallest_k_for_multiple_of_180_l296_296438


namespace sum_of_palindromic_primes_l296_296618

-- Define what it means to be a palindromic prime under the given conditions
def is_palindromic_prime (p : ℕ) : Prop :=
  p > 9 ∧ p < 100 ∧ nat.prime p ∧ (∀ perm : list ℕ, perm ∈ list.permutations (nat.digits 10 p) → nat.prime (nat.of_digits 10 perm))

-- SumOfPalindromicPrimesTheorem: The sum of all palindromic primes less than 100 is 154
theorem sum_of_palindromic_primes : finset.sum (finset.filter is_palindromic_prime (finset.range 100)) = 154 :=
by
  sorry

end sum_of_palindromic_primes_l296_296618


namespace focus_of_hyperbola_l296_296061

-- Define the given hyperbola equation and its conversion to standard form
def hyperbola_eq (x y : ℝ) : Prop := -2 * (x - 2)^2 + 3 * (y + 3)^2 - 28 = 0

-- Define the standard form equation of the hyperbola
def standard_form (x y : ℝ) : Prop :=
  ((y + 3)^2 / (28 / 3)) - ((x - 2)^2 / 14) = 1

-- Define the coordinates of one of the foci of the hyperbola
def focus (x y : ℝ) : Prop :=
  x = 2 ∧ y = -3 + Real.sqrt (70 / 3)

-- The theorem statement proving the given coordinates is a focus of the hyperbola
theorem focus_of_hyperbola :
  ∃ x y, hyperbola_eq x y ∧ standard_form x y → focus x y :=
by
  existsi 2, (-3 + Real.sqrt (70 / 3))
  sorry -- Proof is required to substantiate it, placeholder here.

end focus_of_hyperbola_l296_296061


namespace problem_statement_l296_296204

variables (α β γ : ℝ)

-- We need centroid definitions to use in our main goal
def centroid (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3, (A.3 + B.3 + C.3) / 3)

-- Coordinates of points A, B, and C
def A : ℝ × ℝ × ℝ := (2*α, 0, 0)
def B : ℝ × ℝ × ℝ := (0, β, 0)
def C : ℝ × ℝ × ℝ := (0, 0, -γ)

-- Coordinates of the centroid of triangle ABC
def p : ℝ := (2*α) / 3
def q : ℝ := β / 3
def r : ℝ := -γ / 3

-- Main theorem
theorem problem_statement (h : (1 / (2*α)^2) + (1 / β^2) + (1 / γ^2) = 1 / 4) :
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 9 / 16 :=
by sorry

end problem_statement_l296_296204


namespace brochures_per_box_l296_296732

theorem brochures_per_box (total_brochures : ℕ) (boxes : ℕ) 
  (htotal : total_brochures = 5000) (hboxes : boxes = 5) : 
  (1000 / 5000 : ℚ) = 1 / 5 := 
by sorry

end brochures_per_box_l296_296732


namespace city_rubber_duck_race_money_raised_l296_296265

theorem city_rubber_duck_race_money_raised :
  let regular_ducks_sold := 221
  let regular_duck_cost := 3
  let large_ducks_sold := 185
  let large_duck_cost := 5
  (regular_ducks_sold * regular_duck_cost + large_ducks_sold * large_duck_cost) = 1588 :=
by
  let regular_ducks_sold := 221
  let regular_duck_cost := 3
  let large_ducks_sold := 185
  let large_duck_cost := 5
  calc
  (regular_ducks_sold * regular_duck_cost + large_ducks_sold * large_duck_cost)
      = (221 * 3 + 185 * 5) : by sorry
  ... = 1588 : by sorry

end city_rubber_duck_race_money_raised_l296_296265


namespace find_magnitude_l296_296500

open Real

variables (a b : ℝ × ℝ)

def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  (u.1 * v.1 + u.2 * v.2 = 0)

def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

theorem find_magnitude
  (h₁ : magnitude a = sqrt 2)
  (h₂ : b = (1, 0))
  (h₃ : is_perpendicular a (a - (2 * b))) :
  magnitude (2 * a + b) = sqrt 13 :=
by
  sorry

end find_magnitude_l296_296500


namespace find_a9_l296_296142
-- Import Lean and Mathlib for required mathematical constructs

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := 10^(log10 (a n) + 1 / 2)

-- Define the theorem to prove a_9 = 10000
theorem find_a9 : a 8 = 10000 := by
  sorry

end find_a9_l296_296142


namespace eccentricity_range_l296_296200

noncomputable def ellipse_eccentricity {a b : ℝ} (ha : a > b) (hb : b > 0) 
(C : x² / a² + y² / b² = 1) : set (ℝ) :=
{e : ℝ | 0 < e ∧ e ≤ Real.sqrt 2 / 2}

theorem eccentricity_range (a b : ℝ) (ha : a > b) (hb : b > 0)
(C : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
(hPB : ∀ x y, x^2 + (y - b)^2 ≤ (2 * b)^2) :
  ∃ e, e ∈ ellipse_eccentricity ha hb C :=
sorry

end eccentricity_range_l296_296200


namespace baO_reaction_l296_296084

theorem baO_reaction (H2O_initial : ℕ) (BaOH2_final : ℕ)
  (reaction_equation : ∀ n : ℕ, n = H2O_initial → n = BaOH2_final) :
  BaOH2_final = 3 → H2O_initial = 3 → n = 3 :=
by
  assume h1 : BaOH2_final = 3,
  assume h2 : H2O_initial = 3,
  sorry

end baO_reaction_l296_296084


namespace sequence_an_general_formula_partial_sum_Tn_l296_296107

-- Define the sequence {a_n} and its partial sum S_n
def S (n : ℕ) : ℕ := 2 * a n - 2

-- Prove that the general formula for a_n is 2^n
theorem sequence_an_general_formula (n : ℕ) (h : ∀ n, S n = 2 * a n - 2) : 
  a n = 2^n := 
sorry

-- Define c_n and its partial sum T_n and prove the formula for T_n
def c (n : ℕ) : ℕ := n / a n

def T (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else finset.sum (finset.range n) (λ k, c k)

theorem partial_sum_Tn (n : ℕ) (h : ∀ n, S n = 2 * a n - 2) : 
  T n = 2 - (n + 2) / 2^n := 
sorry

end sequence_an_general_formula_partial_sum_Tn_l296_296107


namespace minimum_distance_value_l296_296255

noncomputable def minimum_distance : ℝ :=
  let A := (3, 0)
  let distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  real.Inf { d : ℝ |
    ∃ (P Q : ℝ × ℝ), 
      (P.2^2 = P.1 ∧ 
       ((Q.1 - 3)^2 + Q.2^2 = 1) ∧
       d = distance P Q) }

theorem minimum_distance_value : minimum_distance = real.sqrt 11 / 2 - 1 :=
by sorry

end minimum_distance_value_l296_296255


namespace pauls_score_is_91_l296_296162

theorem pauls_score_is_91 (q s c w : ℕ) 
  (h1 : q = 35)
  (h2 : s = 35 + 5 * c - 2 * w)
  (h3 : s > 90)
  (h4 : c + w ≤ 35)
  (h5 : ∀ s', 90 < s' ∧ s' < s → ¬ (∃ c' w', s' = 35 + 5 * c' - 2 * w' ∧ c' + w' ≤ 35 ∧ c' ≠ c)) : 
  s = 91 := 
sorry

end pauls_score_is_91_l296_296162


namespace quadrilateral_area_is_11_l296_296159

def point := (ℤ × ℤ)

def A : point := (0, 0)
def B : point := (1, 4)
def C : point := (4, 3)
def D : point := (3, 0)

def area_of_quadrilateral (p1 p2 p3 p4 : point) : ℤ :=
  let ⟨x1, y1⟩ := p1
  let ⟨x2, y2⟩ := p2
  let ⟨x3, y3⟩ := p3
  let ⟨x4, y4⟩ := p4
  (|x1*y2 - y1*x2 + x2*y3 - y2*x3 + x3*y4 - y3*x4 + x4*y1 - y4*x1|) / 2

theorem quadrilateral_area_is_11 : area_of_quadrilateral A B C D = 11 := by 
  sorry

end quadrilateral_area_is_11_l296_296159


namespace first_day_revenue_l296_296252

theorem first_day_revenue :
  ∀ (S : ℕ), (12 * S + 90 = 246) → (4 * S + 3 * 9 = 79) :=
by
  intros S h1
  sorry

end first_day_revenue_l296_296252


namespace impossible_distinct_indicator_numbers_l296_296423

def indicator_number (grid : Fin 6 × Fin 6 → Fin 7) (i j : Fin 5) : Nat :=
  grid (Fin.succ i, Fin.succ j).val +
  grid (Fin.succ i, j).val +
  grid (i, Fin.succ j).val +
  grid (i, j).val

theorem impossible_distinct_indicator_numbers :
  ∀ (grid : Fin 6 × Fin 6 → Fin 7),
  ∃ (i₁ i₂ j₁ j₂ : Fin 5),
  (i₁ ≠ i₂ ∨ j₁ ≠ j₂) ∧
  indicator_number grid i₁ j₁ = indicator_number grid i₂ j₂ :=
by
  sorry

end impossible_distinct_indicator_numbers_l296_296423


namespace who_next_to_boris_l296_296799

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l296_296799


namespace gcd_factorial_7_8_l296_296864

theorem gcd_factorial_7_8 : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := 
by
  sorry

end gcd_factorial_7_8_l296_296864


namespace ankit_solves_prob_first_l296_296038

theorem ankit_solves_prob_first : 
  let P_A := 1 / 3 in
  let P_not_A := 1 - P_A in
  let P_none := P_not_A ^ 3 in
  1 - P_none = (19 : ℚ) / 27 := 
by
  sorry

end ankit_solves_prob_first_l296_296038


namespace limit_at_3_l296_296424

noncomputable def limit_expression (x : ℝ) : ℝ := (Real.log (x^2 - 5*x + 7)) / (x - 3)

theorem limit_at_3 : Tendsto (λ x : ℝ, limit_expression x) (𝓝 3) (𝓝 1) :=
sorry

end limit_at_3_l296_296424


namespace parabola_focus_distance_l296_296270

theorem parabola_focus_distance (M : ℝ × ℝ) (h1 : (M.2)^2 = 4 * M.1) (h2 : dist M (1, 0) = 4) : M.1 = 3 :=
sorry

end parabola_focus_distance_l296_296270


namespace calculate_sum_l296_296839

theorem calculate_sum :
  (1 : ℚ) + 3 / 6 + 5 / 12 + 7 / 20 + 9 / 30 + 11 / 42 + 13 / 56 + 15 / 72 + 17 / 90 = 81 + 2 / 5 :=
sorry

end calculate_sum_l296_296839


namespace triangle_ABC_XY_l296_296553

variable {A B C L K X Y : Type} 
variable {distance : A → A → ℝ} 
variable (AB AC BC AL BK AX BX CX CY : ℝ)
variable {is_angle_bisector : A → A → Prop} 

theorem triangle_ABC_XY :
  AB = 130 ∧ AC = 125 ∧ BC = 118 ∧ 
  is_angle_bisector A L ∧ L ∈ C ∧ 
  is_angle_bisector B K ∧ K ∈ C ∧ 
  distance C X = distance C Y :=
sorry

end triangle_ABC_XY_l296_296553


namespace select_blocks_l296_296021

/-- A $6 \times 6$ grid has 6 rows and 6 columns. -/
constant rows : Finset ℕ := {1, 2, 3, 4, 5, 6}
constant cols : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Theorem: The number of ways to select 4 blocks from a 6x6 grid such that no two are in the same row 
or column is 5400.-/
theorem select_blocks : 
  (rows.card.choose 4) * (cols.card.choose 4) * factorial 4 = 5400 :=
by
  simp only [rows, cols]
  apply sorry

end select_blocks_l296_296021


namespace problem_I_problem_II_l296_296128

open Nat

def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := (∑ i in range (n + 1), a i)

-- Given sequence definitions
def a : ℕ → ℕ := λ n, 2 * n - 1
def Tn (n : ℕ) : ℝ := ∑ i in finset.range n, (2 : ℝ) / (a i * a (i + 1))

theorem problem_I (n : ℕ) : a n = 2 * n - 1 :=
  sorry

theorem problem_II (n : ℕ) : (2 : ℝ) / 3 ≤ Tn n ∧ Tn n < 1 :=
  sorry

end problem_I_problem_II_l296_296128


namespace graph_of_neg_f_l296_296493

-- Define the function f and its segments as given in the problem statement
def f_segment1 (x : ℝ) : ℝ := 
  if x ∈ Icc (-3 : ℝ) 0 then 1 + (-3/3) * x else 0

def f_circle_center : ℝ × ℝ := (2, -2)
def f_circle_radius : ℝ := 2

def f_segment2 (x : ℝ) : ℝ := 
  if (2 : ℝ) ≤ x ∧ x ≤ (3 : ℝ) then 2 * (x - 2) + 2 else 0

-- Define the transformation corresponding to the negation of the function
def neg_f_segment1 (x : ℝ) : ℝ := 
  if x ∈ Icc (-3 : ℝ) 0 then - (1 + (-3/3) * x) else 0

def neg_f_circle_center : ℝ × ℝ := (2, 2)

def neg_f_segment2 (x : ℝ) : ℝ := 
  if (2 : ℝ) ≤ x ∧ x ≤ (3 : ℝ) then - (2 * (x - 2) + 2) else 0

-- The main statement
theorem graph_of_neg_f (x : ℝ) : 
  (x ∈ Icc (-3 : ℝ) 0 → neg_f_segment1 x = - f_segment1 x) ∧ 
  (x = 2 → (neg_f_circle_center = (2, 2))) ∧
  ((2 : ℝ) ≤ x ∧ x ≤ (3 : ℝ) → neg_f_segment2 x = - f_segment2 x) :=
by
  sorry

end graph_of_neg_f_l296_296493


namespace length_16_sequences_l296_296845

def a : ℕ → ℕ
| 0     := 0
| 1     := 0
| n + 2 := a n + b n
and b : ℕ → ℕ
| 0     := 1
| 1     := 1
| n + 2 := a (n + 1) + b n

theorem length_16_sequences : a 16 + b 16 = 377 :=
by
  sorry

end length_16_sequences_l296_296845


namespace breadth_of_rectangular_plot_l296_296280

-- Definitions based on the conditions
def rectangular_plot_breadth (breadth length area : ℕ) : Prop :=
  length = 3 * breadth ∧ area = length * breadth 

-- Theorem statement
theorem breadth_of_rectangular_plot (b L A : ℕ) 
  (h1 : rectangular_plot_breadth b L A)
  (h2 : A = 867) :
  b = 17 :=
by
  -- definitions from conditions
  let h := h1
  have length_eq := and.left h
  have area_eq := and.right h
  -- provide the skipped proof
  sorry

end breadth_of_rectangular_plot_l296_296280


namespace cone_lateral_surface_area_l296_296746

-- Define given conditions
def base_radius : ℝ := 4
def slant_height : ℝ := 5

-- Define the formula for the lateral surface area of the cone
def lateral_surface_area (r l : ℝ) : ℝ := π * r * l

-- Prove the lateral surface area given the conditions
theorem cone_lateral_surface_area :
  lateral_surface_area base_radius slant_height = 20 * π := 
by
  sorry

end cone_lateral_surface_area_l296_296746


namespace total_cost_of_coat_l296_296766

def original_price : ℝ := 150
def sale_discount : ℝ := 0.25
def additional_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_cost_of_coat :
  let sale_price := original_price * (1 - sale_discount)
  let price_after_discount := sale_price - additional_discount
  let final_price := price_after_discount * (1 + sales_tax)
  final_price = 112.75 :=
by
  -- sorry for the actual proof
  sorry

end total_cost_of_coat_l296_296766


namespace team_a_games_played_l296_296636

theorem team_a_games_played (a b: ℕ) (hA_wins : 3 * a = 4 * wins_A)
(hB_wins : 2 * b = 3 * wins_B)
(hB_more_wins : wins_B = wins_A + 8)
(hB_more_loss : b - wins_B = a - wins_A + 8) :
  a = 192 := 
by
  sorry

end team_a_games_played_l296_296636


namespace who_is_next_to_Boris_l296_296816

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l296_296816


namespace correct_propositions_l296_296854

theorem correct_propositions :
  (∀ f : ℝ → ℝ, linear f → (∃! x : ℝ, f x = 0)) ∧
  (∀ f : ℝ → ℝ, quadratic f → (∃ x y : ℝ, f x = 0 ∧ f y = 0 → x = y ∨ f x = 0 ∧ x = y)) ∧
  (∀ f : ℝ → ℝ, exponential f → ¬∃ x : ℝ, f x = 0) ∧
  (∀ f : ℝ → ℝ, logarithmic f → (∃! x : ℝ, f x = 0)) ∧
  (∀ f : ℝ → ℝ, power f → (∃ x : ℝ, f x = 0 ∨ ¬∃ x : ℝ, f x = 0)) :=
sorry

end correct_propositions_l296_296854


namespace hexagon_cyclic_identity_l296_296175

variables (a a' b b' c c' a₁ b₁ c₁ : ℝ)

theorem hexagon_cyclic_identity :
  a₁ * b₁ * c₁ = a * b * c + a' * b' * c' + a * a' * a₁ + b * b' * b₁ + c * c' * c₁ :=
by
  sorry

end hexagon_cyclic_identity_l296_296175


namespace gcd_n_cube_plus_25_n_plus_3_l296_296874

theorem gcd_n_cube_plus_25_n_plus_3 (n : ℕ) (h : n > 3^2) : 
  Int.gcd (n^3 + 25) (n + 3) = if n % 2 = 1 then 2 else 1 :=
by
  sorry

end gcd_n_cube_plus_25_n_plus_3_l296_296874


namespace remainder_when_divided_l296_296473

/-- Given integers T, E, N, S, E', N', S'. When T is divided by E, 
the quotient is N and the remainder is S. When N is divided by E', 
the quotient is N' and the remainder is S'. Prove that the remainder 
when T is divided by E + E' is ES' + S. -/
theorem remainder_when_divided (T E N S E' N' S' : ℤ) (h1 : T = N * E + S) (h2 : N = N' * E' + S') :
  (T % (E + E')) = (E * S' + S) :=
by
  sorry

end remainder_when_divided_l296_296473


namespace no_solution_l296_296244

theorem no_solution (m n : ℕ) : (5 + 3 * Real.sqrt 2) ^ m ≠ (3 + 5 * Real.sqrt 2) ^ n :=
sorry

end no_solution_l296_296244


namespace fit_three_circles_l296_296872

-- Define the problem context
variables {circle : Type} (c_larger : circle) (c_smaller : ℝ) (n : ℕ)

-- Assume we have a function that determines tangency relations
def is_tangent (a b : circle) : Prop := sorry

-- Conditions for the problem
variables (smaller_circles : Fin n → circle)

axiom eq_size (∀ i : Fin n, smaller_circles i = c_smaller)

axiom tangent_to_large (∀ i : Fin n, is_tangent (smaller_circles i) c_larger)

axiom tangent_to_each_other (∀ i j : Fin n, i ≠ j → is_tangent (smaller_circles i) (smaller_circles j))

-- Define the proof problem
theorem fit_three_circles :
  (∀ (n : ℕ), n = 3 → ∃ (smaller_circles : Fin n → circle), 
    (∀ i, is_tangent (smaller_circles i) c_larger) ∧
    (∀ i j, i ≠ j → is_tangent (smaller_circles i) (smaller_circles j))) :=
begin
  sorry
end

end fit_three_circles_l296_296872


namespace find_non_integer_solution_for_q_l296_296062

noncomputable def q (b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 : ℝ) (x y : ℝ) : ℝ :=
  b0 + b1 * x + b2 * y + b3 * x^2 + b4 * x * y + b5 * y^2 + b6 * x^3 + b7 * x^2 * y + b8 * x * y^2 + b9 * y^3

theorem find_non_integer_solution_for_q :
  ∀ (b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 : ℝ),
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 0 0 = 0 →
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 1 0 = 0 →
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 (-1) 0 = 0 →
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 0 1 = 0 →
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 0 (-1) = 0 →
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 1 1 = 0 →
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 1 (-1) = 0 →
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 (-1) 1 = 0 →
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 (-1) (-1) = 0 →
    q b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 (sqrt (3/2)) (sqrt (3/2)) = 0 :=
by
  intros b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end find_non_integer_solution_for_q_l296_296062


namespace number_of_solutions_l296_296069

noncomputable def g (z : ℂ) : ℂ := -complex.I * complex.conj z

theorem number_of_solutions :
  {z : ℂ | complex.abs z = 3 ∧ g z = z}.to_finset.card = 2 := by sorry

end number_of_solutions_l296_296069


namespace binomial_square_identity_evaluate_expression_1_l296_296685

theorem binomial_square_identity (a b : ℕ) (h : a = 15 ∧ b = 5) : 
  a^2 - 2 * a * b + b^2 = (a - b)^2 :=
by {
  cases h,
  rw [h_left, h_right],
  sorry
}

theorem evaluate_expression_1: 15^2 - 2 * 15 * 5 + 5^2 = 100 := 
by {
  have h : 15^2 - 2 * 15 * 5 + 5^2 = (15 - 5)^2,
  {
    apply binomial_square_identity,
    split,
    exact rfl,
    exact rfl,
  },
  rw h,
  norm_num,
}

end binomial_square_identity_evaluate_expression_1_l296_296685


namespace tan_75_degrees_eq_l296_296393

noncomputable def tan_75_degrees : ℝ := Real.tan (75 * Real.pi / 180)

theorem tan_75_degrees_eq : tan_75_degrees = 2 + Real.sqrt 3 := by
  sorry

end tan_75_degrees_eq_l296_296393


namespace solve_log_equation_l296_296295

theorem solve_log_equation (x : ℝ) :
  real.logb 2 (x^2 - 20 * x + 96) = 5 ↔ (x = 16 ∨ x = 4) :=
by
  sorry

end solve_log_equation_l296_296295


namespace not_in_M_in_M_a_pow_x_sin_kx_in_M_iff_l296_296496

-- Problem 1: Prove that f(x) = x does not belong to set M.
theorem not_in_M (T : ℝ) (hT : T ≠ 0) : ¬ ∃ T : ℝ, (T ≠ 0) ∧ (∀ x : ℝ, x + T = T * x) := 
  sorry

-- Problem 2: Prove that f(x) = a^x belongs to set M given conditions.
theorem in_M_a_pow_x {a : ℝ} (ha1 : a > 0) (ha2 : a ≠ 1) (T : ℝ) (hT : T ≠ 0) (haT : a^T = T) : 
  ∀ x : ℝ, a^(x + T) = T * a^x :=
  sorry

-- Problem 3: Determine the range of k for which f(x) = sin(kx) belongs to set M.
theorem sin_kx_in_M_iff (k : ℝ) (T : ℝ) (hT : T ≠ 0) : (∀ x : ℝ, sin(k * (x + T)) = T * sin(k * x)) ↔ (∃ m : ℤ, k = m * π) :=
  sorry

end not_in_M_in_M_a_pow_x_sin_kx_in_M_iff_l296_296496


namespace persons_next_to_Boris_l296_296822

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l296_296822


namespace standing_next_to_boris_l296_296831

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l296_296831


namespace smallest_C_l296_296464

-- Defining the problem and the conditions
theorem smallest_C (k : ℕ) (C : ℕ) :
  (∀ n : ℕ, n ≥ k → (C * Nat.choose (2 * n) (n + k)) % (n + k + 1) = 0) ↔
  C = 2 * k + 1 :=
by sorry

end smallest_C_l296_296464


namespace connie_initial_marbles_l296_296397

theorem connie_initial_marbles (M : ℕ) :
  let T := M + 45 in
  let R_1 := T / 2 in
  let R_2 := R_1 - 73 in
  R_2 = 70 → M = 241 :=
by
  intro h
  sorry

end connie_initial_marbles_l296_296397


namespace ticket_sales_revenue_l296_296264

/-- The circus sells two kinds of tickets: lower seats for $30 and upper seats for $20. 
    On a certain night, the circus sells 80 tickets and sold 50 tickets for lower seats.
    What is the total revenue from the ticket sales? -/
theorem ticket_sales_revenue :
  let lower_seat_price := 30 in
  let upper_seat_price := 20 in
  let total_tickets := 80 in
  let lower_seat_tickets := 50 in
  let upper_seat_tickets := total_tickets - lower_seat_tickets in
  let revenue_lower_seats := lower_seat_tickets * lower_seat_price in
  let revenue_upper_seats := upper_seat_tickets * upper_seat_price in
  let total_revenue := revenue_lower_seats + revenue_upper_seats in
  total_revenue = 2100 := 
by
  sorry

end ticket_sales_revenue_l296_296264


namespace sqrt_of_sum_of_powers_l296_296696

theorem sqrt_of_sum_of_powers :
  sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 :=
sorry

end sqrt_of_sum_of_powers_l296_296696


namespace minimum_pieces_for_inf_op_l296_296461

variables {V : Type*} [fintype V] (G : simple_graph V) (e : ℕ) [fintype G.edge_set]

-- G is a connected simple graph with e edges.
def is_valid_graph (G : simple_graph V) (e : ℕ) : Prop :=
  G.is_connected ∧ fintype.card G.edge_set = e

-- Conditions based on our problem statement.
theorem minimum_pieces_for_inf_op (m : ℕ) (hG : is_valid_graph G e) :
  ∃ (initial_placement : V → ℕ), (∀ v : V, initial_placement v < G.degree v) ∧ m = e :=
sorry

end minimum_pieces_for_inf_op_l296_296461


namespace find_dihedral_angle_cosine_l296_296313

noncomputable def r (R : ℝ) : ℝ := R / 2
noncomputable def distance_between_centers (R : ℝ) : ℝ := (3 * R) / 2
noncomputable def cos_angle_between_centers_and_edge : ℝ := real.cos (real.pi / 6) -- 30 degrees in radians
noncomputable def dihedral_angle_cos : ℝ := 1 / 9

noncomputable def dihedral_angle_cosine := sorry

theorem find_dihedral_angle_cosine (R : ℝ) (r = r R) (d = distance_between_centers R)
    (a = cos_angle_between_centers_and_edge) : dihedral_angle_cosine = dihedral_angle_cos :=
    sorry

end find_dihedral_angle_cosine_l296_296313


namespace bananas_purchased_l296_296361

-- Define the cost per pound for each purchase
def cost_per_pound_first : ℝ := 0.50 / 3
def cost_per_pound_second : ℝ := 1.25 / 5
def selling_price_per_pound : ℝ := 1.00 / 4

-- Define variables for the pounds of bananas purchased
variables (x y : ℝ)

-- Define the total cost and total revenue
def total_cost : ℝ := cost_per_pound_first * x + cost_per_pound_second * y
def total_revenue : ℝ := selling_price_per_pound * (x + y)

-- Define the profit equation
def profit : ℝ := total_revenue - total_cost

-- Given profit is $8.00, prove that total pounds purchased is 96
theorem bananas_purchased (h : profit x y = 8) : x = 96 :=
by {
    sorry
}

end bananas_purchased_l296_296361


namespace triangle_is_isosceles_l296_296597

variables {A B C M N : Type} [pseudo_metric_space A] [pseudo_metric_space B] [pseudo_metric_space C]
[pseudo_metric_space M] [pseudo_metric_space N] 

variables {dist : ∀ {X : Type} [pseudo_metric_space X], X → X → ℝ}
variables {a b c x y : ℝ} -- edge lengths

-- Define the points and their distances
variables {MA MB NA NC : ℝ} (tABC : triangle A B C) 

-- Conditions from the problem
def condition1 : Prop :=
  dist A M + dist M C + dist C A = dist C N + dist N A + dist A C

def condition2 : Prop :=
  dist A N + dist N B + dist B A = dist C M + dist M B + dist B C

-- Proving the triangle is isosceles
theorem triangle_is_isosceles (tABC : triangle A B C) 
    (h1 : condition1)
    (h2 : condition2) : isosceles tABC :=
sorry

end triangle_is_isosceles_l296_296597


namespace sum_of_complex_roots_of_unity_l296_296577

theorem sum_of_complex_roots_of_unity (x : ℂ) (hx : x ≠ 1) (h : x^2013 = 1) :
  ∑ k in Finset.range 2012 | (k + 1), (x ^ (2 * (k + 1)) / (x ^ (k + 1) - 1)) = 3018 :=
by
  sorry

end sum_of_complex_roots_of_unity_l296_296577


namespace complement_of_M_l296_296583

theorem complement_of_M :
  let U := {1, 2, 3, 4, 5, 6}
  let M := {1, 2, 4}
  \complement_U M = {3, 5, 6} :=
by
  sorry

end complement_of_M_l296_296583


namespace triangle_isosceles_l296_296610

theorem triangle_isosceles
  {A B C M N : Point}
  (h_M_on_AB : ∃ t ∈ Set.Icc (0 : ℝ) 1, M = t • A + (1 - t) • B)
  (h_N_on_BC : ∃ t ∈ Set.Icc (0 : ℝ) 1, N = t • B + (1 - t) • C)
  (h_perimeter_AMC_CNA : dist A M + dist M C + dist C A = dist C N + dist N A + dist A C)
  (h_perimeter_ANB_CMB : dist A N + dist N B + dist B A = dist C M + dist M B + dist B C)
  : isosceles_triangle A B C := 
sorry

end triangle_isosceles_l296_296610


namespace distance_to_apex_l296_296672

noncomputable def area_ratios : ℝ := (162 * Real.sqrt 3) / (288 * Real.sqrt 3)
noncomputable def side_ratios : ℝ := Real.sqrt (area_ratios)
noncomputable def H : ℝ := 6 / (1 - side_ratios)

theorem distance_to_apex :
  ∃ H : ℝ, H = 24 ∧ 0 < H ∧ (6 / (1 - (Real.sqrt ((162 * Real.sqrt 3) / (288 * Real.sqrt 3))))) = H :=
by {
  -- placeholder for proof
  existsi 24,
  sorry
}

end distance_to_apex_l296_296672


namespace num_integers_satisfying_inequality_l296_296952

theorem num_integers_satisfying_inequality : 
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 := 
by 
  sorry

end num_integers_satisfying_inequality_l296_296952


namespace sample_size_l296_296026

theorem sample_size (r1 r2 r3 : ℕ) (h_ratio : r1 = 2 ∧ r2 = 3 ∧ r3 = 5) (max_sample : ℕ) (h_max_sample : max_sample = 60) : 
  let total_sample := max_sample * 2
  in total_sample = 120 :=
by
  sorry

end sample_size_l296_296026


namespace find_number_l296_296515

variable {x : ℝ}

theorem find_number (h : (30 / 100) * x = (40 / 100) * 40) : x = 160 / 3 :=
by
  sorry

end find_number_l296_296515


namespace volume_of_inscribed_sphere_l296_296367

/-- A right circular cone sits on a table, pointing upward.
The cross-section triangle, perpendicular to the base, has a vertex angle of 90 degrees.
The diameter of the cone's base is 24 inches.
A sphere is placed inside the cone such that it is tangent to the sides of the cone and rests on the table.
We prove the volume of the sphere in cubic inches, expressing the answer in terms of π. -/
theorem volume_of_inscribed_sphere:
  let d := 24 in
  let r := d / 2 in
  let R := r / 2 in
  let V := (4 / 3) * Real.pi * R^3 in
  V = 288 * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l296_296367


namespace seeds_in_first_plot_l296_296878

theorem seeds_in_first_plot (x : ℕ) (h1 : 0 < x)
  (h2 : 200 = 200)
  (h3 : 0.25 * (x : ℝ) = 0.25 * (x : ℝ))
  (h4 : 0.35 * 200 = 70)
  (h5 : (0.25 * (x : ℝ) + 70) / (x + 200) = 0.29) :
  x = 300 :=
by sorry

end seeds_in_first_plot_l296_296878


namespace triangle_is_isosceles_l296_296614

variables {A B C M N : Type} [EuclideanGeometry A B C M N]

theorem triangle_is_isosceles 
  (hABC : triangle A B C) 
  (hM : OnSide M A B) 
  (hN : OnSide N B C) 
  (h1 : Perimeter (triangle A M C) = Perimeter (triangle C N A))
  (h2 : Perimeter (triangle A N B) = Perimeter (triangle C M B)) :
  IsIsosceles (triangle A B C) := 
sorry

end triangle_is_isosceles_l296_296614


namespace max_odd_sums_l296_296337

theorem max_odd_sums (X Y Z W : ℕ) (hX : X > 0) (hY : Y > 0) (hZ : Z > 0) (hW : W > 0) :
  max (count_odd [X + Y, X + Z, X + W, Y + Z, Y + W, Z + W]) = 4 :=
sorry

end max_odd_sums_l296_296337


namespace determine_y_l296_296151

theorem determine_y (x y : ℝ) (h₁ : x^2 = y - 7) (h₂ : x = 7) : y = 56 :=
sorry

end determine_y_l296_296151


namespace days_left_in_year_is_100_l296_296157

noncomputable def days_left_in_year 
    (daily_average_rain_before : ℝ) 
    (total_rainfall_so_far : ℝ) 
    (average_rain_needed : ℝ) 
    (total_days_in_year : ℕ) : ℕ :=
    sorry

theorem days_left_in_year_is_100 :
    days_left_in_year 2 430 3 365 = 100 := 
sorry

end days_left_in_year_is_100_l296_296157


namespace predicted_temperature_l296_296065

def avg_x (x_vals : List ℕ) : ℕ := (x_vals.foldl (· + ·) 0) / x_vals.length
def avg_y (y_vals : List ℕ) : ℕ := (y_vals.foldl (· + ·) 0) / y_vals.length

theorem predicted_temperature (k : ℚ) (x_vals y_vals : List ℚ) (x : ℕ) (H : (avg_x x_vals = 40) ∧ (avg_y y_vals = 30) ∧ k = 20) :
  0.25 * 80 + k = 40 :=
by
  sorry

end predicted_temperature_l296_296065


namespace lines_through_point_area_condition_l296_296010

theorem lines_through_point_area_condition :
  let P : ℝ × ℝ := (-2, 2)
  let area : ℝ := 8
  let line_eq (k : ℝ) (x : ℝ) : ℝ := k * (x + 2) + 2
  let intersection_x (k : ℝ) : ℝ := - (2 + 2 * k) / k
  let intersection_y (k : ℝ) : ℝ := 2 * k + 2
  let triangle_area (k : ℝ) : ℝ := 0.5 * (| intersection_x k | * | intersection_y k |)
  let k_values := { k : ℝ | k ≠ 0 ∧ (k + 1)^2 = 4 * |k| ∨ (k + 1)^2 = -4 * k }
  cardinality { l : (ℝ → ℝ) | ∃ k : ℝ, k ∈ k_values ∧ (∀ x : ℝ, l(x) = line_eq k x) } = 3 :=
by
  sorry

end lines_through_point_area_condition_l296_296010


namespace arrangement_count_l296_296441

theorem arrangement_count :
  let people := {A, B, C, D, E} in
  let valid_arrangements := { l | l ∈ people.permutations ∧
                            (∀ i j k, l.nth i = some A ∧ l.nth j = some B ∧ l.nth k = some C → 
                             (i < k ∧ j < k) ∨ (i > k ∧ j > k)) } in
  valid_arrangements.card = 80 :=
by sorry

end arrangement_count_l296_296441


namespace product_greater_sum_l296_296216

theorem product_greater_sum (a : ℕ → ℝ) (n : ℕ) (h_pos : ∀ i, 0 < a i)
  (h_lt : ∀ i, a i < 1) (h_n_ge_two : 2 ≤ n) :
  (∏ i in finset.range n, a i) > (∑ i in finset.range n, a i) + 1 - n :=
sorry

end product_greater_sum_l296_296216


namespace cosine_dihedral_angle_result_l296_296311

noncomputable def cosine_dihedral_angle (R : ℝ) : ℝ :=
  let r := R / 2
  let distance_between_centers := 3 * R / 2
  let θ := 30 * Real.pi / 180
  let α := acos ((R / distance_between_centers)) / θ -- this is from interpreting in 3D
  1 / 9  -- result from derived conditions

theorem cosine_dihedral_angle_result (R : ℝ) (α : ℝ) (h1 : r = R / 2) (h2 : distance_between_centers = 3 * R / 2) (h3: θ = 30 * Real.pi / 180) : 
  cosine_dihedral_angle R = 1 / 9 :=
sorry

end cosine_dihedral_angle_result_l296_296311


namespace omega_range_l296_296569

noncomputable def period_sin (ω : ℝ) := (2 * Real.pi) / ω

theorem omega_range (ω : ℝ) 
    (hω : 0 < ω) 
    (hmono : ∀ x₁ x₂ ∈ (Set.Icc (-Real.pi / 3) (Real.pi / 4)), x₁ ≤ x₂ → (2 * Real.sin (ω * x₁) ≤ 2 * Real.sin (ω * x₂))) : 
  0 < ω ∧ ω ≤ (3 / 2) := 
by
  sorry

end omega_range_l296_296569


namespace average_speed_is_11_point_52_l296_296015

-- Define the speeds for each lap
def v1 : ℝ := 6
def v2 : ℝ := 12
def v3 : ℝ := 18
def v4 : ℝ := 24

-- Define the distance of each lap as a positive real number
variable (d : ℝ) (h : d > 0)

-- Calculate the total distance
def total_distance : ℝ := 4 * d

-- Calculate the total time taken
def total_time : ℝ :=
  (d / v1) + (d / v2) + (d / v3) + (d / v4)

-- Calculate the average speed
def average_speed : ℝ :=
  total_distance / total_time

-- Prove that the average speed is 11.52 kmph
theorem average_speed_is_11_point_52 : average_speed d = 11.52 := by
  unfold average_speed total_distance total_time v1 v2 v3 v4
  -- Mathematical calculation confirming the steps from provided solution
  have h1 : (d / 6 + d / 12 + d / 18 + d / 24) = d * (1 / 6 + 1 / 12 + 1 / 18 + 1 / 24) := by ring
  have h2 : (1 / 6 + 1 / 12 + 1 / 18 + 1 / 24) = (12 / 72 + 6 / 72 + 4 / 72 + 3 / 72) := by norm_num
  have h3 : (12 / 72 + 6 / 72 + 4 / 72 + 3 / 72) = (25 / 72) := by norm_num
  rw [h1, h2, h3]
  -- Continue proving the average speed is 11.52
  have h4 : 4 / (25 / 72) = 4 * (72 / 25) := by field_simp
  have h5 : 4 * (72 / 25) = (4 * 72) / 25 := by ring
  norm_num at h4
  rwa [h4, h5]
  
-- sorry to skip the proof
  sorry

end average_speed_is_11_point_52_l296_296015


namespace simplify_expression_l296_296846

theorem simplify_expression : 2 - Real.sqrt 3 + 1 / (2 - Real.sqrt 3) + 1 / (Real.sqrt 3 + 2) = 6 :=
by
  sorry

end simplify_expression_l296_296846


namespace number_of_integers_satisfying_inequality_l296_296936

theorem number_of_integers_satisfying_inequality : 
  {n : Int | (n - 3) * (n + 5) < 0}.card = 7 :=
by
  sorry

end number_of_integers_satisfying_inequality_l296_296936


namespace sum_of_A_plus_B_sum_of_possible_A_plus_B_l296_296492

theorem sum_of_A_plus_B (A B : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) 
  (h : (A + B + 23) % 9 = 0) : A + B = 4 ∨ A + B = 13 :=
sorry

theorem sum_of_possible_A_plus_B : 17 = 
  (if ∃ A B : ℕ, (0 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ (A + B + 23) % 9 = 0 then 
    4 + 13 else 0) :=
by
  have h1 : ∃ A B : ℕ, (0 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ (A + B + 23) % 9 = 0, from sorry
  have h2 : A + B = 4 ∨ A + B = 13, from sorry
  exact 17 = 4 + 13

end sum_of_A_plus_B_sum_of_possible_A_plus_B_l296_296492


namespace sqrt_of_sum_of_powers_l296_296694

theorem sqrt_of_sum_of_powers :
  sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 :=
sorry

end sqrt_of_sum_of_powers_l296_296694


namespace no_valid_conference_division_l296_296170

theorem no_valid_conference_division (num_teams : ℕ) (matches_per_team : ℕ) :
  num_teams = 30 → matches_per_team = 82 → 
  ¬ ∃ (k : ℕ) (x y z : ℕ), k + (num_teams - k) = num_teams ∧
                          x + y + z = (num_teams * matches_per_team) / 2 ∧
                          z = ((x + y + z) / 2) := 
by
  sorry

end no_valid_conference_division_l296_296170


namespace vectors_are_coplanar_l296_296503

noncomputable def vector_a := (-3, 2, 1) : ℝ × ℝ × ℝ
noncomputable def vector_b := (2, 2, -1) : ℝ × ℝ × ℝ
noncomputable def vector_c (m : ℝ) := (m, 4, 0) : ℝ × ℝ × ℝ

def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, c = (x * a.1 + y * b.1, x * a.2 + y * b.2, x * a.3 + y * b.3)

theorem vectors_are_coplanar : coplanar vector_a vector_b (vector_c (-1)) :=
  by
  sorry

end vectors_are_coplanar_l296_296503


namespace sqrt_sum_of_cubes_l296_296689

theorem sqrt_sum_of_cubes :
  √(4^3 + 4^3 + 4^3 + 4^3) = 16 :=
by
  sorry

end sqrt_sum_of_cubes_l296_296689


namespace sqrt_of_sum_of_powers_l296_296692

theorem sqrt_of_sum_of_powers : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_of_sum_of_powers_l296_296692


namespace sin_2012_is_negative_l296_296729

theorem sin_2012_is_negative : sin (2012 * (π / 180)) < 0 :=
by sorry

end sin_2012_is_negative_l296_296729


namespace nat_diff_same_prime_divisors_l296_296622

/-- Every natural number can be expressed as the difference of two natural numbers that have the same number of prime divisors. -/
theorem nat_diff_same_prime_divisors (n : ℕ) : 
  ∃ a b : ℕ, (a - b = n) ∧ (card a.prime_divisors = card b.prime_divisors) := 
sorry

end nat_diff_same_prime_divisors_l296_296622


namespace total_profit_correct_l296_296031

def investment_A : ℝ := 6300
def investment_B : ℝ := 4200
def investment_C : ℝ := 10500
def profit_share_A : ℝ := 3750

def total_profit (investment_A investment_B investment_C profit_share_A : ℝ) : ℝ :=
  let ratio_A := investment_A / 2100
  let ratio_B := investment_B / 2100
  let ratio_C := investment_C / 2100
  let total_ratio := ratio_A + ratio_B + ratio_C
  let value_per_part := profit_share_A / ratio_A
  value_per_part * total_ratio

theorem total_profit_correct :
  total_profit investment_A investment_B investment_C profit_share_A = 12500 := sorry

end total_profit_correct_l296_296031


namespace courtyard_width_is_16_l296_296750

-- Define the conditions
def courtyard_length : ℝ := 25
def brick_length : ℝ := 0.2
def brick_width : ℝ := 0.1
def number_of_bricks : ℕ := 20000

-- Calculate the total area covered by the bricks
def total_area_covered := number_of_bricks * (brick_length * brick_width)

-- The proof statement stating that width equals 16
theorem courtyard_width_is_16 (L : ℝ) (B : ℝ) (W : ℝ) (N : ℕ) : 
  L = courtyard_length → B = brick_length → W = brick_width → N = number_of_bricks → 
  (total_area_covered L B W N / L = 16) :=
  by
    sorry

end courtyard_width_is_16_l296_296750


namespace gcd_cubic_l296_296877

theorem gcd_cubic (n : ℕ) (h1 : n > 9) :
  let k := gcd (n^3 + 25) (n + 3)
  in if (n + 3) % 2 = 1 then k = 1 else k = 2 :=
by
  sorry

end gcd_cubic_l296_296877


namespace multiplication_represents_expansion_l296_296678

theorem multiplication_represents_expansion (a b : ℕ) :
  a * b = expanding a by a factor of b :=
sorry

end multiplication_represents_expansion_l296_296678


namespace sufficient_condition_for_inequality_l296_296292

theorem sufficient_condition_for_inequality (x : ℝ) : (1 - 1/x > 0) → (x > 1) :=
by
  sorry

end sufficient_condition_for_inequality_l296_296292


namespace closest_time_to_1600_mirror_l296_296836

noncomputable def clock_in_mirror_time (hour_hand_minute: ℕ) (minute_hand_minute: ℕ) : (ℕ × ℕ) :=
  let hour_in_mirror := (12 - hour_hand_minute) % 12
  let minute_in_mirror := minute_hand_minute
  (hour_in_mirror, minute_in_mirror)

theorem closest_time_to_1600_mirror (A B C D : (ℕ × ℕ)) :
  clock_in_mirror_time 4 0 = D → D = (8, 0) :=
by
  -- Introduction of hypothesis that clock closest to 16:00 (4:00) is represented by D
  intro h
  -- State the conclusion based on the given hypothesis
  sorry

end closest_time_to_1600_mirror_l296_296836


namespace log_a_b_gt_2_l296_296099

variables (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (ln b) / (a - 1) = (a + 1) / a)

theorem log_a_b_gt_2 : log a b > 2 :=
by
  sorry

end log_a_b_gt_2_l296_296099


namespace donation_percentage_correct_l296_296016

noncomputable def percentage_donated_to_orphan_house (income remaining : ℝ) (given_to_children_percentage : ℝ) (given_to_wife_percentage : ℝ) (remaining_after_donation : ℝ)
    (before_donation_remaining : income * (1 - given_to_children_percentage / 100 - given_to_wife_percentage / 100) = remaining)
    (after_donation_remaining : remaining - remaining_after_donation * remaining = 500) : Prop :=
    100 * (remaining - 500) / remaining = 16.67

theorem donation_percentage_correct 
    (income : ℝ) 
    (child_percentage : ℝ := 10)
    (num_children : ℕ := 2)
    (wife_percentage : ℝ := 20)
    (final_amount : ℝ := 500)
    (income_value : income = 1000 ) : 
    percentage_donated_to_orphan_house income 
    (income * (1 - (child_percentage * num_children) / 100 - wife_percentage / 100)) 
    (child_percentage * num_children)
    wife_percentage 
    final_amount 
    sorry 
    sorry :=
sorry

end donation_percentage_correct_l296_296016


namespace determine_right_hand_coin_l296_296619

theorem determine_right_hand_coin (L R : ℕ) (M_L : ℕ) (M_R : ℕ) (S : ℕ) :
  ((L = 10 ∨ L = 15) ∧ (R = 10 ∨ R = 15) ∧ L ≠ R) →
  (M_L ∈ {4, 10, 12, 26}) →
  (M_R ∈ {7, 13, 21, 35}) →
  S = L * M_L + R * M_R →
  ((S % 2 = 0 ↔ R = 10) ∧ (S % 2 = 1 ↔ R = 15)) :=
by
  intros hLR hML hMR hS
  sorry

end determine_right_hand_coin_l296_296619


namespace sum_S_p_eq_757750_l296_296443

def S_p (p : ℕ) : ℤ :=
  (25 * (149 * p - 49))

theorem sum_S_p_eq_757750 : 
  ∑ p in Finset.range 20 | (p + 1), S_p (p + 1) = 757750 :=
by
  sorry

end sum_S_p_eq_757750_l296_296443


namespace midpoint_arc_perpendicular_l296_296230

open Real
open EuclideanGeometry

noncomputable def midpoint (A B : Point) : Point := sorry -- Assume we have an appropriate implementation.
noncomputable def arc_midpoint (C : Circle) (A B : Point) : Point := sorry -- Assume we have an appropriate implementation.

theorem midpoint_arc_perpendicular {A B C D : Point} (hA : A ∈ Circle) (hB : B ∈ Circle) (hC : C ∈ Circle) (hD : D ∈ Circle) 
    (h_order : cyclic_order A B C D) :
    let A1 := arc_midpoint (Circle A B) A B,
        B1 := arc_midpoint (Circle B C) B C,
        C1 := arc_midpoint (Circle C D) C D,
        D1 := arc_midpoint (Circle D A) D A
    in perpendicular (line_through A1 C1) (line_through B1 D1) :=
sorry

end midpoint_arc_perpendicular_l296_296230


namespace prime_form_4k_plus_3_l296_296580

noncomputable theory

theorem prime_form_4k_plus_3 {p x y : ℕ} (hp : p.prime) (h_form : ∃ k : ℕ, p = 4 * k + 3)
  (hx : 0 < x) (hy : 0 < y) 
  (h_div : p ∣ (x^2 - x * y + ((p + 1) / 4) * y^2)) :
  ∃ u v : ℤ, x^2 - x * y + ((p + 1) / 4 : ℕ) * y^2 = p * (u^2 - u * v + ((p + 1) / 4 : ℕ) * v^2) := 
sorry

end prime_form_4k_plus_3_l296_296580


namespace vertex_of_parabola_l296_296661

theorem vertex_of_parabola : 
  (exists (a b: ℝ), ∀ x: ℝ, (a * (x - 1)^2 + b = (x - 1)^2 - 2)) → (1, -2) = (1, -2) :=
by
  intro h
  sorry

end vertex_of_parabola_l296_296661


namespace hamza_bucket_problem_l296_296146

-- Definitions reflecting the problem conditions
def bucket_2_5_capacity : ℝ := 2.5
def bucket_3_0_capacity : ℝ := 3.0
def bucket_5_6_capacity : ℝ := 5.6
def bucket_6_5_capacity : ℝ := 6.5

def initial_fill_in_5_6 : ℝ := bucket_5_6_capacity
def pour_5_6_to_3_0_remaining : ℝ := 5.6 - 3.0
def remaining_in_5_6_after_second_fill : ℝ := bucket_5_6_capacity - 0.5

-- Main problem statement
theorem hamza_bucket_problem : (bucket_6_5_capacity - 2.6 = 3.9) :=
by sorry

end hamza_bucket_problem_l296_296146


namespace floor_neg_seven_thirds_l296_296079

theorem floor_neg_seven_thirds : Int.floor (-7 / 3 : ℚ) = -3 := by
  sorry

end floor_neg_seven_thirds_l296_296079


namespace custom_op_4_8_l296_296654

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := b + b / a

-- Theorem stating the desired equality
theorem custom_op_4_8 : custom_op 4 8 = 10 :=
by
  -- Proof is omitted
  sorry

end custom_op_4_8_l296_296654


namespace problem_statement_l296_296885

variable {x y : Real}

theorem problem_statement (hx : x * y < 0) (hxy : x > |y|) : x + y > 0 := by
  sorry

end problem_statement_l296_296885


namespace total_people_at_zoo_l296_296741

theorem total_people_at_zoo (A K : ℕ) (ticket_price_adult : ℕ := 28) (ticket_price_kid : ℕ := 12) (total_sales : ℕ := 3864) (number_of_kids : ℕ := 203) :
  (ticket_price_adult * A + ticket_price_kid * number_of_kids = total_sales) → 
  (A + number_of_kids = 254) :=
by
  sorry

end total_people_at_zoo_l296_296741


namespace find_volume_of_rotated_triangle_l296_296092

open Real

noncomputable def height_of_equilateral_triangle (a : ℝ) : ℝ :=
  (a * sqrt 3) / 2

noncomputable def volume_of_rotated_equilateral_triangle (a : ℝ) : ℝ :=
  (pi * (a^3) * sqrt 3) / 24

theorem find_volume_of_rotated_triangle (a : ℝ) (a_pos : 0 < a) : 
  let h := height_of_equilateral_triangle a in 
  volume_of_rotated_equilateral_triangle a = (pi * (a^3) * sqrt 3) / 24 
:= by 
  sorry

end find_volume_of_rotated_triangle_l296_296092


namespace expected_adjacent_black_l296_296263

noncomputable def ExpectedBlackPairs :=
  let totalCards := 104
  let blackCards := 52
  let totalPairs := 103
  let probAdjacentBlack := (blackCards - 1) / (totalPairs)
  blackCards * probAdjacentBlack

theorem expected_adjacent_black :
  ExpectedBlackPairs = 2601 / 103 :=
by
  sorry

end expected_adjacent_black_l296_296263


namespace find_divisor_l296_296089

-- Define the initial number
def num := 1387

-- Define the number to subtract to make it divisible by some divisor
def least_subtract := 7

-- Define the resulting number after subtraction
def remaining_num := num - least_subtract

-- Define the divisor
def divisor := 23

-- The statement to prove: 1380 is divisible by 23
theorem find_divisor (num_subtract_div : num - least_subtract = remaining_num) 
                     (remaining_divisor : remaining_num = 1380) : 
                     ∃ k : ℕ, 1380 = k * divisor := by
  sorry

end find_divisor_l296_296089


namespace NK_bisects_angle_BNC_l296_296166

open EuclideanGeometry

variables {A B C K D M N I : Point}
variable h₁ : Triangle A B C
variable h₂ : acute_triangle h₁
variable h₃ : altitude A D
variable h₄ : midpoint M A D
variable h₅ : incenter I A B C
variable h₆ : touches I A B C B C K
variable h₇ : intersects KM N I

theorem NK_bisects_angle_BNC :
  bisects NK (angle B N C) :=
sorry

end NK_bisects_angle_BNC_l296_296166


namespace arithmetic_sequence_sum_l296_296702

theorem arithmetic_sequence_sum :
  ∃ x y d : ℕ,
    d = 6
    ∧ x = 3 + d * (3 - 1)
    ∧ y = x + d
    ∧ y + d = 39
    ∧ x + y = 60 :=
by
  sorry

end arithmetic_sequence_sum_l296_296702


namespace train_speed_l296_296371

theorem train_speed
  (train_length : ℝ)
  (cross_time : ℝ)
  (man_speed_kmh : ℝ)
  (train_speed_kmh : ℝ) :
  (train_length = 150) →
  (cross_time = 6) →
  (man_speed_kmh = 5) →
  (man_speed_kmh * 1000 / 3600 + (train_speed_kmh * 1000 / 3600)) * cross_time = train_length →
  train_speed_kmh = 85 :=
by
  intros htl hct hmk hs
  sorry

end train_speed_l296_296371


namespace bug_paths_l296_296737

-- Define the problem conditions
structure PathSetup (A B : Type) :=
  (red_arrows : ℕ) -- number of red arrows from point A
  (red_to_blue : ℕ) -- number of blue arrows reachable from each red arrow
  (blue_to_green : ℕ) -- number of green arrows reachable from each blue arrow
  (green_to_orange : ℕ) -- number of orange arrows reachable from each green arrow
  (start_arrows : ℕ) -- starting number of arrows from point A to red arrows
  (orange_arrows : ℕ) -- number of orange arrows equivalent to green arrows

-- Define the conditions for our specific problem setup
def problem_setup : PathSetup Point Point :=
  {
    red_arrows := 3,
    red_to_blue := 2,
    blue_to_green := 2,
    green_to_orange := 1,
    start_arrows := 3,
    orange_arrows := 6 * 2 * 2 -- derived from blue_to_green and red_to_blue steps
  }

-- Prove the number of unique paths from A to B
theorem bug_paths (setup : PathSetup Point Point) : 
  setup.start_arrows * setup.red_to_blue * setup.blue_to_green * setup.green_to_orange * setup.orange_arrows = 1440 :=
by
  -- Calculations are performed; exact values must hold
  sorry

end bug_paths_l296_296737


namespace min_balls_to_ensure_same_color_min_draws_to_ensure_two_same_color_l296_296158

theorem min_balls_to_ensure_same_color (balls_per_color : ℕ) (colors : ℕ) (H_ball_count : balls_per_color = 6) (H_color_count : colors = 4) : ℕ :=
by
  have h := nat.add_one (colors - 1)
  exact h
-- Question to prove: Minimum number of ball draws needed to ensure at least two balls of the same color considering worst-case draw scenarios
theorem min_draws_to_ensure_two_same_color (H_ball_count : ∀ color : Type, ∃ balls : ℕ, color ≠ color → balls = 6) (colors : Type) : ∃ draws_needed : ℕ, draws_needed = 5 :=
by
  sorry

end min_balls_to_ensure_same_color_min_draws_to_ensure_two_same_color_l296_296158


namespace tan_lt_neg_one_implies_range_l296_296856

theorem tan_lt_neg_one_implies_range {x : ℝ} (h1 : 0 < x) (h2 : x < π) (h3 : Real.tan x < -1) :
  (π / 2 < x) ∧ (x < 3 * π / 4) :=
sorry

end tan_lt_neg_one_implies_range_l296_296856


namespace fraction_of_eggs_hatched_l296_296228

variable (x : ℚ)
variable (survived_first_month_fraction : ℚ := 3/4)
variable (survived_first_year_fraction : ℚ := 2/5)
variable (geese_survived : ℕ := 100)
variable (total_eggs : ℕ := 500)

theorem fraction_of_eggs_hatched :
  (x * survived_first_month_fraction * survived_first_year_fraction * total_eggs : ℚ) = geese_survived → x = 2/3 :=
by 
  intro h
  sorry

end fraction_of_eggs_hatched_l296_296228


namespace shiny_penny_problem_l296_296735

theorem shiny_penny_problem :
  let shiny_pennies := 3
      dull_pennies := 4
      total_pennies := shiny_pennies + dull_pennies
      event_probability := (31 : ℚ) / 35
  in (shiny_pennies = 3) ∧ (dull_pennies = 4) ∧
     (total_pennies = 7) ∧ 
     (event_probability.num = 31) ∧ (event_probability.denom = 35) →
     (31 + 35 = 66) :=
by
  sorry

end shiny_penny_problem_l296_296735


namespace intersection_points_range_l296_296487

noncomputable def f (x : ℝ) : ℝ := x^2 / (2 * x - 2 * Real.log x)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * Real.log x + m * x

theorem intersection_points_range (m : ℝ) :
  (0 < m ∧ m < 1 / 2) ↔ ∃ (roots : Fin 4 → ℝ), ∀ i, f (roots i) = g (roots i) :=
sorry

end intersection_points_range_l296_296487


namespace larger_triangle_perimeter_is_65_l296_296783

theorem larger_triangle_perimeter_is_65 (s1 s2 s3 t1 t2 t3 : ℝ)
  (h1 : s1 = 7) (h2 : s2 = 7) (h3 : s3 = 12)
  (h4 : t3 = 30)
  (similar : t1 / s1 = t2 / s2 ∧ t2 / s2 = t3 / s3) :
  t1 + t2 + t3 = 65 := by
  sorry

end larger_triangle_perimeter_is_65_l296_296783


namespace fraction_exponent_simplification_l296_296684

theorem fraction_exponent_simplification (x : ℕ) (h : x = 3) :
  (∏ k in Finset.range' 2 21, x^k) / (∏ k in Finset.range' 3 19, if k % 3 = 0 then x^k else 1) = x^146 :=
by
  rw h
  sorry

end fraction_exponent_simplification_l296_296684


namespace number_with_conditions_eq_66_l296_296506

theorem number_with_conditions_eq_66 :
  let odd_digits := {1, 3, 5, 7, 9}
  let N : ℕ := 1000
  let start := 100
  ∃ cnt,
    cnt = (finset.univ.filter (λ x : ℕ,
      start ≤ x ∧ x < N ∧
      ∃ (d1 d2 d3 : ℕ), 
        (100 * d1 + 10 * d2 + d3 = x) ∧
        ↑d1 ∈ odd_digits ∧
        ↑d2 ∈ odd_digits ∧
        ↑d3 ∈ odd_digits ∧
        d1 < d2 ∧ d2 < d3
    )).card +
    (finset.univ.filter (λ x : ℕ,
      start ≤ x ∧ x < N ∧
      ∃ (d1 d2 d3 : ℕ), 
        (100 * d1 + 10 * d2 + d3 = x) ∧
        (d3 = 0 ∨ d3 = 5) ∧
        (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3) ∧
        (d3 = 0 ∧ (d1 > d2) ∨ d3 = 5 ∧ (d1 > d2))
    )).card = 66 := by
  sorry

end number_with_conditions_eq_66_l296_296506


namespace problem_solution_l296_296323

theorem problem_solution : (121^2 - 110^2) / 11 = 231 := 
by
  sorry

end problem_solution_l296_296323


namespace sqrt_neg2_sq_l296_296047

theorem sqrt_neg2_sq : Real.sqrt ((-2 : ℝ) ^ 2) = 2 := by
  sorry

end sqrt_neg2_sq_l296_296047


namespace distance_between_lines_is_correct_l296_296063

noncomputable def point := (ℝ × ℝ)
def circle := point × ℝ -- Center and radius

def line (A B C : ℝ) := λ (x y : ℝ), A * x + B * y + C = 0

def tangent_line_through_point (C : circle) (M : point) (l1 : line) : Prop :=
  ∃ (A B C1 C2 : ℝ),
    C1 ≠ C2 ∧
    (∀ x y, (x - fst (fst C))^2 + (y - snd (fst C))^2 = snd C^2 →
      (A * x + B * y + C1 = 0 ∧ A * x + B * y + C2 = 0)) ∧
    (A * (fst M) + B * (snd M) + C1 = 0) ∧
    (∃ a, A = a ∧ B = 3 ∧ l1 = line a 3 (2 * a)) ∧
    (ordinal.mk (|C1 - C2|) = 240 / 100)

def main : Prop :=
  let M := (-2, 4) in
  let C := ((2, 1), 5) in
  let l1 := line 4 3 8 in
  distance_between_lines_is_correct C M l1
  
theorem distance_between_lines_is_correct :
  main :=
sorry

end distance_between_lines_is_correct_l296_296063


namespace limit_AO_as_alpha_zero_l296_296543

variables (a b p : ℝ)

theorem limit_AO_as_alpha_zero (α : ℝ) (h₁ : 0 ≤ α) (h₂ : α ≤ π) :
  ∃ L : ℝ, tendsto (λ α, let O := |α| in O) (nhds 0) (nhds L) ∧
  L = \frac{p * \sqrt{a * b}}{\sqrt{a * b} + \sqrt{(p - a) * (p - b)}} := 
sorry

end limit_AO_as_alpha_zero_l296_296543


namespace ant_collision_probability_l296_296413

theorem ant_collision_probability :
  let total_movements := 3^8 in
  let non_collision_cases := 24 in
  let non_collision_probability := (non_collision_cases : ℝ) / total_movements in
  let collision_probability := 1 - non_collision_probability in
  collision_probability = 0.9963 :=
by {
  -- Definitions
  let total_movements := 3^8,
  let non_collision_cases := 24,
  let non_collision_probability := (non_collision_cases : ℝ) / total_movements,
  let collision_probability := 1 - non_collision_probability,
  
  -- Calculation
  have total_movements_eq : total_movements = 6561 := by norm_num,
  rw total_movements_eq at non_collision_probability,
  have non_collision_probability_eq : non_collision_probability = 24 / 6561 := rfl,
  rw non_collision_probability_eq at collision_probability,
  norm_num at collision_probability,
  exact rfl
}

end ant_collision_probability_l296_296413


namespace integer_solutions_count_l296_296960

theorem integer_solutions_count :
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 :=
by sorry

end integer_solutions_count_l296_296960


namespace range_of_a_l296_296887

open Real

def f (a : ℝ) : ℝ → ℝ := λ x : ℝ, 
  if x ≤ 1 then (a-3)*x + 5 
  else (2*a) / x

theorem range_of_a {a : ℝ} :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) < 0) →
  0 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l296_296887


namespace cost_of_general_admission_l296_296767

theorem cost_of_general_admission (G V : ℕ) (x : ℝ) 
      (h1 : G + V = 320) 
      (h2 : V = G - 148) 
      (h3 : 40 * V + x * G = 7500) : 
      x ≈ 17.35 :=  -- Here, "≈" represents approximate equality within a reasonable tolerance
by 
  sorry

end cost_of_general_admission_l296_296767


namespace esperanzas_gross_monthly_salary_l296_296233

def rent : ℝ := 600
def savings : ℝ := 2000
def food_expenses : ℝ := (3/5) * rent
def mortgage_bill : ℝ := 3 * food_expenses
def taxes : ℝ := (2/5) * savings
def gross_monthly_salary : ℝ := rent + food_expenses + mortgage_bill + savings + taxes

theorem esperanzas_gross_monthly_salary : gross_monthly_salary = 4840 := by
  -- proof steps skipped
  sorry

end esperanzas_gross_monthly_salary_l296_296233


namespace other_carton_racket_count_l296_296637

def num_total_cartons : Nat := 38
def num_total_rackets : Nat := 100
def num_specific_cartons : Nat := 24
def num_rackets_per_specific_carton : Nat := 3

def num_remaining_cartons := num_total_cartons - num_specific_cartons
def num_remaining_rackets := num_total_rackets - (num_specific_cartons * num_rackets_per_specific_carton)

theorem other_carton_racket_count :
  (num_remaining_rackets / num_remaining_cartons) = 2 :=
by
  sorry

end other_carton_racket_count_l296_296637


namespace domain1_correct_domain2_correct_l296_296088

-- Function 1: y = sqrt(-x^2 - 3x + 4) / x
def domain1 (x : ℝ) : Prop :=
  -4 ≤ x ∧ x ≤ 1 ∧ x ≠ 0

-- Function 2: y = 1 / sqrt(log_{0.5}(4x-3))
def domain2 (x : ℝ) : Prop :=
  3 / 4 < x ∧ x < 1

-- The statements we need to prove
theorem domain1_correct (x : ℝ) : domain1 x ↔ (-x^2 - 3x + 4 ≥ 0 ∧ x ≠ 0) :=
by
  sorry

theorem domain2_correct (x : ℝ) : domain2 x ↔ (log (4 * x - 3) / log (0.5) > 0) :=
by
  sorry

end domain1_correct_domain2_correct_l296_296088


namespace part_one_part_two_l296_296932

open Real

-- Define set A
def A : Set ℝ := {t | -3 < t ∧ t < -1}

-- Define set B
def B : Set ℝ := {t | t ≥ 0 ∨ t ≤ -2}

-- Define intersection of A and B
def A_intersect_B : Set ℝ := {t | -3 < t ∧ t ≤ -2}

-- Function g(α)
def g (α m : ℝ) : ℝ := -sin α ^ 2 + m * cos α - 2 * m

-- Define set M based on g(α) and intersection of A and B
def M : Set ℝ := {m | ∀ α ∈ Icc π (3/2 * π), g α m ∈ A_intersect_B }

-- Theorems to prove
theorem part_one : A ∩ B = A_intersect_B := 
by sorry

theorem part_two : M = {m | 1/2 < m ∧ m < 1 } := 
by sorry

end part_one_part_two_l296_296932


namespace distance_between_A_and_B_l296_296087

-- Define the two points in 3D space
def pointA : ℝ × ℝ × ℝ := (3, 2, -5)
def pointB : ℝ × ℝ × ℝ := (7, 10, -6)

-- Define the function to calculate the 3D distance
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

-- Define a theorem to state the distance between pointA and pointB is 9
theorem distance_between_A_and_B : distance pointA pointB = 9 :=
by
  -- Proof details would go here
  sorry

end distance_between_A_and_B_l296_296087


namespace persons_next_to_Boris_l296_296821

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l296_296821


namespace total_texts_received_l296_296850

structure TextMessageScenario :=
  (textsBeforeNoon : Nat)
  (textsAtNoon : Nat)
  (textsAfterNoonDoubling : (Nat → Nat) → Nat)
  (textsAfter6pm : (Nat → Nat) → Nat)

def textsBeforeNoon := 21
def textsAtNoon := 2

-- Calculation for texts received from noon to 6 pm
def noonTo6pmTexts (textsAtNoon : Nat) : Nat :=
  let rec doubling (n : Nat) : Nat := match n with
    | 0 => textsAtNoon
    | n + 1 => 2 * (doubling n)
  (doubling 0) + (doubling 1) + (doubling 2) + (doubling 3) + (doubling 4) + (doubling 5)

def textsAfterNoonDoubling : (Nat → Nat) → Nat := λ doubling => noonTo6pmTexts 2

-- Calculation for texts received from 6 pm to midnight
def after6pmTexts (textsAt6pm : Nat) : Nat :=
  let rec decrease (n : Nat) : Nat := match n with
    | 0 => textsAt6pm
    | n + 1 => (decrease n) - 5
  (decrease 0) + (decrease 1) + (decrease 2) + (decrease 3) + (decrease 4) + (decrease 5) + (decrease 6)

def textsAfter6pm : (Nat → Nat) → Nat := λ decrease => after6pmTexts 64

theorem total_texts_received : textsBeforeNoon + (textsAfterNoonDoubling (λ x => x)) + (textsAfter6pm (λ x => x)) = 490 := by
  sorry
 
end total_texts_received_l296_296850


namespace measure_of_sixth_angle_l296_296531

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

def given_angles : list ℝ := [130, 95, 115, 120, 110]

def total_sum (angles : list ℝ) : ℝ := angles.sum

theorem measure_of_sixth_angle (Q : ℝ) : 
  total_sum given_angles + Q = sum_of_interior_angles 6 → Q = 150 :=
by
  sorry

end measure_of_sixth_angle_l296_296531


namespace Johnson_family_seating_l296_296258

theorem Johnson_family_seating : 
  ∃ n : ℕ, number_of_ways_to_seat_Johnson_family = n ∧ n = 288 :=
sorry

end Johnson_family_seating_l296_296258


namespace solve_for_xy_l296_296509

theorem solve_for_xy (x y : ℕ) : 
  (4^x / 2^(x + y) = 16) ∧ (9^(x + y) / 3^(5 * y) = 81) → x * y = 32 :=
by
  sorry

end solve_for_xy_l296_296509


namespace enclosed_area_is_2point28_l296_296218

noncomputable def g (x : ℝ) : ℝ := 2 - real.sqrt (4 - x^2)

theorem enclosed_area_is_2point28 (ε : ℝ) (hε : 0 < ε ∧ ε < 1) :
  ∃ a, abs (a - 2.28) < ε ∧ 
  ∀ x (h₁ : -2 ≤ x) (h₂ : x ≤ 2), 
  true := 
sorry

end enclosed_area_is_2point28_l296_296218


namespace total_prize_money_l296_296365

theorem total_prize_money (prizes : Fin 18 → ℕ) 
  (h1 : prizes 0 = 200) 
  (h2 : prizes 1 = 150) 
  (h3 : prizes 2 = 120) 
  (h_rest : ∀ n : Fin 15, prizes (n.castAdd 3) = 22) :
  (∑ i, prizes i) = 800 :=
by
  sorry

end total_prize_money_l296_296365


namespace circumsphere_radius_of_tetrahedron_l296_296988

theorem circumsphere_radius_of_tetrahedron
  (A B C D : Point)
  (AD_eq : dist A D = 2 * Real.sqrt 3)
  (angle_BAC_eq : ∠ B A C = Real.pi / 3)
  (angle_BAD_eq : ∠ B A D = Real.pi / 4)
  (angle_CAD_eq : ∠ C A D = Real.pi / 4)
  (inner_sphere_radius_eq : let inner_sphere_radius := 1) :
  ∃ r : ℝ, r = 3 := sorry

end circumsphere_radius_of_tetrahedron_l296_296988


namespace acute_angled_triangle_condition_l296_296554

theorem acute_angled_triangle_condition (a b c : ℝ) (A B C : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : A > 0) (h5 : A < π) (h6 : B > 0) (h7 : B < π) (h8 : C > 0) (h9 : C < π)
  (h10 : a / sin A = b / sin B)
  (h11 : b / sin B = c / sin C)
  (h12 : (sin A)^2 + (sin B)^2 - (sin C)^2 > 0)
  (h13 : (sin B)^2 + (sin C)^2 - (sin A)^2 > 0)
  (h14 : (sin C)^2 + (sin A)^2 - (sin B)^2 > 0) : 
  (A < π/2) ∧ (B < π/2) ∧ (C < π/2) :=
sorry

end acute_angled_triangle_condition_l296_296554


namespace log_expression_value_l296_296727

theorem log_expression_value :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 :=
by
  sorry

end log_expression_value_l296_296727


namespace largest_number_Ahn_can_get_l296_296376

theorem largest_number_Ahn_can_get :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (100 ≤ m ∧ m ≤ 999) → 3 * (500 - m) ≤ 1200) := sorry

end largest_number_Ahn_can_get_l296_296376


namespace calligraphy_competition_l296_296660

variable {Ω : Type} [ProbabilitySpace Ω]

-- Definitions for the conditions
variable (A B : Event Ω)
variable (x y : ℕ) -- number of first, second and third prize works (second and third are same)

-- Distribution properties given in the problem
variable (P_A_sen : ℝ := 0.4) -- 40% senior year first prize works
variable (P_AB : ℝ := 0.16) -- P(A and B) = 0.16

-- The goal is to prove P(A | B) = 8 / 23
theorem calligraphy_competition :
  (number_second_eq_third : x = y) →
  (P_A_sen : \frac{0.4 * x}{x + 2 * y} = 0.16) →
  (cond_dist : P(A | B) = 8 / 23) :=
sorry

end calligraphy_competition_l296_296660


namespace igor_painted_all_cells_l296_296977

theorem igor_painted_all_cells :
  ∀ (m n : ℕ), (0 < m ∧ m ≤ 4) →
               (0 < n ∧ n ≤ 3) →
               (3 * m = 4 * n) →
               (∀ i j, 0 ≤ i ∧ i < 6 ∧ 0 ≤ j ∧ j < 6 → painted i j) :=
by
  sorry

end igor_painted_all_cells_l296_296977


namespace max_regular_hours_l296_296739

theorem max_regular_hours (reg_rate : ℕ) (ot_rate : ℕ) (total_compensation : ℝ) 
  (total_hours : ℝ) (H : ℝ) :
  reg_rate = 18 →
  ot_rate = 18 * 1.75 →
  total_compensation = 976 →
  total_hours = 48.12698412698413 →
  total_compensation = reg_rate * H + ot_rate * (total_hours - H) →
  H = 40 :=
by
  intros reg_rate_18 ot_rate_1.75 comp_976 tot_hours_48.12698412698413 eq_total_comp
  sorry

end max_regular_hours_l296_296739


namespace esperanza_gross_salary_l296_296231

def rent : ℕ := 600
def food_expenses (rent : ℕ) : ℕ := 3 * rent / 5
def mortgage_bill (food_expenses : ℕ) : ℕ := 3 * food_expenses
def savings : ℕ := 2000
def taxes (savings : ℕ) : ℕ := 2 * savings / 5
def total_expenses (rent food_expenses mortgage_bill taxes : ℕ) : ℕ :=
  rent + food_expenses + mortgage_bill + taxes
def gross_salary (total_expenses savings : ℕ) : ℕ :=
  total_expenses + savings

theorem esperanza_gross_salary : 
  gross_salary (total_expenses rent (food_expenses rent) (mortgage_bill (food_expenses rent)) (taxes savings)) savings = 4840 :=
by
  sorry

end esperanza_gross_salary_l296_296231


namespace estimate_diagonal_length_l296_296105

noncomputable def length : ℝ := 3
noncomputable def width : ℝ := 2

theorem estimate_diagonal_length : 
  let d := Real.sqrt (length ^ 2 + width ^ 2) in 
  3.6 ≤ d ∧ d ≤ 3.7 := 
by
  sorry

end estimate_diagonal_length_l296_296105


namespace value_of_m_l296_296474

theorem value_of_m (m : ℝ) :
  (∀ A B : ℝ × ℝ, A = (m + 1, -2) → B = (3, m - 1) → (A.snd = B.snd) → m = -1) :=
by
  intros A B hA hB h_parallel
  -- Apply the given conditions and assumptions to prove the value of m.
  sorry

end value_of_m_l296_296474


namespace four_person_apartments_l296_296380

theorem four_person_apartments : 
  ∃ x : ℕ, 
    (4 * (10 + 20 * 2 + 4 * x)) * 3 / 4 = 210 → x = 5 :=
by
  sorry

end four_person_apartments_l296_296380


namespace sum_of_c_with_4_solutions_l296_296060

def g (x : ℝ) : ℝ := ((x - 4) * (x - 2) * x * (x + 2) * (x + 4)) / 120 + 2

theorem sum_of_c_with_4_solutions :
  (finset.sum (finset.filter (λ c, ∃! (x : ℝ), -5 ≤ x ∧ x ≤ 5 ∧ g x = c) (finset.range 101).image (λ c, (c - 50))) id) = 2 :=
sorry

end sum_of_c_with_4_solutions_l296_296060


namespace smallest_positive_multiple_l296_296322

theorem smallest_positive_multiple (a : ℕ) :
  (37 * a) % 97 = 7 → 37 * a = 481 :=
sorry

end smallest_positive_multiple_l296_296322


namespace inequality_problem_l296_296102

theorem inequality_problem {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a = a + b + c) :
  a^2 + b^2 + c^2 + 2 * a * b * c ≥ 5 :=
sorry

end inequality_problem_l296_296102


namespace geometric_seq_problem_l296_296174

theorem geometric_seq_problem (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a (n + 1) = a n * r) 
    (h₄ : a 4 = 5) : a 1 * a 7 = 25 :=
begin
  sorry
end

end geometric_seq_problem_l296_296174


namespace correct_conclusions_l296_296023

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem correct_conclusions :
  (∃ (a b : ℝ), a < b ∧ f a < f b ∧ ∀ x, a < x ∧ x < b → f x < f (x+1)) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = (x₁ - 2012) ∧ f x₂ = (x₂ - 2012)) :=
by
  sorry

end correct_conclusions_l296_296023


namespace find_functions_l296_296427

open Function

theorem find_functions (f g : ℚ → ℚ) :
  (∀ x y : ℚ, f (g x - g y) = f (g x) - y) →
  (∀ x y : ℚ, g (f x - f y) = g (f x) - y) →
  ∃ c : ℚ, c ≠ 0 ∧ (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) :=
by
  sorry

end find_functions_l296_296427


namespace all_7_digit_sequences_divisible_239_l296_296623

theorem all_7_digit_sequences_divisible_239 : 
  (∀ x, 0 ≤ x ∧ x < 10^7 → (10^7 : ℤ) ≡ 1 [MOD 239]) →
  let seq_sum := ∑ A in finset.range (10^7), ∑ B in finset.range (10^7), (A : ℤ) * 10^7 + B
  in seq_sum % 239 = 0 :=
by
  sorry

end all_7_digit_sequences_divisible_239_l296_296623


namespace team_A_wins_at_least_5_matches_l296_296535

open Classical

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

theorem team_A_wins_at_least_5_matches :
  let p := 1 / 2 in
  let n := 7 in
  (binomial_probability n 5 p + binomial_probability n 6 p + binomial_probability n 7 p) = 29 / 128 :=
by
  sorry

end team_A_wins_at_least_5_matches_l296_296535


namespace card_at_58_is_6_l296_296558

theorem card_at_58_is_6 : 
  ∀ (s : ℕ → ℕ), 
  (∀ n, s n = ((n - 1) % 13 + 1)) → 
  s 58 = 6 :=
by
  intro s hs
  have h : s 58 = ((58 - 1) % 13 + 1), from hs 58
  change ((58 - 1) % 13 + 1) = 6 at h
  have : (57 % 13) = 5, from Nat.mod_eq_of_lt (show 57 < 65, by norm_num)
  rw this at h
  norm_num at h
  exact h


end card_at_58_is_6_l296_296558


namespace students_odd_l296_296837

-- Given: list of competitions
structure Competition :=
  (math physics chemistry biology ballroom : ℕ)

-- Given: list of students and their participation
structure StudentList :=
  (participants : list (Competition -> ℕ))
  (total_students : ℕ)

-- Define a condition: each competition has an odd number of participants
def odd_competitions (c : Competition) : Prop :=
  c.math % 2 = 1 ∧ c.physics % 2 = 1 ∧ c.chemistry % 2 = 1 ∧ c.biology % 2 = 1 ∧ c.ballroom % 2 = 1

-- Define a condition: each student participated in an odd number of competitions
def odd_participations (sl : StudentList) : Prop :=
  ∀ p, p ∈ sl.participants -> (p.f math + p.f physics + p.f chemistry + p.f biology + p.f ballroom) % 2 = 1

-- Main theorem statement
theorem students_odd (sl : StudentList) : 
  (∀ c, odd_competitions c) → (odd_participations sl) → sl.total_students % 2 = 1 :=
by
  sorry

end students_odd_l296_296837


namespace percentage_increase_of_gross_l296_296755

theorem percentage_increase_of_gross
  (P R : ℝ)
  (price_drop : ℝ := 0.20)
  (quantity_increase : ℝ := 0.60)
  (original_gross : ℝ := P * R)
  (new_price : ℝ := (1 - price_drop) * P)
  (new_quantity_sold : ℝ := (1 + quantity_increase) * R)
  (new_gross : ℝ := new_price * new_quantity_sold)
  (percentage_increase : ℝ := ((new_gross - original_gross) / original_gross) * 100) :
  percentage_increase = 28 :=
by
  sorry

end percentage_increase_of_gross_l296_296755


namespace square_of_cube_plus_11_l296_296681

def third_smallest_prime : ℕ := 5

theorem square_of_cube_plus_11 : (third_smallest_prime ^ 3)^2 + 11 = 15636 := by
  -- We will provide a proof later
  sorry

end square_of_cube_plus_11_l296_296681


namespace derivative_at_zero_dne_l296_296386

noncomputable def f : ℝ → ℝ
| x := if x ≠ 0 then (Real.sin x * Real.cos (5 / x)) else 0

theorem derivative_at_zero_dne : ¬(DifferentiableAt ℝ f 0) :=
sorry

end derivative_at_zero_dne_l296_296386


namespace a_share_in_gain_l296_296775

noncomputable def investment_share (x: ℝ) (total_gain: ℝ): ℝ := 
  let a_interest := x * 0.1
  let b_interest := (2 * x) * (7 / 100) * (1.5)
  let c_interest := (3 * x) * (10 / 100) * (1.33)
  let total_interest := a_interest + b_interest + c_interest
  a_interest

theorem a_share_in_gain (total_gain: ℝ) (a_share: ℝ) (x: ℝ)
  (hx: 0.709 * x = total_gain):
  investment_share x total_gain = a_share :=
sorry

end a_share_in_gain_l296_296775


namespace g_explicit_expression_inequality_solution_monotonically_increasing_l296_296920

-- Definitions of the functions
def f (x : ℝ) := x^2 + 2 * x
def g (x : ℝ) := -x^2 + 2 * x

-- Proof statement for the first question
theorem g_explicit_expression (x : ℝ) : g(x) = -x^2 + 2 * x :=
sorry

-- Proof statement for the second question
theorem inequality_solution (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1/2) : 
  g(x) ≥ f(x) - |x - 1| :=
sorry

-- Proof statement for the third question
theorem monotonically_increasing (λ : ℝ) (hλ : λ ≤ 0) : 
  ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), 
  deriv (λ x : ℝ, g(x) - λ * f(x) + 1) x ≥ 0 :=
sorry

end g_explicit_expression_inequality_solution_monotonically_increasing_l296_296920


namespace predicted_temperature_l296_296067

-- Define the observation data points
def data_points : List (ℕ × ℝ) :=
  [(20, 25), (30, 27.5), (40, 29), (50, 32.5), (60, 36)]

-- Define the linear regression equation with constant k
def regression (x : ℕ) (k : ℝ) : ℝ :=
  0.25 * x + k

-- Proof statement
theorem predicted_temperature (k : ℝ) (h : regression 40 k = 30) : regression 80 k = 40 :=
by
  sorry

end predicted_temperature_l296_296067


namespace dependent_variable_is_temperature_l296_296549

-- Define the variables involved in the problem
variables (intensity_of_sunlight : ℝ)
variables (temperature_of_water : ℝ)
variables (duration_of_exposure : ℝ)
variables (capacity_of_heater : ℝ)

-- Define the conditions
def changes_with_duration (temp: ℝ) (duration: ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∀ d, temp = f d) ∧ ∀ d₁ d₂, d₁ ≠ d₂ → f d₁ ≠ f d₂

-- The theorem we need to prove
theorem dependent_variable_is_temperature :
  changes_with_duration temperature_of_water duration_of_exposure → 
  (∀ t, ∃ d, temperature_of_water = t → duration_of_exposure = d) :=
sorry

end dependent_variable_is_temperature_l296_296549


namespace ice_formation_l296_296300

-- Definitions for problem setup
variable {T_in T_out T_critical : ℝ}
variable T_inside_surface : ℝ -- The temperature on the inside surface of the trolleybus windows

-- Given conditions
def is_humid (T_in : ℝ) : Prop := true -- 100% humidity condition (always true for this problem)
def condensation_point (T : ℝ) : Prop := T < 0 -- Temperature below which condensation occurs

-- Condition that the inner surface temperature is determined by both inside and outside temperatures
def inner_surface_temp (T_in T_out : ℝ) : ℝ := (T_in + T_out) / 2  -- Simplified model

theorem ice_formation 
  (h_humid : is_humid T_in)
  (h_T_in : T_in < T_critical)
  : condensation_point (inner_surface_temp T_in T_out) :=
sorry

end ice_formation_l296_296300


namespace enrique_speed_l296_296415

theorem enrique_speed (distance : ℝ) (time : ℝ) (speed_diff : ℝ) (E : ℝ) :
  distance = 200 ∧ time = 8 ∧ speed_diff = 7 ∧ 
  (2 * E + speed_diff) * time = distance → 
  E = 9 :=
by
  sorry

end enrique_speed_l296_296415


namespace add_A_to_10_eq_15_l296_296526

theorem add_A_to_10_eq_15 (A : ℕ) (h : A + 10 = 15) : A = 5 :=
sorry

end add_A_to_10_eq_15_l296_296526


namespace angle_between_diagonals_correct_l296_296646

noncomputable def angle_between_diagonals (a b c : ℝ) : ℝ :=
  let cos_alpha := (b^2 - a^2 - c^2) / (a^2 + b^2 + c^2) in
  if a^2 ≤ b^2 + c^2 then 
    real.arccos cos_alpha 
  else 
    180 - real.arccos cos_alpha

theorem angle_between_diagonals_correct (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let α := angle_between_diagonals a b c in
  (a^2 ≤ b^2 + c^2 → α = real.arccos ((b^2 - a^2 - c^2) / (a^2 + b^2 + c^2))) ∧
  (a^2 > b^2 + c^2 → α = 180 - real.arccos ((b^2 - a^2 - c^2) / (a^2 + b^2 + c^2))) :=
sorry

end angle_between_diagonals_correct_l296_296646


namespace G_is_centroid_of_triangle_l296_296180

theorem G_is_centroid_of_triangle
  {A B C A' B' C' G : Point}
  (hA' : lies_on_line A' (line_through B C))
  (hB' : lies_on_line B' (line_through A C))
  (hC' : lies_on_line C' (line_through A B))
  (h_concurrent : concurrent [line_through A A', line_through B B', line_through C C'] G)
  (h_ratios : (distance A G) / (distance G A') = 
              (distance B G) / (distance G B') ∧
              (distance B G) / (distance G B') = 
              (distance C G) / (distance G C')) :
  is_centroid G (triangle A B C) :=
sorry

end G_is_centroid_of_triangle_l296_296180


namespace intersection_point_l296_296267

theorem intersection_point (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : x + y - 3 = 0) : x = 2 ∧ y = 1 :=
by
  sorry

end intersection_point_l296_296267


namespace y_range_l296_296131

noncomputable def y : ℝ := 
  let f : ℝ → ℝ := λ x, sqrt (4 + x)
  limit (λ n, (nat.iterate f n) 0) -- Assuming the nested form converges

lemma y_eq_recursive : y = sqrt (4 + y) := 
  -- Assuming the limit defined above
  sorry

theorem y_range : y = (1 + Real.sqrt 17) / 2 := by
  have h : y^2 = 4 + y := by sorry
  have h_quad : y^2 - y - 4 = 0 := by linarith
  have h_solutions := quadratic_formula 1 (-1) (-4)
  cases h_solutions with y1 y2,
  rw h_solutions at h,
  interval_cases y using h,
  { exact sorry } -- Prove y must be non-negative and equal to the positive root
  { exact sorry } -- Discard the negative root as invalid due to the non-negativity constraint of y

end y_range_l296_296131


namespace parabola_focus_directrix_point_P_coordinates_l296_296465

noncomputable def parabola (p : ℝ) : Type :=
{ eq : ℝ → ℝ → Prop // ∃ F : ℝ × ℝ, ∃ l : ℝ, ∀ x y, eq x y ↔ x^2 = 2*p*y }

noncomputable def circle : Type :=
{ eq : ℝ → ℝ → Prop // ∃ E : ℝ × ℝ, ∀ x y, eq x y ↔ (x^2 + (y - 4)^2 = 1) }

theorem parabola_focus_directrix (C : parabola 2) (F : ℝ × ℝ) 
(hl : F = (0, 1)) : 
  ∀ x y, C.eq x y ↔ x^2 = 4*y :=
sorry

theorem point_P_coordinates (C : parabola 2) (P : ℝ × ℝ) 
(hPQ_symmetry : ∀ x y, C.eq x y → C.eq (-x) y) 
(hE : circle) (MN_parallel_Q : ∀ x y, C.eq x y → x = y):
  P = (0, 0) ∨ P = (4, 4) ∨ P = (-4, 4) :=
sorry

end parabola_focus_directrix_point_P_coordinates_l296_296465


namespace find_100_digit_number_l296_296349

-- Main statement to be proved in Lean 4
theorem find_100_digit_number (b : ℕ) (hb : b ∈ {1, 2, 3}) :
    ∃ (N : ℕ), N = 325 * b * 10^97 ∧ (∃ (M : ℕ), M = N / 13 ∧ N - M*13 = 0) :=
by
    sorry -- Proof to be completed

end find_100_digit_number_l296_296349


namespace descent_phase_duration_l296_296076

noncomputable def start_time_in_seconds : ℕ := 45 * 60 + 39
noncomputable def end_time_in_seconds : ℕ := 47 * 60 + 33

theorem descent_phase_duration :
  end_time_in_seconds - start_time_in_seconds = 114 := by
  sorry

end descent_phase_duration_l296_296076


namespace rohan_house_rent_percentage_l296_296248

theorem rohan_house_rent_percentage :
  let salary : ℕ := 7500
  let food_percentage : ℝ := 0.40
  let entertainment_percentage : ℝ := 0.10
  let conveyance_percentage : ℝ := 0.10
  let savings : ℝ := 1500
  let savings_percentage := savings / salary * 100
  let total_percentage := food_percentage * 100 + entertainment_percentage * 100 + conveyance_percentage * 100 + savings_percentage

  total_percentage + (?H : ℝ) * 100 = 100 →
  (?H : ℝ) = 0.20 :=
by
  sorry

end rohan_house_rent_percentage_l296_296248


namespace heartsuit_example_l296_296512

def heartsuit (x y: ℤ) : ℤ := 4 * x + 6 * y

theorem heartsuit_example : heartsuit 3 8 = 60 :=
by
  sorry

end heartsuit_example_l296_296512


namespace esperanzas_gross_monthly_salary_l296_296234

def rent : ℝ := 600
def savings : ℝ := 2000
def food_expenses : ℝ := (3/5) * rent
def mortgage_bill : ℝ := 3 * food_expenses
def taxes : ℝ := (2/5) * savings
def gross_monthly_salary : ℝ := rent + food_expenses + mortgage_bill + savings + taxes

theorem esperanzas_gross_monthly_salary : gross_monthly_salary = 4840 := by
  -- proof steps skipped
  sorry

end esperanzas_gross_monthly_salary_l296_296234


namespace t_range_l296_296862

noncomputable def exists_nonneg_real_numbers_satisfying_conditions (t : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
  (3 * x^2 + 3 * z * x + z^2 = 1) ∧ 
  (3 * y^2 + 3 * y * z + z^2 = 4) ∧ 
  (x^2 - x * y + y^2 = t)

theorem t_range : ∀ t : ℝ, exists_nonneg_real_numbers_satisfying_conditions t → 
  (t ≥ (3 - Real.sqrt 5) / 2 ∧ t ≤ 1) :=
sorry

end t_range_l296_296862


namespace eldest_child_age_l296_296756

variables (y m e : ℕ)

theorem eldest_child_age (h1 : m = y + 3)
                        (h2 : e = 3 * y)
                        (h3 : e = y + m + 2) : e = 15 :=
by
  sorry

end eldest_child_age_l296_296756


namespace ratio_of_populations_l296_296842

variable (X Y Z : ℕ)

-- Conditions
def condition1 : Prop := X = 8 * Y
def condition2 : Prop := Y = 2 * Z

-- Proof Statement
theorem ratio_of_populations (h1 : condition1 X Y Z) (h2 : condition2 X Y Z) : X / Z = 16 :=
by
  sorry

end ratio_of_populations_l296_296842


namespace rhombus_perimeter_l296_296764

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) :
  let s := (d1 / 2) ^ 2 + (d2 / 2) ^ 2
  let side := real.sqrt s
  in 4 * side = 4 * real.sqrt 61 :=
by
  sorry

end rhombus_perimeter_l296_296764


namespace range_of_power_function_l296_296921

theorem range_of_power_function (k : ℝ) (h : k < 0) : 
  set.range (λ x : ℝ, x^k) ∩ set.Ici 2 = set.Ioc 0 (2^k) := 
sorry

end range_of_power_function_l296_296921


namespace polynomial_divisible_by_6_l296_296929

theorem polynomial_divisible_by_6
  (n : ℕ)
  (a : ℕ → ℤ)
  (f : ℤ → ℤ)
  (Hf_def : ∀ x, f x = ∑ i in finset.range (n+1), a i * x ^ (n - i))
  (H_f2_div : f 2 % 6 = 0)
  (H_f3_div : f 3 % 6 = 0) :
  f 5 % 6 = 0 :=
sorry

end polynomial_divisible_by_6_l296_296929


namespace skew_lines_pairs_count_l296_296890

-- Define the rectangular cuboid and the 12 specific lines.
noncomputable def rectangularCuboid :=
  let points := {"A", "B", "C", "D", "A'", "B'", "C'", "D'"} in
  let lines := {"AB'", "BA'", "CD'", "DC'", "AD'", "DA'", "BC'", "CB'", "AC", "A'C'", "BD", "B'D'"} in
  (points, lines)

-- Define the predicate that checks if two lines are skew.
def are_skew (line1 line2 : String) : Prop :=
  -- Assume a function that checks if line1 and line2 are skew (not parallel and not intersecting).
  sorry

-- The theorem that we need to prove:
theorem skew_lines_pairs_count :
  let (points, lines) := rectangularCuboid in
  ∃ n, n = 30 ∧ ∀ (l₁ l₂ : String) (h₁ : l₁ ∈ lines) (h₂ : l₂ ∈ lines), are_skew l₁ l₂ → n = 30 :=
by
  sorry

end skew_lines_pairs_count_l296_296890


namespace volume_KLMN_l296_296658

-- Defining the problem
variables (ABCD KLMN : Type) [tetrahedron ABCD] [tetrahedron KLMN]
variable (V : ℝ)
-- Conditions
axiom sides_perpendicular (ABCD KLMN : Type) [tetrahedron ABCD] [tetrahedron KLMN] : sides_perpendicular KLMN ABCD 
axiom sides_equal_areas (ABCD KLMN : Type) [tetrahedron ABCD] [tetrahedron KLMN] : sides_equal_areas KLMN ABCD
axiom volume_ABCD (ABCD : Type) [tetrahedron ABCD] : volume ABCD = V

-- The required proof statement
theorem volume_KLMN (ABCD KLMN : Type) [tetrahedron ABCD] [tetrahedron KLMN]
  (h1 : sides_perpendicular ABCD KLMN)
  (h2 : sides_equal_areas ABCD KLMN)
  (h3 : volume ABCD = V) :
  volume KLMN = (3 / 4) * V := 
sorry

end volume_KLMN_l296_296658


namespace area_of_triangle_BQW_l296_296173

-- Declare the known conditions as definitions in Lean
def rectangle (ABCD : Type) (A B C D : ABCD) (AB : ℝ) (AZ WC : ℝ) (area_ZWCD : ℝ) :=
  (AB = 16) ∧ (AZ = 8) ∧ (WC = 8) ∧ (area_ZWCD = 192)

-- Declare the theorem to be proved
theorem area_of_triangle_BQW (ABCD : Type) (A B C D Z W Q : ABCD) (AB AD AZ WC : ℝ) (area_ZWCD area_ABCD : ℝ)
  (h_rect : rectangle ABCD A B C D AB AZ WC area_ZWCD)
  (midpoint : midpoint Z W Q) :
  area_of_triangle B Q W = 0 := 
sorry

end area_of_triangle_BQW_l296_296173


namespace compare_costs_l296_296445

def cost_X (copies: ℕ) : ℝ :=
  if copies >= 40 then
    (copies * 1.25) * 0.95
  else
    copies * 1.25

def cost_Y (copies: ℕ) : ℝ :=
  if copies >= 100 then
    copies * 2.00
  else if copies >= 60 then
    copies * 2.25
  else
    copies * 2.75

def cost_Z (copies: ℕ) : ℝ :=
  if copies >= 50 then
    (copies * 3.00) * 0.90
  else
    copies * 3.00

def cost_W (copies: ℕ) : ℝ :=
  let bulk_groups := copies / 25
  let remainder := copies % 25
  (bulk_groups * 40) + (remainder * 2.00)

theorem compare_costs : 
  cost_X 60 < cost_Y 60 ∧ 
  cost_X 60 < cost_Z 60 ∧ 
  cost_X 60 < cost_W 60 ∧
  cost_Y 60 - cost_X 60 = 63.75 ∧
  cost_Z 60 - cost_X 60 = 90.75 ∧
  cost_W 60 - cost_X 60 = 28.75 :=
  sorry

end compare_costs_l296_296445


namespace triangle_smallest_circle_radius_l296_296468

theorem triangle_smallest_circle_radius :
  ∃ (r : ℝ), let s1 := 2, s2 := 3, s3 := 4 in 
  (r = s3 / 2) := 
begin
  use 2,
  sorry
end

end triangle_smallest_circle_radius_l296_296468


namespace num_integers_satisfying_inequality_l296_296950

theorem num_integers_satisfying_inequality : 
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 := 
by 
  sorry

end num_integers_satisfying_inequality_l296_296950


namespace smallest_positive_period_min_value_of_f_l296_296913

noncomputable def f (x m : ℝ) : ℝ :=
  2 * Math.cos x * (Real.sqrt 3 * Math.sin x + Math.cos x) + m

theorem smallest_positive_period (m : ℝ) : ∃ p > 0, ∀ x, f (x + p) m = f x m := by
  use π
  intros
  simp
  sorry

theorem min_value_of_f (m : ℝ) (x : ℝ) (h_max : ∀ x ∈ Icc 0 (π / 2), f x m ≤ 6) :
  f (π / 2) m = 2 := by
  sorry

end smallest_positive_period_min_value_of_f_l296_296913


namespace problem_statement_l296_296508

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem problem_statement :
  let x := log_base 4 3 in
  (2^x - 2^(-x))^2 = 4 / 3 :=
by
  sorry

end problem_statement_l296_296508


namespace log_half_not_increasing_identify_error_in_reasoning_l296_296283

theorem log_half_not_increasing :
  ∀ x : ℝ, x > 0 → log (1/2) x ≠ log (1/2) (e ^ log (1/2) x) :=
by
  intro x hx
  have h1 : (1/2) < 1 := by norm_num
  have h2 : log (1/2) (e ^ log (1/2) x) = x := by sorry -- using log and exp properties 
  sorry

-- Proof verifying that the function y = log_{1/2} x is not increasing part of equivalent proof
theorem identify_error_in_reasoning (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  a ≠ 1/2 → ∃ x : ℝ, x > 0 ∧ log (1/2) x ≠ log a x :=
by
  intro ha
  use 1
  split
  · norm_num
  · have : log a 1 = 0 := by rw [log_one, mul_zero]
    have : log (1/2) 1 ≠ log a 1 := by
      rw [log_def, log_def]
      intro h
      have := (rpow_eq_rpow a x 2).mpr
      sorry
  sorry

#check identify_error_in_reasoning

end log_half_not_increasing_identify_error_in_reasoning_l296_296283


namespace tangent_line_at_point_l296_296431

noncomputable def curve (x : ℝ) : ℝ := x / (x + 1)

theorem tangent_line_at_point :
  let x₀ : ℝ := 1
  let y₀ : ℝ := 1 / 2
  let line := λ x y : ℝ, x - 4 * y + 1 = 0
  line x₀ (curve x₀) :=
by
  sorry

end tangent_line_at_point_l296_296431


namespace difference_of_interchanged_digits_l296_296644

theorem difference_of_interchanged_digits {x y : ℕ} (h : x - y = 4) :
  (10 * x + y) - (10 * y + x) = 36 :=
by sorry

end difference_of_interchanged_digits_l296_296644


namespace who_is_next_to_boris_l296_296812

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l296_296812


namespace prove_positive_intervals_l296_296455

def expression (a : ℝ) : ℝ :=
  a + (-1 + 9 * a + 4 * a^2) / (a^2 - 3 * a - 10)

theorem prove_positive_intervals (a : ℝ) :
  a + (-1 + 9*a + 4*a^2) / (a^2 - 3*a - 10) > 0 ↔ a ∈ Set.Ioo (Real.ofRat (-2)) (Real.ofRat (-1)) ∪ Set.Ioo (Real.ofRat (-1)) (Real.ofRat 1) ∪ Set.Ioi (Real.ofRat 5) :=
sorry

end prove_positive_intervals_l296_296455


namespace problem1_problem2_problem3_l296_296889

-- Define the general distance formula
def distance_point_to_line (x0 y0 k b : ℝ) : ℝ :=
  |k * x0 - y0 + b| / (Real.sqrt (1 + k^2))

-- Problem 1: Distance from origin to line y = x + 1
theorem problem1 : distance_point_to_line 0 0 1 1 = Real.sqrt 2 / 2 := by
  sorry

-- Problem 2: Distance from P(1, 1) to the line y = x + b is 1
theorem problem2 (b : ℝ) (h : distance_point_to_line 1 1 1 b = 1) : b = Real.sqrt 2 ∨ b = -Real.sqrt 2 := by
  sorry

-- Problem 3: Range of b given distance between two parallel lines
theorem problem3 (d : ℝ) (h : Real.sqrt 2 ≤ d ∧ d ≤ 2 * Real.sqrt 2) : 
  (3 ≤ d * Real.sqrt 2 + 1 ∧ d * Real.sqrt 2 + 1 ≤ 5) ∨ 
  (-3 ≤ d * Real.sqrt 2 - 1 ∧ d * Real.sqrt 2 - 1 ≤ -1) := by
  sorry

end problem1_problem2_problem3_l296_296889


namespace only_solution_l296_296855

def is_solution (n p : ℕ) : Prop :=
  (- n : ℤ) ≤ 2 * p ∧
  (- p : ℤ).minFac = (- p : ℤ) ∧
  ∃ (k : ℕ), (p - 1) ^ n + 1 = k * n ^ (p - 1)

theorem only_solution : is_solution 3 3 :=
by 
  have h_minus_p_prime : (- 3 : ℤ).minFac = (- 3 : ℤ) := sorry
  have h_ineq : (- 3 : ℤ) ≤ 2 * 3 := by norm_num
  have h_div : ∃ (k : ℕ), (3 - 1) ^ 3 + 1 = k * 3 ^ (3 - 1) := by
    use 1
    norm_num
  have solution : is_solution 3 3 := 
    ⟨h_ineq, h_minus_p_prime, h_div⟩
  exact solution

end only_solution_l296_296855


namespace sum_of_roots_l296_296396

theorem sum_of_roots : 
  let f := λ x : ℝ, (3 * x + 2) * (x - 5) + (3 * x + 2) * (x - 7)
  in (f (-2 / 3) = 0 ∧ f 6 = 0) → -2 / 3 + 6 = 16 / 3 :=
by
  let f := λ x : ℝ, (3 * x + 2) * (x - 5) + (3 * x + 2) * (x - 7)
  assume h : f (-2 / 3) = 0 ∧ f 6 = 0
  sorry

end sum_of_roots_l296_296396


namespace isosceles_triangle_l296_296601

variables {A B C M N : Type*}

def is_triangle (A B C : Type*) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (X Y Z : Type*) : ℝ := -- Dummy function to represent perimeter

theorem isosceles_triangle
  {A B C M N : Type*}
  (hABC : is_triangle A B C)
  (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (hM_on_AB : M ∈ [A, B])
  (hN_on_BC : N ∈ [B, C])
  (h_perim_AMC_CNA : perimeter A M C = perimeter C N A)
  (h_perim_ANB_CMB : perimeter A N B = perimeter C M B) :
  (A = B) ∨ (B = C) ∨ (C = A) :=
by sorry

end isosceles_triangle_l296_296601


namespace rectangle_area_percentage_contains_circle_l296_296001

-- Definitions and conditions from step a)
def diameter (d : ℝ) : Prop :=
  d > 0

def width_equals_diameter (w d : ℝ) : Prop :=
  w = d

def length_three_times_width (l w : ℝ) : Prop :=
  l = 3 * w

-- The percentage calculation to be verified
def percentage_area_of_circle_inside_rectangle (π : ℝ) : ℝ :=
  (π / 12) * 100

-- The main statement to be proved
theorem rectangle_area_percentage_contains_circle (d w l : ℝ) (π : ℝ) (hd : diameter d) 
  (hwd : width_equals_diameter w d) (hlw : length_three_times_width l w) :
  percentage_area_of_circle_inside_rectangle π = 26.18 :=
by
  sorry

end rectangle_area_percentage_contains_circle_l296_296001


namespace exponent_multiplication_correct_l296_296711

theorem exponent_multiplication_correct (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end exponent_multiplication_correct_l296_296711


namespace integer_values_of_b_l296_296861

theorem integer_values_of_b (b : ℤ) :
  (∃ x : ℤ, x^3 + 2*x^2 + b*x + 18 = 0) ↔ 
  b = -21 ∨ b = 19 ∨ b = -17 ∨ b = -4 ∨ b = 3 :=
by
  sorry

end integer_values_of_b_l296_296861


namespace m_divisible_by_5_m_divisible_by_7_m_base_10_representation_m_at_least_one_of_each_m_is_smallest_last_four_digits_of_m_l296_296211

noncomputable def m : ℕ :=
  2772

-- Properties of m
theorem m_divisible_by_5 : m % 5 = 0 := by
  sorry

theorem m_divisible_by_7 : m % 7 = 0 := by
  sorry

theorem m_base_10_representation : (10.digitSet.contains 2) ∧ (10.digitSet.contains 7) := by
  sorry

theorem m_at_least_one_of_each : (List.count (λ d : ℕ, d = 2) (m.digits 10) > 0) ∧ 
                                 (List.count (λ d : ℕ, d = 7) (m.digits 10) > 0) := by
  sorry

theorem m_is_smallest : ∀ n, (n % 5 = 0 ∧ n % 7 = 0 ∧
                             (10.digitSet.contains 2) ∧ 
                             (10.digitSet.contains 7) ∧ 
                             (List.count (λ d : ℕ, d = 2) (n.digits 10) > 0) ∧ 
                             (List.count (λ d : ℕ, d = 7) (n.digits 10) > 0)) → n ≥ m := by
  sorry

-- Final proof
theorem last_four_digits_of_m : (m % 10000) = 2772 :=
begin
  rw m,
  exact rfl
end

end m_divisible_by_5_m_divisible_by_7_m_base_10_representation_m_at_least_one_of_each_m_is_smallest_last_four_digits_of_m_l296_296211


namespace find_b_l296_296972

theorem find_b (a b : ℤ) (h : ∀ x : ℤ, (x^2 - 2*x - 1) * (ax^3 + x^2 + b * x + 2)) : b = -6 :=
by
  sorry

end find_b_l296_296972


namespace base_triangle_side_lengths_l296_296262

theorem base_triangle_side_lengths (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_pos : c > 0) :
  ∃(x : ℝ), x = sqrt((a^2 + b^2 + c^2) / 3 - (sqrt 2 / 3) * sqrt( (a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2 )) :=
by
  sorry

end base_triangle_side_lengths_l296_296262


namespace inverse_composition_l296_296634

theorem inverse_composition :
  let g (x : ℝ) := 3 * x^2 + 3 * x + 1 in
  g (g (37)) = -3 + sqrt (129) / 6 := sorry

end inverse_composition_l296_296634


namespace percentage_of_invalid_papers_l296_296051

theorem percentage_of_invalid_papers (total_papers : ℕ) (valid_papers : ℕ) (invalid_papers : ℕ) (percentage_invalid : ℚ) 
  (h1 : total_papers = 400) 
  (h2 : valid_papers = 240) 
  (h3 : invalid_papers = total_papers - valid_ppapers)
  (h4 : percentage_invalid = (invalid_papers : ℚ) / total_papers * 100) : 
  percentage_invalid = 40 :=
by
  sorry

end percentage_of_invalid_papers_l296_296051


namespace sum_divisible_by_power_of_two_l296_296624

theorem sum_divisible_by_power_of_two (n : ℕ) (h : 0 < n) :
  (∑ k in Finset.range (n // 2 + 1), Nat.choose n (2 * k + 1) * 2005^k) % 2^(n-1) = 0 :=
by
  sorry

end sum_divisible_by_power_of_two_l296_296624


namespace square_lattice_black_points_l296_296058

theorem square_lattice_black_points (n : ℕ) : 
    (2 / 7) * (n - 1)^2 ≤ M n ∧ M n ≤ (2 / 7) * n^2 :=
sorry

end square_lattice_black_points_l296_296058


namespace find_division_point_l296_296288

section
variable (A : ℝ × ℝ) (B : ℝ × ℝ) (λ : ℝ) (C : ℝ × ℝ)

-- The condition: points A at (-3, 2) and B at (4, -5) with λ = -3
def conditions : Prop :=
  A = (-3, 2) ∧ B = (4, -5) ∧ λ = -3

-- The theorem: The coordinates of the division point
theorem find_division_point (h : conditions A B λ) : C = (7.5, -8.5) :=
sorry
end

end find_division_point_l296_296288


namespace who_is_next_to_boris_l296_296807

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l296_296807


namespace sector_angle_l296_296484

theorem sector_angle (r : ℝ) (S : ℝ) (α : ℝ) (h₁ : r = 10) (h₂ : S = 50 * π / 3) (h₃ : S = 1 / 2 * r^2 * α) : 
  α = π / 3 :=
by
  sorry

end sector_angle_l296_296484


namespace solution_l296_296628

def question (x : ℝ) : Prop := (x - 5) / ((x - 3) ^ 2) < 0

theorem solution :
  {x : ℝ | question x} = {x : ℝ | x < 3} ∪ {x : ℝ | 3 < x ∧ x < 5} :=
by {
  sorry
}

end solution_l296_296628


namespace incentive_given_to_john_l296_296329

-- Conditions (definitions)
def commission_held : ℕ := 25000
def advance_fees : ℕ := 8280
def amount_given_to_john : ℕ := 18500

-- Problem statement
theorem incentive_given_to_john : (amount_given_to_john - (commission_held - advance_fees)) = 1780 := 
by
  sorry

end incentive_given_to_john_l296_296329


namespace distance_center_to_plane_l296_296237

noncomputable theory
open_locale real_inner_product_space

/-- The distance from the center of the unit sphere with radius 1 to the plane passing through points 
    A, B, and C on the sphere, given the spherical distances d(A, B) = π/2, d(A, C) = π/2, and 
    d(B, C) = π/3, is √21/7. -/
theorem distance_center_to_plane (A B C : EuclideanSpace ℝ (fin 3))
  (h1 : dist A B = π / 2) (h2 : dist A C = π / 2) (h3 : dist B C = π / 3) :
  dist (0 : EuclideanSpace ℝ (fin 3)) (affineSpan ℝ {A, B, C}) = √21 / 7 :=
sorry

end distance_center_to_plane_l296_296237


namespace oreo_solution_l296_296556

noncomputable def oreo_problem : Prop :=
∃ (m : ℤ), (11 + m * 11 + 3 = 36) → m = 2

theorem oreo_solution : oreo_problem :=
sorry

end oreo_solution_l296_296556


namespace sequence_elements_l296_296647

theorem sequence_elements (x : ℕ → ℝ) (h_pos : ∀ n, x n > 0)
    (h_eq : ∀ n, 2 * (∑ i in finset.range n, x i)^4 = (∑ i in finset.range n, (x i)^5) + (∑ i in finset.range n, (x i)^7)) :
    ∀ n, x n = n :=
by
  sorry

end sequence_elements_l296_296647


namespace addition_problem_l296_296999

theorem addition_problem (F I V N E : ℕ) (h1: F = 8) (h2: I % 2 = 0) 
  (h3: 1 ≤ F ∧ F ≤ 9) (h4: 1 ≤ I ∧ I ≤ 9) (h5: 1 ≤ V ∧ V ≤ 9) 
  (h6: 1 ≤ N ∧ N ≤ 9) (h7: 1 ≤ E ∧ E ≤ 9) 
  (h8: F ≠ I ∧ F ≠ V ∧ F ≠ N ∧ F ≠ E) 
  (h9: I ≠ V ∧ I ≠ N ∧ I ≠ E ∧ V ≠ N ∧ V ≠ E ∧ N ≠ E)
  (h10: 2 * F + 2 * I + 2 * V = 1000 * N + 100 * I + 10 * N + E):
  V = 5 :=
sorry

end addition_problem_l296_296999


namespace area_of_trapezoid_DBCE_l296_296984

variable (A B C D E : Type) [real_field A] [real_field B] [real_field C] [real_field D] [real_field E]

variable (area : A → B → C → D)
variable (triangle : Type)
variable (isosceles_triangle : triangle → Prop)
variable (similar : triangle → triangle → Prop)
variable (area_of_triangle_ABC : ℝ := 50)
variable (area_of_smallest_triangle : ℝ := 1)
variable (num_smallest_triangles_ADE : ℝ := 5)
variable (num_smallest_triangles_total : ℝ := 8)

axiom isosceles_and_similar (t : triangle) : isosceles_triangle t → similar t triangle_ABC

theorem area_of_trapezoid_DBCE :
  ∀ (triangle_ABC triangle_ADE : triangle),
  isosceles_triangle triangle_ABC →
  similar triangle_ABC triangle_ADE →
  area triangle_ABC = 50 →
  area triangle_ADE = 5 →
  area_of_smallest_triangles * (num_smallest_triangles_total - num_smallest_triangles_ADE) = 45 :=
by
  sorry

end area_of_trapezoid_DBCE_l296_296984


namespace mary_thought_animals_l296_296586

-- Definitions based on conditions
def double_counted_sheep : ℕ := 7
def forgotten_pigs : ℕ := 3
def actual_animals : ℕ := 56

-- Statement to be proven
theorem mary_thought_animals (double_counted_sheep forgotten_pigs actual_animals : ℕ) :
  (actual_animals + double_counted_sheep - forgotten_pigs) = 60 := 
by 
  -- Proof goes here
  sorry

end mary_thought_animals_l296_296586


namespace max_tanC_is_2root5over5_l296_296978

def triangle_max_tan (A B C : ℝ × ℝ) (AB : ℝ) (diff_AC_BC_squared : ℝ) : ℝ :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xC, yC) := C
  if AB = 2 ∧ diff_AC_BC_squared = 6 then 
    let x = xC
    let y = yC
    if y > 0 then 
      let kAC := (2 / 5) * y
      let kBC := 2 * y
      let tanC := (8 * y) / (5 / y + 4 * y)
      max_tanC := 2 * real.sqrt 5 / 5
      max_tanC
    else 0
  else 0

theorem max_tanC_is_2root5over5 : ∀ A B C : ℝ × ℝ,
  A = (-1, 0) →
  B = (1, 0) →
  (AB : ℝ) →
  (diff_AC_BC_squared : ℝ) →
  AB = 2 →
  diff_AC_BC_squared = 6 →
  triangle_max_tan A B C AB diff_AC_BC_squared = 2 * real.sqrt 5 / 5 :=
  by
    intros A B C hA hB AB diff_AC_BC_squared hAB hdiff 
    sorry

end max_tanC_is_2root5over5_l296_296978


namespace area_of_circle_is_correct_l296_296354

-- Define the side lengths of the rectangle
def length_rect := 26
def width_rect := 18

-- Define the common perimeter both the rectangle and circle have
def perimeter := 2 * (length_rect + width_rect)

-- Define the radius of the circle given the perimeter (circumference)
def radius_circle := perimeter / (2 * Real.pi)

-- Define the area of the circle
def area_circle := Real.pi * (radius_circle ^ 2)

theorem area_of_circle_is_correct :
  (approx area_circle).to_near_eq 616.217 :=
by
  sorry

end area_of_circle_is_correct_l296_296354


namespace simplify_and_substitute_l296_296250

theorem simplify_and_substitute (x : ℝ) (h1 : x ≠ 1) (h3 : x ≠ 3) : 
  ((1 - (2 / (x - 1))) * ((x^2 - x) / (x^2 - 6*x + 9))) = (x / (x - 3)) ∧ 
  (2 / (2 - 3)) = -2 := by
  sorry

end simplify_and_substitute_l296_296250


namespace f_diff_l296_296511

def f (n : ℕ) : ℚ := (1 / 3 : ℚ) * n * (n + 1) * (n + 2)

theorem f_diff (r : ℕ) : f r - f (r - 1) = r * (r + 1) := 
by {
  -- proof goes here
  sorry
}

end f_diff_l296_296511


namespace line_KL_divides_perpendicular_l296_296841

theorem line_KL_divides_perpendicular (A B C D K L M : Point) (h_parallelogram : Parallelogram A B C D)
  (h_touches : touches_circle_on_parallelogram_borders A B C D K L M) (h_perpendicular : Perpendicular C AB) :
  divides_into_equal_parts (line_through_points K L) (perpendicular_from_vertex C AB) :=
sorry

end line_KL_divides_perpendicular_l296_296841


namespace base_of_isosceles_triangle_l296_296266

theorem base_of_isosceles_triangle (b : ℝ) (h1 : 7 + 7 + b = 22) : b = 8 :=
by {
  sorry
}

end base_of_isosceles_triangle_l296_296266


namespace sum_of_decimal_numbers_l296_296358

def recurrence_sequence (a1 : ℕ) (k : ℕ) : List ℕ :=
  let rec aux (n : ℕ) (prev : ℕ) : List ℕ := 
    if n > k then [] else (k * prev % 10) :: aux (n + 1) (k * prev % 10)
  a1 :: aux 2 a1

-- Define A(a1, k) as a rational number
def A (a1 k : ℕ) : ℚ := 
  (recurrence_sequence a1 k).foldr (λ d (acc : ℚ) => d + acc / 10) 0

theorem sum_of_decimal_numbers : 
  (∑ k in Finset.range 9 \set [5], ∑ a1 in Finset.range 9, if a1 + 1 = 5 then A 5 k else 0) +
  (∑ a1 in Finset.range 9, A a1 5) +
  (∑ k in Finset.range 9 \set [5], ∑ a1 in Finset.range 9, if a1 + 1 ≠ 5 then A a1 k else 0) =
  44 + 5/9 :=
by
  sorry

end sum_of_decimal_numbers_l296_296358


namespace volume_inside_sphere_outside_cylinder_l296_296019

noncomputable def volumeDifference (r_cylinder base_radius_sphere : ℝ) :=
  let height := 4 * Real.sqrt 5
  let V_sphere := (4/3) * Real.pi * base_radius_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * height
  V_sphere - V_cylinder

theorem volume_inside_sphere_outside_cylinder
  (base_radius_sphere r_cylinder : ℝ) (h_base_radius_sphere : base_radius_sphere = 6) (h_r_cylinder : r_cylinder = 4) :
  volumeDifference r_cylinder base_radius_sphere = (288 - 64 * Real.sqrt 5) * Real.pi := by
  sorry

end volume_inside_sphere_outside_cylinder_l296_296019


namespace polynomial_roots_expression_l296_296205

theorem polynomial_roots_expression 
  (a b α β γ δ : ℝ)
  (h1 : α^2 - a*α - 1 = 0)
  (h2 : β^2 - a*β - 1 = 0)
  (h3 : γ^2 - b*γ - 1 = 0)
  (h4 : δ^2 - b*δ - 1 = 0) :
  ((α - γ)^2 * (β - γ)^2 * (α + δ)^2 * (β + δ)^2) = (b^2 - a^2)^2 :=
sorry

end polynomial_roots_expression_l296_296205


namespace john_running_distance_l296_296559

theorem john_running_distance:
  ∀ (total_distance days : ℕ), total_distance = 10200 ∧ days = 6 →
  (total_distance / days) = 1700 :=
by
  intros total_distance days h
  cases h with h1 h2
  rw [h1, h2]
  norm_num
  sorry

end john_running_distance_l296_296559


namespace person_next_to_Boris_arkady_galya_l296_296803

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l296_296803


namespace height_of_smaller_cuboid_l296_296751

theorem height_of_smaller_cuboid 
  (L_cuboid : ℝ) (W_cuboid : ℝ) (H_cuboid : ℝ)
  (L_small : ℝ) (W_small : ℝ)
  (n_small : ℝ)
  (volume_relation : L_cuboid * W_cuboid * H_cuboid = n_small * (L_small * W_small * H_small)) :
  H_small = 2 :=
by
  have volume_large : L_cuboid * W_cuboid * H_cuboid = 12 * 14 * 10 := by sorry
  have volume_small : L_small * W_small = 5 * 3 := by sorry
  calc
    H_small = ... := by sorry {
      sorry
    }
  done

end height_of_smaller_cuboid_l296_296751


namespace ranked_choice_voting_total_votes_l296_296168

theorem ranked_choice_voting_total_votes :
  ∃ V : ℝ, let VA := 0.27 * V,
               VB := 0.24 * V,
               VC := 0.20 * V,
               VD := 0.18 * V,
               VE := 0.11 * V,
               VA' := 0.30 * V,
               VB' := 0.27 * V,
               VC' := 0.22 * V,
               VD' := 0.21 * V,
               A_final := VA' + 0.05 * V,
               B_final := VB' + 0.26 * V  -- considering VD' + additional 0.05 from C and D supporters
           in 0.53 * V - 0.35 * V = 1350 ∧ V = 7500 :=
sorry

end ranked_choice_voting_total_votes_l296_296168


namespace original_triangle_area_proof_l296_296774

noncomputable def original_triangle_area (original_triangle : Type) [is_triangle original_triangle] (law_of_sines_equilateral_triangle : triangle) 
(h1 : law_of_sines_equilateral_triangle.is_equilateral) 
(h2 : law_of_sines_equilateral_triangle.side_length = 2) : ℝ :=
  match some_area_calculation_based_on_law_of_sines original_triangle law_of_sines_equilateral_triangle h1 h2 with
  | some calculated_area => calculated_area
  | none => 0 -- placeholder, in actual code this would raise an error or provide more context

theorem original_triangle_area_proof  (original_triangle : Type) [is_triangle original_triangle] 
(law_of_sines_equilateral_triangle : triangle) 
(h1 : law_of_sines_equilateral_triangle.is_equilateral) 
(h2 : law_of_sines_equilateral_triangle.side_length = 2): 
  original_triangle_area original_triangle law_of_sines_equilateral_triangle h1 h2 = real.sqrt 3 :=
  sorry

end original_triangle_area_proof_l296_296774


namespace smallest_number_exists_l296_296680

theorem smallest_number_exists : ∃ b : ℕ, (b % 3 = 2) ∧ (b % 5 = 2) ∧ (b % 7 = 3) ∧ (∀ n : ℕ, (n % 3 = 2) ∧ (n % 5 = 2) ∧ (n % 7 = 3) → n ≥ b) :=
by
  use 17
  repeat {split, trivial}
  sorry

end smallest_number_exists_l296_296680


namespace intersection_when_m_eq_2_range_of_m_l296_296223

open Set

variables (m x : ℝ)

def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}
def intersection (m : ℝ) : Set ℝ := A m ∩ B

-- First proof: When m = 2, the intersection of A and B is [1,2].
theorem intersection_when_m_eq_2 : intersection 2 = {x | 1 ≤ x ∧ x ≤ 2} :=
sorry

-- Second proof: The range of m such that A ⊆ A ∩ B
theorem range_of_m : {m | A m ⊆ B} = {m | -2 ≤ m ∧ m ≤ 1 / 2} :=
sorry

end intersection_when_m_eq_2_range_of_m_l296_296223


namespace complex_conjugate_in_third_quadrant_l296_296991

-- Define the complex numbers and relevant operations
def given_complex_number : ℂ :=
  (Complex.exp (Complex.I * 0)) / (1 + Complex.I) + (1 + 2 * Complex.I) ^ 2

def complex_conjugate (z : ℂ) : ℂ :=
z.re - z.im * Complex.I

def is_in_third_quadrant (z : ℂ) : Prop :=
z.re < 0 ∧ z.im < 0

-- The proof goal: to show that the complex conjugate of the given complex number is in the third quadrant
theorem complex_conjugate_in_third_quadrant :
  is_in_third_quadrant (complex_conjugate given_complex_number) :=
sorry

end complex_conjugate_in_third_quadrant_l296_296991


namespace solve_BSNK_l296_296567

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem solve_BSNK (B S N K : ℝ) (hB : 0 < B) (hS : 0 < S) (hN : 0 < N) (hK : 0 < K)
    (h1 : log_base_10 (B * K) + log_base_10 (B * N) = 3)
    (h2 : log_base_10 (N * K) + log_base_10 (N * S) = 4)
    (h3 : log_base_10 (S * B) + log_base_10 (S * K) = 5) : B * S * N * K = 10000 :=
begin
    sorry
end

end solve_BSNK_l296_296567


namespace point_on_parabola_distance_eq_vertex_l296_296974

theorem point_on_parabola_distance_eq_vertex 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (Hp : P = (a^2, a))
  (Hparabola : ∃ (a : ℝ), P = (a^2, a))
  (Hdist : ∀ P : ℝ × ℝ, (P = (a^2, a)) → 
          (a^2 + 1/4 = real.sqrt (a^4 + a^2))) : 
    P = (1/8, sqrt 2 / 4) ∨ P = (1/8, - sqrt 2 / 4) := 
by
  sorry

end point_on_parabola_distance_eq_vertex_l296_296974


namespace possible_values_of_a2_l296_296020

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n + 2) = abs (a (n + 1) - a n)

theorem possible_values_of_a2 :
  ∃ a2s : Finset ℕ,
    (∀ a2 ∈ a2s, a2 < 1001 ∧ Nat.gcd 1001 a2 = 1 ∧ (a2 % 2 = 0)) ∧
    a2s.card = 360 :=
by
  sorry

end possible_values_of_a2_l296_296020


namespace derivative_at_0_does_not_exist_l296_296389

noncomputable def f : ℝ → ℝ :=
λ x, if x ≠ 0 then sin x * cos (5 / x) else 0

theorem derivative_at_0_does_not_exist : ¬(∃ l, (deriv f 0 = l)) :=
sorry

end derivative_at_0_does_not_exist_l296_296389


namespace needed_blue_cans_l296_296838

-- Definitions derived from the problem conditions
def blue_to_yellow_ratio := 4 / 3
def total_cans := 42

-- The problem's main assertion based on conditions provided
theorem needed_blue_cans 
  (h_ratio : blue_to_yellow_ratio = 4 / 3) 
  (h_total_cans : total_cans = 42) : 
  let blue_fraction := 4 / (4 + 3)
      blue_cans := blue_fraction * total_cans
  in blue_cans = 24 := by
sorry

end needed_blue_cans_l296_296838


namespace average_price_equation_l296_296540

-- Define the variables and conditions
variables (x : ℝ) (hx : x > 8)

def avg_price_popular_science : ℝ := x + 8

-- Number of books purchased equations
def num_books_popular_science : ℝ := 15000 / avg_price_popular_science x
def num_books_literature : ℝ := 12000 / x

-- Theorem statement
theorem average_price_equation (x_pos : 0 < x) (xlit : x > 8) :
    15000 / (x + 8) = 12000 / x := 
sorry

end average_price_equation_l296_296540


namespace count_multiples_5_or_11_not_both_l296_296966

theorem count_multiples_5_or_11_not_both 
    (h₁ : ∀ n, Multiples.of 5 n = n / 5)
    (h₂ : ∀ n, Multiples.of 11 n = n / 11)
    (h₃ : ∀ n, Multiples.of 55 n = n / 55)
    : let count_5_only := Multiples.of 5 200 - Multiples.of 55 200,
          count_11_only := Multiples.of 11 200 - Multiples.of 55 200,
          result := count_5_only + count_11_only
      in result = 52 := 
by
  sorry

end count_multiples_5_or_11_not_both_l296_296966


namespace part_a_part_b_part_c_part_d_l296_296217

variable (k x y : ℝ) (h : 0 < k)

def f (x : ℝ) : ℝ := Real.exp (k * x)

theorem part_a : f k 0 = 1 := by
  sorry

theorem part_b (x : ℝ) : f k (-x) = 1 / f k x := by
  sorry

theorem part_c (x : ℝ) : f k x = Real.root 4 (f k (4 * x)) := by
  sorry

theorem part_d (x y : ℝ) (h1 : y > x) : f k y > f k x := by
  sorry

end part_a_part_b_part_c_part_d_l296_296217


namespace range_of_function_l296_296285

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem range_of_function :
  set.range (λ x, f x) ∩ set.Icc (-2 : ℝ) (1 : ℝ) = set.Icc (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_function_l296_296285


namespace remaining_milk_and_coffee_l296_296356

/-- 
Given:
1. A cup initially contains 1 glass of coffee.
2. A quarter glass of milk is added to the cup.
3. The mixture is thoroughly stirred.
4. One glass of the mixture is poured back.

Prove:
The remaining content in the cup is 1/5 glass of milk and 4/5 glass of coffee. 
--/
theorem remaining_milk_and_coffee :
  let coffee_initial := 1  -- initial volume of coffee
  let milk_added := 1 / 4  -- volume of milk added
  let total_volume := coffee_initial + milk_added  -- total volume after mixing = 5/4 glasses
  let milk_fraction := milk_added / total_volume  -- fraction of milk in the mixture = 1/5
  let coffee_fraction := coffee_initial / total_volume  -- fraction of coffee in the mixture = 4/5
  let volume_poured := 1 / 4  -- volume of mixture poured out
  let milk_poured := (milk_fraction * volume_poured : ℝ)  -- volume of milk poured out = 1/20 glass
  let coffee_poured := (coffee_fraction * volume_poured : ℝ)  -- volume of coffee poured out = 1/5 glass
  let remaining_milk := milk_added - milk_poured  -- remaining volume of milk = 1/5 glass
  let remaining_coffee := coffee_initial - coffee_poured  -- remaining volume of coffee = 4/5 glass
  remaining_milk = 1 / 5 ∧ remaining_coffee = 4 / 5 :=
by
  sorry

end remaining_milk_and_coffee_l296_296356


namespace trapezoid_area_l296_296665

def upper_side : ℝ := 10
def lower_side : ℝ := 2 * upper_side
def height : ℝ := 5

theorem trapezoid_area : 1 / 2 * (upper_side + lower_side) * height = 75 := by
  sorry

end trapezoid_area_l296_296665


namespace period_and_axis_of_symmetry_range_of_AB_plus_AC_l296_296572

-- Definitions for the conditions
def f (x : Real) : Real := sin x * sin (π / 2 + x) + sqrt 3 * cos x ^ 2 - sqrt 3 / 2

-- Theorem for part (1)
theorem period_and_axis_of_symmetry :
  (∀ x : Real, f (x + π) = f x) ∧
  (∃ k : Int, ∀ x : Real, 2 * x + π / 3 = k * π + π / 2 → x = k * π / 2 + π / 12) :=
sorry

-- Theorem for part (2)
theorem range_of_AB_plus_AC (A B C : Real) (R : Real) (area : Real) :
  0 < A ∧ A < π / 2 ∧ f A = 0 ∧ area = π ∧ (∀ x y z : Real, (angle x y z < π / 2) → 
  (∃ s r : Real, (s*r^2 = area) ∧ r > 0)) → 
  (3 < A + B ∧ A + B ≤ 2 * sqrt 3) :=
sorry

end period_and_axis_of_symmetry_range_of_AB_plus_AC_l296_296572


namespace doctor_lindsay_l296_296328

def cost_of_adult_visit (A : ℝ) : ℝ := A

theorem doctor_lindsay (A : ℝ)
  (h1 : ∀ (hours : ℕ), total_adults : ℕ := 4 * hours)
  (h2 : ∀ (hours : ℕ), total_children : ℕ := 3 * hours)
  (h3 : total_adults = 32)
  (h4 : total_children = 24)
  (h5 : total_income := total_adults * A + total_children * 25)
  (h6 : total_income = 2200) :
  cost_of_adult_visit A = 50 :=
by
  sorry

end doctor_lindsay_l296_296328


namespace Yankees_to_Mets_ratio_l296_296981

theorem Yankees_to_Mets_ratio : 
  ∀ (Y M R : ℕ), M = 88 → (M + R + Y = 330) → (4 * R = 5 * M) → (Y : ℚ) / M = 3 / 2 :=
by
  intros Y M R hm htotal hratio
  sorry

end Yankees_to_Mets_ratio_l296_296981


namespace three_angles_right_l296_296161

-- Define the type for an angle in a quadrilateral
structure Quadrilateral :=
(angle1 : ℝ)
(angle2 : ℝ)
(angle3 : ℝ)
(angle4 : ℝ)

-- Define a helper function to check if an angle is a right angle
def is_right_angle (angle : ℝ) : Prop :=
  angle = 90

-- Define a function that checks if a quadrilateral is a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  is_right_angle q.angle1 ∧ is_right_angle q.angle2 ∧ 
  is_right_angle q.angle3 ∧ is_right_angle q.angle4

-- State the main theorem
theorem three_angles_right (q : Quadrilateral) :
  is_right_angle q.angle1 ∧ is_right_angle q.angle2 ∧ 
  is_right_angle q.angle3 → is_rectangle q :=
begin
  sorry -- Proof is omitted as per instructions
end

end three_angles_right_l296_296161


namespace probability_bijection_l296_296501

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2, 3, 4, 5}

theorem probability_bijection : 
  let total_mappings := 5^4
  let bijections := 5 * 4 * 3 * 2
  let probability := bijections / total_mappings
  probability = 24 / 125 := 
by
  sorry

end probability_bijection_l296_296501


namespace annies_classmates_count_l296_296039

theorem annies_classmates_count (spent : ℝ) (cost_per_candy : ℝ) (candies_left : ℕ) (candies_per_classmate : ℕ) (expected_classmates : ℕ):
  spent = 8 ∧ cost_per_candy = 0.1 ∧ candies_left = 12 ∧ candies_per_classmate = 2 ∧ expected_classmates = 34 →
  (spent / cost_per_candy) - candies_left = (expected_classmates * candies_per_classmate) := 
by
  intros h
  sorry

end annies_classmates_count_l296_296039


namespace projection_onto_plane_l296_296869

theorem projection_onto_plane :
  let v := ⟨2, -1, 4⟩ : ℝ × ℝ × ℝ,
  let n := ⟨2, 3, -1⟩ : ℝ × ℝ × ℝ,
  let dot (a b : ℝ × ℝ × ℝ) := a.1 * b.1 + a.2 * b.2 + a.3 * b.3,
  let scale (c : ℝ) (a : ℝ × ℝ × ℝ) := ⟨c * a.1, c * a.2, c * a.3⟩,
  let sub (a b : ℝ × ℝ × ℝ) := ⟨a.1 - b.1, a.2 - b.2, a.3 - b.3⟩,
  let p := sub v (scale (dot v n / dot n n) n)
  in p = ⟨17/7, -5/14, 53/14⟩ := sorry

end projection_onto_plane_l296_296869


namespace sequence_general_formula_and_sum_l296_296466

/-- Given a sequence {a_n} with the sum of its first n terms denoted as S_n,
and S_(n+1) = a_(n+1) + n^2, the formula for the sequence a_n is 2n-1.
Additionally, let b_n = 1 / (a_n * a_(n+1)), the sum of the first n terms
of the sequence {b_n}, denoted as T_n, is n / (2n+1). -/
theorem sequence_general_formula_and_sum (S : ℕ → ℕ) (a : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (∀ n, S n + n^2 = S (n+1) + a (n+1)) →
  a n = 2 * n - 1 →
  b n = 1 / (a n * a (n+1)) →
  T n = ∑ i in finset.range n, b i →
  T n = n / (2 * n + 1) :=
by
  intros
  sorry

end sequence_general_formula_and_sum_l296_296466


namespace six_points_in_circle_distance_l296_296398

theorem six_points_in_circle_distance {r : ℝ} (h_r : r = 1) :
  ∃ b : ℝ, (∀ (points : Fin 6 → (ℝ × ℝ)),
    ∃ (i j : Fin 6), i ≠ j ∧ dist points[i] points[j] ≤ b) ∧ b = (2 * Real.pi) / 5 :=
by
  let circle := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2}
  have hr_eq_1 : r = 1 := h_r
  sorry

end six_points_in_circle_distance_l296_296398


namespace like_terms_exp_l296_296118

theorem like_terms_exp (a b : ℝ) (m n x : ℝ)
  (h₁ : 2 * a ^ x * b ^ (n + 1) = -3 * a * b ^ (2 * m))
  (h₂ : x = 1) (h₃ : n + 1 = 2 * m) : 
  (2 * m - n) ^ x = 1 := 
by
  sorry

end like_terms_exp_l296_296118


namespace karson_needs_to_buy_books_l296_296641

theorem karson_needs_to_buy_books (capacity : ℕ) (current_books : ℕ) (target_percentage : ℝ) : 
  capacity = 400 → current_books = 120 → target_percentage = 0.9 → 
  ∃ books_to_buy : ℕ, books_to_buy = (target_percentage * capacity).toNat - current_books ∧ books_to_buy = 240 :=
by
  intros h_capacity h_current_books h_target_percentage
  use ((target_percentage * capacity).toNat - current_books)
  split
  · refl
  · simp [h_capacity, h_current_books, h_target_percentage]
    norm_num
    sorry

end karson_needs_to_buy_books_l296_296641


namespace x_condition_l296_296454

theorem x_condition (x : ℝ) : x^2 - 2*x ≠ 3 → (x ≠ 3 ∧ x ≠ -1) := 
begin
  intro h,
  -- proof steps would go here
  sorry
end

end x_condition_l296_296454


namespace modulus_of_complex_expr_l296_296867

-- Definitions based on conditions
def complex_expr : ℂ := (2 : ℂ) / (1 + complex.i)^2
def simplified_expr : ℂ := complex.i

-- Statement of the theorem
theorem modulus_of_complex_expr :
  complex_expr = simplified_expr →
  complex.abs complex_expr = 1 :=
by
  intros h
  rw [h]
  smul sorry

end modulus_of_complex_expr_l296_296867


namespace units_digit_2143_pow_752_l296_296967

theorem units_digit_2143_pow_752 : 
  let base := 2143 
  let exp := 752 
  (base % 10 = 3) →
  (exp % 4 = 0) →
  (3^exp % 10 = 1) →
  (base ^ exp % 10 = 1) :=
by
  intros base exp base_units_digit exp_mod_4 power_units_digit
  have units_digit_exp := Nat.pow_mod 3 exp 10
  rw [units_digit_exp, power_units_digit]
  exact pow_mod (pow_mod base exp 10) power_units_digit

end units_digit_2143_pow_752_l296_296967


namespace persons_next_to_Boris_l296_296827

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l296_296827


namespace train_length_l296_296772

/--
A train of a certain length traveling at 60 kmph takes 17.998560115190784 seconds to cross a bridge of 190 m in length.
Prove that the length of the train is approximately 109.976001923 meters.
-/
theorem train_length
    (speed_kmph : ℝ)
    (speed_mps : ℝ)
    (time_s : ℝ)
    (bridge_length_m : ℝ)
    (total_distance_m : ℝ)
    (train_length_m : ℝ)
    (h1 : speed_kmph = 60)
    (h2 : speed_mps = speed_kmph * 1000 / 3600)
    (h3 : time_s = 17.998560115190784)
    (h4 : bridge_length_m = 190)
    (h5 : total_distance_m = speed_mps * time_s)
    (h6 : train_length_m = total_distance_m - bridge_length_m) :
    train_length_m ≈ 109.976001923 :=
by
    sorry

end train_length_l296_296772


namespace find_values_of_p_l296_296863

def geometric_progression (p : ℝ) : Prop :=
  (2 * p)^2 = (4 * p + 5) * |p - 3|

theorem find_values_of_p :
  {p : ℝ | geometric_progression p} = {-1, 15 / 8} :=
by
  sorry

end find_values_of_p_l296_296863


namespace equal_area_division_by_k_l296_296028

noncomputable def triangle := 
  {A : ℝ × ℝ, B : ℝ × ℝ, C : ℝ × ℝ} 

def triangle_area (t : triangle) : ℝ :=
  let A := t.A
  let B := t.B
  let C := t.C
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem equal_area_division_by_k : 
  ∀ (A B C : ℝ × ℝ) (k : ℝ),
  A = (0, 2) → B = (0, 0) → C = (10, 0) →
  let t := {A := A, B := B, C := C} in
  let total_area := triangle_area t in
  total_area = 10 →
  let y_k_triangle_area := (5 * (2 - k)) in
  y_k_triangle_area = (total_area / 2) →
  k = 1 :=
by 
  intros A B C k hA hB hC t total_area total_area_eq y_k_triangle_area y_k_triangle_area_eq
  sorry

end equal_area_division_by_k_l296_296028


namespace max_non_constant_factors_l296_296860

-- Definition of the polynomial x^10 - 1
def poly : ℝ[X] := X^10 - 1

-- Definitions for the polynomials that are factors
def q1 : ℝ[X] := X - 1
def q2 : ℝ[X] := X + 1
def q3 : ℝ[X] := X^4 + X^3 + X^2 + X + 1
def q4 : ℝ[X] := X^4 - X^3 + X^2 - X + 1

-- The maximum number of non-constant factors with real coefficients
theorem max_non_constant_factors : 
  (poly = q1 * q2 * q3 * q4) → 4 = 4 :=
sorry

end max_non_constant_factors_l296_296860


namespace geometric_sequence_general_term_l296_296495

noncomputable def sequence (n : ℕ) : ℕ
| 0     := 2
| (n+1) := 2 * sequence n

theorem geometric_sequence_general_term (n : ℕ) : sequence n = 2^n :=
sorry

end geometric_sequence_general_term_l296_296495


namespace largest_is_5b_l296_296884

def largestInSet (b : ℤ) : ℤ :=
  let set_vals := {-2 * b, 5 * b, 30 / b, b^2 + 1, 2}
  set_vals.max'

theorem largest_is_5b (b : ℤ) (hb : b = 3) : 
  largestInSet b = 5 * b :=
by
  sorry

end largest_is_5b_l296_296884


namespace Ponchik_week_day_l296_296589

theorem Ponchik_week_day (n s : ℕ) (h1 : s = 20) (h2 : s * (4 * n + 1) = 1360) : n = 4 :=
by
  sorry

end Ponchik_week_day_l296_296589


namespace domain_of_k_l296_296679

noncomputable def k (x : ℝ) : ℝ := (1 / (x + 6)) + (1 / (x^2 + 2*x + 9)) + (1 / (x^3 - 27))

theorem domain_of_k : {x : ℝ | k x ≠ 0} = {x : ℝ | x ≠ -6 ∧ x ≠ 3} :=
by
  sorry

end domain_of_k_l296_296679


namespace part1_part2_part3_l296_296209

section part1

variables {a b : ℝ} 

def f (x : ℝ) := a * x^2 + (b - 1) * x + 3

-- (1)
theorem part1 (h : ∀ x, x > -1 ∧ x < 3 → f x > 0) (h₁ : a ≠ 0) : 2 * a + b = -3 := 
sorry

end part1

section part2

variables {a b : ℝ}

def f2 (x : ℝ) := a * x^2 + (b - 1) * x + 3

-- (2)
theorem part2 (h₁ : f2 1 = 5) (h₂ : b > -1) : 
  ∃ x, (∀ a, abs(a) > 0 → (x = (1 / abs(a)) + (4 * abs(a)) / (b + 1)) ) ∧ (∀ y, y >= x) ∧ (x = 2) := 
sorry

end part2

section part3

variables {a : ℝ}

def f3 (x : ℝ) := a * x^2 + (-a - 3 - 1) * x + 3
def g (x : ℝ) := -2 * x + 1

-- (3)
theorem part3 (h : ∀ x, f3 x < g x) 
  : if a < 0 then { x | x < 2 / a ∨ x > 1 } 
    else if 0 < a ∧ a < 2 then { x | 1 < x ∧ x < 2 / a }
    else if a = 2 then ∅
    else { x | 2 / a < x ∧ x < 1 } := 
sorry

end part3

end part1_part2_part3_l296_296209


namespace arcsin_sqrt_three_over_two_l296_296390

theorem arcsin_sqrt_three_over_two : 
  ∃ θ, θ = Real.arcsin (Real.sqrt 3 / 2) ∧ θ = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_three_over_two_l296_296390


namespace initial_girls_is_11_l296_296368

-- Definitions of initial parameters and transformations
def initially_girls_percent : ℝ := 0.35
def final_girls_percent : ℝ := 0.25
def three : ℝ := 3

-- 35% of the initial total is girls
def initially_girls (p : ℝ) : ℝ := initially_girls_percent * p
-- After three girls leave and three boys join, the count of girls
def final_girls (p : ℝ) : ℝ := initially_girls p - three

-- Using the condition that after the change, 25% are girls
def proof_problem : Prop := ∀ (p : ℝ), 
  (final_girls p) / p = final_girls_percent →
  (0.1 * p) = 3 → 
  initially_girls p = 11

-- The statement of the theorem to be proved in Lean 4
theorem initial_girls_is_11 : proof_problem := sorry

end initial_girls_is_11_l296_296368


namespace range_of_x_l296_296903

noncomputable def f : ℝ → ℝ := sorry  -- f is an even function and decreasing on [0, +∞)

theorem range_of_x (x : ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≥ f y) 
  (h_condition : f (Real.log x) > f 1) : 
  1 / 10 < x ∧ x < 10 := 
sorry

end range_of_x_l296_296903


namespace triangle_is_isosceles_l296_296615

variables {A B C M N : Type} [EuclideanGeometry A B C M N]

theorem triangle_is_isosceles 
  (hABC : triangle A B C) 
  (hM : OnSide M A B) 
  (hN : OnSide N B C) 
  (h1 : Perimeter (triangle A M C) = Perimeter (triangle C N A))
  (h2 : Perimeter (triangle A N B) = Perimeter (triangle C M B)) :
  IsIsosceles (triangle A B C) := 
sorry

end triangle_is_isosceles_l296_296615


namespace number_of_sets_in_set_game_l296_296382

theorem number_of_sets_in_set_game :
  (∑ k in finset.range 82, if k = 3 then nat.choose 81 k else 0) = 85320 :=
by
  -- total number of ways to choose 3 cards out of 81
  have h1: nat.choose 81 3 = 85320 := 
    sorry,
  exact h1

end number_of_sets_in_set_game_l296_296382


namespace shiny_pennies_probability_l296_296736

theorem shiny_pennies_probability :
  let n := 5 in
  let m := 6 in
  let total_ways := Nat.choose (n + m) n in
  let prob_more_than_8_draws := (70 * 3 + 56 * 3 + 28) / total_ways in
  let reduced_prob := Rat.num_denom prob_more_than_8_draws in
  let a := reduced_prob.1 in
  let b := reduced_prob.2 in
  a + b = 56 :=
by
  sorry

end shiny_pennies_probability_l296_296736


namespace A_days_to_complete_work_l296_296738

noncomputable def work (W : ℝ) (A_work_per_day B_work_per_day : ℝ) (days_A days_B days_B_alone : ℝ) : ℝ :=
  A_work_per_day * days_A + B_work_per_day * days_B

theorem A_days_to_complete_work 
  (W : ℝ)
  (A_work_per_day B_work_per_day : ℝ)
  (days_A days_B days_B_alone : ℝ)
  (h1 : days_A = 5)
  (h2 : days_B = 12)
  (h3 : days_B_alone = 18)
  (h4 : B_work_per_day = W / days_B_alone)
  (h5 : work W A_work_per_day B_work_per_day days_A days_B days_B_alone = W) :
  W / A_work_per_day = 15 := 
sorry

end A_days_to_complete_work_l296_296738


namespace ice_forms_inner_surface_in_winter_l296_296302

-- Definitions based on conditions
variable (humid_air_inside : Prop) 
variable (heat_transfer_inner_surface : Prop) 
variable (heat_transfer_outer_surface : Prop) 
variable (temp_inner_surface_below_freezing : Prop) 
variable (condensation_inner_surface_below_freezing : Prop)
variable (ice_formation_inner_surface : Prop)
variable (cold_dry_air_outside : Prop)
variable (no_significant_condensation_outside : Prop)

-- Proof of the theorem
theorem ice_forms_inner_surface_in_winter :
  humid_air_inside ∧
  heat_transfer_inner_surface ∧
  heat_transfer_outer_surface ∧
  (¬sufficient_heating → temp_inner_surface_below_freezing) ∧
  (condensation_inner_surface_below_freezing ↔ (temp_inner_surface_below_freezing ∧ humid_air_inside)) ∧
  (ice_formation_inner_surface ↔ (condensation_inner_surface_below_freezing ∧ temp_inner_surface_below_freezing)) ∧
  (cold_dry_air_outside → ¬ice_formation_outer_surface)
  → ice_formation_inner_surface :=
sorry

end ice_forms_inner_surface_in_winter_l296_296302


namespace find_x_for_set_6_l296_296143

theorem find_x_for_set_6 (x : ℝ) (h : 6 ∈ ({2, 4, x^2 - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end find_x_for_set_6_l296_296143


namespace average_age_of_group_l296_296261

theorem average_age_of_group :
  let n_graders := 40
  let n_parents := 50
  let n_teachers := 10
  let avg_age_graders := 12
  let avg_age_parents := 35
  let avg_age_teachers := 45
  let total_individuals := n_graders + n_parents + n_teachers
  let total_age := n_graders * avg_age_graders + n_parents * avg_age_parents + n_teachers * avg_age_teachers
  (total_age : ℚ) / total_individuals = 26.8 :=
by
  sorry

end average_age_of_group_l296_296261


namespace sqrt_of_sum_of_powers_l296_296691

theorem sqrt_of_sum_of_powers : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_of_sum_of_powers_l296_296691


namespace cos_minus_sin_eq_l296_296896

noncomputable def problem (θ α : ℝ) : ℝ :=
  if hθ : α < θ ∧ θ < 5 * Real.pi / 4 then
    let k := Real.sqrt 5 in
    if (2 * tan θ)^2 - 2 * k * 2 * tan θ = 3 - k^2 ∧
       (2 / (2 * tan θ))^2 - 2 * k * 2 / (2 * tan θ) = 3 - k^2 then
      cos θ - sin θ
    else
      0
  else
    0

theorem cos_minus_sin_eq :
  ∀ θ α, (α < θ ∧ θ < 5 * Real.pi / 4) →
  tan θ * (1 / tan θ) = 1 →
  ∃ k, k = Real.sqrt 5 ∧
  (2 * tan θ)^2 - 2 * k * 2 * tan θ = 3 - k^2 ∧
  (2 / (2 * tan θ))^2 - 2 * k * 2 / (2 * tan θ) = 3 - k^2 →
  cos θ - sin θ = -Real.sqrt ((5 - 2 * Real.sqrt 5) / 5) :=
by
  intros
  sorry

end cos_minus_sin_eq_l296_296896


namespace effect_on_revenue_l296_296235

variables (P Q : ℝ)

def original_revenue : ℝ := P * Q
def new_price : ℝ := 1.60 * P
def new_quantity : ℝ := 0.80 * Q
def new_revenue : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue (h1 : new_price P = 1.60 * P) (h2 : new_quantity Q = 0.80 * Q) :
  new_revenue P Q - original_revenue P Q = 0.28 * original_revenue P Q :=
by
  sorry

end effect_on_revenue_l296_296235


namespace range_of_m_l296_296486

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + x else logb (1/3) x

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, f x ≤ (5/4) * m - m^2) : 1/4 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l296_296486


namespace integer_count_in_range_l296_296955

theorem integer_count_in_range (x : Int) : 
  (Set.count (Set.range (λ x, ( -6 ≤ 3*x + 2 ∧ 3*x + 2 ≤ 9))) 5) := 
by 
  sorry

end integer_count_in_range_l296_296955


namespace solve_for_x_l296_296706

theorem solve_for_x :
  ∃ x : ℚ, x + 2/5 = 7/10 + 1/2 ∧ x = 4/5 :=
by {
  use (4/5),
  split,
  {
    linarith,
  },
  {
    refl,
  }
}

end solve_for_x_l296_296706


namespace sqrt_sum_of_cubes_l296_296687

theorem sqrt_sum_of_cubes :
  √(4^3 + 4^3 + 4^3 + 4^3) = 16 :=
by
  sorry

end sqrt_sum_of_cubes_l296_296687


namespace minimum_RS_length_l296_296199

variables {A B C D M R S : Type} [E : EuclideanGeometry]

noncomputable def minimum_distance_RS (ABCD : E.Rhombus A B C D) (AC BD : ℝ) (M : E.PointOnSegment A B) 
  (AM : ℝ) (perpendicular_R : E.PerpendicularFrom M (E.Segment AC)) 
  (perpendicular_S : E.PerpendicularFrom M (E.Segment BD)) : Prop :=
AC = 24 → BD = 40 → AM = 6 → ∃ (RS : ℝ), RS = 3.01

theorem minimum_RS_length (ABCD : E.Rhombus A B C D) (AC BD : ℝ) (M : E.PointOnSegment A B) 
  (AM : ℝ) (R : E.PerpendicularFoot M (E.Segment AC)) 
  (S : E.PerpendicularFoot M (E.Segment BD)) : minimum_distance_RS ABCD AC BD M AM R S :=
by
  intros h1 h2 h3
  existsi 3.01
  sorry

end minimum_RS_length_l296_296199


namespace max_alpha_minus_beta_l296_296517

theorem max_alpha_minus_beta (α β : ℝ) (h1 : 0 ≤ β ∧ β ≤ α) (h2 : α < π / 2) (h3 : α = atan (3 * tan β)) :
  α - β ≤ π / 6 :=
sorry

end max_alpha_minus_beta_l296_296517


namespace most_efficient_packing_l296_296238

theorem most_efficient_packing :
  ∃ box_size, 
  (box_size = 3 ∨ box_size = 6 ∨ box_size = 9) ∧ 
  (∀ q ∈ [21, 18, 15, 12, 9], q % box_size = 0) ∧
  box_size = 3 :=
by
  sorry

end most_efficient_packing_l296_296238


namespace derivative_at_zero_dne_l296_296387

noncomputable def f : ℝ → ℝ
| x := if x ≠ 0 then (Real.sin x * Real.cos (5 / x)) else 0

theorem derivative_at_zero_dne : ¬(DifferentiableAt ℝ f 0) :=
sorry

end derivative_at_zero_dne_l296_296387


namespace will_ferris_wheel_rides_l296_296332

theorem will_ferris_wheel_rides (day_rides night_rides : ℕ) (h_day : day_rides = 7) (h_night : night_rides = 6) :
  day_rides + night_rides = 13 :=
by
  rw [h_day, h_night]
  exact Nat.add_comm 7 6

end will_ferris_wheel_rides_l296_296332


namespace coeff_of_x10_in_expansion_l296_296086

def binomial_coeff (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

theorem coeff_of_x10_in_expansion :
  (finset.sum (finset.range 11) (λ r, binomial_coeff 10 r * 2^(10 - r) * if r = 10 then 1 else if r = 9 then 20 else 0)) = 19 :=
sorry

end coeff_of_x10_in_expansion_l296_296086


namespace trash_cans_veterans_park_l296_296053

theorem trash_cans_veterans_park : 
  ∀ (t_veteran : ℕ) (t_central_half : ℕ) (t_central : ℕ) (t_shift : ℕ), 
    t_veteran = 24 -> 
    t_central_half = t_veteran / 2 -> 
    t_central = t_central_half + 8 -> 
    t_shift = t_central / 2 -> 
    t_veteran + t_shift = 34 := 
by
  intros t_veteran t_central_half t_central t_shift 
  assume h1 : t_veteran = 24
  assume h2 : t_central_half = t_veteran / 2
  assume h3 : t_central = t_central_half + 8
  assume h4 : t_shift = t_central / 2
  sorry

end trash_cans_veterans_park_l296_296053


namespace vertical_asymptotes_count_l296_296072

noncomputable def number_of_vertical_asymptotes (f : ℝ → ℝ) : ℕ :=
  sorry

def function_f (x : ℝ) : ℝ := (x + 3) / (x * x - 4 * x + 3)

theorem vertical_asymptotes_count : 
  number_of_vertical_asymptotes function_f = 2 := 
  sorry

end vertical_asymptotes_count_l296_296072


namespace average_percentage_decrease_is_one_sixth_selling_price_is_125_l296_296005

-- Definitions and conditions for Part 1
def initial_price := 144
def final_price := 100
def average_percentage_decrease (x : ℝ) : Prop := (1 - x) * (1 - x) = real.sqrt (100 / 144)

-- Theorem for Part 1
theorem average_percentage_decrease_is_one_sixth : ∃ x : ℝ, average_percentage_decrease x ∧ x = 1 / 6 :=
sorry

-- Definitions and conditions for Part 2
def initial_selling_price := 140
def initial_sales := 20
def cost_price := 100
def desired_profit := 1250
def sales_quantity (y : ℝ) : ℝ := 20 + 2 * (140 - y)
def profit (y : ℝ) : ℝ := (y - cost_price) * (sales_quantity y)
def valid_selling_price (y : ℝ) : Prop := profit y = desired_profit

-- Theorem for Part 2
theorem selling_price_is_125 : ∃ y : ℝ, valid_selling_price y ∧ y = 125 :=
sorry

end average_percentage_decrease_is_one_sixth_selling_price_is_125_l296_296005


namespace f_odd_function_no_parallel_lines_l296_296916

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - (1 / a^x))

theorem f_odd_function {a : ℝ} (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∀ x : ℝ, f a (-x) = -f a x := 
by
  sorry

theorem no_parallel_lines {a : ℝ} (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∀ x1 x2 : ℝ, x1 ≠ x2 → f a x1 ≠ f a x2 :=
by
  sorry

end f_odd_function_no_parallel_lines_l296_296916


namespace max_revenue_day_l296_296742

noncomputable def p (t : ℕ) : ℕ :=
if 0 < t ∧ t < 25 then t + 20 else
if 25 ≤ t ∧ t ≤ 30 then -t + 100 else 0

noncomputable def Q (t : ℕ) : ℕ :=
if 0 < t ∧ t ≤ 30 then -t + 40 else 0

noncomputable def R (t : ℕ) : ℕ := p t * Q t

theorem max_revenue_day :
  ∃ t, 0 < t ∧ t ≤ 30 ∧ (∀ t', 0 < t' ∧ t' <= 30 → R t ≤ R t') ∧ R t = 1125 :=
sorry

end max_revenue_day_l296_296742


namespace who_next_to_boris_l296_296795

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l296_296795


namespace evaluate_expression_l296_296417

theorem evaluate_expression : (y = 3) → (y^1 * y^2 * y^3 * ... * y^18) / (y^3 * y^6 * y^9 * ... * y^27) = 3^36 :=
by
  sorry

end evaluate_expression_l296_296417


namespace volume_of_sphere_l296_296242

-- Define the problem conditions
variables (A B C O : Point) -- Points A, B, C on the sphere, and the center of the sphere
constant angle_BAC : ℝ := 135 -- Angle BAC is 135 degrees
constant BC : ℝ := 2        -- Distance BC is 2
constant dist_O_to_plane_ABC : ℝ := 1 -- Distance from center O to plane ABC is 1

-- Define the spherical surface condition
axiom on_same_spherical_surface : on_same_sphere A B C O

-- Lean theorem statement for the volume of the sphere calculation
theorem volume_of_sphere : volume_of_sphere O = 4 * sqrt 3 * π :=
sorry

end volume_of_sphere_l296_296242


namespace find_c_l296_296286

noncomputable def parabola_equation (a b c y : ℝ) : ℝ :=
  a * y^2 + b * y + c

theorem find_c (a b c : ℝ) (h_vertex : (-4, 2) = (-4, 2)) (h_point : (-2, 4) = (-2, 4)) :
  ∃ c : ℝ, parabola_equation a b c 0 = -2 :=
  by {
    use -2,
    sorry
  }

end find_c_l296_296286


namespace total_numbers_l296_296298

theorem total_numbers (N : ℕ) (h1 : (∑ i in range N, a i) / N = 9.9)
  (h2 : (∑ i in finset.range 6, a i) / 6 = 10.5)
  (h3 : (∑ i in finset.range 6, a (N - 1 - i)) / 6 = 11.4)
  (h4 : a ((N - 1) / 2) = 22.5) :
  N = 11 := 
sorry

end total_numbers_l296_296298


namespace ratio_spent_to_total_l296_296657

-- David's hourly rate for mowing lawns
def rate_per_hour : ℕ := 14

-- Total hours mowed per day
def hours_per_day : ℕ := 2

-- Total days in a week
def days_per_week : ℕ := 7

-- Total money left after giving half to his mom
def money_left : ℕ := 49

-- The ratio of money spent on shoes to total money made is 1:2
theorem ratio_spent_to_total : 
  let total_hours := hours_per_day * days_per_week
      total_money := total_hours * rate_per_hour
      money_before_giving_half := money_left * 2
      money_spent_on_shoes := total_money - money_before_giving_half in
    money_spent_on_shoes / total_money = 1 / 2 :=
by
  sorry

end ratio_spent_to_total_l296_296657


namespace reduced_number_by_13_l296_296346

-- Definitions directly from the conditions
def original_number (m a n k : ℕ) := m + 10^k * a + 10^(k+1) * n
def replaced_number (m n k : ℕ) := m + 10^(k+1) * n

theorem reduced_number_by_13 (N m a n k : ℕ)
  (h1 : N = original_number m a n k)
  (h2 : N = 13 * replaced_number m n k)
  : ∃ (b : ℕ), b ∈ {1, 2, 3} ∧ N = 325 * b * 10^97 :=
sorry

end reduced_number_by_13_l296_296346


namespace sin_BAD_over_sin_CAD_l296_296187

open Real

variables {A B C D : Type} -- Assume points are some type
variables [BD : HasRatio (⟨B, D⟩ : Point) (2 / 3)] -- B to D divides the whole BC as 2:1 therefore BD = 2/3 BC
variables [CD : HasRatio (⟨C, D⟩ : Point) (1 / 3)] -- C to D divides the whole BC as 2:1 therefore CD = 1/3 BC

-- Triangle ABC conditions
variable hABC : Triangle A B C 
variable hAngB : angle B = 45 -- in degrees
variable hAngC : angle C = 30 -- in degrees

-- Lean can validate angles sum in a triangle, so we put the final statement as theorem
theorem sin_BAD_over_sin_CAD : 
    (2 / Real.sqrt 2) = 1 :=
    sorry

end sin_BAD_over_sin_CAD_l296_296187


namespace maximize_S_l296_296240

-- Define the pentagon and the vertices as complex numbers on the unit circle
def pentagon_vertices : list ℂ := 
  [1, complex.exp (2 * real.pi * complex.I / 5), complex.exp (4 * real.pi * complex.I / 5), 
   complex.exp (6 * real.pi * complex.I / 5), complex.exp (8 * real.pi * complex.I / 5)]

-- Define the point placement function which maximizes S
def point_placement : list ℂ :=
  pentagon_vertices.concat [pentagon_vertices.head, pentagon_vertices.head, pentagon_vertices.head] -- 3 remaining points added to the first vertex

-- Define the function S
def S (points : list ℂ) : ℝ :=
  ∑ i in finset.range points.length.to_nat, 
    ∑ j in finset.Icc (i + 1) (points.length.to_nat), complex.abs (points[i] - points[j])^2

theorem maximize_S : 
  ∀ points, (∀ p, p ∈ points → p ∈ pentagon_vertices) → points.length = 2018 → 
  S points ≤ S point_placement :=
begin
  sorry -- Proof omitted
end

end maximize_S_l296_296240


namespace num_integers_satisfy_inequality_l296_296946

theorem num_integers_satisfy_inequality : 
  ∃ n : ℕ, n = 5 ∧ {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.finite.card = n :=
sorry

end num_integers_satisfy_inequality_l296_296946


namespace sequence_term_l296_296183

theorem sequence_term (x : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, 2 / x n = 1 / x (n - 1) + 1 / x (n + 1))
  (h₁ : x 2 = 2 / 3)
  (h₂ : x 4 = 2 / 5) :
  x 10 = 2 / 11 := 
sorry

end sequence_term_l296_296183


namespace probability_of_non_negative_product_is_correct_l296_296309

noncomputable def probability_non_negative_product : ℝ := (17 : ℝ) / 32

theorem probability_of_non_negative_product_is_correct :
  ∀ (a b : ℝ), a ∈ Icc (-25) 15 → b ∈ Icc (-25) 15 →
  ∃ (p : ℝ), p = (17 / 32) :=
by
  intro a b
  intro ha hb
  use probability_non_negative_product
  sorry

end probability_of_non_negative_product_is_correct_l296_296309


namespace arithmetic_progression_even_terms_l296_296987

theorem arithmetic_progression_even_terms (a d n : ℕ) (h_even : n % 2 = 0)
  (h_last_first_diff : (n - 1) * d = 16)
  (h_sum_odd : n * (a + (n - 2) * d / 2) = 81)
  (h_sum_even : n * (a + d + (n - 2) * d / 2) = 75) :
  n = 8 :=
by sorry

end arithmetic_progression_even_terms_l296_296987


namespace larger_number_is_1617_l296_296269

-- Given conditions
variables (L S : ℤ)
axiom condition1 : L - S = 1515
axiom condition2 : L = 16 * S + 15

-- To prove
theorem larger_number_is_1617 : L = 1617 := by
  sorry

end larger_number_is_1617_l296_296269


namespace two_crows_problem_l296_296163

def Bird := { P | P = "parrot" ∨ P = "crow"} -- Define possible bird species.

-- Define birds and their statements
def Adam_statement (Adam Carl : Bird) : Prop := Carl = Adam
def Bob_statement (Adam : Bird) : Prop := Adam = "crow"
def Carl_statement (Dave : Bird) : Prop := Dave = "crow"
def Dave_statement (Adam Bob Carl Dave: Bird) : Prop := 
  (if Adam = "parrot" then 1 else 0) + 
  (if Bob = "parrot" then 1 else 0) + 
  (if Carl = "parrot" then 1 else 0) + 
  (if Dave = "parrot" then 1 else 0) ≥ 3

-- The main proposition to prove
def main_statement : Prop :=
  ∃ (Adam Bob Carl Dave : Bird), 
    (Adam_statement Adam Carl) ∧ 
    (Bob_statement Adam) ∧ 
    (Carl_statement Dave) ∧ 
    (Dave_statement Adam Bob Carl Dave) ∧ 
    (if Adam = "crow" then 1 else 0) + 
    (if Bob = "crow" then 1 else 0) + 
    (if Carl = "crow" then 1 else 0) + 
    (if Dave = "crow" then 1 else 0) = 2

-- Proof statement to be filled
theorem two_crows_problem : main_statement :=
by {
  sorry
}

end two_crows_problem_l296_296163


namespace rectangle_area_error_83_percent_l296_296169

theorem rectangle_area_error_83_percent (L W : ℝ) :
  let actual_area := L * W
  let measured_length := 1.14 * L
  let measured_width := 0.95 * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 8.3 := by
  sorry

end rectangle_area_error_83_percent_l296_296169


namespace boric_acid_molecular_weight_boric_acid_oxygen_composition_l296_296048

def molecular_weight (H B O : ℝ) : ℝ :=
  3 * H + 10.81 + 3 * O

def percent_composition (atomic_weight element_weight : ℝ) : ℝ :=
  (element_weight / atomic_weight) * 100

theorem boric_acid_molecular_weight :
  ∀ (H B O : ℝ), H = 1.008 → B = 10.81 → O = 16.00 → molecular_weight H B O = 61.834 :=
by
  intros H B O H_eq B_eq O_eq
  rw [H_eq, B_eq, O_eq]
  simp [molecular_weight]
  norm_num
  sorry

theorem boric_acid_oxygen_composition :
  ∀ (H B O : ℝ), H = 1.008 → B = 10.81 → O = 16.00 →
  percent_composition (molecular_weight H B O) (3 * O) ≈ 77.63 :=
by
  intros H B O H_eq B_eq O_eq
  rw [H_eq, B_eq, O_eq]
  simp [molecular_weight, percent_composition]
  norm_num
  sorry

end boric_acid_molecular_weight_boric_acid_oxygen_composition_l296_296048


namespace particular_solution_l296_296868

variables {y : ℝ → ℝ} {x : ℝ}
variable (y')

-- Define the differential equation and initial condition
def diff_eq := ∀ x, y' x * (sin x)^2 * (log (y x)) + y x = 0

def initial_condition := y (π / 4) = 1

-- The theorem we need to prove
theorem particular_solution (h1 : diff_eq y') (h2 : initial_condition) : 
  ∀ x, (log (y x))^2 / 2 = cot x - 1 :=
sorry

end particular_solution_l296_296868


namespace minimum_value_is_one_l296_296414

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  (1 / (3 * a + 2)) + (1 / (3 * b + 2)) + (1 / (3 * c + 2))

theorem minimum_value_is_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  minimum_value a b c = 1 := by
  sorry

end minimum_value_is_one_l296_296414


namespace smallest_denominator_fraction_interval_exists_l296_296865

def interval (a b c d : ℕ) : Prop :=
a = 14 ∧ b = 73 ∧ c = 5 ∧ d = 26

theorem smallest_denominator_fraction_interval_exists :
  ∃ (a b c d : ℕ), 
    a / b < 19 / 99 ∧ b < 99 ∧
    19 / 99 < c / d ∧ d < 99 ∧
    interval a b c d :=
by
  sorry

end smallest_denominator_fraction_interval_exists_l296_296865


namespace find_f_0_max_value_f_f_less_than_linear1_f_less_than_linear2_l296_296256

variable {α : Type*} [LinearOrderedField α] [DecidableEq α]

-- Definitions of the function and conditions
def f (x : α) : α := sorry

-- Condition I
axiom cond_I (x : α) (h : 0 ≤ x ∧ x ≤ 1) : f(x) ≥ 3

-- Condition II
axiom cond_II : f(1) = 4

-- Condition III
axiom cond_III (x1 x2 : α) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3: x1 + x2 ≤ 1) : f(x1 + x2) ≥ f(x1) + f(x2) - 3

-- Proof problems
-- 1. Find the value of f(0)
theorem find_f_0 : f(0) = 3 := by
  sorry

-- 2. Determine the maximum value of the function f(x)
theorem max_value_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x) ≤ 4 := by
  sorry

-- 3. Prove f(x) < 3x + 3 for specific intervals
theorem f_less_than_linear1 {x : α} (h : x ∈ set.Ioc (1/3:α) 1) : f(x) < 3 * x + 3 := by
  sorry

theorem f_less_than_linear2 {x : α} (n : ℕ) (h : x ∈ set.Ioc (1/(3^(n+1):α)) (1/(3^n:α))) : f(x) < 3 * x + 3 := by
  sorry

end find_f_0_max_value_f_f_less_than_linear1_f_less_than_linear2_l296_296256


namespace num_integers_satisfying_ineq_count_l296_296944

theorem num_integers_satisfying_ineq_count :
  {x : ℤ | -6 ≤ 3 * (x : ℤ) + 2 ∧ 3 * (x : ℤ) + 2 ≤ 9}.finite.to_finset.card = 5 :=
by
  sorry

end num_integers_satisfying_ineq_count_l296_296944


namespace sample_mean_correct_l296_296459

def sample_mean (m n p : ℕ) (x₁ x₂ x₃ : ℝ) : ℝ :=
  (m * x₁ + n * x₂ + p * x₃) / (m + n + p)

theorem sample_mean_correct (m n p : ℕ) (x₁ x₂ x₃ : ℝ) :
  sample_mean m n p x₁ x₂ x₃ = (m * x₁ + n * x₂ + p * x₃) / (m + n + p) :=
by simp [sample_mean]

end sample_mean_correct_l296_296459


namespace number_of_not_possible_d_l296_296282

noncomputable def triangle_circle_problem (d : ℕ) : Prop :=
  ∃ (t r : ℝ), 
    3 * t > 0 ∧        -- the triangle has a perimeter greater than 0
    3 * t - 2 * real.pi * r = 500 ∧ 
    t = 2 * r + (d : ℝ) 

theorem number_of_not_possible_d : ∃ (count : ℕ → ℕ), 
  ∀ d, d ≥ 167 → triangle_circle_problem d ↔ false :=
by
  sorry

end number_of_not_possible_d_l296_296282


namespace pet_store_has_70_birds_l296_296760

-- Define the given conditions
def num_cages : ℕ := 7
def parrots_per_cage : ℕ := 4
def parakeets_per_cage : ℕ := 3
def cockatiels_per_cage : ℕ := 2
def canaries_per_cage : ℕ := 1

-- Total number of birds in one cage
def birds_per_cage : ℕ := parrots_per_cage + parakeets_per_cage + cockatiels_per_cage + canaries_per_cage

-- Total number of birds in all cages
def total_birds := birds_per_cage * num_cages

-- Prove that the total number of birds is 70
theorem pet_store_has_70_birds : total_birds = 70 :=
sorry

end pet_store_has_70_birds_l296_296760


namespace sqrt_of_sum_of_powers_l296_296690

theorem sqrt_of_sum_of_powers : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_of_sum_of_powers_l296_296690


namespace determine_functions_l296_296405

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := λ x, 2 * x - f x

theorem determine_functions (x : ℝ) :
  (∀ x y, x < y → f x < f y) ∧
  (∀ x, f x + g x = 2 * x) ∧
  (∀ x, f (g x) = x ∧ g (f x) = x) →
  f x = x :=
begin
  intros h,
  sorry
end

end determine_functions_l296_296405


namespace cost_per_ball_3_balls_l296_296153

def total_cost (n : ℕ) (cost_per_ball : ℚ) : ℚ := n * cost_per_ball

theorem cost_per_ball_3_balls (cost_per_ball : ℚ) : 
  total_cost 3 cost_per_ball = 4.62 → cost_per_ball = 1.54 :=
begin
  intro h,
  calc 
    cost_per_ball = 4.62 / 3 : by rw [←h, mul_div_cancel_left 4.62 (by norm_num : (3 : ℚ) ≠ 0)]
               ... = 1.54 : by norm_num,
  assumption
end

end cost_per_ball_3_balls_l296_296153


namespace triangle_isosceles_l296_296608

theorem triangle_isosceles
  {A B C M N : Point}
  (h_M_on_AB : ∃ t ∈ Set.Icc (0 : ℝ) 1, M = t • A + (1 - t) • B)
  (h_N_on_BC : ∃ t ∈ Set.Icc (0 : ℝ) 1, N = t • B + (1 - t) • C)
  (h_perimeter_AMC_CNA : dist A M + dist M C + dist C A = dist C N + dist N A + dist A C)
  (h_perimeter_ANB_CMB : dist A N + dist N B + dist B A = dist C M + dist M B + dist B C)
  : isosceles_triangle A B C := 
sorry

end triangle_isosceles_l296_296608


namespace linda_total_spent_l296_296225

theorem linda_total_spent : 
  let notebooks := 3 * 1.20
  let pencils := 1.50
  let pens := 1.70
  notebooks + pencils + pens = 6.80 := 
by
  let notebooks := 3 * 1.20
  let pencils := 1.50
  let pens := 1.70
  have h_notebooks : notebooks = 3.60 := by norm_num
  have h_pencils : pencils = 1.50 := by rfl
  have h_pens : pens = 1.70 := by rfl
  calc
    notebooks + pencils + pens = 3.60 + 1.50 + 1.70 : by rw [h_notebooks, h_pencils, h_pens]
                      ... = 6.80 : by norm_num

end linda_total_spent_l296_296225


namespace find_ages_l296_296621

theorem find_ages (P F M : ℕ) 
  (h1 : F - P = 31)
  (h2 : (F + 8) + (P + 8) = 69)
  (h3 : F - M = 4)
  (h4 : (P + 5) + (M + 5) = 65) :
  P = 11 ∧ F = 42 ∧ M = 38 :=
by
  sorry

end find_ages_l296_296621


namespace fourth_root_12960000_eq_60_l296_296057

theorem fourth_root_12960000_eq_60 :
  (6^4 = 1296) →
  (10^4 = 10000) →
  (60^4 = 12960000) →
  (Real.sqrt (Real.sqrt 12960000) = 60) := 
by
  intros h1 h2 h3
  sorry

end fourth_root_12960000_eq_60_l296_296057


namespace range_of_a_l296_296481

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^3 - a * x^2 - 2 * a * x + a^2 - 1 = 0 → (x - 1) ∨ (x^2 + x + 1 < a)) : a < 3/4 :=
sorry

end range_of_a_l296_296481


namespace proof_statements_l296_296144

theorem proof_statements (m : ℝ) (x y : ℝ)
  (h1 : 2 * x + y = 4 - m)
  (h2 : x - 2 * y = 3 * m) :
  (m = 1 → (x = 9 / 5 ∧ y = -3 / 5)) ∧
  (3 * x - y = 4 + 2 * m) ∧
  ¬(∃ (m' : ℝ), (8 + m') / 5 < 0 ∧ (4 - 7 * m') / 5 < 0) :=
sorry

end proof_statements_l296_296144


namespace base_13_satisfies_equation_l296_296409

theorem base_13_satisfies_equation :
  ∃ (a : ℕ), a = 13 ∧ 
             (C_a = 12) →
             (375_a + 592_a = 9C7_a) :=
by
  sorry

end base_13_satisfies_equation_l296_296409


namespace num_integers_satisfying_inequality_l296_296951

theorem num_integers_satisfying_inequality : 
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 := 
by 
  sorry

end num_integers_satisfying_inequality_l296_296951


namespace tangent_line_at_zero_l296_296273

-- Define the function
def f (x : ℝ) : ℝ := Real.exp (2 * x)

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 2 * Real.exp (2 * x)

-- Define the point of tangency
def pointOfTangency : ℝ × ℝ := (0, f 0)

-- Define the tangent line equation
def tangentLine (x : ℝ): ℝ := 2 * x + 1

-- Proof statement to show the tangent line at x = 0 to f(x)
theorem tangent_line_at_zero : ∀ (x : ℝ), 
  tangentLine x = 2 * x + 1 := 
by
  -- Proof goes here
  sorry

end tangent_line_at_zero_l296_296273


namespace range_of_x_in_function_l296_296548

theorem range_of_x_in_function : ∀ (x : ℝ), (2 - x ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ -2) :=
by
  intro x
  sorry

end range_of_x_in_function_l296_296548


namespace angle_C_sin_A_l296_296156

variable {A B C a b c : Real}

-- Assuming given conditions as hypotheses
hypothesis h1 : b * Real.sin (2 * C) = c * Real.sin B
hypothesis h2 : Real.sin (B - Real.pi / 3) = 3 / 5

-- Problem 1: Prove that angle C = π / 3
theorem angle_C (h1 : b * Real.sin (2 * C) = c * Real.sin B) : C = Real.pi / 3 := sorry

-- Problem 2: Prove that sin A = 4√3 - 3 / 10
theorem sin_A (h1 : b * Real.sin (2 * C) = c * Real.sin B)
             (h2 : Real.sin (B - Real.pi / 3) = 3 / 5)
             (h3 : C = Real.pi / 3) : Real.sin A = (4 * Real.sqrt 3 - 3) / 10 := sorry

end angle_C_sin_A_l296_296156


namespace sqrt_of_sum_of_powers_l296_296695

theorem sqrt_of_sum_of_powers :
  sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 :=
sorry

end sqrt_of_sum_of_powers_l296_296695


namespace value_of_M_l296_296150

theorem value_of_M : ∃ M : ℝ, (5 + 6 + 7) / 3 = (1988 + 1989 + 1990) / M ∧ M = 994.5 :=
by
  let lhs := (5 + 6 + 7 : ℝ) / 3
  let rhs := (1988 + 1989 + 1990 : ℝ) / 994.5
  have h : lhs = rhs := sorry
  use 994.5
  split
  · exact h
  · rfl

end value_of_M_l296_296150


namespace sum_without_zeroes_l296_296184

theorem sum_without_zeroes :
  let sum_group := ∑ i in finset.range 503, (4 * i + 1) - (4 * i + 2) - (4 * i + 3) + (4 * i + 4)
  let remaining := 2013 - 2014 - 2015
  let S_pos := ∑ i in finset.range 100, 20 * (i + 1)
  let S_neg := ∑ i in finset.range 101, 10 * (2 * i + 1)
  let original_sum := remaining - 2016 in
  original_sum + S_pos - S_neg = -3026 :=
by
  sorry

end sum_without_zeroes_l296_296184


namespace problem_max_n_l296_296886

open Set

structure Point (α : Type*) := 
  (x : α)
  (y : α)

inductive Color
| red
| green
| yellow

structure SpecialPoint (α : Type*) := 
  (pt : Point α)
  (color : Color)

def inside_triangle {α} [linear_ordered_field α] 
  (tri : list (Point α)) (pt : Point α) : Prop :=
  sorry -- Assume the definition of whether a point is inside a triangle

def max_special_points (points : list (SpecialPoint ℝ)) : Prop :=
  ∀ (n a b c : ℕ) (reds greens yellows : list (SpecialPoint ℝ)),
  list.length points = n ∧ 
  list.length reds = a ∧ 
  list.length greens = b ∧ 
  list.length yellows = c ∧
  (∀ (p : SpecialPoint ℝ), p ∈ points → (p ∈ reds ∨ p ∈ greens ∨ p ∈ yellows)) ∧
  (∀ (tri : list (SpecialPoint ℝ)), uhvectriangle tri ∧ 
   ∃ (p : SpecialPoint ℝ), p ∈ greens ∧ inside_triangle tri.p.pt p.pt) ∧
  (∀ (tri : list (SpecialPoint ℝ)), uhvectriangle tri ∧ 
   ∃ (p : SpecialPoint ℝ), p ∈ yellows ∧ inside_triangle tri.p.pt p.pt) ∧
  (∀ (tri : list (SpecialPoint ℝ)), uhvectriangle tri ∧ 
   ∃ (p : SpecialPoint ℝ), p ∈ reds ∧ inside_triangle tri.p.pt p.pt) →
  n ≤ 18

theorem problem_max_n (points : list (SpecialPoint ℝ)) :
  max_special_points points →
  list.length points ≤ 18 :=
  sorry

end problem_max_n_l296_296886


namespace quadrilateral_not_necessarily_square_l296_296533

/-- In a quadrilateral \(ABCD\):
1. The diagonals \(AC\) and \(BD\) intersect perpendicularly.
2. A circle can be inscribed in the quadrilateral, meaning \(AB + CD = AD + BC\).
3. A circle can be circumscribed around the quadrilateral, meaning \(\angle A + \angle C = 180^\circ\) and \(\angle B + \angle D = 180^\circ\).
These conditions do not necessarily imply the quadrilateral is a square. -/
theorem quadrilateral_not_necessarily_square
  (ABCD : Type)
  [quadrilateral ABCD]
  (h1 : perpendicular_diagonals ABCD)
  (h2 : inscribed_circle ABCD)
  (h3 : circumscribed_circle ABCD) :
  ¬(is_square ABCD) :=
by
  sorry

end quadrilateral_not_necessarily_square_l296_296533


namespace shiela_prepared_paper_stars_l296_296626

theorem shiela_prepared_paper_stars (num_classmates num_stars_per_classmate : ℕ)
  (h1 : num_classmates = 9)
  (h2 : num_stars_per_classmate = 5) :
  num_classmates * num_stars_per_classmate = 45 :=
by
  rw [h1, h2]
  exact nat.mul_comm 9 5
  sorry

end shiela_prepared_paper_stars_l296_296626


namespace person_next_to_Boris_arkady_galya_l296_296802

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l296_296802


namespace math_or_sci_but_not_both_l296_296422

-- Definitions of the conditions
variable (students_math_and_sci : ℕ := 15)
variable (students_math : ℕ := 30)
variable (students_only_sci : ℕ := 18)

-- The theorem to prove
theorem math_or_sci_but_not_both :
  (students_math - students_math_and_sci) + students_only_sci = 33 := by
  -- Proof is omitted.
  sorry

end math_or_sci_but_not_both_l296_296422


namespace find_dihedral_angle_cosine_l296_296312

noncomputable def r (R : ℝ) : ℝ := R / 2
noncomputable def distance_between_centers (R : ℝ) : ℝ := (3 * R) / 2
noncomputable def cos_angle_between_centers_and_edge : ℝ := real.cos (real.pi / 6) -- 30 degrees in radians
noncomputable def dihedral_angle_cos : ℝ := 1 / 9

noncomputable def dihedral_angle_cosine := sorry

theorem find_dihedral_angle_cosine (R : ℝ) (r = r R) (d = distance_between_centers R)
    (a = cos_angle_between_centers_and_edge) : dihedral_angle_cosine = dihedral_angle_cos :=
    sorry

end find_dihedral_angle_cosine_l296_296312


namespace orthocenter_locus_l296_296197

-- Define the circle k with center O, and a fixed point A outside circle k.
variable (O : Point) (R : ℝ) -- R is the radius of the circle
variable (A : Point) -- A is a fixed point outside the circle

-- Define the varying diameter BC
variable (B C : Point)
hypothesis circle_condition : dist O B = R ∧ dist O C = R ∧ ∠ B O C = π

-- Define the orthocenter H of triangle ABC
variable (H : Point)
hypothesis orthocenter_condition : is_orthocenter H A B C

-- State the target proposition: The locus of H is the line GF
theorem orthocenter_locus (G F : Point) (tangents_condition : tangents_from A G F) :
  ∃ GF, locus_of_orthocenter H BC = line GF :=
sorry

end orthocenter_locus_l296_296197


namespace cyclic_quadrilateral_l296_296236

theorem cyclic_quadrilateral (ABCD K L M N : Type) [geometry ABCD K L M N]
    (AK : K.between A L)
    (KN : N.between K L)
    (DN : N.between K D)
    (BL : L.between B C)
    (BC : C.between B L)
    (CM : M.between C N)
    (AK_eq_KN : length A K = length K N)
    (KN_eq_DN : length K N = length D N)
    (BL_eq_BC : length B L = length B C)
    (BC_eq_CM : length B C = length C M)
    (BCNK_cyclic : is_cyclic_quadrilateral B C N K) :
  is_cyclic_quadrilateral A D M L :=
sorry

end cyclic_quadrilateral_l296_296236


namespace unique_positive_integer_k_for_rational_solutions_l296_296074

theorem unique_positive_integer_k_for_rational_solutions :
  ∃ (k : ℕ), (k > 0) ∧ (∀ (x : ℤ), x * x = 256 - 4 * k * k → x = 8) ∧ (k = 7) :=
by
  sorry

end unique_positive_integer_k_for_rational_solutions_l296_296074


namespace inverse_proportion_function_l296_296277

theorem inverse_proportion_function (f : ℝ → ℝ) (h : ∀ x, f x = 1/x) : f 1 = 1 := 
by
  sorry

end inverse_proportion_function_l296_296277


namespace who_is_next_to_Boris_l296_296819

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l296_296819


namespace least_subtraction_divisible_l296_296704

def least_subtrahend (n m : ℕ) : ℕ :=
n % m

theorem least_subtraction_divisible (n : ℕ) (m : ℕ) (sub : ℕ) :
  n = 13604 → m = 87 → sub = least_subtrahend n m → (n - sub) % m = 0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end least_subtraction_divisible_l296_296704


namespace min_n_for_constant_term_l296_296520

/-- If the expansion of (sqrt(x) + (3 / cbrt(x)))^n contains a constant term,
then the minimum value of n is 5. -/
theorem min_n_for_constant_term (n : ℕ) (x : ℝ) :
  (∃ r : ℕ, r ≤ n ∧ 3^r * (Nat.choose n r) * x^((1/2 : ℚ) * n - (5/6 : ℚ) * r) = 1) →
  n = 5 := by
  sorry

end min_n_for_constant_term_l296_296520


namespace percentage_difference_l296_296721

theorem percentage_difference (x y : ℝ) (h : x = 3 * y) : ((x - y) / x) * 100 = 66.67 :=
by
  sorry

end percentage_difference_l296_296721


namespace problem_l296_296516

theorem problem (A : ℕ) (B : ℕ) (hA : A = 2011 ^ 2011) (hB : B = (nat.factorial 2011) ^ 2) : A < B :=
by
  rw [hA, hB]
  sorry

end problem_l296_296516


namespace derivative_at_0_does_not_exist_l296_296388

noncomputable def f : ℝ → ℝ :=
λ x, if x ≠ 0 then sin x * cos (5 / x) else 0

theorem derivative_at_0_does_not_exist : ¬(∃ l, (deriv f 0 = l)) :=
sorry

end derivative_at_0_does_not_exist_l296_296388


namespace prob_at_least_7_is_1_over_9_l296_296518

open BigOperators -- Open namespace for big operators

-- Given definitions
def certain_people: ℕ := 4
def uncertain_people: ℕ := 4
def probability_stay: ℚ := 1 / 3

-- Probability of exactly k out of n people with certain probability staying
def binomial_prob (n k: ℕ) (p: ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1-p)^(n-k))

-- Probability that at least 7 people stayed
def prob_at_least_7 := 
  binomial_prob 4 3 probability_stay + binomial_prob 4 4 probability_stay

-- Theorem: The probability that at least 7 people stayed the entire time is 1/9
theorem prob_at_least_7_is_1_over_9 : 
  prob_at_least_7 = 1 / 9 := by sorry

end prob_at_least_7_is_1_over_9_l296_296518


namespace boris_neighbors_l296_296790

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l296_296790


namespace projection_vector_l296_296206

theorem projection_vector (e a : ℝ^3) (h_e_unit : ∥e∥ = 1) (h_a_mag : ∥a∥ = 2)
  (h_angle : real.angle e a = real.pi / 3) : 
  (a • e / ∥e∥^2) • e = e :=
by sorry

end projection_vector_l296_296206


namespace boris_neighbors_l296_296788

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l296_296788


namespace zeros_of_f_l296_296918

noncomputable def f (x a : ℝ) : ℝ := exp (x - a) + exp (a - x) + (1/2) * x^2 - a^2 * log x - 2

theorem zeros_of_f (a : ℝ) (h : a > 0) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ a > real.sqrt (real.exp 1) :=
sorry

end zeros_of_f_l296_296918


namespace minimum_sum_n_val_l296_296469

variable {α : Type*} [OrderedRing α]
variable (a : ℕ → α) -- Arithmetic sequence
variable (S : ℕ → α) -- Sum of first n terms

def problem_conditions (d : α) : Prop :=
  a 1 = -11 ∧ (a 5 + a 6 = -4) ∧ a = λ n, a 1 + (n - 1) * d ∧ S = λ n, (n / 2) * (2 * a 1 + (n - 1) * d)

theorem minimum_sum_n_val (d : α) (h : problem_conditions a S d) : 
  ∃ n : ℕ, (n = 6) ∧ S n = (n / 2) * (2 * (-11) + (n - 1) * d) := 
by
  sorry

end minimum_sum_n_val_l296_296469


namespace range_a_l296_296924

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (x - 2))

def domain_A : Set ℝ := { x | x < -1 ∨ x > 2 }

def solution_set_B (a : ℝ) : Set ℝ := { x | x < a ∨ x > a + 1 }

theorem range_a (a : ℝ)
  (h : (domain_A ∪ solution_set_B a) = solution_set_B a) :
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l296_296924


namespace two_digit_numbers_l296_296507

theorem two_digit_numbers :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n / 10) > (n % 10) ∧ (n / 10) + (n % 10) > 12}.finite.toFinset.card = 9 :=
by
  sorry

end two_digit_numbers_l296_296507


namespace max_distinct_values_l296_296928

noncomputable def f : Fin 2018 → ℕ := sorry

theorem max_distinct_values :
  (∀ x : ℕ, f ((4 * x + 2) % 2018) = f ((4 * x + 1) % 2018)) →
  (∀ x : ℕ, f ((5 * x + 3) % 2018) = f ((5 * x + 2) % 2018)) →
  (∀ x : ℕ, f ((7 * x + 5) % 2018) = f ((7 * x + 4) % 2018)) →
  ∃ k : ℕ, k = 1138 ∧ (Set.toFinset (Set.range f)).card ≤ k :=
begin
  sorry
end

end max_distinct_values_l296_296928


namespace distance_of_symmetric_points_on_parabola_l296_296480

-- Define what it means for points to be on the given parabola
def onParabola (A B : ℝ × ℝ) : Prop :=
  (∀ x y : ℝ, A = (x, y) → y = 3 - x^2) ∧ 
  (∀ x y : ℝ, B = (x, y) → y = 3 - x^2)

-- Define what it means for points to be symmetric with respect to the line x + y = 0
def symmetric (A B : ℝ × ℝ) : Prop :=
  ∀ xA yA xB yB : ℝ, A = (xA, yA) ∧ B = (xB, yB) → xA + yB = 0 ∧ yA + xB = 0

-- Define the distance function
def distance (A B : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

-- State the theorem
theorem distance_of_symmetric_points_on_parabola (A B : ℝ × ℝ) 
  (h_par : onParabola A B) 
  (h_symm : symmetric A B) : distance A B = 3 * Real.sqrt 2 :=
sorry

end distance_of_symmetric_points_on_parabola_l296_296480


namespace find_f_function_l296_296899

def oddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem find_f_function (f : ℝ → ℝ) (h_odd : oddFunction f) (h_pos : ∀ x, 0 < x → f x = x * (1 + x)) :
  ∀ x, x < 0 → f x = -x - x^2 :=
by
  sorry

end find_f_function_l296_296899


namespace polygon_stats_l296_296399

-- Definitions based on the problem's conditions
def total_number_of_polygons : ℕ := 207
def median_position : ℕ := 104
def m : ℕ := 14
def sum_of_squares_of_sides : ℕ := 2860
def mean_value : ℚ := sum_of_squares_of_sides / total_number_of_polygons
def mode_median : ℚ := 11.5

-- The proof statement
theorem polygon_stats (d μ M : ℚ)
  (h₁ : μ = mean_value)
  (h₂ : d = mode_median)
  (h₃ : M = m) :
  d < μ ∧ μ < M :=
by
  rw [h₁, h₂, h₃]
  -- The exact proof steps are omitted
  sorry

end polygon_stats_l296_296399


namespace sqrt_of_sum_of_powers_l296_296693

theorem sqrt_of_sum_of_powers : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_of_sum_of_powers_l296_296693


namespace persons_next_to_Boris_l296_296823

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l296_296823


namespace number_of_integers_satisfying_inequality_l296_296448

theorem number_of_integers_satisfying_inequality :
  { x : ℤ // 3 * x^2 + 8 * x + 5 ≤ 10 }.card = 5 :=
by
  sorry

end number_of_integers_satisfying_inequality_l296_296448


namespace white_line_longer_l296_296420

-- Define the lengths of the white and blue lines
def white_line_length : ℝ := 7.678934
def blue_line_length : ℝ := 3.33457689

-- State the main theorem
theorem white_line_longer :
  white_line_length - blue_line_length = 4.34435711 :=
by
  sorry

end white_line_longer_l296_296420


namespace constant_sums_l296_296426

theorem constant_sums (n : ℕ) 
  (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : x * y * z = 1) 
  : (x^n + y^n + z^n = 0 ∨ x^n + y^n + z^n = 3) ↔ (n = 1 ∨ n = 3) :=
by sorry

end constant_sums_l296_296426


namespace study_tour_part1_l296_296305

theorem study_tour_part1 (x y : ℕ) 
  (h1 : 45 * y + 15 = x) 
  (h2 : 60 * (y - 3) = x) : 
  x = 600 ∧ y = 13 :=
by sorry

end study_tour_part1_l296_296305


namespace portion_of_cake_l296_296784

theorem portion_of_cake (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) :
  let total_area := a * b in
  let grey_quadrilateral_area := total_area * (2 / 15) in
  grey_quadrilateral_area / total_area = 2 / 15 :=
by
  sorry

end portion_of_cake_l296_296784


namespace train_pass_time_l296_296773

def train_length : ℕ := 250
def train_speed_kmph : ℕ := 36
def station_length : ℕ := 200

def total_distance : ℕ := train_length + station_length

noncomputable def train_speed_mps : ℚ := (train_speed_kmph : ℚ) * 1000 / 3600

noncomputable def time_to_pass_station : ℚ := total_distance / train_speed_mps

theorem train_pass_time : time_to_pass_station = 45 := by
  sorry

end train_pass_time_l296_296773


namespace coeff_a_neg_one_in_expansion_l296_296070

open Finset

noncomputable def binom (n k : ℕ) : ℕ := (n.choose k)

theorem coeff_a_neg_one_in_expansion : 
  ∀ (a : ℝ), (∃ c : ℝ, c = -448 ∧ (∀ x : ℝ, (x + (-((1 + 2 * real.sqrt x) / x)))^8 = 
  ∑ k in range (8 + 1), binom 8 k * x^(8 - k) * (-(1 + 2 * real.sqrt x) / x)^k) → 
  coeff_a_neg_one_in_expansion):
  sorry

end coeff_a_neg_one_in_expansion_l296_296070


namespace constant_term_in_binomial_expansion_l296_296129

noncomputable def minimum_positive_integer_n : ℕ := 6

theorem constant_term_in_binomial_expansion (n : ℕ) (h : ∃ r : ℕ, (x^5 + 1/x)^n = binom(n,r) * x^(5 * n - 6 * r) ∧ 5 * n - 6 * r = 0) : n = minimum_positive_integer_n :=
sorry

end constant_term_in_binomial_expansion_l296_296129


namespace arithmetic_geom_seq_S5_l296_296897

theorem arithmetic_geom_seq_S5 (a_n : ℕ → ℚ) (S_n : ℕ → ℚ)
  (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * (1/2))
  (h_sum : ∀ n, S_n n = n * a_n 1 + (n * (n - 1) / 2) * (1/2))
  (h_geom_seq : (a_n 2) * (a_n 14) = (a_n 6) ^ 2) :
  S_n 5 = 25 / 2 :=
by
  sorry

end arithmetic_geom_seq_S5_l296_296897


namespace eval_infinite_product_l296_296045

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, (3:ℝ)^(2 * n / (3:ℝ)^n)

theorem eval_infinite_product : infinite_product = (3:ℝ)^(9 / 2) := by
  sorry

end eval_infinite_product_l296_296045


namespace rate_of_discount_is_20_l296_296719

def marked_price : ℝ := 150
def selling_price : ℝ := 120
def discount : ℝ := marked_price - selling_price
def rate_of_discount : ℝ := (discount / marked_price) * 100

theorem rate_of_discount_is_20 :
  rate_of_discount = 20 := by
  sorry

end rate_of_discount_is_20_l296_296719


namespace cost_effectiveness_l296_296635

variables (x : ℕ) -- the number of students participating in the competition
def plan1_cost (x : ℕ) : ℕ := 25 * x
def plan2_cost (x : ℕ) : ℕ := 20 * x + 20

theorem cost_effectiveness (x : ℕ) :
  (x > 4 → plan2_cost x < plan1_cost x) ∧
  (x = 4 → plan2_cost x = plan1_cost x) ∧
  (x < 4 → plan1_cost x < plan2_cost x) :=
by {
  split,
  { intro h,
    -- Plan 2 is more cost-effective when x > 4
    sorry
  },
  split,
  { intro h,
    -- Both plans are equally cost-effective when x = 4
    sorry
  },
  { intro h,
    -- Plan 1 is more cost-effective when x < 4
    sorry
  }
}

end cost_effectiveness_l296_296635


namespace measure_angle_ACB_l296_296768

-- Variables and conditions
variables {A B C : Type} [angle A B C : ℝ]
variable {isosceles_triangle : ∀ {A B C : Type}, Type}

-- Definitions based on conditions
def is_isosceles (A B C : Type) [isosceles_triangle A B C] : Prop := angle A = angle B

def angle_A_eq_80 (A B C : Type) [isosceles_triangle A B C] (h : is_isosceles A B C) : Prop :=
  angle A = 80

-- Theorem statement
theorem measure_angle_ACB (A B C : Type) [isosceles_triangle A B C] 
  (h : is_isosceles A B C) 
  (hA : angle_A_eq_80 A B C h) 
  : angle C = 20 := 
  sorry

end measure_angle_ACB_l296_296768


namespace Jane_possible_numbers_l296_296192

def is_factor (a b : ℕ) : Prop := b % a = 0
def in_range (n : ℕ) : Prop := 500 ≤ n ∧ n ≤ 4000

def Jane_number (m : ℕ) : Prop :=
  is_factor 180 m ∧
  is_factor 42 m ∧
  in_range m

theorem Jane_possible_numbers :
  Jane_number 1260 ∧ Jane_number 2520 ∧ Jane_number 3780 :=
by
  sorry

end Jane_possible_numbers_l296_296192


namespace find_a_l296_296570

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x + a

theorem find_a (a : ℝ) :
  (∃! x : ℝ, f x a = 0) → (a = -2 ∨ a = 2) :=
sorry

end find_a_l296_296570


namespace standing_next_to_boris_l296_296828

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l296_296828


namespace num_integers_satisfy_inequality_l296_296945

theorem num_integers_satisfy_inequality : 
  ∃ n : ℕ, n = 5 ∧ {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.finite.card = n :=
sorry

end num_integers_satisfy_inequality_l296_296945


namespace cd_length_correct_l296_296982
noncomputable def length_cd (r : ℝ) (AB : ℝ) (BP : ℝ) (PA : ℝ) : ℝ :=
  let x := sqrt (BP^2 + PA^2) in
  2 * x

noncomputable def given_problem := length_cd 8 16 5 8 = 2 * sqrt 89

theorem cd_length_correct : given_problem :=
by
  -- The proof goes here
  sorry

end cd_length_correct_l296_296982


namespace max_min_sum_eq_two_l296_296489

noncomputable def f (x : ℝ) : ℝ := (exp (2 * x) - (exp x) * (sin x) + 1) / (exp (2 * x) + 1)

theorem max_min_sum_eq_two : 
  let M := ⨆ x, f x in  -- Supremum (maximum value)
  let m := ⨅ x, f x in  -- Infimum (minimum value)
  M + m = 2 :=
sorry

end max_min_sum_eq_two_l296_296489


namespace trapezoid_constructible_l296_296847

theorem trapezoid_constructible (a c e f : ℕ) (ha : a = 8) (hc : c = 4) (he : e = 9) (hf : f = 15) :
  ∃ (A B C D : ℝ × ℝ),
  let AB := a,
      CD := c,
      AC := e,
      BD := f in
  A = (0, 0) ∧ B = (8, 0) ∧ D = (0, some h) ∧ C = (4, some h) ∧ 
  AC = 9 ∧ BD = 15 ∧ AB = 8 ∧ CD = 4 :=
sorry

end trapezoid_constructible_l296_296847


namespace continuous_compound_interest_solution_l296_296316

noncomputable def continuous_compound_interest_rate 
  (A P: ℝ) (t: ℝ) (h_A_value: A = 760) (h_P_value: P = 600) (h_t_value: t = 4) : ℝ :=
  (Real.log (A / P)) / t

theorem continuous_compound_interest_solution :
  continuous_compound_interest_rate 760 600 4 (by norm_num) (by norm_num) (by norm_num) ≈ 0.05909725 :=
by
  unfold continuous_compound_interest_rate
  norm_num
  rw [← Real.log_div]
  sorry

end continuous_compound_interest_solution_l296_296316


namespace number_of_paths_l296_296442

theorem number_of_paths (paths_A_to_B paths_B_to_D paths_D_to_C direct_paths_A_to_C : ℕ) 
  (h1 : paths_A_to_B = 2) (h2 : paths_B_to_D = 2) (h3 : paths_D_to_C = 2) (h4 : direct_paths_A_to_C = 2) :
  paths_A_to_B * paths_B_to_D * paths_D_to_C + direct_paths_A_to_C = 10 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  apply eq.refl

end number_of_paths_l296_296442


namespace triangle_inequality_l296_296627

theorem triangle_inequality (a b c m_A : ℝ)
  (h1 : 2*m_A ≤ b + c)
  (h2 : a^2 + (2*m_A)^2 = (b^2) + (c^2)) :
  a^2 + 4*m_A^2 ≤ (b + c)^2 :=
by {
  sorry
}

end triangle_inequality_l296_296627


namespace calculate_dimensions_l296_296167

noncomputable def room_dimensions (a b c : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let x := real.sqrt (real.sqrt ((b^2 * (b^2 - a^2)) / (b^2 - c^2)))
  let y := a / x
  let z := b / x
  let u := (c * x) / b
  (x, y, z, u)

theorem calculate_dimensions (a b c : ℝ)
  (h_b_pos : b ≠ 0) (h_b_c_neq : b ≠ c) : 
  let res := room_dimensions a b c in
  res.1 = real.sqrt (real.sqrt ((b^2 * (b^2 - a^2)) / (b^2 - c^2))) ∧
  res.2 = a / res.1 ∧
  res.3 = b / res.1 ∧
  res.4 = (c * res.1) / b :=
by
  obtain ⟨x, y, z, u⟩ := room_dimensions a b c
  split; simp only [room_dimensions]; sorry

end calculate_dimensions_l296_296167


namespace quadratic_function_translation_l296_296482

theorem quadratic_function_translation :
  ∃ f : ℝ → ℝ, (∀ x, f(x) = -2 (x+1)^2 + 4 (x+1) + 1 - 2) → (f(x) = -2 x^2 + 1) :=
by
  sorry

end quadratic_function_translation_l296_296482


namespace min_value_expr_l296_296124

theorem min_value_expr (a d : ℝ) (b c : ℝ) (h_a : 0 ≤ a) (h_d : 0 ≤ d) (h_b : 0 < b) (h_c : 0 < c) (h : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end min_value_expr_l296_296124


namespace coeff_linear_term_l296_296776

-- Define the quadratic equation
def quadratic_eqn1 : ℝ → ℝ := λ x, 2 * x^2 - 3 * x + 1

-- Define the second form of quadratic equation
def quadratic_eqn2 : ℝ → ℝ := λ x, 2 * x^2 - 3 * x + 1

-- The coefficient of the linear term in the given quadratic equation as a proof statement
theorem coeff_linear_term (x : ℝ) : 
  quadratic_eqn1 x = quadratic_eqn2 x → 
  by sorry


end coeff_linear_term_l296_296776


namespace esperanza_gross_salary_l296_296232

def rent : ℕ := 600
def food_expenses (rent : ℕ) : ℕ := 3 * rent / 5
def mortgage_bill (food_expenses : ℕ) : ℕ := 3 * food_expenses
def savings : ℕ := 2000
def taxes (savings : ℕ) : ℕ := 2 * savings / 5
def total_expenses (rent food_expenses mortgage_bill taxes : ℕ) : ℕ :=
  rent + food_expenses + mortgage_bill + taxes
def gross_salary (total_expenses savings : ℕ) : ℕ :=
  total_expenses + savings

theorem esperanza_gross_salary : 
  gross_salary (total_expenses rent (food_expenses rent) (mortgage_bill (food_expenses rent)) (taxes savings)) savings = 4840 :=
by
  sorry

end esperanza_gross_salary_l296_296232


namespace rectangle_area_solution_l296_296260

theorem rectangle_area_solution (x : ℝ) (hx : x > 0) 
  (E F G H : ℝ × ℝ)
  (hE : E = (0, 0))
  (hF : F = (0, 5))
  (hG : G = (x, 5))
  (hH : H = (x, 0))
  (h_area : 5 * x = 35) :
  x = 7 :=
begin
  -- proof goes here
  sorry
end

end rectangle_area_solution_l296_296260


namespace angle_opposite_side_c_l296_296652

theorem angle_opposite_side_c (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 3 * a * b) :
  real.cos (real.angle_of_deg 60) = 1 / 2 :=
by sorry

end angle_opposite_side_c_l296_296652


namespace range_of_a_with_six_monotonic_intervals_l296_296095

noncomputable def f (x a b : ℝ) : ℝ := (1 / 3) * |x^3| - (a / 2) * x^2 + (3 - a) * |x| + b

theorem range_of_a_with_six_monotonic_intervals :
  (∃ a b : ℝ, (∀ x : ℝ, f x a b = f (-x) a b) ∧ (∃ x > 0, ∃ y > x,
    ∀ z ∈ Icc (0 : ℝ) x, f z a b ≤ f x a b ∧ f x a b < f y a b) → 2 < a ∧ a < 3) :=
sorry

end range_of_a_with_six_monotonic_intervals_l296_296095


namespace find_a_l296_296698

theorem find_a : 
  ∃ a : ℝ, (a > 0) ∧ (1 / Real.logb 5 a + 1 / Real.logb 6 a + 1 / Real.logb 7 a = 1) ∧ a = 210 :=
by
  sorry

end find_a_l296_296698


namespace intermediate_circle_radius_l296_296547

theorem intermediate_circle_radius (r1 r3: ℝ) (h1: r1 = 5) (h2: r3 = 13) 
  (h3: π * r1 ^ 2 = π * r3 ^ 2 - π * r2 ^ 2) : r2 = 12 := sorry


end intermediate_circle_radius_l296_296547


namespace correct_system_of_equations_l296_296275

variable (x y : ℕ) -- We assume non-negative numbers for counts of chickens and rabbits

theorem correct_system_of_equations :
  (x + y = 35) ∧ (2 * x + 4 * y = 94) ↔
  (∃ (a b : ℕ), a = x ∧ b = y) :=
by
  sorry

end correct_system_of_equations_l296_296275


namespace not_perfect_square_of_sum_300_l296_296182

theorem not_perfect_square_of_sum_300 : ¬(∃ n : ℕ, n = 10^300 - 1 ∧ (∃ m : ℕ, n = m^2)) :=
by
  sorry

end not_perfect_square_of_sum_300_l296_296182


namespace area_of_quadrilateral_DEFG_l296_296973

variable {A B C D E F G : Type} [NormedAddCommGroup A] [NormedAddCommGroup B]
variable (area : B → ℝ) -- Assume a function to calculate the area

-- Equality for area of the parallelogram
variable (h_area_parallelogram : area ABCD = 100)
-- Midpoint properties
variable (h_midpoint_E : to_real A + to_real D = 2 * to_real E)
variable (h_midpoint_G : to_real C + to_real D = 2 * to_real G)
variable (h_midpoint_F : to_real B + to_real C = 2 * to_real F)

theorem area_of_quadrilateral_DEFG :
  area DEFG = 25 :=
sorry

end area_of_quadrilateral_DEFG_l296_296973


namespace max_additional_spheres_in_cone_l296_296373

-- Definition of spheres O_{1} and O_{2} properties
def O₁_radius : ℝ := 2
def O₂_radius : ℝ := 3
def height_cone : ℝ := 8

-- Conditions:
def O₁_on_axis (h : ℝ) := height_cone > 0 ∧ h = O₁_radius
def O₁_tangent_top_base := height_cone = O₁_radius + O₁_radius
def O₂_tangent_O₁ := O₁_radius + O₂_radius = 5
def O₂_on_base := O₂_radius = 3

-- Lean theorem stating mathematically equivalent proof problem
theorem max_additional_spheres_in_cone (h : ℝ) :
  O₁_on_axis h → O₁_tangent_top_base →
  O₂_tangent_O₁ → O₂_on_base →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end max_additional_spheres_in_cone_l296_296373


namespace find_a8_l296_296278

noncomputable def a_seq (a_1 a_2 : ℕ) : ℕ → ℕ
| 0     := a_1
| 1     := a_2
| (n+2) := a_seq n.succ (a_seq n) + a_seq n

theorem find_a8 (a_1 a_2 : ℕ) (h_pos1 : 0 < a_1) (h_pos2 : 0 < a_2) (h7 : a_seq a_1 a_2 6 = 210) :
  a_seq a_1 a_2 7 = 340 :=
sorry

end find_a8_l296_296278


namespace smallest_positive_angle_l296_296437

theorem smallest_positive_angle (y : ℝ) (h : sin (3 * y) * sin (4 * y) = cos (3 * y) * cos (4 * y)) :
  y = 90 / 7 :=
  sorry

end smallest_positive_angle_l296_296437


namespace predicted_temperature_l296_296064

def avg_x (x_vals : List ℕ) : ℕ := (x_vals.foldl (· + ·) 0) / x_vals.length
def avg_y (y_vals : List ℕ) : ℕ := (y_vals.foldl (· + ·) 0) / y_vals.length

theorem predicted_temperature (k : ℚ) (x_vals y_vals : List ℚ) (x : ℕ) (H : (avg_x x_vals = 40) ∧ (avg_y y_vals = 30) ∧ k = 20) :
  0.25 * 80 + k = 40 :=
by
  sorry

end predicted_temperature_l296_296064


namespace trapezium_area_l296_296428

theorem trapezium_area (a b h : ℝ) (h_a : a = 20) (h_b : b = 18) (h_h : h = 5) :
  (1 / 2) * (a + b) * h = 95 :=
by
  rw [h_a, h_b, h_h]
  norm_num
  sorry

end trapezium_area_l296_296428


namespace cherry_tomatoes_ratio_l296_296042

theorem cherry_tomatoes_ratio (T P B : ℕ) (M : ℕ := 3) (h1 : P = 4 * T) (h2 : B = 4 * P) (h3 : B / 3 = 32) :
  (T : ℚ) / M = 2 :=
by
  sorry

end cherry_tomatoes_ratio_l296_296042


namespace geom_prog_common_ratio_unique_l296_296782

theorem geom_prog_common_ratio_unique (b q : ℝ) (hb : b > 0) (hq : q > 1) :
  (∃ b : ℝ, (q = (1 + Real.sqrt 5) / 2) ∧ 
    (0 < b ∧ b * q ≠ b ∧ b * q^2 ≠ b ∧ b * q^3 ≠ b) ∧ 
    ((2 * b * q = b + b * q^2) ∨ (2 * b * q = b + b * q^3) ∨ (2 * b * q^2 = b + b * q^3))) := 
sorry

end geom_prog_common_ratio_unique_l296_296782


namespace number_of_real_values_p_equal_roots_l296_296221

theorem number_of_real_values_p_equal_roots (p : ℝ) :
  let eq_quadratic := (λ x : ℝ, x^2 - p * x + p)
  let roots_equal (p : ℝ) := (p ^ 2 - 4 * p = 0)
  (∃ (a b : ℝ), eq_quadratic a = 0 ∧ eq_quadratic b = 0 ∧ a = b) ↔ (p = 0 ∨ p = 4) :=
by { sorry }

end number_of_real_values_p_equal_roots_l296_296221


namespace largest_equilateral_triangle_from_square_l296_296257

-- Define the setup of the square and necessary folds
structure Square (A B C D: Point)

-- Hypothesize the folding steps to form the triangle AJK

theorem largest_equilateral_triangle_from_square 
  (s : Square A B C D)
  (fold1 : half_fold A B C D)
  (fold2 : AB_Fold_to_FE B fold1)
  (fold3 : Unfold_and_Fold AB AH fold2)
  : ∃ triangle AJK, is_largest_equilateral_triangle AJK :=
sorry

end largest_equilateral_triangle_from_square_l296_296257


namespace number_of_integers_satisfying_inequality_l296_296937

theorem number_of_integers_satisfying_inequality : 
  {n : Int | (n - 3) * (n + 5) < 0}.card = 7 :=
by
  sorry

end number_of_integers_satisfying_inequality_l296_296937


namespace product_of_three_consecutive_integers_is_multiple_of_6_l296_296710

theorem product_of_three_consecutive_integers_is_multiple_of_6 (n : ℕ) (h : n > 0) :
    ∃ k : ℕ, n * (n + 1) * (n + 2) = 6 * k :=
by
  sorry

end product_of_three_consecutive_integers_is_multiple_of_6_l296_296710


namespace least_number_to_subtract_l296_296703

/-- 
Given a number 42398, prove that the least number to be subtracted 
from it, so that the resulting number is divisible by 15, is 8.
-/
theorem least_number_to_subtract (n : ℕ) (h₁ : n = 42398) : 
  ∃ k : ℕ, n - k = 15 * (n / 15) := 
begin
  sorry
end

end least_number_to_subtract_l296_296703


namespace volume_unoccupied_space_l296_296671

-- Definitions of the conditions in Lean

-- The radius of the cones and the cylinder.
def radius_cone_cylinder : ℝ := 15

-- The height of each cone.
def height_cone : ℝ := 10

-- The height of the cylinder.
def height_cylinder : ℝ := 30

-- The volume of a cylinder given its radius and height.
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- The volume of a cone given its radius and height.
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Statement: The volume of space inside the cylinder not occupied by the cones.
theorem volume_unoccupied_space :
  volume_cylinder radius_cone_cylinder height_cylinder - 
  2 * volume_cone radius_cone_cylinder height_cone = 5250 * π :=
by
  sorry

end volume_unoccupied_space_l296_296671


namespace pyramid_properties_l296_296662

open Real EuclideanGeometry

-- Define the points O, A, B, C
def O : EuclideanSpace ℝ (fin 3) := ![0, 0, 0]
def A : EuclideanSpace ℝ (fin 3) := ![5, 2, 0]
def B : EuclideanSpace ℝ (fin 3) := ![2, 5, 0]
def C : EuclideanSpace ℝ (fin 3) := ![1, 2, 4]

-- Define vectors AB, AC, AO
def vector_AB : EuclideanSpace ℝ (fin 3) := B - A
def vector_AC : EuclideanSpace ℝ (fin 3) := C - A
def vector_AO : EuclideanSpace ℝ (fin 3) := O - A

-- Given calculated values
noncomputable def volume_OABC : ℝ := 14
noncomputable def area_ABC : ℝ := 6 * Real.sqrt 3
noncomputable def height_OD : ℝ := 7 * Real.sqrt 3 / 3

-- Proof statement
theorem pyramid_properties :
  let V := 1 / 6 * Real.abs (vector_AB.1 * (vector_AC.2 * vector_AO.3 - vector_AC.3 * vector_AO.2)
                         - vector_AB.2 * (vector_AC.1 * vector_AO.3 - vector_AC.3 * vector_AO.1)
                         + vector_AB.3 * (vector_AC.1 * vector_AO.2 - vector_AC.2 * vector_AO.1)) in
  let S := 1 / 2 * Real.sqrt ((vector_AB.2 * vector_AC.3 - vector_AB.3 * vector_AC.2)^2
                            + (vector_AB.3 * vector_AC.1 - vector_AB.1 * vector_AC.3)^2
                            + (vector_AB.1 * vector_AC.2 - vector_AB.2 * vector_AC.1)^2) in
  V = volume_OABC ∧ S = area_ABC ∧ (3 * V) / S = height_OD := by {
  -- The detailed proof will be provided here
  sorry
}

end pyramid_properties_l296_296662


namespace words_per_page_l296_296734

theorem words_per_page 
    (p : ℕ) 
    (h1 : 150 > 0) 
    (h2 : 150 * p ≡ 200 [MOD 221]) :
    p = 118 := 
by sorry

end words_per_page_l296_296734


namespace average_mpg_is_22_point_7_l296_296041

-- Definitions of the input conditions
def initial_odometer : ℝ := 57300
def final_odometer : ℝ := 58300
def first_refill : ℝ := 8
def second_refill : ℝ := 14
def third_refill : ℝ := 22

-- Total distance traveled
def total_distance : ℝ := final_odometer - initial_odometer

-- Total gasoline used
def total_gasoline : ℝ := first_refill + second_refill + third_refill

-- Average miles per gallon (MPG)
def average_mpg : ℝ := total_distance / total_gasoline

-- Statement to prove
theorem average_mpg_is_22_point_7 : (Float.round (average_mpg * 10) / 10) = 22.7 := by
  sorry

end average_mpg_is_22_point_7_l296_296041


namespace area_of_EFGH_l296_296457

-- Definitions based on given conditions
def shorter_side : ℝ := 4
def longer_side : ℝ := 8
def smaller_rectangle_area : ℝ := shorter_side * longer_side
def larger_rectangle_width : ℝ := longer_side
def larger_rectangle_height : ℝ := 2 * longer_side

-- Theorem stating the area of the larger rectangle
theorem area_of_EFGH : larger_rectangle_width * larger_rectangle_height = 128 := by
  -- Proof goes here
  sorry

end area_of_EFGH_l296_296457


namespace factorize_polynomial_l296_296335

theorem factorize_polynomial :
  (∀ x : ℤ, x^{15} + x^{10} + 1 = (x^2 + x + 1) * (x^{13} - x^{12} + x^{10} - x^9 + x^7 - x^6 + x^4 - x^3 + x) + 1) :=
by sorry

end factorize_polynomial_l296_296335


namespace line_equation_45_deg_through_point_l296_296908

theorem line_equation_45_deg_through_point :
  ∀ (x y : ℝ), 
  (∃ m k: ℝ, m = 1 ∧ k = 5 ∧ y = m * x + k) ∧ (∃ p q : ℝ, p = -2 ∧ q = 3 ∧ y = q ) :=  
  sorry

end line_equation_45_deg_through_point_l296_296908


namespace standing_next_to_boris_l296_296834

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l296_296834


namespace smallest_sum_l296_296056

theorem smallest_sum : min (
  (1/4 + 1/5),
  min (
    (1/4 + 1/6),
    min (
      (1/4 + 1/9),
      min (
        (1/4 + 1/8),
        (1/4 + 1/7)
      )
    )
  )
) = 13/36 := by sorry

end smallest_sum_l296_296056


namespace percentage_sales_other_l296_296259

theorem percentage_sales_other (p_pens p_pencils p_markers p_other : ℕ)
(h_pens : p_pens = 25)
(h_pencils : p_pencils = 30)
(h_markers : p_markers = 20)
(h_other : p_other = 100 - (p_pens + p_pencils + p_markers)): p_other = 25 :=
by
  rw [h_pens, h_pencils, h_markers] at h_other
  exact h_other


end percentage_sales_other_l296_296259


namespace max_points_on_circles_l296_296456

noncomputable def max_intersections (C1 C2 C3 C4 : Circle) (L : Line) : ℕ :=
  let intersections (C : Circle) : ℕ := 2
  intersections C1 + intersections C2 + intersections C3 + intersections C4

theorem max_points_on_circles :
  ∀ (C1 C2 C3 C4 : Circle) (L : Line),
    coplanar {C1, C2, C3, C4} →
    (∀ C : Circle, C ∈ {C1, C2, C3, C4} → radius(C) = r) →
    centers_make_rough_square {center C1, center C2, center C3, center C4} →
    max_intersections C1 C2 C3 C4 L = 8 :=
begin
  intros,
  sorry
end

end max_points_on_circles_l296_296456


namespace john_younger_than_mark_l296_296227

variable (Mark_age John_age Parents_age : ℕ)
variable (h_mark : Mark_age = 18)
variable (h_parents_age_relation : Parents_age = 5 * John_age)
variable (h_parents_when_mark_born : Parents_age = 22 + Mark_age)

theorem john_younger_than_mark : Mark_age - John_age = 10 :=
by
  -- We state the theorem and leave the proof as sorry
  sorry

end john_younger_than_mark_l296_296227


namespace neg_p_necessary_not_sufficient_for_q_l296_296574

-- Conditions
def p (x : ℝ) : Prop := -1 < x ∧ x < 3
def q (x : ℝ) : Prop := x > 5

-- Negation of p
def ¬p (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 3

-- Statement
theorem neg_p_necessary_not_sufficient_for_q (x : ℝ) :
  (∀ x, q x → ¬p x) ∧ ¬(∀ x, ¬p x → q x) :=
by
  sorry

end neg_p_necessary_not_sufficient_for_q_l296_296574


namespace hypotenuse_of_right_triangle_l296_296330

theorem hypotenuse_of_right_triangle (x y : ℝ) (h1 : (1 / 3) * π * y * x^2 = 256 * π) 
(h2 : (1 / 3) * π * x * y^2 = 2304 * π) :
  real.sqrt (x^2 + y^2) = 2 * real.sqrt 82 :=
sorry

end hypotenuse_of_right_triangle_l296_296330


namespace problem_statement_l296_296212

-- Define P(t) as the number of 1's in the binary representation of t
def P (t : ℕ) : ℕ := (t.bits.map (λ b, if b then 1 else 0)).sum

-- Define f_t(n) as the number of C_k^t that are odd, for 1 ≤ k ≤ n and integer k
def f_t (t n : ℕ) : ℕ := ((list.range (n + 1)).filter (λ k, (C(t, k)).val % 2 = 1)).length

-- Main theorem statement
theorem problem_statement (t : ℕ) (n : ℕ) (h : ∃ k, n = 2^k) (ht_pos : 0 < t) (hk_pos : 0 < n) :
  (f_t t n : ℚ) / n = 1 / 2^(P t) :=
sorry

end problem_statement_l296_296212


namespace who_is_next_to_boris_l296_296810

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l296_296810


namespace angle_DPC_is_120_l296_296835

variable (A B C D E P: Type) 
variables [linear_ordered_field A]

/-- 
    In a triangle ABC, BC > AC and ∠A = 60°. Points D and E are the midpoints 
    of AB and AC respectively. The line segment PC bisects ∠ACB and PD bisects ∠ADE. 
    Prove that ∠DPC = 120°.
--/
theorem angle_DPC_is_120 
  (BC_AC : (BC:E → E) > (AC:E → E)) 
  (angle_A : angle A = 60°) 
  (midpoint_D : midpoint D A B) 
  (midpoint_E : midpoint E A C) 
  (bisect_PACB : bisect P C A C B) 
  (bisect_PDADE : bisect P D A D E) : 
  angle D P C = 120° := 
sorry

end angle_DPC_is_120_l296_296835


namespace who_is_next_to_Boris_l296_296814

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l296_296814


namespace monotonicity_increasing_l296_296581

-- Conditions: Define the function and constraints
noncomputable def f (ω φ x : ℝ) : ℝ := sin (ω * x + φ) - sqrt 3 * cos (ω * x + φ)

-- Condition: ω > 0
def ω_pos (ω : ℝ) : Prop := ω > 0

-- Condition: |φ| < π / 2
def φ_bounds (φ : ℝ) : Prop := |φ| < π / 2

-- Condition: The minimum positive period of f(x) is π
def min_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Condition: f(-x) = f(x) (symmetry)
def f_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem: f(x) is monotonically increasing in (0, π / 2)
theorem monotonicity_increasing :
  ∀ (ω φ : ℝ),
    ω_pos ω →
    φ_bounds φ →
    min_period (f ω φ) π →
    f_symmetry (f ω φ) →
    ∀ x, 0 < x → x < π / 2 → f ω φ x < f ω φ (x + π / 4) :=
by sorry

end monotonicity_increasing_l296_296581


namespace range_of_y0_l296_296025

theorem range_of_y0
  (y0 : ℝ)
  (h_tangent : ∃ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2) = 1))
  (h_angle : ∀ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2 = 1)) → (Real.arccos ((Real.sqrt 3 - N.1)/Real.sqrt ((3 - 2 * N.1 * Real.sqrt 3 + N.1^2) + (y0 - N.2)^2)) ≥ π / 6)) :
  -1 ≤ y0 ∧ y0 ≤ 1 :=
by
  sorry

end range_of_y0_l296_296025


namespace triangle_type_l296_296188

theorem triangle_type (A B C : ℝ) (a b c : ℝ)
  (h1 : B = 30) 
  (h2 : c = 15) 
  (h3 : b = 5 * Real.sqrt 3) 
  (h4 : a ≠ 0) 
  (h5 : b ≠ 0)
  (h6 : c ≠ 0) 
  (h7 : 0 < A ∧ A < 180) 
  (h8 : 0 < B ∧ B < 180) 
  (h9 : 0 < C ∧ C < 180) 
  (h10 : A + B + C = 180) : 
  (A = 90 ∨ A = C) ∧ A + B + C = 180 :=
by 
  sorry

end triangle_type_l296_296188


namespace joe_total_paint_used_l296_296193

-- Define the initial amount of paint Joe buys.
def initial_paint : ℕ := 360

-- Define the fraction of paint used during the first week.
def first_week_fraction := 1 / 4

-- Define the fraction of remaining paint used during the second week.
def second_week_fraction := 1 / 2

-- Define the total paint used by Joe in the first week.
def paint_used_first_week := first_week_fraction * initial_paint

-- Define the remaining paint after the first week.
def remaining_paint_after_first_week := initial_paint - paint_used_first_week

-- Define the total paint used by Joe in the second week.
def paint_used_second_week := second_week_fraction * remaining_paint_after_first_week

-- Define the total paint used by Joe.
def total_paint_used := paint_used_first_week + paint_used_second_week

-- The theorem to be proven: the total amount of paint Joe has used is 225 gallons.
theorem joe_total_paint_used : total_paint_used = 225 := by
  sorry

end joe_total_paint_used_l296_296193


namespace f_2015_l296_296122

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (x + 2) = f (2 - x) + 4 * f 2
axiom symmetric_about_neg1 : ∀ x : ℝ, f (x + 1) = f (-2 - (x + 1))
axiom f_at_1 : f 1 = 3

theorem f_2015 : f 2015 = -3 :=
by
  apply sorry

end f_2015_l296_296122


namespace find_t_l296_296504

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the perpendicular condition and solve for t
theorem find_t (t : ℝ) : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 → t = -2 :=
by
  sorry

end find_t_l296_296504


namespace minimal_cost_is_97_5_l296_296625

-- Definitions of costs per flower type
def cost_rose := 1
def cost_tulip := 1.75
def cost_sunflower := 2.25
def cost_peony := 3
def cost_orchid := 3.50

-- Areas of garden sections
def area_top_left := 7 * 2
def area_bottom_left := 7 * 4
def area_top_right := 3 * 2
def area_bottom_right := 3 * 4

-- The least possible cost to plant the flowers in the entire garden
def minimal_cost : Real := 28 * cost_rose + 14 * cost_tulip + 12 * cost_sunflower + 6 * cost_peony

theorem minimal_cost_is_97_5 : minimal_cost = 97.5 := by
  sorry

end minimal_cost_is_97_5_l296_296625


namespace maximum_tickets_l296_296656

-- Define the conditions
def ticket_price : ℝ := 18
def total_money : ℝ := 150

-- Define the maximum number of tickets Tom can buy
def max_tickets (price : ℝ) (money : ℝ) : ℕ :=
  ⌊money / price⌋.to_nat  -- Floor division ensures we get whole number, and to_nat converts to natural number

-- The theorem we want to prove
theorem maximum_tickets : max_tickets ticket_price total_money = 8 :=
  by sorry

end maximum_tickets_l296_296656


namespace rectangle_perimeter_is_28_l296_296419

-- Define the variables and conditions
variables (h w : ℝ)

-- Problem conditions
def rectangle_area (h w : ℝ) : Prop := h * w = 40
def width_greater_than_twice_height (h w : ℝ) : Prop := w > 2 * h
def parallelogram_area (h w : ℝ) : Prop := h * (w - h) = 24

-- The theorem stating the perimeter of the rectangle given the conditions
theorem rectangle_perimeter_is_28 (h w : ℝ) 
  (H1 : rectangle_area h w) 
  (H2 : width_greater_than_twice_height h w) 
  (H3 : parallelogram_area h w) :
  2 * h + 2 * w = 28 :=
sorry

end rectangle_perimeter_is_28_l296_296419


namespace smallest_period_of_f_intervals_where_f_is_decreasing_l296_296133

open Real
open Int

noncomputable def f (x : ℝ) := 4 * sin x * cos (x - π / 3) - sqrt 3

theorem smallest_period_of_f : ∃ T > 0, (∀ x, f(x + T) = f(x)) ∧ T = π := by
  sorry

theorem intervals_where_f_is_decreasing : 
  ∀ k : ℤ, ∃ a b, a = k * π + 5 * π / 12 ∧ b = k * π + 11 * π / 12 ∧ 
  ∀ x ∈ Icc a b, (∀ x₁ ∈ Ioo a b, deriv f x₁ < 0) := by
  sorry

end smallest_period_of_f_intervals_where_f_is_decreasing_l296_296133


namespace general_line_eq_rectangular_curve_eq_segment_length_l296_296907

def parametric_line_eq := {t : ℝ // True} → ℝ × ℝ
| ⟨t, _⟩ => ( (1/2) * t, ( (sqrt 3)/2 ) * t - 1 )

def polar_curve_eq (ρ θ : ℝ) := ρ^2 - 2 * ρ * sin θ - 3 = 0

theorem general_line_eq : ∃ (m b : ℝ), ∀ x y : ℝ,
  (∃ t : ℝ, x = (1/2) * t ∧ y = ( (sqrt 3)/2 ) * t - 1) ↔ y = m * x + b :=
sorry

theorem rectangular_curve_eq : ∃ (x y : ℝ), 
  (∃ (ρ θ : ℝ), polar_curve_eq ρ θ ∧ x = ρ * cos θ ∧ y = ρ * sin θ) ↔ (x - 0)^2 + (y - 1)^2 = 4 :=
sorry

theorem segment_length :
  let d := abs((-1 - 1) / (sqrt(3 + 1))) in
  ∃ length : ℝ, length = 2 * sqrt(2^2 - d^2) :=
sorry

end general_line_eq_rectangular_curve_eq_segment_length_l296_296907


namespace triangle_is_isosceles_l296_296613

variables {A B C M N : Type} [EuclideanGeometry A B C M N]

theorem triangle_is_isosceles 
  (hABC : triangle A B C) 
  (hM : OnSide M A B) 
  (hN : OnSide N B C) 
  (h1 : Perimeter (triangle A M C) = Perimeter (triangle C N A))
  (h2 : Perimeter (triangle A N B) = Perimeter (triangle C M B)) :
  IsIsosceles (triangle A B C) := 
sorry

end triangle_is_isosceles_l296_296613


namespace number_of_values_not_satisfied_l296_296449

-- Define the inequality condition for values of x
def inequality_not_satisfied (x : ℤ) : Prop :=
  3 * x^2 + 8 * x + 5 ≤ 10

-- Define a set of integers in the interval
def integers_in_interval (a b : ℤ) : set ℤ :=
  {x | a ≤ x ∧ x ≤ b}

-- Define the set of values of x where the inequality is not satisfied in the given interval
def values_not_satisfied : finset ℤ :=
  (finset.filter inequality_not_satisfied ⟨{-5, -4, -3, -2, -1, 0, 1}, sorry⟩).to_finset

-- The main theorem stating the number of integer solutions where the inequality is not satisfied
theorem number_of_values_not_satisfied : values_not_satisfied.card = 6 :=
by
  sorry

end number_of_values_not_satisfied_l296_296449


namespace bagel_cuts_l296_296733

theorem bagel_cuts (n : ℕ) (h : n = 10) : pieces_after_cuts n = 11 := sorry

def pieces_after_cuts : ℕ → ℕ
| n => n + 1

end bagel_cuts_l296_296733


namespace distance_point_to_line_l296_296430

theorem distance_point_to_line (x y : ℝ) :
  let distance := abs (y - x) / real.sqrt 2 in
  distance = (abs (y - x) / real.sqrt 2) :=
by
  sorry

end distance_point_to_line_l296_296430


namespace find_sum_of_powers_of_roots_l296_296119

-- Definitions and conditions
def is_root (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

def quadratic_polynomial (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c

variable (α β : ℝ)

-- Given conditions are structured into the Lean statement
theorem find_sum_of_powers_of_roots
  (hα : is_root (quadratic_polynomial 1 (-4) 1) α)
  (hβ : is_root (quadratic_polynomial 1 (-4) 1) β) :
  7 * α^3 + 3 * β^4 = 1019 :=
by
  sorry

end find_sum_of_powers_of_roots_l296_296119


namespace tangents_intersection_on_median_l296_296990

theorem tangents_intersection_on_median
  (ABC : Triangle)
  (B K E F : Point)
  (s : Circle)
  (HAcuteAngle : acute_angle ABC)
  (HAltitude : altitude B K)
  (HcircleSEF : s.diameter = segment B K ∧ s.touches AB at E ∧ s.touches BC at F)
  (Htangents : tangent s at E ∧ tangent s at F) :
  Intersection (tangent s at E) (tangent s at F) ∈ median ABC B :=
sorry

end tangents_intersection_on_median_l296_296990


namespace polygon_sides_l296_296522

theorem polygon_sides (n : ℕ) : 
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by
  sorry

end polygon_sides_l296_296522


namespace solve_triangle_l296_296848

section TriangleConstruction

def TriangleConstruction (m_a m_b c : ℝ) : 
  (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) :=
  let a := 24.82
  let b := 10.818
  let α := 61 + 55/60 + 33/3600
  let β := 22 + 37/60 + 10/3600
  let γ := 95 + 27/60 + 17/3600
  let a_1 := 39
  let b_1 := 17
  let α_1 := 118 + 4/60 + 27/3600
  let β_1 := 22 + 37/60 + 10/3600
  let γ_1 := 39 + 18/60 + 23/3600
  ((a, b, α, β, γ, c), (a_1, b_1, α_1, β_1, γ_1, c))

theorem solve_triangle (m_a m_b c : ℝ) (h_m_a : m_a = 10.769) (h_m_b : m_b = 24.706) (h_c : c = 28) : 
  TriangleConstruction m_a m_b c = 
    ((24.82, 10.818, 61 + 55/60 + 33/3600, 22 + 37/60 + 10/3600, 95 + 27/60 + 17/3600, 28),
     (39, 17, 118 + 4/60 + 27/3600, 22 + 37/60 + 10/3600, 39 + 18/60 + 23/3600, 28)) :=
by
  rw [h_m_a, h_m_b, h_c]
  simp only [TriangleConstruction]

end TriangleConstruction

end solve_triangle_l296_296848


namespace floor_neg_seven_thirds_l296_296080

theorem floor_neg_seven_thirds : Int.floor (-7 / 3 : ℚ) = -3 := by
  sorry

end floor_neg_seven_thirds_l296_296080


namespace more_brunettes_than_blondes_l296_296539

-- Definitions for conditions:
def students : Type := list bool -- Represents a list of students as booleans (true for brunettes, false for blondes)
def is_blonde : bool → Prop := λ b, ¬b
def is_brunette : bool → Prop := λ b, b
def lies : bool → Prop := λ b, ¬b
def tells_truth : bool → Prop := λ b, b
def dyes_hair (d : bool) : bool := λ b, ¬b
def affirmative (c : bool) : Prop := c = true

-- Problem conditions represented in Lean
variables (students : list bool)
#eval list.length students -- There are 20 students in total
def on_monday_20_yes : Prop := list.length (students.filter tells_truth) + list.length (students.filter lies) = 20
def on_friday_10_yes : Prop := list.length (students.filter is_brunette) + list.length (students.filter λ x, lies (dyes_hair x)) = 10
def next_monday_0_yes : Prop := list.length (students.filter (λ x, tells_truth (dyes_hair x))) = 0

-- Proof problem statement
theorem more_brunettes_than_blondes :
  on_monday_20_yes students →
  on_friday_10_yes students →
  next_monday_0_yes students →
  list.length (students.filter is_brunette) > list.length (students.filter is_blonde) :=
sorry

end more_brunettes_than_blondes_l296_296539


namespace general_term_proof_sum_formula_proof_l296_296470

noncomputable def arithmetic_seq_general_term (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) : Prop :=
  (S_n 4 = 4 * S_n 2) →
  (∀ n, a_n (2 * n) = 2 * a_n n + 1) →
  (∀ n, a_n n = 2 * n - 1)

noncomputable def sequence_sum_formula 
  (a_n b_n c_n T_n R_n : ℕ → ℕ) (λ : ℕ) : Prop :=
  (∀ n, T_n n + (a_n n + 1) / 2 ^ n = λ) →
  (c_n = λ n, b_n (2 * n)) →
  (∀ n, R_n n = ∑ i in finset.range n, c_n i) →
  ∀ n, R_n n = (1 / 9) * (4 - (3 * n + 1) / 4^(n - 1))

theorem general_term_proof {a_n : ℕ → ℕ} {S_n : ℕ → ℕ} : 
  arithmetic_seq_general_term a_n S_n := by 
  sorry

theorem sum_formula_proof 
  {a_n b_n c_n T_n R_n : ℕ → ℕ} {λ : ℕ} : 
  sequence_sum_formula a_n b_n c_n T_n R_n λ := by 
  sorry

end general_term_proof_sum_formula_proof_l296_296470


namespace triangle_is_isosceles_l296_296617

variables {A B C M N : Type} [EuclideanGeometry A B C M N]

theorem triangle_is_isosceles 
  (hABC : triangle A B C) 
  (hM : OnSide M A B) 
  (hN : OnSide N B C) 
  (h1 : Perimeter (triangle A M C) = Perimeter (triangle C N A))
  (h2 : Perimeter (triangle A N B) = Perimeter (triangle C M B)) :
  IsIsosceles (triangle A B C) := 
sorry

end triangle_is_isosceles_l296_296617


namespace solution_set_l296_296571

-- Define the conditions
variables {f : ℝ → ℝ}

-- Condition: f(x) is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x

-- Condition: xf'(x) + f(x) < 0 for x in (-∞, 0)
axiom condition1 : ∀ x : ℝ, x < 0 → x * (deriv f x) + f x < 0

-- Condition: f(-2) = 0
axiom f_neg2_zero : f (-2) = 0

-- Goal: Prove the solution set of the inequality xf(x) < 0 is {x | -2 < x < 0 ∨ 0 < x < 2}
theorem solution_set : ∀ x : ℝ, (x * f x < 0) ↔ (-2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2) := by
  sorry

end solution_set_l296_296571


namespace value_of_x0_l296_296519

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x
noncomputable def f_deriv (x : ℝ) : ℝ := ((x - 1) * Real.exp x) / (x * x)

theorem value_of_x0 (x0 : ℝ) (h : f_deriv x0 = -f x0) : x0 = 1 / 2 := by
  sorry

end value_of_x0_l296_296519


namespace gcd_cubic_l296_296876

theorem gcd_cubic (n : ℕ) (h1 : n > 9) :
  let k := gcd (n^3 + 25) (n + 3)
  in if (n + 3) % 2 = 1 then k = 1 else k = 2 :=
by
  sorry

end gcd_cubic_l296_296876


namespace smallest_positive_period_of_f_maximum_value_of_f_on_interval_l296_296485

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + cos (2 * x) + 3

theorem smallest_positive_period_of_f : 
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T := 
sorry

theorem maximum_value_of_f_on_interval : 
  ∃ x ∈ Set.Icc (0:ℝ) (π / 4), f x = 5 ∧ x = π / 6 :=
sorry

end smallest_positive_period_of_f_maximum_value_of_f_on_interval_l296_296485


namespace total_cost_3m3_topsoil_l296_296306

def topsoil_cost (V C : ℕ) : ℕ :=
  V * C

theorem total_cost_3m3_topsoil : topsoil_cost 3 12 = 36 :=
by
  unfold topsoil_cost
  exact rfl

end total_cost_3m3_topsoil_l296_296306


namespace person_next_to_Boris_arkady_galya_l296_296806

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l296_296806


namespace prism_cross_section_angle_volumes_ratio_l296_296891

theorem prism_cross_section_angle_volumes_ratio
  (A A' B B' C C' D E : Point)
  (Hprism : is_right_triangular_prism A A' B B' C C')
  (HDE_BB'_CC' : on_edge D B B' ∧ on_edge E C C')
  (HEC_BC : dist E C = dist B C)
  (HEC_2BD : dist E C = 2 * dist B D)
  (HAA'_2AB : dist A A' = 2 * dist A B) :
  (cross_section_angle (plane_through A D E) (base_plane A B C) = 45) ∧
  (volume_ratio (divide_prism A A' B B' C C' (plane_through A D E)) = 3) :=
sorry

end prism_cross_section_angle_volumes_ratio_l296_296891


namespace persons_next_to_Boris_l296_296824

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l296_296824


namespace integer_solutions_count_l296_296962

theorem integer_solutions_count :
  {x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.to_finset.card = 5 :=
by sorry

end integer_solutions_count_l296_296962


namespace isosceles_triangle_l296_296602

variables {A B C M N : Type*}

def is_triangle (A B C : Type*) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (X Y Z : Type*) : ℝ := -- Dummy function to represent perimeter

theorem isosceles_triangle
  {A B C M N : Type*}
  (hABC : is_triangle A B C)
  (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (hM_on_AB : M ∈ [A, B])
  (hN_on_BC : N ∈ [B, C])
  (h_perim_AMC_CNA : perimeter A M C = perimeter C N A)
  (h_perim_ANB_CMB : perimeter A N B = perimeter C M B) :
  (A = B) ∨ (B = C) ∨ (C = A) :=
by sorry

end isosceles_triangle_l296_296602


namespace puzzle_pieces_count_l296_296307

variable (border_pieces : ℕ) (trevor_pieces : ℕ) (joe_pieces : ℕ) (missing_pieces : ℕ)

def total_puzzle_pieces (border_pieces trevor_pieces joe_pieces missing_pieces : ℕ) : ℕ :=
  border_pieces + trevor_pieces + joe_pieces + missing_pieces

theorem puzzle_pieces_count :
  border_pieces = 75 → 
  trevor_pieces = 105 → 
  joe_pieces = 3 * trevor_pieces → 
  missing_pieces = 5 → 
  total_puzzle_pieces border_pieces trevor_pieces joe_pieces missing_pieces = 500 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  -- proof step to get total_number_pieces = 75 + 105 + (3 * 105) + 5
  -- hence total_puzzle_pieces = 500
  sorry

end puzzle_pieces_count_l296_296307


namespace range_of_a_for_extreme_points_l296_296139

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * Real.exp (2 * x)

theorem range_of_a_for_extreme_points :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    ∀ a : ℝ, 0 < a ∧ a < (1 / 2) →
    (Real.exp x₁ * (x₁ + 1 - 2 * a * Real.exp x₁) = 0) ∧ 
    (Real.exp x₂ * (x₂ + 1 - 2 * a * Real.exp x₂) = 0)) ↔ 
  ∀ a : ℝ, 0 < a ∧ a < (1 / 2) :=
sorry

end range_of_a_for_extreme_points_l296_296139


namespace max_elements_no_square_product_l296_296728

theorem max_elements_no_square_product (M : Finset ℕ) (hM : ∀ (a b c : ℕ), a ∈ M → b ∈ M → c ∈ M → a ≠ b → b ≠ c → a ≠ c → ¬ is_square (a * b * c)) : M.card ≤ 10 :=
sorry

end max_elements_no_square_product_l296_296728


namespace f_zero_f_sequence_geometric_f_monotonic_l296_296754

def f : ℝ → ℝ := sorry

axiom f_prop1 : f(1) = 1
axiom f_prop2 : ∀ x y : ℝ, f(x + y) = f(x) + f(y) + f(x) * f(y)
axiom f_prop3 : ∀ x : ℝ, x > 0 → f(x) > 0

theorem f_zero : f(0) = 0 := sorry

theorem f_sequence_geometric (n : ℕ) (h : n > 0) : 
  ∃ r, (∀ k : ℕ, f(k + 1) + 1 = r * (f(k) + 1)) ∧ r = 2 := sorry

theorem f_monotonic : ∀ x y : ℝ, 0 < x ∧ x < y → f(x) < f(y) := sorry

end f_zero_f_sequence_geometric_f_monotonic_l296_296754


namespace ice_forms_inner_surface_in_winter_l296_296303

-- Definitions based on conditions
variable (humid_air_inside : Prop) 
variable (heat_transfer_inner_surface : Prop) 
variable (heat_transfer_outer_surface : Prop) 
variable (temp_inner_surface_below_freezing : Prop) 
variable (condensation_inner_surface_below_freezing : Prop)
variable (ice_formation_inner_surface : Prop)
variable (cold_dry_air_outside : Prop)
variable (no_significant_condensation_outside : Prop)

-- Proof of the theorem
theorem ice_forms_inner_surface_in_winter :
  humid_air_inside ∧
  heat_transfer_inner_surface ∧
  heat_transfer_outer_surface ∧
  (¬sufficient_heating → temp_inner_surface_below_freezing) ∧
  (condensation_inner_surface_below_freezing ↔ (temp_inner_surface_below_freezing ∧ humid_air_inside)) ∧
  (ice_formation_inner_surface ↔ (condensation_inner_surface_below_freezing ∧ temp_inner_surface_below_freezing)) ∧
  (cold_dry_air_outside → ¬ice_formation_outer_surface)
  → ice_formation_inner_surface :=
sorry

end ice_forms_inner_surface_in_winter_l296_296303


namespace who_is_next_to_boris_l296_296808

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l296_296808


namespace series_sum_l296_296343

theorem series_sum : ∑ i in (Finset.range 5000).filter (fun x => x % 2 = 0), (x + 1) - x = -2500 := 
  by sorry

end series_sum_l296_296343


namespace angle_B_in_triangle_ABC_l296_296108

theorem angle_B_in_triangle_ABC 
  (A B C D E : Type)
  [triangle A B C]
  (on_side_AC : D ∈ AC ∧ E ∈ AC)
  (equilateral_BDE : is_equilateral_triangle B D E)
  (isosceles_ABD : is_isosceles_triangle A B D)
  (isosceles_BEC : is_isosceles_triangle B E C)
  (angle_sum : ∑ (A B C : ℝ), A + B + C = 180) :
  ∃ Bval : ℝ, Bval = 105 ∨ Bval = 120 :=
begin
  sorry
end

end angle_B_in_triangle_ABC_l296_296108


namespace largest_whole_number_lt_150_l296_296279

theorem largest_whole_number_lt_150 : ∃ (x : ℕ), (x <= 16 ∧ ∀ y : ℕ, y < 17 → 9 * y < 150) :=
by
  sorry

end largest_whole_number_lt_150_l296_296279


namespace cristina_catches_up_to_nicky_l296_296339

theorem cristina_catches_up_to_nicky (t : ℕ) :
  (∀ (d_cristina d_nicky : ℕ), d_cristina = 5 * t ∧ d_nicky = 30 + 3 * t) →
  (∃ t = 15, 5 * t = 30 + 3 * t) :=
by
  intros d_cristina d_nicky
  have h1 : d_cristina = 5 * t := by sorry 
  have h2 : d_nicky = 30 + 3 * t := by sorry
  existsi 15
  simp at *

end cristina_catches_up_to_nicky_l296_296339


namespace john_lily_meet_at_midpoint_l296_296194

-- Define John's and Lily's coordinates
def john_coord : (ℝ × ℝ) := (1, 4)
def lily_coord : (ℝ × ℝ) := (5, -2)

-- Define the midpoint function
def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the theorem to prove that the midpoint of John's and Lily's coordinates is (3, 1)
theorem john_lily_meet_at_midpoint : midpoint john_coord lily_coord = (3, 1) := 
by
  sorry

end john_lily_meet_at_midpoint_l296_296194


namespace who_is_next_to_boris_l296_296813

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l296_296813


namespace symmetric_line_eq_l296_296272

-- Definitions for the given line equations
def l1 (x y : ℝ) : Prop := 3 * x - y - 3 = 0
def l2 (x y : ℝ) : Prop := x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x - 3 * y - 1 = 0

-- The theorem to prove
theorem symmetric_line_eq (x y : ℝ) (h1: l1 x y) (h2: l2 x y) : l3 x y :=
sorry

end symmetric_line_eq_l296_296272


namespace find_value_l296_296513

theorem find_value (x y z : ℝ) (h₁ : y = 3 * x) (h₂ : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end find_value_l296_296513


namespace exists_point_on_circle_l296_296113

noncomputable theory

open scoped Classical

theorem exists_point_on_circle 
  (circle : Type) [metric_space circle]
  (A B C D E : circle) (h : Metric.dist A B = Metric.dist B C = Metric.dist C D = Metric.dist D E = Metric.dist E A) :
  ∃ F : circle, 
    (Metric.dist A F = Metric.dist F B) = 
    (Metric.dist B F = Metric.dist F C) = 
    (Metric.dist C F = Metric.dist F D) = 
    (Metric.dist D F = Metric.dist F E) = 
    (Metric.dist E F = Metric.dist F A) :=
begin
  sorry
end

end exists_point_on_circle_l296_296113


namespace fixed_point_l296_296914

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x-1)

theorem fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : ∃ P : ℝ × ℝ, P = (1, 5) ∧ ∀ x, f(a)(x) = P.2 := 
sorry

end fixed_point_l296_296914


namespace max_value_expression_l296_296933

theorem max_value_expression (a x1 x2 : ℝ) (h1 : a < 0) 
(h2 : x1 + x2 = 4 * a) (h3 : x1 * x2 = 3 * a^2)
: x1 + x2 + a / (x1 * x2) ≤ -4 * Real.sqrt(3) / 3 := 
sorry

end max_value_expression_l296_296933


namespace fraction_left_handed_l296_296385

def total_participants (k : ℕ) := 15 * k

def red (k : ℕ) := 5 * k
def blue (k : ℕ) := 5 * k
def green (k : ℕ) := 3 * k
def yellow (k : ℕ) := 2 * k

def left_handed_red (k : ℕ) := (1 / 3) * red k
def left_handed_blue (k : ℕ) := (2 / 3) * blue k
def left_handed_green (k : ℕ) := (1 / 2) * green k
def left_handed_yellow (k : ℕ) := (1 / 4) * yellow k

def total_left_handed (k : ℕ) := left_handed_red k + left_handed_blue k + left_handed_green k + left_handed_yellow k

theorem fraction_left_handed (k : ℕ) : 
  (total_left_handed k) / (total_participants k) = 7 / 15 := 
sorry

end fraction_left_handed_l296_296385


namespace vector_MN_correct_l296_296551

variables (O A B C M N : Point)
variables (a b c : Vector)

-- Given definitions
def vec_OA : Vector := a
def vec_OB : Vector := b
def vec_OC : Vector := c
def vec_OM : Vector := (2 / 3 : ℝ) • a
def BN_eq_NC (B N C : Point) : Prop := (B - N) = (N - C)

-- Given question
def MN (a b c : Vector) : Vector := (-(2 : ℝ)/3) • a + (1 / 2 : ℝ) • b + (1 / 2 : ℝ) • c

-- The theorem to prove
theorem vector_MN_correct :
  BN_eq_NC B N C → 
  vec_OM = (2 / 3 : ℝ) • a →
  MN a b c = (-(2 : ℝ)/3) • a + (1 / 2 : ℝ) • b + (1 / 2 : ℝ) • c :=
by
  sorry

end vector_MN_correct_l296_296551


namespace tangent_area_at_x_eq_1_l296_296137

noncomputable def f (x : ℝ) : ℝ := (Real.log x) + x

def tangent_eq (x y : ℝ) : Prop :=
  y = 2 * x - 1

theorem tangent_area_at_x_eq_1 : 
  let P := (1 : ℝ, f 1)
  let slope := 2
  let y_intercept := -1
  let x_intercept := (1/2)
  area := (1/2) * abs(x_intercept*y_intercept)
in area = 1 / 4 :=
by
  sorry

end tangent_area_at_x_eq_1_l296_296137


namespace find_a_l296_296934

noncomputable theory

open Complex

def z1 : ℂ := 1 + I
def z2 (a : ℝ) : ℂ := 3 + a * I

theorem find_a (a : ℝ) (h : 3 * z1 = z2 a) : a = 3 := by
  sorry

end find_a_l296_296934


namespace hexagon_has_differently_colored_triangle_l296_296412

noncomputable def hexagon_triangle_coloring_probability : ℚ :=
  let probability_diff_colors := 2 / 9
  let total_triangles := 20
  1 - (7/9)^total_triangles

theorem hexagon_has_differently_colored_triangle :
  hexagon_triangle_coloring_probability = 229 / 256 :=
begin
  sorry
end

end hexagon_has_differently_colored_triangle_l296_296412


namespace problem1_l296_296726

theorem problem1 (a b : ℝ) (h1 : a < 0) (h2 : b > 0) (h3 : complex.abs (complex.mk a b) = 2) (h4 : complex.re (complex.mk a b) + complex.re (complex.conj (complex.mk a b)) = -2) :
  complex.mk a b = complex.mk (-1) (real.sqrt 3) := by
  sorry

end problem1_l296_296726


namespace passes_through_point_l296_296650

-- Define the function and conditions.
def exponential_function (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 2

variables (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)

theorem passes_through_point : exponential_function a 1 = 3 :=
by {
  sorry
}

end passes_through_point_l296_296650


namespace lambda_value_Tn_sum_l296_296109

-- Define the arithmetic sequence and its properties
variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (λ : ℕ)

axiom a1_pos : λ > 0
axiom a1_def : a 1 = λ
axiom Sn_def : ∀ n, S n = ∑ i in finset.range n, a i
axiom anplus1_def : ∀ n, a (n + 1) = 2 * int.sqrt (S n) + 1

-- Prove the value of λ
theorem lambda_value : λ = 1 :=
sorry

-- Define the sequence {1/(a_na_{n+1})}
variable (T : ℕ → ℚ)

axiom Tn_def : ∀ n, T n = ∑ i in finset.range n, 1 / (a i * a (i + 1))

-- Prove the sum of the first n terms of {1/(a_na_{n+1})}
theorem Tn_sum : ∀ n, T n = 1/6 - 1/(4*n + 6) :=
sorry

end lambda_value_Tn_sum_l296_296109


namespace rational_square_l296_296243

theorem rational_square (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) : ∃ r : ℚ, (1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2) = r^2 := 
by 
  sorry

end rational_square_l296_296243


namespace length_of_first_train_l296_296027

theorem length_of_first_train
    (speed_first_train_kmph : ℝ) 
    (speed_second_train_kmph : ℝ) 
    (time_to_cross_seconds : ℝ) 
    (length_second_train_meters : ℝ)
    (H1 : speed_first_train_kmph = 120)
    (H2 : speed_second_train_kmph = 80)
    (H3 : time_to_cross_seconds = 9)
    (H4 : length_second_train_meters = 300.04) : 
    ∃ (length_first_train : ℝ), length_first_train = 200 :=
by 
    let relative_speed_m_per_s := (speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600
    let combined_length := relative_speed_m_per_s * time_to_cross_seconds
    let length_first_train := combined_length - length_second_train_meters
    use length_first_train
    sorry

end length_of_first_train_l296_296027


namespace domain_sqrt_log_half_l296_296645

def domain_of_function : Set ℝ :=
  {x : ℝ | x > 0 ∧ Real.log 3 x ≤ 0}

theorem domain_sqrt_log_half : domain_of_function = {x : ℝ | 0 < x ∧ x ≤ 1} :=
sorry

end domain_sqrt_log_half_l296_296645


namespace Rodney_lift_amount_l296_296246

variable (R Ron : ℝ)

def Roger_lifts (R : ℝ) := R
def Ron_lifts (Ron : ℝ) := Ron
def Rodney_lifts (R : ℝ) := 2 * R
def Rebecca_lifts (Ron : ℝ) := 3 * Ron - 20

axiom total_lift : Roger_lifts R + Ron_lifts Ron + Rodney_lifts R + Rebecca_lifts Ron = 375

# First, express the relationship between R and Ron based on the problem:
axiom Roger_over_Ron : R = Ron + 5
  
theorem Rodney_lift_amount (R Ron : ℝ) (h1 : total_lift R Ron) (h2 : Roger_over_Ron R Ron) :
  Rodney_lifts R = 118 :=
by
  sorry

end Rodney_lift_amount_l296_296246


namespace problem_solution_l296_296472

-- Define the points in space.
def A : ℝ × ℝ × ℝ := (1, 1, 0)
def B : ℝ × ℝ × ℝ := (0, 1, 2)
def C : ℝ × ℝ × ℝ := (0, 3, 2)
def D : ℝ × ℝ × ℝ := (-1, 3, 4)

-- Define the vectors between the points.
def vec (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (q.1 - p.1, q.2 - p.2, q.3 - p.3)

def AB : ℝ × ℝ × ℝ := vec A B
def BC : ℝ × ℝ × ℝ := vec B C
def CD : ℝ × ℝ × ℝ := vec C D

-- Define the dot product of two vectors.
def dot (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define parallel vectors.
def parallel (u v : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2, k * v.3)

-- Lean theorem statement.
theorem problem_solution :
  dot AB BC = 0 ∧    -- Statement A: AB ⊥ BC
  parallel AB CD ∧   -- Statement B: AB ∥ CD
  ¬collinear A B C ∧ -- Statement C: A, B, and C are collinear (incorrect, hence negated)
  coplanar A B C D   -- Statement D: A, B, C, and D are coplanar
:= sorry

end problem_solution_l296_296472


namespace range_of_eccentricity_l296_296202

variables (a b : ℝ)
def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)

-- For the upper vertex B
def upper_vertex : ℝ × ℝ := (0, b)

-- Distance condition
def distance_condition (P : ℝ × ℝ) : Prop := 
  let (x, y) := P in 
  (x - 0)^2 + (y - b)^2 ≤ (2 * b)^2

-- Eccentricity
def eccentricity : ℝ := real.sqrt (a^2 - b^2) / a

theorem range_of_eccentricity 
  (h_ellipse : ∀ (x y : ℝ), ellipse_eq a b x y → distance_condition a b (x, y))
  : 0 < eccentricity a b ∧ eccentricity a b ≤ real.sqrt 2 / 2
:= sorry

end range_of_eccentricity_l296_296202


namespace lines_parallel_l296_296483

theorem lines_parallel (a : ℝ) 
  (h₁ : (∀ x y : ℝ, ax + (a + 2) * y + 2 = 0)) 
  (h₂ : (∀ x y : ℝ, x + a * y + 1 = 0)) 
  : a = -1 :=
sorry

end lines_parallel_l296_296483


namespace monkey_distance_l296_296350

-- Define the initial speeds and percentage adjustments
def swing_speed : ℝ := 10
def run_speed : ℝ := 15
def wind_resistance_percentage : ℝ := 0.10
def branch_assistance_percentage : ℝ := 0.05

-- Conditions
def adjusted_swing_speed : ℝ := swing_speed * (1 - wind_resistance_percentage)
def adjusted_run_speed : ℝ := run_speed * (1 + branch_assistance_percentage)
def run_time : ℝ := 5
def swing_time : ℝ := 10

-- Define the distance formulas based on the conditions
def run_distance : ℝ := adjusted_run_speed * run_time
def swing_distance : ℝ := adjusted_swing_speed * swing_time

-- Total distance calculation
def total_distance : ℝ := run_distance + swing_distance

-- Statement for the proof
theorem monkey_distance : total_distance = 168.75 := by
  sorry

end monkey_distance_l296_296350


namespace triangle_sine_half_angle_ineq_l296_296529

theorem triangle_sine_half_angle_ineq
  {A B C : ℝ} 
  (h1 : A + B + C = π)
  (h2 : A > 0) (h3 : B > 0) (h4 : C > 0) :
  3 / 4 ≤ sin (A / 2) ^ 2 + sin (B / 2) ^ 2 + sin (C / 2) ^ 2 ∧
    sin (A / 2) ^ 2 + sin (B / 2) ^ 2 + sin (C / 2) ^ 2 < 1 :=
by
  sorry

end triangle_sine_half_angle_ineq_l296_296529


namespace integral_sqrt_plus_square_l296_296049

open Real

theorem integral_sqrt_plus_square : 
  ∫ x in 0..2, (sqrt (4 - x^2) + x^2) = π + (8 / 3) :=
by
  -- conditions here
  have h1 : ∫ x in 0..2, sqrt (4 - x^2) = π / 2,
    -- Proof or sorry
  sorry,
  
  have h2 : ∫ x in 0..2, x^2 = (8 / 3) := 
    -- Proof or sorry
  sorry,
  
  -- combine the results from h1 and h2 to finish the proof
  sorry

end integral_sqrt_plus_square_l296_296049


namespace triangle_isosceles_l296_296612

theorem triangle_isosceles
  {A B C M N : Point}
  (h_M_on_AB : ∃ t ∈ Set.Icc (0 : ℝ) 1, M = t • A + (1 - t) • B)
  (h_N_on_BC : ∃ t ∈ Set.Icc (0 : ℝ) 1, N = t • B + (1 - t) • C)
  (h_perimeter_AMC_CNA : dist A M + dist M C + dist C A = dist C N + dist N A + dist A C)
  (h_perimeter_ANB_CMB : dist A N + dist N B + dist B A = dist C M + dist M B + dist B C)
  : isosceles_triangle A B C := 
sorry

end triangle_isosceles_l296_296612


namespace distinct_positive_and_conditions_l296_296718

theorem distinct_positive_and_conditions (a b : ℕ) (h_distinct: a ≠ b) (h_pos1: 0 < a) (h_pos2: 0 < b) (h_eq: a^3 - b^3 = a^2 - b^2) : 
  ∃ (c : ℕ), c = 9 * a * b ∧ (c = 1 ∨ c = 2 ∨ c = 3) :=
by
  sorry

end distinct_positive_and_conditions_l296_296718


namespace solve_for_y_l296_296404

def circ (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

theorem solve_for_y : ∃ y : ℝ, circ 2 y = 10 ∧ y = 0 :=
by
  existsi 0
  split
  all_goals
  {
    simp [circ],
    linarith
  }

end solve_for_y_l296_296404


namespace min_value_of_sum_l296_296145

variable {m n : ℝ}
variable h1 : m > 0
variable h2 : n > 0
variable h_parallel : (∀ k : ℝ, k > 0 → (2 * m = 4 - n) ∧ (2 * k * m - k * n = 4))

theorem min_value_of_sum (m_gt0 : m > 0) (n_gt0 : n > 0) (a_parallel_b : (∀ k : ℝ, k > 0 → (2 * m = 4 - n) ∧ (2 * k * m - k * n = 4))) : 
  ∃ m n, (m > 0 ∧ n > 0 ∧ 2 * m + n = 4) ∧ (∀ a b, \frac{1}{a} + \frac{8}{b} ≥ \frac{9}{2}) :=
  sorry

end min_value_of_sum_l296_296145


namespace simple_interest_years_l296_296024

variable (P R T : ℕ)
variable (deltaI : ℕ := 400)
variable (P_value : P = 800)

theorem simple_interest_years 
  (h : (800 * (R + 5) * T / 100) = (800 * R * T / 100) + 400) :
  T = 10 :=
by sorry

end simple_interest_years_l296_296024


namespace central_park_trash_cans_more_than_half_l296_296392

theorem central_park_trash_cans_more_than_half
  (C : ℕ)  -- Original number of trash cans in Central Park
  (V : ℕ := 24)  -- Original number of trash cans in Veteran's Park
  (V_now : ℕ := 34)  -- Number of trash cans in Veteran's Park after the move
  (H_move : (V_now - V) = C / 2)  -- Condition of trash cans moved
  (H_C : C = (1 / 2) * V + x)  -- Central Park had more than half trash cans as Veteran's Park, where x is an excess amount
  : C - (1 / 2) * V = 8 := 
sorry

end central_park_trash_cans_more_than_half_l296_296392


namespace triangle_is_isosceles_l296_296594

variables {A B C M N : Type} [pseudo_metric_space A] [pseudo_metric_space B] [pseudo_metric_space C]
[pseudo_metric_space M] [pseudo_metric_space N] 

variables {dist : ∀ {X : Type} [pseudo_metric_space X], X → X → ℝ}
variables {a b c x y : ℝ} -- edge lengths

-- Define the points and their distances
variables {MA MB NA NC : ℝ} (tABC : triangle A B C) 

-- Conditions from the problem
def condition1 : Prop :=
  dist A M + dist M C + dist C A = dist C N + dist N A + dist A C

def condition2 : Prop :=
  dist A N + dist N B + dist B A = dist C M + dist M B + dist B C

-- Proving the triangle is isosceles
theorem triangle_is_isosceles (tABC : triangle A B C) 
    (h1 : condition1)
    (h2 : condition2) : isosceles tABC :=
sorry

end triangle_is_isosceles_l296_296594
