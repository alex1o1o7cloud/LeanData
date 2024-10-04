import Mathlib
import Mathlib.Algebra.Arithmetic.Basic
import Mathlib.Algebra.CharZero.Quotient
import Mathlib.Algebra.Divisibility
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Geometric.Basic
import Mathlib.Algebra.Order
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Trigonometry
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Finset
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Combinatorics.SimpleGraph.Matching
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Probability.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.MeasureTheory.Function.LpSpace
import Mathlib.MeasureTheory.Integral.SetIntegral
import Mathlib.Probability.ConditionalProbability
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace B_is_quadratic_l382_382105

-- Definitions for the conditions
def equation_A (x : ℝ) : Prop := x^2 + 1/x = 0
def equation_B (x : ℝ) : Prop := x^2 + 6x = 0
def equation_C (a x : ℝ) : Prop := a^2 * x - 5 = 0
def equation_D (x : ℝ) : Prop := 4 * x - x^3 = 2

-- Proof statement: equation_B matches the standard form of a quadratic equation.
theorem B_is_quadratic : equation_B x_iff_exists_coefficients :
  ∃ (a b c : ℝ), a ≠ 0 ∧ equation_B x = a * x * x + b * x + c :=
sorry

end B_is_quadratic_l382_382105


namespace solution_set_inequalities_l382_382826

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382826


namespace prime_squares_5000_9000_l382_382651

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end prime_squares_5000_9000_l382_382651


namespace angle_A_measure_sin_B_plus_sin_C_value_l382_382283

-- Provided definitions and conditions
variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}
variables (triangle_ABC : Triangle) 

-- Given conditions
axiom sides_of_triangle : (triangle_ABC.sides) = (a, b, c)
axiom angle_relationship : 2 * a * Real.cos B + b = 2 * c
axiom side_a_value : a = 2 * Real.sqrt 3
axiom area_value : S = Real.sqrt 3

-- Required proof: Measure of angle A
theorem angle_A_measure : A = Real.pi / 3 := sorry

-- Required proof: Value of sin B + sin C
theorem sin_B_plus_sin_C_value : Real.sin B + Real.sin C = Real.sqrt 6 / 2 := sorry

end angle_A_measure_sin_B_plus_sin_C_value_l382_382283


namespace solution_set_linear_inequalities_l382_382914

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382914


namespace sum_d_e_f_l382_382155

noncomputable def chain_length: ℝ := 25 - Real.sqrt 134

theorem sum_d_e_f
  (h_chain_length : (25 - Real.sqrt 134) / 1 = chain_length)
  (d : ℕ := 25)
  (e : ℕ := 134)
  (f : ℕ := 1)
  (h_prime_f : Nat.Prime f)
  : d + e + f = 160 := by
begin
  sorry
end

end sum_d_e_f_l382_382155


namespace minimum_number_of_guests_l382_382052

theorem minimum_number_of_guests :
  ∀ (total_food : ℝ) (max_food_per_guest : ℝ), total_food = 411 → max_food_per_guest = 2.5 →
  ⌈total_food / max_food_per_guest⌉ = 165 :=
by
  intros total_food max_food_per_guest h1 h2
  rw [h1, h2]
  norm_num
  sorry

end minimum_number_of_guests_l382_382052


namespace time_to_hit_ground_l382_382798

theorem time_to_hit_ground : ∃ t : ℝ, 
  (y = -4.9 * t^2 + 7.2 * t + 8) → (y - (-0.6 * t) * t = 0) → t = 223/110 :=
by
  sorry

end time_to_hit_ground_l382_382798


namespace white_dandelions_on_saturday_l382_382141

theorem white_dandelions_on_saturday 
  (yellow_monday : ℕ) (white_monday : ℕ)
  (yellow_wednesday : ℕ) (white_wednesday : ℕ)
  (life_cycle : ∀ d, (d.bloom == "yellow" ∧ d.age == 3) ∨ (d.bloom == "white" ∧ d.age == 4) ∨ (d.bloom == "dispersed" ∧ d.age == 5))
  (total_monday : yellow_monday = 20 ∧ white_monday = 14)
  (total_wednesday : yellow_wednesday = 15 ∧ white_wednesday = 11):
  let new_dandelions := 26 - 20,
      white_dandelions_saturday := new_dandelions
  in white_dandelions_saturday = 6 := by
  let dandelions_tuesday_wednesday := new_dandelions
  have h : white_dandelions_saturday = 6,
  from sorry
  exact h

end white_dandelions_on_saturday_l382_382141


namespace solution_set_system_of_inequalities_l382_382942

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382942


namespace range_of_m_l382_382398

noncomputable def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop :=
  m > 1

noncomputable def mx2_mx_plus_1_always_pos (m : ℝ) : Prop :=
  ∀ x : ℝ, m * x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) :
  (¬ (is_ellipse_with_foci_on_y_axis m ∧ mx2_mx_plus_1_always_pos m)) ∧
  (is_ellipse_with_foci_on_y_axis m ∨ mx2_mx_plus_1_always_pos m) →
  m ∈ set.Icc 0 1 ∪ set.Ici 4 :=
begin
  sorry
end

end range_of_m_l382_382398


namespace units_digit_120_factorial_l382_382099

theorem units_digit_120_factorial : 
  let n := 120 in ((n.factorial % 10) = 0) := 
by
  let n := 120
  have h : ∃ k, n.factorial = 10 * k := sorry
  exact h

end units_digit_120_factorial_l382_382099


namespace range_of_a_l382_382294

noncomputable def f (x a : ℝ) : ℝ := 
  x * (a - 1 / Real.exp x)

noncomputable def gx (x : ℝ) : ℝ :=
  (1 + x) / Real.exp x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 a = 0 ∧ f x2 a = 0) →
  a < 2 / Real.exp 1 :=
by
  sorry

end range_of_a_l382_382294


namespace unique_poly_with_roots_condition_l382_382639

def poly6 (a b c d e : ℝ) := λ x : ℂ, x^6 + (a : ℂ) * x^5 + (b : ℂ) * x^4 + (c : ℂ) * x^3 + (d : ℂ) * x^2 + (e : ℂ) * x + 2024

theorem unique_poly_with_roots_condition :
  ∃ (a b c d e : ℝ), 
  (∀ r : ℂ, 
    (poly6 a b c d e r = 0) → 
    (poly6 a b c d e ((-1 + real.sqrt 3 * complex.I) / 2 * r) = 0) ∧ 
    (poly6 a b c d e (-r) = 0)) ↔ 
  ∃! (a b c d e : ℝ), (poly6 a b c d e = λ x, (x - r) * (x - ((-1 + real.sqrt 3 * complex.I) / 2 * r)) * 
    (x - ((-1 - real.sqrt 3 * complex.I) / 2 * r)) * (x + r) * 
    (x + ((-1 + real.sqrt 3 * complex.I) / 2 * r)) * (x + ((-1 - real.sqrt 3 * complex.I) / 2 * r))) := 
by
  sorry

end unique_poly_with_roots_condition_l382_382639


namespace solution_set_linear_inequalities_l382_382913

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382913


namespace geometry_problem_l382_382546

open Real

variables {A B C P Q L : Point} (k : Circle)

-- Definitions of the points and circle
def passes_through (c : Circle) (p : Point) : Prop := OnCircle c p
def intersects_sides (c : Circle) (P Q L : Point) (Δ : Triangle) : Prop :=
  IntersectsSide Δ c P ∧ IntersectsSide Δ c Q ∧ IntersectsSide Δ c L

-- Problem Conditions
axiom circle_conditions : passes_through k A ∧ intersects_sides k P Q L ⟨A, B, C⟩

-- Areas of triangles and distance metrics
def area_ABC : Real := Triangle.area ⟨A, B, C⟩
def area_PQL : Real := Triangle.area ⟨P, Q, L⟩

-- Variables for the sides
def distance_PL : Real := Distance P L
def distance_AQ : Real := Distance A Q

-- Problem Statement
theorem geometry_problem : 
  (area_PQL / area_ABC) ≤ (1 / 4) * (distance_PL / distance_AQ)^2 := 
sorry

end geometry_problem_l382_382546


namespace count_primes_squared_in_range_l382_382656

theorem count_primes_squared_in_range : 
  (finset.card (finset.filter (λ p, p^2 ≥ 5000 ∧ p^2 ≤ 9000) 
    (finset.filter nat.prime (finset.range 95)))) = 5 := 
by sorry

end count_primes_squared_in_range_l382_382656


namespace empty_boxes_a_cannot_empty_boxes_b_l382_382450

-- Define the two operations on the boxes
inductive Step
| remove_one : Step
| triple : Step

-- Function to perform a step on the pair of pebbles (p, q)
def perform_step : Step → ℕ × ℕ → ℕ × ℕ
| Step.remove_one (p, q) := (p - 1, q - 1)
| Step.triple (p, q) := (3 * p, q)

-- Part (a): Prove it is possible to empty the boxes for p = 100 and q = 200
theorem empty_boxes_a : ∃ steps : List Step, 
  steps.foldl (fun (pq : ℕ × ℕ) (s : Step) => perform_step s pq) (100, 200) = (0, 0) := sorry

-- Part (b): Prove it is not possible to empty the boxes for p = 101 and q = 200
theorem cannot_empty_boxes_b : ¬ ∃ steps : List Step, 
  steps.foldl (fun (pq : ℕ × ℕ) (s : Step) => perform_step s pq) (101, 200) = (0, 0) := sorry

end empty_boxes_a_cannot_empty_boxes_b_l382_382450


namespace mode_of_sample_data_is_six_l382_382057

def sample_data : List ℤ := [-2, 0, 6, 3, 6]

theorem mode_of_sample_data_is_six : (mode sample_data) = 6 := by
  sorry

end mode_of_sample_data_is_six_l382_382057


namespace p_necessary_not_sufficient_for_q_l382_382272

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def p (a : ℝ) : Prop :=
  collinear (vec a (a^2)) (vec 1 2)

def q (a : ℝ) : Prop := a = 2

theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ ¬(∀ a : ℝ, p a → q a) :=
sorry

end p_necessary_not_sufficient_for_q_l382_382272


namespace all_numbers_in_diagonal_l382_382342

theorem all_numbers_in_diagonal :
  ∀ (a : Fin 15 → Fin 15 → ℕ),
  (∀ i j, 1 ≤ a i j ∧ a i j ≤ 15) ∧ 
  (∀ i j, a i j = a j i) ∧
  (∀ i, Multiset.card (Finset.image (λ j, a i j) Finset.univ) = 15) ∧ 
  (∀ j, Multiset.card (Finset.image (λ i, a i j) Finset.univ) = 15) →
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 15 → 
  ∃ i : Fin 15, a i i = k :=
by
  intros a h_cond k k_range
  sorry

end all_numbers_in_diagonal_l382_382342


namespace continuous_function_exponential_form_l382_382571

variables {f : ℝ → ℝ}

theorem continuous_function_exponential_form :
  continuous f ∧ (∀ x y : ℝ, f(x + y) = f(x) + f(y) + f(x) * f(y)) →
  ∃ (a : ℝ), ∀ x : ℝ, f(x) = a^x - 1 := by
  sorry

end continuous_function_exponential_form_l382_382571


namespace sufficient_not_necessary_l382_382119

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (∃ x, 1 / x < 1 ∧ ¬(x > 1)) :=
by
  sorry

end sufficient_not_necessary_l382_382119


namespace triangle_property_l382_382610

-- Define the triangle ABC and points Q, R, and P such that the given conditions hold
def triangle (A B C : Type) : Prop :=
  ∃ (Q R P : Type),
    -- Conditions for ∠RAB = 15° and ∠RBA = 15°
    (angle_between P B C = 45° ∧ angle_between P C B = 30° ∧
    (angle_between Q A C = 45° ∧ angle_between Q C A = 30° ∧
    angle_between R A B = 15° ∧ angle_between R B A = 15°))

theorem triangle_property (A B C : Type) : triangle A B C → ∃ (Q R P : Type),
  angle_between Q R P = 90° ∧ dist Q R = dist R P :=
by
  sorry

end triangle_property_l382_382610


namespace minimum_moves_to_form_word_l382_382787

-- Definitions
def initial_positions := [1, 2, 3, 4, 5, 6, 7, 8] -- denote initial positions
def target_word := "ФЛАНДРИЯ"
def target_position := 1 -- top position where Ф must be

-- Proof problem
theorem minimum_moves_to_form_word (initial_positions : list ℕ) 
  (Φ_position : ℕ) (target_word : string) (target_position : ℕ) : 
  Φ_position = 5 → target_position = 1 → target_word = "ФЛАНДРИЯ" → 
  ∃ moves : ℕ, moves = 28 :=
by
  sorry

end minimum_moves_to_form_word_l382_382787


namespace mathematically_equivalent_proof_problem_l382_382275

open Set

-- Define the universal set R as real numbers ℝ for convenience
def R := ℝ

-- Define sets A and B with given conditions
def A := {x : ℝ | log (2) (3 - x) ≤ 2}
def B := {x : ℝ | x^2 ≤ 5 * x - 6}

-- Define intersection and complement
def intersection := A ∩ B
def complement := {x : ℝ | x ∉ (A ∩ B)}

-- Define the theorem to be proved
theorem mathematically_equivalent_proof_problem :
  A = {x : ℝ | -1 ≤ x ∧ x < 3} ∧
  B = {x : ℝ | 2 ≤ x ∧ x ≤ 3} ∧
  complement = {x : ℝ | x < 2 ∨ x ≥ 3} :=
by
  sorry

end mathematically_equivalent_proof_problem_l382_382275


namespace jake_sister_weight_ratio_l382_382675

theorem jake_sister_weight_ratio (Jake_initial_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (sister_weight : ℕ) 
(h₁ : Jake_initial_weight = 156) 
(h₂ : total_weight = 224) 
(h₃ : weight_loss = 20) 
(h₄ : total_weight = Jake_initial_weight + sister_weight) :
(Jake_initial_weight - weight_loss) / sister_weight = 2 := by
  sorry

end jake_sister_weight_ratio_l382_382675


namespace train_cross_pole_time_l382_382187

def speed_kmh := 90 -- speed of the train in km/hr
def length_m := 375 -- length of the train in meters

/-- Convert speed from km/hr to m/s -/
def convert_speed (v_kmh : ℕ) : ℕ := v_kmh * 1000 / 3600

/-- Calculate the time it takes for the train to cross the pole -/
def time_to_cross_pole (length_m : ℕ) (speed_m_s : ℕ) : ℕ := length_m / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole length_m (convert_speed speed_kmh) = 15 :=
by
  sorry

end train_cross_pole_time_l382_382187


namespace prime_square_count_l382_382640

-- Define the set of natural numbers whose squares lie between 5000 and 9000
def within_square_range (n : ℕ) : Prop := 5000 < n * n ∧ n * n < 9000

-- Define the predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the set of prime numbers whose squares lie within the range
def primes_within_range : Finset ℕ := (Finset.filter is_prime (Finset.Ico 71 95))

-- The main statement: the number of prime numbers whose squares are between 5000 and 9000
theorem prime_square_count : (primes_within_range.filter within_square_range).card = 6 :=
sorry

end prime_square_count_l382_382640


namespace xy_squared_sum_l382_382421

variable (x y : ℝ) -- assuming x and y are real numbers

theorem xy_squared_sum : 
  (x + y) ^ 2 = 49 ∧ x * y = 8 → x^2 + y^2 = 33 :=
by
  intros h
  cases h with h1 h2
  rw [pow_two, mul_comm x y] at h1
  calc
    x ^ 2 + y ^ 2
        = (x + y) ^ 2 - 2 * x * y : by sorry
    ... = 49 - 2 * 8          : by sorry
    ... = 33                 : by sorry

end xy_squared_sum_l382_382421


namespace petya_wins_prize_probability_atleast_one_wins_probability_l382_382395

/-- Petya and 9 other people each roll a fair six-sided die. 
    A player wins a prize if they roll a number that nobody else rolls more than once.-/
theorem petya_wins_prize_probability : (5 / 6) ^ 9 = 0.194 :=
sorry

/-- The probability that at least one player gets a prize in the game where Petya and
    9 others roll a fair six-sided die is 0.919. -/
theorem atleast_one_wins_probability : 1 - (1 / 6) ^ 9 = 0.919 :=
sorry

end petya_wins_prize_probability_atleast_one_wins_probability_l382_382395


namespace b_alone_completion_days_l382_382111

theorem b_alone_completion_days (Rab : ℝ) (w_12_days : (1 / (Rab + 4 * Rab)) = 12⁻¹) : 
    (1 / Rab = 60) :=
sorry

end b_alone_completion_days_l382_382111


namespace solution_set_linear_inequalities_l382_382923

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382923


namespace a_div_c_eq_neg4_l382_382810

-- Given conditions
def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)
def g_inv (x : ℝ) : ℝ := (4 * x + 2) / (3 - x)

-- Theorem to be proved
theorem a_div_c_eq_neg4 : let a := 4, c := -1 in a / c = -4 :=
by {
  exact rfl
}

end a_div_c_eq_neg4_l382_382810


namespace solution_y_added_is_correct_l382_382780
noncomputable theory

def liquid_x_in_original_solution (solution_y : ℝ) : ℝ := 0.30 * solution_y

def evaporated_solution (solution_y original_water evaporated_water remaining_solution : ℝ) : Prop :=
  original_water = 0.70 * solution_y ∧
  evaporated_water = 2 ∧
  remaining_solution = solution_y - evaporated_water

def final_solution (added_solution_y : ℝ) : Prop :=
  4 + added_solution_y = 4 + 2 ∧ 
  1.8 + 0.30 * added_solution_y = 0.40 * (4 + added_solution_y)

def math_proof_problem : Prop :=
  ∃ (added_solution_y : ℝ), evaporated_solution 6 4 2 4 ∧ final_solution added_solution_y ∧ added_solution_y = 2

theorem solution_y_added_is_correct : math_proof_problem :=
sorry

end solution_y_added_is_correct_l382_382780


namespace area_of_quadrilateral_l382_382404

variables {A B C D E : Type}
variables [field A] [field B] [field C] [field D] [field E]
variables (AC CD AE : ℝ) (angleABC angleACD: ℝ)

-- Conditions
def quadrilateral_conditions : Prop :=
  angleABC = 90 ∧ angleACD = 90 ∧ AC = 24 ∧ CD = 18 ∧ AE = 6

-- Question: Determine the area of quadrilateral ABCD
noncomputable def area_quadrilateral (AC CD AE : ℝ) : ℝ :=
  376 -- This is the answer provided by the solution

-- Proof Problem:
theorem area_of_quadrilateral
  (h : quadrilateral_conditions AC CD AE angleABC angleACD) :
  area_quadrilateral AC CD AE = 376 :=
sorry

end area_of_quadrilateral_l382_382404


namespace solve_inequalities_l382_382886

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382886


namespace number_of_solutions_l382_382438

-- Define the conditions and the proof statement

def equation1 (a x : ℝ) : Prop := 4^x - 4^(-x) = 2 * Real.cos (a * x)

def equation2 (a x : ℝ) : Prop := 4^x + 4^(-x) = 2 * Real.cos (a * x) + 4

theorem number_of_solutions (a : ℝ) (h : ∃ f : ℝ → Prop, (∀ x, f x ↔ equation1 a x) ∧ (∀ s, s ↔ (f.to_fun) = 2015))
: (∃ g : ℝ → Prop, (∀ x, g x ↔ equation2 a x) ∧ (∀ s, s ↔ (g.to_fun) = 4030)) := 
by
sorry

end number_of_solutions_l382_382438


namespace f_increasing_on_neg_infinity_neg_one_l382_382709

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log a (Real.abs (x + 1))

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.abs (x - 1)

theorem f_increasing_on_neg_infinity_neg_one (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) 
  (h_gt_zero : ∀ x : ℝ, -1 < x ∧ x < 0 → g a x > 0) :
  ∀ x y : ℝ, x < y ∧ y < -1 → f a x < f a y := 
by
  sorry

end f_increasing_on_neg_infinity_neg_one_l382_382709


namespace resulting_triangles_are_similar_l382_382346

theorem resulting_triangles_are_similar 
  (triangleABC : Triangle)
  (iterate_draw_medians : Triangle → List Triangle)
  (finite_similarity_classes : Finite (Set (Triangle → Prop)))
  (min_angle_condition : ∀ (triangle : Triangle), min_angle(triangle) ≥ min_angle(triangleABC) / 2) :
  ∃ classes : Finite (Set (Triangle → Prop)),
    (∀ triangle : Triangle, triangle ∈ iterate_draw_medians triangleABC → 
      ∃ class ∈ classes, class triangle) ∧ 
    ∀ triangle ∈ iterate_draw_medians triangleABC, 
      (∀ angle ∈ angles(triangle), angle ≥ min_angle(triangleABC) / 2) := 
sorry

end resulting_triangles_are_similar_l382_382346


namespace sum_of_squares_of_roots_eq_zero_l382_382214

theorem sum_of_squares_of_roots_eq_zero 
  (r : Fin 2020 → ℝ)
  (sum_roots_zero : Finset.univ.sum r = 0)
  (sum_products_zero : ∑ i in Finset.univ.off_diag, r i.1 * r i.2 = 0) :
  Finset.univ.sum (fun i => (r i)^2) = 0 :=
sorry

end sum_of_squares_of_roots_eq_zero_l382_382214


namespace sue_shoes_probability_l382_382419

def sueShoes : List (String × ℕ) := [("black", 7), ("brown", 3), ("gray", 2)]

def total_shoes := 24

def prob_same_color (color : String) (pairs : List (String × ℕ)) : ℚ :=
  let total_pairs := pairs.foldr (λ p acc => acc + p.snd) 0
  let matching_pair := pairs.filter (λ p => p.fst = color)
  if matching_pair.length = 1 then
   let n := matching_pair.head!.snd * 2
   (n / total_shoes) * ((n / 2) / (total_shoes - 1))
  else 0

def prob_total (pairs : List (String × ℕ)) : ℚ :=
  (prob_same_color "black" pairs) + (prob_same_color "brown" pairs) + (prob_same_color "gray" pairs)

theorem sue_shoes_probability :
  prob_total sueShoes = 31 / 138 := by
  sorry

end sue_shoes_probability_l382_382419


namespace arithmetic_sequence_sum_l382_382349

theorem arithmetic_sequence_sum {a : ℕ → ℝ} (d a1 : ℝ)
  (h_arith: ∀ n, a n = a1 + (n - 1) * d)
  (h_condition: a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
by {
  sorry
}

end arithmetic_sequence_sum_l382_382349


namespace boundary_of_set_T_has_4_sides_l382_382719

-- Definitions for the conditions
def in_set_T (a : ℝ) (x y : ℝ) : Prop := 
  a ≤ x ∧ x ≤ 3 * a ∧
  a ≤ y ∧ y ≤ 3 * a ∧
  x + y ≥ 2 * a ∧
  x + 2 * a ≥ 2 * y ∧
  y + 2 * a ≥ 2 * x

-- The problem statement: Boundary of set T is a polygon with 4 sides
theorem boundary_of_set_T_has_4_sides (a : ℝ) (h : 0 < a) :
  ∃ vertices : list (ℝ × ℝ), 
    list.length vertices = 4 ∧
    ∀ x y, in_set_T a x y → ∃ P ∈ vertices, (x, y) = P :=
sorry

end boundary_of_set_T_has_4_sides_l382_382719


namespace y_intercept_of_line_distance_point_to_line_l382_382288

def line_equation (x y : ℝ) : Prop := x - y - 1 = 0

def point_P := (-2 : ℝ, 2 : ℝ)

theorem y_intercept_of_line :
  ∃ y : ℝ, line_equation 0 y ∧ y = -1 :=
by
  sorry

theorem distance_point_to_line :
  let d := (5 * Real.sqrt 2) / 2 in
  ∃ dist : ℝ, dist = d ∧ abs (1 * point_P.1 + -1 * point_P.2 - 1) / Real.sqrt((1:ℝ)^2 + (-1)^2) = d :=
by
  sorry

end y_intercept_of_line_distance_point_to_line_l382_382288


namespace milk_for_750ml_flour_l382_382454

-- Define the conditions from step a)
def ratio_milk_to_flour : ℝ := 50 / 250

-- Define the input amount of flour
def flour_amount (flour : ℝ) : ℝ := 750

-- Calculating the expected amount of milk from the ratio and the input amount of flour
def milk_needed (flour : ℝ) : ℝ := ratio_milk_to_flour * flour / 250 * 250

-- The main theorem to prove
theorem milk_for_750ml_flour : milk_needed 750 = 150 :=
by
  sorry

end milk_for_750ml_flour_l382_382454


namespace proof_problem_l382_382300

variables (a x y : ℝ)

def l1 (x y a : ℝ) : Prop := x + a * y - a = 0
def l2 (x y a : ℝ) : Prop := a * x - (2 * a - 3) * y + a - 2 = 0

-- The product of the slopes of l1 and l2 equals -1 for perpendicularity
def perpendicular (a : ℝ) : Prop := (1 * -a + (a * (2 * a - 3))) = 0

-- Distance between l1 and l2 when a = -3
def distance_between_lines (a : ℝ) : Prop :=
  ∀ x y: ℝ, l1 x y a → l2 x y a → (a = -3) →
  let c1 := a - (a - 1) in
  let c2 := (a + 3) in
  (|c1 - c2| / √((1 ^ 2) + (-3 ^ 2))) = 2 * √(10) / 15

theorem proof_problem :
  (perpendicular 0 ∨ perpendicular 2) ∧
  distance_between_lines -3 :=
by sorry

end proof_problem_l382_382300


namespace solution_set_l382_382842

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382842


namespace solution_set_system_of_inequalities_l382_382938

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382938


namespace johnson_family_seating_l382_382427

def johnson_family_boys : ℕ := 5
def johnson_family_girls : ℕ := 4
def total_chairs : ℕ := 9
def total_arrangements : ℕ := Nat.factorial total_chairs

noncomputable def seating_arrangements_with_at_least_3_boys : ℕ :=
  let three_boys_block_ways := 7 * (5 * 4 * 3) * Nat.factorial 6
  total_arrangements - three_boys_block_ways

theorem johnson_family_seating : seating_arrangements_with_at_least_3_boys = 60480 := by
  unfold seating_arrangements_with_at_least_3_boys
  sorry

end johnson_family_seating_l382_382427


namespace binary_conversion_205_l382_382219

theorem binary_conversion_205 : 
  let b : list ℕ := [1, 1, 0, 0, 1, 1, 0, 1] in -- binary representation of 205
  let u := b.count 0 in -- number of zeros
  let v := b.count 1 in -- number of ones
  2 * (v - u) = 8 := 
by
  sorry

end binary_conversion_205_l382_382219


namespace real_ratio_sum_values_l382_382252

variables (a b c d : ℝ)

theorem real_ratio_sum_values :
  (a / b + b / c + c / d + d / a = 6) ∧
  (a / c + b / d + c / a + d / b = 8) →
  (a / b + c / d = 2 ∨ a / b + c / d = 4) :=
by
  sorry

end real_ratio_sum_values_l382_382252


namespace paul_alex_meet_l382_382394

-- Hypotheses: Coordinates of Paul and Alex
variables (x1 y1 z1 : ℤ) (x2 y2 z2 : ℤ)
axiom h1 : x1 = 2
axiom h2 : y1 = -3
axiom h3 : z1 = 5
axiom h4 : x2 = 8
axiom h5 : y2 = 3
axiom h6 : z2 = -1

-- Midpoint formula in three dimensions
def midpoint (x1 y1 z1 x2 y2 z2 : ℤ) : ℤ × ℤ × ℤ :=
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)

-- The proof goal
theorem paul_alex_meet : midpoint x1 y1 z1 x2 y2 z2 = (5, 0, 2) :=
  by
    rw [h1, h2, h3, h4, h5, h6]
    simp [midpoint]
    -- Final verification steps can occur here but are omitted
    sorry

end paul_alex_meet_l382_382394


namespace banana_final_cost_is_99_l382_382426

def initial_cost : ℝ := 100
def rise_percentage : ℝ := 0.10
def drop_percentage : ℝ := 0.10

noncomputable def final_cost : ℝ :=
  let cost_after_rise := initial_cost * (1 + rise_percentage)
  let cost_after_drop := cost_after_rise * (1 - drop_percentage)
  cost_after_drop

theorem banana_final_cost_is_99 : final_cost = 99 := by
  unfold final_cost initial_cost rise_percentage drop_percentage
  simp
  norm_num
  sorry

end banana_final_cost_is_99_l382_382426


namespace one_prime_in_list_l382_382637

theorem one_prime_in_list (l : List ℕ) (h : l = [1, 11, 111, 1111]) :
  l.count (λ n, Nat.Prime n) = 1 :=
by
  sorry

end one_prime_in_list_l382_382637


namespace janets_litter_box_l382_382712

variable (buy_price weekly_cost total_duration days_per_week weight_per_container cost_per_container real_weight_per_container : ℝ)
variable (weeks weight_per_week : ℝ)

-- Definitions based on the given conditions
def cost_per_week (total_duration days_per_week weekly_cost : ℝ) : ℝ := 
  weekly_cost * (total_duration / days_per_week)

def cost_per_pound (cost_per_container weight_per_container : ℝ) : ℝ := 
  cost_per_container / weight_per_container

def pounds_per_week (weekly_cost cost_per_pound : ℝ) : ℝ := 
  weekly_cost / cost_per_pound 

-- The main theorem to be proved
theorem janets_litter_box: 
  buy_price = 21 → 
  weight_per_container = 45 → 
  total_duration = 210 → 
  days_per_week = 7 → 
  weekly_cost = 210 → 
  real_weight_per_container = 15 :=
begin
  sorry
end

end janets_litter_box_l382_382712


namespace second_group_product_number_l382_382080

theorem second_group_product_number (a₀ : ℕ) (h₀ : 0 ≤ a₀ ∧ a₀ < 20)
  (h₁ : 4 * 20 + a₀ = 94) : 1 * 20 + a₀ = 34 :=
by
  sorry

end second_group_product_number_l382_382080


namespace correct_propositions_incorrect_propositions_l382_382730

variables (l m n : Type) (α β γ : Type)
variables [line l] [line m] [line n] [plane α] [plane β] [plane γ]
variables [perpendicular l α] [perpendicular m α] [lies_in m β]
variables [projection l β n] [perpendicular m l] [lies_in m α]
variables [parallel m n] [perpendicular α γ] [perpendicular β γ]

-- Proposition 1: If l ⊥ α and m ⊥ α, then l ∥ m.
def proposition_1 : Prop := (perpendicular l α ∧ perpendicular m α) → parallel l m

-- Proposition 2: If m lies in β, and n is the projection of l within β, and m ⊥ l, then m ⊥ n.
def proposition_2 : Prop := (lies_in m β ∧ projection l β n ∧ perpendicular m l) → perpendicular m n

-- Proposition 3: If m lies in α, and m ∥ n, then n ∥ α.
def proposition_3 : Prop := (lies_in m α ∧ parallel m n) → parallel n α

-- Proposition 4: If α ⊥ γ and β ⊥ γ, then α ∥ β.
def proposition_4 : Prop := (perpendicular α γ ∧ perpendicular β γ) → parallel α β

-- To prove that Propositions 1 and 2 are true, and Propositions 3 and 4 are false:
theorem correct_propositions : proposition_1 l m n α ∧ proposition_2 l m n α β :=
by {
    -- We'll put the placeholder 'sorry' to represent that the proof is not yet provided.
    sorry
}

theorem incorrect_propositions : ¬ proposition_3 l m n α ∧ ¬ proposition_4 α β γ :=
by {
    -- We'll put the placeholder 'sorry' to represent that the proof is not yet provided.
    sorry
}

end correct_propositions_incorrect_propositions_l382_382730


namespace derivative_at_pi_over_3_l382_382800

def f (x : ℝ) : ℝ := 1 + Real.sin x

theorem derivative_at_pi_over_3 : 
  (Real.cos (Real.pi / 3) = 1 / 2) :=
by 
  sorry

end derivative_at_pi_over_3_l382_382800


namespace red_basket_fruit_count_l382_382081

-- Defining the basket counts
def blue_basket_bananas := 12
def blue_basket_apples := 4
def blue_basket_fruits := blue_basket_bananas + blue_basket_apples
def red_basket_fruits := blue_basket_fruits / 2

-- Statement of the proof problem
theorem red_basket_fruit_count : red_basket_fruits = 8 := by
  sorry

end red_basket_fruit_count_l382_382081


namespace solve_inequalities_l382_382901

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382901


namespace count_primes_between_71_and_95_l382_382647

theorem count_primes_between_71_and_95 : 
  let primes := [71, 73, 79, 83, 89, 97] in
  let filtered_primes := primes.filter (λ p, 71 < p ∧ p < 95) in
  filtered_primes.length = 5 := 
by 
  sorry

end count_primes_between_71_and_95_l382_382647


namespace count_primes_between_71_and_95_l382_382649

theorem count_primes_between_71_and_95 : 
  let primes := [71, 73, 79, 83, 89, 97] in
  let filtered_primes := primes.filter (λ p, 71 < p ∧ p < 95) in
  filtered_primes.length = 5 := 
by 
  sorry

end count_primes_between_71_and_95_l382_382649


namespace linear_inequalities_solution_l382_382874

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382874


namespace proof_new_midpoint_and_distance_l382_382392

open Real EuclideanSpace

variables (p q r s : ℝ)
def initial_midpoint (p q r s : ℝ) : ℝ × ℝ :=
  ((p + r) / 2, (q + s) / 2)

def new_midpoint (p q r s : ℝ) : ℝ × ℝ :=
  (((p + 5) + (r - 15)) / 2, ((q + 10) + (s - 5)) / 2)

def distance (a b : ℝ × ℝ) : ℝ :=
  sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem proof_new_midpoint_and_distance :
  let n := initial_midpoint p q r s,
      n' := new_midpoint p q r s,
      expected_n' := (n.1 - 5, n.2 + 2.5),
      expected_distance := 5.59
  in
  n' = expected_n' ∧ distance n n' = expected_distance := by
  sorry

end proof_new_midpoint_and_distance_l382_382392


namespace percent_of_x_is_y_l382_382673

theorem percent_of_x_is_y 
    (x y : ℝ) 
    (h : 0.30 * (x - y) = 0.20 * (x + y)) : 
    y / x = 0.2 :=
  sorry

end percent_of_x_is_y_l382_382673


namespace systematic_sampling_l382_382991

theorem systematic_sampling :
  let N := 60
  let n := 5
  let k := N / n
  let initial_sample := 5
  let samples := [initial_sample, initial_sample + k, initial_sample + 2 * k, initial_sample + 3 * k, initial_sample + 4 * k] 
  samples = [5, 17, 29, 41, 53] := sorry

end systematic_sampling_l382_382991


namespace intersection_points_of_lines_l382_382558

theorem intersection_points_of_lines :
  let L1 := λ x y : ℝ, 3 * x - 9 * y + 18 = 0,
      L2 := λ x y : ℝ, 6 * x - 18 * y - 36 = 0,
      L3 := λ x : ℝ, x = 3,
      points := Set.univ × Set.univ -- essentially representing ℝ²
  in ∃! (p : ℝ × ℝ), (p ∈ points ∧ (L1 p.1 p.2 ∧ L3 p.1)) ∧ (L2 p.1 p.2 ∧ L3 p.1) :=
  sorry

end intersection_points_of_lines_l382_382558


namespace dividend_calculation_l382_382496

theorem dividend_calculation :
  let divisor := 17
  let quotient := 9
  let remainder := 7
  let dividend := (divisor * quotient) + remainder
  dividend = 160 :=
by
  simp [dividend, divisor, quotient, remainder]
  -- sorry

end dividend_calculation_l382_382496


namespace solution_set_of_linear_inequalities_l382_382864

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382864


namespace cost_per_lunch_is_7_l382_382713

-- Definitions of the conditions
def total_children := 35
def total_chaperones := 5
def janet := 1
def additional_lunches := 3
def total_cost := 308

-- Calculate the total number of lunches
def total_lunches : Int :=
  total_children + total_chaperones + janet + additional_lunches

-- Statement to prove that the cost per lunch is 7
theorem cost_per_lunch_is_7 : total_cost / total_lunches = 7 := by
  sorry

end cost_per_lunch_is_7_l382_382713


namespace white_dandelions_on_saturday_l382_382149

-- Define the life cycle of a dandelion
structure Dandelion where
  status : ℕ  -- 0: yellow, 1: white
  day_observed : ℕ  -- Day it was observed (0: Monday, 1: Tuesday, ...)

-- Define initial conditions for Monday and Wednesday
def monday_yellow := 20
def monday_white := 14
def wednesday_yellow := 15
def wednesday_white := 11

-- Theorem to prove the number of white dandelions on Saturday
theorem white_dandelions_on_saturday 
  (monday_yellow monday_white wednesday_yellow wednesday_white : ℕ)
  (new_dandelions : ℕ) :
  monday_yellow = 20 →
  monday_white = 14 →
  wednesday_yellow = 15 →
  wednesday_white = 11 →
  new_dandelions = (wednesday_yellow + wednesday_white) - (monday_yellow + monday_white) →
  ∃ white_dandelions_on_saturday : ℕ, white_dandelions_on_saturday = new_dandelions  
:=
begin
  intros,
  existsi new_dandelions,
  sorry
end

end white_dandelions_on_saturday_l382_382149


namespace encounter_count_l382_382997

theorem encounter_count (vA vB d : ℝ) (h₁ : 5 * d / vA = 9 * d / vB) :
  ∃ encounters : ℝ, encounters = 3023 :=
by
  sorry

end encounter_count_l382_382997


namespace tangent_perpendicular_slope_l382_382075

theorem tangent_perpendicular_slope (a : ℝ) (h : ∀ x : ℝ, (deriv (λ x, (e^(a*x)) * (cos x))) 0 = a ∧ 
  (∀ x y : ℝ, x + 2*y = 0 → y = 1) ∧ (-1/(2 : ℝ) * a = -1)) : a = 2 :=
sorry

end tangent_perpendicular_slope_l382_382075


namespace intersection_points_of_C1_and_C2_distance_AB_l382_382702

theorem intersection_points_of_C1_and_C2 : 
  ∃ (p1 p2 : ℝ × ℝ), 
    ((p1 = (0, 0) ∨ p1 = (3 / 2, real.sqrt (3) / 2)) ∧ 
     (p2 = (0, 0) ∨ p2 = (3 / 2, real.sqrt (3) / 2)) ∧ 
     (p1 ≠ p2) ∧ 
      (∃ ρ θ, ρ = 2 * real.cos θ ∧ (ρ, θ) = p1) ∧ 
      (∃ ρ θ, ρ = 2 * real.cos (θ - real.pi / 3) ∧ (ρ, θ) = p2))
:= sorry

theorem distance_AB :
  let A := (-1, (2 * real.pi) / 3)
  let B := (1, (2 * real.pi) / 3)
  ∃ AB : ℝ, AB = 2 ∧ dist A B = AB
:= sorry

end intersection_points_of_C1_and_C2_distance_AB_l382_382702


namespace proof_problem_l382_382299

variables (a x y : ℝ)

def l1 (x y a : ℝ) : Prop := x + a * y - a = 0
def l2 (x y a : ℝ) : Prop := a * x - (2 * a - 3) * y + a - 2 = 0

-- The product of the slopes of l1 and l2 equals -1 for perpendicularity
def perpendicular (a : ℝ) : Prop := (1 * -a + (a * (2 * a - 3))) = 0

-- Distance between l1 and l2 when a = -3
def distance_between_lines (a : ℝ) : Prop :=
  ∀ x y: ℝ, l1 x y a → l2 x y a → (a = -3) →
  let c1 := a - (a - 1) in
  let c2 := (a + 3) in
  (|c1 - c2| / √((1 ^ 2) + (-3 ^ 2))) = 2 * √(10) / 15

theorem proof_problem :
  (perpendicular 0 ∨ perpendicular 2) ∧
  distance_between_lines -3 :=
by sorry

end proof_problem_l382_382299


namespace solution_set_l382_382844

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382844


namespace sumata_family_trip_cost_l382_382788

noncomputable def total_cost_trip (distances : List ℝ) (gas_mileage : ℝ) (gas_price : ℝ) (tolls : ℝ) : ℝ :=
  let total_distance := distances.sum
  let total_gallons := total_distance / gas_mileage
  let total_gas_cost := total_gallons * gas_price
  total_gas_cost + tolls

theorem sumata_family_trip_cost
  (distances : List ℝ)
  (gas_mileage : ℝ)
  (gas_price : ℝ)
  (tolls : ℝ)
  (h_distances : distances = [150, 270, 200, 380, 320, 255, 120])
  (h_gas_mileage : gas_mileage = 25)
  (h_gas_price : gas_price = 3.5)
  (h_tolls : tolls = 35) :
  total_cost_trip distances gas_mileage gas_price tolls = 272.30 := 
by
  sorry

end sumata_family_trip_cost_l382_382788


namespace num_diagonals_decagon_l382_382508

theorem num_diagonals_decagon (n : ℕ) (h : n = 10) : 
  let d := n * (n - 3) / 2 in
  d = 35 :=
by
  -- Given n = 10
  rw h
  -- Now calculate d
  let d := 10 * (10 - 3) / 2
  have : d = 35, sorry
  exact this

end num_diagonals_decagon_l382_382508


namespace solution_set_linear_inequalities_l382_382910

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382910


namespace count_primes_with_squares_in_range_l382_382663

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end count_primes_with_squares_in_range_l382_382663


namespace smallest_n_b_n_eq_b_0_l382_382319

theorem smallest_n_b_n_eq_b_0 :
  let b : ℕ → ℝ := λ n, nat.rec_on n (cos (π / 18) ^ 2) (λ n b_n, 4 * b_n * (1 - b_n))
  in ∃ (n : ℕ), n > 0 ∧ b n = b 0 ∧ ∀ m, m > 0 ∧ b m = b 0 → n ≤ m :=
by
  sorry

end smallest_n_b_n_eq_b_0_l382_382319


namespace cut_wood_into_5_pieces_l382_382521

-- Definitions
def pieces_to_cuts (pieces : ℕ) : ℕ := pieces - 1
def time_per_cut (total_time : ℕ) (cuts : ℕ) : ℕ := total_time / cuts
def total_time_for_pieces (pieces : ℕ) (time_per_cut : ℕ) : ℕ := (pieces_to_cuts pieces) * time_per_cut

-- Given conditions
def conditions : Prop :=
  pieces_to_cuts 4 = 3 ∧
  time_per_cut 24 (pieces_to_cuts 4) = 8

-- Problem statement
theorem cut_wood_into_5_pieces (h : conditions) : total_time_for_pieces 5 8 = 32 :=
by sorry

end cut_wood_into_5_pieces_l382_382521


namespace solution_set_of_linear_inequalities_l382_382862

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382862


namespace probability_of_selecting_Zhang_l382_382752

-- Define the set of doctors
inductive Doctor
| Zhang | Wang | Li | Liu

open Doctor

-- Define the selection process
def select_two_doctors (d1 d2 : Doctor) : Prop := 
  d1 ≠ d2 

-- Define the set of all possible selections of 2 doctors out of 4
def possible_selections : list (Doctor × Doctor) := 
  [(Zhang, Wang), (Zhang, Li), (Zhang, Liu), (Wang, Li), (Wang, Liu), (Li, Liu)]

-- Define the set of selections that include Dr. Zhang
def selections_with_Zhang : list (Doctor × Doctor) := 
  [(Zhang, Wang), (Zhang, Li), (Zhang, Liu)]

-- Define the proof statement
theorem probability_of_selecting_Zhang : 
  (selections_with_Zhang.length : ℚ) / (possible_selections.length : ℚ) = 1 / 2 := 
  sorry

end probability_of_selecting_Zhang_l382_382752


namespace mary_lambs_count_l382_382742

def initial_lambs : Nat := 6
def baby_lambs : Nat := 2 * 2
def traded_lambs : Nat := 3
def extra_lambs : Nat := 7

theorem mary_lambs_count : initial_lambs + baby_lambs - traded_lambs + extra_lambs = 14 := by
  sorry

end mary_lambs_count_l382_382742


namespace non_fiction_vs_fiction_diff_l382_382448

def total_books : Nat := 35
def fiction_books : Nat := 5
def picture_books : Nat := 11
def autobiography_books : Nat := 2 * fiction_books

def accounted_books : Nat := fiction_books + autobiography_books + picture_books
def non_fiction_books : Nat := total_books - accounted_books

theorem non_fiction_vs_fiction_diff :
  non_fiction_books - fiction_books = 4 := by 
  sorry

end non_fiction_vs_fiction_diff_l382_382448


namespace river_depth_l382_382525

noncomputable def depth_of_river (w r V : ℝ) (r_conv : r = 8 * 1000 / 60) (V_conv : V = 26666.666666666668) : ℝ :=
  V / (w * r_conv)

theorem river_depth (width : ℝ) (flow_rate : ℝ) (volume : ℝ)
  (h_width : width = 25)
  (h_flow_rate : flow_rate = 8 * 1000 / 60)
  (h_volume : volume = 26666.666666666668) : depth_of_river width flow_rate volume = 8 := 
by {
  rw [h_width, h_flow_rate, h_volume],
  -- calculations
  have flow_rate := 8000 / 60,
  have cross_sectional_area := 25 * 8, -- cross-sectional area here
  have volume := 26666.666666666668,
  rw [cross_sectional_area] at volume,
  have depth := volume / cross_sectional_area,
  rw depth,
  sorry,
}

end river_depth_l382_382525


namespace solution_set_of_inequalities_l382_382963

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382963


namespace shortest_distance_phenomena_explained_l382_382533

def condition1 : Prop :=
  ∀ (a b : ℕ), (exists nail1 : ℕ, exists nail2 : ℕ, nail1 ≠ nail2) → (exists wall : ℕ, wall = a + b)

def condition2 : Prop :=
  ∀ (tree1 tree2 tree3 : ℕ), tree1 ≠ tree2 → tree2 ≠ tree3 → (tree1 + tree2 + tree3) / 3 = tree2

def condition3 : Prop :=
  ∀ (A B : ℕ), ∃ (C : ℕ), C = (B - A) → (A = B - (B - A))

def condition4 : Prop :=
  ∀ (dist : ℕ), dist = 0 → exists shortest : ℕ, shortest < dist

-- The following theorem needs to be proven to match our mathematical problem
theorem shortest_distance_phenomena_explained :
  condition3 ∧ condition4 :=
by
  sorry

end shortest_distance_phenomena_explained_l382_382533


namespace geometric_sequence_a1_l382_382619

theorem geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) 
  (hq : 0 < q)
  (h1 : a 4 * a 8 = 2 * (a 5) ^ 2)
  (h2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end geometric_sequence_a1_l382_382619


namespace smallest_prime_perimeter_scalene_triangle_l382_382562

-- Define prime numbers and properties we'll use for the problem
open nat

def is_prime_gt_3 (n : ℕ) : Prop := prime n ∧ n > 3

-- Define the scaled triangle and conditions in Lean
def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a

-- The main theorem
theorem smallest_prime_perimeter_scalene_triangle : 
  ∃ (a b c : ℕ), is_prime_gt_3 a ∧ is_prime_gt_3 b ∧ is_prime_gt_3 c ∧ is_scalene_triangle a b c ∧ prime (a + b + c) ∧ a + b + c = 23 := 
sorry

end smallest_prime_perimeter_scalene_triangle_l382_382562


namespace number_of_9_step_paths_l382_382158

-- Definitions and conditions
def is_white_square (i j : ℕ) : Bool := (i + j) % 2 = 0

def valid_move (i j k l : ℕ) : Bool :=
  (k = i + 1) ∧ (l = j ∨ l = j + 1 ∨ l = j - 1) ∧ is_white_square k l

def count_paths (n : ℕ) : ℕ :=
  ∑ b in Finset.range (n / 2 + 1),
    Nat.choose n b * Nat.choose (n - b) b

theorem number_of_9_step_paths :
  count_paths 9 = 457 := by
  sorry

end number_of_9_step_paths_l382_382158


namespace sequence_a_10_l382_382604

theorem sequence_a_10 (a : ℕ → ℤ) 
  (H1 : ∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p + a q)
  (H2 : a 2 = -6) : 
  a 10 = -30 :=
sorry

end sequence_a_10_l382_382604


namespace smallest_positive_z_l382_382783

theorem smallest_positive_z (x z : ℝ) (k : ℤ) (hx : sin x = 0) (hxk : x = k * Real.pi) (h_cos : cos (x + z) = -1 / 2) :
  z = 2 * Real.pi / 3 :=
by
  sorry

end smallest_positive_z_l382_382783


namespace incorrect_reasoning_l382_382818

theorem incorrect_reasoning {a : ℝ} (h1 : 0 < a) (h2 : a ≠ 1) :
  ¬ (∀ x : ℝ, a > 1 → a^x increases in x) ∨ 
  ¬ (∀ x : ℝ, 0 < a ∧ a < 1 → a^x decreases in x) :=
by 
  sorry

end incorrect_reasoning_l382_382818


namespace solution_set_linear_inequalities_l382_382929

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382929


namespace angle_YZX_in_incircum_circle_l382_382548

theorem angle_YZX_in_incircum_circle 
  (ABC: Triangle) 
  (XYZ: Triangle) 
  (Γ: Circle) 
  (X Y Z: Point)
  (hΓ_incirc: Γ.isIncircle ABC)
  (hΓ_circum: Γ.isCircumcircle XYZ)
  (hX: X ∈ segment (ABC.b, ABC.c))
  (hY: Y ∈ segment (ABC.a, ABC.b))
  (hZ: Z ∈ segment (ABC.a, ABC.c))
  (h_∠A: ABC.angleA = 40)
  (h_∠B: ABC.angleB = 60)
  (h_∠C: ABC.angleC = 80) : 
  XYZ.angleYZX = 60 := 
sorry

end angle_YZX_in_incircum_circle_l382_382548


namespace proof_f_2017_l382_382374

-- Define the conditions provided in the problem
variable (f : ℝ → ℝ)
variable (hf : ∀ x, f (-x) = -f x) -- f is an odd function
variable (h1 : ∀ x, f (-x + 1) = f (x + 1))
variable (h2 : f (-1) = 1)

-- Define the Lean statement that proves the correct answer
theorem proof_f_2017 : f 2017 = -1 :=
sorry

end proof_f_2017_l382_382374


namespace smallest_positive_z_l382_382420

theorem smallest_positive_z :
  (∃ x z : ℝ, cos x = 0 ∧ cos (x + z) = 1/2 ∧ ∀ w : ℝ, (cos x = 0 ∧ cos (x + w) = 1/2) → (0 < w → z ≤ w)) → z = π / 6 :=
by
  intro h
  sorry

end smallest_positive_z_l382_382420


namespace Cindy_initial_marbles_l382_382545

theorem Cindy_initial_marbles (M : ℕ) 
  (h1 : 4 * (M - 320) = 720) : M = 500 :=
by
  sorry

end Cindy_initial_marbles_l382_382545


namespace white_dandelions_on_saturday_l382_382142

theorem white_dandelions_on_saturday 
  (yellow_monday : ℕ) (white_monday : ℕ)
  (yellow_wednesday : ℕ) (white_wednesday : ℕ)
  (life_cycle : ∀ d, (d.bloom == "yellow" ∧ d.age == 3) ∨ (d.bloom == "white" ∧ d.age == 4) ∨ (d.bloom == "dispersed" ∧ d.age == 5))
  (total_monday : yellow_monday = 20 ∧ white_monday = 14)
  (total_wednesday : yellow_wednesday = 15 ∧ white_wednesday = 11):
  let new_dandelions := 26 - 20,
      white_dandelions_saturday := new_dandelions
  in white_dandelions_saturday = 6 := by
  let dandelions_tuesday_wednesday := new_dandelions
  have h : white_dandelions_saturday = 6,
  from sorry
  exact h

end white_dandelions_on_saturday_l382_382142


namespace bill_amount_is_correct_l382_382076

-- Define the given conditions
def true_discount : ℝ := 189
def rate : ℝ := 0.16
def time : ℝ := 9 / 12

-- Define the true discount formula
def true_discount_formula (FV : ℝ) (R : ℝ) (T : ℝ) : ℝ := 
  (FV * R * T) / (100 + (R * T))

-- State that we want to prove that the Face Value is Rs. 1764
theorem bill_amount_is_correct : ∃ (FV : ℝ), FV = 1764 ∧ true_discount = true_discount_formula FV rate time :=
sorry

end bill_amount_is_correct_l382_382076


namespace rectangle_perimeter_l382_382215

theorem rectangle_perimeter (y z x : ℝ) :
  ∃ p : ℝ, p = 2 * y + 2 * z - 4 * x :=
by
  use 2 * y + 2 * z - 4 * x
  sorry

end rectangle_perimeter_l382_382215


namespace initial_bottle_caps_l382_382566

theorem initial_bottle_caps (end_caps : ℕ) (eaten_caps : ℕ) (start_caps : ℕ) 
  (h1 : end_caps = 61) 
  (h2 : eaten_caps = 4) 
  (h3 : start_caps = end_caps + eaten_caps) : 
  start_caps = 65 := 
by 
  sorry

end initial_bottle_caps_l382_382566


namespace find_a_pure_imaginary_l382_382279

theorem find_a_pure_imaginary (a : ℝ) (i : ℂ) (h1 : i = (0 : ℝ) + I) :
  (∃ b : ℝ, a - (17 / (4 - i)) = (0 + b*I)) → a = 4 :=
by
  sorry

end find_a_pure_imaginary_l382_382279


namespace find_numbers_l382_382980

theorem find_numbers (a b c : ℕ) (h : a + b = 2015) (h' : a = 10 * b + c) (hc : 0 ≤ c ∧ c ≤ 9) :
  (a = 1832 ∧ b = 183) :=
sorry

end find_numbers_l382_382980


namespace range_of_a_l382_382669

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + a * x + 3 ≥ a) ↔ -7 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_a_l382_382669


namespace mary_maximum_hours_l382_382383

def regular_rate : ℝ := 8
def overtime_rate : ℝ := regular_rate * 1.25
def max_earnings : ℝ := 560
def first_20_hours_earnings : ℝ := 20 * regular_rate

theorem mary_maximum_hours :
  let overtime_earnings := max_earnings - first_20_hours_earnings,
      overtime_hours := overtime_earnings / overtime_rate,
      total_hours := 20 + overtime_hours in
  total_hours = 60 := by
  sorry

end mary_maximum_hours_l382_382383


namespace solution_set_inequalities_l382_382833

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382833


namespace determine_x_y_l382_382575

theorem determine_x_y (x y : ℝ) (h1 : x - (11 / 17) * x = 150) (h2 : y - (Real.sqrt 2 / 3) * y = 90) :
  x = 425 ∧ y = (270 + 90 * Real.sqrt 2) / 7 :=
by
  split
  sorry
  sorry

end determine_x_y_l382_382575


namespace value_of_m_l382_382324

theorem value_of_m (m : ℤ) : (∃ x : ℚ, x = 1 / 2 ∧ (10 * x + m = 2)) → m = -3 := by
  intro h
  cases h with x hx
  cases hx with hx1 hx2
  rw [hx1, (show 10 * (1/2 : ℚ) = 5, by norm_num)] at hx2
  linarith

end value_of_m_l382_382324


namespace length_of_first_train_l382_382459

-- Define the given conditions
def speed_train_1_kmph : ℝ := 80
def speed_train_2_kmph : ℝ := 65
def cleartime_seconds : ℝ := 7.695936049253991
def length_train_2_meters : ℝ := 200

-- Conversion constants
def kmph_to_mps_factor : ℝ := 1000 / 3600

-- Calculate relative speed in m/s
def relative_speed_mps : ℝ := (speed_train_1_kmph + speed_train_2_kmph) * kmph_to_mps_factor

-- Calculate total distance covered in given time
def total_distance_meters : ℝ := relative_speed_mps * cleartime_seconds

-- Problem to prove that the length of the first train is 110 meters
theorem length_of_first_train : length_train_2_meters < total_distance_meters :=
by { sorry }

end length_of_first_train_l382_382459


namespace tournament_result_l382_382350

theorem tournament_result :
  let participants := 99
  let alfred_algebra := 16
  let alfred_combinatorics := 30
  let alfred_geometry := 23
  let total_score := alfred_algebra + alfred_combinatorics + alfred_geometry
  (= total_score 69) →
  let B := 1 -- Best possible ranking
  let W := 67 -- Worst possible ranking
  (100 * B + W) = 167 :=
by
  intros _ _ _ _ h_total_score
  have hB : 1 = B := by rfl
  have hW : 67 = W := by rfl
  rw [hB, hW, mul_one]
  exact h_total_score.symm

end tournament_result_l382_382350


namespace find_f_minus_g_squared_l382_382000

-- Definition of the quadratic equation and its relation to f and g
def quadratic_eq (x : ℝ) : Prop := 4 * x^2 + 8 * x - 48 = 0

-- Definitions of the roots
variable (f g : ℝ)
variable h₁ : quadratic_eq f
variable h₂ : quadratic_eq g

-- The theorem to be proved
theorem find_f_minus_g_squared : (f - g) ^ 2 = 49 :=
sorry

end find_f_minus_g_squared_l382_382000


namespace soda_cost_l382_382477

theorem soda_cost (x : ℝ) : 
    (1.5 * 35 + x * (87 - 35) = 78.5) → 
    x = 0.5 := 
by 
  intros h
  sorry

end soda_cost_l382_382477


namespace circle_tangent_problem_l382_382547

theorem circle_tangent_problem
  (o : Set Point) (m p r : Set Point)
  (radius_o radius_m radius_p radius_r : ℝ)
  (center_o center_r : Point)
  (tangent_o_m tangent_o_p tangent_o_r tangent_m_p tangent_m_r tangent_p_r : Prop)
  (passes_center_r : Prop)
  (x : ℝ)
  (cond1 : radius_o > 0)
  (cond2 : radius_m = x)
  (cond3 : radius_p = x)
  (cond4 : radius_r = 1)
  (cond5 : passes_center_r → ∥center_o - center_r∥ = 1)
  (cond6 : tangent_o_m)
  (cond7 : tangent_o_p)
  (cond8 : tangent_o_r)
  (cond9 : tangent_m_p)
  (cond10: tangent_m_r)
  (cond11: tangent_p_r) :
  x = 8 / 9 :=
by
  sorry

end circle_tangent_problem_l382_382547


namespace linear_inequalities_solution_l382_382879

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382879


namespace derivative_at_one_l382_382799

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * Real.log x

theorem derivative_at_one : (deriv f 1) = 5 := 
by 
  sorry

end derivative_at_one_l382_382799


namespace david_trip_expense_l382_382220

theorem david_trip_expense :
  ∃ spent_amount difference : ℕ,
    let initial_amount := 1800 in
    let remaining_amount := 500 in
    spent_amount = initial_amount - remaining_amount ∧
    difference = spent_amount - remaining_amount ∧
    difference = 800 :=
begin
  use [1300, 800],
  simp [initial_amount, remaining_amount],
  split,
  { simp, },
  { split; simp, }
end

end david_trip_expense_l382_382220


namespace solution_set_of_inequalities_l382_382952

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382952


namespace dhoni_spent_on_rent_l382_382222

noncomputable theory

-- Define the conditions for Dhoni's spending
def total_earnings (E : ℝ) : ℝ := E
def percent_spent_on_rent (R : ℝ) (E : ℝ) : ℝ := (R / 100) * E
def percent_spent_on_dishwasher (R : ℝ) (E : ℝ) : ℝ := ((R - 5) / 100) * E
def percent_left_over (L : ℝ) (E : ℝ) : ℝ := (61 / 100) * E

-- Given and required conditions
variables {E R : ℝ}

def condition1 := percent_left_over 61 E = 0.61 * E
def condition2 := percent_spent_on_rent R E + percent_spent_on_dishwasher R E = 0.39 * E
def condition3 := 2 * R - 5 = 39

theorem dhoni_spent_on_rent (E R : ℝ) 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) : R = 22 :=
sorry

end dhoni_spent_on_rent_l382_382222


namespace arithmetic_sequence_no_geometric_progression_l382_382011

theorem arithmetic_sequence_no_geometric_progression {r : ℝ} (a : ℕ → ℝ) (k l : ℕ) (h_arith : ∀ n, a (n+1) - a n = r)
(h_contains_terms : a k = 1 ∧ a l = real.sqrt 2)
: ¬∃ m n p : ℕ, m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ (a m, a n, a p) form_geometric_sequence := sorry

end arithmetic_sequence_no_geometric_progression_l382_382011


namespace solution_set_linear_inequalities_l382_382920

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382920


namespace factorize_expression_l382_382569

variables {a x y : ℝ}

theorem factorize_expression (a x y : ℝ) : 3 * a * x ^ 2 + 6 * a * x * y + 3 * a * y ^ 2 = 3 * a * (x + y) ^ 2 :=
by
  sorry

end factorize_expression_l382_382569


namespace div_by_3kp1_iff_div_by_3k_l382_382399

theorem div_by_3kp1_iff_div_by_3k (m n k : ℕ) (h1 : m > n) :
  (3 ^ (k + 1)) ∣ (4 ^ m - 4 ^ n) ↔ (3 ^ k) ∣ (m - n) := 
sorry

end div_by_3kp1_iff_div_by_3k_l382_382399


namespace find_initial_mangoes_l382_382503

-- Define the initial conditions
def initial_apples : Nat := 7
def initial_oranges : Nat := 8
def apples_taken : Nat := 2
def oranges_taken : Nat := 2 * apples_taken
def remaining_fruits : Nat := 14
def mangoes_remaining (M : Nat) : Nat := M / 3

-- Define the problem statement
theorem find_initial_mangoes (M : Nat) (hM : 7 - apples_taken + 8 - oranges_taken + mangoes_remaining M = remaining_fruits) : M = 15 :=
by
  sorry

end find_initial_mangoes_l382_382503


namespace solve_inequalities_l382_382898

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382898


namespace find_percentage_l382_382124

variable (P : ℝ)

-- Conditions
def condition1 : Prop := (P / 100) * 1265 / 6 = 480.7

theorem find_percentage (h1 : condition1 P) : P ≈ 228.0316 :=
by
  sorry

end find_percentage_l382_382124


namespace four_digit_numbers_containing_at_least_one_even_digit_l382_382314

noncomputable def total_four_digit_numbers : ℕ := 9999 - 1000 + 1

def odd_digits : Finset ℕ := {1, 3, 5, 7, 9}

noncomputable def count_all_odd_digit_numbers (digits_set: Finset ℕ) : ℕ :=
  let count := digits_set.card
  count * count * count * count

theorem four_digit_numbers_containing_at_least_one_even_digit (total : ℕ) (all_odd : ℕ) : total - all_odd = 8375 :=
  by
    let total := total_four_digit_numbers
    let all_odd := count_all_odd_digit_numbers odd_digits
    have h1 : total = 9000 := rfl
    have h2 : all_odd = 5 * 5 * 5 * 5 := rfl
    have h3 : all_odd = 625 := by rw [h2, pow_succ', pow_succ', pow_succ', pow_zero]
    have h4 : total - all_odd = 9000 - 625 := by rw [h1, h3]
    exact h4.symm ▸ rfl

end four_digit_numbers_containing_at_least_one_even_digit_l382_382314


namespace simplify_and_evaluate_expression_l382_382043

theorem simplify_and_evaluate_expression (x : ℚ) (h : x = 3) :
  ((2 * x^2 + 2 * x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2 * x + 1)) / (x / (x + 1)) = 2 :=
by
  sorry

end simplify_and_evaluate_expression_l382_382043


namespace count_primes_squared_in_range_l382_382657

theorem count_primes_squared_in_range : 
  (finset.card (finset.filter (λ p, p^2 ≥ 5000 ∧ p^2 ≤ 9000) 
    (finset.filter nat.prime (finset.range 95)))) = 5 := 
by sorry

end count_primes_squared_in_range_l382_382657


namespace sum_with_probability_point_two_l382_382992

open Finset

def a : Finset ℤ := {2, 3, 4, 5}
def b : Finset ℤ := {4, 5, 6, 7, 8}

theorem sum_with_probability_point_two :
  ∃ sum, (a.product b).filter (λ p : ℤ × ℤ, p.1 + p.2 = sum).card = 4 :=
begin
  use 10,
  sorry -- proof to be filled out
end

end sum_with_probability_point_two_l382_382992


namespace find_k_l382_382370

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a n = a 0 + n * d

theorem find_k (d : ℝ) (h : d ≠ 0) (a : ℕ → ℝ) (h_seq : arithmetic_sequence a d) (h_a1 : a 0 = 4 * d)
  (h_geom : ∀ k, a k = real.sqrt (a 0 * a (2 * k))) :
  ∃ k : ℕ, k = 3 :=
by
  use 3
  sorry

end find_k_l382_382370


namespace probability_abby_bridget_adjacent_l382_382686

theorem probability_abby_bridget_adjacent :
  let total_arrangements := 9!
  let adjacent_row_column_positions := 18
  let permutations_ab_br := 2
  let arrange_remaining_7 := 7!
  let favorable_outcomes := adjacent_row_column_positions * permutations_ab_br * arrange_remaining_7
  total_arrangements = 362880 → -- 9!
  favorable_outcomes = 181440 →  -- 18 * 2 * 7!
  let probability := favorable_outcomes / total_arrangements
  probability = 1 / 2 :=
by
  sorry

end probability_abby_bridget_adjacent_l382_382686


namespace problem1_problem2_l382_382121

open Set

variable {x y z a b : ℝ}

-- Problem 1: Prove the inequality
theorem problem1 (x y z : ℝ) : 
  5 * x^2 + y^2 + z^2 ≥ 2 * x * y + 4 * x + 2 * z - 2 :=
by
  sorry

-- Problem 2: Prove the range of 10a - 5b is [−1, 20]
theorem problem2 (a b : ℝ) 
  (h1 : 1 ≤ 2 * a + b ∧ 2 * a + b ≤ 4)
  (h2 : -1 ≤ a - 2 * b ∧ a - 2 * b ≤ 2) : 
  -1 ≤ 10 * a - 5 * b ∧ 10 * a - 5 * b ≤ 20 :=
by
  sorry

end problem1_problem2_l382_382121


namespace P_subset_Q_l382_382732

def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x > 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end P_subset_Q_l382_382732


namespace total_amount_is_1000_l382_382194

-- Definitions based on the conditions
def amount_Q : ℝ := 125
def amount_R : ℝ := 125

def amount_P := 2 * amount_Q
def amount_S := 4 * amount_R

-- Theorem statement to prove the total amount is 1000
theorem total_amount_is_1000 :
    let total_amount := amount_P + amount_Q + amount_R + amount_S in
    total_amount = 1000 := by
  have amount_P_eq : amount_P = 2 * 125 := rfl
  have amount_S_eq : amount_S = 4 * 125 := rfl
  have total_calc : total_amount = (2 * 125 + 125 + 125 + 4 * 125) := by
    rw [amount_P_eq, amount_S_eq]
  have total_simplified : 2 * 125 + 125 + 125 + 4 * 125 = 1000 := by
    norm_num
  rw [total_calc, total_simplified]
  rfl

end total_amount_is_1000_l382_382194


namespace total_amount_divided_l382_382131

variables (T x : ℝ)
variables (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
variables (h₂ : T - x = 1100)

theorem total_amount_divided (T x : ℝ) 
  (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
  (h₂ : T - x = 1100) : 
  T = 1600 := 
sorry

end total_amount_divided_l382_382131


namespace infinite_primes_p_solutions_eq_p2_l382_382764

theorem infinite_primes_p_solutions_eq_p2 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ 
  (∃ (S : Finset (ZMod p × ZMod p × ZMod p)),
    S.card = p^2 ∧ ∀ (x y z : ZMod p), (3 * x^3 + 4 * y^4 + 5 * z^3 - y^4 * z = 0) ↔ (x, y, z) ∈ S) :=
sorry

end infinite_primes_p_solutions_eq_p2_l382_382764


namespace sum_of_6_digit_numbers_without_0_9_divisible_by_37_l382_382761

theorem sum_of_6_digit_numbers_without_0_9_divisible_by_37 :
  ∀ (N : ℕ), (6 ≤ (nat.log 10 N + 1)) ∧ (nat.log 10 N + 1 ≤ 6)
             → (∀ d : ℕ, (d ∈ N.digits 10) → (d ≠ 0 → d ≠ 9))
             → (∃ k : ℕ, (sum (filter (λ d, d ≠ 0 ∧ d ≠ 9) (digits 10 N)) = k * 37)) :=
by
  -- proof here
  sorry

end sum_of_6_digit_numbers_without_0_9_divisible_by_37_l382_382761


namespace harry_average_sleep_l382_382313

-- Conditions
def sleep_time_monday : ℕ × ℕ := (8, 15)
def sleep_time_tuesday : ℕ × ℕ := (7, 45)
def sleep_time_wednesday : ℕ × ℕ := (8, 10)
def sleep_time_thursday : ℕ × ℕ := (10, 25)
def sleep_time_friday : ℕ × ℕ := (7, 50)

-- Total sleep time calculation
def total_sleep_time : ℕ × ℕ :=
  let (h1, m1) := sleep_time_monday
  let (h2, m2) := sleep_time_tuesday
  let (h3, m3) := sleep_time_wednesday
  let (h4, m4) := sleep_time_thursday
  let (h5, m5) := sleep_time_friday
  (h1 + h2 + h3 + h4 + h5, m1 + m2 + m3 + m4 + m5)

-- Convert minutes to hours and minutes
def convert_minutes (mins : ℕ) : ℕ × ℕ :=
  (mins / 60, mins % 60)

-- Final total sleep time
def final_total_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := total_sleep_time
  let (extra_hours, remaining_minutes) := convert_minutes total_minutes
  (total_hours + extra_hours, remaining_minutes)

-- Average calculation
def average_sleep_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := final_total_time
  (total_hours / 5, (total_hours % 5) * 60 / 5 + total_minutes / 5)

-- The proof statement
theorem harry_average_sleep :
  average_sleep_time = (8, 29) :=
  by
    sorry

end harry_average_sleep_l382_382313


namespace increase_average_by_3_l382_382792

theorem increase_average_by_3 (x : ℕ) (average_initial : ℕ := 32) (matches_initial : ℕ := 10) (score_11th_match : ℕ := 65) :
  (matches_initial * average_initial + score_11th_match = 11 * (average_initial + x)) → x = 3 := 
sorry

end increase_average_by_3_l382_382792


namespace lambda_sum_leq_n_l382_382721

open Real

theorem lambda_sum_leq_n (n : ℕ) (λ : Fin n → ℝ) :
  (∀ θ : ℝ, (∑ i, λ i * cos ((i.val + 1) * θ)) ≥ -1) →
  (∑ i, λ i) ≤ n := by
  sorry

end lambda_sum_leq_n_l382_382721


namespace solution_set_l382_382843

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382843


namespace wilson_theorem_l382_382038

theorem wilson_theorem (p : ℕ) (hp : Nat.Prime p) : (p - 1)! % p = p - 1 := sorry

end wilson_theorem_l382_382038


namespace distinct_integer_pairs_l382_382595

theorem distinct_integer_pairs :
  ∃ pairs : (Nat × Nat) → Prop,
  (∀ x y : Nat, pairs (x, y) → 0 < x ∧ x < y ∧ (8 * Real.sqrt 31 = Real.sqrt x + Real.sqrt y))
  ∧ (∃! p, pairs p) → (∃! q, pairs q) → (∃! r, pairs r) → true := sorry

end distinct_integer_pairs_l382_382595


namespace kim_fraction_of_shirts_given_l382_382360

open Nat

theorem kim_fraction_of_shirts_given (d : ℕ) (s_left : ℕ) (one_dozen := 12) 
  (original_shirts := 4 * one_dozen) 
  (given_shirts := original_shirts - s_left) 
  (fraction_given := given_shirts / original_shirts) 
  (hc1 : d = one_dozen) 
  (hc2 : s_left = 32) 
  : fraction_given = 1 / 3 := 
by 
  sorry

end kim_fraction_of_shirts_given_l382_382360


namespace base_of_exponential_function_l382_382331

theorem base_of_exponential_function (a : ℝ) (h : ∀ x : ℝ, y = a^x) :
  (a > 1 ∧ (a - 1 / a = 1)) ∨ (0 < a ∧ a < 1 ∧ (1 / a - a = 1)) → 
  a = (1 + Real.sqrt 5) / 2 ∨ a = (Real.sqrt 5 - 1) / 2 :=
by sorry

end base_of_exponential_function_l382_382331


namespace solution_set_linear_inequalities_l382_382925

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382925


namespace solution_set_of_linear_inequalities_l382_382865

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382865


namespace difference_of_perimeters_l382_382200

noncomputable def rectangle_length := 6
noncomputable def rectangle_width := 2

structure Point :=
(x : ℝ) (y : ℝ)

structure Rectangle :=
(A B C D E : Point)
(length : ℝ) (width : ℝ)

-- Define the rectangle and its properties
def ABCD : Rectangle :=
{ A := ⟨0, 2⟩,
  B := ⟨6, 2⟩,
  C := ⟨6, 0⟩,
  D := ⟨0, 0⟩,
  E := ⟨6, 1⟩,
  length := rectangle_length,
  width := rectangle_width }

-- Define areas
def area_triangle : ℝ :=  
  (ABCD.C.x - ABCD.A.x) * (ABCD.E.y - ABCD.A.y) / 2            

def area_trapezoid : ℝ :=
  (ABCD.length * ABCD.width) - area_triangle  

-- Define perimeters
def perimeter_trapezoid : ℝ := 
  ABCD.length + ABCD.width + (ABCD.E.y - ABCD.D.y) + (ABCD.B.x - ABCD.E.x)

def perimeter_triangle : ℝ := 
  (ABCD.E.y - ABCD.D.y) + (ABCD.C.x - ABCD.A.x) + sqrt ((ABCD.C.x - ABCD.E.x) * (ABCD.C.x - ABCD.E.x) + ABCD.E.y * ABCD.E.y)

-- Prove the required relationship
theorem difference_of_perimeters :
  (area_trapezoid = 3 * area_triangle) → (perimeter_trapezoid - perimeter_triangle = 6) :=
  by
    sorry  

end difference_of_perimeters_l382_382200


namespace count_primes_with_squares_in_range_l382_382662

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end count_primes_with_squares_in_range_l382_382662


namespace balls_in_boxes_l382_382446

theorem balls_in_boxes : 
  let balls := {1, 2, 3, 4} in
  let boxes := {1, 2, 3, 4} in
  let ways_to_place_balls := 144 in
  ∃ f : balls → option boxes, 
    (∀ b ∈ boxes, ∃! ball, f ball = some b) ∧ (ways_to_place_balls = 144) := 
sorry

end balls_in_boxes_l382_382446


namespace simplify_power_l382_382410

theorem simplify_power (z : ℂ) (h₁ : z = (1 + complex.I) / (1 - complex.I)) : z ^ 1002 = -1 :=
by 
  sorry

end simplify_power_l382_382410


namespace plot_length_is_approx_83_01_l382_382055

/-- Definitions for the problem conditions. -/

noncomputable def breadth (b : ℝ) : Prop := 
  30 * (b + 30) + 35 * (b + 30) + 26.5 * b + 26.5 * (b + 30) = 9000

/-- Proving that the length of the rectangular plot with given conditions is approximately 83.01 meters. -/

theorem plot_length_is_approx_83_01 : 
  ∃ b l : ℝ, breadth b ∧ l = b + 30 ∧ l ≈ 83.01 :=
sorry

end plot_length_is_approx_83_01_l382_382055


namespace smallest_natural_satisfying_conditions_l382_382579

def move_unit_to_front (n : ℕ) : ℕ :=
  let units := n % 10
  let rest := n / 10
  units * 10 ^ (Nat.log10 rest + 1) + rest

theorem smallest_natural_satisfying_conditions :
  ∃ n : ℕ, (∃ x : ℕ, n = 10 * x + 6) ∧ (move_unit_to_front n = 4 * n) ∧ (∀ m : ℕ, (∃ y : ℕ, m = 10 * y + 6) ∧ (move_unit_to_front m = 4 * m) → n ≤ m) :=
sorry

end smallest_natural_satisfying_conditions_l382_382579


namespace prime_square_count_l382_382642

-- Define the set of natural numbers whose squares lie between 5000 and 9000
def within_square_range (n : ℕ) : Prop := 5000 < n * n ∧ n * n < 9000

-- Define the predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the set of prime numbers whose squares lie within the range
def primes_within_range : Finset ℕ := (Finset.filter is_prime (Finset.Ico 71 95))

-- The main statement: the number of prime numbers whose squares are between 5000 and 9000
theorem prime_square_count : (primes_within_range.filter within_square_range).card = 6 :=
sorry

end prime_square_count_l382_382642


namespace solve_inequalities_l382_382894

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382894


namespace profit_percentage_l382_382195

theorem profit_percentage (cost_price selling_price marked_price : ℝ)
  (h1 : cost_price = 47.50)
  (h2 : selling_price = 0.90 * marked_price)
  (h3 : selling_price = 65.97) :
  ((selling_price - cost_price) / cost_price) * 100 = 38.88 := 
by
  sorry

end profit_percentage_l382_382195


namespace biking_race_difference_l382_382343

theorem biking_race_difference
  (chris_speed : ℕ)
  (dana_speed : ℕ)
  (time_hours : ℕ)
  (chris_distance : chris_speed * time_hours)
  (dana_distance : dana_speed * time_hours)
  : chris_speed = 17 → dana_speed = 12 → time_hours = 6 → (chris_distance - dana_distance) = 30 := 
by
  intros
  sorry

end biking_race_difference_l382_382343


namespace normal_prob_interval_l382_382264

variable (ξ : ℝ → ℝ) (μ : ℝ) (σ : ℝ)

def normal_distribution (μ σ : ℝ) := 
  ∀ x, ξ x = 1/(σ * sqrt (2 * π)) * exp (-(x - μ)^2/(2 * σ^2))

noncomputable def event_prob (a b : ℝ) := 
  Classical.P (λ x, a < ξ x ∧ ξ x < b)

theorem normal_prob_interval (h1 : normal_distribution 1 σ) 
  (h2 : event_prob ξ 0 1 = 0.4) :
  event_prob ξ 0 2 = 0.8 :=
sorry

end normal_prob_interval_l382_382264


namespace problem_l382_382271

open Classical

noncomputable theory

variables {α : Type*} [LinearOrder α] [Add α] [One α] [HasLe α] [HasLt α]

theorem problem (a b c d e : α) (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) 
  (h₅ : a^2 + b^2 + c^2 + d^2 + e^2 = ab + ac + ad + ae + bc + bd + be + cd + ce + de) : 
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y ≤ z) ∧ 
  (∃ t₁ t₂ t₃ t₄ t₅ t₆, (t₁ = (a, b, d)) ∨ (t₁ = (a, b, e)) ∨ (t₁ = (a, c, d)) ∨ 
  (t₁ = (a, c, e)) ∨ (t₁ = (b, c, d)) ∨ (t₁ = (b, c, e)) ∨
  (t₂ = (a, c, e)) ∨ (t₂ = (a, d, e)) ∨ (t₂ = (b, c, e)) ∨ (t₂ = (b, d, e)) ∨ 
  (t₂ = (c, d, e)) ∨
  (t₃ = (a, b, d)) ∨ (t₃ = (a, b, e)) ∨ (t₃ = (a, c, e)) ∨ (t₃ = (a, d, e)) ∨ 
  (t₃ = (b, c, e)) ∨ (t₃ = (b, d, e)) ∨
  (t₄ = (b, c, e)) ∨ (t₄ = (b, d, e)) ∨ (t₄ = (c, d, e)) ∨
  (t₅ = (a, b, d)) ∨ (t₅ = (a, b, e)) ∨ (t₅ = (a, c, d)) ∨ (t₅ = (a, c, e)) ∨ 
  (t₅ = (b, c, d)) ∨ (t₆ = (b, c, e))) sorry

end problem_l382_382271


namespace minimize_T_n_l382_382618

def arithmetic_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def a_5 := 15 : ℤ
def a_10 := -10 : ℤ

theorem minimize_T_n :
  ∀ (a_n : ℕ → ℤ), arithmetic_seq a_n →
  a_n 5 = a_5 → a_n 10 = a_10 →
  ∃ n : ℕ, n = 5 ∨ n = 6 :=
by
  sorry

end minimize_T_n_l382_382618


namespace inclination_angle_of_line_l382_382302

-- Definition of the point-slope form condition
def point_slope_form_condition (x y : ℝ) : Prop :=
  y - 2 = -√3 * (x - 1)

-- Statement of the theorem to prove the inclination angle
theorem inclination_angle_of_line : ∃ (α : ℝ), α ∈ set.Icc 0 Real.pi ∧ (∀ (x y : ℝ), point_slope_form_condition x y → ∃ (k : ℝ), k = -√3 ∧ k = Real.tan α) ∧ α = Real.pi / 3 :=
sorry

end inclination_angle_of_line_l382_382302


namespace bill_amount_is_correct_l382_382077

-- Define the given conditions
def true_discount : ℝ := 189
def rate : ℝ := 0.16
def time : ℝ := 9 / 12

-- Define the true discount formula
def true_discount_formula (FV : ℝ) (R : ℝ) (T : ℝ) : ℝ := 
  (FV * R * T) / (100 + (R * T))

-- State that we want to prove that the Face Value is Rs. 1764
theorem bill_amount_is_correct : ∃ (FV : ℝ), FV = 1764 ∧ true_discount = true_discount_formula FV rate time :=
sorry

end bill_amount_is_correct_l382_382077


namespace simplify_cot_15_add_tan_45_l382_382042

theorem simplify_cot_15_add_tan_45 :
  (Real.cot 15 + Real.tan 45) = Real.csc 15 := by
  sorry

end simplify_cot_15_add_tan_45_l382_382042


namespace solution_set_linear_inequalities_l382_382928

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382928


namespace pineapples_bought_l382_382475

-- Define the costs and conditions
def pineapple_cost := 1.25
def shipping_cost := 21.00
def total_cost_per_pineapple := 3.00

-- Define what we need to prove
theorem pineapples_bought : 
  ∀ P : ℕ, 
  (pineapple_cost * P + shipping_cost = total_cost_per_pineapple * P) → 
  P = 12 := 
by
  intro P h
  -- The proof steps will follow but are omitted
  -- Instead we replace it with sorry for now
  sorry

end pineapples_bought_l382_382475


namespace solution_set_linear_inequalities_l382_382932

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382932


namespace solution_set_linear_inequalities_l382_382924

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382924


namespace centroid_locus_hexagon_l382_382112

open Real EuclideanGeometry

def is_locus_of_centroid {A B C : Point} (hA : Tech A) (hB : Tech B) (hC : Tech C) : 
  Prop :=
  ∃ (A' B' C' : Point), 
    Line_on A' B C ∧ Line_on B' C A ∧ Line_on C' A B ∧ 
    centroid_triangle (triangle A' B' C') (hexagon_in_triangle (triangle A B C))

theorem centroid_locus_hexagon (ABC : triangle) : 
  is_locus_of_centroid ABC := 
sorry

end centroid_locus_hexagon_l382_382112


namespace solution_set_inequalities_l382_382834

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382834


namespace parallelogram_area_l382_382489

variables (p q : ℝ^3)
variables a b : ℝ^3
variables θ : ℝ

-- Define the vectors a and b
def a := 7 * p - 2 * q
def b := p + 3 * q

-- Define the magnitudes of p and q
def norm_p := ∥p∥ = 1 / 2
def norm_q := ∥q∥ = 2

-- Define the angle θ between p and q being π / 2
def angle_p_q := θ = Real.pi / 2

-- The main theorem
theorem parallelogram_area : 
  (a = 7 * p - 2 * q) →
  (b = p + 3 * q) →
  (∥p∥ = 1 / 2) →
  (∥q∥ = 2) →
  (↑(Real.sin (Real.pi / 2)) = 1) →
  (θ = Real.pi / 2) →
  abs (∥a ∥ × ∥b∥) = 23 := by
  sorry

end parallelogram_area_l382_382489


namespace order_of_numbers_l382_382321

theorem order_of_numbers (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : 
  -m > n ∧ n > -n ∧ -n > m := 
by
  sorry

end order_of_numbers_l382_382321


namespace perimeter_of_triangle_DEF_l382_382339

open EuclideanGeometry

theorem perimeter_of_triangle_DEF (D E F U V W X : Point)
  (triangle_DEF : Triangle D E F)
  (right_angle_F : angle E F D = π / 2)
  (DE_length : distance D E = 15)
  (square_DEUV : is_square D E U V)
  (square_DFWX : is_square D F W X)
  (points_on_circle : concyclic U V W X) :
  perimeter triangle_DEF = 15 + 15 * real.sqrt 2 := 
  sorry

end perimeter_of_triangle_DEF_l382_382339


namespace find_volume_of_tank_l382_382184

theorem find_volume_of_tank
  (x : ℝ)
  (h1 : 0) -- Ensure x is a non-negative volume (Though not explicitly stated in conditions, this is an implicit assumption)
  (h2 : 0.20 * x / (3 / 4 * x + 15) = 1 / 3) :
  x = 100 :=
sorry

end find_volume_of_tank_l382_382184


namespace calculate_radius_of_circumscribed_sphere_l382_382286

noncomputable def radius_of_circumscribed_sphere
  (volume : ℝ)
  (condition_oa_ob_oc_zero : V3.AddVecs oa ob oc = V3.ZeroVec)
  (circum_center : Point)
  : ℝ :=
  sorry

theorem calculate_radius_of_circumscribed_sphere
  (volume_eq : volume = 1 / 12)
  (center_condition : V3.AddVecs oa ob oc = V3.ZeroVec)
  (circum_center : center_condition)
  :
  radius_of_circumscribed_sphere 1 / 12 (V3.AddVecs oa ob oc = V3.ZeroVec) circum_center = real.sqrt 3 / 3 :=
sorry

end calculate_radius_of_circumscribed_sphere_l382_382286


namespace homothety_point_exists_l382_382995

theorem homothety_point_exists
  (Q₁ O₂ : Point)
  (ω₁ ω₂ : Circle)
  (h_inner_outer : Inscribed ω₁ ω₂)
  (A₁ B₁ A₂ B₂ : Point)
  (diametrically_opposite_ω₁ : DiametricallyOpposite A₁ B₁ ω₁)
  (homothety_points : HomothetyPoints A₁ A₂ B₁ B₂) :
  ∃ O : Point, IsHomothetyCenter O ω₁ ω₂ :=
by
  sorry

end homothety_point_exists_l382_382995


namespace solution_set_inequalities_l382_382827

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382827


namespace solve_inequalities_l382_382900

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382900


namespace second_part_lent_years_l382_382183

theorem second_part_lent_years 
  (P1 P2 T : ℝ)
  (h1 : P1 + P2 = 2743)
  (h2 : P2 = 1688)
  (h3 : P1 * 0.03 * 8 = P2 * 0.05 * T) 
  : T = 3 :=
sorry

end second_part_lent_years_l382_382183


namespace ratio_is_4_to_1_l382_382193

noncomputable def ratio_coach_class_to_first_class 
    (total_seats : ℕ) 
    (coach_seats : ℕ) 
    (condition : coach_seats = 2 + some_multiple * first_class_seats)
    (total_condition : total_seats = coach_seats + first_class_seats) : ℕ × ℕ :=
    let coach_seats_minus_2 := coach_seats - 2;
    let first_class_seats := total_seats - coach_seats;
    let g := Nat.gcd coach_seats_minus_2 first_class_seats;
    (coach_seats_minus_2 / g, first_class_seats / g)

theorem ratio_is_4_to_1 : 
    ∀ (total_seats coach_seats first_class_seats : ℕ) 
      (some_multiple : ℕ),
    total_seats = 387 →
    coach_seats = 310 →
    coach_seats = 2 + some_multiple * first_class_seats →
    total_seats = coach_seats + first_class_seats →
    ratio_coach_class_to_first_class total_seats coach_seats _ _ = (4, 1) :=
by
    sorry

end ratio_is_4_to_1_l382_382193


namespace minimum_value_expression_l382_382318

theorem minimum_value_expression (a : ℝ) (h : a > 0) : 
  a + (a + 4) / a ≥ 5 :=
sorry

end minimum_value_expression_l382_382318


namespace solution_set_linear_inequalities_l382_382922

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382922


namespace rafael_hourly_wage_l382_382033

theorem rafael_hourly_wage
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (remaining_hours : ℕ)
  (total_payment : ℕ)
  (h_monday : monday_hours = 10)
  (h_tuesday : tuesday_hours = 8)
  (h_remaining : remaining_hours = 20)
  (h_payment : total_payment = 760) :
  total_payment / (monday_hours + tuesday_hours + remaining_hours) = 20 := 
by
  rw [h_monday, h_tuesday, h_remaining, h_payment]
  norm_num
  sorry

end rafael_hourly_wage_l382_382033


namespace inverse_fraction_coeff_l382_382805

noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

noncomputable def g_inv (x : ℝ) : ℝ := (4 * x + 2) / (3 - x)

theorem inverse_fraction_coeff : ∃ a b c d : ℝ, g_inv = λ x, (a * x + b) / (c * x + d) ∧ a / c = -4 :=
by
  use [4, 2, -1, 3]
  split
  { funext
    intro x
    calc
    g_inv x = (4 * x + 2) / (3 - x) : rfl
          ... = (4 * x + 2) / (3 + (-x)) : by rw [←sub_eq_add_neg]
          ... = (4 * x + 2) / ((-x) + 3) : by rw [add_comm (-x) 3]},
  { norm_num }

end inverse_fraction_coeff_l382_382805


namespace polar_equation_of_line_l_length_of_segment_AB_l382_382696

-- Definitions from conditions
def parametric_line (t : ℝ) := (1 - t, t)
def C1 (θ : ℝ) := (√3 * Real.cos θ, θ)
def C2 (θ : ℝ) := (3 * Real.sin θ, θ)

-- Mathematical proof problems
theorem polar_equation_of_line_l :
  ∀ (ρ θ : ℝ), (ρ * (Real.cos θ + Real.sin θ) = 1) :=
sorry

theorem length_of_segment_AB :
  ∀ (θ ρA ρB : ℝ),
  θ = π / 6 → ρA = 3 / 2 → ρB = √3 - 1 →
  |ρA - ρB| = (5 / 2) - √3 :=
sorry

end polar_equation_of_line_l_length_of_segment_AB_l382_382696


namespace solution_set_system_of_inequalities_l382_382947

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382947


namespace infinite_sqrt_sol_l382_382429

noncomputable def infinite_sqrt_expr : ℝ := 
  sqrt (2 + sqrt (2 + sqrt (2 + sqrt (2 + sqrt (2 + sqrt (2 + ...))))))

theorem infinite_sqrt_sol : infinite_sqrt_expr = 2 := 
by 
  sorry

end infinite_sqrt_sol_l382_382429


namespace hyperbola_eccentricity_range_l382_382679

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : ∃ (x y : ℝ), y = 2 * x ∧ (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)) :
  sqrt(1 + (b / a) ^ 2) > sqrt 5 :=
sorry

end hyperbola_eccentricity_range_l382_382679


namespace chebyshev_inequality_l382_382493

theorem chebyshev_inequality (n : ℕ) 
  (a b : Fin n → ℝ) 
  (ha : ∀ i j : Fin n, i ≤ j → a i ≤ a j)
  (hb : ∀ i j : Fin n, i ≤ j → b i ≤ b j) : 
  n * ∑ k, a k * b k ≥ (∑ k, a k) * (∑ k, b k) ∧ (∑ k, a k) * (∑ k, b k) ≥ n * ∑ k, a k * b (Fin.ofNat (n - 1 - k)) :=
by {
  sorry
}

end chebyshev_inequality_l382_382493


namespace solution_set_inequality_l382_382269

variable {R : Type*} [LinearOrderedField R]

/-- An auxiliary function f which is even, increasing in nonnegative and f(1) = 0-/
variable (f : R → R)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
variable (hf_value : f 1 = 0)

theorem solution_set_inequality (x : R) : f (x - 2) ≥ 0 ↔ x ≤ 1 ∨ x ≥ 3 :=
by
  sorry

end solution_set_inequality_l382_382269


namespace solution_set_linear_inequalities_l382_382919

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382919


namespace number_of_children_l382_382509

def weekly_husband : ℕ := 335
def weekly_wife : ℕ := 225
def weeks_in_six_months : ℕ := 24
def amount_per_child : ℕ := 1680

theorem number_of_children : (weekly_husband + weekly_wife) * weeks_in_six_months / 2 / amount_per_child = 4 := by
  sorry

end number_of_children_l382_382509


namespace area_inequality_l382_382046

variables {A B C D O M N : Type} [point : Point A] [point : Point B] [point : Point C] [point : Point D] [point : Point O] [point : Point M] [point : Point N]

-- Definitions of segments and areas
noncomputable def area (X Y Z : Type) [Point X] [Point Y] [Point Z] : ℝ := sorry
def isConvexQuadrilateral (A B C D : Type) [Point A] [Point B] [Point C] [Point D] : Prop := sorry

axioms 
  (h1 : isConvexQuadrilateral A B C D)
  (h2 : intersection AC BD O)
  (h3 : line_through O intersect_at M AB)
  (h4 : line_through O intersect_at N CD)
  (h5 : area O M B > area O N D)
  (h6 : area O C N > area O A M)

theorem area_inequality :
  area O A M + area O B C + area O N D > area O D A + area O M B + area O C N :=
begin
  sorry
end

end area_inequality_l382_382046


namespace fedya_initial_deposit_l382_382018

theorem fedya_initial_deposit (n k : ℕ) (h₁ : k < 30) (h₂ : n * (100 - k) = 84700) : 
  n = 1100 :=
by
  sorry

end fedya_initial_deposit_l382_382018


namespace minimize_quadratic_sum_l382_382725

theorem minimize_quadratic_sum (a b : ℝ) : 
  ∃ x : ℝ, y = (x-a)^2 + (x-b)^2 ∧ (∀ x', (x'-a)^2 + (x'-b)^2 ≥ y) ∧ x = (a + b) / 2 := 
sorry

end minimize_quadratic_sum_l382_382725


namespace solution_set_of_inequalities_l382_382970

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382970


namespace solve_logProblem_l382_382317

noncomputable def logProblem : Prop :=
  ∀ (x : ℝ), (log 7 (x + 6) = 2) → (log 13 (3 * x) ≈ 1.74)

theorem solve_logProblem : logProblem := by
  sorry

end solve_logProblem_l382_382317


namespace solution_set_l382_382850

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382850


namespace solution_set_of_inequalities_l382_382971

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382971


namespace binomial_coefficient_x_squared_l382_382050

theorem binomial_coefficient_x_squared (n : ℕ) (x : ℝ) (i : ℂ) (h : i^2 = -1) (h_coeff : binomial_coefficient n 2 * (-1)^2 * i^2 = -28) : n = 8 :=
by sorry

-- Definitions required for the statement
def binomial_coefficient (n k : ℕ) : ℤ := n.choose k

end binomial_coefficient_x_squared_l382_382050


namespace second_hand_travel_distance_l382_382822

theorem second_hand_travel_distance (r : ℝ) (minutes : ℝ) (π : ℝ) (h : r = 10 ∧ minutes = 45 ∧ π = Real.pi) : 
  (minutes / 60) * 60 * (2 * π * r) = 900 * π := 
by sorry

end second_hand_travel_distance_l382_382822


namespace length_of_chord_tangent_to_smaller_circle_l382_382789

-- Given the area of the ring between two concentric circles is 20π square inches
-- We need to prove the length of a chord of the larger circle that is tangent to the smaller circle is 4√5

theorem length_of_chord_tangent_to_smaller_circle 
  (a b : ℝ) -- radii of the larger and smaller circles
  (h : a^2 - b^2 = 20) : 
  ∃ (c : ℝ), c = 4 * Real.sqrt 5 ∧ (c / 2)^2 + b^2 = a^2 :=
by 
  use 4 * Real.sqrt 5
  split
  . rfl
  . sorry

end length_of_chord_tangent_to_smaller_circle_l382_382789


namespace vector_magnitude_sum_l382_382309

open Real

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem vector_magnitude_sum (h1 : ∥a∥ = 1) (h2 : ∥b∥ = sqrt 3) (h3 : a ⬝ (a - 2 • b) = -2) :
  ∥a + b∥ = sqrt 7 :=
sorry

end vector_magnitude_sum_l382_382309


namespace ratio_of_middle_to_tallest_tree_l382_382085

theorem ratio_of_middle_to_tallest_tree :
  (h_tallest : ℝ) (h_shortest : ℝ) (height_ratio : ℝ)
  (ht_condition : h_tallest = 150)
  (hs_condition : h_shortest = 50)
  (shortest_tree_condition : h_shortest = height_ratio / 2) :
  (middle_tree_ratio : (2 / 3) = height_ratio / h_tallest) :=
by
  sorry

end ratio_of_middle_to_tallest_tree_l382_382085


namespace number_of_white_dandelions_on_saturday_l382_382145

variables (D : Type) [decidable_eq D]

-- Define the states of dandelions
inductive DandelionState | Yellow | White | Dispersed

-- Define the life cycle of a dandelion
def life_cycle (d : D) (day : ℕ) : DandelionState :=
if day < 3 then DandelionState.Yellow
else if day = 3 then DandelionState.White
else DandelionState.Dispersed

-- Initial conditions on Monday
def yellow_on_monday : ℕ := 20
def white_on_monday : ℕ := 14

-- Initial conditions on Wednesday
def yellow_on_wednesday : ℕ := 15
def white_on_wednesday : ℕ := 11

-- Theorem stating the number of white dandelions on Saturday
theorem number_of_white_dandelions_on_saturday :
  let total_on_wednesday := yellow_on_wednesday + white_on_wednesday,
      new_dandelions_on_tuesday_wednesday := total_on_wednesday - yellow_on_monday,
      white_dandelions_on_saturday := new_dandelions_on_tuesday_wednesday
  in white_dandelions_on_saturday = 6 :=
sorry

end number_of_white_dandelions_on_saturday_l382_382145


namespace max_Xs_without_three_in_a_row_or_column_l382_382597

-- Define the 5x5 grid using a type alias for easier management
def Grid := Fin 5 × Fin 5

-- Define the placement of X's in the Grid
def is_valid_placement (placements : Finset Grid) : Prop :=
  (placements.card ≤ 25) ∧  -- No more than one X per square
  (∀ i : Fin 5, (placements.filter (λ (p : Grid), p.1 = i)).card < 3) ∧  -- No three X's in any row
  (∀ j : Fin 5, (placements.filter (λ (p : Grid), p.2 = j)).card < 3)    -- No three X's in any column

-- Prove that the maximum valid placements is 10
theorem max_Xs_without_three_in_a_row_or_column (placements : Finset Grid) :
  is_valid_placement placements → placements.card ≤ 10 :=
sorry

end max_Xs_without_three_in_a_row_or_column_l382_382597


namespace terminal_side_of_minus_35_eq_325_l382_382543

noncomputable def terminal_side_angle (α : ℝ) := α - 360 * ⌊α / 360⌋

theorem terminal_side_of_minus_35_eq_325 : 
  ∃ β : ℝ, 0 ≤ β ∧ β < 360 ∧ terminal_side_angle (-35) = β := 
begin
  use 325,
  split,
  { norm_num },
  split,
  { norm_num },
  { rw [terminal_side_angle, real.floor_eq_iff], simp, norm_num }
end

end terminal_side_of_minus_35_eq_325_l382_382543


namespace simplify_sqrt_l382_382412

theorem simplify_sqrt (a b : ℝ) (hb : b > 0) : 
  Real.sqrt (20 * a^3 * b^2) = 2 * a * b * Real.sqrt (5 * a) :=
by
  sorry

end simplify_sqrt_l382_382412


namespace cube_root_of_neg_a_l382_382667

variable (a k : ℝ)
axiom cube_root_a (h : k = a^(1/3)) : (-a)^(1/3) = -k

-- The statement of the problem
theorem cube_root_of_neg_a (h : k = a^(1/3)) : (-a)^(1/3) = -k := by
  apply cube_root_a h
  sorry

end cube_root_of_neg_a_l382_382667


namespace max_value_N_l382_382724

section maximum_value_of_N

variables (n : ℕ) (a : Fin n → ℕ)

noncomputable def S_pi (pi : Equiv.Perm (Fin n)) : Set (Fin n) :=
  { i | (a i) % (pi i).val == 0 }

def distinct_sets : Set (Set (Fin n)) :=
  { S_pi a pi | pi : Equiv.Perm (Fin n) }

theorem max_value_N : 
  ∀ (a : Fin n → ℕ), ∃ (N : ℕ), N = 2^n - n ∧ (distinct_sets a).card = N :=
sorry

end maximum_value_of_N

end max_value_N_l382_382724


namespace matilda_initial_bars_l382_382745

theorem matilda_initial_bars (M : ℕ) 
  (shared_evenly : 5 * M = 20 * 2 / 5)
  (half_given_to_father : M / 2 * 5 = 10)
  (father_bars : 5 + 3 + 2 = 10) :
  M = 4 := 
by
  sorry

end matilda_initial_bars_l382_382745


namespace exists_growing_number_with_perfect_square_sum_of_digits_l382_382522

def is_growing_number (m : ℕ) : Prop :=
  let digits := m.digits 10
  list.all (list.zip digits (list.drop 1 digits)) (λ (p : ℕ × ℕ), p.fst ≥ p.snd)

def sum_of_digits_is_perfect_square (m : ℕ) : Prop :=
  let sum_digits := (m.digits 10).sum
  ∃ k : ℕ, sum_digits = k * k

theorem exists_growing_number_with_perfect_square_sum_of_digits (n : ℕ) :
  ∃ m : ℕ, is_growing_number m ∧ m.digits 10 = list.replicate n 9 ∧ sum_of_digits_is_perfect_square m :=
sorry

end exists_growing_number_with_perfect_square_sum_of_digits_l382_382522


namespace problem_1_problem_2_problem_3_l382_382304

open Nat

noncomputable def a_seq : ℕ → ℝ
| 0     := 4
| (n+1) := 4 - 4 / a_seq n

def a_seq_formula (n : ℕ) : ℝ :=
2 + 2 / (n + 1)

def b_seq (n : ℕ) : ℝ :=
(n + 1) * a_seq_formula n * (1 / 2)^(n + 1)

def S_n (n : ℕ) : ℝ :=
3 - (n + 3) / 2^n

theorem problem_1 (n : ℕ) : ∃ d, (a_seq n - 2) * d = 1 :=
sorry

theorem problem_2 (n : ℕ) : a_seq n = 2 + 2 / (n + 1) :=
sorry

theorem problem_3 (n : ℕ) : ∑ i in range (n + 1), b_seq i = S_n n :=
sorry

end problem_1_problem_2_problem_3_l382_382304


namespace assignment_methods_count_l382_382036

theorem assignment_methods_count :
  let men := 5
  let women := 4
  let reps := 4
  let factories := 4
  let men_range := 2 to reps
  let total_ways := ∑ (m in men_range), (nat.choose men m) * (nat.choose women (reps - m)) * (nat.factorial factories)
  men >= 2 ∧ women >= 1 ∧ total_ways = 2400 :=
by
  sorry

end assignment_methods_count_l382_382036


namespace sum_of_interior_angles_l382_382201

theorem sum_of_interior_angles (n : ℕ) (h : n ≥ 3) : 
  (∑ i in Finset.range n, 180) = (n - 2) * 180 :=
by
  sorry

end sum_of_interior_angles_l382_382201


namespace trig_expression_simplifies_to_zero_l382_382586

-- Define the given condition
def cot_theta_eq_three (θ : ℝ) : Prop := Real.cot θ = 3

-- State the problem as a Lean theorem
theorem trig_expression_simplifies_to_zero (θ : ℝ) (h : cot_theta_eq_three θ) :
  (1 - Real.sin θ) / (Real.cos θ) - (Real.cos θ) / (1 + Real.sin θ) = 0 := 
sorry

end trig_expression_simplifies_to_zero_l382_382586


namespace usual_time_is_25_l382_382487

-- Definitions 
variables {S T : ℝ} (h1 : S * T = 5 / 4 * S * (T - 5))

-- Theorem statement
theorem usual_time_is_25 (h : S * T = 5 / 4 * S * (T - 5)) : T = 25 :=
by 
-- Using the assumption h, we'll derive that T = 25
sorry

end usual_time_is_25_l382_382487


namespace evaluate_expression_l382_382568

theorem evaluate_expression :
  1 + (3 / (4 + (5 / (6 + (7 / 8))))) = 85 / 52 := 
by
  sorry

end evaluate_expression_l382_382568


namespace pool_width_50_l382_382047

noncomputable def pool_width (length depth capacity : ℝ) : ℝ :=
  capacity / (length * depth)

theorem pool_width_50 :
  let capacity_drained := 60 * 1000 in -- in cubic feet
  let total_capacity := capacity_drained / 0.8 in
  pool_width 150 10 total_capacity = 50 :=
by {
  -- Definitions for capacity_drained and total_capacity
  let capacity_drained := 60 * 1000,
  let total_capacity := capacity_drained / 0.8,
  -- Define width calculation
  let width := pool_width 150 10 total_capacity,
  calc
    width = total_capacity / (150 * 10) : by rfl
    ... = 75000 / 1500 : by sorry -- Specific calculations omitted for brevity
    ... = 50 : by linarith
}

end pool_width_50_l382_382047


namespace amount_leaked_during_repairs_l382_382537

theorem amount_leaked_during_repairs:
  let total_leaked := 6206
  let leaked_before_repairs := 2475
  total_leaked - leaked_before_repairs = 3731 :=
by
  sorry

end amount_leaked_during_repairs_l382_382537


namespace gwen_books_collection_l382_382311

theorem gwen_books_collection :
  let mystery_books := 8 * 6
  let picture_books := 5 * 4
  let science_books := 4 * 7
  let non_fiction_books := 3 * 5
  let lent_mystery_books := 2
  let lent_science_books := 3
  let borrowed_picture_books := 5
  mystery_books - lent_mystery_books + picture_books - borrowed_picture_books + borrowed_picture_books + science_books - lent_science_books + non_fiction_books = 106 := by
  sorry

end gwen_books_collection_l382_382311


namespace calculate_m_n_p_l382_382583

theorem calculate_m_n_p (f g : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : ∀ x, g x = -f (90 - x))
  (h3 : ∃ v, g v = f v) -- the graph of g contains the vertex of f
  (h4 : ∃ x1 x2 x3 x4, x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ f x1 = 0 ∧ f x2 = 0 ∧ g x3 = 0 ∧ g x4 = 0)
  (h5 : x3 - x2 = 180) :
  ∃ m n p : ℕ, (x4 - x1 = m + n * real.sqrt p) ∧ (p > 0) ∧ (∀ q : ℕ, q^2 ∣ p → q = 1) ∧ (m + n + p = 902) :=
sorry

end calculate_m_n_p_l382_382583


namespace negation_of_proposition_l382_382435

-- Define the original proposition and its negation
def original_proposition (x : ℝ) : Prop := x^2 - 3*x + 3 > 0
def negated_proposition (x : ℝ) : Prop := x^2 - 3*x + 3 ≤ 0

-- The theorem about the negation of the original proposition
theorem negation_of_proposition :
  ¬ (∀ x : ℝ, original_proposition x) ↔ ∃ x : ℝ, negated_proposition x :=
by
  sorry

end negation_of_proposition_l382_382435


namespace solution_set_l382_382846

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382846


namespace parabola_focus_p_value_l382_382695

theorem parabola_focus_p_value (p : ℝ) (h : focus (λ y, x^2 - 2 * p * y) = (0, 1)) : p = 2 := 
sorry

end parabola_focus_p_value_l382_382695


namespace determine_f_5_l382_382009

noncomputable def S := {x : ℝ // x ≠ 0}

axiom f : S → S

axiom functional_eq (x y : S) (h : (x : ℝ) + (y : ℝ) ≠ 0) : 
  (f x).val + (f y).val = (f ⟨(x : ℝ) * (y : ℝ) * (f (⟨(x : ℝ) + (y : ℝ), h⟩)).val, sorry⟩).val

theorem determine_f_5 : f ⟨5, sorry⟩ = ⟨1/5, sorry⟩ := 
sorry

end determine_f_5_l382_382009


namespace children_that_can_be_catered_l382_382512

/-
We define the mathematical entities based on the conditions given.
-/
def num_adults := 55
def num_children := 70
def meal_for_adults := 70
def meal_for_children := 90
def adults_fed := 21
def dietary_restriction_adults := 0.20 * num_adults
def dietary_restriction_children := 0.15 * num_children
def adult_to_child_ratio := 1.5

theorem children_that_can_be_catered :
  let remaining_adults := meal_for_adults - adults_fed in
  let remaining_children_equiv := remaining_adults * adult_to_child_ratio in
  let remaining_children := Int.floor remaining_children_equiv in
  let children_with_restriction := Int.floor dietary_restriction_children in
  remaining_children - children_with_restriction = 63 := by
sorry

end children_that_can_be_catered_l382_382512


namespace square_area_l382_382093

theorem square_area 
  (P Q R S : ℝ × ℝ)
  (hP : P = (1, 1))
  (hQ : Q = (-2, 3))
  (hR : R = (-1, 8))
  (hS : S = (2, 4))
  (side_eq : dist P Q = dist Q R ∧ dist Q R = dist R S ∧ dist R S = dist S P) :
  let side_len := dist P Q in
  side_len^2 = 13 := 
by 
  sorry

end square_area_l382_382093


namespace sara_total_spent_l382_382023

def ticket_cost : ℝ := 10.62
def num_tickets : ℕ := 2
def rent_cost : ℝ := 1.59
def buy_cost : ℝ := 13.95
def total_spent : ℝ := 36.78

theorem sara_total_spent : (num_tickets * ticket_cost) + rent_cost + buy_cost = total_spent := by
  sorry

end sara_total_spent_l382_382023


namespace part_length_proof_l382_382175

-- Define the scale length in feet and inches
def scale_length_ft : ℕ := 6
def scale_length_inch : ℕ := 8

-- Define the number of equal parts
def num_parts : ℕ := 4

-- Calculate total length in inches
def total_length_inch : ℕ := scale_length_ft * 12 + scale_length_inch

-- Calculate the length of each part in inches
def part_length_inch : ℕ := total_length_inch / num_parts

-- Prove that each part is 1 foot 8 inches long
theorem part_length_proof :
  part_length_inch = 1 * 12 + 8 :=
by
  sorry

end part_length_proof_l382_382175


namespace PQRS_is_parallelogram_l382_382491

theorem PQRS_is_parallelogram
  (O A B C D I P Q R S : Point)
  (circumcircle: Triangle → Circle)
  (cyclic_quad : CyclicQuadrilateral ABCD)
  (O_inside : InsideQuadrilateral O ABCD)
  (O_not_AC : ¬ OnLine O (Line.through A C))
  (intersection_I : DiagonalIntersection I ABCD)
  (circle_AOI : ∀ P Q, OnCircumcircle P (Triangle.mk A O I) ∧ OnCircumcircle Q (Triangle.mk A O I) → On AD P ∧ On AB Q)
  (circle_COI : ∀ R S, OnCircumcircle R (Triangle.mk C O I) ∧ OnCircumcircle S (Triangle.mk C O I) → On CB R ∧ On CD S)
  : Parallelogram PQRS := by
  sorry

end PQRS_is_parallelogram_l382_382491


namespace sam_received_87_l382_382772

def sam_total_money : Nat :=
  sorry

theorem sam_received_87 (spent left_over : Nat) (h1 : spent = 64) (h2 : left_over = 23) :
  sam_total_money = spent + left_over :=
by
  rw [h1, h2]
  sorry

example : sam_total_money = 64 + 23 :=
  sam_received_87 64 23 rfl rfl

end sam_received_87_l382_382772


namespace equal_elevation_locus_l382_382090

noncomputable def point_A : ℝ × ℝ := (-5, 0)
noncomputable def point_B : ℝ × ℝ := (5, 0)
noncomputable def height_A : ℝ := 5
noncomputable def height_B : ℝ := 3

theorem equal_elevation_locus :
  {P : ℝ × ℝ | ∃ x : ℝ, x = P.1 ∧ 0 = P.2 ∧
    (x + 5) * tan (atan (height_A / (x + 5))) = (5 - x) * tan (atan (height_B / (5 - x))))  = 
  {P : ℝ × ℝ | (P.1 - 85 / 8)^2 + P.2^2 = (75 / 8)^2} := 
by
  sorry

end equal_elevation_locus_l382_382090


namespace chris_reaches_andrea_in_52_minutes_l382_382199

theorem chris_reaches_andrea_in_52_minutes
  (distance_initial : ℝ)
  (rate_of_closure_initial : ℝ)
  (rate_decrease_per_minute : ℝ)
  (time_andrea_rides : ℝ)
  (andrea_speed_multiplier : ℝ)
  (total_time_until_andrea_stops : ℝ) :
  distance_initial = 24 →
  rate_of_closure_initial = 1.2 →
  rate_decrease_per_minute = 0.04 →
  time_andrea_rides = 4 →
  andrea_speed_multiplier = 2 →
  total_time_until_andrea_stops = 52 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end chris_reaches_andrea_in_52_minutes_l382_382199


namespace hyperbola_has_eccentricity_2_l382_382274

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (point_symmetric_condition : ∀ F1 F2 : ℝ×ℝ, F1 = (-sqrt (4*a^2), 0) ∧ F2 = (sqrt (4*a^2), 0) 
                           → (∃ M : ℝ×ℝ, M = (-sqrt (4*a^2), 0) ∧ sqrt ( (M.1 - F2.1)^2 + (M.2 - F2.2)^2 ) = sqrt (4*a^2))): 
  ℝ :=
  let F1 := (-sqrt (4*a^2), 0) in
  let F2 := (sqrt (4*a^2), 0) in
  if point_symmetric_condition F1 F2 then 2 else sorry

theorem hyperbola_has_eccentricity_2 (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (point_symmetric_condition : ∀ F1 F2 : ℝ×ℝ, F1 = (-sqrt (4*a^2), 0) ∧ F2 = (sqrt (4*a^2), 0) 
                           → (∃ M : ℝ×ℝ, M = (-sqrt (4*a^2), 0) ∧ sqrt ( (M.1 - F2.1)^2 + (M.2 - F2.2)^2 ) = sqrt (4*a^2))):

  hyperbola_eccentricity a b h_a h_b point_symmetric_condition = 2 :=
by
  sorry

end hyperbola_has_eccentricity_2_l382_382274


namespace linear_inequalities_solution_l382_382876

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382876


namespace paul_correct_prediction_probability_l382_382250

-- Definitions
def probability_ghana_wins : ℚ := 2 / 3
def probability_bolivia_wins : ℚ := 1 / 3

-- Theorem statement
theorem paul_correct_prediction_probability :
  let probability_paul_picks_ghana_correctly := probability_ghana_wins * probability_ghana_wins in
  let probability_paul_picks_bolivia_correctly := probability_bolivia_wins * probability_bolivia_wins in
  probability_paul_picks_ghana_correctly + probability_paul_picks_bolivia_correctly = 5 / 9 :=
by
  sorry

end paul_correct_prediction_probability_l382_382250


namespace face_intersects_corresponding_line_l382_382707

variables {P : Type} [Point P] [ConvexPolyhedron P]
variables {l : ℕ → Type} [Line (l 1)] [Line (l 2)] ... [Line (l n)]
variables {f : Type} [Face f]
variables {A : ℕ → Point} [Intersection {A 1} {l 1} {f}] [Intersection {A 2} {l 2} {f}] ... [Intersection {A n} {l n} {f}]

theorem face_intersects_corresponding_line :
  ∃ (i : ℕ), 1 ≤ i ∧ i ≤ n ∧ (∃ (A : Point), Intersection A (l i) (f i)) :=
by sorry

end face_intersects_corresponding_line_l382_382707


namespace find_sum_l382_382803

noncomputable def f (x : ℝ) : ℝ := sorry

axiom symmetric_about_neg_three_fourths : ∀ x : ℝ, f(x) = -f(-x - 3/2)
axiom functional_equation : ∀ x : ℝ, f(x) = -1 / f(x + 3/2)
axiom f_minus_1 : f(-1) = 1
axiom f_0 : f(0) = -2

theorem find_sum : (f 1 + f 2 + ∑ i in Finset.range 2010, f (i + 2)) = 1 := sorry

end find_sum_l382_382803


namespace cone_height_l382_382519

noncomputable def R : ℝ := sorry -- Custom radius R of the sphere

def V_sphere (R : ℝ) : ℝ := (4.0 / 3.0) * Real.pi * R^3
def S_lat (r : ℝ) : ℝ := 3.0 * Real.pi * r^2
def lateral_surface_area (r l : ℝ) : ℝ := Real.pi * r * l
def pythagorean_theorem (r h l : ℝ) : Prop := l^2 = r^2 + h^2
def V_cone (r h : ℝ) : ℝ := (1.0 / 3.0) * Real.pi * r^2 * h

-- The proof statement
theorem cone_height (R : ℝ) (r h l : ℝ)
  (vol_sphere : V_sphere R = (4.0 / 3.0) * Real.pi * R^3)
  (lat_area : S_lat r = 3.0 * Real.pi * r^2)
  (lat_area_relation : lateral_surface_area r l = Real.pi * r * l)
  (pythagorean : pythagorean_theorem r h l)
  (vol_cone : V_cone r h = (1.0 / 3.0) * Real.pi * r^2 * h)
  (vol_equality : V_sphere R = V_cone r h) : h = 2 * R * Real.cbrt 4 :=
by
  sorry

end cone_height_l382_382519


namespace binary_div_rem_l382_382102

theorem binary_div_rem (n : ℕ) (h : n = 0b101110110101) : n % 8 = 5 := by
  rw [← h]
  -- Convert the binary number to decimal for clarity
  have : 0b101110110101 = 1 * 2^10 + 0 * 2^9 + 1 * 2^8 + 1 * 2^7 + 1 * 2^6 + 0 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 := by native.dec_trivial
  rw this
  -- Calculate the modulo with respect to 8
  native_decide
  sorry -- Omit the proof details

end binary_div_rem_l382_382102


namespace dandelion_white_dandelions_on_saturday_l382_382153

theorem dandelion_white_dandelions_on_saturday :
  ∀ (existsMondayYellow MondayWhite WednesdayYellow WednesdayWhite : ℕ)
    (MondayTotal WednesdayTotal : ℕ)
    (MondayYellow = 20)
    (MondayWhite = 14)
    (MondayTotal = MondayYellow + MondayWhite)
    (WednesdayYellow = 15)
    (WednesdayWhite = 11)
    (WednesdayTotal = WednesdayYellow + WednesdayWhite),
  existsMondayYellow = MondayYellow → existsMondayWhite = MondayWhite →
  WednesdayTotal = 26 →
  (WednesdayTotal - MondayYellow) = 6 →
  WednesdayTotal - MondayYellow - MondayWhite = 6 →
  6 = 6 := 
begin
  intros,
  sorry
end

end dandelion_white_dandelions_on_saturday_l382_382153


namespace median_BC_eq_l382_382983

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 5⟩
def B : Point := ⟨1, -2⟩
def C : Point := ⟨-7, 4⟩

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def D : Point := midpoint B C

theorem median_BC_eq : ∃ (a b c : ℝ), (a = 4) ∧ (b = -3) ∧ (c = 15) ∧ 
  (∀ (P : Point), (P ∈ line_through A D ↔ a * P.x + b * P.y + c = 0)) :=
by
  sorry

end median_BC_eq_l382_382983


namespace number_of_male_students_in_sample_l382_382687

theorem number_of_male_students_in_sample 
  (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (sample_female_students : ℕ) (sample_male_students : ℕ) :
  total_students = 680 →
  male_students = 360 →
  female_students = 320 →
  sample_female_students = 16 →
  (female_students * sample_male_students = male_students * sample_female_students) →
  sample_male_students = 18 :=
by
  intros h_total h_male h_female h_sample_female h_proportion
  sorry

end number_of_male_students_in_sample_l382_382687


namespace mary_friends_chicken_payment_l382_382737

theorem mary_friends_chicken_payment :
  let total_cost := 16
  let beef_cost_per_pound := 4
  let beef_pounds := 3
  let oil_cost := 1
  let chicken_pounds := 2
  let friends := 3
  let beef_cost := beef_cost_per_pound * beef_pounds
  let other_cost := beef_cost + oil_cost
  let chicken_cost := total_cost - other_cost
  let per_person_cost := chicken_cost / friends
  in per_person_cost = 1 := 
by
  sorry

end mary_friends_chicken_payment_l382_382737


namespace Lily_books_on_Wednesday_l382_382389

noncomputable def booksMike : ℕ := 45

noncomputable def booksCorey : ℕ := 2 * booksMike

noncomputable def booksMikeGivenToLily : ℕ := 13

noncomputable def booksCoreyGivenToLily : ℕ := booksMikeGivenToLily + 5

noncomputable def booksEmma : ℕ := 28

noncomputable def booksEmmaGivenToLily : ℕ := booksEmma / 4

noncomputable def totalBooksLilyGot : ℕ := booksMikeGivenToLily + booksCoreyGivenToLily + booksEmmaGivenToLily

theorem Lily_books_on_Wednesday : totalBooksLilyGot = 38 := by
  sorry

end Lily_books_on_Wednesday_l382_382389


namespace quadrangular_pyramid_edge_length_l382_382451

theorem quadrangular_pyramid_edge_length :
  ∃ e : ℝ, 8 * e = 14.8 ∧ e = 1.85 :=
  sorry

end quadrangular_pyramid_edge_length_l382_382451


namespace solve_inequalities_l382_382883

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382883


namespace solution_set_of_linear_inequalities_l382_382856

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382856


namespace maximum_intersections_l382_382382

theorem maximum_intersections (A : Point) (L : Fin 150 -> Line)
  (h_distinct : ∀ i j : Fin 150, i ≠ j → L i ≠ L j)
  (h_par : ∀ n : ℕ, n < 30 → is_parallel (L ⟨5 * (n + 1) - 1, by linarith⟩) (L ⟨5, by norm_num⟩))
  (h_through_A : ∀ n : ℕ, n < 30 → passes_through (L ⟨5 * (n + 1) - 4, by linarith⟩) A)
  (h_perp : ∀ n : ℕ, n < 30 → is_perpendicular (L ⟨5 * (n + 1) - 2, by linarith⟩) (L ⟨0, by norm_num⟩)) :
  maximum_intersections_of_pairs L = 9871 :=
by
  sorry

end maximum_intersections_l382_382382


namespace smallest_number_of_100_numbers_l382_382074

theorem smallest_number_of_100_numbers (a : Fin 100 → ℝ) (M t : ℝ) 
  (h₁ : ∑ i, a i = 1000)
  (h₂ : M = Finset.sup (Finset.univ.image a))
  (h₃ : t ∈ Finset.image a Finset.univ)
  (h₄ : ∑ i, if a i = M then 2 * M else if a i = t then t - 10 else a i = 1000) :
  ∀ i, a i = 10 :=
by
  sorry

end smallest_number_of_100_numbers_l382_382074


namespace triangles_rotation_proof_l382_382457

noncomputable def rotation_transformation (D E F D' E' F' : ℝ × ℝ) (n p q : ℝ) :=
  ∃ (p q : ℝ) (n : ℝ), 0 < n ∧ n < 180 ∧ n = 90 ∧
  ((p - q) = 28 ∧ (q + p) = 14 ∧ D' = (p - q, q + p)) ∧
  ((p + q) = 40 ∧ (q - p) = 14 ∧ E' = (p + q, q - p)) ∧
  ((p - q) = 28 ∧ (q - p) = 4 ∧ F' = (p - q, q - p))

theorem triangles_rotation_proof :
  ∃ p q n : ℝ, rotation_transformation (0, 0) (0, 10) (14, 0) (28, 14) (40, 14) (28, 4) n p q ∧ (n + p + q = 104) :=
by {
  use 21, use -7, use 90,
  split,
  { fsplit, linarith, linarith, ring,
    exact ⟨rfl, rfl, rfl⟩,
  },
  ring,
}

end triangles_rotation_proof_l382_382457


namespace complete_residue_system_mod_l382_382029

open Nat

theorem complete_residue_system_mod (m : ℕ) (x : Fin m → ℕ)
  (h : ∀ i j : Fin m, i ≠ j → ¬ ((x i) % m = (x j) % m)) :
  (Finset.image (λ i => x i % m) (Finset.univ : Finset (Fin m))) = Finset.range m :=
by
  -- Skipping the proof steps.
  sorry

end complete_residue_system_mod_l382_382029


namespace hannah_minimum_correct_answers_l382_382634

-- Define the conditions
def total_questions : ℕ := 50
def student1_score : ℝ := 47.5 / 50
def student2_score : ℝ := 37 / 50
def student3_score : ℝ := 46 / 50
def student4_score : ℝ := 46 / 50
def minimum_to_beat : ℝ := student1_score + 1 / 50

-- Define the highest score
def highest_score : ℝ := 
  max (max (max student1_score student2_score) student3_score) student4_score

-- Prove the required number of correct answers Hannah needs
theorem hannah_minimum_correct_answers : 
  ∀ total_questions student1_score student2_score student3_score student4_score, 
    (total_questions = 50) →
    (student1_score = 47.5 / 50) →
    (student2_score = 37 / 50) →
    (student3_score = 46 / 50) →
    (student4_score = 46 / 50) →
    (minimum_to_beat := student1_score + 1 / 50) → 
    ∀ correct_answers : ℕ, correct_answers = 49 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end hannah_minimum_correct_answers_l382_382634


namespace inequality_proof_l382_382365

theorem inequality_proof (a b c x y z : ℝ) (ha : 0 < a) (hb : a ≤ b) (hc : b ≤ c)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (ax + by + cz) * (x/a + y/b + z/c) ≤ ((x + y + z) ^ 2) * ((a + c) ^ 2) / (4 * a * c) :=
  sorry

end inequality_proof_l382_382365


namespace gear_R_rpm_l382_382208

noncomputable def gear_L_rpm : ℝ := 20
noncomputable def gear_R_revolutions_6sec (gear_L_revolutions_6sec : ℝ) : ℝ :=
  gear_L_revolutions_6sec + 6

noncomputable def gear_L_revolutions_6sec : ℝ :=
  gear_L_rpm * (6 / 60)

theorem gear_R_rpm (gear_L_rpm : ℝ) (gear_L_revolutions_6sec : ℝ) : ℝ :=
  (gear_R_revolutions_6sec gear_L_revolutions_6sec) * 10 = 80 :=
by
  unfold gear_R_revolutions_6sec
  unfold gear_L_revolutions_6sec
  sorry

end gear_R_rpm_l382_382208


namespace solution_set_of_inequalities_l382_382961

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382961


namespace sum_XY_is_seven_l382_382048

theorem sum_XY_is_seven (X Y : ℕ) : 
  (∃ n : ℕ, 16.factorial = 20922000896000 + n * 10^6 + X * 10^3 + Y * 10^2) →
  (38 + X + Y) % 9 = 0 →
  X + Y = 7 :=
by
  sorry

end sum_XY_is_seven_l382_382048


namespace remainder_3_pow_19_mod_10_l382_382488

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 :=
by
  sorry

end remainder_3_pow_19_mod_10_l382_382488


namespace second_hand_travel_distance_l382_382821

theorem second_hand_travel_distance (r : ℝ) (t : ℝ) : 
  r = 10 → t = 45 → 2 * t * π * r = 900 * π :=
by
  intro r_def t_def
  sorry

end second_hand_travel_distance_l382_382821


namespace rectangle_problem_l382_382693

theorem rectangle_problem
  (A B C D P Q S T R : Point)
  (h1 : rectangle A B C D)
  (h2 : P ∈ line B C)
  (h3 : ∠ A P D = 90)
  (h4 : orthogonal T S B C)
  (h5 : distance B P = 2 * distance P T)
  (h6 : line_intersection (line P D) (line T S) Q)
  (h7 : R ∈ line C D)
  (h8 : collinear R A Q)
  (h9 : distance P A = 24)
  (h10 : distance A Q = 18)
  (h11 : distance Q P = 30) : 
  distance S D = 0 :=
sorry

end rectangle_problem_l382_382693


namespace sara_total_spent_l382_382025

def cost_of_tickets := 2 * 10.62
def cost_of_renting := 1.59
def cost_of_buying := 13.95
def total_spent := cost_of_tickets + cost_of_renting + cost_of_buying

theorem sara_total_spent : total_spent = 36.78 := by
  sorry

end sara_total_spent_l382_382025


namespace find_d_over_a_l382_382782

variable (a b c d : ℚ)

-- Conditions
def condition1 : Prop := a / b = 8
def condition2 : Prop := c / b = 4
def condition3 : Prop := c / d = 2 / 3

-- Theorem statement
theorem find_d_over_a (h1 : condition1 a b) (h2 : condition2 c b) (h3 : condition3 c d) : d / a = 3 / 4 :=
by
  -- Proof is omitted
  sorry

end find_d_over_a_l382_382782


namespace angle_quadrant_l382_382615

theorem angle_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  0 < (π - α) ∧ (π - α) < π  :=
by
  sorry

end angle_quadrant_l382_382615


namespace solve_inequalities_l382_382907

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382907


namespace monica_book_ratio_theorem_l382_382385

/-
Given:
1. Monica read 16 books last year.
2. This year, she read some multiple of the number of books she read last year.
3. Next year, she will read 69 books.
4. Next year, she wants to read 5 more than twice the number of books she read this year.

Prove:
The ratio of the number of books she read this year to the number of books she read last year is 2.
-/

noncomputable def monica_book_ratio_proof : Prop :=
  let last_year_books := 16
  let next_year_books := 69
  ∃ (x : ℕ), (∃ (n : ℕ), x = last_year_books * n) ∧ (2 * x + 5 = next_year_books) ∧ (x / last_year_books = 2)

theorem monica_book_ratio_theorem : monica_book_ratio_proof :=
  by
    sorry

end monica_book_ratio_theorem_l382_382385


namespace prime_squares_5000_9000_l382_382652

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end prime_squares_5000_9000_l382_382652


namespace count_special_numbers_l382_382534

-- Define the set of digits and related constraints
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define the property of a number having alternating odd and even digits
def alternating (n : ℕ) : Prop :=
  (λ (i : ℕ), i % 2 = 0 → ∃ (d : ℕ), d ∈ digits ∧ n.digits.get i = some d ∧ d.even) ∧
  (λ (i : ℕ), i % 2 = 1 → ∃ (d : ℕ), d ∈ digits ∧ n.digits.get i = some d ∧ d.odd)

-- Define the property of a number being four digits long
def four_digits (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Define property of a number having no repeated digits
def no_repeat_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i ≠ j → n.digits.get i ≠ n.digits.get j

-- Define the property of a number being divisible by 5
def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Main statement of the proof
theorem count_special_numbers : 
  (∃ (f : ℕ → ℕ) (x : Fin 20),
  ∀ n, (four_digits n ∧ divisible_by_5 n ∧ no_repeat_digits n ∧ alternating n) ↔ (f n < 20)
  ) :=
  sorry

end count_special_numbers_l382_382534


namespace inequality_proof_equality_case_l382_382278

-- Defining that a, b, c are positive real numbers
variables (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The main theorem statement
theorem inequality_proof :
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 >= 6 * Real.sqrt 3 :=
sorry

-- Equality case
theorem equality_case :
  a = b ∧ b = c ∧ a = Real.sqrt 3^(1/4) →
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 = 6 * Real.sqrt 3 :=
sorry

end inequality_proof_equality_case_l382_382278


namespace find_constant_for_odd_function_l382_382242

theorem find_constant_for_odd_function (c : ℝ) :
  (∀ x : ℝ, -1/4 < x ∧ x < 1/4 → f x = arctan ((2 - 2 * x) / (1 + 4 * x)) + c) →
  (∀ x : ℝ, -1/4 < x ∧ x < 1/4 → f x = -f (-x)) →
  c = -arctan 2 := 
  sorry


end find_constant_for_odd_function_l382_382242


namespace equation_of_line_through_trisection_points_l382_382027

-- Definitions of points A, B
def pointA : (ℝ × ℝ) := (1, 2)
def pointB : (ℝ × ℝ) := (8, 3)

-- Trisection points calculation (given as constant for simplicity)
def trisectionPointC : (ℝ × ℝ) := (10 / 3, 7 / 3)
def trisectionPointD : (ℝ × ℝ) := (17 / 3, 8 / 3)

-- Define the point through which the line passes
def throughPoint : (ℝ × ℝ) := (2, 1)

-- Function to verify if a point lies on the line defined by the equation
def on_line (p : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

-- Main theorem statement
theorem equation_of_line_through_trisection_points :
  let line_eqn := (1, -1, 1) in
  on_line throughPoint line_eqn.1 line_eqn.2 line_eqn.3 ∧
  on_line trisectionPointC line_eqn.1 line_eqn.2 line_eqn.3 ∧
  on_line trisectionPointD line_eqn.1 line_eqn.2 line_eqn.3 :=
sorry

end equation_of_line_through_trisection_points_l382_382027


namespace solution_set_of_inequalities_l382_382973

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382973


namespace geese_among_non_herons_l382_382361

-- Define percentages as nonnegative real numbers between 0 and 1.
def geese : ℝ := 0.30
def swans : ℝ := 0.25
def herons : ℝ := 0.20
def ducks : ℝ := 0.25

theorem geese_among_non_herons : (geese / (1 - herons)) * 100 = 37.5 :=
by
  -- condition constraints
  have h_non_herons : 1 - herons = 0.80, by sorry
  have h_geese : geese = 0.30, by sorry
  -- proof
  sorry

end geese_among_non_herons_l382_382361


namespace area_of_triangle_PQR_l382_382994

theorem area_of_triangle_PQR
  (r R : ℝ)
  (h1 : r = 8)
  (h2 : R = 25)
  (cos_P cos_Q cos_R : ℝ)
  (h3 : 2 * cos_Q = cos_P + cos_R) :
  let A : ℝ := 96
  in A = 96 :=
by sorry

end area_of_triangle_PQR_l382_382994


namespace solve_inequalities_l382_382899

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382899


namespace sum_of_6_digit_numbers_without_0_9_divisible_by_37_l382_382760

theorem sum_of_6_digit_numbers_without_0_9_divisible_by_37 :
  ∀ (N : ℕ), (6 ≤ (nat.log 10 N + 1)) ∧ (nat.log 10 N + 1 ≤ 6)
             → (∀ d : ℕ, (d ∈ N.digits 10) → (d ≠ 0 → d ≠ 9))
             → (∃ k : ℕ, (sum (filter (λ d, d ≠ 0 ∧ d ≠ 9) (digits 10 N)) = k * 37)) :=
by
  -- proof here
  sorry

end sum_of_6_digit_numbers_without_0_9_divisible_by_37_l382_382760


namespace solve_inequalities_l382_382888

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382888


namespace max_CA_CB_range_of_k_l382_382490

-- Define the ellipse equation condition
def ellipse (x y : ℝ) : Prop := (x^2 / 8) + (y^2 / 4) = 1

-- Define the line equation condition
def line (k x y : ℝ) : Prop := y = k * x + 1

-- Define the points A and B conditions of intersection
def intersects (k x y : ℝ) : Prop := ellipse x y ∧ line k x y

-- Problem part 1: Maximum value of |CA| + |CB| given the conditions
theorem max_CA_CB (k : ℝ) :
  let A_x := classical.some (exists.intro A_x (intersects k A_x A_y))
  let A_y := k * A_x + 1
  let B_x := classical.some (exists.intro B_x (intersects k B_x B_y))
  let B_y := k * B_x + 1
  let C := (0, 0)
  let dist (x y : ℝ) := sqrt (x^2 + y^2)
  |dist C (A_x, A_y) + dist C (B_x, B_y)| ≤ 4 * sqrt 2 + 1       
:= sorry

-- Problem part 2: Range of values for k given the conditions
theorem range_of_k (k : ℝ) :
  let M := (-1 / k, 0)
  let N := (0, 1)
  let mid_pt := ((-1 / (2 * k)), 1 / 2)
  let radius := sqrt ((-1 / k) ^ 2 + 1)
  let circle_inside_ellipse := ∀ x y, line ((-1 / (2 * k)) - x) ((1 / 2) - y) → ellipse x y
  (k ≤ - (3 + 4 * sqrt 6)/29 ∨ k ≥ (3 + 4 * sqrt 6)/29)
:= sorry

end max_CA_CB_range_of_k_l382_382490


namespace calculate_p_q_r_sum_l382_382368

open Vector3

noncomputable theory

def a := Vector3.mk (2:ℝ) 0 0
def b := Vector3.mk 0 (3:ℝ) 0
def c := Vector3.mk 0 0 (4:ℝ)
def p (a b: Vector3 ℝ) := ⟪a, b⟫ 
def q := (1 / 6 : ℝ)
def r := 0
def are_orthogonal (v1 v2: Vector3 ℝ) : Prop := ⟪v1, v2⟫ = 0

theorem calculate_p_q_r_sum :
  are_orthogonal a b ∧ are_orthogonal b c ∧ are_orthogonal c a ∧
  ⟪a, a⟫ = 4 ∧ ⟪b, b⟫ = 9 ∧ ⟪c, c⟫ = 16 ∧
  ⟪a, b × c⟫ = 24 ∧ 
  a = p a b * (a ⨯ b) + q * (b ⨯ c) + r * (c ⨯ a)
  → p + q + r = (1/6 : ℝ) :=
  sorry

end calculate_p_q_r_sum_l382_382368


namespace linear_inequalities_solution_l382_382869

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382869


namespace value_of_g_at_neg2_l382_382470

def g (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_of_g_at_neg2 : g (-2) = 15 :=
by
  -- This is where the proof steps would go, but we'll skip it
  sorry

end value_of_g_at_neg2_l382_382470


namespace simplify_expression_l382_382777

variable {y : ℝ}
variable (hy : y ≠ 0)

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) : 
  (5 / (2 * y ^ (-4))) * (4 * y ^ 3 / 3) + (y / y ^ (-2)) = (10 * y ^ 7 + 3 * y ^ 3) / 3 :=
by
  sorry

end simplify_expression_l382_382777


namespace mary_friends_chicken_payment_l382_382736

theorem mary_friends_chicken_payment :
  let total_cost := 16
  let beef_cost_per_pound := 4
  let beef_pounds := 3
  let oil_cost := 1
  let chicken_pounds := 2
  let friends := 3
  let beef_cost := beef_cost_per_pound * beef_pounds
  let other_cost := beef_cost + oil_cost
  let chicken_cost := total_cost - other_cost
  let per_person_cost := chicken_cost / friends
  in per_person_cost = 1 := 
by
  sorry

end mary_friends_chicken_payment_l382_382736


namespace smallest_positive_integer_congruence_l382_382466

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 14 [MOD 31] ∧ 0 < x ∧ x < 31 := 
sorry

end smallest_positive_integer_congruence_l382_382466


namespace problem_inequality_l382_382008

theorem problem_inequality 
  (n : ℕ) 
  (hn: n ≥ 2) 
  (a: Fin n.succ → ℝ) 
  (ha: ∀ i j, i ≤ j → a i ≥ a j) 
  (hapos: ∀ i, 0 < a i) 
  :
  (∑ i in Finset.range n, a i / a (i + 1) - n) 
  ≤ (1 / (2 * a 0 * a n)) * ∑ i in Finset.range n, (a i - a (i + 1))^2 := 
by
   sorry

end problem_inequality_l382_382008


namespace total_loads_l382_382312

def shirts_per_load := 3
def sweaters_per_load := 2
def socks_per_load := 4

def white_shirts := 9
def colored_shirts := 12
def white_sweaters := 18
def colored_sweaters := 20
def white_socks := 16
def colored_socks := 24

def white_shirt_loads : ℕ := white_shirts / shirts_per_load
def white_sweater_loads : ℕ := white_sweaters / sweaters_per_load
def white_sock_loads : ℕ := white_socks / socks_per_load

def colored_shirt_loads : ℕ := colored_shirts / shirts_per_load
def colored_sweater_loads : ℕ := colored_sweaters / sweaters_per_load
def colored_sock_loads : ℕ := colored_socks / socks_per_load

def max_white_loads := max (max white_shirt_loads white_sweater_loads) white_sock_loads
def max_colored_loads := max (max colored_shirt_loads colored_sweater_loads) colored_sock_loads

theorem total_loads : max_white_loads + max_colored_loads = 19 := by
  sorry

end total_loads_l382_382312


namespace solution_set_of_linear_inequalities_l382_382860

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382860


namespace part1_part2_i_part2_ii_l382_382439

variable {c : ℝ} (cnonneg : c > 0)

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = a n + c * a n ^ 2

def geom_seq (a : ℕ → ℝ) : Prop :=
  (a 1) * 3 * (a 2 + c * a 2 ^ 2) = (2 * a 2) ^ 2

theorem part1 (a : ℕ → ℝ) (h1 : sequence a) (h2 : geom_seq a) :
  a 1 = (1 + real.sqrt 13) / (6 * c) :=
sorry

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := 1 / (1 + c * a n)
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, b a i

theorem part2_i (a : ℕ → ℝ) (h1 : sequence a) :
  ∀ n : ℕ, n > 0 → 
    (1 / a (n + 1)) - (1 / a n) = - c / (1 + c * a n) :=
sorry

theorem part2_ii (a : ℕ → ℝ) (h1 : sequence a) :
  ∀ n : ℕ, n > 0 →
    S a n < S a (n + 1) ∧ S a (n + 1) < 1 / (c * a 1) :=
sorry

end part1_part2_i_part2_ii_l382_382439


namespace cyclic_quadrilateral_perpendicular_EF_to_OM_l382_382599

-- Definitions based on conditions
def convex_quadrilateral (A B C D : Type) : Prop := sorry
def diagonals_intersect_at (A B C D M : Type) : Prop := sorry
def circumscribed_circle_center (A B C D O : Type) : Prop := sorry
def perpendicular_to_diameter_AC (M A C A1 C1 : Type) : Prop := sorry
def perpendicular_to_diameter_BD (M B D B1 D1 : Type) : Prop := sorry

-- Problem statements
theorem cyclic_quadrilateral 
  (A B C D M O A1 C1 B1 D1 : Type) 
  (h1 : convex_quadrilateral A B C D) 
  (h2 : diagonals_intersect_at A B C D M)
  (h3 : circumscribed_circle_center A B C D O)
  (h4 : perpendicular_to_diameter_AC M A C A1 C1) 
  (h5 : perpendicular_to_diameter_BD M B D B1 D1) :
  cyclic_quadrilateral A1 B1 C1 D1 := sorry

theorem perpendicular_EF_to_OM 
  (A B C D M O A1 C1 B1 D1 E F : Type) 
  (h1 : convex_quadrilateral A B C D) 
  (h2 : diagonals_intersect_at A B C D M)
  (h3 : circumscribed_circle_center A B C D O)
  (h4 : perpendicular_to_diameter_AC M A C A1 C1) 
  (h5 : perpendicular_to_diameter_BD M B D B1 D1) 
  (h6 : cyclic_quadrilateral A1 B1 C1 D1)
  (h7 : E F : circle_intersection_points N1 k) :
  perpendicular EF OM := sorry


end cyclic_quadrilateral_perpendicular_EF_to_OM_l382_382599


namespace find_value_of_a_l382_382002

def pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem find_value_of_a (a : ℝ) :
  pure_imaginary ((a^3 - a) + (a / (1 - a)) * Complex.I) ↔ a = -1 := 
sorry

end find_value_of_a_l382_382002


namespace solution_set_of_inequalities_l382_382958

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382958


namespace fraction_as_power_of_x_l382_382281

variable (x t m n p q r : ℝ)

/-- Conditions -/
variable (hx : x = 3)
variable (ha : x^t ≠ 0)
variable (hb : x^m ≠ 0)
variable (hc : x^n ≠ 0)
variable (h : (x^t)^p * (x^m)^q / (x^n)^r = Real.sqrt 243)

/-- Statement -/
theorem fraction_as_power_of_x :
  t*p + m*q - n*r = 2.5 :=
sorry

end fraction_as_power_of_x_l382_382281


namespace remainder_when_x_plus_2uy_div_y_l382_382101

theorem remainder_when_x_plus_2uy_div_y (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) :
  (x + 2 * u * y) % y = v := 
sorry

end remainder_when_x_plus_2uy_div_y_l382_382101


namespace prob_two_consecutive_jumps_prob_first_time_on_third_attempt_min_attempts_for_99_prob_l382_382513

noncomputable def log10(val : ℝ) : ℝ := Real.log10 val

variable (p : ℝ) (H : p = 0.8)

theorem prob_two_consecutive_jumps : p * p = 0.64 :=
by
  rw [H]
  norm_num

theorem prob_first_time_on_third_attempt : (1 - p) * (1 - p) * p = 0.032 :=
by
  rw [H]
  norm_num

theorem min_attempts_for_99_prob : ∃ n : ℕ, 1 - (1 - p)^n ≥ 0.99 ∧ n = 3 :=
by
  have n_val : ℕ := 3
  use n_val
  split
  · calc
    1 - (1 - p)^n_val = 1 - (1 - 0.8) ^ 3 : by rw [H]
    ... ≥ 0.99 : by norm_num
  · norm_num

end prob_two_consecutive_jumps_prob_first_time_on_third_attempt_min_attempts_for_99_prob_l382_382513


namespace shaded_region_area_l382_382440

noncomputable def area_of_quarter_circle (r : ℝ) : ℝ :=
  (1 / 4) * π * r^2

noncomputable def area_of_semicircle (r : ℝ) : ℝ :=
  (1 / 2) * π * r^2

noncomputable def shaded_area (big_radius : ℝ) (small_radius : ℝ) : ℝ :=
  area_of_quarter_circle big_radius - area_of_semicircle small_radius

theorem shaded_region_area :
  shaded_area 5 2 = (17 * π) / 4 :=
by
  unfold shaded_area
  unfold area_of_quarter_circle
  unfold area_of_semicircle
  -- sorry is used to skip the proof
  sorry

end shaded_region_area_l382_382440


namespace minimum_value_func1_minimum_value_func2_l382_382123

-- Problem (1): 
theorem minimum_value_func1 (x : ℝ) (h : x > -1) : 
  (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

-- Problem (2): 
theorem minimum_value_func2 (x : ℝ) (h : x > 1) : 
  (x^2 + 8) / (x - 1) ≥ 8 :=
sorry

end minimum_value_func1_minimum_value_func2_l382_382123


namespace impossible_labeling_l382_382354

theorem impossible_labeling :
  ¬ (∃ (f : ℤ × ℤ → ℕ), 
    (∀ (a b c : ℤ × ℤ), collinear a b c ↔ (∃ d > 1, d ∣ (f a) ∧ d ∣ (f b) ∧ d ∣ (f c)))) :=
sorry

end impossible_labeling_l382_382354


namespace linear_inequalities_solution_l382_382877

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382877


namespace find_a2_plus_b2_find_a_minus_b2_l382_382590

variable {a b : ℝ}

-- Definition of the conditions
def condition1 := a + b = 4
def condition2 := a * b = 1

-- Proof statement for the first question
theorem find_a2_plus_b2 (h1 : condition1) (h2 : condition2) : a^2 + b^2 = 14 := 
sorry

-- Proof statement for the second question
theorem find_a_minus_b2 (h1 : condition1) (h2 : condition2) : (a - b)^2 = 12 := 
sorry

end find_a2_plus_b2_find_a_minus_b2_l382_382590


namespace antifreeze_percentage_l382_382126

theorem antifreeze_percentage :
  ∀ (p1 p2 p_final : ℝ) (c1 c2 : ℝ),
    p1 = 4 → c1 = 0.05 →
    p2 = 8 → c2 = 0.20 →
    p_final = 12 →
    ((c1 * p1 + c2 * p2) / p_final) * 100 = 15 :=
by
  intros p1 p2 p_final c1 c2
  assume _ h_c1 _ h_c2 _ p_fin_eq
  rw [h_c1, h_c2]
  rw [p_fin_eq]
  suffices h_suff : (0.05 * 4 + 0.20 * 8) / 12 = 0.15
  · calc
      ((0.05 * 4 + 0.20 * 8) / 12) * 100 = 0.15 * 100 := by rw [h_suff]
                                            ... = 15 := by norm_num
  sorry

end antifreeze_percentage_l382_382126


namespace task_completion_l382_382122

   variable A B C : ℕ

   theorem task_completion (x : ℕ) (hxA : A = x) (hxB : B = x + 6) (hxC : C = x + 9)
     (h_work_eq : 3 * B * C + 4 * x * C = 9 * x * B) :
     A = 18 ∧ B = 24 ∧ C = 27 :=
   by
     sorry
   
end task_completion_l382_382122


namespace train_passes_platform_in_160_seconds_l382_382127

def train_length := 1200 -- length of the train in meters
def tree_crossing_time := 120 -- time to cross a tree in seconds
def platform_length := 400 -- length of the platform in meters

theorem train_passes_platform_in_160_seconds :
  let v := train_length / tree_crossing_time in -- speed of the train in m/s
  let total_distance := train_length + platform_length in -- total distance to cover
  let t := total_distance / v in -- time to cross platform
  t = 160 :=
by
  sorry

end train_passes_platform_in_160_seconds_l382_382127


namespace a_div_c_eq_neg4_l382_382811

-- Given conditions
def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)
def g_inv (x : ℝ) : ℝ := (4 * x + 2) / (3 - x)

-- Theorem to be proved
theorem a_div_c_eq_neg4 : let a := 4, c := -1 in a / c = -4 :=
by {
  exact rfl
}

end a_div_c_eq_neg4_l382_382811


namespace exists_four_commissions_with_one_common_deputy_l382_382118

theorem exists_four_commissions_with_one_common_deputy
  (deputies : Fin 100)
  (commissions : Fin 450)
  (h1 : ∀ C1 C2 : Fin 450, C1 ≠ C2 → (commissions C1 ∩ commissions C2).card ≤ 3)
  (h2 : ∀ S : Finset (Fin 450), S.card = 5 → 
        (∃! (d : Fin 100), ∀ C ∈ S, d ∈ commissions C)) :
  ∃ C1 C2 C3 C4 : Fin 450, (commissions C1 ∩ commissions C2 ∩ commissions C3 ∩ commissions C4).card = 1 :=
sorry

end exists_four_commissions_with_one_common_deputy_l382_382118


namespace max_F_value_s_value_l382_382292

noncomputable theory

def f (x : ℝ) : ℝ := (Real.log x) / x

def F (x : ℝ) : ℝ := x^2 - x * f x

def H (s : ℝ) (x : ℝ) : ℝ := if x ≥ s then x / (2 * Real.exp 1) else f x

theorem max_F_value : ∀ x ∈ Set.Icc (1/2 : ℝ) 2, F x ≤ 4 - Real.log 2 :=
by sorry

theorem s_value : ∀ s, (∀ k : ℝ, ∃ x₀ : ℝ, H s x₀ = k) ↔ s = Real.sqrt (Real.exp 1) :=
by sorry

end max_F_value_s_value_l382_382292


namespace sqrt_defined_iff_x_ge_1_l382_382065

theorem sqrt_defined_iff_x_ge_1 (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_defined_iff_x_ge_1_l382_382065


namespace cone_prism_volume_ratio_l382_382173

/--
Given:
- The base of the prism is a rectangle with side lengths 2r and 3r.
- The height of the prism is h.
- The base of the cone is a circle with radius r and height h.

Prove:
- The ratio of the volume of the cone to the volume of the prism is (π / 18).
-/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * Real.pi * r^2 * h) / (6 * r^2 * h) = Real.pi / 18 := by
  sorry

end cone_prism_volume_ratio_l382_382173


namespace simplify_power_l382_382409

theorem simplify_power (z : ℂ) (h₁ : z = (1 + complex.I) / (1 - complex.I)) : z ^ 1002 = -1 :=
by 
  sorry

end simplify_power_l382_382409


namespace inequality_log_range_of_a_l382_382495

open Real

theorem inequality_log (x : ℝ) (h₀ : 0 < x) : 
  1 - 1 / x ≤ log x ∧ log x ≤ x - 1 := sorry

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 < x ∧ x ≤ 1 → a * (1 - x^2) + x^2 * log x ≥ 0) : 
  a ≥ 1/2 := sorry

end inequality_log_range_of_a_l382_382495


namespace distinct_powers_of_two_l382_382759

theorem distinct_powers_of_two (n : ℕ) (h : n ≥ 1) : 
  ∃ (a : ℕ → ℕ) (k : ℕ), n = ∑ i in (Finset.range k), 2^(a i) ∧ Function.injective a :=
begin
  sorry
end

end distinct_powers_of_two_l382_382759


namespace total_trucks_l382_382390

theorem total_trucks {t : ℕ} (h1 : 2 * t + t = 300) : t = 100 := 
by sorry

end total_trucks_l382_382390


namespace inverse_fraction_coeff_l382_382806

noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

noncomputable def g_inv (x : ℝ) : ℝ := (4 * x + 2) / (3 - x)

theorem inverse_fraction_coeff : ∃ a b c d : ℝ, g_inv = λ x, (a * x + b) / (c * x + d) ∧ a / c = -4 :=
by
  use [4, 2, -1, 3]
  split
  { funext
    intro x
    calc
    g_inv x = (4 * x + 2) / (3 - x) : rfl
          ... = (4 * x + 2) / (3 + (-x)) : by rw [←sub_eq_add_neg]
          ... = (4 * x + 2) / ((-x) + 3) : by rw [add_comm (-x) 3]},
  { norm_num }

end inverse_fraction_coeff_l382_382806


namespace parabola_intersects_x_axis_at_two_points_find_m_given_AB_eq_6_l382_382301

variable (m : ℝ)
def parabola (x : ℝ) : ℝ := x^2 + 2*m*x - (5/4)*m^2

theorem parabola_intersects_x_axis_at_two_points
  (h : m > 0) :
  ∃ x1 x2 : ℝ, parabola m x1 = 0 ∧ parabola m x2 = 0 ∧ x1 ≠ x2 :=
by
  sorry

theorem find_m_given_AB_eq_6
  (h : m > 0)
  (h_AB : ∃ x1 x2 : ℝ, parabola m x1 = 0 ∧ parabola m x2 = 0 ∧ x2 - x1 = 6) :
  m = 2 :=
by
  sorry

end parabola_intersects_x_axis_at_two_points_find_m_given_AB_eq_6_l382_382301


namespace old_camera_model_cost_l382_382017

theorem old_camera_model_cost (C new_model_cost discounted_lens_cost : ℝ)
  (h1 : new_model_cost = 1.30 * C)
  (h2 : discounted_lens_cost = 200)
  (h3 : new_model_cost + discounted_lens_cost = 5400)
  : C = 4000 := by
sorry

end old_camera_model_cost_l382_382017


namespace min_mutually_visible_pairs_l382_382125

-- This definition captures the setup of the problem.
def birdsOnCircle (n m : ℕ) (bird_distribution : Fin m → ℕ) :=
  ∑ i, bird_distribution i = n

-- The given condition: Mutual visibility condition for birds at given points.
def mutuallyVisible (bird_distribution : Fin 35 → ℕ) :=
  ∀ i j, (i ≠ j → abs (i - j) ≤ 1 ∨ abs (i - j) ≥ 34) → 
    bird_distribution i * bird_distribution j = 0

-- The main statement asserting the number of mutually visible pairs.
theorem min_mutually_visible_pairs :
  ∃ bird_distribution : Fin 35 → ℕ,
  birdsOnCircle 155 35 bird_distribution ∧
  mutuallyVisible bird_distribution ∧
  (∑ i, bird_distribution i * (bird_distribution i - 1) / 2) = 270 :=
sorry

end min_mutually_visible_pairs_l382_382125


namespace number_of_new_species_imported_is_4_l382_382203

noncomputable def new_species_imported (x : ℕ) : Prop := 
  (5 + x) * 6 = 54

theorem number_of_new_species_imported_is_4 : new_species_imported 4 :=
by {
  sorry,
}

end number_of_new_species_imported_is_4_l382_382203


namespace option_C_is_not_a_correlation_relationship_l382_382107

-- Definitions
def is_correlation_relationship (x y : Type) [Valued x] [Valued y] : Prop := sorry--add the definition if necessary
def is_functional_relationship (f : Type → Type) [Function f] : Prop := sorry--add the definition if necessary

-- Variables and Conditions
variable (A : Type) [Valued A] -- A person's height and weight
variable (B : Type) [Valued B] -- Snowfall and the rate of traffic accidents
variable (C : Type → Type) [Function C] -- The distance traveled by a vehicle moving at a constant speed and time
variable (D : Type) [Valued D] -- The amount of fertilizer applied per acre and the grain yield per acre

-- Statement
theorem option_C_is_not_a_correlation_relationship : 
    ¬ (is_correlation_relationship (C Time Distance)) ∧ (is_functional_relationship (C Time Distance)) := sorry

end option_C_is_not_a_correlation_relationship_l382_382107


namespace percentage_house_rent_l382_382516

theorem percentage_house_rent (x : ℝ) 
  (h_food : x * 0.50) 
  (h_education : x * 0.15) 
  (h_remaining : x * 0.175) :
  (0.50 * (x - x * 0.50 - x * 0.15)) = x * 0.50 := by
  sorry

end percentage_house_rent_l382_382516


namespace train_journey_time_l382_382529

variable (x : ℝ)

def time_first_segment := x / 50
def time_second_segment := (2 * x) / 75
def time_third_segment := (x / 2) / 30
def total_time := time_first_segment x + time_second_segment x + time_third_segment x 

theorem train_journey_time : total_time x = (19 * x) / 300 :=
by
  sorry

end train_journey_time_l382_382529


namespace five_level_pyramid_has_80_pieces_l382_382188

-- Definitions based on problem conditions
def rods_per_level (level : ℕ) : ℕ :=
  if level = 1 then 4
  else if level = 2 then 8
  else if level = 3 then 12
  else if level = 4 then 16
  else if level = 5 then 20
  else 0

def connectors_per_level_transition : ℕ := 4

-- The total rods used for a five-level pyramid
def total_rods_five_levels : ℕ :=
  rods_per_level 1 + rods_per_level 2 + rods_per_level 3 + rods_per_level 4 + rods_per_level 5

-- The total connectors used for a five-level pyramid
def total_connectors_five_levels : ℕ :=
  connectors_per_level_transition * 5

-- The total pieces required for a five-level pyramid
def total_pieces_five_levels : ℕ :=
  total_rods_five_levels + total_connectors_five_levels

-- Main theorem statement for the proof problem
theorem five_level_pyramid_has_80_pieces : 
  total_pieces_five_levels = 80 :=
by
  -- We expect the total_pieces_five_levels to be equal to 80
  sorry

end five_level_pyramid_has_80_pieces_l382_382188


namespace count_distinct_functions_triple_application_cond_l382_382371

theorem count_distinct_functions_triple_application_cond :
  let A := {1, 2, 3, 4, 5, 6}
  in fintype.card {f : A → A // ∀ n : A, f (f (f n)) = n} = 81 :=
by {
  let A : finset ℕ := {1, 2, 3, 4, 5, 6},
  have : A.card = 6 := rfl,
  sorry
}

end count_distinct_functions_triple_application_cond_l382_382371


namespace point_on_x_axis_l382_382330

theorem point_on_x_axis (m : ℝ) (P : ℝ × ℝ) (hP : P = (m + 3, m - 1)) (hx : P.2 = 0) :
  P = (4, 0) :=
by
  sorry

end point_on_x_axis_l382_382330


namespace solution_set_linear_inequalities_l382_382933

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382933


namespace solution_set_linear_inequalities_l382_382926

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382926


namespace min_questions_to_identify_prize_box_l382_382985

-- Define the main problem
theorem min_questions_to_identify_prize_box (n : ℕ) (h : ∃! b : ℕ, 1 ≤ b ∧ b ≤ 100) :
  ∃ q : ℕ, (∀ (answers : Fin q → Bool), ∃! b : ℕ, 1 ≤ b ∧ b ≤ 100 ∧ host_answer (b : Nat) (answers : Fin q → Bool)) ∧ q = 99 := 
by
  sorry

-- Auxiliary definition which might represent the host answering scheme
-- This is a placeholder and would need to be properly formalized
noncomputable def host_answer (b : ℕ) (answers : Fin 99 → Bool) : Bool :=
  sorry

end min_questions_to_identify_prize_box_l382_382985


namespace min_value_x2_sub_xy_add_y2_l382_382239

theorem min_value_x2_sub_xy_add_y2 (x y : ℝ) :
  ∃ x y, x^2 - x * y + y^2 = 0 :=
by {
  use [0, 0],
  simp,
}

end min_value_x2_sub_xy_add_y2_l382_382239


namespace boat_speed_in_still_water_l382_382481

theorem boat_speed_in_still_water :
  ∃ b s : ℝ, b + s = 6 ∧ b - s = 2 ∧ b = 4 :=
by
  existsi (4 : ℝ), existsi (2 : ℝ)
  split
  { rw add_comm, exact add_self 2 }
  split
  { exact sub_self 2 }
  { refl }


end boat_speed_in_still_water_l382_382481


namespace each_person_spent_l382_382326

theorem each_person_spent
  (Tim_cost : ℝ)
  (First_friend_cost : ℝ)
  (Second_friend_cost : ℝ)
  (Third_friend_cost : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (tip_rate : ℝ)
  (num_people : ℝ)
  (Tim_cost_eq : Tim_cost = 50.20)
  (First_friend_cost_eq : First_friend_cost = 45.30)
  (Second_friend_cost_eq : Second_friend_cost = 48.10)
  (Third_friend_cost_eq : Third_friend_cost = 55.00)
  (discount_rate_eq : discount_rate = 0.15)
  (tax_rate_eq : tax_rate = 0.08)
  (tip_rate_eq : tip_rate = 0.25)
  (num_people_eq : num_people = 4) :
  let total_cost := Tim_cost + First_friend_cost + Second_friend_cost + Third_friend_cost,
      discount_amount := total_cost * discount_rate,
      discounted_total := total_cost - discount_amount,
      tax_amount := discounted_total * tax_rate,
      final_bill_before_tip := discounted_total + tax_amount,
      tip_amount := final_bill_before_tip * tip_rate,
      total_bill := final_bill_before_tip + tip_amount,
      amount_per_person := total_bill / num_people
  in amount_per_person = 56.97 :=
sorry

end each_person_spent_l382_382326


namespace largest_divisor_of_expression_l382_382573

theorem largest_divisor_of_expression :
  ∃ k : ℕ, (∀ m : ℕ, (m > k → m ∣ (1991 ^ k * 1990 ^ (1991 ^ 1992) + 1992 ^ (1991 ^ 1990)) = false))
  ∧ k = 1991 := by
sorry

end largest_divisor_of_expression_l382_382573


namespace distinct_sum_and_product_values_l382_382218

theorem distinct_sum_and_product_values :
  let s := {3, 6, 9, 12, 15, 18}
  in ∃ r, set.card r = 15 ∧
     (∀ a b ∈ s, (a + 1) * (b + 1) - 1 ∈ r) ∧
     (∀ x y ∈ r, x = y → x = y) :=
sorry

end distinct_sum_and_product_values_l382_382218


namespace triangle_angle_contradiction_l382_382765

theorem triangle_angle_contradiction (A B C : ℝ) (hA : 60 < A) (hB : 60 < B) (hC : 60 < C) (h_sum : A + B + C = 180) : false :=
by {
  -- This would be the proof part, which we don't need to detail according to the instructions.
  sorry
}

end triangle_angle_contradiction_l382_382765


namespace conic_section_pair_of_lines_l382_382802

theorem conic_section_pair_of_lines : 
  (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = 0 → (2 * x - 3 * y = 0 ∨ 2 * x + 3 * y = 0)) :=
by
  sorry

end conic_section_pair_of_lines_l382_382802


namespace twenty_mul_b_sub_a_not_integer_l382_382223

theorem twenty_mul_b_sub_a_not_integer {a b : ℝ} (hneq : a ≠ b) (hno_roots : ∀ x : ℝ,
  (x^2 + 20 * a * x + 10 * b) * (x^2 + 20 * b * x + 10 * a) ≠ 0) :
  ¬ ∃ n : ℤ, 20 * (b - a) = n :=
sorry

end twenty_mul_b_sub_a_not_integer_l382_382223


namespace circle_equation_l382_382282

theorem circle_equation (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (3, 1))
  (diameter : dist (A.1, A.2) (B.1, B.2) = 2 * dist ((A.1 + B.1) / 2, (A.2 + B.2) / 2) (A.1, A.2)) :
  ∃ C r, C = (2, 3/2) ∧ r = (sqrt 5) / 2 ∧ ∀ x y, (x - 2)^2 + (y - 3/2)^2 = (sqrt 5 / 2)^2 :=
sorry

end circle_equation_l382_382282


namespace equal_area_division_l382_382753

/-- Given a figure on a grid with a total area of 9 squares,
    and a point A on the grid,
    there exists a grid point B (other than A) such that a ray starting at point A
    and passing through point B divides the figure into two regions with equal area of 4.5 squares each. -/
theorem equal_area_division (A : Point) (fig : GridFigure) (total_area : fig.area = 9) :
  ∃ (B : Point), B ≠ A ∧ divides_equal_area A B fig :=
sorry

end equal_area_division_l382_382753


namespace collinearity_of_H_I_M_l382_382053

noncomputable theory

open_locale classical

-- Definitions as per conditions in the problem
-- Define a scalene triangle and the incircle
variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]

def is_scalene_triangle (ABC : triangle A B C) : Prop :=
  ¬(ABC.A = ABC.B ∨ ABC.B = ABC.C ∨ ABC.C = ABC.A)

-- Define the points where the incircle touches the triangle sides
def incircle_touch_points (ABC : triangle A B C) : A × B × C := sorry -- definition of incircle touch points at AB, BC, and CA

-- Define the circumcircle intersecting lines
def circumcircle_intersects_lines (A1 B C1 : A) : A × C := sorry -- definition where circumcircle of A1BC1 intersects lines B1A1 and B1C1

-- Define the orthocenter, incenter, and midpoint
def orthocenter (A0 B C0 : A) : A := sorry
def incenter (ABC : triangle A B C) : A := sorry
def midpoint (A C : A) : A := sorry

-- Now, the theorem statement
theorem collinearity_of_H_I_M 
  (ABC : triangle A B C)
  (h_scalene : is_scalene_triangle ABC)
  (A1 B1 C1 : A)
  (incircle_touches : incircle_touch_points ABC = (C1, A1, B1))
  (A0 C0 : A)
  (circumcircle_intersects : circumcircle_intersects_lines A1 ABC.B C1 = (A0, C0))
  (H := orthocenter A0 ABC.B C0)
  (I := incenter ABC)
  (M := midpoint ABC.A ABC.C) :
  collinear [H, I, M] :=
sorry

end collinearity_of_H_I_M_l382_382053


namespace stratified_sampling_middle_schools_l382_382685

theorem stratified_sampling_middle_schools (high_schools : ℕ) (middle_schools : ℕ) (elementary_schools : ℕ) (total_selected : ℕ) 
    (h_high_schools : high_schools = 10) (h_middle_schools : middle_schools = 30) (h_elementary_schools : elementary_schools = 60)
    (h_total_selected : total_selected = 20) : 
    middle_schools * (total_selected / (high_schools + middle_schools + elementary_schools)) = 6 := 
by 
  sorry

end stratified_sampling_middle_schools_l382_382685


namespace average_of_b_and_c_l382_382791

theorem average_of_b_and_c (a b c : ℝ) 
  (h₁ : (a + b) / 2 = 50) 
  (h₂ : c - a = 40) : 
  (b + c) / 2 = 70 := 
by
  sorry

end average_of_b_and_c_l382_382791


namespace solution_set_linear_inequalities_l382_382936

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382936


namespace starting_even_number_l382_382989

def is_even (n : ℤ) : Prop := n % 2 = 0

def span_covered_by_evens (count : ℤ) : ℤ := count * 2 - 2

theorem starting_even_number
  (count : ℤ)
  (end_num : ℤ)
  (H1 : is_even end_num)
  (H2 : count = 20)
  (H3 : end_num = 55) :
  ∃ start_num, is_even start_num ∧ start_num = end_num - span_covered_by_evens count + 1 := 
sorry

end starting_even_number_l382_382989


namespace orthocentric_tetrahedron_relation_l382_382402

section OrthocentricTetrahedron
variables {T : Type} [EuclideanGeometry T]
variables (O H : T) (R d : ℝ)

def is_circumcenter (O : T) (Tetra : T) : Prop := sorry
def is_orthocenter (H : T) (Tetra : T) : Prop := sorry
def circumradius (R : ℝ) (Tetra : T) : Prop := sorry
def distance_midpoints_opposite_edges (d : ℝ) (Tetra : T) : Prop := sorry

theorem orthocentric_tetrahedron_relation 
  (Tetra : T)
  (hO : is_circumcenter O Tetra)
  (hH : is_orthocenter H Tetra)
  (hR : circumradius R Tetra)
  (hd : distance_midpoints_opposite_edges d Tetra) :
  dist O H ^ 2 = 4 * R ^ 2 - 3 * d ^ 2 :=
  sorry
end OrthocentricTetrahedron

end orthocentric_tetrahedron_relation_l382_382402


namespace solution_set_of_linear_inequalities_l382_382861

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382861


namespace solution_set_of_linear_inequalities_l382_382863

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382863


namespace stratified_sampling_females_l382_382514

theorem stratified_sampling_females :
  let males := 500
  let females := 400
  let total_students := 900
  let total_surveyed := 45
  let males_surveyed := 25
  ((males_surveyed : ℚ) / males) * females = 20 := by
  sorry

end stratified_sampling_females_l382_382514


namespace quadrilateral_area_l382_382386

theorem quadrilateral_area (a b c d : ℝ) (horizontally_vertically_apart : a = b + 1 ∧ b = c + 1 ∧ c = d + 1 ∧ d = a + 1) : 
  area_of_quadrilateral = 6 :=
sorry

end quadrilateral_area_l382_382386


namespace total_lambs_l382_382740

def num_initial_lambs : ℕ := 6
def num_baby_lambs_per_mother : ℕ := 2
def num_mothers : ℕ := 2
def traded_lambs : ℕ := 3
def extra_lambs : ℕ := 7

theorem total_lambs :
  num_initial_lambs + (num_baby_lambs_per_mother * num_mothers) - traded_lambs + extra_lambs = 14 :=
by
  sorry

end total_lambs_l382_382740


namespace bridge_length_correct_l382_382095

def train_length : ℕ := 256
def train_speed_kmh : ℕ := 72
def crossing_time : ℕ := 20

noncomputable def convert_speed (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600 -- Conversion from km/h to m/s

noncomputable def bridge_length (train_length : ℕ) (speed_m : ℕ) (time_s : ℕ) : ℕ :=
  (speed_m * time_s) - train_length

theorem bridge_length_correct :
  bridge_length train_length (convert_speed train_speed_kmh) crossing_time = 144 :=
by
  sorry

end bridge_length_correct_l382_382095


namespace exists_contiguous_subseq_perfect_square_l382_382539

theorem exists_contiguous_subseq_perfect_square (n : ℕ) 
  (nums : list ℕ) (h_length : nums.length = 2^n) (primes : ∀ x ∈ nums, Nat.Prime x)
  (h_distinct : (nums.to_finset.card < n)) :
  ∃ (subseq : list ℕ), subseq ≠ [] ∧ subseq.product ^ 2 = subseq.product :=
by
  sorry

end exists_contiguous_subseq_perfect_square_l382_382539


namespace area_ratio_tangent_l382_382703

-- Define the given conditions and the goal in Lean 4
theorem area_ratio_tangent (α β : Real) (c : Real) (hα : 0 < α ∧ α < Real.pi / 2)
  (hβ : 0 < β ∧ β < Real.pi / 2) (AB_eq_c : AB = c) (D_midpoint : D = midpoint AB) 
  (intersect_AE : lineThrough D intersects AC at E) 
  (angle_DEA_beta : angle D EA = β) 
  (AE_gt_half_AC : AE > 0.5 * AC) : 
  (area quadrilateral_BCED / area triangle_ADE) = 
  ((3 * Real.tan β - Real.tan α) / (Real.tan α + Real.tan β)) := 
by 
  sorry

end area_ratio_tangent_l382_382703


namespace problem1_problem2_l382_382492

theorem problem1 : -1^(2022) + abs (-6) - (-3.14 - Real.pi)^0 + (-1/3)^(-2) = 13 :=
by
  sorry

theorem problem2 (a : ℝ) (h : a ≠ 2) : (1 - a / (a + 2)) / ((a * a - 4) / (a * a + 4 * a + 4)) = 2 / (a - 2) :=
by
  sorry

end problem1_problem2_l382_382492


namespace range_of_m_for_subset_l382_382307

open Set

variable (m : ℝ)

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | (2 * m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_of_m_for_subset (m : ℝ) : B m ⊆ A ↔ m ∈ Icc (-(1 / 2) : ℝ) (2 : ℝ) ∨ m > (2 : ℝ) :=
by
  sorry

end range_of_m_for_subset_l382_382307


namespace distinguishable_octahedrons_l382_382179

noncomputable def number_of_distinguishable_octahedrons (total_colors : ℕ) (used_colors : ℕ) : ℕ :=
  let num_ways_choose_colors := Nat.choose total_colors (used_colors - 1)
  let num_permutations := (used_colors - 1).factorial
  let num_rotations := 3
  (num_ways_choose_colors * num_permutations) / num_rotations

theorem distinguishable_octahedrons (h : number_of_distinguishable_octahedrons 9 8 = 13440) : true := sorry

end distinguishable_octahedrons_l382_382179


namespace circles_disjoint_l382_382306

theorem circles_disjoint :
  ∀ (x y u v : ℝ),
  (x^2 + y^2 = 1) →
  ((u-2)^2 + (v+2)^2 = 1) →
  (2^2 + (-2)^2) > (1 + 1)^2 :=
by sorry

end circles_disjoint_l382_382306


namespace point_in_fourth_quadrant_l382_382287

-- Define the complex number z
def z : ℂ := (3 - I) / (1 + I)

-- Define the conjugate of the complex number z
def z_conjugate : ℂ := conj z

-- Define the function to determine the quadrant of a complex number
def quadrant (c : ℂ) : String :=
  if c.re > 0 ∧ c.im > 0 then "first quadrant"
  else if c.re < 0 ∧ c.im > 0 then "second quadrant"
  else if c.re < 0 ∧ c.im < 0 then "third quadrant"
  else if c.re > 0 ∧ c.im < 0 then "fourth quadrant"
  else "on an axis"

-- Prove that the point corresponding to the conjugate of z is located in the fourth quadrant
theorem point_in_fourth_quadrant : quadrant z_conjugate = "fourth quadrant" :=
by
  sorry

end point_in_fourth_quadrant_l382_382287


namespace part1_eq_of_line_l_part2_eq_of_line_l1_l382_382284

def intersection_point (m n : ℝ × ℝ × ℝ) : ℝ × ℝ := sorry

def line_through_point_eq_dists (P A B : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry
def line_area_triangle (P : ℝ × ℝ) (triangle_area : ℝ) : ℝ × ℝ × ℝ := sorry

-- Conditions defined:
def m : ℝ × ℝ × ℝ := (2, -1, -3)
def n : ℝ × ℝ × ℝ := (1, 1, -3)
def P : ℝ × ℝ := intersection_point m n
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 2)
def triangle_area : ℝ := 4

-- Questions translated into Lean 4 statements:
theorem part1_eq_of_line_l : ∃ l : ℝ × ℝ × ℝ, 
  (l = line_through_point_eq_dists P A B) := sorry

theorem part2_eq_of_line_l1 : ∃ l1 : ℝ × ℝ × ℝ,
  (l1 = line_area_triangle P triangle_area) := sorry

end part1_eq_of_line_l_part2_eq_of_line_l1_l382_382284


namespace intersection_point_of_lines_l382_382432

theorem intersection_point_of_lines (m n : ℝ) :
  let y := x + 1,
      y' := mx + n in 
  ∃ (x y : ℝ), 
    (y = x + 1) ∧ 
    (y = mx + n) ∧
    x = 1 ∧ 
    y = 2 :=
by {
  sorry
}

end intersection_point_of_lines_l382_382432


namespace ratio_of_larger_to_smaller_l382_382981

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) (h3 : 0 < x) (h4 : 0 < y) : x / y = 4 / 3 := by
  sorry

end ratio_of_larger_to_smaller_l382_382981


namespace solve_equation_10_x_squared_plus_x_minus_2_eq_1_l382_382045

theorem solve_equation_10_x_squared_plus_x_minus_2_eq_1 (x : ℝ) : 
  10^(x^2 + x - 2) = 1 ↔ x = -2 ∨ x = 1 :=
by {
  sorry
}

end solve_equation_10_x_squared_plus_x_minus_2_eq_1_l382_382045


namespace double_people_half_work_l382_382483

-- Definitions
def initial_person_count (P : ℕ) : Prop := true
def initial_time (T : ℕ) : Prop := T = 16

-- Theorem
theorem double_people_half_work (P T : ℕ) (hP : initial_person_count P) (hT : initial_time T) : P > 0 → (2 * P) * (T / 2) = P * T / 2 := by
  sorry

end double_people_half_work_l382_382483


namespace solution_set_linear_inequalities_l382_382927

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382927


namespace robert_salary_loss_l382_382769

variable (S : ℝ)

theorem robert_salary_loss : 
  let decreased_salary := 0.80 * S
  let increased_salary := decreased_salary * 1.20
  let percentage_loss := 100 - (increased_salary / S) * 100
  percentage_loss = 4 :=
by
  sorry

end robert_salary_loss_l382_382769


namespace integer_pairs_count_l382_382059

theorem integer_pairs_count : 
  (∃ S : Finset (ℤ × ℤ), 
     (∀ p ∈ S, let (m, n) := p in m * n ≥ 0 ∧ m^3 + n^3 + 99 * m * n = 33^3) ∧
     S.card = 35) :=
sorry

end integer_pairs_count_l382_382059


namespace gumball_guarantee_min_l382_382161

theorem gumball_guarantee_min {R B W G : Nat} (R = 13) (B = 5) (W = 1) (G = 9) :
  ∀ n, (∃ r b g w, r + b + g + w = n ∧ r ≤ 2 ∧ b ≤ 2 ∧ g ≤ 2 ∧ w ≤ 1) →
  n < 8 → ∃ c, c >= 3 :=
by
  sorry

end gumball_guarantee_min_l382_382161


namespace unique_digit_sum_l382_382700

theorem unique_digit_sum
  (V E A : ℕ)
  (h_unique : ∀ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h2_digit_VE : 10 ≤ V * 10 + E ∧ V * 10 + E < 100)
  (h2_digit_AE : 10 ≤ A * 10 + E ∧ A * 10 + E < 100)
  (h_eq : (V * 10 + E) * (A * 10 + E) = 111 * A) :
  V + E + A + A = 26 :=
by
  /- Proof goes here -/
  sorry

end unique_digit_sum_l382_382700


namespace Nick_raising_money_l382_382019

theorem Nick_raising_money :
  let chocolate_oranges := 20
  let oranges_price := 10
  let candy_bars := 160
  let bars_price := 5
  let total_amount := chocolate_oranges * oranges_price + candy_bars * bars_price
  total_amount = 1000 := 
by
  sorry

end Nick_raising_money_l382_382019


namespace solution_set_inequalities_l382_382838

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382838


namespace matrix_eigenvalue_l382_382236

theorem matrix_eigenvalue (k : ℝ) : 
  (k = 1 + 2 * real.sqrt 5 ∨ k = 1 - 2 * real.sqrt 5) →
  ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧ 
    matrix.mul_vec ![![1, 10], ![2, 1]] v = k • v :=
begin
  sorry
end

end matrix_eigenvalue_l382_382236


namespace magnitude_of_z2_l382_382620

open Complex

def z1 : ℂ := -1 - I
def z1_conj : ℂ := conj z1

theorem magnitude_of_z2 (z2 : ℂ) (h : z1_conj * z2 = -2) : Complex.abs z2 = Real.sqrt 2 :=
sorry

end magnitude_of_z2_l382_382620


namespace trigonometric_expression_zero_l382_382587

theorem trigonometric_expression_zero {θ : ℝ} (h : Real.cot θ = 3) :
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 :=
sorry

end trigonometric_expression_zero_l382_382587


namespace probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l382_382775

noncomputable def P_n (n : ℕ) : ℚ :=
  if n = 3 then 1 / 4
  else if n = 4 then 3 / 4
  else 0

theorem probability_center_in_convex_hull_3_points :
  P_n 3 = 1 / 4 :=
by
  sorry

theorem probability_center_in_convex_hull_4_points :
  P_n 4 = 3 / 4 :=
by
  sorry

end probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l382_382775


namespace PQBC_concyclic_l382_382727

theorem PQBC_concyclic 
  {A B C D K L P Q : Type*}
  [trapezoid AB CD] (h_parallel : AB ∥ CD) (h_AB_gt_CD : AB > CD)
  (h_KL_ratio : ∀ {AK KB DL LC : ℝ}, AK / KB = DL / LC)
  (h_PQKL : P ∈ line_KL ∧ Q ∈ line_KL)
  (h_angle_APB_BCD : ∠APB = ∠BCD)
  (h_angle_CQD_ABC : ∠CQD = ∠ABC) :
  cyclic P Q B C := 
sorry

end PQBC_concyclic_l382_382727


namespace circle_tangent_bisector_l382_382134

theorem circle_tangent_bisector
  (O A B C : Point)
  (radius : ℝ)
  (h_radius : radius = 2)
  (h_circle_contains_A : dist O A = radius)
  (h_AB_tangent : AB_perpendicular_to_circle O A B)
  (θ : ℝ)
  (h_angle_AOB : angle O A B = θ)
  (h_C_on_OA : is_on_line C O A)
  (h_BC_bisects_ABO : B C bisects (angle B A O)) :
  OC = 2 / (1 + sin θ) :=
sorry

end circle_tangent_bisector_l382_382134


namespace final_product_is_twelve_l382_382986

theorem final_product_is_twelve :
  (11 + 22 + 33 + 44 = 110) ∧
  (∀ counts : Fin 4 → ℕ,
    ∃ remaining_counts : Fin 4 → ℕ,
      -- Starting counts
      counts 0 = 11 ∧
      counts 1 = 22 ∧
      counts 2 = 33 ∧
      counts 3 = 44 ∧
      
      -- Final remaining counts after applying operations
      (∑ i, remaining_counts i) = 3 ∧
      
      -- The final product
      (remaining_counts 0 + 1) * (remaining_counts 1 + 1) * (remaining_counts 2 + 1) = 12)
sorry

end final_product_is_twelve_l382_382986


namespace complex_quadrant_l382_382262

theorem complex_quadrant (b : ℝ) (z : ℂ) (hz : z = (4 + b * Complex.I) / (1 - Complex.I))
  (hz_re : z.re = -1) : Complex.re (Complex.conj z - b) < 0 ∧ Complex.im (Complex.conj z - b) < 0 :=
by
  sorry

end complex_quadrant_l382_382262


namespace solution_set_of_inequalities_l382_382966

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382966


namespace math_equivalence_problems_l382_382298

def line_parallel_condition (a : ℝ) : Prop :=
  let slope_l1 := - (1 / a)
  let slope_l2 := - (a / (2 * a - 3))
  slope_l1 = slope_l2 ∧ (a ≠ 1 → a = -3)

def line_intercepts_condition (a : ℝ) : Prop :=
  let x_intercept := if (2 * a - 3 != 0) then (a - 2) / (2 * a - 3) else 0
  let y_intercept := if a != 0 then (a - 2) / a else 0
  x_intercept = y_intercept → (a = 1 ∨ a = 2)

def line_perpendicular_condition (a : ℝ) : Prop :=
  let slope_l1 := - (1 / a)
  let slope_l2 := - (a / (2 * a - 3))
  slope_l1 * slope_l2 = -1 → (a = 0 ∨ a = 2)

def line_distance_condition (a : ℝ) : Prop :=
  a = -3 →
  let distance := (|3 - (5/3)|) / (Real.sqrt (1 ^ 2 + (-3) ^ 2))
  distance = 2 * Real.sqrt(10) / 15

theorem math_equivalence_problems :
  (line_parallel_condition a) ∧
  (line_intercepts_condition a) ∧
  (line_perpendicular_condition a) ∧
  (line_distance_condition a) :=
by
  sorry

end math_equivalence_problems_l382_382298


namespace four_fives_to_hundred_case1_four_fives_to_hundred_case2_l382_382108

theorem four_fives_to_hundred_case1 : (5 + 5) * (5 + 5) = 100 :=
by sorry

theorem four_fives_to_hundred_case2 : (5 * 5 - 5) * 5 = 100 :=
by sorry

end four_fives_to_hundred_case1_four_fives_to_hundred_case2_l382_382108


namespace cosine_of_negative_135_l382_382549

theorem cosine_of_negative_135 : Real.cos (-(135 * Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end cosine_of_negative_135_l382_382549


namespace necklace_cost_l382_382035

theorem necklace_cost (total_savings earrings_cost remaining_savings: ℕ) 
                      (h1: total_savings = 80) 
                      (h2: earrings_cost = 23) 
                      (h3: remaining_savings = 9) : 
   total_savings - earrings_cost - remaining_savings = 48 :=
by
  sorry

end necklace_cost_l382_382035


namespace ratio_of_distances_l382_382366

variables (A B C D E : Type)
variables [regular_tetrahedron A B C D] [point_lies_on_midpoint E A B]

-- Definition of distances to planes and edges
def dist_to_planes (E A B C D : Type) [regular_tetrahedron A B C D]
: ℝ := distance E (plane DAC) + distance E (plane DBC) + distance E (plane BCD)

def dist_to_edges (E A B C D : Type) [regular_tetrahedron A B C D]
: ℝ := distance E (edge AC) + distance E (edge BC) + distance E (edge CD) + distance E (edge DA) + distance E (edge DB)

-- The main theorem
theorem ratio_of_distances (E A B C D : Type) 
  [regular_tetrahedron A B C D] [point_lies_on_midpoint E A B] :
  let s := dist_to_planes E A B C D,
      S := dist_to_edges E A B C D in
  s / S = 9 / 10 :=
sorry

end ratio_of_distances_l382_382366


namespace number_divisible_by_77_l382_382520

theorem number_divisible_by_77 (M : ℕ) : ∃ N : ℕ, (M * 10^(digits_sum (N)) + append_digits (N)) % 77 = 0 :=
by
  sorry

-- Helper function to calculate number of digits when 1, 2, ..., N are appended
def digits_sum (N : ℕ) : ℕ := 
  N * (N + 1) / 2  -- Sum of first N natural numbers representing the digits being appended (simplified form)

-- Helper function to calculate the entire number formed after digits are appended
def append_digits (N : ℕ) : ℕ := 
  (List.range (N + 1)).foldl (λ acc x, acc * 10 + x) M  -- Fold over the range from 0 to N to generate appended number


end number_divisible_by_77_l382_382520


namespace regular_pentagon_concyclic_regular_pentagon_circumradius_l382_382408

theorem regular_pentagon_concyclic (x : ℝ) :
  (∃ (A B C D E : ℝ×ℝ), 
    ∃ O : ℝ × ℝ, 
      let center := O in 
      (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ A) ∧
      (dist center A = dist center B ∧ 
       dist center B = dist center C ∧ 
       dist center C = dist center D ∧ 
       dist center D = dist center E ∧ 
       dist center E = dist center A)) :=
sorry

theorem regular_pentagon_circumradius (x : ℝ) :
  (∃ A B C D E : ℝ × ℝ,
    (dist A B = x ∧
     dist B C = x ∧
     dist C D = x ∧
     dist D E = x ∧
     dist E A = x) ∧
    (∃ O : ℝ × ℝ, 
      let center := O in 
      let R := dist center A in 
      (dist center A = dist center B ∧ 
       dist center B = dist center C ∧ 
       dist center C = dist center D ∧ 
       dist center D = dist center E ∧ 
       dist center E = dist center A) ∧
      R = x / (2 * real.sin (real.pi / 5)))) :=
sorry

end regular_pentagon_concyclic_regular_pentagon_circumradius_l382_382408


namespace total_amount_paid_l382_382486

theorem total_amount_paid (quantity_apples : ℕ) (rate_per_kg_apples : ℕ) (quantity_mangoes : ℕ) (rate_per_kg_mangoes : ℕ) :
  quantity_apples = 8 →
  rate_per_kg_apples = 70 →
  quantity_mangoes = 9 →
  rate_per_kg_mangoes = 75 →
  (quantity_apples * rate_per_kg_apples + quantity_mangoes * rate_per_kg_mangoes) = 1235 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_amount_paid_l382_382486


namespace circumscribed_circle_radius_l382_382089

noncomputable def circumradius_of_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℚ :=
  c / 2

theorem circumscribed_circle_radius :
  circumradius_of_right_triangle 30 40 50 (by norm_num : 30^2 + 40^2 = 50^2) = 25 := by
norm_num /- correct answer confirmed -/
sorry

end circumscribed_circle_radius_l382_382089


namespace solution_set_inequalities_l382_382837

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382837


namespace vasya_password_combinations_l382_382999

/-- A function to count the number of valid 4-digit passwords as per the given constraints. -/
def count_valid_passwords : Nat := 
  let digits := {0, 1, 3, 4, 5, 6, 7, 8, 9}
  let valid_A := digits.toFinset.card  -- 9
  let valid_B := valid_A - 1            -- 8 (excluding A)
  let valid_C := valid_B - 1            -- 7 (excluding A and B)
  valid_A * valid_B * valid_C

theorem vasya_password_combinations : count_valid_passwords = 504 := by
  sorry

end vasya_password_combinations_l382_382999


namespace find_present_ratio_l382_382817

noncomputable def present_ratio_of_teachers_to_students : Prop :=
  ∃ (S T S' T' : ℕ),
    (T = 3) ∧
    (S = 50 * T) ∧
    (S' = S + 50) ∧
    (T' = T + 5) ∧
    (S' / T' = 25 / 1) ∧ 
    (T / S = 1 / 50)

theorem find_present_ratio : present_ratio_of_teachers_to_students :=
by
  sorry

end find_present_ratio_l382_382817


namespace bricklayer_wall_l382_382129

/-- 
A bricklayer lays a certain number of meters of wall per day and works for a certain number of days.
Given the daily work rate and the number of days worked, this proof shows that the total meters of 
wall laid equals the product of the daily work rate and the number of days.
-/
theorem bricklayer_wall (daily_rate : ℕ) (days_worked : ℕ) (total_meters : ℕ) 
  (h1 : daily_rate = 8) (h2 : days_worked = 15) : total_meters = 120 :=
by {
  sorry
}

end bricklayer_wall_l382_382129


namespace tina_wins_more_than_losses_l382_382453

theorem tina_wins_more_than_losses :
  let w1 := 10 in
  let w2 := w1 + 5 in
  let w3 := 2 * w2 in
  let l := 2 in
  w3 - l = 28 :=
by
  sorry

end tina_wins_more_than_losses_l382_382453


namespace no_y_satisfies_equation_under_condition_l382_382216

theorem no_y_satisfies_equation_under_condition :
  ∀ y : ℝ, y + 1 = 0 → 3 * y^2 - 8 * y ≠ 5 :=
by
  intro y
  intro h
  rw [←add_eq_zero_iff_eq_neg] at h
  sorry

end no_y_satisfies_equation_under_condition_l382_382216


namespace solution_set_linear_inequalities_l382_382915

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382915


namespace cabbage_area_one_square_foot_l382_382163

-- Define the parameters
def num_cabbages_last_year := 3969
def num_cabbages_this_year := 4096
def increment_cabbages := 127
def area_last_year := 3969
def area_this_year := 4096

-- Properties of a square garden of cabbages
def side_length_last_year := (sqrt area_last_year)
def side_length_this_year := (sqrt area_this_year)

-- Calculate the area per cabbage for this year
def area_per_cabbage := area_this_year / num_cabbages_this_year

-- The goal is to prove that the area per cabbage is 1 square foot
theorem cabbage_area_one_square_foot :
  area_per_cabbage = 1 :=
by
  sorry

end cabbage_area_one_square_foot_l382_382163


namespace find_hourly_charge_l382_382020

variable {x : ℕ}

--Assumptions and conditions
def fixed_charge := 17
def total_paid := 80
def rental_hours := 9

-- Proof problem
theorem find_hourly_charge (h : fixed_charge + rental_hours * x = total_paid) : x = 7 :=
sorry

end find_hourly_charge_l382_382020


namespace solution_set_linear_inequalities_l382_382935

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382935


namespace max_divisibility_by_4_l382_382114

theorem max_divisibility_by_4 (w x y z : ℕ) 
  (hw : w % 2 = 1) (hx : x % 2 = 1) (hy : y % 2 = 1) (hz : z % 2 = 1) 
  (h_distinct : ∀ x y z w, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) : 
  ∃ k, k = 4 ∧ (w^2 + x^2) * (y^2 + z^2) % k = 0 := 
sorry

end max_divisibility_by_4_l382_382114


namespace squares_of_roots_equation_l382_382556

theorem squares_of_roots_equation (a b x : ℂ) 
  (h : ab * x^2 - (a + b) * x + 1 = 0) : 
  a^2 * b^2 * x^2 - (a^2 + b^2) * x + 1 = 0 :=
sorry

end squares_of_roots_equation_l382_382556


namespace cos_alpha_value_l382_382257

theorem cos_alpha_value (α : ℝ) (h1 : π / 4 < α ∧ α < 3 * π / 4) (h2 : sin (α - π / 4) = 4 / 5) :
  cos α = -sqrt 2 / 10 :=
sorry

end cos_alpha_value_l382_382257


namespace white_dandelions_on_saturday_l382_382139

theorem white_dandelions_on_saturday 
  (yellow_monday : ℕ) (white_monday : ℕ)
  (yellow_wednesday : ℕ) (white_wednesday : ℕ)
  (life_cycle : ∀ d, (d.bloom == "yellow" ∧ d.age == 3) ∨ (d.bloom == "white" ∧ d.age == 4) ∨ (d.bloom == "dispersed" ∧ d.age == 5))
  (total_monday : yellow_monday = 20 ∧ white_monday = 14)
  (total_wednesday : yellow_wednesday = 15 ∧ white_wednesday = 11):
  let new_dandelions := 26 - 20,
      white_dandelions_saturday := new_dandelions
  in white_dandelions_saturday = 6 := by
  let dandelions_tuesday_wednesday := new_dandelions
  have h : white_dandelions_saturday = 6,
  from sorry
  exact h

end white_dandelions_on_saturday_l382_382139


namespace problem_1_problem_2_l382_382305

open Real

noncomputable def f (omega : ℝ) (x : ℝ) : ℝ := 
  (cos (omega * x) * cos (omega * x) + sqrt 3 * cos (omega * x) * sin (omega * x) - 1/2)

theorem problem_1 (ω : ℝ) (hω : ω > 0):
 (f ω x = sin (2 * x + π / 6)) ∧ 
 (∀ k : ℤ, ∀ x : ℝ, (-π / 3 + ↑k * π) ≤ x ∧ x ≤ (π / 6 + ↑k * π) → f ω x = sin (2 * x + π / 6)) :=
sorry

theorem problem_2 (A b S a : ℝ) (hA : A / 2 = π / 3)
  (hb : b = 1) (hS: S = sqrt 3) :
  a = sqrt 13 :=
sorry

end problem_1_problem_2_l382_382305


namespace compound_interest_for_two_years_l382_382051

-- Defining the conditions
def P : ℝ := 17000
def SI : ℝ := 10200
def r : ℝ := 3

-- Definition of CI based on the problem conditions
def CI (P r : ℝ) (t : ℝ) := P * (1 + r / 100)^t - P

-- The statement to be proved
theorem compound_interest_for_two_years :
  CI P r 2 = 1053.3 :=
by
  sorry

end compound_interest_for_two_years_l382_382051


namespace each_dolphin_training_hours_l382_382814

theorem each_dolphin_training_hours
  (num_dolphins : ℕ)
  (num_trainers : ℕ)
  (hours_per_trainer : ℕ)
  (total_hours : ℕ := num_trainers * hours_per_trainer)
  (hours_per_dolphin_daily : ℕ := total_hours / num_dolphins)
  (h1 : num_dolphins = 4)
  (h2 : num_trainers = 2)
  (h3 : hours_per_trainer = 6) :
  hours_per_dolphin_daily = 3 :=
  by sorry

end each_dolphin_training_hours_l382_382814


namespace find_longest_interval_l382_382106

theorem find_longest_interval :
  ∀ (t1 t2 t3 : ℕ),  t1 = 1500 * 1 ∧ t2 = 10 * 60 ∧ t3 = 1 * 24 * 60 → max t1 (max t2 t3) = t1 :=
by
  intros t1 t2 t3 h,
  cases h with ht1 h,
  cases h with ht2 ht3,
  rw [ht1, ht2, ht3],
  sorry

end find_longest_interval_l382_382106


namespace ramsey_6_3_3_l382_382401

open Classical

theorem ramsey_6_3_3 (G : SimpleGraph (Fin 6)) :
  ∃ (A : Finset (Fin 6)), A.card = 3 ∧ (∀ (x y : Fin 6), x ∈ A → y ∈ A → x ≠ y → G.Adj x y) ∨ ∃ (B : Finset (Fin 6)), B.card = 3 ∧ (∀ (x y : Fin 6), x ∈ B → y ∈ B → x ≠ y → ¬ G.Adj x y) :=
by
  sorry

end ramsey_6_3_3_l382_382401


namespace trajectory_ellipse_l382_382026

noncomputable theory

variables (P S A B C D : Type) [Point P] [Point S] [Point A] [Point B] [Point C] [Point D]
variables (SBC ABC : Set Point) [Plane ABC] [Face SBC]

-- Conditions
variables (in_sbc : P ∈ SBC)
variables (dist_equal : distance P (Plane ABC) = distance P S)
variables (not_perp : ¬ perpendicular SBC (Plane ABC))

-- Theorem Statement
theorem trajectory_ellipse :
  ∃ (ellipse : Set Point), 
    ∀ P ∈ SBC, distance P (Plane ABC) = distance P S → 
                ¬ perpendicular SBC (Plane ABC) → 
                P ∈ ellipse := 
sorry

end trajectory_ellipse_l382_382026


namespace solution_set_of_inequalities_l382_382965

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382965


namespace regression_coeff_nonzero_l382_382253

theorem regression_coeff_nonzero (a b r : ℝ) (h : b = 0 → r = 0) : b ≠ 0 :=
sorry

end regression_coeff_nonzero_l382_382253


namespace solution_set_system_of_inequalities_l382_382948

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382948


namespace solution_set_l382_382840

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382840


namespace divisors_of_81n4_l382_382249

theorem divisors_of_81n4 (n : ℕ) (h : nat.totient (110 * n^3) = 110) :
  nat.totient (81 * n^4) = 325 :=
sorry

end divisors_of_81n4_l382_382249


namespace sum_of_ages_in_10_years_l382_382538

-- Define the initial conditions about Ann's and Tom's ages
def AnnCurrentAge : ℕ := 6
def TomCurrentAge : ℕ := 2 * AnnCurrentAge

-- Define their ages 10 years later
def AnnAgeIn10Years : ℕ := AnnCurrentAge + 10
def TomAgeIn10Years : ℕ := TomCurrentAge + 10

-- The proof statement
theorem sum_of_ages_in_10_years : AnnAgeIn10Years + TomAgeIn10Years = 38 := by
  sorry

end sum_of_ages_in_10_years_l382_382538


namespace solve_inequalities_l382_382896

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382896


namespace g_g_is_even_l382_382007

variable (g : ℝ → ℝ) (h_even : ∀ x : ℝ, g (-x) = g x)

theorem g_g_is_even : ∀ x : ℝ, g (g (-x)) = g (g x) :=
by
  assume x
  rw [h_even]
  exact h_even (g x)

end g_g_is_even_l382_382007


namespace linear_inequalities_solution_l382_382868

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382868


namespace shaded_area_of_semi_circle_l382_382237

variable (R : ℝ)

theorem shaded_area_of_semi_circle (α : ℝ) (hα : α = 45 * Real.pi / 180) :
  let S0 := (Real.pi * R^2) / 2 in
  let shaded_area := S0 in
  shaded_area = (Real.pi * R^2) / 2 :=
by
  sorry

end shaded_area_of_semi_circle_l382_382237


namespace diff_of_squares_l382_382209

-- Define constants a and b
def a : ℕ := 65
def b : ℕ := 35

-- State the theorem to be proven using the difference of squares formula
theorem diff_of_squares : a^2 - b^2 = 3000 := by
  have h1 : a + b = 100 := by rfl -- Checking a + b
  have h2 : a - b = 30 := by rfl -- Checking a - b
  have h3 : (a+b)*(a-b) = 3000 := by
    rw [h1, h2] -- Substituting the values
    norm_num -- Simplifying the arithmetics
  exact h3 -- Providing the ultimate result

end diff_of_squares_l382_382209


namespace last_number_odd_l382_382463

theorem last_number_odd : 
  (∃ n ∈ {1..2017}, is_odd n) → 
  (∀ a b ∈ {1..2017}, (a ≠ b) → (a, b) → |a - b|^3 ∈ {1..2017}) → 
  (∃ n ∈ {1}, is_odd n) := 
sorry

end last_number_odd_l382_382463


namespace solve_system_l382_382416

open Real

theorem solve_system :
  (∃ x y : ℝ, (sin x) ^ 2 + (cos y) ^ 2 = y ^ 4 ∧ (sin y) ^ 2 + (cos x) ^ 2 = x ^ 2) → 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) := by
  sorry

end solve_system_l382_382416


namespace games_last_month_l382_382714

def games_this_month : ℕ := 11
def games_next_month : ℕ := 16
def total_games : ℕ := 44

theorem games_last_month (G : ℕ) (h : G + games_this_month + games_next_month = total_games) : G = 17 :=
by {
  -- Consider the equations and simplifications given in the solution
  have h1 : G + 27 = 44,
  from h,
  have h2 : G = 44 - 27,
  from nat.sub_eq_of_eq_add h1.symm,
  have h3 : G = 17,
  from h2,
  exact h3
}

end games_last_month_l382_382714


namespace total_sales_last_year_approx_l382_382181

-- Define the conditions
def total_sales_this_year : ℝ := 460
def percent_increase : ℝ := 43.75

-- Define the old value function based on the given information
def old_sales (new_sales : ℝ) (percent_increase : ℝ) : ℝ :=
  new_sales / (1 + (percent_increase / 100))

-- Statement to prove
theorem total_sales_last_year_approx :
  old_sales total_sales_this_year percent_increase ≈ 320 :=
by
  -- This would be the place for the actual proof
  sorry

end total_sales_last_year_approx_l382_382181


namespace solution_set_linear_inequalities_l382_382937

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382937


namespace distance_between_centers_of_circles_l382_382198

theorem distance_between_centers_of_circles (a b : ℝ) (h : a > b) (inscribed_circle_exists : ∃ r : ℝ, r = 1) :
  ∃ d : ℝ, d = (a^2 - b^2) / (8 * sqrt (a * b)) :=
sorry

end distance_between_centers_of_circles_l382_382198


namespace orthocenter_DEF_eq_incenter_ABC_vector_sum_zero_eq_equilateral_l382_382689

-- Assuming the required definitions and structures are in place for geometrical constructs.
variables {A B C D E F : Type}
variables [triangle A B C] [circumcircle_intersection A B C D E F]

/-- Part (a): Prove the orthocenter of triangle DEF coincides with the incenter of triangle ABC. -/
theorem orthocenter_DEF_eq_incenter_ABC :
  orthocenter (triangle D E F) = incenter (triangle A B C) :=
sorry

/-- Part (b): If AD + BE + CF = 0, then triangle ABC is equilateral. -/
theorem vector_sum_zero_eq_equilateral (h : vector_sum_zero (AD BE CF)) :
  is_equilateral (triangle A B C) :=
sorry

end orthocenter_DEF_eq_incenter_ABC_vector_sum_zero_eq_equilateral_l382_382689


namespace solution_set_of_inequalities_l382_382957

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382957


namespace paint_after_third_day_l382_382500

def initial_paint := 2
def paint_used_first_day (x : ℕ) := (1 / 2) * x
def remaining_after_first_day (x : ℕ) := x - paint_used_first_day x
def paint_used_second_day (y : ℕ) := (1 / 4) * y
def remaining_after_second_day (y : ℕ) := y - paint_used_second_day y
def paint_used_third_day (z : ℕ) := (1 / 3) * z
def remaining_after_third_day (z : ℕ) := z - paint_used_third_day z

theorem paint_after_third_day :
  remaining_after_third_day 
    (remaining_after_second_day 
      (remaining_after_first_day initial_paint)) = initial_paint / 2 := 
  by
  sorry

end paint_after_third_day_l382_382500


namespace distance_between_parallel_lines_is_one_l382_382296

-- Define the two lines as conditions
def line1 (x y : ℝ) : Prop := 5 * x + 12 * y - 7 = 0
def line2 (x y : ℝ) : Prop := 5 * x + 12 * y + 6 = 0

-- Define the proof problem: proving the distance between line1 and line2 is 1
theorem distance_between_parallel_lines_is_one : 
  (∀ x y : ℝ, line1 x y → line2 x y → abs (6 - (-7)) / real.sqrt (5^2 + 12^2) = 1) :=
sorry

end distance_between_parallel_lines_is_one_l382_382296


namespace elise_spent_on_puzzle_l382_382229

-- Definitions based on the problem conditions:
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def remaining_money : ℕ := 1

-- Prove that the amount spent on the puzzle is $18.
theorem elise_spent_on_puzzle : initial_money + saved_money - spent_on_comic - remaining_money = 18 := by
  sorry

end elise_spent_on_puzzle_l382_382229


namespace min_hypotenuse_l382_382680

theorem min_hypotenuse {a b : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 10) :
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ c ≥ 5 * Real.sqrt 2 :=
by
  sorry

end min_hypotenuse_l382_382680


namespace mulch_cost_l382_382749

/-- 
Given the conditions:
1. Mulch costs $5 per cubic foot.
2. 1 cubic yard equals 27 cubic feet.
3. The quantity of mulch is 8 cubic yards.
Prove that the cost, in dollars, of 8 cubic yards of mulch is 1080.
-/
theorem mulch_cost : 
  (cost_per_ft³ : ℕ) = 5 → 
  (conversion_factor : ℕ) = 27 → 
  (quantity_yd³ : ℕ) = 8 → 
  (total_cost : ℕ) = quantity_yd³ * conversion_factor * cost_per_ft³ → 
  total_cost = 1080 :=
by
  intros cost_per_ft³_eq conversion_factor_eq quantity_yd³_eq total_cost_eq
  rw [cost_per_ft³_eq, conversion_factor_eq, quantity_yd³_eq] at total_cost_eq
  simp at total_cost_eq
  sorry

end mulch_cost_l382_382749


namespace integer_root_of_polynomial_l382_382217

theorem integer_root_of_polynomial (a b c : ℚ) (h_poly : (x : ℚ) → x^3 + a * x^2 + b * x + c = 0)
  (root1 : ℚ) (root2 : ℚ) (root3 : ℤ)
  (h_root1 : root1 = 3 - real.sqrt 5)
  (h_root2 : root2 = 3 + real.sqrt 5)
  (h_roots : h_poly root1 = 0 ∧ h_poly root2 = 0 ∧ h_poly (root3 : ℚ) = 0) :
  root3 = 0 :=
sorry

end integer_root_of_polynomial_l382_382217


namespace a3_eq_m_squared_plus_m_S_is_geometric_sequence_is_integer_T2k_div_Tk_l382_382704

variable (m : ℝ)

def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else if n = 2 then m else if n = 3 then m^2 + m else sorry

def S : ℕ → ℝ
| 1 => a 1
| n+1 => a (n+1) * S n / (a n)

theorem a3_eq_m_squared_plus_m : a 3 = m^2 + m := by
  unfold a
  simp

theorem S_is_geometric_sequence (n : ℕ) (h : n ≥ 2) : (S n)^2 = (S (n-1)) * (S (n+1)) := by
  sorry

variable (k : ℕ)

def b (n : ℕ) : ℕ :=
  if n ≤ k then a (2*k - n + 1) else a (n) * a (n + 1)

def T : ℕ → ℝ
| 0       => 0
| n + 1   => T n + b n

theorem is_integer_T2k_div_Tk (m = 1) (k : ℕ) (hk : k = 2*n - 1 ∧ n ∈ ℕ) : ∃ (c : ℝ), T (2*k) / T k = 📜 := by
  sorry

end a3_eq_m_squared_plus_m_S_is_geometric_sequence_is_integer_T2k_div_Tk_l382_382704


namespace solve_inequalities_l382_382909

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382909


namespace middle_rungs_widths_l382_382982

theorem middle_rungs_widths (a : ℕ → ℝ) (d : ℝ) :
  a 1 = 33 ∧ a 12 = 110 ∧ (∀ n, a (n + 1) = a n + 7) →
  (a 2 = 40 ∧ a 3 = 47 ∧ a 4 = 54 ∧ a 5 = 61 ∧
   a 6 = 68 ∧ a 7 = 75 ∧ a 8 = 82 ∧ a 9 = 89 ∧
   a 10 = 96 ∧ a 11 = 103) :=
by
  sorry

end middle_rungs_widths_l382_382982


namespace solution_set_linear_inequalities_l382_382912

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382912


namespace solve_inequalities_l382_382908

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382908


namespace total_pencils_l382_382774

/-- The conditions defining the number of pencils Sarah buys each day. -/
def pencils_monday : ℕ := 20
def pencils_tuesday : ℕ := 18
def pencils_wednesday : ℕ := 3 * pencils_tuesday

/-- The hypothesis that the total number of pencils bought by Sarah is 92. -/
theorem total_pencils : pencils_monday + pencils_tuesday + pencils_wednesday = 92 :=
by
  -- calculations skipped
  sorry

end total_pencils_l382_382774


namespace projection_correct_l382_382576

def vector1 : ℝ × ℝ × ℝ := (3, -1, 4)
def vector2 : ℝ × ℝ × ℝ := (1, 0, 2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def norm_squared (v : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2 + v.3 * v.3

def scale_vector (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  scale_vector (dot_product u v / norm_squared v) v

theorem projection_correct :
  projection vector1 vector2 = (11/5, 0, 22/5) :=
by
  sorry

end projection_correct_l382_382576


namespace catching_kolya_l382_382701

-- Define necessary conditions and parameters
variables (park : Type) [MetricSpace park] (A B : park)
variable (v : ℝ) -- speed of Kolya's parents
variable (k_speed : ℝ) -- speed of Kolya
variables (control_region : park → park → Prop) -- region a parent controls

-- Define specific conditions
def conditions : Prop :=
  (∀ p1 p2 : park, ∃ m d : park, m ≠ d ∧ control_region m p1 ∧ control_region d p2) ∧
  k_speed = 3 * v

-- The theorem to be proven
theorem catching_kolya (h : conditions park A B v k_speed control_region) :
  ∃ m d : park, control_region m A ∧ control_region d B := 
by
  sorry

end catching_kolya_l382_382701


namespace min_edges_triangle_l382_382005

-- Definitions
structure Graph :=
  (V : Type)
  (adj : V → V → Prop)
  (finite : Fintype V)
  (simple : ∀ {x y : V}, adj x y → adj y x)
  (undirected : ∀ {x y : V}, adj x y → adj y x)
  (connected : ∀ (v : V), ∃ p : List V, p.head = v ∧ p.last = v ∧ ∀ x y ∈ p, adj x y)

def edge_part_of_triangle (G : Graph) (x y : G.V) : Prop :=
  G.adj x y ∧ ∃ (z : G.V), G.adj x z ∧ G.adj y z

-- Definition to capture number of vertices and edges
def num_vertices (G : Graph) : ℕ := Fintype.card G.V
def num_edges (G : Graph) : ℕ := Fintype.card {e : G.V × G.V // G.adj e.1 e.2}

-- The theorem
theorem min_edges_triangle (G : Graph) (h_triangle : ∀ (x y : G.V), G.adj x y → edge_part_of_triangle G x y)
  (h : num_vertices G = n) : num_edges G = ⌈ (3 * n - 2) / 2 ⌉ :=
by
  sorry

end min_edges_triangle_l382_382005


namespace least_number_to_add_l382_382471

theorem least_number_to_add (x : ℕ) (h1 : x = 1101) : ∃ y, (1101 + y) % 24 = 0 ∧ y = 3 := 
begin
    use (24 - 1101 % 24),
    split,
    {
        -- Proof that (1101 + (24 - 1101 % 24)) % 24 = 0
        sorry
    },
    {
        -- Proof that 24 - 1101 % 24 = 3
        sorry
    }
end

end least_number_to_add_l382_382471


namespace JodiMilesFourthWeek_l382_382358

def JodiMilesFirstWeek := 1 * 6
def JodiMilesSecondWeek := 2 * 6
def JodiMilesThirdWeek := 3 * 6
def TotalMilesFirstThreeWeeks := JodiMilesFirstWeek + JodiMilesSecondWeek + JodiMilesThirdWeek
def TotalMilesFourWeeks := 60

def MilesInFourthWeek := TotalMilesFourWeeks - TotalMilesFirstThreeWeeks
def DaysInWeek := 6

theorem JodiMilesFourthWeek : (MilesInFourthWeek / DaysInWeek) = 4 := by
  sorry

end JodiMilesFourthWeek_l382_382358


namespace count_primes_between_71_and_95_l382_382646

theorem count_primes_between_71_and_95 : 
  let primes := [71, 73, 79, 83, 89, 97] in
  let filtered_primes := primes.filter (λ p, 71 < p ∧ p < 95) in
  filtered_primes.length = 5 := 
by 
  sorry

end count_primes_between_71_and_95_l382_382646


namespace solution_set_of_linear_inequalities_l382_382854

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382854


namespace solve_ax2_2x_a_inequality_l382_382263

noncomputable def solve_inequality (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x > 0 }
  else if 0 < a ∧ a < 1 then { x | (1 - real.sqrt(1 - a^2)) / a < x ∧ x < (1 + real.sqrt(1 - a^2)) / a }
  else if a ≥ 1 then ∅
  else if -1 < a ∧ a < 0 then { x | (x > (1 - real.sqrt(1 - a^2)) / a) ∨ (x < (1 + real.sqrt(1 - a^2)) / a) }
  else if a = -1 then { x | x ≠ 1 / a }
  else { x | true }

theorem solve_ax2_2x_a_inequality (a : ℝ) :
  solve_inequality a = if a = 0 then { x | x > 0 }
  else if 0 < a ∧ a < 1 then { x | (1 - real.sqrt(1 - a^2)) / a < x ∧ x < (1 + real.sqrt(1 - a^2)) / a }
  else if a ≥ 1 then ∅
  else if -1 < a ∧ a < 0 then { x | (x > (1 - real.sqrt(1 - a^2)) / a) ∨ (x < (1 + real.sqrt(1 - a^2)) / a) }
  else if a = -1 then { x | x ≠ 1 / a }
  else { x | true } := sorry

end solve_ax2_2x_a_inequality_l382_382263


namespace solution_set_l382_382852

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382852


namespace solution_set_of_inequalities_l382_382972

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382972


namespace probability_between_lines_in_first_quadrant_l382_382422

theorem probability_between_lines_in_first_quadrant :
  let p : ℝ → ℝ := λ x, -2 * x + 8
      q : ℝ → ℝ := λ x, -3 * x + 9
      area_p := (1 / 2) * 4 * 8
      area_q := (1 / 2) * 3 * 9
      area_pq := area_p - area_q
  in (area_pq / area_p) = 0.16 :=
by
  sorry

end probability_between_lines_in_first_quadrant_l382_382422


namespace solution_set_l382_382841

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382841


namespace pen_and_notebook_cost_l382_382362

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 15 * p + 5 * n = 130 ∧ p > n ∧ p + n = 10 := by
  sorry

end pen_and_notebook_cost_l382_382362


namespace solution_set_of_linear_inequalities_l382_382859

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382859


namespace initial_investment_l382_382132

-- Conditions
def interest := 300 -- Interest earned
def annual_rate := 0.04 -- Annual interest rate
def time_years := 0.75 -- Time in years (9 months)

-- Question and Answer
theorem initial_investment : ∃ P : ℝ, P = interest / (annual_rate * time_years) :=
by
  use 10000
  sorry

end initial_investment_l382_382132


namespace value_subtracted_3_times_number_eq_1_l382_382428

variable (n : ℝ) (v : ℝ)

theorem value_subtracted_3_times_number_eq_1 (h1 : n = 1.0) (h2 : 3 * n - v = 2 * n) : v = 1 :=
by
  sorry

end value_subtracted_3_times_number_eq_1_l382_382428


namespace augmented_matrix_of_system_l382_382790

theorem augmented_matrix_of_system :
  let eqn1 : list Rat := [2, 3, 1],
      eqn2 : list Rat := [1, -2, -1],
      system := [eqn1, eqn2]
  in (system = [[2, 3, 1], [1, -2, -1]]) :=
by
  sorry

end augmented_matrix_of_system_l382_382790


namespace student_attempted_sums_l382_382182

theorem student_attempted_sums (right wrong : ℕ) (h1 : wrong = 2 * right) (h2 : right = 12) : right + wrong = 36 := sorry

end student_attempted_sums_l382_382182


namespace solution_set_linear_inequalities_l382_382930

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382930


namespace max_length_of_cuts_is_1065_l382_382501

-- Define the initial conditions of the problem
def square_side_length : ℕ := 30
def num_parts : ℕ := 225
def part_area : ℕ := (square_side_length * square_side_length) / num_parts

-- Define the function to calculate the maximum length of cuts
def max_cut_length (side_length : ℕ) (parts : ℕ) (part_area : ℕ) : ℕ :=
  /- The total length of cuts maximized under given conditions -/
  (2250 - 4 * side_length) / 2

-- The main theorem statement incorporating the problem and its correct answer
theorem max_length_of_cuts_is_1065 :
  max_cut_length square_side_length num_parts part_area = 1065 :=
begin
  sorry
end

end max_length_of_cuts_is_1065_l382_382501


namespace angle_between_clock_hands_at_3_15_l382_382464

theorem angle_between_clock_hands_at_3_15 :
  let minute_degree := 90
  let hour_degree := 90 + (1 / 4 * 30)
  let angle := abs (hour_degree - minute_degree)
  in angle = 7.5 :=
by
  let minute_degree := 90
  let hour_degree := 90 + (1 / 4 * 30)
  let angle := abs (hour_degree - minute_degree)
  show angle = 7.5
  sorry

end angle_between_clock_hands_at_3_15_l382_382464


namespace vasya_strategy_divisible_by_9_l382_382396

theorem vasya_strategy_divisible_by_9 (d : Fin 6 → Fin 10) :
  (∑ i, (d i) + ∑ i, (9 - d i)) % 9 = 0 :=
by 
  simp [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, bit0, bit1, mul_comm]
  norm_num


end vasya_strategy_divisible_by_9_l382_382396


namespace dandelion_white_dandelions_on_saturday_l382_382151

theorem dandelion_white_dandelions_on_saturday :
  ∀ (existsMondayYellow MondayWhite WednesdayYellow WednesdayWhite : ℕ)
    (MondayTotal WednesdayTotal : ℕ)
    (MondayYellow = 20)
    (MondayWhite = 14)
    (MondayTotal = MondayYellow + MondayWhite)
    (WednesdayYellow = 15)
    (WednesdayWhite = 11)
    (WednesdayTotal = WednesdayYellow + WednesdayWhite),
  existsMondayYellow = MondayYellow → existsMondayWhite = MondayWhite →
  WednesdayTotal = 26 →
  (WednesdayTotal - MondayYellow) = 6 →
  WednesdayTotal - MondayYellow - MondayWhite = 6 →
  6 = 6 := 
begin
  intros,
  sorry
end

end dandelion_white_dandelions_on_saturday_l382_382151


namespace second_hand_travel_distance_l382_382823

theorem second_hand_travel_distance (r : ℝ) (minutes : ℝ) (π : ℝ) (h : r = 10 ∧ minutes = 45 ∧ π = Real.pi) : 
  (minutes / 60) * 60 * (2 * π * r) = 900 * π := 
by sorry

end second_hand_travel_distance_l382_382823


namespace number_of_sides_of_polygon_l382_382060

theorem number_of_sides_of_polygon (n : ℕ) (h : 3 * (n * (n - 3) / 2) - n = 21) : n = 6 :=
by sorry

end number_of_sides_of_polygon_l382_382060


namespace light_flash_time_l382_382515

/--
A light flashes every few seconds. In 3/4 of an hour, it flashes 300 times.
Prove that it takes 9 seconds for the light to flash once.
-/
theorem light_flash_time : 
  (3 / 4 * 60 * 60) / 300 = 9 :=
by
  sorry

end light_flash_time_l382_382515


namespace height_of_triangle_RSC_from_R_to_BC_l382_382456

theorem height_of_triangle_RSC_from_R_to_BC 
  (A B C : ℝ × ℝ)
  (hxA : A = (-2, 10))
  (hxB : B = (2, 0))
  (hxC : C = (10, 0))
  (R S : ℝ × ℝ)
  (hRS : R.1 = S.1)
  (hS_on_BC : S.2 = 0)
  (hR_on_AC : (5 * R.1 + 6 * R.2 = 50))
  (hArea_RSC : ½ * |R.2| * 8 = 24) :
  |R.2| = 6 := 
sorry

end height_of_triangle_RSC_from_R_to_BC_l382_382456


namespace number_of_white_dandelions_on_saturday_l382_382143

variables (D : Type) [decidable_eq D]

-- Define the states of dandelions
inductive DandelionState | Yellow | White | Dispersed

-- Define the life cycle of a dandelion
def life_cycle (d : D) (day : ℕ) : DandelionState :=
if day < 3 then DandelionState.Yellow
else if day = 3 then DandelionState.White
else DandelionState.Dispersed

-- Initial conditions on Monday
def yellow_on_monday : ℕ := 20
def white_on_monday : ℕ := 14

-- Initial conditions on Wednesday
def yellow_on_wednesday : ℕ := 15
def white_on_wednesday : ℕ := 11

-- Theorem stating the number of white dandelions on Saturday
theorem number_of_white_dandelions_on_saturday :
  let total_on_wednesday := yellow_on_wednesday + white_on_wednesday,
      new_dandelions_on_tuesday_wednesday := total_on_wednesday - yellow_on_monday,
      white_dandelions_on_saturday := new_dandelions_on_tuesday_wednesday
  in white_dandelions_on_saturday = 6 :=
sorry

end number_of_white_dandelions_on_saturday_l382_382143


namespace three_consecutive_edges_same_color_l382_382136

variables (V E : Type) [Fintype V] [Fintype E]
variables (polyhedron : E → V × V)
variables (degree : V → ℕ)
variables (coloring : E → ℕ)
variables (A : V)
variables (good_coloring : (E → ℕ) → Prop)

-- Conditions
def vertex_A_degree_5 : Prop := degree A = 5
def all_other_vertices_degree_3 : Prop := ∀ (v : V), v ≠ A → degree v = 3
def good_coloring_def : (E → ℕ) → Prop := λ f, ∀ (v : V), degree v = 3 → pairwise (λ e₁ e₂, f e₁ ≠ f e₂) (filter (λ e, v = (polyhedron e).fst ∨ v = (polyhedron e).snd) (finset.univ : finset E))
def number_good_colorings_not_divisible_by_5 : Prop := ¬ (∃ n : ℕ, n * 5 = finset.card {f // good_coloring f})

-- Statement to prove
theorem three_consecutive_edges_same_color :
  vertex_A_degree_5 degree A ∧
  all_other_vertices_degree_3 degree A ∧
  number_good_colorings_not_divisible_by_5 (λ f, good_coloring f) →
  ∃ (f : E → ℕ), good_coloring f ∧ ∃ (e1 e2 e3 : E), 
  ((polyhedron e1).fst = A ∨ (polyhedron e1).snd = A) ∧
  ((polyhedron e2).fst = A ∨ (polyhedron e2).snd = A) ∧
  ((polyhedron e3).fst = A ∨ (polyhedron e3).snd = A) ∧
  f e1 = f e2 ∧ f e2 = f e3 :=
sorry

end three_consecutive_edges_same_color_l382_382136


namespace eccentricity_of_ellipse_l382_382335

-- Definitions based on the given conditions
structure Ellipse (R : Type) [Real R] :=
  (a c : R) (semi_major_axis : a > 0) (distance_to_focus : c > 0) (max_distance : a + c = 2 * (a - c))

-- Define the eccentricity
def eccentricity {R : Type} [Real R] (e : Ellipse R) : R := e.c / e.a

-- Problem statement: Prove that the eccentricity is 1/3 given the conditions
theorem eccentricity_of_ellipse {R : Type} [Real R] (e : Ellipse R) (h : e.max_distance) : 
  eccentricity e = 1 / 3 :=
sorry

end eccentricity_of_ellipse_l382_382335


namespace evaluate_f_at_2_l382_382592

def f (t : ℝ) : ℝ := (t^2 - t) / (t^2 + 1)

theorem evaluate_f_at_2 : f 2 = 2 / 5 :=
by
  sorry

end evaluate_f_at_2_l382_382592


namespace quadrilateral_perimeter_ratio_l382_382527

theorem quadrilateral_perimeter_ratio (b : ℝ) (hb : b ≠ 0) :
  let A := (-b, -b),
      B := (b, -b),
      C := (-b, b),
      D := (b, b),
      line := λ (x : ℝ), -x / 3
  in (quadrilateral_perimeter A B C D line / b = (14 + real.sqrt 13) / 3) := sorry

end quadrilateral_perimeter_ratio_l382_382527


namespace number_of_oxygen_atoms_l382_382135

theorem number_of_oxygen_atoms (al_weight ph_weight o_weight total_weight : ℝ) (n : ℝ) 
  (h1 : al_weight = 27) 
  (h2 : ph_weight = 31)
  (h3 : o_weight = 16) 
  (h4 : total_weight = 122)
  (h5 : total_weight = al_weight + ph_weight + n * o_weight) :
  n = 4 :=
by {
  rw [h1, h2, h3] at h5,
  have h : 122 = 27 + 31 + n * 16, { exact h5 },
  norm_num at h,
  linarith,
}

end number_of_oxygen_atoms_l382_382135


namespace Peter_bought_green_notebooks_l382_382754

theorem Peter_bought_green_notebooks (total_notebooks : ℕ) (green_notebooks : ℕ) (black_notebook : ℕ) (pink_notebook : ℕ) (total_cost : ℕ) (black_cost : ℕ) (pink_cost : ℕ) :
  total_notebooks = 4 → black_notebook = 1 → pink_notebook = 1 → total_cost = 45 → black_cost = 15 → pink_cost = 10 → 
  total_notebooks = green_notebooks + black_notebook + pink_notebook → 
    (total_cost - (black_cost + pink_cost)) = green_notebooks * (total_cost - (black_cost + pink_cost)) / green_notebooks → 
    green_notebooks = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  sorry

end Peter_bought_green_notebooks_l382_382754


namespace four_fours_expressions_l382_382461

theorem four_fours_expressions :
  (4 * 4 + 4) / 4 = 5 ∧
  4 + (4 + 4) / 2 = 6 ∧
  4 + 4 - 4 / 4 = 7 ∧
  4 + 4 + 4 - 4 = 8 ∧
  4 + 4 + 4 / 4 = 9 :=
by
  sorry

end four_fours_expressions_l382_382461


namespace problem_solution_l382_382674

theorem problem_solution (x : ℝ) (h1 : x = 12) (h2 : 5 + 7 / x = some_number - 5 / x) : some_number = 6 := 
by
  sorry

end problem_solution_l382_382674


namespace exponent_difference_l382_382676

theorem exponent_difference (a b : ℤ) (h : 2 ^ a * 3 ^ b = 8 * 6 ^ 10) : b - a = -3 := by
  sorry

end exponent_difference_l382_382676


namespace solution_set_of_inequalities_l382_382960

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382960


namespace simplify_cot_15_add_tan_45_l382_382041

theorem simplify_cot_15_add_tan_45 :
  (Real.cot 15 + Real.tan 45) = Real.csc 15 := by
  sorry

end simplify_cot_15_add_tan_45_l382_382041


namespace minimum_sum_of_box_dimensions_l382_382238

theorem minimum_sum_of_box_dimensions :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end minimum_sum_of_box_dimensions_l382_382238


namespace solution_set_of_linear_inequalities_l382_382855

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382855


namespace solve_inequalities_l382_382885

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382885


namespace geometric_sequence_term_6_l382_382243

-- Define the geometric sequence conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

variables 
  (a : ℕ → ℝ) -- the geometric sequence
  (r : ℝ) -- common ratio, which is 2
  (h_r : r = 2)
  (h_pos : ∀ n, 0 < a n)
  (h_condition : a 4 * a 10 = 16)

-- The proof statement
theorem geometric_sequence_term_6 :
  a 6 = 2 :=
sorry

end geometric_sequence_term_6_l382_382243


namespace birds_remaining_on_fence_l382_382987

noncomputable def initial_birds : ℝ := 15.3
noncomputable def birds_flew_away : ℝ := 6.5
noncomputable def remaining_birds : ℝ := initial_birds - birds_flew_away

theorem birds_remaining_on_fence : remaining_birds = 8.8 :=
by
  -- sorry is a placeholder for the proof, which is not required
  sorry

end birds_remaining_on_fence_l382_382987


namespace proof_problem_l382_382227

section
open Real

/-- The parametric equation of a line passing through M(2,1) with inclination angle π/4. -/
def parametric_eq_line (t : ℝ) : ℝ × ℝ :=
(2 + (√2 / 2) * t, 1 + (√2 / 2) * t)

/-- The Cartesian equation of a circle given in polar coordinates by ρ = 4√2 sin(θ + π/4). -/
def cartesian_eq_circle (x y : ℝ) : Prop :=
x^2 + y^2 - 4*x - 4*y = 0

/-- The value of 1/|MA| + 1/|MB| given intersections A and B of line and circle. -/
def value_intersections (t1 t2 : ℝ) : ℝ :=
|t1 - t2| / |t1 * t2|

theorem proof_problem :
  (∀ t, parametric_eq_line t = (2 + (√2 / 2) * t, 1 + (√2 / 2) * t)) ∧
  (∀ x y, cartesian_eq_circle x y ↔ x^2 + y^2 - 4*x - 4*y = 0) ∧
  (∀ t1 t2, value_intersections t1 t2 = (√30 / 7)) :=
by
  split
  · intro t
    sorry
  split
  · intro x y
    sorry
  · intro t1 t2
    sorry
end

end proof_problem_l382_382227


namespace solution_set_inequalities_l382_382832

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382832


namespace car_speed_last_third_l382_382504

noncomputable section

def average_speed (d1 d2 d3 t1 t2 t3 : ℝ) : ℝ :=
  (d1 + d2 + d3) / (t1 + t2 + t3)

theorem car_speed_last_third
  (D : ℝ)
  (V : ℝ)
  (D3 : D/3)
  (t1 : D3 / 80)
  (t2 : D3 / 24)
  (t_avg : average_speed D3 D3 D3 t1 t2 (D3 / V) = 39.014778325123146) :
  V ≈ 44 := sorry

end car_speed_last_third_l382_382504


namespace prop_P_prop_Q_main_theorem_l382_382614

theorem prop_P (x : ℝ) : 0 < x ∧ x < 1 ↔ log 10 (x * (1 - x) + 1) > 0 := by
  sorry

theorem prop_Q (A B : ℝ) (h : ∠A > ∠B) : ∠A > ∠B → ¬(cos^2 (A / 2) < cos^2 (B / 2)) := by
  sorry

theorem main_theorem (x A B : ℝ) 
    (hP : 0 < x ∧ x < 1 ↔ log 10 (x * (1 - x) + 1) > 0)
    (hQ : ∠A > ∠B → ¬(cos^2 (A / 2) < cos^2 (B / 2))) :
    (hP ∧ ¬hQ) := by
  sorry

end prop_P_prop_Q_main_theorem_l382_382614


namespace more_bags_found_l382_382993

def bags_Monday : ℕ := 7
def bags_nextDay : ℕ := 12

theorem more_bags_found : bags_nextDay - bags_Monday = 5 := by
  -- Proof Skipped
  sorry

end more_bags_found_l382_382993


namespace range_of_m_l382_382632

open Set

def setM (m : ℝ) : Set ℝ := { x | x ≤ m }
def setP : Set ℝ := { x | x ≥ -1 }

theorem range_of_m (m : ℝ) (h : setM m ∩ setP = ∅) : m < -1 := sorry

end range_of_m_l382_382632


namespace total_signals_l382_382797
-- Step d) Rewrite the math proof problem in c) to a Lean 4 statement.


theorem total_signals (holes : Fin 4) (display : holes -> Bool) :
  (∀ i j, i ≠ j → display i = 1 → display j = 1 → |i - j| ≠ 1) →
  (∑ i, if display i then 1 else 0 = 2) →
  set.card {s : finset (Fin 4) | s.card = 2 ∧ (∀ i j ∈ s, i ≠ j → |i - j| ≠ 1)} = 12 :=
sorry

end total_signals_l382_382797


namespace incorrect_major_premise_l382_382793

noncomputable def Line := Type
noncomputable def Plane := Type

-- Conditions: Definitions
variable (b a : Line) (α : Plane)

-- Assumption: Line b is parallel to Plane α
axiom parallel_to_plane (p : Line) (π : Plane) : Prop

-- Assumption: Line a is in Plane α
axiom line_in_plane (l : Line) (π : Plane) : Prop

-- Define theorem stating the incorrect major premise
theorem incorrect_major_premise 
  (hb_par_α : parallel_to_plane b α)
  (ha_in_α : line_in_plane a α) : ¬ (parallel_to_plane b α → ∀ l, line_in_plane l α → b = l) := 
sorry

end incorrect_major_premise_l382_382793


namespace range_of_k_l382_382603

noncomputable theory

open Nat

def is_monotonically_increasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, 0 < n → a (n + 1) > a n

theorem range_of_k (k : ℤ) (a : ℕ → ℤ) (h : ∀ n : ℕ, 0 < n → a n = n ^ 2 - k * n) :
  is_monotonically_increasing a → k < 3 :=
by
  intros h_mono
  apply lt_of_not_ge
  intro h_ge
  have h' : 2 * 1 + 1 - k ≤ 0 := by sorry  -- To be proved
  sorry -- The remaining proof

end range_of_k_l382_382603


namespace closest_whole_number_l382_382232

theorem closest_whole_number :
  let expr := (4 * 10^150 + 4 * 10^152) / (6 * 10^151) in
  abs (expr - 6) < 0.5 :=
by
  let expr := (4 * 10^150 + 4 * 10^152) / (6 * 10^151)
  have h1 : expr = 101 / 15 := sorry
  have h2 : abs ((101 / 15) - 6) < 0.5 := by
    norm_num
  exact h2

end closest_whole_number_l382_382232


namespace scientific_notation_of_909_000_000_000_l382_382021

theorem scientific_notation_of_909_000_000_000 :
    ∃ (a : ℝ) (n : ℤ), 909000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 9.09 ∧ n = 11 := 
sorry

end scientific_notation_of_909_000_000_000_l382_382021


namespace find_interval_eq_half_l382_382584

open MeasureTheory

variables {f g : ℝ → ℝ} {μ : Measure ℝ}

noncomputable def exists_interval_eq_half (f g : ℝ → ℝ) (μ : Measure ℝ) : Prop :=
  ∃ I : Set ℝ, MeasurableSet I ∧ μ I = 1 / 2 ∧ Integral f μ I = 1 / 2 ∧ Integral g μ I = 1 / 2

theorem find_interval_eq_half 
  (hf : f ∈ Lp ℝ 1 μ) (hg : g ∈ Lp ℝ 1 μ)
  (hf_int : ∫ x in 0..1, f x ∂μ = 1)
  (hg_int : ∫ x in 0..1, g x ∂μ = 1)
  : exists_interval_eq_half f g μ := 
sorry

end find_interval_eq_half_l382_382584


namespace monthly_sales_quantity_expression_monthly_profit_reach_3600_l382_382505

noncomputable def linear_sales_function (x : ℝ) : ℝ := -30 * x + 960

theorem monthly_sales_quantity_expression :
  (∀ x, x = 20 → linear_sales_function x = 360) ∧ (∀ x, x = 30 → linear_sales_function x = 60) :=
by
  intros x hx; subst hx; unfold linear_sales_function;
  -- Proof steps would go here
  sorry

noncomputable def profit (x y : ℝ) : ℝ := (x - 10) * y

theorem monthly_profit_reach_3600 (x : ℝ) :
  (2 * x^2 -980*x + 9600 = 0) → profit x (linear_sales_function x) = 3600 :=
by
  intros h; unfold profit linear_sales_function;
  -- Proof steps would go here
  sorry

end monthly_sales_quantity_expression_monthly_profit_reach_3600_l382_382505


namespace binomial_sum_value_l382_382327

def binomial_expansion (x : ℝ) : ℝ :=
  (1 - 2 * x) ^ 2016
  
noncomputable def a_coeffs : Fin 2017 → ℝ := sorry
  
theorem binomial_sum_value :
  let a_0 := a_coeffs 0 in
  (a_0 + a_coeffs 1 + a_0 + a_coeffs 2 + ... + a_0 + a_coeffs 2016) = 2016 :=
sorry

end binomial_sum_value_l382_382327


namespace min_people_answer_l382_382497

-- Define the problem with the given conditions
namespace TestProblem

variable (people questions : ℕ)

-- There are 21 people and 15 questions
def num_people : ℕ := 21
def num_questions : ℕ := 15

-- Define the property that every pair of people have at least one correct answer in common
def pairwise_common_correct_answer (people : ℕ) (questions : ℕ) : Prop :=
  ∀ (p1 p2 : ℕ), p1 ≠ p2 → ∃ q : ℕ, q ≤ questions ∧ answered_correctly p1 q ∧ answered_correctly p2 q

-- Placeholder for the predicate indicating if a person correctly answered a question
def answered_correctly (p : ℕ) (q : ℕ) : Prop := sorry

-- Define the goal to find the minimum number of people that answered the most common question correctly
def min_people_most_correct (bound : ℕ) : Prop :=
  ∀ (p : ℕ), (∃ q ≤ questions, ∑ j (answered_correctly j q), j <= bound)

-- State the theorem
theorem min_people_answer (h : pairwise_common_correct_answer num_people num_questions) :
  ∃ bound : ℕ, min_people_most_correct 5 :=
sorry

end TestProblem

end min_people_answer_l382_382497


namespace students_walk_fraction_l382_382340

theorem students_walk_fraction
  (school_bus_fraction : ℚ := 1/3)
  (car_fraction : ℚ := 1/5)
  (bicycle_fraction : ℚ := 1/8) :
  (1 - (school_bus_fraction + car_fraction + bicycle_fraction) = 41/120) :=
by
  sorry

end students_walk_fraction_l382_382340


namespace area_of_circular_platform_l382_382518

theorem area_of_circular_platform (d : ℝ) (h : d = 2) : ∃ (A : ℝ), A = Real.pi ∧ A = π *(d / 2)^2 := by
  sorry

end area_of_circular_platform_l382_382518


namespace total_pencils_l382_382773

/-- The conditions defining the number of pencils Sarah buys each day. -/
def pencils_monday : ℕ := 20
def pencils_tuesday : ℕ := 18
def pencils_wednesday : ℕ := 3 * pencils_tuesday

/-- The hypothesis that the total number of pencils bought by Sarah is 92. -/
theorem total_pencils : pencils_monday + pencils_tuesday + pencils_wednesday = 92 :=
by
  -- calculations skipped
  sorry

end total_pencils_l382_382773


namespace points_form_ellipse_l382_382449

variables {n : ℕ}
variable  (h : ℝ)
variables (z : ℕ → ℂ) (w : ℕ → ℂ)
variable  [nontrivial z]
variable  (q : ℂ)
variables (k : ℕ)

-- Conditions from the problem
variables (H1 : ∀ i, 1 ≤ i ∧ i ≤ n → |z i| ≠ 1)
variables (H2 : ∀ i, 1 ≤ i ∧ i ≤ n → w i = z i + (1 / z i) + h)
variables (H3 : ∀ i, 1 ≤ i ∧ i ≤ n → z (i+1) = z 1 * q ^ i)
variable  (H4 : q ≠ 1 ∧ q ≠ -1)

-- Goal statement
theorem points_form_ellipse :
  ∃ (cx : ℝ) (rx ry : ℝ), 
    ∀ i, 1 ≤ i ∧ i ≤ n → 
    (let x := complex.re (w i)) in
    (let y := complex.im (w i)) in
    ((x - cx) ^ 2) / (rx ^ 2) + (y ^ 2) / (ry ^ 2) = 1 := 
sorry

end points_form_ellipse_l382_382449


namespace max_volume_of_sphere_in_cube_l382_382329

theorem max_volume_of_sphere_in_cube (a : ℝ) (h : a = 1) : 
  ∃ V, V = π / 6 ∧ 
        ∀ (r : ℝ), r = a / 2 →
        V = (4 / 3) * π * r^3 :=
by
  sorry

end max_volume_of_sphere_in_cube_l382_382329


namespace smallest_number_divisible_by_12_16_18_21_28_l382_382465

open Nat

def lcm_of_list (lst : List Nat) : Nat :=
  lst.foldl lcm 1

theorem smallest_number_divisible_by_12_16_18_21_28 :
  let lcm := lcm_of_list [12, 16, 18, 21, 28]
  ∃ x, (x - 4) % lcm = 0 ∧ ∀ y, (y - 4) % lcm = 0 → x ≤ y :=
  ∃ x, lcm_of_list [12, 16, 18, 21, 28] = 1008 ∧ x = 1012 :=
begin
  let lcm := lcm_of_list [12, 16, 18, 21, 28],
  show lcm = 1008 from rfl,
  use 1012,
  split,
  { sorry },
  { sorry }
end

end smallest_number_divisible_by_12_16_18_21_28_l382_382465


namespace geometric_sum_S5_l382_382352

theorem geometric_sum_S5 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) (h2 : (a 5 + a 7) / (a 2 + a 4) = 1 / 8) :
  (∑ i in Finset.range 5, a (i + 1)) = 31 / 16 := by
  sorry

end geometric_sum_S5_l382_382352


namespace difference_Q_R_l382_382535

variable (P Q R : ℝ) (x : ℝ)

theorem difference_Q_R (h1 : 11 * x - 5 * x = 12100) : 19 * x - 11 * x = 16133.36 :=
by
  sorry

end difference_Q_R_l382_382535


namespace graph_with_unique_simple_paths_is_tree_l382_382400

-- Definitions based on the conditions
def graph (V : Type u) := V → V → Prop

def is_connected (G : graph V) := 
  ∀ (u v : V), u ≠ v → ∃(p : list V), p.head = u ∧ p.reverse.head = v ∧ ∀ w : V, w ∈ p → G w (p.tail.head) ∧ G (p.tail.head) w

def has_unique_simple_path (G : graph V) := 
  ∀ (u v : V), u ≠ v → ∃!(p : list V), p.head = u ∧ p.reverse.head = v ∧ ∀ w : V, w ∈ p → G w (p.tail.head) ∧ G (p.tail.head) w

def is_tree (G : graph V) :=
  is_connected G ∧ ∀ (C : list V), (∀ w ∈ C, G w (C.tail.head)) → C.head = C.reverse.head → length C = 1

theorem graph_with_unique_simple_paths_is_tree {V : Type u} (G : graph V)
  (h : has_unique_simple_path G) : is_tree G := 
sorry

end graph_with_unique_simple_paths_is_tree_l382_382400


namespace range_of_x_l382_382733

noncomputable def f (x : ℝ) : ℝ := (5 / (x^2)) - (3 * (x^2)) + 2

theorem range_of_x :
  { x : ℝ | f 1 < f (Real.log x / Real.log 3) } = { x : ℝ | (1 / 3) < x ∧ x < 1 ∨ 1 < x ∧ x < 3 } :=
by
  sorry

end range_of_x_l382_382733


namespace ellipse_correct_equation_l382_382607

noncomputable def ellipse_equation (a b x y : ℝ) :=
  (5 * x^2 / 9) + (5 * y^2 / 4)

theorem ellipse_correct_equation (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a^2 - b^2 = 1) :
  let C := { (x, y) : ℝ × ℝ | ellipse_equation a b x y = 1 },
  let F := (1, 0),
  let sym_point := (3 / 5, 4 / 5),
  sym_point ∈ C → ellipse_equation a b 3 5 4 5 = 1 :=
by
  sorry

end ellipse_correct_equation_l382_382607


namespace counterexamples_to_prime_statement_l382_382555

def is_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

def digit_sum_to_five (n : ℕ) : Prop := 
  (n.digits.sum = 5) ∧ (∀ d ∈ n.digits, is_digit d)

noncomputable def count_non_prime_digit_sum_5 (s : Finset ℕ) : ℕ :=
  s.filter (λ n, digit_sum_to_five n ∧ ¬ Prime n).card

theorem counterexamples_to_prime_statement : 
  count_non_prime_digit_sum_5 {n : ℕ | digit_sum_to_five n} = 6 :=
sorry

end counterexamples_to_prime_statement_l382_382555


namespace sum_of_extreme_third_row_numbers_l382_382748

theorem sum_of_extreme_third_row_numbers : 
  let grid_size := 13
  let start_num := 100
  let end_num := 268
  let center := 7
  let third_row := 3
  ∀ (spiral_grid : List (List Nat)),
    (∃ center_num, center_num = start_num) ∧
    (spiral_grid.length = grid_size) ∧
    (∀ row, row.length = grid_size) ∧
    (∀ num, num ∈ spiral_grid.flatten → start_num ≤ num ∧ num ≤ end_num) ∧
    (∃ least_third_row, least_third_row = 107) ∧
    (∃ greatest_third_row, greatest_third_row = 119) →
    least_third_row + greatest_third_row = 226 := sorry

end sum_of_extreme_third_row_numbers_l382_382748


namespace white_dandelions_on_saturday_l382_382140

theorem white_dandelions_on_saturday 
  (yellow_monday : ℕ) (white_monday : ℕ)
  (yellow_wednesday : ℕ) (white_wednesday : ℕ)
  (life_cycle : ∀ d, (d.bloom == "yellow" ∧ d.age == 3) ∨ (d.bloom == "white" ∧ d.age == 4) ∨ (d.bloom == "dispersed" ∧ d.age == 5))
  (total_monday : yellow_monday = 20 ∧ white_monday = 14)
  (total_wednesday : yellow_wednesday = 15 ∧ white_wednesday = 11):
  let new_dandelions := 26 - 20,
      white_dandelions_saturday := new_dandelions
  in white_dandelions_saturday = 6 := by
  let dandelions_tuesday_wednesday := new_dandelions
  have h : white_dandelions_saturday = 6,
  from sorry
  exact h

end white_dandelions_on_saturday_l382_382140


namespace range_of_tan_squared_plus_tan_plus_one_l382_382577

noncomputable def range_of_y (y : ℝ) : Set ℝ := { y | ∃ x : ℝ, y = tan x ^ 2 + tan x + 1 ∧ ∀ k : ℤ, x ≠ k * π + π / 2 }

theorem range_of_tan_squared_plus_tan_plus_one :
  Set.range (λ x : ℝ, tan x ^ 2 + tan x + 1) = { y | y ≥ 3 / 4 } := by
  sorry

end range_of_tan_squared_plus_tan_plus_one_l382_382577


namespace solution_set_inequalities_l382_382828

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382828


namespace factorial_divisibility_l382_382006

theorem factorial_divisibility (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a.factorial + (a + b).factorial) ∣ (a.factorial * (a + b).factorial)) : a ≥ 2 * b + 1 :=
sorry

end factorial_divisibility_l382_382006


namespace ball_distribution_l382_382755

theorem ball_distribution (n : ℕ) (h : n ≥ 1) :
  ∃ (g : ℕ), g = 4^(n-1) ∧ 
  ∃ (A1 A2 A3 A4 : set ℕ), 
  ∃ (f : ℕ → set ℕ), 
  (∀ k, f(k) ⊆ {A1, A2, A3, A4}) ∧ 
  (A1.card ≡ 1 [MOD 2]) ∧ 
  (A2.card ≡ 0 [MOD 2]) ∧ 
  ∀ k ∈ A1 ∪ A2 ∪ A3 ∪ A4, 
  f(k).card = 1 ∧ 
  A1.card + A2.card + A3.card + A4.card = n :=
sorry -- Proof omitted

end ball_distribution_l382_382755


namespace solution_set_l382_382845

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382845


namespace amount_of_bill_is_1575_l382_382078

noncomputable def time_in_years := (9 : ℝ) / 12

noncomputable def true_discount := 189
noncomputable def rate := 16

noncomputable def face_value (TD : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (TD * 100) / (R * T)

theorem amount_of_bill_is_1575 :
  face_value true_discount rate time_in_years = 1575 := by
  sorry

end amount_of_bill_is_1575_l382_382078


namespace sum_elements_is_4_l382_382376

open Matrix

variable {α : Type*} [Field α]

def A (a b c d : α) : Matrix (Fin 3) (Fin 3) α :=
  ![[a, 2, b],
    [-3, 3, c],
    [d, 4, 1]]

def B (e f g h : α) : Matrix (Fin 3) (Fin 3) α :=
  ![[-1, e, 1],
    [f, 0, g],
    [2, h, -1]]

theorem sum_elements_is_4 (a b c d e f g h : α)
  (hAinv : A a b c d ⬝ B e f g h = 1) :
  a + b + c + d + e + f + g + h = 4 := by
  sorry

end sum_elements_is_4_l382_382376


namespace negation_of_universal_proposition_l382_382434

variable {ℝ : Type} [LinearOrderedField ℝ]

-- Defining the original proposition
def original_proposition (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

-- Statement to prove the negation of the original proposition
theorem negation_of_universal_proposition : 
  (¬ (∀ x : ℝ, original_proposition x)) ↔ (∃ x : ℝ, ¬ original_proposition x) :=
by
  sorry

end negation_of_universal_proposition_l382_382434


namespace averaging_property_of_entire_function_l382_382728

noncomputable def entire_function (f : ℂ → ℂ) := ∀ z, differentiable ℂ f ∧ ∀ n : ℕ, differentiable ℂ (λ z, iterated_deriv n f z)

variables {ξ η : ℝ} (f : ℂ → ℂ) [independent ξ η] [normal 0 1 ξ] [normal 0 1 η]

axiom E_finite : ∀ x : ℝ, E (abs (f (x + ξ + complex.I * η))) < ∞

theorem averaging_property_of_entire_function
  (h_f : entire_function f) :
  ∀ x : ℝ, f x = E (f (x + ξ + complex.I * η)) :=
sorry

end averaging_property_of_entire_function_l382_382728


namespace largest_multiple_of_fifteen_less_than_450_l382_382094

theorem largest_multiple_of_fifteen_less_than_450 : 
  ∃ (x : ℕ), x < 450 ∧ x % 15 = 0 ∧ ∀ (y : ℕ), y < 450 ∧ y % 15 = 0 → y ≤ x := 
  exists.intro 435 
  (and.intro (by norm_num) 
  (and.intro (by norm_num) 
  (by intros y h1 h2;
       have h3 : y ≤ 30 * 15, from le_of_lt (by norm_num1 [h1]),
       have h4 : 30 * 15 ≡ 0 [MOD 15], from nat.gcd_mod_right 30 15,
       have h5 : y / 15 < 30, from (by norm_num; exact nat.div_lt_self' h1),
       have h6 : y = y / 15 * 15, from (by norm_num [h1],
       have h7 : y / 15 ≤ 29, from nat.le_of_lt_sub_singleton h5,
       exact nat.mul_le_mul_right 15 h7)))

end largest_multiple_of_fifteen_less_than_450_l382_382094


namespace remaining_blocks_to_walk_l382_382542

noncomputable def total_blocks : ℕ := 11 + 6 + 8
noncomputable def walked_blocks : ℕ := 5

theorem remaining_blocks_to_walk : total_blocks - walked_blocks = 20 := by
  sorry

end remaining_blocks_to_walk_l382_382542


namespace correct_options_are_A_and_B_l382_382473

theorem correct_options_are_A_and_B :
  (let event_occurs (p: ℝ) (n: ℕ) := (1 - p)^n in
   let die_event_A_intersect_B (pA pB: ℝ) := 1 / 6 in
   let specific_red_light_prob (p: ℝ) (n: ℕ) := (1 - p)^(n-1) * p in
   let probability_union (PA PB PAB: ℝ) := PA + PB - PAB in
   event_occurs 0.9 10 > 0 ∧
   (die_event_A_intersect_B (3/6) (2/6) = (3/6) * (2/6)) ∧
   ¬(specific_red_light_prob (2/3) 3 = 2/3) ∧
   ¬(probability_union 0.5 0.5 0.25 ≠ 0.75)
  ) = true :=
by sorry

end correct_options_are_A_and_B_l382_382473


namespace velocity_at_t_one_l382_382289

-- Given position function s
def position (t : ℝ) : ℝ := t^2 + 1/t

-- Velocity function as the derivative of position function
def velocity (t : ℝ) : ℝ := 2*t - 1/(t^2)

-- Prove that the velocity at t = 1 is 1
theorem velocity_at_t_one : velocity 1 = 1 := 
by 
  unfold velocity
  norm_num
  sorry

end velocity_at_t_one_l382_382289


namespace bug_return_probability_twelfth_move_l382_382130

-- Conditions
def P : ℕ → ℚ
| 0       => 1
| (n + 1) => (1 : ℚ) / 3 * (1 - P n)

theorem bug_return_probability_twelfth_move :
  P 12 = 14762 / 59049 := by
sorry

end bug_return_probability_twelfth_move_l382_382130


namespace dandelion_white_dandelions_on_saturday_l382_382154

theorem dandelion_white_dandelions_on_saturday :
  ∀ (existsMondayYellow MondayWhite WednesdayYellow WednesdayWhite : ℕ)
    (MondayTotal WednesdayTotal : ℕ)
    (MondayYellow = 20)
    (MondayWhite = 14)
    (MondayTotal = MondayYellow + MondayWhite)
    (WednesdayYellow = 15)
    (WednesdayWhite = 11)
    (WednesdayTotal = WednesdayYellow + WednesdayWhite),
  existsMondayYellow = MondayYellow → existsMondayWhite = MondayWhite →
  WednesdayTotal = 26 →
  (WednesdayTotal - MondayYellow) = 6 →
  WednesdayTotal - MondayYellow - MondayWhite = 6 →
  6 = 6 := 
begin
  intros,
  sorry
end

end dandelion_white_dandelions_on_saturday_l382_382154


namespace sum_of_roots_l382_382367

theorem sum_of_roots (α β : ℝ) (h1 : α^2 - 4 * α + 3 = 0) (h2 : β^2 - 4 * β + 3 = 0) (h3 : α ≠ β) :
  α + β = 4 :=
sorry

end sum_of_roots_l382_382367


namespace sum_of_coefficients_eq_one_l382_382442

theorem sum_of_coefficients_eq_one :
  ∀ x y : ℤ, (x - 2 * y) ^ 18 = (1 - 2 * 1) ^ 18 → (x - 2 * y) ^ 18 = 1 :=
by
  intros x y h
  sorry

end sum_of_coefficients_eq_one_l382_382442


namespace solution_set_system_of_inequalities_l382_382943

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382943


namespace trigonometric_inequality_proof_l382_382032

noncomputable def trigonometric_inequality (x : ℝ) : Prop :=
  2 * real.sin (real.pi / 4 - real.sqrt 2 / 2) ^ 2 ≤ real.cos (real.sin x) - real.sin (real.cos x) ∧
  real.cos (real.sin x) - real.sin (real.cos x) ≤ 2 * real.sin (real.pi / 4 + real.sqrt 2 / 2) ^ 2

theorem trigonometric_inequality_proof (x : ℝ) : trigonometric_inequality x :=
sorry

end trigonometric_inequality_proof_l382_382032


namespace linear_inequalities_solution_l382_382873

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382873


namespace range_of_quadratic_function_l382_382437

theorem range_of_quadratic_function :
  ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), -x^2 - 4 * x + 1 ∈ Set.Icc (-11) (5) :=
by
  sorry

end range_of_quadratic_function_l382_382437


namespace linear_inequalities_solution_l382_382872

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382872


namespace number_of_white_dandelions_on_saturday_l382_382144

variables (D : Type) [decidable_eq D]

-- Define the states of dandelions
inductive DandelionState | Yellow | White | Dispersed

-- Define the life cycle of a dandelion
def life_cycle (d : D) (day : ℕ) : DandelionState :=
if day < 3 then DandelionState.Yellow
else if day = 3 then DandelionState.White
else DandelionState.Dispersed

-- Initial conditions on Monday
def yellow_on_monday : ℕ := 20
def white_on_monday : ℕ := 14

-- Initial conditions on Wednesday
def yellow_on_wednesday : ℕ := 15
def white_on_wednesday : ℕ := 11

-- Theorem stating the number of white dandelions on Saturday
theorem number_of_white_dandelions_on_saturday :
  let total_on_wednesday := yellow_on_wednesday + white_on_wednesday,
      new_dandelions_on_tuesday_wednesday := total_on_wednesday - yellow_on_monday,
      white_dandelions_on_saturday := new_dandelions_on_tuesday_wednesday
  in white_dandelions_on_saturday = 6 :=
sorry

end number_of_white_dandelions_on_saturday_l382_382144


namespace sum_inequality_l382_382594

-- Definitions
variables {ι : Type*} [linear_ordered_comm_ring ι]

def non_negative_reals (x : ι) := 0 ≤ x

def min_of_sequence (n : ℕ) (x : ι → ι) : ι :=
  finset.min' (finset.range n) (λ j, x j)

-- Theorem statement
theorem sum_inequality (x : ℕ → ι) (n : ℕ) (a : ι) (h_a : a = min_of_sequence n x)
  (h_nonneg : ∀ j, j < n → non_negative_reals (x j)) (h_wrap : x n = x 0) :
  (finset.sum (finset.range n) (λ j, (1 + x j) / (1 + x (nat.succ j % (n + 1)))) ≤ 
  n + (1 / (1 + a)^2) * (finset.sum (finset.range n) (λ j, (x j - a)^2))) :=
begin
  sorry
end

end sum_inequality_l382_382594


namespace subset_with_equal_sum_of_squares_l382_382377

theorem subset_with_equal_sum_of_squares (n : ℕ) (hn : n > 0) :
  ∃ (B : Finset ℕ), 
  B.card = 24 * n + 12 ∧ 
  (∑ b in B, b * b) = ∑ a in (Finset.range (48 * n + 24 + 1) \ B), a * a :=
sorry

end subset_with_equal_sum_of_squares_l382_382377


namespace solution_set_of_inequalities_l382_382953

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382953


namespace solution_set_of_inequalities_l382_382974

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382974


namespace order_of_magnitude_l382_382591

noncomputable def a : ℝ := Real.pi⁻²
noncomputable def b : ℝ := a^a
noncomputable def c : ℝ := a^a^a

theorem order_of_magnitude (a b c : ℝ) (h1 : a = Real.pi⁻²) (h2 : b = a^a) (h3 : c = a^a^a) : b > c ∧ c > a :=
by
  -- proof skipped, using sorry
  sorry

end order_of_magnitude_l382_382591


namespace isothermal_compression_work_l382_382138

variable (H h R : ℝ)
variable (p0 : ℝ := 103300)   -- convert kPa to Pa
variable (S : ℝ := Real.pi * R^2)
variable (V0 : ℝ := S * H)

noncomputable def F (x : ℝ) := (p0 * H / (H - x)) * S

noncomputable def work (h : ℝ) : ℝ := ∫ x in 0..h, F x

theorem isothermal_compression_work (H h R : ℝ) (H_pos : 0 < H) (h_pos : 0 < h) (R_pos : 0 < R) :
  work H h R = 97200 :=
by sorry

end isothermal_compression_work_l382_382138


namespace find_a_when_b_4_l382_382784

noncomputable def a_b_relation (a b : ℝ) : Prop :=
  a^3 * b^4 = 2000

theorem find_a_when_b_4 : ∀ (a b : ℝ), a = 5 ∧ b = 2 → a_b_relation 5 2 ∧ (b = 4 → a = 5 / 2) :=
by
  intros a b h
  have h_ab : a_b_relation a b := sorry
  have h_b4 : b = 4 → a = 5 / 2 := sorry
  exact ⟨h_ab, h_b4⟩

end find_a_when_b_4_l382_382784


namespace prime_squares_5000_9000_l382_382653

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end prime_squares_5000_9000_l382_382653


namespace solution_set_of_linear_inequalities_l382_382866

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382866


namespace exists_integers_m_n_l382_382723

theorem exists_integers_m_n (a b c p q r : ℝ) (h_a : a ≠ 0) (h_p : p ≠ 0) :
  ∃ (m n : ℤ), ∀ (x : ℝ), (a * x^2 + b * x + c = m * (p * x^2 + q * x + r) + n) := sorry

end exists_integers_m_n_l382_382723


namespace cost_price_of_article_l382_382536

-- Define the conditions and goal as a Lean 4 statement
theorem cost_price_of_article (M C : ℝ) (h1 : 0.95 * M = 75) (h2 : 1.25 * C = 75) : 
  C = 60 := 
by 
  sorry

end cost_price_of_article_l382_382536


namespace required_point_is_centroid_l382_382708

variables {A B C M : Type} [circumcenter M]

def triangle_area (A B C M : Type) : ℝ := sorry -- Assume a definition of area calculation.

theorem required_point_is_centroid
  (A B C : Type)
  (M : Type)
  (hM_inside_ABC : M ∈ interior (triangle A B C))
  : triangle_area A B M = triangle_area B C M ∧ triangle_area A C M = triangle_area A B M → M = centroid A B C := sorry

end required_point_is_centroid_l382_382708


namespace solution_set_inequalities_l382_382830

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382830


namespace solution_set_inequalities_l382_382835

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382835


namespace smallest_angle_l382_382433

theorem smallest_angle (x : ℝ) (h : 3 * x + 4 * x + 5 * x = 180) (h2 : 3 * x > 30) : 3 * x = 45 :=
  eq_of_not_ne (by
    intro h3
    have hcalc : 12 * x = 180 := by sorry -- Uses the summing of the angles
    have hx : x = 15 := by sorry -- Solve for x
    have smallest_angle : 3 * x = 45 := by sorry -- Calculate the smallest angle
    contradiction)

end smallest_angle_l382_382433


namespace complex_value_l382_382813

def letterValue (n : Nat) : Int :=
  let pattern := [2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3]
  pattern.get! (n % 26)

def wordValue (word : List Char) : Int :=
  word.foldl (fun accum c => accum + letterValue ((c.val - 'a'.val) % 26)) 0

theorem complex_value : wordValue ['c','o','m','p','l','e','x'] = 9 :=
by
  sorry

end complex_value_l382_382813


namespace garrison_initial_men_l382_382159

theorem garrison_initial_men (M : ℕ) (provisions_days : ℕ) (reinforcement : ℕ) (remaining_days_after_reinforcement : ℕ) 
    (total_days_before_reinforcement : ℕ) (days_after : ℕ) 
    (h1 : provisions_days = 54) 
    (h2 : total_days_before_reinforcement = 21) 
    (h3 : reinforcement = 1300)
    (h4 : remaining_days_after_reinforcement = 20)
    (h5 : days_after = (total_days_before_reinforcement - days_after)) 
    (h6 : M * days_after = (M + reinforcement) * remaining_days_after_reinforcement) : 
    M = 2000 :=
by
  have h7 : M * 33 = (M + 1300) * 20 := by rw [provisions_days, total_days_before_reinforcement, reinforcement, remaining_days_after_reinforcement, days_after] at h6
  sorry

end garrison_initial_men_l382_382159


namespace intersection_is_correct_l382_382316

namespace IntervalProofs

def setA := {x : ℝ | 3 * x^2 - 14 * x + 16 ≤ 0}
def setB := {x : ℝ | (3 * x - 7) / x > 0}

theorem intersection_is_correct :
  {x | 7 / 3 < x ∧ x ≤ 8 / 3} = setA ∩ setB :=
by
  sorry

end IntervalProofs

end intersection_is_correct_l382_382316


namespace positive_yuan_represents_income_l382_382325

def yuan_representation : Prop :=
  ∀ {x : Int}, x < 0 → (x = -150 → "expenditure of 150 yuan") ∧ x > 0 → (x = 200 → "income of 200 yuan")

theorem positive_yuan_represents_income :
  yuan_representation → "200 yuan represents an income of 200 yuan" :=
by
  sorry

end positive_yuan_represents_income_l382_382325


namespace part_length_proof_l382_382176

-- Define the scale length in feet and inches
def scale_length_ft : ℕ := 6
def scale_length_inch : ℕ := 8

-- Define the number of equal parts
def num_parts : ℕ := 4

-- Calculate total length in inches
def total_length_inch : ℕ := scale_length_ft * 12 + scale_length_inch

-- Calculate the length of each part in inches
def part_length_inch : ℕ := total_length_inch / num_parts

-- Prove that each part is 1 foot 8 inches long
theorem part_length_proof :
  part_length_inch = 1 * 12 + 8 :=
by
  sorry

end part_length_proof_l382_382176


namespace total_lambs_l382_382741

def num_initial_lambs : ℕ := 6
def num_baby_lambs_per_mother : ℕ := 2
def num_mothers : ℕ := 2
def traded_lambs : ℕ := 3
def extra_lambs : ℕ := 7

theorem total_lambs :
  num_initial_lambs + (num_baby_lambs_per_mother * num_mothers) - traded_lambs + extra_lambs = 14 :=
by
  sorry

end total_lambs_l382_382741


namespace solution_set_of_inequalities_l382_382977

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382977


namespace expression_decreased_72_22_percent_l382_382699

-- Define variables
variables (x y : ℝ)

-- Original expression
def original_expr (x y : ℝ) := x^2 * y^3

-- Decreased variables
def x' := 0.9 * x
def y' := 0.7 * y

-- Expression with decreased variables
def new_expr := (x' ^ 2) * (y' ^ 3)

-- Prove that the relative decrease is approximately 72.22%
theorem expression_decreased_72_22_percent :
  new_expr x y = 0.27783 * original_expr x y :=
by sorry

end expression_decreased_72_22_percent_l382_382699


namespace combined_tickets_l382_382541

-- Definitions for the initial conditions
def stuffedTigerPrice : ℝ := 43
def keychainPrice : ℝ := 5.5
def discount1 : ℝ := 0.20 * stuffedTigerPrice
def discountedTigerPrice : ℝ := stuffedTigerPrice - discount1
def ticketsLeftDave : ℝ := 55
def spentDave : ℝ := discountedTigerPrice + keychainPrice
def initialTicketsDave : ℝ := spentDave + ticketsLeftDave

def dinoToyPrice : ℝ := 65
def discount2 : ℝ := 0.15 * dinoToyPrice
def discountedDinoToyPrice : ℝ := dinoToyPrice - discount2
def ticketsLeftAlex : ℝ := 42
def spentAlex : ℝ := discountedDinoToyPrice
def initialTicketsAlex : ℝ := spentAlex + ticketsLeftAlex

-- Lean statement proving the combined number of tickets at the start
theorem combined_tickets {dave_alex_combined : ℝ} 
    (h1 : dave_alex_combined = initialTicketsDave + initialTicketsAlex) : 
    dave_alex_combined = 192.15 := 
by 
    -- Placeholder for the actual proof
    sorry

end combined_tickets_l382_382541


namespace solution_set_inequalities_l382_382831

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382831


namespace assistant_professor_pencils_l382_382540

theorem assistant_professor_pencils :
  ∀ (A B P : ℕ), 
    A + B = 7 →
    2 * A + P * B = 10 →
    A + 2 * B = 11 →
    P = 1 :=
by 
  sorry

end assistant_professor_pencils_l382_382540


namespace number_of_white_dandelions_on_saturday_l382_382146

variables (D : Type) [decidable_eq D]

-- Define the states of dandelions
inductive DandelionState | Yellow | White | Dispersed

-- Define the life cycle of a dandelion
def life_cycle (d : D) (day : ℕ) : DandelionState :=
if day < 3 then DandelionState.Yellow
else if day = 3 then DandelionState.White
else DandelionState.Dispersed

-- Initial conditions on Monday
def yellow_on_monday : ℕ := 20
def white_on_monday : ℕ := 14

-- Initial conditions on Wednesday
def yellow_on_wednesday : ℕ := 15
def white_on_wednesday : ℕ := 11

-- Theorem stating the number of white dandelions on Saturday
theorem number_of_white_dandelions_on_saturday :
  let total_on_wednesday := yellow_on_wednesday + white_on_wednesday,
      new_dandelions_on_tuesday_wednesday := total_on_wednesday - yellow_on_monday,
      white_dandelions_on_saturday := new_dandelions_on_tuesday_wednesday
  in white_dandelions_on_saturday = 6 :=
sorry

end number_of_white_dandelions_on_saturday_l382_382146


namespace cupcakes_left_l382_382734

def pack_count := 3
def cupcakes_per_pack := 4
def cupcakes_eaten := 5

theorem cupcakes_left : (pack_count * cupcakes_per_pack - cupcakes_eaten) = 7 := 
by 
  sorry

end cupcakes_left_l382_382734


namespace length_of_each_part_l382_382177

-- Definitions from the conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def number_of_parts : ℕ := 4

-- Proof statement
theorem length_of_each_part : total_length_in_inches / number_of_parts = 20 :=
by
  sorry

end length_of_each_part_l382_382177


namespace largest_expression_l382_382474

theorem largest_expression :
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  sorry

end largest_expression_l382_382474


namespace exists_triangle_sqrt_sqrt_inequality_l382_382824

variable {a b c : ℝ}

theorem exists_triangle_sqrt
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a + b > c)
  (h5 : b + c > a)
  (h6 : c + a > b) :
  (⟨ √a, √b, √c ⟩).1 + (⟨ √a, √b, √c ⟩).2 > (⟨ √a, √b, √c ⟩).3 ∧
  (⟨ √a, √b, √c ⟩).2 + (⟨ √a, √b, √c ⟩).3 > (⟨ √a, √b, √c ⟩).1 ∧
  (⟨ √a, √b, √c ⟩).3 + (⟨ √a, √b, √c ⟩).1 > (⟨ √a, √b, √c ⟩).2 := 
sorry

theorem sqrt_inequality
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a + b > c)
  (h5 : b + c > a)
  (h6 : c + a > b) :
  sqrt(a*b) + sqrt(b*c) + sqrt(c*a) ≤ a + b + c ∧ 
  a + b + c < 2 * (sqrt(a*b) + sqrt(b*c) + sqrt(c*a)) :=
sorry

end exists_triangle_sqrt_sqrt_inequality_l382_382824


namespace cos_neg_135_eq_l382_382552

noncomputable def cosine_neg_135 : ℝ :=
  Real.cos (Real.Angle.ofRealDegree (-135.0))

theorem cos_neg_135_eq :
  cosine_neg_135 = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_135_eq_l382_382552


namespace circle_intersection_l382_382611

-- Definitions of the circles and conditions
def circle_M : ℝ × ℝ → Prop := λ p, let (x, y) := p in x^2 + y^2 - 2 * a * y = 0
def circle_N : ℝ × ℝ → Prop := λ p, let (x, y) := p in (x - 1)^2 + (y - 1)^2 = 1

-- Given chord condition
axiom a_pos : a > 0
axiom chord_condition : (d : ℝ) (h_d : |a| / sqrt(2) = d) (h_chord : d^2 + (sqrt(2))^2 = a^2) : true

-- Prove the position relationship between circle M and circle N is "intersecting"
theorem circle_intersection :
  ∃ (a : ℝ), a > 0 → ∃ (d : ℝ), |a| / sqrt(2) = d ∧ d^2 + (sqrt(2))^2 = a^2 →
  let cx1 := 0 in let cy1 := a in let r1 := a in
  let cx2 := 1 in let cy2 := 1 in let r2 := 1 in
  let dist_centers := sqrt ((1 - 0)^2 + (1 - a)^2) in
  (dist_centers < r1 + r2) ∧ (dist_centers > |r1 - r2|) :=
by
  sorry

end circle_intersection_l382_382611


namespace negation_of_proposition_l382_382064

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_proposition_l382_382064


namespace compound_interest_principal_l382_382472

theorem compound_interest_principal (P r : ℝ) (h1 : 8800 = P * (1 + r)^2) (h2 : 9261 = P * (1 + r)^3) : 
  P ≈ 7945.67 := 
by 
  sorry

end compound_interest_principal_l382_382472


namespace cost_per_person_l382_382738

-- Given conditions
def total_cost := 16
def cost_beef_per_pound := 4
def amount_beef := 3
def cost_oil := 1
def num_people := 3

-- The proof goal stated as a theorem
theorem cost_per_person : 
  (total_cost - (amount_beef * cost_beef_per_pound + cost_oil)) / num_people = 1 := 
begin
  sorry
end

end cost_per_person_l382_382738


namespace certain_number_l382_382110

theorem certain_number (a b : ℕ) (n : ℕ) 
  (h1: a % n = 0) (h2: b % n = 0) 
  (h3: b = a + 9 * n)
  (h4: b = a + 126) : n = 14 :=
by
  sorry

end certain_number_l382_382110


namespace sara_total_spent_l382_382022

def ticket_cost : ℝ := 10.62
def num_tickets : ℕ := 2
def rent_cost : ℝ := 1.59
def buy_cost : ℝ := 13.95
def total_spent : ℝ := 36.78

theorem sara_total_spent : (num_tickets * ticket_cost) + rent_cost + buy_cost = total_spent := by
  sorry

end sara_total_spent_l382_382022


namespace base_number_of_equation_l382_382672

theorem base_number_of_equation (y : ℕ) (b : ℕ) (h1 : 16 ^ y = b ^ 14) (h2 : y = 7) : b = 4 := 
by 
  sorry

end base_number_of_equation_l382_382672


namespace disk_tangency_after_rotations_l382_382796

/-- A structure representing a circular clock face and a rolling disk. -/
structure ClockAndDisk where
  clock_radius : ℝ
  disk_radius : ℝ
  initial_tangent_position : ℕ -- Representing the initial tangency position as hours (12)
  initial_arrow_orientation : ℝ -- Representing the initial arrow orientation in degrees

/-- Proof problem: If a disk with a radius of 5 cm, starting at the 
    12 o'clock position on a clock with a radius of 30 cm, rolls 
    clockwise around the clock face without slipping and makes 
    three complete rotations, it will next be tangent at the 6 o'clock 
    position when the arrow on it points directly upwards. -/
theorem disk_tangency_after_rotations : 
  ∀ (cd : ClockAndDisk), 
    cd.clock_radius = 30 ∧ 
    cd.disk_radius = 5 ∧ 
    cd.initial_tangent_position = 12 ∧ 
    cd.initial_arrow_orientation = 0 → 
    let rotations := 3 
    let disk_rotations := rotations * (cd.clock_radius / cd.disk_radius).toNat
    let final_angle := (disk_rotations * 360) % 360
    final_angle = 180 → 
    (cd.initial_tangent_position + final_angle / 30 - 12) % 12 = 6 :=
by
  intros cd h
  sorry

end disk_tangency_after_rotations_l382_382796


namespace cosine_double_angle_l382_382337

variable {A B C : Real}
variable {k : Real} (hk : k > 0)
variable (h1 : sin A / sin (B) = 2 / 3) (h2 : sin B / sin (C) = 3 / 4)

theorem cosine_double_angle {α β γ : Real} : 
  (sin α / sin β = 2 / 3) → 
  (sin β / sin γ = 3 / 4) → 
  cos (2 * γ) = -7 / 8 := 
by 
  sorry

end cosine_double_angle_l382_382337


namespace complex_problem_l382_382617

noncomputable def i : ℂ := complex.I

theorem complex_problem : 
  (1 - i)^2 - (4 + 2 * i) / (1 - 2 * i) - 4 * i ^ 2014 = -4 - 4 * i := 
by
  sorry

end complex_problem_l382_382617


namespace probability_five_dice_same_l382_382098

-- Define a function that represents the probability problem
noncomputable def probability_all_dice_same : ℚ :=
  (1 / 6) * (1 / 6) * (1 / 6) * (1 / 6)

-- The main theorem to state the proof problem
theorem probability_five_dice_same : probability_all_dice_same = 1 / 1296 :=
by
  sorry

end probability_five_dice_same_l382_382098


namespace count_primes_with_squares_in_range_l382_382664

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end count_primes_with_squares_in_range_l382_382664


namespace length_of_each_part_l382_382178

-- Definitions from the conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def number_of_parts : ℕ := 4

-- Proof statement
theorem length_of_each_part : total_length_in_inches / number_of_parts = 20 :=
by
  sorry

end length_of_each_part_l382_382178


namespace min_perimeter_is_676_l382_382458

-- Definitions and conditions based on the problem statement
def equal_perimeter (a b c : ℕ) : Prop :=
  2 * a + 14 * c = 2 * b + 16 * c

def equal_area (a b c : ℕ) : Prop :=
  7 * Real.sqrt (a^2 - 49 * c^2) = 8 * Real.sqrt (b^2 - 64 * c^2)

def base_ratio (b : ℕ) : ℕ := b * 8 / 7

theorem min_perimeter_is_676 :
  ∃ a b c : ℕ, equal_perimeter a b c ∧ equal_area a b c ∧ base_ratio b = a - b ∧ 
  2 * a + 14 * c = 676 :=
sorry

end min_perimeter_is_676_l382_382458


namespace equal_segments_l382_382071

structure Square (a : ℝ) :=
  (A B C D : ℝ × ℝ)
  (O : ℝ × ℝ)
  (side_length : ℝ)
  (center_origin : O = (0, 0))
  (vertices : A = (-a, a) ∧ B = (a, a) ∧ C = (a, -a) ∧ D = (-a, -a))

structure IsoscelesTriangle (A B : ℝ × ℝ) :=
  (E : ℝ × ℝ)
  (AE_eq_BE : dist A E = dist B E)

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

structure GeometrySetup (a : ℝ) :=
  (square : Square a)
  (isosceles_triangle : IsoscelesTriangle ((-a, a)) ((a, a)))
  (M : ℝ × ℝ)
  (O : ℝ × ℝ)
  (K : ℝ × ℝ)
  (midpoint_property : M = midpoint (-a, a) (IsoscelesTriangle.E isosceles_triangle))
  (center_property : O = (0, 0))
  (intersection : ¬ collinear {((0, 0), M, IsoscelesTriangle.E isosceles_triangle)})

theorem equal_segments (a : ℝ) (setup : GeometrySetup a) : 
  dist (IsoscelesTriangle.E setup.isosceles_triangle) setup.K = 
  dist setup.K setup.O :=
sorry

end equal_segments_l382_382071


namespace count_primes_between_71_and_95_l382_382645

theorem count_primes_between_71_and_95 : 
  let primes := [71, 73, 79, 83, 89, 97] in
  let filtered_primes := primes.filter (λ p, 71 < p ∧ p < 95) in
  filtered_primes.length = 5 := 
by 
  sorry

end count_primes_between_71_and_95_l382_382645


namespace theater_attendance_l382_382530

-- Given conditions
def cost_per_adult := 0.60
def cost_per_child := 0.25
def total_receipts := 140.0
def num_children := 80

theorem theater_attendance :
  ∃ (A : ℝ), cost_per_adult * A + cost_per_child * num_children = total_receipts ∧ A + num_children = 280 :=
by
  sorry

end theater_attendance_l382_382530


namespace average_score_15_percent_is_100_l382_382671

variables (x : ℝ)
variables (pct1 pct2 pct3 avg1 avg2 avg3 : ℝ)

-- Conditions
def conditions := 
  (pct1 = 0.15) ∧
  (pct2 = 0.50) ∧
  (pct3 = 0.35) ∧
  (avg2 = 78) ∧
  (avg3 = 63) ∧
  ( (pct1 * x) + (pct2 * avg2) + (pct3 * avg3) = 76.05 )

-- Proof that x (average score of 15% of the class) is 100%
theorem average_score_15_percent_is_100 (h : conditions) : x = 100 :=
by
  sorry

end average_score_15_percent_is_100_l382_382671


namespace solution_set_system_of_inequalities_l382_382939

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382939


namespace boy_lap_time_l382_382636

def side_length : ℝ := 50 -- meters
def sandy_section_length : ℝ := 20 -- meters
def normal_section_length : ℝ := side_length - sandy_section_length -- 30 meters
def flat_speed : ℝ := 9 * 1000 / 3600 -- 2.5 m/s
def sandy_speed : ℝ := flat_speed * 0.75 -- 1.875 m/s
def hurdle_slowdown_small : ℝ := 2 -- seconds for 0.5-meter hurdles
def hurdle_slowdown_large : ℝ := 3 -- seconds for 1-meter hurdles
def num_small_hurdles : ℕ := 2
def num_large_hurdles : ℕ := 2
def turn_slowdown : ℝ := 2 -- seconds per 110-degree turn
def num_turns : ℕ := 4

def time_flat_section : ℝ := normal_section_length / flat_speed -- 12 seconds
def time_sandy_section : ℝ := sandy_section_length / sandy_speed -- 10.67 seconds
def time_small_hurdles : ℝ := num_small_hurdles * hurdle_slowdown_small -- 4 seconds
def time_large_hurdles : ℝ := num_large_hurdles * hurdle_slowdown_large -- 6 seconds
def time_hurdles_per_side : ℝ := time_small_hurdles + time_large_hurdles -- 10 seconds
def time_turns_total : ℝ := num_turns * turn_slowdown -- 8 seconds

def time_per_side : ℝ := time_flat_section + time_sandy_section + time_hurdles_per_side -- 32.67 seconds
def time_all_sides : ℝ := 4 * time_per_side -- 130.68 seconds
def total_time : ℝ := time_all_sides + time_turns_total -- 138.68 seconds

theorem boy_lap_time : total_time = 138.68 := by
  rw [total_time, time_all_sides, time_per_side, 
      time_flat_section, time_sandy_section, 
      time_hurdles_per_side, time_turns_total,
      flat_speed, sandy_speed, hurdle_slowdown_small,
      hurdle_slowdown_large, num_small_hurdles,
      num_large_hurdles, turn_slowdown, num_turns]
  -- Further calculation simplification
  sorry

end boy_lap_time_l382_382636


namespace third_podcast_length_correct_l382_382406

def first_podcast_length : ℕ := 45
def fourth_podcast_length : ℕ := 60
def next_podcast_length : ℕ := 60
def total_drive_time : ℕ := 360

def second_podcast_length := 2 * first_podcast_length

def total_time_other_than_third := first_podcast_length + second_podcast_length + fourth_podcast_length + next_podcast_length

theorem third_podcast_length_correct :
  total_drive_time - total_time_other_than_third = 105 := by
  -- Proof goes here
  sorry

end third_podcast_length_correct_l382_382406


namespace Mandy_older_than_Jackson_l382_382735

variable (M J A : ℕ)

-- Given conditions
variables (h1 : J = 20)
variables (h2 : A = (3 * J) / 4)
variables (h3 : (M + 10) + (J + 10) + (A + 10) = 95)

-- Prove that Mandy is 10 years older than Jackson
theorem Mandy_older_than_Jackson : M - J = 10 :=
by
  sorry

end Mandy_older_than_Jackson_l382_382735


namespace solve_inequalities_l382_382887

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382887


namespace clara_age_l382_382460

theorem clara_age (x : ℕ) (n m : ℕ) (h1 : x - 2 = n^2) (h2 : x + 3 = m^3) : x = 123 :=
by sorry

end clara_age_l382_382460


namespace complement_union_complement_l382_382012

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

-- The proof problem
theorem complement_union_complement : (U \ (M ∪ N)) = {1, 6} := by
  sorry

end complement_union_complement_l382_382012


namespace num_valid_functions_l382_382717

noncomputable def is_valid_function (f: Fin 12 → Fin 12) : Prop :=
  ∀ i, (f i).val - i.val % 12 ∉ {0, 3, 6, 9}

theorem num_valid_functions :
  ∃ fset : Finset (Fin 12 → Fin 12), 
    (∀ f ∈ fset, is_valid_function f ∧ Function.Injective f) ∧ fset.card = 55392 :=
sorry

end num_valid_functions_l382_382717


namespace find_f_comp_l382_382593

def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then real.pi else 0

theorem find_f_comp : f (f (f (-2))) = real.pi + 1 :=
by
  rw [f, if_neg (by norm_num : ¬(-2 > 0)), if_neg (by norm_num : -2 ≠ 0)]
  rw [f, if_neg (by norm_num : ¬(0 > 0)), if_pos rfl]
  rw [f, if_pos (by linarith : real.pi > 0)]
  norm_num
  sorry

end find_f_comp_l382_382593


namespace probability_at_least_60_cents_heads_l382_382785

theorem probability_at_least_60_cents_heads :
  let outcomes := ([true, false].product ([true, false].product ([true, false].product ([true, false].product [true, false])))) in
  let values := [50, 10, 10, 5, 5] in
  let value (result : list bool) := result.zip values |>.map (λ ⟨head, value⟩ => if head then value else 0) |>.sum in
  let favorable := outcomes.filter (λ outcome => value outcome ≥ 60) in
  (favorable.length / outcomes.length : ℚ) = 9 / 32 := sorry

end probability_at_least_60_cents_heads_l382_382785


namespace boat_speed_in_still_water_l382_382691

/-- Given a boat's speed along the stream and against the stream, prove its speed in still water. -/
theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11)
  (h2 : b - s = 5) : b = 8 :=
sorry

end boat_speed_in_still_water_l382_382691


namespace perpendicular_line_x_intercept_l382_382092

theorem perpendicular_line_x_intercept :
  (∃ x : ℚ, x_intercept_of_perpendicular_line 4 5 15 (-3) = x) :=
by
  use (12 / 5 : ℚ)
  sorry

def x_intercept_of_perpendicular_line (a b c y_intercept : ℚ) : ℚ :=
let slope := -a / b in
let perp_slope := -1 / slope in
let perp_y_intercept := y_intercept in
(perp_y_intercept * 4 / 5).inv

end perpendicular_line_x_intercept_l382_382092


namespace james_running_speed_is_correct_l382_382711

noncomputable def james_running_speed_rounded_to_nearest_hundredth
  (bike_distance : ℝ) 
  (bike_speed_const : ℝ)
  (run_distance : ℝ)
  (run_speed_const : ℝ)
  (total_time_minutes : ℝ)
  (break_time_minutes : ℝ) : ℝ :=
let physical_activity_time_hours := (total_time_minutes - break_time_minutes) / 60 in
let eq := (bike_distance / (3 * bike_speed_const + 1) + run_distance / (bike_speed_const + 2)) = physical_activity_time_hours in
if h : ∃ (x : ℝ), eq then 
  rounded_speed := (classical.some h) + run_speed_const
else
  0

theorem james_running_speed_is_correct : james_running_speed_rounded_to_nearest_hundredth 30 5.9859 10 2 180 10 ≈ 7.99 :=
sorry

end james_running_speed_is_correct_l382_382711


namespace john_total_distance_l382_382482

theorem john_total_distance : 
  ∀ (rate : ℕ) (time1 : ℕ) (time2 : ℕ), 
  rate = 55 ∧ time1 = 2 ∧ time2 = 3 → 
  (rate * time1 + rate * time2) = 275 := 
by 
  intros rate time1 time2 h1; 
  cases h1 with h_rate h_times;
  cases h_times with h_time1 h_time2;
  rw [h_rate, h_time1, h_time2];
  norm_num;
  sorry

end john_total_distance_l382_382482


namespace problem_statement_l382_382010

noncomputable section

open Real

variable {f : ℝ → ℝ} (hf : ContinuousOn f (Icc 0 1)) (hf_bij : Bijective f) (hf0 : f 0 = 0)

theorem problem_statement (α : ℝ) (hα : 0 ≤ α) :
  (α + 2) * ∫ x in 0..1, x^α * (f x + (f⁻¹ x)) ≤ 2 :=
sorry

end problem_statement_l382_382010


namespace cosine_of_negative_135_l382_382550

theorem cosine_of_negative_135 : Real.cos (-(135 * Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end cosine_of_negative_135_l382_382550


namespace solution_set_system_of_inequalities_l382_382944

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382944


namespace area_of_triangle_l382_382756

theorem area_of_triangle (p : ℝ) (h_p : 0 < p ∧ p < 10) : 
    let C := (0, p)
    let O := (0, 0)
    let B := (10, 0)
    (1/2) * 10 * p = 5 * p := 
by
  sorry

end area_of_triangle_l382_382756


namespace inequality_problem_l382_382776

theorem inequality_problem (x : ℝ) (hx : 0 < x) : 
  1 + x ^ 2018 ≥ (2 * x) ^ 2017 / (1 + x) ^ 2016 := 
by
  sorry

end inequality_problem_l382_382776


namespace proof_correct_answers_l382_382191

-- We introduce definitions for the laws and principles needed
constant second_law_of_thermodynamics (heat_transfer feasible: Prop) : Prop
-- This represents the laws of thermodynamics stating certain conditions to be true

constant ideal_gas_law (P V T C : ℝ) : Prop
-- represents the relationship PV/T = C

constant brownian_motion (particles_liquid : Type) (T intensity : ℝ) : Prop
-- represents the dependency of Brownian motion intensity on temperature

-- The statements to verify
constant option_A_incorrect : Prop
constant option_B_correct : Prop
constant option_C_incorrect : Prop
constant option_D_correct : Prop

theorem proof_correct_answers (heat_transfer feasible: Prop) (P V T C : ℝ) (particles_liquid : Type) (T_ex intensity_ex : ℝ) :
  second_law_of_thermodynamics heat_transfer feasible →
  ideal_gas_law P V T C →
  brownian_motion particles_liquid T_ex intensity_ex →
  option_A_incorrect ∧ option_B_correct ∧ option_C_incorrect ∧ option_D_correct :=
by
  intros
  split; try { sorry }

end proof_correct_answers_l382_382191


namespace cost_per_person_l382_382739

-- Given conditions
def total_cost := 16
def cost_beef_per_pound := 4
def amount_beef := 3
def cost_oil := 1
def num_people := 3

-- The proof goal stated as a theorem
theorem cost_per_person : 
  (total_cost - (amount_beef * cost_beef_per_pound + cost_oil)) / num_people = 1 := 
begin
  sorry
end

end cost_per_person_l382_382739


namespace silvia_path_shorter_by_50_percent_l382_382715

def square_side_length : ℝ := 2

def silvia_path (side_length : ℝ) : ℝ :=
  Real.sqrt (2 * side_length ^ 2)

def jerry_path (side_length : ℝ) : ℝ :=
  2 + 1 + (2 + 1 / Real.sqrt 2)

def percentage_difference (jerry_path silvia_path : ℝ) : ℝ :=
  ((jerry_path - silvia_path) / jerry_path) * 100

theorem silvia_path_shorter_by_50_percent :
  percentage_difference (jerry_path square_side_length) (silvia_path square_side_length) = 50 :=
by
  sorry

end silvia_path_shorter_by_50_percent_l382_382715


namespace light_intensity_drop_l382_382381

theorem light_intensity_drop (k : ℝ) (x : ℕ) (H1 : 0.9^x < 1/4) : x ≥ 14 :=
by
  sorry

end light_intensity_drop_l382_382381


namespace line_CO_intersects_segment_AB_at_C₁_l382_382375

variables (A B C : ℝ^3)
variables (m₁ m₂ mₛ : ℝ)

def C₁ : ℝ^3 := (m₁ * A + m₂ * B) / (m₁ + m₂)

noncomputable def O : ℝ^3 := (m₁ * A + m₂ * B + mₛ * C) / (m₁ + m₂ + mₛ)

theorem line_CO_intersects_segment_AB_at_C₁ :
  ∃ t : ℝ, C₁ = t * O + (1 - t) * C :=
sorry

end line_CO_intersects_segment_AB_at_C₁_l382_382375


namespace votes_cast_for_winning_candidate_l382_382485

-- Define the conditions
def two_candidates_in_election := true
def winner_received_62_percent : Prop := ∃ (V : ℝ), 0 < V ∧ winner_votes = 0.62 * V
def winner_won_by_408_votes : Prop := ∃ (V : ℝ), 0 < V ∧ (winner_votes - loser_votes = 408)
def loser_votes := λ (V : ℝ), 0.38 * V
def winner_votes := λ (V : ℝ), 0.62 * V

-- Define the statement to prove the correct answer
theorem votes_cast_for_winning_candidate : 
  winner_received_62_percent ∧ winner_won_by_408_votes → 
  ∃ (V : ℝ), 0 < V ∧ winner_votes V = 1054 := 
by 
  intros h,
  sorry

end votes_cast_for_winning_candidate_l382_382485


namespace complement_intersection_l382_382379

open Set -- Open the Set namespace

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2})
variable (B : Set ℝ := {x | x ≤ -1 ∨ x > 2})

theorem complement_intersection :
  (U \ B) ∩ A = {x | x = 0 ∨ x = 1 ∨ x = 2} :=
by
  sorry -- Proof not required as per the instructions

end complement_intersection_l382_382379


namespace total_number_of_emeralds_l382_382162

variables (D E R : ℕ)
variables (box1 box2 box3 box4 box5 box6 : ℕ)

-- Condition: Total number of rubies exceeds the total number of diamonds by 15
def rubies_exceed_diamonds_by_15 := R = D + 15

-- Condition: Sum of the number of precious stones in the boxes is 39
def total_stones_is_39 := box1 + box2 + box3 + box4 + box5 + box6 = 39
def total_sum_of_stones := D + E + R

-- Given that the sum of the number of precious stones in the boxes should match the sum of the labels from boxes
def total_stones_sum_correct := total_sum_of_stones = box1 + box2 + box3 + box4 + box5 + box6

theorem total_number_of_emeralds :
  rubies_exceed_diamonds_by_15 D R →
  total_stones_is_39 box1 box2 box3 box4 box5 box6 →
  total_stones_sum_correct D E R box1 box2 box3 box4 box5 box6 →
  E = 12 := 
sorry

end total_number_of_emeralds_l382_382162


namespace subtract_29_after_46_l382_382781

theorem subtract_29_after_46 (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end subtract_29_after_46_l382_382781


namespace smallest_special_integer_l382_382168

noncomputable def is_special (N : ℕ) : Prop :=
  N > 1 ∧ 
  (N % 8 = 1) ∧ 
  (2 * 8 ^ Nat.log N (8) / 2 > N / 8 ^ Nat.log N (8)) ∧ 
  (N % 9 = 1) ∧ 
  (2 * 9 ^ Nat.log N (9) / 2 > N / 9 ^ Nat.log N (9))

theorem smallest_special_integer : ∃ (N : ℕ), is_special N ∧ N = 793 :=
by 
  use 793
  sorry

end smallest_special_integer_l382_382168


namespace range_of_independent_variable_l382_382068

noncomputable def range_of_x : (x : ℝ) → Prop := x - 1 ≥ 0

theorem range_of_independent_variable (x : ℝ) :
  range_of_x x ↔ x ≥ 1 :=
by
  sorry

end range_of_independent_variable_l382_382068


namespace solve_for_x_l382_382044

namespace MathProof

theorem solve_for_x (x : ℝ) (h : (1/3) + (1/x) + (1/(x^2)) = 1) : 
  x = (3 + sqrt 33) / 4 ∨ x = (3 - sqrt 33) / 4 :=
by
  sorry

end MathProof

end solve_for_x_l382_382044


namespace simplify_expression_l382_382778

/-- Theorem stating that simplifying the given complex expression results in 111 --/
theorem simplify_expression : 8^0.25 * 42 + (32 * Real.sqrt 3)^6 + log 3 2 * log 2 (log 3 27) = 111 := by
  sorry

end simplify_expression_l382_382778


namespace order_of_abc_l382_382560

noncomputable def a : ℝ := Real.log 3 / Real.log 4
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log (4/3) / Real.log (3/4)

theorem order_of_abc : b > a ∧ a > c := by
  sorry

end order_of_abc_l382_382560


namespace mary_more_candy_initially_l382_382744

-- Definitions of the conditions
def Megan_initial_candy : ℕ := 5
def Mary_candy_after_addition : ℕ := 25
def additional_candy_Mary_adds : ℕ := 10

-- The proof problem statement
theorem mary_more_candy_initially :
  (Mary_candy_after_addition - additional_candy_Mary_adds) / Megan_initial_candy = 3 :=
by
  sorry

end mary_more_candy_initially_l382_382744


namespace direct_proportion_point_l382_382333

theorem direct_proportion_point (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : y₁ = k * x₁) (hx₁ : x₁ = -1) (hy₁ : y₁ = 2) (hx₂ : x₂ = 1) (hy₂ : y₂ = -2) 
  : y₂ = k * x₂ := 
by
  -- sorry will skip the proof
  sorry

end direct_proportion_point_l382_382333


namespace weight_of_3_moles_of_CaI2_is_881_64_l382_382096

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
noncomputable def weight_3_moles_CaI2 : ℝ := 3 * molar_mass_CaI2

theorem weight_of_3_moles_of_CaI2_is_881_64 :
  weight_3_moles_CaI2 = 881.64 :=
by sorry

end weight_of_3_moles_of_CaI2_is_881_64_l382_382096


namespace limit_S_eq_half_l382_382205

noncomputable def S (n : ℕ) := ∑ k in Finset.range (n + 1), 1 / ((2 * k + 1) * (2 * k + 3))

theorem limit_S_eq_half : tendsto S at_top (𝓝 (1 / 2)) :=
sorry

end limit_S_eq_half_l382_382205


namespace third_number_in_product_is_six_l382_382445

-- Given Definitions
def volume_of_cube (a : ℕ) : ℕ := a^3
def material_volume (x : ℕ) : ℕ := 12 * 18 * x
def total_cubes (n : ℕ) (a : ℕ) : ℕ := n * volume_of_cube a

-- Theorem to prove
theorem third_number_in_product_is_six :
  ∀ x, total_cubes 48 3 = material_volume x → x = 6 :=
begin
  intros x h,
  sorry
end

end third_number_in_product_is_six_l382_382445


namespace trigonometric_identity_l382_382258

noncomputable def alpha : ℝ := sorry

axiom tan_alpha_is_2 : Real.tan alpha = 2

theorem trigonometric_identity :
  Real.sin (2 * alpha) / (Real.sin(alpha)^2 + Real.sin(alpha) * Real.cos(alpha) - Real.cos(2 * alpha) - 1) = 1 :=
by
  have h1 := tan_alpha_is_2
  sorry

end trigonometric_identity_l382_382258


namespace value_of_x_minus_y_l382_382668

theorem value_of_x_minus_y 
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : 3 * x - y = 8) :
  x - y = 3 := by
  sorry

end value_of_x_minus_y_l382_382668


namespace h_transform_l382_382627

noncomputable def f : ℝ → ℝ
| x => if -4 ≤ x ∧ x ≤ 0 then -x - 1
      else if 0 ≤ x ∧ x ≤ 6 then sqrt (9 - (x - 3)^2) - 3
      else if 6 ≤ x ∧ x ≤ 7 then 3 * (x - 6)
      else 0

noncomputable def h (x : ℝ) : ℝ := f (5 - x)

theorem h_transform (x : ℝ) : 
  h(x) = f(5 - x) := by
  sorry

end h_transform_l382_382627


namespace Nell_initial_cards_l382_382750

theorem Nell_initial_cards (n : ℕ) (h1 : n - 136 = 106) : n = 242 := 
by
  sorry

end Nell_initial_cards_l382_382750


namespace solve_inequalities_l382_382884

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382884


namespace count_primes_with_squares_in_range_l382_382660

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end count_primes_with_squares_in_range_l382_382660


namespace sum_of_angles_of_spatial_quadrilateral_leq_360_l382_382403

variables (A B C D : Point)

-- Assume that the points A, B, C, and D form a spatial quadrilateral, i.e., they are not coplanar.
def is_spatial_quadrilateral (A B C D : Point) : Prop :=
  ¬ coplanar A B C D

-- Assume some angles properties for the spatial quadrilateral.
variables 
(h1 : ∠ A B C < ∠ A B D + ∠ D B C)
(h2 : ∠ A D C < ∠ A D B + ∠ B D C)

-- Sum of angles in a triangle is 180 degrees.
axiom triangle_angle_sum (A B C : Point) : ∠ A B C + ∠ B A C + ∠ C A B = 180

theorem sum_of_angles_of_spatial_quadrilateral_leq_360 
  (h1 : ∠ A B C < ∠ A B D + ∠ D B C) 
  (h2 : ∠ A D C < ∠ A D B + ∠ B D C)
  (h3 : is_spatial_quadrilateral A B C D):
  ∠ A B C + ∠ A D C + ∠ B A D + ∠ B C D ≤ 360 :=
by
  sorry

end sum_of_angles_of_spatial_quadrilateral_leq_360_l382_382403


namespace brad_running_speed_l382_382746

/-- Maxwell leaves his home and walks toward Brad's house at 3 km/h, while Brad leaves his home
and runs toward Maxwell's house. The distance between their homes is 36 km. They meet in the middle
after Maxwell has traveled 12 km. Prove that Brad's running speed is 1.5 km/h. -/
theorem brad_running_speed :
  ∀ (b : ℝ), (∀ (m : ℝ), m = 3) → (∀ (d : ℝ), d = 36) → (∀ (mt : ℝ), mt = 12) →
  (36 / 2 - 12) / (12 / 3) = b → 
  b = 1.5 :=
by
  intros b m m_def d d_def mt mt_def speed_def
  rw [m_def, d_def, mt_def] at *
  simp_rw [div_eq_mul_inv, mul_comm] at *
  have : 18 - 12 = 6 := by norm_num
  rw this at speed_def
  have : 12 / 3 = 4 := by norm_num
  rw this at speed_def
  simp at speed_def
  exact speed_def

end brad_running_speed_l382_382746


namespace collinear_intersections_l382_382758

variables {P : Type} [linear_ordered_add_comm_group P]

structure Point :=
(x : P) (y : P)

structure Line := 
(a : P) (b : P) (c : P) -- Represents line ax + by + c = 0

def symmetric (l : Line) (P₁ P₂ : Point) : Prop :=
  l.a * (P₁.x + P₂.x) + l.b * (P₁.y + P₂.y) + 2 * l.c = 0

variables (A A1 B B1 C C1 N : Point) (l : Line)
variables (hA : symmetric l A A1) (hB : symmetric l B B1) (hC : symmetric l C C1)
variables (hN : l.a * N.x + l.b * N.y + l.c = 0)

theorem collinear_intersections :
  exists (P₁ P₂ P₃ : Point),
    (line A N = line B1 C1 ∧
     line B N = line A1 C1 ∧
     line C N = line A1 B1) ∧
    collinear P₁ P₂ P₃ :=
begin
  sorry
end

end collinear_intersections_l382_382758


namespace train_speed_is_252_144_l382_382185

/-- Train and pedestrian problem setup -/
noncomputable def train_speed (train_length : ℕ) (cross_time : ℕ) (man_speed_kmph : ℕ) : ℝ :=
  let man_speed_mps := (man_speed_kmph : ℝ) * 1000 / 3600
  let relative_speed_mps := (train_length : ℝ) / (cross_time : ℝ)
  let train_speed_mps := relative_speed_mps - man_speed_mps
  train_speed_mps * 3600 / 1000

theorem train_speed_is_252_144 :
  train_speed 500 7 5 = 252.144 := by
  sorry

end train_speed_is_252_144_l382_382185


namespace exists_four_numbers_with_property_l382_382553

theorem exists_four_numbers_with_property :
  ∀ (numbers : Finset ℕ), numbers.card = 64 ∧ ∀ n ∈ numbers, n ≤ 2012 →
  ∃ (a b c d ∈ numbers), (a + b - c - d) % 2013 = 0 :=
by
  sorry

end exists_four_numbers_with_property_l382_382553


namespace range_of_a_l382_382626

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem range_of_a (a : ℝ) :
  (∃ x, a < x ∧ x < 6 - a^2 ∧ ∀ y, a < y ∧ y < 6 - a^2 → f(x) ≤ f(y)) →
  -2 ≤ a ∧ a < 1 :=
sorry

end range_of_a_l382_382626


namespace equation_of_ellipse_prove_parallel_MN_CD_l382_382268

noncomputable def ellipse : Type := {x : ℝ × ℝ // (x.1^2 / 9) + (x.2^2 / 4) = 1}

theorem equation_of_ellipse
  (a b : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_gt_b : a > b)
  (h_eccentricity : (Real.sqrt 5) / 3 = λ x : ℝ, Real.sqrt (1 - (b^2 / a^2)))
  (h_ac : 2 * b = 4) :
  (a = 3 ∧ b = 2 ∧ ∀ x y : ℝ, (x^2 / 9) + (y^2 / 4) = 1) := sorry

theorem prove_parallel_MN_CD
  (P : ellipse)
  (A B C D : ellipse)
  (h_A : A = (⟨(0, 2), by sorry⟩ : ellipse))
  (h_B : B = (⟨(-3, 0), by sorry⟩ : ellipse))
  (h_C : C = (⟨(0, -2), by sorry⟩ : ellipse))
  (h_D : D = (⟨(3, 0), by sorry⟩ : ellipse))
  (h_P_in_first_quadrant : 0 < P.1.1 ∧ 0 < P.1.2)
  (h_PD_intersects_BC : ∃ M : ellipse, line_intersects P D B C M)
  (h_PA_intersects_y_neg2 : ∃ N : ellipse, line_intersects_AT P A (0, -2) N) :
  parallel MN CD := sorry

end equation_of_ellipse_prove_parallel_MN_CD_l382_382268


namespace num_ways_allocate_friends_l382_382665

theorem num_ways_allocate_friends : ∃ (n : ℕ), n = 65536 ∧ 
  (∀ friends (teams : ℕ → ℕ), friends = 8 ∧ (∀ x, x < 4 → teams x ≤ friends) → 
  (n = 4^friends)) :=
by
  simp [Nat.pow]

sorry -- Proof is omitted

end num_ways_allocate_friends_l382_382665


namespace domain_and_range_of_transformed_function_l382_382621

-- Given conditions: domain and range of f(x)
variables {f : ℝ → ℝ}

axiom domain_of_f : ∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≠ none
axiom range_of_f : ∀ x, -2 ≤ f x ∧ f x ≤ 3

theorem domain_and_range_of_transformed_function :
  (∀ x, -2 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 3 → f(2*x - 1) ≠ none) ∧
  (∀ y, -2 ≤ y ∧ y ≤ 3 → ∃ x, f(2*x - 1) = y) :=
by sorry

end domain_and_range_of_transformed_function_l382_382621


namespace sum_of_distances_l382_382600

variables (X : ℝ × ℝ × ℝ) (vertices : list (ℝ × ℝ × ℝ))
def is_vertex (v : ℝ × ℝ × ℝ) : Prop :=
  v = (0, 0, 0) ∨ v = (0, 0, 1) ∨ v = (0, 1, 0) ∨ v = (0, 1, 1) ∨
  v = (1, 0, 0) ∨ v = (1, 0, 1) ∨ v = (1, 1, 0) ∨ v = (1, 1, 1)

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2 + (q.3 - p.3) ^ 2)

theorem sum_of_distances (hX : X ∈ set.univ) : 
  (∑ v in vertices.filter is_vertex, distance X v) ≥ 4 * real.sqrt 3 := 
begin
  sorry
end

end sum_of_distances_l382_382600


namespace prime_squares_5000_9000_l382_382650

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end prime_squares_5000_9000_l382_382650


namespace solution_in_quadrant_I_l382_382003

theorem solution_in_quadrant_I (k x y : ℝ) (h1 : 2 * x - y = 5) (h2 : k * x^2 + y = 4) (h4 : x > 0) (h5 : y > 0) : k > 0 :=
sorry

end solution_in_quadrant_I_l382_382003


namespace problem_solution_exists_l382_382315

theorem problem_solution_exists {x : ℝ} :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ), 
    1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 =
    a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 +
    a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
    a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7)
  → a_2 = 56 :=
sorry

end problem_solution_exists_l382_382315


namespace quadratic_inequality_solution_l382_382564

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 48*x + 500 ≤ 0} = set.Icc (24 - 2 * Real.sqrt 19) (24 + 2 * Real.sqrt 19) :=
sorry

end quadratic_inequality_solution_l382_382564


namespace cos_x_leq_cos_y_l382_382589

theorem cos_x_leq_cos_y (ε : ℝ) (hx : ε > 0) (x y : ℝ) (hx_interval : x ∈ Ioo (-π / 4) (π / 4)) (hy_interval : y ∈ Ioo (-π / 4) (π / 4)) (h_eq : exp (x + ε) * sin y = exp y * sin x) : 
  cos x ≤ cos y := 
sorry

end cos_x_leq_cos_y_l382_382589


namespace eccentricity_of_hyperbola_l382_382629

variable {a b c x y : ℝ}

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem eccentricity_of_hyperbola
  (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (F1 F2 P Q : ℝ × ℝ)
  (on_hyperbola : hyperbola a b P.1 P.2 ∧ hyperbola a b Q.1 Q.2)
  (line_through_F2 : F2.1 = x ∧ F2.2 = y)
  (perpendicular_PF1_PQ : ∀ (PF1 PQ : ℝ), PF1 = PQ)
  (dist_PF1_EQ_PQ : ∀ (PF1 PQ : ℝ), |PF1| = |PQ|)
  (angle_PQF1_45 : ∠PQF1 = 45)
  (dist_QF1_4a : |QF1| = 4*a)
  (dist_QF2_2a : |QF2| = 2*a)
  (dist_F1F2_2c : |F1 - F2| = 2*c) :
  eccentricity_of_hyperbola a b c = sqrt(5 - 2*sqrt(2)) :=
sorry

end eccentricity_of_hyperbola_l382_382629


namespace incenter_coordinates_l382_382280

noncomputable def x_y_sum (A B C : Point) (AC BC AB : ℝ) (I : Point) : ℝ :=
  let x := 2 / (2 + 3 + 4)
  let y := 4 / (2 + 3 + 4)
  x + y

theorem incenter_coordinates (A B C I : Point) (AC BC AB : ℝ) :
  AC = 2 → BC = 3 → AB = 4 → 
  I = (convex_combination B C (2 / (2 + 3 + 4)) (4 / (2 + 3 + 4))) →
  x_y_sum A B C AC BC AB I = 2 / 3 :=
sorry

end incenter_coordinates_l382_382280


namespace find_x_solution_l382_382423

noncomputable def find_x (x y : ℝ) (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : Prop := 
  x = (3 + Real.sqrt 17) / 2

theorem find_x_solution (x y : ℝ) 
(h1 : x - y^2 = 3) 
(h2 : x^2 + y^4 = 13) 
(hx_pos : 0 < x) 
(hy_pos : 0 < y) : 
  find_x x y h1 h2 :=
sorry

end find_x_solution_l382_382423


namespace area_of_DEF_l382_382353

/--
In triangle ABC, let D, E, and F be the midpoints of sides AB, BC, and CA, respectively.
Given that the area of triangle ABC is 36 square units, we prove that the area of triangle DEF is 9 square units.
-/
theorem area_of_DEF (A B C : Point) (D E F : Point)
  (hD : midpoint A B = D)
  (hE : midpoint B C = E)
  (hF : midpoint C A = F)
  (area_ABC : area ABC = 36) :
  area DEF = 9 :=
sorry

end area_of_DEF_l382_382353


namespace speed_of_man_in_still_water_l382_382517

-- Define the parameters and conditions
def speed_in_still_water (v_m : ℝ) (v_s : ℝ) : Prop :=
    (v_m + v_s = 5) ∧  -- downstream condition
    (v_m - v_s = 7)    -- upstream condition

-- The theorem statement
theorem speed_of_man_in_still_water : 
  ∃ v_m v_s : ℝ, speed_in_still_water v_m v_s ∧ v_m = 6 := 
by
  sorry

end speed_of_man_in_still_water_l382_382517


namespace log2_n_eq_1178_l382_382234

-- Definitions and assumptions (conditions)
variables (teams : ℕ) (games : ℕ) (m n : ℕ)
variables (h_teams : teams = 50)
variables (h_games : games = (nat.choose 50 2))
variables (h_rel_prime : nat.coprime m n)
variables (h_probability : (m : ℚ) / n = (fact 50) / (2^games))

-- The main goal
theorem log2_n_eq_1178 : teams = 50 → games = (nat.choose 50 2) → nat.coprime m n →
  (m : ℚ) / n = (fact 50) / (2^games) → 
  int.log2 n = 1178 :=
by
  -- sorry is used to indicate that the proof is omitted
  sorry

end log2_n_eq_1178_l382_382234


namespace numbers_identification_l382_382387

-- Definitions
def is_natural (n : ℤ) : Prop := n ≥ 0
def is_integer (n : ℤ) : Prop := True

-- Theorem
theorem numbers_identification :
  (is_natural 0 ∧ is_natural 2 ∧ is_natural 6 ∧ is_natural 7) ∧
  (is_integer (-15) ∧ is_integer (-3) ∧ is_integer 0 ∧ is_integer 4) :=
by
  sorry

end numbers_identification_l382_382387


namespace linear_inequalities_solution_l382_382880

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382880


namespace remainder_of_150_div_k_l382_382582

theorem remainder_of_150_div_k (k : ℕ) (hk : k > 0) (h1 : 90 % (k^2) = 10) :
  150 % k = 2 := 
sorry

end remainder_of_150_div_k_l382_382582


namespace Robin_hair_length_l382_382770

variable (initial_length : ℕ) (growth : ℕ) (cut : ℕ)

theorem Robin_hair_length :
  initial_length = 14 →
  growth = 8 →
  cut = 20 →
  initial_length + growth - cut = 2 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end Robin_hair_length_l382_382770


namespace sub_two_three_l382_382206

theorem sub_two_three : 2 - 3 = -1 := 
by 
  sorry

end sub_two_three_l382_382206


namespace moles_of_H2O_combined_l382_382240

theorem moles_of_H2O_combined (mole_NH4Cl mole_NH4OH : ℕ) (reaction : mole_NH4Cl = 1 ∧ mole_NH4OH = 1) : 
  ∃ mole_H2O : ℕ, mole_H2O = 1 :=
by
  sorry

end moles_of_H2O_combined_l382_382240


namespace avg_weight_class_l382_382988

-- Definitions based on the conditions
def students_section_A : Nat := 36
def students_section_B : Nat := 24
def avg_weight_section_A : ℝ := 30.0
def avg_weight_section_B : ℝ := 30.0

-- The statement we want to prove
theorem avg_weight_class :
  (avg_weight_section_A * students_section_A + avg_weight_section_B * students_section_B) / (students_section_A + students_section_B) = 30.0 := 
by
  sorry

end avg_weight_class_l382_382988


namespace smallest_pillage_count_is_two_l382_382397

-- Defining the conditions
def somy_pillage (k: ℤ) : ℤ := 28 * 3^k
def lia_pillage (j: ℤ) : ℤ := 82 * 3^j

-- Statement of the problem
theorem smallest_pillage_count_is_two :
  ∃ (k1 k2 j1 j2: ℤ), (somy_pillage k1 + somy_pillage k2 = lia_pillage j1 + lia_pillage j2) ∧ ∀n ≤ 1, ∃ (k j: ℤ), somy_pillage k ≠ lia_pillage j :=
sorry

end smallest_pillage_count_is_two_l382_382397


namespace arctan_sum_of_roots_l382_382069

theorem arctan_sum_of_roots (u v w : ℝ) (h1 : u + v + w = 0) (h2 : u * v + v * w + w * u = -10) (h3 : u * v * w = -11) :
  Real.arctan u + Real.arctan v + Real.arctan w = π / 4 :=
by
  sorry

end arctan_sum_of_roots_l382_382069


namespace range_of_independent_variable_l382_382067

noncomputable def range_of_x : (x : ℝ) → Prop := x - 1 ≥ 0

theorem range_of_independent_variable (x : ℝ) :
  range_of_x x ↔ x ≥ 1 :=
by
  sorry

end range_of_independent_variable_l382_382067


namespace problem1_problem2_l382_382310
open Real

/-- Given conditions: -/
def exp_a (a : ℝ) : Prop := 2^a = 3
def exp_b (b : ℝ) : Prop := 2^b = 5
def exp_c (c : ℝ) : Prop := 2^c = 75

/-- First problem: Find the value of 2^(c + b - a) given the conditions: -/
theorem problem1 (a b c : ℝ) (h1 : exp_a a) (h2 : exp_b b) (h3 : exp_c c) : 2^(c + b - a) = 125 :=
sorry

/-- Second problem: Prove that a = c - 2b given the conditions: -/
theorem problem2 (a b c : ℝ) (h1 : exp_a a) (h2 : exp_b b) (h3 : exp_c c) : a = c - 2b :=
sorry

end problem1_problem2_l382_382310


namespace Doug_age_l382_382766

theorem Doug_age (Q J D : ℕ) (h1 : Q = J + 6) (h2 : J = D - 3) (h3 : Q = 19) : D = 16 := by
  sorry

end Doug_age_l382_382766


namespace triangle_side_lengths_l382_382338

theorem triangle_side_lengths
  (a : ℝ)
  (B : ℝ)
  (S : ℝ)
  (h1 : a = 1)
  (h2 : B = π / 4)  -- 45 degrees in radians
  (h3 : S = 2) :
  let C := 4 * Real.sqrt 2
  let b := 5 in
  C = 4 * Real.sqrt 2 ∧ b = 5 :=
by
  sorry

end triangle_side_lengths_l382_382338


namespace solution_set_of_inequalities_l382_382959

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382959


namespace solution_set_system_of_inequalities_l382_382950

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382950


namespace solution_set_linear_inequalities_l382_382918

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382918


namespace count_primes_with_squares_in_range_l382_382661

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end count_primes_with_squares_in_range_l382_382661


namespace correct_option_B_l382_382192

-- Definitions based on the given conditions
def lateral_surfaces_of_prism_are_quadrilaterals : Prop :=
  ∀ (P : Prism), lateral_surfaces P are quadrilaterals

def cube_and_cuboid_are_special_cases_of_quadrangular_prisms : Prop :=
  ∀ (P : Prism), (P is a cube ∨ P is a cuboid) → P is a quadrangular prism

def not_all_solids_can_be_unfolded_into_planar_figures : Prop :=
  ∀ (S : Solid), ¬(S can be unfolded into a planar figure)

def lateral_edges_of_prism_are_equal : Prop :=
  ∀ (P : Prism), lateral_edges P are equal

-- The problem statement to prove B is correct given the conditions
theorem correct_option_B :
  lateral_surfaces_of_prism_are_quadrilaterals →
  cube_and_cuboid_are_special_cases_of_quadrangular_prisms →
  not_all_solids_can_be_unfolded_into_planar_figures →
  lateral_edges_of_prism_are_equal →
  cube_and_cuboid_are_special_cases_of_quadrangular_prisms :=
by
  intros
  assumption

end correct_option_B_l382_382192


namespace sequence_a31_value_l382_382631

theorem sequence_a31_value 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h₀ : a 1 = 0) 
  (h₁ : ∀ n, a (n + 1) = a n + b n) 
  (h₂ : b 15 + b 16 = 15)
  (h₃ : ∀ m n : ℕ, (b n - b m) = (n - m) * (b 2 - b 1)) :
  a 31 = 225 :=
by
  sorry

end sequence_a31_value_l382_382631


namespace find_circle_eq_find_a_value_l382_382598

noncomputable def center : ℝ × ℝ := (2, 0)
def radius : ℝ := 5
def line_tangent (x y : ℝ) : Prop := 4 * x + 3 * y - 33 = 0
def circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25
def perpendicular {T : Type*} [inner_product_space ℝ T] (u v : T) : Prop :=
  ⟪u, v⟫ = (0 : ℝ)

theorem find_circle_eq
  (hx : is_integer (fst center) )
  (ht : line_tangent (fst center) (snd center) )
: (circle (fst center) (snd center) )
:= sorry

theorem find_a_value
  (h : ∀ A B : ℝ × ℝ, (circle (fst A) (snd A) )  → (circle (fst B) (snd B) ) 
     → let u := (fst A - fst center, snd A - snd center) in
        let v := (fst B - fst center, snd B - snd center) in
          perpendicular u v):
  a = 1 ∨ a = -73/17  
:= sorry

end find_circle_eq_find_a_value_l382_382598


namespace solution_set_of_inequalities_l382_382968

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382968


namespace right_triangle_hypotenuse_l382_382174

theorem right_triangle_hypotenuse {a b c : ℝ} 
  (h1: a + b + c = 60) 
  (h2: a * b = 96) 
  (h3: a^2 + b^2 = c^2) : 
  c = 28.4 := 
sorry

end right_triangle_hypotenuse_l382_382174


namespace solution_set_l382_382853

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382853


namespace solution_set_of_inequalities_l382_382954

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382954


namespace find_length_AE_l382_382270

noncomputable theory

variables {A B C D E : Type} [Point ABC]

-- Definitions based on given conditions
def is_isosceles_triangle (ABC : Triangle) := ABC.AB = ABC.BC
def point_on_ray_beyond (BA : Ray) (A E : Point) := ∃ t > 1, E = BA.origin + t • BA.direction
def ∠ADC (D A C : Point) : ℝ := 60
def ∠AEC (A E C : Point) : ℝ := 60
def AD (A D : Point) : ℝ := 13
def CE (C E : Point) : ℝ := 13

def BC (B C : Point) : ℝ := 9

theorem find_length_AE (T : Triangle)
  (h1 : is_isosceles_triangle T)
  (h2 : point_on_ray_beyond T.BA T.A E)
  (h3 : point_on_segment T.BC D)
  (h4 : ∠ T.D T.A T.C = 60)
  (h5 : ∠ T.A E T.C = 60)
  (h6 : distance T.A T.D = 13)
  (h7 : distance T.C E = 13)
  (h8 : distance T.D T.C = 9) : 
  distance T.A E = 4 :=
  sorry

end find_length_AE_l382_382270


namespace units_digit_of_147_pow_is_7_some_exponent_units_digit_l382_382468

theorem units_digit_of_147_pow_is_7 (n : ℕ) : (147 ^ 25) % 10 = 7 % 10 :=
by
  sorry

theorem some_exponent_units_digit (n : ℕ) (hn : n % 4 = 2) : ((147 ^ 25) ^ n) % 10 = 9 :=
by
  have base_units_digit := units_digit_of_147_pow_is_7 25
  sorry

end units_digit_of_147_pow_is_7_some_exponent_units_digit_l382_382468


namespace round_to_hundredth_l382_382462

theorem round_to_hundredth (x : ℝ) (h : x = 3.1415926) : Float.round (x * 100) / 100 = 3.14 := by
  sorry

end round_to_hundredth_l382_382462


namespace probability_no_defective_pencils_l382_382344

theorem probability_no_defective_pencils : 
  ∀ (total_pencils: ℕ) (defective_pencils: ℕ) (selected_pencils: ℕ), 
  total_pencils = 10 → defective_pencils = 2 → selected_pencils = 3 →
  let non_defective_pencils := total_pencils - defective_pencils in 
  (nat.choose non_defective_pencils selected_pencils) / (nat.choose total_pencils selected_pencils) = 7 / 15 :=
by 
  intros total_pencils defective_pencils selected_pencils h_total h_defective h_selected;
  let non_defective_pencils := total_pencils - defective_pencils;
  have h_combo_total : nat.choose total_pencils selected_pencils = 120 := sorry;
  have h_combo_non_defective : nat.choose non_defective_pencils selected_pencils = 56 := sorry;
  exact sorry

end probability_no_defective_pencils_l382_382344


namespace units_digit_of_quotient_l382_382563

theorem units_digit_of_quotient :
  let n := 1993,
      u4 := 4^n % 10,
      u5 := 5^n % 10,
      u_sum := (u4 + u5) % 10,
      quotient := (u_sum / 3) in
  u4 = 4 ∧ u5 = 5 ∧ u_sum = 9 → quotient % 10 = 3 :=
by
  intros n u4 u5 u_sum quotient h
  rw [h.1, h.2, h.3]
  sorry

end units_digit_of_quotient_l382_382563


namespace find_complex_z_l382_382235

theorem find_complex_z (z : ℂ) (a b : ℝ) (h : z = a + b * complex.I) (h₁ : z^2 = -45 - 48 * complex.I) :
  z = 3 - 8 * complex.I ∨ z = -3 + 8 * complex.I :=
sorry

end find_complex_z_l382_382235


namespace sum_of_valid_6_digit_numbers_divisible_by_37_l382_382762

def isValidDigit (d : ℕ) : Prop := d ≠ 0 ∧ d ≠ 9

def isValid6DigitNumber (n : ℕ) : Prop :=
  n ≥ 100000 ∧
  n < 1000000 ∧
  (∀ i : ℕ, i < 6 → isValidDigit (n / 10^i % 10))

theorem sum_of_valid_6_digit_numbers_divisible_by_37 :
  ∃ S : ℕ, (∀ n : ℕ, isValid6DigitNumber n → S = Σ n) → S % 37 = 0 :=
sorry

end sum_of_valid_6_digit_numbers_divisible_by_37_l382_382762


namespace solution_set_l382_382847

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382847


namespace winnieKeepsBalloons_l382_382476

-- Given conditions
def redBalloons : Nat := 24
def whiteBalloons : Nat := 39
def greenBalloons : Nat := 72
def chartreuseBalloons : Nat := 91
def totalFriends : Nat := 11

-- Total balloons
def totalBalloons : Nat := redBalloons + whiteBalloons + greenBalloons + chartreuseBalloons

-- Theorem: Prove the number of balloons Winnie keeps for herself
theorem winnieKeepsBalloons :
  totalBalloons % totalFriends = 6 :=
by
  -- Placeholder for the proof
  sorry

end winnieKeepsBalloons_l382_382476


namespace solve_inequalities_l382_382897

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382897


namespace JacobNeed_l382_382356

-- Definitions of the conditions
def jobEarningsBeforeTax : ℝ := 25 * 15
def taxAmount : ℝ := 0.10 * jobEarningsBeforeTax
def jobEarningsAfterTax : ℝ := jobEarningsBeforeTax - taxAmount

def cookieEarnings : ℝ := 5 * 30

def tutoringEarnings : ℝ := 100 * 4

def lotteryWinnings : ℝ := 700 - 20
def friendShare : ℝ := 0.30 * lotteryWinnings
def netLotteryWinnings : ℝ := lotteryWinnings - friendShare

def giftFromSisters : ℝ := 700 * 2

def totalEarnings : ℝ := jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters

def travelGearExpenses : ℝ := 3 + 47

def netSavings : ℝ := totalEarnings - travelGearExpenses

def tripCost : ℝ := 8000

-- Statement to be proven
theorem JacobNeed (jobEarningsBeforeTax taxAmount jobEarningsAfterTax cookieEarnings tutoringEarnings 
netLotteryWinnings giftFromSisters totalEarnings travelGearExpenses netSavings tripCost : ℝ) : 
  (jobEarningsAfterTax == (25 * 15) - (0.10 * (25 * 15))) → 
  (cookieEarnings == 5 * 30) →
  (tutoringEarnings == 100 * 4) →
  (netLotteryWinnings == (700 - 20) - (0.30 * (700 - 20))) →
  (giftFromSisters == 700 * 2) →
  (totalEarnings == jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters) →
  (travelGearExpenses == 3 + 47) →
  (netSavings == totalEarnings - travelGearExpenses) →
  (tripCost == 8000) →
  (tripCost - netSavings = 5286.50) :=
by
  intros
  sorry

end JacobNeed_l382_382356


namespace solution_set_inequalities_l382_382839

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382839


namespace max_q_minus_r_839_l382_382815

theorem max_q_minus_r_839 : ∃ (q r : ℕ), (839 = 19 * q + r) ∧ (0 ≤ r ∧ r < 19) ∧ q - r = 41 :=
by
  sorry

end max_q_minus_r_839_l382_382815


namespace pieces_to_same_point_l382_382086

theorem pieces_to_same_point :
  ∀ (x y z : ℤ), (∃ (final_pos : ℤ), (x = final_pos ∧ y = final_pos ∧ z = final_pos)) ↔ 
  (x, y, z) = (1, 2009, 2010) ∨ 
  (x, y, z) = (0, 2009, 2010) ∨ 
  (x, y, z) = (2, 2009, 2010) ∨ 
  (x, y, z) = (3, 2009, 2010) := 
by {
  sorry
}

end pieces_to_same_point_l382_382086


namespace quadratic_function_example_l382_382256

theorem quadratic_function_example : ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x^2 + b * x + c = 0) ↔ (x = 1 ∨ x = 5)) ∧ 
  (a * 3^2 + b * 3 + c = 8) ∧ 
  (a = -2 ∧ b = 12 ∧ c = -10) :=
by
  sorry

end quadratic_function_example_l382_382256


namespace positive_difference_between_sums_is_zero_l382_382357

def sum_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def round_to_nearest_five (k : ℕ) : ℕ :=
  if k % 5 == 0 then k
  else if k % 5 < 3 then k - k % 5
  else k + (5 - k % 5)

def rounded_sum (n : ℕ) : ℕ :=
  (List.range n).map (λ x, round_to_nearest_five (x + 1)).sum

theorem positive_difference_between_sums_is_zero :
  let jo_sum := sum_integers 100 in
  let kate_sum := rounded_sum 100 in
  |jo_sum - kate_sum| = 0 :=
by
  sorry

end positive_difference_between_sums_is_zero_l382_382357


namespace roadsters_paving_company_total_cement_l382_382405

noncomputable def cement_lexi : ℝ := 10
noncomputable def cement_tess : ℝ := cement_lexi + 0.20 * cement_lexi
noncomputable def cement_ben : ℝ := cement_tess - 0.10 * cement_tess
noncomputable def cement_olivia : ℝ := 2 * cement_ben

theorem roadsters_paving_company_total_cement :
  cement_lexi + cement_tess + cement_ben + cement_olivia = 54.4 := by
  sorry

end roadsters_paving_company_total_cement_l382_382405


namespace white_dandelions_on_saturday_l382_382148

-- Define the life cycle of a dandelion
structure Dandelion where
  status : ℕ  -- 0: yellow, 1: white
  day_observed : ℕ  -- Day it was observed (0: Monday, 1: Tuesday, ...)

-- Define initial conditions for Monday and Wednesday
def monday_yellow := 20
def monday_white := 14
def wednesday_yellow := 15
def wednesday_white := 11

-- Theorem to prove the number of white dandelions on Saturday
theorem white_dandelions_on_saturday 
  (monday_yellow monday_white wednesday_yellow wednesday_white : ℕ)
  (new_dandelions : ℕ) :
  monday_yellow = 20 →
  monday_white = 14 →
  wednesday_yellow = 15 →
  wednesday_white = 11 →
  new_dandelions = (wednesday_yellow + wednesday_white) - (monday_yellow + monday_white) →
  ∃ white_dandelions_on_saturday : ℕ, white_dandelions_on_saturday = new_dandelions  
:=
begin
  intros,
  existsi new_dandelions,
  sorry
end

end white_dandelions_on_saturday_l382_382148


namespace prob_xiao_ming_at_least_3_eq_xiao_yu_distribution_table_xiao_yu_expectation_eq_recommend_participant_l382_382455

namespace AviationCompetition

noncomputable theory

def xiao_ming_prob_correct : ℚ := 3 / 4

def xiao_ming_event (n : ℕ) : ℚ :=
if n = 3 then (Mathlib.Combinatorics.choose 4 3) * (xiao_ming_prob_correct^3) * ((1 - xiao_ming_prob_correct)^1)
else if n = 4 then (Mathlib.Combinatorics.choose 4 4) * (xiao_ming_prob_correct^4)
else 0

def prob_xiao_ming_at_least_3 : ℚ :=
xiao_ming_event 3 + xiao_ming_event 4

def xiao_yu_distribution : list (ℕ × ℚ) :=
[(2, 3 / 14), (3, 4 / 7), (4, 3 / 14)]

def xiao_yu_expectation : ℚ :=
2 * (3 / 14) + 3 * (4 / 7) + 4 * (3 / 14)

def xiao_yu_prob_at_least_3 : ℚ :=
4 / 7 + 3 / 14

theorem prob_xiao_ming_at_least_3_eq : prob_xiao_ming_at_least_3 = 189 / 256 := sorry

theorem xiao_yu_distribution_table : xiao_yu_distribution = [(2, 3 / 14), (3, 4 / 7), (4, 3 / 14)] := sorry

theorem xiao_yu_expectation_eq : xiao_yu_expectation = 3 := sorry

theorem recommend_participant : (prob_xiao_ming_at_least_3 < xiao_yu_prob_at_least_3) :=
by
  exact nat.lt_of_le_of_ne (show prob_xiao_ming_at_least_3 ≤ xiao_yu_prob_at_least_3 from sorry) (λ h, sorry)

end AviationCompetition

end prob_xiao_ming_at_least_3_eq_xiao_yu_distribution_table_xiao_yu_expectation_eq_recommend_participant_l382_382455


namespace triangle_area_l382_382682

theorem triangle_area (b c : ℝ) (C : ℝ)
  (hb : b = 1) (hc : c = sqrt 3) (hC : C = 2/3 * Real.pi) :
  let a := sqrt (c^2 + b^2 - 2 * b * c * cos C) in
  (1/2) * a * b * sin C = sqrt 3 / 4 :=
by
  rw [hb, hc, hC]
  let a := sqrt (3 + 1 - 2 * 1 * sqrt 3 * (-1/2))
  have ha : a = 1 := by -- this proof step is not necessary but provides insight
    rw [Real.sqrt_eq_iff_sq_eq, ← sub_eq_zero]
    ring
    norm_num
  rw [ha, mul_one, mul_one]
  have hsin : sin (2/3 * Real.pi) = sqrt 3 / 2 := by
    exact Real.sin_two_thirds_pi
  rw [hsin]
  norm_num
  sorry -- Omitted the rest of proof steps

end triangle_area_l382_382682


namespace gumball_guarantee_min_l382_382160

theorem gumball_guarantee_min {R B W G : Nat} (R = 13) (B = 5) (W = 1) (G = 9) :
  ∀ n, (∃ r b g w, r + b + g + w = n ∧ r ≤ 2 ∧ b ≤ 2 ∧ g ≤ 2 ∧ w ≤ 1) →
  n < 8 → ∃ c, c >= 3 :=
by
  sorry

end gumball_guarantee_min_l382_382160


namespace distance_between_X_and_Y_l382_382190

theorem distance_between_X_and_Y :
  ∀ (D : ℝ), 
  (10 : ℝ) * (D / (10 : ℝ) + D / (4 : ℝ)) / (10 + 4) = 142.85714285714286 → 
  D = 1000 :=
by
  intro D
  sorry

end distance_between_X_and_Y_l382_382190


namespace continuity_value_at_two_l382_382087

theorem continuity_value_at_two 
  (h₀ : ∀ x, x^4 - 16 = (x^2 - 4) * (x^2 + 4))
  (h₁ : ∀ x, x^2 - 4 = (x - 2) * (x + 2)) :
  ∀ f, f = (λ x, if x ≠ 2 then (x^4 - 16) / (x^2 - 4) else 2) → 
  continuous_at f 2 ∧ f 2 = 2 :=
begin
  sorry
end

end continuity_value_at_two_l382_382087


namespace relationship_abc_l382_382561

def a := Real.log 0.4 / Real.log 2
def b := 0.4 ^ 2
def c := 2 ^ 0.4

theorem relationship_abc : a < b ∧ b < c :=
by
  sorry -- The proof steps are omitted

end relationship_abc_l382_382561


namespace trees_in_park_l382_382084

variable (W O T : Nat)

theorem trees_in_park (h1 : W = 36) (h2 : O = W + 11) (h3 : T = W + O) : T = 83 := by
  sorry

end trees_in_park_l382_382084


namespace weight_units_correct_l382_382570

-- Definitions of weights
def weight_peanut_kernel := 1 -- gram
def weight_truck_capacity := 8 -- ton
def weight_xiao_ming := 30 -- kilogram
def weight_basketball := 580 -- gram

-- Proof that the weights have correct units
theorem weight_units_correct :
  (weight_peanut_kernel = 1 ∧ weight_truck_capacity = 8 ∧ weight_xiao_ming = 30 ∧ weight_basketball = 580) :=
by {
  sorry
}

end weight_units_correct_l382_382570


namespace equilateral_triangle_marked_points_l382_382996

-- Definitions for points, lines, and circles
variables {Point : Type} [MetricSpace Point]
variables (l m : Set Point) (P : Point) (C : MetricSphere P (radius : ℝ)) -- Circle with center P

-- Mutually perpendicular lines
def are_perpendicular (l m : Set Point) : Prop := sorry

-- Point is on circle
def on_circle (C : MetricSphere P radius) (x : Point) : Prop := sorry

-- Tangent line to the circle at a given point
def tangent_line (C : MetricSphere P radius) (x : Point) : Set Point := sorry

-- Equidistant points on the tangent line
def equidistant_points (T : Set Point) (l m : Set Point) (x : Point) : Prop := sorry

-- Points forming an equilateral triangle
def is_equilateral_triangle (A B C : Point) : Prop := sorry

-- Theorem statement
theorem equilateral_triangle_marked_points 
  (l m : Set Point) 
  (P : Point) 
  (C : MetricSphere P radius) 
  (h_perpendicular : are_perpendicular l m) 
  (h_intersection : P ∈ C) 
  (h_points_eqdist : ∀ x ∈ C, equidistant_points (tangent_line C x) l m x) : 
  ∃ A B C : Point, on_circle C A ∧ on_circle C B ∧ on_circle C C ∧ is_equilateral_triangle A B C :=
sorry 

end equilateral_triangle_marked_points_l382_382996


namespace other_x_intercept_l382_382251

noncomputable def quadratic_function_vertex :=
  ∃ (a b c : ℝ), ∀ (x : ℝ), (a ≠ 0) →
  (5, -3) = ((-b) / (2 * a), a * ((-b) / (2 * a))^2 + b * ((-b) / (2 * a)) + c) ∧
  (x = 1) ∧ (a * x^2 + b * x + c = 0) →
  ∃ (x2 : ℝ), x2 = 9

theorem other_x_intercept :
  quadratic_function_vertex :=
sorry

end other_x_intercept_l382_382251


namespace probability_of_drawing_specific_balls_l382_382502

noncomputable def combination (n k : ℕ) : ℕ := n.choose k

theorem probability_of_drawing_specific_balls :
  let total_balls := 6 + 5 + 4 + 3,
      draws := 3,
      ways_to_draw_three := combination total_balls draws,
      ways_to_draw_one_white := combination 5 1,
      ways_to_draw_one_red := combination 4 1,
      ways_to_draw_one_green := combination 3 1 in
  (ways_to_draw_one_white * ways_to_draw_one_red * ways_to_draw_one_green).toRat / ways_to_draw_three.toRat = (5 / 68) := by
  sorry

end probability_of_drawing_specific_balls_l382_382502


namespace robin_hair_length_l382_382034

theorem robin_hair_length
  (l d g : ℕ)
  (h₁ : l = 16)
  (h₂ : d = 11)
  (h₃ : g = 12) :
  (l - d + g = 17) :=
by sorry

end robin_hair_length_l382_382034


namespace escalator_steps_l382_382197

theorem escalator_steps (k : ℝ) (n : ℝ) (h1 : ∀ t, (t > 0) → t = 27/k) (h2 : ∀ t, (t > 0) → t = 18/k) :
  n = 54 :=
by
  -- Definitions
  let A_time := 27 / (2 * k)
  let Z_time := 18 / k
  let esc_speed_A := 2 * k + 1
  let esc_speed_Z := k + 1

  -- Equations for times x speeds
  have eqn_1 : Z_time * esc_speed_Z = n := by
    sorry
  have eqn_2 : A_time * esc_speed_A = n := by
    sorry

  -- Find n is the same from both equations
  have eqn : (Z_time * esc_speed_Z) = (A_time * esc_speed_A) := by
    exact h1 t 27
    exact h2 t 18
  exact sorry

end escalator_steps_l382_382197


namespace greatest_integer_less_than_AD_l382_382692

theorem greatest_integer_less_than_AD
  (AB AD AE : ℝ)
  (hAB : AB = 100)
  (hE_mid : AE = AD / 2)
  (h_perp : is_perpendicular (line (point 0 0) (point AD 0)) (line (point 0 0) (point (AD / 2) 100)))
  :
  ⌊AD⌋ = 141 :=
sorry

end greatest_integer_less_than_AD_l382_382692


namespace average_age_l382_382444

theorem average_age (Jared Molly Hakimi : ℕ) (h1 : Jared = Hakimi + 10) (h2 : Molly = 30) (h3 : Hakimi = 40) :
  (Jared + Molly + Hakimi) / 3 = 40 :=
by
  sorry

end average_age_l382_382444


namespace min_n_subsets_correct_l382_382722

open Finset

noncomputable def min_n_subsets (S : Finset ℕ) : ℕ :=
  -- Calculate the minimum number of subsets
  let n := 15 in -- Final value concluded from problem
  n

theorem min_n_subsets_correct
  {S : Finset ℕ}
  (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
  {A : ℕ → Finset ℕ}
  (h_size : ∀ i, (A i).card = 7)
  (h_intersect : ∀ {i j : ℕ}, i < j → (A i ∩ A j).card ≤ 3)
  (h_triple : ∀ (M : Finset ℕ), M.card = 3 → ∃ i, M ⊆ A i) :
  min_n_subsets S = 15 :=
  sorry  -- Proof omitted

end min_n_subsets_correct_l382_382722


namespace positive_integer_count_l382_382248

-- Define the problem conditions and prove the result
theorem positive_integer_count (h : ∀ n : ℕ, 0 < n ∧ n < 24 → (24 - n) ∣ n) :
  {n : ℕ | 0 < n ∧ n < 24 ∧ (24 - n) ∣ n}.card = 7 :=
by sorry

end positive_integer_count_l382_382248


namespace slope_line_point_l382_382164

theorem slope_line_point (m b : ℝ) (h_slope : m = 3) (h_point : 2 = m * 5 + b) : m + b = -10 :=
by
  sorry

end slope_line_point_l382_382164


namespace max_sides_convex_polygon_with_4_obtuse_angles_l382_382328

theorem max_sides_convex_polygon_with_4_obtuse_angles 
  (n : ℕ) (h_convex: ∀ i j k : ℕ, i < j → j < k → k < n → 
            (angle_between i j + angle_between j k + angle_between k i) < 180) 
  (h_obtuse: ∃ s₁ s₂ s₃ s₄: ℕ, s₁ ≠ s₂ ∧ s₂ ≠ s₃ ∧ s₃ ≠ s₄ ∧ s₁ ≠ s₃ ∧ s₁ ≠ s₄ ∧ s₂ ≠ s₄ ∧ 
              angle_between_consecutive s₁ > 90 ∧ angle_between_consecutive s₂ > 90 ∧ 
              angle_between_consecutive s₃ > 90 ∧ angle_between_consecutive s₄ > 90) : 
     n ≤ 7 := 
sorry

end max_sides_convex_polygon_with_4_obtuse_angles_l382_382328


namespace card_pack_prob_l382_382255

theorem card_pack_prob (N : ℕ) (h : 0 < N):
  let num_suits := 4
  let spades := (N / num_suits)
  let hearts := (N / num_suits)
  let draw_two := N * (N - 1) / 2
  let one_spade_one_heart := spades * hearts
  (one_spade_one_heart / draw_two) = 0.12745098039215685 → N = 52 :=
begin
  sorry
end

end card_pack_prob_l382_382255


namespace second_hand_travel_distance_l382_382820

theorem second_hand_travel_distance (r : ℝ) (t : ℝ) : 
  r = 10 → t = 45 → 2 * t * π * r = 900 * π :=
by
  intro r_def t_def
  sorry

end second_hand_travel_distance_l382_382820


namespace solution_set_linear_inequalities_l382_382911

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382911


namespace distinct_monic_quadratic_polynomials_count_l382_382574

theorem distinct_monic_quadratic_polynomials_count :
  let bound := 122^20
  let max_exp := 50
  ∃ count : ℕ, count = 5699 ∧ 
  (∀ (a b : ℕ), 0 <= a ∧ 0 <= b ∧ 
  (5^a + 5^b <= bound ∧ 5^(a + b) <= bound ∧ 
  (a ≠ b)) →
  (x - 5^a)*(x - 5^b) ∧ monic (x - 5^a)*(x - 5^b) ∧ 
  a + b <= max_exp) := sorry

end distinct_monic_quadratic_polynomials_count_l382_382574


namespace exists_least_number_l382_382484

open Nat

theorem exists_least_number
  (N : ℕ)
  (h1 : N % 5 = 3)
  (h2 : N % 6 = 3)
  (h3 : N % 7 = 3)
  (h4 : N % 8 = 3)
  (h5 : N % 9 = 0)
  : N = 1683 :=
sorry

end exists_least_number_l382_382484


namespace problem_statement_l382_382290

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem problem_statement : f (1 / 2) + f (-1 / 2) = 2 := sorry

end problem_statement_l382_382290


namespace count_primes_squared_in_range_l382_382659

theorem count_primes_squared_in_range : 
  (finset.card (finset.filter (λ p, p^2 ≥ 5000 ∧ p^2 ≤ 9000) 
    (finset.filter nat.prime (finset.range 95)))) = 5 := 
by sorry

end count_primes_squared_in_range_l382_382659


namespace diff_of_squares_l382_382213

-- Define constants a and b
def a : ℕ := 65
def b : ℕ := 35

-- State the theorem to be proven using the difference of squares formula
theorem diff_of_squares : a^2 - b^2 = 3000 := by
  have h1 : a + b = 100 := by rfl -- Checking a + b
  have h2 : a - b = 30 := by rfl -- Checking a - b
  have h3 : (a+b)*(a-b) = 3000 := by
    rw [h1, h2] -- Substituting the values
    norm_num -- Simplifying the arithmetics
  exact h3 -- Providing the ultimate result

end diff_of_squares_l382_382213


namespace edward_initial_amount_l382_382567

-- Defining the conditions
def cost_books : ℕ := 6
def cost_pens : ℕ := 16
def cost_notebook : ℕ := 5
def cost_pencil_case : ℕ := 3
def amount_left : ℕ := 19

-- Mathematical statement to prove
theorem edward_initial_amount : 
  cost_books + cost_pens + cost_notebook + cost_pencil_case + amount_left = 49 :=
by
  sorry

end edward_initial_amount_l382_382567


namespace good_numbers_from_33_and_above_l382_382054

/-- An integer n is called a good number if it can be expressed as a sum of positive integers
    whose reciprocals sum to 1. -/
def is_good_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : fin k → ℕ), (∀ i, 0 < a i) ∧ (∑ i, a i = n) ∧ (∑ i, 1 / (a i : ℚ) = 1)

/-- The main theorem to prove: every integer n ≥ 33 is a good number. -/
theorem good_numbers_from_33_and_above :
  (∀ n : ℕ, 33 ≤ n → is_good_number n) :=
sorry

end good_numbers_from_33_and_above_l382_382054


namespace minimum_sum_of_distances_l382_382612

def point := (ℝ × ℝ)
def P : point := (1, 3)
def Q : point := (-1, 2)

def line (M : point) : Prop := M.1 - M.2 + 1 = 0

def distance (A B : point) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def sum_distances (M : point) : ℝ :=
  distance P M + distance Q M

theorem minimum_sum_of_distances 
  (h : ∀ M : point, line M → sum_distances M ≥ 3) :
  ∃ M : point, line M ∧ sum_distances M = 3 :=
sorry

end minimum_sum_of_distances_l382_382612


namespace solution_set_linear_inequalities_l382_382934

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382934


namespace problem_statement_l382_382028

-- Definitions and assumptions for vectors
variables {a b : ℝ} [nonzero a] [nonzero b]

def dot_product (a b : vector ℝ) : ℝ := a i * b i

def angle_is_obtuse (a b : vector ℝ) : Prop := dot_product a b < 0

-- Proposition p (if dot product is negative, then the angle should be obtuse)
def p : Prop := ∀ (a b : vector ℝ) [nonzero a] [nonzero b], (dot_product a b < 0) → angle_is_obtuse a b

-- Definitions and assumptions for functions
variables {f : ℝ → ℝ} {x₀ : ℝ}

def is_extremum (f : ℝ → ℝ) (x : ℝ) : Prop := f'(x) = 0

-- Proposition q (if the derivative at x₀ is 0, then x₀ is an extremum)
def q : Prop := ∀ (f : ℝ → ℝ) (x₀ : ℝ), (f'(x₀) = 0) → is_extremum f x₀

-- Proof that neither p nor q are true
theorem problem_statement : (¬p) ∧ (¬q) :=
by
  intro hp
  unfold p at hp
  apply hp with (some_example_vectors) -- Provide a counterexample to p

  intro hq
  unfold q at hq
  apply hq with (some_example_function, some_example_x0) -- Provide a counterexample to q
sorry -- Proof omitted

end problem_statement_l382_382028


namespace exists_x1_x2_inequality_l382_382372
noncomputable def exists_x1_x2 (f : ℝ → ℝ) : Prop :=
  (0 ≤ x1 ∧ x1 ≤ 1) ∧ (0 ≤ x2 ∧ x2 <= 1) ∧
  (0 < f x1) ∧ (0 < f x2) ∧
  ∀ x1 x2, ( 0 ≤ x1 ∧ x1 ≤ 1) ∧ (0 ≤ x2 ∧ x2 <= 1) →
    (\frac{(x2 - x1) * (f x1)^2}{f x2} > \frac{f 0}{4} )

-- Main theorem statement
theorem exists_x1_x2_inequality
  (f : ℝ → ℝ) (hf_pos : ∀ x ∈ Icc 0 1, 0 < f x) 
  (hf_bounded : ∃ M, ∀ x ∈ Icc 0 1, f x ≤ M) :
  ∃ x1 x2 ∈ Icc 0 1, (x2 - x1) * (f x1)^2 / f x2 > f 0 / 4 :=
begin
  sorry -- Proof is left to be filled.
end

end exists_x1_x2_inequality_l382_382372


namespace basketball_cards_per_box_l382_382565

-- Given conditions
def num_basketball_boxes : ℕ := 9
def num_football_boxes := num_basketball_boxes - 3
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255
def total_football_cards := num_football_boxes * cards_per_football_box

-- We want to prove that the number of cards in each basketball card box is 15
theorem basketball_cards_per_box :
  (total_cards - total_football_cards) / num_basketball_boxes = 15 := by
  sorry

end basketball_cards_per_box_l382_382565


namespace unique_p_for_equal_roots_l382_382559

theorem unique_p_for_equal_roots : 
  (∃ p : ℝ, (λ A B C p, B^2 - 4 * A * C) 1 (-p) (p^2) = 0) ∧ 
  (∀ q r : ℝ, (λ A B C p, B^2 - 4 * A * C) 1 (-q) (q^2) = 0 → 
                     (λ A B C p, B^2 - 4 * A * C) 1 (-r) (r^2) = 0 → q = r) :=
sorry

end unique_p_for_equal_roots_l382_382559


namespace logarithm_inequality_mistake_l382_382417

theorem logarithm_inequality_mistake:
  let x := (1 / 2 : ℝ) in
  (1 / 4 : ℝ) > (1 / 8 : ℝ) →
  x^2 > x^3 →
  2 * real.log x > 3 * real.log x →
  2 < 3 :=
by
  intros h1 h2 h3
  sorry

end logarithm_inequality_mistake_l382_382417


namespace solution_set_l382_382849

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382849


namespace minimize_surface_area_l382_382605

-- Definitions and assumptions for the problem:
variables (k : ℝ) (V : ℝ) (r h : ℝ)

-- Conditions
def conditions := (k > 0.5) ∧ (V = 2 * k * π * r^2 * h) ∧ (h = r)

-- The proof problem
theorem minimize_surface_area (hk : k > 0.5) (hV : V = 2 * k * π * r^2 * h) (hh : h = r) : 
  h / r = 1 :=
by 
  sorry

end minimize_surface_area_l382_382605


namespace solution_set_of_linear_inequalities_l382_382867

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382867


namespace replace_P_with_1_l382_382341

noncomputable theory

-- Define the structure of the problem
def grid : Type := matrix (fin 4) (fin 4) (fin 5)

def digits_appear_once (m : grid) : Prop :=
  (∀ i : fin 4, (finset.univ.image (λ j, m i j)).card = 4) ∧  -- Each row has distinct digits 1 to 4
  (∀ j : fin 4, (finset.univ.image (λ i, m i j)).card = 4) ∧  -- Each column has distinct digits 1 to 4
  (∀ x y : fin 2, (finset.univ.image (λ i, m ⟨2 * x.1 + i.1, sorry⟩ ⟨2 * y.1 + (i : fin 2), sorry⟩)).card = 4)  -- Each 2x2 sub-grid

-- Define the specific position P
def P : fin 4 × fin 4 := ⟨2, 3⟩  -- Say P is the position (2, 3)

-- State the theorem
theorem replace_P_with_1 (m : grid) (h : digits_appear_once m) : m P.1 P.2 = 1 :=
sorry

end replace_P_with_1_l382_382341


namespace prime_square_count_l382_382643

-- Define the set of natural numbers whose squares lie between 5000 and 9000
def within_square_range (n : ℕ) : Prop := 5000 < n * n ∧ n * n < 9000

-- Define the predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the set of prime numbers whose squares lie within the range
def primes_within_range : Finset ℕ := (Finset.filter is_prime (Finset.Ico 71 95))

-- The main statement: the number of prime numbers whose squares are between 5000 and 9000
theorem prime_square_count : (primes_within_range.filter within_square_range).card = 6 :=
sorry

end prime_square_count_l382_382643


namespace number_of_terms_in_expansion_l382_382061

theorem number_of_terms_in_expansion :
  (∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 c1 c2 c3 : ℕ), (a1 + a2 + a3 + a4 + a5) * (b1 + b2 + b3 + b4) * (c1 + c2 + c3) = 60) :=
by
  sorry

end number_of_terms_in_expansion_l382_382061


namespace solution_set_of_linear_inequalities_l382_382858

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382858


namespace cyclic_quadrilateral_circumcenter_l382_382795

-- Definitions and conditions based on the given problem
variables {A B C D E F G H A' B' C' D' : Type*}
variables [cyclic_quadrilateral A B C D] (non_perpendicular_diagonals : ¬is_perpendicular (diagonal_AC A B C D) (diagonal_BD A B C D))

-- Define the perpendicular feet from the vertices
def feet_perpendicular (a b c d : Type*) := 
  (A' : perpendicular_from A to (diagonal_BD B D)) ×
  (B' : perpendicular_from B to (diagonal_AC A C)) ×
  (C' : perpendicular_from C to (diagonal_BD B D)) ×
  (D' : perpendicular_from D to (diagonal_AC A C))

-- Define the intersection points
def intersects (AA' : Line) (DD' : Line) := E
def intersects (DD' : Line) (CC' : Line) := F
def intersects (CC' : Line) (BB' : Line) := G
def intersects (BB' : Line) (AA' : Line) := H

-- The theorem to prove
theorem cyclic_quadrilateral_circumcenter {α β γ δ : Type*} 
[A B C D E F G H A' B' C' D' : Type*] 
[cyclic_quadrilateral A B C D] 
(non_perpendicular_diagonals : ¬is_perpendicular (diagonal_AC A B C D) (diagonal_BD A B C D))
(feet_perpendicular : feet_perpendicular A B C D)
(intersections : intersects):
  cyclic_quadrilateral A' B' C' D' ∧ circumcenter A' B' C' D' = (intersect (E G) (F H)) :=
sorry

end cyclic_quadrilateral_circumcenter_l382_382795


namespace largest_mersenne_prime_less_than_200_l382_382128

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_mersenne_prime (p : ℕ) : Prop := 
  ∃ n : ℕ, is_prime n ∧ p = 2^n - 1 ∧ is_prime p

theorem largest_mersenne_prime_less_than_200 : 
  ∃ p : ℕ, is_mersenne_prime p ∧ p < 200 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 200 → q ≤ p := 
  exists.intro 127 (and.intro 
    (exists.intro 7 (and.intro 
      (and.intro (by norm_num) sorry) -- Proof of 7 being prime and 127 being prime can be provided
      (by norm_num)
      (by norm_num)) sorry)

end largest_mersenne_prime_less_than_200_l382_382128


namespace white_dandelions_on_saturday_l382_382147

-- Define the life cycle of a dandelion
structure Dandelion where
  status : ℕ  -- 0: yellow, 1: white
  day_observed : ℕ  -- Day it was observed (0: Monday, 1: Tuesday, ...)

-- Define initial conditions for Monday and Wednesday
def monday_yellow := 20
def monday_white := 14
def wednesday_yellow := 15
def wednesday_white := 11

-- Theorem to prove the number of white dandelions on Saturday
theorem white_dandelions_on_saturday 
  (monday_yellow monday_white wednesday_yellow wednesday_white : ℕ)
  (new_dandelions : ℕ) :
  monday_yellow = 20 →
  monday_white = 14 →
  wednesday_yellow = 15 →
  wednesday_white = 11 →
  new_dandelions = (wednesday_yellow + wednesday_white) - (monday_yellow + monday_white) →
  ∃ white_dandelions_on_saturday : ℕ, white_dandelions_on_saturday = new_dandelions  
:=
begin
  intros,
  existsi new_dandelions,
  sorry
end

end white_dandelions_on_saturday_l382_382147


namespace count_primes_squared_in_range_l382_382658

theorem count_primes_squared_in_range : 
  (finset.card (finset.filter (λ p, p^2 ≥ 5000 ∧ p^2 ≤ 9000) 
    (finset.filter nat.prime (finset.range 95)))) = 5 := 
by sorry

end count_primes_squared_in_range_l382_382658


namespace quadratic_two_distinct_real_roots_l382_382441

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  let Δ := (k + 3)^2 - 4 * k in Δ > 0 :=
by
  let Δ := (k + 3)^2 - 4 * k
  have hΔ_pos : Δ = (k + 1)^2 + 8 := sorry
  have h_pos : (k + 1)^2 + 8 > 0 := sorry
  exact h_pos

end quadratic_two_distinct_real_roots_l382_382441


namespace find_m_l382_382431

variable (m : ℝ)

def f (x : ℝ) := (m^2 - m - 1) * x^m

theorem find_m 
  (h1 : ∀ (x : ℝ), x > 0 → (m^2 - m - 1) * x^m = x^m)
  (h2 : ∀ (x : ℝ), x > 0 → f m x > 0) :
  m = 2 :=
by
  sorry

end find_m_l382_382431


namespace number_in_eighth_group_l382_382189

theorem number_in_eighth_group (employees groups n l group_size numbering_drawn starting_number: ℕ) 
(h1: employees = 200) 
(h2: groups = 40) 
(h3: n = 5) 
(h4: number_in_fifth_group = 23) 
(h5: starting_number + 4 * n = number_in_fifth_group) : 
  starting_number + 7 * n = 38 :=
by
  sorry

end number_in_eighth_group_l382_382189


namespace lesser_of_two_numbers_l382_382443

theorem lesser_of_two_numbers (a b : ℕ) (h₁ : a + b = 55) (h₂ : a - b = 7) (h₃ : a > b) : b = 24 :=
by
  sorry

end lesser_of_two_numbers_l382_382443


namespace length_of_DE_l382_382705

theorem length_of_DE 
  (A B C D E : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (hABC : Triangle A B C)
  (hAB : dist A B = 24)
  (hBC : dist B C = 26)
  (hAC : dist A C = 28)
  (hD_on_AB : D ∈ LineSegment A B)
  (hE_on_AC : E ∈ LineSegment A C)
  (hDE_parallel_BC : Parallel (Line D E) (Line B C))
  (hAD_DB_ratio : dist A D / dist D B = 3)
  (hAE_EC_ratio : dist A E / dist E C = 3) :
  dist D E = 19.5 := 
sorry

end length_of_DE_l382_382705


namespace probability_heads_tenth_flip_l382_382747

def molly_flip_tenth_is_head : Prop :=
  let fair_coin : Prop := ∀ n, Pr (flip_coin n = heads) = 1 / 2
  let previous_flips : Prop := ∀ n, Pr (flip_coin 10 = heads | prev_flips = prev_heads_tails) = Pr (flip_coin 10 = heads)
  fair_coin → previous_flips → Pr (flip_coin 10 = heads) = 1 / 2

theorem probability_heads_tenth_flip (h1 : molly_flip_tenth_is_head) : Pr(flip_coin 10 = heads) = 1/2 :=
sorry

end probability_heads_tenth_flip_l382_382747


namespace diff_of_squares_l382_382212

-- Define constants a and b
def a : ℕ := 65
def b : ℕ := 35

-- State the theorem to be proven using the difference of squares formula
theorem diff_of_squares : a^2 - b^2 = 3000 := by
  have h1 : a + b = 100 := by rfl -- Checking a + b
  have h2 : a - b = 30 := by rfl -- Checking a - b
  have h3 : (a+b)*(a-b) = 3000 := by
    rw [h1, h2] -- Substituting the values
    norm_num -- Simplifying the arithmetics
  exact h3 -- Providing the ultimate result

end diff_of_squares_l382_382212


namespace solve_inequalities_l382_382882

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382882


namespace find_a8_l382_382295

open Nat

variables {a_n : ℕ → ℕ} (q a1 : ℕ)

def geometric_sequence (a : ℕ → ℕ) (r : ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n * r

-- Conditions: a_2 = 2 and a_3 * a_4 = 32
def given_conditions : Prop :=
  a_n 2 = 2 ∧ a_n 3 * a_n 4 = 32

-- Goal: To prove that a_8 = 128
theorem find_a8 (h : given_conditions) (h_geom : geometric_sequence a_n q) : a_n 8 = 128 :=
  sorry

end find_a8_l382_382295


namespace length_of_rope_l382_382100

-- Define the given conditions
variable (L : ℝ)
variable (h1 : 0.6 * L = 0.69)

-- The theorem to prove
theorem length_of_rope (L : ℝ) (h1 : 0.6 * L = 0.69) : L = 1.15 :=
by
  sorry

end length_of_rope_l382_382100


namespace solve_sqrt_equation_l382_382414

theorem solve_sqrt_equation (x: ℝ) (h: sqrt (x + 9) - sqrt (x - 5) - 2 = 0): 
  x = 11.25 :=
sorry

end solve_sqrt_equation_l382_382414


namespace solve_problem_l382_382308

-- Define vectors
def vector_a (y : ℝ) : ℝ × ℝ := (1, y)
def vector_b : ℝ × ℝ := (1, -3)

-- Perpendicular condition
def perp_condition (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  let a' := (2 * a.1 + b.1, 2 * a.2 + b.2) in
  a'.1 * b.1 + a'.2 * b.2 = 0

-- Calculate the angle between two vectors
def angle_between (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 in
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2) in
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2) in
  Real.arccos (dot_product / (magnitude_a * magnitude_b))

theorem solve_problem :
  (∃ y : ℝ, vector_a y = (1, 2) ∧ perp_condition (vector_a y) vector_b) ∧ angle_between (vector_a 2) vector_b = 3 * Real.pi / 4 := 
by sorry

end solve_problem_l382_382308


namespace solution_set_of_inequalities_l382_382956

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382956


namespace white_dandelions_on_saturday_l382_382150

-- Define the life cycle of a dandelion
structure Dandelion where
  status : ℕ  -- 0: yellow, 1: white
  day_observed : ℕ  -- Day it was observed (0: Monday, 1: Tuesday, ...)

-- Define initial conditions for Monday and Wednesday
def monday_yellow := 20
def monday_white := 14
def wednesday_yellow := 15
def wednesday_white := 11

-- Theorem to prove the number of white dandelions on Saturday
theorem white_dandelions_on_saturday 
  (monday_yellow monday_white wednesday_yellow wednesday_white : ℕ)
  (new_dandelions : ℕ) :
  monday_yellow = 20 →
  monday_white = 14 →
  wednesday_yellow = 15 →
  wednesday_white = 11 →
  new_dandelions = (wednesday_yellow + wednesday_white) - (monday_yellow + monday_white) →
  ∃ white_dandelions_on_saturday : ℕ, white_dandelions_on_saturday = new_dandelions  
:=
begin
  intros,
  existsi new_dandelions,
  sorry
end

end white_dandelions_on_saturday_l382_382150


namespace all_numbers_equal_l382_382425

-- Define the weird mean
def weird_mean (a b : ℝ) : ℝ := real.sqrt ((2 * a^2 + 3 * b^2) / 5)

-- Define the problem with the circular arrangement and conditions
theorem all_numbers_equal (a : ℕ → ℝ) (n : ℕ)
  (hpos : ∀ i, 0 ≤ a i) -- nonnegativity condition (since they are positive integers)
  (h2009 : n = 2009)
  (hcircular : ∀ i, a (i % n) = weird_mean (a ((i - 1) % n)) (a ((i + 1) % n))) :
  ∀ i j, a i = a j :=
by
  sorry

end all_numbers_equal_l382_382425


namespace total_length_of_table_free_sides_l382_382170

theorem total_length_of_table_free_sides
  (L W : ℕ) -- Define lengths of the sides
  (h1 : L = 2 * W) -- The side opposite the wall is twice the length of each of the other two free sides
  (h2 : L * W = 128) -- The area of the rectangular table is 128 square feet
  : L + 2 * W = 32 -- Prove the total length of the table's free sides is 32 feet
  :=
sorry -- proof omitted

end total_length_of_table_free_sides_l382_382170


namespace max_S_difference_l382_382265

noncomputable def a_sequence (n : ℕ) : ℤ := sorry -- Sequence definition

def S (n : ℕ) : ℤ := ∑ i in Finset.range (n+1), a_sequence i -- Sum of first n terms

lemma recurrence_relation (n : ℕ) : 
  (3 * n - 5) * a_sequence (n+1) = (3 * n - 2) * a_sequence n - 9 * n^2 + 2 * ln - 10 := sorry

lemma initial_condition : a_sequence 1 = -8 := sorry

theorem max_S_difference (n m : ℕ) (h1 : n > m) : ∃ k : ℤ, k = 18 ∧ k = S n - S m := sorry

end max_S_difference_l382_382265


namespace factory_needs_to_produce_l382_382511

-- Define the given conditions
def weekly_production_target : ℕ := 6500
def production_mon_tue_wed : ℕ := 3 * 1200
def production_thu : ℕ := 800
def total_production_mon_thu := production_mon_tue_wed + production_thu
def required_production_fri := weekly_production_target - total_production_mon_thu

-- The theorem we need to prove
theorem factory_needs_to_produce : required_production_fri = 2100 :=
by
  -- The proof would go here
  sorry

end factory_needs_to_produce_l382_382511


namespace verify_solution_l382_382091

theorem verify_solution (C₁ C₂ : ℝ) :
  let y := λ x : ℝ, C₁ * Real.exp x + C₂ * Real.exp x + (1 / 3) * x + (1 / 9)
  let y' := λ x : ℝ, C₁ * Real.exp x + C₂ * Real.exp x + (1 / 3)
  let y'' := λ x : ℝ, C₁ * Real.exp x + C₂ * Real.exp x
  ∀ x : ℝ,
    (y'' x - 4 * y' x + 3 * y x) = (x - 1) :=
by
  sorry

end verify_solution_l382_382091


namespace find_error_position_l382_382447

theorem find_error_position :
  ∃ k : ℕ, (k < 21) ∧
  (51 + 5 * (k-1) - 10 + (∑ n in (list.range 21).erase k, 51 + 5 * n)) = 2021 :=
by
  sorry

end find_error_position_l382_382447


namespace solution_set_of_inequalities_l382_382969

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382969


namespace min_additional_trains_needed_l382_382013

-- Definitions
def current_trains : ℕ := 31
def trains_per_row : ℕ := 8
def smallest_num_additional_trains (current : ℕ) (per_row : ℕ) : ℕ :=
  let next_multiple := ((current + per_row - 1) / per_row) * per_row
  next_multiple - current

-- Theorem
theorem min_additional_trains_needed :
  smallest_num_additional_trains current_trains trains_per_row = 1 :=
by
  sorry

end min_additional_trains_needed_l382_382013


namespace remaining_fruit_is_mandarin_l382_382384

theorem remaining_fruit_is_mandarin
    (initial_mandarins : ℕ := 15)
    (replacement_two_mandarins : ℕ → ℕ)
    (replacement_one_mandarin_one_apple : ℕ → ℕ)
    (replacement_two_apples : ℕ → ℕ)
    : ∃ (final_fruit : String), final_fruit = "mandarin" :=
by
  let replacement_two_mandarins := λ m, m - 2 + 1 -- Two mandarins taken, add one apple
  let replacement_one_mandarin_one_apple := λ m, m -- One mandarin and one apple taken, add one mandarin
  let replacement_two_apples := λ a, a + 1 -- Two apples taken, add one apple
  
  -- Verifying the logical deduction shows the last fruit is a mandarin
  -- State the final fruit is a mandarin according to proof steps provided
  exact ⟨"mandarin", sorry⟩

end remaining_fruit_is_mandarin_l382_382384


namespace increasing_sequence_a_range_l382_382628

theorem increasing_sequence_a_range (a : ℝ) (a_seq : ℕ → ℝ) (h_def : ∀ n, a_seq n = 
  if n ≤ 2 then a * n^2 - ((7 / 8) * a + 17 / 4) * n + 17 / 2
  else a ^ n) : 
  (∀ n, a_seq n < a_seq (n + 1)) → a > 2 :=
by
  sorry

end increasing_sequence_a_range_l382_382628


namespace find_constant_l382_382336

theorem find_constant 
  (t : ℝ) 
  (constant : ℝ) 
  (hx : ∀ t, x = 1 - 2 * t) 
  (hy : ∀ t y, y = 2 * t + constant):
  (hx 0.75 = hy 0.75) → constant = -2 :=
by
  intro h
  sorry

end find_constant_l382_382336


namespace domain_of_f_l382_382221

noncomputable def f (x : ℝ) : ℝ := (1 / (x + 9)) + (1 / (x^2 + 9)) + (1 / (x^3 + 9)) + (1 / (Real.exp x + 9))

def domain_f : Set ℝ := { x : ℝ | x ≠ -9 ∧ x ≠ -Real.cbrt 9 }

theorem domain_of_f : ∀ x : ℝ, x ∈ domain_f ↔ (f x ≠ 0) :=
by 
  sorry

end domain_of_f_l382_382221


namespace linear_inequalities_solution_l382_382870

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382870


namespace solution_set_of_inequalities_l382_382955

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382955


namespace solve_inequalities_l382_382902

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382902


namespace ellipse_standard_form_l382_382072

def ellipse_equation (x y : ℝ) : Prop :=
  sqrt (x^2 + (y - 2)^2) + sqrt (x^2 + (y + 2)^2) = 10

theorem ellipse_standard_form (x y : ℝ) (h : ellipse_equation x y) :
  (y^2 / 25) + (x^2 / 21) = 1 :=
sorry

end ellipse_standard_form_l382_382072


namespace least_k_divisible_480_l382_382677

theorem least_k_divisible_480 (k : ℕ) (h : k^4 % 480 = 0) : k = 101250 :=
sorry

end least_k_divisible_480_l382_382677


namespace ellipse_product_l382_382757

/-- Given conditions:
1. OG = 8
2. The diameter of the inscribed circle of triangle ODG is 4
3. O is the center of an ellipse with major axis AB and minor axis CD
4. Point G is one focus of the ellipse
--/
theorem ellipse_product :
  ∀ (O G D : Point) (a b : ℝ),
    OG = 8 → 
    (a^2 - b^2 = 64) →
    (a - b = 4) →
    (AB = 2*a) →
    (CD = 2*b) →
    (AB * CD = 240) :=
by
  intros O G D a b hOG h1 h2 h3 h4
  sorry

end ellipse_product_l382_382757


namespace joint_savings_l382_382716

/--
Kimmie received $1,950 from her handmade crafts at the supermarket.
Her friend Zahra received 2/3 less money when she sold the same amount of handmade crafts on Etsy.
If both of them save 35% of their earnings in the same savings account,
calculate the total amount of money in the joint savings account.
-/
theorem joint_savings (kim_earnings : ℕ) (kim_earnings_value : kim_earnings = 1950)
                      (zahra_earnings : ℕ) (zahra_earnings_value : zahra_earnings = kim_earnings - 2/3 * kim_earnings)
                      (savings_percentage : ℝ) (savings_percentage_value : savings_percentage = 0.35) :
  (0.35 * kim_earnings + 0.35 * zahra_earnings) = 910 :=
by
  sorry

end joint_savings_l382_382716


namespace sum_of_squares_l382_382998

theorem sum_of_squares (n : ℕ) : (∑ k in range (n+1), k^2) = (n^4 + n^2) / 2 :=
by {
  sorry
}

end sum_of_squares_l382_382998


namespace selection_count_constraint_l382_382254

open_locale big_operators
open_locale classical

noncomputable theory

def group_size := 12
def selection_size := 5
def specific_individuals := 3
def max_allowed_specific_individuals := 2

-- The desired counting function with constraints
def number_of_ways_to_choose_with_constraints : ℕ := 
  nat.choose 9 5 + nat.choose 3 1 * nat.choose 9 4 + nat.choose 3 2 * nat.choose 9 3

theorem selection_count_constraint :
  number_of_ways_to_choose_with_constraints = 756 :=
sorry

end selection_count_constraint_l382_382254


namespace inverse_ratio_l382_382809

def g (x : ℝ) := (3 * x - 2) / (x + 4)

noncomputable def g_inv (x : ℝ) := (4 * x + 2) / (3 - x)

theorem inverse_ratio (a b c d : ℝ) (h1 : g_inv x = (a * x + b) / (c * x + d)) : a / c = -4 :=
by {
    sorry
}

end inverse_ratio_l382_382809


namespace proof_problem_l382_382581

variable {a b x y : ℝ}

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem proof_problem : dollar ((x + y) ^ 2) (y ^ 2 + x ^ 2) = 4 * x ^ 2 * y ^ 2 := by
  sorry

end proof_problem_l382_382581


namespace length_of_leg_of_smallest_triangle_is_5div2_l382_382225

noncomputable def leg_of_smallest_45_45_90_triangle
  (hypotenuse_of_largest_triangle : ℝ) (hypotenuse_of_largest_triangle_eq : hypotenuse_of_largest_triangle = 10) : 
  ℝ :=
  let step1 := 10 * (Real.sqrt 2 / 2) in
  let step2 := step1 * (Real.sqrt 2 / 2) in
  let step3 := step2 * (Real.sqrt 2 / 2) in
  let step4 := step3 * (Real.sqrt 2 / 2) in
  step4

theorem length_of_leg_of_smallest_triangle_is_5div2 
  (hypotenuse_of_largest_triangle : ℝ)
  (hypotenuse_of_largest_triangle_eq : hypotenuse_of_largest_triangle = 10) : 
  leg_of_smallest_45_45_90_triangle hypotenuse_of_largest_triangle hypotenuse_of_largest_triangle_eq = 5 / 2 :=
sorry

end length_of_leg_of_smallest_triangle_is_5div2_l382_382225


namespace box_diagonal_and_areas_l382_382169

theorem box_diagonal_and_areas (a b c : ℝ) :
  let bottom_area := a * b
  let side_area   := b * c
  let front_area  := c * a
  let diagonal    := real.sqrt (a^2 + b^2 + c^2)
  bottom_area * side_area * front_area * (diagonal^2) = a^2 * b^2 * c^2 * (a^2 + b^2 + c^2) := 
  by sorry

end box_diagonal_and_areas_l382_382169


namespace solve_for_constants_l382_382729

def f (x : ℤ) (a b c : ℤ) : ℤ :=
if x > 0 then 2 * a * x + 4
else if x = 0 then a + b
else 3 * b * x + 2 * c

theorem solve_for_constants :
  ∃ a b c : ℤ, 
    f 1 a b c = 6 ∧ 
    f 0 a b c = 7 ∧ 
    f (-1) a b c = -4 ∧ 
    a + b + c = 14 :=
by
  sorry

end solve_for_constants_l382_382729


namespace prime_square_count_l382_382644

-- Define the set of natural numbers whose squares lie between 5000 and 9000
def within_square_range (n : ℕ) : Prop := 5000 < n * n ∧ n * n < 9000

-- Define the predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the set of prime numbers whose squares lie within the range
def primes_within_range : Finset ℕ := (Finset.filter is_prime (Finset.Ico 71 95))

-- The main statement: the number of prime numbers whose squares are between 5000 and 9000
theorem prime_square_count : (primes_within_range.filter within_square_range).card = 6 :=
sorry

end prime_square_count_l382_382644


namespace solve_inequalities_l382_382891

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382891


namespace selling_rate_approx_l382_382165

-- Define the conditions as Lean definitions
def num_bowls_purchased := 114
def cost_per_bowl := 13
def num_bowls_sold := 108
def num_bowls_broken := num_bowls_purchased - num_bowls_sold
def percentage_gain := 23.88663967611336
def total_cost := num_bowls_purchased * cost_per_bowl
def profit := percentage_gain / 100 * total_cost
def total_selling_price := total_cost + profit
def selling_rate_per_bowl := total_selling_price / num_bowls_sold

-- The goal is to state the theorem we aim to prove
theorem selling_rate_approx : selling_rate_per_bowl ≈ 17.00 :=
by sorry

end selling_rate_approx_l382_382165


namespace solve_inequalities_l382_382906

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382906


namespace intersection_M_N_l382_382378

def setM : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def setN : Set ℝ := {x | ∃ y : ℝ, y = sqrt (3 - x^2)}

theorem intersection_M_N :
  {x | x ∈ setM ∧ x ∈ setN} = {x : ℝ | -1 ≤ x ∧ x ≤ sqrt 3} :=
sorry

end intersection_M_N_l382_382378


namespace difference_of_squares_l382_382322

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := 
by
  sorry

end difference_of_squares_l382_382322


namespace solution_set_of_inequalities_l382_382967

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382967


namespace dandelion_white_dandelions_on_saturday_l382_382152

theorem dandelion_white_dandelions_on_saturday :
  ∀ (existsMondayYellow MondayWhite WednesdayYellow WednesdayWhite : ℕ)
    (MondayTotal WednesdayTotal : ℕ)
    (MondayYellow = 20)
    (MondayWhite = 14)
    (MondayTotal = MondayYellow + MondayWhite)
    (WednesdayYellow = 15)
    (WednesdayWhite = 11)
    (WednesdayTotal = WednesdayYellow + WednesdayWhite),
  existsMondayYellow = MondayYellow → existsMondayWhite = MondayWhite →
  WednesdayTotal = 26 →
  (WednesdayTotal - MondayYellow) = 6 →
  WednesdayTotal - MondayYellow - MondayWhite = 6 →
  6 = 6 := 
begin
  intros,
  sorry
end

end dandelion_white_dandelions_on_saturday_l382_382152


namespace area_ratio_l382_382267

variable {α : Type} [LinearOrderedField α]

theorem area_ratio
  (S : α) (λ₁ λ₂ λ₃ : α)
  (h₁ : λ₁ ≥ 0) (h₂ : λ₂ ≥ 0) (h₃ : λ₃ ≥ 0) :
  let S' : α := S * (1 + λ₁ + λ₂ + λ₃ + λ₁ * λ₂ + λ₂ * λ₃ + λ₃ * λ₁)
  in S' / S = 1 + λ₁ + λ₂ + λ₃ + λ₁ * λ₂ + λ₂ * λ₃ + λ₃ * λ₁ :=
by
  sorry

end area_ratio_l382_382267


namespace submarine_rise_l382_382528

theorem submarine_rise (initial_depth final_depth : ℤ) (h_initial : initial_depth = -27) (h_final : final_depth = -18) :
  final_depth - initial_depth = 9 :=
by
  rw [h_initial, h_final]
  norm_num 

end submarine_rise_l382_382528


namespace masking_tape_needed_l382_382228

def wall1_width : ℝ := 4
def wall1_count : ℕ := 2
def wall2_width : ℝ := 6
def wall2_count : ℕ := 2
def door_width : ℝ := 2
def door_count : ℕ := 1
def window_width : ℝ := 1.5
def window_count : ℕ := 2

def total_width_of_walls : ℝ := (wall1_count * wall1_width) + (wall2_count * wall2_width)
def total_width_of_door_and_windows : ℝ := (door_count * door_width) + (window_count * window_width)

theorem masking_tape_needed : total_width_of_walls - total_width_of_door_and_windows = 15 := by
  sorry

end masking_tape_needed_l382_382228


namespace number_of_positive_integers_l382_382247

theorem number_of_positive_integers (n : ℕ) : ∃! k : ℕ, k = 5 ∧
  (∀ n : ℕ, (1 ≤ n) → (12 % (n + 1) = 0)) :=
sorry

end number_of_positive_integers_l382_382247


namespace math_problem_l382_382498

theorem math_problem : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end math_problem_l382_382498


namespace relationship_among_a_b_c_l382_382720

noncomputable def a : ℝ := (1 / 2) ^ (3 / 4)
noncomputable def b : ℝ := (3 / 4) ^ (1 / 2)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_among_a_b_c : a < b ∧ b < c := 
by
  -- Skipping the proof steps
  sorry

end relationship_among_a_b_c_l382_382720


namespace max_blue_cells_n2_max_blue_cells_n25_l382_382049

noncomputable def max_blue_cells (table_size n : ℕ) : ℕ :=
  if h : (table_size = 50 ∧ n = 2) then 2450
  else if h : (table_size = 50 ∧ n = 25) then 1300
  else 0 -- Default case that should not happen for this problem

theorem max_blue_cells_n2 : max_blue_cells 50 2 = 2450 := 
by
  sorry

theorem max_blue_cells_n25 : max_blue_cells 50 25 = 1300 :=
by
  sorry

end max_blue_cells_n2_max_blue_cells_n25_l382_382049


namespace cot15_add_tan45_eq_sqrt2_l382_382040

theorem cot15_add_tan45_eq_sqrt2 :
  (Real.cot (15 * Real.pi / 180) + Real.tan (45 * Real.pi / 180)) = Real.sqrt 2 :=
by
  sorry

end cot15_add_tan45_eq_sqrt2_l382_382040


namespace MN_passes_through_center_of_circumcircle_l382_382726

-- Define the cyclic quadrilateral condition
def cyclic_quadrilateral (A B C D : Point) : Prop :=
  ∃ (O : Point), circle O A ∧ circle O B ∧ circle O C ∧ circle O D

-- Define the perpendicular intersection points
structure perpendicular_intersections (A B C D M N : Point) : Prop :=
  (AM_perpendicular_BA : perpendicular (line_through A M) (line_through B A))
  (AM_inter_CDExt : collinear [A, M, C, D])
  (AN_perpendicular_DA : perpendicular (line_through A N) (line_through D A))
  (AN_inter_BCExt : collinear [A, N, B, C])

-- Define the center of the circumcircle passing through MN
def passes_through_center_of_circumcircle (A B C D M N : Point) : Prop :=
  ∃ (O : Point), cyclic_quadrilateral A B C D ∧ center_of_circumcircle A B C D O ∧
  collinear [M, N, O]

theorem MN_passes_through_center_of_circumcircle
  (A B C D M N : Point)
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : perpendicular_intersections A B C D M N) :
  passes_through_center_of_circumcircle A B C D M N :=
sorry

end MN_passes_through_center_of_circumcircle_l382_382726


namespace minimum_value_of_f_l382_382596

namespace MinimumValueExample

variables {a b c x y z : ℝ}
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variables (h4 : x > 0) (h5 : y > 0) (h6 : z > 0)
variables (h7 : c * y + b * z = a) 
variables (h8 : a * z + c * x = b) 
variables (h9 : b * x + a * y = c) 

def f (x y z : ℝ) := (x^2) / (1 + x) + (y^2) / (1 + y) + (z^2) / (1 + z)

theorem minimum_value_of_f : ∃ (m : ℝ), m = 1 / 2 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → c * y + b * z = a → a * z + c * x = b → b * x + a * y = c → f x y z ≥ m :=
begin
  use 1 / 2,
  intros x y z hx hy hz hc1 hc2 hc3,
  -- Proof goes here
  sorry,
end

end MinimumValueExample

end minimum_value_of_f_l382_382596


namespace abs_inequality_m_eq_neg4_l382_382334

theorem abs_inequality_m_eq_neg4 (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ (m = -4) :=
by
  sorry

end abs_inequality_m_eq_neg4_l382_382334


namespace max_integer_solutions_inf_l382_382167

noncomputable def is_zero_based (p : ℤ[X]) : Prop :=
  p.coefficients.all (λ c, c ∈ ℤ) ∧ p.eval 0 = 0

theorem max_integer_solutions_inf (p : ℤ[X]) 
  (hp : is_zero_based p) : 
  ∃ (S : set ℤ), S.infinite ∧ ∀ k ∈ S, p.eval k = k^2 :=
by
  sorry

end max_integer_solutions_inf_l382_382167


namespace linear_inequalities_solution_l382_382875

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382875


namespace interval_of_monotonic_increase_area_of_triangle_ABC_l382_382625

-- Define the function f(x)
def f (x : ℝ) : ℝ := cos x ^ 2 - sin x ^ 2 + 1 / 2

-- Define the conditions for the triangle ABC
variables (a b : ℝ)
variable H : a = Real.sqrt 19 ∧ b = 5
variable fA_eq : f (a) = 0

noncomputable def area_ABC : ℝ := 1 / 2 * b * 3 * Real.sqrt 3 / 2

-- Statements to prove the two parts of the problem
theorem interval_of_monotonic_increase :
  ∀ (x : ℝ), (0 < x ∧ x < π) → (deriv f x > 0 ↔ (π / 2 ≤ x ∧ x < π)) := sorry

theorem area_of_triangle_ABC :
  ∀ (A : ℝ), (a = Real.sqrt 19 ∧ b = 5 ∧ f (A) = 0) → area_ABC = 15 * Real.sqrt 3 / 4 := sorry

end interval_of_monotonic_increase_area_of_triangle_ABC_l382_382625


namespace solution1_solution2_l382_382690

noncomputable def problem1 (a b : ℝ) (C B c : ℝ) : Prop :=
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B

noncomputable def problem2 (A : ℝ) : Prop :=
  (Real.sqrt 3) * Real.sin (2 * A - (Real.pi / 6)) 
  - 2 * (Real.sin (C - Real.pi / 12)) ^ 2 = 0

theorem solution1 :
  problem1 2 (Real.sqrt 7) C (Real.pi / 3) 3 := sorry

theorem solution2 :
  problem2 (Real.pi / 4) := sorry

end solution1_solution2_l382_382690


namespace solution_set_inequalities_l382_382836

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382836


namespace astronomer_visibility_limit_l382_382180

theorem astronomer_visibility_limit (planet : Type) [is_sphere planet] (asteroids : list point) (h : asteroids.length = 25) :
  ∃ p : point_on planet, ∀ A ∈ asteroids, ¬ (observed_from p A ∧ observed_from (antipodal p) A) →
  ∃ q : point_on planet, (finset.filter (λ a, observed_from q a) (finset.of_list asteroids)).card ≤ 11 :=
sorry

end astronomer_visibility_limit_l382_382180


namespace molecular_weight_of_one_mole_l382_382097

theorem molecular_weight_of_one_mole (molecular_weight_8_moles : ℝ) (h : molecular_weight_8_moles = 992) : 
  molecular_weight_8_moles / 8 = 124 :=
by
  -- proof goes here
  sorry

end molecular_weight_of_one_mole_l382_382097


namespace largest_a_l382_382572

theorem largest_a (a : ℝ) :
  let y := (5 * Real.sqrt 3 - 1 + Real.sqrt ((5 * Real.sqrt 3 - 1)^2 - 20)) / 2 in
  a = Real.sqrt ((y^2 - 2) / 3) ↔
  (5 * Real.sqrt (9 * a^2 + 4) - 5 * a^2 - 2) / (Real.sqrt (2 + 3 * a^2) + 2) = 1 := by
  sorry

end largest_a_l382_382572


namespace solution_set_l382_382851

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382851


namespace smallest_class_number_systematic_sampling_l382_382990

theorem smallest_class_number_systematic_sampling :
  ∀ (x : ℕ), (x > 0) → (x ≤ 18) → (∃ (a b c d : ℕ), a = x ∧ b = x + 6 ∧ c = x + 12 ∧ d = x + 18 ∧ a + b + c + d = 48) → x = 3 :=
by
  intros x hx1 hx2 ⟨a, b, c, d, ha, hb, hc, hd, hs⟩
  sorry

end smallest_class_number_systematic_sampling_l382_382990


namespace solution_set_linear_inequalities_l382_382921

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382921


namespace cartesian_eq_of_curve_length_of_intersection_points_l382_382786

noncomputable def line_param_eq : ℝ × ℝ → Prop :=
λ ⟨x, y⟩, ∃ t : ℝ, x = 1 + t * real.cos (real.pi / 4) ∧ y = t * real.sin (real.pi / 4)

noncomputable def curve_polar_eq : ℝ × ℝ → Prop :=
λ ⟨ρ, θ⟩, ρ * real.sin θ ^ 2 = 4 * real.cos θ

theorem cartesian_eq_of_curve :
  ∀ (x y : ℝ), (∃ ρ θ : ℝ, curve_polar_eq (ρ, θ) ∧ x = ρ * real.cos θ ∧ y = ρ * real.sin θ) ↔ (y^2 = 4 * x) :=
sorry

theorem length_of_intersection_points :
  ∀ (x1 y1 x2 y2 : ℝ), 
  line_param_eq (x1, y1) ∧ line_param_eq (x2, y2) ∧ (y1^2 = 4 * x1) ∧ (y2^2 = 4 * x2) ∧ 
  (y1 = x1 - 1) ∧ (y2 = x2 - 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 1) 
  → real.dist (x1, y1) (x2, y2) = 8 :=
sorry

end cartesian_eq_of_curve_length_of_intersection_points_l382_382786


namespace fraction_of_donations_l382_382452

def max_donation_amount : ℝ := 1200
def total_money_raised : ℝ := 3750000
def donations_from_500_people : ℝ := 500 * max_donation_amount
def fraction_of_money_raised : ℝ := 0.4 * total_money_raised
def num_donors : ℝ := 1500

theorem fraction_of_donations (f : ℝ) :
  donations_from_500_people + num_donors * f * max_donation_amount = fraction_of_money_raised → f = 1 / 2 :=
by
  sorry

end fraction_of_donations_l382_382452


namespace general_term_a_sum_first_n_terms_l382_382276

-- Defining the sequences and their properties
def a (n : ℕ) := 2 * n - 1
def b (n : ℕ) : ℕ := 3^(n - 1)
def c (n : ℕ) :=
  if n ≤ 5 then a n
  else b n

def S (n : ℕ) := if n ≤ 5 then n^2 else (3^n - 679) / 2

-- Theorems based on given conditions
theorem general_term_a (n : ℕ) : a n = 2 * n - 1 :=
begin
  sorry
end

theorem sum_first_n_terms (n : ℕ) : 
  S n =  if n ≤ 5 then n^2 else (3^n - 679) / 2 :=
begin
  sorry
end

end general_term_a_sum_first_n_terms_l382_382276


namespace shopkeeper_profit_percent_l382_382478

theorem shopkeeper_profit_percent
  (cost_price_per_gram : ℝ)
  (faulty_grams : ℝ)
  (correct_grams : ℝ)
  (faulty_grams = 800)
  (correct_grams = 1000)
  (cost_price_per_gram > 0)
  : ((correct_grams - faulty_grams) / faulty_grams) * 100 = 25 := 
sorry

end shopkeeper_profit_percent_l382_382478


namespace solution_set_l382_382848

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l382_382848


namespace find_quadratic_expression_find_range_of_m_find_t_value_l382_382602

-- Condition 1: Minimum value occurs at x = 2
-- Condition 2: Length of line segment on x-axis is 2
-- Function f(x) = ax^2 + bx + 3
def hasMinAt (a b : ℝ) (x : ℝ) : Prop :=
  let f := λ x, a * x^2 + b * x + 3
  ∀ x', f x' ≥ f x

def lengthOfLineSegmentOnXAxis (a b : ℝ) : Prop :=
  let f := λ x, a * x^2 + b * x + 3
  ∃ x1 x2, f x1 = 0 ∧ f x2 = 0 ∧ abs (x1 - x2) = 2

-- Question 1: Find the analytical expression for f(x)
theorem find_quadratic_expression (a b : ℝ) 
  (h1 : hasMinAt a b 2) 
  (h2 : lengthOfLineSegmentOnXAxis a b) :
  ∃ a b, a = 1 ∧ b = -4 ∧ (∀ x, a * x^2 + b * x + 3 = x^2 - 4 * x + 3) := sorry

-- g(x) = f(x) - mx
-- One zero in (0,2), another zero in (2,3)
-- Find the range of m
def g (a b m x : ℝ) : ℝ := (a * x^2 + b * x + 3) - m * x

theorem find_range_of_m (a b : ℝ) 
  (h1 : a = 1) 
  (h2 : b = -4)
  (h3 : ∃ x1 : ℝ, 0 < x1 ∧ x1 < 2 ∧ g a b (-4) x1 = 0)
  (h4 : ∃ x2 : ℝ, 2 < x2 ∧ x2 < 3 ∧ g a b (-4) x2 = 0) :
  -1 / 2 < m ∧ m < 0 := sorry

-- Minimum value of f(x) on [t, t + 1] is -1/2
-- Find the value of t
def minValueInInterval (a b t : ℝ) : Prop :=
  let f := λ x, a * x^2 + b * x + 3
  ∀ x ∈ set.Icc t (t+1), f x ≥ -1/2

theorem find_t_value (a b : ℝ)
  (h1 : a = 1)
  (h2 : b = -4)
  (h3 : minValueInInterval a b t) :
  t = 1 - real.sqrt 2 / 2 ∨ t = 2 + real.sqrt 2 / 2 := sorry

end find_quadratic_expression_find_range_of_m_find_t_value_l382_382602


namespace exist_k_for_sequence_l382_382601

theorem exist_k_for_sequence (n : ℕ) (c : Fin n → ℂ) : 
  ∃ k : Fin n, ∀ (a : Fin n → ℝ), 
    (∀ i : Fin n, 0 ≤ a i ∧ a 0 ≤ 1) ∧ (∀ i j : Fin n, i ≤ j → a i ≥ a j) → 
    (| ∑ i in Finset.range n, a i • c i |) ≤ (| ∑ i in Finset.range (k + 1), c i |) :=
begin
  sorry
end

end exist_k_for_sequence_l382_382601


namespace solution_set_linear_inequalities_l382_382916

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382916


namespace hispanic_population_in_west_l382_382202

theorem hispanic_population_in_west (NE MW South West : ℕ) 
  (Hispanic_NE Hispanic_MW Hispanic_South Hispanic_West : ℕ) 
  (ethnic_table : (NE, MW, South, West) = (Hispanic_NE, Hispanic_MW, Hispanic_South, Hispanic_West)    :=
  (8, 6, 10, 16)) 
  :  (Hispanic_NE + Hispanic_MW + Hispanic_South + Hispanic_West) = 40 → 
     Hispanic_West = 16 → 
     (Hispanic_West : ℝ) / (Hispanic_NE + Hispanic_MW + Hispanic_South + Hispanic_West : ℝ) * 100 = 40 :=
sorry

end hispanic_population_in_west_l382_382202


namespace factory_output_exceeds_by_20_percent_l382_382510

theorem factory_output_exceeds_by_20_percent 
  (planned_output : ℝ) (actual_output : ℝ)
  (h_planned : planned_output = 20)
  (h_actual : actual_output = 24) :
  ((actual_output - planned_output) / planned_output) * 100 = 20 := 
by
  sorry

end factory_output_exceeds_by_20_percent_l382_382510


namespace pairs_count_1432_1433_l382_382580

def PairsCount (n : ℕ) : ℕ :=
  -- The implementation would count the pairs (x, y) such that |x^2 - y^2| = n
  sorry

-- We write down the theorem that expresses what we need to prove
theorem pairs_count_1432_1433 : PairsCount 1432 = 8 ∧ PairsCount 1433 = 4 := by
  sorry

end pairs_count_1432_1433_l382_382580


namespace locus_of_centroids_is_plane_vector_eq_l382_382606

variables {V : Type*} [inner_product_space ℝ V]
variables {O N A B C : V}
variables {u e a b c : V}
variables (x y z : ℝ)

-- Given conditions:
-- O is the vertex of the trihedral angle
-- N is a point such that the sphere passes through O and N
-- Sphere intersects the edges at A, B, and C
-- a, b, c are unit vectors along the edges
-- e = αa + βb + γc is the vector ON
-- Centroid M of triangle ABC
def vector_OA := x • a
def vector_OB := y • b
def vector_OC := z • c
def vector_OM := (1 / 3 : ℝ) • (vector_OA + vector_OB + vector_OC)

-- Proving the locus of centroids M is a plane
theorem locus_of_centroids_is_plane : 
  ∃ p : affine_subspace ℝ V, ∀ O A B C : V, 
    (∃ (a b c e : V), is_unit_vector a ∧ is_unit_vector b ∧ is_unit_vector c ∧
      O = 0 ∧ e = α • a + β • b + γ • c ∧
      A = x • a ∧ B = y • b ∧ C = z • c ∧
      M = (1 / 3 : ℝ) • (O + A + B + C)) -> 
      M ∈ p := 
sorry

noncomputable def is_unit_vector (v : V) : Prop :=
  ⟪v, v⟫ = 1

variables {α β γ : ℝ}

def N_vector := α • a + β • b + γ • c

theorem vector_eq (u v : V) : 
  ⟪u - v, u - v⟫ = 0 ↔ u = v :=
by simp [inner_eq_zero_iff]


end locus_of_centroids_is_plane_vector_eq_l382_382606


namespace sum_mod_9_l382_382544

theorem sum_mod_9 :
  (8 + 77 + 666 + 5555 + 44444 + 333333 + 2222222 + 11111111) % 9 = 3 := 
by sorry

end sum_mod_9_l382_382544


namespace finite_good_not_divisible_by_l382_382373

def tau (n : ℕ) : ℕ :=
  -- Placeholder function for the number of divisors of n,
  -- Assuming an existing implementation in Mathlib or otherwise to be defined.
  sorry

def good (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → tau m < tau n

theorem finite_good_not_divisible_by (k : ℕ) (hk : 1 ≤ k) :
  {n : ℕ | good n ∧ ¬ k ∣ n}.finite :=
  sorry

end finite_good_not_divisible_by_l382_382373


namespace total_value_of_coins_l382_382771

variables {p n : ℕ}

-- Ryan has 17 coins consisting of pennies and nickels
axiom coins_eq : p + n = 17

-- The number of pennies is equal to the number of nickels
axiom pennies_eq_nickels : p = n

-- Prove that the total value of Ryan's coins is 49 cents
theorem total_value_of_coins : (p * 1 + n * 5) = 49 :=
by sorry

end total_value_of_coins_l382_382771


namespace solution_set_linear_inequalities_l382_382931

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l382_382931


namespace zack_traveled_to_18_countries_l382_382109

-- Defining the conditions
variables (countries_traveled_by_george countries_traveled_by_joseph 
           countries_traveled_by_patrick countries_traveled_by_zack : ℕ)

-- Set the conditions as per the problem statement
axiom george_traveled : countries_traveled_by_george = 6
axiom joseph_traveled : countries_traveled_by_joseph = countries_traveled_by_george / 2
axiom patrick_traveled : countries_traveled_by_patrick = 3 * countries_traveled_by_joseph
axiom zack_traveled : countries_traveled_by_zack = 2 * countries_traveled_by_patrick

-- The theorem to prove Zack traveled to 18 countries
theorem zack_traveled_to_18_countries : countries_traveled_by_zack = 18 :=
by
  -- Adding the proof here is unnecessary as per the instructions
  sorry

end zack_traveled_to_18_countries_l382_382109


namespace solution_set_of_inequality_l382_382073

theorem solution_set_of_inequality (x : ℝ) : 
    (∃ x : ℝ, x ∈ setOf (λ x, -1 < x ∧ x < 2)) ↔ (∃ x : ℝ, (x - 2) / (x + 1) < 0) :=
sorry

end solution_set_of_inequality_l382_382073


namespace simplify_power_l382_382411

theorem simplify_power (z : ℂ) (h₁ : z = (1 + complex.I) / (1 - complex.I)) : z ^ 1002 = -1 :=
by 
  sorry

end simplify_power_l382_382411


namespace simplify_expression_l382_382779

theorem simplify_expression (x y : ℝ) (h : (x + 2)^2 + abs (y - 1/2) = 0) :
  (x - 2*y)*(x + 2*y) - (x - 2*y)^2 = -6 :=
by
  -- Proof will be provided here
  sorry

end simplify_expression_l382_382779


namespace vectors_collinear_l382_382116

open_locale big_operators
open_locale real_inner_product_space

variables 
  (a b c₁ c₂ : ℝ × ℝ × ℝ)
  (p : Prop) 

def collinear (v w : ℝ × ℝ × ℝ) : Prop := 
  ∃ γ : ℝ, v.1 = γ * w.1 ∧ v.2 = γ * w.2 ∧ v.3 = γ * w.3

noncomputable def vec_a := (3, -1, 6 : ℝ × ℝ × ℝ)
noncomputable def vec_b := (5, 7, 10 : ℝ × ℝ × ℝ)

noncomputable def vec_c₁ := 
  (4 * vec_a.1 - 2 * vec_b.1, 4 * vec_a.2 - 2 * vec_b.2, 4 * vec_a.3 - 2 * vec_b.3 : ℝ × ℝ × ℝ)
noncomputable def vec_c₂ := 
  (vec_b.1 - 2 * vec_a.1, vec_b.2 - 2 * vec_a.2, vec_b.3 - 2 * vec_a.3 : ℝ × ℝ × ℝ)

theorem vectors_collinear : collinear vec_c₁ vec_c₂ := by
  sorry

end vectors_collinear_l382_382116


namespace zero_in_interval_0_1_l382_382984

noncomputable def f (x : ℝ) := Real.exp x + x - 2

theorem zero_in_interval_0_1 : ∃ x ∈ Ioo 0 1, f x = 0 :=
by
  let a := (0 : ℝ)
  let b := (1 : ℝ)
  have h1 : f a < 0 := by
    simp [f, Real.exp_zero]
    norm_num
  have h2 : f b > 0 := by
    simp [f]
    norm_num
    exact Real.exp_one_lt_e
  refine ⟨c, _, _⟩
  sorry

end zero_in_interval_0_1_l382_382984


namespace calc_result_l382_382819

theorem calc_result : (-3)^2 - (-2)^3 = 17 := 
by
  sorry

end calc_result_l382_382819


namespace no_silver_matrix_for_1997_infinite_silver_matrices_l382_382499

open Finset

-- Definition of a silver matrix
def is_silver_matrix (A : matrix (fin n) (fin n) ℕ) (S : finset ℕ) : Prop :=
  (∀ i : fin n, ∀ j : fin n, A i j ∈ S) ∧ 
  (∀ i : fin n, (univ.bUnion (λ j : fin n, {A i j}) ∪ univ.bUnion (λ j : fin n, {A j i})) = S)

-- Problem (i) statement
theorem no_silver_matrix_for_1997 :
  ¬ ∃ A : matrix (fin 1997) (fin 1997) ℕ, is_silver_matrix A (finset.range (2 * 1997 - 1 + 1)) :=
sorry

-- Problem (ii) statement
theorem infinite_silver_matrices :
  ∃ infinitely_many_n, ∃ A : matrix (fin infinitely_many_n) (fin infinitely_many_n) ℕ, 
  is_silver_matrix A (finset.range (2 * infinitely_many_n - 1 + 1)) :=
sorry

end no_silver_matrix_for_1997_infinite_silver_matrices_l382_382499


namespace area_ratio_and_expression_l382_382751

-- Definitions of the points and conditions in the problem
variable (O : Point) (A B C D M N A' B' C' D' : Point)
variable (Square : Square ABCD) (Midpoints : Midpoints BC AD M N)
variable (Hexagon : EquiangularHexagon A' B' M C' D' N)
variable (Positions : OnLines A' B' C' D' AO BO CO DO)

-- Proof statement
theorem area_ratio_and_expression :
  let ABCD_area := 1
  let A'B'MC'D'N_area := (9 - 4 * Real.sqrt 3) / 8
  let ratio := A'B'MC'D'N_area / ABCD_area
  let a := 9
  let b := -4
  let c := 3
  let d := 8 in
  Int.gcd a b d = 1 →
  1000 * a + 100 * b + 10 * c + d = 8634 := sorry

end area_ratio_and_expression_l382_382751


namespace eddy_climbing_rate_l382_382635

/-- Given the initial conditions:
  - Hillary and Eddy start climbing at 06:00.
  - The summit is 5000 ft from the base camp.
  - Hillary climbs at a rate of 800 ft/hr.
  - Hillary stops 1000 ft short of the summit.
  - Hillary descends at a rate of 1000 ft/hr.
  - Hillary and Eddy pass each other at 12:00.
  Eddie's climbing rate should be 833.3 ft/hr.
-/
theorem eddy_climbing_rate :
  let hillary_rate := 800   -- ft/hr
  let hillary_stop := 1000  -- ft short of summit
  let descent_rate := 1000  -- ft/hr
  let start_time := 0       -- 06:00 as hour 0
  let pass_time := 6        -- 12:00 as hour 6
  let summit_height := 5000 -- ft

  -- hillary climbs 4000 ft in (4000 ft / 800 ft/hr = 5 hr)
  and hillary_descent_time := 1 -- hr
  and hillary_descent_distance := descent_rate * hillary_descent_time
  and total_hillary_distance := summit_height - hillary_stop + hillary_descent_distance

  -- eddy's climbing rate
  in E = total_hillary_distance / pass_time
  -> E = 833.3 :=
sorry

end eddy_climbing_rate_l382_382635


namespace solution_set_system_of_inequalities_l382_382941

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382941


namespace water_volume_container_l382_382524

-- Define variables for the conditions
def side_length : ℝ := 1
def height (h : ℝ) : ℝ := h

-- Definition of cot and tan functions for angles.
noncomputable def cot (θ : ℝ) : ℝ := 1 / (Real.tan θ)
noncomputable def volume (h : ℝ) : ℝ :=
  let cot36 : ℝ := cot (36 * Real.pi / 180)
  let cos108 : ℝ := Real.cos (108 * Real.pi / 180)
  let tan36 : ℝ := Real.tan (36 * Real.pi / 180)
  (5 * h / 8) * cot36 * (1 + cos108 * tan36)^2

-- Theorem statement
theorem water_volume_container (h : ℝ) : volume h = (5 * h / 8) * cot (36 * Real.pi / 180) * (1 + Real.cos (108 * Real.pi / 180) * Real.tan (36 * Real.pi / 180))^2 :=
by
  -- The proof would go here
  sorry

end water_volume_container_l382_382524


namespace solution_set_system_of_inequalities_l382_382945

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382945


namespace evaluate_expression_l382_382231

theorem evaluate_expression:
  let a := 3
  let b := 2
  (a^b)^a - (b^a)^b = 665 :=
by
  sorry

end evaluate_expression_l382_382231


namespace solution_set_of_inequality_l382_382825

theorem solution_set_of_inequality (x : ℝ) : 
  (3 * x - 4 > 2) → (x > 2) :=
by
  intro h
  sorry

end solution_set_of_inequality_l382_382825


namespace range_of_a_l382_382678

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

def has_pos_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, (3 + a * Real.exp (a * x) = 0) ∧ (x > 0)

theorem range_of_a (a : ℝ) : has_pos_extremum a → a < -3 := by
  sorry

end range_of_a_l382_382678


namespace solve_inequalities_l382_382904

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382904


namespace area_difference_is_correct_l382_382348

def right_isosceles_triangle (base_length : ℝ) : Prop :=
  base_length = 1

def area_square (side_length : ℝ) : ℝ :=
  side_length ^ 2

def total_area_fig_2 (side_length : ℝ) : ℝ :=
  2 * area_square side_length

def total_area_fig_3 (side_length : ℝ) : ℝ :=
  2 * area_square side_length

theorem area_difference_is_correct :
  ∀ (x y : ℝ),
    right_isosceles_triangle 1 →
    (x = 1 / 4) →
    (y = real.sqrt 2 / 6) →
    (total_area_fig_2 x - total_area_fig_3 y = 1 / 72) :=
by
  intros x y ht hx hy
  sorry

end area_difference_is_correct_l382_382348


namespace linear_inequalities_solution_l382_382878

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382878


namespace graph_shift_proof_l382_382088

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)
noncomputable def h (x : ℝ) : ℝ := g (x + Real.pi / 8)

theorem graph_shift_proof : ∀ x, h x = f x := by
  sorry

end graph_shift_proof_l382_382088


namespace a_div_c_eq_neg4_l382_382812

-- Given conditions
def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)
def g_inv (x : ℝ) : ℝ := (4 * x + 2) / (3 - x)

-- Theorem to be proved
theorem a_div_c_eq_neg4 : let a := 4, c := -1 in a / c = -4 :=
by {
  exact rfl
}

end a_div_c_eq_neg4_l382_382812


namespace relationship_among_abc_l382_382259

theorem relationship_among_abc
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = 2^(-2/3))
  (hb : b = 2^(-4/3))
  (hc : c = 2^(-1/3)) :
  b < a ∧ a < c :=
by {
  rw [ha, hb, hc],
  sorry
}

end relationship_among_abc_l382_382259


namespace find_inverse_l382_382320

noncomputable def f (x : ℝ) := (x^7 - 1) / 5

theorem find_inverse :
  (f⁻¹ (-1 / 80) = (15 / 16)^(1 / 7)) :=
sorry

end find_inverse_l382_382320


namespace proof1_proof2_l382_382207

open_locale classical

noncomputable theory

variables (ω1 ω2 : Circle) (T : Point)
variables (X : Point) (l1 : Line)
variables (A B S C Y I : Point)
variables (XT YC : Line)

-- Conditions
def conditions := 
  (externally_tangent ω1 ω2 T) ∧  -- Circles ω1 and ω2 are externally tangent at T
  (on_circle X ω1) ∧              -- X is on circle ω1
  (tangent_at l1 ω1 X) ∧          -- l1 is tangent to circle ω1 at X
  (intersects_at l1 ω2 A B) ∧     -- l1 intersects circle ω2 at A and B
  (meets_at XT ω2 S) ∧            -- XT meets circle ω2 at S
  (on_arc C ω2 T S ∧ ¬ (between A C B)) ∧ -- C is on arc TS (not containing A and B)
  (on_circle Y ω1) ∧              -- Y is on circle ω1
  (tangent_at YC ω1 Y) ∧          -- YC is tangent to circle ω1 at Y
  (intersection I (line_combine X Y) (line_combine S C)) -- I is the intersection of lines XY and SC

-- Proof goals
theorem proof1 : conditions ω1 ω2 T X l1 A B S C Y I XT YC →
  ∃ (circ : Circle), on_circle C circ ∧ on_circle T circ ∧ on_circle Y circ ∧ on_circle I circ :=
sorry

theorem proof2 : conditions ω1 ω2 T X l1 A B S C Y I XT YC →
  is_excenter I (triangle A B C) :=
sorry

end proof1_proof2_l382_382207


namespace solve_inequalities_l382_382892

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382892


namespace math_equivalence_problems_l382_382297

def line_parallel_condition (a : ℝ) : Prop :=
  let slope_l1 := - (1 / a)
  let slope_l2 := - (a / (2 * a - 3))
  slope_l1 = slope_l2 ∧ (a ≠ 1 → a = -3)

def line_intercepts_condition (a : ℝ) : Prop :=
  let x_intercept := if (2 * a - 3 != 0) then (a - 2) / (2 * a - 3) else 0
  let y_intercept := if a != 0 then (a - 2) / a else 0
  x_intercept = y_intercept → (a = 1 ∨ a = 2)

def line_perpendicular_condition (a : ℝ) : Prop :=
  let slope_l1 := - (1 / a)
  let slope_l2 := - (a / (2 * a - 3))
  slope_l1 * slope_l2 = -1 → (a = 0 ∨ a = 2)

def line_distance_condition (a : ℝ) : Prop :=
  a = -3 →
  let distance := (|3 - (5/3)|) / (Real.sqrt (1 ^ 2 + (-3) ^ 2))
  distance = 2 * Real.sqrt(10) / 15

theorem math_equivalence_problems :
  (line_parallel_condition a) ∧
  (line_intercepts_condition a) ∧
  (line_perpendicular_condition a) ∧
  (line_distance_condition a) :=
by
  sorry

end math_equivalence_problems_l382_382297


namespace sequence_properties_l382_382609

noncomputable def a_n (n : ℕ) : ℕ → ℝ := 
λ n, (1 / 3)^(n - 1)

noncomputable def b_n (n : ℕ) : ℕ → ℝ := 
λ n, 1 / (n * (n + 1))

noncomputable def T_n (n : ℕ) : ℝ := 
∑ i in finset.range n, b_n i

-- Lean 4 statement for the proof problem
theorem sequence_properties :
  ∀ (n : ℕ), (a_n 1 = 1) ∧ (∑ i in finset.range 1, a_n i,
                               2 * ∑ i in finset.range 2, a_n i,
                               3 * ∑ i in finset.range 3, a_n i).eq ⟨1, _⟩  -- Use constraints for forming arithmetic sequence
  ∧ ( ∀ n, T_n n = ↑n / (↑n + 1) )
:= sorry

end sequence_properties_l382_382609


namespace interval_cardinality_continuum_open_l382_382113

open Set

theorem interval_cardinality_continuum_open {α : Type*} [LinearOrderedField α] : |(Ioo 0 1 : Set α)| = cardinal.cont :=
sorry

end interval_cardinality_continuum_open_l382_382113


namespace equation1_no_solution_equation2_solution_l382_382415

/-- Prove that the equation (4-x)/(x-3) + 1/(3-x) = 1 has no solution. -/
theorem equation1_no_solution (x : ℝ) : x ≠ 3 → ¬ (4 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by intro hx; sorry

/-- Prove that the equation (x+1)/(x-1) - 6/(x^2-1) = 1 has solution x = 2. -/
theorem equation2_solution (x : ℝ) : x = 2 ↔ (x + 1) / (x - 1) - 6 / (x^2 - 1) = 1 :=
by sorry

end equation1_no_solution_equation2_solution_l382_382415


namespace masha_wins_l382_382015

def board_size : Nat := 5

def chip_placement (board : Fin board_size → Fin board_size → Bool) : Prop :=
  ∃ (chip_count : Nat), chip_count = 9 ∧
  (∀ (i j : Fin board_size), board i j = true → 
    ∀ (t : Fin 3 → Fin 3 → Bool), ¬(∀ (x y : Fin 3), t x y → 
        ∃ (dx dy : Fin 3), i + dx < board_size ∧ j + dy < board_size ∧
        board (i + dx) (j + dy) = true))

theorem masha_wins : ∃ (board : Fin board_size → Fin board_size → Bool), chip_placement board :=
sorry

end masha_wins_l382_382015


namespace conditional_probability_partition_l382_382407

open ProbabilityTheory

variables {Ω : Type*} [MeasurableSpace Ω]
variables {A B C : Set Ω} {P : Measure Ω}

-- Assuming P(B) > 0
hypothesis (h : 0 < P B)

theorem conditional_probability_partition 
  (h : 0 < P B) : 
  P[A | B] = P[A | B ∩ C] * P[C | B] + P[A | B ∩ Cᶜ] * P[Cᶜ | B] :=
by
  sorry

end conditional_probability_partition_l382_382407


namespace solution_set_system_of_inequalities_l382_382949

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382949


namespace red_basket_fruit_count_l382_382082

-- Defining the basket counts
def blue_basket_bananas := 12
def blue_basket_apples := 4
def blue_basket_fruits := blue_basket_bananas + blue_basket_apples
def red_basket_fruits := blue_basket_fruits / 2

-- Statement of the proof problem
theorem red_basket_fruit_count : red_basket_fruits = 8 := by
  sorry

end red_basket_fruit_count_l382_382082


namespace true_proposition_l382_382532

universe u

/-- Definition for a line passing through a fixed point P0 -/
def line_through_fixed_point (x0 y0 k : ℝ) := ∀ x y : ℝ, y - y0 = k * (x - x0)

/-- Definition for a line passing through any two different points -/
def line_through_two_points (x1 y1 x2 y2 x y : ℝ) := (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

/-- Definition for a line not passing through the origin -/
def line_not_passing_origin (a b x y : ℝ) := x / a + y / b = 1

/-- Definition for a line passing through a fixed point (0, b) -/
def line_through_point_A (b k x y : ℝ) := y = k * x + b

/-- Theorem: Among the given propositions, the second one is true -/
theorem true_proposition 
  (x0 y0 k x1 y1 x2 y2 a b x y : ℝ)
  (h1 : x ≠ 2)
  (h3 : x ≠ 5)
  (h4 : x1 ≠ x2 ∧ y1 ≠ y2) :
  ¬line_through_fixed_point x0 y0 k x y →
  line_through_two_points x1 y1 x2 y2 x y →
  ¬line_not_passing_origin a b x y →
  ¬line_through_point_A b k x y →
  (∃ (x1 y1 x2 y2 : ℝ), line_through_two_points x1 y1 x2 y2 x y) :=
sorry

end true_proposition_l382_382532


namespace bridge_length_l382_382115

theorem bridge_length (train_length : ℤ) (train_speed_kmph : ℤ) (crossing_time_sec : ℤ) :
  train_length = 156 → train_speed_kmph = 45 → crossing_time_sec = 40 →
  let train_speed_mps := train_speed_kmph * 1000 / 3600 in
  let total_distance := train_speed_mps * crossing_time_sec in
  let bridge_length := total_distance - train_length in
  bridge_length = 344 :=
by
  -- given the known values and conditions
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- calculate speed in m/s, total distance, and bridge length
  let train_speed_mps := 45 * 1000 / 3600
  let total_distance := train_speed_mps * 40
  let bridge_length := total_distance - 156
  -- assert the result
  calc
    bridge_length = (45 * 1000 / 3600) * 40 - 156 : by rw [h2]; reflexivity
                 ... = 500 - 156                     : by norm_num
                 ... = 344                           : by norm_num

end bridge_length_l382_382115


namespace major_axis_length_l382_382196

-- Definitions of the given conditions
structure Ellipse :=
  (focus1 focus2 : ℝ × ℝ)
  (tangent_to_x_axis : Bool)

noncomputable def length_of_major_axis (E : Ellipse) : ℝ :=
  let (x1, y1) := E.focus1
  let (x2, y2) := E.focus2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 + y1) ^ 2)

-- The theorem we want to prove given the conditions
theorem major_axis_length (E : Ellipse)
  (h1 : E.focus1 = (9, 20))
  (h2 : E.focus2 = (49, 55))
  (h3 : E.tangent_to_x_axis = true):
  length_of_major_axis E = 85 :=
by
  sorry

end major_axis_length_l382_382196


namespace _l382_382554

noncomputable def tetrahedron (S A B C : Point) : Prop := sorry
noncomputable def incircle_center (ABC : Triangle) (I : Point) : Prop := sorry
noncomputable def touches_sides (I D E F : Point) (ABC : Triangle) : Prop := sorry 
noncomputable def on_segments_eq (S A' B' C' A B C S I D E F : Point) : Prop := sorry
noncomputable def diametrically_opposite (S S' : Point) (circumsphere : Sphere) : Prop := sorry
noncomputable def altitude (SI : Line) (SABC : Tetrahedron) : Prop := sorry

def main_theorem (S A B C A' B' C' I D E F S' : Point) (SI : Line) (circumsphere : Sphere) :
  tetrahedron S A B C → incircle_center (ABC.triangle) I →
  touches_sides I D E F (ABC.triangle) →
  on_segments_eq S A' B' C' A B C S I D E F →
  diametrically_opposite S S' circumsphere →
  altitude SI (SABC.tetrahedron) →
  S'A' = S'B' ∧ S'B' = S'C' :=
by
  intros
  sorry

end _l382_382554


namespace correct_statements_count_l382_382694

/- Define the statements as Boolean values based on the problem conditions -/
def statement1 : Prop := ¬(correlation_is_deterministic)
def statement2 : Prop := (x_is_explanatory ∧ y_hat_is_predicted)
def statement3 : Prop := (closer_R_squared_is_better_regression)
def statement4 : Prop := ¬(larger_ad_bc_weaker_relationship)
def statement5 : Prop := (uniform_residuals_higher_accuracy)

/- Theorem to assert the number of true statements is 3 -/
theorem correct_statements_count : 
  (statement1 = false) →
  (statement2 = true) →
  (statement3 = true) →
  (statement4 = false) →
  (statement5 = true) →
  3 = 3 :=
by sorry

end correct_statements_count_l382_382694


namespace train_speed_l382_382186

theorem train_speed
  (length_m : ℝ)
  (time_s : ℝ)
  (h_length : length_m = 280.0224)
  (h_time : time_s = 25.2) :
  (length_m / 1000) / (time_s / 3600) = 40.0032 :=
by
  sorry

end train_speed_l382_382186


namespace harpy_count_l382_382688

-- We define the species of birds as either Phoenix (true) or Harpy (false)
inductive Species
| Phoenix : Species
| Harpy : Species

open Species

-- Define each bird's species
def Anne_species : Species := sorry
def Bob_species : Species := sorry
def Charles_species : Species := sorry
def Donna_species : Species := sorry
def Emily_species : Species := sorry

-- Represent the given conditions as Lean statements
def Anne_stmt : Prop := (Anne_species = Charles_species) → False
def Bob_stmt : Prop := (Donna_species = Harpy)
def Charles_stmt : Prop := (Bob_species = Harpy)
def Donna_stmt : Prop := ((Anne_species = Phoenix ∧ Bob_species = Phoenix ∧ Charles_species = Phoenix) ∨ (Anne_species = Phoenix ∧ Bob_species = Phoenix ∧ Emily_species = Phoenix) ∨ (Anne_species = Phoenix ∧ Charles_species = Phoenix ∧ Emily_species = Phoenix) ∨ (Bob_species = Phoenix ∧ Charles_species = Phoenix ∧ Emily_species = Phoenix) ∨ (Anne_species = Phoenix ∧ Donna_species = Phoenix ∧ Emily_species = Phoenix))
def Emily_stmt : Prop := (Anne_species = Phoenix)

-- The proof goal: proving that the number of harpies is 2
def proof_goal : Prop := 
  let n_harpies := [Anne_species, Bob_species, Charles_species, Donna_species, Emily_species].countp (λ s, s = Harpy) in
  n_harpies = 2

-- Combining conditions and proof goal
theorem harpy_count : (Anne_stmt ∧ Bob_stmt ∧ Charles_stmt ∧ Donna_stmt ∧ Emily_stmt) → proof_goal :=
by
  sorry

end harpy_count_l382_382688


namespace solution_set_of_linear_inequalities_l382_382857

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l382_382857


namespace min_value_3x_y_correct_l382_382004

open Real

noncomputable def min_value_3x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : ℝ :=
3x + y

theorem min_value_3x_y_correct {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  min_value_3x_y x y hx hy h = 1 + 8 * sqrt 3 :=
sorry

end min_value_3x_y_correct_l382_382004


namespace num_solutions_l382_382638

def cond1 (x y : ℝ) : Prop := 2 * x + 4 * y = 4
def cond2 (x y : ℝ) : Prop := | |x| - |y| | = 2

theorem num_solutions : 
  {p : ℝ × ℝ // cond1 p.1 p.2 ∧ cond2 p.1 p.2}.toList.length = 2 :=
by
  sorry

end num_solutions_l382_382638


namespace roots_of_equation_l382_382578

theorem roots_of_equation :
  (∃ z : ℂ, z ^ 2 + 2 * z = 10 - 2 * complex.I) ↔
  (z = 3 - complex.I ∨ z = 3 + complex.I ∨ z = -1 - 4 * complex.I ∨ z = -5 - complex.I ∨ z = -5 + complex.I ∨ z = -1 + 4 * complex.I) :=
sorry

end roots_of_equation_l382_382578


namespace negation_of_proposition_l382_382058

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
by 
  sorry

end negation_of_proposition_l382_382058


namespace xy_value_l382_382681

theorem xy_value (x y : ℝ) (h₁ : x + y = 2) (h₂ : x^2 * y^3 + y^2 * x^3 = 32) :
  x * y = 2^(5/3) :=
by
  sorry

end xy_value_l382_382681


namespace solve_inequalities_l382_382889

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382889


namespace choose_four_socks_with_red_l382_382226

open Finset
open Fintype

noncomputable def socks : Finset String := 
  {"blue", "brown", "black", "red", "purple", "orange", "green", "yellow"}

theorem choose_four_socks_with_red :
  ∃ (s : Finset String), s.card = 4 ∧ "red" ∈ s ∧ (s ∈ (socks.powerset.filter (λ t, t.card = 4))) :=
  begin
    sorry
  end

end choose_four_socks_with_red_l382_382226


namespace drawings_on_last_sheet_l382_382014

-- Definitions based on the conditions
def notebooks := 12
def sheets_per_notebook := 36
def drawings_per_sheet_old := 8
def drawings_per_sheet_new := 5
def notebooks_used := 6
def full_sheets_in_last_notebook := 29
def total_drawings := notebooks * sheets_per_notebook * drawings_per_sheet_old
def total_sheets_needed_new := total_drawings / drawings_per_sheet_new
def actual_sheets_needed_new : nat := total_sheets_needed_new.ceil.to_nat -- Rounding up to the next integer
def total_sheets_used_new := notebooks_used * sheets_per_notebook + full_sheets_in_last_notebook + 1
def remaining_drawings := total_drawings - drawings_per_sheet_new * (actual_sheets_needed_new - 1)

-- Statement to prove
theorem drawings_on_last_sheet : remaining_drawings = 1 := by
  sorry

end drawings_on_last_sheet_l382_382014


namespace closest_root_of_equation_l382_382103

theorem closest_root_of_equation : 
  ∃ (x : ℤ), (3 : ℤ) = x ∧ abs (x^3 - 25) ≤ abs (y^3 - 25) ∀ (y : ℤ), y ≠ 3 := 
by
  sorry

end closest_root_of_equation_l382_382103


namespace solution_set_system_of_inequalities_l382_382946

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382946


namespace symmetrical_lines_intersect_l382_382031

theorem symmetrical_lines_intersect :
  ∀ (A B C : Point) (H : Point), is_orthocenter A B C H →
  ∀ (l : Line), 
  let l_BC := symmetric_line l H (line BC) in
  let l_CA := symmetric_line l H (line CA) in
  let l_AB := symmetric_line l H (line AB) in
  ∃ P : Point, 
    P ∈ l_BC ∧ P ∈ l_CA ∧ P ∈ l_AB :=
sorry

end symmetrical_lines_intersect_l382_382031


namespace exponent_multiplication_identity_l382_382469

theorem exponent_multiplication_identity : 2^4 * 3^2 * 5^2 * 7 = 6300 := sorry

end exponent_multiplication_identity_l382_382469


namespace hyperbola_problem_l382_382718

theorem hyperbola_problem
  (F1 F2 : ℝ × ℝ)
  (h k a b : ℝ)
  (hF1 : F1 = (-3, 1 - (Real.sqrt 5) / 4))
  (hF2 : F2 = (-3, 1 + (Real.sqrt 5) / 4))
  (hP : ∀ P : ℝ × ℝ, (Real.abs (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) - Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) = 1)
  (hEq : (1 : ℝ) = 1)
  (hPos_a : a > 0)
  (hPos_b : b > 0)
  (h_center : (h, k) = (-3, 1))
  (h_a : a = 1/2)
  (h_c : Real.sqrt ((k - (1 - Real.sqrt 5 / 4))^2) = Real.sqrt ((k - (1 + Real.sqrt 5 / 4))^2) - 1)
  (b_eq : b = 1/4) :
  h + k + a + b = -5/4 :=
by
  sorry

end hyperbola_problem_l382_382718


namespace factorize_diff_of_squares_l382_382233

theorem factorize_diff_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
  sorry

end factorize_diff_of_squares_l382_382233


namespace mother_to_father_age_ratio_l382_382557

def DarcieAge : ℕ := 4
def FatherAge : ℕ := 30
def MotherAge : ℕ := DarcieAge * 6

theorem mother_to_father_age_ratio :
  (MotherAge : ℚ) / (FatherAge : ℚ) = (4 / 5) := by
  sorry

end mother_to_father_age_ratio_l382_382557


namespace solution_set_of_inequalities_l382_382978

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382978


namespace arithmetic_sequence_a6_l382_382351

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_root1 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 2 = x)
  (h_root2 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 10 = x) : 
  a 6 = -6 := 
by
  sorry

end arithmetic_sequence_a6_l382_382351


namespace sum_of_valid_6_digit_numbers_divisible_by_37_l382_382763

def isValidDigit (d : ℕ) : Prop := d ≠ 0 ∧ d ≠ 9

def isValid6DigitNumber (n : ℕ) : Prop :=
  n ≥ 100000 ∧
  n < 1000000 ∧
  (∀ i : ℕ, i < 6 → isValidDigit (n / 10^i % 10))

theorem sum_of_valid_6_digit_numbers_divisible_by_37 :
  ∃ S : ℕ, (∀ n : ℕ, isValid6DigitNumber n → S = Σ n) → S % 37 = 0 :=
sorry

end sum_of_valid_6_digit_numbers_divisible_by_37_l382_382763


namespace share_of_A_is_23200_l382_382224

-- Definitions of given conditions
def total_amount : ℝ := 232000
def ratio_a_b : ℝ × ℝ := (2, 3)
def ratio_b_c : ℝ × ℝ := (4, 5)
def ratio_c_d : ℝ × ℝ := (3, 4)
def ratio_d_e : ℝ × ℝ := (4, 5)

-- Theorem stating the final result
theorem share_of_A_is_23200 :
  let part_sum := 8 + 12 + 15 + 20 + 25 in
  let part_value := total_amount / part_sum in
  let share_A := 8 * part_value in
  share_A = 23200 :=
by
  let part_sum : ℝ := 8 + 12 + 15 + 20 + 25
  let part_value : ℝ := total_amount / part_sum
  let share_A : ℝ := 8 * part_value
  sorry

end share_of_A_is_23200_l382_382224


namespace range_f_l382_382293

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 1 else 2^x

theorem range_f : set.range f = set.Ici 1 := 
by
  sorry

end range_f_l382_382293


namespace ellipse_proof_l382_382608

noncomputable def ellipse_eq : (a b : ℝ) (M : ℝ × ℝ) (e : ℝ) :=
  a > b ∧ b > 0 ∧ e = 1 / 2 ∧ M = (0, Real.sqrt 3) ∧ 
  (a = 2 ∧ b = Real.sqrt 3 ∧ (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 / 4 + p.2 ^ 2 / 3 = 1})) ∧ 
  (∃ (P : ℝ × ℝ), P = (-Real.sqrt 3, 0) ∧ 
   (∀ (d1 d2 : ℝ), d1 ^ 2 + d2 ^ 2 = (P.1) ^ 2 + (P.2 - Real.sqrt 3) ^ 2 ∧ 
   Real.sqrt (d1 ^ 2 + d2 ^ 2) = 2 * Real.sqrt 3))

theorem ellipse_proof : ellipse_eq 2 (Real.sqrt 3) (0, Real.sqrt 3) (1 / 2) :=
by {
  sorry
}

end ellipse_proof_l382_382608


namespace sara_total_spent_l382_382024

def cost_of_tickets := 2 * 10.62
def cost_of_renting := 1.59
def cost_of_buying := 13.95
def total_spent := cost_of_tickets + cost_of_renting + cost_of_buying

theorem sara_total_spent : total_spent = 36.78 := by
  sorry

end sara_total_spent_l382_382024


namespace f_sum_positive_l382_382291

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem f_sum_positive (x₁ x₂ x₃ : ℝ) (h₁₂ : x₁ + x₂ > 0) (h₂₃ : x₂ + x₃ > 0) (h₃₁ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := 
sorry

end f_sum_positive_l382_382291


namespace probability_divisible_by_3_l382_382479

theorem probability_divisible_by_3 :
  ∀ (n : ℤ), (1 ≤ n) ∧ (n ≤ 99) → 3 ∣ (n * (n + 1)) :=
by
  intros n hn
  -- Detailed proof would follow here
  sorry

end probability_divisible_by_3_l382_382479


namespace last_digit_to_appear_is_1_l382_382380

noncomputable def modified_fib : ℕ → ℕ
| 1     := 2
| 2     := 2
| (n+3) := modified_fib (n+2) + modified_fib (n+1)

def mod_fib (n : ℕ) : ℕ :=
(modified_fib n) % 10

theorem last_digit_to_appear_is_1 :
  ∃ n, ∀ m ≥ n, (∃ k < m, mod_fib k % 10 = 1) → (∃ k < m, mod_fib k % 10 = 2) →
  (∃ k < m, mod_fib k % 10 = 3) → (∃ k < m, mod_fib k % 10 = 4) →
  (∃ k < m, mod_fib k % 10 = 5) → (∃ k < m, mod_fib k % 10 = 6) →
  (∃ k < m, mod_fib k % 10 = 7) → (∃ k < m, mod_fib k % 10 = 8) →
  (∃ k < m, mod_fib k % 10 = 9) ↓ (∃ k < m, mod_fib k % 10 = 0) :=
sorry

end last_digit_to_appear_is_1_l382_382380


namespace solution_set_of_inequalities_l382_382975

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382975


namespace a_10_equals_21_l382_382430

noncomputable def a_1 : ℤ := 3
noncomputable def b (n : ℕ) : ℤ
noncomputable def a (n : ℕ) : ℤ

axiom H1 : b 3 = -2
axiom H2 : b 10 = 12

axiom seq_def : ∀ n : ℕ, a (n + 1) = a n + b n

theorem a_10_equals_21 : a 10 = 21 :=
by
  sorry

end a_10_equals_21_l382_382430


namespace astronaut_arrangement_count_l382_382056

-- Definitions
variables (A B C D : Type) -- Representing the four astronauts
variables (Tianhe Wentian Mengtian : Type) -- Representing the three modules

-- Assumptions
axiom astronaut_count : fintype.card (fin 4) = 4
axiom module_count : fintype.card (fin 3) = 3

-- Condition that each module must have at least one astronaut
-- and A, B must be in different modules.
def valid_arrangement (f : fin 4 → fin 3) : Prop :=
  (∀ m, ∃ i, f i = m) ∧ f (by exact 0) ≠ f (by exact 1)

-- The theorem statement
theorem astronaut_arrangement_count : 
  ∃ (n : ℕ), n = 30 ∧ fintype.card {f // valid_arrangement f} = n :=
begin
  sorry
end

end astronaut_arrangement_count_l382_382056


namespace cos_neg_135_eq_l382_382551

noncomputable def cosine_neg_135 : ℝ :=
  Real.cos (Real.Angle.ofRealDegree (-135.0))

theorem cos_neg_135_eq :
  cosine_neg_135 = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_135_eq_l382_382551


namespace dogs_eat_each_day_l382_382230

theorem dogs_eat_each_day (h1 : 0.125 + 0.125 = 0.25) : true := by
  sorry

end dogs_eat_each_day_l382_382230


namespace pies_sold_in_week_l382_382172

def daily_pies : ℕ := 8
def days_in_week : ℕ := 7

theorem pies_sold_in_week : daily_pies * days_in_week = 56 := by
  sorry

end pies_sold_in_week_l382_382172


namespace solution_set_of_inequalities_l382_382964

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382964


namespace value_of_a_l382_382303

theorem value_of_a (a : ℝ) : (∃ x ∈ set.Icc (1 : ℝ) (3 : ℝ), x^2 + 2 * x - a ≥ 0) → a ≤ 15 :=
by
  sorry

end value_of_a_l382_382303


namespace linear_inequalities_solution_l382_382871

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382871


namespace distance_between_A_and_B_l382_382117

theorem distance_between_A_and_B 
  (v_pas0 v_freight0 : ℝ) -- original speeds of passenger and freight train
  (t_freight : ℝ) -- time taken by freight train
  (d : ℝ) -- distance sought
  (h1 : t_freight = d / v_freight0) 
  (h2 : d + 288 = v_pas0 * t_freight) 
  (h3 : (d / (v_freight0 + 10)) + 2.4 = d / (v_pas0 + 10))
  : d = 360 := 
sorry

end distance_between_A_and_B_l382_382117


namespace rick_iron_clothing_l382_382768

theorem rick_iron_clothing :
  let shirts_per_hour := 4
  let pants_per_hour := 3
  let jackets_per_hour := 2
  let hours_shirts := 3
  let hours_pants := 5
  let hours_jackets := 2
  let total_clothing := (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants) + (jackets_per_hour * hours_jackets)
  total_clothing = 31 := by
  sorry

end rick_iron_clothing_l382_382768


namespace solve_inequalities_l382_382905

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382905


namespace solution_set_system_of_inequalities_l382_382940

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382940


namespace number_of_cuboids_painted_l382_382531

/--
Suppose each cuboid has 6 outer faces and Amelia painted a total of 36 faces.
Prove that the number of cuboids Amelia painted is 6.
-/
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) 
  (h1 : total_faces = 36) (h2 : faces_per_cuboid = 6) :
  total_faces / faces_per_cuboid = 6 := 
by {
  sorry
}

end number_of_cuboids_painted_l382_382531


namespace sum_xy_l382_382767

theorem sum_xy (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 10) : x + y = 14 ∨ x + y = -2 :=
sorry

end sum_xy_l382_382767


namespace amount_of_bill_is_1575_l382_382079

noncomputable def time_in_years := (9 : ℝ) / 12

noncomputable def true_discount := 189
noncomputable def rate := 16

noncomputable def face_value (TD : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (TD * 100) / (R * T)

theorem amount_of_bill_is_1575 :
  face_value true_discount rate time_in_years = 1575 := by
  sorry

end amount_of_bill_is_1575_l382_382079


namespace round_pi_nearest_l382_382244

-- Define the rounding function for non-negative rational numbers
def round_to_nearest (x : ℚ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌊x⌋ + 1

-- The target theorem to prove
theorem round_pi_nearest : round_to_nearest (real.to_rat real.pi) = 3 := 
  sorry

end round_pi_nearest_l382_382244


namespace ratio_of_radii_l382_382526

theorem ratio_of_radii (V_L : ℝ) (V_S : ℝ) (r_L r_S : ℝ) (h1 : V_L = 450 * Real.pi) (h2 : V_S = 0.08 * V_L) 
  (h3 : (4 / 3) * Real.pi * r_L^3 = V_L) (h4 : (4 / 3) * Real.pi * r_S^3 = V_S) :
  r_S / r_L = Real.cbrt(2) / 5 := 
by 
  sorry

end ratio_of_radii_l382_382526


namespace solve_inequalities_l382_382890

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382890


namespace prime_squares_5000_9000_l382_382654

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end prime_squares_5000_9000_l382_382654


namespace solution_set_system_of_inequalities_l382_382951

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l382_382951


namespace gen_term_formula_correct_l382_382801

-- Defining a sequence 'a_n' where the denominator follows a geometric sequence, the numerator follows 2n+1,
-- and the sign alternates based on (-1)^(n+1)
def seq (n : ℕ) : ℝ :=
  (-1)^(n+1) * (2 * n + 1) / 2^n

-- Stating the theorem that the general term 'a_n' matches the given conditions and calculates the correct sequence
theorem gen_term_formula_correct (n : ℕ) : seq n = (-1)^(n+1) * (2 * n + 1) / 2^n :=
by
  -- This will be the place for the proof, but it is left out as per instructions.
  sorry

end gen_term_formula_correct_l382_382801


namespace salt_solution_concentration_l382_382506

theorem salt_solution_concentration :
  ∀ (C : ℝ),
  (∀ (mix_vol : ℝ) (pure_water : ℝ) (salt_solution_vol : ℝ),
    mix_vol = 1.5 →
    pure_water = 1 →
    salt_solution_vol = 0.5 →
    1.5 * 0.15 = 0.5 * (C / 100) →
    C = 45) :=
by
  intros C mix_vol pure_water salt_solution_vol h_mix h_pure h_salt h_eq
  sorry

end salt_solution_concentration_l382_382506


namespace fraction_is_determined_l382_382157

theorem fraction_is_determined (y x : ℕ) (h1 : y * 3 = x - 1) (h2 : (y + 4) * 2 = x) : 
  y = 7 ∧ x = 22 :=
by
  sorry

end fraction_is_determined_l382_382157


namespace polynomial_divisible_by_7_l382_382246

theorem polynomial_divisible_by_7 (n : ℤ) : 7 ∣ ((n + 7)^2 - n^2) :=
sorry

end polynomial_divisible_by_7_l382_382246


namespace inverse_ratio_l382_382808

def g (x : ℝ) := (3 * x - 2) / (x + 4)

noncomputable def g_inv (x : ℝ) := (4 * x + 2) / (3 - x)

theorem inverse_ratio (a b c d : ℝ) (h1 : g_inv x = (a * x + b) / (c * x + d)) : a / c = -4 :=
by {
    sorry
}

end inverse_ratio_l382_382808


namespace roots_exist_l382_382623

theorem roots_exist (a : ℝ) : ∃ x : ℝ, a * x^2 - x = 0 := by
  sorry

end roots_exist_l382_382623


namespace upper_left_region_l382_382697

theorem upper_left_region (t : ℝ) : (2 - 2 * t + 4 ≤ 0) → (t ≤ 3) :=
by
  sorry

end upper_left_region_l382_382697


namespace altitude_in_triangle_l382_382683

open Real

-- Define the sides and angle in the triangle
def AB : ℝ := 2
def angleA : ℝ := π / 3 -- equivalent to 60 degrees in radians
def BC : ℝ := sqrt 7

-- The result we aim to prove
def altitude_length : ℝ := 3 * (sqrt 3) / 2

-- State the theorem
theorem altitude_in_triangle (AB BC angleA : ℝ) (h : ℝ) :
  AB = 2 → BC = sqrt 7 → angleA = π / 3 → h = (3 * (sqrt 3)) / 2 :=
by
  sorry

end altitude_in_triangle_l382_382683


namespace mary_lambs_count_l382_382743

def initial_lambs : Nat := 6
def baby_lambs : Nat := 2 * 2
def traded_lambs : Nat := 3
def extra_lambs : Nat := 7

theorem mary_lambs_count : initial_lambs + baby_lambs - traded_lambs + extra_lambs = 14 := by
  sorry

end mary_lambs_count_l382_382743


namespace kindergarten_children_count_l382_382684

theorem kindergarten_children_count (D B C : ℕ) (hD : D = 18) (hB : B = 6) (hC : C + B = 12) : D + C + B = 30 :=
by
  sorry

end kindergarten_children_count_l382_382684


namespace hyperbola_eccentricity_range_l382_382630

-- Define the hyperbola and its properties
def hyperbola_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the conditions
variables (a b c e : ℝ) (P : ℝ × ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)

-- Define the foci locations
def foci_distance : ℝ := c -- Distance from the center to each focus
def eccentricity : ℝ := c / a

-- Condition that P is on the hyperbola C
variable (P_on_hyperbola : hyperbola_eq a b P.1 P.2)

-- Condition |PF1| = 3a where F1 is the left focus (without loss of generality, assume F1 is on the negative x-axis)
variable (PF1_eq_3a : dist P (-(c, 0)) = 3 * a)

-- Define the range of eccentricity to be proven
def range_eccentricity : Prop := 2 < e ∧ e ≤ 4

-- The theorem statement
theorem hyperbola_eccentricity_range : 
  foci_distance = c ∧ eccentricity = e ∧ PF1_eq_3a → range_eccentricity := sorry

end hyperbola_eccentricity_range_l382_382630


namespace count_primes_squared_in_range_l382_382655

theorem count_primes_squared_in_range : 
  (finset.card (finset.filter (λ p, p^2 ≥ 5000 ∧ p^2 ≤ 9000) 
    (finset.filter nat.prime (finset.range 95)))) = 5 := 
by sorry

end count_primes_squared_in_range_l382_382655


namespace toys_produced_each_day_l382_382156

-- Definitions of conditions
axiom total_weekly_production : ℕ := 6000
axiom days_per_week : ℕ := 4

-- Theorem stating the toys produced each day is 1500
theorem toys_produced_each_day (h1 : total_weekly_production = 6000) 
                              (h2 : days_per_week = 4) :
  6000 / 4 = 1500 :=
by
  sorry

end toys_produced_each_day_l382_382156


namespace polynomial_div_by_derivative_iff_l382_382030

variable {α : Type} [Field α]

theorem polynomial_div_by_derivative_iff (P : Polynomial α) :
  (∃ Q : Polynomial α, P = Q * P.deriv) ↔ ∃ (a_n : α) (x_0 : α) (n : ℕ), P = Polynomial.C a_n * (Polynomial.X - Polynomial.C x_0) ^ n := sorry

end polynomial_div_by_derivative_iff_l382_382030


namespace ratio_of_perimeters_of_similar_triangles_l382_382285

theorem ratio_of_perimeters_of_similar_triangles (A1 A2 P1 P2 : ℝ) (h : A1 / A2 = 16 / 9) : P1 / P2 = 4 / 3 :=
sorry

end ratio_of_perimeters_of_similar_triangles_l382_382285


namespace smallest_cycle_length_l382_382241

theorem smallest_cycle_length (n : ℕ) (h₁ : n ≥ 3) (edges : ℕ) (h₂ : edges > (1 / 2 : ℝ) * n * real.sqrt (n - 1)) : 
  (∃ (l : ℕ), l < 4 ∧ ∃ cycle : finset ℕ, cycle.card = l ∧ cycle ⊆ finset.range n ∧ ∃ x, ∀ i ∈ cycle, adjacency_list x i) ∨ 
  (n ≥ 4 ∧ (∃ (l : ℕ), l < 5 ∧ ∃ cycle : finset ℕ, cycle.card = l ∧ cycle ⊆ finset.range n ∧ ∃ x, ∀ i ∈ cycle, adjacency_list x i)) :=
sorry

end smallest_cycle_length_l382_382241


namespace solve_inequalities_l382_382895

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382895


namespace greatest_percentage_difference_is_january_l382_382418

theorem greatest_percentage_difference_is_january :
  let percentage_difference (D B : ℝ) : ℝ := 
  ((max D B - min D B) / min D B) * 100 in
  let january := percentage_difference 5 2 in
  let february := percentage_difference 6 4 in
  let march := percentage_difference 5 5 in
  let april := percentage_difference 4 6 in
  let may := percentage_difference 3 5 in
  max january (max february (max march (max april may))) = january :=
by
  sorry

end greatest_percentage_difference_is_january_l382_382418


namespace stock_percentage_value_l382_382133

def annual_dividend (yield price_per_share : ℝ) : ℝ :=
  yield * price_per_share

def percentage_value (dividend price_per_share : ℝ) : ℝ :=
  (dividend / price_per_share) * 100

theorem stock_percentage_value (yield : ℝ) (price_per_share : ℝ)
  (h_yield : yield = 0.10) (h_price : price_per_share = 80) :
  percentage_value (annual_dividend yield price_per_share) price_per_share = 10 :=
by
  -- proof steps would go here
  sorry

end stock_percentage_value_l382_382133


namespace _l382_382369

noncomputable def triangle_area_formula (a b c S : ℝ) : Prop :=
  S = (sqrt 3 / 4) * (a^2 + c^2 - b^2)

noncomputable def angle_B (a b c S : ℝ) (B : ℝ) : Prop :=
  triangle_area_formula a b c S → B = π / 3

noncomputable theorem find_B (a b c S : ℝ) (B : ℝ) : triangle_area_formula a b c S → angle_B a b c S B :=
  by sorry

noncomputable def max_value (a c : ℝ) : ℝ :=
  2 * sqrt 6

noncomputable def expression (a c : ℝ) : ℝ :=
  (sqrt 3 - 1) * a + 2 * c

noncomputable def maximum_condition (b : ℝ) : Prop :=
  b = sqrt 3

noncomputable theorem max_expr_value (a b c : ℝ) : maximum_condition b → expression a c ≤ max_value a c :=
  by sorry

end _l382_382369


namespace count_primes_between_71_and_95_l382_382648

theorem count_primes_between_71_and_95 : 
  let primes := [71, 73, 79, 83, 89, 97] in
  let filtered_primes := primes.filter (λ p, 71 < p ∧ p < 95) in
  filtered_primes.length = 5 := 
by 
  sorry

end count_primes_between_71_and_95_l382_382648


namespace minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l382_382323

theorem minimum_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

theorem minimum_value_x_add_2y_achieved (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  ∃ x y, 0 < x ∧ 0 < y ∧ 1/x + 9/y = 1 ∧ x + 2 * y = 19 + 6 * Real.sqrt 2 :=
sorry

end minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l382_382323


namespace sqrt_defined_iff_x_ge_1_l382_382066

theorem sqrt_defined_iff_x_ge_1 (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_defined_iff_x_ge_1_l382_382066


namespace distribute_books_l382_382494

open Finset

-- Definitions for conditions
def books := range 6  -- Representing 6 distinct books as {0, 1, 2, 3, 4, 5}
def individuals := range 3  -- Representing 3 individuals as {0, 1, 2}

-- The main theorem statement
theorem distribute_books : ∃ (f : books → individuals), 
  (∀ i : individuals, (↑(books.filter (λ b, f b = i)).card) = 2) ∧ 
  (∑ i, combinations 6 2 * combinations 4 2 * combinations 2 2 = 90) :=
begin
  sorry
end

end distribute_books_l382_382494


namespace inequality_always_holds_true_l382_382277

theorem inequality_always_holds_true (a b c : ℝ) (h₁ : a > b) (h₂ : b > c) :
  (a / (c^2 + 1)) > (b / (c^2 + 1)) :=
by
  sorry

end inequality_always_holds_true_l382_382277


namespace prove_expression_l382_382666

theorem prove_expression (a : ℝ) (h : a^2 + a - 1 = 0) : 2 * a^2 + 2 * a + 2008 = 2010 := by
  sorry

end prove_expression_l382_382666


namespace diff_of_squares_l382_382210

-- Define constants a and b
def a : ℕ := 65
def b : ℕ := 35

-- State the theorem to be proven using the difference of squares formula
theorem diff_of_squares : a^2 - b^2 = 3000 := by
  have h1 : a + b = 100 := by rfl -- Checking a + b
  have h2 : a - b = 30 := by rfl -- Checking a - b
  have h3 : (a+b)*(a-b) = 3000 := by
    rw [h1, h2] -- Substituting the values
    norm_num -- Simplifying the arithmetics
  exact h3 -- Providing the ultimate result

end diff_of_squares_l382_382210


namespace find_h_l382_382670

noncomputable def h_value (x h : ℝ) : ℝ :=
((10 * x + 2) / 4 - (3 * x - 6) / 18) * 3 - 2 * x

theorem find_h (h : ℝ) : 
  (∀ (x : ℝ), x = 0.3 → 
    (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + h) / 3) 
  → h = 4 := 
by
  intro h_eq
  have : (10 * 0.3 + 2) / 4 - (3 * 0.3 - 6) / 18 = (2 * 0.3 + h) / 3 := h_eq 0.3 rfl
  let h_value_at_03 := (10 * 0.3 + 2) / 4 - (3 * 0.3 - 6) / 18
  have h_simplified : h_value 0.3 h = 4 := by
    unfold h_value
    sorry
  exact h_simplified

end find_h_l382_382670


namespace present_population_l382_382063

theorem present_population (P : ℝ) (h : 1.04 * P = 1289.6) : P = 1240 :=
by
  sorry

end present_population_l382_382063


namespace solution_set_linear_inequalities_l382_382917

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l382_382917


namespace triangle_PQR_area_l382_382507

-- Definitions based on conditions
def circle_1_radius : ℝ := 3
def circle_2_radius : ℝ := 4

def are_tangent (c1_radius c2_radius : ℝ) : Prop :=
  -- hypothetical function to state that two circles are tangent
  sorry

def is_tangent (triangle_side : ℝ) (circle_radius : ℝ) : Prop :=
  -- hypothetical function to state that a side of a triangle is tangent to a circle
  sorry

def are_congruent (side1 side2 : ℝ) : Prop := 
  side1 = side2

-- Tangency of circles
axiom tangent_circles : are_tangent circle_1_radius circle_2_radius

-- Triangle sides’ tangency to circles
axiom side_PQ_tangent : ∀ (triangle_side : ℝ), 
  is_tangent triangle_side circle_1_radius ∧ is_tangent triangle_side circle_2_radius

axiom side_PR_tangent : ∀ (triangle_side : ℝ),
  is_tangent triangle_side circle_1_radius ∧ is_tangent triangle_side circle_2_radius

axiom sides_congruent : ∃ (PQ PR : ℝ), are_congruent PQ PR

-- Theorem stating the area calculation
theorem triangle_PQR_area (PQ PR : ℝ): 
  are_tangent circle_1_radius circle_2_radius →
  (∀ side, is_tangent side circle_1_radius ∧ is_tangent side circle_2_radius) →
  are_congruent PQ PR → ∃ (area : ℝ), area = 54 * real.sqrt 10 :=
by
  -- The proof would be inserted here
  intros
  sorry

end triangle_PQR_area_l382_382507


namespace total_trucks_l382_382391

theorem total_trucks {t : ℕ} (h1 : 2 * t + t = 300) : t = 100 := 
by sorry

end total_trucks_l382_382391


namespace cot15_add_tan45_eq_sqrt2_l382_382039

theorem cot15_add_tan45_eq_sqrt2 :
  (Real.cot (15 * Real.pi / 180) + Real.tan (45 * Real.pi / 180)) = Real.sqrt 2 :=
by
  sorry

end cot15_add_tan45_eq_sqrt2_l382_382039


namespace true_statements_l382_382794

def curve (t : ℝ) : Prop :=
  ∀ x y, (x^2 / (4 - t) + y^2 / (t - 1)) = 1

def statement1 (t : ℝ) : Prop :=
  1 < t ∧ t < 4

def statement2 (t : ℝ) : Prop :=
  t > 4 ∨ t < 1

def statement3 (t : ℝ) : Prop :=
  t ≠ 5 / 2

def statement4 (t : ℝ) : Prop :=
  1 < t ∧ t < 5 / 2

def statement5 (t : ℝ) : Prop :=
  t < 1 ∧ ∀ y, y = sqrt(1 - t)

theorem true_statements (t : ℝ) :
  (statement2 t ∧ statement4 t ∧ statement5 t) ↔
  ((curve t → statement2 t) ∧
  (curve t → statement4 t) ∧
  (curve t → statement5 t)) :=
sorry

end true_statements_l382_382794


namespace angle_measure_l382_382424

theorem angle_measure : ∃ x : ℝ, (180 - x = 4 * (90 - x)) → x = 60 :=
begin
  sorry
end

end angle_measure_l382_382424


namespace solution_set_inequalities_l382_382829

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l382_382829


namespace polygon_formed_by_lines_l382_382816

theorem polygon_formed_by_lines :
  (∀ p : Point, ∃ q : set Point, is_intersec_point q p (y = 2 * x + 3) (y = -4 * x + 3) ∧
    ∃ r : set Point, is_intersec_point r p (y = 2 * x + 3) (y = 1) ∧
    ∃ s : set Point, is_intersec_point s p (y = -4 * x + 3) (y = 1) ∧
    ¬ (equilateral_triangle q r s ∨ isosceles_triangle q r s ∨ right_triangle q r s ∨
       triangle_and_trapezoid q r s ∨ quadrilateral q r s)) :=
sorry

end polygon_formed_by_lines_l382_382816


namespace range_of_m_l382_382260

open Set Real

noncomputable def f (x m : ℝ) : ℝ := abs (x^2 - 4 * x + 9 - 2 * m) + 2 * m

theorem range_of_m
  (h1 : ∀ x ∈ Icc (0 : ℝ) 4, f x m ≤ 9) : m ≤ 7 / 2 :=
by
  sorry

end range_of_m_l382_382260


namespace solution_set_of_inequalities_l382_382976

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382976


namespace probability_objects_meeting_is_020_l382_382388

-- Define the conditions for the movement of objects A and B
noncomputable def probability_meeting_objects :
  ℝ :=
  let total_paths : ℕ := 2^12 in
  let meeting_points := [ (0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1) ] in
  let meeting_prob := 
    ∑ i in finset.range 6, (nat.choose 6 i) * (nat.choose 6 (i + 1)) in
  meeting_prob / total_paths

-- Prove that the probability is 99/512
theorem probability_objects_meeting_is_020 :
  probability_meeting_objects = 99/512 := by
    -- The proof involves calculating the combinatorial sums and 
    -- dividing by the total paths. (Proof omitted)
    sorry

end probability_objects_meeting_is_020_l382_382388


namespace solve_inequalities_l382_382893

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l382_382893


namespace diff_of_squares_l382_382211

-- Define constants a and b
def a : ℕ := 65
def b : ℕ := 35

-- State the theorem to be proven using the difference of squares formula
theorem diff_of_squares : a^2 - b^2 = 3000 := by
  have h1 : a + b = 100 := by rfl -- Checking a + b
  have h2 : a - b = 30 := by rfl -- Checking a - b
  have h3 : (a+b)*(a-b) = 3000 := by
    rw [h1, h2] -- Substituting the values
    norm_num -- Simplifying the arithmetics
  exact h3 -- Providing the ultimate result

end diff_of_squares_l382_382211


namespace solve_inequalities_l382_382903

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l382_382903


namespace nursery_school_students_l382_382393

theorem nursery_school_students :
  (exists S O: ℕ, (O = (S / 10)) ∧ (20 + O = 50) ∧ (S = 300)) :=
begin
  sorry,
end

end nursery_school_students_l382_382393


namespace number_of_students_B_l382_382083

-- Definitions based on the conditions
def avg_weight_A : ℝ := 50
def num_students_A : ℕ := 40
def total_weight_A : ℝ := num_students_A * avg_weight_A

def avg_weight_B : ℝ := 60

-- Given average weight of the whole class
def avg_weight_class : ℝ := 54.285714285714285

-- Converting the average weight of the class to fraction representation
def avg_weight_class_fraction : ℝ := 380 / 7

-- Number of students in section B
def num_students_B : ℕ := 30

-- Total number of students in class
def total_students_class : ℕ := num_students_A + num_students_B

-- Equate the total weight based on the average weight given for the class
def total_weight_B : ℝ := num_students_B * avg_weight_B
def total_weight_class : ℝ := total_weight_A + total_weight_B

-- Prove that num_students_B satisfies the given average weight of the class
theorem number_of_students_B :
  (num_students_B = 30) ∧ 
  (avg_weight_class_fraction * total_students_class = total_weight_class) :=
by
  sorry

end number_of_students_B_l382_382083


namespace trig_expression_simplifies_to_zero_l382_382585

-- Define the given condition
def cot_theta_eq_three (θ : ℝ) : Prop := Real.cot θ = 3

-- State the problem as a Lean theorem
theorem trig_expression_simplifies_to_zero (θ : ℝ) (h : cot_theta_eq_three θ) :
  (1 - Real.sin θ) / (Real.cos θ) - (Real.cos θ) / (1 + Real.sin θ) = 0 := 
sorry

end trig_expression_simplifies_to_zero_l382_382585


namespace range_of_m_l382_382273

namespace MathProof

def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - m * x + m - 1 = 0 }

theorem range_of_m (m : ℝ) (h : A ∪ (B m) = A) : m = 3 :=
  sorry

end MathProof

end range_of_m_l382_382273


namespace complex_number_quadrant_l382_382261

def i := Complex.I
def z := i * (1 + i)

theorem complex_number_quadrant 
  : z.re < 0 ∧ z.im > 0 := 
by
  sorry

end complex_number_quadrant_l382_382261


namespace number_of_nonempty_proper_subsets_of_set_a_b_l382_382070

theorem number_of_nonempty_proper_subsets_of_set_a_b (a b : Type) :
  ∃ A : set (Type), A = {a, b} ∧ {s | s ⊂ A ∧ s ≠ ∅}.card = 2 :=
sorry

end number_of_nonempty_proper_subsets_of_set_a_b_l382_382070


namespace inverse_fraction_coeff_l382_382804

noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

noncomputable def g_inv (x : ℝ) : ℝ := (4 * x + 2) / (3 - x)

theorem inverse_fraction_coeff : ∃ a b c d : ℝ, g_inv = λ x, (a * x + b) / (c * x + d) ∧ a / c = -4 :=
by
  use [4, 2, -1, 3]
  split
  { funext
    intro x
    calc
    g_inv x = (4 * x + 2) / (3 - x) : rfl
          ... = (4 * x + 2) / (3 + (-x)) : by rw [←sub_eq_add_neg]
          ... = (4 * x + 2) / ((-x) + 3) : by rw [add_comm (-x) 3]},
  { norm_num }

end inverse_fraction_coeff_l382_382804


namespace general_formula_a_n_sum_b_n_l382_382616

-- Given conditions and definitions
def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def a_n (n : ℕ) : ℕ :=
  2 * n - 1

def b_n (n : ℕ) : ℕ :=
  a_n n + 2^n

def s_n (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, b_n i

theorem general_formula_a_n :
  (arithmetic_seq a_n ∧ a_n 1 = 1 ∧ a_n 3 = 5) →
    ∀ n : ℕ, a_n n = 2 * n - 1 :=
by
  sorry

theorem sum_b_n :
  (arithmetic_seq a_n ∧ a_n 1 = 1 ∧ a_n 3 = 5) →
    ∀ n : ℕ, s_n n = n^2 + 2^(n + 1) - 2 :=
by
  sorry

end general_formula_a_n_sum_b_n_l382_382616


namespace inverse_ratio_l382_382807

def g (x : ℝ) := (3 * x - 2) / (x + 4)

noncomputable def g_inv (x : ℝ) := (4 * x + 2) / (3 - x)

theorem inverse_ratio (a b c d : ℝ) (h1 : g_inv x = (a * x + b) / (c * x + d)) : a / c = -4 :=
by {
    sorry
}

end inverse_ratio_l382_382807


namespace inequality_solution_l382_382523

noncomputable def quadratic_trinomial (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem inequality_solution (a b c k : ℝ) (hk : k > 0) (hΔ : b^2 - 4 * a * (c - k^2) ≥ 0) :
  {x : ℝ | (quadratic_trinomial a b c x) < k^2} = 
  {x : ℝ | x > ((-b - (√(b^2 - 4 * a * (c - k^2)))) / (2 * a)) ∧ x < ((-b + (√(b^2 - 4 * a * (c - k^2)))) / (2 * a))} := 
sorry

end inequality_solution_l382_382523


namespace geometric_fixed_point_l382_382364

open_locale classical

/-- Let D be an arbitrary point on the side BC of triangle ABC.
    Points E and F are on CA and BA respectively such that CD=CE and BD=BF.
    Lines BE and CF intersect at P. Prove that when point D varies along the line BC, 
    PD passes through a fixed point. -/
theorem geometric_fixed_point
  (A B C D E F P : Type*)
  (h1 : D ∈ BC)
  (h2 : E ∈ CA)
  (h3 : F ∈ BA)
  (h4 : CD = CE)
  (h5 : BD = BF)
  (h6 : ∃ P, BE ∩ CF = {P}) :
  ∃ A', ∀ D, D ∈ BC → ∃ P, (BE ∩ CF) = {P} → PD ∋ A' :=
begin
  -- Proof skipped
  sorry
end

end geometric_fixed_point_l382_382364


namespace area_of_triangle_PF1F2_l382_382731

theorem area_of_triangle_PF1F2 :
  let F1 := (-2, 0)
  let F2 := (2, 0)
  let hyperbola (x y: ℝ) : Prop := x^2 - y^2 / 3 = 1
  let point_P (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2
  P : ℝ × ℝ
  (hP : point_P P)
  (h_dist : 3 * dist P F1 = 4 * dist P F2) :
  let cos_angle (a b c: ℝ) : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)
  let sin_angle (cosθ: ℝ) : ℝ := real.sqrt (1 - cosθ^2)
  let area (a b sinθ: ℝ) : ℝ := 1/2 * a * b * sinθ
  F1 := (-2,0)
  F2 := (2,0)
  area (dist P F1) (dist P F2) (sin_angle (cos_angle 8 6 4)) = 3 * real.sqrt 15 :=
sorry

end area_of_triangle_PF1F2_l382_382731


namespace parallel_lines_a_eq_one_l382_382633

theorem parallel_lines_a_eq_one 
  (a : ℝ)
  (h : (∀ x y : ℝ, ax - y - 2 = 0 ↔ (2 - a)x - y + 1 = 0)) :
  a = 1 := 
sorry

end parallel_lines_a_eq_one_l382_382633


namespace books_left_over_l382_382347

-- Define the conditions as variables in Lean
def total_books : ℕ := 1500
def new_shelf_capacity : ℕ := 28

-- State the theorem based on these conditions
theorem books_left_over : total_books % new_shelf_capacity = 14 :=
by
  sorry

end books_left_over_l382_382347


namespace range_of_x_max_value_of_g_l382_382624

def f (x : ℝ) : ℝ := abs (x - 2) - 3
def g (x : ℝ) : ℝ := 3 * sqrt (x + 4) + 4 * sqrt (abs (x - 6))

theorem range_of_x (x : ℝ) : f x < 0 ↔ -1 < x ∧ x < 5 := 
by sorry

theorem max_value_of_g (x : ℝ) (h : -1 < x ∧ x < 5) : 
  g x ≤ 5 * sqrt 10 ∧ (x = -2 / 5 → g x = 5 * sqrt 10) := 
by sorry

end range_of_x_max_value_of_g_l382_382624


namespace line_intersects_semicircle_at_two_points_l382_382622

theorem line_intersects_semicircle_at_two_points
  (m : ℝ) :
  (3 ≤ m ∧ m < 3 * Real.sqrt 2) ↔ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ (y₁ = -x₁ + m ∧ y₁ = Real.sqrt (9 - x₁^2)) ∧ (y₂ = -x₂ + m ∧ y₂ = Real.sqrt (9 - x₂^2))) :=
by
  -- The proof goes here
  sorry

end line_intersects_semicircle_at_two_points_l382_382622


namespace solution_set_of_inequalities_l382_382962

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l382_382962


namespace linear_inequalities_solution_l382_382881

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l382_382881


namespace continuity_at_one_l382_382332

noncomputable def f (a b : ℝ) : ℝ → ℝ 
| x := if h : x = 1 then 1 else (a / (1 - x)) - (b / (1 - x^2))

theorem continuity_at_one (a b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |f a b x - 1| < ε) ↔ 
  (a = -2 ∧ b = -4) :=
begin
  sorry
end

end continuity_at_one_l382_382332


namespace sum_of_intersection_points_l382_382062

theorem sum_of_intersection_points :
  let p1 := λ x, (x - 2)^2
  let p2 := λ y, (y - 5)^2 - 6
  ∀ (points : List (ℝ × ℝ)), 
    (∀ (x y : ℝ), (y = p1 x ∧ x + 6 = p2 y) → (x, y) ∈ points) →
    points.length = 4 →
    (points.map Prod.fst).sum + (points.map Prod.snd).sum = 10 :=
sorry

end sum_of_intersection_points_l382_382062


namespace direct_proportion_only_neg_x_l382_382104

theorem direct_proportion_only_neg_x :
  ∀ (f : ℝ → ℝ),
  (f = (λ x, x ^ 2) ∨ f = (λ x, -x) ∨ f = (λ x, x + 1) ∨ f = (λ x, 1 / x)) →
  ∃ k : ℝ, f = (λ x, k * x) ↔ f = (λ x, -x) := by
  intro f
  intro h
  existsi -1
  constructor
  intro h1
  sorry
  intro h2
  sorry

end direct_proportion_only_neg_x_l382_382104


namespace pyramid_area_l382_382467

theorem pyramid_area
  (base_length : ℝ)
  (slant_height : ℝ)
  (base_length_eq : base_length = 8)
  (slant_height_eq : slant_height = 10) :
  4 * (1 / 2 * base_length * (real.sqrt (slant_height^2 - (base_length / 2)^2))) = 32 * real.sqrt 21 :=
by
  -- Proof will go here
  sorry

end pyramid_area_l382_382467


namespace problem_1_problem_2_l382_382001

-- Define f as an odd function on ℝ 
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the main property given in the problem
def property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, (a + b) ≠ 0 → (f a + f b) / (a + b) > 0

-- Problem 1: Prove that if a > b then f(a) > f(b)
theorem problem_1 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  ∀ a b : ℝ, a > b → f a > f b := sorry

-- Problem 2: Prove that given f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x in [0, +∞), the range of k is k < 1
theorem problem_2 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  (∀ x : ℝ, 0 ≤ x → f (9 ^ x - 2 * 3 ^ x) + f (2 * 9 ^ x - k) > 0) → k < 1 := sorry

end problem_1_problem_2_l382_382001


namespace lunchroom_tables_l382_382413

/-- Given the total number of students and the number of students per table, 
    prove the number of tables in the lunchroom. -/
theorem lunchroom_tables (total_students : ℕ) (students_per_table : ℕ) 
  (h_total : total_students = 204) (h_per_table : students_per_table = 6) : 
  total_students / students_per_table = 34 := 
by
  sorry

end lunchroom_tables_l382_382413


namespace trig_identity_proof_l382_382120

variable (α β γ : ℝ)

theorem trig_identity_proof :
  3.405 * (sin α + sin β + sin γ - sin (α + β + γ)) / 
  (cos α + cos β + cos γ + cos (α + β + γ)) = 
  tan ((α + β) / 2) * tan ((β + γ) / 2) * tan ((γ + α) / 2) :=
sorry

end trig_identity_proof_l382_382120


namespace area_of_figure_between_curves_l382_382204

noncomputable def area_between_curves :=
  let r1 (φ : ℝ) := (5 / 2) * Real.sin φ
  let r2 (φ : ℝ) := (3 / 2) * Real.sin φ
  (1 / 2) * ∫ φ in -Real.pi / 2..Real.pi / 2, (r1 φ)^2 - (r2 φ)^2

theorem area_of_figure_between_curves : area_between_curves = Real.pi :=
  by
    sorry

end area_of_figure_between_curves_l382_382204


namespace not_p_equiv_exists_leq_sin_l382_382613

-- Define the conditions as a Lean proposition
def p : Prop := ∀ x : ℝ, x > Real.sin x

-- State the problem as a theorem to be proved
theorem not_p_equiv_exists_leq_sin : ¬p = ∃ x : ℝ, x ≤ Real.sin x := 
by sorry

end not_p_equiv_exists_leq_sin_l382_382613


namespace sum_ratios_l382_382706

-- Define the conditions in Lean
variables (A B C D E F : Type) [geometry A B C D E F]

-- D is the midpoint of BC, and E divides AB in the ratio 1:3
axiom D_midpoint_BC : midpoint D B C
axiom E_divides_AB_1_3 : divides E A B (1/3)

-- F is the intersection of the lines AD and CE
axiom F_intersection_AD_CE : intersection F (line_through A D) (line_through C E)

theorem sum_ratios {A B C D E F : Type}
  [geometry A B C D E F]
  (D_midpoint_BC : midpoint D B C)
  (E_divides_AB_1_3 : divides E A B (1/3))
  (F_intersection_AD_CE : intersection F (line_through A D) (line_through C E)) :
  (EF / FC + AF / FD) = 11 / 6 :=
sorry

end sum_ratios_l382_382706


namespace volume_of_inscribed_cube_l382_382137

theorem volume_of_inscribed_cube (R : ℝ) : 
  ∃ x : ℝ, 
    (x = R * Real.sqrt (2 / 3) ∧ 
     (volume : ℝ) = x^3 ∧ 
     volume = 2 * R^3 * Real.sqrt(6) / 9) :=
begin
  let x := R * Real.sqrt (2 / 3),
  let volume := x^3,
  use x,
  split,
  { refl, },
  split,
  { refl, },
  { refl, }
end

end volume_of_inscribed_cube_l382_382137


namespace parallelogram_area_l382_382480

-- Define the given constants for base and height
def base : ℝ := 26
def height : ℝ := 14

-- Statement to prove that the area of the parallelogram is 364 cm²
theorem parallelogram_area :
  (base * height) = 364 := 
by 
  -- The proof is omitted for now 
  sorry

end parallelogram_area_l382_382480


namespace window_treatments_cost_l382_382363

def cost_of_sheers (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def cost_of_drapes (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def total_cost (n : ℕ) (cost_sheers : ℝ) (cost_drapes : ℝ) : ℝ :=
  cost_of_sheers n cost_sheers + cost_of_drapes n cost_drapes

theorem window_treatments_cost :
  total_cost 3 40 60 = 300 :=
by
  sorry

end window_treatments_cost_l382_382363


namespace avg_height_l382_382345

theorem avg_height (num_students : ℕ := 100) (a b c d e : ℕ := 5) (f : ℕ := 40) (g : ℕ := 40) (h : ℕ := 10) :
  let total_students := a + f + g + h + e in
  let height_range1 := (145 + 155) / 2 in
  let height_range2 := (155 + 165) / 2 in
  let height_range3 := (165 + 175) / 2 in
  let height_range4 := (175 + 185) / 2 in
  let height_range5 := (185 + 195) / 2 in
  let average_height := (height_range1 * a + height_range2 * f + height_range3 * g + height_range4 * h + height_range5 * e) / total_students in
  average_height = 167 :=
by
  let total_students := a + f + g + h + e
  let height_range1 := (145 + 155) / 2
  let height_range2 := (155 + 165) / 2
  let height_range3 := (165 + 175) / 2
  let height_range4 := (175 + 185) / 2
  let height_range5 := (185 + 195) / 2
  let average_height := (height_range1 * a + height_range2 * f + height_range3 * g + height_range4 * h + height_range5 * e) / total_students
  have h1 : total_students = 100 := by norm_num
  have h2 : height_range1 = 150 := by norm_num
  have h3 : height_range2 = 160 := by norm_num
  have h4 : height_range3 = 170 := by norm_num
  have h5 : height_range4 = 180 := by norm_num
  have h6 : height_range5 = 190 := by norm_num
  have h7 : average_height = (150 * 5 + 160 * 40 + 170 * 40 + 180 * 10 + 190 * 5) / 100 := by norm_num
  have h8 : (150 * 5 + 160 * 40 + 170 * 40 + 180 * 10 + 190 * 5) / 100 = 167 := by norm_num
  exact h8
  sorry

end avg_height_l382_382345


namespace Mikaela_savings_l382_382016

theorem Mikaela_savings
  (hourly_rate : ℕ)
  (first_month_hours : ℕ)
  (additional_hours_second_month : ℕ)
  (spending_fraction : ℚ)
  (earnings_first_month := hourly_rate * first_month_hours)
  (hours_second_month := first_month_hours + additional_hours_second_month)
  (earnings_second_month := hourly_rate * hours_second_month)
  (total_earnings := earnings_first_month + earnings_second_month)
  (amount_spent := spending_fraction * total_earnings)
  (amount_saved := total_earnings - amount_spent) :
  hourly_rate = 10 →
  first_month_hours = 35 →
  additional_hours_second_month = 5 →
  spending_fraction = 4 / 5 →
  amount_saved = 150 :=
by
  sorry

end Mikaela_savings_l382_382016


namespace q1_even_q1_odd_q2_inequality_q3_unbounded_l382_382698

variable (m n : ℕ)
def S1 : ℝ := sorry  -- Placeholder for total area of black parts
def S2 : ℝ := sorry  -- Placeholder for total area of white parts

def f (m n : ℕ) : ℝ := abs (S1 - S2)

theorem q1_even (hm : even m) (hn : even n) : f m n = 0 := 
sorry

theorem q1_odd (hm : odd m) (hn : odd n) : f m n = 1/2 := 
sorry

theorem q2_inequality (m n : ℕ) : f m n ≤ 1/2 * max m n := 
sorry

theorem q3_unbounded (c : ℝ) : ¬ ∀ m n : ℕ, f m n < c :=
sorry

end q1_even_q1_odd_q2_inequality_q3_unbounded_l382_382698


namespace percentage_profit_is_correct_l382_382166

-- Define the costs and selling price
def cost_tv : ℤ := 16000
def cost_dvd : ℤ := 6250
def selling_price : ℤ := 35600

-- Define the total cost price and profit
def total_cost_price : ℤ := cost_tv + cost_dvd
def profit : ℤ := selling_price - total_cost_price

-- Define the percentage profit calculation
def percentage_profit : ℚ := (profit.toRat / total_cost_price.toRat) * 100

-- State the theorem
theorem percentage_profit_is_correct : percentage_profit = 60 := sorry

end percentage_profit_is_correct_l382_382166


namespace prime_square_count_l382_382641

-- Define the set of natural numbers whose squares lie between 5000 and 9000
def within_square_range (n : ℕ) : Prop := 5000 < n * n ∧ n * n < 9000

-- Define the predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the set of prime numbers whose squares lie within the range
def primes_within_range : Finset ℕ := (Finset.filter is_prime (Finset.Ico 71 95))

-- The main statement: the number of prime numbers whose squares are between 5000 and 9000
theorem prime_square_count : (primes_within_range.filter within_square_range).card = 6 :=
sorry

end prime_square_count_l382_382641


namespace right_triangle_area_l382_382245

theorem right_triangle_area (c a : ℝ) (h1 : c = 5) (h2 : a = 3) : 
  (∃ b : ℝ, b^2 = c^2 - a^2) → 
  (∃ b : ℝ, b = real.sqrt (c^2 - a^2)) → 
  (∃ area : ℝ, area = (1/2) * a * (real.sqrt (c^2 - a^2))) → 
  area = 6 :=
by
  intros
  sorry

end right_triangle_area_l382_382245


namespace number_of_authors_l382_382710

/-- Define the number of books each author has and the total number of books. -/
def books_per_author : ℕ := 33
def total_books : ℕ := 198

/-- Main theorem stating that the number of authors Jack has is derived by dividing total books by the number of books per author. -/
theorem number_of_authors (n : ℕ) (h : total_books = n * books_per_author) : n = 6 := by
  sorry

end number_of_authors_l382_382710


namespace trigonometric_expression_zero_l382_382588

theorem trigonometric_expression_zero {θ : ℝ} (h : Real.cot θ = 3) :
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 :=
sorry

end trigonometric_expression_zero_l382_382588


namespace fraction_of_female_participants_is_correct_l382_382436

-- defining conditions
def last_year_males : ℕ := 30
def male_increase_rate : ℚ := 1.1
def female_increase_rate : ℚ := 1.25
def overall_increase_rate : ℚ := 1.2

-- the statement to prove
theorem fraction_of_female_participants_is_correct :
  ∀ (y : ℕ), 
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  total_this_year = males_this_year + females_this_year →
  (females_this_year / total_this_year) = (25 / 36) :=
by
  intros y
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  intro h
  sorry

end fraction_of_female_participants_is_correct_l382_382436


namespace sqrt_expression_is_integer_l382_382355

theorem sqrt_expression_is_integer 
  (n : ℕ) 
  (h_pos : 0 < n)
  (h_sqrt_int : ∃ (x : ℕ), x * x = 12 * n * n + 1) :
  ∃ (q : ℕ), q * q = (nat.sqrt (12 * n * n + 1) + 1) / 2 :=
sorry

end sqrt_expression_is_integer_l382_382355


namespace divisible_by_five_l382_382037

theorem divisible_by_five (a : ℤ) (h : ¬ (5 ∣ a)) :
    (5 ∣ (a^2 - 1)) ↔ ¬ (5 ∣ (a^2 + 1)) :=
by
  -- Begin the proof here (proof not required according to instructions)
  sorry

end divisible_by_five_l382_382037


namespace triangle_area_and_min_ac_l382_382266

-- Given conditions
variables {A B C a b c : ℝ}

-- Assumptions based on the conditions
axiom cos_eq: (2 * a - c) * Real.cos B = b * Real.cos C
axiom dot_product_eq: ⟦A, B⟧ ⋅ ⟦B, C⟧ = -3

-- The proof problem to be stated
theorem triangle_area_and_min_ac (h_cos_eq : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h_dot_product_eq : ⟦A, B⟧ ⋅ ⟦B, C⟧ = -3) :
  let area := (1 / 2) * a * c * Real.sin (π / 3) in
  area = 3 * Real.sqrt 3 / 2 ∧
  b >= Real.sqrt 6 :=
sorry

end triangle_area_and_min_ac_l382_382266


namespace joe_months_of_play_l382_382359

theorem joe_months_of_play (spending_per_month revenue_per_month initial_amount : ℕ) :
  spending_per_month = 50 →
  revenue_per_month = 30 →
  initial_amount = 240 →
  initial_amount / (spending_per_month - revenue_per_month) = 12 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end joe_months_of_play_l382_382359


namespace solution_set_of_inequalities_l382_382979

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l382_382979


namespace polygon_sides_and_internal_angle_l382_382171

theorem polygon_sides_and_internal_angle
  (perimeter : ℝ) (side_length : ℝ) (claimed_angle : ℝ)
  (h_perimeter : perimeter = 150) (h_side_length : side_length = 10) (h_claimed_angle : claimed_angle = 140) :
  let n := (perimeter / side_length) in
  n = 15 ∧ ((n - 2) * 180 / n) = 156 :=
by
  sorry

end polygon_sides_and_internal_angle_l382_382171
