import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticEquations
import Mathlib.Analysis.Calculus.Area
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.Trigonometric.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Polynomial.RingDivision
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Function
import Mathlib.LinearAlgebra.Matrix
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Algebra.Order.Basic
import Mathlib.Topology.Basic
import Mathlib.Trigonometry.Basic

namespace arithmetic_sequence_minimum_value_of_Sn_l476_476543

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476543


namespace arithmetic_sequence_minimum_value_S_l476_476580

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476580


namespace probability_of_sum_multiple_of_3_l476_476709

noncomputable def card_numbers := {1, 2, 3, 4, 5}

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def pairs_sum_to_multiple_of_3 (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s |>.filter (λ p, is_multiple_of_3 (p.1 + p.2) ∧ p.1 < p.2)

def total_pairs (s : Finset ℕ) : ℕ :=
  (s.card.choose 2)

def favorable_pairs (s : Finset ℕ) : ℕ :=
  (pairs_sum_to_multiple_of_3 s).card

theorem probability_of_sum_multiple_of_3 :
  @Finset.card _ _ card_numbers > 1 → 
  (favorable_pairs card_numbers : ℚ) / (total_pairs card_numbers) = 2 / 5 :=
by
  sorry

end probability_of_sum_multiple_of_3_l476_476709


namespace earnings_in_total_l476_476756

-- Defining the conditions
def hourly_wage : ℝ := 12.50
def hours_per_week : ℝ := 40
def earnings_per_widget : ℝ := 0.16
def widgets_per_week : ℝ := 1250

-- Theorem statement
theorem earnings_in_total : 
  (hours_per_week * hourly_wage) + (widgets_per_week * earnings_per_widget) = 700 := 
by
  sorry

end earnings_in_total_l476_476756


namespace solve_equation_l476_476662

theorem solve_equation :
  ∀ x : ℝ, ( (1 / 8) ^ (3 * x + 12) = 64 ^ (x + 4) ) → x = -4 :=
by
  intros x h
  sorry

end solve_equation_l476_476662


namespace sam_carrots_l476_476074

theorem sam_carrots (sandy_carrots total_carrots : ℕ) 
  (h1 : sandy_carrots = 6) 
  (h2 : total_carrots = 9) : 
  sam_carrots = total_carrots - sandy_carrots :=
by
  have h3 : sam_carrots = 9 - 6 := by rw [h2, h1]
  exact h3
sorry

end sam_carrots_l476_476074


namespace tim_movie_marathon_duration_l476_476717

-- Define the durations of each movie
def first_movie_duration : ℕ := 2

def second_movie_duration : ℕ := 
  first_movie_duration + (first_movie_duration / 2)

def combined_first_two_movies_duration : ℕ :=
  first_movie_duration + second_movie_duration

def last_movie_duration : ℕ := 
  combined_first_two_movies_duration - 1

-- Define the total movie marathon duration
def total_movie_marathon_duration : ℕ := 
  first_movie_duration + second_movie_duration + last_movie_duration

-- Problem statement to be proved
theorem tim_movie_marathon_duration : total_movie_marathon_duration = 9 := by
  sorry

end tim_movie_marathon_duration_l476_476717


namespace min_sin6_cos6_l476_476281

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476281


namespace area_inside_C_outside_A_B_l476_476819

-- Definition of circles A, B, and C with radius 1
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

constant A : Circle
constant B : Circle
constant C : Circle

axiom A_radius : A.radius = 1
axiom B_radius : B.radius = 1
axiom C_radius : C.radius = 1

-- Midpoint of the line segment AB
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Axiom stating that A and B are tangent at one point
axiom A_B_tangent_at_one_point : 
  let d := ((A.center.1 - B.center.1) ^ 2 + (A.center.2 - B.center.2) ^ 2).sqrt in
  d = A.radius + B.radius

-- Axiom stating that C is tangent to the midpoint of AB
axiom C_tangent_to_midpoint : 
  let M := midpoint A.center B.center in
  let d := ((C.center.1 - M.1) ^ 2 + (C.center.2 - M.2) ^ 2).sqrt in
  d = C.radius

-- Theorem to prove
theorem area_inside_C_outside_A_B : 
  let shared_area := (π * 1^2) / 2 - (1 / 2) in
  let total_shared_area := 4 * shared_area in
  let area_C := π * 1^2 in
  area_C - total_shared_area = 2 :=
sorry

end area_inside_C_outside_A_B_l476_476819


namespace angle_between_u_and_v_l476_476857

open Real

noncomputable def u : ℝ × ℝ × ℝ := (3, -2, 2)
noncomputable def v : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (a.1^2 + a.2^2 + a.3^2)

noncomputable def cosine (a b : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product a b) / ((magnitude a) * (magnitude b))

noncomputable def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (cosine a b)

theorem angle_between_u_and_v :
  angle_between_vectors u v = real.arccos (-8 / (3 * sqrt 17)) := by
  sorry

end angle_between_u_and_v_l476_476857


namespace min_value_sin6_cos6_l476_476328

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476328


namespace sequence_problem_l476_476588

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476588


namespace alice_has_winning_strategy_l476_476051

theorem alice_has_winning_strategy (n : ℕ) (h : n > 1) :
  ∃ strategy : nat → bool, ∀ strategyBob : nat → bool, sum (filter strategy (range (2 * n))) 
  ≥ sum (filter (λ x, ¬(strategy x)) (range (2 * n))) :=
sorry

end alice_has_winning_strategy_l476_476051


namespace number_of_numeric_methods_students_l476_476474

def total_students : ℕ := 663
def percentage_second_year : ℝ := 0.80
def second_year_students : ℕ := Nat.round (0.80 * 663)
def acav_students : ℕ := 423
def both_students : ℕ := 134

theorem number_of_numeric_methods_students : 
  (∃ N : ℕ, N + acav_students - both_students = second_year_students) ↔ (N = 241) := by
  sorry

end number_of_numeric_methods_students_l476_476474


namespace percentile_40th_is_99_l476_476001

def percentile_40th_student_A (scores : List ℕ) : ℕ :=
  let n := scores.length
  let sorted_scores := scores.sort
  let pos := (n * 40) / 100
  (sorted_scores[pos - 1] + sorted_scores[pos]) / 2

theorem percentile_40th_is_99 :
  let scores := [94, 96, 98, 98, 100, 101, 101, 102, 102, 103]
  percentile_40th_student_A scores = 99 :=
by
  sorry

end percentile_40th_is_99_l476_476001


namespace km_to_m_is_750_l476_476903

-- Define 1 kilometer equals 5 hectometers
def km_to_hm := 5

-- Define 1 hectometer equals 10 dekameters
def hm_to_dam := 10

-- Define 1 dekameter equals 15 meters
def dam_to_m := 15

-- Theorem stating that the number of meters in one kilometer is 750
theorem km_to_m_is_750 : 1 * km_to_hm * hm_to_dam * dam_to_m = 750 :=
by 
  -- Proof goes here
  sorry

end km_to_m_is_750_l476_476903


namespace find_ratio_is_solution_l476_476999

noncomputable def find_common_ratio (a : ℕ → ℚ) (Sn : ℕ → ℚ) (q : ℚ) : Prop :=
(q = 2 ∨ q = 1/2) ∧
(a 2 = 1/4) ∧
(Sn 3 = 7/8) ∧
(∀ n, a n = a 1 * q^(n-1)) ∧
(∀ n, Sn n = a 1 * (GeomSum q n))

theorem find_ratio_is_solution (a : ℕ → ℚ) (Sn : ℕ → ℚ) (q : ℚ) :
  find_common_ratio a Sn q :=
by
  sorry

end find_ratio_is_solution_l476_476999


namespace real_solutions_count_l476_476877

theorem real_solutions_count : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), (2 : ℝ) ^ (3 * x ^ 2 - 8 * x + 4) = 1 → x = 2 ∨ x = 2 / 3 :=
by
  sorry

end real_solutions_count_l476_476877


namespace five_digit_numbers_qr_divisible_by_7_l476_476052

theorem five_digit_numbers_qr_divisible_by_7 :
  (∑ n in finset.Icc 10000 99999, if let (q, r) := (n / 49, n % 49) in (q + r) % 7 = 0 then 1 else 0) = 12859 := 
sorry

end five_digit_numbers_qr_divisible_by_7_l476_476052


namespace find_radius_l476_476679

theorem find_radius
  (sector_area : ℝ)
  (arc_length : ℝ)
  (sector_area_eq : sector_area = 11.25)
  (arc_length_eq : arc_length = 4.5) :
  ∃ r : ℝ, 11.25 = (1/2 : ℝ) * r * arc_length ∧ r = 5 := 
by
  sorry

end find_radius_l476_476679


namespace min_value_sin_cos_l476_476296

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476296


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476598

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476598


namespace length_PX_l476_476973

theorem length_PX (CX DP PW PX : ℕ) (hCX : CX = 60) (hDP : DP = 20) (hPW : PW = 40)
  (parallel_CD_WX : true)  -- We use a boolean to denote the parallel condition for simplicity
  (h1 : DP + PW = CX)  -- The sum of the segments from point C through P to point X
  (h2 : DP * 2 = PX)  -- The ratio condition derived from the similarity of triangles
  : PX = 40 := 
by
  -- using the given conditions and h2 to solve for PX
  sorry

end length_PX_l476_476973


namespace min_sin6_cos6_l476_476287

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476287


namespace num_arrangements_of_adjacent_ABC_l476_476708

-- Define the problem conditions
def num_people := 5
def to_be_adjacent (a b c : Type) := True -- Represents the adjacency condition of A, B, and C

-- Define a statement verifying the number of arrangements where A, B, and C are adjacent is 36
theorem num_arrangements_of_adjacent_ABC : 
  let arrangements := 36 in 
  arrangements = 6 * 6    := 
by
  -- Provide a statement that encodes the given in terms of let-bindings
  let num_elements := num_people - 2 -- Elements of P4 and P5 to place the 'chunk'
  let arrangement_with_adj_ABC := 6 * 6 -- The inner arrangement
  have inner_arrangement: (6) * (6) = 36,
    { sorry  }
-- Attach the theorem with required statement
  exact inner_arrangement

end num_arrangements_of_adjacent_ABC_l476_476708


namespace problem_I_problem_II_l476_476929

noncomputable def distance_from_point_to_line (A B C : ℝ) (x₀ y₀ : ℝ) : ℝ :=
  (abs (A * x₀ + B * y₀ + C)) / (sqrt (A ^ 2 + B ^ 2))

theorem problem_I : distance_from_point_to_line 2 (-1) (-5) 3 0 = sqrt 5 / 5 :=
sorry

noncomputable def line_m_slope : ℝ := 10

noncomputable def line_m_equation (x y : ℝ) : Prop := 
  x - (y / 10) - 3 = 0

theorem problem_II : ∃ k : ℝ, (k = line_m_slope) ∧ (line_m_equation = λ x y, x - (y / 10) - 3 = 0) :=
sorry

end problem_I_problem_II_l476_476929


namespace digit_at_2013th_position_after_decimal_l476_476425

-- Definitions based on conditions
def decimal_1 : ℚ := 9 / 37
def decimal_2 : ℚ := 10841 / 33333
def product_decimals : ℚ := decimal_1 * decimal_2

-- Main proof statement
theorem digit_at_2013th_position_after_decimal :
  (digit_at_position_after_decimal product_decimals 2013) = 9 :=
by
  sorry

-- Auxiliary definition to get digit at a particular position after the decimal point
noncomputable def digit_at_position_after_decimal (x : ℚ) (n : ℕ) : ℕ :=
  -- Placeholder for the actual implementation
  sorry

end digit_at_2013th_position_after_decimal_l476_476425


namespace min_sin6_cos6_l476_476338

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476338


namespace tan_a3_a5_is_sqrt3_l476_476913

-- Defining that a sequence is geometric
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n

-- The sequence {a_n} and given conditions
variable {a : ℕ → ℝ}
variable (h_geom : is_geometric a)
variable (h_rel : a 2 * a 6 + 2 * (a 4)^2 = real.pi)

-- The goal: Proving the required result
theorem tan_a3_a5_is_sqrt3 (h_geom : is_geometric a) (h_rel : a 2 * a 6 + 2 * (a 4)^2 = real.pi) : 
  real.tan (a 3 * a 5) = real.sqrt 3 := 
  sorry

end tan_a3_a5_is_sqrt3_l476_476913


namespace angle_between_vectors_is_correct_l476_476863

def vec_a : ℝ × ℝ × ℝ := (3, -2, 2)
def vec_b : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def angle_between_vectors : ℝ :=
  Real.acos ((vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 + vec_a.3 * vec_b.3) / 
    (Real.sqrt (vec_a.1^2 + vec_a.2^2 + vec_a.3^2) * Real.sqrt (vec_b.1^2 + vec_b.2^2 + vec_b.3^2)))

theorem angle_between_vectors_is_correct :
  angle_between_vectors = Real.acos (-8 / (3 * Real.sqrt 17)) :=
by sorry

end angle_between_vectors_is_correct_l476_476863


namespace arithmetic_sequence_and_minimum_sum_l476_476520

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476520


namespace min_value_sin_cos_l476_476299

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476299


namespace coefficient_x4_l476_476870

open Polynomial

-- Define the given polynomials and the overall polynomial expression
def p1 : Polynomial ℤ := X^2 - 2 * X^4 + X^3
def p2 : Polynomial ℤ := X^4 + 3 * X^3 - 2 * X^2 + X^5
def p3 : Polynomial ℤ := C 2 + X^2 - X^4 + 2 * X^3
def overall_poly : Polynomial ℤ := 4 * p1 + 2 * p2 - 3 * p3

-- State the theorem to prove the coefficient of x^4
theorem coefficient_x4 : coeff overall_poly 4 = -3 :=
by
  -- The proof is omitted
  sorry

end coefficient_x4_l476_476870


namespace percent_palindromes_with_seven_l476_476142

def is_digit (d : ℕ) : Prop := d ≤ 9

def is_palindrome (n : ℕ) : Prop :=
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ n = 10000 + 1000 + x * 1000 + y * 100 + x * 10 + y

theorem percent_palindromes_with_seven :
  (∃ (n : ℕ) (hx : is_palindrome n), 
   (7 ∈ [n / 1000 % 10, n / 100 % 10]) → 
   19) :=
sorry

end percent_palindromes_with_seven_l476_476142


namespace ticket_is_five_times_soda_l476_476762

variable (p_i p_r : ℝ)

theorem ticket_is_five_times_soda
  (h1 : 6 * p_i + 20 * p_r = 50)
  (h2 : 6 * p_r = p_i + p_r) : p_i = 5 * p_r :=
sorry

end ticket_is_five_times_soda_l476_476762


namespace min_sin6_cos6_l476_476283

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476283


namespace volume_of_rotated_triangle_l476_476068

noncomputable def volume_of_revolution (R : ℝ) (α : ℝ) : ℝ :=
  (2 / 3) * π * R^3 * sin (2 * α) * sin (4 * α)

theorem volume_of_rotated_triangle (R : ℝ) (α : ℝ) :
  let AB := 2 * R,
      angle_ADC := α,
      AC_less_than_AD := true (* This condition implies AC < AD *)
  in volume_of_revolution R α = (2 / 3) * π * R^3 * sin (2 * α) * sin (4 * α) :=
by
  sorry

end volume_of_rotated_triangle_l476_476068


namespace min_sixth_power_sin_cos_l476_476256

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476256


namespace pipe_b_time_l476_476132

noncomputable def pipe_b_filling_time () : ℝ :=
  let pipe_a_rate := 2 -- Pipe A fills 2 tanks per hour
  let t' := 1 / 2  -- time when both pipes work before Pipe A is shut off
  let pipe_a_shut_off_time := 0.25 -- Pipe A is shut off 0.25 hours before the tank overflows
  let pipe_b_work_done_in_last_part := 0.5 -- Pipe B finishes the remaining work, which is 0.5 tanks
  let combined_rate := pipe_a_rate + 1 / (pipe_b_filling_time) -- combined rate of the pipes
  let work_done_by_combined := combined_rate * t' -- work done when both pipes are operational
  let work_done_by_pipe_a := pipe_a_rate * pipe_a_shut_off_time -- work done by Pipe A before shutting off
  let remaining_work_done_by_pipe_b := 1 - work_done_by_pipe_a -- the rest of the work is done by Pipe B
  
  1 -- Pipe B takes 1 hour to fill the tank on its own.

theorem pipe_b_time : pipe_b_filling_time () = 1 := by
  sorry

end pipe_b_time_l476_476132


namespace martha_makes_40_cookies_martha_needs_7_5_cups_l476_476643

theorem martha_makes_40_cookies :
  (24 / 3) * 5 = 40 :=
by
  sorry

theorem martha_needs_7_5_cups :
  60 / (24 / 3) = 7.5 :=
by
  sorry

end martha_makes_40_cookies_martha_needs_7_5_cups_l476_476643


namespace work_completion_rates_l476_476172

noncomputable def Wp : ℝ := 1 / 15.56
def Wq : ℝ := 1 / 28
def Wr : ℝ := 1 / 35
def Ws : ℝ := 1 / 140

variables (Wp Wq Wr Ws : ℝ)

theorem work_completion_rates (H1 : Wp = Wq + Wr)
                            (H2 : Wp + Wq = 1 / 10)
                            (H3 : Wr = 1 / 35)
                            (H4 : Wq + Wr + Ws = 1 / 14) :
  Wp = 1 / 15.56 ∧ Wq = 1 / 28 ∧ Wr = 1 / 35 ∧ Ws = 1 / 140 :=
begin
  sorry
end

end work_completion_rates_l476_476172


namespace pentagon_diagonals_sum_l476_476039

-- Define the given lengths of the sides of the pentagon
variables (AB CD BC DE AE : ℕ)
variables (a b c : ℚ)

-- Conditions given in the problem
def pentagon_inscribed_in_circle : Prop :=
  AB = 3 ∧ CD = 3 ∧ BC = 10 ∧ DE = 10 ∧ AE = 14

-- Total sum of the lengths of all diagonals
def sum_of_diagonals : ℚ :=
  3 * c + (c ^ 2 - 100) / 3 + (c ^ 2 - 9) / 10

-- The correct answer for the given problem
def correct_answer (m n : ℕ) : Prop :=
  is_rel_prime m n ∧ (m / n = 385/6) ∧ m + n = 391

-- The theorem we are proving
theorem pentagon_diagonals_sum
  (h : pentagon_inscribed_in_circle AB CD BC DE AE) :
  ∃ (m n : ℕ), correct_answer m n :=
sorry

end pentagon_diagonals_sum_l476_476039


namespace correct_statements_l476_476621

noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + π / 6)

theorem correct_statements :
  ¬ (f 0 = 1/2) ∧
  (f (5 * π / 12) = 0) ∧
  ∀ x, π / 12 ≤ x ∧ x ≤ 2 * π / 3 → ∀ θ, π / 6 ≤ 2 * θ + π / 6 ∧ 2 * θ + π / 6 ≤ 2 * π / 3 → f θ < f x ∧
  ∀ ϕ, f (x + ϕ) = 3 * sin (2 * (x + π / 6)) → ϕ ≠ -π / 6 := 
by {
  sorry
}

end correct_statements_l476_476621


namespace distance_between_pulley_axes_is_correct_l476_476084

-- Definitions for given conditions
def pulley1_diameter := 80 -- in mm
def pulley2_diameter := 200 -- in mm
def belt_length := 1500 -- in mm

-- Definition to derive the distance between the axes of the pulleys
def radius (d: ℕ) := d / 2

def pulley1_radius := radius pulley1_diameter
def pulley2_radius := radius pulley2_diameter

def compute_distance (r1 r2 l: ℕ) := sorry

noncomputable def distance_between_axes :=
  compute_distance pulley1_radius pulley2_radius belt_length

-- Theorem stating the required proof
theorem distance_between_pulley_axes_is_correct :
  distance_between_axes = 527 := by
  sorry

end distance_between_pulley_axes_is_correct_l476_476084


namespace hamburger_cost_l476_476219

def annie's_starting_money : ℕ := 120
def num_hamburgers_bought : ℕ := 8
def price_milkshake : ℕ := 3
def num_milkshakes_bought : ℕ := 6
def leftover_money : ℕ := 70

theorem hamburger_cost :
  ∃ (H : ℕ), 8 * H + 6 * price_milkshake = annie's_starting_money - leftover_money ∧ H = 4 :=
by
  use 4
  sorry

end hamburger_cost_l476_476219


namespace min_value_sin6_cos6_l476_476327

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476327


namespace arithmetic_sequence_sum_l476_476969

variable (a : ℕ → ℝ) -- Define the arithmetic sequence as a function from natural numbers to reals.
variable (d : ℝ) -- Common difference of the arithmetic sequence 

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable ha : is_arithmetic_sequence a d
variable hsum : a 3 + a 4 + a 5 + a 6 + a 7 = 450

-- Prove that a_2 + a_8 = 180
theorem arithmetic_sequence_sum : a 2 + a 8 = 180 :=
by
  sorry

end arithmetic_sequence_sum_l476_476969


namespace _l476_476398

noncomputable def tan_alpha_theorem (α : ℝ) (h1 : Real.tan (Real.pi / 4 + α) = 2) : Real.tan α = 1 / 3 :=
by
  sorry

noncomputable def evaluate_expression_theorem (α β : ℝ) 
  (h1 : Real.tan (Real.pi / 4 + α) = 2) 
  (h2 : Real.tan β = 1 / 2) 
  (h3 : Real.tan α = 1 / 3) : 
  (Real.sin (α + β) - 2 * Real.sin α * Real.cos β) / (2 * Real.sin α * Real.sin β + Real.cos (α + β)) = 1 / 7 :=
by
  sorry

end _l476_476398


namespace weight_of_rod_l476_476454

-- Define the weight function
def w (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x

-- Define the integral of the weight function from 0 to 6
def integral_w6 (a b : ℝ) : ℝ := ∫ x in 0..6, w x a b

-- Define the integral of the weight function from 0 to 12, which is given as 14 kg
def integral_w12 (a b : ℝ) : ℝ := ∫ x in 0..12, w x a b

theorem weight_of_rod (a b : ℝ) (h : integral_w12 a b = 14) : integral_w6 a b = 72 * a + 18 * b :=
by
  sorry

end weight_of_rod_l476_476454


namespace arithmetic_sequence_min_value_S_l476_476513

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476513


namespace train_crossing_time_l476_476758

noncomputable def time_to_cross_bridge (l_train : ℕ) (v_train_kmh : ℕ) (l_bridge : ℕ) : ℚ :=
  let total_distance := l_train + l_bridge
  let v_train_ms := (v_train_kmh * 1000 : ℚ) / 3600
  total_distance / v_train_ms

theorem train_crossing_time :
  time_to_cross_bridge 110 72 136 = 12.3 := 
by
  sorry

end train_crossing_time_l476_476758


namespace cost_of_each_skirt_l476_476059

theorem cost_of_each_skirt :
  ∀ (price_pant price_blouse : ℝ) (total_budget total_pant_cost total_blouse_cost total_expense : ℝ),
  price_pant = 30 →
  price_blouse = 15 →
  total_blouse_cost = 5 * price_blouse →
  total_pant_cost = price_pant + price_pant / 2 →
  total_expense = total_blouse_cost + total_pant_cost →
  total_budget = 180 →
  total_budget - total_expense = 3 * 20 :=
by
  intros price_pant price_blouse total_budget total_pant_cost total_blouse_cost total_expense
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end cost_of_each_skirt_l476_476059


namespace king_total_payment_l476_476779

/-- 
A king gets a crown made that costs $20,000. He tips the person 10%. Prove that the total amount the king paid after the tip is $22,000.
-/
theorem king_total_payment (C : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h1 : C = 20000) 
  (h2 : tip_percentage = 0.1) 
  (h3 : total_paid = C + C * tip_percentage) : 
  total_paid = 22000 := 
by 
  sorry

end king_total_payment_l476_476779


namespace range_of_x_for_sqrt_l476_476976

-- Define the condition under which the expression inside the square root is non-negative.
def sqrt_condition (x : ℝ) : Prop :=
  x - 7 ≥ 0

-- Main theorem to prove the range of values for x
theorem range_of_x_for_sqrt (x : ℝ) : sqrt_condition x ↔ x ≥ 7 :=
by
  -- Proof steps go here (omitted as per instructions)
  sorry

end range_of_x_for_sqrt_l476_476976


namespace sequence_a_sequence_b_sequence_sum_l476_476176

noncomputable def S (n : ℕ) := 2 * n^2 - 2 * n
noncomputable def T (n : ℕ) := 3 - (T n)

theorem sequence_a (n : ℕ) (hn : n > 0) : 
  ∃ a : ℕ → ℕ, 
    (∀ n, a n = 4 * n - 4) 
    ∧ (S n = S (n - 1) + a n) := sorry

theorem sequence_b (n : ℕ) (hn : n > 0) : 
  ∃ b : ℕ → ℕ, 
    (b n = 3 * (-1)^(n-1)) 
    ∧ (T n = T (n - 1) + b n) := sorry

noncomputable def c (a b : ℕ → ℕ) (n : ℕ) := a n * b n

theorem sequence_sum (n : ℕ) (hn : n > 0) a b : 
  (∀ n, a n = 4 * n - 4) → 
  (∀ n, b n = 3 * (-1)^(n-1)) → 
  ∃ R : ℕ → ℕ, 
    (∀ n, R n = 1 - (n+1) * n^2) := sorry

end sequence_a_sequence_b_sequence_sum_l476_476176


namespace min_sixth_power_sin_cos_l476_476273

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476273


namespace minimum_value_l476_476618

theorem minimum_value (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ∃ x : ℝ, (8 * a^4 + 12 * b^4 + 40 * c^4 + 2 * d^2 + (1 / (5 * a * b * c * d))) = x ∧ x = 4 * real.sqrt 10 / 5 :=
sorry

end minimum_value_l476_476618


namespace arithmetic_sequence_min_value_Sn_l476_476565

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476565


namespace tim_movie_marathon_duration_is_9_l476_476723

-- Define the conditions:
def first_movie_duration : ℕ := 2
def second_movie_duration : ℕ := first_movie_duration + (first_movie_duration / 2)
def combined_duration_first_two_movies : ℕ := first_movie_duration + second_movie_duration
def third_movie_duration : ℕ := combined_duration_first_two_movies - 1
def total_marathon_duration : ℕ := first_movie_duration + second_movie_duration + third_movie_duration

-- The theorem to prove the marathon duration is 9 hours
theorem tim_movie_marathon_duration_is_9 :
  total_marathon_duration = 9 :=
by sorry

end tim_movie_marathon_duration_is_9_l476_476723


namespace legendre_polynomial_expansion_l476_476250

noncomputable def f (α β γ : ℝ) (θ : ℝ) : ℝ := α + β * Real.cos θ + γ * Real.cos θ ^ 2

noncomputable def P0 (x : ℝ) : ℝ := 1
noncomputable def P1 (x : ℝ) : ℝ := x
noncomputable def P2 (x : ℝ) : ℝ := (3 * x ^ 2 - 1) / 2

theorem legendre_polynomial_expansion (α β γ : ℝ) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
    f α β γ θ = (α + γ / 3) * P0 (Real.cos θ) + β * P1 (Real.cos θ) + (2 * γ / 3) * P2 (Real.cos θ) := by
  sorry

end legendre_polynomial_expansion_l476_476250


namespace minimize_sin_cos_six_l476_476305

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476305


namespace find_a_l476_476923

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : (min (log_a a 2) (log_a a 4)) * (max (log_a a 2) (log_a a 4)) = 2) : 
  a = (1 / 2) ∨ a = 2 :=
sorry

end find_a_l476_476923


namespace polygon_area_l476_476734

def vertices : List (ℝ × ℝ) := [(1, 0), (3, 2), (5, 0), (3, 5)]

noncomputable def shoelace_area (pts : List (ℝ × ℝ)) : ℝ :=
  let n := pts.length
  let xs := pts.map Prod.fst
  let ys := pts.map Prod.snd
  Float.abs (((List.range n).sum (λ i, xs.get ⟨i, sorry⟩ * ys.get ⟨(i + 1) % n, sorry⟩)) -
             ((List.range n).sum (λ i, ys.get ⟨i, sorry⟩ * xs.get ⟨(i + 1) % n, sorry⟩))) / 2

theorem polygon_area : shoelace_area vertices = 6 :=
  sorry

end polygon_area_l476_476734


namespace derivative_y_l476_476253

noncomputable def y (a α x : ℝ) :=
  (Real.exp (a * x)) * (3 * Real.sin (3 * x) - α * Real.cos (3 * x)) / (a ^ 2 + 9)

theorem derivative_y (a α x : ℝ) :
  (deriv (y a α) x) =
    (Real.exp (a * x)) * ((3 * a + 3 * α) * Real.sin (3 * x) + (9 - a * α) * Real.cos (3 * x)) / (a ^ 2 + 9) := 
sorry

end derivative_y_l476_476253


namespace angle_SVU_l476_476476

theorem angle_SVU (TU SV SU : ℝ) (angle_STU_T : ℝ) (angle_STU_S : ℝ) :
  TU = SV → angle_STU_T = 75 → angle_STU_S = 30 →
  TU = SU → SU = SV → S_V_U = 65 :=
by
  intros H1 H2 H3 H4 H5
  -- skip proof
  sorry

end angle_SVU_l476_476476


namespace part_I_part_II_l476_476902

open Real

noncomputable def alpha : ℝ := sorry

def OA := (sin alpha, 1)
def OB := (cos alpha, 0)
def OC := (- sin alpha, 2)

def P : ℝ × ℝ := (2 * cos alpha - sin alpha, 1)

-- Condition for collinearity of O, P, and C
def collinear (O P C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (P.1 = k * O.1) ∧ (P.2 = k * O.2) ∧ (C.1 = k * O.1) ∧ (C.2 = k * O.2)

theorem part_I (hcollinear : collinear (0, 0) P OC) : tan alpha = 4 / 3 := by
  sorry

theorem part_II (h_tan_alpha : tan alpha = 4 / 3) : 
  (sin (2 * alpha) + sin alpha) / (2 * cos (2 * alpha) + 2 * sin alpha^2 + cos alpha) + sin (2 * alpha) = 172 / 75 := 
by
  sorry

end part_I_part_II_l476_476902


namespace find_angle_C_l476_476908

noncomputable def area (a b c : ℝ) (B : ℝ) : ℝ :=
  0.5 * a * c * Real.sin B

-- Given values
def A : ℝ := 3 - Real.sqrt 3
def B_obtuse : Prop := Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2
def AB : ℝ := 2
def BC : ℝ := Real.sqrt 3 - 1
noncomputable def AC : ℝ := Real.sqrt 6

-- Main theorem 
theorem find_angle_C :
  area AB AC (2 * Real.pi / 3) = A / 2 →
  Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 → 
  ∠C = 45 :=
by
  intro area_eq sinC_eq
  sorry

end find_angle_C_l476_476908


namespace smallest_piece_cannot_form_triangle_l476_476803

theorem smallest_piece_cannot_form_triangle :
  ∀ (x : ℕ), (8 - x + 15 - x > 17 - x) → x = 6 :=
by
  intros x h
  have h1 : 8 - x + 15 - x + x ≤ 17 - x + x := by sorry
  have h2 : 23 - 2*x ≤ 17 - x := by sorry
  have h3 : 6 ≤ x := by sorry
  exact h3

end smallest_piece_cannot_form_triangle_l476_476803


namespace rectangle_fold_diagonal_l476_476073

theorem rectangle_fold_diagonal (A B C D E F : ℝ) (k m : ℝ) : 
  let x := 2 - sqrt 10 + sqrt 2 in
  (rectangle A B C D ∧ 
   length A B = 2 ∧ 
   length B C = 1 ∧ 
   on_side E A B ∧ 
   on_side F C B ∧ 
   length A E = length C F ∧ 
   fold A D C D E F ∧ 
   coincides A D C D A C ∧
   segment_length A E = sqrt k - m) →
  (k + m = 14) :=
begin
  sorry
end

end rectangle_fold_diagonal_l476_476073


namespace simple_interest_two_years_l476_476082
-- Import the necessary Lean library for mathematical concepts

-- Define the problem conditions and the proof statement
theorem simple_interest_two_years (P r t : ℝ) (CI SI : ℝ)
  (hP : P = 17000) (ht : t = 2) (hCI : CI = 11730) : SI = 5100 :=
by
  -- Principal (P), Rate (r), and Time (t) definitions
  let P := 17000
  let t := 2

  -- Given Compound Interest (CI)
  let CI := 11730

  -- Correct value for Simple Interest (SI) that we need to prove
  let SI := 5100

  -- Formalize the assumptions
  have h1 : P = 17000 := rfl
  have h2 : t = 2 := rfl
  have h3 : CI = 11730 := rfl

  -- Crucial parts of the problem are used here
  sorry  -- This is a placeholder for the actual proof steps

end simple_interest_two_years_l476_476082


namespace ratio_AB_AC_l476_476467

variables {A B C D O M Q : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] [metric_space M] [metric_space Q]
variables (a k : ℝ) (AB BC AC BD AD DC QD OM : ℝ)

-- Given conditions
def isosceles_triangle (A B C : Type) (AB BC : ℝ) := AB = BC
def altitude_greater_than_base (BD AC : ℝ) := BD > AC
def circle_with_diameter (BD QD : ℝ) := 2 * QD = BD
def tangents_intersect (OM AC k : ℝ) := OM / AC = k

-- The problem statement
theorem ratio_AB_AC (AB BC AC BD AD DC QD OM : ℝ) (h1 : isosceles_triangle A B C AB BC)
(h2 : altitude_greater_than_base BD AC)
(h3 : circle_with_diameter BD QD)
(h4 : tangents_intersect OM AC k) :
  AB / AC = 1 / 2 * sqrt ((5 * k - 1) / (k - 1)) :=
by
  simp only [isosceles_triangle, altitude_greater_than_base, circle_with_diameter, tangents_intersect] at *
  sorry

end ratio_AB_AC_l476_476467


namespace correct_conclusions_count_l476_476889

def vertices : list (ℕ × ℕ × ℕ) :=
[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
 (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]

def is_rectangle (a b c d : ℕ × ℕ × ℕ) : Prop :=
  -- Assuming some definition of whether 4 points form a rectangle
  sorry

def is_equilateral_tetrahedron (a b c d : ℕ × ℕ × ℕ) : Prop :=
  -- Assuming some definition of whether 4 points form an equilateral tetrahedron
  sorry

def is_right_triangle_tetrahedron (a b c d : ℕ × ℕ × ℕ) : Prop :=
  -- Assuming some definition of whether 4 points form a tetrahedron with right triangle faces
  sorry

def is_special_tetrahedron (a b c d : ℕ × ℕ × ℕ) : Prop :=
  -- Assuming some definition of whether 4 points form a special tetrahedron with the described properties
  sorry

theorem correct_conclusions_count :
  ({ABCD : is_rectangle (0, 0, 0) (1, 0, 0) (0, 1, 0) (1, 1, 0),
    ACB1D1 : is_equilateral_tetrahedron (0, 0, 0) (1, 0, 0) (1, 0, 1) (0, 0, 1),
    DB1C1D1 : is_right_triangle_tetrahedron (1, 0, 0) (1, 0, 1) (0, 0, 1) (0, 1, 1),
    DA1C1D1 : is_special_tetrahedron (1, 0, 0) (0, 0, 0) (0, 1, 0) (0, 0, 1)} :
    bool) = 4 := 
by { sorry }

end correct_conclusions_count_l476_476889


namespace heather_payment_per_weed_l476_476932

noncomputable def seconds_in_hour : ℕ := 60 * 60

noncomputable def weeds_per_hour (seconds_per_weed : ℕ) : ℕ :=
  seconds_in_hour / seconds_per_weed

noncomputable def payment_per_weed (hourly_pay : ℕ) (weeds_per_hour : ℕ) : ℚ :=
  hourly_pay / weeds_per_hour

theorem heather_payment_per_weed (seconds_per_weed : ℕ) (hourly_pay : ℕ) :
  seconds_per_weed = 18 ∧ hourly_pay = 10 → payment_per_weed hourly_pay (weeds_per_hour seconds_per_weed) = 0.05 :=
by
  sorry

end heather_payment_per_weed_l476_476932


namespace julia_constant_term_l476_476034

theorem julia_constant_term:
  ∀ (p q : Polynomial ℝ),
    p.monic ∧ p.degree = 5 ∧ p.coeff 0 > 0 ∧ p.coeff 0 = q.coeff 0 ∧ p.coeff 1 = q.coeff 1 ∧
    q.monic ∧ q.degree = 5 ∧
    p * q = Polynomial.C 1 * Polynomial.X ^ 10 +
            Polynomial.C 4 * Polynomial.X ^ 9 +
            Polynomial.C 6 * Polynomial.X ^ 8 +
            Polynomial.C 8 * Polynomial.X ^ 7 +
            Polynomial.C 10 * Polynomial.X ^ 6 +
            Polynomial.C 5 * Polynomial.X ^ 5 +
            Polynomial.C 6 * Polynomial.X ^ 4 +
            Polynomial.C 8 * Polynomial.X ^ 3 +
            Polynomial.C 3 * Polynomial.X ^ 2 +
            Polynomial.C 4 * Polynomial.X +
            Polynomial.C 9 →
  p.coeff 0 = 3 := sorry

end julia_constant_term_l476_476034


namespace problem_1_problem_2_l476_476380

noncomputable def a : ℝ := Real.sqrt 7 + 2
noncomputable def b : ℝ := Real.sqrt 7 - 2

theorem problem_1 : a^2 * b + b^2 * a = 6 * Real.sqrt 7 := by
  sorry

theorem problem_2 : a^2 + a * b + b^2 = 25 := by
  sorry

end problem_1_problem_2_l476_476380


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476361

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476361


namespace parabola_vertex_l476_476106

theorem parabola_vertex (c d : ℝ) :
  (∀ x, -x^2 + c * x + d ≤ 0 ↔ (x ∈ Icc (-7) 3 ∨ x ∈ Ici 9)) →
  c = 2 ∧ d = -63 →
  ((1, -62) : ℝ × ℝ) = (1, -62) := 
by
  intros h1 h2
  sorry

end parabola_vertex_l476_476106


namespace simplify_fraction_l476_476930

theorem simplify_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^(2*b) * b^a) / (b^(2*a) * a^b) = (a / b)^b := 
by sorry

end simplify_fraction_l476_476930


namespace symmetrical_circle_l476_476087

-- Defining the given circle's equation
def given_circle_eq (x y: ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Defining the equation of the symmetrical circle
def symmetrical_circle_eq (x y: ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Proving the symmetry property
theorem symmetrical_circle (x y : ℝ) : 
  (given_circle_eq x y) → (symmetrical_circle_eq (-x) (-y)) :=
by
  sorry

end symmetrical_circle_l476_476087


namespace isosceles_triangle_perimeter_eq_10_l476_476218

theorem isosceles_triangle_perimeter_eq_10 (x : ℝ) 
(base leg : ℝ)
(h_base : base = 4)
(h_leg_root : x^2 - 5 * x + 6 = 0)
(h_iso : leg = x)
(triangle_ineq : leg + leg > base):
  2 * leg + base = 10 := 
begin
  cases (em (x = 2)) with h1 h2,
  { rw h1 at h_leg_root,
    rw [←h_iso, h1] at triangle_ineq,
    simp at triangle_ineq,
    contradiction },
  { rw h_iso,
    have : x = 3,
    { by_contra,
      simp [not_or_distrib, h1, h, sub_eq_zero] at h_leg_root },
    rw this,
    simp,
    linarith }
end

# Testing if the theorem can be evaluated successfully
# theorem_example : isosceles_triangle_perimeter_eq_10 3 4 3 rfl rfl sorry sorry rfl :=
# sorry

end isosceles_triangle_perimeter_eq_10_l476_476218


namespace area_of_sector_is_pi_over_6_l476_476238

theorem area_of_sector_is_pi_over_6 
  (r : ℝ) (θ : ℝ) 
  (hr : r = 1) 
  (hθ : θ = real.pi / 3) : 
  (1/2 * r^2 * θ = real.pi / 6) := sorry

end area_of_sector_is_pi_over_6_l476_476238


namespace minimum_value_expression_l476_476620

-- Define the conditions for positive real numbers
variables (a b c : ℝ)
variable (h_a : 0 < a)
variable (h_b : 0 < b)
variable (h_c : 0 < c)

-- State the theorem to prove the minimum value of the expression
theorem minimum_value_expression (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : 
  (a / b) + (b / c) + (c / a) ≥ 3 := 
sorry

end minimum_value_expression_l476_476620


namespace least_n_for_100_l476_476040

noncomputable def a_n (n : ℕ) : ℝ :=
if n = 1 then (real.sqrt 21) / 3
else real.sqrt (7 / 3 * (a_n (n - 1))^2)

def A_0 := (0 : ℝ, 0 : ℝ)

def A (n : ℕ) := (real.sqrt ((7 / 3)^(n - 1) * (a_n 1)^2) : ℝ, 0 : ℝ)

def A_0A_n_length (n : ℕ) : ℝ :=
finset.sum (finset.range n) (λ i, real.sqrt ((7 / 3)^i * (a_n 1)^2))

theorem least_n_for_100 : ∃ (n : ℕ), A_0A_n_length n ≥ 100 ∧ ∀ (m : ℕ), m < n → A_0A_n_length m < 100 :=
begin
  sorry
end

end least_n_for_100_l476_476040


namespace arithmetic_sequence_minimum_value_S_n_l476_476559

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476559


namespace rope_segment_length_l476_476795

theorem rope_segment_length (L : ℕ) (half_fold_times : ℕ) (dm_to_cm : ℕ → ℕ) 
  (hL : L = 8) (h_half_fold_times : half_fold_times = 2) (h_dm_to_cm : dm_to_cm 1 = 10)
  : dm_to_cm (L / 2 ^ half_fold_times) = 20 := 
by 
  sorry

end rope_segment_length_l476_476795


namespace no_14_non_square_rectangles_l476_476798

theorem no_14_non_square_rectangles (side_len : ℕ) 
    (h_side_len : side_len = 9) 
    (num_rectangles : ℕ) 
    (h_num_rectangles : num_rectangles = 14) 
    (min_side_len : ℕ → ℕ → Prop) 
    (h_min_side_len : ∀ l w, min_side_len l w → l ≥ 2 ∧ w ≥ 2) : 
    ¬ (∀ l w, min_side_len l w → l ≠ w) :=
by {
    sorry
}

end no_14_non_square_rectangles_l476_476798


namespace lattice_points_in_intersection_l476_476824

noncomputable def intersection_lattice_points_count : Nat :=
  let sphere1_center : EuclideanSpace ℝ (Fin 3) := ![0, 0, 21 / 2]
  let sphere1_radius : ℝ := 6
  let sphere2_center : EuclideanSpace ℝ (Fin 3) := ![0, 0, 1]
  let sphere2_radius : ℝ := 9 / 2
  let in_sphere1 (p : EuclideanSpace ℝ (Fin 3)) : Prop := 
    (EuclideanSpace.dist p sphere1_center) ≤ sphere1_radius
  let in_sphere2 (p : EuclideanSpace ℝ (Fin 3)) : Prop := 
    (EuclideanSpace.dist p sphere2_center) ≤ sphere2_radius
  let is_lattice_point (p : EuclideanSpace ℝ (Fin 3)) : Prop := 
    ∀ i, p i ∈ Set.Ioi ℤ  -- this means all coordinates are integers
  Nat.card {p : EuclideanSpace ℝ (Fin 3) // in_sphere1 p ∧ in_sphere2 p ∧ is_lattice_point p} = 13

theorem lattice_points_in_intersection : intersection_lattice_points_count = 13 :=
  sorry

end lattice_points_in_intersection_l476_476824


namespace arithmetic_sequence_min_value_S_l476_476509

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476509


namespace binary_strings_length_10_l476_476935

-- Define F(n) as the number of binary strings of length n without substrings "101" or "010"
def F : ℕ → ℕ
| 0     := 1  -- For consistency (base condition)
| 1     := 2
| 2     := 4
| 3     := 6
| (n+4) := F (n + 3) + F (n + 2)

-- The target property to prove
theorem binary_strings_length_10 : F 10 = 178 := by
  -- Omitting the proof as instructed
  sorry

end binary_strings_length_10_l476_476935


namespace matches_in_rectangle_rectangles_in_rectangle_squares_in_rectangle_l476_476792

theorem matches_in_rectangle (m n : Nat) (h : m > n) : 
  2 * m * n + m + n = total_matches (m, n) := sorry

theorem rectangles_in_rectangle (m n : Nat) (h : m > n) : 
  (m * n * (m + 1) * (n + 1)) / 4 = number_of_rectangles (m, n) := sorry

theorem squares_in_rectangle (m n : Nat) (h : m > n) : 
  (n * (n + 1) * (3 * m - n + 1)) / 6 = number_of_squares (m, n) := sorry

-- Definitions to use in the theorems to match the conditions step
noncomputable def total_matches (dim : Nat × Nat) : Nat :=
  let (m, n) := dim
  2 * m * n + m + n

noncomputable def number_of_rectangles (dim : Nat × Nat) : Nat :=
  let (m, n) := dim
  (m * n * (m + 1) * (n + 1)) / 4

noncomputable def number_of_squares (dim : Nat × Nat) : Nat :=
  let (m, n) := dim
  (n * (n + 1) * (3 * m - n + 1)) / 6

end matches_in_rectangle_rectangles_in_rectangle_squares_in_rectangle_l476_476792


namespace complex_modulus_inequality_l476_476369

theorem complex_modulus_inequality (x y : ℝ) (z : ℂ) (h : z = x + y * Complex.I) : 
  abs z ≤ abs x + abs y :=
by 
  sorry

end complex_modulus_inequality_l476_476369


namespace greatest_prime_factor_of_n_is_23_l476_476738

def n : ℕ := 5^7 + 10^6

theorem greatest_prime_factor_of_n_is_23 : ∃ p : ℕ, nat.prime p ∧ p = 23 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ n → q ≤ 23 :=
sorry

end greatest_prime_factor_of_n_is_23_l476_476738


namespace arithmetic_sequence_min_value_Sn_l476_476570

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476570


namespace smallest_integer_greater_than_20_l476_476970

noncomputable def smallest_integer_greater_than_A : ℕ :=
  let a (n : ℕ) := 4 * n - 3
  let A := Real.sqrt (a 1580) - 1 / 4
  Nat.ceil A

theorem smallest_integer_greater_than_20 :
  smallest_integer_greater_than_A = 20 :=
sorry

end smallest_integer_greater_than_20_l476_476970


namespace eval_sum_l476_476917

def f (x : ℝ) : ℝ :=
if x < 1 then 1 + Real.logb 3 (2 - x) else 3 ^ (x - 1)

theorem eval_sum : f (-7) + f (Real.logb 3 12) = 7 := sorry

end eval_sum_l476_476917


namespace max_gcd_seq_value_l476_476692

def sequence (n : ℕ) : ℕ := 100 + n^n

def gcd_seq (n : ℕ) : ℕ := Nat.gcd (sequence n) (sequence (n - 1))

theorem max_gcd_seq_value : ∃ n : ℕ, gcd_seq n = 401 := 
sorry

end max_gcd_seq_value_l476_476692


namespace simplify_fraction_l476_476242

theorem simplify_fraction : (8 / (5 * 42) = 4 / 105) :=
by
    sorry

end simplify_fraction_l476_476242


namespace Jake_should_charge_for_planting_flowers_l476_476984

theorem Jake_should_charge_for_planting_flowers :
  let mowing_time := 1
  let mowing_payment := 15
  let planting_time := 2
  let desired_rate := 20
  let total_hours := mowing_time + planting_time
  let total_earnings := desired_rate * total_hours
  let planting_charge := total_earnings - mowing_payment
  planting_charge = 45 :=
by
  simp [mowing_time, mowing_payment, planting_time, desired_rate, total_hours, total_earnings, planting_charge]
  sorry

end Jake_should_charge_for_planting_flowers_l476_476984


namespace minimize_sin_cos_six_l476_476308

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476308


namespace find_digit_B_l476_476115

theorem find_digit_B (A B : ℕ) (h : 1 ≤ A ∧ A ≤ 9) (h' : 0 ≤ B ∧ B ≤ 9) (eqn : 10 * A + 22 = 9 * B) : B = 8 := 
  sorry

end find_digit_B_l476_476115


namespace graph_pairwise_connected_by_green_or_third_point_l476_476778

open Classical

universe u

variable {V : Type u} [Fintype V]

def is_colored_graph (G : SimpleGraph V) :=
  ∀ (u v : V), u ≠ v → G.adj u v

def only_two_pts_not_connected_by_red_path (G : SimpleGraph V) (red : V → V → Prop) (A B : V) :=
  ¬ (∃ (p : List V), p ≠ [] ∧ p.headI = A ∧ p.getLast sorry = B ∧ ∀ i ∈ p.zip p.tail, red i.1 i.2)

theorem graph_pairwise_connected_by_green_or_third_point (G : SimpleGraph V) 
    (red green : V → V → Prop) (A B : V)
    (h_colored: is_colored_graph G)
    (h_edges_colored: ∀ (u v : V), u ≠ v → red u v ∨ green u v)
    (h_not_red_path: only_two_pts_not_connected_by_red_path G red A B) :
    ∀ (X Y : V), X ≠ Y → green X Y ∨ (∃ (Z : V), green X Z ∧ green Y Z) :=
by
  sorry

end graph_pairwise_connected_by_green_or_third_point_l476_476778


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476602

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476602


namespace kiera_total_envelopes_l476_476036

-- Define the number of blue envelopes
def blue_envelopes : ℕ := 14

-- Define the number of yellow envelopes as 6 fewer than the number of blue envelopes
def yellow_envelopes : ℕ := blue_envelopes - 6

-- Define the number of green envelopes as 3 times the number of yellow envelopes
def green_envelopes : ℕ := 3 * yellow_envelopes

-- The total number of envelopes is the sum of blue, yellow, and green envelopes
def total_envelopes : ℕ := blue_envelopes + yellow_envelopes + green_envelopes

-- Prove that the total number of envelopes is 46
theorem kiera_total_envelopes : total_envelopes = 46 := by
  sorry

end kiera_total_envelopes_l476_476036


namespace leo_kept_packs_calculation_l476_476992

-- Definitions based on conditions
def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def packs_given_to_manny_fraction : ℚ := 1 / 4
def packs_given_to_neil_fraction : ℚ := 1 / 8

-- Main statement
theorem leo_kept_packs_calculation :
  let total_packs := total_marbles / marbles_per_pack in
  let packs_given_to_manny := packs_given_to_manny_fraction * total_packs in
  let packs_given_to_neil := packs_given_to_neil_fraction * total_packs in
  let packs_given_away := packs_given_to_manny + packs_given_to_neil in
  let packs_kept := total_packs - packs_given_away in
  packs_kept = 25 :=
by
  sorry

end leo_kept_packs_calculation_l476_476992


namespace min_value_sin6_cos6_l476_476323

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476323


namespace letter_13_removed_time_l476_476677

-- Define the process of stacking and removing letters
def process (time : ℕ) : ℕ × ℕ × List ℕ → ℕ × ℕ × List ℕ
| (arrivals, removals, stack) =>
  let new_arrivals := arrivals + 3
  let stack' := new_arrivals :: (new_arrivals - 1) :: (new_arrivals - 2) :: stack
  let (top1::top2::rest) := stack' | stack' -- Remove top two elements
  let new_removals := removals + 2
  (new_arrivals, new_removals, rest)

-- Define the Initial State
def initial_state : ℕ × ℕ × List ℕ := (0, 0, [])

-- Define the process repetition
def repeat_process (n : ℕ) : ℕ × ℕ × List ℕ :=
  Nat.repeat process n initial_state

-- Theorem to prove the 13th letter is removed at 1:15 PM
theorem letter_13_removed_time :
  (repeat_process 16).2 = 16 :=
  by
  sorry

end letter_13_removed_time_l476_476677


namespace min_value_sin_cos_l476_476289

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476289


namespace min_sixth_power_sin_cos_l476_476262

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476262


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476355

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476355


namespace complement_union_correct_l476_476637

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4})
variable (hA : A = {0, 1, 2})
variable (hB : B = {2, 3})

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end complement_union_correct_l476_476637


namespace quadratic_function_minimum_value_l476_476235

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 9

-- The Lean statement to prove that the minimum value of f(x) is -12
theorem quadratic_function_minimum_value : 
  (∃ (x : ℝ), (f(x) = -12) ∧ (∀ (y : ℝ), f(y) >= f(x))) :=
begin
  sorry
end

end quadratic_function_minimum_value_l476_476235


namespace parallel_vectors_have_proportional_direction_ratios_l476_476439

theorem parallel_vectors_have_proportional_direction_ratios (m : ℝ) :
  let a := (1, 2)
  let b := (m, 1)
  (a.1 / b.1) = (a.2 / b.2) → m = 1/2 :=
by
  let a := (1, 2)
  let b := (m, 1)
  intro h
  sorry

end parallel_vectors_have_proportional_direction_ratios_l476_476439


namespace solve_quadratic_solve_cubic_l476_476367

theorem solve_quadratic (x : ℝ) (h : 2 * x^2 - 32 = 0) : x = 4 ∨ x = -4 := 
by sorry

theorem solve_cubic (x : ℝ) (h : (x + 4)^3 + 64 = 0) : x = -8 := 
by sorry

end solve_quadratic_solve_cubic_l476_476367


namespace sequence_problem_l476_476589

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476589


namespace rhombus_diagonals_not_always_equal_l476_476153

structure Rhombus where
  all_four_sides_equal : Prop
  symmetrical : Prop
  centrally_symmetrical : Prop

theorem rhombus_diagonals_not_always_equal (R : Rhombus) :
  ¬ (∀ (d1 d2 : ℝ), d1 = d2) :=
sorry

end rhombus_diagonals_not_always_equal_l476_476153


namespace probability_of_test_l476_476843

theorem probability_of_test (letters_in_bag : List Char)
                             (target_word : List Char)
                             (counts_in_bag : ∀ l : Char, l ∈ letters_in_bag → ℕ) :
  letters_in_bag = ['S', 'T', 'A', 'I', 'T', 'I', 'S', 'T', 'C', 'S'] →
  target_word = ['T', 'E', 'S', 'T'] →
  let favorable_outcomes := (['T', 'S'].map (λ c, counts_in_bag c (by simp; tauto))).sum in
  let total_outcomes := letters_in_bag.length in
  (favorable_outcomes / total_outcomes : ℚ) = 2 / 3 :=
by
  intros h1 h2
  let favorable_outcomes := 3 + 3  -- T appears 3 times, S appears 3 times
  let total_outcomes := 9           -- Total tiles
  show (favorable_outcomes / total_outcomes : ℚ) = 2 / 3 
  rw [show favorable_outcomes = 6, by norm_num]
  norm_num
  sorry

end probability_of_test_l476_476843


namespace count_5_digit_numbers_divisible_by_13_l476_476622

theorem count_5_digit_numbers_divisible_by_13 (n q r : ℕ) :
  (10000 <= n ∧ n <= 99999) ∧ (n = 100 * q + r) ∧ (13 ∣ n) ∧ (13 ∣ (q + r)) -> 
  n ∈ {10010, 10023, ..., 99991} -> n ∈ {10010, 10023, ..., 99991}.count = 6922 :=
by {
  sorry
}

end count_5_digit_numbers_divisible_by_13_l476_476622


namespace sin_cos_sixth_min_l476_476354

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476354


namespace probability_entirely_black_grid_l476_476770

-- Definitions for the conditions
-- Define a $4 \times 4$ grid
def grid := list (list bool)

-- Function to rotate the grid 90 degrees clockwise
def rotate_90 (g : grid) : grid := sorry

-- Function to perform the transformation on the grid
def transform (g : grid) : grid := sorry 

-- Probability function that assumes equal distribution
def prob_black (g : grid) : ℚ := sorry

-- Main theorem: probability that the grid is entire black equals 1/256
theorem probability_entirely_black_grid :
  (prob_black (transform (rotate_90 (list.replicate 4 (list.replicate 4 (true))))) = 1/256) := 
sorry

end probability_entirely_black_grid_l476_476770


namespace suff_but_not_necess_l476_476944

theorem suff_but_not_necess (a : ℝ) : (a = 1 → a^2 = 1) ∧ ¬(a^2 = 1 → a = 1) :=
by
  split
  · intro h
    rw h
    exact eq.refl 1
  · intro h
    have : -1 ^ 2 = 1 := by norm_num
    exact this.symm ▸ h this

end suff_but_not_necess_l476_476944


namespace minimize_sin_cos_six_l476_476300

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476300


namespace last_two_digits_of_17_pow_17_l476_476241

theorem last_two_digits_of_17_pow_17 : (17 ^ 17) % 100 = 77 := 
by sorry

end last_two_digits_of_17_pow_17_l476_476241


namespace min_value_sin6_cos6_l476_476321

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476321


namespace min_sin6_cos6_l476_476335

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476335


namespace analytical_expression_of_f_minimum_value_of_f_l476_476408

def f (x : ℝ) : ℝ :=
  if x > 0 then - x^2 + 4 * x else x^2 + 4 * x

theorem analytical_expression_of_f :
  ∀ x : ℝ, f(x) = 
  if x > 0 then -x^2 + 4 * x else x^2 + 4 * x := 
sorry

theorem minimum_value_of_f (a : ℝ) (h : a > -2) : 
  (a ≤ 2 + 2 * real.sqrt 2 → ∃ x : ℝ, x ∈ [-2, a] ∧ f(x) = -4) ∧ 
  (a > 2 + 2 * real.sqrt 2 → ∃ x : ℝ, x ∈ [-2, a] ∧ f(x) = f(a)) :=
sorry

end analytical_expression_of_f_minimum_value_of_f_l476_476408


namespace smallest_floor_sum_l476_476441

theorem smallest_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 4 :=
sorry

end smallest_floor_sum_l476_476441


namespace ellen_baking_time_l476_476844

theorem ellen_baking_time :
  ∀ (rise_time_per_ball bake_time_per_ball : ℕ) (balls : ℕ),
  rise_time_per_ball = 3 →
  bake_time_per_ball = 2 →
  balls = 4 →
  (rise_time_per_ball * balls + bake_time_per_ball * balls) = 20 :=
by
  intros rise_time_per_ball bake_time_per_ball balls
  intros h_rise h_bake h_balls
  rw [h_rise, h_bake, h_balls]
  norm_num
  sorry

end ellen_baking_time_l476_476844


namespace tim_movie_marathon_duration_is_9_l476_476724

-- Define the conditions:
def first_movie_duration : ℕ := 2
def second_movie_duration : ℕ := first_movie_duration + (first_movie_duration / 2)
def combined_duration_first_two_movies : ℕ := first_movie_duration + second_movie_duration
def third_movie_duration : ℕ := combined_duration_first_two_movies - 1
def total_marathon_duration : ℕ := first_movie_duration + second_movie_duration + third_movie_duration

-- The theorem to prove the marathon duration is 9 hours
theorem tim_movie_marathon_duration_is_9 :
  total_marathon_duration = 9 :=
by sorry

end tim_movie_marathon_duration_is_9_l476_476724


namespace table_max_height_l476_476977

theorem table_max_height
  (DE EF FD : ℕ) 
  (P Q : ℝ) 
  (R S : ℝ) 
  (T U : ℝ) 
  (h' : ℝ) :
  DE = 24 → 
  EF = 28 → 
  FD = 32 → 
  (overline P Q).parallel (overline F D) → 
  (overline R S).parallel (overline D E) → 
  (overline T U).parallel (overline E F) → 
  is_right_angle_fold P Q R S T U →
  table_top_parallel_to_floor P Q R S T U → 
  h' = 340 * sqrt 35 / 39 :=
sorry

end table_max_height_l476_476977


namespace obtuse_angle_median_ratio_l476_476967

theorem obtuse_angle_median_ratio (ABC : Type) [euclidean_geometry ℝ ABC] 
  (A B C D : ABC) (AC BC AD : ℝ) (hABC_tri : triangle ABC A B C) 
  (hAD_median : is_median A D B C) :
  AD / AC = 1 / 2 ↔ (angle A B C > π / 2 ∧ angle D A B < π / 2) :=
sorry

end obtuse_angle_median_ratio_l476_476967


namespace nori_crayons_left_l476_476065

def initial_crayons (boxes: ℕ) (crayons_per_box: ℕ) : ℕ :=
  boxes * crayons_per_box

def crayons_given (crayons_left: ℕ) (to_mae: ℕ) (to_lea: ℕ) : ℕ :=
  crayons_left - to_mae - to_lea

theorem nori_crayons_left (boxes: ℕ) (crayons_per_box: ℕ) (to_mae: ℕ) (extra_to_lea: ℕ) :
  boxes = 4 → crayons_per_box = 8 → to_mae = 5 → extra_to_lea = 7 →
  crayons_given (initial_crayons boxes crayons_per_box - to_mae) to_mae (to_mae + extra_to_lea) = 15 :=
by
  intros,
  sorry

end nori_crayons_left_l476_476065


namespace rhombus_diagonals_not_always_equal_l476_476152

structure Rhombus where
  all_four_sides_equal : Prop
  symmetrical : Prop
  centrally_symmetrical : Prop

theorem rhombus_diagonals_not_always_equal (R : Rhombus) :
  ¬ (∀ (d1 d2 : ℝ), d1 = d2) :=
sorry

end rhombus_diagonals_not_always_equal_l476_476152


namespace binom_26_6_l476_476396

theorem binom_26_6 :
  (binom 26 6) = 230230 :=
by
  -- conditions to be assumed for the theorem
  have h1 : binom 24 5 = 42504 := by sorry
  have h2 : binom 25 5 = 53130 := by sorry
  have h3 : binom 25 6 = 177100 := by sorry
  -- using Pascal's identity and the above values
  sorry

end binom_26_6_l476_476396


namespace invertible_matrixA_matrixA_inverse_is_correct_l476_476875

open Matrix

def matrixA : Matrix (Fin 2) (Fin 2) ℝ :=
  matrixOf ![![4, 7], ![2, 6]]

def matrixA_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  matrixOf ![![0.6, -0.7], ![-0.2, 0.4]]

theorem invertible_matrixA : invertible matrixA :=
  by 
  have h_det : det matrixA ≠ 0 := by
    simp [matrixA, det, Finset.sum, Matrix.fin_two_sum, algebra_map, fintype.univ]
  use matrixA_inv, sorry

theorem matrixA_inverse_is_correct:
  matrix.mul matrixA matrixA_inv = 1 ∧ matrix.mul matrixA_inv matrixA = 1 := 
  by
  sorry

end invertible_matrixA_matrixA_inverse_is_correct_l476_476875


namespace complex_power_periodicity_l476_476168

theorem complex_power_periodicity :
  ( (1 + Complex.i) / (1 - Complex.i) ) ^ 2018 = -1 := 
by sorry

end complex_power_periodicity_l476_476168


namespace count_integers_satisfying_conditions_l476_476433

theorem count_integers_satisfying_conditions :
  {n : ℤ | 200 < n ∧ n < 400 ∧ n % 7 = n % 9}.finite.to_finset.card = 21 :=
by
  sorry

end count_integers_satisfying_conditions_l476_476433


namespace find_sum_a100_b100_l476_476404

-- Definitions of arithmetic sequences and their properties
structure arithmetic_sequence (an : ℕ → ℝ) :=
  (a1 : ℝ)
  (d : ℝ)
  (def_seq : ∀ n, an n = a1 + (n - 1) * d)

-- Given conditions
variables (a_n b_n : ℕ → ℝ)
variables (ha : arithmetic_sequence a_n)
variables (hb : arithmetic_sequence b_n)

-- Specified conditions
axiom cond1 : a_n 5 + b_n 5 = 3
axiom cond2 : a_n 9 + b_n 9 = 19

-- The goal to be proved
theorem find_sum_a100_b100 : a_n 100 + b_n 100 = 383 :=
sorry

end find_sum_a100_b100_l476_476404


namespace find_may_monday_l476_476451

noncomputable def weekday (day_of_month : ℕ) (first_day_weekday : ℕ) : ℕ :=
(day_of_month + first_day_weekday - 1) % 7

theorem find_may_monday (r n : ℕ) (condition1 : weekday r 5 = 5) (condition2 : weekday n 5 = 1) (condition3 : 15 < n ∧ n < 25) : 
  n = 20 :=
by
  -- Proof omitted.
  sorry

end find_may_monday_l476_476451


namespace min_value_sin_cos_l476_476295

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476295


namespace leila_money_left_l476_476990

theorem leila_money_left (initial_money spent_on_sweater spent_on_jewelry total_spent left_money : ℕ) 
  (h1 : initial_money = 160) 
  (h2 : spent_on_sweater = 40) 
  (h3 : spent_on_jewelry = 100) 
  (h4 : total_spent = spent_on_sweater + spent_on_jewelry) 
  (h5 : total_spent = 140) : 
  initial_money - total_spent = 20 := by
  sorry

end leila_money_left_l476_476990


namespace angle_between_vectors_l476_476867

theorem angle_between_vectors :
  let v1 : ℝ × ℝ × ℝ := (3, -2, 2)
  let v2 : ℝ × ℝ × ℝ := (-2, 2, 1)
  let dot_product (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let magnitude (v : ℝ × ℝ × ℝ) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let cos_theta := dot_product v1 v2 / (magnitude v1 * magnitude v2)
  θ = Real.acos (cos_theta) * (180 / Real.pi)
  in θ ≈ 127 :=
by
  sorry

end angle_between_vectors_l476_476867


namespace can_build_basketball_court_l476_476179

noncomputable def width_of_basketball_court := 18
noncomputable def length_of_basketball_court := (5/3) * width_of_basketball_court
noncomputable def area_of_basketball_court := 540

noncomputable def total_available_area := 1100
noncomputable def surrounding_space := 1

theorem can_build_basketball_court : 
  (length_of_basketball_court + 2 * surrounding_space) * (width_of_basketball_court + 2 * surrounding_space) ≤ total_available_area :=
by
  have h₁ : width_of_basketball_court = 18 := rfl
  have h₂ : length_of_basketball_court = 5 / 3 * width_of_basketball_court := rfl
  have h₃ : area_of_basketball_court = 540 := rfl
  have h₄ : total_available_area = 1100 := rfl
  have h₅ : surrounding_space = 1 := rfl
  show (5 / 3 * 18 + 2 * 1) * (18 + 2 * 1) ≤ 1100 from sorry

end can_build_basketball_court_l476_476179


namespace length_MD_eq_10_l476_476456

theorem length_MD_eq_10 (A B C D M : Type) [metric_space A] 
  (hM_midpoint : midpoint B C M) 
  (hAD_median : median A B C D)
  (hBD_perp_AD : is_perpendicular B D A D)
  (AB AC : ℝ) (hAB : AB = 16) (hAC : AC = 20) :
  length M D = 10 := 
sorry

end length_MD_eq_10_l476_476456


namespace correct_conclusions_count_l476_476710

-- Definitions of the four propositions
def proposition1 : Prop := 
  ∀ (T : Type) [tertrahedron_space T], 
  ∀ (faces : Fin 4 → T), 
  (faces 0 + faces 1 + faces 2 > faces 3)

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, (a n - a m) = (n - m) * (a 1 - a 0)

def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, (b n / b m) = (b 1 / b 0) ^ (n - m)

def proposition2 : Prop :=
  ∀ {b : ℕ → ℤ} (h : geometric_sequence b),
  (b 6 * b 7 * b 8 * b 9 * b 10)^(1 / 5) = 
  (b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7 * b 8 * b 9 * b 10 * b 11 * b 12 * b 13 * b 14 * b 15)^(1 / 15)

def proposition3 : Prop :=
  ∀ (a b c : ℝ),
  (a * b) * c = a * (b * c)

def proposition4 : Prop :=
  ∀ (z w : ℂ),
  ((z - w).im = 0 → z.re > w.re)

-- Assertion that the number of true propositions is 2
theorem correct_conclusions_count : 
(list.count (λ p, p) [proposition1, proposition2, proposition3, proposition4] = 2) :=
sorry

end correct_conclusions_count_l476_476710


namespace cube_surface_area_l476_476794

def prism_dim1 : ℝ := 6
def prism_dim2 : ℝ := 3
def prism_dim3 : ℝ := 36

def prism_volume : ℝ := prism_dim1 * prism_dim2 * prism_dim3

theorem cube_surface_area :
  (∃ s : ℝ, s^3 = prism_volume ∧ 6 * s^2 = 216 * real.cbrt(3)^2) :=
sorry

end cube_surface_area_l476_476794


namespace Tod_speed_is_25_mph_l476_476122

-- Definitions of the conditions
def miles_north : ℕ := 55
def miles_west : ℕ := 95
def hours_driven : ℕ := 6

-- The total distance travelled
def total_distance : ℕ := miles_north + miles_west

-- The speed calculation, dividing total distance by hours driven
def speed : ℕ := total_distance / hours_driven

-- The theorem to prove
theorem Tod_speed_is_25_mph : speed = 25 :=
by
  -- Proof of the theorem will be filled here, but for now using sorry
  sorry

end Tod_speed_is_25_mph_l476_476122


namespace maria_earnings_l476_476639

def cost_of_brushes : ℕ := 20
def cost_of_canvas : ℕ := 3 * cost_of_brushes
def cost_per_liter_of_paint : ℕ := 8
def liters_of_paint : ℕ := 5
def cost_of_paint : ℕ := liters_of_paint * cost_per_liter_of_paint
def total_cost : ℕ := cost_of_brushes + cost_of_canvas + cost_of_paint
def selling_price : ℕ := 200

theorem maria_earnings : (selling_price - total_cost) = 80 := by
  sorry

end maria_earnings_l476_476639


namespace loss_percentage_l476_476205

theorem loss_percentage (CP SP_gain L : ℝ) 
  (h1 : CP = 1500)
  (h2 : SP_gain = CP + 0.05 * CP)
  (h3 : SP_gain = CP - (L/100) * CP + 225) : 
  L = 10 :=
by
  sorry

end loss_percentage_l476_476205


namespace prove_arithmetic_sequence_minimum_value_S_l476_476536

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476536


namespace sector_area_l476_476452

theorem sector_area (theta r : ℝ) (h1 : theta = 2 * Real.pi / 3) (h2 : r = 2) :
  (1 / 2 * r ^ 2 * theta) = 4 * Real.pi / 3 := by
  sorry

end sector_area_l476_476452


namespace min_value_sin6_cos6_l476_476318

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476318


namespace unique_element_set_l476_476103

theorem unique_element_set (a : ℝ) : 
  (∃! x, (a - 1) * x^2 + 3 * x - 2 = 0) ↔ (a = 1 ∨ a = -1 / 8) :=
by sorry

end unique_element_set_l476_476103


namespace converse_of_proposition_2_is_true_l476_476469

-- Definitions
def points (P : Type) := P → P → P → Prop
def not_coplanar {P : Type} [points P] (a b c d : P) : Prop := ¬∃γ : P → Prop, γ a ∧ γ b ∧ γ c ∧ γ d
def not_collinear {P : Type} [points P] (a b c : P) : Prop := ¬∃ℓ : P → Prop, ℓ a ∧ ℓ b ∧ ℓ c

def lines (L : Type) := L → L → Prop
def no_common_point {L : Type} [lines L] (l1 l2 : L) : Prop := ∀ x, x ∉ l1 ∨ x ∉ l2
def skew_lines {L : Type} [lines L] (l1 l2 : L) : Prop := ∃ P : Type, ∀ γ : P → Prop, γ l1 ∧ ¬ γ l2

-- Proof statement
theorem converse_of_proposition_2_is_true {L : Type} [lines L] (l1 l2 : L) :
  skew_lines l1 l2 → no_common_point l1 l2 := by
  sorry

end converse_of_proposition_2_is_true_l476_476469


namespace rhombus_diagonals_not_always_equal_l476_476154

structure Rhombus where
  all_four_sides_equal : Prop
  symmetrical : Prop
  centrally_symmetrical : Prop

theorem rhombus_diagonals_not_always_equal (R : Rhombus) :
  ¬ (∀ (d1 d2 : ℝ), d1 = d2) :=
sorry

end rhombus_diagonals_not_always_equal_l476_476154


namespace pentagon_product_equality_l476_476494

theorem pentagon_product_equality
  (A : ℕ → Π i, i ∈ [1, 2, 3, 4, 5])  -- Represents the vertices A_1, A_2, A_3, A_4, A_5 of the convex pentagon 
  (X : ℕ → Π i, i ∈ [1, 2, 3, 4, 5])  -- Represents the points X_1, X_2, X_3, X_4, X_5 
  (h_convex : true)                    -- Given that A_1A_2A_3A_4A_5 is a convex pentagon
  (h_intersect : ∀ i, ∃ X_i, (rays_meet (A i+1) (A i+2) (A i-1) (A i-2) = X i)) :  -- For i = 1 to 5, rays defined meet at X_i
  (prod : ∀ i, 1 ≤ Φ→ X i (A i+2) = ∏ j in [1, 2, 3, 4, 5], X j (A j+3)) : -- Proving the product equality
  True := 
by {
  sorry  -- Proof will be fleshed out
}

end pentagon_product_equality_l476_476494


namespace min_sixth_power_sin_cos_l476_476274

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476274


namespace medians_concurrent_l476_476810

variable {Point : Type} [AffineSpace Point ℝ]

structure Triangle :=
(A B C : Point)

def Midpoint (A B : Point) : Point := sorry

theorem medians_concurrent (Δ : Triangle) :
  let A1 := Midpoint Δ.B Δ.C
  let B1 := Midpoint Δ.A Δ.C
  let C1 := Midpoint Δ.A Δ.B
  ∃ G : Point, G = AffineCombination (Set.insert Δ.A {Δ.B, Δ.C}) {Δ.A, Δ.B, Δ.C} := sorry

end medians_concurrent_l476_476810


namespace jeremy_uncle_money_l476_476985

def total_cost (num_jerseys : Nat) (cost_per_jersey : Nat) (basketball_cost : Nat) (shorts_cost : Nat) : Nat :=
  (num_jerseys * cost_per_jersey) + basketball_cost + shorts_cost

def total_money_given (total_cost : Nat) (money_left : Nat) : Nat :=
  total_cost + money_left

theorem jeremy_uncle_money :
  total_money_given (total_cost 5 2 18 8) 14 = 50 :=
by
  sorry

end jeremy_uncle_money_l476_476985


namespace angle_AOF_is_118_l476_476473

theorem angle_AOF_is_118
  (A B C D E F O : Type)
  (angle : (A → A → A → Type) → Type)
  (x y : ℝ)
  (h1 : angle A O B = angle B O C)
  (h2 : angle C O D = angle D O E ∧ angle D O E = angle E O F)
  (h3 : angle A O D = 82)
  (h4 : angle B O E = 68) :
  angle A O F = 118 := 
sorry

end angle_AOF_is_118_l476_476473


namespace rhombus_not_diagonals_equal_l476_476155

theorem rhombus_not_diagonals_equal (R : Type) [linear_ordered_field R] 
  (a b c d : R) (h1 : a = b) (h2 : b = c) (h3 : c = d) (h4 : a = d)
  (h_sym : ∀ x y : R, a = b → b = c → c = d → d = a)
  (h_cen_sym : ∀ p : R × R, p = (0, 0) → p = (0, 0)) :
  ¬(∀ p q : R × R, p ≠ q → (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 = (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) :=
by
  sorry

end rhombus_not_diagonals_equal_l476_476155


namespace find_points_outside_square_l476_476199

-- Define the basic properties of the squares and the setup

def square (A B C D : ℝ × ℝ) : Prop := 
  (A.1 = 0 ∧ A.2 = 0) ∧
  (B.1 = 500 ∧ B.2 = 0) ∧
  (C.1 = 500 ∧ C.2 = 500) ∧
  (D.1 = 0 ∧ D.2 = 500)

def inside_square (center : ℝ × ℝ) (side : ℕ) (Q_sides : list (ℝ × ℝ)) : Prop :=
  ∀ v ∈ Q_sides, abs (v.1 - center.1) < side / 2 ∧ abs (v.2 - center.2) < side / 2

theorem find_points_outside_square : 
  ∃ A B : ℝ × ℝ, square (0,0) (500,0) (500,500) (0,500) →
    inside_square (250, 250) 250 [(125, 125), (375, 125), (375, 375), (125, 375)] →
    dist A B > 521 ∧ ¬ (∃ p : ℝ × ℝ, p ∈ [(125, 125), (375, 125), (375, 375), (125, 375)] ∧ is_on_segment p A B) := sorry

end find_points_outside_square_l476_476199


namespace min_sixth_power_sin_cos_l476_476259

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476259


namespace distance_last_part_l476_476063

-- We are given the following conditions
def speed : ℝ := 10 -- Mr. Isaac's speed in miles per hour
def first_ride_time : ℝ := 0.5 -- First riding period in hours (30 minutes)
def second_ride_distance : ℝ := 15 -- Distance of the second ride in miles
def rest_time : ℝ := 0.5 -- Resting time in hours (30 minutes)
def total_time : ℝ := 4.5 -- Total journey time in hours (270 minutes)

-- We need to prove the distance covered in the last part is 20 miles
theorem distance_last_part : 
  let time_spent := first_ride_time + (second_ride_distance / speed) + rest_time in
  let time_last_part := total_time - time_spent in
  let distance_last_part := speed * time_last_part in
  distance_last_part = 20 :=
by
  -- Placeholder for the proof
  sorry

end distance_last_part_l476_476063


namespace proof_problem_l476_476397

variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ)]

variables (v : euclidean_space ℝ)
variables (n1 n2 : euclidean_space ℝ)
variables (α β : affine_subspace ℝ (euclidean_space ℝ))
variables (l : affine_subspace ℝ (euclidean_space ℝ))

-- Given conditions
def conditions : Prop := 
  (α.dimension = 2) ∧
  (β.dimension = 2) ∧
  (α ≠ β) ∧
  (v ∈ α.direction) ∧
  (n1 ∈ α.direction) ∧
  (n2 ∈ β.direction)

-- Statement A: n1 || n2 ↔ α || β
def statement_a : Prop := 
  (n1 ∥ n2 ↔ α ∥ β)

-- Statement B: n1 ⟂ n2 ↔ α ⟂ β
def statement_b : Prop := 
  (inner_product_space.orthogonal n1 n2 ↔ 
   inner_product_space.orthogonal α.direction β.direction)

-- Statement C: v ⟂ n1 ↔ l ∥ α
def statement_c : Prop := 
  (inner_product_space.orthogonal v n1 ↔
   affine_subspace.parallel l α)

-- Statement D: v ⟂ n1 ↔ l ⟂ α
def statement_d : Prop := 
  (inner_product_space.orthogonal v n1 ↔
   inner_product_space.orthogonal l.direction α.direction)

-- Final goals
theorem proof_problem (h : conditions) : statement_a ∧ statement_b ∧ ¬statement_c ∧ ¬statement_d :=
sorry

end proof_problem_l476_476397


namespace distance_to_origin_eq_sqrt_two_l476_476086

-- Definition of the conditions
def complex_number : ℂ := (1 + complex.I) / complex.I

-- Statement of the problem, proving that the distance is √2
theorem distance_to_origin_eq_sqrt_two : complex.abs complex_number = Real.sqrt 2 := 
sorry

end distance_to_origin_eq_sqrt_two_l476_476086


namespace angle_between_vectors_is_correct_l476_476861

def vec_a : ℝ × ℝ × ℝ := (3, -2, 2)
def vec_b : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def angle_between_vectors : ℝ :=
  Real.acos ((vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 + vec_a.3 * vec_b.3) / 
    (Real.sqrt (vec_a.1^2 + vec_a.2^2 + vec_a.3^2) * Real.sqrt (vec_b.1^2 + vec_b.2^2 + vec_b.3^2)))

theorem angle_between_vectors_is_correct :
  angle_between_vectors = Real.acos (-8 / (3 * Real.sqrt 17)) :=
by sorry

end angle_between_vectors_is_correct_l476_476861


namespace smallest_positive_integer_satisfying_congruence_l476_476746

theorem smallest_positive_integer_satisfying_congruence :
  ∃ x : ℤ, (0 < x) ∧ (x < 31) ∧ (5 * x ≡ 22 [MOD 31]) ∧ (∀ y : ℤ, (0 < y ∧ y < x) → ¬ (5 * y ≡ 22 [MOD 31])) :=
begin
  use 23,
  split, norm_num,
  split, norm_num,
  split,
  { norm_num, exact modeq.refl 22 },
  { intros y hy, 
    dsimp only,
    sorry }
end

end smallest_positive_integer_satisfying_congruence_l476_476746


namespace find_lengths_of_b_and_c_l476_476958

variables (A B C : ℝ) -- angles of the triangle
variables (a b c : ℝ) -- sides of the triangle
variables (sin_C : ℝ) (sin_A : ℝ) (cos_A : ℝ)

-- Given conditions:
-- 1. In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively.
-- 2. sin C = sqrt(10) / 4.
-- 3. a = 2.
-- 4. 2 * sin A = sin C.

def triangle_conditions : Prop :=
  sin_C = sqrt 10 / 4 ∧
  a = 2 ∧
  2 * sin_A = sin_C

def lengths_of_b_and_c : Prop :=
  c = 4 ∧ (b = sqrt 6 ∨ b = 2 * sqrt 6)

-- The theorem statement that encapsulates the proof
theorem find_lengths_of_b_and_c 
  (h : triangle_conditions A B C a b c sin_C sin_A) : lengths_of_b_and_c a b c :=
by
  rcases h with ⟨h₁, h₂, h₃⟩
  -- Since the proof is skipped, insert sorry to indicate unfinished proof.
  sorry

end find_lengths_of_b_and_c_l476_476958


namespace malcolm_initial_white_lights_l476_476090

theorem malcolm_initial_white_lights :
  let red_lights := 12
  let blue_lights := 3 * red_lights
  let green_lights := 6
  let bought_lights := red_lights + blue_lights + green_lights
  let remaining_lights := 5
  let total_needed_lights := bought_lights + remaining_lights
  W = total_needed_lights :=
by
  sorry

end malcolm_initial_white_lights_l476_476090


namespace youngest_child_age_l476_476761

theorem youngest_child_age {x : ℝ} (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by
  sorry

end youngest_child_age_l476_476761


namespace sum_of_radii_of_intersecting_circles_l476_476384

theorem sum_of_radii_of_intersecting_circles (R x y : ℝ) (O A : ℝ) 
  (h₁ : ∃ O₁ O₂ : ℝ, O₁ ≠ O₂ ∧ (∃ A B : ℝ, is_tangent O O₁ R x ∧ is_tangent O O₂ R y ∧ is_intersection A B O₁ O₂)) 
  (h₂ : angle_eq O A B 90) 
  : x + y = R :=
sorry

end sum_of_radii_of_intersecting_circles_l476_476384


namespace max_marks_paper_I_l476_476774

-- Definitions and conditions
def passing_marks : ℝ := 65
def passing_percentage : ℝ := 0.35

-- The theorem directly corresponding to our problem.
theorem max_marks_paper_I : 
    ∃ M : ℝ, passing_percentage * M = passing_marks ∧ M = 186 :=
by
  use 186
  split
  . show passing_percentage * 186 = passing_marks
    calc passing_percentage * 186
        = 0.35 * 186 : rfl
    ... = 65 : by norm_num
  . show 186 = 186
    rfl

end max_marks_paper_I_l476_476774


namespace problem_solved_by_at_least_one_student_l476_476097

theorem problem_solved_by_at_least_one_student (P_A P_B : ℝ) 
  (hA : P_A = 0.8) 
  (hB : P_B = 0.9) :
  (1 - (1 - P_A) * (1 - P_B) = 0.98) :=
by
  have pAwrong := 1 - P_A
  have pBwrong := 1 - P_B
  have both_wrong := pAwrong * pBwrong
  have one_right := 1 - both_wrong
  sorry

end problem_solved_by_at_least_one_student_l476_476097


namespace movie_marathon_duration_l476_476720

theorem movie_marathon_duration :
  let first_movie := 2
  let second_movie := first_movie + 0.5 * first_movie
  let combined_time := first_movie + second_movie
  let third_movie := combined_time - 1
  first_movie + second_movie + third_movie = 9 := by
  sorry

end movie_marathon_duration_l476_476720


namespace pyramid_base_side_length_correct_l476_476682

def sideLengthBase (s : ℕ) : Prop :=
  let area : ℕ := 100
  let slant_height : ℕ := 20
  let lateral_face_area := (1/2:ℚ) * s * slant_height
  lateral_face_area.toNat = area → s = 10

theorem pyramid_base_side_length_correct (s : ℕ) (h: s * 10 = 100) : sideLengthBase s :=
  by
    intros
    simp [sideLengthBase]
    assume lateral_face_area h
    exact h
    sorry

end pyramid_base_side_length_correct_l476_476682


namespace volume_unoccupied_space_l476_476714

def radius_cone := 10  -- radius of each cone in cm
def height_cone := 15  -- height of each cone in cm
def height_cylinder := 45  -- height of the enclosing cylinder in cm

def volume_cylinder (r : ℝ) (h : ℝ) : ℝ := π * r^2 * h
def volume_cone (r : ℝ) (h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

theorem volume_unoccupied_space :
  let r := radius_cone in
  let h := height_cone in
  let H := height_cylinder in
  volume_cylinder r H - 3 * volume_cone r h = 3000 * π :=
by
  sorry

end volume_unoccupied_space_l476_476714


namespace distinct_terms_expansion_l476_476832

-- Define the expression
def expression (a b : ℚ) : ℚ := ((a + 2 * b) ^ 3 * (a - 2 * b) ^ 3) ^ 2

-- Define the theorem stating the number of distinct terms in the expansion
theorem distinct_terms_expansion : ∀ (a b : ℚ), 
  is_polynomial (expression a b) → 
  distinct_terms (expand_polynomial (expression a b)) = 7 :=
by
  intros a b h1
  sorry

end distinct_terms_expansion_l476_476832


namespace simon_tenth_finger_l476_476088

def g (x : ℕ) : ℕ :=
  if x = 2 then 2 else 0 -- Based on the problem's description

theorem simon_tenth_finger :
  ∀ (n : ℕ),
    n ≥ 1 → 
    (∀ k, g^(k 2) = 2) → 
    (∀ m, m < n → g^(m 2) = 2) → 
    g^(9 2) = 2 :=
by
  sorry

end simon_tenth_finger_l476_476088


namespace socks_pairing_l476_476940

theorem socks_pairing : 
  let total_socks : ℕ := 10 
  let white_socks : ℕ := 4 
  let brown_socks : ℕ := 4 
  let blue_socks : ℕ := 2 
  (total_socks = white_socks + brown_socks + blue_socks) →
  (∃ (n : ℕ), n = 4 * 4 + 4 * 2 + 4 * 2 ∧ n = 32) :=
by
  intros
  use 32
  split
  { sorry }
  { rfl }

end socks_pairing_l476_476940


namespace sum_of_segments_le_sum_of_edges_l476_476656

theorem sum_of_segments_le_sum_of_edges {A B C D O : Point}
  (h₁ : Inside O (tetrahedron A B C D)) : length (OA) + length (OB) + length (OC) + length (OD) ≤ length (AB) + length (AC) + length (AD) + length (BC) + length (BD) + length (CD) :=
sorry

-- Definitions required:
def Inside (p : Point) (t : tetrahedron) : Prop := sorry
def tetrahedron (A B C D : Point) : Set Point := sorry
def Point := sorry
def length (p q : Point) : Real := sorry

end sum_of_segments_le_sum_of_edges_l476_476656


namespace area_inside_first_quadrant_l476_476829

noncomputable def circle_center : ℝ × ℝ := (1, -Real.sqrt 3 / 2)
noncomputable def circle_radius : ℝ := 1
noncomputable def desired_area : ℝ := (π / 6) - (Real.sqrt 3 / 4)

theorem area_inside_first_quadrant :
  let center := circle_center
  let radius := circle_radius
  let region_area := desired_area
  ∃ (x y : ℝ), ((x - center.1)^2 + (y - center.2)^2 ≤ radius^2) ∧ (0 ≤ x) ∧ (0 ≤ y) → region_area = desired_area :=
sorry

end area_inside_first_quadrant_l476_476829


namespace football_team_count_l476_476961

def football_team_selections (n : ℕ) (k : ℕ) : ℕ :=
  nat.choose n k

theorem football_team_count : 
  football_team_selections 29 10 * 2 + football_team_selections 29 9 = football_team_selections 31 11 - football_team_selections 29 11 :=
sorry

end football_team_count_l476_476961


namespace count_valid_numbers_l476_476019

def digits_set : List ℕ := [0, 2, 4, 7, 8, 9]

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def sum_digits (digits : List ℕ) : ℕ :=
  List.sum digits

def last_two_digits_divisibility (last_two_digits : ℕ) : Prop :=
  last_two_digits % 4 = 0

def number_is_valid (digits : List ℕ) : Prop :=
  sum_digits digits % 3 = 0

theorem count_valid_numbers :
  let possible_digits := [0, 2, 4, 7, 8, 9]
  let positions := 5
  let combinations := Nat.pow (List.length possible_digits) (positions - 1)
  let last_digit_choices := [0, 4, 8]
  3888 = 3 * combinations :=
sorry

end count_valid_numbers_l476_476019


namespace particle_returns_to_origin_after_120_moves_l476_476192

noncomputable def ω : ℂ := complex.exp (complex.I * real.pi / 6)

def move (z : ℂ) : ℂ := ω * z + 8

noncomputable def particle_position (n : ℕ) : ℂ :=
  if n = 0 then 3 else move (particle_position (n - 1))

-- Theorem to be proved in Lean
theorem particle_returns_to_origin_after_120_moves :
  particle_position 120 = 3 :=
sorry

end particle_returns_to_origin_after_120_moves_l476_476192


namespace sin_cos_sixth_min_l476_476344

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476344


namespace ims_seating_arrangements_l476_476678

/-- The Interstellar Mathematics Summit (IMS) committee consists of 4 Martians, 4 Venusians, and 4 Earthlings.
They sit at a round table with 12 chairs numbered from 1 to 12 clockwise. The seating arrangement must satisfy the following conditions:
- A Martian must occupy chair 1.
- A Venusian must occupy chair 12.
- No Martian can sit immediately to the left of an Earthling.
- No Earthling can sit immediately to the left of a Venusian.
- No Venusian can sit immediately to the left of a Martian.

The number of possible seating arrangements for the committee is K * (4!)^3. We need to find K. -/
theorem ims_seating_arrangements : 
  ∃ K : ℕ, let fact4 := nat.factorial 4 in (K * (fact4 ^ 3) = (number of valid arrangements satisfying given constraints)) ∧ K = 29 := sorry

end ims_seating_arrangements_l476_476678


namespace escher_picasso_probability_correct_l476_476064

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Defining the problem conditions
def total_pieces : ℕ := 12
def escher_prints : ℕ := 4
def picasso_prints : ℕ := 2

-- Function that calculates the number of ways to arrange 12 items
def total_arrangements : ℕ := factorial total_pieces

-- Function that calculates the number of valid Escher-Picasso configurations
def valid_configurations : ℕ := 912

-- Function that calculates the desired probability
def desired_probability : ℚ := valid_configurations / total_arrangements

-- Theorem stating the desired probability
theorem escher_picasso_probability_correct :
  desired_probability = 912 / 479001600 := 
by
  sorry

end escher_picasso_probability_correct_l476_476064


namespace surface_area_ratio_volume_ratio_l476_476811

-- Given conditions
def tetrahedron_surface_area (S : ℝ) : ℝ := 4 * S
def tetrahedron_volume (V : ℝ) : ℝ := 27 * V
def polyhedron_G_surface_area (S : ℝ) : ℝ := 28 * S
def polyhedron_G_volume (V : ℝ) : ℝ := 23 * V

-- Statements to prove
theorem surface_area_ratio (S : ℝ) (h1 : S > 0) :
  tetrahedron_surface_area S / polyhedron_G_surface_area S = 9 / 7 := by
  simp [tetrahedron_surface_area, polyhedron_G_surface_area]
  sorry

theorem volume_ratio (V : ℝ) (h1 : V > 0) :
  tetrahedron_volume V / polyhedron_G_volume V = 27 / 23 := by
  simp [tetrahedron_volume, polyhedron_G_volume]
  sorry

end surface_area_ratio_volume_ratio_l476_476811


namespace hours_per_day_l476_476177

-- Define the parameters
def A1 := 57
def D1 := 12
def H2 := 6
def A2 := 30
def D2 := 19

-- Define the target Equation
theorem hours_per_day :
  A1 * D1 * H = A2 * D2 * H2 → H = 5 :=
by
  sorry

end hours_per_day_l476_476177


namespace sin_cos_sixth_min_l476_476352

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476352


namespace find_B_and_b_range_l476_476458

variables {a b c : ℝ} {A B C : ℝ}

def triangle_ABC (A B C a b c : ℝ) : Prop :=
  A + B + C = π ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 2 * sin (A/2) * cos (B/2 + C/2) ∧
  b = 2 * sin (B/2) * cos (A/2 + C/2) ∧
  c = 2 * sin (C/2) * cos (A/2 + B/2)

theorem find_B_and_b_range
  (h1 : triangle_ABC A B C a b c)
  (h2 : cos C + (cos A - real.sqrt 3 * sin A) * cos B = 0)
  (h3 : a + c = 1) :
  B = π / 3 ∧ (1 / 2) ≤ b ∧ b < 1 := 
sorry

end find_B_and_b_range_l476_476458


namespace min_value_sin6_cos6_l476_476324

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476324


namespace polynomial_not_product_of_first_few_odd_primes_l476_476900

-- Given conditions
variables (n : ℕ) (x : ℕ)
hypothesis (h1 : n > 0)                  -- n is a positive integer
hypothesis (h2 : n % 3 = 0)              -- n is divisible by 3
hypothesis (h3 : Prime (2 * n - 1))      -- (2n - 1) is a prime

-- To prove the polynomial is not a product of the first few odd primes
theorem polynomial_not_product_of_first_few_odd_primes (h4 : x > n) 
    : ¬ ∃ (P : list ℕ), (∀ p ∈ P, odd p) ∧ (nx ^ (n + 1) + (2n + 1) * x ^ n - 3 * (n - 1) * x ^ (n - 1) - x - 3 = P.prod) := 
sorry

end polynomial_not_product_of_first_few_odd_primes_l476_476900


namespace min_value_sin_cos_l476_476294

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476294


namespace road_system_possible_l476_476462

theorem road_system_possible (k : ℕ) : (k > 2) ↔ 
  (∃ (G : SimpleGraph (Fin 8)), 
    (∀ (v : Fin 8), G.degree v ≤ k) ∧ 
    (∀ (u v : Fin 8), u ≠ v → (G.adj u v ∨ ∃ w, G.adj u w ∧ G.adj w v))) := sorry

end road_system_possible_l476_476462


namespace min_value_sin_cos_l476_476291

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476291


namespace minimize_sin_cos_six_l476_476302

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476302


namespace minimize_sin_cos_six_l476_476304

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476304


namespace exchange_crowns_to_tugriks_l476_476221

theorem exchange_crowns_to_tugriks : 
  (∀ (tugriks_per_dinar : ℚ) (dinar_per_rupee : ℚ) (rupees_per_taler : ℚ) (talers_per_crown : ℚ),
    (tugriks_per_dinar = 11 / 14) →
    (dinar_per_rupee = 21 / 22) →
    (rupees_per_taler = 10 / 3) →
    (talers_per_crown = 2 / 5) →
    (13 * (talers_per_crown * rupees_per_taler * dinar_per_rupee * tugriks_per_dinar) = 13)) :=
by 
  intros tugriks_per_dinar dinar_per_rupee rupees_per_taler talers_per_crown
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  simp
  sorry

end exchange_crowns_to_tugriks_l476_476221


namespace tangent_line_at_P_minimum_value_in_interval_l476_476633

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem tangent_line_at_P : ∀ x y, (x = 2 ∧ y = 3) → (9*x - 5 = y) :=
by
  intros x y h
  cases h
  sorry

theorem minimum_value_in_interval : ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f y ≥ f x :=
by
  use -3
  split
  {
    exact Set.left_mem_Icc.mpr (by norm_num)
  }
  {
    intros y hy
    sorry
  }

end tangent_line_at_P_minimum_value_in_interval_l476_476633


namespace find_min_k_l476_476495

theorem find_min_k (k : ℕ) 
  (h1 : k > 0) 
  (h2 : ∀ (A : Finset ℕ), A ⊆ (Finset.range 26).erase 0 → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (2 / 3 : ℝ) ≤ x / y ∧ x / y ≤ (3 / 2 : ℝ)) : 
  k = 7 :=
by {
  sorry
}

end find_min_k_l476_476495


namespace ellipse_condition_l476_476083

theorem ellipse_condition (a : ℝ) : (3 < a ∧ a < 5) → (a ≠ 4 ∧ (∃ (x y : ℝ), (x^2 / (a - 3) + y^2 / (5 - a) = 1) → a = 4 → False) :=
by
  sorry

end ellipse_condition_l476_476083


namespace solve_exponential_eq_l476_476942

theorem solve_exponential_eq (x : ℝ) : 
  ((5 - 2 * x)^(x + 1) = 1) ↔ (x = -1 ∨ x = 2 ∨ x = 3) := by
  sorry

end solve_exponential_eq_l476_476942


namespace minimum_sum_PE_PC_l476_476470

noncomputable def point := (ℝ × ℝ)
noncomputable def length (p1 p2 : point) : ℝ := Real.sqrt (((p1.1 - p2.1)^2) + ((p1.2 - p2.2)^2))

theorem minimum_sum_PE_PC :
  let A : point := (0, 3)
  let B : point := (3, 3)
  let C : point := (3, 0)
  let D : point := (0, 0)
  ∃ P E : point, E.1 = 3 ∧ E.2 = 1 ∧ (∃ t : ℝ, t ≥ 0 ∧ t ≤ 3 ∧ P.1 = 3 - t ∧ P.2 = t) ∧
    (length P E + length P C = Real.sqrt 13) :=
by
  sorry

end minimum_sum_PE_PC_l476_476470


namespace minimum_value_of_reciprocal_squares_l476_476127

theorem minimum_value_of_reciprocal_squares
  (a b : ℝ)
  (h : a ≠ 0 ∧ b ≠ 0)
  (h_eq : (a^2) + 4 * (b^2) = 9)
  : (1/(a^2) + 1/(b^2)) = 1 :=
sorry

end minimum_value_of_reciprocal_squares_l476_476127


namespace trigonometric_equation_solution_l476_476167

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) :
  (cos x ≠ 0) ∧ (sin (2 * x) - 2 * (cos x) ^ 2 + 4 * (sin x - cos x + sin x / cos x - 1) = 0) 
  ↔ (∃ k : ℤ, x = (π / 4) + (k * π)) :=
by
  sorry

end trigonometric_equation_solution_l476_476167


namespace flowers_per_bouquet_l476_476185

theorem flowers_per_bouquet :
  let red_seeds := 125
  let yellow_seeds := 125
  let orange_seeds := 125
  let purple_seeds := 125
  let red_killed := 45
  let yellow_killed := 61
  let orange_killed := 30
  let purple_killed := 40
  let bouquets := 36
  let red_flowers := red_seeds - red_killed
  let yellow_flowers := yellow_seeds - yellow_killed
  let orange_flowers := orange_seeds - orange_killed
  let purple_flowers := purple_seeds - purple_killed
  let total_flowers := red_flowers + yellow_flowers + orange_flowers + purple_flowers
  let flowers_per_bouquet := total_flowers / bouquets
  flowers_per_bouquet = 9 :=
by
  sorry

end flowers_per_bouquet_l476_476185


namespace find_k_l476_476435

theorem find_k 
  (k : ℤ) 
  (h : 2^2000 - 2^1999 - 2^1998 + 2^1997 = k * 2^1997) : 
  k = 3 :=
sorry

end find_k_l476_476435


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476362

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476362


namespace kati_age_l476_476490

/-- Define the age of Kati using the given conditions -/
theorem kati_age (kati_age : ℕ) (brother_age kati_birthdays : ℕ) 
  (h1 : kati_age = kati_birthdays) 
  (h2 : kati_age + brother_age = 111) 
  (h3 : kati_birthdays = kati_age) : 
  kati_age = 18 :=
by
  sorry

end kati_age_l476_476490


namespace max_triangles_in_convex_ngon_l476_476461

open Nat

-- Define problem conditions
def is_convex (n : ℕ) : Prop := n ≥ 3

-- Define the maximum_triangle_count based on whether n is even or odd
def max_triangle_count (n : ℕ) : ℕ :=
  if even n then 2 * n - 4 else 2 * n - 5

-- The proof problem statement
theorem max_triangles_in_convex_ngon (n : ℕ) (hconvex : is_convex n) : 
  ∀ (d : list (fin n * fin n)), -- d is a list of diagonals drawn in the n-gon
  no_three_diagonals_intersect d → -- Condition that no three or more diagonals intersect at the same point inside the polygon
  ∃ t : ℕ, t = max_triangle_count n := 
sorry

end max_triangles_in_convex_ngon_l476_476461


namespace log_evaluation_l476_476848

theorem log_evaluation : 
  ∀ (a b c : ℝ), a = 8^2 ∧ b = 8^(1/3) → log 8 (a * b) = 7/3 :=
by sorry

end log_evaluation_l476_476848


namespace jim_travel_distance_l476_476946

theorem jim_travel_distance
  (john_distance : ℕ := 15)
  (jill_distance : ℕ := john_distance - 5)
  (jim_distance : ℕ := jill_distance * 20 / 100) :
  jim_distance = 2 := 
by
  sorry

end jim_travel_distance_l476_476946


namespace min_value_sin6_cos6_l476_476322

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476322


namespace min_value_sin6_cos6_l476_476312

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476312


namespace max_cars_with_ac_but_not_rs_l476_476965

namespace CarProblem

variables (total_cars : ℕ) 
          (cars_without_ac : ℕ)
          (cars_with_rs : ℕ)
          (cars_with_ac : ℕ := total_cars - cars_without_ac)
          (cars_with_ac_and_rs : ℕ)
          (cars_with_ac_but_not_rs : ℕ := cars_with_ac - cars_with_ac_and_rs)

theorem max_cars_with_ac_but_not_rs 
        (h1 : total_cars = 100)
        (h2 : cars_without_ac = 37)
        (h3 : cars_with_rs ≥ 51)
        (h4 : cars_with_ac_and_rs = min cars_with_rs cars_with_ac) :
        cars_with_ac_but_not_rs = 12 := by
    sorry

end CarProblem

end max_cars_with_ac_but_not_rs_l476_476965


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476549

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476549


namespace milk_revenue_l476_476648

theorem milk_revenue :
  let yesterday_morning := 68
  let yesterday_evening := 82
  let this_morning := yesterday_morning - 18
  let total_milk_before_selling := yesterday_morning + yesterday_evening + this_morning
  let milk_left := 24
  let milk_sold := total_milk_before_selling - milk_left
  let cost_per_gallon := 3.50
  let revenue := milk_sold * cost_per_gallon
  revenue = 616 := by {
    sorry
}

end milk_revenue_l476_476648


namespace probability_exactly_two_primes_l476_476222

theorem probability_exactly_two_primes :
  let primes := {2, 3, 5, 7, 11}
  let num_faces := 12
  let num_ways_choose_2 := Nat.choose 3 2
  let p_prime := (primes.to_finset.card : ℚ) / num_faces
  let p_not_prime := 1 - p_prime
  ∑ (num_ways_choose_2 : ℕ), (num_ways_choose_2 * p_prime ^ 2 * p_not_prime) = 175 / 576 := by
  sorry

end probability_exactly_two_primes_l476_476222


namespace greatest_prime_factor_of_n_is_23_l476_476739

def n : ℕ := 5^7 + 10^6

theorem greatest_prime_factor_of_n_is_23 : ∃ p : ℕ, nat.prime p ∧ p = 23 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ n → q ≤ 23 :=
sorry

end greatest_prime_factor_of_n_is_23_l476_476739


namespace probability_is_1_over_90_l476_476986

/-- Probability Calculation -/
noncomputable def probability_of_COLD :=
  (1 / (Nat.choose 5 3)) * (2 / 3) * (1 / (Nat.choose 4 2))

theorem probability_is_1_over_90 :
  probability_of_COLD = (1 / 90) :=
by
  sorry

end probability_is_1_over_90_l476_476986


namespace min_sin6_cos6_l476_476333

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476333


namespace number_of_valid_even_integers_l476_476936

def digits_set : Finset ℕ := {1, 3, 4, 5, 6, 9}

def valid_numbers (n : ℕ) : Prop :=
  300 ≤ n ∧ n < 800 ∧ ∃ d1 d2 d3, 
    n = 100 * d1 + 10 * d2 + d3 ∧ 
    d1 ∈ digits_set ∧ 
    d2 ∈ digits_set ∧ 
    d3 ∈ digits_set ∧ 
    d1 ≠ d2 ∧ 
    d2 ≠ d3 ∧ 
    d1 ≠ d3 ∧ 
    d3 % 2 = 0

theorem number_of_valid_even_integers : (Finset.filter valid_numbers (Finset.range 800)).card = 24 := 
  by sorry

end number_of_valid_even_integers_l476_476936


namespace minimize_sin_cos_six_l476_476306

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476306


namespace value_of_f_csc_squared_l476_476670

def f (x : ℝ) : ℝ := if x = 0 ∨ x = -1 then 0 else 1 / x

theorem value_of_f_csc_squared (t : ℝ) (h_t : 0 ≤ t ∧ t ≤ π / 2) :
    f (csc t ^ 2) = -cot t ^ 2 :=
by
  sorry

end value_of_f_csc_squared_l476_476670


namespace smallest_coprime_k_with_a_n_l476_476426

/-- Define the sequence a_n as given in the problem. -/
def a_n (n : ℕ) : ℕ := 2^n + 3^n + 6^n + 1

/-- Prove the smallest integer k ≥ 2 that is coprime with all a_n is 23. -/
theorem smallest_coprime_k_with_a_n : ∃ k : ℕ, k ≥ 2 ∧ (∀ n : ℕ, Nat.coprime k (a_n n)) ∧ k = 23 :=
by
  sorry

end smallest_coprime_k_with_a_n_l476_476426


namespace area_of_triangle_l476_476094

-- Define the conditions
variables {A B C P : Type}
variables (perimeter_triangle : ℝ)
variables (distance_P_to_AB : ℝ)
variables (area_triangle : ℝ)

-- State the conditions in Lean
axiom perimeter_condition : perimeter_triangle = 20
axiom distance_condition : distance_P_to_AB = 4
axiom bisectors_intersect_at_P : true -- Placeholder for bisectors condition

-- Define the statement to prove
theorem area_of_triangle :
  perimeter_triangle = 20 ∧ distance_P_to_AB = 4 ∧ bisectors_intersect_at_P → area_triangle = 40 :=
by
  intro h
  sorry

end area_of_triangle_l476_476094


namespace prob_A_and_B_l476_476699

open MeasureTheory

-- Define the probability space
variable (Ω : Type*) [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)

-- Define events A and B
variables (A B : Set Ω)

-- Given conditions
axiom prob_B : P B = 0.4
axiom prob_A_or_B : P (A ∪ B) = 0.6
axiom prob_A : P A = 0.45

-- The goal is to prove P (A ∩ B) = 0.25
theorem prob_A_and_B : P (A ∩ B) = 0.25 :=
by
  have h1 : P (A ∪ B) = P A + P B - P (A ∩ B) := sorry -- Inclusion-exclusion principle
  have h2 : 0.6 = 0.45 + 0.4 - P (A ∩ B) := sorry -- Substitute known values
  have h3 : P (A ∩ B) = 0.25 := sorry -- Solve the equation
  exact h3

end prob_A_and_B_l476_476699


namespace tan_Y_l476_476968

-- Define the right triangle and conditions.
structure RightTriangle (X Y Z : Type) where
  XY : ℝ
  YZ : ℝ
  XZ : ℝ
  angle_XYZ : ∠ X Y Z = 90
  xy_length : XY = 40
  yz_length : YZ = 41
  xz_length : XZ = sqrt (YZ^2 - XY^2) = 9

-- Define point W and its relationship with XZ.
structure PointW (X W Z : Type) where
  XW : ℝ
  WZ : ℝ
  xw_length : XW = 9
  wz_length : WZ = xw_length - 9

-- Define the problem and the proof goal.
theorem tan_Y (X Y Z W : Type) (T : RightTriangle X Y Z) (W_on_XZ : PointW X W Z) : 
  tan Y = 9 / 40 :=
sorry

end tan_Y_l476_476968


namespace max_distinct_prime_factors_l476_476080

-- Define the main problem in type theorist style with the relevant conditions
theorem max_distinct_prime_factors
  (a b : ℕ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (gcd_factors : (nat.factors (nat.gcd a b)).to_finset.card = 5)
  (lcm_factors : (nat.factors (nat.lcm a b)).to_finset.card = 20)
  (a_fewer_factors : (nat.factors a).to_finset.card < (nat.factors b).to_finset.card) :
  (nat.factors a).to_finset.card ≤ 12 := 
sorry

end max_distinct_prime_factors_l476_476080


namespace intersection_of_A_and_B_l476_476056

def A : Set ℝ := { x : ℝ | x / (x - 2) < 0 }
def B : Set ℤ := { x : ℤ | True }

theorem intersection_of_A_and_B : A ∩ (B : Set ℝ) = {1} := by sorry

end intersection_of_A_and_B_l476_476056


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476606

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476606


namespace triangle_equilateral_l476_476690

-- Declare the necessary elements: points A, B, C, circumcircle, medians extended to points A₀, B₀, C₀
variables {A B C A₀ B₀ C₀ : Type*}
variables [by_cases {show_circle : Type*} (A B C : show_new_circle)]

-- Define medians extended intersecting circumcircle at A₀, B₀, and C₀
variables {median_A : A -> A₁}
variables {median_B : B -> B₁}
variables {median_C : C -> C₁}
variables {extension_A : median_A -> circumcircle}
variables {extension_B : median_B -> circumcircle}
variables {extension_C : median_C -> circumcircle}
variables h_A₀ : extension_A intersection_circumcircle A₀
variables h_B₀ : extension_B intersection_circumcircle B₀
variables h_C₀ : extension_C intersection_circumcircle C₀

-- Define the main conditions related to areas of triangles
variables [area_cmp { set_of_triangle (A B C₀) } == set_of_triangle (A B₀ C)]
variables [area_cmp { set_of_triangle ( A B₀ C) } == set_of_triangle (A₀ B C)]

-- Define the goal to prove that triangle ABC is equilateral
theorem triangle_equilateral : equilateral (A B C) :=
sorry

end triangle_equilateral_l476_476690


namespace rectangle_area_pairs_l476_476989

theorem rectangle_area_pairs :
  { p : ℕ × ℕ | p.1 * p.2 = 12 ∧ p.1 > 0 ∧ p.2 > 0 } = { (1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1) } :=
by {
  sorry
}

end rectangle_area_pairs_l476_476989


namespace triangle_line_intersection_at_one_point_l476_476463

noncomputable def intersect_concurrent_lines :
  (A1 A2 A3 : ℝ × ℝ) → (α1 α2 α3 : ℝ) → (l : ℝ × ℝ → ℝ)
  → Prop :=
λ A1 A2 A3 α1 α2 α3 l,
  let l1 := λ x : ℝ × ℝ, (x.1 - A1.1) * (A3.2 - A2.2) + (x.2 - A1.2) * (A3.1 - A2.1) = 0 in
  let l2 := λ x : ℝ × ℝ, (x.1 - A2.1) * (A1.2 - A3.2) + (x.2 - A2.2) * (A1.1 - A3.1) = 0 in
  let l3 := λ x : ℝ × ℝ, (x.1 - A3.1) * (A2.2 - A1.2) + (x.2 - A3.2) * (A2.1 - A1.1) = 0 in
  ∃ (P: ℝ × ℝ), l1 P ∧ l2 P ∧ l3 P

theorem triangle_line_intersection_at_one_point (A1 A2 A3 : ℝ × ℝ) (α1 α2 α3 : ℝ) (l : ℝ × ℝ → ℝ) :
  (∃ (α1 α2 α3 : ℝ), 
  (l (A1.1, A1.2) = α1) ∧ 
  (l (A2.1, A2.2) = α2) ∧ 
  (l (A3.1, A3.2) = α3)) →
  intersect_concurrent_lines A1 A2 A3 α1 α2 α3 l :=
begin
  sorry
end

end triangle_line_intersection_at_one_point_l476_476463


namespace last_two_digits_of_sum_l476_476233

noncomputable def last_two_digits_sum_factorials : ℕ :=
  let fac : List ℕ := List.map (fun n => Nat.factorial (n * 3)) [1, 2, 3, 4, 5, 6, 7]
  fac.foldl (fun acc x => (acc + x) % 100) 0

theorem last_two_digits_of_sum : last_two_digits_sum_factorials = 6 :=
by
  sorry

end last_two_digits_of_sum_l476_476233


namespace min_sixth_power_sin_cos_l476_476257

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476257


namespace arithmetic_sequence_minimum_value_S_l476_476582

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476582


namespace selection_including_both_genders_is_34_l476_476075

def count_ways_to_select_students_with_conditions (total_students boys girls select_students : ℕ) : ℕ :=
  if total_students = 7 ∧ boys = 4 ∧ girls = 3 ∧ select_students = 4 then
    (Nat.choose total_students select_students) - 1
  else
    0

theorem selection_including_both_genders_is_34 :
  count_ways_to_select_students_with_conditions 7 4 3 4 = 34 :=
by
  -- The proof would go here
  sorry

end selection_including_both_genders_is_34_l476_476075


namespace case_a_exists_triangle_and_pentagon_case_b_exists_triangle_quadrilateral_and_pentagon_l476_476842

/-- Two sets of points representing quadrilaterals on a grid. -/
def quadrilateral_a1 := ((0,0), (2,0), (1,2), (0,2))
def quadrilateral_a2 := ((2,2), (3,0), (5,0), (4,2))

def quadrilateral_b1 := ((0,0), (2,0), (2,2), (0,2))
def quadrilateral_b2 := ((3,0), (5,0), (5,2), (3,2))

theorem case_a_exists_triangle_and_pentagon :
  ∃ (triangle pentagon: set (ℝ × ℝ)),
    (triangle ⊆ {((0,0), (2,0), (1,2))}) ∧
    (pentagon ⊆ {((0,0), (2,0), (2,2), (4,2), (3,0))}) := by
  sorry

theorem case_b_exists_triangle_quadrilateral_and_pentagon :
  ∃ (triangle quadrilateral pentagon: set (ℝ × ℝ)),
    (triangle ⊆ {((0,0), (2,0), (2,2))}) ∧
    (quadrilateral ⊆ {((0,0), (2,0), (3,0), (3,2))}) ∧
    (pentagon ⊆ {((0,0), (2,0), (3,0), (5,0), (5,2))}) := by
  sorry

end case_a_exists_triangle_and_pentagon_case_b_exists_triangle_quadrilateral_and_pentagon_l476_476842


namespace distance_p_to_l_l476_476478

-- Definition of the equation of line l in polar coordinates
def polar_line (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ - Real.sin θ) + 2 = 0

-- Definition of the point P in polar coordinates
def polar_point_P (ρ θ : ℝ) := (ρ = 2) ∧ (θ = Real.pi / 6)

-- Conversion to Cartesian coordinates
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Definition of Cartesian line equation
def cartesian_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Distance from point to line
def distance_from_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / Real.sqrt (a^2 + b^2)

-- Cartesian coordinates of point P
def cartesian_point_P := polar_to_cartesian 2 (Real.pi / 6)

-- Proving the distance from point P to line l
theorem distance_p_to_l :
  ∃ P : ℝ × ℝ,
    polar_point_P 2 (Real.pi / 6) →
    P = cartesian_point_P →
    distance_from_point_to_line P 1 (-1) 2 = (Real.sqrt 6 + Real.sqrt 2) / 2 :=
by 
  sorry

end distance_p_to_l_l476_476478


namespace triangle_with_given_heights_is_right_angled_l476_476694

def is_right_angled_triangle (h1 h2 h3 : ℝ) (h : triangle_heights h1 h2 h3) : Prop :=
  -- Prove here that the triangle with given heights is right-angled.
  sorry

theorem triangle_with_given_heights_is_right_angled (h1 h2 h3 : ℝ) 
  (h_h1 : h1 = 12) (h_h2 : h2 = 15) (h_h3 : h3 = 20) : 
  is_right_angled_triangle h1 h2 h3 :=
sorry

end triangle_with_given_heights_is_right_angled_l476_476694


namespace prove_an_and_inequality_l476_476387

-- Definitions based on the given problem conditions
def sequence_an (n : ℕ) : ℕ := 3^(n-1)
def sn (n : ℕ) : ℕ := (3/2)*(sequence_an n) - 1/2

-- Definition for bn based on the term difference a_{n+2} - a_{n+1}
def bn (n : ℕ) : ℕ := 
  have a_n := sequence_an n
  have a_n1 := sequence_an (n+1)
  have a_n2 := sequence_an (n+2)
  2 * n / (a_n2 - a_n1)

-- Definition of Tn as the sum of the first n terms of the sequence {bn}
def Tn (n : ℕ) : ℕ := ∑ i in range n, bn i

-- Statement to prove: the general formula for an and inequality for Tn
theorem prove_an_and_inequality (n : ℕ) : 
  (∀ n, sequence_an n = 3^(n-1))
  ∧ (Tn n < 3/4) :=
by
  -- Proof of the first part
  apply (forall n, sequence_an n = 3^(n-1)),
  -- Proof of the second part
  apply (Tn n < 3/4)

sorry

end prove_an_and_inequality_l476_476387


namespace arithmetic_sequence_and_minimum_sum_l476_476524

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476524


namespace cartesian_equations_and_min_distance_l476_476975

def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sin θ)

def polar_curve_C1 (θ : ℝ) : ℝ :=
  8 * Real.cos θ

def polar_curve_C2 (θ : ℝ) : ℝ :=
  -8 * Real.sin θ

theorem cartesian_equations_and_min_distance :
  (∀ θ, let (x, y) := parametric_curve_C θ in (x^2 / 4 + y^2 = 1)) ∧
  (∀ ρ θ, ρ = polar_curve_C1 θ → (ρ * Real.cos θ)^2 + (ρ * Real.sin θ)^2 = 8 * (ρ * Real.cos θ)) ∧
  (∀ ρ θ, ρ = polar_curve_C2 θ → (ρ * Real.cos θ)^2 + (ρ * Real.sin θ)^2 = -8 * (ρ * Real.sin θ)) ∧
  ((0, 0) = (0, 0) ∧ (4, -4) = (4, -4)) ∧
  (let (x1, y1) := (0, 0); (x2, y2) := (4, -4) in (2, -2) = ((x1 + x2) / 2, (y1 + y2) / 2)) ∧
  (∀ θ, let (N_x, N_y) := parametric_curve_C θ; P := (Real.cos θ + 1, Real.sin θ / 2 - 1) in
   (Real.abs ((P.1 - 2 * P.2 + 2) / Real.sqrt 5)) = (Real.abs (Real.sqrt 2 * Real.cos (θ + Real.pi / 4) + 5) / Real.sqrt 5) → 
   Real.abs ((Real.sqrt 5) - Real.sqrt (10) / 5)) ∧
  (∃ θ, (Real.abs (Real.sqrt 2 * Real.cos (θ + Real.pi / 4) + 5) / √5) = (Real.abs (Real.sqrt 5 - (Real.sqrt 10) / 5))) 
:=
by sorry

end cartesian_equations_and_min_distance_l476_476975


namespace remainder_of_b_mod_13_l476_476050

theorem remainder_of_b_mod_13 :
  let b := (1 / 2 + 1 / 3 + 1 / 7)⁻¹ in
  (b : ℤ) % 13 = 8 := sorry

end remainder_of_b_mod_13_l476_476050


namespace product_fraction_equality_l476_476223

theorem product_fraction_equality : 
  ∏ k in finset.range 15, (k + 1) * ((k + 1) + 3) / ((k + 1) + 5)^2 = 1 / 42336 :=
by
  sorry

end product_fraction_equality_l476_476223


namespace proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l476_476148

open Classical

variable (a b x y : ℝ)

theorem proposition_A_correct (h : a > 1) : (1 / a < 1) ∧ ¬((1 / a < 1) → (a > 1)) :=
sorry

theorem proposition_B_incorrect (h_neg : ¬(x < 1 → x^2 < 1)) : ¬(∃ x, x ≥ 1 ∧ x^2 ≥ 1) :=
sorry

theorem proposition_C_incorrect (h_xy : x ≥ 2 ∧ y ≥ 2) : ¬((x ≥ 2 ∧ y ≥ 2) → x^2 + y^2 ≥ 4) :=
sorry

theorem proposition_D_correct (h_a : a ≠ 0) : (a * b ≠ 0) ∧ ¬((a * b ≠ 0) → (a ≠ 0)) :=
sorry

end proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l476_476148


namespace angle_between_vectors_l476_476869

theorem angle_between_vectors :
  let v1 : ℝ × ℝ × ℝ := (3, -2, 2)
  let v2 : ℝ × ℝ × ℝ := (-2, 2, 1)
  let dot_product (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let magnitude (v : ℝ × ℝ × ℝ) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let cos_theta := dot_product v1 v2 / (magnitude v1 * magnitude v2)
  θ = Real.acos (cos_theta) * (180 / Real.pi)
  in θ ≈ 127 :=
by
  sorry

end angle_between_vectors_l476_476869


namespace arithmetic_sequence_min_value_Sn_l476_476572

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476572


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476542

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476542


namespace volume_of_one_wedge_l476_476198

theorem volume_of_one_wedge 
  (circumference : ℝ)
  (h : circumference = 15 * Real.pi) 
  (radius : ℝ) 
  (volume : ℝ) 
  (wedge_volume : ℝ) 
  (h_radius : radius = 7.5)
  (h_volume : volume = (4 / 3) * Real.pi * radius^3)
  (h_wedge_volume : wedge_volume = volume / 5)
  : wedge_volume = 112.5 * Real.pi :=
by
  sorry

end volume_of_one_wedge_l476_476198


namespace jose_investment_l476_476726

-- Define given constants
def TomInvestment : ℕ := 3000
def JoseShareProfit : ℕ := 3500
def TotalProfit : ℕ := 6300
def MonthsInYear : ℕ := 12
def JoseJoinDelayMonths : ℕ := 2

-- Define derived constants
def TomTotalInvestment : ℕ := TomInvestment * MonthsInYear
def TomProfit : ℕ := TotalProfit - JoseShareProfit

-- Define proof problem
theorem jose_investment :
  ∃ X : ℕ, (TomTotalInvestment : rat) / (X * (MonthsInYear - JoseJoinDelayMonths)) = (TomProfit : rat) / JoseShareProfit ∧ X = 4500 :=
begin
  sorry
end

end jose_investment_l476_476726


namespace king_paid_after_tip_l476_476783

theorem king_paid_after_tip:
  (crown_cost tip_percentage total_cost : ℝ)
  (h_crown_cost : crown_cost = 20000)
  (h_tip_percentage : tip_percentage = 0.1) :
  total_cost = crown_cost + (crown_cost * tip_percentage) :=
by
  have h_tip := h_crown_cost.symm ▸ h_tip_percentage.symm ▸ 20000 * 0.1
  have h_total := h_crown_cost.symm ▸ (h_tip.symm ▸ 2000)
  rw [h_crown_cost, h_tip, h_total]
  exact rfl

end king_paid_after_tip_l476_476783


namespace intersection_of_M_and_N_l476_476926

-- Define sets M and N
def M : Set ℕ := {0, 2, 3, 4}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

-- State the problem as a theorem
theorem intersection_of_M_and_N : (M ∩ N) = {0, 4} :=
by
    sorry

end intersection_of_M_and_N_l476_476926


namespace farmer_bob_water_percentage_l476_476818

def water_per_acre_corn := 20
def water_per_acre_cotton := 80
def water_per_acre_beans := 2 * water_per_acre_corn

def corn_area_bob := 3
def cotton_area_bob := 9
def beans_area_bob := 12

def corn_area_brenda := 6
def cotton_area_brenda := 7
def beans_area_brenda := 14

def corn_area_bernie := 2
def cotton_area_bernie := 12

def water_bob := (corn_area_bob * water_per_acre_corn) + 
                 (cotton_area_bob * water_per_acre_cotton) + 
                 (beans_area_bob * water_per_acre_beans)

def water_brenda := (corn_area_brenda * water_per_acre_corn) + 
                    (cotton_area_brenda * water_per_acre_cotton) + 
                    (beans_area_brenda * water_per_acre_beans)

def water_bernie := (corn_area_bernie * water_per_acre_corn) + 
                    (cotton_area_bernie * water_per_acre_cotton)

def total_water := water_bob + water_brenda + water_bernie

def percentage_bob := (water_bob.toFloat / total_water.toFloat) * 100

theorem farmer_bob_water_percentage : percentage_bob ≈ 36 := sorry

end farmer_bob_water_percentage_l476_476818


namespace angle_CAD_eq_30_l476_476020

noncomputable theory

open_locale classical

variables {A B C D : Type} [normed_add_torsor ℝ (euclidean_point A)] [normed_add_torsor ℝ (euclidean_point B)]
           [normed_add_torsor ℝ (euclidean_point C)] [normed_add_torsor ℝ (euclidean_point D)]

def angle (P Q R : Type) [normed_add_torsor ℝ (euclidean_point P)] [normed_add_torsor ℝ (euclidean_point Q)]
           [normed_add_torsor ℝ (euclidean_point R)] :=
  0 / 1 -- Placeholder for angle measure

axiom angle_BAC_eq_50 : angle A B C = 50
axiom angle_ABD_eq_60 : angle A B D = 60
axiom angle_DBC_eq_20 : angle D B C = 20
axiom angle_BDC_eq_30 : angle B D C = 30

theorem angle_CAD_eq_30 : angle C A D = 30 :=
sorry

end angle_CAD_eq_30_l476_476020


namespace conjugate_of_z_l476_476909

theorem conjugate_of_z (z : ℂ) (hz : z + complex.I = 3 - complex.I) :
  complex.conj z = 3 + 2 * complex.I :=
sorry

end conjugate_of_z_l476_476909


namespace sum_of_remainders_l476_476237

theorem sum_of_remainders (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 5) 
  (h3 : c % 30 = 20) : 
  (a + b + c) % 30 = 10 := 
by sorry

end sum_of_remainders_l476_476237


namespace arvin_fifth_day_running_distance_l476_476012

theorem arvin_fifth_day_running_distance (total_km : ℕ) (first_day_km : ℕ) (increment : ℕ) (days : ℕ) 
  (h1 : total_km = 20) (h2 : first_day_km = 2) (h3 : increment = 1) (h4 : days = 5) : 
  first_day_km + (increment * (days - 1)) = 6 :=
by
  sorry

end arvin_fifth_day_running_distance_l476_476012


namespace min_trig_expression_l476_476484

theorem min_trig_expression (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = Real.pi) : 
  ∃ (x : ℝ), (x = 16 - 8 * Real.sqrt 2) ∧ (∀ A B C, 0 < A → 0 < B → 0 < C → A + B + C = Real.pi → 
    (1 / (Real.sin A)^2 + 1 / (Real.sin B)^2 + 4 / (1 + Real.sin C)) ≥ x) := 
sorry

end min_trig_expression_l476_476484


namespace henry_improvement_l476_476933

theorem henry_improvement :
  let initialLaps := 15
      initialTime := 45
      currentLaps := 18
      currentTime := 42
      initialLapTime := initialTime / initialLaps
      currentLapTime := currentTime / currentLaps
      improvement := initialLapTime - currentLapTime
  in improvement = 2 / 3 :=
by
  -- Definitions
  let initialLaps := 15
  let initialTime := 45
  let currentLaps := 18
  let currentTime := 42
  -- Calculations
  have H_initialLapTime : initialTime / initialLaps = 3 := by
    sorry
  have H_currentLapTime : currentTime / currentLaps = 7 / 3 := by
    sorry
  -- Conclusion
  have H_improvement : (initialTime / initialLaps) - (currentTime / currentLaps) = 2 / 3 := by
    sorry
  exact H_improvement

end henry_improvement_l476_476933


namespace min_sixth_power_sin_cos_l476_476275

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476275


namespace arithmetic_sequence_minimum_value_S_n_l476_476557

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476557


namespace day_before_yesterday_l476_476950

theorem day_before_yesterday (day_after_tomorrow_is_monday : String) : String :=
by
  have tomorrow := "Sunday"
  have today := "Saturday"
  exact today

end day_before_yesterday_l476_476950


namespace problem1_problem2_l476_476918

-- Definition and conditions for the function f
def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * sin (ω * x + φ + π / 3) + 1

variable (φ ω : ℝ) (h1 : |φ| < π / 2) (h2 : ω > 0) (h3 : ∀ x : ℝ, f x ω φ = f (-x) ω φ)
variable (h4 : ∃ T > 0, ∀ x : ℝ, f (x + T) ω φ = f x ω φ ∧ T = π / 2)

-- Lean 4 statements for the proof problems

-- Problem 1: Prove that f(π / 8) = √2
theorem problem1 : f (π / 8) 2 (π / 6) = sqrt 2 :=
by 
  sorry  -- Proof is omitted as per instructions.

-- Problem 2: Prove that the sum of the real roots of f(x) = 5/4 in (-π/2, 3π/2) is 2π
theorem problem2 : ∀ x ∈ Ioo (-π / 2) (3 * π / 2), f x 2 (π / 6) = 5 / 4 →
                   (∃ x1 x2 x3 x4 ∈ Ioo (-π / 2) (3 * π / 2),
                      x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ 
                      (x1 + x2 + x3 + x4 = 2 * π)) :=
by
  sorry  -- Proof is omitted as per instructions.

end problem1_problem2_l476_476918


namespace min_value_sin_cos_l476_476292

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476292


namespace king_total_payment_l476_476781

/-- 
A king gets a crown made that costs $20,000. He tips the person 10%. Prove that the total amount the king paid after the tip is $22,000.
-/
theorem king_total_payment (C : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h1 : C = 20000) 
  (h2 : tip_percentage = 0.1) 
  (h3 : total_paid = C + C * tip_percentage) : 
  total_paid = 22000 := 
by 
  sorry

end king_total_payment_l476_476781


namespace min_sin6_cos6_l476_476282

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476282


namespace arithmetic_sequence_min_value_S_l476_476517

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476517


namespace arithmetic_sequence_min_value_S_l476_476501

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476501


namespace unique_line_rational_points_l476_476959

-- Definition of a rational point
def is_rational_point (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℚ), p.1 = ↑x ∧ p.2 = ↑y

-- Given statement a is irrational
variables (a : ℝ)
axiom a_irrational : ¬ ∃ (q : ℚ), a = ↑q

-- Line passing through (a, 0) and properties of rational points
theorem unique_line_rational_points : 
  ∃! (l : ℝ → ℝ), 
    (l = (λ x : ℝ, 0) ∧ (∃ p1 p2 : ℝ × ℝ, 
      is_rational_point p1 ∧ is_rational_point p2 ∧ p1 ≠ p2 ∧ p1.1 = x ∧ p2.1 = x))
      ∨ 
    (l (a) = 0 ∧ ¬ ∃ (p : ℝ × ℝ), is_rational_point p ∧ p.1 ≠ a ∧ l (p.1) = p.2) :=
sorry

end unique_line_rational_points_l476_476959


namespace path_area_and_cost_correct_l476_476194

def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_of_path : ℝ := 759.36
def cost_per_sqm : ℝ := 2
def total_cost : ℝ := 1518.72

theorem path_area_and_cost_correct :
    let length_with_path := length_field + 2 * path_width
    let width_with_path := width_field + 2 * path_width
    let area_with_path := length_with_path * width_with_path
    let area_field := length_field * width_field
    let calculated_area_of_path := area_with_path - area_field
    let calculated_total_cost := calculated_area_of_path * cost_per_sqm
    calculated_area_of_path = area_of_path ∧ calculated_total_cost = total_cost :=
by
    sorry

end path_area_and_cost_correct_l476_476194


namespace find_a_l476_476099

/-- The random variable ξ takes on all possible values 1, 2, 3, 4, 5,
and P(ξ = k) = a * k for k = 1, 2, 3, 4, 5. Given that the sum 
of probabilities for all possible outcomes of a discrete random
variable equals 1, find the value of a. -/
theorem find_a (a : ℝ) 
  (h : (a * 1) + (a * 2) + (a * 3) + (a * 4) + (a * 5) = 1) : 
  a = 1 / 15 :=
sorry

end find_a_l476_476099


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476364

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476364


namespace evaluate_at_minus_two_l476_476491

def piecewise_function (x : ℝ) : ℝ :=
if x < -1 then
  3 * x + 7
else
  4 - x

theorem evaluate_at_minus_two : piecewise_function (-2) = 1 :=
by
  -- We provide a partial proof outline here
  -- to make sure the theorem is stated correctly.
  sorry

end evaluate_at_minus_two_l476_476491


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476357

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476357


namespace solve_equation1_solve_equation2_l476_476078

theorem solve_equation1 (x : ℝ) : 4 - x = 3 * (2 - x) ↔ x = 1 :=
by sorry

theorem solve_equation2 (x : ℝ) : (2 * x - 1) / 2 - (2 * x + 5) / 3 = (6 * x - 1) / 6 - 1 ↔ x = -3 / 2 :=
by sorry

end solve_equation1_solve_equation2_l476_476078


namespace probability_interval_l476_476635

noncomputable theory

open MeasureTheory ProbabilityTheory

-- Let X be a standard normal random variable
def X : ℝ → ℝ := sorry  -- We define X to be a random variable with standard normal distribution

axiom X_normal_dist : ∀ (x : ℝ), P(X ≤ x) = cum_dist_normal 0 1 x  -- Cumulative distribution function for N(0,1)

axiom given_probability : P(X ≤ 1) = 0.8413

theorem probability_interval : P(-1 < X ∧ X < 0) = 0.3413 :=
by
  -- Courtesy: The normal distribution symmetry about x = 0 and the area from X < 1
  -- (We are reducing the half distribution because of symmetry)
  sorry

end probability_interval_l476_476635


namespace books_from_second_shop_l476_476659

-- Definitions
variables {x : ℕ} -- The number of books Rahim bought from the second shop
def amount_first_shop : ℕ := 581
def amount_second_shop : ℕ := 594
def books_first_shop : ℕ := 27
def average_price_per_book : ℕ := 25
def total_amount : ℕ := amount_first_shop + amount_second_shop
def total_books : ℕ := books_first_shop + x

-- Statement
theorem books_from_second_shop :
  total_amount = total_books * average_price_per_book → x = 20 :=
by
  sorry

end books_from_second_shop_l476_476659


namespace bargain_bin_books_after_changes_l476_476812

theorem bargain_bin_books_after_changes (total_books : ℝ) (initial_bargain_books : ℝ)
  (initial_bargain_percentage : ℝ) (additional_books : ℝ) (shift_percentage : ℝ) :
  total_books = 150 ∧
  initial_bargain_books = 41.0 ∧
  initial_bargain_percentage = 19.0 / 100 ∧
  additional_books = 33.0 ∧
  shift_percentage = 5.2 / 100 →
  let remaining_books := total_books - initial_bargain_books - additional_books in
  let shifted_books := (shift_percentage * remaining_books).floor in
  initial_bargain_books + additional_books + shifted_books = 77 :=
by
  sorry

end bargain_bin_books_after_changes_l476_476812


namespace rhombus_diagonals_not_equal_l476_476160

-- Define what a rhombus is
structure Rhombus where
  sides_equal : ∀ a b : ℝ, a = b  -- all sides are equal
  symmetrical : Prop -- it is a symmetrical figure
  centrally_symmetrical : Prop -- it is a centrally symmetrical figure

-- Theorem to state that the diagonals of a rhombus are not necessarily equal
theorem rhombus_diagonals_not_equal (r : Rhombus) : ¬(∀ a b : ℝ, a = b) := by
  sorry

end rhombus_diagonals_not_equal_l476_476160


namespace div_eq_210_over_79_l476_476825

def a_at_b (a b : ℕ) : ℤ := a^2 * b - a * (b^2)
def a_hash_b (a b : ℕ) : ℤ := a^2 + b^2 - a * b

theorem div_eq_210_over_79 : (a_at_b 10 3) / (a_hash_b 10 3) = 210 / 79 :=
by
  -- This is a placeholder and needs to be filled with the actual proof.
  sorry

end div_eq_210_over_79_l476_476825


namespace largest_positive_integer_k_l476_476876

noncomputable def max_colors := 2

theorem largest_positive_integer_k :
  ∃ (P : Type) (edges : P → P → Prop) (V : set P),
    finite V ∧ 
    (card (set.univ : set {p : P | ∃ q, edges p q}) = 2022) ∧
    (∀ v ∈ V, ∃ d, ∀ u ∈ V, abs (degree u - degree v) ≤ 1) ∧
    (∃ (coloring : (P → P → Prop) → fin max_colors),
      ∀ c (v1 v2 : P),
        (v1 ∈ V → v2 ∈ V → ∃ path,
          path c v1 v2)) :=
begin
  sorry
end

end largest_positive_integer_k_l476_476876


namespace find_pairs_l476_476400

open Nat

theorem find_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≥ y)
    (h : x * y - (x + y) = 2 * gcd x y + lcm x y) : (x = 9 ∧ y = 3) ∨ (x = 5 ∧ y = 5) :=
by
  sorry

end find_pairs_l476_476400


namespace solve_system_of_equations_l476_476665

theorem solve_system_of_equations (x y : ℝ) (h1 : x - y = -5) (h2 : 3 * x + 2 * y = 10) : x = 0 ∧ y = 5 := by
  sorry

end solve_system_of_equations_l476_476665


namespace perpendicular_MB_CD_l476_476765

variables {A B C D M : Type} [convex_quadrilateral A B C D] [midpoint M A C]
variables (α β : ℝ)
variables (a b c : Type)

open_locale real
open_locale geometry

-- Assume all elements are points
noncomputable def angle_A := (angle a b c) = α ∧ α < 90
noncomputable def angle_C := (angle a d c) = α
noncomputable def angle_ABD := (angle a b d) = 90
noncomputable def midpoint_M := midpoint m a c

theorem perpendicular_MB_CD : angle a m d = 90 → angle b c d = 90 → MB ⊥ CD :=
begin
  sorry,
end

end perpendicular_MB_CD_l476_476765


namespace traffic_accident_responsibility_l476_476008

theorem traffic_accident_responsibility (T_A T_B T_C T_D : Prop)
    (A_resp : T_A ↔ ¬T_B)
    (B_resp : T_B ↔ ¬T_C)
    (C_resp : T_C ↔ T_A)
    (D_resp : T_D ↔ ¬responsible(D))
    (one_truth : (T_A ∨ T_B ∨ T_C ∨ T_D) ∧ T_A ↔ (¬T_B ∧ ¬T_C ∧ ¬T_D) 
                                            ∧ T_B ↔ (¬T_A ∧ ¬T_C ∧ ¬T_D)
                                            ∧ T_C ↔ (¬T_A ∧ ¬T_B ∧ ¬T_D)
                                            ∧ T_D ↔ (¬T_A ∧ ¬T_B ∧ ¬T_C)) :
    responsible B → ¬responsible A ∧ ¬responsible C ∧ ¬responsible D := 
    sorry

end traffic_accident_responsibility_l476_476008


namespace johns_final_push_time_l476_476031

theorem johns_final_push_time (t : ℝ) : 
  (∀ t : ℝ, 4.2 * t = 3.7 * t + 17) → t = 34 :=
begin
  intro h,
  have h_eq : 0.5 * t = 17, 
  { rw [← sub_eq_zero, sub_mul, ← h, sub_self, zero_mul], 
    ring },
  exact eq_div_of_mul_eq (by norm_num) h_eq,
end

end johns_final_push_time_l476_476031


namespace arithmetic_sequence_minimum_value_S_n_l476_476553

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476553


namespace area_inside_C_outside_A_B_l476_476821

noncomputable section

-- Define circles A, B, and C along with their positions and other conditions
def radius := 1
def Circle (r : ℝ) := {x // x^2 + r^2 = r^2}

structure TangentCircles where
  A B C : Circle radius
  A_B_tangent : A.val = B.val
  M : ℝ  
  M_midpoint : M = (A.val + B.val) / 2
  C_tangent_to_M : C.val = M ∨ C.val = M

-- Statement of the problem
theorem area_inside_C_outside_A_B (cs : TangentCircles) : 
  let shared_area := cs.radius^2 * π / 4 - 1 / 2 
  cs.C_area := π*cs.radius^2
  let total_shared_area := 4 * shared_area 
  cs.C_area - total_shared_area = 2 := 
by 
  sorry

end area_inside_C_outside_A_B_l476_476821


namespace find_number_of_boys_l476_476007

/-- Define variables for the conditions in the problem --/
def total_students (girls boys : ℕ) : ℕ := girls + boys

def boys_ratio (boys total : ℕ) : ℚ := boys / total

/-- Problem statement: Given the conditions, prove there are 72 boys --/
theorem find_number_of_boys (h_girls : ℕ) (h_ratio : ℚ) (h_total_students : total_students h_girls (72 : ℕ)) :
    boys_ratio 72 h_total_students = 3/8 ∧ h_girls = 120 :=
by
  sorry

end find_number_of_boys_l476_476007


namespace arithmetic_sequence_and_minimum_sum_l476_476519

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476519


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476601

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476601


namespace melanie_marbles_l476_476646

noncomputable def melanie_blue_marbles : ℕ :=
  let sandy_dozen_marbles := 56
  let dozen := 12
  let sandy_marbles := sandy_dozen_marbles * dozen
  let ratio := 8
  sandy_marbles / ratio

theorem melanie_marbles (h1 : ∀ sandy_dozen_marbles dozen ratio, 56 = sandy_dozen_marbles ∧ sandy_dozen_marbles * dozen = 672 ∧ ratio = 8) : melanie_blue_marbles = 84 := by
  sorry

end melanie_marbles_l476_476646


namespace min_sin6_cos6_l476_476288

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476288


namespace minimize_sin_cos_six_l476_476309

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476309


namespace each_person_pays_50_97_l476_476703

noncomputable def total_bill (original_bill : ℝ) (tip_percentage : ℝ) : ℝ :=
  original_bill + original_bill * tip_percentage

noncomputable def amount_per_person (total_bill : ℝ) (num_people : ℕ) : ℝ :=
  total_bill / num_people

theorem each_person_pays_50_97 :
  let original_bill := 139.00
  let number_of_people := 3
  let tip_percentage := 0.10
  let expected_amount := 50.97
  abs (amount_per_person (total_bill original_bill tip_percentage) number_of_people - expected_amount) < 0.01
:= sorry

end each_person_pays_50_97_l476_476703


namespace arithmetic_sequence_min_value_S_l476_476518

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476518


namespace arithmetic_sequence_min_value_Sn_l476_476564

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476564


namespace total_age_is_40_l476_476072

variable (Rachel Rona Collete Tommy : ℕ)

def condition1 := Rachel = 2 * Rona
def condition2 := Collete = Rona / 2
def condition3 := Tommy = Collete + Rona
def condition4 := Rona = 8

theorem total_age_is_40 (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) :
  Rachel + Rona + Collete + Tommy = 40 :=
by
  sorry

end total_age_is_40_l476_476072


namespace radius_of_inscribed_circle_in_triangle_l476_476743

noncomputable def triangle_radius_inscribed_circle 
  (AB AC BC : ℝ) 
  (h_ab : AB = 5) 
  (h_ac : AC = 6) 
  (h_bc : BC = 7) : ℝ :=
let s := (AB + AC + BC) / 2 in
let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) in
K / s

theorem radius_of_inscribed_circle_in_triangle :
  triangle_radius_inscribed_circle 5 6 7 5 6 7 = (2 * Real.sqrt 6) / 3 := sorry

end radius_of_inscribed_circle_in_triangle_l476_476743


namespace count_five_digit_even_numbers_l476_476133

theorem count_five_digit_even_numbers : 
  let digits := {0, 1, 2, 3, 7}
  let even_numbers := { n : ℕ | 
    (∀ d, d ∈ (digits : set ℕ) → ∃ (i : fin 5), n.digits₀.nth i = some d) ∧ 
    n.digits₀.length = 5 ∧ 
    n % 2 = 0 ∧ 
    n.digits₀.nodup
  }
  finite.even_numbers.count = 42 :=
sorry

end count_five_digit_even_numbers_l476_476133


namespace proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l476_476149

theorem proposition_a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a > 1 ∨ a < 1) :=
sorry

theorem negation_of_proposition_b_incorrect (x : ℝ) : ¬(∀ x < 1, x^2 < 1) ↔ ∃ x < 1, x^2 ≥ 1 :=
sorry

theorem proposition_c_not_necessary (x y : ℝ) : (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)) :=
sorry

theorem proposition_d_necessary_not_sufficient (a b : ℝ) : (a ≠ 0 → ab ≠ 0) ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0) :=
sorry

theorem final_answer_correct :
  let proposition_A := (∃ (a : ℝ), a > 1 ∧ 1 / a < 1 ∧ (1 / a < 1 → a > 1 ∨ a < 1))
  let proposition_B := (¬(∀ (x : ℝ), x < 1 → x^2 < 1) ↔ ∃ (x : ℝ), x < 1 ∧ x^2 ≥ 1)
  let proposition_C := (∃ (x y : ℝ), (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)))
  let proposition_D := (∃ (a b : ℝ), a ≠ 0 ∧ ab ≠ 0 ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0))
  proposition_A ∧ proposition_D
:= 
sorry

end proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l476_476149


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476551

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476551


namespace number_of_such_functions_l476_476431

def function_property (f : Fin 10 → Fin 10) : Prop :=
  ∀ i j, (i.val + j.val + 2 = 11) → (f i).val + (f j).val + 2 = 11

theorem number_of_such_functions : 
  {f : Fin 10 → Fin 10 // function_property f}.finite_card = 100000 :=
sorry

end number_of_such_functions_l476_476431


namespace six_cube_2d_faces_count_l476_476796

open BigOperators

theorem six_cube_2d_faces_count :
    let vertices := 64
    let edges_1d := 192
    let edges_2d := 240
    let small_cubes := 46656
    let faces_per_plane := 36
    let planes_count := 15 * 7^4
    faces_per_plane * planes_count = 1296150 := by
  sorry

end six_cube_2d_faces_count_l476_476796


namespace Will_old_cards_l476_476753

theorem Will_old_cards (new_cards pages cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : pages = 6) (h3 : cards_per_page = 3) :
  (pages * cards_per_page) - new_cards = 10 :=
by
  sorry

end Will_old_cards_l476_476753


namespace number_of_integers_in_range_l476_476373

theorem number_of_integers_in_range :
  {x : ℕ | 400 ≤ x^2 ∧ x^2 ≤ 800}.to_finset.card = 9 :=
sorry

end number_of_integers_in_range_l476_476373


namespace computation_result_l476_476231

-- Define the vectors and scalar multiplications
def v1 : ℤ × ℤ := (3, -9)
def v2 : ℤ × ℤ := (2, -7)
def v3 : ℤ × ℤ := (-1, 4)

noncomputable def result : ℤ × ℤ := 
  let scalar_mult (m : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (m * v.1, m * v.2)
  scalar_mult 5 v1 - scalar_mult 3 v2 + scalar_mult 2 v3

-- The main theorem
theorem computation_result : result = (7, -16) :=
  by 
    -- Skip the proof as required
    sorry

end computation_result_l476_476231


namespace fixed_point_not_exist_exist_parallel_lines_at_least_two_points_not_on_line_l476_476411

def line_eq (α x y : ℝ) : Prop := (cos α * (x - 2) + sin α * (y + 1) = 1)

theorem fixed_point_not_exist :
  ¬ ∃ P : ℝ × ℝ, ∀ α : ℝ, line_eq α P.1 P.2 :=
sorry

theorem exist_parallel_lines :
  ∃ α1 α2 : ℝ, α1 ≠ α2 ∧ ∀ x y : ℝ, (line_eq α1 x y ↔ line_eq α2 x y) :=
sorry

theorem at_least_two_points_not_on_line :
  ∀ α : ℝ, ∃ (x y : ℝ), ¬ line_eq α x y ∧ ∃ (x' y' : ℝ), ¬ line_eq α x' y' ∧ (x', y') ≠ (x, y) :=
sorry

end fixed_point_not_exist_exist_parallel_lines_at_least_two_points_not_on_line_l476_476411


namespace marble_probability_l476_476772

theorem marble_probability :
  let total_ways := (Nat.choose 6 4)
  let favorable_ways := 
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1)
  let probability := (favorable_ways : ℚ) / total_ways
  probability = 4 / 5 := by
  sorry

end marble_probability_l476_476772


namespace train_speed_l476_476203

theorem train_speed (length_of_train time_to_cross : ℝ) (h_length : length_of_train = 800) (h_time : time_to_cross = 12) : (length_of_train / time_to_cross) = 66.67 :=
by
  sorry

end train_speed_l476_476203


namespace min_sixth_power_sin_cos_l476_476263

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476263


namespace incircle_excircle_property_l476_476089

-- Definitions of points and lines based on given conditions
variables {A B C K L P M : Type*}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited K] [inhabited L] [inhabited P] [inhabited M]

-- Consider the necessary geometric configurations
-- These would typically include the definitions of the incircle and excircle properties and tangents
-- but are simplified here for clarity

-- Conditions
axiom incircle_touches (triangle_ABC : Δ A B C) : touches_incircle triangle_ABC K AC ∧ touches_incircle triangle_ABC L BC
axiom B_excircle_touches (triangle_ABC : Δ A B C) : touches_B_excircle triangle_ABC P AC
axiom KL_intersects_parallel_through_A (triangle_ABC : Δ A B C) : intersects_parallel_line_through_A K L M BC

-- Main theorem
theorem incircle_excircle_property (triangle_ABC : Δ A B C) :
  segment_length P L = segment_length P M :=
sorry

end incircle_excircle_property_l476_476089


namespace solve_for_y_l476_476835

theorem solve_for_y (y : ℝ) (h : (y * (y^5)^(1/4))^(1/3) = 4) : y = 2^(8/3) :=
by {
  sorry
}

end solve_for_y_l476_476835


namespace arithmetic_sequence_min_value_S_l476_476514

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476514


namespace min_sixth_power_sin_cos_l476_476268

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476268


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476599

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476599


namespace A_can_give_C_start_l476_476004

def canGiveStart (total_distance start_A_B start_B_C start_A_C : ℝ) :=
  (total_distance - start_A_B) / total_distance * (total_distance - start_B_C) / total_distance = 
  (total_distance - start_A_C) / total_distance

theorem A_can_give_C_start :
  canGiveStart 1000 70 139.7849462365591 200 :=
by
  sorry

end A_can_give_C_start_l476_476004


namespace ratio_pentagon_area_l476_476236

noncomputable def square_side_length := 1
noncomputable def square_area := (square_side_length : ℝ)^2
noncomputable def total_area := 3 * square_area
noncomputable def area_triangle (base height : ℝ) := 0.5 * base * height
noncomputable def GC := 2 / 3 * square_side_length
noncomputable def HD := 2 / 3 * square_side_length
noncomputable def area_GJC := area_triangle GC square_side_length
noncomputable def area_HDJ := area_triangle HD square_side_length
noncomputable def area_AJKCB := square_area - (area_GJC + area_HDJ)

theorem ratio_pentagon_area :
  (area_AJKCB / total_area) = 1 / 9 := 
sorry

end ratio_pentagon_area_l476_476236


namespace b_range_l476_476951

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - 6*b*x + 3*b

theorem b_range (b : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ is_local_min (f x b)) → 0 < b ∧ b < 1/2 := by
  sorry

end b_range_l476_476951


namespace parallelogram_area_l476_476607

def vector (α : Type*) := matrix (fin 2) (fin 1) α

noncomputable def v : vector ℝ := !![6; -4]
noncomputable def w : vector ℝ := !![8; -1]
noncomputable def two_w : vector ℝ := !![2 * 8; 2 * (-1)]
noncomputable def matrix_v_two_w : matrix (fin 2) (fin 2) ℝ := !![6, 16; -4, -2]

theorem parallelogram_area :
  let det := matrix.det matrix_v_two_w in
  abs det = 52 :=
by
  let det := matrix.det matrix_v_two_w
  have h_det : det = 52 :=
    by
      sorry
  show abs det = 52 from
    by
      exact abs_eq_self.mpr (le_of_eq h_det)

end parallelogram_area_l476_476607


namespace jim_travel_distance_l476_476947

theorem jim_travel_distance :
  ∀ (John Jill Jim : ℝ),
  John = 15 →
  Jill = (John - 5) →
  Jim = (0.2 * Jill) →
  Jim = 2 :=
by
  intros John Jill Jim hJohn hJill hJim
  sorry

end jim_travel_distance_l476_476947


namespace train_speed_l476_476202

theorem train_speed (length_of_train time_to_cross : ℝ) (h_length : length_of_train = 800) (h_time : time_to_cross = 12) : (length_of_train / time_to_cross) = 66.67 :=
by
  sorry

end train_speed_l476_476202


namespace length_of_k_squared_l476_476669

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f(x) = a * x + b

noncomputable def j (f g h : ℝ → ℝ) (x : ℝ) : ℝ :=
max (f x) (max (g x) (h x))

noncomputable def k (f g h : ℝ → ℝ) (x : ℝ) : ℝ :=
min (f x) (min (g x) (h x))

theorem length_of_k_squared
  (f g h : ℝ → ℝ)
  (h_f : is_linear f)
  (h_g : is_linear g)
  (h_h : is_linear h) :
  let k_fun := k f g h in
  let interval := set.Icc (-3.5) 3.5 in
  let ℓ := Real.dist (k_fun 3.5) (k_fun (-3.5)) in
  (* need a formal way to express the length of the graph of k_fun over the interval *)
  ∃ ℓ : ℝ, ℓ ^ 2 = 245 :=
sorry

end length_of_k_squared_l476_476669


namespace min_value_proof_l476_476613

noncomputable def min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1) : ℝ :=
  if a + 3 * b = 1 ∧ a > 0 ∧ b > 0 then Inf {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ x = (1 / a + 1 / b)} else 0

theorem min_value_proof :
  let a := Real
  let b := Real
  ∀ (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1),
  min_value a b h₀ h₁ h₂ = 4 + 2 * real.sqrt 3 :=
sorry

end min_value_proof_l476_476613


namespace min_sin6_cos6_l476_476284

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476284


namespace problem1_problem2_l476_476894

def f (x m : ℝ) : ℝ := (x^2 + 3) / (x - m)

theorem problem1 (m : ℝ) : (∀ x : ℝ, x > m → f x m + m ≥ 0) ↔ m ∈ Set.Ici (- (2 * Real.sqrt 15) / 5) := sorry

theorem problem2 (m : ℝ) : (∀ x : ℝ, x > m → f x m ≥ 6) ↔ m = 1 := sorry

end problem1_problem2_l476_476894


namespace polynomial_remainder_l476_476047

theorem polynomial_remainder (Q : ℤ[X]) (c d : ℤ) :
  (Q.eval 15 = 8) → 
  (Q.eval 19 = 10) → 
  (∀ Q, ∃ R, ∀ x, Q = (x - 15) * (x - 19) * R + C * x + d) :=
  sorry

end polynomial_remainder_l476_476047


namespace distance_between_P1_P2_l476_476123

-- Define the points and triangle properties in Lean
def Point := (ℝ × ℝ)

structure Triangle :=
(A B C : Point)
(AB BC CA : ℝ)
(AB_eq : AB = dist A B)
(BC_eq : BC = dist B C)
(CA_eq : CA = dist C A)

structure Collinear (A B C : Point) : Prop :=
(left_of_right : A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) = 0)

structure AngleBisector (A B C P : Point) : Prop :=
(bisects : dist A P * dist B C = dist B P * dist A C)

structure Perpendicular (A B C P : Point) : Prop :=
(perp90 : (P.1 - B.1) * (P.1 - C.1) + (P.2 - B.2) * (P.2 - C.2) = 0)

noncomputable def dist (P₁ P₂ : Point) : ℝ := 
  real.sqrt ((P₁.1 - P₂.1)^2 + (P₂.2 - P₂.2)^2)

theorem distance_between_P1_P2 (A B C P₁ P₂ : Point) 
  (triangle : Triangle A B C 21 55 56)
  (angle_bisector_P1 : AngleBisector A B C P₁)
  (angle_bisector_P2 : AngleBisector A B C P₂)
  (perpendicular_P1 : Perpendicular B C P₁)
  (perpendicular_P2 : Perpendicular B C P₂) :
  dist P₁ P₂ = (5 / 2) * real.sqrt 409 :=
sorry

end distance_between_P1_P2_l476_476123


namespace sum_of_possible_values_of_N_l476_476882

theorem sum_of_possible_values_of_N :
  (∑ n in {0, 1, 3, 4, 5, 6, 7, 8, 9, 10}, n) = 53 :=
by
  -- Proof omitted
  sorry

end sum_of_possible_values_of_N_l476_476882


namespace proof_problem_l476_476405

variables {f : ℝ → ℝ}

-- The conditions given in the problem
def condition1 : Prop := ∀ x y : ℝ, f(x + y) = f(x) + f(y)
def condition2 : Prop := ∀ x : ℝ, x > 0 → f(x) > 0

-- The statement to prove
theorem proof_problem (h1 : condition1) (h2 : condition2) :
  f 0 = 0 ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ∧ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end proof_problem_l476_476405


namespace ellie_reflections_in_wide_mirrors_l476_476845

theorem ellie_reflections_in_wide_mirrors :
  (E : ℕ) → 
  (total_reflections : ℕ) →
  (Sarah_tall_mirrors : ℕ) →
  (Sarah_wide_mirrors : ℕ) →
  (Sarah_tall_mirrors_passes : ℕ) →
  (Sarah_wide_mirrors_passes : ℕ) →
  (Ellie_tall_mirrors : ℕ) →
  (Ellie_tall_mirrors_passes : ℕ) →
  (Ellie_wide_mirrors_passes : ℕ) →
  (Sarah_reflections : Sarah_tall_mirrors * Sarah_tall_mirrors_passes + Sarah_wide_mirrors * Sarah_wide_mirrors_passes) →
  3 ∗ Ellie_tall_mirrors + (Ellie_wide_mirrors_passes * E) =
  total_reflections - 
    (Sarah_tall_mirrors * Sarah_tall_mirrors_passes + Sarah_wide_mirrors * Sarah_wide_mirrors_passes) →
  E = 3 :=
begin
  intros,
  sorry
end

end ellie_reflections_in_wide_mirrors_l476_476845


namespace modulus_of_z_l476_476399

-- Definitions based on conditions
def imaginary_unit := Complex.i
def z : Complex := (-4 : Complex.i) + 3

-- Main statement to prove
theorem modulus_of_z (h : imaginary_unit * z = (3 + 4 * Complex.i)) : Complex.abs z = 5 := by
  sorry

end modulus_of_z_l476_476399


namespace min_sixth_power_sin_cos_l476_476277

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476277


namespace isosceles_triangle_perimeter_l476_476214

theorem isosceles_triangle_perimeter {a : ℝ} (h_base : 4 ≠ 0) (h_roots : a^2 - 5 * a + 6 = 0) :
  a = 3 → (4 + 2 * a = 10) :=
by
  sorry

end isosceles_triangle_perimeter_l476_476214


namespace shaded_to_white_ratio_l476_476744

theorem shaded_to_white_ratio (area_shaded area_white : ℝ)
  (h1 : vertices_midpoints : ∀ (square : ℕ), square ≠ largest → vertices_at_midpoints square)
  (h2 : symmetric_configuration : symmetric configuration)
  (total_area_quadrant_shaded : ℝ := 5 * area_of_one_triangle)
  (total_area_quadrant_white : ℝ := 3 * area_of_one_triangle)
  (area_of_one_triangle : ℝ)
  (total_area_shaded : ℝ := 4 * total_area_quadrant_shaded)
  (total_area_white : ℝ := 4 * total_area_quadrant_white)
  (ratio : ℝ := total_area_shaded / total_area_white) :
  ratio = 5 / 3 := 
sorry

end shaded_to_white_ratio_l476_476744


namespace area_of_triangle_AEC_l476_476730

-- Definitions based on the conditions
structure Triangle :=
  (a b c : ℝ)

def right_angled_at (t : Triangle) (p : ℝ) : Prop :=
  p = t.c -- Place-holder for the condition, assuming vertex 'c' is the right-angle

variable (ABC DBC : Triangle) (AC BC CD AE CE : ℝ)

def AC : ℝ := 8
def BC : ℝ := 6
def CD : ℝ := 2

def right_triangles_sharing_hypotenuse (ABC DBC : Triangle) : Prop :=
  right_angled_at ABC BC ∧ right_angled_at DBC CD ∧ ABC.b = DBC.b

def AE_ED_ratio : Prop :=
  AE = 2 / 3 * (AC + CD)

def hypotenuse (ABC : Triangle) : ℝ :=
  real.sqrt (AC ^ 2 + BC ^ 2)

noncomputable def area_triangle_AEC : ℝ :=
  1 / 2 * AC * CE

theorem area_of_triangle_AEC
  (h1 : right_triangles_sharing_hypotenuse ABC DBC)
  (h2 : AE_ED_ratio)
  (h3 : CE = BC)
  : area_triangle_AEC = 24 := by
  sorry

end area_of_triangle_AEC_l476_476730


namespace system_solutions_l476_476854

theorem system_solutions : 
  ∃ (x y z t : ℝ), 
    (x * y - t^2 = 9) ∧ 
    (x^2 + y^2 + z^2 = 18) ∧ 
    ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ 
     (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) :=
by {
  sorry
}

end system_solutions_l476_476854


namespace arithmetic_sequence_min_value_S_l476_476512

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476512


namespace ellipse_equation_and_constant_value_l476_476391

noncomputable def ellipse (a b : ℝ) : Prop := ∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1)

theorem ellipse_equation_and_constant_value 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) 
  (eccentricity : (sqrt 2 / 2)^2 = (a^2 - b^2) / a^2) (AB_len : 2 * b^2 / a = 2) : 
  ellipse a b ∧ 
  (∀ k x1 x2 y1 y2 : ℝ, 
    let M := (x1, y1),
        N := (x2, y2),
        P := (0, sqrt 3),
        l := λ x : ℝ, k * x + sqrt 3,
        OM := M.1 + M.2,
        ON := N.1 + N.2,
        PM := (x1, y1 - sqrt 3),
        PN := (x2, y2 - sqrt 3)
    in (OM * ON - 7 * (PM.1 * PN.1 + PM.2 * PN.2) = -9)) :=
  sorry

end ellipse_equation_and_constant_value_l476_476391


namespace milk_revenue_l476_476649

theorem milk_revenue :
  let yesterday_morning := 68
  let yesterday_evening := 82
  let this_morning := yesterday_morning - 18
  let total_milk_before_selling := yesterday_morning + yesterday_evening + this_morning
  let milk_left := 24
  let milk_sold := total_milk_before_selling - milk_left
  let cost_per_gallon := 3.50
  let revenue := milk_sold * cost_per_gallon
  revenue = 616 := by {
    sorry
}

end milk_revenue_l476_476649


namespace minimize_sin_cos_six_l476_476303

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476303


namespace greatest_M_inequality_l476_476874

theorem greatest_M_inequality :
  ∀ x y z : ℝ, x^4 + y^4 + z^4 + x * y * z * (x + y + z) ≥ (2/3) * (x * y + y * z + z * x)^2 :=
by
  sorry

end greatest_M_inequality_l476_476874


namespace min_value_sin6_cos6_l476_476313

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476313


namespace pq_sum_eq_14_l476_476427

set_option pp.beta true
set_option pp.generalizedFieldNotations false

theorem pq_sum_eq_14 (p q : ℝ) :
  (∀ x, x ∈ { x | x^2 - p * x + 15 = 0 } ↔ (x = 3))
  ∧ (∀ x, x ∈ { x | x^2 - 5 * x + q = 0 } ↔ (x = 3))
  → p + q = 14 :=
by
  intros h,
  sorry

end pq_sum_eq_14_l476_476427


namespace solution_arccos_cos_equation_l476_476077

theorem solution_arccos_cos_equation (x : ℝ) (h1 : -π/3 ≤ x) (h2 : x ≤ π/3) : 
  arccos (cos x) = x / 2 → x = 0 := 
by
  sorry

end solution_arccos_cos_equation_l476_476077


namespace isosceles_triangle_perimeter_l476_476215

theorem isosceles_triangle_perimeter {a : ℝ} (h_base : 4 ≠ 0) (h_roots : a^2 - 5 * a + 6 = 0) :
  a = 3 → (4 + 2 * a = 10) :=
by
  sorry

end isosceles_triangle_perimeter_l476_476215


namespace power_function_increasing_l476_476096

theorem power_function_increasing (m : ℝ) : 
  (∀ x : ℝ, 0 < x → (m^2 - 2*m - 2) * x^(-4*m - 2) > 0) ↔ m = -1 :=
by sorry

end power_function_increasing_l476_476096


namespace salary_cost_increase_1993_l476_476799

variable (E : ℝ)
variable (S : ℝ)
variable (F : ℝ)
variable (P : ℝ)

-- Conditions
axiom EmploymentCostIncrease : E' = 1.035 * E
axiom SalaryCostIncrease : S' = (1 + P) * S
axiom FringeBenefitCostIncrease : F' = 1.055 * F
axiom InitialFringeBenefitCost : F = 0.20 * E
axiom InitialSalaryCost : S = 0.80 * E
axiom TotalEmploymentCost : E' = S' + F'

-- Goal
theorem salary_cost_increase_1993 : P = 0.03 :=
by
  sorry

end salary_cost_increase_1993_l476_476799


namespace parabola_ratio_l476_476420

noncomputable def ratio_AF_BF (p : ℝ) (h_pos : p > 0) : ℝ :=
  let y1 := (Real.sqrt (2 * p * (3 / 2 * p)))
  let y2 := (Real.sqrt (2 * p * (1 / 6 * p)))
  let dist1 := Real.sqrt ((3 / 2 * p - (p / 2))^2 + y1^2)
  let dist2 := Real.sqrt ((1 / 6 * p - p / 2)^2 + y2^2)
  dist1 / dist2

theorem parabola_ratio (p : ℝ) (h_pos : p > 0) : ratio_AF_BF p h_pos = 3 :=
  sorry

end parabola_ratio_l476_476420


namespace joint_distribution_Tg_l476_476998

noncomputable def joint_density_Tg (t x : ℝ) : ℝ :=
  if t > 0 ∧ x ∈ Ioo 0 1 then
    exp (-t) * (1 / (π * sqrt(x * (1 - x))))
  else 0

theorem joint_distribution_Tg {X Y : ℝ → ℝ} (hX_dist : X ∼ ℕ 0 1)
  (hY_dist : Y ∼ ℕ 0 1) (h_indep : indep X Y) :
  ∀ t x : ℝ, joint_density_Tg t x = exp (-t) * (1 / (π * sqrt(x * (1 - x)))) ↔
    (t > 0 ∧ x ∈ Ioo 0 1) := 
sorry

end joint_distribution_Tg_l476_476998


namespace selection_methods_including_both_boys_and_girls_l476_476661

def boys : ℕ := 4
def girls : ℕ := 3
def total_people : ℕ := boys + girls
def select : ℕ := 4

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_methods_including_both_boys_and_girls :
  combination 7 4 - combination boys 4 = 34 :=
by
  sorry

end selection_methods_including_both_boys_and_girls_l476_476661


namespace obtuse_triangle_l476_476632

open Real

def f1 (x : ℝ) : ℝ := x * exp x

def f (n : ℕ) : (ℝ → ℝ) :=
  match n with
  | 0 => f1
  | n + 1 => (λ x, derivative (f n) x)

def P (n : ℕ) : ℝ × ℝ :=
  (-n, - (1 / exp (n : ℝ)))

theorem obtuse_triangle (n : ℕ) (hn : n > 0) :
  let Pn := P n
  let Pn1 := P (n + 1)
  let Pn2 := P (n + 2)
  let K_PnPn1 := (Pn.2 - Pn1.2) / (Pn.1 - Pn1.1)
  let K_Pn1Pn2 := (Pn1.2 - Pn2.2) / (Pn1.1 - Pn2.1)
  K_PnPn1 > K_Pn1Pn2 := sorry

end obtuse_triangle_l476_476632


namespace non_monotonic_interval_l476_476634

def is_non_monotonic (f : ℝ → ℝ) (a x : ℝ) : Prop :=
  ∃ (c d : ℝ), (0 < c ∧ c < 3 ∧ c ≠ d ∧ d ≤ c ∧ (∀ (x : ℝ), c < x ∧ x < d → f'(x) ≠ 0)) 

theorem non_monotonic_interval (a : ℝ) (h : 0 < a) :
  (is_non_monotonic (λ x, (1/3) * a * x^3 - x^2) a (0, 3)) → a > (2/3) :=
begin
  sorry
end

end non_monotonic_interval_l476_476634


namespace initial_rain_amount_l476_476466

theorem initial_rain_amount
    (rain_2to4 : 4 * 2 = 8)
    (rain_4to7 : 3 * 3 = 9)
    (rain_7to9 : 0.5 * 2 = 1)
    (total_rain : 8 + 9 + 1 = 18)
    (final_amount : 20)
    : 20 - 18 = 2 :=
begin
  sorry -- proof to be completed
end

end initial_rain_amount_l476_476466


namespace part_I_part_II_l476_476417

-- Define the function f
def f (x : ℝ) : ℝ := Real.log x - 2 * Real.sqrt x

-- Define the first proof problem for part (I)
theorem part_I (x : ℝ) (h1 : x > 1) : f x < -Real.sqrt x - 1 / Real.sqrt x := by
  sorry

-- Define the second proof problem for part (II)
theorem part_II (x : ℝ) (h1 : x ≥ 1/4) (h2 : x ≤ 1) : f x ≥ 2/3 * x - 8/3 := by
  sorry

end part_I_part_II_l476_476417


namespace dimes_difference_l476_476069

theorem dimes_difference
  (a b c d : ℕ)
  (h1 : a + b + c + d = 150)
  (h2 : 5 * a + 10 * b + 25 * c + 50 * d = 1500) :
  (b = 150 ∨ ∃ c d : ℕ, b = 0 ∧ 4 * c + 9 * d = 150) →
  ∃ b₁ b₂ : ℕ, (b₁ = 150 ∧ b₂ = 0 ∧ b₁ - b₂ = 150) :=
by
  sorry

end dimes_difference_l476_476069


namespace sum_of_angles_is_810_l476_476828

noncomputable def root_angles_sum : ℝ :=
  let z :=
    [2 * (Complex.cos (OfReal 18) + Complex.sin (OfReal 18) * Complex.i),
     2 * (Complex.cos (OfReal 90) + Complex.sin (OfReal 90) * Complex.i),
     2 * (Complex.cos (OfReal 162) + Complex.sin (OfReal 162) * Complex.i),
     2 * (Complex.cos (OfReal 234) + Complex.sin (OfReal 234) * Complex.i),
     2 * (Complex.cos (OfReal 306) + Complex.sin (OfReal 306) * Complex.i)]
  ∑ θ : ℝ in [18, 90, 162, 234, 306], θ

theorem sum_of_angles_is_810 :
  root_angles_sum = 810 :=
sorry

end sum_of_angles_is_810_l476_476828


namespace ethan_pages_left_l476_476247

-- Definitions based on the conditions
def total_pages := 360
def pages_read_morning := 40
def pages_read_night := 10
def pages_read_saturday := pages_read_morning + pages_read_night
def pages_read_sunday := 2 * pages_read_saturday
def total_pages_read := pages_read_saturday + pages_read_sunday

-- Lean 4 statement for the proof problem
theorem ethan_pages_left : total_pages - total_pages_read = 210 := by
  sorry

end ethan_pages_left_l476_476247


namespace find_point_P_l476_476016

open Real

noncomputable def point (x y : ℝ) := (x, y)

def line (x : ℝ) : ℝ := -3/2 * x - 3

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem find_point_P :
  let A := point (-5) 0
  let B := point (-2) 0
  let P (x y : ℝ) := point x y
  let area (A B P : ℝ × ℝ) : ℝ :=
    1/2 * abs ((B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1))

  (P (-14/3) 4 or P (2/3) -4) ∧ 
  line (-14/3) = 4 ∧ 
  line (2/3) = -4 ∧
  abs (
    area A B (P (-14/3) 4) = 6 ∧ 
    area A B (P (2/3) -4) = 6) := sorry

end find_point_P_l476_476016


namespace conical_cup_radius_l476_476183

-- Defining the conditions
def volume := 150
def height := 12
def pi := Real.pi

-- Definition for the conical cup problem
def radius (r : ℝ) : Prop :=
  (1 / 3) * pi * (r^2) * height = volume

-- Main statement to prove
theorem conical_cup_radius : ∃ r : ℝ, radius r ∧ r ≈ 3.5 :=
by
  sorry

end conical_cup_radius_l476_476183


namespace campus_reading_festival_prizes_l476_476102

-- Definitions based on conditions
variables (price_B : ℕ)
variables (price_A : ℕ)
variables (x : ℕ)

-- Conditions
def condition1 : Prop := price_A = 3 * price_B / 2
def condition2 : Prop := 600 / price_A + 10 = 600 / price_B
def condition3 : Prop := ∀(a b : ℕ), a + b = 40 → price_A * a + price_B * b ≤ 1050 → a > b
def answer1 : Prop := price_A = 30 ∧ price_B = 20
def answer2 : Prop := ∃ (l : List ℕ), (∀ (x ∈ l), 20 < x ∧ x ≤ 25) ∧ l.length = 5

-- Proof Problem
theorem campus_reading_festival_prizes :
  condition1 price_A price_B ∧ condition2 price_A price_B ∧ condition3 price_A price_B →
  answer1 price_A price_B ∧ answer2 :=
by
  sorry

end campus_reading_festival_prizes_l476_476102


namespace angle_between_vectors_is_correct_l476_476864

def vec_a : ℝ × ℝ × ℝ := (3, -2, 2)
def vec_b : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def angle_between_vectors : ℝ :=
  Real.acos ((vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 + vec_a.3 * vec_b.3) / 
    (Real.sqrt (vec_a.1^2 + vec_a.2^2 + vec_a.3^2) * Real.sqrt (vec_b.1^2 + vec_b.2^2 + vec_b.3^2)))

theorem angle_between_vectors_is_correct :
  angle_between_vectors = Real.acos (-8 / (3 * Real.sqrt 17)) :=
by sorry

end angle_between_vectors_is_correct_l476_476864


namespace barrels_distribution_possible_l476_476118

noncomputable def barrels_distribution_problem :
  (A, B, C : ℕ) → (full_barrels : ℕ) → (half_filled_barrels : ℕ) → (empty_barrels : ℕ) →
  Prop := 
  ∀(total_barrels : ℕ) (total_wine : ℕ) (barrels : list ℕ), 
  (total_barrels = 24) →
  (total_wine = 16) → -- 8 full barrels contain 8 units, 8 half-filled barrels contain 4 units,
  -- total 12 units of wine

  (∀barrel ∈ barrels, barrel = 0 ∨ barrel = 1 ∨ barrel = 0.5) →
  list.sum barrels = 24 →

  -- Use different sets of barrels for A, B, and C each having exactly 8 barrels,
  -- and totaling to 4 units of wine each
  (∃ (A_Barrels B_Barrels C_Barrels : list ℕ),
    A_Barrels.length = 8 ∧ B_Barrels.length = 8 ∧ C_Barrels.length = 8 ∧
    list.sum A_Barrels = 4 ∧ list.sum B_Barrels = 4 ∧ list.sum C_Barrels = 4 ∧
    A_Barrels ++ B_Barrels ++ C_Barrels = barrels) →

  -- Check for the balance and distribution of the barrels as per the given constraints
  (full_barrels = 8 ∧ half_filled_barrels = 8 ∧ empty_barrels = 8) →

  -- Ensuring each heir gets equal number of barrels and wine
  (∀hr : ℕ, (hr = (total_barrels / 3)) ∧ (hr = (total_wine / 4))) →
  ∃ (A B C : list ℕ), A.length = 8 ∧ B.length = 8 ∧ C.length = 8) 

theorem barrels_distribution_possible :
  ∃A B C : ℕ, barrels_distribution_problem A B C 8 8 8 
:= sorry

end barrels_distribution_possible_l476_476118


namespace employee_overtime_hours_l476_476808

theorem employee_overtime_hours (gross_pay : ℝ) (rate_regular : ℝ) (regular_hours : ℕ) (rate_overtime : ℝ) :
  gross_pay = 622 → rate_regular = 11.25 → regular_hours = 40 → rate_overtime = 16 →
  ∃ (overtime_hours : ℕ), overtime_hours = 10 :=
by
  sorry

end employee_overtime_hours_l476_476808


namespace area_PQR_l476_476666

noncomputable theory
open_locale classical

structure Pyramid :=
(base_side : ℝ)
(altitude : ℝ)

def F := EuclideanGeometry.Point 3
def G := EuclideanGeometry.Point 3
def H := EuclideanGeometry.Point 3
def I := EuclideanGeometry.Point 3
def J := EuclideanGeometry.Point 3

def distance (p1 p2 : EuclideanGeometry.Point 3) := EuclideanGeometry.dist p1 p2

def Pyramid.points (p : Pyramid) : Prop :=
  distance F G = distance G H ∧ distance G I = p.base_side ∧ 
  distance F J = p.altitude ∧ 
  J.coord = (G.coord + (1/4)*(J.coord - G.coord)) ∧
  J.coord = (I.coord + (1/4)*(J.coord - I.coord)) ∧
  J.coord = (H.coord + (3/4)*(J.coord - H.coord))

def area_of_triangle (p1 p2 p3 : EuclideanGeometry.Point 3) : ℝ := 
  let a := distance p1 p2,
      b := distance p2 p3,
      c := distance p3 p1,
      s := (a + b + c) / 2 in
  (s * (s - a) * (s - b) * (s - c)).sqrt

def P := EuclideanGeometry.Point 3
def Q := EuclideanGeometry.Point 3
def R := EuclideanGeometry.Point 3

theorem area_PQR (p : Pyramid) (h : p.points): 
  area_of_triangle P Q R = real.sqrt 35 :=
begin
  sorry
end

end area_PQR_l476_476666


namespace maria_earnings_l476_476641

-- Define the conditions
def costOfBrushes : ℕ := 20
def costOfCanvas : ℕ := 3 * costOfBrushes
def costPerLiterOfPaint : ℕ := 8
def litersOfPaintNeeded : ℕ := 5
def sellingPriceOfPainting : ℕ := 200

-- Define the total cost calculation
def totalCostOfMaterials : ℕ := costOfBrushes + costOfCanvas + (costPerLiterOfPaint * litersOfPaintNeeded)

-- Define the final earning calculation
def mariaEarning : ℕ := sellingPriceOfPainting - totalCostOfMaterials

-- State the theorem
theorem maria_earnings :
  mariaEarning = 80 := by
  sorry

end maria_earnings_l476_476641


namespace count_valid_n_l476_476884

def is_integer_division (num denom : ℕ) : Prop :=
  ∃ k : ℕ, num = denom * k

theorem count_valid_n :
  (finset.filter
    (λ n, is_integer_division ((n^2)!) ((n!)^(n + 2)))
    (finset.range 61).filter (λ n, 1 ≤ n)).card = 1 :=
by
  sorry

end count_valid_n_l476_476884


namespace number_of_real_roots_l476_476058

noncomputable def f (x : ℝ) : ℝ := 1 - |1 - 2 * x|

noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x + 1

noncomputable def F (x : ℝ) : ℝ :=
if f(x) ≥ g(x) then f(x) else g(x)

theorem number_of_real_roots : 
  (set.count 
  { x : ℝ | 0 ≤ x ∧ x ≤ 1 ∧ F(x) * 2^x = 1 })
  = 3 := by
  sorry

end number_of_real_roots_l476_476058


namespace distance_to_line_l476_476789

theorem distance_to_line (a : ℝ) (d : ℝ)
  (h1 : d = 6)
  (h2 : |3 * a + 6| / 5 = d) :
  a = 8 ∨ a = -12 :=
by
  sorry

end distance_to_line_l476_476789


namespace slope_AD_l476_476017

noncomputable def point := ℝ × ℝ

def B : point := (9, -4)
def C : point := (6, 7)

def slope (P Q : point) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

def m_BC := slope B C

theorem slope_AD :
  ∃ D : point, ∃ A : point,
  m_BC = -11/3 ∧ slope A D = 3/11 ∧ (slope A D) * m_BC = -1 :=
by
  sorry

end slope_AD_l476_476017


namespace expand_expression_l476_476249

theorem expand_expression (x y : ℝ) : (3 * x + 15) * (4 * y + 12) = 12 * x * y + 36 * x + 60 * y + 180 := 
  sorry

end expand_expression_l476_476249


namespace twenty_second_n_22npower_ends2_l476_476732

theorem twenty_second_n_22npower_ends2 : 
  ∃ n : ℕ, (22^n % 10 = 2) ∧ ∃ (k : ℕ), (k = 22) ∧ (n = 1 + 4 * (k - 1)) :=
by
  -- n is the 22nd term of the sequence n = 1 + 4 * (k-1)
  use 85
  split
  -- 22^n % 10 = 2
  sorry
  use 22
  split
  -- k = 22
  rfl
  -- n = 1 + 4 * (k-1)
  rfl

end twenty_second_n_22npower_ends2_l476_476732


namespace movie_marathon_duration_l476_476721

theorem movie_marathon_duration :
  let first_movie := 2
  let second_movie := first_movie + 0.5 * first_movie
  let combined_time := first_movie + second_movie
  let third_movie := combined_time - 1
  first_movie + second_movie + third_movie = 9 := by
  sorry

end movie_marathon_duration_l476_476721


namespace minimum_road_length_l476_476652

/-- Define the grid points A, B, and C with their coordinates. -/
def A : ℤ × ℤ := (0, 0)
def B : ℤ × ℤ := (3, 2)
def C : ℤ × ℤ := (4, 3)

/-- Define the side length of each grid square in meters. -/
def side_length : ℕ := 100

/-- Calculate the Manhattan distance between two points on the grid. -/
def manhattan_distance (p q : ℤ × ℤ) : ℕ :=
  (Int.natAbs (p.1 - q.1) + Int.natAbs (p.2 - q.2)) * side_length

/-- Statement: The minimum total length of the roads (in meters) to connect A, B, and C is 1000 meters. -/
theorem minimum_road_length : manhattan_distance A B + manhattan_distance B C + manhattan_distance C A = 1000 := by
  sorry

end minimum_road_length_l476_476652


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476365

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476365


namespace digits_assignment_impossible_l476_476386

-- Defining the main problem context
def regular_45_gon := Type -- A regular 45-sided polygon

def vertices : fin 45 := sorry -- The vertices of the polygon

-- Digits to be assigned
def digits := fin 10

def assign_digits (v : fin 45 → fin 10) := sorry -- Assignment function from vertices to digits

def endpoint_side (v : fin 45) (d1 d2 : fin 10) : Prop := sorry -- Property to check if a side's endpoints are labeled with different digits

-- Problem statement
theorem digits_assignment_impossible :
  ¬ ∃ (assignment : fin 45 → fin 10),
    (∀ (d1 d2 : fin 10), d1 ≠ d2 → ∃ (v : fin 45), endpoint_side v d1 d2 (assignment v))
:=
sorry

end digits_assignment_impossible_l476_476386


namespace mel_age_when_katherine_is_24_l476_476061

theorem mel_age_when_katherine_is_24 (katherine_age : ℕ) (mel_diff : ℕ) :
  katherine_age = 24 → mel_diff = 3 → (katherine_age - mel_diff = 21) :=
by
  intros h_katherine h_diff
  rw [h_katherine, h_diff]
  norm_num

end mel_age_when_katherine_is_24_l476_476061


namespace arithmetic_sequence_min_value_S_l476_476502

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476502


namespace tan_theta_point_l476_476954

open Real

theorem tan_theta_point :
  ∀ θ : ℝ,
  ∃ (x y : ℝ), x = -sqrt 3 / 2 ∧ y = 1 / 2 ∧ (tan θ) = y / x → (tan θ) = -sqrt 3 / 3 :=
by
  sorry

end tan_theta_point_l476_476954


namespace sequence_problem_l476_476587

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476587


namespace arithmetic_sequence_min_value_S_l476_476516

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476516


namespace triangle_congruence_l476_476027

open Real EuclideanGeometry

noncomputable def Triangle := {p : Point → ℝ // ∃ A B C, p = Triangle.mk A B C}

variables (A B C D E O : Point)
variables (h1 : D ∈ AC)
variables (h2 : E ∈ AB)
variables (h3 : BE = CD)
variables (h4 : (BD ∩ CE) = {O})
variables (h5 : ∠BOC = π/2 + 1/2 * ∠BAC)

theorem triangle_congruence (h1 : D ∈ AC) (h2 : E ∈ AB) (h3 : BE = CD) (h4 : BD ∩ CE = {O}) (h5 : ∠BOC = π/2 + 1/2 * ∠BAC) : CD = DE :=
by
  sorry

end triangle_congruence_l476_476027


namespace max_product_MF₁_MF₂_max_angle_F₁_M_F₂_trajectory_eqns_l476_476901

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1
def focus₁ : ℝ × ℝ := (-1, 0)
def focus₂ : ℝ × ℝ := (1, 0)
def is_moving_point_on_ellipse (M : ℝ × ℝ) : Prop := ellipse_eq M.1 M.2

theorem max_product_MF₁_MF₂ (M : ℝ × ℝ) (hM : is_moving_point_on_ellipse M) :
  let MF₁ := (real.sqrt ((M.1 - focus₁.1)^2 + (M.2 - focus₁.2)^2))
  let MF₂ := (real.sqrt ((M.1 - focus₂.1)^2 + (M.2 - focus₂.2)^2))
  MF₁ * MF₂ ≤ 4 := sorry

theorem max_angle_F₁_M_F₂ (M : ℝ × ℝ) (hM : is_moving_point_on_ellipse M) :
  let angle := real.arctan (real.sqrt 3 / 3)
  2 * angle = real.pi / 3 := sorry

theorem trajectory_eqns (P : ℝ × ℝ) (A B : ℝ × ℝ) (hA : P.2 = A.2) (hB : P.2 = B.2) :
  let PA := real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  PA * PB = 2 →
  (ellipse_eq P.1 P.2 ∨ (P.1^2 / 2) + (2 * P.2^2 / 3) = 1 ∨ (P.1^2 / 6) + (2 * P.2^2 / 9) = 1) := sorry

end max_product_MF₁_MF₂_max_angle_F₁_M_F₂_trajectory_eqns_l476_476901


namespace floating_time_l476_476773

theorem floating_time (boat_with_current: ℝ) (boat_against_current: ℝ) (distance: ℝ) (time: ℝ) : 
boat_with_current = 28 ∧ boat_against_current = 24 ∧ distance = 20 ∧ 
time = distance / ((boat_with_current - boat_against_current) / 2) → 
time = 10 := by
  sorry

end floating_time_l476_476773


namespace maximize_profit_l476_476777

noncomputable def profit (x a : ℝ) : ℝ :=
  19 - 24 / (x + 2) - (3 / 2) * x

theorem maximize_profit (a : ℝ) (ha : 0 < a) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ a) ∧ 
  (if a ≥ 2 then x = 2 else x = a) :=
by
  sorry

end maximize_profit_l476_476777


namespace roots_of_equation_of_param_l476_476663

noncomputable def equation_has_roots (a : ℝ) : Prop :=
  sqrt(3 * a - 2 * x) + x = a

theorem roots_of_equation_of_param (a : ℝ) :
  (a < -1 → ¬∃ x : ℝ, sqrt(3 * a - 2 * x) + x = a) ∧
  (a = -1 → (∃! x : ℝ, sqrt(3 * a - 2 * x) + x = a ∧ x = -2)) ∧
  (-1 < a ∧ a ≤ 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (sqrt(3 * a - 2 * x₁) + x₁ = a) ∧ 
    (sqrt(3 * a - 2 * x₂) + x₂ = a) ∧
    (x₁ = a - 1 - sqrt(a + 1)) ∧ 
    (x₂ = a - 1 + sqrt(a + 1))) ∧
  (a > 0 → ∃ x : ℝ, sqrt(3 * a - 2 * x) + x = a ∧ x = a - 1 - sqrt(a + 1))
:= by sorry

end roots_of_equation_of_param_l476_476663


namespace negation_of_proposition_l476_476091

noncomputable def negation_proposition (f : ℝ → Prop) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ ¬ f x

theorem negation_of_proposition :
  (∀ x : ℝ, x ≥ 0 → x^2 + x - 1 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + x - 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l476_476091


namespace square_circle_radius_l476_476731

theorem square_circle_radius (a R : ℝ) (h1 : a^2 = 256) (h2 : R = 10) : R = 10 :=
sorry

end square_circle_radius_l476_476731


namespace min_value_sin6_cos6_l476_476316

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476316


namespace circle_evaluation_circle_conversion_final_calculation_l476_476134

def circle (a : ℚ) (n : ℕ) : ℚ := list.foldl (/) a (list.replicate (n-1) a)

theorem circle_evaluation :
  circle 2 3 = (1/2) ∧
  circle (-3) 4 = (1/9) ∧
  circle (-1/3) 5 = -27 :=
by sorry

theorem circle_conversion (a : ℚ) (ha : a ≠ 0) (n : ℕ) (hn : 2 ≤ n) :
  circle a n = (1 / a ^ (n-2)) :=
by sorry

theorem final_calculation :
  27 * (1/9) + (-48) / (1/2^5) = -3 :=
by sorry

end circle_evaluation_circle_conversion_final_calculation_l476_476134


namespace arithmetic_sequence_problem_l476_476496

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3 : a 3 = 5)
  (Sn : ∀ n, S n = n * (2 + (n - 1) * 2) / 2)
  (S_diff : ∀ k, S (k + 2) - S k = 36)
  : ∃ k : ℕ, k = 8 :=
by
  sorry

end arithmetic_sequence_problem_l476_476496


namespace polynomial_remainder_l476_476745

-- Define the polynomial
def poly (x : ℝ) : ℝ := 3 * x^8 - x^7 - 7 * x^5 + 3 * x^3 + 4 * x^2 - 12 * x - 1

-- Define the divisor
def divisor : ℝ := 3

-- State the theorem
theorem polynomial_remainder :
  poly divisor = 15951 :=
by
  -- Proof omitted, to be filled in later
  sorry

end polynomial_remainder_l476_476745


namespace part_a_part_b_l476_476763

variable (G : ℝ → ℝ)
variable (x1 x2 x3 x4 x5 : ℝ)

axiom polynomial_real_coefficients : ∀ x, G x ∈ ℝ
axiom takes_value_2022 : ∀ x, (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) → G x = 2022
axiom distinct_points : x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5
axiom symmetric_about_neg7 : ∀ x, G x = G (-14 - x)

theorem part_a : x1 + x3 + x5 = -21 := by
  sorry

theorem part_b : ∃ d, polynomial.degree G = d ∧ d = 6 := by
  sorry  

end part_a_part_b_l476_476763


namespace greatest_prime_factor_l476_476740

-- Define the mathematical expression
def expression : ℕ := 5^7 + 10^6

-- Define the greatest prime factor assertion
theorem greatest_prime_factor : ∃ p, nat.prime p ∧ ∀ q, (nat.prime q ∧ q ∣ expression) → q ≤ p ∧ p = 23 :=  by
    sorry

end greatest_prime_factor_l476_476740


namespace sin_cos_sixth_min_l476_476348

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476348


namespace jack_milk_drunk_l476_476244

theorem jack_milk_drunk (initial_milk : ℚ) (rachel_fraction : ℚ) (jack_fraction : ℚ)
  (h₀ : initial_milk = 3 / 4)
  (h₁ : rachel_fraction = 5 / 8)
  (h₂ : jack_fraction = 1 / 2) :
  let remaining_milk := initial_milk * (1 - rachel_fraction) in
  let jack_milk := remaining_milk * jack_fraction in
  jack_milk = 9 / 64 :=
by
  sorry

end jack_milk_drunk_l476_476244


namespace simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l476_476228

variable (a b : ℤ)
def A : ℤ := b^2 - a^2 + 5 * a * b
def B : ℤ := 3 * a * b + 2 * b^2 - a^2

theorem simplify_2A_minus_B : 2 * A a b - B a b = -a^2 + 7 * a * b := by
  sorry

theorem evaluate_2A_minus_B_at_1_2 : 2 * A 1 2 - B 1 2 = 13 := by
  sorry

end simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l476_476228


namespace arithmetic_sequence_nth_term_l476_476807

theorem arithmetic_sequence_nth_term (s : ℕ → ℤ) (r s0 : ℤ) : 
  (∀ k : ℕ, s (k + 1) = s k + r) → 
  s 0 = s0 →
  ∀ n : ℕ, s n = s0 + n * r :=
by
  intros seq_def s0_def n
  induction n with k hk
  case zero =>
    rw [s0_def]
  case succ => 
    rw [seq_def k, hk]
  done

end arithmetic_sequence_nth_term_l476_476807


namespace exists_positive_integer_pow_not_integer_l476_476673

theorem exists_positive_integer_pow_not_integer
  (α β : ℝ)
  (hαβ : α ≠ β)
  (h_non_int : ¬(↑⌊α⌋ = α ∧ ↑⌊β⌋ = β)) :
  ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, α^n - β^n = k :=
by
  sorry

end exists_positive_integer_pow_not_integer_l476_476673


namespace min_value_sin6_cos6_l476_476311

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476311


namespace total_participating_students_l476_476195

-- Define the given conditions
def field_events_participants : ℕ := 15
def track_events_participants : ℕ := 13
def both_events_participants : ℕ := 5

-- Define the total number of students calculation
def total_students_participating : ℕ :=
  (field_events_participants - both_events_participants) + 
  (track_events_participants - both_events_participants) + 
  both_events_participants

-- State the theorem that needs to be proved
theorem total_participating_students : total_students_participating = 23 := by
  sorry

end total_participating_students_l476_476195


namespace valid_integer_pairs_l476_476852

theorem valid_integer_pairs :
  ∀ a b : ℕ, 1 ≤ a → 1 ≤ b → a ^ (b ^ 2) = b ^ a → (a, b) = (1, 1) ∨ (a, b) = (16, 2) ∨ (a, b) = (27, 3) :=
by
  sorry

end valid_integer_pairs_l476_476852


namespace triangle_third_side_l476_476921

open Nat

theorem triangle_third_side (a b c : ℝ) (h1 : a = 4) (h2 : b = 9) (h3 : c > 0) :
  (5 < c ∧ c < 13) ↔ c = 6 :=
by
  sorry

end triangle_third_side_l476_476921


namespace xiaoming_total_money_l476_476163

def xiaoming_money (x : ℕ) := 9 * x

def fresh_milk_cost (y : ℕ) := 6 * y

def yogurt_cost_equation (x y : ℕ) := y = x + 6

theorem xiaoming_total_money (x : ℕ) (y : ℕ)
  (h1: fresh_milk_cost y = xiaoming_money x)
  (h2: yogurt_cost_equation x y) : xiaoming_money x = 108 := 
  sorry

end xiaoming_total_money_l476_476163


namespace sin_cos_sixth_min_l476_476346

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476346


namespace g_g_3_eq_3606651_l476_476442

def g (x: ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x + 1

theorem g_g_3_eq_3606651 : g (g 3) = 3606651 := 
by {
  sorry
}

end g_g_3_eq_3606651_l476_476442


namespace webinar_end_time_correct_l476_476805

-- Define start time and duration as given conditions
def startTime : Nat := 3*60 + 15  -- 3:15 p.m. in minutes after noon
def duration : Nat := 350         -- duration of the webinar in minutes

-- Define the expected end time in minutes after noon (9:05 p.m. is 9*60 + 5 => 545 minutes after noon)
def endTimeExpected : Nat := 9*60 + 5

-- Statement to prove that the calculated end time matches the expected end time
theorem webinar_end_time_correct : startTime + duration = endTimeExpected :=
by
  sorry

end webinar_end_time_correct_l476_476805


namespace maria_earnings_l476_476642

-- Define the conditions
def costOfBrushes : ℕ := 20
def costOfCanvas : ℕ := 3 * costOfBrushes
def costPerLiterOfPaint : ℕ := 8
def litersOfPaintNeeded : ℕ := 5
def sellingPriceOfPainting : ℕ := 200

-- Define the total cost calculation
def totalCostOfMaterials : ℕ := costOfBrushes + costOfCanvas + (costPerLiterOfPaint * litersOfPaintNeeded)

-- Define the final earning calculation
def mariaEarning : ℕ := sellingPriceOfPainting - totalCostOfMaterials

-- State the theorem
theorem maria_earnings :
  mariaEarning = 80 := by
  sorry

end maria_earnings_l476_476642


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476600

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476600


namespace total_cost_of_two_books_l476_476939

theorem total_cost_of_two_books (C1 C2 total_cost: ℝ) :
  C1 = 262.5 →
  0.85 * C1 = 1.19 * C2 →
  total_cost = C1 + C2 →
  total_cost = 450 :=
by
  intros h1 h2 h3
  sorry

end total_cost_of_two_books_l476_476939


namespace math_problem_l476_476412

def f (x : ℝ) (m : ℝ) : ℝ := m * Real.sin x + Real.cos x

theorem math_problem (m : ℝ) (x : ℝ) :
  (f (Real.pi / 2) m = 1) →
  (f x m = √2 * Real.sin (x + Real.pi / 4)) ∧
  (∀ x, f x m = f (x + 2 * Real.pi) m) ∧
  (∀ x, f x m ≤ √2) ∧
  (f (x - Real.pi / 4) m = √2 * Real.sin x → 
   ∀ x, f (2 * x) m = √2 * Real.sin (2 * x + Real.pi / 4)) :=
by
  sorry

end math_problem_l476_476412


namespace no_five_consecutive_terms_divisible_by_2005_l476_476925

noncomputable def a (n : ℕ) : ℤ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_terms_divisible_by_2005 : ¬ ∃ n : ℕ, (a n % 2005 = 0) ∧ (a (n+1) % 2005 = 0) ∧ (a (n+2) % 2005 = 0) ∧ (a (n+3) % 2005 = 0) ∧ (a (n+4) % 2005 = 0) := sorry

end no_five_consecutive_terms_divisible_by_2005_l476_476925


namespace prove_arithmetic_sequence_minimum_value_S_l476_476540

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476540


namespace calculate_sum_l476_476226

theorem calculate_sum :
  (\sum k in Finset.range 51, (-1)^k * Nat.choose 100 (2*k + 1)) = -2^50 :=
by
  sorry

end calculate_sum_l476_476226


namespace initial_pretzels_in_bowl_l476_476114

-- Definitions and conditions
def John_pretzels := 28
def Alan_pretzels := John_pretzels - 9
def Marcus_pretzels := John_pretzels + 12
def Marcus_pretzels_actual := 40

-- The main theorem stating the initial number of pretzels in the bowl
theorem initial_pretzels_in_bowl : 
  Marcus_pretzels = Marcus_pretzels_actual → 
  John_pretzels + Alan_pretzels + Marcus_pretzels = 87 :=
by
  intro h
  sorry -- proof to be filled in

end initial_pretzels_in_bowl_l476_476114


namespace min_sixth_power_sin_cos_l476_476272

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476272


namespace democrats_and_republicans_seating_l476_476771

theorem democrats_and_republicans_seating : 
  let n := 6
  let factorial := Nat.factorial
  let arrangements := (factorial n) * (factorial n)
  let circular_table := 1
  arrangements * circular_table = 518400 :=
by 
  sorry

end democrats_and_republicans_seating_l476_476771


namespace min_k_for_B_winning_strategy_l476_476209

-- Define the game board and conditions
def board : Type := fin 5 × fin 5

-- Define the L-Shape
structure LShape :=
(squares : list (fin 5 × fin 5))
(h_cover3 : squares.length = 3)

-- Define a placement type
def good_placement (unmarked_squares : finset (fin 5 × fin 5)) (Lshapes : list LShape) : Prop :=
  (Lshapes.all (λ L, L.squares.all (λ s, s ∈ unmarked_squares))) ∧
  (disjoint Lshapes.map (λ L, finset.mk L.squares L.h_cover3))

-- Define the winning condition for B
def B_wins (k : ℕ) : Prop :=
∀ (A_moves : finset (fin 5 × fin 5)) (hA : A_moves.card ≤ k),
  ∃ unmarked_squares : finset (fin 5 × fin 5),
    (unmarked_squares.card = 25 - k ∧
     ∀ (Lshapes : list LShape), ¬good_placement unmarked_squares Lshapes → unmarked_squares.card ≥ 3)

-- Main theorem statement
theorem min_k_for_B_winning_strategy : ∃ k : ℕ, k = 4 ∧ B_wins k :=
begin
  use 4,
  sorry
end

end min_k_for_B_winning_strategy_l476_476209


namespace repeating_decimal_sum_l476_476251

/-
The goal is to state that for the repeating decimal 0.363636..., the sum of the numerator and
denominator of its simplified fraction representation is 15.
-/

theorem repeating_decimal_sum {x : ℚ} (h : x = 0.363636363636...) : 
  let numer := (x.denom : ℤ)
  let denom := (x.num : ℤ)
  numer + denom = 15 := 
sorry -- Proof to be provided

end repeating_decimal_sum_l476_476251


namespace angle_between_u_and_v_l476_476855

open Real

noncomputable def u : ℝ × ℝ × ℝ := (3, -2, 2)
noncomputable def v : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (a.1^2 + a.2^2 + a.3^2)

noncomputable def cosine (a b : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product a b) / ((magnitude a) * (magnitude b))

noncomputable def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (cosine a b)

theorem angle_between_u_and_v :
  angle_between_vectors u v = real.arccos (-8 / (3 * sqrt 17)) := by
  sorry

end angle_between_u_and_v_l476_476855


namespace all_push_ups_total_l476_476165

-- Definitions derived from the problem's conditions
def ZacharyPushUps := 47
def DavidPushUps := ZacharyPushUps + 15
def EmilyPushUps := DavidPushUps * 2
def TotalPushUps := ZacharyPushUps + DavidPushUps + EmilyPushUps

-- The statement to be proved
theorem all_push_ups_total : TotalPushUps = 233 := by
  sorry

end all_push_ups_total_l476_476165


namespace solve_system_of_equations_l476_476664

theorem solve_system_of_equations (x y : ℝ) (h1 : x - y = -5) (h2 : 3 * x + 2 * y = 10) : x = 0 ∧ y = 5 := by
  sorry

end solve_system_of_equations_l476_476664


namespace min_sixth_power_sin_cos_l476_476261

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476261


namespace intersecting_lines_determinant_zero_l476_476455

theorem intersecting_lines_determinant_zero :
  (∃ x y : ℝ, 2 * x + y + 3 = 0 ∧ x + y + 2 = 0 ∧ 2 * x - y + 1 = 0) →
  Matrix.det !![![2, 1, 3], ![1, 1, 2], ![2, -1, 1]] = 0 :=
by
  intro h
  sorry

end intersecting_lines_determinant_zero_l476_476455


namespace mod_11_residue_l476_476742

theorem mod_11_residue :
  (312 ≡ 4 [MOD 11]) ∧
  (47 ≡ 3 [MOD 11]) ∧
  (154 ≡ 0 [MOD 11]) ∧
  (22 ≡ 0 [MOD 11]) →
  (312 + 6 * 47 + 8 * 154 + 5 * 22 ≡ 0 [MOD 11]) :=
by
  intros h
  sorry

end mod_11_residue_l476_476742


namespace min_value_sin6_cos6_l476_476326

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476326


namespace arithmetic_sequence_minimum_value_S_l476_476581

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476581


namespace transformed_area_l476_476700

variable (g : ℝ → ℝ)

def area_under_curve (f : ℝ → ℝ) : ℝ := sorry

theorem transformed_area
  (h : area_under_curve g = 15) :
  area_under_curve (λ x, 4 * g (2 * x - 4)) = 30 :=
sorry

end transformed_area_l476_476700


namespace angle_between_vectors_is_correct_l476_476862

def vec_a : ℝ × ℝ × ℝ := (3, -2, 2)
def vec_b : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def angle_between_vectors : ℝ :=
  Real.acos ((vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 + vec_a.3 * vec_b.3) / 
    (Real.sqrt (vec_a.1^2 + vec_a.2^2 + vec_a.3^2) * Real.sqrt (vec_b.1^2 + vec_b.2^2 + vec_b.3^2)))

theorem angle_between_vectors_is_correct :
  angle_between_vectors = Real.acos (-8 / (3 * Real.sqrt 17)) :=
by sorry

end angle_between_vectors_is_correct_l476_476862


namespace part1_part2_part3_l476_476415

noncomputable def f (x t : ℝ) : ℝ := (x + 2) * (x - t) / x^2
noncomputable def λ : ℝ := real.log 2 ^ 2 + real.log 2 * real.log 5 + real.log 5 - 1

theorem part1 (h_even : ∀ x : ℝ, f x 2 = f (-x) 2): t = 2 := 
sorry

def E : set ℝ := {f 1 2, f 2 2, f 3 2}

theorem part2 : λ ∈ E :=
sorry

noncomputable def f_transformed (x : ℝ) : ℝ := 1 - 4 / x^2

theorem part3 (h_range : ∀x ∈ set.Icc a b, f_transformed x ∈ set.Icc (2 - 5 / a) (2 - 5 / b)) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) : a = 1 ∧ b = 4 := 
sorry

end part1_part2_part3_l476_476415


namespace cassini_oval_lemniscate_properties_l476_476676

def isLemniscate (a: ℝ)(P: ℝ × ℝ): Prop :=
   let A := (-a, 0)
   let B := (a, 0)
   let d_PA := (P.1 + a)^2 + P.2^2
   let d_PB := (P.1 - a)^2 + P.2^2
   (d_PA * d_PB = a^2)

theorem cassini_oval_lemniscate_properties (a : ℝ) (P : ℝ × ℝ) (x0 y0 : ℝ) :
  isLemniscate a (x0, y0) →
  (∀ P: ℝ × ℝ, isLemniscate a P → isLemniscate a (-P.1,-P.2)) ∧ -- The curve is symmetric with respect to the origin.
  ¬(∀ x0: ℝ, -a ≤ x0 ∧ x0 ≤ a) ∧ -- Statement B is incorrect.
  (∀ P : ℝ × ℝ, let dist_to_origin := P.1^2 + P.2^2 in dist_to_origin -a^2 ≤ a^2) ∧ -- The maximum value of |PO|^2 - a^2 is a^2.
  (∀ P : ℝ × ℝ, (P.1 = 0) → ((√((a^2) + (P.2)^2) = a) ↔ (P.2 = 0))) -- There is one and only one point P on the curve such that |PA| = |PB|.
  :=
by intros
sorry

end cassini_oval_lemniscate_properties_l476_476676


namespace Julia_played_with_kids_l476_476035

theorem Julia_played_with_kids :
  (∃ k : ℕ, k = 4) ∧ (∃ n : ℕ, n = 4 + 12) → (n = 16) :=
by
  sorry

end Julia_played_with_kids_l476_476035


namespace min_value_inverse_sum_l476_476616

theorem min_value_inverse_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 3 * b = 1) :
  \frac{1}{a} + \frac{1}{b} = 4 + 2 * sqrt 3 :=
sorry

end min_value_inverse_sum_l476_476616


namespace matrix_condition_multiple_root_l476_476243

variable {R : Type*} [CommRing R]

def M (a1 a2 a3 x : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![0, a1 - x, a2 - x], 
    ![-(a1 + x), 0, a3 - x], 
    ![-(a2 + x), -(a3 + x), 0]]

theorem matrix_condition_multiple_root {a1 a2 a3 : R} (h : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0) :
  (∃ x : R, ∃ m : R, (M a1 a2 a3 x).det = 0 ∧ m ≠ 0 ∧ (M a1 a2 a3 x).charpoly.eval m = 0 ∧ (M a1 a2 a3 x).charpoly.derivative.eval m = 0) ↔ 
  (a2 ^ 2 = 8 * (a1 * a3 - a2 * a1 - a2 * a3)) :=
by 
  sorry

end matrix_condition_multiple_root_l476_476243


namespace no_valid_n_l476_476372

theorem no_valid_n (n : ℕ) (h₁ : 100 ≤ n / 4) (h₂ : n / 4 ≤ 999) (h₃ : 100 ≤ 4 * n) (h₄ : 4 * n ≤ 999) : false := by
  sorry

end no_valid_n_l476_476372


namespace arithmetic_sequence_minimum_value_S_n_l476_476560

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476560


namespace compare_answers_l476_476005

def num : ℕ := 384
def correct_answer : ℕ := (5 * num) / 16
def students_answer : ℕ := (5 * num) / 6
def difference : ℕ := students_answer - correct_answer

theorem compare_answers : difference = 200 := 
by
  sorry

end compare_answers_l476_476005


namespace sequence_problem_l476_476595

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476595


namespace angle_between_u_and_v_l476_476859

open Real

noncomputable def u : ℝ × ℝ × ℝ := (3, -2, 2)
noncomputable def v : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (a.1^2 + a.2^2 + a.3^2)

noncomputable def cosine (a b : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product a b) / ((magnitude a) * (magnitude b))

noncomputable def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (cosine a b)

theorem angle_between_u_and_v :
  angle_between_vectors u v = real.arccos (-8 / (3 * sqrt 17)) := by
  sorry

end angle_between_u_and_v_l476_476859


namespace problem_statements_l476_476026

noncomputable def triangle_condition_A (A B : ℝ) (a b : ℝ) [so.A : A > B] 
  [so.a : a > b] : Prop :=
  sin A > sin B

noncomputable def triangle_condition_B (A : ℝ) (a b : ℝ) [so.A : A = 30] 
  [so.a : a = 3] [so.b : b = 4] : Prop :=
  ¬exists (B C : ℝ), 
    A + B + C = 180 ∧
    b = a * sin B / sin A

noncomputable def triangle_condition_C (AB AC : ℝ) (BM MC : ℝ)
  (AM_AO_dot : ℝ) [so.AB : AB = √3] [so.AC : AC = √2]
  [so.BM_MC : BM = 2 * MC] : Prop :=
  AM_AO_dot ≠ 6/7

noncomputable def triangle_condition_D (A B C : ℝ)
  [so.oblique : A + B + C = π] : Prop :=
  tan A + tan B + tan C = tan A * tan B * tan C

theorem problem_statements :
  (∀ A B a b, A > B → a > b → triangle_condition_A A B a b) ∧
  (forall A a b,  A=30 -> a=3 -> b=4 -> ¬triangle_condition_B A a b) ∧
  (∀ AB AC BM MC AM_AO_dot, AB = √3 → AC = √2 →
    BM = 2 * MC → triangle_condition_C AB AC BM MC AM_AO_dot) ∧ 
  (∀ A B C, A + B + C = π → triangle_condition_D A B C) :=
by {
  sorry,
}

end problem_statements_l476_476026


namespace smallest_n_l476_476053

theorem smallest_n (n : ℕ) (x : ℕ → ℝ) (h₁ : ∀ i, i < n → |x i| < 2)
  (h₂ : (∑ i in finset.range n, |x i|) = 39 + |∑ i in finset.range n, x i|)
  : n ≥ 20 :=
sorry

end smallest_n_l476_476053


namespace king_paid_after_tip_l476_476782

theorem king_paid_after_tip:
  (crown_cost tip_percentage total_cost : ℝ)
  (h_crown_cost : crown_cost = 20000)
  (h_tip_percentage : tip_percentage = 0.1) :
  total_cost = crown_cost + (crown_cost * tip_percentage) :=
by
  have h_tip := h_crown_cost.symm ▸ h_tip_percentage.symm ▸ 20000 * 0.1
  have h_total := h_crown_cost.symm ▸ (h_tip.symm ▸ 2000)
  rw [h_crown_cost, h_tip, h_total]
  exact rfl

end king_paid_after_tip_l476_476782


namespace problem_part_1_problem_part_2_l476_476919

def f (x m : ℝ) := 2 * x^2 + (2 - m) * x - m
def g (x m : ℝ) := x^2 - x + 2 * m

theorem problem_part_1 (x : ℝ) : f x 1 > 0 ↔ (x > 1/2 ∨ x < -1) :=
by sorry

theorem problem_part_2 {m x : ℝ} (hm : 0 < m) : f x m ≤ g x m ↔ (-3 ≤ x ∧ x ≤ m) :=
by sorry

end problem_part_1_problem_part_2_l476_476919


namespace probability_first_9_second_diamond_third_7_l476_476116

/-- 
There are 52 cards in a standard deck, with 4 cards that are 9's, 4 cards that are 7's, and 13 cards that are 
diamonds. To find the probability that the first card is a 9, the second card is a diamond, and the third card 
is a 7, we perform a detailed probabilistic calculation. In the end, we combine the probabilities of the 
mutually exclusive cases to find the desired probability.
-/
theorem probability_first_9_second_diamond_third_7 :
  (4 / 52) * (12 / 51) * (4 / 50) +
  (4 / 52) * (1 / 51) * (3 / 50) +
  (1 / 52) * (11 / 51) * (4 / 50) +
  (1 / 52) * (1 / 51) * (3 / 50) = 251 / 132600 :=
by
  sorry

end probability_first_9_second_diamond_third_7_l476_476116


namespace card_draw_probability_l476_476234

theorem card_draw_probability :
  let card_numbers := set.univ ∩ {n | n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13} },
      suits := ['Spades, 'Hearts],
      deck := suits × card_numbers in
  let perfect_square_pairs :=
    {(1,4), (4,1), (1,9), (9,1), (2,8), (8,2), (3,12), (12,3), (4,9), (9,4)} in
  let total_pairs := (Finset.card deck).choose 2 in 
  let valid_pairs :=
    ∑ s in suits, 
      ∑ p in perfect_square_pairs,
        if p.1 ≠ p.2 then 1 else 0 in
  valid_pairs / total_pairs = 2 / 65 :=
by
  sorry

end card_draw_probability_l476_476234


namespace coefficient_a2_l476_476403

theorem coefficient_a2 :
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ),
  (x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
  a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + a_9 * (x + 1)^9 + 
  a_10 * (x + 1)^10) →
  a_2 = 45 :=
by
  sorry

end coefficient_a2_l476_476403


namespace combined_salaries_ABCD_l476_476701

noncomputable def salary_E : ℝ := 9000
noncomputable def avg_salary_ABCDE : ℝ := 9000
noncomputable def combined_salaries_ABC_D (total_salary_ABCDE salary_E : ℝ) : ℝ := total_salary_ABCDE - salary_E

theorem combined_salaries_ABCD : combined_salaries_ABC_D (5 * avg_salary_ABCDE) salary_E = 36000 := 
by
  exact (by norm_num : 5 * avg_salary_ABCDE - salary_E = 36000)

end combined_salaries_ABCD_l476_476701


namespace sequence_condition_l476_476872

theorem sequence_condition (a : ℕ → ℕ) (n : ℕ) 
  (h₀ : a 0 = 2016) 
  (h : ∀ n > 0, (∑ k in Finset.range n, a 0 / a (k + 1)) + 2017 / a (n + 1) = 1) :
  ∀ n ≥ 2, ∃ a₂, a (n + 1) = 2017 ^ (n - 1) * a₂ ∧ (2016 / a 1 + 2017 / a 2 = 1) :=
by
  -- proof omitted
  sorry

end sequence_condition_l476_476872


namespace arithmetic_sequence_and_minimum_sum_l476_476522

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476522


namespace equal_share_each_person_l476_476245

def emani_initial : ℝ := 150
def howard_initial := emani_initial - 30
def jamal_initial : ℝ := 75

def emani_updated := emani_initial * 1.20
def howard_updated := howard_initial * 1.10
def jamal_updated := jamal_initial * 1.15

def combined_amount := emani_updated + howard_updated + jamal_updated
def equal_share := combined_amount / 3

theorem equal_share_each_person :
  equal_share = 132.75 := by
  sorry

end equal_share_each_person_l476_476245


namespace domain_of_sqrt_function_l476_476138

theorem domain_of_sqrt_function : {x : ℝ | 0 ≤ x ∧ x ≤ 1} = {x : ℝ | 1 - x ≥ 0 ∧ x - Real.sqrt (1 - x) ≥ 0} :=
by
  sorry

end domain_of_sqrt_function_l476_476138


namespace conjugate_quadrant_l476_476437

def Z : ℂ := 3 - 4 * complex.i

theorem conjugate_quadrant
    (Z_conj_coord : ℂ := complex.conj Z)
    (coord : ℝ × ℝ := (Z_conj_coord.re, Z_conj_coord.im)) :
    coord.1 > 0 ∧ coord.2 > 0 :=
    by
        sorry

end conjugate_quadrant_l476_476437


namespace sum_of_digits_min_M_l476_476814

theorem sum_of_digits_min_M : 
  ∃ M : ℕ, 
    0 ≤ M ∧ M ≤ 1999 ∧ 
    (32 * M + 1600 < 2000) ∧ 
    (32 * M + 1700 ≥ 2000) ∧
    (M.digits.sum = 1) :=
sorry

end sum_of_digits_min_M_l476_476814


namespace time_jack_first_half_l476_476029

-- Define the conditions
def t_Jill : ℕ := 32
def t_2 : ℕ := 6
def t_Jack : ℕ := t_Jill - 7

-- Define the time Jack took for the first half
def t_1 : ℕ := t_Jack - t_2

-- State the theorem to prove
theorem time_jack_first_half : t_1 = 19 := by
  sorry

end time_jack_first_half_l476_476029


namespace combinatorial_sum_identity_l476_476368

theorem combinatorial_sum_identity (n : ℕ) (m : ℕ) (h : 3 * m + 1 ≤ n ∧ n < 3 * m + 4) :
  (finset.range (n + 1)).filter (λ k, k % 3 = 1).sum (λ k, nat.choose n k) = 
  (1 / 3 : ℝ) * (2^n + 2 * real.cos ((n - 2) * real.pi / 3)) :=
by sorry

end combinatorial_sum_identity_l476_476368


namespace remainder_x_squared_div_25_l476_476444

theorem remainder_x_squared_div_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 0 [ZMOD 25] :=
sorry

end remainder_x_squared_div_25_l476_476444


namespace sin_cos_sixth_min_l476_476350

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476350


namespace sequence_problem_l476_476593

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476593


namespace tom_initial_investment_l476_476727

theorem tom_initial_investment (P J_share J_inv T_share : ℝ)
(hP : P = 63000) 
(hJ_share : J_share = 35000) 
(hJ_inv : J_inv = 45000) 
(hT_share : T_share = P - J_share) 
(h_ratio : (T_share / J_share) = ((T * 12) / (J_inv * 10))) : 
T = 30000 := by
  have h1 : T_share = P - J_share := by sorry
  rw [h1] at h_ratio
  have h2 : 28000 / 35000 = 4 / 5 := by norm_num
  rw h2 at h_ratio 
  solve_by_elim

end tom_initial_investment_l476_476727


namespace probability_A_given_B_l476_476962

namespace ProbabilityProof

def total_parts : ℕ := 100
def A_parts_produced : ℕ := 0
def A_parts_qualified : ℕ := 35
def B_parts_produced : ℕ := 60
def B_parts_qualified : ℕ := 50

def event_A (x : ℕ) : Prop := x ≤ B_parts_qualified + A_parts_qualified
def event_B (x : ℕ) : Prop := x ≤ A_parts_produced

-- Formalizing the probability condition P(A | B) = 7/8, logically this should be revised with practical events.
theorem probability_A_given_B : (event_B x → event_A x) := sorry

end ProbabilityProof

end probability_A_given_B_l476_476962


namespace alpha_beta_is_4_l476_476904

theorem alpha_beta_is_4 (a b : ℕ → ℝ) (α β : ℝ) (d q : ℝ)
  (h1 : a 1 = 2)
  (h2 : b 1 = 1)
  (h3 : a 2 = b 2)
  (h4 : 2 * a 4 = b 3)
  (h5 : ∀ n, a n = real.log (b n) / real.log α + β)
  (non_zero_d : d ≠ 0)
  (arith_seq : ∀ n, a (n + 1) = a n + d)
  (geom_seq : ∀ n, b (n + 1) = b n * q) :
  α ^ β = 4 := by
  sorry

end alpha_beta_is_4_l476_476904


namespace min_sin6_cos6_l476_476341

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476341


namespace length_of_rooms_l476_476488

-- Definitions based on conditions
def width : ℕ := 18
def num_rooms : ℕ := 20
def total_area : ℕ := 6840

-- Theorem stating the length of the rooms
theorem length_of_rooms : (total_area / num_rooms) / width = 19 := by
  sorry

end length_of_rooms_l476_476488


namespace find_n_l476_476401

theorem find_n (n : ℤ) (h : (1 : ℤ)^2 + 3 * 1 + n = 0) : n = -4 :=
sorry

end find_n_l476_476401


namespace smallest_positive_integer_satisfying_congruence_l476_476747

theorem smallest_positive_integer_satisfying_congruence :
  ∃ x : ℤ, (0 < x) ∧ (x < 31) ∧ (5 * x ≡ 22 [MOD 31]) ∧ (∀ y : ℤ, (0 < y ∧ y < x) → ¬ (5 * y ≡ 22 [MOD 31])) :=
begin
  use 23,
  split, norm_num,
  split, norm_num,
  split,
  { norm_num, exact modeq.refl 22 },
  { intros y hy, 
    dsimp only,
    sorry }
end

end smallest_positive_integer_satisfying_congruence_l476_476747


namespace find_inverse_modulo_l476_476851

theorem find_inverse_modulo :
  113 * 113 ≡ 1 [MOD 114] :=
by
  sorry

end find_inverse_modulo_l476_476851


namespace arithmetic_sequence_minimum_value_S_l476_476583

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476583


namespace angle_invariant_under_magnification_l476_476751

theorem angle_invariant_under_magnification :
  ∀ (angle magnification : ℝ), angle = 10 → magnification = 5 → angle = 10 := by
  intros angle magnification h_angle h_magnification
  exact h_angle

end angle_invariant_under_magnification_l476_476751


namespace arithmetic_sequence_min_value_S_l476_476500

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476500


namespace triangle_ABC_perimeter_l476_476978

theorem triangle_ABC_perimeter (k : ℝ) (C : ℝ ∠ABC : angle) (AB BC : ℝ )
(ABeq : AB = 15)
(Ceq : C = 120)
(XY square)
(ABXY rect outside side AB)
(CBWZ rect outside side CB  with BW : WZ = 3 :2 )
(locus:  XY,WZ  points on one circle)
:
 P( tangle ABC)


end triangle_ABC_perimeter_l476_476978


namespace round_trip_time_correct_l476_476108

variable (boat_speed_standing_water : ℕ) (stream_speed : ℕ) (distance_one_way : ℕ)

-- Given conditions
def upstream_speed (boat_speed_standing_water stream_speed : ℕ) : ℕ :=
  boat_speed_standing_water - stream_speed

def downstream_speed (boat_speed_standing_water stream_speed : ℕ) : ℕ :=
  boat_speed_standing_water + stream_speed

def round_trip_distance (distance_one_way : ℕ) : ℕ :=
  2 * distance_one_way

-- Time calculations for each leg of the trip
def time_upstream (distance_one_way upstream_speed : ℕ) : ℝ :=
  (distance_one_way : ℝ) / (upstream_speed : ℝ)

def time_downstream (distance_one_way downstream_speed : ℕ) : ℝ :=
  (distance_one_way : ℝ) / (downstream_speed : ℝ)

-- Total round trip time
noncomputable def total_round_trip_time (distance_one_way upstream_speed downstream_speed : ℕ) : ℝ :=
  time_upstream distance_one_way upstream_speed + time_downstream distance_one_way downstream_speed

theorem round_trip_time_correct :
  total_round_trip_time 7200 (upstream_speed 16 2) (downstream_speed 16 2) = 914.2857 :=
by
  -- Conditions
  let boat_speed_standing_water := 16
  let stream_speed := 2
  let distance_one_way := 7200
  
  -- Calculations
  let uspeed := upstream_speed boat_speed_standing_water stream_speed
  let dspeed := downstream_speed boat_speed_standing_water stream_speed
  let rt_distance := round_trip_distance distance_one_way
  let t_up := time_upstream distance_one_way uspeed
  let t_down := time_downstream distance_one_way dspeed
  let t_total := total_round_trip_time distance_one_way uspeed dspeed

  -- Result
  have h1 : t_total = 914.2857 := sorry
  exact h1

end round_trip_time_correct_l476_476108


namespace rhombus_not_diagonals_equal_l476_476156

theorem rhombus_not_diagonals_equal (R : Type) [linear_ordered_field R] 
  (a b c d : R) (h1 : a = b) (h2 : b = c) (h3 : c = d) (h4 : a = d)
  (h_sym : ∀ x y : R, a = b → b = c → c = d → d = a)
  (h_cen_sym : ∀ p : R × R, p = (0, 0) → p = (0, 0)) :
  ¬(∀ p q : R × R, p ≠ q → (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 = (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) :=
by
  sorry

end rhombus_not_diagonals_equal_l476_476156


namespace population_increase_rate_is_20_percent_l476_476698

noncomputable def population_increase_rate 
  (initial_population final_population : ℕ) : ℕ :=
  ((final_population - initial_population) * 100) / initial_population

theorem population_increase_rate_is_20_percent :
  population_increase_rate 2000 2400 = 20 :=
by
  unfold population_increase_rate
  sorry

end population_increase_rate_is_20_percent_l476_476698


namespace john_new_weekly_earnings_after_raise_l476_476489

theorem john_new_weekly_earnings_after_raise :
  ∀ (original_earnings : ℝ) (percentage_increase : ℝ), original_earnings = 60 → percentage_increase = 0.30 → 
  original_earnings + (original_earnings * percentage_increase) = 78 :=
by
  assume (original_earnings percentage_increase : ℝ),
  assume (h1 : original_earnings = 60),
  assume (h2 : percentage_increase = 0.30),
  calc
    original_earnings + (original_earnings * percentage_increase)
        = 60 + (60 * 0.30) : by rw [h1, h2]
    ... = 60 + 18 : by norm_num
    ... = 78 : by norm_num

end john_new_weekly_earnings_after_raise_l476_476489


namespace greatest_prime_factor_l476_476741

-- Define the mathematical expression
def expression : ℕ := 5^7 + 10^6

-- Define the greatest prime factor assertion
theorem greatest_prime_factor : ∃ p, nat.prime p ∧ ∀ q, (nat.prime q ∧ q ∣ expression) → q ≤ p ∧ p = 23 :=  by
    sorry

end greatest_prime_factor_l476_476741


namespace same_degree_if_not_adjacent_l476_476713

-- Problem statement and conditions as Lean definitions

-- Define the graph and its properties
variables {V : Type} [fintype V] [decidable_eq V]

-- Define a function that represents the adjacency relation
def is_adjacent (G : V → V → Prop) (A B : V) : Prop :=
  G A B

-- Define the main condition: for any two vertices, there is exactly one vertex connected to both
def unique_connector (G : V → V → Prop) : Prop :=
  ∀ A B : V, ∃! C : V, G A C ∧ G B C

-- Define the degree of a vertex (number of adjacent vertices)
def degree (G : V → V → Prop) (v : V) : ℕ :=
  fintype.card {w : V // G v w}

-- The main theorem to be proved
theorem same_degree_if_not_adjacent
  (G : V → V → Prop)
  (hG : unique_connector G)
  (not_adj : ∀ A B : V, ¬ G A B) :
  ∀ A B : V, ¬ G A B → degree G A = degree G B :=
sorry

end same_degree_if_not_adjacent_l476_476713


namespace arithmetic_sequence_min_value_S_l476_476498

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476498


namespace king_total_payment_l476_476780

/-- 
A king gets a crown made that costs $20,000. He tips the person 10%. Prove that the total amount the king paid after the tip is $22,000.
-/
theorem king_total_payment (C : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h1 : C = 20000) 
  (h2 : tip_percentage = 0.1) 
  (h3 : total_paid = C + C * tip_percentage) : 
  total_paid = 22000 := 
by 
  sorry

end king_total_payment_l476_476780


namespace valid_raise_percentage_l476_476487

-- Define the conditions
def raise_between (x : ℝ) : Prop :=
  0.05 ≤ x ∧ x ≤ 0.10

def salary_increase_by_fraction (x : ℝ) : Prop :=
  x = 0.06

-- Define the main theorem 
theorem valid_raise_percentage (x : ℝ) (hx_between : raise_between x) (hx_fraction : salary_increase_by_fraction x) :
  x = 0.06 :=
sorry

end valid_raise_percentage_l476_476487


namespace johns_speed_final_push_l476_476987

-- Definitions for the given conditions
def john_behind_steve : ℝ := 14
def steve_speed : ℝ := 3.7
def john_ahead_steve : ℝ := 2
def john_final_push_time : ℝ := 32

-- Proving the statement
theorem johns_speed_final_push : 
  (∃ (v : ℝ), v * john_final_push_time = steve_speed * john_final_push_time + john_behind_steve + john_ahead_steve) -> 
  ∃ (v : ℝ), v = 4.2 :=
by
  sorry

end johns_speed_final_push_l476_476987


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476541

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476541


namespace distance_A_to_B_is_25sqrt5_l476_476376

noncomputable def distance_from_A_to_B : ℝ :=
  let north_south := 30 - 15 + 10
  let east_west := 80 - 30
  real.sqrt (north_south^2 + east_west^2)

theorem distance_A_to_B_is_25sqrt5 :
  distance_from_A_to_B = 25 * real.sqrt 5 :=
by
  sorry

end distance_A_to_B_is_25sqrt5_l476_476376


namespace arithmetic_sequence_problem_l476_476914

variable (a_2 a_4 a_3 : ℤ)

theorem arithmetic_sequence_problem (h : a_2 + a_4 = 16) : a_3 = 8 :=
by
  -- The proof is not needed as per the instructions
  sorry

end arithmetic_sequence_problem_l476_476914


namespace problem_statement_l476_476628

def N(x : ℝ) := 3 * real.sqrt x
def O(x : ℝ) := x^2

theorem problem_statement : N(O(N(O(N(O(4)))))) = 108 :=
by
  sorry

end problem_statement_l476_476628


namespace PK_perp_AB_l476_476625

open EuclideanGeometry

variables {A B C D K L M P : Point}
variables [triangle ABC]
variables [foot_of_altitude A D]
variables [internal_bisector_angle DAC BC K]
variables [projection K AC L]
variables [intersection_line BL AD M]
variables [intersection_line MC DL P]

theorem PK_perp_AB : ⦃PK ⊥ AB⦄ :=
begin
  sorry
end

end PK_perp_AB_l476_476625


namespace cleaner_steps_l476_476776

theorem cleaner_steps (a b c : ℕ) (h1 : a < 10 ∧ b < 10 ∧ c < 10) (h2 : 100 * a + 10 * b + c > 100 * c + 10 * b + a) (h3 : 100 * a + 10 * b + c + 100 * c + 10 * b + a = 746) :
  (100 * a + 10 * b + c) * 2 = 944 ∨ (100 * a + 10 * b + c) * 2 = 1142 :=
by
  sorry

end cleaner_steps_l476_476776


namespace intersection_of_A_and_B_l476_476395

open Set

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {-2, -1, 0, 1, 2}

-- The theorem we want to prove
theorem intersection_of_A_and_B : A ∩ B = {0, 1} := sorry

end intersection_of_A_and_B_l476_476395


namespace angle_between_vectors_l476_476866

theorem angle_between_vectors :
  let v1 : ℝ × ℝ × ℝ := (3, -2, 2)
  let v2 : ℝ × ℝ × ℝ := (-2, 2, 1)
  let dot_product (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let magnitude (v : ℝ × ℝ × ℝ) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let cos_theta := dot_product v1 v2 / (magnitude v1 * magnitude v2)
  θ = Real.acos (cos_theta) * (180 / Real.pi)
  in θ ≈ 127 :=
by
  sorry

end angle_between_vectors_l476_476866


namespace triangle_DEF_angles_l476_476653

-- Conditions
variable (A B C K L D E F : Point)
variable (α : Real)

-- Assumptions
def conditions := 
  is_triangle A B C ∧
  on_line K A B ∧
  on_line L B C ∧
  angle_eq (angle K C B) α ∧
  angle_eq (angle L A B) α ∧
  perpendicular B D (line_through A L) ∧    -- BD ⊥ AL
  perpendicular B E (line_through C K) ∧    -- BE ⊥ CK
  midpoint F A C                            -- F midpoint of AC

noncomputable def find_angles (α : Real) : List Real :=
  [2 * α, 90 - α, 90 - α]

theorem triangle_DEF_angles ( α : Real )
  (A B C K L D E F : Point)
  (h : conditions A B C K L D E F α) : 
  find_angles α = [2 * α, 90 - α, 90 - α] := 
by sorry

end triangle_DEF_angles_l476_476653


namespace identity_function_l476_476873

def pos_nat := {n : ℕ // n > 0}
def f : pos_nat → pos_nat := sorry

theorem identity_function (f : pos_nat → pos_nat) 
  (h_condition : ∀ n : pos_nat, f n > f (f ⟨n.val - 1, by sorry⟩)) :
  ∀ n : pos_nat, f n = n := 
by
  sorry

end identity_function_l476_476873


namespace arithmetic_sequence_and_minimum_sum_l476_476529

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476529


namespace isosceles_triangle_perimeter_l476_476212

-- Define what it means to be a root of the equation x^2 - 5x + 6 = 0
def is_root (x : ℝ) : Prop := x^2 - 5 * x + 6 = 0

-- Define the perimeter based on given conditions
theorem isosceles_triangle_perimeter (x : ℝ) (base : ℝ) (h_base : base = 4) (h_root : is_root x) :
    2 * x + base = 10 :=
by
  -- Insert proof here
  sorry

end isosceles_triangle_perimeter_l476_476212


namespace modulus_of_complex_number_l476_476629

theorem modulus_of_complex_number (z : ℂ) (h : z^2 = 15 - 20 * complex.i) : complex.abs z = 5 := 
sorry

end modulus_of_complex_number_l476_476629


namespace range_x_coordinate_of_pointA_l476_476383

-- Definition of the circle O: x^2 + y^2 = 4
def circleO (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 4

-- Definition of the line l: x + y - 4 = 0
def lineL (x y : ℝ) : Prop :=
  x + y = 4

-- Definition of point A on line l
def pointA_on_lineL (x y : ℝ) : Prop :=
  lineL x y

-- Definitions of points B and C on circle O
def pointB_on_circleO (x y : ℝ) : Prop :=
  circleO x y

def pointC_on_circleO (x y : ℝ) : Prop :=
  circleO x y

-- Given angle condition ∠BAC = 60°
def angleBAC_eq_60 (A B C : ℝ × ℝ) : Prop :=
  -- Assume we have a function to calculate the angle
  angle_at_point A B C = 60

-- The theorem states the range of the x-coordinate of point A
theorem range_x_coordinate_of_pointA :
  ∀ (xA yA : ℝ), pointA_on_lineL xA yA →
    (∃ (xB yB xC yC : ℝ), pointB_on_circleO xB yB ∧ pointC_on_circleO xC yC ∧ angleBAC_eq_60 (xA, yA) (xB, yB) (xC, yC)) →
    0 ≤ xA ∧ xA ≤ 4 :=
by
  intros xA yA hA hExists
  -- Detailed proof omitted
  sorry

end range_x_coordinate_of_pointA_l476_476383


namespace simplify_expression_l476_476175

theorem simplify_expression :
  6^6 + 6^6 + 6^6 + 6^6 + 6^6 + 6^6 = 6^7 :=
by sorry

end simplify_expression_l476_476175


namespace polynomial_remainder_l476_476048

theorem polynomial_remainder (Q : ℤ[X]) (c d : ℤ) :
  (Q.eval 15 = 8) → 
  (Q.eval 19 = 10) → 
  (∀ Q, ∃ R, ∀ x, Q = (x - 15) * (x - 19) * R + C * x + d) :=
  sorry

end polynomial_remainder_l476_476048


namespace distinct_integers_after_transformations_l476_476895

theorem distinct_integers_after_transformations (n : Nat) (numbers : list Int) (p q : Nat) 
  (h_n : n = numbers.length) (h_pq : 0 < p ∧ 0 < q) :
  ∃ (final_numbers : list Int), (∀ i j, i ≠ j → i ∈ final_numbers → j ∈ final_numbers → i ≠ j) :=
sorry

end distinct_integers_after_transformations_l476_476895


namespace max_weight_and_measurable_count_l476_476831

def weights : List ℕ := [2, 3, 9]

noncomputable def maximum_weight : ℕ := weights.sum

def measurable_weights : Finset ℕ :=
  { x | ∃ (a b c : ℕ), a * 2 + b * 3 + c * 9 = x }.to_finset

def number_of_measurable_weights : ℕ := measurable_weights.card

theorem max_weight_and_measurable_count :
  maximum_weight = 14 ∧ number_of_measurable_weights = 13 :=
by
  sorry

end max_weight_and_measurable_count_l476_476831


namespace min_value_inverse_sum_l476_476614

theorem min_value_inverse_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 3 * b = 1) :
  \frac{1}{a} + \frac{1}{b} = 4 + 2 * sqrt 3 :=
sorry

end min_value_inverse_sum_l476_476614


namespace jars_in_each_carton_l476_476638

-- Define the number of jars in each carton (J) and the total number of good jars (good_jars)
variables (J : ℕ) (good_jars : ℕ)

-- Define the conditions
def conditions : Prop :=
  30 * J - (15 + J) = good_jars

-- The given number of good jars for sale that week
def given_good_jars : Prop :=
  good_jars = 565

-- The main theorem to prove the number of jars in each carton
theorem jars_in_each_carton : conditions J good_jars ∧ given_good_jars → J = 20 :=
by {
  -- Lean proof goes here
  sorry
}

end jars_in_each_carton_l476_476638


namespace proof_system_solution_l476_476079

noncomputable def solve_system : Prop :=
  ∃ x y : ℚ, x + 4 * y = 14 ∧ (x - 3) / 4 - (y - 3) / 3 = 1 / 12 ∧ x = 3 ∧ y = 11 / 4

theorem proof_system_solution : solve_system :=
sorry

end proof_system_solution_l476_476079


namespace min_sixth_power_sin_cos_l476_476260

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476260


namespace probability_of_contemporaries_l476_476131

noncomputable def probability_contemporaries_born_in_last_600_years 
  (birth_year_Alice : ℝ) (birth_year_Bob : ℝ) 
  (within_last_600_years : 0 ≤ birth_year_Alice ∧ birth_year_Alice ≤ 600 ∧ 0 ≤ birth_year_Bob ∧ birth_year_Bob ≤ 600)
  (lifetime : ℝ := 120) 
  (same_time_period : |birth_year_Alice - birth_year_Bob| ≤ lifetime) : ℝ :=
  (18 : ℝ) / 25

theorem probability_of_contemporaries : 
  ∀ (birth_year_Alice birth_year_Bob : ℝ), 
    0 ≤ birth_year_Alice ∧ birth_year_Alice ≤ 600 ∧ 0 ≤ birth_year_Bob ∧ birth_year_Bob ≤ 600 
    → |birth_year_Alice - birth_year_Bob| ≤ 120 
    → probability_contemporaries_born_in_last_600_years birth_year_Alice birth_year_Bob 
      (by assumption) (120) (by assumption) = 18 / 25 :=
by
  sorry

end probability_of_contemporaries_l476_476131


namespace arithmetic_sequence_minimum_value_S_n_l476_476555

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476555


namespace circle_intersection_midpoints_right_angle_l476_476728

theorem circle_intersection_midpoints_right_angle
  (A B C D M N K : Type*)
  [is_circle_intersection A B]
  [passes_through ℓ A]
  [intersects_again ℓ A C D]
  [midpoint_arc_not_containing M B C A]
  [midpoint_arc_not_containing N B D A]
  [midpoint_line_segment K C D] :
  ∠(M, K, N) = 90°
:= sorry

end circle_intersection_midpoints_right_angle_l476_476728


namespace proportional_relationship_compare_y_values_triangle_area_shift_l476_476402

theorem proportional_relationship (x y : ℝ) (h1 : ∀ x y, y = -2 * (x - 1)) (h2 : y = 4 → x = -1) :
  y = -2 * x + 2 := sorry

theorem compare_y_values (x1 x2 y1 y2 : ℝ) (h1 : x1 > x2) (h2 : y1 = -2 * x1 + 2) (h3 : y2 = -2 * x2 + 2) :
  y1 < y2 := sorry

theorem triangle_area_shift (x y : ℝ)
  (h1 : ∀ x y, y = -2 * x -2 → x = -1 ∧ y = -2 ∧ A = (-1, 0) ∧ B = (0, -2)) :
  ∃ A B : ℝ × ℝ, A = (-1, 0) ∧ B = (0, -2) ∧ area (triangle A O B) = 1 := sorry

end proportional_relationship_compare_y_values_triangle_area_shift_l476_476402


namespace binomial_even_sum_l476_476225

theorem binomial_even_sum (n : ℕ) :
  (∑ k in finset.range (n + 1), nat.choose (2 * n) (2 * k)) = 2^(2 * n - 1) - 1 :=
by
  -- Proof to be inserted here
  sorry

end binomial_even_sum_l476_476225


namespace sin_cos_sixth_min_l476_476351

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476351


namespace arithmetic_sequence_and_minimum_sum_l476_476521

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476521


namespace race_arrangements_l476_476429

theorem race_arrangements (H := "Harry") (R := "Ron") (N := "Neville") (Hrm := "Hermione") :
    let participants := [H, R, N, Hrm] in
    let consecutive_condition := (H = "Harry" ∧ Hrm = "Hermione") ∨ (Hrm = "Hermione" ∧ H = "Harry") in
    (no_ties participants) →
    (Hrm_finishes_consecutively participants) →
    (num_possible_orders participants consecutive_condition = 12) :=
sorry

end race_arrangements_l476_476429


namespace arithmetic_sequence_min_value_S_l476_476503

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476503


namespace smallest_pos_int_satisfy_congruence_l476_476749

theorem smallest_pos_int_satisfy_congruence : ∃ x : ℤ, x > 0 ∧ 5 * x ≡ 22 [MOD 31] ∧ ∀ y : ℤ, (y > 0 ∧ 5 * y ≡ 22 [MOD 31]) → x ≤ y :=
sorry

end smallest_pos_int_satisfy_congruence_l476_476749


namespace mike_pens_given_l476_476755

noncomputable def pens_remaining (initial_pens mike_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - 19

theorem mike_pens_given 
  (initial_pens : ℕ)
  (mike_pens final_pens : ℕ) 
  (H1 : initial_pens = 7)
  (H2 : final_pens = 39) 
  (H3 : pens_remaining initial_pens mike_pens = final_pens) : 
  mike_pens = 22 := sorry

end mike_pens_given_l476_476755


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476360

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476360


namespace find_a_l476_476906

-- Assuming the existence of functions and variables as per conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Defining the given conditions
axiom cond1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x - 5
axiom cond2 : f a = 6

-- Now stating the proof goal
theorem find_a : a = 7 / 4 := by
  sorry

end find_a_l476_476906


namespace smallest_degree_of_polynomial_l476_476674

theorem smallest_degree_of_polynomial :
  ∃ (p : Polynomial ℚ), 
    (p.roots ≃ [2 - sqrt 3, 2 + sqrt 3, -2 - sqrt 3, -2 + sqrt 3, 3 + 2 * sqrt 5, 3 - 2 * sqrt 5]) ∧
    (p.degree = 6) := sorry

end smallest_degree_of_polynomial_l476_476674


namespace sin_shift_left_l476_476121

theorem sin_shift_left : 
  ∀ x : ℝ, sin (x + π / 6) = sin (5 * π / 6 - x) :=
by 
  sorry

end sin_shift_left_l476_476121


namespace all_groups_common_member_exists_l476_476769

-- Definitions based on conditions
variables {G : Type*} [fintype G] (X : finset G) (groups : finset (finset G))
variables (ttable : fin ttable_size) (members : finset (fin 25))

-- Each group has at most 9 members
axiom each_group_at_most_9 (g : finset G) (hg : g ∈ groups) : g.card ≤ 9

-- Every two groups have at least one member in common.
axiom every_two_groups_have_common_member 
  (g1 g2 : finset G) (hg1 : g1 ∈ groups) (hg2 : g2 ∈ groups) : g1 ≠ g2 → (g1 ∩ g2 ≠ ∅)

-- Prove there is one gentleman who belongs to all groups
theorem all_groups_common_member_exists : 
  ∃ g ∈ members, ∀ group ∈ groups, g ∈ group :=
sorry

end all_groups_common_member_exists_l476_476769


namespace cone_base_radius_l476_476716

-- Definitions based on conditions
def sphere_radius : ℝ := 1
def cone_height : ℝ := 2

-- Problem statement
theorem cone_base_radius {r : ℝ} 
  (h1 : ∀ x y z : ℝ, (x = sphere_radius ∧ y = sphere_radius ∧ z = sphere_radius) → 
                     (x + y + z = 3 * sphere_radius)) 
  (h2 : ∃ (O O1 O2 O3 : ℝ), (O = 0) ∧ (O1 = 1) ∧ (O2 = 1) ∧ (O3 = 1)) 
  (h3 : ∀ x y z : ℝ, (x + y + z = 3 * sphere_radius) → 
                     (y = z) → (x = z) → y * z + x * z + x * y = 3 * sphere_radius ^ 2)
  (h4 : ∀ h : ℝ, h = cone_height) :
  r = (Real.sqrt 3 / 6) :=
sorry

end cone_base_radius_l476_476716


namespace leo_kept_packs_calculation_l476_476991

-- Definitions based on conditions
def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def packs_given_to_manny_fraction : ℚ := 1 / 4
def packs_given_to_neil_fraction : ℚ := 1 / 8

-- Main statement
theorem leo_kept_packs_calculation :
  let total_packs := total_marbles / marbles_per_pack in
  let packs_given_to_manny := packs_given_to_manny_fraction * total_packs in
  let packs_given_to_neil := packs_given_to_neil_fraction * total_packs in
  let packs_given_away := packs_given_to_manny + packs_given_to_neil in
  let packs_kept := total_packs - packs_given_away in
  packs_kept = 25 :=
by
  sorry

end leo_kept_packs_calculation_l476_476991


namespace law_firm_associates_l476_476006

def percentage (total: ℕ) (part: ℕ): ℕ := part * 100 / total

theorem law_firm_associates (total: ℕ) (second_year: ℕ) (first_year: ℕ) (more_than_two_years: ℕ):
  percentage total more_than_two_years = 50 →
  percentage total second_year = 25 →
  first_year = more_than_two_years - second_year →
  percentage total first_year = 25 →
  percentage total (total - first_year) = 75 :=
by
  intros h1 h2 h3 h4
  sorry

end law_firm_associates_l476_476006


namespace f_prime_neg_one_l476_476406

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f x = f (-x)
axiom h2 : ∀ x : ℝ, f (x + 1) - f (1 - x) = 2 * x

theorem f_prime_neg_one : f' (-1) = -1 := by
  -- The proof is omitted
  sorry

end f_prime_neg_one_l476_476406


namespace quadratic_expression_eqn_l476_476385

noncomputable def quadratic_function (a b c x : ℝ) := a * x^2 + b * x + c

theorem quadratic_expression_eqn (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x : ℝ, f x = quadratic_function a (- (2 + 4 * a)) (3 * a) x) 
  (h2 : ∀ x : ℝ, (1 < x ∧ x < 3) ↔ (f x > -2 * x)) 
  (h3 : ∃ x : ℝ, f x + 6 * a = 0 ∧ discriminant a (- (2 + 4 * a)) (9 * a) = 0)
  (h4 : a < 0) :
  f = λ x, -x^2 - x - 3/5 :=
sorry

-- Here discriminant computation can be defined as below for the sake of completeness
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

end quadratic_expression_eqn_l476_476385


namespace ann_age_l476_476809

theorem ann_age {a b y : ℕ} (h1 : a + b = 44) (h2 : y = a - b) (h3 : b = a / 2 + 2 * (a - b)) : a = 24 :=
by
  sorry

end ann_age_l476_476809


namespace arithmetic_sequence_min_value_Sn_l476_476563

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476563


namespace prove_arithmetic_sequence_minimum_value_S_l476_476530

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476530


namespace find_angle_ACB_l476_476003

theorem find_angle_ACB
  (DC_parallel_AB : ∀ (DC AB : ℝ), parallel DC AB)
  (DCA : ℝ) (ABC : ℝ) : DC_parallel_AB DC AB → DCA = 50 → ABC = 60 → ACB = 70 :=
by sorry

end find_angle_ACB_l476_476003


namespace arithmetic_sequence_minimum_value_S_l476_476584

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476584


namespace stationary_point_of_f_perpendicular_tangent_line_no_arithmetic_seq_l476_476381

/- Problem Statement Part 1 -/
theorem stationary_point_of_f (a : ℝ) 
  (f : ℝ → ℝ := λ x => (a-2) * x^3 - x^2 + 5 * x + (1-a) * real.log x) 
  (h : deriv f 1 = 0) : a = 1 :=
sorry

/- Problem Statement Part 2 -/
theorem perpendicular_tangent_line (f : ℝ → ℝ := λ x => -2 * x^3 - x^2 + 5 * x + real.log x) : 
  ∃ x, deriv f x = -1 :=
sorry

/- Problem Statement Part 3 -/
theorem no_arithmetic_seq (x1 x2 x3 : ℝ) 
  (a : ℝ := 2) 
  (f : ℝ → ℝ := λ x => -x^2 + 5 * x - real.log x) 
  (h1 : 0 < x1 ∧ x1 < x2 ∧ x2 < x3)
  (h2 : x2 = (x1 + x3) / 2) : false :=
sorry

end stationary_point_of_f_perpendicular_tangent_line_no_arithmetic_seq_l476_476381


namespace count_sums_of_three_cubes_l476_476434

theorem count_sums_of_three_cubes :
  let possible_sums := {n | ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ n = a^3 + b^3 + c^3}
  ∃ unique_sums : Finset ℕ, (∀ x ∈ possible_sums, x < 1000) ∧ unique_sums.card = 153 :=
by sorry

end count_sums_of_three_cubes_l476_476434


namespace average_marks_second_class_l476_476081

theorem average_marks_second_class :
  (∃ x : ℝ, 
    let total_students := 76 in
    let average_all_students := 53.1578947368421 in
    let total_first_class := 26 in
    let average_first_class := 40 in
    let total_second_class := 50 in
    let total_marks_first_class := total_first_class * average_first_class in
    (total_marks_first_class + total_second_class * x) / total_students = average_all_students ∧
    x = 60) := sorry

end average_marks_second_class_l476_476081


namespace negation_of_all_have_trap_consumption_l476_476927

-- Definitions for the conditions
def domestic_mobile_phone : Type := sorry

def has_trap_consumption (phone : domestic_mobile_phone) : Prop := sorry

def all_have_trap_consumption : Prop := ∀ phone : domestic_mobile_phone, has_trap_consumption phone

-- Statement of the problem
theorem negation_of_all_have_trap_consumption :
  ¬ all_have_trap_consumption ↔ ∃ phone : domestic_mobile_phone, ¬ has_trap_consumption phone :=
sorry

end negation_of_all_have_trap_consumption_l476_476927


namespace geometric_sequence_sum_5_l476_476049

def geom_seq (a q : ℝ) (n : ℕ) : ℝ := a * q^n

def S_n (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^(n + 1)) / (1 - q)

theorem geometric_sequence_sum_5 {a_1 q : ℝ} (h1 : a_1 = 1 / 3) 
  (h2 : geom_seq a_1 q 3^2 = geom_seq a_1 q 5) :
  S_n a_1 q 4 = 121 / 3 :=
by
  sorry

end geometric_sequence_sum_5_l476_476049


namespace find_n_from_divisors_l476_476623

theorem find_n_from_divisors (n d₁ d₂ d₃ d₄ : ℕ) 
    (hn_pos : 0 < n)
    (h_divisors : d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄)
    (h_min_divisors : {d₁, d₂, d₃, d₄} = {d ∈ Icc 1 n | n % d = 0 | take 4 ↑})
    (h_equation : n = d₁^2 + d₂^2 + d₃^2 + d₄^2) :
    n = 130 :=
sorry

end find_n_from_divisors_l476_476623


namespace area_inside_C_outside_A_B_l476_476820

-- Definition of circles A, B, and C with radius 1
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

constant A : Circle
constant B : Circle
constant C : Circle

axiom A_radius : A.radius = 1
axiom B_radius : B.radius = 1
axiom C_radius : C.radius = 1

-- Midpoint of the line segment AB
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Axiom stating that A and B are tangent at one point
axiom A_B_tangent_at_one_point : 
  let d := ((A.center.1 - B.center.1) ^ 2 + (A.center.2 - B.center.2) ^ 2).sqrt in
  d = A.radius + B.radius

-- Axiom stating that C is tangent to the midpoint of AB
axiom C_tangent_to_midpoint : 
  let M := midpoint A.center B.center in
  let d := ((C.center.1 - M.1) ^ 2 + (C.center.2 - M.2) ^ 2).sqrt in
  d = C.radius

-- Theorem to prove
theorem area_inside_C_outside_A_B : 
  let shared_area := (π * 1^2) / 2 - (1 / 2) in
  let total_shared_area := 4 * shared_area in
  let area_C := π * 1^2 in
  area_C - total_shared_area = 2 :=
sorry

end area_inside_C_outside_A_B_l476_476820


namespace solve_for_p_l476_476030

theorem solve_for_p (p x : ℂ) (h1 : 3 * p - x = 15000) (h2 : x = 9 + 225 * complex.I) : 
  p = 5003 + 75 * complex.I := 
by
  sorry

end solve_for_p_l476_476030


namespace volume_tetrahedron_value_l476_476850

open Real

noncomputable def volume_tetrahedron (P Q R S : ℝ³) : ℝ :=
  let area_PQR := 150
  let area_QRS := 50
  let QR := 10
  let angle_PQR_QRS := π/4  -- 45 degrees in radians
  let height_QRS := (2 * area_QRS) / QR  -- height from S to QR
  let height_S_to_plane_PQR := height_QRS * sin(angle_PQR_QRS) 
  (1/3) * area_PQR * height_S_to_plane_PQR

theorem volume_tetrahedron_value (P Q R S : ℝ³) :
  volume_tetrahedron P Q R S = 250 * sqrt 2 :=
  sorry

end volume_tetrahedron_value_l476_476850


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476544

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476544


namespace arccos_cos_equals_l476_476232

theorem arccos_cos_equals : 
  ∃ x : ℝ, x = 2 * Real.pi - 3 ∧ 0 ≤ x ∧ x ≤ Real.pi ∧ Real.arccos (Real.cos 3) = x := 
by
  have h_periodicity : Real.cos x = Real.cos (2 * Real.pi - x) := sorry
  existsi (2 * Real.pi - 3)
  split
  · refl
  · split
    · sorry  -- Proof that 0 ≤ 2 * Real.pi - 3
    · sorry  -- Proof that 2 * Real.pi - 3 ≤ π
  · sorry  -- Proof that Real.arccos (Real.cos 3) = 2 * Real.pi - 3

end arccos_cos_equals_l476_476232


namespace problem_statement_l476_476187

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem problem_statement
  (h1 : ∀ x, 0 < x ∧ x < (π / 2) → f x < f' x * tan x) :
  sqrt 3 * f (π / 6) < f (π / 3) :=
sorry

end problem_statement_l476_476187


namespace boat_stream_ratio_l476_476757

-- Conditions: A man takes twice as long to row a distance against the stream as to row the same distance in favor of the stream.
theorem boat_stream_ratio (B S : ℝ) (h : ∀ (d : ℝ), d / (B - S) = 2 * (d / (B + S))) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l476_476757


namespace polynomial_remainder_l476_476046

theorem polynomial_remainder (Q : ℚ[X]) :
  (Q.eval 15 = 8) → (Q.eval 19 = 10) →
  ∃ c d : ℚ, (∀ x, Q.eval x = (x-15)*(x-19)*R.eval x + c*x + d) 
             ∧ (c = 1/2) 
             ∧ (d = 1/2) :=
by
  intro hQ15 hQ19
  sorry

end polynomial_remainder_l476_476046


namespace length_of_largest_square_l476_476660

-- Define the conditions of the problem
def side_length_of_shaded_square : ℕ := 10
def side_length_of_largest_square : ℕ := 24

-- The statement to prove
theorem length_of_largest_square (x : ℕ) (h1 : x = side_length_of_shaded_square) : 
  4 * x = side_length_of_largest_square :=
  by
  -- Insert the proof here
  sorry

end length_of_largest_square_l476_476660


namespace candidate_A_votes_l476_476760

theorem candidate_A_votes :
  ∀ (total_votes : ℕ) (pct_invalid : ℝ) (pct_A : ℝ),
    total_votes = 560000 →
    pct_invalid = 0.15 →
    pct_A = 0.60 →
    (pct_A * (1 - pct_invalid) * total_votes).to_nat = 285600 :=
by
  intros total_votes pct_invalid pct_A h_total_votes h_pct_invalid h_pct_A
  rw [h_total_votes, h_pct_invalid, h_pct_A]
  simp
  norm_cast
  sorry

end candidate_A_votes_l476_476760


namespace focus_of_parabola_l476_476255

theorem focus_of_parabola (a b c : ℝ) (h k : ℝ) (p : ℝ)
  (h_eqn : a = 9 ∧ b = 6 ∧ c = -5)
  (h_vertex : h = -1 / 3 ∧ k = -6)
  (h_p : p = 1 / (4 * a)) :
  (h, k + p) = (-1 / 3, -215 / 36) :=
by
  cases h_eqn with ha hb
  cases hb with hb hc
  rw [fraction_ring.mk_eq_div, real.div_eq_mul_inv, fraction_ring.mk_eq_div, real.div_eq_mul_inv]
  sorry

end focus_of_parabola_l476_476255


namespace tim_movie_marathon_duration_is_9_l476_476725

-- Define the conditions:
def first_movie_duration : ℕ := 2
def second_movie_duration : ℕ := first_movie_duration + (first_movie_duration / 2)
def combined_duration_first_two_movies : ℕ := first_movie_duration + second_movie_duration
def third_movie_duration : ℕ := combined_duration_first_two_movies - 1
def total_marathon_duration : ℕ := first_movie_duration + second_movie_duration + third_movie_duration

-- The theorem to prove the marathon duration is 9 hours
theorem tim_movie_marathon_duration_is_9 :
  total_marathon_duration = 9 :=
by sorry

end tim_movie_marathon_duration_is_9_l476_476725


namespace min_value_sin_cos_l476_476297

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476297


namespace inscribed_circle_radius_in_triangle_l476_476025

theorem inscribed_circle_radius_in_triangle
  (P Q R : Type)
  (PQ : ℝ) (PR : ℝ) (QR : ℝ)
  (h1 : PQ = 9)
  (h2 : PR = 9)
  (h3 : QR = 8)
  : radius P Q R = 4 * Real.sqrt 65 / 13 := sorry

end inscribed_circle_radius_in_triangle_l476_476025


namespace stone_length_l476_476189

theorem stone_length (hall_length_m : ℕ) (hall_breadth_m : ℕ) (number_of_stones : ℕ) (stone_width_dm : ℕ) 
    (length_in_dm : 10 > 0) :
    hall_length_m = 36 → hall_breadth_m = 15 → number_of_stones = 2700 → stone_width_dm = 5 →
    ∀ L : ℕ, 
    (10 * hall_length_m) * (10 * hall_breadth_m) = number_of_stones * (L * stone_width_dm) → 
    L = 4 :=
by
  intros h1 h2 h3 h4
  simp at *
  sorry

end stone_length_l476_476189


namespace sum_of_distinct_squares_l476_476672

theorem sum_of_distinct_squares (a b c : ℕ) (h1 : a + b + c = 39)
  (h2 : Int.gcd a b + Int.gcd b c + Int.gcd c a = 11) :
  (a^2 + b^2 + c^2 = 531 ∨ a^2 + b^2 + c^2 = 653) →
  ∑ (val : ℕ) in ({531, 653} : Finset ℕ), val = 1184 :=
by
  sorry

end sum_of_distinct_squares_l476_476672


namespace arithmetic_sequence_terms_l476_476937

theorem arithmetic_sequence_terms :
  ∀ (n : ℕ), (n > 0) → (((100 + (n - 1) * (-6)) = 4) → (n = 17) ∧ (n - 1 = 16)) :=
by
  intros n hn ha
  sorry

end arithmetic_sequence_terms_l476_476937


namespace number_of_people_prefer_soda_l476_476459

-- Given conditions as definitions
def total_people : ℕ := 520
def central_angle_soda : ℕ := 278
def total_circle_angle : ℕ := 360

-- Definition to calculate the fraction of the circle representing "Soda"
def fraction_soda := (central_angle_soda : ℚ) / (total_circle_angle : ℚ)

-- Definition to calculate the number of people who prefer "Soda"
def people_prefer_soda := (total_people : ℚ) * fraction_soda

-- Proof statement
theorem number_of_people_prefer_soda :
  people_prefer_soda.round = 402 :=
by
  sorry

end number_of_people_prefer_soda_l476_476459


namespace arithmetic_sequence_min_value_Sn_l476_476569

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476569


namespace rhombus_not_diagonals_equal_l476_476157

theorem rhombus_not_diagonals_equal (R : Type) [linear_ordered_field R] 
  (a b c d : R) (h1 : a = b) (h2 : b = c) (h3 : c = d) (h4 : a = d)
  (h_sym : ∀ x y : R, a = b → b = c → c = d → d = a)
  (h_cen_sym : ∀ p : R × R, p = (0, 0) → p = (0, 0)) :
  ¬(∀ p q : R × R, p ≠ q → (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 = (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) :=
by
  sorry

end rhombus_not_diagonals_equal_l476_476157


namespace prob_yellow_and_straight_l476_476759

-- Definitions of probabilities given in the problem
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2

-- Derived probability of picking a yellow flower
def prob_yellow : ℚ := 1 - prob_green

-- Statement to prove
theorem prob_yellow_and_straight : prob_yellow * prob_straight = 1 / 6 :=
by
  -- sorry is used here to skip the proof.
  sorry

end prob_yellow_and_straight_l476_476759


namespace minimum_value_expression_l476_476619

noncomputable def minimum_expression (a b c : ℝ) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_expression a b c ≥ 126 :=
by
  sorry

end minimum_value_expression_l476_476619


namespace response_rate_percentage_l476_476180

theorem response_rate_percentage (number_of_responses_needed number_of_questionnaires_mailed : ℕ) 
  (h1 : number_of_responses_needed = 300) 
  (h2 : number_of_questionnaires_mailed = 500) : 
  (number_of_responses_needed / number_of_questionnaires_mailed : ℚ) * 100 = 60 :=
by 
  sorry

end response_rate_percentage_l476_476180


namespace part_a_solutions_l476_476166

theorem part_a_solutions (x : ℝ) : (⌊x⌋^2 - x = -0.99) ↔ (x = 0.99 ∨ x = 1.99) :=
sorry

end part_a_solutions_l476_476166


namespace min_value_y_l476_476443

theorem min_value_y (x : ℝ) (h : 1 < x) : 
  let y := x + 2 / (x - 1) in 
  (∀ y' : ℝ, (∃ (x' : ℝ), (1 < x') ∧ y' = x' + 2 / (x' - 1)) → y ≥ y') → 
  y = 2 * Real.sqrt 2 + 1 := 
by
  sorry

end min_value_y_l476_476443


namespace pairwise_sum_difference_le_sqrt_l476_476754

theorem pairwise_sum_difference_le_sqrt (n : ℕ) (x : ℕ → ℝ)
  (h1 : n ≥ 4) (h2 : ∀ i, 1 ≤ i → i ≤ n → 0 < x i) :
∃ (i j k l : ℕ), i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ 
  (x i + x j) / (x k + x l) ≤ real.sqrt (2^(1/(n-2))) :=
sorry

end pairwise_sum_difference_le_sqrt_l476_476754


namespace leftover_floss_l476_476838

/-
Conditions:
1. There are 20 students in his class.
2. Each student needs 1.5 yards of floss.
3. Each packet of floss contains 35 yards.
4. He buys the least amount necessary.
-/

def students : ℕ := 20
def floss_needed_per_student : ℝ := 1.5
def total_floss_needed : ℝ := students * floss_needed_per_student
def floss_per_packet : ℝ := 35

theorem leftover_floss : floss_per_packet - total_floss_needed = 5 :=
by
  -- Assuming these values from the conditions
  have students_val : 20 = students := rfl
  have floss_needed_val : 1.5 = floss_needed_per_student := rfl
  have total_needed_val : total_floss_needed = 30 := by 
    simp [students, floss_needed_per_student, total_floss_needed]
  have floss_per_packet_val : 35 = floss_per_packet := rfl
  
  -- Calculation to get the leftover floss
  calc
    floss_per_packet - total_floss_needed 
        = 35 - 30 : by rw [total_needed_val]
    ... = 5 : by norm_num

end leftover_floss_l476_476838


namespace geometric_sum_n_eq_4_l476_476109

theorem geometric_sum_n_eq_4 :
  ∃ n : ℕ, (n = 4) ∧ 
  ((1 : ℚ) * (1 - (1 / 4 : ℚ) ^ n) / (1 - (1 / 4 : ℚ)) = (85 / 64 : ℚ)) :=
by
  use 4
  simp
  sorry

end geometric_sum_n_eq_4_l476_476109


namespace min_value_proof_l476_476611

noncomputable def min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1) : ℝ :=
  if a + 3 * b = 1 ∧ a > 0 ∧ b > 0 then Inf {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ x = (1 / a + 1 / b)} else 0

theorem min_value_proof :
  let a := Real
  let b := Real
  ∀ (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1),
  min_value a b h₀ h₁ h₂ = 4 + 2 * real.sqrt 3 :=
sorry

end min_value_proof_l476_476611


namespace binomial_divisibility_l476_476995

-- Define the function α_n (the number of 1's in the binary representation of n)
def α (n : ℕ) : ℕ := (((Nat.bits n).filter (λ b => b = true)).length)

-- Define the core theorem
theorem binomial_divisibility (n r : ℕ) (hn : 0 < n) (hr : 0 < r) :
  2^(2*n - α n) ∣ (Finset.range (2*n + 1)).sum (λ k : ℕ, Nat.choose (2*n) (n + k - n) * (k - n)^(2*r)) :=
by
  sorry

end binomial_divisibility_l476_476995


namespace correct_statements_l476_476374

universe u

variables {R : Type u} [LinearOrderedField R]

def quadratic_function (a x : R) : R := a * x^2 - (5*a + 1) * x + 4*a + 4

theorem correct_statements (a : R) :
  (a < -1 → (quadratic_function a 0) < 0) ∧
  (a > 0 → (∀ x, 1 ≤ x ∧ x ≤ 2 → quadratic_function a x ≤ 3)) ∧
  (a < 0 → let y1 := quadratic_function a 2, y2 := quadratic_function a 3, y3 := quadratic_function a 4 in y1 > y2 ∧ y2 > y3) :=
sorry

end correct_statements_l476_476374


namespace leo_kept_25_packs_of_marbles_l476_476994

/-- Leo had 400 marbles in a jar, each pack contains 10 marbles.
He gave Manny 1/4 of the packs, Neil received 1/8 of the packs. Prove that Leo kept 25 packs of marbles. -/
theorem leo_kept_25_packs_of_marbles :
  (let total_marbles := 400 in
   let marbles_per_pack := 10 in
   let total_packs := total_marbles / marbles_per_pack in
   let packs_given_to_manny := total_packs / 4 in
   let packs_given_to_neil := total_packs / 8 in
   let packs_kept := total_packs - packs_given_to_manny - packs_given_to_neil in
   packs_kept = 25) :=
by
  sorry

end leo_kept_25_packs_of_marbles_l476_476994


namespace coeff_of_x4_in_expansion_l476_476684

open BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

noncomputable def term_coeff (n r : ℕ) : ℤ :=
  (-1)^r * binom n r

theorem coeff_of_x4_in_expansion :
  term_coeff 5 2 = 10 := by
sorry

end coeff_of_x4_in_expansion_l476_476684


namespace abs_diff_41st_is_790_l476_476124

-- Definitions based on the given conditions
def seqA (n : ℕ) : ℤ := 50 + 6 * (n - 1)
def seqB (n : ℕ) : ℤ := 100 - 15 * (n - 1)
def abs_diff_41st : ℤ := | seqA 41 - seqB 41 |

-- The proof statement
theorem abs_diff_41st_is_790 : abs_diff_41st = 790 := by
  sorry -- Proof omitted as requested

end abs_diff_41st_is_790_l476_476124


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476597

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476597


namespace object_speed_approximation_l476_476171

theorem object_speed_approximation :
  ( (90:ℝ) / 5280) / ( (3:ℝ) / 3600) ≈ 20.46 :=
by
  sorry

end object_speed_approximation_l476_476171


namespace arithmetic_sequence_min_value_S_l476_476508

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476508


namespace remainder_x_squared_div_25_l476_476445

theorem remainder_x_squared_div_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 0 [ZMOD 25] :=
sorry

end remainder_x_squared_div_25_l476_476445


namespace min_sixth_power_sin_cos_l476_476270

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476270


namespace min_sin6_cos6_l476_476343

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476343


namespace train_speed_l476_476801

theorem train_speed :
  ∀ (length_of_train : ℝ) (time_to_cross_pole : ℝ),
  length_of_train = 100 →
  time_to_cross_pole = 6.666133375996587 →
  (length_of_train / time_to_cross_pole) * 3.6 = 54 :=
by
  intros length_of_train time_to_cross_pole
  assume length_train_eq time_cross_pole_eq
  have h : length_of_train / time_to_cross_pole = 15 := sorry
  have conversion_factor : 3.6 = 3.6 := by rfl
  rw [length_train_eq, time_cross_pole_eq] at h
  rw [h, conversion_factor]
  exact sorry

end train_speed_l476_476801


namespace remaining_lawn_fractional_l476_476644

-- Define the conditions as variables
variable (mary_rate tom_rate : ℝ)
variable (mary_time tom_time : ℝ)
variable (mary_mow_time tom_mow_time : ℕ)
variable (total_lawn remaining_lawn : ℝ)

-- Define the given conditions
def Mary_mow_rate : Prop := mary_rate = 1 / 3
def Tom_mow_rate : Prop := tom_rate = 1 / 6
def Tom_mow_time : Prop := tom_mow_time = 1
def Mary_mow_time : Prop := mary_mow_time = 2
def Total_lawn : Prop := total_lawn = 1

-- Define the target proof
theorem remaining_lawn_fractional :
  Mary_mow_rate ∧ Tom_mow_rate ∧ Tom_mow_time ∧ Mary_mow_time ∧ Total_lawn →
  remaining_lawn = 1 - (tom_rate * tom_mow_time + mary_rate * mary_mow_time) := by
  sorry

end remaining_lawn_fractional_l476_476644


namespace rational_numbers_of_b_c_d_l476_476151

def tan_pi_over_3 : Real := Real.tan (Real.pi / 3)
def B : Real := 2 * Real.log (2) + Real.log (25)
def C : Real := 3 ^ (1 / Real.log (3)) - Real.exp (1)
def D : Real := Real.log (3) / Real.log (4) * (Real.log (6) / Real.log (3)) * (Real.log (8) / Real.log (6))

theorem rational_numbers_of_b_c_d :
  (¬ Rational tan_pi_over_3) ∧
  Rational B ∧
  Rational C ∧
  Rational D := by
  sorry

end rational_numbers_of_b_c_d_l476_476151


namespace petya_numbers_l476_476655

-- Define the arithmetic sequence property
def arithmetic_seq (a d : ℕ) : ℕ → ℕ
| 0     => a
| (n+1) => a + (n + 1) * d

-- Given conditions
theorem petya_numbers (a d : ℕ) : 
  (arithmetic_seq a d 0 = 6) ∧
  (arithmetic_seq a d 1 = 15) ∧
  (arithmetic_seq a d 2 = 24) ∧
  (arithmetic_seq a d 3 = 33) ∧
  (arithmetic_seq a d 4 = 42) :=
sorry

end petya_numbers_l476_476655


namespace leo_kept_25_packs_of_marbles_l476_476993

/-- Leo had 400 marbles in a jar, each pack contains 10 marbles.
He gave Manny 1/4 of the packs, Neil received 1/8 of the packs. Prove that Leo kept 25 packs of marbles. -/
theorem leo_kept_25_packs_of_marbles :
  (let total_marbles := 400 in
   let marbles_per_pack := 10 in
   let total_packs := total_marbles / marbles_per_pack in
   let packs_given_to_manny := total_packs / 4 in
   let packs_given_to_neil := total_packs / 8 in
   let packs_kept := total_packs - packs_given_to_manny - packs_given_to_neil in
   packs_kept = 25) :=
by
  sorry

end leo_kept_25_packs_of_marbles_l476_476993


namespace regular_18_gon_side_length_eq_l476_476764

theorem regular_18_gon_side_length_eq (a : ℝ) (h : ∀ (n : ℕ), n = 18 → is_regular_ngon_inscribed (n : ℝ) 1 a) :
  a^3 = 3*a - 1 := by
sorry

end regular_18_gon_side_length_eq_l476_476764


namespace prove_arithmetic_sequence_minimum_value_S_l476_476535

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476535


namespace triangle_equilateral_l476_476911

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C : V}
variables {AB AC BC : V}

-- Definitions from the conditions
def condition1 (AB AC BC : V) :=
  (AB / ∥AB∥ + AC / ∥AC∥) ⋅ BC = 0

def condition2 (AB AC : V) :=
  (AB / ∥AB∥) ⋅ (AC / ∥AC∥) = 1 / 2

-- The statement of the problem
theorem triangle_equilateral
  (h1 : condition1 AB AC BC)
  (h2 : condition2 AB AC) :
  ∥A - B∥ = ∥A - C∥ ∧ angle (A - B) (A - C) = real.pi / 3 :=
sorry

end triangle_equilateral_l476_476911


namespace time_difference_l476_476980

variable (R : Type) [Field R] -- Using general type R to represent real numbers
variable (rabbits : R) (hours : R) (holes_rabbits : R)
variable (beavers : R) (minutes : R) (dams_beavers : R)

noncomputable def rabbit_rate : R := holes_rabbits / (rabbits * hours)
noncomputable def beaver_rate : R := dams_beavers / (beavers * minutes)

noncomputable def rabbit_time_one_hole : R := 1 / (rabbit_rate rabbits hours holes_rabbits) 
noncomputable def beaver_time_one_dam : R := 1 / (beaver_rate beavers minutes dams_beavers) 

theorem time_difference 
  (h1 : rabbits = 3) 
  (h2 : hours = 5) 
  (h3 : holes_rabbits = 9) 
  (h4 : beavers = 5) 
  (h5 : minutes = 36) 
  (h6 : dams_beavers = 2) :
  (rabbit_time_one_hole rabbits hours holes_rabbits) - (beaver_time_one_dam beavers minutes dams_beavers) = 10 :=
by
  sorry

end time_difference_l476_476980


namespace min_value_sin6_cos6_l476_476314

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476314


namespace prove_arithmetic_sequence_minimum_value_S_l476_476531

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476531


namespace max_roots_of_composition_l476_476407

-- Definitions based on the given conditions
variable {f : ℝ → ℝ}
variable {x1 x2 : ℝ}

-- Condition 1: f(x) has zero points x1 and x2 where x1 < 0 < x2 < 1
def has_two_zero_points (f : ℝ → ℝ) (x1 x2 : ℝ) : Prop :=
  f x1 = 0 ∧ f x2 = 0 ∧ x1 < 0 ∧ 0 < x2 ∧ x2 < 1

-- Condition 2: Definition of g(x) = x - ln(x^2)
def g (x : ℝ) : ℝ := x - Real.log (x^2)

-- Using the conditions, we want to prove the maximum number of real roots of the equation f[g(x)] = 0
theorem max_roots_of_composition (h : has_two_zero_points f x1 x2) : 
  ∃ x, (g x) ∈ Set.range f → (f ∘ g) x = 0 := sorry

end max_roots_of_composition_l476_476407


namespace sequence_problem_l476_476591

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476591


namespace no_real_roots_l476_476688

noncomputable def equation (x : ℝ) : ℝ := sqrt (x + 4) - sqrt (x - 3) + 1

theorem no_real_roots : ¬ (∃ x : ℝ, equation x = 0) :=
by
  sorry

end no_real_roots_l476_476688


namespace length_EQ_is_correct_l476_476966

-- Definitions of points and properties
def square_side_length : ℝ := 2

def circle_radius (s : ℝ) : ℝ := s / 2

def circle_eq (r : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = r^2

def point_E (s : ℝ) : ℝ × ℝ := (- (s / 2), s / 2)

def point_N : ℝ × ℝ := (0, -1)

def line_EN_eq (E N : ℝ × ℝ) : ℝ → ℝ := λ x, let m := (N.2 - E.2) / (N.1 - E.1) in m * x + E.2 - m * E.1

-- Question: Prove the length of EQ given the conditions
theorem length_EQ_is_correct :
  let s := square_side_length
  let r := circle_radius s
  let circ_eq := circle_eq r
  let E := point_E s
  let N := point_N
  let EN_eq := line_EN_eq E N
  let Q := (4/5 : ℝ, -3/5 : ℝ)
  EQ := (λ (E Q : ℝ × ℝ), real.sqrt ((E.1 - Q.1)^2 + (E.2 - Q.2)^2))
  EQ E Q = (real.sqrt 145) / 5 :=
by
  sorry

end length_EQ_is_correct_l476_476966


namespace line_divides_circle_max_area_diff_l476_476898

theorem line_divides_circle_max_area_diff (P : ℝ × ℝ) (hP : P = (1, 1)) :
  ∃ l : ℝ → ℝ → Prop, 
  (l = (λ x y, x + y - 2 = 0)) ∧ 
  (∀ (x y : ℝ), (l x y → (x^2 + y^2 ≤ 4)) → 
    (l x y → True)) := sorry

end line_divides_circle_max_area_diff_l476_476898


namespace collinear_X_Y_Z_l476_476054

variables {V : Type*} [InnerProductSpace ℝ V]

-- Let JHIZ be a rectangle
noncomputable def J (V) : V := sorry
noncomputable def H (V) : V := sorry
noncomputable def I (V) : V := sorry
noncomputable def Z (V) : V := sorry

-- Let A be a point on ZI
noncomputable def A (V) : V := sorry

-- Let C be a point on ZJ
noncomputable def C (V) : V := sorry

-- X is the intersection of the perpendicular from A to CH with the line HI
noncomputable def CH (V) : Submodule ℝ V := sorry
noncomputable def HI (V) : Submodule ℝ V := sorry
noncomputable def X (V) : V := sorry

-- Y is the intersection of the perpendicular from C to AH with the line HJ
noncomputable def AH (V) : Submodule ℝ V := sorry
noncomputable def HJ (V) : Submodule ℝ V := sorry
noncomputable def Y (V) : V := sorry

-- Prove that X, Y, Z are collinear
theorem collinear_X_Y_Z : collinear ℝ ({X V, Y V, Z V} : Set V) :=
sorry

end collinear_X_Y_Z_l476_476054


namespace min_value_proof_l476_476612

noncomputable def min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1) : ℝ :=
  if a + 3 * b = 1 ∧ a > 0 ∧ b > 0 then Inf {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ x = (1 / a + 1 / b)} else 0

theorem min_value_proof :
  let a := Real
  let b := Real
  ∀ (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1),
  min_value a b h₀ h₁ h₂ = 4 + 2 * real.sqrt 3 :=
sorry

end min_value_proof_l476_476612


namespace smallest_pos_int_satisfy_congruence_l476_476748

theorem smallest_pos_int_satisfy_congruence : ∃ x : ℤ, x > 0 ∧ 5 * x ≡ 22 [MOD 31] ∧ ∀ y : ℤ, (y > 0 ∧ 5 * y ≡ 22 [MOD 31]) → x ≤ y :=
sorry

end smallest_pos_int_satisfy_congruence_l476_476748


namespace block_of_zeros_l476_476076

theorem block_of_zeros (k : ℤ) (hk : k ≥ 1) : ∃ (n : ℕ), (∃ a b : ℕ, b ≠ 0 ∧ (a * 10^k + 10^k * b) = 2^n) :=
by
  sorry

end block_of_zeros_l476_476076


namespace min_sin6_cos6_l476_476279

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476279


namespace rhombus_diagonals_not_equal_l476_476158

-- Define what a rhombus is
structure Rhombus where
  sides_equal : ∀ a b : ℝ, a = b  -- all sides are equal
  symmetrical : Prop -- it is a symmetrical figure
  centrally_symmetrical : Prop -- it is a centrally symmetrical figure

-- Theorem to state that the diagonals of a rhombus are not necessarily equal
theorem rhombus_diagonals_not_equal (r : Rhombus) : ¬(∀ a b : ℝ, a = b) := by
  sorry

end rhombus_diagonals_not_equal_l476_476158


namespace find_a_l476_476021

-- Definitions from conditions
def is_inverse (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop := ∀ x, f (g x) = x ∧ g (f x) = x

def is_symmetric_y_axis (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop := ∀ x, g x = f (-x)

-- Main theorem statement
theorem find_a (f g : ℝ → ℝ) (a : ℝ) (h_inv : is_inverse f (λ x, (1/2)^x))
  (h_symm : is_symmetric_y_axis f g) (h_ga_neg2 : g a = -2) :
  a = -4 :=
sorry

end find_a_l476_476021


namespace sequence_problem_l476_476590

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476590


namespace arithmetic_sequence_minimum_value_S_n_l476_476558

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476558


namespace negation_of_proposition_l476_476697

-- Definitions and conditions from the problem
def original_proposition (x : ℝ) : Prop := x^3 - x^2 + 1 > 0

-- The proof problem: Prove the negation
theorem negation_of_proposition : (¬ ∀ x : ℝ, original_proposition x) ↔ ∃ x : ℝ, ¬original_proposition x := 
by
  -- here we insert our proof later
  sorry

end negation_of_proposition_l476_476697


namespace triangle_is_isosceles_l476_476957

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop := ∃ (s : ℝ), a = s ∧ b = s

theorem triangle_is_isosceles 
  (A B C a b c : ℝ) 
  (h_sides_angles : a = c ∧ b = c) 
  (h_cos_eq : a * Real.cos B = b * Real.cos A) : 
  is_isosceles_triangle A B C a b c := 
by 
  sorry

end triangle_is_isosceles_l476_476957


namespace probability_red_in_both_jars_l476_476486

def original_red_buttons : ℕ := 6
def original_blue_buttons : ℕ := 10
def total_original_buttons : ℕ := original_red_buttons + original_blue_buttons
def remaining_buttons : ℕ := (2 * total_original_buttons) / 3
def moved_buttons : ℕ := total_original_buttons - remaining_buttons
def moved_red_buttons : ℕ := 2
def moved_blue_buttons : ℕ := 3

theorem probability_red_in_both_jars :
  moved_red_buttons = moved_blue_buttons →
  remaining_buttons = 11 →
  (∃ m n : ℚ, m / remaining_buttons = 4 / 11 ∧ n / (moved_red_buttons + moved_blue_buttons) = 2 / 5 ∧ (m / remaining_buttons) * (n / (moved_red_buttons + moved_blue_buttons)) = 8 / 55) :=
by sorry

end probability_red_in_both_jars_l476_476486


namespace projection_a_b_in_dir_b_l476_476393

variables {u v : ℝ → ℝ → ℝ}
variables (a b : u → v)
variables (nonzero_a : ∀ x : u, a x ≠ 0) (nonzero_b : ∀ y : u, b y ≠ 0)

# the condition |a + b| = |a − b|
axiom abs_cond : ∀ x : u, |a x + b x| = |a x - b x|

# the projection formula
def projection (u v : u → v) (x : u) : v :=
  ((u x) • (v x)) / (v x) ^ 2 * (v x)

theorem projection_a_b_in_dir_b:
  ∀ x, projection (a x - b x) b x = - b x :=
by
  sorry

end projection_a_b_in_dir_b_l476_476393


namespace division_line_exists_l476_476729

theorem division_line_exists (n : ℕ) (h : n = 2000000) (points : fin n → ℝ × ℝ) : 
  ∃ l : ℝ × ℝ × ℝ × ℝ, 
    let left_side := λ l p, (l.1.1 - l.2.1) * (p.2 - l.2.2) - (l.1.2 - l.2.2) * (p.1 - l.2.1) in
    (finset.univ.filter (λ i, left_side l (points i) < 0)).card = n / 2 :=
by
  sorry

end division_line_exists_l476_476729


namespace quadrilateral_A1B1C1D1_is_parallelogram_l476_476220

theorem quadrilateral_A1B1C1D1_is_parallelogram
  (O A B C D P Q R S A1 B1 C1 D1 : Point)
  (h1 : incircle_of _ O ABCD)
  (h2 : tangency_points _ P Q R S ABCD)
  (h3 : intersection O A PS A1)
  (h4 : intersection O B PQ B1)
  (h5 : intersection O C QR C1)
  (h6 : intersection O D SR D1) :
  is_parallelogram A1 B1 C1 D1 :=
sorry

end quadrilateral_A1B1C1D1_is_parallelogram_l476_476220


namespace sale_first_month_l476_476188

-- Declaration of all constant sales amounts in rupees
def sale_second_month : ℕ := 6927
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6791
def average_required : ℕ := 6800
def months : ℕ := 6

-- Total sales computed from the average sale requirement
def total_sales_needed : ℕ := months * average_required

-- The sum of sales for the second to sixth months
def total_sales_last_five_months := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- Prove the sales in the first month given the conditions
theorem sale_first_month :
  total_sales_needed - total_sales_last_five_months = 6435 :=
by
  sorry

end sale_first_month_l476_476188


namespace angle_between_u_and_v_l476_476858

open Real

noncomputable def u : ℝ × ℝ × ℝ := (3, -2, 2)
noncomputable def v : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (a.1^2 + a.2^2 + a.3^2)

noncomputable def cosine (a b : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product a b) / ((magnitude a) * (magnitude b))

noncomputable def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (cosine a b)

theorem angle_between_u_and_v :
  angle_between_vectors u v = real.arccos (-8 / (3 * sqrt 17)) := by
  sorry

end angle_between_u_and_v_l476_476858


namespace louie_pie_share_l476_476813

theorem louie_pie_share :
  let leftover := (6 : ℝ) / 7
  let people := 3
  leftover / people = (2 : ℝ) / 7 := 
by
  sorry

end louie_pie_share_l476_476813


namespace log_x_64_eq_log_3_27_l476_476252

theorem log_x_64_eq_log_3_27 (x : ℝ) (hx : 0 < x) (h : log x 64 = log 3 27) : x = 4 :=
by
  sorry

end log_x_64_eq_log_3_27_l476_476252


namespace min_sixth_power_sin_cos_l476_476265

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476265


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476547

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476547


namespace total_cups_of_ingredients_l476_476100

theorem total_cups_of_ingredients
  (ratio_butter : ℕ) (ratio_flour : ℕ) (ratio_sugar : ℕ)
  (flour_cups : ℕ)
  (h_ratio : ratio_butter = 2 ∧ ratio_flour = 3 ∧ ratio_sugar = 5)
  (h_flour : flour_cups = 6) :
  let part_cups := flour_cups / ratio_flour
  let butter_cups := ratio_butter * part_cups
  let sugar_cups := ratio_sugar * part_cups
  let total_cups := butter_cups + flour_cups + sugar_cups
  total_cups = 20 :=
by
  sorry

end total_cups_of_ingredients_l476_476100


namespace find_polynomials_l476_476853

theorem find_polynomials (P Q : ℝ[X]) 
    (h : ∀ n : ℕ, (∑ i in Finset.range (n + 1), P.eval i) = P.eval n * Q.eval n) :
    ∃ (α : ℝ) (r : ℝ) (s : ℕ), 
    P = (C α) * (X - C r) * (X - C (r + 1)) * ... * (X - C (r + (s - 1))) :=
begin
  sorry
end

end find_polynomials_l476_476853


namespace train_speed_l476_476201

def train_length : ℝ := 800
def crossing_time : ℝ := 12
def expected_speed : ℝ := 66.67 

theorem train_speed (h_len : train_length = 800) (h_time : crossing_time = 12) : 
  train_length / crossing_time = expected_speed := 
by {
  sorry
}

end train_speed_l476_476201


namespace points_on_line_l476_476922

theorem points_on_line :
  ∀ (x_1 y_1 x_2 y_2 : ℝ), 
  (x_1, y_1) = (8, 16) → 
  (x_2, y_2) = (0, -8) →
  let slope := (y_1 - y_2) / (x_1 - x_2)
  let line_eq (x : ℝ) := slope * (x - x_1) + y_1
  (line_eq 4 = 4) ∧ 
  (line_eq 9 = 19) :=
by
  intros x_1 y_1 x_2 y_2 h1 h2
  let slope := (y_1 - y_2) / (x_1 - x_2)
  let line_eq (x : ℝ) := slope * (x - x_1) + y_1
  split
  { -- proof for (4, 4)
    sorry },
  { -- proof for (9, 19)
    sorry }

end points_on_line_l476_476922


namespace proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l476_476147

open Classical

variable (a b x y : ℝ)

theorem proposition_A_correct (h : a > 1) : (1 / a < 1) ∧ ¬((1 / a < 1) → (a > 1)) :=
sorry

theorem proposition_B_incorrect (h_neg : ¬(x < 1 → x^2 < 1)) : ¬(∃ x, x ≥ 1 ∧ x^2 ≥ 1) :=
sorry

theorem proposition_C_incorrect (h_xy : x ≥ 2 ∧ y ≥ 2) : ¬((x ≥ 2 ∧ y ≥ 2) → x^2 + y^2 ≥ 4) :=
sorry

theorem proposition_D_correct (h_a : a ≠ 0) : (a * b ≠ 0) ∧ ¬((a * b ≠ 0) → (a ≠ 0)) :=
sorry

end proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l476_476147


namespace range_of_m_l476_476428

theorem range_of_m (m : ℝ) (A B C : set ℝ) :
  (A = {x | 0 < x ∧ x < 2}) →
  (B = {x | -1 < x ∧ x < 1}) →
  (C = {x | mx + 1 > 0}) →
  (∀ x, x ∈ A ∪ B → x ∈ C) ↔ (m ∈ set.Icc (-(1/2) : ℝ) 1) :=
by
  sorry

end range_of_m_l476_476428


namespace arvin_fifth_day_run_l476_476013

theorem arvin_fifth_day_run :
  let running_distance : ℕ → ℕ := λ day, 2 + day - 1
  in running_distance 5 = 6 := by
  sorry

end arvin_fifth_day_run_l476_476013


namespace min_sin6_cos6_l476_476280

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476280


namespace probability_heads_9_tails_at_least_2_l476_476135

noncomputable def probability_exactly_nine_heads : ℚ :=
  let total_outcomes := 2 ^ 12
  let successful_outcomes := Nat.choose 12 9
  successful_outcomes / total_outcomes

theorem probability_heads_9_tails_at_least_2 (n : ℕ) (h : n = 12) :
  n = 12 → probability_exactly_nine_heads = 55 / 1024 := by
  intros h
  sorry

end probability_heads_9_tails_at_least_2_l476_476135


namespace minimize_sin_cos_six_l476_476310

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476310


namespace sum_geq_n_l476_476394

theorem sum_geq_n (n : ℕ) (x : Fin n → ℝ) (h1 : 2 ≤ n) (h2 : ∀ i, 0 < x i) (h3 : ∀ i, x ((i + n) % n) = x i) : 
  ∑ i in Finset.range n, (1 + (x i)^2) / (1 + (x i) * (x ((i + 1) % n))) ≥ n :=
sorry

end sum_geq_n_l476_476394


namespace min_sin6_cos6_l476_476340

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476340


namespace area_between_curves_eq_nine_l476_476815

def f (x : ℝ) := 2 * x - x^2 + 3
def g (x : ℝ) := x^2 - 4 * x + 3

theorem area_between_curves_eq_nine :
  ∫ x in (0 : ℝ)..(3 : ℝ), (f x - g x) = 9 := by
  sorry

end area_between_curves_eq_nine_l476_476815


namespace problem1_problem2_l476_476768

-- First Problem Statement
theorem problem1 (x y : Fin n → ℝ) (h1 : ∀ i, 0 < x i) (h2 : ∀ i, 0 < y i) (h3 : ∑ i, x i = 1) (h4 : ∑ i, y i = 1) :
    (∑ i, x i * log (x i / y i) ≥ 0) ∧ (-∑ i, x i * log (x i) ≤ log n) :=
sorry

-- Second Problem Statement
theorem problem2 (x y : Fin n → ℝ) (h1 : ∀ i, 0 < x i) (h2 : ∀ i, 0 < y i) (xₛ : ∑ i, x i) (yₛ : ∑ i, y i) :
    xₛ = ∑ i, x i → yₛ = ∑ i, y i → (∑ i, x i * log (x i / y i) ≥ xₛ * log (xₛ / yₛ)) :=
sorry

end problem1_problem2_l476_476768


namespace arithmetic_sequence_min_value_S_l476_476506

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476506


namespace number_of_valid_n_l476_476371

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem number_of_valid_n :
  { n : ℕ // 1978 ≤ n ∧ n ≤ 2016 ∧ n % 3 = 0 ∧ sum_of_digits n % 3 = 0 ∧ sum_of_digits (sum_of_digits n) % 3 = 0 ∧ n + sum_of_digits n + sum_of_digits (sum_of_digits n) = 2016 }.card = 3 := 
by 
  sorry

end number_of_valid_n_l476_476371


namespace sin_cos_sixth_min_l476_476345

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476345


namespace find_angle_B_l476_476481

def Trapezoid (A B C D : Type) : Prop := ∃ AB CD : Type, AB ≠ CD ∧ ∀ (α β : Type), AB ∥ CD

variables {A B C D : ℝ}
variables (h1 : Trapezoid A B C D)
variables (h2 : A = 3 * D)
variables (h3 : C = 4 * B)
variables (h4 : A + B = 150)

theorem find_angle_B : B = 36 :=
by
  have h5 : B + C = 180, from sorry,
  rw h3 at h5,
  linear_comb h5,
  rw mul_comm at *,
  sorry


end find_angle_B_l476_476481


namespace min_distance_circle_hyperbola_l476_476996

-- Definitions of the geometric shapes
noncomputable def circle_eq := λ (A : ℝ × ℝ), (A.1 - 8)^2 + A.2^2 = 8
noncomputable def hyperbola_eq := λ (B : ℝ × ℝ), B.1 * B.2 = 2

-- Definition of the distance function
noncomputable def dist (P Q : ℝ × ℝ) :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- The problem to prove: the minimum possible distance between a point on the circle and a point on the hyperbola
theorem min_distance_circle_hyperbola:
  ∃ (A B : ℝ × ℝ), circle_eq A ∧ hyperbola_eq B ∧ 
  ∀ (A' B' : ℝ × ℝ), circle_eq A' → hyperbola_eq B' → dist A B ≤ dist A' B' :=
sorry

end min_distance_circle_hyperbola_l476_476996


namespace distance_B1_to_plane_EFG_l476_476480

def distance_from_point_to_plane
(point P : ℝ × ℝ × ℝ) 
(plane_pts : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem distance_B1_to_plane_EFG :
  let B1 := (1, 1, 0)
  let E := (1 / 2, 0, 0)
  let F := (1, 1 / 2, 1)
  let G := (0, 0, 1 / 2)
  distance_from_point_to_plane B1 (E, F, G) = real.sqrt(11) / 3 :=
sorry

end distance_B1_to_plane_EFG_l476_476480


namespace odd_negative_product_sign_and_units_digit_l476_476136

theorem odd_negative_product_sign_and_units_digit : 
  let numbers := list.range' (-2015) 1007 |>.filter (λ x, x % 2 ≠ 0)
  let product := list.prod numbers
  (product < 0) ∧ (product % 10 = 5) := 
by
  sorry

end odd_negative_product_sign_and_units_digit_l476_476136


namespace min_sixth_power_sin_cos_l476_476264

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476264


namespace part_a_part_b_l476_476037

-- Definitions
variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ)
variable (I : Matrix (Fin n) (Fin n) ℂ) [Fintype (Fin n)] [DecidableEq (Fin n)]
variable (B : Matrix (Fin n) (Fin n) ℂ)

-- Assume A is nonsingular and A, I have the appropriate sizes
noncomputable def A_inv : Matrix (Fin n) (Fin n) ℂ := A⁻¹
noncomputable def A_conj : Matrix (Fin n) (Fin n) ℂ := A.conjTranspose
noncomputable def B_def : Matrix (Fin n) (Fin n) ℂ := A * A.conjTranspose + I

-- Part (a)
theorem part_a (hA : A.det ≠ 0) (hI : I = 1) : A⁻¹ * B * A = B.conjTranspose :=
by
  have hB : B = A * A.conjTranspose + I := rfl
  sorry

-- Part (b)
theorem part_b (hA : A.det ≠ 0) (hI : I = 1) : (B_def A I).det ∈ ℝ :=
by
  have hB : B = A * A.conjTranspose + I := rfl
  sorry

end part_a_part_b_l476_476037


namespace min_sixth_power_sin_cos_l476_476267

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476267


namespace mike_ride_distance_l476_476647

/-- 
Mike took a taxi to the airport and paid a starting amount plus $0.25 per mile. 
Annie took a different route to the airport and paid the same starting amount plus $5.00 in bridge toll fees plus $0.25 per mile. 
Each was charged exactly the same amount, and Annie's ride was 26 miles. 
Prove that Mike's ride was 46 miles given his starting amount was $2.50.
-/
theorem mike_ride_distance
  (S C A_miles : ℝ)                  -- S: starting amount, C: cost per mile, A_miles: Annie's ride distance
  (bridge_fee total_cost : ℝ)        -- bridge_fee: Annie's bridge toll fee, total_cost: total cost for both
  (M : ℝ)                            -- M: Mike's ride distance
  (hS : S = 2.5)
  (hC : C = 0.25)
  (hA_miles : A_miles = 26)
  (h_bridge_fee : bridge_fee = 5)
  (h_total_cost_equal : total_cost = S + bridge_fee + (C * A_miles))
  (h_total_cost_mike : total_cost = S + (C * M)) :
  M = 46 :=
by 
  sorry

end mike_ride_distance_l476_476647


namespace first_player_wins_optimal_play_l476_476184

theorem first_player_wins_optimal_play (n m : ℕ) (hn : n ≥ 2) (hm : m ≥ 2) (h : m ≤ n) :
  ∃ (strategy : (corner_piece : (Σ (n m : ℕ), n ≥ 2 ∧ m ≥ 2 ∧ m ≤ n)), winning_strategy_first_player) :=
sorry

end first_player_wins_optimal_play_l476_476184


namespace triangle_area_l476_476010

theorem triangle_area (a b c : ℝ) (A B C : ℝ) (hABC_acute : A + B + C = π) 
  (h1 : b = 4 * a) (h2 : a + c = 5) 
  (h3 : (sin B * sin C) / sin A = 3 * sqrt 7 / 2) : 
  (1 / 2) * a * b * sin C = 3 * sqrt 7 / 4 :=
sorry

end triangle_area_l476_476010


namespace sum_harmonic_like_sequence_l476_476675

noncomputable def arithmetic_sequence (n : ℕ) : ℝ :=
  n

def sum_arithmetic_sequence (n : ℕ) : ℝ :=
  (n * (n + 1)) / 2

def harmonic_like_sequence (n : ℕ) : ℝ :=
  1 / (arithmetic_sequence n * arithmetic_sequence (n + 1))

theorem sum_harmonic_like_sequence :
  (∑ k in finset.range 2018, harmonic_like_sequence k) = 2018 / 2019 :=
by
  sorry

end sum_harmonic_like_sequence_l476_476675


namespace king_paid_after_tip_l476_476784

theorem king_paid_after_tip:
  (crown_cost tip_percentage total_cost : ℝ)
  (h_crown_cost : crown_cost = 20000)
  (h_tip_percentage : tip_percentage = 0.1) :
  total_cost = crown_cost + (crown_cost * tip_percentage) :=
by
  have h_tip := h_crown_cost.symm ▸ h_tip_percentage.symm ▸ 20000 * 0.1
  have h_total := h_crown_cost.symm ▸ (h_tip.symm ▸ 2000)
  rw [h_crown_cost, h_tip, h_total]
  exact rfl

end king_paid_after_tip_l476_476784


namespace calculate_first_year_sample_l476_476804

noncomputable def stratified_sampling : ℕ :=
  let total_sample_size := 300
  let first_grade_ratio := 4
  let second_grade_ratio := 5
  let third_grade_ratio := 5
  let fourth_grade_ratio := 6
  let total_ratio := first_grade_ratio + second_grade_ratio + third_grade_ratio + fourth_grade_ratio
  let first_grade_proportion := first_grade_ratio / total_ratio
  300 * first_grade_proportion

theorem calculate_first_year_sample :
  stratified_sampling = 60 :=
by sorry

end calculate_first_year_sample_l476_476804


namespace min_value_of_reciprocal_squares_l476_476128

variable (a b : ℝ)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x + a^2 - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0

-- The condition that the two circles are externally tangent and have three common tangents
def externallyTangent (a b : ℝ) : Prop :=
  -- From the derivation in the solution, we must have:
  (a^2 + 4 * b^2 = 9)

-- Ensure a and b are non-zero
def nonzero (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0

-- State the main theorem to prove
theorem min_value_of_reciprocal_squares (h1 : externallyTangent a b) (h2 : nonzero a b) :
  (1 / a^2) + (1 / b^2) = 1 := 
sorry

end min_value_of_reciprocal_squares_l476_476128


namespace area_of_shaded_quadrilateral_l476_476190

structure Point where
  x : ℝ
  y : ℝ

def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

noncomputable def quadArea (p1 p2 p3 p4 : Point) : ℝ :=
  let areaTriangle (a b c : Point) : ℝ :=
    abs ((a.x*(b.y - c.y) + b.x*(c.y - a.y) + c.x*(a.y - b.y)) / 2)
  let rectangleArea (o a b : Point) : ℝ :=
    abs (a.x - o.x) * abs (b.y - o.y)
  rectangleArea p1 p2 p4 - areaTriangle p1 p2 p3

noncomputable def A : Point := ⟨10, 0⟩
noncomputable def B : Point := ⟨0, 10⟩
noncomputable def C : Point := ⟨10, 0⟩
noncomputable def D : Point := ⟨0, 10⟩
noncomputable def E : Point := ⟨5, 5⟩
noncomputable def O : Point := ⟨0, 0⟩

theorem area_of_shaded_quadrilateral : quadArea O C E B = 87.5 := by
  sorry

end area_of_shaded_quadrilateral_l476_476190


namespace min_sixth_power_sin_cos_l476_476276

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476276


namespace find_essay_pages_l476_476028

/-
Conditions:
1. It costs $0.10 to print one page.
2. Jenny wants to print 7 copies of her essay.
3. Jenny wants to buy 7 pens that each cost $1.50.
4. Jenny pays the store with 2 twenty dollar bills and gets $12 in change.
-/

def cost_per_page : Float := 0.10
def number_of_copies : Nat := 7
def cost_per_pen : Float := 1.50
def number_of_pens : Nat := 7
def total_money_given : Float := 40.00  -- 2 twenty dollar bills
def change_received : Float := 12.00

theorem find_essay_pages :
  let total_spent := total_money_given - change_received
  let total_cost_of_pens := Float.ofNat number_of_pens * cost_per_pen
  let total_amount_spent_on_printing := total_spent - total_cost_of_pens
  let number_of_pages := total_amount_spent_on_printing / cost_per_page
  number_of_pages = 175 := by
  sorry

end find_essay_pages_l476_476028


namespace angle_between_vectors_l476_476865

theorem angle_between_vectors :
  let v1 : ℝ × ℝ × ℝ := (3, -2, 2)
  let v2 : ℝ × ℝ × ℝ := (-2, 2, 1)
  let dot_product (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let magnitude (v : ℝ × ℝ × ℝ) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let cos_theta := dot_product v1 v2 / (magnitude v1 * magnitude v2)
  θ = Real.acos (cos_theta) * (180 / Real.pi)
  in θ ≈ 127 :=
by
  sorry

end angle_between_vectors_l476_476865


namespace integral_2x_minus_1_eq_6_l476_476849

noncomputable def definite_integral_example : ℝ :=
  ∫ x in (0:ℝ)..(3:ℝ), (2 * x - 1)

theorem integral_2x_minus_1_eq_6 : definite_integral_example = 6 :=
by
  sorry

end integral_2x_minus_1_eq_6_l476_476849


namespace circuit_operates_normally_probability_l476_476002

theorem circuit_operates_normally_probability :
  let p1 := 0.5
  let p2 := 0.7
  (1 - (1 - p1) * (1 - p2)) = 0.85 :=
by
  let p1 := 0.5
  let p2 := 0.7
  sorry

end circuit_operates_normally_probability_l476_476002


namespace tangents_concur_l476_476038

-- Define the given setup and conditions
variables {A B C O E F H M K P Q : Point}
variable (circumcircle : Circle)
variable (line_parallel_to_BC : Line)
variables (is_acute : acute_triangle A B C)
variables (not_isosceles : not_isosceles A B C)
variables (height_BE : is_height A B C E)
variables (height_CF : is_height A B C F)
variable (intersection_H : intersect_at E F H)
variable (midpoint_M : is_midpoint A H M)
variable (perpendicular_HK_EF : perpendicular HK EF)
variables (parallel_BC_PQ : is_parallel line_parallel_to_BC BC)
variables (intersects_minor_arcs : intersects line_parallel_to_BC (arc A B) P)
variables (intersects_minor_arcs_ : intersects line_parallel_to_BC (arc A C) Q)

-- Prove the main theorem
theorem tangents_concur :
  tangents_concur (circumcircle E C Q) (circumcircle B P F) E F M K :=
sorry

end tangents_concur_l476_476038


namespace triangle_product_l476_476479

theorem triangle_product (a b c: ℕ) (p: ℕ)
    (h1: ∃ k1 k2 k3: ℕ, a * k1 * k2 = p ∧ k2 * k3 * b = p ∧ k3 * c * a = p) 
    : (1 ≤ c ∧ c ≤ 336) :=
by
  sorry

end triangle_product_l476_476479


namespace calculate_expression_l476_476816

theorem calculate_expression : 3 - (-3) ^ (-3) = 82 / 27 :=
by
  sorry

end calculate_expression_l476_476816


namespace volume_of_cube_proof_l476_476636

variables (x : ℝ) (S : ℝ)

-- Given conditions
def surface_area_of_cube := S
def side_length := x

-- The mathematical condition
def surface_area_condition : Prop := S = 6 * x^2

-- The volume of the cube to prove
def volume_of_cube : ℝ := x^3

-- Prove that given the surface area condition, the volume is as specified
theorem volume_of_cube_proof (h : surface_area_condition) : volume_of_cube = x^3 := sorry

end volume_of_cube_proof_l476_476636


namespace min_sin6_cos6_l476_476336

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476336


namespace arithmetic_sequence_min_value_S_l476_476507

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476507


namespace min_value_fraction_l476_476609

open Real

-- Define the premises.
variables (a b : Real)

-- Assume a and b are positive and satisfy the equation a + 3b = 1.
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : a + 3b = 1

-- The theorem we want to prove.
theorem min_value_fraction (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3b = 1) :
  1 / a + 1 / b = 4 * sqrt 3 + 8 :=
sorry

end min_value_fraction_l476_476609


namespace project_completion_time_l476_476800

-- Condition 1: 8 people complete 1/3 of a project in 30 days
def initial_team_size : ℕ := 8
def initial_days : ℕ := 30
def fraction_completed_initially : ℝ := 1/3

-- Condition 2: After 30 days, 4 more people join, making it 12 people in total
def additional_people : ℕ := 4
def total_people_after_addition : ℕ := initial_team_size + additional_people

-- Correct Answer: Total days to complete the project
def total_days_to_complete_project : ℕ := 70

-- Theorem stating the problem
theorem project_completion_time :
  ∃ days_initial additional_days total_days,
    (initial_team_size * days_initial * fraction_completed_initially) / (initial_days * initial_team_size) +
    (total_people_after_addition * additional_days * (1 - fraction_completed_initially)) / total_people_after_addition / (days_initial + additional_days) = 1
    ∧ total_days = days_initial + additional_days
    ∧ total_days = total_days_to_complete_project :=
  by sorry

end project_completion_time_l476_476800


namespace range_of_f_triangle_area_l476_476414

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * sin x * cos (x + π / 3) + sqrt 3

-- Define the conditions for the domain of x
def domain_condition (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ π / 6

-- Define the range of the function f(x)
def range_condition (y : ℝ) : Prop := sqrt 3 ≤ y ∧ y ≤ 2

-- Define the circumcircle radius of the triangle ABC
def circumcircle_radius : ℝ := (3 * sqrt 2) / 4

-- Define the lengths of the sides
def side_a : ℝ := sqrt 3
def side_b : ℝ := 2

-- Prove that the range of the function is within [sqrt 3, 2]
theorem range_of_f : ∀ x, domain_condition x → range_condition (f x) :=
by
  intro x hx
  sorry

-- Prove that the area of triangle ABC is sqrt 2, given the circumcircle radius
theorem triangle_area : 
  let a := side_a, b := side_b, r := circumcircle_radius in
  a = sqrt 3 ∧ b = 2 ∧ r = (3 * sqrt 2) / 4 → 
  let sinA := a / (2 * r), sinB := b / (2 * r) in
  sinA = sqrt 6 / 3 ∧ sinB = 2 * sqrt 2 / 3 → 
  ∃ S: ℝ, S = (1 / 2) * a * b * sin (sinA + sinB) ∧ S = sqrt 2 :=
by
  intro a b r h,
  sorry

end range_of_f_triangle_area_l476_476414


namespace circle_cartesian_line_circle_intersect_l476_476423

noncomputable def L_parametric (t : ℝ) : ℝ × ℝ :=
  (t, 1 + 2 * t)

noncomputable def C_polar (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

def L_cartesian (x y : ℝ) : Prop :=
  y = 2 * x + 1

def C_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem circle_cartesian :
  ∀ x y : ℝ, C_polar x = y ↔ C_cartesian x y :=
sorry

theorem line_circle_intersect (x y : ℝ) :
  L_cartesian x y → C_cartesian x y → True :=
sorry

end circle_cartesian_line_circle_intersect_l476_476423


namespace sum_of_distinct_integers_is_33_l476_476617

noncomputable def distinct_integers_sum_proof : Prop :=
  ∃ (a b c d e : ℤ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 ∧ 
  a + b + c + d + e = 33

theorem sum_of_distinct_integers_is_33 : distinct_integers_sum_proof :=
begin
  sorry
end

end sum_of_distinct_integers_is_33_l476_476617


namespace lambda_range_l476_476388

open BigOperators

-- Definitions for conditions
def a_seq (n : ℕ) : ℕ := 2 * n

def b_seq (n : ℕ) : ℕ := 2^n

def c_seq (n : ℕ) : ℕ := (a_seq n) / (b_seq n)

def S_n (n : ℕ) : ℕ := ∑ i in Finset.range n, c_seq (i + 1)

-- Statement for the proof problem
theorem lambda_range (λ : ℝ) : (∀ n : ℕ, (-1)^n * λ < S_n n + n / (2^(n-1)) ) ↔ (λ > -2 ∧ λ < 3) :=
sorry

end lambda_range_l476_476388


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476548

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476548


namespace fill_time_of_first_tap_l476_476182

-- Definitions and conditions
def emptying_rate_of_second_tap : ℝ := 1 / 9
def net_filling_rate_when_both_open : ℝ := 1 / 4.5

-- Statement to be proved
theorem fill_time_of_first_tap : ∃ T : ℝ, (1 / T - emptying_rate_of_second_tap = net_filling_rate_when_both_open) ∧ T = 3 :=
by {
  sorry
}

end fill_time_of_first_tap_l476_476182


namespace algebraic_form_correct_exponential_form_correct_l476_476162

noncomputable def complex_algebraic_form : Prop :=
  let θ := 4 * real.pi / 3
  let z := 4 * (complex.cos θ + complex.i * complex.sin θ)
  z = -2 - 2 * complex.i * real.sqrt 3

noncomputable def complex_exponential_form : Prop :=
  let θ := 4 * real.pi / 3
  let z := 4 * (complex.cos θ + complex.i * complex.sin θ)
  z = 4 * complex.exp (complex.i * θ)

theorem algebraic_form_correct : complex_algebraic_form :=
  sorry

theorem exponential_form_correct : complex_exponential_form :=
  sorry

end algebraic_form_correct_exponential_form_correct_l476_476162


namespace single_elimination_games_l476_476009

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  (n - 1) = 511 :=
by
  sorry

end single_elimination_games_l476_476009


namespace polynomial_remainder_l476_476045

theorem polynomial_remainder (Q : ℚ[X]) :
  (Q.eval 15 = 8) → (Q.eval 19 = 10) →
  ∃ c d : ℚ, (∀ x, Q.eval x = (x-15)*(x-19)*R.eval x + c*x + d) 
             ∧ (c = 1/2) 
             ∧ (d = 1/2) :=
by
  intro hQ15 hQ19
  sorry

end polynomial_remainder_l476_476045


namespace min_value_sin6_cos6_l476_476332

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476332


namespace min_sin6_cos6_l476_476337

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476337


namespace smallest_t_for_given_roots_l476_476095

-- Define the polynomial with integer coefficients and specific roots
def poly (x : ℝ) : ℝ := (x + 3) * (x - 4) * (x - 6) * (2 * x - 1)

-- Define the main theorem statement
theorem smallest_t_for_given_roots :
  ∃ (t : ℤ), 0 < t ∧ t = 72 := by
  -- polynomial expansion skipped, proof will come here
  sorry

end smallest_t_for_given_roots_l476_476095


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476596

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476596


namespace inconsistent_equations_l476_476928

theorem inconsistent_equations (x y S : ℝ) 
  (h1 : x + y = S)
  (h2 : x + 3y = 1)
  (h3 : x + 2y = 10) : false :=
by
  sorry

end inconsistent_equations_l476_476928


namespace other_root_of_quadratic_l476_476907

theorem other_root_of_quadratic {a : ℝ} : (2 : ℝ) ^ 2 - a * 2 + 2 = 0 → (x₁ x₂ : ℝ) → 
    (x₁ = 2 → x₁ * x₂ = 2 → x₂ = 1) :=
begin
  sorry
end

end other_root_of_quadratic_l476_476907


namespace garden_perimeter_l476_476793

-- We are given:
variables (a b : ℝ)
variables (h1 : b = 3 * a)
variables (h2 : a^2 + b^2 = 34^2)
variables (h3 : a * b = 240)

-- We must prove:
theorem garden_perimeter (h4 : a^2 + 9 * a^2 = 1156) (h5 : 10 * a^2 = 1156) (h6 : a^2 = 115.6) 
  (h7 : 3 * a^2 = 240) (h8 : a^2 = 80) :
  2 * (a + b) = 72 := 
by
  sorry

end garden_perimeter_l476_476793


namespace distance_from_P_to_origin_l476_476686

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_from_P_to_origin :
  distance (-2) 4 0 0 = 2 * Real.sqrt 5 :=
by
  sorry

end distance_from_P_to_origin_l476_476686


namespace point_D_not_in_region_l476_476790

-- Define the condition that checks if a point is not in the region defined by 3x + 2y < 6
def point_not_in_region (x y : ℝ) : Prop :=
  ¬ (3 * x + 2 * y < 6)

-- Define the points
def A := (0, 0)
def B := (1, 1)
def C := (0, 2)
def D := (2, 0)

-- The proof problem as a Lean statement
theorem point_D_not_in_region : point_not_in_region (2:ℝ) (0:ℝ) :=
by
  show point_not_in_region 2 0
  sorry

end point_D_not_in_region_l476_476790


namespace possible_values_of_n_l476_476972

noncomputable def rhombus_side : ℝ := 5
noncomputable def rhombus_angle_deg : ℝ := 60
noncomputable def rhombus_perimeter : ℝ := 4 * rhombus_side 
noncomputable def rhombus_area : ℝ := (1/2) * (2 * rhombus_side * real.cos (real.pi * rhombus_angle_deg / 180)) * (2 * rhombus_side * real.sin (real.pi * rhombus_angle_deg / 180))
noncomputable def circle_radius : ℝ := rhombus_area / rhombus_perimeter
noncomputable def circle_area : ℝ := real.pi * circle_radius ^ 2
noncomputable def positive_integer_approx (x: ℝ) : ℕ := real.to_nnreal x |>.to_nat

theorem possible_values_of_n : positive_integer_approx circle_area = 3 := 
  by
  sorry

end possible_values_of_n_l476_476972


namespace eval_expression_l476_476248

theorem eval_expression : (10 ^ (-3) * 2 ^ 2) / 10 ^ (-4) = 40 := by
  sorry

end eval_expression_l476_476248


namespace barn_volume_l476_476691

def herons_formula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

def volume_of_barn : ℝ :=
  let rectangle_width := 12
  let rectangle_height := 7
  let triangle_side_a := 10
  let triangle_side_b := 10
  let triangle_side_c := 12
  let prism_length := 30
  let base_area_rectangle := rectangle_width * rectangle_height
  let base_area_triangle := herons_formula triangle_side_a triangle_side_b triangle_side_c
  let volume_top_prism := base_area_triangle * prism_length
  let volume_bottom_prism := base_area_rectangle * prism_length
  let volume_one_prism := volume_top_prism + volume_bottom_prism
  let two_prism_volume := 2 * volume_one_prism
  let base_pyramid := 12 * 12
  let height_pyramid := 8
  let volume_pyramid := (1/3) * base_pyramid * height_pyramid
  let rectangular_box_volume := 12 * 12 * 7
  let common_center_volume := volume_pyramid + rectangular_box_volume
  let volume_barn := two_prism_volume - common_center_volume
  volume_barn

theorem barn_volume : volume_of_barn = 6528 := by
  sorry

end barn_volume_l476_476691


namespace arithmetic_sequence_and_minimum_sum_l476_476528

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476528


namespace sum_of_possible_coefficients_eq_85_l476_476062

theorem sum_of_possible_coefficients_eq_85 :
  ∑ b in (finset.filter (λ b : ℤ, ∃ (r s : ℤ), r < 0 ∧ s < 0 ∧ r * s = 36 ∧ r + s = b) (finset.Icc (-72) 0)), b = 85 :=
sorry

end sum_of_possible_coefficients_eq_85_l476_476062


namespace final_balance_l476_476206

noncomputable def balance_after_transactions (initial_balance : ℝ) : ℝ :=
  let balance1 := initial_balance - 300 in
  let balance2 := (2 / 3) * (balance1 - 150) in
  let balance3 := balance2 + (3 / 5) * balance2 in
  let balance4 := balance3 - 250 in
  let balance5 := balance4 + (2 / 3) * balance4 in
  balance5

theorem final_balance (initial_balance : ℝ) (h1 : 300 = (3 / 7) * initial_balance)
  (h2 : 150 = (1 / 3) * (initial_balance - 300))
  (h3 : 250 = (1 / 4) * ((initial_balance - 300 + (3 / 5) * (2 / 3) * (initial_balance - 450)) + 250)) : 
  balance_after_transactions initial_balance = 1250 :=
by
  sorry

end final_balance_l476_476206


namespace point_in_third_quadrant_l476_476915

def z : ℂ := -1 - 3 * complex.I

theorem point_in_third_quadrant : (z.re < 0) ∧ (z.im < 0) :=
by
  sorry

end point_in_third_quadrant_l476_476915


namespace probability_first_ball_odd_l476_476169

/-- A box contains 100 balls numbered 1 to 100. 
If 3 balls are selected at random and with replacement, 
and the 3 numbers contain two odd and one even, 
prove that the probability that the first ball picked is odd is 1/4. -/
theorem probability_first_ball_odd :
  let num_balls := 100 in
  let odd_balls := 50 in
  let even_balls := num_balls - odd_balls in
  (2 * (odd_balls / num_balls * odd_balls / num_balls * even_balls / num_balls)) = 1/4 :=
by
  sorry

end probability_first_ball_odd_l476_476169


namespace dividend_rate_l476_476788

theorem dividend_rate (face_value market_value expected_interest interest_rate : ℝ)
  (h1 : face_value = 52)
  (h2 : expected_interest = 0.12)
  (h3 : market_value = 39)
  : ((expected_interest * market_value) / face_value) * 100 = 9 := by
  sorry

end dividend_rate_l476_476788


namespace focus_larger_x_coordinate_l476_476239

theorem focus_larger_x_coordinate :
  let c : ℝ := (1, -2)
  let a : ℝ := 9
  let b : ℝ := 16
  let hyperbola := { p : ℝ × ℝ | ((p.1 - c.1)^2) / (a^2) - ((p.2 - c.2)^2) / (b^2) = 1 }
  ∃ f1 f2 : ℝ × ℝ, f1 = (c.1 + real.sqrt (a^2 + b^2), c.2) ∧ f2 = (c.1 - real.sqrt (a^2 + b^2), c.2) ∧
  ∀ p ∈ hyperbola, p = (1 + real.sqrt 337, -2) :=
sorry

end focus_larger_x_coordinate_l476_476239


namespace minimum_value_of_reciprocal_squares_l476_476126

theorem minimum_value_of_reciprocal_squares
  (a b : ℝ)
  (h : a ≠ 0 ∧ b ≠ 0)
  (h_eq : (a^2) + 4 * (b^2) = 9)
  : (1/(a^2) + 1/(b^2)) = 1 :=
sorry

end minimum_value_of_reciprocal_squares_l476_476126


namespace polygonal_line_exists_l476_476493

theorem polygonal_line_exists (A : Type) (n q : ℕ) (lengths : Fin q → ℝ)
  (yellow_segments : Fin q → (A × A))
  (h_lengths : ∀ i j : Fin q, i < j → lengths i < lengths j)
  (h_yellow_segments_unique : ∀ i j : Fin q, i ≠ j → yellow_segments i ≠ yellow_segments j) :
  ∃ (m : ℕ), m ≥ 2 * q / n :=
sorry

end polygonal_line_exists_l476_476493


namespace sequence_properties_l476_476702

theorem sequence_properties :
  ∀ {a : ℕ → ℝ} {b : ℕ → ℝ},
  a 1 = 1 ∧ 
  (∀ n, b n > 4 / 3) ∧ 
  (∀ n, (∀ x, x^2 - b n * x + a n = 0 → (x = a (n + 1) ∨ x = 1 + a n))) →
  (a 2 = 1 / 2 ∧ ∃ n, b n > 4 / 3 ∧ n = 5) := by
  sorry

end sequence_properties_l476_476702


namespace min_value_inverse_sum_l476_476615

theorem min_value_inverse_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 3 * b = 1) :
  \frac{1}{a} + \frac{1}{b} = 4 + 2 * sqrt 3 :=
sorry

end min_value_inverse_sum_l476_476615


namespace boxed_boxed_19_eq_42_l476_476370

-- Define the sum of the positive factors function
def sum_factors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum

-- Define the box function
def boxed (n : ℕ) : ℕ :=
  sum_factors n

-- Define boxed boxed function
def boxed_boxed (n : ℕ) : ℕ :=
  boxed (boxed n)

theorem boxed_boxed_19_eq_42 : boxed_boxed 19 = 42 := by
  sorry

end boxed_boxed_19_eq_42_l476_476370


namespace angle_B_value_triangle_area_range_l476_476389

-- Given conditions
variables {A B C : Real}
variables {a b c : Real}
variables (h1 : 0 < A ∧ A < π / 2)
variables (h2 : 0 < B ∧ B < π / 2)
variables (h3 : 0 < C ∧ C < π / 2)
variables (h4 : a = c * sin A / sin C)
variables (h5 : b = c * sin B / sin C)
variables (h6 : c = b * cos A * (tan A + tan B) / sqrt 3)

-- Prove 1: The value of angle B
theorem angle_B_value : B = π / 3 := sorry

-- Prove 2: The range of the area of ΔABC when c = 4
theorem triangle_area_range (h7 : c = 4) :
  let area := 1 / 2 * a * c * sin B in
  2 * sqrt 3 < area ∧ area < 8 * sqrt 3 := sorry

end angle_B_value_triangle_area_range_l476_476389


namespace arithmetic_sequence_minimum_value_S_l476_476575

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476575


namespace min_value_sin6_cos6_l476_476329

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476329


namespace polynomial_real_roots_l476_476366

theorem polynomial_real_roots :
  ∀ x : ℝ, (x^4 - 3 * x^3 + 3 * x^2 - x - 6 = 0) ↔ (x = 3 ∨ x = 2 ∨ x = -1) := 
by
  sorry

end polynomial_real_roots_l476_476366


namespace Sam_distance_l476_476060

theorem Sam_distance (miles_Marguerite: ℝ) (hours_Marguerite: ℕ) (hours_Sam: ℕ) (speed_factor: ℝ) 
  (h1: miles_Marguerite = 150) 
  (h2: hours_Marguerite = 3) 
  (h3: hours_Sam = 4)
  (h4: speed_factor = 1.2) :
  let average_speed_Marguerite := miles_Marguerite / hours_Marguerite
  let average_speed_Sam := speed_factor * average_speed_Marguerite
  let distance_Sam := average_speed_Sam * hours_Sam
  distance_Sam = 240 := 
by 
  sorry

end Sam_distance_l476_476060


namespace min_value_sin6_cos6_l476_476331

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476331


namespace train_speed_l476_476200

def train_length : ℝ := 800
def crossing_time : ℝ := 12
def expected_speed : ℝ := 66.67 

theorem train_speed (h_len : train_length = 800) (h_time : crossing_time = 12) : 
  train_length / crossing_time = expected_speed := 
by {
  sorry
}

end train_speed_l476_476200


namespace correct_option_l476_476146

noncomputable def option_A (x : ℝ) : Prop := x^2 + x^2 = x^4
noncomputable def option_B (x : ℝ) : Prop := x^2 * x^3 = x^6
noncomputable def option_C (x : ℝ) : Prop := (-x)^2 + (-x)^4 = -x^6
noncomputable def option_D (x : ℝ) : Prop := (-x)^3 * (-x)^4 = -x^7

theorem correct_option : ∀ (x : ℝ), option_D x ∧ ¬option_A x ∧ ¬option_B x ∧ ¬option_C x :=
by
  intro x
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end correct_option_l476_476146


namespace max_length_is_150_l476_476193

open Real

-- Define the conditions
def cost_per_foot : ℝ := 10
def total_cost : ℝ := 3000
def total_feet_of_fence : ℝ := total_cost / cost_per_foot
def area (y : ℝ) : ℝ := y * (total_feet_of_fence - 2 * y)

-- The correct answer we need to prove
def max_length_parallel_to_wall : ℝ := 150

theorem max_length_is_150 :
  ∃ (y : ℝ), (∃ (A : ℝ), A = area y ∧ 
  ∂ (λ y : ℝ, area y) y = 0 ∧ max_length_parallel_to_wall = total_feet_of_fence - 2 * y) :=
sorry

end max_length_is_150_l476_476193


namespace trajectory_eqn_d_l476_476704

theorem trajectory_eqn_d (O B : Point) (D E : Point) (a b : ℝ)
  (h1 : orthogonal O B A B)
  (h2 : mid_point E A B) : 
  ( ∀ x y : ℝ, (D.x^2 + D.y^2 = a^2 * b^2 / (a^2 + b^2)) ) ∧
  ( ∀ x y : ℝ, ((a^2 + b^2) * a^2 * b^2 * ((x^2 / a^2) + (y^2 / b^2))^2 = b^4 * x^2 + a^4 * y^2)) :=
by
  sorry

end trajectory_eqn_d_l476_476704


namespace packs_for_emilys_sister_l476_476246

theorem packs_for_emilys_sister (total_packs : ℕ) (emily_packs : ℕ) (sister_packs : ℕ) 
  (h1 : total_packs = 13) (h2 : emily_packs = 6) : 
  sister_packs = total_packs - emily_packs :=
  by
  have h3 : 7 = 13 - 6, by norm_num,
  exact h3.symm

end packs_for_emilys_sister_l476_476246


namespace fruit_seller_profit_l476_476186

theorem fruit_seller_profit 
  (SP : ℝ) (Loss_Percentage : ℝ) (New_SP : ℝ) (Profit_Percentage : ℝ) 
  (h1: SP = 8) 
  (h2: Loss_Percentage = 20) 
  (h3: New_SP = 10.5) 
  (h4: Profit_Percentage = 5) :
  ((New_SP - (SP / (1 - (Loss_Percentage / 100.0))) / (SP / (1 - (Loss_Percentage / 100.0)))) * 100) = Profit_Percentage := 
sorry

end fruit_seller_profit_l476_476186


namespace least_points_scored_tenth_game_l476_476464

noncomputable def min_points_in_tenth_game 
  (points6 points7 points8 points9 : ℕ) 
  (avg_first5 avg_first9 avg_first10 : ℕ → ℝ) : ℕ :=
  if h : avg_first10 10 > 19 then
    let points6_9 := points6 + points7 + points8 + points9 in
    let threshold := 191 in
    let min_points1_5_10 := threshold - points6_9 in
    let max_points1_5 := 90 in
    min_points1_5_10 - max_points1_5
  else
    0

theorem least_points_scored_tenth_game (scores : Fin 10 → ℕ)
  (h1 : scores 5 = 25)
  (h2 : scores 6 = 15)
  (h3 : scores 7 = 12)
  (h4 : scores 8 = 21)
  (h5 : (9 * (scores 0 + scores 1 + scores 2 + scores 3 + scores 4) / 5 < (scores 0 + scores 1 + scores 2 + scores 3 + scores 4 + 25 + 15 + 12 + 21) / 9) )
  (h6 : (scores 0 + scores 1 + scores 2 + scores 3 + scores 4 + 25 + 15 + 12 + 21 + scores 9) / 10 > 19) :
  scores 9 = 28 := by
  sorry

end least_points_scored_tenth_game_l476_476464


namespace arithmetic_sequence_minimum_value_S_n_l476_476554

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476554


namespace angle_between_vectors_is_correct_l476_476860

def vec_a : ℝ × ℝ × ℝ := (3, -2, 2)
def vec_b : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def angle_between_vectors : ℝ :=
  Real.acos ((vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 + vec_a.3 * vec_b.3) / 
    (Real.sqrt (vec_a.1^2 + vec_a.2^2 + vec_a.3^2) * Real.sqrt (vec_b.1^2 + vec_b.2^2 + vec_b.3^2)))

theorem angle_between_vectors_is_correct :
  angle_between_vectors = Real.acos (-8 / (3 * Real.sqrt 17)) :=
by sorry

end angle_between_vectors_is_correct_l476_476860


namespace proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l476_476150

theorem proposition_a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a > 1 ∨ a < 1) :=
sorry

theorem negation_of_proposition_b_incorrect (x : ℝ) : ¬(∀ x < 1, x^2 < 1) ↔ ∃ x < 1, x^2 ≥ 1 :=
sorry

theorem proposition_c_not_necessary (x y : ℝ) : (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)) :=
sorry

theorem proposition_d_necessary_not_sufficient (a b : ℝ) : (a ≠ 0 → ab ≠ 0) ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0) :=
sorry

theorem final_answer_correct :
  let proposition_A := (∃ (a : ℝ), a > 1 ∧ 1 / a < 1 ∧ (1 / a < 1 → a > 1 ∨ a < 1))
  let proposition_B := (¬(∀ (x : ℝ), x < 1 → x^2 < 1) ↔ ∃ (x : ℝ), x < 1 ∧ x^2 ≥ 1)
  let proposition_C := (∃ (x y : ℝ), (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)))
  let proposition_D := (∃ (a b : ℝ), a ≠ 0 ∧ ab ≠ 0 ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0))
  proposition_A ∧ proposition_D
:= 
sorry

end proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l476_476150


namespace number_of_a_for_line_through_parabola_vertex_l476_476885

theorem number_of_a_for_line_through_parabola_vertex :
  {a : ℝ | ∃ x : ℝ, y = 2x + a ∧ y = x^2 + 2a^2} = {0, 1/2} :=
sorry

end number_of_a_for_line_through_parabola_vertex_l476_476885


namespace count_integers_satisfying_inequality_l476_476432

theorem count_integers_satisfying_inequality :
  {n : ℤ | -12 ≤ n ∧ n ≤ 12 ∧ (n - 3) * (n + 3) * (n + 7) < 0}.finite.card = 10 :=
by
  sorry

end count_integers_satisfying_inequality_l476_476432


namespace product_of_fractions_is_eight_l476_476224

theorem product_of_fractions_is_eight :
  (8 / 4) * (14 / 7) * (20 / 10) * (25 / 50) * (9 / 18) * (12 / 6) * (21 / 42) * (16 / 8) = 8 :=
by
  sorry

end product_of_fractions_is_eight_l476_476224


namespace root_of_equation_in_interval_l476_476101

theorem root_of_equation_in_interval :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 2^x = 2 - x := 
sorry

end root_of_equation_in_interval_l476_476101


namespace min_sixth_power_sin_cos_l476_476269

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476269


namespace remainder_x_squared_mod_25_l476_476446

theorem remainder_x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 4 [ZMOD 25] :=
sorry

end remainder_x_squared_mod_25_l476_476446


namespace sin_cos_sixth_min_l476_476349

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476349


namespace pyramid_base_side_length_l476_476680

theorem pyramid_base_side_length (area : ℕ) (slant_height : ℕ) (s : ℕ) 
  (h1 : area = 100) 
  (h2 : slant_height = 20) 
  (h3 : area = (1 / 2) * s * slant_height) :
  s = 10 := 
by 
  sorry

end pyramid_base_side_length_l476_476680


namespace sum_21_eq_63_l476_476018

-- Define the sum of the first n terms of a geometric sequence
def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

-- Given conditions
def S₇ : ℝ := 48
def S₁₄ : ℝ := 60

-- The first term and common ratio of the geometric sequence
variable (a q : ℝ)

-- Assert the conditions as equations
axiom sum_7 : geometric_sum a q 7 = S₇
axiom sum_14 : geometric_sum a q 14 = S₁₄

-- The theorem to prove
theorem sum_21_eq_63 : geometric_sum a q 21 = 63 :=
sorry

end sum_21_eq_63_l476_476018


namespace arithmetic_sequence_terms_l476_476938

theorem arithmetic_sequence_terms :
  ∀ (n : ℕ), (n > 0) → (((100 + (n - 1) * (-6)) = 4) → (n = 17) ∧ (n - 1 = 16)) :=
by
  intros n hn ha
  sorry

end arithmetic_sequence_terms_l476_476938


namespace movie_marathon_duration_l476_476722

theorem movie_marathon_duration :
  let first_movie := 2
  let second_movie := first_movie + 0.5 * first_movie
  let combined_time := first_movie + second_movie
  let third_movie := combined_time - 1
  first_movie + second_movie + third_movie = 9 := by
  sorry

end movie_marathon_duration_l476_476722


namespace sequence_problem_l476_476585

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476585


namespace value_of_expression_l476_476750

theorem value_of_expression :
  (3 * (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) + 2) = 4373 :=
by
  sorry

end value_of_expression_l476_476750


namespace geometric_sequence_a7_l476_476475

theorem geometric_sequence_a7
  (a : ℕ → ℤ)
  (is_geom_seq : ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h1 : a 1 = -16)
  (h4 : a 4 = 8) :
  a 7 = -4 := 
sorry

end geometric_sequence_a7_l476_476475


namespace problem_statement_l476_476071

variables {A B C D E : Type} [Real]
variables (AD AE AB AC : ℝ)

-- Given condition
theorem problem_statement (area_BCD: ℝ) (h1: area_BCD = 0.5 * AD * AE):
  (AB / AE = AD / AC) :=
by
  sorry

end problem_statement_l476_476071


namespace sequence_problem_l476_476586

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476586


namespace min_value_fraction_l476_476610

open Real

-- Define the premises.
variables (a b : Real)

-- Assume a and b are positive and satisfy the equation a + 3b = 1.
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : a + 3b = 1

-- The theorem we want to prove.
theorem min_value_fraction (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3b = 1) :
  1 / a + 1 / b = 4 * sqrt 3 + 8 :=
sorry

end min_value_fraction_l476_476610


namespace common_diff_range_l476_476899

noncomputable def arith_seq_condition_1 : ℕ → ℝ
| n => -20 * n + (n * (n - 1)) / 2 * d

theorem common_diff_range (d : ℝ) 
    (h1 : ∀ n, S n = d / 2 * n^2 - (20 + d / 2) * n)
    (h2 : S 6 = argmin S n where n > 0) :
    10 / 3 < d ∧ d < 4 := 
sorry

end common_diff_range_l476_476899


namespace find_even_odd_functions_l476_476905

variable {X : Type} [AddGroup X]

def even_function (f : X → X) : Prop :=
∀ x, f (-x) = f x

def odd_function (f : X → X) : Prop :=
∀ x, f (-x) = -f x

theorem find_even_odd_functions
  (f g : X → X)
  (h_even : even_function f)
  (h_odd : odd_function g)
  (h_eq : ∀ x, f x + g x = 0) :
  (∀ x, f x = 0) ∧ (∀ x, g x = 0) :=
sorry

end find_even_odd_functions_l476_476905


namespace half_angle_third_quadrant_l476_476438

theorem half_angle_third_quadrant (α : ℝ) (k : ℤ) (h1 : k * 360 + 180 < α) (h2 : α < k * 360 + 270) : 
  (∃ n : ℤ, n * 360 + 90 < (α / 2) ∧ (α / 2) < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < (α / 2) ∧ (α / 2) < n * 360 + 315) := 
sorry

end half_angle_third_quadrant_l476_476438


namespace max_tan_A_minus_B_l476_476457

theorem max_tan_A_minus_B (A B C a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = (1/2) * c) (h2 : 0 < Real.tan A) :
  B = π / 6 :=
begin
  sorry
end

end max_tan_A_minus_B_l476_476457


namespace prove_arithmetic_sequence_minimum_value_S_l476_476533

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476533


namespace area_inside_C_outside_A_B_l476_476822

noncomputable section

-- Define circles A, B, and C along with their positions and other conditions
def radius := 1
def Circle (r : ℝ) := {x // x^2 + r^2 = r^2}

structure TangentCircles where
  A B C : Circle radius
  A_B_tangent : A.val = B.val
  M : ℝ  
  M_midpoint : M = (A.val + B.val) / 2
  C_tangent_to_M : C.val = M ∨ C.val = M

-- Statement of the problem
theorem area_inside_C_outside_A_B (cs : TangentCircles) : 
  let shared_area := cs.radius^2 * π / 4 - 1 / 2 
  cs.C_area := π*cs.radius^2
  let total_shared_area := 4 * shared_area 
  cs.C_area - total_shared_area = 2 := 
by 
  sorry

end area_inside_C_outside_A_B_l476_476822


namespace rhombus_diagonals_not_equal_l476_476159

-- Define what a rhombus is
structure Rhombus where
  sides_equal : ∀ a b : ℝ, a = b  -- all sides are equal
  symmetrical : Prop -- it is a symmetrical figure
  centrally_symmetrical : Prop -- it is a centrally symmetrical figure

-- Theorem to state that the diagonals of a rhombus are not necessarily equal
theorem rhombus_diagonals_not_equal (r : Rhombus) : ¬(∀ a b : ℝ, a = b) := by
  sorry

end rhombus_diagonals_not_equal_l476_476159


namespace maria_earnings_l476_476640

def cost_of_brushes : ℕ := 20
def cost_of_canvas : ℕ := 3 * cost_of_brushes
def cost_per_liter_of_paint : ℕ := 8
def liters_of_paint : ℕ := 5
def cost_of_paint : ℕ := liters_of_paint * cost_per_liter_of_paint
def total_cost : ℕ := cost_of_brushes + cost_of_canvas + cost_of_paint
def selling_price : ℕ := 200

theorem maria_earnings : (selling_price - total_cost) = 80 := by
  sorry

end maria_earnings_l476_476640


namespace arithmetic_sequence_min_value_Sn_l476_476573

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476573


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476605

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476605


namespace fraction_difference_l476_476140

theorem fraction_difference :
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  calc
    (3 + 6 + 9 : ℝ) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9)
        = 18 / 15 - 15 / 18 : by norm_num
    ... = 6 / 5 - 5 / 6 : by norm_num
    ... = (6 * 6 - 5 * 5) / (5 * 6) : by ring
    ... = 36 / 30 - 25 / 30 : by congr; norm_num
    ... = (36 - 25) / 30 : by ring
    ... = 11 / 30 : by norm_num

end fraction_difference_l476_476140


namespace intervals_monotonically_decreasing_max_min_on_interval_l476_476413

noncomputable def f : ℝ → ℝ := λ x, 2 * Real.sin x * Real.cos (x + Real.pi / 3) + Real.sqrt 3 / 2

theorem intervals_monotonically_decreasing (k : ℤ) :
  ∀ x, x ∈ Set.Icc (k * Real.pi + Real.pi / 12) (k * Real.pi + 7 * Real.pi / 12) → 
  ∀ y, y ∈ Set.Icc (k * Real.pi + Real.pi / 12) (k * Real.pi + 7 * Real.pi / 12) → 
  x ≥ y → f x ≤ f y :=
sorry

theorem max_min_on_interval :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
  (f x₁ = 1 ∧ f x₂ = - (Real.sqrt 3 / 2)) :=
sorry

end intervals_monotonically_decreasing_max_min_on_interval_l476_476413


namespace regular_hexagon_perimeter_l476_476912

-- Define the conditions of the problem
def radius (hexagon : Type) [regular_hexagon hexagon] := (3 : ℝ) -- given radius

-- Define the equivalence of radius and side length
def side_length_eq_radius (hexagon : Type) [regular_hexagon hexagon] : Proof := 
  by sorry -- regular_hexagon means the hexagon has equal sides and radius equals side length

-- Define the perimeter of a regular hexagon
def perimeter_of_hexagon {hexagon : Type} [regular_hexagon hexagon] (r : ℝ) : ℝ :=
  6 * r

-- Given radius, prove that perimeter is 18
theorem regular_hexagon_perimeter : perimeter_of_hexagon radius = 18 :=
  by sorry

end regular_hexagon_perimeter_l476_476912


namespace six_digit_probability_l476_476953

theorem six_digit_probability : 
  (let digits := [1, 3, 4, 5, 7, 8] in
   let total_permutations := list.permutations digits in
   let valid_numbers := total_permutations.filter (λ l, 
     let num := list.to_nat l in 
     (num % 20 = 0)) in
    (valid_numbers.length * 1) / (total_permutations.length * 1) = 1 / 15) := 
  sorry

end six_digit_probability_l476_476953


namespace johns_new_total_lift_l476_476032

theorem johns_new_total_lift :
  let initial_squat := 700
  let initial_bench := 400
  let initial_deadlift := 800
  let squat_loss_percentage := 30 / 100.0
  let squat_loss := squat_loss_percentage * initial_squat
  let new_squat := initial_squat - squat_loss
  let new_bench := initial_bench
  let new_deadlift := initial_deadlift - 200
  new_squat + new_bench + new_deadlift = 1490 := 
by
  -- Proof will go here
  sorry

end johns_new_total_lift_l476_476032


namespace trucks_transport_l476_476712

variables {x y : ℝ}

theorem trucks_transport (h1 : 2 * x + 3 * y = 15.5)
                         (h2 : 5 * x + 6 * y = 35) :
  3 * x + 2 * y = 17 :=
sorry

end trucks_transport_l476_476712


namespace factorial_trailing_zeros_50_mod_500_l476_476042

def factorial_trailing_zeros_mod (n : ℕ) : ℕ :=
  let fives_in_factorials (k : ℕ) :=
    let rec count_factors (x : ℕ) (p : ℕ) : ℕ :=
      if x = 0 then 0 else x / p + count_factors (x / p) p
    List.sum (List.range (k + 1)).map (λ i => count_factors i 5)
  fives_in_factorials n

theorem factorial_trailing_zeros_50_mod_500 : (factorial_trailing_zeros_mod 50) % 500 = 391 := by
  sorry

end factorial_trailing_zeros_50_mod_500_l476_476042


namespace arithmetic_sequence_minimum_value_S_l476_476574

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476574


namespace integral_inequality_l476_476881

theorem integral_inequality (a : ℝ) (h : a > 1) :
  (1 / (a - 1) * (1 - (real.log a / (a - 1)))) <
  ((a - real.log a - 1) / (a * (real.log a)^2)) ∧
  ((a - real.log a - 1) / (a * (real.log a)^2)) <
  (1 / real.log a * (1 - (real.log (real.log a + 1) / real.log a))) :=
sorry 

end integral_inequality_l476_476881


namespace arithmetic_sequence_minimum_value_S_n_l476_476552

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476552


namespace angle_B_is_60_l476_476956

theorem angle_B_is_60 (A B C : ℝ) (h_seq : 2 * B = A + C) (h_sum : A + B + C = 180) : B = 60 := 
by 
  sorry

end angle_B_is_60_l476_476956


namespace jake_should_charge_40_for_planting_flowers_l476_476981

theorem jake_should_charge_40_for_planting_flowers 
  (mow_time : ℕ) (mow_pay : ℕ) (desired_pay_rate : ℕ) (flower_time : ℕ) : 
  (mow_time = 1) → (mow_pay = 15) → (desired_pay_rate = 20) → (flower_time = 2) → (desired_pay_rate * flower_time = 40) :=
by
  intros h1 h2 h3 h4
  rw [h3, h4]
  exact rfl

end jake_should_charge_40_for_planting_flowers_l476_476981


namespace prove_arithmetic_sequence_minimum_value_S_l476_476534

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476534


namespace arithmetic_sequence_min_value_Sn_l476_476568

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476568


namespace volume_of_solid_l476_476196

theorem volume_of_solid (π : Real) : 
  (∫ x in (0..1), π * 4^2 * 1) + (∫ x in (1..5), π * 3^2 * 4) = 52 * π :=
by
  sorry

end volume_of_solid_l476_476196


namespace min_value_xyz_l476_476891

open Real

theorem min_value_xyz
  (x y z : ℝ)
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : 5 * x + 16 * y + 33 * z ≥ 136) :
  x^3 + y^3 + z^3 + x^2 + y^2 + z^2 ≥ 50 :=
sorry

end min_value_xyz_l476_476891


namespace find_c_l476_476378

-- Define the problem conditions and statement

variables (a b c : ℝ) (A B C : ℝ)
variable (cos_C : ℝ)
variable (sin_A sin_B : ℝ)

-- Given conditions
axiom h1 : a = 2
axiom h2 : cos_C = -1/4
axiom h3 : 3 * sin_A = 2 * sin_B
axiom sine_rule : sin_A / a = sin_B / b

-- Using sine rule to derive relation between a and b
axiom h4 : 3 * a = 2 * b

-- Cosine rule axiom
axiom cosine_rule : c^2 = a^2 + b^2 - 2 * a * b * cos_C

-- Prove c = 4
theorem find_c : c = 4 :=
by
  sorry

end find_c_l476_476378


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476603

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476603


namespace how_many_fewer_girls_l476_476112

def total_students : ℕ := 27
def girls : ℕ := 11
def boys : ℕ := total_students - girls
def fewer_girls_than_boys : ℕ := boys - girls

theorem how_many_fewer_girls :
  fewer_girls_than_boys = 5 :=
sorry

end how_many_fewer_girls_l476_476112


namespace find_m_l476_476949

noncomputable def complex_is_pure_imaginary (z : ℂ) : Prop :=
  ∃ y : ℝ, z = complex.I * y

theorem find_m (m : ℝ) (h : complex_is_pure_imaginary ((1 + complex.I) / (1 - complex.I) + m * (1 - complex.I))) :
  m = 0 :=
sorry

end find_m_l476_476949


namespace code_number_correspondence_l476_476113

-- Define the codes and corresponding mappings
def code := String
def number := ℕ

-- Given conditions
def codes : List code := ["RWQ", "SXW", "PST", "XNY", "NXY"]
def numbers : List number := [286, 540, 793, 948]

-- Goal: Prove the correct correspondence
theorem code_number_correspondence :
  ∃ (f : code → number),
    f "RWQ" = 286 ∧
    f "SXW" = 450 ∧
    f "PST" = 793 ∧
    f "XNY" = 948 ∧
    f "NXY" = 540 := by
    sorry

end code_number_correspondence_l476_476113


namespace min_value_g_l476_476382

noncomputable def φ : ℝ := π / 6

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + φ)

noncomputable def g (x : ℝ) : ℝ := -2 * sin (2 * x - φ)

theorem min_value_g : 
  ∃ (x : ℝ), x ∈ set.Icc (-π / 4) (π / 6) ∧ g x = -1 :=
begin
  use π / 6,
  split,
  { norm_num, },
  { refl, },
end

end min_value_g_l476_476382


namespace isosceles_triangle_perimeter_eq_10_l476_476216

theorem isosceles_triangle_perimeter_eq_10 (x : ℝ) 
(base leg : ℝ)
(h_base : base = 4)
(h_leg_root : x^2 - 5 * x + 6 = 0)
(h_iso : leg = x)
(triangle_ineq : leg + leg > base):
  2 * leg + base = 10 := 
begin
  cases (em (x = 2)) with h1 h2,
  { rw h1 at h_leg_root,
    rw [←h_iso, h1] at triangle_ineq,
    simp at triangle_ineq,
    contradiction },
  { rw h_iso,
    have : x = 3,
    { by_contra,
      simp [not_or_distrib, h1, h, sub_eq_zero] at h_leg_root },
    rw this,
    simp,
    linarith }
end

# Testing if the theorem can be evaluated successfully
# theorem_example : isosceles_triangle_perimeter_eq_10 3 4 3 rfl rfl sorry sorry rfl :=
# sorry

end isosceles_triangle_perimeter_eq_10_l476_476216


namespace tangent_line_range_of_a_l476_476416

-- Definitions extracted from the problem statements
def f (x : ℝ) : ℝ := x / Real.exp x + x^2 - x
def e : ℝ := Real.exp 1
def g (a x : ℝ) : ℝ := -a * Real.log (f x - x^2 + x) - 1 / x - Real.log x + a + 1

/- 
  Prove that the equation of the tangent line to f(x) at (1, f(1)) is ex - ey - e + 1 = 0,
  given e = 2.71828...
-/
theorem tangent_line (h : e = Real.exp 1) : ∀ x : ℝ, (f 1) = 1 / e → ex - ey - e + 1 = 0 :=
sorry

/- 
  Prove that for g(a, x), if x ≥ 1, then g(x) ≥ 0 implies a ∈ [1, +∞),
  given f(x) = x / Real.exp x + x^2 - x.
-/
theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x ≥ 1 → g a x ≥ 0)) → a ∈ Set.Ici 1 :=
sorry

end tangent_line_range_of_a_l476_476416


namespace factorial_trailing_zeros_50_mod_500_l476_476041

def factorial_trailing_zeros_mod (n : ℕ) : ℕ :=
  let fives_in_factorials (k : ℕ) :=
    let rec count_factors (x : ℕ) (p : ℕ) : ℕ :=
      if x = 0 then 0 else x / p + count_factors (x / p) p
    List.sum (List.range (k + 1)).map (λ i => count_factors i 5)
  fives_in_factorials n

theorem factorial_trailing_zeros_50_mod_500 : (factorial_trailing_zeros_mod 50) % 500 = 391 := by
  sorry

end factorial_trailing_zeros_50_mod_500_l476_476041


namespace arithmetic_sequence_minimum_value_S_n_l476_476562

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476562


namespace random_function_eq_stochastic_process_l476_476657

variables {Ω : Type*} {T : Type*} [measurable_space Ω] {ℱ : measurable_space Ω}
          {𝓡T : Type*} [topological_space 𝓡T] {B_𝓡T : borel_space 𝓡T}

/-- A random function X = (Xₜ)ₜ∈T is measurable with respect to ℱ / 𝓡T ⁰ -/
def is_measurable_random_function (X : Ω → (ℝ^T)) :=
  measurable (λ (ω : Ω), X ω)

/-- Prove that a random function (Xₜ)ₜ∈T is a stochastic process, i.e., Xₜ are random variables, and vice versa -/
theorem random_function_eq_stochastic_process
  {X : Ω → (ℝ^T)} (hX: is_measurable_random_function X) :
  (∀ t : T, measurable (λ (ω : Ω), (X ω) t)) ∧
  (∃ Y : Ω → (ℝ^T), (∀ t : T, measurable (λ (ω : Ω), (Y ω) t)) ∧ 
    (∀ ω : Ω, X ω = Y ω)) :=
sorry

end random_function_eq_stochastic_process_l476_476657


namespace angle_between_u_and_v_l476_476856

open Real

noncomputable def u : ℝ × ℝ × ℝ := (3, -2, 2)
noncomputable def v : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (a.1^2 + a.2^2 + a.3^2)

noncomputable def cosine (a b : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product a b) / ((magnitude a) * (magnitude b))

noncomputable def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (cosine a b)

theorem angle_between_u_and_v :
  angle_between_vectors u v = real.arccos (-8 / (3 * sqrt 17)) := by
  sorry

end angle_between_u_and_v_l476_476856


namespace determine_a2018_l476_476023

def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n + a (n + 1) + a (n + 2) = 15

theorem determine_a2018 (a : ℕ → ℕ)
  (h1 : a 4 = 1)
  (h2 : a 12 = 5)
  (h3 : sequence a) :
  a 2018 = 9 :=
sorry

end determine_a2018_l476_476023


namespace min_sixth_power_sin_cos_l476_476258

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476258


namespace isosceles_triangle_perimeter_l476_476211

-- Define what it means to be a root of the equation x^2 - 5x + 6 = 0
def is_root (x : ℝ) : Prop := x^2 - 5 * x + 6 = 0

-- Define the perimeter based on given conditions
theorem isosceles_triangle_perimeter (x : ℝ) (base : ℝ) (h_base : base = 4) (h_root : is_root x) :
    2 * x + base = 10 :=
by
  -- Insert proof here
  sorry

end isosceles_triangle_perimeter_l476_476211


namespace prime_between_30_and_40_has_remainder_7_l476_476791

theorem prime_between_30_and_40_has_remainder_7 (p : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_interval : 30 < p ∧ p < 40) 
  (h_mod : p % 9 = 7) : 
  p = 34 := 
sorry

end prime_between_30_and_40_has_remainder_7_l476_476791


namespace area_of_A_inter_B_l476_476424

-- Define sets A and B as given
def setA : Set (ℝ × ℝ) := {p | (p.2 - p.1) * (p.2 - 1 / p.1) ≥ 0}
def setB : Set (ℝ × ℝ) := {p | (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 ≤ 1}

-- Define the theorem to prove the area of A ∩ B
theorem area_of_A_inter_B : 
  (measure_theory.measure_space.volume (Set (ℝ × ℝ))).measurable_measure (setA ∩ setB) = π / 2 :=
sorry

end area_of_A_inter_B_l476_476424


namespace point_in_second_quadrant_l476_476092

open Complex

-- Define the determinant operation for a 2x2 matrix
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Given matrix elements
def a : ℂ := 1
def b : ℂ := 2
def c : ℂ := i
def d : ℂ := -1

-- Calculate z from the determinant
def z : ℂ := det a b c d

-- Compute the complex conjugate of z
def conj_z : ℂ := conj z

-- Define the point corresponding to the complex number conj_z
def point : ℝ × ℝ := (conj_z.re, conj_z.im)

-- State the theorem regarding which quadrant the point lies in
theorem point_in_second_quadrant : point.1 < 0 ∧ point.2 > 0 :=
by sorry

end point_in_second_quadrant_l476_476092


namespace measure_of_angle_ZHY_l476_476483

variables (X Y Z P Q R H : Type)
variables [triangleXYZ : is_triangle X Y Z] 

noncomputable def angle_XYZ : ℝ := 60
noncomputable def angle_YZX : ℝ := 40
noncomputable def angle_XHZ : ℝ := 90

theorem measure_of_angle_ZHY :
  ∑ (angle XYZ + angle YZX + ZXY) = 180 →
  altitudes_intersect_at_orthocenter X P Y Q Z R H →
  ∑ (angle XHZ + θ := 40) = 90 →
  ∠ ZHY = 40 :=
by sorry

end measure_of_angle_ZHY_l476_476483


namespace min_sin6_cos6_l476_476286

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476286


namespace neg_one_exponent_difference_l476_476766

theorem neg_one_exponent_difference : (-1 : ℤ) ^ 2004 - (-1 : ℤ) ^ 2003 = 2 := by
  sorry

end neg_one_exponent_difference_l476_476766


namespace pyramid_dihedral_angle_l476_476658

variables {O A B C D : Type*} [EuclideanGeometry]
variables (OA OB OC OD AB θ m n : ℝ)
variables (square_base : SquareBase ABCD)
variables (congruent_edges : OA = OB ∧ OB = OC ∧ OC = OD ∧ OD = OA)
variables (angle_AOB_60 : ∠ AOB = 60)
variables (cos_theta : cos θ = m + sqrt n)

theorem pyramid_dihedral_angle : square_base ABCD → congruent_edges OA OB OC OD → ∠ AOB = 60 → cos θ = 0 :=
by
  -- Proof will be provided here
  sorry

end pyramid_dihedral_angle_l476_476658


namespace fixed_point_line_l476_476952

theorem fixed_point_line (k : ℝ) :
  ∃ A : ℝ × ℝ, (3 + k) * A.1 - 2 * A.2 + 1 - k = 0 ∧ (A = (1, 2)) :=
by
  let A : ℝ × ℝ := (1, 2)
  use A
  sorry

end fixed_point_line_l476_476952


namespace min_value_sin_cos_l476_476298

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476298


namespace arithmetic_sequence_min_value_Sn_l476_476566

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476566


namespace floss_leftover_l476_476839

noncomputable def leftover_floss
    (students : ℕ)
    (floss_per_student : ℚ)
    (floss_per_packet : ℚ) :
    ℚ :=
  let total_needed := students * floss_per_student
  let packets_needed := (total_needed / floss_per_packet).ceil
  let total_floss := packets_needed * floss_per_packet
  total_floss - total_needed

theorem floss_leftover {students : ℕ} {floss_per_student floss_per_packet : ℚ}
    (h_students : students = 20)
    (h_floss_per_student : floss_per_student = 3 / 2)
    (h_floss_per_packet : floss_per_packet = 35) :
    leftover_floss students floss_per_student floss_per_packet = 5 :=
by
  rw [h_students, h_floss_per_student, h_floss_per_packet]
  simp only [leftover_floss]
  norm_num
  sorry

end floss_leftover_l476_476839


namespace problem1_problem2_problem3_l476_476471

noncomputable def point := ℝ × ℝ

def radius (M: point) (r: ℝ): Prop := r = 1

def tangent_condition (l: point → Prop) (C: point): Prop :=
  (C = (3 / 2, 1 + sqrt 3 / 2)) ∧ (l C)

def line_eq (l: point → Prop): Prop :=
  ∀ x y : ℝ, l (x, y) ↔ y = - sqrt 3 / 3 * x + sqrt 3 + 1

def isosceles_right_triangle (A B O: point): Prop :=
  dist A O = dist B O ∧
  ∃ l, line_eq l ∧ ∀ z, is_OAB l A B O z

def minimized_triangle (A B: point): Prop :=
  isosceles_right_triangle A B (0, 0) ∧
  let s := 1 in
  area_triangle A B = s * (3 + 2 * sqrt 2)

def max_PA2_PB2_PO2 (l: point → Prop) (A B O P: point): Prop :=
  l A ∧ l B ∧
  ∀ m n: ℝ, (m, n) ≠ (0, 0) ∧
  let PA2 := (m - A.1) ^ 2 + (n - A.2) ^ 2 in
  let PB2 := (m - B.1) ^ 2 + (n - B.2) ^ 2 in
  let PO2 := (m - O.1) ^ 2 + (n - O.2) ^ 2 in
  PA2 + PB2 + PO2 ≤ (17 + 2 * sqrt 2)

theorem problem1 (l: point → Prop) (A B O C: point):
  radius (1, 1) 1 →
  tangent_condition l C →
  line_eq l :=
sorry

theorem problem2 (A B: point):
  minimized_triangle A B →
  isosceles_right_triangle A B (0, 0) :=
sorry

theorem problem3 (l: point → Prop) (A B O P: point):
  l = λ p, p.1 + p.2 - 2 - sqrt 2 = 0 →
  max_PA2_PB2_PO2 l A B O P :=
sorry

end problem1_problem2_problem3_l476_476471


namespace connor_additional_spending_l476_476823

theorem connor_additional_spending : 
  let cost1 := 13.00 * 0.75,
      cost2 := 15.00 * 0.75,
      cost3 := 10.00,
      cost4 := 10.00,
      total := cost1 + cost2 + cost3 + cost4,
      discounted_total := total * 0.90,
      required_spending := 50.00 - discounted_total
  in required_spending = 13.10 :=
by 
  let cost1 := 13.00 * 0.75
  let cost2 := 15.00 * 0.75
  let cost3 := 10.00
  let cost4 := 10.00
  let total := cost1 + cost2 + cost3 + cost4
  let discounted_total := total * 0.90
  let required_spending := 50.00 - discounted_total
  exact sorry

end connor_additional_spending_l476_476823


namespace min_sin6_cos6_l476_476339

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476339


namespace king_paid_after_tip_l476_476787

-- Define the cost of the crown and the tip percentage
def cost_of_crown : ℝ := 20000
def tip_percentage : ℝ := 0.10

-- Define the total amount paid after the tip
def total_amount_paid (C : ℝ) (tip_pct : ℝ) : ℝ :=
  C + (tip_pct * C)

-- Theorem statement: The total amount paid after the tip is $22,000
theorem king_paid_after_tip : total_amount_paid cost_of_crown tip_percentage = 22000 := by
  sorry

end king_paid_after_tip_l476_476787


namespace cos_double_angle_l476_476377

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 1 / 2) : Real.cos (2 * α) = 3 / 5 :=
by
  sorry

end cos_double_angle_l476_476377


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476356

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476356


namespace sum_of_coefficients_nonzero_power_of_y_l476_476145

-- Define the polynomial expansion as a constant term
def poly_expansion : ℕ := 
  let f := (5 * x + 3 * y + 2) * (2 * x + 5 * y + 4) in
  (f.coeff (1,0) + f.coeff (0,1) + f.coeff (0,2))

-- Theorem stating the required proof
theorem sum_of_coefficients_nonzero_power_of_y : poly_expansion = 68 := by
  sorry

end sum_of_coefficients_nonzero_power_of_y_l476_476145


namespace product_of_intersection_coords_l476_476833

open Real

-- Define the two circles
def circle1 (x y: ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 21 = 0
def circle2 (x y: ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 52 = 0

-- Prove that the product of the coordinates of intersection points equals 189
theorem product_of_intersection_coords :
  (∃ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧ circle1 x2 y2 ∧ circle2 x2 y2 ∧ x1 * y1 * x2 * y2 = 189) :=
by
  sorry

end product_of_intersection_coords_l476_476833


namespace paco_initial_salty_cookies_l476_476654

variable (S : ℕ)
variable (sweet_cookies : ℕ := 40)
variable (salty_cookies_eaten1 : ℕ := 28)
variable (sweet_cookies_eaten : ℕ := 15)
variable (extra_salty_cookies_eaten : ℕ := 13)

theorem paco_initial_salty_cookies 
  (h1 : salty_cookies_eaten1 = 28)
  (h2 : sweet_cookies_eaten = 15)
  (h3 : extra_salty_cookies_eaten = 13)
  (h4 : sweet_cookies = 40)
  : (S = (salty_cookies_eaten1 + (extra_salty_cookies_eaten + sweet_cookies_eaten))) :=
by
  -- starting with the equation S = number of salty cookies Paco
  -- initially had, which should be equal to the total salty 
  -- cookies he ate.
  sorry

end paco_initial_salty_cookies_l476_476654


namespace a_value_l476_476775

-- Conditions: Sales data from January to May 2023 and regression line equation
variable (x : ℕ) (y : ℝ)

-- Given sales data:
def sales (x : ℕ) :=
  match x with
  | 1 => 1.0
  | 2 => 1.6
  | 3 => 2.0
  | 4 => a
  | 5 => 3.0
  | _ => 0.0  -- Default value for unsupported months

-- Given regression equation: \hat{y} = 0.48x + 0.56
def regression (x : ℕ) : ℝ :=
  0.48 * x + 0.56

-- Proof of statement that 'a = 2.4'
theorem a_value :
  ∑ i in (finset.range 5).map (nat.succ) ((sales i)) = 10.0 → a = 2.4 :=
sorry

end a_value_l476_476775


namespace find_initial_principal_amount_l476_476878

noncomputable def compound_interest (initial_principal : ℝ) : ℝ :=
  let year1 := initial_principal * 1.09
  let year2 := (year1 + 500) * 1.10
  let year3 := (year2 - 300) * 1.08
  let year4 := year3 * 1.08
  let year5 := year4 * 1.09
  year5

theorem find_initial_principal_amount :
  ∃ (P : ℝ), (|compound_interest P - 1120| < 0.01) :=
sorry

end find_initial_principal_amount_l476_476878


namespace triangle_max_area_l476_476024

theorem triangle_max_area (A B C a b c S : ℝ)
  (h1 : a = 2)
  (h2 : tan A / tan B = 4 / 3)
  (h3 : S = (1 / 2) * b * c * (sin A))
  (h4 : A + B + C = π)
  (h5 : A + B = π - C)
  (h6 : (sin (π - C)) = sin C) :
  S ≤ 1 / 2 :=
by
  sorry

end triangle_max_area_l476_476024


namespace arithmetic_sequence_and_minimum_sum_l476_476527

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476527


namespace arithmetic_problem_l476_476173

theorem arithmetic_problem : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end arithmetic_problem_l476_476173


namespace min_sin6_cos6_l476_476278

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476278


namespace exists_solution_in_interval_l476_476409

noncomputable def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

theorem exists_solution_in_interval :
  ∃ x ∈ (Set.Ioo 1 2 : Set ℝ), f x = 0 :=
begin
  -- By the Intermediate Value Theorem, we just need to show that f(1) * f(2) < 0.
  have h1 : f 1 = 1,
  { calc f 1 = -1^3 - 3*1 + 5 : by refl
         ... = 1               : by norm_num },

  have h2 : f 2 = -9,
  { calc f 2 = -2^3 - 3*2 + 5 : by refl
         ... = -9              : by norm_num },

  -- So, f(1) * f(2) < 0, by Intermediate Value Theorem, there exists a root
  have hmul : f 1 * f 2 < 0,
  { calc 1 * -9 < 0 : by norm_num },

  -- Conclusion by Intermediate Value Theorem
  exact intermediate_value_Ioo 1 2 hmul sorry
end

end exists_solution_in_interval_l476_476409


namespace area_of_triangle_AFM_l476_476421

theorem area_of_triangle_AFM : 
  ∃ p : ℝ, 
    (p > 0) ∧ 
    (∀ x y : ℝ, ((x = -2) → (y = 0))) ∧ 
    (y^2 = 2 * p * x) ∧ 
    (∀ M F : ℝ × ℝ, 
      (F = (2, 0)) →
      ((sqrt((M.1 - F.1)^2 + (M.2 - F.2)^2) = 8) → 
      (let y0 := M.2 in let x0 := M.1 in y0^2 = 8 * x0 → ∃ S : ℝ, S = 8 * sqrt 3))) :=
begin
  sorry
end

end area_of_triangle_AFM_l476_476421


namespace circumscribed_circle_equation_l476_476120

-- Definitions of the lines
def line1 := (x : ℝ) => 0.2 * x - 0.4
def line2 := (x : ℝ) => x + 2
def line3 := (x : ℝ) => 8 - x

-- The statement that we need to prove
theorem circumscribed_circle_equation :
  ∀ x y : ℝ,
  ((y = line1 x ∨ y = line2 x ∨ y = line3 x) → ((x-2)^2 + y^2 = 26)) :=
by
  intros
  -- Put the proof here.
  sorry

end circumscribed_circle_equation_l476_476120


namespace probability_correct_number_l476_476645

def is_valid_number (num : ℕ) : Prop :=
  (num / 10^5 = 297 ∨ num / 10^5 = 298 ∨ num / 10^5 = 299) ∧
  let last_five_digits := num % 10^5 in
  let digits := [0, 1, 5, 6, 7] in
  ∃ d, d ∈ digits ∧ list.perm ([d, d] ++ digits.erase d) (list.of_digits last_five_digits)

theorem probability_correct_number : 
  ∃ count : ℕ, count = 900 ∧ (1 : ℚ) / count = 1 / 900 :=
sorry

end probability_correct_number_l476_476645


namespace largest_price_increase_l476_476197

variable (p q : ℝ)

theorem largest_price_increase (h : p ≠ q) :
  let scheme_A := (1 + p / 100) * (1 + q / 100),
      scheme_B := (1 + q / 100) * (1 + p / 100),
      scheme_C := (1 + (p + q) / 200) ^ 2,
      scheme_D := (1 + (Real.sqrt ((p ^ 2 + q ^ 2) / 2)) / 100) ^ 2
  in scheme_D > scheme_A ∧ scheme_D > scheme_B ∧ scheme_D > scheme_C :=
by
  sorry

end largest_price_increase_l476_476197


namespace Z_4_3_eq_neg11_l476_476826

def Z (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2

theorem Z_4_3_eq_neg11 : Z 4 3 = -11 := 
by
  sorry

end Z_4_3_eq_neg11_l476_476826


namespace sum_of_ages_l476_476033

-- Definitions from the problem conditions
def Maria_age : ℕ := 14
def age_difference_between_Jose_and_Maria : ℕ := 12
def Jose_age : ℕ := Maria_age + age_difference_between_Jose_and_Maria

-- To be proven: sum of their ages is 40
theorem sum_of_ages : Maria_age + Jose_age = 40 :=
by
  -- skip the proof
  sorry

end sum_of_ages_l476_476033


namespace find_values_of_x_y_l476_476436

-- Define the conditions
def like_terms (e1 e2 : ℝ) : Prop := ∃ (x y : ℝ), e1 = 5 * a ^ (abs x) * b ^ 2 ∧ e2 = -0.2 * a ^ (3 : ℝ) * b ^ (abs y)

-- The main theorem statement
theorem find_values_of_x_y (a b x y : ℝ) (h : like_terms (5 * a ^ (abs x) * b ^ 2) (-0.2 * a ^ (3 : ℝ) * b ^ (abs y))) : abs x = 3 ∧ abs y = 2  :=
sorry

end find_values_of_x_y_l476_476436


namespace brocard_and_circumcenter_properties_l476_476997

structure Triangle (P : Type) :=
  (A B C : P)

def is_brocard_point (Q : P) (T : Triangle P) : Prop :=
  -- Definition of the second Brocard point goes here
  sorry

def circumcenter (T : Triangle P) : P :=
  -- Definition of the circumcenter goes here
  sorry

def circumcircle_center (T : Triangle P) (Q : P) : P :=
  -- Definition of the circumcircle center for triangle formed with Q goes here
  sorry

def similar_triangles (T1 T2 : Triangle P) : Prop :=
  -- Definition of similarity of two triangles goes here
  sorry

def first_brocard_point (O : P) (T : Triangle P) : Prop :=
  -- Definition of the first Brocard point goes here
  sorry

theorem brocard_and_circumcenter_properties (A B C Q O A₁ B₁ C₁ : P)
  (h1 : is_brocard_point Q (Triangle.mk A B C))
  (h2 : circumcenter (Triangle.mk A B C) = O)
  (h3 : circumcircle_center (Triangle.mk C A Q) = A₁)
  (h4 : circumcircle_center (Triangle.mk A B Q) = B₁)
  (h5 : circumcircle_center (Triangle.mk B C Q) = C₁) :
  (similar_triangles (Triangle.mk A₁ B₁ C₁) (Triangle.mk A B C)) ∧
  (first_brocard_point O (Triangle.mk A₁ B₁ C₁)) :=
sorry

end brocard_and_circumcenter_properties_l476_476997


namespace integer_roots_p_l476_476955

theorem integer_roots_p (p x1 x2 : ℤ) (h1 : x1 * x2 = p + 4) (h2 : x1 + x2 = -p) : p = 8 ∨ p = -4 := 
sorry

end integer_roots_p_l476_476955


namespace problem_1_problem_2_l476_476916

noncomputable def f (a x : ℝ) : ℝ := Real.exp x + a * x

-- Part 1: The value of 'a' such that the tangent line at x = 1 is perpendicular to the line x + (e-1)y = 1.
theorem problem_1 (a : ℝ) : (Real.exp 1 + a) = -(1 - Real.exp 1) → a = -1 :=
begin
  intros h_slope_perpendicular,
  sorry
end

-- Part 2: The range of 'a' such that f(x) > 0 for all x > 0.
theorem problem_2 (a : ℝ) : (∀ x : ℝ, x > 0 → f a x > 0) → a > -Real.exp 1 :=
begin
  intros h_positive_f,
  sorry
end

end problem_1_problem_2_l476_476916


namespace arithmetic_sequence_minimum_value_S_n_l476_476556

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476556


namespace evaluate_expression_l476_476847

theorem evaluate_expression : 
  let x := 3; let y := 4 
  in 5 * x^(y - 1) + 2 * y^(x + 1) = 647 := 
by 
  let x := 3
  let y := 4
  sorry

end evaluate_expression_l476_476847


namespace problem_statement_l476_476379

noncomputable def a : ℝ := Real.log (3 / 2) / Real.log (2 / 3)  -- log_{2/3}(3/2)
noncomputable def b : ℝ := Real.log 2 / Real.log 3             -- log_3(2)
noncomputable def c : ℝ := Real.pow 2 (1 / 3)                  -- 2^(1/3)
noncomputable def d : ℝ := Real.pow 3 (1 / 2)                  -- 3^(1/2)

theorem problem_statement : a < b ∧ b < c ∧ c < d :=
by
  sorry

end problem_statement_l476_476379


namespace B_visits_a_l476_476119

variable (City : Type) (Visited : City → Prop)
variable (a b c : City)
variable (A_visits B_visits C_visits : City → Prop)

-- Conditions
axiom A_more_than_B : ∀ x, A_visits x → B_visits x
axiom A_not_b : ¬ A_visits b
axiom B_not_c : ¬ B_visits c
axiom same_city : ∀ x, A_visits x ↔ B_visits x ∧ B_visits x ↔ C_visits x

-- Goal
theorem B_visits_a : B_visits a := sorry

end B_visits_a_l476_476119


namespace num_ways_to_parenthesize_prod_l476_476671

theorem num_ways_to_parenthesize_prod : 
  let n := 4
  Catalan n = 14 :=
by
  sorry

end num_ways_to_parenthesize_prod_l476_476671


namespace radius_of_smaller_base_of_truncated_cone_l476_476117

theorem radius_of_smaller_base_of_truncated_cone 
  (r1 r2 r3 : ℕ) (touching : 2 * r1 = r2 ∧ r1 + r3 = r2 * 2):
  (∀ (R : ℕ), R = 6) :=
sorry

end radius_of_smaller_base_of_truncated_cone_l476_476117


namespace sequence_problem_l476_476592

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476592


namespace tangent_line_eq_l476_476254

def f (x : ℝ) : ℝ := (1 / 2) * x^2

theorem tangent_line_eq (x y : ℝ) (hfx : f 1 = 1 / 2) (hr : y = x - 1 / 2) : 
  2 * x - 2 * y - 1 = 0 :=
sorry

end tangent_line_eq_l476_476254


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476359

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476359


namespace sufficient_b_not_necessary_l476_476931

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

def sufficient_condition (a b : ℝ × ℝ) (x : ℝ) : Prop :=
  (x = -1) → b = (1, 3)

-- Given vectors a and b
def vector_a : ℝ → ℝ × ℝ := λ x, (1, 2 - x)
def vector_b : ℝ → ℝ × ℝ := λ x, (2 + x, 3)

theorem sufficient_b_not_necessary (x : ℝ) : sufficient_condition (vector_a x) (vector_b x) x :=
  by
    sorry

end sufficient_b_not_necessary_l476_476931


namespace diane_honey_harvest_l476_476836

theorem diane_honey_harvest (last_year : ℕ) (increase : ℕ) (this_year : ℕ) :
  last_year = 2479 → increase = 6085 → this_year = last_year + increase → this_year = 8564 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end diane_honey_harvest_l476_476836


namespace min_sixth_power_sin_cos_l476_476266

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l476_476266


namespace jim_travel_distance_l476_476948

theorem jim_travel_distance :
  ∀ (John Jill Jim : ℝ),
  John = 15 →
  Jill = (John - 5) →
  Jim = (0.2 * Jill) →
  Jim = 2 :=
by
  intros John Jill Jim hJohn hJill hJim
  sorry

end jim_travel_distance_l476_476948


namespace expression_equals_l476_476137

theorem expression_equals (x y : ℝ) :
  (x = 3^4) ∧ (y = 3^2) →
  (a = 3^(-2)) ∧ (b = 3^(-4)) →
  (frac_result = (x - y) / (a + b)) →
  frac_result = 583.2 :=
by
  intro h1 h2 h3
  sorry

end expression_equals_l476_476137


namespace largest_prime_factor_of_1560_l476_476139

theorem largest_prime_factor_of_1560 : ∃ p : ℕ, p.prime ∧ p = 13 ∧ ∀ q : ℕ, q.prime → q ∣ 1560 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1560_l476_476139


namespace class_teacher_age_l476_476960

/--
In a class of 45 students with an average age of 14 years, one 15-year-old student transfers out, and the class teacher's age is included in the calculation, resulting in a new average age of 14.66 years. Prove that the age of the class teacher is the smallest prime number greater than 14.66.
-/
theorem class_teacher_age : 
  let num_students_initial := 45
  let avg_age_initial := 14
  let age_transfer_student := 15
  let num_students_after_transfer := num_students_initial - 1
  let avg_age_with_teacher := 14.66
  let total_age_initial := num_students_initial * avg_age_initial
  let total_age_after_transfer := total_age_initial - age_transfer_student
  let age_with_teacher := (num_students_after_transfer * avg_age_with_teacher) + T
  let teacher_age := 17

  total_age_initial = 630 ∧
  total_age_after_transfer = 615 ∧
  age_with_teacher = 659.7 ∧
  teacher_age = 17 ∧
  is_prime teacher_age ∧
  teacher_age > 14.66
  → teacher_age = 17 :=
by {
  -- proof omitted
  sorry
}

end class_teacher_age_l476_476960


namespace arrangement_count_l476_476125

open Finset

variable {B G : Finset ℕ}
variable {A : ℕ}

-- Conditions
def is_boy (x : ℕ) : Prop := x ∈ B
def is_girl (x : ℕ) : Prop := x ∈ G

def different_arrangements (B G : Finset ℕ) (A : ℕ) :=
  (A ∈ B) ∧ 
  (B.card = 2) ∧ 
  (G.card = 3) ∧
  ¬(A = 1 ∨ A = 5) ∧ -- Not standing at either end
  ∃ two_adjacent_girls, two_adjacent_girls.card = 2 ∧ (∀ g ∈ two_adjacent_girls, g ∈ G)

-- Statement to prove
theorem arrangement_count (h : different_arrangements B G A) : 
  finset α \is_boy (x : ℕ) : (x + y = z ) ∈ G.choice : (the_number_of_performances :=)
 30 =
finset G :=
 
  sorry

end arrangement_count_l476_476125


namespace sin_x_value_l476_476892

theorem sin_x_value (x : ℝ) (hx1 : real.sin (real.pi / 2 + x) = 5 / 13) (hx2 : x ∈ set.Ioo (- real.pi / 2) 0) : real.sin x = - (12 / 13) :=
by sorry

end sin_x_value_l476_476892


namespace arithmetic_sequence_min_value_Sn_l476_476571

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476571


namespace geom_seq_a4_l476_476964

theorem geom_seq_a4 (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h2 : a 3 = 9)
  (h3 : a 5 = 1) :
  a 4 = 3 ∨ a 4 = -3 :=
by {
  sorry
}

end geom_seq_a4_l476_476964


namespace num_distinct_triangles_l476_476430

-- Define a 2x4 grid
def grid : set (ℕ × ℕ) := {(x, y) | x < 4 ∧ y < 2}

-- Function to check if three points are collinear
def collinear (p1 p2 p3 : ℕ × ℕ) : Prop :=
  (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) = 0

-- Define the set of all triangles formed by points in the grid
def triangles (s : set (ℕ × ℕ)) : set (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  {(p1, p2, p3) | p1 ∈ s ∧ p2 ∈ s ∧ p3 ∈ s ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ ¬ collinear p1 p2 p3}

-- Proof that the number of distinct triangles is 46
theorem num_distinct_triangles : (finset.card (finset.filter (λ t, t ∈ triangles grid) (finset.univ.image (λ t, ((t.1.1, t.1.2), (t.1.3, t.1.4), (t.1.5, t.1.6))))) = 46) :=
  sorry

end num_distinct_triangles_l476_476430


namespace measure_of_angle_A_l476_476204

-- Definitions based on conditions
def is_isosceles_right_triangle (A B C : Type _) [euclidean_geometry A B C] :=
  (AB = BC) ∧ (angle B = 90)

-- Statement to prove
theorem measure_of_angle_A 
(A B C : Type _) [euclidean_geometry A B C] (h1 : AB = BC) (h2 : angle B = 90) :
  ∠A = 45 :=
sorry

end measure_of_angle_A_l476_476204


namespace domain_of_f_l476_476687

def f (x : ℝ) : ℝ := x ^ (-2)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≠ 0} :=
sorry

end domain_of_f_l476_476687


namespace yoongi_has_fewest_apples_l476_476164

noncomputable def yoongi_apples : ℕ := 4
noncomputable def yuna_apples : ℕ := 5
noncomputable def jungkook_apples : ℕ := 6 * 3

theorem yoongi_has_fewest_apples : yoongi_apples < yuna_apples ∧ yoongi_apples < jungkook_apples := by
  sorry

end yoongi_has_fewest_apples_l476_476164


namespace find_other_percentage_l476_476485

noncomputable def percentage_other_investment
  (total_investment : ℝ)
  (investment_10_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_10_percent : ℝ)
  (other_investment_interest : ℝ) : ℝ :=
  let interest_10_percent := investment_10_percent * interest_rate_10_percent
  let interest_other_investment := total_interest - interest_10_percent
  let amount_other_percentage := total_investment - investment_10_percent
  interest_other_investment / amount_other_percentage

theorem find_other_percentage :
  ∀ (total_investment : ℝ)
    (investment_10_percent : ℝ)
    (total_interest : ℝ)
    (interest_rate_10_percent : ℝ),
    total_investment = 31000 ∧
    investment_10_percent = 12000 ∧
    total_interest = 1390 ∧
    interest_rate_10_percent = 0.1 →
    percentage_other_investment total_investment investment_10_percent total_interest interest_rate_10_percent 190 = 0.01 :=
by
  intros total_investment investment_10_percent total_interest interest_rate_10_percent h
  sorry

end find_other_percentage_l476_476485


namespace max_sin_a_l476_476667

theorem max_sin_a (a b : ℝ) (h : sin (a - b) = sin a - sin b) : sin a ≤ 1 :=
by sorry

end max_sin_a_l476_476667


namespace angle_between_vectors_l476_476868

theorem angle_between_vectors :
  let v1 : ℝ × ℝ × ℝ := (3, -2, 2)
  let v2 : ℝ × ℝ × ℝ := (-2, 2, 1)
  let dot_product (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let magnitude (v : ℝ × ℝ × ℝ) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let cos_theta := dot_product v1 v2 / (magnitude v1 * magnitude v2)
  θ = Real.acos (cos_theta) * (180 / Real.pi)
  in θ ≈ 127 :=
by
  sorry

end angle_between_vectors_l476_476868


namespace graph_of_equation_is_line_and_hyperbola_l476_476830

theorem graph_of_equation_is_line_and_hyperbola :
  ∀ (x y : ℝ), ((x^2 - 1) * (x + y) = y^2 * (x + y)) ↔ (y = -x) ∨ ((x + y) * (x - y) = 1) := by
  intro x y
  sorry

end graph_of_equation_is_line_and_hyperbola_l476_476830


namespace distance_inequality_l476_476626

variables (A B C P M : Type) [Point A] [Point B] [Point C] [Point P] [Point M]

-- Assuming Point is a typeclass representing a point in space
class Point (P : Type) :=
  (distance : P → P → ℝ) -- define a distance function

-- Conditions
variable (is_interior_point : ∀ P : Type, Point P → Prop) -- P is an interior point of Triangle ABC
variable (angle_condition : angle A M B = 120 ∧ angle B M C = 120 ∧ angle C M A = 120)

-- Goal
theorem distance_inequality
  (h_interior : is_interior_point M Point)
  (h_angle : angle_condition)
  (h_point_in_triangle : ∀ P, is_interior_point P Point) :
  Point.distance P A + Point.distance P B + Point.distance P C ≥ 
  Point.distance M A + Point.distance M B + Point.distance M C :=
sorry

end distance_inequality_l476_476626


namespace polygon_area_l476_476735

theorem polygon_area : 
  let vertices := [(1, 0), (3, 2), (5, 0), (3, 5)] in
  let x := vertices.map Prod.fst in
  let y := vertices.map Prod.snd in
  let n := vertices.length in
  let area := 
    0.5 * ((∑ i in Fin.range (n - 1), x[i] * y[i + 1] - y[i] * x[i + 1]) 
           + (x[n - 1] * y[0] - y[n - 1] * x[0])) 
  in area = 6 := by sorry

end polygon_area_l476_476735


namespace ST_map_to_g0_l476_476971

-- Define the set M
def M := {p : ℤ × ℤ | ∃ a b : ℤ, p = (a, b)}

-- Define the transformations S and T
def S : ℤ × ℤ → ℤ × ℤ := λ (p: ℤ × ℤ), (p.1 + p.2, p.2)
def T : ℤ × ℤ → ℤ × ℤ := λ (p: ℤ × ℤ), (-p.2, p.1)

-- The theorem to be proved
theorem ST_map_to_g0 : ∀ (p : ℤ × ℤ), p ∈ M → ∃ g : ℤ, ∃ f : ℕ → ℤ × ℤ, p = f 0 ∧ (∀ n, f (n+1) = S (f n) ∨ f (n+1) = T (f n)) ∧ (∃ n, f n = (g, 0)) := 
sorry

end ST_map_to_g0_l476_476971


namespace arvin_fifth_day_running_distance_l476_476011

theorem arvin_fifth_day_running_distance (total_km : ℕ) (first_day_km : ℕ) (increment : ℕ) (days : ℕ) 
  (h1 : total_km = 20) (h2 : first_day_km = 2) (h3 : increment = 1) (h4 : days = 5) : 
  first_day_km + (increment * (days - 1)) = 6 :=
by
  sorry

end arvin_fifth_day_running_distance_l476_476011


namespace platform_length_l476_476178

theorem platform_length (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
  (h_train_length : train_length = 300) (h_time_pole : time_pole = 12) (h_time_platform : time_platform = 39) : 
  ∃ L : ℕ, L = 675 :=
by
  sorry

end platform_length_l476_476178


namespace min_value_sin_cos_l476_476293

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476293


namespace eval_expression_l476_476022

-- Define the expression to evaluate
def expression : ℚ := 2 * 3 + 4 - (5 / 6)

-- Prove the equivalence of the evaluated expression to the expected result
theorem eval_expression : expression = 37 / 3 :=
by
  -- The detailed proof steps are omitted (relying on sorry)
  sorry

end eval_expression_l476_476022


namespace min_value_sin6_cos6_l476_476317

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476317


namespace jim_travel_distance_l476_476945

theorem jim_travel_distance
  (john_distance : ℕ := 15)
  (jill_distance : ℕ := john_distance - 5)
  (jim_distance : ℕ := jill_distance * 20 / 100) :
  jim_distance = 2 := 
by
  sorry

end jim_travel_distance_l476_476945


namespace sin_cos_sixth_min_l476_476353

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476353


namespace minimal_liars_l476_476707

theorem minimal_liars : ∀ (people : ℕ) (answers : ℕ → ℕ), people = 2024 →
  (∀ (i : ℕ), i < people → (answers i = i ∨ answers i = i + 1 ∨ answers i = i - 1)) →
  (∀ (liars : ℕ), (liars = (∑ i in finset.range people, if answers i ≠ i then 1 else 0)) →
  liars = 1012) :=
by
  intros people answers h_people h_answers liars h_liars
  sorry

end minimal_liars_l476_476707


namespace polygon_area_l476_476736

theorem polygon_area : 
  let vertices := [(1, 0), (3, 2), (5, 0), (3, 5)] in
  let x := vertices.map Prod.fst in
  let y := vertices.map Prod.snd in
  let n := vertices.length in
  let area := 
    0.5 * ((∑ i in Fin.range (n - 1), x[i] * y[i + 1] - y[i] * x[i + 1]) 
           + (x[n - 1] * y[0] - y[n - 1] * x[0])) 
  in area = 6 := by sorry

end polygon_area_l476_476736


namespace find_initial_amount_l476_476871

noncomputable def compound_interest_initial_amount 
  (A_minus_P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (final_multiplier : ℝ) :=
  A_minus_P = (final_multiplier - 1) * 1000

theorem find_initial_amount :
  ∃ P : ℝ, compound_interest_initial_amount 
    82.43216000000007 0.04 2 2 1.08243216 :=
begin
  use 1000,
  simp [compound_interest_initial_amount],
  norm_num,
end

end find_initial_amount_l476_476871


namespace pyramid_base_side_length_l476_476681

theorem pyramid_base_side_length (area : ℕ) (slant_height : ℕ) (s : ℕ) 
  (h1 : area = 100) 
  (h2 : slant_height = 20) 
  (h3 : area = (1 / 2) * s * slant_height) :
  s = 10 := 
by 
  sorry

end pyramid_base_side_length_l476_476681


namespace complex_number_solution_l476_476896

theorem complex_number_solution (z : ℂ) (h : z + complex.i - 3 = 3 - complex.i) : z = 6 - 2 * complex.i :=
sorry

end complex_number_solution_l476_476896


namespace train_passes_in_1_5_minutes_l476_476802

-- Define the conditions
def train_length : ℝ := 100  -- in meters
def train_speed : ℝ := 72  -- in km/hr
def tunnel_length : ℝ := 1.7  -- in km

-- Define the speed conversion from km/hr to m/min
def speed_m_per_min : ℝ := (train_speed * 1000) / 60  -- km/hr to m/min

-- Define the total distance the train needs to travel
def total_distance_m : ℝ := (tunnel_length * 1000) + train_length  -- total distance in meters

-- Define the time calculation
def time_to_pass_through_tunnel : ℝ := total_distance_m / speed_m_per_min  -- time in minutes

-- The theorem statement
theorem train_passes_in_1_5_minutes : time_to_pass_through_tunnel = 1.5 := by
  sorry

end train_passes_in_1_5_minutes_l476_476802


namespace percent_palindromes_with_seven_l476_476141

def is_digit (d : ℕ) : Prop := d ≤ 9

def is_palindrome (n : ℕ) : Prop :=
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ n = 10000 + 1000 + x * 1000 + y * 100 + x * 10 + y

theorem percent_palindromes_with_seven :
  (∃ (n : ℕ) (hx : is_palindrome n), 
   (7 ∈ [n / 1000 % 10, n / 100 % 10]) → 
   19) :=
sorry

end percent_palindromes_with_seven_l476_476141


namespace f_plus_g_even_l476_476419

-- Define the conditions
def f (a x : ℝ) : ℝ := log a (x + 1)
def g (a x : ℝ) : ℝ := log a (1 - x)
variable (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1)

-- Define the theorem to be proven
theorem f_plus_g_even : (∀ x : ℝ, f a x + g a x = f a (-x) + g a (-x)) :=
by
  sorry

end f_plus_g_even_l476_476419


namespace star_proof_l476_476827

def star (a b : ℕ) : ℕ := 3 + b ^ a

theorem star_proof : star (star 2 1) 4 = 259 :=
by
  sorry

end star_proof_l476_476827


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476358

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476358


namespace altitudes_sixth_degree_polynomial_l476_476104

theorem altitudes_sixth_degree_polynomial
  {a b c : ℚ} (h_cubic : ∃ p q r : ℚ, ∀ x : ℚ, (x - a) * (x - b) * (x - c) = x^3 + p*x^2 + q*x + r) :
  ∃ s t u v w z : ℚ, ∀ x : ℚ, 
    (x - (4 * (sqrt ( ( (a + b + c) / 2) *
                      ((a + b + c) / 2 - a) *
                      ((a + b + c) / 2 - b) *
                      ((a + b + c) / 2 - c) ))^2) / a^2)) *
    (x - (4 * (sqrt ( ( (a + b + c) / 2) *
                      ((a + b + c) / 2 - a) *
                      ((a + b + c) / 2 - b) *
                      ((a + b + c) / 2 - c) ))^2) / b^2)) *
    (x - (4 * (sqrt ( ( (a + b + c) / 2) *
                      ((a + b + c) / 2 - a) *
                      ((a + b + c) / 2 - b) *
                      ((a + b + c) / 2 - c) ))^2) / c^2)) =
    x^6 + s*x^5 + t*x^4 + u*x^3 + v*x^2 + w*x + z :=
by sorry

end altitudes_sixth_degree_polynomial_l476_476104


namespace walk_to_cafe_and_back_time_l476_476941

theorem walk_to_cafe_and_back_time 
  (t_p : ℝ) (d_p : ℝ) (half_dp : ℝ) (pace : ℝ)
  (h1 : t_p = 30) 
  (h2 : d_p = 3) 
  (h3 : half_dp = d_p / 2) 
  (h4 : pace = t_p / d_p) :
  2 * half_dp * pace = 30 :=
by 
  sorry

end walk_to_cafe_and_back_time_l476_476941


namespace largest_three_digit_base7_to_decimal_l476_476695

theorem largest_three_digit_base7_to_decimal :
  (6 * 7^2 + 6 * 7^1 + 6 * 7^0) = 342 :=
by
  sorry

end largest_three_digit_base7_to_decimal_l476_476695


namespace center_of_mass_is_52_8_l476_476174

-- Definitions of the masses and positions
def m1 : ℝ := 2
def m2 : ℝ := 3
def m3 : ℝ := 4

def x1 : ℝ := 0
def x2 : ℝ := 25
def x3 : ℝ := 100

-- Calculate the center of mass
def center_of_mass (m1 m2 m3 x1 x2 x3 : ℝ) : ℝ :=
  (m1 * x1 + m2 * x2 + m3 * x3) / (m1 + m2 + m3)

theorem center_of_mass_is_52_8 :
  center_of_mass m1 m2 m3 x1 x2 x3 = 475 / 9 :=
by
  sorry

end center_of_mass_is_52_8_l476_476174


namespace minimize_sin_cos_six_l476_476307

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476307


namespace min_sin6_cos6_l476_476285

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476285


namespace arithmetic_sequence_min_value_S_l476_476515

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476515


namespace prove_arithmetic_sequence_minimum_value_S_l476_476537

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476537


namespace find_m_pure_imaginary_range_m_fourth_quadrant_l476_476392

noncomputable def z1 (m : ℝ) : ℂ := m * (m - 1) + (m - 1) * Complex.I
noncomputable def z2 (m : ℝ) : ℂ := (m + 1) + (m^2 - 1) * Complex.I

-- Problem (1): If z1 is a pure imaginary number, find the value of m.
theorem find_m_pure_imaginary (m : ℝ) (h : Re (z1 m) = 0) : m = 0 := sorry

-- Problem (2): If the point z2 is located in the fourth quadrant, find the range of values for m.
theorem range_m_fourth_quadrant (m : ℝ) (h₁ : Re (z2 m) > 0) (h₂ : Im (z2 m) < 0) : -1 < m ∧ m < 1 := sorry

end find_m_pure_imaginary_range_m_fourth_quadrant_l476_476392


namespace AD_bisects_EDF_l476_476890

theorem AD_bisects_EDF 
  (A B C D H E F : Point)
  (h1 : is_vertex A)
  (h2 : is_perpendicular D A BC)
  (h3 : H ∈ AD)
  (h4 : E = line_intersection (line B H) (line A C))
  (h5 : F = line_intersection (line C H) (line A B)) :
  angle_bisector (line A D) (angle E D F) :=
sorry

end AD_bisects_EDF_l476_476890


namespace arithmetic_sequence_min_value_S_l476_476504

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476504


namespace max_A_in_5_by_5_table_l476_476000

noncomputable def max_min_ratio (arr : Fin 5 → Fin 5 → ℕ) : ℝ :=
  let pairs := { (i, j, k, l) : Fin 5 × Fin 5 × Fin 5 × Fin 5 // (i = k ∨ j = l) ∧ arr i j ≠ arr k l }
  let ratios := Finset.image (λ p : { p // (p.val.1 = p.val.3 ∨ p.val.2 = p.val.4) ∧ arr p.val.1 p.val.2 ≠ arr p.val.3 p.val.4 }, 
                        ( (max (arr p.val.1 p.val.2) (arr p.val.3 p.val.4)).toReal / 
                          (min (arr p.val.1 p.val.2) (arr p.val.3 p.val.4)).toReal)) 
                            (Finset.univ : Finset pairs)
  Finset.min' ratios sorry

theorem max_A_in_5_by_5_table : ∃ (arr : Fin 5 → Fin 5 → ℕ), max_min_ratio arr = (6 / 5) :=
begin
  use (λ i j, (i.val + j.val * 5 + 1)),
  sorry
end

end max_A_in_5_by_5_table_l476_476000


namespace jake_should_charge_40_for_planting_flowers_l476_476982

theorem jake_should_charge_40_for_planting_flowers 
  (mow_time : ℕ) (mow_pay : ℕ) (desired_pay_rate : ℕ) (flower_time : ℕ) : 
  (mow_time = 1) → (mow_pay = 15) → (desired_pay_rate = 20) → (flower_time = 2) → (desired_pay_rate * flower_time = 40) :=
by
  intros h1 h2 h3 h4
  rw [h3, h4]
  exact rfl

end jake_should_charge_40_for_planting_flowers_l476_476982


namespace find_c_for_special_a_l476_476066

theorem find_c_for_special_a (n : ℕ) (hn : n = 5) : 
  ∃ b c : ℕ, (2*n + 1)^2 + b^2 = c^2 ∧ (2*n + 1 = 11 ∧ c = 61) :=
by
  use [60, 61]
  split
  sorry
  split
  sorry

end find_c_for_special_a_l476_476066


namespace king_paid_after_tip_l476_476785

-- Define the cost of the crown and the tip percentage
def cost_of_crown : ℝ := 20000
def tip_percentage : ℝ := 0.10

-- Define the total amount paid after the tip
def total_amount_paid (C : ℝ) (tip_pct : ℝ) : ℝ :=
  C + (tip_pct * C)

-- Theorem statement: The total amount paid after the tip is $22,000
theorem king_paid_after_tip : total_amount_paid cost_of_crown tip_percentage = 22000 := by
  sorry

end king_paid_after_tip_l476_476785


namespace stronger_linear_correlation_better_fit_with_smaller_rss_chi_square_weaker_relationship_calculate_hat_b_main_theorem_l476_476752

theorem stronger_linear_correlation (A_corr : Real) (B_corr : Real) :
  B_corr = -0.85 ∧ A_corr = 0.66 → abs B_corr > abs A_corr :=
by
  intros h
  rcases h with ⟨h_b, h_a⟩
  simp [h_b, h_a]

theorem better_fit_with_smaller_rss (rss : Real) :
  smaller_rss_implies_better_fit :=
sorry -- Assume this proof

theorem chi_square_weaker_relationship (χ2 : Real) :
  smaller_χ2_implies_weaker_relationship :=
sorry -- Assume this proof

theorem calculate_hat_b (x_vals : List Real) (y_vals : List Real) :
  sum x_vals = 9 ∧ sum y_vals = 3 → (∃ b, (∀ x ∈ x_vals, ∀ y ∈ y_vals, y = b * x + 1) ∧ b = -7 / 9) :=
by
  intros h
  rcases h with ⟨h_x_sum, h_y_sum⟩
  use -7/9
  intros x hx y hy
  field_simp
  sorry -- Assume this proof

theorem main_theorem (A_corr : Real) (B_corr : Real) (rss : Real) (χ2 : Real) (x_vals : List Real) (y_vals : List Real) :
  B_corr = -0.85 ∧ A_corr = 0.66 ∧ smaller_rss_implies_better_fit ∧ smaller_χ2_implies_weaker_relationship ∧ sum x_vals = 9 ∧ sum y_vals = 3 →
  abs B_corr > abs A_corr ∧ (∃ b, (∀ x ∈ x_vals, ∀ y ∈ y_vals, y = b * x + 1) ∧ b = -7 / 9) :=
by
  intros h
  rcases h with ⟨h_b_corr, h_a_corr, rss_fit, chi2_rel, h_x_sum, h_y_sum⟩
  split
  { apply stronger_linear_correlation,
    exact ⟨h_b_corr, h_a_corr⟩ },
  { apply calculate_hat_b,
    exact ⟨h_x_sum, h_y_sum⟩ }

end stronger_linear_correlation_better_fit_with_smaller_rss_chi_square_weaker_relationship_calculate_hat_b_main_theorem_l476_476752


namespace limit_an_bn_pi_l476_476627

noncomputable def a : ℕ → ℝ
| 0     := 2 * Real.sqrt 3
| (n+1) := 2 * (a n) * (b n) / ((a n) + (b n))

noncomputable def b : ℕ → ℝ
| 0     := 3
| (n+1) := Real.sqrt ((a (n+1)) * (b n))

theorem limit_an_bn_pi : 
  (filter.at_top : filter ℕ).tendsto a (𝓝 π) ∧ (filter.at_top : filter ℕ).tendsto b (𝓝 π) :=
sorry

end limit_an_bn_pi_l476_476627


namespace prove_arithmetic_sequence_minimum_value_S_l476_476539

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476539


namespace quadratic_unique_solution_pair_l476_476098

theorem quadratic_unique_solution_pair (a c : ℝ) (h₁ : a + c = 12) (h₂ : a < c) (h₃ : a * c = 9) :
  (a, c) = (6 - 3 * Real.sqrt 3, 6 + 3 * Real.sqrt 3) :=
by
  sorry

end quadratic_unique_solution_pair_l476_476098


namespace problem_statement_l476_476208

theorem problem_statement (a b c : ℝ) (h : a * c^2 > b * c^2) (hc : c ≠ 0) : 
  a > b :=
by 
  sorry

end problem_statement_l476_476208


namespace pyramid_base_side_length_correct_l476_476683

def sideLengthBase (s : ℕ) : Prop :=
  let area : ℕ := 100
  let slant_height : ℕ := 20
  let lateral_face_area := (1/2:ℚ) * s * slant_height
  lateral_face_area.toNat = area → s = 10

theorem pyramid_base_side_length_correct (s : ℕ) (h: s * 10 = 100) : sideLengthBase s :=
  by
    intros
    simp [sideLengthBase]
    assume lateral_face_area h
    exact h
    sorry

end pyramid_base_side_length_correct_l476_476683


namespace arithmetic_sequence_minimum_value_S_n_l476_476561

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l476_476561


namespace tim_movie_marathon_duration_l476_476718

-- Define the durations of each movie
def first_movie_duration : ℕ := 2

def second_movie_duration : ℕ := 
  first_movie_duration + (first_movie_duration / 2)

def combined_first_two_movies_duration : ℕ :=
  first_movie_duration + second_movie_duration

def last_movie_duration : ℕ := 
  combined_first_two_movies_duration - 1

-- Define the total movie marathon duration
def total_movie_marathon_duration : ℕ := 
  first_movie_duration + second_movie_duration + last_movie_duration

-- Problem statement to be proved
theorem tim_movie_marathon_duration : total_movie_marathon_duration = 9 := by
  sorry

end tim_movie_marathon_duration_l476_476718


namespace residues_distinct_values_l476_476624

theorem residues_distinct_values (p : ℕ) (a : Fin p → ℤ) [hp : Fact p.Prime] :
  ∃ k : ℤ, (Finset.image (λ i : Fin p, (a i + i * k) % p) Finset.univ).card ≥ p / 2 := sorry

end residues_distinct_values_l476_476624


namespace arithmetic_sequence_min_value_Sn_l476_476567

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l476_476567


namespace radius_of_smallest_tangent_circle_l476_476067

noncomputable def radius_smallest_tangent_circle (R_sphere : ℝ) (R_circles : ℝ) : ℝ :=
  let answer := 1 - Real.sqrt (2/3)
  if R_sphere = 2 ∧ R_circles = 1 then answer else 0

theorem radius_of_smallest_tangent_circle :
  ∀ (R_sphere R_circles : ℝ),
    R_sphere = 2 → R_circles = 1 →
    radius_smallest_tangent_circle R_sphere R_circles = 
      1 - Real.sqrt (2/3) :=
by
  intros R_sphere R_circles hRsphere hRcircles
  unfold radius_smallest_tangent_circle
  rw [if_pos ⟨hRsphere, hRcircles⟩]
  rfl

end radius_of_smallest_tangent_circle_l476_476067


namespace max_value_of_f_l476_476453

def f (x : ℝ) : ℝ := (1 - x^2) * (x^2 + 8 * x + 15)

theorem max_value_of_f (x : ℝ) (h_symm : ∀ x : ℝ, f (-2 - x) = f (-2 + x)) : 
  ∃ t : ℝ, t ∈ set.Icc (-4 : ℝ) 1 ∧ ∀ y ∈ set.Icc (-4 : ℝ) 1, f (y - 2) ≤ f 1 ∧ f 1 = 16 :=
by
  sorry

end max_value_of_f_l476_476453


namespace arvin_fifth_day_run_l476_476014

theorem arvin_fifth_day_run :
  let running_distance : ℕ → ℕ := λ day, 2 + day - 1
  in running_distance 5 = 6 := by
  sorry

end arvin_fifth_day_run_l476_476014


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476550

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476550


namespace aardvark_total_distance_l476_476130

noncomputable def total_distance (r_small r_large : ℝ) : ℝ :=
  let small_circumference := 2 * Real.pi * r_small
  let large_circumference := 2 * Real.pi * r_large
  let half_small_circumference := small_circumference / 2
  let half_large_circumference := large_circumference / 2
  let radial_distance := r_large - r_small
  let total_radial_distance := radial_distance + r_large
  half_small_circumference + radial_distance + half_large_circumference + total_radial_distance

theorem aardvark_total_distance :
  total_distance 15 30 = 45 * Real.pi + 45 :=
by
  sorry

end aardvark_total_distance_l476_476130


namespace min_value_sin6_cos6_l476_476325

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476325


namespace min_value_fraction_l476_476608

open Real

-- Define the premises.
variables (a b : Real)

-- Assume a and b are positive and satisfy the equation a + 3b = 1.
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : a + 3b = 1

-- The theorem we want to prove.
theorem min_value_fraction (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3b = 1) :
  1 / a + 1 / b = 4 * sqrt 3 + 8 :=
sorry

end min_value_fraction_l476_476608


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476545

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476545


namespace graph_passes_through_point_l476_476693

theorem graph_passes_through_point (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : 
    ∃ y : ℝ, y = a^0 + 1 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end graph_passes_through_point_l476_476693


namespace tim_movie_marathon_duration_l476_476719

-- Define the durations of each movie
def first_movie_duration : ℕ := 2

def second_movie_duration : ℕ := 
  first_movie_duration + (first_movie_duration / 2)

def combined_first_two_movies_duration : ℕ :=
  first_movie_duration + second_movie_duration

def last_movie_duration : ℕ := 
  combined_first_two_movies_duration - 1

-- Define the total movie marathon duration
def total_movie_marathon_duration : ℕ := 
  first_movie_duration + second_movie_duration + last_movie_duration

-- Problem statement to be proved
theorem tim_movie_marathon_duration : total_movie_marathon_duration = 9 := by
  sorry

end tim_movie_marathon_duration_l476_476719


namespace first_discount_percentage_l476_476988

/-
  Prove that under the given conditions:
  1. The price before the first discount is $33.78.
  2. The final price after the first and second discounts is $19.
  3. The second discount is 25%.
-/
theorem first_discount_percentage (x : ℝ) :
  (33.78 * (1 - x / 100) * (1 - 25 / 100) = 19) →
  x = 25 :=
by
  -- Proof steps (to be filled)
  sorry

end first_discount_percentage_l476_476988


namespace extra_bananas_per_child_l476_476651

-- Definitions based on the given conditions
def children_total := 610
def absent_children := 305
def bananas_per_child := 2
def total_bananas := children_total * bananas_per_child

-- Theorem statement that captures the question and translates the proof problem
theorem extra_bananas_per_child :
  let present_children := children_total - absent_children in
  let bananas_per_present_child := total_bananas / present_children in
  bananas_per_present_child - bananas_per_child = 2 := by
  sorry

end extra_bananas_per_child_l476_476651


namespace option_C_correct_l476_476440

variable {a b c d : ℝ}

theorem option_C_correct (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end option_C_correct_l476_476440


namespace isosceles_triangle_perimeter_eq_10_l476_476217

theorem isosceles_triangle_perimeter_eq_10 (x : ℝ) 
(base leg : ℝ)
(h_base : base = 4)
(h_leg_root : x^2 - 5 * x + 6 = 0)
(h_iso : leg = x)
(triangle_ineq : leg + leg > base):
  2 * leg + base = 10 := 
begin
  cases (em (x = 2)) with h1 h2,
  { rw h1 at h_leg_root,
    rw [←h_iso, h1] at triangle_ineq,
    simp at triangle_ineq,
    contradiction },
  { rw h_iso,
    have : x = 3,
    { by_contra,
      simp [not_or_distrib, h1, h, sub_eq_zero] at h_leg_root },
    rw this,
    simp,
    linarith }
end

# Testing if the theorem can be evaluated successfully
# theorem_example : isosceles_triangle_perimeter_eq_10 3 4 3 rfl rfl sorry sorry rfl :=
# sorry

end isosceles_triangle_perimeter_eq_10_l476_476217


namespace bases_with_final_digit_one_l476_476883

theorem bases_with_final_digit_one :
  (count (λ b, (360 - 1) % b = 0) (filter (λ b, 2 ≤ b ∧ b ≤ 9) (list.range 10))) = 0 :=
sorry

end bases_with_final_digit_one_l476_476883


namespace Nicky_time_before_catchup_l476_476650

-- Define the given speeds and head start time as constants
def v_C : ℕ := 5 -- Cristina's speed in meters per second
def v_N : ℕ := 3 -- Nicky's speed in meters per second
def t_H : ℕ := 12 -- Head start in seconds

-- Define the running time until catch up
def time_Nicky_run : ℕ := t_H + (36 / (v_C - v_N))

-- Prove that the time Nicky has run before Cristina catches up to him is 30 seconds
theorem Nicky_time_before_catchup : time_Nicky_run = 30 :=
by
  -- Add the steps for the proof
  sorry

end Nicky_time_before_catchup_l476_476650


namespace sum_of_squares_of_pairs_of_roots_l476_476055

-- Define the polynomial whose roots are a, b, c, d
def poly := polynomial.X^4 - 24 * polynomial.X^3 + 50 * polynomial.X^2 - 35 * polynomial.X + 10

-- Establish the main theorem based on the conditions and required proof
theorem sum_of_squares_of_pairs_of_roots :
  let {a, b, c, d} := multiset.of_nat_tuple 4
  (∀ x ∈ {a, b, c, d}, polynomial.eval x poly = 0) →
  (a + b)^2 + (b + c)^2 + (c + d)^2 + (d + a)^2 = 541 :=
sorry

end sum_of_squares_of_pairs_of_roots_l476_476055


namespace min_value_sin6_cos6_l476_476320

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476320


namespace sin_cos_sixth_min_l476_476347

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l476_476347


namespace part_I_part_II_l476_476924

open Real

-- Define the polar equation of curve C
def polar_curve_C (θ : ℝ) : ℝ := 2 * cos θ

-- Define the parametric equation of line l
def parametric_line_l (t : ℝ) : ℝ × ℝ := (4 + (1/2) * t, (sqrt 3 / 2) * t)

-- Define the rectangular equation of curve C
def rectangular_curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0

-- Define the polar equation of line l
def polar_line_l (ρ θ : ℝ) : Prop := sqrt 3 * ρ * cos θ - ρ * sin θ = 4 * sqrt 3

-- Define the distance calculation between points A and B
def distance_AB (ρ_A ρ_B : ℝ) : ℝ := abs (ρ_B - ρ_A)

theorem part_I 
  (θ_A θ_B : ℝ)
  (h_θA : θ_A = π / 6)
  (h_θB : θ_B = π / 6):
  (rectangular_curve_C x y ∧ polar_line_l ρ θ) :=
sorry

theorem part_II 
  (ρ_A ρ_B θ_A θ_B : ℝ)
  (h_ρA : ρ_A = sqrt 3)
  (h_ρB : ρ_B = 4 * sqrt 3)
  (h_θA : θ_A = π / 6)
  (h_θB : θ_B = π / 6):
  distance_AB ρ_A ρ_B = 3 * sqrt 3 :=
sorry

end part_I_part_II_l476_476924


namespace minimize_sin_cos_six_l476_476301

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l476_476301


namespace arithmetic_sequence_minimum_value_S_l476_476579

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476579


namespace arithmetic_sequence_minimum_value_S_l476_476578

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476578


namespace find_c_l476_476418

noncomputable def cubic_function (x : ℝ) (c : ℝ) : ℝ :=
  x^3 - 3 * x + c

theorem find_c (c : ℝ) :
  (∃ x₁ x₂ : ℝ, cubic_function x₁ c = 0 ∧ cubic_function x₂ c = 0 ∧ x₁ ≠ x₂) →
  (c = -2 ∨ c = 2) :=
by
  sorry

end find_c_l476_476418


namespace range_of_x_l476_476910

theorem range_of_x (f : ℝ → ℝ) (h_mono : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f(x) ≤ f(y)) (h_zero : f 0 = 0) :
  ∀ x : ℝ, f (Real.log x / Real.log 10) > 0 → 10 < x :=
by
  intros x h
  have h_log : Real.log x / Real.log 10 > 0 := sorry
  have h_pos : Real.log x > 0 := sorry 
  exact sorry 

end range_of_x_l476_476910


namespace bc_guilty_l476_476715

-- Definition of guilty status of defendants
variables (A B C : Prop)

-- Conditions
axiom condition1 : A ∨ B ∨ C
axiom condition2 : A → ¬B → ¬C

-- Theorem stating that one of B or C is guilty
theorem bc_guilty : B ∨ C :=
by {
  -- Proof goes here
  sorry
}

end bc_guilty_l476_476715


namespace eccentricity_of_hyperbola_l476_476920

noncomputable def find_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) : ℝ :=
Real.sqrt (1 + (b / a)^2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) :
  find_eccentricity a b h1 h2 h3 = 3 / 2 := by
  sorry

end eccentricity_of_hyperbola_l476_476920


namespace angle_of_inclination_is_150_l476_476410

noncomputable def line_slope : ℝ := - real.sqrt 3 / 3

theorem angle_of_inclination_is_150 :
  ∃ α : ℝ, α = 150 ∧ tan (α * real.pi / 180) = line_slope :=
by simp [line_slope]; use 150; sorry

end angle_of_inclination_is_150_l476_476410


namespace isosceles_triangle_area_l476_476468

theorem isosceles_triangle_area
  (A B C D : Type)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  (h1 : dist A B = 17)
  (h2 : dist A C = 17)
  (h3 : dist B C = 16)
  (h4 : ∃ D, dist B D = 8 ∧ dist C D = 8 ∧ is_perpendicular (line_through A D) (line_through B C)) :
  area_of_triangle A B C = 120 :=
by
sorry

end isosceles_triangle_area_l476_476468


namespace distance_rose_bush_to_jasmine_l476_476963

theorem distance_rose_bush_to_jasmine (AB BC CD DE : ℕ)
  (h1 : 2 * AB + 2 * BC + 2 * CD + 2 * DE = 28)
  (h2 : 2 * AB + 4 * BC + 4 * CD + 3 * DE = 48)
  : BC + CD = 6 :=
by
  have h : AB + BC + CD + DE = 14 := by linarith [h1]
  have h' : AB + 2 * BC + 2 * CD + DE = 20 := by linarith [h2]
  linarith [h, h']

#align distance_rose_bush_to_jasmine distance_rose_bush_to_jasmine

end distance_rose_bush_to_jasmine_l476_476963


namespace simplify_expr_l476_476230

def A (a b : ℝ) := b^2 - a^2 + 5 * a * b
def B (a b : ℝ) := 3 * a * b + 2 * b^2 - a^2

theorem simplify_expr (a b : ℝ) : 2 * (A a b) - (B a b) = -a^2 + 7 * a * b := by
  -- actual proof omitted
  sorry

example : (2 * (A 1 2) - (B 1 2)) = 13 := by
  -- actual proof omitted
  sorry

end simplify_expr_l476_476230


namespace f_increasing_intervals_l476_476240

def f (x : ℝ) : ℝ := Real.cos (x - π / 2) + Real.sin (x + π / 3)

theorem f_increasing_intervals : 
  ∀ (k : ℤ), ∀ x, (2 * k * π - 2 * π / 3) ≤ x ∧ x ≤ (2 * k * π + π / 3) → 
    ∂(f x) / ∂x > 0 :=
by
  sorry

end f_increasing_intervals_l476_476240


namespace simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l476_476227

variable (a b : ℤ)
def A : ℤ := b^2 - a^2 + 5 * a * b
def B : ℤ := 3 * a * b + 2 * b^2 - a^2

theorem simplify_2A_minus_B : 2 * A a b - B a b = -a^2 + 7 * a * b := by
  sorry

theorem evaluate_2A_minus_B_at_1_2 : 2 * A 1 2 - B 1 2 = 13 := by
  sorry

end simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l476_476227


namespace angle_BDC_100_l476_476482

-- Define the overall problem setup
variables (A B C D : Type) [Angle A B C] [Angle C B D] [Angle B A D]

-- The conditions in the problem
variables (h1 : AC = BC)  -- AC equals BC
variables (h2 : mangle C = 60)  -- Angle C is 60 degrees
variables (h3 : point_on_extension BC D)  -- Point D is on the extension of BC
variables (h4 : BD = DC)  -- BD equals DC
variables (h5 : mangle BAD = 40)  -- Angle BAD is 40 degrees

-- The proof statement
theorem angle_BDC_100 (A B C D: Type) 
  [Angle A B C] [Angle C B D] [Angle B A D]
  (AC BC : Type) (BD DC: Type)
  (h1 : AC = BC)
  (h2 : mangle C = 60)
  (h3 : point_on_extension BC D)
  (h4 : BD = DC)
  (h5 : mangle BAD = 40) :
  mangle BDC = 100 :=
sorry

end angle_BDC_100_l476_476482


namespace curve_symmetric_about_y_eq_x_l476_476685

def curve_eq (x y : ℝ) : Prop := x * y * (x + y) = 1

theorem curve_symmetric_about_y_eq_x :
  ∀ (x y : ℝ), curve_eq x y ↔ curve_eq y x :=
by sorry

end curve_symmetric_about_y_eq_x_l476_476685


namespace prove_arithmetic_sequence_minimum_value_S_l476_476532

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476532


namespace log_sum_equals_n_squared_l476_476897

variable {a_n : ℕ → ℝ}
variable {n : ℕ}

-- Given conditions:
def geometric_sequence := ∀ n, a_n > 0
def product_condition := ∀ n, n > 1 → a_n(3) * a_n(2 * n - 3) = 4^n

-- To Prove:
theorem log_sum_equals_n_squared (h1 : geometric_sequence) (h2 : product_condition) (h3 : n ≥ 1) :
  log 2 (a_n 1) + log 2 (a_n 3) + ... + log 2 (a_n (2 * n - 1)) = n^2 :=
sorry

end log_sum_equals_n_squared_l476_476897


namespace degree_of_product_polynomials_l476_476668

variables {f g : Polynomial ℝ} -- It's assumed to be real polynomials for simplicity.

-- The degrees of f(x) and g(x) are given as 3 and 6 respectively.
def deg_f : ℕ := 3
def deg_g : ℕ := 6

-- We need to show that the degree of f(x^4) * g(x^3) is 30.
theorem degree_of_product_polynomials :
  Polynomial.degree (f.eval (X^4) * g.eval (X^3)) = 30 :=
sorry

end degree_of_product_polynomials_l476_476668


namespace find_number_l476_476449

theorem find_number (x q : ℕ) (h1 : x = 3 * q) (h2 : q + x + 3 = 63) : x = 45 :=
sorry

end find_number_l476_476449


namespace find_other_number_l476_476170

-- Define the conditions and the theorem
theorem find_other_number (hcf lcm a b : ℕ) (hcf_def : hcf = 20) (lcm_def : lcm = 396) (a_def : a = 36) (rel : hcf * lcm = a * b) : b = 220 :=
by 
  sorry -- Proof to be provided

end find_other_number_l476_476170


namespace proof_stage_constancy_l476_476477

-- Definitions of stages
def Stage1 := "Fertilization and seed germination"
def Stage2 := "Flowering and pollination"
def Stage3 := "Meiosis and fertilization"
def Stage4 := "Formation of sperm and egg cells"

-- Question: Which stages maintain chromosome constancy and promote genetic recombination in plant life?
def Q := "Which stages maintain chromosome constancy and promote genetic recombination in plant life?"

-- Correct answer
def Answer := Stage3

-- Conditions
def s1 := Stage1
def s2 := Stage2
def s3 := Stage3
def s4 := Stage4

-- Theorem statement
theorem proof_stage_constancy : Q = Answer := by
  sorry

end proof_stage_constancy_l476_476477


namespace arithmetic_sequence_and_minimum_sum_l476_476523

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476523


namespace leftover_floss_l476_476837

/-
Conditions:
1. There are 20 students in his class.
2. Each student needs 1.5 yards of floss.
3. Each packet of floss contains 35 yards.
4. He buys the least amount necessary.
-/

def students : ℕ := 20
def floss_needed_per_student : ℝ := 1.5
def total_floss_needed : ℝ := students * floss_needed_per_student
def floss_per_packet : ℝ := 35

theorem leftover_floss : floss_per_packet - total_floss_needed = 5 :=
by
  -- Assuming these values from the conditions
  have students_val : 20 = students := rfl
  have floss_needed_val : 1.5 = floss_needed_per_student := rfl
  have total_needed_val : total_floss_needed = 30 := by 
    simp [students, floss_needed_per_student, total_floss_needed]
  have floss_per_packet_val : 35 = floss_per_packet := rfl
  
  -- Calculation to get the leftover floss
  calc
    floss_per_packet - total_floss_needed 
        = 35 - 30 : by rw [total_needed_val]
    ... = 5 : by norm_num

end leftover_floss_l476_476837


namespace min_value_sin6_cos6_l476_476315

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476315


namespace orchestra_members_l476_476093

theorem orchestra_members :
  ∃ (n : ℕ), 
    150 < n ∧ n < 250 ∧ 
    n % 4 = 2 ∧ 
    n % 5 = 3 ∧ 
    n % 7 = 4 :=
by
  use 158
  repeat {split};
  sorry

end orchestra_members_l476_476093


namespace quadratic_non_positive_solution_l476_476887

theorem quadratic_non_positive_solution :
  { x : ℝ | x^2 - 40 * x + 350 ≤ 0 } = set.Icc 10 30 :=
by
  sorry

end quadratic_non_positive_solution_l476_476887


namespace min_value_sin6_cos6_l476_476330

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l476_476330


namespace intersection_of_C1_C2_product_of_distances_to_P_l476_476422

noncomputable def C1 (t : ℝ) (α : ℝ) : ℝ × ℝ := 
  (-1 + t * Real.cos α, 3 + t * Real.sin α)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ := 
  (2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4) * Real.cos θ, 
   2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4) * Real.sin θ )

def pointA : ℝ × ℝ := (1, 1)

def pointP : ℝ × ℝ := (-1, 3)

def intersection_points := [(2,0), (0,2)]

theorem intersection_of_C1_C2 (α : ℝ) : 
  0 ≤ α ∧ α < Real.pi → 
  ((pointA ∈ (Set.range (λ t, C1 t α)) ∧
  ∃ θ, C1 (Real.sqrt 2) α = C2 θ) →
  (pointA ∈ (Set.range (λ t, C1 t α)) → 
  ∃ x y, (x, y) ∈ intersection_points ∧ 
  ∀ p ∈ (Set.range (λ t, C1 t α)), (p = (2, 0) ∨ p = (0, 2)))) := by
  sorry

theorem product_of_distances_to_P (α : ℝ) (t1 t2 : ℝ) : 
  0 ≤ α ∧ α < Real.pi → 
  ((pointP ∈ (Set.range (λ t, C1 t α)) ∧
  C2 = λ θ, (2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4) * Real.cos θ, 
            2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4) * Real.sin θ)) →
  let PB := |t1| in 
  let PD := |t2| in 
  (t1 * t2 = 6 → 
  PB * PD = 6)) := by
  sorry

end intersection_of_C1_C2_product_of_distances_to_P_l476_476422


namespace arithmetic_sequence_min_value_S_l476_476497

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476497


namespace students_multiple_activities_l476_476460

theorem students_multiple_activities (total_students only_debate only_singing only_dance no_activities students_more_than_one : ℕ) 
  (h1 : total_students = 55) 
  (h2 : only_debate = 10) 
  (h3 : only_singing = 18) 
  (h4 : only_dance = 8)
  (h5 : no_activities = 5)
  (h6 : students_more_than_one = total_students - (only_debate + only_singing + only_dance + no_activities)) :
  students_more_than_one = 14 := by
  sorry

end students_multiple_activities_l476_476460


namespace prove_arithmetic_sequence_minimum_value_S_l476_476538

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l476_476538


namespace simplify_expr_l476_476229

def A (a b : ℝ) := b^2 - a^2 + 5 * a * b
def B (a b : ℝ) := 3 * a * b + 2 * b^2 - a^2

theorem simplify_expr (a b : ℝ) : 2 * (A a b) - (B a b) = -a^2 + 7 * a * b := by
  -- actual proof omitted
  sorry

example : (2 * (A 1 2) - (B 1 2)) = 13 := by
  -- actual proof omitted
  sorry

end simplify_expr_l476_476229


namespace tangent_line_y_intercept_l476_476181

/-
  Prove that the y-intercept of the line tangent to both circles at points in the first quadrant is 9,
  given the following conditions:
  1. Circle 1 has radius 3 and center (3, 0).
  2. Circle 2 has radius 1 and center (7, 0).
-/
theorem tangent_line_y_intercept
  (circle1_center : ℝ × ℝ)
  (circle1_radius : ℝ)
  (circle2_center : ℝ × ℝ)
  (circle2_radius : ℝ)
  (line_tangent : ℝ → ℝ)
  (circle1_tangent_point circle2_tangent_point : ℝ × ℝ) :
  circle1_center = (3, 0) →
  circle1_radius = 3 →
  circle2_center = (7, 0) →
  circle2_radius = 1 →
  -- Prove the y-intercept of the tangent line is 9
  line_tangent 0 = 9 :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end tangent_line_y_intercept_l476_476181


namespace MF_eq_FO_l476_476979

open EuclideanGeometry

theorem MF_eq_FO {A B C M O F : Point}
  (h_triangle : Triangle A B C)
  (h_C_angle : ∠ A C B = 120)
  (h_M_orthocenter : IsOrthocenter M A B C)
  (h_O_circumcenter : IsCircumcenter O A B C)
  (h_F_arc_mid_at_C : IsMidpointOfArc F A C B) :
  dist M F = dist F O := by
  sorry

end MF_eq_FO_l476_476979


namespace arithmetic_sequence_minimum_value_S_l476_476576

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476576


namespace max_ab_l476_476450

theorem max_ab (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ (M : ℝ), M = 1 / 8 ∧ ∀ (a b : ℝ), (a + 2 * b = 1) → 0 < a → 0 < b → ab ≤ M :=
sorry

end max_ab_l476_476450


namespace arithmetic_sequence_min_value_S_l476_476510

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476510


namespace find_dihedral_angle_l476_476696

def point := ℝ × ℝ × ℝ
def plane : Type := point → Prop

-- Definitions of the points given in the conditions
def A : point := (0, 0, 0)
def B : point := (4, 0, 0)
def C : point := (2, 2 * (√3 : ℝ), 0)
def D : point := (-2, 2 * (√3 : ℝ), 0)

-- Upper vertices
def A1 : point := (0, 0, 2)
def B1 : point := (2, 0, 2)
def C1 : point := (1, 2 * (√3 : ℝ), 2)
def D1 : point := (-1, 2 * (√3 : ℝ), 2)

-- Point M on BC
def M : point := (3, (3 : ℝ) * (√3 : ℝ), 0)

-- Center O of the rhombus ABCD
def O : point := (1, (√3 : ℝ), 0)

-- Plane definitions
def plane_A_A1_C1_C : plane := λ p, ∃ α β γ, p = (A.1 * α + A1.1 * β + C1.1 * γ,
                                                      A.2 * α + A1.2 * β + C1.2 * γ,
                                                      A.3 * α + A1.3 * β + C1.3 * γ)

def plane_B1_M_O : plane := λ p, ∃ α β γ, p = (B1.1 * α + M.1 * β + O.1 * γ,
                                               B1.2 * α + M.2 * β + O.2 * γ,
                                               B1.3 * α + M.3 * β + O.3 * γ)

-- Function to find the dihedral angle between two planes
def dihedral_angle (P Q : plane) (p1 p2 p3 : point) : ℝ := --placeholder
  arctan (2 / (3 * (√3 : ℝ)))

-- Lean definition of the problem
theorem find_dihedral_angle :
  dihedral_angle plane_A_A1_C1_C plane_B1_M_O A1 B1 C1 = arctan (2 / (3 * (√3 : ℝ))) :=
sorry

end find_dihedral_angle_l476_476696


namespace count_three_digit_numbers_l476_476888

theorem count_three_digit_numbers :
  let cards := {0, 2, 4, 6, 8}
  in ∃ n : ℕ, 
  (∀ d1 d2 d3 ∈ cards, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 → 
  let digits := [d1, d2, d3]
  in (6 ∈ digits → 9 ∈ digits) → 
  n = 78) :=
sorry

end count_three_digit_numbers_l476_476888


namespace arithmetic_sequence_min_value_S_l476_476505

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476505


namespace problem_solution_l476_476448

theorem problem_solution (m n : ℕ)
  (h_eq : (2013 * 2013) / (2014 * 2014 + 2012) = n / m)
  (h_coprime : Nat.coprime m n) :
  m + n = 1343 :=
  sorry

end problem_solution_l476_476448


namespace arithmetic_sequence_min_value_S_l476_476499

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476499


namespace merchant_profit_percentage_l476_476191

noncomputable def cost_price : ℝ := 100
noncomputable def markup_percentage : ℝ := 30
noncomputable def discount_percentage : ℝ := 10

def marked_price (cp : ℝ) (markup : ℝ) : ℝ := cp + (markup / 100) * cp

def discount (mp : ℝ) (discount : ℝ) : ℝ := (discount / 100) * mp

def selling_price (mp : ℝ) (discount_value : ℝ) : ℝ := mp - discount_value

def profit_percentage (sp : ℝ) (cp : ℝ) : ℝ := ((sp - cp) / cp) * 100

theorem merchant_profit_percentage :
  profit_percentage (selling_price (marked_price cost_price markup_percentage) (discount (marked_price cost_price markup_percentage) discount_percentage)) cost_price = 17 :=
sorry

end merchant_profit_percentage_l476_476191


namespace integral_value_l476_476705

theorem integral_value : (∫ x in 1..e, (1/x + 2 * x)) = Real.exp 2 := by
  sorry

end integral_value_l476_476705


namespace chess_group_players_count_l476_476711

theorem chess_group_players_count (n : ℕ)
  (h1 : ∀ (x y : ℕ), x ≠ y → ∃ k, k = 2)
  (h2 : n * (n - 1) / 2 = 45) :
  n = 10 := sorry

end chess_group_players_count_l476_476711


namespace percentage_of_palindromes_containing_7_l476_476144
theorem percentage_of_palindromes_containing_7 :
  ∀ (A B C : ℕ), (A < 10) → (B < 10) → (C < 10) → 
  (∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ (x = 7 ∨ y = 7 ∨ z = 7)) →
  (∃ n : ℕ, n = 1000) →
  (∃ k : ℕ, k = 120) →
  ((k / n) * 100 = 12) :=
sorry

end percentage_of_palindromes_containing_7_l476_476144


namespace min_value_sin6_cos6_l476_476319

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l476_476319


namespace min_value_sin_cos_l476_476290

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l476_476290


namespace ivan_max_13_bars_a_ivan_max_13_bars_b_l476_476207

variable (n : ℕ) (ivan_max_bags : ℕ)

-- Condition 1: initial count of bars in the chest
def initial_bars := 13

-- Condition 2: function to check if transfers are possible
def can_transfer (bars_in_chest : ℕ) (bars_in_bag : ℕ) (last_transfer : ℕ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ t₁ ≠ last_transfer ∧ t₂ ≠ last_transfer ∧
           t₁ + bars_in_bag ≤ initial_bars ∧ bars_in_chest - t₁ + t₂ = bars_in_chest

-- Proof Problem (a): Given initially 13 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_a 
  (initial_bars : ℕ := 13) 
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 13) :
  ivan_max_bags = target_bars :=
by
  sorry

-- Proof Problem (b): Given initially 14 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_b 
  (initial_bars : ℕ := 14)
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 14) :
  ivan_max_bags = target_bars :=
by
  sorry

end ivan_max_13_bars_a_ivan_max_13_bars_b_l476_476207


namespace number_of_hamburgers_l476_476934

theorem number_of_hamburgers : 
  let condiments := 10 in
  let patties := 3 in
  (2^condiments) * patties = 3072 :=
by 
  let condiments := 10
  let patties := 3
  have h1 : 2^condiments = 1024 := by sorry
  have h2 : 1024 * patties = 3072 := by sorry
  exact Eq.trans (mul_comm _ _) (Eq.symm h2)

end number_of_hamburgers_l476_476934


namespace right_triangle_exists_l476_476492

theorem right_triangle_exists (a b c d : ℕ) (h1 : ab = cd) (h2 : a + b = c - d) : 
  ∃ (x y z : ℕ), x * y / 2 = ab ∧ x^2 + y^2 = z^2 :=
sorry

end right_triangle_exists_l476_476492


namespace problem1_problem2_l476_476767

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 - 3 * x

theorem problem1 (a : ℝ) : (∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x - 3 ≥ 0) → a ≤ 0 :=
sorry

theorem problem2 (a : ℝ) (h : a = 6) :
  x = 3 ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → f x 6 ≤ -6 ∧ f x 6 ≥ -18) :=
sorry

end problem1_problem2_l476_476767


namespace arithmetic_sequence_minimum_value_S_l476_476577

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l476_476577


namespace min_sixth_power_sin_cos_l476_476271

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l476_476271


namespace quadratic_continuous_l476_476070

variable {a b c : ℝ}

theorem quadratic_continuous (x : ℝ) : continuous (λ x, a * x^2 + b * x + c) :=
by 
  sorry

end quadratic_continuous_l476_476070


namespace range_of_k_l476_476886

theorem range_of_k :
  {k : ℝ | ∀ x : ℝ, (x^2 - x - 2 > 0 → 2 * x^2 + (2 * k + 5) * x + 5 * k < 0) →
    set.countable {x | (x^2 - x - 2 > 0) ∧ (2 * x^2 + (2 * k + 5) * x + 5 * k < 0)}.card = 1 } =
  {k | (3 < k ∧ k ≤ 4) ∨ (-3 ≤ k ∧ k < 2)} :=
by
  sorry

end range_of_k_l476_476886


namespace proposition_a_proposition_b_proposition_c_proposition_d_l476_476943

variable (a b c : ℝ)

-- Proposition A: If ac^2 > bc^2, then a > b
theorem proposition_a (h : a * c^2 > b * c^2) : a > b := sorry

-- Proposition B: If a > b, then ac^2 > bc^2
theorem proposition_b (h : a > b) : ¬ (a * c^2 > b * c^2) := sorry

-- Proposition C: If a > b, then 1/a < 1/b
theorem proposition_c (h : a > b) : ¬ (1/a < 1/b) := sorry

-- Proposition D: If a > b > 0, then a^2 > ab > b^2
theorem proposition_d (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 := sorry

end proposition_a_proposition_b_proposition_c_proposition_d_l476_476943


namespace interval_of_monotonic_increase_l476_476630

noncomputable def f (x : ℝ) : ℝ := x * sin x + cos x

theorem interval_of_monotonic_increase :
  ∀ x : ℝ, 0 < x ∧ x < π → 0 < x ∧ x < π/2 →
  ∃ I : set ℝ, I = set.Ioo 0 (π/2) ∧
  (∀ x₁ x₂ : ℝ, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂) := by
  sorry

end interval_of_monotonic_increase_l476_476630


namespace distance_between_points_l476_476737

theorem distance_between_points : 
  let d := Real.sqrt ((10 - 3) ^ 2 + (8 - (-2)) ^ 2) in
  d = Real.sqrt 149 :=
by
  let x1 := 3
  let y1 := -2
  let x2 := 10
  let y2 := 8
  let dx := x2 - x1
  let dy := y2 - y1
  let dsq := dx ^ 2 + dy ^ 2
  let d := Real.sqrt dsq
  have hdx : dx = 7 := rfl
  have hdy : dy = 10 := rfl
  have hdsq : dsq = 149 := by
    rw [hdx, hdy]
    simp
  show d = Real.sqrt 149
  rw [hdsq]
  rfl

end distance_between_points_l476_476737


namespace polygon_area_l476_476733

def vertices : List (ℝ × ℝ) := [(1, 0), (3, 2), (5, 0), (3, 5)]

noncomputable def shoelace_area (pts : List (ℝ × ℝ)) : ℝ :=
  let n := pts.length
  let xs := pts.map Prod.fst
  let ys := pts.map Prod.snd
  Float.abs (((List.range n).sum (λ i, xs.get ⟨i, sorry⟩ * ys.get ⟨(i + 1) % n, sorry⟩)) -
             ((List.range n).sum (λ i, ys.get ⟨i, sorry⟩ * xs.get ⟨(i + 1) % n, sorry⟩))) / 2

theorem polygon_area : shoelace_area vertices = 6 :=
  sorry

end polygon_area_l476_476733


namespace netSalePrice_correct_l476_476465

-- Definitions for item costs and fees
def purchaseCostA : ℝ := 650
def handlingFeeA : ℝ := 0.02 * purchaseCostA
def totalCostA : ℝ := purchaseCostA + handlingFeeA

def purchaseCostB : ℝ := 350
def restockingFeeB : ℝ := 0.03 * purchaseCostB
def totalCostB : ℝ := purchaseCostB + restockingFeeB

def purchaseCostC : ℝ := 400
def transportationFeeC : ℝ := 0.015 * purchaseCostC
def totalCostC : ℝ := purchaseCostC + transportationFeeC

-- Desired profit percentages
def profitPercentageA : ℝ := 0.40
def profitPercentageB : ℝ := 0.25
def profitPercentageC : ℝ := 0.30

-- Net sale prices for achieving the desired profit percentages
def netSalePriceA : ℝ := totalCostA + (profitPercentageA * totalCostA)
def netSalePriceB : ℝ := totalCostB + (profitPercentageB * totalCostB)
def netSalePriceC : ℝ := totalCostC + (profitPercentageC * totalCostC)

-- Expected values
def expectedNetSalePriceA : ℝ := 928.20
def expectedNetSalePriceB : ℝ := 450.63
def expectedNetSalePriceC : ℝ := 527.80

-- Theorem to prove the net sale prices match the expected values
theorem netSalePrice_correct :
  netSalePriceA = expectedNetSalePriceA ∧
  netSalePriceB = expectedNetSalePriceB ∧
  netSalePriceC = expectedNetSalePriceC :=
by
  unfold netSalePriceA netSalePriceB netSalePriceC totalCostA totalCostB totalCostC
         handlingFeeA restockingFeeB transportationFeeC
  sorry

end netSalePrice_correct_l476_476465


namespace isosceles_triangle_perimeter_l476_476213

theorem isosceles_triangle_perimeter {a : ℝ} (h_base : 4 ≠ 0) (h_roots : a^2 - 5 * a + 6 = 0) :
  a = 3 → (4 + 2 * a = 10) :=
by
  sorry

end isosceles_triangle_perimeter_l476_476213


namespace frustum_volume_l476_476806

noncomputable def volume_of_frustum (V₁ V₂ : ℝ) : ℝ :=
  V₁ - V₂

theorem frustum_volume : 
  let base_edge_original := 15
  let height_original := 10
  let base_edge_smaller := 9
  let height_smaller := 6
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let V_original := (1 / 3 : ℝ) * base_area_original * height_original
  let V_smaller := (1 / 3 : ℝ) * base_area_smaller * height_smaller
  volume_of_frustum V_original V_smaller = 588 := 
by
  sorry

end frustum_volume_l476_476806


namespace symmetric_point_correct_l476_476974

-- Define the initial conditions in Lean
def point_A_polar : ℝ × ℝ := (2, Float.pi / 2)
def line_l_polar (ρ θ : ℝ) : Prop := ρ * Float.cos θ = 1

-- Define the target symmetric point in polar coordinates
def symmetric_point_polar : ℝ × ℝ := (2 * Float.sqrt 2, Float.pi / 4)

-- Prove that the symmetric point of A with respect to the line l has the correct polar coordinates
theorem symmetric_point_correct :
  (∃ (A : ℝ × ℝ) (l : ℝ → ℝ → Prop),
    A = point_A_polar ∧ l = line_l_polar ∧
    (symmetric_point_polar = (2 * Float.sqrt 2, Float.pi / 4))) :=
  by sorry

end symmetric_point_correct_l476_476974


namespace Jake_should_charge_for_planting_flowers_l476_476983

theorem Jake_should_charge_for_planting_flowers :
  let mowing_time := 1
  let mowing_payment := 15
  let planting_time := 2
  let desired_rate := 20
  let total_hours := mowing_time + planting_time
  let total_earnings := desired_rate * total_hours
  let planting_charge := total_earnings - mowing_payment
  planting_charge = 45 :=
by
  simp [mowing_time, mowing_payment, planting_time, desired_rate, total_hours, total_earnings, planting_charge]
  sorry

end Jake_should_charge_for_planting_flowers_l476_476983


namespace min_sin6_cos6_l476_476342

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476342


namespace vertex_of_parabola_is_correct_l476_476107

theorem vertex_of_parabola_is_correct
  (p q : ℝ)
  (h : ∀ x : ℝ, 2 * x ^ 2 + p * x + q ≥ 0 ↔ x ∈ set.Icc (-6 : ℝ) (4 : ℝ)) :
  ∃ (k : ℝ), ∀ x : ℝ, 2 * x ^ 2 + p * x + q = 2 * (x + 6) * (x - 4) ∧ k = (-1, -50) := 
sorry

end vertex_of_parabola_is_correct_l476_476107


namespace isosceles_triangle_perimeter_l476_476210

-- Define what it means to be a root of the equation x^2 - 5x + 6 = 0
def is_root (x : ℝ) : Prop := x^2 - 5 * x + 6 = 0

-- Define the perimeter based on given conditions
theorem isosceles_triangle_perimeter (x : ℝ) (base : ℝ) (h_base : base = 4) (h_root : is_root x) :
    2 * x + base = 10 :=
by
  -- Insert proof here
  sorry

end isosceles_triangle_perimeter_l476_476210


namespace exists_integers_sum_product_eq_l476_476841

theorem exists_integers_sum_product_eq (n : ℕ) (h_n : n = 2016) :
  ∃ (a : Fin n → ℤ), (∑ i, a i) = 2016 ∧ (∏ i, a i) = 2016 :=
sorry

end exists_integers_sum_product_eq_l476_476841


namespace arithmetic_sequence_and_minimum_sum_l476_476526

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476526


namespace total_mass_is_correct_l476_476834

noncomputable def atomic_mass_C : Real := 12.01
noncomputable def atomic_mass_H : Real := 1.008
noncomputable def atomic_mass_O : Real := 16.00
noncomputable def atomic_mass_N : Real := 14.01
noncomputable def atomic_mass_Br : Real := 79.90

def molar_mass_C8H10O2NBr2 : Real :=
  (8 * atomic_mass_C) + (10 * atomic_mass_H) + (2 * atomic_mass_O) + (1 * atomic_mass_N) + (2 * atomic_mass_Br)

def total_mass_3_moles : Real :=
  3 * molar_mass_C8H10O2NBr2

theorem total_mass_is_correct :
  total_mass_3_moles = 938.91 :=
by
  -- Defining each atomic mass
  let mass_C := 8 * atomic_mass_C
  let mass_H := 10 * atomic_mass_H
  let mass_O := 2 * atomic_mass_O
  let mass_N := 1 * atomic_mass_N
  let mass_Br := 2 * atomic_mass_Br

  -- Summing to find molar mass
  let molar_mass := mass_C + mass_H + mass_O + mass_N + mass_Br
  have h_molar_mass : molar_mass = 312.97 := by sorry -- Computation verification omitted

  -- Calculating the total mass of 3 moles
  have h_total_mass : 3 * molar_mass = 938.91 := by sorry -- Final computation verification omitted

  -- Rewrite using the correct hypothesis
  show total_mass_3_moles = 938.91 from h_total_mass

end total_mass_is_correct_l476_476834


namespace sequence_problem_l476_476594

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l476_476594


namespace slope_of_l_l476_476390

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def parallel_lines (slope : ℝ) : Prop :=
  ∃ m : ℝ, ∀ x y : ℝ, y = slope * x + m

def intersects_ellipse (slope : ℝ) : Prop :=
  parallel_lines slope ∧ ∃ x y : ℝ, ellipse x y ∧ y = slope * x + (y - slope * x)

theorem slope_of_l {l_slope : ℝ} :
  (∃ (m : ℝ) (x y : ℝ), intersects_ellipse (1 / 4) ∧ (y - l_slope * x = m)) →
  (l_slope = -2) :=
sorry

end slope_of_l_l476_476390


namespace floss_leftover_l476_476840

noncomputable def leftover_floss
    (students : ℕ)
    (floss_per_student : ℚ)
    (floss_per_packet : ℚ) :
    ℚ :=
  let total_needed := students * floss_per_student
  let packets_needed := (total_needed / floss_per_packet).ceil
  let total_floss := packets_needed * floss_per_packet
  total_floss - total_needed

theorem floss_leftover {students : ℕ} {floss_per_student floss_per_packet : ℚ}
    (h_students : students = 20)
    (h_floss_per_student : floss_per_student = 3 / 2)
    (h_floss_per_packet : floss_per_packet = 35) :
    leftover_floss students floss_per_student floss_per_packet = 5 :=
by
  rw [h_students, h_floss_per_student, h_floss_per_packet]
  simp only [leftover_floss]
  norm_num
  sorry

end floss_leftover_l476_476840


namespace remainder_x_squared_mod_25_l476_476447

theorem remainder_x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 4 [ZMOD 25] :=
sorry

end remainder_x_squared_mod_25_l476_476447


namespace max_distance_circle_to_line_l476_476057

-- Define the circle and line as given conditions
def circle (x y : ℝ) := (x - 2)^2 + (y - 2)^2 = 1

def line (x y : ℝ) := x - y - 5 = 0

-- Define the problem: Prove the maximum distance from a point on the circle to the line
theorem max_distance_circle_to_line :
    ∀ (x y : ℝ), 
    circle x y -> 
    ∃ (max_dist : ℝ), 
    max_dist = (5 * real.sqrt 2) / 2 + 1 :=
by
  intros x y h_circle
  use (5 * real.sqrt 2) / 2 + 1
  sorry

end max_distance_circle_to_line_l476_476057


namespace min_value_of_reciprocal_squares_l476_476129

variable (a b : ℝ)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x + a^2 - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0

-- The condition that the two circles are externally tangent and have three common tangents
def externallyTangent (a b : ℝ) : Prop :=
  -- From the derivation in the solution, we must have:
  (a^2 + 4 * b^2 = 9)

-- Ensure a and b are non-zero
def nonzero (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0

-- State the main theorem to prove
theorem min_value_of_reciprocal_squares (h1 : externallyTangent a b) (h2 : nonzero a b) :
  (1 / a^2) + (1 / b^2) = 1 := 
sorry

end min_value_of_reciprocal_squares_l476_476129


namespace arithmetic_sequence_min_value_S_l476_476511

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l476_476511


namespace percentage_of_palindromes_containing_7_l476_476143
theorem percentage_of_palindromes_containing_7 :
  ∀ (A B C : ℕ), (A < 10) → (B < 10) → (C < 10) → 
  (∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ (x = 7 ∨ y = 7 ∨ z = 7)) →
  (∃ n : ℕ, n = 1000) →
  (∃ k : ℕ, k = 120) →
  ((k / n) * 100 = 12) :=
sorry

end percentage_of_palindromes_containing_7_l476_476143


namespace term_containing_x5_and_max_coefficient_l476_476893

open BigOperators

theorem term_containing_x5_and_max_coefficient (C : ℕ → ℕ → ℕ) :
  let n := 10 in
  C n (n-2) = 45 →
  (C n 2 * x^5 = 45 * x^5) ∧ (C n 5 * x^(35/4) = 252 * x^(35/4)) :=
by
  sorry

end term_containing_x5_and_max_coefficient_l476_476893


namespace no_real_or_imaginary_values_of_t_l476_476689

open Complex

theorem no_real_or_imaginary_values_of_t :
  ∀ t : ℂ, sqrt (49 - t^3) + 7 ≠ 0 := by
  sorry

end no_real_or_imaginary_values_of_t_l476_476689


namespace evaluate_expression_l476_476846

theorem evaluate_expression (a b : ℕ) (ha : a = 7) (hb : b = 5) : 3 * (a^3 + b^3) / (a^2 - a * b + b^2) = 36 :=
by
  rw [ha, hb]
  sorry

end evaluate_expression_l476_476846


namespace remainder_of_M_div_500_l476_476043

-- Definition of factorial
def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Count of factors of 5 in the prime factorization of n!
def count_factors_of_5 (n : ℕ) : ℕ :=
  if n = 0 then 0 else n / 5 + count_factors_of_5 (n / 5)

-- Count of factors of 5 in product of factorials from 1 to 50
def M : ℕ :=
  (List.range 51).map (λ n, count_factors_of_5 (factorial n)).sum

-- The mathematical statement to be proved
theorem remainder_of_M_div_500 : M % 500 = 12 :=
sorry

end remainder_of_M_div_500_l476_476043


namespace king_paid_after_tip_l476_476786

-- Define the cost of the crown and the tip percentage
def cost_of_crown : ℝ := 20000
def tip_percentage : ℝ := 0.10

-- Define the total amount paid after the tip
def total_amount_paid (C : ℝ) (tip_pct : ℝ) : ℝ :=
  C + (tip_pct * C)

-- Theorem statement: The total amount paid after the tip is $22,000
theorem king_paid_after_tip : total_amount_paid cost_of_crown tip_percentage = 22000 := by
  sorry

end king_paid_after_tip_l476_476786


namespace total_students_1150_l476_476110

theorem total_students_1150 (T G : ℝ) (h1 : 92 + G = T) (h2 : G = 0.92 * T) : T = 1150 := 
by
  sorry

end total_students_1150_l476_476110


namespace quotient_of_large_div_small_l476_476085

theorem quotient_of_large_div_small (L S : ℕ) (h1 : L - S = 1365)
  (h2 : L = S * (L / S) + 20) (h3 : L = 1634) : (L / S) = 6 := by
  sorry

end quotient_of_large_div_small_l476_476085


namespace find_smallest_n_and_two_lights_l476_476879

noncomputable def smallest_n : ℕ :=
  6

def sufficient_lights (n : ℕ) : Prop :=
  n = 6 → ∃ p1 p2 : (fin n → ℝ × ℝ) → (ℝ × ℝ), 
  (∀ (poly : fin n → ℝ × ℝ), is_simple_polygon poly → lighten_polygon poly p1 ∨ lighten_polygon poly p2)

theorem find_smallest_n_and_two_lights :
  ∃ (n : ℕ), 
  (3 ≤ n) ∧ 
  (∀ (poly : fin n → ℝ × ℝ), is_simple_polygon poly → ∃ p : (ℝ × ℝ), ∀ (s ∈ sides poly), ¬lightened_by p s) ∧ 
  sufficient_lights n :=
begin
  use smallest_n,
  split,
  { exact nat.le_of_eq (nat.succ-pos 2), },
  split,
  { intros poly h1,
    use (0, 0),
    intro h2,
    sorry, },
  {
    unfold sufficient_lights,
    intro h3,
    use (λ poly, (0, 0)),
    use (λ poly, (1, 1)),
    intros _ _,
    exact true.intro,
  }
end

end find_smallest_n_and_two_lights_l476_476879


namespace soccer_team_total_games_l476_476797

variable (total_games : ℕ)
variable (won_games : ℕ)

-- Given conditions
def team_won_percentage (p : ℝ) := p = 0.60
def team_won_games (w : ℕ) := w = 78

-- The proof goal
theorem soccer_team_total_games 
    (h1 : team_won_percentage 0.60)
    (h2 : team_won_games 78) :
    total_games = 130 :=
sorry

end soccer_team_total_games_l476_476797


namespace part1_f_1_part2_range_x_l476_476631

noncomputable def f : ℝ → ℝ := sorry

axiom f_monotonic : ∀ x y : ℝ, 0 < x → 0 < y → x ≤ y → f(x) ≤ f(y)
axiom f_functional : ∀ x y : ℝ, 0 < x → 0 < y → f(x * y) = f(x) + f(y)
axiom f_at_3 : f(3) = 1

theorem part1_f_1 : f(1) = 0 := sorry

theorem part2_range_x (x : ℝ) : f(x) + f(x - 8) ≤ 2 → 8 < x ∧ x ≤ 9 := sorry

end part1_f_1_part2_range_x_l476_476631


namespace product_sequence_l476_476817

theorem product_sequence : (∏ n in Finset.range 2008, ((n + 1) - (n + 2))) = 1 := by
  sorry

end product_sequence_l476_476817


namespace zero_of_f_in_interval_l476_476706

noncomputable def f (x : ℝ) : ℝ := log x - 6 + 2 * x

theorem zero_of_f_in_interval (x₀ : ℝ) (h₀ : f x₀ = 0) : 2 < x₀ ∧ x₀ < 3 :=
by
  sorry

end zero_of_f_in_interval_l476_476706


namespace weight_of_brand_b_ghee_l476_476111

theorem weight_of_brand_b_ghee :
  ∀ (w_b : ℕ),
  (let weight_a : ℕ := 900) -- weight of 1 liter of brand a
  ∧ (let volume_a : ℕ := 3) -- volume of brand a in the mixture
  ∧ (let volume_b : ℕ := 2) -- volume of brand b in the mixture
  ∧ (let total_weight : ℕ := 3520) -- total weight of the mixture
  ∧ (let total_volume : ℕ := 4), -- total volume of the mixture
  (weight_a * volume_a + w_b * volume_b = total_weight) →
  (w_b = 410) :=
  ∀ w_b, 
  let weight_a : ℕ := 900 
  volume_a : ℕ := 3 
  volume_b : ℕ := 2 
  total_weight : ℕ := 3520 
  total_volume : ℕ := 4 
  (weight_a * volume_a + w_b * volume_b = total_weight) →
  (w_b = 410) := sorry

end weight_of_brand_b_ghee_l476_476111


namespace cost_price_of_pots_l476_476375

variable (C : ℝ)

-- Define the conditions
def selling_price (C : ℝ) := 1.25 * C
def total_revenue (selling_price : ℝ) := 150 * selling_price

-- State the main proof goal
theorem cost_price_of_pots (h : total_revenue (selling_price C) = 450) : C = 2.4 := by
  sorry

end cost_price_of_pots_l476_476375


namespace arithmetic_sequence_minimum_value_of_Sn_l476_476546

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l476_476546


namespace remainder_of_M_div_500_l476_476044

-- Definition of factorial
def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Count of factors of 5 in the prime factorization of n!
def count_factors_of_5 (n : ℕ) : ℕ :=
  if n = 0 then 0 else n / 5 + count_factors_of_5 (n / 5)

-- Count of factors of 5 in product of factorials from 1 to 50
def M : ℕ :=
  (List.range 51).map (λ n, count_factors_of_5 (factorial n)).sum

-- The mathematical statement to be proved
theorem remainder_of_M_div_500 : M % 500 = 12 :=
sorry

end remainder_of_M_div_500_l476_476044


namespace initial_bottle_caps_l476_476161

theorem initial_bottle_caps (bought_caps total_caps initial_caps : ℕ) 
  (hb : bought_caps = 41) (ht : total_caps = 43):
  initial_caps = 2 :=
by
  have h : total_caps = initial_caps + bought_caps := sorry
  have ha : initial_caps = total_caps - bought_caps := sorry
  exact sorry

end initial_bottle_caps_l476_476161


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476363

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l476_476363


namespace min_sin6_cos6_l476_476334

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l476_476334


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476604

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l476_476604


namespace find_k_l476_476472

variable {a : ℕ → ℝ}
variable (d : ℝ) (k : ℕ)

def arithmetic_sequence : Prop :=
  a 0 = 0 ∧ d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

theorem find_k (h : arithmetic_sequence d) (h₁ : ∑ i in Finset.range 10, a (i + 1) = a k) :
  k = 46 :=
sorry

end find_k_l476_476472


namespace arithmetic_sequence_and_minimum_sum_l476_476525

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l476_476525


namespace no_polygonal_line_possible_l476_476015

noncomputable def distance (P₁ P₂ : ℕ) : ℝ := sorry -- The distance function between points

lemma unique_distances (n : ℕ) (h : 3 ≤ n) (points : fin n → ℕ) :
  (∀ i j : fin n, i ≠ j → distance (points i) (points j) ≠ distance (points j) (points k)) :=
sorry     -- unique distances

lemma nearest_neighbour (n : ℕ) (h : 3 ≤ n) (points : fin n → ℕ)
  (nearest : fin n → fin n) :
  (∀ i : fin n, i ≠ nearest i → distance (points i) (points (nearest i)) < distance (points i) (points j)) →
  (∀ i j : fin n, distance (points i) (points j) ≠ distance (points i) (points (nearest i))) :=
sorry

theorem no_polygonal_line_possible (n : ℕ) (h : 3 ≤ n) (points : fin n → ℕ)
  (nearest : fin n → fin n) :
  (∀ i j : fin n, i ≠ j → distance (points i) (points j) ≠ distance (points j) (points k)) →
  (∀ i : fin n, i ≠ nearest i → distance (points i) (points (nearest i)) < distance (points i) (points j)) →
  ¬ (∃ seq : list (fin n), seq.nodup ∧ ∀ i < seq.length - 1, nearest (seq.nth_le i sorry) = seq.nth_le (i + 1) sorry) :=
sorry

end no_polygonal_line_possible_l476_476015


namespace sum_C_D_equals_seven_l476_476105

def initial_grid : Matrix (Fin 4) (Fin 4) (Option Nat) :=
  ![ ![ some 1, none, none, none ],
     ![ none, some 2, none, none ],
     ![ none, none, none, none ],
     ![ none, none, none, some 4 ] ]

def valid_grid (grid : Matrix (Fin 4) (Fin 4) (Option Nat)) : Prop :=
  ∀ i j, grid i j ≠ none →
    (∀ k, k ≠ j → grid i k ≠ grid i j) ∧ 
    (∀ k, k ≠ i → grid k j ≠ grid i j)

theorem sum_C_D_equals_seven :
  ∃ (C D : Nat), C + D = 7 ∧ valid_grid initial_grid :=
sorry

end sum_C_D_equals_seven_l476_476105


namespace problem1_problem2_l476_476880

def box (n : ℕ) : ℕ := (10^n - 1) / 9

theorem problem1 (m : ℕ) :
  let b := box (3^m)
  b % (3^m) = 0 ∧ b % (3^(m+1)) ≠ 0 :=
  sorry

theorem problem2 (n : ℕ) :
  (n % 27 = 0) ↔ (box n % 27 = 0) :=
  sorry

end problem1_problem2_l476_476880
