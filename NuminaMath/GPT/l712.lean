import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Field.Real.Basic
import Mathlib.Algebra.OddFunction
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.Logarithm
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.CosSin
import Mathlib.Combinatorics.Derangements.Finite
import Mathlib.Combinatorics.Perm
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Logarithm
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Logic.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution.Normal
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic

namespace range_of_m_l712_712090

-- Define the conditions in Lean 4

def double_point (x y : ℝ) : Prop := y = 2 * x

def quadratic_function (x m : ℝ) : ℝ := x^2 + 2 * m * x - m

noncomputable def M := (x1 : ℝ) (hM : double_point x1 (quadratic_function x1 m)) 
def N := (x2 : ℝ) (hN : double_point x2 (quadratic_function x2 m))
def x1_lt_1_lt_x2 (x1 x2 : ℝ) : Prop := x1 < 1 ∧ 1 < x2

-- Lean 4 theorem statement

theorem range_of_m (x1 x2 m : ℝ) 
  (h_double_point_M : double_point x1 (quadratic_function x1 m))
  (h_double_point_N : double_point x2 (quadratic_function x2 m))
  (h_x1_lt_1_lt_x2 : x1_lt_1_lt_x2 x1 x2) 
: m < 1 := 
sorry

end range_of_m_l712_712090


namespace ratio_of_x_y_l712_712838

theorem ratio_of_x_y (x y : ℚ) (h : (2 * x - y) / (x + y) = 2 / 3) : x / y = 5 / 4 :=
sorry

end ratio_of_x_y_l712_712838


namespace compute_expression_l712_712322

theorem compute_expression :
  23 ^ 12 / 23 ^ 5 + 5 = 148035894 :=
  sorry

end compute_expression_l712_712322


namespace foci_and_vertices_coincide_proposition_true_l712_712787

-- Given: the equation represents an ellipse with foci on the x-axis
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (9 - m) + y^2 / (2 * m) = 1

-- Given: the eccentricity e of the hyperbola is in the interval
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / 5 - y^2 / m = 1

def eccentricity_condition (e : ℝ) : Prop :=
  (sqrt 6) / 2 < e ∧ e < sqrt 2

-- (I) m == 4/3 when the foci of the ellipse and vertices of the hyperbola coincide
theorem foci_and_vertices_coincide (m : ℝ) (h_ellipse : ∀ x y, ellipse_equation x y m) 
  (h_hyperbola : ∀ x y, hyperbola_equation x y m) : m = 4 / 3 :=
sorry

-- (II) The range of values for m when the proposition p ∧ q is true
theorem proposition_true (m : ℝ) 
  (h_ellipse : ∀ x y, ellipse_equation x y m)
  (h_hyperbola : ∀ x y, hyperbola_equation x y m) 
  (h_eccentricity : ∀ e, eccentricity_condition e) : 5 / 2 < m ∧ m < 3 :=
sorry

end foci_and_vertices_coincide_proposition_true_l712_712787


namespace log_f_2_l712_712061

noncomputable def f (a x : ℝ) := a ^ x

def point_condition (a : ℝ) := f a (1 / 2) = (Real.sqrt 2) / 2

theorem log_f_2 (a : ℝ) (h_pos : 0 < a) (h_not_one : a ≠ 1) (h_point : point_condition a) :
  Real.log 2 (f a 2) = -2 :=
sorry

end log_f_2_l712_712061


namespace eccentricity_of_isosceles_right_triangle_l712_712434

theorem eccentricity_of_isosceles_right_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F₁ F₂ : ℝ × ℝ) (A B : ℝ × ℝ)
  (h3 : ∃ x y, A = (x, y) ∧ B = (x, -y) ∧ (x*x)/(a*a) - (y*y)/(b*b) = 1)
  (h4 : EuclideanGeometry.is_isosceles_right (A, B, F₁) (A, F₁, B))
  (e : ℝ) :
  (e^2 = 5 - 2 * real.sqrt 2) :=
sorry

end eccentricity_of_isosceles_right_triangle_l712_712434


namespace age_of_teacher_l712_712580

theorem age_of_teacher (S : ℕ) (T : Real) (n : ℕ) (average_student_age : Real) (new_average_age : Real) : 
  average_student_age = 14 → 
  new_average_age = 14.66 → 
  n = 45 → 
  S = average_student_age * n → 
  T = 44.7 :=
by
  sorry

end age_of_teacher_l712_712580


namespace proof_goal_l712_712773

-- Define the vectors and conditions
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {a1 a2 a3 a4 : V}

-- The sum of the vectors is zero
axiom sum_is_zero : a1 + a2 + a3 + a4 = 0

-- The vectors are pairwise non-parallel
axiom non_parallel : ∀ (i j : ℕ), i ≠ j → 
  let l := [a1, a2, a3, a4] in
  ¬ (∃ k : ℝ, k ≠ 0 ∧ l.i = k • l.j)

-- Declare that we are non-computational 
noncomputable def vectors_form_non_convex_quadrilateral : Prop :=
∃ (p1 p2 p3 p4 : V), 
p1 = a1 ∧ 
p2 = a2 ∧ 
p3 = a3 ∧ 
p4 = a4 ∧ 
(¬ convex {p1, p2, p3, p4})

-- Declare that we are non-computational 
noncomputable def vectors_form_self_intersecting_quadrilateral : Prop :=
∃ (p1 p2 p3 p4 : V), 
p1 = a1 ∧ 
p2 = a2 ∧ 
p3 = a3 ∧ 
p4 = a4 ∧ 
(let l := [p1, p2, p3, p4] in 
  ∃ (i j : ℕ), i ≠ j ∧ segments_intersect l.i l.j)

-- The proof goal
theorem proof_goal :
  vectors_form_non_convex_quadrilateral ∧
  vectors_form_self_intersecting_quadrilateral := 
by {
  -- Proof goes here
  sorry
}

end proof_goal_l712_712773


namespace points_eqidistant_on_perpendicular_bisector_l712_712251

theorem points_eqidistant_on_perpendicular_bisector (A B : Point) (P : Point) :
  (dist P A = dist P B) ↔ P ∈ perpendicular_bisector A B  :=
sorry

end points_eqidistant_on_perpendicular_bisector_l712_712251


namespace collinearity_of_T_K_I_l712_712930

noncomputable def intersection_point (l1 l2 : Line) : Point := sorry

-- Definitions of lines AP, CQ, MP, NQ based on the problem context
variables {A P C Q M N I : Point} (lAP lCQ lMP lNQ : Line)
variables (T : Point) (K : Point)

-- Given conditions
def condition_1 : T = intersection_point lAP lCQ := sorry
def condition_2 : K = intersection_point lMP lNQ := sorry

-- Theorem statement
theorem collinearity_of_T_K_I : T ∈ line_through K I :=
by {
  -- These are the conditions that we're given in the problem
  have hT : T = intersection_point lAP lCQ := sorry,
  have hK : K = intersection_point lMP lNQ := sorry,
  -- Rest of the proof would go here
  sorry
}

end collinearity_of_T_K_I_l712_712930


namespace find_a_l712_712189

theorem find_a : 
  ∃ a : ℝ, 
  (∃ (x y : ℝ), 
    (log a x = y) 
    ∧ (log a (x - 6) = y / 2) 
    ∧ (log a (x - 6) = (y + 6) / 3) 
    ∧ (x - 6)^2 = x 
    ∧ a^6 = 3 
    ∧ x ≠ 0
  ) 
  ∧ a = real.root 6 3 := 
sorry

end find_a_l712_712189


namespace cafeteria_tables_l712_712610

def numberOfTables (seatsPerTable totalSeats: ℕ) : ℕ :=
  totalSeats / seatsPerTable

theorem cafeteria_tables :
  ∃ (tables : ℕ), 
    tables = numberOfTables 10 (135 / (9 / 10 : ℝ) * 10 / 9 * 10) := 
    sorry

end cafeteria_tables_l712_712610


namespace seating_arrangements_l712_712850

theorem seating_arrangements (n : ℕ) (h : n = 7) : (n-1)! = 720 :=
by
  rw [h]
  calc (7 - 1)! = 6!      : rfl
             ... = 720    : by norm_num

end seating_arrangements_l712_712850


namespace num_six_digit_numbers_with_P3_eq_a_l712_712280

def isCyclicPermutation (a b : Nat) : Prop :=
  let digits (x : Nat) := List.toArray (Nat.digits 10 x)
  let n := digits a |>.size
  ∃ (i : Nat), i < n ∧ digits b = Array.append (digits a |>.drop i) (digits a |>.take i)

def P (a : Nat) : Nat :=
  let digits := Nat.digits 10 a
  let n := digits.size
  if n = 0 then 0 else
    let last := digits.ilast!
    Nat.ofDigits 10 (Array.push (Array.dropLast digits) last)

def P3_eq (a : Nat) : Bool :=
  P (P (P a)) = a

theorem num_six_digit_numbers_with_P3_eq_a :
  { a // 100000 ≤ a ∧ a < 1000000 ∧ P3_eq a }.card = 729 := by
  sorry

end num_six_digit_numbers_with_P3_eq_a_l712_712280


namespace solve_quadratic_eq_l712_712567

theorem solve_quadratic_eq (x : ℝ) : (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2) := by
  sorry

end solve_quadratic_eq_l712_712567


namespace compositions_produce_14_distinct_sets_l712_712377

-- Defining the set A according to given conditions
noncomputable def A : Set ℚ := ({x | x ∈ Set.Icc 1 2 ∧ x ∈ ℚ} ∪ Set.Ico 2 3 ∪ Set.Icc 3 4 ∪ {5})

-- Functions f and g should be explicitly defined or treated as variables representing valid set functions
variable (f g : Set ℚ → Set ℚ)

-- The theorem statement should reflect that the compositions result in exactly 14 distinct sets
theorem compositions_produce_14_distinct_sets : 
  ∃ (A : Set ℚ), (∀ (f g : Set ℚ → Set ℚ), -- here we imply the existence of A and definitions of the functions
    let sets := [f A, g (f (g A)), f (g (f (g A))), g (f (g f g A)), g (f g) ∩ A, f (g f g f A), f A ∪ {5}, 
                 g (f A), f (g A), f g A, g f g A, (g f g) ∩ {5}, g (f g (A ∪ {5})), g (A ∪ {5})] in
    set.card sets.to_finset = 14
  ) :=
sorry -- The proof is omitted

end compositions_produce_14_distinct_sets_l712_712377


namespace projection_a_onto_e_l712_712078

def proj_vector (a e : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * e.1 + a.2 * e.2
  let e_magnitude_squared := e.1 ^ 2 + e.2 ^ 2
  let scalar := dot_product / e_magnitude_squared
  (scalar * e.1, scalar * e.2)

theorem projection_a_onto_e :
  let a := (1 : ℝ, Real.sqrt 3)
  let e := (1 / 2 : ℝ, -Real.sqrt 3 / 2)
  proj_vector a e = (-1 / 2, Real.sqrt 3 / 2) :=
by
  sorry

end projection_a_onto_e_l712_712078


namespace negation_of_p_l712_712070

open Real

theorem negation_of_p (p : Prop) : 
  (∃ x : ℝ, x > sin x) → (∀ x : ℝ, x ≤ sin x) :=
sorry

end negation_of_p_l712_712070


namespace smallest_integer_l712_712336

/-- The smallest integer m such that m > 1 and m has a remainder of 1 when divided by any of 5, 7, and 3 is 106. -/
theorem smallest_integer (m : ℕ) : m > 1 ∧ m % 5 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ↔ m = 106 :=
by
    sorry

end smallest_integer_l712_712336


namespace meat_spent_fraction_l712_712717

-- Given conditions
variable (M : ℝ) -- Total money John had
variable (spentFruitsVegetables : ℝ) -- Money spent on fresh fruits and vegetables
variable (spentBakery : ℝ) -- Money spent on bakery products
variable (spentCandy : ℝ) -- Money spent on candy
variable (spentMeat : ℝ) -- Money spent on meat products

-- Conditions based on problem statement
axiom (h1 : M = 120)
axiom (h2 : spentFruitsVegetables = (1/2) * M)
axiom (h3 : spentBakery = (1/10) * M)
axiom (h4 : spentCandy = 8)
axiom (h5 : M - spentFruitsVegetables - spentBakery - spentMeat - spentCandy = 0)

-- Proof that John spent 1/3 of his money on meat products
theorem meat_spent_fraction : spentMeat / M = 1 / 3 :=
by
  sorry

end meat_spent_fraction_l712_712717


namespace carlos_wins_probability_l712_712301

theorem carlos_wins_probability :
  let p_turns := (1 / 2) ^ 4 in
  let sum_geom_series := p_turns / (1 - p_turns) in
  sum_geom_series = 1 / 15 :=
by
  sorry

end carlos_wins_probability_l712_712301


namespace sin_cos_product_l712_712665

theorem sin_cos_product :
  sin (10 * real.pi / 180) * cos (20 * real.pi / 180) * cos (40 * real.pi / 180) = 1 / 8 :=
by
  sorry

end sin_cos_product_l712_712665


namespace solve_eq_l712_712971

theorem solve_eq (x : ℝ) : x^6 - 19*x^3 = 216 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end solve_eq_l712_712971


namespace max_good_family_size_l712_712266

variable (n : ℕ) (E : Finset ℕ) (A : Finset (Finset ℕ))

def is_good_family (A : Finset (Finset ℕ)) : Prop :=
  ∀ ⦃i j⦄, i < j → 
  (A.toList.get? i, A.toList.get? j) ∈ 
  [ (λ Ai Aj, (Ai ∩ Aj = ∅)),
    (λ Ai Aj, ((E \ Ai) ∩ Aj = ∅)),
    (λ Ai Aj, (Ai ∩ (E \ Aj) = ∅)),
    (λ Ai Aj, ((E \ Ai) ∩ (E \ Aj) = ∅)) ]

theorem max_good_family_size (hE : E = Finset.range (n+1)) (hne : 2 ≤ n) : 
  ∃ (A : Finset (Finset ℕ)), is_good_family n E A ∧ A.card = 2 * n - 3 := 
sorry

end max_good_family_size_l712_712266


namespace min_period_f_min_value_f_range_g_l712_712064

noncomputable def f (x : ℝ) := (1/2) * sin (2 * x) - (sqrt 3) * (cos x)^2

theorem min_period_f :
  ∃ T : ℝ, (T = π) ∧ (∀ x : ℝ, f (x + T) = f x) :=
sorry

theorem min_value_f :
  ∃ m : ℝ, (m = (-2 - sqrt 3) / 2) ∧ (∀ x : ℝ, f x ≥ m) :=
sorry

def g (x : ℝ) := sin (x - π / 3) - (sqrt 3) / 2

theorem range_g : 
  ∀ (x : ℝ), (x ∈ Icc (π / 2) π) → (g x ∈ Icc ((1 - sqrt 3) / 2) ((2 - sqrt 3) / 2)) :=
sorry

end min_period_f_min_value_f_range_g_l712_712064


namespace eval_f_four_times_l712_712360

noncomputable def f (z : Complex) : Complex := 
if z.im ≠ 0 then z * z else -(z * z)

theorem eval_f_four_times : 
  f (f (f (f (Complex.mk 2 1)))) = Complex.mk 164833 354192 := 
by 
  sorry

end eval_f_four_times_l712_712360


namespace find_a_l712_712082

theorem find_a (a b : ℝ) (h : (a + b) ^ 2 + real.sqrt (2 * b - 4) = 0) : a = -2 := 
by {
  sorry
}

end find_a_l712_712082


namespace part1_part2_l712_712405

theorem part1 (λ : ℝ) (hλ : λ ≠ 0) (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn : ∀ n, S n = 2 * a n - λ)
  (hSsum : ∀ n, S n = ∑ i in finset.range n, a (i + 1)) :
  ∀ n, a n = λ * 2^(n - 1) :=
sorry

theorem part2 (n : ℕ) (b : ℕ → ℝ) (a : ℕ → ℝ)
  (h_a1 : a 1 = 1)
  (h_an : ∀ n, a n = 2^(n - 1))
  (h_bn : ∀ n, b n = 2 * a n + (-1)^n * real.log (a n) / real.log 2) :
  (finset.range (2 * n)).sum b = 2^(2 * n + 1) - 2 + n :=
sorry

end part1_part2_l712_712405


namespace invest_in_yourself_examples_l712_712374

theorem invest_in_yourself_examples (example1 example2 example3 : String)
  (benefit1 benefit2 benefit3 : String)
  (h1 : example1 = "Investment in Education")
  (h2 : benefit1 = "Spending money on education improves knowledge and skills, leading to better job opportunities and higher salaries. Education appreciates over time, providing financial stability.")
  (h3 : example2 = "Investment in Physical Health")
  (h4 : benefit2 = "Spending on sports activities, fitness programs, or healthcare prevents chronic diseases, saves future medical expenses, and enhances overall well-being.")
  (h5 : example3 = "Time Spent on Reading Books")
  (h6 : benefit3 = "Reading books expands knowledge, improves vocabulary and cognitive abilities, develops critical thinking and analytical skills, and fosters creativity and empathy."):
  "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." = "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." :=
by
  sorry

end invest_in_yourself_examples_l712_712374


namespace collinear_TKI_l712_712889

-- Definitions for points and lines
variables {A P C Q M N I T K : Type}
variable {line : Type → Type}
variables (AP : line A → line P) (CQ : line C → line Q) (MP : line M → line P) (NQ : line N → line Q)

-- Conditions from the problem
-- Assume there exist points T and K which are intersections of the specified lines
axiom intersects_AP_CQ : ∃ (T : Type), AP T = CQ T
axiom intersects_MP_NQ : ∃ (K : Type), MP K = NQ K

-- Collinearity of points T, K, and I
theorem collinear_TKI : ∀ (I : Type) (T : Type) (K : Type),
  intersects_AP_CQ → intersects_MP_NQ → collinear I T K :=
by sorry

end collinear_TKI_l712_712889


namespace exchange_process_cannot_continue_for_more_than_hour_l712_712669

-- Define the alternating sequence
inductive Gender
| girl
| boy

open Gender

def alternating_sequence : List Gender := 
  List.repeat girl 10 ++ List.repeat boy 10

-- Define the exchange process as allowed swaps between pairs
def can_swap (a b : Gender) : Prop := a = girl ∧ b = boy

-- Define the main property to be proved
theorem exchange_process_cannot_continue_for_more_than_hour :
  ∀ (sequence : List Gender), sequence = alternating_sequence →
  (∀ n, n ≠ 60 →
  (∃ m < 60, sequence = List.updateNth (2 * m) boy (List.updateNth (2 * m + 1) girl sequence))) →
  False := 
by
  intros sequence hseq h.
  sorry

end exchange_process_cannot_continue_for_more_than_hour_l712_712669


namespace range_of_f_l712_712637

-- Define the function f
def f (x : ℝ) : ℝ := 1 / x^3

-- Define the range of f
def range_f : Set ℝ := {y | ∃ x : ℝ, x ≠ 0 ∧ f x = y}

-- State the theorem to be proven
theorem range_of_f : range_f = {y : ℝ | y ≠ 0} := sorry

end range_of_f_l712_712637


namespace ellipse_C_properties_l712_712395

open Real

noncomputable def ellipse_eq (b : ℝ) : Prop :=
  (∀ (x y : ℝ), (x = 1 ∧ y = sqrt 3 / 2) → (x^2 / 4 + y^2 / b^2 = 1))

theorem ellipse_C_properties : 
  (∀ (C : ℝ → ℝ → Prop), 
    (C 0 0) ∧ 
    (∀ x y, C x y → (x = 0 ↔ y = 0)) ∧ 
    (∀ x, C x 0) ∧ 
    (∃ x y, C x y ∧ x = 1 ∧ y = sqrt 3 / 2) →
    (∃ b, b > 0 ∧ b^2 = 1 ∧ ellipse_eq b)) ∧
  (∀ P A B : ℝ × ℝ, 
    (P.1 = P.1 ∧ P.1 ≠ 0 ∧ P.2 = 0 ∧ -2 ≤ P.1 ∧ P.1 ≤ 2) →
    (A.2 = 1/2 * (A.1 - P.1) ∧ B.2 = 1/2 * (B.1 - P.1)) →
    ((P.1 - A.1)^2 + A.2^2 + (P.1 - B.1)^2 + B.2^2 = 5)) :=
by sorry

end ellipse_C_properties_l712_712395


namespace calculate_EX_length_l712_712844

-- Define the given conditions
def is_regular_pentagon (ABCDE : ℕ → ℝ × ℝ) : Prop :=
  ∀ i, dist (ABCDE i) (ABCDE ((i + 1) % 5)) = 1

def extended_point (A B X : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ X = B + k * (B - A) ∧ dist A X = 4 * dist A B

-- Define the main theorem
theorem calculate_EX_length (A B C D E X : ℝ × ℝ)
  (ABCDE : ℕ → ℝ × ℝ)
  (h1 : is_regular_pentagon ABCDE)
  (h2 : ABCDE 0 = A ∧ ABCDE 1 = B ∧ ABCDE 2 = C ∧ ABCDE 3 = D ∧ ABCDE 4 = E)
  (h3 : extended_point A B X) :
  dist E X = √68.7255 :=
sorry

end calculate_EX_length_l712_712844


namespace triangle_area_l712_712278

theorem triangle_area {α β γ : ℝ} {A B C : Type} (hcircle : ∃ R = 3, ∀ M M1 M2 : (A → B → C), circle R M M1 M2)
  (hA_angle : α = 60) (hB_angle : β = 45) : 
  ∃ S = 9 * (3 + real.sqrt 3), area_of_triangle A B C = S := 
sorry

end triangle_area_l712_712278


namespace sequence_xn_property_l712_712603

theorem sequence_xn_property :
  (∀ n, x (n + 3) = 2 * x (n + 2) + x (n + 1) - 2 * x n) → 
  x 0 = 0 →
  x 2 = 1 →
  x 100 = (4^50 - 1) / 3 :=
by
  intro h_recurrence h_x0 h_x2
  sorry

end sequence_xn_property_l712_712603


namespace jakes_class_boys_count_l712_712106

theorem jakes_class_boys_count 
    (ratio_girls_boys : ℕ → ℕ → Prop)
    (students_total : ℕ)
    (ratio_condition : ratio_girls_boys 3 4)
    (total_condition : students_total = 35) :
    ∃ boys : ℕ, boys = 20 :=
by
  sorry

end jakes_class_boys_count_l712_712106


namespace min_vertical_segment_length_l712_712204

noncomputable def vertical_segment_length (x : ℝ) : ℝ :=
  abs (|x| - (-x^2 - 4*x - 3))

theorem min_vertical_segment_length :
  ∃ x : ℝ, vertical_segment_length x = 3 / 4 :=
by
  sorry

end min_vertical_segment_length_l712_712204


namespace find_a61_l712_712394

def seq (a : ℕ → ℕ) : Prop :=
  (∀ n, a (2 * n + 1) = a n + a (n + 1)) ∧
  (∀ n, a (2 * n) = a n) ∧
  a 1 = 1

theorem find_a61 (a : ℕ → ℕ) (h : seq a) : a 61 = 9 :=
by
  sorry

end find_a61_l712_712394


namespace sum_of_angles_l712_712035

noncomputable theory

-- Lean statement for our transformed mathematical proof problem
theorem sum_of_angles (α β : ℝ) 
  (h1 : sin (π / 4 - α) = -sqrt 5 / 5)
  (h2 : sin (π / 4 + β) = 3 * sqrt 10 / 10)
  (h3 : 0 < α ∧ α < π / 2)
  (h4 : π / 4 < β ∧ β < π / 2) :
  α + β = 3 * π / 4 :=
sorry

end sum_of_angles_l712_712035


namespace ordered_triples_count_l712_712081
open Nat

def count_ordered_triples : ℕ :=
  let all_triples := [(x, y, z) | x y z : ℕ, x < y ∧ y < z ∧ lcm x y = 210 ∧ lcm x z = 2520 ∧ lcm y z = 630]
  all_triples.length

theorem ordered_triples_count : count_ordered_triples = 8 :=
  sorry

end ordered_triples_count_l712_712081


namespace smallest_sum_of_squares_l712_712990

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l712_712990


namespace fraction_of_menu_items_i_can_eat_l712_712310

def total_dishes (vegan_dishes non_vegan_dishes : ℕ) : ℕ := vegan_dishes + non_vegan_dishes

def vegan_dishes_without_soy (vegan_dishes vegan_with_soy : ℕ) : ℕ := vegan_dishes - vegan_with_soy

theorem fraction_of_menu_items_i_can_eat (vegan_dishes non_vegan_dishes vegan_with_soy : ℕ)
  (h_vegan_dishes : vegan_dishes = 6)
  (h_menu_total : vegan_dishes = (total_dishes vegan_dishes non_vegan_dishes) / 3)
  (h_vegan_with_soy : vegan_with_soy = 4)
  : (vegan_dishes_without_soy vegan_dishes vegan_with_soy) / (total_dishes vegan_dishes non_vegan_dishes) = 1 / 9 :=
by
  sorry

end fraction_of_menu_items_i_can_eat_l712_712310


namespace max_a4_l712_712093

variable {a_n : ℕ → ℝ}

-- Assume a_n is a positive geometric sequence
def is_geometric_seq (a_n : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a_n (n + 1) = a_n n * r

-- Given conditions
def condition1 (a_n : ℕ → ℝ) : Prop := is_geometric_seq a_n
def condition2 (a_n : ℕ → ℝ) : Prop := a_n 3 + a_n 5 = 4

theorem max_a4 (a_n : ℕ → ℝ) (h1 : condition1 a_n) (h2 : condition2 a_n) :
    ∃ max_a4 : ℝ, max_a4 = 2 :=
  sorry

end max_a4_l712_712093


namespace expected_rolls_in_nonleap_year_l712_712710

def probability_stop := (7 : ℚ) / 8
def probability_reroll := (1 : ℚ) / 8
def E : ℚ := (1 + E) * probability_reroll + 1 * probability_stop

theorem expected_rolls_in_nonleap_year : E * 365 = 417.14 := by
  sorry

end expected_rolls_in_nonleap_year_l712_712710


namespace max_f_min_f_l712_712366

noncomputable def f (m n : ℕ) (a : ℕ → ℕ → ℝ) : ℝ :=
  let sum_x := ∑ i in finset.range n, (∑ j in finset.range m, a i j)^2
  let sum_y := ∑ j in finset.range m, (∑ i in finset.range n, a i j)^2
  let f_num := n * sum_x + m * sum_y
  let sum_a := (∑ i in finset.range n, ∑ j in finset.range m, a i j)
  let sum_aa := ∑ i in finset.range n, ∑ j in finset.range m, (a i j)^2
  let f_den := sum_a^2 + m * n * sum_aa
  f_num / f_den

theorem max_f (m n : ℕ) (a : ℕ → ℕ → ℝ) (h₀ : 1 < m) (h₁ : 1 < n) (h₂ : ∀ i j, 0 ≤ a i j) (h₃ : ∃ i j, 0 < a i j) :
  f m n a ≤ 1 := sorry

theorem min_f (m n : ℕ) (a : ℕ → ℕ → ℝ) (h₀ : 1 < m) (h₁ : 1 < n) (h₂ : ∀ i j, 0 ≤ a i j) (h₃ : ∃ i j, 0 < a i j) :
  f m n a ≥ (m + n) / (m * n + n) := sorry

end max_f_min_f_l712_712366


namespace shifted_quadratic_function_l712_712588

theorem shifted_quadratic_function :
  ∀ (x : ℝ), (2 * x ^ 2 + 2 → 2 * (x + 3) ^ 2 + 1) :=
by
  sorry

end shifted_quadratic_function_l712_712588


namespace no_division_result_of_two_l712_712709

theorem no_division_result_of_two :
  ¬ ∃ (A B : ℕ), (A + B = 88 ∧ A = 2 * B) ∧
  (∃ digits : list ℕ, digits = [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9] ∧
    ∃ (l1 l2 : list ℕ),
      l1 ++ l2 = digits ∧
      A = list.digits_to_nat l1 ∧
      B = list.digits_to_nat l2) := sorry

end no_division_result_of_two_l712_712709


namespace train_passing_time_l712_712120

noncomputable def kmph_to_mps (kmph: ℝ) : ℝ :=
  (kmph * 1000) / 3600

theorem train_passing_time :
  let d := 75 -- distance in meters
  let s_kmph := 50 -- speed in kmph
  let s := kmph_to_mps s_kmph -- speed in meters per second
  let t := d / s -- time in seconds
  t ≈ 5.4 :=
by
  sorry

end train_passing_time_l712_712120


namespace number_of_routes_from_A_to_B_l712_712323

-- Define the grid dimensions
def grid_rows : ℕ := 3
def grid_columns : ℕ := 2

-- Define the total number of steps needed to travel from A to B
def total_steps : ℕ := grid_rows + grid_columns

-- Define the number of right moves (R) and down moves (D)
def right_moves : ℕ := grid_rows
def down_moves : ℕ := grid_columns

-- Calculate the number of different routes using combination formula
def number_of_routes : ℕ := Nat.choose total_steps right_moves

-- The main statement to be proven
theorem number_of_routes_from_A_to_B : number_of_routes = 10 :=
by sorry

end number_of_routes_from_A_to_B_l712_712323


namespace triangle_angle_bisectors_eq_sides_l712_712616

theorem triangle_angle_bisectors_eq_sides {A B C F E : Point} 
  (h₁ : triangle A B C)
  (h₂ : angle_bisector A B C F)
  (h₃ : angle_bisector B A C E) : 
  distance B C = distance A C :=
sorry

end triangle_angle_bisectors_eq_sides_l712_712616


namespace acute_angles_relation_l712_712058

theorem acute_angles_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.sin α = (1 / 2) * Real.sin (α + β)) : α < β :=
sorry

end acute_angles_relation_l712_712058


namespace proof_problem_l712_712159

theorem proof_problem
  (n : ℕ)
  (h : n = 16^3018) :
  n / 8 = 2^9032 := by
  sorry

end proof_problem_l712_712159


namespace fish_weight_l712_712355

theorem fish_weight (θ H T : ℝ) (h1 : θ = 4) (h2 : H = θ + 0.5 * T) (h3 : T = H + θ) : H + T + θ = 32 :=
by
  sorry

end fish_weight_l712_712355


namespace problem1_dot_product_problem2_dot_product_problem2_projection_l712_712269

-- Problem (1)
variables {R : Type*} [LinearOrderedField R]
variables (e1 e2 : EuclideanSpace R (Fin 2)) (a b : EuclideanSpace R (Fin 2))

-- Assuming e1 and e2 are unit vectors
hypothesis (h1 : ‖e1‖ = 1)
hypothesis (h2 : ‖e2‖ = 1)
-- Angle between e1 and e2 is 60 degrees -> cosine is 1/2
hypothesis (h3 : InnerProductSpace.inner e1 e2 = 1/2)

-- Definitions for a and b
def a := (3 : R) • e1 - (2 : R) • e2
def b := (2 : R) • e1 - (3 : R) • e2

-- Proof statement for dot product a · b
theorem problem1_dot_product : InnerProductSpace.inner a b = (11 / 2 : R) := 
  sorry

-- Problem (2)
-- Definitions for vectors a and b
def a' := ((3 : R), (4 : R))
def b' := ((2 : R), (-1 : R))

-- Proof statement for dot product a' · b'
theorem problem2_dot_product : a'.1 * b'.1 + a'.2 * b'.2 = (2 : R) := 
  sorry

-- Proof statement for the projection of a' on b'
theorem problem2_projection : (a'.1 * b'.1 + a'.2 * b'.2) / sqrt (b'.1^2 + b'.2^2) = (2 * sqrt(5) / 5 : R) :=
  sorry

end problem1_dot_product_problem2_dot_product_problem2_projection_l712_712269


namespace prime_sequence_constant_l712_712330

open Nat

-- Define a predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the recurrence relation
def recurrence_relation (p : ℕ → ℕ) (k : ℤ) : Prop :=
  ∀ n : ℕ, p (n + 2) = p (n + 1) + p n + k

-- Define the proof problem
theorem prime_sequence_constant (p : ℕ → ℕ) (k : ℤ) : 
  (∀ n, is_prime (p n)) →
  recurrence_relation p k →
  ∃ (q : ℕ), is_prime q ∧ (∀ n, p n = q) ∧ k = -q :=
by
  -- Sorry proof here
  sorry

end prime_sequence_constant_l712_712330


namespace perpendicular_planes_x_value_l712_712459

theorem perpendicular_planes_x_value (x : ℝ) : let a : ℝ × ℝ × ℝ := (-1, 2, 4)
                                                let b : ℝ × ℝ × ℝ := (x, -1, -2)
                                                (a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) →
                                                x = -10 :=
begin
  assume h,
  sorry
end

end perpendicular_planes_x_value_l712_712459


namespace three_digit_numbers_excluding_adjacent_same_digits_is_correct_l712_712818

def num_valid_three_digit_numbers_exclude_adjacent_same_digits : Nat :=
  let total_numbers := 900
  let excluded_numbers_AAB := 81
  let excluded_numbers_BAA := 81
  total_numbers - (excluded_numbers_AAB + excluded_numbers_BAA)

theorem three_digit_numbers_excluding_adjacent_same_digits_is_correct :
  num_valid_three_digit_numbers_exclude_adjacent_same_digits = 738 := by
  sorry

end three_digit_numbers_excluding_adjacent_same_digits_is_correct_l712_712818


namespace fixed_point_condition_l712_712396

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
a > b ∧ b > 0 ∧ (a^2 = 4) ∧ (b^2 = 3) ∧ ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)

theorem fixed_point_condition (k m : ℝ) :
(∀ (x y : ℝ), x ∈ {p : ℝ × ℝ | p.2 = k * p.1 + m} 
→ (x^2 / 4) + (y^2 / 3) = 1) ∧
(∃ A B : ℝ × ℝ, A ≠ B ∧ A ≠ (2, 0) ∧ B ≠ (2, 0) ∧
(ax = A.1 ∧ ay = A.2 
∧ bx = B.1 ∧ by = B.2)
∧ ((1 + k^2) * (ax * bx) - 2 * (ax + bx) + m^2 + 4 = 0)) 
→ ∃ C : ℝ × ℝ, C = (2/7 , 0) := sorry

end fixed_point_condition_l712_712396


namespace selection_count_l712_712019

noncomputable def choose (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_count :
  let boys := 4
  let girls := 3
  let total := boys + girls
  let choose_boys_girls : ℕ := (choose 4 2) * (choose 3 1) + (choose 4 1) * (choose 3 2)
  choose_boys_girls = 30 := 
by
  sorry

end selection_count_l712_712019


namespace parallelogram_condition_l712_712028

theorem parallelogram_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1) →
    ∃ Q R S : ℝ × ℝ, Q ≠ P ∧ R ≠ P ∧ S ≠ P ∧
    (Q.1^2 + Q.2^3 = 1) ∧ (R.1^2 / a^2 + R.2^2 / b^2 = 1) ∧ (S.1^2 / a^2 + S.2^2 / b^2 = 1) ∧
    collinear [P, Q, R] ∧ collinear [Q, R, S]) ↔ (1 / a^2 + 1 / b^2 = 1) :=
sorry

end parallelogram_condition_l712_712028


namespace dot_product_u_v_l712_712721

-- Define the vectors
def u : ℝ × ℝ × ℝ × ℝ := (4, -3, 5, -2)
def v : ℝ × ℝ × ℝ × ℝ := (-6, 3, -1, 4)

-- Define the dot product calculation for 4-dimensional vectors
def dot_product (a b : ℝ × ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 + a.4 * b.4

-- Formulate the proof problem: the dot product of u and v equals -46
theorem dot_product_u_v : dot_product u v = -46 :=
by
  -- Proof goes here
  sorry

end dot_product_u_v_l712_712721


namespace actual_distance_l712_712088

variable {t : ℝ}

-- Conditions
def distance_12kmph := 12 * t
def distance_20kmph := 20 * t
def condition := distance_20kmph = distance_12kmph + 30

-- Question
theorem actual_distance (h : condition) : distance_12kmph = 45 := by
  sorry

end actual_distance_l712_712088


namespace total_books_calculation_l712_712683

-- Definitions and conditions
variable (T : ℝ) -- Total number of books
variable (children_books : ℝ) := 0.35 * T -- 35% of the books are for children
variable (adult_books : ℝ) := 104 -- There are 104 books for adults

-- The proof problem: Prove that T = 160 given the conditions.
theorem total_books_calculation (h : 0.65 * T = 104) : T = 160 := 
by 
  -- Need to prove T = 160
  sorry

end total_books_calculation_l712_712683


namespace length_of_DB_l712_712491

-- Define the data and the conditions
variable (A B C D : Point)
variable (AB AC BC DB AD : ℝ)

-- Given conditions:
variable (h1 : (triangle A B C).is_right_angled A)
variable (h2 : AB = 45)
variable (h3 : AC = 60)
variable (h4 : AD ⊥ (BC))
variable (h5 : A ∈ segment BC)
variable (h6 : B ∈ segment AC)
variable (h7 : C ∈ segment AB)
variable (h8 : D ∈ segment BC)

-- The theorem to prove:
theorem length_of_DB : DB = 48 :=
sorry

end length_of_DB_l712_712491


namespace invest_in_yourself_examples_l712_712373

theorem invest_in_yourself_examples (example1 example2 example3 : String)
  (benefit1 benefit2 benefit3 : String)
  (h1 : example1 = "Investment in Education")
  (h2 : benefit1 = "Spending money on education improves knowledge and skills, leading to better job opportunities and higher salaries. Education appreciates over time, providing financial stability.")
  (h3 : example2 = "Investment in Physical Health")
  (h4 : benefit2 = "Spending on sports activities, fitness programs, or healthcare prevents chronic diseases, saves future medical expenses, and enhances overall well-being.")
  (h5 : example3 = "Time Spent on Reading Books")
  (h6 : benefit3 = "Reading books expands knowledge, improves vocabulary and cognitive abilities, develops critical thinking and analytical skills, and fosters creativity and empathy."):
  "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." = "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." :=
by
  sorry

end invest_in_yourself_examples_l712_712373


namespace math_problem_l712_712515
noncomputable! def f (A : Finset ℕ) (N : Finset ℕ) : ℕ :=
  (A.sum id) * ((N \ A).sum id)

noncomputable! def expected_f (N : Finset ℕ) : ℕ :=
  let k := N.card / 2
  1009 * 1009 * 2018 * k * (k - 1) / (2018 * (2018 - 1) / 2)

theorem math_problem :
  let N := Finset.range (2018 + 1) \ {0}
  ∑ (p : ℕ) in (Nat.factors (expected_f N)).toFinset, p % 1000 = 441 :=
by
  -- N and A definitions
  let N := Finset.range (2018 + 1) \ {0}
  let A := N.subset (by decide)
  -- Expected value computations
  let EF := expected_f N

  -- Sum and factors computations
  have SDF := ∑ (p : ℕ) in (Nat.factors EF).toFinset, p 
  have Hmod : SDF % 1000 = 441 := sorry

  exact Hmod

end math_problem_l712_712515


namespace renata_charity_draw_win_l712_712172

def problem_statement : Prop :=
  ∃ (charity_win : ℤ),
  let initial_amount := 10 in
  let donation := 4 in
  let water_bottle := 1 in
  let lottery_ticket := 1 in
  let lottery_win := 65 in
  let final_amount := 94 in
  let amount_after_donation := initial_amount - donation in
  let amount_after_water := amount_after_donation - water_bottle in
  let amount_after_ticket := amount_after_water - lottery_ticket in
  let amount_after_lottery_win := amount_after_ticket + lottery_win in
  let before_casino := final_amount - amount_after_lottery_win in
  before_casino - amount_after_donation = charity_win ∧ charity_win = 19

theorem renata_charity_draw_win : problem_statement :=
by
  sorry

end renata_charity_draw_win_l712_712172


namespace max_m_n_sq_l712_712737

theorem max_m_n_sq (m n : ℕ) (hm : 1 ≤ m ∧ m ≤ 1981) (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m * n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_n_sq_l712_712737


namespace find_kn_l712_712735

theorem find_kn (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 :=
by
  sorry

end find_kn_l712_712735


namespace distance_A_B_l712_712236

-- Defining the problem conditions
variables (S x y : ℝ)
axiom A_walks_halfway_B_doubles_speed : 
  (S / x = 2 * ((S / 2 / x) + ((S - 1200 - S / 2) / (2 * y))) : Prop)

axiom meet_point_from_B_1200 : 
  (1200 < S / 2)

axiom both_reach_dest_simultaneously : 
  (((S / 2 - 1200) / x) = ((1200 - S / 2) / (2 * y)))

-- Defining the main theorem to prove
theorem distance_A_B (h1 : A_walks_halfway_B_doubles_speed S x y) 
                     (h2 : meet_point_from_B_1200 S)
                     (h3 : both_reach_dest_simultaneously S x y) : 
  S = 2800 :=
by 
  sorry

end distance_A_B_l712_712236


namespace total_earthworms_in_box_l712_712954

-- Definitions of the conditions
def applesPaidByOkeydokey := 5
def applesPaidByArtichokey := 7
def earthwormsReceivedByOkeydokey := 25
def ratio := earthwormsReceivedByOkeydokey / applesPaidByOkeydokey -- which should be 5

-- Theorem statement proving the total number of earthworms in the box
theorem total_earthworms_in_box :
  (applesPaidByOkeydokey + applesPaidByArtichokey) * ratio = 60 :=
by
  sorry

end total_earthworms_in_box_l712_712954


namespace annalise_total_cost_l712_712306

/-- 
Given conditions:
- 25 boxes of tissues.
- Each box contains 18 packs.
- Each pack contains 150 tissues.
- Each tissue costs $0.06.
- A 10% discount on the total price of the packs in each box.

Prove:
The total amount of money Annalise spent is $3645.
-/
theorem annalise_total_cost :
  let boxes := 25
  let packs_per_box := 18
  let tissues_per_pack := 150
  let cost_per_tissue := 0.06
  let discount_rate := 0.10
  let price_per_box := (packs_per_box * tissues_per_pack * cost_per_tissue)
  let discount_per_box := discount_rate * price_per_box
  let discounted_price_per_box := price_per_box - discount_per_box
  let total_cost := discounted_price_per_box * boxes
  total_cost = 3645 :=
by
  sorry

end annalise_total_cost_l712_712306


namespace pascal_triangle_expansion_l712_712662

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k       => 0
| n+1, k+1   => binom n k + binom n (k+1)

-- binom n k gives the binomial coefficient "n choose k"

theorem pascal_triangle_expansion (a : ℕ → ℕ) (n : ℕ) :
  (∀ k, a k = binom n k) →
  (1 + x)^n = ∑ i in Finset.range (n + 1), (a i) * x^i := 
by
  intro h
  sorry

end pascal_triangle_expansion_l712_712662


namespace odd_function_property_l712_712415

noncomputable def f : ℝ → ℝ := sorry -- Define the function to satisfy the given conditions

theorem odd_function_property (h_odd : ∀ x : ℝ, f (-x) = -f x)
                              (h_f_def : ∀ x : ℝ, 0 < x → f x = 2 + f (1/2) * log x / log 2)
                              (h_f_half : f (1/2) = 1) :
  f (-2) = -3 :=
by
  sorry

end odd_function_property_l712_712415


namespace canoeist_downstream_time_l712_712276

-- Definitions of the given conditions
def upstream_distance : ℝ := 12
def upstream_time : ℝ := 6
def current_speed : ℝ := 9
def downstream_distance : ℝ := 12

-- Define the equation for the paddling speed of the canoeist in still water
def paddling_speed_in_still_water : ℝ :=
  (upstream_distance / upstream_time) + current_speed

-- Define the effective speed downstream
def effective_speed_downstream : ℝ :=
  paddling_speed_in_still_water + current_speed

-- Problem statement: Prove that the time it takes to return downstream is 0.6 (or 3/5) hours.
theorem canoeist_downstream_time :
  downstream_distance / effective_speed_downstream = 0.6 := by
  sorry

end canoeist_downstream_time_l712_712276


namespace sum_of_b_for_one_solution_l712_712338

theorem sum_of_b_for_one_solution :
  (∀ b : ℝ, (∃ x : ℝ, 3 * x^2 + (b + 12) * x + 27 = 0) ∧ (∀ x1 x2 : ℝ, 3 * x1^2 + (b + 12) * x1 + 27 = 0 → 
  3 * x2^2 + (b + 12) * x2 + 27 = 0 → x1 = x2)) →
  (∑ b in {6, -30}, b) = -24 :=
by 
  sorry

end sum_of_b_for_one_solution_l712_712338


namespace probability_heads_exactly_8_in_10_l712_712633

def fair_coin_probability (n k : ℕ) : ℚ := (Nat.choose n k : ℚ) / (2 ^ n)

theorem probability_heads_exactly_8_in_10 :
  fair_coin_probability 10 8 = 45 / 1024 :=
by 
  sorry

end probability_heads_exactly_8_in_10_l712_712633


namespace ellipse_perimeter_l712_712796

/--
Given the ellipse (x²/41) + (y²/25) = 1 with two foci F₁ and F₂,
and a chord AB passing through point F₁, 
prove that the perimeter of triangle ABF₂ is 4√41.
-/
theorem ellipse_perimeter (x y : ℝ) (F₁ F₂ : ℝ × ℝ)
  (h₁ : x^2 / 41 + y^2 / 25 = 1)
  (h₂ : F₁ ∈ ({p : ℝ × ℝ | p.1^2 / 41 + p.2^2 / 25 = 1} : set (ℝ × ℝ)))
  (h₃ : F₂ ∈ ({p : ℝ × ℝ | p.1^2 / 41 + p.2^2 / 25 = 1} : set (ℝ × ℝ)))
  (h₄ : ∃ A B : ℝ × ℝ, A ∈ {p : ℝ × ℝ | p.1^2 / 41 + p.2^2 / 25 = 1} ∧ 
                         B ∈ {p : ℝ × ℝ | p.1^2 / 41 + p.2^2 / 25 = 1} ∧
                         (A.1 - F₁.1)^2 + (A.2 - F₁.2)^2 = 0 ∧
                         (B.1 - F₁.1)^2 + (B.2 - F₁.2)^2 = 0) :
  ∃ A B : ℝ × ℝ, (A.1 - F₁.1)^2 + (A.2 - F₁.2)^2 = 0 ∧
                  (B.1 - F₁.1)^2 + (B.2 - F₁.2)^2 = 0 ∧
                  (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
                   sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) + 
                   sqrt ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2)) = 4 * sqrt 41 := 
by
  sorry

end ellipse_perimeter_l712_712796


namespace min_geom_seq_value_l712_712107

noncomputable def geom_seq_min_value : ℝ :=
  let q := (λ (n : ℕ), Classical.epsilon (λ q, q > 0))
  let a : ℕ → ℝ := λ n, if n = 0 then 0 else q^((n-1 : ℕ))
  have h₁ : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8 := sorry,
  ∀ (n : ℕ) (q > 0), 
  ∃ (a : ℕ → ℝ) (h₂ : a n = q ^ (n - 1)),
  (∀ n, a n > 0) →
  2 * a 8 + a 7 = 54 := sorry

theorem min_geom_seq_value : geom_seq_min_value = 54 :=
sorry

end min_geom_seq_value_l712_712107


namespace propositions_correct_l712_712407

theorem propositions_correct (a b c d : ℝ) (ha : a > 0) (hb : b < 0) (hba : b > -a) (hcd : c < d) (hd : d < 0) :
  (ad : ℝ) ≤ (bc : ℝ) → -- proposition (1): ad ≤ bc (false equivalent to ad \not> bc)
  (\frac{a}{b} + \frac{b}{c} < 0) →      -- proposition (2): true
  (a - c > b - d) →                     -- proposition (3): true
  (a * (d - c) > b * (d - c))           -- proposition (4): true
:= by
  intro h1 h2 h3 h4
  sorry

end propositions_correct_l712_712407


namespace leo_basketball_points_l712_712479

theorem leo_basketball_points :
  ∃ x y : ℕ, x + y = 40 ∧ (0.75 * x + 0.80 * y) = 32 :=
by
  sorry

end leo_basketball_points_l712_712479


namespace number_of_possible_triples_l712_712297

-- Given conditions
variables (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)

-- Revenue equation
def revenue_equation : Prop := 10 * x + 5 * y + z = 120

-- Proving the solution
theorem number_of_possible_triples (h : revenue_equation x y z) : 
  ∃ (n : ℕ), n = 121 :=
by
  sorry

end number_of_possible_triples_l712_712297


namespace inequality_preserving_l712_712379

variables {a b : ℝ}

theorem inequality_preserving (h : a > b) : (0.9)^a < (0.9)^b :=
  sorry

end inequality_preserving_l712_712379


namespace collinearity_of_T_K_I_l712_712928

noncomputable def intersection_point (l1 l2 : Line) : Point := sorry

-- Definitions of lines AP, CQ, MP, NQ based on the problem context
variables {A P C Q M N I : Point} (lAP lCQ lMP lNQ : Line)
variables (T : Point) (K : Point)

-- Given conditions
def condition_1 : T = intersection_point lAP lCQ := sorry
def condition_2 : K = intersection_point lMP lNQ := sorry

-- Theorem statement
theorem collinearity_of_T_K_I : T ∈ line_through K I :=
by {
  -- These are the conditions that we're given in the problem
  have hT : T = intersection_point lAP lCQ := sorry,
  have hK : K = intersection_point lMP lNQ := sorry,
  -- Rest of the proof would go here
  sorry
}

end collinearity_of_T_K_I_l712_712928


namespace collinear_T_K_I_l712_712908

noncomputable def T (A P C Q : Point) : Point := intersection (line_through A P) (line_through C Q)
noncomputable def K (M P N Q : Point) : Point := intersection (line_through M P) (line_through N Q)

theorem collinear_T_K_I (A P C Q M N I : Point) :
  collinear [T A P C Q, K M P N Q, I] :=
sorry

end collinear_T_K_I_l712_712908


namespace distinct_arrangements_l712_712223

theorem distinct_arrangements (rocking_chairs : ℕ) (stools : ℕ) (bench : ℕ) (slots : ℕ) :
    rocking_chairs = 7 → stools = 2 → bench = 1 → slots = 10 →
    (∃ (arrangements : ℕ), arrangements = 360) := 
by
  intros h_rc h_st h_b h_slots
  use 360
  sorry

end distinct_arrangements_l712_712223


namespace max_blue_points_l712_712174

/-- There are blue and red points on a line. 
    It satisfies:
    1. There are at least 5 red points.
    2. Any interval with endpoints at red points that contains at least one red point inside contains at least 4 blue points.
    3. Any interval with endpoints at blue points that contains exactly 3 blue points inside contains at least 2 red points.
    Prove: The maximum number of blue points that can be on an interval with endpoints at red points, not containing other red points inside, is 4. -/
theorem max_blue_points (red_points : set ℝ) (blue_points : set ℝ) 
  (h1 : 5 ≤ red_points.to_finset.card)
  (h2 : ∀ (a b : ℝ), a ∈ red_points → b ∈ red_points → (a, b) ∩ red_points \ {a, b} ≠ ∅ → 4 ≤ (set.Icc a b ∩ blue_points).to_finset.card)
  (h3 : ∀ (c d : ℝ), c ∈ blue_points → d ∈ blue_points → 3 = (set.Icc c d ∩ blue_points \ {c, d}).to_finset.card → 2 ≤ (set.Icc c d ∩ red_points).to_finset.card) :
  (∃ a b : ℝ, a ∈ red_points ∧ b ∈ red_points ∧ (set.Icc a b ∩ red_points).to_finset.card = 2 ∧ (set.Icc a b ∩ blue_points).to_finset.card ≤ 4) :=
sorry

end max_blue_points_l712_712174


namespace parabola_equation_l712_712003

theorem parabola_equation (focus_point : ℝ × ℝ) (directrix : set (ℝ × ℝ)) :
  focus_point = (2, 1) → directrix = { p | p.1 = 0 } → (λ x y : ℝ, (y - 1) ^ 2 = 4 * (x - 1)) :=
by
  intros h_focus h_directrix
  have h1 : focus_point = (2, 1) := h_focus
  have h2 : directrix = { p : ℝ × ℝ | p.1 = 0 } := h_directrix
  sorry

end parabola_equation_l712_712003


namespace no_solution_for_equation_l712_712186

theorem no_solution_for_equation : 
  ∀ x : ℝ, (x ≠ 3) → (x-1)/(x-3) = 2 - 2/(3-x) → False :=
by
  intro x hx heq
  sorry

end no_solution_for_equation_l712_712186


namespace problem1_l712_712570

theorem problem1 (x y : ℝ) (h1 : 2^(x + y) = x + 7) (h2 : x + y = 3) : (x = 1 ∧ y = 2) :=
by
  sorry

end problem1_l712_712570


namespace figure_is_square_l712_712184

/-
  Given the following conditions on a figure:
  1. Diagonals bisect each other.
  2. Diagonals are perpendicular.
  3. Diagonals are of equal length.
  
  Prove that the figure is a square.
-/

theorem figure_is_square {F : Type} [Figure F]
  (h1 : bisect_diagonals F)
  (h2 : perpendicular_diagonals F)
  (h3 : equal_length_diagonals F) :
  is_square F :=
sorry

end figure_is_square_l712_712184


namespace melanie_missed_games_l712_712952

-- Define the total number of soccer games played and the number attended by Melanie
def total_games : ℕ := 64
def attended_games : ℕ := 32

-- Statement to be proven
theorem melanie_missed_games : total_games - attended_games = 32 := by
  -- Placeholder for the proof
  sorry

end melanie_missed_games_l712_712952


namespace balls_in_arithmetic_sequence_l712_712224

theorem balls_in_arithmetic_sequence (p q : ℕ) (h_coprime : Nat.gcd p q = 1) :
  (∀ (i : ℕ), 0 < i → probability_of_ball_in_bucket i = 2^(-i)) →
  (probability_three_balls_in_arithmetic_sequence = (6 : ℚ) / 49) :=
by
  -- Definition of the probability function and additional conditions can be embedded here
  -- Example of probability function assumption:
  assume h_prob : ∀ (i : ℕ), 0 < i → probability_of_ball_in_bucket i = 2^(-i),
  -- Embedding the condition such as no bucket contains more than one ball
  -- Derivation of the final statement here
  sorry

end balls_in_arithmetic_sequence_l712_712224


namespace domain_log_f_determined_by_2x_l712_712789

def domain_f (f : ℝ → ℝ) : set ℝ := { x | ∃ y, f y = x }

theorem domain_log_f_determined_by_2x {f : ℝ → ℝ} :
  (domain_f (λ x, f (2^x)) = set.Icc 1 2) → 
  (domain_f (λ x, f (real.log x / real.log 2)) = set.Icc 4 16) :=
by sorry

end domain_log_f_determined_by_2x_l712_712789


namespace solve_special_sine_system_l712_712272

noncomputable def special_sine_conditions1 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  let z := -(Real.pi / 2) + 2 * Real.pi * k
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = (-1)^n * Real.pi / 6 + Real.pi * n ∧
  z = -Real.pi / 2 + 2 * Real.pi * k

noncomputable def special_sine_conditions2 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := -Real.pi / 2 + 2 * Real.pi * k
  let z := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = -Real.pi / 2 + 2 * Real.pi * k ∧
  z = (-1)^n * Real.pi / 6 + Real.pi * n

theorem solve_special_sine_system (m n k : ℤ) :
  special_sine_conditions1 m n k ∨ special_sine_conditions2 m n k :=
sorry

end solve_special_sine_system_l712_712272


namespace ramu_spent_on_repairs_l712_712557

theorem ramu_spent_on_repairs :
  ∃ (RC : ℝ),
  let purchase_price := 42000
      sale_price := 60900
      profit_percent := 10.727272727272727 in
  let TC := purchase_price + RC in
  sale_price = TC * (1 + profit_percent / 100) → 
    RC = 13000 :=
begin
  sorry,
end

end ramu_spent_on_repairs_l712_712557


namespace base_k_for_fraction_l712_712016

-- Definition of a positive integer and conditions
def is_positive_integer (n : ℕ) : Prop := n > 0

-- Definition of the repeating base-k representation
def repeating_fraction_base_k (k : ℕ) : ℚ := 
  have h1 : k > 1 := sorry, -- Placeholder to ensure k > 1
  (2 / k : ℚ) + (4 / (k ^ 2) : ℚ) + (2 / (k ^ 3) : ℚ) + (4 / (k ^ 4) : ℚ) + (2 / (k ^ 5) : ℚ) + (4 / (k ^ 6) : ℚ)

-- Theorem stating that k must be 18
theorem base_k_for_fraction: 
  ∀ (k: ℕ), is_positive_integer k → (repeating_fraction_base_k k = (8 : ℚ) / 65 → k = 18) :=
by {
  intros k pos_k rep_eq,
  sorry
}

end base_k_for_fraction_l712_712016


namespace angle_AEC_proof_l712_712468

-- Defining the setup for the problem
def triangle_isosceles (ABC : Triangle) (AB AC : Real) : Prop :=
  AB = AC

def point_on_line_extended (A B D : Point) : Prop :=
  collinear [A, B, D] ∧ seg_equal (A, D) (D, B)

def parallel_lines (D AC : Line) : Prop :=
  parallel D AC

def angle_equivalence (angle1 angle2 : ℝ) : Prop :=
  angle1 = angle2

-- Main statement for proving the requested angle
theorem angle_AEC_proof (ABC : Triangle) (A B C D E : Point) (AB AC : Real) (m∠ACB : ℝ) : 
  triangle_isosceles ABC AB AC →
  point_on_line_extended A B D →
  E ∈ line AB →
  parallel_lines D (line AC) →
  m∠ACB = 58 →
  angle_equivalence (m∠AEC) 64 :=
by 
  sorry

end angle_AEC_proof_l712_712468


namespace sum_of_edges_l712_712697

theorem sum_of_edges (a r : ℝ) 
  (h_volume : (a^3 = 512))
  (h_surface_area : (2 * (a^2 / r + a^2 + a^2 * r) = 384))
  (h_geometric_progression : true) :
  (4 * ((a / r) + a + (a * r)) = 96) :=
by
  -- It is only necessary to provide the theorem statement
  sorry

end sum_of_edges_l712_712697


namespace least_n_multiple_of_100_l712_712937

def b : ℕ → ℕ 
| 15 := 15
| (n+1) := 50 * b n + (n+1)^2

theorem least_n_multiple_of_100 :
 ∃ n > 15, b n % 100 = 0 ∧ ∀ m > 15, m < n → b m % 100 ≠ 0 := 
begin
  use 16,
  split,
  { exact nat.lt_succ_self 15 },
  split,
  { change b 16 % 100 = 0,
    -- calculation here, proving that b 16 ≡ 0 mod 100
    sorry },
  { intros m hm hmn,
    -- proving no m < 16, m > 15 such that b m ≡ 0 mod 100
    sorry }
end

end least_n_multiple_of_100_l712_712937


namespace jim_profit_percentage_l712_712875

theorem jim_profit_percentage (S C : ℝ) (H1 : S = 670) (H2 : C = 536) :
  ((S - C) / C) * 100 = 25 :=
by
  sorry

end jim_profit_percentage_l712_712875


namespace poly_perfect_fourth_l712_712218

theorem poly_perfect_fourth (a b c : ℤ) (h : ∀ x : ℤ, ∃ k : ℤ, (a * x^2 + b * x + c) = k^4) : 
  a = 0 ∧ b = 0 :=
sorry

end poly_perfect_fourth_l712_712218


namespace inverse_of_problem_matrix_is_zero_matrix_l712_712351

def det (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

def zero_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 0], ![0, 0]]

noncomputable def problem_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, -6], ![-2, 3]]

theorem inverse_of_problem_matrix_is_zero_matrix :
  det problem_matrix = 0 → problem_matrix⁻¹ = zero_matrix :=
by
  intro h
  -- Proof steps will be written here
  sorry

end inverse_of_problem_matrix_is_zero_matrix_l712_712351


namespace minimal_total_distance_l712_712110

variable (A B : ℝ) -- Coordinates of houses A and B on a straight road
variable (h_dist : B - A = 50) -- The distance between A and B is 50 meters

-- Define a point X on the road
variable (X : ℝ)

-- Define the function that calculates the total distance from point X to A and B
def total_distance (A B X : ℝ) := abs (X - A) + abs (X - B)

-- The theorem stating that the total distance is minimized if X lies on the line segment AB
theorem minimal_total_distance : A ≤ X ∧ X ≤ B ↔ total_distance A B X = B - A :=
by
  sorry

end minimal_total_distance_l712_712110


namespace jerry_has_36_stickers_l712_712872

def fred_stickers : ℕ := 18
def george_stickers : ℕ := fred_stickers - 6
def jerry_stickers : ℕ := 3 * george_stickers

theorem jerry_has_36_stickers : jerry_stickers = 36 :=
by
  unfold fred_stickers
  unfold george_stickers
  unfold jerry_stickers
  sorry

end jerry_has_36_stickers_l712_712872


namespace smallest_lucky_number_theorem_specific_lucky_number_theorem_l712_712833

-- Definitions based on the given conditions
def is_lucky_number (M : ℕ) : Prop :=
  ∃ (A B : ℕ), (M = A * B) ∧
               (A ≥ B) ∧
               (A ≥ 10 ∧ A ≤ 99) ∧
               (B ≥ 10 ∧ B ≤ 99) ∧
               (A / 10 = B / 10) ∧
               (A % 10 + B % 10 = 6)

def smallest_lucky_number : ℕ :=
  165

def P (M A B : ℕ) := A + B
def Q (M A B : ℕ) := A - B

def specific_lucky_number (M A B : ℕ) : Prop :=
  M = A * B ∧ (P M A B) / (Q M A B) % 7 = 0

-- Theorems to prove
theorem smallest_lucky_number_theorem :
  ∃ M, is_lucky_number M ∧ M = smallest_lucky_number := by
  sorry

theorem specific_lucky_number_theorem :
  ∃ M A B, is_lucky_number M ∧ specific_lucky_number M A B ∧ M = 3968 := by
  sorry

end smallest_lucky_number_theorem_specific_lucky_number_theorem_l712_712833


namespace rabbit_count_l712_712240

theorem rabbit_count (r1 r2 : ℕ) (h1 : r1 = 8) (h2 : r2 = 5) : r1 + r2 = 13 := 
by 
  sorry

end rabbit_count_l712_712240


namespace vec_dot_product_l712_712042

variables (a b c : ℝ^3)

-- Given conditions:
def condition1 : Prop := sqrt 3 • a + b + 2 • c = 0
def condition2 : Prop := ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥c∥ = 1

-- The target statement to prove:
theorem vec_dot_product : 
  condition1 a b c ∧ condition2 a b c →
  a.dot_product (b + c) = - (sqrt 3 / 2) :=
sorry

end vec_dot_product_l712_712042


namespace collinear_TKI_l712_712890

-- Definitions for points and lines
variables {A P C Q M N I T K : Type}
variable {line : Type → Type}
variables (AP : line A → line P) (CQ : line C → line Q) (MP : line M → line P) (NQ : line N → line Q)

-- Conditions from the problem
-- Assume there exist points T and K which are intersections of the specified lines
axiom intersects_AP_CQ : ∃ (T : Type), AP T = CQ T
axiom intersects_MP_NQ : ∃ (K : Type), MP K = NQ K

-- Collinearity of points T, K, and I
theorem collinear_TKI : ∀ (I : Type) (T : Type) (K : Type),
  intersects_AP_CQ → intersects_MP_NQ → collinear I T K :=
by sorry

end collinear_TKI_l712_712890


namespace no_solution_exists_l712_712010

-- Theorem Statement
theorem no_solution_exists :
  ¬ ∃ (x y : ℝ), 64^(x^2 + y + x) + 64^(x + y^2 + y) = 1 := 
by
  sorry

end no_solution_exists_l712_712010


namespace find_angle_l712_712385

theorem find_angle (α β : ℝ) (h₁ : 0 < α ∧ α < π) (h₂ : 0 < β ∧ β < π)
  (h₃ : tan (α - β) = 1 / 2) (h₄ : tan β = -1 / 7) : 2 * α - β = -3 * π / 4 :=
sorry

end find_angle_l712_712385


namespace parallel_lines_l712_712811

theorem parallel_lines (m : ℝ) 
  (h : 3 * (m - 2) + m * (m + 2) = 0) 
  : m = 1 ∨ m = -6 := 
by 
  sorry

end parallel_lines_l712_712811


namespace power_function_form_l712_712068

theorem power_function_form (f : ℝ → ℝ) 
  (h : f (sqrt 3 / 3) = 3) : 
  f = λ x, x^(-2) :=
  sorry

end power_function_form_l712_712068


namespace student_question_choices_l712_712696

-- Definitions based on conditions
def partA_questions := 10
def partB_questions := 10
def choose_from_partA := 8
def choose_from_partB := 5

-- The proof problem statement
theorem student_question_choices :
  (Nat.choose partA_questions choose_from_partA) * (Nat.choose partB_questions choose_from_partB) = 11340 :=
by
  sorry

end student_question_choices_l712_712696


namespace eight_natural_numbers_exist_l712_712123

theorem eight_natural_numbers_exist :
  ∃ (n : Fin 8 → ℕ), (∀ i j : Fin 8, i ≠ j → ¬(n i ∣ n j)) ∧ (∀ i j : Fin 8, i ≠ j → n i ∣ (n j * n j)) :=
by 
  sorry

end eight_natural_numbers_exist_l712_712123


namespace translate_function_right_l712_712233

theorem translate_function_right 
  (theta varphi : ℝ)
  (h1 : -π / 2 < theta ∧ theta < π / 2)
  (h2 : 0 < varphi ∧ varphi < π)
  (h3 : sin(theta) = sqrt 3 / 2)
  (h4 : sin(-2 * varphi + theta) = sqrt 3 / 2) :
  varphi = 5 * π / 6 :=
sorry

end translate_function_right_l712_712233


namespace fraction_of_male_birds_l712_712655

theorem fraction_of_male_birds (T : ℕ) (h_cond1 : T ≠ 0) :
  let robins := (2 / 5) * T
  let bluejays := T - robins
  let male_robins := (2 / 3) * robins
  let male_bluejays := (1 / 3) * bluejays
  (male_robins + male_bluejays) / T = 7 / 15 :=
by 
  sorry

end fraction_of_male_birds_l712_712655


namespace smallest_sum_of_squares_l712_712989

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l712_712989


namespace smallest_sum_of_squares_l712_712988

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l712_712988


namespace find_range_of_p_l712_712032

noncomputable def range_of_p (p : ℝ) : Prop :=
∀ n : ℕ, n > 0 → 
let S := λ n, (-1) ^ n * n in
let a := λ n, S n - S (n - 1) in
(a (n + 1) - p) * (a n - p) < 0

theorem find_range_of_p (p : ℝ) : -1 < p ∧ p < 3 ↔ range_of_p p :=
sorry

end find_range_of_p_l712_712032


namespace cos_translation_right_shift_l712_712232

theorem cos_translation_right_shift (x : ℝ) : (cos 2x) = (cos (2 (x + π / 6))) :=
by
  sorry

end cos_translation_right_shift_l712_712232


namespace collinearity_of_T_K_I_l712_712898

-- Definitions of the points and lines
variables {A P Q M N C I T K : Type} [Nonempty A] [Nonempty P] [Nonempty Q] 
  [Nonempty M] [Nonempty N] [Nonempty C] [Nonempty I]

-- Intersection points conditions
def intersect (l₁ l₂ : Type) : Type := sorry

-- Given conditions
def condition_1 : T = intersect (line A P) (line C Q) := sorry
def condition_2 : K = intersect (line M P) (line N Q) := sorry

-- Proof that T, K, and I are collinear
theorem collinearity_of_T_K_I : collinear {T, K, I} := by
  have h1 : T = intersect (line A P) (line C Q) := condition_1
  have h2 : K = intersect (line M P) (line N Q) := condition_2
  -- Further steps needed to prove collinearity
  sorry

end collinearity_of_T_K_I_l712_712898


namespace combined_percentage_reduction_approx_required_percentage_increase_approx_l712_712191

-- Define the individual reduction factors
def reduction_factors : List ℝ := [0.88, 0.93, 0.95, 0.90, 0.85]

-- Calculate the combined reduction factor
def combined_reduction_factor : ℝ := reduction_factors.reduce (*)

-- Define the combined percentage reduction
def combined_percentage_reduction : ℝ := (1 - combined_reduction_factor) * 100

-- Define the required percentage increase to return to the original salary
def required_percentage_increase : ℝ := ((1 / combined_reduction_factor) - 1) * 100

-- Prove that the combined percentage reduction is approximately 40.43%
theorem combined_percentage_reduction_approx :
  abs (combined_percentage_reduction - 40.43) < 0.01 := by
  sorry

-- Prove that the required percentage increase to return to the original salary is approximately 67.87%
theorem required_percentage_increase_approx :
  abs (required_percentage_increase - 67.87) < 0.01 := by
  sorry

end combined_percentage_reduction_approx_required_percentage_increase_approx_l712_712191


namespace sequence_property_l712_712461

theorem sequence_property (a : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, (√(a (i + 1)))) = n ^ 2 + 3 * n) :
  a 2 = 36 ∧ (∑ i in Finset.range n, a (i + 1) / (i + 2)) = 2 * n^2 + 6 * n :=
sorry

end sequence_property_l712_712461


namespace find_length_EF_l712_712501

theorem find_length_EF (DE DF : ℝ) (angle_E : ℝ) (angle_E_val : angle_E = 45) (DE_val : DE = 100) (DF_val : DF = 100 * Real.sqrt 2) : ∃ EF, EF = 141.421 :=
by
  exists 141.421
  sorry

end find_length_EF_l712_712501


namespace inclination_angle_phi_l712_712594

noncomputable def R : ℝ := 1 -- Assume some positive value for radius to instantiate the environment

noncomputable def r := (3/4) * R

-- Define h as the height which is equal to the larger radius
noncomputable def h := R

-- The angle of inclination phi, where tan(phi) = R / (R - r)
noncomputable def phi := Real.arctan (R / (R - r))

-- The theorem which we are proving
theorem inclination_angle_phi : phi = Real.arctan(4) := 
by 
  -- The concrete proof steps would go here
  sorry

end inclination_angle_phi_l712_712594


namespace certain_number_division_l712_712653

theorem certain_number_division (x : ℝ) (h : x / 3 + x + 3 = 63) : x = 45 :=
by
  sorry

end certain_number_division_l712_712653


namespace symmetry_center_is_odd_sum_value_l712_712598

noncomputable theory

def f (x : ℝ) : ℝ := x^3 - 3 * x^2

def g (x : ℝ) : ℝ := f(x + 1) + 2

theorem symmetry_center : g(x) = x^3 - 3 * x :=
by {
  simp [f, g],
  sorry
}

theorem is_odd (x : ℝ) : g(-x) = - g(x) :=
by {
  simp [g],
  sorry
}

theorem sum_value : (∑ i in finset.range 4043, f ((-2020 : ℝ) + i)) = -8086 :=
by {
  sorry
}

end symmetry_center_is_odd_sum_value_l712_712598


namespace max_product_of_two_five_digit_numbers_l712_712740

theorem max_product_of_two_five_digit_numbers (d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ)
  (h_digits : {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∃ a b : ℕ, (a * b = 97640 * 85321) ∧ 
            (a = 97420 ∨ 
             (b = 86540 ∨ b = 76430 ∨ b = 97420 ∨ b = 96210 ∨ b = 84321 ∨ a = 86540 ∨ a = 76430 ∨ a = 96210 ∨ a = 84321)) :=
begin
  use [97420, 85321],
  split,
  { refl },
  { left, refl }
end

end max_product_of_two_five_digit_numbers_l712_712740


namespace max_modulus_l712_712046

open Complex

noncomputable def max_modulus_condition (z : ℂ) : Prop :=
  abs (z - (0 + 2*Complex.I)) = 1

theorem max_modulus : ∀ z : ℂ, max_modulus_condition z → abs z ≤ 3 :=
  by sorry

end max_modulus_l712_712046


namespace problem_proof_l712_712968

-- Parameters of the problem
def a := 2^8 + 4^5
def b := (2^3 - (-2)^3) ^ 10
def result := 1342177280

theorem problem_proof : a * b = result :=
by
  calc
    a * b = (2^8 + 4^5) * (2^3 - (-2)^3) ^ 10 : by refl
      ... = 1280 * 1048576                        : by sorry
      ... = 1342177280                            : by sorry

end problem_proof_l712_712968


namespace blue_cross_area_l712_712674

theorem blue_cross_area (A : ℝ) (hA1 : 0.5 * A) (hA2 : 0.2 * A) (h_width_half : true) : 
  (0.3 : ℝ) * A := 
sorry

end blue_cross_area_l712_712674


namespace problem_solution_l712_712401

open Real

def p (x : ℝ) : Prop :=
  log (1/3) x > -1 ∧ x^2 - 6 * x + 8 < 0

def q (x : ℝ) (a : ℝ) : Prop :=
  2 * x^2 - 9 * x + a < 0

theorem problem_solution (a : ℝ) : (∀ x : ℝ, p x → q x a) ↔ (7 ≤ a ∧ a ≤ 8) :=
sorry

end problem_solution_l712_712401


namespace collinear_a_d_x_l712_712855

-- Definitions of points and geometry on the circle
variable {α β : ℝ}

-- Coordinates of the points B, C, and A
def point_B := (ℝ.cos α, -ℝ.sin α)
def point_C := (ℝ.cos α, ℝ.sin α)
def point_A := (ℝ.cos β, ℝ.sin β)

-- Coordinate of point D
def point_D := (1 / ℝ.cos α, 0)

-- Calculation related to the construction of square on AB and AC
structure Point where
  x : ℝ
  y : ℝ

def point_H (B A : Point) : Point :=
  let BA := Point.mk (A.x - B.x) (A.y - B.y)
  Point.mk (B.x - BA.y) (B.y + BA.x)

def point_F (A C : Point) : Point :=
  let AC := Point.mk (C.x - A.x) (C.y - A.y)
  Point.mk (A.x - AC.y) (A.y + AC.x)

-- Coordinates of corresponding points
def H := point_H (Point.mk (ℝ.cos α) (-ℝ.sin α)) (Point.mk (ℝ.cos β) (ℝ.sin β))
def F := point_F (Point.mk (ℝ.cos β) (ℝ.sin β)) (Point.mk (ℝ.cos α) (ℝ.sin α))

-- Tangent lines intersecting to form point X
def line_slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

def point_X := sorry -- coordinates of intersection of lines are omitted

-- Slopes comparisons for collinearity
def line_slope_da (A D : Point) : ℝ :=
  (A.y - D.y) / (A.x - D.x)

def line_slope_ax (A X : Point) : ℝ :=
  (A.y - X.y) / (A.x - X.x)

-- Proof statement: Points A, D, X are collinear
theorem collinear_a_d_x : line_slope_da (Point.mk (ℝ.cos β) (ℝ.sin β)) (Point.mk (1 / ℝ.cos α) 0) =
  line_slope_ax (Point.mk (ℝ.cos β) (ℝ.sin β)) (point_X) :=
  sorry

end collinear_a_d_x_l712_712855


namespace prime_product_divisors_l712_712096

theorem prime_product_divisors (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (m : ℕ) (h : Nat.numDivisors (p^6 * q^m) = 56) : m = 7 := 
sorry

end prime_product_divisors_l712_712096


namespace probability_of_triangle_with_nonagon_side_l712_712370

-- Definitions based on the given conditions
def num_vertices : ℕ := 9

def total_triangles : ℕ := Nat.choose num_vertices 3

def favorable_outcomes : ℕ :=
  let one_side_is_side_of_nonagon := num_vertices * 5
  let two_sides_are_sides_of_nonagon := num_vertices
  one_side_is_side_of_nonagon + two_sides_are_sides_of_nonagon

def probability : ℚ := favorable_outcomes / total_triangles

-- Lean 4 statement to prove the equivalence of the probability calculation
theorem probability_of_triangle_with_nonagon_side :
  probability = 9 / 14 :=
by
  sorry

end probability_of_triangle_with_nonagon_side_l712_712370


namespace law_I_holds_law_II_does_not_hold_law_III_does_not_hold_l712_712527

def star (a b : ℝ) : ℝ := (a * b) / 2

theorem law_I_holds (x y z : ℝ) : star x (y + z) = star x y + star x z :=
by 
  unfold star
  sorry

theorem law_II_does_not_hold (x y z : ℝ) : x + star y z ≠ star (x + y) (x + z) :=
by 
  unfold star
  sorry

theorem law_III_does_not_hold (x y z : ℝ) : star x (star y z) ≠ star (star x y) (star x z) :=
by 
  unfold star
  sorry

end law_I_holds_law_II_does_not_hold_law_III_does_not_hold_l712_712527


namespace angle_DAB_is_45_degrees_l712_712268

theorem angle_DAB_is_45_degrees
  (A B D C E : Type)
  [inner_product_space ℝ A]
  [inner_product_space ℝ B]
  [inner_product_space ℝ C]
  [inner_product_space ℝ D]
  [inner_product_space ℝ E]
  (h1 : ∃ (Δ : triangle B A D), is_right_angle_at Δ B)
  (h2 : ∃ (α : line A D), ∃ (β : point_on line α C), AC = CD)
  (h3 : ∃ (γ : line A B), ∃ (δ : point_on line γ E), AE = EB)
  (h4 : AB = BC)
  (h5 : AD = BD) :
  angle DAB = 45 :=
by
  sorry

end angle_DAB_is_45_degrees_l712_712268


namespace lines_are_perpendicular_l712_712215

theorem lines_are_perpendicular :
  ∀ (l₁ l₂ : ℝ),
  (l₁^2 - 3*l₁ - 1 = 0) ∧ (l₂^2 - 3*l₂ - 1 = 0) →
  l₁ * l₂ = -1 →
  l₁ * l₂ = -1 :=
by
  intros l₁ l₂ h_roots h_product
  show l₁ * l₂ = -1 from h_product
  sorry

end lines_are_perpendicular_l712_712215


namespace correct_regression_equation_l712_712428

-- Problem Statement
def negatively_correlated (x y : ℝ) : Prop := sorry -- Define negative correlation for x, y
def sample_mean_x : ℝ := 3
def sample_mean_y : ℝ := 3.5
def regression_equation (b0 b1 : ℝ) (x : ℝ) : ℝ := b0 + b1 * x

theorem correct_regression_equation 
    (H_neg_corr : negatively_correlated x y) :
    regression_equation 9.5 (-2) sample_mean_x = sample_mean_y :=
by
    -- The proof will go here, skipping with sorry
    sorry

end correct_regression_equation_l712_712428


namespace tangent_line_l712_712437

-- Definitions for the problem setup
variables {α : Type} [EuclideanGeometry α]
variables (A B C D E : α) (e BC : Line α)

-- Assumptions for the conditions stated in the problem
hypothesis (h1 : A ∉ BC)
hypothesis (h2 : e ∥ BC)
hypothesis (h3 : ∃ D, intersection (lineThrough A B) e D)
hypothesis (h4 : ∃ E, secondIntersection (circumcircle A D C) e E)

-- Goal to prove
theorem tangent_line (h_circ : circumcircle A B C) :
  Tangent E C h_circ :=
begin
  sorry
end

end tangent_line_l712_712437


namespace sum_min_max_m_l712_712429

noncomputable def circle (x y : ℝ) : Prop := (x - 3) ^ 2 + (y - 4) ^ 2 = 1

noncomputable def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)

noncomputable def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

theorem sum_min_max_m (m : ℝ) (a b : ℝ) (P : ℝ × ℝ) :
  circle P.1 P.2 ∧ ∃ P : ℝ × ℝ, (P.1 = a ∧ P.2 = b ∧ (P.1 - 3) ^ 2 + (P.2 - 4) ^ 2 = 1) ∧
  (a + m) * (a - m) + b ^ 2 = 0 ∧ m = real.sqrt (a ^ 2 + b ^ 2) →
  max (real.sqrt (a^2 + b^2 + 1)) + min (real.sqrt (a^2 + b^2 - 1)) = 10 :=
by
  sorry

end sum_min_max_m_l712_712429


namespace discriminant_quadratic_eq_l712_712636

-- Given the quadratic equation 5x^2 + 3x - 8
def quadratic_eq (x : ℝ) : ℝ := 5 * x^2 + 3 * x - 8

-- Define the discriminant function for the quadratic equation ax^2 + bx + c
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main statement to prove
theorem discriminant_quadratic_eq :
  discriminant 5 3 (-8) = 169 := by
sorry

end discriminant_quadratic_eq_l712_712636


namespace area_of_triangle_AOT_l712_712546

theorem area_of_triangle_AOT :
  let T := (3, 3)
  let O := (0, 0)
  let A := (real.sqrt(2) * real.cos (real.pi / 4) + 3, real.sqrt(2) * real.sin (real.pi / 4) + 3)
  let OT := real.sqrt ((3 - 0)^2 + (3 - 0)^2)
  let AT := OT
  let AOT := real.pi / 4
  (∠ O A T = real.pi / 2) →
  let Q := 1 / 2 * OT * AT
  Q = 9 :=
by
  -- Definition of Points:
  let T := (3 : ℝ, 3 : ℝ)
  let O := (0 : ℝ, 0 : ℝ)
  let A := (3 + 3, 3 + 3)
  
  -- Distance Calculation:
  let OT := real.sqrt ((3 - 0)^2 + (3 - 0)^2)
  have OT_val : OT = 3 * real.sqrt 2 := by
    unfold OT
    norm_num
    ring
  
  let AT := OT

  -- Helper to assume ∠ O A T = 90°:
  have h_triangle : ∠ O A T = real.pi / 2 := sorry
 
  -- Define Area of Triangle:
  let Q := 1 / 2 * OT * AT

  -- Simplify and Validate the Area Q:
  have Q_val : Q = 9 := by
    unfold Q
    rw OT_val
    norm_num
    ring

  exact Q_val

end area_of_triangle_AOT_l712_712546


namespace savings_needed_for_vacation_l712_712195

-- Define the conditions
def available_funds : ℝ := 150000
def total_vacation_cost : ℝ := 182200

def betta_bank_rate : ℝ := 0.036
def gamma_bank_rate : ℝ := 0.045
def omega_bank_rate : ℝ := 0.0312
def epsilon_bank_rate : ℝ := 0.0025

def betta_bank_interest (P : ℝ) (r : ℝ) (n t : ℝ) := P * (1 + r / n)^(n * t)
def gamma_bank_interest (P : ℝ) (r t : ℝ) := P * (1 + r * t)
def omega_bank_interest (P : ℝ) (r : ℝ) (n t : ℝ) := P * (1 + r / n)^(n * t)
def epsilon_bank_interest (P : ℝ) (r n t : ℝ) := P * (1 + r)^(n * t)

-- Define the final amounts after 6 months for each bank
def betta_bank_amount := betta_bank_interest available_funds betta_bank_rate 12 0.5
def gamma_bank_amount := gamma_bank_interest available_funds gamma_bank_rate 0.5
def omega_bank_amount := omega_bank_interest available_funds omega_bank_rate 4 0.5
def epsilon_bank_amount := epsilon_bank_interest available_funds epsilon_bank_rate 1 6

-- Define the interest earned for each bank
def betta_bank_interest_earned := betta_bank_amount - available_funds
def gamma_bank_interest_earned := gamma_bank_amount - available_funds
def omega_bank_interest_earned := omega_bank_amount - available_funds
def epsilon_bank_interest_earned := epsilon_bank_amount - available_funds

-- Define the amounts that need to be saved from salary for each bank
def betta_bank_savings_needed := total_vacation_cost - available_funds - betta_bank_interest_earned
def gamma_bank_savings_needed := total_vacation_cost - available_funds - gamma_bank_interest_earned
def omega_bank_savings_needed := total_vacation_cost - available_funds - omega_bank_interest_earned
def epsilon_bank_savings_needed := total_vacation_cost - available_funds - epsilon_bank_interest_earned

-- The theorem to be proven
theorem savings_needed_for_vacation :
  betta_bank_savings_needed = 29479.67 ∧
  gamma_bank_savings_needed = 28825 ∧
  omega_bank_savings_needed = 29850.87 ∧
  epsilon_bank_savings_needed = 29935.89 :=
by sorry

end savings_needed_for_vacation_l712_712195


namespace sum_of_indices_l712_712745

theorem sum_of_indices (Φ Ψ : ℝ → ℝ) (hΦ : ∀ x, Φ x = sin x * sin 2) (hΨ : ∀ x, Ψ x = cos x) :
  let θ1 := 1, θ2 := 2, θ3 := 45, θ4 := 46 in 
  (θ1 + θ2 + θ3 + θ4) = 94 := 
by
  --The proof steps are skipped for this statement
  sorry

end sum_of_indices_l712_712745


namespace mod_problem_l712_712007
open Int

theorem mod_problem : 
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [MOD 11] :=
by
  use 3
  split
  · show 0 ≤ 3
    exact le_refl 3
  split
  · show 3 ≤ 10
    exact le_of_lt (by norm_num)
  · show 3 ≡ 123456 [MOD 11]
    exact Nat.modeq.symm (by norm_num [mod_eq_of_lt, Int.mod_def])

end mod_problem_l712_712007


namespace prove_general_formula_l712_712806

-- Define the sequence conditions
def a (n : ℕ) : ℕ := 2 * n + 1

-- Conditions
def a_1 : Prop := a 1 = 3
def a_10 : Prop := a 10 = 21

-- Prove the general formula
def general_formula (n : ℕ) : Prop := a n = 2 * n + 1

-- Specific values to prove
def a_2005 : Prop := a 2005 = 4011

-- Sum of the first 10 terms
def S_10 : Prop := (∑ i in Finset.range 10 + 1, a i) = 120 -- sum from a_1 to a_10

-- Translating the math problem into Lean proof statements

theorem prove_general_formula : a_1 ∧ a_10 ∧ general_formula ∧ a_2005 ∧ S_10 :=
by
  have a_1 : a 1 = 3 := rfl
  have a_10 : a 10 = 21 := rfl
  have general_formula : ∀ n, a n = 2 * n + 1 := by simp [a]
  have a_2005 := rfl
  have S_10 := rfl
  exact ⟨a_1, a_10, general_formula, a_2005, S_10⟩

end prove_general_formula_l712_712806


namespace remaining_seeds_l712_712949

def initial_seeds : Nat := 54000
def seeds_per_zone : Nat := 3123
def number_of_zones : Nat := 7

theorem remaining_seeds (initial_seeds seeds_per_zone number_of_zones : Nat) : 
  initial_seeds - (seeds_per_zone * number_of_zones) = 32139 := 
by 
  sorry

end remaining_seeds_l712_712949


namespace possible_value_of_x_l712_712285

theorem possible_value_of_x
  (x : ℝ)
  (h1 : (x > 0))
  (h2 : (2 - 2)^2 + (10 - 5)^2 = 169) :
  x = 14 :=
by
  have distance_formula := (x - 2)^2 + (10 - 5)^2 = 169
  rw distance_formula at h2
  have simplified : (x - 2)^2 + 25 = 169 := by sorry
  have squared : (x - 2)^2 = 144 := by sorry
  have x_value := (x - 2) = 12 ∨ (x - 2) = -12 := by sorry
  cases x_value with pos neg
  { rw pos at h1
    exact h1 (14) }
  { rw neg at h1
    exact h1 (14) }

end possible_value_of_x_l712_712285


namespace involution_mod_1000_l712_712521

open Function

def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def is_involution (f : ℕ → ℕ) : Prop :=
  ∀ x ∈ A, f (f x) = x

def number_of_involutions : ℕ :=
  -- Placeholder for the actual number of involutions calculation
  -- This should represent the computation of N as explained in the solution
  let N := 227106 in N

theorem involution_mod_1000 : number_of_involutions % 1000 = 106 :=
by
  -- Placeholder for the proof (start with a proof outline)
  sorry

end involution_mod_1000_l712_712521


namespace min_sum_x_correct_l712_712403

noncomputable def min_sum_x : ℝ :=
-5907

theorem min_sum_x_correct
    (x : ℕ → ℝ)
    (h1 : |x 1| = 99)
    (h2 : ∀ n, 2 ≤ n ∧ n ≤ 2014 → |x n| = |x (n-1) + 1|) :
    x 1 + x 2 + ∀ (n : ℕ), n ∈ Finset.range 2014 → real.sum ∑ i in Finset.range n, x (i+1) = -5907 :=
begin
  sorry
end

end min_sum_x_correct_l712_712403


namespace type_B_machine_time_l712_712300

theorem type_B_machine_time :
  (2 * (1 / 5) + 3 * (1 / B) = 5 / 6) → B = 90 / 13 :=
by 
  intro h
  sorry

end type_B_machine_time_l712_712300


namespace trigonometric_identity_proof_l712_712406

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end trigonometric_identity_proof_l712_712406


namespace complex_number_solution_l712_712453

theorem complex_number_solution (z : ℂ) (h : z * (1 + complex.i) = 2 * complex.i) : z = 1 + complex.i :=
sorry

end complex_number_solution_l712_712453


namespace part1_solution_set_part2_value_of_t_l712_712538

open Real

def f (t x : ℝ) : ℝ := x^2 - (t + 1) * x + t

-- Statement for the equivalent proof problem
theorem part1_solution_set (x : ℝ) : 
  (t = 3 → f 3 x > 0 ↔ (x < 1) ∨ (x > 3)) :=
by
  sorry

theorem part2_value_of_t (t : ℝ) :
  (∀ x : ℝ, f t x ≥ 0) → t = 1 :=
by
  sorry

end part1_solution_set_part2_value_of_t_l712_712538


namespace number_of_ways_to_buy_souvenirs_l712_712341

-- Definition of the problem conditions
def types_of_souvenirs : ℕ := 11
def souvenirs_10_yuan : ℕ := 8
def souvenirs_5_yuan : ℕ := 3
def total_spent : ℕ := 50

-- The proof goal that we need to establish as a theorem in Lean
theorem number_of_ways_to_buy_souvenirs : (Σ (count_10_yuan count_5_yuan : ℕ),
((count_10_yuan + count_5_yuan ≤ 11) ∧ (count_10_yuan ≤ souvenirs_10_yuan) ∧ (count_5_yuan ≤ souvenirs_5_yuan)
∧ (count_10_yuan * 10 + count_5_yuan * 5 = total_spent))) 
= 266 := sorry

end number_of_ways_to_buy_souvenirs_l712_712341


namespace triangle_B_and_max_area_l712_712865

theorem triangle_B_and_max_area (a c : ℝ) (B : ℝ) (K : ℝ) (h₁ : B = 120) 
  (h₂ : b = 1) (h₃ : cos C + (2 * a + c) * cos B = 0) :
  B = 120 ∧ K = sqrt 3 / 12 := 
sorry

end triangle_B_and_max_area_l712_712865


namespace area_A_l712_712959

open Real

variables {A B C A' B' C' : Type} [metric_space A] [metric_space B] [metric_space C]
variables [metric_space A'] [metric_space B'] [metric_space C']
variables (AB : ℝ) (BC : ℝ) (CA : ℝ)
variables (AB' : ℝ) (BC' : ℝ) (CA' : ℝ)
variables (R : ℝ)

-- Given the points C', A', and B' on the sides AB, BC, and CA of triangle ABC respectively
-- And given the circumradius R of triangle ABC
-- We want to prove the area of triangle A'B'C'
theorem area_A'B'C'_formula :
  let area : ℝ := (AB' * BC' * CA' + AC' * CB' * BA') / (4 * R) in
  area = (AB' * BC' * CA' + AC' * CB' * BA') / (4 * R) :=
by {
  sorry  -- The proof would be constructed here
}

end area_A_l712_712959


namespace knight_king_moves_incompatible_l712_712631

-- Definitions for moves and chessboards
structure Board :=
  (numbering : Fin 64 → Nat)
  (different_board : Prop)

def knights_move (x y : Fin 64) : Prop :=
  (abs (x / 8 - y / 8) = 2 ∧ abs (x % 8 - y % 8) = 1) ∨
  (abs (x / 8 - y / 8) = 1 ∧ abs (x % 8 - y % 8) = 2)

def kings_move (x y : Fin 64) : Prop :=
  abs (x / 8 - y / 8) ≤ 1 ∧ abs (x % 8 - y % 8) ≤ 1 ∧ (x ≠ y)

-- Theorem stating the proof problem
theorem knight_king_moves_incompatible (vlad_board gosha_board : Board) (h_board_diff: vlad_board.different_board):
  ¬ ∀ i j : Fin 64, (knights_move i j ↔ kings_move (vlad_board.numbering i) (vlad_board.numbering j)) :=
by {
  -- Skipping proofs with sorry
  sorry
}

end knight_king_moves_incompatible_l712_712631


namespace unique_log_constant_l712_712043

theorem unique_log_constant (a : ℝ) (h : a > 1) :
  (∃! c, ∀ x ∈ set.Icc a (3 * a), ∃ y ∈ set.Icc a (a ^ 3), real.logb a x + real.logb a y = c) → a = 3 :=
by
  sorry

end unique_log_constant_l712_712043


namespace piper_hits_ball_count_l712_712950

theorem piper_hits_ball_count 
    (tokens_to_pitches : ℕ → ℕ := λ n, n * 15)
    (macy_tokens : ℕ := 11)
    (piper_tokens : ℕ := 17)
    (macy_hits : ℕ := 50)
    (total_missed : ℕ := 315) :
    let total_pitches := tokens_to_pitches macy_tokens + tokens_to_pitches piper_tokens in
    let total_hits := total_pitches - total_missed in
    let piper_hits := total_hits - macy_hits in
    piper_hits = 55 :=
by
  sorry

end piper_hits_ball_count_l712_712950


namespace triangle_similarity_l712_712508

-- First, define the acute triangle and point P inside it.
variables {A B C P : Point}
variables {A1 B1 C1 A2 B2 C2 A3 B3 C3 : Point}

-- Assume the triangle ABC is acute and point P inside it
axiom acute_triangle (A B C : Point) : is_acute_triangle A B C
axiom point_inside_triangle (P A B C : Point) : inside_triangle P A B C

-- Define the perpendiculars and sequential triangle definitions.
axiom perpendicular_from_P_to_sides (P A B C A1 B1 C1 : Point) : 
  is_perpendicular_drop P A B C A1 B1 C1

axiom perpendicular_repeat_1 (P A1 B1 C1 A2 B2 C2 : Point) : 
  is_perpendicular_drop P A1 B1 C1 A2 B2 C2

axiom perpendicular_repeat_2 (P A2 B2 C2 A3 B3 C3 : Point) : 
  is_perpendicular_drop P A2 B2 C2 A3 B3 C3

-- State the theorem to be proven
theorem triangle_similarity (A B C P A1 B1 C1 A2 B2 C2 A3 B3 C3 : Point)
  (h1 : acute_triangle A B C) 
  (h2 : point_inside_triangle P A B C) 
  (h3 : perpendicular_from_P_to_sides P A B C A1 B1 C1)
  (h4 : perpendicular_repeat_1 P A1 B1 C1 A2 B2 C2)
  (h5 : perpendicular_repeat_2 P A2 B2 C2 A3 B3 C3) :
  similar_triangles A B C A3 B3 C3 :=
sorry

end triangle_similarity_l712_712508


namespace problem1_problem2_l712_712505

-- Definition of the trigonometric identity condition 1
def condition1 (A B C : Real) : Prop :=
  sin A ^ 2 + sin C ^ 2 - sin B ^ 2 - sin A * sin C = 0

-- Definition of the ratio condition 2
def condition2 (a c : Real) : Prop :=
  a / c = 3 / 2

-- Problem 1: Given condition1, prove that B = π/3
theorem problem1 (A B C : Real) (h1 : condition1 A B C) : B = π / 3 :=
by sorry

-- Problem 2: Given that B = π/3 and condition2, prove that tan C = √3 / 2
theorem problem2 (a c C : Real) (h2 : a / c = 3 / 2) (hB : B = π / 3) : tan C = sqrt 3 / 2 :=
by sorry

end problem1_problem2_l712_712505


namespace no_factors_of_p_l712_712738

open Polynomial

noncomputable def p : Polynomial ℝ := X^4 - 4 * X^2 + 16
noncomputable def optionA : Polynomial ℝ := X^2 + 4
noncomputable def optionB : Polynomial ℝ := X + 2
noncomputable def optionC : Polynomial ℝ := X^2 - 4*X + 4
noncomputable def optionD : Polynomial ℝ := X^2 - 4

theorem no_factors_of_p (h : Polynomial ℝ) : h ≠ p / optionA ∧ h ≠ p / optionB ∧ h ≠ p / optionC ∧ h ≠ p / optionD := by
  sorry

end no_factors_of_p_l712_712738


namespace f_of_1_l712_712798

noncomputable def f : ℝ → ℝ :=
λ x, if x >= 0 then real.sqrt x else x + 1

theorem f_of_1 : f 1 = 1 := 
by
  unfold f
  simp
  exact real.sqrt_one

end f_of_1_l712_712798


namespace sufficient_not_necessary_condition_l712_712783

variable (a b : ℝ)

theorem sufficient_not_necessary_condition (h : a > |b|) : a^2 > b^2 :=
by 
  sorry

end sufficient_not_necessary_condition_l712_712783


namespace football_league_matches_l712_712673

theorem football_league_matches (x : ℕ) (h : (x * (x - 1)) / 2 = 15) : 
  (1 / 2 : ℚ) * (x * (x - 1)) = 15 :=
by
  rw [←mul_div_assoc, mul_one, one_mul (x * (x - 1))] at h
  exact h

end football_league_matches_l712_712673


namespace geometry_problem_l712_712507

/-- In triangle ABC with AB > AC, the incircle touches BC at E. 
AE intersects the incircle again at another point D. 
There is a point F on AE different from E such that CE = CF. 
Extend CF to intersect BD at G. Prove that CF = GF. -/
theorem geometry_problem 
  {A B C E D F G : Point} 
  (h1 : AB > AC)
  (h2 : touches_incircle E BC)
  (h3 : AE.intersect_incircle_at D)
  (h4 : on_line F AE ∧ F ≠ E)
  (h5 : CE = CF)
  (h6 : extend CF BD = G) : 
  CF = GF := 
sorry

end geometry_problem_l712_712507


namespace false_statement_l712_712156

variables {Line Plane : Type}

-- Definitions of lines and planes
variables (m n : Line) (α β γ : Plane)

-- Definitions of perpendicular and parallel relations
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) (p1 p2 : Plane) : Prop := sorry

-- Conditions
axiom diff_lines (m n : Line) : m ≠ n
axiom diff_planes (α β γ : Plane) : α ≠ β ∧ β ≠ γ ∧ α ≠ γ 

-- The statement to prove
theorem false_statement (hα_γ : perp α γ) (hβ_γ : perp β γ) : ¬parallel α β :=
sorry

end false_statement_l712_712156


namespace smallest_sum_of_squares_l712_712995

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l712_712995


namespace cinema_seating_estimation_and_exact_l712_712582

variable (r s : ℕ) 

theorem cinema_seating_estimation_and_exact :
  (r = 28 ∧ s = 31) → (r * s ≈ 900 ∧ r * s = 868) :=
by
  sorry

end cinema_seating_estimation_and_exact_l712_712582


namespace Lara_age_in_10_years_l712_712140

theorem Lara_age_in_10_years (current_age: ℕ) (years_ago: ℕ) (years_from_now: ℕ) (age_years_ago: ℕ) (h1: current_age = age_years_ago + years_ago) (h2: age_years_ago = 9) (h3: years_ago = 7) (h4: years_from_now = 10) : current_age + years_from_now = 26 := 
by 
  rw [h2, h3] at h1
  rw [← h1, h4]
  exact rfl

end Lara_age_in_10_years_l712_712140


namespace jill_walking_probability_l712_712510

-- Definitions based on the problem conditions
def platforms := 15
def distance_between_adjacent := 80
def max_walk_distance := 320

-- Predicate to calculate if the distance is within the max walk distance
def within_max_walk_distance (start_platform end_platform : ℕ) : Prop :=
  (abs (end_platform - start_platform) * distance_between_adjacent) ≤ max_walk_distance

-- Main theorem statement
theorem jill_walking_probability :
  let p := 18 in
  let q := 35 in
  let total_scenarios := 210 in
  let valid_scenarios := 108 in
  let probability := valid_scenarios / total_scenarios in
  (nat.gcd p q = 1 ∧ p + q = 53) :=
by {
  sorry -- no proof is required
}

end jill_walking_probability_l712_712510


namespace find_a_plus_2b_l712_712810

-- defining the conditions
variables {a b : ℝ}

-- defining the system of equations
def eq1 : Prop := 2020 * a + 2024 * b = 2040
def eq2 : Prop := 2022 * a + 2026 * b = 2050
def eq3 : Prop := 2025 * a + 2028 * b = 2065

-- proof statement
theorem find_a_plus_2b (h1 : eq1) (h2 : eq2) (h3 : eq3) : a + 2 * b = 5 := 
sorry

end find_a_plus_2b_l712_712810


namespace Lara_age_10_years_from_now_l712_712134

theorem Lara_age_10_years_from_now (current_year_age : ℕ) (age_7_years_ago : ℕ)
  (h1 : age_7_years_ago = 9) (h2 : current_year_age = age_7_years_ago + 7) :
  current_year_age + 10 = 26 :=
by
  sorry

end Lara_age_10_years_from_now_l712_712134


namespace probability_absolute_value_l712_712756

noncomputable theory

-- Define the random variable and its normal distribution
def X : ℝ → ℝ := sorry -- X is normally distributed

axiom X_norm : ∀ x : ℝ, P(X ≤ x) = sorry -- X ~ N(4, σ^2)

-- Given conditions
axiom cond1 : X ∼ N(4, σ^2)
axiom cond2 : P(2 ≤ X ≤ 6) ≈ 0.6827

-- Reference data as axioms (approximations can be treated as axioms for practical purposes)
axiom ref1 : P(4 - σ ≤ X ≤ 4 + σ) ≈ 0.6827
axiom ref2 : P(4 - 2 * σ ≤ X ≤ 4 + 2 * σ) ≈ 0.9545
axiom ref3 : P(4 - 3 * σ ≤ X ≤ 4 + 3 * σ) ≈ 0.9973

-- The proof statement
theorem probability_absolute_value (σ : ℝ) : P(abs (X - 2) ≤ 4) = 0.84 :=
sorry

end probability_absolute_value_l712_712756


namespace age_ratio_in_years_l712_712615

variables (a b : ℕ)  -- Define current ages of Alan and Bella as natural numbers
variables (x : ℕ)    -- Define the number of years as a natural number

-- Conditions given in the problem
def condition1 := a - 3 = 2 * (b - 3)
def condition2 := a - 8 = 3 * (b - 8)

-- Statement to prove
theorem age_ratio_in_years (h1 : condition1) (h2 : condition2) : 
  ∃ x, (a + x : ℚ) / (b + x) = 3 / 2 ∧ x = 7 :=
by {
  sorry -- Proof is omitted, following instructions
}

end age_ratio_in_years_l712_712615


namespace product_of_numbers_with_squares_ending_in_25_l712_712246

theorem product_of_numbers_with_squares_ending_in_25 :
  (∏ n in Finset.filter (λ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 10 = 5)) (Finset.range 100), n) = 3849453751875 := by
  sorry

end product_of_numbers_with_squares_ending_in_25_l712_712246


namespace collinear_T_K_I_l712_712884

noncomputable def intersection (P Q : Set Point) : Point := sorry

variables (A P C Q M N I T K : Point)

-- Definitions based on conditions
def T_def : Point := intersection (line_through A P) (line_through C Q)
def K_def : Point := intersection (line_through M P) (line_through N Q)

-- Proof statement
theorem collinear_T_K_I :
  collinear ({T_def A P C Q, K_def M P N Q, I} : Set Point) := sorry

end collinear_T_K_I_l712_712884


namespace number_of_correct_statements_is_one_l712_712530

variables (l m n : Line) (α β : Plane) 

axiom parallel_line (l1 l2 : Line) : Prop
axiom perpendicular_line_plane (l : Line) (α : Plane) : Prop
axiom contained_line_plane (l : Line) (α : Plane) : Prop

axioms 
  (parallel_lm : parallel_line l m)
  (parallel_mn : parallel_line m n)
  (perpendicular_lalpha : perpendicular_line_plane l α)
  (parallel_mbeta : parallel_line m β)
  (perpendicular_alphabeta : perpendicular_line_plane α β)
  (perpendicular_malpha : perpendicular_line_plane m α)
  (perpendicular_nalpha : perpendicular_line_plane n α)
  (contained_malpha : contained_line_plane m α)
  (contained_nalpha : contained_line_plane n α)
  (perpendicular_lm : perpendicular_line_plane l m)
  (perpendicular_ln : perpendicular_line_plane l n)

theorem number_of_correct_statements_is_one : 
  ((∀ (l m n : Line) (α : Plane), 
    parallel_line l m → parallel_line m n → perpendicular_line_plane l α →
    perpendicular_line_plane n α) ∧ 
  (¬ (∀ (l m β α : Line), 
    parallel_line m β → perpendicular_line_plane α β → perpendicular_line_plane l α →
    perpendicular_line_plane l m)) ∧ 
  (¬ (∀ (m n : Line) (α : Plane), 
    contained_line_plane m α → contained_line_plane n α →
    perpendicular_line_plane l m → perpendicular_line_plane l n →
    perpendicular_line_plane l α)) ∧ 
  (¬ (∀ (l m n : Line) (α : Plane), 
    parallel_line l m → perpendicular_line_plane m α → 
    perpendicular_line_plane n α → 
    perpendicular_line_plane l n)) → 
  (1 = 1)) := 
sorry

end number_of_correct_statements_is_one_l712_712530


namespace polynomial_problem_l712_712879

theorem polynomial_problem
    (a b c : ℤ)
    (h1 : c ≠ 0)
    (h2 : (1 : ℤ)^3 + (a * (1 : ℤ)^2) + (b * (1 : ℤ)) + c = 0)
    (h3 : let f_roots := [root1, root2, root3] in 
          let g_roots := [root1^2, root2^2, root3^2] in 
          f_roots ≠ [] ∧ g_roots ≠ [] ∧
          (∀ froot ∈ f_roots, f froot = 0) ∧ 
          (∀ groot ∈ g_roots, g groot = 0)) :
    (a : ℤ)^(2013) + (b : ℤ)^(2013) + (c : ℤ)^(2013) = -1 := 
sorry

end polynomial_problem_l712_712879


namespace log_ordering_l712_712151

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_ordering (P Q R : ℝ) (h₁ : P = Real.log 3 / Real.log 2)
  (h₂ : Q = Real.log 2 / Real.log 3) (h₃ : R = Real.log (Real.log 2 / Real.log 3) / Real.log 2) :
  R < Q ∧ Q < P := by
  sorry

end log_ordering_l712_712151


namespace catalan_identity_l712_712932

theorem catalan_identity 
  (a : ℕ → ℕ → ℕ)
  (h : ∀ x y, (1 / (1 - x - y + 2 * x * y)) = ∑ p q, a p q * x^p * y^q) 
  (n : ℕ) :
  (-1)^n * a (2*n) (2*n+2) = 1 / (n + 1) * Nat.choose (2*n) n :=
by
  sorry

end catalan_identity_l712_712932


namespace socks_problem_l712_712128

/-
  Theorem: Given x + y + z = 15, 2x + 4y + 5z = 36, and x, y, z ≥ 1, 
  the number of $2 socks Jack bought is x = 4.
-/

theorem socks_problem
  (x y z : ℕ)
  (h1 : x + y + z = 15)
  (h2 : 2 * x + 4 * y + 5 * z = 36)
  (h3 : 1 ≤ x)
  (h4 : 1 ≤ y)
  (h5 : 1 ≤ z) :
  x = 4 :=
  sorry

end socks_problem_l712_712128


namespace fg_of_3_l712_712085

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 3 * x

theorem fg_of_3 : f (g 3) = -2 := by
  sorry

end fg_of_3_l712_712085


namespace vector_sum_correct_l712_712013

-- Define the three vectors
def v1 : ℝ × ℝ := (5, -3)
def v2 : ℝ × ℝ := (-4, 6)
def v3 : ℝ × ℝ := (2, -8)

-- Define the expected result
def expected_sum : ℝ × ℝ := (3, -5)

-- Define vector addition (component-wise)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- The theorem statement
theorem vector_sum_correct : vector_add (vector_add v1 v2) v3 = expected_sum := by
  sorry

end vector_sum_correct_l712_712013


namespace infinite_square_sequence_l712_712867

noncomputable def infinite_square_sequence_exists : Prop :=
  ∃ (a : ℕ → ℕ), (∀ n, ∃ k, a n = k^2) ∧ (∀ m n, m ≠ n → a m ≠ a n) ∧ (∀ n, ∃ k, (∑ i in Finset.range n, a i) = k^2)

theorem infinite_square_sequence :
  infinite_square_sequence_exists :=
sorry

end infinite_square_sequence_l712_712867


namespace midpoint_coordinates_sum_l712_712179

theorem midpoint_coordinates_sum (M A B : ℝ × ℝ) (hM : M = (5, 3)) (hA : A = (2, 2)) (hM_midpoint : M = ((fst B + fst A) / 2, (snd B + snd A) / 2)) :
  fst B + snd B = 12 :=
  sorry

end midpoint_coordinates_sum_l712_712179


namespace number_of_zeros_of_quadratic_l712_712601

variables {a b c : ℝ}

theorem number_of_zeros_of_quadratic (h₁ : a ≠ 0) (h₂ : a * c < 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
begin
  sorry
end

end number_of_zeros_of_quadratic_l712_712601


namespace mashed_potatoes_vs_tomatoes_l712_712718

theorem mashed_potatoes_vs_tomatoes :
  let m := 144
  let t := 79
  m - t = 65 :=
by 
  repeat { sorry }

end mashed_potatoes_vs_tomatoes_l712_712718


namespace total_water_intake_l712_712168

def morning_water : ℝ := 1.5
def afternoon_water : ℝ := 3 * morning_water
def evening_water : ℝ := 0.5 * afternoon_water

theorem total_water_intake : 
  (morning_water + afternoon_water + evening_water) = 8.25 :=
by
  sorry

end total_water_intake_l712_712168


namespace find_a_l712_712944

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then -(Real.log (-x) / Real.log 2) + a else 0

theorem find_a (a : ℝ) :
  (f a (-2) + f a (-4) = 1) → a = 2 :=
by
  sorry

end find_a_l712_712944


namespace sin_add_arcsin_arctan_l712_712725

theorem sin_add_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  Real.sin (a + b) = (2 + 3 * Real.sqrt 3) / 10 :=
by
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  sorry

end sin_add_arcsin_arctan_l712_712725


namespace volume_of_S_l712_712936

-- Define the region S in terms of the conditions
def region_S (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1.5 ∧ 
  abs x + abs y ≤ 1 ∧ 
  abs z ≤ 0.5

-- Define the volume calculation function
noncomputable def volume_S : ℝ :=
  sorry -- This is where the computation/theorem proving for volume would go

-- The theorem stating the volume of S
theorem volume_of_S : volume_S = 2 / 3 :=
  sorry

end volume_of_S_l712_712936


namespace commission_percentage_l712_712681

-- Define the given conditions
def cost_of_item : ℝ := 17
def observed_price : ℝ := 25.50
def desired_profit_percentage : ℝ := 0.20

-- Calculate the desired profit in dollars
def desired_profit : ℝ := desired_profit_percentage * cost_of_item

-- Calculate the total desired price for the distributor
def total_desired_price : ℝ := cost_of_item + desired_profit

-- Calculate the commission in dollars
def commission_in_dollars : ℝ := observed_price - total_desired_price

-- Prove that commission percentage taken by the online store is 20%
theorem commission_percentage :
  (commission_in_dollars / observed_price) * 100 = 20 := 
by
  -- This is the placeholder for the proof
  sorry

end commission_percentage_l712_712681


namespace angle_B_leq_60_l712_712862

variable {k r : ℝ}
axiom k_pos : k > 0
axiom r_pos : r > 0
axiom geom_seq : ∃ (a b c : ℝ), a = k ∧ b = kr ∧ c = kr^2 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem angle_B_leq_60 (A B C : ℝ) (a b c : ℝ)
  (h_geom_seq : a = k ∧ b = kr ∧ c = kr^2)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  B ≤ 60 := sorry

end angle_B_leq_60_l712_712862


namespace calculate_number_of_n_variable_Boolean_functions_evaluate_D_10_g_l712_712149

-- Definitions for part (1)
def n_variable_Boolean_function (n : ℕ) (f : Fin n → Bool) : Bool :=
  true -- just a placeholder; we only define the structure

def number_of_n_variable_Boolean_functions (n : ℕ) : ℕ :=
  2^(2^n)

theorem calculate_number_of_n_variable_Boolean_functions (n : ℕ) :
  ∃ num, num = number_of_n_variable_Boolean_functions n ∧ num = 2^(2^n) :=
  by
  use number_of_n_variable_Boolean_functions n
  sorry

-- Definitions for part (2)
def g (x : Fin 10 → Bool) : Bool :=
  ((1 + x 0 + (x 0 && x 1) + (x 0 && x 1 && x 2) + (x 0 && x 1 && x 2 && x 3)
  + (x 0 && x 1 && x 2 && x 3 && x 4) + (x 0 && x 1 && x 2 && x 3 && x 4 && x 5)
  + (x 0 && x 1 && x 2 && x 3 && x 4 && x 5 && x 6) + (x 0 && x 1 && x 2 && x 3 && x 4 && x 5 && x 6 && x 7)
  + (x 0 && x 1 && x 2 && x 3 && x 4 && x 5 && x 6 && x 7 && x 8)
  + (x 0 && x 1 && x 2 && x 3 && x 4 && x 5 && x 6 && x 7 && x 8 && x 9)) % 2 = 0)

def D_10 (f : Fin 10 → Bool) :=
  { x : Fin 10 → Bool | f x = false }

def size_of_D_10_g : ℕ :=
  2^9

def sum_of_D_10_g : ℕ :=
  28160

theorem evaluate_D_10_g (f : Fin 10 → Bool) :
  (size_of_D_10_g = 2^9) ∧ (sum_of_D_10_g = 28160) :=
  by
  sorry

end calculate_number_of_n_variable_Boolean_functions_evaluate_D_10_g_l712_712149


namespace collinearity_of_T_K_I_l712_712897

-- Definitions of the points and lines
variables {A P Q M N C I T K : Type} [Nonempty A] [Nonempty P] [Nonempty Q] 
  [Nonempty M] [Nonempty N] [Nonempty C] [Nonempty I]

-- Intersection points conditions
def intersect (l₁ l₂ : Type) : Type := sorry

-- Given conditions
def condition_1 : T = intersect (line A P) (line C Q) := sorry
def condition_2 : K = intersect (line M P) (line N Q) := sorry

-- Proof that T, K, and I are collinear
theorem collinearity_of_T_K_I : collinear {T, K, I} := by
  have h1 : T = intersect (line A P) (line C Q) := condition_1
  have h2 : K = intersect (line M P) (line N Q) := condition_2
  -- Further steps needed to prove collinearity
  sorry

end collinearity_of_T_K_I_l712_712897


namespace projection_of_vectors_l712_712354

open Real

theorem projection_of_vectors :
  let u := (5 : ℝ, 7 : ℝ)
  let v := (1 : ℝ, -3 : ℝ)
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_squared := v.1^2 + v.2^2
  let projection_scalar := dot_product / magnitude_squared
  let projection := (projection_scalar * v.1, projection_scalar * v.2)
  projection = (-8 / 5, 24 / 5) :=
by
  sorry

end projection_of_vectors_l712_712354


namespace relationship_among_a_b_c_l712_712382

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := 2^0.1
noncomputable def c : ℝ := 0.2^1.3

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  unfold a b c -- Unfold the definitions
  sorry -- Placeholder for the proof

end relationship_among_a_b_c_l712_712382


namespace find_radius_of_fourth_sphere_l712_712368

noncomputable def radius_of_fourth_sphere (r1 r2 r3 : ℝ) (angle : ℝ) (correct_radius : ℝ) : Prop :=
  -- Conditions
  r1 = 3 ∧ r2 = 3 ∧ r3 = 3 ∧ angle = real.pi / 3 ∧
  -- Question and Correct Answer
  r = correct_radius →

-- State the equivalent proof problem
theorem find_radius_of_fourth_sphere :
  radius_of_fourth_sphere 3 3 3 (real.pi / 3) (9 - 4 * real.sqrt 2) :=
by {
  sorry -- proof to be filled in
}

end find_radius_of_fourth_sphere_l712_712368


namespace P_at_2020_l712_712261

noncomputable def P : ℝ → ℝ := sorry  -- Define polynomial P(x)

axiom leading_coeff_P : leading_coeff P = 1
axiom degree_P : degree P = 10
axiom P_positive : ∀ x : ℝ, P x > 0
axiom neg_P_factors : ∃ factors : ℕ → (ℝ → ℝ), 
  ∀ i : ℕ, i < 5 → (factors i).irreducible ∧ (factors i) 2020 = -3 ∧ 
  ∏ i in finset.range 5, factors i = -P

theorem P_at_2020 : P 2020 = 243 :=
  sorry

end P_at_2020_l712_712261


namespace transform_parabola_l712_712234

theorem transform_parabola (f : ℝ → ℝ) (x : ℝ) :
  (f x = x^2 + 4x - 4) →
  ((λ y, f y - 3) (x + 2)) = (x + 4)^2 - 11 :=
by
  intro h,
  -- The proof would continue here
  sorry

end transform_parabola_l712_712234


namespace third_vertex_coordinates_l712_712626

theorem third_vertex_coordinates (x : ℝ) (h : 6 * |x| = 96) : x = 16 ∨ x = -16 :=
by
  sorry

end third_vertex_coordinates_l712_712626


namespace circle_passing_points_line_intersects_chord_l712_712029

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

theorem circle_passing_points : circle_equation 0 0 ∧ circle_equation (-2) 4 ∧ circle_equation 1 1 :=
by simp [circle_equation]

noncomputable def line_equation (x y m : ℝ) : Prop := 4*x + 3*y + m = 0

theorem line_intersects_chord (m : ℝ) :
  (-4/3) * 4 = -3 ∨ abs ((-1)*4 + 2*3 + m) / sqrt (4^2 + 3^2) = 1 ↔ m = -7 ∨ m = 3 :=
sorry

end circle_passing_points_line_intersects_chord_l712_712029


namespace journey_distance_l712_712286

theorem journey_distance :
  ∃ D : ℝ, (D / 42 + D / 48 = 10) ∧ D = 224 :=
by
  sorry

end journey_distance_l712_712286


namespace distance_from_dormitory_to_city_l712_712497

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h : (1/5) * D + (2/3) * D + 4 = D) : 
  D = 30 :=
sorry

end distance_from_dormitory_to_city_l712_712497


namespace distance_PQ_eq_l712_712774

-- Definition of points P and Q on the line y = kx + b
def point_P (x1 : ℝ) (k : ℝ) (b : ℝ) : ℝ × ℝ := (x1, k * x1 + b)
def point_Q (x2 : ℝ) (k : ℝ) (b : ℝ) : ℝ × ℝ := (x2, k * x2 + b)

-- Distance formula function definition
def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- The theorem to prove the distance between points P and Q
theorem distance_PQ_eq (x1 x2 k b : ℝ) :
  distance (point_P x1 k b) (point_Q x2 k b) = Real.abs (x1 - x2) * Real.sqrt (1 + k ^ 2) :=
by
  sorry -- Proof omitted

end distance_PQ_eq_l712_712774


namespace intersecting_lines_l712_712466

variables (a b c : Line)

-- Assume each pair of lines a, b, and c are skew lines (not parallel and lie in different planes).
axiom skew_lines : ∀ (a b c : Line), (¬Parallel a b) ∧ (¬Parallel b c) ∧ (¬Parallel a c) → 
    (¬coplanar a b) ∧ (¬coplanar b c) ∧ (¬coplanar a c)

theorem intersecting_lines : ∀ (a b c : Line), (¬Parallel a b) ∧ (¬Parallel b c) ∧ (¬Parallel a c) →
    ¬coplanar a b ∧ ¬coplanar b c ∧ ¬coplanar a c → ∃ (l : Line), ∀ p, p ∈ a ∧ p ∈ b ∧ p ∈ c → 
    ∃ (infinitely_many l : Line), ∀ q, q ∈ l →
    (q ∈ a ∧ q ∈ b ∧ q ∈ c) :=
by
  sorry

end intersecting_lines_l712_712466


namespace g_is_odd_l712_712124

def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  sorry

end g_is_odd_l712_712124


namespace geometric_seq_common_ratio_l712_712858

noncomputable def common_ratio (a1 an Sn : ℝ) : ℝ :=
  let q := -(an / a1)^(1 / real.log (1 - Sn * real.log an / (a1 * real.log Sn) / real.log a1))
  q

theorem geometric_seq_common_ratio (q : ℝ) (a1 an Sn : ℝ) (h1 : a1 = 2) (h2 : an = -64) (h3 : Sn = -42) :
  common_ratio a1 an Sn =  -2 :=
by
  sorry

end geometric_seq_common_ratio_l712_712858


namespace second_intersection_on_diagonal_AC_l712_712547

variables {A B C D K L M P : Type} [Square A B C D] 
  (hK : PointOnSegment K A B) 
  (hL : PointOnSegment L C D) 
  (hM : PointOnSegment M K L)

noncomputable def second_intersection AKM MLC : Type := sorry

theorem second_intersection_on_diagonal_AC 
  (h1 : Circumscribed AKM)
  (h2 : Circumscribed MLC) :
  Exists P, P ≠ M ∧ P ∈ AC ∧ second_intersection AKM MLC = P :=
sorry

end second_intersection_on_diagonal_AC_l712_712547


namespace range_of_a_l712_712458

theorem range_of_a 
  (h : ∀ x : ℝ, |x - 1| + |2x + 2| ≥ a^2 + (1 / 2) * a + 2) : -1 / 2 ≤ a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l712_712458


namespace expression_value_l712_712639

theorem expression_value (a b : ℕ) (h1 : a = 36) (h2 : b = 9) : (a + b)^2 - (b^2 + a^2) = 648 :=
by {
  rw [h1, h2],
  norm_num
}

end expression_value_l712_712639


namespace cost_for_5_dozen_apples_l712_712713

-- Define the cost for 4 dozen apples
def cost_for_4_dozen : ℝ := 31.20

-- Define the ratio (cost per dozen apples)
def cost_per_dozen (total_cost : ℝ) (dozens : ℝ) : ℝ := total_cost / dozens

-- Define the cost for n dozen apples
def cost_for_n_dozen (cost_per_dozen : ℝ) (n : ℝ) : ℝ := cost_per_dozen * n

-- Theorem stating the correct answer
theorem cost_for_5_dozen_apples : cost_for_n_dozen (cost_per_dozen cost_for_4_dozen 4) 5 = 39.00 :=
    sorry

end cost_for_5_dozen_apples_l712_712713


namespace elvis_squares_count_l712_712342

theorem elvis_squares_count :
  ∀ (total : ℕ) (Elvis_squares Ralph_squares squares_used_by_Ralph matchsticks_left : ℕ)
  (uses_by_Elvis_per_square uses_by_Ralph_per_square : ℕ),
  total = 50 →
  uses_by_Elvis_per_square = 4 →
  uses_by_Ralph_per_square = 8 →
  Ralph_squares = 3 →
  matchsticks_left = 6 →
  squares_used_by_Ralph = Ralph_squares * uses_by_Ralph_per_square →
  total = (Elvis_squares * uses_by_Elvis_per_square) + squares_used_by_Ralph + matchsticks_left →
  Elvis_squares = 5 :=
by
  sorry

end elvis_squares_count_l712_712342


namespace property_value_decrease_l712_712694

theorem property_value_decrease (P : ℝ) (h : P > 0) :
  let increased_value := 1.30 * P in
  let decreased_value := increased_value * (1 - 3 / 13) in
  decreased_value = P :=
by
  sorry

end property_value_decrease_l712_712694


namespace find_y_l712_712981

theorem find_y (y : ℝ) (h : (15 + 25 + y) / 3 = 23) : y = 29 :=
sorry

end find_y_l712_712981


namespace volume_of_convex_polyhedron_l712_712763

variables {S1 S2 S : ℝ} {h : ℝ}

theorem volume_of_convex_polyhedron (S1 S2 S h : ℝ) :
  (h > 0) → (S1 ≥ 0) → (S2 ≥ 0) → (S ≥ 0) →
  ∃ V, V = (h / 6) * (S1 + S2 + 4 * S) :=
by {
  sorry
}

end volume_of_convex_polyhedron_l712_712763


namespace find_k_value_l712_712345

-- Definition of the integral calculation
def integral_eq_4 (k : ℝ) := ∫ x in 1..2, (2 * x + k) = 4

-- The theorem we need to prove
theorem find_k_value : ∃ (k : ℝ), integral_eq_4 k ∧ k = 1 :=
by
  -- Skipping the proof with sorry
  sorry

end find_k_value_l712_712345


namespace product_def_zero_l712_712728

noncomputable theory

def cos_root_poly (x : ℝ) : ℂ := x^3 + 0 * x^2 + (-1 : ℝ) * x + (-1 / 16 : ℝ)

theorem product_def_zero : 
  let Q := cos_root_poly in
  (roots : list ℂ) -- where roots are the roots of cos_root_poly
  (root1 = complex.cos (π/9))
  (root2 = complex.cos (2 * π / 9))
  (root3 = complex.cos (4 * π / 9))
  def d := 0
  def e := -1
  def f := -1 / 16 
  d * e * f = 0 :=
  sorry

end product_def_zero_l712_712728


namespace inequality_p_l712_712271

theorem inequality_p (p : ℝ) : 
  (∀ (x1 x2 x3 : ℝ), x1^2 + x2^2 + x3^2 ≥ p * (x1 * x2 + x2 * x3)) ↔ p ≤ Real.sqrt 2 := 
sorry

end inequality_p_l712_712271


namespace triangle_line_concurrency_special_case_orthocenter_circumcenter_special_case_parallel_lines_l712_712839

theorem triangle_line_concurrency
  (A B C : Point) -- Vertices of the triangle
  (bisectorA bisectorB bisectorC : Line) -- Angle bisectors at each vertex
  (delta epsilon eta : ℝ) -- Equal angles with angle bisectors
  (lineA1 lineA2 lineB1 lineB2 lineC1 lineC2 : Line) -- Lines drawn through vertices making equal angles with angle bisectors
  (O1 O2 : Point) -- Points of intersection
  (h1 : ThreeLinesIntersectAt lineA1 lineB1 lineC1 O1) :
  ThreeLinesIntersectAt lineA2 lineB2 lineC2 O2 := sorry

-- Special Cases:
theorem special_case_orthocenter_circumcenter
  (triangle : Triangle)
  (intersect_point1 : Point) -- One intersection point is the orthocenter
  (h1 : intersect_point1 = orthocenter triangle) :
  (other_point : Point), other_point = circumcenter triangle := sorry

theorem special_case_parallel_lines
  (triangle : Triangle)
  (intersect_point1 : ∞) -- One intersection point is at infinity (implies three lines are parallel)
  (h1 : ThreeLinesIntersectAtInfinity lineA1 lineB1 lineC1) :
  ThreeLinesIntersectOnCircumcircle triangle lineA2 lineB2 lineC2 := sorry

end triangle_line_concurrency_special_case_orthocenter_circumcenter_special_case_parallel_lines_l712_712839


namespace find_x_value_l712_712412

theorem find_x_value
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1/y = 15) (h2 : y + 1/x = 7 / 20) (h3 : x * y = 2) :
  x = 10 :=
begin
  sorry
end

end find_x_value_l712_712412


namespace problem1_solution_problem2_solution_l712_712470

noncomputable def problem1 (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) : Real := 
  A

noncomputable def problem2 (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) : Real :=
  a

theorem problem1_solution (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) :
  problem1 a b A B h1 = Real.pi / 4 :=
sorry

theorem problem2_solution (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) :
  problem2 a b c A area h1 h2 h3 = Real.sqrt 2 :=
sorry

end problem1_solution_problem2_solution_l712_712470


namespace cos_angle_BAC_l712_712795

variables {V : Type*} [inner_product_space ℝ V]

def circumcenter_property (O A B C : V) : Prop :=
  2 • (O - A) + 3 • (O - B) + 4 • (O - C) = 0

theorem cos_angle_BAC (O A B C : V) (h : circumcenter_property O A B C) :
  ∃ θ : ℝ, cos (angle A B C) = 1 / 4 :=
sorry

end cos_angle_BAC_l712_712795


namespace number_B_expression_l712_712600

theorem number_B_expression (A B : ℝ) (h : A = B - (4/5) * B) : B = (A + B) / (4 / 5) :=
sorry

end number_B_expression_l712_712600


namespace unit_vector_perpendicular_l712_712220

/-- The unit vectors perpendicular to the vector (5, 12) are (-12/13, 5/13) and (12/13, -5/13). -/
theorem unit_vector_perpendicular (x y : ℝ) :
  (x = -12 / 13 ∧ y = 5 / 13) ∨ (x = 12 / 13 ∧ y = -5 / 13) →
  5 * x + 12 * y = 0 ∧ real.sqrt (x^2 + y^2) = 1 :=
by
  rintro (⟨hx, hy⟩ | ⟨hx, hy⟩);
  {
    rw [hx, hy];
    split;
    sorry,
  }

end unit_vector_perpendicular_l712_712220


namespace sum_solutions_eq_zero_l712_712154

noncomputable def f (x : ℝ) : ℝ := x^2 + 3*x + 2

noncomputable def f_inv (y : ℝ) : ℝ := 
  (-3 + Real.sqrt (1 + 4*y)) / 2

theorem sum_solutions_eq_zero :
  (∑ x in Finset.univ.filter (λ x : ℝ, f_inv(x) = f(1 / x) ∧ x ≠ 0), x) = 0 := 
by { sorry }

end sum_solutions_eq_zero_l712_712154


namespace polygon_non_acute_angles_l712_712668

variable {n : ℕ} (O : Point) (A : Fin n → Point)

-- Assume O lies inside the convex polygon A₁A₂...Aₙ
axiom O_in_convex_polygon (h : Point) : inside_convex_polygon h (A ∘ Fin.succ)

-- Prove that at least n - 1 of the angles ∠AᵢOAⱼ, where i, j are distinct, are not acute
theorem polygon_non_acute_angles :
  at_least_non_acute n O A (angles := λ i j => angle (A i) O (A j)) (n - 1) :=
sorry

end polygon_non_acute_angles_l712_712668


namespace sqrt_sum_eq_seven_l712_712575

theorem sqrt_sum_eq_seven (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
  sqrt (64 - x^2) + sqrt (36 - x^2) = 7 := by
  sorry

end sqrt_sum_eq_seven_l712_712575


namespace determinant_of_cos_tan_angles_eq_zero_l712_712536

theorem determinant_of_cos_tan_angles_eq_zero (A B C : ℝ) 
  (h1 : ∃ k: ℕ, A + B + C = k * π) :
  det ![
    ![Real.cos A ^ 2, Real.tan A, 1],
    ![Real.cos B ^ 2, Real.tan B, 1],
    ![Real.cos C ^ 2, Real.tan C, 1]
  ] = 0 := 
sorry

end determinant_of_cos_tan_angles_eq_zero_l712_712536


namespace system_of_equations_solution_l712_712608

theorem system_of_equations_solution (x y : ℝ) (h1 : 4 * x + 3 * y = 11) (h2 : 4 * x - 3 * y = 5) :
  x = 2 ∧ y = 1 :=
by {
  sorry
}

end system_of_equations_solution_l712_712608


namespace smallest_sum_of_squares_l712_712991

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l712_712991


namespace exists_n_and_x_with_f_deriv_negative_l712_712147

noncomputable def f : ℝ → ℝ := sorry

axiom f_infinitely_differentiable : ∀ n : ℕ, differentiable ℝ (f^[n])

axiom f_at_0 : f 0 = 0

axiom f_at_1 : f 1 = 1

axiom f_nonnegative : ∀ x : ℝ, f x ≥ 0

theorem exists_n_and_x_with_f_deriv_negative :
  ∃ (n : ℕ) (x : ℝ), n > 0 ∧ (f^[n]) x < 0 :=
sorry

end exists_n_and_x_with_f_deriv_negative_l712_712147


namespace buicks_count_l712_712873

-- Definitions
def total_cars := 301
def ford_eqn (chevys : ℕ) := 3 + 2 * chevys
def buicks_eqn (chevys : ℕ) := 12 + 8 * chevys

-- Statement
theorem buicks_count (chevys : ℕ) (fords : ℕ) (buicks : ℕ) :
  total_cars = chevys + fords + buicks ∧
  fords = ford_eqn chevys ∧
  buicks = buicks_eqn chevys →
  buicks = 220 :=
by
  intros h
  sorry

end buicks_count_l712_712873


namespace running_percentage_l712_712279

-- Define the conditions
def total_runs := 138
def boundaries := 12
def sixes := 2

-- Calculate the runs from boundaries and sixes
def runs_from_boundaries := boundaries * 4  -- 12 * 4 = 48
def runs_from_sixes := sixes * 6            -- 2 * 6 = 12

-- Total runs from boundaries and sixes
def runs_from_boundaries_and_sixes := runs_from_boundaries + runs_from_sixes -- 48 + 12 = 60

-- Calculate runs made by running between the wickets
def runs_by_running := total_runs - runs_from_boundaries_and_sixes -- 138 - 60 = 78

-- Calculate the percentage of runs made by running between the wickets
def percentage_runs_by_running := (runs_by_running.toFloat / total_runs.toFloat) * 100

-- The final theorem statement to be proven
theorem running_percentage : abs (percentage_runs_by_running - 56.52) < 0.01 :=
sorry

end running_percentage_l712_712279


namespace unique_solution_quadratic_l712_712747

theorem unique_solution_quadratic (q : ℚ) :
  (∃ x : ℚ, q ≠ 0 ∧ q * x^2 - 16 * x + 9 = 0) ∧ (∀ y z : ℚ, (q * y^2 - 16 * y + 9 = 0 ∧ q * z^2 - 16 * z + 9 = 0) → y = z) → q = 64 / 9 :=
by
  sorry

end unique_solution_quadratic_l712_712747


namespace domain_of_ln_x_plus_1_plus_x_minus_2_zero_l712_712584

theorem domain_of_ln_x_plus_1_plus_x_minus_2_zero :
  {x : ℝ | x > -1 ∧ x ≠ 2 } = (-1, 2) ∪ (2, ∞) :=
by sorry

end domain_of_ln_x_plus_1_plus_x_minus_2_zero_l712_712584


namespace car_b_distance_doubled_speed_l712_712619

-- Define the conditions as hypotheses
variable (A B : Point)
variable (vA vB : ℝ)  -- initial speeds of Car A and Car B
variable (d_AB : ℝ) -- initial distance A to B
variable (midpoint_AB : Point := midpoint(A, B))

-- Assuming cars start from A and travel to B
variable (t : ℝ)  -- time for Car A to reach B
variable (distance_B_to_B : ℝ := 15) -- distance of Car B from B when Car A reaches B

-- Define the doubled speed scenario from the midpoint
variable (t_midpoint : ℝ := t / 2) -- time with doubled speed

theorem car_b_distance_doubled_speed :
  (distance_B_to_B = 15) :=
by
  -- restate the problem for clarity
  sorry

end car_b_distance_doubled_speed_l712_712619


namespace rectangle_perimeter_of_triangle_area_l712_712235

theorem rectangle_perimeter_of_triangle_area
  (h_right : ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a = 9 ∧ b = 12 ∧ c = 15)
  (rect_length : ℕ) 
  (rect_area_eq_triangle_area : ∃ (area : ℕ), area = 1/2 * 9 * 12 ∧ area = rect_length * rect_width ) 
  : ∃ (perimeter : ℕ), perimeter = 2 * (6 + rect_width) ∧ perimeter = 30 :=
sorry

end rectangle_perimeter_of_triangle_area_l712_712235


namespace collinearity_of_T_K_I_l712_712927

noncomputable def intersection_point (l1 l2 : Line) : Point := sorry

-- Definitions of lines AP, CQ, MP, NQ based on the problem context
variables {A P C Q M N I : Point} (lAP lCQ lMP lNQ : Line)
variables (T : Point) (K : Point)

-- Given conditions
def condition_1 : T = intersection_point lAP lCQ := sorry
def condition_2 : K = intersection_point lMP lNQ := sorry

-- Theorem statement
theorem collinearity_of_T_K_I : T ∈ line_through K I :=
by {
  -- These are the conditions that we're given in the problem
  have hT : T = intersection_point lAP lCQ := sorry,
  have hK : K = intersection_point lMP lNQ := sorry,
  -- Rest of the proof would go here
  sorry
}

end collinearity_of_T_K_I_l712_712927


namespace jeans_business_hours_l712_712871

theorem jeans_business_hours 
  (hours_per_day_weekday : ℕ) 
  (total_weekday_hours : 5 * hours_per_day_weekday)
  (total_weekend_hours : 2 * 4 = 8) 
  (total_weekly_hours : total_weekday_hours + total_weekend_hours = 38) :
  hours_per_day_weekday = 6 := 
by
  sorry

end jeans_business_hours_l712_712871


namespace S_2019_eq_l712_712854

theorem S_2019_eq : 
  let a3 := 4
  let a2_plus_a5 := 9
  let d := 1
  let a (n : ℕ) := n + 1
  let b (n : ℕ) := 1 / ((a n)^2 - 1)
  let S (n : ℕ) := ∑ i in Finset.range n, b i
  in a3 = 4 ∧ a2_plus_a5 = 9 ∧ d = 1 → S 2019 = 1 / 2 * (1 + 1 / 2 - 1 / 2020 - 1 / 2021) := 
by
  intros; sorry

end S_2019_eq_l712_712854


namespace sum_of_abs_values_ge_n_l712_712148

noncomputable def monic_integer_polynomial (p : Polynomial ℤ) (n : ℕ) : Prop :=
p.degree = n ∧ p.leading_coeff = 1

theorem sum_of_abs_values_ge_n
    (p : Polynomial ℤ) (q : Polynomial ℤ) (n : ℕ) (α : Fin n → ℝ)
    (h1 : monic_integer_polynomial p n)
    (h2 : ∀ i, p.eval (α i) = 0)
    (h3 : ∀ i, q.eval (α i) ≠ 0)
    (h4 : ∀ x, eval x p ∣ eval x q ↔ eval x p = 0)
    :
    ∑ i, |q.eval (α i)| ≥ n := 
sorry

end sum_of_abs_values_ge_n_l712_712148


namespace smallest_d_for_inverse_l712_712155

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse (d : ℝ) : 
  (∀ x1 x2 : ℝ, d ≤ x1 → d ≤ x2 → g x1 = g x2 → x1 = x2) → d = 3 :=
by
  sorry

end smallest_d_for_inverse_l712_712155


namespace T_2017_value_l712_712825

noncomputable def S (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | (n+1) => (n+1) * (n+2) / 2

noncomputable def T (n : ℕ) : ℚ :=
  if h: 2 ≤ n then
    let range_prod := ∏ i in (finset.range (n - 2)).image (λ x, x + 2), 
        (S i) / ((S i) - 1)
    in range_prod
  else 1 / 0 -- undefined for n < 2

theorem T_2017_value : T 2017 = 2017 / 673 := 
by {
  -- sorry for the actual proof
  sorry
}

end T_2017_value_l712_712825


namespace solve_quadratic_from_absolute_value_l712_712513

theorem solve_quadratic_from_absolute_value :
  (∃ (x : ℝ), |x - 8| = 3) →
  (∃ (b c : ℝ), ∀ (x : ℝ), (x - 11) * (x - 5) = x^2 + b * x + c) →
  (-16, 55) :=
by
  sorry

end solve_quadratic_from_absolute_value_l712_712513


namespace rectangle_area_problem_l712_712480

-- Define the conditions
def isDividedIntoSquares (rectangle: ℝ → ℝ → Prop) := 
  ∃ side_len : ℝ, rectangle (5 * side_len) side_len

def perimeterIs200 (rectangle: ℝ → ℝ → Prop) := 
  ∃ width height : ℝ, rectangle width height ∧ 2 * (width + height) = 200

-- Define rectangle property and area calculation
def isRectangleArea (width height area : ℝ) := area = width * height

-- Statement to be proved
theorem rectangle_area_problem : 
  ∃ width height area : ℝ, 
    isDividedIntoSquares (λ w h, w = width ∧ h = height) ∧ 
    perimeterIs200 (λ w h, w = width ∧ h = height) ∧ 
    isRectangleArea width height (12500 / 9) :=
by 
  sorry

end rectangle_area_problem_l712_712480


namespace total_sum_of_transformed_sums_eq_96_l712_712435

def setP : Finset ℕ := {1, 2, 3, 4, 5, 6}
def f (k : ℕ) : ℤ := (-1)^(k : ℤ) * k

theorem total_sum_of_transformed_sums_eq_96 :
  ∑ A in setP.powerset, 
    if A.nonempty then ∑ k in A, f k else 0 
  = 96 := 
sorry

end total_sum_of_transformed_sums_eq_96_l712_712435


namespace collinear_TKI_l712_712892

-- Definitions for points and lines
variables {A P C Q M N I T K : Type}
variable {line : Type → Type}
variables (AP : line A → line P) (CQ : line C → line Q) (MP : line M → line P) (NQ : line N → line Q)

-- Conditions from the problem
-- Assume there exist points T and K which are intersections of the specified lines
axiom intersects_AP_CQ : ∃ (T : Type), AP T = CQ T
axiom intersects_MP_NQ : ∃ (K : Type), MP K = NQ K

-- Collinearity of points T, K, and I
theorem collinear_TKI : ∀ (I : Type) (T : Type) (K : Type),
  intersects_AP_CQ → intersects_MP_NQ → collinear I T K :=
by sorry

end collinear_TKI_l712_712892


namespace factorize_quadratic1_factorize_quadratic2_l712_712239

theorem factorize_quadratic1 {a b c : ℝ} (h : a ≠ 0) :
  (3 * (x - (1 + real.sqrt 2)) * (x - (1 - real.sqrt 2))) = (3 * x^2 - x - 1) :=
sorry

theorem factorize_quadratic2 {a b c : ℝ} (h : a ≠ 0) :
  (2 * (x - (4 + real.sqrt 22) / 2) * (x - (4 - real.sqrt 22) / 2)) = (2 * x^2 - 8 * x - 3) :=
sorry

end factorize_quadratic1_factorize_quadratic2_l712_712239


namespace count_distinct_integer_sums_of_special_fractions_l712_712317

theorem count_distinct_integer_sums_of_special_fractions : 
  let special_fractions := {frac : ℚ | ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a + b = 18 ∧ frac = a / b} in
  let special_sum := {x + y | x y : ℚ, x ∈ special_fractions ∧ y ∈ special_fractions} in
  fintype.card {n : ℤ | ∃ x ∈ special_sum, x = n} = 6 :=
by
  sorry

end count_distinct_integer_sums_of_special_fractions_l712_712317


namespace probability_three_children_one_girl_eldest_l712_712678

theorem probability_three_children_one_girl_eldest :
  let p_boy_or_girl := (1 / 2)
  in p_boy_or_girl * p_boy_or_girl * p_boy_or_girl = 1 / 8 :=
by
  let p_boy_or_girl := (1 / 2)
  sorry

end probability_three_children_one_girl_eldest_l712_712678


namespace savings_amount_correct_l712_712193

-- Define the problem conditions
def principal := 150000.0
def total_expenses := 182200.0

-- Define individual expenses
def airplane_tickets := 61200.0
def accommodation := 65000.0
def food := 36000.0
def excursions := 20000.0

-- Define interest rates
def bettaBank_rate := 0.036
def gammaBank_rate := 0.045
def omegaBank_rate := 0.0312
def epsilonBank_monthly_rate := 0.0025

-- Define compounding periods and time in years
def bettaBank_n := 12
def gammaBank_n := 1
def omegaBank_n := 4
def epsilonBank_n := 12
def t := 0.5

-- Calculate interests and future values
def bettaBank_interest := principal * (1 + bettaBank_rate / bettaBank_n) ^ (bettaBank_n * t) - principal
def gammaBank_interest := principal * (1 + gammaBank_rate * t) - principal
def omegaBank_interest := principal * (1 + omegaBank_rate / omegaBank_n) ^ (omegaBank_n * t) - principal
def epsilonBank_interest := principal * (1 + epsilonBank_monthly_rate) ^ epsilonBank_n / 2 - principal

-- Calculate the required savings from salary
def bettaBank_savings := total_expenses - principal - bettaBank_interest
def gammaBank_savings := total_expenses - principal - gammaBank_interest
def omegaBank_savings := total_expenses - principal - omegaBank_interest
def epsilonBank_savings := total_expenses - principal - epsilonBank_interest

-- The proof problem
theorem savings_amount_correct :
  bettaBank_savings = 29479.67 ∧
  gammaBank_savings = 28825 ∧
  omegaBank_savings = 29850.87 ∧
  epsilonBank_savings = 29935.89 :=
by
  sorry

end savings_amount_correct_l712_712193


namespace quadratic_roots_are_correct_l712_712568

theorem quadratic_roots_are_correct (x: ℝ) : 
    (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2) ∨ (x = (-1 - Real.sqrt 5) / 2) := 
by sorry

end quadratic_roots_are_correct_l712_712568


namespace length_of_DF_l712_712167

theorem length_of_DF
  (D E F P Q: Type)
  (DP: ℝ)
  (EQ: ℝ)
  (h1: DP = 27)
  (h2: EQ = 36)
  (perp: ∀ (u v: Type), u ≠ v):
  ∃ (DF: ℝ), DF = 4 * Real.sqrt 117 :=
by
  sorry

end length_of_DF_l712_712167


namespace complex_number_in_fourth_quadrant_l712_712044

theorem complex_number_in_fourth_quadrant (a : ℝ) (h1 : (a^2 - 1) = 0) (h2 : a + 1 ≠ 0) :
  let z := complex.mk a (a - 2) in
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_number_in_fourth_quadrant_l712_712044


namespace extremum_derivative_zero_l712_712420

theorem extremum_derivative_zero {f : ℝ → ℝ} (h_diff : differentiable ℝ f) (x₀ : ℝ) :
  (∃ (h_extremum : ∀ x, f x₀ ≤ f x ∨ f x₀ ≥ f x, fderiv ℝ f x₀ = 0) → (∀ y, fderiv ℝ f y = 0 → ∀ z, ¬f y₀ ≤ f z ∨ f y₀ ≥ f z)) :=
sorry

end extremum_derivative_zero_l712_712420


namespace find_unknown_rate_l712_712689

theorem find_unknown_rate :
  ∀ (x : ℝ), (3 * 100 + 2 * 150 + 2 * x = 7 * 150) → x = 225 :=
by
  intro x
  assume h
  sorry

end find_unknown_rate_l712_712689


namespace area_of_locus_l712_712119

theorem area_of_locus (PQ PR QR : ℝ) (h1 : PQ = 4) (h2 : PR = 7) (h3 : QR = 9) (MP MQ MR : PQR → ℝ)
  (inside_triangle : Π (M : PQR), MP M ^ 2 + MQ M ^ 2 + MR M ^ 2 ≤ 50) : 
  ∃ (S : ℝ), S = 4 * Real.pi / 9 :=
by
  use (4 * Real.pi / 9)
  sorry

end area_of_locus_l712_712119


namespace sum_inequality_l712_712389

open Real

theorem sum_inequality
  (n : ℕ) (x : Fin n → ℝ) 
  (hn : n > 2) 
  (h_nonneg : ∀ i, 0 ≤ x i) 
  (h_sum : ∑ i, x i = 1) :
  (∑ i j in Finset.range n, i < j, (x i * x j) / ((1 + (n-2) * x i) * (1 + (n-2) * x j)) ≤ n / (8 * (n-1))) :=
by
  sorry

end sum_inequality_l712_712389


namespace sum_T_n_l712_712031

def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n-1)

def b_n (n : ℕ) : ℕ :=
  if n = 1 then 0 else (n-1)*2^(n-1)

def T_n (n : ℕ) : ℕ :=
  ∑ k in Finset.range n, b_n (k + 1)

theorem sum_T_n (n : ℕ) : T_n n = 2 + (n-2)*2^n := sorry

end sum_T_n_l712_712031


namespace smallest_sum_of_squares_l712_712994

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l712_712994


namespace tan_alpha_minus_pi_over_4_l712_712438

open Real

theorem tan_alpha_minus_pi_over_4
  (α : ℝ)
  (a b : ℝ × ℝ)
  (h1 : a = (cos α, -2))
  (h2 : b = (sin α, 1))
  (h3 : ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) :
  tan (α - π / 4) = -3 := 
sorry

end tan_alpha_minus_pi_over_4_l712_712438


namespace factorial_mod_11_l712_712753

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_11 : (factorial 13) % 11 = 0 := by
  sorry

end factorial_mod_11_l712_712753


namespace probability_at_least_30_cents_and_penny_halfdollar_heads_l712_712976

noncomputable def prob_at_least_30_cents_heads : ℚ :=
  let outcomes := {outcome | ∃ (penny nickel dime quarter half_dollar : bool),
    outcome = (penny && nickel && dime && quarter && half_dollar)} in
  let successful_outcomes := {outcome ∈ outcomes |
    let (penny, _, _, _, half_dollar) := outcome in
    penny ∧ half_dollar ∧
    let (nickel, dime, quarter) := outcome in
    (if penny then 1 else 0) + (if nickel then 5 else 0) + 
    (if dime then 10 else 0) + (if quarter then 25 else 0) + 
    (if half_dollar then 50 else 0) ≥ 30 } in
  (successful_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_at_least_30_cents_and_penny_halfdollar_heads :
  prob_at_least_30_cents_heads = 1 / 4 :=
sorry

end probability_at_least_30_cents_and_penny_halfdollar_heads_l712_712976


namespace ratio_c_a_l712_712514

-- Definitions based on the conditions stated
variables {α : Type*} [EuclideanGeometry α] 
variables (A B C D E F : α)
variables (a b c : ℝ)
variables {ω1 ω2 ω3 ω4 : Circle α}

-- Given conditions
axiom angle_ABC_90 : ∠B A C = 90
axiom square_inscribed_in_triangle : square_inscribed_in_triangle B D E F A B C
axiom inradius_EFC_c : inradius (triangle E F C) = c
axiom inradius_EDA_b : inradius (triangle E D A) = b
axiom radius_ω1 : radius ω1 = b
axiom radius_ω2 : radius ω2 = a
axiom radius_ω3 : radius ω3 = b
axiom radius_ω4 : radius ω4 = a
axiom ω1_tangent_ED : tangent ω1 (line_through E D)
axiom ω2_tangent_EF : tangent ω2 (line_through E F)
axiom ω3_tangent_BF : tangent ω3 (line_through B F)
axiom ω4_tangent_BD : tangent ω4 (line_through B D)
axiom tangency_circles : ∀ (i j : nat) (hij : i ≠ j), tangent (circles[i]) (circles[j])

-- Proof goal
theorem ratio_c_a : c = 2 * a :=
sorry

end ratio_c_a_l712_712514


namespace simplify_complex_expression_l712_712185

open Complex

theorem simplify_complex_expression : (1 + 2 * I) / I = -2 + I :=
by
  sorry

end simplify_complex_expression_l712_712185


namespace g_at_neg_two_l712_712451

def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 3)

theorem g_at_neg_two : g (-2) = 4 / 5 := by
  -- the proof is omitted here
  sorry

end g_at_neg_two_l712_712451


namespace normal_char_function_correct_cauchy_char_function_correct_l712_712964

-- Definitions based on given conditions
def char_fn_normal (t : ℝ) : ℝ := Real.exp (-t^2 / 2)
def char_fn_cauchy (t θ : ℝ) : ℝ := Real.exp (-θ * Real.abs t)

-- Proof for normal distribution characteristic function
theorem normal_char_function_correct (t : ℝ) :
  char_fn_normal t = Real.exp (-t^2 / 2) :=
by sorry

-- Proof for Cauchy distribution characteristic function
theorem cauchy_char_function_correct (t θ : ℝ) :
  char_fn_cauchy t θ = Real.exp (-θ * Real.abs t) :=
by sorry


end normal_char_function_correct_cauchy_char_function_correct_l712_712964


namespace find_the_number_l712_712464

theorem find_the_number :
  ∃ (x : ℕ), abs ((x + 121 * 3.125) / 121 - 3.38) < 0.01 ∧ x = 31 := sorry

end find_the_number_l712_712464


namespace prime_9_greater_than_perfect_square_l712_712641

theorem prime_9_greater_than_perfect_square (p : ℕ) (hp : Nat.Prime p) :
  ∃ n m : ℕ, p - 9 = n^2 ∧ p + 2 = m^2 ∧ p = 23 :=
by
  sorry

end prime_9_greater_than_perfect_square_l712_712641


namespace remainder_of_7_pow_51_mod_8_l712_712638

theorem remainder_of_7_pow_51_mod_8 : (7^51 % 8) = 7 := sorry

end remainder_of_7_pow_51_mod_8_l712_712638


namespace collinearity_of_points_l712_712910

noncomputable theory
open_locale classical

variables {A P Q M N I T K : Type*}

-- Conditions given in the problem
variables [IncidenceGeometry A P Q M N I T K]
variable [IntersectionPoint T (Line A P) (Line C Q)]
variable [IntersectionPoint K (Line M P) (Line N Q)]

-- Statement of the proof problem
theorem collinearity_of_points :
  Collinear {T, K, I} :=
sorry

end collinearity_of_points_l712_712910


namespace D_correct_l712_712196
noncomputable def D (n : ℕ) : ℝ :=
  match n with
  | 2 => 2
  | 3 => real.sqrt 3
  | 4 => real.sqrt 2
  | 5 => 2 * real.sin (real.pi / 5)
  | 6 => 1
  | 7 => 1
  | _ => 0 -- This handle cases that are not relevant

theorem D_correct (n : ℕ) (hn : n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7) :
  D n = if n = 2 then 2
  else if n = 3 then real.sqrt 3
  else if n = 4 then real.sqrt 2
  else if n = 5 then 2 * real.sin (real.pi / 5)
  else if n = 6 then 1
  else if n = 7 then 1
  else 0 :=
by sorry

end D_correct_l712_712196


namespace students_catching_up_on_homework_l712_712490

def total_students : ℕ := 24
def silent_reading_students : ℕ := total_students / 2
def board_games_students : ℕ := total_students / 3

theorem students_catching_up_on_homework : 
  total_students - (silent_reading_students + board_games_students) = 4 := by
  sorry

end students_catching_up_on_homework_l712_712490


namespace collinear_T_K_I_l712_712888

noncomputable def intersection (P Q : Set Point) : Point := sorry

variables (A P C Q M N I T K : Point)

-- Definitions based on conditions
def T_def : Point := intersection (line_through A P) (line_through C Q)
def K_def : Point := intersection (line_through M P) (line_through N Q)

-- Proof statement
theorem collinear_T_K_I :
  collinear ({T_def A P C Q, K_def M P N Q, I} : Set Point) := sorry

end collinear_T_K_I_l712_712888


namespace distance_between_A_and_B_l712_712509

-- Define the conditions as constants
constant A : Type
constant B : Type
constant C : Type

constant distance_AC : ℝ := 3  -- Distance from A to C is 3 km
constant distance_BC : ℝ := 3  -- Distance from B to C is 3 km
constant angle_ACB : ℝ := 120  -- Angle ∠ACB is 120 degrees

-- Define and prove the distance between A and B
theorem distance_between_A_and_B :
  let cos_angle_ACB := -1/2 in
  let distance_squared := distance_AC^2 + distance_BC^2 - 2 * distance_AC * distance_BC * cos_angle_ACB in
  sqrt distance_squared = 3 * sqrt 3 :=
by
  sorry

end distance_between_A_and_B_l712_712509


namespace ellipse_eccentricity_proof_l712_712431

noncomputable def ellipse_eccentricity_range {a b x y c : ℝ} (F1 F2 : ℝ) (P : ℝ × ℝ) (h1 : x^2 / a^2 + y^2 / b^2 = 1)
    (h2 : ∃ P : ℝ × ℝ, |P.1 - F1| = 3 * |P.2 - F2|) : ℝ := 
{e : ℝ // 0 < e ∧ e < 1 ∧ 1 / 2 ≤ e}

theorem ellipse_eccentricity_proof : 
  ∀ (a b c F1 F2 : ℝ) (P : ℝ × ℝ),
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
    (∃ P : ℝ × ℝ, |P.1 - F1| = 3 * |P.2 - F2|) →
    ∀ e : ℝ, ∃ (e ∈ ellipse_eccentricity_range F1 F2 P) := 
by sorry

end ellipse_eccentricity_proof_l712_712431


namespace citizen_wealth_ratio_l712_712327

-- Define the conditions as variables first
variables {x y z w : ℝ} (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < w)

-- Define W and P as the total world wealth and population, respectively
noncomputable def world_wealth : ℝ := sorry
noncomputable def world_population : ℝ := sorry

-- Introudce the lean statement that needs to be proved
theorem citizen_wealth_ratio 
  (hxA : 0 < x) (hyA : 0 < y) (hzB : 0 < z) (hwB : 0 < w) 
  (W : ℝ) (P : ℝ) : 
  let wealth_per_citizenA := (y * W) / (x * P),
      subsidyA := 0.1 * wealth_per_citizenA,
      total_wealth_per_citizenA := wealth_per_citizenA + subsidyA, -- 1.1 * wealth_per_citizenA
      wealth_per_citizenB := (w * W) / (z * P)
  in total_wealth_per_citizenA / wealth_per_citizenB = 1.1 * (y * z) / (x * w) :=
by
  sorry

end citizen_wealth_ratio_l712_712327


namespace cone_volume_l712_712055

theorem cone_volume (r h l : ℝ) (π := Real.pi)
  (slant_height : l = 5)
  (lateral_area : π * r * l = 20 * π) :
  (1 / 3) * π * r^2 * h = 16 * π :=
by
  -- Definitions based on conditions
  let slant_height_definition := slant_height
  let lateral_area_definition := lateral_area
  
  -- Need actual proof steps which are omitted using sorry
  sorry

end cone_volume_l712_712055


namespace exists_infinitely_many_triplets_l712_712556

theorem exists_infinitely_many_triplets :
  ∃ᶠ (p q1 q2 : ℕ) in at_top, p > 0 ∧ q1 > 0 ∧ q2 > 0 ∧ (| (Real.sqrt 2 - q1 / p) * (Real.sqrt 3 - q2 / p) | ≤ 1 / (2 * p ^ 3)) := by
  sorry

end exists_infinitely_many_triplets_l712_712556


namespace possible_k_boxes_l712_712647

-- Defining the problem
def box_circle_operation (n : ℕ) (k : ℕ) (box_contents : Fin n → ℕ) : Prop :=
  ∃ (num_operations : ℕ), 
    (∀ op_idx ∈ Fin n, has_consecutive_triple op_idx →
      (all_boxes_have_k_balls (perform_operations num_operations box_contents) k))

-- Main theorem statement
theorem possible_k_boxes (n : ℕ) (h : n ≥ 3) : 
  ∀ k : ℕ, ∃ box_contents : Fin n → ℕ, box_circle_operation n k box_contents ↔ 
    (k % 3 = 1 ∨ k % 3 = 0) :=
by
  sorry

end possible_k_boxes_l712_712647


namespace cost_per_dvd_l712_712165

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) 
  (h1 : total_cost = 4.8) (h2 : num_dvds = 4) : (total_cost / num_dvds) = 1.2 :=
by
  sorry

end cost_per_dvd_l712_712165


namespace savings_needed_for_vacation_l712_712194

-- Define the conditions
def available_funds : ℝ := 150000
def total_vacation_cost : ℝ := 182200

def betta_bank_rate : ℝ := 0.036
def gamma_bank_rate : ℝ := 0.045
def omega_bank_rate : ℝ := 0.0312
def epsilon_bank_rate : ℝ := 0.0025

def betta_bank_interest (P : ℝ) (r : ℝ) (n t : ℝ) := P * (1 + r / n)^(n * t)
def gamma_bank_interest (P : ℝ) (r t : ℝ) := P * (1 + r * t)
def omega_bank_interest (P : ℝ) (r : ℝ) (n t : ℝ) := P * (1 + r / n)^(n * t)
def epsilon_bank_interest (P : ℝ) (r n t : ℝ) := P * (1 + r)^(n * t)

-- Define the final amounts after 6 months for each bank
def betta_bank_amount := betta_bank_interest available_funds betta_bank_rate 12 0.5
def gamma_bank_amount := gamma_bank_interest available_funds gamma_bank_rate 0.5
def omega_bank_amount := omega_bank_interest available_funds omega_bank_rate 4 0.5
def epsilon_bank_amount := epsilon_bank_interest available_funds epsilon_bank_rate 1 6

-- Define the interest earned for each bank
def betta_bank_interest_earned := betta_bank_amount - available_funds
def gamma_bank_interest_earned := gamma_bank_amount - available_funds
def omega_bank_interest_earned := omega_bank_amount - available_funds
def epsilon_bank_interest_earned := epsilon_bank_amount - available_funds

-- Define the amounts that need to be saved from salary for each bank
def betta_bank_savings_needed := total_vacation_cost - available_funds - betta_bank_interest_earned
def gamma_bank_savings_needed := total_vacation_cost - available_funds - gamma_bank_interest_earned
def omega_bank_savings_needed := total_vacation_cost - available_funds - omega_bank_interest_earned
def epsilon_bank_savings_needed := total_vacation_cost - available_funds - epsilon_bank_interest_earned

-- The theorem to be proven
theorem savings_needed_for_vacation :
  betta_bank_savings_needed = 29479.67 ∧
  gamma_bank_savings_needed = 28825 ∧
  omega_bank_savings_needed = 29850.87 ∧
  epsilon_bank_savings_needed = 29935.89 :=
by sorry

end savings_needed_for_vacation_l712_712194


namespace smallest_sum_of_squares_l712_712997

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l712_712997


namespace M_union_N_eq_l712_712808

open Set

def M : Set ℝ := { x | x^2 - 4 * x < 0 }
def N : Set ℝ := { x | abs x ≤ 2 }

theorem M_union_N_eq : M ∪ N = Ico (-2 : ℝ) 4 := by
  sorry

end M_union_N_eq_l712_712808


namespace varies_fix_l712_712829

variable {x y z : ℝ}

theorem varies_fix {k j : ℝ} 
  (h1 : x = k * y^4)
  (h2 : y = j * z^(1/3)) : x = (k * j^4) * z^(4/3) := by
  sorry

end varies_fix_l712_712829


namespace Matt_completes_work_in_100_days_l712_712259

variable (Matt Peter : ℕ → ℝ)

-- Condition 1: Matt and Peter together can do a piece of work in 20 days
def combined_rate : ℝ := 1 / 20

-- Condition 2: They work together for 12 days
def work_together_12_days (Matt Peter : ℕ → ℝ) : ℝ := 12 * combined_rate

-- Condition 3: Peter stops, and Matt completes the remaining work in 10 days
def remaining_work_after_12_days : ℝ := 1 - work_together_12_days Matt Peter
def Peter_rate : ℝ := remaining_work_after_12_days / 10

-- Define Matt's rate
def Matt_rate : ℝ := combined_rate - Peter_rate

-- The required proof statement
theorem Matt_completes_work_in_100_days :
  1 / Matt_rate = 100 := by
sorry

end Matt_completes_work_in_100_days_l712_712259


namespace max_xy_l712_712934

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy <= 81 :=
by {
  sorry
}

end max_xy_l712_712934


namespace tangent_beta_l712_712828

variables {α β : ℝ}

theorem tangent_beta (h1 : tan α = 1 / 3) (h2 : tan (α + β) = 1 / 2) : tan β = 1 / 7 :=
by
  sorry

end tangent_beta_l712_712828


namespace john_correct_answers_l712_712478

theorem john_correct_answers (x : ℕ) (total_questions : ℕ) (attempted_questions : ℕ) 
    (unanswered_points : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (minimum_score : ℕ) :
  total_questions = 25 →
  attempted_questions = 20 →
  unanswered_points = 2 →
  correct_points = 8 →
  incorrect_points = -2 →
  minimum_score = 150 →
  2 * (total_questions - attempted_questions) + correct_points * x + incorrect_points * (attempted_questions - x) ≥ minimum_score ↔
  x ≥ 18 :=
by
  sorry

end john_correct_answers_l712_712478


namespace max_value_fractions_l712_712940

noncomputable def maxFractions (a b c : ℝ) : ℝ :=
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c)

theorem max_value_fractions (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
    (h_sum : a + b + c = 2) :
    maxFractions a b c ≤ 1 ∧ 
    (a = 2 / 3 ∧ b = 2 / 3 ∧ c = 2 / 3 → maxFractions a b c = 1) := 
  by
    sorry

end max_value_fractions_l712_712940


namespace find_S20_l712_712390

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def S : ℕ → ℝ := sorry

axiom a_nonzero (n : ℕ) : a_seq n ≠ 0
axiom a1_eq : a_seq 1 = 1
axiom Sn_eq (n : ℕ) : S n = (a_seq n * a_seq (n + 1)) / 2

theorem find_S20 : S 20 = 210 := sorry

end find_S20_l712_712390


namespace inf_natural_numbers_l712_712145

def s (F : Finset ℤ) : Finset ℤ :=
  {a | (a ∈ F ∨ a - 1 ∈ F) ∧ ¬ (a ∈ F ∧ a - 1 ∈ F)}

noncomputable def s_iter (F : Finset ℤ) (n : ℕ) : Finset ℤ :=
  (Finset.range n).fold (λ acc _, s acc) F

def s_n_eq (F : Finset ℤ) (n : ℕ) : Finset ℤ :=
  F ∪ Finset.image (λ a, a + n) F

theorem inf_natural_numbers (F : Finset ℤ) :
  ∃ᶠ n in at_top, s_iter F n = s_n_eq F n :=
sorry

end inf_natural_numbers_l712_712145


namespace tangent_line_at_origin_l712_712380

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 2) * x

theorem tangent_line_at_origin (a : ℝ) (h : ∀ x, (3 * x^2 + 2 * a * x + (a - 2)) = 3 * (-x)^2 + 2 * a * (-x) + (a - 2)) :
  tangent_at (f a) (0 : ℝ) = -2 * x := 
sorry

end tangent_line_at_origin_l712_712380


namespace exam_score_impossible_l712_712111

theorem exam_score_impossible (x y : ℕ) : 
  (5 * x + y = 97) ∧ (x + y ≤ 20) → false :=
by
  sorry

end exam_score_impossible_l712_712111


namespace sqrt_equality_l712_712648

theorem sqrt_equality (m : ℝ) (n : ℝ) (h1 : 0 < m) (h2 : -3 * m ≤ n) (h3 : n ≤ 3 * m) :
    (Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2))
     - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2))
    = 2 * Real.sqrt (3 * m - n)) :=
sorry

end sqrt_equality_l712_712648


namespace find_socks_cost_l712_712127

variable (S : ℝ)
variable (socks_cost : ℝ := 9.5)
variable (shoe_cost : ℝ := 92)
variable (jack_has : ℝ := 40)
variable (needs_more : ℝ := 71)
variable (total_funds : ℝ := jack_has + needs_more)

theorem find_socks_cost (h : 2 * S + shoe_cost = total_funds) : S = socks_cost :=
by 
  sorry

end find_socks_cost_l712_712127


namespace find_integer_modulo_l712_712005

theorem find_integer_modulo :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 :=
by {
  sorry
}

end find_integer_modulo_l712_712005


namespace odd_log_function_eval_at_neg_four_l712_712051

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then log 2 x else -log 2 (-x)

theorem odd_log_function_eval_at_neg_four :
  (f (-4) = -2) :=
by
sorry

end odd_log_function_eval_at_neg_four_l712_712051


namespace Lara_age_in_10_years_l712_712139

theorem Lara_age_in_10_years (current_age: ℕ) (years_ago: ℕ) (years_from_now: ℕ) (age_years_ago: ℕ) (h1: current_age = age_years_ago + years_ago) (h2: age_years_ago = 9) (h3: years_ago = 7) (h4: years_from_now = 10) : current_age + years_from_now = 26 := 
by 
  rw [h2, h3] at h1
  rw [← h1, h4]
  exact rfl

end Lara_age_in_10_years_l712_712139


namespace find_k_for_solutions_l712_712752

theorem find_k_for_solutions (k : ℝ) :
  (∀ x: ℝ, x = 3 ∨ x = 5 → k * x^2 - 8 * x + 15 = 0) → k = 1 :=
by
  sorry

end find_k_for_solutions_l712_712752


namespace prove_tan_add_2023pi_over_4_l712_712465

-- Define the given conditions
def cosAlpha : ℝ := -3 / 5
def sinAlpha : ℝ := 4 / 5

-- Compute tanAlpha from the defined conditions (sin and cos)
def tanAlpha : ℝ := sinAlpha / cosAlpha

-- Define the actual theorem to prove that tan(α + 2023π/4) = 7
theorem prove_tan_add_2023pi_over_4 (tanAlpha : ℝ) (h_tanAlpha : tanAlpha = -4 / 3) : 
  Real.tan (Real.Angle.toReal (Real.Angle.mk_pi 2023 4 0 sinAlpha cosAlpha) + 2023 * Real.pi / 4) = 7 :=
sorry

end prove_tan_add_2023pi_over_4_l712_712465


namespace difference_is_minus_four_l712_712956

def percentage_scoring_60 : ℝ := 0.15
def percentage_scoring_75 : ℝ := 0.25
def percentage_scoring_85 : ℝ := 0.40
def percentage_scoring_95 : ℝ := 1 - (percentage_scoring_60 + percentage_scoring_75 + percentage_scoring_85)

def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_85 : ℝ := 85
def score_95 : ℝ := 95

def mean_score : ℝ :=
  (percentage_scoring_60 * score_60) +
  (percentage_scoring_75 * score_75) +
  (percentage_scoring_85 * score_85) +
  (percentage_scoring_95 * score_95)

def median_score : ℝ := score_85

def difference_mean_median : ℝ := mean_score - median_score

theorem difference_is_minus_four : difference_mean_median = -4 :=
by
  sorry

end difference_is_minus_four_l712_712956


namespace collinearity_of_points_l712_712911

noncomputable theory
open_locale classical

variables {A P Q M N I T K : Type*}

-- Conditions given in the problem
variables [IncidenceGeometry A P Q M N I T K]
variable [IntersectionPoint T (Line A P) (Line C Q)]
variable [IntersectionPoint K (Line M P) (Line N Q)]

-- Statement of the proof problem
theorem collinearity_of_points :
  Collinear {T, K, I} :=
sorry

end collinearity_of_points_l712_712911


namespace oliver_gave_janet_l712_712358

def initial_candy : ℕ := 78
def remaining_candy : ℕ := 68

theorem oliver_gave_janet : initial_candy - remaining_candy = 10 :=
by
  sorry

end oliver_gave_janet_l712_712358


namespace sum_of_sequence_l712_712392

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) + 1 = (a n + 1) / (2 * a n + 3)

theorem sum_of_sequence {a : ℕ → ℝ} (h_seq : sequence a) (h_a1 : a 1 ≠ 1) :
    let b := λ n, 2 / (a n + 1) in
    (∑ n in Finset.range 20, b n) = 780 :=
sorry

end sum_of_sequence_l712_712392


namespace min_value_of_product_l712_712400

theorem min_value_of_product (n : ℕ) (x : ℕ → ℝ) (h0 : ∀ i, 0 ≤ x i) (h1 : ∑ i in finset.range n, x i ≤ (1/2) : ℝ) :
  ∃ (c : ℝ), c = 1/2 ∧ ∀ (y : fin n → ℝ), (∀ i, y i = 1 - x i) → (finset.prod finset.univ y) = c :=
sorry

end min_value_of_product_l712_712400


namespace resolvent_correct_l712_712238

noncomputable def K (x t : ℝ) : ℝ :=
  x - 2 * t

theorem resolvent_correct (x t λ : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : 0 ≤ t ∧ t ≤ 1) :
  (∃ R : ℝ, R = (x - 2 * t + (x + t - 2 * x * t - 2 / 3) * λ) / (1 + λ / 2 + λ ^ 2 / 6)) :=
by
  sorry

end resolvent_correct_l712_712238


namespace Lara_age_10_years_from_now_l712_712132

theorem Lara_age_10_years_from_now (current_year_age : ℕ) (age_7_years_ago : ℕ)
  (h1 : age_7_years_ago = 9) (h2 : current_year_age = age_7_years_ago + 7) :
  current_year_age + 10 = 26 :=
by
  sorry

end Lara_age_10_years_from_now_l712_712132


namespace quadratic_sum_abc_l712_712203

theorem quadratic_sum_abc 
  (a b c : ℝ) 
  (h1 : ∀ x : ℝ, y = ax^2 + bx + c → ∃ y = 16) 
  (h2 : ax^2 + bx + c = 0 ↔ (x = -3) ∨ (x = 5)) :
  a + b + c = 16 := 
sorry

end quadratic_sum_abc_l712_712203


namespace collinear_T_K_I_l712_712904

noncomputable def T (A P C Q : Point) : Point := intersection (line_through A P) (line_through C Q)
noncomputable def K (M P N Q : Point) : Point := intersection (line_through M P) (line_through N Q)

theorem collinear_T_K_I (A P C Q M N I : Point) :
  collinear [T A P C Q, K M P N Q, I] :=
sorry

end collinear_T_K_I_l712_712904


namespace collinearity_of_T_K_I_l712_712902

-- Definitions of the points and lines
variables {A P Q M N C I T K : Type} [Nonempty A] [Nonempty P] [Nonempty Q] 
  [Nonempty M] [Nonempty N] [Nonempty C] [Nonempty I]

-- Intersection points conditions
def intersect (l₁ l₂ : Type) : Type := sorry

-- Given conditions
def condition_1 : T = intersect (line A P) (line C Q) := sorry
def condition_2 : K = intersect (line M P) (line N Q) := sorry

-- Proof that T, K, and I are collinear
theorem collinearity_of_T_K_I : collinear {T, K, I} := by
  have h1 : T = intersect (line A P) (line C Q) := condition_1
  have h2 : K = intersect (line M P) (line N Q) := condition_2
  -- Further steps needed to prove collinearity
  sorry

end collinearity_of_T_K_I_l712_712902


namespace find_value_of_fraction_l712_712532

variable {x y : ℝ}

theorem find_value_of_fraction (h1 : x > 0) (h2 : y > x) (h3 : y > 0) (h4 : x / y + y / x = 3) : 
  (x + y) / (y - x) = Real.sqrt 5 := 
by sorry

end find_value_of_fraction_l712_712532


namespace max_difference_pairs_l712_712880

-- Define the sets X and Y with given constraints
variables (X Y : Finset ℤ) (hx : X.card = 1000) (hy : Y.card = 1000)
-- Define the functions f and g
variables (f g : ℤ → ℤ → ℝ)

-- Assume if x is not in X and y is not in Y, then f(x,y) = g(x,y)
def condition (x y : ℤ) : Prop := 
  x ∉ X ∧ y ∉ Y → f x y = g x y

-- The maximum number of pairs (x,y) such that f(x,y) ≠ g(x,y) is at most 3000
theorem max_difference_pairs : 
  (∃ X Y : Finset ℤ, (X.card = 1000) ∧ (Y.card = 1000) ∧ 
   (∀ x y, x ∉ X ∧ y ∉ Y → f x y = g x y)) →
  (Finset.card (Finset.filter (λ (p : ℤ × ℤ), f p.1 p.2 ≠ g p.1 p.2) (Finset.product (Finset.range 2000) (Finset.range 2000))) ≤ 3000) :=
begin
  intros h,
  -- Assume the conditions and the proof here would be provided
  sorry -- Placeholder for the actual proof
end

end max_difference_pairs_l712_712880


namespace matrix_N_correct_l712_712333

noncomputable def N : Matrix ℝ 3 3 :=
  ![
    ![0, -3, -2],
    ![2, 0, 3],
    ![-2, -2, 0]
  ]

def vec1 : ℝ^3 := ![3, -4, 6]
def vec2 : ℝ^3 := ![-1, 2, -3]
def vecSum : ℝ^3 := vec1 + vec2

theorem matrix_N_correct : ∀ v : ℝ^3, N.mul_vec v = vecSum.cross_product v := 
by
  sorry

end matrix_N_correct_l712_712333


namespace collinear_T_K_I_l712_712906

noncomputable def T (A P C Q : Point) : Point := intersection (line_through A P) (line_through C Q)
noncomputable def K (M P N Q : Point) : Point := intersection (line_through M P) (line_through N Q)

theorem collinear_T_K_I (A P C Q M N I : Point) :
  collinear [T A P C Q, K M P N Q, I] :=
sorry

end collinear_T_K_I_l712_712906


namespace part_1_part_2_part_3_l712_712790

noncomputable def f (a x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 + 2 * a * x + 1
  else x^2 -2 * a * x + 1

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem part_1 (a x : ℝ) (h_even : is_even_function (f a)) (h_x_lt_0 : x < 0) : f a x = x^2 - 2 * a * x + 1 :=
  sorry

noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ -5/2 then 1
  else 10 * a + 26

theorem part_2 (a : ℝ) : g a = 
    if a ≤ -5/2 then 1
    else 10 * a + 26 :=
  sorry

theorem part_3 (m : ℝ) :
  g (8 * m) = g (1 / m) ↔ (m = real.sqrt 2 / 4 ∨ (-2 / 5 ≤ m ∧ m ≤ -5 / 16)) :=
  sorry

end part_1_part_2_part_3_l712_712790


namespace problem1_l712_712666

theorem problem1 (k : ℝ) : (∃ x : ℝ, k*x^2 + (2*k + 1)*x + (k - 1) = 0) → k ≥ -1/8 := 
sorry

end problem1_l712_712666


namespace problem_statement_l712_712784

noncomputable def x : ℕ := 4
noncomputable def y : ℤ := 3  -- alternatively, we could define y as -3 and the equality would still hold

theorem problem_statement : x^2 + y^2 + x + 2023 = 2052 := by
  sorry  -- Proof goes here

end problem_statement_l712_712784


namespace quadrilateral_is_parallelogram_l712_712632

-- Define the quadrilateral and the point O
variable (A B C D O : Type) [PlaneGeometry A B C D O]

-- Assumptions: 
-- 1. O is the intersection point of diagonals AC and BD.
-- 2. The areas of the triangles formed by the diagonals are equal.
-- 
-- to prove that ABCD is a parallelogram.

theorem quadrilateral_is_parallelogram (h1 : intersection AC BD O)
  (h2 : triangle_area A B O = triangle_area C B O)
  (h3 : triangle_area A D O = triangle_area C D O) : parallelogram A B C D :=
  sorry  

end quadrilateral_is_parallelogram_l712_712632


namespace prime_factorization_min_x_l712_712935

-- Define the conditions
variable (x y : ℕ) (a b e f : ℕ)

-- Given conditions: x and y are positive integers, and 5x^7 = 13y^11
axiom condition1 : 0 < x ∧ 0 < y
axiom condition2 : 5 * x^7 = 13 * y^11

-- Prove the mathematical equivalence
theorem prime_factorization_min_x (a b e f : ℕ) 
    (hx : 5 * x^7 = 13 * y^11)
    (h_prime : a = 13 ∧ b = 5 ∧ e = 6 ∧ f = 1) :
    a + b + e + f = 25 :=
sorry

end prime_factorization_min_x_l712_712935


namespace integral_equality_l712_712744

open Real

theorem integral_equality :
  ∫ x in 1..2, 1 / (x * (x + 1)) = ln (4 / 3) :=
by
  sorry

end integral_equality_l712_712744


namespace evaluate_expression_l712_712344

theorem evaluate_expression :
  (3 - 4 * (4 - 5)⁻¹)⁻¹ = 1 / 7 :=
by
  sorry

end evaluate_expression_l712_712344


namespace number_of_valid_5_digit_numbers_l712_712815

def is_multiple_of_16 (n : Nat) : Prop := 
  n % 16 = 0

theorem number_of_valid_5_digit_numbers : Nat := 
  sorry

example : number_of_valid_5_digit_numbers = 90 :=
  sorry

end number_of_valid_5_digit_numbers_l712_712815


namespace hyperbola_eccentricity_eq_3_sqrt_5_div_5_l712_712793

noncomputable def eccentricity_of_hyperbola (a : ℝ) (h : a > 0) (x y : ℝ) : ℝ :=
  let b := 2
  let c := 3
  c / a

theorem hyperbola_eccentricity_eq_3_sqrt_5_div_5 (a : ℝ) (h : a > 0) :
  (∃ (x y : ℝ), (x^2 / a^2 - y^2 / 4 = 1)) ∧ (∃ k, y^2 = 12*x) → 
  eccentricity_of_hyperbola a h = 3 * real.sqrt 5 / 5 :=
by
  sorry

end hyperbola_eccentricity_eq_3_sqrt_5_div_5_l712_712793


namespace num_unique_three_digit_numbers_l712_712814

theorem num_unique_three_digit_numbers : 
  let available_digits := {2, 3, 3, 5, 5, 6, 6}
  ∃ n, n = 42 ∧ ∀ x : ℕ, (100 ≤ x ∧ x < 1000) → 
    (∀ d, d ∈ available_digits → digit_count x d ≤ count d available_digits) :=
sorry

end num_unique_three_digit_numbers_l712_712814


namespace equal_diagonals_of_isosceles_trapezoids_l712_712621

theorem equal_diagonals_of_isosceles_trapezoids
  {A B C D E F G H : Point}
  {circle : Circle}
  (hAB : A ≠ B) (hBC : B ≠ C) (hCD : C ≠ D) (hDA : D ≠ A)
  (hEF : E ≠ F) (hFG : F ≠ G) (hGH : G ≠ H) (hHE : H ≠ E)
  (hABCD_inscribed : is_inscribed circle {A, B, C, D})
  (hEFGH_inscribed : is_inscribed circle {E, F, G, H})
  (h_parallel_AB_EF : ∃ (l1 l2 : Line), (l1 ∥ l2) ∧ A ∈ l1 ∧ B ∈ l1 ∧ E ∈ l2 ∧ F ∈ l2)
  (h_parallel_BC_FG : ∃ (l1 l2 : Line), (l1 ∥ l2) ∧ B ∈ l1 ∧ C ∈ l1 ∧ F ∈ l2 ∧ G ∈ l2)
  (h_parallel_CD_GH : ∃ (l1 l2 : Line), (l1 ∥ l2) ∧ C ∈ l1 ∧ D ∈ l1 ∧ G ∈ l2 ∧ H ∈ l2)
  (h_parallel_DA_HE : ∃ (l1 l2 : Line), (l1 ∥ l2) ∧ D ∈ l1 ∧ A ∈ l1 ∧ H ∈ l2 ∧ E ∈ l2)
  (isosceles_ABCD : A ≠ C ∧ D ≠ B ∧ (dist A C = dist B D))
  (isosceles_EFGH : E ≠ G ∧ H ≠ F ∧ (dist E G = dist F H))
  : dist A C = dist E G ∧ dist B D = dist F H := sorry

end equal_diagonals_of_isosceles_trapezoids_l712_712621


namespace probability_prime_multiple_of_11_l712_712283

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem probability_prime_multiple_of_11 (card_count : ℕ) (cards : finset ℕ)
  (h_card_count : card_count = 100)
  (h_cards : cards = finset.range (card_count + 1))
  (selected_card : ℕ) :
  (is_prime selected_card ∧ is_multiple_of 11 selected_card) →
  (1 : ℚ) / 100 :=
by
  sorry

end probability_prime_multiple_of_11_l712_712283


namespace smallest_sum_of_squares_l712_712993

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l712_712993


namespace solve_quadratic_eq_l712_712566

theorem solve_quadratic_eq (x : ℝ) : (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2) := by
  sorry

end solve_quadratic_eq_l712_712566


namespace ingrid_income_l712_712129

theorem ingrid_income (I : ℝ) (h1 : 0.30 * 56000 = 16800) 
  (h2 : ∀ (I : ℝ), 0.40 * I = 0.4 * I) 
  (h3 : 0.35625 * (56000 + I) = 16800 + 0.4 * I) : 
  I = 49142.86 := 
by 
  sorry

end ingrid_income_l712_712129


namespace relationship_between_M_and_N_l712_712449
   
   variable (x : ℝ)
   def M := 2*x^2 - 12*x + 15
   def N := x^2 - 8*x + 11
   
   theorem relationship_between_M_and_N : M x ≥ N x :=
   by
     sorry
   
end relationship_between_M_and_N_l712_712449


namespace standard_deviation_distance_l712_712197

-- Definitions and assumptions based on the identified conditions
def mean : ℝ := 12
def std_dev : ℝ := 1.2
def value : ℝ := 9.6

-- Statement to prove
theorem standard_deviation_distance : (value - mean) / std_dev = -2 :=
by sorry

end standard_deviation_distance_l712_712197


namespace arithmetic_sequence_a14_eq_41_l712_712848

theorem arithmetic_sequence_a14_eq_41 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h_a2 : a 2 = 5) 
  (h_a6 : a 6 = 17) : 
  a 14 = 41 :=
sorry

end arithmetic_sequence_a14_eq_41_l712_712848


namespace minimum_box_value_l712_712821

theorem minimum_box_value :
  ∃ (a b : ℤ), a ≠ b ∧ b ≠ 61 ∧ a ≠ 61 ∧ (a * b = 30) ∧ (∃ M : ℤ, (ax + b)*(bx + a) = 30x^2 + M * x + 30) ∧ M = min (a^2 + b^2) 61 :=
by
  sorry

end minimum_box_value_l712_712821


namespace normal_distribution_probability_l712_712755


open ProbabilityTheory MeasureTheory

noncomputable def P_6_lt_X_lt_7 : ℝ :=
  (cdf (normal 5 1) 7 - cdf (normal 5 1) 6)

theorem normal_distribution_probability :
  P_6_lt_X_lt_7 = 0.1359 :=
by
  sorry

end normal_distribution_probability_l712_712755


namespace nine_digit_number_moving_last_digit_condition_l712_712288

theorem nine_digit_number_moving_last_digit_condition (B : ℕ) (h1: B > 222222222)
  (h2: Nat.gcd B 18 = 1) (A : ℕ) :
  (∃ (A_max : ℕ) (A_min : ℕ),
    A_max = 999999998 ∧ A_min = 122222224 ∧
    (A = 10^8 * (B % 10) + (B / 10)) :=
begin
  sorry
end

end nine_digit_number_moving_last_digit_condition_l712_712288


namespace equation_of_line_l_l712_712946

-- The definition of the parabola and its properties
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- The focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- The line with slope k passing through the focus
def line_through_focus (k : ℝ) (k_pos : k > 0) (x y : ℝ) : Prop := 
  y = k * x + 1

-- Midpoint of intersection points (A, B) on the parabola
def is_midpoint (A B P : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- The point M on the parabola such that the perpendicular from midpoint P intersects M
def perpendicular_intersect_at (P M : ℝ × ℝ) : Prop :=
  P.1 = M.1

-- Distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Problem statement: Find the equation of line l given the conditions
theorem equation_of_line_l (A B P M : ℝ × ℝ) (k : ℝ) (k_pos : k > 0) :
  parabola M.1 M.2 →
  line_through_focus k k_pos A.1 A.2 →
  line_through_focus k k_pos B.1 B.2 →
  is_midpoint A B P →
  perpendicular_intersect_at P M →
  distance M focus = 4 →
  ∃ (m : ℝ), m = sqrt 3 ∧ ∀ x y, line_through_focus m (by linarith) x y ↔ y = sqrt 3 * x + 1 :=
sorry

end equation_of_line_l_l712_712946


namespace curve_is_hyperbola_l712_712002

theorem curve_is_hyperbola (r θ : ℝ) (h : r = 3 * Real.tan θ * Real.sec θ) :
  ∃ (x y : ℝ), (x^2 - y^2)^2 = 7*x^2*y^2 :=
by
  sorry

end curve_is_hyperbola_l712_712002


namespace imaginary_part_of_complex_l712_712411

theorem imaginary_part_of_complex :
  let i := Complex.I,
  let z := 5 / (i * (i + 2)),
  Complex.im z = -2 := sorry

end imaginary_part_of_complex_l712_712411


namespace parabola_vertex_and_point_l712_712282

/-- The vertex form of the parabola is at (7, -6) and passes through the point (1,0).
    Verify that the equation parameters a, b, c satisfy a + b + c = -43 / 6. -/
theorem parabola_vertex_and_point (a b c : ℚ)
  (h_eq : ∀ y, (a * y^2 + b * y + c) = a * (y + 6)^2 + 7)
  (h_vertex : ∃ x y, x = a * y^2 + b * y + c ∧ y = -6 ∧ x = 7)
  (h_point : ∃ x y, x = a * y^2 + b * y + c ∧ x = 1 ∧ y = 0) :
  a + b + c = -43 / 6 :=
by
  sorry

end parabola_vertex_and_point_l712_712282


namespace max_area_equilateral_in_rectangle_l712_712605

theorem max_area_equilateral_in_rectangle :
  ∃ (p q r : ℕ), let area := p * (Real.sqrt q) - r in
  (q ≠ 0) ∧ (∀ k : ℕ, k * k ∣ q → k = 1) ∧
  (ABCD_area_pq (r : ℝ = 10) (s : ℝ = 11) (t:=area))  is ≤ 221 * (Real.sqrt 3) - 330
  p + q + r = 554 := sorry

end max_area_equilateral_in_rectangle_l712_712605


namespace isosceles_triangle_proof_l712_712711

open EuclideanGeometry

noncomputable def isosceles_triangle_in_circle (A B C E F D G : Point) (circ : Circle) :=
isosceles △ABC ∧
inscribed △ABC circ ∧
∠ ACB = 90° ∧
chord circ E F ∧
E ≠ C ∧
F ≠ C ∧
meet_at_line CE AB D ∧
meet_at_line CF AB G

theorem isosceles_triangle_proof 
  {A B C E F D G : Point} 
  (h : isosceles_triangle_in_circle A B C E F D G circ) :
  |CE| * |DG| = |EF| * |CG| :=
  sorry

end isosceles_triangle_proof_l712_712711


namespace celebration_women_count_l712_712716

theorem celebration_women_count (num_men : ℕ) (num_pairs : ℕ) (pairs_per_man : ℕ) (pairs_per_woman : ℕ) 
  (hm : num_men = 15) (hpm : pairs_per_man = 4) (hwp : pairs_per_woman = 3) (total_pairs : num_pairs = num_men * pairs_per_man) : 
  num_pairs / pairs_per_woman = 20 :=
by
  sorry

end celebration_women_count_l712_712716


namespace congruent_triangles_implies_corresponding_sides_equal_corresponding_sides_equal_implies_congruent_triangles_not_congruent_triangles_implies_not_corresponding_sides_equal_not_corresponding_sides_equal_implies_not_congruent_triangles_four_equal_sides_implies_is_square_is_square_implies_four_equal_sides_not_four_equal_sides_implies_not_is_square_not_is_square_implies_not_four_equal_sides_l712_712618

namespace GeometricPropositions

-- Definitions for congruence in triangles and quadrilaterals:
def congruent_triangles (Δ1 Δ2 : Type) : Prop := sorry
def corresponding_sides_equal (Δ1 Δ2 : Type) : Prop := sorry

def four_equal_sides (Q : Type) : Prop := sorry
def is_square (Q : Type) : Prop := sorry

-- Propositions and their logical forms for triangles
theorem congruent_triangles_implies_corresponding_sides_equal (Δ1 Δ2 : Type) : congruent_triangles Δ1 Δ2 → corresponding_sides_equal Δ1 Δ2 := sorry

theorem corresponding_sides_equal_implies_congruent_triangles (Δ1 Δ2 : Type) : corresponding_sides_equal Δ1 Δ2 → congruent_triangles Δ1 Δ2 := sorry

theorem not_congruent_triangles_implies_not_corresponding_sides_equal (Δ1 Δ2 : Type) : ¬ congruent_triangles Δ1 Δ2 → ¬ corresponding_sides_equal Δ1 Δ2 := sorry

theorem not_corresponding_sides_equal_implies_not_congruent_triangles (Δ1 Δ2 : Type) : ¬ corresponding_sides_equal Δ1 Δ2 → ¬ congruent_triangles Δ1 Δ2 := sorry

-- Propositions and their logical forms for quadrilaterals
theorem four_equal_sides_implies_is_square (Q : Type) : four_equal_sides Q → is_square Q := sorry

theorem is_square_implies_four_equal_sides (Q : Type) : is_square Q → four_equal_sides Q := sorry

theorem not_four_equal_sides_implies_not_is_square (Q : Type) : ¬ four_equal_sides Q → ¬ is_square Q := sorry

theorem not_is_square_implies_not_four_equal_sides (Q : Type) : ¬ is_square Q → ¬ four_equal_sides Q := sorry

end GeometricPropositions

end congruent_triangles_implies_corresponding_sides_equal_corresponding_sides_equal_implies_congruent_triangles_not_congruent_triangles_implies_not_corresponding_sides_equal_not_corresponding_sides_equal_implies_not_congruent_triangles_four_equal_sides_implies_is_square_is_square_implies_four_equal_sides_not_four_equal_sides_implies_not_is_square_not_is_square_implies_not_four_equal_sides_l712_712618


namespace sequence_a_formula_T_n_formula_l712_712426

noncomputable def sequence_a (n : ℕ) : ℕ := 2 * n - 1
noncomputable def sequence_b (n : ℕ) : ℚ := (2 * n - 1) / (2 ^ n)
noncomputable def T_n (n : ℕ) : ℚ := ∑ k in Finset.range n, sequence_b (k + 1)

theorem sequence_a_formula :
  (∀ n : ℕ, sequence_a n = 2 * n - 1) →
  sequence_a 2 = 3 ∧ sequence_a 4 = 7 ∧ sequence_a 2 * 2 + 1 = 3 * 2 * 2 - 1
  := sorry

theorem T_n_formula :
  ∀ n : ℕ, T_n n = 3 - (↑(2 * n + 3) / (2 ^ n)) := sorry

end sequence_a_formula_T_n_formula_l712_712426


namespace tangent_point_of_exponential_l712_712340

theorem tangent_point_of_exponential {
  : ∃ (x0 : ℝ) (y0 : ℝ), y0 = Real.exp x0 ∧ (0 - y0) = (Real.exp x0) * (0 - x0) →
  (x0 = 1) ∧ (y0 = Real.exp 1) :=
begin
  sorry
end

end tangent_point_of_exponential_l712_712340


namespace reflection_center_of_regular_9_gon_l712_712661

theorem reflection_center_of_regular_9_gon (P : Type) [regular_convex_9_gon P] (A : finite (9 : ℕ)) : 
  final_reflection_center P = A 5 := 
sorry

end reflection_center_of_regular_9_gon_l712_712661


namespace sum_reciprocals_distances_l712_712664

-- Definitions
variables {C : Type*} [circle : MetricSpace C] {O : C} (r : ℝ) (EF : line C)

-- Assume a point F on the extension of OO such that CF * OO = r^2
def point_F_extension (O F : C) := (distance circle.center F) * (distance O circle.center) = r^2

-- Assume EF is parallel to CF
def para_EF_CF (F : C) := line.parallel EF (line.through circle.center F)

-- Theorem to prove
theorem sum_reciprocals_distances {A A' : C} (h_extension : point_F_extension O F) 
  (h_parallel : para_EF_CF F)
  (h_chord : ¬(A = A') ∧ line.through O A = line.through O A') :
  let dA := distance A (EF : Set C),
      dA' := distance A' (EF : Set C) in
  (1 / dA + 1 / dA') = (2 / distance O F) :=
sorry


end sum_reciprocals_distances_l712_712664


namespace cos_alpha_plus_5pi_over_4_eq_16_over_65_l712_712761

theorem cos_alpha_plus_5pi_over_4_eq_16_over_65
  (α β : ℝ)
  (hα : -π / 4 < α ∧ α < 0)
  (hβ : π / 2 < β ∧ β < π)
  (hcos_sum : Real.cos (α + β) = -4/5)
  (hcos_diff : Real.cos (β - π / 4) = 5/13) :
  Real.cos (α + 5 * π / 4) = 16/65 :=
by
  sorry

end cos_alpha_plus_5pi_over_4_eq_16_over_65_l712_712761


namespace even_and_monotonically_decreasing_l712_712304

noncomputable def f := fun x : ℝ => log (1 / |x|)

theorem even_and_monotonically_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, (0 < x ∧ x < y) → f y < f x) :=
by
  sorry

end even_and_monotonically_decreasing_l712_712304


namespace children_neither_happy_nor_sad_l712_712171

theorem children_neither_happy_nor_sad :
  ∀ (total happy sad : ℕ), total = 60 → happy = 30 → sad = 10 →
  happy + sad = 40 → total - (happy + sad) = 20 :=
by
  intros total happy sad ht hh hs hhs
  rw [ht, hh, hs] at hhs
  sorry

end children_neither_happy_nor_sad_l712_712171


namespace ratio_is_three_l712_712870

noncomputable def ratio_tailored_to_off_the_rack (cost_off_the_rack : ℕ) (total_cost : ℕ) (extra_cost : ℕ) : ℚ :=
  let x := (total_cost - cost_off_the_rack - extra_cost) / cost_off_the_rack in x

theorem ratio_is_three :
  ratio_tailored_to_off_the_rack 300 1400 200 = 3 :=
by
  sorry

end ratio_is_three_l712_712870


namespace intersection_M_N_l712_712072

noncomputable def M : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}
def N : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_M_N :
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_l712_712072


namespace length_of_SR_l712_712118

variable (x y : ℝ)
variables (P Q R S : ℝ → ℝ → Prop)
variable [triangle : IsoscelesTriangle P Q R]

-- Given conditions
axiom hyp1 : IsoscelesTriangle P Q R (PQ := x) (PR := x)
axiom hyp2 : ∀ P Q R S, P Q S ∧ P S R (RightTriangleAngle 90)
axiom hyp3 : PS = 2*x
axiom hyp4 : PointOnLineSegment S QR
axiom hyp5 : RightTriangleAngle Q S P 90 
axiom hyp6 : Angle PQS 30 

-- The proposition
theorem length_of_SR : SR = (y / 2) * (1 - real.sqrt 3) := by
  -- Introduce the proof here
  sorry

end length_of_SR_l712_712118


namespace max_distance_from_point_on_circle_to_line_l712_712523

noncomputable def center_of_circle : ℝ × ℝ := (5, 3)
noncomputable def radius_of_circle : ℝ := 3
noncomputable def line_eqn (x y : ℝ) : ℝ := 3 * x + 4 * y - 2
noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ := (|a * px + b * py + c|) / (Real.sqrt (a * a + b * b))

theorem max_distance_from_point_on_circle_to_line :
  let Cx := (center_of_circle.1)
  let Cy := (center_of_circle.2)
  let d := distance_point_to_line Cx Cy 3 4 (-2)
  d + radius_of_circle = 8 := by
  sorry

end max_distance_from_point_on_circle_to_line_l712_712523


namespace south_pole_circumcircle_l712_712150

open Triangle

theorem south_pole_circumcircle (A B C : Point) (S : Point) 
    (h1 : is_triangle A B C) 
    (h2 : angle_bisector_intersection A B C S)
    (h3 : perpendicular_bisector_intersection B C S) :
    lies_on_circumcircle A B C S := 
sorry

end south_pole_circumcircle_l712_712150


namespace equal_segment_sums_l712_712492

-- Define the set of numbers and the circle-structured segments
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the circle structure intersected by 4 segments
structure CircleSegments :=
  (center : ℤ)
  (segments : List (ℤ × ℤ))

-- Define a condition for circle with equal sum in each segment
def valid_circle (circle : CircleSegments) (target_sum : ℤ) : Prop :=
  ∀ (seg : ℤ × ℤ), seg ∈ circle.segments → seg.1 + seg.2 + circle.center = target_sum

-- CircleSegments instantiation
def example_circle : CircleSegments :=
  ⟨5, [(1, 9), (2, 8), (3, 7), (4, 6)]⟩

-- Main theorem statement
theorem equal_segment_sums (circle : CircleSegments) : valid_circle circle 15 :=
by {
  sorry
}

end equal_segment_sums_l712_712492


namespace factor_polynomial_l712_712348

noncomputable def polynomial := 29.52 * (x^2 * y) - (y^2 * z) + (z^2 * x) - (x^2 * z) + (y^2 * x) + (z^2 * y) - (2 * x * y * z)

theorem factor_polynomial (x y z : ℝ) :
    polynomial = (y - z) * (x + y) * (x - z) :=
by
  sorry

end factor_polynomial_l712_712348


namespace vanessa_scored_27_points_l712_712629

variable (P : ℕ) (number_of_players : ℕ) (average_points_per_player : ℚ) (vanessa_points : ℕ)

axiom team_total_points : P = 48
axiom other_players : number_of_players = 6
axiom average_points_per_other_player : average_points_per_player = 3.5

theorem vanessa_scored_27_points 
  (h1 : P = 48)
  (h2 : number_of_players = 6)
  (h3 : average_points_per_player = 3.5)
: vanessa_points = 27 :=
sorry

end vanessa_scored_27_points_l712_712629


namespace solve_inequality_l712_712030

noncomputable theory

def functional_equation (f : ℝ → ℝ) : Prop :=
∀ (a b : ℝ), f (a - b) = f (a) / f (b)

def condition_2 (f : ℝ → ℝ) : Prop :=
∀ x < 0, f x > 1

def condition_3 (f : ℝ → ℝ) : Prop :=
f 4 = 1 / 16

def question (f : ℝ → ℝ) (x : ℝ) : Prop :=
f (x - 3) * f (5 - x^2) ≤ 1 / 4

theorem solve_inequality (f : ℝ → ℝ) :
  (functional_equation f) →
  (condition_2 f) →
  (condition_3 f) →
  ∀ x, 0 ≤ x → x ≤ 1 → question f x :=
by
  intros h1 h2 h3 x hx0 hx1
  sorry

end solve_inequality_l712_712030


namespace collinear_TKI_l712_712891

-- Definitions for points and lines
variables {A P C Q M N I T K : Type}
variable {line : Type → Type}
variables (AP : line A → line P) (CQ : line C → line Q) (MP : line M → line P) (NQ : line N → line Q)

-- Conditions from the problem
-- Assume there exist points T and K which are intersections of the specified lines
axiom intersects_AP_CQ : ∃ (T : Type), AP T = CQ T
axiom intersects_MP_NQ : ∃ (K : Type), MP K = NQ K

-- Collinearity of points T, K, and I
theorem collinear_TKI : ∀ (I : Type) (T : Type) (K : Type),
  intersects_AP_CQ → intersects_MP_NQ → collinear I T K :=
by sorry

end collinear_TKI_l712_712891


namespace ineq1_ineq2_ineq3_l712_712972

theorem ineq1 (x : ℝ) : 3 * x - 2 * x^2 + 2 ≥ 0 → (1 / 2 ≤ x ∧ x ≤ 2) := 
sorry

theorem ineq2 (x : ℝ) : (4 < |2 * x - 3| ∧ |2 * x - 3| ≤ 7) → 
  ((7 / 2 < x ∧ x ≤ 5) ∨ (-2 ≤ x ∧ x < -1 / 2)) := 
sorry

theorem ineq3 (x : ℝ) : (| x - 8 | - | x - 4 | > 2) → 
  (x < 5) := 
sorry

end ineq1_ineq2_ineq3_l712_712972


namespace problem1_problem2_l712_712762

-- Definitions based on conditions
variables {α : Type*} [linear_ordered_field α]

-- Condition: There are 100 positive numbers.
noncomputable def numbers (i : ℕ) (h : 1 ≤ i ∧ i ≤ 100) : α := sorry

-- Extensional Definition for subset sum
def sum_subset (f : ℕ → α) (s : finset ℕ) : α :=
  s.sum f

-- Proof 1: If the sum of any seven numbers is less than 7, then the sum of any ten numbers is less than 10
theorem problem1 (h : ∀ (s : finset ℕ), s.card = 7 → sum_subset numbers s < 7) :
  ∀ (t : finset ℕ), t.card = 10 → sum_subset numbers t < 10 :=
  sorry

-- Proof 2: If the sum of any ten numbers is less than 10, then it is not necessarily true that the sum of any seven numbers is less than 7 (counterexample)
theorem problem2 (h : ∀ (t : finset ℕ), t.card = 10 → sum_subset numbers t < 10) :
  ¬ ∀ (s : finset ℕ), s.card = 7 → sum_subset numbers s < 7 :=
  sorry

end problem1_problem2_l712_712762


namespace count_valid_n_l712_712817

-- Define the condition that n must be a multiple of 5.
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

-- Define the lcm function for natural numbers.
def my_lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Define the gcd function for natural numbers.
def my_gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Specify the factorials that will be used.
def fact5 : ℕ := 5!
def fact10 : ℕ := 10!

-- Define the main problem condition.
def specific_condition (n : ℕ) : Prop := my_lcm fact5 n = 5 * (my_gcd fact10 n)

-- Define the main proof statement.
theorem count_valid_n : {n : ℕ // is_multiple_of_5 n ∧ specific_condition n}.card = 48 := 
begin
  sorry -- The proof goes here
end

end count_valid_n_l712_712817


namespace investments_are_beneficial_l712_712376

-- Definitions of examples and their benefits as given in the conditions
def investment_in_education : Prop :=
  ∃ (benefit : String), 
    benefit = "enhances employability and earning potential"

def investment_in_physical_health : Prop :=
  ∃ (benefit : String), 
    benefit = "reduces future healthcare costs and enhances overall well-being"

def investment_in_reading_books : Prop :=
  ∃ (benefit : String), 
    benefit = "cultivates intellectual growth and contributes to personal and professional success"

-- The theorem combining the three investments and their benefits
theorem investments_are_beneficial :
  investment_in_education ∧ investment_in_physical_health ∧ investment_in_reading_books :=
by
  split;
  { 
    existsi "enhances employability and earning potential", sorry <|>
    existsi "reduces future healthcare costs and enhances overall well-being", sorry <|>
    existsi "cultivates intellectual growth and contributes to personal and professional success", sorry
  }

end investments_are_beneficial_l712_712376


namespace collinear_T_K_I_l712_712886

noncomputable def intersection (P Q : Set Point) : Point := sorry

variables (A P C Q M N I T K : Point)

-- Definitions based on conditions
def T_def : Point := intersection (line_through A P) (line_through C Q)
def K_def : Point := intersection (line_through M P) (line_through N Q)

-- Proof statement
theorem collinear_T_K_I :
  collinear ({T_def A P C Q, K_def M P N Q, I} : Set Point) := sorry

end collinear_T_K_I_l712_712886


namespace find_x_l712_712933

variables {a b x r : ℝ}

noncomputable def triple_base_exponent (a : ℝ) (b : ℕ) : ℝ :=
  (3 * a) ^ (3 * b)

theorem find_x (h_b : b ≠ 0) (h_r : r = triple_base_exponent a b) (h_product : r = a^b * x^b) :
  x = 27 * a^2 :=
by
  sorry

end find_x_l712_712933


namespace unique_solution_l712_712560

theorem unique_solution :
  ∀ a b c : ℕ,
    a > 0 → b > 0 → c > 0 →
    (3 * a * b * c + 11 * (a + b + c) = 6 * (a * b + b * c + c * a) + 18) →
    (a = 1 ∧ b = 2 ∧ c = 3) :=
by
  intros a b c ha hb hc h
  have h1 : a = 1 := sorry
  have h2 : b = 2 := sorry
  have h3 : c = 3 := sorry
  exact ⟨h1, h2, h3⟩

end unique_solution_l712_712560


namespace quadratic_with_roots_1_and_2_l712_712178

theorem quadratic_with_roots_1_and_2 : ∃ (a b c : ℝ), (a = 1 ∧ b = 2) ∧ (∀ x : ℝ, x ≠ 1 → x ≠ 2 → a * x^2 + b * x + c = 0) ∧ (a * x^2 + b * x + c = x^2 - 3 * x + 2) :=
by
  sorry

end quadratic_with_roots_1_and_2_l712_712178


namespace no_power_of_two_rearrangement_l712_712125

-- Definition to determine if a number is a power of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Definition to determine if the digits of two numbers are a permutation of each other
def digits_are_permuted (a b : ℕ) : Prop := multiset.of_nat_digits a = multiset.of_nat_digits b

-- The proof statement 
theorem no_power_of_two_rearrangement : 
  ¬ (∃ n c : ℕ, c > 0 ∧ is_power_of_two (2^n) ∧ is_power_of_two (2^(n+c)) ∧ digits_are_permuted (2^n) (2^(n+c))) := 
by sorry

end no_power_of_two_rearrangement_l712_712125


namespace area_of_garden_l712_712877

theorem area_of_garden :
  ∃ (short_posts long_posts : ℕ), short_posts + long_posts - 4 = 24 → long_posts = 3 * short_posts →
  ∃ (short_length long_length : ℕ), short_length = (short_posts - 1) * 5 → long_length = (long_posts - 1) * 5 →
  (short_length * long_length = 3000) :=
by {
  sorry
}

end area_of_garden_l712_712877


namespace find_k_n_l712_712732

theorem find_k_n (k n : ℕ) (h_kn_pos : 0 < k ∧ 0 < n) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 := 
by {
  sorry
}

end find_k_n_l712_712732


namespace pentagon_not_convex_l712_712263

def Pentagon (α1 α2 α3 α4 α5 : Point) : Prop :=
∃ (A1 A2 A3 A4 A5 : Point)
  (l1 : Line) (l2 : Line) (l3 : Line) (l4 : Line) (l5 : Line)
  (B1 : Point) (B2 : Point) (B3 : Point) (B4 : Point) (B5 : Point),
  -- Conditions for the pentagon and its angle bisectors
  (l1 = angle_bisector A1 A2 A3) ∧ (l2 = angle_bisector A2 A3 A4) ∧
  (l3 = angle_bisector A3 A4 A5) ∧ (l4 = angle_bisector A4 A5 A1) ∧
  (l5 = angle_bisector A5 A1 A2) ∧
  -- Intersection conditions forming points B1, B2, ...
  ((l1 ∩ l2 = B1) ∧ (l2 ∩ l3 = B2) ∧ (l3 ∩ l4 = B3) ∧ (l4 ∩ l5 = B4) ∧ (l5 ∩ l1 = B5)) ∧
  -- The question is whether the Pentagon formed by B1, B2, B3, B4, B5 is convex.
  ¬Convex B1 B2 B3 B4 B5

theorem pentagon_not_convex :
  ∀ (A1 A2 A3 A4 A5 : Point)
    (l1 : Line) (l2 : Line) (l3 : Line) (l4 : Line) (l5 : Line)
    (B1 : Point) (B2 : Point) (B3 : Point) (B4 : Point) (B5 : Point),
  (l1 = angle_bisector A1 A2 A3) →
  (l2 = angle_bisector A2 A3 A4) →
  (l3 = angle_bisector A3 A4 A5) →
  (l4 = angle_bisector A4 A5 A1) →
  (l5 = angle_bisector A5 A1 A2) →
  ((l1 ∩ l2 = B1) ∧ (l2 ∩ l3 = B2) ∧ (l3 ∩ l4 = B3) ∧ (l4 ∩ l5 = B4) ∧ (l5 ∩ l1 = B5)) →
  ¬Convex B1 B2 B3 B4 B5 :=
by
  intros
  sorry

end pentagon_not_convex_l712_712263


namespace sum_first_nine_terms_l712_712853

open ArithmeticSequence

theorem sum_first_nine_terms 
  (a : ℕ → ℕ) 
  (h1 : a 1 + a 4 + a 7 = 39) 
  (h2 : a 3 + a 6 + a 9 = 27) 
  (arith_seq : arithmetic_sequence a) :
  sum_first_nine_terms arith_seq = 99 := 
sorry

end sum_first_nine_terms_l712_712853


namespace decreasing_on_transformed_interval_l712_712053

theorem decreasing_on_transformed_interval
  (f : ℝ → ℝ)
  (h : ∀ ⦃x₁ x₂ : ℝ⦄, 1 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 2 → f x₁ ≤ f x₂) :
  ∀ ⦃x₁ x₂ : ℝ⦄, -1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 0 → f (1 - x₂) ≤ f (1 - x₁) :=
sorry

end decreasing_on_transformed_interval_l712_712053


namespace polynomial_max_bound_l712_712160

variable {n : ℕ} (p : ℝ → ℝ)

-- Define the conditions
def is_degree_at_most (p : ℝ → ℝ) (d : ℕ) : Prop := 
  ∃ coeffs : Fin d.succ → ℝ, ∀ x, p x = ∑ i, coeffs i * x^i

def bounded_on_integers_in_range (p : ℝ → ℝ) (n : ℕ) : Prop := 
  ∀ k : ℤ, - (n : ℤ) ≤ k ∧ k ≤ n → |p k.to_real| ≤ 1

-- State the proof problem
theorem polynomial_max_bound {p : ℝ → ℝ} (n : ℕ)
  (hdeg : is_degree_at_most p (2 * n))
  (hbound : bounded_on_integers_in_range p n) :
  ∀ x : ℝ, - (n : ℝ) ≤ x ∧ x ≤ n → |p x| ≤ 2^(2 * n) := 
begin
  sorry
end

end polynomial_max_bound_l712_712160


namespace jellybeans_in_carrie_box_l712_712719

theorem jellybeans_in_carrie_box :
  let s := (216 : ℝ)^(1/3)
  let bert_volume := s^3
  let carrie_length := 2 * s
  let carrie_width := 2 * s
  let carrie_height := 3 * s
  let carrie_volume := carrie_length * carrie_width * carrie_height
  bert_volume = 216 ∧
  carrie_volume / bert_volume = 12 ∧
  bert_volume * (carrie_volume / bert_volume) = 2592 →
  carrie_volume = 2592 :=
by
  intros s bert_volume carrie_length carrie_width carrie_height carrie_volume h
  cases h with h_bert_volume h
  cases h with h_scaling_factor h_final
  sorry

end jellybeans_in_carrie_box_l712_712719


namespace collinear_TKI_l712_712921

-- Definitions based on conditions
variables {A P Q C M N T K I : Type}
variables (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)

-- Definitions to represent the intersection points 
def T_def (A P Q C : Type) (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) : Prop := 
  ∃ T : Type, line_AP A T ∧ line_CQ C T

def K_def (M P N Q : Type) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop) : Prop := 
  ∃ K : Type, line_MP M K ∧ line_NQ N K

-- Theorem statement
theorem collinear_TKI (A P Q C M N T K I : Type)
  (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)
  (hT : T_def A P Q C line_AP line_CQ) 
  (hK : K_def M P N Q line_MP line_NQ) : 
  collinear T K I :=
sorry

end collinear_TKI_l712_712921


namespace points_on_circle_l712_712731

-- Given definitions
structure Point := (x : ℝ) (y : ℝ)
structure Quadrilateral (A B C D : Point) := (inscribed : ∃ O : Point, ∀ P ∈ {A, B, C, D}, dist O P = dist O A)

def isRhomb (A B C D : Point) (a : ℝ) : Prop :=
  dist A B = a ∧ dist B C = a ∧ dist C D = a ∧ dist D A = a ∧
  ∀ M : Point, (dist M A = dist M C ∧ dist M B = dist M D) → dist M A = a

def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- The problem statement to be proved
theorem points_on_circle (A B C D K L M N : Point) (a : ℝ)
  (hquad : Quadrilateral A B C D) (hneq : dist A B ≠ dist C D) 
  (hrhomb1 : isRhomb A K D L a) (hrhomb2 : isRhomb C M B N a) :
    ∃ O : Point, ∀ P ∈ {K, L, M, N}, dist O P = dist O K := 
sorry

end points_on_circle_l712_712731


namespace collinearity_of_T_K_I_l712_712901

-- Definitions of the points and lines
variables {A P Q M N C I T K : Type} [Nonempty A] [Nonempty P] [Nonempty Q] 
  [Nonempty M] [Nonempty N] [Nonempty C] [Nonempty I]

-- Intersection points conditions
def intersect (l₁ l₂ : Type) : Type := sorry

-- Given conditions
def condition_1 : T = intersect (line A P) (line C Q) := sorry
def condition_2 : K = intersect (line M P) (line N Q) := sorry

-- Proof that T, K, and I are collinear
theorem collinearity_of_T_K_I : collinear {T, K, I} := by
  have h1 : T = intersect (line A P) (line C Q) := condition_1
  have h2 : K = intersect (line M P) (line N Q) := condition_2
  -- Further steps needed to prove collinearity
  sorry

end collinearity_of_T_K_I_l712_712901


namespace fill_hole_l712_712707

def total_gallons_needed := 823
def initial_gallons_in_hole := 676
def additional_gallons_needed := 147

theorem fill_hole (total_gallons_needed initial_gallons_in_hole additional_gallons_needed: ℕ) :
  total_gallons_needed - initial_gallons_in_hole = additional_gallons_needed :=
by
  rw [total_gallons_needed, initial_gallons_in_hole, additional_gallons_needed]
  exact rfl
  done

end fill_hole_l712_712707


namespace salary_increase_l712_712305

theorem salary_increase (new_salary increase : ℝ) (h_new : new_salary = 25000) (h_inc : increase = 5000) : 
  ((increase / (new_salary - increase)) * 100) = 25 :=
by
  -- We will write the proof to satisfy the requirement, but it is currently left out as per the instructions.
  sorry

end salary_increase_l712_712305


namespace sin_alpha_eq_l712_712827

variable (α : Real) (x : Real)

-- Conditions
def isSecondQuadrant (α : Real) : Prop := α ≥ π/2 ∧ α ≤ π
def pointP (x : Real) : Prop := True
def cosAlpha (x : Real) (α : Real) : Prop := cos α = (sqrt 2 * x) / 4

-- Problem Statement
theorem sin_alpha_eq (h1 : isSecondQuadrant α) (h2 : pointP x) (h3 : cosAlpha x α) : 
    sin α = sqrt 10 / 4 := 
sorry

end sin_alpha_eq_l712_712827


namespace triangle_area_l712_712979

theorem triangle_area (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 2) : ∃ A : ℝ, A = 2 :=
by
  use 2
  sorry

end triangle_area_l712_712979


namespace ellipse_intersection_l712_712770

noncomputable def ellipse_standard_eq (x y : ℝ): Prop :=
  (x^2) / 4 + (y^2) / 3 = 1

def line_eq (k x : ℝ): ℝ := k * (x - 1)

theorem ellipse_intersection (k x1 x2 : ℝ) (h_discriminant : (4 * k^2 + 3) * x1^2 - 8 * k^2 * x1 + 4 * k^2 - 12 = 0) 
(h_xsum : x1 + x2 = 8 * k^2 / (4 * k^2 + 3))
(h_xprod : x1 * x2 = (4 * k^2 - 12) / (4 * k^2 + 3)):
let λ := x1 / (1 - x1),
    μ := x2 / (1 - x2)
in λ + μ = -8 / 3 :=
sorry

end ellipse_intersection_l712_712770


namespace symmetric_points_sum_l712_712792

theorem symmetric_points_sum (m n : ℚ)
  (h1 : ∃ l : ℝ → ℝ, symmetric_point (0, 2) l = (4, 0))
  (h2 : ∃ l : ℝ → ℝ, symmetric_point (6, 3) l = (m, n)) :
  m + n = 33 / 5 :=
sorry

noncomputable def symmetric_point : ℚ × ℚ → (ℝ → ℝ) → ℚ × ℚ :=
  fun ⟨x, y⟩ l => (x, y)  -- Placeholder definition, you need the actual formula for symmetric points.

end symmetric_points_sum_l712_712792


namespace schedule_count_l712_712242

-- Definitions of the conditions
def westfield_players : Finset (Fin 4) := {0, 1, 2, 3}
def eastfield_players : Finset (Fin 4) := {0, 1, 2, 3}

-- Theorem to prove the number of different ways to schedule the match.
theorem schedule_count : 
  let games_count := 16,
  let rounds_count := 4,
  let ways_individual_pairing := (4! : ℕ),
  let ways_round_order := (4! : ℕ) in
  games_count = westfield_players.card * eastfield_players.card ∧ 
  westfield_players.card = 4 ∧
  eastfield_players.card = 4 ∧
  (ways_individual_pairing * ways_round_order = 576) :=
by 
  let games_count := westfield_players.card * eastfield_players.card,
  let rounds_count := games_count / 4,
  let ways_individual_pairing := Nat.factorial 4,
  let ways_round_order := Nat.factorial 4,
  have h1 : games_count = 16 := by simp [games_count],
  have h2 : westfield_players.card = 4 := by simp [westfield_players],
  have h3 : eastfield_players.card = 4 := by simp [eastfield_players],
  have h4 : ways_individual_pairing * ways_round_order = 576 := by norm_num,
  exact ⟨h1, h2, h3, h4⟩,
  sorry

end schedule_count_l712_712242


namespace intersection_M_N_l712_712074

open Set

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2 * x + 1}

theorem intersection_M_N : M ∩ N = {-1, 1} :=
by
  sorry

end intersection_M_N_l712_712074


namespace shift_even_function_l712_712163

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem shift_even_function :
  ∀ x : ℝ, f (x - Real.pi / 12) = Real.cos (2 * x) :=
by sorry

end shift_even_function_l712_712163


namespace collinearity_of_T_K_I_l712_712926

noncomputable def intersection_point (l1 l2 : Line) : Point := sorry

-- Definitions of lines AP, CQ, MP, NQ based on the problem context
variables {A P C Q M N I : Point} (lAP lCQ lMP lNQ : Line)
variables (T : Point) (K : Point)

-- Given conditions
def condition_1 : T = intersection_point lAP lCQ := sorry
def condition_2 : K = intersection_point lMP lNQ := sorry

-- Theorem statement
theorem collinearity_of_T_K_I : T ∈ line_through K I :=
by {
  -- These are the conditions that we're given in the problem
  have hT : T = intersection_point lAP lCQ := sorry,
  have hK : K = intersection_point lMP lNQ := sorry,
  -- Rest of the proof would go here
  sorry
}

end collinearity_of_T_K_I_l712_712926


namespace triangle_def_ef_value_l712_712499

theorem triangle_def_ef_value (E F D : ℝ) (DE DF EF : ℝ) (h1 : E = 45)
  (h2 : DE = 100) (h3 : DF = 100 * Real.sqrt 2) :
  EF = Real.sqrt (30000 + 5000*(Real.sqrt 6 - Real.sqrt 2)) := 
sorry 

end triangle_def_ef_value_l712_712499


namespace sqrt_sum_eq_seven_l712_712573

noncomputable def x : ℝ := sorry -- the exact value of x is not necessary

theorem sqrt_sum_eq_seven (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
  sqrt (64 - x^2) + sqrt (36 - x^2) = 7 := sorry

end sqrt_sum_eq_seven_l712_712573


namespace geometric_sequence_a17_l712_712765

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := a n * q

theorem geometric_sequence_a17
  (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : a 2 * a 8 = 16)
  : a 17 = 256 := by
    sorry

end geometric_sequence_a17_l712_712765


namespace max_value_phi_l712_712801

theorem max_value_phi (φ : ℝ) (hφ : -Real.pi / 2 < φ ∧ φ < Real.pi / 2) :
  (∃ k : ℤ, φ = 2 * k * Real.pi + Real.pi / 2 - Real.pi / 3) →
  φ = Real.pi / 6 :=
by 
  intro h
  sorry

end max_value_phi_l712_712801


namespace machine_tasks_l712_712227

theorem machine_tasks (y : ℕ) 
  (h1 : (1 : ℚ)/(y + 4) + (1 : ℚ)/(y + 3) + (1 : ℚ)/(4 * y) = (1 : ℚ)/y) : y = 1 :=
sorry

end machine_tasks_l712_712227


namespace collinear_T_K_I_l712_712885

noncomputable def intersection (P Q : Set Point) : Point := sorry

variables (A P C Q M N I T K : Point)

-- Definitions based on conditions
def T_def : Point := intersection (line_through A P) (line_through C Q)
def K_def : Point := intersection (line_through M P) (line_through N Q)

-- Proof statement
theorem collinear_T_K_I :
  collinear ({T_def A P C Q, K_def M P N Q, I} : Set Point) := sorry

end collinear_T_K_I_l712_712885


namespace angle_between_vecs_l712_712103

open Real

variables {a b : ℝ} (vec_a vec_b : ℝ → ℝ → ℝ)

-- Conditions
axiom mag_a : ∥vec_a∥ = sqrt 3
axiom mag_b : ∥vec_b∥ = 2
axiom orthogonal : (vec_a - vec_b)•vec_a = 0

-- Question and answer tuple
theorem angle_between_vecs : ∃ θ : ℝ, θ = π / 6 :=
by
  -- Utilize the conditions accordingly to show that θ must be π / 6.
  sorry

end angle_between_vecs_l712_712103


namespace problem_correctness_l712_712252

theorem problem_correctness
  (correlation_A : ℝ)
  (correlation_B : ℝ)
  (chi_squared : ℝ)
  (P_chi_squared_5_024 : ℝ)
  (P_chi_squared_6_635 : ℝ)
  (P_X_leq_2 : ℝ)
  (P_X_lt_0 : ℝ) :
  correlation_A = 0.66 →
  correlation_B = -0.85 →
  chi_squared = 6.352 →
  P_chi_squared_5_024 = 0.025 →
  P_chi_squared_6_635 = 0.01 →
  P_X_leq_2 = 0.68 →
  P_X_lt_0 = 0.32 →
  (abs correlation_B > abs correlation_A) ∧
  (1 - P_chi_squared_5_024 < 0.99) ∧
  (P_X_lt_0 = 1 - P_X_leq_2) ∧
  (false) := sorry

end problem_correctness_l712_712252


namespace quadratic_root_real_coeff_l712_712447

theorem quadratic_root_real_coeff (b c : ℝ) : 
  (∃ x : ℂ, x = 1 + (√2)*complex.I ∧ x*x + b*x + c = 0) → b = -2 ∧ c = 3 :=
begin
  sorry
end

end quadratic_root_real_coeff_l712_712447


namespace part1_part2_l712_712027

open Complex

-- Definitions for part 1
def z : ℂ := (-1/2) + ((sqrt 3 * I) / 2)
def z_conj : ℂ := conj z

theorem part1 : z_conj^2 + z_conj + 1 = 0 :=
sorry

-- Definitions for part 2
def S2016 : ℂ := (Finset.range 2016).sum (λ n, z^n)

theorem part2 (hz3 : z^3 = 1) : S2016 = 0 :=
sorry

end part1_part2_l712_712027


namespace no_two_positive_roots_l712_712554

theorem no_two_positive_roots {n : ℕ} {a : Fin n → ℝ}
  (h : ∀ i, 0 ≤ a i) (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hne : x1 ≠ x2) :
  (∑ i in Finset.range n, a i / x1 ^ (i + 1) = 1) → (∑ i in Finset.range n, a i / x2 ^ (i + 1) = 1) → False :=
sorry

end no_two_positive_roots_l712_712554


namespace triangle_inequality_cosine_l712_712961

theorem triangle_inequality_cosine:
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ 0 < α ∧ α < 180 ∧ 0 < β ∧ β < 180 ∧ 0 < γ ∧ γ < 180 →
  cos (α + 60) + cos (β + 60) + cos (γ + 60) + 3/2 ≥ 0 :=
by
  intro α β γ h
  sorry

end triangle_inequality_cosine_l712_712961


namespace peter_catches_rob_t_l712_712549

noncomputable def t_value (a b c d e : ℝ) := (√((a - c) ^ 2 + (b - d) ^ 2)) / e

theorem peter_catches_rob_t (t : ℝ) : 
  (t = t_value 17 39 0 5 6) :=
by
  sorry

end peter_catches_rob_t_l712_712549


namespace Lara_age_in_10_years_l712_712136

theorem Lara_age_in_10_years (Lara_age_7_years_ago : ℕ) (h1 : Lara_age_7_years_ago = 9) : 
  Lara_age_7_years_ago + 7 + 10 = 26 :=
by
  rw [h1]
  norm_num
  sorry

end Lara_age_in_10_years_l712_712136


namespace kimmie_earnings_l712_712131

theorem kimmie_earnings (K : ℚ) (h : (1/2 : ℚ) * K + (1/3 : ℚ) * K = 375) : K = 450 := 
by
  sorry

end kimmie_earnings_l712_712131


namespace sqrt_sum_eq_seven_l712_712574

theorem sqrt_sum_eq_seven (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
  sqrt (64 - x^2) + sqrt (36 - x^2) = 7 := by
  sorry

end sqrt_sum_eq_seven_l712_712574


namespace sum_of_starting_integers_l712_712727

def machine (N : ℤ) : ℤ :=
  if N % 2 = 1 then 3 * N + 1 else N + 3

def machine_apply (N : ℤ) : ℤ × ℤ × ℤ × ℤ :=
  let N1 := machine N
  let N2 := machine N1
  let N3 := machine N2
  let N4 := machine N3
  (N1, N2, N3, N4)

theorem sum_of_starting_integers :
  (N1, N2, N3, N4) = machine_apply (N : ℤ) → N4 = 14 → (N1 = 11 ∨ N1 = 8 ∨ N2 = 5 ∨ N3 = 2) →
  N = 2 ∨ N = 5 →
  \Sigma (N : ℤ) in (N = 2) ∨ (N = 5), N = 7 := 
by
  sorry

end sum_of_starting_integers_l712_712727


namespace none_of_these_true_l712_712531

variable (s r p q : ℝ)
variable (hs : s > 0) (hr : r > 0) (hpq : p * q ≠ 0) (h : s * (p * r) > s * (q * r))

theorem none_of_these_true : ¬ (-p > -q) ∧ ¬ (-p > q) ∧ ¬ (1 > -q / p) ∧ ¬ (1 < q / p) :=
by
  -- The hypothetical theorem to be proven would continue here
  sorry

end none_of_these_true_l712_712531


namespace factor_quadratic_l712_712347

theorem factor_quadratic : ∀ (x : ℝ), 4 * x^2 - 20 * x + 25 = (2 * x - 5)^2 :=
by
  intro x
  sorry

end factor_quadratic_l712_712347


namespace max_value_l712_712009

noncomputable def max_expression (x : ℝ) : ℝ :=
  3^x - 2 * 9^x

theorem max_value : ∃ x : ℝ, max_expression x = 1 / 8 :=
sorry

end max_value_l712_712009


namespace solution_set_of_inequality_l712_712607

theorem solution_set_of_inequality (x : ℝ) : (x^2 ≤ 1) ↔ (-1 ≤ x ∧ x ≤ 1) := 
by 
  sorry

end solution_set_of_inequality_l712_712607


namespace ellipse_property_l712_712430

open Real

noncomputable def ellipse_equation (a b : ℝ) (ha_gt : a > b) (hb_gt : b > 0) (e : ℝ) (he : e = sqrt 2 / 2) : Prop :=
  ∃ (C : set (ℝ × ℝ)), (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / a^2 + y^2 / b^2 = 1)
  ∧ (1 * 1 / a^2 + (1) * (1) / b^2 = 1)
  ∧ (sqrt (a^2 - b^2) / a = e)
  ∧ (C = {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1})

noncomputable def intersect_and_ratio (m : ℝ) : Prop :=
  ∀ (x_1 y_1 x_2 y_2 k : ℝ), 
    (m^2 + 2) * y_2^2 - 2 * m * y_2 - 1 = 0 
    → y_1 + y_2 = 2 * m / (m^2 + 2)
    → y_1 * y_2 = -1 / (m^2 + 2)
    → let x_0 := (x_1 + x_2) / 2 
       y_0 := (y_1 + y_2) / 2 in
    let x_0 := m * y_0 - 1,
      k = y_0 / x_0 in
    k / m = -1 / 2

theorem ellipse_property (a b : ℝ)
  (ha_gt : a > b) (hb_gt : b > 0) (e : ℝ) (he : e = sqrt 2 / 2)
  (m : ℝ) (intersect : m ≠ 0) :
  ellipse_equation a b ha_gt hb_gt e he ∧ intersect_and_ratio m := 
sorry

end ellipse_property_l712_712430


namespace median_free_throws_l712_712274

def free_throws : list ℕ := [8, 20, 16, 12, 21, 13, 23, 17, 21, 14]

theorem median_free_throws : 
  let sorted_free_throws := free_throws.qsort (≤),
      n := list.length sorted_free_throws,
      median := (sorted_free_throws.nth_le (n / 2 - 1) (by norm_num) + 
                 sorted_free_throws.nth_le (n / 2) (by norm_num)) / 2 in
  (median : ℚ) = 16.5 := 
by
  sorry

end median_free_throws_l712_712274


namespace sum_of_values_for_fx_eq_zero_l712_712941

def f (x : ℝ) : ℝ :=
if x ≤ 2 then -x - 5 else x / 3 + 2

theorem sum_of_values_for_fx_eq_zero : 
  (finset.filter (λ x, f x = 0) (finset.Icc (-5) 2)).sum = -5 := 
by
  sorry

end sum_of_values_for_fx_eq_zero_l712_712941


namespace triangle_def_ef_value_l712_712500

theorem triangle_def_ef_value (E F D : ℝ) (DE DF EF : ℝ) (h1 : E = 45)
  (h2 : DE = 100) (h3 : DF = 100 * Real.sqrt 2) :
  EF = Real.sqrt (30000 + 5000*(Real.sqrt 6 - Real.sqrt 2)) := 
sorry 

end triangle_def_ef_value_l712_712500


namespace B_gt_A_A_C_relationship_l712_712021

variable (a : ℝ)
variable (A := 2 * a - 7)
variable (B := a^2 - 4 * a + 3)
variable (C := a^2 + 6 * a - 28)

theorem B_gt_A (h : a > 2) : B > A :=
by
  have h1 : B - A = (a - 3)^2 + 1 := by
    calc B - A = a^2 - 4 * a + 3 - (2 * a - 7) := by sorry 
           ... = a^2 - 6 * a + 10 := by sorry
           ... = (a - 3)^2 + 1 := by sorry
  have h2 : (a - 3)^2 ≥ 0 := by apply pow_two_nonneg
  have h3 : (a - 3)^2 + 1 > 0 := by linarith
  linarith

theorem A_C_relationship (h : a > 2) : (2 < a ∧ a < 3) → A > C ∧ (a = 3) → A = C ∧ (a > 3) → A < C :=
by
  intro h_range h_eq h_gt
  have h1: C - A = a^2 + 4 * a - 21 := by
    calc C - A = a^2 + 6 * a - 28 - (2 * a - 7) := by sorry
           ... = a^2 + 4 * a - 21 := by sorry
  have h2 : C - A = (a + 7) * (a - 3) := by sorry

  by_cases h_cases : (2 < a ∧ a < 3) ∨ (a = 3) ∨ (a > 3)
  case inl ih : {
    sorry -- Assuming '2 < a < 3' implies 'A > C'
  }
  case inr {
    cases h_cases with h_eq h_gt
    case inl ih {
      sorry -- Assuming 'a = 3' implies 'A = C'
    }
    case inr ih {
      sorry -- Assuming 'a > 3' implies 'A < C'
    } 
  }
  linarith

end B_gt_A_A_C_relationship_l712_712021


namespace inequality_proof_l712_712555

theorem inequality_proof : 
  (∑ k in Finset.range 500, (Real.cbrt (2 * (k + 1)) - Real.cbrt (2 * k + 1))) > 9 / 2 :=
by
  sorry

end inequality_proof_l712_712555


namespace incorrect_analysis_l712_712642

noncomputable def genotypeA := (A : Type) × (B : Type)
noncomputable def genotypeB := (A : Type) × (A : Type) × (B : Type) — × (b : Type)

axiom individual_A : genotypeA
axiom individual_B : genotypeB

theorem incorrect_analysis (analysis: Prop):
  analysis = "Gene mutation" →
  ¬("Gene mutation" leads to a decrease in the number of genes for individual A with genotype (AaB : genotypeA) given individual B has genotype (AABb : genotypeB)) :=
by {
  sorry
}

end incorrect_analysis_l712_712642


namespace ratio_triangle_areas_rational_l712_712399

open Real

-- Definitions of the points and segments
variables (A B C D : ℝ × ℝ)
variable h_no_collinear : ¬ collinear {A, B, C} ∧ ¬ collinear {A, B, D} ∧ ¬ collinear {A, C, D} ∧ ¬ collinear {B, C, D}
variables (h_rational_sq_lengths : ∃ a b c d e f : ℚ,
  dist A B ^ 2 = a ∧
  dist A C ^ 2 = b ∧
  dist A D ^ 2 = c ∧
  dist B C ^ 2 = d ∧
  dist B D ^ 2 = e ∧
  dist C D ^ 2 = f)

-- Main statement
theorem ratio_triangle_areas_rational :
  let S_triangle (P Q R : ℝ × ℝ) := 
    let a := dist P Q
    let b := dist P R
    let c := dist Q R
    Real.abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)
  in (S_triangle A B C) / (S_triangle A B D) ∈ ℚ :=
sorry

end ratio_triangle_areas_rational_l712_712399


namespace min_value_of_a_l712_712786

theorem min_value_of_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y))) : 
  a ≥ Real.sqrt 2 :=
sorry -- Proof is omitted

end min_value_of_a_l712_712786


namespace transformed_triangle_area_l712_712975

-- Define the function and the given conditions
variable (x1 x2 x3 : ℝ)
variable (f : ℝ → ℝ)
variable (area : ℝ)

-- Assume the area of the original triangle is given as 50
axiom area_original : area = 50

-- Definition of the points on the graph of y = f(x)
def points_original : List (ℝ × ℝ) := [(x1, f x1), (x2, f x2), (x3, f x3)]

-- Definition of the points on the graph of y = 3f(3x)
def points_transformed : List (ℝ × ℝ) := [(x1 / 3, 3 * f x1), (x2 / 3, 3 * f x2), (x3 / 3, 3 * f x3)]

-- Lean statement to prove the area of the transformed triangle is 50
theorem transformed_triangle_area : area = 50 :=
sorry

end transformed_triangle_area_l712_712975


namespace count_integer_values_for_expression_l712_712364

theorem count_integer_values_for_expression :
  ∃ n_set : Set ℤ, (∀ n ∈ n_set, (3200 * (5 / 2 : ℚ)^n).denom = 1) ∧ n_set.count.to_nat = 11 := sorry

end count_integer_values_for_expression_l712_712364


namespace f_increasing_h_sum_bound_l712_712383

-- Define the function f
def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- Define the monotonicity condition for f
theorem f_increasing : ∀ x y : ℝ, x < y → f(x) < f(y) := sorry

-- Define the function g
def g (x : ℕ) : ℝ := (1 / 2) * (1 - f(x))

-- Define the function h
def h (x : ℕ) : ℝ := 2^x * g(x) * g(x + 1)

-- Define the summation condition for h
theorem h_sum_bound (x : ℕ) (hx : 0 < x) : 
  ∑ k in finset.range(x).filter (λ n, 0 < n), h (k + 1) < 1 / 3 := sorry

end f_increasing_h_sum_bound_l712_712383


namespace time_spent_on_bus_l712_712559

/-
Given conditions:
- Samantha catches the bus at 8:00 a.m. and arrives home at 5:30 p.m.
- She has 7 classes, each lasting 45 minutes.
- She has a lunch break of 45 minutes.
- She spends 1.5 hours in the chess club after classes.
We need to prove that the time Samantha spent on the bus is 120 minutes.
-/

constant bus_departure_time : ℕ := 8 -- 8:00 a.m.
constant bus_arrival_time : ℕ := 17 * 60 + 30 -- 5:30 p.m. in minutes

constant class_count : ℕ := 7
constant class_duration : ℕ := 45 -- each class is 45 minutes
constant lunch_duration : ℕ := 45 -- lunch is 45 minutes
constant chess_club_duration : ℕ := 1.5 * 60 -- chess club is 1.5 hours in minutes

constant total_time_Away : ℕ := bus_arrival_time - (bus_departure_time * 60) -- total time away in minutes
constant total_time_School_Activities : ℕ := (class_count * class_duration) + lunch_duration + chess_club_duration -- total time in school activities in minutes

theorem time_spent_on_bus : total_time_Away - total_time_School_Activities = 120 := sorry

end time_spent_on_bus_l712_712559


namespace ratio_of_largest_to_sum_l712_712730

theorem ratio_of_largest_to_sum :
  let S := {1, 10, 10^2, 10^3, ..., 10^13} in
  let largest := 10^13 in
  let sum_of_others := 1 + 10 + 10^2 + ... + 10^12 in
  let ratio := largest / sum_of_others in
  abs (ratio - 9) < 1 :=
by
  sorry

end ratio_of_largest_to_sum_l712_712730


namespace point_translation_quadrant_l712_712852

theorem point_translation_quadrant :
  let P := (5, 3)
  let translated_P := (P.1, P.2 - 4) 
  (translated_P.1 > 0 ∧ translated_P.2 < 0) := 
  by
    let P := (5, 3)
    let translated_P := (P.1, P.2 - 4)
    show (translated_P.1 > 0 ∧ translated_P.2 < 0)
    from sorry

end point_translation_quadrant_l712_712852


namespace complement_A_union_B_l712_712524

-- Define the universal set U, and the sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- Lean statement to prove the complement of A ∪ B with respect to U
theorem complement_A_union_B : U \ (A ∪ B) = {7, 8} :=
by
sorry

end complement_A_union_B_l712_712524


namespace acute_angle_implies_x_range_l712_712813

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
def vector_b : ℝ × ℝ := (2, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem acute_angle_implies_x_range (x : ℝ) :
  2 * x + 12 > 0 ∧ x ≠ 3 / 2 → x ∈ set.Ioo (-6) (3 / 2) ∪ set.Ioi (3 / 2) :=
sorry

end acute_angle_implies_x_range_l712_712813


namespace tv_price_reduction_l712_712212

theorem tv_price_reduction (x : ℝ) (Q : ℝ) (P : ℝ) (h1 : Q > 0) (h2 : P > 0) (h3 : P*(1 - x/100) * 1.85 * Q = 1.665 * P * Q) : x = 10 :=
by 
  sorry

end tv_price_reduction_l712_712212


namespace distinct_special_sums_l712_712319

def is_special_fraction (a b : ℕ) : Prop := a + b = 18

def is_special_sum (n : ℤ) : Prop :=
  ∃ (a1 b1 a2 b2 : ℕ), is_special_fraction a1 b1 ∧ is_special_fraction a2 b2 ∧ 
  n = (a1 : ℤ) * (b2 : ℤ) * b1 + (a2 : ℤ) * (b1 : ℤ) / a1

theorem distinct_special_sums : 
  (∃ (sums : Finset ℤ), 
    (∀ n, n ∈ sums ↔ is_special_sum n) ∧ 
    sums.card = 7) :=
sorry

end distinct_special_sums_l712_712319


namespace find_triangle_side1_l712_712602

def triangle_side1 (Perimeter Side2 Side3 Side1 : ℕ) : Prop :=
  Perimeter = Side1 + Side2 + Side3

theorem find_triangle_side1 :
  ∀ (Perimeter Side2 Side3 Side1 : ℕ), 
    (Perimeter = 160) → (Side2 = 50) → (Side3 = 70) → triangle_side1 Perimeter Side2 Side3 Side1 → Side1 = 40 :=
by
  intros Perimeter Side2 Side3 Side1 h1 h2 h3 h4
  sorry

end find_triangle_side1_l712_712602


namespace sum_of_first_21_terms_l712_712114

variable (a_n : ℕ → ℝ)

-- Define that the sequence is arithmetic: a_n = a_1 + (n - 1) * d
def arithmetic_sequence (a_1 d : ℝ) (n : ℕ) : ℝ := a_1 + (n - 1) * d 

-- Define condition
def condition (a_1 d : ℝ) : Prop :=
  (arithmetic_sequence a_1 d 6) + (arithmetic_sequence a_1 d 9) + (arithmetic_sequence a_1 d 13) + (arithmetic_sequence a_1 d 16) = 20

-- Define S_n as the sum of first n terms of arithmetic sequence
def sum_arithmetic_sequence (a_1 d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_21_terms (a_1 d : ℝ) (h : condition a_1 d) :
  sum_arithmetic_sequence a_1 d 21 = 105 :=
sorry

end sum_of_first_21_terms_l712_712114


namespace tangent_dot_product_min_value_l712_712386

variables 
  (O : Type*) [metric_space O] [normed_group O] [normed_space ℝ O]
  (P A B : O)
  (radius : ℝ) (h_radius : radius = 1)
  (h_tangent1 : ∃ u : ℝ, (P - A) = u • (A - O) ∧ ∥A - O∥ = radius)
  (h_tangent2 : ∃ v : ℝ, (P - B) = v • (B - O) ∧ ∥B - O∥ = radius)

theorem tangent_dot_product_min_value : 
  let x := ((P - A) / ∥P - A∥) • ((P - B) / ∥P - B∥) in
  (infimum x) = -3 + 2 * real.sqrt 2 := 
sorry

end tangent_dot_product_min_value_l712_712386


namespace energy_saving_lamp_problem_l712_712485

-- Define the conditions as variables
variables {a x y : ℕ} {W : ℕ}

-- Establish the conditions given in the problem
def cost_equation_1 : Prop := 3 * x + 5 * y = 50
def cost_equation_2 : Prop := x + 3 * y = 26
def total_lamps (a : ℕ) : Prop := a + (200 - a) = 200

-- Define the known solutions for the lamp prices
def price_of_lamps : Prop := x = 5 ∧ y = 7

-- Define the functional relationship W = -2a + 1400
def functional_relationship (a : ℕ) : Prop := W = -2 * a + 1400

-- Define the specific case where a = 80
def total_cost_when_a_80 : Prop := a = 80 → W = 1240

-- Combine everything into one theorem to be proven
theorem energy_saving_lamp_problem :
  (cost_equation_1 ∧ cost_equation_2 ∧ total_lamps a) →
  (price_of_lamps ∧ functional_relationship a ∧ total_cost_when_a_80) := 
sorry

end energy_saving_lamp_problem_l712_712485


namespace equivalent_expression_l712_712736

theorem equivalent_expression (x : ℝ) (h : x < 0) :
  (sqrt (x / (1 - (x - 2) / x))) = -x / sqrt 2 :=
sorry

end equivalent_expression_l712_712736


namespace jill_total_tax_percentage_l712_712656

theorem jill_total_tax_percentage :
  ∀ (total_spent : ℝ),
  let clothing_spent := 0.60 * total_spent in
  let food_spent := 0.10 * total_spent in
  let other_spent := 0.30 * total_spent in
  let clothing_tax := 0.04 * clothing_spent in
  let food_tax := 0 in
  let other_tax := 0.08 * other_spent in
  let total_tax := clothing_tax + food_tax + other_tax in
  (total_tax / total_spent) * 100 = 4.8 :=
by
  intros total_spent
  let clothing_spent := 0.60 * total_spent
  let food_spent := 0.10 * total_spent
  let other_spent := 0.30 * total_spent
  let clothing_tax := 0.04 * clothing_spent
  let food_tax := 0
  let other_tax := 0.08 * other_spent
  let total_tax := clothing_tax + food_tax + other_tax
  have : (total_tax / total_spent) * 100 = 4.8 := sorry
  exact this

end jill_total_tax_percentage_l712_712656


namespace triangle_area_ratio_l712_712503

theorem triangle_area_ratio (XY YZ ZX s t u : ℝ)
  (h1 : XY = 14) (h2 : YZ = 16) (h3 : ZX = 18)
  (h4 : s + t + u = 3/4) (h5 : s^2 + t^2 + u^2 = 3/8) :
  let r := xyz_area_ratio XY YZ ZX s t u in r = 9/32  :=
by { sorry }

end triangle_area_ratio_l712_712503


namespace area_original_triangle_l712_712424

theorem area_original_triangle (a : ℝ) :
  let area_projection := (√3 / 4) * a^2 in
  let area_original := (√6 / 2) * a^2 in
  (√2 / 4) * area_original = area_projection :=
by
  let area_projection := (√3 / 4) * a^2
  let area_original := (√6 / 2) * a^2
  have h : (√2 / 4) * area_original = area_projection := by sorry
  exact h

end area_original_triangle_l712_712424


namespace concyclic_amnd_l712_712144

-- Define necessary parameters and conditions
variables {A B C D E M N : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point M] [Point N]

-- Inscribed quadrilateral condition
def inscribed (A B C D : Type) : Prop :=
  Circle.is_cyclic A B C D

-- Tangency condition
def tangent_at (D A : Type) (AB : Line) : Prop :=
  Circle.is_tangent_through D A AB

-- Midpoint condition
def midpoint (X Y P : Type) : Prop :=
  X = (Y + P) / 2

-- Concyclic points
def concyclic (A M N D : Type) : Prop :=
  Circle.is_cyclic A M N D

-- Theorem statement
theorem concyclic_amnd (h1 : inscribed A B C D)
                      (h2 : tangent_at D A (Line AB))
                      (h3 : intersects (Circle_through_D_tangent_to_AB_at_A) (Line CD) E)
                      (h4 : midpoint M A B)
                      (h5 : midpoint N C E) :
    concyclic A M N D :=
sorry

end concyclic_amnd_l712_712144


namespace line_NT_passes_through_midpoint_of_AC_l712_712577

open EuclideanGeometry

variables {A B C K N P T E : Point} (AB BC : Line) {circumcircle : Circle}
          (abc_triangle : Triangle A B C) (mk_angle_bisector : Line) 

def midpoint (X Y : Point) : Point := sorry -- Definition of midpoint

def perp_foot (X Y : Point) (l : Line) : Point := sorry -- Perpendicular foot from X to line l

def bisect_angle (X Y Z : Point) (l : Line) : Prop := sorry -- l is the angle bisector of ∠XYZ

def parallel (l1 l2 : Line) : Prop := sorry -- l1 is parallel to l2

theorem line_NT_passes_through_midpoint_of_AC :
  let circumcircle := circumscribe_circle abc_triangle in
  let K := intersection (angle_bisector A B C) circumcircle in
  let N := perp_foot K A AB in
  let P := midpoint N B in
  let T := intersection (line_through_point_parallel P BC) (line_through_points B K) in
  let E := midpoint A C in
  NT ∧ passes_through E :=
  sorry

end line_NT_passes_through_midpoint_of_AC_l712_712577


namespace sum_mod_9_l712_712012

theorem sum_mod_9 : (7155 + 7156 + 7157 + 7158 + 7159) % 9 = 1 :=
by sorry

end sum_mod_9_l712_712012


namespace avg_of_ab_l712_712980

variable {a b c : ℝ}

-- Conditions
def avg_ab : Prop := (a + b) / 2 = 45
def avg_bc : Prop := (b + c) / 2 = 70
def diff_ca : Prop := c - a = 50

theorem avg_of_ab (h1 : avg_ab) (h2 : avg_bc) (h3 : diff_ca) : (a + b) / 2 = 45 := by
  sorry

end avg_of_ab_l712_712980


namespace cubs_more_home_runs_than_cardinals_l712_712739

theorem cubs_more_home_runs_than_cardinals 
(h1 : 2 + 1 + 2 = 5) 
(h2 : 1 + 1 = 2) : 
5 - 2 = 3 :=
by sorry

end cubs_more_home_runs_than_cardinals_l712_712739


namespace p_squared_plus_one_over_p_squared_plus_six_l712_712087

theorem p_squared_plus_one_over_p_squared_plus_six (p : ℝ) (h : p + 1/p = 10) : p^2 + 1/p^2 + 6 = 104 :=
by {
  sorry
}

end p_squared_plus_one_over_p_squared_plus_six_l712_712087


namespace conditional_probabilities_l712_712183

noncomputable def total_outcomes := 6 * 6 * 6

noncomputable def event_A_outcomes := 6 * 5 * 4

noncomputable def event_B_outcomes := total_outcomes - 5 * 5 * 5

noncomputable def event_A_and_B_outcomes := 3 * 5 * 4

theorem conditional_probabilities :
  (event_A_and_B_outcomes.toRat / event_B_outcomes.toRat = 60 / 91) ∧ 
  (event_A_and_B_outcomes.toRat / event_A_outcomes.toRat = 1 / 2) :=
by
  have total_outcomes_value : total_outcomes = 216 := by decide
  have event_A_value : event_A_outcomes = 120 := by decide
  have event_B_value : event_B_outcomes = 91 := by decide
  have event_A_and_B_value : event_A_and_B_outcomes = 60 := by decide
  rw [event_A_value, event_B_value, event_A_and_B_value]
  split
  case left => exact (by norm_num)
  case right => exact (by norm_num)
  sorry

end conditional_probabilities_l712_712183


namespace calc1_calc2_l712_712316

theorem calc1 : (1 * -11 + 8 + (-14) = -17) := by
  sorry

theorem calc2 : (13 - (-12) + (-21) = 4) := by
  sorry

end calc1_calc2_l712_712316


namespace gcd_le_sqrt_sum_l712_712037

theorem gcd_le_sqrt_sum {a b : ℕ} (h : ∃ k : ℕ, (a + 1) / b + (b + 1) / a = k) :
  ↑(Nat.gcd a b) ≤ Real.sqrt (a + b) := sorry

end gcd_le_sqrt_sum_l712_712037


namespace trigonometric_identities_l712_712025

variable (α : ℝ)

-- Given conditions
axiom cos_pi_plus_alpha : cos (π + α) = -3/5
axiom alpha_range : α ∈ Ioc (3 * π / 2) (2 * π)

-- Prove the following:
theorem trigonometric_identities 
  (h1: cos (π + α) = -3/5) 
  (h2: α ∈ Ioc (3 * π / 2) (2 * π)) : 
  (sin (π / 2 + α) = 3/5) ∧ 
  (cos (2 * α) = -7/25) ∧ 
  (sin (α - π / 4) = -7 * sqrt 2 / 10) ∧ 
  (tan (α / 2) = -1/2) := 
sorry

end trigonometric_identities_l712_712025


namespace smallest_sum_of_squares_l712_712986

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l712_712986


namespace max_value_not_sqrt3_over_2_plus_1_symmetry_about_2pi_over_3_not_monotonically_increasing_minimum_positive_period_pi_l712_712018

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3) + 1

theorem max_value_not_sqrt3_over_2_plus_1 : 
  ¬(∃ x ∈ set.Icc 0 (Real.pi / 2), f x = (Real.sqrt 3 / 2) + 1) := sorry

theorem symmetry_about_2pi_over_3 :
  ∀ x : ℝ, f (2 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x) := sorry

theorem not_monotonically_increasing :
  ¬monotone_on f (set.Ioo 0 (Real.pi / 2)) := sorry

theorem minimum_positive_period_pi :
  ∀ x : ℝ, f (x + Real.pi) = f x := sorry

end max_value_not_sqrt3_over_2_plus_1_symmetry_about_2pi_over_3_not_monotonically_increasing_minimum_positive_period_pi_l712_712018


namespace angle_OQP_eq_90_l712_712418

variables {A B C D P Q O : Type*}
variables [circle O A B C D]

-- Conditions
-- We assume that ABCD is a convex quadrilateral inscribed in a circle O
def convex_quadrilateral_inscribed (O : Type*) (A B C D : Type*) : Prop := sorry

-- Diagonals AC and BD intersect at point P
def diagonals_intersect (A C : Type*) (B D : Type*) (P : Type*) : Prop := sorry 

-- Circumcircles of triangles ABP and CDP intersect at points P and Q
def circumcircles_intersect (A B P : Type*) (C D P : Type*) (Q : Type*) : Prop := sorry 

-- O, P, and Q are distinct
def distinct_points (O P Q : Type*) : Prop := O ≠ P ∧ P ≠ Q ∧ O ≠ Q

-- The statement we aim to prove
theorem angle_OQP_eq_90 (h1 : convex_quadrilateral_inscribed O A B C D)
                        (h2 : diagonals_intersect A C B D P)
                        (h3 : circumcircles_intersect A B P C D P Q)
                        (h4 : distinct_points O P Q) :
  ∠ O Q P = 90 :=
sorry

end angle_OQP_eq_90_l712_712418


namespace tangent_line_at_zero_is_x_plus_one_l712_712050

variable (f : ℝ → ℝ)
variable (mono_f : Monotone f)
variable (h : ∀ x, f (f x - Real.exp x) = 1)

theorem tangent_line_at_zero_is_x_plus_one :
  ∃ (m b : ℝ), m = 1 ∧ b = 1 ∧ (∀ x, f x = Real.exp x → has_deriv_at f (Real.exp 0) 0 ∧ f 0 = 1 → (∀ y : ℝ, y = f(0) + (x - 0) * m) = (y = x + 1)) :=
sorry

end tangent_line_at_zero_is_x_plus_one_l712_712050


namespace no_consecutive_integers_square_difference_2000_l712_712098

theorem no_consecutive_integers_square_difference_2000 :
  ¬ ∃ a : ℤ, (a + 1) ^ 2 - a ^ 2 = 2000 :=
by {
  -- some detailed steps might go here in a full proof
  sorry
}

end no_consecutive_integers_square_difference_2000_l712_712098


namespace HIJK_is_square_l712_712809

noncomputable def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def is_square_HIJK (a : ℝ) : Prop :=
  let A := (0, 0) in
  let B := (a, 0) in
  let C := (a, a) in
  let D := (0, a) in
  let E := (-a, a) in
  let G := (0, 2 * a) in
  let H := midpoint A G in
  let I := midpoint G E in
  let J := midpoint E C in
  let K := midpoint C A in
  ∃ (H I J K : (ℝ × ℝ)),
    H = (0, a) ∧
    I = (-a / 2, 3 * a / 2) ∧
    J = (0, a) ∧
    K = (a / 2, a / 2) ∧
    (∃ (α β : ℝ), α * α = a * a / 2 ∧ β * β = a * a / 2 ∧ α = β ∧ α ≠ 0)

theorem HIJK_is_square (a : ℝ) (h : a ≠ 0) : is_square_HIJK a := by
  sorry

end HIJK_is_square_l712_712809


namespace original_amount_of_money_l712_712181

variable (took : ℕ) (now : ℕ) (initial : ℕ)

-- conditions from the problem
def conditions := (took = 2) ∧ (now = 3)

-- the statement to prove
theorem original_amount_of_money {took now initial : ℕ} (h : conditions took now) :
  initial = now + took ↔ initial = 5 :=
by {
  sorry
}

end original_amount_of_money_l712_712181


namespace triangle_ratio_l712_712861

universe u
variables {α : Type u} [Field α]

structure Point (α : Type u) :=
(x : α) (y : α)

noncomputable def ratio (a b : α) := a / b

theorem triangle_ratio {A B C D F P : Point α}
  (D_on_BC : ∃ t : α, (0 ≤ t ∧ t ≤ 1) ∧ D = Point.mk ((1-t)*B.x + t*C.x) ((1-t)*B.y + t*C.y))
  (F_on_AB : ∃ t : α, (0 ≤ t ∧ t ≤ 1) ∧ F = Point.mk ((1-t)*A.x + t*B.x) ((1-t)*A.y + t*B.y))
  (AD_p_FintP : ∃ t : α, (0 ≤ t ∧ t ≤ 1) ∧ P = Point.mk ((1-t)*A.x + t*D.x) ((1-t)*A.y + t*D.y))
  (CF_p_FintP : ∃ t : α, (0 ≤ t ∧ t ≤ 1) ∧ P = Point.mk ((1-t)*C.x + t*F.x) ((1-t)*C.y + t*F.y))
  (AP_PD : ratio (distance P A) (distance P D) = 3 / 2)
  (FP_PC : ratio (distance P F) (distance P C) = 3 / 1) :
  ratio (distance D B) (distance D C) = 2 / 3 :=
sorry

end triangle_ratio_l712_712861


namespace car_speed_l712_712277

theorem car_speed(
  (normal_fuel_efficiency : ℝ) (normal_fuel_efficiency = 72) 
  (fuel_decrease_gallons : ℝ) (fuel_decrease_gallons = 3.9) 
  (time_hours : ℝ) (time_hours = 5.7)
  (headwind_reduction : ℝ) (headwind_reduction = 0.15)
  (gallon_to_liter : ℝ) (gallon_to_liter = 3.8)
  (km_to_mile : ℝ) (km_to_mile = 1.6)
) : ℝ :=
  let new_fuel_efficiency := normal_fuel_efficiency * (1.0 - headwind_reduction)
  let fuel_used_liters := fuel_decrease_gallons * gallon_to_liter
  let distance_traveled := fuel_used_liters * new_fuel_efficiency
  let speed_kmh := distance_traveled / time_hours
  let speed_mph := speed_kmh / km_to_mile
  speed_mph ≈ 99.488

end car_speed_l712_712277


namespace inequality_proof_l712_712539

theorem inequality_proof 
  (n : ℕ) 
  (a b x : Fin n → ℝ)
  (h_not_proportional : ¬ ∃ k : ℝ, ∀ i, a i = k * b i)
  (h1 : ∑ i, a i * x i = 0) 
  (h2 : ∑ i, b i * x i = 1) :
  (∑ i, (x i)^2) ≥ (∑ i, (a i)^2) / ((∑ i, (a i)^2) * (∑ i, (b i)^2) - (∑ i, a i * b i)^2) :=
sorry

end inequality_proof_l712_712539


namespace savings_amount_correct_l712_712192

-- Define the problem conditions
def principal := 150000.0
def total_expenses := 182200.0

-- Define individual expenses
def airplane_tickets := 61200.0
def accommodation := 65000.0
def food := 36000.0
def excursions := 20000.0

-- Define interest rates
def bettaBank_rate := 0.036
def gammaBank_rate := 0.045
def omegaBank_rate := 0.0312
def epsilonBank_monthly_rate := 0.0025

-- Define compounding periods and time in years
def bettaBank_n := 12
def gammaBank_n := 1
def omegaBank_n := 4
def epsilonBank_n := 12
def t := 0.5

-- Calculate interests and future values
def bettaBank_interest := principal * (1 + bettaBank_rate / bettaBank_n) ^ (bettaBank_n * t) - principal
def gammaBank_interest := principal * (1 + gammaBank_rate * t) - principal
def omegaBank_interest := principal * (1 + omegaBank_rate / omegaBank_n) ^ (omegaBank_n * t) - principal
def epsilonBank_interest := principal * (1 + epsilonBank_monthly_rate) ^ epsilonBank_n / 2 - principal

-- Calculate the required savings from salary
def bettaBank_savings := total_expenses - principal - bettaBank_interest
def gammaBank_savings := total_expenses - principal - gammaBank_interest
def omegaBank_savings := total_expenses - principal - omegaBank_interest
def epsilonBank_savings := total_expenses - principal - epsilonBank_interest

-- The proof problem
theorem savings_amount_correct :
  bettaBank_savings = 29479.67 ∧
  gammaBank_savings = 28825 ∧
  omegaBank_savings = 29850.87 ∧
  epsilonBank_savings = 29935.89 :=
by
  sorry

end savings_amount_correct_l712_712192


namespace instantaneous_velocity_at_3_l712_712834

-- Define the position function s
def position (t : ℝ) : ℝ := 3 * t^2

-- Define the velocity function as the derivative of the position function
def velocity (t : ℝ) : ℝ := deriv position t

-- State the theorem for the given problem
theorem instantaneous_velocity_at_3 : velocity 3 = 18 := by
  sorry

end instantaneous_velocity_at_3_l712_712834


namespace exam_question_correct_count_l712_712849

theorem exam_question_correct_count (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 110) : C = 34 :=
by
  sorry

end exam_question_correct_count_l712_712849


namespace Leonard_is_11_l712_712878

def Leonard_age (L N J P T: ℕ) : Prop :=
  (L = N - 4) ∧
  (N = J / 2) ∧
  (P = 2 * L) ∧
  (T = P - 3) ∧
  (L + N + J + P + T = 75)

theorem Leonard_is_11 (L N J P T : ℕ) (h : Leonard_age L N J P T) : L = 11 :=
by {
  sorry
}

end Leonard_is_11_l712_712878


namespace range_of_m_l712_712760

theorem range_of_m (y : ℝ) (x : ℝ) (xy_ne_zero : x * y ≠ 0) :
  (x^2 + 4 * y^2 = (m^2 + 3 * m) * x * y) → -4 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l712_712760


namespace divides_power_of_odd_l712_712158

theorem divides_power_of_odd (k : ℕ) (hk : k % 2 = 1) (n : ℕ) (hn : n ≥ 1) : 2^(n + 2) ∣ (k^(2^n) - 1) :=
by
  sorry

end divides_power_of_odd_l712_712158


namespace range_of_m_for_two_solutions_l712_712455

theorem range_of_m_for_two_solutions (x m : ℝ) (h₁ : x > 1) :
  2 * log x / log 2 - log (x - 1) / log 2 = m → (2:ℝ) < m := 
sorry

end range_of_m_for_two_solutions_l712_712455


namespace sufficient_and_necessary_condition_l712_712540

def A : Set ℝ := { x | x - 2 > 0 }

def B : Set ℝ := { x | x < 0 }

def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem sufficient_and_necessary_condition :
  ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C :=
sorry

end sufficient_and_necessary_condition_l712_712540


namespace concyclic_OAPQ_l712_712308

variables {α : Type*} [euclidean_geometry α]

-- Definitions and conditions translated into Lean 4

-- Given triangle with AB = AC
variables {A B C P Q O : α}
variables (h1 : eq_dist A B A C)

-- Extended AP = BQ
variables (h2 : collinear C A P)
variables (h3 : collinear A B Q)
variables (h4 : eq_dist A P B Q)

-- O is the circumcenter of triangle ABC
variable (h5 : is_circumcenter O A B C)

-- The statement to be proved
theorem concyclic_OAPQ : concyclic O A P Q :=
sorry  -- Proof should be filled in

end concyclic_OAPQ_l712_712308


namespace increasing_interval_of_f_l712_712008

noncomputable def f (x : ℝ) := real.logb 3 (-x^2 + 5 * x - 6)

lemma domain_of_f : ∀ x, 2 < x ∧ x < 3 → -x^2 + 5 * x - 6 > 0 :=
by sorry

lemma increasing_interval_of_t : ∀ x, 2 < x ∧ x ≤ 5 / 2 → 
  -x^2 + 5 * x - 6 is_strictly_increasing :=
by sorry

theorem increasing_interval_of_f : ∀ x, 2 < x ∧ x ≤ 5 / 2 → 
  monotone_on f (set.Ioo 2 (5 / 2)) :=
by 
  intros x h
  apply increasing_interval_of_t
  exact h
  sorry

end increasing_interval_of_f_l712_712008


namespace points_C_satisfying_conditions_l712_712476

theorem points_C_satisfying_conditions :
  ∀ (A B C : ℝ × ℝ),
    dist A B = 12 →
    ((∃ x y, C = (x, y) ∧ 1/2 * ∣(B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)∣ = 144)
    ∧ (∃ x y, C = (x, y) ∧ dist A C + dist B C + dist A B = 60)) →
    (∃ (a b : ℝ × ℝ), a ≠ b ∧ (∃ x y, a = (x, y) ∧ 1/2 * ∣(B.1 - A.1) * (a.2 - A.2) - (B.2 - A.2) * (a.1 - A.1)∣ = 144)
    ∧ (∃ x y, a = (x, y) ∧ dist A a + dist B a + dist A B = 60)
    ∧ (∃ x y, b = (x, y) ∧ 1/2 * ∣(B.1 - A.1) * (b.2 - A.2) - (B.2 - A.2) * (b.1 - A.1)∣ = 144)
    ∧ (∃ x y, b = (x, y) ∧ dist A b + dist B b + dist A B = 60)) :=
begin
  intros A B C hAB hCconds,
  sorry
end

end points_C_satisfying_conditions_l712_712476


namespace product_of_fractions_l712_712243

theorem product_of_fractions (a b c d e f : ℚ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) 
  (h_d : d = 2) (h_e : e = 3) (h_f : f = 4) :
  (a / b) * (d / e) * (c / f) = 1 / 4 :=
by
  sorry

end product_of_fractions_l712_712243


namespace books_in_special_collection_at_beginning_of_month_l712_712690

theorem books_in_special_collection_at_beginning_of_month
  (loaned_out_real : Real)
  (loaned_out_books : Int)
  (returned_ratio : Real)
  (books_at_end : Int)
  (B : Int)
  (h1 : loaned_out_real = 49.99999999999999)
  (h2 : loaned_out_books = 50)
  (h3 : returned_ratio = 0.70)
  (h4 : books_at_end = 60)
  (h5 : loaned_out_books = Int.floor loaned_out_real)
  (h6 : ∀ (loaned_books : Int), loaned_books ≤ loaned_out_books → returned_ratio * loaned_books + (loaned_books - returned_ratio * loaned_books) = loaned_books)
  : B = 75 :=
by
  sorry

end books_in_special_collection_at_beginning_of_month_l712_712690


namespace perpendicular_line_l712_712201

open Real

-- Definition of slope and point
def line_through_point (m : ℝ) (x₁ y₁ : ℝ) : ℝ → ℝ := λ x, m * (x - x₁) + y₁

theorem perpendicular_line 
  (x₁ y₁ : ℝ)
  (h : x₁ = -1 ∧ y₁ = 0)
  (line : ℝ → ℝ) 
  (h_perpendicular : ∀ x, line x = -x) : 
  ∀ x, (line_through_point 1 (-1) 0) x = x + 1 :=
by
  sorry

end perpendicular_line_l712_712201


namespace general_formula_an_Tn_le_four_ninths_l712_712948

-- Definitions
def a1 : ℤ := 9
def a2 : ℤ := sorry -- Placeholder, as the exact integer is not specified
def S (n : ℕ) : ℕ := sorry -- Placeholder, as we don't have an explicit formula for the sum
def Sn_le_S5 (n : ℕ) : Prop := S n ≤ S 5

-- Propositions
theorem general_formula_an (n : ℕ) (hn : Sn_le_S5 n) : 
  ∃ d : ℤ, d = -2 ∧ (∀ n : ℕ, a n = 11 - 2 * n) := sorry

theorem Tn_le_four_ninths (n : ℕ) (general_formula : ∀ n : ℕ, a n = 11 - 2 * n) : 
  let T (n : ℕ) := ∑ i in finset.range n, 1 / (a i * a (i + 1)) 
  in T n ≤ 4 / 9 := sorry

end general_formula_an_Tn_le_four_ninths_l712_712948


namespace probability_at_least_3_laughs_l712_712837

theorem probability_at_least_3_laughs (p : ℚ) (n : ℕ) (h_p : p = 1/3) (h_n : n = 7) :
  (1 - ((2/3)^7 
       + ((nat.choose 7 1) * (1/3) * (2/3)^6) 
       + ((nat.choose 7 2) * (1/3)^2 * (2/3)^5))) = 939 / 2187 := 
by 
  sorry

end probability_at_least_3_laughs_l712_712837


namespace a_plus_b_eq_l712_712056

-- Define the sets A and B
def A := { x : ℝ | -1 < x ∧ x < 3 }
def B := { x : ℝ | -3 < x ∧ x < 2 }

-- Define the intersection set A ∩ B
def A_inter_B := { x : ℝ | -1 < x ∧ x < 2 }

-- Define a condition
noncomputable def is_solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, (-1 < x ∧ x < 2) ↔ (x^2 + a * x + b < 0)

-- The proof statement
theorem a_plus_b_eq : ∃ a b : ℝ, is_solution_set a b ∧ a + b = -3 := by
  sorry

end a_plus_b_eq_l712_712056


namespace collinear_TKI_l712_712895

-- Definitions for points and lines
variables {A P C Q M N I T K : Type}
variable {line : Type → Type}
variables (AP : line A → line P) (CQ : line C → line Q) (MP : line M → line P) (NQ : line N → line Q)

-- Conditions from the problem
-- Assume there exist points T and K which are intersections of the specified lines
axiom intersects_AP_CQ : ∃ (T : Type), AP T = CQ T
axiom intersects_MP_NQ : ∃ (K : Type), MP K = NQ K

-- Collinearity of points T, K, and I
theorem collinear_TKI : ∀ (I : Type) (T : Type) (K : Type),
  intersects_AP_CQ → intersects_MP_NQ → collinear I T K :=
by sorry

end collinear_TKI_l712_712895


namespace number_of_integer_pairs_l712_712767

theorem number_of_integer_pairs (n : ℕ) : 
  ∃ (count : ℕ), count = 2 * n^2 + 2 * n + 1 ∧ 
  ∀ x y : ℤ, abs x + abs y ≤ n ↔
  count = 2 * n^2 + 2 * n + 1 :=
by
  sorry

end number_of_integer_pairs_l712_712767


namespace identically_zero_on_interval_l712_712553

variable (f : ℝ → ℝ) (a b : ℝ)
variable (h_cont : ContinuousOn f (Set.Icc a b))
variable (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0)

theorem identically_zero_on_interval : ∀ x ∈ Set.Icc a b, f x = 0 := 
by 
  sorry

end identically_zero_on_interval_l712_712553


namespace range_of_m_l712_712089

-- Define the conditions in Lean 4

def double_point (x y : ℝ) : Prop := y = 2 * x

def quadratic_function (x m : ℝ) : ℝ := x^2 + 2 * m * x - m

noncomputable def M := (x1 : ℝ) (hM : double_point x1 (quadratic_function x1 m)) 
def N := (x2 : ℝ) (hN : double_point x2 (quadratic_function x2 m))
def x1_lt_1_lt_x2 (x1 x2 : ℝ) : Prop := x1 < 1 ∧ 1 < x2

-- Lean 4 theorem statement

theorem range_of_m (x1 x2 m : ℝ) 
  (h_double_point_M : double_point x1 (quadratic_function x1 m))
  (h_double_point_N : double_point x2 (quadratic_function x2 m))
  (h_x1_lt_1_lt_x2 : x1_lt_1_lt_x2 x1 x2) 
: m < 1 := 
sorry

end range_of_m_l712_712089


namespace buicks_count_l712_712874

-- Definitions
def total_cars := 301
def ford_eqn (chevys : ℕ) := 3 + 2 * chevys
def buicks_eqn (chevys : ℕ) := 12 + 8 * chevys

-- Statement
theorem buicks_count (chevys : ℕ) (fords : ℕ) (buicks : ℕ) :
  total_cars = chevys + fords + buicks ∧
  fords = ford_eqn chevys ∧
  buicks = buicks_eqn chevys →
  buicks = 220 :=
by
  intros h
  sorry

end buicks_count_l712_712874


namespace log_sin_max_value_l712_712086

theorem log_sin_max_value : ∃ c, c = 0 ∧ ∀ x, 0 < x ∧ x < π → log (sin x) ≤ c :=
by
  use 0
  split
  exact rfl
  intro x hx
  sorry

end log_sin_max_value_l712_712086


namespace total_money_calculation_l712_712688

theorem total_money_calculation (N50 N500 Total_money : ℕ) 
( h₁ : N50 = 37 ) 
( h₂ : N50 + N500 = 54 ) :
Total_money = N50 * 50 + N500 * 500 ↔ Total_money = 10350 := 
by 
  sorry

end total_money_calculation_l712_712688


namespace necessary_but_not_sufficient_condition_l712_712973

variables {α β : Type} [plane α] [plane β]
variable (m : line)
variable (h₁ : m ⊆ α)
variable (h₂ : ∃ (d : plane), d = β ∧ d ≠ α)
variable (h₃ : line_parallel_plane m β)

theorem necessary_but_not_sufficient_condition :
  (plane_parallel_plane α β ↔ line_parallel_plane m β) → false := sorry

end necessary_but_not_sufficient_condition_l712_712973


namespace inequality_solution_l712_712433

open Set

def f (x : ℝ) : ℝ := |x| + x^2 + 2

def solution_set : Set ℝ := { x | x < -2 ∨ x > 4 / 3 }

theorem inequality_solution :
  { x : ℝ | f (2 * x - 1) > f (3 - x) } = solution_set := by
  sorry

end inequality_solution_l712_712433


namespace flower_bed_perimeter_l712_712842

theorem flower_bed_perimeter 
  (a b : ℝ) (h : ℝ)
  (h_triang : a = 3 ∧ b = 4 ∧ h = (Real.sqrt (3^2 + 4^2))) 
  (other_side : ℝ) 
  (h_rect : other_side = 10) : 
  (P : ℝ) (h_perimeter : P = other_side + h + a + b) := 
  P = 22 := 
sorry

end flower_bed_perimeter_l712_712842


namespace _l712_712117

noncomputable def triangle_ABC_def (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : Prop :=
  a = b * sin A * sin B / (a * sin^2 C) ∧ cos A = 5 * real.sqrt 3 / 12

noncomputable theorem cos_B_and_length_BD
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a = b * sin A * sin B / (a * sin^2 C))
  (h2 : cos A = 5 * real.sqrt 3 / 12)
  (area_sqrt23 : real.sqrt 23) 
  (D_midpoint_AC : D = (A + C) / 2) :
  cos B = 3 * real.sqrt 2 / 8 ∧ (segment_length BD = 3)
:=
by {
  sorry
}

end _l712_712117


namespace collinear_TKI_l712_712922

-- Definitions based on conditions
variables {A P Q C M N T K I : Type}
variables (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)

-- Definitions to represent the intersection points 
def T_def (A P Q C : Type) (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) : Prop := 
  ∃ T : Type, line_AP A T ∧ line_CQ C T

def K_def (M P N Q : Type) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop) : Prop := 
  ∃ K : Type, line_MP M K ∧ line_NQ N K

-- Theorem statement
theorem collinear_TKI (A P Q C M N T K I : Type)
  (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)
  (hT : T_def A P Q C line_AP line_CQ) 
  (hK : K_def M P N Q line_MP line_NQ) : 
  collinear T K I :=
sorry

end collinear_TKI_l712_712922


namespace monotonic_increasing_interval_l712_712836

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

def f (a x : ℝ) : ℝ := log_a a (2*x^2 + x)

theorem monotonic_increasing_interval (a : ℝ) (h_a1 : 0 < a) (h_a2 : a ≠ 1) 
    (h_pos_in_interval : ∀ x, 0 < x ∧ x < 1/2 → 0 < f a x) :
    ∃ I : set ℝ, I = set.Iio (-1/2) ∧ ∀ x1 x2 ∈ I, x1 < x2 → f a x1 < f a x2 := 
sorry

end monotonic_increasing_interval_l712_712836


namespace variance_boys_greater_than_girls_l712_712473

def variance (l : List ℝ) : ℝ :=
let mean := (l.foldl (λ acc x => acc + x) 0) / l.length
in (l.foldl (λ acc x => acc + (x - mean) ^ 2) 0) / l.length

theorem variance_boys_greater_than_girls:
  let boys_scores := [86, 94, 88, 92, 90]
  let girls_scores := [88, 93, 93, 88, 93]
  variance boys_scores > variance girls_scores :=
sorry

end variance_boys_greater_than_girls_l712_712473


namespace complex_number_symmetric_division_l712_712398

variable (z1 z2 : ℂ)
variable (h1 : z1 = 1 + 2 * Complex.I)
variable (h2 : z2 = Complex.conj (z1) + Complex.I * Complex.I / 2)

theorem complex_number_symmetric_division :
  (z1 / z2) = (4 / 5) + (3 / 5) * Complex.I :=
by
  sorry

end complex_number_symmetric_division_l712_712398


namespace smallest_value_of_complex_abs_l712_712534

open Complex

noncomputable def smallest_possible_value (z : ℂ) (h : ∥z ^ 2 + 9∥ = ∥z * (z + 3 * Complex.I)∥) : ℝ :=
  infi (λ z, ∥z + Complex.I∥)

theorem smallest_value_of_complex_abs (z : ℂ) (h : ∥z ^ 2 + 9∥ = ∥z * (z + 3 * Complex.I)∥) : 
  smallest_possible_value z h = 2 :=
sorry

end smallest_value_of_complex_abs_l712_712534


namespace number_of_integers_between_sqrts_l712_712443

theorem number_of_integers_between_sqrts :
  let lower_bound := Real.sqrt 10
  let upper_bound := Real.sqrt 75
  let lower_int := Int.ceil lower_bound
  let upper_int := Int.floor upper_bound
  ∃ (count : ℕ), count = upper_int - lower_int + 1 ∧ count = 5 :=
by
  let lower_bound := Real.sqrt 10
  let upper_bound := Real.sqrt 75
  let lower_int := Int.ceil lower_bound
  let upper_int := Int.floor upper_bound
  use upper_int - lower_int + 1
  split
  · sorry
  · sorry

end number_of_integers_between_sqrts_l712_712443


namespace ad_gt_bc_l712_712381

theorem ad_gt_bc 
  (a b c d : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : d > 0)
  (h5 : a + d = b + c)
  (h6 : |a - d| < |b - c|) : ad > bc :=
by
  sorry

end ad_gt_bc_l712_712381


namespace minimum_value_frac_inverse_l712_712409

theorem minimum_value_frac_inverse (a b c : ℝ) (h : a + b + c = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a + b)) + (1 / c) ≥ 4 / 3 :=
by
  sorry

end minimum_value_frac_inverse_l712_712409


namespace min_distance_condition_l712_712095

theorem min_distance_condition (a b : ℝ) (h : 2 * a + b = 1) :
  sqrt ((a - 2) ^ 2 + (b - 2) ^ 2) = sqrt 5 :=
sorry

end min_distance_condition_l712_712095


namespace collinear_TKI_l712_712919

-- Definitions based on conditions
variables {A P Q C M N T K I : Type}
variables (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)

-- Definitions to represent the intersection points 
def T_def (A P Q C : Type) (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) : Prop := 
  ∃ T : Type, line_AP A T ∧ line_CQ C T

def K_def (M P N Q : Type) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop) : Prop := 
  ∃ K : Type, line_MP M K ∧ line_NQ N K

-- Theorem statement
theorem collinear_TKI (A P Q C M N T K I : Type)
  (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)
  (hT : T_def A P Q C line_AP line_CQ) 
  (hK : K_def M P N Q line_MP line_NQ) : 
  collinear T K I :=
sorry

end collinear_TKI_l712_712919


namespace gear_ratio_l712_712484

variable (a b c : ℕ) (ωG ωH ωI : ℚ)

theorem gear_ratio :
  (a * ωG = b * ωH) ∧ (b * ωH = c * ωI) ∧ (a * ωG = c * ωI) →
  ωG / ωH = bc / ac ∧ ωH / ωI = ac / ab ∧ ωG / ωI = bc / ab :=
by
  sorry

end gear_ratio_l712_712484


namespace find_triplets_satisfying_equation_l712_712750

theorem find_triplets_satisfying_equation :
  ∃ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧ (x, y, z) = (2, 251, 252) :=
by
  sorry

end find_triplets_satisfying_equation_l712_712750


namespace find_f_neg2_l712_712416

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then real.log2(x + 2) - 1 else -real.log2(-x + 2) + 1

theorem find_f_neg2 : f(-2) = -1 :=
by 
  sorry

end find_f_neg2_l712_712416


namespace smallest_vertical_distance_l712_712207

def f (x : ℝ) : ℝ := abs x
def g (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem smallest_vertical_distance : 
  ∃ x : ℝ, (∀ y : ℝ, abs (f y - g y) ≥ abs (f x - g x)) ∧ abs (f x - g x) = 3 / 4 :=
begin
  sorry,
end

end smallest_vertical_distance_l712_712207


namespace sum_of_sides_triangle_sum_of_squares_of_sines_triangle_l712_712743

-- Define any necessary variables and concepts
variables {R r A B C a b c : ℝ}

-- Conditions for problem 1
axiom sum_of_sides_condition (h : a + b + c = 4 * R + 2 * r ∧ 
                              a + b + c > 4 * R + 2 * r ∧
                              a + b + c < 4 * R + 2 * r)

-- Proof statement for problem 1
theorem sum_of_sides_triangle (h1 : R > 0) (h2 : r > 0) :
  ∃ A : ℝ, A < 90 → a + b + c > 4 * R + 2 * r ∧
  A = 90 → a + b + c = 4 * R + 2 * r ∧
  A > 90 → a + b + c < 4 * R + 2 * r :=
sorry

-- Conditions for problem 2
axiom sum_of_squares_of_sines_condition (h : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 2 ∧ 
                                        sin A ^ 2 + sin B ^ 2 + sin C ^ 2 > 2 ∧
                                        sin A ^ 2 + sin B ^ 2 + sin C ^ 2 < 2)

-- Proof statement for problem 2
theorem sum_of_squares_of_sines_triangle (hA : A > 0) (hB : B > 0) (hC : C > 0) :
  ∃ A : ℝ, A < 90 → sin A ^ 2 + sin B ^ 2 + sin C ^ 2 > 2 ∧
  A = 90 → sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 2 ∧
  A > 90 → sin A ^ 2 + sin B ^ 2 + sin C ^ 2 < 2 :=
sorry

end sum_of_sides_triangle_sum_of_squares_of_sines_triangle_l712_712743


namespace negation_equivalence_l712_712599

open Classical

variable (R : Type) [LinearOrderedField R]
variable (N : Type) [LinearOrderedSemiring N] [Nontrivial N]

noncomputable def negation_proposition (x : R) (n : N) :=
  ¬ (∀ x : R, ∃ n : N, n ≥ x^2) ↔ (∃ x : R, ∀ n : N, n < x^2)

theorem negation_equivalence : ∀ (x : R) (n : N), negation_proposition R N x n := by
  intros
  sorry

end negation_equivalence_l712_712599


namespace number_of_odd_numbers_l712_712099

theorem number_of_odd_numbers (a : ℤ) (h : 8 < a ∧ a < 27) : 
  (finset.card (finset.filter (λ x, x % 2 = 1) (finset.Ico 9 27))) = 9 :=
by
  sorry

end number_of_odd_numbers_l712_712099


namespace abc_sum_condition_l712_712222

theorem abc_sum_condition {x y a b c d : ℝ} (h₁: x + y = 6) (h₂: 3 * x * y = 6) 
  (hx: x = (a + b * real.sqrt c) / d ∨ x = (a - b * real.sqrt c) / d) 
  (ha: a = 3) (hb: b = 1) (hc: c = 7) (hd: d = 1) : 
  a + b + c + d = 12 := 
by 
  -- Proof is omitted, as per the instructions.
  sorry

end abc_sum_condition_l712_712222


namespace min_value_expression_l712_712352

theorem min_value_expression : 
  ∃ s t : ℝ, (|cos t|^2 + |sin t|^2 = 1) ∧ ((s + 5 - 3 * |cos t|)^2 + (s - 2 * |sin t|)^2 = 2) :=
by
  sorry

end min_value_expression_l712_712352


namespace mean_points_scored_l712_712311

def Mrs_Williams_points : ℝ := 50
def Mr_Adams_points : ℝ := 57
def Mrs_Browns_points : ℝ := 49
def Mrs_Daniels_points : ℝ := 57

def total_points : ℝ := Mrs_Williams_points + Mr_Adams_points + Mrs_Browns_points + Mrs_Daniels_points
def number_of_classes : ℝ := 4

theorem mean_points_scored :
  (total_points / number_of_classes) = 53.25 :=
by
  sorry

end mean_points_scored_l712_712311


namespace find_amplitude_l712_712314

-- Conditions
variables (a b c d : ℝ)

theorem find_amplitude
  (h1 : ∀ x, a * Real.sin (b * x + c) + d ≤ 5)
  (h2 : ∀ x, a * Real.sin (b * x + c) + d ≥ -3) :
  a = 4 :=
by 
  sorry

end find_amplitude_l712_712314


namespace tangent_identity_problem_l712_712378

theorem tangent_identity_problem 
    (α β : ℝ) 
    (h1 : Real.tan (α + β) = 1) 
    (h2 : Real.tan (α - π / 3) = 1 / 3) 
    : Real.tan (β + π / 3) = 1 / 2 := 
sorry

end tangent_identity_problem_l712_712378


namespace average_increase_mpg_correct_l712_712613

-- Definitions of current and required distances, fuel consumptions and their respective efficiencies.
def distanceA : ℝ := 180
def distanceB : ℝ := 225
def distanceC : ℝ := 270

def fuelA : ℝ := 12
def fuelB : ℝ := 15
def fuelC : ℝ := 18

def requiredFuel : ℝ := 10

def current_mpgA : ℝ := distanceA / fuelA
def current_mpgB : ℝ := distanceB / fuelB
def current_mpgC : ℝ := distanceC / fuelC

def required_mpgA : ℝ := distanceA / requiredFuel
def required_mpgB : ℝ := distanceB / requiredFuel
def required_mpgC : ℝ := distanceC / requiredFuel

def increase_mpgA : ℝ := required_mpgA - current_mpgA
def increase_mpgB : ℝ := required_mpgB - current_mpgB
def increase_mpgC : ℝ := required_mpgC - current_mpgC

def average_increase_mpg : ℝ := (increase_mpgA + increase_mpgB + increase_mpgC) / 3

-- The theorem stating the proof problem.
theorem average_increase_mpg_correct :
  average_increase_mpg = 7.5 :=
by
  sorry

end average_increase_mpg_correct_l712_712613


namespace range_of_a_l712_712073

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def B (x : ℝ) (a : ℝ) : Prop := x > a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, A x → B x a) → a < -2 :=
by
  sorry

end range_of_a_l712_712073


namespace largest_four_digit_number_l712_712245

theorem largest_four_digit_number
  (n : ℕ) (hn1 : n % 8 = 2) (hn2 : n % 7 = 4) (hn3 : 1000 ≤ n) (hn4 : n ≤ 9999) :
  n = 9990 :=
sorry

end largest_four_digit_number_l712_712245


namespace collinear_TKI_l712_712918

-- Definitions based on conditions
variables {A P Q C M N T K I : Type}
variables (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)

-- Definitions to represent the intersection points 
def T_def (A P Q C : Type) (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) : Prop := 
  ∃ T : Type, line_AP A T ∧ line_CQ C T

def K_def (M P N Q : Type) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop) : Prop := 
  ∃ K : Type, line_MP M K ∧ line_NQ N K

-- Theorem statement
theorem collinear_TKI (A P Q C M N T K I : Type)
  (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)
  (hT : T_def A P Q C line_AP line_CQ) 
  (hK : K_def M P N Q line_MP line_NQ) : 
  collinear T K I :=
sorry

end collinear_TKI_l712_712918


namespace cube_vertex_numbers_impossible_l712_712122

theorem cube_vertex_numbers_impossible (f : Fin 8 → ℕ) (h : ∀ i : Fin 8, f i ∈ {1, 2, 3, 4, 5, 6, 7, 8}):
  ¬ (∃ (g : Fin 8 → Fin 8), ∀ i : Fin 8, ∃ n1 n2 n3 : Fin 8, n1 ≠ i ∧ n2 ≠ i ∧ n3 ≠ i ∧ g n1 + g n2 + g n3 = g i ∧ g i ∣ (f n1 + f n2 + f n3)) :=
sorry

end cube_vertex_numbers_impossible_l712_712122


namespace problem_statement_l712_712062

def f (x : ℝ) : ℝ :=
  if x < 0 then logBase 3 (-x) else 3^(x - 2)

theorem problem_statement (a : ℝ) (h : f a = 3) :
  f 2 = 1 ∧ (a = 3 ∨ a = -27) :=
by
  sorry

end problem_statement_l712_712062


namespace collinear_T_K_I_l712_712907

noncomputable def T (A P C Q : Point) : Point := intersection (line_through A P) (line_through C Q)
noncomputable def K (M P N Q : Point) : Point := intersection (line_through M P) (line_through N Q)

theorem collinear_T_K_I (A P C Q M N I : Point) :
  collinear [T A P C Q, K M P N Q, I] :=
sorry

end collinear_T_K_I_l712_712907


namespace greatest_non_sum_complex_l712_712543

def is_complex (n : ℕ) : Prop :=
  ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n

theorem greatest_non_sum_complex : ∀ n : ℕ, (¬ ∃ a b : ℕ, is_complex a ∧ is_complex b ∧ a + b = n) → n ≤ 23 :=
by {
  sorry
}

end greatest_non_sum_complex_l712_712543


namespace acute_triangle_side_range_l712_712604

   theorem acute_triangle_side_range (x : ℝ) (h₁ : 3 < x) (h₂ : x < sqrt 34) :
     3 < x ∧ x < 5 :=
   sorry
   
end acute_triangle_side_range_l712_712604


namespace circumscribed_circle_radius_l712_712417

variables (A B C : ℝ) (a b c : ℝ) (R : ℝ) (area : ℝ)

-- Given conditions
def sides_ratio := a / b = 7 / 5 ∧ b / c = 5 / 3
def triangle_area := area = 45 * Real.sqrt 3
def sides := (a, b, c)
def angles := (A, B, C)

-- Prove radius
theorem circumscribed_circle_radius 
  (h_ratio : sides_ratio a b c)
  (h_area : triangle_area area) :
  R = 14 :=
sorry

end circumscribed_circle_radius_l712_712417


namespace smallest_positive_angle_terminal_side_eq_l712_712606

theorem smallest_positive_angle_terminal_side_eq (n : ℤ) :
  (0 ≤ n % 360 ∧ n % 360 < 360) → (∃ k : ℤ, n = -2015 + k * 360 ) → n % 360 = 145 :=
by
  sorry

end smallest_positive_angle_terminal_side_eq_l712_712606


namespace trapezoid_area_l712_712496

variables (a b alpha beta : ℝ)
variables (ABCD : Type) [trapezoid ABCD]
variables (BC AD : ABCD → ℝ) [parallel BC AD]
variables (angleCAD angleBAC : ℝ)
variable [is_angle angleCAD alpha CAD]
variable [is_angle angleBAC beta BAC]

theorem trapezoid_area :
  area ABCD = (a * (a + b) * sin alpha * sin (alpha + beta)) / (2 * sin beta) :=
sorry

end trapezoid_area_l712_712496


namespace monotonicity_and_m_range_l712_712387

variable {f : ℝ → ℝ}

-- Conditions
axiom additivity : ∀ x y : ℝ, f(x) + f(y) = f(x + y)
axiom positivity : ∀ x : ℝ, x > 0 → f(x) > 0
axiom fixed_point : f(1) = 1

-- Prove monotonicity and the range of m
theorem monotonicity_and_m_range 
  (monotonic_f : ∀ x y : ℝ, x < y → f(x) < f(y))
  (m_range : ∀ m : ℝ, (∀ x ∈ set.Icc (-1 : ℝ) 1, ∀ a ∈ set.Icc (-2 : ℝ) 2, f(x) < m^2 - 2*a*m + 1) → (m > 4 ∨ m < -4)) :
  true :=
by sorry

end monotonicity_and_m_range_l712_712387


namespace ratio_surface_area_l712_712422

def cylinder_surface_area (r : ℝ) : ℝ := 2 * π * r^2 + 2 * r * π * 2 * r
def sphere_surface_area (r : ℝ) : ℝ := 4 * π * r^2

theorem ratio_surface_area (r : ℝ) (h : cylinder_surface_area r / sphere_surface_area r = 3 / 2) :
    cylinder_surface_area r / sphere_surface_area r = 3 / 2 := by
    sorry

end ratio_surface_area_l712_712422


namespace lisa_square_cookies_l712_712356

theorem lisa_square_cookies (h_carlos_cookies : ∀ (base height : ℝ) (num : ℕ), 
  base = 4 ∧ height = 5 ∧ num = 20 → 
  (1 / 2) * base * height * num = 200)
  (side_lisa_cookies : ℝ) :
  side_lisa_cookies = 5 → 
  200 / (side_lisa_cookies ^ 2) = 8 :=
by 
  intros h l
  have h_area_carlos := h 4 5 20 ⟨rfl, rfl, rfl⟩
  simp at h_area_carlos
  rw [h_area_carlos, h_l]
  norm_num
  sorry

end lisa_square_cookies_l712_712356


namespace solve_quadratic_l712_712970

theorem solve_quadratic :
  ∀ x : ℂ, 2 * (5 * x^2 + 4 * x + 3) - 6 = -3 * (2 - 4 * x) ↔ x = (1 + complex.I * real.sqrt 14) / 5 ∨ x = (1 - complex.I * real.sqrt 14) / 5 := 
by 
  sorry

end solve_quadratic_l712_712970


namespace collinear_T_K_I_l712_712883

noncomputable def intersection (P Q : Set Point) : Point := sorry

variables (A P C Q M N I T K : Point)

-- Definitions based on conditions
def T_def : Point := intersection (line_through A P) (line_through C Q)
def K_def : Point := intersection (line_through M P) (line_through N Q)

-- Proof statement
theorem collinear_T_K_I :
  collinear ({T_def A P C Q, K_def M P N Q, I} : Set Point) := sorry

end collinear_T_K_I_l712_712883


namespace gallons_of_gas_l712_712512

-- Define the conditions
def mpg : ℕ := 19
def d1 : ℕ := 15
def d2 : ℕ := 6
def d3 : ℕ := 2
def d4 : ℕ := 4
def d5 : ℕ := 11

-- The theorem to prove
theorem gallons_of_gas : (d1 + d2 + d3 + d4 + d5) / mpg = 2 := 
by {
    sorry
}

end gallons_of_gas_l712_712512


namespace maximum_value_inequality_l712_712334

theorem maximum_value_inequality (x y : ℝ) : 
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 50 :=
sorry

end maximum_value_inequality_l712_712334


namespace second_last_digit_modulo_5_is_2_l712_712324

def modified_fibonacci : ℕ → ℕ
| 1       := 2
| 2       := 2
| (n + 3) := modified_fibonacci (n + 2) + modified_fibonacci (n + 1)

theorem second_last_digit_modulo_5_is_2 :
  ∃ (n : ℕ), 
    (∀ m < n, (modified_fibonacci m) % 5 ≠ 2) ∧ 
    ∀ m ≥ n, ∃ k < m, (modified_fibonacci k) % 5 = 2 :=
sorry

end second_last_digit_modulo_5_is_2_l712_712324


namespace smallest_sum_of_squares_l712_712987

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l712_712987


namespace minimum_box_value_l712_712822

theorem minimum_box_value :
  ∃ (a b : ℤ), a ≠ b ∧ b ≠ 61 ∧ a ≠ 61 ∧ (a * b = 30) ∧ (∃ M : ℤ, (ax + b)*(bx + a) = 30x^2 + M * x + 30) ∧ M = min (a^2 + b^2) 61 :=
by
  sorry

end minimum_box_value_l712_712822


namespace tank_salt_solution_l712_712651

theorem tank_salt_solution (x : ℝ) (hx1 : 0.20 * x / (3 / 4 * x + 30) = 1 / 3) : x = 200 :=
by sorry

end tank_salt_solution_l712_712651


namespace final_segment_position_correct_l712_712686

def initial_segment : ℝ × ℝ := (1, 6)
def rotate_180_about (p : ℝ) (x : ℝ) : ℝ := p - (x - p)
def first_rotation_segment : ℝ × ℝ := (rotate_180_about 2 6, rotate_180_about 2 1)
def second_rotation_segment : ℝ × ℝ := (rotate_180_about 1 3, rotate_180_about 1 (-2))

theorem final_segment_position_correct :
  second_rotation_segment = (-1, 4) :=
by
  -- This is a placeholder for the actual proof.
  sorry

end final_segment_position_correct_l712_712686


namespace fifth_term_binomial_expansion_l712_712702

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem fifth_term_binomial_expansion (b x : ℝ) :
  let term := (binomial 7 4) * ((b / x)^(7 - 4)) * ((-x^2 * b)^4)
  term = -35 * b^7 * x^5 := 
by
  sorry

end fifth_term_binomial_expansion_l712_712702


namespace range_k_l712_712454

theorem range_k (k : ℝ) :
  (∀ x : ℝ, (3/8 - k*x - 2*k*x^2) ≥ 0) ↔ (-3 ≤ k ∧ k ≤ 0) :=
sorry

end range_k_l712_712454


namespace area_of_triangle_arithmetic_geometric_sequence_l712_712427

theorem area_of_triangle_arithmetic_geometric_sequence {A B C a b c : ℝ} :
  (A + C) / 2 = B ∧ A + B + C = 180 ∧ a + c + 4/√3 * b = 1 ∧ c = a * r ∧ 4/√3 * b = (a * r) * r → 
  S_ΔABC = (√3 / 2) * a^2 :=
by
  sorry

end area_of_triangle_arithmetic_geometric_sequence_l712_712427


namespace sum_of_scores_with_three_ways_correct_l712_712237

def is_valid_combination (c u: ℕ) (S: ℚ) : Prop :=
  (c + u <= 25) ∧
  (6 * c + 2.5 * u = S)

def count_ways_to_achieve_score (S: ℚ) : ℕ :=
  (List.range 26).countp (λ c => ∃ u, is_valid_combination c u S)

def scores_with_three_ways : List ℚ :=
  (List.range 151).map (λ n => (n : ℚ)).filter (λ S => count_ways_to_achieve_score S = 3)

def sum_of_scores_with_three_ways : ℚ :=
  scores_with_three_ways.sum

theorem sum_of_scores_with_three_ways_correct :
  sum_of_scores_with_three_ways = 182 := -- The actual sum to be specified based on the calculation
  sorry

end sum_of_scores_with_three_ways_correct_l712_712237


namespace moving_circle_trajectory_l712_712423

/-- Given the conditions about tangents and distances from the scenario, show that
the trajectory of the moving circle's center is the specified parabola. --/
theorem moving_circle_trajectory (M_center : ℝ × ℝ) :
  (tangent_to_line : ∃ k : ℝ, M_center = (k, 2)) ∧
  (externally_tangent_to_circle : ∃ r : ℝ, ∃ h : ℝ, M_center = (h, r) ∧ h^2 + (r + 3)^2 = 1)
  → M_center.snd^2 = -12 * M_center.fst :=
by sorry

end moving_circle_trajectory_l712_712423


namespace prove_river_improvement_l712_712663

def river_improvement_equation (x : ℝ) : Prop :=
  4800 / x - 4800 / (x + 200) = 4

theorem prove_river_improvement (x : ℝ) (h : x > 0) : river_improvement_equation x := by
  sorry

end prove_river_improvement_l712_712663


namespace cube_net_shading_count_l712_712984

theorem cube_net_shading_count :
  let P Q R S T U V W : Prop in
  -- The net of a cube cannot simultaneously include both P and Q
  (¬(P ∧ Q)) ∧
  -- Each net must include square T
  (T) ∧
  -- The net of a cube cannot include all four squares T, U, V, and W
  (¬(T ∧ U ∧ V ∧ W)) →
  -- The number of ways to shade six out of the eight squares to form the net of a cube is 6
  ∃ shaded_squares : finset (Prop), 
  shaded_squares.card = 6 ∧
  ∃ exclusion_pairs : finset (Prop × Prop),
  exclusion_pairs.card = 6 ∧
  ∀ (pair : Prop × Prop), pair ∈ exclusion_pairs → (¬(pair.1 ∧ pair.2))
:=
sorry

end cube_net_shading_count_l712_712984


namespace wife_weekly_savings_l712_712284

noncomputable def amount_savings (weeks : ℕ) (amount_per_week : ℕ) := weeks * amount_per_week

def total_savings_couple (husband_weekly : ℕ) (wife_weekly : ℕ) (weeks : ℕ) :=
  amount_savings weeks husband_weekly + amount_savings weeks wife_weekly

theorem wife_weekly_savings (husband_weekly : ℕ) (weeks : ℕ) (total_children_savings : ℕ)
  (children_count : ℕ) (children_share : ℕ) :
  total_children_savings / children_count = children_share →
  total_children_savings * 2 = weekly_savings →
  total_savings_couple husband_weekly wife_weekly weeks = total_savings_couple husband_weekly weekly_savings weeks →
  wife_weekly = 225 :=
by
  let husband_weekly := 335
  let weeks := 24
  let total_children_savings := 6720
  let children_count := 4
  let children_share := 1680

  sorry

end wife_weekly_savings_l712_712284


namespace solution_for_a_l712_712289

theorem solution_for_a :
  ∃ a : ℝ, -6 * a ^ 2 = 3 * (4 * a + 2) ∧ a = -1 :=
begin
  use -1,
  split,
  {
    -- This part confirms the condition
    calc -6 * (-1) ^ 2
        = -6 * 1 : by norm_num
    ... = -6 : by ring
    ... = 3 * (4 * (-1) + 2) : by norm_num
    ... = 3 * (-4 + 2) : by ring
    ... = 3 * -2 : by ring
    ... = -6 : by ring,
  },
  {
    -- This confirms our solution
    refl,
  }
end

end solution_for_a_l712_712289


namespace probability_three_defective_phones_l712_712700

theorem probability_three_defective_phones :
  let total_smartphones := 380
  let defective_smartphones := 125
  let P_def_1 := (defective_smartphones : ℝ) / total_smartphones
  let P_def_2 := (defective_smartphones - 1 : ℝ) / (total_smartphones - 1)
  let P_def_3 := (defective_smartphones - 2 : ℝ) / (total_smartphones - 2)
  let P_all_three_def := P_def_1 * P_def_2 * P_def_3
  abs (P_all_three_def - 0.0351) < 0.001 := 
by
  sorry

end probability_three_defective_phones_l712_712700


namespace complex_division_l712_712726

theorem complex_division (a b : ℂ) (ha : a = (3 - 1 * complex.i)) (hb : b = (1 + 1 * complex.i)) :
  a / b = 1 - 2 * complex.i :=
by sorry

end complex_division_l712_712726


namespace vanessa_points_record_l712_712627

theorem vanessa_points_record 
  (P : ℕ) 
  (H₁ : P = 48) 
  (O : ℕ) 
  (H₂ : O = 6 * 3.5) : V = (P - O) → V = 27 :=
by
  sorry

end vanessa_points_record_l712_712627


namespace e_to_2i_in_second_quadrant_l712_712142

theorem e_to_2i_in_second_quadrant :
  ∃ z : ℂ, z = complex.exp (2 * complex.I) ∧ 
  (z.re < 0 ∧ z.im > 0) :=
begin
  -- Proof would go here
  sorry
end

end e_to_2i_in_second_quadrant_l712_712142


namespace y_mul_k_is_perfect_square_l712_712723

-- Defining y as given in the problem with its prime factorization
def y : Nat := 3^4 * (2^2)^5 * 5^6 * (2 * 3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Since the question asks for an integer k (in this case 75) such that y * k is a perfect square
def k : Nat := 75

-- The statement that needs to be proved
theorem y_mul_k_is_perfect_square : ∃ n : Nat, (y * k) = n^2 := 
by
  sorry

end y_mul_k_is_perfect_square_l712_712723


namespace michael_amc10_score_l712_712977

theorem michael_amc10_score (x : ℕ) (h : 0 ≤ x ∧ x ≤ 20) :
  (7 * x - (20 - x) + 2 * 5) >= 120 → x >= 18 :=
begin
  sorry
end

end michael_amc10_score_l712_712977


namespace induction_base_case_not_necessarily_one_l712_712643

theorem induction_base_case_not_necessarily_one :
  (∀ (P : ℕ → Prop) (n₀ : ℕ), (P n₀) → (∀ n, n ≥ n₀ → P n → P (n + 1)) → ∀ n, n ≥ n₀ → P n) ↔
  (∃ n₀ : ℕ, n₀ ≠ 1) :=
sorry

end induction_base_case_not_necessarily_one_l712_712643


namespace WH_length_l712_712550

open real

noncomputable def length_WH (s area_triangle : ℝ) := 
  let h := sqrt (2 * area_triangle / s) in
  (sqrt (s^2 + h^2))

theorem WH_length :
  ∀ (s area_triangle : ℝ), 
    s^2 = 324 → area_triangle = 234 → length_WH s area_triangle = 12 * sqrt 11 :=
by
  intros s area_triangle hs h_area
  sorry

end WH_length_l712_712550


namespace find_a_minus_b_l712_712448

theorem find_a_minus_b (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023) 
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := 
sorry

end find_a_minus_b_l712_712448


namespace hyperbola_center_l712_712350

theorem hyperbola_center (x y : ℝ) :
  (∃ h k, h = 2 ∧ k = -1 ∧ 
    (∀ x y, (3 * y + 3)^2 / 7^2 - (4 * x - 8)^2 / 6^2 = 1 ↔ 
      (y - (-1))^2 / ((7 / 3)^2) - (x - 2)^2 / ((3 / 2)^2) = 1)) :=
by sorry

end hyperbola_center_l712_712350


namespace new_person_weight_l712_712658

theorem new_person_weight (avg_increase : ℝ) (replaced_weight : ℝ) (new_weight : ℝ) : 
  avg_increase = 5.5 → replaced_weight = 68 → 
  new_weight = replaced_weight + avg_increase * 5 → new_weight = 95.5 :=
begin
  intros h_avg h_replaced h_new,
  rw [h_avg, h_replaced, h_new],
  norm_num,
  sorry
end

end new_person_weight_l712_712658


namespace expand_product_l712_712746

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9 * x + 18 := 
by sorry

end expand_product_l712_712746


namespace students_sampled_l712_712477

def students_in_buildings (n : ℕ) : Prop :=
  n = 400 ∧
  (∀ s : ℕ, 1 ≤ s ∧ s ≤ 400 → 
     (if s ≤ 200 then "A" else if s ≤ 295 then "B" else "C" = 
        (if s ≤ 200 then "A" else if s ≤ 295 then "B" else "C")))

def systematic_sampling (first k : ℕ) (sample : List ℕ) : Prop :=
  sample.length = 50 ∧
  (∀ i : ℕ, i < 50 → sample.nth i = some (first + i * k)) ∧
  all_unique sample

theorem students_sampled :
  ∀ (pop_size sample_size first : ℕ)
    (build_A build_B build_C : List ℕ),
  pop_size = 400 →
  sample_size = 50 →
  first = 3 →
  build_A = [1..200] →
  build_B = [201..295] →
  build_C = [296..400] →
  ∃ (k : ℕ) (sample : List ℕ),
  k = pop_size / sample_size ∧
  systematic_sampling first k sample ∧
  (sample.filter (λ x, x ≤ 200)).length = 25 ∧
  (sample.filter (λ x, 201 ≤ x ∧ x ≤ 295)).length = 12 ∧
  (sample.filter (λ x, 296 ≤ x ∧ x ≤ 400)).length = 13 :=
by
  sorry

end students_sampled_l712_712477


namespace problem_correct_propositions_l712_712645

open Real

theorem problem_correct_propositions (a : ℝ) (f : ℝ → ℝ) :
  (∀ a < 0, ¬ (a * sqrt (-1 / a) = sqrt (-a))) ∧
  (∀ a, (a + 1 / a = 3) → (sqrt a + sqrt (1 / a) = sqrt 5)) ∧
  (∀ a > 0, a ≠ 1 → f 1 = a ^ (1 - 1) + 2 → f 1 = 3) ∧
  (∀ a, (f x = 2 ^ (-x ^ 2 + a * x)) ∧ (∀ x < 1, monotone_increasing (λ x, - x ^ 2 + a * x)) → a ≥ 2) :=
sorry

end problem_correct_propositions_l712_712645


namespace part1_solution_set_part2_range_a_l712_712066

-- Part 1: Proof statement for the inequality f(x) > 4 when a = 2
theorem part1_solution_set (x : ℝ) : (| 3 * x + 3 | + | x - 2 | > 4) ↔ (x > -1 / 2 ∨ x < -5 / 4) :=
sorry

-- Part 2: Proof statement for the range of values for 'a' such that f(x) > 3x + 4 for all x ∈ (-1, +∞)
theorem part2_range_a (a : ℝ) : (∀ x ∈ set.Ioi (-1 : ℝ), | 3 * x + 3 | + | x - a | > 3 * x + 4) ↔ (a ≤ -2) :=
sorry

end part1_solution_set_part2_range_a_l712_712066


namespace shaded_area_of_square_l712_712670

theorem shaded_area_of_square (side_length : ℝ) (midA midB midC : ℝ × ℝ) (h_side : side_length = 12)
  (hA : midA = (6, 12)) (hB : midB = (6, 0)) (hC : midC = (12, 6)) : 
  let half_base := side_length / 2 in 
  let height := side_length in 
  area (triangle midA midB midC) = 18 := 
by
  sorry

end shaded_area_of_square_l712_712670


namespace c_share_of_rent_l712_712257

/-- 
Given the conditions:
- a puts 10 oxen for 7 months,
- b puts 12 oxen for 5 months,
- c puts 15 oxen for 3 months,
- The rent of the pasture is Rs. 210,
Prove that C should pay Rs. 54 as his share of rent.
-/
noncomputable def total_rent : ℝ := 210
noncomputable def oxen_months_a : ℝ := 10 * 7
noncomputable def oxen_months_b : ℝ := 12 * 5
noncomputable def oxen_months_c : ℝ := 15 * 3
noncomputable def total_oxen_months : ℝ := oxen_months_a + oxen_months_b + oxen_months_c

theorem c_share_of_rent : (total_rent / total_oxen_months) * oxen_months_c = 54 :=
by
  sorry

end c_share_of_rent_l712_712257


namespace range_m_max_area_l712_712262

noncomputable theory
open Real

def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 = 1

def right_branch (x0 y0 : ℝ) : Prop :=
  hyperbola_eq x0 y0 ∧ y0 ≥ 1

def foci (a b: ℝ) : set ℝ :=
  {(- sqrt (a^2 + b^2), 0), (sqrt (a^2 + b^2), 0)}

def angle_bisector_point (m x0 y0: ℝ) : Prop :=
  -- This includes the actual relation derived from angle bisector theorem
  ∃ a b : ℝ, 
  (m + sqrt 5) * b = (sqrt 5 - m) * a ∧ 
  a = (sqrt 5 * x0) / 2 + 2 × y0 ∧ 
  b = (sqrt 5 * x0) / 2 - 2 × y0

def max_triangle_area (x0 y0 : ℝ) : ℝ :=
  4 * sqrt 5 * (sqrt ((5 * y0^2 + 1) / (5 * y0^2 - 4)))

theorem range_m (x0 y0 m : ℝ) (h : right_branch x0 y0) :
  0 < m ∧ m ≤ sqrt 2 :=
sorry

theorem max_area (x0 y0 : ℝ) (h : right_branch x0 y0) :
  max_triangle_area x0 y0 = 4 * sqrt 30 :=
sorry

end range_m_max_area_l712_712262


namespace time_to_fill_tank_with_two_pipes_simultaneously_l712_712660

def PipeA : ℝ := 30
def PipeB : ℝ := 45

theorem time_to_fill_tank_with_two_pipes_simultaneously :
  let A := 1 / PipeA
  let B := 1 / PipeB
  let combined_rate := A + B
  let time_to_fill_tank := 1 / combined_rate
  time_to_fill_tank = 18 := 
by
  sorry

end time_to_fill_tank_with_two_pipes_simultaneously_l712_712660


namespace complex_numbers_real_fifth_powers_l712_712255

theorem complex_numbers_real_fifth_powers :
  ∃ (z : Finset ℂ) (hz : z.card = 30), (z.filter (λ x, (x ^ 5).im = 0)).card = 10 :=
by
  sorry

end complex_numbers_real_fifth_powers_l712_712255


namespace books_in_children_section_l712_712868

theorem books_in_children_section (books_start : ℕ) (books_left : ℕ) (books_history : ℕ) (books_fiction : ℕ) (books_misplaced : ℕ) :
  books_start = 51 →
  books_left = 16 →
  books_history = 12 →
  books_fiction = 19 →
  books_misplaced = 4 →
  (books_start + books_misplaced - (books_history + books_fiction) - books_left) = 8 :=
by
  intros h_start h_left h_history h_fiction h_misplaced
  rw [h_start, h_left, h_history, h_fiction, h_misplaced]
  norm_num
  sorry

end books_in_children_section_l712_712868


namespace freshmen_more_than_sophomores_l712_712109

theorem freshmen_more_than_sophomores :
  ∀ (total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores : ℕ),
    total_students = 1200 →
    juniors = 264 →
    not_sophomores = 660 →
    not_freshmen = 300 →
    seniors = 240 →
    adv_grade = 20 →
    freshmen = total_students - not_freshmen - seniors - adv_grade →
    sophomores = total_students - not_sophomores - seniors - adv_grade →
    freshmen - sophomores = 360 :=
by
  intros total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores
  intros h_total h_juniors h_not_sophomores h_not_freshmen h_seniors h_adv_grade h_freshmen h_sophomores
  sorry

end freshmen_more_than_sophomores_l712_712109


namespace calculate_answer_l712_712446

theorem calculate_answer (x : ℕ) (h1 : 7 * x = 70) : 36 - x = 26 :=
by
  -- Calculate x using the condition
  have hx : x = 10 := 
    calc
      x = 70 / 7 : by sorry
      _ = 10      : by sorry
  -- Use the value of x to find the correct answer
  have h_correct : 36 - x = 36 - 10 := congrArg (λ y : ℕ, 36 - y) hx
  -- Verify the final answer is 26
  show 36 - x = 26, by
    rw [hx]
    exact h_correct 
    sorry

end calculate_answer_l712_712446


namespace collinear_TKI_l712_712894

-- Definitions for points and lines
variables {A P C Q M N I T K : Type}
variable {line : Type → Type}
variables (AP : line A → line P) (CQ : line C → line Q) (MP : line M → line P) (NQ : line N → line Q)

-- Conditions from the problem
-- Assume there exist points T and K which are intersections of the specified lines
axiom intersects_AP_CQ : ∃ (T : Type), AP T = CQ T
axiom intersects_MP_NQ : ∃ (K : Type), MP K = NQ K

-- Collinearity of points T, K, and I
theorem collinear_TKI : ∀ (I : Type) (T : Type) (K : Type),
  intersects_AP_CQ → intersects_MP_NQ → collinear I T K :=
by sorry

end collinear_TKI_l712_712894


namespace sandy_took_200_l712_712967

variable (X : ℝ)

/-- Given that Sandy had $140 left after spending 30% of the money she took for shopping,
we want to prove that Sandy took $200 for shopping. -/
theorem sandy_took_200 (h : 0.70 * X = 140) : X = 200 :=
by
  sorry

end sandy_took_200_l712_712967


namespace number_of_white_marbles_l712_712671

theorem number_of_white_marbles
  (total_marbles : ℕ)
  (blue_marbles : ℕ)
  (red_marbles : ℕ)
  (prob_red_or_white : ℚ)
  (W : ℕ)
  (h_total : total_marbles = 20)
  (h_blue : blue_marbles = 5)
  (h_red : red_marbles = 7)
  (h_prob : prob_red_or_white = 3/4) :
  W = 8 :=
by
  have h_combined : (red_marbles + W) / total_marbles = prob_red_or_white :=
    sorry
  have h_eq : red_marbles + W = total_marbles * prob_red_or_white :=
    the conversion from h_combined
    sorry
  sorry -- Complete the remaining proof steps to conclude W = 8

end number_of_white_marbles_l712_712671


namespace positive_integer_expression_l712_712657

theorem positive_integer_expression
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_rec : ∀ n, a (n+1) = (∏ i in finset.range n.succ, (a i)^2) - 3)
  (h_base : ∃ k : ℕ, k = 1 * (a 0 + real.sqrt (a 1 - 1)) / 2) :
  ∀ n, ∃ k : ℕ, k = 1 * (∏ i in finset.range n.succ, a i + real.sqrt (a (n+1) - 1)) / 2 :=
begin
  sorry
end

end positive_integer_expression_l712_712657


namespace largest_common_term_lt_300_l712_712198

theorem largest_common_term_lt_300 :
  ∃ a : ℕ, a < 300 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 3 + 7 * m) ∧ ∀ b : ℕ, b < 300 → (∃ n : ℤ, b = 4 + 5 * n) → (∃ m : ℤ, b = 3 + 7 * m) → b ≤ a :=
sorry

end largest_common_term_lt_300_l712_712198


namespace area_of_ABCD_is_196_l712_712226

-- Define the shorter side length of the smaller rectangles
def shorter_side : ℕ := 7

-- Define the longer side length of the smaller rectangles
def longer_side : ℕ := 2 * shorter_side

-- Define the width of rectangle ABCD
def width_ABCD : ℕ := 2 * shorter_side

-- Define the length of rectangle ABCD
def length_ABCD : ℕ := longer_side

-- Define the area of rectangle ABCD
def area_ABCD : ℕ := length_ABCD * width_ABCD

-- Statement of the problem
theorem area_of_ABCD_is_196 : area_ABCD = 196 :=
by
  -- insert proof here
  sorry

end area_of_ABCD_is_196_l712_712226


namespace collinearity_of_T_K_I_l712_712896

-- Definitions of the points and lines
variables {A P Q M N C I T K : Type} [Nonempty A] [Nonempty P] [Nonempty Q] 
  [Nonempty M] [Nonempty N] [Nonempty C] [Nonempty I]

-- Intersection points conditions
def intersect (l₁ l₂ : Type) : Type := sorry

-- Given conditions
def condition_1 : T = intersect (line A P) (line C Q) := sorry
def condition_2 : K = intersect (line M P) (line N Q) := sorry

-- Proof that T, K, and I are collinear
theorem collinearity_of_T_K_I : collinear {T, K, I} := by
  have h1 : T = intersect (line A P) (line C Q) := condition_1
  have h2 : K = intersect (line M P) (line N Q) := condition_2
  -- Further steps needed to prove collinearity
  sorry

end collinearity_of_T_K_I_l712_712896


namespace sequence_expression_l712_712769

theorem sequence_expression {a : ℕ → ℝ} 
  (h : ∀ n : ℕ, (∏ i in finset.range (n + 1), (i + 1) * a (i + 1)) = 2^(n + 1)) :
  ∀ n : ℕ, a (n + 1) = 2 / (n + 1) :=
by
  intro n
  have H1 : a 1 = 2 / 1 := by sorry
  have Hn : ∀ n ≥ 1, a n = 2 / n := by sorry
  exact sorry

end sequence_expression_l712_712769


namespace segment_length_x_value_l712_712685

theorem segment_length_x_value : ∃ x : ℝ, 
  (dist (2, 2) (x, 5) = 6 ∧ x > 0) ↔ x = 2 + 3 * real.sqrt 3 := by
  sorry

end segment_length_x_value_l712_712685


namespace project_completion_time_l712_712649

theorem project_completion_time (A_days B_days quit_days : ℕ) (A_rate B_rate total_time : ℝ) :
  A_days = 20 →
  B_days = 30 →
  quit_days = 15 →
  A_rate = 1 / 20 →
  B_rate = 1 / 30 →
  (total_time - quit_days) * (A_rate + B_rate) + quit_days * B_rate = 1 →
  total_time = 36 :=
begin
  intros hA_days hB_days hquit_days hA_rate hB_rate heq_total,
  sorry
end

end project_completion_time_l712_712649


namespace circles_coincide_l712_712157

-- Definition of the problem conditions
variables {A1 A2 A3 : Type*} [euclidean_space A1] [euclidean_space A2] [euclidean_space A3]
variable {triangle : triangle A1 A2 A3}
variables (ω1 ω2 ω3 ω4 ω5 ω6 ω7 : circle ℝ)

-- Each circle w_k (k=2, ..., 7) is externally tangent to w_{k-1} 
-- and passes through appropriate points A_i and A_{i+1} (indices mod 3).
def circle_properties (ω: circle ℝ) (A1 A2 A3: Type*): Prop :=
  external_tangent ω ω1 ∧ passes_through ω A2 A3 ∧ -- ω2 passes through A2 and A3
  external_tangent ω ω2 ∧ passes_through ω A3 A1 ∧ -- ω3 passes through A3 and A1
  external_tangent ω ω3 ∧ passes_through ω A1 A2 ∧ -- ω4 passes through A1 and A2
  external_tangent ω ω4 ∧ passes_through ω A2 A3 ∧ -- ω5 passes through A2 and A3
  external_tangent ω ω5 ∧ passes_through ω A3 A1 ∧ -- ω6 passes through A3 and A1
  external_tangent ω ω6 ∧ passes_through ω A1 A2   -- ω7 passes through A1 and A2

-- The Lean theorem statement that needs proof.
theorem circles_coincide (h: circle_properties ω A1 A2 A3): 
  ω7 = ω1 := 
by
  sorry

end circles_coincide_l712_712157


namespace initial_sum_lent_l712_712299

/-- Given:
  A = 650       -- The final amount after 2 years.
  r = 0.05      -- Annual interest rate.
  n = 1         -- Number of times interest is compounded per year.
  t = 2         -- Number of years the money is invested.
  We need to prove:
  P, the initial sum lent, is approximately 589.74.
-/
theorem initial_sum_lent (A r : ℝ) (n t : ℕ) (P : ℝ) (hP : A = P * (1 + r / n) ^ (n * t)) :
  A = 650 ∧ r = 0.05 ∧ n = 1 ∧ t = 2 → P ≈ 589.74 := 
by sorry

end initial_sum_lent_l712_712299


namespace part1_part2_l712_712432

noncomputable def f (a x : ℝ) : ℝ := (a*x^2 + x + a) / real.exp x

-- Defining conditions for Part 1
def monotonic_increasing_interval (a : ℝ) :=
  if a > 0 then
    ∃ (interval : set ℝ), interval = set.Ioo ((a - 1) / a) 1 ∧ ∀ x ∈ interval, derivative (f a) x > 0
  else
    ∃ (interval1 interval2 : set ℝ), interval1 = set.Ioo (-∞) 1 ∧ interval2 = set.Ioo ((a - 1) / a) (∞) ∧
    (∀ x ∈ interval1, derivative (f a) x > 0) ∧ (∀ x ∈ interval2, derivative (f a) x > 0)

theorem part1 (a : ℝ) (h : a ≠ 0) : monotonic_increasing_interval a := sorry

-- Defining conditions for Part 2
noncomputable def f_zero (x : ℝ) : ℝ := x / real.exp x

def g (x x₁ : ℝ) : ℝ := (f_zero x - f_zero x₁) / (x - x₁)

theorem part2 (x₁ x x₂ : ℝ) (h1 : x₁ < x) (h2 : x < x₂) (h3 : x₂ < 2) : g x x₁ > g x₂ x₁ := sorry

end part1_part2_l712_712432


namespace expected_value_Y_l712_712805

noncomputable def X : ℕ → ℝ := sorry -- Define binomial distribution B(5, 0.3)

def Y (X : ℕ → ℝ) : ℕ → ℝ := λ n, 2 * X n - 1

theorem expected_value_Y :
  (∑ i in finset.range (5 + 1), (2 * (binomial 5 0.3) i * 0.3^i *(1 - 0.3)^(5 - i)) i - 1) = 2 := sorry

end expected_value_Y_l712_712805


namespace num_valid_subset_pairs_l712_712164

open Finset

/-- I is the set {1, 2, 3, 4} -/
def I : Finset ℕ := {1, 2, 3, 4}

/-- Conditions for subsets A and B, where A and B are non-empty subsets of I and the largest element of A
    is not greater than the smallest element of B -/
def valid_subset_pair (A B : Finset ℕ) : Prop :=
  A ≠ ∅ ∧ B ≠ ∅ ∧ ∀ a ∈ A, ∀ b ∈ B, a ≤ b

/-- The number of different valid subset pairs (A, B) -/
theorem num_valid_subset_pairs : ∃ n, n = 49 ∧
  ∃ s : Finset (Finset ℕ × Finset ℕ), 
  (∀ p ∈ s, valid_subset_pair p.fst p.snd) ∧
  s.card = 49 :=
sorry

end num_valid_subset_pairs_l712_712164


namespace triangle_inequality_l712_712414

variable {A B C M : Type} [metric_space M] {a b c m : M}
variable {MA MB MC AB BC CA : ℝ}

-- Conditions: M is an arbitrary point inside the triangle ABC
-- MA, MB, and MC are distances from M to the vertices A, B, and C respectively
-- AB, BC, and CA are distances between the vertices

axiom distance_triangle (hx : M) (hABC : ℝ) : MA < AB ∧ MB < BC ∧ MC < CA

theorem triangle_inequality (M : M) (ABC : M) :
  (min {MA, MB, MC} + MA + MB + MC < AB + BC + CA) :=
  sorry

end triangle_inequality_l712_712414


namespace line_intersects_circle_shortest_chord_and_line_eq_l712_712667

open Real

-- Define the line and the circle
def line (m : ℝ) : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), 2 * m * p.1 - p.2 - 8 * m - 3 = 0
def circle : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), (p.1 - 3)^2 + (p.2 + 6)^2 = 25

-- Prove that the line always intersects the circle for any m
theorem line_intersects_circle (m : ℝ) : ∃ (p : ℝ × ℝ), line m p ∧ circle p :=
sorry

-- Find the shortest length of the segment cut by the line on the circle and the equation of the line at this time
theorem shortest_chord_and_line_eq (m : ℝ) : 
  let l := line m in
  let shortest_length := 2 * sqrt (25 - d^2) in
  let intersect_pt := (p : ℝ × ℝ) in
  l (intersect_pt) ∧ circle (intersect_pt) ∧
  eq_line_slope (2 * m) (-1) l -- line equation x + 3y + 5 = 0

end line_intersects_circle_shortest_chord_and_line_eq_l712_712667


namespace power_congruence_l712_712161

theorem power_congruence (a b n : ℕ) (h : a ≡ b [MOD n]) : a^n ≡ b^n [MOD n^2] :=
sorry

end power_congruence_l712_712161


namespace train_crossing_time_l712_712706

def kmph_to_mps (kmph : ℝ) : ℝ := kmph * 1000 / 3600

theorem train_crossing_time (speed_kmph : ℝ) (length_meters : ℝ) : 
  speed_kmph = 72 ∧ length_meters = 140 → (length_meters / (kmph_to_mps speed_kmph) = 7) :=
by
  intro h
  cases h
  have speed_mps : ℝ := kmph_to_mps speed_kmph
  have speed_mps_def : speed_mps = 20 := by
    have conversion_factor := 1000 / 3600
    show speed_mps = 72 * conversion_factor
    sorry
  exact calc 
    length_meters / speed_mps
    _ = 140 / 20 : by rw [speed_mps_def]
    _ = 7 : by norm_num

end train_crossing_time_l712_712706


namespace solve_for_A_l712_712249

theorem solve_for_A (A B : ℕ) (h1 : 4 * 10 + A + 10 * B + 3 = 68) (h2 : 10 ≤ 4 * 10 + A) (h3 : 4 * 10 + A < 100) (h4 : 10 ≤ 10 * B + 3) (h5 : 10 * B + 3 < 100) (h6 : A < 10) (h7 : B < 10) : A = 5 := 
by
  sorry

end solve_for_A_l712_712249


namespace third_position_is_two_l712_712302

theorem third_position_is_two : 
  ∀ (s : Fin 37 → ℕ), 
    (∀ i, 1 ≤ s i ∧ s i ≤ 37) ∧
    (s 0 = 37) ∧ 
    (∀ k, 1 ≤ k → k < 37 → (s k ∣ (∑ i in Finset.range k, s i))) → 
  s 2 = 2 := 
by
  intros s cond
  sorry

end third_position_is_two_l712_712302


namespace pyramid_volume_proof_l712_712581

variable (α β S : ℝ)

noncomputable def pyramid_volume (α β S : ℝ) : ℝ :=
  (2 * S * real.sqrt(S * real.sin(α / 2) * real.cot(β))) / (3 * real.cos(α / 2))

theorem pyramid_volume_proof (α β S : ℝ) :
  volume_of_pyramid α β S = (2 * S * real.sqrt (S * real.sin (α / 2) * real.cot β)) / (3 * real.cos (α / 2)) :=
  sorry

end pyramid_volume_proof_l712_712581


namespace sum_of_coefficients_is_odd_l712_712265

open Polynomial

-- Define the polynomials and the conditions
theorem sum_of_coefficients_is_odd :
  ∀ (g1 g2 g3 g4 g5 : Polynomial ℤ),
  (f : Polynomial ℤ),
  (h_prod : f = g1 * g2 * g3 * g4 * g5) 
  (h_value : f.eval 1999 = 2000) →
  (∃ i ∈ {1, 2, 3, 4, 5}, (g1.sum % 2 = 1) ∨ (g2.sum % 2 = 1) ∨ (g3.sum % 2 = 1) ∨ (g4.sum % 2 = 1) ∨ (g5.sum % 2 = 1)) := 
sorry

end sum_of_coefficients_is_odd_l712_712265


namespace simplify_fraction_l712_712565

theorem simplify_fraction (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 :=
by
  sorry

end simplify_fraction_l712_712565


namespace find_m_l712_712388

open Real

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n-1)

theorem find_m (m : ℕ) :
    (∑ i in (range m).map (λ x, 2*x+1), (geometric_sequence 2 2 (i+1))) = 170 
    ∧ (∑ i in (range m).map (λ x, 2*(x+1)), (geometric_sequence 2 2 (i))) = 340
    → m = 4 :=
by
  sorry

end find_m_l712_712388


namespace elevator_height_after_20_seconds_l712_712094

-- Conditions
def starting_height : ℕ := 120
def descending_speed : ℕ := 4
def time_elapsed : ℕ := 20

-- Statement to prove
theorem elevator_height_after_20_seconds : 
  starting_height - descending_speed * time_elapsed = 40 := 
by 
  sorry

end elevator_height_after_20_seconds_l712_712094


namespace roundness_1000000_l712_712576

-- Definitions based on the conditions in the problem
def prime_factors (n : ℕ) : List (ℕ × ℕ) :=
  if n = 1 then []
  else [(2, 6), (5, 6)] -- Example specifically for 1,000,000

def roundness (n : ℕ) : ℕ :=
  (prime_factors n).map Prod.snd |>.sum

-- The main theorem
theorem roundness_1000000 : roundness 1000000 = 12 := by
  sorry

end roundness_1000000_l712_712576


namespace AE_perp_KL_l712_712108

-- Definitions of the geometrical objects and conditions
variables {ℝ : Type*} [real_field ℝ]

-- Assuming the existence of circles, tangents, and specific points
variables (I1 I2 A B C D K L E : Point ℝ)
variables (K1 K2 K3 : Circle ℝ)

axiom circle_intersect_points :
  (I1, K1).Circle.IntersectAtTwoPoints (I2, K2).Circle A B

axiom angle_is_obtuse :
  obtuse_angle I1 A I2

axiom tangent_K1_at_A :
  tangent_at_point K1 A (C ∈ K2)

axiom tangent_K2_at_A :
  tangent_at_point K2 A (D ∈ K1)

axiom K3_is_circumcircle_of_BCD :
  circumcircle ΔBCD = K3

axiom E_is_midpoint_of_arc_CD :
  midpoint_of_arc K3 C D E B

axiom AC_intersects_K3_at_K :
  intersects_again_on_circumcircle (line_through_points A C) K3 K

axiom AD_intersects_K3_at_L :
  intersects_again_on_circumcircle (line_through_points A D) K3 L

-- The main theorem (proof goal)
theorem AE_perp_KL :
  perpendicular (line_through_points A E) (line_through_points K L) :=
sorry

end AE_perp_KL_l712_712108


namespace maximal_sum_of_arithmetic_sequence_l712_712541

-- Definitions and conditions derived from the problem
def is_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def quadratic_inequality_solution (d a1 : ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 11 → d * x^2 + 2 * a1 * x ≥ 0

def a_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

def S_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a_n a1 d (i + 1)

-- The proof that the value of n that maximizes S_n is 6
theorem maximal_sum_of_arithmetic_sequence
  (a1 d : ℝ)
  (hd : d < 0)
  (hq : quadratic_inequality_solution d a1) :
  ∃ n : ℕ, n = 6 ∧ (∀ m : ℕ, S_n a1 d m ≤ S_n a1 d n) :=
sorry

end maximal_sum_of_arithmetic_sequence_l712_712541


namespace eq_value_l712_712830

theorem eq_value (x y : ℕ) (h1 : x - y = 9) (h2 : x = 9) : 3 ^ x * 4 ^ y = 19683 := by
  sorry

end eq_value_l712_712830


namespace length_CD_determined_l712_712230

theorem length_CD_determined (a r : ℝ) (A B C D O : ℝ → ℝ → Prop)
  (hAB : dist A B = a)
  (hA_perp : ∀ p, A p → ¬ B p)
  (hB_perp : ∀ p, B p → ¬ A p)
  (hC_line : ∀ p, C p → ∃ q, A q ∧ q = ⟨0, p⟩)
  (hD_line : ∀ p, D p → ∃ q, B q ∧ q = ⟨0, p⟩)
  (hCD_plane : ∃ p, O p ∧ dist (proj (midpoint A B)) p = r) :
  dist (line_segment C D) = sqrt (a^2 + 4 * r^2) := 
sorry

end length_CD_determined_l712_712230


namespace limit_of_largest_power_of_5_l712_712945

noncomputable theory
open Real

/-- 
  The largest power of 5 dividing sequence 1, 1, 2, 2, 3, 3, ..., n, n 
  is 5^(f(n)). We want to prove the limit as n tends to infinity of f(n)/n^2 is 1/8.
-/
theorem limit_of_largest_power_of_5 (f : ℕ → ℝ) : 
  (∀ n : ℕ, log 5 (largest_power_of_5_seq n) = f n) → 
  (tendsto (λ n, f(n) / (n^2)) at_top (𝓝 (1/8))) :=
  sorry

/--
  Helper function: largest power of 5 dividing the sequence 1, 1, 2, 2, ..., n, n.
-/
def largest_power_of_5_seq (n : ℕ) : ℕ :=
  sorry

end limit_of_largest_power_of_5_l712_712945


namespace max_value_of_tan_sq_l712_712253

-- We need to state that y = π / (1 + tan^2 x) has a maximum value equal to π
theorem max_value_of_tan_sq (x : ℝ) : ∃ y, y = π / (1 + (Real.tan x)^2) ∧ y ≤ π :=
by
  let y := π / (1 + (Real.tan x)^2)
  use y
  split
  { refl }
  { apply Real.le_of_eq
    sorry
  }

end max_value_of_tan_sq_l712_712253


namespace find_k_value_l712_712785

theorem find_k_value (x k : ℝ) (h : x = 2) (h_sol : (k / (x - 3)) - (1 / (3 - x)) = 1) : k = -2 :=
by
  -- sorry to suppress the actual proof
  sorry

end find_k_value_l712_712785


namespace correct_count_of_valid_integers_l712_712331

def is_valid_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n % 10 = 0) ∧ 
  ¬(∃ k, k == 5 ∧ (n / 100 = k ∨ (n / 10) % 10 = k ∨ n % 10 = k))

def count_valid_3_digit_integers : ℕ :=
  (Finset.range 1000).filter is_valid_integer |>.card

theorem correct_count_of_valid_integers : count_valid_3_digit_integers = 72 :=
  sorry

end correct_count_of_valid_integers_l712_712331


namespace poly_existence_l712_712552

theorem poly_existence (n : ℕ) (hn : 0 < n) : 
∃ p : Polynomial ℝ, p.degree ≤ 100 * n ∧ 
|p.eval 0| > ∑ i in Finset.range (n^2 + 1), |p.eval i| := 
sorry

end poly_existence_l712_712552


namespace min_vertical_segment_length_l712_712205

noncomputable def vertical_segment_length (x : ℝ) : ℝ :=
  abs (|x| - (-x^2 - 4*x - 3))

theorem min_vertical_segment_length :
  ∃ x : ℝ, vertical_segment_length x = 3 / 4 :=
by
  sorry

end min_vertical_segment_length_l712_712205


namespace pyramid_dihedral_angle_problem_l712_712966

theorem pyramid_dihedral_angle_problem
  (O A B C D E : Type)
  (s : ℝ)
  (h_pentagon : regular_pentagonal_base O A B C D E s)
  (h_congruent : congruent_edges O A B C D E s)
  (h_angle_AOB : ∠ O A B = 108°)
  (θ : ℝ)
  (h_cos_theta : cos θ = m + sqrt n)
  : ∃ (m n : ℤ), m + n = 4
:= sorry

end pyramid_dihedral_angle_problem_l712_712966


namespace shift_graph_l712_712617

noncomputable def shift_amount (f g : ℝ → ℝ) :=
  ∃ c, ∀ x, f x = g (x + c)

theorem shift_graph
  (f g : ℝ → ℝ)
  (hf : ∀ x, f x = √2 * cos (3 * x))
  (hg : ∀ x, g x = sin (3 * x) + cos (3 * x)) :
  shift_amount f g :=
begin
  sorry
end

end shift_graph_l712_712617


namespace amon_more_marbles_than_rhonda_l712_712303

theorem amon_more_marbles_than_rhonda (H1 : ∀ A R : ℕ, A + R = 215) (H2 : ∀ R : ℕ, R = 80) :
  ∀ A R : ℕ, A - R = 55 := 
begin
  sorry
end

end amon_more_marbles_than_rhonda_l712_712303


namespace goalies_in_team_l712_712295

-- Definitions based on conditions
def D := 10
def M := 2 * D
def S := 7
def total_players := 40

-- The proof goal
theorem goalies_in_team : ∃ G, G + D + M + S = total_players ∧ G = 3 := by
  -- Assuming the goal is to exist some number G that satisfies the given condition & the solution
  use 3
  simp [D, M, S, total_players]
  sorry

end goalies_in_team_l712_712295


namespace candy_division_l712_712705

theorem candy_division (total_candy : ℕ) (students : ℕ) (per_student : ℕ) 
  (h1 : total_candy = 344) (h2 : students = 43) : 
  total_candy / students = per_student ↔ per_student = 8 := 
by 
  sorry

end candy_division_l712_712705


namespace minimum_shift_value_l712_712457

theorem minimum_shift_value
    (m : ℝ) 
    (h1 : m > 0) :
    (∃ (k : ℤ), m = k * π - π / 3 ∧ k > 0) → (m = (2 * π) / 3) :=
sorry

end minimum_shift_value_l712_712457


namespace collinearity_of_T_K_I_l712_712924

noncomputable def intersection_point (l1 l2 : Line) : Point := sorry

-- Definitions of lines AP, CQ, MP, NQ based on the problem context
variables {A P C Q M N I : Point} (lAP lCQ lMP lNQ : Line)
variables (T : Point) (K : Point)

-- Given conditions
def condition_1 : T = intersection_point lAP lCQ := sorry
def condition_2 : K = intersection_point lMP lNQ := sorry

-- Theorem statement
theorem collinearity_of_T_K_I : T ∈ line_through K I :=
by {
  -- These are the conditions that we're given in the problem
  have hT : T = intersection_point lAP lCQ := sorry,
  have hK : K = intersection_point lMP lNQ := sorry,
  -- Rest of the proof would go here
  sorry
}

end collinearity_of_T_K_I_l712_712924


namespace rats_meet_in_ten_days_l712_712488

theorem rats_meet_in_ten_days : 
  ∀ (n : ℕ), (n ≥ 1) ∧ (2^n - 2 / 2^n - 999 ≥ 0) → (n = 10) :=
by
  assume n,
  intros,
  sorry

end rats_meet_in_ten_days_l712_712488


namespace least_faces_combined_l712_712620

theorem least_faces_combined (a b : ℕ) (h6 : a ≥ 6 ∧ b ≥ 6)
  (h_faces_distinct : ∀ n, n ∈ Finset.range a ∧ n ∈ Finset.range b)
  (h_prob_7_10 : ∃ p7 p10 : ℚ, p7 = 6 / (a * b) ∧ p10 = 8 / (a * b) ∧ p7 = 3 * p10 / 4)
  (h_prob_12 : ∃ n : ℕ, n = (a * b) / 12 ∧ n ≤ 8) :
  a + b = 17 :=
begin
  sorry
end

end least_faces_combined_l712_712620


namespace no_adjacent_enemies_l712_712715

variable {n : ℕ}

-- Assume there are 2n knights.
def knights : Finset (Fin (2 * n)) := Finset.univ

-- Each knight has at most n-1 enemies
variable (enemies : Fin (2 * n) → Finset (Fin (2 * n)))
-- Ensure that each knight has at most n-1 enemies
variable (h_enemies : ∀ k, (enemies k).card ≤ n-1)

theorem no_adjacent_enemies (h_enemies : ∀ k, (enemies k).card ≤ n-1) :
  ∃ (arrangement : Fin (2 * n) → Fin (2 * n)), 
    (∀ i, i < 2 * n → 
    (arrangement (nat_mod (i + 1) (2 * n))) ∉ enemies (arrangement i)) :=
sorry

end no_adjacent_enemies_l712_712715


namespace triangle_sides_ratios_l712_712965

theorem triangle_sides_ratios (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) :
  a / (b + c) = b / (a + c) + c / (a + b) :=
sorry

end triangle_sides_ratios_l712_712965


namespace sqrt_inequality_l712_712542

def solution_set (x : ℝ) : Prop := 1 < x ∧ x < 4

theorem sqrt_inequality (a b : ℝ) (ha : solution_set a) (hb : solution_set b) :
  |sqrt (a * b) - 2| < |2 * sqrt a - sqrt b| := 
by
  sorry

end sqrt_inequality_l712_712542


namespace collinearity_of_points_l712_712915

noncomputable theory
open_locale classical

variables {A P Q M N I T K : Type*}

-- Conditions given in the problem
variables [IncidenceGeometry A P Q M N I T K]
variable [IntersectionPoint T (Line A P) (Line C Q)]
variable [IntersectionPoint K (Line M P) (Line N Q)]

-- Statement of the proof problem
theorem collinearity_of_points :
  Collinear {T, K, I} :=
sorry

end collinearity_of_points_l712_712915


namespace fourth_angle_of_quadrilateral_l712_712228

/-- 
    Given a quadrilateral with three known angles,
    prove that the fourth angle is 85 degrees.
--/
theorem fourth_angle_of_quadrilateral (a b c : ℝ) (h1 : a = 75) (h2 : b = 80) (h3 : c = 120) 
(h4 : a + b + c + d = 360) : d = 85 :=
by
  have h_sum : a + b + c = 275 :=
    calc
      a + b + c = 75 + 80 + 120 : by rw [h1, h2, h3]
                _ = 275 : by norm_num
  have h_total : 360 - (a + b + c) = d := by linarith
  rw [h_sum] at h_total
  exact h_total

end fourth_angle_of_quadrilateral_l712_712228


namespace max_period_initial_phase_function_l712_712587

theorem max_period_initial_phase_function 
  (A ω ϕ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : A = 1/2) 
  (h2 : ω = 6) 
  (h3 : ϕ = π/4) 
  (h4 : ∀ x, f x = A * Real.sin (ω * x + ϕ)) : 
  ∀ x, f x = (1/2) * Real.sin (6 * x + (π/4)) :=
by
  sorry

end max_period_initial_phase_function_l712_712587


namespace shortest_side_proof_l712_712116

noncomputable def shortest_side_length (A B : ℝ) (a b c : ℝ) : Prop :=
  sin(A) = 5 / 13 ∧ 
  cos(B) = 3 / 5 ∧ 
  c = 63 ∧ 
  a = 25

theorem shortest_side_proof :
  ∃ (A B : ℝ) (a b c : ℝ), shortest_side_length A B a b c :=
by
  use [some_angle_A, some_angle_B, 25, some_side_b, 63]
  have h1 : sin(some_angle_A) = 5 / 13 := sorry
  have h2 : cos(some_angle_B) = 3 / 5 := sorry
  have h3 : 63 = 63 := by rfl
  have h4 : 25 = 25 := by rfl
  exact ⟨h1, h2, h3, h4⟩

end shortest_side_proof_l712_712116


namespace min_lambda_inequality_l712_712775

noncomputable def fibSeq (a b : ℕ) (n : ℕ) : ℕ :=
if n = 1 then a
else if n = 2 then b
else fibSeq a b (n - 1) + fibSeq a b (n - 2)

theorem min_lambda_inequality {a b : ℕ} (h : a ≤ b) :
  ∃ λ : ℝ, (λ = 2 + Real.sqrt 5) ∧ (∀ n : ℕ, 1 ≤ n →
  (∑ k in Finset.range (n + 1), fibSeq a b (k + 1))^2 ≤ λ * (fibSeq a b n.succ * fibSeq a b (n + 1))) :=
by
  sorry

end min_lambda_inequality_l712_712775


namespace complement_of_M_l712_712942

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 2*x > 0 }
def complement (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

theorem complement_of_M :
  complement U M = { x | 0 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end complement_of_M_l712_712942


namespace units_digit_subtraction_l712_712595

/-- Statement of the problem -/
theorem units_digit_subtraction (a b c : ℕ) (h : a = c + 3) :
  let original := 100 * a + 10 * b + c,
      reversed := 100 * c + 10 * b + a,
      adjusted_reversed := reversed + 50,
      result := original - adjusted_reversed in
  result % 10 = 7 :=
by {
  sorry
}

end units_digit_subtraction_l712_712595


namespace num_parallel_edges_l712_712445

-- Definition of a rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  width_ne_height : width ≠ height
  length_ne_height : length ≠ height

-- Theorem statement to prove the number of pairs of parallel edges
theorem num_parallel_edges (P : RectangularPrism) : 
  ∃ n : ℕ, n = 12 :=
by 
  -- Assuming the number of pairs of parallel edges is 12
  use 12
  sorry

end num_parallel_edges_l712_712445


namespace probability_outside_interval_N_0_4_9_l712_712722

noncomputable def prob_outside_interval : ℝ :=
  let μ : ℝ := 0
  let σ : ℝ := 2 / 3
  1 - (Mathlib.Probability.normal_cdf (μ, σ^2) 2 - Mathlib.Probability.normal_cdf (μ, σ^2) (-2))

theorem probability_outside_interval_N_0_4_9 : 
  prob_outside_interval = 0.0026 := 
by
  -- Proof goes here
  sorry

end probability_outside_interval_N_0_4_9_l712_712722


namespace find_dot_product_of_AC_BD_l712_712486

section parallelogram

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C D : V)
variables (AB AC AD : V) (ω : V)

-- Given conditions
def parallelogram (A B C D : V) : Prop :=
  ∃ (ω : V), A + ω = B ∧ C + ω = D

noncomputable def AB := 1 : ℝ
noncomputable def AD := 2 : ℝ

-- Main theorem to be proved
theorem find_dot_product_of_AC_BD
  (h_parallelogram : parallelogram A B C D) :
  inner_product (A + (A + AD • ω) - (A + AB • ω)) 
    (D + (A + AD • ω) - (B + AB • ω)) = 3 := 
sorry

end parallelogram

end find_dot_product_of_AC_BD_l712_712486


namespace triangle_problem_l712_712846

-- Given an acute triangle ABC, with angles A, B, C opposed to sides a, b, c respectively,
-- if sin A = 2 sqrt 2 / 3, a = 2, and the area of triangle ABC = sqrt 2,
-- prove that b + c = 2 sqrt 3.
theorem triangle_problem 
  (A B C : ℝ)    -- Angles of the triangle
  (a b c : ℝ)    -- Sides opposite to the angles
  (h_acute : A < (π / 2) ∧ B < (π / 2) ∧ C < (π / 2))  -- Acute angle condition
  (sin_A : sin A = (2 * real.sqrt 2) / 3) -- sin A = 2 sqrt 2 / 3
  (a_eq : a = 2)  -- a = 2
  (area_eq : (1/2) * b * c * sin A = real.sqrt 2)  -- Area of triangle = sqrt 2
  :
  b + c = 2 * real.sqrt 3 := sorry

end triangle_problem_l712_712846


namespace centers_circumcircles_parallel_line_l712_712200

variables {A B C D P Q : Type} [normed_space Mathlib.Real A] [normed_space Mathlib.Real B] [normed_space Mathlib.Real C] [normed_space Mathlib.Real D] [normed_space Mathlib.Real P] [normed_space Mathlib.Real Q]

-- Definitions of the points and conditions
def cyclic_quad (A B C D : Mathlib.Real) : Prop := ∃ ω, circle ω A ∧ circle ω B ∧ circle ω C ∧ circle ω D
def intersect_at (AC BD P : Mathlib.Real) : Prop := ∃ AC BD P, AC ∩ BD = P
def on_segment (Q BC : Mathlib.Real) : Prop := ∃ Q BC, Q ∈ BC
def perp (PQ AC : Mathlib.Real) : Prop := ∃ PQ AC, PQ ⊥ AC

-- Main theorem statement
theorem centers_circumcircles_parallel_line
  (h1 : cyclic_quad A B C D)
  (h2 : intersect_at AC BD P)
  (h3 : on_segment Q BC)
  (h4 : perp PQ AC) :
  ∃ l, centers_of_circumcircles l (triangle A P D) (triangle B Q D) ∧ parallel l AD :=
sorry

end centers_circumcircles_parallel_line_l712_712200


namespace find_kn_l712_712734

theorem find_kn (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 :=
by
  sorry

end find_kn_l712_712734


namespace stable_set_even_subset_count_l712_712517

open Finset

-- Definitions
def is_stable (S : Finset (ℕ × ℕ)) : Prop :=
  ∀ ⦃x y⦄, (x, y) ∈ S → ∀ x' y', x' ≤ x → y' ≤ y → (x', y') ∈ S

-- Main statement
theorem stable_set_even_subset_count (S : Finset (ℕ × ℕ)) (hS : is_stable S):
  (∃ E O : ℕ, E ≥ O ∧ E + O = 2 ^ (S.card)) :=
  sorry

end stable_set_even_subset_count_l712_712517


namespace subset_sum_bounds_l712_712162

theorem subset_sum_bounds (M m n : ℕ) (A : Finset ℕ)
  (h1 : 1 ≤ m) (h2 : m ≤ n) (h3 : 1 ≤ M) (h4 : M ≤ (m * (m + 1)) / 2) (hA : A.card = m) (hA_subset : ∀ x ∈ A, x ∈ Finset.range (n + 1)) :
  ∃ B ⊆ A, 0 ≤ (B.sum id) - M ∧ (B.sum id) - M ≤ n - m :=
by
  sorry

end subset_sum_bounds_l712_712162


namespace sequence_arithmetic_l712_712393

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 2 * n^2 - 2 * n) →
  (∀ n, a n = S n - S (n - 1)) →
  (∀ n, a n - a (n - 1) = 4) :=
by
  intros hS ha
  sorry

end sequence_arithmetic_l712_712393


namespace minimize_power_consumption_l712_712213

def power_consumption (x : ℝ) : ℝ :=
  (1 / 3) * x^3 - (39 / 2) * x^2 - 40 * x

theorem minimize_power_consumption : 
  ∃ x > 0, (∀ y > 0, y ≠ x → power_consumption y > power_consumption x) ∧ x = 40 := 
by
  sorry

end minimize_power_consumption_l712_712213


namespace Lara_age_in_10_years_l712_712137

theorem Lara_age_in_10_years (Lara_age_7_years_ago : ℕ) (h1 : Lara_age_7_years_ago = 9) : 
  Lara_age_7_years_ago + 7 + 10 = 26 :=
by
  rw [h1]
  norm_num
  sorry

end Lara_age_in_10_years_l712_712137


namespace product_sum_equality_l712_712551

open Finset

theorem product_sum_equality {n : ℕ} {x : ℝ} :
  (∏ i in range n, (1 + x^(2^i))) = ∑ j in range (2^n), x^j :=
by sorry

end product_sum_equality_l712_712551


namespace quadrilateral_ACDB_area_l712_712146

variable (O A B C D : Point)
variable (T R : Real)
variable (circle_center : Point)
noncomputable def is_right_angle (O A B : Point) : Prop := sorry
noncomputable def distance (P1 P2 : Point) : Real := sorry

theorem quadrilateral_ACDB_area 
  (h1 : circle_center = O)
  (h2 : distance O A = distance O B = R)
  (h3 : is_right_angle O A B)
  (h4 : distance A C = T - 3)
  (h5 : distance C D = 5)
  (h6 : distance B D = 6)
  (hT : T = 10)
  (hR : 7 < R) :
  calc
    (let OC := R - 7 in
     let OD := R - 6 in
     let area_AOB := 1/2 * R * R * sin (90° : Real) in
     let area_COD := 1/2 * OC * OD * sin (90° : Real) in
     let area_ACDB := area_AOB - area_COD in
     area_ACDB = 44) :=
begin
  sorry
end

end quadrilateral_ACDB_area_l712_712146


namespace radius_of_intersection_yz_plane_l712_712296

theorem radius_of_intersection_yz_plane 
  (center_sphere : ℝ × ℝ × ℝ) 
  (radius_xy_plane : ℝ)
  (center_circle_yz : ℝ × ℝ × ℝ)
  (z_of_center : ℝ) :
  center_sphere = (3, 3, -8) →
  radius_xy_plane = 2 →
  center_circle_yz = (0, 3, -8) →
  z_of_center = -8 →
  let R := sqrt (2^2 + (8:ℝ)^2) in
  sqrt (R^2 - 3^2) = sqrt 59 :=
by {
  intros,
  sorry
}

end radius_of_intersection_yz_plane_l712_712296


namespace cot_tan_equivalence_l712_712469

variable {A B C a b c : ℝ}

def cot (θ : ℝ) : ℝ := real.cos θ / real.sin θ
def tan (θ : ℝ) : ℝ := real.sin θ / real.cos θ

theorem cot_tan_equivalence
  (h : a^2 + b^2 = 6 * c^2)
  (hA : real.sin A ≠ 0)
  (hB : real.sin B ≠ 0)
  (hC : real.cos C ≠ 0) :
  (cot A + cot B) * tan C = 2 / 5 :=
sorry

end cot_tan_equivalence_l712_712469


namespace equation_of_line_l712_712585

-- Defining the conditions
def slope : ℝ := 5
def point : ℝ × ℝ := (0, 2)

-- The theorem we aim to prove
theorem equation_of_line :
  ∃ (a b c : ℝ), a = 5 ∧ b = -1 ∧ c = 2 ∧ (∀ x y, y = slope * x + c → a * x + b * y + c = 0) :=
by
  sorry

end equation_of_line_l712_712585


namespace line_through_points_has_sum_seven_l712_712202

variable (P1 : ℤ × ℤ) (P2 : ℤ × ℤ)

def slope (P1 P2 : ℤ × ℤ) : ℚ :=
  (P2.2 - P1.2) / (P2.1 - P1.1)

def y_intercept (P1 P2 : ℤ × ℤ) : ℚ :=
  P1.2 - (slope P1 P2) * P1.1

theorem line_through_points_has_sum_seven (P1 P2 : ℤ × ℤ) (hP1 : P1 = (-3, 1)) (hP2 : P2 = (1, 7)) :
  slope P1 P2 + y_intercept P1 P2 = 7 :=
by
  sorry

end line_through_points_has_sum_seven_l712_712202


namespace tangent_curves_intersect_at_O_l712_712766

theorem tangent_curves_intersect_at_O (a : ℝ) :
  (∃ l : ℝ × ℝ → bool, ∃ O : ℝ × ℝ, O = (0,0) ∧
    ∀ (x : ℝ), l (x, x^3 - 3*x^2 + 2*x) = true ∧ 
               l (x, x^2 + a) = true) → 
  a = 1 ∨ a = 1/64 :=
by sorry

end tangent_curves_intersect_at_O_l712_712766


namespace esther_biking_speed_l712_712493

theorem esther_biking_speed (d x : ℝ)
  (h_bike_speed : x > 0)
  (h_average_speed : 5 = 2 * d / (d / x + d / 3)) :
  x = 15 :=
by
  sorry

end esther_biking_speed_l712_712493


namespace find_k_n_l712_712733

theorem find_k_n (k n : ℕ) (h_kn_pos : 0 < k ∧ 0 < n) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 := 
by {
  sorry
}

end find_k_n_l712_712733


namespace quadratic_max_passes_points_l712_712593

theorem quadratic_max_passes_points (a b c : ℝ)
  (h1: ∀ x,  ax^2 + bx + c ≤ 72)
  (h2: (0, -1) ∈ { (x:ℝ, y:ℝ) | y = ax^2 + bx + c } )
  (h3: (6, -1) ∈ { (x:ℝ, y:ℝ) | y = ax^2 + bx + c } ) :
  a + b + c = 356/9 := 
sorry

end quadratic_max_passes_points_l712_712593


namespace probability_of_intersection_l712_712115

theorem probability_of_intersection :
  let k_interval := Set.Icc (-1 : ℝ) 1
  let intersection_condition (k : ℝ) :=  - (1/2 : ℝ) ≤ k ∧ k ≤ (1/2 : ℝ)
  let probability :=
    (∫ x in k_interval, if intersection_condition x then 1 else 0)
    / (∫ x in k_interval, 1)
  probability = (1 / 2 : ℝ) :=
by
  let k_interval := Set.Icc (-1 : ℝ) 1
  let intersection_condition (k : ℝ) := - (1/2 : ℝ) ≤ k ∧ k ≤ (1/2 : ℝ)
  let probability :=
    (∫ x in k_interval, if intersection_condition x then 1 else 0)
    / (∫ x in k_interval, 1)
  have h: probability = (1 / 2 : ℝ) := sorry
  exact h

end probability_of_intersection_l712_712115


namespace average_employees_per_week_l712_712676

theorem average_employees_per_week (x : ℝ)
  (h1 : ∀ (x : ℝ), ∃ y : ℝ, y = x + 200)
  (h2 : ∀ (x : ℝ), ∃ z : ℝ, z = x + 150)
  (h3 : ∀ (x : ℝ), ∃ w : ℝ, w = 2 * (x + 150))
  (h4 : ∀ (w : ℝ), w = 400) :
  (250 + 50 + 200 + 400) / 4 = 225 :=
by 
  sorry

end average_employees_per_week_l712_712676


namespace phone_numbers_divisible_by_13_l712_712439

theorem phone_numbers_divisible_by_13 :
  ∃ (x y z : ℕ), (x < 10) ∧ (y < 10) ∧ (z < 10) ∧ (100 * x + 10 * y + z) % 13 = 0 ∧ (2 * y = x + z) :=
  sorry

end phone_numbers_divisible_by_13_l712_712439


namespace triangle_area_l712_712000

theorem triangle_area :
  let line1 (x : ℝ) := 2 * x + 1
  let line2 (x : ℝ) := (16 + x) / 4
  ∃ (base height : ℝ), height = (16 + 2 * base) / 7 ∧ base * height / 2 = 18 / 7 :=
  by
    sorry

end triangle_area_l712_712000


namespace centroid_coordinates_l712_712720

theorem centroid_coordinates (a : ℝ) (h_a_pos : a > 0) :
  let S := ∫ x in 0..1, (a - a * x^3) 
  let M_x := (1 / 2) * ∫ x in 0..1, (a^2 - (a * x^3)^2)
  let M_y := ∫ x in 0..1, x * (a - a * x^3) 
  (S = (3 * a) / 4) →
  (M_x = (3 * a^2) / 7) →
  (M_y = (3 * a) / 10) →
  let x_c := M_y / S
  let y_c := M_x / S
  (x_c = 0.4) ∧ (y_c = (4 * a) / 7) :=
by
  intros S M_x M_y hS hMx hMy
  sorry

end centroid_coordinates_l712_712720


namespace vanessa_points_record_l712_712628

theorem vanessa_points_record 
  (P : ℕ) 
  (H₁ : P = 48) 
  (O : ℕ) 
  (H₂ : O = 6 * 3.5) : V = (P - O) → V = 27 :=
by
  sorry

end vanessa_points_record_l712_712628


namespace david_remaining_money_l712_712652

variable (S : ℝ) -- Assuming S is a real number since it represents money

-- Conditions
axiom initial_amount : S = 1300
axiom spent_less_amount : 1800 - S = S - 800

-- Goal
theorem david_remaining_money : 1800 - S = 500 :=
by
  rw initial_amount at spent_less_amount
  exact spent_less_amount

end david_remaining_money_l712_712652


namespace hyperbola_standard_equation_l712_712060

theorem hyperbola_standard_equation
  (a b : ℝ)
  (asymptotes : ∀ (x y : ℝ), (y = sqrt 3 * x ∨ y = -sqrt 3 * x))
  (focus : ∃ y, (0, y) = (0, -2 * sqrt 2))
  (hyperbola_form : ∀ (x y : ℝ), y^2 / a^2 - x^2 / b^2 = 1) :
  ∃ a b : ℝ, a^2 = 6 ∧ b^2 = 2 ∧ ∀ (x y : ℝ), y^2 / 6 - x^2 / 2 = 1 := 
 sorry

end hyperbola_standard_equation_l712_712060


namespace number_of_true_propositions_l712_712611

def inverse_proposition (x y : ℝ) : Prop :=
  ¬(x + y = 0 → (x ≠ -y))

def contrapositive_proposition (a b : ℝ) : Prop :=
  (a^2 ≤ b^2) → (a ≤ b)

def negation_proposition (x : ℝ) : Prop :=
  (x ≤ -3) → ¬(x^2 + x - 6 > 0)

theorem number_of_true_propositions : 
  (∃ (x y : ℝ), inverse_proposition x y) ∧
  (∃ (a b : ℝ), contrapositive_proposition a b) ∧
  ¬(∃ (x : ℝ), negation_proposition x) → 
  2 = 2 :=
by
  sorry

end number_of_true_propositions_l712_712611


namespace collinear_T_K_I_l712_712903

noncomputable def T (A P C Q : Point) : Point := intersection (line_through A P) (line_through C Q)
noncomputable def K (M P N Q : Point) : Point := intersection (line_through M P) (line_through N Q)

theorem collinear_T_K_I (A P C Q M N I : Point) :
  collinear [T A P C Q, K M P N Q, I] :=
sorry

end collinear_T_K_I_l712_712903


namespace collinearity_of_T_K_I_l712_712899

-- Definitions of the points and lines
variables {A P Q M N C I T K : Type} [Nonempty A] [Nonempty P] [Nonempty Q] 
  [Nonempty M] [Nonempty N] [Nonempty C] [Nonempty I]

-- Intersection points conditions
def intersect (l₁ l₂ : Type) : Type := sorry

-- Given conditions
def condition_1 : T = intersect (line A P) (line C Q) := sorry
def condition_2 : K = intersect (line M P) (line N Q) := sorry

-- Proof that T, K, and I are collinear
theorem collinearity_of_T_K_I : collinear {T, K, I} := by
  have h1 : T = intersect (line A P) (line C Q) := condition_1
  have h2 : K = intersect (line M P) (line N Q) := condition_2
  -- Further steps needed to prove collinearity
  sorry

end collinearity_of_T_K_I_l712_712899


namespace shift_parabola_eq_l712_712561

theorem shift_parabola_eq (x : ℝ) : 
  let y := x^2 + 1
  (y' := (x + 2)^2 - 2) in 
  (∃ y'', y'' = y' ∧ y'' = (x^2 + 1) → y'' = y') :=
by
  sorry

end shift_parabola_eq_l712_712561


namespace trigonometric_identity_l712_712758

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
  sorry

end trigonometric_identity_l712_712758


namespace find_f_2n_l712_712764

variable (f : ℤ → ℤ)
variable (n : ℕ)

axiom axiom1 {x y : ℤ} : f (x + y) = f x + f y + 2 * x * y + 1
axiom axiom2 : f (-2) = 1

theorem find_f_2n (n : ℕ) (h : n > 0) : f (2 * n) = 4 * n^2 + 2 * n - 1 := sorry

end find_f_2n_l712_712764


namespace investments_are_beneficial_l712_712375

-- Definitions of examples and their benefits as given in the conditions
def investment_in_education : Prop :=
  ∃ (benefit : String), 
    benefit = "enhances employability and earning potential"

def investment_in_physical_health : Prop :=
  ∃ (benefit : String), 
    benefit = "reduces future healthcare costs and enhances overall well-being"

def investment_in_reading_books : Prop :=
  ∃ (benefit : String), 
    benefit = "cultivates intellectual growth and contributes to personal and professional success"

-- The theorem combining the three investments and their benefits
theorem investments_are_beneficial :
  investment_in_education ∧ investment_in_physical_health ∧ investment_in_reading_books :=
by
  split;
  { 
    existsi "enhances employability and earning potential", sorry <|>
    existsi "reduces future healthcare costs and enhances overall well-being", sorry <|>
    existsi "cultivates intellectual growth and contributes to personal and professional success", sorry
  }

end investments_are_beneficial_l712_712375


namespace rhombus_area_3_4_l712_712054

-- Define the diagonals of the rhombus
def d1 : ℝ := 3
def d2 : ℝ := 4

-- Define the area formula for the rhombus
def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- The theorem we need to prove
theorem rhombus_area_3_4 : rhombus_area d1 d2 = 6 := by
  -- The proof would go here
  sorry

end rhombus_area_3_4_l712_712054


namespace cube_surface_area_l712_712578

noncomputable def total_surface_area_of_cube (Q : ℝ) : ℝ :=
  8 * Q * Real.sqrt 3 / 3

theorem cube_surface_area (Q : ℝ) (h : Q > 0) :
  total_surface_area_of_cube Q = 8 * Q * Real.sqrt 3 / 3 :=
sorry

end cube_surface_area_l712_712578


namespace max_modulus_l712_712045

open Complex

noncomputable def max_modulus_condition (z : ℂ) : Prop :=
  abs (z - (0 + 2*Complex.I)) = 1

theorem max_modulus : ∀ z : ℂ, max_modulus_condition z → abs z ≤ 3 :=
  by sorry

end max_modulus_l712_712045


namespace min_value_box_l712_712824

theorem min_value_box (a b : ℤ) (h_distinct : a ≠ b) (h_prod : a * b = 30) : 
  ∃ (Box : ℤ), (Box = a^2 + b^2) ∧ Box = 61 :=
by
  use a^2 + b^2
  split
  sorry
  sorry

end min_value_box_l712_712824


namespace student_signup_ways_l712_712273

theorem student_signup_ways : (3 * 3 * 3 * 3 = 81) := by
  exact (nat.mul_comm 3 3 ▸ nat.mul_comm (3 * 3) (3 * 3) ▸ nat.mul_comm 3 9 ▸ nat.mul_comm 9 9 ▸ congr (by refl) (by refl))
  sorry

-- The above Lean theorem states that the product of the multiplication of choices (3 * 3 * 3 * 3) equals 81.
-- The sorry keyword is used to skip the proof steps.

end student_signup_ways_l712_712273


namespace madison_classes_l712_712544

/-- Madison's classes -/
def total_bell_rings : ℕ := 9

/-- Each class requires two bell rings (one to start, one to end) -/
def bell_rings_per_class : ℕ := 2

/-- The number of classes Madison has on Monday -/
theorem madison_classes (total_bell_rings bell_rings_per_class : ℕ) (last_class_start_only : total_bell_rings % bell_rings_per_class = 1) : 
  (total_bell_rings - 1) / bell_rings_per_class + 1 = 5 :=
by
  sorry

end madison_classes_l712_712544


namespace right_triangle_ineq_l712_712963

-- Definitions based on conditions in (a)
variables {a b c m f : ℝ}
variable (h_a : a ≥ 0)
variable (h_b : b ≥ 0)
variable (h_c : c > 0)
variable (h_a_b : a ≤ b)
variable (h_triangle : c = Real.sqrt (a^2 + b^2))
variable (h_m : m = a * b / c)
variable (h_f : f = (Real.sqrt 2 * a * b) / (a + b))

-- Proof goal based on the problem in (c)
theorem right_triangle_ineq : m + f ≤ c :=
sorry

end right_triangle_ineq_l712_712963


namespace set_intersection_complement_l712_712807

-- Definitions of the sets A and B
def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

-- Statement of the problem for Lean 4
theorem set_intersection_complement :
  A ∩ (Set.compl B) = {1, 5, 7} := 
sorry

end set_intersection_complement_l712_712807


namespace collinearity_of_T_K_I_l712_712925

noncomputable def intersection_point (l1 l2 : Line) : Point := sorry

-- Definitions of lines AP, CQ, MP, NQ based on the problem context
variables {A P C Q M N I : Point} (lAP lCQ lMP lNQ : Line)
variables (T : Point) (K : Point)

-- Given conditions
def condition_1 : T = intersection_point lAP lCQ := sorry
def condition_2 : K = intersection_point lMP lNQ := sorry

-- Theorem statement
theorem collinearity_of_T_K_I : T ∈ line_through K I :=
by {
  -- These are the conditions that we're given in the problem
  have hT : T = intersection_point lAP lCQ := sorry,
  have hK : K = intersection_point lMP lNQ := sorry,
  -- Rest of the proof would go here
  sorry
}

end collinearity_of_T_K_I_l712_712925


namespace volunteer_distribution_l712_712724

open nat

/-- The total number of ways to assign 8 volunteers (including 3 females and 5 males) 
to 3 pavilions such that each pavilion has at least one male and one female volunteer is 180. -/
theorem volunteer_distribution :
  ∃ (arrangements: finset (fin 8 → fin 3)),
    arrangements.card = 180 ∧ 
    ∀ (f : fin 8 → fin 3),
    f ∈ arrangements →
      (∀ p : fin 3, ∃ i : fin 8, ∃ j : fin 8, f i = p ∧ f j = p ∧ i < 3 ∧ j ≥ 3) :=
begin
  /-
  The proof goes here.
  -/
  sorry
end

end volunteer_distribution_l712_712724


namespace circles_intersect_l712_712335

-- Definition of the first circle
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

-- Definition of the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 8 = 0

-- Proving that the circles defined by C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
by sorry

end circles_intersect_l712_712335


namespace expression_value_l712_712640

theorem expression_value (a b : ℕ) (h1 : a = 36) (h2 : b = 9) : (a + b)^2 - (b^2 + a^2) = 648 :=
by {
  rw [h1, h2],
  norm_num
}

end expression_value_l712_712640


namespace range_of_a_l712_712776

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc 1 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) →
  (a = 1 ∨ a ≤ -2) :=
by
  sorry

end range_of_a_l712_712776


namespace area_of_two_rotated_congruent_squares_l712_712624

def square_area (side_length : ℝ) : ℝ :=
  side_length ^ 2

def overlapping_area (side_length : ℝ) : ℝ :=
  (side_length * real.sqrt 2 / 2) ^ 2

def total_covered_area (side_length : ℝ) : ℝ :=
  2 * square_area side_length - overlapping_area side_length

theorem area_of_two_rotated_congruent_squares (side_length : ℝ) (h : side_length = 12) :
  total_covered_area side_length = 216 :=
by
  rw [h]
  simp only [total_covered_area, square_area, overlapping_area]
  rw [real.sqrt_eq_rpow, real.rpow_two, pow_two, mul_assoc, mul_div_assoc, mul_div_cancel]
  norm_num
  sorry

end area_of_two_rotated_congruent_squares_l712_712624


namespace sin_cos_sixth_power_sum_l712_712471

theorem sin_cos_sixth_power_sum
  (A B C : Type) [EuclideanTriangle A B C]
  (AB AC BC : ℝ) (alpha : ℝ)
  (hAB : AB = 4) (hBC : BC = 7) (hCA : AC = 5)
  (h_alpha_def : alpha = angle A B C) :
  sin_pow 6 (alpha / 2) + cos_pow 6 (alpha / 2) = 7 / 25 := 
sorry

end sin_cos_sixth_power_sum_l712_712471


namespace xy_system_l712_712083

theorem xy_system (x y : ℚ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 :=
by
  sorry

end xy_system_l712_712083


namespace op_op_k_l712_712329

def op (x y : ℝ) : ℝ := x^3 + x - y

theorem op_op_k (k : ℝ) : op k (op k k) = k := sorry

end op_op_k_l712_712329


namespace case_a_angle_OAC_case_b_angle_OAC_l712_712881

-- Define the given conditions for the triangle and the angles.
variables {α : Type*} [normed_division_ring α] [normed_ring α]
variables (A B C : α) (O : α)

axiom hO_center : ∃ T : triangle α, O = circumcenter T ∧ A, B, C ∈ T.vertices

-- Case a: ∠B = 50° ⇒ ∠OAC = 40°
theorem case_a_angle_OAC :
  ∠ B = 50 * π / 180 → ∠ OAC = 40 * π / 180 :=
sorry

-- Case b: ∠B = 126° ⇒ ∠OAC = 36°
theorem case_b_angle_OAC :
  ∠ B = 126 * π / 180 → ∠ OAC = 36 * π / 180 :=
sorry

end case_a_angle_OAC_case_b_angle_OAC_l712_712881


namespace collinear_TKI_l712_712923

-- Definitions based on conditions
variables {A P Q C M N T K I : Type}
variables (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)

-- Definitions to represent the intersection points 
def T_def (A P Q C : Type) (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) : Prop := 
  ∃ T : Type, line_AP A T ∧ line_CQ C T

def K_def (M P N Q : Type) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop) : Prop := 
  ∃ K : Type, line_MP M K ∧ line_NQ N K

-- Theorem statement
theorem collinear_TKI (A P Q C M N T K I : Type)
  (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)
  (hT : T_def A P Q C line_AP line_CQ) 
  (hK : K_def M P N Q line_MP line_NQ) : 
  collinear T K I :=
sorry

end collinear_TKI_l712_712923


namespace collinear_T_K_I_l712_712909

noncomputable def T (A P C Q : Point) : Point := intersection (line_through A P) (line_through C Q)
noncomputable def K (M P N Q : Point) : Point := intersection (line_through M P) (line_through N Q)

theorem collinear_T_K_I (A P C Q M N I : Point) :
  collinear [T A P C Q, K M P N Q, I] :=
sorry

end collinear_T_K_I_l712_712909


namespace cos_theta_from_point_l712_712057

theorem cos_theta_from_point (theta : Real) (h : (4, -3) ∈ {(x, y) | y = tan_real θ * x}) :
  cos(theta) = 4 / 5 :=
sorry

end cos_theta_from_point_l712_712057


namespace stacy_heather_rate_difference_l712_712190

/-- Stacy and Heather are 10 miles apart and walk towards each other along the same route.
    Heather's constant walking rate is 5 miles/hour.
    Stacy walks at a constant rate some unknown rate faster than Heather.
    Heather starts her journey 24 minutes (or 0.4 hours) after Stacy.
    Heather has walked 3.4545454545454546 miles when they meet.
    Prove that the difference in Stacy's and Heather's walking rates is 1 mile per hour. -/
theorem stacy_heather_rate_difference
  (d_sh : 10)
  (r_h : 5)
  (start_delay : 24 / 60)
  (d_h_meet : 3.4545454545454546) :
  ∃ r_s : ℝ, (r_s - r_h = 1) :=
by
  sorry

end stacy_heather_rate_difference_l712_712190


namespace f_of_3_l712_712052

def f : ℝ → ℝ :=
  λ x, if 1 ≤ x ∧ x < 2 then 1 else if x = 2 then 2 else if 2 < x ∧ x ≤ 4 then 3 else 0

theorem f_of_3 : f 3 = 3 := by
  sorry

end f_of_3_l712_712052


namespace problem_statement_l712_712535

theorem problem_statement (a b c d : ℝ) 
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (hcd : c ≤ d)
  (hsum : a + b + c + d = 0)
  (hinv_sum : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 :=
sorry

end problem_statement_l712_712535


namespace range_of_a_bisection_method_l712_712800

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

theorem range_of_a (a : ℝ) 
  (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f a x = 0) :
  (a ≥ 12 * (27 - 4 * Real.sqrt 6) / 211) ∧ (a ≤ 12 * (27 + 4 * Real.sqrt 6) / 211) :=
sorry

theorem bisection_method (a : ℝ) 
  (h : a = 32 / 17) 
  (h1 : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f a x = 0) :
  ∃ x : ℝ, (0 < x ∧ x < 1) ∧ f a x = 0 ∧ Real.dist x (1 / 2) < 0.1 :=
sorry

end range_of_a_bisection_method_l712_712800


namespace min_expression_l712_712353

theorem min_expression : ∀ x y : ℝ, ∃ x, 4 * x^2 + 4 * x * (Real.sin y) - (Real.cos y)^2 = -1 := by
  sorry

end min_expression_l712_712353


namespace dot_product_of_vectors_l712_712802

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def line_intersection_points : set (ℝ × ℝ) := { p | ∃ x y, 
                                               y^2 = 4 * x ∧ 
                                               y = (2 / 3) * (x + 2) ∧ 
                                               p = (x, y) }

def vector_dot_product (v1 v2 : ℝ × ℝ) :=
  v1.1 * v2.1 + v1.2 * v2.2

def vector_from_focus (v : ℝ × ℝ) := (v.1 - (parabola_focus.1), v.2 - (parabola_focus.2))

theorem dot_product_of_vectors : 
  ∀ M N ∈ line_intersection_points, 
  vector_dot_product (vector_from_focus M) (vector_from_focus N) = 8 :=
by sorry

end dot_product_of_vectors_l712_712802


namespace area_of_square_from_string_l712_712646

theorem area_of_square_from_string (L : ℝ) (h : L = 32) : 
  ∃ (a : ℝ), (a = (L / 4) * (L / 4) ∧ a = 64) :=
by 
  use (L / 4) * (L / 4)
  split
  apply _ -- proof to show (L / 4) * (L / 4) equals proposed a
  apply _ -- proof to show proposed a is equal to 64
  sorry

end area_of_square_from_string_l712_712646


namespace coefficient_x4_in_expansion_l712_712635

theorem coefficient_x4_in_expansion (n : ℕ) (a b : ℝ) (k : ℕ) (h_n : n = 8) (h_a : a = x) (h_b : b = 3 * sqrt 3) (h_k : k = 4) :
  binomial_coefficient n k * (a ^ (n - k)) * (b ^ k) = 51030 * (x ^ 4) :=
by
  sorry

end coefficient_x4_in_expansion_l712_712635


namespace ratio_of_segments_l712_712522

-- Definitions capturing the conditions
variables {A B C D E F : EuclideanGeometry.Point}
variable [EuclideanGeometry.Triangle A B C]

-- Assume D is the midpoint of BC, hence AD is a median
def is_median (A : EuclideanGeometry.Point) (D : EuclideanGeometry.Point) :=
  EuclideanGeometry.midpoint D B C

-- Assume E lies on AD and F lies on AB, with a line through C
def intersection_conditions (E : EuclideanGeometry.Point) (F : EuclideanGeometry.Point) :=
  EuclideanGeometry.line_through E A D ∧ EuclideanGeometry.line_through F A B ∧ EuclideanGeometry.line_through F C

-- The main goal to prove
theorem ratio_of_segments (h_median : is_median A D) (h_intersections : intersection_conditions E F) :
  EuclideanGeometry.length A E / EuclideanGeometry.length E D = 2 * EuclideanGeometry.length A F / EuclideanGeometry.length F B :=
by
  sorry

end ratio_of_segments_l712_712522


namespace Lara_age_10_years_from_now_l712_712133

theorem Lara_age_10_years_from_now (current_year_age : ℕ) (age_7_years_ago : ℕ)
  (h1 : age_7_years_ago = 9) (h2 : current_year_age = age_7_years_ago + 7) :
  current_year_age + 10 = 26 :=
by
  sorry

end Lara_age_10_years_from_now_l712_712133


namespace staircase_toothpicks_l712_712843

theorem staircase_toothpicks (n : ℕ) (total_toothpicks : ℕ) :
  (∑ k in finset.range(n + 1), 3 * k) = total_toothpicks →
  (∑ k in finset.range(6), 3 * k) = 90 →
  total_toothpicks = 300 →
  n = 13 :=
by
  intros h_general h_base h_300
  sorry

end staircase_toothpicks_l712_712843


namespace average_sales_is_91_point_67_l712_712978

def sales : List ℕ := [120, 90, 50, 110, 80, 100]

def total_sales (sales : List ℕ) : ℕ := sales.foldl (· + ·) 0

def number_of_months (sales : List ℕ) : ℕ := sales.length

def average_sales (sales : List ℕ) : ℝ := (total_sales sales : ℝ) / (number_of_months sales : ℝ)

theorem average_sales_is_91_point_67 : average_sales sales = 91.67 := 
  sorry

end average_sales_is_91_point_67_l712_712978


namespace max_andy_l712_712229

def max_cookies_eaten_by_andy (total : ℕ) (k1 k2 a b c : ℤ) : Prop :=
  a + b + c = total ∧ b = 2 * a + 2 ∧ c = a - 3

theorem max_andy (total : ℕ) (a : ℤ) :
  (∀ b c, max_cookies_eaten_by_andy total 2 (-3) a b c) → a ≤ 7 :=
by
  intros H
  sorry

end max_andy_l712_712229


namespace area_of_largest_square_l712_712863

theorem area_of_largest_square (ABC : Triangle) (right_angle : ABC.angle BAC = 90)
  (isosceles : ABC.side AB = ABC.side AC)
  (sum_of_areas_eq : ABC.side AB ^ 2 + ABC.side AC ^ 2 + (ABC.side BC ^ 2) = 450) :
  (ABC.side BC ^ 2) = 225 :=
sorry

end area_of_largest_square_l712_712863


namespace tan_a_div_tan_b_l712_712528

variable {a b : ℝ}

-- Conditions
axiom sin_a_plus_b : Real.sin (a + b) = 1/2
axiom sin_a_minus_b : Real.sin (a - b) = 1/4

-- Proof statement (without the explicit proof)
theorem tan_a_div_tan_b : (Real.tan a) / (Real.tan b) = 3 := by
  sorry

end tan_a_div_tan_b_l712_712528


namespace distance_from_f1_to_line_f2M_l712_712067

-- Defining a hyperbola
def is_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 6 - y^2 / 3 = 1)

-- Defining coordinates of the foci
def foci_f1 : ℝ × ℝ := (-3, 0)
def foci_f2 : ℝ × ℝ := (3, 0)

-- Defining the point M given in the problem
def point_M : ℝ × ℝ := (3, sqrt 6 / 2)

-- Theorem statement for the distance calculation
theorem distance_from_f1_to_line_f2M :
  ∀ (x y : ℝ), is_hyperbola x y ∧ (x, y) = point_M ∧ (x, 0) = foci_f1 ∧ (x = 3) →
  let MF1 := abs (y - 0)
  let MF2 := abs (sqrt 6 / 2 - 0) + sqrt 6 * 2
  let F1F2 := abs (3 - (-3))
  ↑|F1F2 * MF1 / MF2| = 6 / 5 :=
sorry

end distance_from_f1_to_line_f2M_l712_712067


namespace max_possible_percent_error_in_garden_area_l712_712951

open Real

theorem max_possible_percent_error_in_garden_area :
  ∃ (error_max : ℝ), error_max = 21 :=
by
  -- Given conditions
  let accurate_diameter := 30
  let max_error_percent := 10

  -- Defining lower and upper bounds for the diameter
  let lower_diameter := accurate_diameter - accurate_diameter * (max_error_percent / 100)
  let upper_diameter := accurate_diameter + accurate_diameter * (max_error_percent / 100)

  -- Calculating the exact and potential extreme areas
  let exact_area := π * (accurate_diameter / 2) ^ 2
  let lower_area := π * (lower_diameter / 2) ^ 2
  let upper_area := π * (upper_diameter / 2) ^ 2

  -- Calculating the percent errors
  let lower_error_percent := ((exact_area - lower_area) / exact_area) * 100
  let upper_error_percent := ((upper_area - exact_area) / exact_area) * 100

  -- We need to show the maximum error is 21%
  use upper_error_percent -- which should be 21% according to the problem statement
  sorry -- proof goes here

end max_possible_percent_error_in_garden_area_l712_712951


namespace number_of_integers_between_sqrts_l712_712442

theorem number_of_integers_between_sqrts :
  let lower_bound := Real.sqrt 10
  let upper_bound := Real.sqrt 75
  let lower_int := Int.ceil lower_bound
  let upper_int := Int.floor upper_bound
  ∃ (count : ℕ), count = upper_int - lower_int + 1 ∧ count = 5 :=
by
  let lower_bound := Real.sqrt 10
  let upper_bound := Real.sqrt 75
  let lower_int := Int.ceil lower_bound
  let upper_int := Int.floor upper_bound
  use upper_int - lower_int + 1
  split
  · sorry
  · sorry

end number_of_integers_between_sqrts_l712_712442


namespace river_width_l712_712698

theorem river_width
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) (flow_rate_m_per_min : ℝ)
  (H_depth : depth = 5)
  (H_flow_rate_kmph : flow_rate_kmph = 4)
  (H_volume_per_minute : volume_per_minute = 6333.333333333333)
  (H_flow_rate_m_per_min : flow_rate_m_per_min = 66.66666666666667) :
  volume_per_minute / (depth * flow_rate_m_per_min) = 19 :=
by
  -- proof goes here
  sorry

end river_width_l712_712698


namespace obtuse_triangle_ratio_l712_712020

theorem obtuse_triangle_ratio :
  let lengths := [1, 2, 3, 4, 5]
  let n := Nat.choose 5 3
  let obtuse_tris := (lengths.combinations 3).filter (λ triplet =>
    let a := triplet.nthLe 0 sorry
    let b := triplet.nthLe 1 sorry
    let c := triplet.nthLe 2 sorry
    (c * c > a * a + b * b)
  )
  let m := obtuse_tris.length
  m / n = 1 / 5 := by
  let lengths := [1, 2, 3, 4, 5]
  let n := Nat.choose 5 3
  let obtuse_tris := lengths.combinations 3
    |>.filter (λ triplet =>
      match triplet.sorted with
      | [a, b, c] => c^2 > a^2 + b^2
      | _ => False
    )
  let m := obtuse_tris.length
  have : m = 2 := sorry  -- Obtuse triangles are {2, 3, 4} and {2, 4, 5}
  have n_val : n = Nat.choose 5 3 := rfl
  have calc_ratio : m / n = 2 / 10 := sorry
  show 2 / 10 = 1 / 5, from rfl

end obtuse_triangle_ratio_l712_712020


namespace min_value_of_f_when_a_max_l712_712024

noncomputable def f (a x : ℝ) : ℝ := (1 / 4) * a * x^4 - (1 / 2) * x^2

-- Informal statement: Prove that if there exists t ∈ ℝ such that |f'(t+2) - f'(t)| ≤ 1/4,
-- the minimum value of f(x) for the maximum value of a is -2/9.
theorem min_value_of_f_when_a_max (a t : ℝ)
  (h : ∃ t : ℝ, |((a * (t + 2)^3 - (t + 2)) - (a * t^3 - t))| ≤ (1 / 4)) :
  is_max a (9 / 8) → ∃ x : ℝ, ∃ a : ℝ, f a x = -(2 / 9) :=
sorry

end min_value_of_f_when_a_max_l712_712024


namespace minimum_distance_to_recover_cost_l712_712672

theorem minimum_distance_to_recover_cost 
  (initial_consumption : ℝ) (modification_cost : ℝ) (modified_consumption : ℝ) (gas_cost : ℝ) : 
  22000 < (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 ∧ 
  (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 < 26000 :=
by
  let initial_consumption := 8.4
  let modified_consumption := 6.3
  let modification_cost := 400.0
  let gas_cost := 0.80
  sorry

end minimum_distance_to_recover_cost_l712_712672


namespace probability_all_white_balls_l712_712275

def totalBalls : ℕ := 15
def whiteBalls : ℕ := 8
def blackBalls : ℕ := 7
def drawnBalls : ℕ := 7

theorem probability_all_white_balls :
  (nat.choose whiteBalls drawnBalls : ℚ) / (nat.choose totalBalls drawnBalls : ℚ) = 8 / 6435 := sorry

end probability_all_white_balls_l712_712275


namespace range_of_a_l712_712592

def f (a : ℝ) (x : ℝ) : ℝ := log a (abs (a * x^2 - x))

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 ∈ Set.Icc (3 : ℝ) 4, x1 < x2 → f a x1 < f a x2) ↔ (1 < a ∨ (1/6 ≤ a ∧ a < 1/4)) :=
by 
  sorry

end range_of_a_l712_712592


namespace minimum_candies_l712_712675


theorem minimum_candies (students : ℕ) (N : ℕ) (k : ℕ) :
  students = 25 →
  (∀ s1 s2 : ℕ, (s1 < s2 ↔ ∃ n1 n2 : ℕ, n1 ⬝ students = N ∧
                  n2 ⬝ students = N ∧ n1 > n2 ∧ s1 > s2)) →
  ∃ N, N = 600 :=
by
  intros h_students h_condition
  use 600 -- Constructively showing that N = 600 satisfies conditions
  sorry  -- Proof of satisfaction by conditions

end minimum_candies_l712_712675


namespace P_at_2020_l712_712260

noncomputable def P : ℝ → ℝ := sorry  -- Define polynomial P(x)

axiom leading_coeff_P : leading_coeff P = 1
axiom degree_P : degree P = 10
axiom P_positive : ∀ x : ℝ, P x > 0
axiom neg_P_factors : ∃ factors : ℕ → (ℝ → ℝ), 
  ∀ i : ℕ, i < 5 → (factors i).irreducible ∧ (factors i) 2020 = -3 ∧ 
  ∏ i in finset.range 5, factors i = -P

theorem P_at_2020 : P 2020 = 243 :=
  sorry

end P_at_2020_l712_712260


namespace domain_of_f_l712_712583

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - 2 * real.cos x)

theorem domain_of_f :
  {x : ℝ | ∃ k : ℤ, (π / 3 + 2 * k * π ≤ x ∧ x ≤ 5 * π / 3 + 2 * k * π)} =
  {x : ℝ | 1 - 2 * real.cos x ≥ 0} :=
sorry

end domain_of_f_l712_712583


namespace sequence_sum_inequality_l712_712768

-- Define the sequence {a_n} with the given recurrence relation
def a : ℕ → ℝ
| 0       := 0   -- We index from 1, so index 0 is a dummy
| 1       := 1
| (n + 1) := 1 + 2 / a n

-- State the theorem to prove the desired inequality
theorem sequence_sum_inequality (m : ℕ) (hm : 0 < m) : 
  (∑ i in Finset.range (2 * m + 1).succ, a i) > 4 * m - (1/4) := sorry

end sequence_sum_inequality_l712_712768


namespace find_ab_l712_712832

theorem find_ab
  (a b c : ℝ)
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 27)
  (h3 : a + b + c = 10)
  (h4 : a^3 - b^3 = 36)
  : a * b = -15 :=
by
  sorry

end find_ab_l712_712832


namespace find_a_plus_b_l712_712788

noncomputable def f (x : ℝ) : ℝ := sorry

def satisfies_conditions (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x, f(x) = f(4 - x)) ∧
  (∀ x, f(x + 1) = -f(x + 3)) ∧
  (∀ x ∈ Icc 0 4, f(x) = abs(x - a) + b)

theorem find_a_plus_b (f : ℝ → ℝ) (a b : ℝ) (h : satisfies_conditions f a b) :
  a + b = 1 := sorry

end find_a_plus_b_l712_712788


namespace arithmetic_geometric_sequence_l712_712779

noncomputable def a (d : ℝ) : ℝ := 6 + d
noncomputable def b (d : ℝ) : ℝ := 6 + 2 * d
noncomputable def c (r : ℝ) : ℝ := 6 * r
noncomputable def d (r : ℝ) : ℝ := 6 * r ^ 2

theorem arithmetic_geometric_sequence :
  (∃ d r : ℝ, 48 = 6 + 3 * d ∧ 48 = 6 * r ^ 3) →
  (∃ d r : ℝ, a d + b d + c r + d r = 90) :=
by
  intro h
  obtain ⟨d, r, ha, hb⟩ := h
  sorry

end arithmetic_geometric_sequence_l712_712779


namespace angle_C_is_pi_over_3_max_perimeter_l712_712105

variable (C A B a b c : ℝ)

-- Given conditions
def conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  c = sqrt 6 ∧ cos (2 * C) - 3 * cos (A + B) = 1

-- To prove that angle C is π / 3
theorem angle_C_is_pi_over_3 (a b c A B C : ℝ) (h : conditions a b c A B C) : 
  C = π / 3 := sorry

-- To prove that the maximum perimeter of the triangle is 3 * sqrt 6 given c = sqrt 6
theorem max_perimeter (a b c A B C : ℝ) (h : conditions a b c A B C) : 
  let perimeter := a + b + c in
  perimeter = 3 * sqrt 6 := sorry

end angle_C_is_pi_over_3_max_perimeter_l712_712105


namespace vanessa_scored_27_points_l712_712630

variable (P : ℕ) (number_of_players : ℕ) (average_points_per_player : ℚ) (vanessa_points : ℕ)

axiom team_total_points : P = 48
axiom other_players : number_of_players = 6
axiom average_points_per_other_player : average_points_per_player = 3.5

theorem vanessa_scored_27_points 
  (h1 : P = 48)
  (h2 : number_of_players = 6)
  (h3 : average_points_per_player = 3.5)
: vanessa_points = 27 :=
sorry

end vanessa_scored_27_points_l712_712630


namespace parallel_lines_l712_712049

open EuclideanGeometry

variables {Point Line Plane : Type}
variable [incidence_geometry Point Line Plane]

-- Define the conditions
variables (m n : Line) (α : Plane)
-- Assume m is perpendicular to α and n is perpendicular to α
variable (h1 : m ⟂ α) (h2 : n ⟂ α)

-- Proposition to prove
theorem parallel_lines (h1 : m ⟂ α) (h2 : n ⟂ α) : m ∥ n := by
  sorry

end parallel_lines_l712_712049


namespace function_type_l712_712421

theorem function_type (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x < y → (b * x + a) < (b * y + a)) ∧ (b * 0 + a > 0) :=
by
  split
  { intros x y hxy
    calc
      b * x + a < b * y + a := sorry
  }
  { exact sorry
  }

end function_type_l712_712421


namespace smallest_number_approx_l712_712614

-- Given conditions as definitions
def condition1 (x y : ℝ) : Prop := 3 * x - y = 20
def condition2 (y z : ℝ) : Prop := 2 * z = 3 * y
def condition3 (x y z : ℝ) : Prop := x + y + z = 48

-- The main theorem statement
theorem smallest_number_approx (x y z : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 y z) 
  (h3 : condition3 x y z) : 
  x ≈ 11.53 ∧ (x < y ∧ x < z) := 
sorry

end smallest_number_approx_l712_712614


namespace reach_any_composite_from_4_l712_712866

/-- 
Prove that starting from the number \( 4 \), it is possible to reach any given composite number 
through repeatedly adding one of its divisors, different from itself and one. 
-/
theorem reach_any_composite_from_4:
  ∀ n : ℕ, Prime (n) → n ≥ 4 → (∃ k d : ℕ, d ∣ k ∧ k = k + d ∧ k = n) := 
by 
  sorry


end reach_any_composite_from_4_l712_712866


namespace log_a_b_l712_712100

theorem log_a_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_cond1 : 1 / a + 1 / b ≤ 2 * Real.sqrt 2)
  (h_cond2 : (a - b) ^ 2 = 4 * (a * b) ^ 3) : 
  Real.log a b = -1 := 
sorry

end log_a_b_l712_712100


namespace part1_part2_l712_712851

-- Part 1: Prove that x_0^2 = x_1 * x_2 given the conditions
theorem part1
  (x0 α t x1 x2 y1 y2 p : ℝ) (h1: y1^2 = 2 * p * x1) (h2: y2^2 = 2 * p * x2)
  (h3: x1 = x0 + t * cos α) (h4: x2 = x0 + t * cos α) (p_pos: 0 < p):
  x0^2 = x1 * x2 :=
sorry

-- Part 2: Prove that if OA ⊥ OB then x_0 = 2p
theorem part2
  (x0 α t x1 x2 y1 y2 p : ℝ) (h1: y1^2 = 2 * p * x1) (h2: y2^2 = 2 * p * x2)
  (h3: x1 = x0 + t * cos α) (h4: x2 = x0 + t * cos α) (h5: x1 * x2 + y1 * y2 = 0)
  (h6: y1 = t * sin α) (h7: y2 = t * sin α) (p_pos: 0 < p):
  x0 = 2 * p :=
sorry

end part1_part2_l712_712851


namespace scale_division_l712_712256

theorem scale_division (ft : ℕ) (inches : ℕ) (parts : ℕ) (total_inches : ℕ) :
  ft = 7 → inches = 6 → parts = 5 → 
  total_inches = (ft * 12 + inches) → 
  (total_inches / parts) = 18 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have total_in: total_inches = 84 + 6 := by simp [h4]
  simp [total_in]
  have div := (90 / 5)
  show div = 18
  exact rfl

end scale_division_l712_712256


namespace quadratic_roots_are_correct_l712_712569

theorem quadratic_roots_are_correct (x: ℝ) : 
    (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2) ∨ (x = (-1 - Real.sqrt 5) / 2) := 
by sorry

end quadratic_roots_are_correct_l712_712569


namespace total_marks_math_physics_l712_712701

variable (M P C : ℝ)

-- Conditions
def condition1 : Prop := C = P + 20
def condition2 : Prop := (M + C) / 2 = 40

-- Target statement to prove
theorem total_marks_math_physics (h1 : condition1 M P C) (h2 : condition2 M P C) :
  M + P = 60 :=
sorry

end total_marks_math_physics_l712_712701


namespace sum_of_relatively_prime_integers_l712_712180

theorem sum_of_relatively_prime_integers (n : ℕ) (h : n ≥ 7) :
  ∃ a b : ℕ, n = a + b ∧ a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 :=
by
  sorry

end sum_of_relatively_prime_integers_l712_712180


namespace tangent_line_at_point_g_monotonic_intervals_f_less_than_three_halves_g_l712_712036

noncomputable def f (x : ℝ) : ℝ := 4 * sqrt x - x^2
noncomputable def g (x : ℝ) : ℝ := exp x + exp (-x)

theorem tangent_line_at_point :
  let P := (1 : ℝ, f 1) in
  P.2 = 3 :=
by 
  sorry
  
theorem g_monotonic_intervals :
  (∀ x : ℝ, x < 0 → g' x < 0) ∧ (∀ x : ℝ, x > 0 → g' x > 0) :=
by 
  sorry

theorem f_less_than_three_halves_g (x : ℝ) (h : x ≥ 0) :
  f x < (3 / 2) * g x :=
by 
  sorry

end tangent_line_at_point_g_monotonic_intervals_f_less_than_three_halves_g_l712_712036


namespace conclusion_2_conclusion_3_conclusion_4_l712_712558

variable (b : ℝ)

def f (x : ℝ) : ℝ := x^2 - |b| * x - 3

theorem conclusion_2 (h_min : ∃ x, f b x = -3) : b = 0 :=
  sorry

theorem conclusion_3 (h_b : b = -2) (x : ℝ) (hx : -2 < x ∧ x < 2) :
    -4 ≤ f b x ∧ f b x ≤ -3 :=
  sorry

theorem conclusion_4 (hb_ne : b ≠ 0) (m : ℝ) (h_roots : ∃ x1 x2, f b x1 = m ∧ f b x2 = m ∧ x1 ≠ x2) :
    m > -3 ∨ b^2 = -4 * m - 12 :=
  sorry

end conclusion_2_conclusion_3_conclusion_4_l712_712558


namespace smallest_sum_of_squares_l712_712996

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l712_712996


namespace distinct_positive_integers_l712_712518

theorem distinct_positive_integers (a b : ℕ) (h₁ : a > 10^6) (h₂ : b > 10^6) (h₃ : a ≠ b) (h₄ : (a + b)^3 % (a * b) = 0) :
  abs (a - b) > 10^4 := 
sorry

end distinct_positive_integers_l712_712518


namespace double_point_quadratic_l712_712092

theorem double_point_quadratic (m x1 x2 : ℝ) 
  (H1 : x1 < 1) (H2 : 1 < x2)
  (H3 : ∃ (y1 y2 : ℝ), y1 = 2 * x1 ∧ y2 = 2 * x2 ∧ y1 = x1^2 + 2 * m * x1 - m ∧ y2 = x2^2 + 2 * m * x2 - m)
  : m < 1 :=
sorry

end double_point_quadratic_l712_712092


namespace vector_parallel_iff_norm_proposition_p_true_proposition_q_false_logical_statements_l712_712026

-- Definitions of vectors and sets.
def a (m : ℝ) : ℝ × ℝ := (m, 3 * m)
def b (m : ℝ) : ℝ × ℝ := (m + 1, 6)
def A (m : ℝ) : Set ℝ := {x : ℝ | (x - m^2) * (x + m - 2) = 0}

-- Propositions.
def p (m : ℝ) : Prop := (a m).fst * (b m).fst + (a m).snd * (b m).snd = 0 → m = -19
def q (m : ℝ) : Prop := (2 : ℝ) ^ (card (A m)) = 2 → m = 1

-- Proofs of equivalences and logical statements.
theorem vector_parallel_iff_norm : (∀ m ≠ 0, (a m).fst * (b m).snd = (a m).snd * (b m).fst ↔ |a m| = √10) :=
sorry

theorem proposition_p_true : ∀ m ≠ 0, p m :=
sorry

theorem proposition_q_false : ∀ m ≠ 0, ¬q m :=
sorry

theorem logical_statements : ∀ m ≠ 0, (p m ∨ q m) ∧ (¬(p m ∧ q m)) ∧ (¬q m) :=
sorry

end vector_parallel_iff_norm_proposition_p_true_proposition_q_false_logical_statements_l712_712026


namespace cos_midpoint_zero_l712_712591

noncomputable def f : ℝ → ℝ := cos

theorem cos_midpoint_zero {a b : ℝ} (h_inc : ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) 
                         (h_a : f a = -1) (h_b : f b = 1) : cos ((a + b) / 2) = 0 :=
by
  sorry

end cos_midpoint_zero_l712_712591


namespace expenditure_ratio_is_two_l712_712287

-- Define the conditions as constants
constant I : ℝ
constant savings_rate : ℝ := 0.30
constant income_increase_rate : ℝ := 0.30
constant savings_increase_rate : ℝ := 2 -- which corresponds to 100%

-- Define the first year conditions
def first_year_savings := savings_rate * I
def first_year_expenditure := (1 - savings_rate) * I

-- Define the second year conditions based on the given rates
def second_year_income := (1 + income_increase_rate) * I
def second_year_savings := savings_increase_rate * first_year_savings
def second_year_expenditure := second_year_income - second_year_savings

-- Define the total expenditure over two years
def total_expenditure_two_years := first_year_expenditure + second_year_expenditure

-- Define the ratio
def expenditure_ratio := total_expenditure_two_years / first_year_expenditure

-- The proposition we need to prove
theorem expenditure_ratio_is_two : expenditure_ratio = 2 :=
by
  sorry

end expenditure_ratio_is_two_l712_712287


namespace relationship_among_a_b_c_l712_712153

noncomputable def a : ℝ := 0.6^0.6
noncomputable def b : ℝ := 0.6^1.5
noncomputable def c : ℝ := 1.5^0.6

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l712_712153


namespace min_value_AN_plus_2BM_l712_712040

open Real

def point_on_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 4
def not_on_axes (x y : ℝ) : Prop := x ≠ 0 ∧ y ≠ 0
def point_A : ℝ × ℝ := (2, 0)
def point_B : ℝ × ℝ := (0, 2)
def on_y_axis (x y : ℝ) : Prop := y = 0
def on_x_axis (x y : ℝ) : Prop := x = 0

theorem min_value_AN_plus_2BM (P : ℝ × ℝ) 
  (h1 : point_on_circle P.1 P.2)
  (h2 : not_on_axes P.1 P.2) :
  ∃ M N : ℝ × ℝ, 
    on_y_axis M.1 M.2 ∧ on_x_axis N.1 N.2 ∧ 
    |2 - N.2| + 2 * |2 - M.1| = 8 := 
sorry

end min_value_AN_plus_2BM_l712_712040


namespace find_explicit_formula_and_prove_monotonicity_l712_712804

-- Given function passes through (2, 4)
def power_function (x : ℝ) (α : ℝ) : ℝ := x ^ α

def passes_through (α : ℝ) : Prop := power_function 2 α = 4

-- Function is increasing on (0, +∞)
def is_increasing_on (f : ℝ → ℝ) (S : set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y

theorem find_explicit_formula_and_prove_monotonicity :
  ∃ α : ℝ, passes_through α ∧
  ∃ f : ℝ → ℝ, (∀ x, f x = x ^ α) ∧ is_increasing_on f (set.Ioi 0) := sorry

end find_explicit_formula_and_prove_monotonicity_l712_712804


namespace collinearity_of_points_l712_712913

noncomputable theory
open_locale classical

variables {A P Q M N I T K : Type*}

-- Conditions given in the problem
variables [IncidenceGeometry A P Q M N I T K]
variable [IntersectionPoint T (Line A P) (Line C Q)]
variable [IntersectionPoint K (Line M P) (Line N Q)]

-- Statement of the proof problem
theorem collinearity_of_points :
  Collinear {T, K, I} :=
sorry

end collinearity_of_points_l712_712913


namespace carnival_days_l712_712214

-- Define the given conditions
def total_money := 3168
def daily_income := 144

-- Define the main theorem statement
theorem carnival_days : (total_money / daily_income) = 22 := by
  sorry

end carnival_days_l712_712214


namespace finite_sequences_count_l712_712080

def a : ℕ → ℕ
| 0 := 1
| 1 := 1
| n := a (n - 1) + a (n - 2)

theorem finite_sequences_count :
    a 10 = 89 :=
sorry

end finite_sequences_count_l712_712080


namespace double_point_quadratic_l712_712091

theorem double_point_quadratic (m x1 x2 : ℝ) 
  (H1 : x1 < 1) (H2 : 1 < x2)
  (H3 : ∃ (y1 y2 : ℝ), y1 = 2 * x1 ∧ y2 = 2 * x2 ∧ y1 = x1^2 + 2 * m * x1 - m ∧ y2 = x2^2 + 2 * m * x2 - m)
  : m < 1 :=
sorry

end double_point_quadratic_l712_712091


namespace arithmetic_mean_difference_l712_712654

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
sorry

end arithmetic_mean_difference_l712_712654


namespace max_modulus_l712_712047

theorem max_modulus (z : ℂ) (h : |z - 2 * complex.I| = 1) : |z| ≤ 3 :=
begin
  sorry
end

end max_modulus_l712_712047


namespace chocolates_bought_l712_712982

theorem chocolates_bought (C S N : ℕ) (h1 : 4 * C = 7 * (S - C)) (h2 : N * C = 77 * S) :
  N = 121 :=
by
  sorry

end chocolates_bought_l712_712982


namespace sum_even_integers_100_to_200_l712_712217

theorem sum_even_integers_100_to_200 :
  (∑ k in (finset.range 51).map (λ n, 100 + 2 * n) id) = 7650 :=
by {
  sorry
}

end sum_even_integers_100_to_200_l712_712217


namespace collinearity_of_T_K_I_l712_712929

noncomputable def intersection_point (l1 l2 : Line) : Point := sorry

-- Definitions of lines AP, CQ, MP, NQ based on the problem context
variables {A P C Q M N I : Point} (lAP lCQ lMP lNQ : Line)
variables (T : Point) (K : Point)

-- Given conditions
def condition_1 : T = intersection_point lAP lCQ := sorry
def condition_2 : K = intersection_point lMP lNQ := sorry

-- Theorem statement
theorem collinearity_of_T_K_I : T ∈ line_through K I :=
by {
  -- These are the conditions that we're given in the problem
  have hT : T = intersection_point lAP lCQ := sorry,
  have hK : K = intersection_point lMP lNQ := sorry,
  -- Rest of the proof would go here
  sorry
}

end collinearity_of_T_K_I_l712_712929


namespace collinear_T_K_I_l712_712887

noncomputable def intersection (P Q : Set Point) : Point := sorry

variables (A P C Q M N I T K : Point)

-- Definitions based on conditions
def T_def : Point := intersection (line_through A P) (line_through C Q)
def K_def : Point := intersection (line_through M P) (line_through N Q)

-- Proof statement
theorem collinear_T_K_I :
  collinear ({T_def A P C Q, K_def M P N Q, I} : Set Point) := sorry

end collinear_T_K_I_l712_712887


namespace graveling_cost_correct_l712_712650

-- Define the dimensions of the rectangular lawn
def lawn_length : ℕ := 80 -- in meters
def lawn_breadth : ℕ := 50 -- in meters

-- Define the width of each road
def road_width : ℕ := 10 -- in meters

-- Define the cost per square meter for graveling the roads
def cost_per_sq_m : ℕ := 3 -- in Rs. per sq meter

-- Define the area of the road parallel to the length of the lawn
def area_road_parallel_length : ℕ := lawn_length * road_width

-- Define the effective length of the road parallel to the breadth of the lawn
def effective_road_parallel_breadth_length : ℕ := lawn_breadth - road_width

-- Define the area of the road parallel to the breadth of the lawn
def area_road_parallel_breadth : ℕ := effective_road_parallel_breadth_length * road_width

-- Define the total area to be graveled
def total_area_to_be_graveled : ℕ := area_road_parallel_length + area_road_parallel_breadth

-- Define the total cost of graveling
def total_graveling_cost : ℕ := total_area_to_be_graveled * cost_per_sq_m

-- Theorem: The total cost of graveling the two roads is Rs. 3600
theorem graveling_cost_correct : total_graveling_cost = 3600 := 
by
  unfold total_graveling_cost total_area_to_be_graveled area_road_parallel_length area_road_parallel_breadth effective_road_parallel_breadth_length lawn_length lawn_breadth road_width cost_per_sq_m
  exact rfl

end graveling_cost_correct_l712_712650


namespace solve_equation1_solve_equation2_l712_712187

theorem solve_equation1 (x : ℝ) (h1 : 5 * x - 2 * (x - 1) = 3) : x = 1 / 3 := 
sorry

theorem solve_equation2 (x : ℝ) (h2 : (x + 3) / 2 - 1 = (2 * x - 1) / 3) : x = 5 :=
sorry

end solve_equation1_solve_equation2_l712_712187


namespace concurrency_of_AD_BQ_PC_l712_712143

-- Definitions related to the problem
variables {A B C P Q D : Point}
variables [triangle : IsTriangle A B C]
variables [obtuse : IsObtuseTriangle A B C]
variables [largest_side : AB > AC ∧ AB > BC]
variables [angle_bisector : ∃ l, IsAngleBisector l (∠BAC)]
variables [perpendicular_B : IsPerpendicularFromTo B l P]
variables [perpendicular_C : IsPerpendicularFromTo C l Q]
variables [D_on_BC : ∃ D, IsOnLine D BC ∧ AD ⊥ AP]

-- The theorem to prove
theorem concurrency_of_AD_BQ_PC
  (hTriangle : IsTriangle A B C)
  (hObtuse : IsObtuseTriangle A B C)
  (hLargestSide : AB > AC ∧ AB > BC)
  (hAngleBisector : ∃ l, IsAngleBisector l (∠BAC))
  (hPerpendicularB : IsPerpendicularFromTo B l P)
  (hPerpendicularC : IsPerpendicularFromTo C l Q)
  (hDOnBC : ∃ D, IsOnLine D BC ∧ AD ⊥ AP) :
  ConcurrentLines AD BQ PC := 
sorry

end concurrency_of_AD_BQ_PC_l712_712143


namespace simplify_trig_expression_l712_712969

theorem simplify_trig_expression : 
  sqrt (1 - 2 * (sin (π - 2)) * (cos (π - 2))) = sin 2 + cos 2 :=
by
  -- Mathematical proof steps, which are skipped for this statement
  sorry

end simplify_trig_expression_l712_712969


namespace solve_2019_gon_l712_712481

noncomputable def problem_2019_gon (x : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, (x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) + x (i+5) + x (i+6) + x (i+7) + x (i+8) = 300))
  ∧ (x 18 = 19)
  ∧ (x 19 = 20)

theorem solve_2019_gon :
  ∀ x : ℕ → ℕ,
  problem_2019_gon x →
  x 2018 = 61 :=
by sorry

end solve_2019_gon_l712_712481


namespace f_iterated_l712_712361

noncomputable def f (z : ℂ) : ℂ :=
  if (∃ r : ℝ, z = r) then -z^2 else z^2

theorem f_iterated (z : ℂ) (h : z = 2 + 1 * complex.i) : 
  f (f (f (f z))) = 164833 + 354192 * complex.i :=
by 
  subst h
  sorry

end f_iterated_l712_712361


namespace num_integers_between_roots_l712_712440

theorem num_integers_between_roots :
  ∃ n : ℕ, n = 5 ∧ ∀ x : ℝ, (sqrt 10 < x ∧ x < sqrt 75 → ∃ y : ℤ, x < y ∧ y < x + 1) :=
sorry

end num_integers_between_roots_l712_712440


namespace berries_problem_contradiction_l712_712241

def berries_problem_statement : Prop :=
  ∃ (B S G : ℕ), 
    (S - B = 12) ∧ 
    (S - G = 25) ∧ 
    (2 * S - (B + G) = 113)

theorem berries_problem_contradiction : ¬ berries_problem_statement := 
  by {
    intro h,
    obtain ⟨B, S, G, h₁, h₂, h₃⟩ := h,
    have h_add : (S - B) + (S - G) = 37,
    { 
      rw [h₁, h₂], 
      exact add_eq_of_eq_sub _ 12 25,
    },
    have h_contra : 2 * S - (B + G) = 37,
    { 
      rw h_add,
    },
    rw h₃ at h_contra,
    contradiction,
  }

end berries_problem_contradiction_l712_712241


namespace interval_contains_zeros_l712_712071

-- Define the conditions and the function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c 

theorem interval_contains_zeros (a b c : ℝ) (h1 : 2 * a + c / 2 > b) (h2 : c < 0) : 
  ∃ x ∈ Set.Ioc (-2 : ℝ) 0, quadratic a b c x = 0 :=
by
  -- Problem Statement: given conditions, interval (-2, 0) contains a zero
  sorry

end interval_contains_zeros_l712_712071


namespace collinear_TKI_l712_712917

-- Definitions based on conditions
variables {A P Q C M N T K I : Type}
variables (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)

-- Definitions to represent the intersection points 
def T_def (A P Q C : Type) (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) : Prop := 
  ∃ T : Type, line_AP A T ∧ line_CQ C T

def K_def (M P N Q : Type) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop) : Prop := 
  ∃ K : Type, line_MP M K ∧ line_NQ N K

-- Theorem statement
theorem collinear_TKI (A P Q C M N T K I : Type)
  (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)
  (hT : T_def A P Q C line_AP line_CQ) 
  (hK : K_def M P N Q line_MP line_NQ) : 
  collinear T K I :=
sorry

end collinear_TKI_l712_712917


namespace find_numbers_l712_712188

theorem find_numbers :
  ∃ (a b c d : ℕ), 
  (a + 2 = 22) ∧ 
  (b - 2 = 22) ∧ 
  (c * 2 = 22) ∧ 
  (d / 2 = 22) ∧ 
  (a + b + c + d = 99) :=
sorry

end find_numbers_l712_712188


namespace multiply_of_Mari_buttons_l712_712876

-- Define the variables and constants from the problem
def Mari_buttons : ℕ := 8
def Sue_buttons : ℕ := 22
def Kendra_buttons : ℕ := 2 * Sue_buttons

-- Statement that we need to prove
theorem multiply_of_Mari_buttons : ∃ (x : ℕ), Kendra_buttons = 8 * x + 4 ∧ x = 5 := by
  sorry

end multiply_of_Mari_buttons_l712_712876


namespace proof_time_to_change_oil_per_car_is_15_l712_712126

/-- Problem statement: Calculating the time required to change oil on one car --/

def time_to_change_oil_per_car : ℕ :=
  let time_to_wash_car := 10
  let time_to_change_tires := 30
  let cars_washed := 9
  let cars_with_oil_changed := 6
  let tire_sets_changed := 2
  let total_work_time := 4 * 60 -- total work time converted to minutes
  let total_washing_time := cars_washed * time_to_wash_car
  let total_tires_change_time := tire_sets_changed * time_to_change_tires
  let total_oil_change_time := total_work_time - total_washing_time - total_tires_change_time
  total_oil_change_time / cars_with_oil_changed

theorem proof_time_to_change_oil_per_car_is_15 :
  time_to_change_oil_per_car = 15 :=
by
  -- All conditions and their applications in the computation
  let time_to_wash_car := 10
  let time_to_change_tires := 30
  let cars_washed := 9
  let cars_with_oil_changed := 6
  let tire_sets_changed := 2
  let total_work_time := 4 * 60 -- total work time in minutes
  calc
    let total_washing_time := cars_washed * time_to_wash_car
    let total_tires_change_time := tire_sets_changed * time_to_change_tires
    let total_oil_change_time := total_work_time - total_washing_time - total_tires_change_time
    let time_per_oil_change := total_oil_change_time / cars_with_oil_changed
    time_per_oil_change eq 15


end proof_time_to_change_oil_per_car_is_15_l712_712126


namespace measure_angle_WYZ_l712_712525

def angle_XYZ : ℝ := 45
def angle_XYW : ℝ := 15

theorem measure_angle_WYZ : angle_XYZ - angle_XYW = 30 := by
  sorry

end measure_angle_WYZ_l712_712525


namespace collinearity_of_T_K_I_l712_712900

-- Definitions of the points and lines
variables {A P Q M N C I T K : Type} [Nonempty A] [Nonempty P] [Nonempty Q] 
  [Nonempty M] [Nonempty N] [Nonempty C] [Nonempty I]

-- Intersection points conditions
def intersect (l₁ l₂ : Type) : Type := sorry

-- Given conditions
def condition_1 : T = intersect (line A P) (line C Q) := sorry
def condition_2 : K = intersect (line M P) (line N Q) := sorry

-- Proof that T, K, and I are collinear
theorem collinearity_of_T_K_I : collinear {T, K, I} := by
  have h1 : T = intersect (line A P) (line C Q) := condition_1
  have h2 : K = intersect (line M P) (line N Q) := condition_2
  -- Further steps needed to prove collinearity
  sorry

end collinearity_of_T_K_I_l712_712900


namespace seating_arrangements_l712_712221

/-- There are 7 seats on a long bench, and 4 people are to be seated such that 
exactly 2 of the 3 empty seats are adjacent. -/
theorem seating_arrangements : 
  let seats := 7
  let people := 4
  let empty_seats := seats - people
  in (∃ adj_empty_seats: ℕ, adj_empty_seats = 2 ∧ empty_seats - adj_empty_seats = 1) 
     ∧ fintype.card (finset.perms_of_multiset (finset.range seats).val) = 480 := 
by
  sorry

end seating_arrangements_l712_712221


namespace abc_relationship_l712_712759

noncomputable def a : ℝ := Real.log 5 - Real.log 3
noncomputable def b : ℝ := (2/5) * Real.exp (2/3)
noncomputable def c : ℝ := 2/3

theorem abc_relationship : b > c ∧ c > a :=
by
  sorry

end abc_relationship_l712_712759


namespace solution_statement_l712_712298

-- Define the set of courses
inductive Course
| Physics | Chemistry | Literature | History | Philosophy | Psychology

open Course

-- Define the condition that a valid program must include Physics and at least one of Chemistry or Literature
def valid_program (program : Finset Course) : Prop :=
  Course.Physics ∈ program ∧
  (Course.Chemistry ∈ program ∨ Course.Literature ∈ program)

-- Define the problem statement
def problem_statement : Prop :=
  ∃ programs : Finset (Finset Course),
    programs.card = 9 ∧ ∀ program ∈ programs, program.card = 5 ∧ valid_program program

theorem solution_statement : problem_statement := sorry

end solution_statement_l712_712298


namespace probability_single_draw_probability_single_win_3_draws_maximize_probability_winning_exactly_once_l712_712840

theorem probability_single_draw (n : ℕ) (hn : n ≥ 2) : 
  (2 * (n * (n - 1) / 2 + 1) / ((n + 2) * (n + 1) / 2) = (n^2 - n + 2) / (n^2 + 3n + 2)) := 
sorry

theorem probability_single_win_3_draws (P : ℚ) (hP : P = 2 / 5) : 
  (3 * P * (1 - P)^2 = 54 / 125) := 
sorry

theorem maximize_probability_winning_exactly_once (n : ℕ) (hn : n ≥ 2) (hp : (n^2 - n + 2) / (n^2 + 3n + 2) = 1 / 3) : 
  n = 2 := 
sorry

end probability_single_draw_probability_single_win_3_draws_maximize_probability_winning_exactly_once_l712_712840


namespace total_books_calculation_l712_712684

-- Definitions and conditions
variable (T : ℝ) -- Total number of books
variable (children_books : ℝ) := 0.35 * T -- 35% of the books are for children
variable (adult_books : ℝ) := 104 -- There are 104 books for adults

-- The proof problem: Prove that T = 160 given the conditions.
theorem total_books_calculation (h : 0.65 * T = 104) : T = 160 := 
by 
  -- Need to prove T = 160
  sorry

end total_books_calculation_l712_712684


namespace distinct_collections_in_bag_l712_712957

theorem distinct_collections_in_bag :
  let word := "MATHEMATICIANS"
  let vowels := ['A', 'A', 'I', 'I', 'I']
  let consonants := ['M', 'M', 'T', 'T', 'C', 'S', 'N', 'H']
  (count_vowel_collections vowels * count_consonant_collections consonants) = 168 := by
  sorry

def count_vowel_collections (vowels : List Char) : Nat :=
  -- Counting logic for vowels
  sorry

def count_consonant_collections (consonants : List Char) : Nat :=
  -- Counting logic for consonants
  sorry

end distinct_collections_in_bag_l712_712957


namespace min_students_wearing_both_glasses_and_watches_l712_712474

theorem min_students_wearing_both_glasses_and_watches
  (n : ℕ)
  (H_glasses : n * 3 / 5 = 18)
  (H_watches : n * 5 / 6 = 25)
  (H_neither : n * 1 / 10 = 3):
  ∃ (x : ℕ), x = 16 := 
by
  sorry

end min_students_wearing_both_glasses_and_watches_l712_712474


namespace tapB_fill_time_l712_712625

-- Conditions as definitions
def tapA_fill_rate := 1 / 12 -- Tap A fills the cistern in 12 minutes
variable (t : ℝ) -- Time tap B takes to fill the cistern alone
def tapB_fill_rate := 1 / t -- Tap B fills the cistern in t minutes

-- Combined filling for the first 4 minutes
def combined_fill_first4 := 4 * (tapA_fill_rate + tapB_fill_rate)

-- Tap B filling for the next 8 minutes
def tapB_fill_next8 := 8 * tapB_fill_rate

-- Total fill equation
axiom fill_equation : combined_fill_first4 + tapB_fill_next8 = 1

-- The theorem we need to prove
theorem tapB_fill_time : t = 18 :=
by
  sorry

end tapB_fill_time_l712_712625


namespace complement_union_correct_l712_712075

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {2, 4})
variable (hB : B = {3, 4})

theorem complement_union_correct : ((U \ A) ∪ B) = {1, 3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end complement_union_correct_l712_712075


namespace johns_contribution_l712_712831

-- Definitions
variables (A J : ℝ)
axiom h1 : 1.5 * A = 75
axiom h2 : (2 * A + J) / 3 = 75

-- Statement of the proof problem
theorem johns_contribution : J = 125 :=
by
  sorry

end johns_contribution_l712_712831


namespace column_with_at_least_half_zeroes_l712_712845

theorem column_with_at_least_half_zeroes (n : ℕ) (h : n ≥ 2)
  (grid : fin n → fin 6 → ℕ)
  (h01 : ∀ i j, i ≠ j → grid i ≠ grid j)
  (hprod : ∀ i j, ∃ k, ∀ l, grid k l = (grid i l) * (grid j l)) :
  ∃ c : fin 6, ∑ i, if grid i c = 0 then 1 else 0 ≥ n / 2 :=
sorry

end column_with_at_least_half_zeroes_l712_712845


namespace faulty_odometer_correct_mileage_l712_712281

def faulty_reading := 2056
def correct_mileage := 1542

theorem faulty_odometer_correct_mileage (n : ℕ) : n = faulty_reading → (∃ m : ℕ, m = correct_mileage) :=
by
  intro h
  use correct_mileage
  rw h
  sorry

end faulty_odometer_correct_mileage_l712_712281


namespace marcus_baseball_cards_l712_712166

/-- 
Marcus initially has 210.0 baseball cards.
Carter gives Marcus 58.0 more baseball cards.
Prove that Marcus now has 268.0 baseball cards.
-/
theorem marcus_baseball_cards (initial_cards : ℝ) (additional_cards : ℝ) 
  (h_initial : initial_cards = 210.0) (h_additional : additional_cards = 58.0) : 
  initial_cards + additional_cards = 268.0 :=
  by
    sorry

end marcus_baseball_cards_l712_712166


namespace parameterizes_line_l712_712209

-- Define the line equation as a condition
def line_equation (x y : ℝ) : Prop := y = (3 / 2) * x - 25

-- Define the parameterized form
def param_form (f : ℝ → ℝ) (t : ℝ) : ℝ × ℝ := (f(t), 15 * t - 7)

-- Define the function f(t)
def f (t : ℝ) : ℝ := 10 * t + 12

-- We need to prove that for all t, (f(t), 15t - 7) lies on the line y = (3 / 2) x - 25
theorem parameterizes_line : ∀ t : ℝ, line_equation (f t) (15 * t - 7) := by
  intro t
  have : (15 * t - 7) = (3 / 2) * (10 * t + 12) - 25 := sorry
  exact this

end parameterizes_line_l712_712209


namespace range_of_a_l712_712038

variable (x a : ℝ)

def p := x^2 - 5 * x - 6 ≤ 0
def q := x^2 - 2 * x + 1 - 4 * a^2 ≤ 0

theorem range_of_a (h : ¬ p → ¬ q): a ≥ 5 / 2 := sorry

end range_of_a_l712_712038


namespace max_value_y_l712_712835

noncomputable def max_y (a b c d : ℝ) : ℝ :=
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2

theorem max_value_y {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = 10) : max_y a b c d = 40 := 
  sorry

end max_value_y_l712_712835


namespace Lara_age_in_10_years_l712_712138

theorem Lara_age_in_10_years (current_age: ℕ) (years_ago: ℕ) (years_from_now: ℕ) (age_years_ago: ℕ) (h1: current_age = age_years_ago + years_ago) (h2: age_years_ago = 9) (h3: years_ago = 7) (h4: years_from_now = 10) : current_age + years_from_now = 26 := 
by 
  rw [h2, h3] at h1
  rw [← h1, h4]
  exact rfl

end Lara_age_in_10_years_l712_712138


namespace gallons_of_gas_l712_712511

-- Define the conditions
def mpg : ℕ := 19
def d1 : ℕ := 15
def d2 : ℕ := 6
def d3 : ℕ := 2
def d4 : ℕ := 4
def d5 : ℕ := 11

-- The theorem to prove
theorem gallons_of_gas : (d1 + d2 + d3 + d4 + d5) / mpg = 2 := 
by {
    sorry
}

end gallons_of_gas_l712_712511


namespace num_integers_between_roots_l712_712441

theorem num_integers_between_roots :
  ∃ n : ℕ, n = 5 ∧ ∀ x : ℝ, (sqrt 10 < x ∧ x < sqrt 75 → ∃ y : ℤ, x < y ∧ y < x + 1) :=
sorry

end num_integers_between_roots_l712_712441


namespace polynomial_has_zero_of_given_form_l712_712692

open Complex Polynomial

theorem polynomial_has_zero_of_given_form :
  ∃ (P : Polynomial ℂ), 
    P.degree = 4 ∧ 
    P.leadingCoeff = 1 ∧ 
    (∃ r s : ℤ, P.has_root r ∧ P.has_root s ∧ 
    ∃ α β : ℤ, P = Polynomial.real_coeff_of_degree 4 1 (x - r) (x - s) (x^2 + α * x + β) ∧ 
    (Complex.ofReal (2 : ℚ) + Complex.i * Complex.sqrt (8 : ℚ)) / 3) = 0 :=
begin
  sorry
end

end polynomial_has_zero_of_given_form_l712_712692


namespace number_to_scientific_notation_l712_712494

noncomputable def million := 10^6
noncomputable def number := 44.3 * million
noncomputable def scientific_notation := 4.43 * 10^7

theorem number_to_scientific_notation : number = scientific_notation :=
by
  sorry

end number_to_scientific_notation_l712_712494


namespace angle_sum_in_rectangle_l712_712121

theorem angle_sum_in_rectangle (A B C D M : Point) (h_rect : Rectangle A B C D)
  (h_angle : ∠ B M C + ∠ A M D = 180) : ∠ B C M + ∠ D A M = 90 :=
sorry

end angle_sum_in_rectangle_l712_712121


namespace smallest_sum_of_squares_l712_712999

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l712_712999


namespace smallest_vertical_distance_l712_712206

def f (x : ℝ) : ℝ := abs x
def g (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem smallest_vertical_distance : 
  ∃ x : ℝ, (∀ y : ℝ, abs (f y - g y) ≥ abs (f x - g x)) ∧ abs (f x - g x) = 3 / 4 :=
begin
  sorry,
end

end smallest_vertical_distance_l712_712206


namespace minimum_value_inequality_l712_712533

   noncomputable def min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 5) : ℝ :=
     min (1 / x + 4 / y + 9 / z) (frac.mk 36 5)

   theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 5) : 
     min_value x y z hx hy hz hxyz = (36 / 5) := sorry
   
end minimum_value_inequality_l712_712533


namespace line_intersects_circle_no_center_l712_712059

theorem line_intersects_circle_no_center :
  ∀ (θ t : ℝ),
    let x_circle := -1 + 2 * Real.cos θ,
        y_circle := 3 + 2 * Real.sin θ,
        x_line := 2 * t - 1,
        y_line := 6 * t - 1,
        center := (-1, 3),
        radius := 2,
        line_eq := (3 * (-1) - (3) + 2) / (Real.sqrt (3^2 + (-1)^2)) < radius
    in
    (3 * x_line - y_line + 2 = 0) → 
    (x_circle, y_circle) ≠ center → 
    ∃ t, ∃ θ, (x_line, y_line) = (x_circle, y_circle) := sorry

end line_intersects_circle_no_center_l712_712059


namespace inclination_angle_of_line_l712_712596

theorem inclination_angle_of_line : 
  let line_eq := λ x y : ℝ, x - y = 0 in
  ∃ α : ℝ, 0 ≤ α ∧ α ≤ π ∧ tan α = 1 ∧ α = π / 4 :=
sorry

end inclination_angle_of_line_l712_712596


namespace coin_flip_probability_l712_712452

def coin_heads_probability : ℚ := 1 / 2

def flips := 5

theorem coin_flip_probability :
  (coin_heads_probability * 
   coin_heads_probability *
   (1 - coin_heads_probability) *
   (1 - coin_heads_probability) *
   (1 - coin_heads_probability) = 1 / 32) :=
by
  unfold coin_heads_probability
  norm_num
  sorry

end coin_flip_probability_l712_712452


namespace inequality_solution_l712_712749

-- Define the condition
def y_eq_root4_x (x y : ℝ) : Prop :=
  y = x^(1/4:ℝ)

-- Define the inequality condition in terms of y
def inequality_condition (y : ℝ) : Prop :=
  y^2 + 3 * y + 1 >= 0

-- Define the final problem statement in terms of x
def problem_statement (x : ℝ) : Prop :=
  x < ((-3 - Real.sqrt 5) / 2)^4 ∨ x > ((-3 + Real.sqrt 5) / 2)^4

-- The theorem to be proved
theorem inequality_solution (x : ℝ) (y : ℝ) 
  (h1 : y_eq_root4_x x y) (h2 : inequality_condition y) : 
  problem_statement x :=
begin
  sorry -- Proof goes here
end

end inequality_solution_l712_712749


namespace range_a_for_inequality_l712_712597

theorem range_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, 2 ^ (x ^ 2 - 4 * x) > 2 ^ (2 * a * x + a)) ↔ a ∈ set.Ioo (-4 : ℝ) (-1 : ℝ) :=
sorry

end range_a_for_inequality_l712_712597


namespace area_of_triangle_HGF_l712_712504

variables {A B C H G : Type} 
variable [linear_ordered_field A] 
variables {triangle : Type} 
variable [nonempty triangle] 
variable [measurable_space triangle] 
variable [measure_theory.volume.triangle]

/-- Given a triangle ABC with G as the centroid and H as the midpoint of BF. --/
def centroid_division_of_medians (triangle : G) (G : A) : Prop :=
  -- centroid divides medians into 2:1 ratio
  2 * G = 3 * triangle

def area_division_medians (A B C G : triangle) : Prop :=
  -- Each segment connected to centroid divides area into six equal parts
  measurable_space.triangle * 6

def area_half_of_median_division (G F H : triangle) : Prop :=
  -- H is the midpoint of BF thus HGF is \( \frac{1}{12} \) of ABC
  measurable_space.triangle / 12 

theorem area_of_triangle_HGF (A B C H G : triangle) :
  centroid_division_of_medians triangle G ∧ 
  area_division_medians A B C G ∧
  area_half_of_median_division G F H :=
  begin
    exact_resolve_mem
  end

end area_of_triangle_HGF_l712_712504


namespace weeks_per_mouse_correct_l712_712372

def years_in_decade : ℕ := 10
def weeks_per_year : ℕ := 52
def total_mice : ℕ := 130

def total_weeks_in_decade : ℕ := years_in_decade * weeks_per_year
def weeks_per_mouse : ℕ := total_weeks_in_decade / total_mice

theorem weeks_per_mouse_correct : weeks_per_mouse = 4 := 
sorry

end weeks_per_mouse_correct_l712_712372


namespace angle_relation_l712_712548

noncomputable def square (A B C D N P Q: Point) : Prop :=
  is_square A B C D ∧ 
  on_side A B N ∧
  on_side A D P ∧
  dist N C = dist N P ∧
  on_segment A N Q ∧
  angle_eq Q P N N C B

theorem angle_relation (A B C D N P Q : Point) :
  square A B C D N P Q → 2 * angle B C Q = angle A Q P :=
begin
  sorry
end

end angle_relation_l712_712548


namespace solve_inequality_l712_712337

theorem solve_inequality (x : ℝ) : 3 * x^2 + 7 * x + 2 < 0 ↔ -1 < x ∧ x < -2/3 := by
  sorry

end solve_inequality_l712_712337


namespace cos_mul_tan_l712_712463

theorem cos_mul_tan (x y r : ℝ) (hx : x = 3 / 5) (hy : y = -4 / 5) (hr : r = 1) (h_cos : ∃ (α : ℝ), real.cos α = x / r) (h_tan : ∃ (α : ℝ), real.tan α = y / x):
  x * (y / x) = y := 
by
  sorry

end cos_mul_tan_l712_712463


namespace monotonic_increasing_interval_log_function_l712_712210

theorem monotonic_increasing_interval_log_function (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ x : ℝ, (x < 0 → ∃ y : ℝ, y = log a (x^2 - 2*x) ∧ monotone_increasing (x < 0)) :=
by
  sorry

end monotonic_increasing_interval_log_function_l712_712210


namespace clapping_times_l712_712264

theorem clapping_times 
  (jirka_clap : Nat → Prop)
  (petr_clap : Nat → Prop)
  (jirka_interval : 7) 
  (petr_interval : 13) 
  (jirka_time : Nat) 
  (petr_time : Nat)
  (both_clap : 90)
  (start_within : Nat) 
  (start_limit : 15) 
  (h_start_within_jirka : start_within = 6 ∨ start_within = 13)
  (h_start_within_petr : start_within = 12)
  (h_jirka_clap : ∀ t, jirka_clap (jirka_time + t * jirka_interval))
  (h_petr_clap : ∀ t, petr_clap (petr_time + t * petr_interval))
  (h_both_clap : ∃ t, jirka_clap (both_clap + t * jirka_interval) ∧ petr_clap (both_clap + t * petr_interval)) :
  (start_within = 6 ∨ start_within = 13) ∧ (start_within = 12) :=
sorry

end clapping_times_l712_712264


namespace three_points_with_small_angle_l712_712267

-- Theorem: Given n points in the plane, we can always find three which give an angle <= π / n
theorem three_points_with_small_angle (n : ℕ) (h : n ≥ 3) (points : Fin n → ℝ × ℝ) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
  ∃ θ : ℝ, θ = ∠ (points i) (points j) (points k) ∧ θ ≤ Real.pi / n := 
sorry

end three_points_with_small_angle_l712_712267


namespace smallest_area_quadrilateral_l712_712938

noncomputable def maximum_quadrilateral_area (s : ℝ) : ℝ :=
  let cos2θ := (1/s^2 - 1) / (1 + 1) in
  let sin2θ := sqrt(1 - cos2θ^2) in
  s^2 * (cos2θ/2 + (1/2) * sin2θ * cos2θ / 2)

theorem smallest_area_quadrilateral (s : ℝ) (hs : s > 0)
  (P : ℝ → ℝ → ℝ)
  (cond1 : ∀ x y, P x y = sqrt ((x - 0)^2 + (y - 0)^2))
  (cond2 : P 0 1 = 1) :
  maximum_quadrilateral_area s = (sqrt 5 + 2) / 4 :=
by
  sorry

end smallest_area_quadrilateral_l712_712938


namespace james_farmer_walk_distance_l712_712869

theorem james_farmer_walk_distance (d : ℝ) :
  ∃ d : ℝ,
    (∀ w : ℝ, (w = 300 + 50 → d = 20) ∧ 
             (w' = w * 1.30 ∧ w'' = w' * 1.20 → w'' = 546)) :=
by
  sorry

end james_farmer_walk_distance_l712_712869


namespace f_of_x_plus_1_f_of_2_f_of_x_l712_712384

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x := sorry

theorem f_of_2 : f 2 = 3 := sorry

theorem f_of_x (x : ℝ) : f x = x^2 - 1 := sorry

end f_of_x_plus_1_f_of_2_f_of_x_l712_712384


namespace cube_surface_area_l712_712216

/-- Given a cube with a space diagonal of 6, the surface area is 72. -/
theorem cube_surface_area (s : ℝ) (h : s * Real.sqrt 3 = 6) : 6 * s^2 = 72 :=
by
  sorry

end cube_surface_area_l712_712216


namespace translation_coordinates_l712_712077

variable (A B A1 B1 : ℝ × ℝ)

theorem translation_coordinates
  (hA : A = (-1, 0))
  (hB : B = (1, 2))
  (hA1 : A1 = (2, -1))
  (translation_A : A1 = (A.1 + 3, A.2 - 1))
  (translation_B : B1 = (B.1 + 3, B.2 - 1)) :
  B1 = (4, 1) :=
sorry

end translation_coordinates_l712_712077


namespace DE_perpendicular_to_BL_l712_712467

-- Define the problem conditions
variables {A B C M L D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited M] [Inhabited L] [Inhabited D] [Inhabited E]
variable {AB BC AC : ℝ}
variable (ΔABC : Triangle A B C)
variable (H1 : AB > BC)
variable (M_is_midpoint : is_midpoint A M C)
variable (BL_is_angle_bisector : is_angle_bisector B L A C (AB/BC))
variable (ML_parallel : parallel (line A B) (line M D))
variable (LE_parallel : parallel (line B C) (line L E))

-- Define the proof problem
theorem DE_perpendicular_to_BL : perpendicular (line D E) (line B L) :=
sorry

end DE_perpendicular_to_BL_l712_712467


namespace smallest_sum_of_squares_l712_712998

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l712_712998


namespace angle_condition_iff_l712_712498

-- Given a triangle ABC with AD as the median,
-- prove that angle B + angle DAC = pi/2 if and only if A = pi/2 or b = c.

variables {A B C D : Type} [plane_geometry : euclidean_geometry]

open euclidean_geometry

axiom is_median (ABC : triangle) (D : point) (AD : line) : is_median_of_triangle ABC AD

theorem angle_condition_iff (ABC : triangle) (AD : line)
  (h_median : is_median ABC AD) :
  angle (B) + angle (DAC) = π/2 ↔ angle (A) = π/2 ∨ side_length (b) = side_length (c) :=
sorry

end angle_condition_iff_l712_712498


namespace delete_middle_divides_l712_712748

def digits (n : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let a := n / 10000
  let b := (n % 10000) / 1000
  let c := (n % 1000) / 100
  let d := (n % 100) / 10
  let e := n % 10
  (a, b, c, d, e)

def delete_middle_digit (n : ℕ) : ℕ :=
  let (a, b, c, d, e) := digits n
  1000 * a + 100 * b + 10 * d + e

theorem delete_middle_divides (n : ℕ) (hn : 10000 ≤ n ∧ n < 100000) :
  (delete_middle_digit n) ∣ n :=
sorry

end delete_middle_divides_l712_712748


namespace tan_A_max_value_l712_712864

noncomputable def tan_max_A (AB BC : ℝ) (h_AB : AB = 24) (h_BC : BC = 18) : ℝ :=
  if h_AB : AB = 24 and h_BC : BC = 18 then (3 * real.sqrt 7) / 7 else 0

theorem tan_A_max_value :
  ∀ (A B C : Type) (AB BC : ℝ), AB = 24 → BC = 18 → tan_max_A AB BC = (3 * real.sqrt 7) / 7 :=
begin
  intros A B C AB BC h_AB h_BC,
  rw [h_AB, h_BC],
  unfold tan_max_A,
  rw if_pos,
  { refl },
  { exact ⟨rfl, rfl⟩ }
end

end tan_A_max_value_l712_712864


namespace product_of_D_coordinates_l712_712039

-- Define the coordinates of point M and point C
def mid_x : ℝ := 4
def mid_y : ℝ := 3
def mid_z : ℝ := 7

def c_x : ℝ := 5
def c_y : ℝ := -1
def c_z : ℝ := 5

-- Define the coordinates of point D that satisfy the midpoint condition
def D_coordinates (x y z : ℝ) := 
  (x, y, z) -- returns the coordinates to be used for midpoint and product computation

-- Main problem statement to prove
theorem product_of_D_coordinates : 
  ∃ (x y z : ℝ), 
    ((c_x + x) / 2 = mid_x ∧ (c_y + y) / 2 = mid_y ∧ (c_z + z) / 2 = mid_z) ∧
    x * y * z = 189 :=
begin
  -- Provide the conditions
  use 3,       -- x
  use 7,       -- y
  use 9,       -- z
  split,
  { split,
    { -- proving the x-coordinate midpoint condition
      linarith, },
    { split,
      { -- proving the y-coordinate midpoint condition
        linarith, },
      { -- proving the z-coordinate midpoint condition
        linarith, } } },
  { -- proving the product of coordinates is 189
    norm_num, }
end

end product_of_D_coordinates_l712_712039


namespace find_range_of_a_l712_712791

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def is_decreasing_on_domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
a ≤ b → f b ≤ f a

def within_domain (a b x : ℝ): Prop :=
a ≤ x ∧ x ≤ b

-- Main theorem
theorem find_range_of_a (f : ℝ → ℝ) (a : ℝ) :
  is_odd_function f ∧
  is_decreasing_on_domain f (-2) 2 ∧
  within_domain (-2) 2 (2*a + 1) ∧
  within_domain (-2) 2 (4*a - 3) ∧ 
  f(2 * a + 1) + f(4 * a - 3) > 0
  → (1/4:ℝ) ≤ a ∧ a < (1/3:ℝ) :=
sorry

end find_range_of_a_l712_712791


namespace marked_cells_in_grid_l712_712173

theorem marked_cells_in_grid :
  ∀ (grid : Matrix (Fin 5) (Fin 5) Bool), 
  (∀ (i j : Fin 3), ∃! (a b : Fin 3), grid (i + a + 1) (j + b + 1) = true) → ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 4 :=
by
  sorry

end marked_cells_in_grid_l712_712173


namespace find_integer_modulo_l712_712004

theorem find_integer_modulo :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 :=
by {
  sorry
}

end find_integer_modulo_l712_712004


namespace probability_product_even_or_prime_l712_712371

open Classical
open ProbabilityTheory

noncomputable def probability_even_or_prime : ℚ :=
  let outcomes := (Fin 8) × (Fin 8)
  let events := outcomes.filter (λ pair => 
    let prod := (pair.1.succ * pair.2.succ)
    (prod % 2 = 0) ∨ (Nat.Prime prod))
  events.count / outcomes.count

theorem probability_product_even_or_prime :
  probability_even_or_prime = 7 / 8 :=
by
  sorry

end probability_product_even_or_prime_l712_712371


namespace isosceles_right_triangle_area_l712_712208

-- Define the isosceles right triangle and its properties

theorem isosceles_right_triangle_area 
  (h : ℝ)
  (hyp : h = 6) :
  let l : ℝ := h / Real.sqrt 2
  let A : ℝ := (l^2) / 2
  A = 9 :=
by
  -- The proof steps are skipped with sorry
  sorry

end isosceles_right_triangle_area_l712_712208


namespace Lara_age_in_10_years_l712_712135

theorem Lara_age_in_10_years (Lara_age_7_years_ago : ℕ) (h1 : Lara_age_7_years_ago = 9) : 
  Lara_age_7_years_ago + 7 + 10 = 26 :=
by
  rw [h1]
  norm_num
  sorry

end Lara_age_in_10_years_l712_712135


namespace sqrt_meaningful_range_l712_712462

theorem sqrt_meaningful_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end sqrt_meaningful_range_l712_712462


namespace min_value_box_l712_712823

theorem min_value_box (a b : ℤ) (h_distinct : a ≠ b) (h_prod : a * b = 30) : 
  ∃ (Box : ℤ), (Box = a^2 + b^2) ∧ Box = 61 :=
by
  use a^2 + b^2
  split
  sorry
  sorry

end min_value_box_l712_712823


namespace find_total_cows_l712_712258

-- Definitions as per the conditions
variables (D C L H : ℕ)

-- Condition 1: Total number of legs
def total_legs : ℕ := 2 * D + 4 * C

-- Condition 2: Total number of heads
def total_heads : ℕ := D + C

-- Condition 3: Legs are 28 more than twice the number of heads
def legs_heads_relation : Prop := total_legs D C = 2 * total_heads D C + 28

-- The theorem to prove
theorem find_total_cows (h : legs_heads_relation D C) : C = 14 :=
sorry

end find_total_cows_l712_712258


namespace geometric_sequence_a5_l712_712859

-- Definitions from the conditions
def a1 : ℕ := 2
def a9 : ℕ := 8

-- The statement we need to prove
theorem geometric_sequence_a5 (q : ℝ) (h1 : a1 = 2) (h2 : a9 = a1 * q ^ 8) : a1 * q ^ 4 = 4 := by
  have h_q4 : q ^ 4 = 2 := sorry
  -- Proof continues...
  sorry

end geometric_sequence_a5_l712_712859


namespace total_weekly_earnings_l712_712483

/-- Define the intervals and prices for different items. --/
def women_tshirt_interval := 30
def women_tshirt_price := 18

def men_tshirt_interval := 40
def men_tshirt_price := 15

def women_jeans_interval := 45
def women_jeans_price := 40

def men_jeans_interval := 60
def men_jeans_price := 45

def unisex_hoodie_interval := 70
def unisex_hoodie_price := 35

/-- Define the shop opening hours and other conditions. --/
def shop_open_hours := 12
def wednesday_discount := 0.10
def saturday_tax := 0.05

/-- Calculate the total earnings for the shop from selling clothing items per week, considering the discounts and taxes on specific days. --/
theorem total_weekly_earnings : 
  let women_tshirt_sales := (shop_open_hours * 60) / women_tshirt_interval * women_tshirt_price in
  let men_tshirt_sales := (shop_open_hours * 60) / men_tshirt_interval * men_tshirt_price in
  let women_jeans_sales := (shop_open_hours * 60) / women_jeans_interval * women_jeans_price in
  let men_jeans_sales := (shop_open_hours * 60) / men_jeans_interval * men_jeans_price in
  let unisex_hoodie_sales := (shop_open_hours * 60) / unisex_hoodie_interval * unisex_hoodie_price in
  let daily_earnings := women_tshirt_sales + men_tshirt_sales + women_jeans_sales + men_jeans_sales + unisex_hoodie_sales in
  let normal_days_earnings := daily_earnings * 5 in
  let wednesday_earnings := daily_earnings * (1 - wednesday_discount) in
  let saturday_earnings := daily_earnings * (1 + saturday_tax) in
  normal_days_earnings + wednesday_earnings + saturday_earnings = 15512.40 := 
by
  sorry

end total_weekly_earnings_l712_712483


namespace eggs_left_over_l712_712328

def david_eggs : ℕ := 44
def elizabeth_eggs : ℕ := 52
def fatima_eggs : ℕ := 23
def carton_size : ℕ := 12

theorem eggs_left_over : 
  (david_eggs + elizabeth_eggs + fatima_eggs) % carton_size = 11 :=
by sorry

end eggs_left_over_l712_712328


namespace min_value_g_zero_points_condition_l712_712022

-- Given conditions for the first problem
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (4 * a) / (x + a^2) - 2
def g (a : ℝ) : ℝ := f a (a^2)

-- Statement for the minimum value of g(a)
theorem min_value_g (a : ℝ) (h : a > 0) : g a ≥ 0 :=
sorry

-- Given conditions for the second problem
theorem zero_points_condition (a : ℝ) (h : a > 0) : (∀ y : ℝ, ∃ x1 x2 x3 : ℝ,
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ y = f a x1 ∧ y = f a x2 ∧ y = f a x3) → (0 < a ∧ a < 1) :=
sorry

end min_value_g_zero_points_condition_l712_712022


namespace sequence_unique_integers_l712_712520

theorem sequence_unique_integers (a : ℕ → ℤ) 
  (H_inf_pos : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n > N) 
  (H_inf_neg : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n < N)
  (H_diff_remainders : ∀ n : ℕ, n > 0 → ∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) → (1 ≤ j ∧ j ≤ n) → i ≠ j → (a i % ↑n) ≠ (a j % ↑n)) :
  ∀ m : ℤ, ∃! n : ℕ, a n = m := sorry

end sequence_unique_integers_l712_712520


namespace average_temperature_august_l712_712579

theorem average_temperature_august :
  ∃ a b : ℝ,
    (a - 0.5 * b = 22) ∧
    (a + 0.5 * b = 4) ∧
    (let y := a + b * Real.sin (π/6 * 8 + π/6) in y = 31) :=
by 
  sorry

end average_temperature_august_l712_712579


namespace correct_calculation_l712_712250

-- Definitions of the equations
def option_A (a : ℝ) : Prop := a + 2 * a = 3 * a^2
def option_B (a b : ℝ) : Prop := (a^2 * b)^3 = a^6 * b^3
def option_C (a : ℝ) (m : ℕ) : Prop := (a^m)^2 = a^(m+2)
def option_D (a : ℝ) : Prop := a^3 * a^2 = a^6

-- The theorem that states option B is correct and others are incorrect
theorem correct_calculation (a b : ℝ) (m : ℕ) : 
  ¬ option_A a ∧ 
  option_B a b ∧ 
  ¬ option_C a m ∧ 
  ¬ option_D a :=
by sorry

end correct_calculation_l712_712250


namespace exponentiation_of_fraction_l712_712084

theorem exponentiation_of_fraction : 
  (c d : ℝ) (h1 : 90^c = 4) (h2 : 90^d = 7) : 
  18 ^ ((1 - c - d) / (2 * (1 - d))) = 45 / 14 :=
by
  sorry

end exponentiation_of_fraction_l712_712084


namespace circumscribed_quadrilateral_l712_712225

-- Definitions of the conditions and elements involved.
variables {A C B : Point}
variables (γ1 γ2 γ3 : Arc A C)
variables (h1 h2 h3 : Ray B) {P : Point} (d : Point → ℝ)
variables (u12 u23 v12 v23 : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  (arc_betweenness γ1 γ2 γ3) ∧
  (half_plane_defined_by_line A C γ1 γ2 γ3) ∧ 
  (ray_betweenness h1 h2 h3 B) ∧
  (on_segment AC B) ∧
  (tangent_circle_properties h1 h2 γ1 γ2 γ3 u12 u23 v12 v23)

-- Statement we need to prove
theorem circumscribed_quadrilateral :
  conditions →
  (u12 = v12) → (u12 = v23) → (u23 = v12) → 
  (u23 = v23) :=
by
  intro h cond1 cond2 cond3
  sorry

end circumscribed_quadrilateral_l712_712225


namespace coefficient_xy2_in_expansion_l712_712023

theorem coefficient_xy2_in_expansion (a : ℝ) (h : a = ∫ x in 0..(real.pi / 2), real.sin x + real.cos x) : 
    (∃ c : ℝ, c = 72) :=
by
  have h1 : a = 2, from sorry,
  have h2 : ∃ c1 c2 c3, c1 = (1 + a • x)^6 ∧ c2 = (1 + y)^4 ∧ c3 = 72, from sorry,
  exact ⟨72, h2.right.right.right⟩

end coefficient_xy2_in_expansion_l712_712023


namespace percentage_loss_l712_712687

theorem percentage_loss (CP SP : ℝ) (h₁ : CP = 1400) (h₂ : SP = 1232) :
  ((CP - SP) / CP) * 100 = 12 :=
by
  sorry

end percentage_loss_l712_712687


namespace max_m_ratio_l712_712781

theorem max_m_ratio (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : ∀ a b, (4 / a + 1 / b) ≥ m / (a + 4 * b)) :
  (m = 16) → (b / a = 1 / 4) :=
by sorry

end max_m_ratio_l712_712781


namespace polar_equation_of_line_l_distance_AB_l712_712113

noncomputable def line_l_parametric (t : ℝ) : ℝ × ℝ :=
  (t, 2 * t)

noncomputable def curve_C_polar (ρ θ : ℝ) : Prop :=
  ρ^2 + 2 * ρ * real.sin θ - 3 = 0

def to_polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  (real.sqrt (x^2 + y^2), real.arctan y x)

def from_polar_coordinates (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * real.cos θ, ρ * real.sin θ)

theorem polar_equation_of_line_l :
  ∀ (θ : ℝ), 
  let l_polar := from_polar_coordinates 1 θ in
  l_polar.snd = 2 * l_polar.fst → real.sin θ = 2 * real.cos θ := 
by
  intros θ l_polar h
  sorry

theorem distance_AB :
  ∀ (A B : ℝ×ℝ), curve_C_polar (A.fst) (A.snd) ∧ curve_C_polar (B.fst) (B.snd) ∧
  (∀ t ∈ [0, 1], line_l_parametric t ∈ [A,B]) →
  real.dist A B = 2 * real.sqrt 19 / 5 := 
by
  intros A B h
  sorry

end polar_equation_of_line_l_distance_AB_l712_712113


namespace four_spheres_max_intersections_l712_712369

noncomputable def max_intersection_points (n : Nat) : Nat :=
  if h : n > 0 then n * 2 else 0

theorem four_spheres_max_intersections : max_intersection_points 4 = 8 := by
  sorry

end four_spheres_max_intersections_l712_712369


namespace cos_half_pi_minus_2alpha_l712_712041

open Real

theorem cos_half_pi_minus_2alpha (α : ℝ) (h : sin α - cos α = 1 / 3) : cos (π / 2 - 2 * α) = 8 / 9 :=
sorry

end cos_half_pi_minus_2alpha_l712_712041


namespace range_of_x_l712_712456

theorem range_of_x (x : ℝ) : 3 * x - 2 ≥ 0 ↔ x ≥ 2 / 3 := by
  sorry

end range_of_x_l712_712456


namespace triangle_area_correct_l712_712244

-- Define the points (vertices) of the triangle
def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (8, -3)
def point3 : ℝ × ℝ := (2, 7)

-- Function to calculate the area of the triangle given three points (shoelace formula)
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.2 * C.1 - C.2 * A.1 - A.2 * B.1)

-- Prove that the area of the triangle with the given vertices is 18 square units
theorem triangle_area_correct : triangle_area point1 point2 point3 = 18 :=
by
  sorry

end triangle_area_correct_l712_712244


namespace exists_same_color_points_at_unit_distance_l712_712292

theorem exists_same_color_points_at_unit_distance
  (color : ℝ × ℝ → ℕ)
  (coloring : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_at_unit_distance_l712_712292


namespace tangent_line_at_zero_l712_712586

noncomputable def f : ℝ → ℝ := λ x, Real.exp x + Real.sin x + 1

theorem tangent_line_at_zero :
  let f' : ℝ → ℝ := λ x, Real.exp x + Real.cos x in
  let slope_at_zero : ℝ := f' 0 in
  let point_at_zero : ℝ × ℝ := (0, f 0) in
  ∀ x y : ℝ, (y - point_at_zero.2 = slope_at_zero * (x - point_at_zero.1)) ↔ y = 2 * x + 2 :=
by sorry

end tangent_line_at_zero_l712_712586


namespace number_of_zeros_of_f_l712_712419

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2017 ^ x + Real.log x / Real.log 2017 else -2017 ^ (-x) - Real.log (-x) / Real.log 2017

theorem number_of_zeros_of_f :
  (∃ x : ℝ, f x = 0) ∧ (∃ x : ℝ, x > 0 ∧ f x = 0) ∧ (∃ x : ℝ, x < 0 ∧ f x = 0) :=
sorry

end number_of_zeros_of_f_l712_712419


namespace collinearity_of_points_l712_712914

noncomputable theory
open_locale classical

variables {A P Q M N I T K : Type*}

-- Conditions given in the problem
variables [IncidenceGeometry A P Q M N I T K]
variable [IntersectionPoint T (Line A P) (Line C Q)]
variable [IntersectionPoint K (Line M P) (Line N Q)]

-- Statement of the proof problem
theorem collinearity_of_points :
  Collinear {T, K, I} :=
sorry

end collinearity_of_points_l712_712914


namespace no_representation_l712_712519

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem no_representation (a b c : ℕ) 
  (h1 : gcd a b = 1) (h2 : gcd b c = 1) (h3 : gcd c a = 1) : 
  ¬ ∃ (x y z : ℕ), 2 * a * b * c - a * b - b * c - c * a = b * c * x + c * a * y + a * b * z :=
by
  sorry

end no_representation_l712_712519


namespace b_minus_a_is_five_l712_712939

theorem b_minus_a_is_five (a b : ℤ) (h : ∃ (z : ℝ), z^2 + (a : ℝ) * z + (b : ℝ) = 0 ∧ z = Real.sqrt(7 - 4 * Real.sqrt 3)) : b - a = 5 := by
sorry

end b_minus_a_is_five_l712_712939


namespace smallest_area_l712_712247

open Real

noncomputable def f (x : ℝ) : ℝ := 7 - 6 * x - x ^ 2

noncomputable def tangent_line (x₀ : ℝ) : ℝ → ℝ := 
  7 - 6 * x₀ - x₀ ^ 2 + (-6 - 2 * x₀) * (x - x₀)

noncomputable def area (x₀ : ℝ) : ℝ :=
  6 * (x₀ ^ 2 + 4 * x₀ + 19)

theorem smallest_area : 
  ∃ x₀ ∈ Icc (-5 : ℝ) 1, area x₀ = 90 :=
begin
  use -2,
  split,
  { norm_num },
  { norm_num }
end

end smallest_area_l712_712247


namespace collinear_T_K_I_l712_712882

noncomputable def intersection (P Q : Set Point) : Point := sorry

variables (A P C Q M N I T K : Point)

-- Definitions based on conditions
def T_def : Point := intersection (line_through A P) (line_through C Q)
def K_def : Point := intersection (line_through M P) (line_through N Q)

-- Proof statement
theorem collinear_T_K_I :
  collinear ({T_def A P C Q, K_def M P N Q, I} : Set Point) := sorry

end collinear_T_K_I_l712_712882


namespace unattainable_y_l712_712357

theorem unattainable_y (x : ℝ) (hx : x ≠ -2 / 3) : ¬ (∃ x, y = (x - 3) / (3 * x + 2) ∧ y = 1 / 3) := by
  sorry

end unattainable_y_l712_712357


namespace alpha_beta_pi_div_two_iff_l712_712757

theorem alpha_beta_pi_div_two_iff :
  ∀ (α β : ℝ), (0 < α ∧ α < (π / 2)) ∧ (0 < β ∧ β < (π / 2)) →
  (α + β = π / 2 ↔ (sin α ^ 4 / cos β ^ 2 + sin β ^ 4 / cos α ^ 2 = 1)) :=
by
  intros,
  sorry -- Proof omitted

end alpha_beta_pi_div_two_iff_l712_712757


namespace points_on_single_circle_l712_712974

variable (M : Set (ℝ × ℝ)) (f : M → ℤ)
variable (n p : ℤ)

-- Positive integer requirement for n
axiom n_positive : n > 0

-- n ≥ 4
axiom n_ge_4 : n ≥ 4

-- p is a prime number and p ≥ n - 2
axiom p_prime : nat.prime p
axiom p_ge_n_minus_2 : p ≥ n - 2

-- M contains n points with no three collinear
axiom M_has_n_points : card M = n
axiom no_three_collinear : ∀ (A B C : ℝ × ℝ), (A ∈ M) → (B ∈ M) → (C ∈ M) → 
                           (A ≠ B) → (B ≠ C) → (A ≠ C) → ¬collinear {A, B, C}

-- Mapping f satisfies the conditions
axiom unique_zero : ∃! X ∈ M, f X = 0
axiom circumcircle_sum : ∀ (A B C : ℝ × ℝ), (A ∈ M) → (B ∈ M) → (C ∈ M) → (A ≠ B) → (B ≠ C) → 
                        (A ≠ C) → let K = circumcircle A B C in ∑ P in (M ∩ K), f P ≡ 0 [MOD p]

-- The statement to prove
theorem points_on_single_circle : ∀ (M : Set (ℝ × ℝ)), 
                                   (∀ (A B C : ℝ × ℝ), (A ∈ M) → (B ∈ M) → (C ∈ M) → 
                                   (A ≠ B) → (B ≠ C) → (A ≠ C) → ¬collinear {A, B, C}) →
                                   (∀ (A B C : ℝ × ℝ), (A ∈ M) → (B ∈ M) → (C ∈ M) → let K = circumcircle A B C in ∀ P ∈ M ∩ K, f P ≡ 0 [MOD p]) →
                                   (∃! X ∈ M, f X = 0) →
                                   (∃ C : circle, ∀ P ∈ M, P ∈ C) :=
by sorry

end points_on_single_circle_l712_712974


namespace count_distinct_integer_sums_of_special_fractions_l712_712318

theorem count_distinct_integer_sums_of_special_fractions : 
  let special_fractions := {frac : ℚ | ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a + b = 18 ∧ frac = a / b} in
  let special_sum := {x + y | x y : ℚ, x ∈ special_fractions ∧ y ∈ special_fractions} in
  fintype.card {n : ℤ | ∃ x ∈ special_sum, x = n} = 6 :=
by
  sorry

end count_distinct_integer_sums_of_special_fractions_l712_712318


namespace constructible_and_bound_area_l712_712537

theorem constructible_and_bound_area 
  {α β γ : ℝ} 
  (h1 : 0 < α) 
  (h2 : 0 < β) 
  (h3 : 0 < γ) 
  (h4 : α + β + γ < π) 
  (h5 : α + β > γ) 
  (h6 : β + γ > α) 
  (h7 : γ + α > β) :
  (∃ (a b c : ℝ), a = Real.sin α ∧ b = Real.sin β ∧ c = Real.sin γ ∧ a + b > c ∧ b + c > a ∧ c + a > b) ∧
  triangle_area (Real.sin α) (Real.sin β) (Real.sin γ) ≤ 1 / 8 * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) :=
by
  sorry

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

end constructible_and_bound_area_l712_712537


namespace smallest_sum_of_squares_l712_712985

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l712_712985


namespace collinear_TKI_l712_712920

-- Definitions based on conditions
variables {A P Q C M N T K I : Type}
variables (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)

-- Definitions to represent the intersection points 
def T_def (A P Q C : Type) (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) : Prop := 
  ∃ T : Type, line_AP A T ∧ line_CQ C T

def K_def (M P N Q : Type) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop) : Prop := 
  ∃ K : Type, line_MP M K ∧ line_NQ N K

-- Theorem statement
theorem collinear_TKI (A P Q C M N T K I : Type)
  (line_AP : A → P → Prop) (line_CQ : C → Q → Prop) (line_MP : M → P → Prop) (line_NQ : N → Q → Prop)
  (hT : T_def A P Q C line_AP line_CQ) 
  (hK : K_def M P N Q line_MP line_NQ) : 
  collinear T K I :=
sorry

end collinear_TKI_l712_712920


namespace eval_f_four_times_l712_712359

noncomputable def f (z : Complex) : Complex := 
if z.im ≠ 0 then z * z else -(z * z)

theorem eval_f_four_times : 
  f (f (f (f (Complex.mk 2 1)))) = Complex.mk 164833 354192 := 
by 
  sorry

end eval_f_four_times_l712_712359


namespace area_of_triangle_AMB_l712_712847

noncomputable def triangle_area_AMB (OP OQ : ℝ) (h_OP : OP = sqrt 2) (h_OQ : OQ = sqrt 3 / 3) : ℝ :=
  let AM := 2 * OP
  let BM := 2 * OQ
  let angle_AMB := Real.angle.pi - Real.angle.of_degrees 30
  let area := 1 / 2 * AM * BM * Real.sin angle_AMB.toRealAngle
  area

theorem area_of_triangle_AMB :
  triangle_area_AMB (sqrt 2) (sqrt 3 / 3) (rfl : sqrt 2 = sqrt 2) (rfl : sqrt 3 / 3 = sqrt 3 / 3) = sqrt 6 / 3 :=
by
  sorry

end area_of_triangle_AMB_l712_712847


namespace num_sets_A_l712_712211

theorem num_sets_A : 
  { A : set ℕ // {1, 2} ∪ A = {1, 2, 3} }.to_finset.card = 4 := 
by sorry

end num_sets_A_l712_712211


namespace combined_uncovered_fractions_correct_l712_712293

noncomputable def pi := Real.pi

def diameter_x := 16
def diameter_y := 18
def diameter_z := 20

def radius_x := diameter_x / 2
def radius_y := diameter_y / 2
def radius_z := diameter_z / 2

def area_x := pi * radius_x ^ 2
def area_y := pi * radius_y ^ 2
def area_z := pi * radius_z ^ 2

def uncovered_fraction_x := 0
def uncovered_fraction_z := (area_z - area_y) / area_z

def combined_uncovered_fractions := uncovered_fraction_x + uncovered_fraction_z

theorem combined_uncovered_fractions_correct :
  combined_uncovered_fractions = 19 / 100 :=
by
  sorry

end combined_uncovered_fractions_correct_l712_712293


namespace max_subset_elements_l712_712152

theorem max_subset_elements (T : Set ℕ) (hT1 : ∀ x ∈ T, x ≤ 1000) (hT2 : ∀ x y ∈ T, x ≠ y → (x - y) ≠ 4 ∧ (x - y) ≠ 6 ∧ (x - y) ≠ 7) : T.card ≤ 415 :=
sorry

end max_subset_elements_l712_712152


namespace tan_value_l712_712820

theorem tan_value (α : ℝ) 
  (h : (2 * Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) - 1) / (Real.sqrt 2 * Real.sin (2 * α + π / 4)) = 4) : 
  Real.tan (2 * α + π / 4) = 1 / 4 :=
by
  sorry

end tan_value_l712_712820


namespace find_roots_of_quadratic_l712_712741

theorem find_roots_of_quadratic (p q : ℤ) (f : ℤ → ℤ) 
  (h1 : ∀ x, f(x) = x^2 + p * x + q) 
  (h2 : ∃ x1 x2 : ℕ, x1 ≠ x2 ∧ f 3 ∈ Nat.Prime ∧ x1 + x2 = -p ∧ x1 * x2 = q ∧ (x1 > 3 ∧ Nat.Prime x1 ∨ x1 < 3 ∧ x2 < 3)) : 
    ∃ x1 x2 : ℕ, (x1 = 4 ∧ x2 = 5 ∨ x1 = 1 ∧ x2 = 2) :=
by
  sorry

end find_roots_of_quadratic_l712_712741


namespace multiples_count_12_3_2_4_equals_l712_712529

def is_multiple (n k : ℕ) : Prop := ∃ m : ℕ, k = m * n

def count_multiples_under (n bound : ℕ) : ℕ :=
  Nat.count (λ k => is_multiple n k ∧ k < bound) (List.range bound)

theorem multiples_count_12_3_2_4_equals (bound : ℕ) (h : bound = 60) :
  let a := count_multiples_under 12 bound
  let b := count_multiples_under 12 bound
  (a - b) * (a - b) = 0 :=
by
  sorry

end multiples_count_12_3_2_4_equals_l712_712529


namespace minimum_Q_value_l712_712363

def is_valid_n (n k : ℤ) : Prop :=
  ⌊n / k⌋ + ⌊(150 - n) / k⌋ = ⌊150 / k⌋

def Q (k : ℤ) : ℚ :=
  (1 + (Finset.filter (λ n, is_valid_n n k) (Finset.range 150)).card) / 149

def minimum_Q : ℚ :=
  Finset.inf' (Finset.filter (λ k, k % 2 = 1) (Finset.range 150)) (λ k, Q k) (by norm_num)

theorem minimum_Q_value : minimum_Q = 22 / 67 := sorry

end minimum_Q_value_l712_712363


namespace find_b_l712_712001

theorem find_b (b c : ℝ) :
  let a := 2 in
  (3 * X^2 - 2 * X + 4) * (a * X^2 + b * X + c) = (6 * X^4 - 5 * X^3 + 11 * X^2 - 8 * X + 16) →
  b = -1 / 3 := 
by 
  sorry

end find_b_l712_712001


namespace min_value_of_mod_z_add_inv_z_l712_712856

variable (z : ℂ)

def forms_parallelogram (z : ℂ) : Prop :=
  let p1 := (0 : ℂ)
  let p2 := z
  let p3 := 1 / z
  let p4 := z + 1 / z
  true -- as a placeholder, we assume the definition of the parallelogram formation here

def parallelogram_area (z : ℂ) : ℝ :=
  abs (z) * abs (1 / z) * abs (sin (2 * complex.arg(z)))

theorem min_value_of_mod_z_add_inv_z (h₀ : (z : ℂ).re > 0)
    (h₁ : parallelogram_area z = 35 / 37)
    (h₂ : forms_parallelogram z) :
    |z + 1 / z| = 5 * real.sqrt 74 / 37 :=
  sorry

end min_value_of_mod_z_add_inv_z_l712_712856


namespace binomial_coeff_sum_l712_712425

theorem binomial_coeff_sum (n : ℕ) :
  (∑ k in Finset.range (n + 1), (∥binomial n k∥ : ℝ)) - (if n % 2 = 0 then 1 else -1) = 255 → n = 8 :=
by
  intro h
  sorry

end binomial_coeff_sum_l712_712425


namespace sequence_solution_sum_solution_l712_712860

noncomputable def a (n : ℕ) : ℕ := 
    if n = 1 then 0 
    else if n = 2 then 0 
    else ⌊ (4 / 3 : ℚ) * n - (10 / 3 : ℚ) ⌋ * (3 ^ n) + 2

def S (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ k, a (k+1))

theorem sequence_solution (n : ℕ) (hn : 2 ≤ n) :
  a n = ⌊(4 / 3 : ℚ) * n - (10 / 3 : ℚ)⌋ * (3 ^ n) + 2 := 
sorry

theorem sum_solution (n : ℕ) :
  S n = 2 * n * (3 ^ n - 1) :=
sorry

end sequence_solution_sum_solution_l712_712860


namespace suitcase_lock_settings_l712_712703

theorem suitcase_lock_settings : 
  ∀ (d : Fin 8 → Fin 8 → Fin 8 → Fin 8 → Bool), 
  (∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l → d i j k l) →
  (  (number_of_valid_settings d) = 1680 ) := 
sorry

end suitcase_lock_settings_l712_712703


namespace valid_permutations_count_l712_712816

noncomputable def number_of_valid_permutations (n : ℕ) : ℝ :=
  (1 / Real.sqrt 5) * (
    ((1 + Real.sqrt 5) / 2) ^ (n + 1) -
    ((1 - Real.sqrt 5) / 2) ^ (n + 1)
  )

theorem valid_permutations_count (n : ℕ) :
  let perms := number_of_valid_permutations n in
  ∃ (p : ℝ), p = perms :=
sorry

end valid_permutations_count_l712_712816


namespace most_likely_composition_l712_712622

def event_a : Prop := (1 / 3) * (1 / 3) * 2 = (2 / 9)
def event_d : Prop := 2 * (1 / 3 * 1 / 3) = (2 / 9)

theorem most_likely_composition :
  event_a ∧ event_d :=
by sorry

end most_likely_composition_l712_712622


namespace ratio_of_areas_l712_712487

variables {α : ℝ} {A B C D E : Type*}

-- Define points in the Euclidean space 
variable [euclidean_space E]

-- Define conditions for the problem
variable [is_diameter (A B : E)] -- AB is a diameter of the circle
variable [is_chord (C D : E)] -- CD is a chord
variable [intersect_at (C D) (A B) E] -- CD intersects AB at E
variable [perpendicular (A D : E) (D E : E)] -- AD is perpendicular to DE
variable (angle_CEB : angle C E B = α) -- Angle CEB is alpha

-- Define the areas of the triangles based on the vertices
def area_triangle (P Q R : E) : ℝ := sorry

-- Area of triangle ADE
noncomputable def area_ADE : ℝ := area_triangle A D E

-- Area of triangle BCE
noncomputable def area_BCE : ℝ := area_triangle B C E

-- The theorem we need to prove:
theorem ratio_of_areas :
  ∃ (sec_alpha : ℝ), area_ADE / area_BCE = sec_alpha^2 := sorry

end ratio_of_areas_l712_712487


namespace find_new_student_weight_l712_712199

def average_weight (total_weight : ℕ) (num_students : ℕ) : ℝ :=
  total_weight / num_students.to_nat

variable
  (old_avg_weight : ℝ)
  (old_total_students : ℕ)
  (new_avg_weight : ℝ)
  (new_total_students : ℕ)
  (W_new : ℕ)

theorem find_new_student_weight (h1 : old_avg_weight = 28) 
                                (h2 : old_total_students = 29)
                                (h3 : new_avg_weight = 27.5)
                                (h4 : new_total_students = 30)
                                (h5 : old_total_students * old_avg_weight + ℝ.of_nat W_new = new_total_students * new_avg_weight) :
  W_new = 13 :=
by 
  sorry

end find_new_student_weight_l712_712199


namespace sqrt_sum_eq_seven_l712_712572

noncomputable def x : ℝ := sorry -- the exact value of x is not necessary

theorem sqrt_sum_eq_seven (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
  sqrt (64 - x^2) + sqrt (36 - x^2) = 7 := sorry

end sqrt_sum_eq_seven_l712_712572


namespace part1_part2_l712_712063

-- Defining the function f and g
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + (a + 2) * x + a) / (Real.exp x)
def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x * f a x

-- Statement for part 1
theorem part1 (a : ℝ) (h : ∃ x (hx : x ≥ a), g a x ≤ 9) : a ≤ 3 / 2 := sorry

-- Defining M and N as the maximum and minimum values of f respectively
def M (a x1 x2 : ℝ) : ℝ := max (f a x1) (f a x2)
def N (a x1 x2 : ℝ) : ℝ := min (f a x1) (f a x2)

-- Statement for part 2
theorem part2 (a x1 x2 : ℝ) (h1 : a^2 + 8 ≥ 0) (h2 : x1 < x2) :
  ∃ t, t = x1 - x2 ∧ 
  (-2 * Real.sqrt 2 ≤ t) ∧ 
  (M a x1 x2 / N a x1 x2 ∈ Set.Ioo (-(3 + 2 * Real.sqrt 2) * Real.exp (-2 * Real.sqrt 2)) 0) := sorry

end part1_part2_l712_712063


namespace sufficiency_condition_a_gt_b_sq_gt_sq_l712_712450

theorem sufficiency_condition_a_gt_b_sq_gt_sq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a > b → a^2 > b^2) ∧ (∀ (h : a^2 > b^2), ∃ c > 0, ∃ d > 0, c^2 > d^2 ∧ ¬(c > d)) :=
by
  sorry

end sufficiency_condition_a_gt_b_sq_gt_sq_l712_712450


namespace find_cd_l712_712248

noncomputable def g (x : ℝ) (c : ℝ) (d : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) :
  g 2 c d = -9 ∧ g (-1) c d = -19 ↔
  (c = 19/3 ∧ d = -7/3) :=
by
  sorry

end find_cd_l712_712248


namespace partition_positive_integers_into_historical_sets_l712_712699

def is_historical (x y z : ℕ) : Prop :=
  {z - y, y - x} = {1776, 2001}

theorem partition_positive_integers_into_historical_sets :
  ∃ (H : ℕ → ℕ → ℕ → Prop), (∀ n : ℕ, ∃ x y z, H x y z ∧ is_historical x y z ∧ (x = n ∨ y = n ∨ z = n)) ∧ 
  (∀ x1 y1 z1 x2 y2 z2, H x1 y1 z1 → H x2 y2 z2 → (x1 = x2 ∨ y1 = y2 ∨ z1 = z2) → (x1, y1, z1) = (x2, y2, z2)) :=
by
  sorry

end partition_positive_integers_into_historical_sets_l712_712699


namespace ariel_age_problem_l712_712472

theorem ariel_age_problem 
    (current_age : ℕ) 
    (future_multiple : ℕ) 
    (current_age_is_five : current_age = 5)
    (future_multiple_is_four : future_multiple = 4): 
    ∃ (x : ℕ), (current_age + x = future_multiple * current_age) := 
by
  have h1 : current_age = 5 := current_age_is_five
  have h2 : future_multiple = 4 := future_multiple_is_four
  use 15
  rw [h1, h2]
  simp

end ariel_age_problem_l712_712472


namespace line_l1_equation_max_distance_parallel_lines_l712_712612

section LineEquations

variable {R : Type*} [LinearOrderedField R]

-- defining point A(1, 1) and point B(0, -1) as constants
def A : R × R := (1, 1)
def B : R × R := (0, -1)

-- defining the slope function
def slope (p1 p2 : R × R) : R :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- defining a line in point-slope form
def line (point : R × R) (m : R) : R → R := λ x, point.2 + m * (x - point.1)

-- The correct answer to the problem
theorem line_l1_equation_max_distance_parallel_lines :
  let m := slope B A in
  let m_perpendicular := - (1 / m) in  -- negative reciprocal of the slope of AB
  ∀ (l1 l2 : R → R),
    (l1 = line A m_perpendicular) →
    (l2 = line B m_perpendicular) →
    l1 = λ x, 1 / 2 * (x + 3 / 2) :=
sorry

end LineEquations

end line_l1_equation_max_distance_parallel_lines_l712_712612


namespace mod_problem_l712_712006
open Int

theorem mod_problem : 
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [MOD 11] :=
by
  use 3
  split
  · show 0 ≤ 3
    exact le_refl 3
  split
  · show 3 ≤ 10
    exact le_of_lt (by norm_num)
  · show 3 ≡ 123456 [MOD 11]
    exact Nat.modeq.symm (by norm_num [mod_eq_of_lt, Int.mod_def])

end mod_problem_l712_712006


namespace conditional_probability_B_given_A_l712_712691

-- Definitions of sets A and B
def A_set (x : ℝ) : Prop := 0 < x ∧ x < 1/2
def B_set (x : ℝ) : Prop := 1/4 < x ∧ x < 3/4

-- Definition of the interval (0, 1)
def interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Probability of a set on the interval (0, 1)
def probability (S : set ℝ) : ℝ :=
  ∫ x in S, 1

-- Conditional probability of B given A
def conditional_probability (A B : set ℝ) : ℝ :=
  probability (A ∩ B) / probability A

-- Theorem to prove P(B|A) = 1/2
theorem conditional_probability_B_given_A :
  conditional_probability ({ x | A_set x }) ({ x | B_set x }) = 1 / 2 :=
by sorry

end conditional_probability_B_given_A_l712_712691


namespace reconstruct_quadrilateral_l712_712777

theorem reconstruct_quadrilateral 
  (A' B' C' D' : ℝ^3)  -- Assuming points are in three-dimensional space for generality
  (A B C D : ℝ^3)
  (AB_eq : B = (A + A') / 2)
  (BC_eq : C = (B + B') / 2)
  (CD_eq : D = (C + C') / 2)
  (DA_eq : A = (D + D') / 2) :
  A = (1 / 15 : ℝ) • A' + (2 / 15 : ℝ) • B' + (4 / 15 : ℝ) • C' + (8 / 15 : ℝ) • D' :=
sorry

end reconstruct_quadrilateral_l712_712777


namespace stewart_farm_food_l712_712312

variable (S H : ℕ) (HorseFoodPerHorsePerDay : Nat) (TotalSheep : Nat)

theorem stewart_farm_food (ratio_sheep_horses : 6 * H = 7 * S) 
  (total_sheep_count : S = 48) 
  (horse_food : HorseFoodPerHorsePerDay = 230) : 
  HorseFoodPerHorsePerDay * (7 * 48 / 6) = 12880 :=
by
  sorry

end stewart_farm_food_l712_712312


namespace area_of_square_on_AD_l712_712623

theorem area_of_square_on_AD :
  ∃ (AB BC CD AD : ℝ),
    (∃ AB_sq BC_sq CD_sq AD_sq : ℝ,
      AB_sq = 25 ∧ BC_sq = 49 ∧ CD_sq = 64 ∧ 
      AB = Real.sqrt AB_sq ∧ BC = Real.sqrt BC_sq ∧ CD = Real.sqrt CD_sq ∧
      AD_sq = AB^2 + BC^2 + CD^2 ∧ AD = Real.sqrt AD_sq ∧ AD_sq = 138
    ) :=
by
  sorry

end area_of_square_on_AD_l712_712623


namespace angle_between_vectors_l712_712526

variables {V : Type*} [inner_product_space ℝ V] {a b : V}

theorem angle_between_vectors (h : ∥a + b∥ = ∥b∥) :
  let θ := real.arccos ((-3 * ∥b∥^2) / (∥a - 3 • b∥ * ∥b∥)) in
  ∃ θ, θ = real.arccos ((-3 * ∥b∥^2) / (∥a - 3 • b∥ * ∥b∥)) :=
begin
  sorry,
end

end angle_between_vectors_l712_712526


namespace math_problem_l712_712947

variable (a b c d : ℝ)

theorem math_problem 
    (h1 : a + b + c + d = 6)
    (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
    36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
    4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := 
by
    sorry

end math_problem_l712_712947


namespace div_count_27n5_l712_712017

theorem div_count_27n5 
  (n : ℕ) 
  (h : (120 * n^3).divisors.card = 120) 
  : (27 * n^5).divisors.card = 324 :=
sorry

end div_count_27n5_l712_712017


namespace average_points_playoff_teams_l712_712742

theorem average_points_playoff_teams :
  let p1_wins := 12 in
  let p1_ties := 4 in
  let p2_wins := 13 in
  let p2_ties := 1 in
  let p3_wins := 8 in
  let p3_ties := 10 in
  let points_per_win := 2 in
  let points_per_tie := 1 in
  let p1_points := p1_wins * points_per_win + p1_ties * points_per_tie in
  let p2_points := p2_wins * points_per_win + p2_ties * points_per_tie in
  let p3_points := p3_wins * points_per_win + p3_ties * points_per_tie in
  let total_points := p1_points + p2_points + p3_points in
  let num_playoff_teams := 3 in
  total_points / num_playoff_teams = 27 :=
by {
  sorry
}

end average_points_playoff_teams_l712_712742


namespace count_even_only_rows_in_pascals_triangle_l712_712444

theorem count_even_only_rows_in_pascals_triangle : 
  ∃ (n : ℕ), n = 4 ∧ ∀ k, 2 ≤ k ∧ k ≤ 30 → 
    (∃ i, i ∈ {2^n | n : ℕ} ∧ i = k → ∀ j, 0 < j ∧ j < i → even (nat.choose i j)) :=
by
  sorry

end count_even_only_rows_in_pascals_triangle_l712_712444


namespace num_small_boxes_l712_712682

-- Conditions
def chocolates_per_small_box := 25
def total_chocolates := 400

-- Claim: Prove that the number of small boxes is 16
theorem num_small_boxes : (total_chocolates / chocolates_per_small_box) = 16 := 
by sorry

end num_small_boxes_l712_712682


namespace find_a_l712_712076

theorem find_a (a : ℝ) :
  let l1 := λ (x y : ℝ), a * x + 3 * y - 1 = 0
  let l2 := λ (x y : ℝ), 2 * x + (a^2 - a) * y + 3 = 0
  (∀ x1 y1 x2 y2, l1 x1 y1 → l2 x2 y2 → 3 * (a^3 - a^2) + 2 * a = 0) →
  a = 0 ∨ a = 1/3 :=
by
  sorry

end find_a_l712_712076


namespace arithmetic_sequence_general_formula_range_of_m_for_Tn_l712_712033

theorem arithmetic_sequence_general_formula :
  ∀ (a : ℕ → ℝ) (d : ℝ), (d ≠ 0) →
  (∀ n, a (n + 1) = a n + d) →
  (∃ S : ℕ → ℝ, S 5 = 25 ∧ S n = (n • a 1) + ((n * (n - 1)) / 2) * d) →
  (a 1 ≠ 0) →
  (a 2 * a 2 = a 1 * a 5) →
  (∀ n, a n = 2 * n - 1) :=
by
  sorry

theorem range_of_m_for_Tn :
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (m : ℝ),
  (∀ n, a n = 2 * n - 1) →
  (∀ n, b n = 4 * n / ((a n) ^ 2 * (a (n + 1)) ^ 2)) →
  (∀ n, T n = ∑ k in finset.range n, b (k + 1)) →
  (∀ n, T n ≤ m^2 + 0.5 * m) →
  (m < -1 ∨ m ≥ 0.5) :=
by
  sorry

end arithmetic_sequence_general_formula_range_of_m_for_Tn_l712_712033


namespace determine_k_l712_712367

theorem determine_k (k : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = k - 4 * x) ↔ k = 11 :=
by
  sorry

end determine_k_l712_712367


namespace irrational_number_in_list_l712_712644

theorem irrational_number_in_list :
  ∃ x ∈ ({ 7/4, 0.3, real.sqrt 5, real.cbrt 8 } : set ℝ), irrational x ∧ 
  ∀ y ∈ ({ 7/4, 0.3, real.sqrt 5, real.cbrt 8 } : set ℝ), irrational y → y = real.sqrt 5 := 
by
  sorry

end irrational_number_in_list_l712_712644


namespace part_I_part_II_l712_712112

-- Definition of points A_n and B_n
def A_n (n : ℕ) (h : 0 < n) : ℝ × ℝ := (0, 1 / n)
def B_n (n : ℕ) (h : 0 < n) : ℝ × ℝ := let b := (1 / n ^ 2 - 2) ^ (1 / (1 - 2 * n ^ 2))
                                         (b, real.sqrt (2 * b))

-- Given conditions
axiom OA_n_eq_OB_n (n : ℕ) (h : 0 < n) : (0, 1 / n) = B_n n h

noncomputable def a_n (n : ℕ) (h : 0 < n) : ℝ := 
let (b_n, ⟨x_n, y_n⟩) := (1 / n ^ 2 - 2) ^ (1 / (1 - 2 * n ^ 2))
⟨x_n, y_n⟩

theorem part_I (n : ℕ) (h : 0 < n) : a_n n h > a_n (n + 1) h ∧ a_n (n + 1) h > 4 :=
sorry

theorem part_II : ∃ n_0 : ℕ, 0 < n_0 ∧ ∀ n : ℕ, n > n_0 → 
  ∑ i in finset.range (n - n_0 + 1), (λ i, b_n i.succ _ / b_n i _) < n - 2004 :=
sorry

end part_I_part_II_l712_712112


namespace cylindrical_to_rectangular_l712_712325

theorem cylindrical_to_rectangular (r θ z : ℝ) (h1 : r = 6) (h2 : θ = π / 3) (h3 : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, 2) := 
by 
  rw [h1, h2, h3]
  sorry

end cylindrical_to_rectangular_l712_712325


namespace true_discount_l712_712659

theorem true_discount (BG : ℝ) (Rate : ℝ) (Time : ℝ) (TD : ℝ) 
  (hBG : BG = 7.8) 
  (hRate : Rate = 12) 
  (hTime : Time = 1) 
  (hBG_eq : BG = (TD * Rate * Time) / (100 + (Rate * Time))) : TD = 72.8 :=
by
  rw [hBG, hRate, hTime, hBG_eq]
  sorry

end true_discount_l712_712659


namespace projections_of_opposite_sides_equal_l712_712176

-- The theorem statement requires us to define cyclic quadrilateral, its sides, projections and then show their equality.
variable (A B C D O P Q : Point)
variable (circle : Circle) (cyclic_quadrilateral : Quadrilateral)
variable [is_cyclic_quadrilateral : isCyclicQuadrilateral cyclic_quadrilateral]
variable (diameter_AC : isDiameter circle A C)
variable (proj_AB_on_BD : projectionLength A B D P)
variable (proj_CD_on_BD : projectionLength C D B Q)

theorem projections_of_opposite_sides_equal :
  isCyclicQuadrilateral cyclic_quadrilateral →
  isDiameter circle A C →
  projectionLength A B D P →
  projectionLength C D B Q →
  A = C → -- points A and C are endpoints of the diameter
  P = Q := sorry

end projections_of_opposite_sides_equal_l712_712176


namespace general_term_formula_l712_712797

theorem general_term_formula (f : ℕ → ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ x, f x = 1 - 2^x) →
  (∀ n, f n = S n) →
  (∀ n, S n = 1 - 2^n) →
  (∀ n, n = 1 → a n = S 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (∀ n, a n = -2^(n-1)) :=
by
  sorry

end general_term_formula_l712_712797


namespace find_side_difference_l712_712506

def triangle_ABC : Type := ℝ
def angle_B := 20
def angle_C := 40
def length_AD := 2

theorem find_side_difference (ABC : triangle_ABC) (B : ℝ) (C : ℝ) (AD : ℝ) (BC AB : ℝ) :
  B = angle_B → C = angle_C → AD = length_AD → BC - AB = 2 :=
by 
  sorry

end find_side_difference_l712_712506


namespace negation_of_universal_prop_l712_712960

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^3 + 3^x > 0) ↔ (∃ x : ℝ, x^3 + 3^x ≤ 0) :=
by sorry

end negation_of_universal_prop_l712_712960


namespace find_length_EF_l712_712502

theorem find_length_EF (DE DF : ℝ) (angle_E : ℝ) (angle_E_val : angle_E = 45) (DE_val : DE = 100) (DF_val : DF = 100 * Real.sqrt 2) : ∃ EF, EF = 141.421 :=
by
  exists 141.421
  sorry

end find_length_EF_l712_712502


namespace find_remainder_l712_712011

-- Define the numbers
def a := 98134
def b := 98135
def c := 98136
def d := 98137
def e := 98138
def f := 98139

-- Theorem statement
theorem find_remainder :
  (a + b + c + d + e + f) % 9 = 3 :=
by {
  sorry
}

end find_remainder_l712_712011


namespace sum_of_c_and_d_is_6_l712_712695

-- Definitions for the vertices of the quadrilateral
def point1 := (1, 2)
def point2 := (4, 5)
def point3 := (5, 4)
def point4 := (4, 1)

-- Function to calculate the Euclidean distance between two points
def dist (p1 p2 : ℕ × ℕ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The main theorem stating the problem
theorem sum_of_c_and_d_is_6 : 
  let c := 4,
      d := 2,
      perimeter := 4 * real.sqrt 2 + 2 * real.sqrt 10 in
  c + d = 6 :=
by
  sorry

end sum_of_c_and_d_is_6_l712_712695


namespace horner_correct_l712_712231

-- Given definitions and conditions
def P (x : ℝ) (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range (n + 1)).sum (λ k, a k * x ^ k)

-- Horner's scheme for computing P(x_0)
def horner (x_0 : ℝ) (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  let b : ℕ → ℝ → ℝ
  | 0     => a n
  | k + 1 => (b k) * x_0 + a (n - k - 1)
  in b n

-- Theorem statement to be proved
theorem horner_correct (x_0 : ℝ) (a : ℕ → ℝ) (n : ℕ) :
  horner x_0 a n = P x_0 a n :=
sorry

end horner_correct_l712_712231


namespace prove_ellipse_equation_v1_prove_ellipse_equation_v2_l712_712397

noncomputable def ellipse_equation_v1 (a b : ℝ) := (a = 3 * b) ∧ (b = 1) ∧ (a = 3) ∧ (∀ x y : ℝ, ((x^2) / 9 + y^2 = 1))

noncomputable def ellipse_equation_v2 (a b : ℝ) := (a = 3 * b) ∧ (b = 3) ∧ (a = 9) ∧ (∀ x y : ℝ, ((y^2) / 81 + (x^2) / 9 = 1))

theorem prove_ellipse_equation_v1 :
  ∃ a b : ℝ, (a = 3 * b) ∧ (b = 1) ∧ (a = 3) ∧ (∀ x y : ℝ, ((x^2) / 9 + y^2 = 1)) :=
begin
  sorry
end

theorem prove_ellipse_equation_v2 :
  ∃ a b : ℝ, (a = 3 * b) ∧ (b = 3) ∧ (a = 9) ∧ (∀ x y : ℝ, ((y^2) / 81 + (x^2) / 9 = 1)) :=
begin
  sorry
end

end prove_ellipse_equation_v1_prove_ellipse_equation_v2_l712_712397


namespace final_sum_l712_712313

-- Assuming an initial condition for the values on the calculators
def initial_values : List Int := [2, 1, -1]

-- Defining the operations to be applied on the calculators
def operations (vals : List Int) : List Int :=
  match vals with
  | [a, b, c] => [a * a, b * b * b, -c]
  | _ => vals  -- This case handles unexpected input formats

-- Applying the operations for 43 participants
def final_values (vals : List Int) (n : Nat) : List Int :=
  if n = 0 then vals
  else final_values (operations vals) (n - 1)

-- Prove that the final sum of the values on the calculators equals 2 ^ 2 ^ 43
theorem final_sum : 
  final_values initial_values 43 = [2 ^ 2 ^ 43, 1, -1] → 
  List.sum (final_values initial_values 43) = 2 ^ 2 ^ 43 :=
by
  intro h -- This introduces the hypothesis that the final values list equals the expected values
  sorry   -- Provide an ultimate proof for the statement.

end final_sum_l712_712313


namespace collinear_TKI_l712_712893

-- Definitions for points and lines
variables {A P C Q M N I T K : Type}
variable {line : Type → Type}
variables (AP : line A → line P) (CQ : line C → line Q) (MP : line M → line P) (NQ : line N → line Q)

-- Conditions from the problem
-- Assume there exist points T and K which are intersections of the specified lines
axiom intersects_AP_CQ : ∃ (T : Type), AP T = CQ T
axiom intersects_MP_NQ : ∃ (K : Type), MP K = NQ K

-- Collinearity of points T, K, and I
theorem collinear_TKI : ∀ (I : Type) (T : Type) (K : Type),
  intersects_AP_CQ → intersects_MP_NQ → collinear I T K :=
by sorry

end collinear_TKI_l712_712893


namespace fraction_problem_l712_712751

-- Definitions of x and y based on the given conditions
def x : ℚ := 3 / 5
def y : ℚ := 7 / 9

-- The theorem stating the mathematical equivalence to be proven
theorem fraction_problem : (5 * x + 9 * y) / (45 * x * y) = 10 / 21 :=
by
  sorry

end fraction_problem_l712_712751


namespace problem_I_problem_II_l712_712436

-- Definitions of the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m + 3}

-- Problem Part (I)
theorem problem_I (m : ℝ) : (A ∪ B m) = A ↔ (m ∈ Set.Iio (-2) ∪ Set.Ioo (-2, -1/2)) :=
by sorry

-- Problem Part (II)
theorem problem_II (m : ℝ) : (A ∩ B m).Nonempty ↔ (m ∈ Set.Ioo (-2, 1)) :=
by sorry

end problem_I_problem_II_l712_712436


namespace calculate_expression_l712_712315

theorem calculate_expression : 200 * 39.96 * 3.996 * 500 = (3996)^2 :=
by
  sorry

end calculate_expression_l712_712315


namespace cos_alpha_l712_712826

theorem cos_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.sin (α - π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 :=
by
  sorry

end cos_alpha_l712_712826


namespace value_of_m_l712_712101

theorem value_of_m
  (x y m : ℝ)
  (h1 : 2 * x + 3 * y = 4)
  (h2 : 3 * x + 2 * y = 2 * m - 3)
  (h3 : x + y = -3/5) :
  m = -2 :=
sorry

end value_of_m_l712_712101


namespace rhombus_diagonal_l712_712983

variable (d1 : ℝ) (area : ℝ)

theorem rhombus_diagonal (h1: d1 = 160) (h2: area = 5600) : ∃ d2, (area = (d1 * d2) / 2) ∧ d2 = 70 :=
by
  use 70
  rw h1 at *
  rw h2 at *
  sorry

end rhombus_diagonal_l712_712983


namespace find_t_max_value_of_xyz_l712_712270

-- Problem (1)
theorem find_t (t : ℝ) (x : ℝ) (h1 : |2 * x + t| - t ≤ 8) (sol_set : -5 ≤ x ∧ x ≤ 4) : t = 1 :=
sorry

-- Problem (2)
theorem max_value_of_xyz (x y z : ℝ) (h2 : x^2 + (1/4) * y^2 + (1/9) * z^2 = 2) : x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end find_t_max_value_of_xyz_l712_712270


namespace range_of_a_l712_712069

-- Define proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

-- Define the main theorem
theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) → -1 ≤ a ∧ a < 0 :=
by
  sorry

end range_of_a_l712_712069


namespace Pascal_hexagon_l712_712962

noncomputable def hexagon_collinear (A B C D E F : ℝ→ℝ) (conic : ∀ (P : ℝ→ℝ), (P = A) ∨ (P = B) ∨ (P = C) ∨ (P = D) ∨ (P = E) ∨ (P = F)) : Prop :=
  ∃ (X Y Z : ℝ→ℝ), 
    (∃ (l1 : ℝ → ℝ), l1 = line_through A B ∧ (X = intersection l1 (line_through D E))) ∧
    (∃ (l2 : ℝ → ℝ), l2 = line_through B C ∧ (Y = intersection l2 (line_through E F))) ∧
    (∃ (l3 : ℝ → ℝ), l3 = line_through C D ∧ (Z = intersection l3 (line_through A F))) ∧
    collinear [X, Y, Z]

theorem Pascal_hexagon (A B C D E F : ℝ→ℝ) (h_conic : ∀ (P : ℝ→ℝ), P ∈ {A, B, C, D, E, F} → conic P) : 
  hexagon_collinear A B C D E F conic := 
sorry

end Pascal_hexagon_l712_712962


namespace larry_daily_dog_time_l712_712141

-- Definitions from the conditions
def half_hour_in_minutes : ℕ := 30
def twice_a_day (minutes : ℕ) : ℕ := 2 * minutes
def one_fifth_hour_in_minutes : ℕ := 60 / 5

-- Hypothesis resulting from the conditions
def time_walking_and_playing : ℕ := twice_a_day half_hour_in_minutes
def time_feeding : ℕ := one_fifth_hour_in_minutes

-- The theorem to prove
theorem larry_daily_dog_time : time_walking_and_playing + time_feeding = 72 := by
  show time_walking_and_playing + time_feeding = 72
  sorry

end larry_daily_dog_time_l712_712141


namespace coupon_discount_l712_712679

/-- For the listed prices $x$ such that $200 < x < 375$, prove that coupon 1 offers a greater price reduction than either coupon 2 or coupon 3. -/
theorem coupon_discount (x : ℝ) (h1 : x > 200) (h2 : x < 375) :
  (0.15 * x > 30) ∧ (0.15 * x > 0.25 * x - 37.5) :=
by
  have h_coupon1_coupon2 : 0.15 * x > 30 := 
    by linarith
  have h_coupon1_coupon3 : 0.15 * x > 0.25 * x - 37.5 := 
    by linarith
  exact ⟨h_coupon1_coupon2, h_coupon1_coupon3⟩

end coupon_discount_l712_712679


namespace negate_exactly_one_even_l712_712953

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even :
  ¬(is_even a ∧ is_odd b ∧ is_odd c ∨ is_odd a ∧ is_even b ∧ is_odd c ∨ is_odd a ∧ is_odd b ∧ is_even c) ↔
  (is_even a ∧ is_even b ∨ is_even a ∧ is_even c ∨ is_even b ∧ is_even c ∨ is_odd a ∧ is_odd b ∧ is_odd c) := sorry

end negate_exactly_one_even_l712_712953


namespace seq_increasing_seq_inequality_seq_bound_l712_712391

-- Define the sequence {a_n} according to the given recurrence relations
def a_n : ℕ+ → ℝ
| ⟨1, _⟩ := 3 / 2 
| ⟨n + 1, h⟩ := let a_n_k := a_n ⟨n, Nat.succ_pos _⟩ in 
   (1 + 1 / 3^n) * a_n_k + 2 / (n * (n + 1))

noncomputable theory

-- 1. Prove that the sequence {a_n} is strictly increasing
theorem seq_increasing : ∀ n : ℕ+, a_n ⟨n.succ, n.prop⟩ > a_n ⟨n.val, Nat.succ_pos n.val⟩ :=
sorry

-- 2. Prove that for n ≥ 2,
theorem seq_inequality (n : ℕ+) (h : n ≥ 2) : 
  a_n ⟨n.val + 1, n.prop⟩ / a_n ⟨n.val, Nat.succ_pos _⟩ ≤ 
  1 + 1 / 3^n.val + 2 / (3 * n.val * (n.val + 1)) :=
sorry

-- 3. Prove that ∀ n, a_n < 3 * sqrt e
theorem seq_bound : ∀ n : ℕ+, a_n n < 3 * Real.sqrt Real.exp :=
sorry

end seq_increasing_seq_inequality_seq_bound_l712_712391


namespace unit_vector_parallel_to_a_l712_712219

theorem unit_vector_parallel_to_a (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 12 * y = 5 * x) :
  (x = 12 / 13 ∧ y = 5 / 13) ∨ (x = -12 / 13 ∧ y = -5 / 13) := by
  sorry

end unit_vector_parallel_to_a_l712_712219


namespace tangency_points_locus_l712_712516

noncomputable def cyclic_quadrilateral (P Q R S : Point) : Prop :=
  ∃ O : Point, Circle O P ∧ Circle O Q ∧ Circle O R ∧ Circle O S

noncomputable def not_parallel (PQ RS : Line) : Prop :=
  ¬parallel PQ RS

def set_of_circles_through (P Q : Point) : Set Circle :=
  {σ | σ.contains P ∧ σ.contains Q}

def set_of_points_of_tangency (circles1 circles2 : Set Circle) : Set Point :=
  {A | ∃ σ1 ∈ circles1, ∃ σ2 ∈ circles2, tangent σ1 σ2 A}

theorem tangency_points_locus
  (P Q R S : Point) (PQ RS : Line)
  (h1 : cyclic_quadrilateral P Q R S)
  (h2 : not_parallel PQ RS) :
  let X := intersection_point PQ RS in
  ∃ r : Real, set_of_points_of_tangency (set_of_circles_through P Q) (set_of_circles_through R S) =
  {A | ∃ C : Circle, Circle.center C = X ∧ distance X A = r} := sorry

end tangency_points_locus_l712_712516


namespace expected_value_coins_l712_712291

theorem expected_value_coins : 
  let p := 1/2 in
  let v_penny := 1 in
  let v_fifty := 50 in
  let v_dime := 10 in
  let v_quarter := 25 in
  p * v_penny + p * v_fifty + p * v_dime + p * v_quarter = 43 := 
by 
  let p := 1/2
  let v_penny := 1
  let v_fifty := 50
  let v_dime := 10
  let v_quarter := 25
  calc
    (p * v_penny) + (p * v_fifty) + (p * v_dime) + (p * v_quarter)
      = (1/2 * 1) + (1/2 * 50) + (1/2 * 10) + (1/2 * 25) : by rfl
  ... = 0.5 + 25 + 5 + 12.5 : by norm_num
  ... = 43 : by norm_num

end expected_value_coins_l712_712291


namespace cos_value_of_tan_third_quadrant_l712_712780

theorem cos_value_of_tan_third_quadrant (x : ℝ) (h1 : Real.tan x = 4 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -3 / 5 := 
sorry

end cos_value_of_tan_third_quadrant_l712_712780


namespace infinite_danish_numbers_l712_712693

-- Definitions translated from problem conditions
def is_danish (n : ℕ) : Prop :=
  ∃ k, n = 3 * k ∨ n = 2 * 4 ^ k

theorem infinite_danish_numbers :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, is_danish n ∧ is_danish (2^n + n) := sorry

end infinite_danish_numbers_l712_712693


namespace selection_test_arrangements_l712_712495

/-- The number of selection test arrangements given the constraints is 36. -/
theorem selection_test_arrangements : 
  let tests := ["vestibular_function", "overweight_endurance", "weightlessness_flight", "flight_parachute_jumping", "landing_impact"] in
  ∃ (arrangements : List (List String)),
    (∀ (arrangement : List String), arrangement ∈ arrangements → 
     arrangement.length = 5 ∧ 
     ("vestibular_function" ∈ arrangement ∧ "weightlessness_flight" ∈ arrangement ∧ 
      ("vestibular_function" = get (nth arrangement (succ (index_of "weightlessness_flight" arrangement))) arrangement ∨ 
       "weightlessness_flight" = get (nth arrangement (succ (index_of "vestibular_function" arrangement))) arrangement)) ∧ 
     (∀ i, arrangement.nth i = some "overweight_endurance" → 
       ∀ j, arrangement.nth j = some "weightlessness_flight" → (i = j + 1 ∨ j = i + 1) → false)) ∧ 
    arrangements.length = 36 := 
sorry

end selection_test_arrangements_l712_712495


namespace juvy_chives_plants_l712_712130

theorem juvy_chives_plants : 
  let total_rows := 60 in
  let plants_per_row := 18 in
  let parsley_rows := (3 / 5) * total_rows in
  let rosemary_rows := (1 / 4) * total_rows in
  let remaining_after_parsley_rosemary := total_rows - (parsley_rows + rosemary_rows) in
  let mint_rows := (1 / 6) * remaining_after_parsley_rosemary in
  let remaining_after_mint := remaining_after_parsley_rosemary - mint_rows in
  let thyme_rows := (7 / 12) * remaining_after_mint in
  let remaining_after_thyme := remaining_after_mint - thyme_rows in
  let basil_rows := (2 / 3) * remaining_after_thyme in
  let remaining_after_basil := remaining_after_thyme - basil_rows in
  remaining_after_basil * plants_per_row = 36 :=
by
  sorry

end juvy_chives_plants_l712_712130


namespace three_digit_sum_6_l712_712819

theorem three_digit_sum_6 : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 6) ↔ 
  (∃ s : Finset (ℕ × ℕ × ℕ), 
    s.card = 21 ∧ 
    (∀ ⟨a, b, c⟩ ∈ s, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 6)) :=
sorry

end three_digit_sum_6_l712_712819


namespace equidistant_point_position_l712_712102

/-- 
If there exists a point in the plane such that its distances to the four vertices of 
a convex quadrilateral are equal, prove that this point could be inside, outside, or 
on the boundary of the quadrilateral.
-/
theorem equidistant_point_position (P A B C D : Point) (h1 : ConvexQuadrilateral A B C D) 
(h2 : (dist P A = dist P B) ∧ (dist P B = dist P C) ∧ (dist P C = dist P D)) :
  (is_inside P A B C D) ∨ (is_outside P A B C D) ∨ (is_on_boundary P A B C D) :=
sorry

end equidistant_point_position_l712_712102


namespace spheres_volume_ratio_l712_712460

theorem spheres_volume_ratio (r1 r2 : ℝ) (h : (4 * real.pi * r1^2) / (4 * real.pi * r2^2) = 4 / 9) :
    (4 / 3 * real.pi * r1^3) / (4 / 3 * real.pi * r2^3) = 8 / 27 :=
by
  -- We do not include the proof here
  sorry

end spheres_volume_ratio_l712_712460


namespace round_to_two_significant_figures_l712_712714

theorem round_to_two_significant_figures : 
  ∀ {x : ℝ}, x = 0.0984 → round_to_sig_fig x 2 = 0.098 :=
by
  intro x h
  -- Definitions of the rounding function and rules as per the conditions.
  sorry

end round_to_two_significant_figures_l712_712714


namespace find_values_l712_712404

theorem find_values (x : ℝ) (h : 2 * Real.cos x - 5 * Real.sin x = 3) :
  3 * Real.sin x + 2 * Real.cos x = ( -21 + 13 * Real.sqrt 145 ) / 58 ∨
  3 * Real.sin x + 2 * Real.cos x = ( -21 - 13 * Real.sqrt 145 ) / 58 := sorry

end find_values_l712_712404


namespace parallel_vectors_magnitude_l712_712812

open Real

noncomputable def vec_magnitude (v : ℝ × ℝ) : ℝ :=
  (v.1^2 + v.2^2).sqrt

theorem parallel_vectors_magnitude :
  ∀ (x : ℝ), (∃ k : ℝ, (1, 2) = k • (x, -4)) →
  vec_magnitude ((1, 2) + (x, -4)) = sqrt 5 := 
by
  intros x h
  sorry

end parallel_vectors_magnitude_l712_712812


namespace expression_meaningful_l712_712097

theorem expression_meaningful (x : ℝ) : (∃ y, y = 1 / (real.sqrt (x - 3))) ↔ x > 3 :=
by sorry

end expression_meaningful_l712_712097


namespace probability_of_victory_l712_712482

theorem probability_of_victory (p_A p_B : ℝ) (h_A : p_A = 0.3) (h_B : p_B = 0.6) (independent : true) :
  p_A * p_B = 0.18 :=
by
  -- placeholder for proof
  sorry

end probability_of_victory_l712_712482


namespace journey_distance_last_day_l712_712489

theorem journey_distance_last_day (S₆ : ℕ) (q : ℝ) (n : ℕ) (a₁ : ℝ) : 
  S₆ = 378 ∧ q = 1 / 2 ∧ n = 6 ∧ S₆ = a₁ * (1 - q^n) / (1 - q)
  → a₁ * q^(n - 1) = 6 :=
by
  intro h
  sorry

end journey_distance_last_day_l712_712489


namespace pentagon_area_pq_l712_712590

noncomputable def area_pentagon_FGHIJ (length : ℕ) := 4.5 + (9 * Real.sqrt(3) / 4)

theorem pentagon_area_pq {length : ℕ} (h_length : length = 3) 
  (p q : ℕ) (h_area : area_pentagon_FGHIJ length = Real.sqrt p + Real.sqrt q) :
  p + q = 29 :=
by
  sorry

end pentagon_area_pq_l712_712590


namespace quadrilateral_perimeter_l712_712677

theorem quadrilateral_perimeter
  (WXYZ : Set Point)
  (convex_quadrilateral : ConvexHull WXYZ)
  (area_WXYZ : area WXYZ = 2500)
  (Q : Point)
  (W X Y Z : Point)
  (W_in_WXYZ : W ∈ WXYZ)
  (X_in_WXYZ : X ∈ WXYZ)
  (Y_in_WXYZ : Y ∈ WXYZ)
  (Z_in_WXYZ : Z ∈ WXYZ)
  (Q_in_interior : Q ∈ interior WXYZ)
  (WQ : distance W Q = 30)
  (XQ : distance X Q = 40)
  (YQ : distance Y Q = 35)
  (ZQ : distance Z Q = 50) :
  perimeter WXYZ = 222 :=
sorry

end quadrilateral_perimeter_l712_712677


namespace angle_between_IK_and_CL_l712_712034

variables {A B C K L I : Type} [IsoscelesTriangle A B C]
(GreaterThan : Segment B K) 

def incircle_at_I : Triangle A B K → Point I := sorry  -- Assume these functions exist
def tangent_circle_at_B : Point B → Circle := sorry   -- Assume these functions exist
def intersects_segment_at_L : Circle → Segment B K → Point L := sorry  -- Assume this function exists

theorem angle_between_IK_and_CL (h₁ : IsoscelesTriangle A B C)
(h₂ : K ∈ extension AC)
(h₃ : incircle_at_I (Triangle A B K) I)
(h₄ : tangent_circle_at_B B)
(h₅ : intersects_segment_at_L (tangent_circle_at_B B) (Segment B K) L):
angle IK CL = 90 := by
  sorry

end angle_between_IK_and_CL_l712_712034


namespace decimal_equivalent_of_fraction_l712_712680

theorem decimal_equivalent_of_fraction :
  (16 : ℚ) / 50 = 32 / 100 :=
by sorry

end decimal_equivalent_of_fraction_l712_712680


namespace exists_odd_prime_and_positive_k_l712_712754

def norm_distance (x : ℝ) : ℝ :=
  abs (floor (x + 1/2) - x)

theorem exists_odd_prime_and_positive_k (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (p : ℕ) (hp : Fact (Nat.Prime p)) (odd_p : p % 2 = 1) (k : ℕ) (hk : 0 < k),
  norm_distance (a / (p^k : ℝ)) + norm_distance (b / (p^k : ℝ)) + norm_distance ((a + b) / (p^k : ℝ)) = 1 :=
by
  sorry

end exists_odd_prime_and_positive_k_l712_712754


namespace calculate_expression_l712_712079

theorem calculate_expression (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  (4 * (x^y * (7^y * 24^x)) / (x * y) + 5 * (x * (13^y * 15^x)) - 2 * (y * (6^x * 28^y)) + 7 * (x * y * (3^x * 19^y)) / (x + y)) = 11948716.8 :=
by
  subst hx
  subst hy
  norm_num
  -- sorry to skip the proof
  sorry

end calculate_expression_l712_712079


namespace median_length_l712_712634

theorem median_length :
  ∀ (D E F M: Type) [metric_space D] [metric_space E] [metric_space F] [metric_space M],
  ∀ (DE DF EF DM EM : ℝ), DE = 13 → DF = 13 → EF = 14 → DM = 2 * real.sqrt 30 → 
  let M : E × F := (E, 7) in -- M is midpoint of EF with each segment being 7 units.
  let EM^2 := (DE^2 - EM^2) :=
  DM ^ 2 = EM^2 :=
begin
  sorry
end

end median_length_l712_712634


namespace mike_spent_on_baseball_l712_712169

theorem mike_spent_on_baseball:
  ∀ (cost_marbles cost_football total_spent cost_baseball : ℝ),
    cost_marbles = 9.05 →
    cost_football = 4.95 →
    total_spent = 20.52 →
    total_spent = cost_marbles + cost_football + cost_baseball →
    cost_baseball = 6.52 :=
by
  intro cost_marbles cost_football total_spent cost_baseball
  intros hcm hcf hts heq
  rw [hcm, hcf, hts] at heq
  linarith

end mike_spent_on_baseball_l712_712169


namespace largest_divisor_of_consecutive_odd_integers_l712_712332

theorem largest_divisor_of_consecutive_odd_integers :
  ∀ (x : ℤ), (∃ (d : ℤ) (m : ℤ), d = 48 ∧ (x * (x + 2) * (x + 4) * (x + 6)) = d * m) :=
by 
-- We assert that for any integer x, 48 always divides the product of
-- four consecutive odd integers starting from x
sorry

end largest_divisor_of_consecutive_odd_integers_l712_712332


namespace simplify_and_raise_eq_l712_712564

noncomputable def simplify_and_raise (z : ℂ) (n : ℕ) : ℂ :=
  z ^ n

theorem simplify_and_raise_eq :
  simplify_and_raise ((2 + complex.I) / (2 - complex.I)) 200 =
    complex.cos (200 * real.arctan (4 / 3)) +
    complex.I * complex.sin (200 * real.arctan (4 / 3)) :=
by
  sorry

end simplify_and_raise_eq_l712_712564


namespace color_balls_l712_712015

open Nat

theorem color_balls (n : ℕ) (hn : n > 0) (boxes : Fin n → Fin 5 → Fin n) :
  ∃ (red blue : Fin n → Fin 5 → Fin n) (hr : ∀ i j, boxes i j ∈ red i j ∨ boxes i j ∈ blue i j),
  (∑ i in Finset.range n, (∑ j in Finset.range 5, if boxes i j ∈ red i j then boxes i j else 0)) = 
  2 * (∑ i in Finset.range n, (∑ j in Finset.range 5, if boxes i j ∈ blue i j then boxes i j else 0)) :=
by sorry

end color_balls_l712_712015


namespace find_f2_l712_712771

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
noncomputable def a : ℝ := sorry

axiom odd_f : ∀ x, f (-x) = -f x
axiom even_g : ∀ x, g (-x) = g x
axiom fg_eq : ∀ x, f x + g x = a^x - a^(-x) + 2
axiom g2_a : g 2 = a
axiom a_pos : a > 0
axiom a_ne1 : a ≠ 1

theorem find_f2 : f 2 = 15 / 4 := 
by sorry

end find_f2_l712_712771


namespace extreme_value_point_of_f_in_interval_l712_712589

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - (1 / 4) * x ^ 4

theorem extreme_value_point_of_f_in_interval : (0 < x ∧ x < 3) → (x = 1) ∨ ¬ (x = 1) :=
begin
  sorry
end

end extreme_value_point_of_f_in_interval_l712_712589


namespace hyperbola_equation_l712_712413

noncomputable def sqrt_cubed := Real.sqrt 3

theorem hyperbola_equation
  (P : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (hP : P = (1, sqrt_cubed))
  (hAsymptote : (1 / a)^2 - (sqrt_cubed / b)^2 = 0)
  (hAngle : ∀ F : ℝ × ℝ, ∀ O : ℝ × ℝ, (F.1 - 1)^2 + (F.2 - sqrt_cubed)^2 + F.1^2 + F.2^2 = 16) :
  (a^2 = 4) ∧ (b^2 = 12) ∧ (c = 4) →
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 :=
by
  sorry

end hyperbola_equation_l712_712413


namespace first_player_wins_l712_712175

-- Define the initial condition of the game
structure Game where
  board : Array (Array (Option Char)) -- Represent the game board as an array of arrays
  turn : Char -- Whose turn to play, 'W' for White, 'B' for Black

def initialBoard : Array (Array (Option Char)) := 
  #[#[some 'B', some '?', some '?', some '?', some '?', some '?', some '?', 
      some '?', some '?', some '?', some '?', some '?', some '?', some '?', some 'B' ],
   . . . -- Define rows 2 to 14 all containing '?'
   #[some 'W', some '?', some '?', some '?', some '?', some '?', some '?', 
      some '?', some '?', some '?', some '?', some '?', some '?', some '?', some 'W' ]]

def initialGame : Game := 
  { board := initialBoard, turn := 'W' }

-- Define the move ability and game rules
def validMove (g : Game) (r c new_r : Nat): Bool :=
  if r < 0 ∨ r > 14 ∨ c < 0 ∨ c > 14 ∨ new_r < 0 ∨ new_r > 14 then
    false
  else match g.board[r][c], g.board[new_r][c] with
    | some 'W', none => g.turn = 'W' ∧ new_r < r -- White can only move up
    | some 'B', none => g.turn = 'B' ∧ new_r > r -- Black can only move down
    | _, _ => false
  end

def makeMove (g : Game) (r c new_r : Nat): Game :=
  { board := g.board.set! new_r (g.board[new_r].set! c g.board[r][c])
                        .set! r (g.board[r].set! c none),
    turn := if g.turn = 'W' then 'B' else 'W' }

-- Define a strategy for the first player to win
def winningStrategy (g : Game) : Bool := sorry -- Complex strategy proof omitted for brevity

theorem first_player_wins (g : Game) (g = initialGame) : ∃ s, winningStrategy s :=
  sorry

end first_player_wins_l712_712175


namespace probability_defective_correct_l712_712841

noncomputable def probability_defective_item : ℝ :=
let P_H1 := 0.4 in
let P_H2 := 0.6 in
let P_A_given_H1 := 0.03 in
let P_A_given_H2 := 0.02 in
P_H1 * P_A_given_H1 + P_H2 * P_A_given_H2

theorem probability_defective_correct : probability_defective_item = 0.024 :=
by
  sorry

end probability_defective_correct_l712_712841


namespace change_order_of_integration_l712_712321

variable {f : ℝ → ℝ → ℝ}

theorem change_order_of_integration :
  (∫ y in 0..1, ∫ x in 0..(real.sqrt y), f x y) + 
  (∫ y in 1..real.sqrt 2, ∫ x in 0..(real.sqrt (2 - y^2)), f x y) =
  ∫ x in 0..1, ∫ y in x^2..(real.sqrt (2 - x^2)), f x y :=
sorry

end change_order_of_integration_l712_712321


namespace smallest_sum_of_squares_l712_712992

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l712_712992


namespace f_log4_9_eq_neg_one_third_l712_712410

def f (x : ℝ) : ℝ := if x < 0 then 2^x else -(2^(-x))

theorem f_log4_9_eq_neg_one_third :
  f (Real.log 9 / Real.log 4) = -1 / 3 :=
by
  sorry

end f_log4_9_eq_neg_one_third_l712_712410


namespace area_of_triangle_l712_712349

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4 / x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x + 4 / x^2

-- Define the point where the tangent line is considered
def point := (1 : ℝ, f 1)

-- Define the formula for the tangent line at the given point
def tangent_line (x : ℝ) : ℝ := 6 * x - 9

-- Proof of the area of the triangle formed by the tangent line and coordinate axes
theorem area_of_triangle : (1 / 2) * (3 / 2) * 9 = 27 / 4 :=
by
  sorry

end area_of_triangle_l712_712349


namespace distinct_special_sums_l712_712320

def is_special_fraction (a b : ℕ) : Prop := a + b = 18

def is_special_sum (n : ℤ) : Prop :=
  ∃ (a1 b1 a2 b2 : ℕ), is_special_fraction a1 b1 ∧ is_special_fraction a2 b2 ∧ 
  n = (a1 : ℤ) * (b2 : ℤ) * b1 + (a2 : ℤ) * (b1 : ℤ) / a1

theorem distinct_special_sums : 
  (∃ (sums : Finset ℤ), 
    (∀ n, n ∈ sums ↔ is_special_sum n) ∧ 
    sums.card = 7) :=
sorry

end distinct_special_sums_l712_712320


namespace min_value_of_quadratic_fun_min_value_is_reached_l712_712778

theorem min_value_of_quadratic_fun (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1) :
  (3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 ≥ (15 / 782)) :=
sorry

theorem min_value_is_reached (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1)
  (h2 : 3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 = (15 / 782)) :
  true :=
sorry

end min_value_of_quadratic_fun_min_value_is_reached_l712_712778


namespace slope_of_intersection_line_l712_712729

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 40 = 0

-- Define the statement to prove the slope of the line through the intersection points of the circles
theorem slope_of_intersection_line : (∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → (-5 / 2)) :=
  sorry

end slope_of_intersection_line_l712_712729


namespace find_m_value_l712_712014

-- Defining the hyperbola equation and the conditions
def hyperbola_eq (x y : ℝ) (m : ℝ) : Prop :=
  (x^2 / m) - (y^2 / 4) = 1

-- Definition of the focal distance
def focal_distance (c : ℝ) :=
  2 * c = 6

-- Definition of the relationship c^2 = a^2 + b^2 for hyperbolas
def hyperbola_focal_distance_eq (m : ℝ) (c b : ℝ) : Prop :=
  c^2 = m + b^2

-- Stating that the hyperbola has the given focal distance
def given_focal_distance : Prop :=
  focal_distance 3

-- Stating the given condition on b²
def given_b_squared : Prop :=
  4 = 4

-- The main theorem stating that m = 5 given the conditions.
theorem find_m_value (m : ℝ) : 
  (hyperbola_eq 1 1 m) → given_focal_distance → given_b_squared → m = 5 :=
by
  sorry

end find_m_value_l712_712014


namespace eval_imaginary_powers_l712_712343

theorem eval_imaginary_powers :
  let i : ℂ := complex.I in
  i ^ 18 + i ^ 28 + i ^ (-32) = 1 :=
by
  let i : ℂ := complex.I
  sorry

end eval_imaginary_powers_l712_712343


namespace sum_cubes_zero_l712_712346

theorem sum_cubes_zero : 
  (∑ i in Finset.range 51, (i : ℤ)^3) + (∑ i in Finset.range 51, (-i : ℤ)^3) = 0 :=
by
  sorry

end sum_cubes_zero_l712_712346


namespace Ben_more_new_shirts_than_Joe_l712_712708

theorem Ben_more_new_shirts_than_Joe :
  ∀ (alex_shirts joe_shirts ben_shirts : ℕ),
    alex_shirts = 4 →
    joe_shirts = alex_shirts + 3 →
    ben_shirts = 15 →
    ben_shirts - joe_shirts = 8 :=
by
  intros alex_shirts joe_shirts ben_shirts
  intros h_alex h_joe h_ben
  sorry

end Ben_more_new_shirts_than_Joe_l712_712708


namespace divisor_of_number_l712_712290

theorem divisor_of_number : 
  ∃ D, 
    let x := 75 
    let R' := 7 
    let Q := R' + 8 
    x = D * Q + 0 :=
by
  sorry

end divisor_of_number_l712_712290


namespace ball_box_assignment_l712_712772

theorem ball_box_assignment :
  let balls := {1, 2, 3, 4, 5}
  let boxes := {1, 2, 3, 4, 5}
  (∃ f : balls → boxes, (∀ b ∈ balls, ∃! b' ∈ boxes, f b' = b) ∧
  (∀ b₁ b₂ ∈ balls, f b₁ ≠ b₁ → f b₂ ≠ b₂ → b₁ ≠ b₂)) →
  (∃! p : Set (balls → boxes), card p = 1 ∧ (∀ q ∈ p, q = f)) :=
begin
  let count_match := 5, -- Selecting one ball to match its box
  let derangement_four := 9, -- Number of derangements of 4 objects
  have : count_match * derangement_four = 45, by norm_num,
  exact this
end

end ball_box_assignment_l712_712772


namespace rational_m_sign_values_l712_712408

theorem rational_m_sign_values (a b : ℚ) (h : a * b ≠ 0) : 
  let M := (|a| / a) + (b / |b|)
  in M = 0 ∨ M = 2 ∨ M = -2 :=
by
  sorry

end rational_m_sign_values_l712_712408


namespace BG_length_l712_712571

-- Define the conditions and the coordinate system:
noncomputable def point : Type := ℝ × ℝ
def A : point := (0, 0)
def B : point := (0, real.sqrt 12)
def C : point := (2, 0)
def F : point := (1, real.sqrt 3)
def line_AF (x : ℝ) : ℝ := real.sqrt 3 * x
def G : point := (2, line_AF 2)

-- Define the distance formula
def distance (p q : point) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

-- State the proof problem
theorem BG_length :
  distance B G = 2 :=
by sorry

end BG_length_l712_712571


namespace points_on_parabola_l712_712365

theorem points_on_parabola (t : ℝ) : 
  ∃ a b c : ℝ, ∀ (x y: ℝ), (x, y) = (Real.cos t ^ 2, Real.sin (2 * t)) → y^2 = 4 * x - 4 * x^2 := 
by
  sorry

end points_on_parabola_l712_712365


namespace nathan_and_parents_total_cost_l712_712958

/-- Define the total number of people -/
def num_people := 3

/-- Define the cost per object -/
def cost_per_object := 11

/-- Define the number of objects per person -/
def objects_per_person := 2 + 2 + 1

/-- Define the total number of objects -/
def total_objects := num_people * objects_per_person

/-- Define the total cost -/
def total_cost := total_objects * cost_per_object

/-- The main theorem to prove the total cost -/
theorem nathan_and_parents_total_cost : total_cost = 165 := by
  sorry

end nathan_and_parents_total_cost_l712_712958


namespace Randy_trip_distance_l712_712182

noncomputable def total_distance (x : ℝ) :=
  (x / 4) + 40 + 10 + (x / 6)

theorem Randy_trip_distance (x : ℝ) (h : total_distance x = x) : x = 600 / 7 :=
by
  sorry

end Randy_trip_distance_l712_712182


namespace max_modulus_l712_712048

theorem max_modulus (z : ℂ) (h : |z - 2 * complex.I| = 1) : |z| ≤ 3 :=
begin
  sorry
end

end max_modulus_l712_712048


namespace collinear_T_K_I_l712_712905

noncomputable def T (A P C Q : Point) : Point := intersection (line_through A P) (line_through C Q)
noncomputable def K (M P N Q : Point) : Point := intersection (line_through M P) (line_through N Q)

theorem collinear_T_K_I (A P C Q M N I : Point) :
  collinear [T A P C Q, K M P N Q, I] :=
sorry

end collinear_T_K_I_l712_712905


namespace min_value_of_a_plus_b_l712_712402

open Real

theorem min_value_of_a_plus_b (a b : ℝ) (h : log 2 a + log 2 b = -2) : a + b ≥ 1 :=
by
  sorry

end min_value_of_a_plus_b_l712_712402


namespace secret_sharing_l712_712955

theorem secret_sharing :
  ∃ n : ℕ, (n = 7) ∧ (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 = 2186) :=
begin
  use 7,
  split,
  { refl, },
  {
    calc 
    1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 
    = (3^8 - 1) / 2 : by norm_num
    ... = 2186 : by norm_num
  }
end

example : 1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 = 2186 := by norm_num -- verify series sum

end secret_sharing_l712_712955


namespace geometry_problem_l712_712104

-- Define the geometric entities involved
variables {A B C I D E F : Type}
variables {isIncenter : is_incenter I A B C}
variables {ID_perp_BC : perp I D B C}
variables {BE_perp_AI : perp B E A I}
variables {CF_perp_AI : perp C F A I}

-- Define the segments involved
variables {BD DC BE CF : ℝ}

-- Define the conditions on the segments
variables {BD_def : BD = segment_length B D}
variables {DC_def : DC = segment_length D C}
variables {BE_def : BE = segment_length B E}
variables {CF_def : CF = segment_length C F}

-- The theorem to prove
theorem geometry_problem (h1 : is_incenter)
                         (h2 : ID_perp_BC)
                         (h3 : BE_perp_AI)
                         (h4 : CF_perp_AI) :
                         BD * DC = BE * CF :=
begin
  sorry
end

end geometry_problem_l712_712104


namespace range_of_m_l712_712799

noncomputable def f (x : ℝ) : ℝ := 2^|x|

theorem range_of_m (m : ℝ) (hm : f (real.logb 2 m) > f 2) : (0 < m ∧ m < 1 / 4) ∨ (4 < m) :=
by
  sorry

end range_of_m_l712_712799


namespace continuous_values_l712_712339

theorem continuous_values (a b : ℝ) (f : ℝ → ℝ) :
  f = (λ x, (x^4 + a * x^2 + b) / (x^2 - 4)) →
  (∀ (x : ℝ), (x = 2 → continuity_at f 2) ∧ (x = -2 → continuity_at f (-2))) →
  b = -16 - 4 * a := by
  intros hf hc
  sorry

end continuous_values_l712_712339


namespace eccentricity_of_ellipse_l712_712943

open Real

-- Define ellipse and its properties
def ellipse (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions for focus and point relationship on ellipse
def point_on_ellipse_conditions (a b x : ℝ) (h : a > b > 0) (pf2 f1f2 angle : ℝ) :=
  angle = 30 * (π / 180) ∧
  pf2.perpendicular f1f2 ∧
  |PF_1| = 2 * |PF_2| ∧
  |F_1F_2| = √3 * |PF_2|

-- Eccentricity calculation
def calc_eccentricity (a c : ℝ) : ℝ := c / a

-- The main theorem
theorem eccentricity_of_ellipse (h : a > b > 0) (P_on_ellipse : ∃ (x y : ℝ), ellipse a b x y):
  point_on_ellipse_conditions a b x h -> 
  calc_eccentricity a (\sqrt{3} * a / 3) = \sqrt{3} / 3 :=
by sorry

end eccentricity_of_ellipse_l712_712943


namespace find_a_when_xaxis_is_tangent_l712_712065

theorem find_a_when_xaxis_is_tangent
  (a : ℝ)
  (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^3 + a * x + (1 / 4))
  (h₂ : ∃ m : ℝ, f m = 0 ∧ (∀ x, f' m = 0)) :
  a = - (3 / 4) :=
sorry

end find_a_when_xaxis_is_tangent_l712_712065


namespace chloe_wins_probability_l712_712712

theorem chloe_wins_probability :
  let p := 1 / 6
  let q := 5 / 6
  (∑' (n: ℕ), q^(3*n) * q^(-1) * p) = 25 / 91 :=
by
  sorry

end chloe_wins_probability_l712_712712


namespace seven_not_factor_l712_712177

noncomputable def P : ℕ := List.prod (List.firstn 1001 Nat.primes)

def is_factor (a b : ℕ) : Prop := ∃ k, b = a * k

theorem seven_not_factor :
  ¬ is_factor 7007 P :=
by sorry

end seven_not_factor_l712_712177


namespace f_iterated_l712_712362

noncomputable def f (z : ℂ) : ℂ :=
  if (∃ r : ℝ, z = r) then -z^2 else z^2

theorem f_iterated (z : ℂ) (h : z = 2 + 1 * complex.i) : 
  f (f (f (f z))) = 164833 + 354192 * complex.i :=
by 
  subst h
  sorry

end f_iterated_l712_712362


namespace product_divisible_by_10_l712_712309

noncomputable def probability_divisible_by_10 (n : ℕ) (h : n > 1) : ℝ :=
  1 - (8^n + 5^n - 4^n) / 9^n

theorem product_divisible_by_10 (n : ℕ) (h : n > 1) :
  probability_divisible_by_10 n h = 1 - (8^n + 5^n - 4^n)/(9^n) :=
by
  sorry

end product_divisible_by_10_l712_712309


namespace angle_PSR_measure_l712_712857

theorem angle_PSR_measure
  (P Q R S : Type)
  (h1 : ∀ (PQR : Triangle), ∠ PQR = 120)
  (h2 : ∀ (QPS RPS QRS PRS : Angle), QPS = RPS ∧ QRS = PRS)
  (h3 : ∀ (SP SR : Line), bisects SP ∠QPR ∧ bisects SR ∠QRP)
  : ∠ PSR = 150 := 
by
  sorry

end angle_PSR_measure_l712_712857


namespace inclination_angle_l712_712803

theorem inclination_angle (α : ℝ) :
  (∃ t1 t2 : ℝ, t1 + t2 = 2 * real.sin α ∧ t1 * t2 = -3 ∧ (t1 - t2).abs = real.sqrt 15) →
  (α = π / 3 ∨ α = 2 * π / 3) :=
by
  sorry

end inclination_angle_l712_712803


namespace part1_purchase_price_part2_minimum_A_l712_712254

section
variables (x y m : ℝ)

-- Part 1: Purchase price per piece
theorem part1_purchase_price (h1 : 10 * x + 15 * y = 3600) (h2 : 25 * x + 30 * y = 8100) :
  x = 180 ∧ y = 120 :=
sorry

-- Part 2: Minimum number of model A bamboo mats
theorem part2_minimum_A (h3 : x = 180) (h4 : y = 120) 
    (h5 : (260 - x) * m + (180 - y) * (60 - m) ≥ 4400) : 
  m ≥ 40 :=
sorry
end

end part1_purchase_price_part2_minimum_A_l712_712254


namespace find_operation_l712_712609

theorem find_operation (a b : ℝ) (op : ℝ → ℝ → ℝ) (h_add : a = 0.137) (h_bdd : b = 0.098)
  (h_eq : \frac{(a + b)^2 - (a - b)^2}{op a b} = 4)
  : op = has_mul.mul := sorry

end find_operation_l712_712609


namespace total_lives_after_third_level_l712_712475

def initial_lives : ℕ := 2

def extra_lives_first_level : ℕ := 6
def modifier_first_level (lives : ℕ) : ℕ := lives / 2

def extra_lives_second_level : ℕ := 11
def challenge_second_level (lives : ℕ) : ℕ := lives - 3

def reward_third_level (lives_first_two_levels : ℕ) : ℕ := 2 * lives_first_two_levels

theorem total_lives_after_third_level :
  let lives_first_level := modifier_first_level extra_lives_first_level
  let lives_after_first_level := initial_lives + lives_first_level
  let lives_second_level := challenge_second_level extra_lives_second_level
  let lives_after_second_level := lives_after_first_level + lives_second_level
  let total_gained_lives_first_two_levels := lives_first_level + lives_second_level
  let third_level_reward := reward_third_level total_gained_lives_first_two_levels
  lives_after_second_level + third_level_reward = 35 :=
by
  sorry

end total_lives_after_third_level_l712_712475


namespace initial_salt_concentration_l712_712704

theorem initial_salt_concentration (x : ℝ) (C : ℝ) :
  x = 149.99999999999994 →
  (C * x + 20) / (3/4 * x + 10 + 20) = 1/3 →
  C ≈ 0.18333333333333332 :=
by
  intros hx hc
  simp only [hx] at hc
  sorry

end initial_salt_concentration_l712_712704


namespace sin_tan_sum_l712_712794

theorem sin_tan_sum (α : ℝ) (h : ∃ (x y : ℝ), (x, y) = (4, -3) ∧ x * x + y * y = 25) :
  sin α + tan α = -27 / 20 :=
sorry

end sin_tan_sum_l712_712794


namespace simplify_and_raise_eq_l712_712563

noncomputable def simplify_and_raise (z : ℂ) (n : ℕ) : ℂ :=
  z ^ n

theorem simplify_and_raise_eq :
  simplify_and_raise ((2 + complex.I) / (2 - complex.I)) 200 =
    complex.cos (200 * real.arctan (4 / 3)) +
    complex.I * complex.sin (200 * real.arctan (4 / 3)) :=
by
  sorry

end simplify_and_raise_eq_l712_712563


namespace problem_statement_l712_712782

noncomputable def a : ℝ := 13 / 2
noncomputable def b : ℝ := -4

theorem problem_statement :
  ∀ k : ℝ, ∃ x : ℝ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6 ↔ x = 1 :=
by
  sorry

end problem_statement_l712_712782


namespace length_of_AD_l712_712562

noncomputable def side_lengths (AB BC CD : ℝ) := AB = 4 ∧ BC = 5 ∧ CD = 20
noncomputable def angles_obtuse (B C : ℝ) := B > π / 2 ∧ C > π / 2
noncomputable def trig_relations (sinC cosB : ℝ) := sinC = 3 / 5 ∧ cosB = -3 / 5

theorem length_of_AD (AB BC CD AD B C sinC cosB : ℝ) 
  (h1 : side_lengths AB BC CD) 
  (h2 : angles_obtuse B C) 
  (h3 : trig_relations sinC cosB) : 
  AD ≈ 25 :=
begin
  sorry
end

end length_of_AD_l712_712562


namespace pencils_total_l712_712170

theorem pencils_total :
  let Mitchell := 30
  let Antonio := Mitchell * (1 - 0.2)
  let Elizabeth := 2 * Antonio
  Mitchell + Antonio + Elizabeth = 102 :=
by
  let Mitchell := 30
  let Antonio := Mitchell * (1 - 0.2)
  let Elizabeth := 2 * Antonio
  have h : Mitchell + Antonio + Elizabeth = 30 + (30 * 0.8) + (2 * (30 * 0.8))
  sorry

end pencils_total_l712_712170


namespace B₁B₂B₃B₄_is_rectangle_l712_712307

noncomputable theory
open_locale big_operators

-- Definitions for circles and their centers
variable {S S₁ S₂ S₃ S₄ : Type}
variables {O₁ O₂ O₃ O₄ : S} -- centers of S₁, S₂, S₃, S₄
variables {A₁ A₂ A₃ A₄ B₁ B₂ B₃ B₄ : S} -- Intersection points

-- Conditions
variable (OnCircumference : ∀ x ∈ {O₁, O₂, O₃, O₄}, x ∈ S)
variable (Intersects₁₂ : (S₁ ∩ S₂) = {A₁, B₁})
variable (Intersects₂₃ : (S₂ ∩ S₃) = {A₂, B₂})
variable (Intersects₃₄ : (S₃ ∩ S₄) = {A₃, B₃})
variable (Intersects₄₁ : (S₄ ∩ S₁) = {A₄, B₄})
variable (OnCircumference₁ : A₁ ∈ S)
variable (OnCircumference₂ : A₂ ∈ S)
variable (OnCircumference₃ : A₃ ∈ S)
variable (OnCircumference₄ : A₄ ∈ S)
variables (Inside₁ : B₁ ∉ S)
variables (Inside₂ : B₂ ∉ S)
variables (Inside₃ : B₃ ∉ S)
variables (Inside₄ : B₄ ∉ S)

-- Goal
theorem B₁B₂B₃B₄_is_rectangle 
  (OnCircumference : ∀ x ∈ {O₁, O₂, O₃, O₄}, x ∈ S)
  (Intersects₁₂ : (S₁ ∩ S₂) = {A₁, B₁})
  (Intersects₂₃ : (S₂ ∩ S₃) = {A₂, B₂})
  (Intersects₃₄ : (S₃ ∩ S₄) = {A₃, B₃})
  (Intersects₄₁ : (S₄ ∩ S₁) = {A₄, B₄})
  (OnCircumference₁ : A₁ ∈ S)
  (OnCircumference₂ : A₂ ∈ S)
  (OnCircumference₃ : A₃ ∈ S)
  (OnCircumference₄ : A₄ ∈ S)
  (Inside₁ : B₁ ∉ S)
  (Inside₂ : B₂ ∉ S)
  (Inside₃ : B₃ ∉ S)
  (Inside₄ : B₄ ∉ S) :
  IsRectangle (B₁, B₂, B₃, B₄) :=
by sorry

end B₁B₂B₃B₄_is_rectangle_l712_712307


namespace periodic_sequence_l712_712294

def sequence := ℕ → ℕ

def condition (a : sequence) (k : ℕ) : Prop :=
  k > 2016 → (a k = 0 ↔ (∑ i in finset.range 2016, a (k - i - 1)) > 23)

theorem periodic_sequence (a : sequence) (h : ∀ k > 2016, condition a k) :
  ∃ N T > 0, T = 2017 ∧ ∀ k > N, a k = a (k + T) :=
sorry

end periodic_sequence_l712_712294


namespace collinearity_of_points_l712_712916

noncomputable theory
open_locale classical

variables {A P Q M N I T K : Type*}

-- Conditions given in the problem
variables [IncidenceGeometry A P Q M N I T K]
variable [IntersectionPoint T (Line A P) (Line C Q)]
variable [IntersectionPoint K (Line M P) (Line N Q)]

-- Statement of the proof problem
theorem collinearity_of_points :
  Collinear {T, K, I} :=
sorry

end collinearity_of_points_l712_712916


namespace mary_total_nickels_l712_712545

-- Definitions for the conditions
def initial_nickels := 7
def dad_nickels := 5
def mom_nickels := 3 * dad_nickels
def chore_nickels := 2

-- The proof problem statement
theorem mary_total_nickels : 
  initial_nickels + dad_nickels + mom_nickels + chore_nickels = 29 := 
by
  sorry

end mary_total_nickels_l712_712545


namespace collinearity_of_points_l712_712912

noncomputable theory
open_locale classical

variables {A P Q M N I T K : Type*}

-- Conditions given in the problem
variables [IncidenceGeometry A P Q M N I T K]
variable [IntersectionPoint T (Line A P) (Line C Q)]
variable [IntersectionPoint K (Line M P) (Line N Q)]

-- Statement of the proof problem
theorem collinearity_of_points :
  Collinear {T, K, I} :=
sorry

end collinearity_of_points_l712_712912


namespace find_p_of_divisibility_l712_712931

noncomputable def problem_p : ℚ :=
  let T : Finset ℕ := finset_factors 12^7
  let chosen := finset_range.bind (λ _ : Fin 4, T.to_list)
  let valid_sublists := (chosen.filter (λ l, l.length = 4 ∧ (∀ i < 4, ∀ j < 4, i ≤ j → (l.nth i % l.nth j = 0)))).to_list
  let probability_valid := valid_sublists.length / chosen.length
  let (p, q) := probability_valid.q_numden.inert
  p

theorem find_p_of_divisibility : 𝑝-probability problem_p = 101 :=
by
  sorry

end find_p_of_divisibility_l712_712931


namespace polar_to_rectangular_l712_712326

   theorem polar_to_rectangular (ρ θ x y : ℝ) (h1 : ρ = sin θ + cos θ) 
                                (h2 : ρ * cos θ = x) (h3 : ρ * sin θ = y) : 
                                (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
   by
     sorry
   
end polar_to_rectangular_l712_712326
