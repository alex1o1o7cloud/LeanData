import Mathlib
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Functional
import Mathlib.Algebra.Real
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.GraphTheory.Connected
import Mathlib.GraphTheory.Euler
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Probability.Basic
import Mathlib.Probability.Expectation
import Mathlib.Tactic
import Mathlib.Trigonometry.Sine
import analysis.special_functions.integrals

namespace possible_values_of_k_l636_636559

theorem possible_values_of_k (n : ℕ) (k : ℕ) (f : ℕ → ℕ → ℝ)
    (h1 : ∑ i in finset.range n, ∑ j in finset.range n, f i j > 0)
    (h2 : ∀ (a b : ℕ) (ha : a + k ≤ n) (hb : b + k ≤ n), ∑ i in finset.Ico a (a + k), ∑ j in finset.Ico b (b + k), f i j < 0) :
    k < n ∧ ¬ n % k = 0 :=
sorry

end possible_values_of_k_l636_636559


namespace total_distance_l636_636187

-- Define the distances as nonnegative real numbers
noncomputable def distance (a b : ℝ) : Prop := a ≥ 0 ∧ b ≥ 0

-- Define the conditions for the problem
def HomeToStore (d₁ : ℝ) : Prop := distance d₁ 50 -- from Home to Store
def StoreToPeter : ℝ := 50 -- from Store to Peter
def TimeRelation (d₁ d₂ : ℝ) (t₁ t₂ : ℝ) : Prop := 2 * t₂ = t₁ ∧ d₁ = d₂ -- Time and distance relation

-- Prove the total distance
theorem total_distance (d₁ d₂ t₁ t₂ : ℝ) (hHomeStore : HomeToStore d₁) (hStorePeter : StoreToPeter = d₂) (hTimeRel : TimeRelation d₁ d₂ t₁ t₂) : (d₁ + d₂ + d₂) = 150 := 
by 
  -- hHomeStore: distance d₁ 50 -> d₁ = 50 (by definition)
  have h₁ : d₁ = 50 := by sorry

  -- hStorePeter: d₂ = 50 (by definition)
  have h₂ : d₂ = 50 := by sorry

  -- Calc: d₁ + d₂ + d₂ = 50 + 50 + 50 = 150
  calc
    d₁ + d₂ + d₂ = 50 + 50 + 50 := by sorry
    ... = 150 := by sorry

end total_distance_l636_636187


namespace ratio_of_blue_to_purple_beads_l636_636394

theorem ratio_of_blue_to_purple_beads :
  ∃ (B G : ℕ), 
    7 + B + G = 46 ∧ 
    G = B + 11 ∧ 
    B / 7 = 2 :=
by
  sorry

end ratio_of_blue_to_purple_beads_l636_636394


namespace sum_of_n_for_3n_minus_8_eq_5_l636_636742

theorem sum_of_n_for_3n_minus_8_eq_5 : 
  let S := {n | |3 * n - 8| = 5}
  in S.sum id = 16 / 3 :=
by
  sorry

end sum_of_n_for_3n_minus_8_eq_5_l636_636742


namespace probability_of_selecting_cooking_l636_636789

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636789


namespace quadrilaterals_not_necessarily_congruent_l636_636292

-- Define convex quadrilateral type
structure ConvexQuadrilateral where
  sides : ℝ → ℝ → ℝ → ℝ → Prop -- Predicate for side lengths
  diagonals : ℝ → ℝ → Prop      -- Predicate for diagonals
  convex : Prop

-- Statement of the problem in Lean
theorem quadrilaterals_not_necessarily_congruent
  (Q1 Q2 : ConvexQuadrilateral)
  (h1 : Q1.sides (λ a b c d, a < b ∧ b < c ∧ c < d))
  (h2 : Q2.sides (λ a b c d, a < b ∧ b < c ∧ c < d))
  (h3 : Q1.diagonals (λ e f, e < f))
  (h4 : Q2.diagonals (λ e f, e < f))
  (h5 : ∀(a b c d : ℝ), Q1.sides (a, b, c, d) → Q2.sides (a, b, c, d))
  (h6 : ∀(e f : ℝ), Q1.diagonals (e, f) → Q2.diagonals (e, f)) :
  ¬(Q1 = Q2) :=
sorry

end quadrilaterals_not_necessarily_congruent_l636_636292


namespace probability_of_selecting_cooking_l636_636844

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636844


namespace cost_of_painting_l636_636347

def area_of_house : ℕ := 484
def price_per_sqft : ℕ := 20

theorem cost_of_painting : area_of_house * price_per_sqft = 9680 := by
  sorry

end cost_of_painting_l636_636347


namespace determine_function_l636_636571

theorem determine_function (f : ℝ → ℝ) (hf : ∀ x, f x = (32 / 9) * x^2) :
  (∀ x, (∫ t in 0..x, f t) + f x * (sqrt (f x / 2) - x) - (∫ t in 0..(sqrt (f x / 2)), 2 * t^2) =
          (∫ t in 0..(sqrt (f x / 2)), 2 * t^2) - (∫ t in 0..(sqrt (f x / 2)), t^2)) →
  (∀ x, f x = (32 / 9) * x^2) := 
by 
  sorry

end determine_function_l636_636571


namespace total_number_of_squares_l636_636637

variable (x y : ℕ) -- Variables for the number of 10 cm and 20 cm squares

theorem total_number_of_squares
  (h1 : 100 * x + 400 * y = 2500) -- Condition for area
  (h2 : 40 * x + 80 * y = 280)    -- Condition for cutting length
  : (x + y = 16) :=
sorry

end total_number_of_squares_l636_636637


namespace decimal_to_fraction_denominator_l636_636667

theorem decimal_to_fraction_denominator :
  let S := (70 : ℚ) / 90 in 
  denominator (S) = 9 := 
by
  -- Define the repeating decimal as a fraction
  have hS : S = 7 / 9 := by norm_num
  -- Show that its denominator is 9
  exact hS.symm ▸ rfl

end decimal_to_fraction_denominator_l636_636667


namespace missing_vowels_l636_636194

theorem missing_vowels (total_missing_keys : ℕ) (missing_consonants_fraction : ℝ) (total_consonants : ℕ) :
  total_missing_keys = 5 → missing_consonants_fraction = 1/7 → total_consonants = 21 → 
  5 - (total_consonants * missing_consonants_fraction).toNat = 2 :=
by
  sorry

end missing_vowels_l636_636194


namespace geom_series_first_term_l636_636400

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l636_636400


namespace total_marbles_l636_636220

def Mary_marbles : ℕ := 9
def Joan_marbles : ℕ := 3

theorem total_marbles : Mary_marbles + Joan_marbles = 12 :=
by
  -- Please provide the proof here if needed
  sorry

end total_marbles_l636_636220


namespace books_bought_yard_sale_l636_636221

theorem books_bought_yard_sale (initial_books final_books books_bought : ℕ)
  (h1 : initial_books = 35)
  (h2 : final_books = 56)
  (h3 : books_bought = final_books - initial_books) :
  books_bought = 21 := 
by
  rw [h1, h2, h3]
  norm_num

end books_bought_yard_sale_l636_636221


namespace probability_of_cooking_l636_636799

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636799


namespace monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l636_636621

noncomputable def f (x m : ℝ) : ℝ := x - m * (x + 1) * Real.log (x + 1)

theorem monotonicity_intervals_m0 :
  ∀ x : ℝ, x > -1 → f x 0 = x - 0 * (x + 1) * Real.log (x + 1) ∧ f x 0 > 0 := 
sorry

theorem monotonicity_intervals_m_positive (m : ℝ) (hm : m > 0) :
  ∀ x : ℝ, x > -1 → 
  (f x m > f (x + e ^ ((1 - m) / m) - 1) m ∧ 
  f (x + e ^ ((1 - m) / m) - 1) m < f (x + e ^ ((1 - m) / m) - 1 + 1) m) :=
sorry

theorem intersection_points_m1 (t : ℝ) (hx_rng : -1 / 2 ≤ t ∧ t < 1) :
  (∃ x1 x2 : ℝ, x1 > -1/2 ∧ x1 ≤ 1 ∧ x2 > -1/2 ∧ x2 ≤ 1 ∧ f x1 1 = t ∧ f x2 1 = t) ↔ 
  (-1 / 2 + 1 / 2 * Real.log 2 ≤ t ∧ t < 0) :=
sorry

theorem inequality_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (1 + a) ^ b < (1 + b) ^ a :=
sorry

end monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l636_636621


namespace line_equation_through_pointP_and_angle_l636_636673

-- Given conditions
def pointP : ℝ × ℝ := (Real.sqrt 3, -2 * Real.sqrt 3)
def angle : ℝ := 135

-- We need to prove that the line equation is x + y + sqrt(3) = 0
theorem line_equation_through_pointP_and_angle :
  ∃ (a b c : ℝ), a * pointP.1 + b * pointP.2 + c = 0 ∧
                 a = 1 ∧ b = 1 ∧ c = Real.sqrt 3 :=
by
  sorry

end line_equation_through_pointP_and_angle_l636_636673


namespace range_of_f_l636_636677

def floor (x : ℝ) : ℤ := Int.floor x

def f (x : ℝ) : ℝ := floor x - x

theorem range_of_f : set.Icc (-1 : ℝ) 0 = {y : ℝ | ∃ x : ℝ, f x = y} :=
sorry

end range_of_f_l636_636677


namespace total_miles_cycled_l636_636184

theorem total_miles_cycled (D : ℝ) (store_to_peter : ℝ) (same_speed : Prop)
  (time_relation : 2 * D = store_to_peter) (store_to_peter_dist : store_to_peter = 50) :
  D + store_to_peter + store_to_peter = 200 :=
by
  have h1 : D = 100, from sorry  -- Solved from store_to_peter = 50 and time_relation means D = 50*2 = 100
  by rw [h1, store_to_peter_dist]; norm_num; exact h1 

end total_miles_cycled_l636_636184


namespace geometric_locus_eq_union_of_lines_l636_636695

noncomputable def is_perpendicular (line1 line2 : ℝ × ℝ) : Prop :=
  line1.1 * line2.1 + line1.2 * line2.2 = 0

noncomputable def equidistant (p1 p2 o : ℝ × ℝ) : Prop :=
  dist p1 o = dist p2 o

noncomputable def locus_of_points (A B C D : ℝ × ℝ) (AB_CD_equal_nonparallel : dist A B = dist C D ∧ ¬ is_parallel (B - A) (D - C)) : ℝ × ℝ → Prop :=
  λ O, ∃ l1 l2 m1 m2,
    (is_perpendicular l1 ⟨B - A⟩ ∧ equidistant A D O ∧ O ∈ m1 ∧
     is_perpendicular l2 ⟨D - C⟩ ∧ equidistant A C O ∧ O ∈ m2 ∧
     locus O ∈ m1 ∨ locus O ∈ m2)

theorem geometric_locus_eq_union_of_lines (A B C D : ℝ × ℝ)
  (h1 : dist A B = dist C D)
  (h2 : ¬ is_parallel (B - A) (D - C)) :
  let m1 := line (λ O, is_perpendicular (B - A) l1 ∧ equidistant A D O)
  let m2 := line (λ O, is_perpendicular (D - C) l2 ∧ equidistant A C O)
  ∀ O, locus_of_points A B C D (h1, h2) O ↔ (O ∈ m1 ∨ O ∈ m2) :=
sorry

end geometric_locus_eq_union_of_lines_l636_636695


namespace probability_selecting_cooking_l636_636908

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636908


namespace participation_schemes_count_l636_636463

-- Define the conditions
def num_people : ℕ := 6
def num_selected : ℕ := 4
def subjects : List String := ["math", "physics", "chemistry", "english"]
def not_in_english : List String := ["A", "B"]

-- Define the problem 
theorem participation_schemes_count : 
  ∃ total_schemes : ℕ , (total_schemes = 240) :=
by {
  sorry
}

end participation_schemes_count_l636_636463


namespace sequence_general_term_l636_636124

-- Definitions of the conditions
def sequence (a : ℕ → ℤ) :=
  a 1 = 1 ∧ (∀ n, a (n + 1) = 3 * a n + 2)

-- The property to prove
theorem sequence_general_term (a : ℕ → ℤ) (h : sequence a) :
  ∀ n, a n = 2 * 3^(n - 1) - 1 :=
sorry

end sequence_general_term_l636_636124


namespace sum_of_n_for_3n_minus_8_eq_5_l636_636741

theorem sum_of_n_for_3n_minus_8_eq_5 : 
  let S := {n | |3 * n - 8| = 5}
  in S.sum id = 16 / 3 :=
by
  sorry

end sum_of_n_for_3n_minus_8_eq_5_l636_636741


namespace max_value_2x_plus_y_l636_636079

theorem max_value_2x_plus_y (x y : ℝ) (h : y^2 / 4 + x^2 / 3 = 1) : 2 * x + y ≤ 4 :=
by
  sorry

end max_value_2x_plus_y_l636_636079


namespace sum_of_n_values_l636_636737

theorem sum_of_n_values : sum (λ n, |3 * n - 8| = 5) = 16/3 := 
by
  sorry

end sum_of_n_values_l636_636737


namespace probability_cooking_selected_l636_636920

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636920


namespace dx_eq_do_l636_636161

variable {A B C O H D X : Type}
variable [inhabited ABC] [inhabited O] [inhabited H] [inhabited D] [inhabited X]

-- Triangle ABC is an acute triangle with ∠A = 45°
axiom triangle_abc_acute (triangle : Triangle ABC) (angle_A : Angle ABC A = 45)

-- O is the circumcenter of triangle ABC
axiom circumcenter_o (triangle : Triangle ABC) (O : Center ABC)

-- H is the orthocenter of triangle ABC
axiom orthocenter_h (triangle : Triangle ABC) (H : Center ABC)

-- BD is the altitude from B to AC with foot D
axiom altitude_bd (triangle : Triangle ABC) (B D : Type) (altitude : Altitude B AC D)

-- X is the midpoint of arc ADH on the circumcircle of triangle ADH
axiom midpoint_x (triangle : Triangle ADH) (arc_midpoint : MidpointArc ADH X)

-- The final proof that DX = DO
theorem dx_eq_do : ∀ (triangle : Triangle ABC) (circum_o : Center O) (orth_h : Center H) (alt_bd : Altitude B AC D) (mid_x : MidpointArc ADH X), Distance D X = Distance D O :=
by sorry

end dx_eq_do_l636_636161


namespace probability_of_selecting_cooking_l636_636840

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636840


namespace find_B_l636_636055

theorem find_B : 
  ∀ (A B : ℕ), A ≤ 9 → B ≤ 9 → (600 + 10 * A + 5) + (100 + B) = 748 → B = 3 :=
by
  intros A B hA hB hEq
  sorry

end find_B_l636_636055


namespace equal_sum_sequence_S_9_l636_636436

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Conditions taken from the problem statement
def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) :=
  ∀ n : ℕ, a n + a (n + 1) = c

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Lean statement of the problem
theorem equal_sum_sequence_S_9
  (h1 : equal_sum_sequence a 5)
  (h2 : a 1 = 2)
  : sum_first_n_terms a 9 = 22 :=
sorry

end equal_sum_sequence_S_9_l636_636436


namespace problemI_solution_set_problemII_range_of_a_l636_636501

section ProblemI

def f (x : ℝ) := |2 * x - 2| + 2

theorem problemI_solution_set :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end ProblemI

section ProblemII

def f (x a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

theorem problemII_range_of_a :
  {a : ℝ | ∀ x : ℝ, f x a + g x ≥ 3} = {a : ℝ | 2 ≤ a} :=
by
  sorry

end ProblemII

end problemI_solution_set_problemII_range_of_a_l636_636501


namespace general_formula_a_max_S_b_l636_636477

noncomputable def a_n : ℕ → ℕ
| 1 => 3
| 2 => 5
| (n+3) => a_n (n+2) + 2^(n+1)

def S_n (a : ℕ → ℕ) : ℕ → ℕ
| 0 => 0
| 1 => a 1
| (n+2) => S_n a (n+1) + a (n+2)

theorem general_formula_a (n : ℕ) : a_n n = 2^n + 1 :=
sorry

def b_n (n : ℕ) : ℝ :=
real.log2 (256 / (a_n (2 * n) - 1))

def S_b (b : ℕ → ℝ) : ℕ → ℝ
| 0 => 0
| 1 => b 1
| (n+2) => S_b b (n+1) + b (n+2)

theorem max_S_b : ∃ n, S_b b_n n = 12 ∧ n = 3 ∨ n = 4 :=
sorry

end general_formula_a_max_S_b_l636_636477


namespace sin_decreasing_interval_l636_636669

open Real

theorem sin_decreasing_interval :
  ∀ x, x ∈ Icc (π / 2) (3 * π / 2) → sin' x < 0 :=
begin
  sorry
end

end sin_decreasing_interval_l636_636669


namespace inequality_solution_a_range_l636_636500

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| - |x + 2|

theorem inequality_solution (x : ℝ) : f(x) > 0 ↔ x < -1/3 ∨ x > 3 := 
by
  sorry

theorem a_range (a : ℝ) : (∀ x : ℝ, f(x) + 3 * |x + 2| ≥ |a - 1|) ↔ (-4 ≤ a ∧ a ≤ 6) := 
by
  sorry

end inequality_solution_a_range_l636_636500


namespace geometric_series_first_term_l636_636409

theorem geometric_series_first_term (r a s : ℝ) (h₁ : r = 1 / 4) (h₂ : s = 80) (h₃ : s = a / (1 - r)) : a = 60 :=
by
  rw [h₁] at h₃
  rw [h₂] at h₃
  norm_num at h₃
  linarith

# Examples utilized:
-- r : common ratio
-- a : first term of the series
-- s : sum of the series
-- h₁ : condition that the common ratio is 1/4
-- h₂ : condition that the sum is 80
-- h₃ : condition representing the formula for the sum of an infinite geometric series

end geometric_series_first_term_l636_636409


namespace max_quarters_l636_636644

-- Definitions stating the conditions
def total_money_in_dollars : ℝ := 4.80
def value_of_quarter : ℝ := 0.25
def value_of_dime : ℝ := 0.10

-- Theorem statement
theorem max_quarters (q : ℕ) (h1 : total_money_in_dollars = (q * value_of_quarter) + (2 * q * value_of_dime)) : q ≤ 10 :=
by {
  -- Injecting a placeholder to facilitate proof development
  sorry
}

end max_quarters_l636_636644


namespace total_distance_cycled_l636_636190

theorem total_distance_cycled (d_store_peter : ℕ) 
  (store_to_peter_distance : d_store_peter = 50) 
  (home_to_store_twice : ∀ t, distance_from_home_to_store * t = 2 * d_store_peter * t)
  : distance_from_home_to_store = 100 → 
    total_distance = (distance_from_home_to_store + store_to_peter_distance + d_store_peter) :=
by
  /- Let distance_from_home_to_store be denoted as d_home_store for simplicity -/
  let d_home_store := distance_from_home_to_store
  -- Given
  have home_to_store_distance : d_home_store = 2 * d_store_peter := home_to_store_twice 1
  -- We can derive
  have h1 : d_home_store = 100 := home_to_store_distance
  -- Calculate total distance
  have total_distance := d_home_store + d_store_peter + d_store_peter
  -- Conclusion
  have h2 : total_distance = 100 + 50 + 100 := rfl
  -- Final Proposition
  exact h2

end total_distance_cycled_l636_636190


namespace well_defined_group_B_l636_636441

-- Definitions of the groups
def group_A := {x : ℝ | abs (x - Real.sqrt 5) < ε}  -- Placeholder for "sufficiently close"
def group_B := {s : Type | ∃ a b : ℤ, s = { (a, b) ∣ a * a =  b * b }}  -- Definition of squares
def group_C := {m : Type | famous m}  -- Placeholder for "famous mathematicians"
def group_D := {s : Type | tall s ∧ high_school_student s}  -- Placeholder for "tall high school students"

-- Well-definedness criterion placeholder
def is_well_defined (S : Type) : Prop := ∃ (criterion : S → Prop), ∀ x y, criterion x → criterion y → x = y

-- The proof statement
theorem well_defined_group_B :
  is_well_defined group_B ∧
  ¬ is_well_defined group_A ∧
  ¬ is_well_defined group_C ∧
  ¬ is_well_defined group_D :=
by
  sorry

end well_defined_group_B_l636_636441


namespace parabola_intersection_x_coordinate_l636_636970

theorem parabola_intersection_x_coordinate :
  ∃ t : ℝ, Parabola₁.passes_through (10, 0) ∧ Parabola₁.passes_through (13, 0) ∧
            Parabola₂.passes_through (13, 0) ∧
            vertex_of_Parabola₁.bisects (0, 0) (vertex_of_Parabola₂) ∧
            Parabola₂.intersects_x_axis_at t ∧ t = 33 :=
by
  sorry

end parabola_intersection_x_coordinate_l636_636970


namespace slope_of_line_dividing_rectangle_l636_636704

theorem slope_of_line_dividing_rectangle (h_vertices : 
  ∃ (A B C D : ℝ × ℝ), A = (1, 0) ∧ B = (9, 0) ∧ C = (1, 2) ∧ D = (9, 2) ∧ 
  (∃ line : ℝ × ℝ, line = (0, 0) ∧ line = (5, 1))) : 
  ∃ m : ℝ, m = 1 / 5 :=
sorry

end slope_of_line_dividing_rectangle_l636_636704


namespace smaller_two_digit_product_l636_636691

-- Statement of the problem
theorem smaller_two_digit_product (a b : ℕ) (h : a * b = 4536) (h_a : 10 ≤ a ∧ a < 100) (h_b : 10 ≤ b ∧ b < 100) : a = 21 ∨ b = 21 := 
by {
  sorry -- proof not needed
}

end smaller_two_digit_product_l636_636691


namespace probability_selecting_cooking_l636_636902

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636902


namespace eulerian_orientation_exists_l636_636551

theorem eulerian_orientation_exists {V : Type*} (G : SimpleGraph V) [Fintype V] [DecidableRel G.Adj]
  (h_connected : G.Connected) (h_even_degree : ∀ v : V, Even (G.degree v)) :
  ∃ (G_oriented : SimpleGraph V), (∀ v : V, (G_oriented.degree v = G.degree v / 2)) ∧ (∀ u v : V, G_oriented.Connected) :=
by
  sorry

end eulerian_orientation_exists_l636_636551


namespace work_problem_l636_636259

theorem work_problem 
  (P : ℕ) -- Number of persons
  (W : ℕ) -- Amount of work
  (h1 : ∃ n, n = 20) -- P persons can do W work in 20 days
  (h2 : ∃ k, k = W / 20) -- Work rate of P persons
  : (2 * P) can_complete (W / 2) in 5 :=
begin
  sorry
end

end work_problem_l636_636259


namespace final_reduced_price_l636_636974

noncomputable def original_price (P : ℝ) (Q : ℝ) : ℝ := 800 / Q

noncomputable def price_after_first_week (P : ℝ) : ℝ := 0.90 * P
noncomputable def price_after_second_week (price1 : ℝ) : ℝ := 0.85 * price1
noncomputable def price_after_third_week (price2 : ℝ) : ℝ := 0.80 * price2

noncomputable def reduced_price (P : ℝ) : ℝ :=
  let price1 := price_after_first_week P
  let price2 := price_after_second_week price1
  price_after_third_week price2

theorem final_reduced_price :
  ∃ P Q : ℝ, 
    800 = Q * P ∧
    800 = (Q + 5) * reduced_price P ∧
    abs (reduced_price P - 62.06) < 0.01 :=
by
  sorry

end final_reduced_price_l636_636974


namespace find_y_when_x_is_neg2_l636_636119

noncomputable def k : ℝ :=
4 / 2

def f (x : ℝ) : ℝ := k * x

theorem find_y_when_x_is_neg2 : f (-2) = -4 :=
sorry

end find_y_when_x_is_neg2_l636_636119


namespace parallel_lines_slope_l636_636128

theorem parallel_lines_slope (a : ℝ)
  (h₁ : ∀ x : ℝ, y = a * x - 2)
  (h₂ : ∀ x y : ℝ, 3 * x - (a + 2) * y + 1 = 0)
  (parallel : ∀ x : ℝ, slope of the first line = slope of the second line)
  : a = 1 ∨ a = -3 :=
sorry

end parallel_lines_slope_l636_636128


namespace probability_three_out_of_six_greater_than_five_l636_636993

open Nat

def probability_of_exceeding_numbers_greater_than_five (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_three_out_of_six_greater_than_five :
  probability_of_exceeding_numbers_greater_than_five 6 3 (1 / 2) = 5 / 16 := by
  -- Calculation steps to be filled
  sorry

end probability_three_out_of_six_greater_than_five_l636_636993


namespace tan_alpha_value_l636_636528

-- Defining the conditions as hypotheses
variables {α : ℝ}
hypothesis (h_sin : Real.sin α = -5/13)
hypothesis (h_quad: π < α ∧ α < 3 * π / 2)

-- Stating the theorem to prove
theorem tan_alpha_value : Real.tan α = 5/12 :=
by sorry

end tan_alpha_value_l636_636528


namespace integer_values_of_n_l636_636459

theorem integer_values_of_n :
  {n : ℤ | 1 ≤ n ∧ n ≤ 6 ∧ Real.sqrt (n + 1) ≤ Real.sqrt (5 * n - 7) ∧
  Real.sqrt (5 * n - 7) < Real.sqrt (3 * n + 6)}.to_finset.card = 5 :=
by
  sorry

end integer_values_of_n_l636_636459


namespace exists_n_factors_2004_l636_636216

def f : ℤ → ℤ := sorry  -- assuming a non-constant polynomial with integer coefficients

theorem exists_n_factors_2004 (f_is_non_const : ∃ x ∈ ℤ, f(x) ≠ (0 : ℤ))
  (integer_coeffs : ∀ n, n ∈ ℤ → f(n) ∈ ℤ) : 
  ∃ n : ℤ, (nat.prime_factors (int.nat_abs (f(n)))).length ≥ 2004 :=
sorry

end exists_n_factors_2004_l636_636216


namespace probability_of_selecting_cooking_is_one_fourth_l636_636881

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636881


namespace probability_cooking_is_one_fourth_l636_636853
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636853


namespace Ray_gave_25_cents_to_Peter_l636_636643

def Ray_problem (P R : ℕ) : Prop :=
  P + R = 75 ∧ R = 2 * P ∧ P = 25

theorem Ray_gave_25_cents_to_Peter : ∃ P R : ℕ, Ray_problem P R :=
by {
  let P := 25,
  let R := 50,
  use [P, R],
  unfold Ray_problem,
  split,
  { 
    norm_num,
  },
  split,
  { 
    norm_num,
  }, 
  { 
    norm_num,
  }
}

end Ray_gave_25_cents_to_Peter_l636_636643


namespace probability_of_selecting_cooking_l636_636944

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636944


namespace triangle_MNC_is_isosceles_l636_636199

variables {A B C D M N : Type}
variables [PlaneGeometry A] [PlaneGeometry B] [PlaneGeometry C] [PlaneGeometry D] [Midpoint M A D] [Midpoint N A B]

-- Assume the convex quadrilateral ABCD with specific conditions
variables (AD_eq_DC : AD = DC) (AC_eq_AB : AC = AB)
          (angle_ADC_eq_angle_CAB : ∠ADC = ∠CAB)

-- Define midpoints M and N
variable (M_mid_AD : IsMidpoint M A D)
variable (N_mid_AB : IsMidpoint N A B)

-- The final proof
theorem triangle_MNC_is_isosceles : IsoscelesTriangle M N C :=
by
  sorry

end triangle_MNC_is_isosceles_l636_636199


namespace probability_of_cooking_l636_636806

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636806


namespace joe_dropped_score_l636_636593

theorem joe_dropped_score (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 60) (h2 : (A + B + C) / 3 = 65) :
  min A (min B (min C D)) = D → D = 45 :=
by sorry

end joe_dropped_score_l636_636593


namespace problem_i_problem_ii_l636_636127

variables {R r : ℝ} (hRr : R > r)
variables {O P : ℝ × ℝ} (hP : dist O P = r)
variables {B : ℝ × ℝ} (hB : dist O B = R)

-- Problem I: Prove BC^2 + CA^2 + AB^2 = 2r^2 + 6R^2
theorem problem_i (C A : ℝ × ℝ) (hC : ∃ t : ℝ, C = B + t • (B - P)) 
                  (hA : ∃ ℓ : ℝ, ℓ ≠ 0 ∧ line_through P (P + ℓ • perp (B - P)) ∩ circle O r = {A}) :
  dist B C ^ 2 + dist C A ^ 2 + dist A B ^ 2 = 2 * r ^ 2 + 6 * R ^ 2 :=
sorry

-- Problem II: Prove the geometric locus of the midpoint of segment AB is a circle with radius R/2 centered at O
theorem problem_ii : 
  ∃ M : ℝ × ℝ, (∀ A B, dist O (midpoint A B) = R / 2) :=
sorry

end problem_i_problem_ii_l636_636127


namespace arithmetic_sequence_common_difference_l636_636578

-- Arithmetic sequence with condition and proof of common difference
theorem arithmetic_sequence_common_difference (a : ℕ → ℚ) (d : ℚ) :
  (a 2015 = a 2013 + 6) → ((a 2015 - a 2013) = 2 * d) → (d = 3) :=
by
  intro h1 h2
  sorry

end arithmetic_sequence_common_difference_l636_636578


namespace probability_cooking_is_one_fourth_l636_636852
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636852


namespace initial_meals_is_70_l636_636370

-- Define variables and conditions
variables (A : ℕ)
def initial_meals_for_adults := A

-- Given conditions
def condition_1 := true  -- Group of 55 adults and some children (not directly used in proving A)
def condition_2 := true  -- Either a certain number of adults or 90 children (implicitly used in equation)
def condition_3 := (A - 21) * (90 / A) = 63  -- 21 adults have their meal, remaining food serves 63 children

-- The proof statement
theorem initial_meals_is_70 (h : (A - 21) * (90 / A) = 63) : A = 70 :=
sorry

end initial_meals_is_70_l636_636370


namespace larger_segment_correct_l636_636699

-- Define the sides of the triangle
def side1 : ℝ := 40
def side2 : ℝ := 90
def side3 : ℝ := 100

-- Define the larger segment when an altitude is dropped to the side of length 100
def larger_segment (s1 s2 s3 : ℝ) (h : ℝ) : ℝ :=
  let x := (s2^2 - s1^2 + s3^2) / (2 * s3) in s3 - x

-- Statement of the proof problem
theorem larger_segment_correct :
  (larger_segment side1 side2 side3 (sqrt (side1^2 - ((side2^2 - side1^2 + side3^2) / (2 * side3))^2)) = 82.5) :=
by sorry

end larger_segment_correct_l636_636699


namespace probability_selecting_cooking_l636_636906

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636906


namespace probability_cooking_selected_l636_636918

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636918


namespace reimbursement_calculation_l636_636225

variable (total_paid : ℕ) (pieces : ℕ) (cost_per_piece : ℕ)

theorem reimbursement_calculation
  (h1 : total_paid = 20700)
  (h2 : pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (pieces * cost_per_piece) = 600 := 
by
  sorry

end reimbursement_calculation_l636_636225


namespace max_area_triangle_l636_636173

theorem max_area_triangle 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : a = 2)
  (h2 : b^2 - 2 * real.sqrt 3 * b * c * real.sin A + c^2 = 4) :
  ∃ S : ℝ, S = 2 + real.sqrt 3 ∧
    S ≥ (1/2 * b * c * real.sin A) :=
sorry

end max_area_triangle_l636_636173


namespace complex_subset_sum_modulus_l636_636260

open Complex
open BigOperators

theorem complex_subset_sum_modulus {n : ℕ} (z : Fin n → ℂ)
  (h : ∑ k, |z k| = 1) : ∃ S : Finset (Fin n), ∥∑ k in S, z k∥ ≥ 1 / 4 :=
sorry

end complex_subset_sum_modulus_l636_636260


namespace exists_trickster_l636_636155

universe u

def Person := Type u

inductive Role
| Knight
| Liar
| Trickster

def response (r : Role) : Bool :=
match r with
| Role.Knight => true   -- Knight tells the truth
| Role.Liar => false    -- Liar always lies
| Role.Trickster => arbitrary Bool  -- Trickster can choose

def is_knight (r : Role) : Prop :=
r = Role.Knight

def friends (A B C : Person) (f : Person → Role) (responses : Person → Bool) : Prop :=
(responses A = false ∧ responses B = false ∧ responses C = true) 
∨ (responses A = false ∧ responses B = true ∧ responses C = false)
∨ (responses A = true ∧ responses B = false ∧ responses C = false)

theorem exists_trickster (A B C : Person) (f : Person → Role) 
  (responses : Person → Bool) :
  ((responses A = false ∧ responses B = false ∧ responses C = true)
   ∨ (responses A = false ∧ responses B = true ∧ responses C = false)
   ∨ (responses A = true ∧ responses B = false ∧ responses C = false))
  → (∃ x : Person, f x = Role.Trickster) :=
sorry

end exists_trickster_l636_636155


namespace triangle_altitude_ratio_l636_636171

/-
  In triangle ABC, BC=6, AC=3, and angle C=30°. 
  Altitudes AD, BE, and CF intersect at the orthocenter H. 
  Find the ratio AH:HD.
-/
theorem triangle_altitude_ratio :
  ∀ (A B C D H : Point) (BC AC : ℝ) (angleC : ℝ),
  BC = 6 →
  AC = 3 →
  angleC = 30 →
  -- here assuming Point and definition of altitudes with the necessary geometry library
  orthocenter A B C H →
  altitude A D →
  altitude B E →
  altitude C F →
  ratio (segment_length A H) (segment_length H D) = 0.155 := sorry

end triangle_altitude_ratio_l636_636171


namespace a_n_b_n_identity_l636_636697

noncomputable def a : ℕ → ℚ
| 0       := 1 / 2
| (n + 1) := 2 * a n / (1 + (a n)^2)

noncomputable def b : ℕ → ℚ
| 0       := 4
| (n + 1) := (b n)^2 - 2 * b n + 2

theorem a_n_b_n_identity (n : ℕ) :
  (a (n + 1)) * (b (n + 1)) = 2 * ∏ i in Finset.range (n + 1), b i := sorry

end a_n_b_n_identity_l636_636697


namespace currant_yield_increase_l636_636390

-- Conditions
variable (Y : ℝ) -- Initial yield per bush
variable (Y_new : ℝ) -- New yield per bush

-- Define the relationship between yields before and after treatment
def total_yield_15_bushes := 15 * Y
def total_yield_12_bushes := 12 * Y_new
def yield_increase_percentage := 100 * (Y_new - Y) / Y

-- Statement to prove
theorem currant_yield_increase : 
  total_yield_12_bushes Y Y_new = total_yield_15_bushes Y → 
  yield_increase_percentage Y Y_new = 25 := 
by
  sorry

end currant_yield_increase_l636_636390


namespace probability_cooking_is_one_fourth_l636_636859
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636859


namespace cos_3theta_l636_636531

theorem cos_3theta (θ : ℝ) (h : Real.exp (Complex.i * θ) = (3 + Complex.i * Real.sqrt 2) / 4) :
  Real.cos (3 * θ) = 9 / 64 :=
sorry

end cos_3theta_l636_636531


namespace domain_of_f_correct_l636_636438

def domain_of_f (x : ℝ) : Prop :=
  √(x - 1) + √(x + 3) + (x - 4)^(1/3)

theorem domain_of_f_correct (x : ℝ) : domain_of_f x = domain_of_f x :=
  sorry

end domain_of_f_correct_l636_636438


namespace sheep_in_flock_l636_636756

theorem sheep_in_flock :
  ∃ (x : ℕ), 0.4 * x = 60 ∧ 0.6 * x = 0.6 * x :=
begin
  use 150,
  split,
  { norm_num, },
  { exact eq.refl (0.6 * 150), },
end

end sheep_in_flock_l636_636756


namespace machine_fills_2_boxes_in_5_minutes_l636_636375

-- Define the conditions
def rate (boxes : ℕ) (minutes : ℕ) : ℝ := boxes / minutes

-- Define the main proof problem
theorem machine_fills_2_boxes_in_5_minutes :
  rate 24 60 * 5 = 2 :=
by
  -- using arithmetic to derive the result
  sorry

end machine_fills_2_boxes_in_5_minutes_l636_636375


namespace prove_ab_eq_neg_26_l636_636095

theorem prove_ab_eq_neg_26
  (a b : ℚ)
  (H : ∀ k : ℚ, ∃ x : ℚ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6) :
  a * b = -26 := sorry

end prove_ab_eq_neg_26_l636_636095


namespace probability_of_selecting_cooking_l636_636933

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636933


namespace smallest_n_divisible_by_9_l636_636100

def seq : ℕ → ℤ
| 0     := 0 -- This case is unused but required for ℕ → ℤ function.
| 1     := 1
| 2     := 3
| (n+2) := (n + 3) * seq (n + 1) - (n + 2) * seq n

theorem smallest_n_divisible_by_9 :
  ∃ n, (∀ m ≥ n, 9 ∣ seq m) ∧
  (∀ k < n, ∃ m ≥ k, ¬ 9 ∣ seq m) :=
begin
  use 5,
  split,
  { intros m hm,
    sorry }, -- Here you will show that for all m ≥ 5, seq m is divisible by 9.
  { intros k hk,
    cases k,
    { use 1,
      split; linarith,
      norm_num },
    { cases k,
      { use 2,
        split; linarith,
        norm_num },
      { cases k,
        { use 3,
          split; linarith,
          norm_num },
        { cases k,
          { use 4,
            split; linarith,
            norm_num },
          { exfalso, linarith } } } } },
end

end smallest_n_divisible_by_9_l636_636100


namespace polynomial_abc_l636_636540

theorem polynomial_abc {a b c : ℝ} (h : a * x^2 + b * x + c = x^2 - 3 * x + 2) : a * b * c = -6 := by
  sorry

end polynomial_abc_l636_636540


namespace example_problem_l636_636101

variable {R : Type*} [LinearOrderedField R]

def is_monotonically_decreasing (f : R → R) : Prop :=
  ∀ {x y : R}, 0 < x → 0 < y → x < y → f x > f y

theorem example_problem 
  {f : R → R}
  (h : is_monotonically_decreasing (λ x, f x / x)) 
  (x₁ x₂ : R) 
  (hx₁ : 0 < x₁) 
  (hx₂ : 0 < x₂) :
  f x₁ + f x₂ > f (x₁ + x₂) :=
sorry

end example_problem_l636_636101


namespace kendra_earnings_before_discounts_and_taxes_l636_636198

variables {L : ℝ} (L_pos : L > 0) -- assuming L is positive for realism

-- Define the earnings for Kendra (K), Laurel (L), and Max (M) over the years.
def earnings_2014 : Prop := (∃ K : ℝ, K = L - 8000)
def earnings_2015 : Prop := (∃ K : ℝ, K = 1.4667 * L)
def earnings_2016 : Prop := (∃ K : ℝ, K = 1.5529 * L + 17647.06)

-- Prove Kendra's actual sales earnings before discounts and taxes for each year
theorem kendra_earnings_before_discounts_and_taxes :
  earnings_2014 L ∧ earnings_2015 L ∧ earnings_2016 L :=
by
  split
  -- proving earnings_2014
  {
    use (L - 8000),
    exact rfl
  },
  split
  -- proving earnings_2015
  {
    use (1.4667 * L),
    exact rfl
  },
  -- proving earnings_2016
  {
    use (1.5529 * L + 17647.06),
    exact rfl
  }

end kendra_earnings_before_discounts_and_taxes_l636_636198


namespace probability_of_selecting_cooking_l636_636821

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636821


namespace union_M_N_eq_l636_636510

open Set

variable {α : Type} [LinearOrder α]

def U : Set α := {x | -3 ≤ x ∧ x < 2}
def M : Set α := {x | -1 < x ∧ x < 1}
def complement_U_N : Set α := {x | 0 < x ∧ x < 2}
def N : Set α := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_M_N_eq : M ∪ N = {x | -3 ≤ x ∧ x < 1} := by
  sorry

end union_M_N_eq_l636_636510


namespace number_of_squares_l636_636634

def side_plywood : ℕ := 50
def side_square_1 : ℕ := 10
def side_square_2 : ℕ := 20
def total_cut_length : ℕ := 280

/-- Number of squares obtained given the side lengths of the plywood and the cut lengths -/
theorem number_of_squares (x y : ℕ) (h1 : 100 * x + 400 * y = side_plywood^2)
  (h2 : 40 * x + 80 * y = total_cut_length) : x + y = 16 :=
sorry

end number_of_squares_l636_636634


namespace triangle_O1O2O3_equilateral_l636_636989

/- 
  Given:
  - Equilateral triangles ABC, CDE, EFG, DHI.
  - O1 is the midpoint of AH.
  - O2 is the midpoint of BF.
  - O3 is the midpoint of IG.
  Prove that triangle O1O2O3 is equilateral.
-/

variables {A B C D E F G H I : Point}
variables {O1 O2 O3 : Point}

-- Assuming necessary geometric definitions
axiom equilateral_triangle (P Q R : Point) : Prop
axiom midpoint (M X Y : Point) : Prop

-- Condition: Triangles ABC, CDE, EFG, DHI are equilateral
axiom h1 : equilateral_triangle A B C
axiom h2 : equilateral_triangle C D E
axiom h3 : equilateral_triangle E F G
axiom h4 : equilateral_triangle D H I

-- Condition: O1, O2, O3 are midpoints of AH, BF, IG respectively
axiom h5 : midpoint O1 A H
axiom h6 : midpoint O2 B F
axiom h7 : midpoint O3 I G

-- Prove that triangle O1O2O3 is equilateral
theorem triangle_O1O2O3_equilateral : equilateral_triangle O1 O2 O3 :=
by sorry

end triangle_O1O2O3_equilateral_l636_636989


namespace arithmetic_mean_l636_636448

variable (x b : ℝ)

theorem arithmetic_mean (hx : x ≠ 0) :
  ((x + b) / x + (x - 2 * b) / x) / 2 = 1 - b / (2 * x) := by
  sorry

end arithmetic_mean_l636_636448


namespace probability_select_cooking_l636_636883

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636883


namespace emmy_iPods_lost_is_six_l636_636036

noncomputable def emmy_iPods_lost : ℕ :=
  let E := emmy_lost in
  let R := rosa_iPods in

  have h1 : 14 - E = 2 * R := sorry,
  have h2 : (14 - E) + R = 12 := sorry,

  E

theorem emmy_iPods_lost_is_six : emmy_iPods_lost = 6 :=
by {
  sorry
}

end emmy_iPods_lost_is_six_l636_636036


namespace product_of_sums_of_squares_l636_636489

theorem product_of_sums_of_squares (a b c d : ℤ) :
  let m := a^2 + b^2,
      n := c^2 + d^2
  in ∃ x y : ℤ, m * n = x^2 + y^2 :=
by {
  let m := a^2 + b^2,
  let n := c^2 + d^2,
  use (a * c - b * d),
  use (a * d + b * c),
  sorry
}

end product_of_sums_of_squares_l636_636489


namespace find_a_plus_b_l636_636606

open Complex

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∃ (r1 r2 r3 : ℂ),
     r1 = 1 + I * Real.sqrt 3 ∧
     r2 = 1 - I * Real.sqrt 3 ∧
     r3 = -2 ∧
     (r1 + r2 + r3 = 0) ∧
     (r1 * r2 * r3 = -b) ∧
     (r1 * r2 + r2 * r3 + r3 * r1 = -a))

theorem find_a_plus_b (a b : ℝ) (h : problem_statement a b) : a + b = 8 :=
sorry

end find_a_plus_b_l636_636606


namespace probability_select_cooking_l636_636882

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636882


namespace combined_tax_rate_l636_636015

theorem combined_tax_rate (M : ℝ) (hMork : TaxRate M 0.30) (hMindy : TaxRate (3 * M) 0.20) :
  TaxRate (4 * M) 0.225 :=
by
  sorry

end combined_tax_rate_l636_636015


namespace find_C_and_D_l636_636060

variables (C D : ℝ)

theorem find_C_and_D (h : 4 * C + 2 * D + 5 = 30) : C = 5.25 ∧ D = 2 :=
by
  sorry

end find_C_and_D_l636_636060


namespace probability_of_selecting_cooking_l636_636812

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636812


namespace range_of_a_l636_636467

theorem range_of_a (a b : ℝ) (h1 : 0 ≤ a - b ∧ a - b ≤ 1) (h2 : 1 ≤ a + b ∧ a + b ≤ 4) : 
  1 / 2 ≤ a ∧ a ≤ 5 / 2 := 
sorry

end range_of_a_l636_636467


namespace geom_series_first_term_l636_636403

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l636_636403


namespace exists_a_not_divisible_l636_636201

theorem exists_a_not_divisible (p : ℕ) (hp_prime : Prime p) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ (p^2 ∣ (a^(p-1) - 1)) ∧ ¬ (p^2 ∣ ((a+1)^(p-1) - 1))) :=
  sorry

end exists_a_not_divisible_l636_636201


namespace probability_of_selecting_cooking_l636_636947

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636947


namespace sum_of_n_values_l636_636735

theorem sum_of_n_values : sum (λ n, |3 * n - 8| = 5) = 16/3 := 
by
  sorry

end sum_of_n_values_l636_636735


namespace smaller_two_digit_product_l636_636687

theorem smaller_two_digit_product (a b : ℕ) (h1: 10 ≤ a) (h2: a ≤ 99) (h3: 10 ≤ b) (h4: b ≤ 99) (h_prod: a * b = 4536) : a = 54 ∨ b = 54 :=
by sorry

end smaller_two_digit_product_l636_636687


namespace urn_problem_proof_l636_636418

-- Definitions from conditions.
def white_balls : ℕ := 5
def black_balls : ℕ := 10
def total_balls : ℕ := white_balls + black_balls

-- Random variable X representing the number of white balls drawn.
def X (b : Bool) : ℕ := if b then 1 else 0

-- Probability distribution of X.
def P_X (x : ℕ) : ℚ :=
  if x = 0 then 2 / 3
  else if x = 1 then 1 / 3
  else 0

-- Expected value of X.
def expected_value_X : ℚ :=
  0 * (2 / 3) + 1 * (1 / 3)

-- Variance of X.
def variance_X : ℚ :=
  (0^2 * (2 / 3) + 1^2 * (1 / 3)) - (expected_value_X ^ 2)

-- Standard deviation of X.
def std_deviation_X : Real :=
  real.sqrt variance_X

-- Theorem statement.
theorem urn_problem_proof :
  P_X 0 = 2 / 3 ∧
  P_X 1 = 1 / 3 ∧
  expected_value_X = 1 / 3 ∧
  variance_X = 2 / 9 ∧
  std_deviation_X = real.sqrt (2 / 9) :=
  sorry

end urn_problem_proof_l636_636418


namespace voltage_decreases_by_2_percent_l636_636775

noncomputable def initial_side_length (L : ℝ) : ℝ := L
def initial_capacitance (C : ℝ) : ℝ := C
def percentage_increase (L : ℝ) : ℝ := L * 0.01

theorem voltage_decreases_by_2_percent 
  (L C V : ℝ) 
  (h : initial_side_length L = L ∧ initial_capacitance C = C) 
  : let L' := L * 1.01,
        A' := (L')^2,
        C' := C * 1.0201,
        V' := V / 1.0201
    in ((V' - V) / V) * 100 ≈ -2 :=
by
  sorry

end voltage_decreases_by_2_percent_l636_636775


namespace question1_question2_l636_636125

-- Define the sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := { x | 2 * x + a > 0 }
def setB : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Question 1: When a = 2, find the set A ∩ B
theorem question1 : A ∩ B = { x | x > 3 } :=
  sorry

-- Question 2: If A ∩ (complement of B) = ∅, find the range of a
theorem question2 : A ∩ (U \ B) = ∅ → a ≤ -6 :=
  sorry

end question1_question2_l636_636125


namespace probability_of_selecting_cooking_l636_636816

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636816


namespace probability_select_cooking_l636_636884

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636884


namespace evaluate_expression_at_neg3_l636_636039

theorem evaluate_expression_at_neg3 : 
  (let x := -3 in (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)) = -26 :=
by
  simp only
  intro x
  sorry

end evaluate_expression_at_neg3_l636_636039


namespace compare_logs_finite_comparison_l636_636670

-- Part (a)
theorem compare_logs (a b c d : ℝ) (ha : a > 1) (hb : b > a) (hc : c > 1) (hd : d > c) :
  a = 25 → b = 75 → c = 65 → d = 260 → log 25 75 > log 65 260 :=
by
  intros ha hb hc hd h25 h75 h65 h260
  sorry

-- Part (b)
theorem finite_comparison (a b c d : ℝ) (ha : a > 1) (hb : b > a ∨ b < a)
  (hc : c > 1) (hd : d > c ∨ d < c) : 
  ∃ n, (compare_logs_iter a b c d ha hb hc hd n) = (log a b > log c d) :=
by
  sorry

end compare_logs_finite_comparison_l636_636670


namespace union_complement_eq_l636_636513

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {-1, 0, 3}

theorem union_complement_eq :
  A ∪ (U \ B) = {-2, -1, 0, 1, 2} := by
  sorry

end union_complement_eq_l636_636513


namespace probability_of_selecting_cooking_l636_636847

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636847


namespace shaded_region_area_between_7_and_8_l636_636354

-- Definitions and conditions
variables (A B C D : Type) [rectangle : Rectangle A B C D]
variable (circle_centered_at_D : Circle D)
variables (AD CD : ℝ)
variables (AD_eq_4 : AD = 4)
variables (CD_eq_3 : CD = 3)
variable (B_on_circle : OnCircle B circle_centered_at_D)

-- Statement to prove
theorem shaded_region_area_between_7_and_8 :
  7 < (π * (5 ^ 2) / 4 - (AD * CD)) ∧ (π * (5 ^ 2) / 4 - (AD * CD)) < 8 :=
by 
  sorry

end shaded_region_area_between_7_and_8_l636_636354


namespace area_of_fourth_triangle_l636_636180

theorem area_of_fourth_triangle (A B C D P : Point) (area_triangle : Point → Point → Point → ℝ)
  (hABC : Parallelogram A B C D) (hP : InsideParallelogram P A B C D)
  (h_area_PA : area_triangle P A C = 1) (h_area_PB : area_triangle P B C = 2) (h_area_PC : area_triangle P C D = 3) :
  ∃ S4, S4 = 4 :=
by
  sorry

end area_of_fourth_triangle_l636_636180


namespace probability_of_selecting_cooking_l636_636837

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636837


namespace probability_of_cooking_l636_636802

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636802


namespace area_of_triangle_ABC_l636_636715

theorem area_of_triangle_ABC (A B C O : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
  (r A B C : ℝ)
  (triangleABC : IsRightIsoscelesTriangle A B C)
  (centerO : IsIncenter O A B C)
  (circ_area : π * r^2 = π)
  (angleA : Angle A = π / 4)
  (angleB : Angle B = π / 2)
  (angleC : Angle C = π / 4) :
  area A B C = 3 / 2 + Real.sqrt 2 :=
by sorry

end area_of_triangle_ABC_l636_636715


namespace measure_A_l636_636172

noncomputable def angle_A (C B A : ℝ) : Prop :=
  C = 3 / 2 * B ∧ B = 30 ∧ A = 180 - B - C

theorem measure_A (A B C : ℝ) (h : angle_A C B A) : A = 105 :=
by
  -- Extract conditions from h
  obtain ⟨h1, h2, h3⟩ := h
  
  -- Use the conditions to prove the thesis
  simp [h1, h2, h3]
  sorry

end measure_A_l636_636172


namespace ellipse_eccentricity_and_circle_inside_l636_636485

theorem ellipse_eccentricity_and_circle_inside (P Q : Type*) (x y : ℝ)
  (hC : x^2 / 4 + y^2 = 1)
  (hD : (x + 1)^2 + y^2 = 1 / 4) :
  (∃ e : ℝ, e = √3 / 2) ∧ 
  (∃ center : ℝ × ℝ, center = (-1, 0) ∧ (center.fst^2 / 4 + center.snd^2 < 1)) :=
by
  sorry

end ellipse_eccentricity_and_circle_inside_l636_636485


namespace part_of_work_in_one_day_l636_636360

variables (work : Type) (A B : work → Prop)
variables (time_A : ℕ) (time_B : ℕ)

-- Assume A can finish the work in 10 days
axiom hA : time_A = 10

-- Assume B can finish the work in half the time taken by A
axiom hB : time_B = time_A / 2

-- The part of the work A can do in one day
def part_A_per_day := 1 / time_A

-- The part of the work B can do in one day
def part_B_per_day := 1 / time_B

-- Work done by A and B together in one day
def total_work_per_day := part_A_per_day + part_B_per_day

-- Prove that A and B together can finish 3/10 of the work in one day
theorem part_of_work_in_one_day : total_work_per_day = 3 / 10 :=
by
  rw [part_A_per_day, part_B_per_day, hA, hB]
  -- part_A_per_day = 1 / 10
  -- part_B_per_day = 1 / (10 / 2) = 1 / 5 = 2 / 10
  have h1 : 1 / 10 + 2 / 10 = 3 / 10 := sorry
  exact h1

end part_of_work_in_one_day_l636_636360


namespace exists_finitely_many_polynomials_l636_636616

-- Define the polynomial r(x) of odd degree with real coefficients
variable {R : Type*} [CommRing R] [IsDomain R] [CharZero R]
noncomputable def r (x : R) : Polynomial R := sorry

-- Define the property that r(x) is of odd degree
axiom odd_degree_r : (r.degree % 2 = 1)

-- Define the existence of polynomials p(x) and q(x^2) such that p^3(x) + q(x^2) = r(x)
theorem exists_finitely_many_polynomials :
  ∃! (pairs : Finset (Polynomial R × Polynomial R)),
    ∃ (p q : Polynomial R), (p, q) ∈ pairs ∧ (p^3 + Polynomial.eval₂ (Polynomial.C.comp Polynomial.C) q (Polynomial.X ^ 2) = r) :=
sorry

end exists_finitely_many_polynomials_l636_636616


namespace find_pairs_l636_636046

theorem find_pairs (x y : ℕ) (h1 : 0 < x ∧ 0 < y)
  (h2 : ∃ p : ℕ, Prime p ∧ (x + y = 2 * p))
  (h3 : (x! + y!) % (x + y) = 0) : ∃ p : ℕ, Prime p ∧ x = p ∧ y = p :=
by
  sorry

end find_pairs_l636_636046


namespace probability_of_selecting_cooking_l636_636778

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636778


namespace sum_of_cubes_l636_636332

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 17) : a^3 + b^3 = 490 := 
sorry

end sum_of_cubes_l636_636332


namespace tan_half_angle_l636_636602

theorem tan_half_angle (a b : ℝ) (h1 : Real.cos a + Real.cos b = 3/5)
    (h2 : Real.sin a + Real.sin b = 1/5) :
    Real.tan ((a + b) / 2) = 1 / 3 :=
by
  sorry

end tan_half_angle_l636_636602


namespace minimum_meals_needed_l636_636597

theorem minimum_meals_needed (total_jam : ℝ) (max_per_meal : ℝ) (jars : ℕ) (max_jar_weight : ℝ):
  (total_jam = 50) → (max_per_meal = 5) → (jars ≥ 50) → (max_jar_weight ≤ 1) →
  (jars * max_jar_weight = total_jam) →
  jars ≥ 12 := sorry

end minimum_meals_needed_l636_636597


namespace probability_of_selecting_cooking_l636_636815

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636815


namespace vasya_dice_sum_l636_636321

theorem vasya_dice_sum
  (dice : Fin 6 → Fin 6) 
  (rolls : Fin 6 → Fin 6 → ℕ)
  (condition1 : ∀ i, (∑ j, rolls i j) = (∑ i, (i : ℕ) + 1))
  (sum1 : ∑ j, rolls 0 j = 21)
  (sum2 : ∑ j, rolls 1 j = 19)
  (sum3 : ∑ j, rolls 2 j = 20)
  (sum4 : ∑ j, rolls 3 j = 18)
  (sum5 : ∑ j, rolls 4 j = 25) :
  ∑ j, rolls 5 j = 23 := by
  sorry

end vasya_dice_sum_l636_636321


namespace parallelogram_altitude_length_l636_636164

theorem parallelogram_altitude_length
  (A B C D E F : Point)
  (h_parallelogram : parallelogram A B C D)
  (h_DE_altitude : altitude_of D E (line A B))
  (h_DF_altitude : altitude_of D F (line B C))
  (h_angle_ADE : ∠ADE = 60)
  (h_DC : dist D C = 18)
  (h_EB : dist E B = 6)
  (h_DE : dist D E = 9) :
  dist D F = 9 :=
sorry

end parallelogram_altitude_length_l636_636164


namespace min_shift_even_function_l636_636285

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + (π / 6))

noncomputable def g (x : ℝ) (ϕ : ℝ) : ℝ := f (x - ϕ)

theorem min_shift_even_function (ϕ : ℝ) (hϕ : ϕ > 0) : ϕ = π / 3 ↔ ∀ x : ℝ, g x ϕ = g (-x) ϕ :=
by
  sorry

end min_shift_even_function_l636_636285


namespace probability_of_selecting_cooking_is_one_fourth_l636_636877

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636877


namespace probability_selecting_cooking_l636_636905

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636905


namespace probability_cooking_selected_l636_636926

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636926


namespace solve_for_pure_imaginary_l636_636524

theorem solve_for_pure_imaginary (x : ℝ) 
  (h1 : x^2 - 1 = 0) 
  (h2 : x - 1 ≠ 0) 
  : x = -1 :=
sorry

end solve_for_pure_imaginary_l636_636524


namespace ABC_three_digit_number_l636_636990

theorem ABC_three_digit_number : 
    ∃ (A B C : ℕ), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    3 * C % 10 = 8 ∧ 
    3 * B + 1 % 10 = 8 ∧ 
    3 * A + 2 = 8 ∧ 
    100 * A + 10 * B + C = 296 := 
by
  sorry

end ABC_three_digit_number_l636_636990


namespace sum_of_10th_degree_polynomials_is_no_higher_than_10_l636_636304

-- Given definitions of two 10th-degree polynomials
def polynomial1 := ∃p : Polynomial ℝ, p.degree = 10
def polynomial2 := ∃p : Polynomial ℝ, p.degree = 10

-- Statement to prove
theorem sum_of_10th_degree_polynomials_is_no_higher_than_10 :
  ∀ (p q : Polynomial ℝ), p.degree = 10 → q.degree = 10 → (p + q).degree ≤ 10 := by
  sorry

end sum_of_10th_degree_polynomials_is_no_higher_than_10_l636_636304


namespace _l636_636614

noncomputable def Q_and_R_exists : Prop :=
  ∃ Q R : ℤ[X], degree R < 2 ∧ (X^2023 + 1) = (X^2 + 1) * Q + R

noncomputable def degree_R := X + 1

noncomputable theorem find_R :
  Q_and_R_exists →
  (∃ R, degree_R = R) :=
by {
  intro h,
  use (X + 1),
  sorry
}

end _l636_636614


namespace parallelogram_has_four_altitudes_l636_636379

-- Define a Parallelogram
structure Parallelogram (A B C D : Type) :=
  (sides_parallel : (A ≠ B → A ≠ D → B ≠ C → C ≠ D) → ((A,B) ≃ (C,D) ∧ (B,C) ≃ (D,A)))

-- Define the concept of an altitude
def altitude (A B C D : Type) [Parallelogram A B C D] :=
  (∀ (P Q : Type), P ≠ Q → ∃! h : ℝ, ∃ (L : Line P Q), L ⊥ (Line A B) ∧ L ⊥ (Line C D))

-- Question translation to Lean statement
theorem parallelogram_has_four_altitudes (A B C D : Type) [Parallelogram A B C D] : 
  ∃ (altitudes : Fin 4 → ℝ), altitudes ≠ ∅ :=
by
  sorry  -- proof omitted

end parallelogram_has_four_altitudes_l636_636379


namespace tan_diff_is_one_third_l636_636129

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

def vector_b : ℝ × ℝ :=
  (2, -1)

theorem tan_diff_is_one_third (θ : ℝ) (h : vector_a θ.1 * vector_b.1 + vector_a θ.2 * vector_b.2 = 0) :
  Real.tan (θ - Real.pi / 4) = 1 / 3 :=
by
  sorry

end tan_diff_is_one_third_l636_636129


namespace probability_of_selecting_cooking_l636_636807

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636807


namespace score_not_possible_l636_636230

theorem score_not_possible (q_correct q_unanswered q_incorrect : ℕ) 
  (q_correct + q_unanswered + q_incorrect = 25)
  (score : ℕ := 6 * q_correct + q_unanswered)
  (h : score = 139) : false :=
sorry

end score_not_possible_l636_636230


namespace division_addition_correct_l636_636999

-- Define a function that performs the arithmetic operations described
def calculateResult : ℕ :=
  let division := 12 * 4 -- dividing 12 by 1/4 is the same as multiplying by 4
  division + 5 -- then add 5 to the result

-- The theorem statement to prove
theorem division_addition_correct : calculateResult = 53 := by
  sorry

end division_addition_correct_l636_636999


namespace carlos_journey_cost_l636_636425

noncomputable def total_minimum_cost (x y z : ℝ) 
  (hXY : y = 3500) (hXZ : x = 4000)
  (train_cost_per_km : ℝ := 0.2) 
  (taxi_cost_per_km : ℝ := 0.15) 
  (taxi_booking_fee : ℝ := 150) : ℝ :=
  let YZ := real.sqrt (x^2 - y^2) in
  let cost_XY_train : ℝ := y * train_cost_per_km in
  let cost_XY_taxi : ℝ := y * taxi_cost_per_km + taxi_booking_fee in
  let cost_XY := min cost_XY_train cost_XY_taxi in
  let cost_YZ_train : ℝ := YZ * train_cost_per_km in
  let cost_YZ_taxi : ℝ := YZ * taxi_cost_per_km + taxi_booking_fee in
  let cost_YZ := min cost_YZ_train cost_YZ_taxi in
  let cost_XZ_train : ℝ := x * train_cost_per_km in
  let cost_XZ_taxi : ℝ := x * taxi_cost_per_km + taxi_booking_fee in
  let cost_XZ := min cost_XZ_train cost_XZ_taxi in
  cost_XY + cost_YZ + cost_XZ

theorem carlos_journey_cost : total_minimum_cost 4000 3500 (real.sqrt (4000^2 - 3500^2)) = 1812.30 :=
by sorry

end carlos_journey_cost_l636_636425


namespace max_value_a_l636_636094

noncomputable def f (a x : ℝ) := x - a / x

theorem max_value_a (a : ℝ) (h_a : a > 0) :
  (∀ x ∈ Icc (1 : ℝ) 2, |(1 + a / 2) * (x - 1) + 1 - a - (x - a / x)| ≤ 1) → 
  a ≤ 6 + 4 * sqrt 2 :=
begin
  intro h,
  -- Proof omitted
  sorry
end

end max_value_a_l636_636094


namespace total_students_math_exam_l636_636160

noncomputable def total_students 
    (sample_size : ℕ) 
    (probability : ℝ) : ℕ :=
    sample_size / probability

theorem total_students_math_exam :
    total_students 50 0.1 = 500 :=
begin
    sorry
end

end total_students_math_exam_l636_636160


namespace smaller_two_digit_product_l636_636686

theorem smaller_two_digit_product (a b : ℕ) (h1: 10 ≤ a) (h2: a ≤ 99) (h3: 10 ≤ b) (h4: b ≤ 99) (h_prod: a * b = 4536) : a = 54 ∨ b = 54 :=
by sorry

end smaller_two_digit_product_l636_636686


namespace dog_running_direction_undeterminable_l636_636368

/-- Given the conditions:
 1. A dog is tied to a tree with a nylon cord of length 10 feet.
 2. The dog runs from one side of the tree to the opposite side with the cord fully extended.
 3. The dog runs approximately 30 feet.
 Prove that it is not possible to determine the specific starting direction of the dog.
-/
theorem dog_running_direction_undeterminable (r : ℝ) (full_length : r = 10) (distance_ran : ℝ) (approx_distance : distance_ran = 30) : (
  ∀ (d : ℝ), d < 2 * π * r → ¬∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π ∧ (distance_ran = r * θ)
  ) :=
by
  sorry

end dog_running_direction_undeterminable_l636_636368


namespace length_of_CD_l636_636105

theorem length_of_CD
    (AB BC AC AD CD : ℝ)
    (h1 : AB = 6)
    (h2 : BC = 1 / 2 * AB)
    (h3 : AC = AB + BC)
    (h4 : AD = AC)
    (h5 : CD = AD + AC) :
    CD = 18 := by
  sorry

end length_of_CD_l636_636105


namespace num_divisible_by_11_l636_636607

-- Definition of a_k as the number obtained by concatenating digits from 1 to k
def a_k (k : ℕ) : ℕ := 
  -- Placeholder definition, the actual definition would concatenate the numbers
  -- This is for conceptual understanding
  sorry 

-- The main statement to prove the number of a_k divisible by 11
theorem num_divisible_by_11 : (finset.range 200).filter (λ k : ℕ, (a_k k % 11 = 0)).card = 36 :=
  sorry

end num_divisible_by_11_l636_636607


namespace find_selling_price_l636_636650

-- Define the parameters based on the problem conditions
constant cost_price : ℝ := 22
constant selling_price_original : ℝ := 38
constant sales_volume_original : ℝ := 160
constant price_reduction_step : ℝ := 3
constant sales_increase_step : ℝ := 120
constant daily_profit_target : ℝ := 3640

-- Define the function representing the sales volume as a function of price reduction
def sales_volume (x : ℝ) : ℝ :=
  sales_volume_original + (x / price_reduction_step) * sales_increase_step

-- Define the function representing the daily profit as a function of price reduction
def daily_profit (x : ℝ) : ℝ :=
  (selling_price_original - x - cost_price) * (sales_volume x)

-- State the main theorem: the new selling price ensuring the desired profit
theorem find_selling_price : ∃ x : ℝ, daily_profit x = daily_profit_target ∧ (selling_price_original - x = 29) :=
by
  sorry

end find_selling_price_l636_636650


namespace initial_books_l636_636371

theorem initial_books (borrowed_day1 : 5 * 2 = 10)
                      (borrowed_day2 : 20 = 20)
                      (remaining : 70 = 70) :
                      ∃ initial_books, initial_books = 100 :=
by 
    let total_borrowed := 10 + 20
    have total_borrowed_def : total_borrowed = 30 := by sorry
    have initial_count := remaining + total_borrowed
    have initial_count_def : initial_count = 100 := by sorry
    exact ⟨initial_count, initial_count_def⟩

end initial_books_l636_636371


namespace ellipse_eccentricity_and_circle_inside_l636_636486

theorem ellipse_eccentricity_and_circle_inside (P Q : Type*) (x y : ℝ)
  (hC : x^2 / 4 + y^2 = 1)
  (hD : (x + 1)^2 + y^2 = 1 / 4) :
  (∃ e : ℝ, e = √3 / 2) ∧ 
  (∃ center : ℝ × ℝ, center = (-1, 0) ∧ (center.fst^2 / 4 + center.snd^2 < 1)) :=
by
  sorry

end ellipse_eccentricity_and_circle_inside_l636_636486


namespace probability_selecting_cooking_l636_636900

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636900


namespace parabola_intersects_line_exactly_once_l636_636772

theorem parabola_intersects_line_exactly_once (p q : ℚ) : 
  (∀ x : ℝ, 2 * (x - p) ^ 2 = x - 4 ↔ p = 31 / 8) ∧ 
  (∀ x : ℝ, 2 * x ^ 2 - q = x - 4 ↔ q = 31 / 8) := 
by 
  sorry

end parabola_intersects_line_exactly_once_l636_636772


namespace probability_of_selecting_cooking_is_one_fourth_l636_636875

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636875


namespace geometric_sequence_sum_l636_636093

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (n : ℕ) (h_geom : geometric_sequence a) (h_a2 : a 2 = 2) (h_a5 : a 5 = 1/4) :
  (∑ k in finset.range n, a k * a (k + 1)) = (32 / 3) * (1 - 4^(-n)) := 
sorry

end geometric_sequence_sum_l636_636093


namespace evaluate_expression_at_x_neg3_l636_636042

theorem evaluate_expression_at_x_neg3 :
  (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 :=
by
  sorry

end evaluate_expression_at_x_neg3_l636_636042


namespace Nikita_prime_sequence_l636_636223

def is_prime (n : ℕ) : Prop := ¬∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = n

theorem Nikita_prime_sequence :
  ∃ a : Fin 9 → ℕ,
    a 0 = 5 ∧
    a 1 = 3 ∧
    a 2 = 3 ∧
    (∀ n : Fin 6, a n + 3 = a n + a n.succ + a (Fin.succ n.succ)) ∧
    (∀ n : Fin 9, is_prime (a n)) ∧
    (a 0 = 5) ∧
    (a 1 = 3) ∧
    (a 2 = 3) ∧
    (a 3 = 11) ∧
    (a 4 = 17) ∧
    (a 5 = 31) ∧
    (a 6 = 59) ∧
    (a 7 = 107) ∧
    (a 8 = 197)
:=
begin
  sorry
end

end Nikita_prime_sequence_l636_636223


namespace problemI_solution_set_problemII_range_of_a_l636_636502

section ProblemI

def f (x : ℝ) := |2 * x - 2| + 2

theorem problemI_solution_set :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end ProblemI

section ProblemII

def f (x a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

theorem problemII_range_of_a :
  {a : ℝ | ∀ x : ℝ, f x a + g x ≥ 3} = {a : ℝ | 2 ≤ a} :=
by
  sorry

end ProblemII

end problemI_solution_set_problemII_range_of_a_l636_636502


namespace probability_of_selecting_cooking_l636_636780

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636780


namespace xiaoming_selects_cooking_probability_l636_636824

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636824


namespace speed_increase_percentage_l636_636590

variable (T : ℚ)  -- usual travel time in minutes
variable (v : ℚ)  -- usual speed

-- Conditions
-- Ivan usually arrives at 9:00 AM, traveling for T minutes at speed v.
-- When Ivan leaves 40 minutes late and drives 1.6 times his usual speed, he arrives at 8:35 AM
def usual_arrival_time : ℚ := 9 * 60  -- 9:00 AM in minutes

def time_when_late : ℚ := (9 * 60) + 40 - (25 + 40)  -- 8:35 AM in minutes

def increased_speed := 1.6 * v -- 60% increase in speed

def time_taken_with_increased_speed := T - 65

theorem speed_increase_percentage :
  ((T / (T - 40)) = 1.3) :=
by
-- assume the equation for usual time T in terms of increased speed is known
-- Use provided conditions and solve the equation to derive the result.
  sorry

end speed_increase_percentage_l636_636590


namespace maximum_value_tan_l636_636499

-- Given function f(x)
def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x

-- Main theorem statement (proof problem)
theorem maximum_value_tan (x : ℝ) (h_max : ∀ y : ℝ, f(x) ≥ f(y)) : Real.tan x = 3 / 2 :=
sorry

end maximum_value_tan_l636_636499


namespace gcf_of_18_and_10_l636_636319

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b  -- using the definition from Mathlib

theorem gcf_of_18_and_10 :
  (n : ℕ) (hn : n = 18) (hlcm : lcm n 10 = 36) →
  Nat.gcd 18 10 = 5 := by
  sorry

end gcf_of_18_and_10_l636_636319


namespace probability_cooking_is_one_fourth_l636_636857
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636857


namespace probability_of_selecting_cooking_l636_636786

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636786


namespace range_of_angle_A_l636_636177

theorem range_of_angle_A (a b : ℝ) (A : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) 
  (h_triangle : 0 < A ∧ A ≤ Real.pi / 4) :
  (0 < A ∧ A ≤ Real.pi / 4) :=
by
  sorry

end range_of_angle_A_l636_636177


namespace sum_of_solutions_of_absolute_value_l636_636728

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l636_636728


namespace symmetric_line_eq_l636_636282

theorem symmetric_line_eq (x y : ℝ) (h₁ : y = 3 * x + 4) : y = x → y = (1 / 3) * x - (4 / 3) :=
by
  sorry

end symmetric_line_eq_l636_636282


namespace probability_select_cooking_l636_636886

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636886


namespace problem_statement_l636_636530

-- Define the function to calculate the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

-- Define the function f as sum of digits of n^2 + 1
def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

-- Define the iterated function f_k
@[simp]
def f_iter : ℕ → ℕ → ℕ
| 0, n := n
| (k + 1), n := f (f_iter k n)

-- Definition of the problem statement
theorem problem_statement : f_iter 2010 8 = 8 :=
by
  sorry

end problem_statement_l636_636530


namespace sin_neg_2055_eq_l636_636353

theorem sin_neg_2055_eq :
  sin (-2055 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
sorry

end sin_neg_2055_eq_l636_636353


namespace students_like_cricket_l636_636550

-- Definitions based on the given conditions
def B : ℕ := 10 -- Number of students who like basketball
def B_and_C : ℕ := 4 -- Number of students who like both basketball and cricket
def B_or_C : ℕ := 14 -- Number of students who like basketball or cricket or both

-- The proof statement
theorem students_like_cricket : B_or_C = B + sorry - B_and_C → 14 = 10 + C - 4 → 8 := 
by sorry

end students_like_cricket_l636_636550


namespace speed_of_first_car_l636_636315

theorem speed_of_first_car (v : ℝ) (h1 : 2.5 * v + 2.5 * 45 = 175) : v = 25 :=
by
  sorry

end speed_of_first_car_l636_636315


namespace Joseph_next_test_score_l636_636595

theorem Joseph_next_test_score (scores : List ℕ)
  (h : scores = [95, 85, 75, 85, 90]) :
  let current_average := (scores.sum / scores.length : ℕ)
  let target_average := current_average + 5
  let total_score := scores.sum
  let minimum_next_score := target_average * (scores.length + 1) - total_score
  minimum_next_score = 116 :=
by
  rw [h]
  simp [List.sum, List.length, target_average, current_average, minimum_next_score]
  sorry

end Joseph_next_test_score_l636_636595


namespace alternating_sequence_possible_l636_636313

-- Define the problem parameters and conditions
def Box : Type := List Char  -- Representing each box containing either 'A' or 'B'
def op (boxes : Box) (a b : Nat) : Box :=
  -- Function to represent moving adjacent pieces
  sorry -- to be implemented based on piece moving mechanics

-- Statement of the proof problem
theorem alternating_sequence_possible (n : ℕ) (h : n ≥ 3)
  (initial : Box) (config : Box) :
  ∃ (T : List ℤ), (List.length T = n) ∧ 
  (∀ (i : ℤ), i ∈ T → i ≠ 0) ∧ 
  (op_n_times T initial = config) ∧ 
  (alternate config) := 
begin
  -- Details and depths would be implemented in the proof steps
  sorry -- Proof to be implemented
end

end alternating_sequence_possible_l636_636313


namespace common_sum_rectangle_l636_636287

theorem common_sum_rectangle : 
  let rectangle := fin 6 × finite 5
  let integers := fin 30
  let total_sum := ∑ i in integers, i + 1
  let common_sum_per_division := total_sum / 5
  (S : ℕ) × (∀ row : fin 6, ∑ i in fin 5, (rectangle.row row).i) = S × 
  (∀ col : finite 5, ∑ i in fin 6, (rectangle.col col).i) = S × 
  (∀ diagonal, sum of elements in the diagonal is S)
  (∀ quadrant, sum of elements in the quadrant is S)
  S = common_sum_per_division

  (rectangle, integers, total_sum, common_sum_per_division)
  := 93
:= 
1 60 sorry

end common_sum_rectangle_l636_636287


namespace part1_part2_part3_l636_636493

-- Given sequences and conditions
def S (n : ℕ) : ℕ := n * (n + 1) / 2

def b : ℕ → ℕ
| 1 := 3
| (n + 1) := 2 * (b n) - 1

def a (n : ℕ) : ℕ := n

def c (n : ℕ) : ℚ :=
  (-1:ℚ) ^ n * (2 * (a n) + 1) / ((a n) + 1) / (Real.log2 (b n - 1))

def T (n : ℕ) : ℚ :=
  -1 + (-1:ℚ) ^ n / (n + 1)

def d : ℕ → ℚ
| (n + 1) := if n % 2 = 0 then a (n + 2) / (a (n + 1) ^ 2 * a (n + 3) ^ 2) else a (2 * (n + 1)) / b (n + 1)

-- Part (1): Proving {b_n - 1} is a geometric sequence
theorem part1 : ∀ (n : ℕ), (b (n + 1) - 1) = 2 * (b n - 1) :=
sorry

-- Part (2): Finding sum T_n
theorem part2 (n : ℕ) : ∑ k in Finset.range n, c (k + 1) = T n :=
sorry

-- Part (3): Proving sum of d_k less than 9/4
theorem part3 (n : ℕ) : ∑ k in Finset.range (2 * n), d (k + 1) < 9 / 4 :=
sorry

end part1_part2_part3_l636_636493


namespace probability_cooking_is_one_fourth_l636_636861
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636861


namespace dragon_boat_festival_problem_l636_636656

theorem dragon_boat_festival_problem :
  ∃ (x : ℝ), 
    let cp := 22 in
    let sp := 38 in
    let q := 160 in
    let dp := 3640 in
    let profit := (sp - x - cp) in
    let new_q := q + (x / 3) * 120 in
    profit * new_q = dp ∧ 
    sp - x = 29 :=
begin
  sorry
end

end dragon_boat_festival_problem_l636_636656


namespace inequality_proof_l636_636114

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x
noncomputable def g (a x : ℝ) : ℝ := x + 1 / x + a / x
noncomputable def h (a x : ℝ) : ℝ := g a x - f a x

theorem inequality_proof (a x1 x2 m : ℝ) (h_extremum : h a (1 + a) = 3) 
    (h_roots : f a x1 + m * x1 = 0 ∧ f a x2 + m * x2 = 0)
    (h_ratio : x2 / x1 ≥ Real.exp a) :
  (f' (x1 + x2) + m) / f' (x1 - x2) > 6 / 5 :=
by
  sorry

end inequality_proof_l636_636114


namespace max_min_f_on_interval_l636_636068

def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x + 1

theorem max_min_f_on_interval :
  let I := set.Icc (-4 : ℝ) (4 : ℝ) in
  ∃ max_val min_val : ℝ,
    max_val = 77 ∧ min_val = -4 ∧
    (∀ x ∈ I, f x ≤ max_val) ∧
    (∀ x ∈ I, f x ≥ min_val) :=
by
  let I := set.Icc (-4 : ℝ) (4 : ℝ)
  have h1 : ∀ x ∈ I, f x ≤ 77 := sorry
  have h2 : ∀ x ∈ I, f x ≥ -4 := sorry
  exact ⟨77, -4, rfl, rfl, h1, h2⟩

end max_min_f_on_interval_l636_636068


namespace probability_of_selecting_cooking_is_one_fourth_l636_636868

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636868


namespace first_term_of_geometric_series_l636_636416

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l636_636416


namespace real_roots_quadratic_l636_636081

theorem real_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 1 = 0) ↔ (m ≥ -5/4 ∧ m ≠ 1) := by
  sorry

end real_roots_quadratic_l636_636081


namespace percentage_cut_l636_636705

def original_budget : ℝ := 840
def cut_amount : ℝ := 588

theorem percentage_cut : (cut_amount / original_budget) * 100 = 70 :=
by
  sorry

end percentage_cut_l636_636705


namespace calculation_l636_636423

theorem calculation : (⌊|(-5.8:ℝ)|⌋ + |⌊ -5.8 ⌋| + ⌊ -|⌊ -5.8 ⌋| ⌋ : ℤ) = 5 := by
  sorry

end calculation_l636_636423


namespace largest_of_five_consecutive_integers_with_product_15120_l636_636457

theorem largest_of_five_consecutive_integers_with_product_15120 :
  ∃ (a b c d e : ℕ), a * b * c * d * e = 15120 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e = 12 :=
begin
  sorry
end

end largest_of_five_consecutive_integers_with_product_15120_l636_636457


namespace dragon_boat_festival_problem_l636_636654

theorem dragon_boat_festival_problem :
  ∃ (x : ℝ), 
    let cp := 22 in
    let sp := 38 in
    let q := 160 in
    let dp := 3640 in
    let profit := (sp - x - cp) in
    let new_q := q + (x / 3) * 120 in
    profit * new_q = dp ∧ 
    sp - x = 29 :=
begin
  sorry
end

end dragon_boat_festival_problem_l636_636654


namespace line_length_infinite_sum_l636_636373

noncomputable def infinite_series_sum : ℝ :=
  0.5 + (∑ i in (finset.range 100), (1 / (3 : ℝ)^(i+1)) * (if (i % 2 == 0) then real.sqrt 3 else 1))

theorem line_length_infinite_sum :
  infinite_series_sum = (3 + 2 * real.sqrt 3) / 4 :=
  sorry

end line_length_infinite_sum_l636_636373


namespace number_of_possibilities_l636_636384

-- Condition definitions
variable (a b : ℕ)
variable (h1 : b > a)
variable (h2 : a ≥ 6 ∧ b ≥ 6)

noncomputable def painted_area := (a - 4) * (b - 4)
noncomputable def unpainted_area := a * b - painted_area
noncomputable def double_painted_area := 2 * painted_area

-- Main statement
theorem number_of_possibilities (h3 : unpainted_area = double_painted_area) :
  ∃ s : finset (ℕ × ℕ), s.card = 2 ∧ ∀ p ∈ s, p.1 > 6 ∧ p.2 > 6 ∧ p.1 < p.2 :=
by
  sorry

end number_of_possibilities_l636_636384


namespace initial_bacteria_count_l636_636663

theorem initial_bacteria_count (n : ℕ) : 
  (n * 4^10 = 4194304) → n = 4 :=
by
  sorry

end initial_bacteria_count_l636_636663


namespace sin_sum_greater_than_cos_sum_l636_636557

variable {A B C : ℝ}

-- Condition: Triangle ABC is an acute triangle
axiom acute_triangle (hA : A > 0) (hB : B > 0) (hC : C > 0) (hSum : A + B + C = π) : A < π / 2 ∧ B < π / 2 ∧ C < π / 2

theorem sin_sum_greater_than_cos_sum (hA : A > 0) (hB : B > 0) (hC : C > 0) (hSum : A + B + C = π) (acute : acute_triangle hA hB hC hSum) :
  sin A + sin B + sin C > cos A + cos B + cos C := by
  sorry

end sin_sum_greater_than_cos_sum_l636_636557


namespace probability_select_cooking_l636_636890

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636890


namespace solve_for_x_l636_636713

-- Definitions based on conditions
def WorkRateA : Type := ℚ
def WorkRateB : Type := ℚ
def WorkRateC : Type := ℚ

variables (A : WorkRateA) (B : WorkRateB) (C : WorkRateC) (x : ℚ)

-- Conditions
def condition1 : Prop := 15 * A + 7 * B = 1 / x
def condition2 : Prop := 8 * B + 15 * C = 1 / 11
def condition3 : Prop := A + B + C = 1 / 44

-- The theorem we need to prove
theorem solve_for_x (h1 : condition1 A B x) (h2 : condition2 B C) (h3 : condition3 A B C) : x = 4 :=
sorry

end solve_for_x_l636_636713


namespace quadratic_has_root_in_interval_l636_636243

theorem quadratic_has_root_in_interval (a b c : ℝ) (h : 2 * a + 3 * b + 6 * c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end quadratic_has_root_in_interval_l636_636243


namespace number_under_35_sampled_l636_636363

-- Define the conditions
def total_employees : ℕ := 500
def employees_under_35 : ℕ := 125
def employees_35_to_49 : ℕ := 280
def employees_over_50 : ℕ := 95
def sample_size : ℕ := 100

-- Define the theorem stating the desired result
theorem number_under_35_sampled : (employees_under_35 * sample_size / total_employees) = 25 :=
by
  sorry

end number_under_35_sampled_l636_636363


namespace probability_of_selecting_cooking_l636_636936

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636936


namespace find_X_l636_636525

theorem find_X : 
  let M := 3012 / 4
  let N := M / 4
  let X := M - N
  X = 564.75 :=
by
  sorry

end find_X_l636_636525


namespace sum_divide_product_condition_l636_636338

theorem sum_divide_product_condition (n : ℕ) (hn : ∃ k : ℕ, (n! : ℤ) = k * (n * (n+1) / 2)) : 
  ∀ p : ℕ, prime p → n ≠ p - 1 :=
sorry

end sum_divide_product_condition_l636_636338


namespace probability_cooking_selected_l636_636924

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636924


namespace sequence_limit_l636_636696

variable (a b : ℝ)
noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then a else
  if n = 1 then b else
  (sequence (n-2) + (2 * (n-1) - 1) * sequence (n-1)) / (2 * (n-1))

theorem sequence_limit (a b : ℝ) : 
  (∀ n : ℕ, sequence a b n = if n = 0 then a else if n = 1 then b else (sequence a b (n-1) + (2 * n - 1) * sequence a b n) / (2 * n)) →
  lim (sequence a b) = b :=
sorry

end sequence_limit_l636_636696


namespace probability_selecting_cooking_l636_636903

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636903


namespace xiaoming_selects_cooking_probability_l636_636830

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636830


namespace probability_select_cooking_l636_636885

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636885


namespace tangents_parallel_from_non_collinear_points_l636_636514

-- Define the points O, A, and B
variables (O A B : Point)

-- Assume O, A, and B are non-collinear
axiom non_collinear : ¬ collinear O A B

-- The statement of our theorem
theorem tangents_parallel_from_non_collinear_points :
  ∃ (r : ℝ), ∃ (circle : Circle),
    center circle = O ∧
    radius circle = r ∧
    tangent_from A circle ∥ tangent_from B circle :=
sorry

end tangents_parallel_from_non_collinear_points_l636_636514


namespace avg_seq2_eq_l636_636662

-- Definition of the arithmetic sequence and its average
def seq1 := list.range' 1 20     -- This is the list [1, 2, ..., 20]
def seq1_avg := (seq1.sum : ℚ) / seq1.length   -- The average of the sequence

-- Given condition: average of first sequence is a
variable (a : ℚ)
axiom h1 : seq1_avg = a

-- Definition of the second sequence based on the first
def seq2 := seq1.map (λ x, 3 * x + 1)
def seq2_avg := (seq2.sum : ℚ) / seq2.length 

-- The theorem we need to prove
theorem avg_seq2_eq : seq2_avg = 3 * a + 1 :=
sorry

end avg_seq2_eq_l636_636662


namespace count_rational_numbers_l636_636395

def numbers := [3, 0, - Math.PI / 2, 15 / 11, 0.2121121112 , -8.24]

noncomputable def is_rational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem count_rational_numbers : (list.filter is_rational numbers).length = 4 :=
by
  sorry

end count_rational_numbers_l636_636395


namespace find_x_and_y_l636_636623

theorem find_x_and_y :
  (∃ x y : ℕ, y = x^2 + 5 * x - 12 ∧ 15 * x = x + 280 ∧ x = 20 ∧ y = 488) :=
by {
  use 20,
  use 488,
  split,
  {
    -- proof that y = 488
    exact (calc
      488 = (20 ^ 2 + 5 * 20 - 12) : by sorry
    ),
  },
  split,
  {
    -- proof that 15 * x = x + 280
    exact (calc
      15 * 20 = 20 + 280 : by sorry
    ),
  },
  split,
  {
    -- proof that x = 20
    refl,
  },
  {
    -- proof that y = 488
    refl,
  }
}

end find_x_and_y_l636_636623


namespace sum_of_n_for_3n_minus_8_eq_5_l636_636743

theorem sum_of_n_for_3n_minus_8_eq_5 : 
  let S := {n | |3 * n - 8| = 5}
  in S.sum id = 16 / 3 :=
by
  sorry

end sum_of_n_for_3n_minus_8_eq_5_l636_636743


namespace triangle_inequality_theorem_l636_636757

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality_theorem :
  ¬ is_triangle 2 3 5 ∧ is_triangle 5 6 10 ∧ ¬ is_triangle 1 1 3 ∧ ¬ is_triangle 3 4 9 :=
by {
  -- Proof goes here
  sorry
}

end triangle_inequality_theorem_l636_636757


namespace students_chose_water_l636_636556

theorem students_chose_water (total_students : ℕ)
  (h1 : 75 * total_students / 100 = 90)
  (h2 : 25 * total_students / 100 = x) :
  x = 30 := 
sorry

end students_chose_water_l636_636556


namespace xiaoming_selects_cooking_probability_l636_636834

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636834


namespace last_two_digits_of_7_pow_2017_l636_636629

theorem last_two_digits_of_7_pow_2017 : (7^2017) % 100 = 7 :=
by
  -- conditions from the problem
  have h2 : (7^2) % 100 = 49 := by norm_num
  have h3 : (7^3) % 100 = 43 := by norm_num
  have h4 : (7^4) % 100 = 01 := by norm_num
  have h5 : (7^5) % 100 = 07 := by norm_num
  have p : ∀ n ≥ 2, (7^n) % 100 = (7^(n % 4 + 2)) % 100 :=
    sorry -- This condition encodes the periodic behavior with period 4

  show (7^2017) % 100 = 7 from
  by
    have h_period : 2017 % 4 = 1 := by norm_num
    calc
      (7^2017) % 100 = (7^(2017 % 4 + 2)) % 100 : by apply p; norm_num
                  ... = (7^3) % 100             : by rw [h_period]
                  ... = 7                       : by rw [h5]

end last_two_digits_of_7_pow_2017_l636_636629


namespace sum_of_solutions_abs_eq_l636_636733

-- Define the condition on n
def abs_eq (n : ℝ) : Prop := abs (3 * n - 8) = 5

-- Statement we want to prove
theorem sum_of_solutions_abs_eq : ∑ n in { n | abs_eq n }, n = 16/3 :=
by
  sorry

end sum_of_solutions_abs_eq_l636_636733


namespace part_a_part_b_l636_636766

-- Define the strategy type
def Strategy := Nat → (Nat → Bool) → Nat

-- Define the specific bisection strategy S_n
def S_n (N : Nat) : Strategy := λ n ask =>
  let rec bisect_count (low high count : Nat) : Nat :=
    if low >= high then count
    else
      let mid := (low + high) / 2
      if ask mid then bisect_count mid high (count + 1)
      else bisect_count low mid (count + 1)
  bisect_count 1 N 0

-- Define the function f_T for a given strategy
def f_T (T : Strategy) (n : Nat) : Nat :=
  T n (λ x => n ≥ x)

-- Define the function \( \bar{f}_T \)
def bar_f_T (T : Strategy) (n : Nat) : Nat :=
  Nat.maximum (List.map (f_T T) (List.range n).succ)

-- Define the main theorem statements
theorem part_a (N : Nat) : ∀ n, 1 ≤ n ∧ n ≤ N → f_T (S_n N) n = Nat.log2 N + 1 :=
  by sorry

theorem part_b (N : Nat) : ∀ T, ∀ n, 1 ≤ n ∧ n ≤ N → bar_f_T T n ≥ Nat.log2 N + 1 :=
  by sorry

end part_a_part_b_l636_636766


namespace distance_parallel_lines_l636_636051

noncomputable def distance_between_parallel_lines (A B : Real) (C D : Real) : Real :=
  abs(A - C) / sqrt(1 + B^2)

theorem distance_parallel_lines :
  distance_between_parallel_lines 5 (-3) (-4) (-3) = 9 * sqrt 10 / 10 := by
  sorry

end distance_parallel_lines_l636_636051


namespace bridge_length_l636_636683

noncomputable def speed_km_per_hr_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def distance_travelled (speed_m_per_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (train_length_condition : train_length = 150) 
  (train_speed_condition : train_speed_kmph = 45) 
  (crossing_time_condition : crossing_time_s = 30) :
  (distance_travelled (speed_km_per_hr_to_m_per_s train_speed_kmph) crossing_time_s - train_length) = 225 :=
by 
  sorry

end bridge_length_l636_636683


namespace wide_flag_height_is_3_l636_636435

-- Definitions of the conditions
def initial_fabric : ℕ := 1000
def fabric_left : ℕ := 294
def square_flag_side : ℕ := 4
def wide_flag_width : ℕ := 5
def tall_flag_width : ℕ := 3
def tall_flag_height : ℕ := 5
def num_square_flags : ℕ := 16
def num_wide_flags : ℕ := 20
def num_tall_flags : ℕ := 10

-- Main theorem to prove the height of the wide rectangular flags
theorem wide_flag_height_is_3 : 
  let
    square_flag_area := square_flag_side ^ 2,
    total_square_area := num_square_flags * square_flag_area,
    tall_flag_area := tall_flag_width * tall_flag_height,
    total_tall_area := num_tall_flags * tall_flag_area,
    total_used_area := total_square_area + total_tall_area,
    wide_flag_area_sum := initial_fabric - fabric_left - total_used_area,
    height_of_wide_flags := wide_flag_area_sum / (num_wide_flags * wide_flag_width)
  in height_of_wide_flags = 3 := by
  sorry

end wide_flag_height_is_3_l636_636435


namespace number_of_purifiers_purchased_min_selling_price_A_l636_636710

-- Conditions
variables (x y : ℕ)
variables (purchase_price_A purchase_price_B total_cost units_total : ℕ)
variables (gross_profit_A gross_profit_B min_gross_profit : ℕ)
variables (min_selling_price_A min_selling_price_B : ℕ)

-- Given
axiom units_constraint : x + y = 160
axiom cost_constraint : 150 * x + 350 * y = 36000
axiom gross_profit_constraint : 100 * gross_profit_A + 60 * (2 * gross_profit_A) ≥ 11000
axiom purchase_prices : purchase_price_A = 150 ∧ purchase_price_B = 350
axiom total_units : units_total = 160
axiom minimum_gross_profit : min_gross_profit = 11000

-- Prove
theorem number_of_purifiers_purchased : x = 100 ∧ y = 60 :=
by { sorry }

theorem min_selling_price_A :
  let gp_A := gross_profit_A in
  let gp_B := gross_profit_B in
  let sp_A := min_selling_price_A in
  (gp_B = 2 * gp_A) →
  (gp_A ≥ 50) →
  (sp_A = purchase_price_A + gp_A) →
  sp_A = 200 :=
by { sorry }

end number_of_purifiers_purchased_min_selling_price_A_l636_636710


namespace sum_of_solutions_abs_eq_l636_636730

-- Define the condition on n
def abs_eq (n : ℝ) : Prop := abs (3 * n - 8) = 5

-- Statement we want to prove
theorem sum_of_solutions_abs_eq : ∑ n in { n | abs_eq n }, n = 16/3 :=
by
  sorry

end sum_of_solutions_abs_eq_l636_636730


namespace part_i_solution_set_part_ii_range_m_l636_636213

def f (x : ℝ) : ℝ := |x - 2| - |2 * x + 1|

theorem part_i_solution_set :
  { x : ℝ | f x > 0 } = set.Ioo (-3 : ℝ) (1 / 3) :=
by sorry

theorem part_ii_range_m (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ > 2 * m + 1) → m < 3 / 4 :=
by sorry

end part_i_solution_set_part_ii_range_m_l636_636213


namespace solve_quadratic_eq_l636_636647

theorem solve_quadratic_eq (x : ℝ) (h : x > 0) (eq : 4 * x^2 + 8 * x - 20 = 0) : 
  x = Real.sqrt 6 - 1 :=
sorry

end solve_quadratic_eq_l636_636647


namespace probability_of_selecting_cooking_l636_636927

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636927


namespace smallest_sphere_tangent_l636_636052

theorem smallest_sphere_tangent 
(l₁ : ℝ → ℝ × ℝ × ℝ := λ t, (t + 1, 2 * t + 4, -3 * t + 5))
(l₂ : ℝ → ℝ × ℝ × ℝ := λ t, (4 * t - 12, -t + 8, t + 17)) :
∃ (C : ℝ × ℝ × ℝ) (r : ℝ), 
  ( C = (-915 / 502, 791 / 502, 8525 / 502) ) ∧ ( r = 2065 / 251 ) :=
by {
  sorry
}

end smallest_sphere_tangent_l636_636052


namespace probability_of_selecting_cooking_l636_636937

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636937


namespace geom_series_first_term_l636_636404

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l636_636404


namespace fiona_received_59_l636_636987

theorem fiona_received_59 (Dan_riddles : ℕ) (Andy_riddles : ℕ) (Bella_riddles : ℕ) (Emma_riddles : ℕ) (Fiona_riddles : ℕ)
  (h1 : Dan_riddles = 21)
  (h2 : Andy_riddles = Dan_riddles + 12)
  (h3 : Bella_riddles = Andy_riddles - 7)
  (h4 : Emma_riddles = Bella_riddles / 2)
  (h5 : Fiona_riddles = Andy_riddles + Bella_riddles) :
  Fiona_riddles = 59 :=
by
  sorry

end fiona_received_59_l636_636987


namespace trapezoid_split_ratio_l636_636003

-- Definitions of the conditions
def is_trapezoid (B1 B2 a b : ℝ) : Prop := B1 ≠ B2

def equal_perimeter_split (B1 B2 a b : ℝ) (m n : ℝ) : Prop :=
  B1 + m + a * (m/(m + n)) = B2 + n + a * (n/(m + n))

theorem trapezoid_split_ratio
  (B1 B2 a b : ℝ) (h_trap : is_trapezoid B1 B2) (h_base1 : B1 = 3) (h_base2 : B2 = 9)
  (h_leg1 : a = 4) (h_leg2 : b = 6) (m n : ℝ)
  (h_sum : m + n = 6) (h_eq_perimeter : equal_perimeter_split B1 B2 a b m n) :
  m / n = 4 :=
  sorry

end trapezoid_split_ratio_l636_636003


namespace find_smallest_shift_l636_636251

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)

noncomputable def g (x ϕ : ℝ) : ℝ := f (x - ϕ)

def is_odd_function (h : ℝ → ℝ) : Prop :=
  ∀ x, h x = -h (-x)

theorem find_smallest_shift {ϕ : ℝ} (hϕ : ϕ = π / 6) :
  is_odd_function (g ϕ) ↔ ∃ k : ℤ, ϕ = (π / 6 - k * (π / 2)) := sorry

end find_smallest_shift_l636_636251


namespace angle_of_given_vectors_l636_636132

open Real

def vector (α : Type*) := (α × α)

def dot_product (a b : vector ℝ) : ℝ := 
  a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : vector ℝ) : ℝ :=
  sqrt (a.1^2 + a.2^2)

noncomputable def angle_between (a b : vector ℝ) : ℝ :=
  arccos (dot_product a b / (magnitude a * magnitude b))

theorem angle_of_given_vectors : 
  let a := (1, 2) : vector ℝ
  let b := (-2, 1) : vector ℝ
  angle_between a b = π / 2 :=
by sorry

end angle_of_given_vectors_l636_636132


namespace andy_ate_3_cookies_l636_636986

theorem andy_ate_3_cookies (initial_cookies : ℕ) (given_brother : ℕ) 
  (players : ℕ) (first_term : ℕ) (diff : ℕ) (total_cookies_taken_by_players : ℕ) (remaining_cookies: ℕ):
  initial_cookies = 72 →
  given_brother = 5 →
  players = 8 →
  first_term = 1 →
  diff = 2 →
  total_cookies_taken_by_players = 64 →
  remaining_cookies = initial_cookies - (total_cookies_taken_by_players + given_brother) →
  remaining_cookies = 3 :=
by {
  intros,
  subst_vars,
  exact rfl,
}

end andy_ate_3_cookies_l636_636986


namespace member_number_exists_l636_636417

theorem member_number_exists (members : Finset ℕ) (num_countries : ℕ) (total_members : ℕ) 
    (members_per_country : ℕ → Finset ℕ)
    (h_total : total_members = 1978)
    (h_num_countries : num_countries = 6)
    (h_members : members = Finset.range (total_members + 1))
    (h_country_members : ∀ c, members_per_country c ⊆ members ∧ members_per_country c ≠ ∅ ∧
                              members_per_country c.card = ⌈total_members / num_countries⌉) :
  ∃ n ∈ members, ∃ c, ∃ x ∈ members_per_country c, ∃ y ∈ members_per_country c, 
    n = x + y ∨ ∃ z ∈ members_per_country c, n = 2 * z :=
by 
  sorry

end member_number_exists_l636_636417


namespace xiaoming_selects_cooking_probability_l636_636829

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636829


namespace find_linear_odd_increasing_function_l636_636104

theorem find_linear_odd_increasing_function (f : ℝ → ℝ)
    (h1 : ∀ x, f (f x) = 4 * x)
    (h2 : ∀ x, f x = -f (-x))
    (h3 : ∀ x y, x < y → f x < f y)
    (h4 : ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x) : 
    ∀ x, f x = 2 * x :=
by
  sorry

end find_linear_odd_increasing_function_l636_636104


namespace plane_divides_pyramid_l636_636270

variables {V : ℝ} {A B C D P N K M : ℝ} [DecidableEq ℝ]

/-- Given conditions on points in pyramid PABCD -/
def midpoint (x y : ℝ) := (x + y) / 2

def pm_eq_5mb (PB PM MB : ℝ) := PM = 5 * MB

theorem plane_divides_pyramid {a V : ℝ} (h1 : N = midpoint A P) (h2 : K = midpoint P L)
  (h3 : PM = 5 * MB) (h4 : volume_ratio : ℝ) :
  (volume_ratio = 25/227) :=
sorry

end plane_divides_pyramid_l636_636270


namespace mass_percentage_O_HClO₂_is_correct_l636_636054

-- Define the molar masses
def molar_mass_H : ℝ := 1.01
def molar_mass_Cl : ℝ := 35.45
def molar_mass_O : ℝ := 16.00

-- Define the chemical formula and resulting molar mass
def molar_mass_HClO₂ : ℝ := (1 * molar_mass_H) + (1 * molar_mass_Cl) + (2 * molar_mass_O)

-- Define the mass percentage calculation
def mass_percentage_O_in_HClO₂ : ℝ := ((2 * molar_mass_O) / molar_mass_HClO₂) * 100

-- Theorem statement
theorem mass_percentage_O_HClO₂_is_correct : mass_percentage_O_in_HClO₂ = 46.75 := by
  sorry

end mass_percentage_O_HClO₂_is_correct_l636_636054


namespace probability_cooking_selected_l636_636917

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636917


namespace trivia_team_total_points_l636_636389

def totalPoints : Nat := 182

def points_member_A : Nat := 3 * 2
def points_member_B : Nat := 5 * 4 + 1 * 6
def points_member_C : Nat := 2 * 6
def points_member_D : Nat := 4 * 2 + 2 * 4
def points_member_E : Nat := 1 * 2 + 3 * 4
def points_member_F : Nat := 5 * 6
def points_member_G : Nat := 2 * 4 + 1 * 2
def points_member_H : Nat := 3 * 6 + 2 * 2
def points_member_I : Nat := 1 * 4 + 4 * 6
def points_member_J : Nat := 7 * 2 + 1 * 4

theorem trivia_team_total_points : 
  points_member_A + points_member_B + points_member_C + points_member_D + points_member_E + 
  points_member_F + points_member_G + points_member_H + points_member_I + points_member_J = totalPoints := 
by
  repeat { sorry }

end trivia_team_total_points_l636_636389


namespace curve_to_polar_slope_of_line_l636_636570

-- Definition of curve C in parametric form
def curveC (θ : ℝ) : ℝ × ℝ := 
  (3 + (sqrt 5) * Real.cos θ, (sqrt 5) * Real.sin θ)

-- Conversion to polar coordinates
noncomputable def polarEquation (ρ θ : ℝ) : Prop :=
  ρ^2 - 6 * ρ * Real.cos θ + 4 = 0

theorem curve_to_polar :
  ∀ (θ : ℝ), ∃ ρ : ℝ, 
    curveC θ = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ polarEquation ρ θ := 
sorry

-- Definition of line l in parametric form
def lineL (t α : ℝ) : ℝ × ℝ := 
  (1 + t * Real.cos α, t * Real.sin α)

-- Condition for the distance |AB| = 2√3
def distanceAB (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem slope_of_line :
  ∀ (α : ℝ), ∀ (t1 t2 : ℝ), 
    lineL t1 α = (1 + t1 * Real.cos α, t1 * Real.sin α) ∧ 
    lineL t2 α = (1 + t2 * Real.cos α, t2 * Real.sin α) ∧ 
    distanceAB (lineL t1 α) (lineL t2 α) = 2 * sqrt 3 -> 
    let k := Real.tan α in k = -3 / 4 := 
sorry

end curve_to_polar_slope_of_line_l636_636570


namespace speed_increase_percentage_l636_636591

variable (T : ℚ)  -- usual travel time in minutes
variable (v : ℚ)  -- usual speed

-- Conditions
-- Ivan usually arrives at 9:00 AM, traveling for T minutes at speed v.
-- When Ivan leaves 40 minutes late and drives 1.6 times his usual speed, he arrives at 8:35 AM
def usual_arrival_time : ℚ := 9 * 60  -- 9:00 AM in minutes

def time_when_late : ℚ := (9 * 60) + 40 - (25 + 40)  -- 8:35 AM in minutes

def increased_speed := 1.6 * v -- 60% increase in speed

def time_taken_with_increased_speed := T - 65

theorem speed_increase_percentage :
  ((T / (T - 40)) = 1.3) :=
by
-- assume the equation for usual time T in terms of increased speed is known
-- Use provided conditions and solve the equation to derive the result.
  sorry

end speed_increase_percentage_l636_636591


namespace sum_of_common_ratios_eq_three_l636_636215

variable (k p r a2 a3 b2 b3 : ℝ)

-- Conditions on the sequences:
variable (h_nz_k : k ≠ 0)  -- k is nonzero as it is scaling factor
variable (h_seq1 : a2 = k * p)
variable (h_seq2 : a3 = k * p^2)
variable (h_seq3 : b2 = k * r)
variable (h_seq4 : b3 = k * r^2)
variable (h_diff_ratios : p ≠ r)

-- The given equation:
variable (h_eq : a3^2 - b3^2 = 3 * (a2^2 - b2^2))

-- The theorem statement
theorem sum_of_common_ratios_eq_three :
  p^2 + r^2 = 3 :=
by
  -- Introduce the assumptions
  sorry

end sum_of_common_ratios_eq_three_l636_636215


namespace measure_angle_BDA_l636_636396

-- Definitions of geometric structures and properties
variables {α : Type*} [euclidean_geometry α] (O A B C D : α)

-- Conditions from the problem
def conditions (O A B C D : α) :=
triangle.is_equilateral O A B ∧
square.inscribed_in_circle O A B C ∧
triangle O B D ∧
O.is_point_on_circle D ∧
(O.angle_ABC = 90) ∧
(triangle.is_equilateral A B D)

-- Theorem statement based on given conditions
theorem measure_angle_BDA (O A B C D : α) (h : conditions O A B C D) : 
  measure (angle B D A) = 60 :=
sorry

end measure_angle_BDA_l636_636396


namespace false_statements_range_l636_636059

noncomputable def number_of_false_statements (claimed: ℕ → ℕ) (actual: ℕ) : ℕ :=
by
  let total_claims := (claimed 1) + (claimed 2) + (claimed 3) + (claimed 4) + (claimed 5)
  if h : total_claims = actual then
    0
  else
    total_claims - actual

theorem false_statements_range :
  (∀ (claimed: ℕ → ℕ), 
    (claimed 1 = 1 ∧ claimed 2 = 2 ∧ claimed 3 = 3 ∧ claimed 4 = 4 ∧ claimed 5 = 5) → 
    ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 5 ∧ n = number_of_false_statements claimed 10) :=
by
  intros claimed h_claimed
  have h1 : number_of_false_statements claimed 10 = (claimed 1 + claimed 2 + claimed 3 + claimed 4 + claimed 5) - 10 :=
    by rw number_of_false_statements; simp
  rw [number_of_false_statements, dif_neg]
  simp [h_claimed.1, h_claimed.2, h_claimed.3, h_claimed.4, h_claimed.5]
  have : 5 ≤ (1 + 2 + 3 + 4 + 5) - 10 ∧ (1 + 2 + 3 + 4 + 5) - 10 ≤ 5 by norm_num
  exact ⟨(1 + 2 + 3 + 4 + 5) - 10, this⟩
  sorry

end false_statements_range_l636_636059


namespace number_of_candy_tins_l636_636708

theorem number_of_candy_tins (total_strawberry: ℕ) (strawberry_per_tin: ℕ) (tins: ℕ) :
  total_strawberry = 27 ∧ strawberry_per_tin = 3 ∧ tins = total_strawberry / strawberry_per_tin → tins = 9 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3] at h4
  exact h4

end number_of_candy_tins_l636_636708


namespace necessary_not_sufficient_condition_l636_636469

theorem necessary_not_sufficient_condition (a b : ℝ) : (a^2 + b ≥ 0) ↔ (b ≥ 0) := 
begin
  sorry
end

end necessary_not_sufficient_condition_l636_636469


namespace probability_of_selecting_cooking_l636_636953

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636953


namespace problem_statement_l636_636472

theorem problem_statement (m : ℂ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2005 = 2006 :=
  sorry

end problem_statement_l636_636472


namespace initial_mean_of_observations_l636_636293

theorem initial_mean_of_observations (M : ℚ) (h : 50 * M + 11 = 50 * 36.5) : M = 36.28 := 
by
  sorry

end initial_mean_of_observations_l636_636293


namespace probability_select_cooking_l636_636896

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636896


namespace lcm_48_180_l636_636452

theorem lcm_48_180 : Int.lcm 48 180 = 720 := by
  sorry

end lcm_48_180_l636_636452


namespace calc_present_value_l636_636024

noncomputable def presentValue (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  FV / (1 + r)^n

theorem calc_present_value :
  presentValue 750000 0.07 15 ≈ 271971.95 := 
begin
  sorry
end

end calc_present_value_l636_636024


namespace sum_of_solutions_of_absolute_value_l636_636724

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l636_636724


namespace range_of_a_l636_636092

variable (a x : ℝ)

def P : Prop := a < x ∧ x < a + 1
def q : Prop := x^2 - 7 * x + 10 ≤ 0

theorem range_of_a (h₁ : P a x → q x) (h₂ : ∃ x, q x ∧ ¬P a x) : 2 ≤ a ∧ a ≤ 4 := 
sorry

end range_of_a_l636_636092


namespace surface_area_of_circumscribed_sphere_l636_636291

-- Conditions: dimensions of the rectangular solid
def length : ℝ := 2
def width : ℝ := 2
def height : ℝ := 2 * Real.sqrt 2

-- Definition of the diagonal
def diagonal : ℝ := Real.sqrt (length^2 + width^2 + height^2)

-- Definition of the circumscribed sphere's radius
def radius : ℝ := diagonal / 2

-- Definition of the surface area of the sphere
def surface_area_of_sphere : ℝ := 4 * Real.pi * radius^2

-- Assertion to prove
theorem surface_area_of_circumscribed_sphere : surface_area_of_sphere = 16 * Real.pi :=
by
  sorry

end surface_area_of_circumscribed_sphere_l636_636291


namespace triangle_inequality_theorem_l636_636758

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality_theorem :
  ¬ is_triangle 2 3 5 ∧ is_triangle 5 6 10 ∧ ¬ is_triangle 1 1 3 ∧ ¬ is_triangle 3 4 9 :=
by {
  -- Proof goes here
  sorry
}

end triangle_inequality_theorem_l636_636758


namespace total_payment_is_53_l636_636995

-- Conditions
def bobBill : ℝ := 30
def kateBill : ℝ := 25
def bobDiscountRate : ℝ := 0.05
def kateDiscountRate : ℝ := 0.02

-- Calculations
def bobDiscount := bobBill * bobDiscountRate
def kateDiscount := kateBill * kateDiscountRate
def bobPayment := bobBill - bobDiscount
def katePayment := kateBill - kateDiscount

-- Goal
def totalPayment := bobPayment + katePayment

-- Theorem statement
theorem total_payment_is_53 : totalPayment = 53 := by
  sorry

end total_payment_is_53_l636_636995


namespace tangent_slope_at_one_l636_636536

theorem tangent_slope_at_one (c : ℝ) (h : f'(2) = 0) : 
  let f := fun x : ℝ => x^3 - 2*x^2 + c*x + c
  let f' := fun x : ℝ => 3*x^2 - 4*x + c
  f'(1) = -5 :=
sorry

end tangent_slope_at_one_l636_636536


namespace find_vector_b_l636_636209

-- Define the given vectors
def a : ℝ^3 := ![8, -5, -3]
def c : ℝ^3 := ![-1, -2, 3]

-- Define what we expect b to be
def b : ℝ^3 := ![12.5, -3.5, -6]

-- Prove that b is the vector such that a, b, and c are collinear, 
-- and b bisects the angle between a and c
theorem find_vector_b (a c : ℝ^3) (b : ℝ^3) :
  (a = ![8, -5, -3]) →
  (c = ![-1, -2, 3]) →
  (b = ![12.5, -3.5, -6]) →
  (∃ t : ℝ, b = a + t • (c - a)) ∧ 
  ((a.dot b) / (‖a‖ * ‖b‖) = (b.dot c) / (‖b‖ * ‖c‖)) :=
by
  intros ha hc hb
  have : c - a = ![-9, 3, 6], by sorry -- difference of vectors c and a
  have : b = a + (1/2) • (c - a), by sorry -- colinearity condition
  have : ((a.dot b) / (‖a‖ * ‖b‖)) = ((b.dot c) / (‖b‖ * ‖c‖)), by sorry -- angle bisector condition
  exact ⟨1/2, rfl⟩

end find_vector_b_l636_636209


namespace minimum_φ_for_monotonicity_l636_636113

-- Define the original function f(x)
def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x)

-- Define the function g(x) as a shift of f(x) by φ units to the left
def g (φ x : ℝ) : ℝ := 2 * sin (2 * (x + φ) + π / 3)

-- Define the interval of interest
def interval_of_interest (x : ℝ) : Prop := -π / 4 ≤ x ∧ x ≤ π / 6

-- Define the condition for g(x) to be monotonically increasing
def monotonic_increasing (g : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x₁ x₂, a ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ b → g x₁ ≤ g x₂

-- Prove the minimum value of φ for which g(x) is monotonically increasing in the interval
theorem minimum_φ_for_monotonicity :
  ∃ (φ : ℝ), (φ = π / 3) ∧
  ∀ x₁ x₂, interval_of_interest x₁ → interval_of_interest x₂ → x₁ ≤ x₂ → g φ x₁ ≤ g φ x₂ :=
sorry

end minimum_φ_for_monotonicity_l636_636113


namespace find_f_at_1_l636_636480

variable (f : ℝ → ℝ)
variable (a b c d e : ℝ)

-- Given conditions
def is_odd_fn (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

noncomputable def f_expr (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

noncomputable def f_prime_expr (x : ℝ) : ℝ :=
  x^2

theorem find_f_at_1 :
  is_odd_fn f_expr →
  (∀ x, deriv f_expr x = f_prime_expr x) →
  f_expr 1 = 1/3 :=
by
  sorry

end find_f_at_1_l636_636480


namespace percentage_of_kerosene_in_mixture_l636_636233

noncomputable def percentage_kerosene_in_first : ℝ :=
  25

theorem percentage_of_kerosene_in_mixture 
  (x : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ 100) 
  (mix_percentage : (6 * x + 4 * 30) / 10 = 27) : 
  x = percentage_kerosene_in_first :=
by
  have kerosene_weighted_avg_eq := mix_percentage
  have kerosene_in_first_liquid := 6 * x
  have kerosene_in_second_liquid := 4 * 30
  have total_kerosene := kerosene_in_first_liquid + kerosene_in_second_liquid
  have total_parts := 10
  have kerosene_percentage := total_kerosene / total_parts
  rw [total_parts] at kerosene_weighted_avg_eq
  sorry

end percentage_of_kerosene_in_mixture_l636_636233


namespace probability_of_selecting_cooking_l636_636808

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636808


namespace general_term_of_sequence_analytic_expression_of_function_l636_636108

noncomputable def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ := (finset.range (n+1)).sum (λ k, a k)

theorem general_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h : ∀ n : ℕ, 6 * S n = 9 * a n - 1) : ∀ n : ℕ, a n = 3^(n-2) :=
begin
  sorry
end

theorem analytic_expression_of_function
  (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π)
  (h₁ : ∀ x : ℝ, (f : ℝ → ℝ) x = A * sin (ω * x + φ) → f (x + π) = f x)
  (h₂ : f (π / 6) = a 3) : ∃ φ, f x = 3 * sin (2 * x + φ) :=
begin
  sorry
end

end general_term_of_sequence_analytic_expression_of_function_l636_636108


namespace triangle_B_angle_range_triangle_perimeter_l636_636546

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {R : ℝ}

-- Conditions
def triangle_sin_condition (A B C : ℝ) : Prop :=
  sin A + sin C = 2 * sin B

def geometric_sequence_condition (a b c : ℝ) : Prop :=
  9 * b * b = 10 * a * c

def circumcircle_radius_condition (R : ℝ) : Prop :=
  R = 3

-- Problem Statement
theorem triangle_B_angle_range 
  (h1 : triangle_sin_condition A B C)
  : B > 0 ∧ B ≤ π / 3 :=
sorry

theorem triangle_perimeter 
  (h1 : triangle_sin_condition A B C)
  (h2 : geometric_sequence_condition a b c)
  (h3 : circumcircle_radius_condition R)
  : a + b + c = 6 * sqrt 5 :=
sorry

end triangle_B_angle_range_triangle_perimeter_l636_636546


namespace probability_of_selecting_cooking_l636_636784

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636784


namespace probability_of_selecting_cooking_l636_636782

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636782


namespace exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l636_636235

open Real EuclideanGeometry

def is_isosceles_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def is_isosceles_triangle_3D (A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def five_points_isosceles (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 5, is_isosceles_triangle (pts i) (pts j) (pts k)

def six_points_isosceles (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 6, is_isosceles_triangle (pts i) (pts j) (pts k)

def seven_points_isosceles_3D (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∀ i j k : Fin 7, is_isosceles_triangle_3D (pts i) (pts j) (pts k)

theorem exists_five_points_isosceles : ∃ (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)), five_points_isosceles pts :=
sorry

theorem exists_six_points_isosceles : ∃ (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)), six_points_isosceles pts :=
sorry

theorem exists_seven_points_isosceles_3D : ∃ (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)), seven_points_isosceles_3D pts :=
sorry

end exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l636_636235


namespace arithmetic_sequence_S11_l636_636088

theorem arithmetic_sequence_S11 (a1 d : ℝ) 
  (h1 : a1 + d + a1 + 3 * d + 3 * (a1 + 6 * d) + a1 + 8 * d = 24) : 
  let a2 := a1 + d
  let a4 := a1 + 3 * d
  let a7 := a1 + 6 * d
  let a9 := a1 + 8 * d
  let S11 := 11 * (a1 + 5 * d)
  S11 = 44 :=
by
  sorry

end arithmetic_sequence_S11_l636_636088


namespace first_player_winning_strategy_l636_636386

-- Defining the type for the positions on the chessboard
structure Position where
  x : Nat
  y : Nat
  deriving DecidableEq

-- Initial position C1
def C1 : Position := ⟨3, 1⟩

-- Winning position H8
def H8 : Position := ⟨8, 8⟩

-- Function to check if a position is a winning position
-- the target winning position is H8
def isWinningPosition (p : Position) : Bool :=
  p = H8

-- Function to determine the next possible positions
-- from the current position based on the allowed moves
def nextPositions (p : Position) : List Position :=
  (List.range (8 - p.x)).map (λ dx => ⟨p.x + dx + 1, p.y⟩) ++
  (List.range (8 - p.y)).map (λ dy => ⟨p.x, p.y + dy + 1⟩) ++
  (List.range (min (8 - p.x) (8 - p.y))).map (λ d => ⟨p.x + d + 1, p.y + d + 1⟩)

-- Statement of the problem: First player has a winning strategy from C1
theorem first_player_winning_strategy : 
  ∃ move : Position, move ∈ nextPositions C1 ∧
  ∀ next_move : Position, next_move ∈ nextPositions move → isWinningPosition next_move :=
sorry

end first_player_winning_strategy_l636_636386


namespace weight_of_packet_a_l636_636765

theorem weight_of_packet_a
  (A B C D E F : ℝ)
  (h1 : (A + B + C) / 3 = 84)
  (h2 : (A + B + C + D) / 4 = 80)
  (h3 : E = D + 3)
  (h4 : (B + C + D + E) / 4 = 79)
  (h5 : F = (A + E) / 2)
  (h6 : (B + C + D + E + F) / 5 = 81) :
  A = 75 :=
by sorry

end weight_of_packet_a_l636_636765


namespace members_in_third_shift_l636_636035

-- Defining the given conditions
def total_first_shift : ℕ := 60
def percent_first_shift_pension : ℝ := 0.20

def total_second_shift : ℕ := 50
def percent_second_shift_pension : ℝ := 0.40

variable (T : ℕ)
def percent_third_shift_pension : ℝ := 0.10

def percent_total_pension_program : ℝ := 0.24

noncomputable def number_of_members_third_shift : ℕ :=
  T

-- Using the conditions to declare the theorem
theorem members_in_third_shift :
  ((60 * 0.20) + (50 * 0.40) + (number_of_members_third_shift T * percent_third_shift_pension)) / (60 + 50 + number_of_members_third_shift T) = percent_total_pension_program →
  number_of_members_third_shift T = 40 :=
sorry

end members_in_third_shift_l636_636035


namespace probability_of_selecting_cooking_l636_636951

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636951


namespace area_of_triangle_ABF_l636_636322

-- Definitions from the problem statement
structure Point := (x : ℝ) (y : ℝ)

structure Rectangle :=
  (A B C D : Point)
  (length_AB : ℝ)
  (width_AD : ℝ)
  (length_condition : A.x = 0 ∧ A.y = 0 ∧ B.x = length_AB ∧ B.y = 0)
  (width_condition : D.x = 0 ∧ D.y = width_AD ∧ C.x = length_AB ∧ C.y = width_AD)

-- Given isosceles triangle and internal point
def isosceles_triangle (A B E : Point) : Prop := 
  let AB_dist := Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2) in
  let BE_dist := Real.sqrt ((E.x - B.x)^2 + (E.y - B.y)^2) in
  AB_dist = BE_dist

-- Specific instance based on the problem statement
def problem_instance : Prop :=
  ∃ (A B C D E F : Point) (R : Rectangle),
    isosceles_triangle A B E ∧
    R.A = A ∧ R.B = B ∧ R.C = C ∧ R.D = D ∧
    let F := Point.mk ((R.width_AD * √2) / (1 + √2)) ((R.width_AD * √2) / (1 + √2)) in
    let area_ΔABF := (1 / 2) * R.length_AB * F.y in
    area_ΔABF = 1 - (1 / √2)

-- Main theorem based on the problem instance
theorem area_of_triangle_ABF :
  problem_instance :=
sorry

end area_of_triangle_ABF_l636_636322


namespace probability_of_selecting_cooking_l636_636939

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636939


namespace stones_equally_distributed_l636_636308

theorem stones_equally_distributed (n k : ℕ) 
    (h : ∃ piles : Fin n → ℕ, (∀ i j, 2 * piles i + piles j = k * n)) :
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end stones_equally_distributed_l636_636308


namespace geom_series_first_term_l636_636406

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l636_636406


namespace number_count_and_remainder_l636_636205

theorem number_count_and_remainder :
  let M := 561 in 
  (M % 500) = 61 :=
by {
  sorry
}

end number_count_and_remainder_l636_636205


namespace probability_select_cooking_l636_636895

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636895


namespace xiaoming_selects_cooking_probability_l636_636826

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636826


namespace price_of_gift_l636_636061

/-- Prove the price of the gift Lisa wants to buy -/
theorem price_of_gift : 
  let lisa_savings := 1600
  let mother_contribution := (3 / 8) * lisa_savings
  let brother_contribution := (5 / 4) * mother_contribution
  let combined_contribution := mother_contribution + brother_contribution
  let friend_contribution := (2 / 7) * combined_contribution
  let total_contributions := lisa_savings + mother_contribution + brother_contribution + friend_contribution
  let amount_short := 600
  let price_of_gift := total_contributions + amount_short
  in price_of_gift = 3935.71 := by
  sorry

end price_of_gift_l636_636061


namespace max_edge_length_of_cube_in_tetrahedron_l636_636367

theorem max_edge_length_of_cube_in_tetrahedron (edge_length_tetrahedron : ℝ) (h : edge_length_tetrahedron = 6 * Real.sqrt 2) : 
  ∃ (a : ℝ), a = 2 ∧ max_edge_length_of_cube edge_length_tetrahedron a :=
by
  sorry

-- Here you need functions like max_edge_length_of_cube which 
--  calculates the maximum edge length of the cube inside a regular tetrahedron implicitly.
--  You may handle those auxiliary definitions elsewhere in your complete formalization.

end max_edge_length_of_cube_in_tetrahedron_l636_636367


namespace inequality_solution_l636_636047

theorem inequality_solution (x : ℝ) :
  4 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 → x ∈ Set.Ioc (5 / 2 : ℝ) (20 / 7 : ℝ) := by
  sorry

end inequality_solution_l636_636047


namespace probability_of_cooking_l636_636796

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636796


namespace probability_of_cooking_l636_636800

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636800


namespace quadrilateral_area_l636_636146

theorem quadrilateral_area 
  (k : ℝ) 
  (h_perpendicular : 2 * 1 + k * 1 = 0) 
  (h_cyclic : true) : 
  (area_of_quadrilateral 
    ((2 : ℚ), (4 : ℚ)) -- Intersections of 2x + y - 4 = 0 with axes
    ((3 : ℚ), (-3 / 2 : ℚ)) -- Intersections of x - 2y - 3 = 0 with axes
   ) = 41 / 20 := 
by
  have k_value: k = -2 := by linarith [h_perpendicular]
  sorry

noncomputable def area_of_quadrilateral 
  (p1 p2 p3 p4 : ℚ × ℚ) : ℚ :=
  let area_triangle (a b c : ℚ × ℚ) : ℚ := 
    (1 / 2) * abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2))
  in
    sorry

end quadrilateral_area_l636_636146


namespace range_of_m_inequality_empty_l636_636057

theorem range_of_m_inequality_empty (m : ℝ) : 
  (∀ x : ℝ, ((m + 1) * x ^ 2 - m * x + m < 0) → false) ↔
  m ≥ real.sqrt 3 * 2 / 3 :=
by 
  sorry

end range_of_m_inequality_empty_l636_636057


namespace max_length_arith_prog_l636_636200

theorem max_length_arith_prog (p : ℕ) (k : ℕ) (hp : prime p) (hkp : p ≥ 5) (hk : k < p) :
  ∃ (n : ℕ), n = p - 1 := 
sorry

end max_length_arith_prog_l636_636200


namespace problem_statement_l636_636599

def point :=
  ℝ × ℝ

def vector :=
  point

structure hexagon :=
  (A B C D E F : point)
  (side_length : ℝ)
  (is_regular : ∀ (X Y : point), X ≠ Y → dist X Y = side_length)

def vector_sub (p q : point) : vector := (p.1 - q.1, p.2 - q.2)

def vector_scale (c : ℝ) (v : vector) : vector := (c * v.1, c * v.2)

def vector_add (u v : vector) : vector := (u.1 + v.1, u.2 + v.2)

def intersection (p1 p2 p3 p4 : point) : point :=
  -- assume a function that calculates intersection of lines p1p2 and p3p4
  sorry

def K (L F A B : point) : point :=
  let FA := vector_sub A F in
  let FB := vector_sub B F in
  vector_add (vector_scale 3 FA) (vector_scale (-1) FB)

theorem problem_statement 
  (H : hexagon)
  (L : point) (hL : L = intersection H.C H.E H.D H.F)
  (K : point) (hK : K = vector_add (vector_scale 3 (vector_sub H.A H.F)) (vector_scale (-1) (vector_sub H.B H.F))) :
  (K lies outside the hexagon H) ∧ dist K H.A = 4 * real.sqrt 3 / 3 :=
sorry

end problem_statement_l636_636599


namespace length_of_platform_l636_636387

-- Given conditions
def train_length : ℝ := 100
def time_pole : ℝ := 15
def time_platform : ℝ := 40

-- Theorem to prove the length of the platform
theorem length_of_platform (L : ℝ) 
    (h_train_length : train_length = 100)
    (h_time_pole : time_pole = 15)
    (h_time_platform : time_platform = 40)
    (h_speed : (train_length / time_pole) = (100 + L) / time_platform) : 
    L = 500 / 3 :=
by
  sorry

end length_of_platform_l636_636387


namespace sum_of_solutions_of_absolute_value_l636_636726

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l636_636726


namespace sequence_monotonic_decreasing_l636_636478

theorem sequence_monotonic_decreasing (t : ℝ) :
  (∀ n : ℕ, n > 0 → (- (n + 1) ^ 2 + t * (n + 1)) - (- n ^ 2 + t * n) < 0) ↔ (t < 3) :=
by 
  sorry

end sequence_monotonic_decreasing_l636_636478


namespace ball_first_reach_max_height_less_than_2_l636_636359

theorem ball_first_reach_max_height_less_than_2 (initial_height : ℝ) (bounce_ratio : ℝ) (target_height : ℝ) : 
  initial_height = 500 → bounce_ratio = 1/3 → target_height = 2 → 
  ∃ k : ℕ, (500 * (1/3 : ℝ)^k < 2) ∧ ∀ m : ℕ, m < k → ¬(500 * (1/3 : ℝ)^m < 2) := 
by
  intros h₁ h₂ h₃
  use 6
  split
  {
    -- prove 500 * (1/3)^6 < 2
    sorry
  }
  {
    -- prove for all m < 6, not (500 * (1/3)^m < 2)
    sorry
  }

end ball_first_reach_max_height_less_than_2_l636_636359


namespace find_a_l636_636516

theorem find_a
  (a : ℝ)
  (h_perpendicular : ∀ x y : ℝ, ax + 2 * y - 1 = 0 → 3 * x - 6 * y - 1 = 0 → true) :
  a = 4 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end find_a_l636_636516


namespace derivative_f_l636_636049

noncomputable def f (x : ℝ) : ℝ :=
  (1 / (18 * Real.sqrt 2)) * Real.log ((1 + Real.sqrt 2 * Real.coth x) / (1 - Real.sqrt 2 * Real.coth x))

theorem derivative_f (x : ℝ) :
  deriv f x = 1 / (9 * (1 + Real.cosh x ^ 2)) := sorry

end derivative_f_l636_636049


namespace calculate_expression_l636_636022

variables (q : ℝ)

theorem calculate_expression (hq : 0 ≤ q) : 2 * real.sqrt (20 * q) * real.sqrt (10 * q) * real.sqrt (15 * q) = 60 * q * real.sqrt (30 * q) :=
sorry

end calculate_expression_l636_636022


namespace probability_valid_sequence_l636_636464

-- Definitions: Set of digits, condition on the digits.
def digits := {1, 2, 3, 4, 5}

def is_valid_sequence (a b c d e : ℕ) : Prop :=
  a < b ∧ b > c ∧ c < d ∧ d > e

-- The main theorem statement
theorem probability_valid_sequence : 
  (∃ (a b c d e : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
  is_valid_sequence a b c d e) / 
  (5! : ℕ) = (2 / 15) :=
sorry

end probability_valid_sequence_l636_636464


namespace relationship_between_x_and_y_l636_636137

theorem relationship_between_x_and_y (x y : ℝ) (h1 : x - y > x) (h2 : 3 * x + 2 * y < 2 * y) : x < 0 ∧ y < 0 := 
by skip_proof sorry

end relationship_between_x_and_y_l636_636137


namespace find_m_l636_636471

def l1 (m x y: ℝ) : Prop := 2 * x + m * y - 2 = 0
def l2 (m x y: ℝ) : Prop := m * x + 2 * y - 1 = 0
def perpendicular (m : ℝ) : Prop :=
  let slope_l1 := -2 / m
  let slope_l2 := -m / 2
  slope_l1 * slope_l2 = -1

theorem find_m (m : ℝ) (h : perpendicular m) : m = 2 :=
sorry

end find_m_l636_636471


namespace total_payment_is_53_l636_636994

-- Conditions
def bobBill : ℝ := 30
def kateBill : ℝ := 25
def bobDiscountRate : ℝ := 0.05
def kateDiscountRate : ℝ := 0.02

-- Calculations
def bobDiscount := bobBill * bobDiscountRate
def kateDiscount := kateBill * kateDiscountRate
def bobPayment := bobBill - bobDiscount
def katePayment := kateBill - kateDiscount

-- Goal
def totalPayment := bobPayment + katePayment

-- Theorem statement
theorem total_payment_is_53 : totalPayment = 53 := by
  sorry

end total_payment_is_53_l636_636994


namespace remainder_a6_mod_n_eq_1_l636_636613

theorem remainder_a6_mod_n_eq_1 
  (n : ℕ) (a : ℤ) (h₁ : n > 0) (h₂ : a^3 ≡ 1 [MOD n]) : a^6 ≡ 1 [MOD n] := 
by 
  sorry

end remainder_a6_mod_n_eq_1_l636_636613


namespace find_other_root_l636_636533

theorem find_other_root (k : ℝ) (h : 1 ^ 2 + k * 1 - 2 = 0) : ∃ x : ℝ, x ≠ 1 ∧ x ^ 2 + k * x - 2 = 0 :=
by
  use -2
  split
  · sorry
  · sorry

end find_other_root_l636_636533


namespace solve_quadratic_inequality_l636_636657

noncomputable def quadratic_inequality_solution (a : ℝ) : Set ℝ :=
if h1 : a = 0 then Iio 2
else if h2 : 0 < a ∧ a < 3/2 then Iio 2 ∪ Ioi (3/a) 
else if h3 : a = 3/2 then (Iio 2 ∪ Ioi 2)
else if h4 : a > 3/2 then Iio (3/a) ∪ Ioi 2
else Ioo (3/a) 2

theorem solve_quadratic_inequality (a : ℝ) :
  ∀ x: ℝ, ax^2 - (2a + 3)x + 6 > 0 → x ∈ quadratic_inequality_solution a := by
  sorry

end solve_quadratic_inequality_l636_636657


namespace xiaoming_selects_cooking_probability_l636_636832

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636832


namespace probability_selecting_cooking_l636_636909

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636909


namespace areas_equal_iff_O_on_AC_l636_636240

variable (A B C D K L M N O : Point)

-- Definitions of points K, L, M, and N lying on the sides of parallelogram ABCD
def K_on_AB : Prop := lies_on_segment K A B
def L_on_BC : Prop := lies_on_segment L B C
def M_on_CD : Prop := lies_on_segment M C D
def N_on_DA : Prop := lies_on_segment N D A

-- Definitions of segments KM and LN being parallel to sides of the parallelogram
def KM_parallel_to_AB : Prop := segment_parallel KM AB
def LN_parallel_to_AD : Prop := segment_parallel LN AD

-- Definition of intersection at point O
def KM_intersects_LN_at_O : Prop := intersects_at_point KM LN O

-- Definition of the area of parallelogram
def area_of_parallelogram (p q r s : Point) : ℝ := 
  -- Assuming a predefined function to calculate the area of a parallelogram based on four points
  parallelogram_area p q r s

-- Problem statement
theorem areas_equal_iff_O_on_AC (K_on_AB : K_on_AB) (L_on_BC : L_on_BC) (M_on_CD : M_on_CD) 
                                (N_on_DA : N_on_DA) (KM_parallel_to_AB : KM_parallel_to_AB)
                                (LN_parallel_to_AD : LN_parallel_to_AD) 
                                (KM_intersects_LN_at_O : KM_intersects_LN_at_O) :
  area_of_parallelogram K B L O = area_of_parallelogram M D N O ↔ lies_on_segment O A C :=
sorry

end areas_equal_iff_O_on_AC_l636_636240


namespace probability_cooking_selected_l636_636923

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636923


namespace amount_raised_by_ten_students_l636_636709

theorem amount_raised_by_ten_students
  (total_students : ℕ) (special_students : ℕ) (remaining_amount_per_student : ℕ)
  (total_amount_raised : ℕ) (X : ℕ)
  (h1 : total_students = 30)
  (h2 : special_students = 10)
  (h3 : remaining_amount_per_student = 30)
  (h4 : total_amount_raised = 800)
  (h5 : (total_students - special_students) * remaining_amount_per_student = 600)
  : 10 * X + 600 = total_amount_raised → X = 20 := by
  intro h
  have h_eq : 10 * X + 600 = 800 := h.trans h4.symm
  have h6 : 10 * X = 200 := by linarith
  have h7 : X = 20 := by linarith
  exact h7

end amount_raised_by_ten_students_l636_636709


namespace true_proposition_l636_636091

variable {A B : ℝ} -- Angles A and B
variable {x y : ℝ} -- Real numbers x and y

noncomputable def is_obtuse_triangle (A B : ℝ) : Prop :=
  π > A + B ∧ A + B > π / 2

def P (A B : ℝ) : Prop :=
  is_obtuse_triangle A B → ¬ (sin A < cos B)

def q (x y : ℝ) : Prop :=
  (x + y ≠ 2) → (x ≠ -1 ∨ y ≠ 3)

theorem true_proposition :
  (∃ A B, is_obtuse_triangle A B ∧ ¬ (sin A < cos B)) ∧ (∀ x y, q x y) →
  (¬ (∃ A B, is_obtuse_triangle A B ∧ sin A < cos B) ∧ ∀ x y, q x y) := 
by
  -- Placeholder for the proof
  sorry

end true_proposition_l636_636091


namespace trapezoid_area_correct_l636_636960

noncomputable def trapezoid_area:
  Π (long_base : ℝ) (base_angle : ℝ), 
  (long_base = 24) → 
  (base_angle = real.arcsin 0.6) → 
  ℝ :=
λ long_base base_angle h_long_base h_base_angle,
  if h : (long_base = 24 ∧ base_angle = real.arcsin 0.6) then 84 else 0

theorem trapezoid_area_correct:
  trapezoid_area 24 (real.arcsin 0.6) (rfl) (rfl) = 84 := 
by
  trivial

end trapezoid_area_correct_l636_636960


namespace part_a_part_b_l636_636272

open Classical

universes u

variable {α : Type u}

structure Circle (α) :=
(center : α)
(radius : ℝ)

structure Line (α) :=
(points : set α)

structure Point (α) :=
(coord : α)

def tangent_at (p : Point α) (c1 c2 : Circle α) : Prop := sorry

def intersects (l : Line α) (c : Circle α) (p : Point α) : Prop := sorry

def is_diameter (l : Line α) (c : Circle α) : Prop := sorry

def concyclic (p1 p2 p3 p4 : Point α) : Prop := sorry

def collinear (p1 p2 p3 p4 : Point α) : Prop := sorry

theorem part_a (A C1 C2 D1 D2 : Point α) (c1 c2 : Circle α) (l : Line α) :
  tangent_at A c1 c2 →
  intersects l c1 C1 →
  intersects l c2 C2 →
  intersects (Line.mk {x | x = D1 ∨ x = D2}) c1 D1 →
  intersects (Line.mk {x | x = D1 ∨ x = D2}) c2 D2 →
  (concyclic C1 C2 D1 D2 ∨ collinear C1 C2 D1 D2) := sorry

theorem part_b (A B1 B2 D1 D2 C1 C2 : Point α) (c1 c2 : Circle α) (l : Line α) :
  tangent_at A c1 c2 →
  intersects l c1 C1 →
  intersects l c2 C2 →
  intersects (Line.mk {x | x = D1 ∨ x = D2}) c1 D1 →
  intersects (Line.mk {x | x = D1 ∨ x = D2}) c2 D2 →
  is_diameter (Line.mk {x | x = A ∨ x = C1}) c1 →
  is_diameter (Line.mk {x | x = A ∨ x = C2}) c2 →
  concyclic B1 B2 D1 D2 ↔ (is_diameter (Line.mk {x | x = A ∨ x = C1}) c1 ∧ is_diameter (Line.mk {x | x = A ∨ x = C2}) c2) := sorry

end part_a_part_b_l636_636272


namespace cade_initial_marbles_l636_636998

theorem cade_initial_marbles (marbles_from_dylan : ℕ) (total_marbles_now : ℕ) : 
  marbles_from_dylan = 8 → total_marbles_now = 95 → total_marbles_now - marbles_from_dylan = 87 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num
  sorry

end cade_initial_marbles_l636_636998


namespace part_one_part_two_l636_636558

variables {A B C a b c : ℝ} -- Angles and sides of the triangle
variables {S : ℝ} -- Area of the triangle
variables {sin_A : ℝ} -- sin A

-- Conditions
axiom acute_triangle (A B C : ℝ) (triangle : ℝ) : A < π / 2 ∧ B < π / 2 ∧ C < π / 2
axiom side_relations (a b c : ℝ) : a > 0 ∧ b > 0 ∧ c > 0
axiom area_relation (a b c S : ℝ) : 2 * S = a^2 - (b - c)^2

-- Prove for part (1)
theorem part_one (a b c S : ℝ) (sin_A : ℝ) (h_area : 2 * S = a^2 - (b - c)^2) : 
  sin_A = 4 / 5 := sorry

-- Prove for part (2)
theorem part_two (b c : ℝ) (h_range : ∀ A B C : ℝ, A + B + C = π ∧ acute_triangle A B C) :
  2 ≤ (b^2 + c^2) / (b * c) ∧ (b^2 + c^2) / (b * c) < 34 / 15 := sorry

end part_one_part_two_l636_636558


namespace sufficient_parallel_l636_636515

variables {Point Line Plane : Type}
variable [Geometry Line Plane] 

variables α β : Plane
variables a b : Line 

variables (parallel : Plane → Plane → Prop) (incidence : Line → Plane → Prop)
variables (paralLine : Line → Line → Prop) (perpLinePlane : Line → Plane → Prop)

/-- Given two parallel planes α and β, and two non-coincident lines a and b, 
prove that if a ⊥ α and b ⊥ β then a ∥ b. -/
theorem sufficient_parallel (parallelPlanes : parallel α β)
  (line_a_perp_α : perpLinePlane a α) (line_b_perp_β : perpLinePlane b β)
  (lines_non_coincident : a ≠ b) :
  paralLine a b := 
sorry

end sufficient_parallel_l636_636515


namespace find_A_B_l636_636044

theorem find_A_B :
  ∃ (A B : ℝ), (∀ x x ≠ 5 ∧ x ≠ 6, (5 * x - 8) / (x^2 - 11 * x + 30) = A / (x - 5) + B / (x - 6)) :=
begin
  use [-17, 22],
  sorry
end

end find_A_B_l636_636044


namespace speed_increase_needed_l636_636588

-- Definitions based on the conditions
def usual_speed := ℝ
def usual_travel_time := ℝ -- in minutes
def late_departure := 40   -- in minutes
def increased_speed_factor := 1.6
def early_arrival := 9*60 - (8*60 + 35) -- 25 minutes (from 9:00 AM to 8:35 AM)

-- The problem statement in Lean 4
theorem speed_increase_needed (v : usual_speed) (T : usual_travel_time) :
  let T_late := T + late_departure in
  let T_increased_speed := (T / increased_speed_factor) in
  T = usual_travel_time →
  v = usual_speed →
  T_late - T_increased_speed = late_departure + early_arrival → 
  (T - late_departure) / T = 1 / (1 - 40 / (T * (1 / (1.6)))) →
  (v * (3 / 4)) / v = 1.3 := 
sorry

end speed_increase_needed_l636_636588


namespace probability_of_cooking_l636_636795

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636795


namespace xiaoming_selects_cooking_probability_l636_636825

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636825


namespace bryden_receives_amount_l636_636962

variable (q : ℝ) (p : ℝ) (num_quarters : ℝ)

-- Define the conditions
def face_value_of_quarter : Prop := q = 0.25
def percentage_offer : Prop := p = 25 * q
def number_of_quarters : Prop := num_quarters = 5

-- Define the theorem to be proved
theorem bryden_receives_amount (h1 : face_value_of_quarter q) (h2 : percentage_offer q p) (h3 : number_of_quarters num_quarters) :
  (p * num_quarters * q) = 31.25 :=
by
  sorry

end bryden_receives_amount_l636_636962


namespace train_length_approx_l636_636776

noncomputable def length_of_train (t : ℝ) (v_m : ℝ) (v_t : ℝ) : ℝ :=
  let relative_speed := (v_t - v_m) * (5 / 18) in
  relative_speed * t

theorem train_length_approx
  (t : ℝ := 17.998560115190784)
  (v_m : ℝ := 3)
  (v_t : ℝ := 63)
  : (length_of_train t v_m v_t) ≈ 299.98 := 
by
  sorry

end train_length_approx_l636_636776


namespace dragon_boat_festival_problem_l636_636655

theorem dragon_boat_festival_problem :
  ∃ (x : ℝ), 
    let cp := 22 in
    let sp := 38 in
    let q := 160 in
    let dp := 3640 in
    let profit := (sp - x - cp) in
    let new_q := q + (x / 3) * 120 in
    profit * new_q = dp ∧ 
    sp - x = 29 :=
begin
  sorry
end

end dragon_boat_festival_problem_l636_636655


namespace p_over_q_at_neg1_l636_636679

-- Definitions of p(x) and q(x) based on given conditions
noncomputable def q (x : ℝ) := (x + 3) * (x - 2)
noncomputable def p (x : ℝ) := 2 * x

-- Define the main function y = p(x) / q(x)
noncomputable def y (x : ℝ) := p x / q x

-- Statement to prove the value of p(-1) / q(-1)
theorem p_over_q_at_neg1 : y (-1) = (1 : ℝ) / 3 :=
by
  sorry

end p_over_q_at_neg1_l636_636679


namespace fred_remaining_cards_l636_636462

variable (original_cards : ℕ)
variable (bought_cards : ℕ)

theorem fred_remaining_cards (h1 : original_cards = 40) (h2 : bought_cards = 22) :
  original_cards - bought_cards = 18 := by
    have h := h1.symm
    have k := h2.symm
    rw [h, k]
    exact Nat.sub_eq_of_eq_add (by norm_num)

end fred_remaining_cards_l636_636462


namespace find_original_triangle_area_l636_636276

-- Define the conditions and question
def original_triangle_area (A : ℝ) : Prop :=
  let new_area := 4 * A in
  new_area = 32

-- State the problem to prove the area of the original triangle
theorem find_original_triangle_area (A : ℝ) : original_triangle_area A → A = 8 := by
  intro h
  sorry

end find_original_triangle_area_l636_636276


namespace probability_cooking_is_one_fourth_l636_636865
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636865


namespace area_of_triangle_PQR_l636_636374

noncomputable theory
open Real

structure Point :=
(x : ℝ)
(y : ℝ)

def line_eq (P : Point) (m : ℝ) : (ℝ → ℝ) :=
  λ x, m * (x - P.x) + P.y

def x_intercept (f : ℝ → ℝ) : Point :=
  ⟨-(f 0) / (f 1 - f 0), 0⟩

def area_triangle (P Q R : Point) : ℝ :=
  1/2 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

noncomputable def P := ⟨2, 5⟩
noncomputable def Q := x_intercept (line_eq P 3)
noncomputable def R := x_intercept (line_eq P (-1))

theorem area_of_triangle_PQR :
  area_triangle P Q R = 50 / 3 :=
sorry

end area_of_triangle_PQR_l636_636374


namespace Victor_more_scoops_l636_636339

def ground_almonds : ℝ := 1.56
def white_sugar : ℝ := 0.75

theorem Victor_more_scoops :
  ground_almonds - white_sugar = 0.81 :=
by
  sorry

end Victor_more_scoops_l636_636339


namespace probability_selecting_cooking_l636_636897

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636897


namespace original_population_l636_636001

theorem original_population (n : ℕ) (h : 0.85 * (n + 1500) + 45 = n) : n = 8800 :=
sorry

end original_population_l636_636001


namespace rook_placement_non_attacking_l636_636562

theorem rook_placement_non_attacking (n : ℕ) (w b : ℕ) : 
  w = 8 * 8 ∧ b = (8 * (8 - 1) + (8 - 1) * (8 - 1)) → 
  w * b = 3136 :=
by 
  intro h.
  cases h with hw hb.
  sorry

end rook_placement_non_attacking_l636_636562


namespace correct_equation_l636_636577

variables (x y : ℕ)

-- Conditions
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Theorem to prove
theorem correct_equation (h1 : condition1 x y) (h2 : condition2 x y) : (y + 3) / 8 = (y - 4) / 7 :=
sorry

end correct_equation_l636_636577


namespace first_term_of_geometric_series_l636_636413

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l636_636413


namespace probability_cooking_selected_l636_636922

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636922


namespace calculate_probability_l636_636358

-- Definitions
def total_coins : ℕ := 16  -- Total coins (3 pennies + 5 nickels + 8 dimes)
def draw_coins : ℕ := 8    -- Coins drawn
def successful_outcomes : ℕ := 321  -- Number of successful outcomes
def total_outcomes : ℕ := Nat.choose total_coins draw_coins  -- Total number of ways to choose draw_coins from total_coins

-- Question statement in Lean 4: Probability of drawing coins worth at least 75 cents
theorem calculate_probability : (successful_outcomes : ℝ) / (total_outcomes : ℝ) = 321 / 12870 := by
  sorry

end calculate_probability_l636_636358


namespace complement_intersection_l636_636490

-- Definitions
def A : Set ℝ := { x | x^2 + x - 6 < 0 }
def B : Set ℝ := { x | x > 1 }

-- Stating the problem
theorem complement_intersection (x : ℝ) : x ∈ (Aᶜ ∩ B) ↔ x ∈ Set.Ici 2 :=
by sorry

end complement_intersection_l636_636490


namespace no_cyclic_poly_vals_l636_636242

theorem no_cyclic_poly_vals (p : ℤ[X])
  (x : ℤ → ℤ) (n : ℕ) (h : n ≥ 3)
  (h_distinct : ∀ i j, i ≠ j → x i ≠ x j)
  (h_poly_cycle : ∀ i, p.eval (x i) = x ((i + 1) % n)) :
  false :=
sorry

end no_cyclic_poly_vals_l636_636242


namespace probability_of_selecting_cooking_is_one_fourth_l636_636871

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636871


namespace tangent_line_at_one_range_of_a_and_crit_point_l636_636601

noncomputable def f (a x : ℝ) : ℝ := (1 + log a x) / (log a (x + 1))

theorem tangent_line_at_one (a : ℝ) (h : a = 2) : 
    let f (x : ℝ) := (1 + log 2 x) / (log 2 (x + 1)) in
    ∃ (m b : ℝ), (m = (1 : ℝ) / (2 * (log 2 2))) ∧ (f 1 = 1) ∧ (∀ (x y : ℝ), y = f x → y - 1 = m * (x - 1) ↔ x - 2 * log 2 y + 2 * log 2 - 1 = 0) :=
sorry

theorem range_of_a_and_crit_point (a : ℝ) (h1 : a > 1) : 
    ∃ x₀, f a x₀ = 0 ∧ x₀ > 0 ∧ x₀ + f a x₀ ≥ 3 :=
sorry

end tangent_line_at_one_range_of_a_and_crit_point_l636_636601


namespace ratio_of_areas_is_one_fourth_l636_636967

noncomputable def large_square_side_length : ℝ := 1
noncomputable def inscribed_square_side_length : ℝ := large_square_side_length / 2

def area_of_large_square : ℝ := large_square_side_length ^ 2
def area_of_inscribed_square : ℝ := inscribed_square_side_length ^ 2

theorem ratio_of_areas_is_one_fourth :
  (area_of_inscribed_square / area_of_large_square) = 1 / 4 := 
sorry

end ratio_of_areas_is_one_fourth_l636_636967


namespace line_passing_through_point_and_inclined_at_angle_l636_636671

noncomputable def equation_of_line (P : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ × ℝ :=
  let k := Real.tan θ in
  let (x₁, y₁) := P in
  (1, k, y₁ - k * x₁)

theorem line_passing_through_point_and_inclined_at_angle :
  ∀ (P : ℝ × ℝ), P = (Real.sqrt 3, -(2 * Real.sqrt 3)) →
  ∀ (θ : ℝ), θ = 135 * Real.pi / 180 →
  equation_of_line P θ = (1, 1, Real.sqrt 3) := 
by
  intros P hP θ hθ
  rw [hP, hθ]
  sorry

end line_passing_through_point_and_inclined_at_angle_l636_636671


namespace distance_A_to_origin_l636_636475

theorem distance_A_to_origin {A : ℝ × ℝ} (h1 : A.2^2 = 2 * A.1) 
  (h2 : (A.1 + 1 / 2) / |A.2| = 5 / 4) 
  (h3 : (A.1 - 1 / 2)^2 + A.2^2 > 4) : 
  sqrt (A.1^2 + A.2^2) = 2 * sqrt 2 :=
sorry

end distance_A_to_origin_l636_636475


namespace eagles_min_additional_wins_l636_636661

theorem eagles_min_additional_wins {N : ℕ} (eagles_initial_wins falcons_initial_wins : ℕ) (initial_games : ℕ)
  (total_games_won_fraction : ℚ) (required_fraction : ℚ) :
  eagles_initial_wins = 3 →
  falcons_initial_wins = 4 →
  initial_games = eagles_initial_wins + falcons_initial_wins →
  total_games_won_fraction = (3 + N) / (7 + N) →
  required_fraction = 9 / 10 →
  total_games_won_fraction = required_fraction →
  N = 33 :=
by
  sorry

end eagles_min_additional_wins_l636_636661


namespace max_books_combination_l636_636228

theorem max_books_combination 
  (m l : ℕ) 
  (h_total : m + l = 20) 
  (h_m_ge_5 : m ≥ 5) 
  (h_l_ge_5 : l ≥ 5) : 
  (∃ n : ℕ, ∀ k : ℕ, k ≤ 10 → (binomial (10 - k) 5 * binomial (10 + k) 5 ≤ binomial 10 5 * binomial 10 5)) → 
  m = 10 ∧ l = 10 := 
begin
  sorry
end

end max_books_combination_l636_636228


namespace exists_four_points_with_distances_l636_636181

-- Definitions based on conditions
def point := ℝ × ℝ -- considering points in 2D space

-- Distance function
def distance (p1 p2 : point) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Problem statement in Lean
theorem exists_four_points_with_distances :
  ∃ (p1 p2 p3 p4 : point), 
    {distance p1 p2, distance p1 p3, distance p1 p4, distance p2 p3, distance p2 p4, distance p3 p4} = {1, 2, 3, 4, 5, 6} :=
begin
  sorry
end

end exists_four_points_with_distances_l636_636181


namespace part_I_part_II_l636_636504

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
noncomputable def g (x : ℝ) := |2 * x - 1|

theorem part_I (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end part_I_part_II_l636_636504


namespace sum_z1_z2_diff_z1_z2_prod_z1_z2_l636_636481

noncomputable def z1 := (2 : ℂ) + (3 * complex.I)
noncomputable def z2 := (5 : ℂ) - (7 * complex.I)

theorem sum_z1_z2 : z1 + z2 = 7 - 4 * complex.I := by
  sorry

theorem diff_z1_z2 : z1 - z2 = -3 + 10 * complex.I := by
  sorry

theorem prod_z1_z2 : z1 * z2 = 31 + complex.I := by
  sorry

end sum_z1_z2_diff_z1_z2_prod_z1_z2_l636_636481


namespace john_sleep_total_hours_l636_636197

-- Defining the conditions provided in the problem statement
def days_with_3_hours : ℕ := 2
def sleep_per_day_3_hours : ℕ := 3
def remaining_days : ℕ := 7 - days_with_3_hours
def recommended_sleep : ℕ := 8
def percentage_sleep : ℝ := 0.6

-- Expressing the proof problem statement
theorem john_sleep_total_hours :
  (days_with_3_hours * sleep_per_day_3_hours
  + remaining_days * (percentage_sleep * recommended_sleep)) = 30 := by
  sorry

end john_sleep_total_hours_l636_636197


namespace det_transformation_matrix_l636_636600

open Matrix

def dilation_matrix := !![3, 0; 0, 3]

def rotation_matrix_45 := 
  let a := Real.sqrt 2 / 2
  !![a, -a; a, a]

def transformation_matrix := rotation_matrix_45 ⬝ dilation_matrix

theorem det_transformation_matrix :
  det transformation_matrix = 9 := 
sorry

end det_transformation_matrix_l636_636600


namespace partial_fraction_sum_zero_l636_636021

theorem partial_fraction_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 0 :=
by
  sorry

end partial_fraction_sum_zero_l636_636021


namespace brianna_books_l636_636017

theorem brianna_books :
  ∀ (books_per_month : ℕ) (given_books : ℕ) (bought_books : ℕ) (borrowed_books : ℕ) (total_books_needed : ℕ),
    (books_per_month = 2) →
    (given_books = 6) →
    (bought_books = 8) →
    (borrowed_books = bought_books - 2) →
    (total_books_needed = 12 * books_per_month) →
    (total_books_needed - (given_books + bought_books + borrowed_books)) = 4 :=
by
  intros
  sorry

end brianna_books_l636_636017


namespace num_of_new_boarders_l636_636345

def initial_boarders : ℕ := 60
def initial_ratio_boarders_to_daystudents : ℕ × ℕ := (2, 5)
def new_ratio_boarders_to_daystudents : ℕ × ℕ := (1, 2)

theorem num_of_new_boarders : 
  ∃ x : ℕ, let day_students := initial_boarders * initial_ratio_boarders_to_daystudents.2 / initial_ratio_boarders_to_daystudents.1 in
           2 * (initial_boarders + x) = new_ratio_boarders_to_daystudents.2 * day_students ∧
           x = 15
:=
  by 
  let day_students := initial_boarders * initial_ratio_boarders_to_daystudents.2 / initial_ratio_boarders_to_daystudents.1
  have h : 2 * (initial_boarders + 15) = new_ratio_boarders_to_daystudents.2 * day_students := sorry
  existsi 15
  split
  . exact h
  . rfl

end num_of_new_boarders_l636_636345


namespace median_length_is_correct_l636_636141

-- Define the variables and given conditions.
def area := 8
def height_on_hypotenuse := 2

-- Define the length of the hypotenuse calculated from the given conditions.
def hypotenuse_length : ℝ := by 
  -- Solving for the hypotenuse length using the area and the height on the hypotenuse.
  let x := (2 * area) / height_on_hypotenuse
  exact x

-- Define the length of the median on the hypotenuse.
def median_on_hypotenuse := hypotenuse_length / 2

-- State the theorem to prove.
theorem median_length_is_correct :
  median_on_hypotenuse = 4 :=
by
  -- skip the proof for now
  sorry

end median_length_is_correct_l636_636141


namespace shirt_cost_l636_636344

def cost_of_jeans_and_shirts (J S : ℝ) : Prop := (3 * J + 2 * S = 69) ∧ (2 * J + 3 * S = 81)

theorem shirt_cost (J S : ℝ) (h : cost_of_jeans_and_shirts J S) : S = 21 :=
by {
  sorry
}

end shirt_cost_l636_636344


namespace original_triangle_area_l636_636277

theorem original_triangle_area (new_area : ℝ) (scaling_factor : ℝ) (area_ratio : ℝ) : 
  new_area = 32 → scaling_factor = 2 → 
  area_ratio = scaling_factor ^ 2 → 
  new_area / area_ratio = 8 := 
by
  intros
  -- insert your proof logic here
  sorry

end original_triangle_area_l636_636277


namespace sequence_general_term_and_sum_l636_636087

theorem sequence_general_term_and_sum (a_n : ℕ → ℕ) (b_n S_n : ℕ → ℕ) :
  (∀ n, a_n n = 2 ^ n) ∧ (∀ n, b_n n = a_n n * (Real.logb 2 (a_n n)) ∧
  S_n n = (n - 1) * 2 ^ (n + 1) + 2) :=
by
  sorry

end sequence_general_term_and_sum_l636_636087


namespace ln_abs_min_eq_inv_e_squared_l636_636116

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.log x + 1)

theorem ln_abs_min_eq_inv_e_squared :
  let m := f (1 / Real.exp 2)
  ln (|m|) = 1 / Real.exp 2 :=
by
  let m := f (1 / Real.exp 2)
  sorry

end ln_abs_min_eq_inv_e_squared_l636_636116


namespace projection_is_neg3_l636_636071

variable (a b : ℝ)
variable (vec_a vec_b : ℝ → ℝ)

-- Given conditions
def magnitude_a := 5
def magnitude_b := 3
def dot_product_ab := -9

-- Define projection formula
def projection (vec_a vec_b : ℝ → ℝ) : ℝ := dot_product_ab / magnitude_b

-- Theorem statement
theorem projection_is_neg3 : projection vec_a vec_b = -3 := 
by sorry

end projection_is_neg3_l636_636071


namespace correct_equation_l636_636576

variables (x y : ℕ)

-- Conditions
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Theorem to prove
theorem correct_equation (h1 : condition1 x y) (h2 : condition2 x y) : (y + 3) / 8 = (y - 4) / 7 :=
sorry

end correct_equation_l636_636576


namespace work_days_of_A_and_B_l636_636966

theorem work_days_of_A_and_B (B : ℝ) (A : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 27) :
  1 / (A + B) = 9 :=
by
  sorry

end work_days_of_A_and_B_l636_636966


namespace heartsuit_sum_l636_636437

def heartsuit (x : ℝ) : ℝ := (x + x^3) / 2

theorem heartsuit_sum : (heartsuit 1) + (heartsuit 2) + (heartsuit 4) = 40 :=
by
  sorry

end heartsuit_sum_l636_636437


namespace lune_area_equation_l636_636037

theorem lune_area_equation :
  ∃ p q r : ℕ, p + q + r = 10 ∧
    (∀ (p q r : ℕ), p = 3 ∧ q = 1 ∧ r = 6 ∧
      (sqrt (p:ℝ) - (q:ℝ) * real.pi / (r:ℝ) = sqrt 3 - real.pi / 6)) :=
by
  use [3, 1, 6]
  split
  · --Proof for p + q + r
    exact rfl
  sorry

end lune_area_equation_l636_636037


namespace speed_increase_needed_l636_636587

-- Definitions based on the conditions
def usual_speed := ℝ
def usual_travel_time := ℝ -- in minutes
def late_departure := 40   -- in minutes
def increased_speed_factor := 1.6
def early_arrival := 9*60 - (8*60 + 35) -- 25 minutes (from 9:00 AM to 8:35 AM)

-- The problem statement in Lean 4
theorem speed_increase_needed (v : usual_speed) (T : usual_travel_time) :
  let T_late := T + late_departure in
  let T_increased_speed := (T / increased_speed_factor) in
  T = usual_travel_time →
  v = usual_speed →
  T_late - T_increased_speed = late_departure + early_arrival → 
  (T - late_departure) / T = 1 / (1 - 40 / (T * (1 / (1.6)))) →
  (v * (3 / 4)) / v = 1.3 := 
sorry

end speed_increase_needed_l636_636587


namespace acute_triangle_segment_lengths_l636_636538

open Real

theorem acute_triangle_segment_lengths (x : ℝ) (h_condition : (x^2 + 6 > x^2 + 4 ∧ x^2 + 4 ≥ 4x ∧ 4x > 0))
  (h_acute : ∃θ : ℝ, cos θ = (x^2 + 4)^2 + (4x)^2 - (x^2 + 6)^2 / (2 * 4x * (x^2 + 4)) ∧ cos θ > 0) :
  x > sqrt 15 / 3 := 
  sorry

end acute_triangle_segment_lengths_l636_636538


namespace george_earnings_l636_636064

theorem george_earnings (cars_sold : ℕ) (price_per_car : ℕ) (lego_set_price : ℕ) (h1 : cars_sold = 3) (h2 : price_per_car = 5) (h3 : lego_set_price = 30) :
  cars_sold * price_per_car + lego_set_price = 45 :=
by
  sorry

end george_earnings_l636_636064


namespace movie_duration_l636_636377

theorem movie_duration :
  let start_time := (13, 30)
  let end_time := (14, 50)
  let hours := end_time.1 - start_time.1
  let minutes := end_time.2 - start_time.2
  (if minutes < 0 then (hours - 1, minutes + 60) else (hours, minutes)) = (1, 20) := by
    sorry

end movie_duration_l636_636377


namespace probability_of_selecting_cooking_l636_636788

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636788


namespace third_side_length_l636_636145

theorem third_side_length (a b : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  ∃ x : ℝ, (a = 3 ∧ b = 4) ∧ (x = 5 ∨ x = Real.sqrt 7) :=
by
  sorry

end third_side_length_l636_636145


namespace solve_inequality_system_l636_636658

theorem solve_inequality_system (x : ℝ) :
  (4 * x + 5 > x - 1) ∧ ((3 * x - 1) / 2 < x) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

end solve_inequality_system_l636_636658


namespace perpendicular_AA2_KC_l636_636584

variables {A B C K A1 A2 : Type*}

-- Given conditions
variable (triangle_ABC : Triangle A B C)
variable (is_median : IsMedian triangle_ABC A A1)
variable (is_angle_bisector : IsAngleBisector triangle_ABC A A2)
variable (K_on_median : IsOnLine K (Line_through A A1))
variable (KA2_parallel_AC : Parallel (Line_through K A2) (Line_through A C))

-- Goal
theorem perpendicular_AA2_KC 
  (triangle_ABC : Triangle A B C)
  (is_median : IsMedian triangle_ABC A A1)
  (is_angle_bisector : IsAngleBisector triangle_ABC A A2)
  (K_on_median : IsOnLine K (Line_through A A1))
  (KA2_parallel_AC : Parallel (Line_through K A2) (Line_through A C)) :
  Perpendicular (Line_through A A2) (Line_through K C) :=
  sorry

end perpendicular_AA2_KC_l636_636584


namespace find_angles_of_triangle_l636_636178

variables {A B C K L M : Point}
variables (triangle_ABC : Triangle A B C) 
variables (bisector_AK : AngleBisector A K)
variables (bisector_CL : AngleBisector C L)
variables (median_BM : Median B M)
variables (bisector_ML : AngleBisector M L)
variables (bisector_MK : AngleBisector M K)

theorem find_angles_of_triangle :
  \forall (angle_A : angle A = 30^\circ) (angle_B : angle B = 120^\circ) (angle_C : angle C = 30^\circ),
  is_triangle A B C → bisector_of A K → bisector_of C L → median_of B M → bisector_of M L → bisector_of M K → 
  (angle A = 30° ∧ angle B = 120° ∧ angle C = 30°) :=
begin
  sorry
end

end find_angles_of_triangle_l636_636178


namespace primes_between_20_and_30_expression_l636_636029

theorem primes_between_20_and_30_expression :
  ∃ (x y : ℕ), nat.prime x ∧ nat.prime y ∧ (20 < x ∧ x < 30) ∧ (20 < y ∧ y < 30) ∧ x ≠ y ∧
  x * y - (x + y) - (x ^ 2 + y ^ 2) = -755 :=
begin
  use 23, 29,
  repeat {split},
  { exact nat.prime_23 },
  { exact nat.prime_29 },
  { norm_num },
  { norm_num },
  { norm_num },
  { norm_num },
  { norm_num },
  sorry
end

end primes_between_20_and_30_expression_l636_636029


namespace total_bricks_needed_l636_636222

-- Definitions based on conditions
variables (typeA : ℕ) (typeB : ℕ) (otherTypes : ℕ)
axiom typeA_def : typeA = 40
axiom typeB_def : typeB = typeA / 2
axiom otherTypes_def : otherTypes = 90

-- Proof problem statement
theorem total_bricks_needed (typeA : ℕ) (typeB : ℕ) (otherTypes : ℕ) 
  (hA : typeA = 40) (hB : typeB = typeA / 2) (hO : otherTypes = 90) 
  : typeA + typeB + otherTypes = 150 :=
by {
  rw [hA, hB, hO],
  norm_num,
  sorry
}

end total_bricks_needed_l636_636222


namespace reimbursement_calculation_l636_636224

variable (total_paid : ℕ) (pieces : ℕ) (cost_per_piece : ℕ)

theorem reimbursement_calculation
  (h1 : total_paid = 20700)
  (h2 : pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (pieces * cost_per_piece) = 600 := 
by
  sorry

end reimbursement_calculation_l636_636224


namespace find_constant_a_l636_636537

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (Real.exp x - 1)

theorem find_constant_a (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = - f a x) : a = -1 := 
by
  sorry

end find_constant_a_l636_636537


namespace square_area_l636_636568

variables {A B C D P Q R : Type*} [euclidean_geometry E]
variables {AD AB BP CQ : set (euclidean_space E)}
variables {BR PR CQ_len : ℝ}

-- Define the square property
def is_square (A B C D : euclidean_space E) : Prop :=
(dist A B = dist B C) ∧ (dist B C = dist C D) ∧ (dist C D = dist D A)
  ∧ ∠ A B C = π / 2 ∧ ∠ B C D = π / 2 ∧ ∠ C D A = π / 2 ∧ ∠ D A B = π / 2

-- Define the points P and Q on AD and AB respectively
def lies_on (P Q : euclidean_space E) (AD AB : set (euclidean_space E)) : Prop :=
P ∈ AD ∧ Q ∈ AB

-- Define the conditions given in the problem
def problem_conditions (A B C D P Q R: euclidean_space E)
                       (BP CQ: set (euclidean_space E)) (BR PR CQ_len: ℝ) : Prop :=
  is_square A B C D ∧
  lies_on P Q AD ∧
  (BP ∩ CQ).Nonempty ∧ ∠ B P Q = π / 2 ∧ dist B R = BR ∧ dist P R = PR ∧ dist C Q = CQ_len

-- The specific values given in the problem
def specific_values (BR PR CQ_len: ℝ) : Prop :=
  BR = 8 ∧ PR = 5 ∧ CQ_len = 12

-- The final theorem to be proven
theorem square_area (A B C D P Q R : euclidean_space E)
                    (AD AB BP CQ : set (euclidean_space E)) (BR PR CQ_len: ℝ) :
  problem_conditions A B C D P Q R BP CQ BR PR CQ_len ∧ specific_values BR PR CQ_len
  → (dist A B)^2 = 169 :=
by sorry

end square_area_l636_636568


namespace total_distance_cycled_l636_636189

theorem total_distance_cycled (d_store_peter : ℕ) 
  (store_to_peter_distance : d_store_peter = 50) 
  (home_to_store_twice : ∀ t, distance_from_home_to_store * t = 2 * d_store_peter * t)
  : distance_from_home_to_store = 100 → 
    total_distance = (distance_from_home_to_store + store_to_peter_distance + d_store_peter) :=
by
  /- Let distance_from_home_to_store be denoted as d_home_store for simplicity -/
  let d_home_store := distance_from_home_to_store
  -- Given
  have home_to_store_distance : d_home_store = 2 * d_store_peter := home_to_store_twice 1
  -- We can derive
  have h1 : d_home_store = 100 := home_to_store_distance
  -- Calculate total distance
  have total_distance := d_home_store + d_store_peter + d_store_peter
  -- Conclusion
  have h2 : total_distance = 100 + 50 + 100 := rfl
  -- Final Proposition
  exact h2

end total_distance_cycled_l636_636189


namespace atleast_one_genuine_l636_636984

noncomputable def products : ℕ := 12
noncomputable def genuine : ℕ := 10
noncomputable def defective : ℕ := 2
noncomputable def selected : ℕ := 3

theorem atleast_one_genuine :
  (selected = 3) →
  (genuine + defective = 12) →
  (genuine ≥ 3) →
  (selected ≥ 1) →
  ∃ g d : ℕ, g + d = 3 ∧ g > 0 ∧ d ≤ 2 :=
by
  -- Proof will go here.
  sorry

end atleast_one_genuine_l636_636984


namespace intersection_of_M_and_N_l636_636512

noncomputable def M : Set ℝ := {x | x - 2 > 0}
noncomputable def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

theorem intersection_of_M_and_N :
  M ∩ N = {x | x > 2} :=
sorry

end intersection_of_M_and_N_l636_636512


namespace diameter_and_circumference_of_circle_l636_636959

theorem diameter_and_circumference_of_circle (A : ℝ) (hA : A = 4 * Real.pi) :
    ∃ d C : ℝ, d = 4 ∧ C = 4 * Real.pi :=
by
  -- Let r be the radius of the circle
  let r := 2 -- Since π * r^2 = 4 * π, solving for r gives r = 2
  
  -- Define the diameter and circumference
  let d := 2 * r
  let C := 2 * Real.pi * r

  -- Prove the required properties
  use [d, C]
  split
  · show d = 4
    rw [d, r]
    norm_num
  · show C = 4 * Real.pi
    rw [C, r]
    norm_num
  · done

end diameter_and_circumference_of_circle_l636_636959


namespace problem_statement_l636_636089

noncomputable def f : ℝ → ℝ
| x if (0 < x ∧ x ≤ 1)  := 2^x
| x                     := sorry -- definition for other x is determined by properties

theorem problem_statement :
  (∀ x : ℝ, f(-x) = -f(x)) → (∀ x : ℝ, f(x+2) = -f(x)) → f(2016) - f(2015) = 2 :=
by
  intro h_odd h_periodic
  have f_period : ∀ x : ℝ, f(x + 4) = f(x), from
    λ x, by rw [add_assoc, h_periodic, h_periodic (x+2), h_periodic x];
             exact congr_arg (λ y, -(-f(y))) (h_periodic x) -- Utilizing periodicity
  have : f(2016) = f(0), from congr_arg f (show 2016 % 4 = 0, by norm_num)
  have : f(2015) = f(-1), from congr_arg f (show 2015 % 4 = -1, by norm_num)
  have f_0 : f(0) = 0, from sorry -- Using that f is odd and f is 0 at 0
  have f_neg1 : f(-1) = -2, from sorry -- Using that f(1) = 2^1 = 2 and f is odd
  calc
    f(2016) - f(2015) = f(0) - f(-1) : by congr; exact_mod_cast 𝕌-h :- sorry -- Using periodic properties
                ...  = 0 - (-2)     : by rw [f_0, f_neg1]
                ...  = 2,
  sorry

end problem_statement_l636_636089


namespace probability_select_cooking_l636_636887

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636887


namespace problem1_problem2_problem3_l636_636112

noncomputable def f (x : ℝ) (a : ℝ) := (2 / x) + a * Real.log x - 2

theorem problem1 (h : 0 < Real.log 1) : 
  (f'(1) + 1 = 0) → ∃ a : ℝ, a = 1 :=
by
  sorry

noncomputable def g (x : ℝ) (b : ℝ) := (2 / x) + Real.log x + x - 2 - b

theorem problem2 (h₁ : ∞ > e⁻¹) (h₂: x > 0) :
  (∃ x : ℝ, f(x) + x - b = 0) ∧ (∃ x : ℝ, f(x) + x - b = 0) 
  → 1 < b ∧ b ≤ (2 / e + e - 1) :=
by
  sorry

noncomputable def inequality_expression (x : ℝ) :=
  x^2 - 4 * x + 2

theorem problem3 (h : ∀ t : ℝ, |t| ≤ 2 := 
  ∀ x : ℝ, inequality_expression x > 0
  → (0, 2 - Real.sqrt 2) ∪ (2 + Real.sqrt 2, ∞) := 
by
  sorry

end problem1_problem2_problem3_l636_636112


namespace probability_of_selecting_cooking_l636_636941

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636941


namespace sugar_ratio_l636_636426

theorem sugar_ratio (total_sugar : ℕ)  (bags : ℕ) (remaining_sugar : ℕ) (sugar_each_bag : ℕ) (sugar_fell : ℕ)
  (h1 : total_sugar = 24) (h2 : bags = 4) (h3 : total_sugar - remaining_sugar = sugar_fell) 
  (h4 : total_sugar / bags = sugar_each_bag) (h5 : remaining_sugar = 21) : 
  2 * sugar_fell = sugar_each_bag := by
  -- proof goes here
  sorry

end sugar_ratio_l636_636426


namespace probability_cooking_is_one_fourth_l636_636854
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636854


namespace train_length_correct_l636_636979

noncomputable def train_length (time : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time

theorem train_length_correct :
  train_length 2.49980001599872 144 = 99.9920006399488 :=
by
  dsimp [train_length]
  -- Convert 144 km/h to m/s => 40 m/s
  have speed_ms : ℝ := 144 * (1000 / 3600)
  norm_num [speed_ms]
  -- Compute distance: 40 m/s * 2.49980001599872 s => 99.9920006399488 m
  have distance : ℝ := speed_ms * 2.49980001599872
  norm_num [distance]
  -- Assert and verify the final result
  norm_num
  exact sorry

end train_length_correct_l636_636979


namespace sum_of_reciprocals_l636_636212

noncomputable def a_seq (n : ℕ) : ℝ → ℝ → ℝ
| 0, a0, b0 => a0
| (n + 1), a0, b0 => a_seq n a0 b0 + b_seq n a0 b0 + real.cos ((a_seq n a0 b0)^2 + (b_seq n a0 b0)^2)
  
noncomputable def b_seq (n : ℕ) : ℝ → ℝ → ℝ
| 0, a0, b0 => b0
| (n + 1), a0, b0 => a_seq n a0 b0 + b_seq n a0 b0 - real.cos ((a_seq n a0 b0)^2 + (b_seq n a0 b0)^2)

theorem sum_of_reciprocals :
  (a_seq 0 (-3 : ℝ) 2) = -3 → 
  (b_seq 0 (-3 : ℝ) 2) = 2 → 
  (1 / a_seq 2023 (-3 : ℝ) 2 + 1 / b_seq 2023 (-3 : ℝ) 2) = -2 :=
sorry

end sum_of_reciprocals_l636_636212


namespace xiaoming_selects_cooking_probability_l636_636823

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636823


namespace probability_of_selecting_cooking_l636_636819

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636819


namespace biology_to_general_ratio_l636_636706

variable (g b m : ℚ)

theorem biology_to_general_ratio (h1 : g = 30) 
                                (h2 : m = (3/5) * (g + b)) 
                                (h3 : g + b + m = 144) : 
                                b / g = 2 / 1 := 
by 
  sorry

end biology_to_general_ratio_l636_636706


namespace students_brought_only_one_fruit_l636_636154

theorem students_brought_only_one_fruit (a b both : ℕ) (A : a = 12) (B : b = 8) (C : both = 5) :
  (12 - 5) + (8 - 5) = 10 :=
by
  rw [A, B, C]
  norm_num
  sorry

end students_brought_only_one_fruit_l636_636154


namespace sum_of_n_values_l636_636740

theorem sum_of_n_values : sum (λ n, |3 * n - 8| = 5) = 16/3 := 
by
  sorry

end sum_of_n_values_l636_636740


namespace num_ways_to_represent_1500_l636_636563

theorem num_ways_to_represent_1500 :
  let N := 32 in
  ∃ (a b c : ℕ), 
    prime_factors a * prime_factors b * prime_factors c = prime_factors 1500 →
    a * b * c = 1500 ∧ (x1 + x2 + x3 = 2) ∧ 
    (y1 + y2 + y3 = 1) ∧ (z1 + z2 + z3 = 3) →
    N = 32 :=
begin
  sorry
end

end num_ways_to_represent_1500_l636_636563


namespace range_of_k_l636_636474

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x k : ℝ) : ℝ := sin x - cos x - k * x

theorem range_of_k {k : ℝ} (f_mono : ∀ x y : ℝ, x < y → f x < f y) 
  (f_eq : ∀ x : ℝ, f (f x - 2017^x) = 2017)
  (g_mono : ∀ x y : ℝ, x ∈ Icc (-(π / 2)) (π / 2) → y ∈ Icc (-(π / 2)) (π / 2) → x < y → g x k < g y k) :
  k ≤ -1 :=
begin
  sorry
end

end range_of_k_l636_636474


namespace peter_ate_7_over_48_l636_636234

-- Define the initial conditions
def total_slices : ℕ := 16
def slices_peter_ate : ℕ := 2
def shared_slice : ℚ := 1/3

-- Define the first part of the problem
def fraction_peter_ate_alone : ℚ := slices_peter_ate / total_slices

-- Define the fraction Peter ate from sharing one slice
def fraction_peter_ate_shared : ℚ := shared_slice / total_slices

-- Define the total fraction Peter ate
def total_fraction_peter_ate : ℚ := fraction_peter_ate_alone + fraction_peter_ate_shared

-- Create the theorem to be proved (statement only)
theorem peter_ate_7_over_48 :
  total_fraction_peter_ate = 7 / 48 :=
by
  sorry

end peter_ate_7_over_48_l636_636234


namespace sum_of_n_for_3n_minus_8_eq_5_l636_636746

theorem sum_of_n_for_3n_minus_8_eq_5 : 
  let S := {n | |3 * n - 8| = 5}
  in S.sum id = 16 / 3 :=
by
  sorry

end sum_of_n_for_3n_minus_8_eq_5_l636_636746


namespace num_arrangement_schemes_is_126_l636_636458

theorem num_arrangement_schemes_is_126 :
  ∃ (arrangements : Finset (Fin 126)),
    ∀ (A B C D E : arrangements), 
      let job_types := {translation, tour_guide, etiquette, driver},
      (∃ a b, a ≠ b ∧ a ∈ job_types \ {driver} ∧ b ∈ job_types \ {driver} ∧ A ≠ a ∧ B ≠ b ∧ 
        C ∈ job_types ∧ D ∈ job_types ∧ E ∈ job_types ∧
        {A, B, C, D, E}.card = 5) 
      → arrangements.card = 126 := sorry

end num_arrangement_schemes_is_126_l636_636458


namespace find_omega_find_area_triangle_l636_636518
open Real

-- Given conditions
def vector_a (ω x : ℝ) : ℝ × ℝ := (2 * cos (ω * x), -2)
def vector_b (ω x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (ω * x) + cos (ω * x), 1)
def f (ω x : ℝ) : ℝ := (vector_a ω x).1 * (vector_b ω x).1 + (vector_a ω x).2 * (vector_b ω x).2 + 1

-- Part (1)
theorem find_omega (h : ∀ x, f ω x = 2 * sin (2 * ω * x + π / 6)) (distance_cond : ∀ x, x ∈ {y | f ω y = 0}.succ ≠ ∅ → 2 * π / (2 * ω) = π / 2) : ω = 1 :=
  sorry

-- Part (2)
theorem find_area_triangle (h1 : f 1 = 2 * sin (2 * 1 + π / 6)) 
                          (h2: ∀ A B,
                            f 1 A = √3 ∧ f 1 B = √3 
                            → (A = π / 12 ∧ B = π / 4) ∨ (A = π / 4 ∧ B = π / 12))
                          (a : ℝ) (ha : a = sqrt 2) : 
                          ∃ b c, 
                            let area := (1/2) * a * b * sin c in
                            (area = (3 + sqrt 3) / 2) ∨ (area = (3 - sqrt 3) / 4) :=
  sorry

end find_omega_find_area_triangle_l636_636518


namespace draw_at_least_one_red_card_l636_636976

-- Define the deck and properties
def total_cards := 52
def red_cards := 26
def black_cards := 26

-- Define the calculation for drawing three cards sequentially
def total_ways_draw3 := total_cards * (total_cards - 1) * (total_cards - 2)
def black_only_ways_draw3 := black_cards * (black_cards - 1) * (black_cards - 2)

-- Define the main proof statement
theorem draw_at_least_one_red_card : 
    total_ways_draw3 - black_only_ways_draw3 = 117000 := by
    -- Proof is omitted
    sorry

end draw_at_least_one_red_card_l636_636976


namespace shortest_chord_standard_form_l636_636569

-- Definitions for the conditions
def line_parametric (t : ℝ) : ℝ × ℝ := (3 + t, 1 + a * t)
def curve_parametric (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)

-- The main theorem we want to prove
theorem shortest_chord_standard_form (a : ℝ) : (∃ (α1 α2 : ℝ), 
  let A := curve_parametric α1 in
  let B := curve_parametric α2 in
  let P := (3, 1) in
  let CP_slope := (P.2 - 0) / (P.1 - 2) in
  let AB_slope := -1 in
  (Real.dist A B = shortest_dist) ∧
  (CP_slope * AB_slope = -1) ∧
  (∀ t : ℝ, line_parametric t = (x, y))
) → (y - 1 = a * (x - 3)) ∧ (x + y - 4 = 0) :=
by
  sorry

end shortest_chord_standard_form_l636_636569


namespace probability_of_selecting_cooking_is_one_fourth_l636_636880

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636880


namespace sum_of_squares_inductive_step_l636_636720

theorem sum_of_squares_inductive_step (n k : ℕ) :
  (∑ i in Finset.range (k + 1), i^2 + ∑ i in Finset.range k, i^2)
  + (k + 1)^2 + k^2
  = ∑ i in Finset.range (k + 2), i^2 + ∑ i in Finset.range (k + 1), i^2 :=
sorry

end sum_of_squares_inductive_step_l636_636720


namespace distinct_triangle_areas_l636_636033

noncomputable def point := ℝ 
constant G H I J K L : point
constant GH HI IJ KL : ℝ 

axiom GH_val : GH = 0.5
axiom HI_val : HI = 0.5
axiom IJ_val : IJ = 1
axiom KL_val : KL = 2

theorem distinct_triangle_areas : 
  let dist := G ∈ {G,H,I,J} ∧ H ∈ {G,H,I,J} ∧ I ∈ {G,H,I,J} ∧ J ∈ {G,H,I,J} ∧ K ∈ {K,L} ∧ L ∈ {K,L} in
  (distinct_points dist GH HI IJ KL GH_val HI_val IJ_val KL_val = 4) ∧ 
    (positive_areas_distinct dist GH HI IJ KL GH_val HI_val IJ_val KL_val)
  := 
  sorry

end distinct_triangle_areas_l636_636033


namespace fourth_tree_height_l636_636434

-- Define the constants and conditions given.
def first_tree_height := 50
def first_tree_branches := 200
def second_tree_height := 40
def second_tree_branches := 180
def third_tree_height := 60
def third_tree_branches := 180
def fourth_tree_branches := 153
def average_branches_per_foot := 4

-- Define the property we want to prove.
theorem fourth_tree_height : Int.round (fourth_tree_branches / average_branches_per_foot) = 38 := by
  sorry

end fourth_tree_height_l636_636434


namespace sallys_woodworking_llc_reimbursement_l636_636226

/-
Conditions:
1. Remy paid $20,700 for 150 pieces of furniture.
2. The cost of a piece of furniture is $134.
-/
def reimbursement_amount (pieces_paid : ℕ) (total_paid : ℕ) (price_per_piece : ℕ) : ℕ :=
  total_paid - (pieces_paid * price_per_piece)

theorem sallys_woodworking_llc_reimbursement :
  reimbursement_amount 150 20700 134 = 600 :=
by 
  sorry

end sallys_woodworking_llc_reimbursement_l636_636226


namespace remainder_when_a6_divided_by_n_l636_636611

theorem remainder_when_a6_divided_by_n (n : ℕ) (a : ℤ) (h : a^3 ≡ 1 [ZMOD n]) :
  a^6 ≡ 1 [ZMOD n] := 
sorry

end remainder_when_a6_divided_by_n_l636_636611


namespace probability_cooking_is_one_fourth_l636_636862
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636862


namespace geometric_series_first_term_l636_636407

theorem geometric_series_first_term (r a s : ℝ) (h₁ : r = 1 / 4) (h₂ : s = 80) (h₃ : s = a / (1 - r)) : a = 60 :=
by
  rw [h₁] at h₃
  rw [h₂] at h₃
  norm_num at h₃
  linarith

# Examples utilized:
-- r : common ratio
-- a : first term of the series
-- s : sum of the series
-- h₁ : condition that the common ratio is 1/4
-- h₂ : condition that the sum is 80
-- h₃ : condition representing the formula for the sum of an infinite geometric series

end geometric_series_first_term_l636_636407


namespace angle_A_range_l636_636174

theorem angle_A_range (a b : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) :
  ∃ A : ℝ, 0 < A ∧ A ≤ Real.pi / 4 :=
sorry

end angle_A_range_l636_636174


namespace probability_of_selecting_cooking_l636_636956

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636956


namespace problem_solution_l636_636350

variables {p q r : ℝ}

theorem problem_solution (h1 : (p + q) * (q + r) * (r + p) / (p * q * r) = 24)
  (h2 : (p - 2 * q) * (q - 2 * r) * (r - 2 * p) / (p * q * r) = 10) :
  ∃ m n : ℕ, (m.gcd n = 1 ∧ (p/q + q/r + r/p = m/n) ∧ m + n = 39) :=
sorry

end problem_solution_l636_636350


namespace probability_of_selecting_cooking_l636_636938

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636938


namespace overall_profit_refrigerator_mobile_phone_l636_636642

theorem overall_profit_refrigerator_mobile_phone
  (purchase_price_refrigerator : ℕ)
  (purchase_price_mobile_phone : ℕ)
  (loss_percentage_refrigerator : ℕ)
  (profit_percentage_mobile_phone : ℕ)
  (selling_price_refrigerator : ℕ)
  (selling_price_mobile_phone : ℕ)
  (total_cost_price : ℕ)
  (total_selling_price : ℕ)
  (overall_profit : ℕ) :
  purchase_price_refrigerator = 15000 →
  purchase_price_mobile_phone = 8000 →
  loss_percentage_refrigerator = 4 →
  profit_percentage_mobile_phone = 10 →
  selling_price_refrigerator = purchase_price_refrigerator - (purchase_price_refrigerator * loss_percentage_refrigerator / 100) →
  selling_price_mobile_phone = purchase_price_mobile_phone + (purchase_price_mobile_phone * profit_percentage_mobile_phone / 100) →
  total_cost_price = purchase_price_refrigerator + purchase_price_mobile_phone →
  total_selling_price = selling_price_refrigerator + selling_price_mobile_phone →
  overall_profit = total_selling_price - total_cost_price →
  overall_profit = 200 :=
  by sorry

end overall_profit_refrigerator_mobile_phone_l636_636642


namespace MH_greater_than_MK_l636_636553

-- Defining the conditions: BH perpendicular to HK and BH = 2
def BH := 2

-- Defining the conditions: CK perpendicular to HK and CK = 5
def CK := 5

-- M is the midpoint of BC, which implicitly means MB = MC in length
def M_midpoint_BC (MB MC : ℝ) :=
  MB = MC

theorem MH_greater_than_MK (MB MC MH MK : ℝ) 
  (hM_midpoint : M_midpoint_BC MB MC)
  (hMH : MH^2 + BH^2 = MB^2)
  (hMK : MK^2 + CK^2 = MC^2) :
  MH > MK :=
by
  sorry

end MH_greater_than_MK_l636_636553


namespace photosynthesis_pathway_l636_636645

theorem photosynthesis_pathway (co2 C3 Sugar : Type) 
  (optA : co2 -> Chlorophyll -> ADP) 
  (optB : co2 -> Chloroplast -> ATP) 
  (optC : co2 -> LacticAcid -> Sugars) 
  (optD : co2 -> C3 -> Sugars) 
  : (∀ (c : co2), optD c = c -> C3 -> Sugars) := 
by
  sorry

end photosynthesis_pathway_l636_636645


namespace triangles_similar_l636_636598

-- Define the basic geometric setup
variables {A B C D E F P Q R : Type} 

-- Assume that A, B, C form an acute triangle, 
-- and D, E, F are the touchpoints of the incircle.
variables (ABC_acute : is_acute_triangle A B C)
          (incircle_touches : touches_incircle A B C D E F)

-- Assume P, Q, R are the circumcenters
-- of triangles AEF, BDF, and CDE respectively.
variables (P_is_circumcenter : is_circumcenter A E F P)
          (Q_is_circumcenter : is_circumcenter B D F Q)
          (R_is_circumcenter : is_circumcenter C D E R)

-- Prove that triangles ABC and PQR are similar
theorem triangles_similar : similar_triangles A B C P Q R :=
sorry

end triangles_similar_l636_636598


namespace integral_f_eq_l636_636136

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 2 * (2 - π) * x + sin (2 * x)

theorem integral_f_eq :
  (∫ x in 0..1, f x) = (17 / 6) - π - (1 / 2) * cos 2 :=
by
  sorry

end integral_f_eq_l636_636136


namespace line_equation_through_pointP_and_angle_l636_636674

-- Given conditions
def pointP : ℝ × ℝ := (Real.sqrt 3, -2 * Real.sqrt 3)
def angle : ℝ := 135

-- We need to prove that the line equation is x + y + sqrt(3) = 0
theorem line_equation_through_pointP_and_angle :
  ∃ (a b c : ℝ), a * pointP.1 + b * pointP.2 + c = 0 ∧
                 a = 1 ∧ b = 1 ∧ c = Real.sqrt 3 :=
by
  sorry

end line_equation_through_pointP_and_angle_l636_636674


namespace negation_existential_statement_l636_636294

theorem negation_existential_statement : 
  ¬ (∃ x₀ : ℝ, 2^x₀ ≤ 0) ↔ ∀ x : ℝ, 2^x > 0 :=
by sorry

end negation_existential_statement_l636_636294


namespace probability_selecting_cooking_l636_636910

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636910


namespace good_pairs_of_lines_l636_636031

def line1 (x : ℝ) := 4 * x + 6
def line2 (x : ℝ) := 3 * x + 4
def line3 (x : ℝ) := 4 * x - (1 / 3)
def line4 (x : ℝ) := (3 / 2) * x - 2
def line5 (x : ℝ) := (3 / 4) * x - (3 / 2)

theorem good_pairs_of_lines :
  (∃ L1 L2, 
     ((L1 = line1 ∨ L1 = line2 ∨ L1 = line3 ∨ L1 = line4 ∨ L1 = line5) ∧ 
      (L2 = line1 ∨ L2 = line2 ∨ L2 = line3 ∨ L2 = line4 ∨ L2 = line5) ∧ 
      (L1 ≠ L2) ∧ 
      ((∀ x, L1 x = L2 x) ∨ 
       ∃ m1 m2, 
         (L1 = λ x, m1 * x + _ ∧ L2 = λ x, m2 * x + _ ∧ m1 * m2 = -1))) ↔ 
    (L1 = line1 ∧ L2 = line3)) :=
by sorry

end good_pairs_of_lines_l636_636031


namespace probability_of_selecting_cooking_l636_636950

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636950


namespace part_a_part_b_l636_636237

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def canMoveInsideWithReflections (A : ℝ × ℝ) (circle : Circle) (n : ℕ) : Prop :=
  ∃ (A' : ℝ × ℝ), distance A' circle.center < circle.radius ∧ 
                 (∀ i : ℕ, i < n → some_symmetrical_reflection(A, circle, i) = A')

theorem part_a 
  (A : ℝ × ℝ)
  (circle : Circle)
  (hA : distance A circle.center = 50)
  (hR : circle.radius = 1) :
  canMoveInsideWithReflections A circle 25 := 
by 
  sorry

theorem part_b
  (A : ℝ × ℝ)
  (circle : Circle)
  (hA : distance A circle.center = 50)
  (hR : circle.radius = 1) :
  ¬ canMoveInsideWithReflections A circle 24 := 
by 
  sorry

end part_a_part_b_l636_636237


namespace total_number_of_boys_in_class_is_40_l636_636342

theorem total_number_of_boys_in_class_is_40 
  (n : ℕ) (h : 27 - 7 = n / 2):
  n = 40 :=
by
  sorry

end total_number_of_boys_in_class_is_40_l636_636342


namespace repeating_decimal_as_fraction_l636_636138

theorem repeating_decimal_as_fraction :
  let x := 0.5656565656565656 -- representing 0.\overline{56}
  let a := 56
  let b := 99
  x = a / b ∧ gcd a b = 1 →
  a + b = 155 :=
by
  intros h
  sorry -- Proof skipped

end repeating_decimal_as_fraction_l636_636138


namespace cos_arith_seq_l636_636045

theorem cos_arith_seq (a : ℝ) (ha : 0 < a ∧ a < 360) 
  (h_seq : cos a + cos (4 * a) = 2 * cos (2 * a)) : 
  a = 106 ∨ a = 254 := 
sorry

end cos_arith_seq_l636_636045


namespace solve_x_division_l636_636753

theorem solve_x_division :
  ∀ x : ℝ, (3 / x + 4 / x / (8 / x) = 1.5) → x = 3 := 
by
  intro x
  intro h
  sorry

end solve_x_division_l636_636753


namespace range_of_a_l636_636487

variable (f : ℝ → ℝ)

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f y < f x

theorem range_of_a 
  (decreasing_f : is_decreasing f)
  (hfdef : ∀ x, -1 ≤ x ∧ x ≤ 1 → f (2 * x - 3) < f (x - 2)) :
  ∃ a : ℝ, 1 < a ∧ a ≤ 2  :=
by 
  sorry

end range_of_a_l636_636487


namespace average_visitors_on_Sundays_l636_636968

theorem average_visitors_on_Sundays (S : ℕ) 
  (h1 : 30 % 7 = 2)  -- The month begins with a Sunday
  (h2 : 25 = 30 - 5)  -- The month has 25 non-Sundays
  (h3 : (120 * 25) = 3000) -- Total visitors on non-Sundays
  (h4 : (125 * 30) = 3750) -- Total visitors for the month
  (h5 : 5 * 30 > 0) -- There are a positive number of Sundays
  : S = 150 :=
by
  sorry

end average_visitors_on_Sundays_l636_636968


namespace sequence_sum_general_term_l636_636084

theorem sequence_sum_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n) : ∀ n, a n = 2 * n :=
by 
  sorry

end sequence_sum_general_term_l636_636084


namespace speed_of_second_car_l636_636716

theorem speed_of_second_car (d : ℝ) (s1 : ℝ) (t : ℝ) (s2 : ℝ) : d = 333 ∧ s1 = 54 ∧ t = 3 ∧ (s1 * t + s2 * t = d) → s2 = 57 :=
by
  intro h,
  cases h with h1 h,
  cases h with h2 h,
  cases h with h3 h4,
  rw [h1, h2, h3] at h4,
  calc s2 = 57 : by sorry

end speed_of_second_car_l636_636716


namespace sum_series_approx_l636_636330

noncomputable def sum_series :=
  (Finset.sum (Finset.range 1500) (λ n, 3 / ((n+1) * (n+4))))

theorem sum_series_approx :
  abs (sum_series - 1.830) < 0.001 :=
by sorry

end sum_series_approx_l636_636330


namespace area_of_region_l636_636023

theorem area_of_region (x y : ℝ) : (|4 * x - 14| + |3 * y - 9| ≤ 6) → 6 :=
by sorry

end area_of_region_l636_636023


namespace exists_sum53_l636_636309

theorem exists_sum53
  (A : Finset ℕ) 
  (h_card : A.card = 53)
  (h_sum : A.sum id ≤ 1990) : 
  ∃ a b ∈ A, a + b = 53 :=
sorry

end exists_sum53_l636_636309


namespace problem_statement_l636_636603

open Complex

theorem problem_statement (a b : ℝ) (h : (1 + (I : ℂ) * (sqrt 3))^3 + a * (1 + (I * sqrt 3)) + b = 0) :
  a + b = 8 := 
sorry

end problem_statement_l636_636603


namespace probability_of_selecting_cooking_l636_636850

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636850


namespace integral_of_f_eq_7_div_6_l636_636473

-- Define the function according to the conditions
def f (x : ℝ) (f'1 : ℝ) : ℝ := f'1 * x ^ 2 + x + 1

-- State the theorem that needs to be proven
theorem integral_of_f_eq_7_div_6 (f'1 : ℝ) (h : f'1 = -1) : 
  ∫ x in 0..1, f x f'1 = 7 / 6 := by
  sorry

end integral_of_f_eq_7_div_6_l636_636473


namespace greatest_prime_factor_of_176_l636_636327

theorem greatest_prime_factor_of_176 : 
  ∃ p, (Prime p ∧ p ∣ 176) ∧ ∀ q, (Prime q ∧ q ∣ 176) → q ≤ p :=
by
  have h1 : 176 = 2 * 88 := by norm_num
  have h2 : 88 = 2 * 44 := by norm_num
  have h3 : 44 = 2 * 22 := by norm_num
  have h4 : 22 = 2 * 11 := by norm_num
  have h5 : Prime 11 := by norm_num
  existsi 11
  constructor
  constructor
  exact h5
  rw [h1, h2, h3, h4]
  norm_num
  intros q hq
  have hq_prime := hq.1
  have hq_div := hq.2
  sorry

end greatest_prime_factor_of_176_l636_636327


namespace major_axis_of_ellipse_l636_636684

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + y^2 = 16

-- Define the length of the major axis
def major_axis_length : ℝ := 8

-- The theorem to prove
theorem major_axis_of_ellipse : 
  (∀ x y : ℝ, ellipse_eq x y) → major_axis_length = 8 :=
by
  sorry

end major_axis_of_ellipse_l636_636684


namespace statues_created_first_year_l636_636520

-- Definition of the initial conditions and the variable representing the number of statues created in the first year.
variables (S : ℕ)

-- Condition 1: In the second year, statues are quadrupled.
def second_year_statues : ℕ := 4 * S

-- Condition 2: In the third year, 12 statues are added, and 3 statues are broken.
def third_year_statues : ℕ := second_year_statues S + 12 - 3

-- Condition 3: In the fourth year, twice as many new statues are added as had been broken the previous year (2 * 3).
def fourth_year_added_statues : ℕ := 2 * 3
def fourth_year_statues : ℕ := third_year_statues S + fourth_year_added_statues

-- Condition 4: Total number of statues at the end of four years is 31.
def total_statues : ℕ := fourth_year_statues S

theorem statues_created_first_year : total_statues S = 31 → S = 4 :=
by {
  sorry
}

end statues_created_first_year_l636_636520


namespace shoe_cost_l636_636025

def initial_amount : ℕ := 91
def cost_sweater : ℕ := 24
def cost_tshirt : ℕ := 6
def amount_left : ℕ := 50
def cost_shoes : ℕ := 11

theorem shoe_cost :
  initial_amount - (cost_sweater + cost_tshirt) - amount_left = cost_shoes :=
by
  sorry

end shoe_cost_l636_636025


namespace probability_taequan_wins_l636_636264

open Finset

-- Definition of fair 6-sided dice
def fair_six_sided_dice := {1, 2, 3, 4, 5, 6}

-- Definition of rolling three dice (set of triples)
def roll_three_dice := (fair_six_sided_dice.product fair_six_sided_dice).product fair_six_sided_dice

-- Definition of the specific roll {2, 3, 4}
def specific_roll := { (2, 3, 4), (2, 4, 3), (3, 2, 4), (3, 4, 2), (4, 2, 3), (4, 3, 2) }

-- Total possible outcomes when rolling three 6-sided dice
def total_outcomes := 6 * 6 * 6

-- Number of favorable outcomes
def favorable_outcomes := specific_roll.card

-- Probability calculation
def probability_of_winning := favorable_outcomes.to_real / total_outcomes.to_real

theorem probability_taequan_wins :
  probability_of_winning = (1 / 36 : ℝ) :=
begin
  unfold probability_of_winning,
  unfold favorable_outcomes,
  unfold total_outcomes,
  rw [card_eq 6],
  rw [nat.cast_mul, nat.cast_mul, nat.cast_bit0, nat.cast_bit0, nat.cast_bit1],
  norm_num,
  sorry
end

end probability_taequan_wins_l636_636264


namespace complement_of_A_l636_636208

namespace SetTheory

variable (U : Set ℕ) (A : Set ℕ) (complement_A : Set ℕ)

-- Defining the universal set U
def U_def := { x : ℕ | x > -2 ∧ x < 4 ∧ x ∈ Set.univ } -- essentially {0, 1, 2, 3}

-- Defining the set A
def A_def := {0, 2}

-- Defining the complement of A with respect to U
def complement_A_def := { x | x ∈ U ∧ x ∉ A }

theorem complement_of_A :
  complement_A = {1, 3} :=
by
  sorry

end SetTheory

end complement_of_A_l636_636208


namespace num_people_price_item_equation_l636_636572

theorem num_people_price_item_equation
  (x y : ℕ)
  (h1 : 8 * x = y + 3)
  (h2 : 7 * x = y - 4) :
  (y + 3) / 8 = (y - 4) / 7 :=
by
  sorry

end num_people_price_item_equation_l636_636572


namespace probability_of_selecting_cooking_l636_636946

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636946


namespace sunglasses_price_l636_636005

theorem sunglasses_price (P : ℝ) 
  (buy_cost_per_pair : ℝ := 26) 
  (pairs_sold : ℝ := 10) 
  (sign_cost : ℝ := 20) :
  (pairs_sold * P - pairs_sold * buy_cost_per_pair) / 2 = sign_cost →
  P = 30 := 
by
  sorry

end sunglasses_price_l636_636005


namespace sum_of_decimals_l636_636352

theorem sum_of_decimals : (1 / 10) + (9 / 100) + (9 / 1000) + (7 / 10000) = 0.1997 := 
sorry

end sum_of_decimals_l636_636352


namespace Eva_needs_weeks_l636_636034

theorem Eva_needs_weeks (apples : ℕ) (days_in_week : ℕ) (weeks : ℕ) 
  (h1 : apples = 14)
  (h2 : days_in_week = 7) 
  (h3 : apples = weeks * days_in_week) : 
  weeks = 2 := 
by 
  sorry

end Eva_needs_weeks_l636_636034


namespace rajas_monthly_income_l636_636246

theorem rajas_monthly_income (I : ℝ) (h : 0.6 * I + 0.1 * I + 0.1 * I + 5000 = I) : I = 25000 :=
sorry

end rajas_monthly_income_l636_636246


namespace num_people_price_item_equation_l636_636573

theorem num_people_price_item_equation
  (x y : ℕ)
  (h1 : 8 * x = y + 3)
  (h2 : 7 * x = y - 4) :
  (y + 3) / 8 = (y - 4) / 7 :=
by
  sorry

end num_people_price_item_equation_l636_636573


namespace find_y_value_l636_636121

theorem find_y_value (k : ℝ) (h1 : ∀ (x : ℝ), y = k * x) 
(h2 : y = 4 ∧ x = 2) : 
(∀ (x : ℝ), x = -2 → y = -4) := 
by 
  sorry

end find_y_value_l636_636121


namespace hook_squares_l636_636229

theorem hook_squares (k : ℕ) (h : 2 ≤ k)
    (squares : Finset (Finset Color)) (colors : Finset Color)
    (hsquares : ∀ (c : Color), c ∈ colors) :
    (∀ (subset : Finset Color), subset.card = k -> ∃ (c1 c2 : Color), c1 ∈ subset ∧ c2 ∈ subset ∧ hooked (squares.filter(λ s, ∃ (color : Color), color ∈ s).card k).card = 2k-2 :=
by
  sorry

end hook_squares_l636_636229


namespace possible_skew_edges_parallelepiped_l636_636582

theorem possible_skew_edges_parallelepiped (P : Parallelepiped) (L : Line) (face : Plane) 
  (h_P_face : face ∈ faces P) (h_L_face : L ∈ face) :
  ∃ n : ℕ, n ∈ {4, 6, 7, 8} ∧ (∃ edges_not_in_same_plane : ℕ, edges_not_in_same_plane = n ∧ edges_not_in_same_plane ∉ same_plane L) :=
sorry

end possible_skew_edges_parallelepiped_l636_636582


namespace baby_achieves_goal_l636_636231

theorem baby_achieves_goal :
  ∀ (buns : list ℕ), buns.length = 40 →
    (count (λ b, b = 1) buns = 20) →
    (count (λ b, b = 2) buns = 20) →
      ∃ (selection : list ℕ),
        selection.length = 20 ∧
        (count (λ b, b = 1) selection = 10) ∧
        (count (λ b, b = 2) selection = 10) :=
by
  sorry

end baby_achieves_goal_l636_636231


namespace determine_w_minus_y_l636_636420
    
noncomputable def integers (x y z w : ℤ) : ℤ :=
x^3 = y^4 ∧ z^5 = w^2 ∧ z - x = 31
    ∧ w - y = -759439

theorem determine_w_minus_y (x y z w : ℤ)
  (hx : x^3 = y^4)
  (hz : z^5 = w^2)
  (hzx : z - x = 31) :
  w - y = -759439 := by
  apply integers,
  sorry

end determine_w_minus_y_l636_636420


namespace find_angle_B_find_sin_C_l636_636547

-- Statement for proving B = π / 4 given the conditions
theorem find_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.sin A + c * Real.sin C - Real.sqrt 2 * a * Real.sin C = b * Real.sin B) 
  (hABC : A + B + C = Real.pi) :
  B = Real.pi / 4 := 
sorry

-- Statement for proving sin C when cos A = 1 / 3
theorem find_sin_C (A C : ℝ) 
  (hA : Real.cos A = 1 / 3)
  (hABC : A + Real.pi / 4 + C = Real.pi) :
  Real.sin C = (4 + Real.sqrt 2) / 6 := 
sorry

end find_angle_B_find_sin_C_l636_636547


namespace Paige_team_players_l636_636768

/-- Paige's team won their dodgeball game and scored 41 points total.
    If Paige scored 11 points and everyone else scored 6 points each,
    prove that the total number of players on the team was 6. -/
theorem Paige_team_players (total_points paige_points other_points : ℕ) (x : ℕ) (H1 : total_points = 41) (H2 : paige_points = 11) (H3 : other_points = 6) (H4 : paige_points + other_points * x = total_points) : x + 1 = 6 :=
by {
  sorry
}

end Paige_team_players_l636_636768


namespace num_ways_to_join_points_interior_l636_636266

/-- 
Condition:
    - We have 12 points.
    - We need to pair them using 6 line segments.
    - The line segments must pass through the interior of a triangle.

    Goal: Prove that the number of ways to form these line segments is 216.
-/
theorem num_ways_to_join_points_interior : 
    ∃ (points : set ℕ), points.card = 12 ∧ 
    ∃ (pairs : set (set ℕ)), pairs.card = 6 ∧ 
    (∀ p ∈ pairs, ∃ a b ∈ points, a ≠ b ∧ p = {a, b}) ∧
    (∀ p ∈ pairs, p ⊆ {1, 2, ..., 12}) → 
    count_possible_pairings pairs = 216 :=
sorry

end num_ways_to_join_points_interior_l636_636266


namespace largest_of_five_consecutive_integers_l636_636454

theorem largest_of_five_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120) : n + 4 = 9 :=
sorry

end largest_of_five_consecutive_integers_l636_636454


namespace probability_selecting_cooking_l636_636911

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636911


namespace last_digit_2008_pow_2008_l636_636288

theorem last_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := by
  -- Here, the proof would follow the understanding of the cyclic pattern of the last digits of powers of 2008
  sorry

end last_digit_2008_pow_2008_l636_636288


namespace probability_of_cooking_l636_636798

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636798


namespace margo_total_distance_travelled_l636_636625

noncomputable def total_distance_walked (walking_time_in_minutes: ℝ) (stopping_time_in_minutes: ℝ) (additional_walking_time_in_minutes: ℝ) (walking_speed: ℝ) : ℝ :=
  walking_speed * ((walking_time_in_minutes + stopping_time_in_minutes + additional_walking_time_in_minutes) / 60)

noncomputable def total_distance_cycled (cycling_time_in_minutes: ℝ) (cycling_speed: ℝ) : ℝ :=
  cycling_speed * (cycling_time_in_minutes / 60)

theorem margo_total_distance_travelled :
  let walking_time := 10
  let stopping_time := 15
  let additional_walking_time := 10
  let cycling_time := 15
  let walking_speed := 4
  let cycling_speed := 10

  total_distance_walked walking_time stopping_time additional_walking_time walking_speed +
  total_distance_cycled cycling_time cycling_speed = 4.8333 := 
by 
  sorry

end margo_total_distance_travelled_l636_636625


namespace fruit_selling_price_3640_l636_636652

def cost_price := 22
def initial_selling_price := 38
def initial_quantity_sold := 160
def price_reduction := 3
def quantity_increase := 120
def target_profit := 3640

theorem fruit_selling_price_3640 (x : ℝ) :
  ((initial_selling_price - x - cost_price) * (initial_quantity_sold + (x / price_reduction) * quantity_increase) = target_profit) →
  x = 9 →
  initial_selling_price - x = 29 :=
by
  intro h1 h2
  sorry

end fruit_selling_price_3640_l636_636652


namespace greatest_prime_factor_of_176_l636_636328

theorem greatest_prime_factor_of_176 : 
  ∃ p, (Prime p ∧ p ∣ 176) ∧ ∀ q, (Prime q ∧ q ∣ 176) → q ≤ p :=
by
  have h1 : 176 = 2 * 88 := by norm_num
  have h2 : 88 = 2 * 44 := by norm_num
  have h3 : 44 = 2 * 22 := by norm_num
  have h4 : 22 = 2 * 11 := by norm_num
  have h5 : Prime 11 := by norm_num
  existsi 11
  constructor
  constructor
  exact h5
  rw [h1, h2, h3, h4]
  norm_num
  intros q hq
  have hq_prime := hq.1
  have hq_div := hq.2
  sorry

end greatest_prime_factor_of_176_l636_636328


namespace sum_of_n_values_l636_636739

theorem sum_of_n_values : sum (λ n, |3 * n - 8| = 5) = 16/3 := 
by
  sorry

end sum_of_n_values_l636_636739


namespace probability_of_cooking_l636_636803

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636803


namespace sum_of_n_values_l636_636738

theorem sum_of_n_values : sum (λ n, |3 * n - 8| = 5) = 16/3 := 
by
  sorry

end sum_of_n_values_l636_636738


namespace ring_area_to_new_circle_radius_l636_636317

theorem ring_area_to_new_circle_radius
  (r1 r2 : ℝ) (h_r1 : r1 = 24) (h_r2 : r2 = 33) :
  ∃ r, π * r^2 = π * (r2^2 - r1^2) ∧ r = 3*real.sqrt 57 := 
by
  sorry

end ring_area_to_new_circle_radius_l636_636317


namespace water_volume_is_correct_l636_636156

-- Define the base radius of the cylindrical container
def base_radius : ℝ := 1

-- Define the radius of each iron ball
def ball_radius : ℝ := 0.5

-- Define the height of water required to cover the balls
def water_height : ℝ := 1 + (Real.sqrt 2) / 2

-- Define the volume of water (to be poured) calculated including height and base area excluding the volume of four balls.
def V_water : ℝ := π * base_radius^2 * water_height

-- Define the volume of a single iron ball
def V_ball : ℝ := (4 / 3) * π * ball_radius^3

-- Define the volume of four iron balls
def V_four_balls : ℝ := 4 * V_ball

-- The actual volume of water needed to displace the spheres
def V_total : ℝ := V_water - V_four_balls

-- The expected volume as given in the problem statement
def expected_volume : ℝ := π * (1 / 3 + (Real.sqrt 2) / 2)

theorem water_volume_is_correct : V_total = expected_volume := by
  -- The proof stays as sorry since we are focusing only on the statement.
  sorry

end water_volume_is_correct_l636_636156


namespace area_of_triangle_PUT_is_two_thirds_l636_636567

-- Define the vertices of the square PQRS
def P : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (2, 2)
def S : ℝ × ℝ := (0, 2)

-- Given T on PR
def T (t : ℝ) : ℝ × ℝ := (2, t)

-- Equation of diagonal QS: y = -x + 2
def line_QS (x : ℝ) : ℝ := -x + 2

-- Equation of line PT: y = t/2 * x
def line_PT (t x : ℝ) : ℝ := (t / 2) * x

-- Intersection point U
def U (t : ℝ) : ℝ × ℝ :=
  let x_U := 4 / (t + 2)
  let y_U := (2 * t) / (t + 2)
  (x_U, y_U)

-- Area of triangle PUT
def area_PUT (t : ℝ) : ℝ :=
  let base := 2
  let height := (2 * t) / (t + 2)
  (1 / 2) * base * height

-- Proving the area of triangle PUT is 2/3 given t is such that triangle PRT is equilateral
theorem area_of_triangle_PUT_is_two_thirds
  (t : ℝ) (ht : t = 1) :
  area_PUT t = 2 / 3 :=
by
  rw [← ht]
  sorry

end area_of_triangle_PUT_is_two_thirds_l636_636567


namespace total_miles_cycled_l636_636183

theorem total_miles_cycled (D : ℝ) (store_to_peter : ℝ) (same_speed : Prop)
  (time_relation : 2 * D = store_to_peter) (store_to_peter_dist : store_to_peter = 50) :
  D + store_to_peter + store_to_peter = 200 :=
by
  have h1 : D = 100, from sorry  -- Solved from store_to_peter = 50 and time_relation means D = 50*2 = 100
  by rw [h1, store_to_peter_dist]; norm_num; exact h1 

end total_miles_cycled_l636_636183


namespace smaller_two_digit_product_l636_636689

-- Statement of the problem
theorem smaller_two_digit_product (a b : ℕ) (h : a * b = 4536) (h_a : 10 ≤ a ∧ a < 100) (h_b : 10 ≤ b ∧ b < 100) : a = 21 ∨ b = 21 := 
by {
  sorry -- proof not needed
}

end smaller_two_digit_product_l636_636689


namespace bus_total_capacity_l636_636153

-- Definitions based on conditions in a)
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seats_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 12

-- Proof statement
theorem bus_total_capacity : (left_side_seats + right_side_seats) * seats_per_seat + back_seat_capacity = 93 := by
  sorry

end bus_total_capacity_l636_636153


namespace probability_cooking_is_one_fourth_l636_636855
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636855


namespace problem1_problem2_l636_636356

noncomputable def calculate_expression : ℝ :=
  (-2:ℝ)^2 + (real.sqrt 3 - real.pi)^0 + abs (1 - real.sqrt 3)

theorem problem1 : calculate_expression = 4 + real.sqrt 3 := 
  sorry

variables (x y : ℝ)

theorem problem2 :
  (2 * x + y = 1) ∧ (x - 2 * y = 3) → (x = 1) ∧ (y = -1) := 
  sorry

end problem1_problem2_l636_636356


namespace sum_of_n_values_l636_636747

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l636_636747


namespace find_initial_crayons_l636_636633

namespace CrayonProblem

variable (gave : ℕ) (lost : ℕ) (additional_lost : ℕ) 

def correct_answer (gave lost additional_lost : ℕ) :=
  gave + lost = gave + (gave + additional_lost) ∧ gave + lost = 502

theorem find_initial_crayons
  (gave := 90)
  (lost := 412)
  (additional_lost := 322)
  : correct_answer gave lost additional_lost :=
by 
  sorry

end CrayonProblem

end find_initial_crayons_l636_636633


namespace sum_of_roots_of_quadratic_eq_l636_636331

theorem sum_of_roots_of_quadratic_eq (x : ℝ) :
  (x^2 - 7 * x - 10 = 0) → x.sum_of_roots = 7 :=
  sorry

end sum_of_roots_of_quadratic_eq_l636_636331


namespace total_distance_l636_636186

-- Define the distances as nonnegative real numbers
noncomputable def distance (a b : ℝ) : Prop := a ≥ 0 ∧ b ≥ 0

-- Define the conditions for the problem
def HomeToStore (d₁ : ℝ) : Prop := distance d₁ 50 -- from Home to Store
def StoreToPeter : ℝ := 50 -- from Store to Peter
def TimeRelation (d₁ d₂ : ℝ) (t₁ t₂ : ℝ) : Prop := 2 * t₂ = t₁ ∧ d₁ = d₂ -- Time and distance relation

-- Prove the total distance
theorem total_distance (d₁ d₂ t₁ t₂ : ℝ) (hHomeStore : HomeToStore d₁) (hStorePeter : StoreToPeter = d₂) (hTimeRel : TimeRelation d₁ d₂ t₁ t₂) : (d₁ + d₂ + d₂) = 150 := 
by 
  -- hHomeStore: distance d₁ 50 -> d₁ = 50 (by definition)
  have h₁ : d₁ = 50 := by sorry

  -- hStorePeter: d₂ = 50 (by definition)
  have h₂ : d₂ = 50 := by sorry

  -- Calc: d₁ + d₂ + d₂ = 50 + 50 + 50 = 150
  calc
    d₁ + d₂ + d₂ = 50 + 50 + 50 := by sorry
    ... = 150 := by sorry

end total_distance_l636_636186


namespace relative_prime_linear_combination_multiple_l636_636090

theorem relative_prime_linear_combination_multiple (m n k : ℕ) : 
  ∃ r s : ℕ, Nat.coprime r s ∧ k ∣ (r * m + s * n) :=
by
  sorry

end relative_prime_linear_combination_multiple_l636_636090


namespace tangent_line_at_point_sqrt_l636_636675

noncomputable def tangent_line (f : ℝ → ℝ) (p : ℝ × ℝ) : ℝ × ℝ × ℝ :=
let df := (deriv f) p.1;
let (x1, y1) := p;
let m := df in
(x1 - 4 * y1 + 4, m, df)

theorem tangent_line_at_point_sqrt :
  tangent_line (λ x : ℝ, real.sqrt x) (4, 2) = (1, 1/4, 1/4) := by
simp only [tangent_line, real.sqrt, deriv_sqrt];
simp;
sorry

end tangent_line_at_point_sqrt_l636_636675


namespace larger_number_is_299_l636_636286

theorem larger_number_is_299 (A B : ℕ) 
  (HCF_AB : Nat.gcd A B = 23) 
  (LCM_12_13 : Nat.lcm A B = 23 * 12 * 13) : 
  max A B = 299 := 
sorry

end larger_number_is_299_l636_636286


namespace sum_of_solutions_of_absolute_value_l636_636725

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l636_636725


namespace part1_solution_part2_no_solution_l636_636351

theorem part1_solution (x y : ℚ) :
  x + y = 5 ∧ 3 * x + 10 * y = 30 ↔ x = 20 / 7 ∧ y = 15 / 7 :=
by
  sorry

theorem part2_no_solution (x : ℚ) :
  (x + 7) / 2 < 4 ∧ (3 * x - 1) / 2 ≤ 2 * x - 3 ↔ False :=
by
  sorry

end part1_solution_part2_no_solution_l636_636351


namespace probability_of_selecting_cooking_l636_636843

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636843


namespace who_sits_in_middle_l636_636159

-- Definitions for each of the conditions
def seat := ℕ -- Let's represent seat positions by natural numbers

variables (Lauren Aaron Sharon Darren Karen : seat → Prop)
variables (middle : seat)

-- Expressing the problem's conditions
axiom L_last_seat : Lauren 4
axiom A_end : Aaron 0 ∨ Aaron 4
axiom A_behind_S : ∀ (i : seat), Aaron i → Sharon (i - 1)
axiom D_front_of_A : ∀ (i j : seat), Darren i → Aaron j → i < j
axiom K_D_sep : ∀ (i j : seat), Darren i → Karen j → (i < j - 1 ∨ j < i - 1)

-- The main theorem stating who sat in the middle seat
theorem who_sits_in_middle : Sharon 2 :=
sorry

end who_sits_in_middle_l636_636159


namespace probability_of_selecting_cooking_l636_636842

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636842


namespace farmer_land_l636_636630

theorem farmer_land (A : ℝ) (h1 : 0.9 * A = A_cleared) (h2 : 0.3 * A_cleared = A_soybeans) 
  (h3 : 0.6 * A_cleared = A_wheat) (h4 : 0.1 * A_cleared = 540) : A = 6000 :=
by
  sorry

end farmer_land_l636_636630


namespace determine_x_for_collinear_l636_636544

-- Define points and slopes
structure Point where
  x : ℝ
  y : ℝ

def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

def collinear (A B C : Point) : Prop :=
  slope A B = slope A C

-- Create the points
def A : Point := ⟨-1, -2⟩
def B : Point := ⟨4, 8⟩

-- Define the proof goal
theorem determine_x_for_collinear (x : ℝ) (C : Point := ⟨5, x⟩) : x = 10 :=
  collinear A B C → x = 10 := sorry

end determine_x_for_collinear_l636_636544


namespace transformed_function_eq_l636_636681

-- Define the original function
def g(x : ℝ) : ℝ := sin x * cos x

-- Define the transformation function
def y_sqrt3x(x : ℝ) : ℝ := sqrt 3 * x

-- Define the translated function
def f(x : ℝ) : ℝ := (1/2) * sin (2 * x + 2) - sqrt 3

-- The theorem statement
theorem transformed_function_eq :
  (∀ x : ℝ, f(x) = (1/2) * sin (2 * (x + 1)) - sqrt 3) →
  (∀ x : ℝ, g(x) = sin x * cos x) →
  ∀ x : ℝ, (f(x) = (1/2) * sin (2 * x + 2) - √3) :=
by
  intros h_transform h_g x
  -- Skipping the proof with sorry
  sorry

end transformed_function_eq_l636_636681


namespace largest_of_five_consecutive_integers_with_product_15120_l636_636456

theorem largest_of_five_consecutive_integers_with_product_15120 :
  ∃ (a b c d e : ℕ), a * b * c * d * e = 15120 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e = 12 :=
begin
  sorry
end

end largest_of_five_consecutive_integers_with_product_15120_l636_636456


namespace find_common_difference_l636_636560

noncomputable def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 4 = 7 ∧ a 3 + a 6 = 16

theorem find_common_difference (a : ℕ → ℝ) (d : ℝ) (h : common_difference a d) : d = 2 :=
by
  sorry

end find_common_difference_l636_636560


namespace segment_lengths_sum_ge_one_over_k_l636_636296

noncomputable def segment (α : Type*) := set (set α)

noncomputable def length {α : Type*} [metric_space α] (s : segment α) : ℝ := sorry

noncomputable def M (α : Type*) := set (segment α)

def disjoint {α : Type*} (s t : segment α) : Prop := ∀ x ∈ s, x ∉ t

theorem segment_lengths_sum_ge_one_over_k
  {α : Type*}
  [metric_space α]
  (k : ℕ)
  (M : M α)
  (H1 : ∀ (s ∈ M) (t ∈ M), s ≠ t → disjoint s t)
  (H2 : ∀ (l ≤ 1), ∃ (s ∈ M), ∃ (x y ∈ s), dist x y = l)
  : ∑ s in M, length s ≥ 1 / k :=
begin
  sorry
end

end segment_lengths_sum_ge_one_over_k_l636_636296


namespace probability_of_selecting_cooking_l636_636955

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636955


namespace probability_of_selecting_cooking_is_one_fourth_l636_636867

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636867


namespace probability_of_cooking_l636_636792

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636792


namespace find_y_when_x_is_neg2_l636_636118

noncomputable def k : ℝ :=
4 / 2

def f (x : ℝ) : ℝ := k * x

theorem find_y_when_x_is_neg2 : f (-2) = -4 :=
sorry

end find_y_when_x_is_neg2_l636_636118


namespace points_outside_circle_l636_636179

noncomputable def point_set_condition (R : ℝ) (A : ℝ × ℝ) : Prop :=
  let origin : ℝ × ℝ := (0, 0)
  let distance_from_origin := Math.sqrt ((A.1 - origin.1)^2 + (A.2 - origin.2)^2)
  distance_from_origin > R / 3

theorem points_outside_circle (R : ℝ) (A : ℝ × ℝ) (h_ne_center : A ≠ (0,0)) (h_three_reflections : 
  ∃ B C : ℝ × ℝ, -- reflection points
  -- conditions on reflections can be formulated here
  -- for simplicity, assume there exists such points and conditions are met
  True) : 
  point_set_condition R A :=
by
  sorry

end points_outside_circle_l636_636179


namespace definite_integral_value_l636_636424

noncomputable def definite_integral := ∫ x in real.pi / 4 .. real.arctan 3, (1 + real.cot x) / (real.sin x + 2 * real.cos x)^2

theorem definite_integral_value :
  definite_integral = (1/4) * real.log (9 / 5) + (1 / 15) :=
sorry

end definite_integral_value_l636_636424


namespace sum_of_coefficients_l636_636702

def polynomial := (λ x : ℤ, (x^3 + 2 * x + 1) * (3 * x^2 + 4))

theorem sum_of_coefficients : polynomial 1 = 28 :=
  by sorry

end sum_of_coefficients_l636_636702


namespace find_x_if_parallel_vectors_l636_636131

variable (x : ℝ) (a b : ℝ × ℝ)

def a := (3, -2)
def b := (x, 4)

def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_x_if_parallel_vectors : parallel a b → x = -6 := by
  intro h
  unfold parallel at h
  simp at h
  sorry

end find_x_if_parallel_vectors_l636_636131


namespace find_angle_x_l636_636238

theorem find_angle_x (O A B C D : Point) (x : ℝ) (hO : CenterOfCircle O) 
  (hCDA : ∠CDA = 42) (hCyclic : CyclicQuadrilateral A B C D) 
  (hDBC : ∠DBC = 10) : x = 58 :=
sorry

end find_angle_x_l636_636238


namespace sum_of_n_values_l636_636752

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l636_636752


namespace polynomial_abc_l636_636539

theorem polynomial_abc {a b c : ℝ} (h : a * x^2 + b * x + c = x^2 - 3 * x + 2) : a * b * c = -6 := by
  sorry

end polynomial_abc_l636_636539


namespace calculator_display_after_101_presses_l636_636660

theorem calculator_display_after_101_presses :
  let x : ℕ → ℚ := λ n, Nat.recOn n (7 : ℚ) (λ n x_n, 1 / (1 - x_n))
  in x 101 = 6 / 7 :=
by 
  sorry

end calculator_display_after_101_presses_l636_636660


namespace chord_intercept_length_l636_636290

theorem chord_intercept_length (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = 3 → (x + y + a = 0 → true)) →
  ∃ a : ℝ, a = ±(3 * real.sqrt(2) / 2) := 
by
  sorry

end chord_intercept_length_l636_636290


namespace correct_operation_l636_636334

-- Define the conditions
def cond1 (m : ℝ) : Prop := m^2 + m^3 ≠ m^5
def cond2 (m : ℝ) : Prop := m^2 * m^3 = m^5
def cond3 (m : ℝ) : Prop := (m^2)^3 = m^6

-- Main statement that checks the correct operation
theorem correct_operation (m : ℝ) : cond1 m → cond2 m → cond3 m → (m^2 * m^3 = m^5) :=
by
  intros h1 h2 h3
  exact h2

end correct_operation_l636_636334


namespace total_sticks_of_gum_in_12_brown_boxes_l636_636006

-- Definitions based on the conditions
def packs_per_carton := 7
def sticks_per_pack := 5
def cartons_in_full_box := 6
def cartons_in_partial_box := 3
def num_brown_boxes := 12
def num_partial_boxes := 2

-- Calculation definitions
def sticks_per_carton := packs_per_carton * sticks_per_pack
def sticks_per_full_box := cartons_in_full_box * sticks_per_carton
def sticks_per_partial_box := cartons_in_partial_box * sticks_per_carton
def num_full_boxes := num_brown_boxes - num_partial_boxes

-- Final total sticks of gum
def total_sticks_of_gum := (num_full_boxes * sticks_per_full_box) + (num_partial_boxes * sticks_per_partial_box)

-- The theorem to be proved
theorem total_sticks_of_gum_in_12_brown_boxes :
  total_sticks_of_gum = 2310 :=
by
  -- The proof is omitted.
  sorry

end total_sticks_of_gum_in_12_brown_boxes_l636_636006


namespace sum_of_elements_in_B_l636_636509

theorem sum_of_elements_in_B :
  let A := {-3, -2, -1, 0, 1, 2}
  let B := {y | ∃ x ∈ A, y = x^2 - 1}
  (Finset.sum B id) = 10 :=
by {
  -- define set A
  let A : Finset ℤ := {-3, -2, -1, 0, 1, 2},

  -- define set B
  let B : Finset ℤ := A.image (λ x, x^2 - 1),

  -- calculate the sum of elements in B
  have h_sum : B.sum id = 10,
  {
    -- we will provide the proof here
    sorry,
  },
  exact h_sum,
}

end sum_of_elements_in_B_l636_636509


namespace fruit_selling_price_3640_l636_636651

def cost_price := 22
def initial_selling_price := 38
def initial_quantity_sold := 160
def price_reduction := 3
def quantity_increase := 120
def target_profit := 3640

theorem fruit_selling_price_3640 (x : ℝ) :
  ((initial_selling_price - x - cost_price) * (initial_quantity_sold + (x / price_reduction) * quantity_increase) = target_profit) →
  x = 9 →
  initial_selling_price - x = 29 :=
by
  intro h1 h2
  sorry

end fruit_selling_price_3640_l636_636651


namespace total_distance_l636_636188

-- Define the distances as nonnegative real numbers
noncomputable def distance (a b : ℝ) : Prop := a ≥ 0 ∧ b ≥ 0

-- Define the conditions for the problem
def HomeToStore (d₁ : ℝ) : Prop := distance d₁ 50 -- from Home to Store
def StoreToPeter : ℝ := 50 -- from Store to Peter
def TimeRelation (d₁ d₂ : ℝ) (t₁ t₂ : ℝ) : Prop := 2 * t₂ = t₁ ∧ d₁ = d₂ -- Time and distance relation

-- Prove the total distance
theorem total_distance (d₁ d₂ t₁ t₂ : ℝ) (hHomeStore : HomeToStore d₁) (hStorePeter : StoreToPeter = d₂) (hTimeRel : TimeRelation d₁ d₂ t₁ t₂) : (d₁ + d₂ + d₂) = 150 := 
by 
  -- hHomeStore: distance d₁ 50 -> d₁ = 50 (by definition)
  have h₁ : d₁ = 50 := by sorry

  -- hStorePeter: d₂ = 50 (by definition)
  have h₂ : d₂ = 50 := by sorry

  -- Calc: d₁ + d₂ + d₂ = 50 + 50 + 50 = 150
  calc
    d₁ + d₂ + d₂ = 50 + 50 + 50 := by sorry
    ... = 150 := by sorry

end total_distance_l636_636188


namespace arrangements_white_followed_by_black_l636_636011

theorem arrangements_white_followed_by_black :
  ∃ (n : ℕ), n = (6! : ℕ) * (6! : ℕ) :=
begin
  use (6! : ℕ) * (6! : ℕ),
  sorry
end

end arrangements_white_followed_by_black_l636_636011


namespace vanessa_video_files_initial_l636_636722

theorem vanessa_video_files_initial (m v r d t : ℕ) (h1 : m = 13) (h2 : r = 33) (h3 : d = 10) (h4 : t = r + d) (h5 : t = m + v) : v = 30 :=
by
  sorry

end vanessa_video_files_initial_l636_636722


namespace solve_g_inverse_16_l636_636144

noncomputable def f (x : ℝ) : ℝ := log x / log 2 - 1

noncomputable def g (x : ℝ) : ℝ := (2 : ℝ) ^ (x + 1)

theorem solve_g_inverse_16 : g 3 = 16 :=
by
  sorry

end solve_g_inverse_16_l636_636144


namespace solve_for_vee_l636_636139

theorem solve_for_vee (vee : ℝ) (h : 4 * vee ^ 2 = 144) : vee = 6 ∨ vee = -6 :=
by
  -- We state that this theorem should be true for all vee and given the condition h
  sorry

end solve_for_vee_l636_636139


namespace empty_subset_singleton_l636_636985

theorem empty_subset_singleton : (∅ ⊆ ({0} : Set ℕ)) = true :=
by sorry

end empty_subset_singleton_l636_636985


namespace area_of_equilateral_triangle_inscribed_in_square_l636_636048

variables {a : ℝ}

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  a^2 * (2 * Real.sqrt 3 - 3)

theorem area_of_equilateral_triangle_inscribed_in_square (a : ℝ) :
  equilateral_triangle_area a = a^2 * (2 * Real.sqrt 3 - 3) :=
by sorry

end area_of_equilateral_triangle_inscribed_in_square_l636_636048


namespace probability_of_selecting_cooking_l636_636790

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636790


namespace line_equation_l636_636305

-- Define the points A and M
structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 3 1
def M := Point.mk 4 (-3)

def symmetric_point (A M : Point) : Point :=
  Point.mk (2 * M.x - A.x) (2 * M.y - A.y)

def line_through_origin (B : Point) : Prop :=
  7 * B.x + 5 * B.y = 0

theorem line_equation (B : Point) (hB : B = symmetric_point A M) : line_through_origin B :=
  by
  sorry

end line_equation_l636_636305


namespace sampling_is_systematic_l636_636972

-- Conditions
def production_line (units_per_day : ℕ) : Prop := units_per_day = 128

def sampling_inspection (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  samples_per_day = 8 ∧ inspection_time = 30 ∧ inspection_days = 7

-- Question
def sampling_method (method : String) (units_per_day : ℕ) (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  production_line units_per_day ∧ sampling_inspection samples_per_day inspection_time inspection_days → method = "systematic sampling"

-- Theorem stating the question == answer given conditions
theorem sampling_is_systematic : sampling_method "systematic sampling" 128 8 30 7 :=
by
  sorry

end sampling_is_systematic_l636_636972


namespace sugar_amount_first_week_l636_636043

theorem sugar_amount_first_week (s : ℕ → ℕ) (h : s 4 = 3) (h_rec : ∀ n, s (n + 1) = s n / 2) : s 1 = 24 :=
by
  sorry

end sugar_amount_first_week_l636_636043


namespace derivative_of_y_l636_636450

noncomputable def y (x : ℝ) : ℝ :=
  1/2 * Real.tanh x + 1/(4 * Real.sqrt 2) * Real.log ((1 + Real.sqrt 2 * Real.tanh x) / (1 - Real.sqrt 2 * Real.tanh x))

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 1/(Real.cosh x ^ 2 * (1 - Real.sinh x ^ 2)) := 
by
  sorry

end derivative_of_y_l636_636450


namespace geometric_sequence_sum_ratio_l636_636207

noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a q : ℝ) (h : a * q^2 = 8 * a * q^5) :
  (geometric_sum a q 4) / (geometric_sum a q 2) = 5 / 4 :=
by
  -- The proof will go here.
  sorry

end geometric_sequence_sum_ratio_l636_636207


namespace probability_of_selecting_cooking_l636_636851

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636851


namespace flour_weight_range_l636_636284

variable {m : ℝ}

theorem flour_weight_range (h : 25 - 0.02 ≤ m ∧ m ≤ 25 + 0.02) : 24.98 ≤ m ∧ m ≤ 25.02 :=
by
  exact h

end flour_weight_range_l636_636284


namespace parallelogram_area_correct_l636_636232

noncomputable def parallelogram_area (side1 side2 : ℝ) (angle_deg : ℝ) : ℝ :=
  let angle_rad := angle_deg * Real.pi / 180 in
  side1 * side2 * Real.sin angle_rad

theorem parallelogram_area_correct :
  parallelogram_area 10 20 100 ≈ 196.96 := by
  sorry

end parallelogram_area_correct_l636_636232


namespace inequality_proof_l636_636585

theorem inequality_proof (a b c : ℝ) (hab : a * b < 0) : 
  a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := 
by 
  sorry

end inequality_proof_l636_636585


namespace weight_first_sphere_l636_636307

open Real

noncomputable def sphere_surface_area (r : ℝ) : ℝ :=
  4 * pi * r^2

-- conditions
def is_proportional_weight_surface_area : Prop := ∀ (r1 r2 : ℝ) (W1 W2 : ℝ),
  W2 / W1 = (sphere_surface_area r2) / (sphere_surface_area r1)

def radius1 : ℝ := 0.15
def radius2 : ℝ := 0.3
def weight2 : ℝ := 32

-- target statement
theorem weight_first_sphere :
  is_proportional_weight_surface_area →
  (sphere_surface_area radius1 = 4 * pi * 0.15^2) →
  (sphere_surface_area radius2 = 4 * pi * 0.3^2) →
  ∃ W1 : ℝ, W1 = 8 :=
by 
  intros hp hsa1 hsa2
  have h1 := sphere_surface_area radius1
  have h2 := sphere_surface_area radius2
  sorry

end weight_first_sphere_l636_636307


namespace sum_first_49_terms_l636_636303

/-- The sum of the first 49 terms of the sequence \(a_n = \frac{1}{1+2+3+\cdots+n}\). -/
theorem sum_first_49_terms : 
  let a_n := λ n : ℕ, 1 / (Finset.range (n + 1)).sum id in
  (Finset.range 49).sum (λ n, a_n (n + 1)) = 49 / 25 :=
sorry

end sum_first_49_terms_l636_636303


namespace sum_of_n_values_l636_636751

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l636_636751


namespace probability_of_selecting_cooking_is_one_fourth_l636_636870

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636870


namespace maximize_expression_l636_636099

-- Given the condition
theorem maximize_expression (x y : ℝ) (h : x + y = 1) : (x^3 + 1) * (y^3 + 1) ≤ (1)^3 + 1 * (0)^3 + 1 * (0)^3 + 1 :=
sorry

end maximize_expression_l636_636099


namespace shifted_parabola_l636_636283

theorem shifted_parabola (x : ℝ) : (x ^ 2 + 1 = y) ↔ (y = x^2 + 1) :=
begin
  split;
  { intro h, assumption }
end

end shifted_parabola_l636_636283


namespace johns_percentage_increase_l636_636763

def original_amount : ℕ := 60
def new_amount : ℕ := 84

def percentage_increase (original new : ℕ) := ((new - original : ℕ) * 100) / original 

theorem johns_percentage_increase : percentage_increase original_amount new_amount = 40 :=
by
  sorry

end johns_percentage_increase_l636_636763


namespace first_term_of_geometric_series_l636_636414

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l636_636414


namespace probability_of_selecting_cooking_is_one_fourth_l636_636879

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636879


namespace total_payment_correct_l636_636997

theorem total_payment_correct 
  (bob_bill : ℝ) 
  (kate_bill : ℝ) 
  (bob_discount_rate : ℝ) 
  (kate_discount_rate : ℝ) 
  (bob_discount : ℝ := bob_bill * bob_discount_rate / 100) 
  (kate_discount : ℝ := kate_bill * kate_discount_rate / 100) 
  (bob_final_payment : ℝ := bob_bill - bob_discount) 
  (kate_final_payment : ℝ := kate_bill - kate_discount) : 
  (bob_bill = 30) → 
  (kate_bill = 25) → 
  (bob_discount_rate = 5) → 
  (kate_discount_rate = 2) → 
  (bob_final_payment + kate_final_payment = 53) :=
by
  intros
  sorry

end total_payment_correct_l636_636997


namespace main_theorem_l636_636134

noncomputable def prop_polynomial : Prop :=
  ∀ (a b c d : ℝ), let P := λ x : ℂ, x^5 + (a : ℂ)*x^4 + (b : ℂ)*x^3 + (c : ℂ)*x^2 + (d : ℂ)*x + 2019 in
    (∀ r : ℂ, P r = 0 → P (complex.I * r) = 0) →
    ∃! a b c d : ℝ, ∀ r : ℂ, P r = 0 → P (complex.I * r) = 0

theorem main_theorem : prop_polynomial := sorry

end main_theorem_l636_636134


namespace probability_cooking_is_one_fourth_l636_636860
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636860


namespace probability_of_selecting_cooking_l636_636846

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636846


namespace point_on_transformed_graph_l636_636106

theorem point_on_transformed_graph (f : ℝ → ℝ) (Hf : f 8 = 10) : 
  (∃ y, 4 * y = f (3 * 2^3) - 6 ∧ 2 + y = 3) :=
begin
  -- Assume that f(24) = 10 for simplicity in this context
  have : f (3 * 2^3) = 10,
  { exact Hf },
  use 1,
  split,
  { rw this,
    norm_num },
  norm_num,
  sorry
end

end point_on_transformed_graph_l636_636106


namespace probability_of_selecting_cooking_l636_636935

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636935


namespace sum_of_solutions_abs_eq_l636_636732

-- Define the condition on n
def abs_eq (n : ℝ) : Prop := abs (3 * n - 8) = 5

-- Statement we want to prove
theorem sum_of_solutions_abs_eq : ∑ n in { n | abs_eq n }, n = 16/3 :=
by
  sorry

end sum_of_solutions_abs_eq_l636_636732


namespace shortest_side_of_triangle_l636_636388

noncomputable def side_length (x : ℕ) := 2 * x

noncomputable def semiperimeter (a b c : ℕ) := (a + b + c) / 2

noncomputable def triangle_area (s a b c : ℕ) := (s * (s - a) * (s - b) * (s - c)) ^ (1/2)

noncomputable def inradius_formula (Δ s : ℕ) := Δ / s

theorem shortest_side_of_triangle : 
  ∀ (a b r : ℕ), a = 5 ∧ b = 9 ∧ r = 3 → 
  ∃ (x : ℕ), semiperimeter (5 + 9) (side_length x) (14 + 4) = 16 + x ∧ 
             inradius_formula (triangle_area (semiperimeter (5 + 9) (side_length x) (14 + 4)) 14 (side_length x) 18) (16 + x) = r → 
             side_length x = 12 :=
begin
  intros,
  -- proof goes here
  sorry
end

end shortest_side_of_triangle_l636_636388


namespace rook_placement_non_attacking_l636_636561

theorem rook_placement_non_attacking (n : ℕ) (w b : ℕ) : 
  w = 8 * 8 ∧ b = (8 * (8 - 1) + (8 - 1) * (8 - 1)) → 
  w * b = 3136 :=
by 
  intro h.
  cases h with hw hb.
  sorry

end rook_placement_non_attacking_l636_636561


namespace friends_cant_go_to_movies_l636_636628

theorem friends_cant_go_to_movies (total_friends : ℕ) (friends_can_go : ℕ) (H1 : total_friends = 15) (H2 : friends_can_go = 8) : (total_friends - friends_can_go) = 7 :=
by
  sorry

end friends_cant_go_to_movies_l636_636628


namespace distance_focus_to_asymptote_l636_636505

theorem distance_focus_to_asymptote (m : ℝ) (x y : ℝ) (h1 : (x^2) / 9 - (y^2) / m = 1) 
  (h2 : (Real.sqrt 14) / 3 = (Real.sqrt (9 + m)) / 3) : 
  ∃ d : ℝ, d = Real.sqrt 5 := 
by 
  sorry

end distance_focus_to_asymptote_l636_636505


namespace boxes_needed_for_loose_crayons_l636_636461

-- Definitions based on conditions
def boxes_francine : ℕ := 5
def loose_crayons_francine : ℕ := 5
def loose_crayons_friend : ℕ := 27
def total_crayons_francine : ℕ := 85
def total_boxes_needed : ℕ := 2

-- The theorem to prove
theorem boxes_needed_for_loose_crayons 
  (hf : total_crayons_francine = boxes_francine * 16 + loose_crayons_francine)
  (htotal_loose : loose_crayons_francine + loose_crayons_friend = 32)
  (hboxes : boxes_francine = 5) : 
  total_boxes_needed = 2 :=
sorry

end boxes_needed_for_loose_crayons_l636_636461


namespace no_such_quadrilateral_l636_636245

theorem no_such_quadrilateral (A B C D: ℤ × ℤ) (h1 : (distance A C) = 2 * (distance B D))
(h2 : angle_between_diagonals (A, C) (B, D) = 45)
(h3 : integer_coordinates A B C D) : false :=
sorry

end no_such_quadrilateral_l636_636245


namespace floor_expression_eq_zero_l636_636030

theorem floor_expression_eq_zero : 
  (⌊ (2015^2 / (2013 * 2014) - 2013^2 / (2014 * 2015)) ⌋) = 0 :=
by 
  sorry

end floor_expression_eq_zero_l636_636030


namespace probability_of_selecting_cooking_l636_636779

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636779


namespace employee_y_payment_l636_636346

theorem employee_y_payment (X Y : ℝ) (h1 : X + Y = 616) (h2 : X = 1.2 * Y) : Y = 280 :=
by
  sorry

end employee_y_payment_l636_636346


namespace find_number_90_l636_636545

theorem find_number_90 {x y : ℝ} (h1 : x = y + 0.11 * y) (h2 : x = 99.9) : y = 90 :=
sorry

end find_number_90_l636_636545


namespace fly_distance_from_ceiling_l636_636314

theorem fly_distance_from_ceiling (x y z : ℝ) (h1 : x = 3) (h2 : y = 5) (h3 : z = 6)
  (h_dist : (x^2 + y^2 + z^2).sqrt = 10) : z = Real.sqrt 66 :=
by
  -- initiating the coordinates of the fly
  let P := (0, 0, 0)
  -- setting the fly's position relative to P
  let fly_pos := (3, 5, z)
  -- applying the given distance condition from P
  have eq1 : 10 = (Real.sqrt (3^2 + 5^2 + z^2)) := by sorry
  -- deriving the required distance from the ceiling
  show z = Real.sqrt 66 from sorry

end fly_distance_from_ceiling_l636_636314


namespace probability_of_selecting_cooking_l636_636809

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636809


namespace smaller_two_digit_product_l636_636688

theorem smaller_two_digit_product (a b : ℕ) (h1: 10 ≤ a) (h2: a ≤ 99) (h3: 10 ≤ b) (h4: b ≤ 99) (h_prod: a * b = 4536) : a = 54 ∨ b = 54 :=
by sorry

end smaller_two_digit_product_l636_636688


namespace weight_cut_percentage_unknown_l636_636193

-- Define the initial conditions
def original_speed : ℝ := 150
def new_speed : ℝ := 205
def increase_supercharge : ℝ := original_speed * 0.3
def speed_after_supercharge : ℝ := original_speed + increase_supercharge
def increase_weight_cut : ℝ := new_speed - speed_after_supercharge

-- Theorem statement
theorem weight_cut_percentage_unknown : 
  (original_speed = 150) →
  (new_speed = 205) →
  (increase_supercharge = 150 * 0.3) →
  (speed_after_supercharge = 150 + increase_supercharge) →
  (increase_weight_cut = 205 - speed_after_supercharge) →
  increase_weight_cut = 10 →
  sorry := 
by
  intros h_orig h_new h_inc_scharge h_speed_scharge h_inc_weight h_inc_10
  sorry

end weight_cut_percentage_unknown_l636_636193


namespace minimal_sum_distances_at_center_l636_636341

noncomputable def sum_distances_to_vertices (tetrahedron : set Point) (X : Point) : Real :=
  d(X, A) + d(X, B) + d(X, C) + d(X, D)

theorem minimal_sum_distances_at_center (tetrahedron : RegularTetrahedron) (X : Point) :
  sum_distances_to_vertices tetrahedron X = sum_distances_to_vertices tetrahedron (center tetrahedron)
  :=
begin
  sorry
end

end minimal_sum_distances_at_center_l636_636341


namespace count_paths_from_0_0_to_5_5_l636_636380

-- Defining the conditions on the moves.
inductive Move
| Right : Move
| Up : Move
| Diagonal : Move

def isValidMove (current next : Nat × Nat) : Bool :=
  match current, next with
  | (a, b), (a', b') => (a' = a + 1 ∧ b' = b) ∨ (a' = a ∧ b' = b + 1) ∨ (a' = a + 1 ∧ b' = b + 1)

-- Defining no right angle turns.
def noRightAngle (path : List (Nat × Nat)) : Bool :=
  ∀ i, 0 < i → i < path.length - 1 →
       let (curX, curY) := (path.get? i).getOrElse (0, 0)
       let (prevX, prevY) := (path.get? (i - 1)).getOrElse (0, 0)
       let (nextX, nextY) := (path.get? (i + 1)).getOrElse (0, 0)
       (curX ≠ prevX ∨ curX ≠ nextX) ∧ (curY ≠ prevY ∨ curY ≠ nextY)

-- The initial and final points of the path
def start : Nat × Nat := (0, 0)
def end : Nat × Nat := (5, 5)

-- Define the set of possible paths (this is a placeholder for the actual path calculation, which we skip here).
def possiblePaths : List (List (Nat × Nat)) := []

-- Main lemma stating the number of valid paths equals 83
theorem count_paths_from_0_0_to_5_5 : 
  (possiblePaths.filter (λ path => isValidMove path.head start ∧ 
                                      isValidMove path.last end ∧
                                      noRightAngle path)).length = 83 := 
  sorry

end count_paths_from_0_0_to_5_5_l636_636380


namespace probability_cooking_is_one_fourth_l636_636856
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636856


namespace range_of_angle_A_l636_636176

theorem range_of_angle_A (a b : ℝ) (A : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) 
  (h_triangle : 0 < A ∧ A ≤ Real.pi / 4) :
  (0 < A ∧ A ≤ Real.pi / 4) :=
by
  sorry

end range_of_angle_A_l636_636176


namespace imaginary_part_of_conjugate_div_z_l636_636580

noncomputable def z : ℂ := 1 + 3 * complex.I
noncomputable def conj_z : ℂ := complex.conj z

theorem imaginary_part_of_conjugate_div_z :
  complex.im (conj_z / z) = -3/5 :=
by
  -- We skip the detailed proof steps and state the goal.
  sorry

end imaginary_part_of_conjugate_div_z_l636_636580


namespace geometric_series_first_term_l636_636408

theorem geometric_series_first_term (r a s : ℝ) (h₁ : r = 1 / 4) (h₂ : s = 80) (h₃ : s = a / (1 - r)) : a = 60 :=
by
  rw [h₁] at h₃
  rw [h₂] at h₃
  norm_num at h₃
  linarith

# Examples utilized:
-- r : common ratio
-- a : first term of the series
-- s : sum of the series
-- h₁ : condition that the common ratio is 1/4
-- h₂ : condition that the sum is 80
-- h₃ : condition representing the formula for the sum of an infinite geometric series

end geometric_series_first_term_l636_636408


namespace shortest_time_constant_l636_636659

theorem shortest_time_constant 
  (m n : ℕ) (m_ge_two : m ≥ 2)
  (t : ℕ → ℝ) 
  (h : ∀ i j : ℕ, i < j → t i < t j) :
  ∃ k : ℕ, (k ≤ n ∧ k ≥ 1) → 
  (Σ i : ℕ, t i) = Σ i in finset.range (n), t i ∧ 
  k = n → 
  (Σ i in finset.range (n-1), t i) + (m - 1) * t (n-1) = 
  (Σ i in finset.range k, t i) := 
sorry

end shortest_time_constant_l636_636659


namespace sequences_of_length_23_l636_636523

-- Defining the function g and conditions
def g : ℕ → ℕ
| 3 := 1
| 4 := 1
| 5 := 1
| 6 := 2
| 7 := 3
| n := g (n - 4) + 2 * g (n - 5) + 2 * g (n - 6)

-- Main theorem stating the required property
theorem sequences_of_length_23 : g 23 = 130 :=
by
  sorry

end sequences_of_length_23_l636_636523


namespace geometric_problem_l636_636583

variables (A B C A' B' C' O : Type) [triangle A B C] [point_on_side A' B' C' A B C]
variables (AO OA' BO OB' CO OC' : ℝ)
variables (K_A K_B K_C : ℝ)
variables [concurrent AA' BB' CC' O] [sub_triangle_areas A B C O K_A K_B K_C]

theorem geometric_problem
  (hA' : A' ∈ segment B C) (hB' : B' ∈ segment A C) (hC' : C' ∈ segment A B)
  (h_concurrent : concurrent {AA', BB', CC'} O)
  (h_ratios : AO / OA' + BO / OB' + CO / OC' = 41)
  (h_area : K_A = area B O C )
  (h_area_2 : K_B = area C O A )
  (h_area_3 : K_C = area A O B ) :
  (AO / OA') * (BO / OB') * (CO / OC') = 39 :=
begin
  sorry
end

end geometric_problem_l636_636583


namespace probability_cooking_is_one_fourth_l636_636864
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636864


namespace evaluate_expression_at_x_eq_3_l636_636038

theorem evaluate_expression_at_x_eq_3 : (3 ^ 3) ^ (3 ^ 3) = 27 ^ 27 := by
  sorry

end evaluate_expression_at_x_eq_3_l636_636038


namespace cloth_total_30_days_l636_636769

-- Define conditions
def cloth_decrease := λ (d : ℚ), ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 30) → (5 + (n - 1) * d) ≥ 0
def first_day (amount : ℚ) : Prop := amount = 5
def last_day (amount : ℚ) (d : ℚ) : Prop := amount + 29 * d = 1

-- Define the sum of an arithmetic series function
def sum_arith_series (a d : ℚ) (n : ℕ) : ℚ := n * a + (n * (n - 1) * d) / 2

-- Lean 4 statement to prove the total amount of cloth woven is 90 meters
theorem cloth_total_30_days (d : ℚ) (h1 : first_day 5) (h2 : last_day 5 d) :
  sum_arith_series 5 d 30 = 90 :=
by sorry

end cloth_total_30_days_l636_636769


namespace probability_select_cooking_l636_636888

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636888


namespace triangle_area_is_sqrt3_over_4_l636_636482

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem triangle_area_is_sqrt3_over_4
  (a b c A B : ℝ)
  (h1 : A = Real.pi / 3)
  (h2 : b = 2 * a * Real.cos B)
  (h3 : c = 1)
  (h4 : B = Real.pi / 3)
  (h5 : a = 1)
  (h6 : b = 1) :
  area_of_triangle a b c A B (Real.pi - A - B) = Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_is_sqrt3_over_4_l636_636482


namespace derivative_at_pi_over_3_l636_636498

noncomputable def f (x : Real) : Real := 2 * Real.sin x + Real.sqrt 3 * Real.cos x

theorem derivative_at_pi_over_3 : (deriv f) (π / 3) = -1 / 2 := 
by
  sorry

end derivative_at_pi_over_3_l636_636498


namespace binomial_coefficient_x5_l636_636449

theorem binomial_coefficient_x5 :
  let binomial_term (r : ℕ) : ℕ := Nat.choose 7 r * (21 - 4 * r)
  35 = binomial_term 4 :=
by
  sorry

end binomial_coefficient_x5_l636_636449


namespace price_of_mixture_l636_636265

theorem price_of_mixture (P1 P2 P3 : ℝ) (Q1 Q2 Q3 : ℝ) (x : ℝ) :
  P1 = 126 ∧ P2 = 135 ∧ P3 = 175.5 ∧ Q1 = x ∧ Q2 = x ∧ Q3 = 2 * x →
  (let C_total := P1 * Q1 + P2 * Q2 + P3 * Q3 in
   let Q_total := Q1 + Q2 + Q3 in
   let P_mixture := C_total / Q_total in
   P_mixture = 153) :=
by {
  intros,
  sorry
}

end price_of_mixture_l636_636265


namespace compute_expression_l636_636263

noncomputable def g : ℝ → ℝ :=
  fun x => 
    if x = 1 then 3 else
    if x = 2 then 4 else
    if x = 3 then 6 else
    if x = 4 then 8 else
    if x = 5 then 9 else
    if x = 6 then 10 else
    0 -- placeholder for other values

noncomputable def g_inv : ℝ → ℝ :=
  fun y => 
    if y = 3 then 1 else
    if y = 4 then 2 else
    if y = 6 then 3 else
    if y = 8 then 4 else
    if y = 9 then 5 else
    if y = 10 then 6 else
    0 -- placeholder for other values

-- Prove the value of the given expression
theorem compute_expression : g(g(2)) + g(g_inv(9)) + g_inv(g_inv(6)) = 19 :=
by
  sorry

end compute_expression_l636_636263


namespace percentage_passed_all_topics_l636_636554

-- Define the constants and conditions
def total_students : ℕ := 2500
def no_pass_percentage : ℝ := 0.10
def one_topic_percentage : ℝ := 0.20
def two_topics_percentage : ℝ := 0.25
def four_topics_percentage : ℝ := 0.24
def three_topic_students : ℕ := 500
def three_topics_percentage : ℝ := (three_topic_students : ℝ) / (total_students : ℝ)

-- The proof goal
theorem percentage_passed_all_topics :
  ∃ P : ℝ, P + no_pass_percentage + one_topic_percentage + two_topics_percentage + four_topics_percentage + three_topics_percentage = 1 ∧ P = 0.01 :=
by 
  -- Sorry to indicate the proof is omitted
  sorry

end percentage_passed_all_topics_l636_636554


namespace solution_set_l636_636609

noncomputable def f : Real → Real := sorry
noncomputable def f'' : Real → Real := sorry

def odd_fn (f : Real → Real) : Prop := 
  ∀ x, f (-x) = -f x

-- Given conditions
axiom h_odd : odd_fn f
axiom h2 : f (π / 2) = 0
axiom h3 : ∀ x, 0 < x ∧ x < π → f'' x * sin x - f x * cos x < 0

theorem solution_set :
  {x : Real | f x < 2 * f (π / 6) * sin x} = 
    { x : Real | -π / 6 < x ∧ x < 0 } ∪ { x : Real | π / 6 < x ∧ x < π } :=
sorry

end solution_set_l636_636609


namespace prob_none_three_win_prob_at_least_two_not_win_l636_636957

-- Definitions for probabilities
def prob_win : ℚ := 1 / 6
def prob_not_win : ℚ := 1 - prob_win

-- Problem 1: Prove probability that none of the three students win
theorem prob_none_three_win : (prob_not_win ^ 3) = 125 / 216 := by
  sorry

-- Problem 2: Prove probability that at least two of the three students do not win
theorem prob_at_least_two_not_win : 1 - (3 * (prob_win ^ 2) * prob_not_win + prob_win ^ 3) = 25 / 27 := by
  sorry

end prob_none_three_win_prob_at_least_two_not_win_l636_636957


namespace probability_of_selecting_cooking_l636_636791

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636791


namespace probability_selecting_cooking_l636_636898

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636898


namespace find_theta_l636_636468

theorem find_theta 
  (θ : ℝ) 
  (h1 : cos (π + θ) = - (2 / 3)) 
  (h2 : θ ∈ set.Ioo (-π/2) 0) : 
  θ = -real.arccos (2 / 3) :=
sorry

end find_theta_l636_636468


namespace probability_of_selecting_cooking_l636_636954

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636954


namespace median_of_consecutive_integers_l636_636300

theorem median_of_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) (h_sum : sum_of_integers = 5^5) (h_num : num_of_integers = 25) : 
  let median := sum_of_integers / num_of_integers
  in median = 125 :=
by
  let median := sum_of_integers / num_of_integers
  have h1 : sum_of_integers = 3125 := by exact h_sum
  have h2 : num_of_integers = 25 := by exact h_num
  have h3 : median = 125 := by
    calc
      median = 3125 / 25 : by rw [h1, h2]
            ... = 125      : by norm_num
  exact h3

end median_of_consecutive_integers_l636_636300


namespace greatest_prime_factor_of_176_l636_636325

-- Define the number 176
def num : ℕ := 176

-- Define the prime factors of 176
def prime_factors := [2, 11]

-- The definition of the greatest prime factor function
def greatest_prime_factor (n : ℕ) : ℕ := (prime_factors.filter (λ x => x ∣ n)).max' sorry

-- The main theorem stating the greatest prime factor of 176
theorem greatest_prime_factor_of_176 : greatest_prime_factor num = 11 := by
  -- Proof would go here
  sorry

end greatest_prime_factor_of_176_l636_636325


namespace parabola_distance_l636_636279

theorem parabola_distance (p : ℝ) : 
  (∃ p: ℝ, y^2 = 10*x ∧ 2*p = 10) → p = 5 :=
by
  sorry

end parabola_distance_l636_636279


namespace probability_of_selecting_cooking_l636_636839

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636839


namespace more_customers_left_than_remained_l636_636980

theorem more_customers_left_than_remained (initial_customers remaining_customers : ℕ) :
  initial_customers = 25 → remaining_customers = 7 → 
  (initial_customers - remaining_customers) - remaining_customers = 11 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end more_customers_left_than_remained_l636_636980


namespace parabola_segment_limit_l636_636098

theorem parabola_segment_limit :
  (∀ n : ℕ, n ≥ 1 → let a := 2^{2*n}
                        b := -6 * 2^n
                        c := 8
                        in 
                        let sum_roots := (-b / a : ℝ)
                        let prod_roots := (c / a : ℝ)
                        let length_segment := Real.sqrt ((sum_roots)^2 - 4 * prod_roots)
                        let d_n := length_segment
                        in d_n = 1 / 2^(n-1)) →
  (∀ n : ℕ, n ≥ 1 → ∑ i in finset.range n.succ, (1 / 2^(i-1)) = 2) := by
  sorry

end parabola_segment_limit_l636_636098


namespace intersection_chord_line_eq_l636_636126

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
noncomputable def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

theorem intersection_chord_line_eq (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : 
  2 * x + y = 0 :=
sorry

end intersection_chord_line_eq_l636_636126


namespace geom_series_first_term_l636_636402

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l636_636402


namespace xiaoming_selects_cooking_probability_l636_636833

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636833


namespace find_cos_equal_l636_636451

theorem find_cos_equal : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * (real.pi / 180)) = real.cos (315 * (real.pi / 180)) ∧ n = 45 :=
by
  sorry

end find_cos_equal_l636_636451


namespace number_of_factors_l636_636522

theorem number_of_factors (K : ℕ) (hK : K = 2^4 * 3^3 * 5^2 * 7^1) : 
  ∃ n : ℕ, (∀ d e f g : ℕ, (0 ≤ d ∧ d ≤ 4) → (0 ≤ e ∧ e ≤ 3) → (0 ≤ f ∧ f ≤ 2) → (0 ≤ g ∧ g ≤ 1) → n = 120) :=
sorry

end number_of_factors_l636_636522


namespace range_of_a_l636_636110

-- Lean statement that represents the proof problem
theorem range_of_a 
  (h1 : ∀ x y : ℝ, x^2 - 2 * x + Real.log (2 * y^2 - y) = 0 → x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0)
  (h2 : ∀ b : ℝ, 2 * b^2 - b > 0) :
  (∀ a : ℝ, x^2 - 2 * x + Real.log (2 * a^2 - a) = 0 → (- (1:ℝ) / 2) < a ∧ a < 0 ∨ (1 / 2) < a ∧ a < 1) :=
sorry

end range_of_a_l636_636110


namespace trapezium_area_is_correct_l636_636447

-- Define the trapezium problem
def trapezium_area (a b h : ℝ) : ℝ := (1 / 2) * (a + b) * h

-- Constants for the problem conditions
def side1 : ℝ := 20
def side2 : ℝ := 18
def height : ℝ := 12

-- Target area calculation based on given conditions
def target_area : ℝ := 228

-- The proof statement
theorem trapezium_area_is_correct : trapezium_area side1 side2 height = target_area :=
by
  sorry

end trapezium_area_is_correct_l636_636447


namespace geometric_progression_positions_l636_636182

theorem geometric_progression_positions (u1 q : ℝ) (m n p : ℕ)
  (h27 : 27 = u1 * q ^ (m - 1))
  (h8 : 8 = u1 * q ^ (n - 1))
  (h12 : 12 = u1 * q ^ (p - 1)) :
  m = 3 * p - 2 * n :=
sorry

end geometric_progression_positions_l636_636182


namespace increased_speed_l636_636361

theorem increased_speed
  (v : ℝ) (initial_time : ℝ) (stop_time : ℝ) (additional_distance : ℝ)
  (delay : ℝ) (total_distance : ℝ) (d_new : ℝ) (d_after_stop : ℝ)
  (t_remaining : ℝ) (increased_speed : ℝ) :
  v = 32 →
  initial_time = 3 →
  stop_time = 0.25 →
  additional_distance = 28 →
  delay = 0.5 →
  total_distance = 116 →
  d_new = total_distance + additional_distance →
  t_remaining = (3 + stop_time + delay) - initial_time →
  d_after_stop = d_new - (v * initial_time) →
  increased_speed = d_after_stop / t_remaining →
  increased_speed ≈ 34.91 := by
  sorry

end increased_speed_l636_636361


namespace area_of_triangle_A2B2C2_l636_636566

noncomputable def area_DA1B1 : ℝ := 15 / 4
noncomputable def area_DA1C1 : ℝ := 10
noncomputable def area_DB1C1 : ℝ := 6
noncomputable def area_DA2B2 : ℝ := 40
noncomputable def area_DA2C2 : ℝ := 30
noncomputable def area_DB2C2 : ℝ := 50

theorem area_of_triangle_A2B2C2 : ∃ area : ℝ, 
  area = (50 * Real.sqrt 2) ∧ 
  (area_DA1B1 = 15/4 ∧ 
  area_DA1C1 = 10 ∧ 
  area_DB1C1 = 6 ∧ 
  area_DA2B2 = 40 ∧ 
  area_DA2C2 = 30 ∧ 
  area_DB2C2 = 50) := 
by
  sorry

end area_of_triangle_A2B2C2_l636_636566


namespace geom_series_first_term_l636_636399

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l636_636399


namespace rahul_share_l636_636641

theorem rahul_share :
  let total_payment := 370
  let bonus := 30
  let remaining_payment := total_payment - bonus
  let rahul_work_per_day := 1 / 3
  let rajesh_work_per_day := 1 / 2
  let ramesh_work_per_day := 1 / 4
  
  let total_work_per_day := rahul_work_per_day + rajesh_work_per_day + ramesh_work_per_day
  let rahul_share_of_work := rahul_work_per_day / total_work_per_day
  let rahul_payment := rahul_share_of_work * remaining_payment

  rahul_payment = 80 :=
by {
  sorry
}

end rahul_share_l636_636641


namespace ball_box_sequence_l636_636639

variable (Y_box R_box W_box Y_balls W_balls : ℕ)

-- Assuming the conditions
axiom condition1 : Y_box > Y_balls
axiom condition2 : R_box ≠ W_balls
axiom condition3 : W_balls < W_box

theorem ball_box_sequence :
  ∃ R W Y : string, (R, W, Y) = ("yellow", "red", "white") := 
by
  -- translate to ⟨existence⟩ of a sequence of colors
  sorry

end ball_box_sequence_l636_636639


namespace probability_select_cooking_l636_636894

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636894


namespace frog_probability_on_vertical_side_l636_636964

-- Definitions based on the conditions
def start_point : (ℕ × ℕ) := (2, 3)
def jump_length : ℕ := 2
def direction_probs : (ℚ × ℚ × ℚ × ℚ) := (1/3, 1/3, 1/6, 1/6)
def boundary : set (ℕ × ℕ) := {(0, 0), (0, 6), (6, 6), (6, 0)}

-- Definition for the point being on a vertical side of the rectangle
def is_on_vertical_side (p : ℕ × ℕ) : Prop :=
  p.1 = 0 ∨ p.1 = 6

-- Definition for the frog's jumping transition within the rectangle limits
def within_boundary (p : ℕ × ℕ) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6

-- Stating the problem
theorem frog_probability_on_vertical_side :
  let P := λ (p : ℕ × ℕ), (ℚ) in
  P (2, 3) = 8 / 9 :=
sorry

end frog_probability_on_vertical_side_l636_636964


namespace max_lateral_area_cylinder_in_sphere_l636_636492

theorem max_lateral_area_cylinder_in_sphere : 
  (∃ R : ℝ, R^2 = 5) ∧  -- Sphere's radius derived from its area
  (∀ r l : ℝ, r^2 + (l/2)^2 = 5 → 2 * real.pi * r * l ≤ 10 * real.pi) :=
begin
  sorry
end

end max_lateral_area_cylinder_in_sphere_l636_636492


namespace hypotenuse_squared_is_125_l636_636428

noncomputable def right_triangle_hypotenuse_squared (a b c : ℂ) (q r h : ℝ) : Prop :=
  (0 = (a + b + c)) ∧
  (q = (a * b + b * c + c * a)) ∧
  (r = -(a * b * c)) ∧
  (|a|^2 + |b|^2 + |c|^2 = 250) ∧
  (|c|^2 = |a|^2 + |b|^2) ∧
  (h^2 = |c|^2)

theorem hypotenuse_squared_is_125 (a b c : ℂ) (q r h : ℝ) :
  right_triangle_hypotenuse_squared a b c q r h → h^2 = 125 := by
  sorry

end hypotenuse_squared_is_125_l636_636428


namespace counterfeit_coins_weight_comparison_l636_636072

theorem counterfeit_coins_weight_comparison (coins : Finset ℕ) (h : coins.card = 2000) (counterfeit : Finset ℕ) (h_c : counterfeit.card = 2) 
  (lighter heavier : ℕ) (h_l : lighter ∈ counterfeit) (h_h : heavier ∈ counterfeit)
  (h_lighter : lighter < genuine) (h_heavier : heavier > genuine) :
  (total_weight counterfeit < 2 * genuine ∨ total_weight counterfeit = 2 * genuine ∨ total_weight counterfeit > 2 * genuine) := 
sorry

end counterfeit_coins_weight_comparison_l636_636072


namespace smallest_n_for_decimal_314_l636_636664

theorem smallest_n_for_decimal_314 (m n : ℕ) (h1 : Nat.gcd m n = 1) (h2 : m < n) (h3 : String (Nat.floor ((Real.ofNat m) / (Real.ofNat n) * 10^3)) = "314") : n = 159 := 
sorry

end smallest_n_for_decimal_314_l636_636664


namespace area_increases_l636_636170

variables (AB AC BC : ℝ)
variables (AB' AC' BC' : ℝ)

def original_triangle_valid (AB AC BC : ℝ) : Prop :=
  (AB + AC > BC) ∧ (AB + BC > AC) ∧ (AC + BC > AB)

def modified_triangle_valid (AB' AC' BC' : ℝ) : Prop :=
  (AB' + AC' > BC') ∧ (AB' + BC' > AC') ∧ (AC' + BC' > AB')

theorem area_increases (AB AC BC : ℝ) (hAB : AB = 15) (hAC : AC = 9) (hBC : BC = 12)
                       (AB' : AB' = 2 * AB) (AC' : AC' = 0.5 * AC) (BC' : BC' = 2 * BC)
                       (original_valid : original_triangle_valid AB AC BC)
                       (modified_valid : modified_triangle_valid AB' AC' BC') :
  ∃ A' > 0, A' > area ABC :=
sorry

end area_increases_l636_636170


namespace inscribed_sphere_radius_eq_one_l636_636269

-- Define the conditions of the pyramid
variables (A B C D H : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace H]
variables (AB BC CD DA HA : A → B → ℝ) (AB_is_square : ∀ x y, AB x y = 3)
variables (HA_perpendicular : ∀ x y, HA x y = 4)

-- Prove that an inscribed sphere has a radius 1
theorem inscribed_sphere_radius_eq_one :
  ∃ r : ℝ, r = 1 ∧ ∀ h, ∃ s, s = inscribed_sphere H A B C D -> r = 1 :=
sorry

end inscribed_sphere_radius_eq_one_l636_636269


namespace evaluate_expression_at_x_neg3_l636_636041

theorem evaluate_expression_at_x_neg3 :
  (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 :=
by
  sorry

end evaluate_expression_at_x_neg3_l636_636041


namespace jack_current_weight_l636_636592

-- Define the given conditions
def initial_weight : ℝ := 222
def final_weight : ℝ := 180
def duration_goal : ℝ := 45
def duration_current : ℝ := 6

-- Define the goal to prove
def current_weight : ℝ := initial_weight - (duration_current * ((initial_weight - final_weight) / duration_goal))

theorem jack_current_weight : current_weight = 216.4 := 
by
  unfold current_weight
  unfold initial_weight
  unfold final_weight
  unfold duration_goal
  unfold duration_current
  norm_num
  sorry

end jack_current_weight_l636_636592


namespace distance_A_to_B_cm_l636_636718

variable (dist: ℝ) -- distance between A and B in km
variable (vA vB: ℝ) -- speeds of A and B in km/h
variable (t_half h1 h2: ℝ) -- time variables in hours

-- Conditions from the problem:
variables (dA dB: dist > 0) (vA = 12.5) (vB = 10) 
variables (t_half = 0.5) (h1 = 1.5) (h2 = 1/4)
variables (timeA_PL = t + h2) (timeB_PL = t + h1)
variables (t = 19 / 4)

-- Distance calculation
variable (calcDistA: (vA * (timeA_PL)) = dist)
variable (calcDistB: (vB * (timeB_PL)) = dist)
variable (dist_cm: dist * 100000 = 6250000)

noncomputable def lean_problem := 
dist * 100000 = 6250000

theorem distance_A_to_B_cm:
  ∀ (dist: ℝ) (vA vB: ℝ) (t_half h1 h2: ℝ),
    vA = 12.5 → 
    vB = 10 → 
    t_half = 0.5 → 
    h1 = 1.5 →
    h2 = 1/4 →
    t = 19 / 4 →
    (12.5 * (t + 1/4) = dist) → 
    (10 * (t + 1.5) = dist) → 
    (dist * 100000 = 6250000).

-- Proof placeholder
begin
  sorry
end

end distance_A_to_B_cm_l636_636718


namespace probability_of_selecting_cooking_l636_636931

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636931


namespace probability_of_cooking_l636_636804

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636804


namespace cubic_root_equation_solution_l636_636446

theorem cubic_root_equation_solution :
  ∀ (x : ℝ), (∃ y : ℝ, y = real.cbrt x ∧ y - 4 / (y + 4) = 0) ↔ 
  (x = (-2 + 2*real.sqrt 2)^3 ∨ x = (-2 - 2*real.sqrt 2)^3) :=
by
  sorry

end cubic_root_equation_solution_l636_636446


namespace train_crossing_time_l636_636978

/-- Define basic parameters for the problem -/
def train_length : ℝ := 130
def train_speed_km_per_hr : ℝ := 45
def total_length : ℝ := 245

/-- Convert speed from km/hr to m/s -/
def train_speed_m_per_s : ℝ := (train_speed_km_per_hr * 1000) / 3600

/-- Calculate the time to cross the bridge in seconds -/
def time_to_cross_bridge (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Prove that the time to cross the bridge is 19.6 seconds -/
theorem train_crossing_time : time_to_cross_bridge total_length train_speed_m_per_s = 19.6 := by
  /-- We are given the lengths and speed conversion already -/
  sorry

end train_crossing_time_l636_636978


namespace rectangle_side_deficit_l636_636163

theorem rectangle_side_deficit (L W : ℝ) (p : ℝ)
  (h1 : 1.05 * L * (1 - p) * W - L * W = 0.8 / 100 * L * W)
  (h2 : 0 < L) (h3 : 0 < W) : p = 0.04 :=
by {
  sorry
}

end rectangle_side_deficit_l636_636163


namespace probability_of_selecting_cooking_l636_636785

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636785


namespace simplify_fraction_l636_636254

theorem simplify_fraction (x : ℝ) :
  (2 + 3 * sin x - 4 * cos x) / (2 + 3 * sin x + 2 * cos x) =
  (-1 + 3 * sin (x / 2) * cos (x / 2) + 4 * sin (x / 2) ^ 2) /
  (2 + 3 * sin (x / 2) * cos (x / 2) - 2 * sin (x / 2) ^ 2) :=
by sorry

end simplify_fraction_l636_636254


namespace first_term_of_geometric_series_l636_636412

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l636_636412


namespace amount_after_2_years_l636_636760

noncomputable def amount_after_n_years (present_value : ℝ) (rate_of_increase : ℝ) (years : ℕ) : ℝ :=
  present_value * (1 + rate_of_increase)^years

theorem amount_after_2_years :
  amount_after_n_years 6400 (1/8) 2 = 8100 :=
by
  sorry

end amount_after_2_years_l636_636760


namespace find_y_coordinate_l636_636969

-- Definition of line passing through a point with slope
def point_slope_line (x1 y1 m: ℝ): (ℝ → ℝ) := λ x, m * (x - x1) + y1

-- Hypotheses
variables {m x1 x_intercept y_intercept : ℝ}
hypothesis h1 : x1 = 3
hypothesis h2 : m = 2
hypothesis h3 : x_intercept = 1
hypothesis h4 : y_intercept = (point_slope_line x_intercept 0 m)

-- Theorem statement
theorem find_y_coordinate : ∃ y1: ℝ, point_slope_line x1 y1 m x_intercept = 0 :=
begin
  use 4,
  subst h1,
  subst h2,
  subst h3,
  subst h4,
  sorry
end

end find_y_coordinate_l636_636969


namespace problem_conditions_l636_636620

-- Declare the function
def f (a b c x : ℝ) : ℝ := a * x + b * x - c * x

-- Declare the main theorem with given conditions and conclusions
theorem problem_conditions (a b c : ℝ) : 
  c > a -> c > b -> a > 0 -> b > 0 -> 
  (∀ (x : ℝ), (ax, bx, cx) cannot form the sides of a triangle ↔ conclusion 2) ∧ 
  (obtuse_triangle a b c ↔ ∃ (x : ℝ), 1 < x < 2 ∧ f a b c x = 0) := 
by
  sorry

end problem_conditions_l636_636620


namespace convert_255_to_base8_l636_636433

-- Define the conversion function from base 10 to base 8
def base10_to_base8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let r2 := n % 64
  let d1 := r2 / 8
  let r1 := r2 % 8
  d2 * 100 + d1 * 10 + r1

-- Define the specific number and base in the conditions
def num10 : ℕ := 255
def base8_result : ℕ := 377

-- The theorem stating the proof problem
theorem convert_255_to_base8 : base10_to_base8 num10 = base8_result :=
by
  -- You would provide the proof steps here
  sorry

end convert_255_to_base8_l636_636433


namespace evaluate_expression_at_neg3_l636_636040

theorem evaluate_expression_at_neg3 : 
  (let x := -3 in (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)) = -26 :=
by
  simp only
  intro x
  sorry

end evaluate_expression_at_neg3_l636_636040


namespace sum_of_n_values_l636_636736

theorem sum_of_n_values : sum (λ n, |3 * n - 8| = 5) = 16/3 := 
by
  sorry

end sum_of_n_values_l636_636736


namespace find_lighter_ball_min_weighings_l636_636311

noncomputable def min_weighings_to_find_lighter_ball (balls : Fin 9 → ℕ) : ℕ :=
  2

-- Given: 9 balls, where 8 weigh 10 grams and 1 weighs 9 grams, and a balance scale.
theorem find_lighter_ball_min_weighings :
  (∃ i : Fin 9, balls i = 9 ∧ (∀ j : Fin 9, j ≠ i → balls j = 10)) 
  → min_weighings_to_find_lighter_ball balls = 2 :=
by
  intros
  sorry

end find_lighter_ball_min_weighings_l636_636311


namespace circle_equation_l636_636491

theorem circle_equation {C : Type*} [metric_space C] [normed_group C] 
(centre_x centre_y radius : ℝ) (tan_line_x tan_line_y tan_line_c : ℝ)
(h1 : (centre_x, centre_y) = (-1, 0))
(h2 : tan_line_x + tan_line_y + tan_line_c = 0)
(h3 : radius = real.sqrt 2) :
(x + 1)^2 + y^2 = 2 := by 
  sorry

end circle_equation_l636_636491


namespace purely_imaginary_a_l636_636214

theorem purely_imaginary_a (a : ℝ) (h : (a^3 - a) = 0) (h2 : (a / (1 - a)) ≠ 0) : a = -1 := 
sorry

end purely_imaginary_a_l636_636214


namespace divisibility_expression_l636_636646

variable {R : Type*} [CommRing R] (x a b : R)

theorem divisibility_expression :
  ∃ k : R, (x + a + b) ^ 3 - x ^ 3 - a ^ 3 - b ^ 3 = (x + a) * (x + b) * k :=
sorry

end divisibility_expression_l636_636646


namespace imaginary_part_of_z2_div_z1_l636_636619

namespace ComplexNumbersProof

open Complex

def z1 := 1 - 2 * I
def z2 := -1 - 2 * I

lemma symmetry_about_imaginary_axis (a b : ℝ) (z1 z2 : ℂ) (h : z1 = a + b * I) (h' : z2 = -a + b * I) :
  z2 = -a - b * I :=
by {
  simp only [h, h'],
  exact h',
}

theorem imaginary_part_of_z2_div_z1 : im (z2 / z1) = -4 / 5 :=
by {
  have z1_eq : z1 = 1 - 2 * I := by simp [z1],
  have z2_eq : z2 = -1 - 2 * I := by simp [z2],
  rw [z2_eq, z1_eq],
  simp,
  sorry -- skip detailed algebraic manipulation for simplicity
}

end ComplexNumbersProof

end imaginary_part_of_z2_div_z1_l636_636619


namespace exists_nonzero_integers_bound_l636_636617

theorem exists_nonzero_integers_bound
  (n : ℕ)
  (x : fin n → ℝ)
  (hx : ∑ i, (x i) ^ 2 = 1)
  (k : ℤ)
  (hk : k ≥ 2) :
  ∃ (a : fin n → ℤ), (∀ i, 0 < |a i| ∧ |a i| ≤ k - 1) ∧ |∑ i, a i * x i| ≤ (k - 1) * real.sqrt n / (k^n - 1) :=
sorry

end exists_nonzero_integers_bound_l636_636617


namespace minimum_number_of_odd_integers_among_six_l636_636318

theorem minimum_number_of_odd_integers_among_six : 
  ∀ (x y a b m n : ℤ), 
    x + y = 28 →
    x + y + a + b = 45 →
    x + y + a + b + m + n = 63 →
    ∃ (odd_count : ℕ), odd_count = 1 :=
by sorry

end minimum_number_of_odd_integers_among_six_l636_636318


namespace ellipse_eccentricity_circle_inside_ellipse_l636_636484

variables {x y : ℝ}

def ellipse_C (x y : ℝ) := x^2 / 4 + y^2 = 1
def circle_D (x y : ℝ) := (x + 1)^2 + y^2 = 1 / 4

theorem ellipse_eccentricity (x y : ℝ):
  ellipse_C x y → 
  let a := 2 in 
  let b := 1 in 
  let c := Real.sqrt (a^2 - b^2) in 
  let e := c / a in 
  e = Real.sqrt 3 / 2 :=
sorry

theorem circle_inside_ellipse (x y : ℝ):
  ellipse_C x y → circle_D x y → 
  ∃ x y, circle_D x y ∧ ellipse_C x y :=
sorry

end ellipse_eccentricity_circle_inside_ellipse_l636_636484


namespace probability_select_cooking_l636_636889

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636889


namespace ellipse_eccentricity_circle_inside_ellipse_l636_636483

variables {x y : ℝ}

def ellipse_C (x y : ℝ) := x^2 / 4 + y^2 = 1
def circle_D (x y : ℝ) := (x + 1)^2 + y^2 = 1 / 4

theorem ellipse_eccentricity (x y : ℝ):
  ellipse_C x y → 
  let a := 2 in 
  let b := 1 in 
  let c := Real.sqrt (a^2 - b^2) in 
  let e := c / a in 
  e = Real.sqrt 3 / 2 :=
sorry

theorem circle_inside_ellipse (x y : ℝ):
  ellipse_C x y → circle_D x y → 
  ∃ x y, circle_D x y ∧ ellipse_C x y :=
sorry

end ellipse_eccentricity_circle_inside_ellipse_l636_636483


namespace units_digit_of_quotient_l636_636439

theorem units_digit_of_quotient 
  (h1 : ∀ n : ℕ, n % 7 = 0 ↔ (7 ∣ n))
  : Nat.digits 10 (2 ^ 2023 + 3 ^ 2023) % 7 = 0 :=
by
  sorry

end units_digit_of_quotient_l636_636439


namespace probability_of_selecting_cooking_l636_636928

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636928


namespace selected_numbers_satisfy_conditions_l636_636249

theorem selected_numbers_satisfy_conditions :
  ∃ (nums : Finset ℕ), 
  nums = {6, 34, 35, 51, 55, 77} ∧
  (∀ (a b c : ℕ), a ∈ nums → b ∈ nums → c ∈ nums → a ≠ b → a ≠ c → b ≠ c → 
    gcd a b = 1 ∨ gcd b c = 1 ∨ gcd c a = 1) ∧
  (∀ (x y z : ℕ), x ∈ nums → y ∈ nums → z ∈ nums → x ≠ y → x ≠ z → y ≠ z → 
    gcd x y ≠ 1 ∨ gcd y z ≠ 1 ∨ gcd z x ≠ 1) := 
sorry

end selected_numbers_satisfy_conditions_l636_636249


namespace geometric_series_first_term_l636_636410

theorem geometric_series_first_term (r a s : ℝ) (h₁ : r = 1 / 4) (h₂ : s = 80) (h₃ : s = a / (1 - r)) : a = 60 :=
by
  rw [h₁] at h₃
  rw [h₂] at h₃
  norm_num at h₃
  linarith

# Examples utilized:
-- r : common ratio
-- a : first term of the series
-- s : sum of the series
-- h₁ : condition that the common ratio is 1/4
-- h₂ : condition that the sum is 80
-- h₃ : condition representing the formula for the sum of an infinite geometric series

end geometric_series_first_term_l636_636410


namespace curveC_to_rect_line_l1_equations_l636_636122

-- Define the curve C in polar coordinates
def curveC (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Convert the curve C to rectangular coordinates
def curveC_rect (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Define the initial line l in parametric form
def line_l (t : ℝ) : (ℝ × ℝ) := (1 + t, 2 - t)

-- Condition for the line l1 which is parallel to l with specific chord length
def is_parallel (x y : ℝ) (b : ℝ) : Prop := x + y + b = 0
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
def chord_length_condition (d : ℝ) (chord_length : ℝ) : Prop := d = Real.sqrt (1^2 - (chord_length / 2)^2)

-- Proving the Cartesian equations 
theorem curveC_to_rect : ∀ x y : ℝ, curveC_rect x y ↔ (∃ θ : ℝ, (x = 2 * Real.cos θ ∧ y = 2 * Real.sin θ)) :=
sorry

theorem line_l1_equations : ∀ b : ℝ, 
  (is_parallel 1 1 b ∧ chord_length_condition (distance 0 1 1 1) (Real.sqrt 3)) ↔ 
  (b = -1 + (Real.sqrt 2) / 2 ∨ b = -1 - (Real.sqrt 2) / 2) :=
sorry 

end curveC_to_rect_line_l1_equations_l636_636122


namespace square_area_l636_636012

theorem square_area (x : ℝ) (side_length : ℝ) 
  (h1_side_length : side_length = 5 * x - 10)
  (h2_side_length : side_length = 3 * (x + 4)) :
  side_length ^ 2 = 2025 :=
by
  sorry

end square_area_l636_636012


namespace right_triangle_345_no_right_triangle_others_l636_636336

theorem right_triangle_345 :
  ∃ (a b c : ℕ), (a, b, c) = (3, 4, 5) ∧ a * a + b * b = c * c :=
by
  use 3, 4, 5
  split
  · refl
  sorry

theorem no_right_triangle_others :
  ¬(∃ (a b c : ℕ), (a, b, c) = (1, 2, 3) ∧ a * a + b * b = c * c) ∧
  ¬(∃ (a b c : ℕ), (a, b, c) = (2, 3, 4) ∧ a * a + b * b = c * c) ∧
  ¬(∃ (a b c : ℕ), (a, b, c) = (1, 2, 3) ∧ a * a + b * b = c * c) :=
by
  split
  all_goals { try {split}; sorry }

end right_triangle_345_no_right_triangle_others_l636_636336


namespace geom_series_first_term_l636_636398

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l636_636398


namespace probability_selecting_cooking_l636_636899

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636899


namespace total_profit_equals_254000_l636_636981

-- Definitions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 6000
def investment_D : ℕ := 10000

def time_A : ℕ := 12
def time_B : ℕ := 8
def time_C : ℕ := 6
def time_D : ℕ := 9

def capital_months (investment : ℕ) (time : ℕ) : ℕ := investment * time

-- Given conditions
def A_capital_months := capital_months investment_A time_A
def B_capital_months := capital_months investment_B time_B
def C_capital_months := capital_months investment_C time_C
def D_capital_months := capital_months investment_D time_D

def total_capital_months : ℕ := A_capital_months + B_capital_months + C_capital_months + D_capital_months

def C_profit : ℕ := 36000

-- Proportion equation
def total_profit (C_capital_months : ℕ) (total_capital_months : ℕ) (C_profit : ℕ) : ℕ :=
  (C_profit * total_capital_months) / C_capital_months

-- Theorem statement
theorem total_profit_equals_254000 : total_profit C_capital_months total_capital_months C_profit = 254000 := by
  sorry

end total_profit_equals_254000_l636_636981


namespace old_books_to_reread_l636_636020

/-- Brianna problem -/
def total_books_needed : ℕ := 2 * 12

def books_given_as_gift : ℕ := 6

def books_bought : ℕ := 8

def books_borrowed : ℕ := books_bought - 2

def total_new_books : ℕ := books_given_as_gift + books_bought + books_borrowed

theorem old_books_to_reread : 
  let num_old_books := total_books_needed - total_new_books in
  num_old_books = 4 :=
by
  sorry

end old_books_to_reread_l636_636020


namespace common_external_tangent_b_l636_636717

def circle1_center := (1, 3)
def circle1_radius := 3
def circle2_center := (10, 6)
def circle2_radius := 7

theorem common_external_tangent_b :
  ∃ (b : ℝ), ∀ (m : ℝ), m = 3 / 4 ∧ b = 9 / 4 := sorry

end common_external_tangent_b_l636_636717


namespace probability_select_cooking_l636_636893

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636893


namespace tetrahedron_area_relation_l636_636165

theorem tetrahedron_area_relation (A B C D : Point) 
  (hAD_AB : AD > AB) 
  (hAD_perp_AB : AD ⟂ AB) 
  (hAD_perp_AC : AD ⟂ AC) 
  (hAngle_BAC : ∠ BAC = π / 3) 
  (S1 S2 S3 S4 : ℝ) 
  (hAreas : S1 + S2 = S3 + S4) 
  : (S3 / S1 + S3 / S2) = 1.5 :=
sorry

end tetrahedron_area_relation_l636_636165


namespace problem_l636_636479

def sequence_a (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, a n + 1 = a (n + 1)) ∧ (∑ n in Finset.range 5, a n = 15)

def sequence_b (b : ℕ → ℕ) : Prop :=
  (b 1 = 2) ∧ (4 * b 2 + b 4 = 8 * b 3)

theorem problem (a b : ℕ → ℕ) (n : ℕ) (hn : n ≠ 0) :
  (sequence_a a) →
  (sequence_b b) →
  (∀ n, a n = n) ∧
  (∀ n, b n = 2 ^ n) ∧
  ((∑ i in Finset.range n, a i * b (n + 1 - i) = 2 ^ (n + 2) - 2 * n - 4)) ∧
  ((∑ i in Finset.range (2 * n - 1), (-1) ^ (i + 1) * (a i) ^ 2 = 2 * n ^ 2 - n) := 
begin
  sorry
end

end problem_l636_636479


namespace percent_difference_l636_636761

variable (w e y z : ℝ)

-- Definitions based on the given conditions
def condition1 : Prop := w = 0.60 * e
def condition2 : Prop := e = 0.60 * y
def condition3 : Prop := z = 0.54 * y

-- Statement of the theorem to prove
theorem percent_difference (h1 : condition1 w e) (h2 : condition2 e y) (h3 : condition3 z y) : 
  (z - w) / w * 100 = 50 := 
by
  sorry

end percent_difference_l636_636761


namespace min_distance_origin_to_line_l636_636166

noncomputable def distance_from_origin_to_line(A B C : ℝ) : ℝ :=
  let d := |A * 0 + B * 0 + C| / (Real.sqrt (A^2 + B^2))
  d

theorem min_distance_origin_to_line : distance_from_origin_to_line 1 1 (-4) = 2 * Real.sqrt 2 := by 
  sorry

end min_distance_origin_to_line_l636_636166


namespace sum_of_perfect_square_g_l636_636217

noncomputable def P (g : ℕ) : ℕ :=
  g^4 + g^3 + g^2 + g + 1

theorem sum_of_perfect_square_g :
  (∑ g in (Finset.range 1000).filter (λ g, ∃ k : ℕ, P g = k^2), id g) = 3 :=
by {
  sorry
}

end sum_of_perfect_square_g_l636_636217


namespace infinite_prime_divisors_l636_636080

theorem infinite_prime_divisors (n : ℕ) (a : fin n → ℕ) (h : ∀ i, 1 < a i) :
  { p : ℕ | ∃ k : ℕ, prime p ∧ p ∣ (finset.univ.sum (λ i, (a i) ^ k)) }.infinite :=
sorry

end infinite_prime_divisors_l636_636080


namespace intersect_lines_single_point_l636_636076

open Real

-- Define a structure for the problem setup.
structure CircleSetup (n : ℕ) where
  points : Fin n → ℝ × ℝ  -- n points on the circle represented as coordinates (x, y)
  center : ℝ × ℝ          -- center of the circle

-- Define the main theorem to be proved.
theorem intersect_lines_single_point
  (n : ℕ)
  (h_n : 2 < n)  -- n is greater than 2
  (cs : CircleSetup n)
  (M1 : Fin (n - 2) → ℝ × ℝ)  -- center of mass of (n-2) points
  (K : ℝ × ℝ)  -- midpoint of the chord connecting the remaining two points
  (perpendicular_condition : ∀ M1, line_through_center_perpendicular cs.center M1 K) :
  ∃ P : ℝ × ℝ, ∀ M1 : Fin (n - 2) → ℝ × ℝ, (line_through_center_perpendicular cs.center M1 K) → 
  intersect_at_single_point M1 P :=
by
  sorry

end intersect_lines_single_point_l636_636076


namespace find_ab_plus_a_plus_b_l636_636210

-- Define the polynomial
def quartic_poly (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 - 6*x - 1

-- Define the roots conditions
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

-- State the proof problem
theorem find_ab_plus_a_plus_b :
  ∃ a b : ℝ,
    is_root quartic_poly a ∧
    is_root quartic_poly b ∧
    ab = a * b ∧
    a_plus_b = a + b ∧
    ab + a_plus_b = 4 :=
by sorry

end find_ab_plus_a_plus_b_l636_636210


namespace probability_of_selecting_cooking_l636_636814

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636814


namespace probability_of_selecting_cooking_l636_636849

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636849


namespace sum_reciprocals_lt_3_l636_636349

noncomputable def A : Set ℕ := { x | ∀ d ∈ x.digits 10, d ≠ 0 ∧ d ≠ 1 ∧ d ≠ 2 ∧ d ≠ 6 }

theorem sum_reciprocals_lt_3 : (∑ x in A, 1 / (x : ℝ)) < 3 :=
sorry

end sum_reciprocals_lt_3_l636_636349


namespace fraction_of_leo_stickers_given_to_nina_l636_636421

theorem fraction_of_leo_stickers_given_to_nina :
  ∀ (o : ℕ),
  let leo := 12 * o in
  let nina := 3 * o in
  let oli := o in
  let max := o in
  let total_stickers := leo + nina + oli + max in
  let equal_share := total_stickers / 4 in
  let stickers_nina_needs_from_leo := (equal_share - nina) in
  (stickers_nina_needs_from_leo : ℚ) / (leo : ℚ) = 5 / 48 :=
by
  sorry

end fraction_of_leo_stickers_given_to_nina_l636_636421


namespace find_f_neg3_l636_636115

-- Define the functions and the necessary conditions
def f (x : ℝ) : ℝ := a * (Real.sin x) ^ 3 + b * Real.tan x + 1

variables (a b : ℝ) (h : f 3 = 6)

theorem find_f_neg3 : f (-3) = -4 := 
by
  sorry

end find_f_neg3_l636_636115


namespace geometric_series_first_term_l636_636411

theorem geometric_series_first_term (r a s : ℝ) (h₁ : r = 1 / 4) (h₂ : s = 80) (h₃ : s = a / (1 - r)) : a = 60 :=
by
  rw [h₁] at h₃
  rw [h₂] at h₃
  norm_num at h₃
  linarith

# Examples utilized:
-- r : common ratio
-- a : first term of the series
-- s : sum of the series
-- h₁ : condition that the common ratio is 1/4
-- h₂ : condition that the sum is 80
-- h₃ : condition representing the formula for the sum of an infinite geometric series

end geometric_series_first_term_l636_636411


namespace angle_between_a_and_b_is_pi_over_4_l636_636130

noncomputable def angle_between_vectors : ℝ :=
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (1, 2)
  let dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  let norm (x : ℝ × ℝ) : ℝ := real.sqrt (x.1 ^ 2 + x.2 ^ 2)
  let cosine_angle := dot_product a b / (norm a * norm b)
  real.acos cosine_angle

theorem angle_between_a_and_b_is_pi_over_4 :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (1, 2)
  (-1) * (1) + 3 * 2 - (real.sqrt (1 + 9)) * (real.sqrt (1 + 4)) = real.cos (real.pi / 4) := sorry

end angle_between_a_and_b_is_pi_over_4_l636_636130


namespace probability_cooking_selected_l636_636913

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636913


namespace sallys_woodworking_llc_reimbursement_l636_636227

/-
Conditions:
1. Remy paid $20,700 for 150 pieces of furniture.
2. The cost of a piece of furniture is $134.
-/
def reimbursement_amount (pieces_paid : ℕ) (total_paid : ℕ) (price_per_piece : ℕ) : ℕ :=
  total_paid - (pieces_paid * price_per_piece)

theorem sallys_woodworking_llc_reimbursement :
  reimbursement_amount 150 20700 134 = 600 :=
by 
  sorry

end sallys_woodworking_llc_reimbursement_l636_636227


namespace find_vector_decomposition_l636_636564

variables {S A B C M P : Type}
variables [AddGroup S] [AddGroup A] [AddGroup B] [AddGroup C]
variables (AS AC AB : S →+ C)
variables (SC : S →+ C)

-- Conditions
def pyramid_equilateral (S A B C : Type) : Prop := sorry
def centroid (M : A) (ABC : (A × B × C)) : Prop := sorry
def midpoint (P : A) (SC : (S × C)) : Prop := sorry

-- Problem
theorem find_vector_decomposition
  (h1 : pyramid_equilateral S A B C)
  (h2 : centroid M (A, B, C))
  (h3 : midpoint P (S, C)) :
  ∃ k1 k2 k3 : ℚ, 
  ((k1 • (AC : S)) - (k2 • (AB : S)) + (k3 • (AS : S))) = 
  (1/6 • (AC : S) - 1/3 • (AB : S) + 1/2 • (AS : S)) :=
sorry

end find_vector_decomposition_l636_636564


namespace transform_correct_l636_636337

variables (a b m : ℝ)

theorem transform_correct (h : a * (m^2 + 1) = b * (m^2 + 1)) : a = b :=
by
  have h1 : m^2 + 1 ≠ 0 :=
  begin
    have h2 : m^2 ≥ 0 := by apply pow_two_nonneg,
    have h3 : m^2 + 1 > 0 := by linarith,
    exact ne_of_gt h3,
  end,
  calc
    a = (a * (m^2 + 1)) / (m^2 + 1) : by rw [div_self h1, mul_div_cancel_left _ h1]
    ... = (b * (m^2 + 1)) / (m^2 + 1) : by rw h
    ... = b : by rw [div_self h1, mul_div_cancel_left _ h1]

end transform_correct_l636_636337


namespace volume_of_cube_for_tetrahedron_l636_636195

theorem volume_of_cube_for_tetrahedron (h : ℝ) (b1 b2 : ℝ) (V : ℝ) 
  (h_condition : h = 15) (b1_condition : b1 = 8) (b2_condition : b2 = 12)
  (V_condition : V = 3375) : 
  V = (max h (max b1 b2)) ^ 3 := by
  -- To illustrate the mathematical context and avoid concrete steps,
  -- sorry provides the completion of the logical binding to the correct answer
  sorry

end volume_of_cube_for_tetrahedron_l636_636195


namespace lambda_range_l636_636508

theorem lambda_range (λ : ℝ) 
  (h₁ : ∃ n : ℕ, n ≥ 1 ∧ n ≤ 100 ∧ (min (n^2 - (6 + 2 * λ) * n + 2014) = (6 if n = 6 then a₆ else 7 if n = 7 then a₇))) :
  2.5 ≤ λ ∧ λ ≤ 4.5 :=
by
  sorry

end lambda_range_l636_636508


namespace sunzi_wood_problem_l636_636579

theorem sunzi_wood_problem (x y : ℝ) (h1 : y - x = 4.5) (h2 : y / 2 = x - 1) :
  y - x = 4.5 ∧ y / 2 = x - 1 :=
by {
    exact ⟨h1, h2⟩,
    sorry -- Proof is not required as per the instruction
}

end sunzi_wood_problem_l636_636579


namespace trigonometric_identity_l636_636256

theorem trigonometric_identity :
  (tan 20 + tan 30 + tan 40 + tan 60) / sin 80 =
  2 * ((cos 40 / (sqrt 3 * cos 10 * cos 20)) + (2 / cos 40)) :=
by
  sorry

end trigonometric_identity_l636_636256


namespace circle_has_max_area_l636_636241

theorem circle_has_max_area (P : ℝ) (S : ℝ) :
  (∀ (n : ℕ) (hn : n ≥ 3), ∀ (Sn : ℝ), (P^2 / Sn = 4*n*tan(Real.pi / n) → S > Sn))
  → (P > 0)
  → (S < (P^2 / (4*Real.pi))) :=
by
  intros h1 h2
  sorry

end circle_has_max_area_l636_636241


namespace polynomial_abc_value_l636_636542

theorem polynomial_abc_value (a b c : ℝ) (h : a * (x^2) + b * x + c = (x - 1) * (x - 2)) : a * b * c = -6 :=
by
  sorry

end polynomial_abc_value_l636_636542


namespace inverse_trig_expression_eval_l636_636703

theorem inverse_trig_expression_eval :
  let a := Real.arcsin (Real.sqrt 3 / 2)
  let b := Real.arccos (-1 / 2)
  let c := Real.arctan (- Real.sqrt 3)
  a = Real.pi / 3 → b = 2 * Real.pi / 3 → c = -Real.pi / 3 →
  (a + b) / c = -3 :=
by
  intros a_def b_def c_def
  rw [a_def, b_def, c_def]
  have h1 : (Real.pi / 3 + 2 * Real.pi / 3) = Real.pi := by norm_num
  rw h1
  have h2 : Real.pi / (- Real.pi / 3) = -3 := by rw [Real.div_of_mul_inv, Real.mul_comm]; linarith
  exact h2

end inverse_trig_expression_eval_l636_636703


namespace cheese_arrangement_count_l636_636323

theorem cheese_arrangement_count :
  ∃ (count : ℕ), count = 234 :=
by
  let k := [0, 1, 2, 3, 4]
  let arrangements := List.map (fun k => 
    match k with
    | 0 => 2
    | 1 => 2
    | 2 => 32
    | 3 => 98
    | 4 => 100
    | _ => 0) k
  let count := List.sum arrangements
  exact ⟨count, rfl⟩

#eval cheese_arrangement_count -- to check if it evaluates to the expected result

end cheese_arrangement_count_l636_636323


namespace geom_series_first_term_l636_636397

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l636_636397


namespace cone_volume_in_liters_l636_636754

theorem cone_volume_in_liters (d h : ℝ) (pi : ℝ) (liters_conversion : ℝ) :
  d = 12 → h = 10 → liters_conversion = 1000 → (1/3) * pi * (d/2)^2 * h * (1 / liters_conversion) = 0.12 * pi :=
by
  intros hd hh hc
  sorry

end cone_volume_in_liters_l636_636754


namespace sum_of_solutions_of_absolute_value_l636_636727

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l636_636727


namespace valid_interval_for_a_l636_636430

theorem valid_interval_for_a (a : ℝ) :
  (6 - 3 * a > 0) ∧ (a > 0) ∧ (3 * a^2 + a - 2 ≥ 0) ↔ (2 / 3 ≤ a ∧ a < 2 ∧ a ≠ 5 / 3) :=
by
  sorry

end valid_interval_for_a_l636_636430


namespace sum_of_solutions_abs_eq_l636_636731

-- Define the condition on n
def abs_eq (n : ℝ) : Prop := abs (3 * n - 8) = 5

-- Statement we want to prove
theorem sum_of_solutions_abs_eq : ∑ n in { n | abs_eq n }, n = 16/3 :=
by
  sorry

end sum_of_solutions_abs_eq_l636_636731


namespace xizi_set_sum_l636_636511
open Set

def I : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2020}

def W : Set ℕ := {w | ∃ a b ∈ I, w = (a + b) + a * b} ∩ I

def Y : Set ℕ := {y | ∃ a b ∈ I, y = (a + b) * a * b} ∩ I

def X : Set ℕ := W ∩ Y

theorem xizi_set_sum : I.nonempty →
  (∀ w i j, (1 ≤ i ∧ i ≤ 2020) → (1 ≤ j ∧ j ≤ 2020) → w = (i + j) + i * j → w ∈ I) →
  (∀ y i j, (1 ≤ i ∧ i ≤ 2020) → (1 ≤ j ∧ j ≤ 2020) → y = (i + j) * i * j → y ∈ I) →
  ∃ minX maxX ∈ X, minX + maxX = 2020 :=
by
  sorry

end xizi_set_sum_l636_636511


namespace probability_of_selecting_cooking_is_one_fourth_l636_636873

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636873


namespace cost_difference_l636_636196

theorem cost_difference (S : ℕ) (h1 : 15 + S = 24) : 15 - S = 6 :=
by
  sorry

end cost_difference_l636_636196


namespace brianna_books_l636_636018

theorem brianna_books :
  ∀ (books_per_month : ℕ) (given_books : ℕ) (bought_books : ℕ) (borrowed_books : ℕ) (total_books_needed : ℕ),
    (books_per_month = 2) →
    (given_books = 6) →
    (bought_books = 8) →
    (borrowed_books = bought_books - 2) →
    (total_books_needed = 12 * books_per_month) →
    (total_books_needed - (given_books + bought_books + borrowed_books)) = 4 :=
by
  intros
  sorry

end brianna_books_l636_636018


namespace findCircleAndLineEquations_l636_636028

noncomputable def circleEquationThroughPoints (A B C : (ℝ × ℝ)) : Prop :=
  ∃ (h : ℝ) (k : ℝ) (r : ℝ), 
    (A.1 - h)^2 + (A.2 - k)^2 = r^2 ∧
    (B.1 - h)^2 + (B.2 - k)^2 = r^2 ∧
    (C.1 - h)^2 + (C.2 - k)^2 = r^2 ∧
    (h = 3 ∧ k = -1 ∧ r = 5)

noncomputable def lineEquationOfChord (center : ℝ × ℝ) (midpoint : ℝ × ℝ) : Prop :=
  ∃ m b, (b = midpoint.2 - m * midpoint.1) ∧ (m = 1/2) ∧ (b = 1/2)

theorem findCircleAndLineEquations :
  let A := (-1, 2) : (ℝ × ℝ)
  let B := (6, 3)  : (ℝ × ℝ)
  let C := (3, 4)  : (ℝ × ℝ)
  let M := (3, -1) : (ℝ × ℝ)
  let N := (2, 1)  : (ℝ × ℝ)
  circleEquationThroughPoints A B C ∧ lineEquationOfChord M N
:= sorry

end findCircleAndLineEquations_l636_636028


namespace conic_section_hyperbola_l636_636032

theorem conic_section_hyperbola : 
    ∀ (x y : ℝ), 
    (x - 3)^2 = 2 * (y + 1)^2 + 50 → 
    conic_type ((x - 3)^2 = 2 * (y + 1)^2 + 50) = "H" :=
by
    intros x y h
    sorry

end conic_section_hyperbola_l636_636032


namespace quad_abcd_proof_l636_636565

theorem quad_abcd_proof (A B C D E : Type)
  (BC CD AD : ℝ)
  (m_A m_B : ℝ)
  (AB : ℝ) :
  BC = 10 → CD = 15 → AD = 12 → m_A = 60 → m_B = 60 → AB = 22 + real.sqrt 0 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end quad_abcd_proof_l636_636565


namespace error_percent_in_volume_l636_636973

theorem error_percent_in_volume (a b c : ℝ) : 
  let a' := 1.08 * a,
      b' := 0.93 * b,
      c' := 1.05 * c,
      V := a * b * c,
      V' := a' * b' * c' in
  (V' - V) / V * 100 = 0.884 :=
by
  let a' := 1.08 * a
  let b' := 0.93 * b
  let c' := 1.05 * c
  let V := a * b * c
  let V' := a' * b' * c'
  calc
    (V' - V) / V * 100 = ((1.08 * 0.93 * 1.05 * a * b * c) - a * b * c) / (a * b * c) * 100 : by sorry
                      ... = (1.08 * 0.93 * 1.05 - 1) * 100 : by sorry
                      ... = 0.884 : by sorry

end error_percent_in_volume_l636_636973


namespace simplify_90_54_150_90_l636_636257

def simplify_fraction : Fraction := 12 / 5

theorem simplify_90_54_150_90 :
  (90 + 54) / (150 - 90) = simplify_fraction := by
  sorry

end simplify_90_54_150_90_l636_636257


namespace probability_of_selecting_cooking_l636_636945

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636945


namespace min_value_a_2b_l636_636470

theorem min_value_a_2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 3 / b = 1) :
  a + 2 * b = 7 + 2 * Real.sqrt 6 :=
sorry

end min_value_a_2b_l636_636470


namespace equal_pair_l636_636008

theorem equal_pair :
  (−(-2) ≠ -|-2|) ∧
  (-1^2 ≠ (-1)^2) ∧
  ((-2)^3 = -2^3) ∧
  (22 / 3 ≠ (2 / 3)^2) := 
by
  sorry

end equal_pair_l636_636008


namespace frame_percentage_l636_636971

theorem frame_percentage : 
  let side_length := 80
  let frame_width := 4
  let total_area := side_length * side_length
  let picture_side_length := side_length - 2 * frame_width
  let picture_area := picture_side_length * picture_side_length
  let frame_area := total_area - picture_area
  let frame_percentage := (frame_area * 100) / total_area
  frame_percentage = 19 := 
by
  sorry

end frame_percentage_l636_636971


namespace plot_length_is_65_l636_636682

-- Define the known constants
def cost_per_meter : ℝ := 26.50
def total_cost : ℝ := 5300
def cost_per_meter_nonzero : cost_per_meter ≠ 0 := by
  norm_num

-- Define the variables
def breadth_of_plot (b : ℝ) : Prop :=
  -- Let l be the length of the plot
  ∃ l : ℝ, l = b + 30 ∧
              -- Perimeter equation derived from the cost
              4 * b + 60 = total_cost / cost_per_meter ∧
              -- Ensure positive breadth
              b > 0

-- The Target: To prove the length is 65 meters
def length_of_plot : ℝ := 65

-- Proof problem in Lean 4
theorem plot_length_is_65 (b : ℝ) (h : breadth_of_plot b) : length_of_plot = 65 :=
  sorry

end plot_length_is_65_l636_636682


namespace equivalent_proof_problem_l636_636083

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

def a_n := λ n, 2 ^ (n - 1)

def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  3 / (Real.log2 (a (n + 1)) * Real.log2 (a (n + 2)))

def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, b_n a (k + 1))

def f (x a : ℝ) := -x^2 + 2*a*x - a^2 + a - 1

theorem equivalent_proof_problem (a : ℕ → ℝ) (n : ℕ) (x : ℝ) :
  sequence a ∧ a 1 = 1 → a = a_n ∧ T_n a > f x → a < 5 / 2 :=
begin
  sorry
end

end equivalent_proof_problem_l636_636083


namespace hexagonal_peg_board_unique_placement_l636_636310

theorem hexagonal_peg_board_unique_placement :
  ∃! (f : ℕ → ℕ × ℕ), 
    (∀ i j, i ≠ j → f i ≠ f j) ∧ 
    (∀ c ∈ [0, 1, 2, 3, 4],
        (∀ k l, k ≠ l → (f (c * 6 + k)).fst ≠ (f (c * 6 + l)).fst) ∧
        (∀ k l, k ≠ l → (f (c * 6 + k)).snd ≠ (f (c * 6 + l)).snd)) :=
begin
  sorry
end

end hexagonal_peg_board_unique_placement_l636_636310


namespace arrange_in_ascending_order_l636_636097

-- Define the variables a, b, and c as per the conditions.
def a : ℝ := 2^(-3)
def b : ℝ := 3^(1/2)
def c : ℝ := Real.log 5 / Real.log 2 -- log_{2} 5 in terms of natural logarithm.

-- State the theorem that arranges a, b, and c in ascending order.
theorem arrange_in_ascending_order : a < b ∧ b < c := by
  -- Proof goes here
  sorry

end arrange_in_ascending_order_l636_636097


namespace determine_a_l636_636261

noncomputable def is_monotonic (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f(x) ≤ f(y)

theorem determine_a (f : ℝ → ℝ) (x_0 a : ℝ) (h_monotonic : is_monotonic f)
  (h_eq1 : ∀ x, f(f(x) - log x / log 3) = 4)
  (h_eq2 : f(x_0) - 2 * (deriv f x_0) = 3)
  (h_x0 : x_0 ∈ set.Ioo a (a + 1))
  (h_nat : a ∈ set.Ici 1) :
  a = 2 := sorry

end determine_a_l636_636261


namespace probability_cooking_selected_l636_636912

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636912


namespace circle_equation_tangent_to_directrix_l636_636366

noncomputable def parabola_directrix (p : ℝ) : ℝ := -p / 4

noncomputable def standard_circle_eq (h : ℝ) : ℝ × ℝ → Prop :=
  λ (point : ℝ × ℝ), point.1^2 + point.2^2 = h^2

theorem circle_equation_tangent_to_directrix
  (h : ℝ) 
  (origin : ℝ × ℝ := (0, 0))
  (directrix : ℝ := parabola_directrix 4) :
  standard_circle_eq 1 origin := 
sorry

end circle_equation_tangent_to_directrix_l636_636366


namespace calculate_fraction_l636_636755

variables (n_bl: ℕ) (deg_warm: ℕ) (total_deg: ℕ) (total_bl: ℕ)

def blanket_fraction_added := total_deg / deg_warm

theorem calculate_fraction (h1: deg_warm = 3) (h2: total_deg = 21) (h3: total_bl = 14) :
  (blanket_fraction_added total_deg deg_warm) / total_bl = 1 / 2 :=
by {
  sorry
}

end calculate_fraction_l636_636755


namespace probability_selecting_cooking_l636_636904

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636904


namespace fraction_of_positive_number_l636_636362

theorem fraction_of_positive_number (x : ℝ) (f : ℝ) (h : x = 0.4166666666666667 ∧ f * x = (25/216) * (1/x)) : f = 2/3 :=
sorry

end fraction_of_positive_number_l636_636362


namespace right_triangle_AB_length_l636_636151

theorem right_triangle_AB_length {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]
  (h1 : ∠A = 90) (h2 : tan(B) = 5/12) (h3 : dist(B, C) = 39) :
  dist(A, B) = 36 :=
sorry

end right_triangle_AB_length_l636_636151


namespace distance_to_triangle_from_center_l636_636239

theorem distance_to_triangle_from_center {O P Q R : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  (h1 : Metric.ball O 25 = SetOf (\(p : P) -> Metric.dist p O < 25))
  (h2 : Metric.dist P Q = 15)
  (h3 : Metric.dist Q R = 20)
  (h4 : Metric.dist R P = 25)
  (h5 : ∀ x y z : ℕ, x ∈ Nat.Gcd x z = 1)
  (h6 : ∀ (n : ℕ), ¬ n ∈ Nat.Prime*Prime) :
  ∃ x y z : ℕ, (Metric.dist O (SetOf (\(p : Metric.ball O 25) PQR)) = (x*sqrt(y))/z) ∧ x + y + z = 66 :=
by
  sorry

end distance_to_triangle_from_center_l636_636239


namespace magnitude_of_angle_B_sum_of_sequence_l636_636150

-- Definitions
variables {A B C a b c : Real}
variables {a_n : ℕ → Real}
variables {n : ℕ}
variable  {d : Real}

-- Conditions for Question 1
def triangle_condition_1 : a^2 - (b - c)^2 = (2 - Real.sqrt 3) * b * c := sorry
def triangle_condition_2 : Real.sin A * Real.sin B = (Real.cos (C / 2))^2 := sorry

-- Condition for Question 2
def arithmetic_seq_condition : ∀ n, a_n = 2 * n := sorry

-- Solutions
theorem magnitude_of_angle_B 
  (h1 : triangle_condition_1)
  (h2 : triangle_condition_2) :
  B = Real.pi / 6 := sorry

theorem sum_of_sequence 
  (h1 : a_n 1 * Real.cos (2 * B) = 1)
  (h2 : (a_n 2, a_n 4, a_n 8) in arithmetic_seq_condition) :
  ∑ i in Finset.range n, 4 / (a_n i * a_n (i + 1)) = n / (n + 1) := sorry

end magnitude_of_angle_B_sum_of_sequence_l636_636150


namespace handshaking_remainder_div_1000_l636_636157

/-- Given eleven people where each person shakes hands with exactly three others, 
  let handshaking_count be the number of distinct handshaking arrangements.
  Find the remainder when handshaking_count is divided by 1000. -/
def handshaking_count (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) : Nat :=
  sorry

theorem handshaking_remainder_div_1000 (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) :
  (handshaking_count P hP handshakes H) % 1000 = 800 :=
sorry

end handshaking_remainder_div_1000_l636_636157


namespace calc_y_l636_636549

variable (P0 P1 P2 P3 P4 : ℝ)
variable (y : ℝ)

def price_after_january : ℝ := P0 * 1.30
def price_after_february : ℝ := price_after_january P0 * 0.85
def price_after_march : ℝ := price_after_february P0 * 1.10
def price_condition : ℝ := P0 * 1.05

theorem calc_y (P0 : ℝ) (h1 : P1 = price_after_january P0)
               (h2 : P2 = price_after_february P0)
               (h3 : P3 = price_after_march P0)
               (h4 : P4 = price_condition P0)
               (h5 : P4 = P3 * (1 - y / 100)) :
                abs(y - 14) < 1 := by
  sorry

end calc_y_l636_636549


namespace time_to_school_l636_636714

theorem time_to_school (total_distance walk_speed run_speed distance_ran : ℕ) (h_total : total_distance = 1800)
    (h_walk_speed : walk_speed = 70) (h_run_speed : run_speed = 210) (h_distance_ran : distance_ran = 600) :
    total_distance / walk_speed + distance_ran / run_speed = 20 := by
  sorry

end time_to_school_l636_636714


namespace find_original_sales_tax_percentage_l636_636694

noncomputable def original_sales_tax_percentage (x : ℝ) : Prop :=
∃ (x : ℝ),
  let reduced_tax := 10 / 3 / 100;
  let market_price := 9000;
  let difference := 14.999999999999986;
  (x / 100 * market_price - reduced_tax * market_price = difference) ∧ x = 0.5

theorem find_original_sales_tax_percentage : original_sales_tax_percentage 0.5 :=
sorry

end find_original_sales_tax_percentage_l636_636694


namespace frequency_distribution_proportion_l636_636712

def Mean (heights : List ℝ) : ℝ := sorry
def Variance (heights : List ℝ) : ℝ := sorry
def Mode (heights : List ℝ) : ℝ := sorry
def FrequencyDistribution (heights : List ℝ) : List (ℝ × ℕ) := sorry -- Assuming a list of (range, frequency)

theorem frequency_distribution_proportion (heights : List ℝ) :
  (FrequencyDistribution heights).Proportion = "reveals the proportion of students at each height" :=
sorry

-- Note: .Proportion is a placeholder for whatever method reveals proportion.

end frequency_distribution_proportion_l636_636712


namespace hyperbola_eccentricity_l636_636534

theorem hyperbola_eccentricity (h : ∀ x y m : ℝ, x^2 - y^2 / m = 1 → m > 0 → (Real.sqrt (1 + m) = Real.sqrt 3)) : ∃ m : ℝ, m = 2 := sorry

end hyperbola_eccentricity_l636_636534


namespace seq_problem_part1_seq_problem_part2_l636_636169

def seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

theorem seq_problem_part1 (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  a 2008 = 0 := 
sorry

theorem seq_problem_part2 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  ∃ (M : ℤ), 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = 0) ∧ 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = M) := 
sorry

end seq_problem_part1_seq_problem_part2_l636_636169


namespace percentage_calculation_l636_636343

theorem percentage_calculation (amount : ℝ) (percentage : ℝ) (res : ℝ) :
  amount = 400 → percentage = 0.25 → res = amount * percentage → res = 100 := by
  intro h_amount h_percentage h_res
  rw [h_amount, h_percentage] at h_res
  norm_num at h_res
  exact h_res

end percentage_calculation_l636_636343


namespace lines_coplanar_l636_636632

/-
Given:
- Line 1 parameterized as (2 + s, 4 - k * s, -1 + k * s)
- Line 2 parameterized as (2 * t, 2 + t, 3 - t)
Prove: If these lines are coplanar, then k = -1/2
-/
theorem lines_coplanar (k : ℚ) (s t : ℚ)
  (line1 : ℚ × ℚ × ℚ := (2 + s, 4 - k * s, -1 + k * s))
  (line2 : ℚ × ℚ × ℚ := (2 * t, 2 + t, 3 - t))
  (coplanar : ∃ (s t : ℚ), line1 = line2) :
  k = -1 / 2 := 
sorry

end lines_coplanar_l636_636632


namespace probability_of_selecting_cooking_l636_636841

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636841


namespace find_original_triangle_area_l636_636275

-- Define the conditions and question
def original_triangle_area (A : ℝ) : Prop :=
  let new_area := 4 * A in
  new_area = 32

-- State the problem to prove the area of the original triangle
theorem find_original_triangle_area (A : ℝ) : original_triangle_area A → A = 8 := by
  intro h
  sorry

end find_original_triangle_area_l636_636275


namespace total_number_of_squares_l636_636636

variable (x y : ℕ) -- Variables for the number of 10 cm and 20 cm squares

theorem total_number_of_squares
  (h1 : 100 * x + 400 * y = 2500) -- Condition for area
  (h2 : 40 * x + 80 * y = 280)    -- Condition for cutting length
  : (x + y = 16) :=
sorry

end total_number_of_squares_l636_636636


namespace find_remainder_l636_636206

def M : ℕ := 
  let digits := concat (List.range' 1 46) -- range from 1 to 46
  digits.foldl (λ acc d => acc * 10 + d) 0 -- concatenate the digits to form the number

theorem find_remainder (M_div_47 : M = 12345678910111213141516171819202122232425262728293031323334353637383940414243444546) :
  M % 47 = 1 :=
by
  sorry

end find_remainder_l636_636206


namespace jacob_pyramid_sticks_additional_l636_636192

theorem jacob_pyramid_sticks_additional (total_sticks_for_3_steps : ℕ) (step_increase : ℕ) :
  total_sticks_for_3_steps = 20 → step_increase = 3 →
  let sticks_needed n := 5 + (n - 1) * step_increase in
  let additional_sticks := (sticks_needed 4 - sticks_needed 3) + 
                           (sticks_needed 5 - sticks_needed 4) + 
                           (sticks_needed 6 - sticks_needed 5) in
  additional_sticks = 33 :=
by
  intros h1 h2
  let sticks_needed := λ n, 5 + (n - 1) * step_increase
  let additional_sticks := (sticks_needed 4 - sticks_needed 3) + 
                           (sticks_needed 5 - sticks_needed 4) + 
                           (sticks_needed 6 - sticks_needed 5)
  sorry

end jacob_pyramid_sticks_additional_l636_636192


namespace best_estimate_flight_time_l636_636692

def radius_earth : ℝ := 4000
def speed_jet : ℝ := 500
def circumference_earth (r : ℝ) : ℝ := 2 * Real.pi * r
def time_taken (distance speed : ℝ) : ℝ := distance / speed

theorem best_estimate_flight_time :
  let distance := circumference_earth radius_earth in
  let time := time_taken distance speed_jet in
  abs (time - 50) < 1 := sorry

end best_estimate_flight_time_l636_636692


namespace constant_term_of_product_l636_636324

def P(x: ℝ) : ℝ := x^6 + 2 * x^2 + 3
def Q(x: ℝ) : ℝ := x^4 + x^3 + 4
def R(x: ℝ) : ℝ := 2 * x^2 + 3 * x + 7

theorem constant_term_of_product :
  let C := (P 0) * (Q 0) * (R 0)
  C = 84 :=
by
  let C := (P 0) * (Q 0) * (R 0)
  show C = 84
  sorry

end constant_term_of_product_l636_636324


namespace max_zero_coeffs_l636_636381

theorem max_zero_coeffs (P : polynomial ℝ) (n : ℕ) (h_deg : P.degree = n) (h_roots : ∀ i j < n, i ≠ j → root P i → root P j → i ≠ j) :
  ∃ max_zeros: ℕ, max_zeros = if even n then n / 2 else (n + 1) / 2 :=
begin
  sorry
end

end max_zero_coeffs_l636_636381


namespace no_partition_of_interval_l636_636252

theorem no_partition_of_interval (A B : set ℝ) (a : ℝ) :
  (∀ x ∈ A, x ∈ set.Icc 0 1) ∧ (∀ x ∈ B, x ∈ set.Icc 0 1) ∧
  A ∪ B = set.Icc 0 1 ∧ A ∩ B = ∅ ∧ (∀ x, x ∈ B ↔ x - a ∈ A) →
  false :=
by
  sorry

end no_partition_of_interval_l636_636252


namespace probability_select_cooking_l636_636891

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636891


namespace total_books_l636_636640

def numberOfMysteryShelves := 6
def numberOfPictureShelves := 2
def booksPerShelf := 9

theorem total_books (hMystery : numberOfMysteryShelves = 6) 
                    (hPicture : numberOfPictureShelves = 2) 
                    (hBooksPerShelf : booksPerShelf = 9) :
  numberOfMysteryShelves * booksPerShelf + numberOfPictureShelves * booksPerShelf = 72 :=
  by 
  sorry

end total_books_l636_636640


namespace projection_of_a_on_b_magnitude_of_a_plus_2b_range_of_lambda_acute_angle_l636_636070

variables (a b : ℝ^3) (λ : ℝ)
-- Conditions
def cond1 : |a| = Real.sqrt 2 := sorry
def cond2 : |b| = 1 := sorry
def cond3 : Real.angle a b = Real.pi / 4 := sorry

-- Proofs required
theorem projection_of_a_on_b : Real.projection a b = 1 :=
by
  have h_cond1 := cond1 a b
  have h_cond2 := cond2 a b
  have h_cond3 := cond3 a b
  sorry

theorem magnitude_of_a_plus_2b : |a + 2 • b| = Real.sqrt 10 :=
by
  have h_cond1 := cond1 a b
  have h_cond2 := cond2 a b
  have h_cond3 := cond3 a b
  sorry

theorem range_of_lambda_acute_angle : 1 < λ ∧ λ < Real.sqrt 6 ∨ Real.sqrt 6 < λ ∧ λ < 6 :=
by
  have h_cond1 := cond1 a b
  have h_cond2 := cond2 a b
  have h_cond3 := cond3 a b
  sorry

end projection_of_a_on_b_magnitude_of_a_plus_2b_range_of_lambda_acute_angle_l636_636070


namespace polynomial_abc_value_l636_636541

theorem polynomial_abc_value (a b c : ℝ) (h : a * (x^2) + b * x + c = (x - 1) * (x - 2)) : a * b * c = -6 :=
by
  sorry

end polynomial_abc_value_l636_636541


namespace solve_fractional_eq_l636_636258

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (1 / (x + 2) + 4 * x / (x^2 - 4) = 1 / (x - 2)) ↔ (x = 1) :=
by
  split
  sorry

end solve_fractional_eq_l636_636258


namespace probability_of_selecting_cooking_l636_636845

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636845


namespace max_integers_greater_than_26_l636_636302

-- Define the condition that the sum of five integers is 3
def sum_of_five_integers_is_3 (a b c d e : ℤ) : Prop := a + b + c + d + e = 3

-- Define the condition that an integer is greater than 26
def greater_than_26 (x : ℤ) : Prop := x > 26

-- Define the main theorem to be proven
theorem max_integers_greater_than_26 (a b c d e : ℤ) : 
  sum_of_five_integers_is_3 a b c d e → 
  ∃ (k : ℕ), k ≤ 5 ∧ 
  (∀ (i : Fin 5 → ℤ) (P : Fin 5 → Prop), (∀ j, P j → greater_than_26 (i j)) → 
  (sum_of_five_integers_is_3 (i 0) (i 1) (i 2) (i 3) (i 4)) → 
  sum_of_five_integers_is_3 a b c d e →
  k = 4)

end max_integers_greater_than_26_l636_636302


namespace first_player_wins_l636_636719

theorem first_player_wins :
  ∀ {table : Type} {coin : Type} 
  (can_place : table → coin → Prop) -- function defining if a coin can be placed on the table
  (not_overlap : ∀ (t : table) (c1 c2 : coin), (can_place t c1 ∧ can_place t c2) → c1 ≠ c2) -- coins do not overlap
  (first_move_center : table → coin) -- first player places the coin at the center
  (mirror_move : table → coin → coin), -- function to place a coin symmetrically
  (∃ strategy : (table → Prop) → (coin → Prop),
    (∀ (t : table) (p : table → Prop), p t → strategy p (mirror_move t (first_move_center t))) ∧ 
    (∀ (t : table) (p : table → Prop), strategy p (first_move_center t) → p t)) := sorry

end first_player_wins_l636_636719


namespace probability_cooking_is_one_fourth_l636_636866
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636866


namespace bottle_caps_left_l636_636442

theorem bottle_caps_left (initial_caps lost_caps : ℝ) (h1 : initial_caps = 63.75) (h2 : lost_caps = 18.36) : 
  initial_caps - lost_caps = 45.39 :=
by
  rw [h1, h2]
  norm_num
  sorry

end bottle_caps_left_l636_636442


namespace part1_part2_i_part2_ii_l636_636460

def b (a : ℕ → ℝ) (n k : ℕ) := a n + a (n + k)

theorem part1 (a : ℕ → ℝ) (h : ∀ n, b a n 2 - b a n 1 = 1) (n : ℕ) :
  b a n 4 - b a n 1 = 3 :=
sorry

theorem part2_i (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : ∀ n k, b a (n + 1) k = 2 * b a n k) :
  ∀ n, a n = 2 ^ n :=
sorry

theorem part2_ii (a : ℕ → ℝ) (h1 : ∀ n, a n = 2 ^ n)
  (A B : Set ℝ)
  (def_A : ∀ k, A = {x | ∃ n, x = b a n k})
  (def_B : ∀ k, B = {y | ∃ n, y = 5 * b a n (k + 2)}) :
  ∀ k, A ∩ B = ∅ :=
sorry

end part1_part2_i_part2_ii_l636_636460


namespace inequality_divisor_function_l636_636135

-- Define the divisor function
def d (x : ℕ) : ℕ := x.divisors.card

-- State the theorem
theorem inequality_divisor_function (n : ℕ) (h : n > 0) : 
  (∑ k in range n, d(2 * k)) > (∑ k in range n, d(2 * k + 1)) :=
sorry

end inequality_divisor_function_l636_636135


namespace linear_eq_m_minus_2n_zero_l636_636280

theorem linear_eq_m_minus_2n_zero (m n : ℕ) (x y : ℝ) 
  (h1 : 2 * x ^ (m - 1) + 3 * y ^ (2 * n - 1) = 7)
  (h2 : m - 1 = 1) (h3 : 2 * n - 1 = 1) : 
  m - 2 * n = 0 := 
sorry

end linear_eq_m_minus_2n_zero_l636_636280


namespace sum_of_n_for_3n_minus_8_eq_5_l636_636744

theorem sum_of_n_for_3n_minus_8_eq_5 : 
  let S := {n | |3 * n - 8| = 5}
  in S.sum id = 16 / 3 :=
by
  sorry

end sum_of_n_for_3n_minus_8_eq_5_l636_636744


namespace effective_writing_speed_is_750_l636_636378

-- Definitions based on given conditions in problem part a)
def total_words : ℕ := 60000
def total_hours : ℕ := 100
def break_hours : ℕ := 20
def effective_hours : ℕ := total_hours - break_hours
def effective_writing_speed : ℕ := total_words / effective_hours

-- Statement to be proved
theorem effective_writing_speed_is_750 : effective_writing_speed = 750 := by
  sorry

end effective_writing_speed_is_750_l636_636378


namespace josh_money_left_l636_636596

theorem josh_money_left :
  let initial_money := 100.00
  let shirt_cost := 12.67
  let meal_cost := 25.39
  let magazine_cost := 14.25
  let debt_payment := 4.32
  let gadget_cost := 27.50
  let total_spent := shirt_cost + meal_cost + magazine_cost + debt_payment + gadget_cost
  let money_left := initial_money - total_spent
  money_left = 15.87 :=
by
  let initial_money := 100.00
  let shirt_cost := 12.67
  let meal_cost := 25.39
  let magazine_cost := 14.25
  let debt_payment := 4.32
  let gadget_cost := 27.50
  let total_spent := shirt_cost + meal_cost + magazine_cost + debt_payment + gadget_cost
  let money_left := initial_money - total_spent
  have h1 : total_spent = 84.13 := sorry
  have h2 : money_left = initial_money - 84.13 := sorry
  have h3 : money_left = 15.87 := sorry
  exact h3

end josh_money_left_l636_636596


namespace probability_of_selecting_cooking_l636_636787

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636787


namespace find_R_coordinates_l636_636581

-- Definitions given as conditions
def P : ℝ × ℝ := (1, 3)
def Q : ℝ × ℝ := (4, 7)

-- Additional definition
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Theorem statement
theorem find_R_coordinates : ∃ R : ℝ × ℝ, midpoint P R = Q := by
  use (7, 11)
  sorry

end find_R_coordinates_l636_636581


namespace old_books_to_reread_l636_636019

/-- Brianna problem -/
def total_books_needed : ℕ := 2 * 12

def books_given_as_gift : ℕ := 6

def books_bought : ℕ := 8

def books_borrowed : ℕ := books_bought - 2

def total_new_books : ℕ := books_given_as_gift + books_bought + books_borrowed

theorem old_books_to_reread : 
  let num_old_books := total_books_needed - total_new_books in
  num_old_books = 4 :=
by
  sorry

end old_books_to_reread_l636_636019


namespace sum_inv_sqrt_radius_correct_l636_636963

-- Define the circle configuration problem in Lean.
noncomputable def circle_configuration := sorry

-- Define the function to calculate the desired summation.
noncomputable def sum_inv_sqrt_radius : ℝ := 
  ∑ C in S, 1 / (sqrt (r C))

-- Define the main statement:
theorem sum_inv_sqrt_radius_correct :
  sum_inv_sqrt_radius = 105 / 52 :=
sorry

end sum_inv_sqrt_radius_correct_l636_636963


namespace problem_set_equiv_l636_636988

def positive_nats (x : ℕ) : Prop := x > 0

def problem_set : Set ℕ := {x | positive_nats x ∧ x - 3 < 2}

theorem problem_set_equiv : problem_set = {1, 2, 3, 4} :=
by 
  sorry

end problem_set_equiv_l636_636988


namespace count_even_numbers_correct_l636_636320

/-- 
We are given the digits 1, 2, 3, 4, 6 and need to count how many three-digit even numbers less than 600 can be formed using these digits if each digit can be used more than once. 
-/
def count_even_numbers : ℕ :=
  let hundreds_digits := {d | d ∈ {1, 2, 3, 4, 5}} in
  let tens_digits := {d | d ∈ {1, 2, 3, 4, 6}} in
  let units_digits := {d | d ∈ {2, 4, 6}} in
  (hundreds_digits.size * tens_digits.size * units_digits.size)

/-- The total number of such numbers is 75. -/
theorem count_even_numbers_correct : count_even_numbers = 75 :=
by {
  sorry
}

end count_even_numbers_correct_l636_636320


namespace probability_selecting_cooking_l636_636901

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636901


namespace distance_from_circle_center_to_line_is_one_l636_636364

noncomputable def parametricCircle (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.sin θ + 1, 3 * Real.cos θ - 2)

noncomputable def parametricLine (t : ℝ) : ℝ × ℝ :=
  (4 * t - 6, -3 * t + 2)

def circleCenter : ℝ × ℝ :=
  (1, -2)

def lineEquation (x y : ℝ) : Prop :=
  3 * x + 4 * y + 10 = 0

def distanceToLine (p : ℝ × ℝ) : ℝ :=
  abs (3 * p.1 + 4 * p.2 + 10) / sqrt (3 ^ 2 + 4 ^ 2)

theorem distance_from_circle_center_to_line_is_one :
  distanceToLine circleCenter = 1 :=
by 
  sorry

end distance_from_circle_center_to_line_is_one_l636_636364


namespace relationship_among_three_numbers_l636_636065

noncomputable def M (a b : ℝ) : ℝ := a^b
noncomputable def N (a b : ℝ) : ℝ := Real.log a / Real.log b
noncomputable def P (a b : ℝ) : ℝ := b^a

theorem relationship_among_three_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : N a b < M a b ∧ M a b < P a b := 
by
  sorry

end relationship_among_three_numbers_l636_636065


namespace count_distinct_prime_sums_l636_636431

open Finset

def is_sum_of_two_distinct_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p + q

def given_set : Finset ℕ := mk (list.map (fun k => 5 + 10 * k) (list.range 10)) sorry

theorem count_distinct_prime_sums : 
  (filter is_sum_of_two_distinct_primes given_set).card = 1 := sorry

end count_distinct_prime_sums_l636_636431


namespace move_line_down_5_units_l636_636983

theorem move_line_down_5_units (y x : ℝ) : (y = 2 * x + 3) → (y - 5 = 2 * x - 2) :=
begin
  sorry
end

end move_line_down_5_units_l636_636983


namespace value_of_x_l636_636535

-- Define the conditions extracted from problem (a)
def condition1 (x : ℝ) : Prop := x^2 - 1 = 0
def condition2 (x : ℝ) : Prop := x - 1 ≠ 0

-- The statement to be proved
theorem value_of_x : ∀ x : ℝ, condition1 x → condition2 x → x = -1 :=
by
  intros x h1 h2
  sorry

end value_of_x_l636_636535


namespace minimum_cubes_needed_l636_636764

theorem minimum_cubes_needed 
    (volume_of_small_cube : ℕ)
    (length : ℕ)
    (width : ℕ)
    (depth : ℕ) 
    (total_small_cubes : ℕ := length * width * depth) 
    (hollow_inner_cubes : ℕ := (length - 2) * (width - 2) * (depth - 2)) 
    (min_cubes_needed : ℕ := total_small_cubes - hollow_inner_cubes) :
    volume_of_small_cube = 8 →
    length = 3 →
    width = 9 →
    depth = 5 →
    min_cubes_needed = 114 :=
by
  intros h_volume h_length h_width h_depth
  have h_total : total_small_cubes = 135 := by 
    calc
    total_small_cubes = length * width * depth := by rfl
    _ = 3 * 9 * 5 := by rw [h_length, h_width, h_depth]
    _ = 135 := by norm_num
  have h_inner : hollow_inner_cubes = 21 := by 
    calc
    hollow_inner_cubes = (length - 2) * (width - 2) * (depth - 2) := by rfl
    _ = (3 - 2) * (9 - 2) * (5 - 2) := by rw [h_length, h_width, h_depth]
    _ = 1 * 7 * 3 := by norm_num
    _ = 21 := by norm_num
  have h_min : min_cubes_needed = 114 := by 
    calc
    min_cubes_needed = total_small_cubes - hollow_inner_cubes := by rfl
    _ = 135 - 21 := by rw [h_total, h_inner]
    _ = 114 := by norm_num
  exact h_min

end minimum_cubes_needed_l636_636764


namespace probability_of_selecting_cooking_is_one_fourth_l636_636876

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636876


namespace xiaoming_selects_cooking_probability_l636_636836

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636836


namespace no_four_points_all_triangles_acute_l636_636638

theorem no_four_points_all_triangles_acute (A B C D : Point) : 
  ¬(triangle.is_acute A B C ∧ triangle.is_acute B C D ∧ triangle.is_acute C D A ∧ triangle.is_acute D A B) :=
sorry

end no_four_points_all_triangles_acute_l636_636638


namespace stratified_sample_sum_l636_636000

theorem stratified_sample_sum :
  let grains := 40
  let veg_oils := 10
  let animal_foods := 30
  let fruits_veggies := 20
  let total_varieties := grains + veg_oils + animal_foods + fruits_veggies
  let sample_size := 20
  let veg_oils_proportion := (veg_oils:ℚ) / total_varieties
  let fruits_veggies_proportion := (fruits_veggies:ℚ) / total_varieties
  let veg_oils_sample := sample_size * veg_oils_proportion
  let fruits_veggies_sample := sample_size * fruits_veggies_proportion
  veg_oils_sample + fruits_veggies_sample = 6 := sorry

end stratified_sample_sum_l636_636000


namespace probability_of_selecting_cooking_l636_636817

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636817


namespace xiaoming_selects_cooking_probability_l636_636835

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636835


namespace repeating_decimal_denominator_l636_636665

theorem repeating_decimal_denominator :
  let S := 0.77777777 in
  let frac := 7 / 9 in
  denom frac = 9 :=
by
  sorry

end repeating_decimal_denominator_l636_636665


namespace total_questions_in_exam_l636_636548

theorem total_questions_in_exam (hours : ℕ) (typeA_problems : ℕ) (time_per_A : ℝ) (time_A_twice_B : ∀ tB : ℝ, time_per_A = 2 * tB)
: hours = 3 → typeA_problems = 10 → time_per_A = 17.142857142857142 → (typeA_problems + 1) = 11 :=
by
  intros h_hours h_typeA h_time_per_A
  have h_total_time := h_hours * 60
  have h_total_timeA := h_typeA * h_time_per_A
  have tB := h_time_per_A / 2
  have remaining_time := h_total_time - h_total_timeA
  have typeB_problems := remaining_time / tB
  have total_questions := h_typeA + typeB_problems
  calc total_questions = 10 + 1 := by sorry
                          ... = 11 := by rfl

end total_questions_in_exam_l636_636548


namespace complex_pow_simplify_l636_636693

noncomputable def i : ℂ := Complex.I

theorem complex_pow_simplify :
  (1 + Real.sqrt 3 * Complex.I) ^ 3 * Complex.I = -8 * Complex.I :=
by
  sorry

end complex_pow_simplify_l636_636693


namespace max_profit_l636_636267

def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then 2 * x^2 + 10 * x
  else if 5 < x ∧ x ≤ 12 then 200 - 400 / (x - 1)
  else 0

def total_cost (x : ℝ) : ℝ := 4 * x + 4

def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then 2 * x^2 + 6 * x - 4
  else if 5 < x ∧ x ≤ 12 then 196 - 4 * x - 400 / (x - 1)
  else 0

theorem max_profit : ∃ x : ℝ, x = 11 ∧ f x = 112 := by
  sorry

end max_profit_l636_636267


namespace probability_of_selecting_cooking_l636_636942

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636942


namespace probability_of_cooking_l636_636797

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636797


namespace parallel_vectors_xy_l636_636519

theorem parallel_vectors_xy {x y : ℝ} (h : ∃ k : ℝ, (1, y, -3) = (k * x, k * (-2), k * 5)) : x * y = -2 :=
by sorry

end parallel_vectors_xy_l636_636519


namespace find_n_that_satisfy_inequality_l636_636082

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 1 else 2 ^ (n - 1)

def S (n : ℕ) : ℕ :=
if n = 0 then 0 else 2 * a n - 1

def satisfies_inequality (n : ℕ) : Prop :=
a n ≤ 2 * n

theorem find_n_that_satisfy_inequality :
  {n : ℕ | n ≥ 1 ∧ satisfies_inequality n} = {1, 2, 3, 4} :=
sorry

end find_n_that_satisfy_inequality_l636_636082


namespace f_sum_pos_l636_636069

theorem f_sum_pos (f : ℝ → ℝ) (a b c : ℝ) 
(hf_def : ∀ x, f(x) = x^5 + x)
(h1 : a + b > 0)
(h2 : b + c > 0)
(h3 : c + a > 0) : 
  f(a) + f(b) + f(c) > 0 :=
sorry

end f_sum_pos_l636_636069


namespace geom_series_first_term_l636_636401

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l636_636401


namespace systems_solution_l636_636148

    theorem systems_solution : 
      (∃ x y : ℝ, 2 * x + 5 * y = -26 ∧ 3 * x - 5 * y = 36 ∧ 
                 (∃ a b : ℝ, a * x - b * y = -4 ∧ b * x + a * y = -8 ∧ 
                 (2 * a + b) ^ 2020 = 1)) := 
    by
      sorry
    
end systems_solution_l636_636148


namespace distinct_license_plates_l636_636372

noncomputable def license_plates : ℕ :=
  let digits_possibilities := 10^5
  let letters_possibilities := 26^3
  let positions := 6
  positions * digits_possibilities * letters_possibilities

theorem distinct_license_plates : 
  license_plates = 105456000 := by
  sorry

end distinct_license_plates_l636_636372


namespace find_b_l636_636295

variables {a b c p k : ℝ}
variable (h1 : p ≠ 0)
variable (h2 : k ≠ 0)

def parabola_with_given_vertex_and_y_intercept 
  (h_vertex : (λ x : ℝ, a * x^2 + b * x + c) = (λ x : ℝ, a * (x - p)^2 + k * p))
  (h_y_intercept : (λ x : ℝ, a * x^2 + b * x + c) 0 = -k * p): Prop :=
  b = 4 * k / p

-- Statement of the theorem
theorem find_b (h_vertex : (λ x : ℝ, a * x^2 + b * x + c) = (λ x : ℝ, a * (x - p)^2 + k * p))
  (h_y_intercept : (λ x : ℝ, a * x^2 + b * x + c) 0 = -k * p) :
  b = 4 * k / p :=
sorry

end find_b_l636_636295


namespace relationship_among_a_b_c_l636_636466

def a := (1 / 2) ^ 0.3
def b := (1 / 2) ^ (-2)
def c := Real.logb (1 / 2) 2

theorem relationship_among_a_b_c : b > a ∧ a > c := by
  -- Definitions from the problem
  have ha : 0 < a := sorry
  have hb : 1 < b := sorry
  have hc : c < 0 := sorry
  -- Relationships
  exact sorry

end relationship_among_a_b_c_l636_636466


namespace remainder_when_a6_divided_by_n_l636_636610

theorem remainder_when_a6_divided_by_n (n : ℕ) (a : ℤ) (h : a^3 ≡ 1 [ZMOD n]) :
  a^6 ≡ 1 [ZMOD n] := 
sorry

end remainder_when_a6_divided_by_n_l636_636610


namespace xiaoming_selects_cooking_probability_l636_636822

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636822


namespace non_congruent_rectangles_with_even_dimensions_l636_636382

/-- Given a rectangle with perimeter 120 inches and even integer dimensions,
    prove that there are 15 non-congruent rectangles that meet these criteria. -/
theorem non_congruent_rectangles_with_even_dimensions (h w : ℕ) (h_even : h % 2 = 0) (w_even : w % 2 = 0) (perimeter_condition : 2 * (h + w) = 120) :
  ∃ n : ℕ, n = 15 := sorry

end non_congruent_rectangles_with_even_dimensions_l636_636382


namespace possible_value_of_a_l636_636067

variable {a b x : ℝ}

theorem possible_value_of_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x :=
sorry

end possible_value_of_a_l636_636067


namespace cos_beta_eq_neg_16_over_65_l636_636066

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin β = 5 / 13)
variable (h4 : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_eq_neg_16_over_65 : Real.cos β = -16 / 65 := by
  sorry

end cos_beta_eq_neg_16_over_65_l636_636066


namespace line_segments_property_l636_636073

theorem line_segments_property (L : List (ℝ × ℝ)) :
  L.length = 50 →
  (∃ S : List (ℝ × ℝ), S.length = 8 ∧ ∃ x : ℝ, ∀ seg ∈ S, seg.fst ≤ x ∧ x ≤ seg.snd) ∨
  (∃ T : List (ℝ × ℝ), T.length = 8 ∧ ∀ seg1 ∈ T, ∀ seg2 ∈ T, seg1 ≠ seg2 → seg1.snd < seg2.fst ∨ seg2.snd < seg1.fst) :=
by
  -- Theorem proof placeholder
  sorry

end line_segments_property_l636_636073


namespace b_seq_is_arithmetic_a_seq_general_term_l636_636123

variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)

def a_seq (n : ℕ) : ℝ :=
  if n = 1 then 4
  else 4 - 4 / (a_seq (n - 1))

def b_seq (n : ℕ) : ℝ :=
  1 / (a_seq n - 2)

theorem b_seq_is_arithmetic (n : ℕ) :
  (b_seq 1 = 1 / 2) ∧ (∀ n > 1, b_seq n = b_seq 1 + (n - 1) * (1 / 2)) :=
sorry

theorem a_seq_general_term (n : ℕ) :
  a_seq n = (2 / n) + 2 :=
sorry

end b_seq_is_arithmetic_a_seq_general_term_l636_636123


namespace find_number_l636_636056

theorem find_number (x : ℝ) : (x^2 + 4 = 5 * x) → (x = 4 ∨ x = 1) :=
by
  sorry

end find_number_l636_636056


namespace xiao_xin_min_questions_to_pass_l636_636977

theorem xiao_xin_min_questions_to_pass
    (total_questions : ℕ)
    (points_correct : ℕ)
    (points_incorrect : ℕ)
    (passing_score : ℕ)
    (answered_all : ℕ)
    (correct_answers : ℕ)
    (incorrect_or_unanswered : ℕ := total_questions - correct_answers) :
  total_questions = 20 →
  points_correct = 5 →
  points_incorrect = 3 →
  passing_score = 60 →
  answered_all = 20 →
  correct_answers ≥ 15 :=
by
  intros h1 h2 h3 h4 h5
  let x := correct_answers
  have h6 : total_questions = x + incorrect_or_unanswered, from eq.symm (sub_add_cancel (correct_answers) (total_questions).symm)
  have h7 : incorrect_or_unanswered = 20 - correct_answers, by simp [incorrect_or_unanswered]
  have : 5 * correct_answers - 3 * (20 - correct_answers) ≥ 60, by
    calc 5 * correct_answers - 3 * (20 - correct_answers)
      = 5 * x - 3 * (20 - x) : by { rw [h6, h7], simp }
    ... = 5 * x - 60 + 3 * x  : by { ring }
    ... = 8 * x - 60          : by { ring }
    ... ≥ 60                 : by { 
                                  calc 8 * x - 60 ≥ 60
                                  → 8 * x ≥ 120
                                  → x ≥ 15 
                                  ; linarith 
                                }
  exact this
  done

end xiao_xin_min_questions_to_pass_l636_636977


namespace denominator_exceeds_numerator_by_six_l636_636204

noncomputable def G := (837 : ℚ) / 999

theorem denominator_exceeds_numerator_by_six :
  let (numer, denom) := rat.num_denom (837 / 999)
  denom - numer = 6 :=
by
  sorry

end denominator_exceeds_numerator_by_six_l636_636204


namespace principal_trebled_after_5_years_l636_636700

-- Definitions of the conditions
def original_simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100
def total_simple_interest (P R n T : ℕ) : ℕ := (P * R * n) / 100 + (3 * P * R * (T - n)) / 100

-- The theorem statement
theorem principal_trebled_after_5_years :
  ∀ (P R : ℕ), original_simple_interest P R 10 = 800 →
              total_simple_interest P R 5 10 = 1600 →
              5 = 5 :=
by
  intros P R h1 h2
  sorry

end principal_trebled_after_5_years_l636_636700


namespace turkish_mo_2003_inequality_l636_636103

variable 
  (K L M N : Point)
  (A B C D : Point)
  (hK : K ∈ LineSegment A B)
  (hL : L ∈ LineSegment B C)
  (hM : M ∈ LineSegment C D)
  (hN : N ∈ LineSegment D A)
  (S₁ S₂ S₃ S₄ S : ℝ)
  (hS₁ : S₁ = Area (Triangle.mk A K N))
  (hS₂ : S₂ = Area (Triangle.mk B K L))
  (hS₃ : S₃ = Area (Triangle.mk C L M))
  (hS₄ : S₄ = Area (Triangle.mk D M N))
  (hS : S = Area (Quadrilateral.mk A B C D))

theorem turkish_mo_2003_inequality :
  ∛S₁ + ∛S₂ + ∛S₃ + ∛S₄ ≤ 2 * ∛S :=
  sorry

end turkish_mo_2003_inequality_l636_636103


namespace number_of_pieces_l636_636774

theorem number_of_pieces (number_of_boxes : ℕ) (pieces_per_box : ℕ) (h1: number_of_boxes = 6) (h2: pieces_per_box = 500) : (number_of_boxes * pieces_per_box) = 3000 := 
by
  rw [h1, h2]
  exact dec_trivial

end number_of_pieces_l636_636774


namespace largest_prime_factor_of_1197_l636_636329

theorem largest_prime_factor_of_1197 : ∃ (p : ℕ), p.prime ∧ p ∣ 1197 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 1197 → q ≤ p := 
by {
  let n := 1197,
  have h1 : 3 ∣ n := by { norm_num, exact dvd_of_mod_eq_zero rfl },
  let n1 := n / 3,
  have h2 : 3 ∣ n1 := by { norm_num, exact dvd_of_mod_eq_zero rfl },
  let n2 := n1 / 3,
  have h3 : 7 ∣ n2 := by { norm_num, exact dvd_of_mod_eq_zero rfl },
  let n3 := n2 / 7,
  have h4 : n3 = 19 := by { norm_num },
  have h5 : nat.prime 19 := by { norm_num },
  use 19,
  split,
  { exact h5 },
  split,
  { rw h4, exact dvd.intro (n2 / 19) rfl },
  intros q hq1 hq2,
  cases prime_factors n with factors factors_is_prime,
  suffices : q = 19,
  { simp, exact nat.le_of_eq this },
  suffices : q ∈ factors,
  { exact factors_is_prime q this hq1 },
  sorry
}

end largest_prime_factor_of_1197_l636_636329


namespace probability_of_selecting_cooking_l636_636781

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636781


namespace can_capacity_l636_636762

theorem can_capacity (x : ℝ) (milk water : ℝ) (full_capacity : ℝ) : 
  5 * x = milk ∧ 
  3 * x = water ∧ 
  full_capacity = milk + water + 8 ∧ 
  (milk + 8) / water = 2 → 
  full_capacity = 72 := 
sorry

end can_capacity_l636_636762


namespace minimum_value_of_expression_l636_636077

-- Defining the variables and assumptions
variables (a b : ℝ)
#check (a > 0)
#check (b > 0)
#check (6 * a + 3 * b = 1)

-- Stating the theorem
theorem minimum_value_of_expression (h1 : a > 0) (h2 : b > 0) (h3 : 6 * a + 3 * b = 1) : 
  (1 / (5 * a + 2 * b) + 2 / (a + b)) ≥ 3 + 2 * real.sqrt 2 :=
sorry  -- proof not provided

end minimum_value_of_expression_l636_636077


namespace vector_in_same_plane_l636_636770

theorem vector_in_same_plane (x : ℝ) : 
  let a := (1, -1, 0)
  let b := (-1, 0, 1)
  let c := (1, 3, x)
  (λ m n : ℝ, c.1 = m - n ∧ c.2 = -m ∧ c.3 = n) ∃ m n, (c = (m - n, -m, n)) 
  → x = -4 := 
by
  sorry

end vector_in_same_plane_l636_636770


namespace alchemerion_age_problem_l636_636392

theorem alchemerion_age_problem
  (A S F : ℕ)  -- Declare the ages as natural numbers
  (h1 : A = 3 * S)  -- Condition 1: Alchemerion is 3 times his son's age
  (h2 : F = 2 * A + 40)  -- Condition 2: His father’s age is 40 years more than twice his age
  (h3 : A + S + F = 1240)  -- Condition 3: Together they are 1240 years old
  (h4 : A = 360)  -- Condition 4: Alchemerion is 360 years old
  : 40 = F - 2 * A :=  -- Conclusion: The number of years more than twice Alchemerion’s age is 40
by
  sorry  -- Proof can be filled in here

end alchemerion_age_problem_l636_636392


namespace probability_of_selecting_cooking_is_one_fourth_l636_636872

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636872


namespace original_triangle_area_l636_636278

theorem original_triangle_area (new_area : ℝ) (scaling_factor : ℝ) (area_ratio : ℝ) : 
  new_area = 32 → scaling_factor = 2 → 
  area_ratio = scaling_factor ^ 2 → 
  new_area / area_ratio = 8 := 
by
  intros
  -- insert your proof logic here
  sorry

end original_triangle_area_l636_636278


namespace diagonal_difference_l636_636429

def initial_matrix : Matrix (Fin 5) (Fin 5) ℕ := 
  ![![1, 2, 3, 4, 5],
    ![6, 7, 8, 9, 10],
    ![11, 12, 13, 14, 15],
    ![16, 17, 18, 19, 20],
    ![21, 22, 23, 24, 25]]

def reverse_rows (m : Matrix (Fin 5) (Fin 5) ℕ) : Matrix (Fin 5) (Fin 5) ℕ :=
  ![m[0],
    m[1].reverse,
    m[2].reverse,
    m[3],
    m[4].reverse]

def main_diagonal (m : Matrix (Fin 5) (Fin 5) ℕ) : Fin 5 → ℕ
| i => m[i, i]

def secondary_diagonal (m : Matrix (Fin 5) (Fin 5) ℕ) : Fin 5 → ℕ
| i => m[i, 4 - i]

def sum_diagonal (diag : Fin 5 → ℕ) : ℕ :=
  List.sum (List.ofFn diag)

theorem diagonal_difference :
  let new_matrix := reverse_rows initial_matrix
  | sum_diagonal (main_diagonal new_matrix),
    sum_diagonal (secondary_diagonal new_matrix)
  | abs ( sum_diagonal (main_diagonal new_matrix) - sum_diagonal (secondary_diagonal new_matrix) ) = 4 :=
sorry

end diagonal_difference_l636_636429


namespace triangle_angles_determinant_is_zero_l636_636618

theorem triangle_angles_determinant_is_zero (A B C : ℝ)
    (h₁ : A + B + C = π) : 
    det !![ [Real.cos A ^ 2, Real.tan A, 1],
            [Real.cos B ^ 2, Real.tan B, 1],
            [Real.cos C ^ 2, Real.tan C, 1] ] = 0 := 
by 
    sorry

end triangle_angles_determinant_is_zero_l636_636618


namespace length_of_platform_l636_636002

theorem length_of_platform
  (length_train : ℝ)
  (speed_train_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_covered : ℝ)
  (conversion_factor : ℝ) :
  length_train = 250 →
  speed_train_kmph = 90 →
  time_seconds = 20 →
  distance_covered = (speed_train_kmph * 1000 / 3600) * time_seconds →
  conversion_factor = 1000 / 3600 →
  ∃ P : ℝ, distance_covered = length_train + P ∧ P = 250 :=
by
  sorry

end length_of_platform_l636_636002


namespace probability_of_selecting_cooking_l636_636932

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636932


namespace probability_of_selecting_cooking_l636_636952

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636952


namespace most_convincing_method_l636_636007

-- Defining the survey data
def male_participants : Nat := 4258
def male_believe_doping : Nat := 2360
def female_participants : Nat := 3890
def female_believe_framed : Nat := 2386

-- Defining the question-to-answer equivalence related to the most convincing method
theorem most_convincing_method :
  "Independence Test" = "Independence Test" := 
by
  sorry

end most_convincing_method_l636_636007


namespace correct_equation_l636_636575

variables (x y : ℕ)

-- Conditions
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Theorem to prove
theorem correct_equation (h1 : condition1 x y) (h2 : condition2 x y) : (y + 3) / 8 = (y - 4) / 7 :=
sorry

end correct_equation_l636_636575


namespace vector_BC_calculation_l636_636527

/--
If \(\overrightarrow{AB} = (3, 6)\) and \(\overrightarrow{AC} = (1, 2)\),
then \(\overrightarrow{BC} = (-2, -4)\).
-/
theorem vector_BC_calculation (AB AC BC : ℤ × ℤ) 
  (hAB : AB = (3, 6))
  (hAC : AC = (1, 2)) : 
  BC = (-2, -4) := 
by
  sorry

end vector_BC_calculation_l636_636527


namespace exists_fixed_point_Q_l636_636507

-- Definitions from conditions
def is_on_parabola (x y : ℝ) (m : ℝ) : Prop := y^2 = m * x
def tangent_condition (Q P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t l : ℝ, -- placeholders for intermediate line and tangent equations
  -- Condition for external angle bisector to be tangent to parabola
  external_angle_bisector_tangent P Q A B t l

-- Specific constants
def P := (1 : ℝ, -2 : ℝ)
def m := 4
def parabola_eq (x y : ℝ) : Prop := is_on_parabola x y m

-- Proof Problem Statement
theorem exists_fixed_point_Q : 
  ∃ Q : ℝ × ℝ, 
  parabola_eq 1 (-2) → 
  (∀ A B : ℝ × ℝ, intersects_parabola A B)
  ∧ tangent_condition Q P A B :=
sorry

def external_angle_bisector_tangent (P Q A B : ℝ × ℝ) (t l : ℝ) : Prop :=
-- Placeholder for tangent condition
sorry

def intersects_parabola (A B : ℝ × ℝ) : Prop :=
-- Placeholder for the intersection condition with the parabola
sorry

end exists_fixed_point_Q_l636_636507


namespace plane_existence_l636_636432

theorem plane_existence (a b d : ℝ) (A : ℝ × ℝ) (hA : A = (a, b)) (h_dist : d ≤ sqrt (a^2 + b^2)) :
  ∃ (P : AffineSubspace ℝ (EuclideanSpace ℝ 3)), 
  ∃ (V : AffineSpacePhase ℝ (EuclideanSpace ℝ 3)), 
  ∃ (orthogonality : P ≠ ∅ ∧ (∀ x ∈ P, ∃ y ∈ P, y - x = V)),
  ∀ (projection_axis : ℝ × ℝ), 
  P (EuclideanSpace.origin ℝ 3) ∧ dist (A, projection_axis) = sqrt (a^2 + b^2 - d^2) :=
sorry

end plane_existence_l636_636432


namespace number_of_integer_P_in_third_quadrant_l636_636102

def in_third_quadrant (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ P.2 < 0

def point_P (m : ℝ) : ℝ × ℝ := (2 - 4 * m, m - 4)

theorem number_of_integer_P_in_third_quadrant : 
  (ℕ.filter (λ m, m > 1 / 2 ∧ m < 4 ∧ in_third_quadrant (point_P m))).length = 3 := 
by
  sorry

end number_of_integer_P_in_third_quadrant_l636_636102


namespace first_term_of_geometric_series_l636_636415

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l636_636415


namespace fred_games_last_year_proof_l636_636063

def fred_games_last_year (this_year: ℕ) (diff: ℕ) : ℕ := this_year + diff

theorem fred_games_last_year_proof : 
  ∀ (this_year: ℕ) (diff: ℕ),
  this_year = 25 → 
  diff = 11 →
  fred_games_last_year this_year diff = 36 := 
by 
  intros this_year diff h_this_year h_diff
  rw [h_this_year, h_diff]
  sorry

end fred_games_last_year_proof_l636_636063


namespace max_value_l636_636218

open Real

theorem max_value (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha1 : a ≤ 1) (hb1 : b ≤ 1) (hc1 : c ≤ 1/2) :
  sqrt (a * b * c) + sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ (1 / sqrt 2) + (1 / 2) :=
sorry

end max_value_l636_636218


namespace find_number_l636_636773

theorem find_number (x : ℝ) (h : (3 / 4) * (1 / 2) * (2 / 5) * x = 753.0000000000001) : 
  x = 5020.000000000001 :=
by 
  sorry

end find_number_l636_636773


namespace find_b_perpendicular_lines_l636_636440

theorem find_b_perpendicular_lines :
  (∀ (b : ℝ), 
    (∃ (a_1 : ℝ), ∀ x, (λ x, -3 * x + 7) x = a_1) ∧ 
    (∃ (a_2 : ℝ), ∀ x, (λ x, (-b / 9) * x + 2) x = a_2) → 
    (-3) * (-b / 9) = -1 → 
    b = -3) :=
by
  sorry

end find_b_perpendicular_lines_l636_636440


namespace relationship_between_vars_l636_636526

-- Define the variables a, b, c, d as real numbers
variables (a b c d : ℝ)

-- Define the initial condition
def initial_condition := (a + 2 * b) / (2 * b + c) = (c + 2 * d) / (2 * d + a)

-- State the theorem to be proved
theorem relationship_between_vars (h : initial_condition a b c d) : 
  a = c ∨ a + c + 2 * (b + d) = 0 :=
sorry

end relationship_between_vars_l636_636526


namespace count_even_divisors_8_l636_636521

theorem count_even_divisors_8! :
  ∃ (even_divisors total : ℕ),
    even_divisors = 84 ∧
    total = 56 :=
by
  /-
    To formulate the problem in Lean:
    We need to establish two main facts:
    1. The count of even divisors of 8! is 84.
    2. The count of those even divisors that are multiples of both 2 and 3 is 56.
  -/
  sorry

end count_even_divisors_8_l636_636521


namespace sum_of_n_for_3n_minus_8_eq_5_l636_636745

theorem sum_of_n_for_3n_minus_8_eq_5 : 
  let S := {n | |3 * n - 8| = 5}
  in S.sum id = 16 / 3 :=
by
  sorry

end sum_of_n_for_3n_minus_8_eq_5_l636_636745


namespace scientific_notation_of_0_000000023_l636_636152

theorem scientific_notation_of_0_000000023 : 
  0.000000023 = 2.3 * 10^(-8) :=
sorry

end scientific_notation_of_0_000000023_l636_636152


namespace Lisa_flight_time_l636_636624

theorem Lisa_flight_time :
  let distance := 500
  let speed := 45
  (distance : ℝ) / (speed : ℝ) = 500 / 45 := by
  sorry

end Lisa_flight_time_l636_636624


namespace Ap_geq_Ai_l636_636348

variables {A B C P I : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space I]
variables (triangle_ABC : A ≠ B ∧ B ≠ C ∧ C ≠ A)
variables (incenter_I : ∀ (A B C : Type), true) -- This line serves as a placeholder for incenter definition
variables (in_triangle_P : P ∈ triangle_ABC)
variables (angle_condition : ∀ (P B A C : Type), ∠(P B A) + ∠(P C A) = ∠(P B C) + ∠(P C B))

theorem Ap_geq_Ai (triangle_ABC : A ≠ B ∧ B ≠ C ∧ C ≠ A) 
  (incenter_I : ∀ (A B C : Type), true) -- Here, you define I as the incenter
  (in_triangle_P : P ∈ triangle_ABC)
  (angle_condition : ∀ (P B A C : Type), ∠(P B A) + ∠(P C A) = ∠(P B C) + ∠(P C B))
  : ∀ AP AI, AP ≥ AI ∧ (AP = AI ↔ P = I) :=
by sorry

end Ap_geq_Ai_l636_636348


namespace smaller_two_digit_product_l636_636690

-- Statement of the problem
theorem smaller_two_digit_product (a b : ℕ) (h : a * b = 4536) (h_a : 10 ≤ a ∧ a < 100) (h_b : 10 ≤ b ∧ b < 100) : a = 21 ∨ b = 21 := 
by {
  sorry -- proof not needed
}

end smaller_two_digit_product_l636_636690


namespace total_distance_cycled_l636_636191

theorem total_distance_cycled (d_store_peter : ℕ) 
  (store_to_peter_distance : d_store_peter = 50) 
  (home_to_store_twice : ∀ t, distance_from_home_to_store * t = 2 * d_store_peter * t)
  : distance_from_home_to_store = 100 → 
    total_distance = (distance_from_home_to_store + store_to_peter_distance + d_store_peter) :=
by
  /- Let distance_from_home_to_store be denoted as d_home_store for simplicity -/
  let d_home_store := distance_from_home_to_store
  -- Given
  have home_to_store_distance : d_home_store = 2 * d_store_peter := home_to_store_twice 1
  -- We can derive
  have h1 : d_home_store = 100 := home_to_store_distance
  -- Calculate total distance
  have total_distance := d_home_store + d_store_peter + d_store_peter
  -- Conclusion
  have h2 : total_distance = 100 + 50 + 100 := rfl
  -- Final Proposition
  exact h2

end total_distance_cycled_l636_636191


namespace Incorrect_Proposition_D_l636_636335

theorem Incorrect_Proposition_D :
  (∀ (a b m : ℝ), (am < bm → a < b) → ¬ (a < b → am < bm)) :=
begin
  -- The proof goes here
  sorry
end

end Incorrect_Proposition_D_l636_636335


namespace evaluate_expression_l636_636262

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 := by
  sorry

end evaluate_expression_l636_636262


namespace repeating_decimal_denominator_l636_636666

theorem repeating_decimal_denominator :
  let S := 0.77777777 in
  let frac := 7 / 9 in
  denom frac = 9 :=
by
  sorry

end repeating_decimal_denominator_l636_636666


namespace xy_square_sum_l636_636532

theorem xy_square_sum (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 65 :=
by
  sorry

end xy_square_sum_l636_636532


namespace evaluate_expression_l636_636255

theorem evaluate_expression (x : ℝ) (h : x = Real.sqrt 3) : 
  ( (x^2 - 2*x + 1) / (x^2 - x) / (x - 1) ) = Real.sqrt 3 / 3 :=
by
  sorry

end evaluate_expression_l636_636255


namespace distance_from_P_to_center_of_circle_l636_636958

noncomputable def point_on_tangent_lines_symmetric_dist {x y : ℝ} (hx : x = 0) (hy : y = 0)
    (h1 : ∀ x y, (x-3)^2 + (y+1)^2 = 2)
    (h2 : ∀ x y, y = 3 * x) : ℝ :=
  let P := (0, 0 : ℝ × ℝ)
  let C := (3, -1 : ℝ × ℝ)
  let distance := Real.sqrt ((C.1 - P.1) ^ 2 + (C.2 - P.2) ^ 2) in
  distance

theorem distance_from_P_to_center_of_circle : point_on_tangent_lines_symmetric_dist 0 0 sorry sorry = Real.sqrt 10 :=
  by sorry

end distance_from_P_to_center_of_circle_l636_636958


namespace speed_second_half_l636_636376

theorem speed_second_half (total_time : ℝ) (speed_first_half : ℝ) (total_distance : ℝ) : 
  total_time = 10 → speed_first_half = 21 → total_distance = 224 → 
  let distance_half := total_distance / 2 in 
  let time_first_half := distance_half / speed_first_half in 
  let time_second_half := total_time - time_first_half in 
  let speed_second_half := distance_half / time_second_half in 
  speed_second_half = 24 :=
by
  intros ht hs hd
  let distance_half := total_distance / 2
  let time_first_half := distance_half / speed_first_half
  let time_second_half := total_time - time_first_half
  let speed_second_half := distance_half / time_second_half
  sorry

end speed_second_half_l636_636376


namespace proportional_enlargement_height_l636_636627

def initial_width : ℕ := 3
def initial_height : ℕ := 2
def new_width : ℕ := 12

theorem proportional_enlargement_height : 
  (new_width / initial_width) * initial_height = 8 := 
by
  -- conditions setup
  have proportion_factor : ℕ := new_width / initial_width
  have new_height : ℕ := proportion_factor * initial_height
  -- proof (implementation abstracted)
  exact eq.refl new_height

end proportional_enlargement_height_l636_636627


namespace eq_of_line_through_center_and_perpendicular_l636_636078

noncomputable def line_through_center_and_perpendicular (x y : ℝ) : Prop :=
  let circle_eq := x^2 + y^2 - 6*y + 5 = 0
  let perp_line_eq := x + y + 1 = 0
  -- Center of the circle (0,3)
  let center := (0, 3)
  -- Perpendicular line passing through center with the given slope
  (y - 3 = x - 0)

theorem eq_of_line_through_center_and_perpendicular :
  ∃ (l : ℝ → ℝ → Prop), line_through_center_and_perpendicular ∧ (∀ (x y : ℝ), l x y ↔ x - y + 3 = 0) := 
by
  sorry

end eq_of_line_through_center_and_perpendicular_l636_636078


namespace no_square_sum_l636_636202

theorem no_square_sum (x y : ℕ) (hxy_pos : 0 < x ∧ 0 < y)
  (hxy_gcd : Nat.gcd x y = 1)
  (hxy_perf : ∃ k : ℕ, x + 3 * y^2 = k^2) : ¬ ∃ z : ℕ, x^2 + 9 * y^4 = z^2 :=
by
  sorry

end no_square_sum_l636_636202


namespace camper_ratio_l636_636991

theorem camper_ratio (total_campers : ℕ) (G : ℕ) (B : ℕ)
  (h1: total_campers = 96) 
  (h2: G = total_campers / 3) 
  (h3: B = total_campers - G) 
  : B / total_campers = 2 / 3 :=
  by
    sorry

end camper_ratio_l636_636991


namespace probability_even_and_greater_than_14_l636_636250

theorem probability_even_and_greater_than_14 : 
  (∃ (s : finset (ℕ × ℕ)), 
    s = {(x, y) | x ∈ {1, 2, 3, 4, 5, 6, 7} ∧ y ∈ {1, 2, 3, 4, 5, 6, 7} ∧
    x * y % 2 = 0 ∧ x * y > 14}) → 
  (↑(∃ (t : ℕ), t = 16) / 49 : ℚ) :=
sorry

end probability_even_and_greater_than_14_l636_636250


namespace probability_of_cooking_l636_636805

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636805


namespace divisibility_by_8_l636_636244

theorem divisibility_by_8 (a b c d : ℤ) :
  let N := 1000 * a + 100 * b + 10 * c + d in
  (8 ∣ N) ↔ (8 ∣ (4 * b + 2 * c + d)) :=
by
  sorry

end divisibility_by_8_l636_636244


namespace selling_price_per_sweater_correct_l636_636027

-- Definitions based on the problem's conditions
def balls_of_yarn_per_sweater := 4
def cost_per_ball_of_yarn := 6
def number_of_sweaters := 28
def total_gain := 308

-- Defining the required selling price per sweater
def total_cost_of_yarn : Nat := balls_of_yarn_per_sweater * cost_per_ball_of_yarn * number_of_sweaters
def total_revenue : Nat := total_cost_of_yarn + total_gain
def selling_price_per_sweater : ℕ := total_revenue / number_of_sweaters

theorem selling_price_per_sweater_correct :
  selling_price_per_sweater = 35 :=
  by
  sorry

end selling_price_per_sweater_correct_l636_636027


namespace sum_of_zeros_eq_fourteen_l636_636680

def transformed_parabola : ℝ → ℝ := fun x => -(x - 7)^2 + 7

theorem sum_of_zeros_eq_fourteen :
  let a := 7 + Real.sqrt 7 in
  let b := 7 - Real.sqrt 7 in
  a + b = 14 := 
by
  sorry

end sum_of_zeros_eq_fourteen_l636_636680


namespace find_selling_price_l636_636649

-- Define the parameters based on the problem conditions
constant cost_price : ℝ := 22
constant selling_price_original : ℝ := 38
constant sales_volume_original : ℝ := 160
constant price_reduction_step : ℝ := 3
constant sales_increase_step : ℝ := 120
constant daily_profit_target : ℝ := 3640

-- Define the function representing the sales volume as a function of price reduction
def sales_volume (x : ℝ) : ℝ :=
  sales_volume_original + (x / price_reduction_step) * sales_increase_step

-- Define the function representing the daily profit as a function of price reduction
def daily_profit (x : ℝ) : ℝ :=
  (selling_price_original - x - cost_price) * (sales_volume x)

-- State the main theorem: the new selling price ensuring the desired profit
theorem find_selling_price : ∃ x : ℝ, daily_profit x = daily_profit_target ∧ (selling_price_original - x = 29) :=
by
  sorry

end find_selling_price_l636_636649


namespace min_num_pos_announcements_l636_636422

def num_pos_announcements (x y: ℕ) (h1: x * (x - 1) = 90) (h2: y * (y - 1) + (10 - y) * (9 - y) = 42): Prop :=
  y = 4

theorem min_num_pos_announcements : ∃ x y: ℕ, (x * (x - 1) = 90) ∧ (y * (y - 1) + (10 - y) * (9 - y) = 42) ∧ y = 4 :=
  by
    use 10
    use 4
    split
    { -- proof for x * (x - 1) = 90
      sorry
    }
    split
    { -- proof for y * (y - 1) + (10 - y) * (9 - y) = 42
      sorry
    }
    { -- proof for y = 4
      refl
    }

end min_num_pos_announcements_l636_636422


namespace cube_root_fraction_l636_636443

theorem cube_root_fraction : 
  (∛(5 / (63 / 4))) = (∛20) / (∛63) :=
by sorry

end cube_root_fraction_l636_636443


namespace expected_adjacent_red_pairs_l636_636158

theorem expected_adjacent_red_pairs :
  ∀ (deck : list (fin 2)), -- A deck of 60 cards is composed of 30 red (0) and 30 black (1)
  list.length deck = 60 →
  (list.count 0 deck = 30 ∧ list.count 1 deck = 30) →
  (∀ (i : fin 60), deck.nth (i + 1 % 60) = deck.nth (i + 1)) →
  @expected_value (fin 60) {p : list (fin 2) // list.length p = 60 ∧ (list.count 0 p = 30 ∧ list.count 1 p = 30)}
    (λ d, list.sum (list.map (λ i, if(deck.nth i = 0 ∧ deck.nth (i + 1 % 60) = 0)
    then 1 else 0) (list.fin_range 60))) = (30 * 29) / 59 := sorry

end expected_adjacent_red_pairs_l636_636158


namespace probability_cooking_is_one_fourth_l636_636858
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636858


namespace original_grain_amount_l636_636975

def grain_spilled : ℕ := 49952
def grain_remaining : ℕ := 918

theorem original_grain_amount : grain_spilled + grain_remaining = 50870 :=
by
  sorry

end original_grain_amount_l636_636975


namespace tangent_line_intersects_x_axis_l636_636365

theorem tangent_line_intersects_x_axis :
  ∃ x : ℝ, 
  let C1 := (0, 0) 
  let r1 := 3
  let C2 := (20, 0)
  let r2 := 9
  -- assuming the tangent line intersects x-axis to the right of the origin
  (x > 0) ∧
  (let ratio := x / (20 - x) in ratio = r1 / r2) ∧
  (x = 5) :=
by
  let C1 := (0 : ℝ, 0 : ℝ)
  let r1 := 3 : ℝ
  let C2 := (20 : ℝ, 0 : ℝ)
  let r2 := 9 : ℝ
  use 5
  split
  { -- prove x > 0
    sorry
  }
  split
  { -- prove the ratio equation
    sorry
  }
  { -- prove that x = 5
    sorry
  }

end tangent_line_intersects_x_axis_l636_636365


namespace polygon_number_of_sides_l636_636147

theorem polygon_number_of_sides (n : ℕ) 
  (h₁ : (n - 2) * 180 = 3 * 360)
  (h₂ : (n - 2) * 180 = sum_of_interior_angles n)
  (h₃ : 360 = sum_of_exterior_angles n) : 
  n = 8 := 
by
  sorry

-- Definitions to satisfy conditions (These definitions depend on the specific context they're used in)
def sum_of_interior_angles (n : ℕ) : ℤ := (n - 2) * 180
def sum_of_exterior_angles (n : ℕ) : ℤ := 360

end polygon_number_of_sides_l636_636147


namespace rotation_center_exists_l636_636494

noncomputable def f (z : ℂ) : ℂ :=
  ((-2 + 2 * Complex.i) * z + (-3 * Real.sqrt 3 - 15 * Complex.i)) / 3

theorem rotation_center_exists :
  let c : ℂ := (Complex.mk ((-15 * Real.sqrt 3 + 30) / 29) ((-6 * Real.sqrt 3 - 75) / 29)) in
  f c = c :=
by
  sorry

end rotation_center_exists_l636_636494


namespace probability_of_selecting_cooking_is_one_fourth_l636_636874

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636874


namespace lilies_per_centerpiece_l636_636626

def centerpieces := 6
def roses_per_centerpiece := 8
def orchids_per_rose := 2
def total_flowers := 120
def ratio_roses_orchids_lilies_centerpiece := 1 / 2 / 3

theorem lilies_per_centerpiece :
  ∀ (c : ℕ) (r : ℕ) (o : ℕ) (l : ℕ),
  c = centerpieces → r = roses_per_centerpiece →
  o = orchids_per_rose * r →
  total_flowers = 6 * (r + o + l) →
  ratio_roses_orchids_lilies_centerpiece = r / o / l →
  l = 10 := by sorry

end lilies_per_centerpiece_l636_636626


namespace optimal_salary_l636_636167

noncomputable def net_salary (x : ℝ) : ℝ := x - (x ^ 2 / 1000)

theorem optimal_salary : ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → net_salary y ≤ net_salary 500) :=
by
  use 500
  split
  · linarith
  · intro y hy
    sorry

end optimal_salary_l636_636167


namespace xiaoming_selects_cooking_probability_l636_636831

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636831


namespace triangle_properties_l636_636149

noncomputable def angle_A (B C : ℝ) : ℝ :=
  π - (B + C)

noncomputable def side_a (b c : ℝ) (A : ℝ) : ℝ :=
  real.sqrt (b^2 + c^2 - 2 * b * c * real.cos A)

noncomputable def area (b c : ℝ) (A : ℝ) : ℝ :=
  (1 / 2) * b * c * real.sin A

theorem triangle_properties
  (b c : ℝ)
  (cos_BC : ℝ)
  (h_b : b = 2)
  (h_c : c = 4)
  (h_cos_BC : cos_BC = -1 / 2)
  (B C A : ℝ)
  (h_A : A = angle_A B C)
  (h_cos_A : real.cos A = 1 / 2)
  (a : ℝ)
  (h_a : a = side_a b c A)
  (S : ℝ)
  (h_S : S = area b c A) :
  A = π / 3 ∧ a = 2 * real.sqrt 3 ∧ S = 2 * real.sqrt 3 := by {
  sorry
}

end triangle_properties_l636_636149


namespace parallel_segments_between_parallel_planes_equal_l636_636062

-- Definitions for our Lean statement
def parallel_planes (P1 P2 : set (set ℝ³)) : Prop := ∀ p1 ∈ P1, ∀ p2 ∈ P2, ∃ v, p2 - p1 = v
def parallel_segments (S1 S2 : set (ℝ³ × ℝ³)) : Prop := ∀ s1 ∈ S1, ∀ s2 ∈ S2, ∃ v, s2 - s1 = v

-- The theorem statement
theorem parallel_segments_between_parallel_planes_equal {P1 P2 S1 S2 : set (set ℝ³)} :
  parallel_planes P1 P2 → 
  parallel_segments S1 S2 → 
  (∀ s ∈ S1, ∀ p ∈ P1, s ∈ p → ∀ t ∈ S2, ∀ q ∈ P2, t ∈ q → s = t) :=
by 
  intro h1 h2,
  sorry

end parallel_segments_between_parallel_planes_equal_l636_636062


namespace frog_distribution_l636_636013

theorem frog_distribution (n : ℕ) (hn : n ≥ 5) (frogs : ℕ) (hfrogs : frogs = 4 * n + 1) 
  (cells : fin (2 * n) → fin 4) (adj : ∀ c : fin (2 * n), ∃ (a b d : fin (2 * n)), a ≠ b ∧ b ≠ d ∧ d ≠ a ∧ adj c a ∧ adj c b ∧ adj c d)
  (explosion : ∀ c : fin (2 * n), ∃ f : ℕ, f ≥ 3 → ∀ t : set (fin (2 * n)), c ∈ t → |t| ≥ 3 → explodes f t)
  :
  ∀ c : fin (2 * n), (∃ f : ℕ, f = 1) ∨ (∃ na nb nc : fin (2 * n), adj c na ∧ adj c nb ∧ adj c nc ∧ ((∃ n : ℕ, n = 1) ∨ (∃ nf1 nf2 nf3 : fin (2 * n), adj na nf1 ∧ adj nb nf2 ∧ adj nc nf3)))
  :=
sorry

end frog_distribution_l636_636013


namespace f_at_5_point_5_l636_636369

noncomputable def f : ℝ → ℝ
| x@(0 < x < 1) := 4^x
| x@(1 <= x) := 2 * f (x - 1)
| x@_ := 0

theorem f_at_5_point_5 :
  f(5.5) = 64 :=
sorry

end f_at_5_point_5_l636_636369


namespace find_a_plus_b_l636_636605

open Complex

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∃ (r1 r2 r3 : ℂ),
     r1 = 1 + I * Real.sqrt 3 ∧
     r2 = 1 - I * Real.sqrt 3 ∧
     r3 = -2 ∧
     (r1 + r2 + r3 = 0) ∧
     (r1 * r2 * r3 = -b) ∧
     (r1 * r2 + r2 * r3 + r3 * r1 = -a))

theorem find_a_plus_b (a b : ℝ) (h : problem_statement a b) : a + b = 8 :=
sorry

end find_a_plus_b_l636_636605


namespace first_reach_32_l636_636281

-- Define the height function y in terms of t
def height (t : ℝ) : ℝ := -4.9 * t^2 + 29.4 * t

-- Define the condition at y = 32 meters
def condition (t : ℝ) : Prop := height t = 32

-- Define the theorem statement to prove t = 8/7 is the first time the height reaches 32 meters
theorem first_reach_32 : ∀ t₁ t₂ : ℝ,
  condition t₁ → condition t₂ → (0 ≤ t₁) → (0 ≤ t₂) → t₁ ≤ t₂ → t₁ = 8/7 :=
by
  intro t₁ t₂ h1 h2 h3 h4 h5
  sorry

end first_reach_32_l636_636281


namespace eagles_win_probability_l636_636268

/-- 
  The Eagles play the Sharks in a series of nine basketball games.
  Each team has an equal chance of winning each game.
  Prove that the probability the Eagles will win at least five games is 1/2.
-/
theorem eagles_win_probability (n : ℕ) (p : ℚ) (k : ℕ) (h_n : n = 9) (h_p : p = 1/2) (h_k : k = 5) :
  (∑ i in finset.range (n + 1), if i >= k then nat.choose n i * p ^ i * (1 - p) ^ (n - i) else 0) =
  1 / 2 :=
by sorry

end eagles_win_probability_l636_636268


namespace find_selling_price_l636_636648

-- Define the parameters based on the problem conditions
constant cost_price : ℝ := 22
constant selling_price_original : ℝ := 38
constant sales_volume_original : ℝ := 160
constant price_reduction_step : ℝ := 3
constant sales_increase_step : ℝ := 120
constant daily_profit_target : ℝ := 3640

-- Define the function representing the sales volume as a function of price reduction
def sales_volume (x : ℝ) : ℝ :=
  sales_volume_original + (x / price_reduction_step) * sales_increase_step

-- Define the function representing the daily profit as a function of price reduction
def daily_profit (x : ℝ) : ℝ :=
  (selling_price_original - x - cost_price) * (sales_volume x)

-- State the main theorem: the new selling price ensuring the desired profit
theorem find_selling_price : ∃ x : ℝ, daily_profit x = daily_profit_target ∧ (selling_price_original - x = 29) :=
by
  sorry

end find_selling_price_l636_636648


namespace find_f_correct_l636_636678

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_con1 : ∀ x : ℝ, 2 * f x + f (-x) = 2 * x

theorem find_f_correct : ∀ x : ℝ, f x = 2 * x :=
by
  sorry

end find_f_correct_l636_636678


namespace complementary_events_l636_636333

def possible_outcomes := {0, 1, 2, 3}

def no_more_than_one_head (n : ℕ) := n ∈ {0, 1}
def at_least_two_heads (n : ℕ) := n ∈ {2, 3}

theorem complementary_events :
  (∀ n, n ∈ possible_outcomes → no_more_than_one_head n ∨ at_least_two_heads n) ∧
  (∀ n, no_more_than_one_head n → ¬ at_least_two_heads n) ∧
  (∀ n, at_least_two_heads n → ¬ no_more_than_one_head n) :=
by sorry

end complementary_events_l636_636333


namespace probability_of_selecting_cooking_l636_636813

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636813


namespace real_roots_iff_sum_of_squares_l636_636075

theorem real_roots_iff_sum_of_squares (f g h : Polynomial ℝ) (hf : f ∈ Polynomial ℝ)
  (hfg : f^2 = g^2 + h^2) (hdiv : ¬ (f ∣ g)) : 
  (∀ x, IsRoot f x → x ∈ ℝ) ↔ ¬ (∃ g h : Polynomial ℝ, f^2 = g^2 + h^2 ∧ ¬ (f ∣ g)) :=
sorry

end real_roots_iff_sum_of_squares_l636_636075


namespace probability_of_selecting_cooking_l636_636810

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636810


namespace probability_of_selecting_cooking_l636_636783

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636783


namespace length_of_AB_is_8_l636_636506

-- Definition of the parabola equation
def parabola (x y : ℝ) : Prop := x ^ 2 = -4 * y

-- Definition of the line equation
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Definition of the intersection points A and B of parabola and line
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ line A.1 A.2 ∧ parabola B.1 B.2 ∧ line B.1 B.2 ∧ A ≠ B

-- The length of line segment AB
def length_segment (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

-- The proof statement: The length of the segment is 8
theorem length_of_AB_is_8 (A B : ℝ × ℝ) (h : intersection_points A B) : length_segment A B = 8 :=
sorry

end length_of_AB_is_8_l636_636506


namespace tennis_ball_ratio_problem_solution_l636_636385

def tennis_ball_ratio_problem (total_balls ordered_white ordered_yellow dispatched_yellow extra_yellow : ℕ) : Prop :=
  total_balls = 114 ∧ 
  ordered_white = total_balls / 2 ∧ 
  ordered_yellow = total_balls / 2 ∧ 
  dispatched_yellow = ordered_yellow + extra_yellow → 
  (ordered_white / dispatched_yellow = 57 / 107)

theorem tennis_ball_ratio_problem_solution :
  tennis_ball_ratio_problem 114 57 57 107 50 := by 
  sorry

end tennis_ball_ratio_problem_solution_l636_636385


namespace shell_arrangement_symmetry_l636_636594

/-- John draws a regular ten-pointed star in the sand, with 10 outward-pointing 
    and 10 inward-pointing points. He places one of twenty different sea shells 
    at each of these 20 points. This proof shows that the number of distinct 
    ways to place the shells, considering rotations and reflections of the 
    arrangement being equivalent, is 19! -/
theorem shell_arrangement_symmetry : 
  ∃ (star : Type) (shells : Type) (arrangements : star → shells → ℕ),
  (forall (rot : star → star) (ref : star → star), (rotations star = 10) → (reflections star = 2) → 
  (total_arrangements shells = 20!)) → 
  (distinct_arrangements shells = 19!) :=
sorry

end shell_arrangement_symmetry_l636_636594


namespace distance_parallel_lines_l636_636050

noncomputable def distance_between_parallel_lines (A B : Real) (C D : Real) : Real :=
  abs(A - C) / sqrt(1 + B^2)

theorem distance_parallel_lines :
  distance_between_parallel_lines 5 (-3) (-4) (-3) = 9 * sqrt 10 / 10 := by
  sorry

end distance_parallel_lines_l636_636050


namespace probability_cooking_selected_l636_636921

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636921


namespace find_y_value_l636_636120

theorem find_y_value (k : ℝ) (h1 : ∀ (x : ℝ), y = k * x) 
(h2 : y = 4 ∧ x = 2) : 
(∀ (x : ℝ), x = -2 → y = -4) := 
by 
  sorry

end find_y_value_l636_636120


namespace probability_of_selecting_cooking_l636_636930

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636930


namespace sum_of_n_values_l636_636749

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l636_636749


namespace remaining_perimeter_eq_eight_l636_636004

-- Conditions
variables (A B C D E : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables [equilateral A B C] [equilateral D B E]
variable (DB EB : ℝ) (sideABC : ℝ)

-- Given conditions
def side_length_DB_EB_eq_one : Prop := DB = 1 ∧ EB = 1
def side_length_ABC_eq_three : Prop := sideABC = 3

-- Quadrilateral perimeters
def perimeter_ACED : ℝ := 3 + 2 + 2 + 1

-- Proof problem
theorem remaining_perimeter_eq_eight (h1 : side_length_DB_EB_eq_one DB EB) (h2 : side_length_ABC_eq_three sideABC) : 
  perimeter_ACED = 8 :=
  sorry

end remaining_perimeter_eq_eight_l636_636004


namespace area_increase_by_40_percent_l636_636698

theorem area_increase_by_40_percent (s : ℝ) : 
  let A1 := s^2 
  let new_side := 1.40 * s 
  let A2 := new_side^2 
  (A2 - A1) / A1 * 100 = 96 := 
by 
  sorry

end area_increase_by_40_percent_l636_636698


namespace probability_of_selecting_cooking_l636_636940

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636940


namespace carter_plants_l636_636026

theorem carter_plants (packets_A : ℕ) (seeds_per_packet_A : ℕ) (growth_days_A : ℕ)
                      (packets_B : ℕ) (seeds_per_packet_B : ℕ) (growth_days_B : ℕ)
                      (packets_C : ℕ) (seeds_per_packet_C : ℕ) (growth_days_C : ℕ)
                      (total_plants_A : ℕ) (total_plants_B : ℕ) (total_plants_C : ℕ) :
  (packets_A = 2) → (seeds_per_packet_A = 3) → (growth_days_A = 5) →
  (packets_B = 3) → (seeds_per_packet_B = 6) → (growth_days_B = 7) →
  (packets_C = 3) → (seeds_per_packet_C = 9) → (growth_days_C = 4) →
  (total_plants_A = 12) → (total_plants_B = 12) → (total_plants_C = 12) →
  (let initial_plants_A := packets_A * seeds_per_packet_A in
   let initial_plants_B := packets_B * seeds_per_packet_B in
   let initial_plants_C := packets_C * seeds_per_packet_C in
   let additional_plants_A := total_plants_A - initial_plants_A in
   let packets_needed_A := additional_plants_A / seeds_per_packet_A in
   let days_needed_A := growth_days_A in
   (initial_plants_B >= total_plants_B) ∧ 
   (initial_plants_C >= total_plants_C) ∧
   (packets_needed_A = 2) ∧
   (days_needed_A = 5)) :=
by
  intros
  sorry

end carter_plants_l636_636026


namespace integral_eq_sqrt_f_l636_636767

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + (Real.sqrt 2 - 1) * x) ^ 2

theorem integral_eq_sqrt_f (f : ℝ → ℝ)
  (cont : Continuous f)
  (positive : ∀ x, 0 ≤ x → 0 < f x)
  (f1 : f 1 = 1 / 2) :
  (∀ x, ∫ t in 0..x, f t = x * Real.sqrt (f x)) :=
by
  sorry

end integral_eq_sqrt_f_l636_636767


namespace number_of_squares_l636_636635

def side_plywood : ℕ := 50
def side_square_1 : ℕ := 10
def side_square_2 : ℕ := 20
def total_cut_length : ℕ := 280

/-- Number of squares obtained given the side lengths of the plywood and the cut lengths -/
theorem number_of_squares (x y : ℕ) (h1 : 100 * x + 400 * y = side_plywood^2)
  (h2 : 40 * x + 80 * y = total_cut_length) : x + y = 16 :=
sorry

end number_of_squares_l636_636635


namespace time_a_is_390_l636_636552

noncomputable def time_to_finish (V_a V_b V_c : ℝ) (T_a T_b : ℝ) :=
  V_a * T_a = 1000 ∧ V_b * T_a = 975 ∧ V_b * (T_a + 10) = 1000 ∧
  V_a * T_a = 1000 ∧ V_c * T_a = 960 ∧ V_c * (T_a + 8) = 1000 ∧
  V_b * T_b = 1000 ∧ V_c * T_b = 985 ∧ V_c * (T_b + 2) = 1000

theorem time_a_is_390 (V_a V_b V_c : ℝ) (T_a T_b : ℝ):
  time_to_finish V_a V_b V_c T_a T_b → T_a = 390 := 
by 
  intro h,
  cases h with _ h₁,
  cases h₁ with _ h₂,
  cases h₂ with _ h₃,
  cases h₃ with _ h₄,
  cases h₄ with _ h₅,
  cases h₅ with _ h₆,
  cases h₆ with _ h₇,
  cases h₇ with _ h₈,
  sorry

end time_a_is_390_l636_636552


namespace problem_statement_l636_636604

open Complex

theorem problem_statement (a b : ℝ) (h : (1 + (I : ℂ) * (sqrt 3))^3 + a * (1 + (I * sqrt 3)) + b = 0) :
  a + b = 8 := 
sorry

end problem_statement_l636_636604


namespace part_I_part_II_l636_636503

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
noncomputable def g (x : ℝ) := |2 * x - 1|

theorem part_I (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end part_I_part_II_l636_636503


namespace probability_cooking_selected_l636_636914

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636914


namespace arithmetic_sequence_sum_9_is_36_l636_636107

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = r * (a n)
noncomputable def Sn (b : ℕ → ℝ) (n : ℕ) : ℝ := n * (b 1 + b n) / 2

theorem arithmetic_sequence_sum_9_is_36 (a b : ℕ → ℝ) (h_geom : geometric_sequence a) 
    (h_cond : a 4 * a 6 = 2 * a 5) (h_b5 : b 5 = 2 * a 5) : Sn b 9 = 36 :=
by
  sorry

end arithmetic_sequence_sum_9_is_36_l636_636107


namespace fruit_selling_price_3640_l636_636653

def cost_price := 22
def initial_selling_price := 38
def initial_quantity_sold := 160
def price_reduction := 3
def quantity_increase := 120
def target_profit := 3640

theorem fruit_selling_price_3640 (x : ℝ) :
  ((initial_selling_price - x - cost_price) * (initial_quantity_sold + (x / price_reduction) * quantity_increase) = target_profit) →
  x = 9 →
  initial_selling_price - x = 29 :=
by
  intro h1 h2
  sorry

end fruit_selling_price_3640_l636_636653


namespace negation_of_cosine_inequality_l636_636685

theorem negation_of_cosine_inequality :
  ¬ (∀ x : ℝ, cos x ≤ 1) ↔ ∃ x : ℝ, cos x > 1 :=
sorry

end negation_of_cosine_inequality_l636_636685


namespace min_abs_difference_ab_l636_636529

theorem min_abs_difference_ab (a b : ℕ) (h : a > 0 ∧ b > 0 ∧ ab + 5 * a - 2 * b = 193) : 
  abs (a - b) = 15 :=
sorry

end min_abs_difference_ab_l636_636529


namespace remainder_a6_mod_n_eq_1_l636_636612

theorem remainder_a6_mod_n_eq_1 
  (n : ℕ) (a : ℤ) (h₁ : n > 0) (h₂ : a^3 ≡ 1 [MOD n]) : a^6 ≡ 1 [MOD n] := 
by 
  sorry

end remainder_a6_mod_n_eq_1_l636_636612


namespace probability_cooking_selected_l636_636925

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636925


namespace find_a_b_l636_636543

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x + 2 > a ∧ x - 1 < b) ↔ (1 < x ∧ x < 3)) → a = 3 ∧ b = 2 :=
by
  intro h
  sorry

end find_a_b_l636_636543


namespace part1_part1_monotonicity_intervals_part2_l636_636111

noncomputable def f (x a : ℝ) := x * Real.log x - a * (x - 1)^2 - x + 1

-- Part 1: Monotonicity and Extreme values when a = 0
theorem part1 (x : ℝ) : f x 0 = x * Real.log x - x + 1 := sorry

theorem part1_monotonicity_intervals (x : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x < 1 → f x 0 < f 1 0) ∧
  (∀ (x : ℝ), x > 1 → f 1 0 < f x 0) ∧ 
  (f 1 0 = 0) := sorry

-- Part 2: f(x) < 0 for x > 1 and a >= 1/2
theorem part2 (x a : ℝ) (hx : x > 1) (ha : a ≥ 1/2) : f x a < 0 := sorry

end part1_part1_monotonicity_intervals_part2_l636_636111


namespace cyclic_sum_inequality_l636_636074

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∑ cyc in {a, b, c}, (λ x y z : ℝ, (2*x + y + z)^2 / (2*x^2 + (y + z)^2)) a b c ≤ 8 := 
sorry

end cyclic_sum_inequality_l636_636074


namespace larger_number_is_1671_l636_636274

variable (L S : ℕ)

noncomputable def problem_conditions :=
  L - S = 1395 ∧ L = 6 * S + 15

theorem larger_number_is_1671 (h : problem_conditions L S) : L = 1671 := by
  sorry

end larger_number_is_1671_l636_636274


namespace sum_of_solutions_of_absolute_value_l636_636723

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l636_636723


namespace three_digit_number_equality_l636_636444

theorem three_digit_number_equality :
  ∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧
  (100 * x + 10 * y + z = x^2 + y + z^3) ∧
  (100 * x + 10 * y + z = 357) :=
by
  sorry

end three_digit_number_equality_l636_636444


namespace no_solution_m_4_l636_636143

theorem no_solution_m_4 (m : ℝ) : 
  (¬ ∃ x : ℝ, 2/x = m/(2*x + 1)) → m = 4 :=
by
  sorry

end no_solution_m_4_l636_636143


namespace conversion_rate_false_l636_636273

-- Definition of conversion rates between units
def conversion_rate_hour_minute : ℕ := 60
def conversion_rate_minute_second : ℕ := 60

-- Theorem stating that the rate being 100 is false under the given conditions
theorem conversion_rate_false (h1 : conversion_rate_hour_minute = 60) 
  (h2 : conversion_rate_minute_second = 60) : 
  ¬ (conversion_rate_hour_minute = 100 ∧ conversion_rate_minute_second = 100) :=
by {
  sorry
}

end conversion_rate_false_l636_636273


namespace problem_statement_l636_636495

-- Define the function f based on the given condition
def f : ℝ → ℝ := λ y, ∃ x : ℝ, y = sin x ∧ f y = cos (2 * x)

-- Define the specific value of interest and the trigonometric results
def sin_30 := 1 / 2
def cos_60 := 1 / 2

-- State that if y corresponds to sin(30 degrees), then f(y) should be cos(60 degrees)
theorem problem_statement : f (1 / 2) = 1 / 2 :=
  sorry

end problem_statement_l636_636495


namespace chord_length_difference_l636_636316

theorem chord_length_difference 
  (r₁ r₂ : ℝ) 
  (h₁ : r₁ = 5) 
  (h₂ : r₂ = 26) 
  (h₃ : ∃ C₁ C₂ : Point ℝ, dist C₁ C₂ = r₁ + r₂) :
  ∃ L S : ℝ, 
    L = 2 * r₂ ∧ 
    S = 2 * Real.sqrt (r₂^2 - (r₂ - r₁)^2) ∧
    (L - S) = 52 - 2 * Real.sqrt 235 :=
by
  sorry

end chord_length_difference_l636_636316


namespace construct_triangle_l636_636721

-- Define the given conditions
def given_side : Type := ℝ  -- Length of the side AB
def given_angle : Type := ℝ  -- Value of the angle in degrees or radians

-- The side of the triangle and the two adjacent angles
variables (AB : given_side) (alpha beta : given_angle)

-- Define a structure to represent a triangle
structure Triangle :=
  (A B C : Point)
  (AB_len : dist A B = AB)
  (angle_at_A : angle B A C = alpha)
  (angle_at_B : angle A B C = beta)

-- The theorem statement
theorem construct_triangle (AB : given_side) (alpha beta : given_angle) :
  ∃ (T : Triangle), T.AB_len = AB ∧ T.angle_at_A = alpha ∧ T.angle_at_B = beta := 
sorry

end construct_triangle_l636_636721


namespace speed_increase_percentage_l636_636589

variable (T : ℚ)  -- usual travel time in minutes
variable (v : ℚ)  -- usual speed

-- Conditions
-- Ivan usually arrives at 9:00 AM, traveling for T minutes at speed v.
-- When Ivan leaves 40 minutes late and drives 1.6 times his usual speed, he arrives at 8:35 AM
def usual_arrival_time : ℚ := 9 * 60  -- 9:00 AM in minutes

def time_when_late : ℚ := (9 * 60) + 40 - (25 + 40)  -- 8:35 AM in minutes

def increased_speed := 1.6 * v -- 60% increase in speed

def time_taken_with_increased_speed := T - 65

theorem speed_increase_percentage :
  ((T / (T - 40)) = 1.3) :=
by
-- assume the equation for usual time T in terms of increased speed is known
-- Use provided conditions and solve the equation to derive the result.
  sorry

end speed_increase_percentage_l636_636589


namespace length_AB_eq_l636_636517

variables {ℝ : Type*} [inner_product_space ℝ ℝ] -- assuming ℝ is a real inner product space
variables (l1 l2 : set ℝ) (A B C D : ℝ) (n : ℝ)

def is_perpendicular (v1 v2 : ℝ) : Prop := inner v1 v2 = 0

noncomputable def mod (v : ℝ) : ℝ := real.sqrt (inner v v)

axiom perpendicular_segment (l1 l2 : set ℝ) (A B : ℝ) : is_perpendicular A B

axiom point_on_line1 (C : ℝ) (l1 : set ℝ) : C ∈ l1
axiom point_on_line2 (D : ℝ) (l2 : set ℝ) : D ∈ l2

axiom vector_along_segment (A B : ℝ) (n : ℝ) : ∃ t : ℝ, n = t * (B - A)

theorem length_AB_eq (l1 l2 : set ℝ) (A B C D : ℝ) (n : ℝ) 
  (h1 : is_perpendicular A B) 
  (h2 : C ∈ l1) 
  (h3 : D ∈ l2) 
  (h4 : ∃ t : ℝ, n = t * (B - A)) : 
  mod (B - A) = |inner (C - D) n| / mod n := sorry

end length_AB_eq_l636_636517


namespace speed_increase_needed_l636_636586

-- Definitions based on the conditions
def usual_speed := ℝ
def usual_travel_time := ℝ -- in minutes
def late_departure := 40   -- in minutes
def increased_speed_factor := 1.6
def early_arrival := 9*60 - (8*60 + 35) -- 25 minutes (from 9:00 AM to 8:35 AM)

-- The problem statement in Lean 4
theorem speed_increase_needed (v : usual_speed) (T : usual_travel_time) :
  let T_late := T + late_departure in
  let T_increased_speed := (T / increased_speed_factor) in
  T = usual_travel_time →
  v = usual_speed →
  T_late - T_increased_speed = late_departure + early_arrival → 
  (T - late_departure) / T = 1 / (1 - 40 / (T * (1 / (1.6)))) →
  (v * (3 / 4)) / v = 1.3 := 
sorry

end speed_increase_needed_l636_636586


namespace arithmetic_square_root_of_sqrt_16_l636_636771

theorem arithmetic_square_root_of_sqrt_16 :
  (real.sqrt 16) = 4 :=
sorry

end arithmetic_square_root_of_sqrt_16_l636_636771


namespace probability_cooking_selected_l636_636916

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636916


namespace geom_series_first_term_l636_636405

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l636_636405


namespace probability_of_cooking_l636_636793

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636793


namespace problem_statement_l636_636615

variable {a b c : ℝ}
variable (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
variable (h₄ : a * b * c = 1)
variable (h₅ : a + b + c = 9)
variable (h₆ : a * b + b * c + c * a = 11)

noncomputable def s : ℝ := Real.sqrt a + Real.sqrt b + Real.sqrt c

theorem problem_statement : s ^ 4 - 18 * s ^ 2 - 8 * s = -37 :=
  sorry

end problem_statement_l636_636615


namespace hundreds_digit_even_l636_636306

-- Define the given conditions
def units_digit (n : ℕ) : ℕ := n % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- The main theorem to prove
theorem hundreds_digit_even (x : ℕ) 
  (h1 : units_digit (x*x) = 9) 
  (h2 : tens_digit (x*x) = 0) : ((x*x) / 100) % 2 = 0 :=
  sorry

end hundreds_digit_even_l636_636306


namespace num_people_price_item_equation_l636_636574

theorem num_people_price_item_equation
  (x y : ℕ)
  (h1 : 8 * x = y + 3)
  (h2 : 7 * x = y - 4) :
  (y + 3) / 8 = (y - 4) / 7 :=
by
  sorry

end num_people_price_item_equation_l636_636574


namespace hyperbola_equation_l636_636701

theorem hyperbola_equation (a b : ℝ) (hyp1 : (1/a^2) - (1/(b^2)) = 1)
  (hyp2 : b = a * Real.sqrt 2):
  (x y : ℝ) → (x, y) = (1, 1) →
  (x^2 / (1 / 2) - y^2 = 1) ∨ (y^2 / (1 / 2) - x^2 = 1) :=
by
  sorry

end hyperbola_equation_l636_636701


namespace probability_of_selecting_cooking_l636_636949

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636949


namespace maximize_area_ratio_l636_636109

theorem maximize_area_ratio (t : ℝ) (h_t : t ≠ 0)
  (ellipse : ∀ x y : ℝ, x^2 / 4 + y^2 = 1)
  (upper_vertex : ∃ M : ℝ × ℝ, M = (0, 1))
  (lower_vertex : ∃ N : ℝ × ℝ, N = (0, -1))
  (point_T : ∃ T : ℝ × ℝ, T = (t, 2)) 
  (lines_TM_TN : ∀ E F : ℝ × ℝ, 
    (∃ line_TM : ℝ → ℝ, line_TM x = x / t + 1 → line_TM E.1 = E.2) ∧ 
    (∃ line_TN : ℝ → ℝ, line_TN x = -3 * x / t + 3 → line_TN F.1 = F.2)
  )
  (areas : ∃ k : ℝ, area_of_triangle (t, 2) (0, 1) (0, -1) = k * area_of_triangle (t, 2) E F)
  : t = 2 * Real.sqrt 3 ∨ t = -2 * Real.sqrt 3 := 
sorry

end maximize_area_ratio_l636_636109


namespace largest_of_five_consecutive_integers_l636_636455

theorem largest_of_five_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120) : n + 4 = 9 :=
sorry

end largest_of_five_consecutive_integers_l636_636455


namespace avg_salary_difference_l636_636162

theorem avg_salary_difference (factory_payroll : ℕ) (factory_workers : ℕ) (office_payroll : ℕ) (office_workers : ℕ)
  (h1 : factory_payroll = 30000) (h2 : factory_workers = 15)
  (h3 : office_payroll = 75000) (h4 : office_workers = 30) :
  (office_payroll / office_workers) - (factory_payroll / factory_workers) = 500 := by
  sorry

end avg_salary_difference_l636_636162


namespace angle_CED_equals_53_l636_636203

noncomputable theory

variables {O : Type} [circle_center O] {B E F C D : point O}

-- Geometric conditions
variable (AB_diameter := is_diameter B O)
variable (E_on_circle := on_circle E O)
variable (EF_diameter := is_diameter E F O)
variable (F_opposite_E := opposite_points_on_circle E F O)
variable (tangent_B_C := intersects_tangent B C E)
variable (tangent_E_D := intersects_tangent E D E)
variable (tangent_F_G := intersects_tangent F B)
variable (angle_BAE := angle B A E = 37)

-- Goal to prove
theorem angle_CED_equals_53 :
  angle C E D = 53 :=
sorry

end angle_CED_equals_53_l636_636203


namespace range_of_a_l636_636117

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2 - 2 * x

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x - a * x - 1

theorem range_of_a
  (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f_prime x1 a = 0 ∧ f_prime x2 a = 0) ↔
  0 < a ∧ a < Real.exp (-2) :=
sorry

end range_of_a_l636_636117


namespace sum_of_n_values_l636_636748

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l636_636748


namespace range_of_t_l636_636676

theorem range_of_t (t : ℝ) (h : ∀ n : ℕ, 0 < n → 
  (∑ i in finset.range n, (-1) ^ (i + 1) * (2 * i + 1) * (2 * i + 3)) ≥ t * n^2) : 
  t ≤ -12 :=
by sorry

end range_of_t_l636_636676


namespace probability_cooking_selected_l636_636919

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636919


namespace total_right_handed_is_59_l636_636707

def football_team := 70
def throwers := 37
def non_throwers := football_team - throwers
def left_handed_non_throwers := non_throwers / 3
def right_handed_non_throwers := non_throwers - left_handed_non_throwers
def right_handed_throwers := throwers
def total_right_handed := right_handed_throwers + right_handed_non_throwers

theorem total_right_handed_is_59 : total_right_handed = 59 :=
by
  -- Provided in solution
  have h1 : non_throwers = 33 := by rfl
  have h2 : left_handed_non_throwers = 11 := by rw [← h1]; exact (by norm_num : 33 / 3 = 11)
  have h3 : right_handed_non_throwers = 22 := by rw [← h1, ← h2]; exact (by norm_num : 33 - 11 = 22)
  have h4 : right_handed_throwers = 37 := by rfl
  have h5 : total_right_handed = right_handed_throwers + right_handed_non_throwers := by rfl
  rw [h4, h3] at h5
  exact (by norm_num : 37 + 22 = 59)

end total_right_handed_is_59_l636_636707


namespace soil_cost_calculation_l636_636016

variables (num_rose_bushes : ℕ) (cost_per_rose_bush : ℕ)
          (gardener_rate : ℕ) (gardener_hours_per_day : ℕ)
          (num_days : ℕ) (soil_needed : ℕ)
          (total_project_cost : ℕ)

def cost_of_rose_bushes := num_rose_bushes * cost_per_rose_bush
def gardener_hours := gardener_hours_per_day * num_days
def cost_of_gardener := gardener_hours * gardener_rate

def cost_of_rose_bushes_and_gardener := cost_of_rose_bushes + cost_of_gardener
def cost_of_soil := total_project_cost - cost_of_rose_bushes_and_gardener
def cost_per_cubic_foot_of_soil := cost_of_soil / soil_needed

theorem soil_cost_calculation (h1 : num_rose_bushes = 20) 
                              (h2 : cost_per_rose_bush = 150)
                              (h3 : gardener_rate = 30) 
                              (h4 : gardener_hours_per_day = 5) 
                              (h5 : num_days = 4) 
                              (h6 : soil_needed = 100) 
                              (h7 : total_project_cost = 4100) 
                              : cost_per_cubic_foot_of_soil = 5 :=
by {
  sorry
}

end soil_cost_calculation_l636_636016


namespace minimum_length_segment_MX_l636_636236

theorem minimum_length_segment_MX
  (A B C M O X : Type)
  (dist_AB : ℝ = 17)
  (dist_AC : ℝ = 30)
  (dist_BC : ℝ = 19)
  (midpoint_M : IsMidpoint B C M)
  (midpoint_O : IsMidpoint A B O)
  (circle : ℝ → Point → Bool)
  (X_on_circle : circle (dist_AB) X)
  : ∃ (MX_min : ℝ), MX_min = 6.5 := 
sorry

end minimum_length_segment_MX_l636_636236


namespace measure_angle_ACB_l636_636086

-- Define the given setup and conditions
variables {A B C K P Q : Type}
variables [triangle ABC] -- Given that ABC is a triangle
variables (angle_A : ∠A = 42) -- ∠A = 42°
variables (AB_lt_AC : AB < AC)
variables (K_on_AC : K ∈ AC) -- K is a point on AC such that AB = CK
variables (AB_eq_CK : AB = CK)
variables (P_mid_AK : is_midpoint P A K) -- P is the midpoint of AK
variables (Q_mid_BC : is_midpoint Q B C) -- Q is the midpoint of BC
variables (angle_PQC : ∠PQC = 110) -- Given that ∠PQC = 110°

-- Define the proof goal
theorem measure_angle_ACB : ∠ACB = 49 := 
begin
  sorry -- Proof to be completed
end

end measure_angle_ACB_l636_636086


namespace equal_pair_b_l636_636009

def exprA1 := -3^2
def exprA2 := -2^3

def exprB1 := -6^3
def exprB2 := (-6)^3

def exprC1 := -6^2
def exprC2 := (-6)^2

def exprD1 := (-3 * 2)^2
def exprD2 := (-3) * 2^2

theorem equal_pair_b : exprB1 = exprB2 :=
by {
  -- proof steps should go here
  sorry
}

end equal_pair_b_l636_636009


namespace modulus_of_complex_number_l636_636142

theorem modulus_of_complex_number : 
  let z := (1 - complex.i) * (2 - complex.i) / (1 + 2 * complex.i) in
  complex.abs z = real.sqrt 2 :=
by
  sorry

end modulus_of_complex_number_l636_636142


namespace probability_of_cooking_l636_636801

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636801


namespace decimal_to_fraction_denominator_l636_636668

theorem decimal_to_fraction_denominator :
  let S := (70 : ℚ) / 90 in 
  denominator (S) = 9 := 
by
  -- Define the repeating decimal as a fraction
  have hS : S = 7 / 9 := by norm_num
  -- Show that its denominator is 9
  exact hS.symm ▸ rfl

end decimal_to_fraction_denominator_l636_636668


namespace probability_of_selecting_cooking_l636_636948

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636948


namespace find_value_of_3x1_squared_plus_2x2_l636_636488

-- Define the conditions
def quadratic_roots (a b c : ℝ) (x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0

-- Define the given constants and roots for our specific equation
def a : ℝ := 3
def b : ℝ := -2
def c : ℝ := -4
def x1 : ℝ
def x2 : ℝ

-- Lean theorem statement reflecting the proof problem
theorem find_value_of_3x1_squared_plus_2x2
  (h : quadratic_roots a b c x1 x2) :
  3 * x1^2 + 2 * x2 = 16 / 3 := sorry

end find_value_of_3x1_squared_plus_2x2_l636_636488


namespace probability_of_selecting_cooking_l636_636934

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636934


namespace total_miles_cycled_l636_636185

theorem total_miles_cycled (D : ℝ) (store_to_peter : ℝ) (same_speed : Prop)
  (time_relation : 2 * D = store_to_peter) (store_to_peter_dist : store_to_peter = 50) :
  D + store_to_peter + store_to_peter = 200 :=
by
  have h1 : D = 100, from sorry  -- Solved from store_to_peter = 50 and time_relation means D = 50*2 = 100
  by rw [h1, store_to_peter_dist]; norm_num; exact h1 

end total_miles_cycled_l636_636185


namespace XYZ_collinear_l636_636014

variables {A B C H P X Y Z : Point}
variables {PA PB PC : Line}

-- Definitions based on conditions
def is_orthocenter (H : Point) (A B C : Triangle) : Prop := orthocenter_of_triangle H A B C

def draw_perpendicular (H : Point) (PA : Line) : Line := perpendicular_from_point_to_line H PA

def intersect_extension (L1 L2 : Line) : Point := intersection_of_lines L1 L2

-- Given conditions as assumptions
variables (h1 : is_orthocenter H (triangle A B C))
variables (h2 : is_point_on_plane P)

-- HL, HM, HN are perpendiculars drawn from H to PA, PB, PC respectively
def HL := draw_perpendicular H (PA)
def HM := draw_perpendicular H (PB)
def HN := draw_perpendicular H (PC)

-- Points X, Y, Z defined by intersections of extensions
def X := intersect_extension (extension_of_line BC) HL
def Y := intersect_extension (extension_of_line CA) HM
def Z := intersect_extension (extension_of_line AB) HN

-- The theorem to prove collinearity
theorem XYZ_collinear :
  is_collinear X Y Z :=
sorry

end XYZ_collinear_l636_636014


namespace ratio_of_compositions_l636_636608

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem ratio_of_compositions :
  f (g (f 2)) / g (f (g 2)) = 41 / 7 :=
by
  -- Proof will go here
  sorry

end ratio_of_compositions_l636_636608


namespace v_closed_under_multiplication_l636_636622

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

def set_v : set ℕ := { n | is_cube n }

theorem v_closed_under_multiplication :
  ∀ (a b : ℕ), a ∈ set_v ∧ b ∈ set_v → (a * b) ∈ set_v := sorry

end v_closed_under_multiplication_l636_636622


namespace range_f_le_2_l636_636497

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^(1-x) else 1 - log x / log 2

theorem range_f_le_2 : {x : ℝ | f x ≤ 2} = set.Ici 0 := 
sorry

end range_f_le_2_l636_636497


namespace matt_twice_james_age_in_5_years_l636_636357

theorem matt_twice_james_age_in_5_years :
  (∃ x : ℕ, (3 + 27 = 30) ∧ (Matt_current_age = 65) ∧ 
  (Matt_age_in_x_years = Matt_current_age + x) ∧ 
  (James_age_in_x_years = James_current_age + x) ∧ 
  (Matt_age_in_x_years = 2 * James_age_in_x_years) → x = 5) :=
sorry

end matt_twice_james_age_in_5_years_l636_636357


namespace sqrt_inequality_at_least_one_positive_l636_636355

-- Problem 1
theorem sqrt_inequality {a : ℝ} (ha : 0 < a) :
  sqrt (a + 5) - sqrt (a + 3) > sqrt (a + 6) - sqrt (a + 4) :=
  sorry

-- Problem 2
theorem at_least_one_positive (x y z : ℝ) :
  let a := x^2 - 2 * y + (↑(Real.pi / 2))
  let b := y^2 - 2 * z + (↑(Real.pi / 3))
  let c := z^2 - 2 * x + (↑(Real.pi / 6))
  a > 0 ∨ b > 0 ∨ c > 0 :=
  sorry

end sqrt_inequality_at_least_one_positive_l636_636355


namespace sum_of_solutions_abs_eq_l636_636729

-- Define the condition on n
def abs_eq (n : ℝ) : Prop := abs (3 * n - 8) = 5

-- Statement we want to prove
theorem sum_of_solutions_abs_eq : ∑ n in { n | abs_eq n }, n = 16/3 :=
by
  sorry

end sum_of_solutions_abs_eq_l636_636729


namespace initial_cookie_count_l636_636312

variable (cookies_left_after_week : ℕ)
variable (cookies_taken_each_day : ℕ)
variable (total_cookies_taken_in_four_days : ℕ)
variable (initial_cookies : ℕ)
variable (days_per_week : ℕ)

theorem initial_cookie_count :
  cookies_left_after_week = 28 →
  total_cookies_taken_in_four_days = 24 →
  days_per_week = 7 →
  (∀ d (h : d ∈ Finset.range days_per_week), cookies_taken_each_day = 6) →
  initial_cookies = 52 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_cookie_count_l636_636312


namespace principal_amount_borrowed_l636_636759

theorem principal_amount_borrowed (P R T SI : ℕ) (h₀ : SI = (P * R * T) / 100) (h₁ : SI = 5400) (h₂ : R = 12) (h₃ : T = 3) : P = 15000 :=
by
  sorry

end principal_amount_borrowed_l636_636759


namespace find_angle_YZX_l636_636427

-- Given conditions
variables (A B C X Y Z Γ : Type)
variables [Incircle Γ (triangle A B C)]
variables [Circumcircle Γ (triangle X Y Z)]
variables (angle_A angle_B : ℝ)
variables (angle_A_def : angle_A = 50)
variables (angle_B_def : angle_B = 70)

-- Definition of angle C
def angle_C : ℝ := 180 - (angle_A + angle_B)

-- Theorem statement
theorem find_angle_YZX (angle_C_def : angle_C = 60) : angle_YZX = 60 := by
  sorry

end find_angle_YZX_l636_636427


namespace sum_of_n_values_l636_636750

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l636_636750


namespace angle_AMB_is_78_45_l636_636271

open Real

noncomputable def angle_formed_by_chords_intersecting_at_M (θAB θCD : ℝ) : ℝ :=
  (θAB + θCD) / 2

theorem angle_AMB_is_78_45 
  (A B C D M : Type)
  (arc_ratio : ℕ → ℕ)
  (h_ratio : arc_ratio 0 = 2 ∧ arc_ratio 1 = 3 ∧ arc_ratio 2 = 5 ∧ arc_ratio 3 = 6)
  (total_degree : ℝ := 360)
  (arc_length : ℕ → ℝ := λ n, (arc_ratio n).toReal / 16 * total_degree)
  (θAB : ℝ := arc_length 0)
  (θCD : ℝ := arc_length 2) : 
  angle_formed_by_chords_intersecting_at_M θAB θCD = 78.75 := 
  by
    sorry

end angle_AMB_is_78_45_l636_636271


namespace nonnegative_int_solution_count_l636_636133

theorem nonnegative_int_solution_count :
  (∃ n : ℕ, n - 1/2 = 5/2 ∨ n - 1/2 = -5/2) → 
  (∀ x: ℤ, (x^2 + x - 6 = 0) → x ≥ 0) → ∑ n in filter (λ x, x ≥ 0) (roots x^2 + x - 6) 1 :=
sorry

end nonnegative_int_solution_count_l636_636133


namespace infinitely_many_odd_composite_numbers_in_sequence_l636_636253

theorem infinitely_many_odd_composite_numbers_in_sequence :
  ∃ᶠ n in at_top, ∃ k, (∑ i in range (k + 1), i^i) % 2 = 1 ∧
                    ¬ nat.prime (∑ i in range (k + 1), i^i) :=
sorry

end infinitely_many_odd_composite_numbers_in_sequence_l636_636253


namespace number_of_ways_to_choose_officers_l636_636961

-- Define the number of boys and girls.
def num_boys : ℕ := 12
def num_girls : ℕ := 13

-- Define the total number of boys and girls.
def num_members : ℕ := num_boys + num_girls

-- Calculate the number of ways to choose the president, vice-president, and secretary with given conditions.
theorem number_of_ways_to_choose_officers : 
  (num_boys * num_girls * (num_boys - 1)) + (num_girls * num_boys * (num_girls - 1)) = 3588 :=
by
  -- The first part calculates the ways when the president is a boy.
  -- The second part calculates the ways when the president is a girl.
  sorry

end number_of_ways_to_choose_officers_l636_636961


namespace xiaoming_selects_cooking_probability_l636_636827

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636827


namespace positive_solution_count_l636_636168

theorem positive_solution_count :
  (finset.univ.filter (λ p : ℕ × ℕ, 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6 ∧
    (let b := p.1, c := p.2 in c < b ∧ b < ((3 * c + 1) / 2))) ).card = 3 :=
sorry

end positive_solution_count_l636_636168


namespace median_of_consecutive_integers_l636_636299

def sum_of_consecutive_integers (n : ℕ) (a : ℤ) : ℤ :=
  n * (2*a + (n - 1)) / 2

theorem median_of_consecutive_integers (a : ℤ) : 
  (sum_of_consecutive_integers 25 a = 5^5) -> 
  (a + 12 = 125) := 
by
  sorry

end median_of_consecutive_integers_l636_636299


namespace probability_of_selecting_cooking_is_one_fourth_l636_636869

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636869


namespace f_f_e_eq_2_l636_636496

def f : ℝ → ℝ := λ x, if x ≥ 0 then -Real.log x else (1/2)^x

theorem f_f_e_eq_2 : f (f Real.exp) = 2 := by sorry

end f_f_e_eq_2_l636_636496


namespace chess_pieces_missing_l636_636992

theorem chess_pieces_missing (total_pieces present_pieces missing_pieces : ℕ) 
  (h1 : total_pieces = 32)
  (h2 : present_pieces = 22)
  (h3 : missing_pieces = total_pieces - present_pieces) :
  missing_pieces = 10 :=
by
  sorry

end chess_pieces_missing_l636_636992


namespace probability_selecting_cooking_l636_636907

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l636_636907


namespace rectangle_area_in_square_l636_636383

def square_area : ℝ := 24

def rectangle_ratio : ℝ × ℝ := (1, 3)

noncomputable def area_of_rectangle (s : ℝ) (r : ℝ × ℝ) : ℝ :=
  let x := (2 * real.sqrt s * real.sqrt (r.1)) / real.sqrt (r.1 ^ 2 + r.2 ^ 2) in
  x * (r.2 / r.1) * x

theorem rectangle_area_in_square : area_of_rectangle 24 (1, 3) = 18 := by
  sorry

end rectangle_area_in_square_l636_636383


namespace probability_of_selecting_cooking_l636_636838

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636838


namespace probability_of_selecting_cooking_l636_636820

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636820


namespace P_eq_x_plus_1_l636_636476

noncomputable def P : ℕ → ℤ := sorry

def b_seq : ℕ → ℤ
| 0     := 1
| (k+1) := P (b_seq k)

lemma poly_gt_x (x : ℕ) : P x > x := sorry

lemma sequence_divisible (d : ℕ) : ∃ n : ℕ, d ∣ (b_seq n) := sorry

theorem P_eq_x_plus_1 : ∀ x : ℕ, P x = x + 1 :=
begin
  -- The proof goes here.
  sorry
end

end P_eq_x_plus_1_l636_636476


namespace find_x_l636_636058

theorem find_x 
  (h : √x / 0.9 + 1.2 / 0.7 = 2.879628878919216) : 
  x = 1.1 := 
sorry

end find_x_l636_636058


namespace problem_solution_l636_636140

def x : ℤ := -2 + 3
def y : ℤ := abs (-5)
def z : ℤ := 4 * (-1/4)

theorem problem_solution : x + y + z = 5 := 
by
  -- Definitions based on the problem statement
  have h1 : x = -2 + 3 := rfl
  have h2 : y = abs (-5) := rfl
  have h3 : z = 4 * (-1/4) := rfl
  
  -- Exact result required to be proved. Adding placeholder for steps.
  sorry

end problem_solution_l636_636140


namespace xiaoming_selects_cooking_probability_l636_636828

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l636_636828


namespace right_triangle_min_q_l636_636555

open Nat

theorem right_triangle_min_q (p q : ℕ) (hp : Prime p) (hq : p > q) (h : p + q = 90) :
  q = 7 := 
  sorry

end right_triangle_min_q_l636_636555


namespace probability_of_selecting_cooking_l636_636777

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l636_636777


namespace andrew_stickers_correct_l636_636419

noncomputable def total_stickers : ℕ := 1500
def daniel_stickers : ℕ := 250
def fred_stickers : ℕ := daniel_stickers + 120
noncomputable def emily_stickers : ℕ := (daniel_stickers + fred_stickers) / 2
def gina_stickers : ℕ := 80
def friends_stickers : ℕ := daniel_stickers + fred_stickers + emily_stickers + gina_stickers
noncomputable def andrew_kept : ℕ := total_stickers - friends_stickers

theorem andrew_stickers_correct : andrew_kept = 490 :=
by {
  sorry -- Proof goes here
}

end andrew_stickers_correct_l636_636419


namespace max_m_value_l636_636211

theorem max_m_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 35) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m = 3 :=
by
  sorry

end max_m_value_l636_636211


namespace probability_of_cooking_l636_636794

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l636_636794


namespace shaded_area_correct_l636_636453

structure Point (α : Type u) :=
(x : α)
(y : α)

structure Triangle (α : Type u) :=
(A : Point α)
(B : Point α)
(C : Point α)

structure Square (α : Type u) :=
(A : Point α)
(B : Point α)
(C : Point α)
(D : Point α)

def area_of_triangle {α : Type u} [field α] (t : Triangle α) : α :=
  (1 / 2) * abs ((t.A.x * t.B.y + t.B.x * t.C.y + t.C.x * t.A.y) - (t.A.y * t.B.x + t.B.y * t.C.x + t.C.y * t.A.x))

def area_of_square {α : Type u} [field α] (s : Square α) : α :=
  let side := abs (s.A.x - s.B.x) * abs (s.A.y - s.D.y) in
  side

noncomputable def shaded_area : ℕ :=
let sq : Square ℕ := {A := ⟨0, 0⟩, B := ⟨40, 0⟩, C := ⟨40, 40⟩, D := ⟨0, 40⟩},
    unshaded_triangle1 : Triangle ℕ := {A := ⟨0, 0⟩, B := ⟨15, 0⟩, C := ⟨40, 25⟩},
    unshaded_triangle2 : Triangle ℕ := {A := ⟨25, 40⟩, B := ⟨40, 40⟩, C := ⟨40, 25⟩},
    shaded_triangle : Triangle ℕ := {A := ⟨0, 0⟩, B := ⟨0, 15⟩, C := ⟨15, 0⟩} in
area_of_square sq - (area_of_triangle unshaded_triangle1 + area_of_triangle unshaded_triangle2) + area_of_triangle shaded_triangle

theorem shaded_area_correct : shaded_area = 1100 := 
sorry

end shaded_area_correct_l636_636453


namespace line_passing_through_point_and_inclined_at_angle_l636_636672

noncomputable def equation_of_line (P : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ × ℝ :=
  let k := Real.tan θ in
  let (x₁, y₁) := P in
  (1, k, y₁ - k * x₁)

theorem line_passing_through_point_and_inclined_at_angle :
  ∀ (P : ℝ × ℝ), P = (Real.sqrt 3, -(2 * Real.sqrt 3)) →
  ∀ (θ : ℝ), θ = 135 * Real.pi / 180 →
  equation_of_line P θ = (1, 1, Real.sqrt 3) := 
by
  intros P hP θ hθ
  rw [hP, hθ]
  sorry

end line_passing_through_point_and_inclined_at_angle_l636_636672


namespace rebecca_pies_l636_636247

theorem rebecca_pies 
  (P : ℕ) 
  (slices_per_pie : ℕ := 8) 
  (rebecca_slices : ℕ := P) 
  (family_and_friends_slices : ℕ := (7 * P) / 2) 
  (additional_slices : ℕ := 2) 
  (remaining_slices : ℕ := 5) 
  (total_slices : ℕ := slices_per_pie * P) :
  rebecca_slices + family_and_friends_slices + additional_slices + remaining_slices = total_slices → 
  P = 2 := 
by { sorry }

end rebecca_pies_l636_636247


namespace probability_of_selecting_cooking_is_one_fourth_l636_636878

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l636_636878


namespace find_n_l636_636445

theorem find_n (n : ℕ) (h1 : n > 1) (h2 : n + n.gcd(n/factor n) = 2013) :
  n = 1342 := sorry

end find_n_l636_636445


namespace add_pure_fruit_juice_l636_636219

theorem add_pure_fruit_juice (initial_volume : ℝ) (initial_purity : ℝ) (final_purity : ℝ) :
  initial_volume = 2 → 
  initial_purity = 0.1 → 
  final_purity = 0.25 →
  let x := (final_purity * initial_volume - initial_purity * initial_volume) / (1 - final_purity) in
  x = 0.4 :=
by
  intros h1 h2 h3
  dsimp
  have h : x = (0.25 * 2 - 0.1 * 2) / (1 - 0.25), by rw [h1, h2, h3]
  rw h
  norm_num
  sorry

end add_pure_fruit_juice_l636_636219


namespace max_disks_l636_636340

theorem max_disks (n k : ℕ) (h1: n ≥ 1) (h2: k ≥ 1) :
  (∃ (d : ℕ), d = if n > 1 ∧ k > 1 then 2 * (n + k) - 4 else max n k) ∧
  (∀ (p q : ℕ), (p <= n → q <= k → ¬∃ (x y : ℕ), x + 1 = y ∨ x - 1 = y ∨ x + 1 = p ∨ x - 1 = p)) :=
sorry

end max_disks_l636_636340


namespace probability_of_selecting_cooking_l636_636848

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l636_636848


namespace median_of_consecutive_integers_l636_636298

def sum_of_consecutive_integers (n : ℕ) (a : ℤ) : ℤ :=
  n * (2*a + (n - 1)) / 2

theorem median_of_consecutive_integers (a : ℤ) : 
  (sum_of_consecutive_integers 25 a = 5^5) -> 
  (a + 12 = 125) := 
by
  sorry

end median_of_consecutive_integers_l636_636298


namespace irrational_root_two_l636_636010

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_root_two : ∃ x ∈ ({-1, 0, 1 / 2, Real.sqrt 2} : set ℝ), is_irrational x :=
by 
  use Real.sqrt 2
  split
  {
    norm_num,
    exact Real.sqrt_nonneg 2,
  }
  {
    sorry -- Proof that sqrt(2) is irrational
  }

end irrational_root_two_l636_636010


namespace probability_of_selecting_cooking_l636_636818

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636818


namespace probability_of_selecting_cooking_l636_636811

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636811


namespace probability_cooking_selected_l636_636915

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l636_636915


namespace sum_of_solutions_abs_eq_l636_636734

-- Define the condition on n
def abs_eq (n : ℝ) : Prop := abs (3 * n - 8) = 5

-- Statement we want to prove
theorem sum_of_solutions_abs_eq : ∑ n in { n | abs_eq n }, n = 16/3 :=
by
  sorry

end sum_of_solutions_abs_eq_l636_636734


namespace least_common_multiple_lcm_condition_l636_636289

variable (a b c : ℕ)

theorem least_common_multiple_lcm_condition :
  Nat.lcm a b = 18 → Nat.lcm b c = 28 → Nat.lcm a c = 126 :=
by
  intros h1 h2
  sorry

end least_common_multiple_lcm_condition_l636_636289


namespace min_days_to_triple_loan_l636_636393

theorem min_days_to_triple_loan (amount_borrowed : ℕ) (interest_rate : ℝ) :
  ∀ x : ℕ, x ≥ 20 ↔ amount_borrowed + (amount_borrowed * (interest_rate / 10)) * x ≥ 3 * amount_borrowed :=
sorry

end min_days_to_triple_loan_l636_636393


namespace minimum_workers_needed_l636_636711

noncomputable def units_per_first_worker : Nat := 48
noncomputable def units_per_second_worker : Nat := 32
noncomputable def units_per_third_worker : Nat := 28

def minimum_workers_first_process : Nat := 14
def minimum_workers_second_process : Nat := 21
def minimum_workers_third_process : Nat := 24

def lcm_3_nat (a b c : Nat) : Nat :=
  Nat.lcm (Nat.lcm a b) c

theorem minimum_workers_needed (a b c : Nat) (w1 w2 w3 : Nat)
  (h1 : a = 48) (h2 : b = 32) (h3 : c = 28)
  (hw1 : w1 = minimum_workers_first_process )
  (hw2 : w2 = minimum_workers_second_process )
  (hw3 : w3 = minimum_workers_third_process ) :
  lcm_3_nat a b c / a = w1 ∧ lcm_3_nat a b c / b = w2 ∧ lcm_3_nat a b c / c = w3 :=
by
  sorry

end minimum_workers_needed_l636_636711


namespace part_a_part_b_l636_636465

theorem part_a (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2) : abs (a + b + c - a * b * c) ≤ 2 := 
sorry

theorem part_b (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2) : abs (a^3 + b^3 + c^3 - 3 * a * b * c) ≤ 2 * sqrt 2 := 
sorry

end part_a_part_b_l636_636465


namespace sqrt_four_eq_plus_minus_two_l636_636297

theorem sqrt_four_eq_plus_minus_two : ∃ y : ℤ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  -- Proof goes here
  sorry

end sqrt_four_eq_plus_minus_two_l636_636297


namespace probability_of_selecting_cooking_l636_636929

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636929


namespace exists_k0_for_all_k_geq_k0_sufficiently_large_k_implies_Sk_lt_k_sub_1985_l636_636085

-- Definition of the strictly increasing unbounded sequence of positive numbers
def strictly_increasing_unbounded_sequence (a : ℕ → ℝ) :=
  (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ M : ℝ, ∃ n : ℕ, M < a n)

-- Theorem statements for the proof problem
theorem exists_k0_for_all_k_geq_k0 (a : ℕ → ℝ) (h : strictly_increasing_unbounded_sequence a) :
  ∃ k0 : ℕ, ∀ k ≥ k0, (∑ i in finset.range k, (a i) / (a (i + 1))) < k - 1 :=
by sorry

theorem sufficiently_large_k_implies_Sk_lt_k_sub_1985 (a : ℕ → ℝ) (h : strictly_increasing_unbounded_sequence a) :
  ∃ k0 : ℕ, ∀ k ≥ k0, (∑ i in finset.range k, (a i) / (a (i + 1))) < k - 1985 :=
by sorry

end exists_k0_for_all_k_geq_k0_sufficiently_large_k_implies_Sk_lt_k_sub_1985_l636_636085


namespace angle_A_range_l636_636175

theorem angle_A_range (a b : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) :
  ∃ A : ℝ, 0 < A ∧ A ≤ Real.pi / 4 :=
sorry

end angle_A_range_l636_636175


namespace probability_of_selecting_cooking_l636_636943

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l636_636943


namespace total_payment_correct_l636_636996

theorem total_payment_correct 
  (bob_bill : ℝ) 
  (kate_bill : ℝ) 
  (bob_discount_rate : ℝ) 
  (kate_discount_rate : ℝ) 
  (bob_discount : ℝ := bob_bill * bob_discount_rate / 100) 
  (kate_discount : ℝ := kate_bill * kate_discount_rate / 100) 
  (bob_final_payment : ℝ := bob_bill - bob_discount) 
  (kate_final_payment : ℝ := kate_bill - kate_discount) : 
  (bob_bill = 30) → 
  (kate_bill = 25) → 
  (bob_discount_rate = 5) → 
  (kate_discount_rate = 2) → 
  (bob_final_payment + kate_final_payment = 53) :=
by
  intros
  sorry

end total_payment_correct_l636_636996


namespace quadrilateral_parallelogram_midpoints_l636_636982

-- Defining the quadrilateral and points
variables (A B C D P Q E F: Type) 

-- Conditions as hypotheses:
-- 1. Quadrilateral ABCD exists
-- 2. Points P and Q are on diagonal BD such that BP = PQ = QD
-- 3. E is the intersection of AP and BC
-- 4. F is the intersection of AQ and CD

def is_parallelogram (A B C D: Type) : Prop := sorry -- Parallelogram definition
def midpoint (X Y Z: Type) : Prop := sorry -- Midpoint definition

-- Main theorem statement
theorem quadrilateral_parallelogram_midpoints (h1 : is_parallelogram A B C D)
  (h2 : midpoint B P Q) (h3 : midpoint P Q D)
  (h4 : E = intersection (line_through A P) (line_through B C))
  (h5 : F = intersection (line_through A Q) (line_through C D)) :
  is_parallelogram A B C D ↔ (midpoint B C E ∧ midpoint C D F) :=
sorry

end quadrilateral_parallelogram_midpoints_l636_636982


namespace greatest_prime_factor_of_176_l636_636326

-- Define the number 176
def num : ℕ := 176

-- Define the prime factors of 176
def prime_factors := [2, 11]

-- The definition of the greatest prime factor function
def greatest_prime_factor (n : ℕ) : ℕ := (prime_factors.filter (λ x => x ∣ n)).max' sorry

-- The main theorem stating the greatest prime factor of 176
theorem greatest_prime_factor_of_176 : greatest_prime_factor num = 11 := by
  -- Proof would go here
  sorry

end greatest_prime_factor_of_176_l636_636326


namespace ratio_of_seedlings_l636_636248

-- Define conditions
def seedlings_first_day  : ℕ := 200
def total_seedlings      : ℕ := 1200
def seedlings_second_day : ℕ := total_seedlings - seedlings_first_day

-- Statement to be proved
theorem ratio_of_seedlings : seedlings_second_day / seedlings_first_day = 5 :=
by
  -- Seedlings second day: 1200 - 200 = 1000
  have h1 : seedlings_second_day = 1000 := by rw [seedlings_second_day, total_seedlings, seedlings_first_day]; rfl
  -- Compute gcd(1000, 200)
  have h2 : seedlings_second_day % seedlings_first_day = 0 := by norm_num [h1, seedlings_first_day]
  -- 1000 / 200 = 5
  show seedlings_second_day / seedlings_first_day = 5 by rw [h1, seedlings_first_day]; norm_num


end ratio_of_seedlings_l636_636248


namespace probability_cooking_is_one_fourth_l636_636863
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l636_636863


namespace AL_len_eq_l636_636631

open Equiv

-- Definitions of the points, line, and side lengths.
variables {l : Line}  {A D E F G B C : Point} {b c a : ℝ}
variables (triangle1 : EquilTriangle A D E)
variables (triangle2 : EquilTriangle A F G)
variables (triangle3 : EquilTriangle A B C)
variables (G A B : Collinear)

-- given side lengths
hypothesis (h1: length (segment A D) = b)
hypothesis (h2: length (segment A F) = c)
hypothesis (h3: length (segment A B) = a)

-- intersections
variables {N L : Point}
hypothesis (h4 : intersects (line G D) (line A E) N)
hypothesis (h5 : intersects (line B N) (line A C) L)

-- statement to prove:
theorem AL_len_eq : length (segment A L) = abc / (ab + bc + ac) :=
sorry

end AL_len_eq_l636_636631


namespace most_suitable_statistical_graph_for_air_composition_l636_636391

-- Define the conditions given in the problem
variables (Air : Type) [Composition : VariousGasses Air]

-- Define the problem as a theorem stating that the most suitable statistical graph to use is a pie chart
theorem most_suitable_statistical_graph_for_air_composition : 
  most_suitable_statistical_graph Air Composition = PieChart :=
sorry

end most_suitable_statistical_graph_for_air_composition_l636_636391


namespace median_of_consecutive_integers_l636_636301

theorem median_of_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) (h_sum : sum_of_integers = 5^5) (h_num : num_of_integers = 25) : 
  let median := sum_of_integers / num_of_integers
  in median = 125 :=
by
  let median := sum_of_integers / num_of_integers
  have h1 : sum_of_integers = 3125 := by exact h_sum
  have h2 : num_of_integers = 25 := by exact h_num
  have h3 : median = 125 := by
    calc
      median = 3125 / 25 : by rw [h1, h2]
            ... = 125      : by norm_num
  exact h3

end median_of_consecutive_integers_l636_636301


namespace probability_select_cooking_l636_636892

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l636_636892


namespace matrix_inverse_correct_l636_636053

open Matrix

noncomputable def given_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7, -4], ![-3, 2]]

noncomputable def expected_inverse : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 2], ![1.5, 3.5]]

theorem matrix_inverse_correct :
  let det := (given_matrix 0 0) * (given_matrix 1 1) - (given_matrix 0 1) * (given_matrix 1 0)
  det ≠ 0 →
  (1 / det) • adjugate given_matrix = expected_inverse :=
by
  sorry

end matrix_inverse_correct_l636_636053


namespace triangle_is_right_triangle_l636_636096

theorem triangle_is_right_triangle 
  (a b c : ℝ) 
  (h : a^2 * c^2 + b^2 * c^2 = a^4 - b^4) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  c^2 + b^2 = a^2 :=
begin
  sorry
end

end triangle_is_right_triangle_l636_636096


namespace hexagon_diagonals_sum_l636_636965

theorem hexagon_diagonals_sum (x y z : ℕ) 
  (h1 : 81 * y + 31 * 81 = x * z) 
  (h2 : x * z + 81^2 = y^2) 
  (h3 : 81 * y + 81^2 = z^2) 
  (hy : y = 144) 
  (hz : z = 135) 
  (hx : x = 105) : 
  x + y + z = 384 :=
by 
  rw [hx, hy, hz]
  exact rfl

end hexagon_diagonals_sum_l636_636965
