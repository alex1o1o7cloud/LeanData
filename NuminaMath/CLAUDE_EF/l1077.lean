import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l1077_107712

def T : ℕ → ℕ
  | 0 => 5  -- Add this case for 0
  | 1 => 5
  | n + 2 => 3^(T (n + 1))

theorem t_50_mod_7 : T 50 % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l1077_107712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_hexagon_inequality_l1077_107734

theorem circle_triangle_hexagon_inequality (x y : ℝ) :
  Real.sqrt 3 * |x| ≤ Real.sqrt (x^2 + y^2) ∧ Real.sqrt (x^2 + y^2) ≤ |x| + |y| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_hexagon_inequality_l1077_107734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_differences_l1077_107794

/-- A function that checks if a number is a 4-digit integer -/
def is_four_digit (x : ℕ) : Prop := 1000 ≤ x ∧ x ≤ 9999

/-- A function that returns the set of digits in a number -/
def digits (x : ℕ) : Finset ℕ := sorry

/-- The set of allowed digits -/
def allowed_digits : Finset ℕ := {2, 3, 4, 6, 7, 8, 9}

/-- The theorem statement -/
theorem min_sum_of_differences (n m p : ℤ) : 
  is_four_digit n.toNat ∧ is_four_digit m.toNat ∧ is_four_digit p.toNat ∧ 
  n ≠ m ∧ n ≠ p ∧ m ≠ p ∧
  (digits n.toNat ∪ digits m.toNat ∪ digits p.toNat) = allowed_digits →
  14552 ≤ |n - m| + |n - p| + |m - p| :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_differences_l1077_107794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_sin_and_tan_sign_l1077_107770

theorem cos_value_from_sin_and_tan_sign (α : ℝ) 
  (h1 : Real.sin α = -5/13) 
  (h2 : Real.tan α > 0) : 
  Real.cos α = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_sin_and_tan_sign_l1077_107770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_linear_functions_l1077_107754

-- Define the type for function relations
inductive FunctionRelation
| Relation1 : FunctionRelation  -- y = 2x + 1
| Relation2 : FunctionRelation  -- y = 1/x
| Relation3 : FunctionRelation  -- y = (x+1)/2 - x
| Relation4 : FunctionRelation  -- s = 60t
| Relation5 : FunctionRelation  -- y = 100 - 25x

-- Define what it means for a function relation to be linear
def isLinear (f : FunctionRelation) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  match f with
  | FunctionRelation.Relation1 => true  -- y = 2x + 1 is already in the form y = kx + b
  | FunctionRelation.Relation2 => false  -- y = 1/x cannot be expressed as y = kx + b
  | FunctionRelation.Relation3 => true  -- y = (x+1)/2 - x can be simplified to y = -1/2x + 1/2
  | FunctionRelation.Relation4 => true  -- s = 60t can be rewritten as y = 60x
  | FunctionRelation.Relation5 => true  -- y = 100 - 25x is already in the form y = kx + b

-- Theorem statement
theorem exactly_four_linear_functions :
  ∃! (n : ℕ), n = 4 ∧ (∀ f : FunctionRelation, isLinear f) ↔ n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_linear_functions_l1077_107754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1077_107765

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 7 ∧
  t.c = 2 ∧
  2 * t.b * Real.cos t.C = 2 * t.a - t.c

-- Helper function to calculate area
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π / 3 ∧ area t = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1077_107765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_nine_percent_l1077_107767

/-- Calculates the interest rate for years 3-5 given a loan scenario -/
noncomputable def calculate_interest_rate (principal : ℝ) (total_interest : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let interest1 := principal * rate1 * 2
  let interest2 := principal * rate2 * 4
  let interest35 := total_interest - interest1 - interest2
  (interest35 / (principal * 3)) * 100

/-- Theorem stating that the interest rate for years 3-5 is 9% given the problem conditions -/
theorem interest_rate_is_nine_percent :
  calculate_interest_rate 12000 11400 0.06 0.14 = 9 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_nine_percent_l1077_107767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_sum_existence_l1077_107733

theorem partition_sum_existence (n : ℕ) :
  ∀ (A B C : Finset ℕ),
    (A ∪ B ∪ C : Finset ℕ) = Finset.range (3 * n) →
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ →
    Finset.card A = n ∧ Finset.card B = n ∧ Finset.card C = n →
    ∃ a b c, a ∈ A ∧ b ∈ B ∧ c ∈ C ∧ (a = b + c ∨ b = c + a ∨ c = a + b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_sum_existence_l1077_107733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_b_onto_a_l1077_107717

def a : ℝ × ℝ := (0, 5)
def b : ℝ × ℝ := (2, -1)

theorem projection_of_b_onto_a :
  let proj := ((a.1 * b.1 + a.2 * b.2) / (a.1^2 + a.2^2)) • a
  proj = (0, -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_b_onto_a_l1077_107717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l1077_107785

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y : ℝ) (fx fy : ℝ) : ℝ := 
  Real.sqrt ((x - fx)^2 + (y - fy)^2)

-- Theorem statement
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (f1x f1y f2x f2y : ℝ) -- Coordinates of the two foci
  (h1 : is_on_ellipse x y)
  (h2 : distance_to_focus x y f1x f1y = 3)
  : distance_to_focus x y f2x f2y = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l1077_107785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_700_l1077_107798

/-- The cost of a box of pencils -/
def box_cost : ℚ := 50

/-- The number of pencils in a box -/
def pencils_per_box : ℕ := 200

/-- The discount rate for additional sets of 1000 pencils -/
def discount_rate : ℚ := 1/10

/-- The number of pencils the company wants to buy -/
def total_pencils : ℕ := 3000

/-- The number of pencils in each set for discount calculation -/
def pencils_per_set : ℕ := 1000

/-- The cost of pencils without discount -/
def cost_without_discount (n : ℕ) : ℚ :=
  (n : ℚ) * (box_cost / pencils_per_box)

/-- The discounted cost for a set of pencils -/
def discounted_cost (n : ℕ) : ℚ :=
  cost_without_discount n * (1 - discount_rate)

/-- The total cost for buying the specified number of pencils -/
def total_cost : ℚ :=
  cost_without_discount pencils_per_set +
  discounted_cost pencils_per_set +
  discounted_cost pencils_per_set

theorem total_cost_is_700 : total_cost = 700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_700_l1077_107798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marcy_coins_l1077_107787

theorem marcy_coins (total_coins : ℕ) (different_values : ℕ) 
  (h1 : total_coins = 15)
  (h2 : different_values = 20)
  (h3 : ∀ (x y : ℕ), x + y = total_coins → 
    (∃ (combinations : Finset ℕ), 
      (∀ c ∈ combinations, ∃ (a b : ℕ), c = 10 * a + 25 * b ∧ a + b ≤ total_coins) ∧
      Finset.card combinations = different_values)) :
  ∃ (num_25cent : ℕ), num_25cent = 7 ∧ 
    ∃ (num_10cent : ℕ), num_10cent + num_25cent = total_coins :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marcy_coins_l1077_107787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_each_girl_gets_two_dollars_l1077_107783

/-- Represents the savings and debt scenario for Tamara, Nora, and Lulu -/
structure SavingsScenario where
  debt : ℚ
  lulu_savings : ℚ
  nora_savings : ℚ
  tamara_savings : ℚ

/-- Calculates the amount each girl receives after paying off the debt -/
def amount_per_girl (s : SavingsScenario) : ℚ :=
  (s.nora_savings + s.tamara_savings + s.lulu_savings - s.debt) / 3

/-- Theorem stating that each girl receives $2 in the given scenario -/
theorem each_girl_gets_two_dollars (s : SavingsScenario) 
  (h1 : s.debt = 40)
  (h2 : s.nora_savings = 5 * s.lulu_savings)
  (h3 : s.nora_savings = 3 * s.tamara_savings)
  (h4 : s.lulu_savings = 6) : 
  amount_per_girl s = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_each_girl_gets_two_dollars_l1077_107783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_range_l1077_107714

open Set Real

theorem sufficient_condition_range (m : ℝ) : 
  m > 0 → 
  (∀ (A B : Set ℝ), 
    A = {x : ℝ | x^2 - x - 12 < 0} → 
    B = {x : ℝ | |x - 3| ≤ m} → 
    (∀ x : ℝ, x ∈ A → x ∈ B) → 
    m ≥ 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_range_l1077_107714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_cos_shift_l1077_107745

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi/3)

theorem axis_of_symmetry_cos_shift :
  ∃ (x : ℝ), x = 5*Real.pi/3 ∧ 
  (∀ (y : ℝ), f (x - y) = f (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_cos_shift_l1077_107745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l1077_107749

noncomputable def my_sequence (n : ℕ) : ℝ := Real.sqrt (n * (n + 2)) - Real.sqrt (n^2 - 2*n + 3)

theorem my_sequence_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |my_sequence n - 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l1077_107749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_four_consecutive_even_integers_largest_of_four_consecutive_even_integers_proof_l1077_107720

theorem largest_of_four_consecutive_even_integers : ℕ → Prop :=
  fun n =>
    -- Define the sum of first 20 positive even integers
    let sum_20_even := 20 * 21 * 2
    -- Define the four consecutive even integers
    let four_consecutive := [n - 6, n - 4, n - 2, n]
    -- Condition: sum of four consecutive even integers equals sum of first 20 positive even integers
    (four_consecutive.sum = sum_20_even) →
    -- Condition: the integers are even
    (∀ i ∈ four_consecutive, i % 2 = 0) →
    -- Conclusion: the largest of the four integers is 108
    (four_consecutive.maximum? = some 108)

-- The proof of the theorem
theorem largest_of_four_consecutive_even_integers_proof :
  largest_of_four_consecutive_even_integers 108 := by
  sorry

#check largest_of_four_consecutive_even_integers
#check largest_of_four_consecutive_even_integers_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_four_consecutive_even_integers_largest_of_four_consecutive_even_integers_proof_l1077_107720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1077_107748

theorem cube_root_equality : -((-8/27 : Real) ^ (1/3)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1077_107748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1077_107708

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 5) + 1 / (x^2 + 5*x + 6) + 1 / (x^4 + 16)

def domain_f : Set ℝ := {x | x ≠ -5 ∧ x ≠ -3 ∧ x ≠ -2}

theorem domain_of_f :
  domain_f = Set.Iio (-5) ∪ Set.Ioo (-5) (-3) ∪ Set.Ioo (-3) (-2) ∪ Set.Ioi (-2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1077_107708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_no_solution_in_range_l1077_107725

-- Define the infinite series
noncomputable def infiniteSeries (x : ℝ) : ℝ := 2*x + 1 + x^2 / (1 + x)

-- State the theorem
theorem solution_to_equation (x : ℝ) (h1 : |x| < 1) (h2 : infiniteSeries x = 13/6) :
  x = 1/2 ∨ x = -7/9 := by
  sorry

-- For part (b), we can define another infinite series
noncomputable def infiniteSeries2 (x : ℝ) : ℝ := 1/x + x/(1-x)

-- State the theorem for part (b)
theorem no_solution_in_range (x : ℝ) (h1 : |x| < 1) (h2 : infiniteSeries2 x = 7/2) :
  False := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_no_solution_in_range_l1077_107725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1077_107763

/-- The time it takes for two trains to cross each other -/
noncomputable def train_crossing_time (train_length : ℝ) (faster_train_speed : ℝ) : ℝ :=
  let slower_train_speed := faster_train_speed / 2
  let relative_speed := slower_train_speed + faster_train_speed
  let total_distance := 2 * train_length
  let relative_speed_mps := relative_speed * 1000 / 3600
  total_distance / relative_speed_mps

/-- Theorem stating the time it takes for two trains to cross each other -/
theorem train_crossing_time_approx :
  let train_length := (100 : ℝ)
  let faster_train_speed := (59.99520038396929 : ℝ)
  abs (train_crossing_time train_length faster_train_speed - 8.0005120051200512) < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1077_107763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_dice_probability_l1077_107727

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The set of possible outcomes for a single die roll -/
def die_outcomes : Finset ℕ := Finset.range num_sides

/-- The condition for a favorable outcome on a single die -/
def is_favorable (n : ℕ) : Bool := n < 3

/-- The probability of at least one die showing a number less than 3 when two fair 8-sided dice are rolled -/
theorem two_dice_probability : 
  (Finset.card (Finset.filter (fun p => is_favorable p.1 ∨ is_favorable p.2) (die_outcomes.product die_outcomes)) : ℚ) / 
  (Finset.card (die_outcomes.product die_outcomes) : ℚ) = 7/16 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_dice_probability_l1077_107727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_is_two_l1077_107760

/-- The number of intersection points between two polar curves -/
def intersection_count (f g : ℝ → ℝ) : ℕ := sorry

/-- First polar curve: r = 2 cos(2θ) -/
noncomputable def curve1 (θ : ℝ) : ℝ := 2 * Real.cos (2 * θ)

/-- Second polar curve: r = 6 sin(2θ) -/
noncomputable def curve2 (θ : ℝ) : ℝ := 6 * Real.sin (2 * θ)

/-- Theorem stating that the number of intersection points between the two curves is 2 -/
theorem intersection_count_is_two : intersection_count curve1 curve2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_is_two_l1077_107760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1077_107726

-- Define the quadratic function f
def f : ℝ → ℝ := sorry

-- Define the linear function g
def g (m : ℝ) : ℝ → ℝ := fun x ↦ 2 * x + m

-- State the theorem
theorem quadratic_function_properties :
  (∀ x, f (x + 1) - f x = 2 * x) →
  f 0 = 1 →
  (∃ a b c, ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x, f x = x^2 - x + 1) ∧
  (∀ m, (∀ x ∈ Set.Icc (-1) 1, f x > g m x) → m < -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1077_107726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_linear_l1077_107764

theorem gcd_polynomial_and_linear (a : ℤ) (h : ∃ k : ℤ, a = (2 * k + 1) * 7889) :
  Int.gcd (6 * a^2 + 55 * a + 126) (2 * a + 11) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_linear_l1077_107764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_unbounded_l1077_107729

noncomputable def a : ℕ → ℝ
| 0 => 1
| n + 1 => a n + 1 / (Real.sqrt (n + 1 : ℝ) * a n)

theorem sequence_unbounded :
  ∀ M : ℝ, M > 0 → ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a n > M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_unbounded_l1077_107729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l1077_107713

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (1 + 3 * Real.cos α, 3 + 3 * Real.sin α)

-- Define the line l
def line_l (m : ℝ) : ℝ → ℝ := λ x => m - x

-- Theorem statement
theorem curve_and_line_intersection (m : ℝ) :
  -- Part 1: Cartesian equation of C
  (∀ (x y : ℝ), (∃ α : ℝ, curve_C α = (x, y)) ↔ (x - 1)^2 + (y - 3)^2 = 9) ∧
  -- Part 2: Rectangular equation of l
  (∀ (x y : ℝ), y = line_l m x ↔ x + y = m) ∧
  -- Part 3: Intersection condition
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁ - 1)^2 + (y₁ - 3)^2 = 9 ∧ x₁ + y₁ = m ∧
    (x₂ - 1)^2 + (y₂ - 3)^2 = 9 ∧ x₂ + y₂ = m) ↔
  (4 - 3 * Real.sqrt 2 < m ∧ m < 4 + 3 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l1077_107713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diff_100th_term_value_l1077_107771

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  length : ℕ
  sum : ℚ
  min_term : ℚ
  max_term : ℚ

/-- Calculates the difference between the maximum and minimum possible values of the 100th term -/
noncomputable def calc_diff_100th_term (seq : ArithmeticSequence) : ℚ :=
  let avg : ℚ := seq.sum / seq.length
  let max_d : ℚ := (seq.max_term - avg) / (seq.length - 1)
  let min_d : ℚ := (seq.min_term - avg) / (seq.length - 1)
  let max_100th : ℚ := avg + 99 * max_d
  let min_100th : ℚ := avg + 99 * min_d
  max_100th - min_100th

/-- Theorem stating the difference between max and min 100th term -/
theorem diff_100th_term_value (seq : ArithmeticSequence) 
  (h1 : seq.length = 300)
  (h2 : seq.sum = 15000)
  (h3 : seq.min_term ≥ 20)
  (h4 : seq.max_term ≤ 200) :
  calc_diff_100th_term seq = 12060 / 299 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diff_100th_term_value_l1077_107771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l1077_107700

/-- A square in the complex plane centered at the origin with side length 2 -/
def OriginSquare : Set ℂ :=
  {z | Complex.abs z.re ≤ 1 ∧ Complex.abs z.im ≤ 1}

/-- The region outside the square -/
def R : Set ℂ := OriginSquare.compl

/-- The image of R under the reciprocal function -/
def S : Set ℂ := {w | ∃ z ∈ R, w = 1 / z}

/-- The area of a set in the complex plane -/
noncomputable def area (X : Set ℂ) : ℝ := sorry

theorem area_of_S : area S = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l1077_107700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1077_107741

theorem cos_beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α = Real.sqrt 5 / 5) (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  Real.cos β = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1077_107741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_90_l1077_107736

theorem sum_of_factors_90 : (Finset.filter (λ x => 90 % x = 0) (Finset.range 91)).sum id = 234 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_90_l1077_107736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_symmetric_wrt_z_axis_l1077_107746

/-- Two points in ℝ³ are symmetric with respect to the z-axis if their x and y coordinates are negations of each other, while their z coordinates are equal. -/
def symmetric_wrt_z_axis (A B : ℝ × ℝ × ℝ) : Prop :=
  A.fst = -B.fst ∧ A.snd = -B.snd ∧ A.2.2 = B.2.2

/-- Given points A(2,2,4) and B(-2,-2,4) in ℝ³, prove they are symmetric with respect to the z-axis. -/
theorem points_symmetric_wrt_z_axis :
  let A : ℝ × ℝ × ℝ := (2, 2, 4)
  let B : ℝ × ℝ × ℝ := (-2, -2, 4)
  symmetric_wrt_z_axis A B := by
  unfold symmetric_wrt_z_axis
  simp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_symmetric_wrt_z_axis_l1077_107746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_sequence_exists_l1077_107786

/-- A sequence of natural numbers -/
def InterestingSequence := ℕ → ℕ

/-- Check if a sequence is interesting -/
def is_interesting (s : InterestingSequence) : Prop :=
  ∀ n : ℕ, n > 0 → 
    (s (n + 1) = (s n + s (n + 2)) / 2 ∨ 
     (s (n + 1) : ℝ) = Real.sqrt ((s n : ℝ) * (s (n + 2) : ℝ)))

/-- Check if a sequence is purely arithmetic -/
def is_arithmetic (s : InterestingSequence) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, s (n + 1) = s n + d

/-- Check if a sequence is purely geometric -/
def is_geometric (s : InterestingSequence) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, (s (n + 1) : ℝ) = (s n : ℝ) * (q : ℝ)

/-- The main theorem -/
theorem interesting_sequence_exists : 
  ∃ (s : InterestingSequence) (a q : ℕ), 
    q > 1 ∧ 
    s 0 = a ∧ 
    s 1 = a * q ∧ 
    s 2 = a * q * q ∧
    is_interesting s ∧ 
    (∀ n : ℕ, s (n + 1) > s n) ∧
    ¬(is_arithmetic s) ∧ 
    ¬(is_geometric s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_sequence_exists_l1077_107786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_pairwise_sum_implies_equal_power_of_two_size_l1077_107756

/-- A finite collection of positive integers -/
def PositiveIntCollection := Finset ℕ+

/-- The pairwise sum collection of a PositiveIntCollection -/
def pairwiseSumCollection (A : PositiveIntCollection) : Finset ℕ+ :=
  A.product A |>.filter (fun p => p.1 < p.2) |>.image (fun p => p.1 + p.2)

theorem equal_pairwise_sum_implies_equal_power_of_two_size
  (A B : PositiveIntCollection)
  (h_distinct : A ≠ B)
  (h_equal_sums : pairwiseSumCollection A = pairwiseSumCollection B) :
  ∃ k : ℕ, A.card = 2^k ∧ B.card = 2^k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_pairwise_sum_implies_equal_power_of_two_size_l1077_107756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l1077_107706

noncomputable section

theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) : 
  (0 < A) → (A < π) → 
  (0 < B) → (B < π) → 
  (0 < C) → (C < π) → 
  (A + B + C = π) → 
  (a > 0) → (b > 0) → (c > 0) → 
  (a * Real.sin B = b * Real.sin A) → 
  (a * Real.sin C = c * Real.sin A) → 
  (b * Real.sin C = c * Real.sin B) → 
  (Real.sin C / Real.sin A = 2) → 
  (b^2 - a^2 = (3/2) * a * c) → 
  Real.cos B = 1/4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l1077_107706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_with_remainder_l1077_107735

theorem least_positive_integer_with_remainder (n : ℕ) : n = 181 ↔ 
  (n > 1) ∧ 
  (∀ d : ℕ, d ∈ ({2, 3, 4, 5, 6, 9, 10} : Finset ℕ) → n % d = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ d : ℕ, d ∈ ({2, 3, 4, 5, 6, 9, 10} : Finset ℕ) → m % d = 1) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_with_remainder_l1077_107735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_roots_l1077_107740

/-- Given a cubic equation with three positive real roots, this theorem proves
    the minimum value of a specific function of its coefficients. -/
theorem min_value_cubic_roots (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_roots : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
    ∀ t : ℝ, a * t^3 - t^2 + b * t - 1 = 0 ↔ t = x ∨ t = y ∨ t = z) :
  let P := (5 * a^2 - 3 * a * b + 2) / (a^2 * (b - a))
  ∀ p : ℝ, P ≥ p → p ≤ 12 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_roots_l1077_107740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_square_minimum_l1077_107790

theorem tan_cot_square_minimum (x : ℝ) (h : 0 < x ∧ x < Real.pi) :
  (Real.tan x + (Real.tan x)⁻¹)^2 ≥ 4 ∧
  ((Real.tan x + (Real.tan x)⁻¹)^2 = 4 ↔ x = Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_square_minimum_l1077_107790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1077_107781

theorem triangle_angle_measure (a b c A B C : ℝ) :
  0 < B → B < π →
  Real.sqrt 3 * b * Real.sin A - a * Real.cos B - 2 * a = 0 →
  B = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1077_107781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1077_107792

theorem log_inequality (a : ℝ) (h1 : a > 1) : 
  (Real.log a) / (2 * Real.log a) < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1077_107792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1077_107789

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ :=
  (seq.a 1) * (1 - (seq.a 2 / seq.a 1) ^ n) / (1 - (seq.a 2 / seq.a 1))

theorem geometric_sequence_property (seq : GeometricSequence) 
  (h1 : seq.a 1 = 1)
  (h2 : sum_n seq 10 = 3 * sum_n seq 5) :
  seq.a 6 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1077_107789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_duration_l1077_107716

/-- The number of days the work lasted -/
def D : ℕ := sorry

/-- Daily earnings of individual A -/
def A : ℚ := sorry

/-- Daily earnings of individual B -/
def B : ℚ := sorry

/-- Conditions of the problem -/
axiom earnings_A : A * (D - 2) = 80
axiom earnings_B : B * (D - 5) = 63
axiom condition_swap : A * (D - 5) = B * (D - 2) + 2

/-- Theorem: The work lasted 32 days -/
theorem work_duration : D = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_duration_l1077_107716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1077_107774

theorem line_equation_proof :
  ∃ k : ℝ,
    let f := λ x : ℝ ↦ x^2 + 7*x + 10
    let g := λ x : ℝ ↦ 5*x + 3
    let p1 := (k, f k)
    let p2 := (k, g k)
    (abs (p1.2 - p2.2) = 8) ∧
    (g 2 = 8) ∧
    (3 ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1077_107774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l1077_107739

theorem abc_inequality (a b c : ℝ) : 
  a = (1/2)^(1/3) → b = Real.log 2 / Real.log (1/3) → c = Real.log 3 / Real.log (1/2) → c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l1077_107739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1077_107769

-- Define the set of x that satisfy the inequality
def S : Set ℝ := {x | x^2 / (x + 1) ≥ 3 / (x + 1) + 3}

-- Define the solution set
def SolutionSet : Set ℝ := Set.Iic (-6) ∪ Set.Ioo (-1) 3

-- Theorem statement
theorem inequality_solution : S = SolutionSet := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1077_107769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_markup_calculation_l1077_107793

theorem retailer_markup_calculation (R : ℝ) (C M S : ℝ) 
  (h1 : C = 0.7 * R)  -- Cost price is 70% of retail price
  (h2 : S = 0.9 * M)  -- Sale price is 90% of marked price
  (h3 : S = C + 0.3 * S)  -- 30% profit on sale price
  : ∃ ε > 0, |M / R - 1.111| < ε :=  -- Marked price is approximately 111.1% of retail price
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_markup_calculation_l1077_107793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l1077_107784

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, -2 < x ∧ x < -1 → (1/2:ℝ)^x > 1) ∧
  (∃ x : ℝ, (1/2:ℝ)^x > 1 ∧ ¬(-2 < x ∧ x < -1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l1077_107784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_X_Y_equal_X_larger_than_Z_largest_shaded_area_l1077_107751

open Real

-- Define the side length of the squares
def side_length : ℝ := 4

-- Define the shaded area for Figure X
noncomputable def shaded_area_X : ℝ := side_length^2 - Real.pi * (side_length / 2)^2

-- Define the shaded area for Figure Y
noncomputable def shaded_area_Y : ℝ := side_length^2 - 4 * Real.pi

-- Define the shaded area for Figure Z
noncomputable def shaded_area_Z : ℝ := Real.pi * (side_length / 2)^2 - side_length^2

-- Theorem stating that X and Y have equal shaded areas
theorem X_Y_equal : shaded_area_X = shaded_area_Y := by sorry

-- Theorem stating that X (and thus Y) has larger shaded area than Z
theorem X_larger_than_Z : shaded_area_X > shaded_area_Z := by sorry

-- Main theorem combining the above results
theorem largest_shaded_area : 
  shaded_area_X = shaded_area_Y ∧ 
  shaded_area_X > shaded_area_Z ∧ 
  shaded_area_Y > shaded_area_Z := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_X_Y_equal_X_larger_than_Z_largest_shaded_area_l1077_107751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_zero_l1077_107724

def cubic_poly (t : ℝ) : ℝ := t^3 - 2*t - 2

theorem roots_sum_zero (x y z : ℝ) (hx : cubic_poly x = 0) (hy : cubic_poly y = 0) (hz : cubic_poly z = 0) :
  x*(y - z)^2 + y*(z - x)^2 + z*(x - y)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_zero_l1077_107724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_orthogonal_when_y_is_negative_three_l1077_107768

def v₁ : Fin 3 → ℝ := ![2, -4, -3]
def v₂ (y : ℝ) : Fin 3 → ℝ := ![-3, y, 2]

def dot_product (a b : Fin 3 → ℝ) : ℝ :=
  (a 0) * (b 0) + (a 1) * (b 1) + (a 2) * (b 2)

theorem vectors_orthogonal_when_y_is_negative_three :
  dot_product v₁ (v₂ (-3)) = 0 :=
by
  simp [dot_product, v₁, v₂]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_orthogonal_when_y_is_negative_three_l1077_107768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_equals_sin_f_period_4_f_1_is_cos_f_2_is_neg_sin_f_3_is_neg_cos_l1077_107791

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => Real.sin
  | (n + 1) => fun x => deriv (f n) x

-- State the theorem
theorem f_2016_equals_sin : ∀ x : ℝ, f 2016 x = Real.sin x := by
  sorry

-- Prove that f has a period of 4
theorem f_period_4 : ∀ (n : ℕ) (x : ℝ), f (n + 4) x = f n x := by
  sorry

-- Auxiliary lemma: f 1 is cosine
theorem f_1_is_cos : ∀ x : ℝ, f 1 x = Real.cos x := by
  sorry

-- Auxiliary lemma: f 2 is negative sine
theorem f_2_is_neg_sin : ∀ x : ℝ, f 2 x = -Real.sin x := by
  sorry

-- Auxiliary lemma: f 3 is negative cosine
theorem f_3_is_neg_cos : ∀ x : ℝ, f 3 x = -Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_equals_sin_f_period_4_f_1_is_cos_f_2_is_neg_sin_f_3_is_neg_cos_l1077_107791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_read_92_9_percent_l1077_107797

def read_percentage (p : Float) : String :=
  sorry  -- Implementation details omitted for brevity

theorem read_92_9_percent :
  read_percentage 92.9 = "ninety-two point nine percent" :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_read_92_9_percent_l1077_107797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unusual_bicycle_spokes_l1077_107799

/-- The number of spokes on an unusual bicycle. -/
def BicycleSpokes (front_spokes back_spokes : ℕ) : ℕ := front_spokes + back_spokes

/-- Theorem: The total number of spokes on the unusual bicycle is 60. -/
theorem unusual_bicycle_spokes : 
  let front_spokes : ℕ := 20
  let back_spokes : ℕ := 2 * front_spokes
  BicycleSpokes front_spokes back_spokes = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unusual_bicycle_spokes_l1077_107799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1077_107762

theorem remainder_problem (x : ℕ) (h : (8 * x) % 9 = 4) : x % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1077_107762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_C_l1077_107719

-- Define the sets A and B
def A : Finset ℕ := {0, 1}
def B : Finset ℕ := {0, 1, 2}

-- Define the property for set C
def satisfies_condition (C : Finset ℕ) : Prop := A ∪ C = B

-- State the theorem
theorem number_of_sets_C : ∃! (S : Finset (Finset ℕ)), 
  (∀ C, C ∈ S ↔ satisfies_condition C) ∧ Finset.card S = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_C_l1077_107719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_relation_l1077_107722

/-- Given an arithmetic sequence, Sₖ represents the sum of its first k terms. -/
noncomputable def S (a₁ d : ℝ) (k : ℕ) : ℝ := (k : ℝ) / 2 * (2 * a₁ + (k - 1 : ℝ) * d)

/-- For any arithmetic sequence and any positive integer n, 
    the sum of the first 2n terms is equal to the sum of the first n terms 
    plus one-third of the sum of the first 3n terms. -/
theorem arithmetic_series_sum_relation (a₁ d : ℝ) (n : ℕ) :
  S a₁ d (2 * n) = S a₁ d n + (1 / 3) * S a₁ d (3 * n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_relation_l1077_107722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventy_six_million_ninety_l1077_107709

theorem seventy_six_million_ninety : 76000090 = 76000090 := by
  -- The number seventy-six million ninety in standard base-10 representation
  rfl

#eval 76000090

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventy_six_million_ninety_l1077_107709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_winning_percentage_is_55_percent_l1077_107759

/-- Represents a basketball team's game statistics -/
structure BasketballTeam where
  gamesPlayed : ℕ
  totalGames : ℕ
  allowableLosses : ℕ
  targetWinPercentage : ℚ

/-- Calculates the initial winning percentage of a basketball team -/
def initialWinningPercentage (team : BasketballTeam) : ℚ :=
  let targetWins := (team.targetWinPercentage * team.totalGames).floor
  let currentWins := targetWins - team.allowableLosses
  currentWins / team.gamesPlayed

/-- Theorem stating that the initial winning percentage is 55% given the problem conditions -/
theorem initial_winning_percentage_is_55_percent (team : BasketballTeam)
  (h1 : team.gamesPlayed = 40)
  (h2 : team.totalGames = 50)
  (h3 : team.allowableLosses = 8)
  (h4 : team.targetWinPercentage = 3/5) :
  initialWinningPercentage team = 11/20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_winning_percentage_is_55_percent_l1077_107759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l1077_107703

theorem subset_intersection_theorem (n : ℕ) : 
  (∃ (A : Fin n → Finset (Fin n)), 
    (∀ i j : Fin n, i = j → A i = A j) ∧ 
    (∀ i : Fin n, (A i).card = 3) ∧
    (∀ i j : Fin n, i < j → (A i ∩ A j).card ≠ 1)) ↔ 
  ∃ k : ℕ, n = 4 * k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l1077_107703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonic_increase_interval_l1077_107744

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 6))

theorem g_monotonic_increase_interval (k : ℤ) :
  StrictMonoOn g (Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonic_increase_interval_l1077_107744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_even_function_condition_l1077_107730

theorem sine_even_function_condition (ω φ : ℝ) (h : ω ≠ 0) :
  (∃ k : ℤ, φ = k * π + π / 2) ↔
  (∀ x : ℝ, Real.sin (ω * x + φ) = Real.sin (ω * (-x) + φ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_even_function_condition_l1077_107730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_distribution_necessary_for_proportion_l1077_107753

/-- Represents the height of a student -/
def Height : Type := ℝ

/-- Represents a range of heights -/
def HeightRange : Type := Set Height

/-- Represents the frequency distribution of heights -/
def FrequencyDistribution : Type := HeightRange → ℕ

/-- Represents the proportion of students within a height range -/
def Proportion : Type := HeightRange → ℝ

/-- States that the frequency distribution is necessary to determine the proportion of students within a height range -/
theorem frequency_distribution_necessary_for_proportion 
  (students : Finset Height) (range : HeightRange) :
  ∃ (f : FrequencyDistribution), ∀ (p : Proportion), 
    (∀ r : HeightRange, p r = (f r : ℝ) / (students.card : ℝ)) →
    p range = (f range : ℝ) / (students.card : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_distribution_necessary_for_proportion_l1077_107753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_of_specific_geometric_sequence_l1077_107701

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Get the nth term of a geometric sequence -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.firstTerm * seq.commonRatio ^ (n - 1)

theorem sixth_term_of_specific_geometric_sequence :
  ∃ (seq : GeometricSequence),
    seq.firstTerm = 1024 ∧
    seq.nthTerm 8 = 125 ∧
    seq.nthTerm 6 = (5^(15/7)) / 32768 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_of_specific_geometric_sequence_l1077_107701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_halves_pi_l1077_107723

theorem cos_three_halves_pi : Real.cos (3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_halves_pi_l1077_107723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1077_107711

noncomputable section

/-- The area of a semicircle with diameter d -/
def semicircle_area (d : ℝ) : ℝ := (Real.pi * d^2) / 8

/-- The area of the shaded region given the lengths of line segments -/
def shaded_area (ab bc cd de ef : ℝ) : ℝ :=
  let af := ab + bc + cd + de + ef
  semicircle_area af - (semicircle_area ab + semicircle_area bc + 
                        semicircle_area cd + semicircle_area de + 
                        semicircle_area ef)

theorem shaded_area_calculation (ab bc cd de ef : ℝ) 
  (hab : ab = 4) (hbc : bc = 3) (hcd : cd = 5) (hde : de = 2) (hef : ef = 6) :
  shaded_area ab bc cd de ef = (267 / 8) * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1077_107711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_probabilities_l1077_107772

/-- Probability of particle movement between two points -/
structure ParticleMovement where
  p11 : ℝ  -- Probability of staying at A
  p12 : ℝ  -- Probability of moving from A to B
  p21 : ℝ  -- Probability of moving from B to A
  p22 : ℝ  -- Probability of staying at B

/-- Probabilities sum to 1 for each starting point -/
def valid_probabilities (pm : ParticleMovement) : Prop :=
  pm.p11 + pm.p12 = 1 ∧ pm.p21 + pm.p22 = 1

/-- Probability of being at A after n steps, starting from A -/
noncomputable def prob_A_to_A (pm : ParticleMovement) (n : ℕ) : ℝ :=
  pm.p21 / (pm.p12 + pm.p21) + pm.p12 / (pm.p12 + pm.p21) * (pm.p11 - pm.p21) ^ n

/-- Probability of being at A after n steps, starting from B -/
noncomputable def prob_B_to_A (pm : ParticleMovement) (n : ℕ) : ℝ :=
  pm.p21 / (pm.p21 + pm.p12) - pm.p21 / (pm.p21 + pm.p12) * (pm.p22 - pm.p12) ^ n

theorem particle_movement_probabilities (pm : ParticleMovement) (n : ℕ) 
    (h : valid_probabilities pm) : 
  (prob_A_to_A pm n = pm.p21 / (pm.p12 + pm.p21) + pm.p12 / (pm.p12 + pm.p21) * (pm.p11 - pm.p21) ^ n) ∧
  (prob_B_to_A pm n = pm.p21 / (pm.p21 + pm.p12) - pm.p21 / (pm.p21 + pm.p12) * (pm.p22 - pm.p12) ^ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_probabilities_l1077_107772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_midpoint_trajectory_l1077_107702

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 4 = 0

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the line l passing through P
def line (k : ℝ) (x y : ℝ) : Prop := y - P.2 = k * (x - P.1)

-- Define the length of AB
noncomputable def AB_length : ℝ := Real.sqrt 17

-- Define the slope angle of line l
def slope_angle (θ : ℝ) : Prop := θ = Real.pi/3 ∨ θ = 2*Real.pi/3

-- Define the trajectory of midpoint M
def trajectory (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1)^2 = 1/4

theorem line_slope_and_midpoint_trajectory :
  ∀ (A B : ℝ × ℝ) (k θ : ℝ),
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧  -- A and B are on the circle
  line k A.1 A.2 ∧ line k B.1 B.2 ∧  -- A and B are on the line l
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB_length^2 →  -- |AB| = √17
  (slope_angle θ ∧ k = Real.tan θ) ∧  -- Slope angle is π/3 or 2π/3
  (∀ (x y : ℝ), x = (A.1 + B.1)/2 ∧ y = (A.2 + B.2)/2 → trajectory x y)  -- Midpoint trajectory
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_midpoint_trajectory_l1077_107702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picking_cheapest_l1077_107758

/-- Represents the cost and yield of jam production -/
structure JamProduction where
  ticket_cost : ℚ
  berries_collected : ℚ
  berry_market_price : ℚ
  sugar_price : ℚ
  jam_yield : ℚ
  readymade_jam_price : ℚ

/-- Calculates the cost of making jam by picking berries -/
def cost_picking (jp : JamProduction) : ℚ :=
  (jp.ticket_cost / jp.berries_collected) + jp.sugar_price

/-- Calculates the cost of making jam by buying berries -/
def cost_buying (jp : JamProduction) : ℚ :=
  jp.berry_market_price + jp.sugar_price

/-- Calculates the cost of buying ready-made jam -/
def cost_readymade (jp : JamProduction) : ℚ :=
  jp.readymade_jam_price * jp.jam_yield

/-- Theorem stating that picking berries is the cheapest option -/
theorem picking_cheapest (jp : JamProduction) :
  jp.ticket_cost = 200 ∧
  jp.berries_collected = 5 ∧
  jp.berry_market_price = 150 ∧
  jp.sugar_price = 54 ∧
  jp.jam_yield = 3/2 ∧
  jp.readymade_jam_price = 220 →
  cost_picking jp < cost_buying jp ∧
  cost_picking jp < cost_readymade jp :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picking_cheapest_l1077_107758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_stream_ratio_l1077_107738

/-- Represents a water container with a given capacity -/
structure Container where
  capacity : ℚ
  deriving Repr

/-- Represents a water stream with a flow rate -/
structure WaterStream where
  flowRate : ℚ
  deriving Repr

/-- Models the water filling scenario -/
def fillScenario (c1 c2 : Container) (s1 s2 : WaterStream) : Prop :=
  c1.capacity = 10 ∧
  c2.capacity = 8 ∧
  s1.flowRate > s2.flowRate ∧
  ∃ (t1 t2 : ℚ),
    t1 > 0 ∧
    t2 > 0 ∧
    t1 * s1.flowRate + t2 * s2.flowRate = c1.capacity ∧
    t1 * s2.flowRate + t2 * s1.flowRate = c2.capacity ∧
    t1 * s2.flowRate = c2.capacity / 2

theorem water_stream_ratio
  (c1 c2 : Container) (s1 s2 : WaterStream)
  (h : fillScenario c1 c2 s1 s2) :
  s1.flowRate / s2.flowRate = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_stream_ratio_l1077_107738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1077_107728

-- Define the sets A and B
def A : Set ℝ := {x | x^2 > 4}
def B : Set ℝ := {x | Real.exp (x * Real.log 2) > 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1077_107728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1077_107782

/-- Given a polynomial g(x) such that g(x^2 - 1) = x^4 - 4*x^2 + 4,
    prove that g(x^2 - 3) = x^4 - 6*x^2 + 11 -/
theorem polynomial_identity (g : ℝ → ℝ) (h : ∀ x, g (x^2 - 1) = x^4 - 4*x^2 + 4) :
  ∀ x, g (x^2 - 3) = x^4 - 6*x^2 + 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1077_107782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l1077_107757

/-- Given a triangle ABC with external angle ratio α : β : γ, prove the internal angle ratio -/
theorem triangle_angle_ratio 
  (A B C : Real) -- Angles of the triangle
  (α β γ : Real) -- Ratio components of external angles
  (h_pos_α : α > 0) -- α is positive
  (h_pos_β : β > 0) -- β is positive
  (h_pos_γ : γ > 0) -- γ is positive
  (h_triangle : A + B + C = Real.pi) -- Sum of angles in a triangle is π
  (h_external_ratio : ∃ (t : Real), t > 0 ∧ 
    2 * Real.pi - A = α * t ∧ 
    2 * Real.pi - B = β * t ∧ 
    2 * Real.pi - C = γ * t) -- External angle ratio condition
  : ∃ (k : Real), k > 0 ∧ 
    A = k * (β + γ - α) ∧
    B = k * (α - β + γ) ∧
    C = k * (α + β - γ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l1077_107757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_traveled_l1077_107752

/-- The total distance traveled from (-3, 4) to (6, -3) via (2, 2) is equal to √29 + √41. -/
theorem total_distance_traveled : 
  let start : ℝ × ℝ := (-3, 4)
  let mid : ℝ × ℝ := (2, 2)
  let end_point : ℝ × ℝ := (6, -3)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance start mid + distance mid end_point = Real.sqrt 29 + Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_traveled_l1077_107752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_difference_l1077_107780

/-- Proves that Molly takes 150 minutes longer than Xanthia to read a 300-page book -/
theorem reading_time_difference 
  (xanthia_speed : ℕ) 
  (molly_speed : ℕ) 
  (book_pages : ℕ) 
  (h1 : xanthia_speed = 120) 
  (h2 : molly_speed = 60) 
  (h3 : book_pages = 300) : 
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 150 := by
  sorry

#check reading_time_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_difference_l1077_107780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_tree_in_10_seconds_l1077_107743

/-- The time (in seconds) it takes for a train to pass a stationary object -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * (1000 / 3600))

/-- Theorem: A train 500 meters long, traveling at 180 km/hr, will take 10 seconds to pass a tree -/
theorem train_passes_tree_in_10_seconds :
  train_passing_time 500 180 = 10 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_tree_in_10_seconds_l1077_107743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1077_107710

-- Define the cost function
def P (x : ℝ) : ℝ := 12 + 10 * x

-- Define the sales income function
noncomputable def Q (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 16 then -0.5 * x^2 + 22 * x
  else 224

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ := Q x - P x

-- Theorem statement
theorem max_profit :
  ∃ (x : ℝ), x = 12 ∧ f x = 60 ∧ ∀ y, f y ≤ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1077_107710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_range_l1077_107796

noncomputable section

-- Define the two lines
def line1 (x : ℝ) : ℝ := (1/2) * x - 1
def line2 (k : ℝ) (x : ℝ) : ℝ := k * x + 3 * k + 1

-- Define the intersection point
def intersection_point (k : ℝ) (m : ℝ) : Prop :=
  line1 m = line2 k m

-- Define the decreasing property of line2
def line2_decreasing (k : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → line2 k x₁ > line2 k x₂

-- Theorem statement
theorem intersection_point_range (k : ℝ) (m : ℝ) :
  intersection_point k m ∧ line2_decreasing k → -3 < m ∧ m < 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_range_l1077_107796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_calculation_l1077_107718

def original_price : ℝ := 150
def initial_discount_rate : ℝ := 0.25
def coupon_discount : ℝ := 10
def sales_tax_rate : ℝ := 0.10

theorem jacket_price_calculation :
  (original_price * (1 - initial_discount_rate) - coupon_discount) * (1 + sales_tax_rate) = 112.75 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_calculation_l1077_107718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1077_107742

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_subsequence (a : ℕ → ℚ) (indices : List ℕ) :=
  ∀ i j, i < j → i ∈ indices → j ∈ indices → 
    (a (indices.get! j)) / (a (indices.get! i)) = 
    (a (indices.get! 1)) / (a (indices.get! 0))

def sum_of_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (List.range n).map a |>.sum

theorem arithmetic_sequence_properties 
  (a : ℕ → ℚ) (d : ℚ) (h_arith : arithmetic_sequence a d) 
  (h_d_nonzero : d ≠ 0) (h_a1 : a 1 = 2) :
  (∃ indices : List ℕ, indices.length ≥ 3 ∧ 
    3 < indices.get! 2 ∧ 
    geometric_subsequence a (1::3::indices) ∧
    a 3 = 6 →
    ∀ k, indices.get! k = 3^(k+1)) ∧
  (∃ lambda : ℚ, lambda = 5 ∧ d = 4 ∧
    ∀ n : ℕ, sum_of_terms a (3*n) - sum_of_terms a (2*n) = lambda * sum_of_terms a n) :=
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1077_107742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1077_107732

/-- The speed of a train in kmph, given distance in km and time in minutes -/
noncomputable def train_speed (distance : ℝ) (time_minutes : ℝ) : ℝ :=
  (distance / (time_minutes / 60))

/-- Theorem stating that a train covering 20.166666666666664 km in 11 minutes has a speed of approximately 110 kmph -/
theorem train_speed_calculation :
  let distance := 20.166666666666664
  let time_minutes := 11
  let calculated_speed := train_speed distance time_minutes
  abs (calculated_speed - 110) < 0.1 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_speed 20.166666666666664 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1077_107732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_alpha_range_l1077_107779

noncomputable def f (x : ℝ) := Real.cos (2 * x) + Real.sin (2 * x + Real.pi / 6)

theorem f_two_zeros_alpha_range (α : ℝ) :
  (∃ x y, 0 < x ∧ x < y ∧ y < α ∧ f x = 0 ∧ f y = 0 ∧
  ∀ z, 0 < z ∧ z < α ∧ f z = 0 → z = x ∨ z = y) →
  5 * Real.pi / 6 < α ∧ α ≤ 4 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_alpha_range_l1077_107779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1077_107715

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.cos α = -3/5) : Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1077_107715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_existence_and_nonexistence_l1077_107788

def has_property (n k : ℕ) : Prop :=
  ∃ a : ℕ, a > 1 ∧ n = Finset.prod (Finset.range k) (λ i => a + i)

theorem property_existence_and_nonexistence :
  (∃ k n : ℕ, has_property n k ∧ has_property n (k + 2)) ∧
  (¬ ∃ n : ℕ, has_property n 2 ∧ has_property n 4) := by
  sorry

#check property_existence_and_nonexistence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_existence_and_nonexistence_l1077_107788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harkamal_fruit_purchase_l1077_107704

def fruit_purchase (grapes_kg mangoes_kg apples_kg oranges_kg : ℕ)
                   (grapes_price mangoes_price apples_price oranges_price : ℕ)
                   (discount_rate : ℚ) : ℚ :=
  let total_cost := grapes_kg * grapes_price + mangoes_kg * mangoes_price + 
                    apples_kg * apples_price + oranges_kg * oranges_price
  let discount_amount := (discount_rate * total_cost)
  (total_cost : ℚ) - discount_amount

theorem harkamal_fruit_purchase :
  fruit_purchase 8 9 5 2 70 60 50 30 (1/10) = 1269 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harkamal_fruit_purchase_l1077_107704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_inequality_l1077_107755

-- Define the sequence a_n and its partial sum s_n
noncomputable def a : ℕ+ → ℝ := sorry
noncomputable def s : ℕ+ → ℝ := sorry

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + (1/2) * x

-- State the conditions
axiom condition : ∀ n : ℕ+, s n = f n

-- Define T_n
noncomputable def T (n : ℕ+) : ℝ := (3/4) - (1/2) * ((1 / (n + 1)) + (1 / (n + 2)))

-- State the theorem to be proved
theorem sequence_and_inequality :
  (∀ n : ℕ+, a n = n) ∧
  {a : ℝ | 0 < a ∧ a < 1/2} =
  {a : ℝ | 0 < a ∧ a < 1 ∧ ∀ n : ℕ+, T n > (1/3) * Real.log (1 - a) / Real.log a} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_inequality_l1077_107755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_juice_cost_l1077_107776

/-- The cost of a small bottle of mango juice in pesetas -/
def small_bottle_cost (x : ℕ) : Prop := x = 600

/-- The volume of a big bottle in ounces -/
def big_bottle_volume : ℕ := 30

/-- The cost of a big bottle in pesetas -/
def big_bottle_cost : ℕ := 2700

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℕ := 6

/-- The amount saved by buying a big bottle instead of equivalent small bottles, in pesetas -/
def savings : ℕ := 300

theorem mango_juice_cost : small_bottle_cost 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_juice_cost_l1077_107776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_special_addition_l1077_107707

/-- Represents a single-digit integer in base 6 -/
def SingleDigitBase6 : Type := {n : ℕ // n < 6}

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ := sorry

/-- Adds two numbers in base 6 -/
def addBase6 (a b : List ℕ) : List ℕ := sorry

theorem absolute_difference_of_special_addition 
  (A B : SingleDigitBase6) :
  addBase6 (toBase6 A.val) (addBase6 (toBase6 (41 * B.val)) (toBase6 115)) = toBase6 1152 →
  (A.val : ℤ) - (B.val : ℤ) = 1 ∨ (A.val : ℤ) - (B.val : ℤ) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_special_addition_l1077_107707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_area_l1077_107750

/-- The length of the string in centimeters -/
noncomputable def string_length : ℝ := 32

/-- The side length of the square formed by the string -/
noncomputable def side_length : ℝ := string_length / 4

/-- The area of the square formed by the string -/
noncomputable def square_area : ℝ := side_length ^ 2

/-- Theorem: The area of the largest square that can be formed from a 32 cm string is 64 cm² -/
theorem largest_square_area : square_area = 64 := by
  -- Unfold the definitions
  unfold square_area side_length string_length
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_area_l1077_107750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1077_107775

/-- If g(x) = (x - 5) / (x^2 + cx + d) has vertical asymptotes at x = 2 and x = -3, then c + d = -5 -/
theorem asymptote_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -3 → ∃ g : ℝ → ℝ, g x = (x - 5) / (x^2 + c*x + d)) →
  (x^2 + c*x + d = (x - 2)*(x + 3)) →
  c + d = -5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1077_107775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l1077_107777

/-- The distance between two points in 3D space -/
noncomputable def distance3D (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Theorem: The distance between points A(1, 3, -2) and B(-2, 3, 2) is 5 -/
theorem distance_A_to_B : distance3D 1 3 (-2) (-2) 3 2 = 5 := by
  -- Unfold the definition of distance3D
  unfold distance3D
  -- Simplify the expression under the square root
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l1077_107777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_integers_with_average_five_l1077_107795

theorem five_integers_with_average_five (a b c d e : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < e →
  (a + b + c + d + e : ℚ) / 5 = 5 →
  e - a = Nat.max (e - a) (d - b) →
  b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_integers_with_average_five_l1077_107795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_one_l1077_107721

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (1 + x) * Real.log x

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 2 * x - 2

-- Theorem statement
theorem tangent_at_one :
  ∀ x, (deriv f 1) * (x - 1) + f 1 = tangent_line x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_one_l1077_107721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_bisector_distance_l1077_107766

-- Define the space
variable {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] [FiniteDimensional ℝ α]

-- Define points A and B
variable (A B : α)

-- Define the perpendicular bisector line l
def is_perp_bisector (l : Set α) (A B : α) : Prop :=
  ∀ P ∈ l, ‖P - A‖ = ‖P - B‖

-- Define the concept of "same side"
def same_side (P : α) (l : Set α) (X : α) : Prop :=
  ∃ Q ∈ l, ¬ ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q + t • (X - Q) = P

-- State the theorem
theorem perp_bisector_distance (l : Set α) (h : is_perp_bisector l A B) :
  ∀ P, P ∉ l →
    (same_side P l A → ‖P - A‖ < ‖P - B‖) ∧
    (same_side P l B → ‖P - B‖ < ‖P - A‖) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_bisector_distance_l1077_107766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_are_true_l1077_107761

-- Define the propositions
def proposition_p (a b c : ℝ) : Prop :=
  (∀ x y z : ℝ, x*(z^2) > y*(z^2) → x > y) ∧ 
  (∃ x y z : ℝ, x > y ∧ ¬(x*(z^2) > y*(z^2)))

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define angle and sine functions
noncomputable def angle (T : Triangle) (v : Fin 3) : ℝ := sorry
noncomputable def sine (θ : ℝ) : ℝ := sorry

def proposition_q (T : Triangle) : Prop :=
  (angle T 2 > angle T 1 ↔ sine (angle T 2) > sine (angle T 1))

theorem propositions_are_true :
  (∀ a b c : ℝ, proposition_p a b c) ∧
  (∀ T : Triangle, proposition_q T) := by
  sorry

#check propositions_are_true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_are_true_l1077_107761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_180_degrees_l1077_107705

/-- The cosine of an angle in radians -/
noncomputable def cosine (θ : ℝ) : ℝ :=
  (Complex.exp (θ * Complex.I)).re

/-- 180 degrees in radians -/
noncomputable def π : ℝ := Real.pi

theorem cos_180_degrees : cosine π = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_180_degrees_l1077_107705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_product_sign_change_l1077_107778

def sequence_a : ℕ → ℚ
  | 0 => 15  -- Add this case for 0
  | 1 => 15
  | (n + 2) => sequence_a (n + 1) - 2/3

theorem smallest_k_for_product_sign_change :
  ∃ k : ℕ, k > 0 ∧ sequence_a k * sequence_a (k + 1) < 0 ∧
  ∀ j : ℕ, 0 < j ∧ j < k → sequence_a j * sequence_a (j + 1) ≥ 0 :=
by
  use 23
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_product_sign_change_l1077_107778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_profit_percentage_l1077_107731

/-- Calculates the profit percentage given the selling price and purchase price -/
noncomputable def profitPercentage (sellingPrice purchasePrice : ℝ) : ℝ :=
  ((sellingPrice - purchasePrice) / purchasePrice) * 100

/-- Represents the problem scenario -/
theorem bill_profit_percentage (originalSellingPrice : ℝ) 
  (h1 : originalSellingPrice = 989.9999999999992)
  (h2 : ∃ (originalPurchasePrice : ℝ), 
    1.3 * (0.9 * originalPurchasePrice) = originalSellingPrice + 63) :
  ∃ (epsilon : ℝ), epsilon > 0 ∧ epsilon < 0.1 ∧
  ∃ (originalPurchasePrice : ℝ), 
    abs (profitPercentage originalSellingPrice originalPurchasePrice - 10) < epsilon := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_profit_percentage_l1077_107731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_approx_l1077_107773

/-- The rate per meter for fencing a circular field -/
noncomputable def fencing_rate (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (Real.pi * diameter)

/-- Theorem stating that the fencing rate is approximately 2.5 -/
theorem fencing_rate_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |fencing_rate 18 141.37 - 2.5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_approx_l1077_107773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_classmate_skittles_l1077_107737

/-- The number of Skittles Victor split among his classmates -/
noncomputable def total_skittles : ℚ := 25

/-- The number of classmates who received Skittles -/
def num_classmates : ℕ := 5

/-- The number of Skittles each classmate received -/
noncomputable def skittles_per_classmate : ℚ := total_skittles / num_classmates

theorem classmate_skittles : skittles_per_classmate = 5 := by
  rw [skittles_per_classmate, total_skittles, num_classmates]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_classmate_skittles_l1077_107737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1077_107747

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2 * Real.sqrt 3  -- Define for 0 to avoid missing case
  | 1 => 2 * Real.sqrt 3
  | n + 2 => 8 * sequence_a (n + 1) / (4 - (sequence_a (n + 1))^2)

theorem sequence_a_formula (n : ℕ) : 
  n ≥ 1 → sequence_a n = 2 * Real.tan (π / (3 * 2^(n - 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1077_107747
