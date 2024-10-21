import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l407_40737

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃! (z1 z2 z3 : ℝ), z1 ∈ Set.Icc 0 (2 * Real.pi) ∧ 
                            z2 ∈ Set.Icc 0 (2 * Real.pi) ∧ 
                            z3 ∈ Set.Icc 0 (2 * Real.pi) ∧ 
                            f ω z1 = 0 ∧ f ω z2 = 0 ∧ f ω z3 = 0) :
  (∀ x : ℝ, f ω x = f ω (5 * Real.pi / (2 * ω) - x)) ∧ 
  (11 / 8 ≤ ω ∧ ω < 15 / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l407_40737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_plus_minus_one_divisibility_l407_40766

theorem prime_plus_minus_one_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  (∃ n ∈ ({p-1, p+1} : Set ℕ), 4 ∣ n) ∧ 
  ¬(∀ p : ℕ, Nat.Prime p → p > 3 → ∃ n ∈ ({p-1, p+1} : Set ℕ), 5 ∣ n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_plus_minus_one_divisibility_l407_40766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_products_l407_40758

/-- The set of all positive integer divisors of 72000 -/
def T : Finset ℕ := (Finset.range 72001).filter (λ d => d > 0 ∧ 72000 % d = 0)

/-- The set of distinct products of two different elements from T -/
def distinct_products : Finset ℕ := 
  Finset.biUnion T (λ a => Finset.image (λ b => a * b) (T.filter (λ b => b ≠ a)))

theorem count_distinct_products : Finset.card distinct_products = 381 := by
  sorry

#eval Finset.card distinct_products

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_products_l407_40758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_repetition_l407_40742

open Real

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ],
    ![sin θ,  cos θ]]

def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = ![![1, 0],
       ![0, 1]]

theorem smallest_rotation_repetition :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → is_identity ((rotation_matrix (160 * π / 180)) ^ k) → n ≤ k) ∧
  is_identity ((rotation_matrix (160 * π / 180)) ^ n) :=
by sorry

#check smallest_rotation_repetition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_repetition_l407_40742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reasoning_is_deductive_l407_40762

-- Define the structure of a syllogism
structure Syllogism where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

-- Define the properties of numbers
def IsInfiniteDecimal (x : ℝ) : Prop := sorry

-- Define irrational numbers
def IsIrrational (x : ℝ) : Prop := IsInfiniteDecimal x

-- Define the given syllogism
def given_syllogism : Syllogism :=
  { major_premise := ∀ x : ℝ, IsIrrational x → IsInfiniteDecimal x
  , minor_premise := IsInfiniteDecimal (1/6 : ℝ)
  , conclusion := IsIrrational (1/6 : ℝ) }

-- Theorem to prove
theorem reasoning_is_deductive (s : Syllogism) : 
  (s.major_premise → s.minor_premise → s.conclusion) → 
  (s = given_syllogism) → 
  String :=
by
  intro h1 h2
  exact "deduction"

#check reasoning_is_deductive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reasoning_is_deductive_l407_40762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_negative_pi_sixth_f_range_l407_40760

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sin x

-- Theorem for part I
theorem f_at_negative_pi_sixth : f (-π/6) = -1/2 := by sorry

-- Theorem for part II (range)
theorem f_range :
  (∀ x : ℝ, -3 ≤ f x ∧ f x ≤ 3/2) ∧
  (∃ x₁ x₂ : ℝ, f x₁ = -3 ∧ f x₂ = 3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_negative_pi_sixth_f_range_l407_40760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_polynomials_l407_40739

/-- An algebraic expression is a polynomial if it consists of variables and coefficients, 
    that uses only the operations of addition, subtraction, multiplication, 
    and non-negative integer exponents. -/
def IsPolynomial (e : String) : Bool := sorry

def expressions : List String := [
  "1/x",
  "2x+y",
  "1/3*a^2*b",
  "(x-y)/π",
  "5y/(4x)",
  "0.5"
]

theorem count_polynomials :
  (expressions.filter IsPolynomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_polynomials_l407_40739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_16_9_l407_40702

theorem binomial_16_9 (h1 : Nat.choose 15 9 = 5005) 
                       (h2 : Nat.choose 15 8 = 6435) 
                       (h3 : Nat.choose 17 9 = 24310) : 
  Nat.choose 16 9 = 11440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_16_9_l407_40702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_equation_and_prediction_l407_40726

-- Define the number of families
def n : ℕ := 10

-- Define the sums given in the problem
noncomputable def sum_x : ℝ := 80
noncomputable def sum_y : ℝ := 20
noncomputable def sum_xy : ℝ := 184
noncomputable def sum_x_squared : ℝ := 720

-- Define mean functions
noncomputable def mean_x : ℝ := sum_x / n
noncomputable def mean_y : ℝ := sum_y / n

-- Define the regression coefficients
noncomputable def b_hat : ℝ := (sum_xy - n * mean_x * mean_y) / (sum_x_squared - n * mean_x^2)
noncomputable def a_hat : ℝ := mean_y - b_hat * mean_x

-- Define the regression function
noncomputable def regression_function (x : ℝ) : ℝ := b_hat * x + a_hat

-- Theorem to prove
theorem regression_equation_and_prediction :
  (b_hat = 0.3 ∧ a_hat = -0.4) ∧
  (regression_function 12 = 3.2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_equation_and_prediction_l407_40726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l407_40710

/-- The function g(x) = 2^(x^2-4x+3) - 2|x| -/
noncomputable def g (x : ℝ) : ℝ := 2^(x^2 - 4*x + 3) - 2 * |x|

/-- g is neither even nor odd -/
theorem g_neither_even_nor_odd :
  (∀ x, g (-x) = g x) → False ∧ (∀ x, g (-x) = -g x) → False := by
  sorry

#check g_neither_even_nor_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l407_40710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l407_40725

-- Define the function f(x) = log₍₁/₂₎(x² - 3x + 2)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x + 2) / Real.log (1/2)

-- Theorem statement
theorem f_increasing_on_interval :
  ∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l407_40725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_l407_40771

noncomputable section

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + m*x + m) / x

def g (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x^2 + a*x + 1
  else -x^2 + a*x + 1

-- State the theorem
theorem symmetry_and_range (m a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f m x + f m (-x) = 2) ∧
  (∀ x : ℝ, x ≠ 0 → g a x + g a (-x) = 2) ∧
  (∀ x t : ℝ, x < 0 → t > 0 → g a x < f m t) →
  m = 1 ∧
  (∀ x : ℝ, x < 0 → g a x = -x^2 + a*x + 1) ∧
  a > -2 * Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_l407_40771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_leaves_count_l407_40731

/-- Represents a tree on the farm -/
structure FarmTree where
  branches : Nat
  subBranchesPerBranch : Nat
  leavesPerSubBranch : Nat

/-- Calculates the total number of leaves on a single tree -/
def leavesOnTree (t : FarmTree) : Nat :=
  t.branches * t.subBranchesPerBranch * t.leavesPerSubBranch

theorem farm_leaves_count :
  let farmTree : FarmTree := {
    branches := 10,
    subBranchesPerBranch := 40,
    leavesPerSubBranch := 60
  }
  let numTrees : Nat := 4
  leavesOnTree farmTree * numTrees = 96000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_leaves_count_l407_40731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_loss_amount_l407_40714

noncomputable def stock_worth : ℝ := 16000

noncomputable def profit_percentage : ℝ := 20
noncomputable def loss_percentage : ℝ := 5

noncomputable def profit_stock_ratio : ℝ := 10 / 100
noncomputable def loss_stock_ratio : ℝ := 90 / 100

noncomputable def profit_amount : ℝ := stock_worth * profit_stock_ratio * (profit_percentage / 100)
noncomputable def loss_amount : ℝ := stock_worth * loss_stock_ratio * (loss_percentage / 100)

theorem overall_loss_amount :
  loss_amount - profit_amount = 400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_loss_amount_l407_40714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midsegment_avg_bases_equal_heights_isosceles_isosceles_not_imply_perp_diagonals_l407_40777

/-- A structure representing a trapezoid -/
structure Trapezoid where
  -- Let a and b be the lengths of the parallel sides (bases)
  a : ℝ
  b : ℝ
  -- Let c and d be the lengths of the non-parallel sides (legs)
  c : ℝ
  d : ℝ
  -- Let h be the height of the trapezoid
  h : ℝ
  -- Let m be the length of the midsegment
  m : ℝ
  -- Conditions for a valid trapezoid
  ha_pos : 0 < a
  hb_pos : 0 < b
  hc_pos : 0 < c
  hd_pos : 0 < d
  hh_pos : 0 < h
  hm_pos : 0 < m
  ha_ne_b : a ≠ b  -- Parallel sides are not equal

/-- The midsegment of a trapezoid is the average of its bases -/
theorem midsegment_avg_bases (T : Trapezoid) : T.m = (T.a + T.b) / 2 := by sorry

/-- Equal heights in a trapezoid imply it is isosceles -/
theorem equal_heights_isosceles (T : Trapezoid) (h1 h2 : ℝ) 
    (hh1 : h1 > 0) (hh2 : h2 > 0) (heq : h1 = h2) : T.c = T.d := by sorry

/-- A predicate indicating whether the diagonals of a trapezoid are perpendicular -/
def diagonals_perpendicular : Trapezoid → Prop := sorry

/-- An isosceles trapezoid may not have perpendicular diagonals -/
theorem isosceles_not_imply_perp_diagonals : ∃ T : Trapezoid, T.c = T.d ∧ ¬ (diagonals_perpendicular T) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midsegment_avg_bases_equal_heights_isosceles_isosceles_not_imply_perp_diagonals_l407_40777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l407_40729

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (3 * x) + Real.log 3

-- Theorem statement
theorem unique_intersection :
  ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l407_40729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_parked_l407_40791

theorem cars_parked (front_spaces back_spaces available_spaces : ℕ) 
  (h1 : front_spaces = 52)
  (h2 : back_spaces = 38)
  (h3 : available_spaces = 32) : 
  (front_spaces + back_spaces - available_spaces) = 58 := by
  let total_spaces := front_spaces + back_spaces
  let filled_back := back_spaces / 2
  let total_filled := total_spaces - available_spaces
  sorry

#check cars_parked

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_parked_l407_40791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l407_40701

-- Define the function f
def f (a m x : ℝ) : ℝ := x^3 + a*x^2 - a^2*x + m

-- State the theorem
theorem function_properties (a m : ℝ) (h_a : a > 0) :
  -- 1. Intervals of monotonicity
  (∀ x < -a, (deriv (f a m)) x > 0) ∧
  (∀ x > a/3, (deriv (f a m)) x > 0) ∧
  (∀ x ∈ Set.Ioo (-a) (a/3), (deriv (f a m)) x < 0) ∧
  
  -- 2. Range of a for no extremum points in [-1,1]
  (∀ a > 3, ∀ x ∈ Set.Icc (-1) 1, (deriv (f a m)) x ≠ 0) ∧
  
  -- 3. Range of m when f(x) ≤ 1 for x ∈ [-2,2] and a ∈ [3,6]
  (∀ a ∈ Set.Icc 3 6, 
   (∀ x ∈ Set.Icc (-2) 2, f a m x ≤ 1) → 
   m ≤ -87) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l407_40701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotone_f_l407_40799

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x + a)

-- State the theorem
theorem min_a_for_monotone_f :
  (∀ a : ℝ, (∀ x y : ℝ, 1 < x ∧ x < y → f a x ≤ f a y) →
   a ≥ -1) ∧
  (∃ a : ℝ, a = -1 ∧ ∀ x y : ℝ, 1 < x ∧ x < y → f a x ≤ f a y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotone_f_l407_40799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_decreasing_l407_40752

-- Define the function f(x) = 2^|x|
noncomputable def f (x : ℝ) : ℝ := 2^(abs x)

-- Statement to prove
theorem f_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y → x < 0 → y ≤ 0 → f y < f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_decreasing_l407_40752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_n_values_l407_40768

theorem number_of_n_values : 
  {n : ℕ | ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * y * n^2 = 720} = {1, 2, 3, 4, 6, 12} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_n_values_l407_40768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l407_40798

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt (1 + x) + Real.sqrt (1 - x)) * (2 * Real.sqrt (1 - x^2) - 1)

-- State the theorem about the range of f
theorem f_range :
  ∀ m : ℝ, (∃ x : ℝ, f x = m) ↔ -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l407_40798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_captain_age_l407_40719

theorem cricket_team_captain_age (team_size : ℕ) (team_avg_age : ℕ) 
  (h_team_size : team_size = 11)
  (h_team_avg : team_avg_age = 23)
  (captain_age wicket_keeper_age : ℕ)
  (h_wicket_keeper : wicket_keeper_age = captain_age + 1)
  (remaining_players_avg : ℝ)
  (h_remaining_avg : remaining_players_avg = team_avg_age - 1) :
  captain_age = 27 := by
  sorry

#check cricket_team_captain_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_captain_age_l407_40719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l407_40769

noncomputable section

-- Define the vertices A and B
def A : ℝ × ℝ := (5, 0)
def B : ℝ × ℝ := (0, 5)

-- Define the line on which C lies
def line (x y : ℝ) : Prop := x + y = 9

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop := line C.1 C.2

-- Define the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

-- Theorem statement
theorem area_of_triangle_ABC :
  ∀ C : ℝ × ℝ, triangle_ABC C → triangle_area A B C = 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l407_40769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_focus_l407_40789

/-- The conic section C is defined by x = t + 1/t and y = t - 1/t, where t is a parameter -/
noncomputable def conic_section (t : ℝ) : ℝ × ℝ :=
  (t + 1/t, t - 1/t)

/-- The focus of a hyperbola with semi-major axis a and semi-minor axis b -/
noncomputable def hyperbola_focus (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

/-- Theorem: The coordinates of the focus of the given conic section are (±2√2, 0) -/
theorem conic_section_focus :
  ∃ (a b : ℝ), a^2 = 4 ∧ b^2 = 4 ∧ 
  hyperbola_focus a b = 2 * Real.sqrt 2 := by
  sorry

#check conic_section_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_focus_l407_40789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_f_increasing_l407_40757

-- Define the function f(x) with parameter k
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x + k / (x - 1)

-- Theorem for part (1)
theorem min_value_f (x : ℝ) (h : x > 1) :
  f 4 x ≥ 5 ∧ f 4 3 = 5 := by
  sorry

-- Theorem for part (2)
theorem f_increasing (x1 x2 : ℝ) (h1 : x1 > 1) (h2 : x2 > 1) (h3 : x1 < x2) :
  f (-4) x1 < f (-4) x2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_f_increasing_l407_40757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_seven_equals_49_l407_40770

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 5
  fifth_term : a 5 = 9

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: The sum of the first 7 terms of the given arithmetic sequence is 49 -/
theorem sum_seven_equals_49 (seq : ArithmeticSequence) : sum_n seq 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_seven_equals_49_l407_40770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_and_max_value_l407_40793

-- Define the solution set of |x-2| > 3
def solution_set_1 : Set ℝ := {x | |x - 2| > 3}

-- Define the solution set of x^2 - ax - b > 0
def solution_set_2 (a b : ℝ) : Set ℝ := {x | x^2 - a*x - b > 0}

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := a * Real.sqrt (x - 3) + b * Real.sqrt (44 - x)

theorem solution_sets_and_max_value :
  ∃ (a b : ℝ), 
    (solution_set_1 = solution_set_2 a b) ∧ 
    (a = 4 ∧ b = 5) ∧
    (∀ x, x ∈ Set.Icc 3 44 → f a b x ≤ 41) ∧
    (∃ x, x ∈ Set.Icc 3 44 ∧ f a b x = 41) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_and_max_value_l407_40793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_non_defective_pencils_l407_40781

def total_pencils : ℕ := 8
def defective_pencils : ℕ := 2
def purchased_pencils : ℕ := 3

def probability_no_defective : ℚ := 5 / 14

theorem probability_three_non_defective_pencils :
  (Nat.choose (total_pencils - defective_pencils) purchased_pencils : ℚ) /
  (Nat.choose total_pencils purchased_pencils : ℚ) = probability_no_defective := by
  sorry

#eval (Nat.choose (total_pencils - defective_pencils) purchased_pencils : ℚ) /
      (Nat.choose total_pencils purchased_pencils : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_non_defective_pencils_l407_40781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_symmetry_l407_40744

/-- If one of the symmetry centers of y = sin(x + φ) is (π/6, 0), 
    then one of the symmetry axes of y = cos(x + φ) is x = π/6 -/
theorem sin_cos_symmetry (φ : ℝ) :
  (∃ (x : ℝ), x = π/6 ∧ (∀ (y : ℝ), Real.sin (x + φ) = y ↔ Real.sin (-x + φ) = -y)) →
  (∀ (y : ℝ), Real.cos (π/6 + φ + y) = Real.cos (π/6 + φ - y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_symmetry_l407_40744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_sticks_form_triangle_l407_40794

-- Define the set of stick lengths
def stickLengths : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 91) (Finset.range 92)

-- Define a function to check if three lengths can form a triangle
def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem eight_sticks_form_triangle :
  ∀ (S : Finset ℕ),
    S ⊆ stickLengths →
    S.card ≥ 8 →
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ canFormTriangle a b c :=
by
  sorry

#check eight_sticks_form_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_sticks_form_triangle_l407_40794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_lateral_face_angle_value_l407_40750

/-- Regular hexagonal pyramid with height equal to base side length -/
structure RegularHexagonalPyramid where
  side_length : ℝ
  height : ℝ
  height_eq_side : height = side_length

/-- The angle between the slant height (apothem) and an adjacent lateral face -/
noncomputable def apothem_lateral_face_angle (p : RegularHexagonalPyramid) : ℝ :=
  Real.arcsin (Real.sqrt 3 / 7)

/-- Theorem: In a regular hexagonal pyramid where the height is equal to the side length of the base,
    the angle between the slant height (apothem) and the plane of an adjacent lateral face is arcsin(√3/7) -/
theorem apothem_lateral_face_angle_value (p : RegularHexagonalPyramid) :
  apothem_lateral_face_angle p = Real.arcsin (Real.sqrt 3 / 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_lateral_face_angle_value_l407_40750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_iff_l407_40749

/-- The function f(x) = (x+1)e^x - a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.exp x - a

/-- The property of f having exactly two zeros -/
def has_exactly_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧
  ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂

/-- The main theorem stating the condition for f to have exactly two zeros -/
theorem f_two_zeros_iff (a : ℝ) :
  has_exactly_two_zeros a ↔ -1 / Real.exp 2 < a ∧ a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_iff_l407_40749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_product_l407_40709

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope k passing through a point -/
structure Line where
  k : ℝ
  p : Point

noncomputable def focal_length (e : Ellipse) : ℝ := 
  Real.sqrt (e.a^2 - e.b^2)

def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

noncomputable def intersect_ellipse (e : Ellipse) (l : Line) : Point :=
  sorry

theorem ellipse_intersection_product (e : Ellipse) (p : Point) (k₁ k₂ : ℝ) :
  focal_length e = 2 * Real.sqrt 3 →
  on_ellipse e ⟨Real.sqrt 3, 1/2⟩ →
  let l₁ : Line := ⟨k₁, ⟨-2, 0⟩⟩
  let l₂ : Line := ⟨k₂, ⟨-2, 0⟩⟩
  let m := intersect_ellipse e l₁
  let n := intersect_ellipse e l₂
  m.x = n.x →
  k₁ * k₂ = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_product_l407_40709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_approx_74_07_l407_40785

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- The length of the longer base of the trapezoid -/
  longerBase : ℝ
  /-- One of the base angles of the trapezoid -/
  baseAngle : ℝ
  /-- Condition that the trapezoid is isosceles -/
  isIsosceles : True
  /-- Condition that the trapezoid is circumscribed around a circle -/
  isCircumscribed : True
  /-- Condition that the longer base is 20 -/
  longerBaseIs20 : longerBase = 20
  /-- Condition that the base angle is arcsin(0.6) -/
  baseAngleIsArcsin06 : baseAngle = Real.arcsin 0.6

/-- The area of an isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the described trapezoid is approximately 74.07 -/
theorem trapezoid_area_is_approx_74_07 (t : IsoscelesTrapezoid) :
  ‖area t - 74.07‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_approx_74_07_l407_40785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_2125_l407_40728

/-- Represents the selling price of type A disinfectant water -/
def x : ℕ := sorry

/-- The sales volume of type A disinfectant water -/
def sales_volume_A (x : ℕ) : ℤ := 250 - 5 * x

/-- The total cost price of type B disinfectant water -/
def total_cost_B (x : ℕ) : ℤ := 100 * x - 3000

/-- The sales volume of type B disinfectant water -/
def sales_volume_B (x : ℕ) : ℚ := (10 * x : ℚ) / 3 - 100

/-- The total profit from selling both types of disinfectant water -/
def total_profit (x : ℕ) : ℚ := (sales_volume_A x : ℚ) * (x - 20) + sales_volume_B x * 30

theorem max_profit_is_2125 :
  ∃ (max_profit : ℚ), max_profit = 2125 ∧ 
    ∀ y : ℕ, y > 30 → y ≤ 50 → y % 3 = 0 → total_profit y ≤ max_profit :=
by
  sorry

#check max_profit_is_2125

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_2125_l407_40728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_7_to_2023_l407_40764

/-- The function that returns the last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- The function that represents the pattern of last two digits of 7^n -/
def powerSevenPattern (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 49
  | 3 => 43
  | _ => 0  -- Adding a catch-all case to handle all possible inputs

theorem last_two_digits_of_7_to_2023 :
  lastTwoDigits (7^2023) = powerSevenPattern 2023 := by
  sorry

#eval powerSevenPattern 2023  -- This will evaluate to 43

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_7_to_2023_l407_40764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_dihedral_angle_largest_dihedral_angle_deg_cos_120_deg_l407_40704

/-- A pyramid with a square base and a lateral edge coinciding with its height -/
structure SpecialPyramid where
  /-- The side length of the square base -/
  baseSideLength : ℝ
  /-- The length of the lateral edge that coincides with the height -/
  heightLength : ℝ
  /-- The base side length is 1 -/
  base_is_unit : baseSideLength = 1
  /-- The height length is 1 -/
  height_is_unit : heightLength = 1

/-- The maximum dihedral angle in a special pyramid -/
noncomputable def maxDihedralAngle (p : SpecialPyramid) : ℝ :=
  Real.arccos (-1/2)

/-- The maximum dihedral angle in a special pyramid in degrees -/
noncomputable def maxDihedralAngleDegrees (p : SpecialPyramid) : ℝ :=
  180 / Real.pi * maxDihedralAngle p

/-- The largest dihedral angle in a special pyramid is 120° -/
theorem largest_dihedral_angle (p : SpecialPyramid) : 
  Real.cos (maxDihedralAngle p) = -1/2 := by
  sorry

/-- The largest dihedral angle in a special pyramid is 120° -/
theorem largest_dihedral_angle_deg (p : SpecialPyramid) : 
  maxDihedralAngleDegrees p = 120 := by
  sorry

/-- The cosine of 120° is -1/2 -/
theorem cos_120_deg : Real.cos (2 * Real.pi / 3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_dihedral_angle_largest_dihedral_angle_deg_cos_120_deg_l407_40704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l407_40727

/-- Represents a regular hexagonal dartboard with a central hexagon -/
structure HexagonalDartboard where
  /-- Side length of the large hexagon -/
  side_length : ℝ
  /-- Assumption that the side length is positive -/
  side_length_pos : side_length > 0

/-- Calculates the area of a regular hexagon given its side length -/
noncomputable def hexagon_area (s : ℝ) : ℝ := 3 * Real.sqrt 3 / 2 * s^2

/-- Theorem: The probability of a dart landing in the central hexagon is 1/4 -/
theorem dart_probability (board : HexagonalDartboard) : 
  (hexagon_area (board.side_length / 2)) / (hexagon_area board.side_length) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l407_40727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l407_40753

/-- Predicate to represent that a point (x, y) lies on an ellipse -/
def IsEllipse (x y : ℝ) : Prop := sorry

/-- Predicate to represent that the foci of an ellipse are on the y-axis -/
def FociOnYAxis (a b : ℝ) : Prop := sorry

/-- 
Given that x^2 + ky^2 = 2 represents an ellipse with foci on the y-axis,
prove that 0 < k < 1.
-/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 + k*y^2 = 2 → IsEllipse x y) ∧ 
  (∃ a b : ℝ, FociOnYAxis a b) →
  0 < k ∧ k < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l407_40753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_conditions_l407_40776

theorem count_integers_satisfying_conditions : 
  (∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ 7 ∣ n ∧ Nat.lcm (Nat.factorial 7) n = 7 * Nat.gcd (Nat.factorial 14) n) ∧ 
    Finset.card S = 192) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_conditions_l407_40776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_is_five_l407_40759

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 3)
def b (t : ℝ) : ℝ × ℝ := (1, t)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a 2D vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the perpendicularity condition
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- State the theorem
theorem vector_magnitude_is_five :
  ∃ t : ℝ, perpendicular (a - 2 • (b t)) a → magnitude (a + b t) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_is_five_l407_40759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l407_40778

-- Define the vectors a and b
def a : Fin 3 → ℝ := ![5, -3, 2]
def b : Fin 3 → ℝ := ![-2, 4, 1]

-- Theorem statement
theorem vector_subtraction : a - (4 • b) = ![13, -19, -2] := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l407_40778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_triplets_l407_40763

theorem congruence_triplets :
  ∀ (p q : Nat) (n : Nat),
    Nat.Prime p → Nat.Prime q → p % 2 = 1 → q % 2 = 1 → n > 1 →
    (q^(n+2) % p^n = 3^(n+2) % p^n) →
    (p^(n+2) % q^n = 3^(n+2) % q^n) →
    p = 3 ∧ q = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_triplets_l407_40763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l407_40712

/-- The time taken for a group of workers to complete a job -/
noncomputable def time_to_complete (rate : ℝ) : ℝ := 1 / rate

/-- The combined work rate of multiple workers -/
noncomputable def combined_rate (rates : List ℝ) : ℝ := rates.sum

theorem work_completion_time 
  (rate_ab rate_bc rate_ac rate_a rate_b rate_c : ℝ) :
  rate_ab = 1 / 20 →
  rate_bc = 1 / 30 →
  rate_ac = 1 / 30 →
  rate_ab = rate_a + rate_b →
  rate_bc = rate_b + rate_c →
  rate_ac = rate_a + rate_c →
  time_to_complete (combined_rate [rate_a, rate_b, rate_c]) = 120 / 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l407_40712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_store_spending_proof_l407_40773

def weekly_allowance : ℚ := 281/100

def arcade_spending : ℚ := (3/5) * weekly_allowance

def remaining_after_arcade : ℚ := weekly_allowance - arcade_spending

def toy_store_spending : ℚ := (1/3) * remaining_after_arcade

def candy_store_spending : ℚ := remaining_after_arcade - toy_store_spending

theorem candy_store_spending_proof :
  (candy_store_spending * 100000).num / (candy_store_spending * 100000).den = 74933 / 100000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_store_spending_proof_l407_40773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_heights_theorem_l407_40738

def is_valid_height_sequence (heights : List ℕ) : Prop :=
  heights.length = 5 ∧
  ∀ i, i < 4 → (heights.get! i = 2 * heights.get! (i+1) ∨ heights.get! i * 2 = heights.get! (i+1))

theorem tree_heights_theorem (heights : List ℕ) :
  is_valid_height_sequence heights →
  heights.get! 1 = 11 →
  (heights.sum : ℚ) / 5 = 24.2 →
  heights.sum = 121 ∧
  ∃ (h₁ h₃ h₄ h₅ : ℕ), heights = [h₁, 11, h₃, h₄, h₅] ∧
                       h₁ + h₃ = 22 ∧
                       h₄ = 44 ∧
                       h₅ = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_heights_theorem_l407_40738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_max_no_min_l407_40795

/-- Given a function f(x) = sin(ωx + π/4) with ω > 0, prove that f(x) has a maximum value
    but no minimum value on the interval (π/12, π/3) if and only if ω ∈ (3/4, 3). -/
theorem sine_function_max_no_min (ω : ℝ) (h_ω_pos : ω > 0) :
  (∃ (max : ℝ), ∀ x ∈ Set.Ioo (π / 12) (π / 3), Real.sin (ω * x + π / 4) ≤ max) ∧
  (¬∃ (min : ℝ), ∀ x ∈ Set.Ioo (π / 12) (π / 3), Real.sin (ω * x + π / 4) ≥ min) ↔
  ω ∈ Set.Ioo (3 / 4) 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_max_no_min_l407_40795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CPQ_l407_40703

-- Define the parallelogram ABCD
def ABCD : Set (ℝ × ℝ) := sorry

-- Define the area function
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

-- Define the line DQ
def DQ : Set (ℝ × ℝ) := sorry

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the quadrilateral ABPD
def ABPD : Set (ℝ × ℝ) := sorry

-- Define the triangle CPQ
def CPQ : Set (ℝ × ℝ) := sorry

-- Define IsParallelogram
def IsParallelogram (S : Set (ℝ × ℝ)) : Prop := sorry

-- Define Line type and extend function
def Line : Type := sorry
def Line.extend (l : Line) : Set (ℝ × ℝ) := sorry

-- Define AB as a Line
def AB : Line := sorry

-- Theorem statement
theorem area_of_triangle_CPQ 
  (h1 : IsParallelogram ABCD)
  (h2 : area ABCD = 60)
  (h3 : P ∈ DQ ∩ ABCD)
  (h4 : Q ∈ DQ ∩ (Line.extend AB))
  (h5 : area ABPD = 46) :
  area CPQ = 128 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CPQ_l407_40703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_properties_l407_40740

/-- Two distinct points in the plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point2D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Properties of points A and B -/
theorem point_properties (a b : ℝ) : 
  let A : Point2D := ⟨3, a - 1⟩
  let B : Point2D := ⟨b + 1, -2⟩
  (B.x = 0 → b = -1) ∧ 
  (A.x = A.y → a = 4) ∧
  (A.x = B.x ∧ distance A B = 5 → b = 2 ∧ (a = 4 ∨ a = -6)) :=
by
  sorry

#check point_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_properties_l407_40740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_7a_minus_9b_l407_40717

theorem max_value_7a_minus_9b (r₁ r₂ r₃ a b : ℝ) : 
  (∀ i ∈ ({r₁, r₂, r₃} : Set ℝ), 0 < i ∧ i < 1) →
  r₁ * r₂ * r₃ = b →
  r₁ * r₂ + r₂ * r₃ + r₃ * r₁ = a →
  r₁ + r₂ + r₃ = 1 →
  r₁^3 - r₁^2 + a*r₁ - b = 0 →
  r₂^3 - r₂^2 + a*r₂ - b = 0 →
  r₃^3 - r₃^2 + a*r₃ - b = 0 →
  7*a - 9*b ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_7a_minus_9b_l407_40717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l407_40792

/-- The length of each circular arc -/
noncomputable def arc_length : ℝ := 3 * Real.pi / 4

/-- The number of circular arcs -/
def num_arcs : ℕ := 8

/-- The side length of the regular octagon -/
def octagon_side : ℝ := 3

/-- The area enclosed by the curve composed of circular arcs -/
noncomputable def enclosed_area (arc_length : ℝ) (num_arcs : ℕ) (octagon_side : ℝ) : ℝ := 
  54 + 54 * Real.sqrt 2 + 3 * Real.pi

/-- Theorem stating that the enclosed area is equal to the given expression -/
theorem enclosed_area_theorem :
  enclosed_area arc_length num_arcs octagon_side = 54 + 54 * Real.sqrt 2 + 3 * Real.pi := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l407_40792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_one_third_not_increasing_l407_40732

/-- The exponential function with base 1/3 -/
noncomputable def f (x : ℝ) : ℝ := (1/3) ^ x

/-- The exponential function f is not increasing -/
theorem exp_one_third_not_increasing :
  ¬(∀ x y : ℝ, x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_one_third_not_increasing_l407_40732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parameters_l407_40733

/-- Represents a circle in the x-y plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- The general equation of the circle is x^2 + y^2 + D*x + E*y + F = 0 -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y + c.F = 0

/-- The center of the circle -/
noncomputable def Circle.center (c : Circle) : ℝ × ℝ :=
  (-c.D / 2, -c.E / 2)

/-- The radius of the circle -/
noncomputable def Circle.radius (c : Circle) : ℝ :=
  (1 / 2) * Real.sqrt (c.D^2 + c.E^2 - 4 * c.F)

theorem circle_parameters (c : Circle) 
  (h_center : c.center = (-2, 3))
  (h_radius : c.radius = 4) :
  c.D = 4 ∧ c.E = -6 ∧ c.F = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parameters_l407_40733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_plus_two_i_l407_40706

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number expression to be evaluated -/
noncomputable def expression : ℂ := (5 * i) / (2 + i) * i

/-- Theorem stating that the expression equals 1 + 2i -/
theorem expression_equals_one_plus_two_i : expression = 1 + 2 * i := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_plus_two_i_l407_40706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_expression_l407_40700

/-- Two concentric circles with an inscribed equilateral triangle and a point on the outer circle -/
structure CircleTriangleConfig where
  r : ℝ  -- radius of smaller circle
  R : ℝ  -- radius of larger circle
  A : ℝ × ℝ  -- vertex A of equilateral triangle
  B : ℝ × ℝ  -- vertex B of equilateral triangle
  C : ℝ × ℝ  -- vertex C of equilateral triangle
  P : ℝ × ℝ  -- point on larger circle

/-- The area of a triangle with side lengths PA, PB, PC -/
noncomputable def triangleArea (config : CircleTriangleConfig) : ℝ := sorry

/-- Helper function to check if a point is on a circle -/
def isOnCircle (P : ℝ × ℝ) (center : ℝ × ℝ) (R : ℝ) : Prop :=
  (P.1 - center.1)^2 + (P.2 - center.2)^2 = R^2

/-- Helper function to check if three points form an equilateral triangle -/
def isEquilateral (A B C : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let d2 := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let d3 := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  d1 = d2 ∧ d2 = d3

/-- Main theorem: The area can be expressed as (a√b)/c where a = 4013, b = 3, c = 4 -/
theorem area_expression (config : CircleTriangleConfig) 
  (h1 : config.r = 2006)
  (h2 : config.R = 2007)
  (h3 : isEquilateral config.A config.B config.C)
  (h4 : isOnCircle config.P (0, 0) config.R) :
  ∃ (a b c : ℕ), 
    triangleArea config = (a : ℝ) * Real.sqrt b / c ∧ 
    a = 4013 ∧ 
    b = 3 ∧ 
    c = 4 ∧
    Nat.Coprime a c ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ b)) ∧
    a + b + c = 4020 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_expression_l407_40700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_contribution_for_books_l407_40716

/-- The amount Alice needs to contribute to buy books -/
noncomputable def alices_contribution (book_cost : ℝ) (bens_contribution : ℝ) (exchange_rate : ℝ) : ℝ :=
  book_cost - (bens_contribution / exchange_rate)

/-- Theorem stating Alice's required contribution -/
theorem alice_contribution_for_books :
  let book_cost : ℝ := 15
  let bens_contribution : ℝ := 20
  let exchange_rate : ℝ := 1.5
  ∃ ε > 0, |alices_contribution book_cost bens_contribution exchange_rate - 1.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_contribution_for_books_l407_40716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_correct_l407_40786

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 / Real.sqrt (1 - x) + Real.log (2 * x + 1)

-- Define the domain of f
def domain_of_f : Set ℝ := {x | -1/2 < x ∧ x < 1}

-- Theorem statement
theorem domain_of_f_is_correct :
  {x : ℝ | ∃ y, f x = y} = domain_of_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_correct_l407_40786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_from_lcm_gcd_l407_40754

theorem product_from_lcm_gcd (a b : ℕ) : 
  Nat.lcm a b = 24 → Nat.gcd a b = 8 → a * b = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_from_lcm_gcd_l407_40754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_evaluation_l407_40708

theorem ceiling_evaluation : ⌈(5 : ℝ) * ((8 : ℝ) - 3/4) - 5/2⌉ = 34 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_evaluation_l407_40708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_travel_time_l407_40790

/-- Proves that the total travel time for a boat is 19 hours given specific conditions --/
theorem boat_travel_time (stream_velocity boat_speed distance_AB : ℝ) :
  stream_velocity = 4 →
  boat_speed = 14 →
  distance_AB = 180 →
  (distance_AB / (boat_speed + stream_velocity)) +
  ((distance_AB / 2) / (boat_speed - stream_velocity)) = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_travel_time_l407_40790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_l407_40722

/-- Represents a trapezium with given parameters -/
structure Trapezium where
  a : ℝ  -- Length of one parallel side
  b : ℝ  -- Length of the other parallel side
  area : ℝ  -- Area of the trapezium
  h : ℝ  -- Distance between parallel sides

/-- The area formula for a trapezium -/
noncomputable def trapezium_area_formula (t : Trapezium) : ℝ :=
  (1/2) * (t.a + t.b) * t.h

/-- Theorem: For a trapezium with parallel sides 20 cm and 18 cm, 
    and area 380 square cm, the distance between parallel sides is 20 cm -/
theorem trapezium_height (t : Trapezium) 
    (h1 : t.a = 20) 
    (h2 : t.b = 18) 
    (h3 : t.area = 380) 
    (h4 : t.area = trapezium_area_formula t) : 
  t.h = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_l407_40722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_r_value_l407_40734

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def S (n : ℕ+) (q : ℝ) (r : ℝ) : ℝ := q^(n : ℝ) + r

/-- Represents the nth term of the geometric sequence -/
noncomputable def a (n : ℕ+) (q : ℝ) (r : ℝ) : ℝ :=
  if n = 1 then S 1 q r else S n q r - S (n-1) q r

theorem geometric_sequence_r_value (q : ℝ) (h_q_pos : q > 0) (h_q_neq_1 : q ≠ 1) :
  ∃ r : ℝ, ∀ n : ℕ+, ∃ k : ℝ, a (n+1) q r = k * a n q r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_r_value_l407_40734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l407_40761

theorem remainder_sum_mod_30 (a b c : ℕ) 
  (ha : a % 30 = 14)
  (hb : b % 30 = 5)
  (hc : c % 30 = 18) :
  (a + b + c) % 30 = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l407_40761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l407_40787

/-- The function c(x) with parameter a -/
noncomputable def c (a : ℝ) (x : ℝ) : ℝ := (a * x^2 - 3 * x + 4) / (-3 * x^2 - 3 * x + a)

/-- The domain of c(x) is all real numbers iff a < -3/4 -/
theorem domain_c_all_reals (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, c a x = y) ↔ a < -3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l407_40787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_expression_l407_40780

theorem tan_two_expression (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.tan α = 2) : 
  (Real.sin (2 * α) + 1) / (Real.cos α ^ 4 - Real.sin α ^ 4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_expression_l407_40780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l407_40715

/-- The volume of a cone created from a 270-degree sector of a circle with radius 15 cm, divided by π -/
noncomputable def cone_volume_divided_by_pi : ℝ :=
  let r : ℝ := 15 -- radius of the original circle
  let sector_angle : ℝ := 270 -- angle of the sector in degrees
  let base_radius : ℝ := r * (sector_angle / 360) -- radius of the cone's base
  let height : ℝ := Real.sqrt (r^2 - base_radius^2) -- height of the cone
  (1/3) * base_radius^2 * height -- volume of the cone divided by π

/-- Theorem stating that the volume of the cone divided by π is equal to 1184.59375 -/
theorem cone_volume_theorem : 
  cone_volume_divided_by_pi = 1184.59375 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l407_40715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_six_l407_40724

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- A circle in the xy-plane -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℕ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is inside a circle -/
def is_inside (c : Circle) (p : Point) : Prop :=
  distance c.center_x c.center_y p.x p.y < c.radius

/-- Predicate to check if a point is outside a circle -/
def is_outside (c : Circle) (p : Point) : Prop :=
  distance c.center_x c.center_y p.x p.y > c.radius

theorem circle_radius_is_six (c : Circle) (p_inside p_outside : Point) : 
  c.center_x = -2 ∧ 
  c.center_y = -3 ∧ 
  p_inside.x = -2 ∧ 
  p_inside.y = 2 ∧ 
  p_outside.x = 5 ∧ 
  p_outside.y = -3 ∧ 
  is_inside c p_inside ∧ 
  is_outside c p_outside → 
  c.radius = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_six_l407_40724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equation_C_point_D_coordinates_l407_40797

-- Define the semicircle C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the line l
def l (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

-- Theorem for the parametric equation of C
theorem parametric_equation_C :
  ∀ α : ℝ, α ∈ Set.Icc (-Real.pi/2) (Real.pi/2) →
  ∃ θ : ℝ, θ ∈ Set.Icc 0 (Real.pi/2) ∧
  C θ * Real.cos θ = Real.cos α ∧
  C θ * Real.sin θ = 1 + Real.sin α := by
  sorry

-- Theorem for the coordinates of point D
theorem point_D_coordinates :
  ∃ D : ℝ × ℝ,
  D.1 = Real.sqrt 3 / 2 ∧ D.2 = 3 / 2 ∧
  ∃ α : ℝ, α ∈ Set.Icc (-Real.pi/2) (Real.pi/2) ∧
  D.1 = Real.cos α ∧ D.2 = 1 + Real.sin α ∧
  (∀ x y : ℝ, l x y → (x - D.1) * (y - 1) + (y - D.2) * D.1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equation_C_point_D_coordinates_l407_40797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_shaded_areas_l407_40775

theorem equality_of_shaded_areas (φ : Real) (h1 : 0 < φ) (h2 : φ < π / 4) :
  (∃ r : Real, r > 0 ∧
    (r^2 * Real.tan φ / 2 - φ * r^2 / 2 = φ * r^2 / 2)) ↔ Real.tan φ = 2 * φ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_shaded_areas_l407_40775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l407_40747

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

theorem function_properties (ω : ℝ) (h_ω : ω > 0) :
  (∃ x₁ x₂ : ℝ, |f ω x₁ - f ω x₂| = 4 ∧ ∀ y₁ y₂ : ℝ, |f ω y₁ - f ω y₂| = 4 → |y₁ - y₂| ≥ Real.pi / 2) →
  (ω = 2 ∧
   (∀ φ : ℝ, 0 < φ → φ < Real.pi / 2 →
     (∀ x : ℝ, f ω (x + φ) = f ω (-x + φ)) → φ = Real.pi / 6) ∧
   (∃ r₁ r₂ r₃ r₄ : ℝ, r₁ ∈ Set.Ioo (Real.pi / 16) (2 * Real.pi) ∧
                       r₂ ∈ Set.Ioo (Real.pi / 16) (2 * Real.pi) ∧
                       r₃ ∈ Set.Ioo (Real.pi / 16) (2 * Real.pi) ∧
                       r₄ ∈ Set.Ioo (Real.pi / 16) (2 * Real.pi) ∧
                       r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
                       f ω r₁ = Real.sqrt 3 ∧ f ω r₂ = Real.sqrt 3 ∧
                       f ω r₃ = Real.sqrt 3 ∧ f ω r₄ = Real.sqrt 3) ∧
   (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 12) m → f ω x ≥ -2) →
     (∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 12) m → f ω x > -2) →
     m ≥ 2 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l407_40747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radii_relation_l407_40782

theorem triangle_radii_relation (a b c r r₁ r₂ r₃ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hr : r > 0) (hr₁ : r₁ > 0) (hr₂ : r₂ > 0) (hr₃ : r₃ > 0)
  (h_inradius : r = (a + b + c) / (2 * (a * b * c)^(1/2)))
  (h_exradius1 : r₁ = (a + b + c) / (2 * ((b + c - a) * (a * b * c)^(1/2))))
  (h_exradius2 : r₂ = (a + b + c) / (2 * ((a + c - b) * (a * b * c)^(1/2))))
  (h_exradius3 : r₃ = (a + b + c) / (2 * ((a + b - c) * (a * b * c)^(1/2))))
  : 1/r = 1/r₁ + 1/r₂ + 1/r₃ := by
  sorry

#check triangle_radii_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radii_relation_l407_40782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zach_tickets_needed_l407_40730

/-- Calculates the number of tickets needed for three rides with given costs and discounts -/
noncomputable def tickets_needed (ferris_wheel_cost roller_coaster_cost bumper_cars_cost 
                    additional_ride_discount max_additional_ride_discounts
                    newspaper_coupon teacher_discount : ℚ) : ℚ :=
  let total_cost := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost
  let additional_ride_discounts := min 2 max_additional_ride_discounts * additional_ride_discount
  let total_discounts := additional_ride_discounts + newspaper_coupon + teacher_discount
  total_cost - total_discounts

/-- Theorem stating that Zach needs 12.0 tickets for the three rides -/
theorem zach_tickets_needed :
  tickets_needed (35/10) 8 5 (1/2) 2 (3/2) 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zach_tickets_needed_l407_40730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_sequence_determinable_l407_40784

/-- Represents a coin that can be either genuine or counterfeit -/
inductive Coin
  | genuine
  | counterfeit

/-- Represents the weight of a coin -/
def weight (c : Coin) : ℝ :=
  match c with
  | Coin.genuine => 1
  | Coin.counterfeit => 0.5

/-- A row of 100 coins -/
def CoinRow := Fin 100 → Coin

/-- Predicate to check if a coin is counterfeit -/
def is_counterfeit (c : Coin) : Prop :=
  c = Coin.counterfeit

/-- Predicate to check if a subsequence of coins is all counterfeit -/
def all_counterfeit (row : CoinRow) (start : Fin 100) (end_ : Fin 100) : Prop :=
  ∀ i : Fin 100, start ≤ i ∧ i ≤ end_ → is_counterfeit (row i)

/-- The main theorem stating that the counterfeit sequence can be determined -/
theorem counterfeit_sequence_determinable (row : CoinRow) :
  (∃ start end_ : Fin 100, start ≤ end_ ∧ end_ - start = 25 ∧ all_counterfeit row start end_) →
  (∀ i j : Fin 100, i ≠ j → (is_counterfeit (row i) ∧ is_counterfeit (row j)) → 
    ∃ k : Fin 100, ∀ start end_ : Fin 100, (start ≤ end_ ∧ end_ - start = 25 ∧ all_counterfeit row start end_) → start ≤ k ∧ k ≤ end_) →
  (∀ c : Coin, is_counterfeit c → weight c < weight Coin.genuine) →
  ∃ start end_ : Fin 100, start ≤ end_ ∧ end_ - start = 25 ∧ all_counterfeit row start end_ ∧
    (weight (row 25) ≠ weight (row 51) ∨ weight (row 51) ≠ weight (row 77) ∨ weight (row 25) ≠ weight (row 77)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_sequence_determinable_l407_40784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l407_40741

variable (a b : ℝ)

-- Define the solution set of the first inequality
def solution_set_1 (a b : ℝ) : Set ℝ := {x | 2 < x}

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x | x < -1 ∨ 1 < x}

-- Theorem statement
theorem inequality_solution_sets 
  (h : ∀ x, a * x - 2 * b < 0 ↔ x ∈ solution_set_1 a b) : 
  ∀ x, (x - 1) * (a * x + b) < 0 ↔ x ∈ solution_set_2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l407_40741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angle_central_angle_circular_l407_40743

/-- Angle formed by two intersecting chords in a circle -/
noncomputable def chord_intersection_angle (α β : ℝ) : ℝ := (α + β) / 2

/-- Central angle in a circle -/
noncomputable def central_angle (α : ℝ) : ℝ := α

/-- Theorem stating that using the chord intersection angle formula
    to derive the central angle measure leads to circular reasoning -/
theorem chord_angle_central_angle_circular :
  ∀ α : ℝ, 
  (∀ β : ℝ, chord_intersection_angle α β = (α + β) / 2) →
  (central_angle α = α) →
  (∃ γ : ℝ, chord_intersection_angle α α = γ ∧ γ = α) →
  CircularReasoning :=
by
  sorry

/-- Definition of circular reasoning -/
inductive CircularReasoning : Prop where
  | circular : CircularReasoning

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angle_central_angle_circular_l407_40743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l407_40705

theorem polynomial_divisibility (n : ℕ) (k : ℤ) :
  let f : ℤ → ℤ := λ x => x^(n + 2) + (x + 1)^(2*n + 1)
  ∃ m : ℤ, f k = m * (k^2 + k + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l407_40705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_is_e_conditions_hold_for_e_l407_40711

/-- The maximum value of m for which the given conditions hold. -/
noncomputable def max_m : ℝ := Real.exp 1

/-- The theorem stating that e is the maximum value of m for which the given conditions hold. -/
theorem max_m_is_e :
  ∀ m : ℝ,
  (m > 0) →
  (∀ x₁ x₂ : ℝ,
    x₁ ∈ Set.Ioo 0 m →
    x₂ ∈ Set.Ioo 0 m →
    x₁ < x₂ →
    x₁^x₂ < x₂^x₁) →
  m ≤ max_m :=
by sorry

/-- The theorem stating that for m = e, there exist x₁ and x₂ satisfying the conditions. -/
theorem conditions_hold_for_e :
  ∃ x₁ x₂ : ℝ,
    x₁ ∈ Set.Ioo 0 max_m ∧
    x₂ ∈ Set.Ioo 0 max_m ∧
    x₁ < x₂ ∧
    x₁^x₂ < x₂^x₁ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_is_e_conditions_hold_for_e_l407_40711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_relation_l407_40707

-- Define the triangle ABC and points D, E, F in ℝ²
variable (A B C D E F : EuclideanPlane) 

-- Define the properties of the triangle and points
def is_acute_triangle (A B C : EuclideanPlane) : Prop := sorry

def is_height (A B C D : EuclideanPlane) : Prop := sorry

def intersect_at (A B C D E F : EuclideanPlane) : Prop := sorry

noncomputable def angle (A B C : EuclideanPlane) : ℝ := sorry

noncomputable def area (A B C : EuclideanPlane) : ℝ := sorry

-- State the theorem
theorem triangle_area_relation
  (h_acute : is_acute_triangle A B C)
  (h_height_BD : is_height A B C D)
  (h_height_CE : is_height A C B E)
  (h_intersect : intersect_at B C D E F F)
  (h_angle : angle B A C = 45)
  (h_area_DEF : area D E F = S)
  : area B F C = 2 * S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_relation_l407_40707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insphere_touches_centroid_l407_40772

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  -- Add necessary fields (placeholder)
  dummy : Unit

/-- An insphere of a regular tetrahedron -/
structure Insphere (t : RegularTetrahedron) where
  -- Add necessary fields (placeholder)
  dummy : Unit

/-- A face of a regular tetrahedron -/
structure TetrahedronFace (t : RegularTetrahedron) where
  -- Add necessary fields (placeholder)
  dummy : Unit

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The centroid of an equilateral triangle -/
def centroid (t : RegularTetrahedron) (f : TetrahedronFace t) : Point :=
  { x := 0, y := 0, z := 0 } -- Placeholder implementation

/-- The point where the insphere touches a face -/
def touchPoint (t : RegularTetrahedron) (i : Insphere t) (f : TetrahedronFace t) : Point :=
  { x := 0, y := 0, z := 0 } -- Placeholder implementation

/-- Theorem: The insphere of a regular tetrahedron touches each face at the centroid of that face -/
theorem insphere_touches_centroid (t : RegularTetrahedron) (i : Insphere t) (f : TetrahedronFace t) :
  touchPoint t i f = centroid t f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insphere_touches_centroid_l407_40772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cubic_solution_mod_prime_l407_40765

theorem unique_cubic_solution_mod_prime (p : ℕ) (hp : Nat.Prime p) :
  (∃! a : ZMod p, (a^3 - 3*a + 1 : ZMod p) = 0) ↔ p = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cubic_solution_mod_prime_l407_40765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_lower_bound_l407_40721

theorem triangle_perimeter_lower_bound (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_area : (a * b * c) / (4 * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c) / 16)) = 1/2) :
  a + b + c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_lower_bound_l407_40721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt_three_over_two_l407_40736

theorem cos_squared_difference_equals_sqrt_three_over_two :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt_three_over_two_l407_40736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_remainder_l407_40723

theorem pie_remainder : ∃ (carlos_portion maria_fraction remainder_portion maria_portion final_remainder : Real),
  carlos_portion = 0.6 ∧
  maria_fraction = 1/4 ∧
  remainder_portion = 1 - carlos_portion ∧
  maria_portion = maria_fraction * remainder_portion ∧
  final_remainder = remainder_portion - maria_portion ∧
  final_remainder = 0.3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_remainder_l407_40723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l407_40713

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_theorem :
  ∃ α : ℝ, (f α (-2) = -8 ∧ ∃ x : ℝ, f α x = 27 ∧ x = 3) :=
by
  -- Provide the value of α
  use 3
  constructor
  -- Prove f 3 (-2) = -8
  · simp [f]
    norm_num
  -- Prove ∃ x : ℝ, f 3 x = 27 ∧ x = 3
  · use 3
    constructor
    · simp [f]
      norm_num
    · rfl
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l407_40713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_iteration_to_repeat_l407_40756

def f (x : ℕ) : ℕ :=
  if x % 4 = 0 ∧ x % 9 = 0 then x / 36
  else if x % 9 = 0 then 4 * x
  else if x % 4 = 0 then 9 * x
  else x + 4

def f_iterate : ℕ → ℕ → ℕ
| 0, x => x
| (n + 1), x => f (f_iterate n x)

theorem smallest_iteration_to_repeat (a : ℕ) : 
  (a > 1 ∧ f_iterate a 3 = f 3 ∧ ∀ k, 1 < k ∧ k < a → f_iterate k 3 ≠ f 3) ↔ a = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_iteration_to_repeat_l407_40756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ellipse_exists_l407_40755

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in the form y² = 2px -/
structure Parabola where
  p : ℝ

/-- Definition of the problem setup -/
structure ProblemSetup where
  parabola : Parabola
  l : Line
  lPrime : Line
  E : Point
  A : Point
  B : Point
  C : Point
  D : Point
  CPrime : Point
  DPrime : Point

/-- Conditions of the problem -/
axiom parabola_positive (setup : ProblemSetup) : setup.parabola.p > 0

axiom E_on_x_axis (setup : ProblemSetup) : 
  setup.E.x = setup.parabola.p^2 / 4 ∧ setup.E.y = 0

axiom l_intersects_parabola (setup : ProblemSetup) :
  setup.A.y^2 = 2 * setup.parabola.p * setup.A.x ∧
  setup.B.y^2 = 2 * setup.parabola.p * setup.B.x

axiom distance_condition (setup : ProblemSetup) :
  setup.A.x + setup.B.x + setup.parabola.p = 
    Real.sqrt ((setup.A.x - setup.B.x)^2 + (setup.A.y - setup.B.y)^2)

axiom l_prime_perpendicular (setup : ProblemSetup) :
  setup.l.slope * setup.lPrime.slope = -1

axiom l_prime_intersects_parabola (setup : ProblemSetup) :
  setup.C.y^2 = 2 * setup.parabola.p * setup.C.x ∧
  setup.D.y^2 = 2 * setup.parabola.p * setup.D.x

axiom C_D_prime_symmetric (setup : ProblemSetup) :
  setup.CPrime.x = 2 * setup.E.x - setup.C.x ∧
  setup.CPrime.y = -setup.C.y ∧
  setup.DPrime.x = 2 * setup.E.x - setup.D.x ∧
  setup.DPrime.y = -setup.D.y

axiom l_slope_positive (setup : ProblemSetup) : setup.l.slope > 0

/-- Theorem: No ellipse with y-axis symmetry and eccentricity √3/2 passes through A, B, C', D' -/
theorem no_ellipse_exists (setup : ProblemSetup) : 
  ¬∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a^2 = 4*b^2) ∧ 
    (setup.A.x^2 / a^2 + setup.A.y^2 / b^2 = 1) ∧
    (setup.B.x^2 / a^2 + setup.B.y^2 / b^2 = 1) ∧
    (setup.CPrime.x^2 / a^2 + setup.CPrime.y^2 / b^2 = 1) ∧
    (setup.DPrime.x^2 / a^2 + setup.DPrime.y^2 / b^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ellipse_exists_l407_40755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_commutative_field_l407_40751

variable (p : ℕ) [Fact (Nat.Prime p)]
variable (H : Polynomial (ZMod p))

/-- The set of elements of (ℤ/pℤ)[X] considered modulo H(X) -/
def F : Type := Polynomial (ZMod p) ⧸ Ideal.span {H}

/-- F is a commutative field -/
theorem F_is_commutative_field : Field (F p H) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_commutative_field_l407_40751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_response_rate_increase_l407_40746

/-- Calculates the response rate given the number of respondents and total customers --/
noncomputable def responseRate (respondents : ℕ) (totalCustomers : ℕ) : ℝ :=
  (respondents : ℝ) / (totalCustomers : ℝ) * 100

/-- Calculates the percentage increase between two values --/
noncomputable def percentageIncrease (original : ℝ) (new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem survey_response_rate_increase :
  let originalRespondents : ℕ := 7
  let originalTotal : ℕ := 60
  let redesignedRespondents : ℕ := 9
  let redesignedTotal : ℕ := 63
  let originalRate := responseRate originalRespondents originalTotal
  let redesignedRate := responseRate redesignedRespondents redesignedTotal
  let increase := percentageIncrease originalRate redesignedRate
  ∃ ε > 0, |increase - 22.44| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_response_rate_increase_l407_40746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l407_40767

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.c^2 = t.a * t.b ∧ 
  3 * t.b * Real.cos t.C = 2 * (t.a * Real.cos t.C + t.c * Real.cos t.A)

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

-- The theorem to prove
theorem triangle_area_theorem (t : Triangle) 
  (h : satisfies_conditions t) 
  (h_sum : t.a + t.b = 2 * Real.sqrt 13) : 
  triangle_area t = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l407_40767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_implies_m_leq_neg_one_l407_40774

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then
    (Real.sqrt (3 - m * x)) / m
  else if x > 1 then
    x / m - 1
  else
    0  -- arbitrary value for x ≤ 0

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem f_monotone_decreasing_implies_m_leq_neg_one :
  ∀ m : ℝ, monotone_decreasing (f m) → m ≤ -1 := by
  sorry

#check f_monotone_decreasing_implies_m_leq_neg_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_implies_m_leq_neg_one_l407_40774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_is_25_l407_40718

-- Define the given values
noncomputable def purchase_price : ℚ := 900
noncomputable def repair_costs : ℚ := 300
noncomputable def selling_price : ℚ := 1500

-- Define the total cost
noncomputable def total_cost : ℚ := purchase_price + repair_costs

-- Define the gain
noncomputable def gain : ℚ := selling_price - total_cost

-- Define the gain percent
noncomputable def gain_percent : ℚ := (gain / total_cost) * 100

-- Theorem to prove
theorem gain_percent_is_25 : gain_percent = 25 := by
  -- Unfold definitions
  unfold gain_percent
  unfold gain
  unfold total_cost
  -- Perform algebraic simplifications
  simp [purchase_price, repair_costs, selling_price]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_is_25_l407_40718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_coloring_theorem_l407_40788

/-- The number of ways to color n+1 interconnected wheel-shaped regions with m colors -/
def coloringSchemes (m n : ℤ) : ℤ :=
  m * ((-1 : ℤ)^n.toNat * (m - 2) + (m - 2)^n.toNat)

/-- Theorem: The number of ways to color n+1 interconnected wheel-shaped regions 
    with m colors, where adjacent regions have different colors -/
theorem wheel_coloring_theorem (m n : ℤ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (coloringSchemes m n) = m * ((-1 : ℤ)^n.toNat * (m - 2) + (m - 2)^n.toNat) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_coloring_theorem_l407_40788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_three_halves_less_than_reciprocal_l407_40779

theorem only_negative_three_halves_less_than_reciprocal :
  ∀ x : ℚ, x ∈ ({-3/2, -1/2, 1, 3/2, 2} : Set ℚ) →
    (x < 1 / x ↔ x = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_three_halves_less_than_reciprocal_l407_40779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l407_40783

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x)) + Real.sin x

-- State the theorem
theorem solution_set_of_inequality (a : ℝ) :
  (f (a - 2) + f (a^2 - 4) < 0) ↔ (Real.sqrt 3 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l407_40783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_mn_length_y0_range_l407_40796

-- Define the ellipse C
def ellipse (x y a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parameters
variable (a b : ℝ)

-- Define the conditions
axiom a_gt_b : a > b
axiom b_gt_zero : b > 0
axiom right_focus : ellipse 1 0 a b
axiom point_on_ellipse : ellipse 2 0 a b

-- Theorem 1: Equation of the ellipse
theorem ellipse_equation : 
  a = 2 ∧ b = Real.sqrt 3 :=
sorry

-- Theorem 2: Length of MN when slope of l is 1
theorem mn_length (M N : ℝ × ℝ) (l : ℝ → ℝ) :
  (∀ x, l x = x - 1) →
  ellipse M.1 M.2 2 (Real.sqrt 3) →
  ellipse N.1 N.2 2 (Real.sqrt 3) →
  M ≠ N →
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 24 / 7 :=
sorry

-- Theorem 3: Range of y₀
theorem y0_range (y0 : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ 
   ∃ M N : ℝ × ℝ, 
     ellipse M.1 M.2 2 (Real.sqrt 3) ∧
     ellipse N.1 N.2 2 (Real.sqrt 3) ∧
     M ≠ N ∧
     M.2 = k * (M.1 - 1) ∧
     N.2 = k * (N.1 - 1) ∧
     y0 = (M.2 + N.2) / 2 - (1 / k) * ((M.1 + N.1) / 2)) →
  -Real.sqrt 3 / 12 ≤ y0 ∧ y0 ≤ Real.sqrt 3 / 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_mn_length_y0_range_l407_40796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_false_l407_40735

-- Define the types for shapes
structure Shape : Type

-- Define properties for circles and ellipses
def is_circle (s : Shape) : Prop := sorry
def is_ellipse (s : Shape) : Prop := sorry

-- Given statement
axiom circle_is_ellipse : ∀ s : Shape, is_circle s → is_ellipse s

-- Converse statement
def converse : Prop :=
  ∀ s : Shape, is_ellipse s → is_circle s

-- Inverse statement
def inverse : Prop :=
  ∀ s : Shape, ¬is_circle s → ¬is_ellipse s

-- Theorem to prove
theorem converse_and_inverse_false : ¬converse ∧ ¬inverse := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_false_l407_40735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_portia_student_count_l407_40748

/-- The number of students in Portia's high school -/
def portia_students : ℕ := sorry

/-- The number of students in Lara's high school -/
def lara_students : ℕ := sorry

/-- The number of students in Mina's high school -/
def mina_students : ℕ := sorry

/-- Portia's high school has 4 times as many students as Lara's high school -/
axiom portia_lara_relation : portia_students = 4 * lara_students

/-- Portia's high school has 2 times as many students as Mina's high school -/
axiom portia_mina_relation : portia_students = 2 * mina_students

/-- The three high schools together have a total of 4800 students -/
axiom total_students : portia_students + lara_students + mina_students = 4800

/-- Theorem: The number of students in Portia's high school is 2740 (rounded to the nearest 10) -/
theorem portia_student_count : portia_students = 2740 := by
  sorry

#check portia_student_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_portia_student_count_l407_40748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_area_of_triangular_pyramid_l407_40720

-- Define the triangular pyramid
structure TriangularPyramid where
  -- Dihedral angles at the edges of the base
  α : Real
  β : Real
  γ : Real
  -- Areas of the corresponding lateral faces
  Sa : Real
  Sb : Real
  Sc : Real

-- Theorem statement
theorem base_area_of_triangular_pyramid (pyramid : TriangularPyramid) :
  ∃ (base_area : Real),
    base_area = pyramid.Sa * Real.cos pyramid.α + 
                pyramid.Sb * Real.cos pyramid.β + 
                pyramid.Sc * Real.cos pyramid.γ := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_area_of_triangular_pyramid_l407_40720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_is_one_point_five_l407_40745

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.sin (Real.pi * x) - 1

def domain : Set ℝ := {x | -1 < x ∧ x < 3}

theorem sum_of_zeros_is_one_point_five :
  ∃ (S : Finset ℝ), S.toSet ⊆ domain ∧ (∀ x ∈ S, f x = 0) ∧
  (∀ x ∈ domain, f x = 0 → x ∈ S) ∧
  S.sum id = 1.5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_is_one_point_five_l407_40745
