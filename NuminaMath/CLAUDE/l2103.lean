import Mathlib

namespace NUMINAMATH_CALUDE_decimal_56_to_binary_binary_to_decimal_56_decimal_56_binary_equivalence_l2103_210305

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem decimal_56_to_binary :
  to_binary 56 = [false, false, false, true, true, true] :=
by sorry

theorem binary_to_decimal_56 :
  from_binary [false, false, false, true, true, true] = 56 :=
by sorry

theorem decimal_56_binary_equivalence :
  to_binary 56 = [false, false, false, true, true, true] ∧
  from_binary [false, false, false, true, true, true] = 56 :=
by sorry

end NUMINAMATH_CALUDE_decimal_56_to_binary_binary_to_decimal_56_decimal_56_binary_equivalence_l2103_210305


namespace NUMINAMATH_CALUDE_lcm_18_35_l2103_210399

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_35_l2103_210399


namespace NUMINAMATH_CALUDE_y_value_l2103_210352

theorem y_value (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 4 / y) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2103_210352


namespace NUMINAMATH_CALUDE_johns_total_expenses_l2103_210380

/-- The number of days in John's original tour program -/
def original_days : ℕ := 20

/-- The number of additional days if John extends his trip -/
def additional_days : ℕ := 4

/-- The amount by which John must reduce his daily expenses if he extends his trip -/
def expense_reduction : ℕ := 3

/-- John's total expenses remain the same whether he stays for the original duration or extends his trip -/
axiom expense_equality (daily_expense : ℕ) :
  original_days * daily_expense = (original_days + additional_days) * (daily_expense - expense_reduction)

theorem johns_total_expenses : ∃ (total_expense : ℕ), total_expense = 360 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_expenses_l2103_210380


namespace NUMINAMATH_CALUDE_smallest_natural_with_last_four_digits_l2103_210346

theorem smallest_natural_with_last_four_digits : ∃ (N : ℕ), 
  (∀ (k : ℕ), k < N → ¬(47 * k ≡ 1969 [ZMOD 10000])) ∧ 
  (47 * N ≡ 1969 [ZMOD 10000]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_natural_with_last_four_digits_l2103_210346


namespace NUMINAMATH_CALUDE_all_negative_k_purely_imaginary_roots_l2103_210394

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_eq (z k : ℂ) : Prop := 10 * z^2 - 3 * i * z - k = 0

-- Define a purely imaginary number
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem all_negative_k_purely_imaginary_roots :
  ∀ k : ℝ, k < 0 →
    ∃ z₁ z₂ : ℂ, quadratic_eq z₁ k ∧ quadratic_eq z₂ k ∧
               is_purely_imaginary z₁ ∧ is_purely_imaginary z₂ :=
sorry

end NUMINAMATH_CALUDE_all_negative_k_purely_imaginary_roots_l2103_210394


namespace NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_91_l2103_210307

theorem greatest_consecutive_integers_sum_91 :
  (∀ n : ℕ, n > 182 → ¬ (∃ a : ℤ, (Finset.range n).sum (λ i => a + i) = 91)) ∧
  (∃ a : ℤ, (Finset.range 182).sum (λ i => a + i) = 91) :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_91_l2103_210307


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2103_210342

/-- A function satisfying the given functional equation is constant and equal to 2 -/
theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x > 0, f x > 0) → 
  (∀ x y, x > 0 → y > 0 → f x * f y = 2 * f (x + y * f x)) → 
  (∀ x > 0, f x = 2) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2103_210342


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2103_210381

/-- Simple interest rate calculation --/
theorem simple_interest_rate_calculation
  (principal : ℝ)
  (amount : ℝ)
  (time : ℝ)
  (h_principal : principal = 8000)
  (h_amount : amount = 12500)
  (h_time : time = 7)
  (h_simple_interest : amount - principal = principal * (rate / 100) * time) :
  ∃ rate : ℝ, abs (rate - 8.04) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2103_210381


namespace NUMINAMATH_CALUDE_cubic_real_root_l2103_210368

/-- The cubic equation with real coefficients c and d, having -3 - 4i as a root, has -4 as its real root -/
theorem cubic_real_root (c d : ℝ) (h : c * (-3 - 4*I)^3 + 4 * (-3 - 4*I)^2 + d * (-3 - 4*I) - 100 = 0) :
  ∃ x : ℝ, c * x^3 + 4 * x^2 + d * x - 100 = 0 ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_real_root_l2103_210368


namespace NUMINAMATH_CALUDE_equation_solution_l2103_210397

theorem equation_solution : ∃ x : ℝ, 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ∧ 
  x = 31 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2103_210397


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2103_210336

theorem fraction_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/5) = 5/8 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2103_210336


namespace NUMINAMATH_CALUDE_flagpole_height_l2103_210398

/-- Given a 3-meter pole with a 1.2-meter shadow and a flagpole with a 4.8-meter shadow,
    the height of the flagpole is 12 meters. -/
theorem flagpole_height
  (pole_height : Real)
  (pole_shadow : Real)
  (flagpole_shadow : Real)
  (h_pole_height : pole_height = 3)
  (h_pole_shadow : pole_shadow = 1.2)
  (h_flagpole_shadow : flagpole_shadow = 4.8) :
  pole_height / pole_shadow = 12 / flagpole_shadow := by
  sorry

end NUMINAMATH_CALUDE_flagpole_height_l2103_210398


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2103_210370

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x + 1| :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2103_210370


namespace NUMINAMATH_CALUDE_one_lie_one_truth_impossible_l2103_210321

-- Define the propositions
variable (J : Prop) -- Jean is lying
variable (P : Prop) -- Pierre is lying

-- Jean's statement: "When I am not lying, you are not lying either."
axiom jean_statement : ¬J → ¬P

-- Pierre's statement: "When I am lying, you are lying too."
axiom pierre_statement : P → J

-- Theorem: It's impossible for one to lie and the other to tell the truth
theorem one_lie_one_truth_impossible : ¬(J ∧ ¬P) ∧ ¬(¬J ∧ P) :=
sorry

end NUMINAMATH_CALUDE_one_lie_one_truth_impossible_l2103_210321


namespace NUMINAMATH_CALUDE_total_amount_is_1800_l2103_210356

/-- Calculates the total amount spent on courses for two semesters --/
def total_amount_spent (
  units_per_semester : ℕ
  ) (science_cost_per_unit : ℚ
  ) (humanities_cost_per_unit : ℚ
  ) (science_units_first : ℕ
  ) (humanities_units_first : ℕ
  ) (science_units_second : ℕ
  ) (humanities_units_second : ℕ
  ) (scholarship_percentage : ℚ
  ) : ℚ :=
  let first_semester_cost := 
    science_cost_per_unit * science_units_first + 
    humanities_cost_per_unit * humanities_units_first
  let second_semester_cost := 
    (1 - scholarship_percentage) * science_cost_per_unit * science_units_second + 
    humanities_cost_per_unit * humanities_units_second
  first_semester_cost + second_semester_cost

theorem total_amount_is_1800 :
  total_amount_spent 20 60 45 12 8 12 8 (1/2) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_1800_l2103_210356


namespace NUMINAMATH_CALUDE_problem_solution_l2103_210369

/-- The number of ways to distribute n distinct objects to k recipients -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinct objects to k recipients,
    where 2 specific objects must be given to the same recipient -/
def distributeWithPair (n k : ℕ) : ℕ := k * (k^(n - 2))

theorem problem_solution :
  distributeWithPair 8 10 = 10^7 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2103_210369


namespace NUMINAMATH_CALUDE_area_triangle_AEF_l2103_210345

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ :=
  sorry

/-- Divides a line segment in a given ratio -/
def divideLineSegment (A B : Point) (ratio : ℚ) : Point :=
  sorry

/-- Calculates the area of a triangle -/
def areaTriangle (t : Triangle) : ℝ :=
  sorry

theorem area_triangle_AEF (ABCD : Quadrilateral) 
  (hParallelogram : isParallelogram ABCD)
  (hArea : areaQuadrilateral ABCD = 50)
  (E : Point) (hE : E = divideLineSegment ABCD.A ABCD.B (2/5))
  (F : Point) (hF : F = divideLineSegment ABCD.C ABCD.D (3/5))
  (G : Point) (hG : G = divideLineSegment ABCD.B ABCD.C (1/2)) :
  areaTriangle ⟨ABCD.A, E, F⟩ = 12 :=
sorry

end NUMINAMATH_CALUDE_area_triangle_AEF_l2103_210345


namespace NUMINAMATH_CALUDE_max_difference_of_primes_l2103_210359

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_difference_of_primes (a b c : ℕ) : 
  (is_prime a ∧ is_prime b ∧ is_prime c ∧
   is_prime (a + b - c) ∧ is_prime (a + c - b) ∧ is_prime (b + c - a) ∧ is_prime (a + b + c) ∧
   (a + b = 800 ∨ a + c = 800 ∨ b + c = 800) ∧
   a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
   a ≠ (a + b - c) ∧ a ≠ (a + c - b) ∧ a ≠ (b + c - a) ∧ a ≠ (a + b + c) ∧
   b ≠ (a + b - c) ∧ b ≠ (a + c - b) ∧ b ≠ (b + c - a) ∧ b ≠ (a + b + c) ∧
   c ≠ (a + b - c) ∧ c ≠ (a + c - b) ∧ c ≠ (b + c - a) ∧ c ≠ (a + b + c) ∧
   (a + b - c) ≠ (a + c - b) ∧ (a + b - c) ≠ (b + c - a) ∧ (a + b - c) ≠ (a + b + c) ∧
   (a + c - b) ≠ (b + c - a) ∧ (a + c - b) ≠ (a + b + c) ∧
   (b + c - a) ≠ (a + b + c)) →
  (∃ d : ℕ, d ≤ 1594 ∧ 
   d = max (a + b + c) (max (a + c - b) (max (b + c - a) (max a (max b c)))) - 
       min (a + b - c) (min a (min b c)) ∧
   ∀ d' : ℕ, d' ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_max_difference_of_primes_l2103_210359


namespace NUMINAMATH_CALUDE_john_gathered_20_l2103_210349

/-- Given the total number of milk bottles and the number Marcus gathered,
    calculate the number of milk bottles John gathered. -/
def john_bottles (total : ℕ) (marcus : ℕ) : ℕ :=
  total - marcus

/-- Theorem stating that given 45 total milk bottles and 25 gathered by Marcus,
    John gathered 20 milk bottles. -/
theorem john_gathered_20 :
  john_bottles 45 25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_gathered_20_l2103_210349


namespace NUMINAMATH_CALUDE_min_value_of_g_l2103_210396

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define the maximum value M(a) on the interval [1,3]
noncomputable def M (a : ℝ) : ℝ := 
  ⨆ (x : ℝ) (h : x ∈ Set.Icc 1 3), f a x

-- Define the minimum value N(a) on the interval [1,3]
noncomputable def N (a : ℝ) : ℝ := 
  ⨅ (x : ℝ) (h : x ∈ Set.Icc 1 3), f a x

-- Define g(a) as M(a) - N(a)
noncomputable def g (a : ℝ) : ℝ := M a - N a

-- State the theorem
theorem min_value_of_g :
  ∀ a : ℝ, a ∈ Set.Icc (1/3) 1 → 
  ∃ min_g : ℝ, min_g = (⨅ (a : ℝ) (h : a ∈ Set.Icc (1/3) 1), g a) ∧ min_g = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_g_l2103_210396


namespace NUMINAMATH_CALUDE_max_happy_monkeys_theorem_l2103_210383

/-- Represents the number of each fruit type available --/
structure FruitCounts where
  pears : ℕ
  bananas : ℕ
  peaches : ℕ
  tangerines : ℕ

/-- Represents the criteria for a monkey to be happy --/
def happy_monkey (fruits : FruitCounts) : Prop :=
  ∃ (a b c : ℕ), a + b + c = 3 ∧ a + b + c ≤ fruits.pears + fruits.bananas + fruits.peaches + fruits.tangerines

/-- The maximum number of happy monkeys given the fruit counts --/
def max_happy_monkeys (fruits : FruitCounts) : ℕ :=
  Nat.min ((fruits.pears + fruits.bananas + fruits.peaches) / 2) fruits.tangerines

/-- Theorem stating the maximum number of happy monkeys for the given fruit counts --/
theorem max_happy_monkeys_theorem (fruits : FruitCounts) :
  fruits.pears = 20 →
  fruits.bananas = 30 →
  fruits.peaches = 40 →
  fruits.tangerines = 50 →
  max_happy_monkeys fruits = 45 :=
by
  sorry

#eval max_happy_monkeys ⟨20, 30, 40, 50⟩

end NUMINAMATH_CALUDE_max_happy_monkeys_theorem_l2103_210383


namespace NUMINAMATH_CALUDE_lake_superior_weighted_average_l2103_210319

/-- Represents the data for fish caught in a lake -/
structure LakeFishData where
  species : List String
  counts : List Nat
  weights : List Float

/-- Calculates the weighted average weight of fish in a lake -/
def weightedAverageWeight (data : LakeFishData) : Float :=
  let totalWeight := (List.zip data.counts data.weights).map (fun (c, w) => c.toFloat * w) |> List.sum
  let totalCount := data.counts.sum
  totalWeight / totalCount.toFloat

/-- The fish data for Lake Superior -/
def lakeSuperiorData : LakeFishData :=
  { species := ["Perch", "Northern Pike", "Whitefish"]
  , counts := [17, 15, 8]
  , weights := [2.5, 4.0, 3.5] }

/-- Theorem stating that the weighted average weight of fish in Lake Superior is 3.2625kg -/
theorem lake_superior_weighted_average :
  weightedAverageWeight lakeSuperiorData = 3.2625 := by
  sorry

end NUMINAMATH_CALUDE_lake_superior_weighted_average_l2103_210319


namespace NUMINAMATH_CALUDE_sum_of_triangle_ops_l2103_210348

-- Define the triangle operation
def triangle_op (a b c : ℤ) : ℤ := 2*a + 3*b - 4*c

-- State the theorem
theorem sum_of_triangle_ops : 
  triangle_op 2 3 5 + triangle_op 4 6 1 = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangle_ops_l2103_210348


namespace NUMINAMATH_CALUDE_dot_product_zero_l2103_210302

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![4, 3]

theorem dot_product_zero : 
  (Finset.sum Finset.univ (λ i => a i * (2 * a i - b i))) = 0 := by sorry

end NUMINAMATH_CALUDE_dot_product_zero_l2103_210302


namespace NUMINAMATH_CALUDE_triangle_circle_area_l2103_210309

theorem triangle_circle_area (a : ℝ) (h : a > 0) : 
  let base := a
  let angle1 := Real.pi / 4  -- 45 degrees in radians
  let angle2 := Real.pi / 12 -- 15 degrees in radians
  let height := a / (1 + Real.tan (Real.pi / 12))
  let circle_area := Real.pi * height^2
  let sector_angle := 2 * Real.pi / 3 -- 120 degrees in radians
  sector_angle / (2 * Real.pi) * circle_area = (Real.pi * a^2 * (2 - Real.sqrt 3)) / 18
  := by sorry

end NUMINAMATH_CALUDE_triangle_circle_area_l2103_210309


namespace NUMINAMATH_CALUDE_quadratic_problem_l2103_210363

-- Define the quadratic function y₁
def y₁ (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the quadratic function y₂
def y₂ (x m : ℝ) : ℝ := 2*x^2 + x + m

theorem quadratic_problem :
  ∀ (b c m : ℝ),
  (y₁ 0 b c = 4) →                        -- y₁ passes through (0,4)
  (∀ x, y₁ (1 + x) b c = y₁ (1 - x) b c) →  -- symmetry axis x = 1
  (b^2 - c = 0) →                         -- condition b² - c = 0
  (∃ x₀, b - 3 ≤ x₀ ∧ x₀ ≤ b ∧ 
    (∀ x, b - 3 ≤ x ∧ x ≤ b → y₁ x₀ b c ≤ y₁ x b c) ∧
    y₁ x₀ b c = 21) →                     -- minimum value 21 when b-3 ≤ x ≤ b
  (∀ x, 0 ≤ x ∧ x ≤ 1 → y₂ x m ≥ y₁ x b c) →  -- y₂ ≥ y₁ for 0 ≤ x ≤ 1
  ((∀ x, y₁ x b c = x^2 - 2*x + 4) ∧      -- Part 1 result
   (b = -Real.sqrt 7 ∨ b = 4) ∧           -- Part 2 result
   (m = 4))                               -- Part 3 result
  := by sorry

end NUMINAMATH_CALUDE_quadratic_problem_l2103_210363


namespace NUMINAMATH_CALUDE_car_speed_problem_l2103_210327

/-- Proves that given two cars P and R traveling 900 miles, where car P takes 2 hours less
    time than car R and has an average speed 10 miles per hour greater than car R,
    the average speed of car R is 62.25 miles per hour. -/
theorem car_speed_problem (speed_r : ℝ) : 
  (900 / speed_r - 2 = 900 / (speed_r + 10)) → speed_r = 62.25 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2103_210327


namespace NUMINAMATH_CALUDE_number_count_l2103_210387

theorem number_count (avg_all : ℝ) (avg_group1 : ℝ) (avg_group2 : ℝ) (avg_group3 : ℝ) 
  (h1 : avg_all = 2.80)
  (h2 : avg_group1 = 2.4)
  (h3 : avg_group2 = 2.3)
  (h4 : avg_group3 = 3.7) :
  ∃ (n : ℕ), n = 6 ∧ (2 * avg_group1 + 2 * avg_group2 + 2 * avg_group3) / n = avg_all := by
  sorry

end NUMINAMATH_CALUDE_number_count_l2103_210387


namespace NUMINAMATH_CALUDE_shondas_kids_l2103_210386

theorem shondas_kids (friends : ℕ) (other_adults : ℕ) (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_person : ℕ) :
  friends = 10 →
  other_adults = 7 →
  baskets = 15 →
  eggs_per_basket = 12 →
  eggs_per_person = 9 →
  ∃ (shondas_kids : ℕ), shondas_kids = 2 ∧
    (shondas_kids + friends + other_adults + 1) * eggs_per_person = baskets * eggs_per_basket :=
by sorry

end NUMINAMATH_CALUDE_shondas_kids_l2103_210386


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l2103_210334

/-- Represents the systematic sampling of students for a dental health check. -/
def systematicSampling (totalStudents : ℕ) (sampleSize : ℕ) (interval : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => start + i * interval)

/-- Theorem stating that the systematic sampling with given parameters results in the expected list of student numbers. -/
theorem systematic_sampling_result :
  systematicSampling 50 5 10 6 = [6, 16, 26, 36, 46] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l2103_210334


namespace NUMINAMATH_CALUDE_chessboard_squares_l2103_210312

/-- The number of squares of a given size in an 8x8 chessboard -/
def squares_of_size (n : Nat) : Nat :=
  (9 - n) ^ 2

/-- The total number of squares in an 8x8 chessboard -/
def total_squares : Nat :=
  (Finset.range 8).sum squares_of_size

theorem chessboard_squares :
  total_squares = 204 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_squares_l2103_210312


namespace NUMINAMATH_CALUDE_promotion_price_correct_l2103_210318

/-- The price of a medium pizza in the promotion -/
def promotion_price : ℚ := 5

/-- The regular price of a medium pizza -/
def regular_price : ℚ := 18

/-- The number of medium pizzas in the promotion -/
def promotion_quantity : ℕ := 3

/-- The total savings from the promotion -/
def total_savings : ℚ := 39

/-- Theorem stating that the promotion price satisfies the given conditions -/
theorem promotion_price_correct : 
  promotion_quantity * (regular_price - promotion_price) = total_savings :=
by sorry

end NUMINAMATH_CALUDE_promotion_price_correct_l2103_210318


namespace NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l2103_210330

def numBalls : ℕ := 15
def numBins : ℕ := 5

def probability_equal_distribution : ℚ :=
  (Nat.factorial numBalls) / ((Nat.factorial 3)^5 * numBins^numBalls)

def probability_unequal_distribution : ℚ :=
  (Nat.factorial numBalls) / 
  (Nat.factorial 5 * Nat.factorial 4 * (Nat.factorial 2)^3 * numBins^numBalls)

theorem ball_distribution_probability_ratio :
  (probability_equal_distribution / probability_unequal_distribution) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l2103_210330


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2103_210377

/-- The area of an isosceles trapezoid with bases 4x and 3x, and height x, is 7x²/2 -/
theorem isosceles_trapezoid_area (x : ℝ) : 
  let base1 : ℝ := 4 * x
  let base2 : ℝ := 3 * x
  let height : ℝ := x
  let area : ℝ := (base1 + base2) / 2 * height
  area = 7 * x^2 / 2 := by
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2103_210377


namespace NUMINAMATH_CALUDE_fourth_board_score_l2103_210300

/-- Represents a dartboard with its score -/
structure Dartboard :=
  (score : ℕ)

/-- Represents the set of four dartboards -/
def Dartboards := Fin 4 → Dartboard

theorem fourth_board_score (boards : Dartboards) 
  (h1 : boards 0 = ⟨30⟩)
  (h2 : boards 1 = ⟨38⟩)
  (h3 : boards 2 = ⟨41⟩)
  (identical : ∀ (i j : Fin 4), (boards i).score + (boards j).score = 2 * ((boards 0).score + (boards 1).score) / 2) :
  (boards 3).score = 34 := by
  sorry

end NUMINAMATH_CALUDE_fourth_board_score_l2103_210300


namespace NUMINAMATH_CALUDE_min_n_for_S_n_gt_1020_l2103_210351

/-- The sum of the first n terms in the sequence -/
def S_n (n : ℕ) : ℤ := 2 * (2^n - 1) - n

/-- The proposition that 10 is the minimum value of n such that S_n > 1020 -/
theorem min_n_for_S_n_gt_1020 :
  (∀ k < 10, S_n k ≤ 1020) ∧ S_n 10 > 1020 :=
sorry

end NUMINAMATH_CALUDE_min_n_for_S_n_gt_1020_l2103_210351


namespace NUMINAMATH_CALUDE_marys_income_percentage_l2103_210393

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.9)
  (h2 : mary = tim * 1.6) :
  mary = juan * 1.44 := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l2103_210393


namespace NUMINAMATH_CALUDE_sin_negative_three_pi_plus_alpha_l2103_210340

theorem sin_negative_three_pi_plus_alpha (α : ℝ) (h : Real.sin (π + α) = 1/3) :
  Real.sin (-3*π + α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_three_pi_plus_alpha_l2103_210340


namespace NUMINAMATH_CALUDE_wayne_shrimp_cocktail_l2103_210384

/-- Calculates the number of shrimp served per guest given the total spent, cost per pound, shrimp per pound, and number of guests. -/
def shrimp_per_guest (total_spent : ℚ) (cost_per_pound : ℚ) (shrimp_per_pound : ℕ) (num_guests : ℕ) : ℚ :=
  (total_spent / cost_per_pound * shrimp_per_pound) / num_guests

/-- Proves that Wayne plans to serve 5 shrimp per guest given the problem conditions. -/
theorem wayne_shrimp_cocktail :
  let total_spent : ℚ := 170
  let cost_per_pound : ℚ := 17
  let shrimp_per_pound : ℕ := 20
  let num_guests : ℕ := 40
  shrimp_per_guest total_spent cost_per_pound shrimp_per_pound num_guests = 5 := by
  sorry

end NUMINAMATH_CALUDE_wayne_shrimp_cocktail_l2103_210384


namespace NUMINAMATH_CALUDE_kira_morning_downloads_l2103_210301

/-- The number of songs Kira downloaded in the morning -/
def morning_songs : ℕ := sorry

/-- The number of songs Kira downloaded later in the day -/
def afternoon_songs : ℕ := 15

/-- The number of songs Kira downloaded at night -/
def night_songs : ℕ := 3

/-- The size of each song in MB -/
def song_size : ℕ := 5

/-- The total memory space occupied by all songs in MB -/
def total_memory : ℕ := 140

theorem kira_morning_downloads : 
  morning_songs = 10 ∧ 
  song_size * (morning_songs + afternoon_songs + night_songs) = total_memory := by
  sorry

end NUMINAMATH_CALUDE_kira_morning_downloads_l2103_210301


namespace NUMINAMATH_CALUDE_range_of_c_l2103_210313

/-- The statement of the theorem --/
theorem range_of_c (c : ℝ) : c > 0 ∧ c ≠ 1 →
  (((∀ x y : ℝ, x < y → c^x > c^y) ∨ (∀ x : ℝ, x + |x - 2*c| > 1)) ∧
   ¬((∀ x y : ℝ, x < y → c^x > c^y) ∧ (∀ x : ℝ, x + |x - 2*c| > 1))) ↔
  (c ∈ Set.Ioc 0 (1/2) ∪ Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l2103_210313


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2103_210372

theorem probability_at_least_one_multiple_of_four :
  let range := Finset.range 100
  let multiples_of_four := range.filter (λ n => n % 4 = 0)
  let prob_not_multiple := (range.card - multiples_of_four.card : ℚ) / range.card
  1 - prob_not_multiple ^ 2 = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2103_210372


namespace NUMINAMATH_CALUDE_initial_chairs_per_row_l2103_210367

theorem initial_chairs_per_row (rows : ℕ) (extra_chairs : ℕ) (total_chairs : ℕ) :
  rows = 7 →
  extra_chairs = 11 →
  total_chairs = 95 →
  ∃ (chairs_per_row : ℕ), chairs_per_row * rows + extra_chairs = total_chairs ∧ chairs_per_row = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_chairs_per_row_l2103_210367


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2103_210390

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

theorem intersection_complement_equality : B ∩ (U \ A) = {6, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2103_210390


namespace NUMINAMATH_CALUDE_polygon_arrangement_sides_l2103_210378

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ
  sides_positive : sides > 0

/-- Represents the arrangement of polygons as described in the problem. -/
structure PolygonArrangement where
  pentagon : RegularPolygon
  triangle : RegularPolygon
  heptagon : RegularPolygon
  nonagon : RegularPolygon
  dodecagon : RegularPolygon
  pentagon_sides : pentagon.sides = 5
  triangle_sides : triangle.sides = 3
  heptagon_sides : heptagon.sides = 7
  nonagon_sides : nonagon.sides = 9
  dodecagon_sides : dodecagon.sides = 12

/-- The number of exposed sides in the polygon arrangement. -/
def exposed_sides (arrangement : PolygonArrangement) : ℕ :=
  arrangement.pentagon.sides + arrangement.triangle.sides + arrangement.heptagon.sides +
  arrangement.nonagon.sides + arrangement.dodecagon.sides - 7

theorem polygon_arrangement_sides (arrangement : PolygonArrangement) :
  exposed_sides arrangement = 28 := by
  sorry

end NUMINAMATH_CALUDE_polygon_arrangement_sides_l2103_210378


namespace NUMINAMATH_CALUDE_min_rubles_to_win_l2103_210389

/-- Represents the state of the game with current score and rubles spent -/
structure GameState :=
  (score : ℕ)
  (rubles : ℕ)

/-- Defines the possible moves in the game -/
inductive Move
  | oneRuble : Move
  | twoRubles : Move

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.oneRuble => ⟨state.score + 1, state.rubles + 1⟩
  | Move.twoRubles => ⟨state.score * 2, state.rubles + 2⟩

/-- Checks if the game state is valid (score <= 50) -/
def isValidState (state : GameState) : Prop :=
  state.score ≤ 50

/-- Checks if the game state is winning (score = 50) -/
def isWinningState (state : GameState) : Prop :=
  state.score = 50

/-- Defines a sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to the initial game state -/
def applyMoves (moves : MoveSequence) : GameState :=
  moves.foldl applyMove ⟨0, 0⟩

/-- Theorem: The minimum number of rubles to win the game is 11 -/
theorem min_rubles_to_win :
  ∃ (moves : MoveSequence),
    (isValidState (applyMoves moves)) ∧
    (isWinningState (applyMoves moves)) ∧
    ((applyMoves moves).rubles = 11) ∧
    (∀ (other_moves : MoveSequence),
      (isValidState (applyMoves other_moves)) →
      (isWinningState (applyMoves other_moves)) →
      ((applyMoves other_moves).rubles ≥ 11)) :=
sorry

end NUMINAMATH_CALUDE_min_rubles_to_win_l2103_210389


namespace NUMINAMATH_CALUDE_credit_card_more_profitable_min_days_for_credit_card_profitability_l2103_210329

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 20000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.01

/-- Represents the annual interest rate on the debit card -/
def annual_interest_rate : ℝ := 0.06

/-- Represents the number of days in a month (assumed) -/
def days_in_month : ℕ := 30

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 360

/-- Theorem stating the minimum number of days for credit card to be more profitable -/
theorem credit_card_more_profitable (N : ℕ) : 
  (N : ℝ) * annual_interest_rate * purchase_amount / days_in_year + 
  credit_cashback_rate * purchase_amount > 
  debit_cashback_rate * purchase_amount → N ≥ 31 := by
  sorry

/-- Theorem stating that 31 days is the minimum for credit card to be more profitable -/
theorem min_days_for_credit_card_profitability : 
  ∃ (N : ℕ), N = 31 ∧ 
  (∀ (M : ℕ), M < N → 
    (M : ℝ) * annual_interest_rate * purchase_amount / days_in_year + 
    credit_cashback_rate * purchase_amount ≤ 
    debit_cashback_rate * purchase_amount) ∧
  ((N : ℝ) * annual_interest_rate * purchase_amount / days_in_year + 
   credit_cashback_rate * purchase_amount > 
   debit_cashback_rate * purchase_amount) := by
  sorry

end NUMINAMATH_CALUDE_credit_card_more_profitable_min_days_for_credit_card_profitability_l2103_210329


namespace NUMINAMATH_CALUDE_math_contest_correct_answers_l2103_210388

theorem math_contest_correct_answers 
  (total_problems : ℕ) 
  (correct_points : ℤ) 
  (incorrect_points : ℤ) 
  (total_score : ℤ) :
  total_problems = 15 →
  correct_points = 4 →
  incorrect_points = -3 →
  total_score = 25 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_problems ∧
    correct_answers * correct_points + (total_problems - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 10 :=
by sorry

end NUMINAMATH_CALUDE_math_contest_correct_answers_l2103_210388


namespace NUMINAMATH_CALUDE_negation_of_negation_one_l2103_210337

theorem negation_of_negation_one : -(-1) = 1 := by sorry

end NUMINAMATH_CALUDE_negation_of_negation_one_l2103_210337


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_9_l2103_210392

theorem smallest_common_multiple_of_8_and_9 : ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 9 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 8 ∣ m ∧ 9 ∣ m) → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_9_l2103_210392


namespace NUMINAMATH_CALUDE_right_triangle_geometric_sequence_sine_l2103_210357

theorem right_triangle_geometric_sequence_sine (a b c : Real) :
  -- The triangle is right-angled
  a^2 + b^2 = c^2 →
  -- The sides form a geometric sequence
  (b / a = c / b ∨ a / b = b / c) →
  -- The sine of the smallest angle
  min (a / c) (b / c) = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_geometric_sequence_sine_l2103_210357


namespace NUMINAMATH_CALUDE_unknown_number_problem_l2103_210308

theorem unknown_number_problem (x : ℚ) : 
  x + (2/3) * x - (1/3) * (x + (2/3) * x) = 10 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l2103_210308


namespace NUMINAMATH_CALUDE_midpoint_octahedron_volume_ratio_l2103_210331

-- Define a regular tetrahedron
structure RegularTetrahedron where
  -- Add necessary fields here

-- Define an octahedron formed by midpoints of tetrahedron edges
structure MidpointOctahedron (t : RegularTetrahedron) where
  -- Add necessary fields here

-- Define volume calculation functions
def volume_tetrahedron (t : RegularTetrahedron) : ℝ := sorry

def volume_octahedron (o : MidpointOctahedron t) : ℝ := sorry

-- Theorem statement
theorem midpoint_octahedron_volume_ratio 
  (t : RegularTetrahedron) 
  (o : MidpointOctahedron t) : 
  volume_octahedron o / volume_tetrahedron t = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_octahedron_volume_ratio_l2103_210331


namespace NUMINAMATH_CALUDE_expression_value_l2103_210338

theorem expression_value :
  let x : ℕ := 3
  5^3 - 2^x * 3 + 4^2 = 117 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2103_210338


namespace NUMINAMATH_CALUDE_sum_of_odds_l2103_210343

theorem sum_of_odds (sum_of_evens : ℕ) (n : ℕ) :
  (n = 70) →
  (sum_of_evens = n / 2 * (2 + n * 2)) →
  (sum_of_evens = 4970) →
  (n / 2 * (1 + (n * 2 - 1)) = 4900) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odds_l2103_210343


namespace NUMINAMATH_CALUDE_M_lower_bound_l2103_210347

theorem M_lower_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_M_lower_bound_l2103_210347


namespace NUMINAMATH_CALUDE_product_65_55_l2103_210317

theorem product_65_55 : 65 * 55 = 3575 := by
  sorry

end NUMINAMATH_CALUDE_product_65_55_l2103_210317


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2103_210328

theorem triangle_abc_properties (a b c A B C : ℝ) (h1 : a = b * Real.sin A + Real.sqrt 3 * a * Real.cos B)
  (h2 : b = 4) (h3 : (1/2) * a * c = 4) :
  B = Real.pi / 2 ∧ a + b + c = 4 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2103_210328


namespace NUMINAMATH_CALUDE_inequality_proof_l2103_210362

theorem inequality_proof (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2103_210362


namespace NUMINAMATH_CALUDE_apples_handed_out_is_19_l2103_210324

/-- Calculates the number of apples handed out to students in a cafeteria. -/
def apples_handed_out (initial_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (num_pies * apples_per_pie)

/-- Proves that the number of apples handed out to students is 19. -/
theorem apples_handed_out_is_19 :
  apples_handed_out 75 7 8 = 19 := by
  sorry

#eval apples_handed_out 75 7 8

end NUMINAMATH_CALUDE_apples_handed_out_is_19_l2103_210324


namespace NUMINAMATH_CALUDE_two_tangent_circles_through_point_l2103_210364

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an angle formed by three points -/
structure Angle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Checks if a point is inside an angle -/
def isInsideAngle (M : Point) (angle : Angle) : Prop :=
  sorry

/-- Checks if a circle is tangent to both sides of an angle -/
def isTangentToAngle (circle : Circle) (angle : Angle) : Prop :=
  sorry

/-- Checks if a circle passes through a point -/
def passesThrough (circle : Circle) (point : Point) : Prop :=
  sorry

/-- Main theorem -/
theorem two_tangent_circles_through_point 
  (angle : Angle) (M : Point) (h : isInsideAngle M angle) :
  ∃ (c1 c2 : Circle), 
    c1 ≠ c2 ∧ 
    isTangentToAngle c1 angle ∧ 
    isTangentToAngle c2 angle ∧ 
    passesThrough c1 M ∧ 
    passesThrough c2 M ∧
    ∀ (c : Circle), 
      isTangentToAngle c angle → 
      passesThrough c M → 
      (c = c1 ∨ c = c2) :=
  sorry

end NUMINAMATH_CALUDE_two_tangent_circles_through_point_l2103_210364


namespace NUMINAMATH_CALUDE_xy_value_l2103_210379

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 32)
  (h2 : (27:ℝ)^(x+y) / (9:ℝ)^(2*y) = 729) :
  x * y = -63/25 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l2103_210379


namespace NUMINAMATH_CALUDE_at_least_one_solution_l2103_210304

open Complex

-- Define the equation
def satisfies_equation (z : ℂ) : Prop := exp z = z^2 + 1

-- Define the constraint
def within_bound (z : ℂ) : Prop := abs z < 20

-- Theorem statement
theorem at_least_one_solution :
  ∃ z : ℂ, satisfies_equation z ∧ within_bound z :=
sorry

end NUMINAMATH_CALUDE_at_least_one_solution_l2103_210304


namespace NUMINAMATH_CALUDE_rals_current_age_l2103_210335

/-- Given that Ral is three times as old as Suri, and in 6 years Suri's age will be 25,
    prove that Ral's current age is 57 years. -/
theorem rals_current_age (suri_age suri_future_age ral_age : ℕ) 
    (h1 : ral_age = 3 * suri_age)
    (h2 : suri_future_age = suri_age + 6)
    (h3 : suri_future_age = 25) : 
  ral_age = 57 := by
  sorry

end NUMINAMATH_CALUDE_rals_current_age_l2103_210335


namespace NUMINAMATH_CALUDE_expression_value_l2103_210373

theorem expression_value (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x^5 + 3*y^2 + 7) / (x + 4) = 298 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2103_210373


namespace NUMINAMATH_CALUDE_triangle_shape_l2103_210339

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ a = c ∨ b = c) ∨ (A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l2103_210339


namespace NUMINAMATH_CALUDE_unique_natural_number_l2103_210311

theorem unique_natural_number : ∃! n : ℕ, 
  (∃ a : ℕ, n - 45 = a^2) ∧ 
  (∃ b : ℕ, n + 44 = b^2) ∧ 
  n = 1981 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_l2103_210311


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l2103_210333

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 ∧ 
  ∃ k : ℕ, x = 4 * k ∧ 
  x^3 < 8000 → 
  x ≤ 16 ∧ 
  ∃ y : ℕ, y > 0 ∧ ∃ m : ℕ, y = 4 * m ∧ y^3 < 8000 ∧ y = 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l2103_210333


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2103_210382

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - x₁ - k^2 = 0 ∧ x₂^2 - x₂ - k^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2103_210382


namespace NUMINAMATH_CALUDE_books_difference_l2103_210303

theorem books_difference (bobby_books kristi_books : ℕ) 
  (h1 : bobby_books = 142) 
  (h2 : kristi_books = 78) : 
  bobby_books - kristi_books = 64 := by
sorry

end NUMINAMATH_CALUDE_books_difference_l2103_210303


namespace NUMINAMATH_CALUDE_min_value_of_function_l2103_210391

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  ∃ y_min : ℝ, y_min = 7 ∧ ∀ y : ℝ, y = 4*x + 1/(4*x - 5) → y ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2103_210391


namespace NUMINAMATH_CALUDE_quartic_root_theorem_l2103_210355

theorem quartic_root_theorem (a b c d : ℚ) : 
  let f : ℝ → ℝ := λ x => x^4 + a*x^3 + b*x^2 + c*x + d
  (f (3 - Real.sqrt 5) = 0) → 
  (f (3 + Real.sqrt 5) = 0) → 
  (∃ r : ℤ, f r = 0) →
  (f (-3) = 0) := by
sorry

end NUMINAMATH_CALUDE_quartic_root_theorem_l2103_210355


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_l2103_210376

theorem smallest_integer_with_remainder (n : ℕ) : n = 170 ↔ 
  (n > 1 ∧ 
   n % 3 = 2 ∧ 
   n % 7 = 2 ∧ 
   n % 8 = 2 ∧ 
   ∀ m : ℕ, m > 1 → m % 3 = 2 → m % 7 = 2 → m % 8 = 2 → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_l2103_210376


namespace NUMINAMATH_CALUDE_pool_capacity_percentage_l2103_210385

/-- Calculates the current capacity percentage of a pool given its dimensions and draining parameters -/
theorem pool_capacity_percentage
  (width : ℝ) (length : ℝ) (depth : ℝ)
  (drain_rate : ℝ) (drain_time : ℝ)
  (h_width : width = 60)
  (h_length : length = 100)
  (h_depth : depth = 10)
  (h_drain_rate : drain_rate = 60)
  (h_drain_time : drain_time = 800) :
  (drain_rate * drain_time) / (width * length * depth) * 100 = 8 := by
sorry

end NUMINAMATH_CALUDE_pool_capacity_percentage_l2103_210385


namespace NUMINAMATH_CALUDE_a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l2103_210375

theorem a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ∧ ¬(∀ a b : ℝ, a^2 > b^2 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l2103_210375


namespace NUMINAMATH_CALUDE_gcd_294_84_l2103_210326

theorem gcd_294_84 : Nat.gcd 294 84 = 42 := by
  have h1 : 294 = 84 * 3 + 42 := by rfl
  have h2 : 84 = 42 * 2 + 0 := by rfl
  sorry

end NUMINAMATH_CALUDE_gcd_294_84_l2103_210326


namespace NUMINAMATH_CALUDE_probability_ratio_l2103_210320

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability of drawing four slips with the same number -/
def p : ℚ := (distinct_numbers * (slips_per_number.choose drawn_slips)) / (total_slips.choose drawn_slips)

/-- The probability of drawing two pairs of different numbers -/
def q : ℚ := (distinct_numbers.choose 2 * (slips_per_number.choose 2) * (slips_per_number.choose 2)) / (total_slips.choose drawn_slips)

theorem probability_ratio :
  q / p = 90 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l2103_210320


namespace NUMINAMATH_CALUDE_six_digit_repeat_gcd_l2103_210366

theorem six_digit_repeat_gcd : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x < 1000 → 
    n = Nat.gcd (1000 * x + x) (1000 * (x + 1) + (x + 1))) ∧ 
  n = 1001 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_repeat_gcd_l2103_210366


namespace NUMINAMATH_CALUDE_storm_average_rainfall_l2103_210361

theorem storm_average_rainfall 
  (duration : ℝ) 
  (first_30min : ℝ) 
  (next_30min : ℝ) 
  (last_hour : ℝ) :
  duration = 2 →
  first_30min = 5 →
  next_30min = first_30min / 2 →
  last_hour = 1 / 2 →
  (first_30min + next_30min + last_hour) / duration = 4 := by
sorry

end NUMINAMATH_CALUDE_storm_average_rainfall_l2103_210361


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l2103_210314

theorem range_of_a_for_quadratic_inequality :
  (∀ x : ℝ, ∀ a : ℝ, a * x^2 - a * x - 1 ≤ 0) →
  (∀ a : ℝ, (a ∈ Set.Icc (-4 : ℝ) 0) ↔ (∀ x : ℝ, a * x^2 - a * x - 1 ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l2103_210314


namespace NUMINAMATH_CALUDE_expression_evaluation_l2103_210322

theorem expression_evaluation : 
  |5 - 8 * (3 - 12)^2| - |5 - 11| + Real.sqrt 16 + Real.sin (π / 2) = 642 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2103_210322


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l2103_210315

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x = 0 ∨ 3*x - 4*y - 8 = 0

-- Theorem statement
theorem circle_and_line_properties :
  ∃ (a : ℝ),
    -- Circle C has its center on the x-axis
    (∀ x y : ℝ, circle_C x y → y = 0) ∧
    -- Circle C passes through the point (0, √3)
    circle_C 0 (Real.sqrt 3) ∧
    -- Circle C is tangent to the line x=-1
    (∀ y : ℝ, circle_C (-1) y → (x : ℝ) → x = -1 → (circle_C x y → x = -1)) ∧
    -- Line l passes through the point (0,-2)
    line_l 0 (-2) ∧
    -- The chord intercepted by circle C on line l has a length of 2√3
    (∃ x₁ y₁ x₂ y₂ : ℝ, 
      line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ 
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l2103_210315


namespace NUMINAMATH_CALUDE_equation_sum_equals_one_l2103_210350

theorem equation_sum_equals_one 
  (p q r u v w : ℝ) 
  (eq1 : 15 * u + q * v + r * w = 0)
  (eq2 : p * u + 25 * v + r * w = 0)
  (eq3 : p * u + q * v + 50 * w = 0)
  (hp : p ≠ 15)
  (hu : u ≠ 0) :
  p / (p - 15) + q / (q - 25) + r / (r - 50) = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_sum_equals_one_l2103_210350


namespace NUMINAMATH_CALUDE_largest_circle_radius_is_two_l2103_210316

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a circle with center (c, 0) and radius r -/
structure Circle where
  c : ℝ
  r : ℝ
  h_positive_r : 0 < r

/-- Returns true if the circle is entirely contained within the ellipse -/
def circleInEllipse (e : Ellipse) (c : Circle) : Prop :=
  ∀ x y : ℝ, (x - c.c)^2 + y^2 = c.r^2 → x^2 / e.a^2 + y^2 / e.b^2 ≤ 1

/-- Returns true if the circle is tangent to the ellipse -/
def circleTangentToEllipse (e : Ellipse) (c : Circle) : Prop :=
  ∃ x y : ℝ, (x - c.c)^2 + y^2 = c.r^2 ∧ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The theorem stating that the largest circle centered at a focus of the ellipse
    and entirely contained within it has radius 2 -/
theorem largest_circle_radius_is_two (e : Ellipse) (c : Circle) 
    (h_a : e.a = 7) (h_b : e.b = 5) (h_c : c.c = 2 * Real.sqrt 6) : 
    circleInEllipse e c ∧ circleTangentToEllipse e c → c.r = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_radius_is_two_l2103_210316


namespace NUMINAMATH_CALUDE_james_sticker_payment_ratio_l2103_210344

/-- Proves that the ratio of James's payment to the total cost of stickers is 1/2 -/
theorem james_sticker_payment_ratio :
  let num_packs : ℕ := 4
  let stickers_per_pack : ℕ := 30
  let cost_per_sticker : ℚ := 1/10
  let james_payment : ℚ := 6
  let total_stickers : ℕ := num_packs * stickers_per_pack
  let total_cost : ℚ := (total_stickers : ℚ) * cost_per_sticker
  james_payment / total_cost = 1/2 := by
sorry


end NUMINAMATH_CALUDE_james_sticker_payment_ratio_l2103_210344


namespace NUMINAMATH_CALUDE_card_value_decrease_is_57_16_l2103_210306

/-- The percent decrease of a baseball card's value over four years -/
def card_value_decrease : ℝ :=
  let year1_decrease := 0.30
  let year2_decrease := 0.10
  let year3_decrease := 0.20
  let year4_decrease := 0.15
  let remaining_value := (1 - year1_decrease) * (1 - year2_decrease) * (1 - year3_decrease) * (1 - year4_decrease)
  (1 - remaining_value) * 100

/-- Theorem stating that the total percent decrease of the card's value over four years is 57.16% -/
theorem card_value_decrease_is_57_16 : 
  ∃ ε > 0, |card_value_decrease - 57.16| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_card_value_decrease_is_57_16_l2103_210306


namespace NUMINAMATH_CALUDE_bus_rows_l2103_210325

/-- Given a bus with a certain capacity and seats per row, calculate the number of rows. -/
def calculate_rows (total_capacity : ℕ) (children_per_row : ℕ) : ℕ :=
  total_capacity / children_per_row

/-- Theorem: A bus with 36 children capacity and 4 children per row has 9 rows of seats. -/
theorem bus_rows :
  calculate_rows 36 4 = 9 := by
  sorry

#eval calculate_rows 36 4

end NUMINAMATH_CALUDE_bus_rows_l2103_210325


namespace NUMINAMATH_CALUDE_men_on_airplane_l2103_210354

/-- The number of men on an airplane given specific passenger information -/
theorem men_on_airplane (total : ℕ) (children : ℕ) : 
  total = 80 → children = 20 → ∃ (men women : ℕ), 
    men = women ∧ 
    men + women + children = total ∧
    men = 30 := by
  sorry

end NUMINAMATH_CALUDE_men_on_airplane_l2103_210354


namespace NUMINAMATH_CALUDE_tory_cookie_sales_l2103_210332

/-- Proves the number of cookie packs Tory sold to his neighbor -/
theorem tory_cookie_sales (total : ℕ) (grandmother : ℕ) (uncle : ℕ) (left_to_sell : ℕ) 
  (h1 : total = 50)
  (h2 : grandmother = 12)
  (h3 : uncle = 7)
  (h4 : left_to_sell = 26) :
  total - left_to_sell - (grandmother + uncle) = 5 := by
  sorry

#check tory_cookie_sales

end NUMINAMATH_CALUDE_tory_cookie_sales_l2103_210332


namespace NUMINAMATH_CALUDE_problem_solution_l2103_210371

theorem problem_solution : (2200 - 2023)^2 / 196 = 144 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2103_210371


namespace NUMINAMATH_CALUDE_f_equals_g_l2103_210310

-- Define the functions f and g
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := 5 * x^5

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l2103_210310


namespace NUMINAMATH_CALUDE_fifteen_factorial_sum_TMH_l2103_210395

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def base_ten_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 10) :: aux (m / 10)
    (aux n).reverse

theorem fifteen_factorial_sum_TMH :
  ∃ (T M H : ℕ),
    T < 10 ∧ M < 10 ∧ H < 10 ∧
    base_ten_repr (factorial 15) = [1, 3, 0, 7, M, 7, T, 2, 0, 0, H, 0, 0] ∧
    T + M + H = 2 :=
by sorry

end NUMINAMATH_CALUDE_fifteen_factorial_sum_TMH_l2103_210395


namespace NUMINAMATH_CALUDE_range_of_a_l2103_210358

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3/4| ≤ 1/4
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ 0 ≤ a ∧ a ≤ 1/2 ∧ (a ≠ 0 ∨ a ≠ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2103_210358


namespace NUMINAMATH_CALUDE_water_consumption_l2103_210353

theorem water_consumption (W : ℝ) : 
  W > 0 →
  (W - 0.2 * W - 0.35 * (W - 0.2 * W) = 130) →
  W = 250 := by
sorry

end NUMINAMATH_CALUDE_water_consumption_l2103_210353


namespace NUMINAMATH_CALUDE_p_plus_q_equals_27_over_2_l2103_210323

theorem p_plus_q_equals_27_over_2 (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 135 = 0)
  (hq : 12*q^3 - 90*q^2 - 450*q + 4950 = 0) :
  p + q = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_27_over_2_l2103_210323


namespace NUMINAMATH_CALUDE_marbles_left_l2103_210374

theorem marbles_left (total_marbles : ℕ) (num_bags : ℕ) (removed_bags : ℕ) : 
  total_marbles = 28 → 
  num_bags = 4 → 
  removed_bags = 1 → 
  total_marbles % num_bags = 0 → 
  total_marbles - (total_marbles / num_bags * removed_bags) = 21 := by
sorry

end NUMINAMATH_CALUDE_marbles_left_l2103_210374


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2103_210360

theorem absolute_value_inequality (a b c : ℝ) (h : |a + c| < b) : |a| > |c| - |b| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2103_210360


namespace NUMINAMATH_CALUDE_pencil_rows_l2103_210341

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 35) (h2 : pencils_per_row = 5) :
  total_pencils / pencils_per_row = 7 :=
by sorry

end NUMINAMATH_CALUDE_pencil_rows_l2103_210341


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2103_210365

theorem arithmetic_geometric_mean_inequality 
  (a b k : ℝ) 
  (h1 : b = k * a) 
  (h2 : k > 0) 
  (h3 : 1 ≤ k) 
  (h4 : k ≤ 3) : 
  ((a + b) / 2)^2 ≥ (Real.sqrt (a * b))^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2103_210365
