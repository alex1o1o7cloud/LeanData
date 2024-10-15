import Mathlib

namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2595_259514

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : Nat), Prime p ∧ p ∣ (2^12 + 3^14 + 7^4) ∧ ∀ (q : Nat), Prime q → q ∣ (2^12 + 3^14 + 7^4) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2595_259514


namespace NUMINAMATH_CALUDE_total_cost_is_24_l2595_259598

/-- The number of index fingers on a person's hands. -/
def num_index_fingers : ℕ := 2

/-- The cost of one gold ring in dollars. -/
def cost_per_ring : ℕ := 12

/-- The total cost of buying gold rings for all index fingers. -/
def total_cost : ℕ := num_index_fingers * cost_per_ring

/-- Theorem stating that the total cost for buying gold rings for all index fingers is $24. -/
theorem total_cost_is_24 : total_cost = 24 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_24_l2595_259598


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2595_259505

open Real

theorem tangent_line_intersection (x₀ : ℝ) : 
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ 
    (1/2 * x₀^2 - log m = x₀ * (x₀ - m)) ∧ 
    (1/(2*m) = x₀)) →
  (Real.sqrt 3 < x₀ ∧ x₀ < 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2595_259505


namespace NUMINAMATH_CALUDE_minimum_properties_l2595_259562

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem minimum_properties {x₀ : ℝ} (h₀ : x₀ > 0) 
  (h₁ : ∀ x > 0, f x ≥ f x₀) : 
  f x₀ = x₀ + 1 ∧ f x₀ < 3 := by
  sorry

end NUMINAMATH_CALUDE_minimum_properties_l2595_259562


namespace NUMINAMATH_CALUDE_candy_distribution_l2595_259547

theorem candy_distribution (total_candies : ℕ) (num_friends : ℕ) 
  (h1 : total_candies = 27) (h2 : num_friends = 5) :
  total_candies % num_friends = 
    (total_candies - (total_candies / num_friends) * num_friends) := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2595_259547


namespace NUMINAMATH_CALUDE_angle_implies_x_min_value_of_f_l2595_259563

noncomputable section

def a : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

def angle_between (u v : ℝ × ℝ) : ℝ := Real.arccos ((u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2)))

def f (x : ℝ) : ℝ := a.1 * (b x).1 + a.2 * (b x).2

theorem angle_implies_x (x : ℝ) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  angle_between a (b x) = Real.pi / 3 → x = 5 * Real.pi / 12 := by sorry

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 2) ∧ ∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.pi / 2) → f x ≤ f y ∧ f x = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_angle_implies_x_min_value_of_f_l2595_259563


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2595_259549

/-- Given that 36 men can complete a piece of work in 18 days,
    and a smaller group can complete the same work in 72 days,
    prove that the smaller group consists of 9 men. -/
theorem work_completion_theorem :
  ∀ (total_work : ℕ) (smaller_group : ℕ),
  total_work = 36 * 18 →
  total_work = smaller_group * 72 →
  smaller_group = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l2595_259549


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l2595_259597

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) → 
  k = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l2595_259597


namespace NUMINAMATH_CALUDE_b_investment_is_4000_l2595_259579

/-- Represents the investment and profit distribution in a partnership --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that under given conditions, B's investment is 4000 --/
theorem b_investment_is_4000 (p : Partnership)
  (h1 : p.a_investment = 8000)
  (h2 : p.c_investment = 2000)
  (h3 : p.total_profit = 252000)
  (h4 : p.c_profit_share = 36000)
  (h5 : p.c_investment * p.total_profit = p.c_profit_share * (p.a_investment + p.b_investment + p.c_investment)) :
  p.b_investment = 4000 := by
  sorry


end NUMINAMATH_CALUDE_b_investment_is_4000_l2595_259579


namespace NUMINAMATH_CALUDE_a_b_reciprocal_l2595_259560

theorem a_b_reciprocal (a b : ℚ) : 
  (-7/8) / (7/4 - 7/8 - 7/12) = a →
  (7/4 - 7/8 - 7/12) / (-7/8) = b →
  a = -1 / b :=
by
  sorry

end NUMINAMATH_CALUDE_a_b_reciprocal_l2595_259560


namespace NUMINAMATH_CALUDE_october_price_reduction_november_profit_impossible_l2595_259572

def initial_profit_per_box : ℝ := 50
def initial_monthly_sales : ℝ := 500
def sales_increase_per_dollar : ℝ := 20

def profit_function (x : ℝ) : ℝ :=
  (initial_profit_per_box - x) * (initial_monthly_sales + sales_increase_per_dollar * x)

theorem october_price_reduction :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  profit_function x₁ = 28000 ∧
  profit_function x₂ = 28000 ∧
  (x₁ = 10 ∨ x₁ = 15) ∧
  (x₂ = 10 ∨ x₂ = 15) :=
sorry

theorem november_profit_impossible :
  ¬ ∃ x : ℝ, profit_function x = 30000 :=
sorry

end NUMINAMATH_CALUDE_october_price_reduction_november_profit_impossible_l2595_259572


namespace NUMINAMATH_CALUDE_sequence_value_l2595_259529

theorem sequence_value (a : ℕ → ℝ) (h : ∀ n, (3 - a (n + 1)) * (6 + a n) = 18) (h0 : a 0 ≠ 3) :
  ∀ n, a n = 2^(n + 2) - n - 3 :=
sorry

end NUMINAMATH_CALUDE_sequence_value_l2595_259529


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2595_259539

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 3*x - 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2595_259539


namespace NUMINAMATH_CALUDE_race_outcomes_count_l2595_259543

-- Define the number of participants
def num_participants : ℕ := 7

-- Define a function to calculate the number of permutations
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

-- Define a function to calculate the number of combinations
def combinations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem race_outcomes_count :
  (3 * combinations (num_participants - 1) 2 * permutations 2 2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_count_l2595_259543


namespace NUMINAMATH_CALUDE_flyers_left_proof_l2595_259510

/-- The number of flyers left after Jack and Rose hand out some flyers -/
def flyers_left (initial : ℕ) (jack_handed : ℕ) (rose_handed : ℕ) : ℕ :=
  initial - (jack_handed + rose_handed)

/-- Proof that given 1,236 initial flyers, with Jack handing out 120 flyers
    and Rose handing out 320 flyers, the number of flyers left is 796 -/
theorem flyers_left_proof :
  flyers_left 1236 120 320 = 796 := by
  sorry

end NUMINAMATH_CALUDE_flyers_left_proof_l2595_259510


namespace NUMINAMATH_CALUDE_operation_laws_l2595_259573

theorem operation_laws (a b : ℝ) :
  ((25 * b) * 8 = b * (25 * 8)) ∧
  (a * 6 + 6 * 15 = 6 * (a + 15)) ∧
  (1280 / 16 / 8 = 1280 / (16 * 8)) := by
  sorry

end NUMINAMATH_CALUDE_operation_laws_l2595_259573


namespace NUMINAMATH_CALUDE_only_proposition3_is_true_l2595_259574

-- Define the propositions
def proposition1 : Prop := ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0

def proposition2 : Prop := ∀ x y : ℝ, x + y > 2 → (x > 1 ∧ y > 1)

def proposition3 : Prop := ∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3

def proposition4 : Prop := ∀ a b c : ℝ, a ≠ 0 →
  (b^2 - 4*a*c > 0 ↔ ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0)

-- The main theorem
theorem only_proposition3_is_true :
  ¬proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ ¬proposition4 :=
sorry

end NUMINAMATH_CALUDE_only_proposition3_is_true_l2595_259574


namespace NUMINAMATH_CALUDE_subtraction_of_negatives_l2595_259564

theorem subtraction_of_negatives : -14 - (-26) = 12 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negatives_l2595_259564


namespace NUMINAMATH_CALUDE_marias_number_problem_l2595_259536

theorem marias_number_problem (n : ℚ) : 
  (((n + 3) * 3 - 2) / 3 = 10) → (n = 23 / 3) := by
  sorry

end NUMINAMATH_CALUDE_marias_number_problem_l2595_259536


namespace NUMINAMATH_CALUDE_or_proposition_true_l2595_259567

theorem or_proposition_true : 
  let p : Prop := 3 > 4
  let q : Prop := 3 < 4
  p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_or_proposition_true_l2595_259567


namespace NUMINAMATH_CALUDE_quadratic_sum_l2595_259530

/-- Given a quadratic function f(x) = -3x^2 + 24x + 144, prove that when written
    in the form a(x+b)^2 + c, the sum of a, b, and c is 185. -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = -3 * x^2 + 24 * x + 144) →
  (∀ x, f x = a * (x + b)^2 + c) →
  a + b + c = 185 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2595_259530


namespace NUMINAMATH_CALUDE_earliest_82_degrees_l2595_259545

/-- The temperature function modeling the temperature in Denver, CO -/
def temperature (t : ℝ) : ℝ := -2 * t^2 + 16 * t + 40

/-- Theorem stating that the earliest non-negative time when the temperature reaches 82 degrees is 3 hours past noon -/
theorem earliest_82_degrees :
  ∀ t : ℝ, t ≥ 0 → temperature t = 82 → t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_earliest_82_degrees_l2595_259545


namespace NUMINAMATH_CALUDE_first_term_is_one_l2595_259589

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1
  sum_formula : ∀ n : ℕ, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)

/-- Theorem: For a geometric sequence with specific sum values, the first term is 1 -/
theorem first_term_is_one (seq : GeometricSequence) (m : ℕ) 
    (h1 : seq.S (m - 2) = 1)
    (h2 : seq.S m = 3)
    (h3 : seq.S (m + 2) = 5) :
  seq.a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_one_l2595_259589


namespace NUMINAMATH_CALUDE_altitudes_intersect_at_one_point_l2595_259527

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of an acute triangle --/
def isAcute (t : Triangle) : Prop := sorry

/-- Definition of an altitude of a triangle --/
def altitude (t : Triangle) (v : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The three altitudes of an acute triangle intersect at one point --/
theorem altitudes_intersect_at_one_point (t : Triangle) (h : isAcute t) :
  ∃! p : ℝ × ℝ, p ∈ altitude t t.A ∩ altitude t t.B ∩ altitude t t.C :=
sorry

end NUMINAMATH_CALUDE_altitudes_intersect_at_one_point_l2595_259527


namespace NUMINAMATH_CALUDE_product_of_diff_of_squares_l2595_259588

-- Define the property of being a difference of two squares
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, n = x^2 - y^2 ∧ x > y

-- Theorem statement
theorem product_of_diff_of_squares (a b c d : ℕ) 
  (ha : is_diff_of_squares a)
  (hb : is_diff_of_squares b)
  (hc : is_diff_of_squares c)
  (hd : is_diff_of_squares d)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (pos_d : d > 0) :
  is_diff_of_squares (a * b * c * d) :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_diff_of_squares_l2595_259588


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l2595_259531

theorem quadratic_symmetry_axis 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h1 : a * (0 + 4)^2 + b * (0 + 4) + c = 0)
  (h2 : a * (0 - 1)^2 + b * (0 - 1) + c = 0) :
  -b / (2 * a) = 1.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l2595_259531


namespace NUMINAMATH_CALUDE_batsman_final_average_l2595_259519

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  lastInningRuns : Nat
  averageIncrease : Nat

/-- Calculates the average runs of a batsman after their last inning -/
def finalAverage (b : Batsman) : Nat :=
  (b.totalRuns + b.lastInningRuns) / (b.innings + 1)

/-- Theorem stating the final average of the batsman -/
theorem batsman_final_average (b : Batsman) 
  (h1 : b.innings = 10)
  (h2 : b.lastInningRuns = 80)
  (h3 : b.averageIncrease = 5)
  (h4 : finalAverage b = (b.totalRuns / b.innings) + b.averageIncrease) :
  finalAverage b = 30 := by
  sorry

#check batsman_final_average

end NUMINAMATH_CALUDE_batsman_final_average_l2595_259519


namespace NUMINAMATH_CALUDE_mrs_hilt_marbles_l2595_259568

/-- Calculates the final number of marbles Mrs. Hilt has -/
def final_marbles (initial lost given_away found : ℕ) : ℕ :=
  initial - lost - given_away + found

/-- Theorem stating that Mrs. Hilt's final number of marbles is correct -/
theorem mrs_hilt_marbles :
  final_marbles 38 15 6 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_marbles_l2595_259568


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2595_259558

theorem unique_solution_for_equation : ∃! (m n : ℕ), m^m + (m*n)^n = 1984 ∧ m = 4 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2595_259558


namespace NUMINAMATH_CALUDE_annas_remaining_money_l2595_259565

/-- Given Anna's initial amount and her purchases, calculate the amount left --/
theorem annas_remaining_money (initial_amount : ℚ) 
  (gum_price choc_price cane_price : ℚ)
  (gum_quantity choc_quantity cane_quantity : ℕ) : 
  initial_amount = 10 →
  gum_price = 1 →
  choc_price = 1 →
  cane_price = 1/2 →
  gum_quantity = 3 →
  choc_quantity = 5 →
  cane_quantity = 2 →
  initial_amount - (gum_price * gum_quantity + choc_price * choc_quantity + cane_price * cane_quantity) = 1 := by
  sorry

end NUMINAMATH_CALUDE_annas_remaining_money_l2595_259565


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2595_259559

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_properties (a₁ : ℝ) (d : ℝ) (h : d > 0) :
  (∀ n m : ℕ, n < m → arithmetic_sequence a₁ d n < arithmetic_sequence a₁ d m) ∧
  (∀ n m : ℕ, n < m → arithmetic_sequence a₁ d n + 3 * n * d < arithmetic_sequence a₁ d m + 3 * m * d) ∧
  (∃ a₁ d : ℝ, d > 0 ∧ ∃ n m : ℕ, n < m ∧ n * (arithmetic_sequence a₁ d n) ≥ m * (arithmetic_sequence a₁ d m)) ∧
  (∃ a₁ d : ℝ, d > 0 ∧ ∃ n m : ℕ, n < m ∧ arithmetic_sequence a₁ d n / n ≤ arithmetic_sequence a₁ d m / m) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2595_259559


namespace NUMINAMATH_CALUDE_sum_difference_even_odd_l2595_259532

/-- Sum of the first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Sum of the first n positive odd integers -/
def sumFirstOddIntegers (n : ℕ) : ℕ := n * n

/-- The positive difference between the sum of the first 25 positive even integers
    and the sum of the first 20 positive odd integers is 250 -/
theorem sum_difference_even_odd : 
  (sumFirstEvenIntegers 25 : ℤ) - (sumFirstOddIntegers 20 : ℤ) = 250 := by sorry

end NUMINAMATH_CALUDE_sum_difference_even_odd_l2595_259532


namespace NUMINAMATH_CALUDE_line_y_intercept_l2595_259586

/-- A line with slope 6 and x-intercept (8, 0) has y-intercept (0, -48) -/
theorem line_y_intercept (f : ℝ → ℝ) (h_slope : ∀ x y, f y - f x = 6 * (y - x)) 
  (h_x_intercept : f 8 = 0) : f 0 = -48 := by
  sorry

end NUMINAMATH_CALUDE_line_y_intercept_l2595_259586


namespace NUMINAMATH_CALUDE_souvenir_discount_equation_l2595_259548

/-- Proves that for a souvenir with given original and final prices after two consecutive discounts, 
    the equation relating these prices and the discount percentage is correct. -/
theorem souvenir_discount_equation (a : ℝ) : 
  let original_price : ℝ := 168
  let final_price : ℝ := 128
  original_price * (1 - a / 100)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_souvenir_discount_equation_l2595_259548


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2595_259525

theorem simplify_trig_expression (α : Real) (h : α ∈ Set.Ioo (π / 2) π) :
  (Real.sqrt (1 - 2 * Real.sin α * Real.cos α)) / (Real.sin α + Real.sqrt (1 - Real.sin α ^ 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2595_259525


namespace NUMINAMATH_CALUDE_zeros_in_square_of_nines_zeros_count_in_2019_nines_squared_l2595_259522

theorem zeros_in_square_of_nines (n : ℕ) : 
  (10^n - 1)^2 % 10^(n-1) = 1 ∧ (10^n - 1)^2 % 10^n ≠ 0 := by
  sorry

theorem zeros_count_in_2019_nines_squared : 
  ∃ k : ℕ, (10^2019 - 1)^2 = k * 10^2018 + 1 ∧ k % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_nines_zeros_count_in_2019_nines_squared_l2595_259522


namespace NUMINAMATH_CALUDE_box_dimensions_l2595_259578

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  h₁ : 0 < a₁
  h₂ : 0 < a₂
  h₃ : 0 < a₃
  h₄ : a₁ ≤ a₂
  h₅ : a₂ ≤ a₃

/-- The volume of a cube -/
def cubeVolume : ℝ := 2

/-- The proportion of the box filled by cubes -/
def fillProportion : ℝ := 0.4

/-- Checks if the given dimensions satisfy the cube-filling condition -/
def satisfiesCubeFilling (d : BoxDimensions) : Prop :=
  ∃ (n : ℕ), n * cubeVolume = fillProportion * (d.a₁ * d.a₂ * d.a₃)

/-- The theorem stating the possible box dimensions -/
theorem box_dimensions : 
  ∀ d : BoxDimensions, satisfiesCubeFilling d → 
    (d.a₁ = 2 ∧ d.a₂ = 3 ∧ d.a₃ = 5) ∨ (d.a₁ = 2 ∧ d.a₂ = 5 ∧ d.a₃ = 6) := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_l2595_259578


namespace NUMINAMATH_CALUDE_hotel_room_charges_l2595_259512

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * (1 - 0.25))
  (h2 : P = G * (1 - 0.10)) :
  R = G * 1.20 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l2595_259512


namespace NUMINAMATH_CALUDE_positive_expression_l2595_259501

theorem positive_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l2595_259501


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_proof_l2595_259521

/-- Checks if a natural number is a palindrome when represented in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- The smallest 5-digit palindrome in base 2 that is also a 3-digit palindrome in base 5 -/
def smallest_dual_palindrome : ℕ := 27

theorem smallest_dual_palindrome_proof :
  (is_palindrome smallest_dual_palindrome 2) ∧
  (is_palindrome smallest_dual_palindrome 5) ∧
  (to_base smallest_dual_palindrome 2).length = 5 ∧
  (to_base smallest_dual_palindrome 5).length = 3 ∧
  (∀ m : ℕ, m < smallest_dual_palindrome →
    ¬(is_palindrome m 2 ∧ is_palindrome m 5 ∧
      (to_base m 2).length = 5 ∧ (to_base m 5).length = 3)) :=
by
  sorry

#eval smallest_dual_palindrome

end NUMINAMATH_CALUDE_smallest_dual_palindrome_proof_l2595_259521


namespace NUMINAMATH_CALUDE_inscribed_polyhedron_volume_relation_l2595_259507

/-- A polyhedron with an inscribed sphere -/
structure InscribedPolyhedron where
  -- The volume of the polyhedron
  volume : ℝ
  -- The radius of the inscribed sphere
  sphereRadius : ℝ
  -- The surface area of the polyhedron
  surfaceArea : ℝ
  -- Assumption that the sphere is inscribed in the polyhedron
  isInscribed : Prop
  -- Assumption that the polyhedron can be decomposed into pyramids
  canDecompose : Prop
  -- Assumption that each pyramid has a face as base and sphere center as apex
  pyramidProperty : Prop

/-- Theorem stating the volume relation for a polyhedron with an inscribed sphere -/
theorem inscribed_polyhedron_volume_relation (p : InscribedPolyhedron) :
  p.volume = (1 / 3) * p.sphereRadius * p.surfaceArea := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polyhedron_volume_relation_l2595_259507


namespace NUMINAMATH_CALUDE_bus_stop_problem_l2595_259520

/-- The number of children who got on the bus at a stop -/
def children_got_on (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

theorem bus_stop_problem :
  let initial_children : ℕ := 64
  let final_children : ℕ := 78
  children_got_on initial_children final_children = 14 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l2595_259520


namespace NUMINAMATH_CALUDE_perimeter_difference_l2595_259504

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : Real
  width : Real

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : Real :=
  2 * (r.length + r.width)

/-- Represents a cutting configuration for the plywood -/
structure CuttingConfig where
  piece : Rectangle
  num_pieces : Nat

/-- The original plywood dimensions -/
def plywood : Rectangle :=
  { length := 10, width := 6 }

/-- The number of pieces to cut the plywood into -/
def num_pieces : Nat := 6

/-- Checks if a cutting configuration is valid for the given plywood -/
def is_valid_config (config : CuttingConfig) : Prop :=
  config.num_pieces = num_pieces ∧
  config.piece.length * config.piece.width * config.num_pieces = plywood.length * plywood.width

/-- Theorem stating the difference between max and min perimeter -/
theorem perimeter_difference :
  ∃ (max_config min_config : CuttingConfig),
    is_valid_config max_config ∧
    is_valid_config min_config ∧
    (∀ c : CuttingConfig, is_valid_config c →
      perimeter c.piece ≤ perimeter max_config.piece ∧
      perimeter c.piece ≥ perimeter min_config.piece) ∧
    perimeter max_config.piece - perimeter min_config.piece = 11.34 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l2595_259504


namespace NUMINAMATH_CALUDE_unique_solution_iff_prime_l2595_259523

theorem unique_solution_iff_prime (n : ℕ+) :
  (∃! a : ℕ, a < n.val.factorial ∧ (n.val.factorial ∣ a^n.val + 1)) ↔ Nat.Prime n.val := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_prime_l2595_259523


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l2595_259571

theorem right_triangle_leg_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a = 8 → c = 17 →
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l2595_259571


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2595_259518

theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 72 →
  crossing_time = 20 →
  ∃ bridge_length : ℝ,
    bridge_length = 150 ∧
    train_length + bridge_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2595_259518


namespace NUMINAMATH_CALUDE_problem_statement_l2595_259509

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c > 0) :
  (a^2 - b*c > b^2 - a*c) ∧ (a^3 > b^2) ∧ (a + 1/a > b + 1/b) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2595_259509


namespace NUMINAMATH_CALUDE_power_comparison_l2595_259552

theorem power_comparison : 2^1000 < 5^500 ∧ 5^500 < 3^750 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l2595_259552


namespace NUMINAMATH_CALUDE_color_film_fraction_l2595_259526

theorem color_film_fraction (x y : ℝ) (h : x ≠ 0) : 
  let total_bw := 30 * x
  let total_color := 6 * y
  let selected_bw := (y / x) * (1 / 100) * total_bw
  let selected_color := total_color
  (selected_color / (selected_bw + selected_color)) = 20 / 21 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l2595_259526


namespace NUMINAMATH_CALUDE_merry_lambs_l2595_259566

theorem merry_lambs (merry_lambs : ℕ) (brother_lambs : ℕ) : 
  brother_lambs = merry_lambs + 3 →
  merry_lambs + brother_lambs = 23 →
  merry_lambs = 10 := by
sorry

end NUMINAMATH_CALUDE_merry_lambs_l2595_259566


namespace NUMINAMATH_CALUDE_jellybean_problem_l2595_259587

theorem jellybean_problem (initial_count : ℕ) : 
  (((initial_count : ℚ) * (3/4)^3).floor = 27) → initial_count = 64 := by
sorry

end NUMINAMATH_CALUDE_jellybean_problem_l2595_259587


namespace NUMINAMATH_CALUDE_valid_n_characterization_l2595_259528

def is_valid_n (n : ℕ) : Prop :=
  ∃ (k : ℤ), (37.5^n + 26.5^n : ℝ) = k ∧ k > 0

theorem valid_n_characterization :
  ∀ n : ℕ, is_valid_n n ↔ n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_valid_n_characterization_l2595_259528


namespace NUMINAMATH_CALUDE_equation_solution_l2595_259576

theorem equation_solution : 
  let f (x : ℝ) := (x^2 - 3*x + 2) * (x^2 + 3*x - 2)
  let g (x : ℝ) := x^2 * (x + 3) * (x - 3)
  f (1/3) = g (1/3) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2595_259576


namespace NUMINAMATH_CALUDE_range_of_a_l2595_259542

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}
def B : Set ℝ := {x | x^2 - 6*x + 5 < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (A a ∩ B = ∅) → (a ≥ 6 ∨ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2595_259542


namespace NUMINAMATH_CALUDE_smallest_norm_u_l2595_259595

theorem smallest_norm_u (u : ℝ × ℝ) (h : ‖u + (4, 2)‖ = 10) :
  ∃ (v : ℝ × ℝ), ‖v‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ (w : ℝ × ℝ), ‖w + (4, 2)‖ = 10 → ‖v‖ ≤ ‖w‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_u_l2595_259595


namespace NUMINAMATH_CALUDE_grants_age_fraction_l2595_259555

theorem grants_age_fraction (grant_current_age hospital_current_age : ℕ) 
  (h1 : grant_current_age = 25) (h2 : hospital_current_age = 40) :
  (grant_current_age + 5 : ℚ) / (hospital_current_age + 5 : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_grants_age_fraction_l2595_259555


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2595_259596

theorem min_value_quadratic (x y : ℝ) : x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ -41 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2595_259596


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l2595_259554

/-- Represents a normal distribution -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ

/-- The value that is exactly 2 standard deviations less than the mean -/
def two_std_dev_below (d : NormalDistribution) : ℝ :=
  d.mean - 2 * d.std_dev

/-- Theorem: If the mean is 14.0 and the value 2 standard deviations below the mean is 11,
    then the standard deviation is 1.5 -/
theorem normal_distribution_std_dev
  (d : NormalDistribution)
  (h_mean : d.mean = 14.0)
  (h_two_below : two_std_dev_below d = 11) :
  d.std_dev = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l2595_259554


namespace NUMINAMATH_CALUDE_mayoral_election_vote_ratio_l2595_259575

theorem mayoral_election_vote_ratio :
  let votes_Z : ℕ := 25000
  let votes_X : ℕ := 22500
  let votes_Y : ℕ := (3 * votes_Z) / 5
  (votes_X - votes_Y) * 2 = votes_Y := by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_vote_ratio_l2595_259575


namespace NUMINAMATH_CALUDE_point_coordinates_l2595_259553

theorem point_coordinates (x y : ℝ) : 
  (|y| = (1/2) * |x|) → -- distance from x-axis is half the distance from y-axis
  (|x| = 10) →          -- point is 10 units from y-axis
  (y = 5 ∨ y = -5) :=   -- y-coordinate is either 5 or -5
by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2595_259553


namespace NUMINAMATH_CALUDE_project_duration_is_four_days_l2595_259580

/-- Calculates the number of days taken to finish a project given the number of naps, hours per nap, and working hours. -/
def projectDuration (numNaps : ℕ) (hoursPerNap : ℕ) (workingHours : ℕ) : ℚ :=
  let totalHours := numNaps * hoursPerNap + workingHours
  totalHours / 24

/-- Theorem stating that under the given conditions, the project duration is 4 days. -/
theorem project_duration_is_four_days :
  projectDuration 6 7 54 = 4 := by
  sorry

#eval projectDuration 6 7 54

end NUMINAMATH_CALUDE_project_duration_is_four_days_l2595_259580


namespace NUMINAMATH_CALUDE_odd_square_minus_one_l2595_259550

theorem odd_square_minus_one (n : ℕ) : (2*n + 1)^2 - 1 = 4*n*(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_l2595_259550


namespace NUMINAMATH_CALUDE_solution_of_equation_l2595_259569

theorem solution_of_equation (x : ℝ) : (5 / (x + 1) - 4 / x = 0) ↔ (x = 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2595_259569


namespace NUMINAMATH_CALUDE_julia_monday_kids_l2595_259591

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 10

/-- The additional number of kids Julia played with on Monday compared to Tuesday -/
def additional_monday_kids : ℕ := 8

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := tuesday_kids + additional_monday_kids

theorem julia_monday_kids : monday_kids = 18 := by
  sorry

end NUMINAMATH_CALUDE_julia_monday_kids_l2595_259591


namespace NUMINAMATH_CALUDE_remainder_problem_l2595_259508

theorem remainder_problem : (7^6 + 8^7 + 9^8) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2595_259508


namespace NUMINAMATH_CALUDE_seconds_in_minutes_l2595_259516

theorem seconds_in_minutes (minutes : ℚ) (seconds_per_minute : ℕ) :
  minutes = 11 / 3 →
  seconds_per_minute = 60 →
  (minutes * seconds_per_minute : ℚ) = 220 := by
sorry

end NUMINAMATH_CALUDE_seconds_in_minutes_l2595_259516


namespace NUMINAMATH_CALUDE_slab_rate_per_square_meter_l2595_259561

theorem slab_rate_per_square_meter 
  (length : ℝ) 
  (width : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : total_cost = 14437.5) : 
  total_cost / (length * width) = 700 :=
by sorry

end NUMINAMATH_CALUDE_slab_rate_per_square_meter_l2595_259561


namespace NUMINAMATH_CALUDE_element_in_set_l2595_259541

def M : Set (ℕ × ℕ) := {(1, 2)}

theorem element_in_set : (1, 2) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l2595_259541


namespace NUMINAMATH_CALUDE_hiker_first_pack_weight_l2595_259506

/-- Calculates the weight of the first pack for a hiker given specific conditions --/
theorem hiker_first_pack_weight
  (supplies_per_mile : Real)
  (hiking_rate : Real)
  (hours_per_day : Real)
  (days : Real)
  (first_resupply_ratio : Real)
  (second_resupply_ratio : Real)
  (h1 : supplies_per_mile = 0.6)
  (h2 : hiking_rate = 2.5)
  (h3 : hours_per_day = 9)
  (h4 : days = 7)
  (h5 : first_resupply_ratio = 0.3)
  (h6 : second_resupply_ratio = 0.2) :
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := supplies_per_mile * total_distance
  let first_resupply := first_resupply_ratio * total_supplies
  let second_resupply := second_resupply_ratio * total_supplies
  let first_pack_weight := total_supplies - (first_resupply + second_resupply)
  first_pack_weight = 47.25 := by sorry

end NUMINAMATH_CALUDE_hiker_first_pack_weight_l2595_259506


namespace NUMINAMATH_CALUDE_triangle_third_side_l2595_259581

theorem triangle_third_side (a b c : ℕ) : 
  (a - b = 7 ∨ b - a = 7) →  -- difference between two sides is 7
  (a + b + c) % 2 = 1 →      -- perimeter is odd
  c = 8                      -- third side is 8
:= by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l2595_259581


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2595_259556

/-- A regular decagon -/
structure RegularDecagon where
  -- Add necessary properties here

/-- The number of diagonals in a regular decagon -/
def num_diagonals (d : RegularDecagon) : ℕ := 35

/-- The number of ways to choose 3 diagonals from a regular decagon -/
def num_diagonal_trios (d : RegularDecagon) : ℕ := 6545

/-- The number of ways to choose 5 points from a regular decagon such that no three points are consecutive -/
def num_valid_point_sets (d : RegularDecagon) : ℕ := 252

/-- The probability that three randomly chosen diagonals of a regular decagon intersect inside the decagon -/
def intersection_probability (d : RegularDecagon) : ℚ :=
  num_valid_point_sets d / num_diagonal_trios d

theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  intersection_probability d = 252 / 6545 := by
  sorry


end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2595_259556


namespace NUMINAMATH_CALUDE_vision_data_median_l2595_259577

/-- Represents the vision data for a class of students -/
def VisionData : List (Float × Nat) := [
  (4.0, 1), (4.1, 2), (4.2, 6), (4.3, 3), (4.4, 3),
  (4.5, 4), (4.6, 1), (4.7, 2), (4.8, 5), (4.9, 7), (5.0, 5)
]

/-- The total number of students -/
def totalStudents : Nat := 39

/-- Calculates the median of the vision data -/
def median (data : List (Float × Nat)) (total : Nat) : Float :=
  sorry

/-- Theorem stating that the median of the given vision data is 4.6 -/
theorem vision_data_median : median VisionData totalStudents = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_vision_data_median_l2595_259577


namespace NUMINAMATH_CALUDE_function_extrema_condition_l2595_259584

def f (a x : ℝ) : ℝ := x^3 + (a+1)*x^2 + (a+1)*x + a

theorem function_extrema_condition (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) ↔ (a < -1 ∨ a > 2) :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_condition_l2595_259584


namespace NUMINAMATH_CALUDE_equation_solution_l2595_259513

theorem equation_solution :
  ∃ x : ℝ, x - 15 = 30 ∧ x = 45 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2595_259513


namespace NUMINAMATH_CALUDE_chorus_selection_probability_equal_l2595_259540

/-- Represents a two-stage sampling process in a high school chorus selection -/
structure ChorusSelection where
  total_students : ℕ
  eliminated_students : ℕ
  selected_students : ℕ

/-- The probability of a student being selected for the chorus -/
def selection_probability (cs : ChorusSelection) : ℚ :=
  cs.selected_students / cs.total_students

/-- Theorem stating that the selection probability is equal for all students -/
theorem chorus_selection_probability_equal
  (cs : ChorusSelection)
  (h1 : cs.total_students = 1815)
  (h2 : cs.eliminated_students = 15)
  (h3 : cs.selected_students = 30)
  (h4 : cs.total_students = cs.eliminated_students + (cs.total_students - cs.eliminated_students))
  (h5 : cs.selected_students ≤ cs.total_students - cs.eliminated_students) :
  selection_probability cs = 30 / 1815 := by
  sorry

#check chorus_selection_probability_equal

end NUMINAMATH_CALUDE_chorus_selection_probability_equal_l2595_259540


namespace NUMINAMATH_CALUDE_cos_five_pi_thirds_equals_one_half_l2595_259533

theorem cos_five_pi_thirds_equals_one_half : 
  Real.cos (5 * π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_five_pi_thirds_equals_one_half_l2595_259533


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2595_259515

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = b * x + 4) ↔ b ≠ 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2595_259515


namespace NUMINAMATH_CALUDE_train_length_calculation_l2595_259585

/-- Calculates the length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60 →
  person_speed = 6 →
  passing_time = 13.090909090909092 →
  ∃ (train_length : ℝ), abs (train_length - 240) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2595_259585


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2595_259592

theorem polygon_sides_count (n : ℕ) (h : n > 2) : 
  (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2595_259592


namespace NUMINAMATH_CALUDE_circle_sequence_periodic_l2595_259537

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the sequence of circles
def circleSequence (ABC : Triangle) : Fin 7 → Circle
  | 1 => sorry  -- S₁ inscribed in angle A of ABC
  | 2 => sorry  -- S₂ inscribed in triangle formed by tangent from C to S₁
  | 3 => sorry  -- S₃ inscribed in triangle formed by tangent from A to S₂
  | 4 => sorry  -- S₄
  | 5 => sorry  -- S₅
  | 6 => sorry  -- S₆
  | 7 => sorry  -- S₇

-- Theorem statement
theorem circle_sequence_periodic (ABC : Triangle) :
  circleSequence ABC 7 = circleSequence ABC 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_sequence_periodic_l2595_259537


namespace NUMINAMATH_CALUDE_no_x_satisfies_conditions_l2595_259557

theorem no_x_satisfies_conditions : ¬ ∃ x : ℝ, 
  400 ≤ x ∧ x ≤ 600 ∧ 
  Int.floor (Real.sqrt x) = 23 ∧ 
  Int.floor (Real.sqrt (100 * x)) = 480 := by
sorry

end NUMINAMATH_CALUDE_no_x_satisfies_conditions_l2595_259557


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_area_l2595_259582

theorem right_triangle_perimeter_area (a c : ℝ) :
  a > 0 ∧ c > 0 ∧  -- Positive sides
  c > a ∧  -- Hypotenuse is longest side
  Real.sqrt (c - 5) + 2 * Real.sqrt (10 - 2*c) = a - 4 →  -- Given equation
  ∃ b : ℝ, 
    b > 0 ∧  -- Positive side
    a^2 + b^2 = c^2 ∧  -- Pythagorean theorem
    a + b + c = 12 ∧  -- Perimeter
    (1/2) * a * b = 6  -- Area
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_area_l2595_259582


namespace NUMINAMATH_CALUDE_blue_pill_cost_proof_l2595_259544

def total_cost : ℚ := 430
def days : ℕ := 10
def blue_red_diff : ℚ := 3

def blue_pill_cost : ℚ := 23

theorem blue_pill_cost_proof :
  (blue_pill_cost * days + (blue_pill_cost - blue_red_diff) * days = total_cost) ∧
  (blue_pill_cost > 0) ∧
  (blue_pill_cost - blue_red_diff > 0) :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_proof_l2595_259544


namespace NUMINAMATH_CALUDE_cricket_game_target_runs_l2595_259590

/-- Calculates the target number of runs in a cricket game -/
def targetRuns (totalOvers runRateFirst10 runRateRemaining : ℝ) : ℝ :=
  10 * runRateFirst10 + (totalOvers - 10) * runRateRemaining

/-- Theorem stating the target number of runs in the given cricket game -/
theorem cricket_game_target_runs :
  targetRuns 50 6.2 5.5 = 282 := by
  sorry

end NUMINAMATH_CALUDE_cricket_game_target_runs_l2595_259590


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2595_259593

theorem complex_fraction_sum (z : ℂ) (a b : ℝ) : 
  z = (1 + Complex.I) / (1 - Complex.I) → 
  z = Complex.mk a b → 
  a + b = 1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2595_259593


namespace NUMINAMATH_CALUDE_always_positive_l2595_259534

theorem always_positive (x : ℝ) : 3 * x^2 - 6 * x + 3.5 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l2595_259534


namespace NUMINAMATH_CALUDE_abs_x_eq_x_necessary_not_sufficient_l2595_259502

theorem abs_x_eq_x_necessary_not_sufficient :
  (∀ x : ℝ, |x| = x → x^2 ≥ -x) ∧
  ¬(∀ x : ℝ, x^2 ≥ -x → |x| = x) := by
  sorry

end NUMINAMATH_CALUDE_abs_x_eq_x_necessary_not_sufficient_l2595_259502


namespace NUMINAMATH_CALUDE_two_distinct_roots_l2595_259594

/-- The custom operation ⊗ for real numbers -/
def otimes (a b : ℝ) : ℝ := b^2 - a*b

/-- Theorem stating that the equation (k-3) ⊗ x = k-1 has two distinct real roots for any real k -/
theorem two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ otimes (k-3) x₁ = k-1 ∧ otimes (k-3) x₂ = k-1 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l2595_259594


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l2595_259570

/-- The number of sides in a regular decagon -/
def decagon_sides : ℕ := 10

/-- The number of diagonals from one vertex of a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: The number of diagonals from one vertex of a regular decagon is 7 -/
theorem decagon_diagonals_from_vertex : 
  diagonals_from_vertex decagon_sides = 7 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l2595_259570


namespace NUMINAMATH_CALUDE_multiple_of_a_share_l2595_259511

def total_sum : ℚ := 427
def c_share : ℚ := 84

theorem multiple_of_a_share : ∃ (a b : ℚ) (x : ℚ), 
  a + b + c_share = total_sum ∧ 
  x * a = 4 * b ∧ 
  x * a = 7 * c_share ∧
  x = 3 := by sorry

end NUMINAMATH_CALUDE_multiple_of_a_share_l2595_259511


namespace NUMINAMATH_CALUDE_range_of_a_l2595_259524

/-- Given real numbers a, b, c satisfying a system of equations, 
    prove that the range of values for a is [1, 9]. -/
theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  a ∈ Set.Icc 1 9 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l2595_259524


namespace NUMINAMATH_CALUDE_f_inv_is_inverse_unique_solution_is_three_l2595_259551

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem stating that f_inv is indeed the inverse of f
theorem f_inv_is_inverse : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x := by sorry

-- Main theorem
theorem unique_solution_is_three :
  ∃! x : ℝ, f x = f_inv x := by sorry

end NUMINAMATH_CALUDE_f_inv_is_inverse_unique_solution_is_three_l2595_259551


namespace NUMINAMATH_CALUDE_factor_expression_l2595_259503

theorem factor_expression (x : ℝ) : 5*x*(x-2) + 9*(x-2) - 4*(x-2) = 5*(x-2)*(x+1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2595_259503


namespace NUMINAMATH_CALUDE_tan_Y_in_right_triangle_l2595_259538

theorem tan_Y_in_right_triangle (Y : Real) (opposite hypotenuse : ℝ) 
  (h1 : opposite = 8)
  (h2 : hypotenuse = 17)
  (h3 : 0 < opposite)
  (h4 : opposite < hypotenuse) :
  Real.tan Y = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_Y_in_right_triangle_l2595_259538


namespace NUMINAMATH_CALUDE_smallest_value_w3_plus_z3_l2595_259500

theorem smallest_value_w3_plus_z3 (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 18) :
  Complex.abs (w^3 + z^3) ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_w3_plus_z3_l2595_259500


namespace NUMINAMATH_CALUDE_unique_solution_system_l2595_259546

/-- Given positive real numbers a, b, c satisfying √a + √b + √c = √π/2,
    prove that there exists a unique triple (x, y, z) of real numbers
    satisfying the system of equations:
    √(y-a) + √(z-a) = 1
    √(z-b) + √(x-b) = 1
    √(x-c) + √(y-c) = 1 -/
theorem unique_solution_system (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = Real.sqrt (π / 2)) :
  ∃! x y z : ℝ,
    Real.sqrt (y - a) + Real.sqrt (z - a) = 1 ∧
    Real.sqrt (z - b) + Real.sqrt (x - b) = 1 ∧
    Real.sqrt (x - c) + Real.sqrt (y - c) = 1 :=
by sorry


end NUMINAMATH_CALUDE_unique_solution_system_l2595_259546


namespace NUMINAMATH_CALUDE_smallest_z_value_l2595_259599

/-- Given four positive integers w, x, y, and z such that:
    1. w³, x³, y³, and z³ are distinct, consecutive positive perfect cubes
    2. There's a gap of 1 between w, x, and y
    3. There's a gap of 3 between y and z
    4. w³ + x³ + y³ = z³
    Then the smallest possible value of z is 9. -/
theorem smallest_z_value (w x y z : ℕ+) 
  (h1 : w.val + 1 = x.val)
  (h2 : x.val + 1 = y.val)
  (h3 : y.val + 3 = z.val)
  (h4 : w.val^3 + x.val^3 + y.val^3 = z.val^3)
  (h5 : w.val^3 < x.val^3 ∧ x.val^3 < y.val^3 ∧ y.val^3 < z.val^3) :
  z.val ≥ 9 := by
  sorry

#check smallest_z_value

end NUMINAMATH_CALUDE_smallest_z_value_l2595_259599


namespace NUMINAMATH_CALUDE_limit_polynomial_at_2_l2595_259517

theorem limit_polynomial_at_2 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |3*x^2 - 2*x + 7 - 15| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_polynomial_at_2_l2595_259517


namespace NUMINAMATH_CALUDE_driver_net_rate_of_pay_l2595_259583

/-- Calculate the net rate of pay for a driver given specific conditions --/
theorem driver_net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 60)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gas_price = 2.50)
  : ∃ (net_rate : ℝ), net_rate = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_driver_net_rate_of_pay_l2595_259583


namespace NUMINAMATH_CALUDE_new_rectangle_area_l2595_259535

/-- Given a rectangle with sides 3 and 4, prove that a new rectangle
    formed with one side equal to the diagonal of the original rectangle
    and the other side equal to the sum of the original sides has an area of 35. -/
theorem new_rectangle_area (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  let d := Real.sqrt (a^2 + b^2)
  let new_side_sum := a + b
  d * new_side_sum = 35 := by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l2595_259535
