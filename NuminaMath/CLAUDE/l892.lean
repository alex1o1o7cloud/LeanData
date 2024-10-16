import Mathlib

namespace NUMINAMATH_CALUDE_fault_line_exists_l892_89246

/-- Represents a 6x6 grid covered by 18 1x2 dominoes -/
structure DominoCoveredGrid :=
  (grid : Fin 6 → Fin 6 → Bool)
  (dominoes : Fin 18 → (Fin 6 × Fin 6) × (Fin 6 × Fin 6))
  (cover_complete : ∀ i j, ∃ k, (dominoes k).1 = (i, j) ∨ (dominoes k).2 = (i, j))
  (domino_size : ∀ k, 
    ((dominoes k).1.1 = (dominoes k).2.1 ∧ (dominoes k).2.2 = (dominoes k).1.2.succ) ∨
    ((dominoes k).1.2 = (dominoes k).2.2 ∧ (dominoes k).2.1 = (dominoes k).1.1.succ))

/-- A fault line is a row or column that doesn't intersect any domino -/
def has_fault_line (g : DominoCoveredGrid) : Prop :=
  (∃ i : Fin 6, ∀ k, (g.dominoes k).1.1 ≠ i ∧ (g.dominoes k).2.1 ≠ i) ∨
  (∃ j : Fin 6, ∀ k, (g.dominoes k).1.2 ≠ j ∧ (g.dominoes k).2.2 ≠ j)

/-- Theorem: Every 6x6 grid covered by 18 1x2 dominoes has a fault line -/
theorem fault_line_exists (g : DominoCoveredGrid) : has_fault_line g :=
sorry

end NUMINAMATH_CALUDE_fault_line_exists_l892_89246


namespace NUMINAMATH_CALUDE_unique_solution_condition_l892_89201

/-- The equation (3x+5)(x-3) = -55 + kx has exactly one real solution if and only if k = 18 or k = -26 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 5)*(x - 3) = -55 + k*x) ↔ (k = 18 ∨ k = -26) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l892_89201


namespace NUMINAMATH_CALUDE_evaluate_f_l892_89244

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

theorem evaluate_f : 3 * f 5 + 4 * f (-2) = 217 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l892_89244


namespace NUMINAMATH_CALUDE_banana_arrangements_l892_89213

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- Proof that the number of distinct arrangements of "BANANA" is 60 -/
theorem banana_arrangements :
  distinctArrangements 6 [3, 2, 1] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l892_89213


namespace NUMINAMATH_CALUDE_course_size_l892_89202

theorem course_size (total : ℕ) 
  (h1 : total / 5 + total / 4 + total / 2 + 40 = total) : total = 800 := by
  sorry

end NUMINAMATH_CALUDE_course_size_l892_89202


namespace NUMINAMATH_CALUDE_bridget_bakery_profit_l892_89273

/-- Bridget's bakery problem -/
theorem bridget_bakery_profit :
  let total_loaves : ℕ := 60
  let morning_price : ℚ := 3
  let afternoon_discount : ℚ := 1
  let late_afternoon_price : ℚ := 3/2
  let production_cost : ℚ := 4/5
  let morning_sales : ℕ := total_loaves / 2
  let afternoon_sales : ℕ := ((total_loaves - morning_sales) * 3 + 2) / 4 -- Rounding up
  let late_afternoon_sales : ℕ := total_loaves - morning_sales - afternoon_sales
  let total_revenue : ℚ := 
    morning_sales * morning_price + 
    afternoon_sales * (morning_price - afternoon_discount) + 
    late_afternoon_sales * late_afternoon_price
  let total_cost : ℚ := total_loaves * production_cost
  let profit : ℚ := total_revenue - total_cost
  profit = 197/2 := by sorry

end NUMINAMATH_CALUDE_bridget_bakery_profit_l892_89273


namespace NUMINAMATH_CALUDE_digits_difference_in_base_d_l892_89283

/-- Given two digits A and B in base d > 7, such that AB + AA = 172 in base d, prove A - B = 5 in base d -/
theorem digits_difference_in_base_d (d A B : ℕ) : 
  d > 7 →
  A < d →
  B < d →
  (A * d + B) + (A * d + A) = 1 * d^2 + 7 * d + 2 →
  A - B = 5 := by
sorry

end NUMINAMATH_CALUDE_digits_difference_in_base_d_l892_89283


namespace NUMINAMATH_CALUDE_negative_quadratic_inequality_l892_89231

/-- A quadratic polynomial ax^2 + bx + c that is negative for all real x -/
structure NegativeQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  is_negative : ∀ x : ℝ, a * x^2 + b * x + c < 0

/-- Theorem: For a negative quadratic polynomial, b/a < c/a + 1 -/
theorem negative_quadratic_inequality (q : NegativeQuadratic) : q.b / q.a < q.c / q.a + 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_quadratic_inequality_l892_89231


namespace NUMINAMATH_CALUDE_broken_line_intersections_l892_89269

/-- A broken line is represented as a list of points in the plane -/
def BrokenLine := List (Real × Real)

/-- The length of a broken line -/
def length (bl : BrokenLine) : Real :=
  sorry

/-- Checks if a broken line is inside the unit square -/
def isInsideUnitSquare (bl : BrokenLine) : Prop :=
  sorry

/-- Counts the number of intersections between a broken line and a line parallel to the x-axis -/
def intersectionsWithHorizontalLine (bl : BrokenLine) (y : Real) : Nat :=
  sorry

/-- Counts the number of intersections between a broken line and a line parallel to the y-axis -/
def intersectionsWithVerticalLine (bl : BrokenLine) (x : Real) : Nat :=
  sorry

/-- The main theorem -/
theorem broken_line_intersections (bl : BrokenLine) 
  (h1 : length bl = 1000)
  (h2 : isInsideUnitSquare bl) :
  (∃ y : Real, y ∈ Set.Icc 0 1 ∧ intersectionsWithHorizontalLine bl y ≥ 500) ∨
  (∃ x : Real, x ∈ Set.Icc 0 1 ∧ intersectionsWithVerticalLine bl x ≥ 500) :=
sorry

end NUMINAMATH_CALUDE_broken_line_intersections_l892_89269


namespace NUMINAMATH_CALUDE_age_difference_l892_89299

theorem age_difference (a b c d : ℕ) 
  (eq1 : a + b = b + c + 12)
  (eq2 : b + d = c + d + 8)
  (eq3 : d = a + 5) :
  c = a - 12 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l892_89299


namespace NUMINAMATH_CALUDE_birds_on_fence_l892_89227

/-- The total number of birds on a fence given initial birds, additional birds, and additional storks -/
def total_birds (initial : ℕ) (additional : ℕ) (storks : ℕ) : ℕ :=
  initial + additional + storks

/-- Theorem stating that given 6 initial birds, 4 additional birds, and 8 additional storks, 
    the total number of birds on the fence is 18 -/
theorem birds_on_fence : total_birds 6 4 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l892_89227


namespace NUMINAMATH_CALUDE_parentheses_expression_l892_89228

theorem parentheses_expression (a b : ℝ) : (3*b + a) * (3*b - a) = 9*b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_expression_l892_89228


namespace NUMINAMATH_CALUDE_max_profit_theorem_l892_89264

-- Define the cost and profit for each pen type
def cost_A : ℝ := 5
def cost_B : ℝ := 10
def profit_A : ℝ := 2
def profit_B : ℝ := 3

-- Define the total number of pens and the constraint
def total_pens : ℕ := 300
def constraint (x : ℕ) : Prop := x ≥ 4 * (total_pens - x)

-- Define the profit function
def profit (x : ℕ) : ℝ := profit_A * x + profit_B * (total_pens - x)

theorem max_profit_theorem :
  ∃ x : ℕ, x ≤ total_pens ∧ constraint x ∧
  profit x = 660 ∧
  ∀ y : ℕ, y ≤ total_pens → constraint y → profit y ≤ profit x :=
by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l892_89264


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l892_89245

-- Problem 1
theorem problem_1 : (-1)^4 + (1 - 1/2) / 3 * (2 - 2^3) = 2 := by
  sorry

-- Problem 2
theorem problem_2 : (-3/4 - 5/9 + 7/12) / (1/36) = -26 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l892_89245


namespace NUMINAMATH_CALUDE_distance_center_to_line_l892_89271

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the circle C
def circle_C (x y θ : ℝ) : Prop :=
  x = 2 * Real.cos θ ∧ y = 2 * Real.sin θ + 2 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem distance_center_to_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y θ : ℝ, circle_C x y θ → (x - x₀)^2 + (y - y₀)^2 ≤ 4) ∧
    (|x₀ + y₀ - 6| / Real.sqrt 2 = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_distance_center_to_line_l892_89271


namespace NUMINAMATH_CALUDE_saree_sale_price_l892_89215

/-- Applies a discount to a given price -/
def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

/-- Calculates the final price after applying multiple discounts -/
def final_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount original_price

theorem saree_sale_price :
  let original_price : ℝ := 175
  let discounts : List ℝ := [0.30, 0.25, 0.15, 0.10]
  let result := final_price original_price discounts
  ∃ ε > 0, |result - 70.28| < ε :=
sorry

end NUMINAMATH_CALUDE_saree_sale_price_l892_89215


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_three_digit_multiples_of_8_l892_89247

theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  let first := 104  -- First three-digit multiple of 8
  let last := 992   -- Last three-digit multiple of 8
  let step := 8     -- Difference between consecutive multiples
  let count := (last - first) / step + 1  -- Number of terms in the sequence
  let sum := count * (first + last) / 2   -- Sum of arithmetic sequence
  sum / count = 548 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_three_digit_multiples_of_8_l892_89247


namespace NUMINAMATH_CALUDE_bus_speed_l892_89219

/-- The initial average speed of a bus given specific journey conditions -/
theorem bus_speed (D : ℝ) (h : D > 0) : ∃ v : ℝ,
  v > 0 ∧ 
  D = v * (65 / 60) ∧ 
  D = (v + 5) * 1 ∧
  v = 60 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_l892_89219


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l892_89251

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 16/49
  let a₃ : ℚ := 64/343
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → (7^n * a₁ = 4^n)) → r = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l892_89251


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l892_89284

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l892_89284


namespace NUMINAMATH_CALUDE_smallest_a_unique_b_l892_89232

def is_all_real_roots (a b : ℝ) : Prop :=
  ∀ x : ℂ, x^4 - a*x^3 + b*x^2 - a*x + 1 = 0 → x.im = 0

theorem smallest_a_unique_b :
  ∃! (a : ℝ), a > 0 ∧
    (∃ (b : ℝ), b > 0 ∧ is_all_real_roots a b) ∧
    (∀ (a' : ℝ), 0 < a' ∧ a' < a →
      ¬∃ (b : ℝ), b > 0 ∧ is_all_real_roots a' b) ∧
    (∃! (b : ℝ), b > 0 ∧ is_all_real_roots a b) ∧
    a = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_unique_b_l892_89232


namespace NUMINAMATH_CALUDE_line_perpendicular_and_tangent_l892_89276

/-- The given line -/
def given_line (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

/-- The given curve -/
def given_curve (x y : ℝ) : Prop := y = x^3 + 3*x^2 - 5

/-- The line we want to prove is correct -/
def target_line (x y : ℝ) : Prop := 3*x + y + 6 = 0

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- A line is tangent to a curve if it touches the curve at exactly one point -/
def tangent_to_curve (line : (ℝ → ℝ → Prop)) (curve : (ℝ → ℝ → Prop)) : Prop :=
  ∃! p : ℝ × ℝ, line p.1 p.2 ∧ curve p.1 p.2

theorem line_perpendicular_and_tangent :
  (∃ m₁ m₂ : ℝ, perpendicular m₁ m₂ ∧ 
    (∀ x y : ℝ, given_line x y → y = m₁*x + 1/6) ∧
    (∀ x y : ℝ, target_line x y → y = m₂*x - 2)) ∧
  tangent_to_curve target_line given_curve :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_and_tangent_l892_89276


namespace NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l892_89274

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + 42*x + 336 = -48

-- Define the roots of the equation
def root1 : ℝ := -24
def root2 : ℝ := -16

-- Theorem statement
theorem nonnegative_difference_of_roots : 
  (quadratic_equation root1 ∧ quadratic_equation root2) → 
  |root1 - root2| = 8 := by
sorry

end NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l892_89274


namespace NUMINAMATH_CALUDE_michaels_truck_rental_cost_l892_89295

/-- Calculates the total cost of renting a truck -/
def truckRentalCost (rentalFee : ℚ) (chargePerMile : ℚ) (milesDriven : ℕ) : ℚ :=
  rentalFee + chargePerMile * milesDriven

/-- Proves that the total cost for Michael's truck rental is $95.74 -/
theorem michaels_truck_rental_cost :
  truckRentalCost 20.99 0.25 299 = 95.74 := by
  sorry

#eval truckRentalCost 20.99 0.25 299

end NUMINAMATH_CALUDE_michaels_truck_rental_cost_l892_89295


namespace NUMINAMATH_CALUDE_hyperbola_intersection_ratio_difference_l892_89206

/-- The hyperbola with equation x²/2 - y²/2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/2 - y^2/2 = 1

/-- The right branch of the hyperbola -/
def right_branch (x y : ℝ) : Prop := hyperbola x y ∧ x > 0

/-- Point P lies on the right branch of the hyperbola -/
def P_on_right_branch (P : ℝ × ℝ) : Prop := right_branch P.1 P.2

/-- Point A is the intersection of PF₁ and the hyperbola -/
def A_is_intersection (P A : ℝ × ℝ) : Prop :=
  hyperbola A.1 A.2 ∧ ∃ t : ℝ, A = (t * (P.1 + 2) - 2, t * P.2)

/-- Point B is the intersection of PF₂ and the hyperbola -/
def B_is_intersection (P B : ℝ × ℝ) : Prop :=
  hyperbola B.1 B.2 ∧ ∃ t : ℝ, B = (t * (P.1 - 2) + 2, t * P.2)

/-- The main theorem -/
theorem hyperbola_intersection_ratio_difference (P A B : ℝ × ℝ) :
  P_on_right_branch P →
  A_is_intersection P A →
  B_is_intersection P B →
  ∃ (PF₁ AF₁ PF₂ BF₂ : ℝ),
    PF₁ / AF₁ - PF₂ / BF₂ = 6 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_ratio_difference_l892_89206


namespace NUMINAMATH_CALUDE_least_multiple_ending_zero_l892_89255

theorem least_multiple_ending_zero : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (n % 10 = 0) ∧
  (∀ m : ℕ, m < n → (∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∨ m % 10 ≠ 0) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_ending_zero_l892_89255


namespace NUMINAMATH_CALUDE_max_k_for_circle_intersection_l892_89275

/-- The maximum value of k for which a circle with radius 1 centered on the line y = kx - 2
    has a common point with the circle x^2 + y^2 - 8x + 15 = 0 is 4/3 -/
theorem max_k_for_circle_intersection :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 8*p.1 + 15 = 0}
  let line (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k*p.1 - 2}
  let unit_circle_on_line (k : ℝ) : Set (Set (ℝ × ℝ)) :=
    {S | ∃ c ∈ line k, S = {p | (p.1 - c.1)^2 + (p.2 - c.2)^2 = 1}}
  ∀ k > 4/3, ∀ S ∈ unit_circle_on_line k, S ∩ C = ∅ ∧
  ∃ S ∈ unit_circle_on_line (4/3), S ∩ C ≠ ∅ :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_circle_intersection_l892_89275


namespace NUMINAMATH_CALUDE_only_undergraduateGraduates2013_is_well_defined_set_l892_89233

-- Define the universe of discourse
def Universe : Type := Set (Nat → Bool)

-- Define the options
def undergraduateGraduates2013 : Universe := sorry
def highWheatProductionCities2013 : Universe := sorry
def famousMathematicians : Universe := sorry
def numbersCloseToPI : Universe := sorry

-- Define a predicate for well-defined sets
def isWellDefinedSet (S : Universe) : Prop := sorry

-- Theorem statement
theorem only_undergraduateGraduates2013_is_well_defined_set :
  isWellDefinedSet undergraduateGraduates2013 ∧
  ¬isWellDefinedSet highWheatProductionCities2013 ∧
  ¬isWellDefinedSet famousMathematicians ∧
  ¬isWellDefinedSet numbersCloseToPI :=
sorry

end NUMINAMATH_CALUDE_only_undergraduateGraduates2013_is_well_defined_set_l892_89233


namespace NUMINAMATH_CALUDE_team_combinations_l892_89210

theorem team_combinations (n m : ℕ) (h1 : n = 7) (h2 : m = 4) : 
  Nat.choose n m = 35 := by
  sorry

end NUMINAMATH_CALUDE_team_combinations_l892_89210


namespace NUMINAMATH_CALUDE_condition_equivalence_l892_89256

theorem condition_equivalence (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2 ≥ 2*a*b) ↔ (a/b + b/a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_condition_equivalence_l892_89256


namespace NUMINAMATH_CALUDE_unique_solution_l892_89278

theorem unique_solution (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.cos (π * x)^2 + 2 * Real.sin (π * y) = 1)
  (h2 : Real.sin (π * x) + Real.sin (π * y) = 0)
  (h3 : x^2 - y^2 = 12) :
  x = 4 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l892_89278


namespace NUMINAMATH_CALUDE_max_ratio_square_extension_l892_89221

/-- Given a square ABCD with side length a, the ratio MA:MB is maximized 
    when M is positioned on the extension of CD such that MC = 2a / (1 + √5) -/
theorem max_ratio_square_extension (a : ℝ) (h : a > 0) :
  let square := {A : ℝ × ℝ | A.1 ∈ [0, a] ∧ A.2 ∈ [0, a]}
  let C := (a, 0)
  let D := (a, a)
  let M (x : ℝ) := (a + x, 0)
  let ratio (x : ℝ) := ‖M x - (0, a)‖ / ‖M x - (a, a)‖
  ∃ (x_max : ℝ), x_max = 2 * a / (1 + Real.sqrt 5) ∧
    ∀ (x : ℝ), x > 0 → ratio x ≤ ratio x_max :=
by
  sorry


end NUMINAMATH_CALUDE_max_ratio_square_extension_l892_89221


namespace NUMINAMATH_CALUDE_equation_proof_l892_89290

theorem equation_proof (x : ℝ) (h : x = 12) : (17.28 / x) / (3.6 * 0.2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l892_89290


namespace NUMINAMATH_CALUDE_puppies_given_away_l892_89286

/-- Given that Sandy initially had some puppies and now has fewer,
    prove that the number of puppies given away is the difference
    between the initial and current number of puppies. -/
theorem puppies_given_away
  (initial_puppies : ℕ)
  (current_puppies : ℕ)
  (h : current_puppies ≤ initial_puppies) :
  initial_puppies - current_puppies =
  initial_puppies - current_puppies :=
by sorry

end NUMINAMATH_CALUDE_puppies_given_away_l892_89286


namespace NUMINAMATH_CALUDE_miranda_savings_l892_89248

/-- Represents an employee at the Cheesecake factory -/
structure Employee where
  name : String
  savingsFraction : ℚ

/-- Calculates the weekly salary for an employee -/
def weeklySalary (hourlyRate : ℚ) (hoursPerDay : ℕ) (daysPerWeek : ℕ) : ℚ :=
  hourlyRate * hoursPerDay * daysPerWeek

/-- Calculates the savings for an employee over a given number of weeks -/
def savings (e : Employee) (salary : ℚ) (weeks : ℕ) : ℚ :=
  e.savingsFraction * salary * weeks

/-- Theorem: Miranda saves 1/2 of her salary -/
theorem miranda_savings
  (hourlyRate : ℚ)
  (hoursPerDay daysPerWeek weeks : ℕ)
  (robby jaylen miranda : Employee)
  (h1 : hourlyRate = 10)
  (h2 : hoursPerDay = 10)
  (h3 : daysPerWeek = 5)
  (h4 : weeks = 4)
  (h5 : robby.savingsFraction = 2/5)
  (h6 : jaylen.savingsFraction = 3/5)
  (h7 : savings robby (weeklySalary hourlyRate hoursPerDay daysPerWeek) weeks +
        savings jaylen (weeklySalary hourlyRate hoursPerDay daysPerWeek) weeks +
        savings miranda (weeklySalary hourlyRate hoursPerDay daysPerWeek) weeks = 3000) :
  miranda.savingsFraction = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_miranda_savings_l892_89248


namespace NUMINAMATH_CALUDE_convex_polygon_interior_angles_l892_89242

theorem convex_polygon_interior_angles (n : ℕ) :
  n > 2 →
  (∃ x : ℕ, (n - 2) * 180 - x = 2000) →
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_interior_angles_l892_89242


namespace NUMINAMATH_CALUDE_proportion_solution_l892_89258

theorem proportion_solution (x y : ℝ) : 
  (0.60 : ℝ) / x = y / 4 ∧ x = 0.39999999999999997 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l892_89258


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l892_89235

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l892_89235


namespace NUMINAMATH_CALUDE_consecutive_product_square_extension_l892_89200

theorem consecutive_product_square_extension (n : ℕ) (h : n * (n + 1) > 12) : 
  ∃! k : ℕ, k < 100 ∧ ∃ m : ℕ, 100 * (n * (n + 1)) + k = m^2 :=
sorry

end NUMINAMATH_CALUDE_consecutive_product_square_extension_l892_89200


namespace NUMINAMATH_CALUDE_third_number_is_five_l892_89266

def hcf (a b c : ℕ) : ℕ := sorry

def lcm (a b c : ℕ) : ℕ := sorry

theorem third_number_is_five (a b c : ℕ) 
  (ha : a = 30)
  (hb : b = 75)
  (hhcf : hcf a b c = 15)
  (hlcm : lcm a b c = 750) :
  c = 5 := by sorry

end NUMINAMATH_CALUDE_third_number_is_five_l892_89266


namespace NUMINAMATH_CALUDE_correct_control_group_setup_l892_89260

/-- Represents the different media types used in the experiment -/
inductive Medium
| BeefExtractPeptone
| SelectiveUreaDecomposing

/-- Represents the different inoculation methods -/
inductive InoculationMethod
| SoilSample
| SterileWater
| NoInoculation

/-- Represents a control group setup -/
structure ControlGroup :=
  (medium : Medium)
  (inoculation : InoculationMethod)

/-- The correct control group setup for the experiment -/
def correctControlGroup : ControlGroup :=
  { medium := Medium.BeefExtractPeptone,
    inoculation := InoculationMethod.SoilSample }

/-- The experiment setup -/
structure Experiment :=
  (name : String)
  (goal : String)
  (controlGroup : ControlGroup)

/-- Theorem stating that the correct control group is the one that inoculates
    the same soil sample liquid on beef extract peptone medium -/
theorem correct_control_group_setup
  (exp : Experiment)
  (h1 : exp.name = "Separating Bacteria that Decompose Urea in Soil")
  (h2 : exp.goal = "judge whether the separation effect has been achieved")
  : exp.controlGroup = correctControlGroup := by
  sorry

end NUMINAMATH_CALUDE_correct_control_group_setup_l892_89260


namespace NUMINAMATH_CALUDE_alphabet_size_l892_89243

theorem alphabet_size :
  ∀ (dot_and_line dot_only line_only : ℕ),
    dot_and_line = 16 →
    line_only = 30 →
    dot_only = 4 →
    dot_and_line + dot_only + line_only = 50 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_size_l892_89243


namespace NUMINAMATH_CALUDE_solve_equation_l892_89217

theorem solve_equation : ∃ r : ℤ, 19 - 3 = 2 + r ∧ r = 14 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l892_89217


namespace NUMINAMATH_CALUDE_our_sequence_is_valid_common_difference_is_five_l892_89207

/-- An arithmetic sequence with given properties --/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  num_terms : ℕ
  common_diff : ℝ

/-- The properties of our specific arithmetic sequence --/
def our_sequence : ArithmeticSequence where
  first_term := 5
  last_term := 50
  sum := 275
  num_terms := 10  -- Derived from the solution, but could be proven
  common_diff := 5 -- This is what we need to prove

/-- Theorem stating that our sequence satisfies the properties of an arithmetic sequence --/
theorem our_sequence_is_valid : 
  let s := our_sequence
  s.last_term = s.first_term + (s.num_terms - 1) * s.common_diff ∧
  s.sum = (s.num_terms / 2) * (s.first_term + s.last_term) :=
by sorry

/-- Main theorem proving that the common difference must be 5 --/
theorem common_difference_is_five :
  ∀ (s : ArithmeticSequence),
    s.first_term = 5 ∧
    s.last_term = 50 ∧
    s.sum = 275 →
    s.common_diff = 5 :=
by sorry

end NUMINAMATH_CALUDE_our_sequence_is_valid_common_difference_is_five_l892_89207


namespace NUMINAMATH_CALUDE_compare_power_towers_l892_89240

def power_tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (power_tower base n)

theorem compare_power_towers (n : ℕ) :
  (n ≥ 3 → power_tower 3 (n - 1) > power_tower 2 n) ∧
  (n ≥ 2 → power_tower 3 n > power_tower 4 (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_compare_power_towers_l892_89240


namespace NUMINAMATH_CALUDE_travel_cost_theorem_l892_89223

structure City where
  name : String

structure Triangle where
  D : City
  E : City
  F : City
  DE : ℝ
  EF : ℝ
  FD : ℝ
  right_angle_at_D : DE^2 + FD^2 = EF^2

def bus_fare_per_km : ℝ := 0.20

def plane_booking_fee (departure : City) : ℝ :=
  if departure.name = "E" then 150 else 120

def plane_fare_per_km : ℝ := 0.12

def travel_cost (t : Triangle) : ℝ :=
  t.DE * bus_fare_per_km +
  t.EF * plane_fare_per_km +
  plane_booking_fee t.E

theorem travel_cost_theorem (t : Triangle) :
  t.DE = 4000 ∧ t.EF = 4500 ∧ t.FD = 5000 →
  travel_cost t = 1490 := by
  sorry

end NUMINAMATH_CALUDE_travel_cost_theorem_l892_89223


namespace NUMINAMATH_CALUDE_prism_volume_l892_89229

/-- A right rectangular prism with specific face areas and a dimension relation -/
structure RectangularPrism where
  x : ℝ
  y : ℝ
  z : ℝ
  side_area : x * y = 24
  front_area : y * z = 15
  bottom_area : x * z = 8
  dimension_relation : z = 2 * x

/-- The volume of a rectangular prism is equal to 96 cubic inches -/
theorem prism_volume (p : RectangularPrism) : p.x * p.y * p.z = 96 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l892_89229


namespace NUMINAMATH_CALUDE_min_value_theorem_l892_89294

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) (hm : m > 0) (hn : n > 0) 
  (h_fixed_point : a^(2 - 2) = 1) 
  (h_linear : m * 2 + 4 * n = 1) : 
  1 / m + 2 / n ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l892_89294


namespace NUMINAMATH_CALUDE_smaller_side_of_rearranged_rectangle_l892_89285

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the result of dividing and rearranging a rectangle -/
structure RearrangedRectangle where
  original : Rectangle
  new : Rectangle
  is_valid : original.width * original.height = new.width * new.height

/-- The theorem to be proved -/
theorem smaller_side_of_rearranged_rectangle 
  (r : RearrangedRectangle) 
  (h1 : r.original.width = 10) 
  (h2 : r.original.height = 25) :
  min r.new.width r.new.height = 10 := by
  sorry

#check smaller_side_of_rearranged_rectangle

end NUMINAMATH_CALUDE_smaller_side_of_rearranged_rectangle_l892_89285


namespace NUMINAMATH_CALUDE_area_of_region_l892_89234

/-- The region defined by the inequality |4x-24|+|3y+10| ≤ 6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 24| + |3 * p.2 + 10| ≤ 6}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of the region is 12 -/
theorem area_of_region : area Region = 12 := by sorry

end NUMINAMATH_CALUDE_area_of_region_l892_89234


namespace NUMINAMATH_CALUDE_remainder_problem_l892_89203

theorem remainder_problem (n a b c : ℕ) (hn : 0 < n) 
  (ha : n % 3 = a) (hb : n % 5 = b) (hc : n % 7 = c) 
  (heq : 4 * a + 3 * b + 2 * c = 30) : 
  n % 105 = 29 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l892_89203


namespace NUMINAMATH_CALUDE_triangle_side_values_l892_89261

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_values :
  ∀ y : ℕ+, 
    (is_valid_triangle 8 11 (y.val ^ 2)) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l892_89261


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l892_89238

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^(n^2) + n
  let r : ℕ := 3^s - 2*s
  r = 3^515 - 1030 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l892_89238


namespace NUMINAMATH_CALUDE_new_profit_percentage_l892_89254

/-- Calculate the new profit percentage given the original selling price, profit percentage, and additional profit --/
theorem new_profit_percentage
  (original_selling_price : ℝ)
  (original_profit_percentage : ℝ)
  (additional_profit : ℝ)
  (h1 : original_selling_price = 550)
  (h2 : original_profit_percentage = 0.1)
  (h3 : additional_profit = 35) :
  let original_cost_price := original_selling_price / (1 + original_profit_percentage)
  let new_cost_price := original_cost_price * 0.9
  let new_selling_price := original_selling_price + additional_profit
  let new_profit := new_selling_price - new_cost_price
  let new_profit_percentage := new_profit / new_cost_price
  new_profit_percentage = 0.3 := by
sorry


end NUMINAMATH_CALUDE_new_profit_percentage_l892_89254


namespace NUMINAMATH_CALUDE_triangle_area_product_l892_89224

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ a * x + b * y = 6) → 
  (1/2 * (6/a) * (6/b) = 6) → a * b = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_product_l892_89224


namespace NUMINAMATH_CALUDE_a_in_N_necessary_not_sufficient_for_a_in_M_l892_89214

def M : Set ℝ := {x | 0 < x ∧ x < 1}
def N : Set ℝ := {x | -2 < x ∧ x < 1}

theorem a_in_N_necessary_not_sufficient_for_a_in_M :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by sorry

end NUMINAMATH_CALUDE_a_in_N_necessary_not_sufficient_for_a_in_M_l892_89214


namespace NUMINAMATH_CALUDE_product_of_fractions_l892_89291

theorem product_of_fractions : (1 : ℚ) / 3 * 3 / 5 * 5 / 7 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l892_89291


namespace NUMINAMATH_CALUDE_christmas_discount_problem_l892_89249

/-- Represents the Christmas discount problem for an air-conditioning unit. -/
theorem christmas_discount_problem (original_price : ℝ) (price_increase : ℝ) (final_price : ℝ) 
  (h1 : original_price = 470)
  (h2 : price_increase = 0.12)
  (h3 : final_price = 442.18) :
  ∃ (x : ℝ), 
    x ≥ 0 ∧ 
    x ≤ 100 ∧ 
    abs (x - 1.11) < 0.01 ∧
    original_price * (1 - x / 100) * (1 + price_increase) = final_price :=
sorry

end NUMINAMATH_CALUDE_christmas_discount_problem_l892_89249


namespace NUMINAMATH_CALUDE_integer_solutions_of_system_l892_89208

theorem integer_solutions_of_system : 
  ∀ x y z t : ℤ, 
    (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
    ((x, y, z, t) = (1, 0, 3, 1) ∨ 
     (x, y, z, t) = (-1, 0, -3, -1) ∨ 
     (x, y, z, t) = (3, 1, 1, 0) ∨ 
     (x, y, z, t) = (-3, -1, -1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_system_l892_89208


namespace NUMINAMATH_CALUDE_x_greater_than_one_sufficient_not_necessary_l892_89267

theorem x_greater_than_one_sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → x^2 - 2*x + 1 > 0) ∧
  (∃ x : ℝ, x ≤ 1 ∧ x^2 - 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_one_sufficient_not_necessary_l892_89267


namespace NUMINAMATH_CALUDE_monday_rainfall_value_l892_89250

/-- The rainfall recorded over three days in centimeters -/
def total_rainfall : ℝ := 0.6666666666666666

/-- The rainfall recorded on Tuesday in centimeters -/
def tuesday_rainfall : ℝ := 0.4166666666666667

/-- The rainfall recorded on Wednesday in centimeters -/
def wednesday_rainfall : ℝ := 0.08333333333333333

/-- The rainfall recorded on Monday in centimeters -/
def monday_rainfall : ℝ := total_rainfall - (tuesday_rainfall + wednesday_rainfall)

theorem monday_rainfall_value : monday_rainfall = 0.16666666666666663 := by
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_value_l892_89250


namespace NUMINAMATH_CALUDE_only_vertical_angles_true_l892_89288

-- Define the propositions
def vertical_angles_equal : Prop := ∀ (α β : ℝ), α = β → α = β
def corresponding_angles_equal : Prop := ∀ (α β : ℝ), α = β
def product_one_implies_one : Prop := ∀ (a b : ℝ), a * b = 1 → a = 1 ∨ b = 1
def square_root_of_four : Prop := ∀ (x : ℝ), x^2 = 4 → x = 2

-- Theorem stating that only vertical_angles_equal is true
theorem only_vertical_angles_true : 
  vertical_angles_equal ∧ 
  ¬corresponding_angles_equal ∧ 
  ¬product_one_implies_one ∧ 
  ¬square_root_of_four :=
sorry

end NUMINAMATH_CALUDE_only_vertical_angles_true_l892_89288


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l892_89225

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 6) :
  a 5 = 18 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l892_89225


namespace NUMINAMATH_CALUDE_pencils_found_l892_89212

theorem pencils_found (initial bought final misplaced broken : ℕ) : 
  initial = 20 →
  bought = 2 →
  final = 16 →
  misplaced = 7 →
  broken = 3 →
  final = initial - misplaced - broken + bought + (final - (initial - misplaced - broken + bought)) →
  final - (initial - misplaced - broken + bought) = 4 :=
by sorry

end NUMINAMATH_CALUDE_pencils_found_l892_89212


namespace NUMINAMATH_CALUDE_smallest_sum_of_two_distinct_primes_above_70_l892_89289

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_sum_of_two_distinct_primes_above_70 :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p > 70 ∧ 
    q > 70 ∧ 
    p ≠ q ∧ 
    p + q = 144 ∧ 
    (∀ (r s : ℕ), is_prime r → is_prime s → r > 70 → s > 70 → r ≠ s → r + s ≥ 144) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_distinct_primes_above_70_l892_89289


namespace NUMINAMATH_CALUDE_final_sum_is_correct_l892_89230

/-- Represents the state of the three calculators -/
structure CalculatorState where
  calc1 : ℤ
  calc2 : ℤ
  calc3 : ℤ

/-- Applies the operations to the calculator state -/
def applyOperations (state : CalculatorState) : CalculatorState :=
  { calc1 := 2 * state.calc1,
    calc2 := state.calc2 ^ 2,
    calc3 := -state.calc3 }

/-- Iterates the operations n times -/
def iterateOperations (n : ℕ) (state : CalculatorState) : CalculatorState :=
  match n with
  | 0 => state
  | n + 1 => applyOperations (iterateOperations n state)

/-- The initial state of the calculators -/
def initialState : CalculatorState :=
  { calc1 := 2, calc2 := 0, calc3 := -2 }

/-- The main theorem to prove -/
theorem final_sum_is_correct :
  let finalState := iterateOperations 51 initialState
  finalState.calc1 + finalState.calc2 + finalState.calc3 = 2^52 + 2 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_is_correct_l892_89230


namespace NUMINAMATH_CALUDE_largest_two_digit_remainder_two_l892_89265

theorem largest_two_digit_remainder_two : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 99) ∧ 
  n % 13 = 2 ∧ 
  (∀ m : ℕ, (m ≥ 10 ∧ m ≤ 99) ∧ m % 13 = 2 → m ≤ n) ∧
  n = 93 := by
sorry

end NUMINAMATH_CALUDE_largest_two_digit_remainder_two_l892_89265


namespace NUMINAMATH_CALUDE_sqrt_plus_arcsin_equals_pi_half_l892_89241

theorem sqrt_plus_arcsin_equals_pi_half (x : ℝ) :
  Real.sqrt (x * (x + 1)) + Real.arcsin (Real.sqrt (x^2 + x + 1)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_arcsin_equals_pi_half_l892_89241


namespace NUMINAMATH_CALUDE_max_path_length_l892_89220

/-- A rectangular prism with dimensions 1, 2, and 3 -/
structure RectangularPrism where
  length : ℝ := 1
  width : ℝ := 2
  height : ℝ := 3

/-- A path in the rectangular prism -/
structure PrismPath (p : RectangularPrism) where
  -- The path starts and ends at the same corner
  start_end_same : Bool
  -- The path visits each corner exactly once
  visits_all_corners_once : Bool
  -- The path consists of straight lines between corners
  straight_lines : Bool
  -- The length of the path
  length : ℝ

/-- The theorem stating the maximum path length in the rectangular prism -/
theorem max_path_length (p : RectangularPrism) :
  ∃ (path : PrismPath p), 
    path.start_end_same ∧ 
    path.visits_all_corners_once ∧ 
    path.straight_lines ∧
    path.length = 2 * Real.sqrt 14 + 4 * Real.sqrt 13 ∧
    ∀ (other_path : PrismPath p), 
      other_path.start_end_same ∧ 
      other_path.visits_all_corners_once ∧ 
      other_path.straight_lines → 
      other_path.length ≤ path.length :=
sorry

end NUMINAMATH_CALUDE_max_path_length_l892_89220


namespace NUMINAMATH_CALUDE_star_polygon_angles_l892_89236

/-- Given a star polygon where the sum of five angles is 500°, 
    prove that the sum of the other five angles is 140°. -/
theorem star_polygon_angles (p q r s t A B C D E : ℝ) 
  (h1 : p + q + r + s + t = 500) 
  (h2 : A + B + C + D + E = x) : x = 140 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_angles_l892_89236


namespace NUMINAMATH_CALUDE_factor_tree_problem_l892_89297

/-- Factor tree problem -/
theorem factor_tree_problem (F G Y Z X : ℕ) : 
  F = 2 * 5 →
  G = 7 * 3 →
  Y = 7 * F →
  Z = 11 * G →
  X = Y * Z →
  X = 16170 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_problem_l892_89297


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l892_89282

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → perpendicularPlanes α β :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l892_89282


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l892_89280

theorem r_value_when_n_is_3 : 
  let n : ℕ := 3
  let s := 2^n + 2
  let r := 4^s - 2*s
  r = 1048556 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l892_89280


namespace NUMINAMATH_CALUDE_termites_count_workers_composition_l892_89252

/-- The number of termites in the construction project -/
def num_termites : ℕ := 861 - 239

/-- The total number of workers in the construction project -/
def total_workers : ℕ := 861

/-- The number of monkeys in the construction project -/
def num_monkeys : ℕ := 239

/-- Theorem stating that the number of termites is 622 -/
theorem termites_count : num_termites = 622 := by
  sorry

/-- Theorem stating that the total number of workers is the sum of monkeys and termites -/
theorem workers_composition : total_workers = num_monkeys + num_termites := by
  sorry

end NUMINAMATH_CALUDE_termites_count_workers_composition_l892_89252


namespace NUMINAMATH_CALUDE_complex_roots_count_l892_89259

theorem complex_roots_count : ∃! (S : Finset ℂ), 
  (∀ z ∈ S, Complex.abs z < 30 ∧ Complex.exp z = (z - 1) / (z + 1)) ∧ 
  Finset.card S = 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_count_l892_89259


namespace NUMINAMATH_CALUDE_same_color_probability_l892_89292

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (purple : Nat)
  (green : Nat)
  (orange : Nat)
  (glittery : Nat)
  (total : Nat)
  (h1 : purple + green + orange + glittery = total)
  (h2 : total = 30)

/-- The probability of rolling the same color on two identical colored dice -/
def sameProbability (d : ColoredDie) : Rat :=
  (d.purple^2 + d.green^2 + d.orange^2 + d.glittery^2) / d.total^2

/-- Two 30-sided dice with specified colored sides -/
def twoDice : ColoredDie :=
  { purple := 6
    green := 10
    orange := 12
    glittery := 2
    total := 30
    h1 := by rfl
    h2 := by rfl }

theorem same_color_probability :
  sameProbability twoDice = 71 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l892_89292


namespace NUMINAMATH_CALUDE_restaurant_bill_split_l892_89237

-- Define the meal costs and discounts
def sarah_meal : ℝ := 20
def mary_meal : ℝ := 22
def tuan_meal : ℝ := 18
def michael_meal : ℝ := 24
def linda_meal : ℝ := 16
def sarah_coupon : ℝ := 4
def student_discount : ℝ := 0.1
def sales_tax : ℝ := 0.08
def tip_percentage : ℝ := 0.15
def num_people : ℕ := 5

-- Define the theorem
theorem restaurant_bill_split (
  sarah_meal mary_meal tuan_meal michael_meal linda_meal : ℝ)
  (sarah_coupon student_discount sales_tax tip_percentage : ℝ)
  (num_people : ℕ) :
  let total_before_discount := sarah_meal + mary_meal + tuan_meal + michael_meal + linda_meal
  let sarah_discounted := sarah_meal - sarah_coupon
  let tuan_discounted := tuan_meal * (1 - student_discount)
  let linda_discounted := linda_meal * (1 - student_discount)
  let total_after_discount := sarah_discounted + mary_meal + tuan_discounted + michael_meal + linda_discounted
  let tax_amount := total_after_discount * sales_tax
  let tip_amount := total_before_discount * tip_percentage
  let final_bill := total_after_discount + tax_amount + tip_amount
  let individual_contribution := final_bill / num_people
  individual_contribution = 23 :=
by
  sorry


end NUMINAMATH_CALUDE_restaurant_bill_split_l892_89237


namespace NUMINAMATH_CALUDE_water_level_rise_l892_89222

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel -/
theorem water_level_rise
  (cube_edge : ℝ)
  (vessel_length : ℝ)
  (vessel_width : ℝ)
  (h_cube_edge : cube_edge = 10)
  (h_vessel_length : vessel_length = 20)
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_water_level_rise_l892_89222


namespace NUMINAMATH_CALUDE_steel_rod_length_l892_89239

/-- Represents the properties of a uniform steel rod -/
structure SteelRod where
  weight_per_meter : ℝ
  length : ℝ
  weight : ℝ

/-- Theorem: Given a uniform steel rod where 9 m weighs 34.2 kg, 
    the length of the rod that weighs 42.75 kg is 11.25 m -/
theorem steel_rod_length 
  (rod : SteelRod) 
  (h1 : rod.weight_per_meter = 34.2 / 9) 
  (h2 : rod.weight = 42.75) : 
  rod.length = 11.25 := by
  sorry

#check steel_rod_length

end NUMINAMATH_CALUDE_steel_rod_length_l892_89239


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l892_89270

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l892_89270


namespace NUMINAMATH_CALUDE_total_feet_count_l892_89296

/-- Given a total of 50 animals with 30 hens, prove that the total number of feet is 140. -/
theorem total_feet_count (total_animals : ℕ) (num_hens : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) : 
  total_animals = 50 → 
  num_hens = 30 → 
  hen_feet = 2 → 
  cow_feet = 4 → 
  num_hens * hen_feet + (total_animals - num_hens) * cow_feet = 140 := by
sorry

end NUMINAMATH_CALUDE_total_feet_count_l892_89296


namespace NUMINAMATH_CALUDE_fly_path_shortest_distance_l892_89287

/-- Represents a right circular cone. -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone. -/
structure SurfacePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest distance between two points on the surface of a cone. -/
def shortestSurfaceDistance (c : Cone) (p1 p2 : SurfacePoint) : ℝ :=
  sorry

theorem fly_path_shortest_distance :
  let c : Cone := { baseRadius := 600, height := 200 * Real.sqrt 7 }
  let p1 : SurfacePoint := { distanceFromVertex := 125 }
  let p2 : SurfacePoint := { distanceFromVertex := 375 * Real.sqrt 2 }
  shortestSurfaceDistance c p1 p2 = 625 := by sorry

end NUMINAMATH_CALUDE_fly_path_shortest_distance_l892_89287


namespace NUMINAMATH_CALUDE_straight_line_no_dot_l892_89298

/-- Represents the properties of an alphabet with dots and straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  dotOnly : ℕ
  allHaveEither : Bool

/-- Theorem: In the given alphabet, the number of letters with a straight line but no dot is 36 -/
theorem straight_line_no_dot (a : Alphabet) 
  (h1 : a.total = 60)
  (h2 : a.both = 20)
  (h3 : a.dotOnly = 4)
  (h4 : a.allHaveEither = true) : 
  a.total - a.both - a.dotOnly = 36 := by
  sorry

#check straight_line_no_dot

end NUMINAMATH_CALUDE_straight_line_no_dot_l892_89298


namespace NUMINAMATH_CALUDE_disjoint_sets_imply_a_values_l892_89209

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | (p.2 - 3) / (p.1 - 1) = 2 ∧ p.1 ≠ 1}
def B (a : ℝ) : Set (ℝ × ℝ) := {p | 4 * p.1 + a * p.2 = 16}

-- State the theorem
theorem disjoint_sets_imply_a_values (a : ℝ) :
  A ∩ B a = ∅ → a = -2 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_sets_imply_a_values_l892_89209


namespace NUMINAMATH_CALUDE_pizza_toppings_l892_89263

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 16)
  (h2 : pepperoni_slices = 9)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ slice ∈ Finset.range mushroom_slices)) :
  mushroom_slices - (pepperoni_slices + mushroom_slices - total_slices) = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l892_89263


namespace NUMINAMATH_CALUDE_symmetric_point_l892_89218

/-- Given a point P(2,1) and a line x - y + 1 = 0, prove that the point Q(0,3) is symmetric to P with respect to the line. -/
theorem symmetric_point (P Q : ℝ × ℝ) (line : ℝ → ℝ → ℝ) : 
  P = (2, 1) → 
  Q = (0, 3) → 
  line x y = x - y + 1 →
  (Q.1 - P.1) * (Q.2 - P.2) = -1 ∧ 
  line ((P.1 + Q.1) / 2) ((P.2 + Q.2) / 2) = 0 :=
sorry


end NUMINAMATH_CALUDE_symmetric_point_l892_89218


namespace NUMINAMATH_CALUDE_transport_tax_calculation_l892_89253

/-- Calculates the transport tax for a vehicle -/
def transportTax (horsepower : ℕ) (taxRate : ℕ) (ownershipMonths : ℕ) : ℕ :=
  horsepower * taxRate * ownershipMonths / 12

/-- Proves that the transport tax for the given conditions is 2000 rubles -/
theorem transport_tax_calculation :
  transportTax 150 20 8 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_transport_tax_calculation_l892_89253


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l892_89268

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum1 : a 1 + a 4 + a 7 = 39) 
  (h_sum2 : a 2 + a 5 + a 8 = 33) : 
  a 5 + a 8 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l892_89268


namespace NUMINAMATH_CALUDE_problem_statement_l892_89204

theorem problem_statement : 
  let p := (3 + 3 = 5)
  let q := (5 > 2)
  ¬(p ∧ q) ∧ ¬p := by sorry

end NUMINAMATH_CALUDE_problem_statement_l892_89204


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l892_89257

/-- A coloring of a complete graph with 6 vertices using 5 colors -/
def GraphColoring : Type := Fin 6 → Fin 6 → Fin 5

/-- Predicate to check if a coloring is valid -/
def is_valid_coloring (c : GraphColoring) : Prop :=
  ∀ v : Fin 6, ∀ u w : Fin 6, u ≠ v → w ≠ v → u ≠ w → c v u ≠ c v w

/-- Theorem stating that a valid 5-coloring exists for a complete graph with 6 vertices -/
theorem exists_valid_coloring : ∃ c : GraphColoring, is_valid_coloring c := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l892_89257


namespace NUMINAMATH_CALUDE_marys_weight_l892_89281

-- Define the weights as real numbers
variable (mary_weight : ℝ)
variable (john_weight : ℝ)
variable (jamison_weight : ℝ)

-- Define the conditions
axiom john_weight_relation : john_weight = mary_weight + (1/4 * mary_weight)
axiom mary_jamison_relation : mary_weight = jamison_weight - 20
axiom total_weight : mary_weight + john_weight + jamison_weight = 540

-- Theorem to prove
theorem marys_weight : mary_weight = 160 := by
  sorry

end NUMINAMATH_CALUDE_marys_weight_l892_89281


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l892_89205

theorem ceiling_sum_sqrt : ⌈Real.sqrt 20⌉ + ⌈Real.sqrt 200⌉ + ⌈Real.sqrt 2000⌉ = 65 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l892_89205


namespace NUMINAMATH_CALUDE_odd_even_functions_inequality_l892_89277

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_functions_inequality (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g)
  (h_diff : ∀ x, f x - g x = (1/2)^x) :
  g 1 < f 0 ∧ f 0 < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_odd_even_functions_inequality_l892_89277


namespace NUMINAMATH_CALUDE_andrew_stickers_distribution_l892_89226

/-- The number of stickers Andrew bought -/
def total_stickers : ℕ := 750

/-- The number of stickers Andrew kept -/
def kept_stickers : ℕ := 130

/-- The number of additional stickers Fred received compared to Daniel -/
def extra_stickers : ℕ := 120

/-- The number of stickers Daniel received -/
def daniel_stickers : ℕ := 250

theorem andrew_stickers_distribution :
  daniel_stickers + (daniel_stickers + extra_stickers) + kept_stickers = total_stickers :=
by sorry

end NUMINAMATH_CALUDE_andrew_stickers_distribution_l892_89226


namespace NUMINAMATH_CALUDE_sum_of_digits_of_M_l892_89293

/-- Represents a four-digit number -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  is_four_digit : 1000 ≤ d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ d1 * 1000 + d2 * 100 + d3 * 10 + d4 < 10000

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  n.d1 * 1000 + n.d2 * 100 + n.d3 * 10 + n.d4

/-- The product of digits of a four-digit number -/
def FourDigitNumber.digitProduct (n : FourDigitNumber) : Nat :=
  n.d1 * n.d2 * n.d3 * n.d4

/-- The sum of digits of a four-digit number -/
def FourDigitNumber.digitSum (n : FourDigitNumber) : Nat :=
  n.d1 + n.d2 + n.d3 + n.d4

/-- M is the greatest four-digit number whose digits have a product of 24 -/
def M : FourDigitNumber :=
  sorry

theorem sum_of_digits_of_M :
  M.digitProduct = 24 ∧ 
  (∀ n : FourDigitNumber, n.digitProduct = 24 → n.value ≤ M.value) →
  M.digitSum = 13 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_M_l892_89293


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_65_degrees_l892_89262

theorem supplement_of_complement_of_65_degrees : 
  let α : ℝ := 65
  let complement_of_α : ℝ := 90 - α
  let supplement_of_complement : ℝ := 180 - complement_of_α
  supplement_of_complement = 155 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_65_degrees_l892_89262


namespace NUMINAMATH_CALUDE_quadratic_function_range_l892_89216

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- State the theorem
theorem quadratic_function_range (a : ℝ) : 
  (∀ x ∈ Set.Ioc 0 1, |f a x| ≤ 1) → a ∈ Set.Icc (-2) 0 \ {0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l892_89216


namespace NUMINAMATH_CALUDE_even_periodic_increasing_function_inequality_l892_89211

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem even_periodic_increasing_function_inequality (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_period : has_period_two f)
  (h_increasing : increasing_on f (-1) 0) :
  f 3 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f 2 :=
sorry

end NUMINAMATH_CALUDE_even_periodic_increasing_function_inequality_l892_89211


namespace NUMINAMATH_CALUDE_min_value_expression_l892_89279

theorem min_value_expression (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  ∃ (x y z w : ℝ), x ≥ 0 ∧ y > 0 ∧ z > 0 ∧ w ≥ 0 ∧ y + z ≥ x + w ∧
    ∀ (a' d' b' c' : ℝ), a' ≥ 0 → d' ≥ 0 → b' > 0 → c' > 0 → b' + c' ≥ a' + d' →
      (b' / (c' + d')) + (c' / (a' + b')) ≥ (y / (z + w)) + (z / (x + y)) ∧
      (y / (z + w)) + (z / (x + y)) = Real.sqrt 2 - 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_min_value_expression_l892_89279


namespace NUMINAMATH_CALUDE_range_of_m_l892_89272

theorem range_of_m (x m : ℝ) : 
  (∀ x, (2*m - 3 ≤ x ∧ x ≤ 2*m + 1) → x ≤ -5) → 
  m ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l892_89272
