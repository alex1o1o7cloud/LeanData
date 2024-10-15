import Mathlib

namespace NUMINAMATH_CALUDE_addition_subtraction_problem_l3052_305254

theorem addition_subtraction_problem : (5.75 + 3.09) - 1.86 = 6.98 := by
  sorry

end NUMINAMATH_CALUDE_addition_subtraction_problem_l3052_305254


namespace NUMINAMATH_CALUDE_smallest_base_for_inequality_l3052_305265

theorem smallest_base_for_inequality (k : ℕ) (h : k = 6) : 
  ∀ b : ℕ, b > 0 → b ≤ 4 ↔ b^16 ≤ 64^k :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_inequality_l3052_305265


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l3052_305205

/-- Given a triangle ABC with sides a, b, c, where a = 1 and 2cos(C) + c = 2b,
    the perimeter p satisfies 2 < p ≤ 3 -/
theorem triangle_perimeter_range (b c : ℝ) (C : ℝ) : 
  let a : ℝ := 1
  let p := a + b + c
  2 * Real.cos C + c = 2 * b →
  2 < p ∧ p ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l3052_305205


namespace NUMINAMATH_CALUDE_triangle_third_vertex_l3052_305287

/-- Given a triangle with vertices at (0, 0), (10, 5), and (x, 0) where x < 0,
    if the area of the triangle is 50 square units, then x = -20. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * |x * 5| = 50 → x = -20 := by sorry

end NUMINAMATH_CALUDE_triangle_third_vertex_l3052_305287


namespace NUMINAMATH_CALUDE_sock_selection_l3052_305241

theorem sock_selection (n m k : ℕ) : 
  n = 8 → m = 4 → k = 1 →
  (Nat.choose n m) - (Nat.choose (n - k) m) = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_l3052_305241


namespace NUMINAMATH_CALUDE_fourth_term_is_ten_l3052_305272

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1) * seq.d

theorem fourth_term_is_ten
  (seq : ArithmeticSequence)
  (h : seq.nthTerm 2 + seq.nthTerm 6 = 20) :
  seq.nthTerm 4 = 10 := by
  sorry

#check fourth_term_is_ten

end NUMINAMATH_CALUDE_fourth_term_is_ten_l3052_305272


namespace NUMINAMATH_CALUDE_project_work_time_l3052_305242

/-- Calculates the time spent working on a project given the number of days,
    number of naps, hours per nap, and hours per day. -/
def time_spent_working (days : ℕ) (num_naps : ℕ) (hours_per_nap : ℕ) (hours_per_day : ℕ) : ℕ :=
  days * hours_per_day - num_naps * hours_per_nap

/-- Proves that given a 4-day project where 6 seven-hour naps are taken,
    and each day has 24 hours, the time spent working on the project is 54 hours. -/
theorem project_work_time :
  time_spent_working 4 6 7 24 = 54 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_l3052_305242


namespace NUMINAMATH_CALUDE_normal_transform_theorem_l3052_305299

/-- Transforms a standard normal random variable to a normal distribution with given mean and standard deviation -/
def transform (x : ℝ) (μ σ : ℝ) : ℝ := σ * x + μ

/-- The four standard normal random variables -/
def X₁ : ℝ := 0.06
def X₂ : ℝ := -1.10
def X₃ : ℝ := -1.52
def X₄ : ℝ := 0.83

/-- The mean of the target normal distribution -/
def μ : ℝ := 2

/-- The standard deviation of the target normal distribution -/
def σ : ℝ := 3

/-- Theorem stating that the transformation of the given standard normal random variables
    results in the specified values for the target normal distribution -/
theorem normal_transform_theorem :
  (transform X₁ μ σ, transform X₂ μ σ, transform X₃ μ σ, transform X₄ μ σ) =
  (2.18, -1.3, -2.56, 4.49) := by
  sorry

end NUMINAMATH_CALUDE_normal_transform_theorem_l3052_305299


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_800_l3052_305225

theorem greatest_multiple_of_5_and_6_less_than_800 : 
  ∀ n : ℕ, n < 800 ∧ 5 ∣ n ∧ 6 ∣ n → n ≤ 780 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_800_l3052_305225


namespace NUMINAMATH_CALUDE_power_product_rule_l3052_305234

theorem power_product_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by sorry

end NUMINAMATH_CALUDE_power_product_rule_l3052_305234


namespace NUMINAMATH_CALUDE_total_selling_price_l3052_305292

def bicycle_cost : ℚ := 1600
def scooter_cost : ℚ := 8000
def motorcycle_cost : ℚ := 15000

def bicycle_loss_percent : ℚ := 10
def scooter_loss_percent : ℚ := 5
def motorcycle_loss_percent : ℚ := 8

def selling_price (cost : ℚ) (loss_percent : ℚ) : ℚ :=
  cost - (cost * loss_percent / 100)

theorem total_selling_price :
  selling_price bicycle_cost bicycle_loss_percent +
  selling_price scooter_cost scooter_loss_percent +
  selling_price motorcycle_cost motorcycle_loss_percent = 22840 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_l3052_305292


namespace NUMINAMATH_CALUDE_initial_value_problem_l3052_305255

theorem initial_value_problem : ∃! x : ℤ, (x + 82) % 456 = 0 ∧ x = 374 := by sorry

end NUMINAMATH_CALUDE_initial_value_problem_l3052_305255


namespace NUMINAMATH_CALUDE_line_bisects_circle_l3052_305277

/-- The equation of the circle -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 1 = 0

/-- The equation of the bisecting line -/
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

/-- A point is the center of the circle if it satisfies the circle's equation 
    and is equidistant from all points on the circle -/
def is_center (cx cy : ℝ) : Prop :=
  circle_eq cx cy ∧ 
  ∀ x y : ℝ, circle_eq x y → (x - cx)^2 + (y - cy)^2 = 4

/-- A line bisects a circle if and only if it passes through the circle's center -/
axiom bisects_iff_passes_through_center (a b c : ℝ) :
  (∀ x y : ℝ, circle_eq x y → a*x + b*y + c = 0) ↔
  (∃ cx cy : ℝ, is_center cx cy ∧ a*cx + b*cy + c = 0)

/-- The main theorem: the line x - y + 1 = 0 bisects the circle -/
theorem line_bisects_circle :
  ∀ x y : ℝ, circle_eq x y → line_eq x y :=
by sorry

end NUMINAMATH_CALUDE_line_bisects_circle_l3052_305277


namespace NUMINAMATH_CALUDE_math_test_results_l3052_305276

/-- Represents the score distribution for a math test -/
structure ScoreDistribution where
  prob_45 : ℚ
  prob_50 : ℚ
  prob_55 : ℚ
  prob_60 : ℚ

/-- Represents the conditions of the math test -/
structure MathTest where
  total_questions : ℕ
  options_per_question : ℕ
  points_per_correct : ℕ
  certain_correct : ℕ
  uncertain_two_eliminated : ℕ
  uncertain_one_eliminated : ℕ

/-- Calculates the probability of scoring 55 points given the test conditions -/
def prob_55 (test : MathTest) : ℚ :=
  sorry

/-- Calculates the score distribution given the test conditions -/
def score_distribution (test : MathTest) : ScoreDistribution :=
  sorry

/-- Calculates the expected value of the score given the score distribution -/
def expected_value (dist : ScoreDistribution) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem math_test_results (test : MathTest) 
  (h1 : test.total_questions = 12)
  (h2 : test.options_per_question = 4)
  (h3 : test.points_per_correct = 5)
  (h4 : test.certain_correct = 9)
  (h5 : test.uncertain_two_eliminated = 2)
  (h6 : test.uncertain_one_eliminated = 1) :
  prob_55 test = 1/3 ∧ 
  expected_value (score_distribution test) = 165/3 :=
sorry

end NUMINAMATH_CALUDE_math_test_results_l3052_305276


namespace NUMINAMATH_CALUDE_pet_shop_total_cost_l3052_305214

/-- Calculates the total cost of purchasing all pets with discounts -/
def total_cost_with_discounts (puppy1_price puppy2_price kitten1_price kitten2_price 
                               parakeet1_price parakeet2_price parakeet3_price : ℚ) : ℚ :=
  let puppy_total := puppy1_price + puppy2_price
  let puppy_discount := puppy_total * (5 / 100)
  let puppy_cost := puppy_total - puppy_discount

  let kitten_total := kitten1_price + kitten2_price
  let kitten_discount := kitten_total * (10 / 100)
  let kitten_cost := kitten_total - kitten_discount

  let parakeet_total := parakeet1_price + parakeet2_price + parakeet3_price
  let parakeet_discount := min parakeet1_price (min parakeet2_price parakeet3_price) / 2
  let parakeet_cost := parakeet_total - parakeet_discount

  puppy_cost + kitten_cost + parakeet_cost

/-- The theorem stating the total cost of purchasing all pets with discounts -/
theorem pet_shop_total_cost :
  total_cost_with_discounts 72 78 48 52 10 12 14 = 263.5 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_total_cost_l3052_305214


namespace NUMINAMATH_CALUDE_open_box_volume_l3052_305297

theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_square_side : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 38)
  (h3 : cut_square_side = 8) :
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 5632 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l3052_305297


namespace NUMINAMATH_CALUDE_intersection_theorem_l3052_305294

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2 + 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_theorem : A_intersect_B = {x | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3052_305294


namespace NUMINAMATH_CALUDE_book_price_reduction_l3052_305218

theorem book_price_reduction : 
  let initial_discount : ℝ := 0.3
  let price_increase : ℝ := 0.2
  let final_discount : ℝ := 0.5
  let original_price : ℝ := 1
  let discounted_price := original_price * (1 - initial_discount)
  let increased_price := discounted_price * (1 + price_increase)
  let final_price := increased_price * (1 - final_discount)
  let total_reduction := (original_price - final_price) / original_price
  total_reduction = 0.58
:= by sorry

end NUMINAMATH_CALUDE_book_price_reduction_l3052_305218


namespace NUMINAMATH_CALUDE_acute_triangle_cosine_inequality_l3052_305279

theorem acute_triangle_cosine_inequality (A B C : ℝ) 
  (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (Real.cos A / Real.cos (B - C)) + 
  (Real.cos B / Real.cos (C - A)) + 
  (Real.cos C / Real.cos (A - B)) ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_acute_triangle_cosine_inequality_l3052_305279


namespace NUMINAMATH_CALUDE_blueberry_pies_count_l3052_305271

/-- Given a total of 30 pies and a ratio of 2:3:4:1 for apple:blueberry:cherry:peach pies,
    the number of blueberry pies is 9. -/
theorem blueberry_pies_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio peach_ratio : ℕ) :
  total_pies = 30 →
  apple_ratio = 2 →
  blueberry_ratio = 3 →
  cherry_ratio = 4 →
  peach_ratio = 1 →
  blueberry_ratio * (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio + peach_ratio)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_pies_count_l3052_305271


namespace NUMINAMATH_CALUDE_four_digit_odd_divisible_by_digits_l3052_305211

def is_odd (n : Nat) : Prop := ∃ k, n = 2 * k + 1

def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digits_of (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

theorem four_digit_odd_divisible_by_digits :
  ∀ n : Nat,
  is_four_digit n →
  (let d := digits_of n
   d.length = 4 ∧
   (∀ x ∈ d, is_odd x) ∧
   d.toFinset.card = 4 ∧
   (∀ x ∈ d, n % x = 0)) →
  n ∈ [1395, 1935, 3195, 3915, 9135, 9315] := by
sorry

end NUMINAMATH_CALUDE_four_digit_odd_divisible_by_digits_l3052_305211


namespace NUMINAMATH_CALUDE_tank_capacity_l3052_305263

/-- 
Given a tank with an unknown capacity, prove that if it's initially filled to 3/4 of its capacity,
and adding 7 gallons fills it to 9/10 of its capacity, then the tank's total capacity is 140/3 gallons.
-/
theorem tank_capacity (tank_capacity : ℝ) : 
  (3 / 4 * tank_capacity + 7 = 9 / 10 * tank_capacity) ↔ 
  (tank_capacity = 140 / 3) := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l3052_305263


namespace NUMINAMATH_CALUDE_max_product_constrained_l3052_305285

theorem max_product_constrained (x y : ℕ+) (h : 7 * x + 4 * y = 140) :
  x * y ≤ 168 :=
sorry

end NUMINAMATH_CALUDE_max_product_constrained_l3052_305285


namespace NUMINAMATH_CALUDE_jacket_sale_profit_l3052_305256

/-- Calculates the merchant's gross profit for a jacket sale -/
theorem jacket_sale_profit (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  purchase_price = 60 ∧ 
  markup_percent = 0.25 ∧ 
  discount_percent = 0.20 → 
  let selling_price := purchase_price / (1 - markup_percent)
  let discounted_price := selling_price * (1 - discount_percent)
  discounted_price - purchase_price = 4 :=
by sorry

end NUMINAMATH_CALUDE_jacket_sale_profit_l3052_305256


namespace NUMINAMATH_CALUDE_relationship_equation_l3052_305202

/-- Given a relationship "a number that is 3 more than half of x is equal to twice y",
    prove that the equation (1/2)x + 3 = 2y correctly represents this relationship. -/
theorem relationship_equation (x y : ℝ) :
  (∃ (n : ℝ), n = (1/2) * x + 3 ∧ n = 2 * y) ↔ (1/2) * x + 3 = 2 * y :=
by sorry

end NUMINAMATH_CALUDE_relationship_equation_l3052_305202


namespace NUMINAMATH_CALUDE_adoption_cost_theorem_l3052_305260

def cat_cost : ℕ := 50
def adult_dog_cost : ℕ := 100
def puppy_cost : ℕ := 150

def cats_adopted : ℕ := 2
def adult_dogs_adopted : ℕ := 3
def puppies_adopted : ℕ := 2

def total_cost : ℕ := 
  cat_cost * cats_adopted + 
  adult_dog_cost * adult_dogs_adopted + 
  puppy_cost * puppies_adopted

theorem adoption_cost_theorem : total_cost = 700 := by
  sorry

end NUMINAMATH_CALUDE_adoption_cost_theorem_l3052_305260


namespace NUMINAMATH_CALUDE_first_square_with_two_twos_l3052_305296

def starts_with_two_twos (n : ℕ) : Prop :=
  (n / 1000 = 2) ∧ ((n / 100) % 10 = 2)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem first_square_with_two_twos : 
  ∃! n : ℕ, 
    (∀ m : ℕ, m < n → ¬(starts_with_two_twos (m^2))) ∧ 
    (starts_with_two_twos (n^2)) ∧
    (∃ k : ℕ, k > n ∧ starts_with_two_twos (k^2) ∧ sum_of_digits (k^2) = 13) ∧
    n = 47 := by sorry

end NUMINAMATH_CALUDE_first_square_with_two_twos_l3052_305296


namespace NUMINAMATH_CALUDE_tangent_slope_ratio_l3052_305206

-- Define the function f(x) = ax² + b
def f (a b x : ℝ) : ℝ := a * x^2 + b

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 2 * a * x

theorem tangent_slope_ratio (a b : ℝ) :
  f_derivative a b 1 = 2 ∧ f a b 1 = 3 → a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_ratio_l3052_305206


namespace NUMINAMATH_CALUDE_cube_root_equation_product_l3052_305274

theorem cube_root_equation_product (a b : ℤ) : 
  (3 * Real.sqrt (Real.rpow 5 (1/3) - Real.rpow 4 (1/3)) = Real.rpow a (1/3) + Real.rpow b (1/3) + Real.rpow 2 (1/3)) →
  a * b = -500 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_product_l3052_305274


namespace NUMINAMATH_CALUDE_terminal_side_half_angle_l3052_305245

-- Define a function to determine the quadrant of an angle
def quadrant (θ : ℝ) : Set Nat :=
  if 0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2 then {1}
  else if Real.pi / 2 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi then {2}
  else if Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2 then {3}
  else {4}

-- Theorem statement
theorem terminal_side_half_angle (α : ℝ) :
  quadrant α = {3} → quadrant (α / 2) = {2} ∨ quadrant (α / 2) = {4} := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_half_angle_l3052_305245


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l3052_305247

theorem reciprocal_sum_of_quadratic_roots : 
  ∀ x₁ x₂ : ℝ, 
  (x₁^2 + x₁ = 5*x₁ + 6) → 
  (x₂^2 + x₂ = 5*x₂ + 6) → 
  x₁ ≠ x₂ →
  (1/x₁ + 1/x₂ = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l3052_305247


namespace NUMINAMATH_CALUDE_fruit_problem_max_a_l3052_305246

/-- Represents the fruit purchase and sale problem -/
def FruitProblem (totalCost totalWeight cherryPrice cantaloupePrice : ℝ)
  (secondTotalWeight secondMaxCost minProfit : ℝ)
  (cherrySellingPrice cantaloupeSellingPrice : ℝ) :=
  ∀ (a : ℕ),
    let n := (secondMaxCost - 6 * secondTotalWeight) / 29
    (35 * n + 6 * (secondTotalWeight - n) ≤ secondMaxCost) ∧
    (20 * (n - a) + 4 * (secondTotalWeight - n - 2 * a) ≥ minProfit) →
    a ≤ 35

/-- The maximum value of a in the fruit problem is 35 -/
theorem fruit_problem_max_a :
  FruitProblem 9160 560 35 6 300 5280 2120 55 10 :=
sorry

end NUMINAMATH_CALUDE_fruit_problem_max_a_l3052_305246


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l3052_305226

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 50) :
  (original_price - sale_price) / original_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l3052_305226


namespace NUMINAMATH_CALUDE_minimum_satisfying_number_l3052_305280

def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = b * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_conditions (A : ℕ) : Prop :=
  A > 0 ∧
  is_multiple_of A 3 ∧
  ¬is_multiple_of A 9 ∧
  is_multiple_of (A + digit_product A) 9

theorem minimum_satisfying_number :
  satisfies_conditions 138 ∧ ∀ A : ℕ, A < 138 → ¬satisfies_conditions A :=
sorry

end NUMINAMATH_CALUDE_minimum_satisfying_number_l3052_305280


namespace NUMINAMATH_CALUDE_mabel_transactions_l3052_305230

/-- Represents the number of transactions handled by each person -/
structure Transactions where
  mabel : ℕ
  anthony : ℕ
  cal : ℕ
  jade : ℕ

/-- The conditions of the problem -/
def problem_conditions (t : Transactions) : Prop :=
  t.anthony = t.mabel + t.mabel / 10 ∧
  t.cal = (2 * t.anthony) / 3 ∧
  t.jade = t.cal + 16 ∧
  t.jade = 82

/-- The theorem to prove -/
theorem mabel_transactions :
  ∀ t : Transactions, problem_conditions t → t.mabel = 90 := by
  sorry

end NUMINAMATH_CALUDE_mabel_transactions_l3052_305230


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3052_305222

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 1 → 
    1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x + 3) + B / (x - 1) + C / (x - 1)^2) →
  A = 1/16 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3052_305222


namespace NUMINAMATH_CALUDE_investment_problem_l3052_305273

theorem investment_problem (total : ℝ) (rate_a rate_b rate_c : ℝ) 
  (h_total : total = 425)
  (h_rate_a : rate_a = 0.05)
  (h_rate_b : rate_b = 0.08)
  (h_rate_c : rate_c = 0.10)
  (h_equal_increase : ∃ (k : ℝ), k > 0 ∧ 
    ∀ (a b c : ℝ), a + b + c = total → 
    rate_a * a = k ∧ rate_b * b = k ∧ rate_c * c = k) :
  ∃ (a b c : ℝ), a + b + c = total ∧ 
    rate_a * a = rate_b * b ∧ rate_b * b = rate_c * c ∧ 
    c = 100 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3052_305273


namespace NUMINAMATH_CALUDE_seafood_noodles_plates_l3052_305233

/-- Given a chef's banquet with a total of 55 plates, 25 plates of lobster rolls,
    and 14 plates of spicy hot noodles, prove that the number of seafood noodle plates is 16. -/
theorem seafood_noodles_plates (total : ℕ) (lobster : ℕ) (spicy : ℕ) (seafood : ℕ)
  (h1 : total = 55)
  (h2 : lobster = 25)
  (h3 : spicy = 14)
  (h4 : total = lobster + spicy + seafood) :
  seafood = 16 := by
  sorry

end NUMINAMATH_CALUDE_seafood_noodles_plates_l3052_305233


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l3052_305293

-- Define the repeating decimals
def x : ℚ := 0.142857142857142857
def y : ℚ := 2.857142857142857142

-- State the theorem
theorem repeating_decimal_fraction : x / y = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l3052_305293


namespace NUMINAMATH_CALUDE_investment_problem_solution_l3052_305278

/-- Investment problem with two partners -/
structure InvestmentProblem where
  /-- Ratio of investments for partners p and q -/
  investmentRatio : Rat × Rat
  /-- Ratio of profits for partners p and q -/
  profitRatio : Rat × Rat
  /-- Investment period for partner q in months -/
  qPeriod : ℕ

/-- Solution to the investment problem -/
def solveProblem (prob : InvestmentProblem) : ℚ :=
  let (pInvest, qInvest) := prob.investmentRatio
  let (pProfit, qProfit) := prob.profitRatio
  (qProfit * pInvest * prob.qPeriod) / (pProfit * qInvest)

/-- Theorem stating the solution to the specific problem -/
theorem investment_problem_solution :
  let prob : InvestmentProblem := {
    investmentRatio := (7, 5)
    profitRatio := (7, 10)
    qPeriod := 4
  }
  solveProblem prob = 2 := by sorry


end NUMINAMATH_CALUDE_investment_problem_solution_l3052_305278


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3052_305223

theorem sqrt_product_equality : Real.sqrt 125 * Real.sqrt 45 * Real.sqrt 10 = 75 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3052_305223


namespace NUMINAMATH_CALUDE_age_of_b_l3052_305243

/-- Given three people A, B, and C, their average age, and the average age of A and C, prove the age of B. -/
theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 27 →  -- The average age of A, B, and C is 27
  (a + c) / 2 = 29 →      -- The average age of A and C is 29
  b = 23 :=               -- The age of B is 23
by sorry

end NUMINAMATH_CALUDE_age_of_b_l3052_305243


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3052_305217

theorem quadratic_inequality_solution (t a : ℝ) : 
  (∀ x, tx^2 - 6*x + t^2 < 0 ↔ x ∈ Set.Ioi 1 ∪ Set.Iic a) →
  (t*a^2 - 6*a + t^2 = 0 ∧ t*1^2 - 6*1 + t^2 = 0) →
  t < 0 →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3052_305217


namespace NUMINAMATH_CALUDE_sum_of_two_arithmetic_sequences_l3052_305207

/-- Sum of two arithmetic sequences with specific properties -/
theorem sum_of_two_arithmetic_sequences : 
  let seq1 := [2, 14, 26, 38, 50]
  let seq2 := [6, 18, 30, 42, 54]
  (seq1.sum + seq2.sum) = 280 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_arithmetic_sequences_l3052_305207


namespace NUMINAMATH_CALUDE_octagon_intersection_only_hexagonal_prism_l3052_305252

/-- Represents the possible geometric solids --/
inductive GeometricSolid
  | TriangularPrism
  | RectangularPrism
  | PentagonalPrism
  | HexagonalPrism

/-- Represents the possible shapes resulting from a plane intersection --/
inductive IntersectionShape
  | Triangle
  | Quadrilateral
  | Pentagon
  | Hexagon
  | Heptagon
  | Octagon
  | Rectangle

/-- Returns the possible intersection shapes for a given geometric solid --/
def possibleIntersections (solid : GeometricSolid) : List IntersectionShape :=
  match solid with
  | GeometricSolid.TriangularPrism => [IntersectionShape.Quadrilateral, IntersectionShape.Triangle]
  | GeometricSolid.RectangularPrism => [IntersectionShape.Pentagon, IntersectionShape.Quadrilateral, IntersectionShape.Triangle, IntersectionShape.Rectangle]
  | GeometricSolid.PentagonalPrism => [IntersectionShape.Hexagon, IntersectionShape.Pentagon, IntersectionShape.Rectangle, IntersectionShape.Triangle]
  | GeometricSolid.HexagonalPrism => [IntersectionShape.Octagon, IntersectionShape.Heptagon, IntersectionShape.Rectangle]

/-- Theorem: Only the hexagonal prism can produce an octagonal intersection --/
theorem octagon_intersection_only_hexagonal_prism :
  ∀ (solid : GeometricSolid),
    (IntersectionShape.Octagon ∈ possibleIntersections solid) ↔ (solid = GeometricSolid.HexagonalPrism) :=
by sorry


end NUMINAMATH_CALUDE_octagon_intersection_only_hexagonal_prism_l3052_305252


namespace NUMINAMATH_CALUDE_solution_set_of_f_greater_than_one_l3052_305298

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

-- State the theorem
theorem solution_set_of_f_greater_than_one :
  {x : ℝ | f x > 1} = Set.Ioo (2/3) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_greater_than_one_l3052_305298


namespace NUMINAMATH_CALUDE_cone_from_sector_cone_sector_proof_l3052_305258

theorem cone_from_sector (sector_angle : Real) (circle_radius : Real) 
  (base_radius : Real) (slant_height : Real) : Prop :=
  sector_angle = 252 ∧
  circle_radius = 10 ∧
  base_radius = 7 ∧
  slant_height = 10 ∧
  2 * Real.pi * base_radius = (sector_angle / 360) * 2 * Real.pi * circle_radius ∧
  base_radius ^ 2 + (circle_radius ^ 2 - base_radius ^ 2) = slant_height ^ 2

theorem cone_sector_proof : 
  ∃ (sector_angle circle_radius base_radius slant_height : Real),
    cone_from_sector sector_angle circle_radius base_radius slant_height := by
  sorry

end NUMINAMATH_CALUDE_cone_from_sector_cone_sector_proof_l3052_305258


namespace NUMINAMATH_CALUDE_three_heads_probability_l3052_305236

theorem three_heads_probability (p : ℝ) (h_fair : p = 1 / 2) :
  p * p * p = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_probability_l3052_305236


namespace NUMINAMATH_CALUDE_tan_one_condition_l3052_305275

theorem tan_one_condition (x : Real) : 
  (∃ k : Int, x = (k * Real.pi) / 4) ∧ 
  (∃ y : Real, (∃ m : Int, y = (m * Real.pi) / 4) ∧ Real.tan y = 1) ∧ 
  (∃ z : Real, (∃ n : Int, z = (n * Real.pi) / 4) ∧ Real.tan z ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_one_condition_l3052_305275


namespace NUMINAMATH_CALUDE_factorization_equality_l3052_305237

theorem factorization_equality (a b : ℝ) : a^2 * b + 2 * a * b^2 + b^3 = b * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3052_305237


namespace NUMINAMATH_CALUDE_roots_of_equation_l3052_305210

theorem roots_of_equation (x : ℝ) :
  (x^2 - 5*x + 6) * x * (x - 5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3052_305210


namespace NUMINAMATH_CALUDE_triangle_sum_special_case_l3052_305228

def triangle_sum (a b : ℕ) : ℕ :=
  let n_min := a.max b - (a.min b - 1)
  let n_max := a + b - 1
  (n_max - n_min + 1) * (n_max + n_min) / 2

theorem triangle_sum_special_case : triangle_sum 7 10 = 260 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_special_case_l3052_305228


namespace NUMINAMATH_CALUDE_charlie_dana_difference_l3052_305209

/-- Represents the number of games won by each player -/
structure GameWins where
  perry : ℕ
  dana : ℕ
  charlie : ℕ
  phil : ℕ

/-- The conditions of the golf game results -/
def golf_results (g : GameWins) : Prop :=
  g.perry = g.dana + 5 ∧
  g.charlie < g.dana ∧
  g.phil = g.charlie + 3 ∧
  g.phil = 12 ∧
  g.perry = g.phil + 4

theorem charlie_dana_difference (g : GameWins) (h : golf_results g) :
  g.dana - g.charlie = 2 := by
  sorry

end NUMINAMATH_CALUDE_charlie_dana_difference_l3052_305209


namespace NUMINAMATH_CALUDE_platform_length_specific_platform_length_l3052_305248

/-- The length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- Proof of the specific platform length problem -/
theorem specific_platform_length : 
  platform_length 360 45 48 = 840 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_specific_platform_length_l3052_305248


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l3052_305281

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 24)
  (h3 : wicket_keeper_age_diff = 7)
  : ∃ (team_avg_age : ℚ),
    team_avg_age = 23 ∧
    (team_size : ℚ) * team_avg_age = 
      captain_age + (captain_age + wicket_keeper_age_diff) + 
      ((team_size - 2) : ℚ) * (team_avg_age - 1) :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l3052_305281


namespace NUMINAMATH_CALUDE_minimum_point_of_transformed_graph_l3052_305201

-- Define the function representing the transformed graph
def f (x : ℝ) : ℝ := 2 * abs (x + 3) - 7

-- State the theorem
theorem minimum_point_of_transformed_graph :
  ∃ (x : ℝ), f x = f (-3) ∧ ∀ (y : ℝ), f y ≥ f (-3) ∧ f (-3) = -7 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_transformed_graph_l3052_305201


namespace NUMINAMATH_CALUDE_count_threes_up_to_80_l3052_305244

/-- Count of digit 3 in a single number -/
def countThreesInNumber (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 3 then 1 + countThreesInNumber (n / 10)
  else countThreesInNumber (n / 10)

/-- Count of digit 3 in numbers from 1 to n -/
def countThreesUpTo (n : ℕ) : ℕ :=
  List.range n |> List.map (fun i => countThreesInNumber (i + 1)) |> List.sum

/-- The count of the digit 3 in the numbers from 1 to 80 (inclusive) is equal to 9 -/
theorem count_threes_up_to_80 : countThreesUpTo 80 = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_threes_up_to_80_l3052_305244


namespace NUMINAMATH_CALUDE_number_multiplied_by_9999_l3052_305227

theorem number_multiplied_by_9999 : ∃ x : ℚ, x * 9999 = 724787425 ∧ x = 72487.5 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_9999_l3052_305227


namespace NUMINAMATH_CALUDE_gladys_age_ratio_l3052_305208

def gladys_age : ℕ := 30

def billy_age : ℕ := gladys_age / 3

def lucas_age : ℕ := 8 - 3

def sum_billy_lucas : ℕ := billy_age + lucas_age

theorem gladys_age_ratio : 
  gladys_age / sum_billy_lucas = 2 :=
by sorry

end NUMINAMATH_CALUDE_gladys_age_ratio_l3052_305208


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3052_305267

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + 2*y + z = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ ∀ (a b c : ℝ), a + 2*b + c = 1 → x^2 + 4*y^2 + z^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3052_305267


namespace NUMINAMATH_CALUDE_max_sum_pair_contains_96420_l3052_305284

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ 
  a ≥ 10000 ∧ a < 100000 ∧ 
  b ≥ 10000 ∧ b < 100000 ∧
  (∀ d : ℕ, d < 10 → 
    (∃! i : ℕ, i < 5 ∧ (a / 10^i) % 10 = d) ∨
    (∃! i : ℕ, i < 5 ∧ (b / 10^i) % 10 = d))

def is_max_sum_pair (a b : ℕ) : Prop :=
  is_valid_pair a b ∧
  ∀ c d : ℕ, is_valid_pair c d → a + b ≥ c + d

theorem max_sum_pair_contains_96420 :
  ∃ n : ℕ, is_max_sum_pair 96420 n ∨ is_max_sum_pair n 96420 :=
sorry

end NUMINAMATH_CALUDE_max_sum_pair_contains_96420_l3052_305284


namespace NUMINAMATH_CALUDE_curve_and_line_properties_l3052_305262

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x = -2 ∨ 3*x + 4*y - 2 = 0

-- Define the distance ratio condition
def distance_ratio (x y : ℝ) : Prop :=
  (x^2 + y^2) = (1/4) * ((x - 3)^2 + y^2)

-- Define the intersection condition
def intersects_curve (l : ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), l x₁ y₁ ∧ l x₂ y₂ ∧ c x₁ y₁ ∧ c x₂ y₂ ∧ 
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Main theorem
theorem curve_and_line_properties :
  (∀ x y, curve_C x y ↔ distance_ratio x y) ∧
  (line_l (-2) 2) ∧
  (intersects_curve line_l curve_C) ∧
  (∃ x₁ y₁ x₂ y₂, line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) :=
sorry

end NUMINAMATH_CALUDE_curve_and_line_properties_l3052_305262


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l3052_305259

def normal_distribution (mean : ℝ) (std_dev : ℝ) := { μ : ℝ // μ = mean }

theorem two_std_dev_below_mean 
  (μ : ℝ) (σ : ℝ) (h_μ : μ = 14.5) (h_σ : σ = 1.7) :
  ∃ (x : ℝ), x = μ - 2 * σ ∧ x = 11.1 :=
sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l3052_305259


namespace NUMINAMATH_CALUDE_odd_function_solution_set_l3052_305213

/-- An odd function f: ℝ → ℝ satisfying certain conditions -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (f 2 = 0) ∧ 
  (∀ x > 0, x * (deriv f x) - f x < 0)

/-- The solution set for f(x)/x > 0 given the conditions on f -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2}

theorem odd_function_solution_set (f : ℝ → ℝ) (hf : OddFunction f) :
  {x : ℝ | f x / x > 0} = SolutionSet f :=
sorry

end NUMINAMATH_CALUDE_odd_function_solution_set_l3052_305213


namespace NUMINAMATH_CALUDE_pizzeria_sales_l3052_305235

theorem pizzeria_sales (small_price large_price total_revenue small_count : ℕ) 
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_revenue = 40)
  (h4 : small_count = 8) :
  (total_revenue - small_price * small_count) / large_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizzeria_sales_l3052_305235


namespace NUMINAMATH_CALUDE_chocolate_cookie_price_l3052_305229

/-- Given the sale of cookies and total revenue, prove the price of chocolate cookies -/
theorem chocolate_cookie_price
  (chocolate_count : ℕ)
  (vanilla_count : ℕ)
  (vanilla_price : ℚ)
  (total_revenue : ℚ)
  (h1 : chocolate_count = 220)
  (h2 : vanilla_count = 70)
  (h3 : vanilla_price = 2)
  (h4 : total_revenue = 360)
  : ∃ (chocolate_price : ℚ),
    chocolate_price * chocolate_count + vanilla_price * vanilla_count = total_revenue ∧
    chocolate_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cookie_price_l3052_305229


namespace NUMINAMATH_CALUDE_average_female_height_l3052_305251

/-- Given the overall average height of students is 180 cm, the average height of male students
    is 182 cm, and the ratio of men to women is 5:1, prove that the average female height is 170 cm. -/
theorem average_female_height
  (overall_avg : ℝ)
  (male_avg : ℝ)
  (ratio : ℕ)
  (h1 : overall_avg = 180)
  (h2 : male_avg = 182)
  (h3 : ratio = 5)
  : ∃ (female_avg : ℝ), female_avg = 170 ∧
    (ratio * male_avg + female_avg) / (ratio + 1) = overall_avg :=
by
  sorry

end NUMINAMATH_CALUDE_average_female_height_l3052_305251


namespace NUMINAMATH_CALUDE_discount_percentages_l3052_305219

theorem discount_percentages :
  ∃ (x y : ℕ), 0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10 ∧
  69000 * (100 - x) * (100 - y) / 10000 = 60306 ∧
  ((x = 5 ∧ y = 8) ∨ (x = 8 ∧ y = 5)) := by
  sorry

end NUMINAMATH_CALUDE_discount_percentages_l3052_305219


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3052_305239

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + x - 4*m = 0) ↔ m ≥ -1/16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3052_305239


namespace NUMINAMATH_CALUDE_expression_simplification_l3052_305216

theorem expression_simplification (m : ℝ) (h : m = 5) :
  (3*m + 6) / (m^2 + 4*m + 4) / ((m - 2) / (m + 2)) + 1 / (2 - m) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3052_305216


namespace NUMINAMATH_CALUDE_age_difference_l3052_305221

theorem age_difference (age1 age2 : ℕ) : 
  age1 + age2 = 27 → age1 = 13 → age2 = 14 → age2 - age1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3052_305221


namespace NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_l3052_305291

theorem arithmetic_progression_implies_equal (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let g := Real.sqrt (a * b)
  let p := (a + b) / 2
  let q := Real.sqrt ((a^2 + b^2) / 2)
  (g + q = 2 * p) → a = b :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_l3052_305291


namespace NUMINAMATH_CALUDE_remainder_proof_l3052_305269

def smallest_prime_greater_than_1000 : ℕ → Prop :=
  λ x => Prime x ∧ x > 1000 ∧ ∀ y, Prime y ∧ y > 1000 → x ≤ y

theorem remainder_proof (x : ℕ) (h : smallest_prime_greater_than_1000 x) :
  (10000 - 999) % x = 945 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l3052_305269


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3052_305250

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (((64 - 2*x) ^ (1/4) : ℝ) + ((48 + 2*x) ^ (1/4) : ℝ) = 6) ↔ (x = 32 ∨ x = -8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3052_305250


namespace NUMINAMATH_CALUDE_cart_distance_proof_l3052_305268

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ → ℕ :=
  fun i => a₁ + (i - 1) * d

def sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem cart_distance_proof (a₁ d n : ℕ) 
  (h₁ : a₁ = 5) 
  (h₂ : d = 7) 
  (h₃ : n = 30) : 
  sequence_sum a₁ d n = 3195 :=
sorry

end NUMINAMATH_CALUDE_cart_distance_proof_l3052_305268


namespace NUMINAMATH_CALUDE_ratio_problem_l3052_305288

theorem ratio_problem (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1 / 5 * x) / (1 / 6 * y) = 0.72 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3052_305288


namespace NUMINAMATH_CALUDE_furniture_purchase_price_l3052_305224

theorem furniture_purchase_price :
  let marked_price : ℝ := 132
  let discount_rate : ℝ := 0.1
  let profit_rate : ℝ := 0.1
  let purchase_price : ℝ := 108
  marked_price * (1 - discount_rate) - purchase_price = profit_rate * purchase_price :=
by
  sorry

end NUMINAMATH_CALUDE_furniture_purchase_price_l3052_305224


namespace NUMINAMATH_CALUDE_trousers_final_cost_l3052_305264

def calculate_final_cost (original_price : ℝ) (in_store_discount : ℝ) (additional_promotion : ℝ) (sales_tax : ℝ) (handling_fee : ℝ) : ℝ :=
  let price_after_in_store_discount := original_price * (1 - in_store_discount)
  let price_after_additional_promotion := price_after_in_store_discount * (1 - additional_promotion)
  let price_with_tax := price_after_additional_promotion * (1 + sales_tax)
  price_with_tax + handling_fee

theorem trousers_final_cost :
  calculate_final_cost 100 0.20 0.10 0.05 5 = 80.60 := by
  sorry

end NUMINAMATH_CALUDE_trousers_final_cost_l3052_305264


namespace NUMINAMATH_CALUDE_car_start_time_difference_l3052_305257

/-- Two cars traveling at the same speed with specific distance ratios at different times --/
theorem car_start_time_difference
  (speed : ℝ)
  (distance_ratio_10am : ℝ)
  (distance_ratio_12pm : ℝ)
  (h1 : speed > 0)
  (h2 : distance_ratio_10am = 5)
  (h3 : distance_ratio_12pm = 3)
  : ∃ (start_time_diff : ℝ),
    start_time_diff = 8 ∧
    distance_ratio_10am * (10 - start_time_diff) = 10 ∧
    distance_ratio_12pm * (12 - start_time_diff) = 12 :=
by sorry

end NUMINAMATH_CALUDE_car_start_time_difference_l3052_305257


namespace NUMINAMATH_CALUDE_batsman_highest_score_l3052_305212

theorem batsman_highest_score 
  (total_innings : ℕ)
  (average : ℚ)
  (score_difference : ℕ)
  (average_excluding_extremes : ℚ)
  (h : total_innings = 46)
  (h1 : average = 61)
  (h2 : score_difference = 150)
  (h3 : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (total_innings : ℚ) * average = 
      ((total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + lowest_score) ∧
    highest_score = 202 := by
  sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l3052_305212


namespace NUMINAMATH_CALUDE_triangle_proof_l3052_305231

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_proof (t : Triangle) 
  (h1 : (t.a - t.b + t.c) / t.c = t.b / (t.a + t.b - t.c))
  (h2 : t.b - t.c = (Real.sqrt 3 / 3) * t.a) :
  t.A = π / 3 ∧ t.B = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_proof_l3052_305231


namespace NUMINAMATH_CALUDE_relay_race_distance_per_member_l3052_305283

theorem relay_race_distance_per_member 
  (total_distance : ℕ) 
  (team_size : ℕ) 
  (h1 : total_distance = 150) 
  (h2 : team_size = 5) :
  total_distance / team_size = 30 := by
sorry

end NUMINAMATH_CALUDE_relay_race_distance_per_member_l3052_305283


namespace NUMINAMATH_CALUDE_bisection_next_step_l3052_305261

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_continuous : ContinuousOn f (Set.Icc 0 1)
axiom f_neg_zero : f 0 < 0
axiom f_neg_half : f 0.5 < 0
axiom f_pos_one : f 1 > 0

-- Define the theorem
theorem bisection_next_step :
  ∃ x ∈ Set.Ioo 0.5 1, f x = 0 ∧ 
  (∀ y, y ∈ Set.Icc 0 1 → f y = 0 → y ∈ Set.Icc 0.5 1) ∧
  (0.75 = (0.5 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_bisection_next_step_l3052_305261


namespace NUMINAMATH_CALUDE_vector_c_value_l3052_305253

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (-2, 4)

theorem vector_c_value :
  ∀ c : ℝ × ℝ, (4 • a) + (3 • b - 2 • a) + c = (0, 0) → c = (4, -6) := by
  sorry

end NUMINAMATH_CALUDE_vector_c_value_l3052_305253


namespace NUMINAMATH_CALUDE_monotonically_decreasing_implies_a_leq_neg_three_l3052_305295

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1

-- State the theorem
theorem monotonically_decreasing_implies_a_leq_neg_three :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_implies_a_leq_neg_three_l3052_305295


namespace NUMINAMATH_CALUDE_davids_physics_marks_l3052_305232

def marks_english : ℕ := 76
def marks_mathematics : ℕ := 65
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 75
def num_subjects : ℕ := 5

theorem davids_physics_marks :
  ∃ (marks_physics : ℕ),
    marks_physics = average_marks * num_subjects - (marks_english + marks_mathematics + marks_chemistry + marks_biology) ∧
    marks_physics = 82 := by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l3052_305232


namespace NUMINAMATH_CALUDE_paint_ornaments_l3052_305266

/-- Represents the problem of painting star-shaped ornaments on tiles --/
theorem paint_ornaments (num_tiles : ℕ) (paint_coverage : ℝ) (tile_side : ℝ) 
  (pentagon_area : ℝ) (triangle_base triangle_height : ℝ) : 
  num_tiles = 20 → 
  paint_coverage = 750 → 
  tile_side = 12 → 
  pentagon_area = 15 → 
  triangle_base = 4 → 
  triangle_height = 6 → 
  (num_tiles * (tile_side^2 - 4*pentagon_area - 2*triangle_base*triangle_height) ≤ paint_coverage) :=
by sorry

end NUMINAMATH_CALUDE_paint_ornaments_l3052_305266


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l3052_305240

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- State the property of f
axiom f_property : ∀ x : ℝ, f x = f (3 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry :
  is_axis_of_symmetry (3/2) f :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l3052_305240


namespace NUMINAMATH_CALUDE_profit_share_b_profit_share_b_is_1500_l3052_305203

theorem profit_share_b (capital_a capital_b capital_c : ℕ) 
  (profit_diff_ac : ℚ) (profit_share_b : ℚ) : Prop :=
  capital_a = 8000 ∧ 
  capital_b = 10000 ∧ 
  capital_c = 12000 ∧ 
  profit_diff_ac = 600 ∧
  profit_share_b = 1500 ∧
  ∃ (total_profit : ℚ),
    total_profit * (capital_b : ℚ) / (capital_a + capital_b + capital_c : ℚ) = profit_share_b ∧
    total_profit * (capital_c - capital_a : ℚ) / (capital_a + capital_b + capital_c : ℚ) = profit_diff_ac

-- Proof
theorem profit_share_b_is_1500 : 
  profit_share_b 8000 10000 12000 600 1500 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_b_profit_share_b_is_1500_l3052_305203


namespace NUMINAMATH_CALUDE_shanghai_expo_2010_l3052_305204

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)

/-- Calculates the number of days in a year -/
def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

/-- Calculates the day of the week for a given date -/
def dayOfWeek (year month day : Nat) : DayOfWeek := sorry

/-- Calculates the number of days between two dates in the same year -/
def daysBetween (year startMonth startDay endMonth endDay : Nat) : Nat := sorry

theorem shanghai_expo_2010 :
  let year := 2010
  let mayFirst := DayOfWeek.Saturday
  ¬isLeapYear year ∧
  daysInYear year = 365 ∧
  dayOfWeek year 5 31 = DayOfWeek.Monday ∧
  daysBetween year 5 1 10 31 = 184 := by sorry

end NUMINAMATH_CALUDE_shanghai_expo_2010_l3052_305204


namespace NUMINAMATH_CALUDE_num_factors_of_given_number_l3052_305238

/-- The number of distinct natural-number factors of 8^2 * 9^3 * 7^5 -/
def num_factors : ℕ :=
  (7 : ℕ) * (7 : ℕ) * (6 : ℕ)

/-- The given number 8^2 * 9^3 * 7^5 -/
def given_number : ℕ :=
  (8^2 : ℕ) * (9^3 : ℕ) * (7^5 : ℕ)

theorem num_factors_of_given_number :
  (Finset.filter (fun d => given_number % d = 0) (Finset.range (given_number + 1))).card = num_factors :=
by sorry

end NUMINAMATH_CALUDE_num_factors_of_given_number_l3052_305238


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l3052_305282

/-- Given a function f(x) = a*sin(x) - b*cos(x) with symmetry axis x = π/4,
    prove that the inclination angle of the line ax - by + c = 0 is 3π/4 -/
theorem inclination_angle_of_line (a b c : ℝ) :
  (∀ x, a * Real.sin (π/4 + x) - b * Real.cos (π/4 + x) = 
        a * Real.sin (π/4 - x) - b * Real.cos (π/4 - x)) →
  Real.arctan (a / b) = 3 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l3052_305282


namespace NUMINAMATH_CALUDE_worker_travel_time_l3052_305215

/-- Proves that if a worker walking at 5/6 of her normal speed arrives 12 minutes later than usual, her usual travel time is 60 minutes. -/
theorem worker_travel_time (normal_speed : ℝ) (normal_time : ℝ) : 
  normal_speed * normal_time = (5/6 * normal_speed) * (normal_time + 12) → 
  normal_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_worker_travel_time_l3052_305215


namespace NUMINAMATH_CALUDE_optimal_sale_info_l3052_305220

/-- Represents the selling prices and quantities of notebooks and sticky notes -/
structure SaleInfo where
  notebook_price : ℝ
  sticky_note_price : ℝ
  notebook_quantity : ℕ
  sticky_note_quantity : ℕ

/-- Calculates the total income given the sale information -/
def total_income (s : SaleInfo) : ℝ :=
  s.notebook_price * s.notebook_quantity + s.sticky_note_price * s.sticky_note_quantity

/-- Theorem stating the optimal selling prices and quantities for maximum income -/
theorem optimal_sale_info :
  ∃ (s : SaleInfo),
    -- Total number of items is 100
    s.notebook_quantity + s.sticky_note_quantity = 100 ∧
    -- 3 notebooks and 2 sticky notes sold for 65 yuan
    3 * s.notebook_price + 2 * s.sticky_note_price = 65 ∧
    -- 4 notebooks and 3 sticky notes sold for 90 yuan
    4 * s.notebook_price + 3 * s.sticky_note_price = 90 ∧
    -- Number of notebooks does not exceed 3 times the number of sticky notes
    s.notebook_quantity ≤ 3 * s.sticky_note_quantity ∧
    -- Notebook price is 15 yuan
    s.notebook_price = 15 ∧
    -- Sticky note price is 10 yuan
    s.sticky_note_price = 10 ∧
    -- Optimal quantities are 75 notebooks and 25 sticky notes
    s.notebook_quantity = 75 ∧
    s.sticky_note_quantity = 25 ∧
    -- Maximum total income is 1375 yuan
    total_income s = 1375 ∧
    -- This is the maximum income
    ∀ (t : SaleInfo),
      t.notebook_quantity + t.sticky_note_quantity = 100 →
      t.notebook_quantity ≤ 3 * t.sticky_note_quantity →
      total_income t ≤ total_income s := by
  sorry

end NUMINAMATH_CALUDE_optimal_sale_info_l3052_305220


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3052_305286

theorem cube_volume_from_surface_area : 
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3052_305286


namespace NUMINAMATH_CALUDE_triangle_area_l3052_305289

theorem triangle_area (a b c A B C S : ℝ) : 
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a > 0 →
  a = 4 →
  A = π / 4 →
  B = π / 3 →
  S = (1 / 2) * a * b * Real.sin C →
  S = 6 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3052_305289


namespace NUMINAMATH_CALUDE_positive_expression_l3052_305270

theorem positive_expression (x : ℝ) : x^2 * Real.sin x + x * Real.cos x + x^2 + (1/2 : ℝ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l3052_305270


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3052_305290

theorem complex_modulus_problem (w z : ℂ) : 
  w * z = 18 - 24 * I ∧ Complex.abs w = 3 * Real.sqrt 13 → 
  Complex.abs z = 10 / Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3052_305290


namespace NUMINAMATH_CALUDE_percent_of_y_l3052_305200

theorem percent_of_y (y : ℝ) (h : y > 0) : ((6 * y) / 20 + (3 * y) / 10) / y = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3052_305200


namespace NUMINAMATH_CALUDE_permutation_inequality_l3052_305249

theorem permutation_inequality (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (({x, y, z, w} : Finset ℝ) = {a, b, c, d}) ∧
    (2 * (x * y + z * w)^2 > (x^2 + y^2) * (z^2 + w^2)) := by
  sorry

end NUMINAMATH_CALUDE_permutation_inequality_l3052_305249
