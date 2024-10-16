import Mathlib

namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l4176_417669

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 + k*y + 4 = 0 → y = x) → 
  k = 4 ∨ k = -4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l4176_417669


namespace NUMINAMATH_CALUDE_min_value_expression_l4176_417611

theorem min_value_expression (x y : ℝ) : 
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧ 
  ∃ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 25 = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l4176_417611


namespace NUMINAMATH_CALUDE_sons_age_l4176_417638

theorem sons_age (son_age woman_age : ℕ) : 
  woman_age = 2 * son_age + 3 →
  woman_age + son_age = 84 →
  son_age = 27 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l4176_417638


namespace NUMINAMATH_CALUDE_homework_completion_l4176_417692

/-- Fraction of homework done on Monday night -/
def monday_fraction : ℚ := sorry

/-- Fraction of homework done on Tuesday night -/
def tuesday_fraction (x : ℚ) : ℚ := (1 - x) / 3

/-- Fraction of homework done on Wednesday night -/
def wednesday_fraction : ℚ := 4 / 15

theorem homework_completion (x : ℚ) :
  x + tuesday_fraction x + wednesday_fraction = 1 →
  x = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_homework_completion_l4176_417692


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_l4176_417664

theorem sum_reciprocal_squares (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_l4176_417664


namespace NUMINAMATH_CALUDE_sum_of_modified_numbers_l4176_417649

theorem sum_of_modified_numbers (R x y : ℝ) (h : x + y = R) :
  2 * (x + 4) + 2 * (y + 5) = 2 * R + 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_modified_numbers_l4176_417649


namespace NUMINAMATH_CALUDE_chicken_burger_price_proof_l4176_417644

/-- The cost of a chicken burger in won -/
def chicken_burger_cost : ℕ := 3350

/-- The cost of a bulgogi burger in won -/
def bulgogi_burger_cost : ℕ := chicken_burger_cost + 300

/-- The total cost of three bulgogi burgers and three chicken burgers in won -/
def total_cost : ℕ := 21000

theorem chicken_burger_price_proof :
  chicken_burger_cost = 3350 ∧
  bulgogi_burger_cost = chicken_burger_cost + 300 ∧
  3 * chicken_burger_cost + 3 * bulgogi_burger_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_chicken_burger_price_proof_l4176_417644


namespace NUMINAMATH_CALUDE_eulers_pedal_triangle_theorem_l4176_417672

/-- Euler's theorem on the area of pedal triangles -/
theorem eulers_pedal_triangle_theorem (S R d : ℝ) (hR : R > 0) : 
  ∃ (S' : ℝ), S' = (S / 4) * |1 - (d^2 / R^2)| := by
  sorry

end NUMINAMATH_CALUDE_eulers_pedal_triangle_theorem_l4176_417672


namespace NUMINAMATH_CALUDE_simplify_expression_l4176_417655

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 9) - (x + 6)*(3*x - 2) = 5*x - 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4176_417655


namespace NUMINAMATH_CALUDE_workshop_attendance_l4176_417673

/-- Represents the number of scientists at a workshop with various prize distributions -/
structure WorkshopAttendance where
  total : ℕ
  wolfPrize : ℕ
  nobelPrize : ℕ
  wolfAndNobel : ℕ

/-- Theorem stating the total number of scientists at the workshop -/
theorem workshop_attendance (w : WorkshopAttendance) 
  (h1 : w.wolfPrize = 31)
  (h2 : w.wolfAndNobel = 12)
  (h3 : w.nobelPrize = 23)
  (h4 : w.nobelPrize - w.wolfAndNobel = (w.total - w.wolfPrize - (w.nobelPrize - w.wolfAndNobel)) + 3) :
  w.total = 39 := by
  sorry


end NUMINAMATH_CALUDE_workshop_attendance_l4176_417673


namespace NUMINAMATH_CALUDE_teddy_bear_cost_teddy_bear_cost_proof_l4176_417688

theorem teddy_bear_cost (initial_toys : ℕ) (initial_toy_cost : ℕ) 
  (teddy_bears : ℕ) (total_cost : ℕ) : ℕ :=
  let remaining_cost := total_cost - initial_toys * initial_toy_cost
  remaining_cost / teddy_bears

theorem teddy_bear_cost_proof :
  teddy_bear_cost 28 10 20 580 = 15 := by
  sorry

end NUMINAMATH_CALUDE_teddy_bear_cost_teddy_bear_cost_proof_l4176_417688


namespace NUMINAMATH_CALUDE_least_perfect_square_exponent_l4176_417643

theorem least_perfect_square_exponent : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (∃ (k : ℕ), 2^8 + 2^11 + 2^m = k^2) → m ≥ n) ∧
  (∃ (k : ℕ), 2^8 + 2^11 + 2^n = k^2) ∧
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_least_perfect_square_exponent_l4176_417643


namespace NUMINAMATH_CALUDE_unique_ages_l4176_417621

/-- Represents the ages of Gala, Vova, and Katya -/
structure Ages where
  gala : ℕ
  vova : ℕ
  katya : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.gala < 6 ∧
  ages.vova + ages.katya = 112 ∧
  (ages.vova / ages.gala : ℚ) = (ages.katya / ages.vova : ℚ)

/-- Theorem stating that the only ages satisfying all conditions are 2, 14, and 98 -/
theorem unique_ages : ∃! ages : Ages, satisfies_conditions ages ∧ ages.gala = 2 ∧ ages.vova = 14 ∧ ages.katya = 98 := by
  sorry

end NUMINAMATH_CALUDE_unique_ages_l4176_417621


namespace NUMINAMATH_CALUDE_x_axis_fixed_slope_two_invariant_l4176_417646

/-- Transformation that maps a point (x, y) to (x-y, -y) -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - p.2, -p.2)

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

theorem x_axis_fixed :
  ∀ (x : ℝ), transform (x, 0) = (x, 0) := by sorry

theorem slope_two_invariant (b : ℝ) :
  ∀ (x y : ℝ), 
    (Line.contains { slope := 2, intercept := b } (x, y)) →
    (Line.contains { slope := 2, intercept := b } (transform (x, y))) := by sorry

end NUMINAMATH_CALUDE_x_axis_fixed_slope_two_invariant_l4176_417646


namespace NUMINAMATH_CALUDE_ninth_root_unity_sum_l4176_417677

theorem ninth_root_unity_sum (z : ℂ) : 
  z = Complex.exp (Complex.I * (2 * Real.pi / 9)) →
  z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^6 / (1 + z^9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ninth_root_unity_sum_l4176_417677


namespace NUMINAMATH_CALUDE_equation_transformation_l4176_417680

theorem equation_transformation (x : ℝ) :
  (x + 2) / 4 = (2 * x + 3) / 6 →
  12 * ((x + 2) / 4) = 12 * ((2 * x + 3) / 6) →
  3 * (x + 2) = 2 * (2 * x + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l4176_417680


namespace NUMINAMATH_CALUDE_max_not_joined_company_l4176_417603

/-- The maximum number of people who did not join any club -/
def max_not_joined (total : ℕ) (m s z : ℕ) : ℕ :=
  total - (m + max s z)

/-- Proof that the maximum number of people who did not join any club is 26 -/
theorem max_not_joined_company : max_not_joined 60 16 18 11 = 26 := by
  sorry

end NUMINAMATH_CALUDE_max_not_joined_company_l4176_417603


namespace NUMINAMATH_CALUDE_morks_tax_rate_l4176_417620

/-- Given the tax rates and income ratio of Mork and Mindy, prove Mork's tax rate --/
theorem morks_tax_rate (r : ℝ) : 
  (r * 1 + 0.3 * 4) / 5 = 0.32 → r = 0.4 := by sorry

end NUMINAMATH_CALUDE_morks_tax_rate_l4176_417620


namespace NUMINAMATH_CALUDE_cheapest_plan_b_l4176_417689

/-- Represents the cost of a cell phone plan in cents -/
def PlanCost (flatFee minutes : ℕ) (perMinute : ℚ) : ℚ :=
  (flatFee : ℚ) * 100 + perMinute * minutes

theorem cheapest_plan_b (minutes : ℕ) : 
  (minutes ≥ 834) ↔ 
  (PlanCost 25 minutes 6 < PlanCost 0 minutes 12 ∧ 
   PlanCost 25 minutes 6 < PlanCost 0 minutes 9) :=
sorry

end NUMINAMATH_CALUDE_cheapest_plan_b_l4176_417689


namespace NUMINAMATH_CALUDE_smallest_number_proof_l4176_417661

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  b = 28 →                 -- Median is 28
  c = b + 6 →              -- Largest number is 6 more than median
  a ≤ b ∧ b ≤ c →          -- b is the median
  a = 28 :=                -- Smallest number is 28
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l4176_417661


namespace NUMINAMATH_CALUDE_passes_count_is_32_l4176_417632

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the pool and swimming scenario --/
structure SwimmingScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def countPasses (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- The specific swimming scenario from the problem --/
def problemScenario : SwimmingScenario :=
  { poolLength := 100
    swimmer1 := { speed := 4, startPosition := 0 }
    swimmer2 := { speed := 3, startPosition := 100 }
    totalTime := 720 }

/-- Theorem stating that the number of passes in the given scenario is 32 --/
theorem passes_count_is_32 : countPasses problemScenario = 32 :=
  sorry

end NUMINAMATH_CALUDE_passes_count_is_32_l4176_417632


namespace NUMINAMATH_CALUDE_problem_statement_l4176_417671

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : (3 : ℝ)^x = (4 : ℝ)^y) (h2 : 2 * x = a * y) : 
  a = 4 * (Real.log 2 / Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4176_417671


namespace NUMINAMATH_CALUDE_unique_base_solution_l4176_417623

/-- Converts a base-10 number to its representation in base b -/
def toBase (n : ℕ) (b : ℕ) : List ℕ := sorry

/-- Converts a number represented as a list of digits in base b to base 10 -/
def fromBase (digits : List ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if the equation 742_b - 305_b = 43C_b holds for a given base b -/
def equationHolds (b : ℕ) : Prop :=
  let lhs := fromBase (toBase 742 b) b - fromBase (toBase 305 b) b
  let rhs := fromBase (toBase 43 b) b * 12
  lhs = rhs

theorem unique_base_solution :
  ∃! b : ℕ, b > 1 ∧ equationHolds b :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l4176_417623


namespace NUMINAMATH_CALUDE_triangle_largest_angle_and_type_l4176_417619

-- Define the triangle with angle ratio 4:3:2
def triangle_angles (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 180 ∧
  4 * b = 3 * a ∧ 3 * c = 2 * b

-- Theorem statement
theorem triangle_largest_angle_and_type 
  (a b c : ℝ) (h : triangle_angles a b c) : 
  a = 80 ∧ a < 90 ∧ b < 90 ∧ c < 90 := by
  sorry


end NUMINAMATH_CALUDE_triangle_largest_angle_and_type_l4176_417619


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l4176_417667

theorem quadratic_function_a_range (a b c : ℝ) : 
  a ≠ 0 →
  a * (-1)^2 + b * (-1) + c = 3 →
  a * 1^2 + b * 1 + c = 1 →
  0 < c →
  c < 1 →
  1 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l4176_417667


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l4176_417641

theorem repeating_decimal_sum : 
  let x : ℚ := (23 : ℚ) / 99
  let y : ℚ := (14 : ℚ) / 999
  let z : ℚ := (6 : ℚ) / 9999
  x + y + z = (2469 : ℚ) / 9999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l4176_417641


namespace NUMINAMATH_CALUDE_expression_equality_l4176_417653

theorem expression_equality (x y : ℝ) (h : 2 * x - 3 * y = 1) : 
  10 - 4 * x + 6 * y = 8 := by
sorry

end NUMINAMATH_CALUDE_expression_equality_l4176_417653


namespace NUMINAMATH_CALUDE_triangle_equilateral_l4176_417616

theorem triangle_equilateral (m n p : ℝ) (h1 : m + n + p = 180) 
  (h2 : |m - n| + (n - p)^2 = 0) : m = n ∧ n = p := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l4176_417616


namespace NUMINAMATH_CALUDE_number_of_model_X_computers_prove_number_of_model_X_computers_l4176_417607

/-- Represents the time (in minutes) for a model X computer to complete the task -/
def modelXTime : ℝ := 72

/-- Represents the time (in minutes) for a model Y computer to complete the task -/
def modelYTime : ℝ := 36

/-- Represents the total time (in minutes) for the combined computers to complete the task -/
def totalTime : ℝ := 1

/-- Theorem stating that the number of model X computers used is 24 -/
theorem number_of_model_X_computers : ℕ :=
  24

/-- Proof that the number of model X computers used is 24 -/
theorem prove_number_of_model_X_computers :
  (modelXTime : ℝ) > 0 ∧ (modelYTime : ℝ) > 0 ∧ totalTime > 0 →
  ∃ (n : ℕ), n > 0 ∧ n = number_of_model_X_computers ∧
  (n : ℝ) * (1 / modelXTime + 1 / modelYTime) = 1 / totalTime :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_model_X_computers_prove_number_of_model_X_computers_l4176_417607


namespace NUMINAMATH_CALUDE_total_tickets_sold_l4176_417615

/-- Proves the total number of tickets sold given ticket prices, total receipts, and number of senior citizen tickets --/
theorem total_tickets_sold (adult_price senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) :
  adult_price = 25 →
  senior_price = 15 →
  total_receipts = 9745 →
  senior_tickets = 348 →
  ∃ (adult_tickets : ℕ), 
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    adult_tickets + senior_tickets = 529 :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l4176_417615


namespace NUMINAMATH_CALUDE_letter_digit_impossibility_l4176_417657

theorem letter_digit_impossibility :
  ¬ ∃ (f : Fin 7 → Fin 10),
    Function.Injective f ∧
    (f 0 * f 1 * 0 : ℕ) = (f 2 * f 3 * f 4 * f 5 * f 6 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_letter_digit_impossibility_l4176_417657


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l4176_417659

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g (x : ℚ) := c * x^3 - 8 * x^2 + d * x - 7
  (g 2 = -15) ∧ (g (-3) = -140) → c = 36/7 ∧ d = -109/7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l4176_417659


namespace NUMINAMATH_CALUDE_volume_is_304_l4176_417602

/-- The volume of the described set of points -/
def total_volume (central_box : ℝ × ℝ × ℝ) (extension : ℝ) : ℝ :=
  let (l, w, h) := central_box
  let box_volume := l * w * h
  let bounding_boxes_volume := 2 * (l * w + l * h + w * h) * extension
  let edge_prism_volume := 2 * (l + w + h) * extension * extension
  box_volume + bounding_boxes_volume + edge_prism_volume

/-- The theorem stating that the total volume is 304 cubic units -/
theorem volume_is_304 :
  total_volume (2, 3, 4) 2 = 304 := by sorry

end NUMINAMATH_CALUDE_volume_is_304_l4176_417602


namespace NUMINAMATH_CALUDE_product_equality_equal_S_not_imply_equal_Q_l4176_417682

-- Define a structure for a triangle divided by cevians
structure CevianTriangle where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  Q₁ : ℝ
  Q₂ : ℝ
  Q₃ : ℝ
  S_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0
  Q_positive : Q₁ > 0 ∧ Q₂ > 0 ∧ Q₃ > 0

-- Theorem 1: Product of S areas equals product of Q areas
theorem product_equality (t : CevianTriangle) : t.S₁ * t.S₂ * t.S₃ = t.Q₁ * t.Q₂ * t.Q₃ := by
  sorry

-- Theorem 2: Equal S areas do not necessarily imply equal Q areas
theorem equal_S_not_imply_equal_Q :
  ∃ t : CevianTriangle, (t.S₁ = t.S₂ ∧ t.S₂ = t.S₃) ∧ (t.Q₁ ≠ t.Q₂ ∨ t.Q₂ ≠ t.Q₃ ∨ t.Q₁ ≠ t.Q₃) := by
  sorry

end NUMINAMATH_CALUDE_product_equality_equal_S_not_imply_equal_Q_l4176_417682


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l4176_417679

/-- Two lines are parallel if their slopes are equal or if they are both vertical -/
def parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 = 0 ∧ a2 = 0) ∨ (a1 ≠ 0 ∧ a2 ≠ 0 ∧ b1 / a1 = b2 / a2)

/-- The statement of the problem -/
theorem parallel_lines_k_value (k : ℝ) :
  parallel (k - 3) (4 - k) 1 (k - 3) (-1) 1 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l4176_417679


namespace NUMINAMATH_CALUDE_fraction_simplification_l4176_417650

theorem fraction_simplification : 48 / (7 - 3/4) = 192/25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4176_417650


namespace NUMINAMATH_CALUDE_total_amount_spent_l4176_417631

theorem total_amount_spent (num_pens num_pencils : ℕ) (avg_price_pen avg_price_pencil : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  avg_price_pen = 20 →
  avg_price_pencil = 2 →
  (num_pens : ℚ) * avg_price_pen + (num_pencils : ℚ) * avg_price_pencil = 750 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_spent_l4176_417631


namespace NUMINAMATH_CALUDE_multiple_with_ones_and_zeros_multiple_with_only_ones_l4176_417640

def a (k : ℕ) : ℕ := (10^k - 1) / 9

theorem multiple_with_ones_and_zeros (n : ℤ) :
  ∃ k l : ℕ, k < l ∧ n ∣ (a l - a k) :=
sorry

theorem multiple_with_only_ones (n : ℤ) (h_odd : Odd n) (h_not_div_5 : ¬(5 ∣ n)) :
  ∃ d : ℕ, n ∣ (10^d - 1) :=
sorry

end NUMINAMATH_CALUDE_multiple_with_ones_and_zeros_multiple_with_only_ones_l4176_417640


namespace NUMINAMATH_CALUDE_acute_angle_inequalities_l4176_417684

theorem acute_angle_inequalities (α β : Real) 
  (h_α : 0 < α ∧ α < Real.pi / 2) 
  (h_β : 0 < β ∧ β < Real.pi / 2) : 
  (Real.sin (α + β) < Real.cos α + Real.cos β) ∧ 
  (Real.sin (α - β) < Real.cos α + Real.cos β) := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_inequalities_l4176_417684


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l4176_417630

theorem arithmetic_expression_equality : 6 + 18 / 3 - 3^2 - 4 * 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l4176_417630


namespace NUMINAMATH_CALUDE_variance_of_transformed_binomial_l4176_417608

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Variance of a linear transformation of a random variable -/
def variance_linear_transform (X : BinomialRV) (a b : ℝ) : ℝ := a^2 * variance X

/-- Main theorem: Variance of 4ξ + 3 for ξ ~ B(100, 0.2) -/
theorem variance_of_transformed_binomial :
  ∃ (ξ : BinomialRV), ξ.n = 100 ∧ ξ.p = 0.2 ∧ variance_linear_transform ξ 4 3 = 256 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_transformed_binomial_l4176_417608


namespace NUMINAMATH_CALUDE_polynomial_remainder_l4176_417694

def polynomial (x : ℝ) : ℝ := 8*x^4 + 4*x^3 - 9*x^2 + 16*x - 28

def divisor (x : ℝ) : ℝ := 4*x - 12

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 695 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4176_417694


namespace NUMINAMATH_CALUDE_sandys_initial_fish_count_l4176_417675

theorem sandys_initial_fish_count (initial_fish current_fish bought_fish : ℕ) : 
  current_fish = initial_fish + bought_fish →
  current_fish = 32 →
  bought_fish = 6 →
  initial_fish = 26 := by
sorry

end NUMINAMATH_CALUDE_sandys_initial_fish_count_l4176_417675


namespace NUMINAMATH_CALUDE_no_professors_are_student_council_members_l4176_417625

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Professor : U → Prop)
variable (StudentCouncilMember : U → Prop)
variable (Wise : U → Prop)

-- State the theorem
theorem no_professors_are_student_council_members
  (h1 : ∀ x, Professor x → Wise x)
  (h2 : ∀ x, StudentCouncilMember x → ¬Wise x) :
  ∀ x, Professor x → ¬StudentCouncilMember x :=
by sorry

end NUMINAMATH_CALUDE_no_professors_are_student_council_members_l4176_417625


namespace NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_l4176_417639

/-- The number of books Thabo owns -/
def total_books : ℕ := 280

/-- The number of paperback nonfiction books -/
def paperback_nonfiction (hardcover_nonfiction : ℕ) : ℕ := hardcover_nonfiction + 20

/-- The number of paperback fiction books -/
def paperback_fiction (hardcover_nonfiction : ℕ) : ℕ := 2 * (paperback_nonfiction hardcover_nonfiction)

/-- Theorem stating the number of hardcover nonfiction books Thabo owns -/
theorem thabo_hardcover_nonfiction :
  ∃ (hardcover_nonfiction : ℕ),
    hardcover_nonfiction + paperback_nonfiction hardcover_nonfiction + paperback_fiction hardcover_nonfiction = total_books ∧
    hardcover_nonfiction = 55 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_l4176_417639


namespace NUMINAMATH_CALUDE_student_sample_size_l4176_417665

theorem student_sample_size :
  ∀ (total juniors sophomores freshmen seniors : ℕ),
  juniors = (26 * total) / 100 →
  sophomores = (25 * total) / 100 →
  seniors = 160 →
  freshmen = sophomores + 32 →
  total = freshmen + sophomores + juniors + seniors →
  total = 800 := by
sorry

end NUMINAMATH_CALUDE_student_sample_size_l4176_417665


namespace NUMINAMATH_CALUDE_tully_age_proof_l4176_417637

def kate_current_age : ℕ := 29

theorem tully_age_proof (tully_age_year_ago : ℕ) : 
  (tully_age_year_ago + 4 = 2 * (kate_current_age + 3)) → tully_age_year_ago = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_tully_age_proof_l4176_417637


namespace NUMINAMATH_CALUDE_q_plus_r_at_one_eq_neg_47_l4176_417668

/-- The polynomial f(x) = 3x^5 + 4x^4 - 5x^3 + 2x^2 + x + 6 -/
def f (x : ℝ) : ℝ := 3*x^5 + 4*x^4 - 5*x^3 + 2*x^2 + x + 6

/-- The polynomial d(x) = x^3 + 2x^2 - x - 3 -/
def d (x : ℝ) : ℝ := x^3 + 2*x^2 - x - 3

/-- The existence of polynomials q and r satisfying the division algorithm -/
axiom exists_q_r : ∃ (q r : ℝ → ℝ), ∀ x, f x = q x * d x + r x

/-- The degree of r is less than the degree of d -/
axiom deg_r_lt_deg_d : sorry -- We can't easily express polynomial degrees in this simple setup

theorem q_plus_r_at_one_eq_neg_47 : 
  ∃ (q r : ℝ → ℝ), (∀ x, f x = q x * d x + r x) ∧ q 1 + r 1 = -47 := by
  sorry

end NUMINAMATH_CALUDE_q_plus_r_at_one_eq_neg_47_l4176_417668


namespace NUMINAMATH_CALUDE_intersection_A_B_l4176_417601

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}

-- Define set B
def B : Set ℝ := {-4, 1, 3, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4176_417601


namespace NUMINAMATH_CALUDE_estimated_students_above_average_l4176_417678

/-- Represents the time intervals for physical exercise --/
inductive TimeInterval
| LessThan30
| Between30And60
| Between60And90
| Between90And120

/-- Represents the data from the survey --/
structure SurveyData where
  sampleSize : Nat
  totalStudents : Nat
  mean : Nat
  studentsPerInterval : TimeInterval → Nat

/-- Theorem: Given the survey data, prove that the estimated number of students
    spending at least the average time on exercise is 130 --/
theorem estimated_students_above_average (data : SurveyData)
  (h1 : data.sampleSize = 20)
  (h2 : data.totalStudents = 200)
  (h3 : data.mean = 60)
  (h4 : data.studentsPerInterval TimeInterval.LessThan30 = 2)
  (h5 : data.studentsPerInterval TimeInterval.Between30And60 = 5)
  (h6 : data.studentsPerInterval TimeInterval.Between60And90 = 10)
  (h7 : data.studentsPerInterval TimeInterval.Between90And120 = 3) :
  (data.totalStudents * (data.studentsPerInterval TimeInterval.Between60And90 +
   data.studentsPerInterval TimeInterval.Between90And120) / data.sampleSize) = 130 := by
  sorry


end NUMINAMATH_CALUDE_estimated_students_above_average_l4176_417678


namespace NUMINAMATH_CALUDE_reflection_line_l4176_417690

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the triangle vertices
def P : Point := ⟨-2, 3⟩
def Q : Point := ⟨3, 7⟩
def R : Point := ⟨5, 1⟩

-- Define the reflected triangle vertices
def P' : Point := ⟨-6, 3⟩
def Q' : Point := ⟨-9, 7⟩
def R' : Point := ⟨-11, 1⟩

-- Define the line of reflection
def line_of_reflection (x : ℝ) : Prop :=
  (P.x + P'.x) / 2 = x ∧
  (Q.x + Q'.x) / 2 = x ∧
  (R.x + R'.x) / 2 = x

-- Theorem statement
theorem reflection_line : line_of_reflection (-3) := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_l4176_417690


namespace NUMINAMATH_CALUDE_percentage_grade_c_l4176_417695

def scores : List Nat := [49, 58, 65, 77, 84, 70, 88, 94, 55, 82, 60, 86, 68, 74, 99, 81, 73, 79, 53, 91]

def is_grade_c (score : Nat) : Bool :=
  78 ≤ score ∧ score ≤ 86

def count_grade_c (scores : List Nat) : Nat :=
  scores.filter is_grade_c |>.length

theorem percentage_grade_c : 
  (count_grade_c scores : Rat) / (scores.length : Rat) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_grade_c_l4176_417695


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4176_417660

theorem regular_polygon_sides (n : ℕ) (angle : ℝ) : 
  n > 0 → 
  angle > 0 → 
  angle < 180 → 
  (360 : ℝ) / n = angle → 
  angle = 20 → 
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4176_417660


namespace NUMINAMATH_CALUDE_farmer_land_usage_l4176_417652

/-- Represents the ratio of land used for beans, wheat, and corn -/
def land_ratio : Fin 3 → ℕ
  | 0 => 5  -- beans
  | 1 => 2  -- wheat
  | 2 => 4  -- corn
  | _ => 0  -- unreachable

/-- The total parts in the ratio -/
def total_parts : ℕ := (land_ratio 0) + (land_ratio 1) + (land_ratio 2)

/-- The number of acres used for corn -/
def corn_acres : ℕ := 376

theorem farmer_land_usage :
  let total_acres := (total_parts * corn_acres) / (land_ratio 2)
  total_acres = 1034 := by sorry

end NUMINAMATH_CALUDE_farmer_land_usage_l4176_417652


namespace NUMINAMATH_CALUDE_function_derivative_at_zero_l4176_417698

/-- Given a function f where f(x) = x^2 + 2x*f'(1), prove that f'(0) = -4 -/
theorem function_derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 0 = -4 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_at_zero_l4176_417698


namespace NUMINAMATH_CALUDE_larger_square_side_length_l4176_417676

theorem larger_square_side_length 
  (shaded_area unshaded_area : ℝ) 
  (h1 : shaded_area = 18)
  (h2 : unshaded_area = 18) : 
  ∃ (side_length : ℝ), side_length = 6 ∧ side_length^2 = shaded_area + unshaded_area :=
by
  sorry

end NUMINAMATH_CALUDE_larger_square_side_length_l4176_417676


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4176_417610

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 39 →
  a 2 + a 5 + a 8 = 33 →
  a 3 + a 6 + a 9 = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4176_417610


namespace NUMINAMATH_CALUDE_helen_raisin_cookies_l4176_417696

/-- The number of raisin cookies Helen baked yesterday -/
def raisin_cookies_yesterday : ℕ := 300

/-- The number of raisin cookies Helen baked the day before yesterday -/
def raisin_cookies_day_before : ℕ := 280

/-- The difference in raisin cookies between yesterday and the day before -/
def raisin_cookie_difference : ℕ := raisin_cookies_yesterday - raisin_cookies_day_before

theorem helen_raisin_cookies : raisin_cookie_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_helen_raisin_cookies_l4176_417696


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_third_l4176_417663

-- Define the repeating decimal 0.333...
def repeating_third : ℚ := 1/3

-- Theorem statement
theorem reciprocal_of_repeating_third :
  (repeating_third⁻¹ : ℚ) = 3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_third_l4176_417663


namespace NUMINAMATH_CALUDE_portia_school_size_l4176_417629

/-- The number of students in Portia's high school -/
def portia_students : ℕ := sorry

/-- The number of students in Lara's high school -/
def lara_students : ℕ := sorry

/-- Portia's high school has 4 times as many students as Lara's high school -/
axiom portia_lara_ratio : portia_students = 4 * lara_students

/-- The total number of students in both high schools is 3000 -/
axiom total_students : portia_students + lara_students = 3000

/-- Theorem: Portia's high school has 2400 students -/
theorem portia_school_size : portia_students = 2400 := by sorry

end NUMINAMATH_CALUDE_portia_school_size_l4176_417629


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l4176_417686

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that if the difference between the third terms is 5 times the
    difference between the second terms, then the sum of the common ratios is 5. -/
theorem geometric_sequence_ratio_sum (k p r : ℝ) (hk : k ≠ 0) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  k * p^2 - k * r^2 = 5 * (k * p - k * r) → p + r = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l4176_417686


namespace NUMINAMATH_CALUDE_total_bulbs_is_469_l4176_417662

/-- Represents the number of lights of each type -/
structure LightCounts where
  tiny : ℕ
  small : ℕ
  medium : ℕ
  large : ℕ
  extraLarge : ℕ

/-- Calculates the total number of bulbs needed -/
def totalBulbs (counts : LightCounts) : ℕ :=
  counts.tiny * 1 + counts.small * 2 + counts.medium * 3 + counts.large * 4 + counts.extraLarge * 5

theorem total_bulbs_is_469 (counts : LightCounts) :
  counts.large = 2 * counts.medium →
  counts.small = (5 * counts.medium) / 4 →
  counts.extraLarge = counts.small - counts.tiny →
  4 * counts.tiny = 3 * counts.medium →
  2 * counts.small + 3 * counts.medium = 4 * counts.large + 5 * counts.extraLarge →
  counts.extraLarge = 14 →
  totalBulbs counts = 469 := by
  sorry

#eval totalBulbs { tiny := 21, small := 35, medium := 28, large := 56, extraLarge := 14 }

end NUMINAMATH_CALUDE_total_bulbs_is_469_l4176_417662


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_l4176_417648

-- Problem 1
theorem problem_1 : (1) - 27 + (-32) + (-8) + 27 = -40 := by sorry

-- Problem 2
theorem problem_2 : (2) * (-5) + |(-3)| = -2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℤ) (h1 : -x = 3) (h2 : |y| = 5) : 
  x + y = 2 ∨ x + y = -8 := by sorry

-- Problem 4
theorem problem_4 : (-1 - 1/2) + (1 + 1/4) + (-2 - 1/2) - (-3 - 1/4) - (1 + 1/4) = -3/4 := by sorry

-- Problem 5
theorem problem_5 (a b : ℝ) (h : |a - 4| + |b + 5| = 0) : a - b = 9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_l4176_417648


namespace NUMINAMATH_CALUDE_apples_in_market_l4176_417674

theorem apples_in_market (apples oranges : ℕ) : 
  apples = oranges + 27 →
  apples + oranges = 301 →
  apples = 164 := by
sorry

end NUMINAMATH_CALUDE_apples_in_market_l4176_417674


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l4176_417600

/-- Proves that the speed of a boat in still water is 20 km/hr -/
theorem boat_speed_in_still_water : 
  ∀ (x : ℝ), 
    (5 : ℝ) = 5 → -- Rate of current is 5 km/hr
    ((x + 5) * (21 / 60) = (35 / 4 : ℝ)) → -- Distance travelled downstream in 21 minutes is 8.75 km
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l4176_417600


namespace NUMINAMATH_CALUDE_sum_on_real_axis_l4176_417656

theorem sum_on_real_axis (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := 3 + a * I
  (z₁ + z₂).im = 0 → a = -1 := by sorry

end NUMINAMATH_CALUDE_sum_on_real_axis_l4176_417656


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l4176_417647

theorem min_value_sum_of_squares (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l4176_417647


namespace NUMINAMATH_CALUDE_cos_315_degrees_l4176_417693

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l4176_417693


namespace NUMINAMATH_CALUDE_orange_pyramid_count_l4176_417624

/-- Calculates the number of oranges in a pyramid layer given its width and length -/
def layer_oranges (width : ℕ) (length : ℕ) : ℕ := width * length

/-- Calculates the total number of oranges in a pyramid stack -/
def total_oranges (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let base := layer_oranges base_width base_length
  let layer2 := layer_oranges (base_width - 1) (base_length - 1)
  let layer3 := layer_oranges (base_width - 2) (base_length - 2)
  let layer4 := layer_oranges (base_width - 3) (base_length - 3)
  let layer5 := layer_oranges (base_width - 4) (base_length - 4)
  let layer6 := layer_oranges (base_width - 5) (base_length - 5)
  let layer7 := layer_oranges (base_width - 6) (base_length - 6)
  base + layer2 + layer3 + layer4 + layer5 + layer6 + layer7 + 1

theorem orange_pyramid_count :
  total_oranges 7 10 = 225 := by
  sorry

end NUMINAMATH_CALUDE_orange_pyramid_count_l4176_417624


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l4176_417658

theorem cubic_equation_solution (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l4176_417658


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l4176_417699

-- Define the fraction
def fraction : ℚ := 987654321 / (2^30 * 5^5)

-- Define the function to calculate the minimum number of decimal digits
def min_decimal_digits (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem fraction_decimal_digits :
  min_decimal_digits fraction = 30 := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l4176_417699


namespace NUMINAMATH_CALUDE_parallel_intersection_lines_l4176_417622

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallelism relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection operation for a plane and a line
variable (intersect : Plane → Plane → Line)

-- Define the parallelism relation for lines
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_intersection_lines
  (m n : Line)
  (α β γ : Plane)
  (h1 : α ≠ β)
  (h2 : α ≠ γ)
  (h3 : β ≠ γ)
  (h4 : parallel_planes α β)
  (h5 : intersect α γ = m)
  (h6 : intersect β γ = n) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_parallel_intersection_lines_l4176_417622


namespace NUMINAMATH_CALUDE_floor_sqrt_48_squared_l4176_417635

theorem floor_sqrt_48_squared : ⌊Real.sqrt 48⌋^2 = 36 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_48_squared_l4176_417635


namespace NUMINAMATH_CALUDE_trip_time_calculation_l4176_417683

/-- Given a driving time and a traffic time that is twice the driving time, 
    calculate the total trip time. -/
def total_trip_time (driving_time : ℝ) : ℝ :=
  driving_time + 2 * driving_time

theorem trip_time_calculation :
  total_trip_time 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_calculation_l4176_417683


namespace NUMINAMATH_CALUDE_triangle_angle_A_l4176_417651

theorem triangle_angle_A (a b : ℝ) (B : ℝ) (hA : 0 < a) (hB : 0 < b) (hab : a > b) 
  (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 2) (hB : B = π / 4) :
  ∃ A : ℝ, (A = π / 3 ∨ A = 2 * π / 3) ∧ 
    Real.sin A = (a * Real.sin B) / b :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l4176_417651


namespace NUMINAMATH_CALUDE_kylie_stamps_l4176_417642

theorem kylie_stamps (kylie_stamps : ℕ) (nelly_stamps : ℕ) : 
  nelly_stamps = kylie_stamps + 44 →
  kylie_stamps + nelly_stamps = 112 →
  kylie_stamps = 34 := by
sorry

end NUMINAMATH_CALUDE_kylie_stamps_l4176_417642


namespace NUMINAMATH_CALUDE_division_problem_l4176_417626

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 725 →
  divisor = 36 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4176_417626


namespace NUMINAMATH_CALUDE_inequality_range_l4176_417609

theorem inequality_range (a : ℝ) : 
  (∀ (x θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l4176_417609


namespace NUMINAMATH_CALUDE_system_solution_l4176_417654

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = -9) ∧ (5 * x + 4 * y = 14) ∧ (x = 6/31) ∧ (y = 101/31) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4176_417654


namespace NUMINAMATH_CALUDE_ten_apples_left_l4176_417636

/-- The number of apples left after Frank's dog eats some -/
def apples_left (on_tree : ℕ) (on_ground : ℕ) (eaten : ℕ) : ℕ :=
  on_tree + (on_ground - eaten)

/-- Theorem: Given the initial conditions, there are 10 apples left -/
theorem ten_apples_left : apples_left 5 8 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_apples_left_l4176_417636


namespace NUMINAMATH_CALUDE_triangle_properties_l4176_417627

/-- Given a triangle ABC with angle B = 150°, side a = √3c, and side b = 2√7,
    prove that its area is √3 and if sin A + √3 sin C = √2/2, then C = 15° --/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  B = 150 * π / 180 →
  a = Real.sqrt 3 * c →
  b = 2 * Real.sqrt 7 →
  (1/2) * a * c * Real.sin B = Real.sqrt 3 ∧
  (Real.sin A + Real.sqrt 3 * Real.sin C = Real.sqrt 2 / 2 → C = 15 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l4176_417627


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l4176_417617

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1 ∧ c = b + 1 ∧ a * b * c = 990) → a + b + c = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l4176_417617


namespace NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l4176_417613

theorem square_sum_of_difference_and_product (a b : ℝ) 
  (h1 : a - b = 6) 
  (h2 : a * b = 6) : 
  a^2 + b^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l4176_417613


namespace NUMINAMATH_CALUDE_stars_count_theorem_l4176_417614

theorem stars_count_theorem (east : ℕ) (west_percent : ℕ) : 
  east = 120 → west_percent = 473 → 
  east + (east * (west_percent : ℚ) / 100).ceil = 688 := by
sorry

end NUMINAMATH_CALUDE_stars_count_theorem_l4176_417614


namespace NUMINAMATH_CALUDE_bridge_crossing_time_l4176_417605

/-- Proves that a man walking at 10 km/hr takes 15 minutes to cross a 2500-meter bridge -/
theorem bridge_crossing_time :
  let walking_speed : ℝ := 10  -- km/hr
  let bridge_length : ℝ := 2.5  -- km (2500 meters)
  let crossing_time : ℝ := bridge_length / walking_speed * 60  -- minutes
  crossing_time = 15 := by sorry

end NUMINAMATH_CALUDE_bridge_crossing_time_l4176_417605


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l4176_417606

theorem triangle_area_inequality (a b c α β γ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α > 0) (hβ : β > 0) (hγ : γ > 0)
  (hα_def : α = 2 * Real.sqrt (b * c))
  (hβ_def : β = 2 * Real.sqrt (c * a))
  (hγ_def : γ = 2 * Real.sqrt (a * b)) :
  a / α + b / β + c / γ ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l4176_417606


namespace NUMINAMATH_CALUDE_complex_absolute_value_l4176_417633

theorem complex_absolute_value (z : ℂ) (h : (3 - I) / (z - 3*I) = 1 + I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l4176_417633


namespace NUMINAMATH_CALUDE_two_workers_better_l4176_417687

/-- Represents the number of production lines -/
def num_lines : ℕ := 3

/-- Represents the probability of failure for each production line -/
def failure_prob : ℚ := 1/3

/-- Represents the monthly salary of each maintenance worker -/
def worker_salary : ℕ := 10000

/-- Represents the monthly profit of a production line with no failure -/
def profit_no_failure : ℕ := 120000

/-- Represents the monthly profit of a production line with failure and repair -/
def profit_with_repair : ℕ := 80000

/-- Represents the monthly profit of a production line with failure and no repair -/
def profit_no_repair : ℕ := 0

/-- Calculates the expected profit with a given number of maintenance workers -/
def expected_profit (num_workers : ℕ) : ℚ :=
  sorry

/-- Theorem stating that the expected profit with 2 workers is greater than with 1 worker -/
theorem two_workers_better :
  expected_profit 2 > expected_profit 1 := by
  sorry

end NUMINAMATH_CALUDE_two_workers_better_l4176_417687


namespace NUMINAMATH_CALUDE_remainder_3_power_2000_mod_17_l4176_417697

theorem remainder_3_power_2000_mod_17 :
  (3 : ℤ)^2000 ≡ 1 [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_power_2000_mod_17_l4176_417697


namespace NUMINAMATH_CALUDE_shelf_books_count_l4176_417670

/-- The number of books on a shelf after adding more books -/
def total_books (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of books on the shelf is 48 -/
theorem shelf_books_count : total_books 38 10 = 48 := by
  sorry

end NUMINAMATH_CALUDE_shelf_books_count_l4176_417670


namespace NUMINAMATH_CALUDE_square_root_of_difference_l4176_417634

theorem square_root_of_difference (n : ℕ+) :
  Real.sqrt ((10^(2*n.val) - 1)/9 - 2*(10^n.val - 1)/9) = (10^n.val - 1)/3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_difference_l4176_417634


namespace NUMINAMATH_CALUDE_broccoli_production_increase_l4176_417645

theorem broccoli_production_increase :
  ∀ (last_year_side : ℕ) (this_year_side : ℕ),
    last_year_side = 50 →
    this_year_side = 51 →
    this_year_side * this_year_side - last_year_side * last_year_side = 101 :=
by sorry

end NUMINAMATH_CALUDE_broccoli_production_increase_l4176_417645


namespace NUMINAMATH_CALUDE_candy_problem_l4176_417685

/-- Represents a set of candies with three types: hard, chocolate, and gummy -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The total number of candies in a set -/
def total (s : CandySet) : ℕ := s.hard + s.chocolate + s.gummy

theorem candy_problem (s1 s2 s3 : CandySet) 
  (h1 : s1.hard + s2.hard + s3.hard = s1.chocolate + s2.chocolate + s3.chocolate)
  (h2 : s1.hard + s2.hard + s3.hard = s1.gummy + s2.gummy + s3.gummy)
  (h3 : s1.chocolate = s1.gummy)
  (h4 : s1.hard = s1.chocolate + 7)
  (h5 : s2.hard = s2.chocolate)
  (h6 : s2.gummy = s2.hard - 15)
  (h7 : s3.hard = 0) : 
  total s3 = 29 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l4176_417685


namespace NUMINAMATH_CALUDE_fayes_rows_l4176_417666

theorem fayes_rows (pencils_per_row : ℕ) (crayons_per_row : ℕ) (total_items : ℕ) : 
  pencils_per_row = 31 →
  crayons_per_row = 27 →
  total_items = 638 →
  total_items / (pencils_per_row + crayons_per_row) = 11 := by
sorry

end NUMINAMATH_CALUDE_fayes_rows_l4176_417666


namespace NUMINAMATH_CALUDE_inequality_proof_l4176_417612

theorem inequality_proof (x b a : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) :
  x^2 > a*b ∧ a*b > a^2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4176_417612


namespace NUMINAMATH_CALUDE_total_popsicles_l4176_417604

theorem total_popsicles (grape : ℕ) (cherry : ℕ) (banana : ℕ) 
  (h1 : grape = 2) (h2 : cherry = 13) (h3 : banana = 2) : 
  grape + cherry + banana = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_popsicles_l4176_417604


namespace NUMINAMATH_CALUDE_quadratic_system_solution_l4176_417691

theorem quadratic_system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℚ),
    (x₁ = 2/9 ∧ y₁ = 35/117) ∧
    (x₂ = -1 ∧ y₂ = -5/26) ∧
    (∀ x y : ℚ, 9*x^2 + 8*x - 2 = 0 ∧ 27*x^2 + 26*y + 8*x - 14 = 0 →
      (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_system_solution_l4176_417691


namespace NUMINAMATH_CALUDE_binomial_coefficient_floor_divisibility_l4176_417628

theorem binomial_coefficient_floor_divisibility (p n : ℕ) 
  (hp : Nat.Prime p) (hn : n ≥ p) : 
  (Nat.choose n p - n / p) % p = 0 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_floor_divisibility_l4176_417628


namespace NUMINAMATH_CALUDE_total_pieces_four_row_triangle_l4176_417681

/-- Calculates the sum of the first n multiples of 3 -/
def sum_multiples_of_three (n : ℕ) : ℕ := 
  3 * n * (n + 1) / 2

/-- Calculates the sum of the first n even numbers -/
def sum_even_numbers (n : ℕ) : ℕ := 
  n * (n + 1)

/-- Represents the number of rows in the triangle configuration -/
def num_rows : ℕ := 4

/-- Theorem: The total number of pieces in a four-row triangle configuration is 60 -/
theorem total_pieces_four_row_triangle : 
  sum_multiples_of_three num_rows + sum_even_numbers (num_rows + 1) = 60 := by
  sorry

#eval sum_multiples_of_three num_rows + sum_even_numbers (num_rows + 1)

end NUMINAMATH_CALUDE_total_pieces_four_row_triangle_l4176_417681


namespace NUMINAMATH_CALUDE_pan_division_theorem_main_theorem_l4176_417618

/-- Represents the dimensions of a rectangular pan --/
structure PanDimensions where
  length : ℕ
  width : ℕ

/-- Represents the side length of a square piece of cake --/
def PieceSize : ℕ := 3

/-- Calculates the number of square pieces that can be cut from a rectangular pan --/
def numberOfPieces (pan : PanDimensions) (pieceSize : ℕ) : ℕ :=
  (pan.length * pan.width) / (pieceSize * pieceSize)

/-- Theorem stating that a 30x24 inch pan can be divided into 80 3-inch square pieces --/
theorem pan_division_theorem (pan : PanDimensions) (h1 : pan.length = 30) (h2 : pan.width = 24) :
  numberOfPieces pan PieceSize = 80 := by
  sorry

/-- Main theorem to be proved --/
theorem main_theorem : ∃ (pan : PanDimensions), 
  pan.length = 30 ∧ pan.width = 24 ∧ numberOfPieces pan PieceSize = 80 := by
  sorry

end NUMINAMATH_CALUDE_pan_division_theorem_main_theorem_l4176_417618
