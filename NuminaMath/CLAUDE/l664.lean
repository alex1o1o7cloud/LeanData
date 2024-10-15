import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_of_AB_CD_l664_66401

def A : ℝ × ℝ × ℝ := (3, -2, 5)
def B : ℝ × ℝ × ℝ := (13, -12, 10)
def C : ℝ × ℝ × ℝ := (-2, 5, -8)
def D : ℝ × ℝ × ℝ := (3, -1, 12)

def line_intersection (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem intersection_point_of_AB_CD :
  line_intersection A B C D = (-1/11, 1/11, 15/11) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_AB_CD_l664_66401


namespace NUMINAMATH_CALUDE_bookstore_sales_percentage_l664_66405

theorem bookstore_sales_percentage (book_sales magazine_sales other_sales : ℝ) :
  book_sales = 45 →
  magazine_sales = 25 →
  book_sales + magazine_sales + other_sales = 100 →
  other_sales = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_percentage_l664_66405


namespace NUMINAMATH_CALUDE_fuel_cost_per_refill_l664_66412

/-- 
Given the total fuel cost and number of refills, 
calculate the cost of one refilling.
-/
theorem fuel_cost_per_refill 
  (total_cost : ℕ) 
  (num_refills : ℕ) 
  (h1 : total_cost = 63)
  (h2 : num_refills = 3)
  : total_cost / num_refills = 21 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_per_refill_l664_66412


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l664_66457

/-- Given an arithmetic sequence {aₙ} with common difference d and a₃₀, find a₁ -/
theorem arithmetic_sequence_first_term
  (a : ℕ → ℚ)  -- The arithmetic sequence
  (d : ℚ)      -- Common difference
  (h1 : d = 3/4)
  (h2 : a 30 = 63/4)  -- a₃₀ = 15 3/4 = 63/4
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  : a 1 = -6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l664_66457


namespace NUMINAMATH_CALUDE_smallest_890_multiple_of_18_l664_66461

def is_digit_890 (d : ℕ) : Prop := d = 8 ∨ d = 9 ∨ d = 0

def all_digits_890 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_digit_890 d

theorem smallest_890_multiple_of_18 :
  ∃! m : ℕ, m > 0 ∧ m % 18 = 0 ∧ all_digits_890 m ∧
  ∀ n : ℕ, n > 0 → n % 18 = 0 → all_digits_890 n → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_890_multiple_of_18_l664_66461


namespace NUMINAMATH_CALUDE_min_value_sqrt_inverse_equality_condition_l664_66477

theorem min_value_sqrt_inverse (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x^2 ≥ 4 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x^2 = 4 * Real.sqrt 2 ↔ x = 2^(4/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_inverse_equality_condition_l664_66477


namespace NUMINAMATH_CALUDE_unique_function_solution_l664_66497

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) → 
  (∀ x : ℝ, f x = x) := by
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l664_66497


namespace NUMINAMATH_CALUDE_triangle_side_values_l664_66462

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_values :
  ∀ y : ℕ+, 
    (is_valid_triangle 8 11 (y.val ^ 2)) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l664_66462


namespace NUMINAMATH_CALUDE_evaluate_expression_power_sum_given_equation_l664_66455

-- Problem 1
theorem evaluate_expression (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5*y) * (-x - 5*y) - (-x + 5*y)^2 = -5.5 := by sorry

-- Problem 2
theorem power_sum_given_equation (a b : ℝ) (h : a^2 - 2*a + b^2 + 4*b + 5 = 0) :
  (a + b)^2013 = -1 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_power_sum_given_equation_l664_66455


namespace NUMINAMATH_CALUDE_cyclic_fraction_sum_l664_66446

theorem cyclic_fraction_sum (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a / b + b / c + c / a = 100) : 
  b / a + c / b + a / c = -103 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fraction_sum_l664_66446


namespace NUMINAMATH_CALUDE_course_selection_count_l664_66422

/-- The number of courses available -/
def total_courses : ℕ := 7

/-- The number of courses each student must choose -/
def courses_to_choose : ℕ := 4

/-- The number of special courses (A and B) that cannot be chosen together -/
def special_courses : ℕ := 2

/-- The number of different course selection schemes -/
def selection_schemes : ℕ := Nat.choose total_courses courses_to_choose - 
  (Nat.choose special_courses special_courses * Nat.choose (total_courses - special_courses) (courses_to_choose - special_courses))

theorem course_selection_count : selection_schemes = 25 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_count_l664_66422


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_three_digit_multiples_of_8_l664_66453

theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  let first := 104  -- First three-digit multiple of 8
  let last := 992   -- Last three-digit multiple of 8
  let step := 8     -- Difference between consecutive multiples
  let count := (last - first) / step + 1  -- Number of terms in the sequence
  let sum := count * (first + last) / 2   -- Sum of arithmetic sequence
  sum / count = 548 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_three_digit_multiples_of_8_l664_66453


namespace NUMINAMATH_CALUDE_converse_proposition_l664_66476

theorem converse_proposition :
  (∀ x y : ℝ, (x ≤ 2 ∨ y ≤ 2) → x + y ≤ 4) ↔
  (¬∀ x y : ℝ, (x > 2 ∧ y > 2) → x + y > 4) :=
by sorry

end NUMINAMATH_CALUDE_converse_proposition_l664_66476


namespace NUMINAMATH_CALUDE_janet_horses_count_l664_66468

def fertilizer_per_horse_per_day : ℕ := 5
def total_acres : ℕ := 20
def fertilizer_per_acre : ℕ := 400
def acres_fertilized_per_day : ℕ := 4
def days_to_fertilize : ℕ := 25

def janet_horses : ℕ := 64

theorem janet_horses_count : janet_horses = 
  (total_acres * fertilizer_per_acre) / 
  (fertilizer_per_horse_per_day * days_to_fertilize) := by
  sorry

end NUMINAMATH_CALUDE_janet_horses_count_l664_66468


namespace NUMINAMATH_CALUDE_union_equality_implies_m_values_l664_66491

def A : Set ℝ := {2, 3}
def B (m : ℝ) : Set ℝ := {x | m * x - 1 = 0}

theorem union_equality_implies_m_values (m : ℝ) :
  A ∪ B m = A → m = 1/2 ∨ m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_values_l664_66491


namespace NUMINAMATH_CALUDE_parabola_equation_l664_66420

/-- A parabola with vertex at the origin and directrix x = -2 has the equation y^2 = 8x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ y^2 = 8*x) ↔ 
  (∀ x y, p (x, y) → (x, y) ≠ (0, 0)) ∧ 
  (∀ x, x = -2 → ∀ y, ¬p (x, y)) := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l664_66420


namespace NUMINAMATH_CALUDE_ellipse_k_range_l664_66467

-- Define the equation of the ellipse
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) + y^2 / (9 - k) = 1

-- Define the range of k
def valid_k_range (k : ℝ) : Prop :=
  (1 < k ∧ k < 5) ∨ (5 < k ∧ k < 9)

-- Theorem stating the relationship between the ellipse equation and the range of k
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ valid_k_range k :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l664_66467


namespace NUMINAMATH_CALUDE_simplify_expression_l664_66459

theorem simplify_expression (w : ℝ) : 3*w + 6*w + 9*w + 12*w + 15*w + 18 = 45*w + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l664_66459


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l664_66425

theorem monic_quadratic_with_complex_root :
  ∃! (a b : ℝ), ∀ (x : ℂ), x^2 + a*x + b = 0 ↔ x = 2 - I ∨ x = 2 + I :=
by sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l664_66425


namespace NUMINAMATH_CALUDE_fraction_evaluation_l664_66448

theorem fraction_evaluation (a b : ℝ) (h : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^2 + b^2) = a^2 + b^2 - (2 * a^2 * b^2) / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l664_66448


namespace NUMINAMATH_CALUDE_exists_n_divisible_by_1987_l664_66411

theorem exists_n_divisible_by_1987 : ∃ n : ℕ, (1987 : ℕ) ∣ (n^n + (n+1)^n) := by
  sorry

end NUMINAMATH_CALUDE_exists_n_divisible_by_1987_l664_66411


namespace NUMINAMATH_CALUDE_product_equals_one_l664_66414

/-- A geometric sequence with a specific property -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  root_property : a 3 * a 15 = 1 ∧ a 3 + a 15 = 6

/-- The product of five consecutive terms equals 1 -/
theorem product_equals_one (seq : GeometricSequence) :
  seq.a 7 * seq.a 8 * seq.a 9 * seq.a 10 * seq.a 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l664_66414


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l664_66444

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l664_66444


namespace NUMINAMATH_CALUDE_helmet_costs_and_profit_l664_66439

/-- Represents the cost and sales information for helmets --/
structure HelmetData where
  costA3B4 : ℕ  -- Cost of 3 type A and 4 type B helmets
  costA6B2 : ℕ  -- Cost of 6 type A and 2 type B helmets
  basePrice : ℝ  -- Base selling price of type A helmet
  baseSales : ℕ  -- Number of helmets sold at base price
  priceIncrement : ℝ  -- Price increment
  salesDecrement : ℕ  -- Sales decrement per price increment

/-- Theorem about helmet costs and profit --/
theorem helmet_costs_and_profit (data : HelmetData)
  (h1 : data.costA3B4 = 288)
  (h2 : data.costA6B2 = 306)
  (h3 : data.basePrice = 50)
  (h4 : data.baseSales = 100)
  (h5 : data.priceIncrement = 5)
  (h6 : data.salesDecrement = 10) :
  ∃ (costA costB : ℕ) (profitFunc : ℝ → ℝ) (maxProfit : ℝ),
    costA = 36 ∧
    costB = 45 ∧
    (∀ x, 50 ≤ x ∧ x ≤ 100 → profitFunc x = -2 * x^2 + 272 * x - 7200) ∧
    maxProfit = 2048 := by
  sorry

end NUMINAMATH_CALUDE_helmet_costs_and_profit_l664_66439


namespace NUMINAMATH_CALUDE_log_equation_solution_l664_66473

theorem log_equation_solution (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∀ x > 1, 3 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 10 * (Real.log x)^2 / (Real.log a + Real.log b)) →
  b = a^((5 + Real.sqrt 10) / 3) ∨ b = a^((5 - Real.sqrt 10) / 3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l664_66473


namespace NUMINAMATH_CALUDE_diagonals_intersect_l664_66469

-- Define a regular 30-sided polygon
def RegularPolygon30 : Type := Unit

-- Define the sine function (simplified for this context)
noncomputable def sin (angle : ℝ) : ℝ := sorry

-- Define the cosine function (simplified for this context)
noncomputable def cos (angle : ℝ) : ℝ := sorry

-- Theorem statement
theorem diagonals_intersect (polygon : RegularPolygon30) : 
  (sin (6 * π / 180) * sin (18 * π / 180) * sin (84 * π / 180) = 
   sin (12 * π / 180) * sin (12 * π / 180) * sin (48 * π / 180)) ∧
  (sin (6 * π / 180) * sin (36 * π / 180) * sin (54 * π / 180) = 
   sin (30 * π / 180) * sin (12 * π / 180) * sin (12 * π / 180)) ∧
  (sin (36 * π / 180) * sin (18 * π / 180) * sin (6 * π / 180) = 
   cos (36 * π / 180) * cos (36 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_diagonals_intersect_l664_66469


namespace NUMINAMATH_CALUDE_orvin_max_balloons_l664_66443

/-- Represents the maximum number of balloons Orvin can buy given his budget and the sale conditions -/
def max_balloons (regular_price_budget : ℕ) (full_price_ratio : ℕ) (discount_ratio : ℕ) : ℕ :=
  let sets := (regular_price_budget * full_price_ratio) / (full_price_ratio + discount_ratio)
  sets * 2

/-- Proves that Orvin can buy at most 52 balloons given the specified conditions -/
theorem orvin_max_balloons :
  max_balloons 40 2 1 = 52 := by
  sorry

#eval max_balloons 40 2 1

end NUMINAMATH_CALUDE_orvin_max_balloons_l664_66443


namespace NUMINAMATH_CALUDE_min_value_theorem_l664_66407

def f (x : ℝ) := |x - 2| + |x + 1|

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_min : ∀ x, f x ≥ m + n) (h_exists : ∃ x, f x = m + n) :
  (∃ x, f x = 3) ∧ 
  (m^2 + n^2 ≥ 9/2) ∧
  (m^2 + n^2 = 9/2 ↔ m = 3/2 ∧ n = 3/2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l664_66407


namespace NUMINAMATH_CALUDE_inequality_range_l664_66433

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, |x + a| - |x + 1| < 2 * a) ↔ a > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l664_66433


namespace NUMINAMATH_CALUDE_second_last_digit_of_power_of_three_is_even_l664_66465

/-- The second-to-last digit of a natural number -/
def secondLastDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- A natural number is even if it's divisible by 2 -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem second_last_digit_of_power_of_three_is_even (n : ℕ) (h : n > 2) :
  isEven (secondLastDigit (3^n)) := by sorry

end NUMINAMATH_CALUDE_second_last_digit_of_power_of_three_is_even_l664_66465


namespace NUMINAMATH_CALUDE_andrey_stamps_problem_l664_66415

theorem andrey_stamps_problem :
  ∃! x : ℕ, x % 3 = 1 ∧ x % 5 = 3 ∧ x % 7 = 5 ∧ 150 < x ∧ x ≤ 300 ∧ x = 208 := by
  sorry

end NUMINAMATH_CALUDE_andrey_stamps_problem_l664_66415


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_l664_66408

theorem unique_solution_cube_difference (x y : ℤ) :
  (x + 2)^4 - x^4 = y^3 ↔ x = -1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_l664_66408


namespace NUMINAMATH_CALUDE_characterization_of_special_numbers_l664_66400

/-- A natural number n > 1 satisfies the given condition if and only if it's prime or a square of a prime -/
theorem characterization_of_special_numbers (n : ℕ) (h : n > 1) :
  (∀ d : ℕ, d > 1 → d ∣ n → (d - 1) ∣ (n - 1)) ↔ 
  (Nat.Prime n ∨ ∃ p : ℕ, Nat.Prime p ∧ n = p^2) := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_special_numbers_l664_66400


namespace NUMINAMATH_CALUDE_polynomial_factorization_l664_66464

theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  -(a - b) * (b - c) * (c - a) * (a^2 + a*b + b^2 + b*c + c^2 + a*c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l664_66464


namespace NUMINAMATH_CALUDE_ben_old_car_sale_amount_l664_66481

def old_car_cost : ℕ := 1900
def remaining_debt : ℕ := 2000

def new_car_cost : ℕ := 2 * old_car_cost

def amount_paid_off : ℕ := new_car_cost - remaining_debt

theorem ben_old_car_sale_amount : amount_paid_off = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ben_old_car_sale_amount_l664_66481


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l664_66484

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {(x, y) | y = x^2}

/-- The point Q -/
def Q : ℝ × ℝ := (20, 14)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y - Q.2 = m * (x - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m : ℝ) : Prop :=
  line_through_Q m ∩ P = ∅

/-- The theorem statement -/
theorem parabola_line_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 80 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l664_66484


namespace NUMINAMATH_CALUDE_convex_polygon_interior_angles_l664_66489

theorem convex_polygon_interior_angles (n : ℕ) :
  n > 2 →
  (∃ x : ℕ, (n - 2) * 180 - x = 2000) →
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_interior_angles_l664_66489


namespace NUMINAMATH_CALUDE_amoeba_survival_l664_66460

/-- Represents the state of an amoeba with pseudopods and nuclei -/
structure Amoeba where
  pseudopods : Int
  nuclei : Int

/-- Mutation function for an amoeba -/
def mutate (a : Amoeba) : Amoeba :=
  { pseudopods := 2 * a.pseudopods - a.nuclei,
    nuclei := 2 * a.nuclei - a.pseudopods }

/-- Predicate to check if an amoeba is alive -/
def isAlive (a : Amoeba) : Prop :=
  a.pseudopods ≥ 0 ∧ a.nuclei ≥ 0

/-- Theorem stating that only amoebas with equal initial pseudopods and nuclei survive indefinitely -/
theorem amoeba_survival (a : Amoeba) :
  (∀ n : ℕ, isAlive ((mutate^[n]) a)) ↔ a.pseudopods = a.nuclei :=
sorry

end NUMINAMATH_CALUDE_amoeba_survival_l664_66460


namespace NUMINAMATH_CALUDE_max_value_inequality_l664_66483

theorem max_value_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_condition : a + b + c + d ≤ 4) :
  (2*a^2 + a^2*b)^(1/4) + (2*b^2 + b^2*c)^(1/4) + 
  (2*c^2 + c^2*d)^(1/4) + (2*d^2 + d^2*a)^(1/4) ≤ 4 * 3^(1/4) :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l664_66483


namespace NUMINAMATH_CALUDE_connect_points_is_valid_l664_66431

-- Define a type for geometric points
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for geometric drawing operations
inductive DrawingOperation
  | DrawRay (start : Point) (length : ℝ)
  | ConnectPoints (a b : Point)
  | DrawMidpoint (a b : Point)
  | DrawDistance (a b : Point)

-- Define a predicate for valid drawing operations
def IsValidDrawingOperation : DrawingOperation → Prop
  | DrawingOperation.ConnectPoints _ _ => True
  | _ => False

-- Theorem statement
theorem connect_points_is_valid :
  ∀ (a b : Point), IsValidDrawingOperation (DrawingOperation.ConnectPoints a b) :=
by sorry

end NUMINAMATH_CALUDE_connect_points_is_valid_l664_66431


namespace NUMINAMATH_CALUDE_product_abcd_l664_66416

theorem product_abcd (a b c d : ℚ) 
  (eq1 : 4*a - 2*b + 3*c + 5*d = 22)
  (eq2 : 2*(d+c) = b - 2)
  (eq3 : 4*b - c = a + 1)
  (eq4 : c + 1 = 2*d) :
  a * b * c * d = -30751860 / 11338912 := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l664_66416


namespace NUMINAMATH_CALUDE_inequality_range_theorem_l664_66472

theorem inequality_range_theorem (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 5, |2 - x| + |x + 1| ≤ a) ↔ a ∈ Set.Ici 9 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_theorem_l664_66472


namespace NUMINAMATH_CALUDE_almond_butter_ratio_is_one_third_l664_66403

/-- The cost of a jar of peanut butter in dollars -/
def peanut_butter_cost : ℚ := 3

/-- The cost of a jar of almond butter in dollars -/
def almond_butter_cost : ℚ := 3 * peanut_butter_cost

/-- The additional cost per batch for almond butter cookies compared to peanut butter cookies -/
def additional_cost_per_batch : ℚ := 3

/-- The ratio of almond butter needed for a batch to the amount in a jar -/
def almond_butter_ratio : ℚ := additional_cost_per_batch / almond_butter_cost

theorem almond_butter_ratio_is_one_third :
  almond_butter_ratio = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_almond_butter_ratio_is_one_third_l664_66403


namespace NUMINAMATH_CALUDE_root_sum_squares_implies_h_abs_one_l664_66498

theorem root_sum_squares_implies_h_abs_one (h : ℝ) : 
  (∃ r s : ℝ, r^2 + 6*h*r + 8 = 0 ∧ s^2 + 6*h*s + 8 = 0 ∧ r^2 + s^2 = 20) → 
  |h| = 1 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_implies_h_abs_one_l664_66498


namespace NUMINAMATH_CALUDE_range_of_a_l664_66449

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a+3}

theorem range_of_a (a : ℝ) : A ∪ B a = A ↔ a ≤ -4 ∨ (2 ≤ a ∧ a ≤ 3) ∨ a > 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l664_66449


namespace NUMINAMATH_CALUDE_B_power_15_minus_4_power_14_l664_66434

def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 5; 0, 2]

theorem B_power_15_minus_4_power_14 :
  B^15 - 4 • B^14 = !![0, 5; 0, -2] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_4_power_14_l664_66434


namespace NUMINAMATH_CALUDE_circle_tangent_range_l664_66450

/-- The range of k values for which two tangent lines can be drawn from (1, 2) to the circle x^2 + y^2 + kx + 2y + k^2 - 15 = 0 -/
theorem circle_tangent_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + k*x + 2*y + k^2 - 15 = 0) ∧ 
  (∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    (∃ (x₁ y₁ x₂ y₂ : ℝ), 
      x₁^2 + y₁^2 + k*x₁ + 2*y₁ + k^2 - 15 = 0 ∧
      x₂^2 + y₂^2 + k*x₂ + 2*y₂ + k^2 - 15 = 0 ∧
      (y₁ - 2) = t₁ * (x₁ - 1) ∧
      (y₂ - 2) = t₂ * (x₂ - 1))) ↔ 
  (k > -8*Real.sqrt 3/3 ∧ k < -3) ∨ (k > 2 ∧ k < 8*Real.sqrt 3/3) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_range_l664_66450


namespace NUMINAMATH_CALUDE_prob_a_not_less_than_b_expected_tests_scheme_b_l664_66451

/-- Represents the two testing schemes -/
inductive TestScheme
| A
| B

/-- Represents the possible outcomes of a test -/
inductive TestResult
| Positive
| Negative

/-- The total number of swimmers -/
def totalSwimmers : ℕ := 5

/-- The number of swimmers who have taken stimulants -/
def stimulantUsers : ℕ := 1

/-- The number of swimmers tested in the first step of Scheme B -/
def schemeBFirstTest : ℕ := 3

/-- Function to calculate the probability that Scheme A requires no fewer tests than Scheme B -/
def probANotLessThanB : ℚ :=
  18/25

/-- Function to calculate the expected number of tests in Scheme B -/
def expectedTestsSchemeB : ℚ :=
  2.4

/-- Theorem stating the probability that Scheme A requires no fewer tests than Scheme B -/
theorem prob_a_not_less_than_b :
  probANotLessThanB = 18/25 := by sorry

/-- Theorem stating the expected number of tests in Scheme B -/
theorem expected_tests_scheme_b :
  expectedTestsSchemeB = 2.4 := by sorry

end NUMINAMATH_CALUDE_prob_a_not_less_than_b_expected_tests_scheme_b_l664_66451


namespace NUMINAMATH_CALUDE_matrix_power_2023_l664_66499

def A : Matrix (Fin 2) (Fin 2) ℕ :=
  !![1, 0;
     2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1,    0;
                4046, 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l664_66499


namespace NUMINAMATH_CALUDE_volunteer_selection_theorem_l664_66442

def male_students : ℕ := 5
def female_students : ℕ := 4
def total_volunteers : ℕ := 3
def schools : ℕ := 3

def selection_plans : ℕ := 420

theorem volunteer_selection_theorem :
  (male_students.choose 2 * female_students.choose 1 +
   male_students.choose 1 * female_students.choose 2) * schools.factorial = selection_plans :=
by sorry

end NUMINAMATH_CALUDE_volunteer_selection_theorem_l664_66442


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l664_66474

-- Define a function to represent mixed numbers
def mixed_number (whole : Int) (numerator : Int) (denominator : Int) : Rat :=
  whole + (numerator : Rat) / (denominator : Rat)

-- Problem 1
theorem problem_one : 
  mixed_number 28 5 7 + mixed_number (-25) (-1) 7 = mixed_number 3 4 7 := by
  sorry

-- Problem 2
theorem problem_two :
  mixed_number (-2022) (-2) 7 + mixed_number (-2023) (-4) 7 + (4046 : Rat) + (-1 : Rat) / 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l664_66474


namespace NUMINAMATH_CALUDE_positive_sum_product_iff_l664_66466

theorem positive_sum_product_iff (a b : ℝ) : (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_product_iff_l664_66466


namespace NUMINAMATH_CALUDE_function_root_implies_parameter_range_l664_66426

theorem function_root_implies_parameter_range (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ a^2 * x^2 - 2*a*x + 1 = 0) →
  a > 1 := by sorry

end NUMINAMATH_CALUDE_function_root_implies_parameter_range_l664_66426


namespace NUMINAMATH_CALUDE_janet_tile_savings_l664_66428

/-- Calculates the cost difference between two tile options for a given wall area and tile density -/
def tile_cost_difference (
  wall1_length wall1_width wall2_length wall2_width : ℝ)
  (tiles_per_sqft : ℝ)
  (turquoise_cost purple_cost : ℝ) : ℝ :=
  let total_area := wall1_length * wall1_width + wall2_length * wall2_width
  let total_tiles := total_area * tiles_per_sqft
  let cost_diff_per_tile := turquoise_cost - purple_cost
  total_tiles * cost_diff_per_tile

/-- The cost difference between turquoise and purple tiles for Janet's bathroom -/
theorem janet_tile_savings : 
  tile_cost_difference 5 8 7 8 4 13 11 = 768 := by
  sorry

end NUMINAMATH_CALUDE_janet_tile_savings_l664_66428


namespace NUMINAMATH_CALUDE_mike_total_spent_l664_66456

def trumpet_price : Float := 267.35
def song_book_price : Float := 12.95
def trumpet_case_price : Float := 74.50
def cleaning_kit_price : Float := 28.99
def valve_oils_price : Float := 18.75

theorem mike_total_spent : 
  trumpet_price + song_book_price + trumpet_case_price + cleaning_kit_price + valve_oils_price = 402.54 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_spent_l664_66456


namespace NUMINAMATH_CALUDE_prob_less_than_8_l664_66496

/-- The probability of scoring less than 8 in a single shot, given the probabilities of hitting the 10, 9, and 8 rings. -/
theorem prob_less_than_8 (p10 p9 p8 : ℝ) (h1 : p10 = 0.3) (h2 : p9 = 0.3) (h3 : p8 = 0.2) :
  1 - p10 - p9 - p8 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_8_l664_66496


namespace NUMINAMATH_CALUDE_equation_solutions_l664_66418

def equation (x y n : ℤ) : Prop :=
  x^3 - 3*x*y^2 + y^3 = n

theorem equation_solutions (n : ℤ) (hn : n > 0) :
  (∃ (x y : ℤ), equation x y n → 
    equation (y - x) (-x) n ∧ equation (-y) (x - y) n) ∧
  (n = 2891 → ¬∃ (x y : ℤ), equation x y n) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l664_66418


namespace NUMINAMATH_CALUDE_greatest_difference_of_valid_units_digits_l664_66480

/-- A function that checks if a number is divisible by 4 -/
def isDivisibleBy4 (n : ℕ) : Prop := n % 4 = 0

/-- The set of all possible three-digit numbers starting with 47 -/
def threeDigitNumbers : Set ℕ := {n : ℕ | 470 ≤ n ∧ n ≤ 479}

/-- The set of all three-digit numbers starting with 47 that are divisible by 4 -/
def divisibleNumbers : Set ℕ := {n ∈ threeDigitNumbers | isDivisibleBy4 n}

/-- The set of units digits of numbers in divisibleNumbers -/
def validUnitsDigits : Set ℕ := {x : ℕ | ∃ n ∈ divisibleNumbers, n % 10 = x}

theorem greatest_difference_of_valid_units_digits :
  ∃ (a b : ℕ), a ∈ validUnitsDigits ∧ b ∈ validUnitsDigits ∧ 
  ∀ (x y : ℕ), x ∈ validUnitsDigits → y ∈ validUnitsDigits → 
  (max a b - min a b : ℤ) ≥ (max x y - min x y) ∧
  (max a b - min a b : ℤ) = 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_of_valid_units_digits_l664_66480


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l664_66495

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l664_66495


namespace NUMINAMATH_CALUDE_complex_roots_count_l664_66479

theorem complex_roots_count : ∃! (S : Finset ℂ), 
  (∀ z ∈ S, Complex.abs z < 30 ∧ Complex.exp z = (z - 1) / (z + 1)) ∧ 
  Finset.card S = 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_count_l664_66479


namespace NUMINAMATH_CALUDE_ellipse_slope_product_ellipse_fixed_point_l664_66487

/-- Represents an ellipse with eccentricity √6/3 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : (a^2 - b^2) / a^2 = 2/3

/-- A point on the ellipse -/
def PointOnEllipse (e : Ellipse) := {p : ℝ × ℝ // p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1}

theorem ellipse_slope_product (e : Ellipse) (M A B : PointOnEllipse e) 
  (h_sym : A.val = (-B.val.1, -B.val.2)) :
  let k₁ := (A.val.2 - M.val.2) / (A.val.1 - M.val.1)
  let k₂ := (B.val.2 - M.val.2) / (B.val.1 - M.val.1)
  k₁ * k₂ = -1/3 := by sorry

theorem ellipse_fixed_point (e : Ellipse) (M A B : PointOnEllipse e)
  (h_M : M.val = (0, 1))
  (h_slopes : let k₁ := (A.val.2 - M.val.2) / (A.val.1 - M.val.1)
              let k₂ := (B.val.2 - M.val.2) / (B.val.1 - M.val.1)
              k₁ + k₂ = 3) :
  ∃ (k m : ℝ), A.val.2 = k * A.val.1 + m ∧ 
                B.val.2 = k * B.val.1 + m ∧ 
                -2/3 = k * (-2/3) + m ∧ 
                -1 = k * (-2/3) + m := by sorry

end NUMINAMATH_CALUDE_ellipse_slope_product_ellipse_fixed_point_l664_66487


namespace NUMINAMATH_CALUDE_tour_group_composition_l664_66427

/-- Given a group of 18 people where selecting one male (excluding two ineligible men) 
    and one female results in 64 different combinations, prove that there are 10 men 
    and 8 women in the group. -/
theorem tour_group_composition :
  ∀ (num_men : ℕ),
    (num_men - 2) * (18 - num_men) = 64 →
    num_men = 10 ∧ 18 - num_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_tour_group_composition_l664_66427


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l664_66423

theorem girls_to_boys_ratio (total_students : ℕ) (girls_present : ℕ) (boys_absent : ℕ)
  (h1 : total_students = 250)
  (h2 : girls_present = 140)
  (h3 : boys_absent = 40) :
  (girls_present : ℚ) / (total_students - girls_present - boys_absent) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l664_66423


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l664_66410

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x < 0 → x + 1 / x ≤ -2)) ↔ (∃ x : ℝ, x < 0 ∧ x + 1 / x > -2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l664_66410


namespace NUMINAMATH_CALUDE_lock_combination_solution_l664_66486

/-- Represents a digit in base 12 --/
def Digit12 := Fin 12

/-- Represents a mapping from letters to digits --/
def LetterMapping := Char → Digit12

/-- Converts a number in base 12 to base 10 --/
def toBase10 (x : ℕ) : ℕ := x

/-- Checks if all characters in a string are distinct --/
def allDistinct (s : String) : Prop := sorry

/-- Converts a string to a number using the given mapping --/
def stringToNumber (s : String) (m : LetterMapping) : ℕ := sorry

/-- The main theorem --/
theorem lock_combination_solution :
  ∃! (m : LetterMapping),
    (allDistinct "VENUSISNEAR") ∧
    (stringToNumber "VENUS" m + stringToNumber "IS" m + stringToNumber "NEAR" m =
     stringToNumber "SUN" m) ∧
    (toBase10 (stringToNumber "SUN" m) = 655) := by sorry

end NUMINAMATH_CALUDE_lock_combination_solution_l664_66486


namespace NUMINAMATH_CALUDE_problem_statement_l664_66441

theorem problem_statement (x : ℝ) (h : x + 1/x = 6) :
  (x - 3)^2 + 36/((x - 3)^2) = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l664_66441


namespace NUMINAMATH_CALUDE_car_to_stream_distance_l664_66413

/-- The distance from the car to the stream in miles -/
def distance_car_to_stream : ℝ := 0.2

/-- The total distance hiked in miles -/
def total_distance : ℝ := 0.7

/-- The distance from the stream to the meadow in miles -/
def distance_stream_to_meadow : ℝ := 0.4

/-- The distance from the meadow to the campsite in miles -/
def distance_meadow_to_campsite : ℝ := 0.1

theorem car_to_stream_distance :
  distance_car_to_stream = total_distance - distance_stream_to_meadow - distance_meadow_to_campsite :=
by sorry

end NUMINAMATH_CALUDE_car_to_stream_distance_l664_66413


namespace NUMINAMATH_CALUDE_log_inequality_l664_66493

theorem log_inequality (x : ℝ) (hx : x > 0) : Real.log (x + 1) ≥ x - (1/2) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l664_66493


namespace NUMINAMATH_CALUDE_polynomial_identity_l664_66404

theorem polynomial_identity (x : ℝ) : 
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l664_66404


namespace NUMINAMATH_CALUDE_same_grade_percentage_l664_66409

/-- Represents the grade distribution table -/
def gradeDistribution : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![4, 3, 2, 1],
    ![1, 6, 2, 0],
    ![3, 1, 3, 2],
    ![0, 1, 2, 2]]

/-- Total number of students -/
def totalStudents : ℕ := 36

/-- Sum of diagonal elements in the grade distribution table -/
def sameGradeCount : ℕ := (gradeDistribution 0 0) + (gradeDistribution 1 1) + (gradeDistribution 2 2) + (gradeDistribution 3 3)

/-- Theorem stating the percentage of students who received the same grade on both tests -/
theorem same_grade_percentage :
  (sameGradeCount : ℚ) / totalStudents = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l664_66409


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l664_66440

theorem not_divides_power_minus_one (n : ℕ) (h : n > 1) : ¬(n ∣ (2^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l664_66440


namespace NUMINAMATH_CALUDE_functions_intersect_at_negative_six_l664_66494

-- Define the two functions
def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := x - 5

-- State the theorem
theorem functions_intersect_at_negative_six : f (-6) = g (-6) := by
  sorry

end NUMINAMATH_CALUDE_functions_intersect_at_negative_six_l664_66494


namespace NUMINAMATH_CALUDE_problem_solution_l664_66402

def arithmetic_sum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem problem_solution : 
  ∃ x : ℚ, 
    let n : ℕ := (196 - 2) / 2 + 1
    let S : ℕ := arithmetic_sum 2 n 2
    (S + x) / (n + 1 : ℚ) = 50 * x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l664_66402


namespace NUMINAMATH_CALUDE_grocer_average_sale_l664_66458

/-- Given the sales figures for five months, prove that the average sale is 7800 --/
theorem grocer_average_sale
  (sale1 : ℕ) (sale2 : ℕ) (sale3 : ℕ) (sale4 : ℕ) (sale5 : ℕ)
  (h1 : sale1 = 5700)
  (h2 : sale2 = 8550)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 3850)
  (h5 : sale5 = 14045) :
  (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = 7800 := by
  sorry

#check grocer_average_sale

end NUMINAMATH_CALUDE_grocer_average_sale_l664_66458


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l664_66485

theorem wrong_mark_calculation (n : Nat) (initial_avg correct_avg : ℝ) (correct_mark : ℝ) :
  n = 10 ∧ 
  initial_avg = 100 ∧ 
  correct_avg = 95 ∧ 
  correct_mark = 10 →
  ∃ wrong_mark : ℝ,
    n * initial_avg = (n - 1) * correct_avg + wrong_mark ∧
    wrong_mark = 60 := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l664_66485


namespace NUMINAMATH_CALUDE_books_borrowed_by_lunch_correct_l664_66435

/-- Represents the number of books borrowed by lunchtime -/
def books_borrowed_by_lunch : ℕ := 50

/-- Represents the initial number of books on the shelf -/
def initial_books : ℕ := 100

/-- Represents the number of books added after lunch -/
def books_added : ℕ := 40

/-- Represents the number of books borrowed by evening -/
def books_borrowed_by_evening : ℕ := 30

/-- Represents the number of books remaining by evening -/
def books_remaining : ℕ := 60

/-- Proves that the number of books borrowed by lunchtime is correct -/
theorem books_borrowed_by_lunch_correct :
  initial_books - books_borrowed_by_lunch + books_added - books_borrowed_by_evening = books_remaining :=
by sorry


end NUMINAMATH_CALUDE_books_borrowed_by_lunch_correct_l664_66435


namespace NUMINAMATH_CALUDE_sqrt_18_minus_1_bounds_l664_66421

theorem sqrt_18_minus_1_bounds : 3 < Real.sqrt 18 - 1 ∧ Real.sqrt 18 - 1 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_1_bounds_l664_66421


namespace NUMINAMATH_CALUDE_books_ratio_l664_66470

/-- Given the number of books for Loris, Lamont, and Darryl, 
    prove that the ratio of Lamont's books to Darryl's books is 2:1 -/
theorem books_ratio (Loris Lamont Darryl : ℕ) : 
  Loris + 3 = Lamont →  -- Loris needs three more books to have the same as Lamont
  Darryl = 20 →  -- Darryl has 20 books
  Loris + Lamont + Darryl = 97 →  -- Total number of books is 97
  Lamont / Darryl = 2 := by
sorry

end NUMINAMATH_CALUDE_books_ratio_l664_66470


namespace NUMINAMATH_CALUDE_even_function_extension_l664_66471

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the function for x < 0
def f_neg (x : ℝ) : ℝ := x * (2 * x - 1)

-- Theorem statement
theorem even_function_extension 
  (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_neg : ∀ x, x < 0 → f x = f_neg x) : 
  ∀ x, x > 0 → f x = x * (2 * x + 1) := by
sorry


end NUMINAMATH_CALUDE_even_function_extension_l664_66471


namespace NUMINAMATH_CALUDE_big_fifteen_games_l664_66438

/-- Represents the Big Fifteen Basketball Conference -/
structure BigFifteenConference where
  numDivisions : Nat
  teamsPerDivision : Nat
  intraDivisionGames : Nat
  interDivisionGames : Nat
  nonConferenceGames : Nat

/-- Calculates the total number of games in the conference -/
def totalGames (conf : BigFifteenConference) : Nat :=
  let intraDivisionTotal := conf.numDivisions * (conf.teamsPerDivision.choose 2) * conf.intraDivisionGames
  let interDivisionTotal := conf.numDivisions * conf.teamsPerDivision * (conf.numDivisions - 1) * conf.teamsPerDivision / 2
  let nonConferenceTotal := conf.numDivisions * conf.teamsPerDivision * conf.nonConferenceGames
  intraDivisionTotal + interDivisionTotal + nonConferenceTotal

/-- Theorem stating that the total number of games in the Big Fifteen Conference is 270 -/
theorem big_fifteen_games :
  totalGames {
    numDivisions := 3,
    teamsPerDivision := 5,
    intraDivisionGames := 3,
    interDivisionGames := 1,
    nonConferenceGames := 2
  } = 270 := by sorry


end NUMINAMATH_CALUDE_big_fifteen_games_l664_66438


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l664_66430

/-- Represents the number of chocolate bars in the colossal box -/
def chocolate_bars_in_colossal_box : ℕ :=
  let sizable_boxes : ℕ := 350
  let small_boxes_per_sizable : ℕ := 49
  let chocolate_bars_per_small : ℕ := 75
  sizable_boxes * small_boxes_per_sizable * chocolate_bars_per_small

/-- Proves that the number of chocolate bars in the colossal box is 1,287,750 -/
theorem chocolate_bars_count : chocolate_bars_in_colossal_box = 1287750 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l664_66430


namespace NUMINAMATH_CALUDE_ellipse_equation_l664_66417

theorem ellipse_equation (a b : ℝ) (ha : a = 6) (hb : b = Real.sqrt 35) :
  (∃ x y : ℝ, x^2 / 36 + y^2 / 35 = 1) ∧ (∃ x y : ℝ, y^2 / 36 + x^2 / 35 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l664_66417


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_7_times_3_l664_66406

theorem binomial_coefficient_20_7_times_3 : 3 * (Nat.choose 20 7) = 16608 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_7_times_3_l664_66406


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l664_66436

/-- Given a triangle DEF with side lengths d, e, and f satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_of_triangle (d e f : ℝ) (h1 : d + 3*e + 3*f = d^2) (h2 : d + 3*e - 3*f = -4) :
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A ≤ 120 ∧ B ≤ 120 ∧ max A (max B C) = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l664_66436


namespace NUMINAMATH_CALUDE_sqrt_plus_arcsin_equals_pi_half_l664_66488

theorem sqrt_plus_arcsin_equals_pi_half (x : ℝ) :
  Real.sqrt (x * (x + 1)) + Real.arcsin (Real.sqrt (x^2 + x + 1)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_arcsin_equals_pi_half_l664_66488


namespace NUMINAMATH_CALUDE_two_members_absent_l664_66424

/-- Represents a trivia team with its properties and scoring. -/
structure TriviaTeam where
  totalMembers : ℕ
  pointsPerMember : ℕ
  totalPoints : ℕ

/-- Calculates the number of members who didn't show up for a trivia game. -/
def membersAbsent (team : TriviaTeam) : ℕ :=
  team.totalMembers - (team.totalPoints / team.pointsPerMember)

/-- Theorem stating that for the given trivia team, 2 members didn't show up. -/
theorem two_members_absent (team : TriviaTeam)
  (h1 : team.totalMembers = 5)
  (h2 : team.pointsPerMember = 6)
  (h3 : team.totalPoints = 18) :
  membersAbsent team = 2 := by
  sorry

#eval membersAbsent { totalMembers := 5, pointsPerMember := 6, totalPoints := 18 }

end NUMINAMATH_CALUDE_two_members_absent_l664_66424


namespace NUMINAMATH_CALUDE_possible_values_of_a_l664_66490

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 19*x^3) 
  (h3 : a - b = x) : 
  a = 3*x ∨ a = -2*x := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l664_66490


namespace NUMINAMATH_CALUDE_min_draw_theorem_l664_66482

/-- Represents the colors of the balls in the bag -/
inductive BallColor
  | Red
  | White
  | Yellow

/-- Represents the bag of balls -/
structure BallBag where
  red : Nat
  white : Nat
  yellow : Nat

/-- The minimum number of balls to draw to guarantee two different colors -/
def minDrawDifferentColors (bag : BallBag) : Nat :=
  bag.red + 1

/-- The minimum number of balls to draw to guarantee two yellow balls -/
def minDrawTwoYellow (bag : BallBag) : Nat :=
  bag.red + bag.white + 2

/-- Theorem stating the minimum number of balls to draw for different scenarios -/
theorem min_draw_theorem (bag : BallBag) 
  (h_red : bag.red = 10) 
  (h_white : bag.white = 10) 
  (h_yellow : bag.yellow = 10) : 
  minDrawDifferentColors bag = 11 ∧ minDrawTwoYellow bag = 22 := by
  sorry

#check min_draw_theorem

end NUMINAMATH_CALUDE_min_draw_theorem_l664_66482


namespace NUMINAMATH_CALUDE_leonards_age_l664_66437

theorem leonards_age (leonard nina jerome peter natasha : ℕ) : 
  nina = leonard + 4 →
  nina = jerome / 2 →
  peter = 2 * leonard →
  natasha = peter - 3 →
  leonard + nina + jerome + peter + natasha = 75 →
  leonard = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_leonards_age_l664_66437


namespace NUMINAMATH_CALUDE_miranda_savings_l664_66454

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

end NUMINAMATH_CALUDE_miranda_savings_l664_66454


namespace NUMINAMATH_CALUDE_inequality_equivalence_l664_66419

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem inequality_equivalence :
  ∀ x : ℝ, f (x^2 - 4) + f (3 * x) > 0 ↔ x > 1 ∨ x < -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l664_66419


namespace NUMINAMATH_CALUDE_equal_face_areas_not_imply_equal_volumes_l664_66452

/-- A tetrahedron with its volume and face areas -/
structure Tetrahedron where
  volume : ℝ
  face_areas : Fin 4 → ℝ

/-- Two tetrahedrons have equal face areas -/
def equal_face_areas (t1 t2 : Tetrahedron) : Prop :=
  ∀ i : Fin 4, t1.face_areas i = t2.face_areas i

/-- Theorem stating that equal face areas do not imply equal volumes -/
theorem equal_face_areas_not_imply_equal_volumes :
  ∃ (t1 t2 : Tetrahedron), equal_face_areas t1 t2 ∧ t1.volume ≠ t2.volume :=
sorry

end NUMINAMATH_CALUDE_equal_face_areas_not_imply_equal_volumes_l664_66452


namespace NUMINAMATH_CALUDE_ben_needs_14_eggs_l664_66478

/-- Represents the weekly egg requirements for a community -/
structure EggRequirements where
  saly : ℕ
  ben : ℕ
  ked : ℕ
  total_monthly : ℕ

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- Checks if the given egg requirements are valid -/
def is_valid_requirements (req : EggRequirements) : Prop :=
  req.saly = 10 ∧
  req.ked = req.ben / 2 ∧
  req.total_monthly = weeks_in_month * (req.saly + req.ben + req.ked)

/-- Theorem stating that Ben needs 14 eggs per week -/
theorem ben_needs_14_eggs (req : EggRequirements) 
  (h : is_valid_requirements req) (h_total : req.total_monthly = 124) : 
  req.ben = 14 := by
  sorry


end NUMINAMATH_CALUDE_ben_needs_14_eggs_l664_66478


namespace NUMINAMATH_CALUDE_club_size_after_five_years_l664_66463

def club_growth (initial_members : ℕ) (executives : ℕ) (years : ℕ) : ℕ :=
  let regular_members := initial_members - executives
  let final_regular_members := regular_members * (2 ^ years)
  final_regular_members + executives

theorem club_size_after_five_years :
  club_growth 18 6 5 = 390 := by sorry

end NUMINAMATH_CALUDE_club_size_after_five_years_l664_66463


namespace NUMINAMATH_CALUDE_bond_investment_problem_l664_66492

theorem bond_investment_problem (interest_income : ℝ) (rate1 rate2 : ℝ) (amount1 : ℝ) :
  interest_income = 1900 →
  rate1 = 0.0575 →
  rate2 = 0.0625 →
  amount1 = 20000 →
  ∃ amount2 : ℝ,
    amount1 * rate1 + amount2 * rate2 = interest_income ∧
    amount1 + amount2 = 32000 := by
  sorry

#check bond_investment_problem

end NUMINAMATH_CALUDE_bond_investment_problem_l664_66492


namespace NUMINAMATH_CALUDE_total_lemons_l664_66447

/-- Given the number of lemons for each person in terms of x, prove the total number of lemons. -/
theorem total_lemons (x : ℝ) :
  let L := x
  let J := x + 6
  let A := (4/3) * (x + 6)
  let E := (2/3) * (x + 6)
  let I := 2 * (2/3) * (x + 6)
  let N := (3/4) * x
  let O := (3/5) * (4/3) * (x + 6)
  L + J + A + E + I + N + O = (413/60) * x + 30.8 := by
sorry

end NUMINAMATH_CALUDE_total_lemons_l664_66447


namespace NUMINAMATH_CALUDE_cube_difference_l664_66429

theorem cube_difference (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l664_66429


namespace NUMINAMATH_CALUDE_max_value_x_minus_y_l664_66432

theorem max_value_x_minus_y :
  ∃ (max : ℝ), max = 2 * Real.sqrt 3 / 3 ∧
  (∀ x y : ℝ, 3 * (x^2 + y^2) = x + y → x - y ≤ max) ∧
  (∃ x y : ℝ, 3 * (x^2 + y^2) = x + y ∧ x - y = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_minus_y_l664_66432


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l664_66475

theorem quadratic_root_condition (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ x₂ > 1 ∧ 
    3 * x₁^2 + a * (a - 6) * x₁ - 3 = 0 ∧ 
    3 * x₂^2 + a * (a - 6) * x₂ - 3 = 0) ↔ 
  (0 < a ∧ a < 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l664_66475


namespace NUMINAMATH_CALUDE_green_ratio_l664_66445

theorem green_ratio (total : ℕ) (girls : ℕ) (yellow : ℕ) 
  (h_total : total = 30)
  (h_girls : girls = 18)
  (h_yellow : yellow = 9)
  (h_pink : girls / 3 = 6) :
  (total - (girls / 3 + yellow)) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_green_ratio_l664_66445
