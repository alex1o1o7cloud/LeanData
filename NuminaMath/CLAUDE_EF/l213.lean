import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ordering_l213_21315

-- Define the curves
noncomputable def C₁ (x : ℝ) : ℝ := 1 - 1/2 * x
noncomputable def C₂ (x : ℝ) : ℝ := 1 / (x + 1)
noncomputable def C₃ (x : ℝ) : ℝ := 1 - 1/2 * x^2

-- Define the areas
noncomputable def S₁ : ℝ := ∫ x in (0:ℝ)..(1:ℝ), C₁ x
noncomputable def S₂ : ℝ := ∫ x in (0:ℝ)..(1:ℝ), C₂ x
noncomputable def S₃ : ℝ := ∫ x in (0:ℝ)..(1:ℝ), C₃ x

-- Theorem statement
theorem area_ordering : S₂ < S₁ ∧ S₁ < S₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ordering_l213_21315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l213_21383

theorem floor_equation_solution (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊b * ↑n⌋ = b * ⌊a * ↑n⌋) ↔ 
  (a = 0 ∨ b = 0 ∨ a = b ∨ (Int.floor a = a ∧ Int.floor b = b)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l213_21383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_lettuce_is_four_l213_21319

/-- The average number of heads of lettuce purchased by each customer -/
noncomputable def average_lettuce_per_customer (customers : ℕ) (total_revenue : ℝ) (lettuce_price : ℝ) (tomatoes_per_customer : ℕ) (tomato_price : ℝ) : ℝ :=
  (total_revenue - (customers * tomatoes_per_customer * tomato_price)) / (customers * lettuce_price)

/-- Theorem stating that the average number of heads of lettuce purchased by each customer is 4 -/
theorem average_lettuce_is_four :
  average_lettuce_per_customer 500 2000 1 4 0.5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_lettuce_is_four_l213_21319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_first_row_sum_l213_21395

/-- Represents a grid with the given properties -/
structure Grid where
  rows : Nat
  cols : Nat
  max_num : Nat
  occurrences : Nat
  max_diff : Nat

/-- The specific grid from the problem -/
def problem_grid : Grid :=
  { rows := 9
  , cols := 2004
  , max_num := 2004
  , occurrences := 9
  , max_diff := 3 }

/-- The sum of numbers in the first row of the grid -/
def first_row_sum (g : Grid) : Nat := sorry

/-- Checks if a grid satisfies all the required conditions -/
def is_valid_grid (g : Grid) : Prop := sorry

theorem min_first_row_sum :
  ∀ (g : Grid), g = problem_grid → is_valid_grid g →
  first_row_sum g ≥ 2005004 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_first_row_sum_l213_21395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_value_l213_21356

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((Real.exp x - a)^2) / 16 + (x - a)^2

/-- The theorem statement -/
theorem function_minimum_implies_a_value (a : ℝ) :
  (∃ x : ℝ, f a x ≤ 1/17) → a = 1/17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_value_l213_21356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_2017_segments_l213_21374

-- Define the function f(x) = ⌊x⌋{x}
noncomputable def f (x : ℝ) : ℝ := Int.floor x * (x - Int.floor x)

-- Define the composition f(f(f(x)))
noncomputable def f_composed (x : ℝ) : ℝ := f (f (f x))

-- Define a function to count the number of line segments in the graph of f(f(f(x))) on [0,n]
noncomputable def count_line_segments (n : ℕ) : ℕ := sorry

-- State the theorem
theorem smallest_n_for_2017_segments : 
  (∀ m : ℕ, m < 23 → count_line_segments m < 2017) ∧ 
  count_line_segments 23 ≥ 2017 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_2017_segments_l213_21374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_range_l213_21384

/-- The function f(x) = x^2 + e^x - 1/2 for x < 0 -/
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x - 1/2

/-- The function g(x) = x^2 + ln(x + a) -/
noncomputable def g (a x : ℝ) : ℝ := x^2 + Real.log (x + a)

/-- The condition for symmetry about the y-axis -/
def symmetry_condition (a : ℝ) : Prop :=
  ∃ x, x < 0 ∧ f x = g a (-x)

/-- The theorem stating the range of a for which the symmetry condition holds -/
theorem symmetry_range :
  ∀ a, symmetry_condition a ↔ a < Real.sqrt (Real.exp 1) := by
  sorry

#check symmetry_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_range_l213_21384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_age_sum_l213_21394

/-- Represents the ages of three generations in a family -/
structure FamilyAges where
  grandfather_years : ℕ
  son_weeks : ℕ
  grandson_days : ℕ

/-- Calculates the total age of the family in years -/
def total_age (ages : FamilyAges) : ℕ :=
  ages.grandfather_years + (ages.son_weeks / 52) + (ages.grandson_days / 365)

/-- Theorem stating the total age of the family under given conditions -/
theorem family_age_sum (ages : FamilyAges) 
  (h1 : ages.grandfather_years = 72)
  (h2 : ages.grandson_days = ages.son_weeks)
  (h3 : ages.grandson_days / 30 = ages.grandfather_years) : 
  total_age ages = 120 := by
  sorry

#check family_age_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_age_sum_l213_21394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_max_min_equality_l213_21375

-- Define M and m functions as noncomputable
noncomputable def M (x y : ℝ) : ℝ := max x y
noncomputable def m (x y : ℝ) : ℝ := min x y

-- State the theorem
theorem nested_max_min_equality 
  (p q r s t : ℝ) 
  (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) :
  M (M p (m q r)) (m s (m p t)) = q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_max_min_equality_l213_21375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l213_21310

theorem coefficient_x_squared_in_expansion :
  (Polynomial.coeff ((Polynomial.X + 3 : Polynomial ℝ)^40) 2) = 780 * 3^38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l213_21310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l213_21340

def f (x : ℝ) := 2 * x - x^2

theorem range_of_f :
  ∃ (S : Set ℝ), S = {y | ∃ x, 1 < x ∧ x < 3 ∧ f x = y} → S = Set.Ioo (-3) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l213_21340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_work_days_calculation_l213_21325

/-- Represents the number of days it takes for a man to complete a work alone,
    given the time it takes for the man and his son together, and for the son alone. -/
noncomputable def man_work_days (together_days : ℝ) (son_days : ℝ) : ℝ :=
  (together_days * son_days) / (son_days - together_days)

/-- Theorem stating that if a man and his son can complete a work in 4 days together,
    and the son can do it in 6.67 days alone, then the man can do it in 10 days alone. -/
theorem man_work_days_calculation :
  man_work_days 4 (20 / 3) = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_work_days_calculation_l213_21325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_in_fourth_quadrant_l213_21343

def z : ℂ := 1 + 2*Complex.I

def is_in_fourth_quadrant (w : ℂ) : Prop :=
  w.re > 0 ∧ w.im < 0

theorem exists_n_in_fourth_quadrant :
  ∃ n : ℕ+, is_in_fourth_quadrant ((Complex.I^n.val) * z) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_in_fourth_quadrant_l213_21343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l213_21309

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(-x)

-- Define the inverse function
noncomputable def f_inv (a : ℝ) : ℝ → ℝ := Function.invFun (f a)

-- State the theorem
theorem inverse_function_point (a : ℝ) :
  (f_inv a (1/2) = 1) → a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l213_21309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2pi_minus_alpha_l213_21300

theorem cos_2pi_minus_alpha (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) (3 * π / 2))
  (h2 : Real.tan α = -12 / 5) : 
  Real.cos (2 * π - α) = -5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2pi_minus_alpha_l213_21300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_max_value_a_for_inequality_l213_21349

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for part I
theorem min_value_f_on_interval :
  ∀ x ∈ Set.Icc 1 3, f x ≥ 0 ∧ f 1 = 0 := by
  sorry

-- Theorem for part II
theorem max_value_a_for_inequality :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), 2 * f x ≥ -x^2 + a*x - 3) →
  a ≤ -2 + 1/(Real.exp 1) + 3*(Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_max_value_a_for_inequality_l213_21349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_upper_bound_l213_21380

-- Define the function f
noncomputable def f (x : ℝ) := Real.sqrt (1 - x) + Real.sqrt (1 + x)

-- Theorem for the range of f
theorem f_range :
  ∀ y, y ∈ Set.range f → Real.sqrt 2 ≤ y ∧ y ≤ 2 :=
sorry

-- Theorem for the upper bound of f on [0, 1]
theorem f_upper_bound :
  ∀ x, x ∈ Set.Icc 0 1 → f x ≤ 2 - (1/4) * x^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_upper_bound_l213_21380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_5_76_l213_21346

/-- The region in the plane defined by |x + 2y| + |2x - y| ≤ 6 -/
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 + 2*p.2| + |2*p.1 - p.2| ≤ 6}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := (MeasureTheory.volume region).toReal

/-- Theorem stating that the area of the region is 5.76 -/
theorem area_is_5_76 : area_of_region = 5.76 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_5_76_l213_21346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l213_21360

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the focal distance
def focal_distance : ℝ := 4

-- Define the maximum interior angle
noncomputable def max_interior_angle : ℝ := Real.pi / 2

-- Define the condition for points A and B on the line y = kx + m
def line_condition (k m x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m

-- Define the condition |OA + OB| = |OA - OB|
def vector_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂)^2 + (y₁ + y₂)^2 = (x₁ - x₂)^2 + (y₁ - y₂)^2

theorem ellipse_theorem :
  -- The equation of the ellipse is correct
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  -- The range for m is correct
  (∀ k m x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_condition k m x₁ y₁ x₂ y₂ ∧
    vector_condition x₁ y₁ x₂ y₂ →
    m > Real.sqrt 2 ∨ m < -Real.sqrt 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l213_21360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l213_21321

/-- Represents a rectangle -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  area : ℝ
  diagonal : ℝ

/-- Defines similarity between two rectangles -/
def Similar (R1 R2 : Rectangle) : Prop := 
  R1.side1 / R1.side2 = R2.side1 / R2.side2

/-- Given a rectangle R1 with sides 4 inches and area 16 square inches, 
    and a similar rectangle R2 with a diagonal of 10 inches, 
    the area of R2 is 50 square inches. -/
theorem area_of_similar_rectangle (R1 R2 : Rectangle) : 
  R1.side1 = 4 → 
  R1.area = 16 → 
  Similar R1 R2 → 
  R2.diagonal = 10 → 
  R2.area = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l213_21321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l213_21378

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (c = Real.sqrt 6) →
  (Real.cos B = 1 / 3) →
  (f (C / 2) = -1 / 4) →
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (b = 8 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l213_21378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_traveled_l213_21333

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem total_distance_traveled : 
  let p1 : point := (2, 2)
  let p2 : point := (5, 9)
  let p3 : point := (10, 12)
  distance p1 p2 + distance p2 p3 = Real.sqrt 58 + Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_traveled_l213_21333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l213_21328

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := -x + 1 / (2 * x)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ > f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l213_21328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_f_equal_l213_21350

/-- The function f(x) = 2x^2 - 4x + 5 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

/-- g is a function from ℝ to ℝ -/
noncomputable def g : ℝ → ℝ := sorry

/-- The given condition that g(f(2)) = 25 -/
axiom g_f_2 : g (f 2) = 25

/-- The theorem to be proved -/
theorem g_f_equal : g (f (-2)) = g (f 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_f_equal_l213_21350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_payment_for_2000_plus_x_l213_21373

def is_divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def payment (n : ℕ) : ℕ :=
  (if n % 1 = 0 then 1 else 0) +
  (if n % 3 = 0 then 3 else 0) +
  (if n % 5 = 0 then 5 else 0) +
  (if n % 7 = 0 then 7 else 0) +
  (if n % 9 = 0 then 9 else 0) +
  (if n % 11 = 0 then 11 else 0)

theorem max_payment_for_2000_plus_x :
  ∀ x : ℕ, x ≤ 99 → payment (2000 + x) ≤ 31 ∧
  ∃ y : ℕ, y ≤ 99 ∧ payment (2000 + y) = 31 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_payment_for_2000_plus_x_l213_21373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_on_unit_circle_l213_21354

theorem tan_alpha_on_unit_circle (x : ℝ) (α : ℝ) :
  x^2 + (Real.sqrt 3 / 2)^2 = 1 →
  (∃ (s : ℝ), s = 1 ∨ s = -1) ∧ x = s * (1/2) →
  Real.tan α = s * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_on_unit_circle_l213_21354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_distance_after_seven_days_l213_21398

-- Define the driving conditions
def total_distance : ℕ := 8205
def day1_speed : ℕ := 90
def day1_hours : ℕ := 10
def day2_speed : ℕ := 80
def day2_hours : ℕ := 7
def clear_day_speed : ℕ := 100
def foggy_day_speed : ℕ := 75
def daily_driving_hours : ℕ := 7  -- 8 hours with 1-hour break
def rest_stop_interval : ℕ := 1500

-- Define the theorem
theorem remaining_distance_after_seven_days : 
  (let distance_day1 := day1_speed * day1_hours
   let distance_day2 := day2_speed * day2_hours
   let distance_day4 := clear_day_speed * daily_driving_hours
   let distance_day5 := foggy_day_speed * daily_driving_hours
   let distance_day7 := foggy_day_speed * daily_driving_hours
   let total_distance_covered := distance_day1 + distance_day2 + distance_day4 + distance_day5 + distance_day7
   total_distance - total_distance_covered) = 4995 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_distance_after_seven_days_l213_21398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_square_roots_equals_251_54_l213_21347

/-- Represents a circle in the plane -/
structure Circle where
  radius : ℝ

/-- Represents a layer of circles -/
def Layer := List Circle

/-- Constructs a new circle between two given circles -/
noncomputable def constructCircle (c1 c2 : Circle) : Circle :=
  { radius := (c1.radius * c2.radius) / ((Real.sqrt c1.radius + Real.sqrt c2.radius) ^ 2) }

/-- Constructs the next layer given the previous layers -/
def nextLayer (prevLayers : List Layer) : Layer :=
  sorry

/-- Constructs all layers up to the given index -/
def constructLayers (n : ℕ) : List Layer :=
  sorry

/-- The set S of all circles in the first 6 layers -/
def S : List Circle :=
  (constructLayers 6).join

/-- The sum of 1/√(r(C)) for all circles C in S -/
noncomputable def sumInverseSquareRoots : ℝ :=
  (S.map (fun c => 1 / Real.sqrt c.radius)).sum

theorem sum_inverse_square_roots_equals_251_54 :
  sumInverseSquareRoots = 251 / 54 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_square_roots_equals_251_54_l213_21347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_maximized_l213_21389

/-- An arithmetic sequence with its first term and common difference -/
structure ArithmeticSequence where
  a1 : ℚ
  d : ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a1 + (n * (n - 1) / 2) * seq.d

/-- Theorem stating that S_n is maximized when n = 13 or n = 14 -/
theorem sum_maximized (seq : ArithmeticSequence) (h1 : seq.a1 > 0) (h2 : sum_n seq 9 = sum_n seq 18) :
  ∃ n : ℕ, (n = 13 ∨ n = 14) ∧ 
    ∀ m : ℕ, sum_n seq m ≤ sum_n seq n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_maximized_l213_21389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l213_21391

def box1 : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def box2 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

def is_product_even (a b : ℕ) : Bool := Even (a * b)

theorem probability_even_product :
  (Finset.filter (λ a => (Finset.filter (λ b => is_product_even a b) box2).card > 0) box1).card / 
  (box1.card * box2.card : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l213_21391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_solution_part2_solution_l213_21364

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions of the problem
def problem_conditions (t : Triangle) : Prop :=
  t.A = Real.pi/3

-- Define the first part of the problem
def part1 (t : Triangle) : Prop :=
  problem_conditions t ∧ t.B = 5*Real.pi/12 ∧ t.c = Real.sqrt 6

-- Define the second part of the problem
def part2 (t : Triangle) : Prop :=
  problem_conditions t ∧ t.a = Real.sqrt 7 ∧ t.c = 2

-- State the theorem for part 1
theorem part1_solution (t : Triangle) :
  part1 t → t.a = 3 :=
by sorry

-- State the theorem for part 2
theorem part2_solution (t : Triangle) :
  part2 t → t.b = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_solution_part2_solution_l213_21364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_lunks_for_apples_l213_21371

/-- Exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 5 / 4

/-- Exchange rate between kunks and apples -/
def kunk_to_apple_rate : ℚ := 10 / 7

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 12

/-- Calculates the number of lunks needed to buy a given number of apples -/
def lunks_needed (apples : ℕ) : ℕ :=
  (((apples : ℚ) / (kunk_to_apple_rate * lunk_to_kunk_rate)).ceil).toNat

/-- Theorem stating that 7 lunks are needed to buy 12 apples -/
theorem sufficient_lunks_for_apples :
  lunks_needed apples_to_buy = 7 := by
  sorry

#eval lunks_needed apples_to_buy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_lunks_for_apples_l213_21371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sum_divisibility_l213_21316

theorem quadratic_root_sum_divisibility
  (a b : ℤ) (n : ℕ) (u v : ℂ)
  (h1 : u^2 + a*u + b = 0)
  (h2 : v^2 + a*v + b = 0)
  (h3 : b ∣ a^2) :
  ∃ k : ℤ, (u^(2*n) + v^(2*n)) = k * (b^n : ℂ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sum_divisibility_l213_21316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_in_stream_l213_21361

/-- Proves that the distance traveled by a boat in a stream is 6 km given specific conditions -/
theorem boat_distance_in_stream (current_speed : ℝ) (boat_speed : ℝ) (total_time : ℝ) 
  (h1 : current_speed = 4)
  (h2 : boat_speed = 8)
  (h3 : total_time = 2) :
  (total_time * (boat_speed + current_speed) * (boat_speed - current_speed)) / 
  ((boat_speed + current_speed) + (boat_speed - current_speed)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_in_stream_l213_21361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l213_21323

/-- The distance between intersections of x = y³ and x + y² = 1 -/
theorem intersection_distance (a : ℝ) : 
  a^4 + a^3 + a^2 - 1 = 0 → 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ = y₁^3 ∧ x₁ + y₁^2 = 1 ∧
    x₂ = y₂^3 ∧ x₂ + y₂^2 = 1 ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * (a^6 + a^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l213_21323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_CEF_l213_21377

-- Define the parallelogram ABCD
structure Parallelogram :=
  (A B C D : EuclideanSpace ℝ (Fin 2))
  (parallelogram_property : (B - A) = (C - D) ∧ (D - A) = (C - B))

-- Define equilateral triangles
def is_equilateral_triangle (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖P - Q‖ = ‖Q - R‖ ∧ ‖Q - R‖ = ‖R - P‖

-- Define the theorem
theorem equilateral_CEF 
  (ABCD : Parallelogram) 
  (F : EuclideanSpace ℝ (Fin 2))
  (E : EuclideanSpace ℝ (Fin 2))
  (h1 : is_equilateral_triangle ABCD.A ABCD.B F)
  (h2 : is_equilateral_triangle ABCD.A ABCD.D E) :
  is_equilateral_triangle ABCD.C E F :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_CEF_l213_21377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_price_proof_l213_21338

/-- The exchange rate from Euro to USD -/
def exchange_rate : ℝ := 1.1

/-- The original price of the computer in the first store -/
def original_price : ℝ := 1153.47

/-- The discounted price in the first store -/
def first_store_price : ℝ := original_price * (1 - 0.06)

/-- The discounted price in the second store -/
def second_store_price : ℝ := 920

/-- The price difference between the two stores in USD -/
def price_difference : ℝ := 19

theorem computer_price_proof :
  |first_store_price - second_store_price * exchange_rate - price_difference| < 0.01 ∧
  |second_store_price / 0.95 * exchange_rate - original_price| < 1 := by
  sorry

#check computer_price_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_price_proof_l213_21338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juice_price_decrease_percentage_l213_21302

/-- Represents the dimensions of a juice box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

/-- Calculates the mass of juice in a box given its volume and density -/
noncomputable def juiceMass (volume density : ℝ) : ℝ := volume * density

/-- Calculates the price per gram of juice given the total price and mass -/
noncomputable def pricePerGram (price mass : ℝ) : ℝ := price / mass

/-- Calculates the percentage decrease between two values -/
noncomputable def percentageDecrease (oldValue newValue : ℝ) : ℝ :=
  (oldValue - newValue) / oldValue * 100

theorem juice_price_decrease_percentage :
  let smallBox := BoxDimensions.mk 5 10 20
  let largeBox := BoxDimensions.mk 6 10 20
  let smallVolume := boxVolume smallBox
  let largeVolume := boxVolume largeBox
  let smallMass := juiceMass smallVolume 1.1
  let largeMass := juiceMass largeVolume 1.2
  let price : ℝ := 1  -- Arbitrary price, as it cancels out in the calculation
  let smallPricePerGram := pricePerGram price smallMass
  let largePricePerGram := pricePerGram price largeMass
  let decrease := percentageDecrease smallPricePerGram largePricePerGram
  ∃ ε > 0, |decrease - 23.61| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juice_price_decrease_percentage_l213_21302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l213_21330

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function for a triangle
noncomputable def area (t : Triangle) : ℝ := 
  (1/2) * t.a * t.c * Real.sin t.B

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = (Real.sqrt 3 / 3) * t.c * Real.tan t.B)
  (h2 : t.b = 2) :
  t.B = π / 3 ∧ 
  (∀ (s : Triangle), s.b = 2 → area s ≤ Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l213_21330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_plus_area_eq_l213_21314

/-- A parallelogram with vertices at (1,3), (6,3), (4,0), and (-1,0) -/
structure Parallelogram where
  v1 : ℤ × ℤ := (1, 3)
  v2 : ℤ × ℤ := (6, 3)
  v3 : ℤ × ℤ := (4, 0)
  v4 : ℤ × ℤ := (-1, 0)

/-- Calculate the perimeter of the parallelogram -/
noncomputable def perimeter (p : Parallelogram) : ℝ :=
  let d1 := ((p.v2.1 - p.v1.1) ^ 2 + (p.v2.2 - p.v1.2) ^ 2 : ℝ).sqrt
  let d2 := ((p.v3.1 - p.v2.1) ^ 2 + (p.v3.2 - p.v2.2) ^ 2 : ℝ).sqrt
  2 * (d1 + d2)

/-- Calculate the area of the parallelogram -/
def area (p : Parallelogram) : ℝ :=
  let base := (p.v3.1 - p.v4.1).natAbs
  let height := (p.v1.2 - p.v4.2).natAbs
  (base * height : ℝ)

/-- The sum of perimeter and area is equal to 25 + 2√13 -/
theorem perimeter_plus_area_eq (p : Parallelogram) :
    perimeter p + area p = 25 + 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_plus_area_eq_l213_21314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_after_erasing_max_sum_proof_l213_21331

def initial_numbers : List Nat := List.range 11 |>.map (· + 7)

def is_divisible_into_equal_groups (numbers : List Nat) : Prop :=
  let sum := numbers.sum
  ∃ k : Nat, k > 1 ∧ k ≤ numbers.length ∧ sum % k = 0 ∧
    ∃ groups : List (List Nat), groups.length = k ∧
      groups.all (λ group => group.sum = sum / k) ∧
      groups.join.toFinset = numbers.toFinset

theorem max_sum_after_erasing (max_sum : Nat) : Prop :=
  max_sum = 121 ∧
  ∃ numbers : List Nat, numbers ⊆ initial_numbers ∧
    numbers.sum = max_sum ∧
    ¬is_divisible_into_equal_groups numbers ∧
    ∀ other_numbers : List Nat, other_numbers ⊆ initial_numbers →
      other_numbers.sum > max_sum →
      is_divisible_into_equal_groups other_numbers

theorem max_sum_proof : ∃ max_sum : Nat, max_sum_after_erasing max_sum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_after_erasing_max_sum_proof_l213_21331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l213_21385

theorem triangle_properties (a b c S : ℝ) (h1 : S = a^2 - (b - c)^2) (h2 : b + c = 8) :
  ∃ (A : ℝ), 
    (0 < A ∧ A < Real.pi) ∧  -- A is a valid angle measure
    (Real.cos A = 15/17) ∧  -- Part 1: cos A
    (S ≤ 64/17) ∧     -- Part 2: maximum value of S
    (∃ (b' c' : ℝ), b' + c' = 8 ∧ S = 64/17)  -- The maximum is achievable
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l213_21385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_truck_speed_comparison_l213_21301

noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

noncomputable def speed_ratio (speed1 : ℝ) (speed2 : ℝ) : ℝ := speed1 / speed2

theorem taxi_truck_speed_comparison :
  let truck_distance : ℝ := 2.1
  let truck_time : ℝ := 1
  let taxi_distance : ℝ := 10.5
  let taxi_time : ℝ := 4
  let truck_speed := speed truck_distance truck_time
  let taxi_speed := speed taxi_distance taxi_time
  speed_ratio taxi_speed truck_speed = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_truck_speed_comparison_l213_21301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_primes_less_than_125_l213_21303

theorem count_primes_less_than_125 : 
  (Finset.filter (fun n => Nat.Prime n ∧ n < 125) (Finset.range 125)).card = 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_primes_less_than_125_l213_21303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_conjugate_abs_l213_21372

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := (3 - i) * z = 5 * i

-- Theorem statement
theorem z_conjugate_abs (z : ℂ) (h : z_condition z) : Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_conjugate_abs_l213_21372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_interest_rate_theorem_l213_21317

/-- Calculates the annual interest rate given the face value, true discount, and time to maturity of a bill. -/
noncomputable def calculate_annual_interest_rate (face_value : ℝ) (true_discount : ℝ) (months_to_maturity : ℝ) : ℝ :=
  let present_value := face_value - true_discount
  (true_discount * 1200) / (present_value * months_to_maturity)

/-- Theorem stating that for a bill with face value 1764 Rs., true discount 189 Rs., and 9 months to maturity, the annual interest rate is approximately 16%. -/
theorem bill_interest_rate_theorem :
  let face_value : ℝ := 1764
  let true_discount : ℝ := 189
  let months_to_maturity : ℝ := 9
  let calculated_rate := calculate_annual_interest_rate face_value true_discount months_to_maturity
  abs (calculated_rate - 16) < 0.1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_interest_rate_theorem_l213_21317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_win_margin_l213_21341

theorem election_win_margin 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (winner_votes : ℕ) 
  (h1 : winner_percentage = 3/5) 
  (h2 : winner_votes = 864) 
  (h3 : winner_votes = (winner_percentage * ↑total_votes).floor) : 
  winner_votes - (total_votes - winner_votes) = 288 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_win_margin_l213_21341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l213_21336

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Ioo 0 4 :=
sorry

-- Additional lemmas to characterize the range
lemma f_positive (x : ℝ) : 0 < f x :=
sorry

lemma f_upper_bound (x : ℝ) : f x ≤ 4 :=
sorry

lemma f_approaches_zero : ∀ ε > 0, ∃ x : ℝ, 0 < f x ∧ f x < ε :=
sorry

lemma f_approaches_four : ∀ ε > 0, ∃ x : ℝ, |f x - 4| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l213_21336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l213_21306

-- Define the sequence b_n
def b : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => b (n + 1) + b n

-- Define the series
noncomputable def series_sum : ℚ := ∑' n, b n / 3^(n + 1)

-- Theorem statement
theorem fibonacci_series_sum : series_sum = 1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l213_21306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l213_21357

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A 250-meter long train traveling at 162 km/hr takes approximately 5.56 seconds to cross an electric pole -/
theorem train_crossing_pole_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_crossing_time 250 162 - 5.56| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l213_21357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_to_other_schools_l213_21362

def total_applicants : ℕ := 20000
def acceptance_rate : ℚ := 5 / 100
def attendance_rate : ℚ := 90 / 100
def students_attending : ℕ := 900

theorem percentage_to_other_schools :
  let accepted_students : ℚ := (total_applicants : ℚ) * acceptance_rate
  let percentage_to_other_schools : ℚ := 1 - attendance_rate
  (students_attending : ℚ) / accepted_students = attendance_rate →
  percentage_to_other_schools = 10 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_to_other_schools_l213_21362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l213_21392

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs ((x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁))

/-- Theorem: The area of triangle ABC with vertices A(0, 0), B(3, 3), and C(2, 1) is 3/2 -/
theorem triangle_abc_area :
  triangle_area (0, 0) (3, 3) (2, 1) = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l213_21392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_for_tan_negative_two_l213_21348

theorem sin_cos_product_for_tan_negative_two (α : Real) (h : Real.tan α = -2) :
  Real.sin α * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_for_tan_negative_two_l213_21348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_fraction_l213_21307

theorem root_sum_fraction (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + 3*x₁ + 1 = 0 → 
  x₂^3 + 3*x₂ + 1 = 0 → 
  x₃^3 + 3*x₃ + 1 = 0 → 
  x₁^2/((5*x₂+1)*(5*x₃+1)) + x₂^2/((5*x₁+1)*(5*x₃+1)) + x₃^2/((5*x₁+1)*(5*x₂+1)) = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_fraction_l213_21307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_60_degrees_max_area_when_c_is_2sqrt3_l213_21355

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  t.c * Real.tan t.C = Real.sqrt 3 * (t.a * Real.cos t.B + t.b * Real.cos t.A)

-- Theorem 1: Angle C is 60 degrees
theorem angle_C_is_60_degrees (t : Triangle) 
  (h1 : isValidTriangle t) (h2 : satisfiesCondition t) : 
  t.C = Real.pi / 3 := by sorry

-- Theorem 2: Maximum area when c = 2√3
theorem max_area_when_c_is_2sqrt3 (t : Triangle) 
  (h1 : isValidTriangle t) (h2 : satisfiesCondition t) (h3 : t.c = 2 * Real.sqrt 3) :
  ∀ (s : Triangle), isValidTriangle s → satisfiesCondition s → s.c = 2 * Real.sqrt 3 →
    t.a * t.b * Real.sin t.C / 2 ≥ s.a * s.b * Real.sin s.C / 2 ∧
    t.a * t.b * Real.sin t.C / 2 ≤ 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_60_degrees_max_area_when_c_is_2sqrt3_l213_21355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_price_calculation_l213_21399

/-- Proves that given a 15% discount on a set consisting of a $55 table and 4 chairs of equal price, 
    if the final discounted price is $135, then the original price of each chair is approximately $25.96. -/
theorem chair_price_calculation (table_price : ℝ) (chair_count : ℕ) (discount_rate : ℝ) (final_price : ℝ) : 
  table_price = 55 →
  chair_count = 4 →
  discount_rate = 0.15 →
  final_price = 135 →
  ∃ (chair_price : ℝ), (chair_price ≥ 25.95 ∧ chair_price ≤ 25.97) ∧ 
    (1 - discount_rate) * (table_price + chair_count * chair_price) = final_price :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_price_calculation_l213_21399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_55_l213_21376

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => a n + 2 * (n + 1)

theorem a_7_equals_55 : a 7 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_55_l213_21376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_count_l213_21393

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / (x^3 + 6*x^2 - 7*x)

-- Define what a vertical asymptote is
def is_vertical_asymptote (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε) ∧
  (∀ M > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x| > M)

-- Theorem statement
theorem vertical_asymptotes_count :
  ∃! (S : Finset ℝ), (∀ x ∈ S, is_vertical_asymptote f x) ∧ S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_count_l213_21393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blocks_in_box_l213_21366

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- The box dimensions -/
def box : Dimensions := ⟨5, 3, 2⟩

/-- The block dimensions -/
def block : Dimensions := ⟨3, 1, 1⟩

/-- Calculates the maximum number of blocks that can fit in the box -/
def maxBlocks (boxDim blockDim : Dimensions) : ℕ :=
  Int.toNat <| Int.floor ((boxDim.length / blockDim.width) * (boxDim.width / blockDim.height) * (boxDim.height / blockDim.length))

/-- Theorem stating that the maximum number of blocks that can fit in the box is 15 -/
theorem max_blocks_in_box : maxBlocks box block = 15 := by
  sorry

#eval maxBlocks box block

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blocks_in_box_l213_21366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l213_21318

-- Define the domain
def Domain : Type := Fin 3

-- Define function f
def f : Domain → Int
| ⟨0, _⟩ => -1  -- Represents -1
| ⟨1, _⟩ => -1  -- Represents 0
| ⟨2, _⟩ => 1   -- Represents 1

-- Define function g
def g : Domain → Domain
| ⟨0, _⟩ => ⟨2, by norm_num⟩  -- g(-1) = 1
| ⟨1, _⟩ => ⟨2, by norm_num⟩  -- g(0) = 1
| ⟨2, _⟩ => ⟨0, by norm_num⟩  -- g(1) = -1

-- Define the solution set
def SolutionSet : Set Domain := {x | f (g x) > 0}

-- Theorem statement
theorem solution_set_correct : SolutionSet = {⟨0, by norm_num⟩, ⟨1, by norm_num⟩} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l213_21318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chords_area_sum_l213_21353

theorem circle_chords_area_sum (r c d : ℝ) (m n : ℕ) : 
  r = 36 → 
  c = 66 → 
  d = 12 → 
  ∃ (A : ℝ), A = m * Real.pi - n * Real.sqrt (d : ℝ) ∧ 
              m = 216 ∧ 
              n = 162 ∧ 
              d = 15 ∧ 
              m + n + d = 393 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chords_area_sum_l213_21353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l213_21358

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := min (-x + 6) (-2*x^2 + 4*x + 6)

-- State the theorem
theorem f_max_value :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x₀ : ℝ), f x₀ = M) ∧ M = 8 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l213_21358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_conic_sections_l213_21379

/-- The value of m for which the ellipse x^2 + 4y^2 = 4 and the hyperbola x^2 - m(y+2)^2 = 1 are tangent -/
noncomputable def tangent_conic_sections_m : ℝ := 12/13

/-- Ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

/-- Hyperbola equation -/
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+2)^2 = 1

/-- The ellipse and hyperbola are tangent -/
def are_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse x y ∧ hyperbola x y m ∧
  ∀ (x' y' : ℝ), ellipse x' y' ∧ hyperbola x' y' m → (x', y') = (x, y)

/-- Theorem stating that the ellipse and hyperbola are tangent when m = 12/13 -/
theorem tangent_conic_sections :
  are_tangent tangent_conic_sections_m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_conic_sections_l213_21379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_percentage_both_subjects_l213_21381

noncomputable def min_percentage_both (p c : ℝ) : ℝ := max (p + c - 100) 0

theorem min_percentage_both_subjects (physics_percentage chemistry_percentage : ℝ) 
  (h1 : physics_percentage = 68)
  (h2 : chemistry_percentage = 72) :
  min_percentage_both physics_percentage chemistry_percentage = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_percentage_both_subjects_l213_21381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coordinates_l213_21365

/-- A right triangle with sides 8, 6, and 10 -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0
  side_a : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 8
  side_b : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 6
  side_c : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10

/-- The incenter of a triangle -/
noncomputable def incenter (t : RightTriangle) : ℝ × ℝ :=
  let a := 8
  let b := 6
  let c := 10
  let x := a / (a + b + c)
  let y := b / (a + b + c)
  let z := c / (a + b + c)
  (x * t.A.1 + y * t.B.1 + z * t.C.1, x * t.A.2 + y * t.B.2 + z * t.C.2)

theorem incenter_coordinates (t : RightTriangle) :
  incenter t = (1/3 * t.A.1 + 1/4 * t.B.1 + 5/12 * t.C.1,
                1/3 * t.A.2 + 1/4 * t.B.2 + 5/12 * t.C.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coordinates_l213_21365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_domain_conjecture_correct_l213_21397

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x) * Real.exp x

-- Theorem stating that only one conjecture is correct
theorem only_domain_conjecture_correct :
  -- Domain is all real numbers
  (∀ x, f x ≠ 0) ∧
  -- Not increasing on (0, 2)
  (¬ ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y) ∧
  -- Not an odd function
  (¬ ∀ x, f (-x) = -f x) ∧
  -- Does not attain minimum at x = 2
  (¬ ∀ x, f 2 ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_domain_conjecture_correct_l213_21397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l213_21326

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (S : ℝ) : 
  A = 2 * Real.pi / 3 →  -- 120° in radians
  b = 1 →
  S = Real.sqrt 3 →
  S = (1/2) * b * c * Real.sin A →  -- Area formula
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →  -- Law of Cosines
  Real.sin B = b * (Real.sin A) / a →  -- Law of Sines
  a = Real.sqrt 7 ∧ 
  c = 2 ∧ 
  Real.sin (B + Real.pi/6) = (2 * Real.sqrt 7)/7 := by
  sorry

#check triangle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l213_21326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_n_consecutive_composites_iff_n_le_4_l213_21342

/-- A function that checks if there exist n consecutive composite positive integers less than n! -/
def existsNConsecutiveComposites (n : ℕ) : Prop :=
  ∃ k : ℕ, k + n ≤ Nat.factorial n ∧ ∀ i : ℕ, i < n → k + i + 1 > 1 ∧ ¬(Nat.Prime (k + i + 1))

/-- The theorem stating the condition for n -/
theorem no_n_consecutive_composites_iff_n_le_4 :
  ∀ n : ℕ, n > 0 → (¬(existsNConsecutiveComposites n) ↔ n ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_n_consecutive_composites_iff_n_le_4_l213_21342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l213_21359

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the second derivative of f
def f'' : ℝ → ℝ := sorry

-- Axioms based on the given conditions
axiom f_symmetry (x : ℝ) : f x = 6 * x^2 - f (-x)

axiom f''_bound (x : ℝ) (h : x < 0) : 2 * f'' x + 1 < 12 * x

-- Define the inequality condition
def inequality_condition (m : ℝ) : Prop :=
  f (m + 2) ≤ f (-2 * m) + 12 - 9 * m^2

-- Theorem statement
theorem range_of_m :
  {m : ℝ | inequality_condition m} = {m : ℝ | -2/3 ≤ m} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l213_21359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_three_pairs_satisfy_l213_21305

/-- A pair of positive integers (p, q) satisfies the condition if both equations
    x^2 - px + q = 0 and x^2 - qx + p = 0 have integral solutions. -/
def SatisfiesCondition (p q : ℕ) : Prop :=
  (∃ x : ℤ, x^2 - (p : ℤ) * x + (q : ℤ) = 0) ∧
  (∃ y : ℤ, y^2 - (q : ℤ) * y + (p : ℤ) = 0)

/-- The theorem stating that only (4, 4), (6, 5), and (5, 6) satisfy the condition. -/
theorem only_three_pairs_satisfy :
  ∀ p q : ℕ, p > 0 ∧ q > 0 → (SatisfiesCondition p q ↔ (p = 4 ∧ q = 4) ∨ (p = 6 ∧ q = 5) ∨ (p = 5 ∧ q = 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_three_pairs_satisfy_l213_21305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l213_21335

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (x + 1) + Real.sqrt (4 - x^2)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2 ∧ x ≠ 0}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l213_21335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_expression_l213_21337

theorem natural_number_expression (a : ℕ) :
  ∃ (n : ℕ), ((a : ℝ) + 1 + Real.sqrt ((a : ℝ)^5 + 2*(a : ℝ)^2 + 1)) / ((a : ℝ)^2 + 1) = n ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_expression_l213_21337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_intermission_duration_l213_21345

/-- Calculates the duration of an intermission in a concert --/
theorem concert_intermission_duration 
  (total_duration : ℕ) 
  (num_songs : ℕ) 
  (regular_song_duration : ℕ) 
  (special_song_duration : ℕ) : 
  total_duration = 80 ∧ 
  num_songs = 13 ∧ 
  regular_song_duration = 5 ∧ 
  special_song_duration = 10 → 
  total_duration - (regular_song_duration * (num_songs - 1) + special_song_duration) = 10 := by
  intro h
  sorry

#check concert_intermission_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_intermission_duration_l213_21345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_meeting_properties_l213_21370

/-- Represents the properties of two trucks meeting on a road -/
structure TruckMeeting where
  distance : ℝ
  time : ℝ
  speed_ratio : ℝ

/-- Calculates the sum of speeds of two trucks -/
noncomputable def sum_of_speeds (m : TruckMeeting) : ℝ :=
  m.distance / m.time

/-- Calculates the speed of truck A -/
noncomputable def speed_of_A (m : TruckMeeting) : ℝ :=
  m.distance / (m.time * (1 + m.speed_ratio))

/-- Calculates the speed of truck B -/
noncomputable def speed_of_B (m : TruckMeeting) : ℝ :=
  m.speed_ratio * speed_of_A m

/-- Calculates the difference in distance traveled between trucks B and A -/
noncomputable def distance_difference (m : TruckMeeting) : ℝ :=
  (speed_of_B m - speed_of_A m) * m.time

/-- Theorem stating the properties of the truck meeting scenario -/
theorem truck_meeting_properties (m : TruckMeeting) 
  (h1 : m.distance = 300) 
  (h2 : m.time = 3) 
  (h3 : m.speed_ratio = 1.5) : 
  sum_of_speeds m = 100 ∧ 
  speed_of_B m = 60 ∧ 
  distance_difference m = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_meeting_properties_l213_21370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l213_21369

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  -1 ≤ a ∧ a < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l213_21369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grove_cleaning_time_l213_21329

/-- Proves that cleaning a 4x5 grove of trees takes 1 hour when each tree initially takes 6 minutes to clean, but with help it takes half as long. -/
theorem grove_cleaning_time 
  (grove_width : ℕ)
  (grove_height : ℕ)
  (initial_cleaning_time : ℕ)
  (h1 : grove_width = 4)
  (h2 : grove_height = 5)
  (h3 : initial_cleaning_time = 6) :
  let total_trees := grove_width * grove_height
  let cleaning_time_with_help := initial_cleaning_time / 2
  let total_cleaning_time := total_trees * cleaning_time_with_help
  (total_cleaning_time : ℚ) / 60 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grove_cleaning_time_l213_21329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_280_meters_l213_21322

-- Define the given parameters
noncomputable def train_speed_kmph : ℝ := 72
noncomputable def time_to_cross_platform : ℝ := 30
noncomputable def time_to_cross_man : ℝ := 16

-- Convert km/h to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmph * (1000 / 3600)

-- Define the length of the train
noncomputable def train_length : ℝ := train_speed_ms * time_to_cross_man

-- Define the total distance covered while crossing the platform
noncomputable def total_distance : ℝ := train_speed_ms * time_to_cross_platform

-- Theorem to prove
theorem platform_length_is_280_meters :
  total_distance - train_length = 280 := by
  -- Expand definitions
  unfold total_distance train_length train_speed_ms
  -- Perform algebraic manipulations
  simp [train_speed_kmph, time_to_cross_platform, time_to_cross_man]
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_280_meters_l213_21322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_sum_l213_21387

theorem polynomial_coefficient_sum : 
  ∀ (a₅ a₄ a₃ a₂ a₁ a₀ : ℝ), 
  (fun x ↦ (x + 1)^3 * (x + 2)^2) = 
  (fun x ↦ a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ + a₂ + a₀ = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_sum_l213_21387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_parallel_lines_l213_21388

-- Define the basic geometric objects
variable {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]
variable (a : Set α) (β : Set α)

-- Define the parallel relation between a line and a plane
def line_parallel_to_plane (l : Set α) (p : Set α) : Prop := sorry

-- Define the parallel relation between two lines
def lines_parallel (l1 l2 : Set α) : Prop := sorry

-- Define a line being in a plane
def line_in_plane (l : Set α) (p : Set α) : Prop := sorry

-- State the theorem
theorem infinitely_many_non_parallel_lines 
  (h : line_parallel_to_plane a β) :
  ∃ (S : Set (Set α)), 
    (∀ l ∈ S, line_in_plane l β ∧ ¬ lines_parallel l a) ∧ 
    Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_parallel_lines_l213_21388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l213_21339

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (platform_length : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating the time taken for a specific train to cross a platform -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 250
  let train_speed_kmh : ℝ := 55
  let platform_length : ℝ := 520
  let crossing_time := train_crossing_time train_length train_speed_kmh platform_length
  abs (crossing_time - 50.39) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l213_21339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_term_of_arithmetic_sequence_l213_21308

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  arithmetic_property : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem ninth_term_of_arithmetic_sequence 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 5 = 8) 
  (h2 : sum_n_terms seq 3 = 6) : 
  seq.a 9 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_term_of_arithmetic_sequence_l213_21308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l213_21351

noncomputable def f (k x : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem f_properties (k : ℝ) :
  (∀ x : ℝ, f k x ≤ max ((k + 2) / 3) 1) ∧
  (∀ x : ℝ, f k x ≥ min ((k + 2) / 3) 1) ∧
  (∀ a b c : ℝ, ∃ (triangle : ℝ → ℝ → ℝ → Prop),
    triangle (f k a) (f k b) (f k c) ↔ -1/2 < k ∧ k < 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l213_21351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downhill_speed_is_100_l213_21311

/-- Calculates the downhill speed given the uphill speed, total time, and total distance of a round trip. -/
noncomputable def downhill_speed (uphill_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) : ℝ :=
  let half_distance := total_distance / 2
  let uphill_time := half_distance / uphill_speed
  let downhill_time := total_time - uphill_time
  half_distance / downhill_time

/-- Theorem stating that for a round trip with given conditions, the downhill speed is 100 kmph. -/
theorem downhill_speed_is_100 :
  downhill_speed 50 12 800 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downhill_speed_is_100_l213_21311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_cube_root_negative_27_l213_21352

theorem abs_cube_root_negative_27 : |(-27 : ℝ) ^ (1/3 : ℝ)| = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_cube_root_negative_27_l213_21352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_exponential_l213_21344

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem inverse_function_of_exponential :
  (∀ x : ℝ, g (f x) = x) ∧
  (∀ y : ℝ, y > 0 → f (g y) = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_exponential_l213_21344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_equals_result_l213_21324

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![2, 0, -1],
  ![0, 3, -2],
  ![-2, 3, 2]
]

def B : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, -3, 0],
  ![2, 0, -3],
  ![5, 0, 0]
]

def result : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![1, -6, 0],
  ![-4, 0, -9],
  ![10, 6, -9]
]

theorem matrix_product_equals_result : A * B = result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_equals_result_l213_21324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equation_l213_21312

-- Define our own max function to avoid ambiguity with built-in max
def myMax (a b : ℚ) : ℚ := if a ≥ b then a else b

theorem solution_equation : ∃! x : ℚ, myMax x (-x) = 2*x + 1 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equation_l213_21312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_property_l213_21313

theorem rational_property (p : ℕ) (h_prime : Nat.Prime p) :
  let x : ℚ := (p^2 - p - 1 : ℚ) / p^2
  (0 < x ∧ x < 1) ∧
  (∀ (a b : ℕ), x = a / b → Nat.Coprime a b) ∧
  ((x + p) / (1 + p) - x = 1 / p^2) ∧
  (p = 2 → (1 : ℚ) / 2 = 1 / p^2 + (1 : ℚ) / 2 / (1 + p)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_property_l213_21313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arcsin_cos_l213_21390

/-- The area bounded by y = arcsin(cos x) and the x-axis over [π/2, 7π/2] is π² -/
theorem area_arcsin_cos : 
  (∫ x in Set.Icc (π/2) (7*π/2), |Real.arcsin (Real.cos x)|) = π^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arcsin_cos_l213_21390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_cone_l213_21368

/-- Given a cone with base radius 2 and slant height c, containing three externally touching 
    spheres of radius r that touch the lateral surface of the cone, with two of them touching 
    the base, the maximum value of r is √3 - 1. -/
theorem max_sphere_radius_in_cone (c : ℝ) (r : ℝ) 
    (h1 : c > 0) 
    (h2 : r > 0) 
    (h3 : r ≤ 2) 
    (h4 : ∃ (x y z : ℝ), x^2 + y^2 = 4 ∧ (x - r)^2 + y^2 = 4*r^2 ∧ 
         (x + r)^2 + y^2 = 4*r^2 ∧ x^2 + y^2 + z^2 = c^2 ∧ 
         (x - r)^2 + y^2 + (z - r*Real.sqrt 3)^2 = c^2 ∧ 
         (x + r)^2 + y^2 + (z - r*Real.sqrt 3)^2 = c^2) : 
  r ≤ Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_cone_l213_21368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l213_21332

def point1 : ℝ × ℝ := (0, 4)
def point2 : ℝ × ℝ := (3, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points : distance point1 point2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l213_21332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_with_many_roots_l213_21327

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Predicate to check if a real number is a root of the equation -/
def is_root (p q : ℝ) (x : ℝ) : Prop :=
  (floor (x^2) : ℝ) + p * x + q = 0

theorem equation_with_many_roots :
  ∃ (p q : ℝ), p ≠ 0 ∧ (∃ (S : Finset ℝ), (∀ x ∈ S, is_root p q x) ∧ S.card > 100) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_with_many_roots_l213_21327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l213_21334

/-- The distance between two parallel lines in R² --/
noncomputable def distance_parallel_lines (a₁ a₂ b₁ b₂ d₁ d₂ : ℝ) : ℝ :=
  let v := (a₁ - b₁, a₂ - b₂)
  let d := (d₁, d₂)
  let p := ((v.1 * d.1 + v.2 * d.2) / (d.1^2 + d.2^2)) • d
  let c := (v.1 - p.1, v.2 - p.2)
  Real.sqrt (c.1^2 + c.2^2)

/-- Theorem stating the distance between two specific parallel lines --/
theorem distance_specific_parallel_lines :
  distance_parallel_lines 3 (-4) 0 (-1) 2 (-6) = 3 * Real.sqrt 10 / 5 := by
  sorry

#check distance_specific_parallel_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l213_21334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_cube_root_eq_solve_equation_final_l213_21320

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : 9 / x^2 = x / 25) : x^3 = 225 := by
  sorry

-- Helper theorem to connect the main result to the cube root
theorem cube_root_eq (a : ℝ) (h : 0 < a) : (a^(1/3:ℝ))^3 = a := by
  sorry

theorem solve_equation_final (x : ℝ) (h1 : x ≠ 0) (h2 : 9 / x^2 = x / 25) : 
  x = (225 : ℝ)^(1/3:ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_cube_root_eq_solve_equation_final_l213_21320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_enlargement_l213_21363

theorem circle_enlargement (r : ℝ) (h : r > 0) :
  (π * (3 * r)^2) / (π * r^2) = 9 ∧
  (2 * π * (3 * r)) / (2 * π * r) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_enlargement_l213_21363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_period_l213_21304

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x - (Real.cos (x + Real.pi/4))^2

def Smallest_positive_period (f : ℝ → ℝ) : ℝ := sorry
def Triangle_area (A B C a b c : ℝ) : ℝ := sorry

theorem triangle_area_and_function_period 
  (A B C : ℝ) (a b c : ℝ) :
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  f (A/2) = (Real.sqrt 3 - 1)/2 →
  a = 1 →
  b + c = 2 →
  Smallest_positive_period f = Real.pi ∧ 
  Triangle_area A B C a b c = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_period_l213_21304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l213_21386

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ
  c : ℝ

/-- Checks if a point is on the ellipse -/
def on_ellipse (E : Ellipse) (P : Point) : Prop :=
  P.x^2 / E.a^2 + P.y^2 / E.b^2 = 1

/-- Checks if a point is on the line -/
def on_line (L : Line) (P : Point) : Prop :=
  L.m * P.x + P.y - L.m = 0

/-- Theorem statement -/
theorem ellipse_intersection_theorem (E : Ellipse) (F : Point) (l : Line) 
  (A B C D M N : Point) :
  (∀ P : Point, on_ellipse E P → P = A ∨ P = B ∨ P = C ∨ P = D) →
  (E.a^2 - E.b^2) / E.a^2 = 1/4 →
  (∀ P : Point, on_line l P → P = F) →
  (on_ellipse E A ∧ on_ellipse E B ∧ on_ellipse E C ∧ on_ellipse E D) →
  ((C.x - A.x) * (D.x - B.x) + (C.y - A.y) * (D.y - B.y) = 0) →
  (M.x = (A.x + C.x) / 2 ∧ M.y = (A.y + C.y) / 2) →
  (N.x = (B.x + D.x) / 2 ∧ N.y = (B.y + D.y) / 2) →
  ((C.y - A.y) / (C.x - A.x) = 2) →
  (∃ t : ℝ, M.x + t * (N.x - M.x) = 4/7 ∧ M.y + t * (N.y - M.y) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l213_21386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_width_one_l213_21382

-- Define the function f(x) = (ln x)/x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- Define the theorem
theorem channel_width_one :
  ∃ (m₁ m₂ : ℝ), m₂ - m₁ = 1 ∧
  ∀ x : ℝ, x ≥ 1 → m₁ ≤ f x ∧ f x ≤ m₂ := by
  -- We choose m₁ = 0 and m₂ = 1
  use 0, 1
  constructor
  · -- Prove m₂ - m₁ = 1
    simp
  · -- Prove ∀ x : ℝ, x ≥ 1 → 0 ≤ f x ∧ f x ≤ 1
    intro x hx
    constructor
    · -- Prove 0 ≤ f x
      sorry
    · -- Prove f x ≤ 1
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_width_one_l213_21382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_matrix_sum_l213_21367

theorem orthogonal_matrix_sum (θ : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]
  (B.transpose = B⁻¹) → (Real.cos θ) ^ 2 + (Real.sin θ) ^ 2 + (-Real.sin θ) ^ 2 + (Real.cos θ) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_matrix_sum_l213_21367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_is_18_l213_21396

/-- The cost of an album in dollars -/
def album_cost : ℚ := 20

/-- The percentage discount of the CD compared to the album -/
def cd_discount_percent : ℚ := 30

/-- The additional cost of the book compared to the CD in dollars -/
def book_additional_cost : ℚ := 4

/-- The cost of the CD in dollars -/
noncomputable def cd_cost : ℚ := album_cost * (1 - cd_discount_percent / 100)

/-- The cost of the book in dollars -/
noncomputable def book_cost : ℚ := cd_cost + book_additional_cost

theorem book_cost_is_18 : book_cost = 18 := by
  -- Expand the definitions
  unfold book_cost cd_cost
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_is_18_l213_21396
