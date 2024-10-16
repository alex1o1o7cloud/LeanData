import Mathlib

namespace NUMINAMATH_CALUDE_range_of_b_l3343_334310

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x^2 - 2*a*x + 1 else -(x^2 - 2*a*x + 1)

-- State the theorem
theorem range_of_b (a : ℝ) (b : ℝ) :
  (a > 0) →
  (∀ x : ℝ, f a (x^3 + a) = -f a (-(x^3 + a))) →
  (∀ x : ℝ, x ∈ Set.Icc (b - 1) (b + 2) → f a (b * x) ≥ 4 * f a (x + 1)) →
  b ∈ Set.Iic (-Real.sqrt 5) ∪ Set.Ici ((3 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l3343_334310


namespace NUMINAMATH_CALUDE_quadratic_properties_l3343_334313

def quadratic_function (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

theorem quadratic_properties (a h k : ℝ) :
  quadratic_function a h k (-2) = 0 →
  quadratic_function a h k 4 = 0 →
  quadratic_function a h k 1 = -9/2 →
  (a = 1/2 ∧ h = 1 ∧ k = -9/2 ∧ ∀ x, quadratic_function a h k x ≥ -9/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3343_334313


namespace NUMINAMATH_CALUDE_secret_spread_days_l3343_334344

/-- The number of people who know the secret after n days -/
def people_knowing_secret (n : ℕ) : ℕ :=
  (3^(n+1) - 1) / 2

/-- The proposition that it takes 7 days for at least 2186 people to know the secret -/
theorem secret_spread_days : ∃ n : ℕ, n = 7 ∧ 
  people_knowing_secret (n - 1) < 2186 ∧ people_knowing_secret n ≥ 2186 :=
sorry

end NUMINAMATH_CALUDE_secret_spread_days_l3343_334344


namespace NUMINAMATH_CALUDE_triangle_count_l3343_334397

theorem triangle_count : ∃! (n : ℕ), n = 59 ∧
  (∀ (a b c : ℕ), (a < b ∧ b < c) →
    (b = 60) →
    (c - b = b - a) →
    (a + b + c = 180) →
    (0 < a ∧ a < b ∧ b < c) →
    (∃ (d : ℕ), a = 60 - d ∧ c = 60 + d ∧ 0 < d ∧ d < 60)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_count_l3343_334397


namespace NUMINAMATH_CALUDE_function_is_identity_l3343_334321

def IsNonDegenerateTriangle (a b c : ℕ+) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def SatisfiesTriangleCondition (f : ℕ+ → ℕ+) : Prop :=
  ∀ a b : ℕ+, IsNonDegenerateTriangle a (f b) (f (b + f a - 1))

theorem function_is_identity (f : ℕ+ → ℕ+) 
  (h : SatisfiesTriangleCondition f) : 
  ∀ x : ℕ+, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_is_identity_l3343_334321


namespace NUMINAMATH_CALUDE_roadwork_pitch_calculation_l3343_334315

/-- Calculates the number of barrels of pitch needed to pave the remaining road -/
def barrels_of_pitch_needed (total_road_length : ℕ) (truckloads_per_mile : ℕ) (gravel_bags_per_truckload : ℕ) (gravel_to_pitch_ratio : ℕ) (paved_miles : ℕ) : ℕ :=
  let remaining_miles := total_road_length - paved_miles
  let total_truckloads := remaining_miles * truckloads_per_mile
  let total_gravel_bags := total_truckloads * gravel_bags_per_truckload
  total_gravel_bags / gravel_to_pitch_ratio

theorem roadwork_pitch_calculation :
  barrels_of_pitch_needed 16 3 2 5 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roadwork_pitch_calculation_l3343_334315


namespace NUMINAMATH_CALUDE_best_fitting_highest_r_squared_l3343_334360

/-- Represents a regression model with its coefficient of determination -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared : 0 ≤ r_squared ∧ r_squared ≤ 1

/-- Determines if a model is the best-fitting among a list of models -/
def is_best_fitting (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

theorem best_fitting_highest_r_squared 
  (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.98)
  (h2 : model2.r_squared = 0.80)
  (h3 : model3.r_squared = 0.50)
  (h4 : model4.r_squared = 0.25) :
  is_best_fitting model1 [model1, model2, model3, model4] :=
by sorry

end NUMINAMATH_CALUDE_best_fitting_highest_r_squared_l3343_334360


namespace NUMINAMATH_CALUDE_water_pouring_proof_l3343_334369

/-- Calculates the fraction of water remaining after n rounds -/
def water_remaining (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 1/2
  | 2 => 1/3
  | k + 3 => water_remaining (k + 2) * (2 * (k + 3)) / (2 * (k + 3) + 1)

/-- The number of rounds needed to reach exactly 1/5 of the original water -/
def rounds_to_one_fifth : ℕ := 6

theorem water_pouring_proof :
  water_remaining rounds_to_one_fifth = 1/5 :=
sorry

end NUMINAMATH_CALUDE_water_pouring_proof_l3343_334369


namespace NUMINAMATH_CALUDE_solve_adam_allowance_l3343_334358

def adam_allowance_problem (initial_amount spent_amount final_amount : ℕ) : Prop :=
  let remaining_amount := initial_amount - spent_amount
  let allowance := final_amount - remaining_amount
  allowance = 5

theorem solve_adam_allowance :
  adam_allowance_problem 5 2 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_adam_allowance_l3343_334358


namespace NUMINAMATH_CALUDE_wendys_full_face_time_l3343_334359

/-- Calculates the total time for a "full face" routine given the number of products,
    waiting time between products, and make-up application time. -/
def fullFaceTime (numProducts : ℕ) (waitingTime : ℕ) (makeupTime : ℕ) : ℕ :=
  (numProducts - 1) * waitingTime + makeupTime

/-- Proves that Wendy's "full face" routine takes 50 minutes. -/
theorem wendys_full_face_time :
  fullFaceTime 5 5 30 = 50 := by
  sorry

#eval fullFaceTime 5 5 30

end NUMINAMATH_CALUDE_wendys_full_face_time_l3343_334359


namespace NUMINAMATH_CALUDE_fourth_quarter_profits_l3343_334341

/-- Proves that given the annual profits, first quarter profits, and third quarter profits,
    the fourth quarter profits are equal to the difference between the annual profits
    and the sum of the first and third quarter profits. -/
theorem fourth_quarter_profits
  (annual_profits : ℕ)
  (first_quarter_profits : ℕ)
  (third_quarter_profits : ℕ)
  (h1 : annual_profits = 8000)
  (h2 : first_quarter_profits = 1500)
  (h3 : third_quarter_profits = 3000) :
  annual_profits - (first_quarter_profits + third_quarter_profits) = 3500 :=
by sorry

end NUMINAMATH_CALUDE_fourth_quarter_profits_l3343_334341


namespace NUMINAMATH_CALUDE_unique_ages_solution_l3343_334338

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_ages_solution :
  ∃! (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a = 2 * b ∧
    b = c - 7 ∧
    is_prime (a + b + c) ∧
    a + b + c < 70 ∧
    sum_of_digits (a + b + c) = 13 ∧
    a = 30 ∧ b = 15 ∧ c = 22 :=
sorry

end NUMINAMATH_CALUDE_unique_ages_solution_l3343_334338


namespace NUMINAMATH_CALUDE_equation_holds_l3343_334399

theorem equation_holds (n : ℕ+) : 
  (n^2)^2 + n^2 + 1 = (n^2 + n + 1) * ((n-1)^2 + (n-1) + 1) := by
  sorry

#check equation_holds

end NUMINAMATH_CALUDE_equation_holds_l3343_334399


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l3343_334339

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l3343_334339


namespace NUMINAMATH_CALUDE_lines_properties_l3343_334337

/-- Two lines in 2D space -/
structure Lines where
  l1 : ℝ → ℝ → ℝ := fun x y => 2 * x + y + 4
  l2 : ℝ → ℝ → ℝ → ℝ := fun a x y => a * x + 4 * y + 1

/-- The intersection point of two lines when they are perpendicular -/
def intersection (lines : Lines) : ℝ × ℝ := sorry

/-- The distance between two lines when they are parallel -/
def distance (lines : Lines) : ℝ := sorry

/-- Main theorem about the properties of the two lines -/
theorem lines_properties (lines : Lines) :
  (intersection lines = (-3/2, -1) ∧ 
   distance lines = 3 * Real.sqrt 5 / 4) := by sorry

end NUMINAMATH_CALUDE_lines_properties_l3343_334337


namespace NUMINAMATH_CALUDE_fraction_problem_l3343_334311

theorem fraction_problem : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5100 = 765.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3343_334311


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3343_334304

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2 * Complex.I) = 2) : 
  Complex.im z = -4/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3343_334304


namespace NUMINAMATH_CALUDE_number_of_balls_in_box_l3343_334316

theorem number_of_balls_in_box : ∃ x : ℕ, x > 20 ∧ x < 30 ∧ (x - 20 = 30 - x) ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_balls_in_box_l3343_334316


namespace NUMINAMATH_CALUDE_tabletop_qualification_l3343_334353

theorem tabletop_qualification (length width diagonal : ℝ) 
  (h_length : length = 60)
  (h_width : width = 32)
  (h_diagonal : diagonal = 68) : 
  length ^ 2 + width ^ 2 = diagonal ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_tabletop_qualification_l3343_334353


namespace NUMINAMATH_CALUDE_range_of_a_l3343_334346

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 0 ∧ 2^x - a = 1/(x-1)) → 0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3343_334346


namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l3343_334326

theorem geometric_sequence_b_value (b : ℝ) (h₁ : b > 0) :
  (∃ r : ℝ, r ≠ 0 ∧
    b = 10 * r ∧
    10 / 9 = b * r ∧
    10 / 81 = (10 / 9) * r) →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l3343_334326


namespace NUMINAMATH_CALUDE_max_product_two_digit_numbers_l3343_334355

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def unique_digits (a b c d e : ℕ) : Prop :=
  let digits := a.digits 10 ++ b.digits 10 ++ c.digits 10 ++ d.digits 10 ++ e.digits 10
  digits.length = 10 ∧ digits.toFinset.card = 10

theorem max_product_two_digit_numbers :
  ∃ (a b c d e : ℕ),
    is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧ is_two_digit e ∧
    unique_digits a b c d e ∧
    a * b * c * d * e = 1785641760 ∧
    ∀ (x y z w v : ℕ),
      is_two_digit x ∧ is_two_digit y ∧ is_two_digit z ∧ is_two_digit w ∧ is_two_digit v ∧
      unique_digits x y z w v →
      x * y * z * w * v ≤ 1785641760 :=
by sorry

end NUMINAMATH_CALUDE_max_product_two_digit_numbers_l3343_334355


namespace NUMINAMATH_CALUDE_exists_tricolor_right_triangle_l3343_334314

/-- A color type with three possible values -/
inductive Color
| Red
| Green
| Blue

/-- A point in the plane with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point -/
def Coloring := Point → Color

/-- Predicate to check if a triangle is right-angled -/
def isRightTriangle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p3.x - p2.x)^2 + (p3.y - p2.y)^2 = 
  (p3.x - p1.x)^2 + (p3.y - p1.y)^2

/-- Main theorem -/
theorem exists_tricolor_right_triangle (coloring : Coloring) 
  (h1 : ∃ p : Point, coloring p = Color.Red)
  (h2 : ∃ p : Point, coloring p = Color.Green)
  (h3 : ∃ p : Point, coloring p = Color.Blue) :
  ∃ p1 p2 p3 : Point, 
    isRightTriangle p1 p2 p3 ∧ 
    coloring p1 ≠ coloring p2 ∧ 
    coloring p2 ≠ coloring p3 ∧ 
    coloring p3 ≠ coloring p1 :=
sorry

end NUMINAMATH_CALUDE_exists_tricolor_right_triangle_l3343_334314


namespace NUMINAMATH_CALUDE_total_purchase_cost_l3343_334319

def snake_toy_cost : ℚ := 11.76
def cage_cost : ℚ := 14.54

theorem total_purchase_cost : snake_toy_cost + cage_cost = 26.30 := by
  sorry

end NUMINAMATH_CALUDE_total_purchase_cost_l3343_334319


namespace NUMINAMATH_CALUDE_point_in_region_l3343_334365

def is_in_region (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 25 ∧ 
  (x + y ≠ 0 → -1 ≤ x / (x + y) ∧ x / (x + y) ≤ 1) ∧
  (x ≠ 0 → 0 ≤ x / y ∧ x / y ≤ 1)

theorem point_in_region (x y : ℝ) :
  is_in_region x y ↔ 
    (x^2 + y^2 ≤ 25 ∧ 
     ((x ≠ 0 ∧ y ≠ 0 ∧ 0 ≤ x / y ∧ x / y ≤ 1) ∨ 
      (x = 0 ∧ -5 ≤ y ∧ y ≤ 5) ∨ 
      (y = 0 ∧ 0 ≤ x ∧ x ≤ 5))) :=
  sorry

end NUMINAMATH_CALUDE_point_in_region_l3343_334365


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_ge_sqrt2_sum_l3343_334368

theorem sqrt_sum_squares_ge_sqrt2_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_ge_sqrt2_sum_l3343_334368


namespace NUMINAMATH_CALUDE_three_arcs_must_intersect_l3343_334364

/-- Represents a great circle arc on a sphere --/
structure GreatCircleArc where
  length : ℝ
  start_point : Sphere
  end_point : Sphere

/-- Defines a sphere --/
class Sphere where
  center : Point
  radius : ℝ

/-- Checks if two great circle arcs intersect or share an endpoint --/
def arcs_intersect (arc1 arc2 : GreatCircleArc) : Prop :=
  sorry

/-- Theorem: It's impossible to place three 300° great circle arcs on a sphere without intersections --/
theorem three_arcs_must_intersect (s : Sphere) :
  ∀ (arc1 arc2 arc3 : GreatCircleArc),
    arc1.length = 300 ∧ arc2.length = 300 ∧ arc3.length = 300 →
    arcs_intersect arc1 arc2 ∨ arcs_intersect arc2 arc3 ∨ arcs_intersect arc1 arc3 :=
by
  sorry

end NUMINAMATH_CALUDE_three_arcs_must_intersect_l3343_334364


namespace NUMINAMATH_CALUDE_johns_reading_rate_l3343_334354

/-- The number of books John read in 6 weeks -/
def total_books : ℕ := 48

/-- The number of weeks John read -/
def weeks : ℕ := 6

/-- The number of days John reads per week -/
def reading_days_per_week : ℕ := 2

/-- The number of books John can read in a day -/
def books_per_day : ℕ := total_books / (weeks * reading_days_per_week)

theorem johns_reading_rate : books_per_day = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_reading_rate_l3343_334354


namespace NUMINAMATH_CALUDE_min_digit_ratio_l3343_334374

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  digits_bound : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The sum of digits of a three-digit number -/
def ThreeDigitNumber.digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- The ratio of a number to the sum of its digits -/
def digitRatio (n : ThreeDigitNumber) : Rat :=
  n.value / n.digitSum

/-- The condition that the difference between hundreds and tens digit is 8 -/
def diffEight (n : ThreeDigitNumber) : Prop :=
  n.hundreds - n.tens = 8 ∨ n.tens - n.hundreds = 8

theorem min_digit_ratio :
  ∀ k : ThreeDigitNumber,
    diffEight k →
    ∀ m : ThreeDigitNumber,
      diffEight m →
      digitRatio k ≤ digitRatio m →
      k.value = 190 :=
sorry

end NUMINAMATH_CALUDE_min_digit_ratio_l3343_334374


namespace NUMINAMATH_CALUDE_investment_problem_l3343_334301

/-- Represents the investment scenario described in the problem -/
structure Investment where
  fund_a : ℝ  -- Initial investment in Fund A
  fund_b : ℝ  -- Initial investment in Fund B (unknown)
  rate_a : ℝ  -- Total interest rate for Fund A over two years
  rate_b : ℝ  -- Annual interest rate for Fund B

/-- Calculates the final value of Fund A after two years -/
def final_value_a (i : Investment) : ℝ :=
  i.fund_a * (1 + i.rate_a)

/-- Calculates the final value of Fund B after two years -/
def final_value_b (i : Investment) : ℝ :=
  i.fund_b * (1 + i.rate_b)^2

/-- Theorem stating the conditions and the result to be proved -/
theorem investment_problem (i : Investment) 
  (h1 : i.fund_a = 2000)
  (h2 : i.rate_a = 0.12)
  (h3 : i.rate_b = 0.30)
  (h4 : final_value_a i = final_value_b i + 549.9999999999998) :
  i.fund_b = 1000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3343_334301


namespace NUMINAMATH_CALUDE_abs_z2_minus_z1_equals_sqrt2_l3343_334387

theorem abs_z2_minus_z1_equals_sqrt2 : ∀ (z₁ z₂ : ℂ), 
  z₁ = 1 + 2*Complex.I → z₂ = 2 + Complex.I → Complex.abs (z₂ - z₁) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z2_minus_z1_equals_sqrt2_l3343_334387


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l3343_334322

theorem square_area_error_percentage (x : ℝ) (h : x > 0) :
  let measured_side := 1.12 * x
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let error_percentage := (area_error / actual_area) * 100
  error_percentage = 25.44 := by sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l3343_334322


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_l3343_334303

-- Define the conditions
def condition_A (θ : Real) (a : Real) : Prop :=
  Real.sqrt (1 + Real.sin θ) = a

def condition_B (θ : Real) (a : Real) : Prop :=
  Real.sin (θ / 2) + Real.cos (θ / 2) = a

-- Theorem statement
theorem not_necessary_not_sufficient :
  (∃ θ a, condition_A θ a ∧ ¬condition_B θ a) ∧
  (∃ θ a, condition_B θ a ∧ ¬condition_A θ a) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_l3343_334303


namespace NUMINAMATH_CALUDE_problem_solution_l3343_334357

theorem problem_solution (x y z : ℝ) 
  (h1 : 2 * x + y + z = 14)
  (h2 : 2 * x + y = 7)
  (h3 : x + 2 * y + Real.sqrt z = 10) :
  (x + y - z) / 3 = (-4 - Real.sqrt 7) / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3343_334357


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_integers_l3343_334347

theorem sum_of_five_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 5 * n + 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_integers_l3343_334347


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_l3343_334328

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + (a^3 - a) * x + 1

-- State the theorem
theorem increasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → x ≤ -1 → f a x < f a y) →
  -Real.sqrt 3 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_l3343_334328


namespace NUMINAMATH_CALUDE_absolute_value_of_complex_fraction_l3343_334352

theorem absolute_value_of_complex_fraction : 
  Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_complex_fraction_l3343_334352


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3343_334395

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = -1/2 + Real.sqrt 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3343_334395


namespace NUMINAMATH_CALUDE_smallest_third_term_of_gp_l3343_334373

theorem smallest_third_term_of_gp (a b c : ℝ) : 
  (∃ d : ℝ, a = 5 ∧ b = 5 + d ∧ c = 5 + 2*d) →  -- arithmetic progression
  (∃ r : ℝ, 5 * (20 + 2*c - 10) = (8 + b - 5)^2) →  -- geometric progression after modification
  20 + 2*c - 10 ≥ -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_gp_l3343_334373


namespace NUMINAMATH_CALUDE_point_outside_circle_l3343_334388

theorem point_outside_circle (a b : ℝ) 
  (h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    a * x₁ + b * y₁ = 1 ∧ 
    a * x₂ + b * y₂ = 1 ∧ 
    x₁^2 + y₁^2 = 1 ∧ 
    x₂^2 + y₂^2 = 1) : 
  a^2 + b^2 > 1 := by
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3343_334388


namespace NUMINAMATH_CALUDE_lara_future_age_l3343_334333

def lara_age_7_years_ago : ℕ := 9

def lara_current_age : ℕ := lara_age_7_years_ago + 7

def lara_age_10_years_from_now : ℕ := lara_current_age + 10

theorem lara_future_age : lara_age_10_years_from_now = 26 := by
  sorry

end NUMINAMATH_CALUDE_lara_future_age_l3343_334333


namespace NUMINAMATH_CALUDE_root_sum_squares_l3343_334327

theorem root_sum_squares (p q r : ℝ) : 
  (p^3 - 15*p^2 + 22*p - 8 = 0) → 
  (q^3 - 15*q^2 + 22*q - 8 = 0) → 
  (r^3 - 15*r^2 + 22*r - 8 = 0) → 
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 406 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l3343_334327


namespace NUMINAMATH_CALUDE_length_OP_specific_case_l3343_334329

/-- Given a circle with center O and radius r, and two intersecting chords AB and CD,
    this function calculates the length of OP, where P is the intersection point of the chords. -/
def length_OP (r : ℝ) (chord_AB : ℝ) (chord_CD : ℝ) (midpoint_distance : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a circle with radius 20 and two intersecting chords of lengths 24 and 18,
    if the distance between their midpoints is 10, then the length of OP is approximately 14.8. -/
theorem length_OP_specific_case :
  let r := 20
  let chord_AB := 24
  let chord_CD := 18
  let midpoint_distance := 10
  ∃ ε > 0, |length_OP r chord_AB chord_CD midpoint_distance - 14.8| < ε :=
by sorry

end NUMINAMATH_CALUDE_length_OP_specific_case_l3343_334329


namespace NUMINAMATH_CALUDE_problem_statement_l3343_334394

def f (a x : ℝ) : ℝ := |x - 1| + |x - a|

theorem problem_statement (a : ℝ) (h1 : a > 1) :
  (∀ x, f a x ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2) →
  a = 2 ∧ 
  (∀ x, f a x + |x - 1| ≥ 1 → a ∈ Set.Ici 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3343_334394


namespace NUMINAMATH_CALUDE_uncovered_area_of_squares_l3343_334377

theorem uncovered_area_of_squares (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 10 →
  small_square_side = 4 →
  (large_square_side ^ 2) - 2 * (small_square_side ^ 2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_of_squares_l3343_334377


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l3343_334379

def v : ℕ := (List.range 30).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : (3^k ∣ v) ↔ k ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l3343_334379


namespace NUMINAMATH_CALUDE_smallest_number_l3343_334306

theorem smallest_number : 
  let numbers := [-0.991, -0.981, -0.989, -0.9801, -0.9901]
  ∀ x ∈ numbers, -0.991 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3343_334306


namespace NUMINAMATH_CALUDE_annika_hikes_four_km_l3343_334335

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  flatRate : ℝ  -- Rate on flat terrain in minutes per kilometer
  initialDistance : ℝ  -- Initial distance hiked east in kilometers
  totalTime : ℝ  -- Total time available for the round trip in minutes
  uphillDistance : ℝ  -- Distance of uphill section in kilometers
  uphillRate : ℝ  -- Rate on uphill section in minutes per kilometer
  downhillDistance : ℝ  -- Distance of downhill section in kilometers
  downhillRate : ℝ  -- Rate on downhill section in minutes per kilometer

/-- Calculates the total distance hiked east given the hiking scenario -/
def totalDistanceEast (scenario : HikingScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, Annika will hike 4 km east -/
theorem annika_hikes_four_km : 
  let scenario : HikingScenario := {
    flatRate := 10,
    initialDistance := 2.75,
    totalTime := 45,
    uphillDistance := 0.5,
    uphillRate := 15,
    downhillDistance := 0.5,
    downhillRate := 5
  }
  totalDistanceEast scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_annika_hikes_four_km_l3343_334335


namespace NUMINAMATH_CALUDE_species_assignment_theorem_l3343_334305

/-- Represents the compatibility between species -/
def Compatibility := Fin 8 → Finset (Fin 8)

/-- Theorem stating that it's possible to assign 8 species to 4 cages
    given the compatibility constraints -/
theorem species_assignment_theorem (c : Compatibility)
  (h : ∀ s : Fin 8, (c s).card ≤ 4) :
  ∃ (assignment : Fin 8 → Fin 4),
    ∀ s₁ s₂ : Fin 8, assignment s₁ = assignment s₂ → s₂ ∈ c s₁ := by
  sorry

end NUMINAMATH_CALUDE_species_assignment_theorem_l3343_334305


namespace NUMINAMATH_CALUDE_right_triangle_sin_R_l3343_334330

theorem right_triangle_sin_R (P Q R : ℝ) (h_right_triangle : P + Q + R = π) 
  (h_sin_P : Real.sin P = 3/5) (h_sin_Q : Real.sin Q = 1) : Real.sin R = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_R_l3343_334330


namespace NUMINAMATH_CALUDE_h_greater_than_two_l3343_334309

theorem h_greater_than_two (x : ℝ) (hx : x > 0) : Real.exp x - Real.log x > 2 := by
  sorry

end NUMINAMATH_CALUDE_h_greater_than_two_l3343_334309


namespace NUMINAMATH_CALUDE_fraction_addition_l3343_334392

theorem fraction_addition : (8 : ℚ) / 12 + (7 : ℚ) / 15 = (17 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3343_334392


namespace NUMINAMATH_CALUDE_f_neg_two_value_l3343_334370

-- Define f as a function from R to R
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f (2 * x) + x^2

-- State the theorem
theorem f_neg_two_value (h1 : f 2 = 2) (h2 : ∀ x, g x = -g (-x)) : f (-2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_value_l3343_334370


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l3343_334324

theorem fuel_tank_capacity 
  (fuel_a_ethanol_percentage : ℝ)
  (fuel_b_ethanol_percentage : ℝ)
  (total_ethanol : ℝ)
  (fuel_a_volume : ℝ)
  (h1 : fuel_a_ethanol_percentage = 0.12)
  (h2 : fuel_b_ethanol_percentage = 0.16)
  (h3 : total_ethanol = 28)
  (h4 : fuel_a_volume = 99.99999999999999)
  : ∃ (capacity : ℝ), 
    fuel_a_ethanol_percentage * fuel_a_volume + 
    fuel_b_ethanol_percentage * (capacity - fuel_a_volume) = total_ethanol ∧
    capacity = 200 :=
by sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l3343_334324


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l3343_334312

/-- Represents the length of an edge in centimeters -/
def Edge := ℝ

/-- Represents the volume in cubic centimeters -/
def Volume := ℝ

/-- Given a cuboid with edges a, x, and b, and volume v,
    prove that if a = 4, b = 6, and v = 96, then x = 4 -/
theorem cuboid_edge_length (a x b v : ℝ) :
  a = 4 → b = 6 → v = 96 → v = a * x * b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l3343_334312


namespace NUMINAMATH_CALUDE_exp_iff_gt_l3343_334323

-- Define the exponential function as monotonically increasing on ℝ
axiom exp_monotone : ∀ (x y : ℝ), x < y → Real.exp x < Real.exp y

theorem exp_iff_gt (a b : ℝ) : a > b ↔ Real.exp a > Real.exp b := by
  sorry

end NUMINAMATH_CALUDE_exp_iff_gt_l3343_334323


namespace NUMINAMATH_CALUDE_logarithm_expression_evaluation_l3343_334342

theorem logarithm_expression_evaluation :
  Real.log 5 / Real.log 10 + Real.log 2 / Real.log 10 + (3/5)^0 + Real.log (Real.exp (1/2)) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_evaluation_l3343_334342


namespace NUMINAMATH_CALUDE_total_paintable_area_l3343_334386

/-- Calculate the total paintable area for four bedrooms --/
theorem total_paintable_area (
  num_bedrooms : ℕ)
  (length width height : ℝ)
  (window_area : ℝ) :
  num_bedrooms = 4 →
  length = 14 →
  width = 11 →
  height = 9 →
  window_area = 70 →
  (num_bedrooms : ℝ) * ((2 * (length * height + width * height)) - window_area) = 1520 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l3343_334386


namespace NUMINAMATH_CALUDE_bus_rental_combinations_l3343_334367

theorem bus_rental_combinations :
  let total_people : ℕ := 482
  let large_bus_capacity : ℕ := 42
  let medium_bus_capacity : ℕ := 20
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ =>
    p.1 * large_bus_capacity + p.2 * medium_bus_capacity = total_people
  ) (Finset.product (Finset.range (total_people + 1)) (Finset.range (total_people + 1)))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_bus_rental_combinations_l3343_334367


namespace NUMINAMATH_CALUDE_conditionA_not_necessary_nor_sufficient_l3343_334390

/-- Condition A: The square root of 1 plus sine of theta equals a -/
def conditionA (θ : Real) (a : Real) : Prop :=
  Real.sqrt (1 + Real.sin θ) = a

/-- Condition B: The sine of half theta plus the cosine of half theta equals a -/
def conditionB (θ : Real) (a : Real) : Prop :=
  Real.sin (θ / 2) + Real.cos (θ / 2) = a

/-- Theorem stating that Condition A is neither necessary nor sufficient for Condition B -/
theorem conditionA_not_necessary_nor_sufficient :
  ¬(∀ θ a, conditionB θ a → conditionA θ a) ∧
  ¬(∀ θ a, conditionA θ a → conditionB θ a) :=
sorry

end NUMINAMATH_CALUDE_conditionA_not_necessary_nor_sufficient_l3343_334390


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3343_334317

theorem inequality_system_solution (m : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x ≤ 2 ∧ x > m) → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3343_334317


namespace NUMINAMATH_CALUDE_nancy_coffee_expenditure_l3343_334398

/-- Represents Nancy's coffee consumption and expenditure over a period of time. -/
structure CoffeeConsumption where
  morning_price : ℝ
  afternoon_price : ℝ
  days : ℕ

/-- Calculates the total expenditure on coffee given Nancy's consumption pattern. -/
def total_expenditure (c : CoffeeConsumption) : ℝ :=
  c.days * (c.morning_price + c.afternoon_price)

/-- Theorem stating that Nancy's total expenditure on coffee over 20 days is $110.00. -/
theorem nancy_coffee_expenditure :
  let c : CoffeeConsumption := {
    morning_price := 3.00,
    afternoon_price := 2.50,
    days := 20
  }
  total_expenditure c = 110.00 := by
  sorry

end NUMINAMATH_CALUDE_nancy_coffee_expenditure_l3343_334398


namespace NUMINAMATH_CALUDE_percentage_problem_l3343_334376

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3343_334376


namespace NUMINAMATH_CALUDE_parabola_hyperbola_coincident_foci_l3343_334351

/-- Given a parabola and a hyperbola whose foci coincide, we can determine the focal parameter of the parabola. -/
theorem parabola_hyperbola_coincident_foci (p : ℝ) : 
  p > 0 → -- The focal parameter is positive
  (∃ (x y : ℝ), y^2 = 2*p*x) → -- Equation of the parabola
  (∃ (x y : ℝ), x^2 - y^2/3 = 1) → -- Equation of the hyperbola
  (p/2 = 2) → -- The focus of the parabola coincides with the right focus of the hyperbola
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_coincident_foci_l3343_334351


namespace NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l3343_334332

theorem x_equation_implies_polynomial_value (x : ℝ) (h : x + 1/x = Real.sqrt 3) :
  x^7 - 5*x^5 + x^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l3343_334332


namespace NUMINAMATH_CALUDE_cubic_sum_implies_linear_sum_l3343_334331

theorem cubic_sum_implies_linear_sum (x : ℝ) (h : x^3 + 1/x^3 = 52) : x + 1/x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_implies_linear_sum_l3343_334331


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l3343_334385

theorem square_difference_of_integers (a b : ℕ+) 
  (sum_eq : a + b = 70)
  (diff_eq : a - b = 14) :
  a ^ 2 - b ^ 2 = 980 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l3343_334385


namespace NUMINAMATH_CALUDE_jimin_yuna_problem_l3343_334371

/-- Given a line of students ordered by height, calculate the number of students between two specific positions. -/
def students_between (total : ℕ) (pos1 : ℕ) (pos2 : ℕ) : ℕ :=
  if pos1 > pos2 then pos1 - pos2 - 1 else pos2 - pos1 - 1

theorem jimin_yuna_problem :
  let total_students : ℕ := 32
  let jimin_position : ℕ := 27
  let yuna_position : ℕ := 11
  students_between total_students jimin_position yuna_position = 15 := by
  sorry

end NUMINAMATH_CALUDE_jimin_yuna_problem_l3343_334371


namespace NUMINAMATH_CALUDE_gwen_money_left_l3343_334366

/-- The amount of money Gwen has left after spending some of her birthday money -/
def money_left (received : ℕ) (spent : ℕ) : ℕ :=
  received - spent

/-- Theorem stating that Gwen has 2 dollars left -/
theorem gwen_money_left :
  money_left 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gwen_money_left_l3343_334366


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3343_334308

theorem inequality_system_solution :
  let S := {x : ℝ | 2*x > x + 1 ∧ 4*x - 1 > 7}
  S = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3343_334308


namespace NUMINAMATH_CALUDE_complex_number_problem_l3343_334349

theorem complex_number_problem (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) * (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) →
  (a = -1 ∧ Complex.abs (z + Complex.I) = 3) := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3343_334349


namespace NUMINAMATH_CALUDE_miss_darlington_blueberries_l3343_334391

/-- The number of blueberries in Miss Darlington's basket problem -/
theorem miss_darlington_blueberries 
  (initial_basket : ℕ) 
  (additional_baskets : ℕ) 
  (h1 : initial_basket = 20)
  (h2 : additional_baskets = 9) : 
  initial_basket + additional_baskets * initial_basket = 200 := by
  sorry

end NUMINAMATH_CALUDE_miss_darlington_blueberries_l3343_334391


namespace NUMINAMATH_CALUDE_infinitely_many_integers_with_zero_padic_valuation_mod_d_l3343_334396

/-- The p-adic valuation of n! -/
def ν (p : Nat) (n : Nat) : Nat := sorry

theorem infinitely_many_integers_with_zero_padic_valuation_mod_d 
  (d : Nat) (primes : Finset Nat) (h_d : d > 0) (h_primes : ∀ p ∈ primes, Nat.Prime p) :
  ∃ (S : Set Nat), Set.Infinite S ∧ 
    ∀ n ∈ S, ∀ p ∈ primes, (ν p n) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_integers_with_zero_padic_valuation_mod_d_l3343_334396


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3343_334307

/-- Proves that if an item is sold for 1260 with a 16% loss, its cost price was 1500 --/
theorem cost_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1260)
  (h2 : loss_percentage = 16) : 
  (selling_price / (1 - loss_percentage / 100)) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3343_334307


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3343_334318

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x - 1) * (x - 3) < 0} = {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3343_334318


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l3343_334334

theorem smallest_x_for_equation : 
  ∃ (x : ℕ+), x = 9 ∧ 
  (∃ (y : ℕ+), (9 : ℚ) / 10 = (y : ℚ) / (151 + x)) ∧ 
  (∀ (x' : ℕ+), x' < x → 
    ¬∃ (y : ℕ+), (9 : ℚ) / 10 = (y : ℚ) / (151 + x')) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l3343_334334


namespace NUMINAMATH_CALUDE_system_one_solution_set_system_two_solution_set_l3343_334343

-- System 1
theorem system_one_solution_set :
  {x : ℝ | 3*x > x + 6 ∧ (1/2)*x < -x + 5} = {x : ℝ | 3 < x ∧ x < 10/3} := by sorry

-- System 2
theorem system_two_solution_set :
  {x : ℝ | 2*x - 1 < 5 - 2*(x-1) ∧ (3+5*x)/3 > 1} = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_system_one_solution_set_system_two_solution_set_l3343_334343


namespace NUMINAMATH_CALUDE_overlap_area_is_half_unit_l3343_334340

-- Define the grid and triangles
def Grid := Fin 3 × Fin 3

def Triangle1 : Set Grid := {(0, 2), (2, 0), (0, 0)}
def Triangle2 : Set Grid := {(2, 2), (0, 0), (1, 0)}

-- Define the area of overlap
def overlap_area (t1 t2 : Set Grid) : ℝ :=
  sorry

-- Theorem statement
theorem overlap_area_is_half_unit :
  overlap_area Triangle1 Triangle2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_half_unit_l3343_334340


namespace NUMINAMATH_CALUDE_min_horizontal_distance_l3343_334393

/-- Parabola equation: y = x^2 - x - 2 -/
def parabola (x : ℝ) : ℝ := x^2 - x - 2

/-- Point P has y-coordinate 10 -/
def point_P : Set ℝ := {x : ℝ | parabola x = 10}

/-- Point Q has y-coordinate 0 -/
def point_Q : Set ℝ := {x : ℝ | parabola x = 0}

/-- The horizontal distance between two x-coordinates -/
def horizontal_distance (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem min_horizontal_distance :
  ∃ (p q : ℝ), p ∈ point_P ∧ q ∈ point_Q ∧
  ∀ (p' q' : ℝ), p' ∈ point_P → q' ∈ point_Q →
  horizontal_distance p q ≤ horizontal_distance p' q' ∧
  horizontal_distance p q = 2 :=
sorry

end NUMINAMATH_CALUDE_min_horizontal_distance_l3343_334393


namespace NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l3343_334361

/-- Represents a two-digit positive integer -/
structure TwoDigitNumber where
  value : Nat
  is_two_digit : 10 ≤ value ∧ value ≤ 99

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The main theorem -/
theorem smallest_fourth_lucky_number :
  ∀ (n : TwoDigitNumber),
    (sumOfDigits 46 + sumOfDigits 24 + sumOfDigits 85 + sumOfDigits n.value = 
     (46 + 24 + 85 + n.value) / 4) →
    n.value ≥ 59 := by
  sorry

#eval sumOfDigits 59  -- Expected output: 14
#eval (46 + 24 + 85 + 59) / 4  -- Expected output: 53
#eval sumOfDigits 46 + sumOfDigits 24 + sumOfDigits 85 + sumOfDigits 59  -- Expected output: 53

end NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l3343_334361


namespace NUMINAMATH_CALUDE_total_shaded_area_is_one_third_l3343_334336

/-- Represents the fractional area shaded in each step of the square division pattern. -/
def shadedAreaSequence : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => (1/4) * shadedAreaSequence n

/-- The sum of the infinite geometric series representing the total shaded area. -/
noncomputable def totalShadedArea : ℚ := ∑' n, shadedAreaSequence n

/-- Theorem stating that the total shaded area is equal to 1/3. -/
theorem total_shaded_area_is_one_third :
  totalShadedArea = 1/3 := by sorry

end NUMINAMATH_CALUDE_total_shaded_area_is_one_third_l3343_334336


namespace NUMINAMATH_CALUDE_rectangle_area_l3343_334362

/-- Proves that the area of a rectangle is 432 square meters, given that its length is thrice its breadth and its perimeter is 96 meters. -/
theorem rectangle_area (b : ℝ) (l : ℝ) : 
  l = 3 * b →                  -- Length is thrice the breadth
  2 * (l + b) = 96 →           -- Perimeter is 96 meters
  l * b = 432 := by            -- Area is 432 square meters
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3343_334362


namespace NUMINAMATH_CALUDE_solve_equation_l3343_334378

/-- Define the determinant-like operation for four rational numbers -/
def det (a b c d : ℚ) : ℚ := a * d - b * c

/-- Theorem stating that given the condition, x must equal 3 -/
theorem solve_equation : ∃ x : ℚ, det (2*x) (-4) x 1 = 18 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3343_334378


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_six_l3343_334383

theorem no_solution_iff_m_eq_neg_six (m : ℝ) :
  (∀ x : ℝ, x ≠ -2 → (x - 3) / (x + 2) + (x + 1) / (x + 2) ≠ m / (x + 2)) ↔ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_six_l3343_334383


namespace NUMINAMATH_CALUDE_black_squares_in_29th_row_l3343_334375

/-- Represents the number of squares in a row of the pattern -/
def squaresInRow (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- Represents the number of black squares in a row of the pattern -/
def blackSquaresInRow (n : ℕ) : ℕ := (squaresInRow n - 1) / 2

/-- Theorem stating that the 29th row contains 28 black squares -/
theorem black_squares_in_29th_row : blackSquaresInRow 29 = 28 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_in_29th_row_l3343_334375


namespace NUMINAMATH_CALUDE_triangle_side_length_l3343_334348

/-- Given a triangle ABC with the following properties:
  * The product of sides a and b is 60√3
  * The sine of angle B equals the sine of angle C
  * The area of the triangle is 15√3
  This theorem states that the length of side b is 2√15 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a * b = 60 * Real.sqrt 3 →
  Real.sin B = Real.sin C →
  (1/2) * a * b * Real.sin C = 15 * Real.sqrt 3 →
  b = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3343_334348


namespace NUMINAMATH_CALUDE_smallest_x_for_1680x_perfect_cube_l3343_334381

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_x_for_1680x_perfect_cube : 
  (∀ x : ℕ, x > 0 ∧ x < 44100 → ¬(is_perfect_cube (1680 * x))) ∧
  (is_perfect_cube (1680 * 44100)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_1680x_perfect_cube_l3343_334381


namespace NUMINAMATH_CALUDE_matrix_power_4_l3343_334389

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_4 : A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l3343_334389


namespace NUMINAMATH_CALUDE_building_shadow_length_l3343_334384

/-- Given a flagstaff and a building under similar conditions, prove that the length of the shadow
    cast by the building is 28.75 m. -/
theorem building_shadow_length
  (flagstaff_height : ℝ)
  (flagstaff_shadow : ℝ)
  (building_height : ℝ)
  (h1 : flagstaff_height = 17.5)
  (h2 : flagstaff_shadow = 40.25)
  (h3 : building_height = 12.5)
  : (building_height * flagstaff_shadow) / flagstaff_height = 28.75 := by
  sorry

end NUMINAMATH_CALUDE_building_shadow_length_l3343_334384


namespace NUMINAMATH_CALUDE_height_to_hypotenuse_l3343_334380

theorem height_to_hypotenuse (a b c h : ℝ) : 
  a = 6 → b = 8 → c = 10 → a^2 + b^2 = c^2 → (a * b) / 2 = (c * h) / 2 → h = 4.8 :=
by sorry

end NUMINAMATH_CALUDE_height_to_hypotenuse_l3343_334380


namespace NUMINAMATH_CALUDE_town_population_is_300_l3343_334325

/-- The number of females attending the meeting -/
def females_attending : ℕ := 50

/-- The number of males attending the meeting -/
def males_attending : ℕ := 2 * females_attending

/-- The total number of people attending the meeting -/
def total_attending : ℕ := females_attending + males_attending

/-- The total population of the town -/
def town_population : ℕ := 2 * total_attending

theorem town_population_is_300 : town_population = 300 := by
  sorry

end NUMINAMATH_CALUDE_town_population_is_300_l3343_334325


namespace NUMINAMATH_CALUDE_alex_age_l3343_334302

theorem alex_age (alex_age precy_age : ℕ) : 
  (alex_age + 3 = 3 * (precy_age + 3)) →
  (alex_age - 1 = 7 * (precy_age - 1)) →
  alex_age = 15 := by
sorry

end NUMINAMATH_CALUDE_alex_age_l3343_334302


namespace NUMINAMATH_CALUDE_theta_range_l3343_334345

theorem theta_range (θ : Real) : 
  θ ∈ Set.Icc 0 π ∧ 
  (∀ x ∈ Set.Icc (-1) 0, x^2 * Real.cos θ + (x+1)^2 * Real.sin θ + x^2 + x > 0) →
  θ ∈ Set.Ioo (π/12) (5*π/12) := by
sorry

end NUMINAMATH_CALUDE_theta_range_l3343_334345


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3343_334356

theorem quadratic_root_difference (x : ℝ) : 
  (x^2 - 5*x + 6 = 0) → 
  ∃ r₁ r₂ : ℝ, (r₁ - r₂ = 1 ∨ r₂ - r₁ = 1) ∧ (x = r₁ ∨ x = r₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3343_334356


namespace NUMINAMATH_CALUDE_inequality_proof_l3343_334350

theorem inequality_proof (p q r : ℝ) (n : ℕ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0) 
  (h_product : p * q * r = 1) : 
  1 / (p^n + q^n + 1) + 1 / (q^n + r^n + 1) + 1 / (r^n + p^n + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3343_334350


namespace NUMINAMATH_CALUDE_vegetable_ghee_ratio_l3343_334372

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 950

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 850

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3640

/-- The volume of brand 'a' in the mixture -/
def volume_a : ℝ := 2.4

/-- The volume of brand 'b' in the mixture -/
def volume_b : ℝ := 1.6

/-- Theorem stating that the ratio of volumes of brand 'a' to brand 'b' is 1.5:1 -/
theorem vegetable_ghee_ratio :
  volume_a / volume_b = 1.5 ∧
  volume_a + volume_b = total_volume ∧
  weight_a * volume_a + weight_b * volume_b = total_weight :=
by sorry

end NUMINAMATH_CALUDE_vegetable_ghee_ratio_l3343_334372


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3343_334320

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + (4*m + 1)*x₁ + m = 0) ∧
  (x₂^2 + (4*m + 1)*x₂ + m = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3343_334320


namespace NUMINAMATH_CALUDE_student_height_probability_l3343_334300

theorem student_height_probability (p_less_160 p_between_160_175 : ℝ) :
  p_less_160 = 0.2 →
  p_between_160_175 = 0.5 →
  1 - p_less_160 - p_between_160_175 = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_student_height_probability_l3343_334300


namespace NUMINAMATH_CALUDE_park_length_l3343_334363

/-- A rectangular park with given dimensions and tree density. -/
structure Park where
  width : ℝ
  length : ℝ
  treeCount : ℕ
  treeDensity : ℝ

/-- The park satisfies the given conditions. -/
def validPark (p : Park) : Prop :=
  p.width = 2000 ∧
  p.treeCount = 100000 ∧
  p.treeDensity = 1 / 20

/-- The theorem stating the length of the park given the conditions. -/
theorem park_length (p : Park) (h : validPark p) : p.length = 1000 := by
  sorry

#check park_length

end NUMINAMATH_CALUDE_park_length_l3343_334363


namespace NUMINAMATH_CALUDE_minimum_words_to_learn_l3343_334382

theorem minimum_words_to_learn (total_words : ℕ) (required_percentage : ℚ) : 
  total_words = 600 → required_percentage = 90 / 100 → 
  ∃ (min_words : ℕ), min_words * 100 ≥ total_words * required_percentage ∧
    ∀ (n : ℕ), n * 100 ≥ total_words * required_percentage → n ≥ min_words :=
by sorry

end NUMINAMATH_CALUDE_minimum_words_to_learn_l3343_334382
