import Mathlib

namespace NUMINAMATH_CALUDE_broken_marbles_count_l694_69436

/-- The number of marbles in the first set -/
def set1_total : ℕ := 50

/-- The percentage of broken marbles in the first set -/
def set1_broken_percent : ℚ := 1/10

/-- The number of marbles in the second set -/
def set2_total : ℕ := 60

/-- The percentage of broken marbles in the second set -/
def set2_broken_percent : ℚ := 1/5

/-- The total number of broken marbles in both sets -/
def total_broken_marbles : ℕ := 17

theorem broken_marbles_count : 
  ⌊(set1_total : ℚ) * set1_broken_percent⌋ + ⌊(set2_total : ℚ) * set2_broken_percent⌋ = total_broken_marbles :=
by sorry

end NUMINAMATH_CALUDE_broken_marbles_count_l694_69436


namespace NUMINAMATH_CALUDE_equation_rewrite_l694_69406

theorem equation_rewrite (x y : ℝ) : 
  (3 * x + y = 17) → (y = -3 * x + 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l694_69406


namespace NUMINAMATH_CALUDE_divisibility_condition_l694_69437

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l694_69437


namespace NUMINAMATH_CALUDE_gcd_840_1764_l694_69499

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l694_69499


namespace NUMINAMATH_CALUDE_smores_graham_crackers_per_smore_l694_69435

theorem smores_graham_crackers_per_smore (total_graham_crackers : ℕ) 
  (initial_marshmallows : ℕ) (additional_marshmallows : ℕ) :
  total_graham_crackers = 48 →
  initial_marshmallows = 6 →
  additional_marshmallows = 18 →
  (total_graham_crackers / (initial_marshmallows + additional_marshmallows) : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_smores_graham_crackers_per_smore_l694_69435


namespace NUMINAMATH_CALUDE_bottles_per_case_l694_69465

/-- Given a company that produces bottles of water and uses cases to hold them,
    this theorem proves the number of bottles per case. -/
theorem bottles_per_case
  (total_bottles : ℕ)
  (total_cases : ℕ)
  (h1 : total_bottles = 60000)
  (h2 : total_cases = 12000)
  : total_bottles / total_cases = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_case_l694_69465


namespace NUMINAMATH_CALUDE_prime_sequence_l694_69440

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) :
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sequence_l694_69440


namespace NUMINAMATH_CALUDE_max_rectangular_pen_area_l694_69464

/-- Given 50 feet of fencing with 5 feet used for a non-enclosing gate,
    the maximum area of a rectangular pen enclosed by the remaining fencing
    is 126.5625 square feet. -/
theorem max_rectangular_pen_area : 
  ∀ (width height : ℝ),
    width > 0 → height > 0 →
    width + height = (50 - 5) / 2 →
    width * height ≤ 126.5625 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangular_pen_area_l694_69464


namespace NUMINAMATH_CALUDE_geometry_theorem_l694_69443

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lines_parallel : Line → Line → Prop)

-- Define non-coincidence for lines and planes
variable (non_coincident_lines : Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_theorem 
  (m n : Line) (α β : Plane)
  (h_non_coincident_lines : non_coincident_lines m n)
  (h_non_coincident_planes : non_coincident_planes α β) :
  (subset m β ∧ parallel α β → line_parallel m α) ∧
  (perpendicular m α ∧ perpendicular n β ∧ parallel α β → lines_parallel m n) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l694_69443


namespace NUMINAMATH_CALUDE_inequality_solution_l694_69457

/-- The solution set of the inequality x^2 - (a + a^2)x + a^3 > 0 for any real number a -/
def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 ∨ a > 1 then {x | x > a^2 ∨ x < a}
  else if a = 0 then {x | x ≠ 0}
  else if 0 < a ∧ a < 1 then {x | x > a ∨ x < a^2}
  else {x | x ≠ 1}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | x^2 - (a + a^2) * x + a^3 > 0} = solution_set a :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l694_69457


namespace NUMINAMATH_CALUDE_batsman_highest_score_l694_69431

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46) 
  (h1 : average = 60) 
  (h2 : score_difference = 190) 
  (h3 : average_excluding_extremes = 58) : 
  ∃ (highest_score lowest_score : ℕ), 
    highest_score - lowest_score = score_difference ∧ 
    (total_innings : ℚ) * average = (total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + lowest_score ∧
    highest_score = 199 :=
by sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l694_69431


namespace NUMINAMATH_CALUDE_face_value_calculation_l694_69409

/-- Given a banker's discount and true discount, calculate the face value (sum due) -/
def calculate_face_value (bankers_discount true_discount : ℚ) : ℚ :=
  (bankers_discount * true_discount) / (bankers_discount - true_discount)

/-- Theorem stating that given a banker's discount of 144 and a true discount of 120, the face value is 840 -/
theorem face_value_calculation (bankers_discount true_discount : ℚ) 
  (h1 : bankers_discount = 144)
  (h2 : true_discount = 120) :
  calculate_face_value bankers_discount true_discount = 840 := by
sorry

end NUMINAMATH_CALUDE_face_value_calculation_l694_69409


namespace NUMINAMATH_CALUDE_equation_solution_l694_69487

theorem equation_solution : 
  ∃! r : ℚ, (r + 9) / (r - 3) = (r - 2) / (r + 5) ∧ r = -39 / 19 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l694_69487


namespace NUMINAMATH_CALUDE_product_uvw_l694_69412

theorem product_uvw (a c x y : ℝ) (u v w : ℤ) : 
  (a^8*x*y - a^7*y - a^6*x = a^5*(c^5 - 1)) ∧ 
  ((a^u*x - a^v)*(a^w*y - a^3) = a^5*c^5) →
  u*v*w = 6 :=
by sorry

end NUMINAMATH_CALUDE_product_uvw_l694_69412


namespace NUMINAMATH_CALUDE_statement_A_statement_C_statement_D_l694_69482

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 1)^3 - a*x - b + 1

-- Define the function g
def g (a b x : ℝ) : ℝ := f a b x - 3*x + a*x + b

-- Statement A
theorem statement_A (a b : ℝ) :
  a = 3 → (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b x = 0 ∧ f a b y = 0 ∧ f a b z = 0) →
  -4 < b ∧ b < 0 := by sorry

-- Statement C
theorem statement_C (a b m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ∃ k₁ k₂ k₃ : ℝ, 
      k₁ * (2 - x) + m = g a b x ∧
      k₂ * (2 - y) + m = g a b y ∧
      k₃ * (2 - z) + m = g a b z) →
  -5 < m ∧ m < -4 := by sorry

-- Statement D
theorem statement_D (a b : ℝ) :
  (∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧
    (∀ x : ℝ, f a b x₀ ≤ f a b x ∨ f a b x₀ ≥ f a b x) ∧
    f a b x₀ = f a b x₁) →
  ∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ x₁ + 2*x₀ = 3 := by sorry

end NUMINAMATH_CALUDE_statement_A_statement_C_statement_D_l694_69482


namespace NUMINAMATH_CALUDE_kyle_paper_delivery_l694_69471

/-- The number of papers Kyle delivers each week -/
def weekly_papers (
  regular_houses : ℕ
  ) (sunday_skip : ℕ) (sunday_extra : ℕ) : ℕ :=
  (6 * regular_houses) + (regular_houses - sunday_skip + sunday_extra)

theorem kyle_paper_delivery :
  weekly_papers 100 10 30 = 720 := by
  sorry

end NUMINAMATH_CALUDE_kyle_paper_delivery_l694_69471


namespace NUMINAMATH_CALUDE_working_days_is_25_l694_69486

/-- Calculates the number of working days in a month based on given employee wages and tax rates -/
def calculate_working_days (
  num_warehouse_workers : ℕ)
  (num_managers : ℕ)
  (warehouse_hourly_rate : ℚ)
  (manager_hourly_rate : ℚ)
  (hours_per_day : ℕ)
  (tax_rate : ℚ)
  (total_monthly_cost : ℚ) : ℚ :=
  let daily_wage := num_warehouse_workers * warehouse_hourly_rate * hours_per_day +
                    num_managers * manager_hourly_rate * hours_per_day
  let monthly_wage_before_tax := total_monthly_cost / (1 + tax_rate)
  monthly_wage_before_tax / daily_wage

/-- Theorem stating that the number of working days is 25 given the problem conditions -/
theorem working_days_is_25 :
  calculate_working_days 4 2 15 20 8 (1/10) 22000 = 25 := by sorry

end NUMINAMATH_CALUDE_working_days_is_25_l694_69486


namespace NUMINAMATH_CALUDE_ellipse_equation_l694_69462

/-- The standard equation of an ellipse with given major axis and eccentricity -/
theorem ellipse_equation (major_axis : ℝ) (eccentricity : ℝ) :
  major_axis = 8 ∧ eccentricity = 3/4 →
  ∃ (x y : ℝ), (x^2/16 + y^2/7 = 1) ∨ (x^2/7 + y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l694_69462


namespace NUMINAMATH_CALUDE_purely_imaginary_x_equals_one_l694_69450

-- Define a complex number
def complex_number (x : ℝ) : ℂ := (x^2 - 1) + (x + 1) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem purely_imaginary_x_equals_one :
  ∀ x : ℝ, is_purely_imaginary (complex_number x) → x = 1 :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_x_equals_one_l694_69450


namespace NUMINAMATH_CALUDE_sum_of_powers_l694_69496

theorem sum_of_powers (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^10 + ω^12 + ω^14 + ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l694_69496


namespace NUMINAMATH_CALUDE_nina_reading_homework_multiplier_l694_69473

-- Define the given conditions
def ruby_math_homework : ℕ := 6
def ruby_reading_homework : ℕ := 2
def nina_total_homework : ℕ := 48
def nina_math_homework_multiplier : ℕ := 4

-- Define Nina's math homework
def nina_math_homework : ℕ := ruby_math_homework * (nina_math_homework_multiplier + 1)

-- Define Nina's reading homework
def nina_reading_homework : ℕ := nina_total_homework - nina_math_homework

-- Theorem to prove
theorem nina_reading_homework_multiplier :
  nina_reading_homework / ruby_reading_homework = 9 := by
  sorry


end NUMINAMATH_CALUDE_nina_reading_homework_multiplier_l694_69473


namespace NUMINAMATH_CALUDE_clothing_cost_price_l694_69434

theorem clothing_cost_price (original_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) (cost_price : ℝ) : 
  original_price = 132 →
  discount_rate = 0.1 →
  profit_rate = 0.1 →
  original_price * (1 - discount_rate) = cost_price * (1 + profit_rate) →
  cost_price = 108 := by
sorry

end NUMINAMATH_CALUDE_clothing_cost_price_l694_69434


namespace NUMINAMATH_CALUDE_quadratic_function_value_l694_69433

/-- Given a quadratic function f(x) = x^2 + px + q where f(3) = 0 and f(2) = 0, 
    prove that f(0) = 6. -/
theorem quadratic_function_value (p q : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + p*x + q) 
  (h2 : f 3 = 0) 
  (h3 : f 2 = 0) : 
  f 0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l694_69433


namespace NUMINAMATH_CALUDE_sum_of_specific_triangles_l694_69458

/-- Triangle operation that takes three integers and returns their sum minus the last -/
def triangle_op (a b c : ℤ) : ℤ := a + b - c

/-- The sum of two triangle operations -/
def sum_of_triangles (a₁ b₁ c₁ a₂ b₂ c₂ : ℤ) : ℤ :=
  triangle_op a₁ b₁ c₁ + triangle_op a₂ b₂ c₂

/-- Theorem stating that the sum of triangle operations (1,3,4) and (2,5,6) is 1 -/
theorem sum_of_specific_triangles :
  sum_of_triangles 1 3 4 2 5 6 = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_triangles_l694_69458


namespace NUMINAMATH_CALUDE_sum_of_digits_l694_69426

-- Define the variables as natural numbers
variable (a b c d : ℕ)

-- Define the conditions
axiom different_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom sum_hundreds_ones : a + c = 10
axiom sum_tens : b + c = 8
axiom sum_hundreds : a + d = 11
axiom result_sum : 100 * a + 10 * b + c + 100 * d + 10 * c + a = 1180

-- State the theorem
theorem sum_of_digits : a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l694_69426


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_square_integer_l694_69418

theorem sum_and_reciprocal_square_integer (a : ℝ) (h1 : a ≠ 0) (h2 : ∃ k : ℤ, a + 1 / a = k) :
  ∃ m : ℤ, a^2 + 1 / a^2 = m :=
sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_square_integer_l694_69418


namespace NUMINAMATH_CALUDE_remainder_3211_103_l694_69451

theorem remainder_3211_103 : 3211 % 103 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3211_103_l694_69451


namespace NUMINAMATH_CALUDE_ellipse_and_hyperbola_properties_l694_69459

/-- An ellipse with foci on the y-axis -/
structure Ellipse where
  major_axis : ℝ
  minor_axis : ℝ
  foci_on_y_axis : Bool

/-- A hyperbola with foci on the y-axis -/
structure Hyperbola where
  real_axis : ℝ
  imaginary_axis : ℝ
  foci_on_y_axis : Bool

/-- Given ellipse properties, prove its equation, foci coordinates, eccentricity, and related hyperbola equation -/
theorem ellipse_and_hyperbola_properties (e : Ellipse) 
    (h1 : e.major_axis = 10) 
    (h2 : e.minor_axis = 8) 
    (h3 : e.foci_on_y_axis = true) : 
  (∃ (x y : ℝ), x^2/16 + y^2/25 = 1) ∧ 
  (∃ (f1 f2 : ℝ × ℝ), f1 = (0, -3) ∧ f2 = (0, 3)) ∧
  (3/5 : ℝ) = (5^2 - 4^2).sqrt / 5 ∧
  (∃ (h : Hyperbola), h.real_axis = 3 ∧ h.imaginary_axis = 4 ∧ h.foci_on_y_axis = true ∧
    ∃ (x y : ℝ), y^2/9 - x^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_hyperbola_properties_l694_69459


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l694_69474

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- The main theorem -/
theorem smallest_two_digit_prime_with_composite_reverse :
  ∀ p : ℕ, is_two_digit p → Nat.Prime p →
    (∀ q : ℕ, is_two_digit q → Nat.Prime q → q < p →
      Nat.Prime (reverse_digits q)) →
    ¬Nat.Prime (reverse_digits p) →
    p = 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l694_69474


namespace NUMINAMATH_CALUDE_right_triangle_condition_l694_69401

theorem right_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a * Real.cos C + c * Real.cos A = b * Real.sin B) →
  (a * Real.sin A = b * Real.sin B) →
  (b * Real.sin B = c * Real.sin C) →
  (B = π / 2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l694_69401


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l694_69400

theorem complex_fraction_simplification :
  1 + 1 / (1 + 1 / (2 + 2)) = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l694_69400


namespace NUMINAMATH_CALUDE_race_distance_l694_69448

theorem race_distance (a_time b_time : ℝ) (beat_distance : ℝ) : 
  a_time = 28 → b_time = 32 → beat_distance = 16 → 
  ∃ d : ℝ, d = 128 ∧ d / a_time * b_time = d - beat_distance :=
by
  sorry

end NUMINAMATH_CALUDE_race_distance_l694_69448


namespace NUMINAMATH_CALUDE_distance_after_ten_reflections_l694_69421

/-- Represents a circular billiard table with a ball's trajectory -/
structure BilliardTable where
  radius : ℝ
  p_distance : ℝ  -- Distance of point P from the center
  reflection_angle : ℝ  -- Angle of reflection

/-- Calculates the distance between P and the ball's position after n reflections -/
noncomputable def distance_after_reflections (table : BilliardTable) (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the distance after 10 reflections for the given table -/
theorem distance_after_ten_reflections (table : BilliardTable) 
  (h1 : table.radius = 1)
  (h2 : table.p_distance = 0.4)
  (h3 : table.reflection_angle = Real.arcsin ((Real.sqrt 57 - 5) / 8)) :
  ∃ (ε : ℝ), abs (distance_after_reflections table 10 - 0.0425) < ε ∧ ε > 0 ∧ ε < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_distance_after_ten_reflections_l694_69421


namespace NUMINAMATH_CALUDE_gcd_three_numbers_l694_69460

theorem gcd_three_numbers (a b c : ℕ) (h1 : Nat.gcd a b = 18) (h2 : Nat.gcd b c = 18) :
  Nat.gcd a (Nat.gcd b c) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_three_numbers_l694_69460


namespace NUMINAMATH_CALUDE_randy_pictures_l694_69423

theorem randy_pictures (peter_pictures quincy_pictures randy_pictures total_pictures : ℕ) :
  peter_pictures = 8 →
  quincy_pictures = peter_pictures + 20 →
  total_pictures = 41 →
  total_pictures = peter_pictures + quincy_pictures + randy_pictures →
  randy_pictures = 5 := by
sorry

end NUMINAMATH_CALUDE_randy_pictures_l694_69423


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l694_69446

theorem interest_rate_calculation (initial_amount : ℝ) (final_amount : ℝ) 
  (second_year_rate : ℝ) (first_year_rate : ℝ) : 
  initial_amount = 6000 ∧ 
  final_amount = 6552 ∧ 
  second_year_rate = 0.05 ∧
  first_year_rate = 0.04 →
  final_amount = initial_amount + 
    (initial_amount * first_year_rate) + 
    ((initial_amount + initial_amount * first_year_rate) * second_year_rate) :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l694_69446


namespace NUMINAMATH_CALUDE_least_possible_value_x_l694_69403

theorem least_possible_value_x (a b x : ℕ) 
  (h1 : x = 2 * a^5)
  (h2 : x = 3 * b^2)
  (h3 : 0 < a)
  (h4 : 0 < b) :
  ∀ y : ℕ, (∃ c d : ℕ, y = 2 * c^5 ∧ y = 3 * d^2 ∧ 0 < c ∧ 0 < d) → x ≤ y ∧ x = 15552 := by
  sorry

#check least_possible_value_x

end NUMINAMATH_CALUDE_least_possible_value_x_l694_69403


namespace NUMINAMATH_CALUDE_fewest_tiles_needed_l694_69475

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of a single tile -/
def tileDimensions : Dimensions := ⟨6, 2⟩

/-- The dimensions of the rectangular region in feet -/
def regionDimensionsFeet : Dimensions := ⟨3, 6⟩

/-- The dimensions of the rectangular region in inches -/
def regionDimensionsInches : Dimensions :=
  ⟨feetToInches regionDimensionsFeet.length, feetToInches regionDimensionsFeet.width⟩

/-- Calculates the number of tiles needed to cover a given area -/
def tilesNeeded (regionArea tileArea : ℕ) : ℕ :=
  (regionArea + tileArea - 1) / tileArea

theorem fewest_tiles_needed :
  tilesNeeded (area regionDimensionsInches) (area tileDimensions) = 216 := by
  sorry

end NUMINAMATH_CALUDE_fewest_tiles_needed_l694_69475


namespace NUMINAMATH_CALUDE_chess_tournament_games_l694_69432

theorem chess_tournament_games (n : ℕ) (h : n = 8) : 
  n * (n - 1) = 56 ∧ 2 * (n * (n - 1)) = 112 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l694_69432


namespace NUMINAMATH_CALUDE_umbrella_probability_l694_69491

theorem umbrella_probability (p_forget : ℚ) (h1 : p_forget = 5/8) :
  1 - p_forget = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_umbrella_probability_l694_69491


namespace NUMINAMATH_CALUDE_average_gas_mileage_l694_69467

def total_distance : ℝ := 300
def sedan_efficiency : ℝ := 25
def truck_efficiency : ℝ := 15

theorem average_gas_mileage : 
  let sedan_distance := total_distance / 2
  let truck_distance := total_distance / 2
  let sedan_fuel := sedan_distance / sedan_efficiency
  let truck_fuel := truck_distance / truck_efficiency
  let total_fuel := sedan_fuel + truck_fuel
  (total_distance / total_fuel) = 18.75 := by sorry

end NUMINAMATH_CALUDE_average_gas_mileage_l694_69467


namespace NUMINAMATH_CALUDE_snake_sum_squares_geq_n_squared_l694_69402

/-- Represents a snake (python or anaconda) in the grid -/
structure Snake where
  length : ℕ
  is_python : Bool

/-- Represents the n×n grid with snakes -/
structure Grid (n : ℕ) where
  snakes : List Snake
  valid : Bool

/-- The sum of squares of snake lengths -/
def sum_of_squares (grid : Grid n) : ℕ :=
  grid.snakes.map (λ s => s.length * s.length) |>.sum

/-- The theorem to be proved -/
theorem snake_sum_squares_geq_n_squared (n : ℕ) (grid : Grid n) 
  (h1 : n > 0)
  (h2 : grid.valid)
  (h3 : grid.snakes.length > 0) :
  sum_of_squares grid ≥ n * n := by
  sorry


end NUMINAMATH_CALUDE_snake_sum_squares_geq_n_squared_l694_69402


namespace NUMINAMATH_CALUDE_line_passes_through_point_line_has_slope_line_properties_l694_69476

/-- A line in the xy-plane defined by the equation y = k(x+1) for some real k -/
structure Line where
  k : ℝ

/-- The point (-1, 0) in the xy-plane -/
def point : ℝ × ℝ := (-1, 0)

/-- Checks if a given point (x, y) lies on the line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.k * (p.1 + 1)

/-- States that the line passes through the point (-1, 0) -/
theorem line_passes_through_point (l : Line) : l.contains point := by sorry

/-- States that the line has a defined slope -/
theorem line_has_slope (l : Line) : ∃ m : ℝ, ∀ x y : ℝ, y = m * x + l.k := by sorry

/-- Main theorem combining both properties -/
theorem line_properties (l : Line) : l.contains point ∧ ∃ m : ℝ, ∀ x y : ℝ, y = m * x + l.k := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_line_has_slope_line_properties_l694_69476


namespace NUMINAMATH_CALUDE_third_smallest_sum_is_four_l694_69483

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 10) % 10 = 1

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + (n % 10)

theorem third_smallest_sum_is_four :
  ∃ (n : ℕ), is_valid_number n ∧
  (∀ m, is_valid_number m → m < n) ∧
  (∃ k₁ k₂, is_valid_number k₁ ∧ is_valid_number k₂ ∧ k₁ < k₂ ∧ k₂ < n) ∧
  digit_sum n = 4 :=
sorry

end NUMINAMATH_CALUDE_third_smallest_sum_is_four_l694_69483


namespace NUMINAMATH_CALUDE_gilda_marbles_theorem_l694_69425

/-- The percentage of marbles Gilda has left after giving away to her friends and brother -/
def gildasRemainingMarbles : ℝ :=
  let initialMarbles := 100
  let afterPedro := initialMarbles * (1 - 0.30)
  let afterEbony := afterPedro * (1 - 0.20)
  let afterCarlos := afterEbony * (1 - 0.15)
  let afterJimmy := afterCarlos * (1 - 0.10)
  afterJimmy

/-- Theorem stating that Gilda has approximately 43% of her original marbles left -/
theorem gilda_marbles_theorem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |gildasRemainingMarbles - 43| < ε :=
sorry

end NUMINAMATH_CALUDE_gilda_marbles_theorem_l694_69425


namespace NUMINAMATH_CALUDE_athlete_weight_problem_l694_69442

theorem athlete_weight_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 42 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  ∃ k₁ k₂ k₃ : ℕ, a = 5 * k₁ ∧ b = 5 * k₂ ∧ c = 5 * k₃ →
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_athlete_weight_problem_l694_69442


namespace NUMINAMATH_CALUDE_solve_for_b_l694_69470

theorem solve_for_b (y : ℝ) (b : ℝ) (h1 : y > 0) 
  (h2 : (70/100) * y = (8*y) / b + (3*y) / 10) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l694_69470


namespace NUMINAMATH_CALUDE_problem_solution_l694_69477

-- Define the function f
def f (x : ℝ) : ℝ := 6 * x^2 + x - 1

-- State the theorem
theorem problem_solution (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : f (Real.sin α) = 0) :
  Real.sin α = 1 / 3 ∧
  (Real.tan (π + α) * Real.cos (-α)) / (Real.cos (π / 2 - α) * Real.sin (π - α)) = 3 ∧
  Real.sin (α + π / 6) = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l694_69477


namespace NUMINAMATH_CALUDE_urn_problem_l694_69424

/-- Given two urns with different compositions of colored balls, 
    prove that the number of blue balls in the second urn is 15 --/
theorem urn_problem (N : ℕ) : 
  (5 : ℚ) / 10 * (10 : ℚ) / (10 + N) +  -- Probability of both balls being green
  (5 : ℚ) / 10 * (N : ℚ) / (10 + N) =   -- Probability of both balls being blue
  (52 : ℚ) / 100 →                      -- Total probability of same color
  N = 15 := by
sorry

end NUMINAMATH_CALUDE_urn_problem_l694_69424


namespace NUMINAMATH_CALUDE_sum_of_roots_greater_than_two_l694_69497

theorem sum_of_roots_greater_than_two (x₁ x₂ : ℝ) 
  (h₁ : 5 * x₁^3 - 6 = 0) 
  (h₂ : 6 * x₂^3 - 5 = 0) : 
  x₁ + x₂ > 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_greater_than_two_l694_69497


namespace NUMINAMATH_CALUDE_point_on_600_degree_angle_l694_69492

/-- If a point (-4, a) lies on the terminal side of an angle of 600°, then a = -4√3 -/
theorem point_on_600_degree_angle (a : ℝ) : 
  (∃ θ : ℝ, θ = 600 * π / 180 ∧ Real.tan θ = a / (-4)) → a = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_600_degree_angle_l694_69492


namespace NUMINAMATH_CALUDE_solve_salary_problem_l694_69408

def salary_problem (salary_A salary_B : ℝ) : Prop :=
  salary_A + salary_B = 3000 ∧
  0.05 * salary_A = 0.15 * salary_B

theorem solve_salary_problem :
  ∃ (salary_A : ℝ), salary_problem salary_A (3000 - salary_A) ∧ salary_A = 2250 := by
  sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l694_69408


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l694_69447

theorem consecutive_integers_average (highest : ℕ) (h : highest = 36) :
  let set := List.range 7
  let numbers := set.map (λ i => highest - (6 - i))
  (numbers.sum : ℚ) / 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l694_69447


namespace NUMINAMATH_CALUDE_family_gathering_handshakes_l694_69439

/-- The number of unique handshakes in a family gathering with twins and triplets -/
theorem family_gathering_handshakes :
  let twin_sets : ℕ := 12
  let triplet_sets : ℕ := 3
  let twins_per_set : ℕ := 2
  let triplets_per_set : ℕ := 3
  let total_twins : ℕ := twin_sets * twins_per_set
  let total_triplets : ℕ := triplet_sets * triplets_per_set
  let twin_handshakes : ℕ := total_twins * (total_twins - twins_per_set)
  let triplet_handshakes : ℕ := total_triplets * (total_triplets - triplets_per_set)
  let twin_triplet_handshakes : ℕ := total_twins * total_triplets
  let total_handshakes : ℕ := twin_handshakes + triplet_handshakes + twin_triplet_handshakes
  327 = total_handshakes / 2 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_l694_69439


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l694_69493

def arithmetic_sequence (a₁ d n : ℕ) := 
  (fun i => a₁ + (i - 1) * d)

theorem arithmetic_sequence_length : 
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 15 4 n (n) = 95 ∧ n = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l694_69493


namespace NUMINAMATH_CALUDE_scientific_notation_102200_l694_69441

theorem scientific_notation_102200 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 102200 = a * (10 : ℝ) ^ n ∧ a = 1.022 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_102200_l694_69441


namespace NUMINAMATH_CALUDE_women_in_first_class_l694_69420

theorem women_in_first_class (total_passengers : ℕ) 
  (percent_women : ℚ) (percent_women_first_class : ℚ) :
  total_passengers = 180 →
  percent_women = 65 / 100 →
  percent_women_first_class = 15 / 100 →
  ⌈(total_passengers : ℚ) * percent_women * percent_women_first_class⌉ = 18 :=
by sorry

end NUMINAMATH_CALUDE_women_in_first_class_l694_69420


namespace NUMINAMATH_CALUDE_solution_count_l694_69469

-- Define the equations
def equation1 (x y : ℂ) : Prop := y = (x + 2)^3
def equation2 (x y : ℂ) : Prop := x * y + 2 * y = 2

-- Define a solution pair
def is_solution (x y : ℂ) : Prop := equation1 x y ∧ equation2 x y

-- Define the count of real and imaginary solutions
def real_solution_count : ℕ := 2
def imaginary_solution_count : ℕ := 2

-- Theorem statement
theorem solution_count :
  (∃ (s : Finset (ℂ × ℂ)), s.card = real_solution_count + imaginary_solution_count ∧
    (∀ (p : ℂ × ℂ), p ∈ s ↔ is_solution p.1 p.2) ∧
    (∃ (r : Finset (ℂ × ℂ)), r ⊆ s ∧ r.card = real_solution_count ∧
      (∀ (p : ℂ × ℂ), p ∈ r → p.1.im = 0 ∧ p.2.im = 0)) ∧
    (∃ (i : Finset (ℂ × ℂ)), i ⊆ s ∧ i.card = imaginary_solution_count ∧
      (∀ (p : ℂ × ℂ), p ∈ i → p.1.im ≠ 0 ∨ p.2.im ≠ 0))) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l694_69469


namespace NUMINAMATH_CALUDE_gcd_statements_l694_69430

theorem gcd_statements : 
  (Nat.gcd 16 12 = 4) ∧ 
  (Nat.gcd 78 36 = 6) ∧ 
  (Nat.gcd 85 357 ≠ 34) ∧ 
  (Nat.gcd 105 315 = 105) := by
  sorry

end NUMINAMATH_CALUDE_gcd_statements_l694_69430


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l694_69484

/-- Represents a license plate with 4 digits and 2 letters -/
structure LicensePlate where
  digits : Fin 10 → Fin 10
  letters : Fin 2 → Fin 26

/-- Checks if a sequence of 4 digits is a palindrome -/
def isPalindrome4 (s : Fin 4 → Fin 10) : Prop :=
  s 0 = s 3 ∧ s 1 = s 2

/-- Checks if a sequence of 2 letters is a palindrome -/
def isPalindrome2 (s : Fin 2 → Fin 26) : Prop :=
  s 0 = s 1

/-- The probability of a license plate containing at least one palindrome sequence -/
def palindromeProbability : ℚ :=
  5 / 104

/-- The main theorem stating the probability of a license plate containing at least one palindrome sequence -/
theorem license_plate_palindrome_probability :
  palindromeProbability = 5 / 104 := by
  sorry


end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l694_69484


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l694_69404

/-- The amount of money Chris had before his birthday -/
def money_before : ℕ := sorry

/-- The amount Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := 20

/-- The amount Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The amount Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The total amount Chris had after his birthday -/
def total_after : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday -/
theorem chris_money_before_birthday :
  money_before + aunt_uncle_gift + parents_gift + grandmother_gift = total_after ∧
  money_before = 159 := by sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l694_69404


namespace NUMINAMATH_CALUDE_special_number_is_perfect_square_l694_69416

theorem special_number_is_perfect_square (n : ℕ) :
  ∃ k : ℕ, 4 * 10^(2*n+2) - 4 * 10^(n+1) + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_special_number_is_perfect_square_l694_69416


namespace NUMINAMATH_CALUDE_dice_roll_distinct_roots_probability_l694_69413

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

def has_distinct_roots (a b : ℕ) : Prop := a^2 > 8*b

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 9

theorem dice_roll_distinct_roots_probability :
  (∀ a b : ℕ, is_valid_roll a → is_valid_roll b →
    (has_distinct_roots a b ↔ a^2 > 8*b)) →
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_dice_roll_distinct_roots_probability_l694_69413


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l694_69444

theorem parallel_lines_condition (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : a₁^2 + b₁^2 ≠ 0) (h₂ : a₂^2 + b₂^2 ≠ 0) :
  ¬(∀ (x y : ℝ), (a₁*x + b₁*y + c₁ = 0 ↔ a₂*x + b₂*y + c₂ = 0) ↔ 
    (a₁*b₂ - a₂*b₁ ≠ 0)) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l694_69444


namespace NUMINAMATH_CALUDE_tangent_line_fixed_point_l694_69463

/-- The function f(x) = x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

/-- The derivative of f(x) -/
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 2*x + m

/-- Theorem: The tangent line to f(x) at x = 2 passes through (0, -3) for all m -/
theorem tangent_line_fixed_point (m : ℝ) : 
  let x₀ : ℝ := 2
  let y₀ : ℝ := f m x₀
  let slope : ℝ := f_derivative m x₀
  ∃ (k : ℝ), k * slope = y₀ + 3 ∧ k * (-1) = x₀ := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_fixed_point_l694_69463


namespace NUMINAMATH_CALUDE_field_length_correct_l694_69456

/-- The length of a rectangular field -/
def field_length : ℝ := 75

/-- The width of the rectangular field -/
def field_width : ℝ := 15

/-- The number of times the field is circled -/
def laps : ℕ := 3

/-- The total distance covered -/
def total_distance : ℝ := 540

/-- Theorem stating that the field length is correct given the conditions -/
theorem field_length_correct : 
  2 * (field_length + field_width) * laps = total_distance :=
by sorry

end NUMINAMATH_CALUDE_field_length_correct_l694_69456


namespace NUMINAMATH_CALUDE_painter_job_completion_six_to_four_painters_l694_69479

/-- The number of work-days required for a group of painters to complete a job -/
def work_days (painters : ℕ) (days : ℚ) : ℚ := painters * days

theorem painter_job_completion 
  (initial_painters : ℕ) 
  (initial_days : ℚ) 
  (new_painters : ℕ) : 
  initial_painters > 0 → 
  new_painters > 0 → 
  initial_days > 0 → 
  work_days initial_painters initial_days = work_days new_painters ((initial_painters * initial_days) / new_painters) :=
by
  sorry

theorem six_to_four_painters :
  work_days 6 (14/10) = work_days 4 (21/10) :=
by
  sorry

end NUMINAMATH_CALUDE_painter_job_completion_six_to_four_painters_l694_69479


namespace NUMINAMATH_CALUDE_last_digit_is_three_l694_69468

/-- Represents a four-digit number -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  h1 : d1 < 10
  h2 : d2 < 10
  h3 : d3 < 10
  h4 : d4 < 10

/-- Predicate for the first clue -/
def clue1 (n : FourDigitNumber) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ 
    (n.d1 = (Vector.get ⟨[1,3,5,9], rfl⟩ i) ∧ i ≠ 0) ∧
    (n.d2 = (Vector.get ⟨[1,3,5,9], rfl⟩ j) ∧ j ≠ 1)

/-- Predicate for the second clue -/
def clue2 (n : FourDigitNumber) : Prop :=
  n.d1 = 9 ∨ n.d2 = 0 ∨ n.d3 = 1 ∨ n.d4 = 3

/-- Predicate for the third clue -/
def clue3 (n : FourDigitNumber) : Prop :=
  (n.d1 = 9 ∧ (n.d2 = 0 ∨ n.d3 = 1 ∨ n.d4 = 3)) ∨
  (n.d2 = 0 ∧ (n.d1 = 9 ∨ n.d3 = 1 ∨ n.d4 = 3)) ∨
  (n.d3 = 1 ∧ (n.d1 = 9 ∨ n.d2 = 0 ∨ n.d4 = 3)) ∨
  (n.d4 = 3 ∧ (n.d1 = 9 ∨ n.d2 = 0 ∨ n.d3 = 1))

/-- Predicate for the fourth clue -/
def clue4 (n : FourDigitNumber) : Prop :=
  (n.d2 = 1 ∨ n.d3 = 1 ∨ n.d4 = 1) ∧ n.d1 ≠ 1

/-- Predicate for the fifth clue -/
def clue5 (n : FourDigitNumber) : Prop :=
  n.d1 ≠ 7 ∧ n.d1 ≠ 6 ∧ n.d1 ≠ 4 ∧ n.d1 ≠ 2 ∧
  n.d2 ≠ 7 ∧ n.d2 ≠ 6 ∧ n.d2 ≠ 4 ∧ n.d2 ≠ 2 ∧
  n.d3 ≠ 7 ∧ n.d3 ≠ 6 ∧ n.d3 ≠ 4 ∧ n.d3 ≠ 2 ∧
  n.d4 ≠ 7 ∧ n.d4 ≠ 6 ∧ n.d4 ≠ 4 ∧ n.d4 ≠ 2

theorem last_digit_is_three (n : FourDigitNumber) 
  (h1 : clue1 n) (h2 : clue2 n) (h3 : clue3 n) (h4 : clue4 n) (h5 : clue5 n) : 
  n.d4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_is_three_l694_69468


namespace NUMINAMATH_CALUDE_max_balloons_with_promotion_orvin_max_balloons_l694_69428

/-- The maximum number of balloons that can be bought given a promotion --/
theorem max_balloons_with_promotion (full_price_balloons : ℕ) : ℕ :=
  let discounted_sets := (full_price_balloons * 2) / 3
  discounted_sets * 2

/-- Proof that given the conditions, the maximum number of balloons Orvin can buy is 52 --/
theorem orvin_max_balloons : max_balloons_with_promotion 40 = 52 := by
  sorry

end NUMINAMATH_CALUDE_max_balloons_with_promotion_orvin_max_balloons_l694_69428


namespace NUMINAMATH_CALUDE_circle_intersection_perpendicularity_l694_69414

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the properties and relations
variable (center : Circle → Point)
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (tangent : Circle → Circle → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_intersection_perpendicularity
  (O O₁ O₂ : Circle) (M N S T : Point) :
  (intersect O₁ O₂ M N) →
  (tangent O O₁ S) →
  (tangent O O₂ T) →
  (perpendicular (center O) M M N ↔ collinear S N T) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_perpendicularity_l694_69414


namespace NUMINAMATH_CALUDE_max_value_theorem_l694_69480

theorem max_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 - 2*a*b + 9*b^2 - c = 0) :
  ∃ (max_abc : ℝ), 
    (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x^2 - 2*x*y + 9*y^2 - z = 0 → 
      x*y/z ≤ max_abc) →
    (3/a + 1/b - 12/c ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l694_69480


namespace NUMINAMATH_CALUDE_azar_winning_configurations_l694_69422

/-- Represents a tic-tac-toe board configuration -/
def TicTacToeBoard := List (Option Bool)

/-- Checks if a given board configuration is valid according to the game rules -/
def is_valid_board (board : TicTacToeBoard) : Bool :=
  board.length = 9 ∧ 
  (board.filter (· = some true)).length = 4 ∧
  (board.filter (· = some false)).length = 3

/-- Checks if Azar (X) has won in the given board configuration -/
def azar_wins (board : TicTacToeBoard) : Bool :=
  sorry

/-- Counts the number of valid winning configurations for Azar -/
def count_winning_configurations : Nat :=
  sorry

theorem azar_winning_configurations : 
  count_winning_configurations = 100 := by sorry

end NUMINAMATH_CALUDE_azar_winning_configurations_l694_69422


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l694_69410

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of a point being inside a circle -/
def inside_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

/-- Definition of a line being separate from a circle -/
def separate_from_circle (l : Line) (c : Circle) : Prop :=
  let d := |l.a * c.center.1 + l.b * c.center.2 + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d > c.radius

/-- Main theorem -/
theorem line_separate_from_circle 
  (a : ℝ) 
  (h_a : a > 0) 
  (M : ℝ × ℝ) 
  (h_M : inside_circle M ⟨⟨0, 0⟩, a, h_a⟩) 
  (h_M_not_center : M ≠ (0, 0)) :
  separate_from_circle ⟨M.1, M.2, -a^2⟩ ⟨⟨0, 0⟩, a, h_a⟩ :=
sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l694_69410


namespace NUMINAMATH_CALUDE_min_xy_over_x2_plus_y2_l694_69419

theorem min_xy_over_x2_plus_y2 (x y : ℝ) (hx : 1/2 ≤ x ∧ x ≤ 1) (hy : 2/5 ≤ y ∧ y ≤ 1/2) :
  x * y / (x^2 + y^2) ≥ 1/2 ∧ ∃ (x₀ y₀ : ℝ), 1/2 ≤ x₀ ∧ x₀ ≤ 1 ∧ 2/5 ≤ y₀ ∧ y₀ ≤ 1/2 ∧ x₀ * y₀ / (x₀^2 + y₀^2) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_over_x2_plus_y2_l694_69419


namespace NUMINAMATH_CALUDE_ren_faire_amulet_sales_l694_69481

/-- Represents the problem of calculating amulets sold per day at a Ren Faire --/
theorem ren_faire_amulet_sales (selling_price : ℕ) (cost_price : ℕ) (revenue_share : ℚ)
  (total_days : ℕ) (total_profit : ℕ) :
  selling_price = 40 →
  cost_price = 30 →
  revenue_share = 1/10 →
  total_days = 2 →
  total_profit = 300 →
  (selling_price - cost_price - (revenue_share * selling_price)) * total_days * 
    (total_profit / ((selling_price - cost_price - (revenue_share * selling_price)) * total_days)) = 25 :=
by sorry

end NUMINAMATH_CALUDE_ren_faire_amulet_sales_l694_69481


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l694_69453

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ 0 < a ∧ a = b ∧ b < 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l694_69453


namespace NUMINAMATH_CALUDE_total_balloon_cost_l694_69466

def fred_balloons : ℕ := 10
def fred_cost : ℚ := 1

def sam_balloons : ℕ := 46
def sam_cost : ℚ := 3/2

def dan_balloons : ℕ := 16
def dan_cost : ℚ := 3/4

theorem total_balloon_cost :
  (fred_balloons : ℚ) * fred_cost +
  (sam_balloons : ℚ) * sam_cost +
  (dan_balloons : ℚ) * dan_cost = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_balloon_cost_l694_69466


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l694_69461

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 + a 3 + a 10 + a 11 = 48 → a 6 + a 7 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l694_69461


namespace NUMINAMATH_CALUDE_cartesian_coordinates_wrt_origin_l694_69429

/-- In a Cartesian coordinate system, the coordinates of a point with respect to the origin are equal to the point's coordinates. -/
theorem cartesian_coordinates_wrt_origin (x y : ℝ) : 
  let point : ℝ × ℝ := (x, y)
  let origin : ℝ × ℝ := (0, 0)
  let coordinates_wrt_origin : ℝ × ℝ := (x, y)
  coordinates_wrt_origin = point :=
by sorry

end NUMINAMATH_CALUDE_cartesian_coordinates_wrt_origin_l694_69429


namespace NUMINAMATH_CALUDE_functional_equation_l694_69427

theorem functional_equation (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x - y) = f x * f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_l694_69427


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l694_69495

theorem sum_of_three_numbers (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b * c = (a + b) * (a + c)) : a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l694_69495


namespace NUMINAMATH_CALUDE_election_votes_theorem_l694_69490

theorem election_votes_theorem (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    winner_votes - loser_votes = 320) →
  total_votes = 800 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l694_69490


namespace NUMINAMATH_CALUDE_product_without_zero_ending_l694_69415

theorem product_without_zero_ending : ∃ (a b : ℤ), 
  a * b = 100000 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_without_zero_ending_l694_69415


namespace NUMINAMATH_CALUDE_dice_roll_sum_l694_69407

theorem dice_roll_sum (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 360 →
  a + b + c + d ≠ 17 :=
by sorry

end NUMINAMATH_CALUDE_dice_roll_sum_l694_69407


namespace NUMINAMATH_CALUDE_worm_gnawed_pages_in_four_volumes_l694_69411

/-- Represents a book volume with a specific number of pages -/
structure Volume :=
  (pages : ℕ)

/-- Represents a bookshelf with a list of volumes -/
structure Bookshelf :=
  (volumes : List Volume)

/-- Calculates the number of pages a worm gnaws through in a bookshelf -/
def wormGnawedPages (shelf : Bookshelf) : ℕ :=
  match shelf.volumes with
  | [] => 0
  | [_] => 0
  | v1 :: vs :: tail => 
    (vs.pages + (match tail with
                 | [_] => 0
                 | v3 :: _ => v3.pages
                 | _ => 0))

/-- Theorem stating the number of pages gnawed by the worm -/
theorem worm_gnawed_pages_in_four_volumes : 
  ∀ (shelf : Bookshelf),
    shelf.volumes.length = 4 →
    (∀ v ∈ shelf.volumes, v.pages = 200) →
    wormGnawedPages shelf = 400 := by
  sorry


end NUMINAMATH_CALUDE_worm_gnawed_pages_in_four_volumes_l694_69411


namespace NUMINAMATH_CALUDE_f_of_3_equals_23_l694_69488

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- State the theorem
theorem f_of_3_equals_23 (a b : ℝ) :
  f a b 1 = 7 → f a b 2 = 14 → f a b 3 = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_23_l694_69488


namespace NUMINAMATH_CALUDE_rachel_age_when_father_is_60_l694_69445

/-- Given the ages and relationships in Rachel's family, prove that Rachel will be 25 when her father is 60. -/
theorem rachel_age_when_father_is_60 
  (rachel_age : ℕ)
  (grandfather_age : ℕ)
  (mother_age : ℕ)
  (father_age : ℕ)
  (h1 : rachel_age = 12)
  (h2 : grandfather_age = 7 * rachel_age)
  (h3 : mother_age = grandfather_age / 2)
  (h4 : father_age = mother_age + 5) :
  rachel_age + (60 - father_age) = 25 := by
sorry

end NUMINAMATH_CALUDE_rachel_age_when_father_is_60_l694_69445


namespace NUMINAMATH_CALUDE_rational_sqrt_equation_l694_69438

theorem rational_sqrt_equation (a b : ℚ) : 
  a - b * Real.sqrt 2 = (1 + Real.sqrt 2)^2 → a = 3 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_equation_l694_69438


namespace NUMINAMATH_CALUDE_sixth_power_sum_l694_69489

theorem sixth_power_sum (x : ℝ) (hx : x ≠ 0) : x + 1/x = 1 → x^6 + 1/x^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l694_69489


namespace NUMINAMATH_CALUDE_photo_collection_l694_69454

theorem photo_collection (total : ℕ) (tim_less : ℕ) (paul_more : ℕ) : 
  total = 152 →
  tim_less = 100 →
  paul_more = 10 →
  ∃ (tim paul tom : ℕ), 
    tim = total - tim_less ∧
    paul = tim + paul_more ∧
    tom = total - (tim + paul) ∧
    tom = 38 := by
  sorry

end NUMINAMATH_CALUDE_photo_collection_l694_69454


namespace NUMINAMATH_CALUDE_point_M_coordinates_l694_69417

def f (x : ℝ) : ℝ := 2 * x^2 + 1

theorem point_M_coordinates (x₀ y₀ : ℝ) :
  (∃ M : ℝ × ℝ, M.1 = x₀ ∧ M.2 = y₀ ∧ 
   (deriv f) x₀ = -8 ∧ f x₀ = y₀) →
  x₀ = -2 ∧ y₀ = 9 := by
sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l694_69417


namespace NUMINAMATH_CALUDE_original_deck_size_is_52_l694_69449

/-- The number of players among whom the deck is distributed -/
def num_players : ℕ := 3

/-- The number of cards each player receives after distribution -/
def cards_per_player : ℕ := 18

/-- The number of cards added to the original deck -/
def added_cards : ℕ := 2

/-- The original number of cards in the deck -/
def original_deck_size : ℕ := num_players * cards_per_player - added_cards

theorem original_deck_size_is_52 : original_deck_size = 52 := by
  sorry

end NUMINAMATH_CALUDE_original_deck_size_is_52_l694_69449


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_one_l694_69455

theorem fraction_meaningful_iff_not_one (x : ℝ) :
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_one_l694_69455


namespace NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l694_69485

/-- The duration of the red light in seconds -/
def red_duration : ℕ := 30

/-- The duration of the yellow light in seconds -/
def yellow_duration : ℕ := 5

/-- The duration of the green light in seconds -/
def green_duration : ℕ := 40

/-- The total duration of one traffic light cycle -/
def total_duration : ℕ := red_duration + yellow_duration + green_duration

/-- The probability of seeing a red light -/
def red_light_probability : ℚ := red_duration / total_duration

theorem red_light_probability_is_two_fifths :
  red_light_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l694_69485


namespace NUMINAMATH_CALUDE_parabola_vertex_l694_69498

/-- The equation of a parabola in the form y^2 - 4y + 2x + 9 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 4*y + 2*x + 9 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  eq x y ∧ ∀ x' y', eq x' y' → y ≤ y'

theorem parabola_vertex :
  is_vertex (-5/2) 2 parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l694_69498


namespace NUMINAMATH_CALUDE_fifth_set_fraction_approx_three_fourths_l694_69452

-- Define the duration of the whole match in minutes
def whole_match_duration : ℕ := 665

-- Define the duration of the fifth set in minutes
def fifth_set_duration : ℕ := 491

-- Define a function to calculate the fraction
def match_fraction : ℚ := fifth_set_duration / whole_match_duration

-- Define what we consider as "approximately equal" (e.g., within 0.02)
def approximately_equal (x y : ℚ) : Prop := abs (x - y) < 1/50

-- Theorem statement
theorem fifth_set_fraction_approx_three_fourths :
  approximately_equal match_fraction (3/4) :=
sorry

end NUMINAMATH_CALUDE_fifth_set_fraction_approx_three_fourths_l694_69452


namespace NUMINAMATH_CALUDE_sand_container_problem_l694_69494

theorem sand_container_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a * Real.exp (-8 * b) = a / 2) →
  (∃ t : ℝ, a * Real.exp (-b * t) = a / 8 ∧ t > 0) →
  (∃ t : ℝ, a * Real.exp (-b * t) = a / 8 ∧ t = 24) :=
by sorry

end NUMINAMATH_CALUDE_sand_container_problem_l694_69494


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l694_69405

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle_circumference : ℝ),
    rectangle_width = 5 →
    rectangle_height = 12 →
    circle_circumference = π * Real.sqrt (rectangle_width^2 + rectangle_height^2) →
    circle_circumference = 13 * π :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l694_69405


namespace NUMINAMATH_CALUDE_min_value_quadratic_l694_69472

theorem min_value_quadratic (x y : ℝ) : 2 * x^2 + 3 * y^2 - 8 * x + 6 * y + 25 ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l694_69472


namespace NUMINAMATH_CALUDE_circles_internally_tangent_with_one_common_tangent_l694_69478

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 4 = 0
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 12*y + 4 = 0

-- Define the centers and radii of the circles
def center_M : ℝ × ℝ := (-1, 2)
def center_N : ℝ × ℝ := (2, 6)
def radius_M : ℝ := 1
def radius_N : ℝ := 6

-- Define the distance between centers
def distance_between_centers : ℝ := 5

-- Define the common tangent line equation
def common_tangent (x y : ℝ) : Prop := 3*x + 4*y = 0

theorem circles_internally_tangent_with_one_common_tangent :
  (distance_between_centers = radius_N - radius_M) ∧
  (∃! (l : ℝ × ℝ → Prop), ∀ x y, l (x, y) ↔ common_tangent x y) ∧
  (∀ x y, circle_M x y ∧ circle_N x y → common_tangent x y) :=
sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_with_one_common_tangent_l694_69478
