import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l321_32109

theorem rectangular_plot_breadth (area length breadth : ℝ) : 
  area = 24 * breadth →
  length = breadth + 10 →
  area = length * breadth →
  breadth = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l321_32109


namespace NUMINAMATH_CALUDE_race_speed_ratio_l321_32139

/-- Given a race with the following conditions:
  * The total race distance is 600 meters
  * Contestant A has a 100 meter head start
  * Contestant A wins by 200 meters
  This theorem proves that the ratio of the speeds of contestant A to contestant B is 5:4. -/
theorem race_speed_ratio (vA vB : ℝ) (vA_pos : vA > 0) (vB_pos : vB > 0) : 
  (600 - 100) / vA = 400 / vB → vA / vB = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l321_32139


namespace NUMINAMATH_CALUDE_smallest_number_l321_32194

def base_2_to_10 (n : ℕ) : ℕ := n

def base_4_to_10 (n : ℕ) : ℕ := n

def base_8_to_10 (n : ℕ) : ℕ := n

theorem smallest_number :
  let a := base_4_to_10 321
  let b := 58
  let c := base_2_to_10 111000
  let d := base_8_to_10 73
  c < a ∧ c < b ∧ c < d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l321_32194


namespace NUMINAMATH_CALUDE_cost_of_500_apples_l321_32164

/-- The cost of a single apple in cents -/
def apple_cost : ℕ := 5

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of apples we want to calculate the cost for -/
def apple_quantity : ℕ := 500

/-- Theorem stating that the cost of 500 apples is 25.00 dollars -/
theorem cost_of_500_apples : 
  (apple_quantity * apple_cost : ℚ) / cents_per_dollar = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_apples_l321_32164


namespace NUMINAMATH_CALUDE_ellipse_chord_bisector_l321_32111

def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

def point_inside_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 < 1

def bisector_line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

theorem ellipse_chord_bisector :
  ∀ x y : ℝ,
  ellipse x y →
  point_inside_ellipse 3 1 →
  bisector_line 3 4 (-13) x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_chord_bisector_l321_32111


namespace NUMINAMATH_CALUDE_power_equality_l321_32127

theorem power_equality (m : ℕ) : 9^4 = 3^(2*m) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l321_32127


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l321_32108

theorem washing_machine_capacity 
  (shirts : ℕ) 
  (sweaters : ℕ) 
  (loads : ℕ) 
  (h1 : shirts = 19) 
  (h2 : sweaters = 8) 
  (h3 : loads = 3) : 
  (shirts + sweaters) / loads = 9 := by
sorry

end NUMINAMATH_CALUDE_washing_machine_capacity_l321_32108


namespace NUMINAMATH_CALUDE_unique_stamp_denomination_l321_32192

/-- Given a positive integer n, this function checks if a postage value can be formed
    using stamps of denominations 7, n, and n+1 cents. -/
def can_form_postage (n : ℕ+) (postage : ℕ) : Prop :=
  ∃ (a b c : ℕ), postage = 7 * a + n * b + (n + 1) * c

/-- This theorem states that 18 is the unique positive integer n such that,
    given stamps of denominations 7, n, and n+1 cents, 106 cents is the
    greatest postage that cannot be formed. -/
theorem unique_stamp_denomination :
  ∃! (n : ℕ+),
    (¬ can_form_postage n 106) ∧
    (∀ m : ℕ, m > 106 → can_form_postage n m) ∧
    (∀ k : ℕ, k < 106 → ¬ can_form_postage n k → can_form_postage n (k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_unique_stamp_denomination_l321_32192


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_5_l321_32152

theorem quadratic_inequality_implies_a_geq_5 (a : ℝ) : 
  (∀ x : ℝ, 1 < x ∧ x < 5 → x^2 - 2*(a-2)*x + a < 0) → a ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_5_l321_32152


namespace NUMINAMATH_CALUDE_unique_solution_characterization_l321_32183

/-- The set of real numbers a for which the system has a unique solution -/
def UniqueSystemSolutionSet : Set ℝ :=
  {a | a < -5 ∨ a > -1}

/-- The system of equations -/
def SystemEquations (x y a : ℝ) : Prop :=
  x = 4 * Real.sqrt y + a ∧ y^2 - x^2 + 3*y - 5*x - 4 = 0

theorem unique_solution_characterization (a : ℝ) :
  (∃! p : ℝ × ℝ, SystemEquations p.1 p.2 a) ↔ a ∈ UniqueSystemSolutionSet :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_characterization_l321_32183


namespace NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l321_32165

/-- A convex quadrilateral with special diagonal properties -/
structure SpecialQuadrilateral where
  /-- The quadrilateral is convex -/
  convex : Bool
  /-- Any diagonal divides the quadrilateral into two isosceles triangles -/
  diagonal_isosceles : Bool
  /-- Both diagonals divide the quadrilateral into four isosceles triangles -/
  both_diagonals_isosceles : Bool

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The trapezoid has parallel bases -/
  parallel_bases : Bool
  /-- The non-parallel sides are equal -/
  equal_legs : Bool
  /-- The smaller base is equal to the legs -/
  base_equals_legs : Bool

/-- Theorem: There exists a quadrilateral satisfying the special properties that is not a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ (q : SpecialQuadrilateral) (t : IsoscelesTrapezoid),
    q.convex ∧
    q.diagonal_isosceles ∧
    q.both_diagonals_isosceles ∧
    t.parallel_bases ∧
    t.equal_legs ∧
    t.base_equals_legs ∧
    (q ≠ square) := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l321_32165


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_x_sum_l321_32158

/-- The sum of all possible x coordinates of the 4th vertex of a parallelogram 
    with three vertices at (1,2), (3,8), and (4,1) is equal to 8. -/
theorem parallelogram_fourth_vertex_x_sum : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (3, 8)
  let C : ℝ × ℝ := (4, 1)
  let D₁ : ℝ × ℝ := B + C - A
  let D₂ : ℝ × ℝ := A + C - B
  let D₃ : ℝ × ℝ := A + B - C
  (D₁.1 + D₂.1 + D₃.1 : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_x_sum_l321_32158


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_chord_length_implies_line_equation_l321_32186

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

-- Define the line equation
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem for part 1
theorem line_intersects_ellipse (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Theorem for part 2
theorem chord_length_implies_line_equation (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ m ∧ line x₂ y₂ m ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = (2 * Real.sqrt 10 / 5)^2) →
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_chord_length_implies_line_equation_l321_32186


namespace NUMINAMATH_CALUDE_P_equals_59_when_V_is_9_l321_32137

-- Define the relationship between P, h, and V
def P (h V : ℝ) : ℝ := 3 * h * V + 5

-- State the theorem
theorem P_equals_59_when_V_is_9 : 
  ∃ (h : ℝ), (P h 6 = 41) → (P h 9 = 59) := by
  sorry

end NUMINAMATH_CALUDE_P_equals_59_when_V_is_9_l321_32137


namespace NUMINAMATH_CALUDE_special_function_property_l321_32155

/-- A function satisfying the given property for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x^2 - f x * f y + f y^2)

/-- Theorem stating that if f is a special function, then f(1996x) = 1996f(x) for all real x -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x := by
  sorry


end NUMINAMATH_CALUDE_special_function_property_l321_32155


namespace NUMINAMATH_CALUDE_journey_time_proof_l321_32169

/-- Proves that the total time to complete a 24 km journey is 8 hours, 
    given specific speed conditions. -/
theorem journey_time_proof (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ) 
    (remaining_speed : ℝ) : 
  total_distance = 24 →
  initial_speed = 4 →
  initial_time = 4 →
  remaining_speed = 2 →
  ∃ (total_time : ℝ), 
    total_time = 8 ∧ 
    total_distance = initial_speed * initial_time + 
      remaining_speed * (total_time - initial_time) :=
by
  sorry


end NUMINAMATH_CALUDE_journey_time_proof_l321_32169


namespace NUMINAMATH_CALUDE_circle_equation_l321_32173

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x-2)^2 + (y-1)^2 = 1

-- Define that a point is on the line
def point_on_line (x y : ℝ) : Prop := line_l x y

-- Define that a point is on the circle
def point_on_circle (x y : ℝ) : Prop := circle_C x y

-- Theorem statement
theorem circle_equation : 
  (point_on_line 2 1) ∧ 
  (point_on_line 6 3) ∧ 
  (∃ h k : ℝ, point_on_line h k ∧ point_on_circle h k) ∧
  (point_on_circle 2 0) ∧ 
  (point_on_circle 3 1) → 
  ∀ x y : ℝ, circle_C x y ↔ (x-2)^2 + (y-1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l321_32173


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l321_32147

theorem fraction_to_decimal : (7 : ℚ) / 125 = 0.056 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l321_32147


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l321_32134

-- Define the swimming speed in still water
def still_water_speed : ℝ := 6

-- Define the function for downstream speed
def downstream_speed (stream_speed : ℝ) : ℝ := still_water_speed + stream_speed

-- Define the function for upstream speed
def upstream_speed (stream_speed : ℝ) : ℝ := still_water_speed - stream_speed

-- Theorem statement
theorem stream_speed_calculation :
  ∃ (stream_speed : ℝ),
    stream_speed > 0 ∧
    downstream_speed stream_speed / upstream_speed stream_speed = 2 ∧
    stream_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l321_32134


namespace NUMINAMATH_CALUDE_straight_line_five_equal_angles_l321_32181

/-- If PQ is a straight line with 5 equal angles, each measuring x°, then x = 36° -/
theorem straight_line_five_equal_angles (x : ℝ) : 
  (5 * x = 180) → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_straight_line_five_equal_angles_l321_32181


namespace NUMINAMATH_CALUDE_express_delivery_growth_rate_l321_32174

/-- Proves that the equation 5000(1+x)^2 = 7500 correctly represents the average annual growth rate
    for an initial value of 5000, a final value of 7500, over a 2-year period. -/
theorem express_delivery_growth_rate (x : ℝ) : 
  (5000 : ℝ) * (1 + x)^2 = 7500 ↔ 
  (∃ (initial final : ℝ) (years : ℕ), 
    initial = 5000 ∧ 
    final = 7500 ∧ 
    years = 2 ∧ 
    final = initial * (1 + x)^years) :=
by sorry

end NUMINAMATH_CALUDE_express_delivery_growth_rate_l321_32174


namespace NUMINAMATH_CALUDE_extended_tile_ratio_l321_32191

theorem extended_tile_ratio (initial_black : ℕ) (initial_white : ℕ) 
  (h1 : initial_black = 7)
  (h2 : initial_white = 18)
  (h3 : initial_black + initial_white = 25) :
  let side_length : ℕ := (initial_black + initial_white).sqrt
  let extended_side_length : ℕ := side_length + 2
  let extended_black : ℕ := initial_black + 4 * side_length + 4
  let extended_white : ℕ := initial_white
  (extended_black : ℚ) / extended_white = 31 / 18 := by
sorry

end NUMINAMATH_CALUDE_extended_tile_ratio_l321_32191


namespace NUMINAMATH_CALUDE_factory_output_percentage_l321_32116

theorem factory_output_percentage (T X Y : ℝ) : 
  T > 0 →  -- Total output is positive
  X > 0 →  -- Machine-x output is positive
  Y > 0 →  -- Machine-y output is positive
  X + Y = T →  -- Total output is sum of both machines
  0.006 * T = 0.009 * X + 0.004 * Y →  -- Defective units equation
  X = 0.4 * T  -- Machine-x produces 40% of total output
  := by sorry

end NUMINAMATH_CALUDE_factory_output_percentage_l321_32116


namespace NUMINAMATH_CALUDE_train_speed_problem_l321_32177

/-- The speed of Train A in miles per hour -/
def speed_A : ℝ := 30

/-- The time difference between Train A and Train B's departure in hours -/
def time_diff : ℝ := 2

/-- The distance at which Train B overtakes Train A in miles -/
def overtake_distance : ℝ := 360

/-- The speed of Train B in miles per hour -/
def speed_B : ℝ := 42

theorem train_speed_problem :
  speed_A * (overtake_distance / speed_A) = 
  speed_B * (overtake_distance / speed_B - time_diff) ∧
  speed_B * time_diff + speed_A * time_diff = overtake_distance := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l321_32177


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_two_l321_32179

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 2 is -2 -/
theorem opposite_of_two : opposite 2 = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_two_l321_32179


namespace NUMINAMATH_CALUDE_genevieve_coffee_consumption_l321_32112

-- Define the conversion factors
def ml_to_oz : ℝ := 0.0338
def l_to_oz : ℝ := 33.8

-- Define the thermos sizes
def small_thermos_ml : ℝ := 250
def medium_thermos_ml : ℝ := 400
def large_thermos_l : ℝ := 1

-- Calculate the amount of coffee in each thermos type in ounces
def small_thermos_oz : ℝ := small_thermos_ml * ml_to_oz
def medium_thermos_oz : ℝ := medium_thermos_ml * ml_to_oz
def large_thermos_oz : ℝ := large_thermos_l * l_to_oz

-- Define Genevieve's consumption
def genevieve_consumption : ℝ := small_thermos_oz + medium_thermos_oz + large_thermos_oz

-- Theorem statement
theorem genevieve_coffee_consumption :
  genevieve_consumption = 55.77 := by sorry

end NUMINAMATH_CALUDE_genevieve_coffee_consumption_l321_32112


namespace NUMINAMATH_CALUDE_abs_neg_sqrt_two_l321_32199

theorem abs_neg_sqrt_two : |(-Real.sqrt 2)| = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_abs_neg_sqrt_two_l321_32199


namespace NUMINAMATH_CALUDE_solution_check_l321_32110

theorem solution_check (x : ℝ) : 
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 → x = 12 := by
  sorry

#check solution_check

end NUMINAMATH_CALUDE_solution_check_l321_32110


namespace NUMINAMATH_CALUDE_monotonic_decreasing_range_l321_32126

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x^2 - 8 * a * x + 3 else Real.log x / Real.log a

-- State the theorem
theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (1/2 ≤ a ∧ a ≤ 5/8) :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_range_l321_32126


namespace NUMINAMATH_CALUDE_oil_price_reduction_l321_32129

/-- Given a price reduction that allows buying 3 kg more for Rs. 700,
    and a reduced price of Rs. 70 per kg, prove that the percentage
    reduction in the price of oil is 30%. -/
theorem oil_price_reduction (original_price : ℝ) :
  (∃ (reduced_price : ℝ),
    reduced_price = 70 ∧
    700 / original_price + 3 = 700 / reduced_price) →
  (original_price - 70) / original_price * 100 = 30 :=
by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l321_32129


namespace NUMINAMATH_CALUDE_code_cracking_probabilities_l321_32101

/-- The probability of person A succeeding -/
def prob_A : ℚ := 1/2

/-- The probability of person B succeeding -/
def prob_B : ℚ := 3/5

/-- The probability of person C succeeding -/
def prob_C : ℚ := 3/4

/-- The probability of exactly one person succeeding -/
def prob_exactly_one : ℚ := 
  prob_A * (1 - prob_B) * (1 - prob_C) + 
  (1 - prob_A) * prob_B * (1 - prob_C) + 
  (1 - prob_A) * (1 - prob_B) * prob_C

/-- The probability of the code being successfully cracked -/
def prob_success : ℚ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- The minimum number of people like C needed for at least 95% success rate -/
def min_people_C : ℕ := 3

theorem code_cracking_probabilities :
  prob_exactly_one = 11/40 ∧ 
  prob_success = 19/20 ∧
  (∀ n : ℕ, n ≥ min_people_C → 1 - (1 - prob_C)^n ≥ 95/100) ∧
  (∀ n : ℕ, n < min_people_C → 1 - (1 - prob_C)^n < 95/100) :=
sorry

end NUMINAMATH_CALUDE_code_cracking_probabilities_l321_32101


namespace NUMINAMATH_CALUDE_min_value_expression_l321_32130

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = 3 ∧ (∀ x y : ℝ, x > 0 → y > 0 → 
    (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x*y) ≥ min) ∧
    ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
      (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x*y) = min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l321_32130


namespace NUMINAMATH_CALUDE_units_digit_17_1987_l321_32175

theorem units_digit_17_1987 : (17^1987) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_1987_l321_32175


namespace NUMINAMATH_CALUDE_ellipse_equation_l321_32133

/-- Represents an ellipse with focus on the x-axis -/
structure Ellipse where
  /-- Distance from the right focus to the short axis endpoint -/
  short_axis_dist : ℝ
  /-- Distance from the right focus to the left vertex -/
  left_vertex_dist : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

theorem ellipse_equation (e : Ellipse) (h1 : e.short_axis_dist = 2) (h2 : e.left_vertex_dist = 3) :
  ∀ x y : ℝ, standard_equation e x y ↔ x^2 / 4 + y^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l321_32133


namespace NUMINAMATH_CALUDE_restaurant_peppers_total_weight_l321_32182

theorem restaurant_peppers_total_weight 
  (green_peppers : ℝ) 
  (red_peppers : ℝ) 
  (h1 : green_peppers = 0.3333333333333333) 
  (h2 : red_peppers = 0.3333333333333333) : 
  green_peppers + red_peppers = 0.6666666666666666 := by
sorry

end NUMINAMATH_CALUDE_restaurant_peppers_total_weight_l321_32182


namespace NUMINAMATH_CALUDE_largest_prime_sum_of_digits_l321_32170

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_prime_sum_of_digits :
  ∀ A B C D : ℕ,
    isSingleDigit A ∧ isSingleDigit B ∧ isSingleDigit C ∧ isSingleDigit D →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    isPrime (A + B) ∧ isPrime (C + D) →
    (A + B) ≠ (C + D) →
    ∃ k : ℕ, k * (C + D) = A + B →
    ∀ E F : ℕ,
      isSingleDigit E ∧ isSingleDigit F →
      E ≠ F ∧ E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D ∧ F ≠ A ∧ F ≠ B ∧ F ≠ C ∧ F ≠ D →
      isPrime (E + F) →
      (E + F) ≠ (C + D) →
      ∃ m : ℕ, m * (C + D) = E + F →
      A + B ≥ E + F →
    A + B = 11
  := by sorry

end NUMINAMATH_CALUDE_largest_prime_sum_of_digits_l321_32170


namespace NUMINAMATH_CALUDE_difference_between_B_and_C_difference_between_B_and_C_proof_l321_32160

theorem difference_between_B_and_C : ℤ → ℤ → ℤ → Prop :=
  fun A B C =>
    A ≠ B ∧ B ≠ C ∧ A ≠ C →
    C = 1 + 8 →
    B = A + 5 →
    A = 9 - 4 →
    B - C = 1

-- The proof is omitted
theorem difference_between_B_and_C_proof :
  ∃ A B C : ℤ, difference_between_B_and_C A B C :=
by
  sorry

end NUMINAMATH_CALUDE_difference_between_B_and_C_difference_between_B_and_C_proof_l321_32160


namespace NUMINAMATH_CALUDE_intersection_points_count_l321_32187

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x - 2*y + 3) * (4*x + y - 5) = 0
def equation2 (x y : ℝ) : Prop := (x + 2*y - 3) * (3*x - 4*y + 6) = 0

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define what it means for a point to satisfy both equations
def satisfiesBothEquations (p : Point) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- Statement of the theorem
theorem intersection_points_count :
  ∃ (s : Finset Point), (∀ p ∈ s, satisfiesBothEquations p) ∧ s.card = 3 ∧
  (∀ p : Point, satisfiesBothEquations p → p ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l321_32187


namespace NUMINAMATH_CALUDE_angle_at_intersection_point_l321_32188

/-- In a 3x3 grid, given points A, B, C, D, and E where AB and CD intersect at E, 
    prove that the angle at E is 45 degrees. -/
theorem angle_at_intersection_point (A B C D E : ℝ × ℝ) : 
  A = (0, 0) → 
  B = (3, 3) → 
  C = (0, 3) → 
  D = (3, 0) → 
  (E.1 - A.1) / (E.2 - A.2) = (B.1 - A.1) / (B.2 - A.2) →  -- E is on line AB
  (E.1 - C.1) / (E.2 - C.2) = (D.1 - C.1) / (D.2 - C.2) →  -- E is on line CD
  Real.arctan ((B.2 - A.2) / (B.1 - A.1) - (D.2 - C.2) / (D.1 - C.1)) / 
    (1 + (B.2 - A.2) / (B.1 - A.1) * (D.2 - C.2) / (D.1 - C.1)) * (180 / Real.pi) = 45 :=
by sorry

end NUMINAMATH_CALUDE_angle_at_intersection_point_l321_32188


namespace NUMINAMATH_CALUDE_partnership_profit_distribution_l321_32102

/-- Partnership profit distribution problem -/
theorem partnership_profit_distribution (total_profit : ℚ) 
  (hA : ℚ) (hB : ℚ) (hC : ℚ) (hD : ℚ) :
  hA = 1/3 →
  hB = 1/4 →
  hC = 1/5 →
  hD = 1 - (hA + hB + hC) →
  total_profit = 2415 →
  hA * total_profit = 805 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_distribution_l321_32102


namespace NUMINAMATH_CALUDE_inequality_proofs_l321_32198

theorem inequality_proofs (x : ℝ) :
  (x^2 - x - 2 ≥ 0 ∧ Real.sqrt (x^2 - x - 2) ≤ 2*x → x ≥ 2) ∧
  (x^2 - x - 2 ≥ 0 ∧ Real.sqrt (x^2 - x - 2) ≥ 2*x → x ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l321_32198


namespace NUMINAMATH_CALUDE_vector_operation_l321_32151

/-- Given vectors a and b in ℝ², prove that (1/2)a - (3/2)b equals (-1,2) -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (1, -1)) :
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l321_32151


namespace NUMINAMATH_CALUDE_complex_multiplication_l321_32171

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (2 + i) * (3 + i) = 5 + 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l321_32171


namespace NUMINAMATH_CALUDE_oranges_left_uneaten_l321_32104

theorem oranges_left_uneaten (total : Nat) (ripe_fraction : Rat) (ripe_eaten_fraction : Rat) (unripe_eaten_fraction : Rat) :
  total = 96 →
  ripe_fraction = 1/2 →
  ripe_eaten_fraction = 1/4 →
  unripe_eaten_fraction = 1/8 →
  total - (ripe_fraction * total * ripe_eaten_fraction + (1 - ripe_fraction) * total * unripe_eaten_fraction) = 78 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_uneaten_l321_32104


namespace NUMINAMATH_CALUDE_final_expression_l321_32113

theorem final_expression (x y : ℕ) : x + 2*y + x + 3*y + x + 4*y + x + y = 4*x + 10*y := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l321_32113


namespace NUMINAMATH_CALUDE_shooting_competition_stats_l321_32144

/-- Represents the number of shots made and the corresponding number of students -/
structure ShotData :=
  (shots : ℕ)
  (students : ℕ)

/-- Calculates the median of a list of numbers -/
def median (l : List ℕ) : ℚ :=
  sorry

/-- Calculates the mode of a list of numbers -/
def mode (l : List ℕ) : ℕ :=
  sorry

/-- Expands a list of ShotData into a list of individual shots -/
def expandShots (data : List ShotData) : List ℕ :=
  sorry

theorem shooting_competition_stats :
  let data : List ShotData := [
    ⟨6, 2⟩,
    ⟨7, 2⟩,
    ⟨8, 3⟩,
    ⟨9, 1⟩
  ]
  let shots := expandShots data
  median shots = 15/2 ∧ mode shots = 8 := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_stats_l321_32144


namespace NUMINAMATH_CALUDE_factors_of_N_l321_32167

/-- The number of natural-number factors of N, where N = 2^5 * 3^4 * 5^3 * 7^2 * 11^1 -/
def number_of_factors (N : ℕ) : ℕ :=
  (5 + 1) * (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1)

/-- Theorem stating that the number of natural-number factors of N is 720 -/
theorem factors_of_N :
  let N : ℕ := 2^5 * 3^4 * 5^3 * 7^2 * 11^1
  number_of_factors N = 720 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_N_l321_32167


namespace NUMINAMATH_CALUDE_problem_solution_l321_32195

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3*x - 8
def h (r : ℝ) (x : ℝ) : ℝ := 3*x - r

theorem problem_solution :
  (f 2 = 4 ∧ g (f 2) = 4) ∧
  (∀ x : ℝ, f (g x) = g (f x) ↔ x = 2 ∨ x = 6) ∧
  (∀ r : ℝ, f (h r 2) = h r (f 2) ↔ r = 3 ∨ r = 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l321_32195


namespace NUMINAMATH_CALUDE_game_end_not_one_l321_32156

/-- Represents the state of the board with the number of ones and twos -/
structure BoardState where
  ones : Nat
  twos : Nat

/-- Represents a move in the game -/
inductive Move
  | SameDigits : Move
  | DifferentDigits : Move

/-- Applies a move to the board state -/
def applyMove (state : BoardState) (move : Move) : BoardState :=
  match move with
  | Move.SameDigits => 
    if state.ones ≥ 2 then BoardState.mk (state.ones - 2) (state.twos + 1)
    else BoardState.mk state.ones (state.twos - 1)
  | Move.DifferentDigits => 
    if state.ones > 0 && state.twos > 0 
    then BoardState.mk state.ones state.twos
    else state -- This case should not occur in a valid game

/-- The theorem stating that if we start with an even number of ones, 
    the game cannot end with a single one -/
theorem game_end_not_one (initialOnes : Nat) (initialTwos : Nat) :
  initialOnes % 2 = 0 → 
  ∀ (moves : List Move), 
    let finalState := moves.foldl applyMove (BoardState.mk initialOnes initialTwos)
    finalState.ones + finalState.twos = 1 → finalState.ones ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_game_end_not_one_l321_32156


namespace NUMINAMATH_CALUDE_mean_height_of_players_l321_32115

def heights : List ℕ := [47, 48, 50, 51, 51, 54, 55, 56, 56, 57, 61, 63, 64, 64, 65, 67]

theorem mean_height_of_players : 
  (heights.sum : ℚ) / heights.length = 56.8125 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_of_players_l321_32115


namespace NUMINAMATH_CALUDE_initial_women_count_l321_32100

theorem initial_women_count (x y : ℕ) : 
  (y / (x - 15) = 2) → 
  ((y - 45) / (x - 15) = 1 / 5) → 
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_women_count_l321_32100


namespace NUMINAMATH_CALUDE_inequality_and_minimum_l321_32114

theorem inequality_and_minimum (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  let f := fun (t : ℝ) => 2 / t + 9 / (1 - 2 * t)
  -- Part I: Inequality and equality condition
  (a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧
  (a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ a * y = b * x) ∧
  -- Part II: Minimum value and x value for minimum
  (∀ t ∈ Set.Ioo 0 (1/2), f t ≥ 25) ∧
  (f (1/5) = 25) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_l321_32114


namespace NUMINAMATH_CALUDE_dance_attendance_l321_32162

theorem dance_attendance (girls : ℕ) (boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l321_32162


namespace NUMINAMATH_CALUDE_jerry_total_miles_l321_32178

/-- The total miles Jerry walked over three days -/
def total_miles (monday tuesday wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem stating that Jerry walked 45 miles in total -/
theorem jerry_total_miles :
  total_miles 15 18 12 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jerry_total_miles_l321_32178


namespace NUMINAMATH_CALUDE_student_count_l321_32106

theorem student_count (band : ℕ) (sports : ℕ) (both : ℕ) (total : ℕ) : 
  band = 85 → 
  sports = 200 → 
  both = 60 → 
  total = 225 → 
  band + sports - both = total :=
by sorry

end NUMINAMATH_CALUDE_student_count_l321_32106


namespace NUMINAMATH_CALUDE_function_range_exclusion_l321_32148

theorem function_range_exclusion (a : ℕ) : 
  (a > 3 → ∃ x : ℝ, -4 ≤ (8*x - 20) / (a - x^2) ∧ (8*x - 20) / (a - x^2) ≤ -1) ∧ 
  (∀ x : ℝ, (8*x - 20) / (3 - x^2) < -4 ∨ (8*x - 20) / (3 - x^2) > -1) :=
sorry

end NUMINAMATH_CALUDE_function_range_exclusion_l321_32148


namespace NUMINAMATH_CALUDE_fence_cost_l321_32154

/-- The cost of building a fence around a circular plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 289 →
  price_per_foot = 58 →
  cost = 2 * (Real.sqrt (289 * Real.pi)) * price_per_foot →
  cost = 1972 :=
by
  sorry

#check fence_cost

end NUMINAMATH_CALUDE_fence_cost_l321_32154


namespace NUMINAMATH_CALUDE_nba_conference_scheduling_impossibility_l321_32145

theorem nba_conference_scheduling_impossibility 
  (total_teams : ℕ) 
  (games_per_team : ℕ) 
  (h1 : total_teams = 30) 
  (h2 : games_per_team = 82) : 
  ¬ ∃ (eastern_teams western_teams : ℕ) 
      (intra_eastern intra_western inter_conference : ℕ),
    eastern_teams + western_teams = total_teams ∧
    2 * (intra_eastern + intra_western + inter_conference) = total_teams * games_per_team ∧
    2 * inter_conference = total_teams * games_per_team := by
  sorry

end NUMINAMATH_CALUDE_nba_conference_scheduling_impossibility_l321_32145


namespace NUMINAMATH_CALUDE_solution_characterization_l321_32107

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The plane z = x -/
def midplane (p : Point3D) : Prop := p.z = p.x

/-- The sphere with center A and radius r -/
def sphere (A : Point3D) (r : ℝ) (p : Point3D) : Prop :=
  (p.x - A.x)^2 + (p.y - A.y)^2 + (p.z - A.z)^2 = r^2

/-- The set of points satisfying both conditions -/
def solution_set (A : Point3D) (r : ℝ) : Set Point3D :=
  {p : Point3D | sphere A r p ∧ midplane p}

theorem solution_characterization (A : Point3D) (r : ℝ) :
  ∀ p : Point3D, p ∈ solution_set A r ↔ 
    (p.x - A.x)^2 + (p.y - A.y)^2 + (p.x - A.z)^2 = r^2 :=
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l321_32107


namespace NUMINAMATH_CALUDE_heath_carrot_planting_l321_32142

/-- Given the conditions of Heath's carrot planting, prove the number of plants in each row. -/
theorem heath_carrot_planting 
  (total_rows : ℕ) 
  (planting_time : ℕ) 
  (planting_rate : ℕ) 
  (h1 : total_rows = 400)
  (h2 : planting_time = 20)
  (h3 : planting_rate = 6000) :
  (planting_time * planting_rate) / total_rows = 300 := by
  sorry

end NUMINAMATH_CALUDE_heath_carrot_planting_l321_32142


namespace NUMINAMATH_CALUDE_greg_trip_distance_l321_32119

/-- Represents Greg's trip with given distances and speeds -/
structure GregTrip where
  workplace_to_market : ℝ
  market_to_friend : ℝ
  friend_to_aunt : ℝ
  aunt_to_grocery : ℝ
  grocery_to_home : ℝ

/-- Calculates the total distance of Greg's trip -/
def total_distance (trip : GregTrip) : ℝ :=
  trip.workplace_to_market + trip.market_to_friend + trip.friend_to_aunt + 
  trip.aunt_to_grocery + trip.grocery_to_home

/-- Theorem stating that Greg's total trip distance is 100 miles -/
theorem greg_trip_distance :
  ∃ (trip : GregTrip),
    trip.workplace_to_market = 30 ∧
    trip.market_to_friend = trip.workplace_to_market + 10 ∧
    trip.friend_to_aunt = 5 ∧
    trip.aunt_to_grocery = 7 ∧
    trip.grocery_to_home = 18 ∧
    total_distance trip = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_greg_trip_distance_l321_32119


namespace NUMINAMATH_CALUDE_middle_truncated_cone_volume_middle_truncated_cone_volume_is_7V_div_27_l321_32196

/-- Given a cone with volume V whose height is divided into three equal parts by planes parallel to the base, the volume of the middle truncated cone is 7V/27. -/
theorem middle_truncated_cone_volume (V : ℝ) (h : V > 0) : ℝ :=
  let cone_volume := V
  let height_parts := 3
  let middle_truncated_cone_volume := (7 : ℝ) / 27 * V
  middle_truncated_cone_volume

/-- The volume of the middle truncated cone is 7V/27 -/
theorem middle_truncated_cone_volume_is_7V_div_27 (V : ℝ) (h : V > 0) :
  middle_truncated_cone_volume V h = (7 : ℝ) / 27 * V := by
  sorry

end NUMINAMATH_CALUDE_middle_truncated_cone_volume_middle_truncated_cone_volume_is_7V_div_27_l321_32196


namespace NUMINAMATH_CALUDE_two_numbers_difference_l321_32143

theorem two_numbers_difference (x y : ℤ) 
  (sum_eq : x + y = 40)
  (triple_minus_double : 3 * max x y - 2 * min x y = 8) :
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l321_32143


namespace NUMINAMATH_CALUDE_f_is_odd_and_satisfies_conditions_l321_32172

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x + x - 2
  else if x = 0 then 0
  else -2^(-x) + x + 2

-- Theorem statement
theorem f_is_odd_and_satisfies_conditions :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x > 0, f x = 2^x + x - 2) ∧
  (f 0 = 0) ∧
  (∀ x < 0, f x = -2^(-x) + x + 2) := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_and_satisfies_conditions_l321_32172


namespace NUMINAMATH_CALUDE_workers_in_first_group_l321_32180

/-- The number of workers in the first group -/
def W : ℕ := 70

/-- The time taken by the first group to complete the job (in hours) -/
def T1 : ℕ := 3

/-- The number of workers in the second group -/
def W2 : ℕ := 30

/-- The time taken by the second group to complete the job (in hours) -/
def T2 : ℕ := 7

/-- The amount of work done (assumed to be constant for both groups) -/
def work : ℕ := W * T1

theorem workers_in_first_group :
  (W * T1 = W2 * T2) ∧ (W * T2 = W2 * T1) → W = 70 := by
  sorry

end NUMINAMATH_CALUDE_workers_in_first_group_l321_32180


namespace NUMINAMATH_CALUDE_candy_inconsistency_l321_32168

theorem candy_inconsistency :
  ¬∃ (K Y N B : ℕ),
    K + Y + N = 120 ∧
    N + B = 103 ∧
    K + Y + B = 152 :=
by sorry

end NUMINAMATH_CALUDE_candy_inconsistency_l321_32168


namespace NUMINAMATH_CALUDE_function_with_period_3_is_periodic_l321_32161

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

theorem function_with_period_3_is_periodic (f : ℝ → ℝ) 
  (h : ∀ x, f (x + 3) = f x) : is_periodic f := by
  sorry

end NUMINAMATH_CALUDE_function_with_period_3_is_periodic_l321_32161


namespace NUMINAMATH_CALUDE_remainder_div_nine_l321_32190

theorem remainder_div_nine (n : ℕ) (h : n % 18 = 11) : n % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_div_nine_l321_32190


namespace NUMINAMATH_CALUDE_last_day_third_quarter_common_year_l321_32103

/-- Represents a day in a month -/
structure DayInMonth where
  month : Nat
  day : Nat

/-- Definition of a common year -/
def isCommonYear (totalDays : Nat) : Prop := totalDays = 365

/-- Definition of the third quarter -/
def isInThirdQuarter (d : DayInMonth) : Prop :=
  d.month ∈ [7, 8, 9]

/-- The last day of the third quarter in a common year -/
theorem last_day_third_quarter_common_year (totalDays : Nat) 
  (h : isCommonYear totalDays) :
  ∃ (d : DayInMonth), 
    isInThirdQuarter d ∧ 
    d.month = 9 ∧ 
    d.day = 30 ∧ 
    (∀ (d' : DayInMonth), isInThirdQuarter d' → d'.month < d.month ∨ (d'.month = d.month ∧ d'.day ≤ d.day)) :=
sorry

end NUMINAMATH_CALUDE_last_day_third_quarter_common_year_l321_32103


namespace NUMINAMATH_CALUDE_work_completion_time_l321_32146

/-- Given two workers A and B, where A can complete a work in 10 days and B can complete the same work in 7 days, 
    this theorem proves that A and B working together can complete the work in 70/17 days. -/
theorem work_completion_time 
  (work : ℝ) -- Total amount of work
  (a_rate : ℝ) -- A's work rate
  (b_rate : ℝ) -- B's work rate
  (ha : a_rate = work / 10) -- A completes the work in 10 days
  (hb : b_rate = work / 7)  -- B completes the work in 7 days
  : work / (a_rate + b_rate) = 70 / 17 := by
sorry


end NUMINAMATH_CALUDE_work_completion_time_l321_32146


namespace NUMINAMATH_CALUDE_labor_practice_problem_l321_32121

-- Define the problem parameters
def type_a_capacity : ℕ := 35
def type_b_capacity : ℕ := 30
def type_a_rental : ℕ := 400
def type_b_rental : ℕ := 320
def max_rental : ℕ := 3000

-- Define the theorem
theorem labor_practice_problem :
  ∃ (teachers students : ℕ) (type_a_buses : ℕ),
    -- Conditions from the problem
    students = 30 * teachers + 7 ∧
    31 * teachers = students + 1 ∧
    -- Solution part 1
    teachers = 8 ∧
    students = 247 ∧
    -- Solution part 2
    3 ≤ type_a_buses ∧ type_a_buses ≤ 5 ∧
    type_a_capacity * type_a_buses + type_b_capacity * (teachers - type_a_buses) ≥ teachers + students ∧
    type_a_rental * type_a_buses + type_b_rental * (teachers - type_a_buses) ≤ max_rental ∧
    -- Solution part 3
    (∀ m : ℕ, 3 ≤ m ∧ m ≤ 5 →
      type_a_rental * 3 + type_b_rental * 5 ≤ type_a_rental * m + type_b_rental * (8 - m)) :=
by sorry


end NUMINAMATH_CALUDE_labor_practice_problem_l321_32121


namespace NUMINAMATH_CALUDE_quintuplet_babies_count_l321_32131

/-- Represents the number of sets of a given multiple birth type -/
structure MultipleBirthSets where
  twins : ℕ
  triplets : ℕ
  quintuplets : ℕ

/-- Calculates the total number of babies from multiple birth sets -/
def totalBabies (s : MultipleBirthSets) : ℕ :=
  2 * s.twins + 3 * s.triplets + 5 * s.quintuplets

theorem quintuplet_babies_count (s : MultipleBirthSets) :
  s.triplets = 6 * s.quintuplets →
  s.twins = 2 * s.triplets →
  totalBabies s = 1500 →
  5 * s.quintuplets = 160 := by
sorry

end NUMINAMATH_CALUDE_quintuplet_babies_count_l321_32131


namespace NUMINAMATH_CALUDE_remaining_work_days_l321_32193

/-- Given two workers x and y, where x can finish a job in 36 days and y in 24 days,
    prove that x needs 18 days to finish the remaining work after y worked for 12 days. -/
theorem remaining_work_days (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 36) (hy : y_days = 24) (hw : y_worked_days = 12) : 
  (x_days : ℚ) / 2 = 18 := by
  sorry

#check remaining_work_days

end NUMINAMATH_CALUDE_remaining_work_days_l321_32193


namespace NUMINAMATH_CALUDE_sequence_problem_l321_32132

/-- Given a sequence where each term is obtained by doubling the previous term and adding 4,
    if the third term is 52, then the first term is 10. -/
theorem sequence_problem (x : ℝ) : 
  let second_term := 2 * x + 4
  let third_term := 2 * second_term + 4
  third_term = 52 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l321_32132


namespace NUMINAMATH_CALUDE_cubic_equation_roots_of_unity_l321_32149

theorem cubic_equation_roots_of_unity :
  ∃ (a b c : ℤ), (1 : ℂ)^3 + a*(1 : ℂ)^2 + b*(1 : ℂ) + c = 0 ∧
                 (-1 : ℂ)^3 + a*(-1 : ℂ)^2 + b*(-1 : ℂ) + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_of_unity_l321_32149


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l321_32163

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_a1 : a 1 = 2)
  (h_sum : a 2 + a 3 = 13) :
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l321_32163


namespace NUMINAMATH_CALUDE_after_school_program_enrollment_l321_32185

theorem after_school_program_enrollment (drama_students music_students both_students : ℕ) 
  (h1 : drama_students = 41)
  (h2 : music_students = 28)
  (h3 : both_students = 15) :
  drama_students + music_students - both_students = 54 := by
sorry

end NUMINAMATH_CALUDE_after_school_program_enrollment_l321_32185


namespace NUMINAMATH_CALUDE_merchant_profit_l321_32153

theorem merchant_profit (cost : ℝ) (selling_price : ℝ) : 
  cost = 30 ∧ selling_price = 39 → 
  selling_price = cost + (cost * cost / 100) := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l321_32153


namespace NUMINAMATH_CALUDE_sqrt_neg_three_squared_l321_32197

theorem sqrt_neg_three_squared : Real.sqrt ((-3)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_three_squared_l321_32197


namespace NUMINAMATH_CALUDE_sin_plus_cos_range_l321_32117

theorem sin_plus_cos_range : ∀ x : ℝ, -Real.sqrt 2 ≤ Real.sin x + Real.cos x ∧ Real.sin x + Real.cos x ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_range_l321_32117


namespace NUMINAMATH_CALUDE_concert_theorem_l321_32135

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ
  elsa : ℕ

/-- The conditions of the problem -/
def concert_conditions (s : SongCounts) : Prop :=
  s.hanna = 9 ∧ 
  s.mary = 3 ∧ 
  s.alina + s.tina = 16 ∧
  s.hanna > s.alina ∧ s.hanna > s.tina ∧ s.hanna > s.elsa ∧
  s.alina > s.mary ∧ s.tina > s.mary ∧ s.elsa > s.mary

/-- The total number of songs sung -/
def total_songs (s : SongCounts) : ℕ :=
  (s.mary + s.alina + s.tina + s.hanna + s.elsa) / 4

/-- The main theorem: given the conditions, the total number of songs is 8 -/
theorem concert_theorem (s : SongCounts) : 
  concert_conditions s → total_songs s = 8 := by
  sorry

end NUMINAMATH_CALUDE_concert_theorem_l321_32135


namespace NUMINAMATH_CALUDE_rainy_days_count_l321_32122

/-- Proves that the number of rainy days in a week is 2, given the conditions of Mo's drinking habits. -/
theorem rainy_days_count (n : ℤ) : 
  (∃ (R NR : ℕ), 
    R + NR = 7 ∧ 
    n * R + 3 * NR = 20 ∧ 
    3 * NR = n * R + 10) → 
  (∃ (R : ℕ), R = 2) :=
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l321_32122


namespace NUMINAMATH_CALUDE_intersection_characterization_l321_32120

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x + 1 ≥ 0}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- Define the intersection of A and B
def A_inter_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_characterization : 
  A_inter_B = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_characterization_l321_32120


namespace NUMINAMATH_CALUDE_alberts_cabbage_patch_l321_32128

/-- Albert's cabbage patch problem -/
theorem alberts_cabbage_patch (rows : ℕ) (heads_per_row : ℕ) 
  (h1 : rows = 12) (h2 : heads_per_row = 15) : 
  rows * heads_per_row = 180 := by
  sorry

end NUMINAMATH_CALUDE_alberts_cabbage_patch_l321_32128


namespace NUMINAMATH_CALUDE_square_of_1023_l321_32159

theorem square_of_1023 : (1023 : ℕ)^2 = 1045529 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l321_32159


namespace NUMINAMATH_CALUDE_sum_congruence_l321_32124

theorem sum_congruence : 
  (2 + 333 + 5555 + 77777 + 999999 + 11111111 + 222222222) % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l321_32124


namespace NUMINAMATH_CALUDE_age_difference_l321_32184

theorem age_difference (a b c : ℕ) : 
  b = 10 →
  b = 2 * c →
  a + b + c = 27 →
  a = b + 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l321_32184


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l321_32141

/-- 
Given a sum of money that doubles itself in 10 years at simple interest,
prove that the rate percent per annum is 10%.
-/
theorem simple_interest_rate_for_doubling (P : ℝ) (h : P > 0) : 
  ∃ (R : ℝ), R > 0 ∧ R ≤ 100 ∧ P + (P * R * 10) / 100 = 2 * P ∧ R = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l321_32141


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l321_32125

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 1) :
  -1 < a - b ∧ a - b < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l321_32125


namespace NUMINAMATH_CALUDE_paint_area_calculation_l321_32140

/-- Calculates the area to be painted on a wall with a door. -/
def areaToPaint (wallHeight wallLength doorHeight doorWidth : ℝ) : ℝ :=
  wallHeight * wallLength - doorHeight * doorWidth

/-- Proves that the area to be painted on a 10ft by 15ft wall with a 3ft by 5ft door is 135 sq ft. -/
theorem paint_area_calculation :
  areaToPaint 10 15 3 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_paint_area_calculation_l321_32140


namespace NUMINAMATH_CALUDE_inequality_statements_l321_32166

theorem inequality_statements :
  (∀ a b c : ℝ, c ≠ 0 → (a * c^2 < b * c^2 → a < b)) ∧
  (∃ a x y : ℝ, x > y ∧ ¬(-a^2 * x < -a^2 * y)) ∧
  (∀ a b c : ℝ, c ≠ 0 → (a / c^2 < b / c^2 → a < b)) ∧
  (∀ a b : ℝ, a > b → 2 - a < 2 - b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_statements_l321_32166


namespace NUMINAMATH_CALUDE_train_passing_bridge_l321_32136

/-- Time for a train to pass a bridge -/
theorem train_passing_bridge (train_length : Real) (train_speed_kmph : Real) (bridge_length : Real) :
  train_length = 360 ∧ 
  train_speed_kmph = 45 ∧ 
  bridge_length = 140 →
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_bridge_l321_32136


namespace NUMINAMATH_CALUDE_equation_solution_l321_32176

theorem equation_solution :
  ∃ x : ℝ, x ≠ 0 ∧ (2 / x + 3 * ((4 / x) / (8 / x)) = 1.2) ∧ x = -20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l321_32176


namespace NUMINAMATH_CALUDE_g_property_g_2022_l321_32150

/-- A function g that satisfies the given property for all real x and y -/
def g : ℝ → ℝ := fun x ↦ 2021 * x

/-- The theorem stating that g satisfies the required property -/
theorem g_property : ∀ x y : ℝ, g (x - y) = g x + g y - 2021 * (x + y) := by sorry

/-- The main theorem proving that g(2022) equals 4086462 -/
theorem g_2022 : g 2022 = 4086462 := by sorry

end NUMINAMATH_CALUDE_g_property_g_2022_l321_32150


namespace NUMINAMATH_CALUDE_no_zero_root_for_equations_l321_32157

theorem no_zero_root_for_equations :
  (∀ x : ℝ, 3 * x^2 - 5 = 50 → x ≠ 0) ∧
  (∀ x : ℝ, (3*x - 2)^2 = (x - 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, x^2 - 15 = x + 2 → x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_zero_root_for_equations_l321_32157


namespace NUMINAMATH_CALUDE_james_fish_catch_l321_32118

/-- The total pounds of fish James caught -/
def total_fish (trout salmon tuna : ℕ) : ℕ := trout + salmon + tuna

/-- Proves that James caught 900 pounds of fish in total -/
theorem james_fish_catch : 
  let trout : ℕ := 200
  let salmon : ℕ := trout + trout / 2
  let tuna : ℕ := 2 * trout
  total_fish trout salmon tuna = 900 := by sorry

end NUMINAMATH_CALUDE_james_fish_catch_l321_32118


namespace NUMINAMATH_CALUDE_class_grade_average_l321_32189

theorem class_grade_average (N : ℕ) (X : ℝ) : 
  (X * N + 45 * (2 * N)) / (3 * N) = 48 → X = 54 := by
  sorry

end NUMINAMATH_CALUDE_class_grade_average_l321_32189


namespace NUMINAMATH_CALUDE_no_divisible_by_99_ab32_l321_32123

theorem no_divisible_by_99_ab32 : ∀ a b : ℕ, 
  0 ≤ a ∧ a ≤ 9 → 0 ≤ b ∧ b ≤ 9 → 
  ¬(∃ k : ℕ, 1000 * a + 100 * b + 32 = 99 * k) := by
sorry

end NUMINAMATH_CALUDE_no_divisible_by_99_ab32_l321_32123


namespace NUMINAMATH_CALUDE_max_value_on_circle_l321_32105

theorem max_value_on_circle : 
  ∃ (M : ℝ), M = 8 ∧ 
  (∀ (x y : ℝ), (x - 2)^2 + y^2 = 1 → |3*x + 4*y - 3| ≤ M) ∧
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 1 ∧ |3*x + 4*y - 3| = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l321_32105


namespace NUMINAMATH_CALUDE_one_thirds_in_eleven_fifths_l321_32138

theorem one_thirds_in_eleven_fifths : (11 / 5 : ℚ) / (1 / 3 : ℚ) = 33 / 5 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_eleven_fifths_l321_32138
