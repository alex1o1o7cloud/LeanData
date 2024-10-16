import Mathlib

namespace NUMINAMATH_CALUDE_sector_inscribed_circle_area_ratio_l671_67117

theorem sector_inscribed_circle_area_ratio (α : Real) :
  let R := 1  -- We can set R to 1 without loss of generality
  let r := (R * Real.sin (α / 2)) / (1 + Real.sin (α / 2))
  let sector_area := (1 / 2) * R^2 * α
  let inscribed_circle_area := Real.pi * r^2
  sector_area / inscribed_circle_area = (2 * α * (Real.cos (Real.pi / 4 - α / 4))^2) / (Real.pi * (Real.sin (α / 2))^2) :=
by sorry

end NUMINAMATH_CALUDE_sector_inscribed_circle_area_ratio_l671_67117


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l671_67115

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
    (∀ (d e f : ℕ+), (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (d * Real.sqrt 3 + e * Real.sqrt 11) / f) → c ≤ f)) →
  a + b + c = 113 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l671_67115


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l671_67170

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m l : Line) (α β : Plane)
  (h1 : parallel m l)
  (h2 : perpendicular l β)
  (h3 : subset m α) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l671_67170


namespace NUMINAMATH_CALUDE_max_page_number_with_25_threes_l671_67195

/-- Counts the occurrences of a specific digit in a number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Counts the total occurrences of a specific digit in numbers from 1 to n -/
def countDigitUpTo (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The maximum page number that can be reached with a given number of '3's -/
def maxPageNumber (threes : ℕ) : ℕ := sorry

theorem max_page_number_with_25_threes :
  maxPageNumber 25 = 139 := by sorry

end NUMINAMATH_CALUDE_max_page_number_with_25_threes_l671_67195


namespace NUMINAMATH_CALUDE_used_car_clients_l671_67118

theorem used_car_clients (num_cars : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) : 
  num_cars = 10 → cars_per_client = 2 → selections_per_car = 3 →
  (num_cars * selections_per_car) / cars_per_client = 15 := by
  sorry

end NUMINAMATH_CALUDE_used_car_clients_l671_67118


namespace NUMINAMATH_CALUDE_fish_aquarium_problem_l671_67198

theorem fish_aquarium_problem (x y : ℕ) :
  x + y = 100 ∧ x - 30 = y - 40 → x = 45 ∧ y = 55 := by
  sorry

end NUMINAMATH_CALUDE_fish_aquarium_problem_l671_67198


namespace NUMINAMATH_CALUDE_largest_percent_error_circle_area_l671_67143

/-- The largest possible percent error in the computed area of a circle -/
theorem largest_percent_error_circle_area (actual_circumference : ℝ) (max_error_percent : ℝ) :
  actual_circumference = 30 →
  max_error_percent = 15 →
  ∃ (computed_area actual_area : ℝ),
    computed_area ≠ actual_area ∧
    abs ((computed_area - actual_area) / actual_area) ≤ 0.3225 ∧
    ∀ (other_area : ℝ),
      abs ((other_area - actual_area) / actual_area) ≤ abs ((computed_area - actual_area) / actual_area) :=
by sorry

end NUMINAMATH_CALUDE_largest_percent_error_circle_area_l671_67143


namespace NUMINAMATH_CALUDE_stock_worth_l671_67101

-- Define the total worth of the stock
variable (X : ℝ)

-- Define the profit percentage on 20% of stock
def profit_percent : ℝ := 0.10

-- Define the loss percentage on 80% of stock
def loss_percent : ℝ := 0.05

-- Define the overall loss
def overall_loss : ℝ := 200

-- Theorem statement
theorem stock_worth :
  (0.20 * X * (1 + profit_percent) + 0.80 * X * (1 - loss_percent) = X - overall_loss) →
  X = 10000 := by
sorry

end NUMINAMATH_CALUDE_stock_worth_l671_67101


namespace NUMINAMATH_CALUDE_base_subtraction_l671_67177

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement --/
theorem base_subtraction :
  let base_8_num := to_base_10 [0, 1, 2, 3, 4, 5] 8
  let base_9_num := to_base_10 [2, 3, 4, 5, 6] 9
  base_8_num - base_9_num = 136532 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l671_67177


namespace NUMINAMATH_CALUDE_water_bottles_cost_l671_67175

/-- The total cost of water bottles given the number of bottles, liters per bottle, and price per liter. -/
def total_cost (num_bottles : ℕ) (liters_per_bottle : ℕ) (price_per_liter : ℕ) : ℕ :=
  num_bottles * liters_per_bottle * price_per_liter

/-- Theorem stating that the total cost of six 2-liter bottles of water is $12 when the price is $1 per liter. -/
theorem water_bottles_cost :
  total_cost 6 2 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_cost_l671_67175


namespace NUMINAMATH_CALUDE_rice_yield_increase_l671_67140

theorem rice_yield_increase : 
  let yield_changes : List Int := [50, -35, 10, -16, 27, -5, -20, 35]
  yield_changes.sum = 46 := by sorry

end NUMINAMATH_CALUDE_rice_yield_increase_l671_67140


namespace NUMINAMATH_CALUDE_line_translation_coincidence_l671_67180

/-- 
Given a line y = kx + 2 in the Cartesian plane,
prove that if the line is translated upward by 3 units
and then rightward by 2 units, and the resulting line
coincides with the original line, then k = 3/2.
-/
theorem line_translation_coincidence (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 2 ↔ y = k * (x - 2) + 5) → k = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_coincidence_l671_67180


namespace NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_four_l671_67124

theorem no_real_roots_x_squared_plus_four :
  ¬ ∃ (x : ℝ), x^2 + 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_four_l671_67124


namespace NUMINAMATH_CALUDE_three_digit_number_divided_by_11_l671_67163

theorem three_digit_number_divided_by_11 : 
  ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 → 
  (n / 11 = (n / 100)^2 + ((n / 10) % 10)^2 + (n % 10)^2) ↔ 
  (n = 550 ∨ n = 803) := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_divided_by_11_l671_67163


namespace NUMINAMATH_CALUDE_quadratic_inequality_result_l671_67158

theorem quadratic_inequality_result (y : ℝ) (h : y^2 - 7*y + 12 < 0) :
  44 < y^2 + 7*y + 14 ∧ y^2 + 7*y + 14 < 58 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_result_l671_67158


namespace NUMINAMATH_CALUDE_school_bus_distance_l671_67172

/-- Calculates the total distance traveled by a school bus under specific conditions -/
theorem school_bus_distance : 
  let initial_velocity := 0
  let acceleration := 2
  let acceleration_time := 30
  let constant_speed_time := 20 * 60
  let deceleration := 1
  let final_velocity := acceleration * acceleration_time
  let distance_constant_speed := final_velocity * constant_speed_time
  let distance_deceleration := final_velocity^2 / (2 * deceleration)
  distance_constant_speed + distance_deceleration = 73800 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_distance_l671_67172


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l671_67152

-- Problem 1
theorem problem_one : (9/4)^(3/2) - (-9.6)^0 - (27/8)^(2/3) + (3/2)^(-2) = 1/2 := by
  sorry

-- Problem 2
theorem problem_two (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l671_67152


namespace NUMINAMATH_CALUDE_five_fold_f_of_one_l671_67142

def f (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem five_fold_f_of_one : f (f (f (f (f 1)))) = 4687 := by
  sorry

end NUMINAMATH_CALUDE_five_fold_f_of_one_l671_67142


namespace NUMINAMATH_CALUDE_even_function_range_l671_67112

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem even_function_range (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 2, f a b x = f a b (-x)) →
  Set.range (f a b) = Set.Icc (-10) 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_range_l671_67112


namespace NUMINAMATH_CALUDE_min_speed_to_arrive_earlier_l671_67174

/-- Proves the minimum speed required for the second person to arrive before the first person --/
theorem min_speed_to_arrive_earlier (distance : ℝ) (speed_A : ℝ) (delay : ℝ) :
  distance = 120 →
  speed_A = 30 →
  delay = 1.5 →
  ∀ speed_B : ℝ, speed_B > 48 → 
    distance / speed_B < distance / speed_A - delay := by
  sorry

end NUMINAMATH_CALUDE_min_speed_to_arrive_earlier_l671_67174


namespace NUMINAMATH_CALUDE_log_equation_solution_l671_67122

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l671_67122


namespace NUMINAMATH_CALUDE_smallest_positive_angle_solution_l671_67153

/-- The equation that needs to be satisfied -/
def equation (y : ℝ) : Prop :=
  6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 1

/-- The smallest positive angle in degrees that satisfies the equation -/
def smallest_angle : ℝ := 10.4525

theorem smallest_positive_angle_solution :
  equation (smallest_angle * π / 180) ∧
  ∀ y, 0 < y ∧ y < smallest_angle * π / 180 → ¬equation y :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_solution_l671_67153


namespace NUMINAMATH_CALUDE_zoo_animals_l671_67103

theorem zoo_animals (M P L : ℕ) : 
  (26 ≤ M + P + L ∧ M + P + L ≤ 32) →
  M + L > P →
  P + L = 2 * M →
  M + P > 3 * L →
  P < 2 * L →
  P = 12 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l671_67103


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l671_67126

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + a*c = 131) : 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l671_67126


namespace NUMINAMATH_CALUDE_oplus_inequality_range_l671_67190

def oplus (x y : ℝ) : ℝ := x * (2 - y)

theorem oplus_inequality_range (a : ℝ) :
  (∀ t : ℝ, oplus (t - a) (t + a) < 1) ↔ (0 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_oplus_inequality_range_l671_67190


namespace NUMINAMATH_CALUDE_unique_solution_condition_l671_67128

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 5)*(x - 6) = -53 + k*x + x^2) ↔ (k = -1 ∨ k = -25) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l671_67128


namespace NUMINAMATH_CALUDE_three_students_four_groups_signup_ways_l671_67166

/-- The number of different ways for students to sign up for interest groups -/
def signUpWays (numStudents : ℕ) (numGroups : ℕ) : ℕ :=
  numGroups ^ numStudents

/-- Theorem: 3 students signing up for 4 interest groups results in 64 different ways -/
theorem three_students_four_groups_signup_ways :
  signUpWays 3 4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_three_students_four_groups_signup_ways_l671_67166


namespace NUMINAMATH_CALUDE_unique_solution_k_values_l671_67191

theorem unique_solution_k_values (k : ℝ) :
  (∃! x : ℝ, 1 ≤ k * x^2 + 2 ∧ x + k ≤ 2) ↔ 
  (k = 1 + Real.sqrt 2 ∨ k = (1 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_k_values_l671_67191


namespace NUMINAMATH_CALUDE_lyle_notebook_cost_l671_67164

/-- The cost of a pen in dollars -/
def pen_cost : ℝ := 1.50

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := 3 * pen_cost

/-- The number of notebooks Lyle wants to buy -/
def num_notebooks : ℕ := 4

/-- The total cost of notebooks Lyle will pay -/
def total_cost : ℝ := num_notebooks * notebook_cost

theorem lyle_notebook_cost : total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_lyle_notebook_cost_l671_67164


namespace NUMINAMATH_CALUDE_not_always_cylinder_l671_67100

/-- A cylinder in 3D space -/
structure Cylinder where
  base : Set (ℝ × ℝ)  -- Base of the cylinder
  height : ℝ          -- Height of the cylinder

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ  -- Normal vector of the plane
  point : ℝ × ℝ × ℝ   -- A point on the plane

/-- Two planes are parallel if their normal vectors are parallel -/
def parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), p1.normal = k • p2.normal

/-- The result of cutting a cylinder with two parallel planes -/
def cut_cylinder (c : Cylinder) (p1 p2 : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry  -- Definition of the cut cylinder

/-- Theorem: Cutting a cylinder with two arbitrary parallel planes 
    does not always result in a cylinder -/
theorem not_always_cylinder (c : Cylinder) :
  ∃ (p1 p2 : Plane), parallel p1 p2 ∧ ¬∃ (c' : Cylinder), cut_cylinder c p1 p2 = {(x, y, z) | (x, y) ∈ c'.base ∧ 0 ≤ z ∧ z ≤ c'.height} :=
sorry


end NUMINAMATH_CALUDE_not_always_cylinder_l671_67100


namespace NUMINAMATH_CALUDE_division_remainder_problem_l671_67179

theorem division_remainder_problem :
  let dividend : ℕ := 12401
  let divisor : ℕ := 163
  let quotient : ℕ := 76
  dividend = quotient * divisor + 13 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l671_67179


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l671_67135

def sum_of_first_n_even_integers (n : ℕ) : ℕ := 2 * n * (n + 1)

def sum_of_five_consecutive_even_integers (largest : ℕ) : ℕ :=
  (largest - 8) + (largest - 6) + (largest - 4) + (largest - 2) + largest

theorem largest_of_five_consecutive_even_integers :
  ∃ (largest : ℕ), 
    sum_of_first_n_even_integers 15 = sum_of_five_consecutive_even_integers largest ∧
    largest = 52 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l671_67135


namespace NUMINAMATH_CALUDE_remainder_4672_div_34_l671_67189

theorem remainder_4672_div_34 : 4672 % 34 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4672_div_34_l671_67189


namespace NUMINAMATH_CALUDE_decimal_0_03_is_3_percent_l671_67193

/-- Converts a decimal fraction to a percentage -/
def decimal_to_percentage (d : ℝ) : ℝ := d * 100

/-- The decimal fraction we're working with -/
def given_decimal : ℝ := 0.03

/-- Theorem: The percentage equivalent of 0.03 is 3% -/
theorem decimal_0_03_is_3_percent :
  decimal_to_percentage given_decimal = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_0_03_is_3_percent_l671_67193


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l671_67110

theorem complex_modulus_problem (z : ℂ) : 2 + z * Complex.I = z - 2 * Complex.I → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l671_67110


namespace NUMINAMATH_CALUDE_prob_two_tails_after_HHT_is_correct_l671_67131

/-- A fair coin flip sequence that stops when two consecutive heads or tails are obtained -/
def CoinFlipSequence : Type := List Bool

/-- The probability of getting a specific sequence of coin flips -/
def prob_sequence (s : CoinFlipSequence) : ℚ :=
  (1 / 2) ^ s.length

/-- The probability of getting two tails after HHT -/
def prob_two_tails_after_HHT : ℚ :=
  1 / 24

/-- The theorem stating that the probability of getting two tails after HHT is 1/24 -/
theorem prob_two_tails_after_HHT_is_correct :
  prob_two_tails_after_HHT = 1 / 24 := by
  sorry

#check prob_two_tails_after_HHT_is_correct

end NUMINAMATH_CALUDE_prob_two_tails_after_HHT_is_correct_l671_67131


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l671_67137

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth : (12 : ℚ) / (1 / 6) = 72 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l671_67137


namespace NUMINAMATH_CALUDE_tangent_point_and_inequality_condition_l671_67144

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem tangent_point_and_inequality_condition (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, f a x₀ + (a - 1 / x₀) * (x - x₀) = 0 → x = 0) ∧ 
    x₀ = Real.exp 1) ∧
  (∀ x : ℝ, x ≥ 1 → f a x ≥ a * (2 * x - x^2) → a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_and_inequality_condition_l671_67144


namespace NUMINAMATH_CALUDE_sweater_markup_l671_67106

theorem sweater_markup (wholesale : ℝ) (retail : ℝ) (h1 : retail > 0) (h2 : wholesale > 0) :
  (0.4 * retail = 1.35 * wholesale) →
  (retail - wholesale) / wholesale * 100 = 237.5 := by
sorry

end NUMINAMATH_CALUDE_sweater_markup_l671_67106


namespace NUMINAMATH_CALUDE_dividing_line_theorem_l671_67134

/-- A configuration of six unit squares in two rows of three in the coordinate plane -/
structure SquareGrid :=
  (width : ℕ := 3)
  (height : ℕ := 2)

/-- A line extending from (2,0) to (k,k) -/
structure DividingLine :=
  (k : ℝ)

/-- The area above and below the line formed by the DividingLine -/
def areas (grid : SquareGrid) (line : DividingLine) : ℝ × ℝ :=
  sorry

/-- Theorem stating that k = 4 divides the grid such that the area above is twice the area below -/
theorem dividing_line_theorem (grid : SquareGrid) :
  ∃ (line : DividingLine), 
    let (area_below, area_above) := areas grid line
    line.k = 4 ∧ area_above = 2 * area_below := by sorry

end NUMINAMATH_CALUDE_dividing_line_theorem_l671_67134


namespace NUMINAMATH_CALUDE_parallelogram_area_v_w_l671_67171

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : Fin 2 → ℝ) : ℝ :=
  |v 0 * w 1 - v 1 * w 0|

/-- Vectors v and w -/
def v : Fin 2 → ℝ := ![4, -6]
def w : Fin 2 → ℝ := ![7, -1]

/-- Theorem stating that the area of the parallelogram formed by v and w is 38 -/
theorem parallelogram_area_v_w : parallelogramArea v w = 38 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_v_w_l671_67171


namespace NUMINAMATH_CALUDE_quadratic_shift_theorem_l671_67182

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal and vertical shift to a quadratic function -/
def shift_quadratic (f : QuadraticFunction) (h_shift v_shift : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := -2 * f.a * h_shift
  , c := f.a * h_shift^2 + f.c - v_shift }

theorem quadratic_shift_theorem (f : QuadraticFunction) 
  (h : f.a = -2 ∧ f.b = 0 ∧ f.c = 1) : 
  shift_quadratic f 3 2 = { a := -2, b := 12, c := -1 } := by
  sorry

#check quadratic_shift_theorem

end NUMINAMATH_CALUDE_quadratic_shift_theorem_l671_67182


namespace NUMINAMATH_CALUDE_circular_class_properties_l671_67105

/-- Represents a circular seating arrangement of students -/
structure CircularClass where
  totalStudents : ℕ
  boyOppositePositions : (ℕ × ℕ)
  everyOtherIsBoy : Bool

/-- Calculates the number of boys in the class -/
def numberOfBoys (c : CircularClass) : ℕ :=
  c.totalStudents / 2

/-- Theorem stating the properties of the circular class -/
theorem circular_class_properties (c : CircularClass) 
  (h1 : c.boyOppositePositions = (10, 40))
  (h2 : c.everyOtherIsBoy = true) :
  c.totalStudents = 60 ∧ numberOfBoys c = 30 := by
  sorry

#check circular_class_properties

end NUMINAMATH_CALUDE_circular_class_properties_l671_67105


namespace NUMINAMATH_CALUDE_yoongi_has_smaller_number_l671_67176

theorem yoongi_has_smaller_number : 
  let jungkook_number := 6 + 3
  let yoongi_number := 4
  yoongi_number < jungkook_number := by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_smaller_number_l671_67176


namespace NUMINAMATH_CALUDE_polygon_sides_l671_67197

theorem polygon_sides (sum_interior_angles : ℝ) (h : sum_interior_angles = 1680) :
  ∃ (n : ℕ), n = 12 ∧ (n - 2) * 180 > sum_interior_angles ∧ (n - 2) * 180 ≤ sum_interior_angles + 180 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l671_67197


namespace NUMINAMATH_CALUDE_decimal_representation_of_fraction_l671_67187

theorem decimal_representation_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.36 ↔ n = 9 ∧ d = 25 :=
sorry

end NUMINAMATH_CALUDE_decimal_representation_of_fraction_l671_67187


namespace NUMINAMATH_CALUDE_river_current_speed_l671_67113

/-- Given a boat's travel times and distances, calculates the current's speed -/
theorem river_current_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 24) 
  (h2 : upstream_distance = 24) 
  (h3 : downstream_time = 4) 
  (h4 : upstream_time = 6) :
  ∃ (boat_speed current_speed : ℝ),
    boat_speed > 0 ∧ 
    (boat_speed + current_speed) * downstream_time = downstream_distance ∧
    (boat_speed - current_speed) * upstream_time = upstream_distance ∧
    current_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l671_67113


namespace NUMINAMATH_CALUDE_x_squared_when_y_is_4_l671_67178

-- Define the inverse variation relationship between x² and y³
def inverse_variation (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x^2 * y^3 = k

-- State the theorem
theorem x_squared_when_y_is_4
  (h1 : ∀ x y, inverse_variation x y)
  (h2 : inverse_variation 10 2) :
  ∃ x : ℝ, inverse_variation x 4 ∧ x^2 = 12.5 := by
sorry


end NUMINAMATH_CALUDE_x_squared_when_y_is_4_l671_67178


namespace NUMINAMATH_CALUDE_sequence_a_property_l671_67168

def sequence_a (n : ℕ) : ℚ :=
  1 / (n * (n + 1))

def S (n : ℕ) : ℚ :=
  n^2 * sequence_a n

theorem sequence_a_property :
  ∀ n : ℕ, n ≥ 1 →
    (sequence_a 1 = 1) ∧
    (S n = n^2 * sequence_a n) ∧
    (sequence_a n = 1 / (n * (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_property_l671_67168


namespace NUMINAMATH_CALUDE_square_area_is_400_l671_67184

/-- A square is cut into five rectangles of equal area, with one rectangle having a width of 5. -/
structure CutSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The width of the rectangle with known width -/
  known_width : ℝ
  /-- The area of each rectangle -/
  rectangle_area : ℝ
  /-- The known width is 5 -/
  known_width_is_5 : known_width = 5
  /-- The square is divided into 5 rectangles of equal area -/
  five_equal_rectangles : side * side = 5 * rectangle_area

/-- The area of the square is 400 -/
theorem square_area_is_400 (s : CutSquare) : s.side * s.side = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_400_l671_67184


namespace NUMINAMATH_CALUDE_quadratic_root_product_l671_67165

theorem quadratic_root_product (b : ℝ) : ∃ x₁ x₂ : ℝ, 
  (x₁ * x₂ = 8) ∧ (x₁^2 + b*x₁ + 8 = 0) ∧ (x₂^2 + b*x₂ + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l671_67165


namespace NUMINAMATH_CALUDE_inequality_proof_l671_67139

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a + b < 2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l671_67139


namespace NUMINAMATH_CALUDE_test_probabilities_l671_67107

/-- Given probabilities of answering questions correctly on a test, 
    calculate the probability of answering neither question correctly. -/
theorem test_probabilities (p_first p_second p_both : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.55)
  (h3 : p_both = 0.50) :
  1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_test_probabilities_l671_67107


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l671_67132

theorem diophantine_equation_solutions
  (a b c : ℤ) 
  (d : ℕ) 
  (h_d : d = Int.gcd a b) 
  (h_div : c % d = 0) 
  (x₀ y₀ : ℤ) 
  (h_particular : a * x₀ + b * y₀ = c) :
  ∀ (x y : ℤ), 
    (a * x + b * y = c) ↔ 
    (∃ (k : ℤ), x = x₀ + k * (b / d) ∧ y = y₀ - k * (a / d)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l671_67132


namespace NUMINAMATH_CALUDE_selling_price_calculation_l671_67173

/-- Given an article with a gain of $15 and a gain percentage of 20%,
    prove that the selling price is $90. -/
theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) :
  gain = 15 →
  gain_percentage = 20 →
  ∃ (cost_price selling_price : ℝ),
    gain = (gain_percentage / 100) * cost_price ∧
    selling_price = cost_price + gain ∧
    selling_price = 90 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l671_67173


namespace NUMINAMATH_CALUDE_inequality_system_solution_l671_67194

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, (x - b < 0 ∧ x + a > 0) ↔ (2 < x ∧ x < 3)) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l671_67194


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l671_67155

theorem arithmetic_calculations :
  ((-8) + 10 - (-2) = 12) ∧
  (42 * (-2/3) + (-3/4) / (-0.25) = -25) ∧
  ((-2.5) / (-5/8) * (-0.25) = -1) ∧
  ((1 + 3/4 - 7/8 - 7/12) / (-7/8) = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l671_67155


namespace NUMINAMATH_CALUDE_ABD_collinear_l671_67102

/-- Given vectors in 2D space -/
def a : ℝ × ℝ := sorry
def b : ℝ × ℝ := sorry

/-- Define vectors AB, BC, and CD -/
def AB : ℝ × ℝ := a + 5 • b
def BC : ℝ × ℝ := -2 • a + 8 • b
def CD : ℝ × ℝ := 3 • (a - b)

/-- Define points A, B, C, and D -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := A + AB
def C : ℝ × ℝ := B + BC
def D : ℝ × ℝ := C + CD

/-- Theorem: Points A, B, and D are collinear -/
theorem ABD_collinear : ∃ (t : ℝ), D = A + t • (B - A) := by sorry

end NUMINAMATH_CALUDE_ABD_collinear_l671_67102


namespace NUMINAMATH_CALUDE_pet_food_difference_l671_67114

theorem pet_food_difference (dog_food : ℕ) (cat_food : ℕ) 
  (h1 : dog_food = 600) (h2 : cat_food = 327) : 
  dog_food - cat_food = 273 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_difference_l671_67114


namespace NUMINAMATH_CALUDE_milk_sharing_l671_67162

theorem milk_sharing (don_milk : ℚ) (rachel_portion : ℚ) (rachel_milk : ℚ) : 
  don_milk = 3 / 7 → 
  rachel_portion = 1 / 2 → 
  rachel_milk = rachel_portion * don_milk → 
  rachel_milk = 3 / 14 := by
sorry

end NUMINAMATH_CALUDE_milk_sharing_l671_67162


namespace NUMINAMATH_CALUDE_permutations_of_four_distinct_elements_l671_67130

theorem permutations_of_four_distinct_elements : 
  Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_four_distinct_elements_l671_67130


namespace NUMINAMATH_CALUDE_polynomial_equality_l671_67146

/-- Given a polynomial Q(x) = Q(0) + Q(1)x + Q(3)x^2 where Q(-1) = 2, 
    prove that Q(x) = 0.6x^2 - 2x - 0.6 -/
theorem polynomial_equality (Q : ℝ → ℝ) (h1 : ∀ x, Q x = Q 0 + Q 1 * x + Q 3 * x^2)
    (h2 : Q (-1) = 2) : ∀ x, Q x = 0.6 * x^2 - 2 * x - 0.6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l671_67146


namespace NUMINAMATH_CALUDE_min_value_2m_plus_n_solution_set_f_gt_5_l671_67125

-- Define the function f
def f (x m n : ℝ) : ℝ := |x + m| + |2*x - n|

-- Theorem for part I
theorem min_value_2m_plus_n (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f x m n ≥ 1) → 2*m + n ≥ 2 :=
sorry

-- Theorem for part II
theorem solution_set_f_gt_5 :
  {x : ℝ | f x 2 3 > 5} = {x : ℝ | x < 0 ∨ x > 2} :=
sorry

end NUMINAMATH_CALUDE_min_value_2m_plus_n_solution_set_f_gt_5_l671_67125


namespace NUMINAMATH_CALUDE_beast_sports_meeting_l671_67120

theorem beast_sports_meeting (total : ℕ) (tigers lions leopards : ℕ) : 
  total = 220 →
  lions = 2 * tigers + 5 →
  leopards = 2 * lions - 5 →
  total = tigers + lions + leopards →
  leopards - tigers = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_beast_sports_meeting_l671_67120


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l671_67133

/-- 
If the cost price of 50 articles is equal to the selling price of 15 articles, 
then the gain percent is 233.33%.
-/
theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 15 * S) : 
  (S - C) / C * 100 = 233.33 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l671_67133


namespace NUMINAMATH_CALUDE_minutes_before_noon_l671_67199

/-- 
Given that 20 minutes ago it was 3 times as many minutes after 9 am, 
and there are 180 minutes between 9 am and 12 noon, 
prove that it is 130 minutes before 12 noon.
-/
theorem minutes_before_noon : 
  ∀ x : ℕ, 
  (x + 20 = 3 * (180 - x)) → 
  x = 130 := by
sorry

end NUMINAMATH_CALUDE_minutes_before_noon_l671_67199


namespace NUMINAMATH_CALUDE_y_value_proof_l671_67136

theorem y_value_proof (y : ℝ) (h : 8 / y^3 = y / 32) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l671_67136


namespace NUMINAMATH_CALUDE_divide_by_approximate_700_l671_67183

-- Define the approximation tolerance
def tolerance : ℝ := 0.001

-- Define the condition from the problem
def condition (x : ℝ) : Prop :=
  abs (49 / x - 700) < tolerance

-- State the theorem
theorem divide_by_approximate_700 :
  ∃ x : ℝ, condition x ∧ abs (x - 0.07) < tolerance :=
sorry

end NUMINAMATH_CALUDE_divide_by_approximate_700_l671_67183


namespace NUMINAMATH_CALUDE_car_dealership_ratio_l671_67109

/-- Given a car dealership with economy cars, luxury cars, and sport utility vehicles,
    where the ratio of economy to luxury cars is 3:2 and the ratio of economy cars
    to sport utility vehicles is 4:1, prove that the ratio of luxury cars to sport
    utility vehicles is 8:3. -/
theorem car_dealership_ratio (E L S : ℚ) 
    (h1 : E / L = 3 / 2)
    (h2 : E / S = 4 / 1) :
    L / S = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_ratio_l671_67109


namespace NUMINAMATH_CALUDE_no_solutions_for_system_l671_67119

theorem no_solutions_for_system : 
  ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 2) := by
sorry

end NUMINAMATH_CALUDE_no_solutions_for_system_l671_67119


namespace NUMINAMATH_CALUDE_paco_cookies_l671_67145

/-- The number of cookies Paco initially had -/
def initial_cookies : ℕ := sorry

/-- The number of cookies Paco gave to his friend -/
def cookies_given : ℕ := 9

/-- The number of cookies Paco ate -/
def cookies_eaten : ℕ := 18

/-- The difference between cookies eaten and given -/
def cookies_difference : ℕ := 9

theorem paco_cookies : 
  initial_cookies = cookies_given + cookies_eaten ∧
  cookies_eaten = cookies_given + cookies_difference ∧
  initial_cookies = 27 := by sorry

end NUMINAMATH_CALUDE_paco_cookies_l671_67145


namespace NUMINAMATH_CALUDE_max_rides_both_days_l671_67108

/-- Represents the prices of rides on a given day -/
structure RidePrices where
  ferrisWheel : ℕ
  rollerCoaster : ℕ
  bumperCars : ℕ
  carousel : ℕ
  logFlume : ℕ
  hauntedHouse : Option ℕ

/-- Calculates the maximum number of rides within a budget -/
def maxRides (prices : RidePrices) (budget : ℕ) : ℕ :=
  sorry

/-- The daily budget -/
def dailyBudget : ℕ := 10

/-- Ride prices for the first day -/
def firstDayPrices : RidePrices :=
  { ferrisWheel := 4
  , rollerCoaster := 5
  , bumperCars := 3
  , carousel := 2
  , logFlume := 6
  , hauntedHouse := none }

/-- Ride prices for the second day -/
def secondDayPrices : RidePrices :=
  { ferrisWheel := 4
  , rollerCoaster := 7
  , bumperCars := 3
  , carousel := 2
  , logFlume := 6
  , hauntedHouse := some 4 }

theorem max_rides_both_days :
  maxRides firstDayPrices dailyBudget = 3 ∧
  maxRides secondDayPrices dailyBudget = 3 :=
sorry

end NUMINAMATH_CALUDE_max_rides_both_days_l671_67108


namespace NUMINAMATH_CALUDE_joes_total_lift_weight_l671_67154

/-- The total weight of Joe's two lifts is 600 pounds -/
theorem joes_total_lift_weight :
  ∀ (first_lift second_lift : ℕ),
  first_lift = 300 →
  2 * first_lift = second_lift + 300 →
  first_lift + second_lift = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_joes_total_lift_weight_l671_67154


namespace NUMINAMATH_CALUDE_inequality_proof_l671_67127

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l671_67127


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l671_67141

theorem factorization_of_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l671_67141


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l671_67151

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - 4*m + 3)
  (z.re = 0 ∧ z.im ≠ 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l671_67151


namespace NUMINAMATH_CALUDE_graph_transform_properties_l671_67185

/-- A graph in a 2D plane -/
structure Graph where
  -- We don't need to define the internal structure of the graph
  -- as we're only concerned with its properties under transformations

/-- Properties of a graph that may or may not change under transformations -/
structure GraphProperties where
  shape : Bool  -- True if shape is preserved
  size : Bool   -- True if size is preserved
  direction : Bool  -- True if direction is preserved

/-- Rotation of a graph -/
def rotate (g : Graph) : Graph :=
  sorry

/-- Translation of a graph -/
def translate (g : Graph) : Graph :=
  sorry

/-- Properties preserved under rotation and translation -/
def properties_after_transform (g : Graph) : GraphProperties :=
  sorry

theorem graph_transform_properties :
  ∀ g : Graph,
    let props := properties_after_transform g
    props.shape = true ∧ props.size = true ∧ props.direction = false :=
by sorry

end NUMINAMATH_CALUDE_graph_transform_properties_l671_67185


namespace NUMINAMATH_CALUDE_sunTzu_nests_count_l671_67157

/-- Geometric sequence with first term a and common ratio r -/
def geometricSeq (a : ℕ) (r : ℕ) : ℕ → ℕ := fun n => a * r ^ (n - 1)

/-- The number of nests in Sun Tzu's Arithmetic problem -/
def sunTzuNests : ℕ := geometricSeq 9 9 4

theorem sunTzu_nests_count : sunTzuNests = 6561 := by
  sorry

end NUMINAMATH_CALUDE_sunTzu_nests_count_l671_67157


namespace NUMINAMATH_CALUDE_alberts_to_bettys_age_ratio_l671_67104

/-- Proves that the ratio of Albert's age to Betty's age is 4:1 given the specified conditions -/
theorem alberts_to_bettys_age_ratio :
  ∀ (albert_age mary_age betty_age : ℕ),
    albert_age = 2 * mary_age →
    mary_age = albert_age - 14 →
    betty_age = 7 →
    (albert_age : ℚ) / betty_age = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_alberts_to_bettys_age_ratio_l671_67104


namespace NUMINAMATH_CALUDE_wedding_attendance_l671_67123

theorem wedding_attendance (actual_attendance : ℕ) (show_up_rate : ℚ) : 
  actual_attendance = 209 → show_up_rate = 95/100 → 
  ∃ expected_attendance : ℕ, expected_attendance = 220 ∧ 
  (↑actual_attendance : ℚ) = show_up_rate * expected_attendance := by
sorry

end NUMINAMATH_CALUDE_wedding_attendance_l671_67123


namespace NUMINAMATH_CALUDE_largest_common_divisor_462_231_l671_67160

theorem largest_common_divisor_462_231 : Nat.gcd 462 231 = 231 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_462_231_l671_67160


namespace NUMINAMATH_CALUDE_bandages_left_in_box_l671_67111

/-- The number of bandages in a box before use -/
def initial_bandages : ℕ := 24 - 8

/-- The number of bandages used on the left knee -/
def left_knee_bandages : ℕ := 2

/-- The number of bandages used on the right knee -/
def right_knee_bandages : ℕ := 3

/-- The total number of bandages used -/
def total_used_bandages : ℕ := left_knee_bandages + right_knee_bandages

theorem bandages_left_in_box : initial_bandages - total_used_bandages = 11 := by
  sorry

end NUMINAMATH_CALUDE_bandages_left_in_box_l671_67111


namespace NUMINAMATH_CALUDE_polynomial_value_at_one_l671_67121

theorem polynomial_value_at_one (a b c : ℝ) : 
  (-a - b - c + 1 = 6) → (a + b + c + 1 = -4) := by sorry

end NUMINAMATH_CALUDE_polynomial_value_at_one_l671_67121


namespace NUMINAMATH_CALUDE_geometric_series_sum_l671_67186

theorem geometric_series_sum : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  let series_sum : ℚ := (a * (1 - r^n)) / (1 - r)
  series_sum = 341/1024 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l671_67186


namespace NUMINAMATH_CALUDE_circular_arcs_in_regular_ngon_l671_67129

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A point inside a regular n-gon -/
def PointInside (E : RegularNGon n) (P : ℝ × ℝ) : Prop := sorry

/-- A circular arc inside a regular n-gon -/
def CircularArcInside (E : RegularNGon n) (arc : ℝ × ℝ → ℝ × ℝ → Prop) : Prop := sorry

/-- The angle between two circular arcs at their intersection point -/
def AngleBetweenArcs (arc1 arc2 : ℝ × ℝ → ℝ × ℝ → Prop) (P : ℝ × ℝ) : ℝ := sorry

theorem circular_arcs_in_regular_ngon (n : ℕ) (E : RegularNGon n) (P₁ P₂ : ℝ × ℝ) 
  (h₁ : PointInside E P₁) (h₂ : PointInside E P₂) :
  ∃ (arc1 arc2 : ℝ × ℝ → ℝ × ℝ → Prop),
    CircularArcInside E arc1 ∧ 
    CircularArcInside E arc2 ∧
    arc1 P₁ P₂ ∧ 
    arc2 P₁ P₂ ∧
    AngleBetweenArcs arc1 arc2 P₁ ≥ (1 - 2 / n) * π ∧
    AngleBetweenArcs arc1 arc2 P₂ ≥ (1 - 2 / n) * π :=
sorry

end NUMINAMATH_CALUDE_circular_arcs_in_regular_ngon_l671_67129


namespace NUMINAMATH_CALUDE_A_intersect_B_empty_l671_67181

/-- The set A defined by the equation (y-3)/(x-2) = a+1 -/
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = a + 1}

/-- The set B defined by the equation (a^2-1)x + (a-1)y = 15 -/
def B (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a^2 - 1) * p.1 + (a - 1) * p.2 = 15}

/-- The theorem stating that A ∩ B is empty if and only if a is in the set {-1, -4, 1, 5/2} -/
theorem A_intersect_B_empty (a : ℝ) :
  A a ∩ B a = ∅ ↔ a ∈ ({-1, -4, 1, (5:ℝ)/2} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_empty_l671_67181


namespace NUMINAMATH_CALUDE_two_blue_gumballs_probability_l671_67116

/-- The probability of drawing a pink gumball from the jar -/
def prob_pink : ℝ := 0.5714285714285714

/-- The probability of drawing a blue gumball from the jar -/
def prob_blue : ℝ := 1 - prob_pink

/-- The probability of drawing two blue gumballs in a row -/
def prob_two_blue : ℝ := prob_blue * prob_blue

theorem two_blue_gumballs_probability :
  prob_two_blue = 0.1836734693877551 := by sorry

end NUMINAMATH_CALUDE_two_blue_gumballs_probability_l671_67116


namespace NUMINAMATH_CALUDE_gala_trees_l671_67167

/-- Represents the orchard with Fuji and Gala apple trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- Conditions of the orchard -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji + o.cross_pollinated = 170 ∧
  o.pure_fuji = (3 * o.total) / 4 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

theorem gala_trees (o : Orchard) (h : orchard_conditions o) : o.pure_gala = 50 := by
  sorry

#check gala_trees

end NUMINAMATH_CALUDE_gala_trees_l671_67167


namespace NUMINAMATH_CALUDE_problem_solution_l671_67159

/-- An arithmetic sequence with positive terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem problem_solution (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
    (h_geom : geometric_sequence b)
    (h_equal : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l671_67159


namespace NUMINAMATH_CALUDE_rectangle_area_l671_67192

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 206) :
  L * B = 2520 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l671_67192


namespace NUMINAMATH_CALUDE_square_remainder_mod_16_l671_67148

theorem square_remainder_mod_16 (n : ℤ) : ∃ k : ℤ, 0 ≤ k ∧ k < 4 ∧ (n^2) % 16 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_mod_16_l671_67148


namespace NUMINAMATH_CALUDE_vector_AB_coordinates_and_magnitude_l671_67150

def OA : ℝ × ℝ := (1, 2)
def OB : ℝ × ℝ := (3, 1)

def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_AB_coordinates_and_magnitude :
  AB = (2, -1) ∧ Real.sqrt ((AB.1)^2 + (AB.2)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_AB_coordinates_and_magnitude_l671_67150


namespace NUMINAMATH_CALUDE_prime_gap_2015_l671_67161

theorem prime_gap_2015 : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p < q ∧ q - p > 2015 ∧ 
  ∀ k : ℕ, p < k ∧ k < q → ¬(Prime k) :=
sorry

end NUMINAMATH_CALUDE_prime_gap_2015_l671_67161


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_two_plus_half_inverse_l671_67147

theorem absolute_value_sqrt_two_plus_half_inverse :
  |1 - Real.sqrt 2| + (1/2)⁻¹ = Real.sqrt 2 + 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_two_plus_half_inverse_l671_67147


namespace NUMINAMATH_CALUDE_even_function_interval_sum_zero_l671_67169

/-- A function f is even on an interval [a, b] if for all x in [a, b], f(x) = f(-x) -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, f x = f (-x)

/-- If f is an even function on the interval [a, b], then a + b = 0 -/
theorem even_function_interval_sum_zero (f : ℝ → ℝ) (a b : ℝ) 
  (h : IsEvenOn f a b) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_interval_sum_zero_l671_67169


namespace NUMINAMATH_CALUDE_group_size_l671_67156

/-- The number of people in a group, given weight changes. -/
theorem group_size (weight_increase_per_person : ℝ) (new_person_weight : ℝ) (replaced_person_weight : ℝ) :
  weight_increase_per_person * 10 = new_person_weight - replaced_person_weight →
  10 = (new_person_weight - replaced_person_weight) / weight_increase_per_person :=
by
  sorry

#check group_size 7.2 137 65

end NUMINAMATH_CALUDE_group_size_l671_67156


namespace NUMINAMATH_CALUDE_cost_per_bag_l671_67149

def num_friends : ℕ := 3
def num_bags : ℕ := 5
def payment_per_friend : ℚ := 5

theorem cost_per_bag : 
  (num_friends * payment_per_friend) / num_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_bag_l671_67149


namespace NUMINAMATH_CALUDE_function_zero_nonpositive_l671_67196

/-- A function satisfying the given inequality property -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) ≤ y * f x + f (f x)

/-- The main theorem to prove -/
theorem function_zero_nonpositive (f : ℝ → ℝ) (h : SatisfiesInequality f) :
    ∀ x : ℝ, x ≤ 0 → f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_nonpositive_l671_67196


namespace NUMINAMATH_CALUDE_line_AB_passes_through_fixed_point_l671_67188

-- Define the hyperbola D
def hyperbolaD (x y : ℝ) : Prop := y^2/2 - x^2 = 1/3

-- Define the parabola C
def parabolaC (x y : ℝ) : Prop := x^2 = 4*y

-- Define the point P on parabola C
def P : ℝ × ℝ := (2, 1)

-- Define a point on parabola C
def pointOnParabolaC (x y : ℝ) : Prop := parabolaC x y

-- Define the perpendicular condition for PA and PB
def perpendicularCondition (x1 y1 x2 y2 : ℝ) : Prop :=
  ((y1 - 1) / (x1 - 2)) * ((y2 - 1) / (x2 - 2)) = -1

-- The main theorem
theorem line_AB_passes_through_fixed_point :
  ∀ (x1 y1 x2 y2 : ℝ),
  pointOnParabolaC x1 y1 →
  pointOnParabolaC x2 y2 →
  perpendicularCondition x1 y1 x2 y2 →
  ∃ (t : ℝ), t ∈ (Set.Icc 0 1) ∧ 
  (t * x1 + (1 - t) * x2 = -2) ∧
  (t * y1 + (1 - t) * y2 = 5) :=
sorry

end NUMINAMATH_CALUDE_line_AB_passes_through_fixed_point_l671_67188


namespace NUMINAMATH_CALUDE_min_value_theorem_l671_67138

theorem min_value_theorem (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_cond : x + y + z + w = 2)
  (prod_cond : x * y * z * w = 1/16) :
  (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → 
    a + b + c + d = 2 → a * b * c * d = 1/16 → 
    (x + y + z) / (x * y * z * w) ≤ (a + b + c) / (a * b * c * d)) →
  (x + y + z) / (x * y * z * w) = 24 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l671_67138
