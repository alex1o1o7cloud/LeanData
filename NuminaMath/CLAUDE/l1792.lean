import Mathlib

namespace NUMINAMATH_CALUDE_circle_plus_solution_l1792_179265

def circle_plus (a b : ℝ) : ℝ := a * b - 2 * b + 3 * a

theorem circle_plus_solution :
  ∃ x : ℝ, circle_plus 7 x = 61 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_solution_l1792_179265


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1792_179237

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties :
  a ≠ 0 →
  (∀ x : ℝ, f a b c x ≠ x) →
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1792_179237


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_k_range_l1792_179227

/-- A function f is monotonic on an interval [a, +∞) if for all x, y in [a, +∞) with x ≤ y, we have f(x) ≤ f(y) or f(x) ≥ f(y) -/
def IsMonotonicOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x ≥ a → y ≥ a → x ≤ y → (f x ≤ f y ∨ f x ≥ f y)

/-- The main theorem stating that if f(x) = x^2 - 2kx - 2 is monotonic on [5, +∞), then k ∈ (-∞, 5] -/
theorem monotonic_quadratic_function_k_range :
  ∀ k : ℝ, IsMonotonicOn (fun x => x^2 - 2*k*x - 2) 5 → k ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_k_range_l1792_179227


namespace NUMINAMATH_CALUDE_no_xyz_solution_l1792_179206

theorem no_xyz_solution : ¬∃ (x y z : ℕ), 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ 
  100 * x + 10 * y + z = x * (10 * y + z) := by
sorry

end NUMINAMATH_CALUDE_no_xyz_solution_l1792_179206


namespace NUMINAMATH_CALUDE_train_speed_and_length_l1792_179202

-- Define the bridge length
def bridge_length : ℝ := 1000

-- Define the time to completely cross the bridge
def cross_time : ℝ := 60

-- Define the time spent on the bridge
def bridge_time : ℝ := 40

-- Define the train's speed
def train_speed : ℝ := 20

-- Define the train's length
def train_length : ℝ := 200

theorem train_speed_and_length :
  bridge_length = 1000 ∧ 
  cross_time = 60 ∧ 
  bridge_time = 40 →
  train_speed * cross_time = bridge_length + train_length ∧
  train_speed * bridge_time = bridge_length ∧
  train_speed = 20 ∧
  train_length = 200 := by sorry

end NUMINAMATH_CALUDE_train_speed_and_length_l1792_179202


namespace NUMINAMATH_CALUDE_complex_magnitude_quadratic_l1792_179272

theorem complex_magnitude_quadratic (z : ℂ) : z^2 - 6*z + 25 = 0 → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_quadratic_l1792_179272


namespace NUMINAMATH_CALUDE_point_A_on_circle_point_B_on_circle_point_C_on_circle_circle_equation_unique_l1792_179279

/-- A circle passes through points A(2, 0), B(4, 0), and C(0, 2) -/
def circle_through_points (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 3)^2 = 10

/-- Point A lies on the circle -/
theorem point_A_on_circle : circle_through_points 2 0 := by sorry

/-- Point B lies on the circle -/
theorem point_B_on_circle : circle_through_points 4 0 := by sorry

/-- Point C lies on the circle -/
theorem point_C_on_circle : circle_through_points 0 2 := by sorry

/-- The equation (x - 3)² + (y - 3)² = 10 represents the unique circle 
    passing through points A(2, 0), B(4, 0), and C(0, 2) -/
theorem circle_equation_unique : 
  ∀ x y : ℝ, circle_through_points x y ↔ (x - 3)^2 + (y - 3)^2 = 10 := by sorry

end NUMINAMATH_CALUDE_point_A_on_circle_point_B_on_circle_point_C_on_circle_circle_equation_unique_l1792_179279


namespace NUMINAMATH_CALUDE_pineapple_juice_theorem_l1792_179204

/-- Represents the juice bar problem -/
structure JuiceBarProblem where
  total_spent : ℕ
  mango_price : ℕ
  pineapple_price : ℕ
  total_people : ℕ

/-- Calculates the amount spent on pineapple juice -/
def pineapple_juice_spent (problem : JuiceBarProblem) : ℕ :=
  let mango_people := problem.total_people - (problem.total_spent - problem.mango_price * problem.total_people) / (problem.pineapple_price - problem.mango_price)
  let pineapple_people := problem.total_people - mango_people
  pineapple_people * problem.pineapple_price

/-- Theorem stating that the amount spent on pineapple juice is $54 -/
theorem pineapple_juice_theorem (problem : JuiceBarProblem) 
  (h1 : problem.total_spent = 94)
  (h2 : problem.mango_price = 5)
  (h3 : problem.pineapple_price = 6)
  (h4 : problem.total_people = 17) :
  pineapple_juice_spent problem = 54 := by
  sorry

#eval pineapple_juice_spent { total_spent := 94, mango_price := 5, pineapple_price := 6, total_people := 17 }

end NUMINAMATH_CALUDE_pineapple_juice_theorem_l1792_179204


namespace NUMINAMATH_CALUDE_phoebes_servings_is_one_l1792_179222

/-- The number of servings per jar of peanut butter -/
def servings_per_jar : ℕ := 15

/-- The number of jars needed -/
def jars_needed : ℕ := 4

/-- The number of days the peanut butter should last -/
def days_to_last : ℕ := 30

/-- Phoebe's serving amount equals her dog's serving amount -/
axiom phoebe_dog_equal_servings : True

/-- The number of servings Phoebe eats each night -/
def phoebes_nightly_servings : ℚ :=
  (servings_per_jar * jars_needed : ℚ) / (2 * days_to_last)

theorem phoebes_servings_is_one :
  phoebes_nightly_servings = 1 := by sorry

end NUMINAMATH_CALUDE_phoebes_servings_is_one_l1792_179222


namespace NUMINAMATH_CALUDE_length_breadth_difference_is_24_l1792_179228

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ

/-- The difference between length and breadth of a rectangular plot. -/
def lengthBreadthDifference (plot : RectangularPlot) : ℝ :=
  plot.length - plot.breadth

/-- The perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

/-- Theorem stating the difference between length and breadth for a specific plot. -/
theorem length_breadth_difference_is_24 (plot : RectangularPlot) 
    (h1 : plot.length = 62)
    (h2 : plot.fencingCostPerMeter = 26.5)
    (h3 : plot.totalFencingCost = 5300)
    (h4 : plot.totalFencingCost = plot.fencingCostPerMeter * perimeter plot) :
  lengthBreadthDifference plot = 24 := by
  sorry

#eval lengthBreadthDifference { length := 62, breadth := 38, fencingCostPerMeter := 26.5, totalFencingCost := 5300 }

end NUMINAMATH_CALUDE_length_breadth_difference_is_24_l1792_179228


namespace NUMINAMATH_CALUDE_complex_quadrant_l1792_179268

theorem complex_quadrant (z : ℂ) (h : (z + Complex.I) * Complex.I = 1 + z) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1792_179268


namespace NUMINAMATH_CALUDE_dogsled_race_speed_difference_l1792_179295

theorem dogsled_race_speed_difference 
  (course_length : ℝ) 
  (team_w_speed : ℝ) 
  (time_difference : ℝ) :
  course_length = 300 →
  team_w_speed = 20 →
  time_difference = 3 →
  let team_w_time := course_length / team_w_speed
  let team_a_time := team_w_time - time_difference
  let team_a_speed := course_length / team_a_time
  team_a_speed - team_w_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_speed_difference_l1792_179295


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l1792_179238

theorem sine_cosine_relation (x : ℝ) (h : Real.cos (5 * Real.pi / 6 - x) = 1 / 3) :
  Real.sin (x - Real.pi / 3) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l1792_179238


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1792_179289

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 4/y = 1 ∧ x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1792_179289


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1792_179229

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 5*x + 6 > 0 ↔ x < 2 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1792_179229


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_10000_l1792_179200

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sequence_sum_of_digits (n : Nat) : Nat :=
  (List.range n).map sum_of_digits |> List.sum

theorem sum_of_digits_1_to_10000 :
  sequence_sum_of_digits 10000 = 180001 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_10000_l1792_179200


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1792_179251

-- System 1
theorem system_one_solution (x y : ℝ) : 
  (4 * x - 2 * y = 14) ∧ (3 * x + 2 * y = 7) → x = 3 ∧ y = -1 :=
by sorry

-- System 2
theorem system_two_solution (x y : ℝ) :
  (y = x + 1) ∧ (2 * x + y = 10) → x = 3 ∧ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1792_179251


namespace NUMINAMATH_CALUDE_problem_statement_l1792_179239

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

theorem problem_statement :
  (3 ∈ A) ∧ (∀ k : ℤ, 4*k - 2 ∉ A) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1792_179239


namespace NUMINAMATH_CALUDE_pentagon_regular_if_equal_altitudes_and_medians_l1792_179213

/-- A pentagon is a polygon with five vertices and five edges. -/
structure Pentagon where
  vertices : Fin 5 → ℝ × ℝ

/-- An altitude of a pentagon is the perpendicular drop from a vertex to the opposite side. -/
def altitude (p : Pentagon) (i : Fin 5) : ℝ := sorry

/-- A median of a pentagon is the line joining a vertex to the midpoint of the opposite side. -/
def median (p : Pentagon) (i : Fin 5) : ℝ := sorry

/-- A pentagon is regular if all its sides are equal and all its interior angles are equal. -/
def is_regular (p : Pentagon) : Prop := sorry

/-- Theorem: If all altitudes and all medians of a pentagon have the same length, then the pentagon is regular. -/
theorem pentagon_regular_if_equal_altitudes_and_medians (p : Pentagon) 
  (h1 : ∀ i j : Fin 5, altitude p i = altitude p j) 
  (h2 : ∀ i j : Fin 5, median p i = median p j) : 
  is_regular p := by sorry

end NUMINAMATH_CALUDE_pentagon_regular_if_equal_altitudes_and_medians_l1792_179213


namespace NUMINAMATH_CALUDE_felicity_gas_usage_l1792_179207

theorem felicity_gas_usage (adhira : ℝ) 
  (h1 : 4 * adhira - 5 + adhira = 30) : 
  4 * adhira - 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_felicity_gas_usage_l1792_179207


namespace NUMINAMATH_CALUDE_circle_radius_when_area_is_250_percent_of_circumference_l1792_179230

theorem circle_radius_when_area_is_250_percent_of_circumference (r : ℝ) : 
  r > 0 → π * r^2 = 2.5 * (2 * π * r) → r = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_when_area_is_250_percent_of_circumference_l1792_179230


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l1792_179219

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_b_value :
  ∀ b : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, b⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → b = -9 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l1792_179219


namespace NUMINAMATH_CALUDE_concentric_circles_properties_l1792_179226

/-- Two concentric circles with a width of 15 feet between them -/
structure ConcentricCircles where
  inner_diameter : ℝ
  width : ℝ
  width_is_15 : width = 15

theorem concentric_circles_properties (c : ConcentricCircles) :
  let outer_diameter := c.inner_diameter + 2 * c.width
  (π * outer_diameter - π * c.inner_diameter = 30 * π) ∧
  (π * (15 * c.inner_diameter + 225) = 
   π * ((outer_diameter / 2)^2 - (c.inner_diameter / 2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_properties_l1792_179226


namespace NUMINAMATH_CALUDE_prob_not_both_white_l1792_179254

theorem prob_not_both_white (prob_white_A prob_white_B : ℚ) 
  (h1 : prob_white_A = 1/3)
  (h2 : prob_white_B = 1/2) :
  1 - prob_white_A * prob_white_B = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_both_white_l1792_179254


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1792_179250

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 - 9 * p^2 + 27 * p - 6 = 0) →
  (3 * q^3 - 9 * q^2 + 27 * q - 6 = 0) →
  (3 * r^3 - 9 * r^2 + 27 * r - 6 = 0) →
  (p + q + r = 3) →
  (p * q + q * r + r * p = 9) →
  (p * q * r = 2) →
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 585 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1792_179250


namespace NUMINAMATH_CALUDE_min_large_buses_correct_l1792_179260

/-- The minimum number of large buses required to transport students --/
def min_large_buses (total_students : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) (min_small_buses : ℕ) : ℕ :=
  let remaining_students := total_students - min_small_buses * small_capacity
  (remaining_students + large_capacity - 1) / large_capacity

theorem min_large_buses_correct (total_students : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) (min_small_buses : ℕ)
  (h1 : total_students = 523)
  (h2 : large_capacity = 45)
  (h3 : small_capacity = 30)
  (h4 : min_small_buses = 5) :
  min_large_buses total_students large_capacity small_capacity min_small_buses = 9 := by
  sorry

#eval min_large_buses 523 45 30 5

end NUMINAMATH_CALUDE_min_large_buses_correct_l1792_179260


namespace NUMINAMATH_CALUDE_toy_purchase_with_discount_l1792_179280

theorem toy_purchase_with_discount (num_toys : ℕ) (cost_per_toy : ℝ) (discount_percent : ℝ) :
  num_toys = 5 →
  cost_per_toy = 3 →
  discount_percent = 20 →
  (num_toys * cost_per_toy) * (1 - discount_percent / 100) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_with_discount_l1792_179280


namespace NUMINAMATH_CALUDE_xy_value_l1792_179234

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1792_179234


namespace NUMINAMATH_CALUDE_max_stores_visited_l1792_179286

theorem max_stores_visited (
  total_stores : ℕ)
  (total_shoppers : ℕ)
  (double_visitors : ℕ)
  (total_visits : ℕ)
  (h1 : total_stores = 7)
  (h2 : total_shoppers = 11)
  (h3 : double_visitors = 7)
  (h4 : total_visits = 21)
  (h5 : double_visitors * 2 + (total_shoppers - double_visitors) ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧
    ∀ (individual_visits : ℕ), individual_visits ≤ max_visits :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l1792_179286


namespace NUMINAMATH_CALUDE_product_remainder_l1792_179257

theorem product_remainder (a b c d : ℕ) (h1 : a = 1729) (h2 : b = 1865) (h3 : c = 1912) (h4 : d = 2023) :
  (a * b * c * d) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1792_179257


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l1792_179235

theorem consecutive_page_numbers (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 15300 → n + (n + 1) = 247 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l1792_179235


namespace NUMINAMATH_CALUDE_range_of_m_l1792_179262

theorem range_of_m (m : ℝ) : 
  ¬((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  m ≤ -2 ∨ m > -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1792_179262


namespace NUMINAMATH_CALUDE_second_price_increase_l1792_179276

theorem second_price_increase (P : ℝ) (x : ℝ) (h : P > 0) :
  (1.15 * P) * (1 + x / 100) = 1.4375 * P → x = 25 := by
sorry

end NUMINAMATH_CALUDE_second_price_increase_l1792_179276


namespace NUMINAMATH_CALUDE_y_value_l1792_179233

theorem y_value (x : ℝ) : 
  Real.sqrt ((2008 * x + 2009) / (2010 * x - 2011)) + 
  Real.sqrt ((2008 * x + 2009) / (2011 - 2010 * x)) + 2010 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1792_179233


namespace NUMINAMATH_CALUDE_value_of_2a_plus_3b_l1792_179282

theorem value_of_2a_plus_3b (a b : ℚ) 
  (eq1 : 3 * a + 6 * b = 48) 
  (eq2 : 8 * a + 4 * b = 84) : 
  2 * a + 3 * b = 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_2a_plus_3b_l1792_179282


namespace NUMINAMATH_CALUDE_balls_after_2010_steps_l1792_179288

/-- Converts a natural number to its base-6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  if n < 6 then [n]
  else (n % 6) :: toBase6 (n / 6)

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.sum

theorem balls_after_2010_steps :
  sumDigits (toBase6 2010) = 10 := by
  sorry

end NUMINAMATH_CALUDE_balls_after_2010_steps_l1792_179288


namespace NUMINAMATH_CALUDE_interview_probability_l1792_179293

/-- The number of students enrolled in at least one language class -/
def total_students : ℕ := 30

/-- The number of students enrolled in the German class -/
def german_students : ℕ := 20

/-- The number of students enrolled in the Italian class -/
def italian_students : ℕ := 22

/-- The probability of selecting two students such that at least one is enrolled in German
    and at least one is enrolled in Italian -/
def prob_both_classes : ℚ := 362 / 435

theorem interview_probability :
  prob_both_classes = 1 - (Nat.choose (german_students + italian_students - total_students) 2 +
                           Nat.choose (german_students - (german_students + italian_students - total_students)) 2 +
                           Nat.choose (italian_students - (german_students + italian_students - total_students)) 2) /
                          Nat.choose total_students 2 :=
by sorry

end NUMINAMATH_CALUDE_interview_probability_l1792_179293


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1792_179255

/-- The quadratic function f(x) = x^2 + 1774x + 235 satisfies f(f(x) + x) / f(x) = x^2 + 1776x + 2010 for all x. -/
theorem quadratic_function_property : ∀ x : ℝ,
  let f : ℝ → ℝ := λ x ↦ x^2 + 1774*x + 235
  (f (f x + x)) / (f x) = x^2 + 1776*x + 2010 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1792_179255


namespace NUMINAMATH_CALUDE_box_surface_area_l1792_179215

theorem box_surface_area (side_area1 side_area2 volume : ℝ) 
  (h1 : side_area1 = 120)
  (h2 : side_area2 = 72)
  (h3 : volume = 720) :
  ∃ (length width height : ℝ),
    length * width = side_area1 ∧
    length * height = side_area2 ∧
    length * width * height = volume ∧
    length * width = 120 :=
by sorry

end NUMINAMATH_CALUDE_box_surface_area_l1792_179215


namespace NUMINAMATH_CALUDE_k_range_l1792_179271

/-- The function y = |log₂ x| is meaningful and not monotonic in the interval (k-1, k+1) -/
def is_meaningful_and_not_monotonic (k : ℝ) : Prop :=
  (k - 1 > 0) ∧ (1 ∈ Set.Ioo (k - 1) (k + 1))

/-- The theorem stating the range of k -/
theorem k_range :
  ∀ k : ℝ, is_meaningful_and_not_monotonic k ↔ k ∈ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l1792_179271


namespace NUMINAMATH_CALUDE_division_ways_count_l1792_179292

def number_of_people : ℕ := 6
def number_of_cars : ℕ := 2
def max_capacity_per_car : ℕ := 4

theorem division_ways_count :
  (Finset.sum (Finset.range (min number_of_people (max_capacity_per_car + 1)))
    (λ i => (number_of_people.choose i) * ((number_of_people - i).choose (number_of_people - i)))) = 60 := by
  sorry

end NUMINAMATH_CALUDE_division_ways_count_l1792_179292


namespace NUMINAMATH_CALUDE_betty_needs_five_more_l1792_179263

-- Define the cost of the wallet
def wallet_cost : ℕ := 100

-- Define Betty's initial savings
def betty_initial_savings : ℕ := wallet_cost / 2

-- Define the amount Betty's parents give her
def parents_contribution : ℕ := 15

-- Define the amount Betty's grandparents give her
def grandparents_contribution : ℕ := 2 * parents_contribution

-- Define Betty's total savings after contributions
def betty_total_savings : ℕ := betty_initial_savings + parents_contribution + grandparents_contribution

-- Theorem: Betty needs $5 more to buy the wallet
theorem betty_needs_five_more : wallet_cost - betty_total_savings = 5 := by
  sorry

end NUMINAMATH_CALUDE_betty_needs_five_more_l1792_179263


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l1792_179247

theorem base_10_to_base_7 : 
  ∃ (a b c d : ℕ), 
    784 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l1792_179247


namespace NUMINAMATH_CALUDE_factor_condition_l1792_179266

theorem factor_condition (x t : ℝ) : 
  (∃ k : ℝ, 6 * x^2 + 13 * x - 5 = (x - t) * k) ↔ (t = -5/2 ∨ t = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_factor_condition_l1792_179266


namespace NUMINAMATH_CALUDE_jia_steps_to_meet_yi_l1792_179278

theorem jia_steps_to_meet_yi (distance : ℝ) (speed_ratio : ℝ) (step_length : ℝ) :
  distance = 10560 ∧ speed_ratio = 5 ∧ step_length = 2.5 →
  (distance / (1 + speed_ratio)) / step_length = 704 := by
  sorry

end NUMINAMATH_CALUDE_jia_steps_to_meet_yi_l1792_179278


namespace NUMINAMATH_CALUDE_notebook_cost_l1792_179267

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buyers : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
  total_students = 35 ∧
  total_cost = 2013 ∧
  buyers > total_students / 2 ∧
  notebooks_per_student % 2 = 0 ∧
  notebooks_per_student > 2 ∧
  cost_per_notebook > notebooks_per_student ∧
  buyers * notebooks_per_student * cost_per_notebook = total_cost ∧
  cost_per_notebook = 61 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l1792_179267


namespace NUMINAMATH_CALUDE_profit_analysis_l1792_179298

-- Define the profit function
def profit (x : ℝ) : ℝ := x * (10 * x + 90)

-- Define the original monthly profit
def original_profit : ℝ := 1.2

theorem profit_analysis :
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 12 ∧ 10 * x^2 + 90 * x = 700 ∧ x = 5) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 12 ∧ 10 * x^2 + 90 * x = 120 * x ∧ x = 3) ∧
  (profit 12 + 12 * (profit 12 / 12) = 6360) :=
sorry

end NUMINAMATH_CALUDE_profit_analysis_l1792_179298


namespace NUMINAMATH_CALUDE_kevins_record_is_72_l1792_179259

/-- Calculates the number of wings in Kevin's hot wing eating record --/
def kevins_record (duration : ℕ) (alans_rate : ℕ) (additional_wings_needed : ℕ) : ℕ :=
  duration * (alans_rate + additional_wings_needed)

theorem kevins_record_is_72 :
  kevins_record 8 5 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_kevins_record_is_72_l1792_179259


namespace NUMINAMATH_CALUDE_cyclist_energized_time_l1792_179294

/-- Given a cyclist who rides at different speeds when energized and exhausted,
    prove the time spent energized for a specific total distance and time. -/
theorem cyclist_energized_time
  (speed_energized : ℝ)
  (speed_exhausted : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (h_speed_energized : speed_energized = 22)
  (h_speed_exhausted : speed_exhausted = 15)
  (h_total_distance : total_distance = 154)
  (h_total_time : total_time = 9)
  : ∃ (time_energized : ℝ),
    time_energized * speed_energized +
    (total_time - time_energized) * speed_exhausted = total_distance ∧
    time_energized = 19 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_energized_time_l1792_179294


namespace NUMINAMATH_CALUDE_adam_tickets_bought_l1792_179277

def tickets_bought (tickets_left : ℕ) (ticket_cost : ℕ) (amount_spent : ℕ) : ℕ :=
  tickets_left + amount_spent / ticket_cost

theorem adam_tickets_bought :
  tickets_bought 4 9 81 = 13 := by
  sorry

end NUMINAMATH_CALUDE_adam_tickets_bought_l1792_179277


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1792_179225

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1 / 3 : ℂ) + (2 / 5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1 / 3 : ℂ) - (2 / 5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1792_179225


namespace NUMINAMATH_CALUDE_sqrt_six_equality_l1792_179224

theorem sqrt_six_equality (r : ℝ) (h : r = Real.sqrt 2 + Real.sqrt 3) :
  Real.sqrt 6 = (r^2 - 5) / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_six_equality_l1792_179224


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1792_179208

theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) :
  diagonal = 16 →
  area = diagonal^2 / 2 →
  area = 128 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1792_179208


namespace NUMINAMATH_CALUDE_monster_consumption_l1792_179209

theorem monster_consumption (a : ℕ → ℕ) (h1 : a 0 = 121) (h2 : ∀ n, a (n + 1) = 2 * a n) : 
  a 0 + a 1 + a 2 = 847 := by
sorry

end NUMINAMATH_CALUDE_monster_consumption_l1792_179209


namespace NUMINAMATH_CALUDE_origin_fixed_under_dilation_l1792_179211

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square defined by its four vertices -/
structure Square where
  s : Point
  t : Point
  u : Point
  v : Point

/-- Defines a dilation transformation -/
def dilation (center : Point) (k : ℝ) (p : Point) : Point :=
  { x := center.x + k * (p.x - center.x)
  , y := center.y + k * (p.y - center.y) }

theorem origin_fixed_under_dilation (original : Square) (dilated : Square) :
  original.s = Point.mk 3 3 ∧
  original.t = Point.mk 7 3 ∧
  original.u = Point.mk 7 7 ∧
  original.v = Point.mk 3 7 ∧
  dilated.s = Point.mk 6 6 ∧
  dilated.t = Point.mk 12 6 ∧
  dilated.u = Point.mk 12 12 ∧
  dilated.v = Point.mk 6 12 →
  ∃ (k : ℝ), ∀ (p : Point),
    dilation (Point.mk 0 0) k original.s = dilated.s ∧
    dilation (Point.mk 0 0) k original.t = dilated.t ∧
    dilation (Point.mk 0 0) k original.u = dilated.u ∧
    dilation (Point.mk 0 0) k original.v = dilated.v :=
by sorry

end NUMINAMATH_CALUDE_origin_fixed_under_dilation_l1792_179211


namespace NUMINAMATH_CALUDE_square_area_with_circles_l1792_179273

theorem square_area_with_circles (r : ℝ) (h : r = 3) : 
  let d := 2 * r
  let s := 2 * d
  s^2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_square_area_with_circles_l1792_179273


namespace NUMINAMATH_CALUDE_swap_result_l1792_179253

def swap_values (a b : ℕ) : ℕ × ℕ :=
  let t := a
  let a := b
  let b := t
  (a, b)

theorem swap_result : swap_values 3 2 = (2, 3) := by sorry

end NUMINAMATH_CALUDE_swap_result_l1792_179253


namespace NUMINAMATH_CALUDE_butterfat_mixture_l1792_179269

/-- Proves that adding 16 gallons of 10% butterfat milk to 8 gallons of 40% butterfat milk 
    results in a mixture with 20% butterfat. -/
theorem butterfat_mixture : 
  let initial_volume : ℝ := 8
  let initial_butterfat_percent : ℝ := 40
  let added_volume : ℝ := 16
  let added_butterfat_percent : ℝ := 10
  let final_butterfat_percent : ℝ := 20
  let total_volume := initial_volume + added_volume
  let total_butterfat := (initial_volume * initial_butterfat_percent / 100) + 
                         (added_volume * added_butterfat_percent / 100)
  (total_butterfat / total_volume) * 100 = final_butterfat_percent :=
by
  sorry

#check butterfat_mixture

end NUMINAMATH_CALUDE_butterfat_mixture_l1792_179269


namespace NUMINAMATH_CALUDE_wynter_bicycle_count_l1792_179258

/-- The number of bicycles Wynter counted -/
def num_bicycles : ℕ := 50

/-- The number of tricycles Wynter counted -/
def num_tricycles : ℕ := 20

/-- The total number of wheels from all vehicles -/
def total_wheels : ℕ := 160

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

/-- Theorem stating that the number of bicycles Wynter counted is 50 -/
theorem wynter_bicycle_count :
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_wynter_bicycle_count_l1792_179258


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1792_179203

theorem rationalize_denominator : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1792_179203


namespace NUMINAMATH_CALUDE_angle_properties_l1792_179275

-- Define the angle α
variable (α : Real)

-- Define the condition that the terminal side of α lies on y = -√3x
def terminal_side (α : Real) : Prop :=
  ∃ (x y : Real), y = -Real.sqrt 3 * x ∧ x = Real.cos α ∧ y = Real.sin α

-- Define the set S of angles with the same terminal side as α
def S : Set Real :=
  {β | ∃ (k : ℤ), β = k * Real.pi + 2 * Real.pi / 3}

-- State the theorem
theorem angle_properties (h : terminal_side α) :
  (Real.tan α = -Real.sqrt 3) ∧
  (S = {β | ∃ (k : ℤ), β = k * Real.pi + 2 * Real.pi / 3}) ∧
  ((Real.sqrt 3 * Real.sin (α - Real.pi) + 5 * Real.cos (2 * Real.pi - α)) /
   (-Real.sqrt 3 * Real.cos (3 * Real.pi / 2 + α) + Real.cos (Real.pi + α)) = 4) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l1792_179275


namespace NUMINAMATH_CALUDE_parallelograms_from_congruent_triangles_l1792_179248

/-- Represents a triangle -/
structure Triangle :=
  (vertices : Fin 3 → ℝ × ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a quadrilateral is a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop := sorry

/-- Forms quadrilaterals from two triangles -/
def form_quadrilaterals (t1 t2 : Triangle) : Set Quadrilateral := sorry

/-- Counts the number of parallelograms in a set of quadrilaterals -/
def count_parallelograms (qs : Set Quadrilateral) : ℕ := sorry

theorem parallelograms_from_congruent_triangles 
  (t1 t2 : Triangle) 
  (h : are_congruent t1 t2) : 
  count_parallelograms (form_quadrilaterals t1 t2) = 3 := sorry

end NUMINAMATH_CALUDE_parallelograms_from_congruent_triangles_l1792_179248


namespace NUMINAMATH_CALUDE_simplify_and_substitute_l1792_179216

theorem simplify_and_substitute :
  let expression (x : ℝ) := (1 + 1 / x) / ((x^2 - 1) / x)
  expression 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_substitute_l1792_179216


namespace NUMINAMATH_CALUDE_max_piles_for_660_stones_l1792_179217

/-- Represents the stone splitting game -/
structure StoneSplittingGame where
  initial_stones : ℕ
  max_piles : ℕ

/-- Checks if a list of pile sizes is valid according to the game rules -/
def is_valid_configuration (piles : List ℕ) : Prop :=
  ∀ i j, i < piles.length → j < piles.length → 
    2 * piles[i]! > piles[j]! ∧ 2 * piles[j]! > piles[i]!

/-- Theorem stating the maximum number of piles for 660 stones -/
theorem max_piles_for_660_stones (game : StoneSplittingGame) 
  (h1 : game.initial_stones = 660)
  (h2 : game.max_piles = 30) :
  ∃ (piles : List ℕ), 
    piles.length = game.max_piles ∧ 
    piles.sum = game.initial_stones ∧
    is_valid_configuration piles ∧
    ∀ (other_piles : List ℕ), 
      other_piles.sum = game.initial_stones → 
      is_valid_configuration other_piles →
      other_piles.length ≤ game.max_piles :=
sorry


end NUMINAMATH_CALUDE_max_piles_for_660_stones_l1792_179217


namespace NUMINAMATH_CALUDE_fraction_value_given_equation_l1792_179256

theorem fraction_value_given_equation (a b : ℝ) : 
  |5 - a| + (b + 3)^2 = 0 → b / a = -3 / 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_given_equation_l1792_179256


namespace NUMINAMATH_CALUDE_triangle_area_l1792_179287

def a : ℝ × ℝ := (7, 3)
def b : ℝ × ℝ := (-1, 5)

theorem triangle_area : 
  let det := a.1 * b.2 - a.2 * b.1
  (1/2 : ℝ) * |det| = 19 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1792_179287


namespace NUMINAMATH_CALUDE_kingfisher_pelican_fish_difference_l1792_179241

theorem kingfisher_pelican_fish_difference (pelican_fish : ℕ) (kingfisher_fish : ℕ) (fisherman_fish : ℕ) : 
  pelican_fish = 13 →
  kingfisher_fish > pelican_fish →
  fisherman_fish = 3 * (pelican_fish + kingfisher_fish) →
  fisherman_fish = pelican_fish + 86 →
  kingfisher_fish - pelican_fish = 7 := by
sorry

end NUMINAMATH_CALUDE_kingfisher_pelican_fish_difference_l1792_179241


namespace NUMINAMATH_CALUDE_complex_exponential_185_54_l1792_179291

theorem complex_exponential_185_54 :
  (Complex.exp (185 * Real.pi / 180 * Complex.I))^54 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_185_54_l1792_179291


namespace NUMINAMATH_CALUDE_count_five_divisors_l1792_179264

theorem count_five_divisors (n : ℕ) (h : n = 50000) : 
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125) + (n / 15625) = 12499 :=
by sorry

end NUMINAMATH_CALUDE_count_five_divisors_l1792_179264


namespace NUMINAMATH_CALUDE_eight_hash_four_eq_eighteen_l1792_179205

-- Define the operation #
def hash (a b : ℚ) : ℚ := 2 * a + a / b

-- Theorem statement
theorem eight_hash_four_eq_eighteen : hash 8 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_eight_hash_four_eq_eighteen_l1792_179205


namespace NUMINAMATH_CALUDE_kindergartners_with_orange_shirts_l1792_179236

-- Define the constants from the problem
def first_graders : ℕ := 113
def second_graders : ℕ := 107
def third_graders : ℕ := 108

def yellow_shirt_cost : ℚ := 5
def blue_shirt_cost : ℚ := 56/10
def green_shirt_cost : ℚ := 21/4
def orange_shirt_cost : ℚ := 29/5

def total_spent : ℚ := 2317

-- Theorem to prove
theorem kindergartners_with_orange_shirts :
  (total_spent -
   (first_graders * yellow_shirt_cost +
    second_graders * blue_shirt_cost +
    third_graders * green_shirt_cost)) / orange_shirt_cost = 101 := by
  sorry

end NUMINAMATH_CALUDE_kindergartners_with_orange_shirts_l1792_179236


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l1792_179283

theorem sin_pi_minus_alpha (α : Real) :
  (∃ (x y : Real), x = -4 ∧ y = 3 ∧ x^2 + y^2 = 25 ∧ 
   (∃ (r : Real), r > 0 ∧ x = r * (Real.cos α) ∧ y = r * (Real.sin α))) →
  Real.sin (π - α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l1792_179283


namespace NUMINAMATH_CALUDE_limit_exists_l1792_179220

/-- Prove the existence of δ(ε) for the limit of (5x^2 - 24x - 5) / (x - 5) as x approaches 5 -/
theorem limit_exists (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, x ≠ 5 → |x - 5| < δ →
    |(5 * x^2 - 24 * x - 5) / (x - 5) - 26| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_exists_l1792_179220


namespace NUMINAMATH_CALUDE_range_of_S_l1792_179231

theorem range_of_S (x : ℝ) (y : ℝ) (S : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : 0 ≤ x ∧ x ≤ 1/2) 
  (h3 : S = x * y) : 
  -1/8 ≤ S ∧ S ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_S_l1792_179231


namespace NUMINAMATH_CALUDE_field_trip_cost_calculation_l1792_179270

def field_trip_cost (num_students : ℕ) (num_teachers : ℕ) (student_ticket_price : ℚ) 
  (adult_ticket_price : ℚ) (discount_rate : ℚ) (min_tickets_for_discount : ℕ) 
  (transportation_cost : ℚ) (meal_cost_per_person : ℚ) : ℚ :=
  sorry

theorem field_trip_cost_calculation : 
  field_trip_cost 25 6 1 3 0.2 20 100 7.5 = 366.9 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_cost_calculation_l1792_179270


namespace NUMINAMATH_CALUDE_citadel_school_earnings_l1792_179281

/-- Represents the total earnings for a school in the summer project. -/
def schoolEarnings (totalPayment : ℚ) (totalStudentDays : ℕ) (schoolStudentDays : ℕ) : ℚ :=
  (totalPayment / totalStudentDays) * schoolStudentDays

/-- Theorem: The earnings for Citadel school in the summer project. -/
theorem citadel_school_earnings :
  let apexDays : ℕ := 9 * 5
  let beaconDays : ℕ := 3 * 4
  let citadelDays : ℕ := 6 * 7
  let totalDays : ℕ := apexDays + beaconDays + citadelDays
  let totalPayment : ℚ := 864
  schoolEarnings totalPayment totalDays citadelDays = 864 / 99 * 42 :=
by sorry

#eval schoolEarnings 864 99 42

end NUMINAMATH_CALUDE_citadel_school_earnings_l1792_179281


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1792_179252

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 12) (h2 : x ≠ -4) :
  (7 * x - 5) / (x^2 - 8*x - 48) = (79/16) / (x - 12) + (33/16) / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1792_179252


namespace NUMINAMATH_CALUDE_cosine_graph_shift_l1792_179285

/-- Given a cosine function f(x) = 3cos(2x), prove that shifting its graph π/6 units 
    to the right results in the graph of g(x) = 3cos(2x - π/3) -/
theorem cosine_graph_shift (x : ℝ) :
  3 * Real.cos (2 * (x - π / 6)) = 3 * Real.cos (2 * x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_cosine_graph_shift_l1792_179285


namespace NUMINAMATH_CALUDE_total_age_now_l1792_179299

-- Define Xavier's and Yasmin's ages as natural numbers
variable (xavier_age yasmin_age : ℕ)

-- Define the conditions
axiom xavier_twice_yasmin : xavier_age = 2 * yasmin_age
axiom xavier_future_age : xavier_age + 6 = 30

-- Theorem to prove
theorem total_age_now : xavier_age + yasmin_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_age_now_l1792_179299


namespace NUMINAMATH_CALUDE_product_pqr_l1792_179246

theorem product_pqr (p q r : ℤ) 
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_sum : p + q + r = 26)
  (h_eq : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 360 / (p * q * r) = 1) :
  p * q * r = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_pqr_l1792_179246


namespace NUMINAMATH_CALUDE_min_sum_of_even_factors_l1792_179284

theorem min_sum_of_even_factors (a b : ℤ) : 
  Even a → Even b → a * b = 144 → (∀ x y : ℤ, Even x → Even y → x * y = 144 → a + b ≤ x + y) → a + b = -74 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_even_factors_l1792_179284


namespace NUMINAMATH_CALUDE_tan_order_l1792_179290

theorem tan_order : 
  (1 < Real.pi / 2) → 
  (Real.pi / 2 < 2) → 
  (2 < 3) → 
  (3 < Real.pi) → 
  (∀ x y, Real.pi / 2 < x → x < y → y < Real.pi → Real.tan x < Real.tan y) →
  Real.tan 1 > Real.tan 3 ∧ Real.tan 3 > Real.tan 2 :=
by sorry

end NUMINAMATH_CALUDE_tan_order_l1792_179290


namespace NUMINAMATH_CALUDE_three_pairs_same_difference_l1792_179296

theorem three_pairs_same_difference (X : Finset ℕ) 
  (h1 : X ⊆ Finset.range 18 \ {0})
  (h2 : X.card = 8) : 
  ∃ (a b c d e f : ℕ), a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ d ∈ X ∧ e ∈ X ∧ f ∈ X ∧ 
  a ≠ b ∧ c ≠ d ∧ e ≠ f ∧
  (a - b : ℤ) = (c - d : ℤ) ∧ (c - d : ℤ) = (e - f : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_three_pairs_same_difference_l1792_179296


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_36_l1792_179243

theorem ceiling_neg_sqrt_36 : ⌈-Real.sqrt 36⌉ = -6 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_36_l1792_179243


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1792_179212

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a^2 + b^2 + c^2 = 2500 →  -- Given condition
  c = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1792_179212


namespace NUMINAMATH_CALUDE_greatest_integer_in_ratio_l1792_179240

theorem greatest_integer_in_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 84 →
  2 * b = 5 * a →
  7 * a = 2 * c →
  max a (max b c) = 42 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_in_ratio_l1792_179240


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_one_l1792_179274

/-- The equation has exactly one solution if and only if a = 1 -/
theorem unique_solution_iff_a_eq_one (a : ℝ) :
  (∃! x : ℝ, 5^(x^2 - 6*a*x + 9*a^2) = a*x^2 - 6*a^2*x + 9*a^3 + a^2 - 6*a + 6) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_one_l1792_179274


namespace NUMINAMATH_CALUDE_min_box_value_l1792_179297

theorem min_box_value (a b Box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + Box * x + 15) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  (∃ m : ℤ, Box ≥ m ∧ (∃ a' b' : ℤ, 
    (∀ x, (a' * x + b') * (b' * x + a') = 15 * x^2 + m * x + 15) ∧
    a' ≠ b' ∧ b' ≠ m ∧ a' ≠ m)) →
  Box ≥ 34 := by
sorry

end NUMINAMATH_CALUDE_min_box_value_l1792_179297


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_l1792_179218

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x - 1|

-- Theorem 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Theorem 2
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ {x : ℝ | 1/2 ≤ x ∧ x ≤ 1}, f a x ≤ |2*x + 1|) →
  -1 ≤ a ∧ a ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_l1792_179218


namespace NUMINAMATH_CALUDE_inverse_256_mod_101_l1792_179223

theorem inverse_256_mod_101 (h : (16⁻¹ : ZMod 101) = 31) :
  (256⁻¹ : ZMod 101) = 52 := by
  sorry

end NUMINAMATH_CALUDE_inverse_256_mod_101_l1792_179223


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_three_tosses_prob_at_least_one_head_three_tosses_is_seven_eighths_l1792_179201

/-- The probability of getting at least one head when tossing a fair coin three times -/
theorem prob_at_least_one_head_three_tosses : ℚ :=
  let S := Finset.powerset {1, 2, 3}
  let favorable_outcomes := S.filter (λ s => s.card > 0)
  favorable_outcomes.card / S.card

theorem prob_at_least_one_head_three_tosses_is_seven_eighths :
  prob_at_least_one_head_three_tosses = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_head_three_tosses_prob_at_least_one_head_three_tosses_is_seven_eighths_l1792_179201


namespace NUMINAMATH_CALUDE_trivia_team_distribution_l1792_179232

theorem trivia_team_distribution (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total_students = 58)
  (h2 : not_picked = 10)
  (h3 : num_groups = 8) :
  (total_students - not_picked) / num_groups = 6 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_distribution_l1792_179232


namespace NUMINAMATH_CALUDE_sign_up_options_count_l1792_179221

/-- The number of students. -/
def num_students : ℕ := 5

/-- The number of teams. -/
def num_teams : ℕ := 3

/-- The total number of sign-up options. -/
def total_options : ℕ := num_teams ^ num_students

/-- 
Theorem: Given 5 students and 3 teams, where each student must choose exactly one team,
the total number of possible sign-up combinations is 3^5.
-/
theorem sign_up_options_count :
  total_options = 243 := by sorry

end NUMINAMATH_CALUDE_sign_up_options_count_l1792_179221


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1792_179261

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_z : ℝ), min_z = 12 ∧ ∀ z : ℝ, z = 4*x^2 + 8*x + 16 → z ≥ min_z :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1792_179261


namespace NUMINAMATH_CALUDE_pen_retailer_profit_percentage_pen_retailer_profit_is_ten_percent_l1792_179245

/-- Calculates the profit percentage for a pen retailer. -/
theorem pen_retailer_profit_percentage 
  (num_pens_bought : ℕ) 
  (num_pens_price : ℕ) 
  (discount_percent : ℚ) : ℚ :=
  let cost_price := num_pens_price
  let market_price_per_pen := 1
  let selling_price_per_pen := market_price_per_pen * (1 - discount_percent / 100)
  let total_selling_price := num_pens_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

/-- Proves that the profit percentage is 10% for the given scenario. -/
theorem pen_retailer_profit_is_ten_percent :
  pen_retailer_profit_percentage 40 36 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pen_retailer_profit_percentage_pen_retailer_profit_is_ten_percent_l1792_179245


namespace NUMINAMATH_CALUDE_shirt_cost_l1792_179214

theorem shirt_cost (J S K : ℚ) 
  (eq1 : 3 * J + 2 * S + K = 110)
  (eq2 : 2 * J + 3 * S + 2 * K = 176)
  (eq3 : 4 * J + S + 3 * K = 254) :
  S = 5.6 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l1792_179214


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l1792_179210

theorem factorial_equation_solution (n : ℕ) : n * n! + n! = 5040 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l1792_179210


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_and_a_div_b_l1792_179244

theorem range_of_a_minus_b_and_a_div_b (a b : ℝ) 
  (ha : 12 < a ∧ a < 60) (hb : 15 < b ∧ b < 36) : 
  (-24 < a - b ∧ a - b < 45) ∧ (1/3 < a/b ∧ a/b < 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_and_a_div_b_l1792_179244


namespace NUMINAMATH_CALUDE_subtraction_result_l1792_179249

theorem subtraction_result : 500000000000 - 3 * 111111111111 = 166666666667 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l1792_179249


namespace NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l1792_179242

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_percentage_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  ((1.3 * l) * (1.2 * w) - l * w) / (l * w) = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l1792_179242
