import Mathlib

namespace NUMINAMATH_CALUDE_product_eleven_cubed_sum_l841_84198

theorem product_eleven_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 11^3 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 133 := by sorry

end NUMINAMATH_CALUDE_product_eleven_cubed_sum_l841_84198


namespace NUMINAMATH_CALUDE_expression_simplification_l841_84129

theorem expression_simplification (y : ℝ) :
  3 * y + 12 * y^2 + 18 - (6 - 3 * y - 12 * y^2) + 5 * y^3 = 5 * y^3 + 24 * y^2 + 6 * y + 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l841_84129


namespace NUMINAMATH_CALUDE_boys_without_calculators_l841_84162

theorem boys_without_calculators (total_students : Nat) (boys : Nat) (students_with_calculators : Nat) (girls_with_calculators : Nat)
  (h1 : total_students = 30)
  (h2 : boys = 20)
  (h3 : students_with_calculators = 25)
  (h4 : girls_with_calculators = 18)
  : total_students - boys - (students_with_calculators - girls_with_calculators) = 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l841_84162


namespace NUMINAMATH_CALUDE_min_groups_for_class_l841_84114

/-- Given a class of 30 students and a maximum group size of 12,
    proves that the minimum number of equal-sized groups is 3. -/
theorem min_groups_for_class (total_students : ℕ) (max_group_size : ℕ) :
  total_students = 30 →
  max_group_size = 12 →
  ∃ (group_size : ℕ), 
    group_size ≤ max_group_size ∧
    total_students % group_size = 0 ∧
    (total_students / group_size = 3) ∧
    ∀ (other_size : ℕ), 
      other_size ≤ max_group_size →
      total_students % other_size = 0 →
      total_students / other_size ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_groups_for_class_l841_84114


namespace NUMINAMATH_CALUDE_flower_count_l841_84118

theorem flower_count (num_bees : ℕ) (num_flowers : ℕ) : 
  num_bees = 3 → num_bees = num_flowers - 2 → num_flowers = 5 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l841_84118


namespace NUMINAMATH_CALUDE_quadrilaterals_form_polygons_l841_84160

/-- A point in 2D space --/
structure Point :=
  (x : ℤ)
  (y : ℤ)

/-- A polygon defined by its vertices --/
structure Polygon :=
  (vertices : List Point)

/-- Definition of a square --/
def is_square (p : Polygon) : Prop :=
  p.vertices.length = 4 ∧
  ∃ (x y : ℤ), p.vertices = [Point.mk x y, Point.mk (x+2) y, Point.mk (x+2) (y+2), Point.mk x (y+2)]

/-- Definition of a triangle --/
def is_triangle (p : Polygon) : Prop :=
  p.vertices.length = 3

/-- Definition of a pentagon --/
def is_pentagon (p : Polygon) : Prop :=
  p.vertices.length = 5

/-- The two squares from the problem --/
def square1 : Polygon :=
  Polygon.mk [Point.mk 0 0, Point.mk 2 0, Point.mk 2 2, Point.mk 0 2]

def square2 : Polygon :=
  Polygon.mk [Point.mk 2 2, Point.mk 4 2, Point.mk 4 4, Point.mk 2 4]

/-- Main theorem --/
theorem quadrilaterals_form_polygons :
  (is_square square1 ∧ is_square square2) →
  (∃ (t : Polygon) (p : Polygon), is_triangle t ∧ is_pentagon p ∧
    (∀ v : Point, v ∈ t.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices) ∧
    (∀ v : Point, v ∈ p.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices)) ∧
  (∃ (t : Polygon) (q : Polygon) (p : Polygon), 
    is_triangle t ∧ p.vertices.length = 4 ∧ is_pentagon p ∧
    (∀ v : Point, v ∈ t.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices) ∧
    (∀ v : Point, v ∈ q.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices) ∧
    (∀ v : Point, v ∈ p.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices)) :=
by sorry

end NUMINAMATH_CALUDE_quadrilaterals_form_polygons_l841_84160


namespace NUMINAMATH_CALUDE_c_necessary_not_sufficient_l841_84101

-- Define the proposition p
def p (x : ℝ) : Prop := x^2 - x < 0

-- Define the condition c
def c (x : ℝ) : Prop := -1 < x ∧ x < 1

-- Theorem stating that c is a necessary but not sufficient condition for p
theorem c_necessary_not_sufficient :
  (∀ x : ℝ, p x → c x) ∧ 
  (∃ x : ℝ, c x ∧ ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_c_necessary_not_sufficient_l841_84101


namespace NUMINAMATH_CALUDE_triangle_parallelogram_relation_l841_84117

theorem triangle_parallelogram_relation (triangle_area : ℝ) (parallelogram_height : ℝ) : 
  triangle_area = 15 → parallelogram_height = 5 → 
  ∃ (parallelogram_area parallelogram_base : ℝ),
    parallelogram_area = 2 * triangle_area ∧
    parallelogram_area = parallelogram_height * parallelogram_base ∧
    parallelogram_area = 30 ∧
    parallelogram_base = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_relation_l841_84117


namespace NUMINAMATH_CALUDE_smallest_pair_with_six_coins_l841_84193

/-- Represents the value of a coin in half-pennies -/
inductive Coin : Nat → Type where
  | halfpenny : Coin 1
  | penny : Coin 2
  | threepence : Coin 6
  | fourpence : Coin 8
  | sixpence : Coin 12
  | shilling : Coin 24

/-- Checks if an amount can be represented with exactly 6 coins -/
def representableWithSixCoins (amount : Nat) : Prop :=
  ∃ (c₁ c₂ c₃ c₄ c₅ c₆ : Nat),
    (∃ (coin₁ : Coin c₁) (coin₂ : Coin c₂) (coin₃ : Coin c₃)
        (coin₄ : Coin c₄) (coin₅ : Coin c₅) (coin₆ : Coin c₆),
      c₁ + c₂ + c₃ + c₄ + c₅ + c₆ = amount)

/-- The main theorem to prove -/
theorem smallest_pair_with_six_coins :
  ∀ (a b : Nat),
    a < 60 ∧ b < 60 ∧ a < b ∧
    representableWithSixCoins a ∧
    representableWithSixCoins b ∧
    representableWithSixCoins (a + b) →
    a ≥ 23 ∧ b ≥ 47 :=
sorry

end NUMINAMATH_CALUDE_smallest_pair_with_six_coins_l841_84193


namespace NUMINAMATH_CALUDE_ajays_income_l841_84141

/-- Ajay's monthly income in Rupees -/
def monthly_income : ℝ := 90000

/-- Percentage of income spent on household items -/
def household_percentage : ℝ := 0.50

/-- Percentage of income spent on clothes -/
def clothes_percentage : ℝ := 0.25

/-- Percentage of income spent on medicines -/
def medicines_percentage : ℝ := 0.15

/-- Amount saved in Rupees -/
def savings : ℝ := 9000

theorem ajays_income :
  monthly_income * household_percentage +
  monthly_income * clothes_percentage +
  monthly_income * medicines_percentage +
  savings = monthly_income :=
by sorry

end NUMINAMATH_CALUDE_ajays_income_l841_84141


namespace NUMINAMATH_CALUDE_inequality_solution_l841_84103

theorem inequality_solution (a : ℝ) : 
  (∀ x > 0, (a * x - 9) * Real.log (2 * a / x) ≤ 0) ↔ a = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l841_84103


namespace NUMINAMATH_CALUDE_t_shaped_figure_perimeter_l841_84136

/-- A geometric figure composed of four identical squares in a T shape -/
structure TShapedFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 144 cm² -/
  area_eq : 4 * side_length ^ 2 = 144

/-- The perimeter of a T-shaped figure -/
def perimeter (f : TShapedFigure) : ℝ :=
  5 * f.side_length

theorem t_shaped_figure_perimeter (f : TShapedFigure) : perimeter f = 30 :=
sorry

end NUMINAMATH_CALUDE_t_shaped_figure_perimeter_l841_84136


namespace NUMINAMATH_CALUDE_cubic_function_c_value_l841_84143

theorem cubic_function_c_value (a b c d y₁ y₂ : ℝ) :
  y₁ = a + b + c + d →
  y₂ = 8*a + 4*b + 2*c + d →
  y₁ - y₂ = -17 →
  c = -17 + 7*a + 3*b :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_c_value_l841_84143


namespace NUMINAMATH_CALUDE_find_x_l841_84150

theorem find_x : ∃ x : ℝ, (85 + x / 113) * 113 = 9637 ∧ x = 9552 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l841_84150


namespace NUMINAMATH_CALUDE_sum_of_odd_naturals_900_l841_84107

theorem sum_of_odd_naturals_900 :
  ∃ n : ℕ, n^2 = 900 ∧ (∀ k : ℕ, k ≤ n → (2*k - 1) ≤ n^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odd_naturals_900_l841_84107


namespace NUMINAMATH_CALUDE_max_value_of_fraction_difference_l841_84122

theorem max_value_of_fraction_difference (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 4 * a - b ≥ 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → 4 * x - y ≥ 2 → 1 / x - 1 / y ≤ 1 / 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x - y ≥ 2 ∧ 1 / x - 1 / y = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_difference_l841_84122


namespace NUMINAMATH_CALUDE_existence_of_non_dividing_sum_l841_84171

theorem existence_of_non_dividing_sum (n : ℕ) (a : Fin n → ℕ+) (h_n : n ≥ 3) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∃ i j, i ≠ j ∧ ∀ k, ¬((a i + a j : ℕ) ∣ (3 * (a k : ℕ))) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_dividing_sum_l841_84171


namespace NUMINAMATH_CALUDE_set_equality_l841_84196

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {3, 5}

-- Theorem statement
theorem set_equality : U = A ∪ (U \ B) := by sorry

end NUMINAMATH_CALUDE_set_equality_l841_84196


namespace NUMINAMATH_CALUDE_even_decreasing_function_inequality_l841_84161

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on (0, +∞) if for all x, y ∈ (0, +∞),
    x < y implies f(x) > f(y) -/
def IsDecreasingOnPositiveReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x > f y

theorem even_decreasing_function_inequality
  (f : ℝ → ℝ)
  (heven : IsEven f)
  (hdecr : IsDecreasingOnPositiveReals f) :
  f (-5) < f (-4) ∧ f (-4) < f 3 :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_function_inequality_l841_84161


namespace NUMINAMATH_CALUDE_late_condition_early_condition_60_early_condition_50_exact_time_speed_correct_l841_84111

/-- Represents the speed Mr. Bird needs to drive to arrive exactly on time -/
def exact_time_speed : ℚ :=
  160 / 3

/-- Represents the distance to Mr. Bird's workplace in miles -/
def distance_to_work : ℚ :=
  40 / 3

/-- Represents the ideal time to reach work on time in hours -/
def ideal_time : ℚ :=
  1 / 4

/-- Given that driving at 40 mph makes Mr. Bird 5 minutes late -/
theorem late_condition (speed : ℚ) (time : ℚ) :
  speed = 40 → time = ideal_time + 5 / 60 → speed * time = distance_to_work :=
sorry

/-- Given that driving at 60 mph makes Mr. Bird 2 minutes early -/
theorem early_condition_60 (speed : ℚ) (time : ℚ) :
  speed = 60 → time = ideal_time - 2 / 60 → speed * time = distance_to_work :=
sorry

/-- Given that driving at 50 mph makes Mr. Bird 1 minute early -/
theorem early_condition_50 (speed : ℚ) (time : ℚ) :
  speed = 50 → time = ideal_time - 1 / 60 → speed * time = distance_to_work :=
sorry

/-- Theorem stating that the exact_time_speed is the speed required to arrive exactly on time -/
theorem exact_time_speed_correct :
  exact_time_speed * ideal_time = distance_to_work :=
sorry

end NUMINAMATH_CALUDE_late_condition_early_condition_60_early_condition_50_exact_time_speed_correct_l841_84111


namespace NUMINAMATH_CALUDE_correct_pricing_l841_84119

/-- A hotel's pricing structure -/
structure HotelPricing where
  flat_fee : ℝ
  additional_fee : ℝ
  discount : ℝ := 10

/-- Calculate the cost of a stay given the pricing and number of nights -/
def stay_cost (p : HotelPricing) (nights : ℕ) : ℝ :=
  if nights ≤ 4 then
    p.flat_fee + p.additional_fee * (nights - 1 : ℝ)
  else
    p.flat_fee + p.additional_fee * 3 + (p.additional_fee - p.discount) * ((nights - 4) : ℝ)

/-- The theorem stating the correct pricing structure -/
theorem correct_pricing :
  ∃ (p : HotelPricing),
    stay_cost p 4 = 180 ∧
    stay_cost p 7 = 302 ∧
    p.flat_fee = 28 ∧
    p.additional_fee = 50.67 := by
  sorry

end NUMINAMATH_CALUDE_correct_pricing_l841_84119


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_l841_84152

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P -/
def P : CartesianPoint :=
  { x := 1, y := -4 }

/-- Theorem: P lies in the fourth quadrant -/
theorem P_in_fourth_quadrant : is_in_fourth_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_l841_84152


namespace NUMINAMATH_CALUDE_at_most_one_root_l841_84194

-- Define a monotonically increasing function on an interval
def MonoIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Theorem statement
theorem at_most_one_root (f : ℝ → ℝ) (a b : ℝ) (h : MonoIncreasing f a b) :
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_at_most_one_root_l841_84194


namespace NUMINAMATH_CALUDE_constant_function_from_parallel_tangent_l841_84158

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the property that the tangent line is parallel to the x-axis at every point
def tangent_parallel_to_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, deriv f x = 0

-- Theorem statement
theorem constant_function_from_parallel_tangent :
  tangent_parallel_to_x_axis f → ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by
  sorry

end NUMINAMATH_CALUDE_constant_function_from_parallel_tangent_l841_84158


namespace NUMINAMATH_CALUDE_arithmetic_computation_l841_84185

theorem arithmetic_computation : 143 - 13 + 31 + 17 = 178 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l841_84185


namespace NUMINAMATH_CALUDE_farmer_randy_cotton_acres_l841_84170

/-- The number of acres a single tractor can plant in one day -/
def acres_per_day : ℕ := 68

/-- The number of tractors working for the first two days -/
def tractors_first_two_days : ℕ := 2

/-- The number of tractors working for the last three days -/
def tractors_last_three_days : ℕ := 7

/-- The number of days in the first period -/
def first_period_days : ℕ := 2

/-- The number of days in the second period -/
def second_period_days : ℕ := 3

/-- The total number of acres Farmer Randy needs to have planted -/
def total_acres : ℕ := 1700

theorem farmer_randy_cotton_acres :
  total_acres = 
    acres_per_day * first_period_days * tractors_first_two_days + 
    acres_per_day * second_period_days * tractors_last_three_days :=
by sorry

end NUMINAMATH_CALUDE_farmer_randy_cotton_acres_l841_84170


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l841_84134

/-- The shortest distance from a point on the parabola y = x^2 to the line 2x - y = 4 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  let line := {p : ℝ × ℝ | 2 * p.1 - p.2 = 4}
  let distance (p : ℝ × ℝ) := |2 * p.1 - p.2 - 4| / Real.sqrt 5
  (∀ p ∈ parabola, distance p ≥ 3 * Real.sqrt 5 / 5) ∧
  (∃ p ∈ parabola, distance p = 3 * Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l841_84134


namespace NUMINAMATH_CALUDE_garden_theorem_l841_84148

def garden_problem (initial_plants : ℕ) (day1_eaten : ℕ) (day3_eaten : ℕ) : ℕ :=
  let remaining_day1 := initial_plants - day1_eaten
  let remaining_day2 := remaining_day1 / 2
  remaining_day2 - day3_eaten

theorem garden_theorem :
  garden_problem 30 20 1 = 4 := by
  sorry

#eval garden_problem 30 20 1

end NUMINAMATH_CALUDE_garden_theorem_l841_84148


namespace NUMINAMATH_CALUDE_hvac_cost_per_vent_l841_84130

/-- Calculates the cost per vent of an HVAC system -/
def cost_per_vent (total_cost : ℕ) (num_zones : ℕ) (vents_per_zone : ℕ) : ℕ :=
  total_cost / (num_zones * vents_per_zone)

/-- Proves that the cost per vent of the given HVAC system is $2,000 -/
theorem hvac_cost_per_vent :
  cost_per_vent 20000 2 5 = 2000 := by
  sorry

#eval cost_per_vent 20000 2 5

end NUMINAMATH_CALUDE_hvac_cost_per_vent_l841_84130


namespace NUMINAMATH_CALUDE_num_men_is_seven_l841_84180

/-- Represents the amount of work a person can do per hour -/
structure WorkRate where
  amount : ℝ

/-- The number of men working with 2 boys -/
def numMen : ℕ := sorry

/-- The work rate of a man -/
def manWorkRate : WorkRate := sorry

/-- The work rate of a boy -/
def boyWorkRate : WorkRate := sorry

/-- The ratio of work done by a man to a boy is 4:1 -/
axiom work_ratio : manWorkRate.amount = 4 * boyWorkRate.amount

/-- The group (numMen men and 2 boys) can do 6 times as much work per hour as a man and a boy together -/
axiom group_work_rate : 
  numMen * manWorkRate.amount + 2 * boyWorkRate.amount = 
  6 * (manWorkRate.amount + boyWorkRate.amount)

theorem num_men_is_seven : numMen = 7 := by sorry

end NUMINAMATH_CALUDE_num_men_is_seven_l841_84180


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l841_84173

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 10
  let θ : ℝ := 4 * π / 3
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x = -5 * Real.sqrt 3) ∧ (y = -15 / 2) ∧ (z = 5) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l841_84173


namespace NUMINAMATH_CALUDE_correct_addition_with_digit_change_l841_84187

theorem correct_addition_with_digit_change :
  ∃ (d e : ℕ), d ≠ e ∧ d < 10 ∧ e < 10 ∧
  ((853697 + 930541 = 1383238 ∧ d = 8 ∧ e = 4) ∨
   (453697 + 930541 = 1383238 ∧ d = 8 ∧ e = 4)) ∧
  d + e = 12 := by
sorry

end NUMINAMATH_CALUDE_correct_addition_with_digit_change_l841_84187


namespace NUMINAMATH_CALUDE_function_properties_l841_84153

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom cond1 : ∀ x, f (10 + x) = f (10 - x)
axiom cond2 : ∀ x, f (20 - x) = -f (20 + x)

-- Theorem to prove
theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 40) = f x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l841_84153


namespace NUMINAMATH_CALUDE_children_attendance_l841_84179

/-- Proves the number of children attending a concert given ticket prices and total revenue -/
theorem children_attendance (adult_price : ℕ) (adult_count : ℕ) (total_revenue : ℕ) : 
  adult_price = 26 →
  adult_count = 183 →
  total_revenue = 5122 →
  ∃ (child_count : ℕ), 
    adult_price * adult_count + (adult_price / 2) * child_count = total_revenue ∧
    child_count = 28 := by
  sorry

end NUMINAMATH_CALUDE_children_attendance_l841_84179


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l841_84178

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_abc_theorem (t : Triangle) 
  (h1 : t.b * Real.sin t.C = Real.sqrt 3)
  (h2 : t.B = π / 4)
  (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = 9 / 2) :
  t.c = Real.sqrt 6 ∧ t.b = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l841_84178


namespace NUMINAMATH_CALUDE_matrix_transpose_inverse_sum_squares_l841_84133

theorem matrix_transpose_inverse_sum_squares (p q r s : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![p, q; r, s]
  B.transpose = B⁻¹ →
  p^2 + q^2 + r^2 + s^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_matrix_transpose_inverse_sum_squares_l841_84133


namespace NUMINAMATH_CALUDE_jills_net_salary_l841_84156

/-- Calculates the net monthly salary given the discretionary income ratio and remaining amount --/
def calculate_net_salary (discretionary_ratio : ℚ) (vacation_ratio : ℚ) (savings_ratio : ℚ) 
  (socializing_ratio : ℚ) (remaining_amount : ℚ) : ℚ :=
  remaining_amount / (discretionary_ratio * (1 - (vacation_ratio + savings_ratio + socializing_ratio)))

/-- Proves that given the specified conditions, Jill's net monthly salary is $3700 --/
theorem jills_net_salary :
  let discretionary_ratio : ℚ := 1/5
  let vacation_ratio : ℚ := 30/100
  let savings_ratio : ℚ := 20/100
  let socializing_ratio : ℚ := 35/100
  let remaining_amount : ℚ := 111
  calculate_net_salary discretionary_ratio vacation_ratio savings_ratio socializing_ratio remaining_amount = 3700 := by
  sorry

#eval calculate_net_salary (1/5) (30/100) (20/100) (35/100) 111

end NUMINAMATH_CALUDE_jills_net_salary_l841_84156


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_l841_84124

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 1

-- State the theorem
theorem range_of_f_on_interval (m : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ -2 → f m x₁ > f m x₂) ∧ 
  (∀ x₁ x₂, -2 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →
  Set.Icc (f m 1) (f m 2) = Set.Icc (-11) 33 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_l841_84124


namespace NUMINAMATH_CALUDE_variance_2xi_plus_3_l841_84120

variable (ξ : ℝ → ℝ)

-- D represents the variance operator
def D (X : ℝ → ℝ) : ℝ := sorry

-- Given condition
axiom variance_xi : D ξ = 2

-- Theorem to prove
theorem variance_2xi_plus_3 : D (fun ω => 2 * ξ ω + 3) = 8 := by sorry

end NUMINAMATH_CALUDE_variance_2xi_plus_3_l841_84120


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l841_84125

-- Define a circle
def Circle : Type := Unit

-- Define a color
inductive Color
| Red
| Blue

-- Define a point on the circle
structure Point (c : Circle) where
  color : Color

-- Define a coloring of the circle
def Coloring (c : Circle) := Point c → Color

-- Define an isosceles triangle
structure IsoscelesTriangle (c : Circle) where
  a : Point c
  b : Point c
  c : Point c
  isIsosceles : True  -- Placeholder for the isosceles property

-- Theorem statement
theorem isosceles_triangle_exists (c : Circle) (coloring : Coloring c) :
  ∃ (t : IsoscelesTriangle c), t.a.color = t.b.color ∧ t.b.color = t.c.color :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l841_84125


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l841_84166

theorem furniture_shop_cost_price (selling_price : ℕ) (markup_percentage : ℕ) : 
  selling_price = 1000 → markup_percentage = 100 → 
  ∃ (cost_price : ℕ), cost_price * (100 + markup_percentage) / 100 = selling_price ∧ cost_price = 500 := by
sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l841_84166


namespace NUMINAMATH_CALUDE_smallest_x_congruence_and_divisible_l841_84142

theorem smallest_x_congruence_and_divisible (x : ℕ) : x = 45 ↔ 
  (x > 0 ∧ 
   (x + 6721) % 12 = 3458 % 12 ∧ 
   x % 5 = 0 ∧
   ∀ y : ℕ, y > 0 → (y + 6721) % 12 = 3458 % 12 → y % 5 = 0 → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_congruence_and_divisible_l841_84142


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l841_84132

/-- Given two similar triangles, where the first triangle has sides of 8, 10, and 12,
    and the second triangle has a perimeter of 150, prove that the longest side
    of the second triangle is 60. -/
theorem similar_triangle_longest_side
  (triangle1 : ℝ × ℝ × ℝ)
  (triangle2 : ℝ × ℝ × ℝ)
  (h_triangle1 : triangle1 = (8, 10, 12))
  (h_similar : ∃ (k : ℝ), triangle2 = (8*k, 10*k, 12*k))
  (h_perimeter : triangle2.1 + triangle2.2.1 + triangle2.2.2 = 150)
  : triangle2.2.2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l841_84132


namespace NUMINAMATH_CALUDE_inequality_proof_l841_84157

theorem inequality_proof (a b c : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  (a / (b + c + 1)) + (b / (c + a + 1)) + (c / (a + b + 1)) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l841_84157


namespace NUMINAMATH_CALUDE_tennis_ball_storage_l841_84154

theorem tennis_ball_storage (n : ℕ) : n = 105 ↔ 
  (n % 25 = 5 ∧ n % 20 = 5 ∧ ∀ m : ℕ, m < n → (m % 25 ≠ 5 ∨ m % 20 ≠ 5)) :=
by sorry

end NUMINAMATH_CALUDE_tennis_ball_storage_l841_84154


namespace NUMINAMATH_CALUDE_sum_product_bound_l841_84159

theorem sum_product_bound (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  -1/2 ≤ a*b + b*c + c*a ∧ a*b + b*c + c*a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bound_l841_84159


namespace NUMINAMATH_CALUDE_peggy_left_knee_bandages_l841_84123

/-- The number of bandages Peggy used on her left knee -/
def bandages_on_left_knee (initial_bandages : ℕ) (remaining_bandages : ℕ) (right_knee_bandages : ℕ) : ℕ :=
  initial_bandages - remaining_bandages - right_knee_bandages

/-- Proof that Peggy used 2 bandages on her left knee -/
theorem peggy_left_knee_bandages : 
  let initial_bandages := 24 - 8
  let remaining_bandages := 11
  let right_knee_bandages := 3
  bandages_on_left_knee initial_bandages remaining_bandages right_knee_bandages = 2 := by
sorry

#eval bandages_on_left_knee (24 - 8) 11 3

end NUMINAMATH_CALUDE_peggy_left_knee_bandages_l841_84123


namespace NUMINAMATH_CALUDE_contractor_absence_solution_l841_84168

/-- Represents the problem of calculating a contractor's absence days --/
def ContractorAbsenceProblem (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_received : ℚ) : Prop :=
  ∃ (absent_days : ℕ),
    (absent_days ≤ total_days) ∧
    (daily_pay * (total_days - absent_days : ℚ) - daily_fine * (absent_days : ℚ) = total_received)

/-- Theorem stating the solution to the contractor absence problem --/
theorem contractor_absence_solution :
  ContractorAbsenceProblem 30 25 7.5 425 → ∃ (absent_days : ℕ), absent_days = 10 := by
  sorry

#check contractor_absence_solution

end NUMINAMATH_CALUDE_contractor_absence_solution_l841_84168


namespace NUMINAMATH_CALUDE_count_pairs_equals_210_l841_84147

def count_pairs : ℕ := 
  (Finset.range 20).sum (fun a => 21 - a)

theorem count_pairs_equals_210 : count_pairs = 210 := by sorry

end NUMINAMATH_CALUDE_count_pairs_equals_210_l841_84147


namespace NUMINAMATH_CALUDE_distinct_values_count_l841_84126

def parenthesization1 : ℕ := 3^(3^(3^3))
def parenthesization2 : ℕ := 3^((3^3)^3)
def parenthesization3 : ℕ := ((3^3)^3)^3
def parenthesization4 : ℕ := 3^((3^3)^(3^2))

def distinctValues : Finset ℕ := {parenthesization1, parenthesization2, parenthesization3, parenthesization4}

theorem distinct_values_count :
  Finset.card distinctValues = 3 := by sorry

end NUMINAMATH_CALUDE_distinct_values_count_l841_84126


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l841_84199

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {1, 4}

-- Theorem statement
theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {4} :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l841_84199


namespace NUMINAMATH_CALUDE_M_equals_singleton_l841_84106

def M : Set (ℝ × ℝ) := {p | 2 * p.1 + p.2 = 2 ∧ p.1 - p.2 = 1}

theorem M_equals_singleton : M = {(1, 0)} := by sorry

end NUMINAMATH_CALUDE_M_equals_singleton_l841_84106


namespace NUMINAMATH_CALUDE_binary_addition_theorem_l841_84151

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its binary representation as a list of bits -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_addition_theorem :
  let a := [true, false, true]  -- 101₂
  let b := [true, true]         -- 11₂
  let c := [false, false, true, true]  -- 1100₂
  let d := [true, false, true, true, true]  -- 11101₂
  let result := [true, false, false, false, false, true, true]  -- 110001₂
  binary_to_decimal a + binary_to_decimal b + binary_to_decimal c + binary_to_decimal d =
  binary_to_decimal result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_theorem_l841_84151


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l841_84121

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Triangle inequality
  cos B = 2/5 →
  sin A * cos B - (2*c - cos A) * sin B = 0 →
  b = 1/2 ∧
  ∀ a' c', 0 < a' ∧ 0 < c' →
    a' + b + c' ≤ Real.sqrt 30 / 6 + 1/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l841_84121


namespace NUMINAMATH_CALUDE_initial_strawberry_weight_l841_84169

/-- The initial total weight of strawberries collected by Marco and his dad -/
def initial_total (marco_weight dad_weight lost_weight : ℕ) : ℕ :=
  marco_weight + dad_weight + lost_weight

/-- Proof that the initial total weight of strawberries is 36 pounds -/
theorem initial_strawberry_weight :
  ∀ (marco_weight dad_weight lost_weight : ℕ),
    marco_weight = 12 →
    dad_weight = 16 →
    lost_weight = 8 →
    initial_total marco_weight dad_weight lost_weight = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_strawberry_weight_l841_84169


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l841_84102

theorem complex_modulus_problem (θ : ℝ) (z : ℂ) : 
  z = (Complex.I * (Real.sin θ - Complex.I)) / Complex.I →
  Real.cos θ = 1/3 →
  Complex.abs z = Real.sqrt 17 / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l841_84102


namespace NUMINAMATH_CALUDE_largest_integer_in_sequence_l841_84165

theorem largest_integer_in_sequence (n : ℕ) (start : ℤ) (h1 : n = 40) (h2 : start = -11) :
  (start + (n - 1) : ℤ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_sequence_l841_84165


namespace NUMINAMATH_CALUDE_pentagon_ink_length_l841_84192

/-- Ink length of a regular pentagon with side length n -/
def inkLength (n : ℕ) : ℕ := 5 * n

theorem pentagon_ink_length :
  (inkLength 4 = 20) ∧
  (inkLength 9 - inkLength 8 = 5) ∧
  (inkLength 100 = 500) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_ink_length_l841_84192


namespace NUMINAMATH_CALUDE_interest_rate_approximately_three_percent_l841_84139

/-- Calculates the interest rate for the first part of a loan given the total sum,
    the amount of the second part, and the interest rate for the second part. -/
def calculate_interest_rate (total_sum second_part second_rate : ℚ) : ℚ :=
  let first_part := total_sum - second_part
  let second_interest := second_part * second_rate * 3
  second_interest / (first_part * 8)

/-- Theorem stating that under the given conditions, the interest rate
    for the first part is approximately 3%. -/
theorem interest_rate_approximately_three_percent :
  let total_sum : ℚ := 2678
  let second_part : ℚ := 1648
  let second_rate : ℚ := 5 / 100
  let calculated_rate := calculate_interest_rate total_sum second_part second_rate
  abs (calculated_rate - 3 / 100) < 1 / 1000 :=
by
  sorry

#eval calculate_interest_rate 2678 1648 (5/100)

end NUMINAMATH_CALUDE_interest_rate_approximately_three_percent_l841_84139


namespace NUMINAMATH_CALUDE_people_in_house_l841_84137

theorem people_in_house : 
  ∀ (initial_bedroom : ℕ) (entering_bedroom : ℕ) (living_room : ℕ),
    initial_bedroom = 2 →
    entering_bedroom = 5 →
    living_room = 8 →
    initial_bedroom + entering_bedroom + living_room = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_people_in_house_l841_84137


namespace NUMINAMATH_CALUDE_fraction_inequality_l841_84140

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  a / d > b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l841_84140


namespace NUMINAMATH_CALUDE_division_remainder_l841_84189

theorem division_remainder (N : ℕ) : 
  (∃ R : ℕ, N = 44 * 432 + R) ∧ 
  (∃ Q : ℕ, N = 39 * Q + 15) → 
  ∃ Q' : ℕ, N = 44 * Q' + 0 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l841_84189


namespace NUMINAMATH_CALUDE_expression_simplification_l841_84144

theorem expression_simplification (x y : ℚ) 
  (hx : x = -1/2) (hy : y = -3) : 
  3 * (x^2 - 2*x*y) - (3*x^2 - 2*y + 2*(x*y + y)) = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l841_84144


namespace NUMINAMATH_CALUDE_journey_distance_l841_84127

theorem journey_distance (speed1 speed2 time1 total_time : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 70)
  (h3 : time1 = 3.2)
  (h4 : total_time = 8) :
  speed1 * time1 + speed2 * (total_time - time1) = 400 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l841_84127


namespace NUMINAMATH_CALUDE_union_of_sets_l841_84104

theorem union_of_sets : 
  let A : Set ℕ := {1, 3, 7, 8}
  let B : Set ℕ := {1, 5, 8}
  A ∪ B = {1, 3, 5, 7, 8} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l841_84104


namespace NUMINAMATH_CALUDE_inequality_proof_l841_84146

theorem inequality_proof (x y z : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l841_84146


namespace NUMINAMATH_CALUDE_rational_point_coloring_l841_84188

/-- A coloring function for rational points in the plane -/
def coloringFunction (n : ℕ) (p : ℚ × ℚ) : Fin n :=
  sorry

/-- A predicate to check if a point is on a line segment -/
def isOnLineSegment (p q r : ℚ × ℚ) : Prop :=
  sorry

theorem rational_point_coloring (n : ℕ) (hn : n > 0) :
  ∃ (f : ℚ × ℚ → Fin n),
    ∀ (p q : ℚ × ℚ) (c : Fin n),
      ∃ (r : ℚ × ℚ), isOnLineSegment p q r ∧ f r = c :=
sorry

end NUMINAMATH_CALUDE_rational_point_coloring_l841_84188


namespace NUMINAMATH_CALUDE_monotonic_decreasing_range_l841_84164

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (2 * k - 1) * x + 1

-- State the theorem
theorem monotonic_decreasing_range (k : ℝ) :
  (∀ x y : ℝ, x < y → f k x > f k y) →
  k < (1 / 2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_range_l841_84164


namespace NUMINAMATH_CALUDE_unique_operation_equals_one_l841_84138

theorem unique_operation_equals_one :
  ((-3 + (-3) = 1) = False) ∧
  ((-3 - (-3) = 1) = False) ∧
  ((-3 / (-3) = 1) = True) ∧
  ((-3 * (-3) = 1) = False) := by
  sorry

end NUMINAMATH_CALUDE_unique_operation_equals_one_l841_84138


namespace NUMINAMATH_CALUDE_point_translation_point_translation_proof_l841_84110

/-- Given a point B with coordinates (-5, 1), moving it 4 units right and 2 units up
    results in a point B' with coordinates (-1, 3). -/
theorem point_translation : ℝ × ℝ → ℝ × ℝ → Prop :=
  fun B B' => B = (-5, 1) → B' = (B.1 + 4, B.2 + 2) → B' = (-1, 3)

/-- The proof of the theorem. -/
theorem point_translation_proof : point_translation (-5, 1) (-1, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_translation_point_translation_proof_l841_84110


namespace NUMINAMATH_CALUDE_tysons_ocean_speed_l841_84108

/-- Tyson's swimming speed problem -/
theorem tysons_ocean_speed (lake_speed : ℝ) (total_races : ℕ) (race_distance : ℝ) (total_time : ℝ) :
  lake_speed = 3 →
  total_races = 10 →
  race_distance = 3 →
  total_time = 11 →
  ∃ (ocean_speed : ℝ),
    ocean_speed = 2.5 ∧
    (lake_speed * (total_races / 2 * race_distance) + ocean_speed * (total_races / 2 * race_distance)) / total_races = race_distance / (total_time / total_races) :=
by sorry

end NUMINAMATH_CALUDE_tysons_ocean_speed_l841_84108


namespace NUMINAMATH_CALUDE_smallest_multiple_of_3_5_7_9_l841_84172

theorem smallest_multiple_of_3_5_7_9 (n : ℕ) :
  (∀ m : ℕ, m > 0 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 9 ∣ m → n ≤ m) ↔ n = 315 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_3_5_7_9_l841_84172


namespace NUMINAMATH_CALUDE_fourth_number_12th_row_l841_84116

/-- Given a number pattern where each row has 8 numbers, and the last number of each row is 8 times the row number, this function calculates the nth number in the mth row. -/
def patternNumber (m n : ℕ) : ℕ :=
  8 * (m - 1) + n

/-- Theorem stating that the fourth number in the 12th row of the described pattern is 92. -/
theorem fourth_number_12th_row : patternNumber 12 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_12th_row_l841_84116


namespace NUMINAMATH_CALUDE_cistern_leak_time_l841_84128

/-- Represents the cistern problem -/
def CisternProblem (capacity : ℝ) (tapRate : ℝ) (timeWithTap : ℝ) : Prop :=
  let leakRate := capacity / timeWithTap + tapRate
  let timeWithoutTap := capacity / leakRate
  timeWithoutTap = 20

/-- Theorem stating the solution to the cistern problem -/
theorem cistern_leak_time :
  CisternProblem 480 4 24 := by sorry

end NUMINAMATH_CALUDE_cistern_leak_time_l841_84128


namespace NUMINAMATH_CALUDE_fifteen_points_max_planes_l841_84149

def max_planes (n : ℕ) : ℕ := n.choose 3

theorem fifteen_points_max_planes :
  max_planes 15 = 455 :=
by sorry

end NUMINAMATH_CALUDE_fifteen_points_max_planes_l841_84149


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l841_84109

theorem least_k_for_inequality (k : ℤ) : 
  (∀ n : ℤ, n < k → (0.0010101 : ℝ) * (10 : ℝ) ^ (n : ℝ) ≤ 100) ∧ 
  (0.0010101 : ℝ) * (10 : ℝ) ^ (k : ℝ) > 100 → 
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l841_84109


namespace NUMINAMATH_CALUDE_no_distinct_natural_power_sum_equality_l841_84181

theorem no_distinct_natural_power_sum_equality :
  ∀ (x y z t : ℕ),
    x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t →
    x^x + y^y ≠ z^z + t^t :=
by
  sorry

end NUMINAMATH_CALUDE_no_distinct_natural_power_sum_equality_l841_84181


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_ab_product_l841_84182

-- Define the ellipse and hyperbola equations
def ellipse_equation (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def hyperbola_equation (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the theorem
theorem ellipse_hyperbola_ab_product 
  (a b : ℝ) 
  (h_ellipse : ∃ (x y : ℝ), ellipse_equation x y a b ∧ (x = 0 ∧ y = 5 ∨ x = 0 ∧ y = -5))
  (h_hyperbola : ∃ (x y : ℝ), hyperbola_equation x y a b ∧ (x = 7 ∧ y = 0 ∨ x = -7 ∧ y = 0)) :
  |a * b| = 2 * Real.sqrt 111 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_ab_product_l841_84182


namespace NUMINAMATH_CALUDE_can_determine_contents_l841_84177

/-- Represents the possible contents of a box -/
inductive BoxContent
  | Red
  | White
  | Mixed

/-- Represents a box with a label and actual content -/
structure Box where
  label : BoxContent
  content : BoxContent

/-- The result of opening a box and drawing a ball -/
inductive DrawResult
  | Red
  | White

/-- Represents the state of the puzzle -/
structure PuzzleState where
  boxes : Fin 3 → Box
  all_labels_incorrect : ∀ i, (boxes i).label ≠ (boxes i).content
  contents_distinct : ∀ i j, i ≠ j → (boxes i).content ≠ (boxes j).content

/-- Function to determine the contents of all boxes based on the draw result -/
def determineContents (state : PuzzleState) (draw : DrawResult) : Fin 3 → BoxContent :=
  sorry

theorem can_determine_contents (state : PuzzleState) :
  ∃ (i : Fin 3) (draw : DrawResult),
    determineContents state draw = λ j => (state.boxes j).content :=
  sorry

end NUMINAMATH_CALUDE_can_determine_contents_l841_84177


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l841_84163

theorem quadratic_coefficient (a : ℝ) : 
  (a * (1/2)^2 + 9 * (1/2) - 5 = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l841_84163


namespace NUMINAMATH_CALUDE_amount_A_to_B_plus_ratio_l841_84197

/-- The amount promised for a B+ grade -/
def amount_B_plus : ℝ := 5

/-- The number of courses in Paul's scorecard -/
def num_courses : ℕ := 10

/-- The flat amount received for each A+ grade -/
def amount_A_plus : ℝ := 15

/-- The maximum amount Paul could receive -/
def max_amount : ℝ := 190

/-- The amount promised for an A grade -/
noncomputable def amount_A : ℝ := 
  (max_amount - 2 * amount_A_plus) / (2 * (num_courses - 2))

/-- Theorem stating that the ratio of amount promised for an A to a B+ is 2:1 -/
theorem amount_A_to_B_plus_ratio : 
  amount_A / amount_B_plus = 2 := by sorry

end NUMINAMATH_CALUDE_amount_A_to_B_plus_ratio_l841_84197


namespace NUMINAMATH_CALUDE_distance_and_angle_from_origin_l841_84174

/-- In a rectangular coordinate system, for a point (12, 5): -/
theorem distance_and_angle_from_origin :
  let x : ℝ := 12
  let y : ℝ := 5
  let distance := Real.sqrt (x^2 + y^2)
  let angle := Real.arctan (y / x)
  (distance = 13 ∧ angle = Real.arctan (5 / 12)) := by
  sorry

end NUMINAMATH_CALUDE_distance_and_angle_from_origin_l841_84174


namespace NUMINAMATH_CALUDE_gcd_of_specific_squares_l841_84191

theorem gcd_of_specific_squares : Nat.gcd (130^2 + 240^2 + 350^2) (131^2 + 241^2 + 349^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_squares_l841_84191


namespace NUMINAMATH_CALUDE_probability_inner_circle_l841_84105

theorem probability_inner_circle (R : ℝ) (r : ℝ) (h1 : R = 6) (h2 : r = 2) :
  (π * r^2) / (π * R^2) = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_inner_circle_l841_84105


namespace NUMINAMATH_CALUDE_investment_percentage_l841_84135

/-- Given two investments with a total of $2000, where $600 is invested at 8%,
    and the annual income from the first investment exceeds the second by $92,
    prove that the percentage of the first investment is 10%. -/
theorem investment_percentage : 
  ∀ (total_investment first_investment_amount first_investment_rate : ℝ),
  total_investment = 2000 →
  first_investment_amount = 1400 →
  first_investment_rate * first_investment_amount - 0.08 * 600 = 92 →
  first_investment_rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_l841_84135


namespace NUMINAMATH_CALUDE_frank_defeated_six_enemies_l841_84145

/-- The number of enemies Frank defeated in the game --/
def enemies_defeated : ℕ := sorry

/-- The points earned per enemy defeated --/
def points_per_enemy : ℕ := 9

/-- The bonus points for completing the level --/
def bonus_points : ℕ := 8

/-- The total points Frank earned --/
def total_points : ℕ := 62

/-- Theorem stating that Frank defeated 6 enemies --/
theorem frank_defeated_six_enemies :
  enemies_defeated = 6 ∧
  enemies_defeated * points_per_enemy + bonus_points = total_points :=
sorry

end NUMINAMATH_CALUDE_frank_defeated_six_enemies_l841_84145


namespace NUMINAMATH_CALUDE_min_time_35_minutes_l841_84115

/-- Represents a rectangular parallelepiped -/
structure Brick where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a point moving on the surface of a brick -/
structure MovingPoint where
  v_x : ℝ → ℝ
  v_y : ℝ → ℝ
  velocity_constraint : ∀ t, (v_x t)^2 + 4*(v_y t)^2 = 1

/-- The minimum time for a point to travel from one vertex of the lower base
    to the opposite vertex of the upper base of a brick -/
def min_travel_time (b : Brick) (p : MovingPoint) : ℝ := sorry

/-- The theorem stating the minimum travel time for the given problem -/
theorem min_time_35_minutes (b : Brick) (p : MovingPoint)
    (h1 : b.length = 28)
    (h2 : b.width = 9)
    (h3 : b.height = 6) :
  min_travel_time b p = 35 := by sorry

end NUMINAMATH_CALUDE_min_time_35_minutes_l841_84115


namespace NUMINAMATH_CALUDE_linear_system_solution_l841_84176

theorem linear_system_solution (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h₁ : a₁ * ((c₁ * b₂ - c₂ * b₁) / (a₁ * b₂ - a₂ * b₁)) + b₁ * y = c₁)
  (h₂ : a₂ * ((c₁ * b₂ - c₂ * b₁) / (a₁ * b₂ - a₂ * b₁)) + b₂ * y = c₂)
  (h₃ : a₁ * b₂ ≠ a₂ * b₁) :
  y = (c₁ * a₂ - c₂ * a₁) / (b₁ * a₂ - b₂ * a₁) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l841_84176


namespace NUMINAMATH_CALUDE_constant_term_is_integer_coefficients_not_necessarily_integer_l841_84186

/-- A real quadratic polynomial that takes integer values for all integer inputs -/
structure IntegerValuedQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  integer_valued : ∀ (x : ℤ), ∃ (y : ℤ), a * x^2 + b * x + c = y

theorem constant_term_is_integer (p : IntegerValuedQuadratic) : ∃ (n : ℤ), p.c = n := by
  sorry

theorem coefficients_not_necessarily_integer : 
  ∃ (p : IntegerValuedQuadratic), ¬(∃ (m n : ℤ), p.a = m ∧ p.b = n) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_integer_coefficients_not_necessarily_integer_l841_84186


namespace NUMINAMATH_CALUDE_combinatorial_identities_l841_84184

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of k-permutations of n -/
def permutation (n k : ℕ) : ℕ := sorry

theorem combinatorial_identities :
  (∀ n k : ℕ, k > 0 → k * binomial n k = n * binomial (n - 1) (k - 1)) ∧
  binomial 2014 2013 + permutation 5 3 = 2074 := by sorry

end NUMINAMATH_CALUDE_combinatorial_identities_l841_84184


namespace NUMINAMATH_CALUDE_range_of_a_l841_84175

/-- A quadratic function y = x^2 + 2(a-1)x + 2 -/
def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The function decreases monotonically on (-∞, 4] -/
def decreases_on_left (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ ∧ x₂ ≤ 4 → f a x₁ ≥ f a x₂

/-- The function increases monotonically on [5, +∞) -/
def increases_on_right (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 5 ≤ x₁ ∧ x₁ ≤ x₂ → f a x₁ ≤ f a x₂

/-- The range of a given the monotonicity conditions -/
theorem range_of_a (a : ℝ) 
  (h1 : decreases_on_left a) 
  (h2 : increases_on_right a) : 
  -4 ≤ a ∧ a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l841_84175


namespace NUMINAMATH_CALUDE_tree_height_problem_l841_84155

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 16 →  -- One tree is 16 feet taller than the other
  h₂ / h₁ = 3 / 4 →  -- The heights are in the ratio 3:4
  h₁ = 64 :=  -- The taller tree is 64 feet tall
by
  sorry

end NUMINAMATH_CALUDE_tree_height_problem_l841_84155


namespace NUMINAMATH_CALUDE_position_of_2008_l841_84113

/-- Define the position of a number in the pattern -/
structure Position where
  row : Nat
  column : Nat

/-- Function to calculate the position of a number in the pattern -/
noncomputable def calculatePosition (n : Nat) : Position :=
  sorry  -- The actual implementation would go here

/-- Theorem stating that 2008 is in row 18, column 45 -/
theorem position_of_2008 : calculatePosition 2008 = ⟨18, 45⟩ := by
  sorry

#check position_of_2008

end NUMINAMATH_CALUDE_position_of_2008_l841_84113


namespace NUMINAMATH_CALUDE_min_value_sum_l841_84167

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b^2)) + (b / (4 * c^3)) + (c / (5 * a^4)) ≥ 1 ∧
  ((a / (3 * b^2)) + (b / (4 * c^3)) + (c / (5 * a^4)) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l841_84167


namespace NUMINAMATH_CALUDE_oil_tank_capacity_l841_84112

theorem oil_tank_capacity (t : ℝ) (h1 : t > 0) : 
  (1/4 : ℝ) * t + 6 = (1/3 : ℝ) * t → t = 72 := by
  sorry

end NUMINAMATH_CALUDE_oil_tank_capacity_l841_84112


namespace NUMINAMATH_CALUDE_isosceles_triangle_l841_84131

theorem isosceles_triangle (A B C : ℝ) (h_sum : A + B + C = π) :
  let f := fun x : ℝ => x^2 - x * Real.cos A * Real.cos B + 2 * Real.sin (C/2)^2
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = (1/2) * x₁ * x₂ → A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l841_84131


namespace NUMINAMATH_CALUDE_mary_found_two_seashells_l841_84190

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 7 - 5

/-- The number of seashells Keith found -/
def keith_seashells : ℕ := 5

/-- The total number of seashells Mary and Keith found together -/
def total_seashells : ℕ := 7

theorem mary_found_two_seashells :
  mary_seashells = 2 :=
sorry

end NUMINAMATH_CALUDE_mary_found_two_seashells_l841_84190


namespace NUMINAMATH_CALUDE_sequence_inequality_l841_84100

-- Define the sequence a_n
def a (n k : ℤ) : ℝ := |n - k| + |n + 2*k|

-- State the theorem
theorem sequence_inequality (k : ℤ) :
  (∀ n : ℕ, a n k ≥ a 3 k) ∧ (a 3 k = a 4 k) →
  k ≤ -2 ∨ k ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l841_84100


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_count_l841_84195

def p (x : ℤ) : ℤ := 4*x^4 - 12*x^3 + 17*x^2 - 6*x - 14

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

theorem polynomial_perfect_square_count :
  ∃! (S : Finset ℤ), S.card = 2 ∧ ∀ x : ℤ, x ∈ S ↔ is_perfect_square (p x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_count_l841_84195


namespace NUMINAMATH_CALUDE_function_equality_l841_84183

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x ≤ x) 
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_equality_l841_84183
