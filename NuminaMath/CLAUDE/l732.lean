import Mathlib

namespace NUMINAMATH_CALUDE_men_to_women_ratio_l732_73277

/-- Proves that the ratio of men to women is 2:1 given the average heights -/
theorem men_to_women_ratio (M W : ℕ) (h_total : M * 185 + W * 170 = (M + W) * 180) :
  M / W = 2 / 1 := by
  sorry

#check men_to_women_ratio

end NUMINAMATH_CALUDE_men_to_women_ratio_l732_73277


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l732_73257

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |4 - 3*x| - 5 ≤ 0} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l732_73257


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l732_73213

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = 7 ∧
  ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1) →
  ((3/4) * x^2 + 2*x - y^2 ≤ M) :=
sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l732_73213


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l732_73211

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The foci of the ellipse are on the x-axis -/
  foci_on_x_axis : Bool
  /-- The center of the ellipse is at the origin -/
  center_at_origin : Bool
  /-- The four vertices of a square with side length 2 are on the minor axis and coincide with the foci -/
  square_vertices_on_minor_axis : Bool
  /-- The side length of the square is 2 -/
  square_side_length : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the special ellipse -/
theorem special_ellipse_equation (E : SpecialEllipse) (x y : ℝ) :
  E.foci_on_x_axis ∧
  E.center_at_origin ∧
  E.square_vertices_on_minor_axis ∧
  E.square_side_length = 2 →
  standard_equation 4 2 x y :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l732_73211


namespace NUMINAMATH_CALUDE_exists_valid_road_configuration_l732_73260

/-- A configuration of roads connecting four villages at the vertices of a square -/
structure RoadConfiguration where
  /-- The side length of the square -/
  side_length : ℝ
  /-- The total length of roads in the configuration -/
  total_length : ℝ
  /-- Ensure that all villages are connected -/
  all_connected : Bool

/-- Theorem stating that there exists a valid road configuration with total length less than 5.5 km -/
theorem exists_valid_road_configuration :
  ∃ (config : RoadConfiguration),
    config.side_length = 2 ∧
    config.all_connected = true ∧
    config.total_length < 5.5 := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_road_configuration_l732_73260


namespace NUMINAMATH_CALUDE_find_a1_l732_73290

def recurrence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n / (2 * a n + 1)

theorem find_a1 (a : ℕ → ℚ) (h1 : recurrence a) (h2 : a 3 = 1/5) :
  a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a1_l732_73290


namespace NUMINAMATH_CALUDE_anthony_lunch_money_l732_73299

theorem anthony_lunch_money (initial_money juice_cost cupcake_cost : ℕ) 
  (h1 : initial_money = 75)
  (h2 : juice_cost = 27)
  (h3 : cupcake_cost = 40) :
  initial_money - (juice_cost + cupcake_cost) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_anthony_lunch_money_l732_73299


namespace NUMINAMATH_CALUDE_identity_matrix_solution_l732_73285

def matrix_equation (N : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  N^4 - 3 • N^3 + 3 • N^2 - N = !![5, 15; 0, 5]

theorem identity_matrix_solution :
  ∃! N : Matrix (Fin 2) (Fin 2) ℝ, matrix_equation N ∧ N = 1 := by sorry

end NUMINAMATH_CALUDE_identity_matrix_solution_l732_73285


namespace NUMINAMATH_CALUDE_punch_mixture_theorem_l732_73234

/-- Given a 2-liter mixture that is 15% fruit juice, adding 0.125 liters of pure fruit juice
    results in a new mixture that is 20% fruit juice. -/
theorem punch_mixture_theorem :
  let initial_volume : ℝ := 2
  let initial_concentration : ℝ := 0.15
  let added_juice : ℝ := 0.125
  let target_concentration : ℝ := 0.20
  let final_volume : ℝ := initial_volume + added_juice
  let final_juice_amount : ℝ := initial_volume * initial_concentration + added_juice
  final_juice_amount / final_volume = target_concentration := by
sorry


end NUMINAMATH_CALUDE_punch_mixture_theorem_l732_73234


namespace NUMINAMATH_CALUDE_rationalize_denominator_l732_73202

theorem rationalize_denominator : 
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 81 (1/3)) = (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l732_73202


namespace NUMINAMATH_CALUDE_grandmother_five_times_lingling_age_l732_73262

/-- Represents the current age of Lingling -/
def lingling_age : ℕ := 8

/-- Represents the current age of the grandmother -/
def grandmother_age : ℕ := 60

/-- Represents the number of years after which the grandmother's age will be 5 times Lingling's age -/
def years_until_five_times : ℕ := 5

/-- Proves that after 'years_until_five_times' years, the grandmother's age will be 5 times Lingling's age -/
theorem grandmother_five_times_lingling_age : 
  grandmother_age + years_until_five_times = 5 * (lingling_age + years_until_five_times) := by
  sorry

end NUMINAMATH_CALUDE_grandmother_five_times_lingling_age_l732_73262


namespace NUMINAMATH_CALUDE_equation_solutions_l732_73215

theorem equation_solutions : 
  (∃ x1 x2 : ℝ, x1 = 4 ∧ x2 = -1 ∧ x1^2 - 3*x1 - 4 = 0 ∧ x2^2 - 3*x2 - 4 = 0) ∧
  (∃ y1 y2 : ℝ, y1 = 1 + Real.sqrt 2 ∧ y2 = 1 - Real.sqrt 2 ∧ y1*(y1-2) = 1 ∧ y2*(y2-2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l732_73215


namespace NUMINAMATH_CALUDE_minimum_weights_l732_73253

def is_valid_weight_set (weights : List ℕ) : Prop :=
  ∀ w : ℕ, 1 ≤ w ∧ w ≤ 20 →
    ∃ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ (w = a ∨ w = a + b)

theorem minimum_weights :
  ∃ (weights : List ℕ),
    weights.length = 6 ∧
    is_valid_weight_set weights ∧
    ∀ (other_weights : List ℕ),
      is_valid_weight_set other_weights →
      other_weights.length ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_minimum_weights_l732_73253


namespace NUMINAMATH_CALUDE_dorothy_sea_glass_count_l732_73200

-- Define the sea glass counts for Blanche and Rose
def blanche_green : ℕ := 12
def blanche_red : ℕ := 3
def rose_red : ℕ := 9
def rose_blue : ℕ := 11

-- Define Dorothy's sea glass counts based on the conditions
def dorothy_red : ℕ := 2 * (blanche_red + rose_red)
def dorothy_blue : ℕ := 3 * rose_blue

-- Define Dorothy's total sea glass count
def dorothy_total : ℕ := dorothy_red + dorothy_blue

-- Theorem to prove
theorem dorothy_sea_glass_count : dorothy_total = 57 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_sea_glass_count_l732_73200


namespace NUMINAMATH_CALUDE_asymptote_sum_l732_73288

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 2 ∧ x ≠ 3 → 
    (x^3 + A*x^2 + B*x + C ≠ 0)) →
  ((x + 1) * (x - 2) * (x - 3) = x^3 + A*x^2 + B*x + C) →
  A + B + C = -5 := by
sorry

end NUMINAMATH_CALUDE_asymptote_sum_l732_73288


namespace NUMINAMATH_CALUDE_num_employees_correct_l732_73206

/-- The number of employees in an organization, excluding the manager -/
def num_employees : ℕ := 15

/-- The average monthly salary of employees, excluding the manager -/
def avg_salary : ℕ := 1800

/-- The increase in average salary when the manager's salary is added -/
def avg_increase : ℕ := 150

/-- The manager's monthly salary -/
def manager_salary : ℕ := 4200

/-- Theorem stating that the number of employees is correct given the conditions -/
theorem num_employees_correct :
  (avg_salary * num_employees + manager_salary) / (num_employees + 1) = avg_salary + avg_increase :=
by sorry

end NUMINAMATH_CALUDE_num_employees_correct_l732_73206


namespace NUMINAMATH_CALUDE_card_number_sum_l732_73229

theorem card_number_sum (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
  sorry

end NUMINAMATH_CALUDE_card_number_sum_l732_73229


namespace NUMINAMATH_CALUDE_walk_distance_proof_l732_73226

/-- Given a constant walking speed and time, calculates the distance walked. -/
def distance_walked (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that walking at 4 miles per hour for 2 hours results in a distance of 8 miles. -/
theorem walk_distance_proof :
  let speed : ℝ := 4
  let time : ℝ := 2
  distance_walked speed time = 8 := by
sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l732_73226


namespace NUMINAMATH_CALUDE_julia_stairs_difference_l732_73223

theorem julia_stairs_difference (jonny_stairs julia_stairs : ℕ) : 
  jonny_stairs = 1269 →
  julia_stairs < jonny_stairs / 3 →
  jonny_stairs + julia_stairs = 1685 →
  (jonny_stairs / 3) - julia_stairs = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_julia_stairs_difference_l732_73223


namespace NUMINAMATH_CALUDE_min_value_a_l732_73218

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, |y + 4| - |y| ≤ 2^x + a / (2^x)) → 
  a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l732_73218


namespace NUMINAMATH_CALUDE_rectangle_max_area_l732_73209

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) →
  l * w = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l732_73209


namespace NUMINAMATH_CALUDE_angle_E_measure_l732_73292

structure Parallelogram where
  E : Real
  F : Real
  G : Real
  H : Real

def external_angle (p : Parallelogram) : Real := 50

theorem angle_E_measure (p : Parallelogram) :
  external_angle p = 50 → p.E = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_E_measure_l732_73292


namespace NUMINAMATH_CALUDE_soda_price_theorem_l732_73280

/-- Calculates the price of soda cans given specific discount conditions -/
def sodaPrice (regularPrice : ℝ) (caseDiscount : ℝ) (bulkDiscount : ℝ) (caseSize : ℕ) (numCans : ℕ) : ℝ :=
  let discountedPrice := regularPrice * (1 - caseDiscount)
  let fullCases := numCans / caseSize
  let remainingCans := numCans % caseSize
  let fullCasePrice := if fullCases ≥ 3
                       then (fullCases * caseSize * discountedPrice) * (1 - bulkDiscount)
                       else fullCases * caseSize * discountedPrice
  let remainingPrice := remainingCans * discountedPrice
  fullCasePrice + remainingPrice

/-- The price of 70 cans of soda under given discount conditions is $26.895 -/
theorem soda_price_theorem :
  sodaPrice 0.55 0.25 0.10 24 70 = 26.895 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_theorem_l732_73280


namespace NUMINAMATH_CALUDE_water_bucket_problem_l732_73252

theorem water_bucket_problem (a b : ℝ) : 
  (a - 6 = (1/3) * (b + 6)) →
  (b - 6 = (1/2) * (a + 6)) →
  a = 13.2 := by
  sorry

end NUMINAMATH_CALUDE_water_bucket_problem_l732_73252


namespace NUMINAMATH_CALUDE_other_number_proof_l732_73263

theorem other_number_proof (A B : ℕ+) (hcf lcm : ℕ+) : 
  hcf = 12 →
  lcm = 396 →
  A = 48 →
  Nat.gcd A.val B.val = hcf.val →
  Nat.lcm A.val B.val = lcm.val →
  B = 99 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l732_73263


namespace NUMINAMATH_CALUDE_symmetric_point_sum_l732_73242

/-- A point is symmetric to the line x+y+1=0 if its symmetric point is also on this line -/
def is_symmetric_point (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), x + y + 1 = 0 ∧ (a + x) / 2 + (b + y) / 2 + 1 = 0

/-- Theorem: If a point (a,b) is symmetric to the line x+y+1=0 and its symmetric point
    is also on this line, then a+b=-1 -/
theorem symmetric_point_sum (a b : ℝ) (h : is_symmetric_point a b) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_sum_l732_73242


namespace NUMINAMATH_CALUDE_additional_people_needed_l732_73254

/-- Represents the number of person-hours required to mow a lawn -/
def lawn_work : ℕ := 24

/-- The number of people who can mow the lawn in 3 hours -/
def initial_people : ℕ := 8

/-- The initial time taken to mow the lawn -/
def initial_time : ℕ := 3

/-- The desired time to mow the lawn -/
def target_time : ℕ := 2

theorem additional_people_needed : 
  ∃ (additional : ℕ), 
    initial_people * initial_time = lawn_work ∧
    (initial_people + additional) * target_time = lawn_work ∧
    additional = 4 :=
by sorry

end NUMINAMATH_CALUDE_additional_people_needed_l732_73254


namespace NUMINAMATH_CALUDE_volume_of_sphere_with_radius_three_l732_73212

/-- The volume of a sphere with radius 3 is 36π. -/
theorem volume_of_sphere_with_radius_three : 
  (4 / 3 : ℝ) * Real.pi * 3^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_of_sphere_with_radius_three_l732_73212


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l732_73294

-- Define the polynomial and the divisor
def f (x : ℝ) : ℝ := x^6 - x^5 - x^4 + x^3 + x^2
def divisor (x : ℝ) : ℝ := (x^2 - 1) * (x - 2)

-- Define the remainder
def remainder (x : ℝ) : ℝ := 9 * x^2 - 8

-- Theorem statement
theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = divisor x * q x + remainder x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l732_73294


namespace NUMINAMATH_CALUDE_youngest_member_age_l732_73243

theorem youngest_member_age (n : ℕ) (current_avg : ℚ) (birth_avg : ℚ) 
  (h1 : n = 5)
  (h2 : current_avg = 20)
  (h3 : birth_avg = 25/2) :
  (n : ℚ) * current_avg - (n - 1 : ℚ) * birth_avg = 10 := by
  sorry

end NUMINAMATH_CALUDE_youngest_member_age_l732_73243


namespace NUMINAMATH_CALUDE_max_cities_in_network_l732_73286

/-- Represents a city in the airline network -/
structure City where
  id : Nat

/-- Represents the airline network -/
structure AirlineNetwork where
  cities : Finset City
  connections : City → Finset City

/-- The maximum number of direct connections a city can have -/
def maxDirectConnections : Nat := 3

/-- Defines a valid airline network based on the given conditions -/
def isValidNetwork (network : AirlineNetwork) : Prop :=
  ∀ c ∈ network.cities,
    (network.connections c).card ≤ maxDirectConnections ∧
    ∀ d ∈ network.cities, 
      c ≠ d → (d ∈ network.connections c ∨ 
               ∃ e ∈ network.cities, e ∈ network.connections c ∧ d ∈ network.connections e)

/-- The theorem stating the maximum number of cities in a valid network -/
theorem max_cities_in_network (network : AirlineNetwork) 
  (h : isValidNetwork network) : network.cities.card ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_cities_in_network_l732_73286


namespace NUMINAMATH_CALUDE_largest_number_divisible_by_88_has_4_digits_l732_73296

theorem largest_number_divisible_by_88_has_4_digits :
  let n : ℕ := 9944
  (∀ m : ℕ, m > n → m % 88 ≠ 0 ∨ (String.length (toString m) > String.length (toString n))) →
  n % 88 = 0 →
  String.length (toString n) = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_divisible_by_88_has_4_digits_l732_73296


namespace NUMINAMATH_CALUDE_hassan_apple_trees_count_l732_73201

/-- The number of apple trees Hassan has -/
def hassan_apple_trees : ℕ := 1

/-- The number of orange trees Ahmed has -/
def ahmed_orange_trees : ℕ := 8

/-- The number of orange trees Hassan has -/
def hassan_orange_trees : ℕ := 2

/-- The number of apple trees Ahmed has -/
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees

/-- The total number of trees in Ahmed's orchard -/
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees

/-- The total number of trees in Hassan's orchard -/
def hassan_total_trees : ℕ := hassan_orange_trees + hassan_apple_trees

theorem hassan_apple_trees_count :
  hassan_apple_trees = 1 ∧
  ahmed_orange_trees = 8 ∧
  hassan_orange_trees = 2 ∧
  ahmed_apple_trees = 4 * hassan_apple_trees ∧
  ahmed_total_trees = hassan_total_trees + 9 := by
  sorry

end NUMINAMATH_CALUDE_hassan_apple_trees_count_l732_73201


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l732_73276

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2 = 1

-- Define the circle ⊙G
def circle_G (x y r : ℝ) : Prop := (x-2)^2 + y^2 = r^2

-- Define the incircle property
def is_incircle (r : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse C.1 C.2 ∧
    circle_G 2 0 r ∧
    A.1 = -4 -- Left vertex of ellipse

-- Define the tangent line EF
def line_EF (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b

-- Define the tangency condition
def is_tangent (m b r : ℝ) : Prop :=
  abs (m*2 - b) / Real.sqrt (1 + m^2) = r

-- State the theorem
theorem circle_and_tangent_line :
  ∀ r : ℝ,
  is_incircle r →
  r = 2/3 ∧
  ∃ m b : ℝ,
    line_EF m b 0 1 ∧  -- Line passes through M(0,1)
    is_tangent m b r   -- Line is tangent to ⊙G
:= by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l732_73276


namespace NUMINAMATH_CALUDE_count_six_digit_numbers_with_seven_l732_73205

def digits : Finset ℕ := Finset.range 10

def multiset_count (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) k

def six_digit_numbers_with_seven : ℕ :=
  (multiset_count 9 5) +
  (multiset_count 9 4) +
  (multiset_count 9 3) +
  (multiset_count 9 2) +
  (multiset_count 9 1) +
  (multiset_count 9 0)

theorem count_six_digit_numbers_with_seven :
  six_digit_numbers_with_seven = 2002 := by sorry

end NUMINAMATH_CALUDE_count_six_digit_numbers_with_seven_l732_73205


namespace NUMINAMATH_CALUDE_possible_a_values_l732_73241

theorem possible_a_values :
  ∀ (a : ℤ), 
    (∃ (b c : ℤ), ∀ (x : ℤ), (x - a) * (x - 15) + 4 = (x + b) * (x + c)) ↔ 
    (a = 16 ∨ a = 21) := by
sorry

end NUMINAMATH_CALUDE_possible_a_values_l732_73241


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l732_73279

/-- An isosceles triangle with given altitude and perimeter has area 75 -/
theorem isosceles_triangle_area (b s : ℝ) : 
  b > 0 → s > 0 → 
  2 * s + 2 * b = 40 → -- perimeter condition
  b ^ 2 + 10 ^ 2 = s ^ 2 → -- Pythagorean theorem
  (2 * b) * 10 / 2 = 75 := by 
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l732_73279


namespace NUMINAMATH_CALUDE_perpendicular_angle_values_l732_73289

theorem perpendicular_angle_values (α : Real) : 
  (4 * Real.pi < α ∧ α < 6 * Real.pi) →
  (∃ k : ℤ, α = -Real.pi/6 + k * Real.pi) →
  (α = 29 * Real.pi / 6 ∨ α = 35 * Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_angle_values_l732_73289


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l732_73259

-- Define the circles
def circle1_center : ℝ × ℝ := (3, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (8, 0)
def circle2_radius : ℝ := 2

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := sorry

-- Define the property of being tangent to a circle in the fourth quadrant
def is_tangent_in_fourth_quadrant (line : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : Prop := sorry

-- Theorem statement
theorem tangent_y_intercept :
  is_tangent_in_fourth_quadrant tangent_line circle1_center circle1_radius ∧
  is_tangent_in_fourth_quadrant tangent_line circle2_center circle2_radius →
  ∃ (y : ℝ), y = 6/5 ∧ (0, y) ∈ tangent_line :=
sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l732_73259


namespace NUMINAMATH_CALUDE_hyperbrick_probability_l732_73244

open Set
open Real
open Finset

-- Define the set of numbers
def S : Finset ℕ := Finset.range 500

-- Define the type for our 9 randomly selected numbers
structure NineNumbers :=
  (numbers : Finset ℕ)
  (size_eq : numbers.card = 9)
  (subset_S : numbers ⊆ S)

-- Define the probability function
def probability (n : NineNumbers) : ℚ :=
  -- Implementation details omitted
  sorry

-- The main theorem
theorem hyperbrick_probability :
  ∀ n : NineNumbers, probability n = 16 / 63 :=
sorry

end NUMINAMATH_CALUDE_hyperbrick_probability_l732_73244


namespace NUMINAMATH_CALUDE_range_of_a_l732_73273

theorem range_of_a (x a : ℝ) : 
  (∀ x, (a ≤ x ∧ x < a + 2) → (|x| ≠ 1)) ∧ 
  (∃ x, |x| ≠ 1 ∧ ¬(a ≤ x ∧ x < a + 2)) →
  a ∈ Set.Iic (-3) ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l732_73273


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l732_73216

theorem triangle_angle_calculation (A B C : Real) (a b c : Real) :
  A = π / 3 →
  a = Real.sqrt 6 →
  b = 2 →
  a > b →
  (a / Real.sin A = b / Real.sin B) →
  B = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l732_73216


namespace NUMINAMATH_CALUDE_max_area_inscribed_triangle_l732_73250

/-- The ellipse in which the triangle is inscribed -/
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

/-- A point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- The triangle inscribed in the ellipse -/
structure InscribedTriangle where
  A : EllipsePoint
  B : EllipsePoint
  C : EllipsePoint

/-- The condition that line segment AB passes through point P(1,0) -/
def passes_through_P (t : InscribedTriangle) : Prop :=
  ∃ k : ℝ, t.A.x + k * (t.B.x - t.A.x) = 1 ∧ t.A.y + k * (t.B.y - t.A.y) = 0

/-- The area of the triangle -/
noncomputable def triangle_area (t : InscribedTriangle) : ℝ :=
  abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)) / 2

/-- The theorem to be proved -/
theorem max_area_inscribed_triangle :
  ∃ (t : InscribedTriangle), passes_through_P t ∧
    (∀ (t' : InscribedTriangle), passes_through_P t' → triangle_area t' ≤ triangle_area t) ∧
    triangle_area t = 16 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_area_inscribed_triangle_l732_73250


namespace NUMINAMATH_CALUDE_abs_inequality_range_l732_73258

theorem abs_inequality_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = |x - 2| + |x + a|) →
  (∀ x : ℝ, f x ≥ 3) →
  (a ≤ -5 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_range_l732_73258


namespace NUMINAMATH_CALUDE_train_speed_l732_73228

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) :
  train_length = 900 →
  crossing_time = 53.99568034557235 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63.0036) < 0.0001 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l732_73228


namespace NUMINAMATH_CALUDE_triangle_median_and_altitude_l732_73232

/-- Triangle ABC with given vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of median -/
def isMedian (t : Triangle) (l : LineEquation) : Prop :=
  -- The line passes through vertex B and the midpoint of AC
  sorry

/-- Definition of altitude -/
def isAltitude (t : Triangle) (l : LineEquation) : Prop :=
  -- The line passes through vertex A and is perpendicular to BC
  sorry

/-- Main theorem -/
theorem triangle_median_and_altitude (t : Triangle) 
    (h1 : t.A = (-5, 0))
    (h2 : t.B = (4, -4))
    (h3 : t.C = (0, 2)) :
    ∃ (median altitude : LineEquation),
      isMedian t median ∧
      isAltitude t altitude ∧
      median = LineEquation.mk 1 7 5 ∧
      altitude = LineEquation.mk 2 (-3) 10 := by
  sorry


end NUMINAMATH_CALUDE_triangle_median_and_altitude_l732_73232


namespace NUMINAMATH_CALUDE_average_seeds_per_grape_l732_73214

/-- Theorem: Average number of seeds per grape -/
theorem average_seeds_per_grape 
  (total_seeds : ℕ) 
  (apple_seeds : ℕ) 
  (pear_seeds : ℕ) 
  (apples : ℕ) 
  (pears : ℕ) 
  (grapes : ℕ) 
  (seeds_needed : ℕ) 
  (h1 : total_seeds = 60)
  (h2 : apple_seeds = 6)
  (h3 : pear_seeds = 2)
  (h4 : apples = 4)
  (h5 : pears = 3)
  (h6 : grapes = 9)
  (h7 : seeds_needed = 3)
  : (total_seeds - (apples * apple_seeds + pears * pear_seeds) - seeds_needed) / grapes = 3 :=
by sorry

end NUMINAMATH_CALUDE_average_seeds_per_grape_l732_73214


namespace NUMINAMATH_CALUDE_peruvian_coffee_cost_l732_73231

/-- Proves that the cost of Peruvian coffee beans per pound is approximately $2.29 given the specified conditions --/
theorem peruvian_coffee_cost (colombian_cost : ℝ) (total_weight : ℝ) (mix_price : ℝ) (colombian_weight : ℝ) :
  colombian_cost = 5.50 →
  total_weight = 40 →
  mix_price = 4.60 →
  colombian_weight = 28.8 →
  ∃ (peruvian_cost : ℝ), abs (peruvian_cost - 2.29) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_peruvian_coffee_cost_l732_73231


namespace NUMINAMATH_CALUDE_rectangular_box_with_spheres_l732_73210

theorem rectangular_box_with_spheres (h : ℝ) : 
  let box_base : ℝ := 4
  let large_sphere_radius : ℝ := 2
  let small_sphere_radius : ℝ := 1
  let num_small_spheres : ℕ := 8
  h > 0 ∧ 
  box_base > 0 ∧
  large_sphere_radius > 0 ∧
  small_sphere_radius > 0 ∧
  num_small_spheres > 0 ∧
  (∃ (box : Set (ℝ × ℝ × ℝ)) (large_sphere : Set (ℝ × ℝ × ℝ)) (small_spheres : Finset (Set (ℝ × ℝ × ℝ))),
    -- Box properties
    (∀ (x y z : ℝ), (x, y, z) ∈ box ↔ 0 ≤ x ∧ x ≤ box_base ∧ 0 ≤ y ∧ y ≤ box_base ∧ 0 ≤ z ∧ z ≤ h) ∧
    -- Large sphere properties
    (∃ (cx cy cz : ℝ), large_sphere = {(x, y, z) | (x - cx)^2 + (y - cy)^2 + (z - cz)^2 ≤ large_sphere_radius^2}) ∧
    -- Small spheres properties
    (small_spheres.card = num_small_spheres) ∧
    (∀ s ∈ small_spheres, ∃ (cx cy cz : ℝ), s = {(x, y, z) | (x - cx)^2 + (y - cy)^2 + (z - cz)^2 ≤ small_sphere_radius^2}) ∧
    -- Tangency conditions
    (∀ s ∈ small_spheres, ∃ (face1 face2 face3 : Set (ℝ × ℝ × ℝ)), face1 ∪ face2 ∪ face3 ⊆ box ∧ s ∩ face1 ≠ ∅ ∧ s ∩ face2 ≠ ∅ ∧ s ∩ face3 ≠ ∅) ∧
    (∀ s ∈ small_spheres, large_sphere ∩ s ≠ ∅)) →
  h = 2 + 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_with_spheres_l732_73210


namespace NUMINAMATH_CALUDE_jackie_additional_amount_l732_73278

/-- The amount required for free shipping -/
def free_shipping_threshold : ℝ := 50

/-- The cost of a bottle of shampoo -/
def shampoo_cost : ℝ := 10

/-- The cost of a bottle of conditioner -/
def conditioner_cost : ℝ := 10

/-- The cost of a bottle of lotion -/
def lotion_cost : ℝ := 6

/-- The number of shampoo bottles Jackie ordered -/
def shampoo_quantity : ℕ := 1

/-- The number of conditioner bottles Jackie ordered -/
def conditioner_quantity : ℕ := 1

/-- The number of lotion bottles Jackie ordered -/
def lotion_quantity : ℕ := 3

/-- The additional amount Jackie needs to spend for free shipping -/
def additional_amount_needed : ℝ :=
  free_shipping_threshold - (shampoo_cost * shampoo_quantity + conditioner_cost * conditioner_quantity + lotion_cost * lotion_quantity)

theorem jackie_additional_amount : additional_amount_needed = 12 := by
  sorry

end NUMINAMATH_CALUDE_jackie_additional_amount_l732_73278


namespace NUMINAMATH_CALUDE_seven_pencils_of_one_color_l732_73266

/-- A box of colored pencils -/
structure ColoredPencilBox where
  pencils : Finset ℕ
  colors : ℕ → Finset ℕ
  total_pencils : pencils.card = 25
  color_property : ∀ s : Finset ℕ, s ⊆ pencils → s.card = 5 → ∃ c, (s ∩ colors c).card ≥ 2

/-- There are at least seven pencils of one color in the box -/
theorem seven_pencils_of_one_color (box : ColoredPencilBox) : 
  ∃ c, (box.pencils ∩ box.colors c).card ≥ 7 := by
  sorry


end NUMINAMATH_CALUDE_seven_pencils_of_one_color_l732_73266


namespace NUMINAMATH_CALUDE_right_triangle_area_l732_73249

theorem right_triangle_area (hypotenuse base : ℝ) (h1 : hypotenuse = 15) (h2 : base = 9) :
  let height : ℝ := Real.sqrt (hypotenuse^2 - base^2)
  let area : ℝ := (base * height) / 2
  area = 54 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l732_73249


namespace NUMINAMATH_CALUDE_num_divisors_not_div_by_3_eq_8_l732_73269

/-- The number of positive divisors of 210 that are not divisible by 3 -/
def num_divisors_not_div_by_3 : ℕ :=
  (Finset.filter (fun d => d ∣ 210 ∧ ¬(3 ∣ d)) (Finset.range 211)).card

/-- Theorem: The number of positive divisors of 210 that are not divisible by 3 is 8 -/
theorem num_divisors_not_div_by_3_eq_8 : num_divisors_not_div_by_3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_not_div_by_3_eq_8_l732_73269


namespace NUMINAMATH_CALUDE_triangle_existence_l732_73208

/-- Given a perimeter, inscribed circle radius, and an angle, 
    there exists a triangle with these properties -/
theorem triangle_existence (s ρ α : ℝ) (h1 : s > 0) (h2 : ρ > 0) (h3 : 0 < α ∧ α < π) :
  ∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧  -- Positive side lengths
    a + b + c = 2 * s ∧      -- Perimeter condition
    ρ = (a * b * c) / (4 * s) ∧  -- Inscribed circle radius formula
    α = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) :=  -- Cosine law for angle
by sorry

end NUMINAMATH_CALUDE_triangle_existence_l732_73208


namespace NUMINAMATH_CALUDE_logarithm_inequality_l732_73297

theorem logarithm_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  Real.log a ^ 2 / Real.log (b + c) + Real.log b ^ 2 / Real.log (c + a) + Real.log c ^ 2 / Real.log (a + b) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l732_73297


namespace NUMINAMATH_CALUDE_triangle_side_length_l732_73233

theorem triangle_side_length (a b : ℝ) (A B C : ℝ) :
  b = 2 →
  B = 30 * (π / 180) →
  C = 135 * (π / 180) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a = Real.sqrt 6 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l732_73233


namespace NUMINAMATH_CALUDE_sin_2theta_value_l732_73272

theorem sin_2theta_value (θ : Real) (h : Real.tan θ + 1 / Real.tan θ = 2) : 
  Real.sin (2 * θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l732_73272


namespace NUMINAMATH_CALUDE_rind_papyrus_fraction_decomposition_l732_73236

theorem rind_papyrus_fraction_decomposition : 
  (2 : ℚ) / 73 = 1 / 60 + 1 / 219 + 1 / 292 + 1 / 365 := by
  sorry

end NUMINAMATH_CALUDE_rind_papyrus_fraction_decomposition_l732_73236


namespace NUMINAMATH_CALUDE_total_cost_matches_expected_l732_73221

/-- Calculate the total cost of an order with given conditions --/
def calculate_total_cost (burger_price : ℚ) (soda_price : ℚ) (chicken_sandwich_price : ℚ) 
  (happy_hour_discount : ℚ) (coupon_discount : ℚ) (sales_tax : ℚ) 
  (paulo_burgers : ℕ) (paulo_sodas : ℕ) (jeremy_burgers : ℕ) (jeremy_sodas : ℕ) 
  (stephanie_burgers : ℕ) (stephanie_sodas : ℕ) (stephanie_chicken : ℕ) : ℚ :=
  let total_burgers := paulo_burgers + jeremy_burgers + stephanie_burgers
  let total_sodas := paulo_sodas + jeremy_sodas + stephanie_sodas
  let subtotal := burger_price * total_burgers + soda_price * total_sodas + 
                  chicken_sandwich_price * stephanie_chicken
  let tax_amount := sales_tax * subtotal
  let total_with_tax := subtotal + tax_amount
  let coupon_applied := if total_with_tax > 25 then total_with_tax - coupon_discount else total_with_tax
  let happy_hour_discount_amount := if total_burgers > 2 then happy_hour_discount * (burger_price * total_burgers) else 0
  coupon_applied - happy_hour_discount_amount

/-- Theorem stating that the total cost matches the expected result --/
theorem total_cost_matches_expected : 
  calculate_total_cost 6 2 7.5 0.1 5 0.05 1 1 2 2 3 1 1 = 45.48 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_matches_expected_l732_73221


namespace NUMINAMATH_CALUDE_solution_in_first_quadrant_l732_73207

theorem solution_in_first_quadrant (d : ℝ) :
  (∃ x y : ℝ, x - 2*y = 5 ∧ d*x + y = 6 ∧ x > 0 ∧ y > 0) ↔ -1/2 < d ∧ d < 6/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_first_quadrant_l732_73207


namespace NUMINAMATH_CALUDE_min_xy_value_l732_73245

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 / (2 + x) + 3 / (2 + y) = 1) : 
  ∀ z, z = x * y → z ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l732_73245


namespace NUMINAMATH_CALUDE_largest_indecomposable_amount_l732_73295

/-- Represents the set of coin denominations in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |> List.map (fun k => 3^(n - k) * 5^k)

/-- Predicate to check if a number is decomposable using given coin denominations -/
def is_decomposable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), 
    coeffs.length = n + 1 ∧ 
    (List.zip coeffs (coin_denominations n) |> List.map (fun (c, d) => c * d) |> List.sum) = s

/-- The main theorem stating the largest indecomposable amount -/
theorem largest_indecomposable_amount (n : ℕ) : 
  ¬(is_decomposable (5^(n+1) - 2 * 3^(n+1)) n) ∧ 
  ∀ m : ℕ, m > (5^(n+1) - 2 * 3^(n+1)) → is_decomposable m n :=
by sorry

end NUMINAMATH_CALUDE_largest_indecomposable_amount_l732_73295


namespace NUMINAMATH_CALUDE_fraction_simplification_l732_73282

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (x^2 + x) / (x^2 - 1) = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l732_73282


namespace NUMINAMATH_CALUDE_tangent_line_properties_l732_73220

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 + 2 * x + 1

-- Define the derivative of the function
def f' (m : ℝ) (x : ℝ) : ℝ := 3 * m * x^2 + 2

theorem tangent_line_properties (m : ℝ) :
  -- Part 1: Parallel to y = 3x
  (f' m 1 = 3 → m = 1/3) ∧
  -- Part 2: Perpendicular to y = -1/2x
  (f' m 1 = 2 → ∃ b : ℝ, ∀ x y : ℝ, y = 2 * x + b ↔ y - f m 1 = f' m 1 * (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l732_73220


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l732_73275

/-- Represents a trader selling pens -/
structure PenTrader where
  sold : ℕ
  gainInPens : ℕ

/-- Calculates the gain percentage for a pen trader -/
def gainPercentage (trader : PenTrader) : ℚ :=
  (trader.gainInPens : ℚ) / (trader.sold : ℚ) * 100

/-- Theorem stating that for a trader selling 250 pens and gaining the cost of 65 pens, 
    the gain percentage is 26% -/
theorem trader_gain_percentage :
  ∀ (trader : PenTrader), 
    trader.sold = 250 → 
    trader.gainInPens = 65 → 
    gainPercentage trader = 26 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l732_73275


namespace NUMINAMATH_CALUDE_kristy_cookies_theorem_l732_73247

/-- The number of cookies Kristy baked -/
def total_cookies : ℕ := 22

/-- The number of cookies Kristy ate -/
def cookies_eaten : ℕ := 2

/-- The number of cookies Kristy gave to her brother -/
def cookies_given_to_brother : ℕ := 1

/-- The number of cookies taken by the first friend -/
def cookies_taken_by_first_friend : ℕ := 3

/-- The number of cookies taken by the second friend -/
def cookies_taken_by_second_friend : ℕ := 5

/-- The number of cookies taken by the third friend -/
def cookies_taken_by_third_friend : ℕ := 5

/-- The number of cookies left -/
def cookies_left : ℕ := 6

/-- Theorem stating that the total number of cookies equals the sum of all distributed cookies and those left -/
theorem kristy_cookies_theorem : 
  total_cookies = 
    cookies_eaten + 
    cookies_given_to_brother + 
    cookies_taken_by_first_friend + 
    cookies_taken_by_second_friend + 
    cookies_taken_by_third_friend + 
    cookies_left :=
by
  sorry

end NUMINAMATH_CALUDE_kristy_cookies_theorem_l732_73247


namespace NUMINAMATH_CALUDE_house_worth_problem_l732_73246

theorem house_worth_problem (initial_price final_price : ℝ) 
  (h1 : final_price = initial_price * 1.1 * 0.9)
  (h2 : final_price = 99000) : initial_price = 100000 := by
  sorry

end NUMINAMATH_CALUDE_house_worth_problem_l732_73246


namespace NUMINAMATH_CALUDE_problem_solution_l732_73238

theorem problem_solution : ∀ (P Q Y : ℚ),
  P = 3012 / 4 →
  Q = P / 2 →
  Y = P - Q →
  Y = 376.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l732_73238


namespace NUMINAMATH_CALUDE_theta_range_l732_73256

theorem theta_range (θ : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l732_73256


namespace NUMINAMATH_CALUDE_anthony_pencils_l732_73203

theorem anthony_pencils (x : ℕ) : x + 56 = 65 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencils_l732_73203


namespace NUMINAMATH_CALUDE_sanya_towels_per_wash_l732_73222

/-- The number of bath towels Sanya can wash in one wash -/
def towels_per_wash : ℕ := sorry

/-- The number of hours Sanya has per day for washing -/
def hours_per_day : ℕ := 2

/-- The total number of bath towels Sanya has -/
def total_towels : ℕ := 98

/-- The number of days it takes to wash all towels -/
def days_to_wash_all : ℕ := 7

/-- Theorem stating that Sanya can wash 7 towels in one wash -/
theorem sanya_towels_per_wash :
  towels_per_wash = 7 := by sorry

end NUMINAMATH_CALUDE_sanya_towels_per_wash_l732_73222


namespace NUMINAMATH_CALUDE_special_function_uniqueness_l732_73230

/-- A function satisfying the given properties -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 2 = 2 ∧ ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

/-- The main theorem stating that any function satisfying the special properties
    is equivalent to the function f(x) = 2x -/
theorem special_function_uniqueness (g : ℝ → ℝ) (hg : special_function g) :
  ∀ x : ℝ, g x = 2 * x :=
sorry

end NUMINAMATH_CALUDE_special_function_uniqueness_l732_73230


namespace NUMINAMATH_CALUDE_min_button_presses_l732_73265

/-- Represents the state of the room --/
structure RoomState where
  armedMines : ℕ
  closedDoors : ℕ

/-- Represents the actions of pressing buttons --/
inductive ButtonPress
  | Red
  | Yellow
  | Green

/-- Defines the effect of pressing a button on the room state --/
def pressButton (state : RoomState) (button : ButtonPress) : RoomState :=
  match button with
  | ButtonPress.Red => ⟨state.armedMines + 1, state.closedDoors⟩
  | ButtonPress.Yellow => 
      if state.armedMines ≥ 2 
      then ⟨state.armedMines - 2, state.closedDoors + 1⟩ 
      else ⟨3, 3⟩  -- Reset condition
  | ButtonPress.Green => 
      if state.closedDoors ≥ 2 
      then ⟨state.armedMines, state.closedDoors - 2⟩ 
      else ⟨3, 3⟩  -- Reset condition

/-- Defines the initial state of the room --/
def initialState : RoomState := ⟨3, 3⟩

/-- Defines the goal state (all mines disarmed and all doors opened) --/
def goalState : RoomState := ⟨0, 0⟩

/-- Theorem stating that the minimum number of button presses to reach the goal state is 9 --/
theorem min_button_presses : 
  ∃ (sequence : List ButtonPress), 
    sequence.length = 9 ∧ 
    (sequence.foldl pressButton initialState = goalState) ∧
    (∀ (otherSequence : List ButtonPress), 
      otherSequence.foldl pressButton initialState = goalState → 
      otherSequence.length ≥ 9) := by
  sorry


end NUMINAMATH_CALUDE_min_button_presses_l732_73265


namespace NUMINAMATH_CALUDE_sum_of_digits_315_base_2_l732_73291

/-- The sum of the digits in the base-2 expression of 315₁₀ is equal to 6. -/
theorem sum_of_digits_315_base_2 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_315_base_2_l732_73291


namespace NUMINAMATH_CALUDE_remainder_4032_divided_by_125_l732_73283

theorem remainder_4032_divided_by_125 : 
  4032 % 125 = 32 := by sorry

end NUMINAMATH_CALUDE_remainder_4032_divided_by_125_l732_73283


namespace NUMINAMATH_CALUDE_lewis_harvest_earnings_l732_73271

/-- Calculates the total earnings during a harvest season given regular weekly earnings, overtime weekly earnings, and the number of weeks. -/
def total_harvest_earnings (regular_weekly : ℕ) (overtime_weekly : ℕ) (weeks : ℕ) : ℕ :=
  (regular_weekly + overtime_weekly) * weeks

/-- Theorem stating that Lewis's total earnings during the harvest season equal $1,055,497 -/
theorem lewis_harvest_earnings :
  total_harvest_earnings 28 939 1091 = 1055497 := by
  sorry

#eval total_harvest_earnings 28 939 1091

end NUMINAMATH_CALUDE_lewis_harvest_earnings_l732_73271


namespace NUMINAMATH_CALUDE_largest_divisible_n_l732_73227

theorem largest_divisible_n : ∃ (n : ℕ), n = 15544 ∧ 
  (∀ m : ℕ, m > n → ¬(n + 26 ∣ n^3 + 2006)) ∧
  (n + 26 ∣ n^3 + 2006) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l732_73227


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l732_73274

def simple_interest : ℝ := 4016.25
def interest_rate : ℝ := 0.09
def time_period : ℝ := 5

theorem principal_amount_calculation :
  simple_interest / (interest_rate * time_period) = 8925 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l732_73274


namespace NUMINAMATH_CALUDE_prob_at_least_four_same_l732_73240

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the probability of all five dice showing the same number
def prob_all_same : ℚ := 1 / die_sides^(num_dice - 1)

-- Define the probability of exactly four dice showing the same number
def prob_four_same : ℚ := 
  (num_dice : ℚ) * (1 / die_sides^(num_dice - 2)) * ((die_sides - 1 : ℚ) / die_sides)

-- Theorem statement
theorem prob_at_least_four_same : 
  prob_all_same + prob_four_same = 13 / 648 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_four_same_l732_73240


namespace NUMINAMATH_CALUDE_square_value_l732_73261

theorem square_value (triangle circle square : ℕ) 
  (h1 : triangle + circle = square) 
  (h2 : triangle + circle + square = 100) : 
  square = 50 := by sorry

end NUMINAMATH_CALUDE_square_value_l732_73261


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l732_73217

/-- 
Given a parabola defined by the equation x = a * y^2 where a ≠ 0,
prove that the coordinates of its focus are (1/(4*a), 0).
-/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  let parabola := {p : ℝ × ℝ | p.1 = a * p.2^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (1 / (4 * a), 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l732_73217


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l732_73235

/-- Given a rectangular plot with the following properties:
  - The length is 32 meters more than the breadth
  - The cost of fencing at 26.50 per meter is Rs. 5300
  Prove that the length of the plot is 66 meters. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) :
  length = breadth + 32 →
  perimeter = 2 * (length + breadth) →
  perimeter * 26.5 = 5300 →
  length = 66 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_plot_length_l732_73235


namespace NUMINAMATH_CALUDE_num_sandwich_combinations_l732_73224

/-- Represents the number of different types of bread available. -/
def num_bread : ℕ := 3

/-- Represents the number of different types of meat available. -/
def num_meat : ℕ := 5

/-- Represents the number of different types of cheese available. -/
def num_cheese : ℕ := 4

/-- Represents the number of sandwiches excluded due to the turkey/swiss cheese combination. -/
def turkey_swiss_exclusions : ℕ := num_bread

/-- Represents the number of sandwiches excluded due to the roast beef/rye bread combination. -/
def roast_beef_rye_exclusions : ℕ := num_cheese

/-- Calculates the total number of possible sandwich combinations without restrictions. -/
def total_combinations : ℕ := num_bread * num_meat * num_cheese

/-- Theorem stating that the number of different sandwiches that can be ordered is 53. -/
theorem num_sandwich_combinations : 
  total_combinations - turkey_swiss_exclusions - roast_beef_rye_exclusions = 53 := by
  sorry

end NUMINAMATH_CALUDE_num_sandwich_combinations_l732_73224


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l732_73237

theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l732_73237


namespace NUMINAMATH_CALUDE_vasya_irrational_sequence_l732_73287

theorem vasya_irrational_sequence (r : ℚ) (hr : 0 < r) : 
  ∃ n : ℕ, ¬ (∃ q : ℚ, q = (λ x => Real.sqrt (x + 1))^[n] r) :=
sorry

end NUMINAMATH_CALUDE_vasya_irrational_sequence_l732_73287


namespace NUMINAMATH_CALUDE_village_population_l732_73268

theorem village_population (initial_population : ℕ) 
  (h1 : initial_population = 4599) :
  let died := (initial_population : ℚ) * (1/10)
  let remained_after_death := initial_population - ⌊died⌋
  let left := (remained_after_death : ℚ) * (1/5)
  initial_population - ⌊died⌋ - ⌊left⌋ = 3312 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l732_73268


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l732_73293

/-- The probability of selecting two non-defective pens from a box of 12 pens, where 6 are defective -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 12)
  (h2 : defective_pens = 6) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5 / 22 := by
  sorry

#check prob_two_non_defective_pens

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l732_73293


namespace NUMINAMATH_CALUDE_inconsistent_pricing_problem_l732_73298

theorem inconsistent_pricing_problem (shirt trouser tie : ℕ → ℚ) :
  (∃ x : ℕ, 6 * shirt 1 + 4 * trouser 1 + x * tie 1 = 80) →
  (4 * shirt 1 + 2 * trouser 1 + 2 * tie 1 = 140) →
  (5 * shirt 1 + 3 * trouser 1 + 2 * tie 1 = 110) →
  False :=
by
  sorry

end NUMINAMATH_CALUDE_inconsistent_pricing_problem_l732_73298


namespace NUMINAMATH_CALUDE_sum_of_products_bounds_l732_73225

/-- Represents a table of -1s and 1s -/
def Table (n : ℕ) := Fin n → Fin n → Int

/-- Defines the valid entries for the table -/
def validEntry (x : Int) : Prop := x = 1 ∨ x = -1

/-- Defines a valid table where all entries are either 1 or -1 -/
def validTable (A : Table n) : Prop :=
  ∀ i j, validEntry (A i j)

/-- Product of elements in a row -/
def rowProduct (A : Table n) (i : Fin n) : Int :=
  (Finset.univ.prod fun j => A i j)

/-- Product of elements in a column -/
def colProduct (A : Table n) (j : Fin n) : Int :=
  (Finset.univ.prod fun i => A i j)

/-- Sum of products S for a given table -/
def sumOfProducts (A : Table n) : Int :=
  (Finset.univ.sum fun i => rowProduct A i) + (Finset.univ.sum fun j => colProduct A j)

/-- Theorem stating that the sum of products is always even and bounded -/
theorem sum_of_products_bounds (n : ℕ) (A : Table n) (h : validTable A) :
  ∃ k : Int, sumOfProducts A = 2 * k ∧ -n ≤ k ∧ k ≤ n :=
sorry

end NUMINAMATH_CALUDE_sum_of_products_bounds_l732_73225


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l732_73255

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x - k ≠ 0) → k < -9/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l732_73255


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_l732_73264

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | 4 - x^2 ≤ 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

-- Theorem for (¬_U A) ∪ (¬_U B)
theorem complement_union :
  (Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x < -2 ∨ x > -1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_l732_73264


namespace NUMINAMATH_CALUDE_quadratic_min_values_l732_73248

-- Define the quadratic function
def f (x a : ℝ) : ℝ := 2 * x^2 - 4 * a * x + a^2 + 2 * a + 2

-- State the theorem
theorem quadratic_min_values (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → f x a ≥ 2) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f x a = 2) →
  a = 0 ∨ a = 2 ∨ a = -3 - Real.sqrt 7 ∨ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_min_values_l732_73248


namespace NUMINAMATH_CALUDE_candy_distribution_l732_73219

theorem candy_distribution (total_candy : ℕ) (candy_per_friend : ℕ) (h1 : total_candy = 45) (h2 : candy_per_friend = 5) :
  total_candy / candy_per_friend = 9 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l732_73219


namespace NUMINAMATH_CALUDE_peach_difference_l732_73251

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 5 →
  steven_peaches = jill_peaches + 18 →
  jake_peaches = 17 →
  steven_peaches - jake_peaches = 6 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l732_73251


namespace NUMINAMATH_CALUDE_polar_to_rect_transformation_l732_73267

/-- Given a point with rectangular coordinates (10, 3) and polar coordinates (r, θ),
    prove that the point with polar coordinates (r², 2θ) has rectangular coordinates (91, 60). -/
theorem polar_to_rect_transformation (r θ : ℝ) (h1 : r * Real.cos θ = 10) (h2 : r * Real.sin θ = 3) :
  (r^2 * Real.cos (2*θ), r^2 * Real.sin (2*θ)) = (91, 60) := by
  sorry


end NUMINAMATH_CALUDE_polar_to_rect_transformation_l732_73267


namespace NUMINAMATH_CALUDE_cases_purchased_is_13_l732_73204

/-- The number of cases of water purchased initially for a children's camp --/
def cases_purchased (group1 group2 group3 : ℕ) 
  (bottles_per_case bottles_per_child_per_day camp_days additional_bottles : ℕ) : ℕ :=
  let group4 := (group1 + group2 + group3) / 2
  let total_children := group1 + group2 + group3 + group4
  let total_bottles_needed := total_children * bottles_per_child_per_day * camp_days
  let bottles_already_have := total_bottles_needed - additional_bottles
  bottles_already_have / bottles_per_case

/-- Theorem stating that the number of cases purchased initially is 13 --/
theorem cases_purchased_is_13 :
  cases_purchased 14 16 12 24 3 3 255 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cases_purchased_is_13_l732_73204


namespace NUMINAMATH_CALUDE_dot_product_perpendiculars_l732_73281

/-- Given a point P(x₀, y₀) on the curve y = x + 2/x for x > 0,
    and points A and B as the feet of perpendiculars from P to y = x and y-axis respectively,
    prove that the dot product of PA and PB is -1. -/
theorem dot_product_perpendiculars (x₀ : ℝ) (h₀ : x₀ > 0) : 
  let y₀ := x₀ + 2 / x₀
  let P := (x₀, y₀)
  let A := ((x₀ + y₀) / 2, (x₀ + y₀) / 2)  -- Foot of perpendicular to y = x
  let B := (0, y₀)  -- Foot of perpendicular to y-axis
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = -1 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_perpendiculars_l732_73281


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l732_73270

/-- Given a line with equation 3x-4y+5=0, this theorem states that its symmetric line
    with respect to the x-axis has the equation 3x+4y+5=0 -/
theorem symmetric_line_wrt_x_axis : 
  ∀ (x y : ℝ), (3 * x - 4 * y + 5 = 0) → 
  ∃ (x' y' : ℝ), (x' = x ∧ y' = -y) ∧ (3 * x' + 4 * y' + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l732_73270


namespace NUMINAMATH_CALUDE_brand_preference_ratio_l732_73239

theorem brand_preference_ratio (total : ℕ) (brand_x : ℕ) : 
  total = 400 → brand_x = 360 → 
  (brand_x : ℚ) / (total - brand_x : ℚ) = 9 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_ratio_l732_73239


namespace NUMINAMATH_CALUDE_marilyn_bananas_count_l732_73284

/-- The number of boxes Marilyn has for her bananas. -/
def num_boxes : ℕ := 8

/-- The number of bananas required in each box. -/
def bananas_per_box : ℕ := 5

/-- Theorem stating that Marilyn has 40 bananas in total. -/
theorem marilyn_bananas_count :
  num_boxes * bananas_per_box = 40 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bananas_count_l732_73284
