import Mathlib

namespace NUMINAMATH_CALUDE_max_triangle_area_l3580_358010

/-- Given a triangle ABC where BC = 2 ∛3 and ∠BAC = π/3, the maximum possible area is 3 -/
theorem max_triangle_area (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt (Real.sqrt 3) * 2
  let BAC := π / 3
  let area := Real.sqrt 3 * BC^2 / 4
  BC = Real.sqrt (Real.sqrt 3) * 2 →
  BAC = π / 3 →
  area ≤ 3 ∧ ∃ (A' B' C' : ℝ × ℝ), 
    let BC' := Real.sqrt (Real.sqrt 3) * 2
    let BAC' := π / 3
    let area' := Real.sqrt 3 * BC'^2 / 4
    BC' = Real.sqrt (Real.sqrt 3) * 2 ∧
    BAC' = π / 3 ∧
    area' = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3580_358010


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3580_358088

theorem election_votes_theorem (V : ℕ) (W L : ℕ) : 
  W + L = V →  -- Total votes
  W - L = V / 10 →  -- Initial margin
  (L + 1500) - (W - 1500) = V / 10 →  -- New margin after vote change
  V = 30000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3580_358088


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3580_358080

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 98 ∧ 
  (∀ (y : ℕ), y < x → ¬(769 ∣ (157673 - y))) ∧ 
  (769 ∣ (157673 - x)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3580_358080


namespace NUMINAMATH_CALUDE_coupon_probability_l3580_358001

theorem coupon_probability (n m k : ℕ) (hn : n = 17) (hm : m = 9) (hk : k = 6) :
  (Nat.choose k k * Nat.choose (n - k) (m - k)) / Nat.choose n m = 3 / 442 := by
  sorry

end NUMINAMATH_CALUDE_coupon_probability_l3580_358001


namespace NUMINAMATH_CALUDE_hotel_room_encoding_l3580_358025

theorem hotel_room_encoding (x : ℕ) : 
  1 ≤ x ∧ x ≤ 30 → x % 5 = 3 → x % 7 = 6 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_encoding_l3580_358025


namespace NUMINAMATH_CALUDE_savings_ratio_l3580_358058

def savings_problem (monday tuesday wednesday thursday : ℚ) : Prop :=
  let total_savings := monday + tuesday + wednesday
  let ratio := thursday / total_savings
  (monday = 15) ∧ (tuesday = 28) ∧ (wednesday = 13) ∧ (thursday = 28) → ratio = 1/2

theorem savings_ratio : ∀ (monday tuesday wednesday thursday : ℚ),
  savings_problem monday tuesday wednesday thursday :=
λ monday tuesday wednesday thursday => by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l3580_358058


namespace NUMINAMATH_CALUDE_inequality_solution_l3580_358002

theorem inequality_solution (a x : ℝ) : ax^2 - ax + x > 0 ↔
  (a = 0 ∧ x > 0) ∨
  (a = 1 ∧ x ≠ 0) ∨
  (a < 0 ∧ 0 < x ∧ x < 1 - 1/a) ∨
  (a > 1 ∧ (x < 0 ∨ x > 1 - 1/a)) ∨
  (0 < a ∧ a < 1 ∧ (x < 1 - 1/a ∨ x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3580_358002


namespace NUMINAMATH_CALUDE_completing_square_transform_l3580_358079

theorem completing_square_transform (x : ℝ) :
  (x^2 - 8*x + 5 = 0) ↔ ((x - 4)^2 = 11) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transform_l3580_358079


namespace NUMINAMATH_CALUDE_initial_kittens_count_l3580_358089

/-- The number of kittens Tim initially had -/
def initial_kittens : ℕ := 18

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- Theorem stating that the initial number of kittens is equal to
    the sum of kittens given away and kittens left -/
theorem initial_kittens_count :
  initial_kittens = kittens_to_jessica + kittens_to_sara + kittens_left :=
by sorry

end NUMINAMATH_CALUDE_initial_kittens_count_l3580_358089


namespace NUMINAMATH_CALUDE_negation_of_existential_quadratic_l3580_358076

theorem negation_of_existential_quadratic (p : Prop) : 
  (p ↔ ∃ x : ℝ, x^2 + 2*x + 2 = 0) → 
  (¬p ↔ ∀ x : ℝ, x^2 + 2*x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_quadratic_l3580_358076


namespace NUMINAMATH_CALUDE_russia_canada_size_comparison_l3580_358016

theorem russia_canada_size_comparison 
  (us canada russia : ℝ) 
  (h1 : canada = 1.5 * us) 
  (h2 : russia = 2 * us) : 
  (russia - canada) / canada = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_russia_canada_size_comparison_l3580_358016


namespace NUMINAMATH_CALUDE_sum_of_factorials_last_two_digits_l3580_358051

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def S : ℕ := (List.range 2012).map factorial |>.sum

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem sum_of_factorials_last_two_digits :
  last_two_digits S = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_factorials_last_two_digits_l3580_358051


namespace NUMINAMATH_CALUDE_height_difference_l3580_358084

/-- Given three heights in ratio 4 : 5 : 6 with the shortest being 120 cm, 
    prove that the sum of shortest and tallest minus the middle equals 150 cm -/
theorem height_difference (h₁ h₂ h₃ : ℝ) : 
  h₁ / h₂ = 4 / 5 → 
  h₂ / h₃ = 5 / 6 → 
  h₁ = 120 → 
  h₁ + h₃ - h₂ = 150 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l3580_358084


namespace NUMINAMATH_CALUDE_walking_distance_l3580_358027

theorem walking_distance (speed1 speed2 time_diff : ℝ) (h1 : speed1 = 4)
  (h2 : speed2 = 3) (h3 : time_diff = 1/2) :
  let distance := speed1 * (time_diff + distance / speed2)
  distance = 6 := by sorry

end NUMINAMATH_CALUDE_walking_distance_l3580_358027


namespace NUMINAMATH_CALUDE_garden_max_area_exists_max_area_garden_l3580_358065

/-- Represents a rectangular garden with fencing on three sides --/
structure Garden where
  width : ℝ
  length : ℝ
  fencing : ℝ
  fence_constraint : fencing = 2 * width + length

/-- The area of a rectangular garden --/
def Garden.area (g : Garden) : ℝ := g.width * g.length

/-- The maximum possible area of a garden with 400 feet of fencing --/
def max_garden_area : ℝ := 20000

/-- Theorem stating that the maximum area of a garden with 400 feet of fencing is 20000 square feet --/
theorem garden_max_area :
  ∀ g : Garden, g.fencing = 400 → g.area ≤ max_garden_area :=
by
  sorry

/-- Theorem stating that there exists a garden configuration achieving the maximum area --/
theorem exists_max_area_garden :
  ∃ g : Garden, g.fencing = 400 ∧ g.area = max_garden_area :=
by
  sorry

end NUMINAMATH_CALUDE_garden_max_area_exists_max_area_garden_l3580_358065


namespace NUMINAMATH_CALUDE_b_equals_two_l3580_358043

theorem b_equals_two (x y z a b : ℝ) 
  (eq1 : x + y = 2)
  (eq2 : x * y - z^2 = a)
  (eq3 : b = x + y + z) :
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_b_equals_two_l3580_358043


namespace NUMINAMATH_CALUDE_math_city_intersections_l3580_358083

/-- Represents a city with a number of straight, non-parallel streets -/
structure City where
  num_streets : ℕ
  streets_straight : Bool
  streets_non_parallel : Bool

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (city : City) : ℕ :=
  (city.num_streets * (city.num_streets - 1)) / 2

/-- Theorem: A city with 10 straight, non-parallel streets has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 10 ∧ c.streets_straight ∧ c.streets_non_parallel →
  max_intersections c = 45 := by
  sorry

end NUMINAMATH_CALUDE_math_city_intersections_l3580_358083


namespace NUMINAMATH_CALUDE_trigonometric_ratio_equals_one_l3580_358093

theorem trigonometric_ratio_equals_one :
  (Real.cos (70 * π / 180) * Real.cos (10 * π / 180) + Real.cos (80 * π / 180) * Real.cos (20 * π / 180)) /
  (Real.cos (69 * π / 180) * Real.cos (9 * π / 180) + Real.cos (81 * π / 180) * Real.cos (21 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_equals_one_l3580_358093


namespace NUMINAMATH_CALUDE_percent_equality_l3580_358098

theorem percent_equality (x : ℝ) : (75 / 100 * 600 = 50 / 100 * x) → x = 900 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l3580_358098


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3580_358030

def largest_three_digit_multiple_of_4 : ℕ := 996

def smallest_four_digit_multiple_of_3 : ℕ := 1002

theorem sum_of_multiples : 
  largest_three_digit_multiple_of_4 + smallest_four_digit_multiple_of_3 = 1998 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3580_358030


namespace NUMINAMATH_CALUDE_total_cost_of_collars_l3580_358014

/-- Represents the material composition and cost of a collar --/
structure Collar :=
  (nylon_inches : ℕ)
  (leather_inches : ℕ)
  (nylon_cost_per_inch : ℕ)
  (leather_cost_per_inch : ℕ)

/-- Calculates the total cost of a single collar --/
def collar_cost (c : Collar) : ℕ :=
  c.nylon_inches * c.nylon_cost_per_inch + c.leather_inches * c.leather_cost_per_inch

/-- Defines a dog collar according to the problem specifications --/
def dog_collar : Collar :=
  { nylon_inches := 18
  , leather_inches := 4
  , nylon_cost_per_inch := 1
  , leather_cost_per_inch := 2 }

/-- Defines a cat collar according to the problem specifications --/
def cat_collar : Collar :=
  { nylon_inches := 10
  , leather_inches := 2
  , nylon_cost_per_inch := 1
  , leather_cost_per_inch := 2 }

/-- Theorem stating the total cost of materials for 9 dog collars and 3 cat collars --/
theorem total_cost_of_collars :
  9 * collar_cost dog_collar + 3 * collar_cost cat_collar = 276 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_collars_l3580_358014


namespace NUMINAMATH_CALUDE_gcd_282_470_l3580_358056

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by sorry

end NUMINAMATH_CALUDE_gcd_282_470_l3580_358056


namespace NUMINAMATH_CALUDE_peter_soda_purchase_l3580_358045

/-- The amount of money Peter has left after buying soda -/
def money_left (cost_per_ounce : ℚ) (initial_money : ℚ) (ounces_bought : ℚ) : ℚ :=
  initial_money - cost_per_ounce * ounces_bought

/-- Theorem: Peter has $0.50 left after buying soda -/
theorem peter_soda_purchase : 
  let cost_per_ounce : ℚ := 25 / 100
  let initial_money : ℚ := 2
  let ounces_bought : ℚ := 6
  money_left cost_per_ounce initial_money ounces_bought = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_peter_soda_purchase_l3580_358045


namespace NUMINAMATH_CALUDE_a_work_time_l3580_358015

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12
def work_rate_C : ℚ := 1 / 4

-- Define the theorem
theorem a_work_time : 
  work_rate_A + work_rate_C = 1 / 2 ∧ 
  work_rate_B + work_rate_C = 1 / 3 ∧ 
  work_rate_B = 1 / 12 →
  1 / work_rate_A = 4 := by
  sorry


end NUMINAMATH_CALUDE_a_work_time_l3580_358015


namespace NUMINAMATH_CALUDE_four_at_three_equals_thirty_l3580_358041

-- Define the operation @
def at_op (a b : ℤ) : ℤ := 3 * a^2 - 2 * b^2

-- Theorem statement
theorem four_at_three_equals_thirty : at_op 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_four_at_three_equals_thirty_l3580_358041


namespace NUMINAMATH_CALUDE_net_effect_on_revenue_l3580_358094

theorem net_effect_on_revenue 
  (original_price original_sales : ℝ) 
  (price_reduction : ℝ) 
  (sales_increase : ℝ) 
  (h1 : price_reduction = 0.2) 
  (h2 : sales_increase = 0.8) : 
  let new_price := original_price * (1 - price_reduction)
  let new_sales := original_sales * (1 + sales_increase)
  let original_revenue := original_price * original_sales
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.44 := by
sorry

end NUMINAMATH_CALUDE_net_effect_on_revenue_l3580_358094


namespace NUMINAMATH_CALUDE_cos_150_degrees_l3580_358023

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l3580_358023


namespace NUMINAMATH_CALUDE_snowball_difference_l3580_358018

def charlie_snowballs : ℕ := 50
def lucy_snowballs : ℕ := 19

theorem snowball_difference : charlie_snowballs - lucy_snowballs = 31 := by
  sorry

end NUMINAMATH_CALUDE_snowball_difference_l3580_358018


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3580_358008

theorem fractional_equation_solution : 
  ∃ x : ℝ, (2 * x) / (x - 1) = 3 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3580_358008


namespace NUMINAMATH_CALUDE_reflection_squared_is_identity_l3580_358042

-- Define a reflection matrix over a non-zero vector
def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

-- Theorem: The square of a reflection matrix is the identity matrix
theorem reflection_squared_is_identity (v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  (reflection_matrix v) ^ 2 = !![1, 0; 0, 1] :=
sorry

end NUMINAMATH_CALUDE_reflection_squared_is_identity_l3580_358042


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3580_358029

theorem perfect_square_condition (a b k : ℝ) :
  (∃ (n : ℝ), a^2 + k*a*b + 9*b^2 = n^2) → (k = 6 ∨ k = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3580_358029


namespace NUMINAMATH_CALUDE_number_problem_l3580_358013

theorem number_problem (n : ℚ) : (1/2 : ℚ) * (3/5 : ℚ) * n = 36 → n = 120 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3580_358013


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3580_358073

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * y - 4 > 2 * y + 5) → y ≥ 10 ∧ (3 * 10 - 4 > 2 * 10 + 5) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3580_358073


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3580_358070

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3580_358070


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_l3580_358069

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials :
  units_digit (sum_factorials 99) = units_digit (sum_factorials 4) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_l3580_358069


namespace NUMINAMATH_CALUDE_amusement_park_problem_l3580_358000

/-- Proves that given the conditions of the amusement park problem, 
    the number of parents is 10 and the number of students is 5 -/
theorem amusement_park_problem 
  (total_people : ℕ)
  (adult_ticket_price : ℕ)
  (student_discount : ℚ)
  (total_spent : ℕ)
  (h1 : total_people = 15)
  (h2 : adult_ticket_price = 50)
  (h3 : student_discount = 0.6)
  (h4 : total_spent = 650) :
  ∃ (parents students : ℕ),
    parents + students = total_people ∧
    parents * adult_ticket_price + 
    students * (adult_ticket_price * (1 - student_discount)) = total_spent ∧
    parents = 10 ∧
    students = 5 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_problem_l3580_358000


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l3580_358006

/-- Given a linear function y = (m² + 1)x + 2n where m and n are constants,
    and two points A(2a - 1, y₁) and B(a² + 1, y₂) on this function,
    prove that y₁ < y₂ -/
theorem y1_less_than_y2 (m n a : ℝ) (y₁ y₂ : ℝ) 
  (h1 : y₁ = (m^2 + 1) * (2*a - 1) + 2*n) 
  (h2 : y₂ = (m^2 + 1) * (a^2 + 1) + 2*n) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l3580_358006


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_l3580_358050

theorem negation_of_existence_is_forall :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_l3580_358050


namespace NUMINAMATH_CALUDE_sphere_radius_from_cylinder_l3580_358007

/-- The radius of a sphere formed by recasting a cylindrical iron block -/
theorem sphere_radius_from_cylinder (cylinder_radius : ℝ) (cylinder_height : ℝ) (sphere_radius : ℝ) : 
  cylinder_radius = 2 →
  cylinder_height = 9 →
  (4 / 3) * Real.pi * sphere_radius ^ 3 = Real.pi * cylinder_radius ^ 2 * cylinder_height →
  sphere_radius = 3 := by
  sorry

#check sphere_radius_from_cylinder

end NUMINAMATH_CALUDE_sphere_radius_from_cylinder_l3580_358007


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_l3580_358062

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Check if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Check if three points form an equilateral triangle -/
def isEquilateral (t : EquilateralTriangle) : Prop :=
  let d12 := ((t.v1.x - t.v2.x)^2 + (t.v1.y - t.v2.y)^2)
  let d23 := ((t.v2.x - t.v3.x)^2 + (t.v2.y - t.v3.y)^2)
  let d31 := ((t.v3.x - t.v1.x)^2 + (t.v3.y - t.v1.y)^2)
  d12 = d23 ∧ d23 = d31

theorem equilateral_triangle_third_vertex 
  (t : EquilateralTriangle)
  (h1 : t.v1 = ⟨0, 3⟩)
  (h2 : t.v2 = ⟨6, 3⟩)
  (h3 : isInFirstQuadrant t.v3)
  (h4 : isEquilateral t) :
  t.v3 = ⟨6, 3 + 3 * Real.sqrt 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_l3580_358062


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_neg_one_evaluate_zero_undefined_for_one_undefined_for_two_l3580_358090

theorem simplify_expression (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ 2) :
  (1 + 3 / (a - 1)) / ((a^2 - 4) / (a - 1)) = 1 / (a - 2) := by
  sorry

-- Evaluation for a = -1
theorem evaluate_neg_one :
  (1 + 3 / (-1 - 1)) / ((-1^2 - 4) / (-1 - 1)) = -1/3 := by
  sorry

-- Evaluation for a = 0
theorem evaluate_zero :
  (1 + 3 / (0 - 1)) / ((0^2 - 4) / (0 - 1)) = -1/2 := by
  sorry

-- Undefined for a = 1
theorem undefined_for_one (h : (1 : ℝ) ≠ 2) :
  ¬∃x, (1 + 3 / (1 - 1)) / ((1^2 - 4) / (1 - 1)) = x := by
  sorry

-- Undefined for a = 2
theorem undefined_for_two (h : (2 : ℝ) ≠ 1) :
  ¬∃x, (1 + 3 / (2 - 1)) / ((2^2 - 4) / (2 - 1)) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_neg_one_evaluate_zero_undefined_for_one_undefined_for_two_l3580_358090


namespace NUMINAMATH_CALUDE_coffee_order_total_cost_l3580_358019

/-- The total cost of a coffee order -/
def coffee_order_cost (drip_coffee_price : ℝ) (drip_coffee_quantity : ℕ)
                      (espresso_price : ℝ) (espresso_quantity : ℕ)
                      (latte_price : ℝ) (latte_quantity : ℕ)
                      (vanilla_syrup_price : ℝ) (vanilla_syrup_quantity : ℕ)
                      (cold_brew_price : ℝ) (cold_brew_quantity : ℕ)
                      (cappuccino_price : ℝ) (cappuccino_quantity : ℕ) : ℝ :=
  drip_coffee_price * drip_coffee_quantity +
  espresso_price * espresso_quantity +
  latte_price * latte_quantity +
  vanilla_syrup_price * vanilla_syrup_quantity +
  cold_brew_price * cold_brew_quantity +
  cappuccino_price * cappuccino_quantity

/-- The theorem stating that the given coffee order costs $25.00 -/
theorem coffee_order_total_cost :
  coffee_order_cost 2.25 2 3.50 1 4.00 2 0.50 1 2.50 2 3.50 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_coffee_order_total_cost_l3580_358019


namespace NUMINAMATH_CALUDE_tangent_line_theorem_l3580_358036

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a-2)*x

-- Define the tangent line at the origin
def tangent_line_at_origin (a : ℝ) (x : ℝ) : ℝ := -2*x

-- Theorem statement
theorem tangent_line_theorem (a : ℝ) :
  ∀ x : ℝ, (tangent_line_at_origin a x) = 
    (deriv (f a)) 0 * x + (f a 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_l3580_358036


namespace NUMINAMATH_CALUDE_unique_solution_l3580_358099

theorem unique_solution (x y z : ℝ) 
  (hx : x > 3) (hy : y > 3) (hz : z > 3)
  (h : ((x + 2)^2) / (y + z - 2) + ((y + 4)^2) / (z + x - 4) + ((z + 6)^2) / (x + y - 6) = 36) :
  x = 10 ∧ y = 8 ∧ z = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l3580_358099


namespace NUMINAMATH_CALUDE_base7_divisibility_by_19_l3580_358026

def base7ToDecimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

theorem base7_divisibility_by_19 :
  ∃ (x : ℕ), x < 7 ∧ 19 ∣ (base7ToDecimal 2 5 x 3) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_base7_divisibility_by_19_l3580_358026


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l3580_358092

/-- Calculates the average runs for a batsman over multiple sets of matches -/
def average_runs (runs_per_set : List ℕ) (matches_per_set : List ℕ) : ℚ :=
  (runs_per_set.zip matches_per_set).map (fun (r, m) => r * m)
    |> List.sum
    |> (fun total_runs => total_runs / matches_per_set.sum)

theorem batsman_average_theorem (first_10_avg : ℕ) (next_10_avg : ℕ) :
  first_10_avg = 40 →
  next_10_avg = 30 →
  average_runs [first_10_avg, next_10_avg] [10, 10] = 35 := by
  sorry

#eval average_runs [40, 30] [10, 10]

end NUMINAMATH_CALUDE_batsman_average_theorem_l3580_358092


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_60_l3580_358077

theorem largest_multiple_of_8_less_than_60 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 60 → n ≤ 56 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_60_l3580_358077


namespace NUMINAMATH_CALUDE_inequality_proof_l3580_358097

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (a + 1/b)^2 > (b + 1/a)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3580_358097


namespace NUMINAMATH_CALUDE_probability_both_selected_l3580_358011

theorem probability_both_selected (prob_ram : ℚ) (prob_ravi : ℚ) 
  (h1 : prob_ram = 3/7) (h2 : prob_ravi = 1/5) : 
  prob_ram * prob_ravi = 3/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l3580_358011


namespace NUMINAMATH_CALUDE_total_weekly_sleep_is_123_l3580_358054

/-- Represents the type of day (odd or even) -/
inductive DayType
| odd
| even

/-- Calculates the sleep time for a cougar based on the day type -/
def cougarSleep (day : DayType) : ℕ :=
  match day with
  | DayType.odd => 6
  | DayType.even => 4

/-- Calculates the sleep time for a zebra based on the cougar's sleep time -/
def zebraSleep (cougarSleepTime : ℕ) : ℕ :=
  cougarSleepTime + 2

/-- Calculates the sleep time for a lion based on the day type and other animals' sleep times -/
def lionSleep (day : DayType) (cougarSleepTime zebraSleepTime : ℕ) : ℕ :=
  match day with
  | DayType.odd => cougarSleepTime + 1
  | DayType.even => zebraSleepTime - 3

/-- Calculates the total weekly sleep time for all three animals -/
def totalWeeklySleep : ℕ :=
  let oddDays := 4
  let evenDays := 3
  let cougarTotal := oddDays * cougarSleep DayType.odd + evenDays * cougarSleep DayType.even
  let zebraTotal := oddDays * zebraSleep (cougarSleep DayType.odd) + evenDays * zebraSleep (cougarSleep DayType.even)
  let lionTotal := oddDays * lionSleep DayType.odd (cougarSleep DayType.odd) (zebraSleep (cougarSleep DayType.odd)) +
                   evenDays * lionSleep DayType.even (cougarSleep DayType.even) (zebraSleep (cougarSleep DayType.even))
  cougarTotal + zebraTotal + lionTotal

theorem total_weekly_sleep_is_123 : totalWeeklySleep = 123 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_sleep_is_123_l3580_358054


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_l3580_358064

theorem real_part_of_reciprocal (z : ℂ) : 
  z ≠ (1 : ℂ) →
  Complex.abs z = 1 → 
  z = Complex.exp (Complex.I * Real.pi / 3) →
  Complex.re (1 / (1 - z)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_l3580_358064


namespace NUMINAMATH_CALUDE_abc_sum_theorem_l3580_358071

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 6

def to_base_6 (n : ℕ) : ℕ := n

theorem abc_sum_theorem (A B C : ℕ) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_valid : is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C)
  (h_equation : to_base_6 (A * 36 + B * 6 + C) + to_base_6 (B * 6 + C) = to_base_6 (A * 36 + C * 6 + A)) :
  to_base_6 (A + B + C) = 11 :=
sorry

end NUMINAMATH_CALUDE_abc_sum_theorem_l3580_358071


namespace NUMINAMATH_CALUDE_otimes_four_otimes_four_four_l3580_358003

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^3 + 3*x*y - y

-- Theorem statement
theorem otimes_four_otimes_four_four : otimes 4 (otimes 4 4) = 1252 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_otimes_four_four_l3580_358003


namespace NUMINAMATH_CALUDE_cone_height_for_right_angle_vertex_l3580_358022

/-- Represents a cone with given volume and vertex angle -/
structure Cone where
  volume : ℝ
  vertexAngle : ℝ

/-- The height of a cone given its volume and vertex angle -/
def coneHeight (c : Cone) : ℝ :=
  sorry

theorem cone_height_for_right_angle_vertex (c : Cone) 
  (h_volume : c.volume = 20000 * Real.pi)
  (h_angle : c.vertexAngle = Real.pi / 2) :
  ∃ (r : ℝ), coneHeight c = r * Real.sqrt 2 ∧ 
  r^3 * Real.sqrt 2 = 60000 :=
sorry

end NUMINAMATH_CALUDE_cone_height_for_right_angle_vertex_l3580_358022


namespace NUMINAMATH_CALUDE_robin_hair_growth_l3580_358082

/-- Calculates hair growth given initial length, cut length, and final length -/
def hair_growth (initial_length cut_length final_length : ℕ) : ℕ :=
  final_length - (initial_length - cut_length)

/-- Theorem: Given the problem conditions, hair growth is 12 inches -/
theorem robin_hair_growth :
  hair_growth 16 11 17 = 12 := by sorry

end NUMINAMATH_CALUDE_robin_hair_growth_l3580_358082


namespace NUMINAMATH_CALUDE_f_at_neg_one_eq_78_l3580_358044

/-- The polynomial g(x) -/
def g (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 - 5*x + 15

/-- The polynomial f(x) -/
def f (q r : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 50*x + r

/-- Theorem stating that f(-1) = 78 given the conditions -/
theorem f_at_neg_one_eq_78 
  (p q r : ℝ) 
  (h1 : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g p x = 0 ∧ g p y = 0 ∧ g p z = 0)
  (h2 : ∀ x : ℝ, g p x = 0 → f q r x = 0) :
  f q r (-1) = 78 := by
  sorry

end NUMINAMATH_CALUDE_f_at_neg_one_eq_78_l3580_358044


namespace NUMINAMATH_CALUDE_labourer_income_l3580_358037

/-- The monthly income of a labourer given specific expenditure and savings patterns -/
theorem labourer_income (
  first_period : ℕ) 
  (second_period : ℕ)
  (first_expenditure : ℚ)
  (second_expenditure : ℚ)
  (savings : ℚ)
  (h1 : first_period = 8)
  (h2 : second_period = 6)
  (h3 : first_expenditure = 80)
  (h4 : second_expenditure = 65)
  (h5 : savings = 50)
  : ∃ (income : ℚ), 
    income * ↑first_period < first_expenditure * ↑first_period ∧ 
    income * ↑second_period = second_expenditure * ↑second_period + 
      (first_expenditure * ↑first_period - income * ↑first_period) + savings ∧
    income = 1080 / 14 := by
  sorry


end NUMINAMATH_CALUDE_labourer_income_l3580_358037


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3580_358095

-- Define set P
def P : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Define set Q
def Q : Set ℝ := {y | ∃ x ∈ P, y = (1/2) * x^2 - 1}

-- Theorem statement
theorem intersection_of_P_and_Q :
  P ∩ Q = {m : ℝ | m ≥ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3580_358095


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l3580_358012

theorem greatest_three_digit_number : ∃ (n : ℕ), n = 953 ∧
  n ≤ 999 ∧
  ∃ (k : ℕ), n = 9 * k + 2 ∧
  ∃ (m : ℕ), n = 5 * m + 3 ∧
  ∃ (l : ℕ), n = 7 * l + 4 ∧
  ∀ (x : ℕ), x ≤ 999 → 
    (∃ (a b c : ℕ), x = 9 * a + 2 ∧ x = 5 * b + 3 ∧ x = 7 * c + 4) → 
    x ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l3580_358012


namespace NUMINAMATH_CALUDE_ultramarathon_training_l3580_358031

theorem ultramarathon_training (initial_time initial_speed : ℝ)
  (time_increase_percent speed_increase : ℝ)
  (h1 : initial_time = 8)
  (h2 : initial_speed = 8)
  (h3 : time_increase_percent = 75)
  (h4 : speed_increase = 4) :
  let new_time := initial_time * (1 + time_increase_percent / 100)
  let new_speed := initial_speed + speed_increase
  new_time * new_speed = 168 := by
  sorry

end NUMINAMATH_CALUDE_ultramarathon_training_l3580_358031


namespace NUMINAMATH_CALUDE_set_b_forms_triangle_l3580_358086

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A set of line segments can form a triangle if they satisfy the triangle inequality. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem set_b_forms_triangle :
  can_form_triangle 8 6 3 := by
  sorry

end NUMINAMATH_CALUDE_set_b_forms_triangle_l3580_358086


namespace NUMINAMATH_CALUDE_unopened_box_cards_l3580_358091

theorem unopened_box_cards (initial_cards given_away_cards final_total_cards : ℕ) :
  initial_cards = 26 →
  given_away_cards = 18 →
  final_total_cards = 48 →
  final_total_cards = (initial_cards - given_away_cards) + (final_total_cards - (initial_cards - given_away_cards)) :=
by
  sorry

end NUMINAMATH_CALUDE_unopened_box_cards_l3580_358091


namespace NUMINAMATH_CALUDE_remaining_pennies_l3580_358096

theorem remaining_pennies (initial : ℝ) (spent : ℝ) (remaining : ℝ) 
  (h1 : initial = 98.5) 
  (h2 : spent = 93.25) 
  (h3 : remaining = initial - spent) : 
  remaining = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pennies_l3580_358096


namespace NUMINAMATH_CALUDE_total_values_count_l3580_358048

theorem total_values_count (initial_mean correct_mean : ℝ) 
  (incorrect_value correct_value : ℝ) (n : ℕ) : 
  initial_mean = 150 →
  correct_mean = 151.25 →
  incorrect_value = 135 →
  correct_value = 160 →
  (n : ℝ) * initial_mean = (n : ℝ) * correct_mean - (correct_value - incorrect_value) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_total_values_count_l3580_358048


namespace NUMINAMATH_CALUDE_not_equal_necessary_not_sufficient_l3580_358021

-- Define the relationship between α and β
def not_equal (α β : Real) : Prop := α ≠ β

-- Define the relationship between sin α and sin β
def sin_not_equal (α β : Real) : Prop := Real.sin α ≠ Real.sin β

-- Theorem stating that not_equal is a necessary but not sufficient condition for sin_not_equal
theorem not_equal_necessary_not_sufficient :
  (∀ α β : Real, sin_not_equal α β → not_equal α β) ∧
  ¬(∀ α β : Real, not_equal α β → sin_not_equal α β) :=
sorry

end NUMINAMATH_CALUDE_not_equal_necessary_not_sufficient_l3580_358021


namespace NUMINAMATH_CALUDE_max_profit_theorem_l3580_358057

/-- Represents the unit prices and quantities of exercise books -/
structure BookPrices where
  regular : ℝ
  deluxe : ℝ

/-- Represents the purchase quantities of exercise books -/
structure PurchaseQuantities where
  regular : ℝ
  deluxe : ℝ

/-- Defines the conditions of the problem -/
def problem_conditions (prices : BookPrices) : Prop :=
  150 * prices.regular + 100 * prices.deluxe = 1450 ∧
  200 * prices.regular + 50 * prices.deluxe = 1100

/-- Defines the profit function -/
def profit_function (prices : BookPrices) (quantities : PurchaseQuantities) : ℝ :=
  (prices.regular - 2) * quantities.regular + (prices.deluxe - 7) * quantities.deluxe

/-- Defines the purchase constraints -/
def purchase_constraints (quantities : PurchaseQuantities) : Prop :=
  quantities.regular + quantities.deluxe = 500 ∧
  quantities.regular ≥ 3 * quantities.deluxe

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit_theorem (prices : BookPrices) 
  (h_conditions : problem_conditions prices) :
  ∃ (quantities : PurchaseQuantities),
    purchase_constraints quantities ∧
    profit_function prices quantities = 750 ∧
    ∀ (other_quantities : PurchaseQuantities),
      purchase_constraints other_quantities →
      profit_function prices other_quantities ≤ 750 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l3580_358057


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_third_l3580_358072

theorem sin_alpha_minus_pi_third (α : ℝ) (h : Real.cos (α + π/6) = -1/3) : 
  Real.sin (α - π/3) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_third_l3580_358072


namespace NUMINAMATH_CALUDE_arrangement_theorem_l3580_358066

/-- The number of ways to arrange 3 individuals on 7 steps --/
def arrangement_count : ℕ := 336

/-- The number of steps --/
def num_steps : ℕ := 7

/-- The number of individuals --/
def num_individuals : ℕ := 3

/-- The maximum number of people allowed on each step --/
def max_per_step : ℕ := 2

/-- Function to calculate the number of arrangements --/
def calculate_arrangements (steps : ℕ) (individuals : ℕ) (max_per_step : ℕ) : ℕ := 
  sorry

theorem arrangement_theorem : 
  calculate_arrangements num_steps num_individuals max_per_step = arrangement_count := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l3580_358066


namespace NUMINAMATH_CALUDE_at_least_two_positive_roots_l3580_358046

def f (x : ℝ) : ℝ := x^11 + 8*x^10 + 15*x^9 - 1729*x^8 + 1379*x^7 - 172*x^6

theorem at_least_two_positive_roots :
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a ≠ b ∧ f a = 0 ∧ f b = 0 :=
sorry

end NUMINAMATH_CALUDE_at_least_two_positive_roots_l3580_358046


namespace NUMINAMATH_CALUDE_total_cans_donated_l3580_358035

/-- The number of homeless shelters -/
def num_shelters : ℕ := 6

/-- The number of people served by each shelter -/
def people_per_shelter : ℕ := 30

/-- The number of cans of soup bought per person -/
def cans_per_person : ℕ := 10

/-- Theorem: The total number of cans of soup Mark donates is 1800 -/
theorem total_cans_donated : 
  num_shelters * people_per_shelter * cans_per_person = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_donated_l3580_358035


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_once_l3580_358078

/-- The ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l with equation 3x₀x + 4y₀y - 12 = 0, where (x₀, y₀) is on the ellipse -/
def line_l (x₀ y₀ x y : ℝ) : Prop := 3*x₀*x + 4*y₀*y - 12 = 0

/-- The point P(x₀, y₀) is on the ellipse C -/
def point_on_ellipse (x₀ y₀ : ℝ) : Prop := ellipse_C x₀ y₀

theorem line_intersects_ellipse_once (x₀ y₀ : ℝ) 
  (h : point_on_ellipse x₀ y₀) :
  ∃! p : ℝ × ℝ, ellipse_C p.1 p.2 ∧ line_l x₀ y₀ p.1 p.2 := by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_once_l3580_358078


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3580_358005

theorem no_positive_integer_solutions : 
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^4 * y^4 - 8 * x^2 * y^2 + 12 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3580_358005


namespace NUMINAMATH_CALUDE_marias_towels_l3580_358038

theorem marias_towels (green_towels white_towels given_towels : ℕ) : 
  green_towels = 40 →
  white_towels = 44 →
  given_towels = 65 →
  green_towels + white_towels - given_towels = 19 := by
sorry

end NUMINAMATH_CALUDE_marias_towels_l3580_358038


namespace NUMINAMATH_CALUDE_mistaken_calculation_l3580_358059

theorem mistaken_calculation (x : ℝ) : x + 2 = 6 → x - 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l3580_358059


namespace NUMINAMATH_CALUDE_multiple_reals_less_than_negative_one_l3580_358052

theorem multiple_reals_less_than_negative_one :
  ∃ (x y : ℝ), x < -1 ∧ y < -1 ∧ x ≠ y :=
sorry

end NUMINAMATH_CALUDE_multiple_reals_less_than_negative_one_l3580_358052


namespace NUMINAMATH_CALUDE_twenty_paise_coins_count_l3580_358032

theorem twenty_paise_coins_count 
  (total_coins : ℕ) 
  (total_value : ℚ) 
  (h_total_coins : total_coins = 334)
  (h_total_value : total_value = 71)
  : ∃ (coins_20p coins_25p : ℕ), 
    coins_20p + coins_25p = total_coins ∧ 
    (1/5 : ℚ) * coins_20p + (1/4 : ℚ) * coins_25p = total_value ∧
    coins_20p = 250 := by
  sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_count_l3580_358032


namespace NUMINAMATH_CALUDE_elections_with_at_least_two_past_officers_l3580_358017

def total_candidates : ℕ := 20
def past_officers : ℕ := 10
def positions : ℕ := 6

def total_elections : ℕ := Nat.choose total_candidates positions

def elections_no_past_officers : ℕ := Nat.choose (total_candidates - past_officers) positions

def elections_one_past_officer : ℕ := 
  Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions - 1)

theorem elections_with_at_least_two_past_officers : 
  total_elections - elections_no_past_officers - elections_one_past_officer = 36030 := by
  sorry

end NUMINAMATH_CALUDE_elections_with_at_least_two_past_officers_l3580_358017


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3580_358024

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3580_358024


namespace NUMINAMATH_CALUDE_extreme_point_iff_a_eq_zero_l3580_358061

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

/-- Definition of an extreme point -/
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f y ≥ f x ∨ f y ≤ f x

/-- The main theorem stating that x=1 is an extreme point of f(x) iff a=0 -/
theorem extreme_point_iff_a_eq_zero (a : ℝ) :
  is_extreme_point (f a) 1 ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_extreme_point_iff_a_eq_zero_l3580_358061


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3580_358028

theorem complex_fraction_simplification :
  (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3580_358028


namespace NUMINAMATH_CALUDE_problem_statement_l3580_358033

theorem problem_statement (x y : ℝ) (h : x + 2*y = 30) : 
  x/5 + 2*y/3 + 2*y/5 + x/3 = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3580_358033


namespace NUMINAMATH_CALUDE_robert_ate_more_chocolates_l3580_358055

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 13

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 4

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_chocolates : chocolate_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_more_chocolates_l3580_358055


namespace NUMINAMATH_CALUDE_range_of_m_l3580_358060

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : A ∪ B m = A → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3580_358060


namespace NUMINAMATH_CALUDE_problem_statement_l3580_358075

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3580_358075


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l3580_358049

theorem triangle_area_from_squares (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2 : ℝ) * a * b = 24 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l3580_358049


namespace NUMINAMATH_CALUDE_smallest_sum_of_products_l3580_358020

theorem smallest_sum_of_products (b : Fin 100 → Int) 
  (h : ∀ i, b i = 1 ∨ b i = -1) :
  22 = (Finset.range 100).sum (λ i => 
    (Finset.range 100).sum (λ j => 
      if i < j then b i * b j else 0)) ∧
  ∀ (c : Fin 100 → Int) (hc : ∀ i, c i = 1 ∨ c i = -1),
    0 < (Finset.range 100).sum (λ i => 
      (Finset.range 100).sum (λ j => 
        if i < j then c i * c j else 0)) →
    22 ≤ (Finset.range 100).sum (λ i => 
      (Finset.range 100).sum (λ j => 
        if i < j then c i * c j else 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_products_l3580_358020


namespace NUMINAMATH_CALUDE_randy_biscuits_left_l3580_358004

/-- The number of biscuits Randy is left with after receiving and losing some -/
def biscuits_left (initial : ℕ) (from_father : ℕ) (from_mother : ℕ) (eaten_by_brother : ℕ) : ℕ :=
  initial + from_father + from_mother - eaten_by_brother

/-- Theorem stating that Randy is left with 40 biscuits -/
theorem randy_biscuits_left : biscuits_left 32 13 15 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_randy_biscuits_left_l3580_358004


namespace NUMINAMATH_CALUDE_iv_bottle_capacity_l3580_358039

/-- Calculates the total capacity of an IV bottle given initial volume, flow rate, and elapsed time. -/
def totalCapacity (initialVolume : ℝ) (flowRate : ℝ) (elapsedTime : ℝ) : ℝ :=
  initialVolume + flowRate * elapsedTime

/-- Theorem stating that given the specified conditions, the total capacity of the IV bottle is 150 mL. -/
theorem iv_bottle_capacity :
  let initialVolume : ℝ := 100
  let flowRate : ℝ := 2.5
  let elapsedTime : ℝ := 12
  totalCapacity initialVolume flowRate elapsedTime = 150 := by
  sorry

#eval totalCapacity 100 2.5 12

end NUMINAMATH_CALUDE_iv_bottle_capacity_l3580_358039


namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_formula_l3580_358067

/-- The area of a rectangle with a rectangular hole -/
def rectangle_with_hole_area (x : ℝ) : ℝ :=
  let large_length : ℝ := 2 * x + 8
  let large_width : ℝ := x + 6
  let hole_length : ℝ := 3 * x - 4
  let hole_width : ℝ := x - 3
  (large_length * large_width) - (hole_length * hole_width)

/-- Theorem: The area of the rectangle with a hole is equal to -x^2 + 33x + 36 -/
theorem rectangle_with_hole_area_formula (x : ℝ) :
  rectangle_with_hole_area x = -x^2 + 33*x + 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_formula_l3580_358067


namespace NUMINAMATH_CALUDE_polygon_angles_theorem_l3580_358040

theorem polygon_angles_theorem (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 2 * 360) →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_angles_theorem_l3580_358040


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3580_358074

theorem chess_tournament_games (n : ℕ) (h : n = 25) : 
  n * (n - 1) = 600 → 2 * (n * (n - 1)) = 1200 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l3580_358074


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3580_358068

theorem exponent_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3580_358068


namespace NUMINAMATH_CALUDE_divide_fraction_by_integer_l3580_358034

theorem divide_fraction_by_integer :
  (3 : ℚ) / 7 / 4 = 3 / 28 := by sorry

end NUMINAMATH_CALUDE_divide_fraction_by_integer_l3580_358034


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3580_358063

-- Define the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Finset ℕ := {1, 2, 3}

-- Define set B
def B : Finset ℕ := {2, 3, 4}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3580_358063


namespace NUMINAMATH_CALUDE_stating_angle_edge_to_face_special_case_l3580_358085

/-- Represents a trihedral angle with vertex A and edges AB, AC, and AD -/
structure TrihedralAngle where
  BAC : ℝ  -- Angle between AB and AC
  CAD : ℝ  -- Angle between AC and AD
  BAD : ℝ  -- Angle between AB and AD

/-- 
Calculates the angle between edge AB and face ACD in a trihedral angle
given the measures of angles BAC, CAD, and BAD
-/
def angleEdgeToFace (t : TrihedralAngle) : ℝ :=
  sorry

/-- 
Theorem stating that for a trihedral angle with BAC = 45°, CAD = 90°, and BAD = 60°,
the angle between edge AB and face ACD is 30°
-/
theorem angle_edge_to_face_special_case :
  let t : TrihedralAngle := { BAC := Real.pi / 4, CAD := Real.pi / 2, BAD := Real.pi / 3 }
  angleEdgeToFace t = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_stating_angle_edge_to_face_special_case_l3580_358085


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3580_358009

theorem sum_of_cubes (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 270 → a + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3580_358009


namespace NUMINAMATH_CALUDE_stair_climbing_and_descending_l3580_358087

def climbStairs (n : ℕ) : ℕ :=
  if n ≤ 2 then n else climbStairs (n - 1) + climbStairs (n - 2)

def descendStairs (n : ℕ) : ℕ := 2^(n - 1)

theorem stair_climbing_and_descending :
  (climbStairs 10 = 89) ∧ (descendStairs 10 = 512) := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_and_descending_l3580_358087


namespace NUMINAMATH_CALUDE_subtraction_result_l3580_358053

theorem subtraction_result : (1000000000000 : ℕ) - 777777777777 = 222222222223 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l3580_358053


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3580_358081

/-- A function satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x + y) / 2) = (f x + f y) / 2

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = ax + b for some constants a and b. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3580_358081


namespace NUMINAMATH_CALUDE_abc_inequality_l3580_358047

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a / (a * b + 1) + b / (b * c + 1) + c / (c * a + 1) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3580_358047
