import Mathlib

namespace show_revenue_calculation_l4110_411010

def first_showing_attendance : ℕ := 200
def second_showing_multiplier : ℕ := 3
def ticket_price : ℕ := 25

theorem show_revenue_calculation :
  let second_showing_attendance := first_showing_attendance * second_showing_multiplier
  let total_attendance := first_showing_attendance + second_showing_attendance
  let total_revenue := total_attendance * ticket_price
  total_revenue = 20000 := by
  sorry

end show_revenue_calculation_l4110_411010


namespace profit_loss_ratio_l4110_411083

theorem profit_loss_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end profit_loss_ratio_l4110_411083


namespace hall_volume_theorem_l4110_411053

/-- Represents the dimensions of a rectangular hall. -/
structure HallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular hall given its dimensions. -/
def hallVolume (d : HallDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the sum of the areas of the floor and ceiling of a rectangular hall. -/
def floorCeilingArea (d : HallDimensions) : ℝ :=
  2 * d.length * d.width

/-- Calculates the sum of the areas of the four walls of a rectangular hall. -/
def wallsArea (d : HallDimensions) : ℝ :=
  2 * d.height * (d.length + d.width)

/-- Theorem stating the volume of a specific rectangular hall with given conditions. -/
theorem hall_volume_theorem (d : HallDimensions) 
    (h_length : d.length = 15)
    (h_width : d.width = 12)
    (h_area_equality : floorCeilingArea d = wallsArea d) :
    ∃ ε > 0, |hallVolume d - 1201.8| < ε := by
  sorry

end hall_volume_theorem_l4110_411053


namespace fraction_power_product_equals_three_halves_l4110_411029

theorem fraction_power_product_equals_three_halves :
  (3 / 2 : ℝ) ^ 2023 * (2 / 3 : ℝ) ^ 2022 = 3 / 2 := by
  sorry

end fraction_power_product_equals_three_halves_l4110_411029


namespace distance_ratio_of_cars_l4110_411081

-- Define the speeds and travel times for both cars
def speed_A : ℝ := 50
def time_A : ℝ := 8
def speed_B : ℝ := 25
def time_B : ℝ := 4

-- Define a function to calculate distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem distance_ratio_of_cars :
  (distance speed_A time_A) / (distance speed_B time_B) = 4 := by
  sorry


end distance_ratio_of_cars_l4110_411081


namespace scale_division_l4110_411091

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 10 * 12 + 5

/-- The number of parts the scale is divided into -/
def num_parts : ℕ := 5

/-- Calculates the length of each part when the scale is divided equally -/
def part_length : ℕ := scale_length / num_parts

/-- Theorem stating that each part of the scale is 25 inches long -/
theorem scale_division :
  part_length = 25 := by sorry

end scale_division_l4110_411091


namespace rectangle_perimeter_l4110_411040

theorem rectangle_perimeter (a b : ℝ) :
  let area := 3 * a^2 - 3 * a * b + 6 * a
  let side1 := 3 * a
  let side2 := area / side1
  side1 > 0 → side2 > 0 →
  2 * (side1 + side2) = 8 * a - 2 * b + 4 := by sorry

end rectangle_perimeter_l4110_411040


namespace range_of_m_l4110_411031

-- Define the propositions p and q
def p (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x)

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (sufficient_not_necessary (p) (q m)) →
  m ≥ 9 :=
by sorry

end range_of_m_l4110_411031


namespace sum_two_angles_gt_90_implies_acute_l4110_411068

-- Define a triangle type
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_180 : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the property of sum of any two angles greater than 90°
def sum_two_angles_gt_90 (t : Triangle) : Prop :=
  t.A + t.B > 90 ∧ t.B + t.C > 90 ∧ t.C + t.A > 90

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.A < 90 ∧ t.B < 90 ∧ t.C < 90

-- Theorem statement
theorem sum_two_angles_gt_90_implies_acute (t : Triangle) :
  sum_two_angles_gt_90 t → is_acute_triangle t :=
by sorry

end sum_two_angles_gt_90_implies_acute_l4110_411068


namespace count_solutions_l4110_411023

def positive_integer_solutions : Nat :=
  let n := 25
  let k := 5
  let min_values := [2, 3, 1, 2, 4]
  let remaining := n - (min_values.sum)
  Nat.choose (remaining + k - 1) (k - 1)

theorem count_solutions :
  positive_integer_solutions = 1190 := by
  sorry

end count_solutions_l4110_411023


namespace root_in_interval_implies_a_range_l4110_411002

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 1

-- State the theorem
theorem root_in_interval_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0) → a ∈ Set.Ici (-1) :=
by sorry

end root_in_interval_implies_a_range_l4110_411002


namespace cookie_jar_problem_l4110_411056

/-- Represents the number of raisins in the larger cookie -/
def larger_cookie_raisins : ℕ := 12

/-- Represents the total number of raisins in the jar -/
def total_raisins : ℕ := 100

/-- Represents the range of cookies in the jar -/
def cookie_range : Set ℕ := {n | 5 ≤ n ∧ n ≤ 10}

theorem cookie_jar_problem (n : ℕ) (h_n : n ∈ cookie_range) :
  ∃ (r : ℕ),
    r + 1 = larger_cookie_raisins ∧
    (n - 1) * r + (r + 1) = total_raisins :=
sorry

end cookie_jar_problem_l4110_411056


namespace disinfectant_transport_theorem_l4110_411032

/-- Represents the number of bottles a box can hold -/
structure BoxCapacity where
  large : Nat
  small : Nat

/-- Represents the cost of a box in yuan -/
structure BoxCost where
  large : Nat
  small : Nat

/-- Represents the carrying capacity of a vehicle -/
structure VehicleCapacity where
  large : Nat
  small : Nat

/-- Represents the number of boxes purchased -/
structure Boxes where
  large : Nat
  small : Nat

/-- Represents the number of vehicles of each type -/
structure Vehicles where
  typeA : Nat
  typeB : Nat

def total_bottles : Nat := 3250
def total_cost : Nat := 1700
def total_vehicles : Nat := 10

def box_capacity : BoxCapacity := { large := 10, small := 5 }
def box_cost : BoxCost := { large := 5, small := 3 }
def vehicle_capacity_A : VehicleCapacity := { large := 30, small := 10 }
def vehicle_capacity_B : VehicleCapacity := { large := 20, small := 40 }

def is_valid_box_purchase (boxes : Boxes) : Prop :=
  boxes.large * box_capacity.large + boxes.small * box_capacity.small = total_bottles ∧
  boxes.large * box_cost.large + boxes.small * box_cost.small = total_cost

def is_valid_vehicle_arrangement (vehicles : Vehicles) (boxes : Boxes) : Prop :=
  vehicles.typeA + vehicles.typeB = total_vehicles ∧
  vehicles.typeA * vehicle_capacity_A.large + vehicles.typeB * vehicle_capacity_B.large ≥ boxes.large ∧
  vehicles.typeA * vehicle_capacity_A.small + vehicles.typeB * vehicle_capacity_B.small ≥ boxes.small

def is_optimal_arrangement (vehicles : Vehicles) (boxes : Boxes) : Prop :=
  is_valid_vehicle_arrangement vehicles boxes ∧
  ∀ (other : Vehicles), is_valid_vehicle_arrangement other boxes → vehicles.typeA ≥ other.typeA

theorem disinfectant_transport_theorem : 
  ∃ (boxes : Boxes) (vehicles : Vehicles),
    is_valid_box_purchase boxes ∧
    boxes.large = 250 ∧
    boxes.small = 150 ∧
    is_optimal_arrangement vehicles boxes ∧
    vehicles.typeA = 8 ∧
    vehicles.typeB = 2 := by sorry

end disinfectant_transport_theorem_l4110_411032


namespace largest_whole_number_times_eleven_less_than_150_l4110_411044

theorem largest_whole_number_times_eleven_less_than_150 :
  (∃ x : ℕ, x = 13 ∧ 11 * x < 150 ∧ ∀ y : ℕ, y > x → 11 * y ≥ 150) :=
by sorry

end largest_whole_number_times_eleven_less_than_150_l4110_411044


namespace interest_difference_relation_l4110_411016

/-- Represents the compound interest scenario -/
structure CompoundInterest where
  P : ℝ  -- Principal amount
  r : ℝ  -- Interest rate

/-- Calculate the difference in compound interest between year 2 and year 1 -/
def interestDifference (ci : CompoundInterest) : ℝ :=
  ci.P * ci.r^2

/-- The theorem stating the relationship between the original and tripled interest rate scenarios -/
theorem interest_difference_relation (ci : CompoundInterest) :
  interestDifference { P := ci.P, r := 3 * ci.r } = 360 →
  interestDifference ci = 40 :=
by
  sorry

#check interest_difference_relation

end interest_difference_relation_l4110_411016


namespace annie_televisions_correct_l4110_411067

/-- The number of televisions Annie bought at a liquidation sale -/
def num_televisions : ℕ := 5

/-- The cost of each television -/
def television_cost : ℕ := 50

/-- The number of figurines Annie bought -/
def num_figurines : ℕ := 10

/-- The cost of each figurine -/
def figurine_cost : ℕ := 1

/-- The total amount Annie spent -/
def total_spent : ℕ := 260

/-- Theorem stating that the number of televisions Annie bought is correct -/
theorem annie_televisions_correct : 
  num_televisions * television_cost + num_figurines * figurine_cost = total_spent :=
by sorry

end annie_televisions_correct_l4110_411067


namespace probability_of_red_and_flag_in_three_draws_l4110_411086

/-- Represents a single draw from the bag -/
inductive Ball : Type
| wind : Ball
| exhibition : Ball
| red : Ball
| flag : Ball

/-- Represents a set of three draws -/
def DrawSet := (Ball × Ball × Ball)

/-- The sample data of 20 draw sets -/
def sampleData : List DrawSet := [
  (Ball.wind, Ball.red, Ball.red),
  (Ball.exhibition, Ball.flag, Ball.red),
  (Ball.flag, Ball.exhibition, Ball.wind),
  (Ball.wind, Ball.red, Ball.exhibition),
  (Ball.red, Ball.red, Ball.exhibition),
  (Ball.wind, Ball.wind, Ball.flag),
  (Ball.exhibition, Ball.red, Ball.flag),
  (Ball.red, Ball.wind, Ball.wind),
  (Ball.flag, Ball.flag, Ball.red),
  (Ball.red, Ball.exhibition, Ball.flag),
  (Ball.red, Ball.red, Ball.wind),
  (Ball.red, Ball.wind, Ball.exhibition),
  (Ball.red, Ball.red, Ball.red),
  (Ball.flag, Ball.wind, Ball.wind),
  (Ball.flag, Ball.red, Ball.exhibition),
  (Ball.flag, Ball.flag, Ball.wind),
  (Ball.exhibition, Ball.exhibition, Ball.flag),
  (Ball.red, Ball.exhibition, Ball.exhibition),
  (Ball.red, Ball.red, Ball.flag),
  (Ball.red, Ball.flag, Ball.flag)
]

/-- Checks if a draw set contains both red and flag balls -/
def containsRedAndFlag (s : DrawSet) : Bool :=
  match s with
  | (Ball.red, Ball.flag, _) | (Ball.red, _, Ball.flag) | (Ball.flag, Ball.red, _) 
  | (Ball.flag, _, Ball.red) | (_, Ball.red, Ball.flag) | (_, Ball.flag, Ball.red) => true
  | _ => false

/-- Counts the number of draw sets containing both red and flag balls -/
def countRedAndFlag (data : List DrawSet) : Nat :=
  data.filter containsRedAndFlag |>.length

/-- The theorem to be proved -/
theorem probability_of_red_and_flag_in_three_draws : 
  (countRedAndFlag sampleData : ℚ) / sampleData.length = 3 / 20 := by
  sorry


end probability_of_red_and_flag_in_three_draws_l4110_411086


namespace walmart_sales_l4110_411049

theorem walmart_sales (thermometer_price hot_water_bottle_price total_sales : ℕ)
  (thermometer_ratio : ℕ) (h1 : thermometer_price = 2)
  (h2 : hot_water_bottle_price = 6) (h3 : total_sales = 1200)
  (h4 : thermometer_ratio = 7) :
  ∃ (thermometers hot_water_bottles : ℕ),
    thermometer_price * thermometers + hot_water_bottle_price * hot_water_bottles = total_sales ∧
    thermometers = thermometer_ratio * hot_water_bottles ∧
    hot_water_bottles = 60 := by
  sorry

end walmart_sales_l4110_411049


namespace min_value_expression_l4110_411090

theorem min_value_expression (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y, x > 1 ∧ y > 0 ∧ x + y = 2 → 1 / (a - 1) + 1 / (2 * b) ≤ 1 / (x - 1) + 1 / (2 * y)) ∧
  1 / (a - 1) + 1 / (2 * b) = 3 / 2 + Real.sqrt 2 :=
by sorry

end min_value_expression_l4110_411090


namespace imaginary_part_of_z_l4110_411004

theorem imaginary_part_of_z (m : ℝ) : 
  let z : ℂ := 1 - m * Complex.I
  (z ^ 2 = -2 * Complex.I) → (z.im = -1) := by
  sorry

end imaginary_part_of_z_l4110_411004


namespace paint_cost_per_kg_paint_cost_is_36_5_l4110_411059

/-- The cost of paint per kg, given the coverage rate and the cost to paint a cube. -/
theorem paint_cost_per_kg (coverage : ℝ) (cube_side : ℝ) (total_cost : ℝ) : ℝ :=
  let surface_area := 6 * cube_side^2
  let paint_needed := surface_area / coverage
  let cost_per_kg := total_cost / paint_needed
  by
    -- Proof goes here
    sorry

/-- The cost of paint is Rs. 36.5 per kg -/
theorem paint_cost_is_36_5 :
  paint_cost_per_kg 16 8 876 = 36.5 := by
  -- Proof goes here
  sorry

end paint_cost_per_kg_paint_cost_is_36_5_l4110_411059


namespace sum_mod_nine_l4110_411087

theorem sum_mod_nine : (3612 + 3613 + 3614 + 3615 + 3616) % 9 = 1 := by
  sorry

end sum_mod_nine_l4110_411087


namespace carnival_ticket_cost_l4110_411043

/-- Calculate the total cost of carnival tickets --/
theorem carnival_ticket_cost (kids_ticket_price : ℚ) (kids_ticket_quantity : ℕ)
  (adult_ticket_price : ℚ) (adult_ticket_quantity : ℕ)
  (kids_tickets_bought : ℕ) (adult_tickets_bought : ℕ) :
  kids_ticket_price * (kids_tickets_bought / kids_ticket_quantity : ℚ) +
  adult_ticket_price * (adult_tickets_bought / adult_ticket_quantity : ℚ) = 9 :=
by
  sorry

#check carnival_ticket_cost (1/4) 4 (2/3) 3 12 9

end carnival_ticket_cost_l4110_411043


namespace derivative_f_at_1_l4110_411017

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 2 := by sorry

end derivative_f_at_1_l4110_411017


namespace tax_free_items_cost_l4110_411052

/-- Given a total purchase amount, sales tax paid, and tax rate,
    calculate the cost of tax-free items. -/
def cost_of_tax_free_items (total_purchase : ℚ) (sales_tax : ℚ) (tax_rate : ℚ) : ℚ :=
  total_purchase - sales_tax / tax_rate

/-- Theorem stating that given the specific conditions in the problem,
    the cost of tax-free items is 22. -/
theorem tax_free_items_cost :
  let total_purchase : ℚ := 25
  let sales_tax : ℚ := 30 / 100  -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 10 / 100   -- 10% = 0.10
  cost_of_tax_free_items total_purchase sales_tax tax_rate = 22 := by
  sorry


end tax_free_items_cost_l4110_411052


namespace quadratic_equation_solution_l4110_411065

theorem quadratic_equation_solution : 
  let x₁ : ℝ := (-1 + Real.sqrt 5) / 2
  let x₂ : ℝ := (-1 - Real.sqrt 5) / 2
  ∀ x : ℝ, x^2 + x - 1 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end quadratic_equation_solution_l4110_411065


namespace product_of_smallest_primes_l4110_411088

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def smallest_one_digit_primes : (ℕ × ℕ) :=
  (2, 3)

def smallest_two_digit_prime : ℕ :=
  11

theorem product_of_smallest_primes :
  let (p1, p2) := smallest_one_digit_primes
  p1 * p2 * smallest_two_digit_prime = 66 ∧
  is_prime p1 ∧ is_prime p2 ∧ is_prime smallest_two_digit_prime ∧
  p1 < 10 ∧ p2 < 10 ∧ smallest_two_digit_prime ≥ 10 ∧ smallest_two_digit_prime < 100 :=
by
  sorry

end product_of_smallest_primes_l4110_411088


namespace third_power_four_five_l4110_411072

theorem third_power_four_five (x y : ℚ) : 
  x = 5/6 → y = 6/5 → (1/3) * x^4 * y^5 = 44/111 := by
  sorry

end third_power_four_five_l4110_411072


namespace triangle_properties_l4110_411019

theorem triangle_properties (a b c A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  c * Real.sin C / Real.sin A - c = b * Real.sin B / Real.sin A - a →
  b = 2 →
  (B = π / 3 ∧
   (a = 2 * Real.sqrt 6 / 3 →
    1/2 * a * b * Real.sin C = 1 + Real.sqrt 3 / 3)) :=
by sorry

end triangle_properties_l4110_411019


namespace probability_three_colors_l4110_411045

/-- The probability of picking at least one ball of each color when selecting 3 balls from a jar
    containing 8 black, 5 white, and 3 red balls is 3/14. -/
theorem probability_three_colors (black white red : ℕ) (total : ℕ) (h1 : black = 8) (h2 : white = 5) (h3 : red = 3) 
    (h4 : total = black + white + red) : 
  (black * white * red : ℚ) / (total * (total - 1) * (total - 2) / 6) = 3 / 14 := by
  sorry

end probability_three_colors_l4110_411045


namespace sum_of_exponents_15_factorial_l4110_411033

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def sumOfPrimeFactorExponents (n : ℕ) : ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  sumOfPrimeFactorExponents (largestPerfectSquareDivisor (factorial 15).sqrt) = 10 := by
  sorry

end sum_of_exponents_15_factorial_l4110_411033


namespace parabola_axis_of_symmetry_l4110_411006

/-- The axis of symmetry of a parabola y = a(x+1)(x-3) where a ≠ 0 -/
def axisOfSymmetry (a : ℝ) (h : a ≠ 0) : ℝ := 1

/-- Theorem stating that the axis of symmetry of the parabola y = a(x+1)(x-3) where a ≠ 0 is x = 1 -/
theorem parabola_axis_of_symmetry (a : ℝ) (h : a ≠ 0) :
  axisOfSymmetry a h = 1 := by sorry

end parabola_axis_of_symmetry_l4110_411006


namespace log_sum_equals_two_l4110_411098

-- Define the logarithm functions
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_sum_equals_two : lg 0.01 + log2 16 = 2 := by
  sorry

end log_sum_equals_two_l4110_411098


namespace angle_bisector_theorem_l4110_411096

noncomputable section

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define point S on PR
def S (t : Triangle) : ℝ × ℝ := sorry

-- Define the lengths
def PR (t : Triangle) : ℝ := sorry
def PQ (t : Triangle) : ℝ := sorry
def QR (t : Triangle) : ℝ := sorry
def PS (t : Triangle) : ℝ := sorry

-- Define the angle bisector property
def bisects_angle_Q (t : Triangle) : Prop := sorry

-- Theorem statement
theorem angle_bisector_theorem (t : Triangle) 
  (h1 : PR t = 72)
  (h2 : PQ t = 32)
  (h3 : QR t = 64)
  (h4 : bisects_angle_Q t) :
  PS t = 24 := by sorry

end

end angle_bisector_theorem_l4110_411096


namespace alice_bob_meet_l4110_411021

/-- The number of points on the circular track -/
def n : ℕ := 15

/-- Alice's movement in clockwise direction per turn -/
def a : ℕ := 7

/-- Bob's movement in counterclockwise direction per turn -/
def b : ℕ := 10

/-- The function that calculates the position after k turns -/
def position (movement : ℕ) (k : ℕ) : ℕ :=
  (movement * k) % n

/-- The theorem stating that Alice and Bob meet after 8 turns -/
theorem alice_bob_meet :
  (∀ k : ℕ, k < 8 → position a k ≠ position (n - b) k) ∧
  position a 8 = position (n - b) 8 :=
sorry

end alice_bob_meet_l4110_411021


namespace ellipse_parabola_intersection_range_l4110_411092

theorem ellipse_parabola_intersection_range (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) →
  -1 ≤ a ∧ a ≤ 17/8 := by
  sorry

end ellipse_parabola_intersection_range_l4110_411092


namespace tiles_required_to_cover_floor_l4110_411064

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℚ :=
  d.length * d.width

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ :=
  inches / 12

/-- The dimensions of the floor in feet -/
def floorDimensions : Dimensions :=
  { length := 12, width := 9 }

/-- The dimensions of a tile in inches -/
def tileDimensions : Dimensions :=
  { length := 8, width := 6 }

/-- Theorem stating that 324 tiles are required to cover the floor -/
theorem tiles_required_to_cover_floor :
  (area floorDimensions) / (area { length := inchesToFeet tileDimensions.length,
                                   width := inchesToFeet tileDimensions.width }) = 324 := by
  sorry

end tiles_required_to_cover_floor_l4110_411064


namespace pattern_circle_area_ratio_l4110_411036

-- Define the circle
def circle_radius : ℝ := 3

-- Define the rectangle
def rectangle_length : ℝ := 12
def rectangle_width : ℝ := 6

-- Define the number of arcs
def num_arcs : ℕ := 6

-- Theorem statement
theorem pattern_circle_area_ratio :
  let circle_area := π * circle_radius^2
  let pattern_area := circle_area  -- Assumption: rearranged arcs preserve total area
  pattern_area / circle_area = 1 := by sorry

end pattern_circle_area_ratio_l4110_411036


namespace spaceship_age_conversion_l4110_411035

/-- Converts an octal number represented as a list of digits to its decimal equivalent. -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

/-- The octal representation of the spaceship's age -/
def spaceship_age_octal : List Nat := [3, 5, 1]

theorem spaceship_age_conversion :
  octal_to_decimal spaceship_age_octal = 233 := by
  sorry

end spaceship_age_conversion_l4110_411035


namespace maintenance_team_schedule_l4110_411095

theorem maintenance_team_schedule : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 9 11)) = 3960 := by
  sorry

end maintenance_team_schedule_l4110_411095


namespace h_more_efficient_l4110_411007

/-- The daily harvest rate of a K combine in hectares -/
def k_rate : ℝ := sorry

/-- The daily harvest rate of an H combine in hectares -/
def h_rate : ℝ := sorry

/-- The total harvest of 4 K combines and 3 H combines in 5 days -/
def harvest1 : ℝ := 5 * (4 * k_rate + 3 * h_rate)

/-- The total harvest of 3 K combines and 5 H combines in 4 days -/
def harvest2 : ℝ := 4 * (3 * k_rate + 5 * h_rate)

/-- The theorem stating that H combines harvest more per day than K combines -/
theorem h_more_efficient : harvest1 = harvest2 → h_rate > k_rate := by
  sorry

end h_more_efficient_l4110_411007


namespace chad_cracker_boxes_l4110_411015

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sandwiches Chad eats per night -/
def sandwiches_per_night : ℕ := 5

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

/-- The number of crackers in each sleeve -/
def crackers_per_sleeve : ℕ := 28

/-- The number of nights the crackers will last -/
def nights_lasting : ℕ := 56

/-- Calculates the number of boxes of crackers Chad has -/
def boxes_of_crackers : ℕ :=
  (crackers_per_sandwich * sandwiches_per_night * nights_lasting) /
  (sleeves_per_box * crackers_per_sleeve)

theorem chad_cracker_boxes :
  boxes_of_crackers = 5 := by
  sorry

end chad_cracker_boxes_l4110_411015


namespace derivative_of_f_l4110_411071

-- Define the function f(x) = (5x - 4)^3
def f (x : ℝ) : ℝ := (5 * x - 4) ^ 3

-- State the theorem that the derivative of f(x) is 15(5x - 4)^2
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 15 * (5 * x - 4) ^ 2 := by
  sorry

end derivative_of_f_l4110_411071


namespace jason_pokemon_cards_l4110_411030

theorem jason_pokemon_cards (initial_cards : ℕ) (bought_cards : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 676 → bought_cards = 224 → remaining_cards = initial_cards - bought_cards → 
  remaining_cards = 452 := by
  sorry

end jason_pokemon_cards_l4110_411030


namespace cost_of_socks_socks_cost_proof_l4110_411089

theorem cost_of_socks (initial_amount : ℕ) (shirt_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  initial_amount - shirt_cost - remaining_amount

theorem socks_cost_proof (initial_amount : ℕ) (shirt_cost : ℕ) (remaining_amount : ℕ) 
    (h1 : initial_amount = 100)
    (h2 : shirt_cost = 24)
    (h3 : remaining_amount = 65) :
  cost_of_socks initial_amount shirt_cost remaining_amount = 11 := by
  sorry

end cost_of_socks_socks_cost_proof_l4110_411089


namespace interest_calculation_l4110_411028

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

theorem interest_calculation (principal : ℚ) (rate : ℚ) (time : ℚ) 
  (h1 : principal = 3000)
  (h2 : rate = 5)
  (h3 : time = 5)
  (h4 : simple_interest principal rate time = principal - 2250) :
  simple_interest principal rate time = 750 := by
  sorry

end interest_calculation_l4110_411028


namespace aperture_radius_ratio_l4110_411011

theorem aperture_radius_ratio (r : ℝ) (h : r > 0) : 
  ∃ (r_new : ℝ), (π * r_new^2 = 2 * π * r^2) ∧ (r_new / r = Real.sqrt 2) :=
by sorry

end aperture_radius_ratio_l4110_411011


namespace unique_perfect_square_in_range_l4110_411078

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem unique_perfect_square_in_range :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 14 →
    (is_perfect_square (n.factorial * (n + 1).factorial / 3) ↔ n = 11) :=
by sorry

end unique_perfect_square_in_range_l4110_411078


namespace stripe_area_on_cylindrical_silo_l4110_411000

/-- The area of a stripe wrapped around a cylindrical silo. -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 40) 
  (h2 : stripe_width = 4) 
  (h3 : revolutions = 3) : 
  stripe_width * revolutions * Real.pi * diameter = 480 * Real.pi := by
  sorry

#check stripe_area_on_cylindrical_silo

end stripe_area_on_cylindrical_silo_l4110_411000


namespace selection_theorem_l4110_411099

/-- The number of athletes who can play both basketball and soccer -/
def both_sports (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  basketball + soccer - total

/-- The number of athletes who can only play basketball -/
def only_basketball (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  basketball - both_sports total basketball soccer

/-- The number of athletes who can only play soccer -/
def only_soccer (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  soccer - both_sports total basketball soccer

/-- The number of ways to select two athletes for basketball and soccer -/
def selection_ways (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  let b := both_sports total basketball soccer
  let ob := only_basketball total basketball soccer
  let os := only_soccer total basketball soccer
  Nat.choose b 2 + b * ob + b * os + ob * os

theorem selection_theorem (total basketball soccer : ℕ) 
  (h1 : total = 9) (h2 : basketball = 5) (h3 : soccer = 6) :
  selection_ways total basketball soccer = 28 := by
  sorry

end selection_theorem_l4110_411099


namespace correct_verb_form_surround_is_correct_verb_l4110_411085

/-- Represents the grammatical form of a verb --/
inductive VerbForm
| Base
| PresentParticiple
| PastParticiple
| PresentPerfect

/-- Represents the structure of a sentence --/
structure Sentence :=
  (verb : VerbForm)
  (isImperative : Bool)
  (hasFutureTense : Bool)

/-- Determines if a given sentence structure is correct --/
def isCorrectSentenceStructure (s : Sentence) : Prop :=
  s.isImperative ∧ s.hasFutureTense ∧ s.verb = VerbForm.Base

/-- The specific sentence structure in the problem --/
def givenSentence : Sentence :=
  { verb := VerbForm.Base,
    isImperative := true,
    hasFutureTense := true }

/-- Theorem stating that the given sentence structure is correct --/
theorem correct_verb_form :
  isCorrectSentenceStructure givenSentence :=
sorry

/-- Theorem stating that "Surround" is the correct verb to use --/
theorem surround_is_correct_verb :
  givenSentence.verb = VerbForm.Base →
  isCorrectSentenceStructure givenSentence →
  "Surround" = "Surround" :=
sorry

end correct_verb_form_surround_is_correct_verb_l4110_411085


namespace smallest_positive_root_floor_l4110_411009

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x - 3 * Real.cos x + Real.tan x

def is_smallest_positive_root (s : ℝ) : Prop :=
  s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem smallest_positive_root_floor :
  ∃ s, is_smallest_positive_root s ∧ ⌊s⌋ = 3 := by sorry

end smallest_positive_root_floor_l4110_411009


namespace conical_cylinder_volume_l4110_411097

/-- The volume of a conical cylinder with base radius 3 cm and slant height 5 cm is 12π cm³ -/
theorem conical_cylinder_volume : 
  ∀ (r h s : ℝ), 
  r = 3 → s = 5 → h^2 + r^2 = s^2 →
  (1/3) * π * r^2 * h = 12 * π := by
sorry

end conical_cylinder_volume_l4110_411097


namespace complex_equation_solution_l4110_411082

/-- Given a real number a and the imaginary unit i, if (2+ai)/(1+i) = 3+i, then a = 4 -/
theorem complex_equation_solution (a : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I →
  a = 4 := by
sorry

end complex_equation_solution_l4110_411082


namespace altitude_segment_length_l4110_411037

/-- An acute triangle with two altitudes dividing two sides -/
structure AcuteTriangleWithAltitudes where
  -- Sides
  AC : ℝ
  BC : ℝ
  -- Segments created by altitudes
  AD : ℝ
  DC : ℝ
  CE : ℝ
  EB : ℝ
  -- Conditions
  acute : AC > 0 ∧ BC > 0  -- Simplification for acute triangle
  altitude_division : AD + DC = AC ∧ CE + EB = BC
  given_lengths : AD = 6 ∧ DC = 4 ∧ CE = 3

/-- The theorem stating that y (EB) equals 11/3 -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) : t.EB = 11/3 := by
  sorry

end altitude_segment_length_l4110_411037


namespace sin_90_plus_alpha_eq_neg_half_l4110_411058

/-- Given that α is an angle in the second quadrant and tan α = -√3, prove that sin(90° + α) = -1/2 -/
theorem sin_90_plus_alpha_eq_neg_half (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.tan α = -Real.sqrt 3) : -- tan α = -√3
  Real.sin (π/2 + α) = -1/2 := by sorry

end sin_90_plus_alpha_eq_neg_half_l4110_411058


namespace chapter_page_difference_l4110_411046

theorem chapter_page_difference (first_chapter_pages second_chapter_pages : ℕ) 
  (h1 : first_chapter_pages = 37)
  (h2 : second_chapter_pages = 80) : 
  second_chapter_pages - first_chapter_pages = 43 := by
sorry

end chapter_page_difference_l4110_411046


namespace max_leftover_grapes_l4110_411084

theorem max_leftover_grapes (n : ℕ) : 
  ∃ (q r : ℕ), n = 5 * q + r ∧ r < 5 ∧ r ≤ 4 :=
by sorry

end max_leftover_grapes_l4110_411084


namespace sum_of_positive_numbers_l4110_411074

theorem sum_of_positive_numbers (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + y + x * y = 8)
  (eq2 : y + z + y * z = 15)
  (eq3 : z + x + z * x = 35) :
  x + y + z + x * y = 15 := by sorry

end sum_of_positive_numbers_l4110_411074


namespace bus_stoppage_time_l4110_411005

theorem bus_stoppage_time (s1 s2 s3 v1 v2 v3 : ℝ) 
  (h1 : s1 = 54) (h2 : s2 = 60) (h3 : s3 = 72)
  (h4 : v1 = 36) (h5 : v2 = 40) (h6 : v3 = 48) :
  (1 - v1 / s1) + (1 - v2 / s2) + (1 - v3 / s3) = 1 := by
  sorry

end bus_stoppage_time_l4110_411005


namespace largest_k_for_g_range_l4110_411008

/-- The function g(x) defined as x^2 + 5x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + k

/-- The property that -5 is in the range of g(x) -/
def inRange (k : ℝ) : Prop := ∃ x, g k x = -5

/-- The theorem stating that 5/4 is the largest value of k such that -5 is in the range of g(x) -/
theorem largest_k_for_g_range :
  (∀ k > 5/4, ¬ inRange k) ∧ inRange (5/4) :=
sorry

end largest_k_for_g_range_l4110_411008


namespace complex_number_quadrant_l4110_411093

theorem complex_number_quadrant (z : ℂ) (h : (2 + 3*I)*z = 1 + I) : 
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end complex_number_quadrant_l4110_411093


namespace one_third_of_1206_percent_of_200_l4110_411027

theorem one_third_of_1206_percent_of_200 : (1206 / 3) / 200 * 100 = 201 := by
  sorry

end one_third_of_1206_percent_of_200_l4110_411027


namespace ice_melting_problem_l4110_411062

theorem ice_melting_problem (V : ℝ) : 
  V > 0 → 
  ((1 - 3/4) * (1 - 3/4) * V = 0.75) → 
  V = 12 := by
  sorry

end ice_melting_problem_l4110_411062


namespace tree_planting_correct_l4110_411069

/-- Represents the number of trees each person should plant in different scenarios -/
structure TreePlanting where
  average : ℝ  -- Average number of trees per person for the whole class
  female : ℝ   -- Number of trees per person if only females plant
  male : ℝ     -- Number of trees per person if only males plant

/-- The tree planting scenario for the ninth-grade class -/
def class_planting : TreePlanting :=
  { average := 6
  , female := 15
  , male := 10 }

/-- Theorem stating that the given values satisfy the tree planting scenario -/
theorem tree_planting_correct (tp : TreePlanting) (h : tp = class_planting) :
  1 / tp.male + 1 / tp.female = 1 / tp.average :=
by sorry

end tree_planting_correct_l4110_411069


namespace certain_number_proof_l4110_411022

theorem certain_number_proof : ∃ x : ℝ, x * 16 = 3408 ∧ x * 1.6 = 340.8 ∧ x = 213 := by
  sorry

end certain_number_proof_l4110_411022


namespace hotel_charges_l4110_411051

-- Define the charges for each hotel
variable (P R G S T : ℝ)

-- Define the relationships between the charges
axiom p_r : P = 0.75 * R
axiom p_g : P = 0.90 * G
axiom s_r : S = 1.15 * R
axiom t_g : T = 0.80 * G

-- Theorem to prove
theorem hotel_charges :
  S = 1.5333 * P ∧ 
  T = 0.8888 * P ∧ 
  (R - G) / G = 0.18 := by sorry

end hotel_charges_l4110_411051


namespace valid_numbers_l4110_411018

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧  -- 4-digit number
  (n % 6 = 0) ∧ (n % 7 = 0) ∧ (n % 8 = 0) ∧  -- divisible by 6, 7, and 8
  (n % 4 ≠ 0) ∧ (n % 3 ≠ 0) ∧  -- not divisible by 4 or 3
  (n / 100 = 55) ∧  -- first two digits are 55
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 22) ∧  -- sum of digits is 22
  (∃ (a b : Nat), n = a * 1100 + b * 11)  -- two digits repeat twice

theorem valid_numbers : 
  ∀ n : Nat, is_valid_number n ↔ (n = 5566 ∨ n = 6655) := by
  sorry

end valid_numbers_l4110_411018


namespace missing_number_implies_next_prime_l4110_411080

/-- Definition of the table entry function -/
def table_entry (r s : ℕ) : ℕ := r * s - (r + s)

/-- Theorem: If n > 3 is not in the table, then n + 1 is prime -/
theorem missing_number_implies_next_prime (n : ℕ) (h1 : n > 3) 
  (h2 : ∀ r s, r ≥ 3 → s ≥ 3 → table_entry r s ≠ n) : 
  Nat.Prime (n + 1) := by
  sorry

end missing_number_implies_next_prime_l4110_411080


namespace empty_set_implies_m_zero_l4110_411001

theorem empty_set_implies_m_zero (m : ℝ) : (∀ x : ℝ, m * x ≠ 1) → m = 0 := by
  sorry

end empty_set_implies_m_zero_l4110_411001


namespace correct_operation_l4110_411048

theorem correct_operation (x y : ℝ) : y * x - 2 * x * y = -x * y := by
  sorry

end correct_operation_l4110_411048


namespace distinct_prime_factors_of_30_factorial_l4110_411039

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem distinct_prime_factors_of_30_factorial :
  (Finset.filter (Nat.Prime) (Finset.range 31)).card = 10 ∧
  ∀ p : ℕ, Nat.Prime p → p ∣ factorial 30 ↔ p ≤ 30 :=
by sorry

end distinct_prime_factors_of_30_factorial_l4110_411039


namespace base_k_is_seven_l4110_411025

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := 
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Converts a number from base k to base 10 -/
def baseKToBase10 (n : ℕ) (k : ℕ) : ℕ := 
  (n / 100) * k^2 + ((n / 10) % 10) * k + (n % 10)

/-- The theorem stating that 7 is the base k where (524)₈ = (664)ₖ -/
theorem base_k_is_seven : 
  ∃ k : ℕ, k > 1 ∧ base8ToBase10 524 = baseKToBase10 664 k → k = 7 := by
  sorry

end base_k_is_seven_l4110_411025


namespace inequality_proof_l4110_411076

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) :
  x^2 + y^2 + x^2 * y^2 ≤ 3 :=
by sorry

end inequality_proof_l4110_411076


namespace convenience_store_soda_sales_l4110_411070

/-- Represents the weekly soda sales of a convenience store -/
structure SodaSales where
  gallons_per_box : ℕ
  cost_per_box : ℕ
  weekly_syrup_cost : ℕ

/-- Calculates the number of gallons of soda sold per week -/
def gallons_sold_per_week (s : SodaSales) : ℕ :=
  (s.weekly_syrup_cost / s.cost_per_box) * s.gallons_per_box

/-- Theorem: Given the conditions, the store sells 180 gallons of soda per week -/
theorem convenience_store_soda_sales :
  ∀ (s : SodaSales),
    s.gallons_per_box = 30 →
    s.cost_per_box = 40 →
    s.weekly_syrup_cost = 240 →
    gallons_sold_per_week s = 180 := by
  sorry

end convenience_store_soda_sales_l4110_411070


namespace f_min_at_neg_seven_l4110_411014

/-- The quadratic function f(x) = x^2 + 14x + 6 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 6

/-- Theorem: The minimum value of f(x) occurs when x = -7 -/
theorem f_min_at_neg_seven :
  ∀ x : ℝ, f x ≥ f (-7) := by sorry

end f_min_at_neg_seven_l4110_411014


namespace sum_of_five_consecutive_even_integers_l4110_411041

theorem sum_of_five_consecutive_even_integers (n : ℤ) :
  (2*n) + (2*n + 2) + (2*n + 4) + (2*n + 6) + (2*n + 8) = 10*n + 20 := by
  sorry

end sum_of_five_consecutive_even_integers_l4110_411041


namespace tire_price_problem_l4110_411012

theorem tire_price_problem (regular_price : ℝ) : 
  (3 * regular_price + 10 = 310) → regular_price = 100 := by
  sorry

end tire_price_problem_l4110_411012


namespace kannon_apples_difference_kannon_apples_difference_proof_l4110_411026

theorem kannon_apples_difference : ℕ → Prop :=
  fun x => 
    let apples_last_night : ℕ := 3
    let bananas_last_night : ℕ := 1
    let oranges_last_night : ℕ := 4
    let apples_today : ℕ := x
    let bananas_today : ℕ := 10 * bananas_last_night
    let oranges_today : ℕ := 2 * apples_today
    let total_fruits : ℕ := 39
    (apples_last_night + bananas_last_night + oranges_last_night + 
     apples_today + bananas_today + oranges_today = total_fruits) →
    (apples_today > apples_last_night) →
    (apples_today - apples_last_night = 4)

-- Proof
theorem kannon_apples_difference_proof : kannon_apples_difference 7 := by
  sorry

end kannon_apples_difference_kannon_apples_difference_proof_l4110_411026


namespace sixth_face_configuration_l4110_411066

structure Cube where
  size : Nat
  black_cubes : Nat
  white_cubes : Nat

structure Face where
  center_white : Nat
  edge_white : Nat
  corner_white : Nat

def valid_face (f : Face) : Prop :=
  f.center_white = 1 ∧ f.edge_white = 2 ∧ f.corner_white = 1

def cube_configuration (c : Cube) (known_faces : List Face) : Prop :=
  c.size = 3 ∧
  c.black_cubes = 15 ∧
  c.white_cubes = 12 ∧
  known_faces.length = 5

theorem sixth_face_configuration
  (c : Cube)
  (known_faces : List Face)
  (h_config : cube_configuration c known_faces) :
  ∃ (sixth_face : Face), valid_face sixth_face :=
by sorry

end sixth_face_configuration_l4110_411066


namespace m_range_l4110_411047

theorem m_range (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) (h_ineq : x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
  sorry

end m_range_l4110_411047


namespace smallest_degree_for_horizontal_asymptote_l4110_411055

/-- 
Given a rational function f(x) = (5x^7 + 4x^4 - 3x + 2) / q(x),
prove that the smallest degree of q(x) for f(x) to have a horizontal asymptote is 7.
-/
theorem smallest_degree_for_horizontal_asymptote 
  (q : ℝ → ℝ) -- q is a real-valued function of a real variable
  (f : ℝ → ℝ) -- f is the rational function
  (hf : ∀ x, f x = (5*x^7 + 4*x^4 - 3*x + 2) / q x) -- definition of f
  : (∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x, abs x > M → abs (f x - L) < ε) ↔ 
    (∃ n : ℕ, n ≥ 7 ∧ ∀ x, abs (q x) ≤ abs x^n + 1 ∧ abs x^n ≤ abs (q x) + 1) :=
sorry

end smallest_degree_for_horizontal_asymptote_l4110_411055


namespace alex_sandwiches_l4110_411094

/-- The number of different sandwiches Alex can make -/
def num_sandwiches (num_meats : ℕ) (num_cheeses : ℕ) (num_breads : ℕ) : ℕ :=
  num_meats * (1 + num_cheeses + (num_cheeses.choose 2)) * num_breads

/-- Theorem stating the number of different sandwiches Alex can make -/
theorem alex_sandwiches :
  num_sandwiches 12 11 3 = 2412 := by sorry

end alex_sandwiches_l4110_411094


namespace set_equality_implies_sum_of_powers_l4110_411020

theorem set_equality_implies_sum_of_powers (a b : ℝ) : 
  ({b, b/a, 0} : Set ℝ) = {a, a+b, 1} → a^2018 + b^2018 = 2 := by
  sorry

end set_equality_implies_sum_of_powers_l4110_411020


namespace difference_of_squares_l4110_411057

theorem difference_of_squares (a : ℝ) : (a + 3) * (a - 3) = a^2 - 9 := by
  sorry

end difference_of_squares_l4110_411057


namespace angle_sum_in_triangle_l4110_411013

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- State the theorem
theorem angle_sum_in_triangle (t : Triangle) 
  (h1 : t.A = 65)
  (h2 : t.B = 40) : 
  t.C = 75 := by
  sorry

end angle_sum_in_triangle_l4110_411013


namespace exact_three_ones_between_zeros_l4110_411054

/-- A sequence of 10 elements consisting of 8 ones and 2 zeros -/
def Sequence := Fin 10 → Fin 2

/-- The number of sequences with exactly three ones between two zeros -/
def favorable_sequences : ℕ := 12

/-- The total number of possible sequences -/
def total_sequences : ℕ := Nat.choose 10 2

/-- The probability of having exactly three ones between two zeros -/
def probability : ℚ := favorable_sequences / total_sequences

theorem exact_three_ones_between_zeros :
  probability = 2 / 15 := by
  sorry

end exact_three_ones_between_zeros_l4110_411054


namespace regular_ngon_triangle_property_l4110_411034

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Triangle type: acute, right, or obtuse -/
inductive TriangleType
  | Acute
  | Right
  | Obtuse

/-- Determine the type of a triangle given its vertices -/
def triangleType (A B C : ℝ × ℝ) : TriangleType :=
  sorry

/-- The main theorem -/
theorem regular_ngon_triangle_property (n : ℕ) (hn : n > 0) :
  ∀ (P : RegularNGon n) (σ : Fin n → Fin n),
  Function.Bijective σ →
  ∃ (i j k : Fin n),
    triangleType (P.vertices i) (P.vertices j) (P.vertices k) =
    triangleType (P.vertices (σ i)) (P.vertices (σ j)) (P.vertices (σ k)) :=
sorry

end regular_ngon_triangle_property_l4110_411034


namespace complex_modulus_problem_l4110_411077

theorem complex_modulus_problem (z : ℂ) (h : z * (Complex.I + 1) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l4110_411077


namespace tank_insulation_problem_l4110_411038

/-- Proves that for a rectangular tank with given dimensions and insulation cost, 
    the third dimension is 2 feet. -/
theorem tank_insulation_problem (x : ℝ) : 
  x > 0 → 
  (2 * 3 * 5 + 2 * 3 * x + 2 * 5 * x) * 20 = 1240 → 
  x = 2 := by
sorry

end tank_insulation_problem_l4110_411038


namespace z_properties_l4110_411079

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := 2 * m + (4 - m^2) * Complex.I

/-- z lies on the imaginary axis -/
def on_imaginary_axis (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- z lies in the first or third quadrant -/
def in_first_or_third_quadrant (z : ℂ) : Prop := z.re * z.im < 0

theorem z_properties (m : ℝ) :
  (on_imaginary_axis (z m) ↔ m = 0) ∧
  (in_first_or_third_quadrant (z m) ↔ m > 2 ∨ (-2 < m ∧ m < 0)) := by
  sorry

end z_properties_l4110_411079


namespace inscribed_square_rectangle_l4110_411073

theorem inscribed_square_rectangle (a b : ℝ) : 
  (∃ (s r_short r_long : ℝ),
    s^2 = 9 ∧                     -- Area of square is 9
    r_long = 2 * r_short ∧        -- One side of rectangle is double the other
    r_short * r_long = 18 ∧       -- Area of rectangle is 18
    a + b = r_short ∧             -- a and b divide the shorter side
    a^2 + b^2 = s^2)              -- Pythagorean theorem for the right triangle formed
  → a * b = 0 := by
  sorry

end inscribed_square_rectangle_l4110_411073


namespace three_numbers_sum_l4110_411060

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 7 → 
  (a + b + c) / 3 = a + 8 → 
  (a + b + c) / 3 = c - 20 → 
  a + b + c = 57 := by
sorry

end three_numbers_sum_l4110_411060


namespace number_of_possible_D_values_l4110_411050

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the addition operation
def add (a b : Digit) : ℕ := a.val + b.val

-- Define the property of being distinct
def distinct (a b c d : Digit) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Define the main theorem
theorem number_of_possible_D_values :
  ∃ (s : Finset Digit),
    (∀ d ∈ s, ∃ (a b c e : Digit),
      distinct a b c d ∧
      add a b = d.val ∧
      add c e = d.val) ∧
    s.card = 8 := by
  sorry

end number_of_possible_D_values_l4110_411050


namespace spontaneous_low_temp_signs_l4110_411063

/-- Represents the change in enthalpy -/
def ΔH : ℝ := sorry

/-- Represents the change in entropy -/
def ΔS : ℝ := sorry

/-- Represents temperature -/
def T : ℝ := sorry

/-- Represents the change in Gibbs free energy -/
def ΔG (T : ℝ) : ℝ := ΔH - T * ΔS

/-- Represents that the reaction is spontaneous -/
def is_spontaneous (T : ℝ) : Prop := ΔG T < 0

/-- Represents that the reaction is spontaneous only at low temperatures -/
def spontaneous_at_low_temp : Prop :=
  ∃ T₀ > 0, ∀ T, 0 < T → T < T₀ → is_spontaneous T

theorem spontaneous_low_temp_signs :
  spontaneous_at_low_temp → ΔH < 0 ∧ ΔS < 0 := by
  sorry

end spontaneous_low_temp_signs_l4110_411063


namespace prob_at_least_one_boy_one_girl_l4110_411003

-- Define the probability of having a boy or a girl
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_one_girl :
  1 - (prob_boy_or_girl ^ num_children + prob_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end prob_at_least_one_boy_one_girl_l4110_411003


namespace unique_m_satisfying_lcm_conditions_l4110_411075

theorem unique_m_satisfying_lcm_conditions (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by
  sorry

end unique_m_satisfying_lcm_conditions_l4110_411075


namespace power_product_equality_l4110_411024

theorem power_product_equality : (0.125^8 * (-8)^7) = -0.125 := by
  sorry

end power_product_equality_l4110_411024


namespace coffee_consumption_theorem_l4110_411061

/-- Represents the relationship between coffee consumption, sleep, and work intensity -/
def coffee_relation (sleep : ℝ) (work : ℝ) (coffee : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ coffee * sleep * work = k

/-- Theorem stating the relationship between coffee consumption on two different days -/
theorem coffee_consumption_theorem (sleep_mon sleep_tue work_mon work_tue coffee_mon : ℝ) :
  sleep_mon = 8 →
  work_mon = 4 →
  coffee_mon = 1 →
  sleep_tue = 5 →
  work_tue = 7 →
  coffee_relation sleep_mon work_mon coffee_mon →
  coffee_relation sleep_tue work_tue ((32 : ℝ) / 35) :=
by sorry

end coffee_consumption_theorem_l4110_411061


namespace company_fund_problem_l4110_411042

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (initial_fund = 60 * n - 10) →  -- The fund initially contained $10 less than needed for $60 bonuses
  (initial_fund = 50 * n + 140) → -- Each employee received a $50 bonus, and $140 remained
  initial_fund = 890 := by
  sorry

end company_fund_problem_l4110_411042
