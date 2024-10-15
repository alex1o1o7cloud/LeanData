import Mathlib

namespace NUMINAMATH_CALUDE_sand_weight_difference_l1255_125501

/-- Proves that the sand in the box is heavier than the sand in the barrel by 260 grams --/
theorem sand_weight_difference 
  (barrel_weight : ℕ) 
  (barrel_with_sand_weight : ℕ) 
  (box_weight : ℕ) 
  (box_with_sand_weight : ℕ) 
  (h1 : barrel_weight = 250)
  (h2 : barrel_with_sand_weight = 1780)
  (h3 : box_weight = 460)
  (h4 : box_with_sand_weight = 2250) :
  (box_with_sand_weight - box_weight) - (barrel_with_sand_weight - barrel_weight) = 260 := by
  sorry

#check sand_weight_difference

end NUMINAMATH_CALUDE_sand_weight_difference_l1255_125501


namespace NUMINAMATH_CALUDE_cylinder_height_l1255_125554

/-- A cylinder with given lateral area and volume has height 3 -/
theorem cylinder_height (r h : ℝ) (h_positive : h > 0) (r_positive : r > 0) 
  (lateral_area : 2 * π * r * h = 12 * π) 
  (volume : π * r^2 * h = 12 * π) : h = 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l1255_125554


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_11_12_13_eight_five_eight_zero_divisible_smallest_positive_integer_divisible_by_10_11_12_13_l1255_125593

theorem smallest_divisible_by_10_11_12_13 : 
  ∀ n : ℕ, n > 0 ∧ 10 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n ∧ 13 ∣ n → n ≥ 8580 := by
  sorry

theorem eight_five_eight_zero_divisible :
  10 ∣ 8580 ∧ 11 ∣ 8580 ∧ 12 ∣ 8580 ∧ 13 ∣ 8580 := by
  sorry

theorem smallest_positive_integer_divisible_by_10_11_12_13 :
  ∃! n : ℕ, n > 0 ∧ 
    (∀ m : ℕ, m > 0 ∧ 10 ∣ m ∧ 11 ∣ m ∧ 12 ∣ m ∧ 13 ∣ m → n ≤ m) ∧
    10 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n ∧ 13 ∣ n ∧ n = 8580 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_11_12_13_eight_five_eight_zero_divisible_smallest_positive_integer_divisible_by_10_11_12_13_l1255_125593


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1255_125568

theorem necessary_not_sufficient (a b : ℝ) : 
  (a > b → a > b - 1) ∧ ¬(a > b - 1 → a > b) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1255_125568


namespace NUMINAMATH_CALUDE_question_selection_probability_l1255_125533

/-- The probability of selecting an algebra question first and a geometry question second -/
def prob_AB (total_questions : ℕ) (algebra_questions : ℕ) (geometry_questions : ℕ) : ℚ :=
  (algebra_questions : ℚ) / total_questions * (geometry_questions : ℚ) / (total_questions - 1)

/-- The probability of selecting a geometry question second given an algebra question was selected first -/
def prob_B_given_A (total_questions : ℕ) (algebra_questions : ℕ) (geometry_questions : ℕ) : ℚ :=
  (geometry_questions : ℚ) / (total_questions - 1)

theorem question_selection_probability :
  let total_questions := 5
  let algebra_questions := 2
  let geometry_questions := 3
  prob_AB total_questions algebra_questions geometry_questions = 3 / 10 ∧
  prob_B_given_A total_questions algebra_questions geometry_questions = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_question_selection_probability_l1255_125533


namespace NUMINAMATH_CALUDE_jills_salary_l1255_125511

/-- Proves that given the conditions of Jill's income allocation, her net monthly salary is $3600 -/
theorem jills_salary (salary : ℝ) 
  (h1 : salary / 5 * 0.15 = 108) : salary = 3600 := by
  sorry

end NUMINAMATH_CALUDE_jills_salary_l1255_125511


namespace NUMINAMATH_CALUDE_max_value_of_a_l1255_125510

theorem max_value_of_a (x y a : ℝ) 
  (h1 : x > 1/3) 
  (h2 : y > 1) 
  (h3 : ∀ (x y : ℝ), x > 1/3 → y > 1 → 
    (9 * x^2) / (a^2 * (y-1)) + (y^2) / (a^2 * (3*x-1)) ≥ 1) : 
  a ≤ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1255_125510


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l1255_125530

theorem cost_price_per_meter (cloth_length : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) 
  (h1 : cloth_length = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_meter = 25) : 
  (selling_price - cloth_length * profit_per_meter) / cloth_length = 80 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l1255_125530


namespace NUMINAMATH_CALUDE_no_three_digit_even_sum_27_l1255_125574

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is even -/
def is_even (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has three digits -/
def is_three_digit (n : ℕ) : Prop := sorry

theorem no_three_digit_even_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ is_even n ∧ digit_sum n = 27 := by sorry

end NUMINAMATH_CALUDE_no_three_digit_even_sum_27_l1255_125574


namespace NUMINAMATH_CALUDE_skating_speed_ratio_l1255_125556

/-- The ratio of skating speeds between a father and son -/
theorem skating_speed_ratio (v_f v_s : ℝ) (h1 : v_f > 0) (h2 : v_s > 0) 
  (h3 : v_f > v_s) (h4 : (v_f + v_s) / (v_f - v_s) = 5) : v_f / v_s = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_skating_speed_ratio_l1255_125556


namespace NUMINAMATH_CALUDE_max_inscribed_rectangle_area_l1255_125523

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-coordinate for a given x on the parabola -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Theorem: Maximum inscribed rectangle area in a parabola -/
theorem max_inscribed_rectangle_area
  (p : Parabola)
  (vertex_x : p.y_at 3 = -5)
  (point_on_parabola : p.y_at 5 = 15) :
  ∃ (area : ℝ), area = 10 ∧ 
  ∀ (rect_area : ℝ), 
    (∃ (x1 x2 : ℝ), 
      x1 < x2 ∧ 
      p.y_at x1 = 0 ∧ 
      p.y_at x2 = 0 ∧ 
      rect_area = (x2 - x1) * min (p.y_at ((x1 + x2) / 2)) 0) →
    rect_area ≤ area :=
by sorry

end NUMINAMATH_CALUDE_max_inscribed_rectangle_area_l1255_125523


namespace NUMINAMATH_CALUDE_mike_ride_distance_l1255_125516

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  base_fare : ℝ
  per_mile_rate : ℝ
  additional_fee : ℝ
  distance : ℝ

/-- Calculates the total fare for a taxi ride -/
def total_fare (ride : TaxiRide) : ℝ :=
  ride.base_fare + ride.per_mile_rate * ride.distance + ride.additional_fee

/-- Proves that Mike's ride was 42 miles long given the conditions -/
theorem mike_ride_distance (mike annie : TaxiRide) 
    (h1 : mike.base_fare = 2.5)
    (h2 : mike.per_mile_rate = 0.25)
    (h3 : mike.additional_fee = 0)
    (h4 : annie.base_fare = 2.5)
    (h5 : annie.per_mile_rate = 0.25)
    (h6 : annie.additional_fee = 5)
    (h7 : annie.distance = 22)
    (h8 : total_fare mike = total_fare annie) : mike.distance = 42 := by
  sorry

#check mike_ride_distance

end NUMINAMATH_CALUDE_mike_ride_distance_l1255_125516


namespace NUMINAMATH_CALUDE_range_of_f_l1255_125503

-- Define the function f
def f (x : ℝ) : ℝ := x - x^3

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a = -6 ∧ b = 2 * Real.sqrt 3 / 9 ∧
  (∀ y, (∃ x ∈ Set.Icc 0 2, f x = y) ↔ a ≤ y ∧ y ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1255_125503


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_l1255_125547

-- Define the conditions
def p (a x : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 5

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, p 1 x ∧ q x → 2 < x ∧ x < 4 := by sorry

-- Part 2
theorem range_of_a :
  (∀ x : ℝ, p a x → q x) ∧ (∃ x : ℝ, q x ∧ ¬(p a x)) →
  5/4 < a ∧ a ≤ 2 := by sorry

#check range_of_x_when_a_is_one
#check range_of_a

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_l1255_125547


namespace NUMINAMATH_CALUDE_power_function_property_l1255_125550

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) (h2 : f 4 = 2) : f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l1255_125550


namespace NUMINAMATH_CALUDE_xiaoming_estimate_larger_l1255_125531

/-- Rounds a number up to the nearest ten -/
def roundUp (n : ℤ) : ℤ := sorry

/-- Rounds a number down to the nearest ten -/
def roundDown (n : ℤ) : ℤ := sorry

theorem xiaoming_estimate_larger (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  roundUp x - roundDown y > x - y := by sorry

end NUMINAMATH_CALUDE_xiaoming_estimate_larger_l1255_125531


namespace NUMINAMATH_CALUDE_six_sufficient_not_necessary_l1255_125508

-- Define the binomial expansion term
def binomialTerm (n : ℕ) (r : ℕ) : ℚ → ℚ := λ x => x^(2*n - 3*r)

-- Define the condition for a constant term
def hasConstantTerm (n : ℕ) : Prop := ∃ r : ℕ, 2*n = 3*r

-- Theorem stating that n=6 is sufficient but not necessary
theorem six_sufficient_not_necessary :
  (hasConstantTerm 6) ∧ (∃ m : ℕ, m ≠ 6 ∧ hasConstantTerm m) :=
sorry

end NUMINAMATH_CALUDE_six_sufficient_not_necessary_l1255_125508


namespace NUMINAMATH_CALUDE_determinant_max_value_l1255_125513

theorem determinant_max_value (θ : ℝ) :
  (∀ θ', -Real.sin (4 * θ') / 2 ≤ -Real.sin (4 * θ) / 2) →
  -Real.sin (4 * θ) / 2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_determinant_max_value_l1255_125513


namespace NUMINAMATH_CALUDE_company_demographics_l1255_125514

theorem company_demographics (total : ℕ) (total_pos : 0 < total) :
  let men_percent : ℚ := 48 / 100
  let union_percent : ℚ := 60 / 100
  let union_men_percent : ℚ := 70 / 100
  let men := (men_percent * total).floor
  let union := (union_percent * total).floor
  let union_men := (union_men_percent * union).floor
  let non_union := total - union
  let non_union_men := men - union_men
  let non_union_women := non_union - non_union_men
  (non_union_women : ℚ) / non_union = 85 / 100 :=
by sorry

end NUMINAMATH_CALUDE_company_demographics_l1255_125514


namespace NUMINAMATH_CALUDE_second_group_size_l1255_125594

/-- Represents a tour group with a number of people -/
structure TourGroup where
  people : ℕ

/-- Represents a day's tour schedule -/
structure TourSchedule where
  group1 : TourGroup
  group2 : TourGroup
  group3 : TourGroup
  group4 : TourGroup

def questions_per_tourist : ℕ := 2

def total_questions : ℕ := 68

theorem second_group_size (schedule : TourSchedule) : 
  schedule.group1.people = 6 ∧ 
  schedule.group3.people = 8 ∧ 
  schedule.group4.people = 7 ∧
  questions_per_tourist * (schedule.group1.people + schedule.group2.people + schedule.group3.people + schedule.group4.people) = total_questions →
  schedule.group2.people = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l1255_125594


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1255_125525

theorem rectangular_field_area (w : ℝ) (d : ℝ) (h1 : w = 15) (h2 : d = 17) :
  ∃ l : ℝ, w * l = 120 ∧ d^2 = w^2 + l^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1255_125525


namespace NUMINAMATH_CALUDE_unique_solution_equation_l1255_125579

theorem unique_solution_equation : 
  ∃! (x y z : ℕ+), 1 + 2^(x.val) + 3^(y.val) = z.val^3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l1255_125579


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1255_125587

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = 6) ∧ (x + 2 * y = -2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1255_125587


namespace NUMINAMATH_CALUDE_max_exterior_sum_is_34_l1255_125535

/-- Represents a rectangular prism with a pyramid added to one face -/
structure PrismWithPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_new_faces : Nat
  pyramid_new_edges : Nat
  pyramid_new_vertices : Nat

/-- Calculates the total number of exterior elements (faces, edges, vertices) -/
def totalExteriorElements (shape : PrismWithPyramid) : Nat :=
  shape.prism_faces - 1 + shape.pyramid_new_faces +
  shape.prism_edges + shape.pyramid_new_edges +
  shape.prism_vertices + shape.pyramid_new_vertices

/-- The maximum sum of exterior faces, vertices, and edges -/
def maxExteriorSum : Nat := 34

/-- Theorem stating that the maximum sum of exterior elements is 34 -/
theorem max_exterior_sum_is_34 :
  ∀ shape : PrismWithPyramid,
    shape.prism_faces = 6 ∧
    shape.prism_edges = 12 ∧
    shape.prism_vertices = 8 ∧
    shape.pyramid_new_faces ≤ 4 ∧
    shape.pyramid_new_edges ≤ 4 ∧
    shape.pyramid_new_vertices = 1 →
    totalExteriorElements shape ≤ maxExteriorSum :=
by
  sorry


end NUMINAMATH_CALUDE_max_exterior_sum_is_34_l1255_125535


namespace NUMINAMATH_CALUDE_pipeA_rate_correct_l1255_125532

/-- Represents the rate at which Pipe A fills the tank -/
def pipeA_rate : ℝ := 40

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 750

/-- The rate at which Pipe B fills the tank in liters per minute -/
def pipeB_rate : ℝ := 30

/-- The rate at which Pipe C drains the tank in liters per minute -/
def pipeC_rate : ℝ := 20

/-- The time in minutes it takes to fill the tank -/
def fill_time : ℝ := 45

/-- The duration of one cycle in minutes -/
def cycle_duration : ℝ := 3

/-- Theorem stating that the rate of Pipe A is correct given the conditions -/
theorem pipeA_rate_correct : 
  tank_capacity = (fill_time / cycle_duration) * (pipeA_rate + pipeB_rate - pipeC_rate) :=
by sorry

end NUMINAMATH_CALUDE_pipeA_rate_correct_l1255_125532


namespace NUMINAMATH_CALUDE_parking_theorem_l1255_125588

/-- The number of parking spaces -/
def total_spaces : ℕ := 7

/-- The number of cars to be parked -/
def num_cars : ℕ := 3

/-- The number of spaces that must remain empty and connected -/
def empty_spaces : ℕ := 4

/-- The number of possible positions for the block of empty spaces -/
def empty_block_positions : ℕ := total_spaces - empty_spaces + 1

/-- The number of distinct parking arrangements -/
def parking_arrangements : ℕ := empty_block_positions * (Nat.factorial num_cars)

theorem parking_theorem : parking_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_parking_theorem_l1255_125588


namespace NUMINAMATH_CALUDE_intersection_range_l1255_125529

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0 ∧ y > 0

-- Define the line
def line (k x y : ℝ) : Prop := y = k*(x + 2)

-- Define the intersection condition
def intersects (k : ℝ) : Prop :=
  ∃ x y, curve x y ∧ line k x y

-- State the theorem
theorem intersection_range :
  ∀ k, intersects k ↔ k > 0 ∧ k ≤ 3/4 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l1255_125529


namespace NUMINAMATH_CALUDE_trapezoid_ab_length_l1255_125548

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- The sum of AB and CD is 280
  sum_sides : ab + cd = 280
  -- The ratio of areas is 5:2
  ratio_constraint : area_ratio = 5 / 2

/-- Theorem: In a trapezoid with given properties, AB = 200 -/
theorem trapezoid_ab_length (t : Trapezoid) : t.ab = 200 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_ab_length_l1255_125548


namespace NUMINAMATH_CALUDE_total_price_is_23_l1255_125528

/-- The price of cucumbers in dollars per kilogram -/
def cucumber_price : ℝ := 5

/-- The price of tomatoes in dollars per kilogram -/
def tomato_price : ℝ := cucumber_price * (1 - 0.2)

/-- The total price of tomatoes and cucumbers -/
def total_price : ℝ := 2 * tomato_price + 3 * cucumber_price

theorem total_price_is_23 : total_price = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_price_is_23_l1255_125528


namespace NUMINAMATH_CALUDE_circle_area_difference_l1255_125563

theorem circle_area_difference (π : ℝ) (h_π : π > 0) : 
  let R := 18 / π  -- Radius of larger circle
  let r := R / 2   -- Radius of smaller circle
  (π * R^2 - π * r^2) = 243 / π := by
sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1255_125563


namespace NUMINAMATH_CALUDE_sum_of_square_roots_bound_l1255_125560

theorem sum_of_square_roots_bound (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) : 
  Real.sqrt (7 * x + 3) + Real.sqrt (7 * y + 3) + Real.sqrt (7 * z + 3) ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_bound_l1255_125560


namespace NUMINAMATH_CALUDE_x_squared_minus_two_is_quadratic_l1255_125559

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - 2 = 0 is a quadratic equation -/
theorem x_squared_minus_two_is_quadratic :
  is_quadratic_equation (λ x : ℝ ↦ x^2 - 2) :=
by
  sorry


end NUMINAMATH_CALUDE_x_squared_minus_two_is_quadratic_l1255_125559


namespace NUMINAMATH_CALUDE_basketball_team_starters_l1255_125581

theorem basketball_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) :
  total_players = 12 →
  quadruplets = 4 →
  starters = 5 →
  quad_starters = 2 →
  (Nat.choose quadruplets quad_starters) * (Nat.choose (total_players - quadruplets) (starters - quad_starters)) = 336 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_starters_l1255_125581


namespace NUMINAMATH_CALUDE_first_reading_takes_15_days_l1255_125521

/-- The number of pages in the book -/
def total_pages : ℕ := 480

/-- The additional pages read per day in the second reading -/
def additional_pages_per_day : ℕ := 16

/-- The number of days saved in the second reading -/
def days_saved : ℕ := 5

/-- The number of days taken for the first reading -/
def first_reading_days : ℕ := 15

/-- The number of pages read per day in the first reading -/
def pages_per_day_first : ℕ := total_pages / first_reading_days

/-- The number of pages read per day in the second reading -/
def pages_per_day_second : ℕ := pages_per_day_first + additional_pages_per_day

/-- Theorem stating that the given conditions result in 15 days for the first reading -/
theorem first_reading_takes_15_days :
  (total_pages / pages_per_day_first = first_reading_days) ∧
  (total_pages / pages_per_day_second = first_reading_days - days_saved) :=
by sorry

end NUMINAMATH_CALUDE_first_reading_takes_15_days_l1255_125521


namespace NUMINAMATH_CALUDE_bertha_family_without_daughters_l1255_125519

/-- Represents a family tree starting from Bertha -/
structure BerthaFamily where
  daughters : Nat
  daughters_with_children : Nat
  total_descendants : Nat

/-- The conditions of Bertha's family -/
def bertha_family : BerthaFamily := {
  daughters := 6,
  daughters_with_children := 4,
  total_descendants := 30
}

/-- Theorem: The number of Bertha's daughters and granddaughters who have no daughters is 26 -/
theorem bertha_family_without_daughters : 
  (bertha_family.total_descendants - bertha_family.daughters_with_children * bertha_family.daughters) + 
  (bertha_family.daughters - bertha_family.daughters_with_children) = 26 := by
  sorry

#check bertha_family_without_daughters

end NUMINAMATH_CALUDE_bertha_family_without_daughters_l1255_125519


namespace NUMINAMATH_CALUDE_baguettes_left_at_end_of_day_l1255_125551

/-- The number of baguettes left at the end of the day in a bakery --/
def baguettes_left (batches_per_day : ℕ) (baguettes_per_batch : ℕ) 
  (sold_after_first : ℕ) (sold_after_second : ℕ) (sold_after_third : ℕ) : ℕ :=
  let total_baguettes := batches_per_day * baguettes_per_batch
  let left_after_first := baguettes_per_batch - sold_after_first
  let left_after_second := (baguettes_per_batch + left_after_first) - sold_after_second
  let left_after_third := (baguettes_per_batch + left_after_second) - sold_after_third
  left_after_third

/-- Theorem stating the number of baguettes left at the end of the day --/
theorem baguettes_left_at_end_of_day :
  baguettes_left 3 48 37 52 49 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baguettes_left_at_end_of_day_l1255_125551


namespace NUMINAMATH_CALUDE_abs_diff_eq_diff_implies_leq_l1255_125591

theorem abs_diff_eq_diff_implies_leq (x y : ℝ) : |x - y| = y - x → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_eq_diff_implies_leq_l1255_125591


namespace NUMINAMATH_CALUDE_divide_into_three_unequal_groups_divide_into_three_equal_groups_divide_among_three_people_l1255_125576

-- Define the number of books
def n : ℕ := 6

-- Theorem for the first question
theorem divide_into_three_unequal_groups :
  (n.choose 1) * ((n - 1).choose 2) * ((n - 3).choose 3) = 60 := by sorry

-- Theorem for the second question
theorem divide_into_three_equal_groups :
  (n.choose 2 * (n - 2).choose 2 * (n - 4).choose 2) / 6 = 15 := by sorry

-- Theorem for the third question
theorem divide_among_three_people :
  n.choose 2 * (n - 2).choose 2 * (n - 4).choose 2 = 90 := by sorry

end NUMINAMATH_CALUDE_divide_into_three_unequal_groups_divide_into_three_equal_groups_divide_among_three_people_l1255_125576


namespace NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l1255_125543

/-- Represents the number of bahs equivalent to 36 rahs -/
def bahs_per_36_rahs : ℕ := 24

/-- Represents the number of rahs equivalent to 18 yahs -/
def rahs_per_18_yahs : ℕ := 12

/-- Represents the number of yahs we want to convert to bahs -/
def yahs_to_convert : ℕ := 1500

/-- Theorem stating the equivalence between 1500 yahs and 667 bahs -/
theorem yahs_to_bahs_conversion :
  ∃ (bahs : ℕ), bahs = 667 ∧
  (bahs * bahs_per_36_rahs * rahs_per_18_yahs : ℚ) / 36 / 18 = yahs_to_convert / 1 :=
sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l1255_125543


namespace NUMINAMATH_CALUDE_joans_cat_kittens_l1255_125538

/-- The number of kittens Joan has now -/
def total_kittens : ℕ := 10

/-- The number of kittens Joan got from her friends -/
def kittens_from_friends : ℕ := 2

/-- The number of kittens Joan's cat had -/
def cat_kittens : ℕ := total_kittens - kittens_from_friends

theorem joans_cat_kittens : cat_kittens = 8 := by
  sorry

end NUMINAMATH_CALUDE_joans_cat_kittens_l1255_125538


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1255_125578

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 18 players, where each player plays twice with every other player, the total number of games played is 612 -/
theorem chess_tournament_games :
  tournament_games 18 * 2 = 612 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1255_125578


namespace NUMINAMATH_CALUDE_L_quotient_property_l1255_125537

/-- L(a,b) is defined as the exponent c such that a^c = b, for positive numbers a and b -/
noncomputable def L (a b : ℝ) : ℝ :=
  Real.log b / Real.log a

/-- Theorem: For positive real numbers a, m, and n, L(a, m/n) = L(a,m) - L(a,n) -/
theorem L_quotient_property (a m n : ℝ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  L a (m/n) = L a m - L a n := by
  sorry

end NUMINAMATH_CALUDE_L_quotient_property_l1255_125537


namespace NUMINAMATH_CALUDE_square_area_with_point_l1255_125512

/-- A square with a point inside satisfying certain distance conditions -/
structure SquareWithPoint where
  -- The side length of the square
  a : ℝ
  -- Coordinates of point P
  x : ℝ
  y : ℝ
  -- Conditions
  square_positive : 0 < a
  inside_square : 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a
  distance_to_A : x^2 + y^2 = 4
  distance_to_B : (a - x)^2 + y^2 = 9
  distance_to_C : (a - x)^2 + (a - y)^2 = 16

/-- The area of a square with a point inside satisfying certain distance conditions is 10 + √63 -/
theorem square_area_with_point (s : SquareWithPoint) : s.a^2 = 10 + Real.sqrt 63 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_point_l1255_125512


namespace NUMINAMATH_CALUDE_isosceles_triangle_n_value_l1255_125524

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : (side1 = side2 ∧ side3 ≠ side1) ∨ (side1 = side3 ∧ side2 ≠ side1) ∨ (side2 = side3 ∧ side1 ≠ side2)

/-- The quadratic equation x^2 - 8x + n = 0 -/
def quadratic_equation (x n : ℝ) : Prop :=
  x^2 - 8*x + n = 0

/-- Theorem statement -/
theorem isosceles_triangle_n_value :
  ∀ (t : IsoscelesTriangle) (n : ℝ),
    ((t.side1 = 3 ∨ t.side2 = 3 ∨ t.side3 = 3) ∧
     (quadratic_equation t.side1 n ∧ quadratic_equation t.side2 n) ∨
     (quadratic_equation t.side1 n ∧ quadratic_equation t.side3 n) ∨
     (quadratic_equation t.side2 n ∧ quadratic_equation t.side3 n)) →
    n = 15 ∨ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_n_value_l1255_125524


namespace NUMINAMATH_CALUDE_triangle_side_length_range_l1255_125534

theorem triangle_side_length_range : ∃ (min max : ℤ),
  (∀ x : ℤ, (x + 8 > 10 ∧ x + 10 > 8 ∧ 8 + 10 > x) → min ≤ x ∧ x ≤ max) ∧
  min = 3 ∧ max = 17 ∧ max - min = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_range_l1255_125534


namespace NUMINAMATH_CALUDE_annual_mischief_convention_handshakes_l1255_125544

/-- The number of handshakes at the Annual Mischief Convention -/
theorem annual_mischief_convention_handshakes (n_gremlins : ℕ) (n_imps : ℕ) : 
  n_gremlins = 30 → n_imps = 15 → 
  (n_gremlins * (n_gremlins - 1)) / 2 + n_imps * (n_gremlins / 2) = 660 := by
  sorry

end NUMINAMATH_CALUDE_annual_mischief_convention_handshakes_l1255_125544


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l1255_125590

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l1255_125590


namespace NUMINAMATH_CALUDE_correct_transformation_l1255_125561

theorem correct_transformation (x : ℝ) : (x / 2 - x / 3 = 1) ↔ (3 * x - 2 * x = 6) := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1255_125561


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1255_125509

theorem arithmetic_mean_problem (x : ℝ) : 
  (12 + 18 + 24 + 36 + 6 + x) / 6 = 16 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1255_125509


namespace NUMINAMATH_CALUDE_jack_email_difference_l1255_125573

theorem jack_email_difference : 
  let morning_emails : ℕ := 6
  let afternoon_emails : ℕ := 2
  morning_emails - afternoon_emails = 4 :=
by sorry

end NUMINAMATH_CALUDE_jack_email_difference_l1255_125573


namespace NUMINAMATH_CALUDE_missing_root_l1255_125552

theorem missing_root (x : ℝ) : x^2 - 2*x = 0 → (x = 2 ∨ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_missing_root_l1255_125552


namespace NUMINAMATH_CALUDE_base_five_product_theorem_l1255_125565

def base_five_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (5 ^ i)) 0

def decimal_to_base_five (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec convert (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else convert (m / 5) ((m % 5) :: acc)
  convert n []

def base_five_multiply (a b : List Nat) : List Nat :=
  decimal_to_base_five ((base_five_to_decimal a) * (base_five_to_decimal b))

theorem base_five_product_theorem :
  base_five_multiply [1, 3, 1] [2, 1] = [2, 2, 1, 2] := by sorry

end NUMINAMATH_CALUDE_base_five_product_theorem_l1255_125565


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l1255_125569

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number consisting of n repetitions of a digit d in base 10 -/
def repeatedDigit (d : ℕ) (n : ℕ) : ℕ := sorry

theorem sum_of_digits_9ab : 
  let a := repeatedDigit 6 2023
  let b := repeatedDigit 4 2023
  sumOfDigits (9 * a * b) = 20225 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l1255_125569


namespace NUMINAMATH_CALUDE_stock_price_fluctuation_l1255_125502

theorem stock_price_fluctuation (original_price : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) :
  increase_percent = 0.40 →
  decrease_percent = 2 / 7 →
  original_price * (1 + increase_percent) * (1 - decrease_percent) = original_price :=
by sorry

end NUMINAMATH_CALUDE_stock_price_fluctuation_l1255_125502


namespace NUMINAMATH_CALUDE_problem_solution_l1255_125539

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 4| + |x - 4|

-- Theorem statement
theorem problem_solution :
  -- Part 1: Solution set of f(x) ≥ 10
  (∀ x, f x ≥ 10 ↔ x ∈ Set.Iic (-10/3) ∪ Set.Ici 2) ∧
  -- Part 2: Minimum value of f(x) is 6
  (∃ x, f x = 6 ∧ ∀ y, f y ≥ f x) ∧
  -- Part 3: Inequality for positive real numbers a, b, c
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 6 →
    1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1255_125539


namespace NUMINAMATH_CALUDE_max_sum_abc_l1255_125505

theorem max_sum_abc (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1)
  (h : a * b * c + 2 * a^2 + 2 * b^2 + 2 * c^2 + c * a - c * b - 4 * a + 4 * b - c = 28) :
  a + b + c ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_sum_abc_l1255_125505


namespace NUMINAMATH_CALUDE_trapezoid_circle_properties_l1255_125570

/-- Represents a trapezoid ABCD with a circle centered at P on AB and tangent to BC and AD -/
structure Trapezoid :=
  (AB CD BC AD : ℝ)
  (AP : ℝ)
  (r : ℝ)

/-- The theorem stating the properties of the trapezoid and circle -/
theorem trapezoid_circle_properties (T : Trapezoid) :
  T.AB = 105 ∧
  T.BC = 65 ∧
  T.CD = 27 ∧
  T.AD = 80 ∧
  T.AP = 175 / 3 ∧
  T.r = 35 / 6 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_circle_properties_l1255_125570


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l1255_125536

theorem logarithmic_equation_solution (a : ℝ) (ha : a > 0) :
  ∃ x : ℝ, x > 1 ∧ Real.log (a * x) = 2 * Real.log (x - 1) ↔
  ∃ x : ℝ, x = (2 + a + Real.sqrt (a^2 + 4*a)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l1255_125536


namespace NUMINAMATH_CALUDE_distance_to_square_center_l1255_125586

-- Define the right triangle ABC
structure RightTriangle where
  a : ℝ  -- length of BC
  b : ℝ  -- length of AC
  h : 0 < a ∧ 0 < b  -- positive lengths

-- Define the square ABDE on the hypotenuse
structure SquareOnHypotenuse (t : RightTriangle) where
  center : ℝ × ℝ  -- coordinates of the center of the square

-- Theorem statement
theorem distance_to_square_center (t : RightTriangle) (s : SquareOnHypotenuse t) :
  Real.sqrt ((s.center.1 ^ 2) + (s.center.2 ^ 2)) = (t.a + t.b) / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_square_center_l1255_125586


namespace NUMINAMATH_CALUDE_find_a_value_l1255_125558

def f (x a : ℝ) : ℝ := |x + 1| + |x - a|

theorem find_a_value (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, f x a ≥ 5 ↔ x ≤ -2 ∨ x > 3) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l1255_125558


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l1255_125592

theorem average_of_abc_is_three (A B C : ℚ) 
  (eq1 : 101 * C - 202 * A = 404)
  (eq2 : 101 * B + 303 * A = 505)
  (eq3 : 101 * A + 101 * B + 101 * C = 303) :
  (A + B + C) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l1255_125592


namespace NUMINAMATH_CALUDE_fold_points_area_l1255_125520

-- Define the triangle DEF
def triangle_DEF (D E F : ℝ × ℝ) : Prop :=
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  let angle_e := Real.arccos ((de^2 + df^2 - (F.1 - E.1)^2 - (F.2 - E.2)^2) / (2 * de * df))
  de = 48 ∧ df = 96 ∧ angle_e = Real.pi / 2

-- Define the area of fold points
def area_fold_points (D E F : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem fold_points_area (D E F : ℝ × ℝ) :
  triangle_DEF D E F →
  area_fold_points D E F = 432 * Real.pi - 518 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_fold_points_area_l1255_125520


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l1255_125596

-- Define the total number of chocolate bars
def total_bars : ℕ := 9

-- Define the number of unsold bars
def unsold_bars : ℕ := 3

-- Define the total amount made from the sale
def total_amount : ℕ := 18

-- Theorem to prove
theorem chocolate_bar_cost :
  ∃ (cost : ℚ), cost * (total_bars - unsold_bars) = total_amount ∧ cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l1255_125596


namespace NUMINAMATH_CALUDE_direction_vector_k_l1255_125546

/-- The direction vector of the line passing through points A(0,2) and B(-1,0) is (1,k). -/
theorem direction_vector_k (k : ℝ) : 
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (-1, 0)
  let direction_vector : ℝ × ℝ := (1, k)
  (direction_vector.1 * (B.1 - A.1) = direction_vector.2 * (B.2 - A.2)) → k = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_direction_vector_k_l1255_125546


namespace NUMINAMATH_CALUDE_equations_represent_same_curve_l1255_125571

-- Define the two equations
def equation1 (x y : ℝ) : Prop := |y| = |x|
def equation2 (x y : ℝ) : Prop := y^2 = x^2

-- Theorem statement
theorem equations_represent_same_curve :
  ∀ (x y : ℝ), equation1 x y ↔ equation2 x y := by
  sorry

end NUMINAMATH_CALUDE_equations_represent_same_curve_l1255_125571


namespace NUMINAMATH_CALUDE_complex_magnitude_l1255_125518

theorem complex_magnitude (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : (1 - z) * i = 2) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1255_125518


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l1255_125595

theorem largest_n_satisfying_conditions : 
  ∃ (n : ℤ), n = 313 ∧ 
  (∀ (x : ℤ), x > n → 
    (¬∃ (m : ℤ), x^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ (k : ℤ), 2*x + 103 = k^2)) ∧
  (∃ (m : ℤ), n^2 = (m+1)^3 - m^3) ∧
  (∃ (k : ℤ), 2*n + 103 = k^2) := by
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l1255_125595


namespace NUMINAMATH_CALUDE_amanda_drawer_pulls_l1255_125545

/-- Proves that the number of drawer pulls Amanda is replacing is 8 --/
theorem amanda_drawer_pulls (num_cabinet_knobs : ℕ) (cost_cabinet_knob : ℚ) 
  (cost_drawer_pull : ℚ) (total_cost : ℚ) 
  (h1 : num_cabinet_knobs = 18)
  (h2 : cost_cabinet_knob = 5/2)
  (h3 : cost_drawer_pull = 4)
  (h4 : total_cost = 77) :
  (total_cost - num_cabinet_knobs * cost_cabinet_knob) / cost_drawer_pull = 8 := by
  sorry

#eval (77 : ℚ) - 18 * (5/2 : ℚ)

end NUMINAMATH_CALUDE_amanda_drawer_pulls_l1255_125545


namespace NUMINAMATH_CALUDE_paths_in_10x5_grid_avoiding_point_l1255_125598

/-- The number of paths in a grid that avoid a specific point -/
def grid_paths_avoiding_point (m n a b c d : ℕ) : ℕ :=
  Nat.choose (m + n) n - Nat.choose (a + b) b * Nat.choose ((m - a) + (n - b)) (n - b)

/-- Theorem stating the number of paths in a 10x5 grid from (0,0) to (10,5) avoiding (5,3) -/
theorem paths_in_10x5_grid_avoiding_point : 
  grid_paths_avoiding_point 10 5 5 3 5 2 = 1827 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_10x5_grid_avoiding_point_l1255_125598


namespace NUMINAMATH_CALUDE_sin_15_deg_squared_value_l1255_125599

theorem sin_15_deg_squared_value : 
  7/16 - 7/8 * (Real.sin (15 * π / 180))^2 = 7 * Real.sqrt 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_deg_squared_value_l1255_125599


namespace NUMINAMATH_CALUDE_orange_boxes_weight_l1255_125597

/-- Calculates the total weight of oranges in three boxes given their capacities, fill ratios, and orange weights. -/
theorem orange_boxes_weight (capacity1 capacity2 capacity3 : ℕ)
                            (fill1 fill2 fill3 : ℚ)
                            (weight1 weight2 weight3 : ℚ) :
  capacity1 = 80 →
  capacity2 = 50 →
  capacity3 = 60 →
  fill1 = 3/4 →
  fill2 = 3/5 →
  fill3 = 2/3 →
  weight1 = 1/4 →
  weight2 = 3/10 →
  weight3 = 2/5 →
  (capacity1 * fill1 * weight1 + capacity2 * fill2 * weight2 + capacity3 * fill3 * weight3 : ℚ) = 40 := by
  sorry

#eval (80 * (3/4) * (1/4) + 50 * (3/5) * (3/10) + 60 * (2/3) * (2/5) : ℚ)

end NUMINAMATH_CALUDE_orange_boxes_weight_l1255_125597


namespace NUMINAMATH_CALUDE_cube_edge_length_l1255_125585

/-- A prism made up of six squares -/
structure Cube where
  edge_length : ℝ
  edge_sum : ℝ

/-- The sum of the lengths of all edges is 72 cm -/
def total_edge_length (c : Cube) : Prop :=
  c.edge_sum = 72

/-- Theorem: If the sum of the lengths of all edges is 72 cm, 
    then the length of one edge is 6 cm -/
theorem cube_edge_length (c : Cube) 
    (h : total_edge_length c) : c.edge_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1255_125585


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1255_125549

theorem gcd_polynomial_and_multiple (x : ℤ) : 
  36000 ∣ x → 
  Nat.gcd ((5*x + 3)*(11*x + 2)*(6*x + 7)*(3*x + 8) : ℤ).natAbs x.natAbs = 144 := by
sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1255_125549


namespace NUMINAMATH_CALUDE_smallest_inverse_domain_l1255_125517

def g (x : ℝ) : ℝ := -3 * (x - 1)^2 + 4

theorem smallest_inverse_domain (c : ℝ) : 
  (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_domain_l1255_125517


namespace NUMINAMATH_CALUDE_yujeong_drank_most_l1255_125584

/-- Represents the amount of water drunk by each person in liters. -/
structure WaterConsumption where
  eunji : ℚ
  yujeong : ℚ
  yuna : ℚ

/-- Determines who drank the most water given the water consumption of three people. -/
def who_drank_most (consumption : WaterConsumption) : String :=
  if consumption.yujeong > consumption.eunji ∧ consumption.yujeong > consumption.yuna then
    "Yujeong"
  else if consumption.eunji > consumption.yujeong ∧ consumption.eunji > consumption.yuna then
    "Eunji"
  else
    "Yuna"

/-- Theorem stating that Yujeong drank the most water given the specific amounts. -/
theorem yujeong_drank_most :
  who_drank_most ⟨(1/2), (7/10), (6/10)⟩ = "Yujeong" := by
  sorry

#eval who_drank_most ⟨(1/2), (7/10), (6/10)⟩

end NUMINAMATH_CALUDE_yujeong_drank_most_l1255_125584


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1255_125526

theorem cubic_equation_solutions :
  let f (x : ℝ) := (10 * x - 1) ^ (1/3) + (20 * x + 1) ^ (1/3) - 3 * (5 * x) ^ (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 1/10 ∨ x = -45/973 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1255_125526


namespace NUMINAMATH_CALUDE_greatest_integer_c_l1255_125562

-- Define the numerator and denominator of the expression
def numerator (x : ℝ) : ℝ := 16 * x^3 + 5 * x^2 + 28 * x + 12
def denominator (c x : ℝ) : ℝ := x^2 + c * x + 12

-- Define the condition for the expression to have a domain of all real numbers
def has_full_domain (c : ℝ) : Prop :=
  ∀ x : ℝ, denominator c x ≠ 0

-- State the theorem
theorem greatest_integer_c :
  (∃ c : ℤ, has_full_domain (c : ℝ) ∧ 
   ∀ d : ℤ, d > c → ¬has_full_domain (d : ℝ)) ∧
  (∃ c : ℤ, c = 6 ∧ has_full_domain (c : ℝ) ∧ 
   ∀ d : ℤ, d > c → ¬has_full_domain (d : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_c_l1255_125562


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l1255_125527

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smallest circle
  let d₂ : ℝ := 4  -- diameter of middle circle
  let d₃ : ℝ := 6  -- diameter of largest circle
  let r₁ : ℝ := d₁ / 2  -- radius of smallest circle
  let r₂ : ℝ := d₂ / 2  -- radius of middle circle
  let r₃ : ℝ := d₃ / 2  -- radius of largest circle
  let A₁ : ℝ := π * r₁^2  -- area of smallest circle
  let A₂ : ℝ := π * r₂^2  -- area of middle circle
  let A₃ : ℝ := π * r₃^2  -- area of largest circle
  (A₃ - A₂) / A₁ = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l1255_125527


namespace NUMINAMATH_CALUDE_division_remainder_l1255_125522

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 3086) (h2 : divisor = 85) (h3 : quotient = 36) :
  dividend - divisor * quotient = 26 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1255_125522


namespace NUMINAMATH_CALUDE_complement_of_union_equals_set_l1255_125500

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

-- Theorem statement
theorem complement_of_union_equals_set (U A B : Set Int) :
  U = {-2, -1, 0, 1, 2, 3} →
  A = {-1, 2} →
  B = {x : Int | x^2 - 4*x + 3 = 0} →
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

-- Note: We use \ for set difference (complement) in Lean

end NUMINAMATH_CALUDE_complement_of_union_equals_set_l1255_125500


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1255_125583

-- Given identity
axiom identity (a b : ℝ) : (a - b) * (a + b) = a^2 - b^2

-- Theorem 1
theorem problem_1 : (2 - 1) * (2 + 1) = 3 := by sorry

-- Theorem 2
theorem problem_2 : (2 + 1) * (2^2 + 1) = 15 := by sorry

-- Helper function to generate the product series
def product_series (n : ℕ) : ℝ :=
  if n = 0 then 2 + 1
  else (2^(2^n) + 1) * product_series (n-1)

-- Theorem 3
theorem problem_3 : product_series 5 = 2^64 - 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1255_125583


namespace NUMINAMATH_CALUDE_exact_two_females_one_male_probability_l1255_125515

def total_contestants : ℕ := 8
def female_contestants : ℕ := 5
def male_contestants : ℕ := 3
def selected_contestants : ℕ := 3

theorem exact_two_females_one_male_probability :
  (Nat.choose female_contestants 2 * Nat.choose male_contestants 1) / 
  Nat.choose total_contestants selected_contestants = 15 / 28 := by
  sorry

end NUMINAMATH_CALUDE_exact_two_females_one_male_probability_l1255_125515


namespace NUMINAMATH_CALUDE_focal_radii_common_points_l1255_125575

/-- An ellipse and hyperbola sharing the same foci -/
structure EllipseHyperbola where
  a : ℝ  -- semi-major axis of the ellipse
  e : ℝ  -- semi-major axis of the hyperbola

/-- The focal radii of the common points of an ellipse and hyperbola sharing the same foci -/
def focal_radii (eh : EllipseHyperbola) : ℝ × ℝ :=
  (eh.a + eh.e, eh.a - eh.e)

/-- Theorem: The focal radii of the common points of an ellipse and hyperbola 
    sharing the same foci are a + e and a - e -/
theorem focal_radii_common_points (eh : EllipseHyperbola) :
  focal_radii eh = (eh.a + eh.e, eh.a - eh.e) := by
  sorry

end NUMINAMATH_CALUDE_focal_radii_common_points_l1255_125575


namespace NUMINAMATH_CALUDE_cube_root_difference_theorem_l1255_125506

theorem cube_root_difference_theorem (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (1 - x^3)^(1/3) - (1 + x^3)^(1/3) = 1) : 
  x^3 = (x^2 * (28^(1/9))) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_difference_theorem_l1255_125506


namespace NUMINAMATH_CALUDE_board_sum_l1255_125572

theorem board_sum : ∀ (numbers : List ℕ),
  (numbers.length = 9) →
  (∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 5) →
  (numbers.filter (λ n => n ≥ 2)).length ≥ 7 →
  (numbers.filter (λ n => n > 2)).length ≥ 6 →
  (numbers.filter (λ n => n ≥ 4)).length ≥ 3 →
  (numbers.filter (λ n => n ≥ 5)).length ≥ 1 →
  numbers.sum = 26 := by
sorry

end NUMINAMATH_CALUDE_board_sum_l1255_125572


namespace NUMINAMATH_CALUDE_complex_ln_def_l1255_125557

-- Define the complex logarithm function
def complex_ln (z : ℂ) : Set ℂ :=
  {w : ℂ | ∃ k : ℤ, w = Complex.log (Complex.abs z) + Complex.I * (Complex.arg z + 2 * k * Real.pi)}

-- State the theorem
theorem complex_ln_def (z : ℂ) :
  ∀ w ∈ complex_ln z, Complex.exp w = z :=
by sorry

end NUMINAMATH_CALUDE_complex_ln_def_l1255_125557


namespace NUMINAMATH_CALUDE_missing_figure_proof_l1255_125542

theorem missing_figure_proof (x : ℝ) : (0.75 / 100) * x = 0.06 ↔ x = 8 := by sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l1255_125542


namespace NUMINAMATH_CALUDE_shirt_cost_theorem_l1255_125577

theorem shirt_cost_theorem (first_shirt_cost second_shirt_cost total_cost : ℕ) : 
  first_shirt_cost = 15 →
  first_shirt_cost = second_shirt_cost + 6 →
  total_cost = first_shirt_cost + second_shirt_cost →
  total_cost = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_theorem_l1255_125577


namespace NUMINAMATH_CALUDE_pursuer_catches_target_l1255_125580

/-- Represents a point on an infinite straight line --/
structure Point where
  position : ℝ

/-- Represents a moving object on the line --/
structure MovingObject where
  initialPosition : Point
  speed : ℝ
  direction : Bool  -- True for positive direction, False for negative

/-- The pursuer (police car) --/
def pursuer : MovingObject :=
  { initialPosition := { position := 0 },
    speed := 1,
    direction := true }

/-- The target (stolen car) --/
def target : MovingObject :=
  { initialPosition := { position := 0 },  -- Initial position unknown
    speed := 0.9,
    direction := true }  -- Direction unknown

/-- Theorem stating that the pursuer will eventually catch the target --/
theorem pursuer_catches_target :
  ∃ (t : ℝ), t > 0 ∧ 
  (pursuer.initialPosition.position + t * pursuer.speed = 
   target.initialPosition.position + t * target.speed ∨
   pursuer.initialPosition.position - t * pursuer.speed = 
   target.initialPosition.position - t * target.speed) :=
sorry

end NUMINAMATH_CALUDE_pursuer_catches_target_l1255_125580


namespace NUMINAMATH_CALUDE_minimum_shoeing_time_l1255_125555

theorem minimum_shoeing_time 
  (num_blacksmiths : ℕ) 
  (num_horses : ℕ) 
  (time_per_horseshoe : ℕ) 
  (horseshoes_per_horse : ℕ) : 
  num_blacksmiths = 48 →
  num_horses = 60 →
  time_per_horseshoe = 5 →
  horseshoes_per_horse = 4 →
  (num_horses * horseshoes_per_horse * time_per_horseshoe) / num_blacksmiths = 25 :=
by sorry

end NUMINAMATH_CALUDE_minimum_shoeing_time_l1255_125555


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l1255_125582

theorem cubic_roots_sum_squares (p q r : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (∀ x, x^3 - p*x^2 + q*x - r = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  x₁^2 + x₂^2 + x₃^2 = p^2 - 2*q := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l1255_125582


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l1255_125589

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ) : 
  A = 225 →
  Nat.gcd A B = 15 →
  Nat.lcm A B = 15 * X →
  X = 15 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l1255_125589


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l1255_125504

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalPopulation : Nat
  sampleSize : Nat
  interval : Nat
  startingNumber : Nat

/-- Calculates the selected number within a given range for a systematic sampling -/
def selectedNumber (s : SystematicSampling) (rangeStart rangeEnd : Nat) : Nat :=
  let adjustedStart := (rangeStart - s.startingNumber) / s.interval * s.interval + s.startingNumber
  if adjustedStart < rangeStart then
    adjustedStart + s.interval
  else
    adjustedStart

/-- Theorem stating that for the given systematic sampling, the selected number in the range 033 to 048 is 039 -/
theorem systematic_sampling_result :
  let s : SystematicSampling := {
    totalPopulation := 800,
    sampleSize := 50,
    interval := 16,
    startingNumber := 7
  }
  selectedNumber s 33 48 = 39 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_result_l1255_125504


namespace NUMINAMATH_CALUDE_solution_set_equiv_l1255_125553

/-- The solution set of ax^2 + 2ax > 0 -/
def SolutionSet (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * a * x > 0}

/-- The proposition that 0 < a < 1 -/
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

/-- The theorem stating that q is necessary and sufficient for the solution set to be ℝ -/
theorem solution_set_equiv (a : ℝ) : SolutionSet a = Set.univ ↔ q a := by sorry

end NUMINAMATH_CALUDE_solution_set_equiv_l1255_125553


namespace NUMINAMATH_CALUDE_radii_and_circles_regions_l1255_125564

/-- The number of regions created by radii and concentric circles inside a larger circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem radii_and_circles_regions :
  num_regions 16 10 = 176 := by
  sorry

end NUMINAMATH_CALUDE_radii_and_circles_regions_l1255_125564


namespace NUMINAMATH_CALUDE_molecular_weight_C4H10_is_58_12_l1255_125541

/-- The atomic weight of carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The number of carbon atoms in C4H10 -/
def carbon_count : ℕ := 4

/-- The number of hydrogen atoms in C4H10 -/
def hydrogen_count : ℕ := 10

/-- The molecular weight of C4H10 in atomic mass units (amu) -/
def molecular_weight_C4H10 : ℝ := carbon_weight * carbon_count + hydrogen_weight * hydrogen_count

/-- Theorem stating that the molecular weight of C4H10 is 58.12 amu -/
theorem molecular_weight_C4H10_is_58_12 : 
  molecular_weight_C4H10 = 58.12 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_C4H10_is_58_12_l1255_125541


namespace NUMINAMATH_CALUDE_function_range_l1255_125566

/-- Given a function f(x) = x^2 + ax + 3 - a, if f(x) ≥ 0 for all x in [-2, 2],
    then a is in the range [-7, 2]. -/
theorem function_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + a*x + 3 - a ≥ 0) → 
  a ∈ Set.Icc (-7 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1255_125566


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1255_125507

theorem quadratic_equation_equivalence :
  ∀ (x : ℝ), 3 * x^2 + 1 = 6 * x ↔ 3 * x^2 - 6 * x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1255_125507


namespace NUMINAMATH_CALUDE_inverse_65_mod_66_l1255_125567

theorem inverse_65_mod_66 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 65 ∧ (65 * x) % 66 = 1 :=
by
  use 65
  sorry

end NUMINAMATH_CALUDE_inverse_65_mod_66_l1255_125567


namespace NUMINAMATH_CALUDE_smallest_band_size_l1255_125540

theorem smallest_band_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 6 = 5 ∧ 
  n % 5 = 4 ∧ 
  n % 7 = 6 ∧ 
  (∀ m : ℕ, m > 0 → m % 6 = 5 → m % 5 = 4 → m % 7 = 6 → m ≥ n) ∧
  n = 119 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_band_size_l1255_125540
