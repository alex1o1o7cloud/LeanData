import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_area_l3696_369606

def u : Fin 3 → ℝ := ![4, 2, -3]
def v : Fin 3 → ℝ := ![2, -4, 5]

theorem parallelogram_area : 
  Real.sqrt ((u 0 * v 1 - u 1 * v 0)^2 + (u 0 * v 2 - u 2 * v 0)^2 + (u 1 * v 2 - u 2 * v 1)^2) = 20 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3696_369606


namespace NUMINAMATH_CALUDE_equation_solution_l3696_369630

theorem equation_solution : ∃ x : ℚ, (x - 2)^2 - (x + 3)*(x - 3) = 4*x - 1 ∧ x = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3696_369630


namespace NUMINAMATH_CALUDE_rectangular_box_width_l3696_369604

/-- Proves that the width of rectangular boxes is 5 cm given the conditions of the problem -/
theorem rectangular_box_width (wooden_length wooden_width wooden_height : ℕ)
                               (box_length box_height : ℕ)
                               (max_boxes : ℕ) :
  wooden_length = 800 →
  wooden_width = 1000 →
  wooden_height = 600 →
  box_length = 4 →
  box_height = 6 →
  max_boxes = 4000000 →
  ∃ (box_width : ℕ),
    box_width = 5 ∧
    wooden_length * wooden_width * wooden_height =
    max_boxes * (box_length * box_width * box_height) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_width_l3696_369604


namespace NUMINAMATH_CALUDE_stating_river_width_determination_l3696_369602

/-- Represents the width of a river and the meeting points of two ferries --/
structure RiverCrossing where
  width : ℝ
  first_meeting : ℝ
  second_meeting : ℝ

/-- 
Theorem stating that if two ferries meet at specific points during their crossings, 
the river width can be determined.
-/
theorem river_width_determination (r : RiverCrossing) 
  (h1 : r.first_meeting = 720)
  (h2 : r.second_meeting = 400)
  (h3 : r.first_meeting + (r.width - r.first_meeting) = r.width)
  (h4 : r.width + r.second_meeting = 3 * r.first_meeting) :
  r.width = 1760 := by
  sorry

#check river_width_determination

end NUMINAMATH_CALUDE_stating_river_width_determination_l3696_369602


namespace NUMINAMATH_CALUDE_equation_solutions_l3696_369693

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, 3*x*(x - 2) = x - 2 ↔ x = 2 ∨ x = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3696_369693


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3696_369636

/-- A sequence is geometric if there exists a constant r such that a_{n+1} = r * a_n for all n -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The property that a_n^2 = a_{n-1} * a_{n+1} for all n -/
def HasSquareProperty (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_property :
  (∀ a : ℕ → ℝ, IsGeometricSequence a → HasSquareProperty a) ∧
  (∃ a : ℕ → ℝ, HasSquareProperty a ∧ ¬IsGeometricSequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3696_369636


namespace NUMINAMATH_CALUDE_bill_amount_calculation_l3696_369699

/-- Given a true discount and a banker's discount, calculate the amount of the bill. -/
def billAmount (trueDiscount : ℚ) (bankersDiscount : ℚ) : ℚ :=
  trueDiscount + trueDiscount

/-- Theorem: Given a true discount of 360 and a banker's discount of 428.21, the amount of the bill is 720. -/
theorem bill_amount_calculation :
  let trueDiscount : ℚ := 360
  let bankersDiscount : ℚ := 428.21
  billAmount trueDiscount bankersDiscount = 720 := by
  sorry

#eval billAmount 360 428.21

end NUMINAMATH_CALUDE_bill_amount_calculation_l3696_369699


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3696_369607

theorem product_of_three_numbers (x y z : ℚ) 
  (sum_eq : x + y + z = 30)
  (first_eq : x = 3 * (y + z))
  (second_eq : y = 6 * z) :
  x * y * z = 23625 / 686 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3696_369607


namespace NUMINAMATH_CALUDE_material_left_proof_l3696_369627

theorem material_left_proof (material1 material2 material_used : ℚ) :
  material1 = 4 / 17 →
  material2 = 3 / 10 →
  material_used = 0.23529411764705882 →
  material1 + material2 - material_used = 51 / 170 := by
  sorry

end NUMINAMATH_CALUDE_material_left_proof_l3696_369627


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3696_369666

theorem quadratic_roots_sum (m n p : ℤ) : 
  (∃ x : ℝ, 4 * x * (2 * x - 5) = -4) →
  (∃ x : ℝ, x = (m + Real.sqrt n : ℝ) / p ∧ 4 * x * (2 * x - 5) = -4) →
  (∃ x : ℝ, x = (m - Real.sqrt n : ℝ) / p ∧ 4 * x * (2 * x - 5) = -4) →
  m + n + p = 26 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3696_369666


namespace NUMINAMATH_CALUDE_cars_return_to_start_l3696_369675

/-- Represents the state of cars on a circular track -/
def TrackState (n : ℕ) := Fin n → Fin n

/-- The permutation of car positions after one hour -/
def hourlyPermutation (n : ℕ) : TrackState n → TrackState n := sorry

/-- Theorem: There exists a time when all cars return to their original positions -/
theorem cars_return_to_start (n : ℕ) : 
  ∃ d : ℕ+, ∀ initial : TrackState n, (hourlyPermutation n)^[d] initial = initial := by
  sorry


end NUMINAMATH_CALUDE_cars_return_to_start_l3696_369675


namespace NUMINAMATH_CALUDE_faster_walking_speed_l3696_369629

theorem faster_walking_speed 
  (actual_speed : ℝ) 
  (actual_distance : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_speed = 8) 
  (h2 : actual_distance = 40) 
  (h3 : additional_distance = 20) : 
  ∃ (faster_speed : ℝ), 
    faster_speed = (actual_distance + additional_distance) / (actual_distance / actual_speed) ∧ 
    faster_speed = 12 := by
  sorry


end NUMINAMATH_CALUDE_faster_walking_speed_l3696_369629


namespace NUMINAMATH_CALUDE_dolphin_training_hours_l3696_369678

theorem dolphin_training_hours 
  (num_dolphins : ℕ) 
  (hours_per_dolphin : ℕ) 
  (num_trainers : ℕ) 
  (h1 : num_dolphins = 4)
  (h2 : hours_per_dolphin = 3)
  (h3 : num_trainers = 2)
  : (num_dolphins * hours_per_dolphin) / num_trainers = 6 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_training_hours_l3696_369678


namespace NUMINAMATH_CALUDE_power_multiplication_l3696_369694

theorem power_multiplication (a b : ℕ) : (10 : ℕ) ^ 655 * (10 : ℕ) ^ 652 = (10 : ℕ) ^ (655 + 652) := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3696_369694


namespace NUMINAMATH_CALUDE_circle_center_l3696_369647

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Theorem statement
theorem circle_center :
  ∃ (c : ℝ × ℝ), (c.1 = 1 ∧ c.2 = 0) ∧
  ∀ (x y : ℝ), circle_equation x y ↔ (x - c.1)^2 + (y - c.2)^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3696_369647


namespace NUMINAMATH_CALUDE_additional_spend_for_free_delivery_l3696_369641

/-- The minimum amount required for free delivery -/
def min_for_free_delivery : ℚ := 35

/-- The price of chicken per pound -/
def chicken_price_per_pound : ℚ := 6

/-- The amount of chicken in pounds -/
def chicken_amount : ℚ := 3/2

/-- The price of lettuce -/
def lettuce_price : ℚ := 3

/-- The price of cherry tomatoes -/
def cherry_tomatoes_price : ℚ := 5/2

/-- The price of a sweet potato -/
def sweet_potato_price : ℚ := 3/4

/-- The number of sweet potatoes -/
def sweet_potato_count : ℕ := 4

/-- The price of a broccoli head -/
def broccoli_price : ℚ := 2

/-- The number of broccoli heads -/
def broccoli_count : ℕ := 2

/-- The price of Brussel sprouts -/
def brussel_sprouts_price : ℚ := 5/2

/-- The total cost of items in Alice's cart -/
def cart_total : ℚ :=
  chicken_price_per_pound * chicken_amount +
  lettuce_price +
  cherry_tomatoes_price +
  sweet_potato_price * sweet_potato_count +
  broccoli_price * broccoli_count +
  brussel_sprouts_price

/-- The theorem stating how much more Alice needs to spend for free delivery -/
theorem additional_spend_for_free_delivery :
  min_for_free_delivery - cart_total = 11 := by sorry

end NUMINAMATH_CALUDE_additional_spend_for_free_delivery_l3696_369641


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3696_369686

theorem no_integer_solutions : 
  ¬ ∃ (m n : ℤ), m^3 + n^4 + 130*m*n = 35^3 ∧ m*n ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3696_369686


namespace NUMINAMATH_CALUDE_bank_savings_exceed_two_dollars_l3696_369670

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem bank_savings_exceed_two_dollars :
  let a : ℚ := 1/100  -- 1 cent in dollars
  let r : ℚ := 2      -- doubling each day
  (geometric_sum a r 8 > 2) ∧ (geometric_sum a r 7 ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_bank_savings_exceed_two_dollars_l3696_369670


namespace NUMINAMATH_CALUDE_triangle_inequality_ac_not_fourteen_l3696_369619

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  c < a + b ∧ b < a + c ∧ a < b + c :=
sorry

theorem ac_not_fourteen (ab bc : ℝ) (hab : ab = 5) (hbc : bc = 8) :
  ¬ (∃ (ac : ℝ), ac = 14 ∧ 
    (ac < ab + bc ∧ bc < ab + ac ∧ ab < bc + ac) ∧
    (0 < ab ∧ 0 < bc ∧ 0 < ac)) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_ac_not_fourteen_l3696_369619


namespace NUMINAMATH_CALUDE_train_travel_rate_l3696_369632

/-- Given a train's travel information, prove the rate of additional hours per mile -/
theorem train_travel_rate (initial_distance : ℝ) (initial_time : ℝ) 
  (additional_distance : ℝ) (additional_time : ℝ) 
  (h1 : initial_distance = 360) 
  (h2 : initial_time = 3) 
  (h3 : additional_distance = 240) 
  (h4 : additional_time = 2) :
  (additional_time / additional_distance) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_rate_l3696_369632


namespace NUMINAMATH_CALUDE_boris_candy_distribution_l3696_369688

/-- Given the initial conditions of Boris's candy distribution, 
    prove that the final number of pieces in each bowl is 83. -/
theorem boris_candy_distribution (initial_candy : ℕ) 
  (daughter_eats : ℕ) (set_aside : ℕ) (num_bowls : ℕ) (take_away : ℕ) :
  initial_candy = 300 →
  daughter_eats = 25 →
  set_aside = 10 →
  num_bowls = 6 →
  take_away = 5 →
  let remaining := initial_candy - daughter_eats - set_aside
  let per_bowl := remaining / num_bowls
  let doubled := per_bowl * 2
  doubled - take_away = 83 := by
  sorry

end NUMINAMATH_CALUDE_boris_candy_distribution_l3696_369688


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l3696_369618

theorem sum_of_number_and_its_square (x : ℝ) : x = 18 → x + x^2 = 342 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l3696_369618


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l3696_369625

/-- Given a triangle XYZ with points D on XY and E on YZ satisfying certain ratios,
    prove that DE:EF = 1:4 when DE intersects XZ at F. -/
theorem triangle_ratio_theorem (X Y Z D E F : ℝ × ℝ) : 
  -- Triangle XYZ exists
  (∃ (a b c : ℝ), X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X) →
  -- D is on XY with XD:DY = 4:1
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ D = (1 - t) • X + t • Y ∧ t = 1/5) →
  -- E is on YZ with YE:EZ = 4:1
  (∃ s : ℝ, s ∈ Set.Icc 0 1 ∧ E = (1 - s) • Y + s • Z ∧ s = 4/5) →
  -- DE intersects XZ at F
  (∃ r : ℝ, F = (1 - r) • D + r • E ∧ 
            ∃ q : ℝ, F = (1 - q) • X + q • Z) →
  -- Then DE:EF = 1:4
  ‖E - D‖ / ‖F - E‖ = 1/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l3696_369625


namespace NUMINAMATH_CALUDE_cubic_function_range_l3696_369660

/-- Given a cubic function f(x) = ax³ + bx satisfying certain conditions,
    prove that its range is [-2, 18] -/
theorem cubic_function_range (a b : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a * x^3 + b * x)
    (h_point : f 2 = 2) (h_slope : (fun x ↦ 3 * a * x^2 + b) 2 = 9) :
    Set.range f = Set.Icc (-2) 18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_range_l3696_369660


namespace NUMINAMATH_CALUDE_cattle_transport_speed_l3696_369633

/-- Proves that the speed of a truck transporting cattle is 60 miles per hour given specific conditions -/
theorem cattle_transport_speed (total_cattle : ℕ) (distance : ℕ) (truck_capacity : ℕ) (total_time : ℕ) :
  total_cattle = 400 →
  distance = 60 →
  truck_capacity = 20 →
  total_time = 40 →
  (distance * 2 * (total_cattle / truck_capacity)) / total_time = 60 := by
  sorry

#check cattle_transport_speed

end NUMINAMATH_CALUDE_cattle_transport_speed_l3696_369633


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l3696_369601

/-- A circle tangent to the coordinate axes and passing through (2, 1) -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = center.2
  passes_through : (2 - center.1)^2 + (1 - center.2)^2 = radius^2

/-- The equation of the circle -/
def circle_equation (c : TangentCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Theorem stating the possible equations of the circle -/
theorem tangent_circle_equation :
  ∀ c : TangentCircle,
  (∀ x y : ℝ, circle_equation c x y ↔ (x - 5)^2 + (y - 5)^2 = 25) ∨
  (∀ x y : ℝ, circle_equation c x y ↔ (x - 1)^2 + (y - 1)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l3696_369601


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_inequality_l3696_369680

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 2*a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 2} = {x : ℝ | x < 1/2 ∨ x > 5/2} := by sorry

-- Part II
theorem range_of_a_given_inequality (a : ℝ) :
  (∀ x : ℝ, f a x ≥ a^2 - 3*a - 3) → a ∈ Set.Icc (-1) (2 + Real.sqrt 7) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_inequality_l3696_369680


namespace NUMINAMATH_CALUDE_bijection_property_l3696_369646

theorem bijection_property (k : ℕ) (f : ℤ → ℤ) 
  (h_bij : Function.Bijective f)
  (h_prop : ∀ i j : ℤ, |i - j| ≤ k → |f i - f j| ≤ k) :
  ∀ i j : ℤ, |f i - f j| = |i - j| := by
  sorry

end NUMINAMATH_CALUDE_bijection_property_l3696_369646


namespace NUMINAMATH_CALUDE_function_value_at_2017_l3696_369658

/-- Given a function f(x) = x^2 - x * f'(0) - 1, prove that f(2017) = 2016 * 2018 -/
theorem function_value_at_2017 (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - x * (deriv f 0) - 1) : 
  f 2017 = 2016 * 2018 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_2017_l3696_369658


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l3696_369653

theorem chocolate_gain_percent (C S : ℝ) (h : 24 * C = 16 * S) : 
  (S - C) / C * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l3696_369653


namespace NUMINAMATH_CALUDE_boat_cost_correct_l3696_369612

/-- The cost of taking a boat to the Island of Mysteries -/
def boat_cost : ℚ := 254

/-- The cost of taking a plane to the Island of Mysteries -/
def plane_cost : ℚ := 600

/-- The amount saved by taking the boat instead of the plane -/
def savings : ℚ := 346

/-- Theorem stating that the boat cost is correct given the plane cost and savings -/
theorem boat_cost_correct : boat_cost = plane_cost - savings := by sorry

end NUMINAMATH_CALUDE_boat_cost_correct_l3696_369612


namespace NUMINAMATH_CALUDE_bus_stop_timing_l3696_369684

theorem bus_stop_timing (distance : ℝ) (speed1 speed2 : ℝ) (T : ℝ) : 
  distance = 9.999999999999993 →
  speed1 = 5 →
  speed2 = 6 →
  distance / speed1 * 60 - distance / speed2 * 60 = 2 * T →
  T = 10 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_timing_l3696_369684


namespace NUMINAMATH_CALUDE_delores_initial_money_l3696_369698

/-- Calculates the final price of an item after applying discount and sales tax -/
def finalPrice (originalPrice discount salesTax : ℚ) : ℚ :=
  (originalPrice * (1 - discount)) * (1 + salesTax)

/-- Represents the problem of calculating Delores' initial amount of money -/
theorem delores_initial_money (computerPrice printerPrice headphonesPrice : ℚ)
  (computerDiscount computerTax printerTax headphonesTax leftoverMoney : ℚ) :
  computerPrice = 400 →
  printerPrice = 40 →
  headphonesPrice = 60 →
  computerDiscount = 0.1 →
  computerTax = 0.08 →
  printerTax = 0.05 →
  headphonesTax = 0.06 →
  leftoverMoney = 10 →
  ∃ initialMoney : ℚ,
    initialMoney = 
      finalPrice computerPrice computerDiscount computerTax +
      finalPrice printerPrice 0 printerTax +
      finalPrice headphonesPrice 0 headphonesTax +
      leftoverMoney ∧
    initialMoney = 504.4 := by
  sorry

end NUMINAMATH_CALUDE_delores_initial_money_l3696_369698


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l3696_369673

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_7 : a 7 = 8)
  (h_23 : a 23 = 22) :
  a 15 = 15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l3696_369673


namespace NUMINAMATH_CALUDE_joyce_land_theorem_l3696_369668

/-- Calculates the suitable land for growing vegetables given the previous property size,
    the factor by which the new property is larger, and the size of the pond. -/
def suitable_land (previous_property : ℝ) (size_factor : ℝ) (pond_size : ℝ) : ℝ :=
  previous_property * size_factor - pond_size

/-- Theorem stating that given a previous property of 2 acres, a new property 8 times larger,
    and a 3-acre pond, the land suitable for growing vegetables is 13 acres. -/
theorem joyce_land_theorem :
  suitable_land 2 8 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_joyce_land_theorem_l3696_369668


namespace NUMINAMATH_CALUDE_angie_leftover_money_l3696_369661

def angie_finances (salary : ℕ) (necessities : ℕ) (taxes : ℕ) : ℕ :=
  salary - (necessities + taxes)

theorem angie_leftover_money :
  angie_finances 80 42 20 = 18 := by sorry

end NUMINAMATH_CALUDE_angie_leftover_money_l3696_369661


namespace NUMINAMATH_CALUDE_toy_cars_in_second_box_l3696_369610

theorem toy_cars_in_second_box :
  let total_boxes : ℕ := 3
  let cars_in_first_box : ℕ := 21
  let cars_in_third_box : ℕ := 19
  let total_cars : ℕ := 71
  let cars_in_second_box : ℕ := total_cars - cars_in_first_box - cars_in_third_box
  cars_in_second_box = 31 := by
  sorry

end NUMINAMATH_CALUDE_toy_cars_in_second_box_l3696_369610


namespace NUMINAMATH_CALUDE_intersection_A_B_l3696_369626

-- Define set A
def A : Set ℝ := {x | ∃ y, (x^2)/4 + (3*y^2)/4 = 1}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = x^2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3696_369626


namespace NUMINAMATH_CALUDE_subset_condition_l3696_369683

theorem subset_condition (a : ℝ) : 
  let A := {x : ℝ | |x - (a+1)^2/2| ≤ (a-1)^2/2}
  let B := {x : ℝ | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0}
  (A ⊆ B) ↔ (1 ≤ a ∧ a ≤ 3) ∨ a = -1 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l3696_369683


namespace NUMINAMATH_CALUDE_series_sum_l3696_369643

def series (n : ℕ) : ℚ :=
  if n = 0 then 2
  else if n = 1 then 3 + (1/2) * (series 0)
  else (1005 - n + 1 : ℚ) + (1/2) * (series (n-1))

theorem series_sum : series 1003 = 2008 := by sorry

end NUMINAMATH_CALUDE_series_sum_l3696_369643


namespace NUMINAMATH_CALUDE_fraction_equality_l3696_369664

theorem fraction_equality (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (1/x - 1/y) / (1/x + 1/y) = 2023 → (x + y) / (x - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3696_369664


namespace NUMINAMATH_CALUDE_product_of_roots_l3696_369640

theorem product_of_roots (x : ℝ) : 
  let equation := (16 : ℝ) * x^2 + 60 * x - 200
  let product_of_roots := -200 / 16
  equation = 0 → product_of_roots = -(25 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3696_369640


namespace NUMINAMATH_CALUDE_intersection_theorem_union_theorem_l3696_369648

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

theorem intersection_theorem : 
  A ∩ B a = {2} → a = -1 ∨ a = -3 := by sorry

theorem union_theorem : 
  A ∪ B a = A → a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_union_theorem_l3696_369648


namespace NUMINAMATH_CALUDE_fold_line_length_squared_l3696_369696

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  let d_AB := ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2)^(1/2)
  let d_BC := ((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2)^(1/2)
  let d_CA := ((t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2)^(1/2)
  d_AB = d_BC ∧ d_BC = d_CA

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)^(1/2)

/-- Theorem: The square of the length of the fold line in the given triangle problem -/
theorem fold_line_length_squared (t : Triangle) (P : Point) :
  isEquilateral t →
  distance t.A t.B = 15 →
  distance t.B P = 11 →
  P.x = t.B.x + 11 * (t.C.x - t.B.x) / 15 →
  P.y = t.B.y + 11 * (t.C.y - t.B.y) / 15 →
  ∃ Q : Point,
    Q.x = t.A.x + (P.x - t.A.x) / 2 ∧
    Q.y = t.A.y + (P.y - t.A.y) / 2 ∧
    (distance Q P)^2 = 1043281 / 31109 :=
sorry

end NUMINAMATH_CALUDE_fold_line_length_squared_l3696_369696


namespace NUMINAMATH_CALUDE_mass_BaSO4_produced_l3696_369692

-- Define the molar masses of elements (in g/mol)
def molar_mass_Ba : ℝ := 137.327
def molar_mass_S : ℝ := 32.065
def molar_mass_O : ℝ := 15.999

-- Define the molar mass of Barium sulfate
def molar_mass_BaSO4 : ℝ := molar_mass_Ba + molar_mass_S + 4 * molar_mass_O

-- Define the number of moles of Barium bromide
def moles_BaBr2 : ℝ := 4

-- Theorem statement
theorem mass_BaSO4_produced (excess_Na2SO4 : Prop) (double_displacement : Prop) :
  moles_BaBr2 * molar_mass_BaSO4 = 933.552 := by
  sorry


end NUMINAMATH_CALUDE_mass_BaSO4_produced_l3696_369692


namespace NUMINAMATH_CALUDE_minimum_students_l3696_369603

theorem minimum_students (boys girls : ℕ) : 
  boys > 0 → 
  girls > 0 → 
  (3 * boys) / 4 = (2 * girls) / 3 → 
  ∃ (total : ℕ), total = boys + girls ∧ total ≥ 17 ∧ 
    ∀ (b g : ℕ), b > 0 → g > 0 → (3 * b) / 4 = (2 * g) / 3 → b + g ≥ total :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_students_l3696_369603


namespace NUMINAMATH_CALUDE_triangle_right_angled_l3696_369613

theorem triangle_right_angled (A B C : Real) : 
  (Real.sin A) ^ 2 + (Real.sin B) ^ 2 + (Real.sin C) ^ 2 = 2 * ((Real.cos A) ^ 2 + (Real.cos B) ^ 2 + (Real.cos C) ^ 2) → 
  A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l3696_369613


namespace NUMINAMATH_CALUDE_max_value_of_fraction_difference_l3696_369679

theorem max_value_of_fraction_difference (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 4 * a - b ≥ 2) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 * x - y ≥ 2 ∧ 1 / x - 1 / y ≤ 1 / 2) ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 * x - y ≥ 2 ∧ 1 / x - 1 / y = 1 / 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_difference_l3696_369679


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l3696_369687

theorem greatest_integer_inequality :
  ∀ x : ℤ, (3 * x + 2 < 7 - 2 * x) → x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l3696_369687


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_y_l3696_369677

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
structure FinalMixture where
  x_percentage : ℝ
  y_percentage : ℝ
  ryegrass_percentage : ℝ

/-- Theorem stating the percentage of ryegrass in seed mixture Y -/
theorem ryegrass_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (final : FinalMixture)
  (hx_ryegrass : x.ryegrass = 0.4)
  (hx_bluegrass : x.bluegrass = 0.6)
  (hy_fescue : y.fescue = 0.75)
  (hfinal_x : final.x_percentage = 0.13333333333333332)
  (hfinal_y : final.y_percentage = 1 - final.x_percentage)
  (hfinal_ryegrass : final.ryegrass_percentage = 0.27)
  : y.ryegrass = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ryegrass_percentage_in_y_l3696_369677


namespace NUMINAMATH_CALUDE_garden_tiles_count_l3696_369656

/-- Represents a square garden covered with square tiles -/
structure SquareGarden where
  side_length : ℕ
  diagonal_tiles : ℕ

/-- The total number of tiles in a square garden -/
def total_tiles (garden : SquareGarden) : ℕ :=
  garden.side_length * garden.side_length

/-- The number of tiles on both diagonals of a square garden -/
def diagonal_tiles_count (garden : SquareGarden) : ℕ :=
  2 * garden.side_length - 1

theorem garden_tiles_count (garden : SquareGarden) 
  (h : diagonal_tiles_count garden = 25) : 
  total_tiles garden = 169 := by
  sorry

end NUMINAMATH_CALUDE_garden_tiles_count_l3696_369656


namespace NUMINAMATH_CALUDE_absolute_value_four_l3696_369645

theorem absolute_value_four (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_four_l3696_369645


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l3696_369691

theorem solution_set_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, ((a * x - 1) * (x + 2) > 0) ↔ (-3 < x ∧ x < -2)) →
  a = -1/3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l3696_369691


namespace NUMINAMATH_CALUDE_one_prime_in_alternating_series_l3696_369639

/-- The nth number in the alternating 1-0 series -/
def A (n : ℕ) : ℕ := 
  (10^(2*n) - 1) / 99

/-- The series of alternating 1-0 numbers -/
def alternating_series : Set ℕ :=
  {x | ∃ n : ℕ, x = A n}

/-- Theorem: There is exactly one prime number in the alternating 1-0 series -/
theorem one_prime_in_alternating_series : 
  ∃! p, p ∈ alternating_series ∧ Nat.Prime p :=
sorry

end NUMINAMATH_CALUDE_one_prime_in_alternating_series_l3696_369639


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3696_369634

def is_odd_prime (n : ℕ) : Prop := Nat.Prime n ∧ n % 2 = 1

def is_scalene_triangle (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    is_odd_prime a →
    is_odd_prime b →
    is_odd_prime c →
    is_scalene_triangle a b c →
    Nat.Prime (a + b + c) →
    a + b + c ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3696_369634


namespace NUMINAMATH_CALUDE_rounding_effect_on_expression_l3696_369609

theorem rounding_effect_on_expression (a b c a' b' c' : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha' : a' ≥ a) (hb' : b' ≤ b) (hc' : c' ≤ c) : 
  2 * (a' / b') + 2 * c' > 2 * (a / b) + 2 * c :=
sorry

end NUMINAMATH_CALUDE_rounding_effect_on_expression_l3696_369609


namespace NUMINAMATH_CALUDE_frog_jump_distance_l3696_369623

/-- The distance jumped by the grasshopper in inches -/
def grasshopper_jump : ℕ := 19

/-- The difference between the grasshopper's jump and the frog's jump in inches -/
def grasshopper_frog_diff : ℕ := 4

/-- The difference between the frog's jump and the mouse's jump in inches -/
def frog_mouse_diff : ℕ := 44

/-- The distance jumped by the frog in inches -/
def frog_jump : ℕ := grasshopper_jump - grasshopper_frog_diff

theorem frog_jump_distance : frog_jump = 15 := by sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l3696_369623


namespace NUMINAMATH_CALUDE_recycling_problem_l3696_369615

/-- Given a total number of cans and a number of bags, calculates the number of cans per bag -/
def cans_per_bag (total_cans : ℕ) (num_bags : ℕ) : ℕ :=
  total_cans / num_bags

theorem recycling_problem (total_cans : ℕ) (num_bags : ℕ) 
  (h1 : total_cans = 122) (h2 : num_bags = 2) : 
  cans_per_bag total_cans num_bags = 61 := by
  sorry

end NUMINAMATH_CALUDE_recycling_problem_l3696_369615


namespace NUMINAMATH_CALUDE_num_intersection_points_is_correct_l3696_369642

/-- The number of distinct intersection points of two equations -/
def num_intersection_points : ℕ := 3

/-- First equation -/
def equation1 (x y : ℝ) : Prop :=
  (x - y + 3) * (2*x + 3*y - 9) = 0

/-- Second equation -/
def equation2 (x y : ℝ) : Prop :=
  (2*x - y + 2) * (x + 3*y - 6) = 0

/-- A point satisfies both equations -/
def is_intersection_point (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

theorem num_intersection_points_is_correct :
  ∃ (points : Finset (ℝ × ℝ)),
    points.card = num_intersection_points ∧
    (∀ p ∈ points, is_intersection_point p) ∧
    (∀ p : ℝ × ℝ, is_intersection_point p → p ∈ points) :=
  sorry

end NUMINAMATH_CALUDE_num_intersection_points_is_correct_l3696_369642


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a1_l3696_369614

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a1 (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 9)
  (h_a3_a2 : 2 * a 3 - a 2 = 6) :
  a 1 = -3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a1_l3696_369614


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3696_369637

/-- A geometric sequence with a_2 = 2 and a_5 = 1/4 has a common ratio of 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1))
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1/4) :
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3696_369637


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3696_369649

theorem quadratic_inequality (x : ℝ) : x^2 + 3*x - 18 > 0 ↔ x < -6 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3696_369649


namespace NUMINAMATH_CALUDE_final_sign_is_minus_l3696_369654

/-- Represents the state of the board with plus and minus signs -/
structure BoardState where
  plus_count : Nat
  minus_count : Nat

/-- Represents an operation on the board -/
inductive Operation
  | same_sign
  | different_sign

/-- Applies an operation to the board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.same_sign => 
      if state.plus_count ≥ 2 then 
        { plus_count := state.plus_count - 1, minus_count := state.minus_count }
      else 
        { plus_count := state.plus_count + 1, minus_count := state.minus_count - 2 }
  | Operation.different_sign => 
      { plus_count := state.plus_count - 1, minus_count := state.minus_count }

/-- Theorem: After 24 operations, the final sign is a minus sign -/
theorem final_sign_is_minus (initial_state : BoardState) 
    (h_initial : initial_state.plus_count = 10 ∧ initial_state.minus_count = 15) 
    (operations : List Operation) 
    (h_operations : operations.length = 24) : 
    (operations.foldl apply_operation initial_state).plus_count = 0 ∧ 
    (operations.foldl apply_operation initial_state).minus_count = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_sign_is_minus_l3696_369654


namespace NUMINAMATH_CALUDE_larger_cuboid_height_l3696_369695

/-- The height of a larger cuboid given its dimensions and the number and dimensions of smaller cuboids it contains. -/
theorem larger_cuboid_height (length width : ℝ) (num_small_cuboids : ℕ) 
  (small_length small_width small_height : ℝ) : 
  length = 12 →
  width = 14 →
  num_small_cuboids = 56 →
  small_length = 5 →
  small_width = 3 →
  small_height = 2 →
  (length * width * (num_small_cuboids * small_length * small_width * small_height) / (length * width)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_height_l3696_369695


namespace NUMINAMATH_CALUDE_work_completion_proof_l3696_369608

/-- The number of days A takes to complete the work alone -/
def a_days : ℕ := 45

/-- The number of days B takes to complete the work alone -/
def b_days : ℕ := 40

/-- The number of days B takes to complete the remaining work after A leaves -/
def b_remaining_days : ℕ := 23

/-- The number of days A works before leaving -/
def x : ℕ := 9

theorem work_completion_proof :
  let total_work := 1
  let a_rate := total_work / a_days
  let b_rate := total_work / b_days
  x * (a_rate + b_rate) + b_remaining_days * b_rate = total_work :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3696_369608


namespace NUMINAMATH_CALUDE_combined_average_age_l3696_369690

theorem combined_average_age (x_count y_count : ℕ) (x_avg y_avg : ℝ) 
  (hx : x_count = 8) (hy : y_count = 5) 
  (hxa : x_avg = 30) (hya : y_avg = 45) : 
  (x_count * x_avg + y_count * y_avg) / (x_count + y_count) = 36 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l3696_369690


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l3696_369617

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 4

/-- The number of people that can sit in each seat -/
def people_per_seat : ℕ := 4

/-- The total number of people that can ride the Ferris wheel at the same time -/
def total_people : ℕ := num_seats * people_per_seat

theorem ferris_wheel_capacity : total_people = 16 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l3696_369617


namespace NUMINAMATH_CALUDE_trivia_game_win_probability_l3696_369651

def num_questions : ℕ := 4
def num_choices : ℕ := 4
def min_correct : ℕ := 3

def probability_correct_guess : ℚ := 1 / num_choices

def probability_win : ℚ :=
  (probability_correct_guess ^ num_questions) +
  (num_questions * (probability_correct_guess ^ (num_questions - 1)) * (1 - probability_correct_guess))

theorem trivia_game_win_probability :
  probability_win = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_win_probability_l3696_369651


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l3696_369662

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < exp 1 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l3696_369662


namespace NUMINAMATH_CALUDE_floor_sqrt_116_l3696_369681

theorem floor_sqrt_116 : ⌊Real.sqrt 116⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_116_l3696_369681


namespace NUMINAMATH_CALUDE_integer_solutions_x4_minus_2y2_eq_1_l3696_369667

theorem integer_solutions_x4_minus_2y2_eq_1 :
  ∀ x y : ℤ, x^4 - 2*y^2 = 1 ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_x4_minus_2y2_eq_1_l3696_369667


namespace NUMINAMATH_CALUDE_percentage_runs_by_running_is_fifty_percent_l3696_369655

def total_runs : ℕ := 120
def boundaries : ℕ := 3
def sixes : ℕ := 8
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

def runs_from_boundaries_and_sixes : ℕ := boundaries * runs_per_boundary + sixes * runs_per_six

def runs_by_running : ℕ := total_runs - runs_from_boundaries_and_sixes

theorem percentage_runs_by_running_is_fifty_percent :
  (runs_by_running : ℚ) / total_runs * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_percentage_runs_by_running_is_fifty_percent_l3696_369655


namespace NUMINAMATH_CALUDE_two_letter_language_max_words_l3696_369663

/-- A language with two letters and specific word formation rules -/
structure TwoLetterLanguage where
  alphabet : Finset Char
  max_word_length : ℕ
  is_valid_word : List Char → Prop
  no_concatenation : ∀ (w1 w2 : List Char), is_valid_word w1 → is_valid_word w2 → ¬is_valid_word (w1 ++ w2)

/-- The maximum number of words in the specific two-letter language -/
def max_word_count (L : TwoLetterLanguage) : ℕ := 16056

/-- Theorem stating the maximum number of words in the specific two-letter language -/
theorem two_letter_language_max_words (L : TwoLetterLanguage) 
  (h1 : L.alphabet.card = 2)
  (h2 : L.max_word_length = 13)
  : max_word_count L = 16056 := by
  sorry

end NUMINAMATH_CALUDE_two_letter_language_max_words_l3696_369663


namespace NUMINAMATH_CALUDE_rectangle_length_equal_square_side_l3696_369697

/-- The length of a rectangle with width 4 cm and area equal to a square with side length 4 cm -/
theorem rectangle_length_equal_square_side : ∀ (length : ℝ), 
  (4 : ℝ) * length = (4 : ℝ) * (4 : ℝ) → length = (4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equal_square_side_l3696_369697


namespace NUMINAMATH_CALUDE_sales_at_540_l3696_369635

/-- Represents the sales model for a product -/
structure SalesModel where
  originalPrice : ℕ
  initialSales : ℕ
  reductionStep : ℕ
  salesIncreasePerStep : ℕ

/-- Calculates the sales volume given a price reduction -/
def salesVolume (model : SalesModel) (priceReduction : ℕ) : ℕ :=
  model.initialSales + (priceReduction / model.reductionStep) * model.salesIncreasePerStep

/-- Theorem stating the sales volume at a specific price point -/
theorem sales_at_540 (model : SalesModel) 
  (h1 : model.originalPrice = 600)
  (h2 : model.initialSales = 750)
  (h3 : model.reductionStep = 5)
  (h4 : model.salesIncreasePerStep = 30) :
  salesVolume model 60 = 1110 := by
  sorry

#eval salesVolume { originalPrice := 600, initialSales := 750, reductionStep := 5, salesIncreasePerStep := 30 } 60

end NUMINAMATH_CALUDE_sales_at_540_l3696_369635


namespace NUMINAMATH_CALUDE_tile_arrangements_count_l3696_369657

def brown_tiles : ℕ := 2
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 2
def yellow_tiles : ℕ := 2
def orange_tiles : ℕ := 1

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles + orange_tiles

theorem tile_arrangements_count :
  (Nat.factorial total_tiles) / (Nat.factorial brown_tiles * Nat.factorial purple_tiles * 
   Nat.factorial green_tiles * Nat.factorial yellow_tiles * Nat.factorial orange_tiles) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_count_l3696_369657


namespace NUMINAMATH_CALUDE_sequence_formula_l3696_369616

def geometric_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ k, k ≥ 1 → a (k + 1) - a k = (a 2 - a 1) * (2 ^ (k - 1))

theorem sequence_formula (a : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 →
  geometric_sequence a n →
  a 2 - a 1 = 2 →
  ∀ k, k ≥ 1 → a k = 2^k - 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l3696_369616


namespace NUMINAMATH_CALUDE_extremum_point_condition_l3696_369611

open Real

theorem extremum_point_condition (a : ℝ) :
  (∀ b : ℝ, ∃! x : ℝ, x > 0 ∧ 
    (∀ y : ℝ, y > 0 → (exp (a * x) * (log x + b) ≥ exp (a * y) * (log y + b)) ∨
                      (exp (a * x) * (log x + b) ≤ exp (a * y) * (log y + b))))
  → a < 0 := by
sorry

end NUMINAMATH_CALUDE_extremum_point_condition_l3696_369611


namespace NUMINAMATH_CALUDE_students_just_passed_l3696_369659

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 28 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_all_passed : first_div_percent + second_div_percent ≤ 1) :
  total - (total * (first_div_percent + second_div_percent)).floor = 54 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l3696_369659


namespace NUMINAMATH_CALUDE_speed_is_pi_over_three_l3696_369665

/-- Represents a rectangular track with looped ends -/
structure Track where
  width : ℝ
  straightLength : ℝ

/-- Calculates the speed of a person walking around the track -/
def calculateSpeed (track : Track) (timeDifference : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the specific track conditions, the calculated speed is π/3 -/
theorem speed_is_pi_over_three (track : Track) (h1 : track.width = 8)
    (h2 : track.straightLength = 100) (timeDifference : ℝ) (h3 : timeDifference = 48) :
    calculateSpeed track timeDifference = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_speed_is_pi_over_three_l3696_369665


namespace NUMINAMATH_CALUDE_inches_to_represent_distance_l3696_369650

/-- Represents the scale of a map in miles per inch -/
def map_scale : ℝ := 28

/-- Represents the relationship between inches and miles on the map -/
theorem inches_to_represent_distance (D : ℝ) :
  ∃ I : ℝ, I * map_scale = D ∧ I = D / map_scale :=
sorry

end NUMINAMATH_CALUDE_inches_to_represent_distance_l3696_369650


namespace NUMINAMATH_CALUDE_solve_for_x_l3696_369671

theorem solve_for_x (x y : ℚ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3696_369671


namespace NUMINAMATH_CALUDE_four_integer_average_l3696_369624

theorem four_integer_average (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧ 
  d = 90 ∧ 
  a ≥ 37 → 
  (a + b + c + d) / 4 ≥ 51 := by
sorry

end NUMINAMATH_CALUDE_four_integer_average_l3696_369624


namespace NUMINAMATH_CALUDE_horse_race_equation_l3696_369676

/-- Represents the scenario of two horses racing --/
structure HorseRace where
  fast_speed : ℕ  -- Speed of the fast horse in miles per day
  slow_speed : ℕ  -- Speed of the slow horse in miles per day
  head_start : ℕ  -- Number of days the slow horse starts earlier

/-- The equation for when the fast horse catches up to the slow horse --/
def catch_up_equation (race : HorseRace) (x : ℕ) : Prop :=
  race.slow_speed * (x + race.head_start) = race.fast_speed * x

/-- The theorem stating the correct equation for the given scenario --/
theorem horse_race_equation :
  let race := HorseRace.mk 240 150 12
  ∀ x, catch_up_equation race x ↔ 150 * (x + 12) = 240 * x :=
by sorry

end NUMINAMATH_CALUDE_horse_race_equation_l3696_369676


namespace NUMINAMATH_CALUDE_tomatoes_eaten_by_birds_l3696_369638

theorem tomatoes_eaten_by_birds 
  (total_grown : ℕ) 
  (remaining : ℕ) 
  (h1 : total_grown = 127) 
  (h2 : remaining = 54) 
  (h3 : remaining * 2 = total_grown - (total_grown - remaining * 2)) : 
  total_grown - remaining * 2 = 19 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_eaten_by_birds_l3696_369638


namespace NUMINAMATH_CALUDE_f_of_3x_plus_2_l3696_369622

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_of_3x_plus_2 (x : ℝ) : f (3 * x + 2) = 9 * x^2 + 12 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3x_plus_2_l3696_369622


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3696_369600

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, (-1 < x ∧ x < 3) → x < 3) ∧
  (∃ x : ℝ, x < 3 ∧ ¬(-1 < x ∧ x < 3)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3696_369600


namespace NUMINAMATH_CALUDE_unique_unbeatable_city_l3696_369631

/-- Represents a city with two bulldozers -/
structure City where
  leftBulldozer : ℕ
  rightBulldozer : ℕ

/-- Represents the road with n cities -/
def Road (n : ℕ) := Fin n → City

/-- A city i overtakes city j if its right bulldozer can reach j -/
def overtakes (road : Road n) (i j : Fin n) : Prop :=
  i < j ∧ ∀ k, i < k ∧ k ≤ j → (road i).rightBulldozer > (road k).leftBulldozer

/-- There exists a unique city that cannot be overtaken -/
theorem unique_unbeatable_city (n : ℕ) (road : Road n)
  (h1 : ∀ i j : Fin n, i ≠ j → (road i).leftBulldozer ≠ (road j).leftBulldozer)
  (h2 : ∀ i j : Fin n, i ≠ j → (road i).rightBulldozer ≠ (road j).rightBulldozer)
  (h3 : ∀ i : Fin n, (road i).leftBulldozer ≠ (road i).rightBulldozer) :
  ∃! i : Fin n, ∀ j : Fin n, j ≠ i → ¬(overtakes road j i) :=
sorry

end NUMINAMATH_CALUDE_unique_unbeatable_city_l3696_369631


namespace NUMINAMATH_CALUDE_distinct_values_count_l3696_369689

def odd_integers_less_than_15 : Finset ℕ :=
  {1, 3, 5, 7, 9, 11, 13}

def expression (p q : ℕ) : ℤ :=
  p * q - (p + q)

theorem distinct_values_count :
  Finset.card (Finset.image₂ expression odd_integers_less_than_15 odd_integers_less_than_15) = 28 :=
by sorry

end NUMINAMATH_CALUDE_distinct_values_count_l3696_369689


namespace NUMINAMATH_CALUDE_square_root_difference_limit_l3696_369628

theorem square_root_difference_limit : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |Real.sqrt (n + 1) - Real.sqrt n| < ε := by
sorry

end NUMINAMATH_CALUDE_square_root_difference_limit_l3696_369628


namespace NUMINAMATH_CALUDE_complex_number_equality_l3696_369672

theorem complex_number_equality (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (2 - Complex.I)
  Complex.re z = Complex.im z → a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3696_369672


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l3696_369620

/-- A triangle with sides 13, 13, and 10 units -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 13
  hb : b = 13
  hc : c = 10

/-- The sum of squares of medians of the triangle -/
def sumOfSquaresOfMedians (t : IsoscelesTriangle) : ℝ := sorry

/-- The area of the triangle -/
def triangleArea (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem stating the sum of squares of medians and area of the specific triangle -/
theorem isosceles_triangle_properties (t : IsoscelesTriangle) :
  sumOfSquaresOfMedians t = 278.5 ∧ triangleArea t = 60 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_properties_l3696_369620


namespace NUMINAMATH_CALUDE_solve_for_d_l3696_369652

theorem solve_for_d (y : ℝ) (d : ℝ) (h1 : y > 0) (h2 : (6 * y) / 20 + (3 * y) / d = 0.60 * y) : d = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_d_l3696_369652


namespace NUMINAMATH_CALUDE_motorist_gas_plan_l3696_369669

/-- The number of gallons a motorist initially planned to buy given certain conditions -/
theorem motorist_gas_plan (actual_price expected_price_difference affordable_gallons : ℚ) :
  actual_price = 150 ∧ 
  expected_price_difference = 30 ∧ 
  affordable_gallons = 10 →
  (actual_price * affordable_gallons) / (actual_price - expected_price_difference) = 25/2 :=
by sorry

end NUMINAMATH_CALUDE_motorist_gas_plan_l3696_369669


namespace NUMINAMATH_CALUDE_system_solution_l3696_369644

theorem system_solution (x y : ℝ) : 
  (2 * x + y = 5) ∧ (x - 3 * y = 6) ↔ (x = 3 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3696_369644


namespace NUMINAMATH_CALUDE_value_added_after_doubling_l3696_369605

theorem value_added_after_doubling (x : ℝ) (v : ℝ) : 
  x = 4 → 2 * x + v = x / 2 + 20 → v = 14 := by
  sorry

end NUMINAMATH_CALUDE_value_added_after_doubling_l3696_369605


namespace NUMINAMATH_CALUDE_max_five_sunday_months_correct_five_is_max_l3696_369682

/-- Represents a year, which can be either common (365 days) or leap (366 days) -/
inductive Year
| Common
| Leap

/-- Represents a month in a year -/
structure Month where
  days : Nat
  h1 : days ≥ 28
  h2 : days ≤ 31

/-- The number of Sundays in a month -/
def sundays (m : Month) : Nat :=
  if m.days ≥ 35 then 5 else 4

/-- The maximum number of months with 5 Sundays in a year -/
def max_five_sunday_months (y : Year) : Nat :=
  match y with
  | Year.Common => 4
  | Year.Leap => 5

theorem max_five_sunday_months_correct (y : Year) :
  max_five_sunday_months y = 
    match y with
    | Year.Common => 4
    | Year.Leap => 5 :=
by
  sorry

theorem five_is_max (y : Year) :
  max_five_sunday_months y ≤ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_five_sunday_months_correct_five_is_max_l3696_369682


namespace NUMINAMATH_CALUDE_equation_solution_l3696_369621

theorem equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 3 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 3 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3696_369621


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3696_369685

theorem product_of_three_numbers (p q r m : ℝ) 
  (sum_eq : p + q + r = 180)
  (p_eq : 8 * p = m)
  (q_eq : q - 10 = m)
  (r_eq : r + 10 = m)
  (p_smallest : p < q ∧ p < r) :
  p * q * r = 90000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3696_369685


namespace NUMINAMATH_CALUDE_probability_abs_diff_gt_half_l3696_369674

/-- A coin flip result -/
inductive CoinFlip
| Heads
| Tails

/-- The result of the number selection process -/
inductive NumberSelection
| Uniform : ℝ → NumberSelection
| Zero
| One

/-- The process of selecting a number based on coin flips -/
def selectNumber (flip1 : CoinFlip) (flip2 : CoinFlip) (u : ℝ) : NumberSelection :=
  match flip1 with
  | CoinFlip.Heads => match flip2 with
    | CoinFlip.Heads => NumberSelection.Zero
    | CoinFlip.Tails => NumberSelection.One
  | CoinFlip.Tails => NumberSelection.Uniform u

/-- The probability measure for the problem -/
noncomputable def P : Set (NumberSelection × NumberSelection) → ℝ := sorry

/-- The event that |x-y| > 1/2 -/
def event : Set (NumberSelection × NumberSelection) :=
  {pair | let (x, y) := pair
          match x, y with
          | NumberSelection.Uniform x', NumberSelection.Uniform y' => |x' - y'| > 1/2
          | NumberSelection.Zero, NumberSelection.Uniform y' => y' < 1/2
          | NumberSelection.One, NumberSelection.Uniform y' => y' < 1/2
          | NumberSelection.Uniform x', NumberSelection.Zero => x' > 1/2
          | NumberSelection.Uniform x', NumberSelection.One => x' < 1/2
          | NumberSelection.Zero, NumberSelection.One => true
          | NumberSelection.One, NumberSelection.Zero => true
          | _, _ => false}

theorem probability_abs_diff_gt_half :
  P event = 7/16 := by sorry

end NUMINAMATH_CALUDE_probability_abs_diff_gt_half_l3696_369674
