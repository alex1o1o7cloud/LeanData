import Mathlib

namespace NUMINAMATH_CALUDE_no_refuel_needed_l4054_405482

-- Define the parameters
def total_distance : ℕ := 156
def distance_driven : ℕ := 48
def gas_added : ℕ := 12
def fuel_consumption : ℕ := 24

-- Define the remaining distance
def remaining_distance : ℕ := total_distance - distance_driven

-- Define the range with added gas
def range_with_added_gas : ℕ := gas_added * fuel_consumption

-- Theorem statement
theorem no_refuel_needed : range_with_added_gas ≥ remaining_distance := by
  sorry

end NUMINAMATH_CALUDE_no_refuel_needed_l4054_405482


namespace NUMINAMATH_CALUDE_nina_total_spending_l4054_405466

/-- The total cost of Nina's purchases --/
def total_cost (toy_price toy_quantity card_price card_quantity shirt_price shirt_quantity : ℕ) : ℕ :=
  toy_price * toy_quantity + card_price * card_quantity + shirt_price * shirt_quantity

/-- Theorem stating that Nina's total spending is $70 --/
theorem nina_total_spending :
  total_cost 10 3 5 2 6 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_nina_total_spending_l4054_405466


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l4054_405453

/-- Given three lines in the 2D plane:
    l₁: 3x + 4y - 2 = 0
    l₂: 2x + y + 2 = 0
    l₃: 3x - 2y + 4 = 0
    Prove that the line l: 2x - 3y - 22 = 0 passes through the intersection of l₁ and l₂,
    and is perpendicular to l₃. -/
theorem intersection_and_perpendicular_line 
  (l₁ : Real → Real → Prop) 
  (l₂ : Real → Real → Prop)
  (l₃ : Real → Real → Prop)
  (l : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 3*x + 4*y - 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 2*x + y + 2 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ 3*x - 2*y + 4 = 0)
  (h : ∀ x y, l x y ↔ 2*x - 3*y - 22 = 0) :
  (∃ x y, l₁ x y ∧ l₂ x y ∧ l x y) ∧ 
  (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → l₃ x₁ y₁ → l₃ x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l4054_405453


namespace NUMINAMATH_CALUDE_geometric_series_sum_2_to_2048_l4054_405484

def geometric_series_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def last_term (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * r^(n - 1)

theorem geometric_series_sum_2_to_2048 :
  ∃ n : ℕ, 
    last_term 2 2 n = 2048 ∧ 
    geometric_series_sum 2 2 n = 4094 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_2_to_2048_l4054_405484


namespace NUMINAMATH_CALUDE_circle_C_theorem_l4054_405446

/-- Definition of the circle C with parameter t -/
def circle_C (t : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*t*x - 2*t^2*y + 4*t - 4 = 0

/-- Definition of the line on which the center of C lies -/
def center_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

/-- First possible equation of circle C -/
def circle_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 8 = 0

/-- Second possible equation of circle C -/
def circle_C2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y + 4 = 0

/-- The fixed point that C passes through -/
def fixed_point : ℝ × ℝ := (2, 0)

theorem circle_C_theorem :
  ∀ t : ℝ,
  (∃ x y : ℝ, circle_C t x y ∧ center_line x y) →
  ((∀ x y : ℝ, circle_C t x y ↔ circle_C1 x y) ∨
   (∀ x y : ℝ, circle_C t x y ↔ circle_C2 x y)) ∧
  circle_C t fixed_point.1 fixed_point.2 :=
sorry

end NUMINAMATH_CALUDE_circle_C_theorem_l4054_405446


namespace NUMINAMATH_CALUDE_sphere_volume_increase_on_doubling_radius_l4054_405422

theorem sphere_volume_increase_on_doubling_radius :
  ∀ (r : ℝ), r > 0 →
  (4 / 3 * Real.pi * (2 * r)^3) = 8 * (4 / 3 * Real.pi * r^3) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_on_doubling_radius_l4054_405422


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l4054_405473

-- Define a type for lines in 3D space
structure Line3D where
  -- We don't need to specify the exact representation of a line here
  -- as we're only interested in their relationships

-- Define the relationships between lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry
def parallel (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem perpendicular_parallel_implies_perpendicular 
  (l1 l2 l3 : Line3D) 
  (h1 : perpendicular l1 l2) 
  (h2 : parallel l2 l3) : 
  perpendicular l1 l3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l4054_405473


namespace NUMINAMATH_CALUDE_quadratic_integer_root_set_characterization_l4054_405465

/-- The set of positive integers a for which the quadratic equation
    ax^2 + 2(2a-1)x + 4(a-3) = 0 has at least one integer root -/
def QuadraticIntegerRootSet : Set ℕ+ :=
  {a | ∃ x : ℤ, a * x^2 + 2*(2*a-1)*x + 4*(a-3) = 0}

/-- Theorem stating that the QuadraticIntegerRootSet contains exactly 1, 3, 6, and 10 -/
theorem quadratic_integer_root_set_characterization :
  QuadraticIntegerRootSet = {1, 3, 6, 10} := by
  sorry

#check quadratic_integer_root_set_characterization

end NUMINAMATH_CALUDE_quadratic_integer_root_set_characterization_l4054_405465


namespace NUMINAMATH_CALUDE_weighted_average_percentage_l4054_405477

def bag1_popped : ℕ := 60
def bag1_total : ℕ := 75

def bag2_popped : ℕ := 42
def bag2_total : ℕ := 50

def bag3_popped : ℕ := 112
def bag3_total : ℕ := 130

def bag4_popped : ℕ := 68
def bag4_total : ℕ := 90

def bag5_popped : ℕ := 82
def bag5_total : ℕ := 100

def total_kernels : ℕ := bag1_total + bag2_total + bag3_total + bag4_total + bag5_total

def weighted_sum : ℚ :=
  (bag1_popped : ℚ) / (bag1_total : ℚ) * (bag1_total : ℚ) +
  (bag2_popped : ℚ) / (bag2_total : ℚ) * (bag2_total : ℚ) +
  (bag3_popped : ℚ) / (bag3_total : ℚ) * (bag3_total : ℚ) +
  (bag4_popped : ℚ) / (bag4_total : ℚ) * (bag4_total : ℚ) +
  (bag5_popped : ℚ) / (bag5_total : ℚ) * (bag5_total : ℚ)

theorem weighted_average_percentage (ε : ℚ) (hε : ε = 1 / 10000) :
  ∃ (x : ℚ), abs (x - (weighted_sum / (total_kernels : ℚ))) < ε ∧ x = 7503 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_percentage_l4054_405477


namespace NUMINAMATH_CALUDE_star_chain_evaluation_l4054_405494

def star (a b : ℤ) : ℤ := a * b + a + b

theorem star_chain_evaluation :
  ∃ f : ℕ → ℤ, f 1 = star 1 2 ∧ 
  (∀ n : ℕ, n ≥ 2 → f n = star (f (n-1)) (n+1)) ∧
  f 99 = Nat.factorial 101 - 1 := by sorry

end NUMINAMATH_CALUDE_star_chain_evaluation_l4054_405494


namespace NUMINAMATH_CALUDE_post_office_distance_l4054_405410

/-- Proves that the distance of a round trip journey is 10 km given specific conditions -/
theorem post_office_distance (outward_speed return_speed total_time : ℝ) 
  (h1 : outward_speed = 12.5)
  (h2 : return_speed = 2)
  (h3 : total_time = 5.8) : 
  (total_time * outward_speed * return_speed) / (outward_speed + return_speed) = 10 := by
  sorry

end NUMINAMATH_CALUDE_post_office_distance_l4054_405410


namespace NUMINAMATH_CALUDE_packs_per_carton_is_five_l4054_405469

/-- The number of sticks of gum in each pack -/
def sticks_per_pack : ℕ := 3

/-- The number of cartons in each brown box -/
def cartons_per_box : ℕ := 4

/-- The total number of sticks of gum in all brown boxes -/
def total_sticks : ℕ := 480

/-- The number of brown boxes -/
def num_boxes : ℕ := 8

/-- The number of packs of gum in each carton -/
def packs_per_carton : ℕ := total_sticks / (num_boxes * cartons_per_box * sticks_per_pack)

theorem packs_per_carton_is_five : packs_per_carton = 5 := by sorry

end NUMINAMATH_CALUDE_packs_per_carton_is_five_l4054_405469


namespace NUMINAMATH_CALUDE_cylinder_height_from_cube_water_l4054_405419

/-- The height of a cylinder filled with water from a cube -/
theorem cylinder_height_from_cube_water (cube_edge : ℝ) (cylinder_base_area : ℝ) 
  (h_cube_edge : cube_edge = 6)
  (h_cylinder_base : cylinder_base_area = 18)
  (h_water_conserved : cube_edge ^ 3 = cylinder_base_area * cylinder_height) :
  cylinder_height = 12 := by
  sorry

#check cylinder_height_from_cube_water

end NUMINAMATH_CALUDE_cylinder_height_from_cube_water_l4054_405419


namespace NUMINAMATH_CALUDE_rice_profit_l4054_405498

/-- Calculates the profit from selling a sack of rice -/
theorem rice_profit (weight : ℝ) (cost : ℝ) (price_per_kg : ℝ) :
  weight = 50 ∧ cost = 50 ∧ price_per_kg = 1.20 →
  weight * price_per_kg - cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_rice_profit_l4054_405498


namespace NUMINAMATH_CALUDE_jan_water_collection_l4054_405487

theorem jan_water_collection :
  ∀ (initial_water : ℕ) 
    (car_water : ℕ) 
    (plant_water : ℕ) 
    (plates_clothes_water : ℕ),
  car_water = 7 * 2 →
  plant_water = car_water - 11 →
  plates_clothes_water = 24 →
  plates_clothes_water * 2 = initial_water - (car_water + plant_water) →
  initial_water = 65 :=
by
  sorry


end NUMINAMATH_CALUDE_jan_water_collection_l4054_405487


namespace NUMINAMATH_CALUDE_A_intersect_B_is_singleton_one_l4054_405421

def A : Set ℝ := {0.1, 1, 10}

def B : Set ℝ := { y | ∃ x ∈ A, y = Real.log x / Real.log 10 }

theorem A_intersect_B_is_singleton_one : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_singleton_one_l4054_405421


namespace NUMINAMATH_CALUDE_point_on_curve_with_slope_l4054_405447

def curve (x : ℝ) : ℝ := x^2 + x - 2

def tangent_slope (x : ℝ) : ℝ := 2*x + 1

theorem point_on_curve_with_slope : 
  ∃ (x y : ℝ), curve x = y ∧ tangent_slope x = 3 ∧ x = 1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_with_slope_l4054_405447


namespace NUMINAMATH_CALUDE_pet_sitting_charge_per_night_l4054_405462

def num_cats : ℕ := 2
def num_dogs : ℕ := 3
def total_payment : ℕ := 65

theorem pet_sitting_charge_per_night :
  (total_payment : ℚ) / (num_cats + num_dogs : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_pet_sitting_charge_per_night_l4054_405462


namespace NUMINAMATH_CALUDE_one_third_of_five_times_seven_l4054_405406

theorem one_third_of_five_times_seven :
  (1/3 : ℚ) * (5 * 7) = 35/3 := by
sorry

end NUMINAMATH_CALUDE_one_third_of_five_times_seven_l4054_405406


namespace NUMINAMATH_CALUDE_hall_width_to_length_ratio_l4054_405463

/-- Represents a rectangular hall -/
structure RectangularHall where
  width : ℝ
  length : ℝ

/-- Properties of the rectangular hall -/
def HallProperties (hall : RectangularHall) : Prop :=
  hall.width > 0 ∧ 
  hall.length > 0 ∧
  hall.width * hall.length = 128 ∧ 
  hall.length - hall.width = 8

theorem hall_width_to_length_ratio 
  (hall : RectangularHall) 
  (h : HallProperties hall) : 
  hall.width / hall.length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hall_width_to_length_ratio_l4054_405463


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_a_l4054_405472

-- Define the quadratic function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Theorem 1
theorem min_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) 
  (h3 : ∀ x : ℝ, f a (Real.sin x) ≤ 5/4) :
  ∃ x : ℝ, f a x = -1 ∧ ∀ y : ℝ, f a y ≥ -1 := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) 
  (h : ∀ x : ℝ, x ∈ Set.Icc (-π/2) 0 → 
    a/2 * Real.sin x * Real.cos x + 1/2 * Real.sin x + 1/2 * Real.cos x + a/4 ≤ 1) :
  a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_range_of_a_l4054_405472


namespace NUMINAMATH_CALUDE_sum_reciprocals_l4054_405480

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^3 = 1 →
  ω ≠ 1 →
  (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω →
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l4054_405480


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l4054_405409

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 991 ∧ 
  n < 1000 ∧ 
  n > 99 ∧
  55 * n ≡ 165 [MOD 260] ∧
  ∀ (m : ℕ), m < 1000 ∧ m > 99 ∧ 55 * m ≡ 165 [MOD 260] → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l4054_405409


namespace NUMINAMATH_CALUDE_euler_line_intersection_l4054_405490

structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

def isAcute (t : Triangle) : Prop := sorry

def isObtuse (t : Triangle) : Prop := sorry

def eulerLine (t : Triangle) : Set (ℝ × ℝ) := sorry

def sideLength (t : Triangle) (i : Fin 3) : ℝ := sorry

def largestSide (t : Triangle) : Fin 3 := sorry

def smallestSide (t : Triangle) : Fin 3 := sorry

def medianSide (t : Triangle) : Fin 3 := sorry

def intersects (line : Set (ℝ × ℝ)) (side : Fin 3) (t : Triangle) : Prop := sorry

theorem euler_line_intersection (t : Triangle) :
  (isAcute t → intersects (eulerLine t) (largestSide t) t ∧ intersects (eulerLine t) (smallestSide t) t) ∧
  (isObtuse t → intersects (eulerLine t) (largestSide t) t ∧ intersects (eulerLine t) (medianSide t) t) := by
  sorry

end NUMINAMATH_CALUDE_euler_line_intersection_l4054_405490


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l4054_405400

/-- Calculates the man's speed against the current with wind resistance -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) (wind_resistance_factor : ℝ) (current_increase_factor : ℝ) : ℝ :=
  let speed_still_water := speed_with_current - current_speed
  let effective_speed_still_water := speed_still_water * (1 - wind_resistance_factor)
  let new_current_speed := current_speed * (1 + current_increase_factor)
  effective_speed_still_water - new_current_speed

/-- Theorem stating the man's speed against the current -/
theorem mans_speed_against_current :
  speed_against_current 22 5 0.15 0.1 = 8.95 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l4054_405400


namespace NUMINAMATH_CALUDE_expression_equals_sixteen_times_twelve_to_1001_l4054_405412

theorem expression_equals_sixteen_times_twelve_to_1001 :
  (3^1001 + 4^1002)^2 - (3^1001 - 4^1002)^2 = 16 * 12^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_sixteen_times_twelve_to_1001_l4054_405412


namespace NUMINAMATH_CALUDE_obrienHatsAfterLoss_l4054_405468

/-- The number of hats Policeman O'Brien has after losing one -/
def obrienHats (simpsonHats : ℕ) : ℕ :=
  2 * simpsonHats + 5 - 1

theorem obrienHatsAfterLoss (simpsonHats : ℕ) (h : simpsonHats = 15) : 
  obrienHats simpsonHats = 34 := by
  sorry

end NUMINAMATH_CALUDE_obrienHatsAfterLoss_l4054_405468


namespace NUMINAMATH_CALUDE_factorable_implies_even_b_l4054_405401

/-- A quadratic expression of the form 15x^2 + bx + 15 -/
def quadratic_expr (b : ℤ) (x : ℝ) : ℝ := 15 * x^2 + b * x + 15

/-- Represents a linear binomial factor with integer coefficients -/
structure LinearFactor where
  c : ℤ
  d : ℤ

/-- Checks if a quadratic expression can be factored into two linear binomial factors -/
def is_factorable (b : ℤ) : Prop :=
  ∃ (f1 f2 : LinearFactor), ∀ x, 
    quadratic_expr b x = (f1.c * x + f1.d) * (f2.c * x + f2.d)

theorem factorable_implies_even_b :
  ∀ b : ℤ, is_factorable b → Even b :=
sorry

end NUMINAMATH_CALUDE_factorable_implies_even_b_l4054_405401


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_1000_l4054_405440

theorem units_digit_of_7_power_1000 : (7^(10^3)) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_1000_l4054_405440


namespace NUMINAMATH_CALUDE_negation_of_forall_product_nonzero_l4054_405499

theorem negation_of_forall_product_nonzero (f g : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x * g x ≠ 0) ↔ (∃ x : ℝ, f x = 0 ∨ g x = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_product_nonzero_l4054_405499


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l4054_405461

/-- The polynomial we're working with -/
def p (x : ℝ) : ℝ := 5 * (2 * x^5 - x^3 + 2 * x^2 - 3)

/-- The coefficients of the expanded polynomial -/
def coefficients : List ℝ := [10, 0, -5, 10, 0, -15]

/-- Sum of squares of coefficients -/
def sum_of_squares (l : List ℝ) : ℝ := (l.map (λ x => x^2)).sum

theorem sum_of_squares_of_coefficients :
  sum_of_squares coefficients = 450 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l4054_405461


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_i_l4054_405493

theorem imaginary_part_of_i_times_one_minus_i (i : ℂ) : 
  i * i = -1 → Complex.im (i * (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_i_l4054_405493


namespace NUMINAMATH_CALUDE_eunji_class_size_l4054_405438

/-- The number of students who can play instrument (a) -/
def students_a : ℕ := 24

/-- The number of students who can play instrument (b) -/
def students_b : ℕ := 17

/-- The number of students who can play both instruments -/
def students_both : ℕ := 8

/-- The total number of students in Eunji's class -/
def total_students : ℕ := students_a + students_b - students_both

theorem eunji_class_size :
  total_students = 33 ∧
  students_a = 24 ∧
  students_b = 17 ∧
  students_both = 8 ∧
  total_students = students_a + students_b - students_both :=
by sorry

end NUMINAMATH_CALUDE_eunji_class_size_l4054_405438


namespace NUMINAMATH_CALUDE_circle_ratio_l4054_405478

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 4 * (π * a^2) → a / b = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l4054_405478


namespace NUMINAMATH_CALUDE_free_trade_superior_l4054_405448

/-- Represents a country with its production capacity for zucchinis and cauliflower -/
structure Country where
  zucchini_capacity : ℝ
  cauliflower_capacity : ℝ

/-- Represents the market conditions and consumption preferences -/
structure MarketConditions where
  price_ratio : ℝ  -- Price of zucchini / Price of cauliflower
  consumption_ratio : ℝ  -- Ratio of zucchini to cauliflower consumption

/-- Calculates the total vegetable consumption under free trade conditions -/
def free_trade_consumption (a b : Country) (market : MarketConditions) : ℝ :=
  sorry

/-- Calculates the total vegetable consumption for the unified country without trade -/
def unified_consumption (a b : Country) : ℝ :=
  sorry

/-- Theorem stating that free trade leads to higher total consumption -/
theorem free_trade_superior (a b : Country) (market : MarketConditions) :
  free_trade_consumption a b market > unified_consumption a b :=
sorry

end NUMINAMATH_CALUDE_free_trade_superior_l4054_405448


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l4054_405451

/-- A number is composite if it has more than two distinct positive divisors. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

/-- A natural number can be expressed as the sum of two composite numbers. -/
def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ a + b = n

/-- 11 is the largest natural number that cannot be expressed as the sum of two composite numbers. -/
theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l4054_405451


namespace NUMINAMATH_CALUDE_lychee_ratio_l4054_405413

theorem lychee_ratio (total : ℕ) (remaining : ℕ) : 
  total = 500 → 
  remaining = 100 → 
  (total / 2 - remaining : ℚ) / (total / 2 : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_lychee_ratio_l4054_405413


namespace NUMINAMATH_CALUDE_problem_solution_l4054_405457

-- Define A and B as functions of a and b
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 + 6 * a * b + 8 * a

-- Theorem for the three parts of the problem
theorem problem_solution :
  (∀ a b : ℝ, 2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a) ∧
  (2 * A (-2) 1 - B (-2) 1 = 54) ∧
  (∀ a b : ℝ, (∀ a' : ℝ, 2 * A a' b - B a' b = 2 * A a b - B a b) ↔ b = -1/2) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l4054_405457


namespace NUMINAMATH_CALUDE_expression_factorization_l4054_405415

theorem expression_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + 3 * a * b * c * (a - b) * (b - c) * (c - a) =
  (a - b) * (b - c) * (c - a) * (a + b + c + 3 * a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l4054_405415


namespace NUMINAMATH_CALUDE_twelve_person_tournament_matches_l4054_405430

/-- Calculate the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 12-person round-robin tournament has 66 matches -/
theorem twelve_person_tournament_matches : 
  roundRobinMatches 12 = 66 := by
sorry

end NUMINAMATH_CALUDE_twelve_person_tournament_matches_l4054_405430


namespace NUMINAMATH_CALUDE_map_scale_l4054_405449

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (map_cm : ℝ) (map_km : ℝ) (actual_cm : ℝ)
  (h1 : map_cm = 15)
  (h2 : map_km = 90)
  (h3 : actual_cm = 20) :
  (actual_cm / map_cm) * map_km = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l4054_405449


namespace NUMINAMATH_CALUDE_two_draw_probability_l4054_405416

/-- The probability of drawing either a red and blue chip or a blue and green chip
    in two draws with replacement from a bag containing 6 red, 4 blue, and 2 green chips -/
theorem two_draw_probability (red blue green : ℕ) (total : ℕ) : 
  red = 6 → blue = 4 → green = 2 → total = red + blue + green →
  (red / total * blue / total + blue / total * red / total +
   blue / total * green / total + green / total * blue / total : ℚ) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_two_draw_probability_l4054_405416


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l4054_405425

variable (a b c : ℝ)

def matrix1 : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2*c, -2*b],
    ![-2*c, 0, 2*a],
    ![2*b, -2*a, 0]]

def matrix2 : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a^2, 2*a*b, 2*a*c],
    ![2*a*b, b^2, 2*b*c],
    ![2*a*c, 2*b*c, c^2]]

theorem matrix_product_is_zero :
  matrix1 a b c * matrix2 a b c = 0 := by sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l4054_405425


namespace NUMINAMATH_CALUDE_tan_half_product_l4054_405408

theorem tan_half_product (a b : Real) :
  7 * (Real.sin a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 1) ∨ (Real.tan (a / 2) * Real.tan (b / 2) = -1) :=
by sorry

end NUMINAMATH_CALUDE_tan_half_product_l4054_405408


namespace NUMINAMATH_CALUDE_minimum_value_a2_plus_b2_l4054_405489

theorem minimum_value_a2_plus_b2 (a b : ℝ) : 
  (∃ k : ℕ, (20 : ℝ) = k * a^3 * b^3 ∧ Nat.choose 6 k * a^(6-k) * b^k = (20 : ℝ) * a^(6-k) * b^k) → 
  a^2 + b^2 ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_a2_plus_b2_l4054_405489


namespace NUMINAMATH_CALUDE_win_rate_problem_l4054_405495

/-- Represents the win rate problem for a sports team -/
theorem win_rate_problem (first_third_win_rate : ℚ) (total_matches : ℕ) :
  first_third_win_rate = 55 / 100 →
  (∃ (remaining_win_rate : ℚ),
    remaining_win_rate = 85 / 100 ∧
    first_third_win_rate * (1 / 3) + remaining_win_rate * (2 / 3) = 3 / 4) ∧
  (first_third_win_rate * (1 / 3) + 1 * (2 / 3) = 85 / 100) :=
by sorry

end NUMINAMATH_CALUDE_win_rate_problem_l4054_405495


namespace NUMINAMATH_CALUDE_mean_problem_l4054_405439

theorem mean_problem (x : ℝ) :
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_problem_l4054_405439


namespace NUMINAMATH_CALUDE_abs_value_inequality_solution_set_l4054_405460

theorem abs_value_inequality_solution_set :
  {x : ℝ | |x| > 1} = {x : ℝ | x > 1 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_abs_value_inequality_solution_set_l4054_405460


namespace NUMINAMATH_CALUDE_tinas_time_l4054_405417

/-- Represents the mile times of three runners with specific speed relationships -/
structure RunnerTimes where
  tom : ℝ
  tina : ℝ
  tony : ℝ
  tina_slower : tina = 3 * tom
  tony_faster : tony = tina / 2
  total_time : tom + tina + tony = 11

/-- Theorem stating that given the conditions, Tina's mile time is 6 minutes -/
theorem tinas_time (rt : RunnerTimes) : rt.tina = 6 := by
  sorry

end NUMINAMATH_CALUDE_tinas_time_l4054_405417


namespace NUMINAMATH_CALUDE_series_sum_equality_l4054_405418

/-- Given real numbers c and d satisfying a specific equation, 
    prove that the sum of a certain series equals a specific fraction. -/
theorem series_sum_equality (c d : ℝ) 
    (h : (c / d) / (1 - 1 / d) + (1 / d) / (1 - 1 / d) = 6) :
  c / (c + 2 * d) / (1 - 1 / (c + 2 * d)) = (6 * d - 7) / (8 * (d - 1)) := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equality_l4054_405418


namespace NUMINAMATH_CALUDE_problem_statement_l4054_405481

theorem problem_statement : 
  (Real.sin (15 * π / 180)) / (Real.cos (75 * π / 180)) + 
  1 / (Real.sin (75 * π / 180))^2 - 
  (Real.tan (15 * π / 180))^2 = 2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l4054_405481


namespace NUMINAMATH_CALUDE_point_P_coordinates_l4054_405434

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem point_P_coordinates :
  ∃ (P : ℝ × ℝ), (f' P.1 = 3) ∧ ((P = (-1, -1)) ∨ (P = (1, 1))) :=
sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l4054_405434


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l4054_405456

/-- The area of a cross-section created by cutting a right circular cylinder -/
theorem cylinder_cross_section_area
  (r : ℝ) -- radius of the cylinder
  (h : ℝ) -- height of the cylinder
  (θ : ℝ) -- angle of the arc in radians
  (hr : r = 8) -- given radius
  (hh : h = 10) -- given height
  (hθ : θ = π / 2) -- 90° in radians
  : r^2 * θ * h = 320 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l4054_405456


namespace NUMINAMATH_CALUDE_least_months_to_triple_l4054_405431

/-- The initial borrowed amount in dollars -/
def initial_amount : ℝ := 1000

/-- The monthly interest rate as a decimal -/
def monthly_rate : ℝ := 0.06

/-- The function that calculates the amount owed after t months -/
def amount_owed (t : ℕ) : ℝ := initial_amount * (1 + monthly_rate) ^ t

/-- Theorem stating that 17 is the least number of months for which the amount owed exceeds three times the initial amount -/
theorem least_months_to_triple : 
  (∀ k : ℕ, k < 17 → amount_owed k ≤ 3 * initial_amount) ∧ 
  amount_owed 17 > 3 * initial_amount :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l4054_405431


namespace NUMINAMATH_CALUDE_sine_cosine_product_l4054_405426

theorem sine_cosine_product (α : Real) : 
  (∃ P : ℝ × ℝ, P.1 = Real.cos α ∧ P.2 = Real.sin α ∧ P.2 = -2 * P.1) →
  Real.sin α * Real.cos α = -2/5 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_product_l4054_405426


namespace NUMINAMATH_CALUDE_denis_neighbors_l4054_405405

-- Define the students
inductive Student : Type
| Anya : Student
| Borya : Student
| Vera : Student
| Gena : Student
| Denis : Student

-- Define the line as a list of students
def Line := List Student

-- Define a function to check if two students are next to each other in the line
def next_to (s1 s2 : Student) (line : Line) : Prop :=
  ∃ i, (line.get? i = some s1 ∧ line.get? (i+1) = some s2) ∨
       (line.get? i = some s2 ∧ line.get? (i+1) = some s1)

-- Define the conditions
def valid_line (line : Line) : Prop :=
  (line.length = 5) ∧
  (line.head? = some Student.Borya) ∧
  (next_to Student.Vera Student.Anya line) ∧
  (¬ next_to Student.Vera Student.Gena line) ∧
  (¬ next_to Student.Anya Student.Borya line) ∧
  (¬ next_to Student.Anya Student.Gena line) ∧
  (¬ next_to Student.Borya Student.Gena line)

-- Theorem to prove
theorem denis_neighbors (line : Line) (h : valid_line line) :
  next_to Student.Denis Student.Anya line ∧ next_to Student.Denis Student.Gena line :=
sorry

end NUMINAMATH_CALUDE_denis_neighbors_l4054_405405


namespace NUMINAMATH_CALUDE_tangent_sum_product_l4054_405407

theorem tangent_sum_product (α β γ : Real) (h : α + β + γ = 2 * Real.pi) :
  Real.tan (α / 2) + Real.tan (β / 2) + Real.tan (γ / 2) = 
  Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_product_l4054_405407


namespace NUMINAMATH_CALUDE_power_of_two_equation_l4054_405414

theorem power_of_two_equation (l : ℤ) : 
  2^2000 - 2^1999 - 3 * 2^1998 + 2^1997 = l * 2^1997 → l = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l4054_405414


namespace NUMINAMATH_CALUDE_odd_red_faces_count_l4054_405432

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with odd number of red faces -/
def count_odd_red_faces (dims : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the correct number of cubes with odd red faces -/
theorem odd_red_faces_count (block : BlockDimensions) 
  (h1 : block.length = 5)
  (h2 : block.width = 5)
  (h3 : block.height = 1) : 
  count_odd_red_faces block = 13 := by
  sorry

end NUMINAMATH_CALUDE_odd_red_faces_count_l4054_405432


namespace NUMINAMATH_CALUDE_mrs_heine_dogs_l4054_405429

theorem mrs_heine_dogs (total_biscuits : ℕ) (biscuits_per_dog : ℕ) (num_dogs : ℕ) : 
  total_biscuits = 6 →
  biscuits_per_dog = 3 →
  total_biscuits = num_dogs * biscuits_per_dog →
  num_dogs = 2 := by
sorry

end NUMINAMATH_CALUDE_mrs_heine_dogs_l4054_405429


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l4054_405486

theorem quadratic_equation_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - x + a = 0) → a ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l4054_405486


namespace NUMINAMATH_CALUDE_point_on_line_l4054_405458

theorem point_on_line (m n p : ℝ) : 
  (m = n / 3 - 2 / 5) → 
  (m + p = (n + 9) / 3 - 2 / 5) → 
  p = 3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l4054_405458


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l4054_405475

/-- A quadratic function f(x) = ax² + bx + c with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The value of the quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_coefficient (f : QuadraticFunction) 
  (vertex_x : f.eval 2 = 5)
  (point : f.eval 1 = 2) : 
  f.a = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l4054_405475


namespace NUMINAMATH_CALUDE_number_of_boys_l4054_405474

/-- Given a school with girls and boys, prove the number of boys. -/
theorem number_of_boys (girls boys : ℕ) 
  (h1 : girls = 635)
  (h2 : boys = girls + 510) : 
  boys = 1145 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l4054_405474


namespace NUMINAMATH_CALUDE_breads_after_five_thieves_l4054_405455

/-- The number of breads remaining after a thief takes their share. -/
def remaining_breads (initial : ℕ) (thief : ℕ) : ℚ :=
  if thief = 0 then initial
  else (remaining_breads initial (thief - 1) / 2) - 1/2

/-- The theorem stating that after 5 thieves, 3 breads remain from an initial count of 127. -/
theorem breads_after_five_thieves :
  ⌊remaining_breads 127 5⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_breads_after_five_thieves_l4054_405455


namespace NUMINAMATH_CALUDE_ping_pong_tournament_participants_ping_pong_tournament_solution_l4054_405479

/-- Represents a ping-pong tournament with elimination rules -/
structure PingPongTournament where
  participants : ℕ
  games_played : ℕ
  remaining_players : ℕ

/-- Conditions for our specific tournament -/
def tournament_conditions (t : PingPongTournament) : Prop :=
  t.games_played = 29 ∧ t.remaining_players = 2

/-- Theorem stating the number of participants in the tournament -/
theorem ping_pong_tournament_participants 
  (t : PingPongTournament) 
  (h : tournament_conditions t) : 
  t.participants = 16 := by
  sorry

/-- Main theorem combining the structure and the proof -/
theorem ping_pong_tournament_solution : 
  ∃ t : PingPongTournament, tournament_conditions t ∧ t.participants = 16 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_tournament_participants_ping_pong_tournament_solution_l4054_405479


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4054_405403

def A : Set ℝ := {x | -5 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 9 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | -3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4054_405403


namespace NUMINAMATH_CALUDE_order_of_abc_l4054_405428

-- Define the constants
noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log (1/2)
noncomputable def c : ℝ := Real.log 2 / Real.log (1/3)

-- State the theorem
theorem order_of_abc : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l4054_405428


namespace NUMINAMATH_CALUDE_quadratic_congruence_solution_l4054_405420

theorem quadratic_congruence_solution (p : ℕ) (hp : Nat.Prime p) :
  ∃ n : ℤ, (6 * n^2 + 5 * n + 1) % p = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_congruence_solution_l4054_405420


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l4054_405459

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define a predicate for a line being in a plane
variable (lineInPlane : Line → Plane → Prop)

variable (α β : Plane)
variable (l : Line)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary :
  (α ≠ β) →
  (lineInPlane l α) →
  (parallelPlanes α β → parallelLinePlane l β) ∧
  ∃ γ : Plane, (parallelLinePlane l γ ∧ ¬parallelPlanes α γ) :=
by sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l4054_405459


namespace NUMINAMATH_CALUDE_comparison_theorem_l4054_405435

theorem comparison_theorem (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : n ≥ 2) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l4054_405435


namespace NUMINAMATH_CALUDE_store_discount_l4054_405442

theorem store_discount (original_price : ℝ) (original_price_pos : 0 < original_price) : 
  let sale_price := 0.5 * original_price
  let coupon_discount := 0.2
  let promotion_discount := 0.1
  let final_price := (1 - promotion_discount) * ((1 - coupon_discount) * sale_price)
  (original_price - final_price) / original_price = 0.64 :=
by sorry

end NUMINAMATH_CALUDE_store_discount_l4054_405442


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l4054_405404

/-- A linear function y = (m-3)x + 6 + 2m decreases as x increases if and only if m < 3 -/
theorem linear_function_decreasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((m - 3) * x₁ + 6 + 2 * m) > ((m - 3) * x₂ + 6 + 2 * m)) ↔ m < 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l4054_405404


namespace NUMINAMATH_CALUDE_exists_word_with_multiple_associations_l4054_405433

-- Define the alphabet A
def A : Type := Char

-- Define the set of all words over A
def A_star : Type := List A

-- Define the transducer T'
def T' : A_star → Set A_star := sorry

-- Define the property of a word having multiple associations
def has_multiple_associations (v : A_star) : Prop :=
  ∃ (w1 w2 : A_star), w1 ∈ T' v ∧ w2 ∈ T' v ∧ w1 ≠ w2

-- Theorem statement
theorem exists_word_with_multiple_associations :
  ∃ (v : A_star), has_multiple_associations v := by sorry

end NUMINAMATH_CALUDE_exists_word_with_multiple_associations_l4054_405433


namespace NUMINAMATH_CALUDE_rental_cost_difference_l4054_405454

/-- Calculates the total cost of renting a boat for two days with a discount on the second day -/
def total_cost (daily_rental : ℝ) (hourly_rental : ℝ) (hourly_fuel : ℝ) (hours_per_day : ℝ) (discount_rate : ℝ) : ℝ :=
  let first_day := daily_rental + hourly_fuel * hours_per_day
  let second_day := (daily_rental + hourly_fuel * hours_per_day) * (1 - discount_rate)
  first_day + second_day + hourly_rental * hours_per_day * 2

/-- The difference in cost between renting a ski boat and a sailboat -/
theorem rental_cost_difference : 
  let sailboat_cost := total_cost 60 0 10 3 0.1
  let ski_boat_cost := total_cost 0 80 20 3 0.1
  ski_boat_cost - sailboat_cost = 402 := by sorry

end NUMINAMATH_CALUDE_rental_cost_difference_l4054_405454


namespace NUMINAMATH_CALUDE_mean_home_runs_l4054_405423

def home_runs : List ℕ := [5, 6, 7, 8, 9]
def players : List ℕ := [4, 5, 3, 2, 2]

theorem mean_home_runs :
  let total_hrs := (List.zip home_runs players).map (fun (hr, p) => hr * p) |>.sum
  let total_players := players.sum
  (total_hrs : ℚ) / total_players = 105 / 16 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l4054_405423


namespace NUMINAMATH_CALUDE_std_dev_of_scaled_data_l4054_405436

-- Define the type for our data set
def DataSet := Fin 100 → ℝ

-- Define the variance of a data set
noncomputable def variance (data : DataSet) : ℝ := sorry

-- Define the standard deviation of a data set
noncomputable def std_dev (data : DataSet) : ℝ := Real.sqrt (variance data)

-- Define a function that multiplies each element of a data set by 3
def scale_by_3 (data : DataSet) : DataSet := λ i => 3 * data i

-- Our theorem
theorem std_dev_of_scaled_data (original_data : DataSet) 
  (h : variance original_data = 2) : 
  std_dev (scale_by_3 original_data) = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_std_dev_of_scaled_data_l4054_405436


namespace NUMINAMATH_CALUDE_bell_weight_ratio_l4054_405411

/-- Given three bells with specific weight relationships, prove the ratio of the third to second bell's weight --/
theorem bell_weight_ratio :
  ∀ (bell1 bell2 bell3 : ℝ),
  bell1 = 50 →
  bell2 = 2 * bell1 →
  bell1 + bell2 + bell3 = 550 →
  bell3 / bell2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bell_weight_ratio_l4054_405411


namespace NUMINAMATH_CALUDE_check_max_value_l4054_405496

theorem check_max_value (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →  -- x is a two-digit number
  (10 ≤ y ∧ y ≤ 99) →  -- y is a two-digit number
  (100 * x + y) - (100 * y + x) = 2061 →  -- difference between correct and incorrect amounts
  x ≤ 78 :=
by sorry

end NUMINAMATH_CALUDE_check_max_value_l4054_405496


namespace NUMINAMATH_CALUDE_marbleSelectionWays_l4054_405485

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 marbles from 3 pairs of different colored marbles -/
def twoFromThreePairs : ℕ := sorry

/-- The total number of marbles -/
def totalMarbles : ℕ := 15

/-- The number of special colored marbles (red, green, blue) -/
def specialColoredMarbles : ℕ := 6

/-- The number of marbles to be chosen -/
def marblesToChoose : ℕ := 5

/-- The number of special colored marbles that must be chosen -/
def specialMarblesToChoose : ℕ := 2

theorem marbleSelectionWays : 
  twoFromThreePairs * choose (totalMarbles - specialColoredMarbles) (marblesToChoose - specialMarblesToChoose) = 1008 :=
sorry

end NUMINAMATH_CALUDE_marbleSelectionWays_l4054_405485


namespace NUMINAMATH_CALUDE_library_visitors_month_length_l4054_405497

theorem library_visitors_month_length :
  ∀ (s : ℕ) (d : ℕ),
    s > 0 →  -- At least one Sunday
    s + d > 0 →  -- Total days in month is positive
    150 * s + 120 * d = 125 * (s + d) →  -- Equation balancing total visitors
    s = 5 ∧ d = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_library_visitors_month_length_l4054_405497


namespace NUMINAMATH_CALUDE_seating_arrangements_l4054_405491

/-- The number of ways to arrange n people in a row --/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row with k specific people in consecutive seats --/
def consecutiveArrangements (n k : ℕ) : ℕ := 
  Nat.factorial (n - k + 1) * Nat.factorial k

/-- The number of people to be seated --/
def totalPeople : ℕ := 10

/-- The number of people who refuse to sit consecutively --/
def refusingPeople : ℕ := 4

theorem seating_arrangements :
  totalArrangements totalPeople - consecutiveArrangements totalPeople refusingPeople = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l4054_405491


namespace NUMINAMATH_CALUDE_bathing_suits_for_men_l4054_405441

theorem bathing_suits_for_men (total : ℕ) (women : ℕ) (men : ℕ) : 
  total = 19766 → women = 4969 → men = total - women → men = 14797 := by
  sorry

end NUMINAMATH_CALUDE_bathing_suits_for_men_l4054_405441


namespace NUMINAMATH_CALUDE_base6_265_equals_base10_113_l4054_405444

/-- Converts a base-6 number to base 10 --/
def base6ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 6^2 + tens * 6^1 + ones * 6^0

/-- Theorem: The base-6 number 265₆ is equal to 113 in base 10 --/
theorem base6_265_equals_base10_113 : base6ToBase10 2 6 5 = 113 := by
  sorry

end NUMINAMATH_CALUDE_base6_265_equals_base10_113_l4054_405444


namespace NUMINAMATH_CALUDE_homework_students_l4054_405445

theorem homework_students (total : ℕ) (silent_reading : ℕ) (board_games : ℕ) : 
  total = 24 →
  silent_reading = total / 2 →
  board_games = total / 3 →
  total - (silent_reading + board_games) = 4 := by
sorry

end NUMINAMATH_CALUDE_homework_students_l4054_405445


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4054_405464

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 6*x₁ + 9*k = 0 ∧ x₂^2 - 6*x₂ + 9*k = 0) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4054_405464


namespace NUMINAMATH_CALUDE_sharon_coffee_cost_l4054_405437

/-- Calculates the total cost of coffee pods for a vacation. -/
def coffee_cost (days : ℕ) (pods_per_day : ℕ) (pods_per_box : ℕ) (cost_per_box : ℚ) : ℚ :=
  let total_pods := days * pods_per_day
  let boxes_needed := (total_pods + pods_per_box - 1) / pods_per_box  -- Ceiling division
  boxes_needed * cost_per_box

/-- Proves that Sharon's coffee cost for her vacation is $32.00 -/
theorem sharon_coffee_cost :
  coffee_cost 40 3 30 8 = 32 :=
by
  sorry

#eval coffee_cost 40 3 30 8

end NUMINAMATH_CALUDE_sharon_coffee_cost_l4054_405437


namespace NUMINAMATH_CALUDE_problem_solution_l4054_405467

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x > -3 ∧ x ≤ 3}

theorem problem_solution :
  (A = {x | -2 < x ∧ x < 3}) ∧
  (Set.compl (A ∩ B) = {x | x ≤ -2 ∨ x ≥ 3}) ∧
  ((Set.compl A) ∩ B = {x | -3 < x ∧ x ≤ -2 ∨ x = 3}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4054_405467


namespace NUMINAMATH_CALUDE_chessboard_coverage_l4054_405488

/-- Represents a chessboard square --/
inductive Square
| Black
| White

/-- Represents a 2x1 tile --/
structure Tile :=
  (first : Square)
  (second : Square)

/-- Represents a chessboard --/
def Chessboard := List (List Square)

/-- Creates a standard 8x8 chessboard --/
def standardChessboard : Chessboard :=
  sorry

/-- Removes two squares of different colors from the chessboard --/
def removeSquares (board : Chessboard) (pos1 pos2 : Nat × Nat) : Chessboard :=
  sorry

/-- Checks if a given tile placement is valid on the board --/
def isValidPlacement (board : Chessboard) (tile : Tile) (pos : Nat × Nat) : Bool :=
  sorry

/-- Theorem: A chessboard with two squares of different colors removed can always be covered with 2x1 tiles --/
theorem chessboard_coverage (board : Chessboard) (pos1 pos2 : Nat × Nat) :
  let removedBoard := removeSquares standardChessboard pos1 pos2
  ∃ (tilePlacements : List (Tile × (Nat × Nat))),
    (∀ (placement : Tile × (Nat × Nat)), placement ∈ tilePlacements →
      isValidPlacement removedBoard placement.fst placement.snd) ∧
    (∀ (square : Nat × Nat), square ≠ pos1 ∧ square ≠ pos2 →
      ∃ (placement : Tile × (Nat × Nat)), placement ∈ tilePlacements ∧
        (placement.snd = square ∨ (placement.snd.1 + 1, placement.snd.2) = square)) :=
  sorry


end NUMINAMATH_CALUDE_chessboard_coverage_l4054_405488


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l4054_405452

theorem max_value_of_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  (∀ x y : ℝ, x * y = 1 → x > y → y ≥ 2/3 → (x - y) / (x^2 + y^2) ≤ (a - b) / (a^2 + b^2)) →
  (a - b) / (a^2 + b^2) = 30/97 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l4054_405452


namespace NUMINAMATH_CALUDE_divisors_of_8_factorial_l4054_405443

/-- The number of positive divisors of n! -/
def num_divisors_factorial (n : ℕ) : ℕ :=
  sorry

/-- 8! has 96 positive divisors -/
theorem divisors_of_8_factorial :
  num_divisors_factorial 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8_factorial_l4054_405443


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l4054_405471

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a^3 + b^3 = 152) : 
  a * b = 15 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l4054_405471


namespace NUMINAMATH_CALUDE_triangle_properties_l4054_405450

/-- Given a triangle ABC with the following properties:
    - a, b, c are sides opposite to angles A, B, C respectively
    - a = 2√3
    - A = π/3
    - Area S = 2√3
    - sin(C-B) = sin(2B) - sin(A)
    Prove the properties of sides b, c and the shape of the triangle -/
theorem triangle_properties (a b c A B C S : Real) : 
  a = 2 * Real.sqrt 3 →
  A = π / 3 →
  S = 2 * Real.sqrt 3 →
  S = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  Real.sin (C - B) = Real.sin (2*B) - Real.sin A →
  ((b = 2 ∧ c = 4) ∨ (b = 4 ∧ c = 2)) ∧
  (B = π / 2 ∨ C = B) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4054_405450


namespace NUMINAMATH_CALUDE_elaine_rent_percentage_l4054_405470

/-- The percentage of Elaine's annual earnings spent on rent last year -/
def rent_percentage_last_year : ℝ := 20

/-- Elaine's earnings this year as a percentage of last year's earnings -/
def earnings_ratio : ℝ := 115

/-- The percentage of Elaine's earnings spent on rent this year -/
def rent_percentage_this_year : ℝ := 25

/-- The ratio of this year's rent expenditure to last year's rent expenditure -/
def rent_expenditure_ratio : ℝ := 143.75

theorem elaine_rent_percentage :
  rent_percentage_this_year * earnings_ratio / 100 = 
  rent_expenditure_ratio * rent_percentage_last_year / 100 :=
sorry

end NUMINAMATH_CALUDE_elaine_rent_percentage_l4054_405470


namespace NUMINAMATH_CALUDE_number_of_small_boxes_correct_number_of_small_boxes_l4054_405492

theorem number_of_small_boxes 
  (dolls_per_big_box : ℕ) 
  (dolls_per_small_box : ℕ) 
  (num_big_boxes : ℕ) 
  (total_dolls : ℕ) : ℕ :=
  let remaining_dolls := total_dolls - dolls_per_big_box * num_big_boxes
  remaining_dolls / dolls_per_small_box

theorem correct_number_of_small_boxes :
  number_of_small_boxes 7 4 5 71 = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_small_boxes_correct_number_of_small_boxes_l4054_405492


namespace NUMINAMATH_CALUDE_oblique_square_area_l4054_405476

/-- Represents a square in an oblique projection drawing as a parallelogram -/
structure ObliqueSquare where
  parallelogram_side : ℝ

/-- The possible areas of the original square given its oblique projection -/
def possible_areas (os : ObliqueSquare) : Set ℝ :=
  {16, 64}

/-- 
Given a square represented as a parallelogram in an oblique projection drawing,
if one side of the parallelogram is 4, then the area of the original square
is either 16 or 64.
-/
theorem oblique_square_area (os : ObliqueSquare) 
  (h : os.parallelogram_side = 4) : 
  ∀ a ∈ possible_areas os, a = 16 ∨ a = 64 := by
  sorry

end NUMINAMATH_CALUDE_oblique_square_area_l4054_405476


namespace NUMINAMATH_CALUDE_men_per_table_l4054_405483

theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ)
  (h1 : num_tables = 7)
  (h2 : women_per_table = 7)
  (h3 : total_customers = 63) :
  (total_customers - num_tables * women_per_table) / num_tables = 2 :=
by sorry

end NUMINAMATH_CALUDE_men_per_table_l4054_405483


namespace NUMINAMATH_CALUDE_max_value_constraint_l4054_405427

theorem max_value_constraint (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) : 
  ∃ (M : ℝ), M = 3.2 ∧ ∀ (x y z : ℝ), 9*x^2 + 4*y^2 + 25*z^2 = 1 → 6*x + 3*y + 10*z ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l4054_405427


namespace NUMINAMATH_CALUDE_trigonometric_sum_zero_l4054_405424

theorem trigonometric_sum_zero (α : ℝ) : 
  Real.sin (2 * α - 3/2 * Real.pi) + Real.cos (2 * α - 8/3 * Real.pi) + Real.cos (2/3 * Real.pi + 2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_zero_l4054_405424


namespace NUMINAMATH_CALUDE_chip_sales_ratio_l4054_405402

/-- Represents the sales data for a convenience store's chip sales over a month. -/
structure ChipSales where
  total : ℕ
  first_week : ℕ
  third_week : ℕ
  fourth_week : ℕ

/-- Calculates the ratio of second week sales to first week sales. -/
def sales_ratio (sales : ChipSales) : ℚ :=
  let second_week := sales.total - sales.first_week - sales.third_week - sales.fourth_week
  (second_week : ℚ) / sales.first_week

/-- Theorem stating that given the specific sales conditions, the ratio of second week to first week sales is 3:1. -/
theorem chip_sales_ratio :
  ∀ (sales : ChipSales),
    sales.total = 100 ∧
    sales.first_week = 15 ∧
    sales.third_week = 20 ∧
    sales.fourth_week = 20 →
    sales_ratio sales = 3 := by
  sorry

end NUMINAMATH_CALUDE_chip_sales_ratio_l4054_405402
