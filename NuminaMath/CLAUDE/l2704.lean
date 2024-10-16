import Mathlib

namespace NUMINAMATH_CALUDE_slope_range_l2704_270411

-- Define the points
def A₁ : ℝ × ℝ := (-2, 0)
def A₂ : ℝ × ℝ := (2, 0)
def B₁ (x : ℝ) : ℝ × ℝ := (x, 2)
def B₂ (x : ℝ) : ℝ × ℝ := (x, -2)
def P (x y : ℝ) : ℝ × ℝ := (x, y)
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the dot product
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Define the equation of the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the condition for the line passing through B and intersecting the ellipse
def intersects_ellipse (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    on_ellipse x₁ y₁ ∧
    on_ellipse x₂ y₂ ∧
    y₁ = k * x₁ + 2 ∧
    y₂ = k * x₂ + 2 ∧
    x₁ ≠ x₂

-- Define the condition for the ratio of triangle areas
def area_ratio_condition (x₁ x₂ : ℝ) : Prop :=
  1/2 < |x₁| / |x₂| ∧ |x₁| / |x₂| < 1

-- Main theorem
theorem slope_range :
  ∀ k : ℝ,
    intersects_ellipse k ∧
    (∃ x₁ x₂ : ℝ, area_ratio_condition x₁ x₂) ↔
    (k > Real.sqrt 2 / 2 ∧ k < 3 * Real.sqrt 14 / 14) ∨
    (k < -Real.sqrt 2 / 2 ∧ k > -3 * Real.sqrt 14 / 14) :=
sorry

end NUMINAMATH_CALUDE_slope_range_l2704_270411


namespace NUMINAMATH_CALUDE_square_sum_eq_double_product_implies_zero_l2704_270495

theorem square_sum_eq_double_product_implies_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eq_double_product_implies_zero_l2704_270495


namespace NUMINAMATH_CALUDE_cooler_cans_count_l2704_270472

/-- Given a cooler with cherry soda and orange pop, where there are twice as many
    cans of orange pop as cherry soda, and there are 8 cherry sodas,
    prove that the total number of cans in the cooler is 24. -/
theorem cooler_cans_count (cherry_soda orange_pop : ℕ) : 
  cherry_soda = 8 →
  orange_pop = 2 * cherry_soda →
  cherry_soda + orange_pop = 24 := by
  sorry

end NUMINAMATH_CALUDE_cooler_cans_count_l2704_270472


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l2704_270422

theorem min_value_quadratic_form (x y z : ℝ) :
  x^2 + x*y + y^2 + y*z + z^2 ≥ 0 ∧
  (x^2 + x*y + y^2 + y*z + z^2 = 0 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l2704_270422


namespace NUMINAMATH_CALUDE_log_equation_solution_l2704_270482

theorem log_equation_solution (x : ℝ) :
  x > 1 →
  (Real.log (x^3 - 9*x + 8) / Real.log (x + 1)) * (Real.log (x + 1) / Real.log (x - 1)) = 3 →
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2704_270482


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2704_270432

/-- Given two vectors a and b in ℝ², if a is perpendicular to (a - b), then the x-coordinate of b is 9. -/
theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.2 = -2 →
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) →
  b.1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2704_270432


namespace NUMINAMATH_CALUDE_bus_patrons_count_l2704_270451

/-- The number of patrons a golf cart can fit -/
def golf_cart_capacity : ℕ := 3

/-- The number of patrons who came in cars -/
def car_patrons : ℕ := 12

/-- The number of golf carts needed to transport all patrons -/
def golf_carts_needed : ℕ := 13

/-- The number of patrons who came from a bus -/
def bus_patrons : ℕ := golf_carts_needed * golf_cart_capacity - car_patrons

theorem bus_patrons_count : bus_patrons = 27 := by
  sorry

end NUMINAMATH_CALUDE_bus_patrons_count_l2704_270451


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l2704_270418

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l2704_270418


namespace NUMINAMATH_CALUDE_teacher_fills_thermos_once_per_day_l2704_270443

/-- Represents the teacher's coffee drinking habits --/
structure CoffeeDrinkingHabits where
  thermos_capacity : ℝ
  school_days_per_week : ℕ
  current_weekly_consumption : ℝ
  consumption_reduction_factor : ℝ

/-- Calculates the number of times the thermos is filled per day --/
def thermos_fills_per_day (habits : CoffeeDrinkingHabits) : ℕ :=
  sorry

/-- Theorem stating that the teacher fills her thermos once per day --/
theorem teacher_fills_thermos_once_per_day (habits : CoffeeDrinkingHabits) 
  (h1 : habits.thermos_capacity = 20)
  (h2 : habits.school_days_per_week = 5)
  (h3 : habits.current_weekly_consumption = 40)
  (h4 : habits.consumption_reduction_factor = 1/4) :
  thermos_fills_per_day habits = 1 := by
  sorry

end NUMINAMATH_CALUDE_teacher_fills_thermos_once_per_day_l2704_270443


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_7_l2704_270476

theorem smallest_lcm_with_gcd_7 :
  ∃ (m n : ℕ), 
    1000 ≤ m ∧ m < 10000 ∧
    1000 ≤ n ∧ n < 10000 ∧
    Nat.gcd m n = 7 ∧
    Nat.lcm m n = 144001 ∧
    ∀ (a b : ℕ), 
      1000 ≤ a ∧ a < 10000 ∧
      1000 ≤ b ∧ b < 10000 ∧
      Nat.gcd a b = 7 →
      Nat.lcm a b ≥ 144001 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_7_l2704_270476


namespace NUMINAMATH_CALUDE_vision_data_median_l2704_270430

structure VisionData where
  values : List Float
  frequencies : List Nat
  total_students : Nat

def median (data : VisionData) : Float :=
  sorry

theorem vision_data_median :
  let data : VisionData := {
    values := [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    frequencies := [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5],
    total_students := 39
  }
  median data = 4.6 := by sorry

end NUMINAMATH_CALUDE_vision_data_median_l2704_270430


namespace NUMINAMATH_CALUDE_pizza_fraction_l2704_270466

theorem pizza_fraction (pieces_per_day : ℕ) (days : ℕ) (whole_pizzas : ℕ) :
  pieces_per_day = 3 →
  days = 72 →
  whole_pizzas = 27 →
  (1 : ℚ) / (pieces_per_day * days / whole_pizzas) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l2704_270466


namespace NUMINAMATH_CALUDE_major_axis_length_tangent_ellipse_major_axis_l2704_270435

/-- An ellipse with foci at (4, 1 + 2√3) and (4, 1 - 2√3), tangent to both x and y axes -/
structure TangentEllipse where
  /-- The ellipse is tangent to the x-axis -/
  tangent_x : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_y : Bool
  /-- The x-coordinate of both foci -/
  focus_x : ℝ
  /-- The y-coordinate of the first focus -/
  focus_y1 : ℝ
  /-- The y-coordinate of the second focus -/
  focus_y2 : ℝ
  /-- Ensure the foci are correctly positioned -/
  foci_constraint : focus_x = 4 ∧ focus_y1 = 1 + 2 * Real.sqrt 3 ∧ focus_y2 = 1 - 2 * Real.sqrt 3

/-- The length of the major axis of the ellipse is 2 -/
theorem major_axis_length (e : TangentEllipse) : ℝ :=
  2

/-- The theorem stating that the major axis length of the given ellipse is 2 -/
theorem tangent_ellipse_major_axis (e : TangentEllipse) (h1 : e.tangent_x = true) (h2 : e.tangent_y = true) :
  major_axis_length e = 2 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_tangent_ellipse_major_axis_l2704_270435


namespace NUMINAMATH_CALUDE_tens_digit_of_23_pow_1987_l2704_270408

theorem tens_digit_of_23_pow_1987 : ∃ k : ℕ, 23^1987 ≡ 40 + k [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_pow_1987_l2704_270408


namespace NUMINAMATH_CALUDE_manny_marbles_l2704_270452

theorem manny_marbles (total : ℕ) (mario_ratio manny_ratio given : ℕ) : 
  total = 36 →
  mario_ratio = 4 →
  manny_ratio = 5 →
  given = 2 →
  (manny_ratio * (total / (mario_ratio + manny_ratio))) - given = 18 :=
by sorry

end NUMINAMATH_CALUDE_manny_marbles_l2704_270452


namespace NUMINAMATH_CALUDE_intersection_complement_and_B_l2704_270403

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

theorem intersection_complement_and_B : 
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_and_B_l2704_270403


namespace NUMINAMATH_CALUDE_camp_girls_count_l2704_270479

theorem camp_girls_count (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 133 → difference = 33 → girls + (girls + difference) = total → girls = 50 := by
  sorry

end NUMINAMATH_CALUDE_camp_girls_count_l2704_270479


namespace NUMINAMATH_CALUDE_fraction_equality_l2704_270492

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 4) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2704_270492


namespace NUMINAMATH_CALUDE_reginalds_apple_sales_l2704_270464

/-- Represents the problem of calculating the number of apples sold by Reginald --/
theorem reginalds_apple_sales :
  let apple_price : ℚ := 5 / 4  -- $1.25 per apple
  let bike_cost : ℚ := 80
  let repair_cost_ratio : ℚ := 1 / 4  -- 25% of bike cost
  let remaining_ratio : ℚ := 1 / 5  -- 1/5 of earnings remain after repairs
  let apples_per_set : ℕ := 6  -- 5 paid + 1 free
  let paid_apples_per_set : ℕ := 5

  ∃ (total_apples : ℕ),
    total_apples = 120 ∧
    total_apples % apples_per_set = 0 ∧
    let total_sets := total_apples / apples_per_set
    let total_earnings := (total_sets * paid_apples_per_set : ℚ) * apple_price
    let repair_cost := bike_cost * repair_cost_ratio
    total_earnings * remaining_ratio = total_earnings - repair_cost :=
by
  sorry

end NUMINAMATH_CALUDE_reginalds_apple_sales_l2704_270464


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_l2704_270470

-- Define the function f(x)
def f (t : ℝ) (x : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

-- Define the derivative of f(x)
def f_deriv (t : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (t - 1) * x

theorem tangent_lines_theorem (t k : ℝ) (hk : k ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    f_deriv t x₁ = k ∧
    f_deriv t x₂ = k ∧
    f t x₁ = 2 * x₁ - 1 ∧
    f t x₂ = 2 * x₂ - 1) →
  t + k = 7 := by
sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_l2704_270470


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_with_roots_product_one_l2704_270490

theorem infinitely_many_pairs_with_roots_product_one :
  ∀ n : ℕ, n > 2 →
  ∃ a b : ℤ,
    (∃ x y : ℝ, x ≠ y ∧ x * y = 1 ∧
      x^2019 = a * x + b ∧ y^2019 = a * y + b) ∧
    (∀ m : ℕ, m > 2 → m ≠ n →
      ∃ c d : ℤ, c ≠ a ∨ d ≠ b ∧
        (∃ u v : ℝ, u ≠ v ∧ u * v = 1 ∧
          u^2019 = c * u + d ∧ v^2019 = c * v + d)) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_with_roots_product_one_l2704_270490


namespace NUMINAMATH_CALUDE_escalator_steps_l2704_270436

/-- The number of steps Al counts walking down the escalator -/
def al_steps : ℕ := 150

/-- The number of steps Bob counts walking up the escalator -/
def bob_steps : ℕ := 75

/-- The ratio of Al's walking speed to Bob's walking speed -/
def speed_ratio : ℕ := 3

/-- The number of steps visible on the escalator at any given time -/
def visible_steps : ℕ := 120

/-- Theorem stating that given the conditions, the number of visible steps on the escalator is 120 -/
theorem escalator_steps : 
  ∀ (al_count bob_count : ℕ) (speed_ratio : ℕ),
    al_count = al_steps →
    bob_count = bob_steps →
    speed_ratio = 3 →
    visible_steps = 120 := by sorry

end NUMINAMATH_CALUDE_escalator_steps_l2704_270436


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_inequality_l2704_270407

theorem smallest_positive_integer_satisfying_inequality :
  ∀ x : ℕ, x > 0 → (x + 3 < 2 * x - 7) → x ≥ 11 ∧
  (11 + 3 < 2 * 11 - 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_inequality_l2704_270407


namespace NUMINAMATH_CALUDE_farmland_equations_correct_l2704_270440

/-- Represents the farmland purchase problem -/
structure FarmlandProblem where
  total_acres : ℕ
  good_cost_per_acre : ℚ
  bad_cost_per_seven_acres : ℚ
  total_spent : ℚ

/-- Represents the system of equations for the farmland problem -/
def farmland_equations (p : FarmlandProblem) (x y : ℚ) : Prop :=
  x + y = p.total_acres ∧
  p.good_cost_per_acre * x + (p.bad_cost_per_seven_acres / 7) * y = p.total_spent

/-- Theorem stating that the system of equations correctly represents the farmland problem -/
theorem farmland_equations_correct (p : FarmlandProblem) (x y : ℚ) :
  p.total_acres = 100 →
  p.good_cost_per_acre = 300 →
  p.bad_cost_per_seven_acres = 500 →
  p.total_spent = 10000 →
  farmland_equations p x y ↔
    (x + y = 100 ∧ 300 * x + (500 / 7) * y = 10000) :=
by sorry

end NUMINAMATH_CALUDE_farmland_equations_correct_l2704_270440


namespace NUMINAMATH_CALUDE_peanuts_remaining_l2704_270480

def initial_peanuts : ℕ := 148
def bonita_eaten : ℕ := 29

theorem peanuts_remaining : 
  initial_peanuts - (initial_peanuts / 4) - bonita_eaten = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_peanuts_remaining_l2704_270480


namespace NUMINAMATH_CALUDE_only_solution_is_48_l2704_270437

/-- Product of digits function -/
def p (A : ℕ) : ℕ :=
  sorry

/-- Theorem: 48 is the only natural number satisfying A = 1.5 * p(A) -/
theorem only_solution_is_48 :
  ∀ A : ℕ, A = (3/2 : ℚ) * p A ↔ A = 48 :=
by sorry

end NUMINAMATH_CALUDE_only_solution_is_48_l2704_270437


namespace NUMINAMATH_CALUDE_truck_gas_calculation_l2704_270484

/-- Calculates the amount of gas already in a truck's tank given the truck's fuel efficiency, 
    distance to travel, and additional gas needed to complete the journey. -/
def gas_in_tank (miles_per_gallon : ℚ) (distance : ℚ) (additional_gas : ℚ) : ℚ :=
  distance / miles_per_gallon - additional_gas

/-- Theorem stating that for a truck traveling 3 miles per gallon, needing to cover 90 miles,
    and requiring 18 more gallons, the amount of gas already in the tank is 12 gallons. -/
theorem truck_gas_calculation :
  gas_in_tank 3 90 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_truck_gas_calculation_l2704_270484


namespace NUMINAMATH_CALUDE_day_before_day_after_tomorrow_l2704_270499

-- Define the days of the week
inductive Day : Type
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day
  | sunday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday
  | Day.sunday => Day.monday

-- Define a function to get the previous day
def prevDay (d : Day) : Day :=
  match d with
  | Day.monday => Day.sunday
  | Day.tuesday => Day.monday
  | Day.wednesday => Day.tuesday
  | Day.thursday => Day.wednesday
  | Day.friday => Day.thursday
  | Day.saturday => Day.friday
  | Day.sunday => Day.saturday

-- Theorem statement
theorem day_before_day_after_tomorrow (today : Day) :
  today = Day.thursday →
  prevDay (nextDay (nextDay today)) = Day.friday :=
by
  sorry


end NUMINAMATH_CALUDE_day_before_day_after_tomorrow_l2704_270499


namespace NUMINAMATH_CALUDE_clothing_colors_l2704_270497

-- Define the colors
inductive Color
| Red
| Blue

-- Define a structure for a child's clothing
structure Clothing where
  tshirt : Color
  shorts : Color

-- Define the four children
def Alyna : Clothing := sorry
def Bohdan : Clothing := sorry
def Vika : Clothing := sorry
def Grysha : Clothing := sorry

-- Define the theorem
theorem clothing_colors :
  -- Conditions
  (Alyna.tshirt = Color.Red) →
  (Bohdan.tshirt = Color.Red) →
  (Alyna.shorts ≠ Bohdan.shorts) →
  (Vika.tshirt ≠ Grysha.tshirt) →
  (Vika.shorts = Color.Blue) →
  (Grysha.shorts = Color.Blue) →
  (Alyna.tshirt ≠ Vika.tshirt) →
  (Alyna.shorts ≠ Vika.shorts) →
  -- Conclusion
  (Alyna = ⟨Color.Red, Color.Red⟩ ∧
   Bohdan = ⟨Color.Red, Color.Blue⟩ ∧
   Vika = ⟨Color.Blue, Color.Blue⟩ ∧
   Grysha = ⟨Color.Red, Color.Blue⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_clothing_colors_l2704_270497


namespace NUMINAMATH_CALUDE_smallest_y_value_l2704_270469

theorem smallest_y_value (y : ℝ) (h : y > 0) :
  (y / 7 + 2 / (7 * y) = 1 / 3) → y ≥ 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_y_value_l2704_270469


namespace NUMINAMATH_CALUDE_prime_composite_property_l2704_270459

theorem prime_composite_property (n : ℕ) :
  (∀ (a : Fin n → ℕ), Function.Injective a →
    ∃ (i j : Fin n), (a i + a j) / Nat.gcd (a i) (a j) ≥ 2 * n - 1) ∨
  (∃ (a : Fin n → ℕ), Function.Injective a ∧
    ∀ (i j : Fin n), (a i + a j) / Nat.gcd (a i) (a j) < 2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_composite_property_l2704_270459


namespace NUMINAMATH_CALUDE_new_boat_travel_distance_l2704_270421

/-- Calculates the distance traveled by a new boat given the speed increase and the distance traveled by an old boat -/
def new_boat_distance (speed_increase : ℝ) (old_distance : ℝ) : ℝ :=
  old_distance * (1 + speed_increase)

/-- Theorem: Given a new boat traveling 30% faster than an old boat, and the old boat traveling 150 miles,
    the new boat will travel 195 miles in the same time -/
theorem new_boat_travel_distance :
  new_boat_distance 0.3 150 = 195 := by
  sorry

#eval new_boat_distance 0.3 150

end NUMINAMATH_CALUDE_new_boat_travel_distance_l2704_270421


namespace NUMINAMATH_CALUDE_no_solution_iff_m_equals_seven_l2704_270405

theorem no_solution_iff_m_equals_seven (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) ≠ (x - m) / (x - 8)) ↔ m = 7 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_equals_seven_l2704_270405


namespace NUMINAMATH_CALUDE_max_area_parallelogram_in_circle_l2704_270434

/-- A right-angled parallelogram inscribed in a circle of radius r has maximum area when its sides are r√2 -/
theorem max_area_parallelogram_in_circle (r : ℝ) (h : r > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a * b ≤ x * y) ∧
  (x^2 + y^2 = (2*r)^2) ∧
  x = r * Real.sqrt 2 ∧ y = r * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_area_parallelogram_in_circle_l2704_270434


namespace NUMINAMATH_CALUDE_total_winter_clothing_l2704_270448

/-- The number of boxes of winter clothing -/
def num_boxes : ℕ := 3

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 3

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 4

/-- Theorem: The total number of winter clothing pieces is 21 -/
theorem total_winter_clothing : 
  num_boxes * (scarves_per_box + mittens_per_box) = 21 := by
sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l2704_270448


namespace NUMINAMATH_CALUDE_students_just_passed_l2704_270429

/-- The number of students who just passed an examination -/
theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 26 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_all_passed : first_div_percent + second_div_percent < 1) :
  total - (first_div_percent * total).floor - (second_div_percent * total).floor = 60 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l2704_270429


namespace NUMINAMATH_CALUDE_projectile_max_height_l2704_270410

/-- The height function of the projectile -/
def f (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- Theorem stating that the maximum height of the projectile is 161 meters -/
theorem projectile_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 161 := by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2704_270410


namespace NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2704_270423

/-- A type representing a cell in the grid -/
structure Cell where
  row : Fin 22
  col : Fin 22

/-- A function representing the number in each cell -/
def gridValue : Cell → Fin (22^2) := sorry

/-- Two cells are adjacent if they share an edge or vertex -/
def adjacent (c1 c2 : Cell) : Prop := sorry

theorem adjacent_sum_divisible_by_four :
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (gridValue c1 + gridValue c2) % 4 = 0 := by sorry

end NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2704_270423


namespace NUMINAMATH_CALUDE_race_overtake_points_l2704_270456

-- Define the race parameters
def kelly_head_start : ℝ := 3
def kelly_speed : ℝ := 9
def abel_speed : ℝ := 9.5
def chris_speed : ℝ := 10
def chris_start_behind : ℝ := 2
def abel_loss_distance : ℝ := 0.75

-- Define the overtake points
def abel_overtake_kelly : ℝ := 54.75
def chris_overtake_both : ℝ := 56

-- Theorem statement
theorem race_overtake_points : 
  kelly_head_start = 3 ∧ 
  kelly_speed = 9 ∧ 
  abel_speed = 9.5 ∧ 
  chris_speed = 10 ∧ 
  chris_start_behind = 2 ∧
  abel_loss_distance = 0.75 →
  (abel_overtake_kelly = 54.75 ∧ chris_overtake_both = 56) := by
  sorry

end NUMINAMATH_CALUDE_race_overtake_points_l2704_270456


namespace NUMINAMATH_CALUDE_volunteer_allocation_l2704_270487

theorem volunteer_allocation (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 4) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 84 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_allocation_l2704_270487


namespace NUMINAMATH_CALUDE_problem_solution_l2704_270491

def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|

def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

theorem problem_solution :
  (∀ x : ℝ, g x < 6 ↔ -7/5 < x ∧ x < 3) ∧
  (∃ x₁ x₂ : ℝ, f a x₁ = -g x₂) → -13/12 ≤ a ∧ a ≤ 5/12 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2704_270491


namespace NUMINAMATH_CALUDE_day_crew_fraction_of_boxes_l2704_270473

/-- Represents the fraction of boxes loaded by the day crew given the relative productivity
    and size of the night crew compared to the day crew. -/
theorem day_crew_fraction_of_boxes
  (night_worker_productivity : ℚ)  -- Productivity of night worker relative to day worker
  (night_crew_size : ℚ)            -- Size of night crew relative to day crew
  (h1 : night_worker_productivity = 1 / 4)
  (h2 : night_crew_size = 4 / 5) :
  (1 : ℚ) / (1 + night_worker_productivity * night_crew_size) = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_day_crew_fraction_of_boxes_l2704_270473


namespace NUMINAMATH_CALUDE_exactly_three_blue_marbles_l2704_270413

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_picks : ℕ := 7
def num_blue_picks : ℕ := 3

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exactly_three_blue_marbles :
  Nat.choose num_picks num_blue_picks *
  (prob_blue ^ num_blue_picks) *
  (prob_red ^ (num_picks - num_blue_picks)) =
  640 / 1547 := by sorry

end NUMINAMATH_CALUDE_exactly_three_blue_marbles_l2704_270413


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2704_270433

theorem arithmetic_expression_evaluation : 6 + 15 / 3 - 4^2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2704_270433


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l2704_270461

theorem trapezoid_side_length (square_side : ℝ) (trapezoid_area hexagon_area : ℝ) 
  (x : ℝ) : 
  square_side = 1 →
  trapezoid_area = hexagon_area →
  trapezoid_area = 1/4 →
  x = trapezoid_area * 4 / (1 + square_side) →
  x = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l2704_270461


namespace NUMINAMATH_CALUDE_money_duration_l2704_270449

def mowing_earnings : ℕ := 14
def weed_eating_earnings : ℕ := 26
def weekly_spending : ℕ := 5

theorem money_duration : 
  (mowing_earnings + weed_eating_earnings) / weekly_spending = 8 := by
sorry

end NUMINAMATH_CALUDE_money_duration_l2704_270449


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2704_270445

def is_valid (n : ℕ) : Prop :=
  n % 9 = 0 ∧
  n % 2 = 1 ∧
  n % 3 = 1 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 6 = 1

theorem smallest_valid_number : 
  is_valid 361 ∧ ∀ m : ℕ, m < 361 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2704_270445


namespace NUMINAMATH_CALUDE_car_problem_solution_l2704_270498

def car_problem (t : ℝ) : Prop :=
  let v1 : ℝ := 60  -- First speed in km/h
  let v2 : ℝ := 90  -- Second speed in km/h
  let t2 : ℝ := 2/3 -- Time at second speed in hours (40 minutes = 2/3 hour)
  let v_avg : ℝ := 80 -- Average speed in km/h
  
  -- Total distance
  let d_total : ℝ := v1 * t + v2 * t2
  
  -- Total time
  let t_total : ℝ := t + t2
  
  -- Average speed equation
  v_avg = d_total / t_total

theorem car_problem_solution :
  ∃ t : ℝ, car_problem t ∧ t = 1/3 := by sorry

end NUMINAMATH_CALUDE_car_problem_solution_l2704_270498


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2704_270483

theorem infinite_series_sum : 
  let r : ℝ := (1 : ℝ) / 1000
  let series_sum := ∑' n, (n : ℝ)^2 * r^(n - 1)
  series_sum = (r + 1) / ((1 - r)^3) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2704_270483


namespace NUMINAMATH_CALUDE_vietnam_2007_solution_l2704_270420

open Real

/-- The functional equation from the 2007 Vietnam Mathematical Olympiad -/
def functional_equation (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x * 3^(b + f y - 1) + b^x * 3^(b^3 + f y - 1) - b^(x + y)

/-- The theorem statement for the 2007 Vietnam Mathematical Olympiad problem -/
theorem vietnam_2007_solution (b : ℝ) (hb : b > 0) :
  ∀ f : ℝ → ℝ, functional_equation f b ↔ (∀ x, f x = -b^x) ∨ (∀ x, f x = 1 - b^x) :=
sorry

end NUMINAMATH_CALUDE_vietnam_2007_solution_l2704_270420


namespace NUMINAMATH_CALUDE_always_satisfies_condition_l2704_270488

-- Define the set of colors
inductive Color
| Red
| Blue
| Green
| Yellow

-- Define a point with a color
structure ColoredPoint where
  color : Color

-- Define a colored line segment
structure ColoredSegment where
  endpoint1 : ColoredPoint
  endpoint2 : ColoredPoint
  color : Color

-- Define the coloring property for segments
def validSegmentColoring (segment : ColoredSegment) : Prop :=
  segment.color = segment.endpoint1.color ∨ segment.color = segment.endpoint2.color

-- Define the configuration of points and segments
structure Configuration where
  points : Fin 4 → ColoredPoint
  segments : Fin 6 → ColoredSegment
  allColorsUsed : ∀ c : Color, ∃ s : Fin 6, (segments s).color = c
  distinctPointColors : ∀ i j : Fin 4, i ≠ j → (points i).color ≠ (points j).color
  validSegments : ∀ s : Fin 6, validSegmentColoring (segments s)

-- Define the conditions to be satisfied
def satisfiesConditionA (config : Configuration) (p : Fin 4) : Prop :=
  ∃ s1 s2 s3 : Fin 6,
    (config.segments s1).color = Color.Red ∧
    (config.segments s2).color = Color.Blue ∧
    (config.segments s3).color = Color.Green ∧
    ((config.segments s1).endpoint1 = config.points p ∨ (config.segments s1).endpoint2 = config.points p) ∧
    ((config.segments s2).endpoint1 = config.points p ∨ (config.segments s2).endpoint2 = config.points p) ∧
    ((config.segments s3).endpoint1 = config.points p ∨ (config.segments s3).endpoint2 = config.points p)

def satisfiesConditionB (config : Configuration) (p : Fin 4) : Prop :=
  ∃ s1 s2 s3 : Fin 6,
    (config.segments s1).color = Color.Red ∧
    (config.segments s2).color = Color.Blue ∧
    (config.segments s3).color = Color.Green ∧
    (config.segments s1).endpoint1 ≠ config.points p ∧
    (config.segments s1).endpoint2 ≠ config.points p ∧
    (config.segments s2).endpoint1 ≠ config.points p ∧
    (config.segments s2).endpoint2 ≠ config.points p ∧
    (config.segments s3).endpoint1 ≠ config.points p ∧
    (config.segments s3).endpoint2 ≠ config.points p

-- The main theorem
theorem always_satisfies_condition (config : Configuration) :
  ∃ p : Fin 4, satisfiesConditionA config p ∨ satisfiesConditionB config p := by
  sorry

end NUMINAMATH_CALUDE_always_satisfies_condition_l2704_270488


namespace NUMINAMATH_CALUDE_triangle_side_length_l2704_270450

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  -- Sides form an arithmetic sequence
  2 * b = a + c →
  -- Angle B is 30°
  B = π / 6 →
  -- Area of triangle is 3/2
  (1 / 2) * a * c * Real.sin B = 3 / 2 →
  -- Side b has length √3 + 1
  b = Real.sqrt 3 + 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2704_270450


namespace NUMINAMATH_CALUDE_water_consumption_days_l2704_270460

/-- Represents the daily water consumption of each sibling -/
structure SiblingWaterConsumption where
  theo : ℕ
  mason : ℕ
  roxy : ℕ

/-- Calculates the number of days it takes for the siblings to drink a given amount of water -/
def calculateDays (consumption : SiblingWaterConsumption) (totalWater : ℕ) : ℕ :=
  totalWater / (consumption.theo + consumption.mason + consumption.roxy)

/-- Theorem stating that it takes 7 days for the siblings to drink 168 cups of water -/
theorem water_consumption_days :
  let consumption : SiblingWaterConsumption := ⟨8, 7, 9⟩
  calculateDays consumption 168 = 7 := by
  sorry

#eval calculateDays ⟨8, 7, 9⟩ 168

end NUMINAMATH_CALUDE_water_consumption_days_l2704_270460


namespace NUMINAMATH_CALUDE_ant_travel_distance_l2704_270426

/-- The number of nodes on the bamboo -/
def num_nodes : ℕ := 30

/-- The height of the first node in feet -/
def first_node_height : ℝ := 0.5

/-- The increase in height between consecutive nodes in feet -/
def node_height_diff : ℝ := 0.03

/-- The circumference of the first circle in feet -/
def first_circle_circumference : ℝ := 1.3

/-- The decrease in circumference between consecutive circles in feet -/
def circle_circumference_diff : ℝ := 0.013

/-- The total distance traveled by the ant in feet -/
def total_distance : ℝ := 61.395

/-- Theorem stating the total distance traveled by the ant -/
theorem ant_travel_distance :
  (num_nodes : ℝ) * first_node_height + 
  (num_nodes * (num_nodes - 1) / 2) * node_height_diff +
  (num_nodes : ℝ) * first_circle_circumference - 
  (num_nodes * (num_nodes - 1) / 2) * circle_circumference_diff = 
  total_distance :=
sorry

end NUMINAMATH_CALUDE_ant_travel_distance_l2704_270426


namespace NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l2704_270447

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (P : ConvexPolyhedron) : ℕ :=
  (P.vertices.choose 2) - P.edges - (2 * P.quadrilateral_faces)

/-- Theorem stating the number of space diagonals in the specific polyhedron -/
theorem specific_polyhedron_space_diagonals :
  ∃ (P : ConvexPolyhedron),
    P.vertices = 26 ∧
    P.edges = 60 ∧
    P.faces = 36 ∧
    P.triangular_faces = 24 ∧
    P.quadrilateral_faces = 12 ∧
    space_diagonals P = 241 := by
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l2704_270447


namespace NUMINAMATH_CALUDE_triangle_obtuse_l2704_270441

theorem triangle_obtuse (a b c : ℝ) (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    A + B + C = π ∧
    c^2 = a^2 + b^2 - 2 * a * b * Real.cos C ∧
    Real.cos C < 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_obtuse_l2704_270441


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2704_270454

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  x^6 + 2*x^5 - 3*x^4 + x^3 - 2*x^2 + 5*x - 1 =
  (x - 1) * (x + 2) * (x - 3) * q + (17*x^2 - 52*x + 38) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2704_270454


namespace NUMINAMATH_CALUDE_car_not_speeding_l2704_270457

/-- Braking distance function -/
def braking_distance (x : ℝ) : ℝ := 0.01 * x + 0.002 * x^2

/-- Speed limit in km/h -/
def speed_limit : ℝ := 120

/-- Measured braking distance in meters -/
def measured_distance : ℝ := 26.5

/-- Theorem: There exists a speed less than the speed limit that results in the measured braking distance -/
theorem car_not_speeding : ∃ x : ℝ, x < speed_limit ∧ braking_distance x = measured_distance := by
  sorry


end NUMINAMATH_CALUDE_car_not_speeding_l2704_270457


namespace NUMINAMATH_CALUDE_point_on_line_point_twelve_seven_on_line_l2704_270474

/-- Given three points in the plane, this theorem states that if the first two points
    determine a line, then the third point lies on that line. -/
theorem point_on_line (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁) →
  ∃ (m b : ℝ), y₁ = m * x₁ + b ∧ y₂ = m * x₂ + b ∧ y₃ = m * x₃ + b :=
by sorry

/-- The point (12,7) lies on the line passing through (0,1) and (-6,-2). -/
theorem point_twelve_seven_on_line : 
  ∃ (m b : ℝ), 1 = m * 0 + b ∧ -2 = m * (-6) + b ∧ 7 = m * 12 + b :=
by
  apply point_on_line 0 1 (-6) (-2) 12 7
  -- Proof that the points are collinear
  sorry

end NUMINAMATH_CALUDE_point_on_line_point_twelve_seven_on_line_l2704_270474


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2704_270446

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 1 / x + 1 / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2704_270446


namespace NUMINAMATH_CALUDE_solution_set_f_min_m2_n2_l2704_270428

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Theorem for the solution set of f(x) > 7
theorem solution_set_f : 
  {x : ℝ | f x > 7} = {x : ℝ | x > 4 ∨ x < -3} := by sorry

-- Theorem for the minimum value of m^2 + n^2
theorem min_m2_n2 (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_min : ∀ x, f x ≥ m + n) (h_eq : m + n = 3) :
  m^2 + n^2 ≥ 9/2 ∧ (m^2 + n^2 = 9/2 ↔ m = 3/2 ∧ n = 3/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_min_m2_n2_l2704_270428


namespace NUMINAMATH_CALUDE_ascending_order_abc_l2704_270478

theorem ascending_order_abc (a b c : ℝ) : 
  a = (2 * Real.tan (70 * π / 180)) / (1 + Real.tan (70 * π / 180)^2) →
  b = Real.sqrt ((1 + Real.cos (109 * π / 180)) / 2) →
  c = (Real.sqrt 3 / 2) * Real.cos (81 * π / 180) + (1 / 2) * Real.sin (99 * π / 180) →
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l2704_270478


namespace NUMINAMATH_CALUDE_similar_polygon_area_sum_l2704_270444

/-- Given two similar polygons, constructs a third similar polygon with area equal to the sum of the given polygons' areas -/
theorem similar_polygon_area_sum 
  (t₁ t₂ : ℝ) 
  (a₁ a₂ : ℝ) 
  (h_positive : t₁ > 0 ∧ t₂ > 0 ∧ a₁ > 0 ∧ a₂ > 0)
  (h_similar : t₁ / (a₁^2) = t₂ / (a₂^2)) :
  let b := Real.sqrt (a₁^2 + a₂^2)
  let t₃ := t₁ + t₂
  t₃ / b^2 = t₁ / a₁^2 := by sorry

end NUMINAMATH_CALUDE_similar_polygon_area_sum_l2704_270444


namespace NUMINAMATH_CALUDE_probability_not_face_card_l2704_270442

theorem probability_not_face_card (total_cards : ℕ) (face_cards : ℕ) :
  total_cards = 52 →
  face_cards = 12 →
  (total_cards - face_cards : ℚ) / total_cards = 10 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_not_face_card_l2704_270442


namespace NUMINAMATH_CALUDE_third_term_is_five_l2704_270412

/-- An arithmetic sequence where the sum of the first and fifth terms is 10 -/
def ArithmeticSequence (a : ℝ) (d : ℝ) : Prop :=
  a + (a + 4 * d) = 10

/-- The third term of the arithmetic sequence -/
def ThirdTerm (a : ℝ) (d : ℝ) : ℝ :=
  a + 2 * d

theorem third_term_is_five {a d : ℝ} (h : ArithmeticSequence a d) :
  ThirdTerm a d = 5 := by
  sorry


end NUMINAMATH_CALUDE_third_term_is_five_l2704_270412


namespace NUMINAMATH_CALUDE_initial_people_at_table_l2704_270489

theorem initial_people_at_table (initial : ℕ) 
  (h1 : initial ≥ 6)
  (h2 : initial - 6 + 5 = 10) : initial = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_people_at_table_l2704_270489


namespace NUMINAMATH_CALUDE_standard_deviation_of_commute_times_l2704_270400

def commute_times : List ℝ := [12, 8, 10, 11, 9]

theorem standard_deviation_of_commute_times :
  let n : ℕ := commute_times.length
  let mean : ℝ := (commute_times.sum) / n
  let variance : ℝ := (commute_times.map (λ x => (x - mean)^2)).sum / n
  Real.sqrt variance = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_of_commute_times_l2704_270400


namespace NUMINAMATH_CALUDE_jenny_jellybeans_l2704_270416

/-- The fraction of jellybeans remaining after eating 25% -/
def remainingFraction : ℝ := 0.75

/-- The number of days that passed -/
def days : ℕ := 3

/-- The number of jellybeans remaining after 3 days -/
def remainingJellybeans : ℕ := 27

/-- The original number of jellybeans in Jenny's jar -/
def originalJellybeans : ℕ := 64

theorem jenny_jellybeans :
  (remainingFraction ^ days) * (originalJellybeans : ℝ) = remainingJellybeans := by
  sorry

end NUMINAMATH_CALUDE_jenny_jellybeans_l2704_270416


namespace NUMINAMATH_CALUDE_unique_solution_l2704_270467

theorem unique_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : 2*x - 2*y + 1/z = 1/2014)
  (eq2 : 2*y - 2*z + 1/x = 1/2014)
  (eq3 : 2*z - 2*x + 1/y = 1/2014) :
  x = 2014 ∧ y = 2014 ∧ z = 2014 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2704_270467


namespace NUMINAMATH_CALUDE_count_theorem_l2704_270406

/-- The count of positive integers less than 3000 with at most three different digits -/
def count_numbers_with_at_most_three_digits : ℕ :=
  -- Definition placeholder
  0

/-- The upper bound of the considered range -/
def upper_bound : ℕ := 3000

/-- Predicate to check if a number has at most three different digits -/
def has_at_most_three_digits (n : ℕ) : Prop :=
  -- Definition placeholder
  True

theorem count_theorem :
  count_numbers_with_at_most_three_digits = 891 :=
sorry


end NUMINAMATH_CALUDE_count_theorem_l2704_270406


namespace NUMINAMATH_CALUDE_prime_between_40_and_50_and_largest_below_100_l2704_270462

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_between_40_and_50_and_largest_below_100 :
  (∀ p : ℕ, 40 < p ∧ p < 50 ∧ isPrime p ↔ p = 41 ∨ p = 43 ∨ p = 47) ∧
  (∀ q : ℕ, q < 100 ∧ isPrime q → q ≤ 97) ∧
  isPrime 97 :=
sorry

end NUMINAMATH_CALUDE_prime_between_40_and_50_and_largest_below_100_l2704_270462


namespace NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l2704_270458

theorem least_n_with_gcd_conditions : 
  ∃ n : ℕ, n > 500 ∧ 
    Nat.gcd 42 (n + 80) = 14 ∧ 
    Nat.gcd (n + 42) 80 = 40 ∧
    (∀ m : ℕ, m > 500 → 
      Nat.gcd 42 (m + 80) = 14 → 
      Nat.gcd (m + 42) 80 = 40 → 
      n ≤ m) ∧
    n = 638 :=
by sorry

end NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l2704_270458


namespace NUMINAMATH_CALUDE_fraction_problem_l2704_270425

theorem fraction_problem (N : ℝ) (x y : ℤ) :
  N = 30 →
  0.5 * N = (x / y : ℝ) * N + 10 →
  (x / y : ℝ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2704_270425


namespace NUMINAMATH_CALUDE_square_traffic_sign_perimeter_l2704_270415

/-- A square traffic sign with sides of 4 feet has a perimeter of 16 feet. -/
theorem square_traffic_sign_perimeter : 
  ∀ (side_length : ℝ), side_length = 4 → 4 * side_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_traffic_sign_perimeter_l2704_270415


namespace NUMINAMATH_CALUDE_regular_triangle_counts_l2704_270419

/-- Regular triangle with sides divided into n segments -/
structure RegularTriangle (n : ℕ) where
  -- Add any necessary fields

/-- Count of regular triangles in a RegularTriangle -/
def countRegularTriangles (t : RegularTriangle n) : ℕ :=
  if n % 2 = 0 then
    (n * (n + 2) * (2 * n + 1)) / 8
  else
    ((n + 1) * (2 * n^2 + 3 * n - 1)) / 8

/-- Count of rhombuses in a RegularTriangle -/
def countRhombuses (t : RegularTriangle n) : ℕ :=
  if n % 2 = 0 then
    (n * (n + 2) * (2 * n - 1)) / 8
  else
    ((n - 1) * (n + 1) * (2 * n + 3)) / 8

/-- Theorem stating the counts are correct -/
theorem regular_triangle_counts (n : ℕ) (t : RegularTriangle n) :
  (countRegularTriangles t = if n % 2 = 0 then (n * (n + 2) * (2 * n + 1)) / 8
                             else ((n + 1) * (2 * n^2 + 3 * n - 1)) / 8) ∧
  (countRhombuses t = if n % 2 = 0 then (n * (n + 2) * (2 * n - 1)) / 8
                      else ((n - 1) * (n + 1) * (2 * n + 3)) / 8) :=
by sorry

end NUMINAMATH_CALUDE_regular_triangle_counts_l2704_270419


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2704_270404

theorem shaded_fraction_of_rectangle (length width : ℝ) (shaded_quarter : ℝ) :
  length = 15 →
  width = 20 →
  shaded_quarter = (1 / 4) * (length * width) →
  shaded_quarter = (1 / 5) * (length * width) →
  shaded_quarter / (length * width) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2704_270404


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l2704_270493

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) - a n = 3

theorem arithmetic_sequence_50th_term 
  (a : ℕ → ℕ) (n : ℕ) (h : arithmetic_sequence a) (h_50 : a n = 50) : n = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l2704_270493


namespace NUMINAMATH_CALUDE_unique_prime_triple_l2704_270424

theorem unique_prime_triple : 
  ∀ p q r : ℕ+, 
    Prime p.val → Prime q.val → 
    (r.val^2 - 5*q.val^2) / (p.val^2 - 1) = 2 → 
    (p, q, r) = (3, 2, 6) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l2704_270424


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l2704_270453

theorem power_of_power_at_three :
  (3^3)^(3^3) = 27^27 := by sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l2704_270453


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l2704_270477

theorem scientific_notation_equivalence : 
  ∃ (a : ℝ) (n : ℤ), 0.0000907 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 9.07 ∧ n = -5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l2704_270477


namespace NUMINAMATH_CALUDE_investment_rate_proof_l2704_270402

def total_investment : ℝ := 12000
def first_investment : ℝ := 5000
def second_investment : ℝ := 4000
def first_rate : ℝ := 0.03
def second_rate : ℝ := 0.045
def desired_income : ℝ := 600

theorem investment_rate_proof :
  let remaining_investment := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let remaining_income := desired_income - income_from_first - income_from_second
  remaining_income / remaining_investment = 0.09 := by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l2704_270402


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2704_270439

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) → -3 < k ∧ k < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2704_270439


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2704_270496

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + k * y + 12 = 0 → y = x) → 
  k = 12 ∨ k = -12 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2704_270496


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2704_270427

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = (1 + Complex.I) / 2) :
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2704_270427


namespace NUMINAMATH_CALUDE_jacob_age_theorem_l2704_270494

/-- Proves that Jacob's current age is the age at which he is twice as old as his brother -/
theorem jacob_age_theorem (jacob_age : ℕ) (brother_age : ℕ) 
  (h1 : jacob_age = 18) 
  (h2 : jacob_age = 2 * brother_age) : 
  jacob_age = 2 * brother_age := by
  sorry

#check jacob_age_theorem

end NUMINAMATH_CALUDE_jacob_age_theorem_l2704_270494


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2704_270481

open Function Real

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x - f y) = 1 - x - y) →
  (∀ y : ℝ, f y = 1/2 - y) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2704_270481


namespace NUMINAMATH_CALUDE_sin_4theta_from_exp_itheta_l2704_270485

theorem sin_4theta_from_exp_itheta (θ : ℝ) :
  Complex.exp (Complex.I * θ) = (4 + Complex.I * Real.sqrt 7) / 5 →
  Real.sin (4 * θ) = 144 * Real.sqrt 7 / 625 := by
  sorry

end NUMINAMATH_CALUDE_sin_4theta_from_exp_itheta_l2704_270485


namespace NUMINAMATH_CALUDE_expr_is_symmetrical_l2704_270431

/-- Definition of a symmetrical expression -/
def is_symmetrical (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b, f a b = f b a

/-- The expression we want to prove is symmetrical -/
def expr (a b : ℝ) : ℝ := 4*a^2 + 4*b^2 - 4*a*b

/-- Theorem: The expression 4a^2 + 4b^2 - 4ab is symmetrical -/
theorem expr_is_symmetrical : is_symmetrical expr := by sorry

end NUMINAMATH_CALUDE_expr_is_symmetrical_l2704_270431


namespace NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l2704_270463

theorem largest_number_from_hcf_lcm_factors (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 23)
  (lcm_eq : Nat.lcm a b = 23 * 13 * 14) :
  max a b = 322 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l2704_270463


namespace NUMINAMATH_CALUDE_function_transformation_l2704_270486

-- Define the given function
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem function_transformation :
  (∀ x, f (2 * x + 1) = x^2 - 2*x) →
  (∀ x, f x = x^2 / 4 - (3/2) * x + 5/4) := by sorry

end NUMINAMATH_CALUDE_function_transformation_l2704_270486


namespace NUMINAMATH_CALUDE_perpendicular_bisector_covered_l2704_270465

def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

def perpendicular_bisector (O P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | (q.1 - (O.1 + P.1)/2)^2 + (q.2 - (O.2 + P.2)/2)^2 < ((O.1 - P.1)^2 + (O.2 - P.2)^2) / 4}

def plane_region (m : ℝ) : Set (ℝ × ℝ) := {p | |p.1| + |p.2| ≥ m}

theorem perpendicular_bisector_covered (m : ℝ) :
  (∀ P ∈ circle_O, perpendicular_bisector (0, 0) P ⊆ plane_region m) →
  m ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_covered_l2704_270465


namespace NUMINAMATH_CALUDE_bowling_team_weight_problem_l2704_270417

theorem bowling_team_weight_problem (original_players : ℕ) (original_avg_weight : ℝ)
  (new_player1_weight : ℝ) (new_avg_weight : ℝ) :
  original_players = 7 →
  original_avg_weight = 121 →
  new_player1_weight = 110 →
  new_avg_weight = 113 →
  ∃ new_player2_weight : ℝ,
    new_player2_weight = 60 ∧
    (original_players : ℝ) * original_avg_weight + new_player1_weight + new_player2_weight =
      ((original_players : ℝ) + 2) * new_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_weight_problem_l2704_270417


namespace NUMINAMATH_CALUDE_linear_systems_solution_and_expression_l2704_270468

theorem linear_systems_solution_and_expression (a b : ℝ) : 
  (∃ x y : ℝ, (3 * x - 5 * y = 36 ∧ b * x + a * y = -8) ∧
              (2 * x + 5 * y = -26 ∧ a * x - b * y = -4)) →
  (∃ x y : ℝ, x = 2 ∧ y = -6 ∧
              (3 * x - 5 * y = 36 ∧ b * x + a * y = -8) ∧
              (2 * x + 5 * y = -26 ∧ a * x - b * y = -4)) ∧
  (2 * a + b)^2023 = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_systems_solution_and_expression_l2704_270468


namespace NUMINAMATH_CALUDE_fraction_simplification_l2704_270475

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) :
  (x - 3) / (2 * x * (x - 3)) = 1 / (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2704_270475


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2704_270438

/-- The distance from a real number to the nearest integer -/
noncomputable def distToNearestInt (x : ℝ) : ℝ := min (x - ⌊x⌋) (⌈x⌉ - x)

/-- The sum of squares of x * (distance of x to nearest integer) -/
noncomputable def sumOfSquares (xs : Finset ℝ) : ℝ :=
  Finset.sum xs (λ x => (x * distToNearestInt x)^2)

/-- The maximum value of the sum of squares given the constraints -/
theorem max_sum_of_squares (n : ℕ) :
  ∃ (xs : Finset ℝ),
    (∀ x ∈ xs, 0 ≤ x) ∧
    (Finset.sum xs id = n) ∧
    (Finset.card xs = n) ∧
    (∀ ys : Finset ℝ,
      (∀ y ∈ ys, 0 ≤ y) →
      (Finset.sum ys id = n) →
      (Finset.card ys = n) →
      sumOfSquares ys ≤ sumOfSquares xs) ∧
    (sumOfSquares xs = (n^2 - n + 1/2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2704_270438


namespace NUMINAMATH_CALUDE_work_completion_time_l2704_270409

/-- A can do a piece of work in some days. A does the work for 5 days only and leaves the job. 
    B does the remaining work in 3 days. B alone can do the work in 4.5 days. 
    This theorem proves that A alone can do the work in 15 days. -/
theorem work_completion_time (W : ℝ) (A_work_per_day B_work_per_day : ℝ) : 
  (B_work_per_day = W / 4.5) →
  (5 * A_work_per_day + 3 * B_work_per_day = W) →
  (A_work_per_day = W / 15) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2704_270409


namespace NUMINAMATH_CALUDE_algebraic_sum_equals_one_l2704_270401

theorem algebraic_sum_equals_one (a b c x : ℝ) 
  (ha : a + x^2 = 2006)
  (hb : b + x^2 = 2007)
  (hc : c + x^2 = 2008)
  (habc : a * b * c = 3) :
  a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_sum_equals_one_l2704_270401


namespace NUMINAMATH_CALUDE_product_equals_693_over_256_l2704_270471

theorem product_equals_693_over_256 : 
  (1 + 1/2) * (1 + 1/4) * (1 + 1/6) * (1 + 1/8) * (1 + 1/10) = 693/256 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_693_over_256_l2704_270471


namespace NUMINAMATH_CALUDE_equation_solution_l2704_270414

theorem equation_solution :
  ∃ x : ℝ, 3.5 * ((3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5)) = 2800.0000000000005 ∧ x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2704_270414


namespace NUMINAMATH_CALUDE_tangency_condition_l2704_270455

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse x y ∧ hyperbola x y m ∧
  ∀ x' y' : ℝ, ellipse x' y' ∧ hyperbola x' y' m → (x', y') = (x, y)

-- State the theorem
theorem tangency_condition :
  ∀ m : ℝ, are_tangent m ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_tangency_condition_l2704_270455
