import Mathlib

namespace NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l2604_260497

theorem hundred_power_ten_as_sum_of_tens (n : ℕ) : (100 ^ 10) = n * 10 → n = 10 ^ 19 := by
  sorry

end NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l2604_260497


namespace NUMINAMATH_CALUDE_total_population_is_56000_l2604_260496

/-- The total population of Boise, Seattle, and Lake View --/
def total_population (boise seattle lakeview : ℕ) : ℕ :=
  boise + seattle + lakeview

/-- Theorem: Given the conditions, the total population of the three cities is 56000 --/
theorem total_population_is_56000 :
  ∀ (boise seattle lakeview : ℕ),
    boise = (3 * seattle) / 5 →
    lakeview = seattle + 4000 →
    lakeview = 24000 →
    total_population boise seattle lakeview = 56000 := by
  sorry

end NUMINAMATH_CALUDE_total_population_is_56000_l2604_260496


namespace NUMINAMATH_CALUDE_max_elements_sum_to_target_l2604_260478

/-- The sequence of consecutive odd numbers from 1 to 101 -/
def oddSequence : List Nat := List.range 51 |>.map (fun n => 2 * n + 1)

/-- The sum of selected numbers should be 2013 -/
def targetSum : Nat := 2013

/-- The maximum number of elements that can be selected -/
def maxElements : Nat := 43

theorem max_elements_sum_to_target :
  ∃ (selected : List Nat),
    selected.length = maxElements ∧
    selected.all (· ∈ oddSequence) ∧
    selected.sum = targetSum ∧
    ∀ (other : List Nat),
      other.all (· ∈ oddSequence) →
      other.sum = targetSum →
      other.length ≤ maxElements :=
sorry

end NUMINAMATH_CALUDE_max_elements_sum_to_target_l2604_260478


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2604_260422

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2604_260422


namespace NUMINAMATH_CALUDE_computer_operations_l2604_260415

/-- Represents the computer's specifications and performance --/
structure ComputerSpec where
  mult_rate : ℕ  -- multiplications per second
  add_rate : ℕ   -- additions per second
  switch_time : ℕ  -- time in seconds when switching from multiplications to additions
  total_time : ℕ  -- total operation time in seconds

/-- Calculates the total number of operations performed by the computer --/
def total_operations (spec : ComputerSpec) : ℕ :=
  let mult_ops := spec.mult_rate * spec.switch_time
  let add_ops := spec.add_rate * (spec.total_time - spec.switch_time)
  mult_ops + add_ops

/-- Theorem stating that the computer performs 63,000,000 operations in 2 hours --/
theorem computer_operations :
  let spec : ComputerSpec := {
    mult_rate := 5000,
    add_rate := 10000,
    switch_time := 1800,
    total_time := 7200
  }
  total_operations spec = 63000000 := by
  sorry


end NUMINAMATH_CALUDE_computer_operations_l2604_260415


namespace NUMINAMATH_CALUDE_total_players_l2604_260426

theorem total_players (cricket : ℕ) (hockey : ℕ) (football : ℕ) (softball : ℕ)
  (h1 : cricket = 16)
  (h2 : hockey = 12)
  (h3 : football = 18)
  (h4 : softball = 13) :
  cricket + hockey + football + softball = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l2604_260426


namespace NUMINAMATH_CALUDE_equation_solution_l2604_260439

theorem equation_solution : ∃ x : ℝ, 7 * x - 5 = 6 * x ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2604_260439


namespace NUMINAMATH_CALUDE_triangle_area_l2604_260447

/-- Given a triangle with perimeter 40 and inradius 2.5, prove its area is 50 -/
theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) : 
  P = 40 → r = 2.5 → A = r * (P / 2) → A = 50 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2604_260447


namespace NUMINAMATH_CALUDE_bruce_fruit_purchase_cost_l2604_260418

/-- Calculates the total cost of Bruce's fruit purchase in US dollars -/
def fruit_purchase_cost (grapes_kg : ℝ) (grapes_price : ℝ) (mangoes_kg : ℝ) (mangoes_price : ℝ)
  (oranges_kg : ℝ) (oranges_price : ℝ) (apples_kg : ℝ) (apples_price : ℝ)
  (grapes_discount : ℝ) (mangoes_tax : ℝ) (oranges_premium : ℝ)
  (euro_to_usd : ℝ) (pound_to_usd : ℝ) (yen_to_usd : ℝ) : ℝ :=
  let grapes_cost := grapes_kg * grapes_price * (1 - grapes_discount)
  let mangoes_cost := mangoes_kg * mangoes_price * euro_to_usd * (1 + mangoes_tax)
  let oranges_cost := oranges_kg * oranges_price * pound_to_usd * (1 + oranges_premium)
  let apples_cost := apples_kg * apples_price * yen_to_usd
  grapes_cost + mangoes_cost + oranges_cost + apples_cost

/-- Theorem stating that Bruce's fruit purchase cost is $1563.10 -/
theorem bruce_fruit_purchase_cost :
  fruit_purchase_cost 8 70 8 55 5 40 10 3000 0.1 0.05 0.03 1.15 1.25 0.009 = 1563.10 := by
  sorry

end NUMINAMATH_CALUDE_bruce_fruit_purchase_cost_l2604_260418


namespace NUMINAMATH_CALUDE_sprinkles_problem_l2604_260432

theorem sprinkles_problem (initial_cans remaining_cans subtracted_number : ℕ) : 
  initial_cans = 12 →
  remaining_cans = 3 →
  remaining_cans = initial_cans / 2 - subtracted_number →
  subtracted_number = 3 := by
  sorry

end NUMINAMATH_CALUDE_sprinkles_problem_l2604_260432


namespace NUMINAMATH_CALUDE_weight_of_smaller_cube_l2604_260401

/-- Given two cubes of the same material, where the second cube has sides twice
    as long as the first and weighs 40 pounds, prove that the weight of the first
    cube is 5 pounds. -/
theorem weight_of_smaller_cube (s : ℝ) (w : ℝ → ℝ → ℝ) :
  (∀ x y, w x y = (y / x^3) * w 1 1) →  -- weight is proportional to volume
  w (2*s) (8*s^3) = 40 →                -- weight of larger cube
  w s (s^3) = 5 := by
sorry


end NUMINAMATH_CALUDE_weight_of_smaller_cube_l2604_260401


namespace NUMINAMATH_CALUDE_min_value_theorem_l2604_260419

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 3 * y = 8) :
  (2 / x + 3 / y) ≥ 25 / 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + 3 * y = 8 ∧ 2 / x + 3 / y = 25 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2604_260419


namespace NUMINAMATH_CALUDE_hypothetical_town_population_l2604_260482

theorem hypothetical_town_population : ∃ n : ℕ, 
  (∃ m k : ℕ, 
    n^2 + 150 = m^2 + 1 ∧ 
    n^2 + 300 = k^2) ∧ 
  n^2 = 5476 := by
  sorry

end NUMINAMATH_CALUDE_hypothetical_town_population_l2604_260482


namespace NUMINAMATH_CALUDE_f_extremum_and_monotonicity_l2604_260473

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

theorem f_extremum_and_monotonicity :
  (∀ x > 0, f (-1/2) x ≤ f (-1/2) 1) ∧ f (-1/2) 1 = 0 ∧
  (∀ a : ℝ, (∀ x > 0, ∀ y > 0, x < y → f a x < f a y) ↔ a ≥ 1 / (2 * Real.exp 2)) :=
sorry

end NUMINAMATH_CALUDE_f_extremum_and_monotonicity_l2604_260473


namespace NUMINAMATH_CALUDE_factory_output_percentage_l2604_260441

theorem factory_output_percentage (may_output june_output : ℝ) : 
  may_output = june_output * (1 - 0.2) → 
  (june_output - may_output) / may_output = 0.25 := by
sorry

end NUMINAMATH_CALUDE_factory_output_percentage_l2604_260441


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2604_260498

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  B = π / 3 ∧ a = Real.sqrt 3 ∧ c = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2604_260498


namespace NUMINAMATH_CALUDE_girls_in_class_l2604_260481

/-- In a class with the following properties:
    - There are 18 boys over 160 cm tall
    - These 18 boys constitute 3/4 of all boys
    - The total number of boys is 2/3 of the total number of students
    Then the number of girls in the class is 12 -/
theorem girls_in_class (tall_boys : ℕ) (total_boys : ℕ) (total_students : ℕ) 
  (h1 : tall_boys = 18)
  (h2 : tall_boys = (3 / 4 : ℚ) * total_boys)
  (h3 : total_boys = (2 / 3 : ℚ) * total_students) :
  total_students - total_boys = 12 := by
  sorry

#check girls_in_class

end NUMINAMATH_CALUDE_girls_in_class_l2604_260481


namespace NUMINAMATH_CALUDE_construction_materials_cost_l2604_260440

/-- Calculates the total amount paid for construction materials --/
def total_amount_paid (cement_bags : ℕ) (cement_price : ℚ) (cement_discount : ℚ)
                      (sand_lorries : ℕ) (sand_tons_per_lorry : ℕ) (sand_price_per_ton : ℚ)
                      (sand_tax : ℚ) : ℚ :=
  let cement_cost := cement_bags * cement_price
  let cement_discount_amount := cement_cost * cement_discount
  let cement_total := cement_cost - cement_discount_amount
  let sand_tons := sand_lorries * sand_tons_per_lorry
  let sand_cost := sand_tons * sand_price_per_ton
  let sand_tax_amount := sand_cost * sand_tax
  let sand_total := sand_cost + sand_tax_amount
  cement_total + sand_total

/-- The total amount paid for construction materials is $13,310 --/
theorem construction_materials_cost :
  total_amount_paid 500 10 (5/100) 20 10 40 (7/100) = 13310 := by
  sorry

end NUMINAMATH_CALUDE_construction_materials_cost_l2604_260440


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2604_260468

theorem perfect_square_trinomial (x : ℝ) : 
  (x + 9)^2 = x^2 + 18*x + 81 ∧ 
  ∃ (a b : ℝ), (x + 9)^2 = a^2 + 2*a*b + b^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2604_260468


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l2604_260403

theorem cosine_sine_identity (θ : Real) (h : Real.tan θ = 1/3) :
  Real.cos θ ^ 2 + (1/2) * Real.sin (2 * θ) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l2604_260403


namespace NUMINAMATH_CALUDE_angle_A_range_l2604_260493

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the convexity property
def is_convex (q : Quadrilateral) : Prop := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle measure function
def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_A_range (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_AB : distance q.A q.B = 8)
  (h_BC : distance q.B q.C = 4)
  (h_CD : distance q.C q.D = 6)
  (h_DA : distance q.D q.A = 6) :
  0 < angle_measure q.B q.A q.D ∧ angle_measure q.B q.A q.D < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_range_l2604_260493


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_five_l2604_260462

def expression (n : ℕ) : ℤ :=
  8 * (n - 2)^6 - 3 * n^2 + 20 * n - 36

theorem largest_n_divisible_by_five :
  ∀ n : ℕ, n < 100000 →
    (expression n % 5 = 0 → n ≤ 99997) ∧
    (expression 99997 % 5 = 0) ∧
    99997 < 100000 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_five_l2604_260462


namespace NUMINAMATH_CALUDE_overhead_cost_reduction_l2604_260448

/-- Represents the cost components of manufacturing a car --/
structure CarCost where
  raw_material : ℝ
  labor : ℝ
  overhead : ℝ

/-- Calculates the total cost of manufacturing a car --/
def total_cost (cost : CarCost) : ℝ :=
  cost.raw_material + cost.labor + cost.overhead

theorem overhead_cost_reduction 
  (initial_cost : CarCost) 
  (new_cost : CarCost) 
  (h1 : initial_cost.raw_material = (4/9) * total_cost initial_cost)
  (h2 : initial_cost.labor = (3/9) * total_cost initial_cost)
  (h3 : initial_cost.overhead = (2/9) * total_cost initial_cost)
  (h4 : new_cost.raw_material = 1.1 * initial_cost.raw_material)
  (h5 : new_cost.labor = 1.08 * initial_cost.labor)
  (h6 : total_cost new_cost = 1.06 * total_cost initial_cost) :
  new_cost.overhead = 0.95 * initial_cost.overhead :=
sorry

end NUMINAMATH_CALUDE_overhead_cost_reduction_l2604_260448


namespace NUMINAMATH_CALUDE_negative_integer_product_l2604_260457

theorem negative_integer_product (a b : ℤ) : ∃ n : ℤ,
  n < 0 ∧
  n * a < 0 ∧
  -8 * b < 0 ∧
  n * a * (-8 * b) + a * b = 89 ∧
  n = -11 := by sorry

end NUMINAMATH_CALUDE_negative_integer_product_l2604_260457


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2604_260476

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) →
  ((a + 2) * (a - 2) * a = a^3 - 12) →
  (a^3 = 27) := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2604_260476


namespace NUMINAMATH_CALUDE_integral_of_4x_plus_7_cos_3x_l2604_260437

theorem integral_of_4x_plus_7_cos_3x (x : ℝ) :
  deriv (fun x => (1/3) * (4*x + 7) * Real.sin (3*x) + (4/9) * Real.cos (3*x)) x
  = (4*x + 7) * Real.cos (3*x) := by
sorry

end NUMINAMATH_CALUDE_integral_of_4x_plus_7_cos_3x_l2604_260437


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_implies_a_eq_one_l2604_260494

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is nonzero. -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number constructed from the real number a. -/
def complexNumber (a : ℝ) : ℂ :=
  ⟨a^2 - 3*a + 2, a - 2⟩

/-- If the complex number ((a^2 - 3a + 2) + (a - 2)i) is purely imaginary, then a = 1. -/
theorem complex_purely_imaginary_implies_a_eq_one (a : ℝ) :
  isPurelyImaginary (complexNumber a) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_complex_purely_imaginary_implies_a_eq_one_l2604_260494


namespace NUMINAMATH_CALUDE_subtraction_problem_l2604_260445

theorem subtraction_problem : 3.609 - 2.5 - 0.193 = 0.916 := by sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2604_260445


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2604_260477

theorem fraction_to_decimal : (58 : ℚ) / 160 = (3625 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2604_260477


namespace NUMINAMATH_CALUDE_circle_equation_and_m_range_l2604_260454

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define the points A and B
def point_A : ℝ × ℝ := (3, 3)
def point_B : ℝ × ℝ := (2, 4)

-- Define the line y = 3x - 5
def line_center (x y : ℝ) : Prop := y = 3*x - 5

-- Define the circle with diameter PQ
def circle_PQ (x y m : ℝ) : Prop := x^2 + y^2 = m^2

theorem circle_equation_and_m_range :
  ∃ (center_x center_y : ℝ),
    -- The center of C is on the line y = 3x - 5
    line_center center_x center_y ∧
    -- C passes through A and B
    circle_C point_A.1 point_A.2 ∧
    circle_C point_B.1 point_B.2 ∧
    -- For any m > 0, if there exists a point M on both circles
    (∀ m : ℝ, m > 0 →
      (∃ x y : ℝ, circle_C x y ∧ circle_PQ x y m) →
      -- Then m is in the range [4, 6]
      4 ≤ m ∧ m ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_and_m_range_l2604_260454


namespace NUMINAMATH_CALUDE_parallelogram_area_l2604_260469

def vector_a : Fin 2 → ℝ := ![6, -8]
def vector_b : Fin 2 → ℝ := ![15, 4]

theorem parallelogram_area : 
  |vector_a 0 * vector_b 1 - vector_a 1 * vector_b 0| = 144 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2604_260469


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l2604_260458

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

end NUMINAMATH_CALUDE_angle_bisector_theorem_l2604_260458


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2604_260430

theorem complex_number_quadrant (z : ℂ) (h : (2 + 3*I)*z = 1 + I) : 
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2604_260430


namespace NUMINAMATH_CALUDE_train_length_l2604_260488

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (h1 : speed_kmph = 90) (h2 : time_sec = 5) :
  speed_kmph * (1000 / 3600) * time_sec = 125 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2604_260488


namespace NUMINAMATH_CALUDE_ten_pound_bag_cost_l2604_260461

/-- Represents the cost and weight of a bag of grass seed -/
structure Bag where
  weight : ℕ
  cost : ℚ

/-- Represents the purchase constraints and known information -/
structure PurchaseInfo where
  minWeight : ℕ
  maxWeight : ℕ
  leastCost : ℚ
  bag5lb : Bag
  bag25lb : Bag

/-- Calculates the cost of a 10-pound bag given the purchase information -/
def calculate10lbBagCost (info : PurchaseInfo) : ℚ :=
  info.leastCost - 3 * info.bag25lb.cost

/-- Theorem stating that the cost of the 10-pound bag is $1.98 -/
theorem ten_pound_bag_cost (info : PurchaseInfo) 
  (h1 : info.minWeight = 65)
  (h2 : info.maxWeight = 80)
  (h3 : info.leastCost = 98.73)
  (h4 : info.bag5lb = ⟨5, 13.80⟩)
  (h5 : info.bag25lb = ⟨25, 32.25⟩) :
  calculate10lbBagCost info = 1.98 := by sorry

end NUMINAMATH_CALUDE_ten_pound_bag_cost_l2604_260461


namespace NUMINAMATH_CALUDE_total_peaches_l2604_260472

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 19

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 4

/-- The total number of baskets -/
def number_of_baskets : ℕ := 15

/-- Theorem: The total number of peaches in all baskets is 345 -/
theorem total_peaches :
  (red_peaches_per_basket + green_peaches_per_basket) * number_of_baskets = 345 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l2604_260472


namespace NUMINAMATH_CALUDE_special_function_properties_l2604_260431

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y - 1

theorem special_function_properties (f : ℝ → ℝ) 
  (h1 : special_function f) (h2 : f 1 = 4) : 
  f 0 = 1 ∧ ∀ n : ℕ, f n = (n + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l2604_260431


namespace NUMINAMATH_CALUDE_inventory_and_profit_calculation_l2604_260495

/-- Represents the inventory and financial data of a supermarket --/
structure Supermarket where
  total_cost : ℕ
  cost_A : ℕ
  cost_B : ℕ
  price_A : ℕ
  price_B : ℕ

/-- Calculates the number of items A and B, and the total profit --/
def calculate_inventory_and_profit (s : Supermarket) : ℕ × ℕ × ℕ :=
  let items_A := 150
  let items_B := 90
  let profit := items_A * (s.price_A - s.cost_A) + items_B * (s.price_B - s.cost_B)
  (items_A, items_B, profit)

/-- Theorem stating the correct inventory and profit calculation --/
theorem inventory_and_profit_calculation (s : Supermarket) 
  (h1 : s.total_cost = 6000)
  (h2 : s.cost_A = 22)
  (h3 : s.cost_B = 30)
  (h4 : s.price_A = 29)
  (h5 : s.price_B = 40) :
  calculate_inventory_and_profit s = (150, 90, 1950) :=
by
  sorry

#eval calculate_inventory_and_profit ⟨6000, 22, 30, 29, 40⟩

end NUMINAMATH_CALUDE_inventory_and_profit_calculation_l2604_260495


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_36_5_l2604_260465

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

end NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_36_5_l2604_260465


namespace NUMINAMATH_CALUDE_complex_product_modulus_l2604_260433

theorem complex_product_modulus : Complex.abs (4 - 5 * Complex.I) * Complex.abs (4 + 5 * Complex.I) = 41 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l2604_260433


namespace NUMINAMATH_CALUDE_pet_owners_proof_l2604_260475

theorem pet_owners_proof (total_pet_owners : Nat) 
                         (only_dog_owners : Nat)
                         (only_cat_owners : Nat)
                         (cat_dog_snake_owners : Nat)
                         (total_snakes : Nat)
                         (h1 : total_pet_owners = 99)
                         (h2 : only_dog_owners = 15)
                         (h3 : only_cat_owners = 10)
                         (h4 : cat_dog_snake_owners = 3)
                         (h5 : total_snakes = 69) : 
  total_pet_owners = only_dog_owners + only_cat_owners + cat_dog_snake_owners + (total_snakes - cat_dog_snake_owners) + 5 :=
by
  sorry

#check pet_owners_proof

end NUMINAMATH_CALUDE_pet_owners_proof_l2604_260475


namespace NUMINAMATH_CALUDE_tree_planting_correct_l2604_260412

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

end NUMINAMATH_CALUDE_tree_planting_correct_l2604_260412


namespace NUMINAMATH_CALUDE_book_arrangement_ways_l2604_260487

theorem book_arrangement_ways (n m : ℕ) (h : n + m = 9) (hn : n = 4) (hm : m = 5) :
  Nat.choose (n + m) n = 126 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_ways_l2604_260487


namespace NUMINAMATH_CALUDE_ricciana_long_jump_l2604_260456

/-- Ricciana's long jump problem -/
theorem ricciana_long_jump :
  ∀ (ricciana_run margarita_run ricciana_jump margarita_jump : ℕ),
  ricciana_run = 20 →
  margarita_run = 18 →
  margarita_jump = 2 * ricciana_jump - 1 →
  margarita_run + margarita_jump = ricciana_run + ricciana_jump + 1 →
  ricciana_jump = 22 := by
sorry

end NUMINAMATH_CALUDE_ricciana_long_jump_l2604_260456


namespace NUMINAMATH_CALUDE_original_triangle_area_l2604_260464

theorem original_triangle_area (original_side : ℝ) (new_side : ℝ) (new_area : ℝ) :
  new_side = 5 * original_side →
  new_area = 125 →
  (original_side^2 * Real.sqrt 3) / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_triangle_area_l2604_260464


namespace NUMINAMATH_CALUDE_f_properties_l2604_260451

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + Real.pi/2) * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3/4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ 1/4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1/2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = 1/4) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2604_260451


namespace NUMINAMATH_CALUDE_coefficient_a2_l2604_260436

theorem coefficient_a2 (x a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (x - 1)^5 + (x - 1)^3 + (x - 1) = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a →
  a₂ = -13 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a2_l2604_260436


namespace NUMINAMATH_CALUDE_ned_chocolate_pieces_l2604_260499

theorem ned_chocolate_pieces : 
  ∀ (boxes_bought boxes_given pieces_per_box : ℝ),
    boxes_bought = 14.0 →
    boxes_given = 7.0 →
    pieces_per_box = 6.0 →
    (boxes_bought - boxes_given) * pieces_per_box = 42.0 := by
  sorry

end NUMINAMATH_CALUDE_ned_chocolate_pieces_l2604_260499


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l2604_260485

theorem divisors_of_2_pow_48_minus_1 :
  ∃! (a b : ℕ), 60 < a ∧ a < b ∧ b < 70 ∧ (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l2604_260485


namespace NUMINAMATH_CALUDE_unitPrice_is_constant_l2604_260405

/-- Represents the data from a fuel dispenser --/
structure FuelDispenser :=
  (amount : ℝ)
  (unitPrice : ℝ)
  (unitPricePerYuanPerLiter : ℝ)

/-- The fuel dispenser data from the problem --/
def fuelData : FuelDispenser :=
  { amount := 116.64,
    unitPrice := 18,
    unitPricePerYuanPerLiter := 6.48 }

/-- Predicate to check if a value is constant in the fuel dispenser context --/
def isConstant (f : FuelDispenser → ℝ) : Prop :=
  ∀ (d1 d2 : FuelDispenser), d1.unitPrice = d2.unitPrice → f d1 = f d2

/-- Theorem stating that the unit price is constant --/
theorem unitPrice_is_constant :
  isConstant (λ d : FuelDispenser => d.unitPrice) :=
sorry

end NUMINAMATH_CALUDE_unitPrice_is_constant_l2604_260405


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2604_260450

theorem gcd_of_three_numbers : Nat.gcd 7254 (Nat.gcd 10010 22554) = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2604_260450


namespace NUMINAMATH_CALUDE_soccer_ball_max_height_l2604_260414

/-- The height function of a soccer ball's trajectory -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 11

/-- Theorem stating the maximum height of the soccer ball -/
theorem soccer_ball_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 136 :=
sorry

end NUMINAMATH_CALUDE_soccer_ball_max_height_l2604_260414


namespace NUMINAMATH_CALUDE_difference_of_squares_ratio_l2604_260404

theorem difference_of_squares_ratio : 
  (1732^2 - 1725^2) / (1739^2 - 1718^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_ratio_l2604_260404


namespace NUMINAMATH_CALUDE_smallest_number_l2604_260471

theorem smallest_number (a b c d : ℝ) (h1 : a = 3) (h2 : b = -2) (h3 : c = 1/2) (h4 : d = 2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l2604_260471


namespace NUMINAMATH_CALUDE_fair_coin_five_tosses_l2604_260452

/-- A fair coin is a coin with equal probability of landing on either side. -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of a specific sequence of n tosses for a fair coin. -/
def prob_sequence (n : ℕ) (p : ℝ) : ℝ := p ^ n

/-- The probability of landing on the same side for n tosses of a fair coin. -/
def prob_same_side (n : ℕ) (p : ℝ) : ℝ := 2 * (prob_sequence n p)

theorem fair_coin_five_tosses (p : ℝ) (h : fair_coin p) :
  prob_same_side 5 p = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_five_tosses_l2604_260452


namespace NUMINAMATH_CALUDE_smallest_sum_of_five_primes_with_unique_digits_l2604_260400

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the set of digits used in a number -/
def digitsUsed (n : ℕ) : Finset ℕ := sorry

/-- A function that returns the sum of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem smallest_sum_of_five_primes_with_unique_digits :
  ∃ (primes : List ℕ),
    primes.length = 5 ∧
    (∀ p ∈ primes, isPrime p) ∧
    (digitsUsed (sumList primes) = Finset.range 9) ∧
    (∀ s : List ℕ,
      s.length = 5 →
      (∀ p ∈ s, isPrime p) →
      (digitsUsed (sumList s) = Finset.range 9) →
      sumList primes ≤ sumList s) ∧
    sumList primes = 106 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_five_primes_with_unique_digits_l2604_260400


namespace NUMINAMATH_CALUDE_exam_student_count_l2604_260474

theorem exam_student_count 
  (total_average : ℝ)
  (excluded_count : ℕ)
  (excluded_average : ℝ)
  (remaining_average : ℝ)
  (h1 : total_average = 80)
  (h2 : excluded_count = 5)
  (h3 : excluded_average = 20)
  (h4 : remaining_average = 95)
  : ∃ N : ℕ, N > 0 ∧ 
    N * total_average = 
    (N - excluded_count) * remaining_average + excluded_count * excluded_average :=
by
  sorry

end NUMINAMATH_CALUDE_exam_student_count_l2604_260474


namespace NUMINAMATH_CALUDE_triple_solution_l2604_260442

theorem triple_solution (a b c : ℝ) :
  a^2 + 2*b^2 - 2*b*c = 16 ∧ 2*a*b - c^2 = 16 →
  (a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = -4 ∧ b = -4 ∧ c = -4) := by
sorry

end NUMINAMATH_CALUDE_triple_solution_l2604_260442


namespace NUMINAMATH_CALUDE_scale_division_l2604_260428

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 10 * 12 + 5

/-- The number of parts the scale is divided into -/
def num_parts : ℕ := 5

/-- Calculates the length of each part when the scale is divided equally -/
def part_length : ℕ := scale_length / num_parts

/-- Theorem stating that each part of the scale is 25 inches long -/
theorem scale_division :
  part_length = 25 := by sorry

end NUMINAMATH_CALUDE_scale_division_l2604_260428


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2604_260459

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | x - 2 ≥ -5 ∧ 3*x < x + 2}
  S = {x | -3 ≤ x ∧ x < 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2604_260459


namespace NUMINAMATH_CALUDE_alice_additional_spend_l2604_260479

/-- Represents the grocery store cart with various items and their prices -/
structure GroceryCart where
  chicken : Float
  lettuce : Float
  cherryTomatoes : Float
  sweetPotatoes : Float
  broccoli : Float
  brusselSprouts : Float
  strawberries : Float
  cereal : Float
  groundBeef : Float

/-- Calculates the pre-tax total of the grocery cart -/
def calculatePreTaxTotal (cart : GroceryCart) : Float :=
  cart.chicken + cart.lettuce + cart.cherryTomatoes + cart.sweetPotatoes +
  cart.broccoli + cart.brusselSprouts + cart.strawberries + cart.cereal +
  cart.groundBeef

/-- Theorem: The difference between the minimum spend for free delivery and
    Alice's pre-tax total is $3.02 -/
theorem alice_additional_spend (minSpend : Float) (cart : GroceryCart)
    (h1 : minSpend = 50.00)
    (h2 : cart.chicken = 10.80)
    (h3 : cart.lettuce = 3.50)
    (h4 : cart.cherryTomatoes = 5.00)
    (h5 : cart.sweetPotatoes = 3.75)
    (h6 : cart.broccoli = 6.00)
    (h7 : cart.brusselSprouts = 2.50)
    (h8 : cart.strawberries = 4.80)
    (h9 : cart.cereal = 4.00)
    (h10 : cart.groundBeef = 5.63) :
    minSpend - calculatePreTaxTotal cart = 3.02 := by
  sorry

end NUMINAMATH_CALUDE_alice_additional_spend_l2604_260479


namespace NUMINAMATH_CALUDE_card_selection_count_l2604_260463

theorem card_selection_count (n : ℕ) (h : n > 0) : 
  (Nat.choose (2 * n) n : ℚ) = (Nat.factorial (2 * n)) / ((Nat.factorial n) * (Nat.factorial n)) :=
by sorry

end NUMINAMATH_CALUDE_card_selection_count_l2604_260463


namespace NUMINAMATH_CALUDE_S_not_union_of_finite_arithmetic_progressions_l2604_260421

-- Define the set S
def S : Set ℕ := {n : ℕ | ∀ p q : ℕ, (3 : ℚ) / n ≠ 1 / p + 1 / q}

-- Define what it means for a set to be the union of finitely many arithmetic progressions
def is_union_of_finite_arithmetic_progressions (T : Set ℕ) : Prop :=
  ∃ (k : ℕ) (a b : Fin k → ℕ), T = ⋃ i, {n : ℕ | ∃ m : ℕ, n = a i + m * b i}

-- State the theorem
theorem S_not_union_of_finite_arithmetic_progressions :
  ¬(is_union_of_finite_arithmetic_progressions S) := by
  sorry

end NUMINAMATH_CALUDE_S_not_union_of_finite_arithmetic_progressions_l2604_260421


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2604_260449

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (-1 + Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2604_260449


namespace NUMINAMATH_CALUDE_bag_of_balls_l2604_260423

/-- Given a bag of balls, prove that the total number of balls is 15 -/
theorem bag_of_balls (total_balls : ℕ) 
  (prob_red : ℚ) 
  (num_red : ℕ) : 
  prob_red = 1/5 → 
  num_red = 3 → 
  total_balls = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_bag_of_balls_l2604_260423


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l2604_260490

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ (3 * x^2 + 1 = 4) ↔ (x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l2604_260490


namespace NUMINAMATH_CALUDE_circle_on_parabola_through_focus_l2604_260427

/-- A circle with center on the parabola y² = 8x and tangent to x + 2 = 0 passes through (2, 0) -/
theorem circle_on_parabola_through_focus (x y : ℝ) :
  y^2 = 8*x →  -- center (x, y) is on the parabola
  (x + 2)^2 + y^2 = (x + 4)^2 →  -- circle is tangent to x + 2 = 0
  (2 - x)^2 + y^2 = (x + 4)^2 :=  -- circle passes through (2, 0)
by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_through_focus_l2604_260427


namespace NUMINAMATH_CALUDE_number_division_problem_l2604_260446

theorem number_division_problem (x : ℝ) : x / 5 = 30 + x / 6 ↔ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2604_260446


namespace NUMINAMATH_CALUDE_semicircle_roll_path_length_l2604_260438

theorem semicircle_roll_path_length (r : ℝ) (h : r = 4 / Real.pi) : 
  let semicircle_arc_length := r * Real.pi
  semicircle_arc_length = 4 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_roll_path_length_l2604_260438


namespace NUMINAMATH_CALUDE_two_digit_number_reverse_sum_l2604_260489

theorem two_digit_number_reverse_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 7 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_reverse_sum_l2604_260489


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l2604_260416

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 40 → 
  a * b + b * c + c * d + d * a ≤ 800 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l2604_260416


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l2604_260429

theorem ellipse_parabola_intersection_range (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) →
  -1 ≤ a ∧ a ≤ 17/8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l2604_260429


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l2604_260417

theorem proposition_false_iff_a_in_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l2604_260417


namespace NUMINAMATH_CALUDE_inscribed_square_rectangle_l2604_260434

theorem inscribed_square_rectangle (a b : ℝ) : 
  (∃ (s r_short r_long : ℝ),
    s^2 = 9 ∧                     -- Area of square is 9
    r_long = 2 * r_short ∧        -- One side of rectangle is double the other
    r_short * r_long = 18 ∧       -- Area of rectangle is 18
    a + b = r_short ∧             -- a and b divide the shorter side
    a^2 + b^2 = s^2)              -- Pythagorean theorem for the right triangle formed
  → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_rectangle_l2604_260434


namespace NUMINAMATH_CALUDE_doughnuts_given_away_is_30_l2604_260455

/-- The number of doughnuts in each box -/
def doughnuts_per_box : ℕ := 10

/-- The total number of doughnuts made for the day -/
def total_doughnuts : ℕ := 300

/-- The number of boxes sold -/
def boxes_sold : ℕ := 27

/-- The number of doughnuts given away at the end of the day -/
def doughnuts_given_away : ℕ := total_doughnuts - (boxes_sold * doughnuts_per_box)

theorem doughnuts_given_away_is_30 : doughnuts_given_away = 30 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_given_away_is_30_l2604_260455


namespace NUMINAMATH_CALUDE_sum_two_angles_gt_90_implies_acute_l2604_260411

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

end NUMINAMATH_CALUDE_sum_two_angles_gt_90_implies_acute_l2604_260411


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2604_260466

/-- Given an ellipse and a hyperbola with the same vertices, prove the equation of the hyperbola -/
theorem hyperbola_equation (e : ℝ) (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2 / 16 + y^2 / 9 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) 
  (h_eccentricity : e = 2) (h_vertices : a = 4 ∧ b = 3) :
  (∀ x y : ℝ, x^2 / 16 - y^2 / 48 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / (e^2 * a^2 - a^2) = 1}) ∨
  (∀ x y : ℝ, y^2 / 9 - x^2 / 27 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.2^2 / b^2 - p.1^2 / (e^2 * b^2 - b^2) = 1}) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2604_260466


namespace NUMINAMATH_CALUDE_af_equals_kc_l2604_260409

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points
variable (O : ℝ × ℝ)  -- Center of the circle
variable (G H E D B A C F K : ℝ × ℝ)

-- Define the circle
variable (circle : Circle)

-- Define conditions
variable (gh_diameter : (G.1 - H.1)^2 + (G.2 - H.2)^2 = 4 * circle.radius^2)
variable (ed_diameter : (E.1 - D.1)^2 + (E.2 - D.2)^2 = 4 * circle.radius^2)
variable (perpendicular_diameters : (G.1 - H.1) * (E.1 - D.1) + (G.2 - H.2) * (E.2 - D.2) = 0)
variable (b_outside : (B.1 - circle.center.1)^2 + (B.2 - circle.center.2)^2 > circle.radius^2)
variable (a_on_circle : (A.1 - circle.center.1)^2 + (A.2 - circle.center.2)^2 = circle.radius^2)
variable (c_on_circle : (C.1 - circle.center.1)^2 + (C.2 - circle.center.2)^2 = circle.radius^2)
variable (a_on_gh : A.2 = G.2 ∧ A.2 = H.2)
variable (c_on_gh : C.2 = G.2 ∧ C.2 = H.2)
variable (f_on_gh : F.2 = G.2 ∧ F.2 = H.2)
variable (k_on_gh : K.2 = G.2 ∧ K.2 = H.2)
variable (ba_tangent : (B.1 - A.1) * (A.1 - circle.center.1) + (B.2 - A.2) * (A.2 - circle.center.2) = 0)
variable (bc_tangent : (B.1 - C.1) * (C.1 - circle.center.1) + (B.2 - C.2) * (C.2 - circle.center.2) = 0)
variable (be_intersects_gh_at_f : (B.1 - E.1) * (F.2 - B.2) = (F.1 - B.1) * (B.2 - E.2))
variable (bd_intersects_gh_at_k : (B.1 - D.1) * (K.2 - B.2) = (K.1 - B.1) * (B.2 - D.2))

-- Theorem statement
theorem af_equals_kc : (A.1 - F.1)^2 + (A.2 - F.2)^2 = (K.1 - C.1)^2 + (K.2 - C.2)^2 := by sorry

end NUMINAMATH_CALUDE_af_equals_kc_l2604_260409


namespace NUMINAMATH_CALUDE_stones_per_bracelet_l2604_260470

theorem stones_per_bracelet (total_stones : ℕ) (num_bracelets : ℕ) 
  (h1 : total_stones = 140) (h2 : num_bracelets = 10) :
  total_stones / num_bracelets = 14 := by
  sorry

end NUMINAMATH_CALUDE_stones_per_bracelet_l2604_260470


namespace NUMINAMATH_CALUDE_calculation_proof_l2604_260424

theorem calculation_proof : Real.sqrt 2 * Real.sqrt 2 - 4 * Real.sin (π / 6) + (1 / 2)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2604_260424


namespace NUMINAMATH_CALUDE_prime_triplet_theorem_l2604_260413

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_geometric_progression (a b c : ℕ) : Prop := (b + 1)^2 = (a + 1) * (c + 1)

def valid_prime_triplet (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  is_geometric_progression a b c

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 5, 11), (2, 11, 47), (5, 11, 23), (5, 17, 53), (7, 11, 17), (7, 23, 71),
   (11, 23, 47), (17, 23, 31), (17, 41, 97), (31, 47, 71), (71, 83, 97)}

theorem prime_triplet_theorem :
  {x : ℕ × ℕ × ℕ | valid_prime_triplet x.1 x.2.1 x.2.2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_theorem_l2604_260413


namespace NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l2604_260406

/-- The original parabola function -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 4

/-- The shifted parabola function -/
def g (x : ℝ) : ℝ := -x^2 + 2

/-- Theorem stating that the shifted parabola passes through (-1, 1) -/
theorem shifted_parabola_passes_through_point :
  g (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l2604_260406


namespace NUMINAMATH_CALUDE_sum_of_positive_numbers_l2604_260435

theorem sum_of_positive_numbers (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + y + x * y = 8)
  (eq2 : y + z + y * z = 15)
  (eq3 : z + x + z * x = 35) :
  x + y + z + x * y = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_positive_numbers_l2604_260435


namespace NUMINAMATH_CALUDE_cone_from_sector_l2604_260492

theorem cone_from_sector (sector_angle : Real) (sector_radius : Real) 
  (h_angle : sector_angle = 270) (h_radius : sector_radius = 12) :
  ∃ (base_radius : Real),
    base_radius = 9 ∧ 
    2 * Real.pi * base_radius = (sector_angle / 360) * (2 * Real.pi * sector_radius) :=
by sorry

end NUMINAMATH_CALUDE_cone_from_sector_l2604_260492


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l2604_260425

/-- The parabola equation: x = -y^2 + 2y + 3 -/
def parabola (x y : ℝ) : Prop := x = -y^2 + 2*y + 3

/-- An x-intercept is a point where the parabola crosses the x-axis (y = 0) -/
def is_x_intercept (x : ℝ) : Prop := parabola x 0

/-- The parabola has exactly one x-intercept -/
theorem parabola_has_one_x_intercept : ∃! x : ℝ, is_x_intercept x := by sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l2604_260425


namespace NUMINAMATH_CALUDE_minhyung_top_choices_l2604_260453

/-- The number of cases for choosing a top -/
def choose_top_cases (initial_tops : ℕ) (new_tops : ℕ) : ℕ :=
  initial_tops + new_tops

theorem minhyung_top_choices (initial_tops new_tops : ℕ) :
  choose_top_cases initial_tops new_tops = initial_tops + new_tops :=
by sorry

#eval choose_top_cases 2 4  -- Should output 6

end NUMINAMATH_CALUDE_minhyung_top_choices_l2604_260453


namespace NUMINAMATH_CALUDE_average_height_is_10_8_l2604_260408

def tree_heights (h1 h2 h3 h4 h5 : ℕ) : Prop :=
  h2 = 18 ∧
  (h1 = 3 * h2 ∨ h1 * 3 = h2) ∧
  (h2 = 3 * h3 ∨ h2 * 3 = h3) ∧
  (h3 = 3 * h4 ∨ h3 * 3 = h4) ∧
  (h4 = 3 * h5 ∨ h4 * 3 = h5)

theorem average_height_is_10_8 :
  ∃ (h1 h2 h3 h4 h5 : ℕ), tree_heights h1 h2 h3 h4 h5 ∧
  (h1 + h2 + h3 + h4 + h5) / 5 = 54 / 5 :=
sorry

end NUMINAMATH_CALUDE_average_height_is_10_8_l2604_260408


namespace NUMINAMATH_CALUDE_final_fruit_juice_percentage_l2604_260444

/-- Given an initial mixture of punch and some added pure fruit juice,
    calculate the final percentage of fruit juice in the punch. -/
theorem final_fruit_juice_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_juice : ℝ)
  (h1 : initial_volume = 2)
  (h2 : initial_percentage = 0.1)
  (h3 : added_juice = 0.4)
  : (initial_volume * initial_percentage + added_juice) / (initial_volume + added_juice) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_final_fruit_juice_percentage_l2604_260444


namespace NUMINAMATH_CALUDE_dog_weight_l2604_260467

theorem dog_weight (d l s : ℝ) 
  (total_weight : d + l + s = 36)
  (larger_comparison : d + l = 3 * s)
  (smaller_comparison : d + s = l) : 
  d = 9 := by
  sorry

end NUMINAMATH_CALUDE_dog_weight_l2604_260467


namespace NUMINAMATH_CALUDE_sibling_age_equation_l2604_260460

/-- Represents the ages of two siblings -/
structure SiblingAges where
  sister : ℕ
  brother : ℕ

/-- The condition of the ages this year -/
def this_year (ages : SiblingAges) : Prop :=
  ages.brother = 2 * ages.sister

/-- The condition of the ages four years ago -/
def four_years_ago (ages : SiblingAges) : Prop :=
  (ages.brother - 4) = 3 * (ages.sister - 4)

/-- The theorem representing the problem -/
theorem sibling_age_equation (x : ℕ) :
  ∃ (ages : SiblingAges),
    ages.sister = x ∧
    this_year ages ∧
    four_years_ago ages →
    2 * x - 4 = 3 * (x - 4) :=
by
  sorry

end NUMINAMATH_CALUDE_sibling_age_equation_l2604_260460


namespace NUMINAMATH_CALUDE_multiply_fractions_l2604_260443

theorem multiply_fractions (a b : ℝ) : (1 / 3) * a^2 * (-6 * a * b) = -2 * a^3 * b := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_l2604_260443


namespace NUMINAMATH_CALUDE_triangle_area_l2604_260410

/-- A triangle with sides in ratio 3:4:5 and perimeter 60 has area 150 -/
theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (3, 4, 5)) 
  (h_perimeter : a + b + c = 60) : 
  (1/2) * a * b = 150 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2604_260410


namespace NUMINAMATH_CALUDE_solutions_count_l2604_260483

/-- The number of solutions to the equation √(x+3) = ax + 2 depends on the value of parameter a -/
theorem solutions_count (a : ℝ) : 
  (∀ x, Real.sqrt (x + 3) ≠ a * x + 2) ∨ 
  (∃! x, Real.sqrt (x + 3) = a * x + 2) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ Real.sqrt (x₁ + 3) = a * x₁ + 2 ∧ Real.sqrt (x₂ + 3) = a * x₂ + 2 ∧ 
    ∀ x, Real.sqrt (x + 3) = a * x + 2 → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    Real.sqrt (x₁ + 3) = a * x₁ + 2 ∧ Real.sqrt (x₂ + 3) = a * x₂ + 2 ∧ Real.sqrt (x₃ + 3) = a * x₃ + 2 ∧
    ∀ x, Real.sqrt (x + 3) = a * x + 2 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∨
  (∃ x₁ x₂ x₃ x₄, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    Real.sqrt (x₁ + 3) = a * x₁ + 2 ∧ Real.sqrt (x₂ + 3) = a * x₂ + 2 ∧ 
    Real.sqrt (x₃ + 3) = a * x₃ + 2 ∧ Real.sqrt (x₄ + 3) = a * x₄ + 2 ∧
    ∀ x, Real.sqrt (x + 3) = a * x + 2 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) :=
by
  sorry

end NUMINAMATH_CALUDE_solutions_count_l2604_260483


namespace NUMINAMATH_CALUDE_max_value_theorem_l2604_260484

theorem max_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2*x*y + 3*y^2 = 12) :
  ∃ (M : ℝ), M = 24 + 24*Real.sqrt 3 ∧ x^2 + 2*x*y + 3*y^2 ≤ M ∧
  ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x'^2 - 2*x'*y' + 3*y'^2 = 12 ∧ x'^2 + 2*x'*y' + 3*y'^2 = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2604_260484


namespace NUMINAMATH_CALUDE_badminton_partitions_l2604_260486

def number_of_partitions (n : ℕ) : ℕ := (n.choose 2) * ((n - 2).choose 2) / 2

theorem badminton_partitions :
  number_of_partitions 6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_badminton_partitions_l2604_260486


namespace NUMINAMATH_CALUDE_woodworker_tables_l2604_260420

/-- Calculates the number of tables made given the total number of furniture legs,
    number of chairs, legs per chair, and legs per table. -/
def tables_made (total_legs : ℕ) (chairs : ℕ) (legs_per_chair : ℕ) (legs_per_table : ℕ) : ℕ :=
  (total_legs - chairs * legs_per_chair) / legs_per_table

/-- Theorem stating that given 40 total furniture legs, 6 chairs made,
    4 legs per chair, and 4 legs per table, the number of tables made is 4. -/
theorem woodworker_tables :
  tables_made 40 6 4 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_tables_l2604_260420


namespace NUMINAMATH_CALUDE_deposit_this_month_is_34_l2604_260491

/-- Represents a bank account with deposits and withdrawals -/
structure BankAccount where
  initial_balance : ℤ
  deposit_last_month : ℤ
  withdrawal_last_month : ℤ
  deposit_this_month : ℤ

/-- Calculates the final balance of the bank account -/
def final_balance (account : BankAccount) : ℤ :=
  account.initial_balance + account.deposit_last_month - account.withdrawal_last_month + account.deposit_this_month

/-- Theorem stating that the deposit this month is $34 -/
theorem deposit_this_month_is_34 (account : BankAccount) :
  account.initial_balance = 150 →
  account.deposit_last_month = 17 →
  final_balance account = account.initial_balance + 16 →
  account.deposit_this_month = 34 := by
  sorry

#check deposit_this_month_is_34

end NUMINAMATH_CALUDE_deposit_this_month_is_34_l2604_260491


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2604_260402

theorem sum_of_decimals : 0.4 + 0.02 + 0.006 = 0.426 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2604_260402


namespace NUMINAMATH_CALUDE_expression_evaluation_l2604_260480

theorem expression_evaluation (x y : ℤ) (hx : x = -1) (hy : y = 2) :
  2 * x * y + (3 * x * y - 2 * y^2) - 2 * (x * y - y^2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2604_260480


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l2604_260407

theorem students_taking_neither_music_nor_art 
  (total_students : ℕ) 
  (music_students : ℕ) 
  (art_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 500) 
  (h2 : music_students = 20) 
  (h3 : art_students = 20) 
  (h4 : both_students = 10) : 
  total_students - (music_students + art_students - both_students) = 470 := by
  sorry

#check students_taking_neither_music_nor_art

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l2604_260407
