import Mathlib

namespace NUMINAMATH_CALUDE_line_through_points_l1183_118305

/-- A line with slope 4 passing through points (3,5), (a,7), and (-1,b) has a = 7/2 and b = -11 -/
theorem line_through_points (a b : ℚ) : 
  (((7 - 5) / (a - 3) = 4) ∧ ((b - 5) / (-1 - 3) = 4)) → 
  (a = 7/2 ∧ b = -11) := by
sorry

end NUMINAMATH_CALUDE_line_through_points_l1183_118305


namespace NUMINAMATH_CALUDE_multiple_problem_l1183_118379

theorem multiple_problem (x y : ℕ) (k m : ℕ) : 
  x = 11 → 
  x + y = 55 → 
  y = k * x + m → 
  k = 4 ∧ m = 0 := by
sorry

end NUMINAMATH_CALUDE_multiple_problem_l1183_118379


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1183_118381

theorem smallest_number_of_eggs (n : ℕ) (c : ℕ) : 
  n > 150 →
  n = 15 * c - 6 →
  c ≥ 11 →
  (∀ m : ℕ, m > 150 ∧ (∃ k : ℕ, m = 15 * k - 6) → m ≥ n) →
  n = 159 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1183_118381


namespace NUMINAMATH_CALUDE_comparison_sqrt_l1183_118369

theorem comparison_sqrt : 3 * Real.sqrt 2 > Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_comparison_sqrt_l1183_118369


namespace NUMINAMATH_CALUDE_shaded_area_in_circle_l1183_118364

theorem shaded_area_in_circle (r : ℝ) (h : r = 6) : 
  let angle : ℝ := π / 3  -- 60° in radians
  let triangle_area : ℝ := (1/2) * r * r * Real.sin angle
  let sector_area : ℝ := (angle / (2 * π)) * π * r^2
  2 * triangle_area + 2 * sector_area = 36 * Real.sqrt 3 + 12 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_in_circle_l1183_118364


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1183_118371

theorem opposite_of_negative_two :
  ∃ x : ℝ, x + (-2) = 0 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1183_118371


namespace NUMINAMATH_CALUDE_shaded_area_is_thirty_l1183_118395

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_ten : leg_length = 10

/-- The large triangle partitioned into 25 congruent smaller triangles -/
def num_partitions : ℕ := 25

/-- The number of shaded smaller triangles -/
def num_shaded : ℕ := 15

/-- Theorem stating the area of the shaded region -/
theorem shaded_area_is_thirty (t : IsoscelesRightTriangle) : 
  (t.leg_length^2 / 2) * (num_shaded / num_partitions) = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_thirty_l1183_118395


namespace NUMINAMATH_CALUDE_ellipsoid_sum_center_axes_l1183_118347

/-- The equation of a tilted three-dimensional ellipsoid -/
def ellipsoid_equation (x y z x₀ y₀ z₀ A B C : ℝ) : Prop :=
  (x - x₀)^2 / A^2 + (y - y₀)^2 / B^2 + (z - z₀)^2 / C^2 = 1

/-- Theorem: Sum of center coordinates and semi-major axes lengths -/
theorem ellipsoid_sum_center_axes :
  ∀ (x₀ y₀ z₀ A B C : ℝ),
  ellipsoid_equation x y z x₀ y₀ z₀ A B C →
  x₀ = -2 →
  y₀ = 3 →
  z₀ = 1 →
  A = 6 →
  B = 4 →
  C = 2 →
  x₀ + y₀ + z₀ + A + B + C = 14 :=
by sorry

end NUMINAMATH_CALUDE_ellipsoid_sum_center_axes_l1183_118347


namespace NUMINAMATH_CALUDE_special_matrix_product_l1183_118394

/-- A 5x5 matrix with special properties -/
structure SpecialMatrix where
  a : Fin 5 → Fin 5 → ℝ
  first_row_arithmetic : ∀ i j k : Fin 5, a 0 j - a 0 i = a 0 k - a 0 j
    → j - i = k - j
  columns_geometric : ∃ q : ℝ, ∀ i j : Fin 5, a (i+1) j = q * a i j
  a24_eq_4 : a 1 3 = 4
  a41_eq_neg2 : a 3 0 = -2
  a43_eq_10 : a 3 2 = 10

/-- The product of a₁₁ and a₅₅ is -11 -/
theorem special_matrix_product (m : SpecialMatrix) : m.a 0 0 * m.a 4 4 = -11 := by
  sorry

end NUMINAMATH_CALUDE_special_matrix_product_l1183_118394


namespace NUMINAMATH_CALUDE_mildred_blocks_l1183_118397

theorem mildred_blocks (initial_blocks found_blocks : ℕ) : 
  initial_blocks = 2 → found_blocks = 84 → initial_blocks + found_blocks = 86 := by
  sorry

end NUMINAMATH_CALUDE_mildred_blocks_l1183_118397


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1183_118309

theorem similar_triangle_perimeter (a b c d e f : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for the original triangle
  d^2 + e^2 = f^2 →  -- Pythagorean theorem for the similar triangle
  d / a = e / b →    -- Similarity condition
  d / a = f / c →    -- Similarity condition
  a = 6 →            -- Given length of shorter leg of original triangle
  b = 8 →            -- Given length of longer leg of original triangle
  d = 15 →           -- Given length of shorter leg of similar triangle
  d + e + f = 60 :=  -- Perimeter of the similar triangle
by
  sorry


end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1183_118309


namespace NUMINAMATH_CALUDE_triangle_value_proof_l1183_118302

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

theorem triangle_value_proof :
  ∀ (triangle : Nat),
  (triangle < 7) →
  (triangle < 9) →
  (base_to_decimal [triangle, 5] 7 = base_to_decimal [3, triangle] 9) →
  triangle = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_value_proof_l1183_118302


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1183_118373

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a, b, and c are constants, and a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + x - 4 = 0 -/
def f (x : ℝ) : ℝ := x^2 + x - 4

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1183_118373


namespace NUMINAMATH_CALUDE_shortest_wire_for_given_poles_l1183_118399

/-- Represents a cylindrical pole with a given diameter -/
structure Pole where
  diameter : ℝ

/-- Calculates the shortest wire length to wrap around three poles -/
def shortestWireLength (pole1 pole2 pole3 : Pole) : ℝ :=
  sorry

/-- The theorem stating the shortest wire length for the given poles -/
theorem shortest_wire_for_given_poles :
  let pole1 : Pole := ⟨6⟩
  let pole2 : Pole := ⟨18⟩
  let pole3 : Pole := ⟨12⟩
  shortestWireLength pole1 pole2 pole3 = 6 * Real.sqrt 3 + 6 * Real.sqrt 6 + 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shortest_wire_for_given_poles_l1183_118399


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1183_118350

theorem quadratic_equation_solution :
  ∀ x : ℝ, (x - 2)^2 - 4 = 0 ↔ x = 4 ∨ x = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1183_118350


namespace NUMINAMATH_CALUDE_translation_problem_l1183_118356

/-- A translation of the complex plane -/
def ComplexTranslation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

theorem translation_problem (t : ℂ → ℂ) (h : t (1 + 3*I) = 4 + 2*I) :
  ∃ w : ℂ, t = ComplexTranslation w ∧ t (3 - 2*I) = 6 - 3*I := by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l1183_118356


namespace NUMINAMATH_CALUDE_security_deposit_is_1110_l1183_118353

/-- Calculates the security deposit for a cabin rental --/
def calculate_security_deposit (weeks : ℕ) (daily_rate : ℚ) (pet_fee : ℚ) (service_fee_rate : ℚ) (deposit_rate : ℚ) : ℚ :=
  let days := weeks * 7
  let rental_fee := daily_rate * days
  let total_rental := rental_fee + pet_fee
  let service_fee := service_fee_rate * total_rental
  let total_cost := total_rental + service_fee
  deposit_rate * total_cost

/-- Theorem: The security deposit for the given conditions is $1,110.00 --/
theorem security_deposit_is_1110 :
  calculate_security_deposit 2 125 100 (1/5) (1/2) = 1110 := by
  sorry

end NUMINAMATH_CALUDE_security_deposit_is_1110_l1183_118353


namespace NUMINAMATH_CALUDE_percentage_of_whole_l1183_118310

theorem percentage_of_whole (whole : ℝ) (part : ℝ) (h : whole = 450 ∧ part = 229.5) :
  (part / whole) * 100 = 51 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_whole_l1183_118310


namespace NUMINAMATH_CALUDE_min_value_theorem_l1183_118361

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ x : ℝ, 1 ≤ x → x ≤ 4 → a * x + b - 3 ≤ 0) : 
  1 / a - b ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1183_118361


namespace NUMINAMATH_CALUDE_probability_at_least_three_white_is_550_715_l1183_118334

def white_balls : ℕ := 8
def black_balls : ℕ := 7
def total_balls : ℕ := white_balls + black_balls
def drawn_balls : ℕ := 6

def probability_at_least_three_white : ℚ :=
  (Nat.choose white_balls 3 * Nat.choose black_balls 3 +
   Nat.choose white_balls 4 * Nat.choose black_balls 2 +
   Nat.choose white_balls 5 * Nat.choose black_balls 1 +
   Nat.choose white_balls 6 * Nat.choose black_balls 0) /
  Nat.choose total_balls drawn_balls

theorem probability_at_least_three_white_is_550_715 :
  probability_at_least_three_white = 550 / 715 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_three_white_is_550_715_l1183_118334


namespace NUMINAMATH_CALUDE_max_intersecting_chords_2017_l1183_118392

/-- The maximum number of intersecting chords for a circle with n points -/
def max_intersecting_chords (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  k * (n - 1 - k) + (n - 1)

/-- Theorem stating the maximum number of intersecting chords for 2017 points -/
theorem max_intersecting_chords_2017 :
  max_intersecting_chords 2017 = 1018080 := by
  sorry

#eval max_intersecting_chords 2017

end NUMINAMATH_CALUDE_max_intersecting_chords_2017_l1183_118392


namespace NUMINAMATH_CALUDE_melissa_commission_l1183_118352

/-- Calculates the commission earned by Melissa based on vehicle sales --/
def calculate_commission (coupe_price suv_price luxury_sedan_price motorcycle_price truck_price : ℕ)
  (coupe_sold suv_sold luxury_sedan_sold motorcycle_sold truck_sold : ℕ) : ℕ :=
  let total_sales := coupe_price * coupe_sold + (2 * coupe_price) * suv_sold +
                     luxury_sedan_price * luxury_sedan_sold + motorcycle_price * motorcycle_sold +
                     truck_price * truck_sold
  let total_vehicles := coupe_sold + suv_sold + luxury_sedan_sold + motorcycle_sold + truck_sold
  let commission_rate := if total_vehicles ≤ 2 then 2
                         else if total_vehicles ≤ 4 then 25
                         else 3
  (total_sales * commission_rate) / 100

theorem melissa_commission :
  calculate_commission 30000 60000 80000 15000 40000 3 2 1 4 2 = 12900 :=
by sorry

end NUMINAMATH_CALUDE_melissa_commission_l1183_118352


namespace NUMINAMATH_CALUDE_jerry_age_l1183_118382

/-- Given that Mickey's age is 8 years less than 200% of Jerry's age,
    and Mickey is 16 years old, prove that Jerry is 12 years old. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 16)
  (h2 : mickey_age = 2 * jerry_age - 8) : 
  jerry_age = 12 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l1183_118382


namespace NUMINAMATH_CALUDE_F_3_f_4_equals_7_l1183_118360

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 2
def F (a b : ℝ) : ℝ := b^2 + a

-- State the theorem
theorem F_3_f_4_equals_7 : F 3 (f 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_F_3_f_4_equals_7_l1183_118360


namespace NUMINAMATH_CALUDE_photographer_profit_percentage_l1183_118390

/-- Calculates the profit percentage for a photographer's business --/
theorem photographer_profit_percentage
  (selling_price : ℝ)
  (production_cost : ℝ)
  (sale_probability : ℝ)
  (h1 : selling_price = 600)
  (h2 : production_cost = 100)
  (h3 : sale_probability = 1/4)
  : (((sale_probability * selling_price - production_cost) / production_cost) * 100 = 50) :=
by sorry

end NUMINAMATH_CALUDE_photographer_profit_percentage_l1183_118390


namespace NUMINAMATH_CALUDE_cubic_inequality_l1183_118389

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 - 7*x + 10 > 0 ↔ x < -2 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1183_118389


namespace NUMINAMATH_CALUDE_craig_travel_difference_l1183_118383

theorem craig_travel_difference :
  let bus_distance : ℝ := 3.83
  let walk_distance : ℝ := 0.17
  bus_distance - walk_distance = 3.66 := by sorry

end NUMINAMATH_CALUDE_craig_travel_difference_l1183_118383


namespace NUMINAMATH_CALUDE_average_speed_two_segment_trip_l1183_118324

theorem average_speed_two_segment_trip (d1 d2 v1 v2 : ℝ) 
  (h1 : d1 = 45) (h2 : d2 = 15) (h3 : v1 = 15) (h4 : v2 = 45) :
  (d1 + d2) / ((d1 / v1) + (d2 / v2)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_segment_trip_l1183_118324


namespace NUMINAMATH_CALUDE_arrangement_count_proof_l1183_118349

/-- The number of ways to arrange 2 female students and 4 male students in a row,
    such that female student A is to the left of female student B. -/
def arrangement_count : ℕ := 360

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of female students -/
def female_students : ℕ := 2

/-- The number of male students -/
def male_students : ℕ := 4

theorem arrangement_count_proof :
  arrangement_count = (Nat.factorial total_students) / 2 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_proof_l1183_118349


namespace NUMINAMATH_CALUDE_helmet_sales_theorem_l1183_118301

/-- Represents the helmet sales scenario -/
structure HelmetSales where
  initialPrice : ℝ
  initialSales : ℝ
  priceReductionEffect : ℝ
  costPrice : ℝ

/-- Calculates the number of helmets sold after a price reduction -/
def helmetsSold (hs : HelmetSales) (priceReduction : ℝ) : ℝ :=
  hs.initialSales + hs.priceReductionEffect * priceReduction

/-- Calculates the monthly profit -/
def monthlyProfit (hs : HelmetSales) (priceReduction : ℝ) : ℝ :=
  (hs.initialPrice - priceReduction - hs.costPrice) * (helmetsSold hs priceReduction)

/-- The main theorem about helmet sales -/
theorem helmet_sales_theorem (hs : HelmetSales) 
    (h1 : hs.initialPrice = 80)
    (h2 : hs.initialSales = 200)
    (h3 : hs.priceReductionEffect = 20)
    (h4 : hs.costPrice = 50) : 
    (helmetsSold hs 10 = 400 ∧ monthlyProfit hs 10 = 8000) ∧
    ∃ x, x > 0 ∧ monthlyProfit hs x = 7500 ∧ hs.initialPrice - x = 65 := by
  sorry


end NUMINAMATH_CALUDE_helmet_sales_theorem_l1183_118301


namespace NUMINAMATH_CALUDE_bus_passenger_ratio_l1183_118300

/-- Represents the number of passengers on a bus --/
structure BusPassengers where
  men : ℕ
  women : ℕ

/-- The initial state of passengers on the bus --/
def initial : BusPassengers := sorry

/-- The state of passengers after changes in city Y --/
def after_city_y : BusPassengers := sorry

/-- The total number of passengers at the start --/
def total_passengers : ℕ := 72

/-- Changes in passenger numbers at city Y --/
def men_leave : ℕ := 16
def women_enter : ℕ := 8

theorem bus_passenger_ratio :
  initial.men = 2 * initial.women ∧
  initial.men + initial.women = total_passengers ∧
  after_city_y.men = initial.men - men_leave ∧
  after_city_y.women = initial.women + women_enter ∧
  after_city_y.men = after_city_y.women :=
by sorry

end NUMINAMATH_CALUDE_bus_passenger_ratio_l1183_118300


namespace NUMINAMATH_CALUDE_no_real_solution_cubic_equation_l1183_118328

theorem no_real_solution_cubic_equation :
  ∀ x : ℝ, x > 0 → 4 * x^(1/3) - 3 * (x / x^(2/3)) ≠ 10 + 2 * x^(1/3) + x^(2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_cubic_equation_l1183_118328


namespace NUMINAMATH_CALUDE_solve_for_y_l1183_118336

theorem solve_for_y (x y : ℝ) (h1 : 3 * x^2 = y - 6) (h2 : x = 4) : y = 54 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1183_118336


namespace NUMINAMATH_CALUDE_function_value_proof_l1183_118308

theorem function_value_proof (f : ℝ → ℝ) :
  (3 : ℝ) + 17 = 60 * f 3 → f 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_proof_l1183_118308


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1183_118365

-- Define the variables
variable (P : ℝ) -- Principal amount
variable (R : ℝ) -- Original interest rate in percentage

-- Define the theorem
theorem simple_interest_problem :
  (P * (R + 3) * 2) / 100 - (P * R * 2) / 100 = 300 →
  P = 5000 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1183_118365


namespace NUMINAMATH_CALUDE_line_only_count_l1183_118351

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total_letters : ℕ)
  (dot_and_line : ℕ)
  (dot_only : ℕ)
  (h_total : total_letters = 40)
  (h_dot_and_line : dot_and_line = 9)
  (h_dot_only : dot_only = 7)
  (h_all_contain : total_letters = dot_and_line + dot_only + (total_letters - (dot_and_line + dot_only)))

/-- The number of letters containing a straight line but not a dot -/
def line_only (α : Alphabet) : ℕ := α.total_letters - (α.dot_and_line + α.dot_only)

theorem line_only_count (α : Alphabet) : line_only α = 24 := by
  sorry

end NUMINAMATH_CALUDE_line_only_count_l1183_118351


namespace NUMINAMATH_CALUDE_f_lower_bound_l1183_118311

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 / 2 - (a + 1) * x

theorem f_lower_bound :
  ∀ x : ℝ, x > 0 → f (-1) x ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_l1183_118311


namespace NUMINAMATH_CALUDE_binary_subtraction_l1183_118340

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def a : List Bool := [true, true, true, true, true, true, true, true, true]
def b : List Bool := [true, true, true, true]

theorem binary_subtraction :
  binary_to_decimal a - binary_to_decimal b = 496 := by
  sorry

end NUMINAMATH_CALUDE_binary_subtraction_l1183_118340


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l1183_118335

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_min_value
  (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0) (h_geom : geometric_sequence a q)
  (h_relation : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n, Real.sqrt (a m * a n) = 4 * a 1) :
  (∀ m n, Real.sqrt (a m * a n) = 4 * a 1 → 1 / m + 5 / n ≥ 7 / 4) ∧
  (∃ m n, Real.sqrt (a m * a n) = 4 * a 1 ∧ 1 / m + 5 / n = 7 / 4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l1183_118335


namespace NUMINAMATH_CALUDE_sport_water_amount_l1183_118303

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  cornSyrup : ℚ
  water : ℚ

/-- Represents the amount of ingredients in ounces -/
structure DrinkAmount where
  flavoring : ℚ
  cornSyrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standardRatio : DrinkRatio :=
  { flavoring := 1, cornSyrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sportRatio (standard : DrinkRatio) : DrinkRatio :=
  { flavoring := standard.flavoring,
    cornSyrup := standard.cornSyrup / 3,
    water := standard.water * 2 }

/-- Theorem stating the amount of water in the sport formulation -/
theorem sport_water_amount 
  (standard : DrinkRatio)
  (sport : DrinkRatio)
  (sportAmount : DrinkAmount)
  (h1 : sport = sportRatio standard)
  (h2 : sportAmount.cornSyrup = 2) :
  sportAmount.water = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_sport_water_amount_l1183_118303


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1183_118372

def A : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1183_118372


namespace NUMINAMATH_CALUDE_hyperbola_equation_hyperbola_eccentricity_l1183_118391

/-- Represents a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_eq : c = 2
  h_asymptote : b = a

/-- The equation of the hyperbola is (x^2 / 2) - (y^2 / 2) = 1 -/
theorem hyperbola_equation (h : Hyperbola) : 
  h.a^2 = 2 ∧ h.b^2 = 2 :=
sorry

/-- The eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  Real.sqrt (h.c^2 / h.a^2) = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_hyperbola_eccentricity_l1183_118391


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_quadratic_l1183_118341

theorem negation_of_universal_positive_quadratic :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_quadratic_l1183_118341


namespace NUMINAMATH_CALUDE_quadratic_root_value_l1183_118312

theorem quadratic_root_value (x : ℝ) : x = -4 → Real.sqrt (1 - 2*x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1183_118312


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l1183_118316

theorem quadratic_form_minimum : ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -4.45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l1183_118316


namespace NUMINAMATH_CALUDE_fraction_product_l1183_118396

theorem fraction_product : (2 : ℚ) / 9 * 5 / 8 = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1183_118396


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1183_118374

theorem simplify_complex_fraction :
  1 / (1 / (Real.sqrt 2 + 1) + 2 / (Real.sqrt 3 - 1) + 3 / (Real.sqrt 5 + 2)) =
  1 / (Real.sqrt 2 + 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 5) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1183_118374


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4872_l1183_118342

theorem largest_prime_factor_of_4872 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4872 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4872 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4872_l1183_118342


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l1183_118348

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - (a + 3) * x + 2 = 0 ∧ 
               a * y^2 - (a + 3) * y + 2 = 0 ∧ 
               x * y < 0) ↔ 
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l1183_118348


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1183_118306

theorem arithmetic_expression_equality : -6 * 5 - (-4 * -2) + (-12 * -6) / 3 = -14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1183_118306


namespace NUMINAMATH_CALUDE_balls_removed_l1183_118377

def initial_balls : ℕ := 8
def current_balls : ℕ := 6

theorem balls_removed : initial_balls - current_balls = 2 := by
  sorry

end NUMINAMATH_CALUDE_balls_removed_l1183_118377


namespace NUMINAMATH_CALUDE_max_snacks_l1183_118358

theorem max_snacks (S : ℕ) : 
  (∀ n : ℕ, n ≤ S → n > 6 * 18 ∧ n < 7 * 18) → 
  S = 125 := by
  sorry

end NUMINAMATH_CALUDE_max_snacks_l1183_118358


namespace NUMINAMATH_CALUDE_sum_xy_given_condition_l1183_118398

theorem sum_xy_given_condition (x y : ℝ) : 
  |x + 3| + (y - 2)^2 = 0 → x + y = -1 := by sorry

end NUMINAMATH_CALUDE_sum_xy_given_condition_l1183_118398


namespace NUMINAMATH_CALUDE_expression_range_l1183_118387

theorem expression_range (x y : ℝ) (h : (x - 1)^2 + (y - 4)^2 = 1) :
  0 ≤ (x*y - x) / (x^2 + (y - 1)^2) ∧ (x*y - x) / (x^2 + (y - 1)^2) ≤ 12/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_range_l1183_118387


namespace NUMINAMATH_CALUDE_couples_matching_l1183_118378

structure Couple where
  wife : String
  husband : String
  wife_bottles : ℕ
  husband_bottles : ℕ

def total_bottles : ℕ := 44

def couples : List Couple := [
  ⟨"Anna", "", 2, 0⟩,
  ⟨"Betty", "", 3, 0⟩,
  ⟨"Carol", "", 4, 0⟩,
  ⟨"Dorothy", "", 5, 0⟩
]

def husbands : List String := ["Brown", "Green", "White", "Smith"]

theorem couples_matching :
  ∃ (matched_couples : List Couple),
    matched_couples.length = 4 ∧
    (matched_couples.map (λ c => c.wife_bottles + c.husband_bottles)).sum = total_bottles ∧
    (∃ c ∈ matched_couples, c.wife = "Anna" ∧ c.husband = "Smith" ∧ c.husband_bottles = 4 * c.wife_bottles) ∧
    (∃ c ∈ matched_couples, c.wife = "Betty" ∧ c.husband = "White" ∧ c.husband_bottles = 3 * c.wife_bottles) ∧
    (∃ c ∈ matched_couples, c.wife = "Carol" ∧ c.husband = "Green" ∧ c.husband_bottles = 2 * c.wife_bottles) ∧
    (∃ c ∈ matched_couples, c.wife = "Dorothy" ∧ c.husband = "Brown" ∧ c.husband_bottles = c.wife_bottles) ∧
    (matched_couples.map (λ c => c.husband)).toFinset = husbands.toFinset :=
by sorry

end NUMINAMATH_CALUDE_couples_matching_l1183_118378


namespace NUMINAMATH_CALUDE_pencil_count_l1183_118357

theorem pencil_count (num_students : ℕ) (pencils_per_student : ℕ) 
  (h1 : num_students = 2) 
  (h2 : pencils_per_student = 9) : 
  num_students * pencils_per_student = 18 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1183_118357


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1183_118344

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a - 2 * i) * i = b - i → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1183_118344


namespace NUMINAMATH_CALUDE_triangle_area_l1183_118367

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ × ℝ)

-- Define the point L on EF
def L (t : Triangle) : ℝ × ℝ := sorry

-- State that DL is an altitude of triangle DEF
def is_altitude (t : Triangle) : Prop :=
  let (dx, dy) := t.D
  let (lx, ly) := L t
  (lx - dx) * (t.F.1 - t.E.1) + (ly - dy) * (t.F.2 - t.E.2) = 0

-- Define the lengths
def DE (t : Triangle) : ℝ := sorry
def EL (t : Triangle) : ℝ := sorry
def EF (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_area (t : Triangle) 
  (h1 : is_altitude t)
  (h2 : DE t = 14)
  (h3 : EL t = 9)
  (h4 : EF t = 17) :
  let area := (EF t * Real.sqrt ((DE t)^2 - (EL t)^2)) / 2
  area = (17 * Real.sqrt 115) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l1183_118367


namespace NUMINAMATH_CALUDE_safari_animal_ratio_l1183_118376

theorem safari_animal_ratio :
  let antelopes : ℕ := 80
  let rabbits : ℕ := antelopes + 34
  let hyenas : ℕ := antelopes + rabbits - 42
  let wild_dogs : ℕ := hyenas + 50
  let total_animals : ℕ := 605
  let leopards : ℕ := total_animals - (antelopes + rabbits + hyenas + wild_dogs)
  leopards * 2 = rabbits :=
by sorry

end NUMINAMATH_CALUDE_safari_animal_ratio_l1183_118376


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l1183_118393

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

theorem f_composition_negative_two : f (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l1183_118393


namespace NUMINAMATH_CALUDE_money_left_l1183_118315

def initial_amount : ℕ := 48
def num_books : ℕ := 5
def book_cost : ℕ := 2

theorem money_left : initial_amount - (num_books * book_cost) = 38 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l1183_118315


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1183_118380

/-- Given a quadratic function y = x^2 - 1840x + 2009 with roots m and n,
    prove that (m^2 - 1841m + 2009)(n^2 - 1841n + 2009) = 2009 -/
theorem quadratic_roots_property (m n : ℝ) : 
  m^2 - 1840*m + 2009 = 0 →
  n^2 - 1840*n + 2009 = 0 →
  (m^2 - 1841*m + 2009) * (n^2 - 1841*n + 2009) = 2009 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1183_118380


namespace NUMINAMATH_CALUDE_derivative_at_e_l1183_118327

open Real

theorem derivative_at_e (f : ℝ → ℝ) (h : Differentiable ℝ f) :
  (∀ x, f x = 2 * x * (deriv f e) - log x) →
  deriv f e = 1 / e :=
by sorry

end NUMINAMATH_CALUDE_derivative_at_e_l1183_118327


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l1183_118317

-- Proposition A
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → 1 / a < 1) ∧
  (∃ a, 1 / a < 1 ∧ ¬(a > 1)) :=
sorry

-- Proposition D
theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b, a ≠ 0 ∧ a * b = 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l1183_118317


namespace NUMINAMATH_CALUDE_book_price_comparison_l1183_118319

theorem book_price_comparison (price_second : ℝ) (price_first : ℝ) :
  price_first = price_second * 1.5 →
  (price_first - price_second) / price_first * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_book_price_comparison_l1183_118319


namespace NUMINAMATH_CALUDE_line_equation_through_bisecting_point_l1183_118322

/-- Given a parabola and a line with specific properties, prove the equation of the line -/
theorem line_equation_through_bisecting_point (x y : ℝ) :
  (∀ x y, y^2 = 16*x) → -- parabola equation
  (∃ x1 y1 x2 y2 : ℝ, 
    y1^2 = 16*x1 ∧ y2^2 = 16*x2 ∧ -- intersection points on parabola
    (x1 + x2)/2 = 2 ∧ (y1 + y2)/2 = 1) → -- midpoint is (2, 1)
  (8*x - y - 15 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_line_equation_through_bisecting_point_l1183_118322


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l1183_118370

theorem cube_sum_inequality (x y z : ℝ) : 
  x^3 + y^3 + z^3 + 3*x*y*z ≥ x^2*(y+z) + y^2*(z+x) + z^2*(x+y) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l1183_118370


namespace NUMINAMATH_CALUDE_parallelogram_construction_l1183_118339

-- Define a structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a structure for a circle
structure Circle where
  center : Point2D
  radius : ℝ

-- Define the problem statement
theorem parallelogram_construction (A B C : Point2D) (r : ℝ) 
  (h1 : ∃ (circle : Circle), circle.center = A ∧ circle.radius = r ∧ 
    (B.x - A.x)^2 + (B.y - A.y)^2 ≤ r^2 ∧ 
    (C.x - A.x)^2 + (C.y - A.y)^2 ≤ r^2) :
  ∃ (D : Point2D), 
    (A.x + C.x = B.x + D.x) ∧ 
    (A.y + C.y = B.y + D.y) ∧
    (A.x - B.x = D.x - C.x) ∧ 
    (A.y - B.y = D.y - C.y) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_construction_l1183_118339


namespace NUMINAMATH_CALUDE_power_equation_solution_l1183_118385

theorem power_equation_solution : ∃ k : ℕ, 3 * 2^2001 - 3 * 2^2000 - 2^1999 + 2^1998 = k * 2^1998 ∧ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1183_118385


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1183_118384

theorem sum_of_reciprocals (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 15) (h4 : a * b = 225) : 
  1 / a + 1 / b = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1183_118384


namespace NUMINAMATH_CALUDE_three_digit_to_four_digit_l1183_118321

theorem three_digit_to_four_digit (a : ℕ) (h : 100 ≤ a ∧ a ≤ 999) :
  (10 * a + 1 : ℕ) = 1000 + (a - 100) * 10 + 1 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_to_four_digit_l1183_118321


namespace NUMINAMATH_CALUDE_quadratic_solution_l1183_118355

theorem quadratic_solution (x : ℝ) (h1 : 2 * x^2 - 6 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1183_118355


namespace NUMINAMATH_CALUDE_range_of_a_l1183_118314

theorem range_of_a (a : ℝ) : 
  (∀ (x y : ℝ), x ≠ 0 → |x + 1/x| ≥ |a - 2| + Real.sin y) ↔ a ∈ Set.Icc 1 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1183_118314


namespace NUMINAMATH_CALUDE_dividend_mod_31_l1183_118345

theorem dividend_mod_31 (divisor quotient remainder dividend : ℕ) : 
  divisor = 37 → 
  quotient = 214 → 
  remainder = 12 → 
  dividend = divisor * quotient + remainder →
  dividend % 31 = 25 := by
  sorry

end NUMINAMATH_CALUDE_dividend_mod_31_l1183_118345


namespace NUMINAMATH_CALUDE_probability_two_number_cards_sum_15_l1183_118330

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of number cards (2 through 10) in each suit -/
def numberCardsPerSuit : ℕ := 9

/-- The number of suits in a standard deck -/
def numberOfSuits : ℕ := 4

/-- The total number of number cards (2 through 10) in a standard deck -/
def totalNumberCards : ℕ := numberCardsPerSuit * numberOfSuits

/-- The possible first card values that can sum to 15 with another number card -/
def validFirstCards : List ℕ := [5, 6, 7, 8, 9]

/-- The number of ways to choose two number cards that sum to 15 -/
def waysToSum15 : ℕ := validFirstCards.length * numberOfSuits

theorem probability_two_number_cards_sum_15 :
  (waysToSum15 : ℚ) / (standardDeckSize * (standardDeckSize - 1)) = 100 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_number_cards_sum_15_l1183_118330


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l1183_118354

/-- Given a point (x, y) and a line y = mx + b, 
    returns the symmetric point with respect to the line -/
def symmetricPoint (x y m b : ℝ) : ℝ × ℝ := sorry

/-- The line of symmetry y = x - 1 -/
def lineOfSymmetry : ℝ → ℝ := fun x ↦ x - 1

theorem symmetric_point_correct : 
  symmetricPoint (-1) 2 1 (-1) = (3, -2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l1183_118354


namespace NUMINAMATH_CALUDE_product_remainder_by_10_l1183_118307

theorem product_remainder_by_10 : (4219 * 2675 * 394082 * 5001) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_10_l1183_118307


namespace NUMINAMATH_CALUDE_function_inequality_l1183_118333

-- Define the condition (1-x)/f'(x) ≥ 0
def condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (1 - x) / (deriv f x) ≥ 0

-- State the theorem
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) (h : condition f) :
  f 0 + f 2 < 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1183_118333


namespace NUMINAMATH_CALUDE_cistern_filling_time_l1183_118331

theorem cistern_filling_time (x : ℝ) : 
  x > 0 ∧                            -- x is positive (time can't be negative or zero)
  (1 / x + 1 / 12 - 1 / 20 = 1 / 7.5) -- combined rate equation
  → x = 10 := by
sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l1183_118331


namespace NUMINAMATH_CALUDE_equation_solution_l1183_118359

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x + 30 / (x - 4)
  ∃ (x₁ x₂ : ℝ), (f x₁ = -8 ∧ f x₂ = -8) ∧ 
    x₁ = -2 + Real.sqrt 6 ∧ x₂ = -2 - Real.sqrt 6 ∧
    ∀ x : ℝ, f x = -8 → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1183_118359


namespace NUMINAMATH_CALUDE_octal_to_decimal_fraction_l1183_118368

theorem octal_to_decimal_fraction (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are base-10 digits
  (435 : Nat) = 4 * 8^2 + 3 * 8 + 5 →  -- 435 in octal
  285 = 200 + 10 * c + d →  -- 2cd in decimal
  (c + d) / 12 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_fraction_l1183_118368


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l1183_118375

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 6) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l1183_118375


namespace NUMINAMATH_CALUDE_fraction_equality_l1183_118304

-- Define the @ operation
def at_op (a b : ℕ) : ℕ := a * b + b^2

-- Define the # operation
def hash_op (a b : ℕ) : ℕ := a + b + a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 5 3 : ℚ) / (hash_op 5 3) = 24 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1183_118304


namespace NUMINAMATH_CALUDE_third_group_men_count_l1183_118366

/-- Represents the work rate of a person -/
structure WorkRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work rate of a group -/
def totalWorkRate (m w : WorkRate) (g : WorkGroup) : ℝ :=
  g.men * m.rate + g.women * w.rate

theorem third_group_men_count 
  (m w : WorkRate) 
  (g1 g2 : WorkGroup) 
  (h1 : totalWorkRate m w g1 = totalWorkRate m w g2)
  (h2 : g1.men = 3 ∧ g1.women = 8)
  (h3 : g2.men = 6 ∧ g2.women = 2)
  (g3 : WorkGroup)
  (h4 : g3.women = 3)
  (h5 : totalWorkRate m w g3 = 0.5 * totalWorkRate m w g1) :
  g3.men = 2 := by
sorry

end NUMINAMATH_CALUDE_third_group_men_count_l1183_118366


namespace NUMINAMATH_CALUDE_midpoint_locus_of_square_l1183_118323

/-- The locus of the midpoint of a square with side length 2a, where two consecutive vertices
    are always on the x- and y-axes respectively in the first quadrant, is a circle with
    radius a centered at the origin. -/
theorem midpoint_locus_of_square (a : ℝ) (h : a > 0) :
  ∃ (C : ℝ × ℝ), (∀ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ x^2 + y^2 = (2*a)^2 →
    C = (x/2, y/2) ∧ C.1^2 + C.2^2 = a^2) :=
sorry

end NUMINAMATH_CALUDE_midpoint_locus_of_square_l1183_118323


namespace NUMINAMATH_CALUDE_impossible_equal_tokens_l1183_118363

/-- Represents the state of tokens --/
structure TokenState where
  green : ℕ
  red : ℕ

/-- Represents a token exchange operation --/
inductive Exchange
  | GreenToRed
  | RedToGreen

/-- Applies an exchange to a token state --/
def applyExchange (state : TokenState) (ex : Exchange) : TokenState :=
  match ex with
  | Exchange.GreenToRed => 
      if state.green ≥ 1 then ⟨state.green - 1, state.red + 5⟩ else state
  | Exchange.RedToGreen => 
      if state.red ≥ 1 then ⟨state.green + 5, state.red - 1⟩ else state

/-- A sequence of exchanges --/
def ExchangeSequence := List Exchange

/-- Applies a sequence of exchanges to a token state --/
def applyExchangeSequence (state : TokenState) (seq : ExchangeSequence) : TokenState :=
  seq.foldl applyExchange state

/-- The theorem to be proved --/
theorem impossible_equal_tokens : 
  ∀ (seq : ExchangeSequence), 
  let finalState := applyExchangeSequence ⟨1, 0⟩ seq
  finalState.green ≠ finalState.red :=
sorry

end NUMINAMATH_CALUDE_impossible_equal_tokens_l1183_118363


namespace NUMINAMATH_CALUDE_system_solution_l1183_118325

theorem system_solution (x y : ℝ) : 
  (4 * x + y = 6 ∧ 3 * x - y = 1) ↔ (x = 1 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1183_118325


namespace NUMINAMATH_CALUDE_trig_calculation_l1183_118343

theorem trig_calculation : 
  (6 * (Real.tan (45 * π / 180))) - (2 * (Real.cos (60 * π / 180))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_calculation_l1183_118343


namespace NUMINAMATH_CALUDE_inscribed_circle_hypotenuse_length_l1183_118313

/-- A circle inscribed on the hypotenuse of a right triangle -/
structure InscribedCircle where
  /-- The right triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The inscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point M, tangency point of the circle with AB -/
  M : ℝ × ℝ
  /-- Point N, intersection of the circle with AC -/
  N : ℝ × ℝ
  /-- Center of the circle -/
  O : ℝ × ℝ
  /-- The triangle is right-angled at B -/
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  /-- The circle touches AB at M -/
  touches_AB : M ∈ circle ∩ {p | (p.1 - A.1) * (B.2 - A.2) = (p.2 - A.2) * (B.1 - A.1)}
  /-- The circle touches BC -/
  touches_BC : ∃ p ∈ circle, (p.1 - B.1) * (C.2 - B.2) = (p.2 - B.2) * (C.1 - B.1)
  /-- The circle lies on AC -/
  on_AC : O ∈ {p | (p.1 - A.1) * (C.2 - A.2) = (p.2 - A.2) * (C.1 - A.1)}
  /-- AM = 20/9 -/
  AM_length : Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 20/9
  /-- AN:MN = 6:1 -/
  AN_MN_ratio : Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2) / Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = 6

/-- The main theorem -/
theorem inscribed_circle_hypotenuse_length (ic : InscribedCircle) :
  Real.sqrt ((ic.C.1 - ic.A.1)^2 + (ic.C.2 - ic.A.2)^2) = Real.sqrt 5 + 1/4 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_hypotenuse_length_l1183_118313


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l1183_118337

theorem cubic_factorization_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 1001 * x^3 - 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 3458 :=
by sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l1183_118337


namespace NUMINAMATH_CALUDE_no_reciprocal_roots_l1183_118329

theorem no_reciprocal_roots (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬∃ (n : ℕ), a * (1 / n : ℚ)^2 + b * (1 / n : ℚ) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_reciprocal_roots_l1183_118329


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1183_118338

theorem digit_sum_problem (a b c d : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧  -- Digits are less than 10
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- All digits are different
  100 * a + 10 * b + c + 100 * d + 10 * c + a = 1100  -- The equation
  → a + b + c + d = 19 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1183_118338


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l1183_118388

theorem rowing_time_ratio (man_speed stream_speed : ℝ) 
  (h1 : man_speed = 36)
  (h2 : stream_speed = 18) :
  (man_speed - stream_speed) / (man_speed + stream_speed) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_ratio_l1183_118388


namespace NUMINAMATH_CALUDE_remainder_plus_3255_l1183_118332

theorem remainder_plus_3255 (n : ℤ) (h : n % 5 = 2) : (n + 3255) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_plus_3255_l1183_118332


namespace NUMINAMATH_CALUDE_line_through_point_with_angle_l1183_118326

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parametric line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Checks if a point lies on a parametric line -/
def pointOnLine (p : Point) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = p.x ∧ l.y t = p.y

/-- Calculates the angle between a parametric line and the positive x-axis -/
noncomputable def lineAngle (l : ParametricLine) : ℝ :=
  Real.arctan ((l.y 1 - l.y 0) / (l.x 1 - l.x 0))

theorem line_through_point_with_angle (M : Point) (θ : ℝ) :
  let l : ParametricLine := {
    x := λ t => 1 + (1/2) * t,
    y := λ t => 5 + (Real.sqrt 3 / 2) * t
  }
  pointOnLine M l ∧ lineAngle l = θ ∧ M.x = 1 ∧ M.y = 5 ∧ θ = π/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_with_angle_l1183_118326


namespace NUMINAMATH_CALUDE_sqrt_88200_simplification_l1183_118386

theorem sqrt_88200_simplification : Real.sqrt 88200 = 882 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_88200_simplification_l1183_118386


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1183_118346

-- Define a random variable following normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define probability function
noncomputable def probability {α : Type} (event : Set α) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_probability 
  (ξ : normal_distribution 1 σ) 
  (h1 : probability {x | x < 1} = 1/2) 
  (h2 : probability {x | x > 2} = p) :
  probability {x | 0 < x ∧ x < 1} = 1/2 - p :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1183_118346


namespace NUMINAMATH_CALUDE_correct_distinct_arrangements_l1183_118318

/-- The number of distinct arrangements to distribute 5 students into two dormitories,
    with each dormitory accommodating at least 2 students. -/
def distinct_arrangements : ℕ := 20

/-- The total number of students to be distributed. -/
def total_students : ℕ := 5

/-- The number of dormitories. -/
def num_dormitories : ℕ := 2

/-- The minimum number of students that must be in each dormitory. -/
def min_students_per_dormitory : ℕ := 2

/-- Theorem stating that the number of distinct arrangements is correct. -/
theorem correct_distinct_arrangements :
  distinct_arrangements = 20 ∧
  total_students = 5 ∧
  num_dormitories = 2 ∧
  min_students_per_dormitory = 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_distinct_arrangements_l1183_118318


namespace NUMINAMATH_CALUDE_combined_time_calculation_l1183_118320

/-- The time taken by the car to reach station B -/
def car_time : ℝ := 4.5

/-- The additional time taken by the train compared to the car -/
def train_additional_time : ℝ := 2

/-- The time taken by the train to reach station B -/
def train_time : ℝ := car_time + train_additional_time

/-- The combined time taken by both the car and the train to reach station B -/
def combined_time : ℝ := car_time + train_time

theorem combined_time_calculation : combined_time = 11 := by sorry

end NUMINAMATH_CALUDE_combined_time_calculation_l1183_118320


namespace NUMINAMATH_CALUDE_angle_after_rotation_l1183_118362

def rotation_result (initial_angle rotation : ℕ) : ℕ :=
  (rotation - initial_angle) % 360

theorem angle_after_rotation (initial_angle : ℕ) (h1 : initial_angle = 70) (rotation : ℕ) (h2 : rotation = 960) :
  rotation_result initial_angle rotation = 170 := by
  sorry

end NUMINAMATH_CALUDE_angle_after_rotation_l1183_118362
