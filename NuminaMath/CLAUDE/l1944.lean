import Mathlib

namespace NUMINAMATH_CALUDE_intersection_sum_l1944_194411

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y - 5)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧
    (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 10 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l1944_194411


namespace NUMINAMATH_CALUDE_cyclist_speed_l1944_194483

-- Define the parameters of the problem
def first_distance : ℝ := 8
def second_distance : ℝ := 10
def second_speed : ℝ := 8
def total_average_speed : ℝ := 8.78

-- Define the theorem
theorem cyclist_speed (v : ℝ) (h : v > 0) :
  (first_distance + second_distance) / ((first_distance / v) + (second_distance / second_speed)) = total_average_speed →
  v = 10 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_l1944_194483


namespace NUMINAMATH_CALUDE_fishing_problem_solution_l1944_194400

/-- Represents the fishing problem scenario -/
structure FishingProblem where
  totalCatch : ℝ
  plannedDays : ℝ
  dailyCatch : ℝ
  stormDuration : ℝ
  stormCatchReduction : ℝ
  normalCatchIncrease : ℝ
  daysAheadOfSchedule : ℝ

/-- Theorem stating the solution to the fishing problem -/
theorem fishing_problem_solution (p : FishingProblem) 
  (h1 : p.totalCatch = 1800)
  (h2 : p.stormDuration = p.plannedDays / 3)
  (h3 : p.stormCatchReduction = 20)
  (h4 : p.normalCatchIncrease = 20)
  (h5 : p.daysAheadOfSchedule = 1)
  (h6 : p.plannedDays * p.dailyCatch = p.totalCatch)
  (h7 : p.stormDuration * (p.dailyCatch - p.stormCatchReduction) + 
        (p.plannedDays - p.stormDuration - p.daysAheadOfSchedule) * 
        (p.dailyCatch + p.normalCatchIncrease) = p.totalCatch) :
  p.dailyCatch = 100 := by
  sorry


end NUMINAMATH_CALUDE_fishing_problem_solution_l1944_194400


namespace NUMINAMATH_CALUDE_parallel_condition_l1944_194447

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ * c₂ ≠ m₂ * c₁

/-- The theorem stating that a=1 is a necessary and sufficient condition for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  parallel a 2 (-1) 1 2 4 ↔ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_condition_l1944_194447


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1944_194434

-- Problem 1
theorem equation_one_solution (x : ℝ) : 
  x / (2 * x - 5) + 5 / (5 - 2 * x) = 1 ↔ x = 0 :=
sorry

-- Problem 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℝ), (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1944_194434


namespace NUMINAMATH_CALUDE_solve_lollipop_problem_l1944_194485

def lollipop_problem (alison henry diane emily : ℕ) (daily_rate : ℝ) : Prop :=
  henry = alison + 30 ∧
  alison = 60 ∧
  alison * 2 = diane ∧
  emily = 50 ∧
  emily + 10 = diane ∧
  daily_rate = 1.5 ∧
  ∃ (days : ℕ), days = 4 ∧
    (let total := alison + henry + diane + emily
     let first_day := 45
     let rec consumed (n : ℕ) : ℝ :=
       if n = 0 then 0
       else if n = 1 then first_day
       else consumed (n - 1) * daily_rate
     consumed days > total ∧ consumed (days - 1) ≤ total)

theorem solve_lollipop_problem :
  ∃ (alison henry diane emily : ℕ) (daily_rate : ℝ),
    lollipop_problem alison henry diane emily daily_rate :=
by
  sorry

end NUMINAMATH_CALUDE_solve_lollipop_problem_l1944_194485


namespace NUMINAMATH_CALUDE_train_bus_cost_difference_l1944_194419

/-- The cost difference between a train ride and a bus ride -/
def cost_difference (train_cost bus_cost : ℝ) : ℝ := train_cost - bus_cost

/-- Theorem stating the cost difference between a train ride and a bus ride -/
theorem train_bus_cost_difference :
  ∀ (train_cost bus_cost : ℝ),
    train_cost > bus_cost →
    train_cost + bus_cost = 9.85 →
    bus_cost = 1.75 →
    cost_difference train_cost bus_cost = 6.35 := by
  sorry

end NUMINAMATH_CALUDE_train_bus_cost_difference_l1944_194419


namespace NUMINAMATH_CALUDE_gerald_wood_pieces_l1944_194417

/-- The number of pieces of wood needed to make a table -/
def wood_per_table : ℕ := 12

/-- The number of pieces of wood needed to make a chair -/
def wood_per_chair : ℕ := 8

/-- The number of chairs Gerald can make -/
def chairs : ℕ := 48

/-- The number of tables Gerald can make -/
def tables : ℕ := 24

/-- Theorem stating the total number of wood pieces Gerald has -/
theorem gerald_wood_pieces : 
  wood_per_table * tables + wood_per_chair * chairs = 672 := by
  sorry

end NUMINAMATH_CALUDE_gerald_wood_pieces_l1944_194417


namespace NUMINAMATH_CALUDE_difference_median_mode_l1944_194445

def data : List ℕ := [36, 37, 37, 38, 40, 40, 40, 41, 42, 43, 54, 55, 57, 59, 61, 61, 65, 68, 69]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℝ := sorry

theorem difference_median_mode : 
  |median data - mode data| = 2 := by sorry

end NUMINAMATH_CALUDE_difference_median_mode_l1944_194445


namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l1944_194430

theorem cricket_bat_selling_price 
  (profit : ℝ) 
  (profit_percentage : ℝ) 
  (selling_price : ℝ) : 
  profit = 300 → 
  profit_percentage = 50 → 
  selling_price = profit * (100 / profit_percentage) + profit → 
  selling_price = 900 := by
sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l1944_194430


namespace NUMINAMATH_CALUDE_point_outside_circle_l1944_194492

theorem point_outside_circle (a b : ℝ) 
  (h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    a * x₁ + b * y₁ = 1 ∧ 
    a * x₂ + b * y₂ = 1 ∧ 
    x₁^2 + y₁^2 = 1 ∧ 
    x₂^2 + y₂^2 = 1) : 
  a^2 + b^2 > 1 := by
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1944_194492


namespace NUMINAMATH_CALUDE_profit_percent_from_cost_price_ratio_l1944_194413

/-- Profit percent calculation given cost price as a percentage of selling price -/
theorem profit_percent_from_cost_price_ratio (selling_price : ℝ) (cost_price_ratio : ℝ) 
  (h : cost_price_ratio = 0.8) : 
  (selling_price - cost_price_ratio * selling_price) / (cost_price_ratio * selling_price) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_from_cost_price_ratio_l1944_194413


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1944_194432

/-- A function f : ℝ → ℝ is quadratic if there exist real numbers a, b, and c
    with a ≠ 0 such that f(x) = ax^2 + bx + c for all x ∈ ℝ. -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = (x + 1)^2 - 5 -/
def f (x : ℝ) : ℝ := (x + 1)^2 - 5

/-- Theorem: The function f(x) = (x + 1)^2 - 5 is a quadratic function -/
theorem f_is_quadratic : IsQuadratic f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1944_194432


namespace NUMINAMATH_CALUDE_point_A_on_curve_l1944_194449

/-- The equation of curve C is x^2 + x + y - 1 = 0 -/
def curve_equation (x y : ℝ) : Prop := x^2 + x + y - 1 = 0

/-- Point A has coordinates (0, 1) -/
def point_A : ℝ × ℝ := (0, 1)

/-- Theorem: Point A lies on curve C -/
theorem point_A_on_curve : curve_equation point_A.1 point_A.2 := by sorry

end NUMINAMATH_CALUDE_point_A_on_curve_l1944_194449


namespace NUMINAMATH_CALUDE_kite_area_is_40_l1944_194435

/-- A point in 2D space represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- A kite defined by its four vertices -/
structure Kite where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculate the area of a kite given its vertices -/
def kiteArea (k : Kite) : ℝ := sorry

/-- The specific kite from the problem -/
def problemKite : Kite := {
  v1 := { x := 0, y := 6 }
  v2 := { x := 4, y := 10 }
  v3 := { x := 8, y := 6 }
  v4 := { x := 4, y := 0 }
}

theorem kite_area_is_40 : kiteArea problemKite = 40 := by sorry

end NUMINAMATH_CALUDE_kite_area_is_40_l1944_194435


namespace NUMINAMATH_CALUDE_f_negative_nine_halves_l1944_194444

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def periodic_2 (f : ℝ → ℝ) := ∀ x, f (x + 2) = f x

def f_on_unit_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

theorem f_negative_nine_halves 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : periodic_2 f) 
  (h_unit_interval : f_on_unit_interval f) : 
  f (-9/2) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_f_negative_nine_halves_l1944_194444


namespace NUMINAMATH_CALUDE_line_x_intercept_l1944_194429

/-- Given a line with slope 3/4 passing through (-12, -39), prove its x-intercept is 40 -/
theorem line_x_intercept :
  let m : ℚ := 3/4  -- slope
  let x₀ : ℤ := -12
  let y₀ : ℤ := -39
  let b : ℚ := y₀ - m * x₀  -- y-intercept
  let x_intercept : ℚ := -b / m  -- x-coordinate where y = 0
  x_intercept = 40 := by
sorry

end NUMINAMATH_CALUDE_line_x_intercept_l1944_194429


namespace NUMINAMATH_CALUDE_valentines_calculation_l1944_194424

/-- The number of Valentines Mrs. Franklin initially had -/
def initial_valentines : ℕ := 58

/-- The number of Valentines Mrs. Franklin gave to her students -/
def given_valentines : ℕ := 42

/-- The number of Valentines Mrs. Franklin has now -/
def remaining_valentines : ℕ := initial_valentines - given_valentines

theorem valentines_calculation : remaining_valentines = 16 := by
  sorry

end NUMINAMATH_CALUDE_valentines_calculation_l1944_194424


namespace NUMINAMATH_CALUDE_whittlesworth_band_size_l1944_194463

theorem whittlesworth_band_size (n : ℕ) : 
  (20 * n % 28 = 6) →
  (20 * n % 19 = 5) →
  (20 * n < 1200) →
  (∀ m : ℕ, (20 * m % 28 = 6) → (20 * m % 19 = 5) → (20 * m < 1200) → m ≤ n) →
  20 * n = 2000 :=
by sorry

end NUMINAMATH_CALUDE_whittlesworth_band_size_l1944_194463


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l1944_194451

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {m₁ m₂ : ℝ} : 
  (m₁ = m₂) ↔ (∀ (x y : ℝ), m₁ * x + y = 0 ↔ m₂ * x + y = 0)

/-- The slope of a line ax + by = c is -a/b -/
axiom slope_of_line {a b c : ℝ} (hb : b ≠ 0) : 
  ∀ (x y : ℝ), a * x + b * y = c → -a/b * x + y = c/b

theorem parallel_lines_condition (m : ℝ) :
  (∀ (x y : ℝ), (m - 1) * x + y = 4 * m - 1 ↔ 2 * x - 3 * y = 5) ↔ m = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l1944_194451


namespace NUMINAMATH_CALUDE_roots_differ_by_one_l1944_194427

theorem roots_differ_by_one (a : ℝ) : 
  (∃ x y : ℝ, x^2 - a*x + 1 = 0 ∧ y^2 - a*y + 1 = 0 ∧ y - x = 1) → a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_differ_by_one_l1944_194427


namespace NUMINAMATH_CALUDE_angle_B_measure_l1944_194499

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b*cos(C) + (2a+c)*cos(B) = 0, then the measure of angle B is 2π/3 -/
theorem angle_B_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b * Real.cos C + (2 * a + c) * Real.cos B = 0 →
  B = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l1944_194499


namespace NUMINAMATH_CALUDE_function_inequality_range_l1944_194455

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x - a| + |x + 3*a - 2|
def g (a x : ℝ) : ℝ := -x^2 + 2*a*x + 1

-- State the theorem
theorem function_inequality_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, f a x₁ > g a x₂) ↔ 
  (a ∈ Set.Ioo (-2 - Real.sqrt 5) (-2 + Real.sqrt 5) ∪ Set.Ioo 1 3) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_range_l1944_194455


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1944_194466

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + 2 * x + y = 8) :
  x + y ≥ 2 * Real.sqrt 10 - 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ + 2 * x₀ + y₀ = 8 ∧ x₀ + y₀ = 2 * Real.sqrt 10 - 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1944_194466


namespace NUMINAMATH_CALUDE_triangle_area_l1944_194439

/-- The area of a triangle with sides 9, 40, and 41 is 180 square units. -/
theorem triangle_area : ∀ (a b c : ℝ), a = 9 ∧ b = 40 ∧ c = 41 →
  (a * a + b * b = c * c) → (1/2 : ℝ) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1944_194439


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1944_194472

theorem rectangular_plot_breadth (length width area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 2028 → 
  width = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1944_194472


namespace NUMINAMATH_CALUDE_total_flowers_l1944_194416

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) (h1 : num_pots = 141) (h2 : flowers_per_pot = 71) : 
  num_pots * flowers_per_pot = 10011 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l1944_194416


namespace NUMINAMATH_CALUDE_whole_number_between_fractions_l1944_194496

theorem whole_number_between_fractions (N : ℤ) :
  (3.5 < (N : ℚ) / 5 ∧ (N : ℚ) / 5 < 4.5) ↔ (N = 18 ∨ N = 19 ∨ N = 20 ∨ N = 21 ∨ N = 22) :=
by sorry

end NUMINAMATH_CALUDE_whole_number_between_fractions_l1944_194496


namespace NUMINAMATH_CALUDE_abcdef_hex_bits_l1944_194422

def hex_to_decimal (hex : String) : ℕ :=
  -- Convert hexadecimal string to decimal
  sorry

def bits_required (n : ℕ) : ℕ :=
  -- Calculate the number of bits required to represent n
  sorry

theorem abcdef_hex_bits :
  bits_required (hex_to_decimal "ABCDEF") = 24 := by
  sorry

end NUMINAMATH_CALUDE_abcdef_hex_bits_l1944_194422


namespace NUMINAMATH_CALUDE_relatively_prime_dates_february_leap_year_count_l1944_194471

/-- The number of days in February during a leap year -/
def leap_year_february_days : ℕ := 29

/-- The month number for February -/
def february_number : ℕ := 2

/-- A function that returns the number of relatively prime dates in February of a leap year -/
def relatively_prime_dates_february_leap_year : ℕ := 
  leap_year_february_days - (leap_year_february_days / february_number)

/-- Theorem stating that the number of relatively prime dates in February of a leap year is 15 -/
theorem relatively_prime_dates_february_leap_year_count : 
  relatively_prime_dates_february_leap_year = 15 := by sorry

end NUMINAMATH_CALUDE_relatively_prime_dates_february_leap_year_count_l1944_194471


namespace NUMINAMATH_CALUDE_fifteen_percent_of_thousand_is_150_l1944_194433

theorem fifteen_percent_of_thousand_is_150 :
  (15 / 100) * 1000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_thousand_is_150_l1944_194433


namespace NUMINAMATH_CALUDE_popsicle_stick_sum_l1944_194420

theorem popsicle_stick_sum : 
  ∀ (gino ana sam speaker : ℕ),
    gino = 63 →
    ana = 128 →
    sam = 75 →
    speaker = 50 →
    gino + ana + sam + speaker = 316 :=
by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_sum_l1944_194420


namespace NUMINAMATH_CALUDE_yard_length_l1944_194414

theorem yard_length (n : ℕ) (d : ℝ) (h1 : n = 26) (h2 : d = 14) : 
  (n - 1) * d = 350 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_l1944_194414


namespace NUMINAMATH_CALUDE_incorrect_statement_is_false_l1944_194465

/-- Represents the method used for separation and counting of bacteria. -/
inductive SeparationMethod
| DilutionPlating
| StreakPlate

/-- Represents a biotechnology practice. -/
structure BiotechPractice where
  soil_bacteria_method : SeparationMethod
  fruit_vinegar_air : Bool
  nitrite_detection : Bool
  dna_extraction : Bool

/-- The correct biotechnology practices. -/
def correct_practices : BiotechPractice := {
  soil_bacteria_method := SeparationMethod.DilutionPlating,
  fruit_vinegar_air := true,
  nitrite_detection := true,
  dna_extraction := true
}

/-- The statement to be proven false. -/
def incorrect_statement : BiotechPractice := {
  soil_bacteria_method := SeparationMethod.StreakPlate,
  fruit_vinegar_air := true,
  nitrite_detection := true,
  dna_extraction := true
}

/-- Theorem stating that the incorrect statement is indeed incorrect. -/
theorem incorrect_statement_is_false : incorrect_statement ≠ correct_practices := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_is_false_l1944_194465


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1944_194415

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ (x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1944_194415


namespace NUMINAMATH_CALUDE_smaller_integer_is_49_l1944_194493

theorem smaller_integer_is_49 (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧  -- m is a 2-digit positive integer
  10 ≤ n ∧ n < 100 ∧  -- n is a 2-digit positive integer
  ∃ k : ℕ, n = 25 * k ∧ -- n is a multiple of 25
  m < n ∧  -- n is larger than m
  (m + n) / 2 = m + n / 100  -- their average equals the decimal number
  → m = 49 := by sorry

end NUMINAMATH_CALUDE_smaller_integer_is_49_l1944_194493


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1944_194454

/-- Two vectors in ℝ² are parallel if the ratio of their components is constant -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

/-- The sum of two vectors in ℝ² -/
def vec_sum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

theorem parallel_vectors_sum :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → vec_sum a b = (-2, -1) := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1944_194454


namespace NUMINAMATH_CALUDE_current_gas_in_car_l1944_194438

/-- Represents the fuel efficiency of a car in miles per gallon -/
def fuel_efficiency : ℝ := 20

/-- Represents the total distance to be traveled in miles -/
def total_distance : ℝ := 1200

/-- Represents the additional gallons of gas needed for the trip -/
def additional_gas_needed : ℝ := 52

/-- Theorem stating the current amount of gas in the car -/
theorem current_gas_in_car : 
  (total_distance / fuel_efficiency) - additional_gas_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_current_gas_in_car_l1944_194438


namespace NUMINAMATH_CALUDE_polygon_contains_half_unit_segment_l1944_194408

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon
  area : ℝ
  isConvex : Bool

/-- A square with side length 1 -/
structure UnitSquare where
  -- Add necessary fields for a unit square

/-- Represents the placement of a polygon inside a square -/
structure PolygonInSquare where
  polygon : ConvexPolygon
  square : UnitSquare
  isInside : Bool

/-- A line segment -/
structure LineSegment where
  length : ℝ
  isParallelToSquareSide : Bool
  isInsidePolygon : Bool

/-- The main theorem -/
theorem polygon_contains_half_unit_segment 
  (p : PolygonInSquare) 
  (h1 : p.polygon.area > 0.5) 
  (h2 : p.polygon.isConvex) 
  (h3 : p.isInside) :
  ∃ (s : LineSegment), s.length = 0.5 ∧ s.isParallelToSquareSide ∧ s.isInsidePolygon :=
by sorry

end NUMINAMATH_CALUDE_polygon_contains_half_unit_segment_l1944_194408


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1944_194495

-- Define the quadratic function
def f (x : ℝ) := 2 * x^2 + 4 * x - 6

-- Define the solution set
def solution_set := {x : ℝ | f x < 0}

-- State the theorem
theorem quadratic_inequality_solution :
  solution_set = Set.Ioo (-3 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1944_194495


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l1944_194479

/-- The set of possible slopes for a line with y-intercept (0, -3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (2/110) ∨ m ≥ Real.sqrt (2/110)}

/-- Theorem stating the possible slopes of the line -/
theorem line_ellipse_intersection_slopes :
  ∀ (m : ℝ), (∃ (x y : ℝ), 4*x^2 + 25*y^2 = 100 ∧ y = m*x - 3) ↔ m ∈ possible_slopes := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l1944_194479


namespace NUMINAMATH_CALUDE_supply_duration_with_three_leaks_l1944_194403

/-- Represents a water tank with its supply duration and leak information -/
structure WaterTank where
  normalDuration : ℕ  -- Duration in days without leaks
  singleLeakDuration : ℕ  -- Duration in days with a single leak
  singleLeakRate : ℕ  -- Rate of the single leak in liters per day
  leakRates : List ℕ  -- List of leak rates for multiple leaks

/-- Calculates the duration of water supply given multiple leaks -/
def supplyDurationWithLeaks (tank : WaterTank) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the correct supply duration for the given scenario -/
theorem supply_duration_with_three_leaks 
  (tank : WaterTank) 
  (h1 : tank.normalDuration = 60)
  (h2 : tank.singleLeakDuration = 45)
  (h3 : tank.singleLeakRate = 10)
  (h4 : tank.leakRates = [10, 15, 20]) :
  supplyDurationWithLeaks tank = 24 := by
  sorry

end NUMINAMATH_CALUDE_supply_duration_with_three_leaks_l1944_194403


namespace NUMINAMATH_CALUDE_fraction_equality_l1944_194481

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 15) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1944_194481


namespace NUMINAMATH_CALUDE_isosceles_triangle_cosine_l1944_194450

/-- Theorem: In an isosceles triangle with two sides of length 3 and the third side of length √15 - √3,
    the cosine of the angle opposite the third side is equal to √5/3. -/
theorem isosceles_triangle_cosine (a b c : ℝ) (h1 : a = 3) (h2 : b = 3) (h3 : c = Real.sqrt 15 - Real.sqrt 3) :
  let cosC := (a^2 + b^2 - c^2) / (2 * a * b)
  cosC = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_cosine_l1944_194450


namespace NUMINAMATH_CALUDE_orange_profit_theorem_l1944_194410

/-- Represents the orange selling scenario --/
structure OrangeSelling where
  buy_price : ℚ  -- Price to buy 4 oranges in cents
  sell_price : ℚ  -- Price to sell 7 oranges in cents
  free_oranges : ℕ  -- Number of free oranges per 8 bought
  target_profit : ℚ  -- Target profit in cents
  oranges_to_sell : ℕ  -- Number of oranges to sell

/-- Calculates the profit from selling oranges --/
def calculate_profit (scenario : OrangeSelling) : ℚ :=
  let cost_per_9 := scenario.buy_price * 2  -- Cost for 8 bought + 1 free
  let cost_per_orange := cost_per_9 / 9
  let revenue_per_orange := scenario.sell_price / 7
  let profit_per_orange := revenue_per_orange - cost_per_orange
  profit_per_orange * scenario.oranges_to_sell

/-- Theorem: Selling 120 oranges results in a profit of at least 200 cents --/
theorem orange_profit_theorem (scenario : OrangeSelling) 
  (h1 : scenario.buy_price = 15)
  (h2 : scenario.sell_price = 35)
  (h3 : scenario.free_oranges = 1)
  (h4 : scenario.target_profit = 200)
  (h5 : scenario.oranges_to_sell = 120) :
  calculate_profit scenario ≥ scenario.target_profit :=
sorry

end NUMINAMATH_CALUDE_orange_profit_theorem_l1944_194410


namespace NUMINAMATH_CALUDE_function_inequality_l1944_194475

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition f''(x) < f(x) for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv (deriv f)) x < f x)

-- State the theorem
theorem function_inequality (f : ℝ → ℝ) (h : ∀ x : ℝ, (deriv (deriv f)) x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2001 < Real.exp 2001 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1944_194475


namespace NUMINAMATH_CALUDE_base10_512_to_base5_l1944_194473

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

theorem base10_512_to_base5 :
  toBase5 512 = [4, 0, 2, 2] :=
sorry

end NUMINAMATH_CALUDE_base10_512_to_base5_l1944_194473


namespace NUMINAMATH_CALUDE_initial_snatch_weight_l1944_194488

/-- Represents John's weightlifting progress --/
structure Weightlifter where
  initialCleanAndJerk : ℝ
  initialSnatch : ℝ
  newCleanAndJerk : ℝ
  newSnatch : ℝ
  newTotal : ℝ

/-- Theorem stating that given the conditions, John's initial Snatch weight was 50 kg --/
theorem initial_snatch_weight (john : Weightlifter) :
  john.initialCleanAndJerk = 80 ∧
  john.newCleanAndJerk = 2 * john.initialCleanAndJerk ∧
  john.newSnatch = 1.8 * john.initialSnatch ∧
  john.newTotal = 250 ∧
  john.newTotal = john.newCleanAndJerk + john.newSnatch →
  john.initialSnatch = 50 := by
  sorry

#check initial_snatch_weight

end NUMINAMATH_CALUDE_initial_snatch_weight_l1944_194488


namespace NUMINAMATH_CALUDE_zero_overtime_accidents_l1944_194409

/-- Represents the linear relationship between overtime hours and accidents -/
structure AccidentModel where
  slope : ℝ
  intercept : ℝ

/-- Calculates the expected number of accidents for a given number of overtime hours -/
def expected_accidents (model : AccidentModel) (hours : ℝ) : ℝ :=
  model.slope * hours + model.intercept

/-- Theorem stating the expected number of accidents when no overtime is logged -/
theorem zero_overtime_accidents 
  (model : AccidentModel)
  (h1 : expected_accidents model 1000 = 8)
  (h2 : expected_accidents model 400 = 5) :
  expected_accidents model 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_zero_overtime_accidents_l1944_194409


namespace NUMINAMATH_CALUDE_train_crossing_time_l1944_194418

/-- The time taken for a train to cross a man running in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 450 ∧ 
  train_speed = 60 * (1000 / 3600) ∧ 
  man_speed = 6 * (1000 / 3600) → 
  train_length / (train_speed - man_speed) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1944_194418


namespace NUMINAMATH_CALUDE_disjunction_true_l1944_194467

theorem disjunction_true (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_l1944_194467


namespace NUMINAMATH_CALUDE_gcd_problem_l1944_194470

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2142 * k) : 
  Nat.gcd (Int.natAbs (b^2 + 11*b + 28)) (Int.natAbs (b + 6)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1944_194470


namespace NUMINAMATH_CALUDE_max_points_2079_l1944_194441

def points (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem max_points_2079 :
  ∀ x : ℕ, 2017 ≤ x → x ≤ 2117 → points x ≤ points 2079 :=
by
  sorry

end NUMINAMATH_CALUDE_max_points_2079_l1944_194441


namespace NUMINAMATH_CALUDE_solution_and_minimum_value_l1944_194494

def A (a : ℕ) : Set ℝ := {x : ℝ | |x - 2| < a}

theorem solution_and_minimum_value (a : ℕ) (h1 : a > 0) 
  (h2 : (3/2 : ℝ) ∈ A a) (h3 : (1/2 : ℝ) ∉ A a) :
  (a = 1) ∧ 
  (∀ x : ℝ, |x + a| + |x - 2| ≥ 3) ∧ 
  (∃ x : ℝ, |x + a| + |x - 2| = 3) := by
sorry

end NUMINAMATH_CALUDE_solution_and_minimum_value_l1944_194494


namespace NUMINAMATH_CALUDE_james_truck_trip_distance_l1944_194459

/-- 
Given:
- James gets paid $0.50 per mile to drive a truck.
- Gas costs $4.00 per gallon.
- The truck gets 20 miles per gallon.
- James made a profit of $180 from a trip.

Prove: The length of the trip was 600 miles.
-/
theorem james_truck_trip_distance : 
  let pay_rate : ℝ := 0.50  -- pay rate in dollars per mile
  let gas_price : ℝ := 4.00  -- gas price in dollars per gallon
  let fuel_efficiency : ℝ := 20  -- miles per gallon
  let profit : ℝ := 180  -- profit in dollars
  ∃ distance : ℝ, 
    distance * pay_rate - (distance / fuel_efficiency) * gas_price = profit ∧ 
    distance = 600 := by
  sorry

end NUMINAMATH_CALUDE_james_truck_trip_distance_l1944_194459


namespace NUMINAMATH_CALUDE_range_of_m_l1944_194402

/-- The quadratic function p(x) = x^2 + 2x - m -/
def p (m : ℝ) (x : ℝ) : Prop := x^2 + 2*x - m > 0

/-- Given p(x): x^2 + 2x - m > 0, if p(1) is false and p(2) is true, 
    then the range of values for m is [3, 8) -/
theorem range_of_m (m : ℝ) : 
  (¬ p m 1) ∧ (p m 2) → 3 ≤ m ∧ m < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1944_194402


namespace NUMINAMATH_CALUDE_leo_weight_l1944_194406

/-- Given the weights of Leo (L), Kendra (K), and Ethan (E) satisfying the following conditions:
    1. L + K + E = 210
    2. L + 10 = 1.5K
    3. L + 10 = 0.75E
    We prove that Leo's weight (L) is approximately 63.33 pounds. -/
theorem leo_weight (L K E : ℝ) 
    (h1 : L + K + E = 210)
    (h2 : L + 10 = 1.5 * K)
    (h3 : L + 10 = 0.75 * E) : 
    ∃ ε > 0, |L - 63.33| < ε := by
  sorry

end NUMINAMATH_CALUDE_leo_weight_l1944_194406


namespace NUMINAMATH_CALUDE_total_molecular_weight_l1944_194443

-- Define molecular weights of elements
def mw_C : ℝ := 12.01
def mw_H : ℝ := 1.008
def mw_O : ℝ := 16.00
def mw_Na : ℝ := 22.99

-- Define composition of compounds
def acetic_acid_C : ℕ := 2
def acetic_acid_H : ℕ := 4
def acetic_acid_O : ℕ := 2

def sodium_hydroxide_Na : ℕ := 1
def sodium_hydroxide_O : ℕ := 1
def sodium_hydroxide_H : ℕ := 1

-- Define number of moles
def moles_acetic_acid : ℝ := 7
def moles_sodium_hydroxide : ℝ := 10

-- Theorem statement
theorem total_molecular_weight :
  let mw_acetic_acid := acetic_acid_C * mw_C + acetic_acid_H * mw_H + acetic_acid_O * mw_O
  let mw_sodium_hydroxide := sodium_hydroxide_Na * mw_Na + sodium_hydroxide_O * mw_O + sodium_hydroxide_H * mw_H
  let total_weight := moles_acetic_acid * mw_acetic_acid + moles_sodium_hydroxide * mw_sodium_hydroxide
  total_weight = 820.344 := by
  sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l1944_194443


namespace NUMINAMATH_CALUDE_previous_average_production_l1944_194456

theorem previous_average_production (n : ℕ) (today_production : ℕ) (new_average : ℚ) :
  n = 9 →
  today_production = 100 →
  new_average = 55 →
  let previous_total := n * (((n + 1) : ℚ) * new_average - today_production) / n
  previous_total / n = 50 := by sorry

end NUMINAMATH_CALUDE_previous_average_production_l1944_194456


namespace NUMINAMATH_CALUDE_exponential_function_extrema_l1944_194462

theorem exponential_function_extrema (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  let max_val := max (f 1) (f 2)
  let min_val := min (f 1) (f 2)
  max_val + min_val = 12 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_extrema_l1944_194462


namespace NUMINAMATH_CALUDE_line_segment_intersection_condition_l1944_194486

/-- A line in 2D space defined by the equation ax + y + 2 = 0 -/
structure Line2D where
  a : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Checks if a line intersects a line segment -/
def intersects (l : Line2D) (p q : Point2D) : Prop :=
  sorry

/-- The theorem to be proved -/
theorem line_segment_intersection_condition (l : Line2D) (p q : Point2D) :
  p = Point2D.mk (-2) 1 →
  q = Point2D.mk 3 2 →
  intersects l p q →
  l.a ∈ Set.Ici (3/2) ∪ Set.Iic (-4/3) :=
sorry

end NUMINAMATH_CALUDE_line_segment_intersection_condition_l1944_194486


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l1944_194476

theorem sqrt_inequality_solution_set (x : ℝ) :
  Real.sqrt (2 * x + 2) > x - 1 ↔ -1 ≤ x ∧ x ≤ 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l1944_194476


namespace NUMINAMATH_CALUDE_intersection_characterization_l1944_194484

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def matches_x_squared_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2

def has_two_distinct_intersections (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = x₁ + a ∧ f x₂ = x₂ + a

theorem intersection_characterization (f : ℝ → ℝ) (a : ℝ) :
  is_even_function f ∧ has_period_two f ∧ matches_x_squared_on_unit_interval f →
  has_two_distinct_intersections f a ↔ ∃ n : ℤ, a = 2 * n ∨ a = 2 * n - 1/4 :=
sorry

end NUMINAMATH_CALUDE_intersection_characterization_l1944_194484


namespace NUMINAMATH_CALUDE_octagon_area_reduction_l1944_194460

theorem octagon_area_reduction (x : ℝ) : 
  x > 0 ∧ x < 1 →  -- The smaller square's side length is positive and less than the original square
  4 + 2*x = 1.4 * 4 →  -- Perimeter condition
  (1 - x^2) / 1 = 0.36 :=  -- Area reduction
by sorry

end NUMINAMATH_CALUDE_octagon_area_reduction_l1944_194460


namespace NUMINAMATH_CALUDE_ternary_to_decimal_l1944_194448

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

def ternary_number : List Nat := [1, 0, 2, 0, 1, 2]

theorem ternary_to_decimal :
  to_decimal ternary_number 3 = 320 := by
  sorry

end NUMINAMATH_CALUDE_ternary_to_decimal_l1944_194448


namespace NUMINAMATH_CALUDE_exactly_one_hit_probability_l1944_194426

theorem exactly_one_hit_probability (p : ℝ) (h : p = 0.6) :
  p * (1 - p) + (1 - p) * p = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_hit_probability_l1944_194426


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1944_194442

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : M ∩ (U \ N) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1944_194442


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_76_l1944_194478

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (4, -3)
def v2 : ℝ × ℝ := (4, 7)
def v3 : ℝ × ℝ := (12, 2)
def v4 : ℝ × ℝ := (12, -7)

-- Define the function to calculate the area of the quadrilateral
def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area_is_76 : 
  quadrilateralArea v1 v2 v3 v4 = 76 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_76_l1944_194478


namespace NUMINAMATH_CALUDE_largest_integer_problem_l1944_194458

theorem largest_integer_problem (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℕ) = {57, 70, 83} →
  max a (max b (max c (max d e))) = 48 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_problem_l1944_194458


namespace NUMINAMATH_CALUDE_smallest_third_term_of_gp_l1944_194497

theorem smallest_third_term_of_gp (a b c : ℝ) : 
  (∃ d : ℝ, a = 5 ∧ b = 5 + d ∧ c = 5 + 2*d) →  -- arithmetic progression
  (∃ r : ℝ, 5 * (20 + 2*c - 10) = (8 + b - 5)^2) →  -- geometric progression after modification
  20 + 2*c - 10 ≥ -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_gp_l1944_194497


namespace NUMINAMATH_CALUDE_min_value_fraction_l1944_194477

theorem min_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∃ (m : ℝ), m = 3 ∧ ∀ z, z = (x * y) / x → m ≤ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1944_194477


namespace NUMINAMATH_CALUDE_parabola_vertex_not_in_second_quadrant_l1944_194404

/-- The vertex of the parabola y = 4x^2 - 4(a+1)x + a cannot lie in the second quadrant for any real value of a. -/
theorem parabola_vertex_not_in_second_quadrant (a : ℝ) : 
  let f (x : ℝ) := 4 * x^2 - 4 * (a + 1) * x + a
  let vertex_x := (a + 1) / 2
  let vertex_y := f vertex_x
  ¬(vertex_x < 0 ∧ vertex_y > 0) := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_not_in_second_quadrant_l1944_194404


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l1944_194437

def systematic_sample_count (total_population : ℕ) (sample_size : ℕ) (range_start : ℕ) (range_end : ℕ) : ℕ :=
  let group_size := total_population / sample_size
  ((range_end - range_start + 1) / group_size)

theorem systematic_sample_theorem :
  systematic_sample_count 800 20 121 400 = 7 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l1944_194437


namespace NUMINAMATH_CALUDE_even_function_implies_A_equals_one_l1944_194480

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x + 1)(x - A) -/
def f (A : ℝ) : ℝ → ℝ := λ x ↦ (x + 1) * (x - A)

/-- If f(x) = (x + 1)(x - A) is an even function, then A = 1 -/
theorem even_function_implies_A_equals_one :
  IsEven (f A) → A = 1 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_A_equals_one_l1944_194480


namespace NUMINAMATH_CALUDE_interest_rate_difference_l1944_194421

/-- Proves that for a sum of $700 at simple interest for 4 years, 
    if a higher rate fetches $56 more interest, 
    then the difference between the higher rate and the original rate is 2 percentage points. -/
theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (original_rate : ℝ) 
  (higher_rate : ℝ) 
  (h1 : principal = 700) 
  (h2 : time = 4) 
  (h3 : higher_rate * principal * time / 100 = original_rate * principal * time / 100 + 56) : 
  higher_rate - original_rate = 2 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l1944_194421


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1944_194425

theorem unique_solution_absolute_value_equation :
  ∃! y : ℝ, |y - 25| + |y - 15| = |2*y - 40| :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1944_194425


namespace NUMINAMATH_CALUDE_system_solution_unique_l1944_194446

theorem system_solution_unique : 
  ∃! (x y : ℝ), (2 * x - y = 5) ∧ (3 * x + 2 * y = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1944_194446


namespace NUMINAMATH_CALUDE_haley_final_lives_l1944_194431

/-- Calculates the final number of lives in a video game scenario. -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Proves that for the given scenario, the final number of lives is 46. -/
theorem haley_final_lives : final_lives 14 4 36 = 46 := by
  sorry

end NUMINAMATH_CALUDE_haley_final_lives_l1944_194431


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l1944_194440

/-- Two parallel lines in the plane -/
structure ParallelLines :=
  (a : ℝ)
  (l₁ : ℝ → ℝ → Prop)
  (l₂ : ℝ → ℝ → Prop)
  (h_l₁ : ∀ x y, l₁ x y ↔ x + (a - 1) * y + 2 = 0)
  (h_l₂ : ∀ x y, l₂ x y ↔ a * x + 2 * y + 1 = 0)
  (h_parallel : ∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → (x₁ - x₂) * 2 = (y₁ - y₂) * (1 - a))

/-- Distance between two lines -/
def distance (l₁ l₂ : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem: If the distance between two parallel lines is 3√5/5, then a = -1 -/
theorem parallel_lines_distance (lines : ParallelLines) :
  distance lines.l₁ lines.l₂ = 3 * Real.sqrt 5 / 5 → lines.a = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l1944_194440


namespace NUMINAMATH_CALUDE_correct_sticker_distribution_l1944_194469

/-- Represents the number of stickers Miss Walter has and distributes -/
structure StickerDistribution where
  gold : Nat
  silver : Nat
  bronze : Nat
  students : Nat

/-- Calculates the number of stickers each student receives -/
def stickersPerStudent (sd : StickerDistribution) : Nat :=
  (sd.gold + sd.silver + sd.bronze) / sd.students

/-- Theorem stating the correct number of stickers each student receives -/
theorem correct_sticker_distribution :
  ∀ sd : StickerDistribution,
    sd.gold = 50 →
    sd.silver = 2 * sd.gold →
    sd.bronze = sd.silver - 20 →
    sd.students = 5 →
    stickersPerStudent sd = 46 := by
  sorry

end NUMINAMATH_CALUDE_correct_sticker_distribution_l1944_194469


namespace NUMINAMATH_CALUDE_no_fraternity_member_is_club_member_l1944_194453

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (ClubMember : U → Prop)
variable (FraternityMember : U → Prop)
variable (Honest : U → Prop)

-- State the theorem
theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, ClubMember x → Honest x)
  (h2 : ∃ x, Student x ∧ ¬Honest x)
  (h3 : ∀ x, Student x → FraternityMember x → ¬ClubMember x) :
  ∀ x, FraternityMember x → ¬ClubMember x :=
by
  sorry

end NUMINAMATH_CALUDE_no_fraternity_member_is_club_member_l1944_194453


namespace NUMINAMATH_CALUDE_ten_point_circle_chords_l1944_194482

/-- The number of chords between non-adjacent points on a circle with n points -/
def non_adjacent_chords (n : ℕ) : ℕ :=
  Nat.choose n 2 - n

/-- Theorem: Given 10 points on a circle, there are 35 chords connecting non-adjacent points -/
theorem ten_point_circle_chords :
  non_adjacent_chords 10 = 35 := by
  sorry

#eval non_adjacent_chords 10  -- This should output 35

end NUMINAMATH_CALUDE_ten_point_circle_chords_l1944_194482


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1944_194464

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1944_194464


namespace NUMINAMATH_CALUDE_triangle_max_sin2A_tan2C_l1944_194489

theorem triangle_max_sin2A_tan2C (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are positive sides of the triangle
  0 < a ∧ 0 < b ∧ 0 < c →
  -- -c cosB is the arithmetic mean of √2a cosB and √2b cosA
  -c * Real.cos B = (Real.sqrt 2 * a * Real.cos B + Real.sqrt 2 * b * Real.cos A) / 2 →
  -- Maximum value of sin2A•tan²C
  ∃ (max : Real), ∀ (A' B' C' : Real) (a' b' c' : Real),
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = π →
    0 < a' ∧ 0 < b' ∧ 0 < c' →
    -c' * Real.cos B' = (Real.sqrt 2 * a' * Real.cos B' + Real.sqrt 2 * b' * Real.cos A') / 2 →
    Real.sin (2 * A') * (Real.tan C')^2 ≤ max ∧
    max = 3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_sin2A_tan2C_l1944_194489


namespace NUMINAMATH_CALUDE_hike_length_is_48_l1944_194474

/-- Represents the length of a multi-day hike --/
structure HikeLength where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hike as described in the problem --/
def hike_conditions (h : HikeLength) : Prop :=
  h.day1 + h.day2 + h.day3 = 34 ∧
  (h.day2 + h.day3) / 2 = 12 ∧
  h.day3 + h.day4 + h.day5 = 40 ∧
  h.day1 + h.day3 + h.day5 = 38 ∧
  h.day4 = 14

/-- The theorem stating that given the conditions, the total length of the trail is 48 miles --/
theorem hike_length_is_48 (h : HikeLength) (hc : hike_conditions h) :
  h.day1 + h.day2 + h.day3 + h.day4 + h.day5 = 48 := by
  sorry


end NUMINAMATH_CALUDE_hike_length_is_48_l1944_194474


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1944_194412

theorem expression_simplification_and_evaluation :
  let a : ℚ := -2
  let b : ℚ := 1/3
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1944_194412


namespace NUMINAMATH_CALUDE_min_digit_ratio_l1944_194498

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  digits_bound : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The sum of digits of a three-digit number -/
def ThreeDigitNumber.digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- The ratio of a number to the sum of its digits -/
def digitRatio (n : ThreeDigitNumber) : Rat :=
  n.value / n.digitSum

/-- The condition that the difference between hundreds and tens digit is 8 -/
def diffEight (n : ThreeDigitNumber) : Prop :=
  n.hundreds - n.tens = 8 ∨ n.tens - n.hundreds = 8

theorem min_digit_ratio :
  ∀ k : ThreeDigitNumber,
    diffEight k →
    ∀ m : ThreeDigitNumber,
      diffEight m →
      digitRatio k ≤ digitRatio m →
      k.value = 190 :=
sorry

end NUMINAMATH_CALUDE_min_digit_ratio_l1944_194498


namespace NUMINAMATH_CALUDE_no_perfect_square_9999xxxx_l1944_194457

theorem no_perfect_square_9999xxxx : ¬∃ x : ℕ, 
  99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ y : ℕ, x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_9999xxxx_l1944_194457


namespace NUMINAMATH_CALUDE_right_triangle_properties_l1944_194401

theorem right_triangle_properties (a b c h : ℝ) : 
  a = 12 → b = 5 → c^2 = a^2 + b^2 → (1/2) * a * b = (1/2) * c * h →
  c = 13 ∧ h = 60/13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l1944_194401


namespace NUMINAMATH_CALUDE_muffin_division_l1944_194487

theorem muffin_division (num_friends : ℕ) (total_muffins : ℕ) : 
  num_friends = 4 → total_muffins = 20 → (total_muffins / (num_friends + 1) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_muffin_division_l1944_194487


namespace NUMINAMATH_CALUDE_brooke_has_eight_customers_l1944_194490

/-- Represents Brooke's milk and butter business --/
structure MilkBusiness where
  num_cows : ℕ
  milk_price : ℚ
  butter_price : ℚ
  milk_per_cow : ℕ
  milk_per_customer : ℕ
  total_revenue : ℚ

/-- Calculates the number of customers in Brooke's milk business --/
def calculate_customers (business : MilkBusiness) : ℕ :=
  let total_milk := business.num_cows * business.milk_per_cow
  total_milk / business.milk_per_customer

/-- Theorem stating that Brooke has 8 customers --/
theorem brooke_has_eight_customers :
  let brooke_business : MilkBusiness := {
    num_cows := 12,
    milk_price := 3,
    butter_price := 3/2,
    milk_per_cow := 4,
    milk_per_customer := 6,
    total_revenue := 144
  }
  calculate_customers brooke_business = 8 := by
  sorry

end NUMINAMATH_CALUDE_brooke_has_eight_customers_l1944_194490


namespace NUMINAMATH_CALUDE_abs_z2_minus_z1_equals_sqrt2_l1944_194491

theorem abs_z2_minus_z1_equals_sqrt2 : ∀ (z₁ z₂ : ℂ), 
  z₁ = 1 + 2*Complex.I → z₂ = 2 + Complex.I → Complex.abs (z₂ - z₁) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z2_minus_z1_equals_sqrt2_l1944_194491


namespace NUMINAMATH_CALUDE_complement_P_wrt_U_l1944_194468

-- Define the sets U and P
def U : Set ℝ := Set.univ
def P : Set ℝ := Set.Ioo 0 (1/2)

-- State the theorem
theorem complement_P_wrt_U :
  (U \ P) = Set.Iic 0 ∪ Set.Ici (1/2) := by
  sorry

end NUMINAMATH_CALUDE_complement_P_wrt_U_l1944_194468


namespace NUMINAMATH_CALUDE_system_solution_l1944_194436

theorem system_solution (a b x y : ℝ) 
  (h1 : (x - y) / (1 - x * y) = 2 * a / (1 + a^2))
  (h2 : (x + y) / (1 + x * y) = 2 * b / (1 + b^2))
  (ha : a^2 ≠ 1)
  (hb : b^2 ≠ 1)
  (hab : a ≠ b)
  (hnr : a * b ≠ 1) :
  ((x = (a * b + 1) / (a + b) ∧ y = (a * b - 1) / (a - b)) ∨
   (x = (a + b) / (a * b + 1) ∧ y = (a - b) / (a * b - 1))) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1944_194436


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1944_194405

theorem initial_number_of_persons (avg_weight_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_weight_increase = 1.5 →
  old_weight = 65 →
  new_weight = 78.5 →
  (new_weight - old_weight) / avg_weight_increase = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l1944_194405


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1944_194407

-- Define an isosceles triangle with sides of lengths 3 and 7
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ c = 3) ∨ (a = c ∧ b = 3) ∨ (b = c ∧ a = 3)

-- Triangle inequality theorem
axiom triangle_inequality (a b c : ℝ) : a > 0 → b > 0 → c > 0 → a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ,
  IsoscelesTriangle a b c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 7 ∨ b = 7 ∨ c = 7) →
  a + b + c = 17 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1944_194407


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1944_194423

open Real

-- Define the property p
def property_p (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + π) = -f x

-- Define the property q
def property_q (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2*π) = f x

-- Theorem: p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ f : ℝ → ℝ, property_p f → property_q f) ∧
  (∃ f : ℝ → ℝ, property_q f ∧ ¬property_p f) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1944_194423


namespace NUMINAMATH_CALUDE_alberts_earnings_increase_l1944_194428

/-- Proves that given Albert's earnings of $495 after a 36% increase and $454.96 after an unknown percentage increase, the unknown percentage increase is 25%. -/
theorem alberts_earnings_increase (original : ℝ) (increased : ℝ) (percentage : ℝ) : 
  (original * 1.36 = 495) → 
  (original * (1 + percentage) = 454.96) → 
  percentage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_alberts_earnings_increase_l1944_194428


namespace NUMINAMATH_CALUDE_arrange_plates_eq_365240_l1944_194461

/-- Number of ways to arrange plates around a circular table with constraints -/
def arrange_plates : ℕ :=
  let total_plates : ℕ := 13
  let blue_plates : ℕ := 6
  let red_plates : ℕ := 3
  let green_plates : ℕ := 3
  let orange_plates : ℕ := 1
  let total_arrangements : ℕ := (Nat.factorial (total_plates - 1)) / (Nat.factorial blue_plates * Nat.factorial red_plates * Nat.factorial green_plates)
  let green_adjacent : ℕ := (Nat.factorial (total_plates - green_plates)) / (Nat.factorial blue_plates * Nat.factorial red_plates)
  let red_adjacent : ℕ := (Nat.factorial (total_plates - red_plates)) / (Nat.factorial blue_plates * Nat.factorial green_plates)
  total_arrangements - green_adjacent - red_adjacent

theorem arrange_plates_eq_365240 : arrange_plates = 365240 := by
  sorry

end NUMINAMATH_CALUDE_arrange_plates_eq_365240_l1944_194461


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1944_194452

theorem completing_square_quadratic (x : ℝ) : 
  (∃ c, (x^2 + 4*x + 2 = 0) ↔ ((x + 2)^2 = c)) → 
  (∃ c, ((x + 2)^2 = c) ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1944_194452
