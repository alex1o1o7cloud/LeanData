import Mathlib

namespace NUMINAMATH_CALUDE_tank_initial_water_l912_91233

def tank_capacity : ℚ := 100
def day1_collection : ℚ := 15
def day2_collection : ℚ := 20
def day3_overflow : ℚ := 25

theorem tank_initial_water (initial_water : ℚ) :
  initial_water + day1_collection + day2_collection = tank_capacity ∧
  (initial_water / tank_capacity = 13 / 20) := by
  sorry

end NUMINAMATH_CALUDE_tank_initial_water_l912_91233


namespace NUMINAMATH_CALUDE_range_of_a_l912_91289

-- Define the propositions p and q
def p (a x : ℝ) : Prop := 3 * a < x ∧ x < a
def q (x : ℝ) : Prop := x^2 - x - 6 < 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ,
  (a < 0) →
  (∀ x : ℝ, ¬(p a x) → ¬(q x)) →
  (∃ x : ℝ, ¬(p a x) ∧ q x) →
  -2/3 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l912_91289


namespace NUMINAMATH_CALUDE_dusting_team_combinations_l912_91283

theorem dusting_team_combinations (n : ℕ) (k : ℕ) : n = 5 → k = 3 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_dusting_team_combinations_l912_91283


namespace NUMINAMATH_CALUDE_project_distribution_count_l912_91296

/-- The number of ways to distribute 8 projects among 4 companies -/
def distribute_projects : ℕ :=
  -- Total projects
  let total := 8
  -- Projects for each company
  let company_A := 3
  let company_B := 1
  let company_C := 2
  let company_D := 2
  -- The actual calculation would go here
  1680

/-- Theorem stating that the number of ways to distribute the projects is 1680 -/
theorem project_distribution_count : distribute_projects = 1680 := by
  sorry

end NUMINAMATH_CALUDE_project_distribution_count_l912_91296


namespace NUMINAMATH_CALUDE_min_value_of_function_l912_91225

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  (4 * x^2 + 12 * x + 25) / (6 * (1 + x)) ≥ 8/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l912_91225


namespace NUMINAMATH_CALUDE_number_theory_problem_l912_91257

theorem number_theory_problem :
  (∃ n : ℤ, 35 = 5 * n) ∧
  (∃ n : ℤ, 252 = 21 * n) ∧ ¬(∃ m : ℤ, 48 = 21 * m) ∧
  (∃ k : ℤ, 180 = 9 * k) := by
  sorry

end NUMINAMATH_CALUDE_number_theory_problem_l912_91257


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_one_two_left_closed_l912_91288

/-- The set M of real numbers x such that (x + 3)(x - 2) < 0 -/
def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}

/-- The set N of real numbers x such that 1 ≤ x ≤ 3 -/
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

/-- The theorem stating that the intersection of M and N is equal to the interval [1, 2) -/
theorem M_intersect_N_eq_one_two_left_closed :
  M ∩ N = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_one_two_left_closed_l912_91288


namespace NUMINAMATH_CALUDE_complex_equation_solution_l912_91234

theorem complex_equation_solution (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l912_91234


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l912_91248

/-- An isosceles triangle with two side lengths of 6 and 8 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 6 ∧ b = 8 ∧ (c = a ∨ c = b) →  -- Triangle is isosceles with sides 6 and 8
  a + b + c = 22 :=                  -- Perimeter is 22
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l912_91248


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l912_91202

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

-- Define the conditions of the problem
def sequence_conditions (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧
  a 1 = 1 ∧
  a 3 = Real.sqrt (a 1 * a 9)

-- State the theorem
theorem arithmetic_sequence_theorem (a : ℕ → ℝ) :
  sequence_conditions a → ∀ n : ℕ, a n = n := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l912_91202


namespace NUMINAMATH_CALUDE_library_visits_total_l912_91219

/-- The number of times William goes to the library per week -/
def william_freq : ℕ := 2

/-- The number of times Jason goes to the library per week -/
def jason_freq : ℕ := 4 * william_freq

/-- The number of times Emma goes to the library per week -/
def emma_freq : ℕ := 3 * jason_freq

/-- The number of times Zoe goes to the library per week -/
def zoe_freq : ℕ := william_freq / 2

/-- The number of times Chloe goes to the library per week -/
def chloe_freq : ℕ := emma_freq / 3

/-- The number of weeks -/
def weeks : ℕ := 8

/-- The total number of times Jason, Emma, Zoe, and Chloe go to the library over 8 weeks -/
def total_visits : ℕ := (jason_freq + emma_freq + zoe_freq + chloe_freq) * weeks

theorem library_visits_total : total_visits = 328 := by
  sorry

end NUMINAMATH_CALUDE_library_visits_total_l912_91219


namespace NUMINAMATH_CALUDE_diamond_two_three_l912_91221

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a * b^2 - b + a^2 + 1

-- Theorem statement
theorem diamond_two_three : diamond 2 3 = 20 := by sorry

end NUMINAMATH_CALUDE_diamond_two_three_l912_91221


namespace NUMINAMATH_CALUDE_rationalize_denominator_l912_91217

theorem rationalize_denominator :
  ∃ (a b : ℝ), a + b * Real.sqrt 3 = -Real.sqrt 3 - 2 ∧ 
  (a + b * Real.sqrt 3) * (Real.sqrt 3 - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l912_91217


namespace NUMINAMATH_CALUDE_sum_of_sequences_l912_91201

def sequence1 : List ℕ := [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
def sequence2 : List ℕ := [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum = 1000) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l912_91201


namespace NUMINAMATH_CALUDE_install_remaining_windows_time_l912_91236

/-- Calculates the time needed to install remaining windows -/
def timeToInstallRemaining (totalWindows installedWindows timePerWindow : ℕ) : ℕ :=
  (totalWindows - installedWindows) * timePerWindow

/-- Proves that the time to install the remaining windows is 18 hours -/
theorem install_remaining_windows_time :
  timeToInstallRemaining 9 6 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_install_remaining_windows_time_l912_91236


namespace NUMINAMATH_CALUDE_cooking_time_per_potato_l912_91264

theorem cooking_time_per_potato 
  (total_potatoes : ℕ) 
  (cooked_potatoes : ℕ) 
  (time_for_rest : ℕ) 
  (h1 : total_potatoes = 16) 
  (h2 : cooked_potatoes = 7) 
  (h3 : time_for_rest = 45) : 
  (time_for_rest : ℚ) / ((total_potatoes - cooked_potatoes) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cooking_time_per_potato_l912_91264


namespace NUMINAMATH_CALUDE_harkamal_payment_l912_91267

/-- The total amount Harkamal paid to the shopkeeper for grapes and mangoes. -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1125 to the shopkeeper. -/
theorem harkamal_payment : total_amount_paid 9 70 9 55 = 1125 := by
  sorry

#eval total_amount_paid 9 70 9 55

end NUMINAMATH_CALUDE_harkamal_payment_l912_91267


namespace NUMINAMATH_CALUDE_ordering_of_special_values_l912_91253

theorem ordering_of_special_values :
  let a := Real.exp (1/2)
  let b := Real.log (1/2)
  let c := Real.sin (1/2)
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ordering_of_special_values_l912_91253


namespace NUMINAMATH_CALUDE_identify_six_genuine_coins_l912_91286

/-- Represents the result of a weighing on a balance scale -/
inductive WeighResult
| Equal : WeighResult
| LeftHeavier : WeighResult
| RightHeavier : WeighResult

/-- Represents a group of coins -/
structure CoinGroup where
  total : Nat
  genuine : Nat
  counterfeit : Nat

/-- Represents a weighing action on the balance scale -/
def weigh (left right : CoinGroup) : WeighResult :=
  sorry

/-- Represents the process of identifying genuine coins -/
def identifyGenuineCoins (coins : CoinGroup) (maxWeighings : Nat) : Option (Fin 6 → Bool) :=
  sorry

theorem identify_six_genuine_coins :
  ∀ (coins : CoinGroup),
    coins.total = 25 →
    coins.genuine = 22 →
    coins.counterfeit = 3 →
    ∃ (result : Fin 6 → Bool),
      identifyGenuineCoins coins 2 = some result ∧
      (∀ i, result i = true → i.val < 6) :=
by
  sorry

end NUMINAMATH_CALUDE_identify_six_genuine_coins_l912_91286


namespace NUMINAMATH_CALUDE_letitia_order_l912_91244

theorem letitia_order (julie_order anton_order individual_tip tip_percentage : ℚ) 
  (h1 : julie_order = 10)
  (h2 : anton_order = 30)
  (h3 : individual_tip = 4)
  (h4 : tip_percentage = 1/5)
  : ∃ letitia_order : ℚ, 
    tip_percentage * (julie_order + letitia_order + anton_order) = 3 * individual_tip ∧ 
    letitia_order = 20 := by
  sorry

end NUMINAMATH_CALUDE_letitia_order_l912_91244


namespace NUMINAMATH_CALUDE_inequality_condition_max_value_l912_91251

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

-- Statement 1
theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 :=
sorry

-- Statement 2
theorem max_value (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ 
    ∀ y ∈ Set.Icc 0 2, h a x ≥ h a y) ∧
  (∃ m : ℝ, (∀ x ∈ Set.Icc 0 2, h a x ≤ m) ∧
    m = if a ≥ -3 then a + 3 else 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_max_value_l912_91251


namespace NUMINAMATH_CALUDE_evaluate_expression_l912_91279

theorem evaluate_expression (x : ℝ) (h : x = 2) : x^3 + x^2 + x + Real.exp x = 14 + Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l912_91279


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l912_91298

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive sides
  a + b + c = 40 ∧  -- perimeter condition
  (1/2) * a * b = 24 ∧  -- area condition
  a^2 + b^2 = c^2 ∧  -- right triangle (Pythagorean theorem)
  c = 18.8 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l912_91298


namespace NUMINAMATH_CALUDE_range_of_m_l912_91292

def is_hyperbola (m : ℝ) : Prop := (m + 2) * (m - 3) > 0

def no_positive_roots (m : ℝ) : Prop :=
  m = 0 ∨ (m ≠ 0 ∧ (∀ x : ℝ, x > 0 → m * x^2 + (m + 3) * x + 4 ≠ 0))

theorem range_of_m (m : ℝ) :
  (is_hyperbola m ∨ no_positive_roots m) ∧
  ¬(is_hyperbola m ∧ no_positive_roots m) →
  m < -2 ∨ (0 ≤ m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l912_91292


namespace NUMINAMATH_CALUDE_car_distribution_l912_91238

theorem car_distribution (total_cars_per_column : ℕ) 
                         (total_zhiguli : ℕ) 
                         (zhiguli_first : ℕ) 
                         (zhiguli_second : ℕ) :
  total_cars_per_column = 28 →
  total_zhiguli = 11 →
  zhiguli_first + zhiguli_second = total_zhiguli →
  (total_cars_per_column - zhiguli_first) = 2 * (total_cars_per_column - zhiguli_second) →
  (total_cars_per_column - zhiguli_first = 21 ∧ total_cars_per_column - zhiguli_second = 24) :=
by
  sorry

#check car_distribution

end NUMINAMATH_CALUDE_car_distribution_l912_91238


namespace NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l912_91282

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (-3, 1)

theorem a_perpendicular_to_a_minus_b : a • (a - b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l912_91282


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l912_91210

def rectangle_lengths : List ℕ := [1, 9, 25, 49, 81, 121]
def common_width : ℕ := 3

theorem sum_of_rectangle_areas :
  (rectangle_lengths.map (λ l => l * common_width)).sum = 858 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l912_91210


namespace NUMINAMATH_CALUDE_isabellas_paintable_area_l912_91272

/-- Calculates the total paintable area for a set of identical rooms -/
def totalPaintableArea (
  numRooms : ℕ
  ) (length width height : ℝ
  ) (unpaintableAreaPerRoom : ℝ
  ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let paintableAreaPerRoom := wallArea - unpaintableAreaPerRoom
  numRooms * paintableAreaPerRoom

/-- Proves that the total paintable area for Isabella's bedrooms is 1592 square feet -/
theorem isabellas_paintable_area :
  totalPaintableArea 4 15 11 9 70 = 1592 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_paintable_area_l912_91272


namespace NUMINAMATH_CALUDE_problem_solution_l912_91229

def is_arithmetic_sequence (s : Fin 5 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 4, s (i + 1) - s i = d

def is_geometric_sequence (s : Fin 5 → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, s (i + 1) / s i = r

theorem problem_solution (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) :
  is_arithmetic_sequence (λ i => match i with
    | 0 => 1
    | 1 => a₁
    | 2 => a₂
    | 3 => a₃
    | 4 => 9) →
  is_geometric_sequence (λ i => match i with
    | 0 => -9
    | 1 => b₁
    | 2 => b₂
    | 3 => b₃
    | 4 => -1) →
  b₂ / (a₁ + a₃) = -3/10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l912_91229


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l912_91239

/-- A linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def onLine (p : Point) (f : LinearFunction) : Prop :=
  p.y = f.m * p.x + f.b

theorem y1_less_than_y2 
  (f : LinearFunction)
  (p1 p2 : Point)
  (h1 : f.m = 8)
  (h2 : f.b = -1)
  (h3 : p1.x = 3)
  (h4 : p2.x = 4)
  (h5 : onLine p1 f)
  (h6 : onLine p2 f) :
  p1.y < p2.y := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l912_91239


namespace NUMINAMATH_CALUDE_right_triangle_min_perimeter_l912_91240

theorem right_triangle_min_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 1 →
  a^2 + b^2 = c^2 →
  a + b + c ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_min_perimeter_l912_91240


namespace NUMINAMATH_CALUDE_seaweed_livestock_amount_l912_91205

-- Define the total amount of seaweed harvested
def total_seaweed : ℝ := 400

-- Define the percentage of seaweed used for fires
def fire_percentage : ℝ := 0.5

-- Define the percentage of remaining seaweed for human consumption
def human_percentage : ℝ := 0.25

-- Function to calculate the amount of seaweed fed to livestock
def seaweed_for_livestock : ℝ :=
  let remaining_after_fire := total_seaweed * (1 - fire_percentage)
  let for_humans := remaining_after_fire * human_percentage
  remaining_after_fire - for_humans

-- Theorem stating the amount of seaweed fed to livestock
theorem seaweed_livestock_amount : seaweed_for_livestock = 150 := by
  sorry

end NUMINAMATH_CALUDE_seaweed_livestock_amount_l912_91205


namespace NUMINAMATH_CALUDE_pure_imaginary_magnitude_l912_91297

theorem pure_imaginary_magnitude (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 9) (m^2 + 2*m - 3)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 12 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_magnitude_l912_91297


namespace NUMINAMATH_CALUDE_roy_sports_time_l912_91254

/-- Calculates the total time spent on sports activities for a specific week --/
def total_sports_time (
  basketball_time : ℝ)
  (swimming_time : ℝ)
  (track_time : ℝ)
  (school_days : ℕ)
  (missed_days : ℕ)
  (weekend_soccer : ℝ)
  (weekend_basketball : ℝ)
  (canceled_swimming : ℕ) : ℝ :=
  let school_sports := (basketball_time + swimming_time + track_time) * (school_days - missed_days : ℝ) - 
                       swimming_time * canceled_swimming
  let weekend_sports := weekend_soccer + weekend_basketball
  school_sports + weekend_sports

/-- Theorem stating that Roy's total sports time for the specific week is 13.5 hours --/
theorem roy_sports_time : 
  total_sports_time 1 1.5 1 5 2 1.5 3 1 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_roy_sports_time_l912_91254


namespace NUMINAMATH_CALUDE_max_triangle_sum_l912_91200

/-- Represents the arrangement of numbers on the vertices of the triangles -/
def TriangleArrangement := Fin 6 → Fin 6

/-- The sum of three numbers on a side of a triangle -/
def sideSum (arr : TriangleArrangement) (i j k : Fin 6) : ℕ :=
  (arr i).val + 12 + (arr j).val + 12 + (arr k).val + 12

/-- Predicate to check if an arrangement is valid -/
def isValidArrangement (arr : TriangleArrangement) : Prop :=
  (∀ i j, i ≠ j → arr i ≠ arr j) ∧
  (∀ i, arr i < 6)

/-- Predicate to check if all sides have the same sum -/
def allSidesEqual (arr : TriangleArrangement) (S : ℕ) : Prop :=
  sideSum arr 0 1 2 = S ∧
  sideSum arr 2 3 4 = S ∧
  sideSum arr 4 5 0 = S

theorem max_triangle_sum :
  ∃ (S : ℕ) (arr : TriangleArrangement),
    isValidArrangement arr ∧
    allSidesEqual arr S ∧
    (∀ (S' : ℕ) (arr' : TriangleArrangement),
      isValidArrangement arr' → allSidesEqual arr' S' → S' ≤ S) ∧
    S = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_sum_l912_91200


namespace NUMINAMATH_CALUDE_min_value_theorem_l912_91252

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 9/y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 2 ∧ 1/x₀ + 9/y₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l912_91252


namespace NUMINAMATH_CALUDE_juice_sales_theorem_l912_91237

/-- Represents the capacity of a can in liters -/
structure CanCapacity where
  large : ℝ
  medium : ℝ
  liter : ℝ

/-- Represents the daily sales data -/
structure DailySales where
  large : ℕ
  medium : ℕ
  liter : ℕ

/-- Calculates the total volume of juice sold in a day -/
def dailyVolume (c : CanCapacity) (s : DailySales) : ℝ :=
  c.large * s.large + c.medium * s.medium + c.liter * s.liter

theorem juice_sales_theorem (c : CanCapacity) 
  (s1 s2 s3 : DailySales) : 
  c.liter = 1 →
  s1 = ⟨1, 4, 0⟩ →
  s2 = ⟨2, 0, 6⟩ →
  s3 = ⟨1, 3, 3⟩ →
  dailyVolume c s1 = dailyVolume c s2 →
  dailyVolume c s2 = dailyVolume c s3 →
  (dailyVolume c s1 + dailyVolume c s2 + dailyVolume c s3) = 54 := by
  sorry

#check juice_sales_theorem

end NUMINAMATH_CALUDE_juice_sales_theorem_l912_91237


namespace NUMINAMATH_CALUDE_counterexample_exists_l912_91276

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l912_91276


namespace NUMINAMATH_CALUDE_parallel_line_plane_perpendicular_transitivity_l912_91261

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Axioms for the properties of lines and planes
axiom different_lines : ∀ (m n : Line), m ≠ n
axiom different_planes : ∀ (α β γ : Plane), α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Theorem 1
theorem parallel_line_plane (m n : Line) (α : Plane) :
  parallel m n → parallelLP n α → (parallelLP m α ∨ subset m α) := by sorry

-- Theorem 2
theorem perpendicular_transitivity (m : Line) (α β γ : Plane) :
  parallelPP α β → parallelPP β γ → perpendicular m α → perpendicular m γ := by sorry

end NUMINAMATH_CALUDE_parallel_line_plane_perpendicular_transitivity_l912_91261


namespace NUMINAMATH_CALUDE_elderly_workers_in_sample_l912_91275

/-- Represents the composition of workers in a company --/
structure WorkforceComposition where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents a stratified sample from the workforce --/
structure StratifiedSample where
  youngInSample : ℕ
  elderlyInSample : ℕ

/-- Theorem stating the number of elderly workers in the stratified sample --/
theorem elderly_workers_in_sample 
  (wc : WorkforceComposition) 
  (sample : StratifiedSample) : 
  wc.total = 430 →
  wc.young = 160 →
  wc.middleAged = 2 * wc.elderly →
  sample.youngInSample = 32 →
  sample.elderlyInSample = 18 := by
  sorry

end NUMINAMATH_CALUDE_elderly_workers_in_sample_l912_91275


namespace NUMINAMATH_CALUDE_problem_solution_l912_91245

theorem problem_solution : 
  (2/3 - 3/4 + 1/6) / (-1/24) = -2 ∧ 
  -2^3 + 3 * (-1)^2023 - |3-7| = -15 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l912_91245


namespace NUMINAMATH_CALUDE_porch_length_calculation_l912_91280

/-- Given the dimensions of a house and porch, and the total area needing shingles,
    calculate the length of the porch. -/
theorem porch_length_calculation
  (house_length : ℝ)
  (house_width : ℝ)
  (porch_width : ℝ)
  (total_area : ℝ)
  (h1 : house_length = 20.5)
  (h2 : house_width = 10)
  (h3 : porch_width = 4.5)
  (h4 : total_area = 232) :
  (total_area - house_length * house_width) / porch_width = 6 := by
  sorry

end NUMINAMATH_CALUDE_porch_length_calculation_l912_91280


namespace NUMINAMATH_CALUDE_sin_cos_transformation_given_condition_l912_91290

theorem sin_cos_transformation (x : ℝ) :
  4 * Real.sin x * Real.cos x = 2 * Real.sin (2 * x + π / 6) :=
by
  sorry

-- Additional theorem to represent the given condition
theorem given_condition (x : ℝ) :
  Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x) = 2 * Real.sin (2 * x - π / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_sin_cos_transformation_given_condition_l912_91290


namespace NUMINAMATH_CALUDE_no_single_digit_fraction_l912_91299

theorem no_single_digit_fraction :
  ¬ ∃ (n : ℕ+) (a b : ℕ),
    1 ≤ a ∧ a < 10 ∧
    1 ≤ b ∧ b < 10 ∧
    (1234 - n) * b = (6789 - n) * a :=
by sorry

end NUMINAMATH_CALUDE_no_single_digit_fraction_l912_91299


namespace NUMINAMATH_CALUDE_inequality_solution_set_l912_91249

theorem inequality_solution_set (x : ℝ) : (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l912_91249


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_multiples_l912_91277

theorem largest_of_three_consecutive_multiples (a b c : ℕ) : 
  (∃ n : ℕ, a = 3 * n ∧ b = 3 * n + 3 ∧ c = 3 * n + 6) →  -- Consecutive multiples of 3
  a + b + c = 117 →                                      -- Sum is 117
  c = 42 ∧ c ≥ a ∧ c ≥ b                                 -- c is the largest and equals 42
  := by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_multiples_l912_91277


namespace NUMINAMATH_CALUDE_jacks_remaining_money_l912_91224

def remaining_money (initial_amount snack_cost ride_multiplier game_multiplier : ℝ) : ℝ :=
  initial_amount - (snack_cost + ride_multiplier * snack_cost + game_multiplier * snack_cost)

theorem jacks_remaining_money :
  remaining_money 100 15 3 1.5 = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_jacks_remaining_money_l912_91224


namespace NUMINAMATH_CALUDE_lillian_candy_count_l912_91206

def initial_candies : ℕ := 88
def additional_candies : ℕ := 5

theorem lillian_candy_count :
  initial_candies + additional_candies = 93 := by sorry

end NUMINAMATH_CALUDE_lillian_candy_count_l912_91206


namespace NUMINAMATH_CALUDE_carbonic_acid_formation_l912_91273

-- Define the molecules and their quantities
structure Molecule where
  name : String
  moles : ℕ

-- Define the reaction
def reaction (reactant1 reactant2 product : Molecule) : Prop :=
  reactant1.name = "CO2" ∧ 
  reactant2.name = "H2O" ∧ 
  product.name = "H2CO3" ∧
  reactant1.moles = reactant2.moles ∧
  product.moles = min reactant1.moles reactant2.moles

-- Theorem statement
theorem carbonic_acid_formation 
  (co2 : Molecule) 
  (h2o : Molecule) 
  (h2co3 : Molecule) :
  co2.name = "CO2" →
  h2o.name = "H2O" →
  h2co3.name = "H2CO3" →
  co2.moles = 3 →
  h2o.moles = 3 →
  reaction co2 h2o h2co3 →
  h2co3.moles = 3 :=
by sorry

end NUMINAMATH_CALUDE_carbonic_acid_formation_l912_91273


namespace NUMINAMATH_CALUDE_people_visited_neither_l912_91285

theorem people_visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 100 →
  iceland = 55 →
  norway = 43 →
  both = 61 →
  total - (iceland + norway - both) = 63 :=
by sorry

end NUMINAMATH_CALUDE_people_visited_neither_l912_91285


namespace NUMINAMATH_CALUDE_negation_of_existence_implication_l912_91278

theorem negation_of_existence_implication :
  ¬(∃ n : ℤ, ∀ m : ℤ, n^2 = m^2 → n = m) ↔
  (∀ n : ℤ, ∃ m : ℤ, n^2 = m^2 ∧ n ≠ m) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_implication_l912_91278


namespace NUMINAMATH_CALUDE_constraint_extrema_l912_91241

def constraint (x y : ℝ) : Prop :=
  Real.sqrt (x - 3) + Real.sqrt (y - 4) = 4

def objective (x y : ℝ) : ℝ :=
  2 * x + 3 * y

theorem constraint_extrema :
  ∃ (x_min y_min x_max y_max : ℝ),
    constraint x_min y_min ∧
    constraint x_max y_max ∧
    (∀ x y, constraint x y → objective x y ≥ objective x_min y_min) ∧
    (∀ x y, constraint x y → objective x y ≤ objective x_max y_max) ∧
    x_min = 219 / 25 ∧
    y_min = 264 / 25 ∧
    x_max = 3 ∧
    y_max = 20 ∧
    objective x_min y_min = 37.2 ∧
    objective x_max y_max = 66 :=
  sorry

#check constraint_extrema

end NUMINAMATH_CALUDE_constraint_extrema_l912_91241


namespace NUMINAMATH_CALUDE_election_result_l912_91209

/-- Represents the total number of valid votes cast in the election -/
def total_votes : ℕ := sorry

/-- Represents the number of votes received by the winning candidate -/
def winning_votes : ℕ := 7320

/-- Represents the percentage of votes received by the winning candidate after redistribution -/
def winning_percentage : ℚ := 43 / 100

theorem election_result :
  total_votes * winning_percentage = winning_votes ∧
  total_votes ≥ 17023 ∧
  total_votes < 17024 :=
sorry

end NUMINAMATH_CALUDE_election_result_l912_91209


namespace NUMINAMATH_CALUDE_muffin_cost_is_correct_l912_91218

/-- The cost of a muffin given the total cost and the cost of juice -/
def muffin_cost (total_cost juice_cost : ℚ) : ℚ :=
  (total_cost - juice_cost) / 3

theorem muffin_cost_is_correct (total_cost juice_cost : ℚ) 
  (h1 : total_cost = 370/100) 
  (h2 : juice_cost = 145/100) : 
  muffin_cost total_cost juice_cost = 75/100 := by
  sorry

#eval muffin_cost (370/100) (145/100)

end NUMINAMATH_CALUDE_muffin_cost_is_correct_l912_91218


namespace NUMINAMATH_CALUDE_a_is_editor_l912_91213

-- Define the professions
inductive Profession
| Doctor
| Teacher
| Editor

-- Define the volunteers
structure Volunteer where
  name : String
  profession : Profession
  age : Nat

-- Define the fair
structure Fair where
  volunteers : List Volunteer

-- Define the proposition
theorem a_is_editor (f : Fair) : 
  (∃ a b c : Volunteer, 
    a ∈ f.volunteers ∧ b ∈ f.volunteers ∧ c ∈ f.volunteers ∧
    a.name = "A" ∧ b.name = "B" ∧ c.name = "C" ∧
    a.profession ≠ b.profession ∧ b.profession ≠ c.profession ∧ c.profession ≠ a.profession ∧
    (∃ d : Volunteer, d ∈ f.volunteers ∧ d.profession = Profession.Doctor ∧ d.age ≠ a.age) ∧
    (∃ e : Volunteer, e ∈ f.volunteers ∧ e.profession = Profession.Editor ∧ e.age > c.age) ∧
    (∃ d : Volunteer, d ∈ f.volunteers ∧ d.profession = Profession.Doctor ∧ d.age > b.age)) →
  (∃ a : Volunteer, a ∈ f.volunteers ∧ a.name = "A" ∧ a.profession = Profession.Editor) :=
by sorry

end NUMINAMATH_CALUDE_a_is_editor_l912_91213


namespace NUMINAMATH_CALUDE_cos_alpha_value_l912_91294

theorem cos_alpha_value (α : Real) (h : Real.sin (α - Real.pi/2) = 3/5) :
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l912_91294


namespace NUMINAMATH_CALUDE_honor_guard_subsets_l912_91223

theorem honor_guard_subsets (n : ℕ) (h : n = 60) :
  Finset.card (Finset.powerset (Finset.range n)) = 2^n := by sorry

end NUMINAMATH_CALUDE_honor_guard_subsets_l912_91223


namespace NUMINAMATH_CALUDE_tangent_ellipse_solution_l912_91287

/-- An ellipse with semi-major axis a and semi-minor axis b that is tangent to a rectangle with area 48 -/
structure TangentEllipse where
  a : ℝ
  b : ℝ
  area_eq : a * b = 12
  a_pos : a > 0
  b_pos : b > 0

/-- The theorem stating that the ellipse with a = 4 and b = 3 satisfies the conditions -/
theorem tangent_ellipse_solution :
  ∃ (e : TangentEllipse), e.a = 4 ∧ e.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ellipse_solution_l912_91287


namespace NUMINAMATH_CALUDE_winston_initial_gas_l912_91284

/-- The amount of gas in gallons used for a trip -/
structure Trip where
  gas_used : ℝ

/-- The gas tank of a car -/
structure GasTank where
  capacity : ℝ
  initial_amount : ℝ
  remaining_amount : ℝ

/-- Winston's car trips and gas tank -/
def winston_scenario (store_trip doctor_trip : Trip) (tank : GasTank) : Prop :=
  store_trip.gas_used = 6 ∧
  doctor_trip.gas_used = 2 ∧
  tank.capacity = 12 ∧
  tank.initial_amount = tank.remaining_amount + store_trip.gas_used + doctor_trip.gas_used ∧
  tank.remaining_amount > 0 ∧
  tank.initial_amount ≤ tank.capacity

theorem winston_initial_gas 
  (store_trip doctor_trip : Trip) (tank : GasTank) 
  (h : winston_scenario store_trip doctor_trip tank) : 
  tank.initial_amount = 12 :=
sorry

end NUMINAMATH_CALUDE_winston_initial_gas_l912_91284


namespace NUMINAMATH_CALUDE_supermarket_distribution_l912_91281

/-- Proves that given a total of 420 supermarkets divided between two countries,
    with one country having 56 more supermarkets than the other,
    the country with more supermarkets has 238 supermarkets. -/
theorem supermarket_distribution (total : ℕ) (difference : ℕ) (more : ℕ) (less : ℕ) :
  total = 420 →
  difference = 56 →
  more = less + difference →
  total = more + less →
  more = 238 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_distribution_l912_91281


namespace NUMINAMATH_CALUDE_magazine_subscription_cost_l912_91204

theorem magazine_subscription_cost (reduced_cost : ℝ) (reduction_percentage : ℝ) (original_cost : ℝ) : 
  reduced_cost = 752 ∧ 
  reduction_percentage = 0.20 ∧ 
  reduced_cost = original_cost * (1 - reduction_percentage) →
  original_cost = 940 := by
sorry

end NUMINAMATH_CALUDE_magazine_subscription_cost_l912_91204


namespace NUMINAMATH_CALUDE_absolute_quadratic_inequality_l912_91226

/-- The set of real numbers x satisfying |x^2 - 4x + 3| ≤ 3 is equal to the closed interval [0, 4]. -/
theorem absolute_quadratic_inequality (x : ℝ) :
  |x^2 - 4*x + 3| ≤ 3 ↔ x ∈ Set.Icc 0 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_quadratic_inequality_l912_91226


namespace NUMINAMATH_CALUDE_circle_radius_with_tangent_parabola_l912_91255

theorem circle_radius_with_tangent_parabola :
  ∀ r : ℝ,
  (∃ x : ℝ, x^2 + r = x) →  -- Parabola y = x^2 + r is tangent to line y = x
  (∀ x : ℝ, x^2 + r ≥ x) →  -- Parabola lies above or on the line
  r = (1 : ℝ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_with_tangent_parabola_l912_91255


namespace NUMINAMATH_CALUDE_sandal_pairs_bought_l912_91270

def shirt_price : ℕ := 5
def sandal_price : ℕ := 3
def num_shirts : ℕ := 10
def total_paid : ℕ := 100
def change_received : ℕ := 41

theorem sandal_pairs_bought : ℕ := by
  sorry

end NUMINAMATH_CALUDE_sandal_pairs_bought_l912_91270


namespace NUMINAMATH_CALUDE_sum_of_coefficients_factorization_l912_91274

theorem sum_of_coefficients_factorization (x y : ℝ) : 
  ∃ (a b c d e f g h j k : ℤ),
    27 * x^6 - 512 * y^6 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2) ∧
    a + b + c + d + e + f + g + h + j + k = 55 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_factorization_l912_91274


namespace NUMINAMATH_CALUDE_centroid_of_S_l912_91266

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 ≤ p.2 ∧ p.2 ≤ abs p.1 + 3 ∧ p.2 ≤ 4}

-- Define the centroid of a set
def centroid (T : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Theorem statement
theorem centroid_of_S :
  centroid S = (0, 13/5) := by sorry

end NUMINAMATH_CALUDE_centroid_of_S_l912_91266


namespace NUMINAMATH_CALUDE_smallest_number_l912_91230

theorem smallest_number (a b c d e : ℝ) 
  (ha : a = 0.997) 
  (hb : b = 0.979) 
  (hc : c = 0.999) 
  (hd : d = 0.9797) 
  (he : e = 0.9709) : 
  e ≤ a ∧ e ≤ b ∧ e ≤ c ∧ e ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l912_91230


namespace NUMINAMATH_CALUDE_other_number_l912_91291

theorem other_number (x : ℝ) : 
  0.5 > x ∧ 0.5 - x = 0.16666666666666669 → x = 0.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_other_number_l912_91291


namespace NUMINAMATH_CALUDE_infinitely_many_primary_triplets_l912_91242

/-- A primary triplet is a triplet of positive integers (x, y, z) satisfying
    x, y, z > 1 and x^3 - yz^3 = 2021, where at least two of x, y, z are prime numbers. -/
def PrimaryTriplet (x y z : ℕ) : Prop :=
  x > 1 ∧ y > 1 ∧ z > 1 ∧
  x^3 - y*z^3 = 2021 ∧
  (Nat.Prime x ∧ Nat.Prime y) ∨ (Nat.Prime x ∧ Nat.Prime z) ∨ (Nat.Prime y ∧ Nat.Prime z)

/-- There exist infinitely many primary triplets. -/
theorem infinitely_many_primary_triplets :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ∃ x y z : ℕ, PrimaryTriplet x y z :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primary_triplets_l912_91242


namespace NUMINAMATH_CALUDE_max_value_of_function_l912_91259

theorem max_value_of_function (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  (2*x^2 - 2*x + 3) / (x^2 - x + 1) ≤ 10/3 ∧
  ∃ y : ℝ, (2*y^2 - 2*y + 3) / (y^2 - y + 1) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l912_91259


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l912_91203

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31/13 - (1/13) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l912_91203


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l912_91293

theorem unique_five_digit_number : ∃! n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧ 
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧
  (∀ i, 0 ≤ i ∧ i < 5 → (n / 10^i) % 10 ≠ 0) ∧
  ((n % 1000) = 7 * (n / 100)) ∧
  n = 12946 := by
sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l912_91293


namespace NUMINAMATH_CALUDE_may_profit_max_profit_l912_91263

-- Define the profit function
def profit (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 6 then 12 * x + 28
  else if 6 < x ∧ x ≤ 12 then 200 - 14 * x
  else 0

-- Theorem for May's profit
theorem may_profit : profit 5 = 88 := by sorry

-- Theorem for maximum profit
theorem max_profit :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 12 → profit x ≤ profit 7 ∧ profit 7 = 102 := by sorry

end NUMINAMATH_CALUDE_may_profit_max_profit_l912_91263


namespace NUMINAMATH_CALUDE_hexagon_angle_problem_l912_91231

/-- Given a hexagon with specific angle conditions, prove that the unknown angle is 25 degrees. -/
theorem hexagon_angle_problem (a b c d e x : ℝ) : 
  -- Sum of interior angles of a hexagon
  a + b + c + d + e + x = (6 - 2) * 180 →
  -- Sum of five known angles
  a + b + c + d + e = 100 →
  -- Two adjacent angles are 75° each
  75 + x + 75 = 360 →
  -- Conclusion: x is 25°
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_problem_l912_91231


namespace NUMINAMATH_CALUDE_train_length_l912_91227

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) : 
  speed_kmph = 18 → time_sec = 5 → (speed_kmph * 1000 / 3600) * time_sec = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l912_91227


namespace NUMINAMATH_CALUDE_may_greatest_drop_l912_91258

/-- Represents the months of the year --/
inductive Month
| january
| february
| march
| april
| may
| june

/-- Price change for a given month --/
def price_change : Month → ℝ
| Month.january  => -1.00
| Month.february => 3.50
| Month.march    => -3.00
| Month.april    => 4.00
| Month.may      => -5.00
| Month.june     => 2.00

/-- Returns true if the price change is negative (a drop) --/
def is_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- The month with the greatest price drop --/
def greatest_drop : Month :=
  Month.may

theorem may_greatest_drop :
  ∀ m : Month, is_price_drop m → price_change greatest_drop ≤ price_change m :=
by sorry

end NUMINAMATH_CALUDE_may_greatest_drop_l912_91258


namespace NUMINAMATH_CALUDE_sqrt_product_equals_240_l912_91235

theorem sqrt_product_equals_240 : Real.sqrt 128 * Real.sqrt 50 * (27 ^ (1/3 : ℝ)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_240_l912_91235


namespace NUMINAMATH_CALUDE_k_squared_minus_3k_minus_4_l912_91232

theorem k_squared_minus_3k_minus_4 (a b c d k : ℝ) :
  (2 * a / (b + c + d) = k) ∧
  (2 * b / (a + c + d) = k) ∧
  (2 * c / (a + b + d) = k) ∧
  (2 * d / (a + b + c) = k) →
  (k^2 - 3*k - 4 = -50/9) ∨ (k^2 - 3*k - 4 = 6) :=
by sorry

end NUMINAMATH_CALUDE_k_squared_minus_3k_minus_4_l912_91232


namespace NUMINAMATH_CALUDE_centroid_sum_l912_91212

def vertex1 : Fin 3 → ℚ := ![9, 2, -1]
def vertex2 : Fin 3 → ℚ := ![5, -2, 3]
def vertex3 : Fin 3 → ℚ := ![1, 6, 5]

def centroid (v1 v2 v3 : Fin 3 → ℚ) : Fin 3 → ℚ :=
  fun i => (v1 i + v2 i + v3 i) / 3

theorem centroid_sum :
  (centroid vertex1 vertex2 vertex3 0 +
   centroid vertex1 vertex2 vertex3 1 +
   centroid vertex1 vertex2 vertex3 2) = 28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_centroid_sum_l912_91212


namespace NUMINAMATH_CALUDE_range_of_a_l912_91208

/-- Given functions f and g, prove the range of a -/
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, |2 * x₁ - a| + |2 * x₁ + 3| = |x₂ - 1| + 2) →
  (a ≥ -1 ∨ a ≤ -5) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l912_91208


namespace NUMINAMATH_CALUDE_class_size_problem_l912_91268

/-- Given classes A, B, and C with the following properties:
  * Class A is twice as big as Class B
  * Class A is a third the size of Class C
  * Class B has 20 people
  Prove that Class C has 120 people -/
theorem class_size_problem (class_A class_B class_C : ℕ) : 
  class_A = 2 * class_B →
  class_A = class_C / 3 →
  class_B = 20 →
  class_C = 120 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l912_91268


namespace NUMINAMATH_CALUDE_circle_properties_l912_91247

/-- Given a circle C with equation x^2 + 8x - 2y = 1 - y^2, 
    prove that its center is (-4, 1), its radius is 3√2, 
    and the sum of its center coordinates and radius is -3 + 3√2 -/
theorem circle_properties : 
  ∃ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + 8*x - 2*y = 1 - y^2) ∧
    center = (-4, 1) ∧
    radius = 3 * Real.sqrt 2 ∧
    center.1 + center.2 + radius = -3 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l912_91247


namespace NUMINAMATH_CALUDE_regular_soda_bottles_count_l912_91265

/-- The number of regular soda bottles in a grocery store -/
def regular_soda_bottles : ℕ := 30

/-- The total number of bottles in the store -/
def total_bottles : ℕ := 38

/-- The number of diet soda bottles in the store -/
def diet_soda_bottles : ℕ := 8

/-- Theorem stating that the number of regular soda bottles is correct -/
theorem regular_soda_bottles_count : 
  regular_soda_bottles = total_bottles - diet_soda_bottles :=
by sorry

end NUMINAMATH_CALUDE_regular_soda_bottles_count_l912_91265


namespace NUMINAMATH_CALUDE_total_blue_balloons_l912_91214

theorem total_blue_balloons (joan sally jessica : ℕ) 
  (h1 : joan = 9) 
  (h2 : sally = 5) 
  (h3 : jessica = 2) : 
  joan + sally + jessica = 16 := by
sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l912_91214


namespace NUMINAMATH_CALUDE_rectangle_area_l912_91211

theorem rectangle_area (x : ℝ) (w : ℝ) (h : w > 0) : 
  (3 * w)^2 + w^2 = x^2 → 3 * w^2 = 3 * x^2 / 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l912_91211


namespace NUMINAMATH_CALUDE_fixed_point_sets_l912_91243

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | f a b x = x}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = x}

-- Theorem statement
theorem fixed_point_sets (a b : ℝ) :
  A a b = {-1, 3} →
  B a b = {-Real.sqrt 3, -1, Real.sqrt 3, 3} ∧ A a b ⊆ B a b := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_sets_l912_91243


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l912_91271

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℝ) 
  (error1 error2 error3 error4 error5 : ℝ) : 
  n = 70 →
  initial_mean = 350 →
  error1 = 215.5 - 195.5 →
  error2 = -30 - 30 →
  error3 = 720.8 - 670.8 →
  error4 = -95.4 - (-45.4) →
  error5 = 124.2 - 114.2 →
  (n : ℝ) * initial_mean + (error1 + error2 + error3 + error4 + error5) = n * 349.57 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l912_91271


namespace NUMINAMATH_CALUDE_train_crossing_time_l912_91295

/-- Time for a train to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 50 → 
  train_speed_kmh = 360 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l912_91295


namespace NUMINAMATH_CALUDE_water_content_in_fresh_grapes_fresh_grapes_water_percentage_l912_91246

theorem water_content_in_fresh_grapes 
  (dried_water_content : Real) 
  (fresh_weight : Real) 
  (dried_weight : Real) : Real :=
  let solid_content := dried_weight * (1 - dried_water_content)
  let water_content := fresh_weight - solid_content
  let water_percentage := (water_content / fresh_weight) * 100
  90

theorem fresh_grapes_water_percentage :
  let dried_water_content := 0.20
  let fresh_weight := 10
  let dried_weight := 1.25
  water_content_in_fresh_grapes dried_water_content fresh_weight dried_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_water_content_in_fresh_grapes_fresh_grapes_water_percentage_l912_91246


namespace NUMINAMATH_CALUDE_tile_border_ratio_l912_91220

/-- Proves that for a square tiled surface with n^2 tiles, each tile of side length s,
    surrounded by a border of width d, if n = 30 and the tiles cover 81% of the total area,
    then d/s = 1/18. -/
theorem tile_border_ratio (n s d : ℝ) (h1 : n = 30) 
    (h2 : (n^2 * s^2) / ((n*s + 2*n*d)^2) = 0.81) : d/s = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l912_91220


namespace NUMINAMATH_CALUDE_rhombus_count_in_triangle_l912_91216

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ
  positive : sideLength > 0

/-- Represents a rhombus composed of smaller triangles -/
structure Rhombus where
  smallTriangles : ℕ

/-- The number of rhombuses in a large equilateral triangle -/
def countRhombuses (largeTriangle : EquilateralTriangle) (smallTriangleSideLength : ℝ) (rhombusSize : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem rhombus_count_in_triangle :
  let largeTriangle : EquilateralTriangle := ⟨10, by norm_num⟩
  let smallTriangleSideLength : ℝ := 1
  let rhombusSize : ℕ := 8
  countRhombuses largeTriangle smallTriangleSideLength rhombusSize = 84 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_count_in_triangle_l912_91216


namespace NUMINAMATH_CALUDE_problem_statement_l912_91222

theorem problem_statement (p q r : ℝ) 
  (h1 : p * r / (p + q) + q * p / (q + r) + r * q / (r + p) = -8)
  (h2 : q * r / (p + q) + r * p / (q + r) + p * q / (r + p) = 9) :
  q / (p + q) + r / (q + r) + p / (r + p) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l912_91222


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l912_91215

theorem pure_imaginary_solutions : 
  let f (x : ℂ) := x^6 - 6*x^5 + 15*x^4 - 20*x^3 + 27*x^2 - 18*x - 8
  let y := Real.sqrt ((Real.sqrt 52 - 5) / 3)
  f (Complex.I * y) = 0 ∧ f (-Complex.I * y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l912_91215


namespace NUMINAMATH_CALUDE_line_equation_length_BC_l912_91250

-- Problem 1
def projection_point : ℝ × ℝ := (2, -1)

theorem line_equation (l : Set (ℝ × ℝ)) (h : projection_point ∈ l) :
  l = {(x, y) | 2*x - y - 5 = 0} := by sorry

-- Problem 2
def point_A : ℝ × ℝ := (4, -1)
def midpoint_AB : ℝ × ℝ := (3, 2)
def centroid : ℝ × ℝ := (4, 2)

theorem length_BC :
  let B : ℝ × ℝ := (2*midpoint_AB.1 - point_A.1, 2*midpoint_AB.2 - point_A.2)
  let C : ℝ × ℝ := (3*centroid.1 - point_A.1 - B.1, 3*centroid.2 - point_A.2 - B.2)
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_line_equation_length_BC_l912_91250


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l912_91207

-- Define the polynomials
def p (x : ℝ) : ℝ := 7 * x^2 + 5 * x + 3
def q (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + 1

-- State the theorem
theorem polynomial_product_expansion :
  ∀ x : ℝ, p x * q x = 21 * x^5 + 29 * x^4 + 19 * x^3 + 13 * x^2 + 5 * x + 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l912_91207


namespace NUMINAMATH_CALUDE_plough_time_for_A_l912_91269

/-- Given two workers A and B who can plough a field together in 10 hours,
    and B alone takes 30 hours, prove that A alone would take 15 hours. -/
theorem plough_time_for_A (time_together time_B : ℝ) (time_together_pos : time_together > 0)
    (time_B_pos : time_B > 0) (h1 : time_together = 10) (h2 : time_B = 30) :
    ∃ time_A : ℝ, time_A > 0 ∧ 1 / time_A + 1 / time_B = 1 / time_together ∧ time_A = 15 := by
  sorry

end NUMINAMATH_CALUDE_plough_time_for_A_l912_91269


namespace NUMINAMATH_CALUDE_deepak_present_age_l912_91228

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age -/
theorem deepak_present_age 
  (ratio_rahul : ℕ) 
  (ratio_deepak : ℕ) 
  (rahul_future_age : ℕ) 
  (years_difference : ℕ) : 
  ratio_rahul = 4 → 
  ratio_deepak = 3 → 
  rahul_future_age = 22 → 
  years_difference = 6 → 
  (ratio_deepak * (rahul_future_age - years_difference)) / ratio_rahul = 12 := by
  sorry

end NUMINAMATH_CALUDE_deepak_present_age_l912_91228


namespace NUMINAMATH_CALUDE_roots_of_equation_l912_91260

theorem roots_of_equation : ∃ x₁ x₂ : ℝ,
  (88 * (x₁ - 2)^2 = 95) ∧
  (88 * (x₂ - 2)^2 = 95) ∧
  (x₁ < 1) ∧
  (x₂ > 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l912_91260


namespace NUMINAMATH_CALUDE_st_length_l912_91262

/-- Triangle PQR with given side lengths and points S, T on its sides --/
structure TrianglePQR where
  /-- Side length PQ --/
  pq : ℝ
  /-- Side length PR --/
  pr : ℝ
  /-- Side length QR --/
  qr : ℝ
  /-- Point S on side PQ --/
  s : ℝ
  /-- Point T on side PR --/
  t : ℝ
  /-- PQ = 13 --/
  pq_eq : pq = 13
  /-- PR = 14 --/
  pr_eq : pr = 14
  /-- QR = 15 --/
  qr_eq : qr = 15
  /-- S is between P and Q --/
  s_between : 0 ≤ s ∧ s ≤ pq
  /-- T is between P and R --/
  t_between : 0 ≤ t ∧ t ≤ pr
  /-- ST is parallel to QR --/
  st_parallel_qr : (s / pq) = (t / pr)
  /-- ST contains the incenter of triangle PQR --/
  st_contains_incenter : ∃ (k : ℝ), 0 < k ∧ k < 1 ∧
    k * s / (1 - k) * (pq - s) = pr / (pr + qr) ∧
    k * t / (1 - k) * (pr - t) = pq / (pq + qr)

/-- The main theorem --/
theorem st_length (tri : TrianglePQR) : (tri.s * tri.pr + tri.t * tri.pq) / (tri.pq + tri.pr) = 135 / 14 := by
  sorry

end NUMINAMATH_CALUDE_st_length_l912_91262


namespace NUMINAMATH_CALUDE_basketball_tournament_handshakes_l912_91256

/-- Calculates the total number of handshakes in a basketball tournament --/
def total_handshakes (num_teams : ℕ) (players_per_team : ℕ) (num_referees : ℕ) : ℕ :=
  let total_players := num_teams * players_per_team
  let player_handshakes := (total_players * (total_players - players_per_team)) / 2
  let referee_handshakes := total_players * num_referees
  player_handshakes + referee_handshakes

/-- Theorem stating the total number of handshakes in the specific basketball tournament scenario --/
theorem basketball_tournament_handshakes :
  total_handshakes 3 5 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tournament_handshakes_l912_91256
