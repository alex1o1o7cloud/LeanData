import Mathlib

namespace NUMINAMATH_CALUDE_emmanuel_regular_plan_cost_l2845_284598

/-- Calculates the regular plan cost given the stay duration, international data cost per day, and total charges. -/
def regular_plan_cost (stay_duration : ℕ) (intl_data_cost_per_day : ℚ) (total_charges : ℚ) : ℚ :=
  total_charges - (stay_duration : ℚ) * intl_data_cost_per_day

/-- Proves that Emmanuel's regular plan cost is $175 given the problem conditions. -/
theorem emmanuel_regular_plan_cost :
  regular_plan_cost 10 (350/100) 210 = 175 := by
  sorry

end NUMINAMATH_CALUDE_emmanuel_regular_plan_cost_l2845_284598


namespace NUMINAMATH_CALUDE_floor_greater_than_fraction_l2845_284581

theorem floor_greater_than_fraction (a : ℝ) (n : ℤ) 
  (h1 : a ≥ 1) (h2 : 0 ≤ n) (h3 : n ≤ a) :
  Int.floor a > (n / (n + 1 : ℝ)) * a := by
  sorry

end NUMINAMATH_CALUDE_floor_greater_than_fraction_l2845_284581


namespace NUMINAMATH_CALUDE_right_triangle_vector_property_l2845_284520

-- Define a right-angled triangle ABC
structure RightTriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angled : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the theorem
theorem right_triangle_vector_property (t : RightTriangleABC) (x : ℝ) 
  (h1 : t.C.1 - t.A.1 = 2 ∧ t.C.2 - t.A.2 = 4)
  (h2 : t.C.1 - t.B.1 = -6 ∧ t.C.2 - t.B.2 = x) :
  x = 3 := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_right_triangle_vector_property_l2845_284520


namespace NUMINAMATH_CALUDE_purchase_ways_l2845_284552

/-- The number of oreo flavors available -/
def oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def milk_flavors : ℕ := 3

/-- The total number of item options -/
def total_options : ℕ := oreo_flavors + milk_flavors

/-- The maximum number of items of the same flavor one person can order -/
def max_same_flavor : ℕ := 2

/-- The maximum number of milk flavors one person can order -/
def max_milk : ℕ := 1

/-- The total number of items they purchase collectively -/
def total_items : ℕ := 3

/-- Function to calculate the number of ways to choose k items from n options -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The main theorem: the number of ways Charlie and Delta can purchase exactly 3 items -/
theorem purchase_ways : 
  (choose total_options total_items) + 
  (choose total_options 2 * oreo_flavors) + 
  (choose total_options 1 * choose total_options 2) + 
  (choose total_options total_items) = 708 := by sorry

end NUMINAMATH_CALUDE_purchase_ways_l2845_284552


namespace NUMINAMATH_CALUDE_geometric_series_relation_l2845_284590

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 3/5. -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (c/d) / (1 - 1/d) = 3) :
    (c/(c+2*d)) / (1 - 1/(c+2*d)) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l2845_284590


namespace NUMINAMATH_CALUDE_number_equality_l2845_284567

theorem number_equality (x : ℚ) (h : (30 / 100) * x = (40 / 100) * 50) : x = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2845_284567


namespace NUMINAMATH_CALUDE_sum_of_coordinates_is_50_l2845_284526

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions for the points
def satisfiesConditions (p : Point) : Prop :=
  (p.y = 18 ∨ p.y = 12) ∧ 
  (p.x - 5)^2 + (p.y - 15)^2 = 10^2

-- Define the set of points satisfying the conditions
def validPoints : Set Point :=
  {p : Point | satisfiesConditions p}

-- Theorem statement
theorem sum_of_coordinates_is_50 :
  ∃ (a b c d : Point),
    a ∈ validPoints ∧ b ∈ validPoints ∧ c ∈ validPoints ∧ d ∈ validPoints ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a.x + a.y + b.x + b.y + c.x + c.y + d.x + d.y = 50 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_is_50_l2845_284526


namespace NUMINAMATH_CALUDE_B_largest_at_200_l2845_284508

/-- B_k is defined as the binomial coefficient (800 choose k) multiplied by 0.3^k -/
def B (k : ℕ) : ℝ := (Nat.choose 800 k : ℝ) * (0.3 ^ k)

/-- Theorem stating that B_k is largest when k = 200 -/
theorem B_largest_at_200 : ∀ k : ℕ, k ≤ 800 → B k ≤ B 200 :=
sorry

end NUMINAMATH_CALUDE_B_largest_at_200_l2845_284508


namespace NUMINAMATH_CALUDE_solution_for_m_eq_one_solution_satisfies_equation_l2845_284506

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  2 * x + y = 4 - m ∧ x - 2 * y = 3 * m

-- Statement 1: When m = 1, the solution is x = 9/5 and y = -3/5
theorem solution_for_m_eq_one :
  system (9/5) (-3/5) 1 := by sorry

-- Statement 2: For any m, the solution satisfies 3x - y = 4 + 2m
theorem solution_satisfies_equation (m : ℝ) (x y : ℝ) :
  system x y m → 3 * x - y = 4 + 2 * m := by sorry

end NUMINAMATH_CALUDE_solution_for_m_eq_one_solution_satisfies_equation_l2845_284506


namespace NUMINAMATH_CALUDE_altitudes_intersect_at_one_point_l2845_284597

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of an acute triangle --/
def isAcute (t : Triangle) : Prop := sorry

/-- Definition of an altitude of a triangle --/
def altitude (t : Triangle) (v : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The three altitudes of an acute triangle intersect at one point --/
theorem altitudes_intersect_at_one_point (t : Triangle) (h : isAcute t) :
  ∃! p : ℝ × ℝ, p ∈ altitude t t.A ∩ altitude t t.B ∩ altitude t t.C :=
sorry

end NUMINAMATH_CALUDE_altitudes_intersect_at_one_point_l2845_284597


namespace NUMINAMATH_CALUDE_village_population_l2845_284547

theorem village_population (P : ℝ) : 
  (P > 0) →
  (0.85 * (0.9 * P) = 3213) →
  P = 4200 := by
sorry

end NUMINAMATH_CALUDE_village_population_l2845_284547


namespace NUMINAMATH_CALUDE_two_digit_three_digit_sum_l2845_284550

theorem two_digit_three_digit_sum : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 100 ≤ y ∧ y < 1000 ∧ 
  100000 * x + y = 7 * x * y ∧ 
  x + y = 18 := by
sorry

end NUMINAMATH_CALUDE_two_digit_three_digit_sum_l2845_284550


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2845_284517

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I + 1) / (Complex.I - 1) = Complex.mk a b → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2845_284517


namespace NUMINAMATH_CALUDE_jeffrey_farm_chickens_l2845_284501

/-- Calculates the total number of chickens on Jeffrey's farm -/
def total_chickens (num_hens : ℕ) (hen_to_rooster_ratio : ℕ) (chicks_per_hen : ℕ) : ℕ :=
  let num_roosters := num_hens / hen_to_rooster_ratio
  let num_chicks := num_hens * chicks_per_hen
  num_hens + num_roosters + num_chicks

/-- Proves that the total number of chickens on Jeffrey's farm is 76 -/
theorem jeffrey_farm_chickens :
  total_chickens 12 3 5 = 76 := by
  sorry

end NUMINAMATH_CALUDE_jeffrey_farm_chickens_l2845_284501


namespace NUMINAMATH_CALUDE_second_number_value_l2845_284509

theorem second_number_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b = 1.2 * a) (h4 : a / b = 5 / 6) : b = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l2845_284509


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2845_284566

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  let l1 : Line := { a := 2, b := -1, c := -1 }  -- 2x - y - 1 = 0
  let l2 : Line := { a := 2, b := -1, c := 0 }   -- 2x - y = 0
  parallel l1 l2 ∧ point_on_line 1 2 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2845_284566


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2845_284561

theorem sum_of_squares_16_to_30 :
  let sum_squares : (n : ℕ) → ℕ := λ n => n * (n + 1) * (2 * n + 1) / 6
  let sum_1_to_15 := 1280
  let sum_1_to_30 := sum_squares 30
  sum_1_to_30 - sum_1_to_15 = 8215 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2845_284561


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l2845_284542

theorem sum_of_odd_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (∀ x, (2*x + 1)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₃ + a₅ = 122 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l2845_284542


namespace NUMINAMATH_CALUDE_tens_digit_of_11_power_12_power_13_l2845_284569

-- Define the exponentiation operation
def power (base : ℕ) (exponent : ℕ) : ℕ := base ^ exponent

-- Define a function to get the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Theorem statement
theorem tens_digit_of_11_power_12_power_13 :
  tens_digit (power 11 (power 12 13)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_11_power_12_power_13_l2845_284569


namespace NUMINAMATH_CALUDE_minimize_sum_of_squares_l2845_284528

theorem minimize_sum_of_squares (s : ℝ) (hs : s > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = s ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = s → x^2 + y^2 ≤ a^2 + b^2 ∧
  x^2 + y^2 = s^2 / 2 ∧ x = s / 2 ∧ y = s / 2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_sum_of_squares_l2845_284528


namespace NUMINAMATH_CALUDE_marcy_cat_time_l2845_284537

/-- Given that Marcy spends 12 minutes petting her cat and 1/3 of that time combing it,
    prove that she spends 16 minutes in total with her cat. -/
theorem marcy_cat_time (petting_time : ℝ) (combing_ratio : ℝ) : 
  petting_time = 12 → combing_ratio = 1/3 → petting_time + combing_ratio * petting_time = 16 := by
sorry

end NUMINAMATH_CALUDE_marcy_cat_time_l2845_284537


namespace NUMINAMATH_CALUDE_modulus_of_specific_complex_l2845_284512

open Complex

theorem modulus_of_specific_complex : ‖(1 - I) / I‖ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_specific_complex_l2845_284512


namespace NUMINAMATH_CALUDE_symmetric_line_passes_through_fixed_point_l2845_284591

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The point of symmetry -/
def symmetryPoint : ℝ × ℝ := (2, 1)

/-- Line l₁ defined as y = k(x-4) -/
def l₁ (k : ℝ) : Line :=
  { slope := k, point := (4, 0) }

/-- Line l₂ symmetric to l₁ with respect to the symmetry point -/
def l₂ (k : ℝ) : Line :=
  sorry -- definition omitted as it's not directly given in the problem

theorem symmetric_line_passes_through_fixed_point (k : ℝ) :
  (0, 2) ∈ {p : ℝ × ℝ | p.2 = (l₂ k).slope * (p.1 - (l₂ k).point.1) + (l₂ k).point.2} :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_passes_through_fixed_point_l2845_284591


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2845_284574

/-- Given vectors a and b, prove that if k*a + b is parallel to a - 3*b, then k = -1/3 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : ∃ (t : ℝ), t ≠ 0 ∧ (k • a + b) = t • (a - 3 • b)) :
  k = -1/3 := by
  sorry

#check parallel_vectors_k_value

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2845_284574


namespace NUMINAMATH_CALUDE_smallest_representable_l2845_284522

/-- The representation function for k -/
def representation (n m : ℕ+) : ℤ := 19^(n:ℕ) - 5^(m:ℕ)

/-- The property that k is representable -/
def is_representable (k : ℕ) : Prop :=
  ∃ (n m : ℕ+), representation n m = k

/-- The main theorem statement -/
theorem smallest_representable : 
  (is_representable 14) ∧ (∀ k : ℕ, 0 < k ∧ k < 14 → ¬(is_representable k)) := by
  sorry

#check smallest_representable

end NUMINAMATH_CALUDE_smallest_representable_l2845_284522


namespace NUMINAMATH_CALUDE_cookie_count_bounds_l2845_284588

/-- Represents the number of cookies in a package -/
inductive PackageSize
| small : PackageSize  -- 6 cookies
| large : PackageSize  -- 12 cookies

/-- Profit from selling a package -/
def profit : PackageSize → ℕ
| PackageSize.small => 4
| PackageSize.large => 9

/-- Number of cookies in a package -/
def cookiesInPackage : PackageSize → ℕ
| PackageSize.small => 6
| PackageSize.large => 12

/-- Total profit from selling packages -/
def totalProfit : ℕ → ℕ → ℕ := λ x y => x * profit PackageSize.large + y * profit PackageSize.small

/-- Total number of cookies in packages -/
def totalCookies : ℕ → ℕ → ℕ := λ x y => x * cookiesInPackage PackageSize.large + y * cookiesInPackage PackageSize.small

theorem cookie_count_bounds :
  ∃ (x_min y_min x_max y_max : ℕ),
    totalProfit x_min y_min = 219 ∧
    totalProfit x_max y_max = 219 ∧
    totalCookies x_min y_min = 294 ∧
    totalCookies x_max y_max = 324 ∧
    (∀ x y, totalProfit x y = 219 → totalCookies x y ≥ 294 ∧ totalCookies x y ≤ 324) :=
by sorry

end NUMINAMATH_CALUDE_cookie_count_bounds_l2845_284588


namespace NUMINAMATH_CALUDE_min_trucks_for_given_problem_l2845_284502

/-- Represents the problem of transporting crates with trucks -/
structure CrateTransportProblem where
  totalWeight : ℝ
  maxCrateWeight : ℝ
  truckCapacity : ℝ

/-- Calculates the minimum number of trucks required -/
def minTrucksRequired (problem : CrateTransportProblem) : ℕ :=
  sorry

/-- Theorem stating the minimum number of trucks required for the given problem -/
theorem min_trucks_for_given_problem :
  let problem : CrateTransportProblem := {
    totalWeight := 10,
    maxCrateWeight := 1,
    truckCapacity := 3
  }
  minTrucksRequired problem = 5 := by sorry

end NUMINAMATH_CALUDE_min_trucks_for_given_problem_l2845_284502


namespace NUMINAMATH_CALUDE_delta_theta_solution_l2845_284544

theorem delta_theta_solution :
  ∃ (Δ Θ : ℤ), 4 * 3 = Δ - 5 + Θ ∧ Θ = 14 ∧ Δ = 3 := by
  sorry

end NUMINAMATH_CALUDE_delta_theta_solution_l2845_284544


namespace NUMINAMATH_CALUDE_simplify_fraction_l2845_284539

theorem simplify_fraction (x y z : ℚ) (hx : x = 5) (hy : y = 2) (hz : z = 4) :
  (10 * x^2 * y^3 * z) / (15 * x * y^2 * z^2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2845_284539


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2845_284532

/-- Proves that a train of given length, traveling at a given speed, will take the calculated time to cross a bridge of given length. -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (bridge_length : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length = 100)
  (h2 : bridge_length = 200)
  (h3 : train_speed_kmph = 36)
  : (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 30 :=
by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2845_284532


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2845_284595

theorem simplify_trig_expression (α : Real) (h : α ∈ Set.Ioo (π / 2) π) :
  (Real.sqrt (1 - 2 * Real.sin α * Real.cos α)) / (Real.sin α + Real.sqrt (1 - Real.sin α ^ 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2845_284595


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2845_284516

/-- Given that a sum of money becomes 7/6 of itself in 2 years under simple interest,
    prove that the rate of interest per annum is 100/12. -/
theorem simple_interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  ∃ R : ℝ, R = 100 / 12 ∧ P * (1 + R * 2 / 100) = 7 / 6 * P :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2845_284516


namespace NUMINAMATH_CALUDE_distance_rowed_is_90km_l2845_284557

/-- Calculates the distance rowed downstream given the rowing speed in still water,
    the stream speed, and the time spent rowing. -/
def distance_rowed_downstream (rowing_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (rowing_speed + stream_speed) * time

/-- Theorem stating that given the specific conditions of the problem,
    the distance rowed downstream is 90 km. -/
theorem distance_rowed_is_90km
  (rowing_speed : ℝ)
  (stream_speed : ℝ)
  (time : ℝ)
  (h1 : rowing_speed = 10)
  (h2 : stream_speed = 8)
  (h3 : time = 5) :
  distance_rowed_downstream rowing_speed stream_speed time = 90 := by
  sorry

#check distance_rowed_is_90km

end NUMINAMATH_CALUDE_distance_rowed_is_90km_l2845_284557


namespace NUMINAMATH_CALUDE_complex_power_trig_l2845_284568

theorem complex_power_trig : (2 * Complex.cos (π / 6) + 2 * Complex.I * Complex.sin (π / 6)) ^ 10 = 512 - 512 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_trig_l2845_284568


namespace NUMINAMATH_CALUDE_system_solution_sum_of_squares_l2845_284513

theorem system_solution_sum_of_squares (x y : ℝ) : 
  x * y = 6 → x^2 * y + x * y^2 + x + y = 63 → x^2 + y^2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_sum_of_squares_l2845_284513


namespace NUMINAMATH_CALUDE_unique_solution_iff_l2845_284551

/-- The function f(x) = x^2 + 2ax + 3a -/
def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 3*a

/-- The inequality |f(x)| ≤ 2 -/
def inequality (a x : ℝ) : Prop := |f a x| ≤ 2

/-- The theorem stating that the inequality has exactly one solution if and only if a = 1 or a = 2 -/
theorem unique_solution_iff (a : ℝ) : 
  (∃! x, inequality a x) ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_l2845_284551


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2845_284529

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (n = 626) ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (Real.sqrt (m : ℝ) - Real.sqrt ((m - 1) : ℝ) ≥ 0.02 ∨ 
     Real.sin (Real.pi / Real.sqrt (m : ℝ)) ≤ 0.5)) ∧
  (Real.sqrt (n : ℝ) - Real.sqrt ((n - 1) : ℝ) < 0.02) ∧
  (Real.sin (Real.pi / Real.sqrt (n : ℝ)) > 0.5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2845_284529


namespace NUMINAMATH_CALUDE_trains_at_start_after_2016_minutes_all_trains_at_start_after_2016_minutes_l2845_284556

/-- Represents a metro line with a given round trip time -/
structure MetroLine where
  roundTripTime : ℕ

/-- Represents the metro system of city N -/
structure MetroSystem where
  redLine : MetroLine
  blueLine : MetroLine
  greenLine : MetroLine

/-- Theorem stating that after 2016 minutes, all trains will be at their starting positions -/
theorem trains_at_start_after_2016_minutes (system : MetroSystem) 
  (h_red : system.redLine.roundTripTime = 14)
  (h_blue : system.blueLine.roundTripTime = 16)
  (h_green : system.greenLine.roundTripTime = 18) :
  2016 % system.redLine.roundTripTime = 0 ∧
  2016 % system.blueLine.roundTripTime = 0 ∧
  2016 % system.greenLine.roundTripTime = 0 := by
  sorry

/-- Function to check if a train is at its starting position after a given time -/
def isAtStartPosition (line : MetroLine) (time : ℕ) : Bool :=
  time % line.roundTripTime = 0

/-- Theorem stating that all trains are at their starting positions after 2016 minutes -/
theorem all_trains_at_start_after_2016_minutes (system : MetroSystem) 
  (h_red : system.redLine.roundTripTime = 14)
  (h_blue : system.blueLine.roundTripTime = 16)
  (h_green : system.greenLine.roundTripTime = 18) :
  isAtStartPosition system.redLine 2016 ∧
  isAtStartPosition system.blueLine 2016 ∧
  isAtStartPosition system.greenLine 2016 := by
  sorry

end NUMINAMATH_CALUDE_trains_at_start_after_2016_minutes_all_trains_at_start_after_2016_minutes_l2845_284556


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2845_284511

/-- An arithmetic sequence with sum S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ

/-- Theorem: For an arithmetic sequence with S_3 = 9 and S_6 = 27, S_9 = 54 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 3 = 9) 
  (h2 : a.S 6 = 27) : 
  a.S 9 = 54 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2845_284511


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2845_284523

/-- The remainder when (x^1001 - 1) is divided by (x^4 + x^3 + 2x^2 + x + 1) -/
def remainder1 (x : ℝ) : ℝ := x^2 * (1 - x)

/-- The remainder when (x^1001 - 1) is divided by (x^8 + x^6 + 2x^4 + x^2 + 1) -/
def remainder2 (x : ℝ) : ℝ := -2*x^7 - x^5 - 2*x^3 - 1

/-- The first divisor polynomial -/
def divisor1 (x : ℝ) : ℝ := x^4 + x^3 + 2*x^2 + x + 1

/-- The second divisor polynomial -/
def divisor2 (x : ℝ) : ℝ := x^8 + x^6 + 2*x^4 + x^2 + 1

/-- The dividend polynomial -/
def dividend (x : ℝ) : ℝ := x^1001 - 1

theorem polynomial_division_theorem :
  ∀ x : ℝ,
  ∃ q1 q2 : ℝ → ℝ,
  dividend x = q1 x * divisor1 x + remainder1 x ∧
  dividend x = q2 x * divisor2 x + remainder2 x :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2845_284523


namespace NUMINAMATH_CALUDE_blue_marbles_count_l2845_284514

theorem blue_marbles_count (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 20 →
  red = 9 →
  prob_red_or_white = 7/10 →
  ∃ (blue white : ℕ),
    blue + red + white = total ∧
    (red + white : ℚ) / total = prob_red_or_white ∧
    blue = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l2845_284514


namespace NUMINAMATH_CALUDE_set_propositions_equivalence_l2845_284576

theorem set_propositions_equivalence (A B : Set α) :
  (((A ∪ B ≠ B) → (A ∩ B ≠ A)) ∧
   ((A ∩ B ≠ A) → (A ∪ B ≠ B)) ∧
   ((A ∪ B = B) → (A ∩ B = A)) ∧
   ((A ∩ B = A) → (A ∪ B = B))) := by
  sorry

end NUMINAMATH_CALUDE_set_propositions_equivalence_l2845_284576


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00000428_l2845_284564

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_bounds : 1 ≤ |mantissa| ∧ |mantissa| < 10

/-- Conversion function from a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00000428 :
  toScientificNotation 0.00000428 = ScientificNotation.mk 4.28 (-6) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00000428_l2845_284564


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2845_284565

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → (l * w ≤ 100) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2845_284565


namespace NUMINAMATH_CALUDE_rectangle_discrepancy_exists_l2845_284573

/-- Represents a point in a 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle with sides parallel to the axes -/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The side length of the square -/
def squareSideLength : ℝ := 10^2019

/-- The total number of marked points -/
def totalPoints : ℕ := 10^4038

/-- A set of points marked in the square -/
def markedPoints : Set Point := sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.x2 - r.x1) * (r.y2 - r.y1)

/-- Counts the number of points inside a rectangle -/
def pointsInRectangle (r : Rectangle) (points : Set Point) : ℕ := sorry

/-- The main theorem to be proved -/
theorem rectangle_discrepancy_exists :
  ∃ (r : Rectangle),
    r.x1 ≥ 0 ∧ r.y1 ≥ 0 ∧ r.x2 ≤ squareSideLength ∧ r.y2 ≤ squareSideLength ∧
    |rectangleArea r - (pointsInRectangle r markedPoints : ℝ)| ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_discrepancy_exists_l2845_284573


namespace NUMINAMATH_CALUDE_distinct_points_on_curve_l2845_284531

theorem distinct_points_on_curve (e a b : ℝ) : 
  e > 0 →
  (a^2 + e^2 = 3 * e * a + 1) →
  (b^2 + e^2 = 3 * e * b + 1) →
  a ≠ b →
  |a - b| = Real.sqrt (5 * e^2 + 4) :=
by sorry

end NUMINAMATH_CALUDE_distinct_points_on_curve_l2845_284531


namespace NUMINAMATH_CALUDE_hexagon_coverage_percentage_l2845_284507

structure Tile :=
  (grid_size : Nat)
  (square_count : Nat)
  (hexagon_count : Nat)

def Region :=
  {t : Tile // t.grid_size = 4 ∧ t.square_count = 8 ∧ t.hexagon_count = 8}

theorem hexagon_coverage_percentage (r : Region) : 
  (r.val.hexagon_count : ℚ) / (r.val.grid_size^2 : ℚ) * 100 = 50 :=
sorry

end NUMINAMATH_CALUDE_hexagon_coverage_percentage_l2845_284507


namespace NUMINAMATH_CALUDE_pencils_remaining_l2845_284541

theorem pencils_remaining (initial_pencils : ℕ) (pencils_removed : ℕ) : 
  initial_pencils = 9 → pencils_removed = 4 → initial_pencils - pencils_removed = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remaining_l2845_284541


namespace NUMINAMATH_CALUDE_half_quarter_difference_l2845_284584

theorem half_quarter_difference (n : ℝ) (h : n = 8) : 0.5 * n - 0.25 * n = 2 := by
  sorry

end NUMINAMATH_CALUDE_half_quarter_difference_l2845_284584


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2845_284592

theorem rectangle_perimeter (width length : ℝ) (h1 : width = Real.sqrt 3) (h2 : length = Real.sqrt 6) :
  2 * (width + length) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2845_284592


namespace NUMINAMATH_CALUDE_yellow_purple_difference_l2845_284587

/-- Represents the composition of candies in a box of rainbow nerds -/
structure RainbowNerdsBox where
  purple : ℕ
  yellow : ℕ
  green : ℕ
  total : ℕ
  green_yellow_relation : green = yellow - 2
  total_sum : total = purple + yellow + green

/-- Theorem stating the difference between yellow and purple candies -/
theorem yellow_purple_difference (box : RainbowNerdsBox) 
  (h_purple : box.purple = 10) 
  (h_total : box.total = 36) : 
  box.yellow - box.purple = 4 := by
  sorry


end NUMINAMATH_CALUDE_yellow_purple_difference_l2845_284587


namespace NUMINAMATH_CALUDE_golden_ratio_logarithm_l2845_284533

theorem golden_ratio_logarithm (r s : ℝ) (hr : r > 0) (hs : s > 0) :
  (Real.log r / Real.log 4 = Real.log s / Real.log 18) ∧
  (Real.log s / Real.log 18 = Real.log (r + s) / Real.log 24) →
  s / r = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_golden_ratio_logarithm_l2845_284533


namespace NUMINAMATH_CALUDE_color_film_fraction_l2845_284596

theorem color_film_fraction (x y : ℝ) (h : x ≠ 0) : 
  let total_bw := 30 * x
  let total_color := 6 * y
  let selected_bw := (y / x) * (1 / 100) * total_bw
  let selected_color := total_color
  (selected_color / (selected_bw + selected_color)) = 20 / 21 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l2845_284596


namespace NUMINAMATH_CALUDE_leila_payment_l2845_284519

/-- The total cost of Leila's cake order --/
def total_cost (chocolate_cakes strawberry_cakes : ℕ) 
               (chocolate_price strawberry_price : ℚ) : ℚ :=
  chocolate_cakes * chocolate_price + strawberry_cakes * strawberry_price

/-- Theorem stating that Leila should pay $168 for her cake order --/
theorem leila_payment : 
  total_cost 3 6 12 22 = 168 := by sorry

end NUMINAMATH_CALUDE_leila_payment_l2845_284519


namespace NUMINAMATH_CALUDE_line_equation_from_triangle_l2845_284589

/-- Given a line passing through (-a, b) and intersecting the y-axis in the second quadrant,
    forming a triangle with area T and base ka along the x-axis, prove that
    the equation of the line is 2Tx - ka²y + ka²b + 2aT = 0 -/
theorem line_equation_from_triangle (a T k : ℝ) (b : ℝ) (hb : b ≠ 0) :
  ∃ (m c : ℝ), 
    (∀ x y, y = m * x + c ↔ 2 * T * x - k * a^2 * y + k * a^2 * b + 2 * a * T = 0) ∧
    m * (-a) + c = b ∧
    m > 0 ∧
    c > 0 ∧
    k > 0 ∧
    T = (1/2) * k * a * (c - b) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_triangle_l2845_284589


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2845_284510

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

/-- The main theorem -/
theorem collinear_points_sum (a b : ℝ) : 
  collinear (Point3D.mk 1 a b) (Point3D.mk a 2 3) (Point3D.mk a b 3) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2845_284510


namespace NUMINAMATH_CALUDE_min_blue_chips_l2845_284553

theorem min_blue_chips (r w b : ℕ) : 
  r ≥ 2 * w →
  r ≤ 2 * b / 3 →
  r + w ≥ 72 →
  ∀ b' : ℕ, (∃ r' w' : ℕ, r' ≥ 2 * w' ∧ r' ≤ 2 * b' / 3 ∧ r' + w' ≥ 72) → b' ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_min_blue_chips_l2845_284553


namespace NUMINAMATH_CALUDE_rob_baseball_cards_l2845_284554

theorem rob_baseball_cards (rob_total : ℕ) (rob_doubles : ℕ) (jess_doubles : ℕ) :
  rob_doubles = rob_total / 3 →
  jess_doubles = 5 * rob_doubles →
  jess_doubles = 40 →
  rob_total = 24 := by
sorry

end NUMINAMATH_CALUDE_rob_baseball_cards_l2845_284554


namespace NUMINAMATH_CALUDE_simplify_expression_l2845_284578

theorem simplify_expression :
  Real.sqrt 5 * 5^(1/2) + 18 / 3 * 4 - 8^(3/2) + 10 - 3^2 = 30 - 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2845_284578


namespace NUMINAMATH_CALUDE_ghee_mixture_original_quantity_l2845_284530

/-- Proves that the original quantity of ghee mixture was 10 kg -/
theorem ghee_mixture_original_quantity :
  ∀ x : ℝ,
  (0.6 * x = x - 0.4 * x) →  -- 60% pure ghee, 40% vanaspati
  (0.4 * x = 0.2 * (x + 10)) →  -- After adding 10 kg pure ghee, vanaspati becomes 20%
  x = 10 := by
  sorry

#check ghee_mixture_original_quantity

end NUMINAMATH_CALUDE_ghee_mixture_original_quantity_l2845_284530


namespace NUMINAMATH_CALUDE_intersection_theorem_l2845_284503

-- Define the curves
def curve1 (x y a : ℝ) : Prop := (x - 1)^2 + y^2 = a^2
def curve2 (x y a : ℝ) : Prop := y = x^2 - a

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve1 p.1 p.2 a ∧ curve2 p.1 p.2 a}

-- Define the condition for exactly three intersection points
def has_exactly_three_intersections (a : ℝ) : Prop :=
  ∃ p q r : ℝ × ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  intersection_points a = {p, q, r}

-- Theorem statement
theorem intersection_theorem :
  ∀ a : ℝ, has_exactly_three_intersections a ↔ 
  (a = (3 + Real.sqrt 5) / 2 ∨ a = (3 - Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2845_284503


namespace NUMINAMATH_CALUDE_cookie_distribution_l2845_284524

theorem cookie_distribution (chris kenny glenn terry dan anne : ℕ) : 
  chris = kenny / 3 →
  glenn = 4 * chris →
  glenn = 24 →
  terry = Int.floor (Real.sqrt (glenn : ℝ) + 3) →
  dan = 2 * (chris + kenny) →
  anne = kenny / 2 →
  anne ≥ 7 →
  kenny % 2 = 1 →
  ∀ k : ℕ, k % 2 = 1 ∧ k / 2 ≥ 7 → kenny ≤ k →
  chris = 6 ∧ 
  kenny = 18 ∧ 
  glenn = 24 ∧ 
  terry = 8 ∧ 
  dan = 48 ∧ 
  anne = 9 ∧
  chris + kenny + glenn + terry + dan + anne = 113 :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l2845_284524


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2845_284518

theorem rectangular_box_volume (x : ℕ) (h : x > 0) :
  let volume := x * (2 * x) * (5 * x)
  (volume = 80 ∨ volume = 250 ∨ volume = 500 ∨ volume = 1000 ∨ volume = 2000) →
  volume = 80 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2845_284518


namespace NUMINAMATH_CALUDE_probability_different_suits_pinochle_l2845_284540

/-- A pinochle deck of cards -/
structure PinochleDeck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h1 : cards = suits * cards_per_suit)

/-- The probability of drawing three cards of different suits from a pinochle deck -/
def probability_different_suits (deck : PinochleDeck) : Rat :=
  let remaining_after_first := deck.cards - 1
  let suitable_for_second := deck.cards - deck.cards_per_suit
  let remaining_after_second := deck.cards - 2
  let suitable_for_third := deck.cards - 2 * deck.cards_per_suit + 1
  (suitable_for_second : Rat) / remaining_after_first *
  (suitable_for_third : Rat) / remaining_after_second

theorem probability_different_suits_pinochle :
  let deck : PinochleDeck := ⟨48, 4, 12, rfl⟩
  probability_different_suits deck = 414 / 1081 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_suits_pinochle_l2845_284540


namespace NUMINAMATH_CALUDE_speedboat_drift_time_l2845_284546

/-- The time taken for a speedboat to drift along a river --/
theorem speedboat_drift_time 
  (L : ℝ) -- Total length of the river
  (v : ℝ) -- Speed of the speedboat in still water
  (u : ℝ) -- Speed of the water flow when reservoir is discharging
  (h1 : v = L / 150) -- Speed of boat in still water
  (h2 : v + u = L / 60) -- Speed of boat with water flow
  (h3 : u > 0) -- Water flow is positive
  : (L / 3) / u = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_speedboat_drift_time_l2845_284546


namespace NUMINAMATH_CALUDE_complement_of_M_l2845_284559

-- Define the set M
def M : Set ℝ := {x | x^2 - x > 0}

-- State the theorem
theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2845_284559


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l2845_284504

-- Part 1
theorem factorization_1 (x y : ℝ) : -x^5 * y^3 + x^3 * y^5 = -x^3 * y^3 * (x + y) * (x - y) := by sorry

-- Part 2
theorem factorization_2 (a : ℝ) : (a^2 + 1)^2 - 4 * a^2 = (a + 1)^2 * (a - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l2845_284504


namespace NUMINAMATH_CALUDE_least_four_digit_solution_l2845_284572

theorem least_four_digit_solution (x : ℕ) : 
  (x ≥ 1000 ∧ x < 10000) →
  (5 * x ≡ 15 [ZMOD 20]) →
  (3 * x + 7 ≡ 19 [ZMOD 8]) →
  (-3 * x + 2 ≡ x [ZMOD 14]) →
  (∀ y : ℕ, y ≥ 1000 ∧ y < x →
    ¬(5 * y ≡ 15 [ZMOD 20] ∧
      3 * y + 7 ≡ 19 [ZMOD 8] ∧
      -3 * y + 2 ≡ y [ZMOD 14])) →
  x = 1032 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_solution_l2845_284572


namespace NUMINAMATH_CALUDE_mirka_has_more_pears_l2845_284500

-- Define the number of pears in the bowl at the start
def initial_pears : ℕ := 14

-- Define Ivan's strategy
def ivan_takes : ℕ := 2

-- Define Mirka's strategy
def mirka_takes (remaining : ℕ) : ℕ := remaining / 2

-- Define the sequence of pear-taking
def pear_sequence (pears : ℕ) : ℕ × ℕ :=
  let after_ivan1 := pears - ivan_takes
  let after_mirka1 := after_ivan1 - mirka_takes after_ivan1
  let after_ivan2 := after_mirka1 - ivan_takes
  let after_mirka2 := after_ivan2 - mirka_takes after_ivan2
  let after_ivan3 := after_mirka2 - ivan_takes
  (3 * ivan_takes, mirka_takes after_ivan1 + mirka_takes after_ivan2)

theorem mirka_has_more_pears :
  let (ivan_total, mirka_total) := pear_sequence initial_pears
  mirka_total = ivan_total + 2 := by sorry

end NUMINAMATH_CALUDE_mirka_has_more_pears_l2845_284500


namespace NUMINAMATH_CALUDE_square_sum_of_solution_l2845_284525

theorem square_sum_of_solution (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 75) : 
  x^2 + y^2 = 3205 / 121 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_solution_l2845_284525


namespace NUMINAMATH_CALUDE_dobarulho_solutions_l2845_284583

def is_dobarulho (A B C D : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 8 ∧
  1 ≤ B ∧ B ≤ 9 ∧
  1 ≤ C ∧ C ≤ 9 ∧
  D > 1 ∧
  D ∣ (100*A + 10*B + C) ∧
  D ∣ (100*B + 10*C + A) ∧
  D ∣ (100*C + 10*A + B) ∧
  D ∣ (100*(A+1) + 10*C + B) ∧
  D ∣ (100*C + 10*B + (A+1)) ∧
  D ∣ (100*B + 10*(A+1) + C)

theorem dobarulho_solutions :
  ∀ A B C D : ℕ, is_dobarulho A B C D ↔ 
    ((A = 3 ∧ B = 7 ∧ C = 0 ∧ D = 37) ∨
     (A = 4 ∧ B = 8 ∧ C = 1 ∧ D = 37) ∨
     (A = 5 ∧ B = 9 ∧ C = 2 ∧ D = 37)) :=
by sorry

end NUMINAMATH_CALUDE_dobarulho_solutions_l2845_284583


namespace NUMINAMATH_CALUDE_solve_for_t_l2845_284527

theorem solve_for_t (s t : ℝ) (eq1 : 7 * s + 3 * t = 84) (eq2 : s = t - 3) : t = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l2845_284527


namespace NUMINAMATH_CALUDE_michael_money_left_l2845_284534

-- Define the initial amount and expenses
def initial_amount : ℕ := 100
def snack_expense : ℕ := 20
def game_expense : ℕ := 5

-- Define the ride expense as a function of snack expense
def ride_expense : ℕ := 3 * snack_expense

-- Define the total expense
def total_expense : ℕ := snack_expense + ride_expense + game_expense

-- Theorem to prove
theorem michael_money_left : initial_amount - total_expense = 15 := by
  sorry

end NUMINAMATH_CALUDE_michael_money_left_l2845_284534


namespace NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_factorization_problem3_factorization_problem4_l2845_284505

-- Problem 1
theorem factorization_problem1 (x y : ℝ) : 
  8 * x^2 + 26 * x * y - 15 * y^2 = (2 * x - y) * (4 * x + 15 * y) := by sorry

-- Problem 2
theorem factorization_problem2 (x y : ℝ) : 
  x^6 - y^6 - 2 * x^3 + 1 = (x^3 - y^3 - 1) * (x^3 + y^3 - 1) := by sorry

-- Problem 3
theorem factorization_problem3 (a b c : ℝ) : 
  a^3 + a^2 * c + b^2 * c - a * b * c + b^3 = (a + b + c) * (a^2 - a * b + b^2) := by sorry

-- Problem 4
theorem factorization_problem4 (x : ℝ) : 
  x^3 - 11 * x^2 + 31 * x - 21 = (x - 1) * (x - 3) * (x - 7) := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_factorization_problem3_factorization_problem4_l2845_284505


namespace NUMINAMATH_CALUDE_rachels_apple_tree_l2845_284586

theorem rachels_apple_tree (initial : ℕ) : 
  (initial - 2 + 3 = 5) → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachels_apple_tree_l2845_284586


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2845_284571

/-- The equation has exactly one real solution if and only if b < -4 -/
theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0) ↔ b < -4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2845_284571


namespace NUMINAMATH_CALUDE_macaron_fraction_l2845_284577

theorem macaron_fraction (mitch joshua miles renz : ℕ) (total_kids : ℕ) :
  mitch = 20 →
  joshua = mitch + 6 →
  2 * joshua = miles →
  total_kids = 68 →
  2 * total_kids = mitch + joshua + miles + renz →
  renz + 1 = miles * 19 / 26 :=
by sorry

end NUMINAMATH_CALUDE_macaron_fraction_l2845_284577


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2845_284582

theorem complex_equation_solution (z : ℂ) : 
  z * (1 - 2 * Complex.I) = 2 + 4 * Complex.I → 
  z = -2/5 + 8/5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2845_284582


namespace NUMINAMATH_CALUDE_carrot_picking_l2845_284562

theorem carrot_picking (carol_carrots : ℕ) (good_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : good_carrots = 38)
  (h3 : bad_carrots = 7) :
  good_carrots + bad_carrots - carol_carrots = 16 := by
  sorry

end NUMINAMATH_CALUDE_carrot_picking_l2845_284562


namespace NUMINAMATH_CALUDE_star_difference_sum_l2845_284558

/-- The ⋆ operation for real numbers -/
def star (a b : ℝ) : ℝ := a^2 - b

/-- Theorem stating the result of (x - y) ⋆ (x + y) -/
theorem star_difference_sum (x y : ℝ) : 
  star (x - y) (x + y) = x^2 - x - 2*x*y + y^2 - y := by
  sorry

end NUMINAMATH_CALUDE_star_difference_sum_l2845_284558


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seats_l2845_284548

/-- The number of seats on a Ferris wheel -/
def ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) : ℕ :=
  total_people / people_per_seat

/-- Theorem: The Ferris wheel in paradise park has 4 seats -/
theorem paradise_park_ferris_wheel_seats :
  ferris_wheel_seats 16 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seats_l2845_284548


namespace NUMINAMATH_CALUDE_union_of_sets_l2845_284515

theorem union_of_sets : 
  let A : Set ℤ := {1, 3, 5, 6}
  let B : Set ℤ := {-1, 5, 7}
  A ∪ B = {-1, 1, 3, 5, 6, 7} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2845_284515


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2845_284536

/-- Given two 2D vectors a and b, where a = (2,1), a + b = (1,k), and a ⟂ b, prove that k = 3 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  a + b = (1, k) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2845_284536


namespace NUMINAMATH_CALUDE_arrangement_count_is_432_l2845_284563

/-- The number of ways to arrange players from four teams in a row. -/
def arrangement_count : ℕ :=
  let celtics := 3  -- Number of Celtics players
  let lakers := 3   -- Number of Lakers players
  let warriors := 2 -- Number of Warriors players
  let nuggets := 2  -- Number of Nuggets players
  let team_count := 4 -- Number of teams
  let specific_warrior := 1 -- One specific Warrior must sit at the left end
  
  -- Arrangements of teams (excluding Warriors who are fixed at the left)
  (team_count - 1).factorial *
  -- Arrangement of the non-specific Warrior
  (warriors - specific_warrior).factorial *
  -- Arrangements within each team
  celtics.factorial * lakers.factorial * nuggets.factorial

/-- Theorem stating that the number of arrangements is 432. -/
theorem arrangement_count_is_432 : arrangement_count = 432 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_432_l2845_284563


namespace NUMINAMATH_CALUDE_t_formula_correct_t_2022_last_digit_l2845_284594

/-- The number of unordered triples of non-empty and pairwise disjoint subsets of a set with n elements -/
def t (n : ℕ+) : ℚ :=
  (4^n.val - 3 * 3^n.val + 3 * 2^n.val - 1) / 6

/-- The closed form formula for t_n is correct -/
theorem t_formula_correct (n : ℕ+) :
  t n = (4^n.val - 3 * 3^n.val + 3 * 2^n.val - 1) / 6 := by sorry

/-- The last digit of t_2022 is 1 -/
theorem t_2022_last_digit :
  t 2022 % 1 = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_t_formula_correct_t_2022_last_digit_l2845_284594


namespace NUMINAMATH_CALUDE_chris_bowling_score_l2845_284543

/-- Proves Chris's bowling score given Sarah and Greg's score conditions -/
theorem chris_bowling_score (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 60 →
  (sarah_score + greg_score) / 2 = 110 →
  let avg := (sarah_score + greg_score) / 2
  let chris_score := (avg * 120) / 100
  chris_score = 132 := by
sorry

end NUMINAMATH_CALUDE_chris_bowling_score_l2845_284543


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l2845_284535

theorem greatest_integer_less_than_negative_fraction :
  ⌊-21/5⌋ = -5 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l2845_284535


namespace NUMINAMATH_CALUDE_inequality_solution_l2845_284585

theorem inequality_solution (x : ℝ) : 
  x ≥ 0 → (2021 * (x^2020)^(1/202) - 1 ≥ 2020*x ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2845_284585


namespace NUMINAMATH_CALUDE_family_composition_l2845_284570

/-- A family where one member has an equal number of brothers and sisters,
    and another member has twice as many brothers as sisters. -/
structure Family where
  boys : ℕ
  girls : ℕ
  tony_equal_siblings : boys - 1 = girls
  alice_double_brothers : boys = 2 * (girls - 1)

/-- The family has 4 boys and 3 girls. -/
theorem family_composition (f : Family) : f.boys = 4 ∧ f.girls = 3 := by
  sorry

end NUMINAMATH_CALUDE_family_composition_l2845_284570


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l2845_284560

theorem inscribed_circle_square_area (s : ℝ) (r : ℝ) : 
  r > 0 → s = 2 * r → r^2 * Real.pi = 9 * Real.pi → s^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l2845_284560


namespace NUMINAMATH_CALUDE_basketball_team_subjects_l2845_284593

theorem basketball_team_subjects (P C B : Finset Nat) : 
  (P ∪ C ∪ B).card = 18 →
  P.card = 10 →
  B.card = 7 →
  C.card = 5 →
  (P ∩ B).card = 3 →
  (B ∩ C).card = 2 →
  (P ∩ C).card = 1 →
  (P ∩ C ∩ B).card = 2 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_subjects_l2845_284593


namespace NUMINAMATH_CALUDE_parabola_reflects_to_parallel_l2845_284538

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a curve in 2D space -/
def CurveEquation : Type := Point → Prop

/-- The equation of a parabola y^2 = 2Cx + C^2 -/
def ParabolaEquation (C : ℝ) : CurveEquation :=
  fun p => p.y^2 = 2*C*p.x + C^2

/-- A ray of light -/
structure Ray where
  origin : Point
  direction : Point

/-- The reflection of a ray off a curve at a point -/
def ReflectedRay (curve : CurveEquation) (incidentRay : Ray) (reflectionPoint : Point) : Ray :=
  sorry

/-- The theorem stating that a parabola reflects rays from the origin into parallel rays -/
theorem parabola_reflects_to_parallel (C : ℝ) :
  ∀ (p : Point), ParabolaEquation C p →
  ∀ (incidentRay : Ray),
    incidentRay.origin = ⟨0, 0⟩ →
    (ReflectedRay (ParabolaEquation C) incidentRay p).direction.y = 0 :=
  sorry

end NUMINAMATH_CALUDE_parabola_reflects_to_parallel_l2845_284538


namespace NUMINAMATH_CALUDE_negation_equivalence_l2845_284521

theorem negation_equivalence (m : ℝ) : 
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2845_284521


namespace NUMINAMATH_CALUDE_function_inequality_l2845_284555

theorem function_inequality (m n : ℝ) (hm : m < 0) :
  (∃ x : ℝ, x > 0 ∧ Real.log x + m * x + n ≥ 0) →
  n - 1 ≥ Real.log (-m) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2845_284555


namespace NUMINAMATH_CALUDE_book_words_per_page_l2845_284599

theorem book_words_per_page :
  ∀ (words_per_page : ℕ),
    words_per_page ≤ 120 →
    (150 * words_per_page) % 221 = 210 →
    words_per_page = 98 := by
  sorry

end NUMINAMATH_CALUDE_book_words_per_page_l2845_284599


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2845_284580

/-- The longest segment in a cylinder with radius 5 cm and height 6 cm is √136 cm. -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 6) :
  Real.sqrt ((2 * r)^2 + h^2) = Real.sqrt 136 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2845_284580


namespace NUMINAMATH_CALUDE_initial_work_plan_l2845_284575

/-- Proves that the initial plan was to complete the work in 28 days given the conditions of the problem. -/
theorem initial_work_plan (total_men : Nat) (absent_men : Nat) (days_with_reduced_men : Nat) 
  (h1 : total_men = 42)
  (h2 : absent_men = 6)
  (h3 : days_with_reduced_men = 14) : 
  (total_men * ((total_men - absent_men) * days_with_reduced_men)) / (total_men - absent_men) = 28 := by
  sorry

#eval (42 * ((42 - 6) * 14)) / (42 - 6)

end NUMINAMATH_CALUDE_initial_work_plan_l2845_284575


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2845_284549

theorem weight_of_new_person (initial_weight : ℝ) (weight_increase : ℝ) :
  initial_weight = 65 →
  weight_increase = 4.5 →
  ∃ (new_weight : ℝ), new_weight = initial_weight + 2 * weight_increase :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2845_284549


namespace NUMINAMATH_CALUDE_unknown_number_value_l2845_284579

theorem unknown_number_value (x n : ℤ) : 
  x = 88320 →
  x + n + 9211 - 1569 = 11901 →
  n = -84061 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_value_l2845_284579


namespace NUMINAMATH_CALUDE_inscribed_circle_inequality_l2845_284545

variable (a b c u v w : ℝ)

-- a, b, c are positive real numbers representing side lengths of a triangle
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- u, v, w are positive real numbers representing distances from incenter to opposite vertices
variable (hu : u > 0) (hv : v > 0) (hw : w > 0)

-- Triangle inequality
variable (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b)

theorem inscribed_circle_inequality :
  (a + b + c) * (1/u + 1/v + 1/w) ≤ 3 * (a/u + b/v + c/w) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_inequality_l2845_284545
