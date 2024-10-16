import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2019_l3340_334000

theorem smallest_n_divisible_by_2019 : ∃ (n : ℕ), n = 2000 ∧ 
  (∀ (m : ℕ), m < n → ¬(2019 ∣ (m^2 + 20*m + 19))) ∧ 
  (2019 ∣ (n^2 + 20*n + 19)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2019_l3340_334000


namespace NUMINAMATH_CALUDE_sqrt_range_l3340_334047

theorem sqrt_range (x : ℝ) : x ∈ {y : ℝ | ∃ (z : ℝ), z^2 = y - 7} ↔ x ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_range_l3340_334047


namespace NUMINAMATH_CALUDE_problem_statement_l3340_334065

theorem problem_statement (a b : ℝ) : 
  |a + 2| + (b - 1)^2 = 0 → (a + b)^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3340_334065


namespace NUMINAMATH_CALUDE_segment_AE_length_l3340_334030

-- Define the quadrilateral ABCD and point E
structure Quadrilateral :=
  (A B C D E : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let d_AB := Real.sqrt ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2)
  let d_CD := Real.sqrt ((q.D.1 - q.C.1)^2 + (q.D.2 - q.C.2)^2)
  let d_AC := Real.sqrt ((q.C.1 - q.A.1)^2 + (q.C.2 - q.A.2)^2)
  let d_AE := Real.sqrt ((q.E.1 - q.A.1)^2 + (q.E.2 - q.A.2)^2)
  let d_EC := Real.sqrt ((q.C.1 - q.E.1)^2 + (q.C.2 - q.E.2)^2)
  d_AB = 10 ∧ d_CD = 15 ∧ d_AC = 18 ∧
  (q.E.1 - q.A.1) * (q.C.1 - q.A.1) + (q.E.2 - q.A.2) * (q.C.2 - q.A.2) = d_AE * d_AC ∧
  (q.E.1 - q.B.1) * (q.D.1 - q.B.1) + (q.E.2 - q.B.2) * (q.D.2 - q.B.2) = 
    Real.sqrt ((q.E.1 - q.B.1)^2 + (q.E.2 - q.B.2)^2) * Real.sqrt ((q.D.1 - q.B.1)^2 + (q.D.2 - q.B.2)^2) ∧
  d_AE / d_EC = 10 / 15

theorem segment_AE_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  Real.sqrt ((q.E.1 - q.A.1)^2 + (q.E.2 - q.A.2)^2) = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_segment_AE_length_l3340_334030


namespace NUMINAMATH_CALUDE_seniors_in_three_sports_l3340_334041

theorem seniors_in_three_sports 
  (total_seniors : ℕ) 
  (football : ℕ) 
  (baseball : ℕ) 
  (football_lacrosse : ℕ) 
  (baseball_football : ℕ) 
  (baseball_lacrosse : ℕ) 
  (h1 : total_seniors = 85)
  (h2 : football = 74)
  (h3 : baseball = 26)
  (h4 : football_lacrosse = 17)
  (h5 : baseball_football = 18)
  (h6 : baseball_lacrosse = 13)
  : ∃ (n : ℕ), n = 11 ∧ 
    total_seniors = football + baseball + 2*n - baseball_football - football_lacrosse - baseball_lacrosse + n :=
by sorry

end NUMINAMATH_CALUDE_seniors_in_three_sports_l3340_334041


namespace NUMINAMATH_CALUDE_smallest_cube_divisor_l3340_334057

theorem smallest_cube_divisor (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let m := a^3 * b^5 * c^7
  ∀ k : ℕ, k^3 ∣ m → (a * b * c^3)^3 ≤ k^3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_divisor_l3340_334057


namespace NUMINAMATH_CALUDE_function_value_determines_a_l3340_334080

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

-- State the theorem
theorem function_value_determines_a (a : ℝ) : f a (f a 0) = 3*a → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_value_determines_a_l3340_334080


namespace NUMINAMATH_CALUDE_complex_distance_sum_l3340_334091

theorem complex_distance_sum (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 4) :
  Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 3*I))^2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_sum_l3340_334091


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3340_334098

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem problem_solution :
  ∃ (k : ℕ), k < 17 ∧ (9857621 - k) % 17 = 0 ∧ ∀ (m : ℕ), m < k → (9857621 - m) % 17 ≠ 0 ∧ k = 8 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3340_334098


namespace NUMINAMATH_CALUDE_car_speed_proof_l3340_334093

theorem car_speed_proof (reduced_speed : ℝ) (distance : ℝ) (time : ℝ) (actual_speed : ℝ) : 
  reduced_speed = 5 / 7 * actual_speed →
  distance = 42 →
  time = 42 / 25 →
  reduced_speed = distance / time →
  actual_speed = 35 := by
sorry


end NUMINAMATH_CALUDE_car_speed_proof_l3340_334093


namespace NUMINAMATH_CALUDE_min_cups_to_fill_container_l3340_334019

def container_capacity : ℝ := 640
def cup_capacity : ℝ := 120

theorem min_cups_to_fill_container : 
  ∃ n : ℕ, (n : ℝ) * cup_capacity ≥ container_capacity ∧ 
  ∀ m : ℕ, (m : ℝ) * cup_capacity ≥ container_capacity → n ≤ m ∧ 
  n = 6 :=
sorry

end NUMINAMATH_CALUDE_min_cups_to_fill_container_l3340_334019


namespace NUMINAMATH_CALUDE_pythagorean_triple_check_l3340_334003

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_check :
  ¬ is_pythagorean_triple 1 1 2 ∧
  ¬ is_pythagorean_triple 3 4 5 ∧
  is_pythagorean_triple 6 8 10 ∧
  ¬ is_pythagorean_triple 1 1 1 :=
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_check_l3340_334003


namespace NUMINAMATH_CALUDE_number_with_specific_remainder_l3340_334048

theorem number_with_specific_remainder : ∃ x : ℕ, ∃ k : ℕ, 
  x = 29 * k + 8 ∧ 
  1490 % 29 = 11 ∧ 
  (∀ m : ℕ, m > 29 → (x % m ≠ 8 ∨ 1490 % m ≠ 11)) :=
by sorry

end NUMINAMATH_CALUDE_number_with_specific_remainder_l3340_334048


namespace NUMINAMATH_CALUDE_max_consecutive_sum_l3340_334099

/-- The sum of n consecutive integers starting from k -/
def consecutiveSum (n k : ℕ) : ℕ := n * (2 * k + (n - 1)) / 2

/-- The maximum number of consecutive positive integers starting from 3 
    that can be added together before the sum exceeds 500 -/
theorem max_consecutive_sum : 
  (∀ m : ℕ, m ≤ 29 → consecutiveSum m 3 ≤ 500) ∧ 
  consecutiveSum 30 3 > 500 := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_l3340_334099


namespace NUMINAMATH_CALUDE_solve_system_l3340_334087

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := 2 * a + b

-- Theorem statement
theorem solve_system (x y : ℝ) 
  (h1 : otimes x (-y) = 2) 
  (h2 : otimes (2 * y) x = 1) : 
  x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3340_334087


namespace NUMINAMATH_CALUDE_range_of_a_l3340_334007

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3340_334007


namespace NUMINAMATH_CALUDE_diamond_example_l3340_334016

/-- The diamond operation -/
def diamond (a b : ℤ) : ℤ := a * b^2 - b + 1

/-- Theorem stating the result of (3 ◇ 4) ◇ 2 -/
theorem diamond_example : diamond (diamond 3 4) 2 = 179 := by
  sorry

end NUMINAMATH_CALUDE_diamond_example_l3340_334016


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3340_334082

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (l : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : parallel l β) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3340_334082


namespace NUMINAMATH_CALUDE_fraction_equality_l3340_334070

theorem fraction_equality (x y p q : ℚ) : 
  (7 * x + 6 * y) / (x - 2 * y) = 27 → x / (2 * y) = p / q → p / q = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3340_334070


namespace NUMINAMATH_CALUDE_quadratic_equation_has_solution_l3340_334052

theorem quadratic_equation_has_solution (a b : ℝ) :
  ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_solution_l3340_334052


namespace NUMINAMATH_CALUDE_annes_speed_ratio_l3340_334061

/-- Proves that the ratio of Anne's new cleaning rate to her original rate is 2:1 --/
theorem annes_speed_ratio :
  -- Bruce and Anne's original combined rate
  ∀ (B A : ℚ), B + A = 1/4 →
  -- Anne's original rate
  A = 1/12 →
  -- Bruce and Anne's new combined rate (with Anne's changed speed)
  ∀ (A' : ℚ), B + A' = 1/3 →
  -- The ratio of Anne's new rate to her original rate
  A' / A = 2 := by
sorry

end NUMINAMATH_CALUDE_annes_speed_ratio_l3340_334061


namespace NUMINAMATH_CALUDE_mary_earnings_l3340_334086

/-- Mary's earnings from cleaning homes -/
theorem mary_earnings (total_earnings : ℕ) (homes_cleaned : ℕ) 
  (h1 : total_earnings = 276)
  (h2 : homes_cleaned = 6) :
  total_earnings / homes_cleaned = 46 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_l3340_334086


namespace NUMINAMATH_CALUDE_two_number_difference_l3340_334095

theorem two_number_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 3 * x = 27) : |y - x| = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l3340_334095


namespace NUMINAMATH_CALUDE_set_equality_gt_one_set_equality_odd_integers_l3340_334054

-- Statement 1
theorem set_equality_gt_one : {x : ℝ | x > 1} = {y : ℝ | y > 1} := by sorry

-- Statement 2
theorem set_equality_odd_integers : {x : ℤ | ∃ k : ℤ, x = 2*k + 1} = {x : ℤ | ∃ k : ℤ, x = 2*k - 1} := by sorry

end NUMINAMATH_CALUDE_set_equality_gt_one_set_equality_odd_integers_l3340_334054


namespace NUMINAMATH_CALUDE_caterpillar_final_position_l3340_334078

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction as a unit vector -/
inductive Direction
  | West
  | North
  | East
  | South

/-- Represents the state of the caterpillar -/
structure CaterpillarState where
  position : Point
  direction : Direction
  moveDistance : Nat

/-- Performs a single move and turn -/
def move (state : CaterpillarState) : CaterpillarState :=
  sorry

/-- Performs n moves and turns -/
def moveNTimes (initialState : CaterpillarState) (n : Nat) : CaterpillarState :=
  sorry

/-- The main theorem to prove -/
theorem caterpillar_final_position :
  let initialState : CaterpillarState := {
    position := { x := 15, y := -15 },
    direction := Direction.West,
    moveDistance := 1
  }
  let finalState := moveNTimes initialState 1010
  finalState.position = { x := -491, y := 489 } :=
sorry

end NUMINAMATH_CALUDE_caterpillar_final_position_l3340_334078


namespace NUMINAMATH_CALUDE_problem_solution_l3340_334026

theorem problem_solution : 3 * 3^4 + 9^30 / 9^28 = 324 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3340_334026


namespace NUMINAMATH_CALUDE_surface_area_ratio_l3340_334049

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  /-- The edge length of the tetrahedron -/
  edge_length : ℝ
  /-- Assumption that the edge length is positive -/
  edge_positive : edge_length > 0

/-- The surface area of a regular tetrahedron -/
def surface_area_tetrahedron (t : RegularTetrahedron) : ℝ := sorry

/-- The surface area of the inscribed sphere of a regular tetrahedron -/
def surface_area_inscribed_sphere (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the ratio of the surface areas -/
theorem surface_area_ratio (t : RegularTetrahedron) :
  surface_area_tetrahedron t / surface_area_inscribed_sphere t = 6 * Real.sqrt 3 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_surface_area_ratio_l3340_334049


namespace NUMINAMATH_CALUDE_calculation_result_l3340_334071

theorem calculation_result : 50 + 50 / 50 + 50 = 101 := by sorry

end NUMINAMATH_CALUDE_calculation_result_l3340_334071


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l3340_334060

theorem complex_magnitude_squared (a b : ℝ) (z : ℂ) : 
  z = Complex.mk a b → z + Complex.abs z = 3 + 7*Complex.I → Complex.abs z^2 = 841/9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l3340_334060


namespace NUMINAMATH_CALUDE_problem_solution_l3340_334002

theorem problem_solution (x : ℝ) : 
  3 - (1/4)*2 - (1/3)*3 - (1/7)*x = 27 → (10/100) * x = 17.85 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3340_334002


namespace NUMINAMATH_CALUDE_circle_radius_l3340_334018

theorem circle_radius (x y : ℝ) (h : x + y = 150 * Real.pi) : 
  ∃ (r : ℝ), r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = Real.sqrt 151 - 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3340_334018


namespace NUMINAMATH_CALUDE_min_tangent_length_l3340_334027

/-- The minimum length of a tangent drawn from a point on the line y = x - 1 
    to the circle x^2 + y^2 - 6x + 8 = 0 is equal to 1. -/
theorem min_tangent_length : 
  let line : Set (ℝ × ℝ) := {p | p.2 = p.1 - 1}
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 + 8 = 0}
  ∃ (min_length : ℝ), 
    (∀ (p : ℝ × ℝ) (t : ℝ × ℝ), 
      p ∈ line → t ∈ circle → 
      dist p t ≥ min_length) ∧
    (∃ (p : ℝ × ℝ) (t : ℝ × ℝ), 
      p ∈ line ∧ t ∈ circle ∧ 
      dist p t = min_length) ∧
    min_length = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_tangent_length_l3340_334027


namespace NUMINAMATH_CALUDE_smallest_number_l3340_334088

theorem smallest_number (a b c d : ℚ) (ha : a = 0) (hb : b = -2/3) (hc : c = 1) (hd : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3340_334088


namespace NUMINAMATH_CALUDE_profit_sharing_multiple_l3340_334012

/-- Given the conditions of a profit-sharing scenario, prove that the multiple of R's capital is 10. -/
theorem profit_sharing_multiple (P Q R k : ℚ) (total_profit : ℚ) : 
  4 * P = 6 * Q ∧ 
  4 * P = k * R ∧ 
  total_profit = 4340 ∧ 
  R * (total_profit / (P + Q + R)) = 840 →
  k = 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_multiple_l3340_334012


namespace NUMINAMATH_CALUDE_group_interval_calculation_l3340_334042

/-- Given a group [a,b) in a frequency distribution histogram with frequency 0.3 and height 0.06, |a-b| = 5 -/
theorem group_interval_calculation (a b : ℝ) 
  (frequency : ℝ) (height : ℝ) 
  (h1 : frequency = 0.3) 
  (h2 : height = 0.06) : 
  |a - b| = 5 := by sorry

end NUMINAMATH_CALUDE_group_interval_calculation_l3340_334042


namespace NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l3340_334017

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

-- Theorem for intervals of monotonicity and minimum value
theorem f_monotonicity_and_minimum :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f x < f y) ∧
  (∀ x y, x < y ∧ x > 3 ∧ y > 3 → f x < f y) ∧
  (∀ x y, x < y ∧ x > -1 ∧ y < 3 → f x > f y) ∧
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≥ -20) ∧
  (f 2 = -20) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l3340_334017


namespace NUMINAMATH_CALUDE_milk_cartons_consumption_l3340_334092

theorem milk_cartons_consumption (total_cartons : ℕ) 
  (younger_sister_fraction : ℚ) (older_sister_fraction : ℚ) :
  total_cartons = 24 →
  younger_sister_fraction = 1 / 8 →
  older_sister_fraction = 3 / 8 →
  (younger_sister_fraction * total_cartons : ℚ) = 3 ∧
  (older_sister_fraction * total_cartons : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_milk_cartons_consumption_l3340_334092


namespace NUMINAMATH_CALUDE_carwash_problem_l3340_334097

/-- Represents the carwash problem with modified constraints to ensure consistency --/
theorem carwash_problem 
  (car_price SUV_price truck_price motorcycle_price bus_price : ℕ)
  (total_raised : ℕ)
  (num_SUVs num_trucks num_motorcycles : ℕ)
  (max_vehicles : ℕ)
  (h1 : car_price = 7)
  (h2 : SUV_price = 12)
  (h3 : truck_price = 10)
  (h4 : motorcycle_price = 15)
  (h5 : bus_price = 18)
  (h6 : total_raised = 500)
  (h7 : num_SUVs = 3)
  (h8 : num_trucks = 8)
  (h9 : num_motorcycles = 5)
  (h10 : max_vehicles = 20)  -- Modified to make the problem consistent
  : ∃ (num_cars num_buses : ℕ), 
    (num_cars + num_buses + num_SUVs + num_trucks + num_motorcycles ≤ max_vehicles) ∧ 
    (num_cars % 2 = 0) ∧ 
    (num_buses % 2 = 1) ∧
    (car_price * num_cars + bus_price * num_buses + 
     SUV_price * num_SUVs + truck_price * num_trucks + 
     motorcycle_price * num_motorcycles = total_raised) := by
  sorry


end NUMINAMATH_CALUDE_carwash_problem_l3340_334097


namespace NUMINAMATH_CALUDE_power_of_sixteen_five_fourths_l3340_334036

theorem power_of_sixteen_five_fourths : (16 : ℝ) ^ (5/4 : ℝ) = 32 := by sorry

end NUMINAMATH_CALUDE_power_of_sixteen_five_fourths_l3340_334036


namespace NUMINAMATH_CALUDE_xy_value_l3340_334044

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x + y) = 32)
  (h2 : (16 : ℝ)^(x + y) / (4 : ℝ)^(3 * y) = 256) : 
  x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3340_334044


namespace NUMINAMATH_CALUDE_complex_inequality_l3340_334069

theorem complex_inequality (x y a b : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l3340_334069


namespace NUMINAMATH_CALUDE_water_ratio_is_half_l3340_334066

/-- Represents the water flow problem --/
structure WaterFlow where
  flow_rate_1 : ℚ  -- Flow rate for the first hour (cups per 10 minutes)
  flow_rate_2 : ℚ  -- Flow rate for the second hour (cups per 10 minutes)
  duration_1 : ℚ   -- Duration of first flow rate (hours)
  duration_2 : ℚ   -- Duration of second flow rate (hours)
  water_left : ℚ   -- Amount of water left after dumping (cups)

/-- Calculates the total water collected before dumping --/
def total_water (wf : WaterFlow) : ℚ :=
  wf.flow_rate_1 * 6 * wf.duration_1 + wf.flow_rate_2 * 6 * wf.duration_2

/-- Theorem stating the ratio of water left to total water collected is 1/2 --/
theorem water_ratio_is_half (wf : WaterFlow) 
  (h1 : wf.flow_rate_1 = 2)
  (h2 : wf.flow_rate_2 = 4)
  (h3 : wf.duration_1 = 1)
  (h4 : wf.duration_2 = 1)
  (h5 : wf.water_left = 18) :
  wf.water_left / total_water wf = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_water_ratio_is_half_l3340_334066


namespace NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l3340_334067

-- Define a real-valued function on the real line
variable (f : ℝ → ℝ)

-- Assume f is differentiable
variable (hf : Differentiable ℝ f)

-- Define what it means for f to have an extremum at a point
def has_extremum_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem derivative_zero_necessary_not_sufficient :
  (∀ x : ℝ, has_extremum_at f x → deriv f x = 0) ∧
  ¬(∀ x : ℝ, deriv f x = 0 → has_extremum_at f x) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l3340_334067


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3340_334020

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x + y = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3340_334020


namespace NUMINAMATH_CALUDE_lauren_revenue_l3340_334014

def commercial_revenue (per_commercial : ℚ) (num_commercials : ℕ) : ℚ :=
  per_commercial * num_commercials

def subscription_revenue (per_subscription : ℚ) (num_subscriptions : ℕ) : ℚ :=
  per_subscription * num_subscriptions

theorem lauren_revenue 
  (per_commercial : ℚ) 
  (per_subscription : ℚ) 
  (num_commercials : ℕ) 
  (num_subscriptions : ℕ) 
  (total_revenue : ℚ) :
  per_subscription = 1 →
  num_commercials = 100 →
  num_subscriptions = 27 →
  total_revenue = 77 →
  commercial_revenue per_commercial num_commercials + 
    subscription_revenue per_subscription num_subscriptions = total_revenue →
  per_commercial = 1/2 := by
sorry

end NUMINAMATH_CALUDE_lauren_revenue_l3340_334014


namespace NUMINAMATH_CALUDE_lindas_outfits_l3340_334056

/-- The number of different outfits that can be created from a given number of skirts, blouses, and shoes. -/
def number_of_outfits (skirts blouses shoes : ℕ) : ℕ :=
  skirts * blouses * shoes

/-- Theorem stating that with 5 skirts, 8 blouses, and 2 pairs of shoes, 80 different outfits can be created. -/
theorem lindas_outfits :
  number_of_outfits 5 8 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lindas_outfits_l3340_334056


namespace NUMINAMATH_CALUDE_sin_ratio_comparison_l3340_334028

open Real

theorem sin_ratio_comparison : 
  (sin (2016 * π / 180)) / (sin (2017 * π / 180)) < 
  (sin (2018 * π / 180)) / (sin (2019 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_sin_ratio_comparison_l3340_334028


namespace NUMINAMATH_CALUDE_min_value_expression_l3340_334006

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1 / y^2) * (x + 1 / y^2 - 500) + (y + 1 / x^2) * (y + 1 / x^2 - 500) ≥ -125000 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3340_334006


namespace NUMINAMATH_CALUDE_treehouse_rope_length_l3340_334023

theorem treehouse_rope_length : 
  let rope_lengths : List Nat := [24, 20, 14, 12, 18, 22]
  List.sum rope_lengths = 110 := by
  sorry

end NUMINAMATH_CALUDE_treehouse_rope_length_l3340_334023


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3340_334034

-- Define the volume of the cube
def cube_volume : ℝ := 125

-- Theorem stating the relationship between volume and surface area of one side
theorem cube_surface_area_from_volume :
  ∃ (side_length : ℝ), 
    side_length^3 = cube_volume ∧ 
    side_length^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3340_334034


namespace NUMINAMATH_CALUDE_coefficient_x4y2_in_expansion_coefficient_equals_60_l3340_334081

/-- The coefficient of x^4y^2 in the expansion of (x-2y)^6 is 60 -/
theorem coefficient_x4y2_in_expansion : ℕ :=
  60

/-- The binomial coefficient "6 choose 2" -/
def binomial_6_2 : ℕ := 15

/-- The expansion of (x-2y)^6 -/
def expansion (x y : ℝ) : ℝ := (x - 2*y)^6

/-- The coefficient of x^4y^2 in the expansion -/
def coefficient (x y : ℝ) : ℝ := binomial_6_2 * (-2)^2

theorem coefficient_equals_60 :
  coefficient = λ _ _ ↦ 60 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x4y2_in_expansion_coefficient_equals_60_l3340_334081


namespace NUMINAMATH_CALUDE_scientist_news_sharing_l3340_334068

/-- Represents the state of scientists' knowledge before and after pairing -/
structure ScientistState where
  total : Nat
  initial_knowledgeable : Nat
  final_knowledgeable : Nat

/-- Probability of a specific final state given initial conditions -/
def probability (s : ScientistState) : Rat :=
  sorry

/-- Expected number of scientists knowing the news after pairing -/
def expected_final_knowledgeable (total : Nat) (initial_knowledgeable : Nat) : Rat :=
  sorry

/-- Main theorem about scientists and news sharing -/
theorem scientist_news_sharing :
  let s₁ : ScientistState := ⟨18, 10, 13⟩
  let s₂ : ScientistState := ⟨18, 10, 14⟩
  probability s₁ = 0 ∧
  probability s₂ = 1120 / 2431 ∧
  expected_final_knowledgeable 18 10 = 14^12 / 17 :=
by sorry

end NUMINAMATH_CALUDE_scientist_news_sharing_l3340_334068


namespace NUMINAMATH_CALUDE_smallest_integer_below_sqrt5_plus_sqrt3_to_6th_l3340_334013

theorem smallest_integer_below_sqrt5_plus_sqrt3_to_6th :
  ∃ n : ℤ, n = 3322 ∧ n < (Real.sqrt 5 + Real.sqrt 3)^6 ∧ ∀ m : ℤ, m < (Real.sqrt 5 + Real.sqrt 3)^6 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_below_sqrt5_plus_sqrt3_to_6th_l3340_334013


namespace NUMINAMATH_CALUDE_find_other_number_l3340_334022

theorem find_other_number (a b : ℕ+) (hcf lcm : ℕ+) : 
  Nat.gcd a.val b.val = hcf.val →
  Nat.lcm a.val b.val = lcm.val →
  hcf * lcm = a * b →
  a = 154 →
  hcf = 14 →
  lcm = 396 →
  b = 36 := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l3340_334022


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3340_334024

theorem sufficient_not_necessary (a : ℝ) (h : a > 0) :
  (∀ a, a > 2 → a^a > a^2) ∧
  (∃ a, 0 < a ∧ a < 2 ∧ a^a > a^2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3340_334024


namespace NUMINAMATH_CALUDE_optimal_invitation_strategy_l3340_334096

/-- Represents an invitation strategy for a social gathering. -/
structure InvitationStrategy where
  total_acquaintances : Nat
  ladies : Nat
  gentlemen : Nat
  ladies_per_invite : Nat
  gentlemen_per_invite : Nat
  invitations : Nat

/-- Checks if the invitation strategy is valid and optimal. -/
def is_valid_and_optimal (strategy : InvitationStrategy) : Prop :=
  strategy.total_acquaintances = strategy.ladies + strategy.gentlemen
  ∧ strategy.ladies_per_invite + strategy.gentlemen_per_invite < strategy.total_acquaintances
  ∧ strategy.invitations * strategy.ladies_per_invite ≥ strategy.ladies * (strategy.total_acquaintances - 1)
  ∧ strategy.invitations * strategy.gentlemen_per_invite ≥ strategy.gentlemen * (strategy.total_acquaintances - 1)
  ∧ ∀ n : Nat, n < strategy.invitations →
    n * strategy.ladies_per_invite < strategy.ladies * (strategy.total_acquaintances - 1)
    ∨ n * strategy.gentlemen_per_invite < strategy.gentlemen * (strategy.total_acquaintances - 1)

theorem optimal_invitation_strategy :
  ∃ (strategy : InvitationStrategy),
    strategy.total_acquaintances = 20
    ∧ strategy.ladies = 9
    ∧ strategy.gentlemen = 11
    ∧ strategy.ladies_per_invite = 3
    ∧ strategy.gentlemen_per_invite = 2
    ∧ strategy.invitations = 11
    ∧ is_valid_and_optimal strategy
    ∧ (strategy.invitations * strategy.ladies_per_invite) / strategy.ladies = 7
    ∧ (strategy.invitations * strategy.gentlemen_per_invite) / strategy.gentlemen = 2 :=
  sorry

end NUMINAMATH_CALUDE_optimal_invitation_strategy_l3340_334096


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3340_334043

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Given a_13 = S_13 = 13, prove a_1 = -11 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 13 = 13) (h2 : seq.S 13 = 13) : seq.a 1 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3340_334043


namespace NUMINAMATH_CALUDE_log_two_x_equals_neg_two_l3340_334032

theorem log_two_x_equals_neg_two (x : ℝ) : 
  x = (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4) → Real.log x / Real.log 2 = -2 :=
by sorry

end NUMINAMATH_CALUDE_log_two_x_equals_neg_two_l3340_334032


namespace NUMINAMATH_CALUDE_girls_left_auditorium_l3340_334025

theorem girls_left_auditorium (initial_boys : ℕ) (initial_girls : ℕ) (remaining_students : ℕ) : 
  initial_boys = 24 →
  initial_girls = 14 →
  remaining_students = 30 →
  ∃ (left_girls : ℕ), left_girls = 4 ∧ 
    ∃ (left_boys : ℕ), left_boys = left_girls ∧
    initial_boys + initial_girls - (left_boys + left_girls) = remaining_students :=
by sorry

end NUMINAMATH_CALUDE_girls_left_auditorium_l3340_334025


namespace NUMINAMATH_CALUDE_unique_intersection_point_l3340_334009

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies all four linear equations -/
def satisfiesAllEquations (p : Point2D) : Prop :=
  3 * p.x - 2 * p.y = 12 ∧
  2 * p.x + 5 * p.y = -1 ∧
  p.x + 4 * p.y = 8 ∧
  5 * p.x - 3 * p.y = 15

/-- Theorem stating that there exists exactly one point satisfying all equations -/
theorem unique_intersection_point :
  ∃! p : Point2D, satisfiesAllEquations p :=
sorry


end NUMINAMATH_CALUDE_unique_intersection_point_l3340_334009


namespace NUMINAMATH_CALUDE_inequality_and_optimization_l3340_334075

theorem inequality_and_optimization (m : ℝ) :
  (∀ x : ℝ, |x + 3| + |x + m| ≥ 2*m) →
  m ≤ 1 ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    a^2 + 2*b^2 + 3*c^2 ≥ 6/11 ∧
    (a^2 + 2*b^2 + 3*c^2 = 6/11 ↔ a = 6/11 ∧ b = 3/11 ∧ c = 2/11)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_optimization_l3340_334075


namespace NUMINAMATH_CALUDE_circle_properties_l3340_334010

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 0)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3340_334010


namespace NUMINAMATH_CALUDE_square_equation_solution_l3340_334094

theorem square_equation_solution : 
  ∃ x : ℝ, (2010 + x)^2 = 2*x^2 ∧ (x = 4850 ∨ x = -830) :=
by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3340_334094


namespace NUMINAMATH_CALUDE_yogurt_combinations_l3340_334004

/-- The number of yogurt flavors -/
def num_flavors : ℕ := 5

/-- The number of toppings -/
def num_toppings : ℕ := 7

/-- The number of toppings to choose -/
def toppings_to_choose : ℕ := 2

/-- The number of doubling options (double first, double second, or no doubling) -/
def doubling_options : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem yogurt_combinations :
  num_flavors * choose num_toppings toppings_to_choose * doubling_options = 315 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l3340_334004


namespace NUMINAMATH_CALUDE_triangle_segment_equality_l3340_334085

theorem triangle_segment_equality (AB AC : ℝ) (n : ℕ) :
  AB = 33 →
  AC = 21 →
  (∃ (D E : ℝ), 0 ≤ D ∧ D ≤ AB ∧ 0 ≤ E ∧ E ≤ AC ∧ D = n ∧ AB - D = n ∧ E = n ∧ AC - E = n) →
  (∃ (BC : ℕ), BC = 30) :=
by sorry

end NUMINAMATH_CALUDE_triangle_segment_equality_l3340_334085


namespace NUMINAMATH_CALUDE_petya_wins_l3340_334059

/-- Represents the game between Petya and Vasya -/
structure CandyGame where
  /-- Total number of candies in both boxes -/
  total_candies : Nat
  /-- Probability of Vasya getting two caramels -/
  prob_two_caramels : ℝ

/-- Petya has a higher chance of winning if his winning probability is greater than 0.5 -/
def petya_has_higher_chance (game : CandyGame) : Prop :=
  1 - (1 - game.prob_two_caramels) > 0.5

/-- Given the conditions of the game, prove that Petya has a higher chance of winning -/
theorem petya_wins (game : CandyGame) 
    (h1 : game.total_candies = 25)
    (h2 : game.prob_two_caramels = 0.54) : 
  petya_has_higher_chance game := by
  sorry

#check petya_wins

end NUMINAMATH_CALUDE_petya_wins_l3340_334059


namespace NUMINAMATH_CALUDE_largest_five_digit_number_with_product_180_l3340_334055

/-- Represents a five-digit number as a list of its digits -/
def FiveDigitNumber := List Nat

/-- Checks if a given list represents a valid five-digit number -/
def is_valid_five_digit_number (n : FiveDigitNumber) : Prop :=
  n.length = 5 ∧ n.all (· < 10) ∧ n.head! ≠ 0

/-- Computes the product of the digits of a number -/
def digit_product (n : FiveDigitNumber) : Nat :=
  n.prod

/-- Computes the sum of the digits of a number -/
def digit_sum (n : FiveDigitNumber) : Nat :=
  n.sum

/-- Compares two five-digit numbers -/
def is_greater (a b : FiveDigitNumber) : Prop :=
  a.foldl (fun acc d => acc * 10 + d) 0 > b.foldl (fun acc d => acc * 10 + d) 0

theorem largest_five_digit_number_with_product_180 :
  ∃ (M : FiveDigitNumber),
    is_valid_five_digit_number M ∧
    digit_product M = 180 ∧
    (∀ (N : FiveDigitNumber), is_valid_five_digit_number N → digit_product N = 180 → is_greater M N) ∧
    digit_sum M = 19 :=
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_number_with_product_180_l3340_334055


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l3340_334090

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l₂ (a x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) :
  (∀ x y, l₁ a x y ∧ l₂ a x y → (a * 1 + 2 * (a - 1) = 0)) →
  a = 2/3 :=
sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) :
  (∀ x y, l₁ a x y ∧ l₂ a x y → (a / 1 = 2 / (a - 1) ∧ a / 1 ≠ 6 / (a^2 - 1))) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l3340_334090


namespace NUMINAMATH_CALUDE_parabola_vertex_range_l3340_334089

/-- Represents a parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The vertex of a parabola -/
structure Vertex where
  s : ℝ
  t : ℝ

theorem parabola_vertex_range 
  (p : Parabola) 
  (v : Vertex) 
  (y₁ y₂ : ℝ)
  (h1 : p.a * (-2)^2 + p.b * (-2) + p.c = y₁)
  (h2 : p.a * 4^2 + p.b * 4 + p.c = y₂)
  (h3 : y₁ > y₂)
  (h4 : y₂ > v.t)
  : v.s > 1 ∧ v.s ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_range_l3340_334089


namespace NUMINAMATH_CALUDE_power_of_three_difference_l3340_334001

theorem power_of_three_difference : 3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_difference_l3340_334001


namespace NUMINAMATH_CALUDE_quarrel_between_opposite_houses_l3340_334083

/-- Represents a house in the square yard -/
inductive House : Type
| A : House
| B : House
| C : House
| D : House

/-- Represents a quarrel between two houses -/
structure Quarrel :=
  (house1 : House)
  (house2 : House)

/-- Checks if two houses are neighbors -/
def are_neighbors (h1 h2 : House) : Prop :=
  (h1 = House.A ∧ (h2 = House.B ∨ h2 = House.D)) ∨
  (h1 = House.B ∧ (h2 = House.A ∨ h2 = House.C)) ∨
  (h1 = House.C ∧ (h2 = House.B ∨ h2 = House.D)) ∨
  (h1 = House.D ∧ (h2 = House.A ∨ h2 = House.C))

/-- Checks if two houses are opposite -/
def are_opposite (h1 h2 : House) : Prop :=
  (h1 = House.A ∧ h2 = House.C) ∨ (h1 = House.C ∧ h2 = House.A) ∨
  (h1 = House.B ∧ h2 = House.D) ∨ (h1 = House.D ∧ h2 = House.B)

theorem quarrel_between_opposite_houses 
  (total_friends : Nat)
  (quarrels : List Quarrel)
  (h_total_friends : total_friends = 77)
  (h_quarrels_count : quarrels.length = 365)
  (h_different_houses : ∀ q ∈ quarrels, q.house1 ≠ q.house2)
  (h_no_neighbor_friends : ∀ h1 h2, are_neighbors h1 h2 → 
    ∃ q ∈ quarrels, (q.house1 = h1 ∧ q.house2 = h2) ∨ (q.house1 = h2 ∧ q.house2 = h1))
  : ∃ q ∈ quarrels, are_opposite q.house1 q.house2 :=
by sorry

end NUMINAMATH_CALUDE_quarrel_between_opposite_houses_l3340_334083


namespace NUMINAMATH_CALUDE_max_area_is_one_l3340_334053

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ
  h_nonzero : m ≠ 0

/-- The maximum area of triangle PMN for the given ellipse and line configuration -/
def max_area (e : Ellipse) (l : Line) : ℝ := 1

/-- Theorem stating the maximum area of triangle PMN is 1 -/
theorem max_area_is_one (e : Ellipse) (l : Line) 
  (h_focus : e.a^2 - e.b^2 = 9)
  (h_vertex : e.a^2 = 12)
  (h_line : l.c = 3) :
  max_area e l = 1 := by sorry

end NUMINAMATH_CALUDE_max_area_is_one_l3340_334053


namespace NUMINAMATH_CALUDE_third_layer_sugar_l3340_334039

/-- The amount of sugar needed for each layer of the cake -/
def sugar_amount (layer : Nat) : ℕ :=
  match layer with
  | 1 => 2  -- First layer requires 2 cups of sugar
  | 2 => 2 * sugar_amount 1  -- Second layer is twice as big as the first
  | 3 => 3 * sugar_amount 2  -- Third layer is three times larger than the second
  | _ => 0  -- We only consider 3 layers in this problem

theorem third_layer_sugar : sugar_amount 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_layer_sugar_l3340_334039


namespace NUMINAMATH_CALUDE_calculation_proof_l3340_334073

theorem calculation_proof : 4 * Real.sqrt 24 * (Real.sqrt 6 / 8) / Real.sqrt 3 - 3 * Real.sqrt 3 = - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3340_334073


namespace NUMINAMATH_CALUDE_snickers_bought_l3340_334011

-- Define the cost of a single Snickers
def snickers_cost : ℚ := 3/2

-- Define the number of M&M packs bought
def mm_packs : ℕ := 3

-- Define the total amount paid
def total_paid : ℚ := 20

-- Define the change received
def change : ℚ := 8

-- Define the relationship between M&M pack cost and Snickers cost
def mm_pack_cost (s : ℚ) : ℚ := 2 * s

-- Theorem to prove
theorem snickers_bought :
  ∃ (n : ℕ), (n : ℚ) * snickers_cost + mm_packs * mm_pack_cost snickers_cost = total_paid - change ∧ n = 2 := by
  sorry


end NUMINAMATH_CALUDE_snickers_bought_l3340_334011


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l3340_334029

theorem division_multiplication_problem : 
  ∃ x : ℝ, (244.8 / x = 51) ∧ (x * 15 = 72) :=
by sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l3340_334029


namespace NUMINAMATH_CALUDE_parallel_lines_and_not_always_parallel_planes_l3340_334021

-- Define the line equations
def line1 (a x y : ℝ) : Prop := a * x + 3 * y + 1 = 0
def line2 (a x y : ℝ) : Prop := 2 * x + (a + 1) * y + 1 = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y, line1 a x y ↔ line2 a x y

-- Define a plane
def Plane : Type := ℝ × ℝ × ℝ

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define distance between a point and a plane
def distance (p : Point) (plane : Plane) : ℝ := sorry

-- Define non-collinear points
def nonCollinear (p1 p2 p3 : Point) : Prop := sorry

-- Define parallel planes
def parallelPlanes (α β : Plane) : Prop := sorry

-- Statement of the theorem
theorem parallel_lines_and_not_always_parallel_planes :
  (∀ a, parallel a ↔ a = -3) ∧
  ¬(∀ α β : Plane, ∀ p1 p2 p3 : Point,
    nonCollinear p1 p2 p3 →
    distance p1 β = distance p2 β ∧ distance p2 β = distance p3 β →
    parallelPlanes α β) := by sorry

end NUMINAMATH_CALUDE_parallel_lines_and_not_always_parallel_planes_l3340_334021


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l3340_334050

def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {1, 3, 9}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3, 9} := by
  sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l3340_334050


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3340_334064

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I * Real.sqrt 3) = 1) :
  Complex.abs z = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3340_334064


namespace NUMINAMATH_CALUDE_range_theorem_fixed_point_theorem_l3340_334051

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Define the range function
def in_range (z : ℝ) : Prop := (6 - 2*Real.sqrt 3) / 3 ≤ z ∧ z ≤ (6 + 2*Real.sqrt 3) / 3

-- Theorem 1: Range of (y+3)/x for points on circle C
theorem range_theorem (x y : ℝ) : 
  circle_C x y → in_range ((y + 3) / x) :=
sorry

-- Define a point on line l
def point_on_line_l (t : ℝ) : ℝ × ℝ := (t, 2*t)

-- Define the circle passing through P, A, C, and B
def circle_PACB (t x y : ℝ) : Prop :=
  (x - (t + 2) / 2)^2 + (y - t)^2 = (5*t^2 - 4*t + 4) / 4

-- Theorem 2: Circle PACB passes through (2/5, 4/5)
theorem fixed_point_theorem (t : ℝ) :
  circle_PACB t (2/5) (4/5) :=
sorry

end NUMINAMATH_CALUDE_range_theorem_fixed_point_theorem_l3340_334051


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3340_334035

/-- The measure of an interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_interior_angles : ℝ := (n - 2) * 180  -- sum of interior angles formula
  let interior_angle : ℝ := sum_interior_angles / n  -- each interior angle measure
  135

/-- Proof of the theorem -/
lemma prove_regular_octagon_interior_angle : 
  regular_octagon_interior_angle = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3340_334035


namespace NUMINAMATH_CALUDE_divisible_by_five_problem_l3340_334015

theorem divisible_by_five_problem (n : ℕ) : 
  n % 5 = 0 ∧ n / 5 = 96 → (n + 17) * 69 = 34293 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_problem_l3340_334015


namespace NUMINAMATH_CALUDE_smallest_angle_trig_equation_l3340_334072

theorem smallest_angle_trig_equation :
  let θ := Real.pi / 14
  (∀ φ > 0, φ < θ → Real.sin (3 * φ) * Real.sin (4 * φ) ≠ Real.cos (3 * φ) * Real.cos (4 * φ)) ∧
  Real.sin (3 * θ) * Real.sin (4 * θ) = Real.cos (3 * θ) * Real.cos (4 * θ) := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_trig_equation_l3340_334072


namespace NUMINAMATH_CALUDE_probability_not_all_same_dice_probability_not_all_same_five_eight_sided_dice_l3340_334084

theorem probability_not_all_same_dice (n : ℕ) (s : ℕ) (hn : n > 0) (hs : s > 0) : 
  1 - (s : ℚ) / (s ^ n : ℚ) = (s ^ n - s : ℚ) / (s ^ n : ℚ) :=
by sorry

-- The probability of not getting all the same numbers when rolling five fair 8-sided dice
theorem probability_not_all_same_five_eight_sided_dice : 
  1 - (8 : ℚ) / (8^5 : ℚ) = 4095 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_probability_not_all_same_dice_probability_not_all_same_five_eight_sided_dice_l3340_334084


namespace NUMINAMATH_CALUDE_add_10000_seconds_to_5_45_00_l3340_334040

def seconds_to_time (seconds : ℕ) : ℕ × ℕ × ℕ :=
  let total_minutes := seconds / 60
  let remaining_seconds := seconds % 60
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes, remaining_seconds)

def add_time (start : ℕ × ℕ × ℕ) (duration : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (start_h, start_m, start_s) := start
  let (duration_h, duration_m, duration_s) := duration
  let total_seconds := start_s + start_m * 60 + start_h * 3600 +
                       duration_s + duration_m * 60 + duration_h * 3600
  seconds_to_time total_seconds

theorem add_10000_seconds_to_5_45_00 :
  add_time (5, 45, 0) (seconds_to_time 10000) = (8, 31, 40) :=
sorry

end NUMINAMATH_CALUDE_add_10000_seconds_to_5_45_00_l3340_334040


namespace NUMINAMATH_CALUDE_lucas_150_mod_5_l3340_334031

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

/-- The 150th term of the Lucas sequence modulo 5 is equal to 3 -/
theorem lucas_150_mod_5 : lucas 149 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lucas_150_mod_5_l3340_334031


namespace NUMINAMATH_CALUDE_union_of_sets_l3340_334063

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | 3 * x^2 + p * x - 7 = 0}
def B (q : ℝ) : Set ℝ := {x | 3 * x^2 - 7 * x + q = 0}

-- State the theorem
theorem union_of_sets (p q : ℝ) :
  (∃ (p q : ℝ), A p ∩ B q = {-1/3}) →
  (∃ (p q : ℝ), A p ∪ B q = {-1/3, 8/3, 7}) :=
by sorry

end NUMINAMATH_CALUDE_union_of_sets_l3340_334063


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l3340_334038

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_increasing_neg : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y

-- Define the solution set
def solution_set := {x : ℝ | f (3 - 2*x) > f 1}

-- State the theorem
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l3340_334038


namespace NUMINAMATH_CALUDE_mike_distance_equals_46_l3340_334005

def mike_start_fee : ℚ := 2.5
def annie_start_fee : ℚ := 2.5
def annie_toll_fee : ℚ := 5
def per_mile_rate : ℚ := 0.25
def annie_distance : ℚ := 26

theorem mike_distance_equals_46 (mike_distance : ℚ) :
  mike_start_fee + per_mile_rate * mike_distance =
  annie_start_fee + annie_toll_fee + per_mile_rate * annie_distance →
  mike_distance = 46 := by
sorry

end NUMINAMATH_CALUDE_mike_distance_equals_46_l3340_334005


namespace NUMINAMATH_CALUDE_house_cost_is_280k_l3340_334079

/-- Calculates the total cost of a house given the initial deposit, mortgage duration, and monthly payment. -/
def house_cost (deposit : ℕ) (duration_years : ℕ) (monthly_payment : ℕ) : ℕ :=
  deposit + duration_years * 12 * monthly_payment

/-- Proves that the total cost of the house is $280,000 given the specified conditions. -/
theorem house_cost_is_280k :
  house_cost 40000 10 2000 = 280000 :=
by sorry

end NUMINAMATH_CALUDE_house_cost_is_280k_l3340_334079


namespace NUMINAMATH_CALUDE_system_solution_l3340_334062

theorem system_solution :
  ∀ x y z : ℝ,
  (x * y + x * z = 8 - x^2) ∧
  (x * y + y * z = 12 - y^2) ∧
  (y * z + z * x = -4 - z^2) →
  ((x = 2 ∧ y = 3 ∧ z = -1) ∨ (x = -2 ∧ y = -3 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3340_334062


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l3340_334076

/-- Proves that adding 1.2 liters of pure alcohol to a 6-liter solution
    that is 40% alcohol results in a 50% alcohol solution -/
theorem alcohol_solution_proof (initial_volume : ℝ) (initial_concentration : ℝ)
    (added_alcohol : ℝ) (final_concentration : ℝ)
    (h1 : initial_volume = 6)
    (h2 : initial_concentration = 0.4)
    (h3 : added_alcohol = 1.2)
    (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) /
  (initial_volume + added_alcohol) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l3340_334076


namespace NUMINAMATH_CALUDE_cannot_be_B_l3340_334058

-- Define set A
def A : Set ℝ := {x : ℝ | x ≠ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -1}

-- Theorem statement
theorem cannot_be_B (h : A ∪ B = Set.univ) : False := by
  sorry

end NUMINAMATH_CALUDE_cannot_be_B_l3340_334058


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l3340_334077

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the perimeter function
def perimeter (q : Quadrilateral) : ℝ :=
  sorry

-- Define the perpendicular property
def perpendicular (v w : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem quadrilateral_perimeter :
  ∀ (q : Quadrilateral),
    perpendicular (q.B - q.A) (q.C - q.B) →
    perpendicular (q.C - q.D) (q.C - q.B) →
    ‖q.B - q.A‖ = 9 →
    ‖q.D - q.C‖ = 4 →
    ‖q.C - q.B‖ = 12 →
    perimeter q = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l3340_334077


namespace NUMINAMATH_CALUDE_total_cats_l3340_334008

theorem total_cats (white : ℕ) (black : ℕ) (gray : ℕ) 
  (h_white : white = 2) 
  (h_black : black = 10) 
  (h_gray : gray = 3) : 
  white + black + gray = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l3340_334008


namespace NUMINAMATH_CALUDE_students_at_start_correct_l3340_334033

/-- The number of students at the start of the year in fourth grade -/
def students_at_start : ℕ := 10

/-- The number of students added during the year -/
def students_added : ℝ := 4.0

/-- The number of new students who came to school -/
def new_students : ℝ := 42.0

/-- The total number of students at the end of the year -/
def students_at_end : ℕ := 56

/-- Theorem stating that the number of students at the start of the year is correct -/
theorem students_at_start_correct :
  students_at_start + (students_added + new_students) = students_at_end := by
  sorry

end NUMINAMATH_CALUDE_students_at_start_correct_l3340_334033


namespace NUMINAMATH_CALUDE_jellybean_purchase_l3340_334046

theorem jellybean_purchase (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n ≥ 164 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_purchase_l3340_334046


namespace NUMINAMATH_CALUDE_optimal_system_is_best_l3340_334074

/-- Represents a monetary system with three coin denominations -/
structure MonetarySystem where
  d1 : ℕ
  d2 : ℕ
  d3 : ℕ
  h1 : 0 < d1 ∧ d1 < d2 ∧ d2 < d3
  h2 : d3 ≤ 100

/-- Calculates the minimum number of coins required for a given monetary system -/
def minCoinsRequired (system : MonetarySystem) : ℕ := sorry

/-- The optimal monetary system -/
def optimalSystem : MonetarySystem :=
  { d1 := 1, d2 := 7, d3 := 14,
    h1 := by simp,
    h2 := by simp }

theorem optimal_system_is_best :
  (∀ system : MonetarySystem, minCoinsRequired system ≥ minCoinsRequired optimalSystem) ∧
  minCoinsRequired optimalSystem = 14 := by sorry

end NUMINAMATH_CALUDE_optimal_system_is_best_l3340_334074


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3340_334037

/-- Returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Given two natural numbers m and n, returns true if m has a units digit of 4 -/
def hasUnitsDigitFour (m : ℕ) : Prop := unitsDigit m = 4

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : hasUnitsDigitFour m) :
  unitsDigit n = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3340_334037


namespace NUMINAMATH_CALUDE_probability_neighboring_points_l3340_334045

/-- The probability of choosing neighboring points on a circle -/
theorem probability_neighboring_points (n : ℕ) (h : n ≥ 3) :
  (2 : ℚ) / (n - 1) = (n : ℚ) / (n.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_probability_neighboring_points_l3340_334045
