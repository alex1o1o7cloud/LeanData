import Mathlib

namespace NUMINAMATH_CALUDE_tiffany_cans_l2439_243930

theorem tiffany_cans (bags_monday : ℕ) : bags_monday = 12 :=
  by
    have h1 : bags_monday + 12 = 2 * bags_monday := by sorry
    -- The number of bags on Tuesday (bags_monday + 12) is double the number of bags on Monday (2 * bags_monday)
    sorry

end NUMINAMATH_CALUDE_tiffany_cans_l2439_243930


namespace NUMINAMATH_CALUDE_water_tank_problem_l2439_243944

/-- Calculates the water volume in a tank after a given number of hours, 
    with specified initial volume, loss rate, and water additions. -/
def water_volume (initial_volume : ℝ) (loss_rate : ℝ) (additions : List ℝ) : ℝ :=
  initial_volume - loss_rate * additions.length + additions.sum

/-- The water volume problem -/
theorem water_tank_problem : 
  let initial_volume : ℝ := 40
  let loss_rate : ℝ := 2
  let additions : List ℝ := [0, 0, 1, 3]
  water_volume initial_volume loss_rate additions = 36 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_problem_l2439_243944


namespace NUMINAMATH_CALUDE_percentage_of_C_grades_l2439_243936

def gradeC (score : ℕ) : Bool :=
  76 ≤ score ∧ score ≤ 85

def scores : List ℕ := [93, 71, 55, 98, 81, 89, 77, 72, 78, 62, 87, 80, 68, 82, 91, 67, 76, 84, 70, 95]

theorem percentage_of_C_grades (scores : List ℕ) : 
  (100 * (scores.filter gradeC).length) / scores.length = 35 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_C_grades_l2439_243936


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_3_and_5_l2439_243990

theorem smallest_four_digit_multiple_of_3_and_5 : ∃ n : ℕ,
  (n ≥ 1000 ∧ n < 10000) ∧  -- 4-digit number
  n % 3 = 0 ∧               -- multiple of 3
  n % 5 = 0 ∧               -- multiple of 5
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 3 = 0 ∧ m % 5 = 0) → n ≤ m) ∧
  n = 1005 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_3_and_5_l2439_243990


namespace NUMINAMATH_CALUDE_discount_is_eleven_l2439_243941

/-- The discount on the first two books when ordering four books online --/
def discount_on_first_two_books : ℝ :=
  let free_shipping_threshold : ℝ := 50
  let book1_price : ℝ := 13
  let book2_price : ℝ := 15
  let book3_price : ℝ := 10
  let book4_price : ℝ := 10
  let additional_spend_needed : ℝ := 9
  let total_without_discount : ℝ := book1_price + book2_price + book3_price + book4_price
  let total_with_discount : ℝ := free_shipping_threshold + additional_spend_needed
  total_with_discount - total_without_discount

/-- Theorem stating that the discount on the first two books is $11.00 --/
theorem discount_is_eleven : discount_on_first_two_books = 11 := by
  sorry

end NUMINAMATH_CALUDE_discount_is_eleven_l2439_243941


namespace NUMINAMATH_CALUDE_first_negative_term_l2439_243988

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem first_negative_term (a₁ d : ℝ) (h₁ : a₁ = 51) (h₂ : d = -4) :
  ∀ k < 14, arithmetic_sequence a₁ d k ≥ 0 ∧
  arithmetic_sequence a₁ d 14 < 0 := by
sorry

end NUMINAMATH_CALUDE_first_negative_term_l2439_243988


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2439_243960

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₁ + a₉ = 10, prove that a₅ = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 9 = 10) : 
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2439_243960


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_80_l2439_243913

theorem twenty_percent_greater_than_80 (x : ℝ) : 
  x = 80 * (1 + 0.2) → x = 96 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_80_l2439_243913


namespace NUMINAMATH_CALUDE_cotton_amount_l2439_243980

/-- Given:
  * Kevin plants corn and cotton
  * He harvests 30 pounds of corn and x pounds of cotton
  * Corn sells for $5 per pound
  * Cotton sells for $10 per pound
  * Total revenue from selling all corn and cotton is $640
Prove that x = 49 -/
theorem cotton_amount (x : ℝ) : 
  (30 * 5 + x * 10 = 640) → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_cotton_amount_l2439_243980


namespace NUMINAMATH_CALUDE_calculate_expression_l2439_243997

theorem calculate_expression : (35 / (5 * 2 + 5)) * 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2439_243997


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2439_243909

/-- Given a rectangular box with dimensions a, b, and c, if the sum of the lengths
    of the twelve edges is 140 and the distance from one corner to the farthest
    corner is 21, then the total surface area of the box is 784. -/
theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : a + b + c = 35)
  (diagonal : a^2 + b^2 + c^2 = 441) :
  2 * (a * b + b * c + c * a) = 784 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2439_243909


namespace NUMINAMATH_CALUDE_equation_solution_l2439_243933

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- State the theorem
theorem equation_solution :
  ∃ (x : ℝ), (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ∧ x = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2439_243933


namespace NUMINAMATH_CALUDE_max_k_inequality_l2439_243993

theorem max_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_condition : a^2 + b^2 + c^2 = 2*(a*b + b*c + c*a)) :
  ∃ (k : ℝ), k > 0 ∧ k = 2 ∧
  ∀ (k' : ℝ), k' > 0 →
    (1 / (k'*a*b + c^2) + 1 / (k'*b*c + a^2) + 1 / (k'*c*a + b^2) ≥ (k' + 3) / (a^2 + b^2 + c^2)) →
    k' ≤ k :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l2439_243993


namespace NUMINAMATH_CALUDE_complex_sum_simplification_l2439_243926

theorem complex_sum_simplification :
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^12 + z₂^12 = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_simplification_l2439_243926


namespace NUMINAMATH_CALUDE_expression_simplification_l2439_243902

theorem expression_simplification (x y : ℝ) : 
  2 * x^2 * y - 4 * x * y^2 - (-3 * x * y^2 + x^2 * y) = x^2 * y - x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2439_243902


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l2439_243929

/-- Given a cylindrical tank with radius 5 inches and height 6 inches, 
    this theorem proves that increasing the radius by x inches 
    results in the same volume increase as increasing the height by 2x inches 
    when x = 10/3. -/
theorem cylinder_volume_increase (x : ℝ) : x = 10/3 ↔ 
  π * (5 + x)^2 * 6 = π * 5^2 * (6 + 2*x) := by
  sorry

#check cylinder_volume_increase

end NUMINAMATH_CALUDE_cylinder_volume_increase_l2439_243929


namespace NUMINAMATH_CALUDE_crow_eating_time_l2439_243973

/-- The time it takes for a crow to eat a fraction of nuts -/
def eat_time (total_fraction : ℚ) (time : ℚ) : ℚ := total_fraction / time

theorem crow_eating_time :
  let quarter_time : ℚ := 5
  let quarter_fraction : ℚ := 1/4
  let fifth_fraction : ℚ := 1/5
  let rate := eat_time quarter_fraction quarter_time
  eat_time fifth_fraction rate = 4 := by sorry

end NUMINAMATH_CALUDE_crow_eating_time_l2439_243973


namespace NUMINAMATH_CALUDE_all_cells_equal_l2439_243906

/-- Represents a 10x10 board with integer values -/
def Board := Fin 10 → Fin 10 → ℤ

/-- Predicate to check if a board satisfies the given conditions -/
def satisfies_conditions (b : Board) : Prop :=
  ∃ d : ℤ,
    (∀ i : Fin 10, b i i = d) ∧
    (∀ i j : Fin 10, b i j ≤ d)

/-- Theorem stating that if a board satisfies the conditions, all cells are equal -/
theorem all_cells_equal (b : Board) (h : satisfies_conditions b) :
    ∃ d : ℤ, ∀ i j : Fin 10, b i j = d := by
  sorry


end NUMINAMATH_CALUDE_all_cells_equal_l2439_243906


namespace NUMINAMATH_CALUDE_max_diagonal_length_l2439_243958

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_diagonal_length (PQRS : Quadrilateral) : 
  distance PQRS.P PQRS.Q = 7 →
  distance PQRS.Q PQRS.R = 13 →
  distance PQRS.R PQRS.S = 7 →
  distance PQRS.S PQRS.P = 10 →
  ∃ (pr : ℕ), pr ≤ 19 ∧ 
    distance PQRS.P PQRS.R = pr ∧
    ∀ (x : ℕ), distance PQRS.P PQRS.R = x → x ≤ pr :=
by sorry

end NUMINAMATH_CALUDE_max_diagonal_length_l2439_243958


namespace NUMINAMATH_CALUDE_solution_set_implies_a_bound_l2439_243965

theorem solution_set_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |x + 1| ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_bound_l2439_243965


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_three_l2439_243924

theorem subset_implies_a_equals_three (A B : Set ℕ) (a : ℕ) 
  (hA : A = {1, 3}) 
  (hB : B = {1, 2, a}) 
  (hSubset : A ⊆ B) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_three_l2439_243924


namespace NUMINAMATH_CALUDE_function_symmetry_l2439_243950

/-- Given a function f and a real number a, proves that if f(x) = |x|(e^(ax) - e^(-ax)) + 2 
    and f(10) = 1, then f(-10) = 3 -/
theorem function_symmetry (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f x = |x| * (Real.exp (a * x) - Real.exp (-a * x)) + 2)
    (h2 : f 10 = 1) : 
  f (-10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2439_243950


namespace NUMINAMATH_CALUDE_josie_cabinet_unlock_time_l2439_243963

/-- Represents the shopping trip details -/
structure ShoppingTrip where
  total_time : ℕ
  shopping_time : ℕ
  cart_wait_time : ℕ
  restocking_wait_time : ℕ
  checkout_wait_time : ℕ

/-- Calculates the time spent waiting for the cabinet to be unlocked -/
def cabinet_unlock_time (trip : ShoppingTrip) : ℕ :=
  trip.total_time - trip.shopping_time - trip.cart_wait_time - trip.restocking_wait_time - trip.checkout_wait_time

/-- Theorem stating that Josie waited 13 minutes for the cabinet to be unlocked -/
theorem josie_cabinet_unlock_time :
  let trip := ShoppingTrip.mk 90 42 3 14 18
  cabinet_unlock_time trip = 13 := by sorry

end NUMINAMATH_CALUDE_josie_cabinet_unlock_time_l2439_243963


namespace NUMINAMATH_CALUDE_curve_intersection_perpendicular_l2439_243932

/-- The curve C: x^2 + y^2 - 2x - 4y + m = 0 -/
def curve_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

/-- The line: x + 2y - 3 = 0 -/
def line (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem curve_intersection_perpendicular (m : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    curve_C x1 y1 m ∧ curve_C x2 y2 m ∧
    line x1 y1 ∧ line x2 y2 ∧
    perpendicular x1 y1 x2 y2) →
  m = 12/5 := by sorry

end NUMINAMATH_CALUDE_curve_intersection_perpendicular_l2439_243932


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l2439_243991

theorem p_or_q_is_true : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≥ 1) ∨ 
  (∃ x : ℝ, x^2 + x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l2439_243991


namespace NUMINAMATH_CALUDE_traces_bag_weight_is_two_l2439_243921

/-- The weight of one of Trace's shopping bags -/
def traces_bag_weight (
  trace_bag_count : ℕ
  ) (
  gordon_bag1_weight : ℕ
  ) (
  gordon_bag2_weight : ℕ
  ) : ℚ :=
  (gordon_bag1_weight + gordon_bag2_weight) / trace_bag_count

/-- Theorem stating the weight of one of Trace's shopping bags -/
theorem traces_bag_weight_is_two :
  traces_bag_weight 5 3 7 = 2 := by
  sorry

#eval traces_bag_weight 5 3 7

end NUMINAMATH_CALUDE_traces_bag_weight_is_two_l2439_243921


namespace NUMINAMATH_CALUDE_geese_ratio_l2439_243915

/-- Represents the number of ducks and geese bought by a person -/
structure DucksAndGeese where
  ducks : ℕ
  geese : ℕ

/-- The problem setup -/
def market_problem (lily rayden : DucksAndGeese) : Prop :=
  rayden.ducks = 3 * lily.ducks ∧
  lily.ducks = 20 ∧
  lily.geese = 10 ∧
  rayden.ducks + rayden.geese = lily.ducks + lily.geese + 70

/-- The theorem to prove -/
theorem geese_ratio (lily rayden : DucksAndGeese) 
  (h : market_problem lily rayden) : 
  rayden.geese = 4 * lily.geese := by
  sorry


end NUMINAMATH_CALUDE_geese_ratio_l2439_243915


namespace NUMINAMATH_CALUDE_pipe_cutting_time_l2439_243994

/-- The time needed to cut a pipe into sections -/
def cut_time (sections : ℕ) (time_per_cut : ℕ) : ℕ :=
  (sections - 1) * time_per_cut

/-- Theorem: The time needed to cut a pipe into 5 sections is 24 minutes -/
theorem pipe_cutting_time : cut_time 5 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_pipe_cutting_time_l2439_243994


namespace NUMINAMATH_CALUDE_spiral_staircase_handrail_length_l2439_243984

/-- The length of a spiral staircase handrail -/
theorem spiral_staircase_handrail_length 
  (turn_angle : Real) 
  (rise : Real) 
  (radius : Real) 
  (handrail_length : Real) : 
  turn_angle = 315 ∧ 
  rise = 12 ∧ 
  radius = 4 → 
  abs (handrail_length - Real.sqrt (rise^2 + (turn_angle / 360 * 2 * Real.pi * radius)^2)) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_spiral_staircase_handrail_length_l2439_243984


namespace NUMINAMATH_CALUDE_equation_is_circle_l2439_243949

/-- A conic section type -/
inductive ConicSection
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines if an equation represents a circle -/
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ h k r, ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def equation (x y : ℝ) : Prop :=
  (x - 3)^2 = -(3*y + 1)^2 + 45

theorem equation_is_circle :
  is_circle equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_circle_l2439_243949


namespace NUMINAMATH_CALUDE_expression_simplification_l2439_243970

theorem expression_simplification (a b : ℝ) (h : a * b ≠ 0) :
  (3 * a^3 * b - 12 * a^2 * b^2 - 6 * a * b^3) / (-3 * a * b) - 4 * a * b = -a^2 + 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2439_243970


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2439_243911

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), k > 0 ∧ x = 2*k ∧ y = k ∧ z = k :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2439_243911


namespace NUMINAMATH_CALUDE_concentric_circles_radius_l2439_243971

theorem concentric_circles_radius (r₁ r₂ AB : ℝ) : 
  r₁ > 0 → r₂ > 0 →
  r₂ / r₁ = 7 / 3 →
  AB = 20 →
  ∃ (AC BC : ℝ), 
    AC = 2 * r₂ ∧
    BC^2 + AB^2 = AC^2 ∧
    BC^2 = r₂^2 - r₁^2 →
  r₂ = 70 / 3 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_l2439_243971


namespace NUMINAMATH_CALUDE_largest_product_of_three_l2439_243998

def S : Finset Int := {-4, -3, -1, 5, 6}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z → x * y * z ≤ 72) ∧
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 72) :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l2439_243998


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2439_243987

theorem line_intersects_circle (a : ℝ) : ∃ (x y : ℝ), 
  y = a * x - a + 1 ∧ x^2 + y^2 = 8 := by
  sorry

#check line_intersects_circle

end NUMINAMATH_CALUDE_line_intersects_circle_l2439_243987


namespace NUMINAMATH_CALUDE_greatest_base12_divisible_by_7_l2439_243928

/-- Converts a base 12 number to decimal --/
def base12ToDecimal (a b c : Nat) : Nat :=
  a * 12^2 + b * 12 + c

/-- Checks if a number is divisible by 7 --/
def isDivisibleBy7 (n : Nat) : Prop :=
  n % 7 = 0

/-- Theorem: BB6₁₂ is the greatest 3-digit base 12 positive integer divisible by 7 --/
theorem greatest_base12_divisible_by_7 :
  let bb6 := base12ToDecimal 11 11 6
  isDivisibleBy7 bb6 ∧
  ∀ n, n > bb6 → n ≤ base12ToDecimal 11 11 11 →
    ¬(isDivisibleBy7 n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_base12_divisible_by_7_l2439_243928


namespace NUMINAMATH_CALUDE_power_four_2024_mod_11_l2439_243952

theorem power_four_2024_mod_11 : 4^2024 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_four_2024_mod_11_l2439_243952


namespace NUMINAMATH_CALUDE_tom_waits_six_months_l2439_243983

/-- Represents Tom's medication and doctor visit costs --/
structure MedicationCosts where
  pills_per_day : ℕ
  doctor_visit_cost : ℕ
  pill_cost : ℕ
  insurance_coverage : ℚ
  total_annual_cost : ℕ

/-- Calculates the number of months between doctor visits --/
def months_between_visits (costs : MedicationCosts) : ℚ :=
  let annual_medication_cost := costs.pills_per_day * 365 * costs.pill_cost * (1 - costs.insurance_coverage)
  let annual_doctor_cost := costs.total_annual_cost - annual_medication_cost
  let visits_per_year := annual_doctor_cost / costs.doctor_visit_cost
  12 / visits_per_year

/-- Theorem stating that Tom waits 6 months between doctor visits --/
theorem tom_waits_six_months (costs : MedicationCosts) 
  (h1 : costs.pills_per_day = 2)
  (h2 : costs.doctor_visit_cost = 400)
  (h3 : costs.pill_cost = 5)
  (h4 : costs.insurance_coverage = 4/5)
  (h5 : costs.total_annual_cost = 1530) :
  months_between_visits costs = 6 := by
  sorry


end NUMINAMATH_CALUDE_tom_waits_six_months_l2439_243983


namespace NUMINAMATH_CALUDE_power_of_negative_one_difference_l2439_243979

theorem power_of_negative_one_difference : (-1)^2004 - (-1)^2003 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_one_difference_l2439_243979


namespace NUMINAMATH_CALUDE_blackboard_numbers_l2439_243981

theorem blackboard_numbers (n : ℕ) (h1 : n = 2004) (h2 : (List.range n).sum % 167 = 0)
  (x : ℕ) (h3 : x ≤ 166) (h4 : (x + 999) % 167 = 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l2439_243981


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l2439_243976

theorem rectangle_dimension_change (original_length original_width : ℝ) 
  (h_positive_length : original_length > 0)
  (h_positive_width : original_width > 0) :
  let new_length := 1.4 * original_length
  let new_width := original_width * (1 - 0.2857)
  new_length * new_width = original_length * original_width := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l2439_243976


namespace NUMINAMATH_CALUDE_lemonade_glasses_l2439_243947

/-- Calculates the total number of glasses of lemonade that can be served -/
def total_glasses (glasses_per_pitcher : ℕ) (num_pitchers : ℕ) : ℕ :=
  glasses_per_pitcher * num_pitchers

/-- Theorem: Given 5 glasses per pitcher and 6 pitchers, the total glasses served is 30 -/
theorem lemonade_glasses : total_glasses 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_glasses_l2439_243947


namespace NUMINAMATH_CALUDE_a_18_value_l2439_243919

/-- An equal sum sequence with common sum c -/
def EqualSumSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = c

theorem a_18_value (a : ℕ → ℝ) (h : EqualSumSequence a 5) (h1 : a 1 = 2) :
  a 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_18_value_l2439_243919


namespace NUMINAMATH_CALUDE_cos_inequality_range_l2439_243959

theorem cos_inequality_range (θ : Real) : 
  θ ∈ Set.Icc (-Real.pi) Real.pi →
  (3 * Real.sqrt 2 * Real.cos (θ + Real.pi / 4) < 4 * (Real.sin θ ^ 3 - Real.cos θ ^ 3)) ↔
  θ ∈ Set.Ioc (-Real.pi) (-3 * Real.pi / 4) ∪ Set.Ioo (Real.pi / 4) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cos_inequality_range_l2439_243959


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l2439_243904

/-- Represents the age ratio between a man and his son after two years -/
def age_ratio (son_age : ℕ) (age_difference : ℕ) : ℚ :=
  let man_age := son_age + age_difference
  (man_age + 2) / (son_age + 2)

/-- Theorem stating the age ratio between a man and his son after two years -/
theorem man_son_age_ratio :
  age_ratio 22 24 = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l2439_243904


namespace NUMINAMATH_CALUDE_min_value_theorem_l2439_243946

def f (a x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

theorem min_value_theorem (a : ℝ) : 
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f a x ≤ f a y) ∧ 
  (∀ x ∈ Set.Icc 0 1, f a x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f a x = 2) ↔ 
  a = 0 ∨ a = 3 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2439_243946


namespace NUMINAMATH_CALUDE_tricycle_wheels_l2439_243935

theorem tricycle_wheels (num_bicycles num_tricycles bicycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 16)
  (h2 : num_tricycles = 7)
  (h3 : bicycle_wheels = 2)
  (h4 : total_wheels = 53)
  : (total_wheels - num_bicycles * bicycle_wheels) / num_tricycles = 3 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_wheels_l2439_243935


namespace NUMINAMATH_CALUDE_pqr_value_l2439_243957

theorem pqr_value (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 390 / (p * q * r) = 1) :
  p * q * r = 1680 := by
sorry

end NUMINAMATH_CALUDE_pqr_value_l2439_243957


namespace NUMINAMATH_CALUDE_function_inequality_l2439_243953

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, 3 * f x - f' x > 0) :
  f 1 < Real.exp 3 * f 0 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l2439_243953


namespace NUMINAMATH_CALUDE_lee_proposal_time_l2439_243945

/-- Calculates the number of months needed to save for an engagement ring based on annual salary and monthly savings. -/
def months_to_save_for_ring (annual_salary : ℕ) (monthly_savings : ℕ) : ℕ :=
  let monthly_salary := annual_salary / 12
  let ring_cost := 2 * monthly_salary
  ring_cost / monthly_savings

/-- Proves that given the specified conditions, it takes 10 months to save for the ring. -/
theorem lee_proposal_time : months_to_save_for_ring 60000 1000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lee_proposal_time_l2439_243945


namespace NUMINAMATH_CALUDE_negative_one_is_root_l2439_243966

/-- The polynomial f(x) = x^3 + x^2 - 6x - 6 -/
def f (x : ℝ) : ℝ := x^3 + x^2 - 6*x - 6

/-- Theorem: -1 is a root of the polynomial f(x) = x^3 + x^2 - 6x - 6 -/
theorem negative_one_is_root : f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_is_root_l2439_243966


namespace NUMINAMATH_CALUDE_diamond_value_l2439_243925

/-- Given that ◇5₉ = ◇3₁₁ where ◇ represents a digit, prove that ◇ = 1 -/
theorem diamond_value : ∃ (d : ℕ), d < 10 ∧ d * 9 + 5 = d * 11 + 3 ∧ d = 1 := by sorry

end NUMINAMATH_CALUDE_diamond_value_l2439_243925


namespace NUMINAMATH_CALUDE_waiter_customers_theorem_l2439_243948

/-- The number of initial customers that satisfies the given condition -/
def initial_customers : ℕ := 33

/-- The condition given in the problem -/
theorem waiter_customers_theorem : 
  (initial_customers - 31 + 26 = 28) := by sorry

end NUMINAMATH_CALUDE_waiter_customers_theorem_l2439_243948


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2439_243975

theorem concentric_circles_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : π * b^2 - π * a^2 = 4 * (π * a^2)) : a / b = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2439_243975


namespace NUMINAMATH_CALUDE_special_function_is_negation_l2439_243982

/-- A function satisfying the given functional equation -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x - y) = f x + f (f y - f (-x)) + x

/-- The main theorem: if f satisfies the functional equation, then f(x) = -x for all x -/
theorem special_function_is_negation (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f x = -x :=
sorry

end NUMINAMATH_CALUDE_special_function_is_negation_l2439_243982


namespace NUMINAMATH_CALUDE_square_difference_l2439_243931

theorem square_difference (m n : ℕ+) 
  (h : (2001 : ℕ) * m ^ 2 + m = (2002 : ℕ) * n ^ 2 + n) : 
  ∃ k : ℕ, m - n = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2439_243931


namespace NUMINAMATH_CALUDE_unique_sequence_exists_l2439_243961

def sequence_property (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a n * a (n + 2) = (a (n + 1))^3 + 1

theorem unique_sequence_exists : ∃! a : ℕ → ℕ, sequence_property a := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_exists_l2439_243961


namespace NUMINAMATH_CALUDE_equation_solution_l2439_243951

theorem equation_solution (a : ℕ) : 
  (∃ x y : ℕ, (x + y)^2 + 3*x + y = 2*a) ↔ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2439_243951


namespace NUMINAMATH_CALUDE_walking_problem_l2439_243967

/-- Represents the ratio of steps taken by the good walker to the bad walker in the same time -/
def step_ratio : ℚ := 100 / 60

/-- Represents the head start of the bad walker in steps -/
def head_start : ℕ := 100

/-- Represents the walking problem described in "Nine Chapters on the Mathematical Art" -/
theorem walking_problem (x y : ℚ) :
  (x - y = head_start) ∧ (x = step_ratio * y) ↔
  x - y = head_start ∧ x = (100 : ℚ) / 60 * y :=
sorry

end NUMINAMATH_CALUDE_walking_problem_l2439_243967


namespace NUMINAMATH_CALUDE_rhinoceros_preserve_watering_area_l2439_243917

theorem rhinoceros_preserve_watering_area 
  (initial_population : ℕ)
  (grazing_area_per_rhino : ℕ)
  (population_increase_percent : ℚ)
  (total_preserve_area : ℕ) :
  initial_population = 8000 →
  grazing_area_per_rhino = 100 →
  population_increase_percent = 1/10 →
  total_preserve_area = 890000 →
  let increased_population := initial_population + (initial_population * population_increase_percent).floor
  let total_grazing_area := increased_population * grazing_area_per_rhino
  let watering_area := total_preserve_area - total_grazing_area
  watering_area = 10000 := by
sorry

end NUMINAMATH_CALUDE_rhinoceros_preserve_watering_area_l2439_243917


namespace NUMINAMATH_CALUDE_fraction_problem_l2439_243905

theorem fraction_problem (a b : ℚ) : 
  b / (a - 2) = 3 / 4 →
  b / (a + 9) = 5 / 7 →
  b / a = 165 / 222 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l2439_243905


namespace NUMINAMATH_CALUDE_election_invalid_votes_l2439_243995

theorem election_invalid_votes 
  (total_polled : ℕ) 
  (vote_difference : ℕ) 
  (losing_percentage : ℚ) :
  total_polled = 850 →
  vote_difference = 500 →
  losing_percentage = 1/5 →
  (∃ (invalid_votes : ℕ), invalid_votes = 17) :=
by sorry

end NUMINAMATH_CALUDE_election_invalid_votes_l2439_243995


namespace NUMINAMATH_CALUDE_coin_machine_theorem_l2439_243968

/-- Represents the coin-changing machine's rules --/
structure CoinMachine where
  quarter_to_nickels : ℕ → ℕ
  nickel_to_pennies : ℕ → ℕ
  penny_to_quarters : ℕ → ℕ

/-- Represents the possible amounts in cents --/
def possible_amounts (m : CoinMachine) (n : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, x = 1 + 74 * k}

/-- The set of given options in cents --/
def given_options : Set ℕ := {175, 325, 449, 549, 823}

theorem coin_machine_theorem (m : CoinMachine) 
  (h1 : m.quarter_to_nickels 1 = 5)
  (h2 : m.nickel_to_pennies 1 = 5)
  (h3 : m.penny_to_quarters 1 = 3) :
  given_options ∩ (possible_amounts m 1) = {823} := by
  sorry

end NUMINAMATH_CALUDE_coin_machine_theorem_l2439_243968


namespace NUMINAMATH_CALUDE_opposite_number_problem_l2439_243922

theorem opposite_number_problem (x : ℤ) : (x + 1 = -(-10)) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_problem_l2439_243922


namespace NUMINAMATH_CALUDE_sum_of_segment_lengths_divisible_by_four_l2439_243908

/-- Represents a square sheet of graph paper -/
structure GraphPaper where
  sideLength : ℕ

/-- The sum of lengths of all segments in the graph paper -/
def sumOfSegmentLengths (paper : GraphPaper) : ℕ :=
  2 * paper.sideLength * (paper.sideLength + 1)

/-- Theorem stating that the sum of segment lengths is divisible by 4 -/
theorem sum_of_segment_lengths_divisible_by_four (paper : GraphPaper) :
  4 ∣ sumOfSegmentLengths paper := by
  sorry

end NUMINAMATH_CALUDE_sum_of_segment_lengths_divisible_by_four_l2439_243908


namespace NUMINAMATH_CALUDE_jill_weekly_earnings_l2439_243996

/-- Calculates Jill's earnings as a waitress for a week --/
def jill_earnings (hourly_wage : ℝ) (tip_rate : ℝ) (sales_tax : ℝ) 
                  (shifts : ℕ) (hours_per_shift : ℕ) (avg_orders_per_hour : ℝ) : ℝ :=
  let total_hours := shifts * hours_per_shift
  let wage_earnings := hourly_wage * total_hours
  let total_orders := avg_orders_per_hour * total_hours
  let orders_with_tax := total_orders * (1 + sales_tax)
  let tip_earnings := orders_with_tax * tip_rate
  wage_earnings + tip_earnings

/-- Theorem stating Jill's earnings for the week --/
theorem jill_weekly_earnings : 
  jill_earnings 4 0.15 0.1 3 8 40 = 254.4 := by
  sorry

end NUMINAMATH_CALUDE_jill_weekly_earnings_l2439_243996


namespace NUMINAMATH_CALUDE_solve_equation_l2439_243969

theorem solve_equation : ∃ y : ℝ, 4 * y + 6 * y = 450 - 10 * (y - 5) ∧ y = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2439_243969


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_p_plus_s_zero_l2439_243900

/-- Given a curve y = (px + q)/(rx + s) with y = 2x as its axis of symmetry,
    where p, q, r, s are nonzero real numbers, prove that p + s = 0. -/
theorem symmetry_axis_implies_p_plus_s_zero
  (p q r s : ℝ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h_symmetry : ∀ (x y : ℝ), y = (p * x + q) / (r * x + s) → 2 * x = (p * (2 * y) + q) / (r * (2 * y) + s)) :
  p + s = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_p_plus_s_zero_l2439_243900


namespace NUMINAMATH_CALUDE_CaBr2_molecular_weight_l2439_243940

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.904

/-- The number of calcium atoms in CaBr2 -/
def num_Ca_atoms : ℕ := 1

/-- The number of bromine atoms in CaBr2 -/
def num_Br_atoms : ℕ := 2

/-- The molecular weight of CaBr2 in g/mol -/
def molecular_weight_CaBr2 : ℝ :=
  atomic_weight_Ca * num_Ca_atoms + atomic_weight_Br * num_Br_atoms

theorem CaBr2_molecular_weight :
  molecular_weight_CaBr2 = 199.888 := by
  sorry

end NUMINAMATH_CALUDE_CaBr2_molecular_weight_l2439_243940


namespace NUMINAMATH_CALUDE_candy_distribution_l2439_243901

/-- Given the number of candies for each type and the number of cousins,
    calculates the number of candies left after equal distribution. -/
def candies_left (apple orange lemon grape cousins : ℕ) : ℕ :=
  (apple + orange + lemon + grape) % cousins

theorem candy_distribution (apple orange lemon grape cousins : ℕ) 
    (h : cousins > 0) : 
  candies_left apple orange lemon grape cousins = 
  (apple + orange + lemon + grape) % cousins := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2439_243901


namespace NUMINAMATH_CALUDE_function_identity_l2439_243907

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h_continuous : Continuous f)
variable (h_inequality : ∀ (a b c : ℝ) (x : ℝ), f (a * x^2 + b * x + c) ≥ a * (f x)^2 + b * (f x) + c)

-- Theorem statement
theorem function_identity : f = id := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2439_243907


namespace NUMINAMATH_CALUDE_numberOfWaysTheorem_l2439_243939

/-- The number of ways to choose sets S_ij satisfying the given conditions -/
def numberOfWays (n : ℕ) : ℕ :=
  (Nat.factorial (2 * n)) * (2 ^ (n ^ 2))

/-- The theorem stating the number of ways to choose sets S_ij -/
theorem numberOfWaysTheorem (n : ℕ) :
  numberOfWays n = (Nat.factorial (2 * n)) * (2 ^ (n ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_numberOfWaysTheorem_l2439_243939


namespace NUMINAMATH_CALUDE_sum_of_positive_reals_l2439_243934

theorem sum_of_positive_reals (p q r s : ℝ) : 
  0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s →
  p^2 + q^2 = 2500 →
  r^2 + s^2 = 2500 →
  p * r = 1200 →
  q * s = 1200 →
  p + q + r + s = 140 := by
sorry

end NUMINAMATH_CALUDE_sum_of_positive_reals_l2439_243934


namespace NUMINAMATH_CALUDE_problem_statement_l2439_243938

theorem problem_statement (x y : ℝ) (h : (y + 1)^2 + Real.sqrt (x - 2) = 0) : y^x = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2439_243938


namespace NUMINAMATH_CALUDE_starters_count_l2439_243927

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of twins
def num_twins : ℕ := 2

-- Define the number of triplets
def num_triplets : ℕ := 3

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the function to calculate the number of ways to choose starters
def choose_starters (total : ℕ) (twins : ℕ) (triplets : ℕ) (starters : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem starters_count :
  choose_starters total_players num_twins num_triplets num_starters = 5148 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_l2439_243927


namespace NUMINAMATH_CALUDE_students_walking_home_l2439_243978

theorem students_walking_home (bus auto bike scooter : ℚ)
  (h_bus : bus = 2/5)
  (h_auto : auto = 1/5)
  (h_bike : bike = 1/10)
  (h_scooter : scooter = 1/10)
  : 1 - (bus + auto + bike + scooter) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l2439_243978


namespace NUMINAMATH_CALUDE_remaining_credit_after_call_prove_remaining_credit_l2439_243955

/-- Calculates the remaining credit on a prepaid phone card after a call. -/
theorem remaining_credit_after_call 
  (initial_value : ℝ) 
  (cost_per_minute : ℝ) 
  (call_duration : ℕ) 
  (remaining_credit : ℝ) : Prop :=
  initial_value = 30 ∧ 
  cost_per_minute = 0.16 ∧ 
  call_duration = 22 ∧ 
  remaining_credit = initial_value - (cost_per_minute * call_duration) → 
  remaining_credit = 26.48

/-- Proof of the remaining credit calculation. -/
theorem prove_remaining_credit : 
  ∃ (initial_value cost_per_minute : ℝ) (call_duration : ℕ) (remaining_credit : ℝ),
    remaining_credit_after_call initial_value cost_per_minute call_duration remaining_credit :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_credit_after_call_prove_remaining_credit_l2439_243955


namespace NUMINAMATH_CALUDE_sum_and_product_membership_l2439_243962

def P : Set ℤ := {x | ∃ k, x = 2 * k - 1}
def Q : Set ℤ := {y | ∃ n, y = 2 * n}

theorem sum_and_product_membership (x y : ℤ) (hx : x ∈ P) (hy : y ∈ Q) :
  (x + y) ∈ P ∧ (x * y) ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_membership_l2439_243962


namespace NUMINAMATH_CALUDE_inequality_property_equivalence_l2439_243989

theorem inequality_property_equivalence (t : ℝ) (ht : t > 0) :
  (∃ X : Set ℝ, Set.Infinite X ∧
    ∀ (x y z : ℝ) (a : ℝ) (d : ℝ), x ∈ X → y ∈ X → z ∈ X → d > 0 →
      max (|x - (a - d)|) (max (|y - a|) (|z - (a + d)|)) > t * d) ↔
  t < (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_property_equivalence_l2439_243989


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l2439_243914

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

theorem min_value_reciprocal_sum_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ 
    (1 / a + 1 / b) < 4 + 2 * Real.sqrt 3 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l2439_243914


namespace NUMINAMATH_CALUDE_sum_to_k_perfect_square_l2439_243910

theorem sum_to_k_perfect_square (k : ℕ) :
  (∃ n : ℕ, n < 100 ∧ k * (k + 1) / 2 = n^2) → k = 1 ∨ k = 8 ∨ k = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_k_perfect_square_l2439_243910


namespace NUMINAMATH_CALUDE_pauls_crayons_l2439_243916

/-- Paul's crayon problem -/
theorem pauls_crayons (initial given lost broken traded : ℕ) : 
  initial = 250 → 
  given = 150 → 
  lost = 512 → 
  broken = 75 → 
  traded = 35 → 
  lost - (given + broken + traded) = 252 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l2439_243916


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2439_243912

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = (b / a) * x ∨ y = -(b / a) * x}

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2 / 3 = 1

theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola_equation x y →
  (x, y) ∈ asymptote_equation 1 (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2439_243912


namespace NUMINAMATH_CALUDE_x_range_when_a_is_one_a_range_when_q_necessary_not_sufficient_l2439_243920

-- Define propositions p and q
def p (a x : ℝ) : Prop := a < x ∧ x < 3 * a

def q (x : ℝ) : Prop := 2 < x ∧ x < 3

-- Theorem 1
theorem x_range_when_a_is_one (x : ℝ) (h1 : p 1 x) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Theorem 2
theorem a_range_when_q_necessary_not_sufficient (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, q x → p a x)
  (h3 : ∃ x, p a x ∧ ¬q x) :
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_when_a_is_one_a_range_when_q_necessary_not_sufficient_l2439_243920


namespace NUMINAMATH_CALUDE_line_through_point_l2439_243974

/-- Proves that the value of k is -10 for a line passing through (-1/3, -2) --/
theorem line_through_point (k : ℝ) : 
  (2 - 3 * k * (-1/3) = 4 * (-2)) → k = -10 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2439_243974


namespace NUMINAMATH_CALUDE_root_in_interval_l2439_243964

def f (x : ℝ) : ℝ := x^3 + x + 3

theorem root_in_interval :
  ∃ x ∈ Set.Ioo (-2 : ℝ) (-1), f x = 0 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l2439_243964


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l2439_243954

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) : 
  total_caps = 12 → num_groups = 6 → caps_per_group = total_caps / num_groups → caps_per_group = 2 := by
  sorry

#check bottle_cap_distribution

end NUMINAMATH_CALUDE_bottle_cap_distribution_l2439_243954


namespace NUMINAMATH_CALUDE_initial_speed_problem_l2439_243977

theorem initial_speed_problem (v : ℝ) : 
  (0.5 * v + 1 * (2 * v) = 75) → v = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_problem_l2439_243977


namespace NUMINAMATH_CALUDE_sin_239_equals_neg_cos_31_l2439_243992

theorem sin_239_equals_neg_cos_31 (a : ℝ) (h : Real.cos (31 * π / 180) = a) :
  Real.sin (239 * π / 180) = -a := by
  sorry

end NUMINAMATH_CALUDE_sin_239_equals_neg_cos_31_l2439_243992


namespace NUMINAMATH_CALUDE_percent_of_x_l2439_243942

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25) / x * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l2439_243942


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_eight_l2439_243986

theorem ceiling_neg_sqrt_eight : ⌈-Real.sqrt 8⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_eight_l2439_243986


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l2439_243943

/-- Given a line expressed as a dot product of vectors, prove it can be rewritten in slope-intercept form -/
theorem line_equation_equivalence (x y : ℝ) : 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-5)) = 0 ↔ y = 2 * x - 11 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l2439_243943


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2439_243956

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (3 + I) / (2 - I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2439_243956


namespace NUMINAMATH_CALUDE_product_divisible_by_four_probability_l2439_243999

/-- The set of integers from 6 to 18, inclusive -/
def IntegerRange : Set ℤ := {n : ℤ | 6 ≤ n ∧ n ≤ 18}

/-- The set of integers in IntegerRange that are divisible by 4 -/
def DivisibleBy4 : Set ℤ := {n : ℤ | n ∈ IntegerRange ∧ n % 4 = 0}

/-- The set of even integers in IntegerRange -/
def EvenInRange : Set ℤ := {n : ℤ | n ∈ IntegerRange ∧ n % 2 = 0}

/-- The number of ways to choose 2 distinct integers from IntegerRange -/
def TotalChoices : ℕ := Nat.choose (Finset.card (Finset.range 13)) 2

/-- The number of ways to choose 2 distinct integers from IntegerRange 
    such that their product is divisible by 4 -/
def FavorableChoices : ℕ := 33

theorem product_divisible_by_four_probability : 
  (FavorableChoices : ℚ) / TotalChoices = 33 / 78 := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_four_probability_l2439_243999


namespace NUMINAMATH_CALUDE_sixty_has_twelve_divisors_l2439_243985

/-- The number of positive divisors of 60 -/
def num_divisors_60 : ℕ := Finset.card (Nat.divisors 60)

/-- Theorem stating that 60 has exactly 12 positive divisors -/
theorem sixty_has_twelve_divisors : num_divisors_60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sixty_has_twelve_divisors_l2439_243985


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2439_243903

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (16 / x) + (108 / y) + x * y ≥ 36 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ x y, (16 / x) + (108 / y) + x * y = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2439_243903


namespace NUMINAMATH_CALUDE_reyn_placed_25_pieces_l2439_243918

/-- Represents the puzzle distribution and placement problem --/
structure PuzzleProblem where
  total_pieces : Nat
  num_sons : Nat
  pieces_left : Nat
  rhys_multiplier : Nat
  rory_multiplier : Nat

/-- Calculates the number of pieces Reyn placed --/
def reyn_pieces (p : PuzzleProblem) : Nat :=
  let pieces_per_son := p.total_pieces / p.num_sons
  let total_placed := p.total_pieces - p.pieces_left
  total_placed / (1 + p.rhys_multiplier + p.rory_multiplier)

/-- Theorem stating that Reyn placed 25 pieces --/
theorem reyn_placed_25_pieces : 
  let p : PuzzleProblem := {
    total_pieces := 300,
    num_sons := 3,
    pieces_left := 150,
    rhys_multiplier := 2,
    rory_multiplier := 3
  }
  reyn_pieces p = 25 := by
  sorry

end NUMINAMATH_CALUDE_reyn_placed_25_pieces_l2439_243918


namespace NUMINAMATH_CALUDE_parabola_focus_l2439_243937

/-- A parabola is defined by the equation y = 4x^2 -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
def is_focus (a : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∧ p.2 = 1 / (4 * a)

/-- Theorem: The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem parabola_focus :
  ∃ (f : ℝ × ℝ), (∀ x y, parabola x y → is_focus 4 f) ∧ f = (0, 1/16) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l2439_243937


namespace NUMINAMATH_CALUDE_geometric_series_sum_is_four_thirds_l2439_243972

/-- The sum of the infinite geometric series with first term 1 and common ratio 1/4 -/
def geometric_series_sum : ℚ := 4/3

/-- The first term of the geometric series -/
def a : ℚ := 1

/-- The common ratio of the geometric series -/
def r : ℚ := 1/4

/-- Theorem stating that the sum of the infinite geometric series
    1 + (1/4) + (1/4)² + (1/4)³ + ... is equal to 4/3 -/
theorem geometric_series_sum_is_four_thirds :
  geometric_series_sum = (a / (1 - r)) := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_is_four_thirds_l2439_243972


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l2439_243923

theorem bobby_candy_problem (x : ℕ) : 
  (x - 5 - 9 = 7) → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l2439_243923
