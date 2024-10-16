import Mathlib

namespace NUMINAMATH_CALUDE_second_agency_daily_charge_correct_l2324_232401

/-- The daily charge of the first agency -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first agency -/
def first_agency_mile_charge : ℝ := 0.14

/-- The per-mile charge of the second agency -/
def second_agency_mile_charge : ℝ := 0.22

/-- The number of miles at which the agencies' costs are equal -/
def equal_cost_miles : ℝ := 25

/-- The daily charge of the second agency -/
def second_agency_daily_charge : ℝ := 18.25

theorem second_agency_daily_charge_correct :
  first_agency_daily_charge + first_agency_mile_charge * equal_cost_miles =
  second_agency_daily_charge + second_agency_mile_charge * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_second_agency_daily_charge_correct_l2324_232401


namespace NUMINAMATH_CALUDE_team_selection_ways_l2324_232480

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem team_selection_ways : 
  let group_size := 6
  let selection_size := 3
  let num_groups := 2
  (choose group_size selection_size) ^ num_groups = 400 := by sorry

end NUMINAMATH_CALUDE_team_selection_ways_l2324_232480


namespace NUMINAMATH_CALUDE_tim_running_hours_l2324_232477

/-- The number of days Tim originally ran per week -/
def original_days : ℕ := 3

/-- The number of extra days Tim added to her running schedule -/
def extra_days : ℕ := 2

/-- The number of hours Tim runs in the morning each day she runs -/
def morning_hours : ℕ := 1

/-- The number of hours Tim runs in the evening each day she runs -/
def evening_hours : ℕ := 1

/-- Theorem stating that Tim now runs 10 hours a week -/
theorem tim_running_hours : 
  (original_days + extra_days) * (morning_hours + evening_hours) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tim_running_hours_l2324_232477


namespace NUMINAMATH_CALUDE_max_value_theorem_l2324_232440

theorem max_value_theorem (x y z : ℝ) (h : x + 2 * y + z = 4) :
  ∃ (max : ℝ), max = 4 ∧ ∀ (a b c : ℝ), a + 2 * b + c = 4 → a * b + a * c + b * c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2324_232440


namespace NUMINAMATH_CALUDE_maggies_income_l2324_232430

/-- Maggie's weekly income calculation -/
theorem maggies_income
  (office_rate : ℝ)
  (tractor_rate : ℝ)
  (tractor_hours : ℝ)
  (total_income : ℝ)
  (h1 : tractor_rate = 12)
  (h2 : tractor_hours = 13)
  (h3 : total_income = 416)
  (h4 : office_rate * (2 * tractor_hours) + tractor_rate * tractor_hours = total_income) :
  office_rate = 10 := by
sorry

end NUMINAMATH_CALUDE_maggies_income_l2324_232430


namespace NUMINAMATH_CALUDE_xy_minimum_value_l2324_232450

theorem xy_minimum_value (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_geom : (1/4 * Real.log x) * (Real.log y) = (1/4)^2) : x * y ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_minimum_value_l2324_232450


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_implies_a_value_l2324_232412

theorem infinitely_many_solutions_implies_a_value 
  (a b : ℝ) 
  (h : ∀ x : ℝ, 2*a*(x-1) = (5-a)*x + 3*b) :
  a = 5/3 := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_implies_a_value_l2324_232412


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2324_232423

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  a 1 + a 2017 = 10 →
  a 1 * a 2017 = 16 →
  a 2 + a 1009 + a 2016 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2324_232423


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l2324_232459

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem stating that i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 -/
theorem sum_of_powers_of_i_is_zero :
  i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l2324_232459


namespace NUMINAMATH_CALUDE_ellipse_and_line_problem_l2324_232438

-- Define the ellipse C
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the line l
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the problem statement
theorem ellipse_and_line_problem 
  (C : Ellipse)
  (l₁ : Line)
  (l₂ : Line)
  (h₁ : l₁.slope = Real.sqrt 3)
  (h₂ : l₁.intercept = -2 * Real.sqrt 3)
  (h₃ : C.a^2 - C.b^2 = 4)
  (h₄ : (C.a^2 - C.b^2) / C.a^2 = 6 / 9)
  (h₅ : l₂.intercept = -3) :
  (C.a^2 = 6 ∧ C.b^2 = 2) ∧ 
  ((l₂.slope = Real.sqrt 3 ∧ l₂.intercept = -3) ∨ 
   (l₂.slope = -Real.sqrt 3 ∧ l₂.intercept = -3)) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_and_line_problem_l2324_232438


namespace NUMINAMATH_CALUDE_hidden_dots_sum_l2324_232411

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of all numbers on a standard die -/
def sumOfDie : ℕ := 21

/-- The total number of dice -/
def numberOfDice : ℕ := 3

/-- The visible numbers on the dice -/
def visibleNumbers : List ℕ := [6, 5, 3, 1, 4, 2, 1]

/-- The number of visible faces -/
def visibleFaces : ℕ := 7

/-- The number of hidden faces -/
def hiddenFaces : ℕ := 11

/-- Theorem: The total number of dots on the hidden faces is 41 -/
theorem hidden_dots_sum :
  (numberOfDice * sumOfDie) - (visibleNumbers.sum) = 41 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_sum_l2324_232411


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_fourth_powers_l2324_232422

theorem highest_power_of_two_dividing_difference_of_fourth_powers :
  ∃ k : ℕ, k = 7 ∧ 2^k = (Nat.gcd (17^4 - 15^4) (2^64)) :=
by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_fourth_powers_l2324_232422


namespace NUMINAMATH_CALUDE_a_seq_divisibility_l2324_232451

/-- Given a natural number a ≥ 2, define the sequence a_n recursively -/
def a_seq (a : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => a ^ (a_seq a n)

/-- The main theorem stating the divisibility property of the sequence -/
theorem a_seq_divisibility (a : ℕ) (h : a ≥ 2) (n : ℕ) :
  (a_seq a (n + 1) - a_seq a n) ∣ (a_seq a (n + 2) - a_seq a (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_a_seq_divisibility_l2324_232451


namespace NUMINAMATH_CALUDE_travel_distance_l2324_232488

/-- Proves that given a person traveling equal distances at speeds of 5 km/hr, 10 km/hr, and 15 km/hr,
    and taking a total time of 11 minutes, the total distance traveled is 1.5 km. -/
theorem travel_distance (d : ℝ) : 
  d / 5 + d / 10 + d / 15 = 11 / 60 → 3 * d = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_travel_distance_l2324_232488


namespace NUMINAMATH_CALUDE_apex_angle_of_regular_quad_pyramid_l2324_232425

-- Define a regular quadrilateral pyramid
structure RegularQuadPyramid where
  -- We don't need to specify all properties, just the relevant ones
  apex_angle : ℝ
  dihedral_angle : ℝ

-- State the theorem
theorem apex_angle_of_regular_quad_pyramid 
  (pyramid : RegularQuadPyramid)
  (h : pyramid.dihedral_angle = 2 * pyramid.apex_angle) :
  pyramid.apex_angle = Real.arccos ((Real.sqrt 5 - 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_apex_angle_of_regular_quad_pyramid_l2324_232425


namespace NUMINAMATH_CALUDE_inequality_assignment_exists_l2324_232458

/-- Represents the inequality symbols on even-positioned cards -/
def InequalitySequence := Fin 50 → Bool

/-- Represents the assignment of numbers to odd-positioned cards -/
def NumberAssignment := Fin 51 → Fin 51

/-- Checks if a number assignment satisfies the given inequality sequence -/
def is_valid_assignment (ineq : InequalitySequence) (assign : NumberAssignment) : Prop :=
  ∀ i : Fin 50, 
    (ineq i = true → assign i < assign (i + 1)) ∧
    (ineq i = false → assign i > assign (i + 1))

/-- The main theorem stating that a valid assignment always exists -/
theorem inequality_assignment_exists (ineq : InequalitySequence) :
  ∃ (assign : NumberAssignment), is_valid_assignment ineq assign ∧ Function.Bijective assign :=
sorry

end NUMINAMATH_CALUDE_inequality_assignment_exists_l2324_232458


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2324_232403

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (0 < a ∧ a < b → 1 / a > 1 / b) ∧
  ¬(1 / a > 1 / b → 0 < a ∧ a < b) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2324_232403


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2324_232453

/-- Given a quadratic function f(x) = ax^2 + bx + c where f(1) = 3 and f(2) = 12, prove that f(3) = 21 -/
theorem quadratic_function_value (a b c : ℝ) (f : ℝ → ℝ) 
  (h_quad : ∀ x, f x = a * x^2 + b * x + c)
  (h_f1 : f 1 = 3)
  (h_f2 : f 2 = 12) : 
  f 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2324_232453


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2324_232419

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 20 → x^2 - y^2 = 200 → x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2324_232419


namespace NUMINAMATH_CALUDE_first_oil_price_l2324_232462

/-- Given two oils mixed together, prove the price of the first oil. -/
theorem first_oil_price 
  (first_oil_volume : ℝ) 
  (second_oil_volume : ℝ) 
  (second_oil_price : ℝ) 
  (mixture_price : ℝ)
  (h1 : first_oil_volume = 10)
  (h2 : second_oil_volume = 5)
  (h3 : second_oil_price = 66)
  (h4 : mixture_price = 58) :
  ∃ (first_oil_price : ℝ), 
    first_oil_price = 54 ∧ 
    first_oil_price * first_oil_volume + second_oil_price * second_oil_volume = 
      mixture_price * (first_oil_volume + second_oil_volume) := by
  sorry

end NUMINAMATH_CALUDE_first_oil_price_l2324_232462


namespace NUMINAMATH_CALUDE_nina_walking_distance_l2324_232495

/-- Proves that Nina's walking distance to school is 0.4 miles, given John's distance and the difference between their distances. -/
theorem nina_walking_distance
  (john_distance : ℝ)
  (difference : ℝ)
  (h1 : john_distance = 0.7)
  (h2 : difference = 0.3)
  (h3 : john_distance = nina_distance + difference)
  : nina_distance = 0.4 :=
by
  sorry

end NUMINAMATH_CALUDE_nina_walking_distance_l2324_232495


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2324_232468

theorem initial_money_calculation (remaining_money : ℝ) (spent_percentage : ℝ) 
  (h1 : remaining_money = 840)
  (h2 : spent_percentage = 0.3)
  : (remaining_money / (1 - spent_percentage)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2324_232468


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2324_232469

theorem cubic_sum_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 0) : 
  (x^3 + y^3 + z^3) / (x*y*z) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2324_232469


namespace NUMINAMATH_CALUDE_yeast_population_growth_l2324_232472

/-- The yeast population growth problem -/
theorem yeast_population_growth
  (initial_population : ℕ)
  (growth_factor : ℕ)
  (time_increments : ℕ)
  (h1 : initial_population = 30)
  (h2 : growth_factor = 3)
  (h3 : time_increments = 3) :
  initial_population * growth_factor ^ time_increments = 810 :=
by sorry

end NUMINAMATH_CALUDE_yeast_population_growth_l2324_232472


namespace NUMINAMATH_CALUDE_solution_set_inequality_empty_solution_set_l2324_232409

-- Part 1
theorem solution_set_inequality (x : ℝ) :
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 := by sorry

-- Part 2
theorem empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*a*x + 4*a^2 + a > 0) ↔ a > 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_empty_solution_set_l2324_232409


namespace NUMINAMATH_CALUDE_odd_function_properties_l2324_232473

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an increasing function on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Define the minimum value of a function on an interval
def MinValueOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

-- Define the maximum value of a function on an interval
def MaxValueOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

-- Theorem statement
theorem odd_function_properties (f : ℝ → ℝ) :
  OddFunction f →
  IncreasingOn f 3 7 →
  MinValueOn f 3 7 1 →
  IncreasingOn f (-7) (-3) ∧ MaxValueOn f (-7) (-3) (-1) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2324_232473


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l2324_232408

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l2324_232408


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2324_232400

def is_valid_base_6_digit (n : ℕ) : Prop := n < 6
def is_valid_base_8_digit (n : ℕ) : Prop := n < 8

def value_in_base_6 (a : ℕ) : ℕ := 6 * a + a
def value_in_base_8 (b : ℕ) : ℕ := 8 * b + b

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ),
    is_valid_base_6_digit a ∧
    is_valid_base_8_digit b ∧
    value_in_base_6 a = 63 ∧
    value_in_base_8 b = 63 ∧
    (∀ (x y : ℕ),
      is_valid_base_6_digit x ∧
      is_valid_base_8_digit y ∧
      value_in_base_6 x = value_in_base_8 y →
      value_in_base_6 x ≥ 63) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2324_232400


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_is_three_tenths_l2324_232486

def total_students : ℕ := 5
def selected_students : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (Nat.choose (total_students - 2) (selected_students - 2)) /
  (Nat.choose total_students selected_students)

theorem probability_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_and_B_selected_is_three_tenths_l2324_232486


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2324_232427

theorem chocolate_distribution (total : ℕ) (michael_share : ℕ) (paige_share : ℕ) (mandy_share : ℕ) : 
  total = 60 →
  michael_share = total / 2 →
  paige_share = (total - michael_share) / 2 →
  mandy_share = total - michael_share - paige_share →
  mandy_share = 15 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2324_232427


namespace NUMINAMATH_CALUDE_systematic_sampling_milk_powder_l2324_232424

/-- Represents a systematic sampling selection -/
def SystematicSample (totalItems : ℕ) (sampleSize : ℕ) : List ℕ :=
  let interval := totalItems / sampleSize
  List.range sampleSize |>.map (fun i => (i + 1) * interval)

/-- The problem statement -/
theorem systematic_sampling_milk_powder :
  SystematicSample 50 5 = [5, 15, 25, 35, 45] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_milk_powder_l2324_232424


namespace NUMINAMATH_CALUDE_polygon_angles_l2324_232463

theorem polygon_angles (n : ℕ) (h : n > 2) :
  (180 * (n - 2) : ℝ) = 3 * 360 →
  n = 8 ∧ (180 * (n - 2) : ℝ) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l2324_232463


namespace NUMINAMATH_CALUDE_macaroon_problem_l2324_232476

theorem macaroon_problem (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (total_bags : ℕ) (remaining_weight : ℕ) :
  total_macaroons = 12 →
  weight_per_macaroon = 5 →
  total_bags = 4 →
  remaining_weight = 45 →
  total_macaroons % total_bags = 0 →
  (total_macaroons * weight_per_macaroon - remaining_weight) / (total_macaroons / total_bags * weight_per_macaroon) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_macaroon_problem_l2324_232476


namespace NUMINAMATH_CALUDE_min_value_theorem_l2324_232402

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  arithmetic_sequence a →
  (∀ k : ℕ, a k > 0) →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 2 * Real.sqrt 2 * a 1 →
  (2 : ℝ) / m + 8 / n ≥ 18 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2324_232402


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2324_232466

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2324_232466


namespace NUMINAMATH_CALUDE_cube_surface_area_for_given_volume_l2324_232406

def cube_volume : ℝ := 3375

def cube_surface_area (v : ℝ) : ℝ :=
  6 * (v ^ (1/3)) ^ 2

theorem cube_surface_area_for_given_volume :
  cube_surface_area cube_volume = 1350 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_for_given_volume_l2324_232406


namespace NUMINAMATH_CALUDE_worker_payment_l2324_232457

theorem worker_payment (total_days : ℕ) (days_not_worked : ℕ) (return_amount : ℕ) 
  (h1 : total_days = 30)
  (h2 : days_not_worked = 24)
  (h3 : return_amount = 25)
  : ∃ x : ℕ, 
    (total_days - days_not_worked) * x = days_not_worked * return_amount ∧ 
    x = 100 := by
  sorry

end NUMINAMATH_CALUDE_worker_payment_l2324_232457


namespace NUMINAMATH_CALUDE_inequality_solution_l2324_232404

theorem inequality_solution (x : ℝ) : 2 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ x ∈ Set.Ioo (35 / 13) (10 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2324_232404


namespace NUMINAMATH_CALUDE_fraction_equality_l2324_232482

theorem fraction_equality : (5 + 4 - 3) / (5 + 4 + 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2324_232482


namespace NUMINAMATH_CALUDE_trig_equation_result_l2324_232446

theorem trig_equation_result (x : Real) : 
  2 * Real.cos x - 5 * Real.sin x = 3 → 
  Real.sin x + 2 * Real.cos x = 1/2 ∨ Real.sin x + 2 * Real.cos x = 83/29 := by
sorry

end NUMINAMATH_CALUDE_trig_equation_result_l2324_232446


namespace NUMINAMATH_CALUDE_similar_triangles_solution_l2324_232421

/-- Two similar right triangles with legs 15 and 12 in the first triangle, 
    and y and 9 in the second triangle. -/
def similar_triangles (y : ℝ) : Prop :=
  15 / y = 12 / 9

theorem similar_triangles_solution :
  ∃ y : ℝ, similar_triangles y ∧ y = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_solution_l2324_232421


namespace NUMINAMATH_CALUDE_power_division_equals_one_l2324_232443

theorem power_division_equals_one (a : ℝ) (h : a ≠ 0) : a^5 / a^5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_one_l2324_232443


namespace NUMINAMATH_CALUDE_box_counting_l2324_232441

theorem box_counting (initial_boxes : ℕ) (boxes_per_operation : ℕ) (final_nonempty_boxes : ℕ) : 
  initial_boxes = 2013 → 
  boxes_per_operation = 13 → 
  final_nonempty_boxes = 2013 →
  initial_boxes + boxes_per_operation * final_nonempty_boxes = 28182 := by
  sorry

#check box_counting

end NUMINAMATH_CALUDE_box_counting_l2324_232441


namespace NUMINAMATH_CALUDE_mixture_composition_l2324_232432

/-- Represents a solution with a given percentage of carbonated water -/
structure Solution :=
  (carbonated_water_percent : ℝ)
  (h_percent : 0 ≤ carbonated_water_percent ∧ carbonated_water_percent ≤ 1)

/-- Represents a mixture of two solutions -/
structure Mixture (P Q : Solution) :=
  (p_volume : ℝ)
  (q_volume : ℝ)
  (h_positive : 0 < p_volume ∧ 0 < q_volume)
  (carbonated_water_percent : ℝ)
  (h_mixture_percent : 0 ≤ carbonated_water_percent ∧ carbonated_water_percent ≤ 1)
  (h_balance : p_volume * P.carbonated_water_percent + q_volume * Q.carbonated_water_percent = 
               (p_volume + q_volume) * carbonated_water_percent)

/-- The main theorem to prove -/
theorem mixture_composition 
  (P : Solution) 
  (Q : Solution) 
  (mix : Mixture P Q) 
  (h_P : P.carbonated_water_percent = 0.8) 
  (h_Q : Q.carbonated_water_percent = 0.55) 
  (h_mix : mix.carbonated_water_percent = 0.6) : 
  mix.p_volume / (mix.p_volume + mix.q_volume) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l2324_232432


namespace NUMINAMATH_CALUDE_scientific_notation_600000_l2324_232491

theorem scientific_notation_600000 : 600000 = 6 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_600000_l2324_232491


namespace NUMINAMATH_CALUDE_permutations_of_five_l2324_232455

theorem permutations_of_five (d : ℕ) : d = Nat.factorial 5 → d = 120 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_five_l2324_232455


namespace NUMINAMATH_CALUDE_gake_uses_fewer_boards_l2324_232431

/-- Represents the width of a character in centimeters -/
def char_width : ℕ := 9

/-- Represents the width of a board in centimeters -/
def board_width : ℕ := 5

/-- Calculates the number of boards needed for a given total width -/
def boards_needed (total_width : ℕ) : ℕ :=
  (total_width + board_width - 1) / board_width

/-- Represents Tom's message -/
def tom_message : String := "MMO"

/-- Represents Gake's message -/
def gake_message : String := "2020"

/-- Calculates the total width needed for a message -/
def message_width (msg : String) : ℕ :=
  msg.length * char_width

theorem gake_uses_fewer_boards :
  boards_needed (message_width gake_message) < boards_needed (message_width tom_message) := by
  sorry

#eval boards_needed (message_width tom_message)
#eval boards_needed (message_width gake_message)

end NUMINAMATH_CALUDE_gake_uses_fewer_boards_l2324_232431


namespace NUMINAMATH_CALUDE_percentage_calculation_l2324_232413

theorem percentage_calculation (A B : ℝ) (x : ℝ) 
  (h1 : A - B = 1670)
  (h2 : A = 2505)
  (h3 : (7.5 / 100) * A = (x / 100) * B) :
  x = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2324_232413


namespace NUMINAMATH_CALUDE_max_expression_value_l2324_232420

def expression (a b c d : ℕ) : ℕ := c * a^b - d

theorem max_expression_value :
  ∃ (a b c d : ℕ), 
    a ∈ ({0, 1, 2, 3} : Set ℕ) ∧ 
    b ∈ ({0, 1, 2, 3} : Set ℕ) ∧ 
    c ∈ ({0, 1, 2, 3} : Set ℕ) ∧ 
    d ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 9 ∧
    ∀ (w x y z : ℕ), 
      w ∈ ({0, 1, 2, 3} : Set ℕ) → 
      x ∈ ({0, 1, 2, 3} : Set ℕ) → 
      y ∈ ({0, 1, 2, 3} : Set ℕ) → 
      z ∈ ({0, 1, 2, 3} : Set ℕ) →
      w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
      expression w x y z ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_expression_value_l2324_232420


namespace NUMINAMATH_CALUDE_solution_set_m_zero_range_of_m_x_in_2_3_l2324_232445

-- Define the inequality
def inequality (x m : ℝ) : Prop := x * abs (x - m) - 2 ≥ m

-- Part 1: Solution set when m = 0
theorem solution_set_m_zero :
  {x : ℝ | inequality x 0} = {x : ℝ | x ≥ Real.sqrt 2} := by sorry

-- Part 2: Range of m when x ∈ [2, 3]
theorem range_of_m_x_in_2_3 :
  {m : ℝ | ∀ x ∈ Set.Icc 2 3, inequality x m} = 
  {m : ℝ | m ≤ 2/3 ∨ m ≥ 6} := by sorry

end NUMINAMATH_CALUDE_solution_set_m_zero_range_of_m_x_in_2_3_l2324_232445


namespace NUMINAMATH_CALUDE_area_of_large_square_l2324_232494

/-- Given three squares with side lengths a, b, and c, prove that the area of the largest square is 100 --/
theorem area_of_large_square (a b c : ℝ) 
  (h1 : a^2 = b^2 + 32)
  (h2 : 4*a = 4*c + 16) : 
  a^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_area_of_large_square_l2324_232494


namespace NUMINAMATH_CALUDE_function_properties_l2324_232492

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem function_properties (a b c : ℝ) :
  (∃ y, -3 < y ∧ y ≤ 0 ∧ f a b c (-1) = y ∧ f a b c 1 = y ∧ f a b c 2 = y) →
  a = -2 ∧ b = -1 ∧ -1 < c ∧ c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2324_232492


namespace NUMINAMATH_CALUDE_cone_from_sector_l2324_232454

/-- Given a 270° sector of a circle with radius 8, prove that the cone formed by aligning
    the straight sides of the sector has a base radius of 6 and a slant height of 8. -/
theorem cone_from_sector (r : ℝ) (angle : ℝ) (h1 : r = 8) (h2 : angle = 270) :
  let sector_arc_length := (angle / 360) * (2 * Real.pi * r)
  let cone_base_circumference := sector_arc_length
  let cone_base_radius := cone_base_circumference / (2 * Real.pi)
  let cone_slant_height := r
  (cone_base_radius = 6) ∧ (cone_slant_height = 8) :=
by sorry

end NUMINAMATH_CALUDE_cone_from_sector_l2324_232454


namespace NUMINAMATH_CALUDE_carols_mother_carrots_l2324_232474

theorem carols_mother_carrots : 
  ∀ (carol_carrots good_carrots bad_carrots total_carrots mother_carrots : ℕ),
    carol_carrots = 29 →
    good_carrots = 38 →
    bad_carrots = 7 →
    total_carrots = good_carrots + bad_carrots →
    mother_carrots = total_carrots - carol_carrots →
    mother_carrots = 16 := by
  sorry

end NUMINAMATH_CALUDE_carols_mother_carrots_l2324_232474


namespace NUMINAMATH_CALUDE_probability_of_either_test_l2324_232479

theorem probability_of_either_test (p_math p_english : ℚ) 
  (h_math : p_math = 5/8)
  (h_english : p_english = 1/4)
  (h_independent : True) -- We don't need to express independence in the theorem statement
  : 1 - (1 - p_math) * (1 - p_english) = 23/32 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_either_test_l2324_232479


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l2324_232417

theorem max_value_x_plus_2y (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 12) 
  (h2 : 3 * x + 6 * y ≤ 9) : 
  ∃ (max : ℝ), max = 3 ∧ x + 2 * y ≤ max ∧ 
  ∀ (z : ℝ), (∃ (a b : ℝ), 4 * a + 3 * b ≤ 12 ∧ 3 * a + 6 * b ≤ 9 ∧ z = a + 2 * b) → z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l2324_232417


namespace NUMINAMATH_CALUDE_count_solutions_eq_two_l2324_232414

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem count_solutions_eq_two :
  (∃ (A : Finset ℕ), (∀ n ∈ A, n + S n + S (S n) = 2050) ∧ A.card = 2 ∧
   ∀ n : ℕ, n + S n + S (S n) = 2050 → n ∈ A) := by sorry

end NUMINAMATH_CALUDE_count_solutions_eq_two_l2324_232414


namespace NUMINAMATH_CALUDE_exists_circumscribing_square_l2324_232449

/-- A type representing a bounded convex shape in a plane -/
structure BoundedConvexShape where
  -- Add necessary fields/axioms to define a bounded convex shape
  is_bounded : Bool
  is_convex : Bool

/-- A type representing a square in a plane -/
structure Square where
  -- Add necessary fields to define a square

/-- Predicate to check if a square circumscribes a bounded convex shape -/
def circumscribes (s : Square) (shape : BoundedConvexShape) : Prop :=
  sorry -- Define the circumscription condition

/-- Theorem stating that every bounded convex shape can be circumscribed by a square -/
theorem exists_circumscribing_square (shape : BoundedConvexShape) :
  shape.is_bounded ∧ shape.is_convex → ∃ s : Square, circumscribes s shape := by
  sorry


end NUMINAMATH_CALUDE_exists_circumscribing_square_l2324_232449


namespace NUMINAMATH_CALUDE_miranda_pillows_l2324_232410

/-- Represents the number of pillows Miranda can stuff -/
def stuff_pillows (goose_feathers_per_pound : ℕ) (duck_feathers_per_pound : ℕ) 
  (goose_total_feathers : ℕ) (duck_total_feathers : ℕ) (feathers_per_pillow : ℕ) : ℕ :=
  ((goose_total_feathers / goose_feathers_per_pound + 
    duck_total_feathers / duck_feathers_per_pound) / feathers_per_pillow)

/-- Theorem: Miranda can stuff 10 pillows given the conditions -/
theorem miranda_pillows : 
  stuff_pillows 300 500 3600 4000 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_miranda_pillows_l2324_232410


namespace NUMINAMATH_CALUDE_digit_product_equation_l2324_232426

def digit_product (k : ℕ) : ℕ :=
  if k = 0 then 0
  else if k < 10 then k
  else (k % 10) * digit_product (k / 10)

theorem digit_product_equation : 
  ∀ k : ℕ, k > 0 → (digit_product k = (25 * k) / 8 - 211) ↔ (k = 72 ∨ k = 88) :=
sorry

end NUMINAMATH_CALUDE_digit_product_equation_l2324_232426


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_and_negation_l2324_232478

theorem sufficient_not_necessary_and_negation :
  (∀ a : ℝ, a > 1 → (1 / a < 1)) ∧
  (∃ a : ℝ, 1 / a < 1 ∧ a ≤ 1) ∧
  (¬(∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_and_negation_l2324_232478


namespace NUMINAMATH_CALUDE_sheryll_book_purchase_l2324_232460

/-- Calculates the number of books purchased given the original price, discount, and total payment -/
def books_purchased (original_price : ℚ) (discount : ℚ) (total_payment : ℚ) : ℚ :=
  total_payment / (original_price - discount)

/-- Theorem stating that given the specific conditions, the number of books purchased is 10 -/
theorem sheryll_book_purchase :
  let original_price : ℚ := 5
  let discount : ℚ := 0.5
  let total_payment : ℚ := 45
  books_purchased original_price discount total_payment = 10 := by
  sorry

end NUMINAMATH_CALUDE_sheryll_book_purchase_l2324_232460


namespace NUMINAMATH_CALUDE_area_18_rectangles_l2324_232481

def rectangle_pairs : Set (ℕ × ℕ) :=
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)}

theorem area_18_rectangles :
  ∀ (w l : ℕ), w > 0 → l > 0 → w * l = 18 ↔ (w, l) ∈ rectangle_pairs := by
  sorry

end NUMINAMATH_CALUDE_area_18_rectangles_l2324_232481


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2324_232471

theorem complex_equation_solution (a b : ℝ) 
  (h : (a - 1 : ℂ) + 2*a*I = -4 + b*I) : b = -6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2324_232471


namespace NUMINAMATH_CALUDE_cooking_time_proof_l2324_232497

/-- The time it takes to cook a batch of waffles in minutes -/
def waffle_time : ℕ := 10

/-- The time it takes to cook 3 steaks and a batch of waffles in minutes -/
def total_time : ℕ := 28

/-- The number of steaks cooked -/
def num_steaks : ℕ := 3

/-- The time it takes to cook a single chicken-fried steak in minutes -/
def steak_time : ℕ := (total_time - waffle_time) / num_steaks

theorem cooking_time_proof : steak_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_cooking_time_proof_l2324_232497


namespace NUMINAMATH_CALUDE_third_person_teeth_removal_l2324_232484

theorem third_person_teeth_removal (total_teeth : ℕ) (total_removed : ℕ) 
  (first_person_fraction : ℚ) (second_person_fraction : ℚ) (last_person_removed : ℕ) :
  total_teeth = 32 →
  total_removed = 40 →
  first_person_fraction = 1/4 →
  second_person_fraction = 3/8 →
  last_person_removed = 4 →
  (total_removed - 
    (first_person_fraction * total_teeth + 
     second_person_fraction * total_teeth + 
     last_person_removed)) / total_teeth = 1/2 := by
  sorry

#check third_person_teeth_removal

end NUMINAMATH_CALUDE_third_person_teeth_removal_l2324_232484


namespace NUMINAMATH_CALUDE_operation_property_l2324_232461

theorem operation_property (h : ℕ → ℝ) (k : ℝ) (n : ℕ) 
  (h_def : ∀ m l : ℕ, h (m + l) = h m * h l) 
  (h_2 : h 2 = k) 
  (k_nonzero : k ≠ 0) : 
  h (2 * n) * h 2024 = k^(n + 1012) := by
  sorry

end NUMINAMATH_CALUDE_operation_property_l2324_232461


namespace NUMINAMATH_CALUDE_equidistant_function_property_l2324_232416

/-- A function that scales a complex number by a complex factor -/
def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * Complex.I) * z

/-- The theorem stating the properties and result of the problem -/
theorem equidistant_function_property (c d : ℝ) :
  (c > 0) →
  (d > 0) →
  (∀ z : ℂ, Complex.abs (g c d z - z) = Complex.abs (g c d z)) →
  Complex.abs (c + d * Complex.I) = 10 →
  d^2 = 99.75 := by sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l2324_232416


namespace NUMINAMATH_CALUDE_mad_hatter_waiting_time_l2324_232499

/-- Represents a clock with a rate different from real time -/
structure AdjustedClock where
  rate : ℚ  -- Rate of the clock compared to real time

/-- The Mad Hatter's clock -/
def madHatterClock : AdjustedClock :=
  { rate := 75 / 60 }

/-- The March Hare's clock -/
def marchHareClock : AdjustedClock :=
  { rate := 50 / 60 }

/-- Calculates the real time passed for a given clock time -/
def realTimePassed (clock : AdjustedClock) (clockTime : ℚ) : ℚ :=
  clockTime / clock.rate

/-- The agreed meeting time on their clocks -/
def meetingTime : ℚ := 5

theorem mad_hatter_waiting_time :
  realTimePassed madHatterClock meetingTime + 2 = realTimePassed marchHareClock meetingTime :=
by sorry

end NUMINAMATH_CALUDE_mad_hatter_waiting_time_l2324_232499


namespace NUMINAMATH_CALUDE_nigel_money_problem_l2324_232448

theorem nigel_money_problem (initial_amount : ℕ) (mother_gift : ℕ) (final_amount : ℕ) : 
  initial_amount = 45 →
  mother_gift = 80 →
  final_amount = 2 * initial_amount + 10 →
  initial_amount - (final_amount - mother_gift) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_nigel_money_problem_l2324_232448


namespace NUMINAMATH_CALUDE_binary_decimal_octal_equivalence_l2324_232487

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation as a list of digits -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_decimal_octal_equivalence :
  let binary := [true, true, false, true]  -- 1011 in binary (least significant bit first)
  let decimal := 11
  let octal := [1, 3]
  binary_to_decimal binary = decimal ∧
  decimal_to_octal decimal = octal :=
by sorry

end NUMINAMATH_CALUDE_binary_decimal_octal_equivalence_l2324_232487


namespace NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l2324_232464

/-- The area of a circle circumscribing an isosceles triangle -/
theorem circle_area_isosceles_triangle (a b : ℝ) (h1 : a = 4) (h2 : b = 3) :
  let r := Real.sqrt ((a^2 / 4 + b^2 / 16))
  π * r^2 = 5.6875 * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l2324_232464


namespace NUMINAMATH_CALUDE_banana_distribution_l2324_232418

theorem banana_distribution (B N : ℕ) : 
  B = 2 * N ∧ B = 4 * (N - 320) → N = 640 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l2324_232418


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l2324_232456

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≥ |a - 4|) ↔ a ∈ Set.Icc (-1 : ℝ) 9 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l2324_232456


namespace NUMINAMATH_CALUDE_data_transmission_time_l2324_232493

-- Define the number of packets
def num_packets : ℕ := 100

-- Define the number of bytes per packet
def bytes_per_packet : ℕ := 256

-- Define the transmission rate in bytes per second
def transmission_rate : ℕ := 200

-- Define the number of seconds in a minute
def seconds_per_minute : ℕ := 60

-- Theorem to prove
theorem data_transmission_time :
  (num_packets * bytes_per_packet) / transmission_rate / seconds_per_minute = 2 := by
  sorry


end NUMINAMATH_CALUDE_data_transmission_time_l2324_232493


namespace NUMINAMATH_CALUDE_plumbing_job_washers_remaining_l2324_232442

/-- Calculates the number of washers remaining after a plumbing job. -/
def washers_remaining (copper_pipe : ℕ) (pvc_pipe : ℕ) (steel_pipe : ℕ) 
  (copper_bolt_length : ℕ) (pvc_bolt_length : ℕ) (steel_bolt_length : ℕ)
  (copper_washers_per_bolt : ℕ) (pvc_washers_per_bolt : ℕ) (steel_washers_per_bolt : ℕ)
  (total_washers : ℕ) : ℕ :=
  let copper_bolts := (copper_pipe + copper_bolt_length - 1) / copper_bolt_length
  let pvc_bolts := (pvc_pipe + pvc_bolt_length - 1) / pvc_bolt_length * 2
  let steel_bolts := (steel_pipe + steel_bolt_length - 1) / steel_bolt_length
  let washers_used := copper_bolts * copper_washers_per_bolt + 
                      pvc_bolts * pvc_washers_per_bolt + 
                      steel_bolts * steel_washers_per_bolt
  total_washers - washers_used

theorem plumbing_job_washers_remaining :
  washers_remaining 40 30 20 5 10 8 2 3 4 80 = 43 := by
  sorry

end NUMINAMATH_CALUDE_plumbing_job_washers_remaining_l2324_232442


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_not_l2324_232498

theorem negation_of_existence_is_forall_not :
  (¬ ∃ x : ℝ, x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_not_l2324_232498


namespace NUMINAMATH_CALUDE_range_of_a_l2324_232415

theorem range_of_a (p q : Prop) (h_p : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (h_q : ∃ x : ℝ, x^2 - 4*x + a ≤ 0) (h_pq : p ∧ q) :
  a ∈ Set.Icc (Real.exp 1) 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2324_232415


namespace NUMINAMATH_CALUDE_granger_grocery_bill_l2324_232444

def spam_price : ℕ := 3
def peanut_butter_price : ℕ := 5
def bread_price : ℕ := 2

def spam_quantity : ℕ := 12
def peanut_butter_quantity : ℕ := 3
def bread_quantity : ℕ := 4

def total_cost : ℕ := spam_price * spam_quantity + 
                      peanut_butter_price * peanut_butter_quantity + 
                      bread_price * bread_quantity

theorem granger_grocery_bill : total_cost = 59 := by
  sorry

end NUMINAMATH_CALUDE_granger_grocery_bill_l2324_232444


namespace NUMINAMATH_CALUDE_stratified_sampling_l2324_232405

theorem stratified_sampling (total_families : ℕ) (high_income : ℕ) (middle_income : ℕ) (low_income : ℕ) 
  (high_income_sampled : ℕ) (h1 : total_families = 500) (h2 : high_income = 125) (h3 : middle_income = 280) 
  (h4 : low_income = 95) (h5 : high_income_sampled = 25) :
  (high_income_sampled * low_income) / high_income = 19 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2324_232405


namespace NUMINAMATH_CALUDE_two_intersection_points_l2324_232437

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2*y - 3*x = 3
def line2 (x y : ℝ) : Prop := x + 3*y = 3
def line3 (x y : ℝ) : Prop := 5*x - 3*y = 6

-- Define a function to check if a point lies on at least two lines
def onAtLeastTwoLines (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem two_intersection_points :
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧ 
    onAtLeastTwoLines p1.1 p1.2 ∧ 
    onAtLeastTwoLines p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), onAtLeastTwoLines p.1 p.2 → p = p1 ∨ p = p2 :=
sorry

end NUMINAMATH_CALUDE_two_intersection_points_l2324_232437


namespace NUMINAMATH_CALUDE_rachel_homework_l2324_232452

/-- Rachel's homework problem -/
theorem rachel_homework (math_pages reading_pages total_pages : ℕ) : 
  math_pages = 8 ∧ 
  math_pages = reading_pages + 3 →
  total_pages = math_pages + reading_pages ∧
  total_pages = 13 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_l2324_232452


namespace NUMINAMATH_CALUDE_apple_bag_price_l2324_232407

-- Define the given quantities
def total_harvest : ℕ := 405
def juice_amount : ℕ := 90
def restaurant_amount : ℕ := 60
def bag_size : ℕ := 5
def total_revenue : ℕ := 408

-- Define the selling price of one bag
def selling_price : ℚ := 8

-- Theorem to prove
theorem apple_bag_price :
  (total_harvest - juice_amount - restaurant_amount) / bag_size * selling_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_apple_bag_price_l2324_232407


namespace NUMINAMATH_CALUDE_sin_square_plus_sin_minus_one_range_l2324_232489

theorem sin_square_plus_sin_minus_one_range :
  ∀ x : ℝ, -5/4 ≤ Real.sin x ^ 2 + Real.sin x - 1 ∧ Real.sin x ^ 2 + Real.sin x - 1 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_square_plus_sin_minus_one_range_l2324_232489


namespace NUMINAMATH_CALUDE_octal_4652_to_decimal_l2324_232475

def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

theorem octal_4652_to_decimal :
  octal_to_decimal [2, 5, 6, 4] = 2474 := by
  sorry

end NUMINAMATH_CALUDE_octal_4652_to_decimal_l2324_232475


namespace NUMINAMATH_CALUDE_red_light_runners_estimate_l2324_232465

/-- Represents the survey results and conditions -/
structure SurveyData where
  total_students : ℕ
  yes_answers : ℕ
  id_range : Set ℕ

/-- Calculates the estimated number of people who have run a red light -/
def estimate_red_light_runners (data : SurveyData) : ℕ :=
  2 * (data.yes_answers - data.total_students / 4)

/-- Theorem stating the estimated number of red light runners -/
theorem red_light_runners_estimate (data : SurveyData) 
  (h1 : data.total_students = 800)
  (h2 : data.yes_answers = 240)
  (h3 : data.id_range = {n : ℕ | 1 ≤ n ∧ n ≤ 800}) :
  estimate_red_light_runners data = 80 := by
  sorry

end NUMINAMATH_CALUDE_red_light_runners_estimate_l2324_232465


namespace NUMINAMATH_CALUDE_max_plus_min_equals_13_l2324_232439

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem max_plus_min_equals_13 :
  ∃ (a b : ℝ), (∀ x ∈ domain, f x ≤ a) ∧
               (∀ x ∈ domain, b ≤ f x) ∧
               (∃ x₁ ∈ domain, f x₁ = a) ∧
               (∃ x₂ ∈ domain, f x₂ = b) ∧
               a + b = 13 :=
by sorry

end NUMINAMATH_CALUDE_max_plus_min_equals_13_l2324_232439


namespace NUMINAMATH_CALUDE_sugar_solution_sweetness_l2324_232490

theorem sugar_solution_sweetness (a b m : ℝ) 
  (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : a / b < (a + m) / (b + m) := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_sweetness_l2324_232490


namespace NUMINAMATH_CALUDE_ellipse_equation_and_chord_length_l2324_232434

noncomputable section

-- Define the ellipse C₁
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the left endpoint of the ellipse
def left_endpoint : ℝ × ℝ := (-Real.sqrt 6, 0)

-- Define the line l₂
def line_l2 (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - 2)

theorem ellipse_equation_and_chord_length 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ellipse a b (Prod.fst parabola_focus) (Prod.snd parabola_focus))
  (h4 : ellipse a b (Prod.fst left_endpoint) (Prod.snd left_endpoint)) :
  (∀ x y, ellipse a b x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    ellipse a b A.1 A.2 ∧ 
    ellipse a b B.1 B.2 ∧
    line_l2 A.1 A.2 ∧
    line_l2 B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 6 / 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_chord_length_l2324_232434


namespace NUMINAMATH_CALUDE_losing_candidate_percentage_approx_33_percent_l2324_232496

/-- Calculates the percentage of votes received by a losing candidate -/
def losingCandidatePercentage (totalVotes : ℕ) (lossMargin : ℕ) : ℚ :=
  let candidateVotes := (totalVotes - lossMargin) / 2
  (candidateVotes : ℚ) / totalVotes * 100

/-- Theorem stating that given the conditions, the losing candidate's vote percentage is approximately 33% -/
theorem losing_candidate_percentage_approx_33_percent 
  (totalVotes : ℕ) (lossMargin : ℕ) 
  (h1 : totalVotes = 2450) 
  (h2 : lossMargin = 833) : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |losingCandidatePercentage totalVotes lossMargin - 33| < ε :=
sorry

end NUMINAMATH_CALUDE_losing_candidate_percentage_approx_33_percent_l2324_232496


namespace NUMINAMATH_CALUDE_binary_1010_equals_decimal_10_l2324_232428

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.zip b (List.reverse (List.range b.length))).foldl
    (fun acc (digit, power) => acc + if digit then 2^power else 0) 0

theorem binary_1010_equals_decimal_10 :
  binary_to_decimal [true, false, true, false] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_equals_decimal_10_l2324_232428


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2324_232483

/-- The length of the major axis of the ellipse x²/5 + y²/2 = 1 is 2√5 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 5 + y^2 / 2 = 1}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1} ∧
    2 * a = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2324_232483


namespace NUMINAMATH_CALUDE_investment_problem_l2324_232433

/-- Proves that given the conditions of the investment problem, the invested sum is 15000 --/
theorem investment_problem (P : ℝ) 
  (h1 : P * (15 / 100) * 2 - P * (12 / 100) * 2 = 900) : 
  P = 15000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l2324_232433


namespace NUMINAMATH_CALUDE_parallel_resistors_combined_resistance_l2324_232470

theorem parallel_resistors_combined_resistance :
  let r1 : ℚ := 2
  let r2 : ℚ := 5
  let r3 : ℚ := 6
  let r : ℚ := (1 / r1 + 1 / r2 + 1 / r3)⁻¹
  r = 15 / 13 := by sorry

end NUMINAMATH_CALUDE_parallel_resistors_combined_resistance_l2324_232470


namespace NUMINAMATH_CALUDE_cell_division_result_l2324_232467

/-- Represents the cell division process over time -/
def cellDivision (initialOrganisms : ℕ) (initialCellsPerOrganism : ℕ) (divisionRatio : ℕ) (daysBetweenDivisions : ℕ) (totalDays : ℕ) : ℕ :=
  let initialCells := initialOrganisms * initialCellsPerOrganism
  let numDivisions := totalDays / daysBetweenDivisions
  initialCells * divisionRatio ^ numDivisions

/-- Theorem stating the result of the cell division process -/
theorem cell_division_result :
  cellDivision 8 4 3 3 12 = 864 := by
  sorry

end NUMINAMATH_CALUDE_cell_division_result_l2324_232467


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2324_232447

theorem decimal_to_fraction : (2.75 : ℚ) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2324_232447


namespace NUMINAMATH_CALUDE_orange_harvest_l2324_232435

/-- The number of sacks of oranges kept after a given number of harvest days -/
def sacksKept (harvestedPerDay discardedPerDay harvestDays : ℕ) : ℕ :=
  (harvestedPerDay - discardedPerDay) * harvestDays

/-- Theorem: The number of sacks of oranges kept after 51 days of harvest is 153 -/
theorem orange_harvest :
  sacksKept 74 71 51 = 153 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_l2324_232435


namespace NUMINAMATH_CALUDE_sphere_cone_equal_volume_l2324_232429

/-- Given a cone with radius 2 inches and height 6 inches, prove that a sphere
    with radius ∛6 inches has the same volume as the cone. -/
theorem sphere_cone_equal_volume :
  let cone_radius : ℝ := 2
  let cone_height : ℝ := 6
  let sphere_radius : ℝ := (6 : ℝ) ^ (1/3)
  (1/3 : ℝ) * Real.pi * cone_radius^2 * cone_height = (4/3 : ℝ) * Real.pi * sphere_radius^3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_cone_equal_volume_l2324_232429


namespace NUMINAMATH_CALUDE_initial_eggs_count_l2324_232485

/-- Given a person shares eggs among 8 friends, with each friend receiving 2 eggs,
    prove that the initial number of eggs is 16. -/
theorem initial_eggs_count (num_friends : ℕ) (eggs_per_friend : ℕ) 
  (h1 : num_friends = 8) (h2 : eggs_per_friend = 2) : 
  num_friends * eggs_per_friend = 16 := by
  sorry

#check initial_eggs_count

end NUMINAMATH_CALUDE_initial_eggs_count_l2324_232485


namespace NUMINAMATH_CALUDE_scientific_notation_of_1300000_l2324_232436

/-- Express 1,300,000 in scientific notation -/
theorem scientific_notation_of_1300000 :
  (1300000 : ℝ) = 1.3 * (10 : ℝ) ^ 6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1300000_l2324_232436
