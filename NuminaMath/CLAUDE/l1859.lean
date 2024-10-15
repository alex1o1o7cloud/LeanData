import Mathlib

namespace NUMINAMATH_CALUDE_carpet_breadth_l1859_185961

/-- The breadth of the first carpet in meters -/
def b : ℝ := 6

/-- The length of the first carpet in meters -/
def l : ℝ := 1.44 * b

/-- The length of the second carpet in meters -/
def l2 : ℝ := 1.4 * l

/-- The breadth of the second carpet in meters -/
def b2 : ℝ := 1.25 * b

/-- The cost of the second carpet in rupees -/
def cost : ℝ := 4082.4

/-- The rate of the carpet in rupees per square meter -/
def rate : ℝ := 45

theorem carpet_breadth :
  b = 6 ∧
  l = 1.44 * b ∧
  l2 = 1.4 * l ∧
  b2 = 1.25 * b ∧
  cost = rate * l2 * b2 :=
by sorry

end NUMINAMATH_CALUDE_carpet_breadth_l1859_185961


namespace NUMINAMATH_CALUDE_exists_painted_subpolygon_l1859_185913

/-- Represents a convex polygon --/
structure ConvexPolygon where
  -- Add necessary fields

/-- Represents a diagonal of a polygon --/
structure Diagonal where
  -- Add necessary fields

/-- Represents a subpolygon formed by diagonals --/
structure Subpolygon where
  -- Add necessary fields

/-- A function to check if a subpolygon is entirely painted on the outside --/
def is_entirely_painted_outside (sp : Subpolygon) : Prop :=
  sorry

/-- The main theorem --/
theorem exists_painted_subpolygon 
  (P : ConvexPolygon) 
  (sides_painted_outside : Prop) 
  (diagonals : List Diagonal)
  (no_three_intersect : Prop)
  (diagonals_painted_one_side : Prop) :
  ∃ (sp : Subpolygon), is_entirely_painted_outside sp :=
sorry

end NUMINAMATH_CALUDE_exists_painted_subpolygon_l1859_185913


namespace NUMINAMATH_CALUDE_zoo_trip_average_bus_capacity_l1859_185971

theorem zoo_trip_average_bus_capacity (total_students : ℕ) (num_buses : ℕ) 
  (car1_capacity car2_capacity car3_capacity car4_capacity : ℕ) : 
  total_students = 396 →
  num_buses = 7 →
  car1_capacity = 5 →
  car2_capacity = 4 →
  car3_capacity = 3 →
  car4_capacity = 6 →
  (total_students - (car1_capacity + car2_capacity + car3_capacity + car4_capacity)) / num_buses = 54 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_average_bus_capacity_l1859_185971


namespace NUMINAMATH_CALUDE_function_values_l1859_185909

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else x^2

theorem function_values (a : ℝ) : f (-1) = 2 * f a → a = Real.sqrt 3 ∨ a = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_values_l1859_185909


namespace NUMINAMATH_CALUDE_inequality_proof_l1859_185943

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_ineq : 2 * (a + b + c + d) ≥ a * b * c * d) : 
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1859_185943


namespace NUMINAMATH_CALUDE_total_selling_price_calculation_craig_appliance_sales_l1859_185918

/-- Calculates the total selling price of appliances given commission details --/
theorem total_selling_price_calculation 
  (fixed_commission : ℝ) 
  (variable_commission_rate : ℝ) 
  (num_appliances : ℕ) 
  (total_commission : ℝ) : ℝ :=
  let total_fixed_commission := fixed_commission * num_appliances
  let variable_commission := total_commission - total_fixed_commission
  variable_commission / variable_commission_rate

/-- Proves that the total selling price is $3620 given the problem conditions --/
theorem craig_appliance_sales : 
  total_selling_price_calculation 50 0.1 6 662 = 3620 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_calculation_craig_appliance_sales_l1859_185918


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l1859_185964

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

-- State the theorem
theorem derivative_f_at_2 : 
  deriv f 2 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l1859_185964


namespace NUMINAMATH_CALUDE_fibonacci_sum_identity_l1859_185908

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_sum_identity (n m : ℕ) (h1 : n ≥ 1) (h2 : m ≥ 0) :
  fib (n + m) = fib (n - 1) * fib m + fib n * fib (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_identity_l1859_185908


namespace NUMINAMATH_CALUDE_unitedNations75thAnniversary_l1859_185980

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (advanceDay d n)

-- Define the founding day of the United Nations
def unitedNationsFoundingDay : DayOfWeek := DayOfWeek.Wednesday

-- Define the number of days to advance for the 75th anniversary
def daysToAdvance : Nat := 93

-- Theorem statement
theorem unitedNations75thAnniversary :
  advanceDay unitedNationsFoundingDay daysToAdvance = DayOfWeek.Friday :=
sorry

end NUMINAMATH_CALUDE_unitedNations75thAnniversary_l1859_185980


namespace NUMINAMATH_CALUDE_sum_of_numbers_ge_1_1_l1859_185974

theorem sum_of_numbers_ge_1_1 : 
  let numbers : List ℝ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let filtered_numbers := numbers.filter (λ x => x ≥ 1.1)
  filtered_numbers.sum = 3.9 := by sorry

end NUMINAMATH_CALUDE_sum_of_numbers_ge_1_1_l1859_185974


namespace NUMINAMATH_CALUDE_roses_picked_later_l1859_185981

/-- Calculates the number of roses picked later by a florist -/
theorem roses_picked_later (initial : ℕ) (sold : ℕ) (final : ℕ) : 
  initial ≥ sold → final > initial - sold → final - (initial - sold) = 21 := by
  sorry

end NUMINAMATH_CALUDE_roses_picked_later_l1859_185981


namespace NUMINAMATH_CALUDE_sine_inequality_l1859_185941

theorem sine_inequality (x : ℝ) : 
  (∃ k : ℤ, (π / 6 + k * π < x ∧ x < π / 2 + k * π) ∨ 
            (5 * π / 6 + k * π < x ∧ x < 3 * π / 2 + k * π)) ↔ 
  (Real.sin x)^2 + (Real.sin (2 * x))^2 > (Real.sin (3 * x))^2 := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l1859_185941


namespace NUMINAMATH_CALUDE_expected_pairs_for_given_deck_l1859_185933

/-- Represents a deck of cards with numbered pairs and Joker pairs -/
structure Deck :=
  (num_pairs : ℕ)
  (joker_pairs : ℕ)

/-- Calculates the expected number of complete pairs when drawing until a Joker pair is found -/
def expected_complete_pairs (d : Deck) : ℚ :=
  (d.num_pairs : ℚ) / 3 + 1

theorem expected_pairs_for_given_deck :
  let d : Deck := ⟨7, 2⟩
  expected_complete_pairs d = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_expected_pairs_for_given_deck_l1859_185933


namespace NUMINAMATH_CALUDE_cart_distance_l1859_185929

/-- The distance traveled by a cart with three wheels of different circumferences -/
theorem cart_distance (front_circ rear_circ third_circ : ℕ)
  (h1 : front_circ = 30)
  (h2 : rear_circ = 32)
  (h3 : third_circ = 34)
  (rev_rear : ℕ)
  (h4 : front_circ * (rev_rear + 5) = rear_circ * rev_rear)
  (h5 : third_circ * (rev_rear - 8) = rear_circ * rev_rear) :
  rear_circ * rev_rear = 2400 :=
sorry

end NUMINAMATH_CALUDE_cart_distance_l1859_185929


namespace NUMINAMATH_CALUDE_coals_per_bag_prove_coals_per_bag_l1859_185979

-- Define the constants from the problem
def coals_per_set : ℕ := 15
def minutes_per_set : ℕ := 20
def total_minutes : ℕ := 240
def num_bags : ℕ := 3

-- Define the theorem
theorem coals_per_bag : ℕ :=
  let sets_burned := total_minutes / minutes_per_set
  let total_coals_burned := sets_burned * coals_per_set
  total_coals_burned / num_bags

-- State the theorem to be proved
theorem prove_coals_per_bag : coals_per_bag = 60 := by
  sorry

end NUMINAMATH_CALUDE_coals_per_bag_prove_coals_per_bag_l1859_185979


namespace NUMINAMATH_CALUDE_bicycle_distance_l1859_185923

theorem bicycle_distance (front_circ rear_circ : ℚ) (extra_revs : ℕ) : 
  front_circ = 4/3 →
  rear_circ = 3/2 →
  extra_revs = 25 →
  (front_circ * (extra_revs + (rear_circ * extra_revs) / (front_circ - rear_circ))) = 300 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_distance_l1859_185923


namespace NUMINAMATH_CALUDE_first_year_exceeding_target_l1859_185973

-- Define the initial investment and growth rate
def initial_investment : ℝ := 1.3
def growth_rate : ℝ := 0.12

-- Define the target investment
def target_investment : ℝ := 2.0

-- Define the function to calculate the investment for a given year
def investment (year : ℕ) : ℝ := initial_investment * (1 + growth_rate) ^ (year - 2015)

-- Theorem statement
theorem first_year_exceeding_target : 
  (∀ y : ℕ, y < 2019 → investment y ≤ target_investment) ∧ 
  investment 2019 > target_investment :=
sorry

end NUMINAMATH_CALUDE_first_year_exceeding_target_l1859_185973


namespace NUMINAMATH_CALUDE_store_pricing_l1859_185950

theorem store_pricing (shirts_total : ℝ) (sweaters_total : ℝ) (jeans_total : ℝ)
  (shirts_count : ℕ) (sweaters_count : ℕ) (jeans_count : ℕ)
  (shirt_discount : ℝ) (sweater_discount : ℝ) (jeans_discount : ℝ)
  (h1 : shirts_total = 360)
  (h2 : sweaters_total = 900)
  (h3 : jeans_total = 1200)
  (h4 : shirts_count = 20)
  (h5 : sweaters_count = 45)
  (h6 : jeans_count = 30)
  (h7 : shirt_discount = 2)
  (h8 : sweater_discount = 4)
  (h9 : jeans_discount = 3) :
  let shirt_avg := (shirts_total / shirts_count) - shirt_discount
  let sweater_avg := (sweaters_total / sweaters_count) - sweater_discount
  let jeans_avg := (jeans_total / jeans_count) - jeans_discount
  shirt_avg = sweater_avg ∧ jeans_avg - sweater_avg = 21 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_l1859_185950


namespace NUMINAMATH_CALUDE_vector_subtraction_l1859_185975

/-- Given two vectors AB and AC in 2D space, prove that BC is their difference. -/
theorem vector_subtraction (AB AC : Fin 2 → ℝ) (h1 : AB = ![3, 6]) (h2 : AC = ![1, 2]) :
  AB - AC = ![-2, -4] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1859_185975


namespace NUMINAMATH_CALUDE_cos_3x_minus_pi_3_equals_sin_3x_plus_pi_18_l1859_185928

theorem cos_3x_minus_pi_3_equals_sin_3x_plus_pi_18 (x : ℝ) :
  Real.cos (3 * x - π / 3) = Real.sin (3 * (x + π / 18)) := by
  sorry

end NUMINAMATH_CALUDE_cos_3x_minus_pi_3_equals_sin_3x_plus_pi_18_l1859_185928


namespace NUMINAMATH_CALUDE_three_fifths_of_difference_l1859_185932

theorem three_fifths_of_difference : (3 : ℚ) / 5 * ((7 * 9) - (4 * 3)) = 153 / 5 := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_of_difference_l1859_185932


namespace NUMINAMATH_CALUDE_curve_crosses_at_point_l1859_185938

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^2 - 2

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 9*t + 5

/-- The curve crosses itself if there exist two distinct real numbers that yield the same point -/
def curve_crosses_itself : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ x a = x b ∧ y a = y b

/-- The point where the curve crosses itself -/
def crossing_point : ℝ × ℝ := (7, 5)

/-- Theorem stating that the curve crosses itself at the specified point -/
theorem curve_crosses_at_point :
  curve_crosses_itself ∧ ∃ t : ℝ, (x t, y t) = crossing_point :=
sorry

end NUMINAMATH_CALUDE_curve_crosses_at_point_l1859_185938


namespace NUMINAMATH_CALUDE_calculate_brads_speed_l1859_185988

/-- Given two people walking towards each other, calculate the speed of one person given the other's speed and distance traveled. -/
theorem calculate_brads_speed (maxwell_speed brad_speed : ℝ) (total_distance maxwell_distance : ℝ) : 
  maxwell_speed = 2 →
  total_distance = 36 →
  maxwell_distance = 12 →
  2 * maxwell_distance = total_distance →
  brad_speed = 4 := by sorry

end NUMINAMATH_CALUDE_calculate_brads_speed_l1859_185988


namespace NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l1859_185960

theorem opposite_signs_abs_sum_less_abs_diff (x y : ℝ) (h : x * y < 0) :
  |x + y| < |x - y| := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l1859_185960


namespace NUMINAMATH_CALUDE_hypotenuse_ratio_from_area_ratio_l1859_185924

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  area : ℝ

-- Theorem statement
theorem hypotenuse_ratio_from_area_ratio
  (t1 t2 : IsoscelesRightTriangle)
  (h_area : t2.area = 2 * t1.area) :
  t2.hypotenuse = Real.sqrt 2 * t1.hypotenuse :=
sorry

end NUMINAMATH_CALUDE_hypotenuse_ratio_from_area_ratio_l1859_185924


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l1859_185962

/-- The perimeter of a square garden with area 900 square meters is 120 meters and 12000 centimeters. -/
theorem square_garden_perimeter :
  ∀ (side : ℝ), 
  side^2 = 900 →
  (4 * side = 120) ∧ (4 * side * 100 = 12000) := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l1859_185962


namespace NUMINAMATH_CALUDE_function_properties_l1859_185959

noncomputable def f (x φ : ℝ) : ℝ := Real.sin x * Real.cos φ + Real.cos x * Real.sin φ

theorem function_properties (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  -- The smallest positive period of f is 2π
  (∃ (T : ℝ), T > 0 ∧ T = 2 * π ∧ ∀ (x : ℝ), f x φ = f (x + T) φ) ∧
  -- If the graph of y = f(2x + π/4) is symmetric about x = π/6, then φ = 11π/12
  (∀ (x : ℝ), f (2 * (π/6 - x) + π/4) φ = f (2 * (π/6 + x) + π/4) φ → φ = 11 * π / 12) ∧
  -- If f(α - 2π/3) = √2/4, then sin 2α = -3/4
  (∀ (α : ℝ), f (α - 2 * π / 3) φ = Real.sqrt 2 / 4 → Real.sin (2 * α) = -3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1859_185959


namespace NUMINAMATH_CALUDE_investment_problem_l1859_185993

/-- Proves that given the conditions of the investment problem, the invested sum is 4200 --/
theorem investment_problem (P : ℝ) 
  (h1 : P * (15 / 100) * 2 - P * (10 / 100) * 2 = 840) : 
  P = 4200 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1859_185993


namespace NUMINAMATH_CALUDE_tan_five_pi_over_four_l1859_185934

theorem tan_five_pi_over_four : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_over_four_l1859_185934


namespace NUMINAMATH_CALUDE_fuel_food_ratio_l1859_185902

theorem fuel_food_ratio 
  (fuel_cost : ℝ) 
  (distance_per_tank : ℝ) 
  (total_distance : ℝ) 
  (total_spent : ℝ) 
  (h1 : fuel_cost = 45)
  (h2 : distance_per_tank = 500)
  (h3 : total_distance = 2000)
  (h4 : total_spent = 288) :
  (total_spent - (total_distance / distance_per_tank * fuel_cost)) / 
  (total_distance / distance_per_tank * fuel_cost) = 3 / 5 := by
sorry


end NUMINAMATH_CALUDE_fuel_food_ratio_l1859_185902


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l1859_185951

theorem fixed_point_parabola :
  ∃ (a b : ℝ), ∀ (k : ℝ), 9 * a^2 + k * a - 5 * k + 3 = b ∧ a = 5 ∧ b = 228 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l1859_185951


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1859_185989

/-- The sum of terms in an arithmetic sequence with first term 2, common difference 12, and last term 182 is 1472 -/
theorem arithmetic_sequence_sum : 
  let a₁ : ℕ := 2  -- First term
  let d : ℕ := 12  -- Common difference
  let aₙ : ℕ := 182  -- Last term
  let n : ℕ := (aₙ - a₁) / d + 1  -- Number of terms
  (n : ℝ) * (a₁ + aₙ) / 2 = 1472 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1859_185989


namespace NUMINAMATH_CALUDE_triangle_side_constraint_l1859_185955

theorem triangle_side_constraint (a : ℝ) : 
  (6 > 0 ∧ 1 - 3*a > 0 ∧ 10 > 0) ∧  -- positive side lengths
  (6 + (1 - 3*a) > 10 ∧ 6 + 10 > 1 - 3*a ∧ 10 + (1 - 3*a) > 6) →  -- triangle inequality
  -5 < a ∧ a < -1 :=
by sorry


end NUMINAMATH_CALUDE_triangle_side_constraint_l1859_185955


namespace NUMINAMATH_CALUDE_classroom_attendance_l1859_185982

theorem classroom_attendance (students_in_restroom : ℕ) 
  (total_students : ℕ) (rows : ℕ) (desks_per_row : ℕ) 
  (occupancy_rate : ℚ) :
  students_in_restroom = 2 →
  total_students = 23 →
  rows = 4 →
  desks_per_row = 6 →
  occupancy_rate = 2/3 →
  ∃ (m : ℕ), m * students_in_restroom - 1 = 
    total_students - (↑(rows * desks_per_row) * occupancy_rate).floor - students_in_restroom ∧
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_classroom_attendance_l1859_185982


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l1859_185984

theorem tan_double_angle_special_case (θ : ℝ) :
  3 * Real.cos (π / 2 - θ) + Real.cos (π + θ) = 0 →
  Real.tan (2 * θ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l1859_185984


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1859_185995

theorem fractional_equation_solution (x : ℝ) :
  x ≠ 2 → x ≠ 0 → (1 / (x - 2) = 3 / x) ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1859_185995


namespace NUMINAMATH_CALUDE_polynomial_mapping_l1859_185930

def polynomial_equation (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) : Prop :=
  ∀ x, x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄ = (x+1)^4 + b₁*(x+1)^3 + b₂*(x+1)^2 + b₃*(x+1) + b₄

def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let b₁ := 0
  let b₂ := -3
  let b₃ := 4
  let b₄ := -1
  (b₁, b₂, b₃, b₄)

theorem polynomial_mapping :
  polynomial_equation 4 3 2 1 0 (-3) 4 (-1) → f 4 3 2 1 = (0, -3, 4, -1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_mapping_l1859_185930


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_sum_less_than_16_sixteen_is_smallest_l1859_185978

theorem smallest_whole_number_above_sum : ℕ → Prop :=
  fun n => (2 + 1/2 + 3 + 1/3 + 4 + 1/4 + 5 + 1/5 : ℚ) < n ∧
           ∀ m : ℕ, (2 + 1/2 + 3 + 1/3 + 4 + 1/4 + 5 + 1/5 : ℚ) < m → n ≤ m

theorem sum_less_than_16 :
  (2 + 1/2 + 3 + 1/3 + 4 + 1/4 + 5 + 1/5 : ℚ) < 16 :=
sorry

theorem sixteen_is_smallest : smallest_whole_number_above_sum 16 :=
sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_sum_less_than_16_sixteen_is_smallest_l1859_185978


namespace NUMINAMATH_CALUDE_cricket_match_analysis_l1859_185983

-- Define the cricket match parameters
def total_overs : ℕ := 50
def initial_overs : ℕ := 10
def remaining_overs : ℕ := total_overs - initial_overs
def initial_run_rate : ℚ := 32/10
def initial_wickets : ℕ := 2
def target_score : ℕ := 320
def min_additional_wickets : ℕ := 5

-- Define the theorem
theorem cricket_match_analysis :
  let initial_score := initial_run_rate * initial_overs
  let remaining_score := target_score - initial_score
  let required_run_rate := remaining_score / remaining_overs
  let total_wickets_needed := initial_wickets + min_additional_wickets
  (required_run_rate = 72/10) ∧ (total_wickets_needed = 7) := by
  sorry

end NUMINAMATH_CALUDE_cricket_match_analysis_l1859_185983


namespace NUMINAMATH_CALUDE_goats_gifted_count_l1859_185996

/-- Represents the number of goats gifted by Jeremy to Fred -/
def goats_gifted (initial_horses initial_sheep initial_chickens : ℕ) 
  (male_animals : ℕ) : ℕ :=
  let initial_total := initial_horses + initial_sheep + initial_chickens
  let after_brian_sale := initial_total - initial_total / 2
  let final_total := male_animals * 2
  final_total - after_brian_sale

/-- Theorem stating the number of goats gifted by Jeremy -/
theorem goats_gifted_count : 
  goats_gifted 100 29 9 53 = 37 := by
  sorry

#eval goats_gifted 100 29 9 53

end NUMINAMATH_CALUDE_goats_gifted_count_l1859_185996


namespace NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_36_l1859_185912

/-- The sum of consecutive integers starting from a given integer -/
def sumConsecutiveIntegers (start : ℤ) (count : ℕ) : ℤ :=
  (count : ℤ) * (2 * start + (count : ℤ) - 1) / 2

/-- The property that the sum of a sequence of consecutive integers is 36 -/
def hasSumThirtySix (start : ℤ) (count : ℕ) : Prop :=
  sumConsecutiveIntegers start count = 36

/-- The theorem stating that 72 is the greatest number of consecutive integers whose sum is 36 -/
theorem greatest_consecutive_integers_sum_36 :
  (∃ start : ℤ, hasSumThirtySix start 72) ∧
  (∀ n : ℕ, n > 72 → ∀ start : ℤ, ¬hasSumThirtySix start n) :=
sorry

end NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_36_l1859_185912


namespace NUMINAMATH_CALUDE_cloud_9_diving_refund_l1859_185911

/-- Cloud 9 Diving Company Cancellation Refund Problem -/
theorem cloud_9_diving_refund (individual_bookings group_bookings total_after_cancellations : ℕ) 
  (h1 : individual_bookings = 12000)
  (h2 : group_bookings = 16000)
  (h3 : total_after_cancellations = 26400) :
  individual_bookings + group_bookings - total_after_cancellations = 1600 := by
  sorry

end NUMINAMATH_CALUDE_cloud_9_diving_refund_l1859_185911


namespace NUMINAMATH_CALUDE_num_multicolor_ducks_l1859_185954

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolored duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def num_white_ducks : ℕ := 3

/-- The number of black ducks -/
def num_black_ducks : ℕ := 7

/-- The total number of fish in the lake -/
def total_fish : ℕ := 157

/-- The theorem stating the number of multicolored ducks -/
theorem num_multicolor_ducks : ℕ := by
  sorry

#check num_multicolor_ducks

end NUMINAMATH_CALUDE_num_multicolor_ducks_l1859_185954


namespace NUMINAMATH_CALUDE_prob_two_sunny_days_value_l1859_185915

/-- The probability of exactly 2 sunny days out of 5 days, where each day has a 75% chance of rain -/
def prob_two_sunny_days : ℚ :=
  (Nat.choose 5 2 : ℚ) * (1/4)^2 * (3/4)^3

/-- The main theorem stating that the probability is equal to 135/512 -/
theorem prob_two_sunny_days_value : prob_two_sunny_days = 135/512 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_sunny_days_value_l1859_185915


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l1859_185956

def num_cows : ℕ := 45
def num_bags : ℕ := 90
def num_days : ℕ := 60

theorem one_cow_one_bag_days : 
  (num_days * num_cows) / num_bags = 30 := by
  sorry

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l1859_185956


namespace NUMINAMATH_CALUDE_m_minus_n_squared_l1859_185945

theorem m_minus_n_squared (m n : ℝ) (h1 : m + n = 6) (h2 : m^2 + n^2 = 26) : 
  (m - n)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_m_minus_n_squared_l1859_185945


namespace NUMINAMATH_CALUDE_yoongi_position_l1859_185922

/-- Calculates the number of students behind a runner after passing others. -/
def students_behind (total : ℕ) (initial_position : ℕ) (passed : ℕ) : ℕ :=
  total - (initial_position - passed)

/-- Theorem stating the number of students behind Yoongi after passing others. -/
theorem yoongi_position (total : ℕ) (initial_position : ℕ) (passed : ℕ) 
  (h_total : total = 9)
  (h_initial : initial_position = 7)
  (h_passed : passed = 4) :
  students_behind total initial_position passed = 6 := by
sorry

end NUMINAMATH_CALUDE_yoongi_position_l1859_185922


namespace NUMINAMATH_CALUDE_principal_exists_l1859_185968

/-- The principal amount that satisfies the given conditions -/
def find_principal : ℝ → Prop := fun P =>
  let first_year_rate : ℝ := 0.10
  let second_year_rate : ℝ := 0.12
  let semi_annual_rate1 : ℝ := first_year_rate / 2
  let semi_annual_rate2 : ℝ := second_year_rate / 2
  let compound_factor : ℝ := (1 + semi_annual_rate1)^2 * (1 + semi_annual_rate2)^2
  let simple_interest_factor : ℝ := first_year_rate + second_year_rate
  P * (compound_factor - 1 - simple_interest_factor) = 15

/-- Theorem stating the existence of a principal amount satisfying the given conditions -/
theorem principal_exists : ∃ P : ℝ, find_principal P := by
  sorry

end NUMINAMATH_CALUDE_principal_exists_l1859_185968


namespace NUMINAMATH_CALUDE_same_rate_different_time_l1859_185927

/-- Given that a person drives 150 miles in 3 hours, 
    prove that another person driving at the same rate for 4 hours will cover 200 miles. -/
theorem same_rate_different_time (distance₁ : ℝ) (time₁ : ℝ) (time₂ : ℝ) 
  (h₁ : distance₁ = 150) 
  (h₂ : time₁ = 3) 
  (h₃ : time₂ = 4) : 
  (distance₁ / time₁) * time₂ = 200 := by
  sorry

end NUMINAMATH_CALUDE_same_rate_different_time_l1859_185927


namespace NUMINAMATH_CALUDE_divide_by_four_twice_l1859_185904

theorem divide_by_four_twice (x : ℝ) : x = 166.08 → (x / 4) / 4 = 10.38 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_four_twice_l1859_185904


namespace NUMINAMATH_CALUDE_journey_theorem_l1859_185946

/-- Represents the journey to Koschei's kingdom -/
structure Journey where
  total_distance : ℝ
  first_day_distance : ℝ
  second_day_distance : ℝ
  third_day_distance : ℝ
  fourth_day_distance : ℝ

/-- The conditions of Leshy's journey -/
def leshy_journey (j : Journey) : Prop :=
  j.first_day_distance = j.total_distance / 3 ∧
  j.second_day_distance = j.first_day_distance / 2 ∧
  j.third_day_distance = j.first_day_distance ∧
  j.fourth_day_distance = 100 ∧
  j.total_distance = j.first_day_distance + j.second_day_distance + j.third_day_distance + j.fourth_day_distance

theorem journey_theorem (j : Journey) (h : leshy_journey j) :
  j.total_distance = 600 ∧ j.fourth_day_distance = 100 := by
  sorry

#check journey_theorem

end NUMINAMATH_CALUDE_journey_theorem_l1859_185946


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1859_185987

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2*x, -3)
  are_parallel a b → x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1859_185987


namespace NUMINAMATH_CALUDE_log_inequality_l1859_185958

theorem log_inequality (a b : ℝ) (ha : a = Real.log 0.3 / Real.log 0.2) (hb : b = Real.log 0.3 / Real.log 2) :
  a * b < a + b ∧ a + b < 0 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1859_185958


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1859_185969

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 3) : 
  3 * π * r^2 = 2 * π * r^2 + π * r^2 := by
sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l1859_185969


namespace NUMINAMATH_CALUDE_zhang_bing_age_problem_l1859_185940

def current_year : ℕ := 2023  -- Assuming current year is 2023

def birth_year : ℕ := 1953

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem zhang_bing_age_problem :
  ∃! x : ℕ, 
    birth_year < x ∧ 
    x ≤ current_year ∧
    (x - birth_year) % 9 = 0 ∧
    x - birth_year = sum_of_digits x ∧
    x - birth_year = 18 := by
  sorry

end NUMINAMATH_CALUDE_zhang_bing_age_problem_l1859_185940


namespace NUMINAMATH_CALUDE_book_distribution_l1859_185914

theorem book_distribution (n : ℕ) (b : ℕ) : 
  (3 * n + 6 = b) →                     -- Condition 1
  (5 * n - 5 ≤ b) →                     -- Condition 2 (lower bound)
  (b < 5 * n - 2) →                     -- Condition 2 (upper bound)
  (n = 5 ∧ b = 21) :=                   -- Conclusion
by sorry

end NUMINAMATH_CALUDE_book_distribution_l1859_185914


namespace NUMINAMATH_CALUDE_smallest_k_for_same_color_square_l1859_185999

/-- 
Given a positive integer n, this theorem states that 2n^2 - n + 1 is the smallest positive 
integer k such that any coloring of a 2n × k table with n colors contains two rows and 
two columns intersecting in four squares of the same color.
-/
theorem smallest_k_for_same_color_square (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k = 2*n^2 - n + 1 ∧ 
  (∀ (table : Fin (2*n) → Fin k → Fin n), 
    ∃ (r1 r2 : Fin (2*n)) (c1 c2 : Fin k),
      r1 ≠ r2 ∧ c1 ≠ c2 ∧
      table r1 c1 = table r1 c2 ∧
      table r1 c1 = table r2 c1 ∧
      table r1 c1 = table r2 c2) ∧
  (∀ k' : ℕ, k' < k → 
    ∃ (table : Fin (2*n) → Fin k' → Fin n), 
      ∀ (r1 r2 : Fin (2*n)) (c1 c2 : Fin k'),
        r1 = r2 ∨ c1 = c2 ∨
        table r1 c1 ≠ table r1 c2 ∨
        table r1 c1 ≠ table r2 c1 ∨
        table r1 c1 ≠ table r2 c2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_same_color_square_l1859_185999


namespace NUMINAMATH_CALUDE_soft_drink_pack_size_l1859_185903

/-- The number of cans in a pack of soft drinks -/
def num_cans : ℕ := 11

/-- The cost of a pack of soft drinks in dollars -/
def pack_cost : ℚ := 299/100

/-- The cost of an individual can in dollars -/
def can_cost : ℚ := 1/4

/-- Theorem stating that the number of cans in a pack is 11 -/
theorem soft_drink_pack_size :
  num_cans = ⌊pack_cost / can_cost⌋ := by sorry

end NUMINAMATH_CALUDE_soft_drink_pack_size_l1859_185903


namespace NUMINAMATH_CALUDE_point_on_765_degree_angle_l1859_185977

/-- Given that a point (4, m) lies on the terminal side of an angle of 765°, prove that m = 4 -/
theorem point_on_765_degree_angle (m : ℝ) : 
  (∃ (θ : ℝ), θ = 765 * Real.pi / 180 ∧ Real.tan θ = m / 4) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_765_degree_angle_l1859_185977


namespace NUMINAMATH_CALUDE_four_digit_diff_divisible_iff_middle_digits_same_l1859_185998

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9
  h4 : d ≥ 0 ∧ d ≤ 9

/-- Calculates the value of a four-digit number -/
def fourDigitValue (n : FourDigitNumber) : ℕ :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Calculates the value of the reversed four-digit number -/
def reversedValue (n : FourDigitNumber) : ℕ :=
  1000 * n.d + 100 * n.c + 10 * n.b + n.a

/-- Theorem: For a four-digit number, the difference between the number and its reverse
    is divisible by 37 if and only if the two middle digits are the same -/
theorem four_digit_diff_divisible_iff_middle_digits_same (n : FourDigitNumber) :
  (fourDigitValue n - reversedValue n) % 37 = 0 ↔ n.b = n.c := by
  sorry

end NUMINAMATH_CALUDE_four_digit_diff_divisible_iff_middle_digits_same_l1859_185998


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_k_is_1_A_intersect_B_nonempty_iff_k_geq_neg_1_l1859_185965

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def B (k : ℝ) : Set ℝ := {x : ℝ | x - k ≤ 0}

-- Define the complement of B in the universal set U (which is ℝ in this case)
def C_U_B (k : ℝ) : Set ℝ := {x : ℝ | x - k > 0}

theorem intersection_A_complement_B_when_k_is_1 :
  A ∩ C_U_B 1 = {x : ℝ | 1 < x ∧ x < 3} := by sorry

theorem A_intersect_B_nonempty_iff_k_geq_neg_1 :
  ∀ k : ℝ, (A ∩ B k).Nonempty ↔ k ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_k_is_1_A_intersect_B_nonempty_iff_k_geq_neg_1_l1859_185965


namespace NUMINAMATH_CALUDE_katies_first_stopover_l1859_185948

/-- Calculates the distance to the first stopover given the total distance,
    distance to the second stopover, and additional distance to the final destination -/
def distance_to_first_stopover (total_distance : ℕ) (second_stopover : ℕ) (additional_distance : ℕ) : ℕ :=
  second_stopover - (total_distance - second_stopover - additional_distance)

/-- Proves that given the specific distances in Katie's trip,
    the distance to the first stopover is 104 miles -/
theorem katies_first_stopover :
  distance_to_first_stopover 436 236 68 = 104 := by
  sorry

#eval distance_to_first_stopover 436 236 68

end NUMINAMATH_CALUDE_katies_first_stopover_l1859_185948


namespace NUMINAMATH_CALUDE_odd_function_condition_l1859_185963

def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -(f a b x)) ↔ a = 0 ∧ b = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_condition_l1859_185963


namespace NUMINAMATH_CALUDE_granger_age_difference_l1859_185947

theorem granger_age_difference : 
  let granger_age : ℕ := 42
  let son_age : ℕ := 16
  granger_age - 2 * son_age = 10 := by sorry

end NUMINAMATH_CALUDE_granger_age_difference_l1859_185947


namespace NUMINAMATH_CALUDE_first_grade_boys_count_l1859_185957

theorem first_grade_boys_count (num_classrooms : ℕ) (num_girls : ℕ) (students_per_classroom : ℕ) :
  num_classrooms = 4 →
  num_girls = 44 →
  students_per_classroom = 25 →
  (∀ classroom, classroom ≤ num_classrooms →
    (num_girls / num_classrooms = students_per_classroom / 2)) →
  num_girls = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_first_grade_boys_count_l1859_185957


namespace NUMINAMATH_CALUDE_f_of_one_f_of_a_f_of_f_of_a_l1859_185907

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 3

-- Theorem statements
theorem f_of_one : f 1 = 5 := by sorry

theorem f_of_a (a : ℝ) : f a = 2 * a + 3 := by sorry

theorem f_of_f_of_a (a : ℝ) : f (f a) = 4 * a + 9 := by sorry

end NUMINAMATH_CALUDE_f_of_one_f_of_a_f_of_f_of_a_l1859_185907


namespace NUMINAMATH_CALUDE_original_fraction_l1859_185966

theorem original_fraction (N D : ℚ) : 
  (N * (1 + 30/100)) / (D * (1 - 15/100)) = 25/21 →
  N / D = 425/546 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l1859_185966


namespace NUMINAMATH_CALUDE_crossword_puzzle_subset_l1859_185901

def is_three_identical_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ ∃ d, n = d * 100 + d * 10 + d

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def has_three_middle_threes (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ ∃ a b, n = a * 10000 + 3 * 1000 + 3 * 100 + 3 * 10 + b

theorem crossword_puzzle_subset :
  ∀ x y z : ℕ,
  is_three_identical_digits x →
  y = x^2 →
  digit_sum z = 18 →
  has_three_middle_threes z →
  x = 111 ∧ y = 12321 ∧ z = 33333 :=
by sorry

end NUMINAMATH_CALUDE_crossword_puzzle_subset_l1859_185901


namespace NUMINAMATH_CALUDE_solve_system_l1859_185997

theorem solve_system (c d : ℝ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 6 + d = 9 + c) : 
  5 - c = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1859_185997


namespace NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l1859_185905

/-- Represents a repeating decimal -/
def RepeatingDecimal (whole : ℕ) (repeating : List ℕ) : ℚ :=
  sorry

/-- The sum of three specific repeating decimals -/
theorem sum_of_specific_repeating_decimals :
  RepeatingDecimal 0 [1] + RepeatingDecimal 0 [1, 2] + RepeatingDecimal 0 [1, 2, 3] =
  RepeatingDecimal 0 [3, 5, 5, 4, 4, 6] :=
sorry

end NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l1859_185905


namespace NUMINAMATH_CALUDE_expected_remaining_bullets_value_l1859_185917

/-- The probability of hitting the target with each shot -/
def hit_probability : ℝ := 0.6

/-- The total number of available bullets -/
def total_bullets : ℕ := 4

/-- The expected number of remaining bullets after the first hit -/
def expected_remaining_bullets : ℝ :=
  3 * hit_probability +
  2 * hit_probability * (1 - hit_probability) +
  1 * hit_probability * (1 - hit_probability)^2 +
  0 * (1 - hit_probability)^3

theorem expected_remaining_bullets_value :
  expected_remaining_bullets = 2.376 := by sorry

end NUMINAMATH_CALUDE_expected_remaining_bullets_value_l1859_185917


namespace NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l1859_185994

-- Define the angle α
def α : Real := sorry

-- Define the point P₀
def P₀ : ℝ × ℝ := (-3, -4)

-- Theorem statement
theorem cos_pi_half_plus_alpha (h : (Real.cos α * (-3) = Real.sin α * (-4))) : 
  Real.cos (π / 2 + α) = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l1859_185994


namespace NUMINAMATH_CALUDE_max_length_sum_l1859_185920

def length (k : ℕ) : ℕ := sorry

def has_even_power_prime_factor (n : ℕ) : Prop := sorry

def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem max_length_sum (x y : ℕ) 
  (hx : x > 1) 
  (hy : y > 1) 
  (hsum : x + 3 * y < 1000) 
  (hx_even : has_even_power_prime_factor x) 
  (hy_even : has_even_power_prime_factor y) 
  (hp : smallest_prime_factor x + smallest_prime_factor y ≡ 0 [MOD 3]) :
  ∀ (a b : ℕ), a > 1 → b > 1 → a + 3 * b < 1000 → 
    has_even_power_prime_factor a → has_even_power_prime_factor b → 
    smallest_prime_factor a + smallest_prime_factor b ≡ 0 [MOD 3] →
    length x + length y ≥ length a + length b :=
sorry

end NUMINAMATH_CALUDE_max_length_sum_l1859_185920


namespace NUMINAMATH_CALUDE_cos_seven_pi_six_plus_x_l1859_185944

theorem cos_seven_pi_six_plus_x (x : Real) (h : Real.sin (2 * Real.pi / 3 + x) = 3 / 5) :
  Real.cos (7 * Real.pi / 6 + x) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_six_plus_x_l1859_185944


namespace NUMINAMATH_CALUDE_third_to_second_night_ratio_l1859_185925

/-- Represents Billy's sleep pattern over four nights -/
structure SleepPattern where
  first_night : ℝ
  second_night : ℝ
  third_night : ℝ
  fourth_night : ℝ

/-- Calculates the total sleep over four nights -/
def total_sleep (sp : SleepPattern) : ℝ :=
  sp.first_night + sp.second_night + sp.third_night + sp.fourth_night

/-- Theorem stating the ratio of third to second night's sleep -/
theorem third_to_second_night_ratio 
  (sp : SleepPattern)
  (h1 : sp.first_night = 6)
  (h2 : sp.second_night = sp.first_night + 2)
  (h3 : sp.fourth_night = 3 * sp.third_night)
  (h4 : total_sleep sp = 30) :
  sp.third_night / sp.second_night = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_third_to_second_night_ratio_l1859_185925


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l1859_185916

/-- The minimum sum of dimensions for a rectangular box with volume 1645 and positive integer dimensions -/
theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 1645 → l + w + h ≥ 129 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l1859_185916


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1859_185991

theorem sphere_surface_area (r : ℝ) (h : r = 4) : 
  4 * π * r^2 = 64 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1859_185991


namespace NUMINAMATH_CALUDE_egg_roll_ratio_l1859_185935

-- Define the number of egg rolls each person ate
def matthew_egg_rolls : ℕ := 6
def alvin_egg_rolls : ℕ := 4

-- Define Patrick's egg rolls based on the condition
def patrick_egg_rolls : ℕ := matthew_egg_rolls / 3

-- Theorem to prove the ratio
theorem egg_roll_ratio :
  patrick_egg_rolls / alvin_egg_rolls = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_egg_roll_ratio_l1859_185935


namespace NUMINAMATH_CALUDE_gear_revolution_theorem_l1859_185942

/-- The number of revolutions per minute for gear p -/
def p_rpm : ℝ := 10

/-- The duration in minutes -/
def duration : ℝ := 0.5

/-- The difference in revolutions between gear q and gear p after the given duration -/
def revolution_difference : ℝ := 15

/-- The number of revolutions per minute for gear q -/
def q_rpm : ℝ := 40

theorem gear_revolution_theorem :
  q_rpm = 2 * (p_rpm * duration + revolution_difference) := by
  sorry

end NUMINAMATH_CALUDE_gear_revolution_theorem_l1859_185942


namespace NUMINAMATH_CALUDE_can_measure_four_liters_l1859_185921

/-- Represents the state of water in the buckets -/
structure BucketState :=
  (small : ℕ)  -- Amount of water in the 3-liter bucket
  (large : ℕ)  -- Amount of water in the 5-liter bucket

/-- Represents the possible operations on the buckets -/
inductive BucketOperation
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | PourSmallToLarge
  | PourLargeToSmall

/-- Applies a single operation to a bucket state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  match op with
  | BucketOperation.FillSmall => { small := 3, large := state.large }
  | BucketOperation.FillLarge => { small := state.small, large := 5 }
  | BucketOperation.EmptySmall => { small := 0, large := state.large }
  | BucketOperation.EmptyLarge => { small := state.small, large := 0 }
  | BucketOperation.PourSmallToLarge =>
      let amount := min state.small (5 - state.large)
      { small := state.small - amount, large := state.large + amount }
  | BucketOperation.PourLargeToSmall =>
      let amount := min state.large (3 - state.small)
      { small := state.small + amount, large := state.large - amount }

/-- Theorem: It is possible to measure exactly 4 liters using buckets of 3 and 5 liters -/
theorem can_measure_four_liters : ∃ (ops : List BucketOperation), 
  let final_state := ops.foldl applyOperation { small := 0, large := 0 }
  final_state.small + final_state.large = 4 := by
  sorry

end NUMINAMATH_CALUDE_can_measure_four_liters_l1859_185921


namespace NUMINAMATH_CALUDE_magician_decks_l1859_185976

-- Define the problem parameters
def price_per_deck : ℕ := 2
def total_earnings : ℕ := 4
def decks_left : ℕ := 3

-- Define the theorem
theorem magician_decks : 
  ∃ (initial_decks : ℕ), 
    initial_decks * price_per_deck - total_earnings = decks_left * price_per_deck ∧ 
    initial_decks = 5 := by
  sorry

end NUMINAMATH_CALUDE_magician_decks_l1859_185976


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1859_185970

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1859_185970


namespace NUMINAMATH_CALUDE_simplify_expression_exponent_calculation_l1859_185900

-- Part 1
theorem simplify_expression (x : ℝ) : 
  (-2*x)^3 * x^2 + (3*x^4)^2 / x^3 = x^5 := by sorry

-- Part 2
theorem exponent_calculation (a m n : ℝ) 
  (hm : a^m = 2) (hn : a^n = 3) : a^(m+2*n) = 18 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_exponent_calculation_l1859_185900


namespace NUMINAMATH_CALUDE_figure_with_perimeter_91_has_11_tiles_l1859_185919

/-- Represents a figure in the sequence --/
structure Figure where
  tiles : ℕ
  perimeter : ℕ

/-- The side length of each equilateral triangle tile in cm --/
def tileSideLength : ℕ := 7

/-- The first figure in the sequence --/
def firstFigure : Figure :=
  { tiles := 1
  , perimeter := 3 * tileSideLength }

/-- Generates the next figure in the sequence --/
def nextFigure (f : Figure) : Figure :=
  { tiles := f.tiles + 1
  , perimeter := f.perimeter + tileSideLength }

/-- Theorem: The figure with perimeter 91 cm consists of 11 tiles --/
theorem figure_with_perimeter_91_has_11_tiles :
  ∃ (n : ℕ), (n.iterate nextFigure firstFigure).perimeter = 91 ∧
             (n.iterate nextFigure firstFigure).tiles = 11 := by
  sorry

end NUMINAMATH_CALUDE_figure_with_perimeter_91_has_11_tiles_l1859_185919


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l1859_185985

theorem geometric_series_first_term 
  (a r : ℝ) 
  (h1 : a / (1 - r) = 30) 
  (h2 : a^2 / (1 - r^2) = 150) : 
  a = 60 / 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l1859_185985


namespace NUMINAMATH_CALUDE_polynomial_equation_l1859_185972

theorem polynomial_equation (a : ℝ) (A : ℝ → ℝ) :
  (∀ x, A x * (x + 1) = x^2 - 1) → A = fun x ↦ x - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_l1859_185972


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1859_185986

theorem perfect_square_condition (n : ℕ+) (p : ℕ) :
  (Nat.Prime p) → (∃ k : ℕ, p^2 + 7^n.val = k^2) ↔ (n = 1 ∧ p = 3) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1859_185986


namespace NUMINAMATH_CALUDE_f_increasing_on_negative_reals_l1859_185906

-- Define the function f(x) = -|x|
def f (x : ℝ) : ℝ := -|x|

-- State the theorem
theorem f_increasing_on_negative_reals :
  ∀ (x₁ x₂ : ℝ), x₁ < 0 → x₂ < 0 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_negative_reals_l1859_185906


namespace NUMINAMATH_CALUDE_inequality_proof_l1859_185936

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1859_185936


namespace NUMINAMATH_CALUDE_vovochka_candy_theorem_l1859_185992

/-- Represents the candy distribution problem --/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- The maximum number of candies that can be kept while satisfying the distribution condition --/
def max_kept_candies (cd : CandyDistribution) : ℕ :=
  cd.total_candies - (cd.num_classmates * (cd.min_group_candies / cd.min_group_size))

/-- Theorem stating the maximum number of candies that can be kept in the given scenario --/
theorem vovochka_candy_theorem (cd : CandyDistribution) 
  (h1 : cd.total_candies = 200)
  (h2 : cd.num_classmates = 25)
  (h3 : cd.min_group_size = 16)
  (h4 : cd.min_group_candies = 100) :
  max_kept_candies cd = 37 := by
  sorry

#eval max_kept_candies { total_candies := 200, num_classmates := 25, min_group_size := 16, min_group_candies := 100 }

end NUMINAMATH_CALUDE_vovochka_candy_theorem_l1859_185992


namespace NUMINAMATH_CALUDE_sector_inscribed_circle_area_ratio_l1859_185939

/-- 
Given a sector with a central angle of 120°, 
the ratio of the area of the sector to the area of its inscribed circle is (7 + 4√3) / 9.
-/
theorem sector_inscribed_circle_area_ratio :
  ∀ R r : ℝ,
  R > 0 → r > 0 →
  r / (R - r) = Real.sqrt 3 / 2 →
  (1/3 * π * R^2) / (π * r^2) = (7 + 4 * Real.sqrt 3) / 9 :=
by sorry

end NUMINAMATH_CALUDE_sector_inscribed_circle_area_ratio_l1859_185939


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l1859_185952

/-- Given a triangle ADE with point C on AD and point B on AC, prove that FC = 10 -/
theorem triangle_similarity_theorem 
  (DC : ℝ) (CB : ℝ) (AD : ℝ) (AB : ℝ) (ED : ℝ) (FC : ℝ) :
  DC = 9 →
  CB = 7 →
  AB = (1/3) * AD →
  ED = (2/3) * AD →
  FC = 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l1859_185952


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1859_185926

theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = π / 5 ∧  -- Given condition
  a * Real.cos B - b * Real.cos A = c →  -- Given equation
  B = 3 * π / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l1859_185926


namespace NUMINAMATH_CALUDE_count_complementary_sets_l1859_185937

/-- Represents a card with four attributes -/
structure Card :=
  (shape : Fin 3)
  (color : Fin 3)
  (shade : Fin 3)
  (size : Fin 3)

/-- The deck of all possible cards -/
def deck : Finset Card := sorry

/-- A set of three cards -/
def ThreeCardSet := Finset Card

/-- Predicate for a complementary set -/
def is_complementary (s : ThreeCardSet) : Prop := sorry

/-- The set of all complementary three-card sets -/
def complementary_sets : Finset ThreeCardSet := sorry

theorem count_complementary_sets :
  Finset.card complementary_sets = 6483 := by sorry

end NUMINAMATH_CALUDE_count_complementary_sets_l1859_185937


namespace NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l1859_185967

/-- The points of intersection of the given lines form a hyperbola -/
theorem intersection_points_form_hyperbola :
  ∀ (s x y : ℝ), 
    (2*s*x - 3*y - 5*s = 0) → 
    (2*x - 3*s*y + 4 = 0) → 
    ∃ (a b : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l1859_185967


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l1859_185953

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  equal_roots : ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r)
  derivative : ∀ x, HasDerivAt f (2 * x + 2) x

/-- The main theorem: if f satisfies the given conditions, then f(x) = x^2 + 2x + 1 -/
theorem quadratic_function_unique (qf : QuadraticFunction) :
  ∀ x, qf.f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l1859_185953


namespace NUMINAMATH_CALUDE_f_properties_l1859_185931

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (1/3) * x^3 + ((1-a)/2) * x^2 - a^2 * Real.log x + a^2 * Real.log a

theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x > 0, f 1 x ≥ 1/3 ∧ f 1 1 = 1/3) ∧
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a > 3 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1859_185931


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1859_185910

theorem sin_330_degrees : 
  Real.sin (330 * π / 180) = -(1/2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1859_185910


namespace NUMINAMATH_CALUDE_sequence_properties_l1859_185949

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

def T (n : ℕ) : ℝ := sorry

theorem sequence_properties (n : ℕ) :
  n > 0 →
  (S n = 2 * sequence_a n - 2) →
  (sequence_a n = 2^n ∧ T n = 2^(n+2) - 4 - 2*n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1859_185949


namespace NUMINAMATH_CALUDE_zeros_imply_b_and_c_b_in_interval_l1859_185990

-- Define the quadratic function f(x)
def f (b c x : ℝ) : ℝ := x^2 + 2*b*x + c

-- Part 1: Prove that if -1 and 1 are zeros of f(x), then b = 0 and c = -1
theorem zeros_imply_b_and_c (b c : ℝ) :
  f b c (-1) = 0 ∧ f b c 1 = 0 → b = 0 ∧ c = -1 := by sorry

-- Part 2: Prove that given the conditions, b is in the interval (1/5, 5/7)
theorem b_in_interval (b c : ℝ) :
  f b c 1 = 0 ∧ 
  (∃ x₁ x₂, -3 < x₁ ∧ x₁ < -2 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
    f b c x₁ + x₁ + b = 0 ∧ f b c x₂ + x₂ + b = 0) →
  1/5 < b ∧ b < 5/7 := by sorry

end NUMINAMATH_CALUDE_zeros_imply_b_and_c_b_in_interval_l1859_185990
