import Mathlib

namespace NUMINAMATH_CALUDE_sum_odd_plus_even_l4001_400174

def sum_odd_integers (n : ℕ) : ℕ :=
  (n + 1) * n

def sum_even_integers (n : ℕ) : ℕ :=
  n * (n + 1)

def m : ℕ := sum_odd_integers 56

def t : ℕ := sum_even_integers 25

theorem sum_odd_plus_even : m + t = 3786 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_plus_even_l4001_400174


namespace NUMINAMATH_CALUDE_restaurant_group_adults_l4001_400146

/-- Calculates the number of adults in a restaurant group given the total bill, 
    number of children, and cost per meal. -/
theorem restaurant_group_adults 
  (total_bill : ℕ) 
  (num_children : ℕ) 
  (cost_per_meal : ℕ) : 
  total_bill = 56 → 
  num_children = 5 → 
  cost_per_meal = 8 → 
  (total_bill - num_children * cost_per_meal) / cost_per_meal = 2 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_adults_l4001_400146


namespace NUMINAMATH_CALUDE_marbles_redistribution_l4001_400183

/-- The number of marbles Tyrone initially had -/
def tyrone_initial : ℕ := 120

/-- The number of marbles Eric initially had -/
def eric_initial : ℕ := 18

/-- The ratio of Tyrone's marbles to Eric's after redistribution -/
def final_ratio : ℕ := 3

/-- The number of marbles Tyrone gave to Eric -/
def marbles_given : ℚ := 16.5

theorem marbles_redistribution :
  let tyrone_final := tyrone_initial - marbles_given
  let eric_final := eric_initial + marbles_given
  tyrone_final = final_ratio * eric_final := by sorry

end NUMINAMATH_CALUDE_marbles_redistribution_l4001_400183


namespace NUMINAMATH_CALUDE_limestone_amount_l4001_400179

/-- Represents the composition of a cement compound -/
structure CementCompound where
  limestone : ℝ
  shale : ℝ
  total_weight : ℝ
  limestone_cost : ℝ
  shale_cost : ℝ
  compound_cost : ℝ

/-- Theorem stating the correct amount of limestone in the compound -/
theorem limestone_amount (c : CementCompound) 
  (h1 : c.total_weight = 100)
  (h2 : c.limestone_cost = 3)
  (h3 : c.shale_cost = 5)
  (h4 : c.compound_cost = 4.25)
  (h5 : c.limestone + c.shale = c.total_weight)
  (h6 : c.limestone * c.limestone_cost + c.shale * c.shale_cost = c.total_weight * c.compound_cost) :
  c.limestone = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_limestone_amount_l4001_400179


namespace NUMINAMATH_CALUDE_polygon_with_40_degree_exterior_angles_has_9_sides_l4001_400161

/-- A polygon with exterior angles of 40° has 9 sides -/
theorem polygon_with_40_degree_exterior_angles_has_9_sides :
  ∀ (n : ℕ), 
  (n > 2) →
  (360 / n = 40) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_polygon_with_40_degree_exterior_angles_has_9_sides_l4001_400161


namespace NUMINAMATH_CALUDE_twelve_ways_to_choose_l4001_400151

/-- The number of ways to choose one female student from a group of 4
    and one male student from a group of 3 -/
def waysToChoose (female_count male_count : ℕ) : ℕ :=
  female_count * male_count

/-- Theorem stating that there are 12 ways to choose one female student
    from a group of 4 and one male student from a group of 3 -/
theorem twelve_ways_to_choose :
  waysToChoose 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_ways_to_choose_l4001_400151


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l4001_400168

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear

/-- The proportion of angular speeds for a system of four meshed gears -/
def angular_speed_proportion (g : GearSystem) : Prop :=
  ∃ (k : ℝ), k > 0 ∧
    g.A.speed = k * g.B.teeth * g.C.teeth * g.D.teeth ∧
    g.B.speed = k * g.A.teeth * g.C.teeth * g.D.teeth ∧
    g.C.speed = k * g.A.teeth * g.B.teeth * g.D.teeth ∧
    g.D.speed = k * g.A.teeth * g.B.teeth * g.C.teeth

theorem gear_speed_proportion (g : GearSystem) :
  angular_speed_proportion g → True :=
by
  sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l4001_400168


namespace NUMINAMATH_CALUDE_p_neither_necessary_nor_sufficient_l4001_400121

-- Define p and q as propositions depending on x and y
def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x : ℝ) : Prop := x ≠ 0  -- Assuming 'x' means x is non-zero

-- Define the theorem
theorem p_neither_necessary_nor_sufficient :
  ∃ (x y : ℝ), y ≠ -1 ∧
  (q x ∧ ¬(p x y)) ∧  -- p is not necessary
  (p x y ∧ ¬(q x))    -- p is not sufficient
  := by sorry

end NUMINAMATH_CALUDE_p_neither_necessary_nor_sufficient_l4001_400121


namespace NUMINAMATH_CALUDE_woman_lawyer_probability_l4001_400126

/-- Represents a study group with given proportions of women and women lawyers -/
structure StudyGroup where
  totalMembers : ℕ
  womenPercentage : ℚ
  womenLawyerPercentage : ℚ

/-- Calculates the probability of selecting a woman lawyer at random from the study group -/
def probWomanLawyer (group : StudyGroup) : ℚ :=
  group.womenPercentage * group.womenLawyerPercentage

theorem woman_lawyer_probability (group : StudyGroup) 
  (h1 : group.womenPercentage = 9/10)
  (h2 : group.womenLawyerPercentage = 6/10) :
  probWomanLawyer group = 54/100 := by
  sorry

#eval probWomanLawyer { totalMembers := 100, womenPercentage := 9/10, womenLawyerPercentage := 6/10 }

end NUMINAMATH_CALUDE_woman_lawyer_probability_l4001_400126


namespace NUMINAMATH_CALUDE_rental_hours_proof_l4001_400187

/-- Represents a bike rental service with a base cost and hourly rate. -/
structure BikeRental where
  baseCost : ℕ
  hourlyRate : ℕ

/-- Calculates the total cost for a given number of hours. -/
def totalCost (rental : BikeRental) (hours : ℕ) : ℕ :=
  rental.baseCost + rental.hourlyRate * hours

/-- Proves that for the given bike rental conditions and total cost, the number of hours rented is 9. -/
theorem rental_hours_proof (rental : BikeRental) 
    (h1 : rental.baseCost = 17)
    (h2 : rental.hourlyRate = 7)
    (h3 : totalCost rental 9 = 80) : 
  ∃ (hours : ℕ), totalCost rental hours = 80 ∧ hours = 9 := by
  sorry

#check rental_hours_proof

end NUMINAMATH_CALUDE_rental_hours_proof_l4001_400187


namespace NUMINAMATH_CALUDE_minimum_guests_l4001_400123

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (h1 : total_food = 411) (h2 : max_per_guest = 2.5) :
  ⌈total_food / max_per_guest⌉ = 165 := by
  sorry

end NUMINAMATH_CALUDE_minimum_guests_l4001_400123


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_y_coordinates_correct_tangent_points_are_correct_l4001_400190

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_points_parallel_to_line (x : ℝ) :
  (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
sorry

-- Theorem to verify the y-coordinates
theorem y_coordinates_correct :
  f 1 = 0 ∧ f (-1) = -4 :=
sorry

-- Main theorem combining both conditions
theorem tangent_points_are_correct :
  ∀ x y : ℝ, (f' x = 4 ∧ f x = y) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) :=
sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_y_coordinates_correct_tangent_points_are_correct_l4001_400190


namespace NUMINAMATH_CALUDE_john_lewis_meeting_point_l4001_400104

/-- Represents the journey between two cities --/
structure Journey where
  distance : ℝ
  johnSpeed : ℝ
  lewisOutboundSpeed : ℝ
  lewisReturnSpeed : ℝ
  johnBreakFrequency : ℝ
  johnBreakDuration : ℝ
  lewisBreakFrequency : ℝ
  lewisBreakDuration : ℝ

/-- Calculates the meeting point of John and Lewis --/
def meetingPoint (j : Journey) : ℝ :=
  sorry

/-- Theorem stating the meeting point of John and Lewis --/
theorem john_lewis_meeting_point :
  let j : Journey := {
    distance := 240,
    johnSpeed := 40,
    lewisOutboundSpeed := 60,
    lewisReturnSpeed := 50,
    johnBreakFrequency := 2,
    johnBreakDuration := 0.25,
    lewisBreakFrequency := 2.5,
    lewisBreakDuration := 1/3
  }
  ∃ (ε : ℝ), ε > 0 ∧ |meetingPoint j - 23.33| < ε :=
sorry

end NUMINAMATH_CALUDE_john_lewis_meeting_point_l4001_400104


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l4001_400115

theorem x_squared_plus_y_squared (x y : ℝ) :
  |x - 1/2| + (2*y + 1)^2 = 0 → x^2 + y^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l4001_400115


namespace NUMINAMATH_CALUDE_fraction_meaningful_l4001_400138

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 5)) ↔ x ≠ 5 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l4001_400138


namespace NUMINAMATH_CALUDE_vector_operation_result_l4001_400176

/-- Proves that the given vector operation results in (4, -7) -/
theorem vector_operation_result : 
  4 • !![3, -9] - 3 • !![2, -7] + 2 • !![-1, 4] = !![4, -7] := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l4001_400176


namespace NUMINAMATH_CALUDE_department_store_sales_multiple_l4001_400118

theorem department_store_sales_multiple (M : ℝ) :
  (∀ (A : ℝ), A > 0 →
    M * A = 0.15384615384615385 * (11 * A + M * A)) →
  M = 2 := by
sorry

end NUMINAMATH_CALUDE_department_store_sales_multiple_l4001_400118


namespace NUMINAMATH_CALUDE_cos_22_5_squared_minus_sin_22_5_squared_l4001_400101

theorem cos_22_5_squared_minus_sin_22_5_squared : 
  Real.cos (22.5 * π / 180) ^ 2 - Real.sin (22.5 * π / 180) ^ 2 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_22_5_squared_minus_sin_22_5_squared_l4001_400101


namespace NUMINAMATH_CALUDE_charity_book_donation_l4001_400177

theorem charity_book_donation (initial_books : ℕ) (books_per_donation : ℕ) 
  (borrowed_books : ℕ) (final_books : ℕ) : 
  initial_books = 300 →
  books_per_donation = 5 →
  borrowed_books = 140 →
  final_books = 210 →
  (final_books + borrowed_books - initial_books) / books_per_donation = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_book_donation_l4001_400177


namespace NUMINAMATH_CALUDE_non_zero_digits_count_l4001_400143

def expression : ℚ := 180 / (2^4 * 5^6 * 3^2)

def count_non_zero_decimal_digits (q : ℚ) : ℕ :=
  sorry

theorem non_zero_digits_count : count_non_zero_decimal_digits expression = 1 := by
  sorry

end NUMINAMATH_CALUDE_non_zero_digits_count_l4001_400143


namespace NUMINAMATH_CALUDE_composite_function_solution_l4001_400178

theorem composite_function_solution (h k : ℝ → ℝ) (b : ℝ) :
  (∀ x, h x = x / 3 + 2) →
  (∀ x, k x = 5 - 2 * x) →
  h (k b) = 4 →
  b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_composite_function_solution_l4001_400178


namespace NUMINAMATH_CALUDE_garden_trees_l4001_400169

/-- The number of trees in a garden with given specifications -/
def num_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  yard_length / tree_distance + 1

/-- Theorem: The number of trees in a 500-metre garden with 20-metre spacing is 26 -/
theorem garden_trees : num_trees 500 20 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_trees_l4001_400169


namespace NUMINAMATH_CALUDE_ellipse_equation_l4001_400171

theorem ellipse_equation (a b : ℝ) (M : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    A.1 = 0 ∧ B.1 = 0 ∧
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = (2 * Real.sqrt 6 / 3)^2 ∧
    (M.1 - B.1)^2 + (M.2 - B.2)^2 = (2 * Real.sqrt 6 / 3)^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 6 / 3)^2) →
  a^2 = 6 ∧ b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4001_400171


namespace NUMINAMATH_CALUDE_toys_after_game_purchase_l4001_400186

theorem toys_after_game_purchase (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : 
  initial_amount = 57 → game_cost = 27 → toy_cost = 6 → 
  (initial_amount - game_cost) / toy_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_toys_after_game_purchase_l4001_400186


namespace NUMINAMATH_CALUDE_solution_pairs_l4001_400148

theorem solution_pairs (x y a n m : ℕ) (h1 : x + y = a^n) (h2 : x^2 + y^2 = a^m) :
  ∃ k : ℕ, x = 2^k ∧ y = 2^k := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l4001_400148


namespace NUMINAMATH_CALUDE_change_is_three_l4001_400198

/-- Calculates the change received after a restaurant visit -/
def calculate_change (lee_amount : ℕ) (friend_amount : ℕ) (wings_cost : ℕ) (salad_cost : ℕ) (soda_cost : ℕ) (soda_quantity : ℕ) (tax : ℕ) : ℕ :=
  let total_amount := lee_amount + friend_amount
  let food_cost := wings_cost + salad_cost + soda_cost * soda_quantity
  let total_cost := food_cost + tax
  total_amount - total_cost

/-- Proves that the change received is $3 given the specific conditions -/
theorem change_is_three :
  calculate_change 10 8 6 4 1 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_change_is_three_l4001_400198


namespace NUMINAMATH_CALUDE_solution_count_l4001_400194

/-- The number of different integer solutions (x, y) for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := sorry

theorem solution_count :
  (num_solutions 1 = 4) →
  (num_solutions 2 = 8) →
  (num_solutions 20 = 80) := by sorry

end NUMINAMATH_CALUDE_solution_count_l4001_400194


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l4001_400122

theorem popped_kernel_probability (white yellow blue : ℝ)
  (white_pop yellow_pop blue_pop : ℝ) :
  white = 1/2 →
  yellow = 1/4 →
  blue = 1/4 →
  white_pop = 1/3 →
  yellow_pop = 3/4 →
  blue_pop = 2/3 →
  (white * white_pop) / (white * white_pop + yellow * yellow_pop + blue * blue_pop) = 2/11 := by
  sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l4001_400122


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_max_negative_l4001_400133

def f (x : ℝ) := |x - 1| + |x + 1|

theorem f_inequality_solution (x : ℝ) :
  f x ≤ 4 ↔ x ∈ Set.Icc (-2) 2 :=
sorry

theorem f_max_negative (b : ℝ) (hb : b ≠ 0) :
  (∀ x, f x ≥ (|2*b + 1| + |1 - b|) / |b|) →
  (∃ x, x < 0 ∧ f x ≥ (|2*b + 1| + |1 - b|) / |b| ∧
    ∀ y, y < 0 → f y ≥ (|2*b + 1| + |1 - b|) / |b| → y ≤ x) →
  x = -1.5 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_max_negative_l4001_400133


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4001_400159

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2015_2013 : a 2015 = a 2013 + 6) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4001_400159


namespace NUMINAMATH_CALUDE_ratio_problem_l4001_400152

theorem ratio_problem (x y : ℝ) (h : (3*x - 2*y) / (2*x + y) = 5/4) : y / x = 2/13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l4001_400152


namespace NUMINAMATH_CALUDE_classroom_chairs_l4001_400132

theorem classroom_chairs (blue_chairs : ℕ) (green_chairs : ℕ) (white_chairs : ℕ) :
  blue_chairs = 10 →
  green_chairs = 3 * blue_chairs →
  white_chairs = blue_chairs + green_chairs - 13 →
  blue_chairs + green_chairs + white_chairs = 67 := by
sorry

end NUMINAMATH_CALUDE_classroom_chairs_l4001_400132


namespace NUMINAMATH_CALUDE_a_values_in_A_l4001_400150

def A : Set ℝ := {2, 4, 6}

theorem a_values_in_A : {a : ℝ | a ∈ A ∧ (6 - a) ∈ A} = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_a_values_in_A_l4001_400150


namespace NUMINAMATH_CALUDE_jim_toads_difference_l4001_400173

theorem jim_toads_difference (tim_toads sarah_toads : ℕ) 
  (h1 : tim_toads = 30)
  (h2 : sarah_toads = 100)
  (h3 : sarah_toads = 2 * jim_toads)
  (h4 : jim_toads > tim_toads) : 
  jim_toads - tim_toads = 20 := by
sorry

end NUMINAMATH_CALUDE_jim_toads_difference_l4001_400173


namespace NUMINAMATH_CALUDE_arc_length_for_60_degrees_l4001_400139

/-- Given a circle with radius 10 cm and a central angle of 60°, 
    the length of the corresponding arc is 10π/3 cm. -/
theorem arc_length_for_60_degrees (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 10 → θ = 60 * π / 180 → l = r * θ → l = 10 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_60_degrees_l4001_400139


namespace NUMINAMATH_CALUDE_subgroup_samples_is_ten_l4001_400130

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  subgroup_size : ℕ
  total_samples : ℕ
  subgroup_samples : ℕ

/-- Calculates the number of samples from a subgroup in stratified sampling -/
def calculate_subgroup_samples (s : StratifiedSample) : ℚ :=
  s.total_samples * (s.subgroup_size : ℚ) / s.total_population

/-- Theorem stating that for the given scenario, the number of subgroup samples is 10 -/
theorem subgroup_samples_is_ten : 
  let s : StratifiedSample := {
    total_population := 1200,
    subgroup_size := 200,
    total_samples := 60,
    subgroup_samples := 10
  }
  calculate_subgroup_samples s = 10 := by
  sorry


end NUMINAMATH_CALUDE_subgroup_samples_is_ten_l4001_400130


namespace NUMINAMATH_CALUDE_work_completion_time_l4001_400188

/-- The time taken for A to complete the work alone -/
def time_A : ℝ := 10

/-- The time taken for A and B to complete the work together -/
def time_AB : ℝ := 4.444444444444445

/-- The time taken for B to complete the work alone -/
def time_B : ℝ := 8

/-- Theorem stating that given the time for A alone and A and B together, 
    the time for B alone is 8 days -/
theorem work_completion_time : 
  (1 / time_A + 1 / time_B = 1 / time_AB) → time_B = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4001_400188


namespace NUMINAMATH_CALUDE_peanuts_added_l4001_400117

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) 
  (h1 : initial_peanuts = 4)
  (h2 : final_peanuts = 10) :
  final_peanuts - initial_peanuts = 6 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_added_l4001_400117


namespace NUMINAMATH_CALUDE_rhombus_area_l4001_400192

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 6*d₁ + 8 = 0 → 
  d₂^2 - 6*d₂ + 8 = 0 → 
  d₁ ≠ d₂ →
  (1/2) * d₁ * d₂ = 4 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_area_l4001_400192


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l4001_400185

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The given line y = -3x + 6 -/
def givenLine : Line :=
  { slope := -3, yIntercept := 6 }

theorem y_intercept_of_parallel_line_through_point :
  ∃ (b : Line), 
    parallel b givenLine ∧ 
    pointOnLine b 4 (-2) ∧ 
    b.yIntercept = 10 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l4001_400185


namespace NUMINAMATH_CALUDE_derivative_of_linear_function_l4001_400163

theorem derivative_of_linear_function (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x + 5
  HasDerivAt f 3 x := by sorry

end NUMINAMATH_CALUDE_derivative_of_linear_function_l4001_400163


namespace NUMINAMATH_CALUDE_difference_of_squares_625_575_l4001_400124

theorem difference_of_squares_625_575 : 625^2 - 575^2 = 60000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_625_575_l4001_400124


namespace NUMINAMATH_CALUDE_mirror_side_length_l4001_400112

/-- Given a rectangular wall and a square mirror, proves that the mirror's side length is 34 inches -/
theorem mirror_side_length (wall_width wall_length mirror_area : ℝ) : 
  wall_width = 54 →
  wall_length = 42.81481481481482 →
  mirror_area = (wall_width * wall_length) / 2 →
  Real.sqrt mirror_area = 34 := by
  sorry

end NUMINAMATH_CALUDE_mirror_side_length_l4001_400112


namespace NUMINAMATH_CALUDE_farm_area_calculation_l4001_400116

/-- The total area of a farm with given sections and section area -/
def farm_total_area (num_sections : ℕ) (section_area : ℕ) : ℕ :=
  num_sections * section_area

/-- Theorem: The total area of a farm with 5 sections of 60 acres each is 300 acres -/
theorem farm_area_calculation :
  farm_total_area 5 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_farm_area_calculation_l4001_400116


namespace NUMINAMATH_CALUDE_college_students_count_l4001_400167

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 6) (h2 : girls = 200) :
  boys + girls = 440 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l4001_400167


namespace NUMINAMATH_CALUDE_intersection_and_subset_condition_l4001_400160

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (6 + 5*x - x^2)}
def B (m : ℝ) : Set ℝ := {x | (x - 1 + m) * (x - 1 - m) ≤ 0}

theorem intersection_and_subset_condition :
  (∃ m : ℝ, m = 3 ∧ A ∩ B m = {x | -1 ≤ x ∧ x ≤ 4}) ∧
  (∀ m : ℝ, m > 0 → (A ⊆ B m → m ≥ 5)) := by sorry

end NUMINAMATH_CALUDE_intersection_and_subset_condition_l4001_400160


namespace NUMINAMATH_CALUDE_greyEyedBlackHairedCount_l4001_400197

/-- Represents the characteristics of students in a class. -/
structure ClassCharacteristics where
  total : ℕ
  greenEyedRedHaired : ℕ
  blackHaired : ℕ
  greyEyed : ℕ

/-- Calculates the number of grey-eyed black-haired students given the class characteristics. -/
def greyEyedBlackHaired (c : ClassCharacteristics) : ℕ :=
  c.greyEyed - (c.total - c.blackHaired - c.greenEyedRedHaired)

/-- Theorem stating that for the given class characteristics, 
    the number of grey-eyed black-haired students is 20. -/
theorem greyEyedBlackHairedCount : 
  let c : ClassCharacteristics := {
    total := 60,
    greenEyedRedHaired := 20,
    blackHaired := 35,
    greyEyed := 25
  }
  greyEyedBlackHaired c = 20 := by
  sorry


end NUMINAMATH_CALUDE_greyEyedBlackHairedCount_l4001_400197


namespace NUMINAMATH_CALUDE_length_breadth_difference_l4001_400193

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Theorem stating that for a rectangular plot with given conditions, 
    the length is 12 meters more than the breadth. -/
theorem length_breadth_difference (plot : RectangularPlot) 
  (h1 : plot.length = 56)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : plot.total_fencing_cost = (2 * (plot.length + plot.breadth)) * plot.fencing_cost_per_meter) :
  plot.length - plot.breadth = 12 := by
  sorry

#check length_breadth_difference

end NUMINAMATH_CALUDE_length_breadth_difference_l4001_400193


namespace NUMINAMATH_CALUDE_weekly_earnings_theorem_l4001_400135

/-- Represents the shop's T-shirt sales and operating conditions -/
structure ShopConditions where
  women_tshirt_interval : ℕ := 30 -- minutes between women's T-shirt sales
  women_tshirt_price : ℕ := 18 -- price of women's T-shirt
  men_tshirt_interval : ℕ := 40 -- minutes between men's T-shirt sales
  men_tshirt_price : ℕ := 15 -- price of men's T-shirt
  daily_operating_minutes : ℕ := 720 -- minutes of operation per day (12 hours)
  days_per_week : ℕ := 7 -- number of operating days per week

/-- Calculates the weekly earnings from T-shirt sales given the shop conditions -/
def calculate_weekly_earnings (conditions : ShopConditions) : ℕ :=
  let women_daily_sales := conditions.daily_operating_minutes / conditions.women_tshirt_interval
  let men_daily_sales := conditions.daily_operating_minutes / conditions.men_tshirt_interval
  let daily_earnings := women_daily_sales * conditions.women_tshirt_price +
                        men_daily_sales * conditions.men_tshirt_price
  daily_earnings * conditions.days_per_week

/-- Theorem stating that the weekly earnings from T-shirt sales is $4914 -/
theorem weekly_earnings_theorem (shop : ShopConditions) :
  calculate_weekly_earnings shop = 4914 := by
  sorry


end NUMINAMATH_CALUDE_weekly_earnings_theorem_l4001_400135


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l4001_400134

theorem sum_smallest_largest_prime_1_to_50 : 
  ∃ (p q : Nat), 
    Prime p ∧ Prime q ∧ 
    p ≤ 50 ∧ q ≤ 50 ∧
    (∀ r, Prime r ∧ r ≤ 50 → p ≤ r) ∧
    (∀ r, Prime r ∧ r ≤ 50 → r ≤ q) ∧
    p + q = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l4001_400134


namespace NUMINAMATH_CALUDE_greatest_number_l4001_400128

theorem greatest_number : 
  let a := 1000 + 0.01
  let b := 1000 * 0.01
  let c := 1000 / 0.01
  let d := 0.01 / 1000
  let e := 1000 - 0.01
  (c > a) ∧ (c > b) ∧ (c > d) ∧ (c > e) := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_l4001_400128


namespace NUMINAMATH_CALUDE_product_of_geometric_terms_l4001_400164

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def geometric_sequence (n : ℕ) : ℕ := 2^(n - 1)

theorem product_of_geometric_terms : 
  geometric_sequence (arithmetic_sequence 1) * 
  geometric_sequence (arithmetic_sequence 3) * 
  geometric_sequence (arithmetic_sequence 5) = 4096 := by
sorry

end NUMINAMATH_CALUDE_product_of_geometric_terms_l4001_400164


namespace NUMINAMATH_CALUDE_work_completion_time_l4001_400144

/-- Given that:
    1. Ravi can do a piece of work in 15 days
    2. Ravi and another person together can do the work in 10 days
    Prove that the other person can do the work alone in 30 days -/
theorem work_completion_time (ravi_time : ℝ) (joint_time : ℝ) (other_time : ℝ) :
  ravi_time = 15 →
  joint_time = 10 →
  (1 / ravi_time + 1 / other_time = 1 / joint_time) →
  other_time = 30 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l4001_400144


namespace NUMINAMATH_CALUDE_kelvin_winning_strategy_l4001_400119

/-- Represents a player in the game -/
inductive Player
| Kelvin
| Alex

/-- Represents a single move in the game -/
structure Move where
  digit : Nat
  position : Nat

/-- Represents the state of the game -/
structure GameState where
  number : Nat
  currentPlayer : Player

/-- A strategy for Kelvin -/
def KelvinStrategy := GameState → Move

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Bool :=
  sorry

/-- Plays the game given Kelvin's strategy and Alex's moves -/
def playGame (strategy : KelvinStrategy) (alexMoves : List Move) : Bool :=
  sorry

/-- Theorem stating that Kelvin has a winning strategy -/
theorem kelvin_winning_strategy :
  ∃ (strategy : KelvinStrategy),
    ∀ (alexMoves : List Move),
      ¬(playGame strategy alexMoves) :=
sorry

end NUMINAMATH_CALUDE_kelvin_winning_strategy_l4001_400119


namespace NUMINAMATH_CALUDE_remove_fifteen_for_average_seven_point_five_l4001_400165

theorem remove_fifteen_for_average_seven_point_five :
  let sequence := List.range 15
  let sum := sequence.sum
  let removed := 15
  let remaining_sum := sum - removed
  let remaining_count := sequence.length - 1
  (remaining_sum : ℚ) / remaining_count = 15/2 := by
    sorry

end NUMINAMATH_CALUDE_remove_fifteen_for_average_seven_point_five_l4001_400165


namespace NUMINAMATH_CALUDE_complex_fraction_power_l4001_400180

theorem complex_fraction_power (i : ℂ) : i * i = -1 → (((1 + i) / (1 - i)) ^ 2014 : ℂ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l4001_400180


namespace NUMINAMATH_CALUDE_ball_count_theorem_l4001_400120

theorem ball_count_theorem (n : ℕ) : 
  (18 : ℝ) / (18 + 9 + n) = (30 : ℝ) / 100 → n = 42 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l4001_400120


namespace NUMINAMATH_CALUDE_reach_probability_is_5_128_l4001_400158

/-- Represents a point in the 2D coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a step direction -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of taking a specific step -/
def stepProbability : ℚ := 1 / 4

/-- The starting point -/
def start : Point := ⟨0, 0⟩

/-- The target point -/
def target : Point := ⟨3, 1⟩

/-- The maximum number of steps allowed -/
def maxSteps : ℕ := 5

/-- Calculates the probability of reaching the target point from the start point
    in at most maxSteps steps -/
def reachProbability (start target : Point) (maxSteps : ℕ) : ℚ :=
  sorry

theorem reach_probability_is_5_128 :
  reachProbability start target maxSteps = 5 / 128 :=
sorry

end NUMINAMATH_CALUDE_reach_probability_is_5_128_l4001_400158


namespace NUMINAMATH_CALUDE_trig_simplification_l4001_400107

theorem trig_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trig_simplification_l4001_400107


namespace NUMINAMATH_CALUDE_height_relation_l4001_400141

/-- Two right circular cylinders with equal volume and related radii -/
structure CylinderPair where
  r₁ : ℝ  -- radius of the first cylinder
  h₁ : ℝ  -- height of the first cylinder
  r₂ : ℝ  -- radius of the second cylinder
  h₂ : ℝ  -- height of the second cylinder
  r₁_pos : 0 < r₁  -- r₁ is positive
  h₁_pos : 0 < h₁  -- h₁ is positive
  r₂_pos : 0 < r₂  -- r₂ is positive
  h₂_pos : 0 < h₂  -- h₂ is positive
  volume_eq : π * r₁^2 * h₁ = π * r₂^2 * h₂  -- volumes are equal
  radius_relation : r₂ = 1.2 * r₁  -- r₂ is 20% more than r₁

/-- The theorem stating the relationship between the heights of the cylinders -/
theorem height_relation (cp : CylinderPair) : cp.h₁ = 1.44 * cp.h₂ := by
  sorry

#check height_relation

end NUMINAMATH_CALUDE_height_relation_l4001_400141


namespace NUMINAMATH_CALUDE_bookshelf_theorem_l4001_400156

/-- Given that:
    - A algebra books and H geometry books fill a bookshelf
    - S algebra books and M geometry books fill the same bookshelf
    - E algebra books alone fill the same bookshelf
    - A, H, S, M, E are different positive integers
    - Geometry books are thicker than algebra books
    Prove that E = (A * M - S * H) / (M - H) -/
theorem bookshelf_theorem (A H S M E : ℕ) 
    (hA : A > 0) (hH : H > 0) (hS : S > 0) (hM : M > 0) (hE : E > 0)
    (hDiff : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ 
             H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ 
             S ≠ M ∧ S ≠ E ∧ 
             M ≠ E)
    (hFillAH : ∃ d e : ℚ, d > 0 ∧ e > 0 ∧ e > d ∧ A * d + H * e = E * d)
    (hFillSM : ∃ d e : ℚ, d > 0 ∧ e > 0 ∧ e > d ∧ S * d + M * e = E * d) :
  E = (A * M - S * H) / (M - H) := by
sorry

end NUMINAMATH_CALUDE_bookshelf_theorem_l4001_400156


namespace NUMINAMATH_CALUDE_line_product_l4001_400147

/-- Given a line y = mx + b passing through points (0, -3) and (3, 6), prove that mb = -9 -/
theorem line_product (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b ∧ 
  (-3 : ℝ) = m * 0 + b ∧ 
  (6 : ℝ) = m * 3 + b → 
  m * b = -9 := by
  sorry

end NUMINAMATH_CALUDE_line_product_l4001_400147


namespace NUMINAMATH_CALUDE_bridget_apples_l4001_400109

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 4 + 6 = x → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_l4001_400109


namespace NUMINAMATH_CALUDE_seashell_count_l4001_400106

theorem seashell_count (sam_shells joan_shells : ℕ) 
  (h1 : sam_shells = 35) 
  (h2 : joan_shells = 18) : 
  sam_shells + joan_shells = 53 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_l4001_400106


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l4001_400110

theorem min_sum_of_squares (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  ∃ (m : ℝ), (∀ a b : ℝ, (a + 5) * (b - 5) = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 50 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l4001_400110


namespace NUMINAMATH_CALUDE_reflection_of_M_across_x_axis_l4001_400157

/-- The reflection of a point across the x-axis --/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point M --/
def M : ℝ × ℝ := (1, 2)

theorem reflection_of_M_across_x_axis :
  reflect_x M = (1, -2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_across_x_axis_l4001_400157


namespace NUMINAMATH_CALUDE_y_relationship_l4001_400113

theorem y_relationship : ∀ y₁ y₂ y₃ : ℝ,
  y₁ = (0.5 : ℝ) ^ (1/4 : ℝ) →
  y₂ = (0.6 : ℝ) ^ (1/4 : ℝ) →
  y₃ = (0.6 : ℝ) ^ (1/5 : ℝ) →
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l4001_400113


namespace NUMINAMATH_CALUDE_fraction_greater_than_one_necessary_not_sufficient_l4001_400170

theorem fraction_greater_than_one_necessary_not_sufficient :
  (∀ a b : ℝ, a > b ∧ b > 0 → a / b > 1) ∧
  (∃ a b : ℝ, a / b > 1 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_greater_than_one_necessary_not_sufficient_l4001_400170


namespace NUMINAMATH_CALUDE_ratio_percentage_difference_l4001_400166

theorem ratio_percentage_difference (A B : ℝ) (h : A / B = 5 / 8) :
  (B - A) / B = 37.5 / 100 ∧ (B - A) / A = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ratio_percentage_difference_l4001_400166


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l4001_400149

theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 2 * field_width →
    field_length = 112 →
    pond_side = 8 →
    (pond_side^2) / (field_length * field_width) = 1 / 98 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l4001_400149


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l4001_400155

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.black + b.yellow + b.blue

/-- Calculates the probability of drawing a specific color from a bag -/
def Bag.prob (b : Bag) (color : ℕ) : ℚ :=
  (color : ℚ) / (b.total : ℚ)

/-- The configuration of Bag A -/
def bagA : Bag := { white := 5, black := 6 }

/-- The configuration of Bag B -/
def bagB : Bag := { yellow := 8, blue := 6 }

/-- The configuration of Bag C -/
def bagC : Bag := { yellow := 3, blue := 9 }

/-- The probability of drawing a yellow marble as the second marble -/
def yellowProbability : ℚ :=
  bagA.prob bagA.white * bagB.prob bagB.yellow +
  bagA.prob bagA.black * bagC.prob bagC.yellow

theorem yellow_marble_probability :
  yellowProbability = 61 / 154 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l4001_400155


namespace NUMINAMATH_CALUDE_centipede_dressing_sequences_l4001_400114

/-- The number of legs a centipede has -/
def num_legs : ℕ := 10

/-- The total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- The number of valid sequences for a centipede to wear its socks and shoes -/
def valid_sequences : ℕ := Nat.factorial total_items / (2^ num_legs)

/-- Theorem stating the number of valid sequences for a centipede to wear its socks and shoes -/
theorem centipede_dressing_sequences :
  valid_sequences = Nat.factorial total_items / (2 ^ num_legs) :=
by sorry

end NUMINAMATH_CALUDE_centipede_dressing_sequences_l4001_400114


namespace NUMINAMATH_CALUDE_workers_in_first_group_l4001_400108

/-- Given two groups of workers building walls, this theorem proves the number of workers in the first group. -/
theorem workers_in_first_group 
  (wall_length_1 : ℝ) 
  (days_1 : ℝ) 
  (wall_length_2 : ℝ) 
  (days_2 : ℝ) 
  (workers_2 : ℕ) 
  (h1 : wall_length_1 = 66) 
  (h2 : days_1 = 12) 
  (h3 : wall_length_2 = 189.2) 
  (h4 : days_2 = 8) 
  (h5 : workers_2 = 86) :
  ∃ (workers_1 : ℕ), workers_1 = 57 ∧ 
    (workers_1 : ℝ) * days_1 * wall_length_2 = (workers_2 : ℝ) * days_2 * wall_length_1 :=
by sorry

end NUMINAMATH_CALUDE_workers_in_first_group_l4001_400108


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_condition_l4001_400102

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def consecutive_pair (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_condition :
  ∃ (n : ℕ) (a : ℕ),
    n = 2329089562800 ∧
    a ≥ 1 ∧ a < 30 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → (k = a ∨ k = a + 1 ∨ is_divisible n k)) ∧
    consecutive_pair a (a + 1) ∧
    ¬(is_divisible n a) ∧
    ¬(is_divisible n (a + 1)) ∧
    (∀ m : ℕ, m < n →
      ¬(∃ b : ℕ, b ≥ 1 ∧ b < 30 ∧
        (∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → (k = b ∨ k = b + 1 ∨ is_divisible m k)) ∧
        consecutive_pair b (b + 1) ∧
        ¬(is_divisible m b) ∧
        ¬(is_divisible m (b + 1)))) :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_condition_l4001_400102


namespace NUMINAMATH_CALUDE_tenth_minus_ninth_square_tiles_l4001_400199

-- Define the sequence of square side lengths
def squareSideLength (n : ℕ) : ℕ := n

-- Define the number of tiles in the nth square
def tilesInSquare (n : ℕ) : ℕ := (squareSideLength n) ^ 2

-- Theorem statement
theorem tenth_minus_ninth_square_tiles : 
  tilesInSquare 10 - tilesInSquare 9 = 19 := by sorry

end NUMINAMATH_CALUDE_tenth_minus_ninth_square_tiles_l4001_400199


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l4001_400145

theorem sum_of_roots_cubic_equation :
  let f (x : ℝ) := 3 * x^3 - 6 * x^2 - 9 * x
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ + r₂ + r₃ = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l4001_400145


namespace NUMINAMATH_CALUDE_dragon_rope_problem_l4001_400162

theorem dragon_rope_problem (a b c : ℕ) (h_prime : Nat.Prime c) :
  let tower_radius : ℝ := 10
  let rope_length : ℝ := 25
  let height_difference : ℝ := 3
  let rope_touching_tower : ℝ := (a - Real.sqrt b) / c
  (tower_radius > 0 ∧ rope_length > tower_radius ∧ height_difference > 0 ∧
   rope_touching_tower > 0 ∧ rope_touching_tower < rope_length) →
  a + b + c = 352 :=
by sorry

end NUMINAMATH_CALUDE_dragon_rope_problem_l4001_400162


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l4001_400184

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_x_1 :
  ∃ (m c : ℝ), 
    (∀ x y : ℝ, y = m * x + c ↔ m * x - y + c = 0) ∧
    (m = f' 1) ∧
    (f 1 = m * 1 + c) ∧
    (m * x - y + c = 0 ↔ 4 * x - y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l4001_400184


namespace NUMINAMATH_CALUDE_total_spider_legs_l4001_400189

/-- The number of spiders in Ivy's room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs in Ivy's room is 32 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l4001_400189


namespace NUMINAMATH_CALUDE_constant_sequence_l4001_400111

theorem constant_sequence (a : ℕ → ℝ) : 
  (∀ (b : ℕ → ℕ), (∀ n : ℕ, b n ≠ b (n + 1) ∧ (b n ∣ b (n + 1))) → 
    ∃ (d : ℝ), ∀ n : ℕ, a (b (n + 1)) - a (b n) = d) →
  ∃ (c : ℝ), ∀ n : ℕ, a n = c :=
by sorry

end NUMINAMATH_CALUDE_constant_sequence_l4001_400111


namespace NUMINAMATH_CALUDE_fraction_equality_l4001_400182

theorem fraction_equality : (45 : ℚ) / (8 - 3 / 7) = 315 / 53 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l4001_400182


namespace NUMINAMATH_CALUDE_inscribed_squares_side_length_l4001_400103

/-- Right triangle ABC with two inscribed squares -/
structure RightTriangleWithSquares where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side AC (hypotenuse) -/
  ac : ℝ
  /-- Side length of the inscribed squares -/
  s : ℝ
  /-- AB = 6 -/
  ab_eq : ab = 6
  /-- BC = 8 -/
  bc_eq : bc = 8
  /-- AC = 10 -/
  ac_eq : ac = 10
  /-- Pythagorean theorem holds -/
  pythagorean : ab ^ 2 + bc ^ 2 = ac ^ 2
  /-- The two squares do not overlap -/
  non_overlapping : 2 * s ≤ (ab * bc) / ac

/-- The side length of each inscribed square is 2.4 -/
theorem inscribed_squares_side_length (t : RightTriangleWithSquares) : t.s = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_side_length_l4001_400103


namespace NUMINAMATH_CALUDE_unique_assignment_l4001_400137

-- Define the polyhedron structure
structure Polyhedron :=
  (faces : Fin 2022 → ℝ)
  (adjacent : Fin 2022 → Finset (Fin 2022))
  (adjacent_symmetric : ∀ i j, j ∈ adjacent i ↔ i ∈ adjacent j)

-- Define the property of being a valid number assignment
def ValidAssignment (p : Polyhedron) : Prop :=
  ∀ i, p.faces i = if i = 0 then 26
                   else if i = 1 then 4
                   else if i = 2 then 2022
                   else (p.adjacent i).sum p.faces / (p.adjacent i).card

-- Theorem statement
theorem unique_assignment (p : Polyhedron) :
  ∃! f : Fin 2022 → ℝ, ValidAssignment { faces := f, adjacent := p.adjacent, adjacent_symmetric := p.adjacent_symmetric } :=
sorry

end NUMINAMATH_CALUDE_unique_assignment_l4001_400137


namespace NUMINAMATH_CALUDE_divisibility_count_l4001_400196

def count_numbers (n : ℕ) : ℕ :=
  (n / 6) - ((n / 12) + (n / 18) - (n / 36))

theorem divisibility_count : count_numbers 2018 = 112 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_count_l4001_400196


namespace NUMINAMATH_CALUDE_all_less_than_one_l4001_400181

theorem all_less_than_one (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a^2 < b) (hbc : b^2 < c) (hca : c^2 < a) :
  a < 1 ∧ b < 1 ∧ c < 1 := by
  sorry

end NUMINAMATH_CALUDE_all_less_than_one_l4001_400181


namespace NUMINAMATH_CALUDE_square_of_complex_l4001_400154

theorem square_of_complex (z : ℂ) : z = 2 - 3*I → z^2 = -5 - 12*I := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_l4001_400154


namespace NUMINAMATH_CALUDE_not_fourth_power_prime_minus_four_l4001_400142

theorem not_fourth_power_prime_minus_four (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ¬ ∃ (a : ℕ), p - 4 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_not_fourth_power_prime_minus_four_l4001_400142


namespace NUMINAMATH_CALUDE_tim_income_percentage_l4001_400100

theorem tim_income_percentage (tim juan mary : ℝ) 
  (h1 : mary = 1.7 * tim) 
  (h2 : mary = 1.02 * juan) : 
  (juan - tim) / juan = 0.4 := by
sorry

end NUMINAMATH_CALUDE_tim_income_percentage_l4001_400100


namespace NUMINAMATH_CALUDE_problem_figure_perimeter_l4001_400105

/-- Represents a figure made of unit squares -/
structure UnitSquareFigure where
  bottom_row : Nat
  left_column : Nat
  top_row : Nat
  right_column : Nat

/-- The specific figure described in the problem -/
def problem_figure : UnitSquareFigure :=
  { bottom_row := 3
  , left_column := 2
  , top_row := 4
  , right_column := 3 }

/-- Calculates the perimeter of a UnitSquareFigure -/
def perimeter (figure : UnitSquareFigure) : Nat :=
  figure.bottom_row + figure.left_column + figure.top_row + figure.right_column

theorem problem_figure_perimeter : perimeter problem_figure = 12 := by
  sorry

#eval perimeter problem_figure

end NUMINAMATH_CALUDE_problem_figure_perimeter_l4001_400105


namespace NUMINAMATH_CALUDE_stratified_sample_category_a_l4001_400131

/-- Represents the number of students in each school category -/
structure SchoolCategories where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the number of students to be sampled from Category A schools
    using stratified sampling -/
def sampleSizeA (categories : SchoolCategories) (totalSample : ℕ) : ℕ :=
  (categories.a * totalSample) / (categories.a + categories.b + categories.c)

/-- Theorem stating that for the given school categories and sample size,
    the number of students to be selected from Category A is 200 -/
theorem stratified_sample_category_a 
  (categories : SchoolCategories)
  (h1 : categories.a = 2000)
  (h2 : categories.b = 3000)
  (h3 : categories.c = 4000)
  (totalSample : ℕ)
  (h4 : totalSample = 900) :
  sampleSizeA categories totalSample = 200 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_category_a_l4001_400131


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l4001_400191

theorem sum_of_reciprocals_of_quadratic_roots :
  ∀ p q : ℝ, p^2 - 10*p + 3 = 0 → q^2 - 10*q + 3 = 0 → p ≠ q →
  1/p + 1/q = 10/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l4001_400191


namespace NUMINAMATH_CALUDE_nonnegative_integer_solutions_x_squared_eq_6x_l4001_400172

theorem nonnegative_integer_solutions_x_squared_eq_6x :
  ∃! n : ℕ, (∃ s : Finset ℕ, s.card = n ∧
    ∀ x : ℕ, x ∈ s ↔ x^2 = 6*x) ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_nonnegative_integer_solutions_x_squared_eq_6x_l4001_400172


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l4001_400153

theorem rectangle_area_difference (A B a b : ℕ) 
  (h1 : A = 20) (h2 : B = 30) (h3 : a = 4) (h4 : b = 7) : 
  (A * B - a * b) - ((A - a) * B + A * (B - b)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l4001_400153


namespace NUMINAMATH_CALUDE_five_topping_pizzas_l4001_400195

theorem five_topping_pizzas (n : ℕ) (k : ℕ) : n = 8 ∧ k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_topping_pizzas_l4001_400195


namespace NUMINAMATH_CALUDE_multiply_three_six_and_quarter_l4001_400175

theorem multiply_three_six_and_quarter : 3.6 * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_six_and_quarter_l4001_400175


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l4001_400136

theorem cubic_root_equation_solution :
  ∃ x : ℝ, (((3 - x) ^ (1/3 : ℝ)) + ((x - 1) ^ (1/2 : ℝ)) = 2) ∧ (x = 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l4001_400136


namespace NUMINAMATH_CALUDE_probability_all_selected_l4001_400140

/-- The probability of Ram being selected -/
def p_ram : ℚ := 6/7

/-- The initial probability of Ravi being selected -/
def p_ravi_initial : ℚ := 1/5

/-- The probability of Ravi being selected given Ram is selected -/
def p_ravi_given_ram : ℚ := 2/5

/-- The initial probability of Rajesh being selected -/
def p_rajesh_initial : ℚ := 2/3

/-- The probability of Rajesh being selected given Ravi is selected -/
def p_rajesh_given_ravi : ℚ := 1/2

/-- The theorem stating the probability of all three brothers being selected -/
theorem probability_all_selected : 
  p_ram * p_ravi_given_ram * p_rajesh_given_ravi = 6/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_selected_l4001_400140


namespace NUMINAMATH_CALUDE_p_shape_points_count_l4001_400125

/-- Represents the "П" shape formed from a square --/
structure PShape :=
  (side_length : ℕ)

/-- Calculates the number of points along the "П" shape --/
def count_points (p : PShape) : ℕ :=
  3 * (p.side_length + 1) - 2

theorem p_shape_points_count :
  ∀ (p : PShape), p.side_length = 10 → count_points p = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_p_shape_points_count_l4001_400125


namespace NUMINAMATH_CALUDE_polynomial_root_k_value_l4001_400127

theorem polynomial_root_k_value :
  ∀ k : ℚ, (3 : ℚ)^4 + k * (3 : ℚ)^2 - 26 = 0 → k = -55/9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_k_value_l4001_400127


namespace NUMINAMATH_CALUDE_complex_product_theorem_l4001_400129

theorem complex_product_theorem (α₁ α₂ α₃ : ℝ) : 
  let z₁ : ℂ := Complex.exp (I * α₁)
  let z₂ : ℂ := Complex.exp (I * α₂)
  let z₃ : ℂ := Complex.exp (I * α₃)
  z₁ * z₂ = Complex.exp (I * (α₁ + α₂)) →
  z₂ * z₃ = Complex.exp (I * (α₂ + α₃)) →
  z₁ * z₂ * z₃ = Complex.exp (I * (α₁ + α₂ + α₃)) :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l4001_400129
