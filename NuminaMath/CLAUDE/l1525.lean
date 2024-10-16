import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1525_152567

theorem equation_solution : ∃! x : ℝ, (1 + x) / 4 - (x - 2) / 8 = 1 := by
  use 4
  constructor
  · -- Prove that x = 4 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l1525_152567


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l1525_152548

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l1525_152548


namespace NUMINAMATH_CALUDE_fermat_class_size_l1525_152537

theorem fermat_class_size : ∀ (total : ℕ),
  (total > 0) →
  (0.2 * (total : ℝ) = (total * 20 / 100 : ℕ)) →
  (0.35 * (total : ℝ) = (total * 35 / 100 : ℕ)) →
  (total - (total * 20 / 100) - (total * 35 / 100) = 9) →
  total = 20 := by
sorry

end NUMINAMATH_CALUDE_fermat_class_size_l1525_152537


namespace NUMINAMATH_CALUDE_triangle_properties_l1525_152556

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_properties (t : Triangle) :
  (t.a^2 + t.b^2 < t.c^2 → π/2 < t.C) ∧
  (Real.sin t.A > Real.sin t.B → t.a > t.b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1525_152556


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1525_152566

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ s = -b / a) :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 2000 * x - 2001
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ s = -2000) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1525_152566


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l1525_152526

/-- The number of ways to select k items from n distinct items, where order matters. -/
def permutations (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

/-- The number of ways to select 2 books from 5 different books and give them to 2 students. -/
def book_distribution_ways : ℕ := permutations 5 2

theorem book_distribution_theorem : book_distribution_ways = 20 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l1525_152526


namespace NUMINAMATH_CALUDE_third_column_second_row_l1525_152510

/-- Represents a position in a classroom grid -/
structure Position :=
  (column : ℕ)
  (row : ℕ)

/-- The coordinate system for the classroom -/
def classroom_coordinate_system : Position → Bool
  | ⟨1, 2⟩ => true  -- This represents the condition that (1,2) is a valid position
  | _ => false

/-- Theorem: In the given coordinate system, (3,2) represents the 3rd column and 2nd row -/
theorem third_column_second_row :
  classroom_coordinate_system ⟨1, 2⟩ → 
  (∃ p : Position, p.column = 3 ∧ p.row = 2 ∧ classroom_coordinate_system p) :=
sorry

end NUMINAMATH_CALUDE_third_column_second_row_l1525_152510


namespace NUMINAMATH_CALUDE_arrangement_count_proof_l1525_152594

/-- The number of ways to arrange 4 distinct digits in a 2 × 3 grid with 2 empty cells -/
def arrangement_count : ℕ := 360

/-- The size of the grid -/
def grid_size : ℕ × ℕ := (2, 3)

/-- The number of available digits -/
def digit_count : ℕ := 4

/-- The number of empty cells -/
def empty_cell_count : ℕ := 2

/-- The total number of cells in the grid -/
def total_cells : ℕ := grid_size.1 * grid_size.2

theorem arrangement_count_proof :
  arrangement_count = (Nat.choose total_cells empty_cell_count) * (Nat.factorial digit_count) :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_proof_l1525_152594


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_one_l1525_152579

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The first line ax + 3y + 3 = 0 -/
def line1 (a : ℝ) : Line :=
  { a := a, b := 3, c := 3 }

/-- The second line x + (a-2)y + l = 0 -/
def line2 (a l : ℝ) : Line :=
  { a := 1, b := a - 2, c := l }

/-- Theorem stating that the lines are parallel if and only if a = -1 -/
theorem lines_parallel_iff_a_eq_neg_one (a l : ℝ) :
  parallel (line1 a) (line2 a l) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_one_l1525_152579


namespace NUMINAMATH_CALUDE_train_length_l1525_152517

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 →
  time_s = 3.9996800255979523 →
  ∃ (length_m : ℝ), abs (length_m - 99.992) < 0.001 ∧ 
    length_m = speed_kmh * (1000 / 3600) * time_s :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1525_152517


namespace NUMINAMATH_CALUDE_elena_savings_theorem_l1525_152541

/-- The amount Elena saves when buying binders with a discount and rebate -/
def elenaSavings (numBinders : ℕ) (pricePerBinder : ℚ) (discountRate : ℚ) (rebateThreshold : ℚ) (rebateAmount : ℚ) : ℚ :=
  let originalCost := numBinders * pricePerBinder
  let discountedPrice := originalCost * (1 - discountRate)
  let finalPrice := if originalCost > rebateThreshold then discountedPrice - rebateAmount else discountedPrice
  originalCost - finalPrice

/-- Theorem stating that Elena saves $10.25 under the given conditions -/
theorem elena_savings_theorem :
  elenaSavings 7 3 (25 / 100) 20 5 = (41 / 4) := by
  sorry

end NUMINAMATH_CALUDE_elena_savings_theorem_l1525_152541


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l1525_152568

theorem tan_alpha_plus_pi_sixth (α : Real) 
  (h : Real.cos α + 2 * Real.cos (α + π/3) = 0) : 
  Real.tan (α + π/6) = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l1525_152568


namespace NUMINAMATH_CALUDE_boat_race_distance_l1525_152542

/-- The distance between two points A and B traveled by two boats with different speeds and start times -/
theorem boat_race_distance 
  (a b d n : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d ≥ 0) 
  (hn : n > 0) 
  (hab : a > b) :
  ∃ x : ℝ, x > 0 ∧ x = (a * (d + b * n)) / (a - b) ∧
    x / a + n = (x - d) / b :=
by sorry

end NUMINAMATH_CALUDE_boat_race_distance_l1525_152542


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_3_4_l1525_152596

/-- Given an angle α where a point on its terminal side has coordinates (3,4), prove that sin α = 4/5 -/
theorem sin_alpha_for_point_3_4 (α : Real) :
  (∃ (x y : Real), x = 3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_3_4_l1525_152596


namespace NUMINAMATH_CALUDE_bubble_sort_iterations_for_given_list_l1525_152553

def bubble_sort_iterations (list : List Int) : Nat :=
  sorry

theorem bubble_sort_iterations_for_given_list :
  bubble_sort_iterations [6, -3, 0, 15] = 3 := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_iterations_for_given_list_l1525_152553


namespace NUMINAMATH_CALUDE_closest_integer_to_thirteen_minus_sqrt_thirteen_l1525_152571

theorem closest_integer_to_thirteen_minus_sqrt_thirteen : 
  ∃ (n : ℤ), ∀ (m : ℤ), |13 - Real.sqrt 13 - n| ≤ |13 - Real.sqrt 13 - m| → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_thirteen_minus_sqrt_thirteen_l1525_152571


namespace NUMINAMATH_CALUDE_average_price_approx_1_70_l1525_152534

/-- The average price per bottle given the purchase of large and small bottles -/
def average_price_per_bottle (large_bottles : ℕ) (small_bottles : ℕ) 
  (large_price : ℚ) (small_price : ℚ) : ℚ :=
  ((large_bottles : ℚ) * large_price + (small_bottles : ℚ) * small_price) / 
  ((large_bottles : ℚ) + (small_bottles : ℚ))

/-- Theorem stating that the average price per bottle is approximately $1.70 -/
theorem average_price_approx_1_70 :
  let large_bottles : ℕ := 1300
  let small_bottles : ℕ := 750
  let large_price : ℚ := 189/100  -- $1.89
  let small_price : ℚ := 138/100  -- $1.38
  abs (average_price_per_bottle large_bottles small_bottles large_price small_price - 17/10) < 1/100
  := by sorry

end NUMINAMATH_CALUDE_average_price_approx_1_70_l1525_152534


namespace NUMINAMATH_CALUDE_karen_total_distance_l1525_152535

/-- The number of shelves in the library. -/
def num_shelves : ℕ := 4

/-- The number of books on each shelf. -/
def books_per_shelf : ℕ := 400

/-- The total number of books in the library. -/
def total_books : ℕ := num_shelves * books_per_shelf

/-- The distance in miles from the library to Karen's home. -/
def distance_to_home : ℕ := total_books

/-- The total distance Karen bikes from home to library and back. -/
def total_distance : ℕ := 2 * distance_to_home

/-- Theorem stating that the total distance Karen bikes is 3200 miles. -/
theorem karen_total_distance : total_distance = 3200 := by
  sorry

end NUMINAMATH_CALUDE_karen_total_distance_l1525_152535


namespace NUMINAMATH_CALUDE_two_blue_gumballs_probability_l1525_152519

theorem two_blue_gumballs_probability 
  (pink_prob : ℝ) 
  (h_pink_prob : pink_prob = 1/3) : 
  let blue_prob := 1 - pink_prob
  (blue_prob * blue_prob) = 4/9 := by
sorry

end NUMINAMATH_CALUDE_two_blue_gumballs_probability_l1525_152519


namespace NUMINAMATH_CALUDE_unique_solution_system_l1525_152569

theorem unique_solution_system (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (2 * x₁ = x₅^2 - 23) ∧
  (4 * x₂ = x₁^2 + 7) ∧
  (6 * x₃ = x₂^2 + 14) ∧
  (8 * x₄ = x₃^2 + 23) ∧
  (10 * x₅ = x₄^2 + 34) →
  x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧ x₅ = 5 :=
by sorry

#check unique_solution_system

end NUMINAMATH_CALUDE_unique_solution_system_l1525_152569


namespace NUMINAMATH_CALUDE_binomial_largest_coefficient_l1525_152560

/-- 
Given a positive integer n, if the binomial coefficient in the expansion of (2+x)^n 
is largest in the 4th and 5th terms, then n = 7.
-/
theorem binomial_largest_coefficient (n : ℕ+) : 
  (∀ k : ℕ, k ≠ 3 ∧ k ≠ 4 → Nat.choose n k ≤ Nat.choose n 3) ∧
  (∀ k : ℕ, k ≠ 3 ∧ k ≠ 4 → Nat.choose n k ≤ Nat.choose n 4) ∧
  Nat.choose n 3 = Nat.choose n 4 →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_binomial_largest_coefficient_l1525_152560


namespace NUMINAMATH_CALUDE_phenomena_explanation_l1525_152591

-- Define the basic fact
def two_points_determine_line : Prop := sorry

-- Define the phenomena
def phenomenon_1 : Prop := sorry
def phenomenon_2 : Prop := sorry
def phenomenon_3 : Prop := sorry
def phenomenon_4 : Prop := sorry

-- Define the explanations
def explains (fact : Prop) (phenomenon : Prop) : Prop := sorry

-- Theorem statement
theorem phenomena_explanation :
  two_points_determine_line →
  (¬ explains two_points_determine_line phenomenon_1) ∧
  (explains two_points_determine_line phenomenon_2) ∧
  (explains two_points_determine_line phenomenon_3) ∧
  (explains two_points_determine_line phenomenon_4) :=
by sorry

end NUMINAMATH_CALUDE_phenomena_explanation_l1525_152591


namespace NUMINAMATH_CALUDE_vehicle_Y_ahead_distance_l1525_152598

-- Define the vehicles and their properties
structure Vehicle where
  speed : ℝ
  initialPosition : ℝ

-- Define the problem parameters
def time : ℝ := 5
def vehicleX : Vehicle := { speed := 36, initialPosition := 22 }
def vehicleY : Vehicle := { speed := 45, initialPosition := 0 }

-- Define the function to calculate the position of a vehicle after a given time
def position (v : Vehicle) (t : ℝ) : ℝ :=
  v.initialPosition + v.speed * t

-- Theorem statement
theorem vehicle_Y_ahead_distance : 
  position vehicleY time - position vehicleX time = 23 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_Y_ahead_distance_l1525_152598


namespace NUMINAMATH_CALUDE_annual_salary_calculation_l1525_152565

theorem annual_salary_calculation (hourly_wage : ℝ) (hours_per_day : ℕ) (days_per_month : ℕ) :
  hourly_wage = 8.50 →
  hours_per_day = 8 →
  days_per_month = 20 →
  hourly_wage * hours_per_day * days_per_month * 12 = 16320 :=
by
  sorry

end NUMINAMATH_CALUDE_annual_salary_calculation_l1525_152565


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1525_152529

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≤ -2 → ∀ x y, -1 ≤ x ∧ x ≤ y → f a x ≤ f a y) ∧
  (∃ a', a' > -2 ∧ ∀ x y, -1 ≤ x ∧ x ≤ y → f a' x ≤ f a' y) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1525_152529


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1525_152570

/-- An arithmetic sequence (b_n) with given first three terms -/
def arithmetic_sequence (x : ℝ) (n : ℕ) : ℝ :=
  x^2 + (n - 1) * x

theorem arithmetic_sequence_eighth_term (x : ℝ) :
  arithmetic_sequence x 8 = 2 * x^2 + 7 * x := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1525_152570


namespace NUMINAMATH_CALUDE_walk_in_closet_doorway_width_l1525_152588

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a rectangular opening (door or window) -/
structure Opening where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangular surface -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- Calculates the total wall area of a room -/
def totalWallArea (room : RoomDimensions) : ℝ :=
  2 * (room.width + room.length) * room.height

/-- Calculates the area of an opening -/
def openingArea (opening : Opening) : ℝ := rectangleArea opening.width opening.height

theorem walk_in_closet_doorway_width 
  (room : RoomDimensions)
  (doorway1 : Opening)
  (window : Opening)
  (closetDoorwayHeight : ℝ)
  (areaToPaint : ℝ)
  (h1 : room.width = 20)
  (h2 : room.length = 20)
  (h3 : room.height = 8)
  (h4 : doorway1.width = 3)
  (h5 : doorway1.height = 7)
  (h6 : window.width = 6)
  (h7 : window.height = 4)
  (h8 : closetDoorwayHeight = 7)
  (h9 : areaToPaint = 560) :
  ∃ (closetDoorwayWidth : ℝ), 
    closetDoorwayWidth = 5 ∧
    areaToPaint = totalWallArea room - openingArea doorway1 - openingArea window - rectangleArea closetDoorwayWidth closetDoorwayHeight :=
by sorry

end NUMINAMATH_CALUDE_walk_in_closet_doorway_width_l1525_152588


namespace NUMINAMATH_CALUDE_problem_statement_l1525_152549

theorem problem_statement (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 11) 
  (h2 : y = 1) : 
  5 * x + 3 = 18 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1525_152549


namespace NUMINAMATH_CALUDE_newspaper_profit_is_550_l1525_152536

/-- Calculate the profit from selling newspapers given the following conditions:
  * Total number of newspapers bought
  * Selling price per newspaper
  * Percentage of newspapers sold
  * Percentage discount on buying price compared to selling price
-/
def calculate_profit (total_newspapers : ℕ) (selling_price : ℚ) (sold_percentage : ℚ) (discount_percentage : ℚ) : ℚ :=
  let buying_price := selling_price * (1 - discount_percentage)
  let total_cost := buying_price * total_newspapers
  let newspapers_sold := (sold_percentage * total_newspapers).floor
  let revenue := selling_price * newspapers_sold
  revenue - total_cost

/-- Theorem stating that under the given conditions, the profit is $550 -/
theorem newspaper_profit_is_550 :
  calculate_profit 500 2 0.8 0.75 = 550 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_profit_is_550_l1525_152536


namespace NUMINAMATH_CALUDE_sliding_chord_annulus_area_l1525_152516

/-- The area of the annulus formed by a sliding chord on a circle -/
theorem sliding_chord_annulus_area
  (R : ℝ) -- radius of the outer circle
  (a b : ℝ) -- distances from point C to ends A and B of the chord
  (h1 : R > 0) -- radius is positive
  (h2 : a > 0) -- distance a is positive
  (h3 : b > 0) -- distance b is positive
  (h4 : a + b ≤ 2 * R) -- chord length constraint
  : ∃ (r : ℝ), -- radius of the inner circle
    r^2 = R^2 - a * b ∧ 
    π * R^2 - π * r^2 = π * a * b :=
by sorry

end NUMINAMATH_CALUDE_sliding_chord_annulus_area_l1525_152516


namespace NUMINAMATH_CALUDE_volume_change_with_pressure_increase_l1525_152593

theorem volume_change_with_pressure_increase {P V P' V' : ℝ} (h1 : P > 0) (h2 : V > 0) :
  (P * V = P' * V') → -- inverse proportionality
  (P' = 1.2 * P) → -- 20% increase in pressure
  (V' = V * (5/6)) -- 16.67% decrease in volume
  := by sorry

end NUMINAMATH_CALUDE_volume_change_with_pressure_increase_l1525_152593


namespace NUMINAMATH_CALUDE_brother_age_proof_l1525_152515

/-- Trevor's current age -/
def Trevor_current_age : ℕ := 11

/-- Trevor's future age when the condition is met -/
def Trevor_future_age : ℕ := 24

/-- Trevor's older brother's current age -/
def Brother_current_age : ℕ := 20

theorem brother_age_proof :
  (Trevor_future_age - Trevor_current_age = Brother_current_age - Trevor_current_age) ∧
  (Brother_current_age + (Trevor_future_age - Trevor_current_age) = 3 * Trevor_current_age) :=
by sorry

end NUMINAMATH_CALUDE_brother_age_proof_l1525_152515


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l1525_152508

theorem quadratic_equation_1 : ∃ x₁ x₂ : ℝ, x₁ = 6 ∧ x₂ = -1 ∧ x₁^2 - 5*x₁ - 6 = 0 ∧ x₂^2 - 5*x₂ - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l1525_152508


namespace NUMINAMATH_CALUDE_delta_value_l1525_152530

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ - 1 → Δ = -11 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l1525_152530


namespace NUMINAMATH_CALUDE_variance_implies_fluctuation_l1525_152552

-- Define a type for our data set
def DataSet := List ℝ

-- Define variance
def variance (data : DataSet) : ℝ := sorry

-- Define a measure of fluctuation
def fluctuation (data : DataSet) : ℝ := sorry

-- Theorem statement
theorem variance_implies_fluctuation (data1 data2 : DataSet) :
  variance data1 > variance data2 → fluctuation data1 > fluctuation data2 := by
  sorry

end NUMINAMATH_CALUDE_variance_implies_fluctuation_l1525_152552


namespace NUMINAMATH_CALUDE_evaluate_expression_l1525_152503

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1525_152503


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l1525_152581

theorem linear_systems_solutions :
  -- System 1
  (2 : ℝ) + 2 * (1 : ℝ) = (4 : ℝ) ∧
  (2 : ℝ) + 3 * (1 : ℝ) = (5 : ℝ) ∧
  -- System 2
  2 * (2 : ℝ) - 5 * (5 : ℝ) = (-21 : ℝ) ∧
  4 * (2 : ℝ) + 3 * (5 : ℝ) = (23 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l1525_152581


namespace NUMINAMATH_CALUDE_charity_event_selection_l1525_152575

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of students -/
def total_students : ℕ := 10

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of students excluding A and B -/
def remaining_students : ℕ := total_students - 2

theorem charity_event_selection :
  choose total_students selected_students - choose remaining_students selected_students = 140 := by
  sorry

end NUMINAMATH_CALUDE_charity_event_selection_l1525_152575


namespace NUMINAMATH_CALUDE_fair_prize_distribution_l1525_152502

/-- Represents the probability of winning a single game -/
def win_probability : ℚ := 1 / 2

/-- Represents the total prize money in yuan -/
def total_prize : ℕ := 12000

/-- Represents the score of the leading player -/
def leading_score : ℕ := 2

/-- Represents the score of the trailing player -/
def trailing_score : ℕ := 1

/-- Represents the number of games needed to win the series -/
def games_to_win : ℕ := 3

/-- Calculates the fair prize for the leading player -/
def fair_prize (p : ℚ) (total : ℕ) : ℚ := p * total

/-- Theorem stating the fair prize distribution for the leading player -/
theorem fair_prize_distribution :
  fair_prize ((win_probability + win_probability * win_probability) : ℚ) total_prize =
  (3 / 4 : ℚ) * total_prize :=
sorry

end NUMINAMATH_CALUDE_fair_prize_distribution_l1525_152502


namespace NUMINAMATH_CALUDE_distance_between_squares_l1525_152561

/-- Given two squares, one with perimeter 8 cm and another with area 36 cm²,
    prove that the distance between opposite corners is √80 cm. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ)
    (h1 : small_square_perimeter = 8)
    (h2 : large_square_area = 36) :
    Real.sqrt ((small_square_perimeter / 4 + Real.sqrt large_square_area) ^ 2 +
               (Real.sqrt large_square_area - small_square_perimeter / 4) ^ 2) = Real.sqrt 80 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_squares_l1525_152561


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1525_152544

open Set

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1525_152544


namespace NUMINAMATH_CALUDE_sin_90_degrees_l1525_152572

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l1525_152572


namespace NUMINAMATH_CALUDE_angle_line_plane_range_l1525_152551

/-- The angle between a line and a plane is the acute angle between the line and its projection onto the plane. -/
def angle_line_plane (line : Line) (plane : Plane) : ℝ := sorry

theorem angle_line_plane_range (line : Line) (plane : Plane) :
  let θ := angle_line_plane line plane
  0 ≤ θ ∧ θ ≤ 90 :=
sorry

end NUMINAMATH_CALUDE_angle_line_plane_range_l1525_152551


namespace NUMINAMATH_CALUDE_y_equals_five_l1525_152527

/-- A line passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

/-- The theorem stating that under the given conditions, y must equal 5 -/
theorem y_equals_five (x y : ℝ) :
  line_k x 6 →
  line_k 10 y →
  x * y = 60 →
  y = 5 := by sorry

end NUMINAMATH_CALUDE_y_equals_five_l1525_152527


namespace NUMINAMATH_CALUDE_inequality_proof_l1525_152540

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1525_152540


namespace NUMINAMATH_CALUDE_product_of_distances_to_asymptotes_l1525_152585

/-- Represents a hyperbola with equation y²/2 - x²/b = 1 -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0

/-- A point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : y^2 / 2 - x^2 / h.b = 1

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The distance from a point to an asymptote of the hyperbola -/
def distance_to_asymptote (h : Hyperbola) (p : PointOnHyperbola h) : ℝ := sorry

/-- The theorem stating the product of distances to asymptotes -/
theorem product_of_distances_to_asymptotes (h : Hyperbola) 
  (h_ecc : eccentricity h = 2) (p : PointOnHyperbola h) : 
  (distance_to_asymptote h p) * (distance_to_asymptote h p) = 3/2 := 
sorry

end NUMINAMATH_CALUDE_product_of_distances_to_asymptotes_l1525_152585


namespace NUMINAMATH_CALUDE_catfish_dinner_price_l1525_152562

/-- The price of a catfish dinner at River Joe's Seafood Diner -/
def catfish_price : ℚ := 6

/-- The price of a popcorn shrimp dinner at River Joe's Seafood Diner -/
def popcorn_shrimp_price : ℚ := 7/2

/-- The total number of orders filled -/
def total_orders : ℕ := 26

/-- The number of popcorn shrimp orders sold -/
def popcorn_shrimp_orders : ℕ := 9

/-- The total revenue collected -/
def total_revenue : ℚ := 267/2

theorem catfish_dinner_price :
  catfish_price * (total_orders - popcorn_shrimp_orders) + 
  popcorn_shrimp_price * popcorn_shrimp_orders = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_catfish_dinner_price_l1525_152562


namespace NUMINAMATH_CALUDE_brick_width_is_four_l1525_152597

/-- The surface area of a rectangular prism given its length, width, and height -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a rectangular prism with length 10, height 2, and surface area 136 is 4 -/
theorem brick_width_is_four :
  ∃ w : ℝ, w > 0 ∧ surfaceArea 10 w 2 = 136 → w = 4 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_is_four_l1525_152597


namespace NUMINAMATH_CALUDE_matrix_power_2023_l1525_152554

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 : A ^ 2023 = !![1, 0; 4046, 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l1525_152554


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1525_152507

/-- The area of a rectangular field with one side of 15 m and a diagonal of 18 m -/
theorem rectangular_field_area : 
  ∀ (a b : ℝ), 
  a = 15 → 
  a^2 + b^2 = 18^2 → 
  a * b = 45 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1525_152507


namespace NUMINAMATH_CALUDE_power_of_sum_squares_and_abs_l1525_152564

theorem power_of_sum_squares_and_abs (a b : ℝ) : 
  (a - 4)^2 + |2 - b| = 0 → a^b = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_sum_squares_and_abs_l1525_152564


namespace NUMINAMATH_CALUDE_count_words_with_vowels_l1525_152501

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 7

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 2

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The number of 5-letter words with at least one vowel -/
def words_with_vowels : ℕ := alphabet_size ^ word_length - (alphabet_size - vowel_count) ^ word_length

theorem count_words_with_vowels :
  words_with_vowels = 13682 :=
sorry

end NUMINAMATH_CALUDE_count_words_with_vowels_l1525_152501


namespace NUMINAMATH_CALUDE_influenza_transmission_rate_l1525_152595

theorem influenza_transmission_rate (initial_infected : ℕ) (total_infected : ℕ) : 
  initial_infected = 4 →
  total_infected = 256 →
  ∃ (x : ℕ), 
    x > 0 ∧
    initial_infected + initial_infected * x + (initial_infected + initial_infected * x) * x = total_infected →
    x = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_influenza_transmission_rate_l1525_152595


namespace NUMINAMATH_CALUDE_lcm_of_20_45_60_l1525_152573

theorem lcm_of_20_45_60 : Nat.lcm (Nat.lcm 20 45) 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_60_l1525_152573


namespace NUMINAMATH_CALUDE_distance_difference_l1525_152505

-- Define the travel parameters
def grayson_speed1 : ℝ := 25
def grayson_time1 : ℝ := 1
def grayson_speed2 : ℝ := 20
def grayson_time2 : ℝ := 0.5
def rudy_speed : ℝ := 10
def rudy_time : ℝ := 3

-- Calculate distances
def grayson_distance1 : ℝ := grayson_speed1 * grayson_time1
def grayson_distance2 : ℝ := grayson_speed2 * grayson_time2
def grayson_total_distance : ℝ := grayson_distance1 + grayson_distance2
def rudy_distance : ℝ := rudy_speed * rudy_time

-- Theorem to prove
theorem distance_difference :
  grayson_total_distance - rudy_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1525_152505


namespace NUMINAMATH_CALUDE_two_thousand_second_non_diff_of_squares_non_diff_of_squares_form_eight_thousand_six_form_main_theorem_l1525_152559

/-- A function that returns true if a number is the difference of two squares -/
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, n = x^2 - y^2

/-- A function that returns the nth number of the form 4k + 2 -/
def nth_non_diff_of_squares (n : ℕ) : ℕ :=
  4 * n - 2

/-- Theorem stating that 8006 is the 2002nd positive integer that is not the difference of two squares -/
theorem two_thousand_second_non_diff_of_squares :
  nth_non_diff_of_squares 2002 = 8006 ∧ ¬(is_diff_of_squares 8006) :=
sorry

/-- Theorem stating that numbers of the form 4k + 2 cannot be expressed as the difference of two squares -/
theorem non_diff_of_squares_form (k : ℕ) :
  ¬(is_diff_of_squares (4 * k + 2)) :=
sorry

/-- Theorem stating that 8006 is of the form 4k + 2 -/
theorem eight_thousand_six_form :
  ∃ k : ℕ, 8006 = 4 * k + 2 :=
sorry

/-- Main theorem combining the above results -/
theorem main_theorem :
  ∃ n : ℕ, n = 2002 ∧ nth_non_diff_of_squares n = 8006 ∧ ¬(is_diff_of_squares 8006) :=
sorry

end NUMINAMATH_CALUDE_two_thousand_second_non_diff_of_squares_non_diff_of_squares_form_eight_thousand_six_form_main_theorem_l1525_152559


namespace NUMINAMATH_CALUDE_smallest_k_satisfies_condition_no_smaller_k_satisfies_condition_l1525_152533

/-- The smallest positive real number k satisfying the given condition -/
def smallest_k : ℝ := 4

/-- Predicate to check if a quadratic equation has two distinct real roots -/
def has_distinct_real_roots (p q : ℝ) : Prop :=
  p^2 - 4*q > 0

/-- Predicate to check if four real numbers are distinct -/
def are_distinct (a b c d : ℝ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Main theorem stating that smallest_k satisfies the required condition -/
theorem smallest_k_satisfies_condition :
  ∀ a b c d : ℝ,
  are_distinct a b c d →
  a ≥ smallest_k → b ≥ smallest_k → c ≥ smallest_k → d ≥ smallest_k →
  ∃ p q r s : ℝ,
    ({p, q, r, s} : Set ℝ) = {a, b, c, d} ∧
    has_distinct_real_roots p q ∧
    has_distinct_real_roots r s ∧
    (∀ x : ℝ, (x^2 + p*x + q = 0 ∨ x^2 + r*x + s = 0) →
      (∀ y : ℝ, y ≠ x → (y^2 + p*y + q ≠ 0 ∧ y^2 + r*y + s ≠ 0))) :=
by sorry

/-- Theorem stating that no smaller positive real number than smallest_k satisfies the condition -/
theorem no_smaller_k_satisfies_condition :
  ∀ k : ℝ, 0 < k → k < smallest_k →
  ∃ a b c d : ℝ,
    are_distinct a b c d ∧
    a ≥ k ∧ b ≥ k ∧ c ≥ k ∧ d ≥ k ∧
    (∀ p q r s : ℝ,
      ({p, q, r, s} : Set ℝ) = {a, b, c, d} →
      ¬(has_distinct_real_roots p q ∧
        has_distinct_real_roots r s ∧
        (∀ x : ℝ, (x^2 + p*x + q = 0 ∨ x^2 + r*x + s = 0) →
          (∀ y : ℝ, y ≠ x → (y^2 + p*y + q ≠ 0 ∧ y^2 + r*y + s ≠ 0))))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_satisfies_condition_no_smaller_k_satisfies_condition_l1525_152533


namespace NUMINAMATH_CALUDE_weed_eating_money_l1525_152539

def mowing_money : ℕ := 14
def weeks_lasted : ℕ := 8
def weekly_spending : ℕ := 5

def total_money : ℕ := weeks_lasted * weekly_spending

theorem weed_eating_money :
  total_money - mowing_money = 26 := by sorry

end NUMINAMATH_CALUDE_weed_eating_money_l1525_152539


namespace NUMINAMATH_CALUDE_hexagon_perimeter_value_l1525_152547

/-- The perimeter of a hexagon ABCDEF with given side lengths -/
def hexagon_perimeter (AB BC CD DE EF FA : ℝ) : ℝ :=
  AB + BC + CD + DE + EF + FA

/-- Theorem: The perimeter of hexagon ABCDEF is 7.5 + √3 -/
theorem hexagon_perimeter_value :
  hexagon_perimeter 1 1.5 1.5 1.5 (Real.sqrt 3) 2 = 7.5 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_value_l1525_152547


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1525_152524

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 3) = 1 / 5) : 
  Real.sin α = (2 * Real.sqrt 6 - Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1525_152524


namespace NUMINAMATH_CALUDE_max_area_inscribed_circle_l1525_152538

/-- A quadrilateral with given angles and perimeter -/
structure Quadrilateral where
  angles : Fin 4 → ℝ
  perimeter : ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 2 * Real.pi
  positive_perimeter : perimeter > 0

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- A predicate indicating whether a circle can be inscribed in the quadrilateral -/
def has_inscribed_circle (q : Quadrilateral) : Prop := sorry

/-- The theorem stating that the quadrilateral with an inscribed circle has the largest area -/
theorem max_area_inscribed_circle (q : Quadrilateral) :
  has_inscribed_circle q ↔ ∀ (q' : Quadrilateral), q'.angles = q.angles ∧ q'.perimeter = q.perimeter → area q ≥ area q' :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_circle_l1525_152538


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1525_152511

theorem train_bridge_crossing_time
  (train_length : ℝ)
  (bridge_length : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length = 165)
  (h2 : bridge_length = 660)
  (h3 : train_speed_kmph = 90) :
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 33 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1525_152511


namespace NUMINAMATH_CALUDE_reducible_fraction_implies_divisibility_l1525_152543

theorem reducible_fraction_implies_divisibility 
  (a b c d l k p q : ℤ) 
  (h1 : a * l + b = k * p) 
  (h2 : c * l + d = k * q) : 
  k ∣ (a * d - b * c) := by
  sorry

end NUMINAMATH_CALUDE_reducible_fraction_implies_divisibility_l1525_152543


namespace NUMINAMATH_CALUDE_school_gender_ratio_l1525_152531

theorem school_gender_ratio (boys girls : ℕ) : 
  (boys : ℚ) / girls = 7.5 / 15.4 →
  girls = boys + 174 →
  boys = 165 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l1525_152531


namespace NUMINAMATH_CALUDE_first_player_wins_l1525_152586

/-- Represents a board in the game -/
structure Board :=
  (m : ℕ)

/-- Represents a position on the board -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents a move in the game -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Represents the game state -/
structure GameState :=
  (board : Board)
  (currentPosition : Position)
  (usedSegments : List (Position × Position))

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.Up => state.currentPosition.y < state.board.m
  | Move.Down => state.currentPosition.y > 0
  | Move.Left => state.currentPosition.x > 0
  | Move.Right => state.currentPosition.x < state.board.m - 1

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Represents a winning strategy for the first player -/
def winningStrategy (board : Board) : Prop :=
  ∃ (strategy : List Move),
    ∀ (opponentMoves : List Move),
      let finalState := (strategy ++ opponentMoves).foldl applyMove
        { board := board
        , currentPosition := ⟨0, 0⟩
        , usedSegments := []
        }
      ¬∃ (move : Move), isValidMove finalState move

/-- The main theorem: there exists a winning strategy for the first player -/
theorem first_player_wins (m : ℕ) (h : m > 1) :
  winningStrategy { m := m } :=
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l1525_152586


namespace NUMINAMATH_CALUDE_lilies_count_l1525_152590

def flowers_problem (roses sunflowers daisies total : ℕ) : Prop :=
  roses = 40 ∧ sunflowers = 40 ∧ daisies = 40 ∧ total = 160

theorem lilies_count (roses sunflowers daisies total : ℕ) 
  (h : flowers_problem roses sunflowers daisies total) : 
  total - (roses + sunflowers + daisies) = 40 := by
  sorry

end NUMINAMATH_CALUDE_lilies_count_l1525_152590


namespace NUMINAMATH_CALUDE_square_perimeter_transformation_l1525_152532

-- Define a square type
structure Square where
  perimeter : ℝ

-- Define the transformation function
def transform (s : Square) : Square :=
  { perimeter := 12 * s.perimeter }

-- Theorem statement
theorem square_perimeter_transformation (s : Square) :
  (transform s).perimeter = 12 * s.perimeter := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_transformation_l1525_152532


namespace NUMINAMATH_CALUDE_positive_expressions_l1525_152550

theorem positive_expressions (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + b^2 ∧ 0 < b + 3*b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expressions_l1525_152550


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1525_152522

theorem gcd_of_three_numbers : Nat.gcd 84 (Nat.gcd 294 315) = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1525_152522


namespace NUMINAMATH_CALUDE_irrational_element_exists_l1525_152546

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ (a (n + 1))^2 = a n + 1

theorem irrational_element_exists (a : ℕ → ℝ) (h : sequence_condition a) :
  ∃ i : ℕ, ¬ (∃ q : ℚ, a i = q) :=
sorry

end NUMINAMATH_CALUDE_irrational_element_exists_l1525_152546


namespace NUMINAMATH_CALUDE_divisibility_implication_l1525_152580

theorem divisibility_implication (x y : ℤ) : (2*x + 1) ∣ (8*y) → (2*x + 1) ∣ y := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1525_152580


namespace NUMINAMATH_CALUDE_triangle_side_length_l1525_152545

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  b = Real.sqrt 3 →
  (a / Real.sin A = b / Real.sin B) →
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1525_152545


namespace NUMINAMATH_CALUDE_new_failing_grades_saturday_is_nine_l1525_152557

/-- The number of new failing grades that appear on Saturday -/
def new_failing_grades_saturday : ℕ :=
  let group1_students : ℕ := 7
  let group2_students : ℕ := 9
  let days_mon_to_sat : ℕ := 6
  let failing_grades_mon_to_fri : ℕ := 30
  let group1_failing_grades := group1_students * (days_mon_to_sat / 2)
  let group2_failing_grades := group2_students * (days_mon_to_sat / 3)
  let total_failing_grades := group1_failing_grades + group2_failing_grades
  total_failing_grades - failing_grades_mon_to_fri

theorem new_failing_grades_saturday_is_nine :
  new_failing_grades_saturday = 9 := by
  sorry

end NUMINAMATH_CALUDE_new_failing_grades_saturday_is_nine_l1525_152557


namespace NUMINAMATH_CALUDE_bisecting_line_theorem_l1525_152525

/-- The pentagon vertices -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (11, 0)
def C : ℝ × ℝ := (11, 2)
def D : ℝ × ℝ := (6, 2)
def E : ℝ × ℝ := (0, 8)

/-- The area of the pentagon -/
noncomputable def pentagonArea : ℝ := sorry

/-- The x-coordinate of the bisecting line -/
noncomputable def bisectingLineX : ℝ := 8 - 2 * Real.sqrt 6

/-- The area of the left part of the pentagon when divided by the line x = bisectingLineX -/
noncomputable def leftArea : ℝ := sorry

/-- The area of the right part of the pentagon when divided by the line x = bisectingLineX -/
noncomputable def rightArea : ℝ := sorry

/-- Theorem stating that the line x = 8 - 2√6 bisects the area of the pentagon -/
theorem bisecting_line_theorem : leftArea = rightArea ∧ leftArea + rightArea = pentagonArea := by sorry

end NUMINAMATH_CALUDE_bisecting_line_theorem_l1525_152525


namespace NUMINAMATH_CALUDE_cost_of_four_enchiladas_five_tacos_l1525_152521

/-- The price of an enchilada -/
def enchilada_price : ℝ := sorry

/-- The price of a taco -/
def taco_price : ℝ := sorry

/-- The first condition: 5 enchiladas and 2 tacos cost $4.30 -/
axiom condition1 : 5 * enchilada_price + 2 * taco_price = 4.30

/-- The second condition: 4 enchiladas and 3 tacos cost $4.50 -/
axiom condition2 : 4 * enchilada_price + 3 * taco_price = 4.50

/-- The theorem to prove -/
theorem cost_of_four_enchiladas_five_tacos :
  4 * enchilada_price + 5 * taco_price = 6.01 := by sorry

end NUMINAMATH_CALUDE_cost_of_four_enchiladas_five_tacos_l1525_152521


namespace NUMINAMATH_CALUDE_skate_cost_is_65_l1525_152576

/-- The cost of renting skates for one visit -/
def rental_cost : ℚ := 2.5

/-- The number of visits needed to justify buying skates -/
def visits : ℕ := 26

/-- The cost of a new pair of skates -/
def skate_cost : ℚ := rental_cost * visits

/-- Theorem stating that the cost of a new pair of skates is $65 -/
theorem skate_cost_is_65 : skate_cost = 65 := by sorry

end NUMINAMATH_CALUDE_skate_cost_is_65_l1525_152576


namespace NUMINAMATH_CALUDE_tan_alpha_eq_neg_one_l1525_152584

theorem tan_alpha_eq_neg_one (α : ℝ) (h : Real.sin (π/6 - α) = Real.cos (π/6 + α)) : 
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_neg_one_l1525_152584


namespace NUMINAMATH_CALUDE_trig_value_comparison_l1525_152558

theorem trig_value_comparison :
  let a : ℝ := Real.tan (-7 * π / 6)
  let b : ℝ := Real.cos (23 * π / 4)
  let c : ℝ := Real.sin (-33 * π / 4)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_trig_value_comparison_l1525_152558


namespace NUMINAMATH_CALUDE_complex_simplification_l1525_152520

/-- Given that i is the imaginary unit, prove that (1+3i)/(1+i) = 2+i -/
theorem complex_simplification (i : ℂ) (hi : i^2 = -1) :
  (1 + 3*i) / (1 + i) = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l1525_152520


namespace NUMINAMATH_CALUDE_triangle_inequality_l1525_152582

/-- Prove that for positive integers x, y, z and angles α, β, γ in [0, π) where any two angles 
    sum to more than the third, the following inequality holds:
    √(x²+y²-2xy cos α) + √(y²+z²-2yz cos β) ≥ √(z²+x²-2zx cos γ) -/
theorem triangle_inequality (x y z : ℕ+) (α β γ : ℝ)
  (h_α : 0 ≤ α ∧ α < π)
  (h_β : 0 ≤ β ∧ β < π)
  (h_γ : 0 ≤ γ ∧ γ < π)
  (h_sum1 : α + β > γ)
  (h_sum2 : β + γ > α)
  (h_sum3 : γ + α > β) :
  Real.sqrt (x.val^2 + y.val^2 - 2*x.val*y.val*(Real.cos α)) + 
  Real.sqrt (y.val^2 + z.val^2 - 2*y.val*z.val*(Real.cos β)) ≥
  Real.sqrt (z.val^2 + x.val^2 - 2*z.val*x.val*(Real.cos γ)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1525_152582


namespace NUMINAMATH_CALUDE_function_periodicity_l1525_152518

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_periodicity (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = f (2 - x))
  (h2 : ∀ x, f (x + 7) = f (7 - x)) :
  is_periodic f 10 := by
sorry

end NUMINAMATH_CALUDE_function_periodicity_l1525_152518


namespace NUMINAMATH_CALUDE_part1_part2_l1525_152528

/-- Definition of the function f -/
def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

/-- Theorem for part 1 -/
theorem part1 (a : ℝ) : 
  f a 1 > 0 ↔ (3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3) := by sorry

/-- Theorem for part 2 -/
theorem part2 (a b : ℝ) : 
  (∀ x, f a x > b ↔ -1 < x ∧ x < 3) → 
  ((a = 3 - Real.sqrt 3 ∨ a = 3 + Real.sqrt 3) ∧ b = -3) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1525_152528


namespace NUMINAMATH_CALUDE_not_special_2013_l1525_152574

/-- A year is special if there exists a month and day such that their product
    equals the last two digits of the year. -/
def is_special_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

/-- The last two digits of 2013. -/
def last_two_digits_2013 : ℕ := 13

/-- Theorem stating that 2013 is not a special year. -/
theorem not_special_2013 : ¬(is_special_year 2013) := by
  sorry

end NUMINAMATH_CALUDE_not_special_2013_l1525_152574


namespace NUMINAMATH_CALUDE_g_of_3_eq_3_l1525_152500

/-- The function g is defined as g(x) = x^2 - 2x for all real x. -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- Theorem: The value of g(3) is 3. -/
theorem g_of_3_eq_3 : g 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_3_l1525_152500


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l1525_152506

theorem marble_fraction_after_tripling (total : ℚ) (h_pos : total > 0) : 
  let green := (4/7) * total
  let blue := (1/7) * total
  let initial_white := total - green - blue
  let new_white := 3 * initial_white
  let new_total := green + blue + new_white
  new_white / new_total = 6/11 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l1525_152506


namespace NUMINAMATH_CALUDE_log_exponent_sum_l1525_152512

theorem log_exponent_sum (a : ℝ) (h : a = Real.log 3 / Real.log 4) : 
  2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_log_exponent_sum_l1525_152512


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l1525_152589

theorem simplify_sqrt_difference : 
  Real.sqrt 300 / Real.sqrt 75 - Real.sqrt 220 / Real.sqrt 55 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l1525_152589


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1525_152578

theorem geometric_sequence_sum (a : ℕ) : 
  let seq := [a, 2*a, 4*a, 8*a, 16*a, 32*a]
  ∀ (x y z w : ℕ), x ∈ seq → y ∈ seq → z ∈ seq → w ∈ seq →
  x ≠ y ∧ z ≠ w ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w →
  x + y = 136 →
  z + w = 272 →
  ∃ (p q : ℕ), p ∈ seq ∧ q ∈ seq ∧ p ≠ q ∧ p + q = 96 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1525_152578


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l1525_152592

theorem quadratic_form_h_value : ∃ (a k : ℝ), ∀ x : ℝ, 
  3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l1525_152592


namespace NUMINAMATH_CALUDE_f_properties_l1525_152513

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi ∧
  (∀ (x y : ℝ), x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) →
    y ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1525_152513


namespace NUMINAMATH_CALUDE_function_identity_proof_l1525_152523

theorem function_identity_proof (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, (n - 1)^2 < f n * f (f n) ∧ f n * f (f n) < n^2 + n) : 
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_proof_l1525_152523


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l1525_152514

def U : Set ℕ := {x | x^2 - 4*x - 5 ≤ 0}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3, 5}

theorem intersection_complement_equals_set : A ∩ (U \ B) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l1525_152514


namespace NUMINAMATH_CALUDE_three_quarters_of_fifteen_fifths_minus_half_l1525_152509

theorem three_quarters_of_fifteen_fifths_minus_half (x : ℚ) : x = (3 / 4) * (15 / 5) - (1 / 2) → x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_three_quarters_of_fifteen_fifths_minus_half_l1525_152509


namespace NUMINAMATH_CALUDE_power_of_1024_is_16_l1525_152577

theorem power_of_1024_is_16 :
  (1024 : ℝ) ^ (2/5 : ℝ) = 16 :=
by
  have h : 1024 = 2^10 := by norm_num
  sorry

end NUMINAMATH_CALUDE_power_of_1024_is_16_l1525_152577


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1525_152599

/-- Proves that a train with given length crossing a bridge with given length in a given time has a specific speed. -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 275 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1525_152599


namespace NUMINAMATH_CALUDE_triangle_third_side_l1525_152583

theorem triangle_third_side (a b : ℝ) (n : ℕ) : 
  a = 3.14 → b = 0.67 → 
  (n : ℝ) + b > a ∧ (n : ℝ) + a > b ∧ a + b > (n : ℝ) →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l1525_152583


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l1525_152587

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2) * (z + 2)^2) ≤ 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l1525_152587


namespace NUMINAMATH_CALUDE_tommy_has_100_nickels_l1525_152504

/-- Represents Tommy's coin collection --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ
  dollar_coins : ℕ

/-- Calculates the total value of the coin collection in cents --/
def total_value (c : CoinCollection) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes + 25 * c.quarters + 50 * c.half_dollars + 100 * c.dollar_coins

/-- Tommy's coin collection satisfies the given conditions --/
def tommy_collection (c : CoinCollection) : Prop :=
  c.dimes = c.pennies + 10 ∧
  c.nickels = 2 * c.dimes ∧
  c.quarters = 4 ∧
  c.pennies = 10 * c.quarters ∧
  c.half_dollars = c.quarters + 5 ∧
  c.dollar_coins = 3 * c.half_dollars ∧
  total_value c = 2000

theorem tommy_has_100_nickels :
  ∀ c : CoinCollection, tommy_collection c → c.nickels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommy_has_100_nickels_l1525_152504


namespace NUMINAMATH_CALUDE_vacation_cost_splitting_l1525_152555

/-- Prove that the difference between what Tom and Dorothy owe Sammy is 20 dollars -/
theorem vacation_cost_splitting (tom_paid dorothy_paid sammy_paid t d : ℚ) : 
  tom_paid = 105 →
  dorothy_paid = 125 →
  sammy_paid = 175 →
  (tom_paid + dorothy_paid + sammy_paid) / 3 = tom_paid + t →
  (tom_paid + dorothy_paid + sammy_paid) / 3 = dorothy_paid + d →
  t - d = 20 := by
sorry


end NUMINAMATH_CALUDE_vacation_cost_splitting_l1525_152555


namespace NUMINAMATH_CALUDE_shopping_mall_problem_l1525_152563

/-- Shopping mall product purchase problem -/
theorem shopping_mall_problem 
  (cost_price_A cost_price_B : ℚ)
  (quantity_A quantity_B : ℕ)
  (selling_price_A selling_price_B : ℚ) :
  cost_price_A = cost_price_B - 2 →
  80 / cost_price_A = 100 / cost_price_B →
  quantity_A = 3 * quantity_B - 5 →
  quantity_A + quantity_B ≤ 95 →
  selling_price_A = 12 →
  selling_price_B = 15 →
  (selling_price_A - cost_price_A) * quantity_A + 
  (selling_price_B - cost_price_B) * quantity_B > 380 →
  (cost_price_A = 8 ∧ cost_price_B = 10) ∧
  (∀ n : ℕ, n ≤ quantity_B → n ≤ 25) ∧
  ((quantity_A = 67 ∧ quantity_B = 24) ∨ 
   (quantity_A = 70 ∧ quantity_B = 25)) := by
sorry


end NUMINAMATH_CALUDE_shopping_mall_problem_l1525_152563
