import Mathlib

namespace NUMINAMATH_CALUDE_cross_product_equals_l2618_261893

def vector1 : ℝ × ℝ × ℝ := (3, -4, 5)
def vector2 : ℝ × ℝ × ℝ := (-2, 7, 1)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := v1
  let (d, e, f) := v2
  (b * f - c * e, c * d - a * f, a * e - b * d)

theorem cross_product_equals : cross_product vector1 vector2 = (-39, -13, 13) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_equals_l2618_261893


namespace NUMINAMATH_CALUDE_myrtle_eggs_theorem_l2618_261820

/-- The number of eggs Myrtle has after all collections and drops -/
def eggs_remaining (num_hens : ℕ) (days_gone : ℕ) (neighbor_took : ℕ) (dropped_eggs : List ℕ)
  (daily_eggs : List ℕ) : ℕ :=
  let total_eggs := (List.sum daily_eggs) * days_gone
  let remaining_after_neighbor := total_eggs - neighbor_took
  remaining_after_neighbor - (List.sum dropped_eggs)

/-- Theorem stating the number of eggs Myrtle has after all collections and drops -/
theorem myrtle_eggs_theorem : 
  eggs_remaining 5 12 32 [3, 5, 2] [3, 4, 2, 5, 3] = 162 := by
  sorry

#eval eggs_remaining 5 12 32 [3, 5, 2] [3, 4, 2, 5, 3]

end NUMINAMATH_CALUDE_myrtle_eggs_theorem_l2618_261820


namespace NUMINAMATH_CALUDE_guard_distance_proof_l2618_261861

/-- Calculates the total distance walked by a guard around a rectangular warehouse -/
def total_distance_walked (length width : ℕ) (total_circles skipped_circles : ℕ) : ℕ :=
  2 * (length + width) * (total_circles - skipped_circles)

/-- Proves that the guard walks 16000 feet given the specific conditions -/
theorem guard_distance_proof :
  total_distance_walked 600 400 10 2 = 16000 := by
  sorry

end NUMINAMATH_CALUDE_guard_distance_proof_l2618_261861


namespace NUMINAMATH_CALUDE_m_range_for_g_l2618_261803

/-- Definition of an (a, b) type function -/
def is_ab_type_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) * f (a - x) = b

/-- Definition of the function g -/
def g (m : ℝ) (x : ℝ) : ℝ :=
  x^2 - m * (x - 1) + 1

/-- Main theorem -/
theorem m_range_for_g :
  ∀ m : ℝ,
  (m > 0) →
  (is_ab_type_function (g m) 1 4) →
  (∀ x ∈ Set.Icc 0 2, 1 ≤ g m x ∧ g m x ≤ 3) →
  (2 - 2 * Real.sqrt 6 / 3 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_m_range_for_g_l2618_261803


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l2618_261845

theorem power_fraction_simplification : (3^2040 + 3^2038) / (3^2040 - 3^2038) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l2618_261845


namespace NUMINAMATH_CALUDE_johns_remaining_budget_l2618_261877

/-- Calculates the remaining budget after a purchase -/
def remaining_budget (initial : ℚ) (spent : ℚ) : ℚ :=
  initial - spent

/-- Proves that given an initial budget of $999.00 and a purchase of $165.00, the remaining amount is $834.00 -/
theorem johns_remaining_budget :
  remaining_budget 999 165 = 834 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_budget_l2618_261877


namespace NUMINAMATH_CALUDE_shopping_money_left_l2618_261881

theorem shopping_money_left (initial_amount : ℝ) (final_amount : ℝ) 
  (spent_percentage : ℝ) (h1 : initial_amount = 4000) 
  (h2 : final_amount = 2800) (h3 : spent_percentage = 0.3) : 
  initial_amount * (1 - spent_percentage) = final_amount := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_left_l2618_261881


namespace NUMINAMATH_CALUDE_planes_lines_false_implications_l2618_261805

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

theorem planes_lines_false_implications 
  (α β : Plane) (l m : Line) :
  ∃ (α β : Plane) (l m : Line),
    α ≠ β ∧ l ≠ m ∧
    subset l α ∧ subset m β ∧
    ¬(¬(parallel α β) → ¬(line_parallel l m)) ∧
    ¬(perpendicular l m → plane_perpendicular α β) := by
  sorry

end NUMINAMATH_CALUDE_planes_lines_false_implications_l2618_261805


namespace NUMINAMATH_CALUDE_teena_current_distance_l2618_261853

/-- Represents the current situation and future state of two drivers on a road -/
structure DrivingSituation where
  teena_speed : ℝ  -- Teena's speed in miles per hour
  poe_speed : ℝ    -- Poe's speed in miles per hour
  time : ℝ          -- Time in hours
  future_distance : ℝ  -- Distance Teena will be ahead of Poe after the given time

/-- Calculates the current distance between two drivers given their future state -/
def current_distance (s : DrivingSituation) : ℝ :=
  ((s.teena_speed - s.poe_speed) * s.time) - s.future_distance

/-- Theorem stating that Teena is currently 7.5 miles behind Poe -/
theorem teena_current_distance :
  let s : DrivingSituation := {
    teena_speed := 55,
    poe_speed := 40,
    time := 1.5,  -- 90 minutes = 1.5 hours
    future_distance := 15
  }
  current_distance s = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_teena_current_distance_l2618_261853


namespace NUMINAMATH_CALUDE_bus_capacity_l2618_261816

theorem bus_capacity :
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let regular_seat_capacity : ℕ := 3
  let back_seat_capacity : ℕ := 12
  let total_regular_seats : ℕ := left_seats + right_seats
  let regular_seats_capacity : ℕ := total_regular_seats * regular_seat_capacity
  let total_capacity : ℕ := regular_seats_capacity + back_seat_capacity
  total_capacity = 93 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_l2618_261816


namespace NUMINAMATH_CALUDE_car_efficiency_problem_l2618_261814

/-- The combined fuel efficiency of two cars -/
def combined_efficiency (e1 e2 : ℚ) : ℚ :=
  2 / (1 / e1 + 1 / e2)

theorem car_efficiency_problem :
  let ray_efficiency : ℚ := 50
  let tom_efficiency : ℚ := 15
  combined_efficiency ray_efficiency tom_efficiency = 300 / 13 := by
sorry

end NUMINAMATH_CALUDE_car_efficiency_problem_l2618_261814


namespace NUMINAMATH_CALUDE_larger_cuboid_height_l2618_261896

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem larger_cuboid_height :
  let smallerCuboid : CuboidDimensions := ⟨5, 6, 3⟩
  let largerCuboidBase : CuboidDimensions := ⟨18, 15, 2⟩
  let numSmallerCuboids : ℕ := 6
  cuboidVolume largerCuboidBase = numSmallerCuboids * cuboidVolume smallerCuboid := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_height_l2618_261896


namespace NUMINAMATH_CALUDE_bobby_child_jumps_l2618_261835

/-- The number of jumps Bobby can do per minute as an adult -/
def adult_jumps : ℕ := 60

/-- The number of additional jumps Bobby can do as an adult compared to when he was a child -/
def additional_jumps : ℕ := 30

/-- The number of jumps Bobby could do per minute as a child -/
def child_jumps : ℕ := adult_jumps - additional_jumps

theorem bobby_child_jumps : child_jumps = 30 := by sorry

end NUMINAMATH_CALUDE_bobby_child_jumps_l2618_261835


namespace NUMINAMATH_CALUDE_f_monotonicity_and_b_range_l2618_261854

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

def g (b : ℝ) (x : ℝ) : ℝ := x^2 + (2*b + 1)*x - b - 1

def prop_p (a b : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*b

def prop_q (b : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo (-3) (-2) ∧ x₂ ∈ Set.Ioo 0 1 ∧
  g b x₁ = 0 ∧ g b x₂ = 0

theorem f_monotonicity_and_b_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) ∧
  {b : ℝ | (prop_p a b ∨ prop_q b) ∧ ¬(prop_p a b ∧ prop_q b)} =
  Set.Ioo (1/5) (1/2) ∪ Set.Ici (5/7) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_b_range_l2618_261854


namespace NUMINAMATH_CALUDE_max_value_of_function_l2618_261846

theorem max_value_of_function (x : ℝ) (h : x > 0) : 2 - 9*x - 4/x ≤ -10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2618_261846


namespace NUMINAMATH_CALUDE_strawberry_juice_problem_l2618_261843

theorem strawberry_juice_problem (T : ℚ) 
  (h1 : T > 0)
  (h2 : (5/6 * T - 2/5 * (5/6 * T) - 2/3 * (5/6 * T - 2/5 * (5/6 * T))) = 120) : 
  T = 720 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_juice_problem_l2618_261843


namespace NUMINAMATH_CALUDE_xiao_li_commute_l2618_261849

/-- Xiao Li's commute problem -/
theorem xiao_li_commute
  (distance : ℝ)
  (walk_late : ℝ)
  (bike_early : ℝ)
  (bike_speed_ratio : ℝ)
  (bike_breakdown_distance : ℝ)
  (h_distance : distance = 4.5)
  (h_walk_late : walk_late = 5 / 60)
  (h_bike_early : bike_early = 10 / 60)
  (h_bike_speed_ratio : bike_speed_ratio = 1.5)
  (h_bike_breakdown_distance : bike_breakdown_distance = 1.5) :
  ∃ (walk_speed bike_speed min_run_speed : ℝ),
    walk_speed = 6 ∧
    bike_speed = 9 ∧
    min_run_speed = 7.2 ∧
    distance / walk_speed - walk_late = distance / bike_speed + bike_early ∧
    bike_speed = bike_speed_ratio * walk_speed ∧
    bike_breakdown_distance / bike_speed +
      (distance - bike_breakdown_distance) / min_run_speed ≤
        distance / bike_speed + bike_early - 5 / 60 :=
by sorry

end NUMINAMATH_CALUDE_xiao_li_commute_l2618_261849


namespace NUMINAMATH_CALUDE_tshirt_cost_l2618_261859

theorem tshirt_cost (original_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  original_price = 240 ∧ 
  discount_rate = 0.2 ∧ 
  profit_rate = 0.2 →
  ∃ (cost : ℝ), cost = 160 ∧ 
    cost * (1 + profit_rate) = original_price * (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_tshirt_cost_l2618_261859


namespace NUMINAMATH_CALUDE_function_properties_monotonicity_condition_l2618_261866

/-- The function f(x) = ax³ + bx² -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem function_properties (a b : ℝ) :
  (f a b 1 = 4) ∧ 
  (f_derivative a b 1 * 1 = -9) →
  (a = 1 ∧ b = 3) :=
sorry

theorem monotonicity_condition (m : ℝ) :
  (∀ x ∈ Set.Icc m (m + 1), f_derivative 1 3 x > 0) →
  (m ≥ 0 ∨ m ≤ -3) :=
sorry

end NUMINAMATH_CALUDE_function_properties_monotonicity_condition_l2618_261866


namespace NUMINAMATH_CALUDE_square_area_and_diagonal_l2618_261817

/-- Given a square with perimeter 40 feet, prove its area and diagonal length -/
theorem square_area_and_diagonal (perimeter : ℝ) (h : perimeter = 40) :
  let side := perimeter / 4
  (side ^ 2 = 100) ∧ (side * Real.sqrt 2 = 10 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_square_area_and_diagonal_l2618_261817


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fourths_l2618_261824

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fraction_greater_than_three_fourths :
  ∃ (a b : ℕ), 
    is_two_digit a ∧ 
    is_two_digit b ∧ 
    (a : ℚ) / b > 3 / 4 ∧
    (∀ (c d : ℕ), is_two_digit c → is_two_digit d → (c : ℚ) / d > 3 / 4 → a ≤ c) ∧
    a = 73 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fourths_l2618_261824


namespace NUMINAMATH_CALUDE_fraction_equality_l2618_261875

theorem fraction_equality (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2618_261875


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2618_261895

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x - y) + 4 * x * y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x^2 + c :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2618_261895


namespace NUMINAMATH_CALUDE_line_through_points_l2618_261888

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_through_points : 
  let p1 : Point := ⟨2, 5⟩
  let p2 : Point := ⟨6, 17⟩
  let p3 : Point := ⟨10, 29⟩
  let p4 : Point := ⟨34, s⟩
  collinear p1 p2 p3 ∧ collinear p1 p2 p4 → s = 101 := by
  sorry


end NUMINAMATH_CALUDE_line_through_points_l2618_261888


namespace NUMINAMATH_CALUDE_exam_results_l2618_261804

/-- Represents a student in the autonomous recruitment exam -/
structure Student where
  writtenProb : ℝ  -- Probability of passing the written exam
  oralProb : ℝ     -- Probability of passing the oral exam

/-- The autonomous recruitment exam setup -/
def ExamSetup : (Student × Student × Student) :=
  (⟨0.6, 0.5⟩, ⟨0.5, 0.6⟩, ⟨0.4, 0.75⟩)

/-- Calculates the probability of exactly one student passing the written exam -/
noncomputable def probExactlyOnePassWritten (setup : Student × Student × Student) : ℝ :=
  sorry

/-- Calculates the expected number of pre-admitted students -/
noncomputable def expectedPreAdmitted (setup : Student × Student × Student) : ℝ :=
  sorry

/-- Main theorem stating the results of the calculations -/
theorem exam_results :
  let setup := ExamSetup
  probExactlyOnePassWritten setup = 0.38 ∧
  expectedPreAdmitted setup = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l2618_261804


namespace NUMINAMATH_CALUDE_four_number_sequence_l2618_261882

/-- Given four real numbers satisfying specific sequence and sum conditions, 
    prove they are one of two specific quadruples -/
theorem four_number_sequence (a b c d : ℝ) 
  (geom_seq : b / c = c / a)  -- a, b, c form a geometric sequence
  (geom_sum : a + b + c = 19)
  (arith_seq : b - c = c - d)  -- b, c, d form an arithmetic sequence
  (arith_sum : b + c + d = 12) :
  ((a, b, c, d) = (25, -10, 4, 18)) ∨ ((a, b, c, d) = (9, 6, 4, 2)) := by
  sorry

end NUMINAMATH_CALUDE_four_number_sequence_l2618_261882


namespace NUMINAMATH_CALUDE_inequality_chain_l2618_261809

theorem inequality_chain (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  9 / (a + b + c) ≤ 2 / (a + b) + 2 / (b + c) + 2 / (c + a) ∧
  2 / (a + b) + 2 / (b + c) + 2 / (c + a) ≤ 1 / a + 1 / b + 1 / c :=
by sorry

end NUMINAMATH_CALUDE_inequality_chain_l2618_261809


namespace NUMINAMATH_CALUDE_tower_of_hanoi_minimum_moves_five_disks_minimum_moves_l2618_261891

/-- Minimum number of moves required to solve the Tower of Hanoi puzzle with n disks -/
def tower_of_hanoi_moves (n : ℕ) : ℕ := 2^n - 1

/-- The number of disks in our specific problem -/
def num_disks : ℕ := 5

theorem tower_of_hanoi_minimum_moves :
  ∀ n : ℕ, tower_of_hanoi_moves n = 2^n - 1 :=
sorry

theorem five_disks_minimum_moves :
  tower_of_hanoi_moves num_disks = 31 :=
sorry

end NUMINAMATH_CALUDE_tower_of_hanoi_minimum_moves_five_disks_minimum_moves_l2618_261891


namespace NUMINAMATH_CALUDE_total_rectangles_is_176_l2618_261839

/-- The number of gray cells in the picture -/
def total_gray_cells : ℕ := 40

/-- The number of "blue" cells (a subset of gray cells) -/
def blue_cells : ℕ := 36

/-- The number of "red" cells (the remaining subset of gray cells) -/
def red_cells : ℕ := total_gray_cells - blue_cells

/-- The number of unique rectangles containing each blue cell -/
def rectangles_per_blue_cell : ℕ := 4

/-- The number of unique rectangles containing each red cell -/
def rectangles_per_red_cell : ℕ := 8

/-- The total number of checkered rectangles containing exactly one gray cell -/
def total_rectangles : ℕ := blue_cells * rectangles_per_blue_cell + red_cells * rectangles_per_red_cell

theorem total_rectangles_is_176 : total_rectangles = 176 := by
  sorry

end NUMINAMATH_CALUDE_total_rectangles_is_176_l2618_261839


namespace NUMINAMATH_CALUDE_passengers_at_third_station_l2618_261840

/-- Calculates the number of passengers at the third station given the initial number of passengers and the changes at each station. -/
def passengersAtThirdStation (initialPassengers : ℕ) : ℕ :=
  let afterFirstDrop := initialPassengers - initialPassengers / 3
  let afterFirstAdd := afterFirstDrop + 280
  let afterSecondDrop := afterFirstAdd - afterFirstAdd / 2
  afterSecondDrop + 12

/-- Theorem stating that given 270 initial passengers, the number of passengers at the third station is 242. -/
theorem passengers_at_third_station :
  passengersAtThirdStation 270 = 242 := by
  sorry

#eval passengersAtThirdStation 270

end NUMINAMATH_CALUDE_passengers_at_third_station_l2618_261840


namespace NUMINAMATH_CALUDE_units_digit_of_65_plus_37_in_octal_l2618_261880

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNumber :=
  sorry

/-- Adds two octal numbers --/
def octalAdd (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Gets the units digit of an octal number --/
def unitsDigit (n : OctalNumber) : ℕ :=
  sorry

/-- Theorem: The units digit of 65₈ + 37₈ in base 8 is 4 --/
theorem units_digit_of_65_plus_37_in_octal :
  unitsDigit (octalAdd (toOctal 65) (toOctal 37)) = 4 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_65_plus_37_in_octal_l2618_261880


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2618_261836

theorem fraction_sum_equality : (7 : ℚ) / 10 + (3 : ℚ) / 100 + (9 : ℚ) / 1000 = 739 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2618_261836


namespace NUMINAMATH_CALUDE_students_walking_home_l2618_261892

theorem students_walking_home (bus_fraction car_fraction scooter_fraction : ℚ)
  (h1 : bus_fraction = 2/5)
  (h2 : car_fraction = 1/5)
  (h3 : scooter_fraction = 1/8)
  (h4 : bus_fraction + car_fraction + scooter_fraction < 1) :
  1 - (bus_fraction + car_fraction + scooter_fraction) = 11/40 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l2618_261892


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_3780_l2618_261807

/-- The largest perfect square factor of a natural number -/
def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The largest perfect square factor of 3780 is 36 -/
theorem largest_perfect_square_factor_of_3780 :
  largest_perfect_square_factor 3780 = 36 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_3780_l2618_261807


namespace NUMINAMATH_CALUDE_inequality_proof_l2618_261830

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2618_261830


namespace NUMINAMATH_CALUDE_shiela_paintings_l2618_261897

/-- The number of paintings Shiela can give to each grandmother -/
def paintings_per_grandmother : ℕ := 9

/-- The number of grandmothers Shiela has -/
def number_of_grandmothers : ℕ := 2

/-- The total number of paintings Shiela has -/
def total_paintings : ℕ := paintings_per_grandmother * number_of_grandmothers

theorem shiela_paintings : total_paintings = 18 := by
  sorry

end NUMINAMATH_CALUDE_shiela_paintings_l2618_261897


namespace NUMINAMATH_CALUDE_worker_y_fraction_l2618_261870

theorem worker_y_fraction (total_products : ℝ) (x_products y_products : ℝ) 
  (h1 : x_products + y_products = total_products)
  (h2 : 0.005 * x_products + 0.008 * y_products = 0.007 * total_products) :
  y_products / total_products = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_worker_y_fraction_l2618_261870


namespace NUMINAMATH_CALUDE_problem_statement_l2618_261874

theorem problem_statement (a b c : ℝ) (h : (2:ℝ)^a = (3:ℝ)^b ∧ (3:ℝ)^b = (18:ℝ)^c ∧ (18:ℝ)^c < 1) :
  b < 2*c ∧ (a + b)/c > 3 + 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2618_261874


namespace NUMINAMATH_CALUDE_quadratic_equation_propositions_l2618_261862

theorem quadratic_equation_propositions (a b : ℝ) : 
  ∃! (prop_a prop_b prop_c prop_d : Prop),
    prop_a = (1 ^ 2 + a * 1 + b = 0) ∧
    prop_b = (∃ x y : ℝ, x ^ 2 + a * x + b = 0 ∧ y ^ 2 + a * y + b = 0 ∧ x + y = 2) ∧
    prop_c = (3 ^ 2 + a * 3 + b = 0) ∧
    prop_d = (∃ x y : ℝ, x ^ 2 + a * x + b = 0 ∧ y ^ 2 + a * y + b = 0 ∧ x * y < 0) ∧
    (¬prop_a ∧ prop_b ∧ prop_c ∧ prop_d) ∨
    (prop_a ∧ ¬prop_b ∧ prop_c ∧ prop_d) ∨
    (prop_a ∧ prop_b ∧ ¬prop_c ∧ prop_d) ∨
    (prop_a ∧ prop_b ∧ prop_c ∧ ¬prop_d) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_propositions_l2618_261862


namespace NUMINAMATH_CALUDE_non_defective_products_percentage_l2618_261868

/-- Represents a machine in the factory -/
structure Machine where
  production_percentage : ℝ
  defective_percentage : ℝ

/-- The factory setup -/
def factory : List Machine := [
  ⟨0.25, 0.02⟩,  -- m1
  ⟨0.35, 0.04⟩,  -- m2
  ⟨0.40, 0.05⟩   -- m3
]

/-- Calculate the percentage of non-defective products -/
def non_defective_percentage (machines : List Machine) : ℝ :=
  1 - (machines.map (λ m => m.production_percentage * m.defective_percentage)).sum

/-- Theorem stating the percentage of non-defective products -/
theorem non_defective_products_percentage :
  non_defective_percentage factory = 0.961 := by
  sorry

#eval non_defective_percentage factory

end NUMINAMATH_CALUDE_non_defective_products_percentage_l2618_261868


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2618_261855

theorem sqrt_equation_solution (w : ℝ) :
  (Real.sqrt 1.1 / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt w = 2.879628878919216) →
  w = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2618_261855


namespace NUMINAMATH_CALUDE_distance_between_walkers_l2618_261889

/-- Proves the distance between two people walking towards each other after a given time -/
theorem distance_between_walkers 
  (playground_length : ℝ) 
  (speed_hyosung : ℝ) 
  (speed_mimi : ℝ) 
  (time : ℝ) 
  (h1 : playground_length = 2.5)
  (h2 : speed_hyosung = 0.08)
  (h3 : speed_mimi = 2.4 / 60)
  (h4 : time = 15) :
  playground_length - (speed_hyosung + speed_mimi) * time = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_walkers_l2618_261889


namespace NUMINAMATH_CALUDE_tamil_speakers_l2618_261815

theorem tamil_speakers (total_population : ℕ) (english_speakers : ℕ) (both_speakers : ℕ) (hindi_probability : ℚ) : 
  total_population = 1024 →
  english_speakers = 562 →
  both_speakers = 346 →
  hindi_probability = 0.0859375 →
  ∃ tamil_speakers : ℕ, tamil_speakers = 720 ∧ 
    tamil_speakers = total_population - (english_speakers + (total_population * hindi_probability).floor - both_speakers) :=
by
  sorry

end NUMINAMATH_CALUDE_tamil_speakers_l2618_261815


namespace NUMINAMATH_CALUDE_fertilizer_prices_l2618_261865

/-- Represents the price per ton of fertilizer A -/
def price_A : ℝ := sorry

/-- Represents the price per ton of fertilizer B -/
def price_B : ℝ := sorry

/-- The price difference between fertilizer A and B is $100 -/
axiom price_difference : price_A = price_B + 100

/-- The total cost of 2 tons of fertilizer A and 1 ton of fertilizer B is $1700 -/
axiom total_cost : 2 * price_A + price_B = 1700

theorem fertilizer_prices :
  price_A = 600 ∧ price_B = 500 := by sorry

end NUMINAMATH_CALUDE_fertilizer_prices_l2618_261865


namespace NUMINAMATH_CALUDE_ratio_difference_l2618_261818

theorem ratio_difference (a b c : ℝ) : 
  a / 3 = b / 5 ∧ b / 5 = c / 7 ∧ c = 56 → c - a = 32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_l2618_261818


namespace NUMINAMATH_CALUDE_nearest_integer_to_cube_root_five_sixth_power_l2618_261802

theorem nearest_integer_to_cube_root_five_sixth_power :
  ∃ (n : ℕ), n = 74608 ∧ ∀ (m : ℕ), |((3 : ℝ) + Real.sqrt 5)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_cube_root_five_sixth_power_l2618_261802


namespace NUMINAMATH_CALUDE_vector_addition_l2618_261890

theorem vector_addition (a b : Fin 2 → ℝ) 
  (ha : a = ![2, 1]) 
  (hb : b = ![1, 3]) : 
  a + b = ![3, 4] := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l2618_261890


namespace NUMINAMATH_CALUDE_no_lower_bound_l2618_261869

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence :=
  (a₁ : ℝ)
  (d : ℝ)

/-- The second term of the arithmetic sequence -/
def ArithmeticSequence.a₂ (seq : ArithmeticSequence) : ℝ := seq.a₁ + seq.d

/-- The third term of the arithmetic sequence -/
def ArithmeticSequence.a₃ (seq : ArithmeticSequence) : ℝ := seq.a₁ + 2 * seq.d

/-- The expression to be minimized -/
def expression (seq : ArithmeticSequence) : ℝ := 3 * seq.a₂ + 7 * seq.a₃

/-- The theorem stating that the expression has no lower bound -/
theorem no_lower_bound :
  ∀ (b : ℝ), ∃ (seq : ArithmeticSequence), seq.a₁ = 3 ∧ expression seq < b :=
sorry

end NUMINAMATH_CALUDE_no_lower_bound_l2618_261869


namespace NUMINAMATH_CALUDE_product_integer_part_l2618_261832

theorem product_integer_part : 
  ⌊(1.1 : ℝ) * 1.2 * 1.3 * 1.4 * 1.5 * 1.6⌋ = 1 := by sorry

end NUMINAMATH_CALUDE_product_integer_part_l2618_261832


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l2618_261884

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- The first line: y = ax - 2 -/
def line1 (a x : ℝ) : ℝ := a * x - 2

/-- The second line: y = (2-a)x + 1 -/
def line2 (a x : ℝ) : ℝ := (2 - a) * x + 1

theorem parallel_lines_imply_a_equals_one (a : ℝ) :
  parallel_lines a (2 - a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l2618_261884


namespace NUMINAMATH_CALUDE_child_b_share_child_b_share_is_552_l2618_261899

/-- Calculates the share of child B given the total amount, tax rate, interest rate, and distribution ratio. -/
theorem child_b_share (total_amount : ℝ) (tax_rate : ℝ) (interest_rate : ℝ) (ratio_a ratio_b ratio_c : ℕ) : ℝ :=
  let tax := total_amount * tax_rate
  let interest := total_amount * interest_rate
  let remaining_amount := total_amount - (tax + interest)
  let total_parts := ratio_a + ratio_b + ratio_c
  let part_value := remaining_amount / total_parts
  ratio_b * part_value

/-- Proves that given the specific conditions, B's share is $552. -/
theorem child_b_share_is_552 : 
  child_b_share 1800 0.05 0.03 2 3 4 = 552 := by
  sorry

end NUMINAMATH_CALUDE_child_b_share_child_b_share_is_552_l2618_261899


namespace NUMINAMATH_CALUDE_sum_first_two_terms_l2618_261856

/-- A geometric sequence with third term 12 and fourth term 18 -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), q ≠ 0 ∧ (∀ n, a (n + 1) = a n * q) ∧ a 3 = 12 ∧ a 4 = 18

/-- The sum of the first and second terms of the geometric sequence is 40/3 -/
theorem sum_first_two_terms (a : ℕ → ℚ) (h : GeometricSequence a) :
  a 1 + a 2 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_two_terms_l2618_261856


namespace NUMINAMATH_CALUDE_compute_d_l2618_261860

-- Define the polynomial
def f (c d : ℚ) (x : ℝ) : ℝ := x^3 + c*x^2 + d*x - 36

-- State the theorem
theorem compute_d (c : ℚ) :
  ∃ d : ℚ, f c d (3 + Real.sqrt 2) = 0 → d = -23 - 6/7 :=
by sorry

end NUMINAMATH_CALUDE_compute_d_l2618_261860


namespace NUMINAMATH_CALUDE_son_age_l2618_261842

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 22 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end NUMINAMATH_CALUDE_son_age_l2618_261842


namespace NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l2618_261833

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 2 * a / x

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → 0 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l2618_261833


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_isosceles_triangle_area_proof_l2618_261822

/-- The area of an isosceles triangle with two sides of length 13 and a base of 10 is 60 -/
theorem isosceles_triangle_area : ℝ → Prop :=
  fun area =>
    ∃ (x y z : ℝ),
      x = 13 ∧ y = 13 ∧ z = 10 ∧  -- Two sides are 13, base is 10
      x = y ∧                     -- Isosceles condition
      area = (z * (x ^ 2 - (z / 2) ^ 2).sqrt) / 2 ∧  -- Area formula
      area = 60

/-- Proof of the theorem -/
theorem isosceles_triangle_area_proof : isosceles_triangle_area 60 := by
  sorry

#check isosceles_triangle_area_proof

end NUMINAMATH_CALUDE_isosceles_triangle_area_isosceles_triangle_area_proof_l2618_261822


namespace NUMINAMATH_CALUDE_total_smoothie_ingredients_l2618_261838

def strawberries : ℝ := 0.2
def yogurt : ℝ := 0.1
def orange_juice : ℝ := 0.2
def spinach : ℝ := 0.15
def protein_powder : ℝ := 0.05

theorem total_smoothie_ingredients :
  strawberries + yogurt + orange_juice + spinach + protein_powder = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_total_smoothie_ingredients_l2618_261838


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l2618_261851

theorem mary_baseball_cards 
  (promised_to_fred : ℝ) 
  (bought : ℝ) 
  (left_after_giving : ℝ) 
  (h1 : promised_to_fred = 26.0)
  (h2 : bought = 40.0)
  (h3 : left_after_giving = 32.0) :
  ∃ initial : ℝ, initial = 18.0 ∧ 
    (initial + bought - promised_to_fred = left_after_giving) :=
by sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l2618_261851


namespace NUMINAMATH_CALUDE_system_solution_fractional_equation_solution_l2618_261831

-- System of equations
theorem system_solution :
  ∃ (x y : ℚ), 3 * x - 5 * y = 3 ∧ x / 2 - y / 3 = 1 ∧ x = 8 / 3 ∧ y = 1 := by sorry

-- Fractional equation
theorem fractional_equation_solution :
  ∃ (x : ℚ), x ≠ 1 ∧ x / (x - 1) + 1 = 3 / (2 * x - 2) ∧ x = 5 / 4 := by sorry

end NUMINAMATH_CALUDE_system_solution_fractional_equation_solution_l2618_261831


namespace NUMINAMATH_CALUDE_min_variance_of_sample_l2618_261823

theorem min_variance_of_sample (x y : ℝ) : 
  (x + 1 + y + 5) / 4 = 2 → 
  ((x - 2)^2 + (1 - 2)^2 + (y - 2)^2 + (5 - 2)^2) / 4 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_variance_of_sample_l2618_261823


namespace NUMINAMATH_CALUDE_least_coins_seventeen_coins_least_possible_coins_l2618_261864

theorem least_coins (a : ℕ) : (a % 7 = 3 ∧ a % 4 = 1) → a ≥ 17 := by
  sorry

theorem seventeen_coins : 17 % 7 = 3 ∧ 17 % 4 = 1 := by
  sorry

theorem least_possible_coins : ∃ (a : ℕ), a % 7 = 3 ∧ a % 4 = 1 ∧ ∀ (b : ℕ), (b % 7 = 3 ∧ b % 4 = 1) → a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_least_coins_seventeen_coins_least_possible_coins_l2618_261864


namespace NUMINAMATH_CALUDE_line_graph_most_suitable_for_forest_data_l2618_261826

/-- Represents types of statistical graphs -/
inductive StatisticalGraph
| LineGraph
| BarChart
| PieChart
| ScatterPlot
| Histogram

/-- Represents characteristics of data and analysis requirements -/
structure DataCharacteristics where
  continuous : Bool
  timeSpan : ℕ
  decreasingTrend : Bool

/-- Determines the most suitable graph type for given data characteristics -/
def mostSuitableGraph (data : DataCharacteristics) : StatisticalGraph :=
  sorry

/-- Theorem stating that a line graph is the most suitable for the given forest area data -/
theorem line_graph_most_suitable_for_forest_data :
  let forestData : DataCharacteristics := {
    continuous := true,
    timeSpan := 20,
    decreasingTrend := true
  }
  mostSuitableGraph forestData = StatisticalGraph.LineGraph :=
sorry

end NUMINAMATH_CALUDE_line_graph_most_suitable_for_forest_data_l2618_261826


namespace NUMINAMATH_CALUDE_profit_percentage_theorem_l2618_261850

theorem profit_percentage_theorem (selling_price purchase_price : ℝ) 
  (h1 : selling_price > 0) 
  (h2 : purchase_price > 0) 
  (h3 : selling_price > purchase_price) :
  let original_profit_percentage := (selling_price - purchase_price) / purchase_price * 100
  let new_purchase_price := purchase_price * 0.95
  let new_profit_percentage := (selling_price - new_purchase_price) / new_purchase_price * 100
  (new_profit_percentage - original_profit_percentage = 15) → 
  (original_profit_percentage = 185) := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_theorem_l2618_261850


namespace NUMINAMATH_CALUDE_greatest_n_value_exists_n_value_l2618_261886

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 := by
  sorry

theorem exists_n_value : ∃ (n : ℤ), 101 * n^2 ≤ 12100 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_exists_n_value_l2618_261886


namespace NUMINAMATH_CALUDE_evenAdjacentCellsCount_l2618_261812

/-- The number of cells with an even number of adjacent cells in an equilateral triangle -/
def evenAdjacentCells (sideLength : ℕ) : ℕ :=
  sideLength * sideLength - (sideLength - 3) * (sideLength - 3) - 3

/-- The side length of the large equilateral triangle -/
def largeSideLength : ℕ := 2022

theorem evenAdjacentCellsCount :
  evenAdjacentCells largeSideLength = 12120 := by
  sorry

end NUMINAMATH_CALUDE_evenAdjacentCellsCount_l2618_261812


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l2618_261857

/-- Proves that the percentage increase in rent for one friend is 16% given the conditions of the problem -/
theorem rent_increase_percentage (num_friends : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (initial_rent : ℝ) : 
  num_friends = 4 →
  initial_avg = 800 →
  new_avg = 850 →
  initial_rent = 1250 →
  let total_initial := initial_avg * num_friends
  let new_rent := (new_avg * num_friends) - (total_initial - initial_rent)
  let percentage_increase := (new_rent - initial_rent) / initial_rent * 100
  percentage_increase = 16 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l2618_261857


namespace NUMINAMATH_CALUDE_abc_mod_nine_l2618_261894

theorem abc_mod_nine (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9)
  (h1 : (a + 2*b + 3*c) % 9 = 0)
  (h2 : (2*a + 3*b + c) % 9 = 3)
  (h3 : (3*a + b + 2*c) % 9 = 8) :
  (a * b * c) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_abc_mod_nine_l2618_261894


namespace NUMINAMATH_CALUDE_min_text_length_for_symbol_occurrence_l2618_261837

theorem min_text_length_for_symbol_occurrence : 
  ∃ (x : ℕ), (19 : ℝ) * (21 : ℝ) / 200 < (x : ℝ) ∧ (x : ℝ) < (19 : ℝ) * (11 : ℝ) / 100 ∧
  ∀ (L : ℕ), L < 19 → ¬∃ (y : ℕ), (L : ℝ) * (21 : ℝ) / 200 < (y : ℝ) ∧ (y : ℝ) < (L : ℝ) * (11 : ℝ) / 100 :=
by sorry

end NUMINAMATH_CALUDE_min_text_length_for_symbol_occurrence_l2618_261837


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l2618_261844

theorem polar_to_cartesian (p θ x y : ℝ) :
  (p = 8 * Real.cos θ) ∧ (x = p * Real.cos θ) ∧ (y = p * Real.sin θ) →
  x^2 + y^2 = 8*x :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l2618_261844


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2618_261887

theorem child_ticket_cost (adult_price : ℕ) (num_adults num_children : ℕ) (total_price : ℕ) :
  adult_price = 22 →
  num_adults = 2 →
  num_children = 2 →
  total_price = 58 →
  (total_price - num_adults * adult_price) / num_children = 7 := by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2618_261887


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_l2618_261876

theorem quadratic_roots_opposite (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k-2)*x - 1 = 0 ∧ y^2 + (k-2)*y - 1 = 0 ∧ x = -y) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_opposite_l2618_261876


namespace NUMINAMATH_CALUDE_ladder_length_proof_l2618_261878

/-- The length of a ladder leaning against a wall. -/
def ladder_length : ℝ := 18.027756377319946

/-- The initial distance of the ladder's bottom from the wall. -/
def initial_bottom_distance : ℝ := 6

/-- The distance the ladder's bottom moves when the top slips. -/
def bottom_slip_distance : ℝ := 12.480564970698127

/-- The distance the ladder's top slips down the wall. -/
def top_slip_distance : ℝ := 4

/-- Theorem stating the length of the ladder given the conditions. -/
theorem ladder_length_proof : 
  ∃ (initial_height : ℝ),
    ladder_length ^ 2 = initial_height ^ 2 + initial_bottom_distance ^ 2 ∧
    ladder_length ^ 2 = (initial_height - top_slip_distance) ^ 2 + bottom_slip_distance ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ladder_length_proof_l2618_261878


namespace NUMINAMATH_CALUDE_elberta_money_l2618_261847

theorem elberta_money (granny_smith : ℕ) (elberta anjou : ℕ) : 
  granny_smith = 72 →
  elberta = anjou + 5 →
  anjou = granny_smith / 4 →
  elberta = 23 := by
sorry

end NUMINAMATH_CALUDE_elberta_money_l2618_261847


namespace NUMINAMATH_CALUDE_cheryl_walk_distance_l2618_261834

/-- Calculates the total distance walked by a person who walks at a constant speed
    for a given time in one direction and then returns along the same path. -/
def total_distance_walked (speed : ℝ) (time : ℝ) : ℝ :=
  2 * speed * time

/-- Theorem: Given a person walking at 2 miles per hour for 3 hours in one direction
    and then returning along the same path, the total distance walked is 12 miles. -/
theorem cheryl_walk_distance :
  total_distance_walked 2 3 = 12 := by
  sorry

#eval total_distance_walked 2 3

end NUMINAMATH_CALUDE_cheryl_walk_distance_l2618_261834


namespace NUMINAMATH_CALUDE_fraction_sum_l2618_261872

theorem fraction_sum : (1 : ℚ) / 6 + (2 : ℚ) / 9 + (1 : ℚ) / 3 = (13 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2618_261872


namespace NUMINAMATH_CALUDE_bus_ride_distance_l2618_261858

theorem bus_ride_distance 
  (total_time : ℝ) 
  (bus_speed : ℝ) 
  (walking_speed : ℝ) 
  (h1 : total_time = 8) 
  (h2 : bus_speed = 9) 
  (h3 : walking_speed = 3) : 
  ∃ d : ℝ, d = 18 ∧ d / bus_speed + d / walking_speed = total_time :=
by
  sorry

end NUMINAMATH_CALUDE_bus_ride_distance_l2618_261858


namespace NUMINAMATH_CALUDE_cookie_flour_weight_l2618_261825

/-- Given the conditions of Matt's cookie baking, prove that each bag of flour weighs 5 pounds -/
theorem cookie_flour_weight 
  (cookies_per_batch : ℕ) 
  (flour_per_batch : ℕ) 
  (num_bags : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_left : ℕ) 
  (h1 : cookies_per_batch = 12)
  (h2 : flour_per_batch = 2)
  (h3 : num_bags = 4)
  (h4 : cookies_eaten = 15)
  (h5 : cookies_left = 105) :
  (cookies_eaten + cookies_left) * flour_per_batch / (cookies_per_batch * num_bags) = 5 := by
  sorry

#check cookie_flour_weight

end NUMINAMATH_CALUDE_cookie_flour_weight_l2618_261825


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_product_absolute_value_equation_solution_product_holds_l2618_261821

theorem absolute_value_equation_solution_product : ℝ → Prop :=
  fun x ↦ (|2 * x - 14| - 5 = 1) → 
    ∃ y, (|2 * y - 14| - 5 = 1) ∧ x * y = 40 ∧ 
    ∀ z, (|2 * z - 14| - 5 = 1) → (z = x ∨ z = y)

-- Proof
theorem absolute_value_equation_solution_product_holds :
  ∃ a b : ℝ, absolute_value_equation_solution_product a ∧
             absolute_value_equation_solution_product b ∧
             a ≠ b :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_product_absolute_value_equation_solution_product_holds_l2618_261821


namespace NUMINAMATH_CALUDE_product_equals_sum_implies_y_value_l2618_261829

theorem product_equals_sum_implies_y_value :
  ∀ y : ℚ, (2 * 3 * 5 * y = 2 + 3 + 5 + y) → y = 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_sum_implies_y_value_l2618_261829


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l2618_261800

theorem purely_imaginary_complex (m : ℝ) : 
  (Complex.mk (m^2 - m) m).im ≠ 0 ∧ (Complex.mk (m^2 - m) m).re = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l2618_261800


namespace NUMINAMATH_CALUDE_cabinet_ratio_proof_l2618_261863

/-- Proves the ratio of new cabinets per counter to initial cabinets is 2:1 --/
theorem cabinet_ratio_proof (initial_cabinets : ℕ) (total_cabinets : ℕ) (additional_cabinets : ℕ) 
  (h1 : initial_cabinets = 3)
  (h2 : total_cabinets = 26)
  (h3 : additional_cabinets = 5)
  : ∃ (new_cabinets_per_counter : ℕ), 
    initial_cabinets + 3 * new_cabinets_per_counter + additional_cabinets = total_cabinets ∧ 
    new_cabinets_per_counter = 2 * initial_cabinets :=
by
  sorry


end NUMINAMATH_CALUDE_cabinet_ratio_proof_l2618_261863


namespace NUMINAMATH_CALUDE_inheritance_tax_calculation_l2618_261879

theorem inheritance_tax_calculation (x : ℝ) : 
  (0.2 * x + 0.1 * (x - 0.2 * x) = 10500) → x = 37500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_calculation_l2618_261879


namespace NUMINAMATH_CALUDE_order_of_xyz_l2618_261885

-- Define the variables and their relationships
theorem order_of_xyz (a b c d : ℝ) 
  (h_order : a > b ∧ b > c ∧ c > d ∧ d > 0) 
  (x : ℝ) (hx : x = Real.sqrt (a * b) + Real.sqrt (c * d))
  (y : ℝ) (hy : y = Real.sqrt (a * c) + Real.sqrt (b * d))
  (z : ℝ) (hz : z = Real.sqrt (a * d) + Real.sqrt (b * c)) :
  x > y ∧ y > z :=
by sorry

end NUMINAMATH_CALUDE_order_of_xyz_l2618_261885


namespace NUMINAMATH_CALUDE_billion_difference_value_l2618_261841

/-- Arnaldo's definition of a billion -/
def arnaldo_billion : ℕ := 1000000 * 1000000

/-- Correct definition of a billion -/
def correct_billion : ℕ := 1000 * 1000000

/-- The difference between Arnaldo's definition and the correct definition -/
def billion_difference : ℕ := arnaldo_billion - correct_billion

theorem billion_difference_value : billion_difference = 999000000000 := by
  sorry

end NUMINAMATH_CALUDE_billion_difference_value_l2618_261841


namespace NUMINAMATH_CALUDE_squares_different_areas_l2618_261819

-- Define what a square is
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define properties of squares
def Square.isEquiangular (s : Square) : Prop := true
def Square.isRectangle (s : Square) : Prop := true
def Square.isRegularPolygon (s : Square) : Prop := true
def Square.isSimilarTo (s1 s2 : Square) : Prop := true

-- Define the area of a square
def Square.area (s : Square) : ℝ := s.side * s.side

-- Theorem: There exist squares with different areas
theorem squares_different_areas :
  ∃ (s1 s2 : Square), 
    Square.isEquiangular s1 ∧ 
    Square.isEquiangular s2 ∧
    Square.isRectangle s1 ∧ 
    Square.isRectangle s2 ∧
    Square.isRegularPolygon s1 ∧ 
    Square.isRegularPolygon s2 ∧
    Square.isSimilarTo s1 s2 ∧
    Square.area s1 ≠ Square.area s2 :=
by
  sorry

end NUMINAMATH_CALUDE_squares_different_areas_l2618_261819


namespace NUMINAMATH_CALUDE_solve_equation_l2618_261871

theorem solve_equation (x : ℝ) : 3*x - 4*x + 7*x = 210 → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2618_261871


namespace NUMINAMATH_CALUDE_angle_inequality_l2618_261883

theorem angle_inequality : 
  let a : ℝ := (1/2) * Real.cos (6 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * π / 180)
  let b : ℝ := (2 * Real.tan (13 * π / 180)) / (1 - Real.tan (13 * π / 180)^2)
  let c : ℝ := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_angle_inequality_l2618_261883


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2618_261873

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 6 + x) ↔ x ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2618_261873


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l2618_261828

/-- Proves that the number of meters of cloth sold is 40, given the conditions of the problem -/
theorem cloth_sale_meters : 
  -- C represents the cost price of 1 meter of cloth
  ∀ (C : ℝ), C > 0 →
  -- S represents the selling price of 1 meter of cloth
  ∀ (S : ℝ), S > C →
  -- The gain is the selling price of 10 meters
  let G := 10 * S
  -- The gain percentage is 1/3 (33.33333333333333%)
  let gain_percentage := (1 : ℝ) / 3
  -- M represents the number of meters sold
  ∃ (M : ℝ),
    -- The gain is equal to the gain percentage times the total cost
    G = gain_percentage * (M * C) ∧
    -- The selling price is the cost price plus the gain per meter
    S = C + G / M ∧
    -- The number of meters sold is 40
    M = 40 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_meters_l2618_261828


namespace NUMINAMATH_CALUDE_f_composition_value_l2618_261852

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 5 * x + 4 else 2^x

def angle_terminal_side_point (α : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 / p.1 = Real.tan α

theorem f_composition_value (α : ℝ) :
  angle_terminal_side_point α (4, -3) →
  f (f (Real.sin α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l2618_261852


namespace NUMINAMATH_CALUDE_john_zoo_snakes_l2618_261811

/-- The number of snakes John has in his zoo --/
def num_snakes : ℕ := 15

/-- The total number of animals in John's zoo --/
def total_animals : ℕ := 114

/-- Theorem stating that the number of snakes in John's zoo is correct --/
theorem john_zoo_snakes :
  (num_snakes : ℚ) +
  (2 * num_snakes : ℚ) +
  ((2 * num_snakes : ℚ) - 5) +
  ((2 * num_snakes : ℚ) - 5 + 8) +
  (1/3 * ((2 * num_snakes : ℚ) - 5 + 8)) = total_animals := by
  sorry

#check john_zoo_snakes

end NUMINAMATH_CALUDE_john_zoo_snakes_l2618_261811


namespace NUMINAMATH_CALUDE_garden_border_perimeter_l2618_261808

/-- The total perimeter of Mrs. Hilt's garden border -/
theorem garden_border_perimeter :
  let num_rocks_a : ℝ := 125.0
  let circumference_a : ℝ := 0.5
  let num_rocks_b : ℝ := 64.0
  let circumference_b : ℝ := 0.7
  let total_perimeter : ℝ := num_rocks_a * circumference_a + num_rocks_b * circumference_b
  total_perimeter = 107.3 := by
sorry

end NUMINAMATH_CALUDE_garden_border_perimeter_l2618_261808


namespace NUMINAMATH_CALUDE_two_sarees_four_shirts_cost_l2618_261827

/-- The price of a single saree -/
def saree_price : ℝ := sorry

/-- The price of a single shirt -/
def shirt_price : ℝ := sorry

/-- The cost of 2 sarees and 4 shirts equals the cost of 1 saree and 6 shirts -/
axiom price_equality : 2 * saree_price + 4 * shirt_price = saree_price + 6 * shirt_price

/-- The price of 12 shirts is $2400 -/
axiom twelve_shirts_price : 12 * shirt_price = 2400

/-- The theorem stating that 2 sarees and 4 shirts cost $1600 -/
theorem two_sarees_four_shirts_cost : 2 * saree_price + 4 * shirt_price = 1600 := by sorry

end NUMINAMATH_CALUDE_two_sarees_four_shirts_cost_l2618_261827


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l2618_261898

theorem weekend_rain_probability
  (p_rain_saturday : ℝ)
  (p_rain_sunday : ℝ)
  (p_rain_sunday_given_no_saturday : ℝ)
  (h1 : p_rain_saturday = 0.6)
  (h2 : p_rain_sunday = 0.4)
  (h3 : p_rain_sunday_given_no_saturday = 0.7)
  : ℝ :=
by
  -- Probability of rain over the weekend
  sorry

#check weekend_rain_probability

end NUMINAMATH_CALUDE_weekend_rain_probability_l2618_261898


namespace NUMINAMATH_CALUDE_power_of_prime_iff_only_prime_factor_l2618_261848

theorem power_of_prime_iff_only_prime_factor (p n : ℕ) : 
  Prime p → (∃ k : ℕ, n = p ^ k) ↔ (∀ q : ℕ, Prime q → q ∣ n → q = p) :=
sorry

end NUMINAMATH_CALUDE_power_of_prime_iff_only_prime_factor_l2618_261848


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_range_l2618_261801

theorem complex_in_second_quadrant_range (x : ℝ) : 
  (x^2 - 6*x + 5 < 0) ∧ (x - 2 > 0) → 2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_range_l2618_261801


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2618_261813

/-- Given a positive geometric sequence {a_n} where a_7 = a_6 + 2a_5, 
    and there exist two terms a_m and a_n such that √(a_m * a_n) = 4a_1,
    the minimum value of 1/m + 4/n is 3/2. -/
theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive sequence
  (∃ q > 0, ∀ k, a (k + 1) = q * a k) →  -- Geometric sequence
  a 7 = a 6 + 2 * a 5 →  -- Given condition
  Real.sqrt (a m * a n) = 4 * a 1 →  -- Given condition
  (∀ i j : ℕ, 1 / i + 4 / j ≥ 3 / 2) ∧  -- Minimum value is at least 3/2
  (∃ i j : ℕ, 1 / i + 4 / j = 3 / 2) :=  -- Minimum value of 3/2 is achievable
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2618_261813


namespace NUMINAMATH_CALUDE_sum_of_three_fourth_powers_not_end_2019_l2618_261867

theorem sum_of_three_fourth_powers_not_end_2019 :
  ∀ a b c : ℤ, ¬ (∃ k : ℤ, a^4 + b^4 + c^4 = 10000 * k + 2019) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_fourth_powers_not_end_2019_l2618_261867


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2618_261810

theorem min_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2618_261810


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2618_261806

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_cond : 2 * a 1 + a 2 = a 3) :
  (a 4 + a 5) / (a 3 + a 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2618_261806
