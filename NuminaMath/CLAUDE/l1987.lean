import Mathlib

namespace NUMINAMATH_CALUDE_andrey_stamps_l1987_198702

theorem andrey_stamps :
  ∃ (x : ℕ), 
    x % 3 = 1 ∧ 
    x % 5 = 3 ∧ 
    x % 7 = 5 ∧ 
    150 < x ∧ 
    x ≤ 300 ∧ 
    x = 208 := by
  sorry

end NUMINAMATH_CALUDE_andrey_stamps_l1987_198702


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1987_198763

theorem sqrt_inequality (a : ℝ) : (0 < a ∧ a < 1) ↔ a < Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1987_198763


namespace NUMINAMATH_CALUDE_no_natural_number_with_three_prime_divisors_l1987_198725

theorem no_natural_number_with_three_prime_divisors :
  ¬ ∃ (m p q r : ℕ),
    (Prime p ∧ Prime q ∧ Prime r) ∧
    (∃ (a b c : ℕ), m = p^a * q^b * r^c) ∧
    (p - 1 ∣ m) ∧
    (q * r - 1 ∣ m) ∧
    ¬(q - 1 ∣ m) ∧
    ¬(r - 1 ∣ m) ∧
    ¬(3 ∣ q + r) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_with_three_prime_divisors_l1987_198725


namespace NUMINAMATH_CALUDE_greatest_distance_between_sets_l1987_198711

def set_A : Set ℂ := {z : ℂ | z^4 - 16 = 0}
def set_B : Set ℂ := {z : ℂ | z^4 - 16*z^3 + 64*z^2 - 16*z + 16 = 0}

theorem greatest_distance_between_sets : 
  ∃ (a : ℂ) (b : ℂ), a ∈ set_A ∧ b ∈ set_B ∧ 
    (∀ (x : ℂ) (y : ℂ), x ∈ set_A → y ∈ set_B → Complex.abs (x - y) ≤ Complex.abs (a - b)) ∧
    Complex.abs (a - b) = 3 :=
sorry

end NUMINAMATH_CALUDE_greatest_distance_between_sets_l1987_198711


namespace NUMINAMATH_CALUDE_point_placement_theorem_l1987_198768

theorem point_placement_theorem : ∃ n : ℕ+, 9 * n - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_point_placement_theorem_l1987_198768


namespace NUMINAMATH_CALUDE_five_bikes_in_driveway_l1987_198712

/-- Calculates the number of bikes in the driveway given the total number of wheels and other vehicles --/
def number_of_bikes (total_wheels car_count tricycle_count trash_can_count roller_skate_wheels : ℕ) : ℕ :=
  let car_wheels := 4 * car_count
  let tricycle_wheels := 3 * tricycle_count
  let remaining_wheels := total_wheels - (car_wheels + tricycle_wheels + roller_skate_wheels)
  let bike_and_trash_can_wheels := remaining_wheels - (2 * trash_can_count)
  bike_and_trash_can_wheels / 2

/-- Theorem stating that there are 5 bikes in the driveway --/
theorem five_bikes_in_driveway :
  number_of_bikes 25 2 1 1 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_bikes_in_driveway_l1987_198712


namespace NUMINAMATH_CALUDE_iris_pants_purchase_l1987_198757

/-- Represents the number of pairs of pants Iris bought -/
def num_pants : ℕ := sorry

/-- The cost of each jacket -/
def jacket_cost : ℕ := 10

/-- The number of jackets bought -/
def num_jackets : ℕ := 3

/-- The cost of each pair of shorts -/
def shorts_cost : ℕ := 6

/-- The number of pairs of shorts bought -/
def num_shorts : ℕ := 2

/-- The cost of each pair of pants -/
def pants_cost : ℕ := 12

/-- The total amount spent -/
def total_spent : ℕ := 90

theorem iris_pants_purchase :
  num_pants = 4 ∧
  num_pants * pants_cost + num_jackets * jacket_cost + num_shorts * shorts_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_iris_pants_purchase_l1987_198757


namespace NUMINAMATH_CALUDE_two_distinct_roots_condition_l1987_198781

theorem two_distinct_roots_condition (b : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ |2^x - 1| = b ∧ |2^y - 1| = b) ↔ 0 < b ∧ b < 1 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_roots_condition_l1987_198781


namespace NUMINAMATH_CALUDE_coes_speed_l1987_198794

theorem coes_speed (teena_speed : ℝ) (initial_distance : ℝ) (time : ℝ) (final_distance : ℝ) :
  teena_speed = 55 →
  initial_distance = 7.5 →
  time = 1.5 →
  final_distance = 15 →
  ∃ coe_speed : ℝ,
    coe_speed = 50 ∧
    teena_speed * time - coe_speed * time = final_distance + initial_distance :=
by
  sorry

end NUMINAMATH_CALUDE_coes_speed_l1987_198794


namespace NUMINAMATH_CALUDE_tangent_line_equation_maximum_value_l1987_198769

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 8

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m*x + b ↔ 3*x + y - 9 = 0) ∧
  (∃ ε > 0, ∀ h : ℝ, 0 < |h| → |h| < ε →
    |f (1 + h) - (f 1 + m*h)| / |h| < ε) :=
sorry

-- Theorem for the maximum value
theorem maximum_value :
  ∀ x : ℝ, f x ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_maximum_value_l1987_198769


namespace NUMINAMATH_CALUDE_ned_garage_sale_games_l1987_198762

/-- The number of games Ned bought from a friend -/
def games_from_friend : ℕ := 50

/-- The number of games that didn't work -/
def bad_games : ℕ := 74

/-- The number of good games Ned ended up with -/
def good_games : ℕ := 3

/-- The number of games Ned bought at the garage sale -/
def games_from_garage_sale : ℕ := (good_games + bad_games) - games_from_friend

theorem ned_garage_sale_games :
  games_from_garage_sale = 27 := by sorry

end NUMINAMATH_CALUDE_ned_garage_sale_games_l1987_198762


namespace NUMINAMATH_CALUDE_circle_through_three_points_l1987_198772

/-- The equation of a circle passing through three given points -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y - 12 = 0

/-- Point A coordinates -/
def point_A : ℝ × ℝ := (5, 1)

/-- Point B coordinates -/
def point_B : ℝ × ℝ := (6, 0)

/-- Point C coordinates -/
def point_C : ℝ × ℝ := (-1, 1)

/-- Theorem stating that the given equation represents the unique circle passing through the three points -/
theorem circle_through_three_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 ∧
  (∀ (D E F : ℝ), (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 ↔ circle_equation x y) →
    D = -4 ∧ E = 6 ∧ F = -12) :=
by sorry

end NUMINAMATH_CALUDE_circle_through_three_points_l1987_198772


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l1987_198791

theorem x_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → (x - 1) * (x + 2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l1987_198791


namespace NUMINAMATH_CALUDE_trajectory_equation_l1987_198753

/-- The trajectory of point P satisfies x² + y² = 1, given a line l: x cos θ + y sin θ = 1,
    where OP is perpendicular to l at P, and O is the origin. -/
theorem trajectory_equation (θ : ℝ) (x y : ℝ) :
  (∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y ∧
    (x * Real.cos θ + y * Real.sin θ = 1) ∧
    (∃ (t : ℝ), P = (t * Real.cos θ, t * Real.sin θ))) →
  x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1987_198753


namespace NUMINAMATH_CALUDE_unique_intersection_l1987_198787

/-- The coefficient of x^2 in the quadratic equation -/
def b : ℚ := 49 / 16

/-- The quadratic function -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 2

/-- The linear function -/
def g (x : ℝ) : ℝ := -2 * x - 2

/-- The difference between the quadratic and linear functions -/
def h (x : ℝ) : ℝ := f x - g x

theorem unique_intersection :
  ∃! x, h x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l1987_198787


namespace NUMINAMATH_CALUDE_exists_integer_between_sqrt2_and_sqrt17_l1987_198740

theorem exists_integer_between_sqrt2_and_sqrt17 : ∃ n : ℤ, Real.sqrt 2 < n ∧ n < Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_between_sqrt2_and_sqrt17_l1987_198740


namespace NUMINAMATH_CALUDE_surface_area_of_combined_solid_l1987_198733

/-- Calculates the surface area of a rectangular solid -/
def surfaceAreaRect (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Represents the combined solid formed by attaching a rectangular prism to a rectangular solid -/
structure CombinedSolid where
  mainLength : ℝ
  mainWidth : ℝ
  mainHeight : ℝ
  attachedLength : ℝ
  attachedWidth : ℝ
  attachedHeight : ℝ

/-- Calculates the total surface area of the combined solid -/
def totalSurfaceArea (s : CombinedSolid) : ℝ :=
  surfaceAreaRect s.mainLength s.mainWidth s.mainHeight +
  surfaceAreaRect s.attachedLength s.attachedWidth s.attachedHeight -
  2 * (s.attachedLength * s.attachedWidth)

/-- The specific combined solid from the problem -/
def problemSolid : CombinedSolid :=
  { mainLength := 4
    mainWidth := 3
    mainHeight := 2
    attachedLength := 2
    attachedWidth := 1
    attachedHeight := 1 }

theorem surface_area_of_combined_solid :
  totalSurfaceArea problemSolid = 58 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_combined_solid_l1987_198733


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_l1987_198747

/-- Represents the cost of pencils and erasers -/
structure PencilEraserCost where
  pencil : ℕ
  eraser : ℕ

/-- The total cost of 10 pencils and 4 erasers in cents -/
def totalCost : ℕ := 120

/-- Condition: A pencil costs more than an eraser -/
def pencilCostsMore (cost : PencilEraserCost) : Prop :=
  cost.pencil > cost.eraser

/-- Condition: The total cost equation must be satisfied -/
def satisfiesTotalCost (cost : PencilEraserCost) : Prop :=
  10 * cost.pencil + 4 * cost.eraser = totalCost

/-- The main theorem to prove -/
theorem pencil_eraser_cost :
  ∃ (cost : PencilEraserCost),
    pencilCostsMore cost ∧
    satisfiesTotalCost cost ∧
    cost.pencil + cost.eraser = 15 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_l1987_198747


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1987_198759

theorem arithmetic_computation : 5 + 4 * (4 - 9)^2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1987_198759


namespace NUMINAMATH_CALUDE_dance_girls_fraction_l1987_198705

theorem dance_girls_fraction (colfax_total : ℕ) (winthrop_total : ℕ)
  (colfax_boy_ratio colfax_girl_ratio : ℕ)
  (winthrop_boy_ratio winthrop_girl_ratio : ℕ)
  (h1 : colfax_total = 270)
  (h2 : winthrop_total = 180)
  (h3 : colfax_boy_ratio = 5 ∧ colfax_girl_ratio = 4)
  (h4 : winthrop_boy_ratio = 4 ∧ winthrop_girl_ratio = 5) :
  let colfax_girls := colfax_total * colfax_girl_ratio / (colfax_boy_ratio + colfax_girl_ratio)
  let winthrop_girls := winthrop_total * winthrop_girl_ratio / (winthrop_boy_ratio + winthrop_girl_ratio)
  let total_girls := colfax_girls + winthrop_girls
  let total_students := colfax_total + winthrop_total
  (total_girls : ℚ) / total_students = 22 / 45 := by
sorry

end NUMINAMATH_CALUDE_dance_girls_fraction_l1987_198705


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1987_198760

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (hyp : a^2 + b^2 = c^2) -- Pythagorean theorem
  (hypotenuse : c = 13) 
  (side : a = 12) : 
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1987_198760


namespace NUMINAMATH_CALUDE_fraction_equality_l1987_198745

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 3) 
  (h2 : r / t = 8 / 15) : 
  (4 * m * r - 2 * n * t) / (5 * n * t - 9 * m * r) = -14 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1987_198745


namespace NUMINAMATH_CALUDE_farm_area_and_planned_days_correct_l1987_198729

/-- Represents the farm field and ploughing scenario -/
structure FarmField where
  planned_daily_area : ℝ
  actual_daily_area : ℝ
  type_a_percentage : ℝ
  type_b_percentage : ℝ
  type_c_percentage : ℝ
  type_a_hours_per_hectare : ℝ
  type_b_hours_per_hectare : ℝ
  type_c_hours_per_hectare : ℝ
  extra_days_worked : ℕ
  area_left_to_plough : ℝ
  max_hours_per_day : ℝ

/-- Calculates the total area of the farm field and the initially planned work days -/
def calculate_farm_area_and_planned_days (field : FarmField) : ℝ × ℕ :=
  sorry

/-- Theorem stating the correct total area and initially planned work days -/
theorem farm_area_and_planned_days_correct (field : FarmField) 
  (h1 : field.planned_daily_area = 260)
  (h2 : field.actual_daily_area = 85)
  (h3 : field.type_a_percentage = 0.4)
  (h4 : field.type_b_percentage = 0.3)
  (h5 : field.type_c_percentage = 0.3)
  (h6 : field.type_a_hours_per_hectare = 4)
  (h7 : field.type_b_hours_per_hectare = 6)
  (h8 : field.type_c_hours_per_hectare = 3)
  (h9 : field.extra_days_worked = 2)
  (h10 : field.area_left_to_plough = 40)
  (h11 : field.max_hours_per_day = 12) :
  calculate_farm_area_and_planned_days field = (340, 2) :=
by
  sorry

end NUMINAMATH_CALUDE_farm_area_and_planned_days_correct_l1987_198729


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_half_l1987_198789

/-- Represents a cube constructed from smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  white_cubes : ℕ
  black_cubes : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def white_surface_fraction (c : CompositeCube) : ℚ :=
  sorry

/-- The specific composite cube from the problem -/
def problem_cube : CompositeCube :=
  { edge_length := 4
  , total_small_cubes := 64
  , white_cubes := 48
  , black_cubes := 16 }

theorem white_surface_fraction_is_half :
  white_surface_fraction problem_cube = 1/2 :=
sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_half_l1987_198789


namespace NUMINAMATH_CALUDE_clock_gains_five_minutes_per_hour_l1987_198766

/-- A clock that gains time -/
structure GainingClock where
  start_time : ℕ  -- Start time in hours (24-hour format)
  end_time : ℕ    -- End time in hours (24-hour format)
  total_gain : ℕ  -- Total minutes gained

/-- Calculate the minutes gained per hour -/
def minutes_gained_per_hour (clock : GainingClock) : ℚ :=
  clock.total_gain / (clock.end_time - clock.start_time)

/-- Theorem: A clock that starts at 9 a.m. and gains 45 minutes by 6 p.m. gains 5 minutes per hour -/
theorem clock_gains_five_minutes_per_hour (clock : GainingClock) 
    (h1 : clock.start_time = 9)
    (h2 : clock.end_time = 18)
    (h3 : clock.total_gain = 45) :
  minutes_gained_per_hour clock = 5 := by
  sorry

end NUMINAMATH_CALUDE_clock_gains_five_minutes_per_hour_l1987_198766


namespace NUMINAMATH_CALUDE_pool_filling_time_l1987_198701

/-- Calculates the time in hours required to fill a pool given its capacity and the rate of water flow. -/
theorem pool_filling_time 
  (pool_capacity : ℚ)  -- Pool capacity in gallons
  (num_hoses : ℕ)      -- Number of hoses
  (flow_rate : ℚ)      -- Flow rate per hose in gallons per minute
  (h : pool_capacity = 36000 ∧ num_hoses = 6 ∧ flow_rate = 3) :
  (pool_capacity / (↑num_hoses * flow_rate * 60)) = 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l1987_198701


namespace NUMINAMATH_CALUDE_books_about_trains_l1987_198774

def books_about_animals : ℕ := 10
def books_about_space : ℕ := 1
def cost_per_book : ℕ := 16
def total_spent : ℕ := 224

theorem books_about_trains : ℕ := by
  sorry

end NUMINAMATH_CALUDE_books_about_trains_l1987_198774


namespace NUMINAMATH_CALUDE_amy_biking_distance_l1987_198716

theorem amy_biking_distance (x : ℝ) : 
  x + (2 * x - 3) = 33 → x = 12 := by sorry

end NUMINAMATH_CALUDE_amy_biking_distance_l1987_198716


namespace NUMINAMATH_CALUDE_exists_determining_question_l1987_198742

-- Define the types of guests
inductive GuestType
| Human
| Vampire

-- Define the possible answers
inductive Answer
| Bal
| Da

-- Define a question as a function that takes a GuestType and returns an Answer
def Question := GuestType → Answer

-- Define a function to determine the guest type based on the answer
def determineGuestType (q : Question) (a : Answer) : GuestType := 
  match a with
  | Answer.Bal => GuestType.Human
  | Answer.Da => GuestType.Vampire

-- Theorem statement
theorem exists_determining_question : 
  ∃ (q : Question), 
    (∀ (g : GuestType), (determineGuestType q (q g)) = g) :=
sorry

end NUMINAMATH_CALUDE_exists_determining_question_l1987_198742


namespace NUMINAMATH_CALUDE_log_sqrt2_and_inequality_l1987_198777

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sqrt2_and_inequality :
  (log 4 (Real.sqrt 2) = 1/4) ∧
  (∀ x : ℝ, log x (Real.sqrt 2) > 1 ↔ 1 < x ∧ x < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_log_sqrt2_and_inequality_l1987_198777


namespace NUMINAMATH_CALUDE_eleven_team_league_games_l1987_198734

/-- The number of games played in a league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 11 teams, where each team plays every other team exactly once, 
    the total number of games played is 55. -/
theorem eleven_team_league_games : games_played 11 = 55 := by
  sorry

end NUMINAMATH_CALUDE_eleven_team_league_games_l1987_198734


namespace NUMINAMATH_CALUDE_complex_conjugate_root_l1987_198703

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- A complex number is a root of a polynomial if the polynomial evaluates to zero at that number -/
def is_root (f : RealPolynomial) (z : ℂ) : Prop := f z.re = 0 ∧ f z.im = 0

theorem complex_conjugate_root (f : RealPolynomial) (a b : ℝ) :
  is_root f (Complex.mk a b) → is_root f (Complex.mk a (-b)) :=
by sorry

end NUMINAMATH_CALUDE_complex_conjugate_root_l1987_198703


namespace NUMINAMATH_CALUDE_trapezoid_height_l1987_198735

/-- Represents a trapezoid with height x, bases 3x and 5x, and area 40 -/
structure Trapezoid where
  x : ℝ
  base1 : ℝ := 3 * x
  base2 : ℝ := 5 * x
  area : ℝ := 40

/-- The height of a trapezoid with the given properties is √10 -/
theorem trapezoid_height (t : Trapezoid) : t.x = Real.sqrt 10 := by
  sorry

#check trapezoid_height

end NUMINAMATH_CALUDE_trapezoid_height_l1987_198735


namespace NUMINAMATH_CALUDE_infinitely_many_rationals_between_one_sixth_and_five_sixths_l1987_198723

theorem infinitely_many_rationals_between_one_sixth_and_five_sixths :
  ∃ (S : Set ℚ), Set.Infinite S ∧ ∀ q ∈ S, 1/6 < q ∧ q < 5/6 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_rationals_between_one_sixth_and_five_sixths_l1987_198723


namespace NUMINAMATH_CALUDE_coconut_to_mango_ratio_l1987_198704

/-- Proves that the ratio of coconut trees to mango trees is 1:2 given the conditions --/
theorem coconut_to_mango_ratio :
  ∀ (mango_trees coconut_trees total_trees : ℕ) (ratio : ℚ),
    mango_trees = 60 →
    total_trees = 85 →
    coconut_trees = mango_trees * ratio - 5 →
    total_trees = mango_trees + coconut_trees →
    coconut_trees * 2 = mango_trees := by
  sorry

end NUMINAMATH_CALUDE_coconut_to_mango_ratio_l1987_198704


namespace NUMINAMATH_CALUDE_lanas_tickets_l1987_198795

/-- The number of tickets Lana bought for herself and friends -/
def tickets_for_friends : ℕ := sorry

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 6

/-- The number of extra tickets Lana bought -/
def extra_tickets : ℕ := 2

/-- The total amount Lana spent in dollars -/
def total_spent : ℕ := 60

theorem lanas_tickets : 
  (tickets_for_friends + extra_tickets) * ticket_cost = total_spent ∧ 
  tickets_for_friends = 8 := by
  sorry

end NUMINAMATH_CALUDE_lanas_tickets_l1987_198795


namespace NUMINAMATH_CALUDE_plumber_salary_percentage_l1987_198709

-- Define the daily salaries and total labor cost
def construction_worker_salary : ℝ := 100
def electrician_salary : ℝ := 2 * construction_worker_salary
def total_labor_cost : ℝ := 650

-- Define the number of workers
def num_construction_workers : ℕ := 2
def num_electricians : ℕ := 1
def num_plumbers : ℕ := 1

-- Calculate the plumber's salary
def plumber_salary : ℝ :=
  total_labor_cost - (num_construction_workers * construction_worker_salary + num_electricians * electrician_salary)

-- Define the theorem
theorem plumber_salary_percentage :
  plumber_salary / construction_worker_salary * 100 = 250 := by
  sorry


end NUMINAMATH_CALUDE_plumber_salary_percentage_l1987_198709


namespace NUMINAMATH_CALUDE_max_distance_complex_l1987_198771

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  (⨆ z, |(2 + 3*I)*z^2 - z^4|) = 81 + 9 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_l1987_198771


namespace NUMINAMATH_CALUDE_problem_statement_l1987_198708

theorem problem_statement (x : ℝ) (h : x^2 + 8 * (x / (x - 3))^2 = 53) :
  ((x - 3)^3 * (x + 4)) / (2 * x - 5) = 17000 / 21 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1987_198708


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1987_198770

/-- Given that the solution set of ax² + bx - 2 > 0 is (-4, 1), prove that a + b = 2 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x - 2 > 0 ↔ -4 < x ∧ x < 1) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1987_198770


namespace NUMINAMATH_CALUDE_sunflower_seeds_majority_l1987_198731

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  sunflowerSeeds : Rat
  otherSeeds : Rat

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    sunflowerSeeds := state.sunflowerSeeds * (4/5) + (2/5),
    otherSeeds := 3/5 }

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1,
    sunflowerSeeds := 2/5,
    otherSeeds := 3/5 }

/-- Theorem stating that on the third day, more than half the seeds are sunflower seeds -/
theorem sunflower_seeds_majority : 
  let state3 := nextDay (nextDay initialState)
  state3.sunflowerSeeds > (state3.sunflowerSeeds + state3.otherSeeds) / 2 := by
  sorry


end NUMINAMATH_CALUDE_sunflower_seeds_majority_l1987_198731


namespace NUMINAMATH_CALUDE_property_1_property_2_property_3_f_satisfies_all_properties_l1987_198744

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Property 1: f(xy) = f(x)f(y)
theorem property_1 : ∀ x y : ℝ, f (x * y) = f x * f y := by sorry

-- Property 2: f'(x) is an even function
theorem property_2 : ∀ x : ℝ, (deriv f) (-x) = (deriv f) x := by sorry

-- Property 3: f(x) is monotonically increasing on (0, +∞)
theorem property_3 : ∀ x y : ℝ, 0 < x → x < y → f x < f y := by sorry

-- Main theorem: f(x) = x^3 satisfies all three properties
theorem f_satisfies_all_properties :
  (∀ x y : ℝ, f (x * y) = f x * f y) ∧
  (∀ x : ℝ, (deriv f) (-x) = (deriv f) x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_property_1_property_2_property_3_f_satisfies_all_properties_l1987_198744


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1987_198713

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) →
  (a = 1 ∧ b = 2) ∧
  (∀ c : ℝ, 
    (∀ x, x^2 - (c + 2) * x + 2 * c ≤ 0 ↔ 
      (c < 2 ∧ c ≤ x ∧ x ≤ 2) ∨
      (c = 2 ∧ x = 2) ∨
      (c > 2 ∧ 2 ≤ x ∧ x ≤ c))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1987_198713


namespace NUMINAMATH_CALUDE_integral_equality_l1987_198785

theorem integral_equality : ∫ x in (1 : ℝ)..Real.sqrt 3, 
  (x^(2*x^2 + 1) + Real.log (x^(2*x^(2*x^2 + 1)))) = 13 := by sorry

end NUMINAMATH_CALUDE_integral_equality_l1987_198785


namespace NUMINAMATH_CALUDE_seven_digit_multiple_of_each_l1987_198743

/-- A function that returns the set of digits of a positive integer -/
def digits (n : ℕ+) : Finset ℕ :=
  sorry

/-- The theorem statement -/
theorem seven_digit_multiple_of_each : ∃ (n : ℕ+),
  (digits n).card = 7 ∧
  ∀ d ∈ digits n, d > 0 ∧ n % d = 0 →
  digits n = {1, 2, 3, 6, 7, 8, 9} :=
sorry

end NUMINAMATH_CALUDE_seven_digit_multiple_of_each_l1987_198743


namespace NUMINAMATH_CALUDE_optimal_distribution_part1_optimal_distribution_part2_l1987_198732

/-- Represents the types of vegetables -/
inductive VegetableType
| A
| B
| C

/-- Properties of each vegetable type -/
def tons_per_truck (v : VegetableType) : ℚ :=
  match v with
  | .A => 2
  | .B => 1
  | .C => 2.5

def profit_per_ton (v : VegetableType) : ℚ :=
  match v with
  | .A => 5
  | .B => 7
  | .C => 4

/-- Theorem for part 1 -/
theorem optimal_distribution_part1 :
  ∃ (b c : ℕ),
    b + c = 14 ∧
    b * tons_per_truck VegetableType.B + c * tons_per_truck VegetableType.C = 17 ∧
    b = 12 ∧ c = 2 := by sorry

/-- Theorem for part 2 -/
theorem optimal_distribution_part2 :
  ∃ (a b c : ℕ) (max_profit : ℚ),
    a + b + c = 30 ∧
    1 ≤ a ∧ a ≤ 10 ∧
    a * tons_per_truck VegetableType.A + b * tons_per_truck VegetableType.B + c * tons_per_truck VegetableType.C = 48 ∧
    a = 9 ∧ b = 15 ∧ c = 6 ∧
    max_profit = 255 ∧
    (∀ (a' b' c' : ℕ),
      a' + b' + c' = 30 →
      1 ≤ a' ∧ a' ≤ 10 →
      a' * tons_per_truck VegetableType.A + b' * tons_per_truck VegetableType.B + c' * tons_per_truck VegetableType.C = 48 →
      a' * tons_per_truck VegetableType.A * profit_per_ton VegetableType.A +
      b' * tons_per_truck VegetableType.B * profit_per_ton VegetableType.B +
      c' * tons_per_truck VegetableType.C * profit_per_ton VegetableType.C ≤ max_profit) := by sorry

end NUMINAMATH_CALUDE_optimal_distribution_part1_optimal_distribution_part2_l1987_198732


namespace NUMINAMATH_CALUDE_beth_sold_coins_l1987_198796

-- Define the initial number of coins Beth had
def initial_coins : ℕ := 125

-- Define the number of coins Carl gave to Beth
def gifted_coins : ℕ := 35

-- Define the total number of coins Beth had after receiving the gift
def total_coins : ℕ := initial_coins + gifted_coins

-- Define the number of coins Beth sold (half of her total coins)
def sold_coins : ℕ := total_coins / 2

-- Theorem stating that the number of coins Beth sold is equal to 80
theorem beth_sold_coins : sold_coins = 80 := by
  sorry

end NUMINAMATH_CALUDE_beth_sold_coins_l1987_198796


namespace NUMINAMATH_CALUDE_jane_score_is_12_l1987_198784

/-- Represents the score calculation for a modified AMC 8 contest --/
def modified_amc_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - (incorrect : ℚ) / 2

/-- Theorem stating that Jane's score in the modified AMC 8 contest is 12 --/
theorem jane_score_is_12 :
  let total_questions : ℕ := 35
  let correct_answers : ℕ := 18
  let incorrect_answers : ℕ := 12
  let unanswered_questions : ℕ := 5
  modified_amc_score correct_answers incorrect_answers unanswered_questions = 12 := by
  sorry

#eval modified_amc_score 18 12 5

end NUMINAMATH_CALUDE_jane_score_is_12_l1987_198784


namespace NUMINAMATH_CALUDE_solution_set_eq_singleton_l1987_198710

/-- The set of solutions to the system of equations x + y = 2 and x - y = 0 -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 2 ∧ p.1 - p.2 = 0}

/-- Theorem stating that the solution set is equal to {(1, 1)} -/
theorem solution_set_eq_singleton : solution_set = {(1, 1)} := by
  sorry

#check solution_set_eq_singleton

end NUMINAMATH_CALUDE_solution_set_eq_singleton_l1987_198710


namespace NUMINAMATH_CALUDE_unreachable_corner_l1987_198706

/-- A point in 3D space with integer coordinates -/
structure Point3D where
  x : Int
  y : Int
  z : Int

/-- The set of 7 vertices of a cube, excluding (1,1,1) -/
def cube_vertices : Set Point3D :=
  { ⟨0,0,0⟩, ⟨0,0,1⟩, ⟨0,1,0⟩, ⟨1,0,0⟩, ⟨0,1,1⟩, ⟨1,0,1⟩, ⟨1,1,0⟩ }

/-- Symmetry transformation with respect to another point -/
def symmetry_transform (p : Point3D) (center : Point3D) : Point3D :=
  ⟨2 * center.x - p.x, 2 * center.y - p.y, 2 * center.z - p.z⟩

/-- The set of points reachable through symmetry transformations -/
def reachable_points : Set Point3D :=
  sorry -- Definition of reachable points through symmetry transformations

theorem unreachable_corner : ⟨1,1,1⟩ ∉ reachable_points := by
  sorry

#check unreachable_corner

end NUMINAMATH_CALUDE_unreachable_corner_l1987_198706


namespace NUMINAMATH_CALUDE_square_area_from_corners_l1987_198783

/-- The area of a square with adjacent corners at (1, 2) and (-2, 2) is 9 -/
theorem square_area_from_corners : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (-2, 2)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := side_length^2
  area = 9 := by sorry

end NUMINAMATH_CALUDE_square_area_from_corners_l1987_198783


namespace NUMINAMATH_CALUDE_headcount_analysis_l1987_198793

/-- Student headcount data for spring terms -/
structure HeadcountData where
  y02_03 : ℕ
  y03_04 : ℕ
  y04_05 : ℕ
  y05_06 : ℕ

/-- Calculate average headcount -/
def average_headcount (data : HeadcountData) : ℚ :=
  (data.y02_03 + data.y03_04 + data.y04_05 + data.y05_06) / 4

/-- Calculate percentage change -/
def percentage_change (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Theorem stating the average headcount and percentage change -/
theorem headcount_analysis (data : HeadcountData)
  (h1 : data.y02_03 = 10000)
  (h2 : data.y03_04 = 11000)
  (h3 : data.y04_05 = 9500)
  (h4 : data.y05_06 = 10500) :
  average_headcount data = 10125 ∧ percentage_change data.y02_03 data.y05_06 = 5 := by
  sorry

#eval average_headcount ⟨10000, 11000, 9500, 10500⟩
#eval percentage_change 10000 10500

end NUMINAMATH_CALUDE_headcount_analysis_l1987_198793


namespace NUMINAMATH_CALUDE_plane_line_parallel_l1987_198799

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- Line is subset of plane
variable (parallel : Line → Line → Prop) -- Lines are parallel
variable (parallel_plane : Line → Plane → Prop) -- Line is parallel to plane
variable (intersect : Plane → Plane → Line → Prop) -- Planes intersect in a line

-- State the theorem
theorem plane_line_parallel 
  (α β : Plane) (m n : Line) 
  (h1 : intersect α β m) 
  (h2 : parallel n m) 
  (h3 : ¬ subset n α) 
  (h4 : ¬ subset n β) : 
  parallel_plane n α ∧ parallel_plane n β :=
sorry

end NUMINAMATH_CALUDE_plane_line_parallel_l1987_198799


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1987_198746

theorem decimal_to_fraction (x : ℚ) : x = 336/100 → x = 84/25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1987_198746


namespace NUMINAMATH_CALUDE_probability_red_then_white_l1987_198754

/-- The probability of drawing a red ball followed by a white ball in two successive draws with replacement -/
theorem probability_red_then_white (total : ℕ) (red : ℕ) (white : ℕ) 
  (h_total : total = 9)
  (h_red : red = 3)
  (h_white : white = 2) :
  (red : ℚ) / total * (white : ℚ) / total = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_then_white_l1987_198754


namespace NUMINAMATH_CALUDE_distance_to_larger_section_l1987_198749

/-- Right pentagonal pyramid with two parallel cross sections -/
structure RightPentagonalPyramid where
  /-- Area of smaller cross section in square feet -/
  area_small : ℝ
  /-- Area of larger cross section in square feet -/
  area_large : ℝ
  /-- Distance between the two cross sections in feet -/
  distance_between : ℝ

/-- Theorem: Distance from apex to larger cross section -/
theorem distance_to_larger_section (pyramid : RightPentagonalPyramid) 
  (h_area_small : pyramid.area_small = 100 * Real.sqrt 3)
  (h_area_large : pyramid.area_large = 225 * Real.sqrt 3)
  (h_distance : pyramid.distance_between = 5) :
  ∃ (d : ℝ), d = 15 ∧ d * d * pyramid.area_small = (d - 5) * (d - 5) * pyramid.area_large :=
by sorry

end NUMINAMATH_CALUDE_distance_to_larger_section_l1987_198749


namespace NUMINAMATH_CALUDE_constant_sum_zero_l1987_198752

theorem constant_sum_zero (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / (x + 1)) →
  (2 = a + b / (1 + 1)) →
  (3 = a + b / (3 + 1)) →
  a + b = 0 := by sorry

end NUMINAMATH_CALUDE_constant_sum_zero_l1987_198752


namespace NUMINAMATH_CALUDE_area_triangle_EYH_l1987_198718

/-- Represents a trapezoid with bases and diagonals -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Theorem: Area of triangle EYH in trapezoid EFGH -/
theorem area_triangle_EYH (EFGH : Trapezoid) (h1 : EFGH.base1 = 15) (h2 : EFGH.base2 = 35) (h3 : EFGH.area = 400) :
  ∃ (area_EYH : ℝ), area_EYH = 84 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_EYH_l1987_198718


namespace NUMINAMATH_CALUDE_smartpup_academy_total_dogs_l1987_198761

/-- Represents the number of dogs at Smartpup Tricks Academy with various skill combinations -/
structure DogSkills where
  fetch : ℕ
  jump : ℕ
  play_dead : ℕ
  fetch_and_jump : ℕ
  jump_and_play_dead : ℕ
  fetch_and_play_dead : ℕ
  all_three : ℕ
  none : ℕ

/-- Calculates the total number of dogs at the academy -/
def total_dogs (skills : DogSkills) : ℕ :=
  skills.all_three +
  (skills.fetch_and_play_dead - skills.all_three) +
  (skills.jump_and_play_dead - skills.all_three) +
  (skills.fetch_and_jump - skills.all_three) +
  (skills.fetch - skills.fetch_and_jump - skills.fetch_and_play_dead + skills.all_three) +
  (skills.jump - skills.fetch_and_jump - skills.jump_and_play_dead + skills.all_three) +
  (skills.play_dead - skills.fetch_and_play_dead - skills.jump_and_play_dead + skills.all_three) +
  skills.none

/-- The main theorem stating that the total number of dogs is 75 -/
theorem smartpup_academy_total_dogs :
  let skills : DogSkills := {
    fetch := 40,
    jump := 35,
    play_dead := 22,
    fetch_and_jump := 14,
    jump_and_play_dead := 10,
    fetch_and_play_dead := 16,
    all_three := 6,
    none := 12
  }
  total_dogs skills = 75 := by
  sorry

end NUMINAMATH_CALUDE_smartpup_academy_total_dogs_l1987_198761


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1987_198738

theorem max_value_quadratic :
  ∃ (M : ℝ), M = 26 ∧ ∀ (x : ℝ), -3 * x^2 + 18 * x - 1 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1987_198738


namespace NUMINAMATH_CALUDE_amys_tickets_proof_l1987_198788

/-- The total number of tickets Amy has after buying more at the fair -/
def amys_total_tickets (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that Amy's total tickets is 54 given her initial and additional tickets -/
theorem amys_tickets_proof :
  amys_total_tickets 33 21 = 54 := by
  sorry

end NUMINAMATH_CALUDE_amys_tickets_proof_l1987_198788


namespace NUMINAMATH_CALUDE_additive_inverses_and_quadratic_roots_l1987_198719

theorem additive_inverses_and_quadratic_roots :
  (∀ x y : ℝ, (∃ z : ℝ, x + z = 0 ∧ y + z = 0) → x + y = 0) ∧
  (∀ q : ℝ, (∀ x : ℝ, x^2 + x + q ≠ 0) → q > -1) := by
  sorry

end NUMINAMATH_CALUDE_additive_inverses_and_quadratic_roots_l1987_198719


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1987_198778

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1987_198778


namespace NUMINAMATH_CALUDE_product_w_k_value_l1987_198727

theorem product_w_k_value : 
  let a : ℕ := 105
  let b : ℕ := 60
  let c : ℕ := 42
  let w : ℕ := (a^3 * b^4) / (21 * 25 * 45 * 50)
  let k : ℕ := c^5 / (35 * 28 * 56)
  w * k = 18458529600 := by sorry

end NUMINAMATH_CALUDE_product_w_k_value_l1987_198727


namespace NUMINAMATH_CALUDE_odd_cube_difference_divisible_by_24_l1987_198715

theorem odd_cube_difference_divisible_by_24 (n : ℤ) : 
  24 ∣ ((2 * n + 1)^3 - (2 * n + 1)) := by sorry

end NUMINAMATH_CALUDE_odd_cube_difference_divisible_by_24_l1987_198715


namespace NUMINAMATH_CALUDE_expression_evaluation_l1987_198741

theorem expression_evaluation (a b : ℝ) (h : |a + 1| + (b - 1/2)^2 = 0) :
  5 * (a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1987_198741


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1987_198722

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def GeometricSequence (x y z : ℚ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ArithmeticSequence a d)
  (h_geom : GeometricSequence (a 5) (a 9) (a 15)) :
  a 15 / a 9 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1987_198722


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l1987_198764

-- Define the sum of the first n terms of an arithmetic sequence
def T (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

-- State the theorem
theorem arithmetic_sequence_constant_ratio (a : ℚ) :
  (∀ n : ℕ, n > 0 → ∃ k : ℚ, T a (4 * n) / T a n = k) →
  a = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l1987_198764


namespace NUMINAMATH_CALUDE_recipe_solution_l1987_198728

def recipe_problem (sugar : ℕ) (flour_put : ℕ) (flour_sugar_diff : ℕ) : Prop :=
  let total_flour := sugar + flour_sugar_diff
  total_flour = flour_put + (total_flour - flour_put)

theorem recipe_solution :
  let sugar := 7
  let flour_put := 2
  let flour_sugar_diff := 2
  recipe_problem sugar flour_put flour_sugar_diff ∧
  (sugar + flour_sugar_diff = 9) :=
by sorry

end NUMINAMATH_CALUDE_recipe_solution_l1987_198728


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l1987_198750

/-- The sum of distinct powers of 2 that equals 700 -/
def sum_of_powers (powers : List ℕ) : Prop :=
  (powers.map (λ x => 2^x)).sum = 700 ∧ powers.Nodup

/-- The proposition that 30 is the least possible sum of exponents -/
theorem least_sum_of_exponents :
  ∀ powers : List ℕ,
    sum_of_powers powers →
    powers.length ≥ 3 →
    powers.sum ≥ 30 ∧
    ∃ optimal_powers : List ℕ,
      sum_of_powers optimal_powers ∧
      optimal_powers.length ≥ 3 ∧
      optimal_powers.sum = 30 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l1987_198750


namespace NUMINAMATH_CALUDE_line_point_y_coordinate_l1987_198798

/-- Given a line passing through points (3, -1, 0) and (8, -4, -5),
    the y-coordinate of a point on this line with z-coordinate -3 is -14/5 -/
theorem line_point_y_coordinate :
  let p₁ : ℝ × ℝ × ℝ := (3, -1, 0)
  let p₂ : ℝ × ℝ × ℝ := (8, -4, -5)
  let line := {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = p₁ + t • (p₂ - p₁)}
  ∃ p : ℝ × ℝ × ℝ, p ∈ line ∧ p.2.2 = -3 ∧ p.2.1 = -14/5 :=
by sorry

end NUMINAMATH_CALUDE_line_point_y_coordinate_l1987_198798


namespace NUMINAMATH_CALUDE_right_triangle_345_l1987_198765

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_triangle_345 :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5) ∧
  ¬ is_right_triangle 4 6 9 ∧
  is_right_triangle 3 4 5 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_345_l1987_198765


namespace NUMINAMATH_CALUDE_smallest_positive_integer_form_l1987_198730

theorem smallest_positive_integer_form (m n : ℤ) :
  (∃ k : ℕ+, k = |4509 * m + 27981 * n| ∧ 
   ∀ j : ℕ+, (∃ a b : ℤ, j = |4509 * a + 27981 * b|) → k ≤ j) ↔ 
  Nat.gcd 4509 27981 = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_form_l1987_198730


namespace NUMINAMATH_CALUDE_remaining_dogs_l1987_198707

theorem remaining_dogs (total_pets : ℕ) (dogs_given : ℕ) : 
  total_pets = 189 → dogs_given = 10 → 
  (10 : ℚ) / 27 * total_pets - dogs_given = 60 := by
sorry

end NUMINAMATH_CALUDE_remaining_dogs_l1987_198707


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l1987_198792

/-- The function f(x) = (x+1)^2 -/
def f (x : ℝ) : ℝ := (x + 1)^2

/-- The theorem stating that the derivative of f(x) at x = 1 is 4 -/
theorem derivative_f_at_1 : 
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l1987_198792


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1987_198751

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1987_198751


namespace NUMINAMATH_CALUDE_determine_down_speed_man_down_speed_l1987_198782

/-- The speed of a man traveling up and down a hill -/
structure TravelSpeed where
  up : ℝ
  down : ℝ
  average : ℝ

/-- Theorem stating that given the up speed and average speed, we can determine the down speed -/
theorem determine_down_speed (s : TravelSpeed) (h1 : s.up = 24) (h2 : s.average = 28.8) :
  s.down = 36 := by
  sorry

/-- Main theorem proving the specific case in the problem -/
theorem man_down_speed :
  ∃ s : TravelSpeed, s.up = 24 ∧ s.average = 28.8 ∧ s.down = 36 := by
  sorry

end NUMINAMATH_CALUDE_determine_down_speed_man_down_speed_l1987_198782


namespace NUMINAMATH_CALUDE_percentage_puppies_greater_profit_l1987_198717

/-- Calculates the percentage of puppies that can be sold for a greater profit -/
theorem percentage_puppies_greater_profit (total_puppies : ℕ) (puppies_more_than_4_spots : ℕ) 
  (h1 : total_puppies = 10)
  (h2 : puppies_more_than_4_spots = 6) :
  (puppies_more_than_4_spots : ℚ) / total_puppies * 100 = 60 := by
  sorry


end NUMINAMATH_CALUDE_percentage_puppies_greater_profit_l1987_198717


namespace NUMINAMATH_CALUDE_g_equals_g_l1987_198776

/-- Two triangles are similar isosceles triangles with vertex A and angle α -/
def similarIsoscelesA (t1 t2 : Set (ℝ × ℝ)) (A : ℝ × ℝ) (α : ℝ) : Prop :=
  sorry

/-- Two triangles are similar isosceles triangles with angle π - α at the vertex -/
def similarIsoscelesVertex (t1 t2 : Set (ℝ × ℝ)) (α : ℝ) : Prop :=
  sorry

/-- The theorem stating that G = G' given the conditions -/
theorem g_equals_g' (A K L M N G G' : ℝ × ℝ) (α : ℝ) 
    (h1 : similarIsoscelesA {A, K, L} {A, M, N} A α)
    (h2 : similarIsoscelesVertex {G, N, K} {G', L, M} α) :
    G = G' :=
  sorry

end NUMINAMATH_CALUDE_g_equals_g_l1987_198776


namespace NUMINAMATH_CALUDE_equation_solution_l1987_198714

theorem equation_solution :
  ∀ x : ℚ, x ≠ 2 →
  (7 * x / (x - 2) - 5 / (x - 2) = 3 / (x - 2)) ↔ x = 8 / 7 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1987_198714


namespace NUMINAMATH_CALUDE_f_minimum_g_solution_set_l1987_198700

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem for the minimum value of f
theorem f_minimum : ∀ x : ℝ, f x ≥ -3 :=
sorry

-- Define the inequality function g
def g (x : ℝ) : ℝ := x^2 - 8*x + 15 + f x

-- Theorem for the solution set of g(x) ≤ 0
theorem g_solution_set : 
  ∀ x : ℝ, g x ≤ 0 ↔ 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_g_solution_set_l1987_198700


namespace NUMINAMATH_CALUDE_bear_population_l1987_198775

/-- The number of black bears in the park -/
def black_bears : ℕ := 60

/-- The number of white bears in the park -/
def white_bears : ℕ := black_bears / 2

/-- The number of brown bears in the park -/
def brown_bears : ℕ := black_bears + 40

/-- The total population of bears in the park -/
def total_bears : ℕ := white_bears + black_bears + brown_bears

theorem bear_population : total_bears = 190 := by
  sorry

end NUMINAMATH_CALUDE_bear_population_l1987_198775


namespace NUMINAMATH_CALUDE_solve_for_a_l1987_198724

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = 2 → 3 * x - a * y = 1) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1987_198724


namespace NUMINAMATH_CALUDE_no_real_solutions_l1987_198726

theorem no_real_solutions : ¬∃ (a b : ℝ), 
  (a - 8 = b - a) ∧ (ab - b = b - a) ∧ (a * (1 + b) = 3 * b) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1987_198726


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l1987_198790

theorem composite_sum_of_powers (a b c d m n : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a)
  (h_div : (a + b - c + d) ∣ (a * c + b * d))
  (h_m_pos : 0 < m)
  (h_n_odd : n % 2 = 1) :
  ∃ (k : ℕ), k > 1 ∧ k ∣ (a^n * b^m + c^m * d^n) :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_powers_l1987_198790


namespace NUMINAMATH_CALUDE_triangle_circumscribed_circle_diameter_l1987_198737

/-- Given a triangle with one side of 12 inches and the opposite angle of 30°,
    the diameter of the circumscribed circle is 24 inches. -/
theorem triangle_circumscribed_circle_diameter
  (side : ℝ) (angle : ℝ) :
  side = 12 →
  angle = 30 * π / 180 →
  side / Real.sin angle = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumscribed_circle_diameter_l1987_198737


namespace NUMINAMATH_CALUDE_azalea_profit_l1987_198780

/-- Calculates the shearer's payment based on the amount of wool produced -/
def shearer_payment (wool_amount : ℕ) : ℕ :=
  1000 + 
  (if wool_amount > 1000 then 1500 else 0) + 
  (if wool_amount > 2000 then (wool_amount - 2000) / 2 else 0)

/-- Calculates the revenue from wool sales based on quality distribution -/
def wool_revenue (total_wool : ℕ) : ℕ :=
  (total_wool / 2) * 30 +  -- High-quality
  (total_wool * 3 / 10) * 20 +  -- Medium-quality
  (total_wool / 5) * 10  -- Low-quality

theorem azalea_profit :
  let total_wool := 2400
  let revenue := wool_revenue total_wool
  let payment := shearer_payment total_wool
  revenue - payment = 52500 := by sorry

end NUMINAMATH_CALUDE_azalea_profit_l1987_198780


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l1987_198797

/-- Given a tetrahedron with face areas S₁, S₂, S₃, S₄, and inscribed sphere radius R,
    the volume V is (1/3)(S₁ + S₂ + S₃ + S₄)R -/
theorem tetrahedron_volume (S₁ S₂ S₃ S₄ R : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) (h₃ : S₃ > 0) (h₄ : S₄ > 0) (hR : R > 0) :
  ∃ V : ℝ, V = (1/3) * (S₁ + S₂ + S₃ + S₄) * R := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l1987_198797


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1987_198779

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_first_term 
  (a : ℕ → ℚ) 
  (h_geometric : is_geometric_sequence a) 
  (h_third_term : a 2 = 8)
  (h_fifth_term : a 4 = 27 / 4) :
  a 0 = 256 / 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1987_198779


namespace NUMINAMATH_CALUDE_freely_falling_body_time_l1987_198720

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 9.808

/-- The additional distance fallen in meters -/
def additional_distance : ℝ := 49.34

/-- The additional time of fall in seconds -/
def additional_time : ℝ := 1.3

/-- The initial time of fall in seconds -/
def initial_time : ℝ := 7.088

theorem freely_falling_body_time :
  g * (initial_time * additional_time + 0.5 * additional_time^2) = additional_distance := by
  sorry

end NUMINAMATH_CALUDE_freely_falling_body_time_l1987_198720


namespace NUMINAMATH_CALUDE_rectangle_area_with_circles_l1987_198767

/-- The area of a rectangle surrounded by four circles -/
theorem rectangle_area_with_circles (r : ℝ) (h1 : r = 3) : ∃ (length width : ℝ),
  length = 2 * r * 2 ∧ 
  width = 2 * r ∧
  length * width = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_circles_l1987_198767


namespace NUMINAMATH_CALUDE_simplify_expression_l1987_198748

theorem simplify_expression : (18 * 10^9) / (6 * 10^4) = 300000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1987_198748


namespace NUMINAMATH_CALUDE_ash_cloud_ratio_l1987_198786

/-- Given a volcanic eruption where ashes are shot into the sky, this theorem proves
    the ratio of the ash cloud's diameter to the eruption height. -/
theorem ash_cloud_ratio (eruption_height : ℝ) (cloud_radius : ℝ) 
    (h1 : eruption_height = 300)
    (h2 : cloud_radius = 2700) : 
    (2 * cloud_radius) / eruption_height = 18 := by
  sorry

end NUMINAMATH_CALUDE_ash_cloud_ratio_l1987_198786


namespace NUMINAMATH_CALUDE_function_inequality_l1987_198758

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * (deriv f x) > -f x) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1987_198758


namespace NUMINAMATH_CALUDE_jason_work_experience_l1987_198755

/-- Calculates the total work experience in months given years as bartender and years and months as manager -/
def total_work_experience (bartender_years : ℕ) (manager_years : ℕ) (manager_months : ℕ) : ℕ := 
  bartender_years * 12 + manager_years * 12 + manager_months

/-- Proves that Jason's total work experience is 150 months -/
theorem jason_work_experience : 
  total_work_experience 9 3 6 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jason_work_experience_l1987_198755


namespace NUMINAMATH_CALUDE_ratio_problem_l1987_198756

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a * b * c / (d * e * f) = 1.875)
  (h2 : a / b = 5 / 2)
  (h3 : b / c = 1 / 2)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  c / d = 0.375 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l1987_198756


namespace NUMINAMATH_CALUDE_exactly_one_white_and_all_white_mutually_exclusive_but_not_complementary_l1987_198773

/-- Represents the outcome of drawing balls from a bag -/
inductive BallDraw
  | oneWhite
  | twoWhite
  | threeWhite
  | allBlack

/-- The set of all possible outcomes when drawing 3 balls from a bag with 3 white and 4 black balls -/
def allOutcomes : Set BallDraw := {BallDraw.oneWhite, BallDraw.twoWhite, BallDraw.threeWhite, BallDraw.allBlack}

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite : Set BallDraw := {BallDraw.oneWhite}

/-- The event of drawing all white balls -/
def allWhite : Set BallDraw := {BallDraw.threeWhite}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set BallDraw) : Prop := A ∩ B = ∅

/-- Two events are complementary if their union is the set of all outcomes -/
def complementary (A B : Set BallDraw) : Prop := A ∪ B = allOutcomes

theorem exactly_one_white_and_all_white_mutually_exclusive_but_not_complementary :
  mutuallyExclusive exactlyOneWhite allWhite ∧ ¬complementary exactlyOneWhite allWhite :=
sorry

end NUMINAMATH_CALUDE_exactly_one_white_and_all_white_mutually_exclusive_but_not_complementary_l1987_198773


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1987_198739

theorem inequality_equivalence (x y : ℝ) : 
  y^2 - x*y < 0 ↔ (0 < y ∧ y < x) ∨ (y < x ∧ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1987_198739


namespace NUMINAMATH_CALUDE_jane_hector_meeting_l1987_198721

/-- Represents the points around the block area -/
inductive Point := | A | B | C | D | E

/-- The total distance around the block area -/
def total_distance : ℕ := 24

/-- Hector's walking speed -/
def hector_speed : ℝ := 1

/-- Jane's walking speed -/
def jane_speed : ℝ := 3 * hector_speed

/-- The distance walked by Hector when they meet -/
def hector_distance : ℝ := 6

/-- The distance walked by Jane when they meet -/
def jane_distance : ℝ := 18

/-- The point where Jane and Hector meet -/
def meeting_point : Point := Point.C

theorem jane_hector_meeting :
  (jane_speed = 3 * hector_speed) →
  (hector_distance + jane_distance = total_distance) →
  (jane_distance = 3 * hector_distance) →
  meeting_point = Point.C :=
by sorry

end NUMINAMATH_CALUDE_jane_hector_meeting_l1987_198721


namespace NUMINAMATH_CALUDE_min_people_for_valid_seating_l1987_198736

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone. -/
def valid_seating (table : CircularTable) : Prop :=
  ∀ i : ℕ, i < table.total_chairs → ∃ j : ℕ, j < table.total_chairs ∧ 
    (((i + 1) % table.total_chairs = j) ∨ ((i + table.total_chairs - 1) % table.total_chairs = j))

/-- The main theorem stating the minimum number of people required for a valid seating arrangement. -/
theorem min_people_for_valid_seating :
  ∃ (table : CircularTable), table.total_chairs = 100 ∧ 
    valid_seating table ∧ table.seated_people = 25 ∧
    (∀ (smaller_table : CircularTable), smaller_table.total_chairs = 100 → 
      valid_seating smaller_table → smaller_table.seated_people ≥ 25) :=
sorry

end NUMINAMATH_CALUDE_min_people_for_valid_seating_l1987_198736
