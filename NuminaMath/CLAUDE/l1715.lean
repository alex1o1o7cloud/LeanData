import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l1715_171590

theorem simplify_expression (a : ℝ) (h : a ≠ -1) :
  a - 1 + 1 / (a + 1) = a^2 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1715_171590


namespace NUMINAMATH_CALUDE_min_value_rational_function_l1715_171504

theorem min_value_rational_function (x : ℤ) (h : x > 10) :
  (4 * x^2) / (x - 10) ≥ 160 ∧
  ((4 * x^2) / (x - 10) = 160 ↔ x = 20) :=
by sorry

end NUMINAMATH_CALUDE_min_value_rational_function_l1715_171504


namespace NUMINAMATH_CALUDE_product_remainder_l1715_171505

theorem product_remainder (x y : ℤ) 
  (hx : x % 792 = 62) 
  (hy : y % 528 = 82) : 
  (x * y) % 66 = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1715_171505


namespace NUMINAMATH_CALUDE_equation_solution_l1715_171550

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 1 ∧ (x^2 - x + 2) / (x - 1) = x + 3 ∧ x = 5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1715_171550


namespace NUMINAMATH_CALUDE_principal_is_250_l1715_171570

/-- Proves that the principal is 250 given the conditions of the problem -/
theorem principal_is_250 (P : ℝ) (I : ℝ) : 
  I = P * 0.04 * 8 →  -- Simple interest formula for 4% per annum over 8 years
  I = P - 170 →       -- Interest is 170 less than the principal
  P = 250 := by
sorry

end NUMINAMATH_CALUDE_principal_is_250_l1715_171570


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1715_171537

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  -- The side length of the equilateral triangle
  side_length : ℝ
  -- The perpendicular distances from the point to each side
  dist_to_side1 : ℝ
  dist_to_side2 : ℝ
  dist_to_side3 : ℝ
  -- Ensure the triangle is equilateral and the point is inside
  is_equilateral : side_length > 0
  point_inside : dist_to_side1 > 0 ∧ dist_to_side2 > 0 ∧ dist_to_side3 > 0

/-- The theorem to be proved -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangleWithPoint) 
  (h1 : triangle.dist_to_side1 = 2) 
  (h2 : triangle.dist_to_side2 = 3) 
  (h3 : triangle.dist_to_side3 = 4) : 
  triangle.side_length = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1715_171537


namespace NUMINAMATH_CALUDE_rebecca_egg_marble_difference_l1715_171552

/-- Given that Rebecca has 20 eggs and 6 marbles, prove that she has 14 more eggs than marbles. -/
theorem rebecca_egg_marble_difference :
  let eggs : ℕ := 20
  let marbles : ℕ := 6
  eggs - marbles = 14 := by sorry

end NUMINAMATH_CALUDE_rebecca_egg_marble_difference_l1715_171552


namespace NUMINAMATH_CALUDE_total_amount_proof_l1715_171517

/-- The total amount shared among p, q, and r -/
def total_amount : ℝ := 5400.000000000001

/-- The amount r has -/
def r_amount : ℝ := 3600.0000000000005

/-- Theorem stating that given r has two-thirds of the total amount and r's amount is 3600.0000000000005,
    the total amount is 5400.000000000001 -/
theorem total_amount_proof :
  (2 / 3 : ℝ) * total_amount = r_amount →
  total_amount = 5400.000000000001 := by
sorry

end NUMINAMATH_CALUDE_total_amount_proof_l1715_171517


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_25_l1715_171502

/-- The area of the quadrilateral formed by the lines y=8, y=x+3, y=-x+3, and x=5 -/
def quadrilateralArea : ℝ := 25

/-- Line y = 8 -/
def line1 (x : ℝ) : ℝ := 8

/-- Line y = x + 3 -/
def line2 (x : ℝ) : ℝ := x + 3

/-- Line y = -x + 3 -/
def line3 (x : ℝ) : ℝ := -x + 3

/-- Line x = 5 -/
def line4 : ℝ := 5

theorem quadrilateral_area_is_25 : quadrilateralArea = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_25_l1715_171502


namespace NUMINAMATH_CALUDE_product_properties_l1715_171511

-- Define the range of two-digit numbers
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define the range of three-digit numbers
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define the number of digits in a natural number
def NumDigits (n : ℕ) : ℕ := (Nat.log 10 n).succ

-- Define approximate equality
def ApproxEqual (x y : ℕ) (ε : ℕ) : Prop := (x : ℤ) - (y : ℤ) ≤ ε ∧ (y : ℤ) - (x : ℤ) ≤ ε

theorem product_properties :
  (NumDigits (52 * 403) = 5) ∧
  (ApproxEqual (52 * 403) 20000 1000) ∧
  (∀ a b, ThreeDigitNumber a → TwoDigitNumber b →
    (NumDigits (a * b) = 4 ∨ NumDigits (a * b) = 5)) :=
by sorry

end NUMINAMATH_CALUDE_product_properties_l1715_171511


namespace NUMINAMATH_CALUDE_water_level_lowered_l1715_171580

/-- Proves that removing 4500 gallons of water from a 60ft by 20ft pool lowers the water level by 6 inches -/
theorem water_level_lowered (pool_length pool_width : ℝ) 
  (water_removed : ℝ) (conversion_factor : ℝ) :
  pool_length = 60 →
  pool_width = 20 →
  water_removed = 4500 →
  conversion_factor = 7.5 →
  (water_removed / conversion_factor) / (pool_length * pool_width) * 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_water_level_lowered_l1715_171580


namespace NUMINAMATH_CALUDE_nedy_crackers_l1715_171599

/-- The number of cracker packs Nedy ate from Monday to Thursday -/
def monday_to_thursday : ℕ := 8

/-- The number of cracker packs Nedy ate on Friday -/
def friday : ℕ := 2 * monday_to_thursday

/-- The total number of cracker packs Nedy ate from Monday to Friday -/
def total : ℕ := monday_to_thursday + friday

theorem nedy_crackers : total = 24 := by sorry

end NUMINAMATH_CALUDE_nedy_crackers_l1715_171599


namespace NUMINAMATH_CALUDE_exists_initial_points_for_82_l1715_171519

/-- The function that calculates the number of points after one application of the procedure -/
def points_after_one_procedure (n : ℕ) : ℕ := 3 * n - 2

/-- The function that calculates the number of points after two applications of the procedure -/
def points_after_two_procedures (n : ℕ) : ℕ := 9 * n - 8

/-- Theorem stating that there exists an initial number of points that results in 82 points after two procedures -/
theorem exists_initial_points_for_82 : ∃ n : ℕ, points_after_two_procedures n = 82 := by
  sorry

end NUMINAMATH_CALUDE_exists_initial_points_for_82_l1715_171519


namespace NUMINAMATH_CALUDE_bushes_for_sixty_zucchinis_l1715_171536

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 12

/-- The number of containers of blueberries that can be traded for 3 zucchinis -/
def containers_per_trade : ℕ := 8

/-- The number of zucchinis received in one trade -/
def zucchinis_per_trade : ℕ := 3

/-- The target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- The function to calculate the number of bushes needed for a given number of zucchinis -/
def bushes_needed (zucchinis : ℕ) : ℕ :=
  (zucchinis * containers_per_trade + containers_per_bush * zucchinis_per_trade - 1) / 
  (containers_per_bush * zucchinis_per_trade)

theorem bushes_for_sixty_zucchinis :
  bushes_needed target_zucchinis = 14 := by
  sorry

end NUMINAMATH_CALUDE_bushes_for_sixty_zucchinis_l1715_171536


namespace NUMINAMATH_CALUDE_point_trajectory_l1715_171586

/-- The trajectory of a point satisfying a specific equation -/
theorem point_trajectory (x y : ℝ) :
  (Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) →
  ((x^2 / 16 - y^2 / 9 = 1) ∧ (x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_trajectory_l1715_171586


namespace NUMINAMATH_CALUDE_variance_implies_stability_l1715_171571

-- Define a structure for a data set
structure DataSet where
  variance : ℝ
  stability : ℝ

-- Define a relation for comparing stability
def more_stable (a b : DataSet) : Prop :=
  a.stability > b.stability

-- Theorem statement
theorem variance_implies_stability (a b : DataSet) 
  (h : a.variance < b.variance) : more_stable a b :=
sorry

end NUMINAMATH_CALUDE_variance_implies_stability_l1715_171571


namespace NUMINAMATH_CALUDE_shoe_tying_time_difference_l1715_171573

theorem shoe_tying_time_difference (jack_shoe_time toddler_count total_time : ℕ) :
  jack_shoe_time = 4 →
  toddler_count = 2 →
  total_time = 18 →
  (total_time - jack_shoe_time) / toddler_count - jack_shoe_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_shoe_tying_time_difference_l1715_171573


namespace NUMINAMATH_CALUDE_cos_315_degrees_l1715_171544

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l1715_171544


namespace NUMINAMATH_CALUDE_max_distance_line_equation_l1715_171513

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Returns true if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Returns true if a point (x, y) is on the given line --/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- Returns the distance between two parallel lines --/
noncomputable def distance_between_parallel_lines (l1 l2 : Line) : ℝ :=
  sorry

/-- The theorem to be proved --/
theorem max_distance_line_equation (l1 l2 : Line) (A B : ℝ × ℝ) :
  are_parallel l1 l2 →
  point_on_line l1 A.1 A.2 →
  point_on_line l2 B.1 B.2 →
  A = (1, 3) →
  B = (2, 4) →
  (∀ l1' l2' : Line, are_parallel l1' l2' →
    point_on_line l1' A.1 A.2 →
    point_on_line l2' B.1 B.2 →
    distance_between_parallel_lines l1' l2' ≤ distance_between_parallel_lines l1 l2) →
  l1 = { slope := -1, y_intercept := 4 } :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_equation_l1715_171513


namespace NUMINAMATH_CALUDE_trigonometric_equality_l1715_171548

theorem trigonometric_equality : 
  3.427 * Real.cos (50 * π / 180) + 
  8 * Real.cos (200 * π / 180) * Real.cos (220 * π / 180) * Real.cos (80 * π / 180) = 
  2 * Real.sin (65 * π / 180) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l1715_171548


namespace NUMINAMATH_CALUDE_service_cost_is_correct_l1715_171565

/-- Represents the service cost per vehicle at a fuel station. -/
def service_cost_per_vehicle : ℝ := 2.30

/-- Represents the cost of fuel per liter. -/
def fuel_cost_per_liter : ℝ := 0.70

/-- Represents the number of mini-vans. -/
def num_mini_vans : ℕ := 4

/-- Represents the number of trucks. -/
def num_trucks : ℕ := 2

/-- Represents the total cost for all vehicles. -/
def total_cost : ℝ := 396

/-- Represents the capacity of a mini-van's fuel tank in liters. -/
def mini_van_tank_capacity : ℝ := 65

/-- Represents the percentage by which a truck's tank is larger than a mini-van's tank. -/
def truck_tank_percentage : ℝ := 120

/-- Theorem stating that the service cost per vehicle is $2.30 given the problem conditions. -/
theorem service_cost_is_correct :
  let truck_tank_capacity := mini_van_tank_capacity * (1 + truck_tank_percentage / 100)
  let total_fuel_volume := num_mini_vans * mini_van_tank_capacity + num_trucks * truck_tank_capacity
  let total_fuel_cost := total_fuel_volume * fuel_cost_per_liter
  let total_service_cost := total_cost - total_fuel_cost
  service_cost_per_vehicle = total_service_cost / (num_mini_vans + num_trucks) := by
  sorry


end NUMINAMATH_CALUDE_service_cost_is_correct_l1715_171565


namespace NUMINAMATH_CALUDE_chocolate_bars_left_chocolate_problem_l1715_171522

theorem chocolate_bars_left (initial_bars : ℕ) 
  (thomas_and_friends : ℕ) (piper_reduction : ℕ) 
  (friend_return : ℕ) : ℕ :=
  let thomas_take := initial_bars / 4
  let friend_take := thomas_take / thomas_and_friends
  let total_taken := thomas_take - friend_return
  let piper_take := total_taken - piper_reduction
  initial_bars - total_taken - piper_take

theorem chocolate_problem : 
  chocolate_bars_left 200 5 5 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_left_chocolate_problem_l1715_171522


namespace NUMINAMATH_CALUDE_missing_number_is_1745_l1715_171520

def known_numbers : List ℕ := [744, 747, 748, 749, 752, 752, 753, 755, 755]

theorem missing_number_is_1745 :
  let total_count : ℕ := 10
  let average : ℕ := 750
  let sum_known : ℕ := known_numbers.sum
  let missing_number : ℕ := total_count * average - sum_known
  missing_number = 1745 := by sorry

end NUMINAMATH_CALUDE_missing_number_is_1745_l1715_171520


namespace NUMINAMATH_CALUDE_abc_sum_problem_l1715_171556

theorem abc_sum_problem (a b c d : ℝ) 
  (eq1 : a + b + c = 6)
  (eq2 : a + b + d = 2)
  (eq3 : a + c + d = 3)
  (eq4 : b + c + d = -3) :
  a * b + c * d = -37 / 9 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_problem_l1715_171556


namespace NUMINAMATH_CALUDE_points_above_line_t_range_l1715_171500

def P (t : ℝ) : ℝ × ℝ := (1, t)
def Q (t : ℝ) : ℝ × ℝ := (t^2, t - 1)

def above_line (p : ℝ × ℝ) : Prop := p.1 + p.2 - 1 > 0

theorem points_above_line_t_range :
  ∀ t : ℝ, (above_line (P t) ∧ above_line (Q t)) ↔ t > 1 := by
sorry

end NUMINAMATH_CALUDE_points_above_line_t_range_l1715_171500


namespace NUMINAMATH_CALUDE_alyssa_cut_roses_l1715_171553

/-- Represents the number of roses Alyssa cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Proves that Alyssa cut 11 roses given the initial and final number of roses -/
theorem alyssa_cut_roses : roses_cut 3 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_cut_roses_l1715_171553


namespace NUMINAMATH_CALUDE_car_waiting_time_l1715_171568

/-- Proves that a car waiting for a cyclist to catch up after 18 minutes must have initially waited 4.5 minutes -/
theorem car_waiting_time 
  (cyclist_speed : ℝ) 
  (car_speed : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : cyclist_speed = 15) 
  (h2 : car_speed = 60) 
  (h3 : catch_up_time = 18 / 60) : 
  let relative_speed := car_speed - cyclist_speed
  let distance := cyclist_speed * catch_up_time
  let initial_wait_time := distance / car_speed
  initial_wait_time * 60 = 4.5 := by sorry

end NUMINAMATH_CALUDE_car_waiting_time_l1715_171568


namespace NUMINAMATH_CALUDE_find_number_l1715_171561

theorem find_number : ∃ X : ℝ, (50 : ℝ) = 0.2 * X + 47 ∧ X = 15 := by sorry

end NUMINAMATH_CALUDE_find_number_l1715_171561


namespace NUMINAMATH_CALUDE_special_function_property_l1715_171575

/-- A function from positive reals to positive reals satisfying the given condition -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x y, x > 0 → y > 0 → f (x + f y) ≥ f (x + y) + f y)

/-- The main theorem -/
theorem special_function_property (f : ℝ → ℝ) (h : SpecialFunction f) :
  ∀ x > 0, f x > x :=
by sorry

end NUMINAMATH_CALUDE_special_function_property_l1715_171575


namespace NUMINAMATH_CALUDE_smallest_satisfying_arrangement_l1715_171593

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_guests : ℕ

/-- Checks if a seating arrangement satisfies the condition -/
def satisfies_condition (seating : CircularSeating) : Prop :=
  ∀ (i : ℕ), i < seating.total_chairs →
    ∃ (j : ℕ), j < seating.seated_guests ∧
      (i % (seating.total_chairs / seating.seated_guests) = 0 ∨
       (i + 1) % (seating.total_chairs / seating.seated_guests) = 0)

/-- The main theorem to be proved -/
theorem smallest_satisfying_arrangement :
  ∀ (n : ℕ), n < 20 →
    ¬(satisfies_condition { total_chairs := 120, seated_guests := n }) ∧
    satisfies_condition { total_chairs := 120, seated_guests := 20 } :=
by sorry


end NUMINAMATH_CALUDE_smallest_satisfying_arrangement_l1715_171593


namespace NUMINAMATH_CALUDE_sum_factorials_mod_12_l1715_171582

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_mod_12 :
  sum_factorials 7 % 12 = (factorial 1 + factorial 2 + factorial 3) % 12 :=
sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_12_l1715_171582


namespace NUMINAMATH_CALUDE_part_one_part_two_l1715_171515

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - (a + 1/a)*x + 1 < 0

def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Part 1
theorem part_one (x : ℝ) (h1 : p x 2) (h2 : q x) : 1 ≤ x ∧ x < 2 :=
sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, q x → p x a) (h_not_sufficient : ¬(∀ x, p x a → q x)) : 3 < a :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1715_171515


namespace NUMINAMATH_CALUDE_floor_times_self_equals_72_l1715_171574

theorem floor_times_self_equals_72 (x : ℝ) :
  x > 0 ∧ ⌊x⌋ * x = 72 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_equals_72_l1715_171574


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1715_171554

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  ∃ (max : ℝ), max = 72.25 ∧ 
  ∀ (a b : ℝ), a + b = 5 → 
    a^5*b + a^4*b + a^3*b + a*b + a*b^2 + a*b^3 + a*b^5 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1715_171554


namespace NUMINAMATH_CALUDE_football_players_count_l1715_171510

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 36)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 7) :
  total - neither - (tennis - both) = 26 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l1715_171510


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1715_171508

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n - (250 : ℝ)^(1/3)| ≥ |(6 : ℝ) - (250 : ℝ)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1715_171508


namespace NUMINAMATH_CALUDE_paul_filled_six_bags_saturday_l1715_171592

/-- The number of bags Paul filled on Saturday -/
def bags_saturday : ℕ := sorry

/-- The number of bags Paul filled on Sunday -/
def bags_sunday : ℕ := 3

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 8

/-- The total number of cans collected -/
def total_cans : ℕ := 72

/-- Theorem stating that Paul filled 6 bags on Saturday -/
theorem paul_filled_six_bags_saturday : 
  bags_saturday = 6 := by sorry

end NUMINAMATH_CALUDE_paul_filled_six_bags_saturday_l1715_171592


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1715_171523

theorem solution_set_inequality (x : ℝ) : 
  (2 * x) / (x + 1) ≤ 1 ↔ x ∈ Set.Ioc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1715_171523


namespace NUMINAMATH_CALUDE_chicken_egg_production_l1715_171555

theorem chicken_egg_production 
  (num_chickens : ℕ) 
  (price_per_dozen : ℚ) 
  (total_revenue : ℚ) 
  (num_weeks : ℕ) 
  (h1 : num_chickens = 46)
  (h2 : price_per_dozen = 3)
  (h3 : total_revenue = 552)
  (h4 : num_weeks = 8) :
  (total_revenue / (price_per_dozen / 12) / num_weeks / num_chickens : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_egg_production_l1715_171555


namespace NUMINAMATH_CALUDE_negation_of_existence_l1715_171579

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - 2*a*x - 1 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 - 2*a*x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1715_171579


namespace NUMINAMATH_CALUDE_unique_reverse_half_ceiling_l1715_171527

/-- Function that reverses the digits of an integer -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Ceiling function -/
def ceil (x : ℚ) : ℕ := sorry

theorem unique_reverse_half_ceiling :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 10000 ∧ reverse_digits n = ceil (n / 2) ∧ n = 7993 := by sorry

end NUMINAMATH_CALUDE_unique_reverse_half_ceiling_l1715_171527


namespace NUMINAMATH_CALUDE_bus_journey_max_time_l1715_171525

/-- Represents the transportation options available to Jenny --/
inductive TransportOption
  | Bus
  | Walk
  | Bike
  | Carpool
  | Train

/-- Calculates the total time for a given transportation option --/
def total_time (option : TransportOption) : ℝ :=
  match option with
  | .Bus => 30 + 15  -- Bus time + walking time
  | .Walk => 30
  | .Bike => 20
  | .Carpool => 25
  | .Train => 45

/-- Jenny's walking speed in miles per minute --/
def walking_speed : ℝ := 0.05

/-- The maximum time allowed for any transportation option --/
def max_allowed_time : ℝ := 45

theorem bus_journey_max_time :
  ∀ (option : TransportOption),
    total_time TransportOption.Bus ≤ max_allowed_time ∧
    total_time TransportOption.Bus = total_time option →
    30 = max_allowed_time - (0.75 / walking_speed) := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_max_time_l1715_171525


namespace NUMINAMATH_CALUDE_power_three_mod_ten_l1715_171546

theorem power_three_mod_ten : 3^24 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_ten_l1715_171546


namespace NUMINAMATH_CALUDE_M_intersect_N_l1715_171584

def M : Set ℕ := {1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a - 1}

theorem M_intersect_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1715_171584


namespace NUMINAMATH_CALUDE_equation_solution_l1715_171542

theorem equation_solution :
  let f (x : ℝ) := (7 * x + 3) / (3 * x^2 + 7 * x - 6)
  let g (x : ℝ) := (3 * x) / (3 * x - 2)
  ∀ x : ℝ, f x = g x ↔ x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1715_171542


namespace NUMINAMATH_CALUDE_carlos_singles_percentage_l1715_171597

/-- Represents the hit statistics for Carlos during the baseball season -/
structure HitStats :=
  (total_hits : ℕ)
  (home_runs : ℕ)
  (triples : ℕ)
  (doubles : ℕ)
  (strikeouts : ℕ)

/-- Calculates the percentage of singles among successful hits -/
def percentage_singles (stats : HitStats) : ℚ :=
  let successful_hits := stats.total_hits - stats.strikeouts
  let non_single_hits := stats.home_runs + stats.triples + stats.doubles
  let singles := successful_hits - non_single_hits
  (singles : ℚ) / (successful_hits : ℚ) * 100

/-- The hit statistics for Carlos -/
def carlos_stats : HitStats :=
  { total_hits := 50
  , home_runs := 4
  , triples := 2
  , doubles := 8
  , strikeouts := 6 }

/-- Theorem stating that the percentage of singles for Carlos is approximately 68.18% -/
theorem carlos_singles_percentage :
  abs (percentage_singles carlos_stats - 68.18) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_carlos_singles_percentage_l1715_171597


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1715_171576

/-- The distance between the vertices of the hyperbola x^2/16 - y^2/9 = 1 is 8 -/
theorem hyperbola_vertices_distance : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/16 - y^2/9 = 1
  ∃ (v₁ v₂ : ℝ × ℝ), 
    (h v₁.1 v₁.2 ∧ h v₂.1 v₂.2) ∧ 
    (v₁.2 = 0 ∧ v₂.2 = 0) ∧
    (v₁.1 = -v₂.1) ∧
    abs (v₁.1 - v₂.1) = 8 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1715_171576


namespace NUMINAMATH_CALUDE_total_additions_in_half_hour_l1715_171581

/-- The number of additions a single computer can perform per second -/
def additions_per_second : ℕ := 15000

/-- The number of computers -/
def num_computers : ℕ := 3

/-- The number of seconds in half an hour -/
def seconds_in_half_hour : ℕ := 1800

/-- The total number of additions performed by all computers in half an hour -/
def total_additions : ℕ := additions_per_second * num_computers * seconds_in_half_hour

theorem total_additions_in_half_hour :
  total_additions = 81000000 := by sorry

end NUMINAMATH_CALUDE_total_additions_in_half_hour_l1715_171581


namespace NUMINAMATH_CALUDE_sqrt_two_cos_thirty_degrees_l1715_171514

theorem sqrt_two_cos_thirty_degrees : 
  Real.sqrt 2 * Real.cos (30 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_cos_thirty_degrees_l1715_171514


namespace NUMINAMATH_CALUDE_password_count_l1715_171572

def password_length : ℕ := 4
def available_digits : ℕ := 9  -- digits 0-6, 8-9

def total_passwords : ℕ := available_digits ^ password_length

def passwords_with_distinct_digits : ℕ := 
  (Nat.factorial available_digits) / (Nat.factorial (available_digits - password_length))

theorem password_count : 
  total_passwords - passwords_with_distinct_digits = 3537 := by
  sorry

end NUMINAMATH_CALUDE_password_count_l1715_171572


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1715_171567

def is_geometric_sequence (a : Fin 4 → ℝ) : Prop :=
  ∃ q : ℝ, ∀ i : Fin 3, a (i + 1) = a i * q

theorem geometric_sequence_property
  (a : Fin 4 → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 = Real.log (a 0 + a 1 + a 2))
  (h_a1 : a 0 > 1) :
  a 0 > a 2 ∧ a 1 < a 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1715_171567


namespace NUMINAMATH_CALUDE_game_score_theorem_l1715_171569

theorem game_score_theorem (a b : ℕ) (h1 : 0 < b) (h2 : b < a) (h3 : a < 1986)
  (h4 : ∀ x : ℕ, x ≥ 1986 → ∃ (m n : ℕ), x = m * a + n * b)
  (h5 : ¬∃ (m n : ℕ), 1985 = m * a + n * b)
  (h6 : ¬∃ (m n : ℕ), 663 = m * a + n * b) :
  a = 332 ∧ b = 7 := by
sorry

end NUMINAMATH_CALUDE_game_score_theorem_l1715_171569


namespace NUMINAMATH_CALUDE_append_two_to_three_digit_number_l1715_171559

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  is_valid : hundreds < 10 ∧ tens < 10 ∧ units < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNum (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Appends a digit to a number -/
def appendDigit (n : ℕ) (d : ℕ) : ℕ :=
  10 * n + d

theorem append_two_to_three_digit_number (n : ThreeDigitNumber) :
  appendDigit (ThreeDigitNumber.toNum n) 2 =
  1000 * n.hundreds + 100 * n.tens + 10 * n.units + 2 := by
  sorry

end NUMINAMATH_CALUDE_append_two_to_three_digit_number_l1715_171559


namespace NUMINAMATH_CALUDE_garden_area_l1715_171524

/-- Represents a rectangular garden with given properties -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  length_walk : length * 20 = 1000
  perimeter_walk : (length + width) * 2 * 8 = 1000

/-- The area of a rectangular garden with the given properties is 625 square meters -/
theorem garden_area (g : RectangularGarden) : g.length * g.width = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l1715_171524


namespace NUMINAMATH_CALUDE_avg_cost_rounded_to_13_l1715_171521

/- Define the number of pencils -/
def num_pencils : ℕ := 200

/- Define the cost of pencils in cents -/
def pencil_cost : ℕ := 1990

/- Define the shipping cost in cents -/
def shipping_cost : ℕ := 695

/- Define the function to calculate the average cost per pencil in cents -/
def avg_cost_per_pencil : ℚ :=
  (pencil_cost + shipping_cost : ℚ) / num_pencils

/- Define the function to round to the nearest whole number -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/- Theorem statement -/
theorem avg_cost_rounded_to_13 :
  round_to_nearest avg_cost_per_pencil = 13 := by
  sorry


end NUMINAMATH_CALUDE_avg_cost_rounded_to_13_l1715_171521


namespace NUMINAMATH_CALUDE_copper_percentage_in_first_alloy_l1715_171594

/-- Prove that the percentage of copper in the first alloy is 20% -/
theorem copper_percentage_in_first_alloy
  (final_mixture_weight : ℝ)
  (final_copper_percentage : ℝ)
  (first_alloy_weight : ℝ)
  (second_alloy_copper_percentage : ℝ)
  (h1 : final_mixture_weight = 100)
  (h2 : final_copper_percentage = 24.9)
  (h3 : first_alloy_weight = 30)
  (h4 : second_alloy_copper_percentage = 27)
  : ∃ (first_alloy_copper_percentage : ℝ),
    first_alloy_copper_percentage = 20 ∧
    (first_alloy_copper_percentage / 100) * first_alloy_weight +
    (second_alloy_copper_percentage / 100) * (final_mixture_weight - first_alloy_weight) =
    (final_copper_percentage / 100) * final_mixture_weight :=
by sorry

end NUMINAMATH_CALUDE_copper_percentage_in_first_alloy_l1715_171594


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l1715_171535

theorem fraction_sum_zero (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h : (a - d) / (b - c) + (b - d) / (c - a) + (c - d) / (a - b) = 0) :
  (a + d) / (b - c)^3 + (b + d) / (c - a)^3 + (c + d) / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l1715_171535


namespace NUMINAMATH_CALUDE_letter_150_is_B_l1715_171531

def repeating_pattern : ℕ → Char
  | n => match n % 4 with
    | 0 => 'D'
    | 1 => 'A'
    | 2 => 'B'
    | _ => 'C'

theorem letter_150_is_B : repeating_pattern 150 = 'B' := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_B_l1715_171531


namespace NUMINAMATH_CALUDE_race_start_relation_l1715_171587

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race conditions -/
structure RaceCondition where
  a : Runner
  b : Runner
  c : Runner
  race_length : ℝ
  a_c_start : ℝ
  b_c_start : ℝ

/-- Theorem stating the relation between the starts given by runners -/
theorem race_start_relation (cond : RaceCondition) 
  (h1 : cond.race_length = 1000)
  (h2 : cond.a_c_start = 600)
  (h3 : cond.b_c_start = 428.57) :
  ∃ (a_b_start : ℝ), a_b_start = 750 ∧ 
    (cond.race_length - a_b_start) / cond.race_length = 
    (cond.race_length - cond.b_c_start) / (cond.race_length - cond.a_c_start) :=
by sorry

end NUMINAMATH_CALUDE_race_start_relation_l1715_171587


namespace NUMINAMATH_CALUDE_f_increasing_f_sum_positive_l1715_171539

-- Define the function f(x) = x + x^3
def f (x : ℝ) : ℝ := x + x^3

-- Theorem 1: f is an increasing function
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

-- Theorem 2: For any a, b ∈ ℝ where a + b > 0, f(a) + f(b) > 0
theorem f_sum_positive (a b : ℝ) (h : a + b > 0) : f a + f b > 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_f_sum_positive_l1715_171539


namespace NUMINAMATH_CALUDE_boatman_distance_along_current_l1715_171532

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  stationary : ℝ
  against_current : ℝ
  current : ℝ
  along_current : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (speed time : ℝ) : ℝ := speed * time

/-- Theorem: The boatman travels 1 km along the current -/
theorem boatman_distance_along_current 
  (speed : BoatSpeed)
  (h1 : distance speed.against_current 4 = 4) -- 4 km against current in 4 hours
  (h2 : distance speed.stationary 3 = 6)      -- 6 km in stationary water in 3 hours
  (h3 : speed.current = speed.stationary - speed.against_current)
  (h4 : speed.along_current = speed.stationary + speed.current)
  : distance speed.along_current (1/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_boatman_distance_along_current_l1715_171532


namespace NUMINAMATH_CALUDE_total_spent_is_2100_l1715_171588

/-- Calculates the total amount spent on a computer setup -/
def total_spent (computer_cost monitor_peripheral_ratio original_video_card_cost new_video_card_ratio : ℚ) : ℚ :=
  computer_cost + 
  (monitor_peripheral_ratio * computer_cost) + 
  (new_video_card_ratio * original_video_card_cost - original_video_card_cost)

/-- Proves that the total amount spent is $2100 given the specified costs and ratios -/
theorem total_spent_is_2100 : 
  total_spent 1500 (1/5) 300 2 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_2100_l1715_171588


namespace NUMINAMATH_CALUDE_quadratic_negative_value_l1715_171543

theorem quadratic_negative_value (a b c : ℝ) :
  (∃ x : ℝ, x^2 + b*x + c = 0) →
  (∃ x : ℝ, a*x^2 + x + c = 0) →
  (∃ x : ℝ, a*x^2 + b*x + 1 = 0) →
  (∃ x : ℝ, a*x^2 + b*x + c < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_negative_value_l1715_171543


namespace NUMINAMATH_CALUDE_square_side_length_l1715_171534

-- Define the right triangle PQR
def triangle_PQR (PQ PR : ℝ) : Prop := PQ = 5 ∧ PR = 12

-- Define the square on the hypotenuse
def square_on_hypotenuse (s : ℝ) (PQ PR : ℝ) : Prop :=
  ∃ (x : ℝ), 
    s / (PQ^2 + PR^2).sqrt = x / PR ∧
    s / (PR - PQ * PR / (PQ^2 + PR^2).sqrt) = (PR - x) / PR

-- Theorem statement
theorem square_side_length (PQ PR s : ℝ) : 
  triangle_PQR PQ PR →
  square_on_hypotenuse s PQ PR →
  s = 96.205 / 20.385 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1715_171534


namespace NUMINAMATH_CALUDE_sin_cos_difference_45_15_l1715_171512

theorem sin_cos_difference_45_15 :
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) -
  Real.cos (45 * π / 180) * Real.sin (15 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_45_15_l1715_171512


namespace NUMINAMATH_CALUDE_geometric_mean_problem_l1715_171557

theorem geometric_mean_problem : 
  let a := 7 + 3 * Real.sqrt 5
  let b := 7 - 3 * Real.sqrt 5
  ∃ x : ℝ, x^2 = a * b ∧ (x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_problem_l1715_171557


namespace NUMINAMATH_CALUDE_excellent_set_properties_l1715_171516

-- Definition of an excellent set
def IsExcellentSet (M : Set ℝ) : Prop :=
  ∀ x y, x ∈ M → y ∈ M → (x + y) ∈ M ∧ (x - y) ∈ M

-- Theorem statement
theorem excellent_set_properties
  (A B : Set ℝ)
  (hA : IsExcellentSet A)
  (hB : IsExcellentSet B) :
  (IsExcellentSet (A ∩ B)) ∧
  (IsExcellentSet (A ∪ B) → (A ⊆ B ∨ B ⊆ A)) ∧
  (IsExcellentSet (A ∪ B) → IsExcellentSet (A ∩ B)) :=
by sorry

end NUMINAMATH_CALUDE_excellent_set_properties_l1715_171516


namespace NUMINAMATH_CALUDE_root_sum_quotient_l1715_171564

theorem root_sum_quotient (m₁ m₂ : ℝ) : 
  m₁^2 - 21*m₁ + 4 = 0 → 
  m₂^2 - 21*m₂ + 4 = 0 → 
  m₁ / m₂ + m₂ / m₁ = 108.25 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_quotient_l1715_171564


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1715_171558

/-- 
Given a rectangular plot where:
- The area is 21 times its breadth
- The difference between the length and breadth is 10 metres
This theorem proves that the breadth of the plot is 11 metres.
-/
theorem rectangular_plot_breadth (length width : ℝ) 
  (h1 : length * width = 21 * width) 
  (h2 : length - width = 10) : 
  width = 11 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1715_171558


namespace NUMINAMATH_CALUDE_max_squares_covered_l1715_171503

/-- Represents a square card with a given side length -/
structure Card where
  side : ℝ
  side_positive : side > 0

/-- Represents a square on a checkerboard -/
structure CheckerboardSquare where
  side : ℝ
  side_positive : side > 0

/-- Calculates the number of checkerboard squares covered by a card -/
def squares_covered (card : Card) (square : CheckerboardSquare) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a 1-inch checkerboard -/
theorem max_squares_covered :
  let card := Card.mk 2 (by norm_num)
  let square := CheckerboardSquare.mk 1 (by norm_num)
  ∀ n : ℕ, squares_covered card square ≤ n → n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_squares_covered_l1715_171503


namespace NUMINAMATH_CALUDE_original_flock_size_l1715_171560

/-- Represents a flock of sheep --/
structure Flock where
  rams : ℕ
  ewes : ℕ

/-- The original flock of sheep --/
def original_flock : Flock := sorry

/-- The flock after one ram runs away --/
def flock_minus_ram : Flock := 
  { rams := original_flock.rams - 1, ewes := original_flock.ewes }

/-- The flock after the ram returns and one ewe runs away --/
def flock_minus_ewe : Flock := 
  { rams := original_flock.rams, ewes := original_flock.ewes - 1 }

/-- The theorem to be proved --/
theorem original_flock_size : 
  (flock_minus_ram.rams : ℚ) / flock_minus_ram.ewes = 7 / 5 ∧
  (flock_minus_ewe.rams : ℚ) / flock_minus_ewe.ewes = 5 / 3 →
  original_flock.rams + original_flock.ewes = 25 := by
  sorry


end NUMINAMATH_CALUDE_original_flock_size_l1715_171560


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1715_171540

theorem infinite_geometric_series_first_term
  (r : ℚ)
  (S : ℚ)
  (h1 : r = 1 / 4)
  (h2 : S = 20)
  (h3 : S = a / (1 - r)) :
  a = 15 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1715_171540


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1715_171547

/-- The cost of a candy bar given initial and remaining amounts -/
theorem candy_bar_cost (initial_amount remaining_amount : ℕ) :
  initial_amount = 5 ∧ remaining_amount = 3 →
  initial_amount - remaining_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1715_171547


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l1715_171518

/-- A regular dodecahedron -/
structure Dodecahedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 20
  edge_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of randomly selecting two vertices that are endpoints of an edge -/
def edge_endpoint_probability (d : Dodecahedron) : ℚ :=
  (d.edges.card : ℚ) / (d.vertices.card.choose 2 : ℚ)

/-- The main theorem -/
theorem dodecahedron_edge_probability (d : Dodecahedron) :
  edge_endpoint_probability d = 3/19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l1715_171518


namespace NUMINAMATH_CALUDE_sausage_pepperoni_difference_l1715_171585

def pizza_problem (pepperoni ham sausage : ℕ) : Prop :=
  let total_slices : ℕ := 6
  let meat_per_slice : ℕ := 22
  pepperoni = 30 ∧
  ham = 2 * pepperoni ∧
  sausage > pepperoni ∧
  (pepperoni + ham + sausage) / total_slices = meat_per_slice

theorem sausage_pepperoni_difference :
  ∀ (pepperoni ham sausage : ℕ),
    pizza_problem pepperoni ham sausage →
    sausage - pepperoni = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sausage_pepperoni_difference_l1715_171585


namespace NUMINAMATH_CALUDE_raquel_has_40_dollars_l1715_171583

-- Define the amounts of money for each person
def raquel_money : ℝ := sorry
def nataly_money : ℝ := sorry
def tom_money : ℝ := sorry

-- State the theorem
theorem raquel_has_40_dollars :
  -- Conditions
  (tom_money = (1/4) * nataly_money) →
  (nataly_money = 3 * raquel_money) →
  (tom_money + nataly_money + raquel_money = 190) →
  -- Conclusion
  raquel_money = 40 := by
  sorry

end NUMINAMATH_CALUDE_raquel_has_40_dollars_l1715_171583


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1715_171578

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (3 - x) - Real.sqrt (x + 1) > (1 : ℝ) / 2 ↔ 
  -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8 :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1715_171578


namespace NUMINAMATH_CALUDE_fraction_of_boys_l1715_171563

theorem fraction_of_boys (total_students : ℕ) (girls_no_pets : ℕ) 
  (dog_owners_percent : ℚ) (cat_owners_percent : ℚ) :
  total_students = 30 →
  girls_no_pets = 8 →
  dog_owners_percent = 40 / 100 →
  cat_owners_percent = 20 / 100 →
  (17 : ℚ) / 30 = (total_students - (girls_no_pets / ((1 : ℚ) - dog_owners_percent - cat_owners_percent))) / total_students :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_boys_l1715_171563


namespace NUMINAMATH_CALUDE_max_removable_edges_in_complete_graph_l1715_171541

theorem max_removable_edges_in_complete_graph :
  ∀ (n : ℕ), n = 30 →
  ∃ (k : ℕ), k = 406 ∧
  (((n * (n - 1)) / 2) - k = n - 1) ∧
  k = ((n * (n - 1)) / 2) - (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_max_removable_edges_in_complete_graph_l1715_171541


namespace NUMINAMATH_CALUDE_total_weight_is_56_7_l1715_171509

/-- The total weight of five plastic rings in grams -/
def total_weight_in_grams : ℝ :=
  let orange_weight := 0.08333333333333333
  let purple_weight := 0.3333333333333333
  let white_weight := 0.4166666666666667
  let blue_weight := 0.5416666666666666
  let red_weight := 0.625
  let conversion_factor := 28.35
  (orange_weight + purple_weight + white_weight + blue_weight + red_weight) * conversion_factor

/-- Theorem stating that the total weight of the five plastic rings is 56.7 grams -/
theorem total_weight_is_56_7 : total_weight_in_grams = 56.7 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_56_7_l1715_171509


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1715_171591

theorem fixed_point_on_line (m : ℝ) : 
  (m - 1) * (7/2 : ℝ) - (m + 3) * (5/2 : ℝ) - (m - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1715_171591


namespace NUMINAMATH_CALUDE_y_minimized_at_b_over_2_l1715_171566

variable (a b : ℝ)

def y (x : ℝ) := (x - a)^2 + (x - b)^2 + 2*(a - b)*x

theorem y_minimized_at_b_over_2 :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y a b x_min ≤ y a b x ∧ x_min = b/2 :=
sorry

end NUMINAMATH_CALUDE_y_minimized_at_b_over_2_l1715_171566


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l1715_171528

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_specific :
  let a₁ : ℤ := -3
  let d : ℤ := 6
  let n : ℕ := 8
  let aₙ : ℤ := a₁ + (n - 1) * d
  aₙ = 39 →
  arithmetic_sequence_sum a₁ d n = 144 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l1715_171528


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_75_6300_l1715_171526

theorem gcd_lcm_sum_75_6300 : Nat.gcd 75 6300 + Nat.lcm 75 6300 = 6375 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_75_6300_l1715_171526


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1715_171596

theorem min_value_on_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 12 = 0) :
  ∃ (min : ℝ), (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' + 6*y' + 12 = 0 →
    |2*x' - y' - 2| ≥ min) ∧ min = 5 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1715_171596


namespace NUMINAMATH_CALUDE_distance_cos80_sin80_to_cos20_sin20_l1715_171562

/-- The distance between points (cos 80°, sin 80°) and (cos 20°, sin 20°) is 1. -/
theorem distance_cos80_sin80_to_cos20_sin20 : 
  let A : ℝ × ℝ := (Real.cos (80 * π / 180), Real.sin (80 * π / 180))
  let B : ℝ × ℝ := (Real.cos (20 * π / 180), Real.sin (20 * π / 180))
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_distance_cos80_sin80_to_cos20_sin20_l1715_171562


namespace NUMINAMATH_CALUDE_tom_rental_hours_l1715_171538

/-- Represents the rental fees and total amount paid --/
structure RentalInfo where
  baseFee : ℕ
  bikeHourlyFee : ℕ
  helmetFee : ℕ
  lockHourlyFee : ℕ
  totalPaid : ℕ

/-- Calculates the number of hours rented based on the rental information --/
def calculateHoursRented (info : RentalInfo) : ℕ :=
  ((info.totalPaid - info.baseFee - info.helmetFee) / (info.bikeHourlyFee + info.lockHourlyFee))

/-- Theorem stating that Tom rented the bike and accessories for 8 hours --/
theorem tom_rental_hours (info : RentalInfo) 
    (h1 : info.baseFee = 17)
    (h2 : info.bikeHourlyFee = 7)
    (h3 : info.helmetFee = 5)
    (h4 : info.lockHourlyFee = 2)
    (h5 : info.totalPaid = 95) : 
  calculateHoursRented info = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_rental_hours_l1715_171538


namespace NUMINAMATH_CALUDE_floor_slab_rate_l1715_171577

/-- Proves that for a rectangular room with given dimensions and total flooring cost,
    the rate per square meter is 900 Rs. -/
theorem floor_slab_rate (length width total_cost : ℝ) :
  length = 5 →
  width = 4.75 →
  total_cost = 21375 →
  total_cost / (length * width) = 900 := by
sorry

end NUMINAMATH_CALUDE_floor_slab_rate_l1715_171577


namespace NUMINAMATH_CALUDE_kelly_cheese_days_l1715_171551

/-- The number of weeks Kelly needs to cover -/
def weeks : ℕ := 4

/-- The number of packages of string cheese Kelly buys -/
def packages : ℕ := 2

/-- The number of string cheeses in each package -/
def cheeses_per_package : ℕ := 30

/-- The number of string cheeses the oldest child needs per day -/
def oldest_child_cheeses : ℕ := 2

/-- The number of string cheeses the youngest child needs per day -/
def youngest_child_cheeses : ℕ := 1

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: Kelly puts string cheeses in her kids' lunches 5 days per week -/
theorem kelly_cheese_days : 
  (packages * cheeses_per_package) / (oldest_child_cheeses + youngest_child_cheeses) / weeks = 5 := by
  sorry

end NUMINAMATH_CALUDE_kelly_cheese_days_l1715_171551


namespace NUMINAMATH_CALUDE_easier_decryption_with_more_unique_letters_l1715_171529

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem easier_decryption_with_more_unique_letters 
  (word1 : String) (word2 : String) 
  (h1 : word1 = "термометр") (h2 : word2 = "ремонт") :
  (unique_letters word2).card > (unique_letters word1).card :=
by sorry

end NUMINAMATH_CALUDE_easier_decryption_with_more_unique_letters_l1715_171529


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_two_l1715_171506

theorem fraction_zero_implies_x_two (x : ℝ) : 
  (x^2 - 4) / (x + 2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_two_l1715_171506


namespace NUMINAMATH_CALUDE_largest_six_digit_with_factorial_product_l1715_171501

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).foldl (· * ·) 1

theorem largest_six_digit_with_factorial_product :
  ∃ (n : ℕ), 
    100000 ≤ n ∧ 
    n ≤ 999999 ∧ 
    digit_product n = factorial 8 ∧
    ∀ (m : ℕ), 100000 ≤ m ∧ m ≤ 999999 ∧ digit_product m = factorial 8 → m ≤ n :=
by
  use 987542
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_with_factorial_product_l1715_171501


namespace NUMINAMATH_CALUDE_aaron_matthews_more_cows_l1715_171595

/-- Represents the number of cows each person has -/
structure CowCounts where
  aaron : ℕ
  matthews : ℕ
  marovich : ℕ

/-- The conditions of the problem -/
def cow_problem (c : CowCounts) : Prop :=
  c.aaron = 4 * c.matthews ∧
  c.matthews = 60 ∧
  c.aaron + c.matthews + c.marovich = 570

/-- The theorem to prove -/
theorem aaron_matthews_more_cows (c : CowCounts) 
  (h : cow_problem c) : c.aaron + c.matthews - c.marovich = 30 := by
  sorry


end NUMINAMATH_CALUDE_aaron_matthews_more_cows_l1715_171595


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1715_171598

theorem repeating_decimal_sum (b c : ℕ) : 
  b < 10 → c < 10 →
  (10 * b + c : ℚ) / 99 + (100 * c + 10 * b + c : ℚ) / 999 = 83 / 222 →
  b = 1 ∧ c = 1 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1715_171598


namespace NUMINAMATH_CALUDE_third_sum_third_term_ratio_l1715_171589

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ
  sum : ℕ → ℝ
  first_third_sum : a 1 + a 3 = 5/2
  second_fourth_sum : a 2 + a 4 = 5/4
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The theorem stating that S₃/a₃ = 6 for the given arithmetic progression -/
theorem third_sum_third_term_ratio (ap : ArithmeticProgression) :
  ap.sum 3 / ap.a 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_sum_third_term_ratio_l1715_171589


namespace NUMINAMATH_CALUDE_calculation_result_l1715_171530

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The result of the calculation in base 10 --/
def result : Rat :=
  (toBase10 [3, 1, 0, 2] 5 : Rat) / (toBase10 [1, 1] 3) -
  (toBase10 [4, 2, 1, 3] 6 : Rat) +
  (toBase10 [1, 2, 3, 4] 7 : Rat)

theorem calculation_result : result = 898.5 := by sorry

end NUMINAMATH_CALUDE_calculation_result_l1715_171530


namespace NUMINAMATH_CALUDE_least_marbles_count_l1715_171507

theorem least_marbles_count (n : ℕ) : n ≥ 402 →
  (n % 7 = 3 ∧ n % 4 = 2 ∧ n % 6 = 1) →
  n = 402 :=
by sorry

end NUMINAMATH_CALUDE_least_marbles_count_l1715_171507


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1715_171545

def polynomial (x : ℝ) : ℝ := 8*x^4 + 4*x^3 - 9*x^2 + 16*x - 28

def divisor (x : ℝ) : ℝ := 4*x - 12

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 695 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1715_171545


namespace NUMINAMATH_CALUDE_chessboard_constant_l1715_171549

/-- A function representing the numbers on an infinite chessboard. -/
def ChessboardFunction := ℤ × ℤ → ℝ

/-- The property that each number is the arithmetic mean of its four neighbors. -/
def IsMeanValue (f : ChessboardFunction) : Prop :=
  ∀ m n : ℤ, f (m, n) = (f (m+1, n) + f (m-1, n) + f (m, n+1) + f (m, n-1)) / 4

/-- The property that all values of the function are nonnegative. -/
def IsNonnegative (f : ChessboardFunction) : Prop :=
  ∀ m n : ℤ, 0 ≤ f (m, n)

/-- Theorem stating that a nonnegative function satisfying the mean value property is constant. -/
theorem chessboard_constant (f : ChessboardFunction) 
  (h_mean : IsMeanValue f) (h_nonneg : IsNonnegative f) : 
  ∃ c : ℝ, ∀ m n : ℤ, f (m, n) = c :=
sorry

end NUMINAMATH_CALUDE_chessboard_constant_l1715_171549


namespace NUMINAMATH_CALUDE_polynomial_division_problem_l1715_171533

theorem polynomial_division_problem (a : ℤ) : 
  (∃ p : Polynomial ℤ, (X^2 - 2*X + a) * p = X^13 + 2*X + 180) ↔ a = 3 :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_problem_l1715_171533
