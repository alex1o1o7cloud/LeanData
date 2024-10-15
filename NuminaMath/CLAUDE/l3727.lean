import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l3727_372739

/-- Given two rectangles A and B, prove the ratio of their sides -/
theorem rectangle_side_ratio 
  (a b c d : ℝ) 
  (h1 : a * b / (c * d) = 0.16) 
  (h2 : a / c = 2 / 5) : 
  b / d = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l3727_372739


namespace NUMINAMATH_CALUDE_f_negative_five_halves_l3727_372714

def f (x : ℝ) : ℝ := sorry

theorem f_negative_five_halves :
  (∀ x, f (-x) = -f x) →                     -- f is odd
  (∀ x, f (x + 2) = f x) →                   -- f has period 2
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2*x*(1 - x)) → -- f(x) = 2x(1-x) for 0 ≤ x ≤ 1
  f (-5/2) = -1/2 := by sorry

end NUMINAMATH_CALUDE_f_negative_five_halves_l3727_372714


namespace NUMINAMATH_CALUDE_prime_triple_uniqueness_l3727_372713

theorem prime_triple_uniqueness : 
  ∀ p : ℕ, p > 0 → Prime p → Prime (p + 2) → Prime (p + 4) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_triple_uniqueness_l3727_372713


namespace NUMINAMATH_CALUDE_yeast_growth_20_minutes_l3727_372702

/-- Represents the population growth of yeast cells over time -/
def yeast_population (initial_population : ℕ) (growth_factor : ℕ) (intervals : ℕ) : ℕ :=
  initial_population * growth_factor ^ intervals

theorem yeast_growth_20_minutes :
  let initial_population := 30
  let growth_factor := 3
  let intervals := 5
  yeast_population initial_population growth_factor intervals = 7290 := by sorry

end NUMINAMATH_CALUDE_yeast_growth_20_minutes_l3727_372702


namespace NUMINAMATH_CALUDE_work_completion_time_l3727_372752

/-- Proves the time taken to complete a work when two people work together -/
theorem work_completion_time (rahul_rate meena_rate : ℚ) 
  (hrahul : rahul_rate = 1 / 5)
  (hmeena : meena_rate = 1 / 10) :
  1 / (rahul_rate + meena_rate) = 10 / 3 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3727_372752


namespace NUMINAMATH_CALUDE_two_car_garage_count_l3727_372707

theorem two_car_garage_count (total : ℕ) (pool : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 90 → pool = 40 → both = 35 → neither = 35 → 
  ∃ (garage : ℕ), garage = 50 ∧ garage + pool - both = total - neither :=
by
  sorry

end NUMINAMATH_CALUDE_two_car_garage_count_l3727_372707


namespace NUMINAMATH_CALUDE_shara_shell_collection_l3727_372797

/-- Calculates the total number of shells Shara has after her vacations -/
def total_shells (initial : ℕ) (vacation1 : ℕ) (vacation2 : ℕ) (vacation3 : ℕ) : ℕ :=
  initial + vacation1 + vacation2 + vacation3

/-- The number of shells Shara collected during her first vacation -/
def first_vacation : ℕ := 5 * 3 + 6

/-- The number of shells Shara collected during her second vacation -/
def second_vacation : ℕ := 4 * 2 + 7

/-- The number of shells Shara collected during her third vacation -/
def third_vacation : ℕ := 8 + 4 + 3 * 2

theorem shara_shell_collection :
  total_shells 20 first_vacation second_vacation third_vacation = 74 := by
  sorry

end NUMINAMATH_CALUDE_shara_shell_collection_l3727_372797


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_50_l3727_372721

/-- The last two digits of a natural number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- The sum of factorials from 1 to n -/
def sumFactorials (n : ℕ) : ℕ := (List.range n).map Nat.factorial |>.sum

/-- The last two digits of the sum of factorials from 1 to 50 are 13 -/
theorem last_two_digits_sum_factorials_50 :
  lastTwoDigits (sumFactorials 50) = 13 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_50_l3727_372721


namespace NUMINAMATH_CALUDE_convenience_store_choices_l3727_372745

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (set1 : Nat) (set2 : Nat) : Nat :=
  set1 * set2

/-- Theorem: Choosing one item from a set of 4 and one from a set of 3 results in 12 possibilities -/
theorem convenience_store_choices :
  choose_one_from_each 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_convenience_store_choices_l3727_372745


namespace NUMINAMATH_CALUDE_two_digit_numbers_property_l3727_372766

-- Define a function to calculate the truncated square of a number
def truncatedSquare (n : ℕ) : ℕ := n * n + n * (n % 10) + (n % 10) * (n % 10)

-- Define the property for a two-digit number
def satisfiesProperty (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n = truncatedSquare (n / 10 + n % 10)

-- Theorem statement
theorem two_digit_numbers_property :
  satisfiesProperty 13 ∧
  satisfiesProperty 63 ∧
  63 - 13 = 50 ∧
  (∃ (x : ℕ), satisfiesProperty x ∧ x ≠ 13 ∧ x ≠ 63) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_property_l3727_372766


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3727_372751

/-- Given a cylinder with volume 81π cm³, prove that a cone with the same base radius
    and twice the height of the cylinder has a volume of 54π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 81 * π) → 
  ((1/3) * π * r^2 * (2*h) = 54 * π) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3727_372751


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3727_372730

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3727_372730


namespace NUMINAMATH_CALUDE_scientific_notation_conversion_l3727_372715

theorem scientific_notation_conversion :
  (380180000000 : ℝ) = 3.8018 * (10 : ℝ)^11 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_conversion_l3727_372715


namespace NUMINAMATH_CALUDE_optimal_walking_distance_ratio_l3727_372783

-- Define the problem setup
structure TravelProblem where
  totalDistance : ℝ
  speedA : ℝ
  speedB : ℝ
  speedC : ℝ
  mk_travel_problem : totalDistance > 0 ∧ speedA > 0 ∧ speedB > 0 ∧ speedC > 0

-- Define the optimal solution
def OptimalSolution (p : TravelProblem) :=
  ∃ (x : ℝ),
    0 < x ∧ x < p.totalDistance ∧
    (p.totalDistance - x) / p.speedA = x / (2 * p.speedC) + (p.totalDistance - x) / p.speedC

-- Theorem statement
theorem optimal_walking_distance_ratio 
  (p : TravelProblem) 
  (h_speeds : p.speedA = 4 ∧ p.speedB = 5 ∧ p.speedC = 12) 
  (h_optimal : OptimalSolution p) : 
  ∃ (distA distB : ℝ),
    distA > 0 ∧ distB > 0 ∧
    distA / distB = 17 / 10 ∧
    distA + distB = p.totalDistance :=
  sorry

end NUMINAMATH_CALUDE_optimal_walking_distance_ratio_l3727_372783


namespace NUMINAMATH_CALUDE_distance_traveled_l3727_372772

/-- Given a person traveling at a constant speed for a certain time,
    prove that the distance traveled is equal to the product of speed and time. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 25) (h2 : time = 5) :
  speed * time = 125 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3727_372772


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l3727_372720

/-- Calculates the percentage of total seeds that germinated given the number of seeds and germination rates for two plots. -/
theorem seed_germination_percentage 
  (seeds_plot1 : ℕ) 
  (seeds_plot2 : ℕ) 
  (germination_rate_plot1 : ℚ) 
  (germination_rate_plot2 : ℚ) 
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 25 / 100)
  (h4 : germination_rate_plot2 = 40 / 100) :
  (((seeds_plot1 : ℚ) * germination_rate_plot1 + (seeds_plot2 : ℚ) * germination_rate_plot2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) * 100 = 31 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l3727_372720


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3727_372737

theorem complex_fraction_simplification : 
  (((3875/1000) * (1/5) + (155/4) * (9/100) - (155/400)) / 
   ((13/6) + (((108/25) - (42/25) - (33/25)) * (5/11) - (2/7)) / (44/35) + (35/24))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3727_372737


namespace NUMINAMATH_CALUDE_min_value_of_sum_reciprocals_l3727_372782

theorem min_value_of_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_reciprocals_l3727_372782


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3727_372738

theorem inequality_solution_set (x : ℝ) :
  x ≠ 3/2 →
  ((x - 4) / (3 - 2*x) < 0) ↔ (x < 3/2 ∨ x > 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3727_372738


namespace NUMINAMATH_CALUDE_sophias_book_length_l3727_372734

theorem sophias_book_length (P : ℕ) : 
  (2 : ℚ) / 3 * P = (1 : ℚ) / 3 * P + 90 → P = 270 := by
  sorry

end NUMINAMATH_CALUDE_sophias_book_length_l3727_372734


namespace NUMINAMATH_CALUDE_unique_root_condition_l3727_372718

theorem unique_root_condition (k : ℝ) : 
  (∃! x : ℝ, (1/2) * Real.log (k * x) = Real.log (x + 1)) ↔ (k = 4 ∨ k < 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_condition_l3727_372718


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3727_372700

theorem quadratic_solution_difference_squared :
  ∀ p q : ℝ,
  (2 * p^2 + 11 * p - 21 = 0) →
  (2 * q^2 + 11 * q - 21 = 0) →
  p ≠ q →
  (p - q)^2 = 289/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3727_372700


namespace NUMINAMATH_CALUDE_complex_number_equality_l3727_372722

theorem complex_number_equality (z : ℂ) : 
  (Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
   Complex.abs (z - 2) = Complex.abs (z - 2*I)) ↔ 
  z = -1 - I :=
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3727_372722


namespace NUMINAMATH_CALUDE_calendar_box_sum_divisible_by_four_l3727_372743

/-- Represents a box of four numbers in a 7-column calendar --/
structure CalendarBox where
  top_right : ℕ
  top_left : ℕ
  bottom_left : ℕ
  bottom_right : ℕ

/-- Creates a calendar box given the top right number --/
def make_calendar_box (a : ℕ) : CalendarBox :=
  { top_right := a
  , top_left := a - 1
  , bottom_left := a + 6
  , bottom_right := a + 7 }

/-- The sum of numbers in a calendar box --/
def box_sum (box : CalendarBox) : ℕ :=
  box.top_right + box.top_left + box.bottom_left + box.bottom_right

/-- Theorem: The sum of numbers in any calendar box is divisible by 4 --/
theorem calendar_box_sum_divisible_by_four (a : ℕ) :
  4 ∣ box_sum (make_calendar_box a) := by
  sorry

end NUMINAMATH_CALUDE_calendar_box_sum_divisible_by_four_l3727_372743


namespace NUMINAMATH_CALUDE_sum_of_medians_l3727_372799

def player_A_median : ℝ := 36
def player_B_median : ℝ := 27

theorem sum_of_medians : player_A_median + player_B_median = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_medians_l3727_372799


namespace NUMINAMATH_CALUDE_largest_fraction_sum_inequality_l3727_372784

theorem largest_fraction_sum_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≥ b) (hac : a ≥ c)
  (h_eq : a / b = c / d) : 
  a + d > b + c := by
sorry

end NUMINAMATH_CALUDE_largest_fraction_sum_inequality_l3727_372784


namespace NUMINAMATH_CALUDE_average_age_problem_l3727_372767

theorem average_age_problem (a c : ℝ) : 
  (a + c) / 2 = 29 →
  ((a + c) + 26) / 3 = 28 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l3727_372767


namespace NUMINAMATH_CALUDE_vasya_fool_count_l3727_372759

theorem vasya_fool_count (misha petya kolya vasya : ℕ) : 
  misha + petya + kolya + vasya = 16 →
  misha ≥ 1 → petya ≥ 1 → kolya ≥ 1 → vasya ≥ 1 →
  petya + kolya = 9 →
  misha > petya → misha > kolya → misha > vasya →
  vasya = 1 := by
sorry

end NUMINAMATH_CALUDE_vasya_fool_count_l3727_372759


namespace NUMINAMATH_CALUDE_woodworker_extra_parts_l3727_372746

/-- A woodworker's production scenario -/
structure WoodworkerProduction where
  normal_days : ℕ
  normal_parts : ℕ
  productivity_increase : ℕ
  new_days : ℕ

/-- Calculate the extra parts made by the woodworker -/
def extra_parts (w : WoodworkerProduction) : ℕ :=
  let normal_daily := w.normal_parts / w.normal_days
  let new_daily := normal_daily + w.productivity_increase
  new_daily * w.new_days - w.normal_parts

/-- Theorem stating the extra parts made by the woodworker -/
theorem woodworker_extra_parts :
  ∀ (w : WoodworkerProduction),
    w.normal_days = 24 ∧
    w.normal_parts = 360 ∧
    w.productivity_increase = 5 ∧
    w.new_days = 22 →
    extra_parts w = 80 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_extra_parts_l3727_372746


namespace NUMINAMATH_CALUDE_problem_statements_l3727_372776

theorem problem_statements :
  (∀ x : ℝ, (x^2 - 4*x + 3 = 0 → x = 3) ↔ (x ≠ 3 → x^2 - 4*x + 3 ≠ 0)) ∧
  (¬(∀ x : ℝ, x^2 - x + 2 > 0) ↔ (∃ x : ℝ, x^2 - x + 2 ≤ 0)) ∧
  (∀ p q : Prop, (p ∧ q) → (p ∧ q)) ∧
  (∀ x : ℝ, x > -1 → x^2 + 4*x + 3 > 0) ∧
  (∃ x : ℝ, x^2 + 4*x + 3 > 0 ∧ x ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l3727_372776


namespace NUMINAMATH_CALUDE_elevator_capacity_l3727_372781

/-- Proves that the number of people in an elevator is 20, given the weight limit,
    average weight, and excess weight. -/
theorem elevator_capacity
  (weight_limit : ℝ)
  (average_weight : ℝ)
  (excess_weight : ℝ)
  (h1 : weight_limit = 1500)
  (h2 : average_weight = 80)
  (h3 : excess_weight = 100)
  : (weight_limit + excess_weight) / average_weight = 20 := by
  sorry

#check elevator_capacity

end NUMINAMATH_CALUDE_elevator_capacity_l3727_372781


namespace NUMINAMATH_CALUDE_lines_skew_when_one_parallel_to_plane_other_in_plane_l3727_372719

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line in 3D space
  -- (We'll leave this abstract for now)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane in 3D space
  -- (We'll leave this abstract for now)

/-- Proposition that a line is parallel to a plane -/
def is_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Define what it means for a line to be parallel to a plane
  sorry

/-- Proposition that a line is contained within a plane -/
def is_contained_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Define what it means for a line to be contained in a plane
  sorry

/-- Proposition that two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be skew
  sorry

/-- Theorem statement -/
theorem lines_skew_when_one_parallel_to_plane_other_in_plane 
  (a b : Line3D) (α : Plane3D) 
  (h1 : is_parallel_to_plane a α) 
  (h2 : is_contained_in_plane b α) : 
  are_skew a b :=
sorry

end NUMINAMATH_CALUDE_lines_skew_when_one_parallel_to_plane_other_in_plane_l3727_372719


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3727_372740

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {y | y^2 + y = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3727_372740


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l3727_372744

theorem angle_bisector_theorem (a b : Real) (h : b - a = 100) :
  (b / 2) - (a / 2) = 50 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l3727_372744


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l3727_372748

structure Orangeade where
  orange_juice : ℝ
  water : ℝ
  price_per_glass : ℝ
  glasses_sold : ℝ

def revenue (o : Orangeade) : ℝ := o.price_per_glass * o.glasses_sold

theorem orangeade_price_day2 (day1 day2 : Orangeade) :
  day1.orange_juice > 0 →
  day1.orange_juice = day1.water →
  day2.orange_juice = day1.orange_juice →
  day2.water = 2 * day1.water →
  day1.price_per_glass = 0.9 →
  revenue day1 = revenue day2 →
  day2.glasses_sold = (3/2) * day1.glasses_sold →
  day2.price_per_glass = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l3727_372748


namespace NUMINAMATH_CALUDE_chloe_carrot_problem_l3727_372773

theorem chloe_carrot_problem :
  ∀ (initial_carrots picked_next_day final_carrots thrown_out : ℕ),
    initial_carrots = 48 →
    picked_next_day = 42 →
    final_carrots = 45 →
    initial_carrots - thrown_out + picked_next_day = final_carrots →
    thrown_out = 45 := by
  sorry

end NUMINAMATH_CALUDE_chloe_carrot_problem_l3727_372773


namespace NUMINAMATH_CALUDE_tower_surface_area_l3727_372753

/-- Represents a cube in the tower --/
structure Cube where
  volume : ℕ
  sideLength : ℕ
  deriving Repr

/-- Represents the tower of cubes --/
def Tower : List Cube := [
  { volume := 343, sideLength := 7 },
  { volume := 125, sideLength := 5 },
  { volume := 27,  sideLength := 3 },
  { volume := 64,  sideLength := 4 },
  { volume := 1,   sideLength := 1 }
]

/-- Calculates the visible surface area of a cube in the tower --/
def visibleSurfaceArea (cube : Cube) (aboveCube : Option Cube) : ℕ := sorry

/-- Calculates the total visible surface area of the tower --/
def totalVisibleSurfaceArea (tower : List Cube) : ℕ := sorry

/-- Theorem stating that the total visible surface area of the tower is 400 square units --/
theorem tower_surface_area : totalVisibleSurfaceArea Tower = 400 := by sorry

end NUMINAMATH_CALUDE_tower_surface_area_l3727_372753


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3727_372769

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((2496 + y) % 7 = 0 ∧ (2496 + y) % 11 = 0)) ∧ 
  (2496 + x) % 7 = 0 ∧ (2496 + x) % 11 = 0 → 
  x = 37 := by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3727_372769


namespace NUMINAMATH_CALUDE_factory_production_equation_l3727_372717

/-- Represents the production equation for a factory with monthly growth rate --/
theorem factory_production_equation (april_production : ℝ) (quarter_production : ℝ) (x : ℝ) :
  april_production = 500000 →
  quarter_production = 1820000 →
  50 + 50 * (1 + x) + 50 * (1 + x)^2 = 182 :=
by sorry

end NUMINAMATH_CALUDE_factory_production_equation_l3727_372717


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l3727_372708

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a1 : a 1 = 2)
  (h_a2 : a 2 = 4) :
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l3727_372708


namespace NUMINAMATH_CALUDE_kebul_family_children_l3727_372706

/-- Represents a family with children -/
structure Family where
  total_members : ℕ
  father_age : ℕ
  average_age : ℚ
  average_age_without_father : ℚ

/-- Calculates the number of children in a family -/
def number_of_children (f : Family) : ℕ :=
  f.total_members - 2

/-- Theorem stating the number of children in the Kebul family -/
theorem kebul_family_children (f : Family) 
  (h1 : f.average_age = 18)
  (h2 : f.father_age = 38)
  (h3 : f.average_age_without_father = 14) :
  number_of_children f = 4 := by
  sorry

#eval number_of_children { total_members := 6, father_age := 38, average_age := 18, average_age_without_father := 14 }

end NUMINAMATH_CALUDE_kebul_family_children_l3727_372706


namespace NUMINAMATH_CALUDE_fruit_bag_probabilities_l3727_372791

theorem fruit_bag_probabilities (apples oranges : ℕ) (h1 : apples = 7) (h2 : oranges = 1) :
  let total := apples + oranges
  (apples : ℚ) / total = 7 / 8 ∧ (oranges : ℚ) / total = 1 / 8 := by
sorry


end NUMINAMATH_CALUDE_fruit_bag_probabilities_l3727_372791


namespace NUMINAMATH_CALUDE_product_factor_proof_l3727_372704

theorem product_factor_proof (w : ℕ+) (h1 : 2^5 ∣ (936 * w)) (h2 : 3^3 ∣ (936 * w)) (h3 : w ≥ 144) :
  ∃ x : ℕ, 12^x ∣ (936 * w) ∧ ∀ y : ℕ, 12^y ∣ (936 * w) → y ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_product_factor_proof_l3727_372704


namespace NUMINAMATH_CALUDE_factorization_theorem_1_factorization_theorem_2_l3727_372762

-- Theorem 1
theorem factorization_theorem_1 (m n : ℝ) : m^3*n - 9*m*n = m*n*(m+3)*(m-3) := by
  sorry

-- Theorem 2
theorem factorization_theorem_2 (a : ℝ) : a^3 + a - 2*a^2 = a*(a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_1_factorization_theorem_2_l3727_372762


namespace NUMINAMATH_CALUDE_intersecting_lines_angles_l3727_372789

-- Define a structure for a line
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for an angle
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function for alternate interior angles
def alternate_interior_angles (l1 l2 : Line) (t : Line) : Angle × Angle :=
  sorry

-- Define a function for corresponding angles
def corresponding_angles (l1 l2 : Line) (t : Line) : Angle × Angle :=
  sorry

-- Define a function for consecutive interior angles
def consecutive_interior_angles (l1 l2 : Line) (t : Line) : Angle × Angle :=
  sorry

-- Main theorem
theorem intersecting_lines_angles (l1 l2 t : Line) 
  (h : ¬ are_parallel l1 l2) : 
  ∃ (a1 a2 : Angle), 
    (alternate_interior_angles l1 l2 t = (a1, a2) ∧ a1.measure ≠ a2.measure) ∨
    (corresponding_angles l1 l2 t = (a1, a2) ∧ a1.measure ≠ a2.measure) ∨
    (consecutive_interior_angles l1 l2 t = (a1, a2) ∧ a1.measure + a2.measure ≠ 180) :=
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_angles_l3727_372789


namespace NUMINAMATH_CALUDE_midpoint_specific_segment_l3727_372733

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_specific_segment :
  let (r, θ) := polar_midpoint 10 (π/4) 10 (3*π/4)
  r = 5 * Real.sqrt 2 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π := by
  sorry

end NUMINAMATH_CALUDE_midpoint_specific_segment_l3727_372733


namespace NUMINAMATH_CALUDE_vector_magnitude_l3727_372768

/-- Given plane vectors a, b, and c, if (a + b) is parallel to c, then the magnitude of c is 2√17. -/
theorem vector_magnitude (a b c : ℝ × ℝ) (h : ∃ (t : ℝ), a + b = t • c) : 
  a = (-1, 1) → b = (2, 3) → c.1 = -2 → ‖c‖ = 2 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3727_372768


namespace NUMINAMATH_CALUDE_transistors_2010_count_l3727_372786

/-- The number of transistors in a CPU triples every two years -/
def tripling_period : ℕ := 2

/-- The initial number of transistors in 1990 -/
def initial_transistors : ℕ := 500000

/-- The number of years between 1990 and 2010 -/
def years_passed : ℕ := 20

/-- Calculate the number of transistors in 2010 -/
def transistors_2010 : ℕ := initial_transistors * (3 ^ (years_passed / tripling_period))

/-- Theorem stating that the number of transistors in 2010 is 29,524,500,000 -/
theorem transistors_2010_count : transistors_2010 = 29524500000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_2010_count_l3727_372786


namespace NUMINAMATH_CALUDE_product_and_sum_of_factors_l3727_372754

theorem product_and_sum_of_factors : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 1540 ∧ 
  a + b = 97 := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_of_factors_l3727_372754


namespace NUMINAMATH_CALUDE_sample_size_definition_l3727_372735

/-- Represents a population of students' exam scores -/
structure Population where
  scores : Set ℝ

/-- Represents a sample drawn from a population -/
structure Sample where
  elements : Finset ℝ

/-- Simple random sampling function -/
def simpleRandomSampling (pop : Population) (n : ℕ) : Sample :=
  sorry

theorem sample_size_definition 
  (pop : Population) 
  (sample : Sample) 
  (n : ℕ) 
  (h1 : sample = simpleRandomSampling pop n) 
  (h2 : n = 100) : 
  n = Finset.card sample.elements :=
sorry

end NUMINAMATH_CALUDE_sample_size_definition_l3727_372735


namespace NUMINAMATH_CALUDE_triangle_perimeter_ratio_l3727_372750

theorem triangle_perimeter_ratio (X Y Z D J : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : 
  let XZ := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  let YZ := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  let XY := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  -- XYZ is a right triangle with hypotenuse XY
  (X.1 - Z.1) * (Y.1 - Z.1) + (X.2 - Z.2) * (Y.2 - Z.2) = 0 →
  -- XZ = 8, YZ = 15
  XZ = 8 →
  YZ = 15 →
  -- ZD is the altitude to XY
  (X.1 - Y.1) * (D.1 - Z.1) + (X.2 - Y.2) * (D.2 - Z.2) = 0 →
  -- ω is the circle with ZD as diameter
  ω = {P : ℝ × ℝ | (P.1 - Z.1)^2 + (P.2 - Z.2)^2 = (D.1 - Z.1)^2 + (D.2 - Z.2)^2} →
  -- J is outside XYZ
  (J.1 - X.1) * (Y.2 - X.2) - (J.2 - X.2) * (Y.1 - X.1) ≠ 0 →
  -- XJ and YJ are tangent to ω
  ∃ P ∈ ω, (J.1 - X.1) * (P.1 - X.1) + (J.2 - X.2) * (P.2 - X.2) = 0 →
  ∃ Q ∈ ω, (J.1 - Y.1) * (Q.1 - Y.1) + (J.2 - Y.2) * (Q.2 - Y.2) = 0 →
  -- The ratio of the perimeter of XYJ to XY is 30/17
  (Real.sqrt ((X.1 - J.1)^2 + (X.2 - J.2)^2) + 
   Real.sqrt ((Y.1 - J.1)^2 + (Y.2 - J.2)^2) + XY) / XY = 30/17 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_ratio_l3727_372750


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3727_372788

/-- Given a circle with equation x^2 + y^2 = 6x + 8y - 15, 
    prove that the sum of the x and y coordinates of its center is 7. -/
theorem circle_center_coordinate_sum : 
  ∀ (h k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 6*x + 8*y - 15 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h - 8*k + 15)) →
  h + k = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3727_372788


namespace NUMINAMATH_CALUDE_tom_trout_count_l3727_372731

/-- Given that Melanie catches 8 trout and Tom catches 2 times as many trout as Melanie,
    prove that Tom catches 16 trout. -/
theorem tom_trout_count (melanie_trout : ℕ) (tom_multiplier : ℕ) 
    (h1 : melanie_trout = 8)
    (h2 : tom_multiplier = 2) : 
  tom_multiplier * melanie_trout = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_trout_count_l3727_372731


namespace NUMINAMATH_CALUDE_investment_return_calculation_l3727_372774

theorem investment_return_calculation (total_investment : ℝ) (combined_return_rate : ℝ) 
  (investment_1 : ℝ) (return_rate_1 : ℝ) (investment_2 : ℝ) :
  total_investment = 2000 →
  combined_return_rate = 0.22 →
  investment_1 = 500 →
  return_rate_1 = 0.07 →
  investment_2 = 1500 →
  let total_return := combined_return_rate * total_investment
  let return_1 := return_rate_1 * investment_1
  let return_2 := total_return - return_1
  return_2 / investment_2 = 0.27 := by sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l3727_372774


namespace NUMINAMATH_CALUDE_line_equidistant_points_l3727_372728

/-- Given a line passing through (4, 4) with slope 0.5, equidistant from (0, 2) and (A, 8), prove A = -3 -/
theorem line_equidistant_points (A : ℝ) : 
  let line_point : ℝ × ℝ := (4, 4)
  let line_slope : ℝ := 0.5
  let P : ℝ × ℝ := (0, 2)
  let Q : ℝ × ℝ := (A, 8)
  let midpoint : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let perpendicular_slope : ℝ := -1 / ((Q.2 - P.2) / (Q.1 - P.1))
  (line_slope = perpendicular_slope) ∧ 
  (midpoint.2 - line_point.2 = line_slope * (midpoint.1 - line_point.1)) →
  A = -3 := by
sorry

end NUMINAMATH_CALUDE_line_equidistant_points_l3727_372728


namespace NUMINAMATH_CALUDE_sphere_water_volume_l3727_372705

def hemisphere_volume : ℝ := 4
def num_hemispheres : ℕ := 2749

theorem sphere_water_volume :
  let total_volume := (num_hemispheres : ℝ) * hemisphere_volume
  total_volume = 10996 := by sorry

end NUMINAMATH_CALUDE_sphere_water_volume_l3727_372705


namespace NUMINAMATH_CALUDE_defeat_dragon_l3727_372787

def dragonHeads (n : ℕ) : ℕ → ℕ
  | 0 => n
  | m + 1 => 
    let remaining := dragonHeads n m - 5
    if remaining ≤ 5 then 0
    else remaining + (remaining % 9)

theorem defeat_dragon (initialHeads : ℕ) (swings : ℕ) : 
  initialHeads = 198 →
  (∀ k < swings, dragonHeads initialHeads k > 5) →
  dragonHeads initialHeads swings ≤ 5 →
  swings = 40 :=
sorry

#check defeat_dragon

end NUMINAMATH_CALUDE_defeat_dragon_l3727_372787


namespace NUMINAMATH_CALUDE_milk_students_l3727_372770

theorem milk_students (total_students : ℕ) (soda_students : ℕ) (milk_percent : ℚ) (soda_percent : ℚ) :
  soda_percent = 1/2 →
  milk_percent = 3/10 →
  soda_students = 90 →
  (milk_percent / soda_percent) * soda_students = 54 :=
by sorry

end NUMINAMATH_CALUDE_milk_students_l3727_372770


namespace NUMINAMATH_CALUDE_oldest_sister_clothing_amount_l3727_372763

/-- Proves that the oldest sister's clothing amount is the difference between Nicole's final amount and the sum of the younger sisters' amounts. -/
theorem oldest_sister_clothing_amount 
  (nicole_initial : ℕ) 
  (nicole_final : ℕ) 
  (first_older_sister : ℕ) 
  (next_oldest_sister : ℕ) 
  (h1 : nicole_initial = 10)
  (h2 : first_older_sister = nicole_initial / 2)
  (h3 : next_oldest_sister = nicole_initial + 2)
  (h4 : nicole_final = 36) :
  nicole_final - (nicole_initial + first_older_sister + next_oldest_sister) = 9 := by
sorry

end NUMINAMATH_CALUDE_oldest_sister_clothing_amount_l3727_372763


namespace NUMINAMATH_CALUDE_valid_numbers_divisible_by_36_l3727_372755

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 52000 + a * 100 + 20 + b

def is_divisible_by_36 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 36 * k

theorem valid_numbers_divisible_by_36 :
  ∀ n : ℕ, is_valid_number n ∧ is_divisible_by_36 n ↔ 
    n = 52524 ∨ n = 52128 ∨ n = 52020 ∨ n = 52920 :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_divisible_by_36_l3727_372755


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l3727_372793

theorem unique_solution_logarithmic_equation :
  ∃! (x : ℝ), x > 0 ∧ x^(Real.log x / Real.log 10) = x^4 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l3727_372793


namespace NUMINAMATH_CALUDE_stating_greatest_N_with_property_l3727_372727

/-- 
Given a positive integer N, this function type represents the existence of 
N integers x_1, ..., x_N such that x_i^2 - x_i x_j is not divisible by 1111 for any i ≠ j.
-/
def HasProperty (N : ℕ+) : Prop :=
  ∃ (x : Fin N → ℤ), ∀ (i j : Fin N), i ≠ j → ¬(1111 ∣ (x i)^2 - (x i) * (x j))

/-- 
Theorem stating that 1000 is the greatest positive integer satisfying the property.
-/
theorem greatest_N_with_property : 
  HasProperty 1000 ∧ ∀ (N : ℕ+), N > 1000 → ¬HasProperty N :=
sorry

end NUMINAMATH_CALUDE_stating_greatest_N_with_property_l3727_372727


namespace NUMINAMATH_CALUDE_max_of_min_values_l3727_372796

/-- The function f(x) for a given m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 8*m + 4

/-- The minimum value of f(x) for a given m -/
def min_value (m : ℝ) : ℝ := f m m

/-- The function representing all minimum values of f(x) for different m -/
def g (m : ℝ) : ℝ := -m^2 + 8*m + 4

/-- The maximum of all minimum values of f(x) -/
theorem max_of_min_values :
  (⨆ (m : ℝ), min_value m) = 20 :=
sorry

end NUMINAMATH_CALUDE_max_of_min_values_l3727_372796


namespace NUMINAMATH_CALUDE_chord_count_for_concentric_circles_l3727_372726

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle formed by two adjacent chords is 60°, then exactly 3 such chords are needed to complete a full circle. -/
theorem chord_count_for_concentric_circles (angle : ℝ) (n : ℕ) : 
  angle = 60 → n * angle = 360 → n = 3 := by sorry

end NUMINAMATH_CALUDE_chord_count_for_concentric_circles_l3727_372726


namespace NUMINAMATH_CALUDE_intersection_line_passes_through_circles_l3727_372758

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4*x
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0

-- Define the line
def line (x y : ℝ) : Prop := y = -x

-- Theorem statement
theorem intersection_line_passes_through_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_passes_through_circles_l3727_372758


namespace NUMINAMATH_CALUDE_bird_count_l3727_372736

theorem bird_count (crows : ℕ) (hawk_percentage : ℚ) : 
  crows = 30 → 
  hawk_percentage = 60 / 100 → 
  crows + (crows + hawk_percentage * crows) = 78 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_l3727_372736


namespace NUMINAMATH_CALUDE_sum_of_xyz_l3727_372764

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 40) (hxz : x * z = 80) (hyz : y * z = 120) :
  x + y + z = 22 * Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l3727_372764


namespace NUMINAMATH_CALUDE_divisible_by_five_l3727_372729

theorem divisible_by_five (n : ℕ) : ∃ k : ℤ, (n^5 : ℤ) + 4*n = 5*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l3727_372729


namespace NUMINAMATH_CALUDE_octahedron_side_length_l3727_372724

/-- A unit cube in 3D space -/
structure UnitCube where
  A : ℝ × ℝ × ℝ := (0, 0, 0)
  A' : ℝ × ℝ × ℝ := (1, 1, 1)

/-- A regular octahedron inscribed in a unit cube -/
structure InscribedOctahedron (cube : UnitCube) where
  vertices : List (ℝ × ℝ × ℝ)

/-- The side length of an inscribed octahedron -/
def sideLength (octahedron : InscribedOctahedron cube) : ℝ :=
  sorry

/-- Theorem: The side length of the inscribed octahedron is √2/3 -/
theorem octahedron_side_length (cube : UnitCube) 
  (octahedron : InscribedOctahedron cube) : 
  sideLength octahedron = Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_side_length_l3727_372724


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3727_372725

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := x^2 + 25*y + 19*z = -471
def equation2 (x y z : ℝ) : Prop := y^2 + 23*x + 21*z = -397
def equation3 (x y z : ℝ) : Prop := z^2 + 21*x + 21*y = -545

-- Theorem statement
theorem solution_satisfies_system :
  equation1 (-22) (-23) (-20) ∧
  equation2 (-22) (-23) (-20) ∧
  equation3 (-22) (-23) (-20) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3727_372725


namespace NUMINAMATH_CALUDE_translation_problem_l3727_372711

def complex_translation (z w : ℂ) : ℂ := z + w

theorem translation_problem (t : ℂ → ℂ) :
  (t (1 + 3*I) = -2 + 4*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = complex_translation z w) →
  (t (3 + 7*I) = 8*I) :=
by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l3727_372711


namespace NUMINAMATH_CALUDE_sqrt_expressions_l3727_372779

theorem sqrt_expressions :
  (2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2) ∧
  (Real.sqrt ((-3)^2) ≠ -3) ∧
  (Real.sqrt 24 / Real.sqrt 6 ≠ 4) ∧
  (Real.sqrt 3 + Real.sqrt 2 ≠ Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l3727_372779


namespace NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l3727_372795

/-- The sum of an arithmetic sequence with given parameters -/
def arithmeticSequenceSum (a1 : ℤ) (an : ℤ) (d : ℤ) : ℤ :=
  let n : ℤ := (an - a1) / d + 1
  n * (a1 + an) / 2

/-- Theorem: The sum of the specific arithmetic sequence is -440 -/
theorem specific_arithmetic_sequence_sum :
  arithmeticSequenceSum (-41) 1 2 = -440 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l3727_372795


namespace NUMINAMATH_CALUDE_combined_average_marks_l3727_372778

/-- Given two classes with the specified number of students and average marks,
    calculate the combined average mark of all students in both classes. -/
theorem combined_average_marks
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ)
  (h1 : n1 = 55) (h2 : n2 = 48) (h3 : avg1 = 60) (h4 : avg2 = 58) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℝ) = (55 * 60 + 48 * 58) / (55 + 48 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_combined_average_marks_l3727_372778


namespace NUMINAMATH_CALUDE_fran_required_speed_l3727_372749

/-- Calculates the required average speed for Fran to travel the same distance as Joann -/
theorem fran_required_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5) :
  (joann_speed * joann_time) / fran_time = 120 / 7 :=
by sorry

end NUMINAMATH_CALUDE_fran_required_speed_l3727_372749


namespace NUMINAMATH_CALUDE_marbles_given_proof_l3727_372723

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := sorry

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := 143

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := 70

/-- Theorem stating that the number of marbles given is equal to the difference
between the initial number of marbles and the remaining marbles -/
theorem marbles_given_proof : 
  marbles_given = initial_marbles - remaining_marbles :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_proof_l3727_372723


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3727_372798

open Complex

theorem imaginary_part_of_z : ∃ (z : ℂ), z = (1 + I)^2 + I^2010 ∧ z.im = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3727_372798


namespace NUMINAMATH_CALUDE_probability_two_hits_l3727_372742

def probability_at_least_one_hit : ℚ := 65/81

def number_of_shots : ℕ := 4

def probability_single_hit : ℚ := 1/3

theorem probability_two_hits :
  (1 - probability_at_least_one_hit) = (1 - probability_single_hit) ^ number_of_shots →
  Nat.choose number_of_shots 2 * probability_single_hit^2 * (1 - probability_single_hit)^2 = 8/27 := by
sorry

end NUMINAMATH_CALUDE_probability_two_hits_l3727_372742


namespace NUMINAMATH_CALUDE_greg_total_distance_l3727_372716

/-- The total distance Greg travels given his individual trip distances -/
theorem greg_total_distance (d1 d2 d3 : ℝ) 
  (h1 : d1 = 30) -- Distance from workplace to farmer's market
  (h2 : d2 = 20) -- Distance from farmer's market to friend's house
  (h3 : d3 = 25) -- Distance from friend's house to home
  : d1 + d2 + d3 = 75 := by sorry

end NUMINAMATH_CALUDE_greg_total_distance_l3727_372716


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3727_372732

/-- Given a rectangle with vertices at (-2, y), (6, y), (-2, 2), and (6, 2),
    where y is positive, and an area of 64 square units, y must equal 10. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  (6 - (-2)) * (y - 2) = 64 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3727_372732


namespace NUMINAMATH_CALUDE_two_red_one_blue_probability_l3727_372756

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 3

theorem two_red_one_blue_probability :
  (Nat.choose red_marbles 2 * blue_marbles) / Nat.choose total_marbles selected_marbles = 44 / 95 :=
by sorry

end NUMINAMATH_CALUDE_two_red_one_blue_probability_l3727_372756


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l3727_372701

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l3727_372701


namespace NUMINAMATH_CALUDE_problem_l3727_372780

theorem problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1/a + 1/b) :
  (a + b ≥ 2) ∧ ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_l3727_372780


namespace NUMINAMATH_CALUDE_greatest_third_side_length_l3727_372765

theorem greatest_third_side_length (a b : ℝ) (ha : a = 5) (hb : b = 11) :
  ∃ (c : ℕ), c = 15 ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
  ∀ (d : ℕ), d > c → ¬((a + b > d) ∧ (a + d > b) ∧ (b + d > a)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_third_side_length_l3727_372765


namespace NUMINAMATH_CALUDE_total_spent_with_tip_l3727_372747

def lunch_cost : ℝ := 60.50
def tip_percentage : ℝ := 0.20

theorem total_spent_with_tip : 
  lunch_cost * (1 + tip_percentage) = 72.60 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_with_tip_l3727_372747


namespace NUMINAMATH_CALUDE_quadratic_function_b_range_l3727_372761

/-- Given a quadratic function f(x) = x^2 + 2bx + c where b and c are real numbers,
    if f(1) = 0 and the equation f(x) + x + b = 0 has two real roots
    in the intervals (-3,-2) and (0,1), then b is in the open interval (1/5, 5/7). -/
theorem quadratic_function_b_range (b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*b*x + c
  (f 1 = 0) →
  (∃ x₁ x₂, -3 < x₁ ∧ x₁ < -2 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
    f x₁ + x₁ + b = 0 ∧ f x₂ + x₂ + b = 0) →
  1/5 < b ∧ b < 5/7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_b_range_l3727_372761


namespace NUMINAMATH_CALUDE_rectangle_area_and_range_l3727_372709

/-- Represents the area of a rectangle formed by a rope of length 10cm -/
def area (x : ℝ) : ℝ := -x^2 + 5*x

/-- The length of the rope forming the rectangle -/
def ropeLength : ℝ := 10

theorem rectangle_area_and_range :
  ∀ x : ℝ, 0 < x ∧ x < 5 →
  (2 * (x + (ropeLength / 2 - x)) = ropeLength) ∧
  (area x = x * (ropeLength / 2 - x)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_and_range_l3727_372709


namespace NUMINAMATH_CALUDE_average_weight_increase_l3727_372760

theorem average_weight_increase (group_size : ℕ) (original_weight new_weight : ℝ) :
  group_size = 4 →
  original_weight = 65 →
  new_weight = 71 →
  (new_weight - original_weight) / group_size = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3727_372760


namespace NUMINAMATH_CALUDE_fewer_heads_probability_l3727_372790

/-- The number of coins being flipped -/
def n : ℕ := 8

/-- The probability of getting the same number of heads and tails -/
def p_equal : ℚ := (n.choose (n / 2)) / 2^n

/-- The probability of getting fewer heads than tails -/
def p_fewer_heads : ℚ := (1 - p_equal) / 2

theorem fewer_heads_probability :
  p_fewer_heads = 93 / 256 := by sorry

end NUMINAMATH_CALUDE_fewer_heads_probability_l3727_372790


namespace NUMINAMATH_CALUDE_clock_hand_overlaps_in_day_l3727_372777

/-- Represents the number of overlaps between clock hands in a given time period -/
def clockHandOverlaps (hourRotations minuteRotations : ℕ) : ℕ :=
  minuteRotations - hourRotations

theorem clock_hand_overlaps_in_day :
  clockHandOverlaps 2 24 = 22 := by
  sorry

#eval clockHandOverlaps 2 24

end NUMINAMATH_CALUDE_clock_hand_overlaps_in_day_l3727_372777


namespace NUMINAMATH_CALUDE_base8_sum_l3727_372775

/-- Base 8 representation of a three-digit number -/
def base8Rep (x y z : ℕ) : ℕ := 64 * x + 8 * y + z

/-- Proposition: If X, Y, and Z are non-zero distinct digits in base 8 such that 
    XYZ₈ + YZX₈ + ZXY₈ = XXX0₈, then Y + Z = 7₈ -/
theorem base8_sum (X Y Z : ℕ) 
  (h1 : X ≠ 0 ∧ Y ≠ 0 ∧ Z ≠ 0)
  (h2 : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z)
  (h3 : X < 8 ∧ Y < 8 ∧ Z < 8)
  (h4 : base8Rep X Y Z + base8Rep Y Z X + base8Rep Z X Y = 8 * base8Rep X X X) :
  Y + Z = 7 := by
sorry

end NUMINAMATH_CALUDE_base8_sum_l3727_372775


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l3727_372757

/-- The weight difference between Kelly's chemistry and geometry textbooks -/
theorem textbook_weight_difference :
  let chemistry_weight : ℚ := 712 / 100
  let geometry_weight : ℚ := 62 / 100
  chemistry_weight - geometry_weight = 650 / 100 :=
by sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l3727_372757


namespace NUMINAMATH_CALUDE_park_fencing_cost_l3727_372712

/-- Represents the dimensions and fencing costs of a park with a flower bed -/
structure ParkWithFlowerBed where
  park_ratio : Rat -- Ratio of park's length to width
  park_area : ℝ -- Area of the park in square meters
  park_fence_cost : ℝ -- Cost of fencing the park per meter
  flowerbed_fence_cost : ℝ -- Cost of fencing the flower bed per meter

/-- Calculates the total fencing cost for a park with a flower bed -/
def total_fencing_cost (p : ParkWithFlowerBed) : ℝ :=
  sorry

/-- Theorem stating the total fencing cost for the given park configuration -/
theorem park_fencing_cost :
  let p : ParkWithFlowerBed := {
    park_ratio := 3/2,
    park_area := 3750,
    park_fence_cost := 0.70,
    flowerbed_fence_cost := 0.90
  }
  total_fencing_cost p = 245.65 := by
  sorry

end NUMINAMATH_CALUDE_park_fencing_cost_l3727_372712


namespace NUMINAMATH_CALUDE_largest_divisor_n_plus_10_divisibility_condition_l3727_372741

theorem largest_divisor_n_plus_10 :
  ∀ n : ℕ, n > 0 → (n + 10) ∣ (n^3 + 2011) → n ≤ 1001 :=
by sorry

theorem divisibility_condition :
  (1001 + 10) ∣ (1001^3 + 2011) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_n_plus_10_divisibility_condition_l3727_372741


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l3727_372703

theorem oranges_thrown_away (initial_oranges : ℕ) (new_oranges : ℕ) (final_oranges : ℕ)
  (h1 : initial_oranges = 50)
  (h2 : new_oranges = 24)
  (h3 : final_oranges = 34)
  : initial_oranges - (initial_oranges - new_oranges + final_oranges) = 40 := by
  sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l3727_372703


namespace NUMINAMATH_CALUDE_carla_cooking_time_l3727_372771

def total_time (sharpening_time peeling_time chopping_time first_break fruits_time second_break salad_time : ℝ) : ℝ :=
  sharpening_time + peeling_time + chopping_time + first_break + fruits_time + second_break + salad_time

theorem carla_cooking_time : ∃ (total : ℝ),
  let sharpening_time : ℝ := 15
  let peeling_time : ℝ := 3 * sharpening_time
  let chopping_time : ℝ := (1 / 4) * peeling_time
  let first_break : ℝ := 5
  let fruits_time : ℝ := 2 * chopping_time
  let second_break : ℝ := 10
  let previous_activities_time : ℝ := sharpening_time + peeling_time + chopping_time + first_break + fruits_time + second_break
  let salad_time : ℝ := (3 / 5) * previous_activities_time
  total = total_time sharpening_time peeling_time chopping_time first_break fruits_time second_break salad_time ∧
  total = 174.6 := by
    sorry

end NUMINAMATH_CALUDE_carla_cooking_time_l3727_372771


namespace NUMINAMATH_CALUDE_BD_range_l3727_372794

/-- Triangle ABC with median AD to side BC -/
structure Triangle :=
  (A B C D : ℝ × ℝ)
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)
  (BD : ℝ)
  (is_median : BD = BC / 2)
  (AB_eq : AB = 5)
  (AC_eq : AC = 7)

/-- The length of BD in a triangle ABC with median AD to side BC, 
    where AB = 5 and AC = 7, satisfies 1 < BD < 6 -/
theorem BD_range (t : Triangle) : 1 < t.BD ∧ t.BD < 6 := by
  sorry

end NUMINAMATH_CALUDE_BD_range_l3727_372794


namespace NUMINAMATH_CALUDE_lucy_groceries_l3727_372792

/-- The number of packs of groceries Lucy bought -/
def total_groceries (cookies cake chocolate : ℕ) : ℕ :=
  cookies + cake + chocolate

/-- Theorem stating that Lucy bought 42 packs of groceries in total -/
theorem lucy_groceries : total_groceries 4 22 16 = 42 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l3727_372792


namespace NUMINAMATH_CALUDE_max_ab_l3727_372710

theorem max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4*a + b = 1) :
  ab ≤ 1/16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4*a₀ + b₀ = 1 ∧ a₀*b₀ = 1/16 :=
sorry


end NUMINAMATH_CALUDE_max_ab_l3727_372710


namespace NUMINAMATH_CALUDE_total_profit_calculation_total_profit_is_630_l3727_372785

/-- Calculates the total profit given investment conditions and A's share of profit -/
theorem total_profit_calculation (a_initial : ℕ) (b_initial : ℕ) (a_withdrawal : ℕ) (b_addition : ℕ) (months : ℕ) (a_share : ℕ) : ℕ :=
  let a_investment_months := a_initial * 8 + (a_initial - a_withdrawal) * 4
  let b_investment_months := b_initial * 8 + (b_initial + b_addition) * 4
  let total_ratio_parts := a_investment_months + b_investment_months
  let total_profit := a_share * total_ratio_parts / a_investment_months
  total_profit

/-- The total profit at the end of the year is 630 Rs -/
theorem total_profit_is_630 :
  total_profit_calculation 3000 4000 1000 1000 12 240 = 630 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_calculation_total_profit_is_630_l3727_372785
