import Mathlib

namespace NUMINAMATH_CALUDE_x_not_equal_one_l3398_339870

theorem x_not_equal_one (x : ℝ) (h : (x - 1)^0 = 1) : x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_x_not_equal_one_l3398_339870


namespace NUMINAMATH_CALUDE_unique_intersection_point_l3398_339830

/-- The function g(x) = x^3 + 5x^2 + 12x + 20 -/
def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 12*x + 20

/-- Theorem: The unique intersection point of g(x) and its inverse is (-4, -4) -/
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-4, -4) := by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l3398_339830


namespace NUMINAMATH_CALUDE_set_equality_implies_x_values_l3398_339837

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- State the theorem
theorem set_equality_implies_x_values (x : ℝ) :
  A x ∪ B x = A x → x = 2 ∨ x = -2 ∨ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_x_values_l3398_339837


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3398_339892

theorem algebraic_expression_equality (x : ℝ) (h : x = 5) :
  3 / (x - 4) - 24 / (x^2 - 16) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3398_339892


namespace NUMINAMATH_CALUDE_max_sum_of_four_digits_l3398_339865

def is_valid_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem max_sum_of_four_digits :
  ∀ A B C D : ℕ,
    is_valid_digit A → is_valid_digit B → is_valid_digit C → is_valid_digit D →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A + B) + (C + D) ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_four_digits_l3398_339865


namespace NUMINAMATH_CALUDE_work_completion_time_l3398_339836

/-- If a group can complete a task in 12 days, then twice that group can complete half the task in 3 days. -/
theorem work_completion_time 
  (people : ℕ) 
  (work : ℝ) 
  (h : people > 0) 
  (completion_time : ℝ := 12) 
  (h_completion : work = people * completion_time) : 
  work / 2 = (2 * people) * 3 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3398_339836


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l3398_339846

/-- An isosceles right triangle with perimeter 14 + 14√2 has a hypotenuse of length 28 -/
theorem isosceles_right_triangle_hypotenuse : ∀ a c : ℝ,
  a > 0 → c > 0 →
  a = c / Real.sqrt 2 →  -- Condition for isosceles right triangle
  2 * a + c = 14 + 14 * Real.sqrt 2 →  -- Perimeter condition
  c = 28 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l3398_339846


namespace NUMINAMATH_CALUDE_rectangles_in_35_44_grid_l3398_339848

/-- The number of rectangles in a grid -/
def count_rectangles (m n : ℕ) : ℕ :=
  (m * (m + 1) * n * (n + 1)) / 4

/-- Theorem: The number of rectangles in a 35 · 44 grid is 87 -/
theorem rectangles_in_35_44_grid :
  count_rectangles 35 44 = 87 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_35_44_grid_l3398_339848


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_120_over_8_l3398_339874

theorem largest_whole_number_less_than_120_over_8 : 
  (∀ n : ℕ, n > 14 → 8 * n ≥ 120) ∧ (8 * 14 < 120) := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_120_over_8_l3398_339874


namespace NUMINAMATH_CALUDE_set_operations_l3398_339863

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

theorem set_operations :
  (A ∪ B = {x | x ≥ 3}) ∧
  (A ∩ B = {x | 4 ≤ x ∧ x < 10}) ∧
  ((Aᶜ ∩ B) ∩ (A ∪ B) = {x | (3 ≤ x ∧ x < 4) ∨ x ≥ 10}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3398_339863


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l3398_339884

theorem smallest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → n ≥ 1013 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l3398_339884


namespace NUMINAMATH_CALUDE_empty_vessel_possible_l3398_339831

/-- Represents a state of water distribution among three vessels --/
structure WaterState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a pouring operation from one vessel to another --/
inductive PouringOperation
  | FromAToB
  | FromAToC
  | FromBToA
  | FromBToC
  | FromCToA
  | FromCToB

/-- Applies a pouring operation to a water state --/
def applyPouring (state : WaterState) (op : PouringOperation) : WaterState :=
  match op with
  | PouringOperation.FromAToB => 
      if state.a ≤ state.b then {a := 0, b := state.b + state.a, c := state.c}
      else {a := state.a - state.b, b := 2 * state.b, c := state.c}
  | PouringOperation.FromAToC => 
      if state.a ≤ state.c then {a := 0, b := state.b, c := state.c + state.a}
      else {a := state.a - state.c, b := state.b, c := 2 * state.c}
  | PouringOperation.FromBToA => 
      if state.b ≤ state.a then {a := state.a + state.b, b := 0, c := state.c}
      else {a := 2 * state.a, b := state.b - state.a, c := state.c}
  | PouringOperation.FromBToC => 
      if state.b ≤ state.c then {a := state.a, b := 0, c := state.c + state.b}
      else {a := state.a, b := state.b - state.c, c := 2 * state.c}
  | PouringOperation.FromCToA => 
      if state.c ≤ state.a then {a := state.a + state.c, b := state.b, c := 0}
      else {a := 2 * state.a, b := state.b, c := state.c - state.a}
  | PouringOperation.FromCToB => 
      if state.c ≤ state.b then {a := state.a, b := state.b + state.c, c := 0}
      else {a := state.a, b := 2 * state.b, c := state.c - state.b}

/-- Predicate to check if a water state has an empty vessel --/
def hasEmptyVessel (state : WaterState) : Prop :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- The main theorem stating that it's always possible to empty a vessel --/
theorem empty_vessel_possible (initialState : WaterState) : 
  ∃ (operations : List PouringOperation), 
    hasEmptyVessel (operations.foldl applyPouring initialState) :=
  sorry

end NUMINAMATH_CALUDE_empty_vessel_possible_l3398_339831


namespace NUMINAMATH_CALUDE_tan_5460_deg_equals_sqrt_3_l3398_339890

theorem tan_5460_deg_equals_sqrt_3 : Real.tan (5460 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_5460_deg_equals_sqrt_3_l3398_339890


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3398_339879

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -4 ∧ x₂ = -4.5 ∧ 
  (∀ x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 7) ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3398_339879


namespace NUMINAMATH_CALUDE_least_prime_factor_of_9_5_plus_9_4_l3398_339826

theorem least_prime_factor_of_9_5_plus_9_4 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (9^5 + 9^4) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (9^5 + 9^4) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_9_5_plus_9_4_l3398_339826


namespace NUMINAMATH_CALUDE_divisibility_of_powers_l3398_339815

-- Define the polynomial and its greatest positive root
def f (x : ℝ) := x^3 - 3*x^2 + 1

def a : ℝ := sorry

axiom a_is_root : f a = 0

axiom a_is_greatest_positive_root : 
  ∀ x > 0, f x = 0 → x ≤ a

-- Define the floor function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem divisibility_of_powers : 
  (17 ∣ floor (a^1788)) ∧ (17 ∣ floor (a^1988)) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_powers_l3398_339815


namespace NUMINAMATH_CALUDE_rhombuses_count_in_5x5_grid_l3398_339813

/-- A grid composed of equilateral triangles -/
structure TriangleGrid where
  size : ℕ
  /-- The grid is composed of size^2 small equilateral triangles -/
  triangles_count : size^2 = 25

/-- A function to count the number of rhombuses in a TriangleGrid -/
def count_rhombuses (grid : TriangleGrid) : ℕ := sorry

/-- Theorem stating that the number of rhombuses in a 5x5 TriangleGrid is 30 -/
theorem rhombuses_count_in_5x5_grid (grid : TriangleGrid) :
  grid.size = 5 → count_rhombuses grid = 30 := by sorry

end NUMINAMATH_CALUDE_rhombuses_count_in_5x5_grid_l3398_339813


namespace NUMINAMATH_CALUDE_percentage_increase_l3398_339851

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 500 → final = 650 → (final - initial) / initial * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3398_339851


namespace NUMINAMATH_CALUDE_joan_next_birthday_age_l3398_339855

theorem joan_next_birthday_age
  (joan larry kim : ℝ)
  (h1 : joan = 1.3 * larry)
  (h2 : larry = 0.75 * kim)
  (h3 : joan + larry + kim = 39)
  : ⌊joan⌋ + 1 = 15 :=
sorry

end NUMINAMATH_CALUDE_joan_next_birthday_age_l3398_339855


namespace NUMINAMATH_CALUDE_line_point_z_coordinate_l3398_339857

/-- Given a line passing through two points in 3D space, find the z-coordinate of a point on the line with a specific x-coordinate. -/
theorem line_point_z_coordinate 
  (p1 : ℝ × ℝ × ℝ) 
  (p2 : ℝ × ℝ × ℝ) 
  (x : ℝ) : 
  p1 = (1, 3, 2) → 
  p2 = (4, 2, -1) → 
  x = 3 → 
  ∃ (y z : ℝ), (∃ (t : ℝ), 
    (1 + 3*t, 3 - t, 2 - 3*t) = (x, y, z) ∧ 
    z = 0) := by
  sorry

#check line_point_z_coordinate

end NUMINAMATH_CALUDE_line_point_z_coordinate_l3398_339857


namespace NUMINAMATH_CALUDE_power_of_four_remainder_l3398_339835

theorem power_of_four_remainder (a : ℕ+) (p : ℕ) :
  p = 4^(a : ℕ) → p % 10 = 6 → ∃ k : ℕ, (a : ℕ) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_remainder_l3398_339835


namespace NUMINAMATH_CALUDE_number_cube_problem_l3398_339801

theorem number_cube_problem (N : ℝ) : 
  (0.05 * N = 220) → 
  ((1.3 * N / 20) ^ 3 = 23393616) := by
sorry

end NUMINAMATH_CALUDE_number_cube_problem_l3398_339801


namespace NUMINAMATH_CALUDE_sphere_tangent_plane_distance_l3398_339834

/-- Given three spheres where two smaller spheres touch each other externally and
    each touches a larger sphere internally, with radii as specified,
    the distance from the center of the largest sphere to the tangent plane
    at the touching point of the smaller spheres is R/5. -/
theorem sphere_tangent_plane_distance (R : ℝ) : ℝ := by
  -- Define the radii of the smaller spheres
  let r₁ := R / 2
  let r₂ := R / 3
  
  -- Define the distance from the center of the largest sphere
  -- to the tangent plane at the touching point of the smaller spheres
  let d : ℝ := R / 5
  
  -- The proof would go here
  sorry

#check sphere_tangent_plane_distance

end NUMINAMATH_CALUDE_sphere_tangent_plane_distance_l3398_339834


namespace NUMINAMATH_CALUDE_pool_width_l3398_339878

theorem pool_width (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 30 ∧ length = 10 ∧ area = length * width → width = 3 := by
  sorry

end NUMINAMATH_CALUDE_pool_width_l3398_339878


namespace NUMINAMATH_CALUDE_license_plate_count_l3398_339821

/-- The number of consonants available for the first character -/
def num_consonants : ℕ := 20

/-- The number of vowels available for the second and third characters -/
def num_vowels : ℕ := 6

/-- The number of digits and special symbols available for the fourth character -/
def num_digits_and_symbols : ℕ := 12

/-- The total number of possible license plates -/
def total_plates : ℕ := num_consonants * num_vowels * num_vowels * num_digits_and_symbols

theorem license_plate_count : total_plates = 103680 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3398_339821


namespace NUMINAMATH_CALUDE_b_formula_l3398_339862

/-- Sequence a_n defined recursively --/
def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 / (a n + 1)

/-- Sequence b_n defined in terms of a_n --/
def b (n : ℕ) : ℚ := |((a n + 2) / (a n - 1))|

/-- The main theorem to be proved --/
theorem b_formula (n : ℕ) : b n = 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_b_formula_l3398_339862


namespace NUMINAMATH_CALUDE_lcm_gcd_product_12_15_l3398_339897

theorem lcm_gcd_product_12_15 : Nat.lcm 12 15 * Nat.gcd 12 15 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_12_15_l3398_339897


namespace NUMINAMATH_CALUDE_square_root_of_1708249_l3398_339853

theorem square_root_of_1708249 : 
  Real.sqrt 1708249 = 1307 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1708249_l3398_339853


namespace NUMINAMATH_CALUDE_certain_number_l3398_339891

theorem certain_number (X : ℝ) (h : 45 * 8 = 0.40 * X) : X = 900 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_l3398_339891


namespace NUMINAMATH_CALUDE_cook_selection_l3398_339804

theorem cook_selection (total : ℕ) (vegetarians : ℕ) (cooks : ℕ) :
  total = 10 → vegetarians = 3 → cooks = 2 →
  (Nat.choose vegetarians 1) * (Nat.choose (total - 1) 1) = 27 :=
by sorry

end NUMINAMATH_CALUDE_cook_selection_l3398_339804


namespace NUMINAMATH_CALUDE_organization_size_after_five_years_l3398_339824

def organization_growth (initial_members : ℕ) (initial_leaders : ℕ) (years : ℕ) : ℕ :=
  let rec growth (year : ℕ) (members : ℕ) : ℕ :=
    if year = 0 then
      members
    else
      growth (year - 1) (4 * members - 18)
  growth years initial_members

theorem organization_size_after_five_years :
  organization_growth 12 6 5 = 6150 := by
  sorry

end NUMINAMATH_CALUDE_organization_size_after_five_years_l3398_339824


namespace NUMINAMATH_CALUDE_prime_condition_characterization_l3398_339802

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The main theorem statement -/
theorem prime_condition_characterization (n : ℕ) :
  (n > 0 ∧ ∀ k : ℕ, k < n → is_prime (4 * k^2 + n)) ↔ (n = 3 ∨ n = 7) :=
sorry

end NUMINAMATH_CALUDE_prime_condition_characterization_l3398_339802


namespace NUMINAMATH_CALUDE_allan_bought_three_balloons_l3398_339822

/-- The number of balloons Allan bought at the park -/
def balloons_bought_at_park (initial_balloons final_balloons : ℕ) : ℕ :=
  final_balloons - initial_balloons

/-- Theorem stating that Allan bought 3 balloons at the park -/
theorem allan_bought_three_balloons :
  balloons_bought_at_park 5 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_allan_bought_three_balloons_l3398_339822


namespace NUMINAMATH_CALUDE_stratified_sampling_arrangements_l3398_339889

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4

theorem stratified_sampling_arrangements :
  (Nat.choose total_balls black_balls) = number_of_arrangements :=
by sorry

#check stratified_sampling_arrangements

end NUMINAMATH_CALUDE_stratified_sampling_arrangements_l3398_339889


namespace NUMINAMATH_CALUDE_shift_increasing_interval_l3398_339833

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem shift_increasing_interval :
  IncreasingOn f (-2) 3 → IncreasingOn (fun x ↦ f (x + 5)) (-7) (-2) :=
by
  sorry

end NUMINAMATH_CALUDE_shift_increasing_interval_l3398_339833


namespace NUMINAMATH_CALUDE_region_is_lower_left_l3398_339887

-- Define the line
def line (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x + y - 6 < 0

-- Define a point on the lower left side of the line
def lower_left_point (x y : ℝ) : Prop := x + y < 6

-- Theorem stating that the region is on the lower left side of the line
theorem region_is_lower_left :
  ∀ (x y : ℝ), region x y ↔ lower_left_point x y :=
sorry

end NUMINAMATH_CALUDE_region_is_lower_left_l3398_339887


namespace NUMINAMATH_CALUDE_vectors_opposite_direction_l3398_339843

def a : ℝ × ℝ := (-2, 4)
def b : ℝ × ℝ := (1, -2)

theorem vectors_opposite_direction :
  ∃ k : ℝ, k < 0 ∧ a = (k • b) := by sorry

end NUMINAMATH_CALUDE_vectors_opposite_direction_l3398_339843


namespace NUMINAMATH_CALUDE_min_groups_for_athletes_l3398_339825

theorem min_groups_for_athletes (total_athletes : ℕ) (max_group_size : ℕ) (h1 : total_athletes = 30) (h2 : max_group_size = 12) : 
  ∃ (num_groups : ℕ), 
    num_groups ≥ 1 ∧ 
    num_groups ≤ total_athletes ∧
    ∃ (group_size : ℕ), 
      group_size > 0 ∧
      group_size ≤ max_group_size ∧
      total_athletes = num_groups * group_size ∧
      ∀ (n : ℕ), n < num_groups → 
        ¬∃ (g : ℕ), g > 0 ∧ g ≤ max_group_size ∧ total_athletes = n * g :=
by
  sorry

end NUMINAMATH_CALUDE_min_groups_for_athletes_l3398_339825


namespace NUMINAMATH_CALUDE_zoe_gre_exam_month_l3398_339827

-- Define the months as an enumeration
inductive Month
  | January | February | March | April | May | June | July | August | September | October | November | December

-- Define a function to add months
def addMonths (start : Month) (n : Nat) : Month :=
  match n with
  | 0 => start
  | Nat.succ m => addMonths (match start with
    | Month.January => Month.February
    | Month.February => Month.March
    | Month.March => Month.April
    | Month.April => Month.May
    | Month.May => Month.June
    | Month.June => Month.July
    | Month.July => Month.August
    | Month.August => Month.September
    | Month.September => Month.October
    | Month.October => Month.November
    | Month.November => Month.December
    | Month.December => Month.January
  ) m

-- Theorem statement
theorem zoe_gre_exam_month :
  addMonths Month.April 2 = Month.June :=
by sorry

end NUMINAMATH_CALUDE_zoe_gre_exam_month_l3398_339827


namespace NUMINAMATH_CALUDE_circus_ticket_price_l3398_339844

theorem circus_ticket_price :
  let total_tickets : ℕ := 522
  let child_ticket_price : ℚ := 8
  let total_receipts : ℚ := 5086
  let adult_tickets_sold : ℕ := 130
  let child_tickets_sold : ℕ := total_tickets - adult_tickets_sold
  let adult_ticket_price : ℚ := (total_receipts - child_ticket_price * child_tickets_sold) / adult_tickets_sold
  adult_ticket_price = 15 := by
sorry

end NUMINAMATH_CALUDE_circus_ticket_price_l3398_339844


namespace NUMINAMATH_CALUDE_blackboard_numbers_l3398_339861

/-- Represents the state of the blackboard after n steps -/
def BlackboardState (n : ℕ) : Type := List ℕ

/-- The rule for updating the blackboard -/
def updateBlackboard (state : BlackboardState n) : BlackboardState (n + 1) :=
  sorry

/-- The number of numbers on the blackboard after n steps -/
def f (n : ℕ) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem blackboard_numbers (n : ℕ) : 
  f n = (1 / 2 : ℚ) * Nat.choose (2 * n + 2) (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l3398_339861


namespace NUMINAMATH_CALUDE_major_axis_length_l3398_339893

/-- Represents a right circular cylinder --/
structure RightCircularCylinder where
  radius : ℝ

/-- Represents an ellipse formed by intersecting a plane with a cylinder --/
structure Ellipse where
  minorAxis : ℝ
  majorAxis : ℝ

/-- The ellipse formed by intersecting a plane with a right circular cylinder --/
def cylinderEllipse (c : RightCircularCylinder) : Ellipse :=
  { minorAxis := 2 * c.radius,
    majorAxis := 3 * c.radius }

theorem major_axis_length (c : RightCircularCylinder) 
  (h : c.radius = 1) :
  (cylinderEllipse c).majorAxis = 3 ∧
  (cylinderEllipse c).majorAxis = 1.5 * (cylinderEllipse c).minorAxis :=
by sorry

end NUMINAMATH_CALUDE_major_axis_length_l3398_339893


namespace NUMINAMATH_CALUDE_intersection_M_N_l3398_339896

def M : Set ℝ := {x : ℝ | x^2 - 3*x = 0}
def N : Set ℝ := {-1, 1, 3}

theorem intersection_M_N : M ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3398_339896


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1140_l3398_339875

theorem sum_of_largest_and_smallest_prime_factors_of_1140 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 1140 ∧ largest ∣ 1140 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1140 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1140 → p ≥ smallest) ∧
    smallest + largest = 21 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1140_l3398_339875


namespace NUMINAMATH_CALUDE_function_property_l3398_339832

theorem function_property (k : ℝ) (h_k : k > 0) :
  ∀ (f : ℝ → ℝ), 
  (∀ (x : ℝ), x > 0 → (f (x^2 + 1))^(Real.sqrt x) = k) →
  ∀ (y : ℝ), y > 0 → (f ((9 + y^2) / y^2))^(Real.sqrt (12 / y)) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3398_339832


namespace NUMINAMATH_CALUDE_complex_equation_product_l3398_339886

theorem complex_equation_product (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (a + 2*i)/i = b + i) : a * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_product_l3398_339886


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3398_339820

/-- Proves that the speed of a boat in still water is 22 km/hr, given the conditions -/
theorem boat_speed_in_still_water :
  let stream_speed : ℝ := 5
  let downstream_distance : ℝ := 108
  let downstream_time : ℝ := 4
  let downstream_speed : ℝ := downstream_distance / downstream_time
  let boat_speed_still : ℝ := downstream_speed - stream_speed
  boat_speed_still = 22 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3398_339820


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l3398_339842

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical :
  let x : ℝ := -5
  let y : ℝ := 0
  let z : ℝ := -8
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.pi
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 5 ∧ θ = Real.pi ∧ z = -8 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l3398_339842


namespace NUMINAMATH_CALUDE_car_speed_is_45_l3398_339845

/-- Represents the scenario of a car and motorcyclist journey --/
structure Journey where
  distance : ℝ  -- Distance from A to B in km
  moto_speed : ℝ  -- Motorcyclist's speed in km/h
  delay : ℝ  -- Delay before motorcyclist starts in hours
  car_speed : ℝ  -- Car's speed in km/h (to be proven)

/-- Theorem stating that under given conditions, the car's speed is 45 km/h --/
theorem car_speed_is_45 (j : Journey) 
  (h1 : j.distance = 82.5)
  (h2 : j.moto_speed = 60)
  (h3 : j.delay = 1/3)
  (h4 : ∃ t : ℝ, 
    t > 0 ∧ 
    j.car_speed * (t + j.delay) = j.moto_speed * t ∧ 
    (j.distance - j.moto_speed * t) / j.car_speed = t / 2) :
  j.car_speed = 45 := by
sorry


end NUMINAMATH_CALUDE_car_speed_is_45_l3398_339845


namespace NUMINAMATH_CALUDE_quiz_probability_l3398_339810

theorem quiz_probability : 
  let n : ℕ := 6  -- number of questions
  let m : ℕ := 6  -- number of possible answers per question
  let p : ℚ := 1 - (m - 1 : ℚ) / m  -- probability of getting one question right
  1 - (1 - p) ^ n = 31031 / 46656 :=
by sorry

end NUMINAMATH_CALUDE_quiz_probability_l3398_339810


namespace NUMINAMATH_CALUDE_dragon_jewel_ratio_l3398_339819

theorem dragon_jewel_ratio :
  ∀ (initial_jewels : ℕ),
    initial_jewels - 3 + 6 = 24 →
    (6 : ℚ) / initial_jewels = 2 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_dragon_jewel_ratio_l3398_339819


namespace NUMINAMATH_CALUDE_book_club_snack_fee_l3398_339840

theorem book_club_snack_fee (members : ℕ) (hardcover_price paperback_price : ℚ)
  (hardcover_count paperback_count : ℕ) (total_collected : ℚ) :
  members = 6 →
  hardcover_price = 30 →
  paperback_price = 12 →
  hardcover_count = 6 →
  paperback_count = 6 →
  total_collected = 2412 →
  (total_collected - members * (hardcover_price * hardcover_count + paperback_price * paperback_count)) / members = 150 := by
  sorry

#check book_club_snack_fee

end NUMINAMATH_CALUDE_book_club_snack_fee_l3398_339840


namespace NUMINAMATH_CALUDE_florist_bouquets_l3398_339871

/-- The number of flower colors --/
def num_colors : ℕ := 4

/-- The number of flowers in each bouquet --/
def flowers_per_bouquet : ℕ := 9

/-- The number of seeds planted for each color --/
def seeds_per_color : ℕ := 125

/-- The number of red flowers killed by fungus --/
def red_killed : ℕ := 45

/-- The number of yellow flowers killed by fungus --/
def yellow_killed : ℕ := 61

/-- The number of orange flowers killed by fungus --/
def orange_killed : ℕ := 30

/-- The number of purple flowers killed by fungus --/
def purple_killed : ℕ := 40

/-- Theorem: The florist can make 36 bouquets --/
theorem florist_bouquets :
  (num_colors * seeds_per_color - (red_killed + yellow_killed + orange_killed + purple_killed)) / flowers_per_bouquet = 36 :=
by sorry

end NUMINAMATH_CALUDE_florist_bouquets_l3398_339871


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l3398_339858

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 143 → x^2 + y^2 ≥ 145 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l3398_339858


namespace NUMINAMATH_CALUDE_select_five_from_eight_l3398_339850

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l3398_339850


namespace NUMINAMATH_CALUDE_golden_state_total_points_l3398_339805

/-- The Golden State Team's total points calculation -/
theorem golden_state_total_points :
  let draymond_points : ℕ := 12
  let curry_points : ℕ := 2 * draymond_points
  let kelly_points : ℕ := 9
  let durant_points : ℕ := 2 * kelly_points
  let klay_points : ℕ := draymond_points / 2
  draymond_points + curry_points + kelly_points + durant_points + klay_points = 69 := by
  sorry

end NUMINAMATH_CALUDE_golden_state_total_points_l3398_339805


namespace NUMINAMATH_CALUDE_negation_equivalence_l3398_339899

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3398_339899


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3398_339895

theorem quadratic_roots_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3398_339895


namespace NUMINAMATH_CALUDE_horatio_sonnets_l3398_339876

def sonnet_lines : ℕ := 16
def sonnets_read : ℕ := 9
def unread_lines : ℕ := 126

theorem horatio_sonnets :
  ∃ (total_sonnets : ℕ),
    total_sonnets * sonnet_lines = sonnets_read * sonnet_lines + unread_lines ∧
    total_sonnets = 16 := by
  sorry

end NUMINAMATH_CALUDE_horatio_sonnets_l3398_339876


namespace NUMINAMATH_CALUDE_same_number_probability_l3398_339894

/-- The upper bound for the selected numbers -/
def upperBound : ℕ := 300

/-- Billy's number is a multiple of this value -/
def billyMultiple : ℕ := 36

/-- Bobbi's number is a multiple of this value -/
def bobbiMultiple : ℕ := 48

/-- The probability of Billy and Bobbi selecting the same number -/
def sameProbability : ℚ := 1 / 24

theorem same_number_probability :
  (∃ (b₁ b₂ : ℕ), b₁ > 0 ∧ b₂ > 0 ∧ b₁ < upperBound ∧ b₂ < upperBound ∧
    b₁ % billyMultiple = 0 ∧ b₂ % bobbiMultiple = 0) →
  (∃ (n : ℕ), n > 0 ∧ n < upperBound ∧ n % billyMultiple = 0 ∧ n % bobbiMultiple = 0) →
  sameProbability = (Nat.card {n : ℕ | n > 0 ∧ n < upperBound ∧ n % billyMultiple = 0 ∧ n % bobbiMultiple = 0} : ℚ) /
                    ((Nat.card {n : ℕ | n > 0 ∧ n < upperBound ∧ n % billyMultiple = 0} : ℚ) *
                     (Nat.card {n : ℕ | n > 0 ∧ n < upperBound ∧ n % bobbiMultiple = 0} : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_same_number_probability_l3398_339894


namespace NUMINAMATH_CALUDE_john_quilt_cost_l3398_339872

/-- Calculates the cost of a rectangular quilt given its dimensions and cost per square foot -/
def quiltCost (length width costPerSqFt : ℝ) : ℝ :=
  length * width * costPerSqFt

/-- Proves that a 7ft by 8ft quilt at $40 per square foot costs $2240 -/
theorem john_quilt_cost :
  quiltCost 7 8 40 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_john_quilt_cost_l3398_339872


namespace NUMINAMATH_CALUDE_intersection_S_T_l3398_339849

def S : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}
def T : Set ℝ := {x : ℝ | x + 2 ≤ 3}

theorem intersection_S_T : S ∩ T = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l3398_339849


namespace NUMINAMATH_CALUDE_value_of_y_l3398_339803

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 14 ∧ y = 98 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3398_339803


namespace NUMINAMATH_CALUDE_basketball_substitutions_l3398_339839

/-- The number of possible substitution methods in a basketball game with specific rules -/
def substitution_methods (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitute_players := total_players - starting_players
  1 + -- No substitutions
  (starting_players * substitute_players) + -- One substitution
  (starting_players * (starting_players - 1) * substitute_players * (substitute_players - 1) / 2) + -- Two substitutions
  (starting_players * (starting_players - 1) * (starting_players - 2) * substitute_players * (substitute_players - 1) * (substitute_players - 2) / 6) -- Three substitutions

/-- The main theorem stating the number of substitution methods and its remainder when divided by 1000 -/
theorem basketball_substitutions :
  let m := substitution_methods 18 9 3
  m = 45010 ∧ m % 1000 = 10 := by
  sorry


end NUMINAMATH_CALUDE_basketball_substitutions_l3398_339839


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3398_339816

theorem other_root_of_quadratic (m : ℝ) : 
  ((-1)^2 + (-1) + m = 0) → 
  (∃ (x : ℝ), x ≠ -1 ∧ x^2 + x + m = 0 ∧ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3398_339816


namespace NUMINAMATH_CALUDE_fred_has_61_cards_l3398_339859

/-- The number of baseball cards Fred has after all transactions -/
def fred_final_cards (initial : ℕ) (given_to_mary : ℕ) (found_in_box : ℕ) (given_to_john : ℕ) (purchased : ℕ) : ℕ :=
  initial - given_to_mary + found_in_box - given_to_john + purchased

/-- Theorem stating that Fred ends up with 61 cards -/
theorem fred_has_61_cards :
  fred_final_cards 26 18 40 12 25 = 61 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_61_cards_l3398_339859


namespace NUMINAMATH_CALUDE_percentage_equality_l3398_339882

theorem percentage_equality (x : ℝ) : (15 / 100 * 75 = 2.5 / 100 * x) → x = 450 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3398_339882


namespace NUMINAMATH_CALUDE_max_teams_is_six_l3398_339873

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played in the tournament -/
def max_games : ℕ := 150

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- The function to calculate the number of games for a given number of teams -/
def total_games (n : ℕ) : ℕ := n.choose 2 * games_between_teams

/-- The theorem stating that 6 is the maximum number of teams that can participate -/
theorem max_teams_is_six :
  ∀ n : ℕ, n > 6 → total_games n > max_games ∧
  total_games 6 ≤ max_games :=
sorry

end NUMINAMATH_CALUDE_max_teams_is_six_l3398_339873


namespace NUMINAMATH_CALUDE_solution_to_logarithmic_equation_l3398_339881

theorem solution_to_logarithmic_equation :
  ∃ x : ℝ, (3 * Real.log x - 4 * Real.log 5 = -1) ∧ (x = (62.5 : ℝ) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_logarithmic_equation_l3398_339881


namespace NUMINAMATH_CALUDE_relationship_order_l3398_339885

theorem relationship_order (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := by
  sorry

end NUMINAMATH_CALUDE_relationship_order_l3398_339885


namespace NUMINAMATH_CALUDE_sum_of_prime_factor_exponents_l3398_339867

/-- The sum of exponents in the given expression of prime factors -/
def sum_of_exponents : ℕ :=
  9 + 5 + 7 + 4 + 6 + 3 + 5 + 2

/-- The theorem states that the sum of exponents in the given expression equals 41 -/
theorem sum_of_prime_factor_exponents : sum_of_exponents = 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factor_exponents_l3398_339867


namespace NUMINAMATH_CALUDE_geometric_sum_n1_l3398_339868

theorem geometric_sum_n1 (a : ℝ) (h : a ≠ 1) :
  1 + a + a^2 = (1 - a^3) / (1 - a) := by sorry

end NUMINAMATH_CALUDE_geometric_sum_n1_l3398_339868


namespace NUMINAMATH_CALUDE_linear_regression_change_l3398_339814

/-- Given a linear regression equation y = 2 - 1.5x, prove that when x increases by 1, y decreases by 1.5. -/
theorem linear_regression_change (x y : ℝ) : 
  y = 2 - 1.5 * x → (2 - 1.5 * (x + 1)) = y - 1.5 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_change_l3398_339814


namespace NUMINAMATH_CALUDE_complex_power_2015_l3398_339818

/-- Given a complex number i such that i^2 = -1, i^3 = -i, and i^4 = 1,
    prove that i^2015 = -i -/
theorem complex_power_2015 (i : ℂ) (hi2 : i^2 = -1) (hi3 : i^3 = -i) (hi4 : i^4 = 1) :
  i^2015 = -i := by sorry

end NUMINAMATH_CALUDE_complex_power_2015_l3398_339818


namespace NUMINAMATH_CALUDE_total_cars_produced_l3398_339829

/-- The total number of cars produced in North America, Europe, and Asia is 9972. -/
theorem total_cars_produced (north_america europe asia : ℕ) 
  (h1 : north_america = 3884)
  (h2 : europe = 2871)
  (h3 : asia = 3217) : 
  north_america + europe + asia = 9972 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_produced_l3398_339829


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l3398_339888

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A regular 9-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l3398_339888


namespace NUMINAMATH_CALUDE_duck_profit_is_170_l3398_339807

/-- Calculates the profit from selling ducks given the specified conditions -/
def duck_profit : ℕ :=
  let first_group_weight := 10 * 3
  let second_group_weight := 10 * 4
  let third_group_weight := 10 * 5
  let total_weight := first_group_weight + second_group_weight + third_group_weight

  let first_group_cost := 10 * 9
  let second_group_cost := 10 * 10
  let third_group_cost := 10 * 12
  let total_cost := first_group_cost + second_group_cost + third_group_cost

  let selling_price_per_pound := 5
  let total_selling_price := total_weight * selling_price_per_pound
  let discount_rate := 20
  let discount_amount := total_selling_price * discount_rate / 100
  let final_selling_price := total_selling_price - discount_amount

  final_selling_price - total_cost

theorem duck_profit_is_170 : duck_profit = 170 := by
  sorry

end NUMINAMATH_CALUDE_duck_profit_is_170_l3398_339807


namespace NUMINAMATH_CALUDE_factorization_problems_l3398_339800

theorem factorization_problems (a m x y : ℝ) :
  (ax^2 - 4*a = a*(x+2)*(x-2)) ∧
  (m*x^2 + 2*m*x*y + m*y^2 = m*(x+y)^2) ∧
  ((1/2)*a^2 - a*b + (1/2)*b^2 = (1/2)*(a-b)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l3398_339800


namespace NUMINAMATH_CALUDE_field_ratio_l3398_339809

/-- Proves that a rectangular field with perimeter 360 meters and width 75 meters has a length-to-width ratio of 7:5 -/
theorem field_ratio (perimeter width : ℝ) (h_perimeter : perimeter = 360) (h_width : width = 75) :
  let length := (perimeter - 2 * width) / 2
  (length / width) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l3398_339809


namespace NUMINAMATH_CALUDE_average_speed_last_segment_l3398_339869

theorem average_speed_last_segment (total_distance : ℝ) (total_time : ℝ) 
  (speed_first : ℝ) (speed_second : ℝ) :
  total_distance = 108 ∧ 
  total_time = 1.5 ∧ 
  speed_first = 70 ∧ 
  speed_second = 60 → 
  ∃ speed_last : ℝ, 
    speed_last = 86 ∧ 
    (speed_first + speed_second + speed_last) / 3 = total_distance / total_time :=
by sorry

end NUMINAMATH_CALUDE_average_speed_last_segment_l3398_339869


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3398_339880

theorem simplify_and_rationalize : 
  (Real.sqrt 7 / Real.sqrt 3) * (Real.sqrt 8 / Real.sqrt 5) * (Real.sqrt 9 / Real.sqrt 7) = 2 * Real.sqrt 30 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3398_339880


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3398_339817

theorem p_sufficient_not_necessary_for_q 
  (h1 : p → q) 
  (h2 : ¬(¬p → ¬q)) : 
  (∃ (x : Prop), x → q) ∧ (∃ (y : Prop), q ∧ ¬y) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3398_339817


namespace NUMINAMATH_CALUDE_cubic_equation_solution_range_l3398_339838

theorem cubic_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^3 - 3*x + m = 0) → m ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_range_l3398_339838


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l3398_339828

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = 180 ∧ -- Angle sum theorem
  a / Real.sin A = b / Real.sin B ∧ -- Sine rule
  a / Real.sin A = c / Real.sin C -- Sine rule

-- State the theorem
theorem longest_side_of_triangle :
  ∀ (A B C a b c : ℝ),
  triangle_ABC A B C a b c →
  B = 135 →
  C = 15 →
  a = 5 →
  b = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l3398_339828


namespace NUMINAMATH_CALUDE_final_comfortable_butterflies_l3398_339823

/-- Represents a point in the 2D lattice -/
structure LatticePoint where
  x : ℕ
  y : ℕ

/-- Represents the state of the lattice at any given time -/
def LatticeState := LatticePoint → Bool

/-- The neighborhood of a lattice point -/
def neighborhood (n : ℕ) (c : LatticePoint) : Set LatticePoint :=
  sorry

/-- Checks if a butterfly at a given point is lonely -/
def isLonely (n : ℕ) (state : LatticeState) (p : LatticePoint) : Bool :=
  sorry

/-- Simulates the process of lonely butterflies flying away -/
def simulateProcess (n : ℕ) (initialState : LatticeState) : LatticeState :=
  sorry

/-- Counts the number of comfortable butterflies in the final state -/
def countComfortableButterflies (n : ℕ) (finalState : LatticeState) : ℕ :=
  sorry

/-- The main theorem stating that the number of comfortable butterflies in the final state is n -/
theorem final_comfortable_butterflies (n : ℕ) (h : n > 0) :
  countComfortableButterflies n (simulateProcess n (λ _ => true)) = n :=
sorry

end NUMINAMATH_CALUDE_final_comfortable_butterflies_l3398_339823


namespace NUMINAMATH_CALUDE_pizza_sharing_ratio_l3398_339808

theorem pizza_sharing_ratio (total_slices : ℕ) (waiter_slices : ℕ) : 
  total_slices = 78 → 
  waiter_slices - 20 = 28 → 
  (total_slices - waiter_slices) / waiter_slices = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_sharing_ratio_l3398_339808


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3398_339860

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define the theorem
theorem quadratic_function_properties (a : ℝ) (h1 : 1/3 ≤ a) (h2 : a ≤ 1) :
  -- 1. Minimum value of f(x) on [1, 3]
  (∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ ∀ (y : ℝ), y ∈ Set.Icc 1 3 → f a x ≤ f a y) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ f a x = 1 - 1/a) ∧
  -- 2. Minimum value of M(a) - N(a)
  (∃ (M N : ℝ → ℝ),
    (∀ (x : ℝ), x ∈ Set.Icc 1 3 → f a x ≤ M a) ∧
    (∃ (y : ℝ), y ∈ Set.Icc 1 3 ∧ f a y = M a) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 1 3 → N a ≤ f a x) ∧
    (∃ (z : ℝ), z ∈ Set.Icc 1 3 ∧ f a z = N a) ∧
    (∀ (b : ℝ), 1/3 ≤ b ∧ b ≤ 1 → 1/2 ≤ M b - N b) ∧
    (∃ (c : ℝ), 1/3 ≤ c ∧ c ≤ 1 ∧ M c - N c = 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3398_339860


namespace NUMINAMATH_CALUDE_unfactorable_quadratic_l3398_339854

theorem unfactorable_quadratic : ¬ ∃ (a b c : ℝ), (∀ x : ℝ, x^2 - 10*x - 25 = (a*x + b)*(c*x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_unfactorable_quadratic_l3398_339854


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3398_339883

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0

-- Define the given line
def givenLine (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Define the result circle
def resultCircle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ, 
    (∃ c : ℝ, c > 0 ∧ 
      (∀ x' y' : ℝ, givenCircle x' y' ↔ (x' - 1)^2 + (y' + 2)^2 = c)) ∧ 
    (∃ x₀ y₀ : ℝ, givenLine x₀ y₀ ∧ resultCircle x₀ y₀) ∧
    (∀ x' y' : ℝ, givenLine x' y' → ¬(resultCircle x' y' ∧ ¬(x' = x₀ ∧ y' = y₀))) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3398_339883


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3398_339841

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (1 - 1 / (x + 1)) / (x / (x - 1)) = (x - 1) / (x + 1) ∧
  (2 - 1) / (2 + 1) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3398_339841


namespace NUMINAMATH_CALUDE_smallest_n_terminating_with_2_l3398_339866

def is_terminating_decimal (n : ℕ+) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit_2 (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + 2 + 10 * m

theorem smallest_n_terminating_with_2 :
  ∃ n : ℕ+, 
    is_terminating_decimal n ∧ 
    contains_digit_2 n.val ∧ 
    (∀ m : ℕ+, m < n → ¬(is_terminating_decimal m ∧ contains_digit_2 m.val)) ∧
    n = 2 :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_terminating_with_2_l3398_339866


namespace NUMINAMATH_CALUDE_sine_tangent_inequality_l3398_339856

theorem sine_tangent_inequality (x y : ℝ) : 
  (0 < Real.sin (50 * π / 180) ∧ Real.sin (50 * π / 180) < 1) →
  Real.tan (50 * π / 180) > 1 →
  (Real.sin (50 * π / 180))^x - (Real.tan (50 * π / 180))^x ≤ 
  (Real.sin (50 * π / 180))^(-y) - (Real.tan (50 * π / 180))^(-y) →
  x + y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_tangent_inequality_l3398_339856


namespace NUMINAMATH_CALUDE_min_value_theorem_l3398_339864

theorem min_value_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 →
    a / (Real.sin θ)^(3/2) + b / (Real.cos θ)^(3/2) ≥ (a^(4/7) + b^(4/7))^(7/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3398_339864


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l3398_339898

theorem power_of_three_plus_five_mod_eight :
  (3^100 + 5) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l3398_339898


namespace NUMINAMATH_CALUDE_jordan_run_time_l3398_339812

/-- Given that Jordan runs 4 miles in 1/3 of the time Steve takes to run 6 miles,
    and Steve takes 36 minutes to run 6 miles, prove that Jordan would take 30 minutes to run 10 miles. -/
theorem jordan_run_time (jordan_distance : ℝ) (steve_distance : ℝ) (steve_time : ℝ) 
  (h1 : jordan_distance = 4)
  (h2 : steve_distance = 6)
  (h3 : steve_time = 36)
  (h4 : jordan_distance * steve_time = steve_distance * (steve_time / 3)) :
  (10 : ℝ) * (steve_time / jordan_distance) = 30 := by
  sorry

end NUMINAMATH_CALUDE_jordan_run_time_l3398_339812


namespace NUMINAMATH_CALUDE_product_sum_reciprocals_l3398_339877

theorem product_sum_reciprocals : (3 * 5 * 7) * (1/3 + 1/5 + 1/7) = 71 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_reciprocals_l3398_339877


namespace NUMINAMATH_CALUDE_smallest_n_for_seating_arrangement_l3398_339811

theorem smallest_n_for_seating_arrangement (k : ℕ) : 
  (2 ≤ k) → 
  (∃ n : ℕ, 
    k < n ∧ 
    (2 * (n - 1).factorial * (n - k + 2) = n * (n - 1).factorial) ∧
    (∀ m : ℕ, m < n → 
      (2 ≤ m ∧ k < m) → 
      (2 * (m - 1).factorial * (m - k + 2) ≠ m * (m - 1).factorial))) → 
  (∃ n : ℕ, 
    k < n ∧ 
    (2 * (n - 1).factorial * (n - k + 2) = n * (n - 1).factorial) ∧
    n = 12) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_seating_arrangement_l3398_339811


namespace NUMINAMATH_CALUDE_complex_not_purely_imaginary_l3398_339847

theorem complex_not_purely_imaginary (a : ℝ) : 
  (Complex.mk (a^2 - a - 2) (|a - 1| - 1) ≠ Complex.I * (Complex.mk 0 (|a - 1| - 1))) ↔ 
  (a ≠ -1) := by
  sorry

end NUMINAMATH_CALUDE_complex_not_purely_imaginary_l3398_339847


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3398_339806

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / ((a + b + c)^4 - 79 * (a * b * c)^(4/3))
  A ≤ 3 ∧ ∃ (x : ℝ), x > 0 ∧ 
    let A' := (3 * x^3 * (2 * x)) / ((3 * x)^4 - 79 * x^4)
    A' = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3398_339806


namespace NUMINAMATH_CALUDE_congruence_solution_sum_l3398_339852

theorem congruence_solution_sum (a m : ℕ) : 
  m ≥ 2 → 
  0 ≤ a → 
  a < m → 
  (∀ x : ℤ, (8 * x + 1) % 12 = 5 % 12 ↔ x % m = a % m) → 
  a + m = 5 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_sum_l3398_339852
