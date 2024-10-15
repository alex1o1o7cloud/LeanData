import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_foci_l3799_379924

-- Define the three known endpoints of the ellipse's axes
def point1 : ℝ × ℝ := (-3, 5)
def point2 : ℝ × ℝ := (4, -3)
def point3 : ℝ × ℝ := (9, 5)

-- Define the ellipse based on these points
def ellipse_from_points (p1 p2 p3 : ℝ × ℝ) : Type := sorry

-- Theorem stating the distance between foci
theorem distance_between_foci 
  (e : ellipse_from_points point1 point2 point3) : 
  ∃ (f1 f2 : ℝ × ℝ), dist f1 f2 = 4 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_distance_between_foci_l3799_379924


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_product_l3799_379998

theorem quadratic_equation_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 8 ∧ x * y = 9) →
  m + n = 51 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_product_l3799_379998


namespace NUMINAMATH_CALUDE_relationship_abc_l3799_379901

theorem relationship_abc : 
  let a : ℝ := (3/7)^(2/7)
  let b : ℝ := (2/7)^(3/7)
  let c : ℝ := (2/7)^(2/7)
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3799_379901


namespace NUMINAMATH_CALUDE_worker_overtime_hours_l3799_379926

/-- A worker's pay calculation --/
theorem worker_overtime_hours (regular_rate : ℚ) (regular_hours : ℚ) (overtime_rate : ℚ) (total_pay : ℚ) : 
  regular_rate = 3 →
  regular_hours = 40 →
  overtime_rate = 2 * regular_rate →
  total_pay = 180 →
  (total_pay - regular_rate * regular_hours) / overtime_rate = 10 := by
sorry

end NUMINAMATH_CALUDE_worker_overtime_hours_l3799_379926


namespace NUMINAMATH_CALUDE_cereal_box_price_calculation_l3799_379929

theorem cereal_box_price_calculation 
  (initial_price : ℕ) 
  (price_reduction : ℕ) 
  (num_boxes : ℕ) : 
  initial_price = 104 → 
  price_reduction = 24 → 
  num_boxes = 20 → 
  (initial_price - price_reduction) * num_boxes = 1600 := by
sorry

end NUMINAMATH_CALUDE_cereal_box_price_calculation_l3799_379929


namespace NUMINAMATH_CALUDE_gray_area_trees_count_l3799_379990

/-- Represents a rectangle with trees -/
structure TreeRectangle where
  total_trees : ℕ
  white_area_trees : ℕ

/-- Represents the setup of three overlapping rectangles -/
structure ThreeRectangles where
  rect1 : TreeRectangle
  rect2 : TreeRectangle
  rect3 : TreeRectangle

/-- The total number of trees in the gray (overlapping) areas -/
def gray_area_trees (setup : ThreeRectangles) : ℕ :=
  setup.rect1.total_trees - setup.rect1.white_area_trees +
  setup.rect2.total_trees - setup.rect2.white_area_trees

/-- Theorem stating the total number of trees in the gray areas -/
theorem gray_area_trees_count (setup : ThreeRectangles)
  (h1 : setup.rect1.total_trees = 100)
  (h2 : setup.rect2.total_trees = 100)
  (h3 : setup.rect3.total_trees = 100)
  (h4 : setup.rect1.white_area_trees = 82)
  (h5 : setup.rect2.white_area_trees = 82) :
  gray_area_trees setup = 26 := by
  sorry

end NUMINAMATH_CALUDE_gray_area_trees_count_l3799_379990


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3799_379908

theorem polynomial_simplification (x : ℝ) : 
  (12 * x^10 - 3 * x^9 + 8 * x^8 - 5 * x^7) - 
  (2 * x^10 + 2 * x^9 - x^8 + x^7 + 4 * x^4 + 6 * x^2 + 9) = 
  10 * x^10 - 5 * x^9 + 9 * x^8 - 6 * x^7 - 4 * x^4 - 6 * x^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3799_379908


namespace NUMINAMATH_CALUDE_square_route_distance_l3799_379967

/-- Represents a square route with given side length -/
structure SquareRoute where
  side_length : ℝ

/-- Calculates the total distance traveled in a square route -/
def total_distance (route : SquareRoute) : ℝ :=
  4 * route.side_length

/-- Theorem: The total distance traveled in a square route with sides of 2000 km is 8000 km -/
theorem square_route_distance :
  let route := SquareRoute.mk 2000
  total_distance route = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_route_distance_l3799_379967


namespace NUMINAMATH_CALUDE_currency_notes_problem_l3799_379936

theorem currency_notes_problem :
  ∃ (D : ℕ+) (x y : ℕ),
    x + y = 100 ∧
    70 * x + D * y = 5000 :=
by sorry

end NUMINAMATH_CALUDE_currency_notes_problem_l3799_379936


namespace NUMINAMATH_CALUDE_debut_attendance_is_200_l3799_379985

/-- The number of people who bought tickets for the debut show -/
def debut_attendance : ℕ := sorry

/-- The number of people who bought tickets for the second showing -/
def second_showing_attendance : ℕ := 3 * debut_attendance

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 25

/-- The total revenue from both shows in dollars -/
def total_revenue : ℕ := 20000

/-- Theorem stating that the number of people who bought tickets for the debut show is 200 -/
theorem debut_attendance_is_200 : debut_attendance = 200 := by
  sorry

end NUMINAMATH_CALUDE_debut_attendance_is_200_l3799_379985


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l3799_379904

theorem arithmetic_geometric_sum (a₁ d : ℚ) (g₁ r : ℚ) (n : ℕ) 
  (h₁ : a₁ = 15)
  (h₂ : d = 0.2)
  (h₃ : g₁ = 15)
  (h₄ : r = 2)
  (h₅ : n = 101) :
  (n : ℚ) * (a₁ + (a₁ + (n - 1) * d)) / 2 + g₁ * (r^n - 1) / (r - 1) = 15 * (2^101 - 1) + 2525 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l3799_379904


namespace NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l3799_379961

theorem degenerate_ellipse_max_y_coordinate (x y : ℝ) :
  (x - 3)^2 / 49 + (y - 2)^2 / 25 = 0 → y ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l3799_379961


namespace NUMINAMATH_CALUDE_three_in_range_of_f_l3799_379992

/-- The function f(x) = x^2 + bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 1

/-- Theorem: For all real b, there exists a real x such that f(x) = 3 -/
theorem three_in_range_of_f (b : ℝ) : ∃ x : ℝ, f b x = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_in_range_of_f_l3799_379992


namespace NUMINAMATH_CALUDE_sokka_fish_count_l3799_379902

theorem sokka_fish_count (aang_fish : ℕ) (toph_fish : ℕ) (average_fish : ℕ) (total_people : ℕ) :
  aang_fish = 7 →
  toph_fish = 12 →
  average_fish = 8 →
  total_people = 3 →
  ∃ sokka_fish : ℕ, sokka_fish = total_people * average_fish - (aang_fish + toph_fish) :=
by
  sorry

end NUMINAMATH_CALUDE_sokka_fish_count_l3799_379902


namespace NUMINAMATH_CALUDE_chocolate_differences_l3799_379935

/-- Given the number of chocolates eaten by Robert, Nickel, and Jessica,
    prove the differences between Robert's and Nickel's chocolates,
    and Jessica's and Nickel's chocolates. -/
theorem chocolate_differences (robert nickel jessica : ℕ) 
    (h_robert : robert = 23)
    (h_nickel : nickel = 8)
    (h_jessica : jessica = 15) :
    robert - nickel = 15 ∧ jessica - nickel = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_differences_l3799_379935


namespace NUMINAMATH_CALUDE_fencing_probability_theorem_l3799_379991

/-- Represents the increase in winning probability for player A in a fencing match -/
def fencing_probability_increase (k l : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose (k + l) k) * (1 - p)^(l + 1) * p^k

/-- Theorem stating the increase in winning probability for player A in a fencing match -/
theorem fencing_probability_theorem (k l : ℕ) (p : ℝ) 
    (h1 : 0 ≤ k ∧ k ≤ 14) (h2 : 0 ≤ l ∧ l ≤ 14) (h3 : 0 ≤ p ∧ p ≤ 1) : 
  fencing_probability_increase k l p = 
    (Nat.choose (k + l) k) * (1 - p)^(l + 1) * p^k := by
  sorry

#check fencing_probability_theorem

end NUMINAMATH_CALUDE_fencing_probability_theorem_l3799_379991


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l3799_379951

theorem base_10_to_base_7 : 
  (2 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0 : ℕ) = 789 := by
sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l3799_379951


namespace NUMINAMATH_CALUDE_daves_initial_apps_daves_initial_apps_proof_l3799_379945

theorem daves_initial_apps : ℕ :=
  let initial_files : ℕ := 9
  let final_files : ℕ := 5
  let final_apps : ℕ := 12
  let app_file_difference : ℕ := 7

  have h1 : final_apps = final_files + app_file_difference := by sorry
  have h2 : ∃ (initial_apps : ℕ), initial_apps - final_apps = initial_files - final_files := by sorry

  16

theorem daves_initial_apps_proof : daves_initial_apps = 16 := by sorry

end NUMINAMATH_CALUDE_daves_initial_apps_daves_initial_apps_proof_l3799_379945


namespace NUMINAMATH_CALUDE_min_p_plus_q_l3799_379955

def is_repeating_decimal (p q : ℕ+) : Prop :=
  (p : ℚ) / q = 0.198

theorem min_p_plus_q (p q : ℕ+) (h : is_repeating_decimal p q) 
  (h_min : ∀ (p' q' : ℕ+), is_repeating_decimal p' q' → q ≤ q') : 
  p + q = 121 := by
  sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l3799_379955


namespace NUMINAMATH_CALUDE_students_not_in_biology_l3799_379922

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880) 
  (h2 : biology_percentage = 325 / 1000) : 
  total_students - (total_students * biology_percentage).floor = 594 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l3799_379922


namespace NUMINAMATH_CALUDE_no_integer_solution_for_equation_l3799_379953

theorem no_integer_solution_for_equation : ∀ x y : ℤ, x^2 - y^2 ≠ 210 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_equation_l3799_379953


namespace NUMINAMATH_CALUDE_composition_ratio_l3799_379997

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composition_ratio :
  (f (g (f 1))) / (g (f (g 1))) = -23 / 5 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l3799_379997


namespace NUMINAMATH_CALUDE_cafeteria_apples_l3799_379978

theorem cafeteria_apples (initial_apples : ℕ) (bought_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 17)
  (h2 : bought_apples = 23)
  (h3 : final_apples = 38) :
  initial_apples - (initial_apples - (final_apples - bought_apples)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l3799_379978


namespace NUMINAMATH_CALUDE_least_skilled_painter_is_granddaughter_l3799_379910

-- Define the family members
inductive FamilyMember
  | Grandmother
  | Niece
  | Nephew
  | Granddaughter

-- Define the skill levels
inductive SkillLevel
  | Best
  | Least

-- Define the gender
inductive Gender
  | Male
  | Female

-- Function to get the gender of a family member
def gender (m : FamilyMember) : Gender :=
  match m with
  | FamilyMember.Grandmother => Gender.Female
  | FamilyMember.Niece => Gender.Female
  | FamilyMember.Nephew => Gender.Male
  | FamilyMember.Granddaughter => Gender.Female

-- Function to determine if two family members can be twins
def canBeTwins (m1 m2 : FamilyMember) : Prop :=
  (m1 = FamilyMember.Niece ∧ m2 = FamilyMember.Nephew) ∨
  (m1 = FamilyMember.Nephew ∧ m2 = FamilyMember.Niece) ∨
  (m1 = FamilyMember.Granddaughter ∧ m2 = FamilyMember.Nephew) ∨
  (m1 = FamilyMember.Nephew ∧ m2 = FamilyMember.Granddaughter)

-- Function to determine if two family members can be the same age
def canBeSameAge (m1 m2 : FamilyMember) : Prop :=
  (m1 = FamilyMember.Niece ∧ m2 = FamilyMember.Nephew) ∨
  (m1 = FamilyMember.Nephew ∧ m2 = FamilyMember.Niece) ∨
  (m1 = FamilyMember.Niece ∧ m2 = FamilyMember.Granddaughter) ∨
  (m1 = FamilyMember.Granddaughter ∧ m2 = FamilyMember.Niece)

-- Theorem statement
theorem least_skilled_painter_is_granddaughter :
  ∀ (best least : FamilyMember),
    (gender best ≠ gender least) →
    (∃ twin, canBeTwins twin least ∧ twin ≠ least) →
    canBeSameAge best least →
    least = FamilyMember.Granddaughter :=
by
  sorry

end NUMINAMATH_CALUDE_least_skilled_painter_is_granddaughter_l3799_379910


namespace NUMINAMATH_CALUDE_inequality_proof_l3799_379930

theorem inequality_proof (a b : ℝ) (h : a > b) : 3 - 2*a < 3 - 2*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3799_379930


namespace NUMINAMATH_CALUDE_fraction_sum_denominator_l3799_379984

theorem fraction_sum_denominator (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 1) :
  let f1 := 3 * a / (5 * b)
  let f2 := 2 * a / (9 * b)
  let f3 := 4 * a / (15 * b)
  (f1 + f2 + f3 : ℚ) = 28 / 45 →
  5 * b + 9 * b + 15 * b = 203 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_denominator_l3799_379984


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l3799_379905

-- Define the given constants
def total_distance : ℝ := 36
def brad_speed : ℝ := 4
def maxwell_distance : ℝ := 12

-- Define Maxwell's speed as a variable
def maxwell_speed : ℝ := sorry

-- Theorem statement
theorem maxwell_walking_speed :
  maxwell_speed = 8 :=
by
  -- The proof would go here, but we're using sorry to skip it
  sorry

end NUMINAMATH_CALUDE_maxwell_walking_speed_l3799_379905


namespace NUMINAMATH_CALUDE_geometric_sequence_grouping_l3799_379949

/-- Given a geometric sequence with common ratio q ≠ 1, prove that the sequence
    formed by grouping every three terms is also geometric with ratio q^3 -/
theorem geometric_sequence_grouping (q : ℝ) (hq : q ≠ 1) :
  ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = q * a n) →
  ∃ (b : ℕ → ℝ), (∀ n, b n = a (3*n - 2) + a (3*n - 1) + a (3*n)) ∧
                 (∀ n, b (n + 1) = q^3 * b n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_grouping_l3799_379949


namespace NUMINAMATH_CALUDE_weighted_average_two_groups_l3799_379948

/-- Weighted average calculation for two groups of students -/
theorem weighted_average_two_groups 
  (x y : ℝ) -- x and y are real numbers representing average scores
  (total_students : ℕ := 25) -- total number of students
  (group_a_students : ℕ := 15) -- number of students in Group A
  (group_b_students : ℕ := 10) -- number of students in Group B
  (h1 : total_students = group_a_students + group_b_students) -- condition: total students is sum of both groups
  : (group_a_students * x + group_b_students * y) / total_students = (3 * x + 2 * y) / 5 :=
by
  sorry

#check weighted_average_two_groups

end NUMINAMATH_CALUDE_weighted_average_two_groups_l3799_379948


namespace NUMINAMATH_CALUDE_function_value_2009_l3799_379923

theorem function_value_2009 (f : ℝ → ℝ) 
  (h1 : f 3 = -Real.sqrt 3) 
  (h2 : ∀ x : ℝ, f (x + 2) * (1 - f x) = 1 + f x) : 
  f 2009 = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_function_value_2009_l3799_379923


namespace NUMINAMATH_CALUDE_correct_number_of_sons_l3799_379933

/-- Represents the problem of dividing land among sons --/
structure LandDivision where
  total_land : ℝ  -- Total land in hectares
  hectare_to_sqm : ℝ  -- Conversion factor from hectare to square meters
  profit_area : ℝ  -- Area in square meters that yields a certain profit
  profit_per_quarter : ℝ  -- Profit in dollars per quarter for profit_area
  son_yearly_profit : ℝ  -- Yearly profit for each son in dollars

/-- Calculate the number of sons based on land division --/
def calculate_sons (ld : LandDivision) : ℕ :=
  sorry

/-- Theorem stating the correct number of sons --/
theorem correct_number_of_sons (ld : LandDivision) 
  (h1 : ld.total_land = 3)
  (h2 : ld.hectare_to_sqm = 10000)
  (h3 : ld.profit_area = 750)
  (h4 : ld.profit_per_quarter = 500)
  (h5 : ld.son_yearly_profit = 10000) :
  calculate_sons ld = 8 := by
    sorry

end NUMINAMATH_CALUDE_correct_number_of_sons_l3799_379933


namespace NUMINAMATH_CALUDE_max_isosceles_triangles_correct_l3799_379915

/-- Represents a set of points on a line and a point not on the line -/
structure PointConfiguration where
  n : ℕ  -- number of points on the line
  h : n = 100

/-- The maximum number of isosceles triangles that can be formed -/
def max_isosceles_triangles (config : PointConfiguration) : ℕ := 150

/-- Theorem stating the maximum number of isosceles triangles -/
theorem max_isosceles_triangles_correct (config : PointConfiguration) :
  max_isosceles_triangles config = 150 := by
  sorry

end NUMINAMATH_CALUDE_max_isosceles_triangles_correct_l3799_379915


namespace NUMINAMATH_CALUDE_cos_2000_in_terms_of_tan_20_l3799_379980

theorem cos_2000_in_terms_of_tan_20 (a : ℝ) (h : Real.tan (20 * π / 180) = a) :
  Real.cos (2000 * π / 180) = -1 / Real.sqrt (1 + a^2) := by
  sorry

end NUMINAMATH_CALUDE_cos_2000_in_terms_of_tan_20_l3799_379980


namespace NUMINAMATH_CALUDE_smallest_divisible_by_2022_l3799_379979

theorem smallest_divisible_by_2022 : 
  ∀ n : ℕ, n > 1 ∧ n < 79 → ¬(2022 ∣ (n^7 - 1)) ∧ (2022 ∣ (79^7 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_2022_l3799_379979


namespace NUMINAMATH_CALUDE_function_passes_through_first_and_fourth_quadrants_l3799_379989

-- Define the conditions
def condition (a b c k : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (b + c - a) / a = k ∧
  (a + c - b) / b = k ∧
  (a + b - c) / c = k

-- Define the function
def f (k : ℝ) (x : ℝ) : ℝ := k * x - k

-- Define what it means for a function to pass through a quadrant
def passes_through_first_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y > 0 ∧ f x = y

def passes_through_fourth_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y < 0 ∧ f x = y

-- The theorem to be proved
theorem function_passes_through_first_and_fourth_quadrants
  (a b c k : ℝ) (h : condition a b c k) :
  passes_through_first_quadrant (f k) ∧
  passes_through_fourth_quadrant (f k) := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_first_and_fourth_quadrants_l3799_379989


namespace NUMINAMATH_CALUDE_president_and_committee_from_eight_l3799_379913

/-- The number of ways to choose a president and a 2-person committee from a group of people. -/
def choose_president_and_committee (n : ℕ) : ℕ :=
  n * (n - 1).choose 2

/-- The theorem stating that choosing a president and a 2-person committee from 8 people results in 168 ways. -/
theorem president_and_committee_from_eight :
  choose_president_and_committee 8 = 168 := by
  sorry

#eval choose_president_and_committee 8

end NUMINAMATH_CALUDE_president_and_committee_from_eight_l3799_379913


namespace NUMINAMATH_CALUDE_medical_supply_transport_l3799_379942

/-- Given two locations A and B that are 360 kilometers apart, a truck carrying 6 boxes of medical supplies
    traveling from A to B at 40 km/h, and a motorcycle departing from B towards the truck at 80 km/h,
    this theorem proves that the total time needed to transport all 6 boxes to location B is 26/3 hours
    and the total distance traveled by the motorcycle is 2080/3 kilometers. -/
theorem medical_supply_transport (distance_AB : ℝ) (truck_speed : ℝ) (motorcycle_speed : ℝ) 
  (boxes : ℕ) (boxes_per_trip : ℕ) :
  distance_AB = 360 →
  truck_speed = 40 →
  motorcycle_speed = 80 →
  boxes = 6 →
  boxes_per_trip = 2 →
  ∃ (total_time : ℝ) (total_distance : ℝ),
    total_time = 26/3 ∧
    total_distance = 2080/3 :=
by sorry


end NUMINAMATH_CALUDE_medical_supply_transport_l3799_379942


namespace NUMINAMATH_CALUDE_ninth_root_of_unity_sum_l3799_379941

theorem ninth_root_of_unity_sum (ω : ℂ) (h1 : ω ^ 9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^27 + ω^36 + ω^45 + ω^54 + ω^63 + ω^72 + ω^81 + ω^90 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ninth_root_of_unity_sum_l3799_379941


namespace NUMINAMATH_CALUDE_unicorn_count_correct_l3799_379959

/-- The number of unicorns in the Enchanted Forest --/
def num_unicorns : ℕ := 6

/-- The number of flowers that bloom with each unicorn step --/
def flowers_per_step : ℕ := 4

/-- The length of the journey in kilometers --/
def journey_length : ℕ := 9

/-- The length of each unicorn step in meters --/
def step_length : ℕ := 3

/-- The total number of flowers that bloom during the journey --/
def total_flowers : ℕ := 72000

/-- Theorem stating that the number of unicorns is correct given the conditions --/
theorem unicorn_count_correct : 
  num_unicorns * flowers_per_step * (journey_length * 1000 / step_length) = total_flowers :=
by sorry

end NUMINAMATH_CALUDE_unicorn_count_correct_l3799_379959


namespace NUMINAMATH_CALUDE_dinner_pizzas_count_l3799_379981

-- Define the variables
def lunch_pizzas : ℕ := 9
def total_pizzas : ℕ := 15

-- Define the theorem
theorem dinner_pizzas_count : total_pizzas - lunch_pizzas = 6 := by
  sorry

end NUMINAMATH_CALUDE_dinner_pizzas_count_l3799_379981


namespace NUMINAMATH_CALUDE_solution_set_implies_b_value_l3799_379903

theorem solution_set_implies_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - b*x + 6 < 0 ↔ 2 < x ∧ x < 3) → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_b_value_l3799_379903


namespace NUMINAMATH_CALUDE_class_size_is_25_l3799_379999

/-- Represents the number of students in a class with preferences for French fries and burgers. -/
structure ClassPreferences where
  frenchFries : ℕ  -- Number of students who like French fries
  burgers : ℕ      -- Number of students who like burgers
  both : ℕ         -- Number of students who like both
  neither : ℕ      -- Number of students who like neither

/-- Calculates the total number of students in the class. -/
def totalStudents (prefs : ClassPreferences) : ℕ :=
  prefs.frenchFries + prefs.burgers + prefs.neither - prefs.both

/-- Theorem stating that given the specific preferences, the total number of students is 25. -/
theorem class_size_is_25 (prefs : ClassPreferences)
  (h1 : prefs.frenchFries = 15)
  (h2 : prefs.burgers = 10)
  (h3 : prefs.both = 6)
  (h4 : prefs.neither = 6) :
  totalStudents prefs = 25 := by
  sorry

#eval totalStudents { frenchFries := 15, burgers := 10, both := 6, neither := 6 }

end NUMINAMATH_CALUDE_class_size_is_25_l3799_379999


namespace NUMINAMATH_CALUDE_percentage_less_than_twice_yesterday_l3799_379983

def students_yesterday : ℕ := 70
def students_absent_today : ℕ := 30
def students_registered : ℕ := 156

def students_today : ℕ := students_registered - students_absent_today
def twice_students_yesterday : ℕ := 2 * students_yesterday
def difference : ℕ := twice_students_yesterday - students_today

theorem percentage_less_than_twice_yesterday (h : difference = 14) :
  (difference : ℚ) / (twice_students_yesterday : ℚ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_than_twice_yesterday_l3799_379983


namespace NUMINAMATH_CALUDE_sum_must_be_odd_l3799_379918

theorem sum_must_be_odd (x y : ℤ) (h : 7 * x + 5 * y = 11111) : 
  ¬(Even (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_sum_must_be_odd_l3799_379918


namespace NUMINAMATH_CALUDE_first_sample_is_three_l3799_379917

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalPopulation : Nat
  sampleSize : Nat
  lastSample : Nat

/-- Calculates the first sample in a systematic sampling scenario -/
def firstSample (s : SystematicSampling) : Nat :=
  s.lastSample - (s.sampleSize - 1) * (s.totalPopulation / s.sampleSize)

/-- Theorem: In the given systematic sampling scenario, the first sample is 3 -/
theorem first_sample_is_three :
  let s : SystematicSampling := ⟨300, 60, 298⟩
  firstSample s = 3 := by sorry

end NUMINAMATH_CALUDE_first_sample_is_three_l3799_379917


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3799_379974

theorem arithmetic_calculation : 4 * 6 * 9 - 18 / 3 + 2^3 = 218 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3799_379974


namespace NUMINAMATH_CALUDE_pentagonal_prism_edges_l3799_379950

/-- A pentagonal prism is a three-dimensional shape with two pentagonal bases connected by lateral edges. -/
structure PentagonalPrism where
  base_edges : ℕ  -- Number of edges in one pentagonal base
  lateral_edges : ℕ  -- Number of lateral edges connecting the two bases

/-- Theorem: A pentagonal prism has 15 edges. -/
theorem pentagonal_prism_edges (p : PentagonalPrism) : 
  p.base_edges = 5 → p.lateral_edges = 5 → p.base_edges * 2 + p.lateral_edges = 15 := by
  sorry

#check pentagonal_prism_edges

end NUMINAMATH_CALUDE_pentagonal_prism_edges_l3799_379950


namespace NUMINAMATH_CALUDE_flow_rate_increase_l3799_379916

/-- Proves that the percentage increase in flow rate from the first to the second hour is 50% -/
theorem flow_rate_increase (r1 r2 r3 : ℝ) : 
  r2 = 36 →  -- Second hour flow rate
  r3 = 1.25 * r2 →  -- Third hour flow rate is 25% more than second
  r1 + r2 + r3 = 105 →  -- Total flow for all three hours
  r1 < r2 →  -- Second hour rate faster than first
  (r2 - r1) / r1 * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_flow_rate_increase_l3799_379916


namespace NUMINAMATH_CALUDE_jasper_candy_count_jasper_candy_proof_l3799_379971

theorem jasper_candy_count : ℕ → Prop :=
  fun initial_candies =>
    let day1_remaining := initial_candies - (initial_candies / 4) - 3
    let day2_remaining := day1_remaining - (day1_remaining / 5) - 5
    let day3_remaining := day2_remaining - (day2_remaining / 6) - 2
    day3_remaining = 10 → initial_candies = 537

theorem jasper_candy_proof : jasper_candy_count 537 := by
  sorry

end NUMINAMATH_CALUDE_jasper_candy_count_jasper_candy_proof_l3799_379971


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_implies_a_greater_half_l3799_379921

/-- The intersection point of two lines is in the fourth quadrant if and only if
    its x-coordinate is positive and its y-coordinate is negative. -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Given two lines y = -x + 1 and y = x - 2a, their intersection point
    is in the fourth quadrant implies a > 1/2 -/
theorem intersection_in_fourth_quadrant_implies_a_greater_half (a : ℝ) :
  (∃ x y : ℝ, y = -x + 1 ∧ y = x - 2 * a ∧ in_fourth_quadrant x y) →
  a > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_implies_a_greater_half_l3799_379921


namespace NUMINAMATH_CALUDE_fraction_simplification_l3799_379932

theorem fraction_simplification : (-150 + 50) / (-50) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3799_379932


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3799_379940

theorem min_value_of_expression (x y : ℝ) :
  (x + y + x * y)^2 + (x - y - x * y)^2 ≥ 0 ∧
  ∃ a b : ℝ, (a + b + a * b)^2 + (a - b - a * b)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3799_379940


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3799_379987

/-- An arithmetic sequence with a positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- The first and seventh terms are roots of x^2 - 10x + 16 = 0 -/
def RootsProperty (a : ℕ → ℝ) : Prop :=
  a 1 ^ 2 - 10 * a 1 + 16 = 0 ∧ a 7 ^ 2 - 10 * a 7 + 16 = 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : RootsProperty a) : 
  a 2 + a 4 + a 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3799_379987


namespace NUMINAMATH_CALUDE_aero_tees_count_l3799_379946

/-- The number of people golfing -/
def num_people : ℕ := 4

/-- The number of tees in a package of generic tees -/
def generic_package_size : ℕ := 12

/-- The maximum number of generic packages Bill will buy -/
def max_generic_packages : ℕ := 2

/-- The minimum number of tees needed per person -/
def min_tees_per_person : ℕ := 20

/-- The number of aero flight tee packages Bill must purchase -/
def aero_packages : ℕ := 28

/-- The number of aero flight tees in one package -/
def aero_tees_per_package : ℕ := 2

theorem aero_tees_count : 
  num_people * min_tees_per_person ≤ 
  max_generic_packages * generic_package_size + 
  aero_packages * aero_tees_per_package ∧
  num_people * min_tees_per_person > 
  max_generic_packages * generic_package_size + 
  aero_packages * (aero_tees_per_package - 1) :=
by sorry

end NUMINAMATH_CALUDE_aero_tees_count_l3799_379946


namespace NUMINAMATH_CALUDE_fraction_sum_l3799_379944

theorem fraction_sum (x y : ℚ) (h : x / y = 4 / 7) : (x + y) / y = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3799_379944


namespace NUMINAMATH_CALUDE_karen_cookies_l3799_379956

/-- Given Karen's cookie distribution, prove she kept 10 for herself -/
theorem karen_cookies (total : ℕ) (grandparents : ℕ) (class_size : ℕ) (per_person : ℕ)
  (h1 : total = 50)
  (h2 : grandparents = 8)
  (h3 : class_size = 16)
  (h4 : per_person = 2) :
  total - (grandparents + class_size * per_person) = 10 := by
  sorry

#eval 50 - (8 + 16 * 2)  -- Expected output: 10

end NUMINAMATH_CALUDE_karen_cookies_l3799_379956


namespace NUMINAMATH_CALUDE_positive_implies_increasing_exists_increasing_not_always_positive_l3799_379982

-- Define a differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Part 1: Sufficiency
theorem positive_implies_increasing :
  (∀ x, f x > 0) → MonotoneOn f Set.univ := by sorry

-- Part 2: Not Necessary
theorem exists_increasing_not_always_positive :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ MonotoneOn f Set.univ ∧ ∃ x, f x ≤ 0 := by sorry

end NUMINAMATH_CALUDE_positive_implies_increasing_exists_increasing_not_always_positive_l3799_379982


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3799_379975

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3799_379975


namespace NUMINAMATH_CALUDE_exists_factorial_with_124_zeros_l3799_379965

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- There exists a positive integer n such that n! has exactly 124 trailing zeros -/
theorem exists_factorial_with_124_zeros : ∃ n : ℕ, n > 0 ∧ trailingZeros n = 124 := by
  sorry

end NUMINAMATH_CALUDE_exists_factorial_with_124_zeros_l3799_379965


namespace NUMINAMATH_CALUDE_boy_girl_sum_equal_l3799_379957

/-- Represents a child in the line -/
inductive Child
  | Boy : Child
  | Girl : Child

/-- The line of children -/
def Line (n : ℕ) := Vector Child (2 * n)

/-- Count children to the right of a position -/
def countRight (line : Line n) (pos : Fin (2 * n)) : ℕ := sorry

/-- Count children to the left of a position -/
def countLeft (line : Line n) (pos : Fin (2 * n)) : ℕ := sorry

/-- Sum of counts for boys -/
def boySum (line : Line n) : ℕ := sorry

/-- Sum of counts for girls -/
def girlSum (line : Line n) : ℕ := sorry

/-- The main theorem: boySum equals girlSum for any valid line -/
theorem boy_girl_sum_equal (n : ℕ) (line : Line n) 
  (h : ∀ i : Fin (2 * n), (i.val < n → line.get i = Child.Boy) ∧ (i.val ≥ n → line.get i = Child.Girl)) :
  boySum line = girlSum line := by sorry

end NUMINAMATH_CALUDE_boy_girl_sum_equal_l3799_379957


namespace NUMINAMATH_CALUDE_jar_weight_percentage_l3799_379911

theorem jar_weight_percentage (jar_weight : ℝ) (full_weight : ℝ) 
  (h1 : jar_weight = 0.4 * full_weight)
  (h2 : jar_weight > 0)
  (h3 : full_weight > jar_weight) :
  let beans_weight := full_weight - jar_weight
  let remaining_beans_weight := (1/3) * beans_weight
  let new_total_weight := jar_weight + remaining_beans_weight
  new_total_weight / full_weight = 0.6 := by
sorry

end NUMINAMATH_CALUDE_jar_weight_percentage_l3799_379911


namespace NUMINAMATH_CALUDE_tan_alpha_three_implies_sin_2alpha_over_cos_squared_alpha_eq_six_l3799_379934

theorem tan_alpha_three_implies_sin_2alpha_over_cos_squared_alpha_eq_six
  (α : Real)
  (h : Real.tan α = 3) :
  (Real.sin (2 * α)) / (Real.cos α)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_three_implies_sin_2alpha_over_cos_squared_alpha_eq_six_l3799_379934


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l3799_379960

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The main theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → ∀ x : ℝ, f x = x :=
by sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l3799_379960


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_reciprocal_min_value_achieved_l3799_379928

theorem min_value_of_x_plus_reciprocal (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) = 3 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_reciprocal_min_value_achieved_l3799_379928


namespace NUMINAMATH_CALUDE_optimal_feed_consumption_l3799_379931

/-- Represents the nutritional content and cost of animal feeds -/
structure Feed where
  nutrientA : ℝ
  nutrientB : ℝ
  cost : ℝ

/-- Represents the daily nutritional requirements for an animal -/
structure Requirements where
  minNutrientA : ℝ
  minNutrientB : ℝ

/-- Represents the daily consumption of feeds -/
structure Consumption where
  feedI : ℝ
  feedII : ℝ

/-- Calculates the total cost of a given consumption -/
def totalCost (c : Consumption) : ℝ := c.feedI + c.feedII

/-- Checks if a given consumption meets the nutritional requirements -/
def meetsRequirements (f1 f2 : Feed) (r : Requirements) (c : Consumption) : Prop :=
  c.feedI * f1.nutrientA + c.feedII * f2.nutrientA ≥ r.minNutrientA ∧
  c.feedI * f1.nutrientB + c.feedII * f2.nutrientB ≥ r.minNutrientB

/-- Theorem stating the optimal solution for the animal feed problem -/
theorem optimal_feed_consumption 
  (feedI feedII : Feed)
  (req : Requirements)
  (h1 : feedI.nutrientA = 5 ∧ feedI.nutrientB = 2.5 ∧ feedI.cost = 1)
  (h2 : feedII.nutrientA = 3 ∧ feedII.nutrientB = 3 ∧ feedII.cost = 1)
  (h3 : req.minNutrientA = 30 ∧ req.minNutrientB = 22.5) :
  ∃ (c : Consumption), 
    meetsRequirements feedI feedII req c ∧ 
    totalCost c = 8 ∧
    ∀ (c' : Consumption), meetsRequirements feedI feedII req c' → totalCost c' ≥ totalCost c :=
by sorry

end NUMINAMATH_CALUDE_optimal_feed_consumption_l3799_379931


namespace NUMINAMATH_CALUDE_new_teacher_age_proof_l3799_379966

/-- The number of teachers initially -/
def initial_teachers : ℕ := 20

/-- The average age of initial teachers -/
def initial_average_age : ℕ := 49

/-- The number of teachers after a new teacher joins -/
def final_teachers : ℕ := 21

/-- The new average age after a new teacher joins -/
def final_average_age : ℕ := 48

/-- The age of the new teacher -/
def new_teacher_age : ℕ := 28

theorem new_teacher_age_proof :
  initial_teachers * initial_average_age + new_teacher_age = final_teachers * final_average_age :=
sorry

end NUMINAMATH_CALUDE_new_teacher_age_proof_l3799_379966


namespace NUMINAMATH_CALUDE_pear_sales_problem_l3799_379919

theorem pear_sales_problem (morning_sales afternoon_sales total_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  afternoon_sales = 320 →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 480 :=
by sorry

end NUMINAMATH_CALUDE_pear_sales_problem_l3799_379919


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l3799_379947

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * (x + 1)^2 - 2 = 25
def equation2 (x : ℝ) : Prop := (x - 1)^3 = 64

-- Theorem for equation1
theorem equation1_solutions :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = 2 ∧ x₂ = -4 :=
sorry

-- Theorem for equation2
theorem equation2_solution :
  ∃ x : ℝ, equation2 x ∧ x = 5 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l3799_379947


namespace NUMINAMATH_CALUDE_cameron_fruit_arrangements_l3799_379996

/-- The number of ways to arrange n objects, where there are k groups of indistinguishable objects with sizes a₁, a₂, ..., aₖ -/
def multinomial (n : ℕ) (a : List ℕ) : ℕ :=
  Nat.factorial n / (a.map Nat.factorial).prod

/-- The number of ways Cameron can eat his fruit -/
def cameronFruitArrangements : ℕ :=
  multinomial 9 [4, 3, 2]

theorem cameron_fruit_arrangements :
  cameronFruitArrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_cameron_fruit_arrangements_l3799_379996


namespace NUMINAMATH_CALUDE_coefficient_x3y5_times_two_l3799_379993

theorem coefficient_x3y5_times_two (x y : ℝ) : 2 * (Finset.range 9).sum (λ k => if k = 5 then Nat.choose 8 k else 0) = 112 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_times_two_l3799_379993


namespace NUMINAMATH_CALUDE_picnic_group_size_l3799_379962

theorem picnic_group_size (initial_group : ℕ) (new_avg_age : ℝ) (final_avg_age : ℝ) :
  initial_group = 15 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  ∃ (new_members : ℕ), 
    new_members = 1 ∧
    (initial_group : ℝ) * final_avg_age = 
      (initial_group : ℝ) * new_avg_age + (new_members : ℝ) * (final_avg_age - new_avg_age) :=
by sorry

end NUMINAMATH_CALUDE_picnic_group_size_l3799_379962


namespace NUMINAMATH_CALUDE_range_of_m_given_inequality_and_point_l3799_379970

/-- Given a planar region defined by an inequality and a point within that region,
    this theorem states the range of the parameter m. -/
theorem range_of_m_given_inequality_and_point (m : ℝ) : 
  (∀ x y : ℝ, x - (m^2 - 2*m + 4)*y + 6 > 0 → 
    (1 : ℝ) - (m^2 - 2*m + 4)*(1 : ℝ) + 6 > 0) → 
  m ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_given_inequality_and_point_l3799_379970


namespace NUMINAMATH_CALUDE_speed_ratio_l3799_379909

-- Define the speeds of A and B
def v_A : ℝ := sorry
def v_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := -800

-- Define the equidistant condition at 3 minutes
def equidistant_3min : Prop :=
  3 * v_A = abs (initial_B_position + 3 * v_B)

-- Define the equidistant condition at 9 minutes
def equidistant_9min : Prop :=
  9 * v_A = abs (initial_B_position + 9 * v_B)

-- Theorem statement
theorem speed_ratio :
  equidistant_3min →
  equidistant_9min →
  v_A / v_B = 9 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_l3799_379909


namespace NUMINAMATH_CALUDE_checker_moves_fibonacci_checker_moves_10_checker_moves_11_l3799_379972

def checkerMoves : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => checkerMoves (n + 1) + checkerMoves n

theorem checker_moves_fibonacci (n : ℕ) :
  checkerMoves n = checkerMoves (n - 1) + checkerMoves (n - 2) :=
by sorry

theorem checker_moves_10 : checkerMoves 10 = 89 :=
by sorry

theorem checker_moves_11 : checkerMoves 11 = 144 :=
by sorry

end NUMINAMATH_CALUDE_checker_moves_fibonacci_checker_moves_10_checker_moves_11_l3799_379972


namespace NUMINAMATH_CALUDE_system_solution_l3799_379988

theorem system_solution :
  let eq1 (x y : ℚ) := x * y^2 - 2 * y^2 + 3 * x = 18
  let eq2 (x y : ℚ) := 3 * x * y + 5 * x - 6 * y = 24
  (eq1 3 3 ∧ eq2 3 3) ∧
  (eq1 (75/13) (-3/7) ∧ eq2 (75/13) (-3/7)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3799_379988


namespace NUMINAMATH_CALUDE_ice_cream_melt_height_l3799_379914

/-- The height of a cylinder with radius 9 inches, having the same volume as a sphere with radius 3 inches, is 4/9 inches. -/
theorem ice_cream_melt_height : 
  let sphere_radius : ℝ := 3
  let cylinder_radius : ℝ := 9
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius ^ 3
  let cylinder_volume (h : ℝ) := Real.pi * cylinder_radius ^ 2 * h
  ∃ h : ℝ, cylinder_volume h = sphere_volume ∧ h = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_melt_height_l3799_379914


namespace NUMINAMATH_CALUDE_unique_grouping_l3799_379925

def numbers : List ℕ := [12, 30, 42, 44, 57, 91, 95, 143]

def is_valid_grouping (group1 group2 : List ℕ) : Prop :=
  group1.prod = group2.prod ∧
  (group1 ++ group2).toFinset = numbers.toFinset ∧
  group1.toFinset ∩ group2.toFinset = ∅

theorem unique_grouping :
  ∀ (group1 group2 : List ℕ),
    is_valid_grouping group1 group2 →
    ((group1.toFinset = {12, 42, 95, 143} ∧ group2.toFinset = {30, 44, 57, 91}) ∨
     (group2.toFinset = {12, 42, 95, 143} ∧ group1.toFinset = {30, 44, 57, 91})) :=
by sorry

end NUMINAMATH_CALUDE_unique_grouping_l3799_379925


namespace NUMINAMATH_CALUDE_complex_equation_implies_exponent_one_l3799_379927

theorem complex_equation_implies_exponent_one (x y : ℝ) 
  (h : (x + y) * Complex.I = x - 1) : 
  (2 : ℝ) ^ (x + y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_exponent_one_l3799_379927


namespace NUMINAMATH_CALUDE_can_distribution_l3799_379958

theorem can_distribution (total_cans : Nat) (volume_difference : Real) (total_volume : Real) :
  total_cans = 140 →
  volume_difference = 2.5 →
  total_volume = 60 →
  ∃ (large_cans small_cans : Nat) (small_volume : Real),
    large_cans + small_cans = total_cans ∧
    large_cans * (small_volume + volume_difference) = total_volume ∧
    small_cans * small_volume = total_volume ∧
    large_cans = 20 ∧
    small_cans = 120 := by
  sorry

#check can_distribution

end NUMINAMATH_CALUDE_can_distribution_l3799_379958


namespace NUMINAMATH_CALUDE_children_at_track_meet_l3799_379907

theorem children_at_track_meet (total_seats : ℕ) (empty_seats : ℕ) (adults : ℕ) 
  (h1 : total_seats = 95)
  (h2 : empty_seats = 14)
  (h3 : adults = 29) :
  total_seats - empty_seats - adults = 52 := by
  sorry

end NUMINAMATH_CALUDE_children_at_track_meet_l3799_379907


namespace NUMINAMATH_CALUDE_additive_inverses_imply_x_equals_one_l3799_379994

theorem additive_inverses_imply_x_equals_one :
  ∀ x : ℝ, (4 * x - 1) + (3 * x - 6) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverses_imply_x_equals_one_l3799_379994


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3799_379976

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ (x < 1 ∨ x > b)

-- Define the constraint for x and y
def constraint (a b x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ a / x + b / y = 1

-- Main theorem
theorem quadratic_inequality_theorem :
  ∃ a b : ℝ,
    -- Part I: Values of a and b
    solution_set a b ∧ a = 1 ∧ b = 2 ∧
    -- Part II: Minimum value of 2x + y
    (∀ x y, constraint a b x y → 2 * x + y ≥ 8) ∧
    -- Part II: Range of k
    (∀ k, (∀ x y, constraint a b x y → 2 * x + y ≥ k^2 + k + 2) ↔ -3 ≤ k ∧ k ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3799_379976


namespace NUMINAMATH_CALUDE_five_digit_sum_l3799_379943

def is_valid_digit (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 9

def sum_of_digits (x : ℕ) : ℕ := 120 * (1 + 3 + 4 + 6 + x)

theorem five_digit_sum (x : ℕ) (h1 : is_valid_digit x) (h2 : sum_of_digits x = 2640) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_sum_l3799_379943


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l3799_379900

theorem least_number_with_remainders : ∃ n : ℕ, 
  n > 0 ∧
  n % 5 = 3 ∧
  n % 6 = 3 ∧
  n % 7 = 3 ∧
  n % 8 = 3 ∧
  n % 9 = 0 ∧
  (∀ m : ℕ, m > 0 ∧ m % 5 = 3 ∧ m % 6 = 3 ∧ m % 7 = 3 ∧ m % 8 = 3 ∧ m % 9 = 0 → n ≤ m) ∧
  n = 1683 :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l3799_379900


namespace NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l3799_379969

/-- Calculates the cost per pouch in cents -/
def cost_per_pouch (num_boxes : ℕ) (pouches_per_box : ℕ) (total_cost_dollars : ℕ) : ℕ :=
  (total_cost_dollars * 100) / (num_boxes * pouches_per_box)

/-- Theorem: The cost per pouch is 20 cents -/
theorem capri_sun_cost_per_pouch :
  cost_per_pouch 10 6 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l3799_379969


namespace NUMINAMATH_CALUDE_hash_problem_l3799_379995

-- Define the operation #
def hash (a b : ℕ) : ℕ := 4*a^2 + 4*b^2 + 8*a*b

-- Theorem statement
theorem hash_problem (a b : ℕ) :
  hash a b = 100 ∧ (a + b) + 6 = 11 → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_hash_problem_l3799_379995


namespace NUMINAMATH_CALUDE_range_of_b_l3799_379977

theorem range_of_b (a b : ℝ) (h1 : a * b^2 > a) (h2 : a > a * b) : b < -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l3799_379977


namespace NUMINAMATH_CALUDE_train_crossing_time_l3799_379963

/-- Given a train and platform with specified lengths and crossing time, 
    calculate the time taken for the train to cross a signal pole. -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 285)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3799_379963


namespace NUMINAMATH_CALUDE_inequality_proof_l3799_379954

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π/2) : 
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3799_379954


namespace NUMINAMATH_CALUDE_equation_solution_l3799_379973

-- Define the equation
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + 3 * x * y + y^2 + x = 1

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = -x - 1
def line2 (x y : ℝ) : Prop := y = -2*x + 1

-- Theorem statement
theorem equation_solution (x y : ℝ) :
  satisfies_equation x y → line1 x y ∨ line2 x y :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3799_379973


namespace NUMINAMATH_CALUDE_total_money_is_36_l3799_379938

/-- Given Joanna's money, calculate the total money of Joanna, her brother, and her sister -/
def total_money (joanna_money : ℕ) : ℕ :=
  joanna_money + 3 * joanna_money + joanna_money / 2

/-- Theorem: The total money of Joanna, her brother, and her sister is $36 when Joanna has $8 -/
theorem total_money_is_36 : total_money 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_36_l3799_379938


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l3799_379906

theorem sum_of_solutions_quadratic (a b c d e : ℝ) :
  (∀ x, a * x^2 + b * x + c = d * x + e) →
  (a ≠ 0) →
  (∃ x y, a * x^2 + b * x + c = d * x + e ∧ 
          a * y^2 + b * y + c = d * y + e ∧ 
          x ≠ y) →
  (x + y = -(b - d) / a) :=
by sorry

theorem sum_of_solutions_specific_quadratic :
  let a : ℝ := 2
  let b : ℝ := -8
  let c : ℝ := 5
  let d : ℝ := 4
  let e : ℝ := -20
  (∀ x, a * x^2 + b * x + c = d * x + e) →
  (∃ x y, a * x^2 + b * x + c = d * x + e ∧ 
          a * y^2 + b * y + c = d * y + e ∧ 
          x ≠ y) →
  (x + y = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l3799_379906


namespace NUMINAMATH_CALUDE_smallest_aab_value_l3799_379986

theorem smallest_aab_value (A B : ℕ) : 
  (1 ≤ A ∧ A ≤ 9) →  -- A is a digit from 1 to 9
  (1 ≤ B ∧ B ≤ 9) →  -- B is a digit from 1 to 9
  A + 1 = B →        -- A and B are consecutive digits
  (10 * A + B : ℕ) = (110 * A + B) / 7 →  -- AB = AAB / 7
  (∀ A' B' : ℕ, 
    (1 ≤ A' ∧ A' ≤ 9) → 
    (1 ≤ B' ∧ B' ≤ 9) → 
    A' + 1 = B' → 
    (10 * A' + B' : ℕ) = (110 * A' + B') / 7 → 
    110 * A + B ≤ 110 * A' + B') →
  110 * A + B = 889 := by
sorry

end NUMINAMATH_CALUDE_smallest_aab_value_l3799_379986


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3799_379920

theorem discount_percentage_proof (wholesale_price retail_price : ℝ) 
  (profit_percentage : ℝ) (h1 : wholesale_price = 81) 
  (h2 : retail_price = 108) (h3 : profit_percentage = 0.2) : 
  (retail_price - (wholesale_price + wholesale_price * profit_percentage)) / retail_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3799_379920


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3799_379937

theorem quadratic_roots_relation (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*x + 12 = 0 ∧ y^2 - k*y + 12 = 0 → y = x + 7) → 
  k = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3799_379937


namespace NUMINAMATH_CALUDE_workbook_problems_l3799_379939

theorem workbook_problems (T : ℕ) : 
  (T : ℚ) / 2 + T / 4 + T / 6 + 20 = T → T = 240 := by
  sorry

end NUMINAMATH_CALUDE_workbook_problems_l3799_379939


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3799_379968

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, (m - 1) * x^2 + 5 * x + m^2 - 1 = 0) → 
  (m^2 - 1 = 0) → 
  m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3799_379968


namespace NUMINAMATH_CALUDE_solution_of_quadratic_equation_l3799_379964

theorem solution_of_quadratic_equation :
  ∀ x : ℝ, 2 * x^2 = 4 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_quadratic_equation_l3799_379964


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3799_379912

theorem triangle_inequality_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt (a / (b + c - a)) + Real.sqrt (b / (c + a - b)) + Real.sqrt (c / (a + b - c)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3799_379912


namespace NUMINAMATH_CALUDE_fractional_equation_to_polynomial_l3799_379952

theorem fractional_equation_to_polynomial (x y : ℝ) (h1 : (2*x - 1)/x^2 + x^2/(2*x - 1) = 5) (h2 : (2*x - 1)/x^2 = y) : y^2 - 5*y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_to_polynomial_l3799_379952
