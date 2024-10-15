import Mathlib

namespace NUMINAMATH_CALUDE_reading_time_calculation_l3175_317533

theorem reading_time_calculation (total_time math_time spelling_time : ℕ) 
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18) :
  total_time - (math_time + spelling_time) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l3175_317533


namespace NUMINAMATH_CALUDE_eagles_volleyball_games_l3175_317515

theorem eagles_volleyball_games :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
  initial_wins = (0.4 : ℝ) * initial_games →
  (initial_wins + 9 : ℝ) / (initial_games + 10) = 0.55 →
  initial_games + 10 = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_eagles_volleyball_games_l3175_317515


namespace NUMINAMATH_CALUDE_gene_mutation_not_valid_for_AaB_l3175_317508

/-- Represents a genotype --/
inductive Genotype
  | AaB
  | AABb

/-- Represents possible reasons for lacking a gene --/
inductive Reason
  | GeneMutation
  | ChromosomalVariation
  | ChromosomalStructuralVariation
  | MaleIndividual

/-- Determines if a reason is valid for explaining the lack of a gene --/
def is_valid_reason (g : Genotype) (r : Reason) : Prop :=
  match g, r with
  | Genotype.AaB, Reason.GeneMutation => False
  | _, _ => True

/-- Theorem stating that gene mutation is not a valid reason for individual A's genotype --/
theorem gene_mutation_not_valid_for_AaB :
  ¬(is_valid_reason Genotype.AaB Reason.GeneMutation) :=
by
  sorry


end NUMINAMATH_CALUDE_gene_mutation_not_valid_for_AaB_l3175_317508


namespace NUMINAMATH_CALUDE_y_value_l3175_317547

theorem y_value : (2023^2 - 1012) / 2023 = 2023 - 1012/2023 := by sorry

end NUMINAMATH_CALUDE_y_value_l3175_317547


namespace NUMINAMATH_CALUDE_equation_solution_set_l3175_317531

theorem equation_solution_set : 
  {x : ℝ | x^6 + x^2 = (2*x + 3)^3 + 2*x + 3} = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3175_317531


namespace NUMINAMATH_CALUDE_subtraction_of_negative_l3175_317566

theorem subtraction_of_negative : 12.345 - (-3.256) = 15.601 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_l3175_317566


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3175_317545

/-- Given two hyperbolas with the same asymptotes, prove that M = 576/25 -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, y^2/16 - x^2/25 = 1 ↔ x^2/36 - y^2/M = 1) → 
  (∀ x y : ℝ, y = (4/5)*x ↔ y = (Real.sqrt M / 6)*x) → 
  M = 576/25 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3175_317545


namespace NUMINAMATH_CALUDE_f_increasing_implies_F_decreasing_l3175_317538

/-- A function f is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Definition of F in terms of f -/
def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (1 - x) - f (1 + x)

/-- A function f is decreasing on ℝ -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem f_increasing_implies_F_decreasing (f : ℝ → ℝ) (h : IsIncreasing f) : IsDecreasing (F f) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_implies_F_decreasing_l3175_317538


namespace NUMINAMATH_CALUDE_modified_sequence_last_term_l3175_317511

def sequence_rule (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | i + 1 => 
    let prev := sequence_rule n i
    if prev < 10 then
      2 * prev
    else
      (prev % 10) + 5

def modified_sequence (n : ℕ) (m : ℕ) : ℕ → ℕ
  | i => if i = 99 then sequence_rule n i + m else sequence_rule n i

theorem modified_sequence_last_term (n : ℕ) :
  ∃ m : ℕ, m < 10 ∧ modified_sequence 2012 m 2011 = 5 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_modified_sequence_last_term_l3175_317511


namespace NUMINAMATH_CALUDE_first_car_departure_time_l3175_317581

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv : minutes < 60

/-- Represents a car with its speed -/
structure Car where
  speed : ℝ  -- speed in miles per hour

def problem (first_car : Car) (second_car : Car) (trip_distance : ℝ) (time_difference : ℝ) (meeting_time : Time) : Prop :=
  first_car.speed = 30 ∧
  second_car.speed = 60 ∧
  trip_distance = 80 ∧
  time_difference = 1/6 ∧  -- 10 minutes in hours
  meeting_time.hours = 10 ∧
  meeting_time.minutes = 30

theorem first_car_departure_time 
  (first_car : Car) (second_car : Car) (trip_distance : ℝ) (time_difference : ℝ) (meeting_time : Time) :
  problem first_car second_car trip_distance time_difference meeting_time →
  ∃ (departure_time : Time), 
    departure_time.hours = 10 ∧ departure_time.minutes = 10 :=
sorry

end NUMINAMATH_CALUDE_first_car_departure_time_l3175_317581


namespace NUMINAMATH_CALUDE_two_circles_exist_l3175_317598

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions for the circle
def satisfiesConditions (c : Circle) : Prop :=
  let (a, b) := c.center
  -- Circle is tangent to the x-axis
  c.radius = |b| ∧
  -- Center is on the line 3x - y = 0
  3 * a = b ∧
  -- Intersects x - y = 0 to form a chord of length 2√7
  2 * c.radius^2 = (a - b)^2 + 14

-- State the theorem
theorem two_circles_exist :
  ∃ (c1 c2 : Circle),
    satisfiesConditions c1 ∧
    satisfiesConditions c2 ∧
    c1.center = (1, 3) ∧
    c2.center = (-1, -3) ∧
    c1.radius = 3 ∧
    c2.radius = 3 :=
  sorry

end NUMINAMATH_CALUDE_two_circles_exist_l3175_317598


namespace NUMINAMATH_CALUDE_smallest_pencil_collection_l3175_317591

theorem smallest_pencil_collection (P : ℕ) : 
  P > 2 ∧ 
  P % 5 = 2 ∧ 
  P % 9 = 2 ∧ 
  P % 11 = 2 ∧ 
  (∀ Q : ℕ, Q > 2 ∧ Q % 5 = 2 ∧ Q % 9 = 2 ∧ Q % 11 = 2 → P ≤ Q) →
  P = 497 := by
sorry

end NUMINAMATH_CALUDE_smallest_pencil_collection_l3175_317591


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l3175_317528

/-- An isosceles triangle with given perimeter and base -/
structure IsoscelesTriangle where
  perimeter : ℝ
  base : ℝ
  legs_equal : ℝ
  perimeter_eq : perimeter = 2 * legs_equal + base

/-- Theorem: In an isosceles triangle with perimeter 26 cm and base 11 cm, each leg is 7.5 cm -/
theorem isosceles_triangle_leg_length 
  (triangle : IsoscelesTriangle) 
  (h_perimeter : triangle.perimeter = 26) 
  (h_base : triangle.base = 11) : 
  triangle.legs_equal = 7.5 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l3175_317528


namespace NUMINAMATH_CALUDE_probability_multiple_of_four_l3175_317537

/-- A set of digits from 1 to 5 -/
def DigitSet : Finset ℕ := {1, 2, 3, 4, 5}

/-- A function to check if a three-digit number is divisible by 4 -/
def isDivisibleByFour (a b c : ℕ) : Prop := (10 * b + c) % 4 = 0

/-- The total number of ways to draw three digits from five -/
def totalWays : ℕ := 5 * 4 * 3

/-- The number of ways to draw three digits that form a number divisible by 4 -/
def validWays : ℕ := 15

theorem probability_multiple_of_four :
  (validWays : ℚ) / totalWays = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_four_l3175_317537


namespace NUMINAMATH_CALUDE_two_digit_number_special_property_l3175_317505

theorem two_digit_number_special_property : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (∃ x y : ℕ, n = 10 * x + y ∧ x < 10 ∧ y < 10 ∧ n = x^3 + y^2) ∧
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_special_property_l3175_317505


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3175_317542

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 4*x + 8) / ((x - 1)*(x - 4)*(x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) ∧
      P = 1/3 ∧ Q = -4/3 ∧ R = 2 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3175_317542


namespace NUMINAMATH_CALUDE_students_not_in_sports_l3175_317588

/-- The number of students in the class -/
def total_students : ℕ := 50

/-- The number of students playing basketball -/
def basketball : ℕ := total_students / 2

/-- The number of students playing volleyball -/
def volleyball : ℕ := total_students / 3

/-- The number of students playing soccer -/
def soccer : ℕ := total_students / 5

/-- The number of students playing badminton -/
def badminton : ℕ := total_students / 8

/-- The number of students playing both basketball and volleyball -/
def basketball_and_volleyball : ℕ := total_students / 10

/-- The number of students playing both basketball and soccer -/
def basketball_and_soccer : ℕ := total_students / 12

/-- The number of students playing both basketball and badminton -/
def basketball_and_badminton : ℕ := total_students / 16

/-- The number of students playing both volleyball and soccer -/
def volleyball_and_soccer : ℕ := total_students / 8

/-- The number of students playing both volleyball and badminton -/
def volleyball_and_badminton : ℕ := total_students / 10

/-- The number of students playing both soccer and badminton -/
def soccer_and_badminton : ℕ := total_students / 20

/-- The number of students playing all four sports -/
def all_four_sports : ℕ := total_students / 25

/-- The theorem stating that 16 students do not engage in any of the four sports -/
theorem students_not_in_sports : 
  total_students - (basketball + volleyball + soccer + badminton 
  - basketball_and_volleyball - basketball_and_soccer - basketball_and_badminton 
  - volleyball_and_soccer - volleyball_and_badminton - soccer_and_badminton 
  + all_four_sports) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_sports_l3175_317588


namespace NUMINAMATH_CALUDE_books_per_week_after_second_l3175_317599

theorem books_per_week_after_second (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (total_weeks : ℕ) :
  total_books = 54 →
  first_week = 6 →
  second_week = 3 →
  total_weeks = 7 →
  (total_books - (first_week + second_week)) / (total_weeks - 2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_books_per_week_after_second_l3175_317599


namespace NUMINAMATH_CALUDE_cone_height_from_sphere_waste_l3175_317582

/-- Given a sphere and a cone carved from it, prove the height of the cone when 75% of wood is wasted -/
theorem cone_height_from_sphere_waste (r : ℝ) (h : ℝ) : 
  r = 9 →  -- sphere radius
  (4/3) * Real.pi * r^3 * (1 - 0.75) = (1/3) * Real.pi * r^2 * h → -- 75% wood wasted
  h = 27 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_sphere_waste_l3175_317582


namespace NUMINAMATH_CALUDE_perimeter_difference_rectangles_l3175_317535

/-- Calculate the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculate the positive difference between two natural numbers -/
def positiveDifference (a b : ℕ) : ℕ :=
  max a b - min a b

theorem perimeter_difference_rectangles :
  positiveDifference (rectanglePerimeter 3 4) (rectanglePerimeter 1 8) = 4 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_rectangles_l3175_317535


namespace NUMINAMATH_CALUDE_rational_fraction_equality_l3175_317504

theorem rational_fraction_equality (a b : ℚ) 
  (h1 : (a + 2*b) / (2*a - b) = 2)
  (h2 : 3*a - 2*b ≠ 0) :
  (3*a + 2*b) / (3*a - 2*b) = 3 := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_equality_l3175_317504


namespace NUMINAMATH_CALUDE_kyro_are_fylol_and_glyk_l3175_317534

-- Define the types
variable (U : Type) -- Universe of discourse
variable (Fylol Glyk Kyro Mylo : Set U)

-- State the given conditions
variable (h1 : Fylol ⊆ Glyk)
variable (h2 : Kyro ⊆ Glyk)
variable (h3 : Mylo ⊆ Fylol)
variable (h4 : Kyro ⊆ Mylo)

-- Theorem to prove
theorem kyro_are_fylol_and_glyk : Kyro ⊆ Fylol ∩ Glyk := by sorry

end NUMINAMATH_CALUDE_kyro_are_fylol_and_glyk_l3175_317534


namespace NUMINAMATH_CALUDE_lines_are_parallel_l3175_317589

/-- Two lines a₁x + b₁y + c₁ = 0 and a₂x + b₂y + c₂ = 0 are parallel if and only if a₁b₂ = a₂b₁ -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * c₂ ≠ a₂ * c₁

/-- The line x - 2y + 1 = 0 -/
def line1 : ℝ → ℝ → ℝ := λ x y => x - 2*y + 1

/-- The line 2x - 4y + 1 = 0 -/
def line2 : ℝ → ℝ → ℝ := λ x y => 2*x - 4*y + 1

theorem lines_are_parallel : parallel 1 (-2) 1 2 (-4) 1 :=
  sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l3175_317589


namespace NUMINAMATH_CALUDE_joan_dimes_l3175_317564

/-- The number of dimes Joan has after spending some -/
def remaining_dimes (initial : ℕ) (spent : ℕ) : ℕ := initial - spent

/-- Theorem: If Joan had 5 dimes initially and spent 2 dimes, she now has 3 dimes -/
theorem joan_dimes : remaining_dimes 5 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_joan_dimes_l3175_317564


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l3175_317575

/-- Given three polynomial functions f, g, and h, prove their sum equals a specific polynomial. -/
theorem sum_of_polynomials (x : ℝ) : 
  let f := fun (x : ℝ) => -4*x^2 + 2*x - 5
  let g := fun (x : ℝ) => -6*x^2 + 4*x - 9
  let h := fun (x : ℝ) => 6*x^2 + 6*x + 2
  f x + g x + h x = -4*x^2 + 12*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l3175_317575


namespace NUMINAMATH_CALUDE_divisibility_pairs_l3175_317551

theorem divisibility_pairs : 
  {p : ℕ × ℕ | (p.1 + 1) % p.2 = 0 ∧ (p.2^2 - p.2 + 1) % p.1 = 0} = 
  {(1, 1), (1, 2), (3, 2)} := by
sorry

end NUMINAMATH_CALUDE_divisibility_pairs_l3175_317551


namespace NUMINAMATH_CALUDE_pascal_ratio_row_34_l3175_317520

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Check if three consecutive entries in a row are in the ratio 2:3:4 -/
def hasRatio234 (n : ℕ) (r : ℕ) : Prop :=
  4 * pascal n r = 3 * pascal n (r+1) ∧
  4 * pascal n (r+1) = 3 * pascal n (r+2)

theorem pascal_ratio_row_34 : ∃ r, hasRatio234 34 r := by
  sorry

#check pascal_ratio_row_34

end NUMINAMATH_CALUDE_pascal_ratio_row_34_l3175_317520


namespace NUMINAMATH_CALUDE_intersection_M_N_l3175_317500

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x / (x - 1) ≤ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3175_317500


namespace NUMINAMATH_CALUDE_blue_marble_difference_l3175_317574

theorem blue_marble_difference (total_green : ℕ) 
  (ratio_a_blue ratio_a_green ratio_b_blue ratio_b_green : ℕ) : 
  total_green = 162 →
  ratio_a_blue = 5 →
  ratio_a_green = 3 →
  ratio_b_blue = 4 →
  ratio_b_green = 1 →
  ∃ (a b : ℕ), 
    ratio_a_green * a + ratio_b_green * b = total_green ∧
    (ratio_a_blue + ratio_a_green) * a = (ratio_b_blue + ratio_b_green) * b ∧
    ratio_b_blue * b - ratio_a_blue * a = 49 :=
by
  sorry

#check blue_marble_difference

end NUMINAMATH_CALUDE_blue_marble_difference_l3175_317574


namespace NUMINAMATH_CALUDE_mark_spent_40_l3175_317539

/-- The total amount Mark spent on tomatoes and apples -/
def total_spent (tomato_price : ℝ) (tomato_weight : ℝ) (apple_price : ℝ) (apple_weight : ℝ) : ℝ :=
  tomato_price * tomato_weight + apple_price * apple_weight

/-- Theorem stating that Mark spent $40 in total -/
theorem mark_spent_40 : 
  total_spent 5 2 6 5 = 40 := by sorry

end NUMINAMATH_CALUDE_mark_spent_40_l3175_317539


namespace NUMINAMATH_CALUDE_smallest_m_for_tax_price_l3175_317596

theorem smallest_m_for_tax_price : ∃ (x : ℕ), x > 0 ∧ x + (6 * x) / 100 = 2 * 53 * 100 ∧
  ∀ (m : ℕ) (y : ℕ), m > 0 ∧ m < 53 → y > 0 → y + (6 * y) / 100 ≠ 2 * m * 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_tax_price_l3175_317596


namespace NUMINAMATH_CALUDE_logistics_personnel_in_sample_l3175_317560

theorem logistics_personnel_in_sample
  (total_staff : ℕ)
  (logistics_staff : ℕ)
  (sample_size : ℕ)
  (h1 : total_staff = 160)
  (h2 : logistics_staff = 24)
  (h3 : sample_size = 20) :
  (logistics_staff : ℚ) / (total_staff : ℚ) * (sample_size : ℚ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_logistics_personnel_in_sample_l3175_317560


namespace NUMINAMATH_CALUDE_smallest_divisible_by_3_and_4_l3175_317522

theorem smallest_divisible_by_3_and_4 : 
  ∀ n : ℕ, n > 0 ∧ 3 ∣ n ∧ 4 ∣ n → n ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_3_and_4_l3175_317522


namespace NUMINAMATH_CALUDE_feb_1_is_sunday_l3175_317583

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

-- Define a function to get the previous day
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

-- Define a function to get the day n days before a given day
def daysBefore (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => daysBefore (prevDay d) n

-- Theorem statement
theorem feb_1_is_sunday (h : DayOfWeek.Saturday = daysBefore DayOfWeek.Saturday 13) :
  DayOfWeek.Sunday = daysBefore DayOfWeek.Saturday 13 :=
by sorry

end NUMINAMATH_CALUDE_feb_1_is_sunday_l3175_317583


namespace NUMINAMATH_CALUDE_circle_line_distance_l3175_317592

theorem circle_line_distance (a : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x - 4*y = 0}
  let line := {(x, y) : ℝ × ℝ | x - y + a = 0}
  let center := (1, 2)
  let distance := |1 - 2 + a| / Real.sqrt 2
  (distance = Real.sqrt 2 / 2) → (a = 2 ∨ a = 0) := by
sorry

end NUMINAMATH_CALUDE_circle_line_distance_l3175_317592


namespace NUMINAMATH_CALUDE_endpoint_sum_thirteen_l3175_317595

/-- Given a line segment with one endpoint (6,1) and midpoint (3,7),
    the sum of the coordinates of the other endpoint is 13. -/
theorem endpoint_sum_thirteen (x y : ℝ) : 
  (6 + x) / 2 = 3 ∧ (1 + y) / 2 = 7 → x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_thirteen_l3175_317595


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3175_317552

/-- Given an arithmetic sequence {a_n} where a_n ≠ 0 for all n,
    if a_1, a_3, and a_4 form a geometric sequence,
    then the common ratio of this geometric sequence is either 1 or 1/2. -/
theorem arithmetic_geometric_sequence_ratio
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arith : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)  -- Arithmetic sequence condition
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)  -- Non-zero condition
  (h_geom : ∃ q : ℝ, a 3 = a 1 * q ∧ a 4 = a 3 * q)  -- Geometric sequence condition
  : ∃ q : ℝ, (q = 1 ∨ q = 1/2) ∧ a 3 = a 1 * q ∧ a 4 = a 3 * q :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3175_317552


namespace NUMINAMATH_CALUDE_twenty_dollars_combinations_l3175_317569

/-- The number of ways to make 20 dollars with nickels, dimes, and quarters -/
def ways_to_make_20_dollars : ℕ :=
  (Finset.filter (fun (n, d, q) => 
    5 * n + 10 * d + 25 * q = 2000 ∧ 
    n ≥ 2 ∧ 
    q ≥ 1) 
  (Finset.product (Finset.range 401) (Finset.product (Finset.range 201) (Finset.range 81)))).card

/-- Theorem stating that there are exactly 130 ways to make 20 dollars 
    with nickels, dimes, and quarters, using at least two nickels and one quarter -/
theorem twenty_dollars_combinations : ways_to_make_20_dollars = 130 := by
  sorry

end NUMINAMATH_CALUDE_twenty_dollars_combinations_l3175_317569


namespace NUMINAMATH_CALUDE_train_cars_count_l3175_317562

/-- Represents a train with a consistent speed --/
structure Train where
  cars_per_12_seconds : ℕ
  total_passing_time : ℕ

/-- Calculates the total number of cars in the train --/
def total_cars (t : Train) : ℕ :=
  (t.cars_per_12_seconds * t.total_passing_time) / 12

/-- Theorem stating that a train with 8 cars passing in 12 seconds 
    and taking 210 seconds to pass has 140 cars --/
theorem train_cars_count :
  ∀ (t : Train), t.cars_per_12_seconds = 8 ∧ t.total_passing_time = 210 → 
  total_cars t = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l3175_317562


namespace NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_l3175_317572

/-- Represents the number of blocks Ray walks in different segments of his route -/
structure RayWalk where
  to_park : Nat
  to_school : Nat
  to_home : Nat

/-- Represents Ray's daily dog walking routine -/
structure DailyWalk where
  route : RayWalk
  walks_per_day : Nat

/-- Calculates the total number of blocks Ray's dog walks in a day -/
def total_blocks_walked (daily : DailyWalk) : Nat :=
  (daily.route.to_park + daily.route.to_school + daily.route.to_home) * daily.walks_per_day

/-- Theorem stating that Ray's dog walks 66 blocks each day -/
theorem rays_dog_walks_66_blocks (daily : DailyWalk) 
  (h1 : daily.route.to_park = 4)
  (h2 : daily.route.to_school = 7)
  (h3 : daily.route.to_home = 11)
  (h4 : daily.walks_per_day = 3) : 
  total_blocks_walked daily = 66 := by
  sorry

end NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_l3175_317572


namespace NUMINAMATH_CALUDE_probability_n_power_16_mod_6_equals_1_l3175_317580

theorem probability_n_power_16_mod_6_equals_1 (N : ℕ) (h : 1 ≤ N ∧ N ≤ 2000) :
  (Nat.card {n : ℕ | 1 ≤ n ∧ n ≤ 2000 ∧ n^16 % 6 = 1}) / 2000 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_n_power_16_mod_6_equals_1_l3175_317580


namespace NUMINAMATH_CALUDE_p_percentage_of_x_l3175_317526

theorem p_percentage_of_x (x y z w t u p : ℝ) 
  (h1 : 0.37 * z = 0.84 * y)
  (h2 : y = 0.62 * x)
  (h3 : 0.47 * w = 0.73 * z)
  (h4 : w = t - u)
  (h5 : u = 0.25 * t)
  (h6 : p = z + t + u) :
  p = 5.05675 * x := by sorry

end NUMINAMATH_CALUDE_p_percentage_of_x_l3175_317526


namespace NUMINAMATH_CALUDE_two_x_less_than_one_necessary_not_sufficient_l3175_317510

theorem two_x_less_than_one_necessary_not_sufficient :
  (∀ x : ℝ, -1 < x ∧ x < 0 → 2*x < 1) ∧
  (∃ x : ℝ, 2*x < 1 ∧ ¬(-1 < x ∧ x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_two_x_less_than_one_necessary_not_sufficient_l3175_317510


namespace NUMINAMATH_CALUDE_max_value_cos_sin_sum_l3175_317570

theorem max_value_cos_sin_sum :
  ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 ∧ 
  ∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_sum_l3175_317570


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3175_317571

/-- Given a cubic equation x³ + px + q = 0 where p and q are real numbers,
    if 2 + i is a root, then p + q = 9 -/
theorem cubic_root_sum (p q : ℝ) : 
  (Complex.I : ℂ) ^ 3 + p * (Complex.I : ℂ) + q = 0 → p + q = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3175_317571


namespace NUMINAMATH_CALUDE_sum_two_longest_altitudes_eq_14_l3175_317521

/-- A triangle with sides 6, 8, and 10 -/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side1_eq : side1 = 6)
  (side2_eq : side2 = 8)
  (side3_eq : side3 = 10)

/-- The length of an altitude in a triangle -/
def altitude_length (t : Triangle) : ℝ → ℝ :=
  sorry

/-- The sum of the two longest altitudes in the triangle -/
def sum_two_longest_altitudes (t : Triangle) : ℝ :=
  sorry

/-- Theorem: The sum of the two longest altitudes in a triangle with sides 6, 8, and 10 is 14 -/
theorem sum_two_longest_altitudes_eq_14 (t : Triangle) :
  sum_two_longest_altitudes t = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_two_longest_altitudes_eq_14_l3175_317521


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_minimal_m_l3175_317546

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 10) ≤ 0

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Define the sufficient condition
def sufficient (m : ℝ) : Prop :=
  ∀ x, q x m → p x

-- Define the not necessary condition
def not_necessary (m : ℝ) : Prop :=
  ∃ x, p x ∧ ¬(q x m)

-- Main theorem
theorem sufficient_but_not_necessary_condition (m : ℝ) 
  (h1 : m ≥ 3) (h2 : m > 0) : 
  sufficient m ∧ not_necessary m := by
  sorry

-- Prove that this is the minimal value of m
theorem minimal_m :
  ∀ m < 3, ¬(sufficient m ∧ not_necessary m) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_minimal_m_l3175_317546


namespace NUMINAMATH_CALUDE_convex_polygon_perimeter_bound_l3175_317540

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool  -- We simplify the convexity check to a boolean for this statement

/-- A square in 2D space -/
structure Square where
  center : Real × Real
  side_length : Real

/-- Check if a point is inside or on the boundary of a square -/
def point_in_square (p : Real × Real) (s : Square) : Prop :=
  let (x, y) := p
  let (cx, cy) := s.center
  let half_side := s.side_length / 2
  x ≥ cx - half_side ∧ x ≤ cx + half_side ∧
  y ≥ cy - half_side ∧ y ≤ cy + half_side

/-- Check if a polygon is contained in a square -/
def polygon_in_square (p : ConvexPolygon) (s : Square) : Prop :=
  ∀ v ∈ p.vertices, point_in_square v s

/-- Calculate the perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : Real :=
  sorry  -- The actual calculation is omitted for brevity

/-- The main theorem -/
theorem convex_polygon_perimeter_bound (p : ConvexPolygon) (s : Square) :
  p.is_convex = true →
  s.side_length = 1 →
  polygon_in_square p s →
  perimeter p ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_perimeter_bound_l3175_317540


namespace NUMINAMATH_CALUDE_pirate_costume_cost_l3175_317590

theorem pirate_costume_cost (num_friends : ℕ) (total_spent : ℕ) : 
  num_friends = 8 → total_spent = 40 → total_spent / num_friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_pirate_costume_cost_l3175_317590


namespace NUMINAMATH_CALUDE_equal_numbers_l3175_317550

theorem equal_numbers (a b c : ℝ) (h : |a - b| = 2*|b - c| ∧ |a - b| = 3*|c - a|) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_l3175_317550


namespace NUMINAMATH_CALUDE_no_real_solutions_log_equation_l3175_317554

theorem no_real_solutions_log_equation :
  ¬∃ (x : ℝ), (x + 3 > 0 ∧ x - 1 > 0 ∧ x^2 - 2*x - 3 > 0) ∧
  (Real.log (x + 3) + Real.log (x - 1) = Real.log (x^2 - 2*x - 3)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_log_equation_l3175_317554


namespace NUMINAMATH_CALUDE_shorter_leg_length_l3175_317586

/-- A right triangle that can be cut and rearranged into a square -/
structure CuttableRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  is_right_triangle : shorter_leg > 0 ∧ longer_leg > 0
  can_form_square : shorter_leg * 2 = longer_leg

/-- Theorem: If a right triangle with longer leg 10 can be cut and rearranged 
    to form a square, then its shorter leg has length 5 -/
theorem shorter_leg_length (t : CuttableRightTriangle) 
    (h : t.longer_leg = 10) : t.shorter_leg = 5 := by
  sorry

end NUMINAMATH_CALUDE_shorter_leg_length_l3175_317586


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3175_317544

/-- Given a geometric sequence with 7 terms, where the first term is 8 and the last term is 5832,
    prove that the fifth term is 648. -/
theorem fifth_term_of_geometric_sequence (a : Fin 7 → ℝ) :
  (∀ i j, a (i + 1) / a i = a (j + 1) / a j) →  -- geometric sequence condition
  a 0 = 8 →                                     -- first term is 8
  a 6 = 5832 →                                  -- last term is 5832
  a 4 = 648 := by                               -- fifth term (index 4) is 648
sorry


end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3175_317544


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3175_317597

theorem logarithm_expression_equality : 2 * Real.log 2 / Real.log 10 + Real.log (5/8) / Real.log 10 - Real.log 25 / Real.log 10 = -1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3175_317597


namespace NUMINAMATH_CALUDE_common_root_of_equations_l3175_317512

theorem common_root_of_equations : ∃ x : ℚ, 
  2 * x^3 - 5 * x^2 + 6 * x - 2 = 0 ∧ 
  6 * x^3 - 3 * x^2 - 2 * x + 1 = 0 := by
  use 1/2
  sorry

#eval (2 * (1/2)^3 - 5 * (1/2)^2 + 6 * (1/2) - 2 : ℚ)
#eval (6 * (1/2)^3 - 3 * (1/2)^2 - 2 * (1/2) + 1 : ℚ)

end NUMINAMATH_CALUDE_common_root_of_equations_l3175_317512


namespace NUMINAMATH_CALUDE_octal_sum_equality_l3175_317548

/-- Represents a number in base 8 --/
def OctalNumber : Type := List Nat

/-- Converts an OctalNumber to a natural number --/
def octal_to_nat (n : OctalNumber) : Nat :=
  n.foldl (fun acc d => 8 * acc + d) 0

/-- Adds two OctalNumbers in base 8 --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

theorem octal_sum_equality : 
  octal_add [1, 4, 6, 3] [2, 7, 5] = [1, 7, 5, 0] :=
sorry

end NUMINAMATH_CALUDE_octal_sum_equality_l3175_317548


namespace NUMINAMATH_CALUDE_certain_number_problem_l3175_317558

theorem certain_number_problem (x : ℤ) (h : x + 5 * 8 = 340) : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3175_317558


namespace NUMINAMATH_CALUDE_f_triple_3_l3175_317529

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_triple_3 : f (f (f 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_triple_3_l3175_317529


namespace NUMINAMATH_CALUDE_race_participants_l3175_317536

theorem race_participants (total : ℕ) (finished : ℕ) : 
  finished = 52 →
  (3/4 : ℚ) * total * (1/3 : ℚ) + 
  (3/4 : ℚ) * total * (2/3 : ℚ) * (4/5 : ℚ) = finished →
  total = 130 := by
  sorry

end NUMINAMATH_CALUDE_race_participants_l3175_317536


namespace NUMINAMATH_CALUDE_greatest_integer_value_five_satisfies_condition_no_greater_integer_greatest_integer_is_five_l3175_317561

theorem greatest_integer_value (x : ℤ) : (3 * Int.natAbs x + 4 ≤ 19) → x ≤ 5 :=
by sorry

theorem five_satisfies_condition : 3 * Int.natAbs 5 + 4 ≤ 19 :=
by sorry

theorem no_greater_integer (y : ℤ) : y > 5 → (3 * Int.natAbs y + 4 > 19) :=
by sorry

theorem greatest_integer_is_five : 
  ∃ (x : ℤ), (3 * Int.natAbs x + 4 ≤ 19) ∧ (∀ (y : ℤ), (3 * Int.natAbs y + 4 ≤ 19) → y ≤ x) ∧ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_value_five_satisfies_condition_no_greater_integer_greatest_integer_is_five_l3175_317561


namespace NUMINAMATH_CALUDE_max_cells_hit_five_times_l3175_317587

/-- Represents a triangular cell in the grid -/
structure TriangularCell :=
  (id : ℕ)

/-- Represents the entire triangular grid -/
structure TriangularGrid :=
  (cells : List TriangularCell)

/-- Represents a shot fired by the marksman -/
structure Shot :=
  (target : TriangularCell)

/-- Function to determine if two cells are adjacent -/
def areAdjacent (c1 c2 : TriangularCell) : Bool :=
  sorry

/-- Function to determine where a shot lands -/
def shotLands (s : Shot) (g : TriangularGrid) : TriangularCell :=
  sorry

/-- Function to count the number of hits on a cell -/
def countHits (c : TriangularCell) (shots : List Shot) : ℕ :=
  sorry

/-- Theorem stating the maximum number of cells that can be hit exactly five times -/
theorem max_cells_hit_five_times (g : TriangularGrid) :
  (∃ (shots : List Shot), 
    (∀ c : TriangularCell, c ∈ g.cells → countHits c shots ≤ 5) ∧ 
    (∃ cells : List TriangularCell, 
      cells.length = 25 ∧ 
      (∀ c : TriangularCell, c ∈ cells → countHits c shots = 5))) ∧
  (∀ (shots : List Shot),
    ¬∃ cells : List TriangularCell, 
      cells.length > 25 ∧ 
      (∀ c : TriangularCell, c ∈ cells → countHits c shots = 5)) :=
  sorry

end NUMINAMATH_CALUDE_max_cells_hit_five_times_l3175_317587


namespace NUMINAMATH_CALUDE_smarties_remainder_l3175_317565

theorem smarties_remainder (n : ℕ) (h : n % 11 = 6) : (4 * n) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_smarties_remainder_l3175_317565


namespace NUMINAMATH_CALUDE_petrol_price_reduction_l3175_317516

/-- The original price of petrol in dollars per gallon -/
def P : ℝ := sorry

/-- The amount spent on petrol in dollars -/
def amount_spent : ℝ := 250

/-- The price reduction percentage as a decimal -/
def price_reduction : ℝ := 0.1

/-- The additional gallons that can be bought after the price reduction -/
def additional_gallons : ℝ := 5

/-- Theorem stating the relationship between the original price and the additional gallons that can be bought after the price reduction -/
theorem petrol_price_reduction (P : ℝ) (amount_spent : ℝ) (price_reduction : ℝ) (additional_gallons : ℝ) :
  amount_spent / ((1 - price_reduction) * P) - amount_spent / P = additional_gallons :=
sorry

end NUMINAMATH_CALUDE_petrol_price_reduction_l3175_317516


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3175_317553

theorem complementary_angles_difference (a b : ℝ) (h1 : a + b = 90) (h2 : a / b = 5 / 4) :
  |a - b| = 10 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3175_317553


namespace NUMINAMATH_CALUDE_slope_of_tan_45_degrees_line_l3175_317509

theorem slope_of_tan_45_degrees_line (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.tan (45 * π / 180)
  (deriv f) x = 0 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_tan_45_degrees_line_l3175_317509


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3175_317563

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

/-- The 150th term of the specific arithmetic sequence -/
def term_150 : ℝ :=
  arithmetic_sequence 3 5 150

theorem arithmetic_sequence_150th_term :
  term_150 = 748 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3175_317563


namespace NUMINAMATH_CALUDE_stock_price_increase_l3175_317557

theorem stock_price_increase (X : ℝ) : 
  (1 + X / 100) * (1 - 25 / 100) * (1 + 15 / 100) = 103.5 / 100 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l3175_317557


namespace NUMINAMATH_CALUDE_system_solutions_l3175_317541

def is_solution (x y z u : ℤ) : Prop :=
  x + y + z + u = 12 ∧
  x^2 + y^2 + z^2 + u^2 = 170 ∧
  x^3 + y^3 + z^3 + u^3 = 1764 ∧
  x * y = z * u

def solutions : List (ℤ × ℤ × ℤ × ℤ) :=
  [(12, -1, 4, -3), (12, -1, -3, 4), (-1, 12, 4, -3), (-1, 12, -3, 4),
   (4, -3, 12, -1), (4, -3, -1, 12), (-3, 4, 12, -1), (-3, 4, -1, 12)]

theorem system_solutions :
  (∀ x y z u : ℤ, is_solution x y z u ↔ (x, y, z, u) ∈ solutions) ∧
  solutions.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3175_317541


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3175_317556

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (2, y)
  parallel a b → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3175_317556


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3175_317577

/-- Calculates the speed of a train given the lengths of two trains, the speed of the second train, and the time taken for the first train to pass the second train. -/
theorem train_speed_calculation (length1 length2 : ℝ) (speed2 : ℝ) (time : ℝ) :
  length1 = 250 →
  length2 = 300 →
  speed2 = 36 * (1000 / 3600) →
  time = 54.995600351971845 →
  ∃ (speed1 : ℝ), speed1 = 72 * (1000 / 3600) ∧
    (length1 + length2) / time = speed1 - speed2 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3175_317577


namespace NUMINAMATH_CALUDE_average_existence_l3175_317568

theorem average_existence : ∃ N : ℝ, 12 < N ∧ N < 18 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_existence_l3175_317568


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3175_317578

theorem absolute_value_simplification (a : ℝ) (h : a < 3) : |a - 3| = 3 - a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3175_317578


namespace NUMINAMATH_CALUDE_volleyball_match_probability_l3175_317527

-- Define the probability of Team A winning a single game
def p_win_game : ℚ := 2/3

-- Define the probability of Team A winning the match
def p_win_match : ℚ := 20/27

-- Theorem statement
theorem volleyball_match_probability :
  (p_win_game = 2/3) →  -- Probability of Team A winning a single game
  (p_win_match = p_win_game * p_win_game + 2 * p_win_game * (1 - p_win_game) * p_win_game) :=
by
  sorry

#check volleyball_match_probability

end NUMINAMATH_CALUDE_volleyball_match_probability_l3175_317527


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3175_317567

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where a_2 + a_6 = 10, prove that a_4 = 5. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 2 + a 6 = 10) : 
  a 4 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3175_317567


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3175_317503

theorem absolute_value_inequality (x : ℝ) :
  |x + 3| > 1 ↔ x < -4 ∨ x > -2 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3175_317503


namespace NUMINAMATH_CALUDE_three_cubes_exposed_faces_sixty_cubes_exposed_faces_l3175_317502

/-- The number of exposed faces for n cubes in a row on a table -/
def exposed_faces (n : ℕ) : ℕ := 3 * n + 2

/-- Theorem stating that for 3 cubes, there are 11 exposed faces -/
theorem three_cubes_exposed_faces : exposed_faces 3 = 11 := by sorry

/-- Theorem to prove the number of exposed faces for 60 cubes -/
theorem sixty_cubes_exposed_faces : exposed_faces 60 = 182 := by sorry

end NUMINAMATH_CALUDE_three_cubes_exposed_faces_sixty_cubes_exposed_faces_l3175_317502


namespace NUMINAMATH_CALUDE_vector_operation_l3175_317543

def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

theorem vector_operation :
  (2 : ℝ) • a - b = (5, 7) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3175_317543


namespace NUMINAMATH_CALUDE_digging_time_for_second_hole_l3175_317517

/-- Proves that given the conditions of the digging problem, the time required to dig the second hole is 6 hours -/
theorem digging_time_for_second_hole 
  (workers_first : ℕ) 
  (hours_first : ℕ) 
  (depth_first : ℕ) 
  (extra_workers : ℕ) 
  (depth_second : ℕ) 
  (h : workers_first = 45)
  (i : hours_first = 8)
  (j : depth_first = 30)
  (k : extra_workers = 65)
  (l : depth_second = 55) :
  (workers_first + extra_workers) * (660 / (workers_first + extra_workers) : ℚ) * depth_second = 
  workers_first * hours_first * depth_second := by
sorry

#eval (45 + 65) * (660 / (45 + 65) : ℚ)

end NUMINAMATH_CALUDE_digging_time_for_second_hole_l3175_317517


namespace NUMINAMATH_CALUDE_golden_ratio_comparison_l3175_317519

theorem golden_ratio_comparison : (Real.sqrt 5 - 1) / 2 > 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_comparison_l3175_317519


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l3175_317501

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Theorem: The result of (101110₂) × (110100₂) ÷ (110₂) is 101011100₂ -/
theorem binary_multiplication_division :
  let a := binaryToNat [true, false, true, true, true, false]  -- 101110₂
  let b := binaryToNat [true, true, false, true, false, false] -- 110100₂
  let c := binaryToNat [true, true, false]                     -- 110₂
  let result := binaryToNat [true, false, true, false, true, true, true, false, false] -- 101011100₂
  a * b / c = result := by
  sorry


end NUMINAMATH_CALUDE_binary_multiplication_division_l3175_317501


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3175_317584

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 3
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3175_317584


namespace NUMINAMATH_CALUDE_tetrahedron_acute_angle_vertex_l3175_317594

/-- A tetrahedron is represented by its four vertices in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- The plane angle at a vertex of a tetrahedron -/
def planeAngle (t : Tetrahedron) (v : Fin 4) (e1 e2 : Fin 4) : ℝ :=
  sorry

/-- Theorem: In any tetrahedron, there exists at least one vertex where all plane angles are acute -/
theorem tetrahedron_acute_angle_vertex (t : Tetrahedron) : 
  ∃ v : Fin 4, ∀ e1 e2 : Fin 4, e1 ≠ e2 → e1 ≠ v → e2 ≠ v → planeAngle t v e1 e2 < π / 2 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_acute_angle_vertex_l3175_317594


namespace NUMINAMATH_CALUDE_abc_is_zero_l3175_317532

theorem abc_is_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) :
  a * b * c = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_is_zero_l3175_317532


namespace NUMINAMATH_CALUDE_distance_A_to_B_l3175_317549

/-- The distance between points A(1, 0) and B(0, -1) is √2. -/
theorem distance_A_to_B : Real.sqrt 2 = Real.sqrt ((0 - 1)^2 + (-1 - 0)^2) := by sorry

end NUMINAMATH_CALUDE_distance_A_to_B_l3175_317549


namespace NUMINAMATH_CALUDE_corporation_employee_count_l3175_317514

/-- The number of employees at a corporation. -/
structure Corporation where
  female_employees : ℕ
  total_managers : ℕ
  male_associates : ℕ
  female_managers : ℕ

/-- The total number of employees in the corporation. -/
def Corporation.total_employees (c : Corporation) : ℕ :=
  c.female_employees + c.male_associates + (c.total_managers - c.female_managers)

/-- Theorem stating that the total number of employees is 250 given the specific conditions. -/
theorem corporation_employee_count (c : Corporation)
  (h1 : c.female_employees = 90)
  (h2 : c.total_managers = 40)
  (h3 : c.male_associates = 160)
  (h4 : c.female_managers = 40) :
  c.total_employees = 250 := by
  sorry

#check corporation_employee_count

end NUMINAMATH_CALUDE_corporation_employee_count_l3175_317514


namespace NUMINAMATH_CALUDE_division_remainder_and_divisibility_l3175_317506

theorem division_remainder_and_divisibility : 
  let n : ℕ := 1234567
  let d : ℕ := 256
  let r : ℕ := n % d
  (r = 2) ∧ (r % 7 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_division_remainder_and_divisibility_l3175_317506


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3175_317585

theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 27*x + a = (3*x + b)^2) → a = 20.25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3175_317585


namespace NUMINAMATH_CALUDE_total_spent_is_638_l3175_317530

/-- The total amount spent by Elizabeth, Emma, and Elsa -/
def total_spent (emma_spent : ℕ) : ℕ :=
  let elsa_spent := 2 * emma_spent
  let elizabeth_spent := 4 * elsa_spent
  emma_spent + elsa_spent + elizabeth_spent

/-- Theorem stating that the total amount spent is 638 -/
theorem total_spent_is_638 : total_spent 58 = 638 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_638_l3175_317530


namespace NUMINAMATH_CALUDE_circle_radius_proof_l3175_317573

theorem circle_radius_proof (chord_length : ℝ) (center_to_intersection : ℝ) (ratio_left : ℝ) (ratio_right : ℝ) :
  chord_length = 18 →
  center_to_intersection = 7 →
  ratio_left = 2 * ratio_right →
  ratio_left + ratio_right = chord_length →
  ∃ (radius : ℝ), radius = 11 ∧ 
    (radius - center_to_intersection) * (radius + center_to_intersection) = ratio_left * ratio_right :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l3175_317573


namespace NUMINAMATH_CALUDE_vishal_investment_percentage_l3175_317576

def total_investment : ℝ := 6358
def raghu_investment : ℝ := 2200
def trishul_investment_percentage : ℝ := 90  -- 100% - 10%

theorem vishal_investment_percentage (vishal_investment trishul_investment : ℝ) : 
  vishal_investment + trishul_investment + raghu_investment = total_investment →
  trishul_investment = raghu_investment * trishul_investment_percentage / 100 →
  (vishal_investment - trishul_investment) / trishul_investment * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_vishal_investment_percentage_l3175_317576


namespace NUMINAMATH_CALUDE_total_enjoyable_gameplay_l3175_317523

/-- Calculates the total enjoyable gameplay time given the conditions of the game, expansion, and mod. -/
theorem total_enjoyable_gameplay 
  (original_game_hours : ℝ)
  (original_game_boring_percent : ℝ)
  (expansion_hours : ℝ)
  (expansion_load_screen_percent : ℝ)
  (expansion_inventory_percent : ℝ)
  (mod_skip_percent : ℝ)
  (h1 : original_game_hours = 150)
  (h2 : original_game_boring_percent = 0.7)
  (h3 : expansion_hours = 50)
  (h4 : expansion_load_screen_percent = 0.25)
  (h5 : expansion_inventory_percent = 0.25)
  (h6 : mod_skip_percent = 0.15) :
  let original_enjoyable := original_game_hours * (1 - original_game_boring_percent)
  let expansion_enjoyable := expansion_hours * (1 - expansion_load_screen_percent) * (1 - expansion_inventory_percent)
  let total_tedious := original_game_hours * original_game_boring_percent + 
                       expansion_hours * (expansion_load_screen_percent + (1 - expansion_load_screen_percent) * expansion_inventory_percent)
  let mod_skipped := total_tedious * mod_skip_percent
  original_enjoyable + expansion_enjoyable + mod_skipped = 92.15625 := by
  sorry


end NUMINAMATH_CALUDE_total_enjoyable_gameplay_l3175_317523


namespace NUMINAMATH_CALUDE_total_trees_on_farm_l3175_317579

def farm_trees (mango_trees : ℕ) (coconut_trees : ℕ) : ℕ :=
  mango_trees + coconut_trees

theorem total_trees_on_farm :
  let mango_trees : ℕ := 60
  let coconut_trees : ℕ := mango_trees / 2 - 5
  farm_trees mango_trees coconut_trees = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_on_farm_l3175_317579


namespace NUMINAMATH_CALUDE_time_sum_after_increment_l3175_317518

-- Define a type for time on a 12-hour digital clock
structure Time12Hour where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ
  is_pm : Bool

-- Function to add hours, minutes, and seconds to a given time
def addTime (start : Time12Hour) (hours minutes seconds : ℕ) : Time12Hour :=
  sorry

-- Function to calculate A + B + C for a given time
def sumTime (t : Time12Hour) : ℕ :=
  t.hours + t.minutes + t.seconds

-- Theorem statement
theorem time_sum_after_increment :
  let start_time := Time12Hour.mk 3 0 0 true
  let end_time := addTime start_time 190 45 30
  sumTime end_time = 76 := by sorry

end NUMINAMATH_CALUDE_time_sum_after_increment_l3175_317518


namespace NUMINAMATH_CALUDE_min_t_for_equations_l3175_317507

theorem min_t_for_equations (a b c d e : ℝ) 
  (h_non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0) 
  (h_sum_pos : a + b + c + d + e > 0) :
  (∃ t : ℝ, t = Real.sqrt 2 ∧ 
    a + c = t * b ∧ 
    b + d = t * c ∧ 
    c + e = t * d) ∧
  (∀ s : ℝ, (a + c = s * b ∧ b + d = s * c ∧ c + e = s * d) → s ≥ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_t_for_equations_l3175_317507


namespace NUMINAMATH_CALUDE_fraction_equality_implies_relationship_l3175_317555

theorem fraction_equality_implies_relationship (a b c d : ℝ) :
  (a + b + 1) / (b + c + 2) = (c + d + 1) / (d + a + 2) →
  (a - c) * (a + b + c + d + 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_relationship_l3175_317555


namespace NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l3175_317524

theorem polynomial_multiplication_simplification (x : ℝ) :
  (3 * x - 2) * (5 * x^12 + 3 * x^11 + 5 * x^10 + 3 * x^9) =
  15 * x^13 - x^12 + 9 * x^11 - x^10 - 6 * x^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l3175_317524


namespace NUMINAMATH_CALUDE_david_catches_cory_l3175_317593

/-- The length of the track in meters -/
def track_length : ℝ := 600

/-- Cory's initial lead in meters -/
def initial_lead : ℝ := 50

/-- David's speed relative to Cory's -/
def speed_ratio : ℝ := 1.5

/-- Number of laps David runs when he first catches up to Cory -/
def david_laps : ℝ := 2

theorem david_catches_cory :
  ∃ (cory_speed : ℝ), cory_speed > 0 →
  let david_speed := speed_ratio * cory_speed
  let catch_up_distance := david_laps * track_length
  catch_up_distance * (1 / david_speed - 1 / cory_speed) = initial_lead := by
  sorry

end NUMINAMATH_CALUDE_david_catches_cory_l3175_317593


namespace NUMINAMATH_CALUDE_y_not_between_l3175_317525

theorem y_not_between (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ x y : ℝ, y = (a * Real.sin x + b) / (a * Real.sin x - b) →
  (a > b → (y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b))) :=
by sorry

end NUMINAMATH_CALUDE_y_not_between_l3175_317525


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3175_317559

/-- A random variable following a normal distribution with mean 2 and variance 4 -/
def X : Real → Real := sorry

/-- The probability density function of X -/
def pdf_X : Real → Real := sorry

/-- The cumulative distribution function of X -/
def cdf_X : Real → Real := sorry

/-- The value 'a' such that P(X < a) = 0.2 -/
def a : Real := sorry

theorem normal_distribution_symmetry (h1 : ∀ x, pdf_X x = pdf_X (4 - x))
  (h2 : cdf_X a = 0.2) : cdf_X (4 - a) = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3175_317559


namespace NUMINAMATH_CALUDE_north_american_stamps_cost_is_91_cents_l3175_317513

/-- Represents a country --/
inductive Country
| China
| Japan
| Canada
| Mexico

/-- Represents a continent --/
inductive Continent
| Asia
| NorthAmerica

/-- Represents a decade --/
inductive Decade
| D1960s
| D1970s

/-- Maps a country to its continent --/
def country_continent : Country → Continent
| Country.China => Continent.Asia
| Country.Japan => Continent.Asia
| Country.Canada => Continent.NorthAmerica
| Country.Mexico => Continent.NorthAmerica

/-- Cost of stamps in cents for each country --/
def stamp_cost : Country → ℕ
| Country.China => 7
| Country.Japan => 7
| Country.Canada => 3
| Country.Mexico => 4

/-- Number of stamps for each country and decade --/
def stamp_count : Country → Decade → ℕ
| Country.China => fun
  | Decade.D1960s => 5
  | Decade.D1970s => 9
| Country.Japan => fun
  | Decade.D1960s => 6
  | Decade.D1970s => 7
| Country.Canada => fun
  | Decade.D1960s => 7
  | Decade.D1970s => 6
| Country.Mexico => fun
  | Decade.D1960s => 8
  | Decade.D1970s => 5

/-- Total cost of North American stamps from 1960s and 1970s --/
def north_american_stamps_cost : ℚ :=
  let north_american_countries := [Country.Canada, Country.Mexico]
  let decades := [Decade.D1960s, Decade.D1970s]
  (north_american_countries.map fun country =>
    (decades.map fun decade =>
      (stamp_count country decade) * (stamp_cost country)
    ).sum
  ).sum / 100

theorem north_american_stamps_cost_is_91_cents :
  north_american_stamps_cost = 91 / 100 := by sorry

end NUMINAMATH_CALUDE_north_american_stamps_cost_is_91_cents_l3175_317513
