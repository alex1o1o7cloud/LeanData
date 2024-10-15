import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l375_37578

-- Define a plane
class Plane where
  -- Add any necessary properties for a plane

-- Define a line
class Line where
  -- Add any necessary properties for a line

-- Define perpendicularity between a line and a plane
def perpendicular_to_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Definition of perpendicularity between a line and a plane

-- Define parallel lines
def parallel_lines (l1 l2 : Line) : Prop :=
  sorry -- Definition of parallel lines

-- Theorem statement
theorem perpendicular_lines_parallel (p : Plane) (l1 l2 : Line) :
  perpendicular_to_plane l1 p → perpendicular_to_plane l2 p → parallel_lines l1 l2 :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l375_37578


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l375_37507

/-- The surface area of the circumscribed sphere of a rectangular solid with face diagonals √3, √5, and 2 -/
theorem circumscribed_sphere_surface_area (a b c : ℝ) : 
  a^2 + b^2 = 3 → b^2 + c^2 = 5 → c^2 + a^2 = 4 → 
  4 * Real.pi * ((a^2 + b^2 + c^2) / 4) = 6 * Real.pi := by
  sorry

#check circumscribed_sphere_surface_area

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l375_37507


namespace NUMINAMATH_CALUDE_uniform_profit_percentage_clock_sales_l375_37555

/-- Uniform profit percentage calculation for clock sales --/
theorem uniform_profit_percentage_clock_sales
  (total_clocks : ℕ)
  (clocks_10_percent : ℕ)
  (clocks_20_percent : ℕ)
  (cost_price : ℚ)
  (revenue_difference : ℚ)
  (h1 : total_clocks = clocks_10_percent + clocks_20_percent)
  (h2 : total_clocks = 90)
  (h3 : clocks_10_percent = 40)
  (h4 : clocks_20_percent = 50)
  (h5 : cost_price = 79.99999999999773)
  (h6 : revenue_difference = 40) :
  let actual_revenue := clocks_10_percent * (cost_price * (1 + 10 / 100)) +
                        clocks_20_percent * (cost_price * (1 + 20 / 100))
  let uniform_revenue := actual_revenue - revenue_difference
  let uniform_profit_percentage := (uniform_revenue / (total_clocks * cost_price) - 1) * 100
  uniform_profit_percentage = 15 :=
sorry

end NUMINAMATH_CALUDE_uniform_profit_percentage_clock_sales_l375_37555


namespace NUMINAMATH_CALUDE_basketball_tryouts_l375_37525

theorem basketball_tryouts (girls : ℕ) (boys : ℕ) (called_back : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : called_back = 26) :
  girls + boys - called_back = 17 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l375_37525


namespace NUMINAMATH_CALUDE_custom_op_solution_l375_37562

/-- Custom operation "*" for positive integers -/
def custom_op (k n : ℕ+) : ℕ := (n : ℕ) * (2 * k + n - 1) / 2

/-- Theorem stating that if 3 * n = 150 using the custom operation, then n = 15 -/
theorem custom_op_solution :
  ∃ (n : ℕ+), custom_op 3 n = 150 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l375_37562


namespace NUMINAMATH_CALUDE_specific_theater_seats_l375_37547

/-- A theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculate the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- The theorem stating the total number of seats in the specific theater -/
theorem specific_theater_seats :
  let t : Theater := { first_row_seats := 14, seat_increase := 3, last_row_seats := 50 }
  total_seats t = 416 := by
  sorry


end NUMINAMATH_CALUDE_specific_theater_seats_l375_37547


namespace NUMINAMATH_CALUDE_other_triangle_area_ratio_l375_37563

/-- Represents a right triangle with a point on its hypotenuse and parallel lines dividing it -/
structure DividedRightTriangle where
  /-- The area of one small right triangle -/
  smallTriangleArea : ℝ
  /-- The area of the rectangle -/
  rectangleArea : ℝ
  /-- The ratio of the small triangle area to the rectangle area -/
  n : ℝ
  /-- The ratio of the longer side to the shorter side of the rectangle -/
  k : ℝ
  /-- The small triangle area is n times the rectangle area -/
  area_relation : smallTriangleArea = n * rectangleArea
  /-- The sides of the rectangle are in the ratio 1:k -/
  rectangle_ratio : k > 0

/-- The ratio of the area of the other small right triangle to the area of the rectangle is n -/
theorem other_triangle_area_ratio (t : DividedRightTriangle) :
    ∃ otherTriangleArea : ℝ, otherTriangleArea / t.rectangleArea = t.n := by
  sorry

end NUMINAMATH_CALUDE_other_triangle_area_ratio_l375_37563


namespace NUMINAMATH_CALUDE_roots_less_than_one_l375_37540

theorem roots_less_than_one (a b : ℝ) 
  (h1 : |a| + |b| < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x : ℝ, x^2 + a*x + b = 0 → |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_less_than_one_l375_37540


namespace NUMINAMATH_CALUDE_total_count_is_2552_l375_37560

/-- Represents the total count for a week given the number of items and the counting schedule. -/
def weeklyCount (tiles books windows chairs lightBulbs : ℕ) : ℕ :=
  let monday := tiles * 2 + books * 2 + windows * 2
  let tuesday := tiles * 3 + books * 2 + windows * 1
  let wednesday := chairs * 4 + lightBulbs * 5
  let thursday := tiles * 1 + chairs * 2 + books * 3 + windows * 4 + lightBulbs * 5
  let friday := tiles * 1 + books * 2 + chairs * 2 + windows * 3 + lightBulbs * 3
  monday + tuesday + wednesday + thursday + friday

/-- Theorem stating that the total count for the week is 2552 given the specific item counts. -/
theorem total_count_is_2552 : weeklyCount 60 120 10 80 24 = 2552 := by
  sorry

#eval weeklyCount 60 120 10 80 24

end NUMINAMATH_CALUDE_total_count_is_2552_l375_37560


namespace NUMINAMATH_CALUDE_six_hundred_million_scientific_notation_l375_37518

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem six_hundred_million_scientific_notation :
  toScientificNotation 600000000 = ScientificNotation.mk 6 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_six_hundred_million_scientific_notation_l375_37518


namespace NUMINAMATH_CALUDE_andy_remaining_demerits_l375_37584

/-- The maximum number of demerits allowed in a month before firing -/
def max_demerits : ℕ := 50

/-- The number of demerits per late instance -/
def demerits_per_late : ℕ := 2

/-- The number of times Andy was late -/
def times_late : ℕ := 6

/-- The number of demerits for the inappropriate joke -/
def joke_demerits : ℕ := 15

/-- The number of additional demerits Andy can get before being fired -/
def remaining_demerits : ℕ := max_demerits - (demerits_per_late * times_late + joke_demerits)

theorem andy_remaining_demerits : remaining_demerits = 23 := by
  sorry

end NUMINAMATH_CALUDE_andy_remaining_demerits_l375_37584


namespace NUMINAMATH_CALUDE_valid_12_letter_words_mod_1000_l375_37512

/-- Represents a letter in Zuminglish -/
inductive ZumLetter
| M
| O
| P

/-- Represents a Zuminglish word -/
def ZumWord := List ZumLetter

/-- Checks if a letter is a vowel -/
def isVowel (l : ZumLetter) : Bool :=
  match l with
  | ZumLetter.O => true
  | _ => false

/-- Checks if a Zuminglish word is valid -/
def isValidWord (w : ZumWord) : Bool :=
  sorry

/-- Counts the number of valid n-letter Zuminglish words -/
def countValidWords (n : Nat) : Nat :=
  sorry

/-- The main theorem: number of valid 12-letter Zuminglish words modulo 1000 -/
theorem valid_12_letter_words_mod_1000 :
  countValidWords 12 % 1000 = 416 := by
  sorry

end NUMINAMATH_CALUDE_valid_12_letter_words_mod_1000_l375_37512


namespace NUMINAMATH_CALUDE_positive_intervals_l375_37554

-- Define the expression
def f (x : ℝ) : ℝ := (x + 2) * (x - 2)

-- State the theorem
theorem positive_intervals (x : ℝ) : f x > 0 ↔ x < -2 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_intervals_l375_37554


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l375_37520

/-- Represents a 4x4x4 cube composed of unit cubes -/
structure LargeCube where
  size : ℕ
  size_eq : size = 4

/-- Represents a plane intersecting the large cube -/
structure IntersectingPlane where
  cube : LargeCube
  ratio : ℚ
  ratio_eq : ratio = 1 / 3

/-- Counts the number of unit cubes intersected by the plane -/
def count_intersected_cubes (plane : IntersectingPlane) : ℕ := sorry

/-- Theorem stating that the plane intersects 32 unit cubes -/
theorem intersected_cubes_count (plane : IntersectingPlane) : 
  count_intersected_cubes plane = 32 := by sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l375_37520


namespace NUMINAMATH_CALUDE_eva_orange_count_l375_37549

/-- Calculates the number of oranges Eva needs to buy given her dietary requirements --/
def calculate_oranges (total_days : ℕ) (orange_frequency : ℕ) : ℕ :=
  total_days / orange_frequency

/-- Theorem stating that Eva needs to buy 10 oranges given her dietary requirements --/
theorem eva_orange_count : calculate_oranges 30 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_eva_orange_count_l375_37549


namespace NUMINAMATH_CALUDE_fifteen_star_positive_integer_count_l375_37564

def star (a b : ℤ) : ℚ := a^3 / b

theorem fifteen_star_positive_integer_count :
  (∃ (S : Finset ℤ), (∀ x ∈ S, x > 0 ∧ (star 15 x).isInt) ∧ S.card = 16) :=
sorry

end NUMINAMATH_CALUDE_fifteen_star_positive_integer_count_l375_37564


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l375_37509

theorem sum_of_coefficients (a b c d : ℤ) :
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 11*x + 6) →
  a + b + c + d = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l375_37509


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l375_37550

theorem rectangle_area_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (2 * (a + c) = 2 * (2 * (b + c))) → (a = 2 * b) →
  ((a * c) = 2 * (b * c)) := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l375_37550


namespace NUMINAMATH_CALUDE_cone_volume_from_sector_l375_37524

/-- Given a cone whose lateral surface area is a sector with central angle 120° and area 3π,
    prove that the volume of the cone is (2√2π)/3 -/
theorem cone_volume_from_sector (θ : Real) (A : Real) (V : Real) : 
  θ = 2 * π / 3 →  -- 120° in radians
  A = 3 * π →
  V = (2 * Real.sqrt 2 * π) / 3 →
  ∃ (r l h : Real),
    r > 0 ∧ l > 0 ∧ h > 0 ∧
    A = (1/2) * l^2 * θ ∧  -- Area of sector
    r = l * θ / (2 * π) ∧  -- Relation between radius and arc length
    h^2 = l^2 - r^2 ∧     -- Pythagorean theorem
    V = (1/3) * π * r^2 * h  -- Volume of cone
    := by sorry

end NUMINAMATH_CALUDE_cone_volume_from_sector_l375_37524


namespace NUMINAMATH_CALUDE_candy_expenditure_l375_37516

theorem candy_expenditure (total_spent : ℚ) :
  total_spent = 75 →
  (1 / 2 : ℚ) + (1 / 3 : ℚ) + (1 / 10 : ℚ) + (candy_fraction : ℚ) = 1 →
  candy_fraction * total_spent = 5 :=
by sorry

end NUMINAMATH_CALUDE_candy_expenditure_l375_37516


namespace NUMINAMATH_CALUDE_pen_cost_l375_37528

theorem pen_cost (pen_price pencil_price : ℚ) 
  (h1 : 6 * pen_price + 2 * pencil_price = 348/100)
  (h2 : 3 * pen_price + 4 * pencil_price = 234/100) :
  pen_price = 51/100 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l375_37528


namespace NUMINAMATH_CALUDE_rectangle_breadth_l375_37503

theorem rectangle_breadth (area : ℝ) (length_ratio : ℝ) :
  area = 460 →
  length_ratio = 1.15 →
  ∃ (breadth : ℝ), 
    area = length_ratio * breadth * breadth ∧
    breadth = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l375_37503


namespace NUMINAMATH_CALUDE_strip_length_is_one_million_l375_37579

/-- The number of meters in a kilometer -/
def meters_per_km : ℕ := 1000

/-- The number of cubic meters in a cubic kilometer -/
def cubic_meters_in_cubic_km : ℕ := meters_per_km ^ 3

/-- The length of the strip in kilometers -/
def strip_length_km : ℕ := cubic_meters_in_cubic_km / meters_per_km

theorem strip_length_is_one_million :
  strip_length_km = 1000000 := by
  sorry


end NUMINAMATH_CALUDE_strip_length_is_one_million_l375_37579


namespace NUMINAMATH_CALUDE_basketball_game_ratio_l375_37582

theorem basketball_game_ratio :
  let girls : ℕ := 30
  let boys : ℕ := girls + 18
  let ratio : ℚ := boys / girls
  ratio = 8 / 5 := by sorry

end NUMINAMATH_CALUDE_basketball_game_ratio_l375_37582


namespace NUMINAMATH_CALUDE_cost_per_meal_is_8_l375_37510

-- Define the number of adults
def num_adults : ℕ := 2

-- Define the number of children
def num_children : ℕ := 5

-- Define the total bill amount
def total_bill : ℚ := 56

-- Define the total number of people
def total_people : ℕ := num_adults + num_children

-- Theorem to prove
theorem cost_per_meal_is_8 : 
  total_bill / total_people = 8 := by sorry

end NUMINAMATH_CALUDE_cost_per_meal_is_8_l375_37510


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l375_37588

/-- The greatest distance between centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 24)
  (h_height : rectangle_height = 18)
  (h_diameter : circle_diameter = 8)
  (h_nonneg_width : 0 ≤ rectangle_width)
  (h_nonneg_height : 0 ≤ rectangle_height)
  (h_nonneg_diameter : 0 ≤ circle_diameter)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ d : ℝ, d = Real.sqrt 356 ∧
    ∀ d' : ℝ, d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 ∧
        d' = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l375_37588


namespace NUMINAMATH_CALUDE_rational_equal_to_reciprocal_l375_37519

theorem rational_equal_to_reciprocal (x : ℚ) : x = 1 ∨ x = -1 ↔ x = 1 / x := by sorry

end NUMINAMATH_CALUDE_rational_equal_to_reciprocal_l375_37519


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l375_37535

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬(is_divisible (y + 3) 18 ∧ is_divisible (y + 3) 70 ∧ 
                     is_divisible (y + 3) 25 ∧ is_divisible (y + 3) 21)) ∧
  (is_divisible (x + 3) 18 ∧ is_divisible (x + 3) 70 ∧ 
   is_divisible (x + 3) 25 ∧ is_divisible (x + 3) 21) ∧
  x = 3147 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l375_37535


namespace NUMINAMATH_CALUDE_exactly_one_multiple_of_five_l375_37522

theorem exactly_one_multiple_of_five (a b : ℤ) (h : 24 * a^2 + 1 = b^2) :
  (a % 5 = 0 ∧ b % 5 ≠ 0) ∨ (a % 5 ≠ 0 ∧ b % 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_multiple_of_five_l375_37522


namespace NUMINAMATH_CALUDE_system_solution_l375_37576

theorem system_solution :
  ∀ x y z : ℝ,
  (x * y + x * z = 8 - x^2) ∧
  (x * y + y * z = 12 - y^2) ∧
  (y * z + z * x = -4 - z^2) →
  ((x = 2 ∧ y = 3 ∧ z = -1) ∨ (x = -2 ∧ y = -3 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l375_37576


namespace NUMINAMATH_CALUDE_unique_solution_equation_l375_37565

theorem unique_solution_equation (x : ℝ) : 
  x > 12 ∧ (x - 5) / 12 = 5 / (x - 12) ↔ x = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l375_37565


namespace NUMINAMATH_CALUDE_winter_sales_is_seven_million_l375_37539

/-- The number of pizzas sold in millions for each season --/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The percentage of pizzas sold in fall --/
def fall_percentage : ℝ := 0.20

/-- The given seasonal sales data --/
def given_sales : SeasonalSales where
  spring := 6
  summer := 7
  fall := fall_percentage * (6 + 7 + fall_percentage * (6 + 7 + 5 + 7) + 7)
  winter := 7

/-- Theorem stating that the winter sales is 7 million pizzas --/
theorem winter_sales_is_seven_million (s : SeasonalSales) :
  s.spring = 6 →
  s.summer = 7 →
  s.fall = fall_percentage * (s.spring + s.summer + s.fall + s.winter) →
  s.winter = 7 := by
  sorry

#eval given_sales.winter

end NUMINAMATH_CALUDE_winter_sales_is_seven_million_l375_37539


namespace NUMINAMATH_CALUDE_sin_arccos_12_13_l375_37544

theorem sin_arccos_12_13 : Real.sin (Real.arccos (12/13)) = 5/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_12_13_l375_37544


namespace NUMINAMATH_CALUDE_max_graduates_few_calls_l375_37548

/-- The number of graduates -/
def total_graduates : ℕ := 100

/-- The number of universities -/
def num_universities : ℕ := 5

/-- The number of graduates each university attempts to contact -/
def contacts_per_university : ℕ := 50

/-- The total number of contact attempts made by all universities -/
def total_contacts : ℕ := num_universities * contacts_per_university

/-- The maximum number of graduates who received at most 2 calls -/
def max_graduates_with_few_calls : ℕ := 83

theorem max_graduates_few_calls :
  ∀ n : ℕ,
  n ≤ total_graduates →
  2 * n + 5 * (total_graduates - n) ≥ total_contacts →
  n ≤ max_graduates_with_few_calls :=
by sorry

end NUMINAMATH_CALUDE_max_graduates_few_calls_l375_37548


namespace NUMINAMATH_CALUDE_continued_fraction_value_l375_37523

theorem continued_fraction_value : 
  ∃ x : ℝ, x = 3 + 4 / (2 + 4 / x) ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l375_37523


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_multiple_l375_37587

def isDivisibleBy (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisible_by_multiple : 
  ∃! n : ℕ, (∀ m : ℕ, m < n → 
    ¬(isDivisibleBy (m - 6) 12 ∧ 
      isDivisibleBy (m - 6) 16 ∧ 
      isDivisibleBy (m - 6) 18 ∧ 
      isDivisibleBy (m - 6) 21 ∧ 
      isDivisibleBy (m - 6) 28)) ∧ 
    isDivisibleBy (n - 6) 12 ∧ 
    isDivisibleBy (n - 6) 16 ∧ 
    isDivisibleBy (n - 6) 18 ∧ 
    isDivisibleBy (n - 6) 21 ∧ 
    isDivisibleBy (n - 6) 28 ∧
    n = 1014 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_multiple_l375_37587


namespace NUMINAMATH_CALUDE_pauls_reading_rate_l375_37537

theorem pauls_reading_rate (total_books : ℕ) (total_weeks : ℕ) 
  (h1 : total_books = 20) (h2 : total_weeks = 5) : 
  total_books / total_weeks = 4 := by
sorry

end NUMINAMATH_CALUDE_pauls_reading_rate_l375_37537


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l375_37506

theorem gain_percent_calculation (C S : ℝ) (h : C > 0) :
  50 * C = 20 * S → (S - C) / C * 100 = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l375_37506


namespace NUMINAMATH_CALUDE_population_average_age_l375_37533

theorem population_average_age 
  (k : ℕ) 
  (h_k_pos : k > 0) 
  (men_count : ℕ := 7 * k)
  (women_count : ℕ := 8 * k)
  (men_avg_age : ℚ := 36)
  (women_avg_age : ℚ := 30) :
  let total_population := men_count + women_count
  let total_age := men_count * men_avg_age + women_count * women_avg_age
  total_age / total_population = 164 / 5 := by
sorry

#eval (164 : ℚ) / 5  -- Should evaluate to 32.8

end NUMINAMATH_CALUDE_population_average_age_l375_37533


namespace NUMINAMATH_CALUDE_xiao_dong_jump_distance_l375_37598

/-- Given a standard jump distance and a recorded result, calculate the actual jump distance. -/
def actual_jump_distance (standard : ℝ) (recorded : ℝ) : ℝ :=
  standard + recorded

/-- Theorem: For a standard jump distance of 4.00 meters and a recorded result of -0.32,
    the actual jump distance is 3.68 meters. -/
theorem xiao_dong_jump_distance :
  let standard : ℝ := 4.00
  let recorded : ℝ := -0.32
  actual_jump_distance standard recorded = 3.68 := by
  sorry

end NUMINAMATH_CALUDE_xiao_dong_jump_distance_l375_37598


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_property_l375_37532

theorem two_digit_number_interchange_property (a b k : ℕ) (h1 : 10 * a + b = 2 * k * (a + b)) :
  10 * b + a - 3 * (a + b) = (9 - 4 * k) * (a + b) := by sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_property_l375_37532


namespace NUMINAMATH_CALUDE_whale_length_from_relative_speed_l375_37571

/-- The length of a whale can be determined by the relative speed of two whales
    and the time taken for one to cross the other. -/
theorem whale_length_from_relative_speed (v_fast v_slow t : ℝ) (h1 : v_fast > v_slow) :
  (v_fast - v_slow) * t = (v_fast - v_slow) * 15 → v_fast = 18 → v_slow = 15 → (v_fast - v_slow) * 15 = 45 := by
  sorry

#check whale_length_from_relative_speed

end NUMINAMATH_CALUDE_whale_length_from_relative_speed_l375_37571


namespace NUMINAMATH_CALUDE_prob_at_least_two_correct_l375_37570

def num_questions : ℕ := 30
def num_guessed : ℕ := 5
def num_choices : ℕ := 6

def prob_correct : ℚ := 1 / num_choices
def prob_incorrect : ℚ := 1 - prob_correct

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem prob_at_least_two_correct :
  (1 : ℚ) - (binomial num_guessed 0 : ℚ) * prob_incorrect ^ num_guessed
          - (binomial num_guessed 1 : ℚ) * prob_correct * prob_incorrect ^ (num_guessed - 1)
  = 1526 / 7776 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_correct_l375_37570


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_ratio_l375_37568

noncomputable def f₁ (a : ℝ) (x : ℝ) : ℝ := x^2 - x + 2*a
noncomputable def f₂ (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + 3
noncomputable def f₃ (a b : ℝ) (x : ℝ) : ℝ := 4*x^2 + (2*b-3)*x + 6*a + 3
noncomputable def f₄ (a b : ℝ) (x : ℝ) : ℝ := 4*x^2 + (6*b-1)*x + 9 + 2*a

noncomputable def A (a : ℝ) : ℝ := Real.sqrt (1 - 8*a)
noncomputable def B (b : ℝ) : ℝ := Real.sqrt (4*b^2 - 12)
noncomputable def C (a b : ℝ) : ℝ := (1/4) * Real.sqrt ((2*b - 3)^2 - 64*(6*a + 3))
noncomputable def D (a b : ℝ) : ℝ := (1/4) * Real.sqrt ((6*b - 1)^2 - 64*(9 + 2*a))

theorem quadratic_roots_difference_ratio (a b : ℝ) (h : A a ≠ B b) :
  (C a b^2 - D a b^2) / (A a^2 - B b^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_ratio_l375_37568


namespace NUMINAMATH_CALUDE_ines_peaches_bought_l375_37558

def peaches_bought (initial_amount : ℕ) (remaining_amount : ℕ) (price_per_pound : ℕ) : ℕ :=
  (initial_amount - remaining_amount) / price_per_pound

theorem ines_peaches_bought :
  peaches_bought 20 14 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ines_peaches_bought_l375_37558


namespace NUMINAMATH_CALUDE_mod_9_sum_of_digits_mod_9_sum_mod_9_product_l375_37574

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Property 1
theorem mod_9_sum_of_digits (n : ℕ) : n % 9 = sumOfDigits n % 9 := by
  sorry

-- Property 2
theorem mod_9_sum (ns : List ℕ) : 
  (ns.sum % 9) = (ns.map (· % 9)).sum % 9 := by
  sorry

-- Property 3
theorem mod_9_product (ns : List ℕ) : 
  (ns.prod % 9) = (ns.map (· % 9)).prod % 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_9_sum_of_digits_mod_9_sum_mod_9_product_l375_37574


namespace NUMINAMATH_CALUDE_seashell_collection_l375_37504

/-- Calculates the remaining number of seashells after Leo gives away a quarter of his collection -/
def remaining_seashells (henry_shells : ℕ) (paul_shells : ℕ) (total_shells : ℕ) : ℕ :=
  let leo_shells := total_shells - henry_shells - paul_shells
  let leo_gave_away := leo_shells / 4
  total_shells - leo_gave_away

theorem seashell_collection (henry_shells paul_shells total_shells : ℕ) 
  (h1 : henry_shells = 11)
  (h2 : paul_shells = 24)
  (h3 : total_shells = 59) :
  remaining_seashells henry_shells paul_shells total_shells = 53 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l375_37504


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l375_37557

theorem sum_of_fractions_equals_one
  (a b c d w x y z : ℝ)
  (eq1 : 17*w + b*x + c*y + d*z = 0)
  (eq2 : a*w + 29*x + c*y + d*z = 0)
  (eq3 : a*w + b*x + 37*y + d*z = 0)
  (eq4 : a*w + b*x + c*y + 53*z = 0)
  (ha : a ≠ 17)
  (hb : b ≠ 29)
  (hc : c ≠ 37)
  (h_not_all_zero : ¬(w = 0 ∧ x = 0 ∧ y = 0)) :
  a / (a - 17) + b / (b - 29) + c / (c - 37) + d / (d - 53) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l375_37557


namespace NUMINAMATH_CALUDE_range_of_negative_values_l375_37586

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_decreasing : ∀ x y, x < y → y < 0 → f x > f y)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l375_37586


namespace NUMINAMATH_CALUDE_product_of_radicals_l375_37583

theorem product_of_radicals (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q^3) = 14 * q^2 * Real.sqrt (42 * q) :=
by sorry

end NUMINAMATH_CALUDE_product_of_radicals_l375_37583


namespace NUMINAMATH_CALUDE_car_speed_problem_l375_37566

/-- The speed of Car B in km/h -/
def speed_B : ℝ := 35

/-- The time it takes Car A to catch up with Car B when traveling at 50 km/h -/
def time_1 : ℝ := 6

/-- The time it takes Car A to catch up with Car B when traveling at 80 km/h -/
def time_2 : ℝ := 2

/-- The speed of Car A in the first scenario (km/h) -/
def speed_A_1 : ℝ := 50

/-- The speed of Car A in the second scenario (km/h) -/
def speed_A_2 : ℝ := 80

theorem car_speed_problem :
  speed_B * time_1 = speed_A_1 * time_1 - (time_1 - time_2) * speed_B ∧
  speed_B * time_2 = speed_A_2 * time_2 - (time_1 - time_2) * speed_B :=
by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l375_37566


namespace NUMINAMATH_CALUDE_distance_traveled_l375_37572

/-- Given a car's fuel efficiency and the amount of fuel used, calculate the distance traveled. -/
theorem distance_traveled (efficiency : ℝ) (fuel_used : ℝ) (h1 : efficiency = 20) (h2 : fuel_used = 3) :
  efficiency * fuel_used = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l375_37572


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l375_37530

theorem right_triangle_hypotenuse (longer_leg shorter_leg hypotenuse : ℝ) : 
  shorter_leg = longer_leg - 3 →
  (1 / 2) * longer_leg * shorter_leg = 120 →
  longer_leg > 0 →
  shorter_leg > 0 →
  hypotenuse^2 = longer_leg^2 + shorter_leg^2 →
  hypotenuse = Real.sqrt 425 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l375_37530


namespace NUMINAMATH_CALUDE_range_of_g_l375_37521

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 5

-- Define the function g as a composition of f
def g (x : ℝ) : ℝ := f (f (f x))

-- Theorem statement
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ g x = y) ↔ -41 ≤ y ∧ y ≤ 87 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l375_37521


namespace NUMINAMATH_CALUDE_sum_zero_fraction_l375_37517

theorem sum_zero_fraction (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (h_sum : x + y + z = 0) : 
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_fraction_l375_37517


namespace NUMINAMATH_CALUDE_union_of_sets_l375_37577

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | 3 * x^2 + p * x - 7 = 0}
def B (q : ℝ) : Set ℝ := {x | 3 * x^2 - 7 * x + q = 0}

-- State the theorem
theorem union_of_sets (p q : ℝ) :
  (∃ (p q : ℝ), A p ∩ B q = {-1/3}) →
  (∃ (p q : ℝ), A p ∪ B q = {-1/3, 8/3, 7}) :=
by sorry

end NUMINAMATH_CALUDE_union_of_sets_l375_37577


namespace NUMINAMATH_CALUDE_investment_interest_proof_l375_37590

/-- Calculate compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Calculate total interest earned --/
def total_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  compound_interest principal rate years - principal

theorem investment_interest_proof :
  let principal := 1500
  let rate := 0.08
  let years := 5
  ∃ ε > 0, abs (total_interest principal rate years - 704) < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l375_37590


namespace NUMINAMATH_CALUDE_hannah_lost_eight_pieces_l375_37553

-- Define the initial state of the chess game
def initial_pieces : ℕ := 32
def initial_pieces_per_player : ℕ := 16

-- Define the given conditions
def scarlett_lost : ℕ := 6
def total_pieces_left : ℕ := 18

-- Define Hannah's lost pieces
def hannah_lost : ℕ := initial_pieces_per_player - (total_pieces_left - (initial_pieces_per_player - scarlett_lost))

-- Theorem to prove
theorem hannah_lost_eight_pieces : hannah_lost = 8 := by
  sorry

end NUMINAMATH_CALUDE_hannah_lost_eight_pieces_l375_37553


namespace NUMINAMATH_CALUDE_max_value_theorem_l375_37531

theorem max_value_theorem (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1)
  (hax : a^x = 2) (hby : b^y = 2) (hab : a + Real.sqrt b = 4) :
  ∃ (M : ℝ), M = 4 ∧ ∀ (z : ℝ), (2/x + 1/y) ≤ z → z ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l375_37531


namespace NUMINAMATH_CALUDE_trajectory_of_product_slopes_l375_37585

/-- The trajectory of a moving point P whose product of slopes to fixed points A(-1,0) and B(1,0) is -1 -/
theorem trajectory_of_product_slopes (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -1 → x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_product_slopes_l375_37585


namespace NUMINAMATH_CALUDE_square_side_length_l375_37543

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * Real.sqrt 2 = diagonal ∧ side = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l375_37543


namespace NUMINAMATH_CALUDE_inequality_proof_l375_37534

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l375_37534


namespace NUMINAMATH_CALUDE_union_equals_A_l375_37513

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equals_A (m : ℝ) : 
  (A m ∪ B m = A m) → (m = 0 ∨ m = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l375_37513


namespace NUMINAMATH_CALUDE_largest_value_l375_37599

theorem largest_value (a b c d : ℝ) 
  (h : a + 1 = b - 2 ∧ a + 1 = c + 3 ∧ a + 1 = d - 4) : 
  d = max a (max b (max c d)) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l375_37599


namespace NUMINAMATH_CALUDE_power_of_power_l375_37529

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l375_37529


namespace NUMINAMATH_CALUDE_remainder_theorem_l375_37569

/-- The polynomial for which we want to find the remainder -/
def f (x : ℝ) : ℝ := 5*x^5 - 8*x^4 + 3*x^3 - x^2 + 4*x - 15

/-- The theorem stating that the remainder of f(x) divided by (x - 2) is 45 -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 45 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l375_37569


namespace NUMINAMATH_CALUDE_cosine_value_l375_37581

theorem cosine_value (α : Real) 
  (h : Real.cos (α - π/6) - Real.sin α = 2 * Real.sqrt 3 / 5) : 
  Real.cos (α + 7*π/6) = -(2 * Real.sqrt 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_l375_37581


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_l375_37527

/-- The cost of paint per kg for a cube painting problem -/
theorem paint_cost_per_kg (coverage : ℝ) (total_cost : ℝ) (side_length : ℝ) : 
  coverage = 20 →
  total_cost = 10800 →
  side_length = 30 →
  (total_cost / (6 * side_length^2 / coverage)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_l375_37527


namespace NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l375_37502

/-- The distance Arthur walked in miles -/
def arthur_distance (blocks_west blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 4.5 miles -/
theorem arthur_walked_four_point_five_miles :
  arthur_distance 8 10 (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l375_37502


namespace NUMINAMATH_CALUDE_original_group_size_l375_37508

-- Define the work completion rate for a group
def work_rate (num_men : ℕ) (days : ℕ) : ℚ := 1 / (num_men * days)

-- Define the theorem
theorem original_group_size :
  ∃ (x : ℕ),
    -- Condition 1: Original group completes work in 20 days
    work_rate x 20 =
    -- Condition 2 & 3: Remaining group (x - 10) completes work in 40 days
    work_rate (x - 10) 40 ∧
    -- Answer: The original group size is 20
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l375_37508


namespace NUMINAMATH_CALUDE_factor_expression_l375_37538

theorem factor_expression (y : ℝ) : 64 - 16 * y^3 = 16 * (2 - y) * (4 + 2*y + y^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l375_37538


namespace NUMINAMATH_CALUDE_number_division_l375_37573

theorem number_division (x : ℝ) : x + 8 = 88 → x / 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l375_37573


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l375_37542

def grape_purchase : ℕ := 8
def mango_purchase : ℕ := 10
def mango_rate : ℕ := 55
def total_paid : ℕ := 1110

theorem grape_rate_calculation :
  ∃ (grape_rate : ℕ),
    grape_rate * grape_purchase + mango_rate * mango_purchase = total_paid ∧
    grape_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l375_37542


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l375_37597

theorem complex_number_quadrant : 
  let z : ℂ := (Complex.I) / (2 * Complex.I - 1)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l375_37597


namespace NUMINAMATH_CALUDE_tangent_line_slope_l375_37575

/-- Given a function f(x) = x^3 + ax^2 + x with a tangent line at (1, f(1)) having slope 6, 
    prove that a = 1. -/
theorem tangent_line_slope (a : ℝ) : 
  let f := λ x : ℝ => x^3 + a*x^2 + x
  let f' := λ x : ℝ => 3*x^2 + 2*a*x + 1
  f' 1 = 6 → a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l375_37575


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_two_l375_37589

/-- 
Given a triangle ABC where:
- The sides opposite to angles A, B, C are a, b, c respectively
- b = √5
- c = 2
- cos B = 2/3

Prove that the measure of angle A is π/2
-/
theorem angle_A_is_pi_over_two (a b c : ℝ) (A B C : ℝ) : 
  b = Real.sqrt 5 → 
  c = 2 → 
  Real.cos B = 2/3 → 
  A + B + C = π → 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  a * Real.sin B = b * Real.sin A → 
  b^2 = a^2 + c^2 - 2*a*c * Real.cos B → 
  A = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_two_l375_37589


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l375_37551

-- Problem 1
theorem problem_one : Real.sqrt 3 - Real.sqrt 3 * (1 - Real.sqrt 3) = 3 := by sorry

-- Problem 2
theorem problem_two : (Real.sqrt 3 - 2)^2 + Real.sqrt 12 + 6 * Real.sqrt (1/3) = 7 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l375_37551


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l375_37594

-- Define the polynomial
def f (x : ℝ) := 3 * x^3 - 3 * x^2 - 3 * x - 9

-- Define p, q, r as roots of the polynomial
theorem roots_of_polynomial (p q r : ℝ) : f p = 0 ∧ f q = 0 ∧ f r = 0 → p^2 + q^2 + r^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l375_37594


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l375_37511

def population : ℕ := 1203
def sample_size : ℕ := 40

theorem systematic_sampling_interval :
  ∃ (k : ℕ) (eliminated : ℕ),
    eliminated ≤ sample_size ∧
    (population - eliminated) % sample_size = 0 ∧
    k = (population - eliminated) / sample_size ∧
    k = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l375_37511


namespace NUMINAMATH_CALUDE_circle_covering_theorem_l375_37500

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a function to check if a point is inside or on a circle
def pointInCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

-- Main theorem
theorem circle_covering_theorem 
  (points : Finset Point) 
  (outer_circle : Circle) 
  (h1 : outer_circle.radius = 2)
  (h2 : points.card = 15)
  (h3 : ∀ p ∈ points, pointInCircle p outer_circle) :
  ∃ (inner_circle : Circle), 
    inner_circle.radius = 1 ∧ 
    (∃ (subset : Finset Point), subset ⊆ points ∧ subset.card ≥ 3 ∧ 
      ∀ p ∈ subset, pointInCircle p inner_circle) := by
  sorry

end NUMINAMATH_CALUDE_circle_covering_theorem_l375_37500


namespace NUMINAMATH_CALUDE_amy_music_files_l375_37546

/-- Represents the number of music files Amy initially had -/
def initial_music_files : ℕ := sorry

/-- Represents the initial total number of files -/
def initial_total_files : ℕ := initial_music_files + 36

/-- Represents the number of deleted files -/
def deleted_files : ℕ := 48

/-- Represents the number of remaining files after deletion -/
def remaining_files : ℕ := 14

theorem amy_music_files :
  initial_total_files - deleted_files = remaining_files ∧
  initial_music_files = 26 := by
  sorry

end NUMINAMATH_CALUDE_amy_music_files_l375_37546


namespace NUMINAMATH_CALUDE_cos_negative_420_degrees_l375_37595

theorem cos_negative_420_degrees : Real.cos (-(420 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_420_degrees_l375_37595


namespace NUMINAMATH_CALUDE_equality_condition_l375_37559

theorem equality_condition (a b c : ℝ) :
  2 * a + 3 * b * c = (a + 2 * b) * (2 * a + 3 * c) ↔ a = 0 ∨ a + 2 * b + 1.5 * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l375_37559


namespace NUMINAMATH_CALUDE_complex_equation_solution_l375_37561

theorem complex_equation_solution (z : ℂ) :
  (1 + 2*I) * z = 4 + 3*I → z = 2 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l375_37561


namespace NUMINAMATH_CALUDE_geometric_sequence_308th_term_l375_37591

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_308th_term :
  let a₁ := 12
  let a₂ := -24
  let r := a₂ / a₁
  geometric_sequence a₁ r 308 = -2^307 * 12 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_308th_term_l375_37591


namespace NUMINAMATH_CALUDE_max_distinct_pairs_l375_37552

theorem max_distinct_pairs (k : ℕ) 
  (a b : Fin k → ℕ)
  (h_range : ∀ i : Fin k, 1 ≤ a i ∧ a i < b i ∧ b i ≤ 150)
  (h_distinct : ∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)
  (h_sum_distinct : ∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j)
  (h_sum_bound : ∀ i : Fin k, a i + b i ≤ 150) :
  k ≤ 59 :=
sorry

end NUMINAMATH_CALUDE_max_distinct_pairs_l375_37552


namespace NUMINAMATH_CALUDE_book_e_chapters_l375_37541

/-- Represents the number of chapters in each book --/
structure BookChapters where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The problem statement --/
def book_chapters_problem (total : ℕ) (books : BookChapters) : Prop :=
  books.a = 17 ∧
  books.b = books.a + 5 ∧
  books.c = books.b - 7 ∧
  books.d = 2 * books.c ∧
  total = 97 ∧
  total = books.a + books.b + books.c + books.d + books.e

/-- The theorem to prove --/
theorem book_e_chapters (total : ℕ) (books : BookChapters) 
  (h : book_chapters_problem total books) : books.e = 13 := by
  sorry


end NUMINAMATH_CALUDE_book_e_chapters_l375_37541


namespace NUMINAMATH_CALUDE_game_savings_ratio_l375_37567

theorem game_savings_ratio (game_cost : ℝ) (tax_rate : ℝ) (weekly_allowance : ℝ) (weeks_to_save : ℕ) :
  game_cost = 50 →
  tax_rate = 0.1 →
  weekly_allowance = 10 →
  weeks_to_save = 11 →
  (weekly_allowance / weekly_allowance : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_game_savings_ratio_l375_37567


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l375_37592

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l375_37592


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l375_37580

theorem multiply_and_add_equality : 45 * 56 + 54 * 45 = 4950 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l375_37580


namespace NUMINAMATH_CALUDE_binomial_coefficient_10_3_l375_37545

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_10_3_l375_37545


namespace NUMINAMATH_CALUDE_inequality_problem_l375_37505

theorem inequality_problem (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : m > 0)
  (h4 : ∀ a b c, a > b → b > c → (1 / (a - b) + m / (b - c) ≥ 9 / (a - c))) :
  m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l375_37505


namespace NUMINAMATH_CALUDE_tadpoles_let_go_75_percent_l375_37515

/-- The percentage of tadpoles let go, given the total caught and number kept -/
def tadpoles_let_go_percentage (total : ℕ) (kept : ℕ) : ℚ :=
  (total - kept : ℚ) / total * 100

/-- Theorem stating that the percentage of tadpoles let go is 75% -/
theorem tadpoles_let_go_75_percent (total : ℕ) (kept : ℕ) 
  (h1 : total = 180) (h2 : kept = 45) : 
  tadpoles_let_go_percentage total kept = 75 := by
  sorry

#eval tadpoles_let_go_percentage 180 45

end NUMINAMATH_CALUDE_tadpoles_let_go_75_percent_l375_37515


namespace NUMINAMATH_CALUDE_hyperbola_C_properties_l375_37556

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2/3 - y^2/12 = 1

-- Define the reference hyperbola
def ref_hyperbola (x y : ℝ) : Prop := y^2/4 - x^2 = 1

-- Theorem statement
theorem hyperbola_C_properties :
  -- C passes through (2,2)
  C 2 2 ∧
  -- C has the same asymptotes as the reference hyperbola
  (∀ x y : ℝ, C x y ↔ ∃ k : ℝ, k ≠ 0 ∧ ref_hyperbola (x/k) (y/k)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_C_properties_l375_37556


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l375_37526

theorem sufficient_condition_for_inequality (a : ℝ) (h : a ≥ 5) :
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l375_37526


namespace NUMINAMATH_CALUDE_jimmy_can_lose_five_more_points_l375_37536

def passing_score : ℕ := 50
def exams_count : ℕ := 3
def points_per_exam : ℕ := 20
def points_lost : ℕ := 5

def max_additional_points_to_lose : ℕ :=
  exams_count * points_per_exam - points_lost - passing_score

theorem jimmy_can_lose_five_more_points :
  max_additional_points_to_lose = 5 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_can_lose_five_more_points_l375_37536


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l375_37596

theorem max_stamps_purchasable (stamp_price : ℕ) (discounted_price : ℕ) (budget : ℕ) :
  stamp_price = 50 →
  discounted_price = 45 →
  budget = 5000 →
  (∀ n : ℕ, n ≤ 50 → n * stamp_price ≤ budget) →
  (∀ n : ℕ, n > 50 → 50 * stamp_price + (n - 50) * discounted_price ≤ budget) →
  (∃ n : ℕ, n = 105 ∧
    (∀ m : ℕ, m > n → 
      (m ≤ 50 → m * stamp_price > budget) ∧
      (m > 50 → 50 * stamp_price + (m - 50) * discounted_price > budget))) :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l375_37596


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l375_37514

/-- Proves that a faster speed allowing 20 km more distance in the same time as 50 km at 10 km/hr is 14 km/hr -/
theorem faster_speed_calculation (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ) :
  actual_distance = 50 →
  actual_speed = 10 →
  additional_distance = 20 →
  ∃ (faster_speed : ℝ),
    (actual_distance / actual_speed = (actual_distance + additional_distance) / faster_speed) ∧
    faster_speed = 14 :=
by sorry

end NUMINAMATH_CALUDE_faster_speed_calculation_l375_37514


namespace NUMINAMATH_CALUDE_first_girl_siblings_l375_37593

-- Define the number of girls in the survey
def num_girls : ℕ := 9

-- Define the mean number of siblings
def mean_siblings : ℚ := 5.7

-- Define the list of known sibling counts
def known_siblings : List ℕ := [6, 10, 4, 3, 3, 11, 3, 10]

-- Define the sum of known sibling counts
def sum_known_siblings : ℕ := known_siblings.sum

-- Theorem to prove
theorem first_girl_siblings :
  ∃ (x : ℕ), x + sum_known_siblings = Int.floor (mean_siblings * num_girls) ∧ x = 1 := by
  sorry


end NUMINAMATH_CALUDE_first_girl_siblings_l375_37593


namespace NUMINAMATH_CALUDE_treadmill_price_correct_l375_37501

/-- The price of the treadmill at Toby's garage sale. -/
def treadmill_price : ℝ := 133.33

/-- The total sum of money Toby made at the garage sale. -/
def total_money : ℝ := 600

/-- Theorem stating that the treadmill price is correct given the conditions of the garage sale. -/
theorem treadmill_price_correct : 
  treadmill_price + 0.5 * treadmill_price + 3 * treadmill_price = total_money :=
by sorry

end NUMINAMATH_CALUDE_treadmill_price_correct_l375_37501
