import Mathlib

namespace NUMINAMATH_CALUDE_barney_sit_ups_time_l3028_302815

/-- Proves that Barney did sit-ups for 1 minute given the problem conditions --/
theorem barney_sit_ups_time (barney_rate : ℕ) (carrie_time jerrie_time : ℕ) (total_sit_ups : ℕ)
  (h1 : barney_rate = 45)
  (h2 : carrie_time = 2)
  (h3 : jerrie_time = 3)
  (h4 : total_sit_ups = 510) :
  ∃ (barney_time : ℕ),
    barney_time * barney_rate +
    carrie_time * (2 * barney_rate) +
    jerrie_time * (2 * barney_rate + 5) = total_sit_ups ∧
    barney_time = 1 := by
  sorry

end NUMINAMATH_CALUDE_barney_sit_ups_time_l3028_302815


namespace NUMINAMATH_CALUDE_circle_triangle_perimeter_l3028_302805

structure Circle :=
  (points : Fin 6 → ℝ × ℝ)

structure Triangle :=
  (vertices : Fin 3 → ℝ × ℝ)

def perimeter (t : Triangle) : ℝ := sorry

theorem circle_triangle_perimeter
  (c : Circle)
  (x y z : ℝ × ℝ)
  (h1 : x ∈ Set.Icc (c.points 0) (c.points 3) ∩ Set.Icc (c.points 1) (c.points 4))
  (h2 : y ∈ Set.Icc (c.points 0) (c.points 3) ∩ Set.Icc (c.points 2) (c.points 5))
  (h3 : z ∈ Set.Icc (c.points 2) (c.points 5) ∩ Set.Icc (c.points 1) (c.points 4))
  (h4 : x ∈ Set.Icc z (c.points 1))
  (h5 : x ∈ Set.Icc y (c.points 0))
  (h6 : y ∈ Set.Icc z (c.points 2))
  (h7 : dist (c.points 0) x = 3)
  (h8 : dist (c.points 1) x = 2)
  (h9 : dist (c.points 2) y = 4)
  (h10 : dist (c.points 3) y = 10)
  (h11 : dist (c.points 4) z = 16)
  (h12 : dist (c.points 5) z = 12)
  : perimeter { vertices := ![x, y, z] } = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_perimeter_l3028_302805


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l3028_302809

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l3028_302809


namespace NUMINAMATH_CALUDE_expand_and_compare_l3028_302869

theorem expand_and_compare (m n : ℝ) :
  (∀ x : ℝ, (x + 2) * (x + 3) = x^2 + m*x + n) → m = 5 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_compare_l3028_302869


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l3028_302831

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem third_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 8) :
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l3028_302831


namespace NUMINAMATH_CALUDE_election_majority_l3028_302806

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 1400 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).floor - 
  ((1 - winning_percentage) * total_votes : ℚ).floor = 280 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_l3028_302806


namespace NUMINAMATH_CALUDE_target_seat_representation_l3028_302880

/-- Represents a seat in a cinema -/
structure CinemaSeat where
  row : Nat
  seatNumber : Nat

/-- Given representation for seat number 4 in row 6 -/
def givenSeat : CinemaSeat := ⟨6, 4⟩

/-- The seat we want to represent (seat number 1 in row 5) -/
def targetSeat : CinemaSeat := ⟨5, 1⟩

/-- Theorem stating that the target seat is correctly represented -/
theorem target_seat_representation : targetSeat = ⟨5, 1⟩ := by
  sorry

end NUMINAMATH_CALUDE_target_seat_representation_l3028_302880


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_point_satisfies_equation_l3028_302800

/-- Proves that the equation of a line passing through the point (1, 2) with a slope of 3 is y = 3x - 1 -/
theorem line_equation_through_point_with_slope (x y : ℝ) : 
  (y - 2 = 3 * (x - 1)) ↔ (y = 3 * x - 1) := by
  sorry

/-- Verifies that the point (1, 2) satisfies the equation y = 3x - 1 -/
theorem point_satisfies_equation : 
  2 = 3 * 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_point_satisfies_equation_l3028_302800


namespace NUMINAMATH_CALUDE_sphere_surface_area_doubling_l3028_302820

theorem sphere_surface_area_doubling (r : ℝ) :
  (4 * Real.pi * r^2 = 2464) →
  (4 * Real.pi * (2*r)^2 = 39376) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_doubling_l3028_302820


namespace NUMINAMATH_CALUDE_sally_fries_theorem_l3028_302868

def sally_fries_problem (sally_initial : ℕ) (mark_total : ℕ) (jessica_total : ℕ) : Prop :=
  let mark_share := mark_total / 3
  let jessica_share := jessica_total / 2
  sally_initial + mark_share + jessica_share = 38

theorem sally_fries_theorem :
  sally_fries_problem 14 36 24 :=
by
  sorry

end NUMINAMATH_CALUDE_sally_fries_theorem_l3028_302868


namespace NUMINAMATH_CALUDE_asymptote_sum_l3028_302844

/-- Given a rational function y = x / (x^3 + Ax^2 + Bx + C) where A, B, C are integers,
    if the graph has vertical asymptotes at x = -3, 0, 3, then A + B + C = -9 -/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    ∃ y : ℝ, y = x / (x^3 + A*x^2 + B*x + C)) →
  A + B + C = -9 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l3028_302844


namespace NUMINAMATH_CALUDE_set_element_value_l3028_302874

theorem set_element_value (a : ℝ) : 2 ∈ ({0, a, a^2 - 3*a + 2} : Set ℝ) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_element_value_l3028_302874


namespace NUMINAMATH_CALUDE_roses_cut_correct_l3028_302857

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

theorem roses_cut_correct (initial_roses final_roses : ℕ) 
  (h : initial_roses ≤ final_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 6 16  -- Should output 10

end NUMINAMATH_CALUDE_roses_cut_correct_l3028_302857


namespace NUMINAMATH_CALUDE_amber_max_ounces_l3028_302892

def amber_money : ℚ := 7
def candy_price : ℚ := 1
def candy_ounces : ℚ := 12
def chips_price : ℚ := 1.4
def chips_ounces : ℚ := 17

def candy_bags : ℚ := amber_money / candy_price
def chips_bags : ℚ := amber_money / chips_price

def total_candy_ounces : ℚ := candy_bags * candy_ounces
def total_chips_ounces : ℚ := chips_bags * chips_ounces

theorem amber_max_ounces :
  max total_candy_ounces total_chips_ounces = 85 :=
by sorry

end NUMINAMATH_CALUDE_amber_max_ounces_l3028_302892


namespace NUMINAMATH_CALUDE_zoo_count_l3028_302841

theorem zoo_count (total_heads : ℕ) (total_legs : ℕ) : 
  total_heads = 300 → 
  total_legs = 710 → 
  ∃ (birds mammals unique : ℕ), 
    birds + mammals + unique = total_heads ∧
    2 * birds + 4 * mammals + 3 * unique = total_legs ∧
    birds = 230 := by
  sorry

end NUMINAMATH_CALUDE_zoo_count_l3028_302841


namespace NUMINAMATH_CALUDE_sum_diff_difference_is_six_l3028_302839

/-- A two-digit number with specific properties -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_two_digit : 10 ≤ 10 * tens + ones ∧ 10 * tens + ones < 100
  digit_ratio : ones = 2 * tens
  interchange_diff : 10 * ones + tens - (10 * tens + ones) = 36

/-- The difference between the sum and difference of digits for a TwoDigitNumber -/
def sum_diff_difference (n : TwoDigitNumber) : Nat :=
  (n.tens + n.ones) - (n.ones - n.tens)

theorem sum_diff_difference_is_six (n : TwoDigitNumber) :
  sum_diff_difference n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_diff_difference_is_six_l3028_302839


namespace NUMINAMATH_CALUDE_kaydence_sister_age_l3028_302850

/-- The ages of Kaydence's family members -/
structure FamilyAges where
  total : ℕ
  father : ℕ
  mother : ℕ
  brother : ℕ
  kaydence : ℕ
  sister : ℕ

/-- The conditions given in the problem -/
def family_conditions (ages : FamilyAges) : Prop :=
  ages.total = 200 ∧
  ages.father = 60 ∧
  ages.mother = ages.father - 2 ∧
  ages.brother = ages.father / 2 ∧
  ages.kaydence = 12

/-- Theorem stating that given the conditions, Kaydence's sister is 40 years old -/
theorem kaydence_sister_age (ages : FamilyAges) :
  family_conditions ages → ages.sister = 40 :=
by sorry

end NUMINAMATH_CALUDE_kaydence_sister_age_l3028_302850


namespace NUMINAMATH_CALUDE_a_gt_b_neither_sufficient_nor_necessary_for_abs_a_gt_abs_b_l3028_302873

theorem a_gt_b_neither_sufficient_nor_necessary_for_abs_a_gt_abs_b :
  ¬(∀ a b : ℝ, a > b → |a| > |b|) ∧ ¬(∀ a b : ℝ, |a| > |b| → a > b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_neither_sufficient_nor_necessary_for_abs_a_gt_abs_b_l3028_302873


namespace NUMINAMATH_CALUDE_total_handshakes_l3028_302829

/-- The number of teams in the tournament -/
def num_teams : ℕ := 3

/-- The number of players in each team -/
def players_per_team : ℕ := 4

/-- The number of referees -/
def num_referees : ℕ := 3

/-- The number of coaches -/
def num_coaches : ℕ := 1

/-- The total number of players -/
def total_players : ℕ := num_teams * players_per_team

/-- The number of officials (referees + coaches) -/
def num_officials : ℕ := num_referees + num_coaches

/-- Theorem stating the total number of handshakes in the tournament -/
theorem total_handshakes : 
  (num_teams * players_per_team * (num_teams - 1) * players_per_team) / 2 + 
  (total_players * num_officials) = 144 := by
  sorry

#eval (num_teams * players_per_team * (num_teams - 1) * players_per_team) / 2 + 
      (total_players * num_officials)

end NUMINAMATH_CALUDE_total_handshakes_l3028_302829


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3028_302819

-- Define the sections of the trip
def section1_distance : ℝ := 600
def section1_speed : ℝ := 30
def section2_distance : ℝ := 300
def section2_speed : ℝ := 15
def section3_distance : ℝ := 500
def section3_speed : ℝ := 25
def section4_distance : ℝ := 400
def section4_speed : ℝ := 40

-- Define the total distance
def total_distance : ℝ := section1_distance + section2_distance + section3_distance + section4_distance

-- Theorem statement
theorem average_speed_calculation :
  let time1 := section1_distance / section1_speed
  let time2 := section2_distance / section2_speed
  let time3 := section3_distance / section3_speed
  let time4 := section4_distance / section4_speed
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  abs (average_speed - 25.71) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3028_302819


namespace NUMINAMATH_CALUDE_triangle_area_ordering_l3028_302866

/-- The area of the first triangle -/
def m : ℚ := 15/2

/-- The area of the second triangle -/
def n : ℚ := 13/2

/-- The area of the third triangle -/
def p : ℚ := 7

/-- The side length of the square -/
def square_side : ℚ := 4

/-- The area of the square -/
def square_area : ℚ := square_side * square_side

/-- Theorem stating that the areas of the triangles satisfy n < p < m -/
theorem triangle_area_ordering : n < p ∧ p < m := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ordering_l3028_302866


namespace NUMINAMATH_CALUDE_students_making_stars_l3028_302813

theorem students_making_stars (stars_per_student : ℕ) (total_stars : ℕ) (h1 : stars_per_student = 3) (h2 : total_stars = 372) :
  total_stars / stars_per_student = 124 := by
  sorry

end NUMINAMATH_CALUDE_students_making_stars_l3028_302813


namespace NUMINAMATH_CALUDE_circle_bisection_and_symmetric_points_l3028_302804

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k*x - 1

-- Define symmetry with respect to a line
def symmetric_wrt_line (x1 y1 x2 y2 k : ℝ) : Prop :=
  (x1 + x2) * (k + 1/k) = (y1 + y2) * (1 - 1/k)

-- Define perpendicularity
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_bisection_and_symmetric_points :
  -- Part 1: The line y = -x - 1 bisects the circle
  (∀ x y : ℝ, circle_C x y → line_l x y (-1)) ∧
  -- Part 2: There exist points A and B on the circle satisfying the conditions
  ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    symmetric_wrt_line x1 y1 x2 y2 (-1) ∧
    perpendicular x1 y1 x2 y2 ∧
    ((x1 - y1 + 1 = 0 ∧ x2 - y2 + 1 = 0) ∨ (x1 - y1 - 4 = 0 ∧ x2 - y2 - 4 = 0)) :=
sorry

end NUMINAMATH_CALUDE_circle_bisection_and_symmetric_points_l3028_302804


namespace NUMINAMATH_CALUDE_sum_of_amp_operations_l3028_302832

-- Define the operation &
def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

-- Theorem statement
theorem sum_of_amp_operations : amp 12 5 + amp 8 3 = 174 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_amp_operations_l3028_302832


namespace NUMINAMATH_CALUDE_divisibility_by_three_l3028_302877

theorem divisibility_by_three (n : ℕ+) : 
  (∃ k : ℤ, n = 6*k + 1 ∨ n = 6*k + 2) ↔ 
  (∃ m : ℤ, n * 2^(n : ℕ) + 1 = 3 * m) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l3028_302877


namespace NUMINAMATH_CALUDE_implicit_second_derivative_l3028_302867

noncomputable def y (x : ℝ) : ℝ := Real.exp x

theorem implicit_second_derivative 
  (h : ∀ x, y x * Real.exp x + Real.exp (y x) = Real.exp 1 + 1) :
  ∀ x, deriv (deriv y) x = 
    (-2 * Real.exp (2*x) * y x * (Real.exp x + Real.exp (y x)) + 
     y x * Real.exp x * (Real.exp x + Real.exp (y x))^2 + 
     (y x)^2 * Real.exp (y x) * Real.exp (2*x)) / 
    (Real.exp x + Real.exp (y x))^3 :=
by
  sorry

end NUMINAMATH_CALUDE_implicit_second_derivative_l3028_302867


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3028_302827

theorem sqrt_50_between_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3028_302827


namespace NUMINAMATH_CALUDE_tiffany_cans_problem_l3028_302837

theorem tiffany_cans_problem (monday_bags : ℕ) (next_day_bags : ℕ) : 
  (monday_bags = next_day_bags + 1) → (next_day_bags = 7) → (monday_bags = 8) :=
by sorry

end NUMINAMATH_CALUDE_tiffany_cans_problem_l3028_302837


namespace NUMINAMATH_CALUDE_sum_of_squares_l3028_302812

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 90)
  (h2 : x^2 * y + x * y^2 = 1122) : 
  x^2 + y^2 = 1044 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3028_302812


namespace NUMINAMATH_CALUDE_james_payment_l3028_302803

theorem james_payment (adoption_fee : ℝ) (friend_contribution_percent : ℝ) : 
  adoption_fee = 200 →
  friend_contribution_percent = 25 →
  adoption_fee - (friend_contribution_percent / 100 * adoption_fee) = 150 := by
  sorry

end NUMINAMATH_CALUDE_james_payment_l3028_302803


namespace NUMINAMATH_CALUDE_surface_area_of_large_cube_l3028_302875

/-- Given 8 cubes with volume 512 cm³ each, prove that the surface area of the large cube formed by joining them is 1536 cm² -/
theorem surface_area_of_large_cube (small_cube_volume : ℝ) (num_small_cubes : ℕ) :
  small_cube_volume = 512 →
  num_small_cubes = 8 →
  let small_cube_side := small_cube_volume ^ (1/3 : ℝ)
  let large_cube_side := 2 * small_cube_side
  let large_cube_surface_area := 6 * large_cube_side^2
  large_cube_surface_area = 1536 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_large_cube_l3028_302875


namespace NUMINAMATH_CALUDE_lamp_distance_in_specific_classroom_l3028_302890

/-- Represents a classroom with two lamps -/
structure Classroom where
  length : ℝ
  ceiling_height : ℝ
  lamp1_position : ℝ
  lamp2_position : ℝ
  lamp1_circle_diameter : ℝ
  lamp2_illumination_length : ℝ

/-- The distance between two lamps in the classroom -/
def lamp_distance (c : Classroom) : ℝ :=
  |c.lamp1_position - c.lamp2_position|

/-- Theorem stating the distance between lamps in the specific classroom setup -/
theorem lamp_distance_in_specific_classroom :
  ∀ (c : Classroom),
    c.length = 10 ∧
    c.lamp1_circle_diameter = 6 ∧
    c.lamp2_illumination_length = 10 ∧
    c.lamp1_position = c.length / 2 ∧
    c.lamp2_position = 1 →
    lamp_distance c = 4 := by
  sorry

#check lamp_distance_in_specific_classroom

end NUMINAMATH_CALUDE_lamp_distance_in_specific_classroom_l3028_302890


namespace NUMINAMATH_CALUDE_comics_reassembly_l3028_302848

theorem comics_reassembly (pages_per_comic : ℕ) (torn_pages : ℕ) (untorn_comics : ℕ) : 
  pages_per_comic = 25 →
  torn_pages = 150 →
  untorn_comics = 5 →
  (torn_pages / pages_per_comic + untorn_comics : ℕ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_comics_reassembly_l3028_302848


namespace NUMINAMATH_CALUDE_exist_triangle_area_le_two_l3028_302872

-- Define a lattice point
def LatticePoint := ℤ × ℤ

-- Define the condition for points within the square region
def WithinSquare (p : LatticePoint) : Prop :=
  |p.1| ≤ 2 ∧ |p.2| ≤ 2

-- Define a function to calculate the area of a triangle given three points
def TriangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Define the property of three points not being collinear
def NotCollinear (p1 p2 p3 : LatticePoint) : Prop :=
  TriangleArea p1 p2 p3 ≠ 0

-- Main theorem
theorem exist_triangle_area_le_two 
  (points : Fin 6 → LatticePoint)
  (h1 : ∀ i, WithinSquare (points i))
  (h2 : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → NotCollinear (points i) (points j) (points k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ TriangleArea (points i) (points j) (points k) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_exist_triangle_area_le_two_l3028_302872


namespace NUMINAMATH_CALUDE_quick_multiply_correct_l3028_302845

/-- Represents a two-digit number -/
def TwoDigitNumber (tens ones : Nat) : Nat :=
  10 * tens + ones

/-- The quick multiplication formula for two-digit numbers with reversed digits -/
def quickMultiply (x y : Nat) : Nat :=
  101 * x * y + 10 * (x^2 + y^2)

/-- Theorem stating that the quick multiplication formula is correct -/
theorem quick_multiply_correct (x y : Nat) (h1 : x < 10) (h2 : y < 10) :
  (TwoDigitNumber x y) * (TwoDigitNumber y x) = quickMultiply x y :=
by
  sorry

end NUMINAMATH_CALUDE_quick_multiply_correct_l3028_302845


namespace NUMINAMATH_CALUDE_power_product_equality_l3028_302810

theorem power_product_equality : 2000 * (2000 ^ 2000) = 2000 ^ 2001 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3028_302810


namespace NUMINAMATH_CALUDE_zunyi_temperature_difference_l3028_302807

/-- The temperature difference between the highest and lowest temperatures in Zunyi City on June 1, 2019 -/
def temperature_difference (highest lowest : ℝ) : ℝ := highest - lowest

/-- Theorem stating that the temperature difference is 10°C given the highest and lowest temperatures -/
theorem zunyi_temperature_difference :
  let highest : ℝ := 25
  let lowest : ℝ := 15
  temperature_difference highest lowest = 10 := by
  sorry

end NUMINAMATH_CALUDE_zunyi_temperature_difference_l3028_302807


namespace NUMINAMATH_CALUDE_two_sunny_days_probability_l3028_302855

/-- The probability of exactly k successes in n independent trials,
    each with probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem two_sunny_days_probability :
  binomial_probability 5 2 (3/10 : ℝ) = 3087/10000 := by
  sorry

end NUMINAMATH_CALUDE_two_sunny_days_probability_l3028_302855


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l3028_302821

def entrance_ticket_cost (num_students : ℕ) (num_teachers : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (num_students + num_teachers)

theorem museum_ticket_cost :
  entrance_ticket_cost 20 3 115 = 5 := by
sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l3028_302821


namespace NUMINAMATH_CALUDE_curve_length_l3028_302859

/-- The length of a curve defined by the intersection of a plane and a sphere --/
theorem curve_length (x y z : ℝ) : 
  x + y + z = 8 → 
  x * y + y * z + x * z = -18 → 
  (∃ (l : ℝ), l = 4 * Real.pi * Real.sqrt (59 / 3) ∧ 
    l = 2 * Real.pi * Real.sqrt (100 - (8 * 8) / 3)) := by
  sorry

end NUMINAMATH_CALUDE_curve_length_l3028_302859


namespace NUMINAMATH_CALUDE_b_sixth_mod_n_l3028_302878

theorem b_sixth_mod_n (n : ℕ+) (b : ℤ) (h : b^3 ≡ 1 [ZMOD n]) :
  b^6 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_b_sixth_mod_n_l3028_302878


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l3028_302895

theorem set_equality_implies_a_equals_three (a : ℝ) : 
  let A : Set ℝ := {0, 1, a^2}
  let B : Set ℝ := {1, 0, 2*a+3}
  A ∩ B = A ∪ B → a = 3 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l3028_302895


namespace NUMINAMATH_CALUDE_pizza_topping_combinations_l3028_302864

theorem pizza_topping_combinations (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) + (n.choose 3) = 92 := by sorry

end NUMINAMATH_CALUDE_pizza_topping_combinations_l3028_302864


namespace NUMINAMATH_CALUDE_component_probability_l3028_302888

theorem component_probability (p : ℝ) : 
  p ∈ Set.Icc 0 1 →
  (1 - (1 - p)^3 = 0.999) →
  p = 0.9 := by
sorry

end NUMINAMATH_CALUDE_component_probability_l3028_302888


namespace NUMINAMATH_CALUDE_equation_solution_l3028_302884

theorem equation_solution : 
  ∃ c : ℚ, (c - 35) / 14 = (2 * c + 9) / 49 ∧ c = 1841 / 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3028_302884


namespace NUMINAMATH_CALUDE_maple_trees_after_cutting_l3028_302836

/-- The number of maple trees remaining after cutting -/
def remaining_maple_trees (initial : ℝ) (cut : ℝ) : ℝ :=
  initial - cut

/-- Proof that the number of maple trees remaining is 7.0 -/
theorem maple_trees_after_cutting :
  remaining_maple_trees 9.0 2.0 = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_after_cutting_l3028_302836


namespace NUMINAMATH_CALUDE_exponent_product_equals_twentyfive_l3028_302825

theorem exponent_product_equals_twentyfive :
  (5 ^ 0.4) * (5 ^ 0.6) * (5 ^ 0.2) * (5 ^ 0.3) * (5 ^ 0.5) = 25 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_equals_twentyfive_l3028_302825


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3028_302856

theorem greatest_power_of_two_factor (n : ℕ) : n = 1003 →
  ∃ k : ℕ, (2^n : ℤ) ∣ (10^n - 4^(n/2)) ∧
  ∀ m : ℕ, m > n → ¬((2^m : ℤ) ∣ (10^n - 4^(n/2))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3028_302856


namespace NUMINAMATH_CALUDE_system_equations_proof_l3028_302853

theorem system_equations_proof (x y a : ℝ) : 
  (3 * x + y = 2 + 3 * a) →
  (x + 3 * y = 2 + a) →
  (x + y < 0) →
  (a < -1) ∧ (|1 - a| + |a + 1/2| = 1/2 - 2 * a) := by
sorry

end NUMINAMATH_CALUDE_system_equations_proof_l3028_302853


namespace NUMINAMATH_CALUDE_beadshop_profit_l3028_302847

theorem beadshop_profit (monday_profit_ratio : ℚ) (tuesday_profit_ratio : ℚ) (wednesday_profit : ℚ) 
  (h1 : monday_profit_ratio = 1/3)
  (h2 : tuesday_profit_ratio = 1/4)
  (h3 : wednesday_profit = 500) :
  ∃ total_profit : ℚ, 
    total_profit * (1 - monday_profit_ratio - tuesday_profit_ratio) = wednesday_profit ∧
    total_profit = 1200 := by
sorry

end NUMINAMATH_CALUDE_beadshop_profit_l3028_302847


namespace NUMINAMATH_CALUDE_min_cuts_for_100_polygons_l3028_302885

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- Represents the state of the paper after cutting -/
structure PaperState where
  pieces : ℕ
  total_vertices : ℕ

/-- Initial state of the square paper -/
def initial_state : PaperState :=
  { pieces := 1, total_vertices := 4 }

/-- Function to model a single cut -/
def cut (state : PaperState) (new_vertices : ℕ) : PaperState :=
  { pieces := state.pieces + 1,
    total_vertices := state.total_vertices + new_vertices }

/-- Predicate to check if the final state is valid -/
def is_valid_final_state (state : PaperState) : Prop :=
  state.pieces = 100 ∧ state.total_vertices = 100 * 20

/-- Theorem stating the minimum number of cuts required -/
theorem min_cuts_for_100_polygons :
  ∃ (n : ℕ), n = 1699 ∧
  ∃ (cut_sequence : List ℕ),
    cut_sequence.length = n ∧
    (cut_sequence.all (λ x => x ∈ [2, 3, 4])) ∧
    is_valid_final_state (cut_sequence.foldl cut initial_state) ∧
    ∀ (m : ℕ) (other_sequence : List ℕ),
      m < n →
      other_sequence.length = m →
      (other_sequence.all (λ x => x ∈ [2, 3, 4])) →
      ¬is_valid_final_state (other_sequence.foldl cut initial_state) :=
sorry


end NUMINAMATH_CALUDE_min_cuts_for_100_polygons_l3028_302885


namespace NUMINAMATH_CALUDE_cricket_average_l3028_302898

theorem cricket_average (current_innings : Nat) (next_innings_runs : Nat) (average_increase : Nat) (current_average : Nat) : 
  current_innings = 20 →
  next_innings_runs = 116 →
  average_increase = 4 →
  (current_innings * current_average + next_innings_runs) / (current_innings + 1) = current_average + average_increase →
  current_average = 32 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l3028_302898


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_sufficient_not_necessary_condition_l3028_302899

-- Define the sets M and P
def M : Set ℝ := {x | x < -3 ∨ x > 5}
def P (a : ℝ) : Set ℝ := {x | (x - a) * (x - 8) ≤ 0}

-- Theorem 1: Necessary and sufficient condition
theorem necessary_and_sufficient_condition (a : ℝ) :
  M ∩ P a = {x | 5 < x ∧ x ≤ 8} ↔ -3 ≤ a ∧ a ≤ 5 := by sorry

-- Theorem 2: Sufficient but not necessary condition
theorem sufficient_not_necessary_condition :
  ∃ a : ℝ, (M ∩ P a = {x | 5 < x ∧ x ≤ 8}) ∧
  ¬(∀ b : ℝ, M ∩ P b = {x | 5 < x ∧ x ≤ 8} → b = a) := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_sufficient_not_necessary_condition_l3028_302899


namespace NUMINAMATH_CALUDE_num_terms_eq_508020_l3028_302870

/-- The number of terms in the simplified expression of (x+y+z+w)^2008 + (x-y-z-w)^2008 -/
def num_terms : ℕ :=
  let n := 2008
  let sum := (n / 2 + 1)^2 - (n / 2) * (n / 2 + 1) / 2
  sum

/-- Theorem stating that the number of terms in the simplified expression
    of (x+y+z+w)^2008 + (x-y-z-w)^2008 is equal to 508020 -/
theorem num_terms_eq_508020 : num_terms = 508020 := by
  sorry

end NUMINAMATH_CALUDE_num_terms_eq_508020_l3028_302870


namespace NUMINAMATH_CALUDE_epsilon_max_ratio_l3028_302860

/-- Represents a contestant's performance in a math contest --/
structure ContestPerformance where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ
  day3_score : ℕ
  day3_attempted : ℕ

/-- Calculates the total score for a contestant --/
def totalScore (p : ContestPerformance) : ℕ := p.day1_score + p.day2_score + p.day3_score

/-- Calculates the total attempted points for a contestant --/
def totalAttempted (p : ContestPerformance) : ℕ := 
  p.day1_attempted + p.day2_attempted + p.day3_attempted

/-- Calculates the success ratio for a contestant --/
def successRatio (p : ContestPerformance) : ℚ := 
  (totalScore p : ℚ) / (totalAttempted p : ℚ)

/-- Delta's performance in the contest --/
def delta : ContestPerformance := {
  day1_score := 210,
  day1_attempted := 350,
  day2_score := 320, -- Assumed based on total score
  day2_attempted := 450, -- Assumed based on total attempted
  day3_score := 0, -- Placeholder
  day3_attempted := 0 -- Placeholder
}

theorem epsilon_max_ratio :
  ∀ epsilon : ContestPerformance,
  totalAttempted epsilon = 800 →
  totalAttempted delta = 800 →
  successRatio delta = 530 / 800 →
  epsilon.day1_attempted ≠ 350 →
  epsilon.day1_score > 0 →
  epsilon.day2_score > 0 →
  epsilon.day3_score > 0 →
  (epsilon.day1_score : ℚ) / (epsilon.day1_attempted : ℚ) < 210 / 350 →
  (epsilon.day2_score : ℚ) / (epsilon.day2_attempted : ℚ) < (delta.day2_score : ℚ) / (delta.day2_attempted : ℚ) →
  (epsilon.day3_score : ℚ) / (epsilon.day3_attempted : ℚ) < (delta.day3_score : ℚ) / (delta.day3_attempted : ℚ) →
  successRatio epsilon ≤ 789 / 800 :=
by sorry

end NUMINAMATH_CALUDE_epsilon_max_ratio_l3028_302860


namespace NUMINAMATH_CALUDE_doritos_distribution_l3028_302816

theorem doritos_distribution (total_bags : ℕ) (doritos_fraction : ℚ) (num_piles : ℕ) : 
  total_bags = 200 →
  doritos_fraction = 2 / 5 →
  num_piles = 5 →
  (total_bags : ℚ) * doritos_fraction / num_piles = 16 := by
  sorry

end NUMINAMATH_CALUDE_doritos_distribution_l3028_302816


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3028_302808

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 > 0 →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3028_302808


namespace NUMINAMATH_CALUDE_pencil_price_l3028_302882

theorem pencil_price (price : ℝ) : 
  price = 5000 - 20 → price / 10000 = 0.5 := by sorry

end NUMINAMATH_CALUDE_pencil_price_l3028_302882


namespace NUMINAMATH_CALUDE_corn_cobs_picked_l3028_302826

theorem corn_cobs_picked (bushel_weight : ℝ) (ear_weight : ℝ) (bushels_picked : ℝ) : 
  bushel_weight = 56 → 
  ear_weight = 0.5 → 
  bushels_picked = 2 → 
  (bushels_picked * bushel_weight / ear_weight : ℝ) = 224 := by
sorry

end NUMINAMATH_CALUDE_corn_cobs_picked_l3028_302826


namespace NUMINAMATH_CALUDE_incorrect_calculation_l3028_302840

theorem incorrect_calculation (a : ℝ) (n : ℕ) : 
  a^(2*n) * (a^(2*n))^3 / a^(4*n) ≠ a^2 :=
sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l3028_302840


namespace NUMINAMATH_CALUDE_smallest_cube_multiplier_l3028_302876

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_cube_multiplier (k : ℕ) : 
  (∀ m : ℕ, m < 4500 → ¬ ∃ n : ℕ, m * y = n^3) ∧ 
  (∃ n : ℕ, 4500 * y = n^3) := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_multiplier_l3028_302876


namespace NUMINAMATH_CALUDE_triangle_side_length_l3028_302865

namespace TriangleProof

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  AC : ℝ
  BC : ℝ
  cos_A : ℝ
  cos_B : ℝ
  h_cos_A : cos_A = 3/5
  h_cos_B : cos_B = 5/13
  h_AC : AC = 3

/-- The main theorem to prove -/
theorem triangle_side_length (t : Triangle) : t.AB = 14/5 := by
  sorry

end TriangleProof

end NUMINAMATH_CALUDE_triangle_side_length_l3028_302865


namespace NUMINAMATH_CALUDE_natural_number_triples_l3028_302818

theorem natural_number_triples (a b c : ℕ) :
  (∃ m n p : ℕ, (a + b : ℚ) / c = m ∧ (b + c : ℚ) / a = n ∧ (c + a : ℚ) / b = p) →
  (∃ k : ℕ, (a = k ∧ b = k ∧ c = k) ∨
            (a = k ∧ b = k ∧ c = 2 * k) ∨
            (a = k ∧ b = 2 * k ∧ c = 3 * k) ∨
            (a = k ∧ c = 2 * k ∧ b = 3 * k) ∨
            (b = k ∧ a = 2 * k ∧ c = 3 * k) ∨
            (b = k ∧ c = 2 * k ∧ a = 3 * k) ∨
            (c = k ∧ a = 2 * k ∧ b = 3 * k) ∨
            (c = k ∧ b = 2 * k ∧ a = 3 * k)) :=
sorry

end NUMINAMATH_CALUDE_natural_number_triples_l3028_302818


namespace NUMINAMATH_CALUDE_isabella_marble_problem_l3028_302883

def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem isabella_marble_problem :
  ∀ k : ℕ, k < 45 → P k ≥ 1 / 2023 ∧ P 45 < 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_isabella_marble_problem_l3028_302883


namespace NUMINAMATH_CALUDE_floor_painting_theorem_l3028_302842

/-- The number of integer pairs (a, b) satisfying the floor painting conditions -/
def floor_painting_solutions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let a := p.1
    let b := p.2
    b > a ∧ b % 3 = 0 ∧ (a - 6) * (b - 6) = 36 ∧ a > 6 ∧ b > 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 2 solutions to the floor painting problem -/
theorem floor_painting_theorem : floor_painting_solutions = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_theorem_l3028_302842


namespace NUMINAMATH_CALUDE_first_place_points_l3028_302843

def second_place_points : Nat := 7
def third_place_points : Nat := 5
def fourth_place_points : Nat := 2
def total_participations : Nat := 7
def product_of_points : Nat := 38500

theorem first_place_points (first_place : Nat) 
  (h1 : ∃ (a b c d : Nat), a + b + c + d = total_participations ∧ 
                           a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
                           first_place^a * second_place_points^b * 
                           third_place_points^c * fourth_place_points^d = product_of_points) : 
  first_place = 11 := by
sorry

end NUMINAMATH_CALUDE_first_place_points_l3028_302843


namespace NUMINAMATH_CALUDE_picture_book_shelves_l3028_302896

theorem picture_book_shelves 
  (books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) 
  (total_books : ℕ) : 
  books_per_shelf = 4 → 
  mystery_shelves = 5 → 
  total_books = 32 → 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 3 := by
sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l3028_302896


namespace NUMINAMATH_CALUDE_reflection_of_point_A_l3028_302852

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflect a point over the origin -/
def reflectOverOrigin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

theorem reflection_of_point_A :
  let A : Point3D := { x := 2, y := 3, z := 4 }
  reflectOverOrigin A = { x := -2, y := -3, z := -4 } := by
  sorry

#check reflection_of_point_A

end NUMINAMATH_CALUDE_reflection_of_point_A_l3028_302852


namespace NUMINAMATH_CALUDE_highest_probability_A_l3028_302838

-- Define the sample space
variable (Ω : Type)
-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define events A, B, and C
variable (A B C : Set Ω)

-- State the theorem
theorem highest_probability_A (hCB : C ⊆ B) (hBA : B ⊆ A) :
  P A ≥ P B ∧ P A ≥ P C := by
  sorry

end NUMINAMATH_CALUDE_highest_probability_A_l3028_302838


namespace NUMINAMATH_CALUDE_grandfather_age_proof_l3028_302897

/-- The age of the grandfather -/
def grandfather_age : ℕ := 84

/-- The age of the older grandson -/
def older_grandson_age : ℕ := grandfather_age / 3

/-- The age of the younger grandson -/
def younger_grandson_age : ℕ := grandfather_age / 4

theorem grandfather_age_proof :
  (grandfather_age = 3 * older_grandson_age) ∧
  (grandfather_age = 4 * younger_grandson_age) ∧
  (older_grandson_age + younger_grandson_age = 49) →
  grandfather_age = 84 :=
by sorry

end NUMINAMATH_CALUDE_grandfather_age_proof_l3028_302897


namespace NUMINAMATH_CALUDE_second_question_percentage_l3028_302823

theorem second_question_percentage
  (first_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : first_correct = 75)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 20) :
  ∃ (second_correct : ℝ),
    second_correct = 25 ∧
    first_correct + second_correct - both_correct = 100 - neither_correct :=
sorry

end NUMINAMATH_CALUDE_second_question_percentage_l3028_302823


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3028_302851

theorem inequality_solution_range :
  ∀ d : ℝ, (∃ x : ℝ, |2017 - x| + |2018 - x| ≤ d) ↔ d ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3028_302851


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3028_302886

/-- Given a cylinder with base area S whose lateral surface unfolds into a square,
    prove that its lateral surface area is 4πS. -/
theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let h := 2 * Real.pi * r
  (h = 2 * r)  → 2 * Real.pi * r * h = 4 * Real.pi * S :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3028_302886


namespace NUMINAMATH_CALUDE_egg_difference_l3028_302862

/-- Given that Megan bought 2 dozen eggs, 3 eggs broke, and twice as many cracked,
    prove that the difference between eggs in perfect condition and cracked eggs is 9. -/
theorem egg_difference (total : ℕ) (broken : ℕ) (cracked : ℕ) :
  total = 2 * 12 →
  broken = 3 →
  cracked = 2 * broken →
  total - (broken + cracked) - cracked = 9 := by
  sorry

end NUMINAMATH_CALUDE_egg_difference_l3028_302862


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3028_302879

-- Define the hyperbola parameters
def hyperbola_equation (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 20 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = 2 * x

-- Define the focal length calculation
def focal_length (a : ℝ) : ℝ := 2 * a

-- Theorem statement
theorem hyperbola_focal_length (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x y : ℝ, hyperbola_equation x y a → asymptote_equation x y) :
  focal_length a = 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3028_302879


namespace NUMINAMATH_CALUDE_exists_row_or_column_with_sqrt_n_distinct_l3028_302889

/-- Represents a grid with n rows and n columns -/
structure Grid (n : ℕ) where
  entries : Fin n → Fin n → Fin n

/-- A grid is valid if each number from 1 to n appears exactly n times -/
def isValidGrid {n : ℕ} (g : Grid n) : Prop :=
  ∀ k : Fin n, (Finset.sum Finset.univ (λ i => Finset.sum Finset.univ (λ j => if g.entries i j = k then 1 else 0))) = n

/-- The number of distinct elements in a row -/
def distinctInRow {n : ℕ} (g : Grid n) (i : Fin n) : ℕ :=
  Finset.card (Finset.image (g.entries i) Finset.univ)

/-- The number of distinct elements in a column -/
def distinctInColumn {n : ℕ} (g : Grid n) (j : Fin n) : ℕ :=
  Finset.card (Finset.image (λ i => g.entries i j) Finset.univ)

/-- The main theorem -/
theorem exists_row_or_column_with_sqrt_n_distinct {n : ℕ} (g : Grid n) (h : isValidGrid g) :
  (∃ i : Fin n, distinctInRow g i ≥ Int.ceil (Real.sqrt n)) ∨
  (∃ j : Fin n, distinctInColumn g j ≥ Int.ceil (Real.sqrt n)) := by
  sorry

end NUMINAMATH_CALUDE_exists_row_or_column_with_sqrt_n_distinct_l3028_302889


namespace NUMINAMATH_CALUDE_appetizer_price_l3028_302817

def total_spent : ℚ := 50
def entree_percentage : ℚ := 80 / 100
def num_entrees : ℕ := 4
def num_appetizers : ℕ := 2

theorem appetizer_price :
  let entree_cost : ℚ := total_spent * entree_percentage
  let appetizer_total : ℚ := total_spent - entree_cost
  let single_appetizer_price : ℚ := appetizer_total / num_appetizers
  single_appetizer_price = 5 := by
sorry

end NUMINAMATH_CALUDE_appetizer_price_l3028_302817


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_l3028_302861

theorem smallest_perfect_square_divisible (n : ℕ) (h : n = 14400) :
  (∃ k : ℕ, k * k = n ∧ n / 5 = 2880) ∧
  (∀ m : ℕ, m < n → m / 5 = 2880 → ¬∃ j : ℕ, j * j = m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_l3028_302861


namespace NUMINAMATH_CALUDE_max_area_MPNQ_l3028_302801

noncomputable section

-- Define the curves C₁ and C₂ in polar coordinates
def C₁ (θ : Real) : Real := 2 * Real.sqrt 2

def C₂ (θ : Real) : Real := 4 * Real.sqrt 2 * (Real.cos θ + Real.sin θ)

-- Define the area of quadrilateral MPNQ as a function of α
def area_MPNQ (α : Real) : Real :=
  4 * Real.sqrt 2 * Real.sin (2 * α + Real.pi / 4) + 4 - 2 * Real.sqrt 2

-- Theorem statement
theorem max_area_MPNQ :
  ∃ α, 0 < α ∧ α < Real.pi / 2 ∧
  ∀ β, 0 < β → β < Real.pi / 2 →
  area_MPNQ β ≤ area_MPNQ α ∧
  area_MPNQ α = 4 + 2 * Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_max_area_MPNQ_l3028_302801


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l3028_302891

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l3028_302891


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l3028_302849

/-- Calculates the mean daily profit for a month given the mean profits of two halves --/
def mean_daily_profit (days : ℕ) (first_half_mean : ℚ) (second_half_mean : ℚ) : ℚ :=
  (first_half_mean * (days / 2) + second_half_mean * (days / 2)) / days

/-- Proves that the mean daily profit for the given scenario is 350 --/
theorem shopkeeper_profit : mean_daily_profit 30 245 455 = 350 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l3028_302849


namespace NUMINAMATH_CALUDE_sum_of_digits_of_special_number_l3028_302894

/-- The least 6-digit number -/
def least_six_digit : ℕ := 100000

/-- Function to check if a number is 6-digit -/
def is_six_digit (n : ℕ) : Prop := n ≥ least_six_digit ∧ n < 1000000

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem sum_of_digits_of_special_number :
  ∃ n : ℕ,
    is_six_digit n ∧
    n % 4 = 2 ∧
    n % 610 = 2 ∧
    n % 15 = 2 ∧
    (∀ m : ℕ, m < n → ¬(is_six_digit m ∧ m % 4 = 2 ∧ m % 610 = 2 ∧ m % 15 = 2)) ∧
    sum_of_digits n = 17 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_special_number_l3028_302894


namespace NUMINAMATH_CALUDE_marks_garden_l3028_302893

theorem marks_garden (yellow purple green : ℕ) : 
  purple = yellow + (yellow * 4 / 5) →
  green = (yellow + purple) / 4 →
  yellow + purple + green = 35 →
  yellow = 10 := by
sorry

end NUMINAMATH_CALUDE_marks_garden_l3028_302893


namespace NUMINAMATH_CALUDE_water_bottle_boxes_l3028_302802

theorem water_bottle_boxes (bottles_per_box : ℕ) (bottle_capacity : ℚ) (fill_ratio : ℚ) (total_water : ℚ) 
  (h1 : bottles_per_box = 50)
  (h2 : bottle_capacity = 12)
  (h3 : fill_ratio = 3/4)
  (h4 : total_water = 4500) :
  (total_water / (bottle_capacity * fill_ratio)) / bottles_per_box = 10 := by
sorry

end NUMINAMATH_CALUDE_water_bottle_boxes_l3028_302802


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l3028_302828

/-- The symmetric point of (a, b) with respect to the y-axis is (-a, b) -/
def symmetric_point_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- The given point A -/
def point_A : ℝ × ℝ := (2, -3)

/-- The expected symmetric point -/
def expected_symmetric_point : ℝ × ℝ := (-2, -3)

theorem symmetric_point_correct :
  symmetric_point_y_axis point_A = expected_symmetric_point := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l3028_302828


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3028_302871

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a2 : a 2 = 2) 
  (h_a5 : a 5 = 1/4) : 
  a 1 / a 0 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3028_302871


namespace NUMINAMATH_CALUDE_yellow_score_mixture_l3028_302887

theorem yellow_score_mixture (white black : ℕ) : 
  white * 6 = black * 7 →
  2 * (white - black) = 3 * 4 →
  white + black = 78 := by
sorry

end NUMINAMATH_CALUDE_yellow_score_mixture_l3028_302887


namespace NUMINAMATH_CALUDE_sequence_range_theorem_l3028_302830

def sequence_sum (n : ℕ) : ℚ := (-1)^(n+1) * (1 / 2^n)

def sequence_term (n : ℕ) : ℚ := sequence_sum n - sequence_sum (n-1)

theorem sequence_range_theorem (p : ℚ) : 
  (∃ n : ℕ, (p - sequence_term n) * (p - sequence_term (n+1)) < 0) ↔ 
  (-3/4 < p ∧ p < 1/2) :=
sorry

end NUMINAMATH_CALUDE_sequence_range_theorem_l3028_302830


namespace NUMINAMATH_CALUDE_integral_exp_2x_l3028_302881

theorem integral_exp_2x : ∫ x in (0)..(1/2), Real.exp (2*x) = (1/2) * (Real.exp 1 - 1) := by sorry

end NUMINAMATH_CALUDE_integral_exp_2x_l3028_302881


namespace NUMINAMATH_CALUDE_tricycle_count_l3028_302824

theorem tricycle_count (num_bicycles : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  num_bicycles = 24 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  total_wheels = 90 →
  ∃ num_tricycles : ℕ, num_tricycles = 14 ∧ 
    num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l3028_302824


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3028_302863

theorem quadratic_equation_solution (t s : ℝ) : t = 8 * s^2 + 2 * s → t = 5 →
  s = (-1 + Real.sqrt 41) / 8 ∨ s = (-1 - Real.sqrt 41) / 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3028_302863


namespace NUMINAMATH_CALUDE_ellipse_special_point_l3028_302834

def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

def line_intersects_ellipse (m t x y : ℝ) : Prop :=
  ellipse x y ∧ x = t*y + m

def distance_squared (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2

theorem ellipse_special_point :
  ∃ (m : ℝ), 
    (∀ (t x₁ y₁ x₂ y₂ : ℝ),
      line_intersects_ellipse m t x₁ y₁ ∧ 
      line_intersects_ellipse m t x₂ y₂ ∧ 
      (x₁, y₁) ≠ (x₂, y₂) →
      ∃ (k : ℝ), 
        1 / distance_squared m 0 x₁ y₁ + 1 / distance_squared m 0 x₂ y₂ = k) ∧
    m = 2 * Real.sqrt 15 / 5 ∧
    (∀ (t x₁ y₁ x₂ y₂ : ℝ),
      line_intersects_ellipse m t x₁ y₁ ∧ 
      line_intersects_ellipse m t x₂ y₂ ∧ 
      (x₁, y₁) ≠ (x₂, y₂) →
      1 / distance_squared m 0 x₁ y₁ + 1 / distance_squared m 0 x₂ y₂ = 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_special_point_l3028_302834


namespace NUMINAMATH_CALUDE_percentage_problem_l3028_302858

theorem percentage_problem : ∃ x : ℝ, (0.001 * x = 0.24) ∧ (x = 240) := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3028_302858


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l3028_302846

theorem trigonometric_expression_equality : 
  (Real.sin (92 * π / 180) - Real.sin (32 * π / 180) * Real.cos (60 * π / 180)) / Real.cos (32 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l3028_302846


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l3028_302835

-- Define the function f
def f (f'2 : ℝ) : ℝ → ℝ := λ x ↦ 3 * x^2 - 2 * x * f'2

-- State the theorem
theorem f_derivative_at_2 : 
  ∃ f'2 : ℝ, (deriv (f f'2)) 2 = 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l3028_302835


namespace NUMINAMATH_CALUDE_sum_753_326_base8_l3028_302822

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent. -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits. -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

/-- The theorem stating that the sum of 753₈ and 326₈ in base 8 is 1301₈. -/
theorem sum_753_326_base8 :
  decimalToBase8 (base8ToDecimal [7, 5, 3] + base8ToDecimal [3, 2, 6]) = [1, 3, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_753_326_base8_l3028_302822


namespace NUMINAMATH_CALUDE_water_added_proof_l3028_302814

/-- The amount of water added to fill a container -/
def water_added (capacity : ℝ) (initial_percent : ℝ) (final_percent : ℝ) : ℝ :=
  capacity * final_percent - capacity * initial_percent

/-- Theorem: The amount of water added to a 20-liter container to change it from 30% full to 3/4 full is 9 liters -/
theorem water_added_proof :
  water_added 20 0.3 0.75 = 9 := by
  sorry

end NUMINAMATH_CALUDE_water_added_proof_l3028_302814


namespace NUMINAMATH_CALUDE_rectangle_area_l3028_302811

theorem rectangle_area (w : ℝ) (h₁ : w > 0) (h₂ : 10 * w = 200) : w * (4 * w) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3028_302811


namespace NUMINAMATH_CALUDE_last_two_digits_of_A_power_20_l3028_302833

theorem last_two_digits_of_A_power_20 (A : ℤ) 
  (h1 : A % 2 = 0) 
  (h2 : A % 10 ≠ 0) : 
  A^20 % 100 = 76 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_A_power_20_l3028_302833


namespace NUMINAMATH_CALUDE_melies_remaining_money_l3028_302854

/-- Calculates the remaining money after Méliès buys groceries -/
theorem melies_remaining_money :
  let meat_weight : ℝ := 3.5
  let meat_price_per_kg : ℝ := 95
  let vegetable_weight : ℝ := 4
  let vegetable_price_per_kg : ℝ := 18
  let fruit_weight : ℝ := 2.5
  let fruit_price_per_kg : ℝ := 12
  let initial_money : ℝ := 450
  let total_cost : ℝ := meat_weight * meat_price_per_kg +
                        vegetable_weight * vegetable_price_per_kg +
                        fruit_weight * fruit_price_per_kg
  let remaining_money : ℝ := initial_money - total_cost
  remaining_money = 15.5 := by sorry

end NUMINAMATH_CALUDE_melies_remaining_money_l3028_302854
