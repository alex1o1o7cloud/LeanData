import Mathlib

namespace lolita_weekday_milk_l1056_105626

/-- The number of milk boxes Lolita drinks on a single weekday -/
def weekday_milk : ℕ := 3

/-- The number of milk boxes Lolita drinks on Saturday -/
def saturday_milk : ℕ := 2 * weekday_milk

/-- The number of milk boxes Lolita drinks on Sunday -/
def sunday_milk : ℕ := 3 * weekday_milk

/-- The total number of milk boxes Lolita drinks in a week -/
def total_weekly_milk : ℕ := 30

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

theorem lolita_weekday_milk :
  weekdays * weekday_milk = 15 ∧
  weekdays * weekday_milk + saturday_milk + sunday_milk = total_weekly_milk :=
sorry

end lolita_weekday_milk_l1056_105626


namespace donation_conversion_l1056_105696

theorem donation_conversion (usd_donation : ℝ) (exchange_rate : ℝ) (cny_donation : ℝ) : 
  usd_donation = 1.2 →
  exchange_rate = 6.25 →
  cny_donation = usd_donation * exchange_rate →
  cny_donation = 7.5 :=
by sorry

end donation_conversion_l1056_105696


namespace truncated_cube_edges_l1056_105627

/-- A truncated cube is a polyhedron obtained by truncating each vertex of a cube
    such that a small square face replaces each vertex, and no cutting planes
    intersect each other inside the cube. -/
structure TruncatedCube where
  -- We don't need to define the internal structure, just the concept

/-- The number of edges in a truncated cube -/
def num_edges (tc : TruncatedCube) : ℕ := 16

/-- Theorem stating that the number of edges in a truncated cube is 16 -/
theorem truncated_cube_edges (tc : TruncatedCube) :
  num_edges tc = 16 := by sorry

end truncated_cube_edges_l1056_105627


namespace tiles_cut_to_square_and_rectangle_l1056_105637

/-- Represents a rectangular tile with width and height -/
structure Tile where
  width : ℝ
  height : ℝ

/-- Represents a rectangle formed by tiles -/
structure Rectangle where
  width : ℝ
  height : ℝ
  tiles : List Tile

/-- Theorem stating that tiles can be cut to form a square and a rectangle -/
theorem tiles_cut_to_square_and_rectangle 
  (n : ℕ) 
  (original : Rectangle) 
  (h_unequal_sides : original.width ≠ original.height) 
  (h_tile_count : original.tiles.length = n) :
  ∃ (square : Rectangle) (remaining : Rectangle),
    square.width = square.height ∧
    square.tiles.length = n ∧
    remaining.tiles.length = n ∧
    (∀ t ∈ original.tiles, ∃ t1 t2, t1 ∈ square.tiles ∧ t2 ∈ remaining.tiles) :=
sorry

end tiles_cut_to_square_and_rectangle_l1056_105637


namespace portias_school_size_l1056_105671

-- Define variables for the number of students in each school
variable (L : ℕ) -- Lara's high school
variable (P : ℕ) -- Portia's high school
variable (M : ℕ) -- Mia's high school

-- Define the conditions
axiom portia_students : P = 4 * L
axiom mia_students : M = 2 * L
axiom total_students : P + L + M = 4200

-- Theorem to prove
theorem portias_school_size : P = 2400 := by
  sorry

end portias_school_size_l1056_105671


namespace irrational_equation_root_l1056_105630

theorem irrational_equation_root (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ Real.sqrt (2 * x + m) = x) → m = -1 := by
  sorry

end irrational_equation_root_l1056_105630


namespace family_age_theorem_l1056_105633

/-- Calculates the average age of a family given initial conditions --/
def average_family_age (initial_average_age : ℚ) (years_passed : ℕ) (child_age : ℕ) : ℚ :=
  let initial_total_age := initial_average_age * 2
  let current_total_age := initial_total_age + years_passed * 2 + child_age
  current_total_age / 3

/-- Proves that the average age of the family is 19 years --/
theorem family_age_theorem (initial_average_age : ℚ) (years_passed : ℕ) (child_age : ℕ)
  (h1 : initial_average_age = 23)
  (h2 : years_passed = 5)
  (h3 : child_age = 1) :
  average_family_age initial_average_age years_passed child_age = 19 := by
  sorry

#eval average_family_age 23 5 1

end family_age_theorem_l1056_105633


namespace simplify_sqrt_x_squared_y_l1056_105694

theorem simplify_sqrt_x_squared_y (x y : ℝ) (h : x * y < 0) :
  Real.sqrt (x^2 * y) = -x * Real.sqrt y := by
  sorry

end simplify_sqrt_x_squared_y_l1056_105694


namespace inequality_system_solutions_l1056_105658

theorem inequality_system_solutions :
  let S : Set ℤ := {x | x ≥ 0 ∧ 2*x + 5 ≤ 3*(x + 2) ∧ 2*x - (1 + 3*x)/2 < 1}
  S = {0, 1, 2, 3} := by
  sorry

end inequality_system_solutions_l1056_105658


namespace angle_372_in_first_quadrant_l1056_105604

/-- An angle is in the first quadrant if it is between 0° and 90° (exclusive) when reduced to the range [0°, 360°) -/
def is_in_first_quadrant (angle : ℝ) : Prop :=
  0 ≤ (angle % 360) ∧ (angle % 360) < 90

/-- Theorem: An angle of 372° is located in the first quadrant -/
theorem angle_372_in_first_quadrant :
  is_in_first_quadrant 372 := by
  sorry


end angle_372_in_first_quadrant_l1056_105604


namespace negation_equivalence_l1056_105675

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≥ 0) :=
by sorry

end negation_equivalence_l1056_105675


namespace rooks_placement_formula_l1056_105664

/-- The number of ways to place k non-attacking rooks on an n × n chessboard -/
def rooks_placement (n k : ℕ) : ℕ :=
  Nat.choose n k * Nat.descFactorial n k

/-- An n × n chessboard -/
structure Chessboard (n : ℕ) where
  size : ℕ := n

theorem rooks_placement_formula {n k : ℕ} (C : Chessboard n) (h : k ≤ n) :
  rooks_placement n k = Nat.choose n k * Nat.descFactorial n k := by
  sorry

end rooks_placement_formula_l1056_105664


namespace chocolate_cost_proof_l1056_105648

/-- The cost of the chocolate -/
def chocolate_cost : ℝ := 3

/-- The cost of the candy bar -/
def candy_bar_cost : ℝ := 6

theorem chocolate_cost_proof :
  chocolate_cost = 3 ∧
  candy_bar_cost = 6 ∧
  candy_bar_cost = chocolate_cost + 3 :=
by sorry

end chocolate_cost_proof_l1056_105648


namespace last_date_theorem_l1056_105651

/-- Represents a date in DD.MM.YYYY format -/
structure Date :=
  (day : Nat)
  (month : Nat)
  (year : Nat)

/-- Check if a date is valid -/
def is_valid_date (d : Date) : Bool :=
  d.day ≥ 1 && d.day ≤ 31 && d.month ≥ 1 && d.month ≤ 12 && d.year ≥ 1

/-- Get the set of digits used in a date -/
def date_digits (d : Date) : Finset Nat :=
  sorry

/-- Check if a date is before another date -/
def is_before (d1 d2 : Date) : Bool :=
  sorry

/-- Find the last date before a given date with the same set of digits -/
def last_date_with_same_digits (d : Date) : Date :=
  sorry

theorem last_date_theorem (current_date : Date) :
  let target_date := Date.mk 15 12 2012
  current_date = Date.mk 22 11 2015 →
  is_valid_date target_date ∧
  is_before target_date current_date ∧
  date_digits target_date = date_digits current_date ∧
  (∀ d : Date, is_valid_date d ∧ is_before d current_date ∧ date_digits d = date_digits current_date →
    is_before d target_date ∨ d = target_date) :=
by sorry

end last_date_theorem_l1056_105651


namespace arccos_cos_eleven_l1056_105608

theorem arccos_cos_eleven : 
  Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end arccos_cos_eleven_l1056_105608


namespace cos_pi_twelfth_l1056_105695

theorem cos_pi_twelfth : Real.cos (π / 12) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_pi_twelfth_l1056_105695


namespace max_log_sum_l1056_105616

theorem max_log_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 6) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 6 → Real.log x + 2 * Real.log y ≤ 3 * Real.log 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 6 ∧ Real.log x + 2 * Real.log y = 3 * Real.log 2) :=
by sorry

end max_log_sum_l1056_105616


namespace imaginary_unit_sum_l1056_105606

theorem imaginary_unit_sum (i : ℂ) : i * i = -1 → (i⁻¹ : ℂ) + i^2015 = -2*i := by
  sorry

end imaginary_unit_sum_l1056_105606


namespace ball_probability_l1056_105638

/-- Given a bag of 100 balls with specified colors, prove the probability of choosing a ball that is neither red nor purple -/
theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 100)
  (h2 : white = 50)
  (h3 : green = 30)
  (h4 : yellow = 8)
  (h5 : red = 9)
  (h6 : purple = 3)
  (h7 : total = white + green + yellow + red + purple) :
  (white + green + yellow : ℚ) / total = 88 / 100 := by
sorry

end ball_probability_l1056_105638


namespace complex_multiplication_l1056_105631

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 * i - 1) = -2 - i := by
  sorry

end complex_multiplication_l1056_105631


namespace fresh_fruit_weight_l1056_105657

theorem fresh_fruit_weight (total_fruit : ℕ) (fresh_ratio frozen_ratio : ℕ) 
  (h_total : total_fruit = 15000)
  (h_ratio : fresh_ratio = 7 ∧ frozen_ratio = 3) :
  (fresh_ratio * total_fruit) / (fresh_ratio + frozen_ratio) = 10500 :=
by sorry

end fresh_fruit_weight_l1056_105657


namespace composition_equation_solution_l1056_105607

theorem composition_equation_solution (p q : ℝ → ℝ) (c : ℝ) :
  (∀ x, p x = 3 * x - 8) →
  (∀ x, q x = 5 * x - c) →
  p (q 3) = 14 →
  c = 23 / 3 := by
sorry

end composition_equation_solution_l1056_105607


namespace balls_distribution_theorem_l1056_105697

def distribute_balls (n : ℕ) (k : ℕ) (min : ℕ) (max : ℕ) : ℕ :=
  sorry

theorem balls_distribution_theorem :
  distribute_balls 6 2 1 4 = 35 := by
  sorry

end balls_distribution_theorem_l1056_105697


namespace rachel_homework_pages_l1056_105640

theorem rachel_homework_pages (math_pages reading_pages biology_pages : ℕ) 
  (h1 : math_pages = 2)
  (h2 : reading_pages = 3)
  (h3 : biology_pages = 10) :
  math_pages + reading_pages + biology_pages = 15 := by
  sorry

end rachel_homework_pages_l1056_105640


namespace speed_limit_exceeders_l1056_105672

/-- The percentage of motorists who exceed the speed limit -/
def exceed_limit_percent : ℝ := 50

/-- The percentage of all motorists who receive speeding tickets -/
def receive_ticket_percent : ℝ := 40

/-- The percentage of speed limit exceeders who do not receive tickets -/
def no_ticket_percent : ℝ := 20

theorem speed_limit_exceeders :
  exceed_limit_percent = 50 :=
by
  sorry

#check speed_limit_exceeders

end speed_limit_exceeders_l1056_105672


namespace vector_operation_l1056_105653

theorem vector_operation (a b : ℝ × ℝ) :
  a = (2, 4) → b = (-1, 1) → 2 • a - b = (5, 7) := by sorry

end vector_operation_l1056_105653


namespace symmetry_about_yOz_plane_l1056_105669

/-- The symmetry of a point about the yOz plane in a rectangular coordinate system -/
theorem symmetry_about_yOz_plane (x y z : ℝ) : 
  let original_point := (x, y, z)
  let symmetric_point := (-x, y, z)
  symmetric_point = (- (x : ℝ), y, z) := by
  sorry

end symmetry_about_yOz_plane_l1056_105669


namespace jasons_stove_repair_cost_l1056_105641

theorem jasons_stove_repair_cost (stove_cost : ℝ) (wall_repair_ratio : ℝ) : 
  stove_cost = 1200 →
  wall_repair_ratio = 1 / 6 →
  stove_cost + (wall_repair_ratio * stove_cost) = 1400 := by
  sorry

end jasons_stove_repair_cost_l1056_105641


namespace marys_story_characters_l1056_105650

theorem marys_story_characters (total : ℕ) (a c d e : ℕ) : 
  total = 60 →
  a = total / 2 →
  c = a / 2 →
  d + e = total - a - c →
  d = 2 * e →
  d = 10 := by
  sorry

end marys_story_characters_l1056_105650


namespace exam_pass_count_l1056_105644

theorem exam_pass_count (total_students : ℕ) (total_average : ℚ) 
  (pass_average : ℚ) (fail_average : ℚ) (weight_ratio : ℚ × ℚ) :
  total_students = 150 ∧ 
  total_average = 40 ∧ 
  pass_average = 45 ∧ 
  fail_average = 20 ∧ 
  weight_ratio = (3, 1) →
  ∃ (pass_count : ℕ), pass_count = 85 ∧ 
    (pass_count : ℚ) * weight_ratio.1 * pass_average + 
    (total_students - pass_count : ℚ) * weight_ratio.2 * fail_average = 
    total_average * (pass_count * weight_ratio.1 + (total_students - pass_count) * weight_ratio.2) :=
by sorry

end exam_pass_count_l1056_105644


namespace complex_square_eq_neg45_neg48i_l1056_105652

theorem complex_square_eq_neg45_neg48i (z : ℂ) : 
  z^2 = -45 - 48*I ↔ z = 3 - 8*I ∨ z = -3 + 8*I := by
  sorry

end complex_square_eq_neg45_neg48i_l1056_105652


namespace marshas_pay_per_mile_l1056_105620

theorem marshas_pay_per_mile :
  let first_package_miles : ℝ := 10
  let second_package_miles : ℝ := 28
  let third_package_miles : ℝ := second_package_miles / 2
  let total_miles : ℝ := first_package_miles + second_package_miles + third_package_miles
  let total_pay : ℝ := 104
  total_pay / total_miles = 2 := by
sorry

end marshas_pay_per_mile_l1056_105620


namespace boys_in_second_grade_is_20_l1056_105628

/-- The number of boys in the second grade -/
def boys_in_second_grade : ℕ := sorry

/-- The number of girls in the second grade -/
def girls_in_second_grade : ℕ := 11

/-- The total number of students in the second grade -/
def students_in_second_grade : ℕ := boys_in_second_grade + girls_in_second_grade

/-- The number of students in the third grade -/
def students_in_third_grade : ℕ := 2 * students_in_second_grade

/-- The total number of students in grades 2 and 3 -/
def total_students : ℕ := 93

theorem boys_in_second_grade_is_20 :
  boys_in_second_grade = 20 ∧
  students_in_second_grade + students_in_third_grade = total_students :=
sorry

end boys_in_second_grade_is_20_l1056_105628


namespace jellybean_ratio_l1056_105613

/-- Proves the ratio of jellybeans Lorelai has eaten to the total jellybeans Rory and Gigi have -/
theorem jellybean_ratio (gigi_jellybeans : ℕ) (rory_extra_jellybeans : ℕ) (lorelai_jellybeans : ℕ) :
  gigi_jellybeans = 15 →
  rory_extra_jellybeans = 30 →
  lorelai_jellybeans = 180 →
  ∃ (m : ℕ), m * (gigi_jellybeans + (gigi_jellybeans + rory_extra_jellybeans)) = lorelai_jellybeans →
  (lorelai_jellybeans : ℚ) / (gigi_jellybeans + (gigi_jellybeans + rory_extra_jellybeans) : ℚ) = 3 := by
  sorry

#check jellybean_ratio

end jellybean_ratio_l1056_105613


namespace nandan_earning_is_2000_l1056_105618

/-- Represents the business investment scenario of Krishan and Nandan -/
structure BusinessInvestment where
  nandan_investment : ℝ
  nandan_time : ℝ
  total_gain : ℝ

/-- Calculates Nandan's earning based on the given business investment scenario -/
def nandan_earning (b : BusinessInvestment) : ℝ :=
  b.nandan_investment * b.nandan_time

/-- Theorem stating that Nandan's earning is 2000 given the specified conditions -/
theorem nandan_earning_is_2000 (b : BusinessInvestment) 
  (h1 : b.total_gain = 26000)
  (h2 : b.nandan_investment * b.nandan_time + 
        (4 * b.nandan_investment) * (3 * b.nandan_time) = b.total_gain) :
  nandan_earning b = 2000 := by
  sorry

#check nandan_earning_is_2000

end nandan_earning_is_2000_l1056_105618


namespace gear_speed_proportion_l1056_105685

/-- Represents a gear in the system -/
structure Gear where
  teeth : ℕ
  angular_speed : ℝ

/-- Represents the system of gears -/
structure GearSystem where
  P : Gear
  Q : Gear
  R : Gear
  efficiency : ℝ

/-- The theorem stating the correct proportion of angular speeds -/
theorem gear_speed_proportion (sys : GearSystem) 
  (h1 : sys.efficiency = 0.9)
  (h2 : sys.P.teeth * sys.P.angular_speed = sys.Q.teeth * sys.Q.angular_speed)
  (h3 : sys.R.angular_speed = sys.efficiency * sys.Q.angular_speed) :
  ∃ (k : ℝ), k > 0 ∧ 
    sys.P.angular_speed = k * sys.Q.teeth ∧
    sys.Q.angular_speed = k * sys.P.teeth ∧
    sys.R.angular_speed = k * sys.efficiency * sys.P.teeth :=
sorry

end gear_speed_proportion_l1056_105685


namespace eagle_speed_proof_l1056_105629

/-- The speed of the eagle in miles per hour -/
def eagle_speed : ℝ := 15

/-- The speed of the falcon in miles per hour -/
def falcon_speed : ℝ := 46

/-- The speed of the pelican in miles per hour -/
def pelican_speed : ℝ := 33

/-- The speed of the hummingbird in miles per hour -/
def hummingbird_speed : ℝ := 30

/-- The time all birds flew in hours -/
def flight_time : ℝ := 2

/-- The total distance covered by all birds in miles -/
def total_distance : ℝ := 248

theorem eagle_speed_proof :
  eagle_speed * flight_time +
  falcon_speed * flight_time +
  pelican_speed * flight_time +
  hummingbird_speed * flight_time =
  total_distance :=
sorry

end eagle_speed_proof_l1056_105629


namespace concentric_circles_radii_l1056_105643

theorem concentric_circles_radii 
  (chord_length : ℝ) 
  (ring_width : ℝ) 
  (h_chord : chord_length = 32) 
  (h_width : ring_width = 8) :
  ∃ (r R : ℝ), 
    r > 0 ∧ 
    R > r ∧
    R = r + ring_width ∧
    (r + ring_width)^2 = r^2 + (chord_length/2)^2 ∧
    r = 12 ∧ 
    R = 20 := by
  sorry

end concentric_circles_radii_l1056_105643


namespace shorter_leg_length_in_30_60_90_triangle_l1056_105622

theorem shorter_leg_length_in_30_60_90_triangle (median_length : ℝ) :
  median_length = 5 * Real.sqrt 3 →
  ∃ (shorter_leg hypotenuse : ℝ),
    shorter_leg = 5 ∧
    hypotenuse = 2 * shorter_leg ∧
    median_length = hypotenuse / 2 :=
by sorry

end shorter_leg_length_in_30_60_90_triangle_l1056_105622


namespace greatest_four_digit_number_l1056_105668

theorem greatest_four_digit_number : ∃ (n : ℕ), 
  (n = 9997) ∧ 
  (n < 10000) ∧ 
  (∃ (k : ℕ), n = 7 * k + 1) ∧ 
  (∃ (j : ℕ), n = 8 * j + 5) ∧ 
  (∀ (m : ℕ), m < 10000 → (∃ (k : ℕ), m = 7 * k + 1) → (∃ (j : ℕ), m = 8 * j + 5) → m ≤ n) :=
by sorry

end greatest_four_digit_number_l1056_105668


namespace opposite_of_difference_l1056_105690

theorem opposite_of_difference (a b : ℝ) : -(a - b) = b - a := by sorry

end opposite_of_difference_l1056_105690


namespace square_difference_equality_l1056_105612

theorem square_difference_equality : (2 + 3)^2 - (2^2 + 3^2) = 12 := by
  sorry

end square_difference_equality_l1056_105612


namespace star_calculation_l1056_105688

/-- The ⋆ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 3 ⋆ (5 ⋆ 6) = -112 -/
theorem star_calculation : star 3 (star 5 6) = -112 := by
  sorry

end star_calculation_l1056_105688


namespace intersection_complement_theorem_l1056_105692

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {1, 2, 3, 4}

-- Define set B
def B : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 4} := by sorry

end intersection_complement_theorem_l1056_105692


namespace f_of_two_equals_two_l1056_105625

theorem f_of_two_equals_two (f : ℝ → ℝ) (h : ∀ x ≥ 0, f (1 + Real.sqrt x) = x + 1) : f 2 = 2 := by
  sorry

end f_of_two_equals_two_l1056_105625


namespace x_minus_q_equals_three_minus_two_q_l1056_105673

theorem x_minus_q_equals_three_minus_two_q (x q : ℝ) 
  (h1 : |x - 3| = q) 
  (h2 : x < 3) : 
  x - q = 3 - 2*q := by
  sorry

end x_minus_q_equals_three_minus_two_q_l1056_105673


namespace polynomial_expansion_l1056_105621

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^2 - 4 * t + 3) * (-4 * t^2 + 2 * t - 6) = 
  -12 * t^4 + 22 * t^3 - 38 * t^2 + 30 * t - 18 := by
  sorry

end polynomial_expansion_l1056_105621


namespace magnitude_of_complex_power_l1056_105662

theorem magnitude_of_complex_power (z : ℂ) : 
  z = 4 + 2 * Complex.I * Real.sqrt 5 → Complex.abs (z^4) = 1296 := by
  sorry

end magnitude_of_complex_power_l1056_105662


namespace sugar_solution_percentage_l1056_105683

theorem sugar_solution_percentage (x : ℝ) :
  (3/4 * x + 1/4 * 26 = 14) → x = 10 := by
  sorry

end sugar_solution_percentage_l1056_105683


namespace robin_water_consumption_l1056_105689

theorem robin_water_consumption 
  (morning : ℝ) 
  (afternoon : ℝ) 
  (evening : ℝ) 
  (night : ℝ) 
  (m : ℝ) 
  (e : ℝ) 
  (t : ℝ) 
  (h1 : morning = 7.5) 
  (h2 : afternoon = 9.25) 
  (h3 : evening = 5.75) 
  (h4 : night = 3.5) 
  (h5 : m = morning + afternoon) 
  (h6 : e = evening + night) 
  (h7 : t = m + e) : 
  t = 16.75 + 9.25 := by
  sorry

end robin_water_consumption_l1056_105689


namespace polygon_sides_count_l1056_105663

-- Define the number of sides of the polygon
variable (n : ℕ)

-- Define the sum of interior angles
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Define the sum of exterior angles (always 360°)
def sum_exterior_angles : ℝ := 360

-- State the theorem
theorem polygon_sides_count :
  sum_interior_angles n = sum_exterior_angles + 720 → n = 8 := by
  sorry

end polygon_sides_count_l1056_105663


namespace tan_half_sum_l1056_105647

theorem tan_half_sum (a b : ℝ) 
  (h1 : Real.cos a + Real.cos b = (1 : ℝ) / 2)
  (h2 : Real.sin a + Real.sin b = (3 : ℝ) / 11) : 
  Real.tan ((a + b) / 2) = (6 : ℝ) / 11 := by
  sorry

end tan_half_sum_l1056_105647


namespace triangle_angle_calculation_l1056_105686

theorem triangle_angle_calculation (x : ℝ) : 
  x > 0 ∧ 
  40 + 3 * x + x = 180 →
  x = 35 := by
  sorry

end triangle_angle_calculation_l1056_105686


namespace sum_of_roots_quadratic_l1056_105680

theorem sum_of_roots_quadratic (x : ℝ) (h : x^2 + 12*x = 64) : 
  ∃ y : ℝ, y^2 + 12*y = 64 ∧ x + y = -12 := by
sorry

end sum_of_roots_quadratic_l1056_105680


namespace linear_is_bounded_multiple_rational_is_bounded_multiple_odd_lipschitz_is_bounded_multiple_l1056_105656

/-- A function is a bounded multiple function if there exists a constant M > 0 
    such that |f(x)| ≤ M|x| for all real x. -/
def BoundedMultipleFunction (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≤ M * |x|

/-- The function f(x) = 2x is a bounded multiple function. -/
theorem linear_is_bounded_multiple : BoundedMultipleFunction (fun x ↦ 2 * x) := by
  sorry

/-- The function f(x) = x/(x^2 - x + 3) is a bounded multiple function. -/
theorem rational_is_bounded_multiple : BoundedMultipleFunction (fun x ↦ x / (x^2 - x + 3)) := by
  sorry

/-- An odd function f(x) defined on ℝ that satisfies |f(x₁) - f(x₂)| ≤ 2|x₁ - x₂| 
    for all x₁, x₂ ∈ ℝ is a bounded multiple function. -/
theorem odd_lipschitz_is_bounded_multiple 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_lipschitz : ∀ x₁ x₂, |f x₁ - f x₂| ≤ 2 * |x₁ - x₂|) : 
  BoundedMultipleFunction f := by
  sorry

end linear_is_bounded_multiple_rational_is_bounded_multiple_odd_lipschitz_is_bounded_multiple_l1056_105656


namespace metal_waste_l1056_105603

theorem metal_waste (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let circle_area := Real.pi * (s/2)^2
  let inner_square_side := s / Real.sqrt 2
  let inner_square_area := inner_square_side^2
  let waste := square_area - circle_area + (circle_area - inner_square_area)
  waste = square_area / 2 :=
by sorry

end metal_waste_l1056_105603


namespace power_congruence_l1056_105615

theorem power_congruence (h : 5^200 ≡ 1 [ZMOD 1000]) :
  5^6000 ≡ 1 [ZMOD 1000] := by
  sorry

end power_congruence_l1056_105615


namespace rectangle_to_cylinder_volume_l1056_105678

/-- The volume of a cylinder formed by rolling a rectangle with length 6 and width 3 -/
theorem rectangle_to_cylinder_volume :
  ∃ (V : ℝ), (V = 27 / π ∨ V = 27 / (4 * π)) ∧
  ∃ (R h : ℝ), (R * h = 18 ∨ R * h = 9) ∧ V = π * R^2 * h := by
  sorry

end rectangle_to_cylinder_volume_l1056_105678


namespace investment_problem_l1056_105677

/-- The investment problem -/
theorem investment_problem 
  (x_investment : ℕ) 
  (z_investment : ℕ) 
  (z_join_time : ℕ) 
  (total_profit : ℕ) 
  (z_profit_share : ℕ) 
  (total_time : ℕ) 
  (h1 : x_investment = 36000)
  (h2 : z_investment = 48000)
  (h3 : z_join_time = 4)
  (h4 : total_profit = 13860)
  (h5 : z_profit_share = 4032)
  (h6 : total_time = 12) :
  ∃ y_investment : ℕ, 
    y_investment * total_time * (total_profit - z_profit_share) = 
    x_investment * total_time * z_profit_share - 
    z_investment * (total_time - z_join_time) * (total_profit - z_profit_share) ∧
    y_investment = 25000 := by
  sorry


end investment_problem_l1056_105677


namespace triangle_properties_l1056_105655

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c)
  (h2 : t.c = Real.sqrt 7)
  (h3 : t.a * t.b = 6) :
  t.C = π / 3 ∧ t.a + t.b + t.c = Real.sqrt 37 + Real.sqrt 7 := by
  sorry

end triangle_properties_l1056_105655


namespace max_min_on_interval_l1056_105667

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = min) ∧
    max = 5 ∧ min = 1 :=
by sorry

end max_min_on_interval_l1056_105667


namespace expression_equals_negative_one_l1056_105619

theorem expression_equals_negative_one (a y : ℝ) 
  (h1 : a ≠ 0) (h2 : a ≠ 2*y) (h3 : a ≠ -2*y) :
  (a / (a + 2*y) + y / (a - 2*y)) / (y / (a + 2*y) - a / (a - 2*y)) = -1 ↔ y = -a/3 :=
by sorry

end expression_equals_negative_one_l1056_105619


namespace three_subject_average_l1056_105665

/-- Given that the average score of Korean and mathematics is 86 points,
    and the English score is 98 points, prove that the average score
    of all three subjects is 90 points. -/
theorem three_subject_average (korean math english : ℝ) : 
  (korean + math) / 2 = 86 → 
  english = 98 → 
  (korean + math + english) / 3 = 90 := by
sorry

end three_subject_average_l1056_105665


namespace minor_premise_identification_l1056_105693

-- Define the types
def Shape : Type := String

-- Define the properties
def IsRectangle (s : Shape) : Prop := s = "rectangle"
def IsParallelogram (s : Shape) : Prop := s = "parallelogram"
def IsTriangle (s : Shape) : Prop := s = "triangle"

-- Define the syllogism statements
def MajorPremise : Prop := ∀ s : Shape, IsRectangle s → IsParallelogram s
def MinorPremise : Prop := ∃ s : Shape, IsTriangle s ∧ ¬IsParallelogram s
def Conclusion : Prop := ∃ s : Shape, IsTriangle s ∧ ¬IsRectangle s

-- Theorem to prove
theorem minor_premise_identification :
  MinorPremise = (∃ s : Shape, IsTriangle s ∧ ¬IsParallelogram s) :=
by sorry

end minor_premise_identification_l1056_105693


namespace brenda_skittles_l1056_105670

theorem brenda_skittles (initial : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 7 → bought = 8 → final = initial + bought → final = 15 := by
  sorry

end brenda_skittles_l1056_105670


namespace four_points_no_obtuse_triangle_l1056_105617

noncomputable def probability_no_obtuse_triangle (n : ℕ) : ℝ :=
  sorry

theorem four_points_no_obtuse_triangle :
  probability_no_obtuse_triangle 4 = 3 / 32 :=
sorry

end four_points_no_obtuse_triangle_l1056_105617


namespace f_properties_l1056_105659

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - x - 1/a) * Real.exp (a * x)

theorem f_properties (h : a ≠ 0) :
  -- Part I
  (a = 1/2 → (f a x = 0 ↔ x = -1 ∨ x = 2)) ∧
  -- Part II
  (∀ x, f a x = 0 → x = 1 ∨ x = -2/a) ∧
  -- Part III
  (a > 0 → (∀ x, f a x + 2/a ≥ 0) ↔ 0 < a ∧ a ≤ Real.log 2) :=
sorry

end

end f_properties_l1056_105659


namespace white_balls_count_l1056_105632

theorem white_balls_count (red blue white : ℕ) : 
  red = 80 → blue = 40 → red = blue + white - 12 → white = 52 := by
  sorry

end white_balls_count_l1056_105632


namespace correct_answers_count_l1056_105687

/-- Represents a mathematics contest with scoring rules and results. -/
structure MathContest where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  answered_problems : ℕ
  total_score : ℤ

/-- Theorem stating that given the contest conditions, 11 correct answers result in a score of 54. -/
theorem correct_answers_count (contest : MathContest) 
  (h1 : contest.total_problems = 15)
  (h2 : contest.correct_points = 6)
  (h3 : contest.incorrect_points = -3)
  (h4 : contest.answered_problems = contest.total_problems)
  (h5 : contest.total_score = 54) :
  ∃ (correct : ℕ), correct = 11 ∧ 
    contest.correct_points * correct + contest.incorrect_points * (contest.total_problems - correct) = contest.total_score :=
by sorry

end correct_answers_count_l1056_105687


namespace range_of_f_l1056_105601

def f (x : ℝ) : ℝ := x^2 - 6*x + 10

theorem range_of_f :
  Set.range f = {y | y ≥ 1} := by sorry

end range_of_f_l1056_105601


namespace maddies_mom_coffee_cost_l1056_105661

/-- Represents the weekly coffee consumption and cost for Maddie's mom -/
structure CoffeeConsumption where
  cups_per_day : ℕ
  beans_per_cup : ℚ
  beans_per_bag : ℚ
  cost_per_bag : ℚ
  milk_per_week : ℚ
  cost_per_gallon_milk : ℚ

/-- Calculates the weekly cost of coffee -/
def weekly_coffee_cost (c : CoffeeConsumption) : ℚ :=
  let beans_per_week := c.cups_per_day * c.beans_per_cup * 7
  let bags_per_week := beans_per_week / c.beans_per_bag
  let coffee_cost := bags_per_week * c.cost_per_bag
  let milk_cost := c.milk_per_week * c.cost_per_gallon_milk
  coffee_cost + milk_cost

/-- Theorem stating that Maddie's mom's weekly coffee cost is $18 -/
theorem maddies_mom_coffee_cost :
  let c : CoffeeConsumption := {
    cups_per_day := 2,
    beans_per_cup := 3/2,
    beans_per_bag := 21/2,
    cost_per_bag := 8,
    milk_per_week := 1/2,
    cost_per_gallon_milk := 4
  }
  weekly_coffee_cost c = 18 := by
  sorry

end maddies_mom_coffee_cost_l1056_105661


namespace cans_per_row_is_twelve_l1056_105645

/-- The number of rows on one shelf -/
def rows_per_shelf : ℕ := 4

/-- The number of shelves in one closet -/
def shelves_per_closet : ℕ := 10

/-- The total number of cans Jack can store in one closet -/
def cans_per_closet : ℕ := 480

/-- The number of cans Jack can fit in one row -/
def cans_per_row : ℕ := cans_per_closet / (shelves_per_closet * rows_per_shelf)

theorem cans_per_row_is_twelve : cans_per_row = 12 := by
  sorry

end cans_per_row_is_twelve_l1056_105645


namespace gcd_9011_4403_l1056_105636

theorem gcd_9011_4403 : Nat.gcd 9011 4403 = 1 := by
  sorry

end gcd_9011_4403_l1056_105636


namespace sara_apples_l1056_105691

theorem sara_apples (total : ℕ) (ali_ratio : ℕ) (sara_apples : ℕ) : 
  total = 80 →
  ali_ratio = 4 →
  total = sara_apples + ali_ratio * sara_apples →
  sara_apples = 16 := by
  sorry

end sara_apples_l1056_105691


namespace percent_calculation_l1056_105624

theorem percent_calculation (x : ℝ) (h : 0.20 * x = 200) : 1.20 * x = 1200 := by
  sorry

end percent_calculation_l1056_105624


namespace sequence_problem_l1056_105666

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a : a 1000 + a 1018 = 2 * Real.pi)
  (h_b : b 6 * b 2012 = 2) :
  Real.tan ((a 2 + a 2016) / (1 + b 3 * b 2015)) = -Real.sqrt 3 := by
  sorry

end sequence_problem_l1056_105666


namespace binomial_12_choose_2_l1056_105679

theorem binomial_12_choose_2 : Nat.choose 12 2 = 66 := by
  sorry

end binomial_12_choose_2_l1056_105679


namespace arithmetic_square_root_of_four_l1056_105698

theorem arithmetic_square_root_of_four :
  Real.sqrt 4 = 2 := by
  sorry

end arithmetic_square_root_of_four_l1056_105698


namespace num_valid_colorings_is_7776_l1056_105649

/-- A graph representing the extended figure described in the problem -/
def ExtendedFigureGraph : Type := Unit

/-- The number of vertices in the extended figure graph -/
def num_vertices : Nat := 12

/-- The number of available colors -/
def num_colors : Nat := 4

/-- A function that determines if two vertices are adjacent in the extended figure graph -/
def are_adjacent (v1 v2 : Fin num_vertices) : Bool := sorry

/-- A coloring of the graph is a function from vertices to colors -/
def Coloring := Fin num_vertices → Fin num_colors

/-- A predicate that determines if a coloring is valid (no adjacent vertices have the same color) -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ v1 v2 : Fin num_vertices, are_adjacent v1 v2 → c v1 ≠ c v2

/-- The number of valid colorings for the extended figure graph -/
def num_valid_colorings : Nat := sorry

/-- The main theorem stating that the number of valid colorings is 7776 -/
theorem num_valid_colorings_is_7776 : num_valid_colorings = 7776 := by sorry

end num_valid_colorings_is_7776_l1056_105649


namespace sum_of_cubes_of_roots_l1056_105639

theorem sum_of_cubes_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 5 * x₁ - 2 = 0) → 
  (3 * x₂^2 - 5 * x₂ - 2 = 0) → 
  x₁^3 + x₂^3 = 215 / 27 := by
sorry

end sum_of_cubes_of_roots_l1056_105639


namespace power_multiplication_l1056_105642

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end power_multiplication_l1056_105642


namespace finite_subsequence_exists_infinite_subsequence_not_exists_l1056_105605

/-- The sequence 1, 1/2, 1/3, ... -/
def harmonic_sequence : ℕ → ℚ 
  | n => 1 / n

/-- A subsequence of the harmonic sequence -/
structure Subsequence :=
  (indices : ℕ → ℕ)
  (strictly_increasing : ∀ n, indices n < indices (n + 1))

/-- The property that each term from the third is the difference of the two preceding terms -/
def has_difference_property (s : Subsequence) : Prop :=
  ∀ k ≥ 3, harmonic_sequence (s.indices k) = 
    harmonic_sequence (s.indices (k - 2)) - harmonic_sequence (s.indices (k - 1))

theorem finite_subsequence_exists : ∃ s : Subsequence, 
  (∀ n, n ≤ 100 → s.indices n ≤ 100) ∧ has_difference_property s :=
sorry

theorem infinite_subsequence_not_exists : ¬∃ s : Subsequence, has_difference_property s :=
sorry

end finite_subsequence_exists_infinite_subsequence_not_exists_l1056_105605


namespace smallest_d_inequality_l1056_105634

theorem smallest_d_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∃ (d : ℝ), d > 0 ∧ 
  (∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x^2 * y^2) + d * |x^2 - y^2| + x + y ≥ x^2 + y^2) ∧
  (∀ (d' : ℝ), d' > 0 → 
    (∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x^2 * y^2) + d' * |x^2 - y^2| + x + y ≥ x^2 + y^2) → 
    d ≤ d') ∧
  d = 1 :=
sorry

end smallest_d_inequality_l1056_105634


namespace jean_side_spots_l1056_105646

/-- Represents the number of spots on different parts of Jean the jaguar's body. -/
structure JeanSpots where
  total : ℕ
  upperTorso : ℕ
  backHindquarters : ℕ
  sides : ℕ

/-- Theorem stating the number of spots on Jean's sides given the conditions. -/
theorem jean_side_spots (j : JeanSpots) 
    (h1 : j.upperTorso = j.total / 2)
    (h2 : j.backHindquarters = j.total / 3)
    (h3 : j.upperTorso = 30)
    (h4 : j.total = j.upperTorso + j.backHindquarters + j.sides) :
    j.sides = 10 := by
  sorry

end jean_side_spots_l1056_105646


namespace distribute_unique_items_l1056_105654

theorem distribute_unique_items 
  (num_items : ℕ) 
  (num_recipients : ℕ) 
  (h1 : num_items = 6) 
  (h2 : num_recipients = 8) :
  (num_recipients ^ num_items : ℕ) = 262144 := by
  sorry

end distribute_unique_items_l1056_105654


namespace arithmetic_geometric_sequences_l1056_105699

theorem arithmetic_geometric_sequences :
  -- Arithmetic sequence
  ∃ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (a 8 = 6 ∧ a 10 = 0) →
    (∀ n, a n = 30 - 3 * n) ∧
    (∀ n, S n = -3/2 * n^2 + 57/2 * n) ∧
    (∀ n, n ≠ 9 ∧ n ≠ 10 → S n < S 9) ∧
  -- Geometric sequence
  ∃ (b : ℕ → ℝ) (T : ℕ → ℝ),
    (b 1 = 1/2 ∧ b 4 = 4) →
    (∀ n, b n = 2^(n-2)) ∧
    (∀ n, T n = 2^(n-1) - 1/2) :=
by
  sorry

end arithmetic_geometric_sequences_l1056_105699


namespace vector_dot_product_l1056_105623

/-- Given two vectors a and b in ℝ², prove that their dot product is 25 -/
theorem vector_dot_product (a b : ℝ × ℝ) : 
  a = (1, 2) → a - (1/5 : ℝ) • b = (-2, 1) → a.1 * b.1 + a.2 * b.2 = 25 := by
  sorry

end vector_dot_product_l1056_105623


namespace science_club_board_selection_l1056_105614

theorem science_club_board_selection (total_members : Nat) (prev_served : Nat) (board_size : Nat)
  (h1 : total_members = 20)
  (h2 : prev_served = 9)
  (h3 : board_size = 6) :
  (Nat.choose total_members board_size) - (Nat.choose (total_members - prev_served) board_size) = 38298 := by
  sorry

end science_club_board_selection_l1056_105614


namespace necessary_but_not_sufficient_condition_l1056_105674

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a > b - 1) ∧ 
  (∃ a b, a > b - 1 ∧ ¬(a > b)) :=
sorry

end necessary_but_not_sufficient_condition_l1056_105674


namespace marble_fraction_after_doubling_red_l1056_105684

theorem marble_fraction_after_doubling_red (total : ℚ) (h : total > 0) :
  let blue := (2 / 3) * total
  let red := total - blue
  let new_red := 2 * red
  let new_total := blue + new_red
  new_red / new_total = 1 / 2 := by sorry

end marble_fraction_after_doubling_red_l1056_105684


namespace dads_toothpaste_usage_l1056_105681

/-- Represents the amount of toothpaste used by Anne's dad at each brushing -/
def dads_toothpaste_use : ℝ := 3

/-- Theorem stating that Anne's dad uses 3 grams of toothpaste at each brushing -/
theorem dads_toothpaste_usage 
  (total_toothpaste : ℝ) 
  (moms_usage : ℝ)
  (kids_usage : ℝ)
  (brushings_per_day : ℕ)
  (days_to_empty : ℕ)
  (h1 : total_toothpaste = 105)
  (h2 : moms_usage = 2)
  (h3 : kids_usage = 1)
  (h4 : brushings_per_day = 3)
  (h5 : days_to_empty = 5)
  : dads_toothpaste_use = 3 := by
  sorry

#check dads_toothpaste_usage

end dads_toothpaste_usage_l1056_105681


namespace water_carriers_capacity_l1056_105609

/-- Represents the water-carrying capacity and trip ratio of two people --/
structure WaterCarriers where
  bucket_capacity : ℕ
  jack_buckets_per_trip : ℕ
  jill_buckets_per_trip : ℕ
  jack_trips_ratio : ℕ
  jill_trips_ratio : ℕ
  jill_total_trips : ℕ

/-- Calculates the total capacity of water carried by both people --/
def total_capacity (w : WaterCarriers) : ℕ :=
  let jack_trips := w.jack_trips_ratio * w.jill_total_trips / w.jill_trips_ratio
  let jack_capacity := w.bucket_capacity * w.jack_buckets_per_trip * jack_trips
  let jill_capacity := w.bucket_capacity * w.jill_buckets_per_trip * w.jill_total_trips
  jack_capacity + jill_capacity

/-- The theorem states that given the specified conditions, the total capacity is 600 gallons --/
theorem water_carriers_capacity : 
  ∀ (w : WaterCarriers), 
  w.bucket_capacity = 5 ∧ 
  w.jack_buckets_per_trip = 2 ∧ 
  w.jill_buckets_per_trip = 1 ∧ 
  w.jack_trips_ratio = 3 ∧ 
  w.jill_trips_ratio = 2 ∧ 
  w.jill_total_trips = 30 → 
  total_capacity w = 600 := by
  sorry

end water_carriers_capacity_l1056_105609


namespace largest_four_digit_negative_congruent_to_3_mod_29_l1056_105635

theorem largest_four_digit_negative_congruent_to_3_mod_29 :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n ≡ 3 [ZMOD 29] → n ≤ -1012 :=
by
  sorry

end largest_four_digit_negative_congruent_to_3_mod_29_l1056_105635


namespace sqrt_pattern_l1056_105611

theorem sqrt_pattern (n : ℕ+) : 
  Real.sqrt (1 + 1 / (n : ℝ)^2 + 1 / ((n + 1) : ℝ)^2) = 1 + 1 / ((n : ℝ) * (n + 1)) := by
  sorry

end sqrt_pattern_l1056_105611


namespace son_work_time_l1056_105602

-- Define the work rates
def man_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 3

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem to prove
theorem son_work_time : (1 : ℚ) / son_rate = 6 := by
  sorry

end son_work_time_l1056_105602


namespace max_sum_of_factors_l1056_105660

theorem max_sum_of_factors (x y : ℕ+) (h : x * y = 48) :
  x + y ≤ 49 ∧ ∃ (a b : ℕ+), a * b = 48 ∧ a + b = 49 := by
  sorry

end max_sum_of_factors_l1056_105660


namespace grape_juice_in_drink_l1056_105600

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ

/-- Calculates the amount of grape juice in the drink -/
def grape_juice_amount (drink : FruitDrink) : ℝ :=
  drink.total * (1 - drink.orange_percent - drink.watermelon_percent)

/-- Theorem stating the amount of grape juice in the specific drink -/
theorem grape_juice_in_drink : 
  let drink : FruitDrink := { total := 150, orange_percent := 0.35, watermelon_percent := 0.35 }
  grape_juice_amount drink = 45 := by
  sorry

end grape_juice_in_drink_l1056_105600


namespace arithmetic_sequence_product_l1056_105676

/-- 
Given an arithmetic sequence where:
- a₇ is the 7th term
- d is the common difference
- a₁ is the first term
- a₂ is the second term

This theorem states that if a₇ = 17 and d = 2, then a₁ * a₂ = 35.
-/
theorem arithmetic_sequence_product (a : ℕ → ℝ) (d : ℝ) :
  (a 7 = 17) → (∀ n, a (n + 1) - a n = d) → (d = 2) → (a 1 * a 2 = 35) := by
  sorry

#check arithmetic_sequence_product

end arithmetic_sequence_product_l1056_105676


namespace base_c_sum_equals_47_l1056_105610

/-- Represents a number in base c -/
structure BaseC (c : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < c

/-- Converts a base c number to its decimal (base 10) representation -/
def to_decimal (c : ℕ) (n : BaseC c) : ℕ := sorry

/-- Converts a decimal (base 10) number to its base c representation -/
def from_decimal (c : ℕ) (n : ℕ) : BaseC c := sorry

/-- Multiplies two numbers in base c -/
def mul_base_c (c : ℕ) (a b : BaseC c) : BaseC c := sorry

/-- Adds two numbers in base c -/
def add_base_c (c : ℕ) (a b : BaseC c) : BaseC c := sorry

theorem base_c_sum_equals_47 (c : ℕ) 
  (h_prod : mul_base_c c (mul_base_c c (from_decimal c 13) (from_decimal c 18)) (from_decimal c 17) = from_decimal c 4357) :
  let s := add_base_c c (add_base_c c (from_decimal c 13) (from_decimal c 18)) (from_decimal c 17)
  s = from_decimal c 47 := by
  sorry

end base_c_sum_equals_47_l1056_105610


namespace positive_A_value_l1056_105682

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 4 = 65) : A = 7 := by
  sorry

end positive_A_value_l1056_105682
