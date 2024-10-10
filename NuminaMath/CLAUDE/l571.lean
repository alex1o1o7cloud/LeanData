import Mathlib

namespace correct_num_raised_beds_l571_57116

/-- The number of raised beds Abby is building -/
def num_raised_beds : ℕ := 2

/-- The length of each raised bed in feet -/
def bed_length : ℕ := 8

/-- The width of each raised bed in feet -/
def bed_width : ℕ := 4

/-- The height of each raised bed in feet -/
def bed_height : ℕ := 1

/-- The volume of soil in each bag in cubic feet -/
def soil_per_bag : ℕ := 4

/-- The total number of soil bags needed -/
def total_soil_bags : ℕ := 16

/-- Theorem stating that the number of raised beds Abby is building is correct -/
theorem correct_num_raised_beds :
  num_raised_beds * (bed_length * bed_width * bed_height) = total_soil_bags * soil_per_bag :=
by sorry

end correct_num_raised_beds_l571_57116


namespace work_time_B_l571_57112

theorem work_time_B (time_A time_BC time_AC : ℝ) (h1 : time_A = 4) (h2 : time_BC = 3) (h3 : time_AC = 2) : 
  (1 / time_A + 1 / time_BC - 1 / time_AC)⁻¹ = 12 := by
sorry

end work_time_B_l571_57112


namespace simple_interest_rate_calculation_l571_57149

/-- Simple interest calculation --/
theorem simple_interest_rate_calculation 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 5000)
  (h2 : interest = 2500)
  (h3 : time = 5)
  : (interest * 100) / (principal * time) = 10 := by
  sorry

end simple_interest_rate_calculation_l571_57149


namespace senior_citizen_tickets_l571_57125

theorem senior_citizen_tickets (total_tickets : ℕ) (adult_price senior_price : ℕ) (total_receipts : ℕ) 
  (h1 : total_tickets = 529)
  (h2 : adult_price = 25)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 9745) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 348 := by
  sorry

end senior_citizen_tickets_l571_57125


namespace concentric_circles_radii_difference_l571_57139

theorem concentric_circles_radii_difference
  (s L : ℝ)
  (h_positive : s > 0)
  (h_ratio : L^2 / s^2 = 4) :
  L - s = s :=
sorry

end concentric_circles_radii_difference_l571_57139


namespace quadratic_roots_opposite_l571_57175

theorem quadratic_roots_opposite (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (k^2 - 4)*x₁ + k - 1 = 0 ∧
    x₂^2 + (k^2 - 4)*x₂ + k - 1 = 0 ∧
    x₁ = -x₂) →
  k = -2 := by
sorry

end quadratic_roots_opposite_l571_57175


namespace illuminated_area_of_cube_l571_57152

/-- The area of the illuminated part of a cube's surface when illuminated by a cylindrical beam -/
theorem illuminated_area_of_cube (a ρ : ℝ) (h_a : a = Real.sqrt (2 + Real.sqrt 3)) (h_ρ : ρ = Real.sqrt 2) :
  let S := ρ^2 * Real.sqrt 3 * (Real.pi - 6 * Real.arccos (a / (ρ * Real.sqrt 2)) + 
           6 * (a / (ρ * Real.sqrt 2)) * Real.sqrt (1 - (a / (ρ * Real.sqrt 2))^2))
  S = Real.sqrt 3 * (Real.pi + 3) :=
sorry

end illuminated_area_of_cube_l571_57152


namespace sum_reciprocal_l571_57169

theorem sum_reciprocal (x : ℝ) (w : ℝ) (h1 : x ≠ 0) (h2 : w = x^2 + (1/x)^2) (h3 : w = 23) :
  x + (1/x) = 5 := by
  sorry

end sum_reciprocal_l571_57169


namespace gcd_of_specific_numbers_l571_57107

theorem gcd_of_specific_numbers : 
  let m : ℕ := 3333333
  let n : ℕ := 66666666
  gcd m n = 3 := by sorry

end gcd_of_specific_numbers_l571_57107


namespace power_division_equals_one_l571_57170

theorem power_division_equals_one (a : ℝ) (h : a ≠ 0) : a^5 / a^5 = 1 := by
  sorry

end power_division_equals_one_l571_57170


namespace polynomial_coefficient_B_l571_57151

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ x : ℂ, x^6 - 12*x^5 + A*x^4 + (-162)*x^3 + C*x^2 + D*x + 36 = 
      (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) * (x - r₅) * (x - r₆)) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12 := by
  sorry

end polynomial_coefficient_B_l571_57151


namespace part_one_part_two_l571_57123

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x - 3) + m / (3 - x) = 3

-- Part 1
theorem part_one (m : ℝ) :
  equation 2 m → m = 5 := by
  sorry

-- Part 2
theorem part_two (x m : ℝ) :
  equation x m → x > 0 → m < 9 ∧ m ≠ 3 := by
  sorry

end part_one_part_two_l571_57123


namespace susan_homework_start_time_l571_57183

def volleyball_practice_start : Nat := 16 * 60 -- 4:00 p.m. in minutes since midnight

def homework_duration : Nat := 96 -- in minutes

def time_between_homework_and_practice : Nat := 25 -- in minutes

def homework_start_time : Nat := volleyball_practice_start - homework_duration - time_between_homework_and_practice

theorem susan_homework_start_time :
  homework_start_time = 13 * 60 + 59 -- 1:59 p.m. in minutes since midnight
  := by sorry

end susan_homework_start_time_l571_57183


namespace mario_earnings_l571_57110

/-- Mario's work hours and earnings over two weeks in July --/
theorem mario_earnings :
  ∀ (third_week_hours second_week_hours : ℕ) 
    (hourly_rate third_week_earnings second_week_earnings : ℚ),
  third_week_hours = 28 →
  third_week_hours = second_week_hours + 10 →
  third_week_earnings = second_week_earnings + 68 →
  hourly_rate * (third_week_hours : ℚ) = third_week_earnings →
  hourly_rate * (second_week_hours : ℚ) = second_week_earnings →
  hourly_rate * ((third_week_hours + second_week_hours) : ℚ) = 312.8 :=
by sorry

end mario_earnings_l571_57110


namespace multiples_of_four_between_100_and_300_l571_57164

theorem multiples_of_four_between_100_and_300 :
  (Finset.filter (fun n => n % 4 = 0) (Finset.range 300 \ Finset.range 101)).card = 49 := by
  sorry

end multiples_of_four_between_100_and_300_l571_57164


namespace base7_calculation_l571_57134

/-- Represents a number in base 7 --/
def Base7 : Type := Nat

/-- Converts a base 7 number to its decimal representation --/
def toDecimal (n : Base7) : Nat := sorry

/-- Converts a decimal number to its base 7 representation --/
def toBase7 (n : Nat) : Base7 := sorry

/-- Adds two base 7 numbers --/
def addBase7 (a b : Base7) : Base7 := sorry

/-- Subtracts two base 7 numbers --/
def subBase7 (a b : Base7) : Base7 := sorry

theorem base7_calculation : 
  let a := toBase7 2000
  let b := toBase7 1256
  let c := toBase7 345
  let d := toBase7 1042
  subBase7 (addBase7 (subBase7 a b) c) d = toBase7 0 := by sorry

end base7_calculation_l571_57134


namespace contractor_absent_days_l571_57117

/-- Proves the number of absent days for a contractor under specific conditions -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (daily_pay : ℚ) 
  (daily_fine : ℚ) 
  (total_received : ℚ) 
  (h1 : total_days = 30)
  (h2 : daily_pay = 25)
  (h3 : daily_fine = 7.5)
  (h4 : total_received = 360) :
  ∃ (absent_days : ℕ), 
    (absent_days : ℚ) * daily_fine + (total_days - absent_days : ℚ) * daily_pay = total_received ∧ 
    absent_days = 12 := by
  sorry


end contractor_absent_days_l571_57117


namespace problem_solution_l571_57101

theorem problem_solution (x y : ℝ) (h1 : x = 3) (h2 : y = 3) : 
  x - y^((x - y) / 3) = 2 := by
  sorry

end problem_solution_l571_57101


namespace f_seven_equals_negative_two_l571_57148

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_seven_equals_negative_two :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, f (x + 4) = f x) →  -- f has period 4
  (∀ x ∈ Set.Ioo 0 2, f x = 2 * x^2) →  -- f(x) = 2x^2 for x in (0,2)
  f 7 = -2 := by
sorry

end f_seven_equals_negative_two_l571_57148


namespace incorrect_observation_value_l571_57161

theorem incorrect_observation_value (n : ℕ) (original_mean corrected_mean correct_value : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 36)
  (h3 : corrected_mean = 36.5)
  (h4 : correct_value = 43) :
  ∃ x : ℝ, 
    (n : ℝ) * original_mean = (n : ℝ) * corrected_mean - correct_value + x :=
by
  sorry

end incorrect_observation_value_l571_57161


namespace race_speed_ratio_l571_57165

theorem race_speed_ratio (course_length : ℝ) (head_start : ℝ) 
  (h1 : course_length = 84)
  (h2 : head_start = 63)
  (h3 : course_length > head_start)
  (h4 : head_start > 0) :
  ∃ (speed_a speed_b : ℝ),
    speed_a > 0 ∧ speed_b > 0 ∧
    (course_length / speed_a = (course_length - head_start) / speed_b) ∧
    speed_a = 4 * speed_b :=
by sorry

end race_speed_ratio_l571_57165


namespace min_framing_for_picture_l571_57168

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with a border -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  (perimeter_inches + 11) / 12  -- Round up to the nearest foot

/-- The minimum number of linear feet of framing needed for the given picture specifications -/
theorem min_framing_for_picture : min_framing_feet 4 6 4 3 = 9 := by
  sorry

end min_framing_for_picture_l571_57168


namespace mike_total_cards_l571_57194

/-- The total number of baseball cards Mike has after his birthday -/
def total_cards (initial_cards birthday_cards : ℕ) : ℕ :=
  initial_cards + birthday_cards

/-- Theorem stating that Mike has 82 cards in total -/
theorem mike_total_cards : 
  total_cards 64 18 = 82 := by
  sorry

end mike_total_cards_l571_57194


namespace malcolm_lights_problem_l571_57178

theorem malcolm_lights_problem (initial_white : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : initial_white = 59)
  (h2 : red = 12)
  (h3 : green = 6) :
  initial_white - (red + 3 * red + green) = 5 := by
  sorry

end malcolm_lights_problem_l571_57178


namespace max_cylinder_radius_in_crate_l571_57137

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder fits inside a crate -/
def cylinderFitsInCrate (c : Cylinder) (d : CrateDimensions) : Prop :=
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.width ∧ c.height ≤ d.height) ∨
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.height ∧ c.height ≤ d.width) ∨
  (2 * c.radius ≤ d.width ∧ 2 * c.radius ≤ d.height ∧ c.height ≤ d.length)

/-- The main theorem stating that the maximum radius of a cylinder that fits in the given crate is 1.5 feet -/
theorem max_cylinder_radius_in_crate :
  let d := CrateDimensions.mk 3 8 12
  ∀ c : Cylinder, cylinderFitsInCrate c d → c.radius ≤ 1.5 := by
  sorry

end max_cylinder_radius_in_crate_l571_57137


namespace flower_count_l571_57160

/-- The number of pots -/
def num_pots : ℕ := 141

/-- The number of flowers in each pot -/
def flowers_per_pot : ℕ := 71

/-- The total number of flowers -/
def total_flowers : ℕ := num_pots * flowers_per_pot

theorem flower_count : total_flowers = 10011 := by
  sorry

end flower_count_l571_57160


namespace chips_note_taking_schedule_l571_57184

/-- Chip's note-taking schedule --/
theorem chips_note_taking_schedule 
  (pages_per_class : ℕ) 
  (num_classes : ℕ) 
  (sheets_per_pack : ℕ) 
  (num_weeks : ℕ) 
  (packs_used : ℕ) 
  (h1 : pages_per_class = 2)
  (h2 : num_classes = 5)
  (h3 : sheets_per_pack = 100)
  (h4 : num_weeks = 6)
  (h5 : packs_used = 3) :
  (packs_used * sheets_per_pack) / (pages_per_class * num_classes * num_weeks) = 5 := by
  sorry

#check chips_note_taking_schedule

end chips_note_taking_schedule_l571_57184


namespace quadratic_integer_solutions_l571_57133

theorem quadratic_integer_solutions (p q x₁ x₂ : ℝ) : 
  (∃ (x : ℝ), x^2 + p*x + q = 0) →  -- Quadratic equation has real solutions
  (x₁^2 + p*x₁ + q = 0) →           -- x₁ is a solution
  (x₂^2 + p*x₂ + q = 0) →           -- x₂ is a solution
  (x₁ ≠ x₂) →                       -- Solutions are distinct
  |x₁ - x₂| = 1 →                   -- Absolute difference of solutions is 1
  |p - q| = 1 →                     -- Absolute difference of p and q is 1
  (∃ (m n k l : ℤ), (↑m : ℝ) = p ∧ (↑n : ℝ) = q ∧ (↑k : ℝ) = x₁ ∧ (↑l : ℝ) = x₂) :=
by sorry

end quadratic_integer_solutions_l571_57133


namespace perpendicular_to_horizontal_is_vertical_l571_57108

/-- The angle of inclination of a line -/
def angle_of_inclination (l : Line2D) : ℝ := sorry

/-- A line is horizontal if its angle of inclination is 0 -/
def is_horizontal (l : Line2D) : Prop := angle_of_inclination l = 0

/-- Two lines are perpendicular if their angles of inclination sum to 90° -/
def are_perpendicular (l1 l2 : Line2D) : Prop :=
  angle_of_inclination l1 + angle_of_inclination l2 = 90

theorem perpendicular_to_horizontal_is_vertical (l1 l2 : Line2D) :
  is_horizontal l1 → are_perpendicular l1 l2 → angle_of_inclination l2 = 90 := by
  sorry

end perpendicular_to_horizontal_is_vertical_l571_57108


namespace original_polygon_sides_l571_57157

theorem original_polygon_sides (n : ℕ) : 
  (n + 1 - 2) * 180 = 1620 → n = 10 := by sorry

end original_polygon_sides_l571_57157


namespace function_equality_l571_57191

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom cond1 : ∀ x : ℝ, f x ≤ x
axiom cond2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y

-- State the theorem
theorem function_equality : ∀ x : ℝ, f x = x := by
  sorry

end function_equality_l571_57191


namespace icecream_cost_theorem_l571_57159

def chapati_count : ℕ := 16
def rice_count : ℕ := 5
def vegetable_count : ℕ := 7
def icecream_count : ℕ := 6

def chapati_cost : ℕ := 6
def rice_cost : ℕ := 45
def vegetable_cost : ℕ := 70

def total_paid : ℕ := 1015

theorem icecream_cost_theorem : 
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / icecream_count = 34 := by
  sorry

end icecream_cost_theorem_l571_57159


namespace diana_etienne_money_comparison_l571_57163

/-- Proves that Diana's money is 21.25% greater than Etienne's after euro appreciation --/
theorem diana_etienne_money_comparison :
  let initial_rate : ℝ := 1.25  -- 1 euro = 1.25 dollars
  let diana_dollars : ℝ := 600
  let etienne_euros : ℝ := 350
  let appreciation_rate : ℝ := 1.08  -- 8% appreciation
  let new_rate : ℝ := initial_rate * appreciation_rate
  let etienne_dollars : ℝ := etienne_euros * new_rate
  let difference_percent : ℝ := (diana_dollars - etienne_dollars) / etienne_dollars * 100
  difference_percent = 21.25 := by
sorry

end diana_etienne_money_comparison_l571_57163


namespace probability_one_red_one_white_l571_57188

/-- The probability of drawing 1 red ball and 1 white ball when drawing two balls with replacement 
    from a bag containing 2 red balls and 3 white balls is equal to 2/5. -/
theorem probability_one_red_one_white (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
  (h_total : total_balls = red_balls + white_balls)
  (h_red : red_balls = 2)
  (h_white : white_balls = 3) :
  (red_balls / total_balls) * (white_balls / total_balls) * 2 = 2 / 5 := by
  sorry

end probability_one_red_one_white_l571_57188


namespace next_perfect_square_sum_of_digits_l571_57143

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def begins_with_three_twos (n : ℕ) : Prop :=
  n ≥ 222000 ∧ n < 223000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem next_perfect_square_sum_of_digits :
  ∃ n : ℕ, is_perfect_square n ∧ 
           begins_with_three_twos n ∧
           (∀ m : ℕ, is_perfect_square m ∧ begins_with_three_twos m → n ≤ m) ∧
           sum_of_digits n = 18 :=
sorry

end next_perfect_square_sum_of_digits_l571_57143


namespace f_monotonicity_and_intersection_l571_57189

/-- The cubic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem f_monotonicity_and_intersection (a : ℝ) :
  (∀ x : ℝ, a ≥ 1/3 → Monotone (f a)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = a + 1 ∧ f a x = y ∧ f' a x * (-x) + y = 0) ∧
  (∃ x y : ℝ, x = -1 ∧ y = -a - 1 ∧ f a x = y ∧ f' a x * (-x) + y = 0) := by
  sorry

end f_monotonicity_and_intersection_l571_57189


namespace handshake_count_l571_57166

theorem handshake_count (n : ℕ) (total_handshakes : ℕ) : 
  n = 7 ∧ total_handshakes = n * (n - 1) / 2 → total_handshakes = 21 := by
  sorry

#check handshake_count

end handshake_count_l571_57166


namespace square_diagonals_equal_l571_57106

-- Define a structure for a parallelogram
structure Parallelogram :=
  (diagonals_equal : Bool)

-- Define a structure for a square that is a parallelogram
structure Square extends Parallelogram

-- State the theorem
theorem square_diagonals_equal (s : Square) : s.diagonals_equal = true := by
  sorry


end square_diagonals_equal_l571_57106


namespace solution_to_system_l571_57155

theorem solution_to_system (x y : ℝ) :
  x^5 + y^5 = 33 ∧ x + y = 3 →
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
sorry

end solution_to_system_l571_57155


namespace quadratic_equation_roots_l571_57136

theorem quadratic_equation_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 - (k-3)*x₁ - k + 1 = 0) ∧ 
  (x₂^2 - (k-3)*x₂ - k + 1 = 0) :=
sorry

end quadratic_equation_roots_l571_57136


namespace arctan_equation_solution_l571_57162

theorem arctan_equation_solution :
  ∀ x : ℝ, 2 * Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/x) = π/4 → x = -19/5 := by
  sorry

end arctan_equation_solution_l571_57162


namespace square_area_from_vertices_l571_57156

/-- The area of a square with adjacent vertices at (1,3) and (4,6) is 18 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (4, 6)
  let distance_squared := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
  distance_squared = 18 :=
by sorry

end square_area_from_vertices_l571_57156


namespace polyhedron_property_l571_57147

/-- Represents a convex polyhedron with the given properties -/
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  t : ℕ  -- Number of triangular faces
  s : ℕ  -- Number of square faces
  euler_formula : V - E + F = 2
  face_count : F = 42
  face_types : F = t + s
  edge_relation : E = (3 * t + 4 * s) / 2
  vertex_degree : 13 * V = 2 * E

/-- The main theorem to be proved -/
theorem polyhedron_property (p : ConvexPolyhedron) : 100 * 3 + 10 * 2 + p.V = 337 := by
  sorry

end polyhedron_property_l571_57147


namespace stratified_sampling_result_count_l571_57197

def junior_population : ℕ := 400
def senior_population : ℕ := 200
def total_sample_size : ℕ := 60

def proportional_allocation (total_pop : ℕ) (stratum_pop : ℕ) (sample_size : ℕ) : ℕ :=
  (stratum_pop * sample_size) / total_pop

theorem stratified_sampling_result_count :
  Nat.choose junior_population (proportional_allocation (junior_population + senior_population) junior_population total_sample_size) *
  Nat.choose senior_population (proportional_allocation (junior_population + senior_population) senior_population total_sample_size) =
  Nat.choose junior_population 40 * Nat.choose senior_population 20 :=
by sorry

end stratified_sampling_result_count_l571_57197


namespace smallest_cookie_count_l571_57146

theorem smallest_cookie_count : ∃ (x : ℕ), x > 0 ∧
  x % 6 = 5 ∧ x % 8 = 7 ∧ x % 9 = 2 ∧
  ∀ (y : ℕ), y > 0 → y % 6 = 5 → y % 8 = 7 → y % 9 = 2 → x ≤ y :=
by
  -- Proof goes here
  sorry

end smallest_cookie_count_l571_57146


namespace f_of_g_eight_l571_57129

def g (x : ℝ) : ℝ := 4 * x + 5

def f (x : ℝ) : ℝ := 6 * x - 11

theorem f_of_g_eight : f (g 8) = 211 := by
  sorry

end f_of_g_eight_l571_57129


namespace arithmetic_progression_sum_l571_57113

def arithmetic_progression (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_sum
  (a : ℕ → ℚ)
  (h_ap : arithmetic_progression a)
  (h_sum1 : a 1 + a 4 + a 7 = 45)
  (h_sum2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 27 :=
sorry

end arithmetic_progression_sum_l571_57113


namespace min_discount_factor_l571_57128

def cost_price : ℝ := 800
def marked_price : ℝ := 1200
def min_profit_margin : ℝ := 0.2

theorem min_discount_factor (x : ℝ) : 
  (cost_price * (1 + min_profit_margin) = marked_price * x) → x = 0.8 :=
by sorry

end min_discount_factor_l571_57128


namespace jellybean_problem_l571_57172

/-- Calculates the number of jellybeans removed after adding some back -/
def jellybeans_removed_after_adding_back (initial : ℕ) (removed : ℕ) (added_back : ℕ) (final : ℕ) : ℕ :=
  initial - removed + added_back - final

theorem jellybean_problem (initial : ℕ) (removed : ℕ) (added_back : ℕ) (final : ℕ)
  (h1 : initial = 37)
  (h2 : removed = 15)
  (h3 : added_back = 5)
  (h4 : final = 23) :
  jellybeans_removed_after_adding_back initial removed added_back final = 4 := by
  sorry

end jellybean_problem_l571_57172


namespace problem_statement_l571_57105

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 4)
  (h2 : x * y = 64) :
  (x + y) / 2 = (64^(1/(3+Real.sqrt 3)) + 64^((2+Real.sqrt 3)/(3+Real.sqrt 3))) / 2 := by
sorry

end problem_statement_l571_57105


namespace apple_sale_discrepancy_l571_57111

/-- Represents the number of apples sold for one cent by the first vendor -/
def apples_per_cent_vendor1 : ℕ := 3

/-- Represents the number of apples sold for one cent by the second vendor -/
def apples_per_cent_vendor2 : ℕ := 2

/-- Represents the number of unsold apples each vendor had -/
def unsold_apples_per_vendor : ℕ := 30

/-- Represents the total number of apples to be sold -/
def total_apples : ℕ := 2 * unsold_apples_per_vendor

/-- Represents the number of apples sold for two cents by the friend -/
def apples_per_two_cents_friend : ℕ := 5

/-- Calculates the revenue when apples are sold individually by vendors -/
def revenue_individual : ℕ := 
  (unsold_apples_per_vendor / apples_per_cent_vendor1) + 
  (unsold_apples_per_vendor / apples_per_cent_vendor2)

/-- Calculates the revenue when apples are sold by the friend -/
def revenue_friend : ℕ := 
  2 * (total_apples / apples_per_two_cents_friend)

theorem apple_sale_discrepancy : 
  revenue_individual = revenue_friend + 1 := by
  sorry

end apple_sale_discrepancy_l571_57111


namespace value_of_a_l571_57132

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 2*(x + 1)

-- State the theorem
theorem value_of_a (a : ℝ) (h : f a = 3) : a = 2 ∨ a = -2 := by
  sorry

end value_of_a_l571_57132


namespace x_value_proof_l571_57174

theorem x_value_proof (x : ℝ) : 
  (⌊x⌋ + ⌈x⌉ = 7) ∧ (10 ≤ 3*x - 5 ∧ 3*x - 5 ≤ 13) → 3 < x ∧ x < 4 :=
by sorry

end x_value_proof_l571_57174


namespace hyperbola_standard_form_l571_57198

/-- Given a hyperbola with equation x²/m - y²/(m+6) = 1 where m > 0,
    and its conjugate axis is twice the length of its transverse axis,
    prove that the standard form of the hyperbola's equation is x²/2 - y²/8 = 1 -/
theorem hyperbola_standard_form (m : ℝ) (h_m : m > 0) 
  (h_eq : ∀ x y : ℝ, x^2 / m - y^2 / (m + 6) = 1)
  (h_axis : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = m ∧ b^2 = m + 6 ∧ b = 2*a) :
  ∀ x y : ℝ, x^2 / 2 - y^2 / 8 = 1 := by
  sorry

end hyperbola_standard_form_l571_57198


namespace p_sufficient_not_necessary_for_q_l571_57121

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end p_sufficient_not_necessary_for_q_l571_57121


namespace unique_perimeter_l571_57104

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  AB : ℕ+
  BC : ℕ+
  CD : ℕ+
  AD : ℕ+
  right_angle_B : True
  right_angle_C : True
  AB_equals_3 : AB = 3
  CD_equals_AD : CD = AD

/-- The perimeter of a SpecialQuadrilateral -/
def perimeter (q : SpecialQuadrilateral) : ℕ :=
  q.AB + q.BC + q.CD + q.AD

/-- Theorem stating that there's exactly one valid perimeter less than 2015 -/
theorem unique_perimeter :
  ∃! p : ℕ, p < 2015 ∧ ∃ q : SpecialQuadrilateral, perimeter q = p :=
by sorry

end unique_perimeter_l571_57104


namespace barney_towel_count_l571_57179

/-- The number of towels Barney owns -/
def num_towels : ℕ := 18

/-- The number of towels Barney uses per day -/
def towels_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days Barney can use clean towels before running out -/
def days_before_running_out : ℕ := 9

/-- Theorem stating that Barney owns 18 towels -/
theorem barney_towel_count : 
  num_towels = towels_per_day * days_before_running_out :=
by sorry

end barney_towel_count_l571_57179


namespace intersection_A_B_equals_open_interval_2_3_l571_57103

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

-- Define the open interval (2, 3)
def open_interval_2_3 : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B_equals_open_interval_2_3 : A ∩ B = open_interval_2_3 := by
  sorry

end intersection_A_B_equals_open_interval_2_3_l571_57103


namespace seafood_price_proof_l571_57126

/-- The regular price of seafood given the sale price and discount -/
def regular_price (sale_price : ℚ) (discount_percent : ℚ) : ℚ :=
  sale_price / (1 - discount_percent)

/-- The price for a given weight of seafood at the regular price -/
def price_for_weight (price_per_unit : ℚ) (weight : ℚ) : ℚ :=
  price_per_unit * weight

theorem seafood_price_proof :
  let sale_price_per_pack : ℚ := 4
  let pack_weight : ℚ := 3/4
  let discount_percent : ℚ := 3/4
  let target_weight : ℚ := 3/2

  let regular_price_per_pack := regular_price sale_price_per_pack discount_percent
  let regular_price_per_pound := regular_price_per_pack / pack_weight
  
  price_for_weight regular_price_per_pound target_weight = 32 := by
  sorry

end seafood_price_proof_l571_57126


namespace chicks_increase_l571_57138

theorem chicks_increase (first_day : ℕ) (second_day : ℕ) : first_day = 23 → second_day = 12 → first_day + second_day = 35 := by
  sorry

end chicks_increase_l571_57138


namespace f_increasing_and_odd_l571_57171

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end f_increasing_and_odd_l571_57171


namespace milestone_number_l571_57142

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  ones : ℕ
  h1 : tens ≥ 1 ∧ tens ≤ 9
  h2 : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to a natural number -/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : ℕ := 10 * n.tens + n.ones

/-- Theorem: Given the conditions of the problem, the initial number must be 16 -/
theorem milestone_number (initial : TwoDigitNumber) 
  (h1 : initial.toNat + initial.toNat = 100 * initial.ones + initial.tens + 100 * initial.tens + initial.ones) :
  initial.tens = 1 ∧ initial.ones = 6 := by
  sorry

#check milestone_number

end milestone_number_l571_57142


namespace xy_equals_one_l571_57167

theorem xy_equals_one (x y : ℝ) (h : x + y = 1/x + 1/y ∧ x + y ≠ 0) : x * y = 1 := by
  sorry

end xy_equals_one_l571_57167


namespace derivatives_at_zero_l571_57140

open Function Real

/-- A function f satisfying the given conditions -/
def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → f (1 / n) = n^2 / (n^2 + 1)

/-- The theorem statement -/
theorem derivatives_at_zero
  (f : ℝ → ℝ)
  (h_smooth : ContDiff ℝ ⊤ f)
  (h_cond : f_condition f) :
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  deriv^[2] f 0 = -2 ∧
  ∀ k : ℕ, k ≥ 3 → deriv^[k] f 0 = 0 :=
by sorry

end derivatives_at_zero_l571_57140


namespace cauchy_schwarz_inequality_2d_l571_57190

theorem cauchy_schwarz_inequality_2d (a₁ a₂ b₁ b₂ : ℝ) :
  a₁ * b₁ + a₂ * b₂ ≤ Real.sqrt (a₁^2 + a₂^2) * Real.sqrt (b₁^2 + b₂^2) := by
  sorry

end cauchy_schwarz_inequality_2d_l571_57190


namespace bowling_team_weight_l571_57122

theorem bowling_team_weight (original_players : ℕ) (original_avg_weight : ℝ)
  (new_players : ℕ) (second_player_weight : ℝ) (new_avg_weight : ℝ)
  (h1 : original_players = 7)
  (h2 : original_avg_weight = 112)
  (h3 : new_players = 2)
  (h4 : second_player_weight = 60)
  (h5 : new_avg_weight = 106) :
  let total_players := original_players + new_players
  let original_total_weight := original_players * original_avg_weight
  let new_total_weight := total_players * new_avg_weight
  let first_player_weight := new_total_weight - original_total_weight - second_player_weight
  first_player_weight = 110 := by
sorry

end bowling_team_weight_l571_57122


namespace quadratic_equation_solution_l571_57180

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end quadratic_equation_solution_l571_57180


namespace factorization_problem_1_factorization_problem_2_l571_57193

-- Problem 1
theorem factorization_problem_1 (x : ℝ) :
  x^4 - 16 = (x-2)*(x+2)*(x^2+4) := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  -9*x^2*y + 12*x*y^2 - 4*y^3 = -y*(3*x-2*y)^2 := by sorry

end factorization_problem_1_factorization_problem_2_l571_57193


namespace simplify_square_root_l571_57192

theorem simplify_square_root (x y : ℝ) (h : x * y < 0) :
  x * Real.sqrt (-y / x^2) = Real.sqrt (-y) := by
  sorry

end simplify_square_root_l571_57192


namespace exponential_function_inequality_l571_57102

theorem exponential_function_inequality (m n : ℝ) : 
  let a : ℝ := (Real.sqrt 5 - Real.sqrt 2) / 2
  let f : ℝ → ℝ := fun x ↦ a^x
  0 < a ∧ a < 1 → f m > f n → m < n := by sorry

end exponential_function_inequality_l571_57102


namespace largest_angle_in_pentagon_l571_57154

theorem largest_angle_in_pentagon (P Q R S T : ℝ) : 
  P = 70 → 
  Q = 100 → 
  R = S → 
  T = 3 * R - 25 → 
  P + Q + R + S + T = 540 → 
  max P (max Q (max R (max S T))) = 212 :=
sorry

end largest_angle_in_pentagon_l571_57154


namespace cylinder_cone_height_relation_l571_57131

/-- Given a right cylinder and a cone with equal base radii, volumes, surface areas, 
    and heights in the ratio of 1:3, prove that the height of the cylinder 
    is 4/5 of the base radius. -/
theorem cylinder_cone_height_relation 
  (r : ℝ) -- base radius
  (h_cyl : ℝ) -- height of cylinder
  (h_cone : ℝ) -- height of cone
  (h_ratio : h_cone = 3 * h_cyl) -- height ratio condition
  (h_vol : π * r^2 * h_cyl = 1/3 * π * r^2 * h_cone) -- equal volumes
  (h_area : 2 * π * r^2 + 2 * π * r * h_cyl = 
            π * r^2 + π * r * Real.sqrt (r^2 + h_cone^2)) -- equal surface areas
  : h_cyl = 4/5 * r :=
by sorry

end cylinder_cone_height_relation_l571_57131


namespace cubic_function_property_l571_57124

/-- Given a cubic function f(x) = ax^3 + bx + 1 where f(-2) = 2, prove that f(2) = 0 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 1
  f (-2) = 2 → f 2 = 0 := by
sorry

end cubic_function_property_l571_57124


namespace num_solutions_for_quadratic_l571_57196

/-- A line y = x + a that does not pass through the second quadrant -/
structure Line where
  a : ℝ
  not_in_second_quadrant : a ≤ 0

/-- The number of real solutions for a quadratic equation -/
inductive NumRealSolutions
  | zero
  | one
  | two

/-- The theorem stating the number of real solutions for the equation ax^2 + 2x + 1 = 0 -/
theorem num_solutions_for_quadratic (l : Line) :
  ∃ n : NumRealSolutions, (n = NumRealSolutions.one ∨ n = NumRealSolutions.two) ∧
  (∃ x : ℝ, l.a * x^2 + 2*x + 1 = 0) ∧
  (n = NumRealSolutions.two → ∃ x y : ℝ, x ≠ y ∧ l.a * x^2 + 2*x + 1 = 0 ∧ l.a * y^2 + 2*y + 1 = 0) :=
sorry

end num_solutions_for_quadratic_l571_57196


namespace prove_train_car_capacity_l571_57145

/-- The number of passengers a 747 airplane can carry -/
def airplane_capacity : ℕ := 366

/-- The number of cars in the train -/
def train_cars : ℕ := 16

/-- The additional passengers a train can carry compared to 2 airplanes -/
def additional_passengers : ℕ := 228

/-- The number of passengers a single train car can carry -/
def train_car_capacity : ℕ := 60

theorem prove_train_car_capacity : 
  train_car_capacity * train_cars = 2 * airplane_capacity + additional_passengers :=
sorry

end prove_train_car_capacity_l571_57145


namespace hazel_walk_l571_57144

/-- The distance Hazel walked in the first hour -/
def first_hour_distance : ℝ := 2

/-- The distance Hazel walked in the second hour -/
def second_hour_distance (x : ℝ) : ℝ := 2 * x

/-- The total distance Hazel walked in 2 hours -/
def total_distance : ℝ := 6

theorem hazel_walk :
  first_hour_distance + second_hour_distance first_hour_distance = total_distance :=
by sorry

end hazel_walk_l571_57144


namespace cos_50_tan_40_equals_sqrt_3_l571_57135

theorem cos_50_tan_40_equals_sqrt_3 : 
  4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end cos_50_tan_40_equals_sqrt_3_l571_57135


namespace power_mod_thirteen_l571_57153

theorem power_mod_thirteen : 7^1234 ≡ 4 [ZMOD 13] := by
  sorry

end power_mod_thirteen_l571_57153


namespace mutually_exclusive_but_not_complementary_l571_57141

-- Define the sample space
def SampleSpace := Finset (Fin 4 × Fin 4)

-- Define the event of selecting exactly one girl
def exactlyOneGirl (s : SampleSpace) : Prop :=
  (s.card = 2) ∧ (s.filter (λ p => p.1 > 1 ∨ p.2 > 1)).card = 1

-- Define the event of selecting exactly two girls
def exactlyTwoGirls (s : SampleSpace) : Prop :=
  (s.card = 2) ∧ (s.filter (λ p => p.1 > 1 ∧ p.2 > 1)).card = 2

-- State the theorem
theorem mutually_exclusive_but_not_complementary :
  (∀ s : SampleSpace, ¬(exactlyOneGirl s ∧ exactlyTwoGirls s)) ∧
  (∃ s : SampleSpace, ¬(exactlyOneGirl s ∨ exactlyTwoGirls s)) :=
sorry

end mutually_exclusive_but_not_complementary_l571_57141


namespace purple_valley_skirts_l571_57187

def azure_skirts : ℕ := 60

def seafoam_skirts (azure : ℕ) : ℕ := (2 * azure) / 3

def purple_skirts (seafoam : ℕ) : ℕ := seafoam / 4

theorem purple_valley_skirts : 
  purple_skirts (seafoam_skirts azure_skirts) = 10 := by
  sorry

end purple_valley_skirts_l571_57187


namespace water_depth_calculation_l571_57127

-- Define the heights of Ron and Dean
def ron_height : ℝ := 13
def dean_height : ℝ := ron_height + 4

-- Define the maximum depth at high tide
def max_depth : ℝ := 15 * dean_height

-- Define the current tide percentage and current percentage
def tide_percentage : ℝ := 0.75
def current_percentage : ℝ := 0.20

-- Theorem statement
theorem water_depth_calculation :
  let current_tide_depth := tide_percentage * max_depth
  let additional_depth := current_percentage * current_tide_depth
  current_tide_depth + additional_depth = 229.5 := by
  sorry

end water_depth_calculation_l571_57127


namespace meals_given_away_l571_57177

theorem meals_given_away (initial_meals : ℕ) (additional_meals : ℕ) (meals_left : ℕ) : 
  initial_meals = 113 → additional_meals = 50 → meals_left = 78 → 
  initial_meals + additional_meals - meals_left = 85 := by
  sorry

end meals_given_away_l571_57177


namespace power_equality_l571_57130

theorem power_equality (n : ℕ) : 3^n = 3^2 * 9^4 * 81^3 → n = 22 := by
  sorry

end power_equality_l571_57130


namespace corvette_trip_speed_l571_57186

theorem corvette_trip_speed (total_distance : ℝ) (average_speed : ℝ) 
  (h1 : total_distance = 640)
  (h2 : average_speed = 40) : ℝ :=
  let first_half_distance := total_distance / 2
  let second_half_time_ratio := 3
  let first_half_speed := 
    (2 * total_distance * average_speed) / (total_distance + 2 * first_half_distance)
  have h3 : first_half_speed = 80 := by sorry
  first_half_speed

#check corvette_trip_speed

end corvette_trip_speed_l571_57186


namespace train_length_l571_57176

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 179.99999999999997) (h2 : time = 3) :
  speed * time = 540 := by
  sorry

end train_length_l571_57176


namespace rectangle_perimeter_l571_57119

/-- The perimeter of a rectangle with length 100 and breadth 500 is 1200. -/
theorem rectangle_perimeter : 
  ∀ (length breadth perimeter : ℕ), 
    length = 100 → 
    breadth = 500 → 
    perimeter = 2 * (length + breadth) → 
    perimeter = 1200 := by
  sorry

end rectangle_perimeter_l571_57119


namespace min_towns_for_22_routes_l571_57158

/-- A graph representing a country's airline network -/
structure AirlineNetwork where
  towns : Finset ℕ
  connections : towns → towns → Bool
  paid_direction : towns → towns → Bool

/-- The number of free routes between two towns in an airline network -/
def free_routes (g : AirlineNetwork) (a b : g.towns) : ℕ :=
  sorry

/-- The theorem stating that the minimum number of towns for 22 free routes is 7 -/
theorem min_towns_for_22_routes :
  ∃ (g : AirlineNetwork) (a b : g.towns),
    free_routes g a b = 22 ∧
    g.towns.card = 7 ∧
    (∀ (h : AirlineNetwork) (x y : h.towns),
      free_routes h x y = 22 → h.towns.card ≥ 7) :=
  sorry

end min_towns_for_22_routes_l571_57158


namespace amy_current_age_l571_57114

/-- Given that Mark is 7 years older than Amy and Mark will be 27 years old in 5 years,
    prove that Amy's current age is 15 years old. -/
theorem amy_current_age :
  ∀ (mark_age amy_age : ℕ),
  mark_age = amy_age + 7 →
  mark_age + 5 = 27 →
  amy_age = 15 :=
by
  sorry

end amy_current_age_l571_57114


namespace walters_age_l571_57181

theorem walters_age (walter_age_1994 : ℝ) (grandmother_age_1994 : ℝ) : 
  walter_age_1994 = grandmother_age_1994 / 3 →
  (1994 - walter_age_1994) + (1994 - grandmother_age_1994) = 3750 →
  walter_age_1994 + 6 = 65.5 := by
sorry

end walters_age_l571_57181


namespace largest_multiple_of_15_under_400_l571_57173

theorem largest_multiple_of_15_under_400 : 
  ∀ n : ℕ, n * 15 < 400 → n * 15 ≤ 390 :=
by
  sorry

end largest_multiple_of_15_under_400_l571_57173


namespace intersection_S_complement_T_l571_57150

-- Define the universal set U
def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}

-- Define set S
def S : Set ℕ := {1, 2, 4, 5}

-- Define set T
def T : Set ℕ := {3, 4, 5, 7}

-- Theorem statement
theorem intersection_S_complement_T : S ∩ (U \ T) = {1, 2} := by sorry

end intersection_S_complement_T_l571_57150


namespace sum_of_squares_zero_implies_sum_l571_57100

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 5)^2 + (y - 6)^2 + (z - 7)^2 + 2 = 2 →
  x + y + z = 18 := by
  sorry

end sum_of_squares_zero_implies_sum_l571_57100


namespace evaluate_expression_l571_57199

theorem evaluate_expression (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (4 * y^2) * (1 / (2*y)^3) = 9 * y^2 := by
sorry

end evaluate_expression_l571_57199


namespace sum_of_smallest_multiples_l571_57185

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def smallest_two_digit_multiple_of_5 (a : ℕ) : Prop :=
  is_two_digit a ∧ 5 ∣ a ∧ ∀ m : ℕ, is_two_digit m → 5 ∣ m → a ≤ m

def smallest_three_digit_multiple_of_7 (b : ℕ) : Prop :=
  is_three_digit b ∧ 7 ∣ b ∧ ∀ m : ℕ, is_three_digit m → 7 ∣ m → b ≤ m

theorem sum_of_smallest_multiples (a b : ℕ) :
  smallest_two_digit_multiple_of_5 a →
  smallest_three_digit_multiple_of_7 b →
  a + b = 115 := by
  sorry

end sum_of_smallest_multiples_l571_57185


namespace unique_determination_by_gcds_l571_57109

theorem unique_determination_by_gcds :
  ∀ X : ℕ, X ≤ 100 →
  ∃ (M N : Fin 7 → ℕ), (∀ i, M i < 100 ∧ N i < 100) ∧
    ∀ Y : ℕ, Y ≤ 100 →
      (∀ i : Fin 7, Nat.gcd (X + M i) (N i) = Nat.gcd (Y + M i) (N i)) →
      X = Y :=
by sorry

end unique_determination_by_gcds_l571_57109


namespace hannahs_peppers_l571_57195

theorem hannahs_peppers (green_peppers red_peppers : ℝ) 
  (h1 : green_peppers = 0.3333333333333333)
  (h2 : red_peppers = 0.3333333333333333) :
  green_peppers + red_peppers = 0.6666666666666666 := by
  sorry

end hannahs_peppers_l571_57195


namespace intersection_of_A_and_B_l571_57115

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end intersection_of_A_and_B_l571_57115


namespace class_overall_score_l571_57182

/-- Calculates the overall score for a class based on four aspects --/
def calculate_overall_score (study_score hygiene_score discipline_score activity_score : ℝ) : ℝ :=
  0.4 * study_score + 0.25 * hygiene_score + 0.25 * discipline_score + 0.1 * activity_score

/-- Theorem stating that the overall score for the given class is 84 --/
theorem class_overall_score :
  calculate_overall_score 85 90 80 75 = 84 := by
  sorry

#eval calculate_overall_score 85 90 80 75

end class_overall_score_l571_57182


namespace smallest_absolute_value_of_z_l571_57118

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 10) + Complex.abs (z + 3*I) = 17) :
  ∃ (w : ℂ), Complex.abs (z - 10) + Complex.abs (z + 3*I) = 17 ∧ Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 30/17 :=
by sorry

end smallest_absolute_value_of_z_l571_57118


namespace alligators_hiding_l571_57120

/-- Given a zoo cage with alligators, prove the number of hiding alligators -/
theorem alligators_hiding (total_alligators : ℕ) (not_hiding : ℕ) 
  (h1 : total_alligators = 75)
  (h2 : not_hiding = 56) :
  total_alligators - not_hiding = 19 := by
  sorry

#check alligators_hiding

end alligators_hiding_l571_57120
