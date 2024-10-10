import Mathlib

namespace exam_boys_count_total_boys_is_120_l117_11707

/-- The number of boys who passed the examination -/
def passed_boys : ℕ := 100

/-- The average marks of all boys -/
def total_average : ℚ := 35

/-- The average marks of passed boys -/
def passed_average : ℚ := 39

/-- The average marks of failed boys -/
def failed_average : ℚ := 15

/-- The total number of boys who took the examination -/
def total_boys : ℕ := sorry

theorem exam_boys_count :
  total_boys = passed_boys +
    (total_boys * total_average - passed_boys * passed_average) / (failed_average - total_average) :=
by sorry

theorem total_boys_is_120 : total_boys = 120 :=
by sorry

end exam_boys_count_total_boys_is_120_l117_11707


namespace purely_imaginary_z_equals_one_l117_11751

theorem purely_imaginary_z_equals_one (x : ℝ) :
  let z : ℂ := (x + (x^2 - 1) * Complex.I) / Complex.I
  (∃ (y : ℝ), z = Complex.I * y) → z = 1 := by
  sorry

end purely_imaginary_z_equals_one_l117_11751


namespace no_perfect_square_natural_l117_11701

theorem no_perfect_square_natural (n : ℕ) : ¬∃ (m : ℕ), n^5 - 5*n^3 + 4*n + 7 = m^2 := by
  sorry

end no_perfect_square_natural_l117_11701


namespace negation_of_no_left_handed_in_chess_club_l117_11731

-- Define the universe of students
variable (Student : Type)

-- Define predicates for left-handedness and chess club membership
variable (isLeftHanded : Student → Prop)
variable (isInChessClub : Student → Prop)

-- State the theorem
theorem negation_of_no_left_handed_in_chess_club :
  (¬ ∀ (s : Student), isLeftHanded s → ¬ isInChessClub s) ↔
  (∃ (s : Student), isLeftHanded s ∧ isInChessClub s) :=
by sorry

end negation_of_no_left_handed_in_chess_club_l117_11731


namespace fewer_green_marbles_percentage_l117_11717

/-- Proves that the percentage of fewer green marbles compared to yellow marbles is 50% -/
theorem fewer_green_marbles_percentage (total : ℕ) (white yellow green red : ℕ) :
  total = 50 ∧
  white = total / 2 ∧
  yellow = 12 ∧
  red = 7 ∧
  green = total - (white + yellow + red) →
  (yellow - green) / yellow * 100 = 50 := by
  sorry

end fewer_green_marbles_percentage_l117_11717


namespace moles_NaHCO3_equals_moles_HCl_l117_11715

/-- Represents a chemical species in the reaction -/
inductive Species
| NaHCO3
| HCl
| NaCl
| H2O
| CO2

/-- Represents the balanced chemical equation -/
def balanced_equation (reactants products : Species → ℕ) : Prop :=
  reactants Species.NaHCO3 = 1 ∧
  reactants Species.HCl = 1 ∧
  products Species.NaCl = 1 ∧
  products Species.H2O = 1 ∧
  products Species.CO2 = 1

/-- The number of moles of HCl given -/
def moles_HCl : ℕ := 3

/-- The number of moles of products formed -/
def moles_products : Species → ℕ
| Species.NaCl => 3
| Species.H2O => 3
| Species.CO2 => 3
| _ => 0

/-- Theorem stating that the number of moles of NaHCO3 required equals the number of moles of HCl -/
theorem moles_NaHCO3_equals_moles_HCl 
  (eq : balanced_equation (λ _ => 1) (λ _ => 1))
  (prod : ∀ s, moles_products s = moles_HCl ∨ moles_products s = 0) :
  moles_HCl = moles_HCl := by sorry

end moles_NaHCO3_equals_moles_HCl_l117_11715


namespace inscribed_sphere_radius_bound_l117_11774

/-- A tetrahedron with an inscribed sphere --/
structure Tetrahedron :=
  (r : ℝ) -- radius of inscribed sphere
  (a b : ℝ) -- lengths of a pair of opposite edges
  (r_pos : r > 0)
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- Theorem: The radius of the inscribed sphere is less than ab/(2(a+b)) --/
theorem inscribed_sphere_radius_bound (t : Tetrahedron) : t.r < (t.a * t.b) / (2 * (t.a + t.b)) := by
  sorry

end inscribed_sphere_radius_bound_l117_11774


namespace cube_root_simplification_l117_11769

theorem cube_root_simplification : 
  (54880000 : ℝ)^(1/3) = 140 * 20^(1/3) := by sorry

end cube_root_simplification_l117_11769


namespace complement_intersection_theorem_l117_11702

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem complement_intersection_theorem : 
  (U \ A) ∩ (U \ B) = {4, 8} := by sorry

end complement_intersection_theorem_l117_11702


namespace sum_invested_is_15000_l117_11753

/-- The sum invested that satisfies the given conditions -/
def find_sum (interest_rate_high : ℚ) (interest_rate_low : ℚ) (time : ℚ) (interest_difference : ℚ) : ℚ :=
  interest_difference / (time * (interest_rate_high - interest_rate_low))

/-- Theorem stating that the sum invested is 15000 given the problem conditions -/
theorem sum_invested_is_15000 :
  find_sum (15/100) (12/100) 2 900 = 15000 := by
  sorry

#eval find_sum (15/100) (12/100) 2 900

end sum_invested_is_15000_l117_11753


namespace floor_plus_double_eq_sixteen_l117_11733

theorem floor_plus_double_eq_sixteen (r : ℝ) : (⌊r⌋ : ℝ) + 2 * r = 16 ↔ r = (5.5 : ℝ) := by
  sorry

end floor_plus_double_eq_sixteen_l117_11733


namespace degree_of_specific_monomial_l117_11789

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

/-- The monomial -1/4 * π * x^2 * y^3 has degree 5 -/
theorem degree_of_specific_monomial :
  degree_of_monomial 2 3 = 5 := by sorry

end degree_of_specific_monomial_l117_11789


namespace award_distribution_l117_11721

theorem award_distribution (n : ℕ) (k : ℕ) :
  n = 6 ∧ k = 3 →
  (Finset.univ.powerset.filter (λ s : Finset (Fin n) => s.card = 2)).card.choose k = 15 :=
by sorry

end award_distribution_l117_11721


namespace largest_number_with_equal_costs_l117_11735

/-- Calculates the sum of squares of decimal digits for a given number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates the number of 1's in the binary representation of a given number -/
def count_ones_in_binary (n : ℕ) : ℕ := sorry

/-- Theorem stating that 503 is the largest number less than 2000 where 
    sum of squares of digits equals the number of 1's in binary representation -/
theorem largest_number_with_equal_costs : 
  ∀ n : ℕ, n < 2000 → n > 503 → 
    sum_of_squares_of_digits n ≠ count_ones_in_binary n := by
  sorry

end largest_number_with_equal_costs_l117_11735


namespace flooring_boxes_needed_l117_11743

def room_length : ℝ := 16
def room_width : ℝ := 20
def flooring_per_box : ℝ := 10
def flooring_laid : ℝ := 250

theorem flooring_boxes_needed : 
  ⌈(room_length * room_width - flooring_laid) / flooring_per_box⌉ = 7 := by
  sorry

end flooring_boxes_needed_l117_11743


namespace train_speed_l117_11716

/-- The speed of a train crossing a platform of equal length -/
theorem train_speed (train_length platform_length : ℝ) (crossing_time : ℝ) : 
  train_length = platform_length → 
  train_length = 600 → 
  crossing_time = 1 / 60 → 
  (train_length + platform_length) / crossing_time / 1000 = 72 :=
by
  sorry

#check train_speed

end train_speed_l117_11716


namespace inequality_proof_l117_11740

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_proof_l117_11740


namespace intersection_of_sets_l117_11767

theorem intersection_of_sets :
  let P : Set ℕ := {1, 3, 5}
  let Q : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}
  P ∩ Q = {3, 5} := by
sorry

end intersection_of_sets_l117_11767


namespace linear_equation_solution_l117_11741

theorem linear_equation_solution (a b : ℝ) (h1 : a - b = 0) (h2 : a ≠ 0) :
  ∃! x : ℝ, a * x + b = 0 ∧ x = -1 := by
  sorry

end linear_equation_solution_l117_11741


namespace abs_5x_minus_2_zero_l117_11744

theorem abs_5x_minus_2_zero (x : ℚ) : |5*x - 2| = 0 ↔ x = 2/5 := by
  sorry

end abs_5x_minus_2_zero_l117_11744


namespace school_trip_students_l117_11738

/-- The number of students in a school given the number of classrooms, bus seats, and buses needed for a trip -/
theorem school_trip_students (classrooms : ℕ) (seats_per_bus : ℕ) (buses_needed : ℕ) 
  (h1 : classrooms = 87)
  (h2 : seats_per_bus = 2)
  (h3 : buses_needed = 29)
  (h4 : ∀ c1 c2 : ℕ, c1 < classrooms → c2 < classrooms → 
        (seats_per_bus * buses_needed) % classrooms = 0) :
  seats_per_bus * buses_needed * classrooms = 5046 := by
sorry

end school_trip_students_l117_11738


namespace triangle_side_product_range_l117_11763

theorem triangle_side_product_range (x y : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ y) 
  (h3 : y < x + 1) : 
  let t := max (1/x) (max (x/y) y) * min (1/x) (min (x/y) y)
  1 ≤ t ∧ t < (1 + Real.sqrt 5) / 2 := by
  sorry

end triangle_side_product_range_l117_11763


namespace swimmer_speed_is_five_l117_11749

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  manSpeed : ℝ  -- Speed of the man in still water (km/h)
  streamSpeed : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.manSpeed + s.streamSpeed else s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 5 km/h. -/
theorem swimmer_speed_is_five 
  (s : SwimmerSpeed)
  (h1 : effectiveSpeed s true = 30 / 5)  -- Downstream condition
  (h2 : effectiveSpeed s false = 20 / 5) -- Upstream condition
  : s.manSpeed = 5 := by
  sorry

#check swimmer_speed_is_five

end swimmer_speed_is_five_l117_11749


namespace square_of_difference_l117_11729

theorem square_of_difference (a b : ℝ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by
  sorry

end square_of_difference_l117_11729


namespace rectangle_square_length_difference_l117_11720

/-- Given a square and a rectangle with specific perimeter and width relationships,
    prove that the length of the rectangle is 4 centimeters longer than the side of the square. -/
theorem rectangle_square_length_difference
  (s : ℝ) -- side length of the square
  (l w : ℝ) -- length and width of the rectangle
  (h1 : 2 * (l + w) = 4 * s + 4) -- perimeter relationship
  (h2 : w = s - 2) -- width relationship
  : l = s + 4 := by
  sorry

end rectangle_square_length_difference_l117_11720


namespace quadratic_inequality_always_true_l117_11703

theorem quadratic_inequality_always_true (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2) :=
by sorry

end quadratic_inequality_always_true_l117_11703


namespace lawn_mowing_earnings_l117_11785

theorem lawn_mowing_earnings (total_lawns : ℕ) (forgotten_lawns : ℕ) (total_earned : ℕ) :
  total_lawns = 12 →
  forgotten_lawns = 8 →
  total_earned = 36 →
  (total_earned : ℚ) / ((total_lawns - forgotten_lawns) : ℚ) = 9 :=
by sorry

end lawn_mowing_earnings_l117_11785


namespace cake_frosting_time_difference_l117_11783

/-- The time difference for frosting 10 cakes between normal and sprained wrist conditions -/
theorem cake_frosting_time_difference 
  (normal_time : ℕ) 
  (sprained_time : ℕ) 
  (num_cakes : ℕ) 
  (h1 : normal_time = 5)
  (h2 : sprained_time = 8)
  (h3 : num_cakes = 10) : 
  (sprained_time * num_cakes) - (normal_time * num_cakes) = 30 := by
  sorry

#check cake_frosting_time_difference

end cake_frosting_time_difference_l117_11783


namespace nested_percentage_calculation_l117_11752

-- Define the initial amount
def initial_amount : ℝ := 3000

-- Define the percentages
def percent_1 : ℝ := 0.20
def percent_2 : ℝ := 0.35
def percent_3 : ℝ := 0.05

-- State the theorem
theorem nested_percentage_calculation :
  percent_3 * (percent_2 * (percent_1 * initial_amount)) = 10.50 := by
  sorry

end nested_percentage_calculation_l117_11752


namespace team_total_score_l117_11794

def team_score (connor_score : ℕ) (amy_score : ℕ) (jason_score : ℕ) : ℕ :=
  connor_score + amy_score + jason_score

theorem team_total_score :
  ∀ (connor_score amy_score jason_score : ℕ),
    connor_score = 2 →
    amy_score = connor_score + 4 →
    jason_score = 2 * amy_score →
    team_score connor_score amy_score jason_score = 20 :=
by
  sorry

end team_total_score_l117_11794


namespace june_design_white_tiles_l117_11706

/-- Calculates the number of white tiles in June's design -/
theorem june_design_white_tiles :
  let total_tiles : ℕ := 20
  let yellow_tiles : ℕ := 3
  let blue_tiles : ℕ := yellow_tiles + 1
  let purple_tiles : ℕ := 6
  let colored_tiles : ℕ := yellow_tiles + blue_tiles + purple_tiles
  let white_tiles : ℕ := total_tiles - colored_tiles
  white_tiles = 7 := by
  sorry

end june_design_white_tiles_l117_11706


namespace vector_square_difference_l117_11709

theorem vector_square_difference (a b : ℝ × ℝ) (h1 : a + b = (-3, 6)) (h2 : a - b = (-3, 2)) :
  (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 := by
  sorry

end vector_square_difference_l117_11709


namespace perpendicular_sufficient_not_necessary_l117_11734

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)

-- Define the "within" relation for a line being in a plane
variable (within : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_sufficient_not_necessary 
  (l m n : Line) (α : Plane)
  (m_in_α : within m α) (n_in_α : within n α) :
  (∀ l m n α, perp_plane l α → perp_line l m ∧ perp_line l n) ∧
  ¬(∀ l m n α, perp_line l m ∧ perp_line l n → perp_plane l α) :=
sorry

end perpendicular_sufficient_not_necessary_l117_11734


namespace largest_hexagon_angle_l117_11723

-- Define the hexagon's properties
def is_valid_hexagon (angles : List ℕ) : Prop :=
  angles.length = 6 ∧
  angles.sum = 720 ∧
  ∃ (a d : ℕ), angles = [a, a + d, a + 2*d, a + 3*d, a + 4*d, a + 5*d] ∧
  ∀ x ∈ angles, 0 < x ∧ x < 180

-- Theorem statement
theorem largest_hexagon_angle (angles : List ℕ) :
  is_valid_hexagon angles →
  (∀ x ∈ angles, x ≤ 175) ∧
  (∃ x ∈ angles, x = 175) :=
by sorry

end largest_hexagon_angle_l117_11723


namespace stamp_exhibition_l117_11796

theorem stamp_exhibition (people : ℕ) (total_stamps : ℕ) : 
  (3 * people + 24 = total_stamps) →
  (4 * people = total_stamps + 26) →
  total_stamps = 174 := by
sorry

end stamp_exhibition_l117_11796


namespace danny_bottle_caps_l117_11725

theorem danny_bottle_caps (thrown_away old_caps new_caps_initial new_caps_additional : ℕ) 
  (h1 : thrown_away = 6)
  (h2 : new_caps_initial = 50)
  (h3 : new_caps_additional = thrown_away + 44) :
  new_caps_initial + new_caps_additional - thrown_away = 94 := by
  sorry

end danny_bottle_caps_l117_11725


namespace horner_method_operations_l117_11724

def horner_polynomial (x : ℝ) : ℝ := ((((((9 * x + 12) * x + 7) * x + 54) * x + 34) * x + 9) * x + 1)

theorem horner_method_operations :
  let f := λ (x : ℝ) => 9 * x^6 + 12 * x^5 + 7 * x^4 + 54 * x^3 + 34 * x^2 + 9 * x + 1
  ∃ (mult_ops add_ops : ℕ), 
    (∀ x : ℝ, f x = horner_polynomial x) ∧
    mult_ops = 6 ∧
    add_ops = 6 :=
sorry

end horner_method_operations_l117_11724


namespace tristan_study_schedule_l117_11718

/-- Tristan's study schedule problem -/
theorem tristan_study_schedule (monday tuesday wednesday thursday friday goal saturday sunday : ℝ) 
  (h1 : monday = 4)
  (h2 : tuesday = 5)
  (h3 : wednesday = 6)
  (h4 : thursday = tuesday / 2)
  (h5 : friday = 2 * monday)
  (h6 : goal = 41.5)
  (h7 : saturday = sunday)
  (h8 : monday + tuesday + wednesday + thursday + friday + saturday + sunday = goal) :
  saturday = 8 := by
sorry


end tristan_study_schedule_l117_11718


namespace grade_assignments_12_students_4_grades_l117_11710

/-- The number of possible grade assignments for a class -/
def gradeAssignments (numStudents : ℕ) (numGrades : ℕ) : ℕ :=
  numGrades ^ numStudents

/-- Theorem stating the number of ways to assign 4 grades to 12 students -/
theorem grade_assignments_12_students_4_grades :
  gradeAssignments 12 4 = 16777216 := by
  sorry

end grade_assignments_12_students_4_grades_l117_11710


namespace gcd_98_63_l117_11700

theorem gcd_98_63 : Int.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l117_11700


namespace total_age_problem_l117_11766

/-- Given three people a, b, and c, where a is two years older than b, 
    b is twice as old as c, and b is 10 years old, 
    prove that the total of their ages is 27 years. -/
theorem total_age_problem (a b c : ℕ) : 
  b = 10 → a = b + 2 → b = 2 * c → a + b + c = 27 := by
sorry

end total_age_problem_l117_11766


namespace slope_intercept_product_l117_11719

/-- Given points A, B, C in a plane, and D as the midpoint of AB,
    prove that the product of the slope and y-intercept of line CD is -5/2 -/
theorem slope_intercept_product (A B C D : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 0) →
  C = (10, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let m := (C.2 - D.2) / (C.1 - D.1)
  let b := D.2
  m * b = -5/2 := by sorry

end slope_intercept_product_l117_11719


namespace hexagonal_grid_consecutive_circles_l117_11756

/-- Represents a hexagonal grid of circles -/
structure HexagonalGrid :=
  (num_circles : ℕ)

/-- Counts the number of ways to choose 3 consecutive circles in a row -/
def count_horizontal_ways (grid : HexagonalGrid) : ℕ :=
  (1 + 2 + 3 + 4 + 5 + 6)

/-- Counts the number of ways to choose 3 consecutive circles in one diagonal direction -/
def count_diagonal_ways (grid : HexagonalGrid) : ℕ :=
  (4 + 4 + 4 + 3 + 2 + 1)

/-- Counts the total number of ways to choose 3 consecutive circles in all directions -/
def count_total_ways (grid : HexagonalGrid) : ℕ :=
  count_horizontal_ways grid + 2 * count_diagonal_ways grid

/-- Theorem: The total number of ways to choose 3 consecutive circles in a hexagonal grid of 33 circles is 57 -/
theorem hexagonal_grid_consecutive_circles (grid : HexagonalGrid) 
  (h : grid.num_circles = 33) : count_total_ways grid = 57 := by
  sorry

end hexagonal_grid_consecutive_circles_l117_11756


namespace similar_triangle_shorter_sides_sum_l117_11714

theorem similar_triangle_shorter_sides_sum (a b c : ℝ) (k : ℝ) :
  a = 8 ∧ b = 10 ∧ c = 12 →
  k * (a + b + c) = 180 →
  k * a + k * b = 108 := by
  sorry

end similar_triangle_shorter_sides_sum_l117_11714


namespace train_crossing_platforms_l117_11786

/-- A train crosses two platforms of different lengths. -/
theorem train_crossing_platforms
  (train_length : ℝ)
  (platform1_length : ℝ)
  (platform2_length : ℝ)
  (time1 : ℝ)
  (h1 : train_length = 350)
  (h2 : platform1_length = 100)
  (h3 : platform2_length = 250)
  (h4 : time1 = 15)
  : (train_length + platform2_length) / ((train_length + platform1_length) / time1) = 20 := by
  sorry

#check train_crossing_platforms

end train_crossing_platforms_l117_11786


namespace sector_properties_l117_11793

/-- Represents a circular sector --/
structure Sector where
  radius : ℝ
  centralAngle : ℝ

/-- Calculates the perimeter of a sector --/
def sectorPerimeter (s : Sector) : ℝ :=
  2 * s.radius + s.radius * s.centralAngle

/-- Calculates the area of a sector --/
def sectorArea (s : Sector) : ℝ :=
  0.5 * s.radius * s.radius * s.centralAngle

theorem sector_properties :
  ∃ (s : Sector),
    sectorPerimeter s = 8 ∧
    (s.centralAngle = 2 → sectorArea s = 4) ∧
    (∀ (t : Sector), sectorPerimeter t = 8 → sectorArea t ≤ 4) ∧
    (sectorArea s = 4 ∧ s.centralAngle = 2) := by
  sorry

end sector_properties_l117_11793


namespace birch_not_adjacent_probability_l117_11787

def maple_count : ℕ := 3
def oak_count : ℕ := 4
def birch_count : ℕ := 5

def total_trees : ℕ := maple_count + oak_count + birch_count

def total_arrangements : ℕ := Nat.factorial total_trees / (Nat.factorial maple_count * Nat.factorial oak_count * Nat.factorial birch_count)

def favorable_arrangements : ℕ := (Nat.choose (maple_count + oak_count + 1) birch_count) * (Nat.factorial (maple_count + oak_count))

theorem birch_not_adjacent_probability : 
  (favorable_arrangements : ℚ) / total_arrangements = 7 / 99 := by sorry

end birch_not_adjacent_probability_l117_11787


namespace vector_square_difference_l117_11776

theorem vector_square_difference (a b : ℝ × ℝ) (h1 : a + b = (-3, 6)) (h2 : a - b = (-3, 2)) :
  (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 := by sorry

end vector_square_difference_l117_11776


namespace man_money_calculation_l117_11705

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def total_amount (n_50 n_500 : ℕ) : ℕ :=
  50 * n_50 + 500 * n_500

/-- Proves that a man with 36 notes, 17 of which are 50 rupee notes and the rest are 500 rupee notes, has 10350 rupees in total -/
theorem man_money_calculation :
  let total_notes : ℕ := 36
  let n_50 : ℕ := 17
  let n_500 : ℕ := total_notes - n_50
  total_amount n_50 n_500 = 10350 := by
  sorry

#eval total_amount 17 19

end man_money_calculation_l117_11705


namespace circle_center_l117_11742

/-- Given a circle with equation (x-2)^2 + (y-3)^2 = 1, its center is at (2, 3) -/
theorem circle_center (x y : ℝ) : 
  ((x - 2)^2 + (y - 3)^2 = 1) → (2, 3) = (x, y) := by sorry

end circle_center_l117_11742


namespace jerrys_shelf_books_l117_11781

/-- The number of books on Jerry's shelf -/
def books : ℕ := 9

/-- The initial number of action figures -/
def initial_figures : ℕ := 5

/-- The number of action figures added -/
def added_figures : ℕ := 7

/-- The difference between action figures and books -/
def figure_book_difference : ℕ := 3

theorem jerrys_shelf_books :
  books = initial_figures + added_figures - figure_book_difference := by
  sorry

end jerrys_shelf_books_l117_11781


namespace tire_price_calculation_l117_11777

theorem tire_price_calculation (num_tires sale_price total_savings : ℕ) 
  (h1 : num_tires = 4)
  (h2 : sale_price = 75)
  (h3 : total_savings = 36)
  : sale_price + total_savings / num_tires = 84 := by
  sorry

end tire_price_calculation_l117_11777


namespace shipping_cost_for_five_pounds_l117_11788

/-- Calculates the shipping cost based on weight and rates -/
def shipping_cost (flat_fee : ℝ) (per_pound_rate : ℝ) (weight : ℝ) : ℝ :=
  flat_fee + per_pound_rate * weight

/-- Proves that the shipping cost for a 5-pound package is $9.00 -/
theorem shipping_cost_for_five_pounds :
  shipping_cost 5 0.8 5 = 9 := by
  sorry

end shipping_cost_for_five_pounds_l117_11788


namespace tournament_max_matches_l117_11779

/-- Represents a round-robin tennis tournament -/
structure TennisTournament where
  players : ℕ
  original_days : ℕ
  rest_days : ℕ

/-- Calculates the maximum number of matches that can be completed in a tournament -/
def max_matches (t : TennisTournament) : ℕ :=
  min
    ((t.players * (t.players - 1)) / 2)
    ((t.players / 2) * (t.original_days - t.rest_days))

/-- Theorem: In a tournament with 10 players, 9 original days, and 1 rest day, 
    the maximum number of matches is 40 -/
theorem tournament_max_matches :
  let t : TennisTournament := ⟨10, 9, 1⟩
  max_matches t = 40 := by
  sorry


end tournament_max_matches_l117_11779


namespace systematic_sampling_probabilities_l117_11711

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  removed : ℕ
  (population_positive : population > 0)
  (sample_size_le_population : sample_size ≤ population)
  (removed_le_population : removed ≤ population)

/-- The probability of an individual being removed in a systematic sampling scenario -/
def prob_removed (s : SystematicSampling) : ℚ :=
  s.removed / s.population

/-- The probability of an individual being sampled in a systematic sampling scenario -/
def prob_sampled (s : SystematicSampling) : ℚ :=
  s.sample_size / s.population

/-- Theorem stating the probabilities for the given systematic sampling scenario -/
theorem systematic_sampling_probabilities :
  let s : SystematicSampling :=
    { population := 1003
    , sample_size := 50
    , removed := 3
    , population_positive := by norm_num
    , sample_size_le_population := by norm_num
    , removed_le_population := by norm_num }
  prob_removed s = 3 / 1003 ∧ prob_sampled s = 50 / 1003 := by
  sorry

end systematic_sampling_probabilities_l117_11711


namespace triangle_side_length_l117_11730

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given conditions
  a * (1 - Real.cos B) = b * Real.cos A →
  c = 3 →
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 2 →
  -- Conclusion
  b = 4 * Real.sqrt 2 ∨ b = 2 := by
sorry

end triangle_side_length_l117_11730


namespace connie_blue_markers_l117_11755

/-- Given that Connie has 41 red markers and a total of 105 markers,
    prove that she has 64 blue markers. -/
theorem connie_blue_markers :
  let red_markers : ℕ := 41
  let total_markers : ℕ := 105
  let blue_markers := total_markers - red_markers
  blue_markers = 64 := by
  sorry

end connie_blue_markers_l117_11755


namespace special_isosceles_inscribed_circle_radius_l117_11745

/-- An isosceles triangle with a specific inscribed circle property -/
structure SpecialIsoscelesTriangle where
  -- Base of the triangle
  base : ℝ
  -- Ratio of the parts of the altitude divided by the center of the inscribed circle
  altitude_ratio : ℝ × ℝ
  -- The triangle is isosceles
  isIsosceles : True
  -- The base is 60
  base_is_60 : base = 60
  -- The ratio is 17:15
  ratio_is_17_15 : altitude_ratio = (17, 15)

/-- The radius of the inscribed circle in the special isosceles triangle -/
def inscribed_circle_radius (t : SpecialIsoscelesTriangle) : ℝ := 7.5

/-- Theorem: The radius of the inscribed circle in the special isosceles triangle is 7.5 -/
theorem special_isosceles_inscribed_circle_radius (t : SpecialIsoscelesTriangle) :
  inscribed_circle_radius t = 7.5 := by
  sorry

end special_isosceles_inscribed_circle_radius_l117_11745


namespace sum_p_q_form_l117_11732

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  h1 : ∀ x, ∃ a b c, q x = a * x^2 + b * x + c  -- q(x) is quadratic
  h2 : p 1 = 4  -- p(1) = 4
  h3 : q 3 = 0  -- q(3) = 0
  h4 : ∃ k, ∀ x, q x = k * (x - 3)^2  -- q(x) has a double root at x = 3

/-- The main theorem about the sum of p(x) and q(x) -/
theorem sum_p_q_form (f : RationalFunction) :
  ∃ a c : ℝ, (∀ x, f.p x + f.q x = x^2 + (a - 6) * x + 13) ∧ a + c = 4 := by
  sorry


end sum_p_q_form_l117_11732


namespace solution_set_f_range_of_m_l117_11739

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

-- Theorem for part (1)
theorem solution_set_f (x : ℝ) :
  f x ≤ 2 ↔ x ∈ Set.Icc (-4) 10 :=
sorry

-- Theorem for part (2)
theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, f x - g x ≥ m - 3) ↔ m ≤ 5 :=
sorry

end solution_set_f_range_of_m_l117_11739


namespace sum_coordinates_of_D_l117_11798

/-- Given that N(5,9) is the midpoint of line segment CD and C has coordinates (11,5),
    prove that the sum of the coordinates of point D is 12. -/
theorem sum_coordinates_of_D (C D N : ℝ × ℝ) : 
  C = (11, 5) → 
  N = (5, 9) → 
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 12 := by
sorry

end sum_coordinates_of_D_l117_11798


namespace cleaning_time_proof_l117_11765

def grove_width : ℕ := 4
def grove_length : ℕ := 5
def cleaning_time_per_tree : ℕ := 6
def minutes_per_hour : ℕ := 60

theorem cleaning_time_proof :
  let total_trees := grove_width * grove_length
  let total_cleaning_time := total_trees * cleaning_time_per_tree
  let cleaning_time_hours := total_cleaning_time / minutes_per_hour
  let actual_cleaning_time := cleaning_time_hours / 2
  actual_cleaning_time = 1 := by sorry

end cleaning_time_proof_l117_11765


namespace tina_postcard_earnings_l117_11758

/-- Tina's postcard business earnings calculation --/
theorem tina_postcard_earnings :
  let postcards_per_day : ℕ := 30
  let price_per_postcard : ℕ := 5
  let days_worked : ℕ := 6
  let total_postcards : ℕ := postcards_per_day * days_worked
  let total_earnings : ℕ := total_postcards * price_per_postcard
  total_earnings = 900 :=
by
  sorry

#check tina_postcard_earnings

end tina_postcard_earnings_l117_11758


namespace initial_geese_count_l117_11792

theorem initial_geese_count (initial_count : ℕ) : 
  (initial_count / 2 + 4 = 12) → initial_count = 16 := by
  sorry

end initial_geese_count_l117_11792


namespace problem_statement_l117_11764

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = x + y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a + b → 1/a + 1/b ≥ 2) ∧
  (1/x + 1/y = 2 ↔ x = 1 ∧ y = 1) ∧
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = a + b ∧ (a + 1) * (b + 1) = 5 :=
by sorry

end problem_statement_l117_11764


namespace marshmallow_challenge_l117_11778

/-- The Marshmallow Challenge Theorem -/
theorem marshmallow_challenge (haley michael brandon : ℕ) 
  (haley_holds : haley = 8)
  (michael_holds : michael = 3 * haley)
  (brandon_holds : brandon = michael / 2) :
  haley + michael + brandon = 44 := by
  sorry

end marshmallow_challenge_l117_11778


namespace floor_ceil_sum_l117_11784

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.3 : ℝ)⌉ = 27 := by sorry

end floor_ceil_sum_l117_11784


namespace watch_cost_price_proof_l117_11795

/-- The cost price of a watch satisfying certain selling conditions -/
def watch_cost_price : ℝ := 875

/-- The selling price of the watch at a loss -/
def selling_price_loss : ℝ := watch_cost_price * (1 - 0.12)

/-- The selling price of the watch at a gain -/
def selling_price_gain : ℝ := watch_cost_price * (1 + 0.04)

/-- Theorem stating the cost price of the watch given the selling conditions -/
theorem watch_cost_price_proof :
  (selling_price_loss = watch_cost_price * (1 - 0.12)) ∧
  (selling_price_gain = watch_cost_price * (1 + 0.04)) ∧
  (selling_price_gain - selling_price_loss = 140) →
  watch_cost_price = 875 := by
  sorry

end watch_cost_price_proof_l117_11795


namespace equation_solutions_l117_11762

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -Real.sqrt 2 ∧ x₁^2 + Real.sqrt 2 * x₁ - Real.sqrt 6 = Real.sqrt 3 * x₁) ∧
  (x₂ = Real.sqrt 3 ∧ x₂^2 + Real.sqrt 2 * x₂ - Real.sqrt 6 = Real.sqrt 3 * x₂) :=
by sorry

end equation_solutions_l117_11762


namespace gift_wrapping_l117_11790

theorem gift_wrapping (total_rolls total_gifts first_roll_gifts second_roll_gifts : ℕ) :
  total_rolls = 3 →
  total_gifts = 12 →
  first_roll_gifts = 3 →
  second_roll_gifts = 5 →
  total_gifts = first_roll_gifts + second_roll_gifts + (total_gifts - (first_roll_gifts + second_roll_gifts)) →
  (total_gifts - (first_roll_gifts + second_roll_gifts)) = 4 :=
by
  sorry

end gift_wrapping_l117_11790


namespace seats_taken_l117_11771

theorem seats_taken (rows : ℕ) (chairs_per_row : ℕ) (unoccupied : ℕ) : 
  rows = 40 → chairs_per_row = 20 → unoccupied = 10 → 
  rows * chairs_per_row - unoccupied = 790 := by
  sorry

end seats_taken_l117_11771


namespace total_crayons_lost_l117_11736

/-- Represents a box of crayons with initial and final counts -/
structure CrayonBox where
  initial : Nat
  final : Nat

/-- Calculates the number of crayons lost or given away from a box -/
def crayonsLost (box : CrayonBox) : Nat :=
  box.initial - box.final

theorem total_crayons_lost (box1 box2 box3 : CrayonBox)
  (h1 : box1.initial = 479 ∧ box1.final = 134)
  (h2 : box2.initial = 352 ∧ box2.final = 221)
  (h3 : box3.initial = 621 ∧ box3.final = 487) :
  crayonsLost box1 + crayonsLost box2 + crayonsLost box3 = 610 := by
  sorry

#eval crayonsLost ⟨479, 134⟩ + crayonsLost ⟨352, 221⟩ + crayonsLost ⟨621, 487⟩

end total_crayons_lost_l117_11736


namespace sallys_initial_cards_l117_11770

/-- Proves that Sally initially had 27 Pokemon cards given the problem conditions --/
theorem sallys_initial_cards (x : ℕ) : x + 20 = 41 + 6 → x = 27 := by
  sorry

end sallys_initial_cards_l117_11770


namespace geometric_progression_solutions_l117_11748

theorem geometric_progression_solutions : 
  ∃ (x₁ x₂ a₁ a₂ : ℝ), 
    (x₁ = 2 ∧ a₁ = 3 ∧ 3 * |x₁| * Real.sqrt (x₁ + 2) = 5 * x₁ + 2) ∧
    (x₂ = -2/9 ∧ a₂ = 1/2 ∧ 3 * |x₂| * Real.sqrt (x₂ + 2) = 5 * x₂ + 2) := by
  sorry

end geometric_progression_solutions_l117_11748


namespace sqrt5_irrational_and_greater_than_sqrt3_l117_11713

theorem sqrt5_irrational_and_greater_than_sqrt3 : 
  Irrational (Real.sqrt 5) ∧ Real.sqrt 5 > Real.sqrt 3 := by
  sorry

end sqrt5_irrational_and_greater_than_sqrt3_l117_11713


namespace man_speed_in_still_water_l117_11754

/-- Represents the speed of the man in still water -/
def man_speed : ℝ := 9

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 3

/-- The distance traveled downstream -/
def downstream_distance : ℝ := 36

/-- The distance traveled upstream -/
def upstream_distance : ℝ := 18

/-- The time taken for both downstream and upstream journeys -/
def journey_time : ℝ := 3

theorem man_speed_in_still_water :
  (man_speed + stream_speed) * journey_time = downstream_distance ∧
  (man_speed - stream_speed) * journey_time = upstream_distance →
  man_speed = 9 := by
sorry

end man_speed_in_still_water_l117_11754


namespace school_year_weekly_hours_l117_11737

def summer_weekly_hours : ℕ := 60
def summer_weeks : ℕ := 8
def summer_earnings : ℕ := 6000
def school_year_weeks : ℕ := 40
def school_year_earnings : ℕ := 7500

theorem school_year_weekly_hours : ℕ := by
  sorry

end school_year_weekly_hours_l117_11737


namespace inequality_theorem_l117_11757

theorem inequality_theorem (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + (n^n : ℝ)/(x^n) ≥ n + 1 := by
  sorry

end inequality_theorem_l117_11757


namespace circle_equation_a_range_l117_11772

/-- A circle in the xy-plane can be represented by the equation x^2 + y^2 - 2x + 2y + a = 0,
    where a is a real number. This theorem states that the range of a for which this equation
    represents a circle is (-∞, 2). -/
theorem circle_equation_a_range :
  ∀ a : ℝ, (∃ x y : ℝ, x^2 + y^2 - 2*x + 2*y + a = 0 ∧ 
    ∀ x' y' : ℝ, x'^2 + y'^2 - 2*x' + 2*y' + a = 0 → (x' - x)^2 + (y' - y)^2 = Constant)
  ↔ a < 2 :=
sorry

end circle_equation_a_range_l117_11772


namespace trig_identity_l117_11791

theorem trig_identity (α : Real) 
  (h : Real.sin α - Real.cos α = -7/5) : 
  (Real.sin α * Real.cos α = -12/25) ∧ 
  ((Real.tan α = -3/4) ∨ (Real.tan α = -4/3)) := by
  sorry

end trig_identity_l117_11791


namespace rhombus_side_length_l117_11722

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3). -/
theorem rhombus_side_length (K : ℝ) (d₁ d₂ s : ℝ) (h₁ : K > 0) (h₂ : d₁ > 0) (h₃ : d₂ > 0) (h₄ : s > 0) :
  d₂ = 3 * d₁ →
  K = (1/2) * d₁ * d₂ →
  s^2 = (d₁/2)^2 + (d₂/2)^2 →
  s = Real.sqrt ((5 * K) / 3) :=
by sorry

end rhombus_side_length_l117_11722


namespace square_sum_geq_product_sum_l117_11768

theorem square_sum_geq_product_sum (x y z : ℝ) : x^2 + y^2 + z^2 ≥ x*y + y*z + z*x := by
  sorry

end square_sum_geq_product_sum_l117_11768


namespace average_b_c_is_70_l117_11780

/-- Given two numbers a and b with an average of 50, and a third number c such that c - a = 40,
    prove that the average of b and c is 70. -/
theorem average_b_c_is_70 (a b c : ℝ) 
    (h1 : (a + b) / 2 = 50)
    (h2 : c - a = 40) : 
  (b + c) / 2 = 70 := by
  sorry

end average_b_c_is_70_l117_11780


namespace candidate_vote_percentage_l117_11727

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15/100)
  (h3 : candidate_valid_votes = 333200) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) * 100 = 70 := by
sorry

end candidate_vote_percentage_l117_11727


namespace random_walk_properties_l117_11747

/-- Represents a random walk on a line. -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of the random walk. -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk. -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences that achieve the maximum range. -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Theorem stating the properties of the random walk. -/
theorem random_walk_properties (w : RandomWalk) :
  (max_range w = w.a) ∧
  (min_range w = w.a - w.b) ∧
  (max_range_sequences w = w.b + 1) := by sorry

end random_walk_properties_l117_11747


namespace new_alcohol_concentration_l117_11746

/-- Represents a vessel containing an alcohol mixture -/
structure Vessel where
  capacity : ℝ
  alcohol_concentration : ℝ

/-- Calculates the amount of alcohol in a vessel -/
def alcohol_amount (v : Vessel) : ℝ := v.capacity * v.alcohol_concentration

theorem new_alcohol_concentration
  (vessel1 : Vessel)
  (vessel2 : Vessel)
  (final_capacity : ℝ)
  (h1 : vessel1.capacity = 2)
  (h2 : vessel1.alcohol_concentration = 0.4)
  (h3 : vessel2.capacity = 6)
  (h4 : vessel2.alcohol_concentration = 0.6)
  (h5 : final_capacity = 10)
  (h6 : vessel1.capacity + vessel2.capacity = 8) :
  let total_alcohol := alcohol_amount vessel1 + alcohol_amount vessel2
  let new_concentration := total_alcohol / final_capacity
  new_concentration = 0.44 := by
  sorry

#check new_alcohol_concentration

end new_alcohol_concentration_l117_11746


namespace stating_first_player_strategy_l117_11712

/-- 
Represents a game where two players fill coefficients of quadratic equations.
n is the number of equations.
-/
def QuadraticGame (n : ℕ) :=
  { rootless : ℕ // rootless ≤ n }

/-- 
The maximum number of rootless equations the first player can guarantee.
-/
def maxRootlessEquations (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- 
Theorem stating that the first player can always ensure at least (n+1)/2 
equations have no roots, regardless of the second player's actions.
-/
theorem first_player_strategy (n : ℕ) :
  ∃ (strategy : QuadraticGame n), 
    (strategy.val ≥ maxRootlessEquations n) :=
sorry

end stating_first_player_strategy_l117_11712


namespace right_triangle_shorter_leg_l117_11775

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a ^ 2 + b ^ 2 = c ^ 2 →  -- Pythagorean theorem
  c = 65 →                 -- Hypotenuse length
  a < b →                  -- a is the shorter leg
  a = 39 :=                -- Shorter leg length
by sorry

end right_triangle_shorter_leg_l117_11775


namespace tangent_slope_acute_implies_a_equals_one_l117_11773

/-- Given a curve C: y = x^3 - 2ax^2 + 2ax, if the slope of the tangent line
    at any point on the curve is acute, then a = 1, where a is an integer. -/
theorem tangent_slope_acute_implies_a_equals_one (a : ℤ) : 
  (∀ x : ℝ, 0 < 3*x^2 - 4*a*x + 2*a) → a = 1 := by
  sorry

end tangent_slope_acute_implies_a_equals_one_l117_11773


namespace largest_product_digit_sum_l117_11750

def is_single_digit_prime (n : ℕ) : Prop :=
  n < 10 ∧ Nat.Prime n

def largest_product (a b : ℕ) : ℕ :=
  a * b * (a * b + 3)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_digit_sum :
  ∃ (a b : ℕ),
    is_single_digit_prime a ∧
    is_single_digit_prime b ∧
    a ≠ b ∧
    Nat.Prime (a * b + 3) ∧
    (∀ (x y : ℕ),
      is_single_digit_prime x ∧
      is_single_digit_prime y ∧
      x ≠ y ∧
      Nat.Prime (x * y + 3) →
      largest_product x y ≤ largest_product a b) ∧
    sum_of_digits (largest_product a b) = 13 :=
  sorry

end largest_product_digit_sum_l117_11750


namespace alloy_mixture_problem_l117_11761

/-- Represents the composition of an alloy -/
structure Alloy where
  lead : ℝ
  tin : ℝ
  copper : ℝ

/-- The total weight of an alloy -/
def Alloy.weight (a : Alloy) : ℝ := a.lead + a.tin + a.copper

/-- The problem statement -/
theorem alloy_mixture_problem (alloyA alloyB : Alloy) 
  (h1 : alloyA.weight = 170)
  (h2 : alloyB.weight = 250)
  (h3 : alloyB.tin / alloyB.copper = 3 / 5)
  (h4 : alloyA.tin + alloyB.tin = 221.25)
  (h5 : alloyA.copper = 0)
  (h6 : alloyB.lead = 0) :
  alloyA.lead / alloyA.tin = 1 / 3 := by
  sorry

end alloy_mixture_problem_l117_11761


namespace cuboid_first_edge_length_l117_11760

/-- The length of the first edge of a cuboid with volume 30 cm³ and other edges 5 cm and 3 cm -/
def first_edge_length : ℝ := 2

/-- The volume of the cuboid -/
def cuboid_volume : ℝ := 30

/-- The width of the cuboid -/
def cuboid_width : ℝ := 5

/-- The height of the cuboid -/
def cuboid_height : ℝ := 3

theorem cuboid_first_edge_length :
  first_edge_length * cuboid_width * cuboid_height = cuboid_volume :=
by sorry

end cuboid_first_edge_length_l117_11760


namespace infinite_prime_factors_and_non_factors_l117_11782

def sequence_a : ℕ → ℕ
  | 0 => 4
  | n + 1 => sequence_a n * (sequence_a n - 1)

def prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ p ∣ n}

def prime_factors_of_sequence : Set ℕ :=
  ⋃ n, prime_factors (sequence_a n)

def primes_not_dividing_sequence : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∀ n, ¬(p ∣ sequence_a n)}

theorem infinite_prime_factors_and_non_factors :
  (Set.Infinite prime_factors_of_sequence) ∧
  (Set.Infinite primes_not_dividing_sequence) :=
sorry

end infinite_prime_factors_and_non_factors_l117_11782


namespace trig_identity_l117_11726

theorem trig_identity (α : Real) 
  (h : (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 2) : 
  1 + 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 1/10 := by
  sorry

end trig_identity_l117_11726


namespace fairy_tale_book_weighs_1_1_kg_l117_11704

/-- The weight of the fairy tale book in kilograms -/
def fairy_tale_book_weight : ℝ := sorry

/-- The total weight on the other side of the scale in kilograms -/
def other_side_weight : ℝ := 0.5 + 0.3 + 0.3

/-- The scale is level, so the weights on both sides are equal -/
axiom scale_balance : fairy_tale_book_weight = other_side_weight

/-- Theorem: The fairy tale book weighs 1.1 kg -/
theorem fairy_tale_book_weighs_1_1_kg : fairy_tale_book_weight = 1.1 := by sorry

end fairy_tale_book_weighs_1_1_kg_l117_11704


namespace n_times_n_plus_one_divisible_by_two_l117_11728

theorem n_times_n_plus_one_divisible_by_two (n : ℤ) (h : 1 ≤ n ∧ n ≤ 99) : 
  2 ∣ (n * (n + 1)) := by
  sorry

end n_times_n_plus_one_divisible_by_two_l117_11728


namespace sinusoidal_function_properties_l117_11799

/-- Given a sinusoidal function with specific properties, prove its exact form and the set of x-values where it equals 1. -/
theorem sinusoidal_function_properties (A ω φ : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h2 : A > 0)
  (h3 : ω > 0)
  (h4 : |φ| < π)
  (h5 : f (π/8) = 2)
  (h6 : f (5*π/8) = -2) :
  (∀ x, f x = 2 * Real.sin (2*x + π/4)) ∧ 
  (∀ x, f x = 1 ↔ ∃ k : ℤ, x = -π/24 + k*π ∨ x = 7*π/24 + k*π) := by
  sorry

end sinusoidal_function_properties_l117_11799


namespace negative_root_range_l117_11708

theorem negative_root_range (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ (3/2)^x = (2+3*a)/(5-a)) → 
  a ∈ Set.Ioo (-2/3 : ℝ) (3/4 : ℝ) :=
by sorry

end negative_root_range_l117_11708


namespace convex_pentagon_probability_l117_11759

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 5

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def ways_to_select_chords : ℕ := total_chords.choose k

/-- The number of convex pentagons that can be formed -/
def convex_pentagons : ℕ := n.choose k

/-- The probability of forming a convex pentagon -/
def probability : ℚ := convex_pentagons / ways_to_select_chords

theorem convex_pentagon_probability : probability = 1 / 1755 := by
  sorry

end convex_pentagon_probability_l117_11759


namespace train_speed_increase_l117_11797

theorem train_speed_increase (distance : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) (speed_limit : ℝ)
  (h1 : distance = 1600)
  (h2 : speed_increase = 20)
  (h3 : time_reduction = 4)
  (h4 : speed_limit = 140) :
  ∃ (original_speed : ℝ),
    original_speed > 0 ∧
    distance / original_speed = distance / (original_speed + speed_increase) + time_reduction ∧
    original_speed + speed_increase < speed_limit :=
by sorry

#check train_speed_increase

end train_speed_increase_l117_11797
