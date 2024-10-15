import Mathlib

namespace NUMINAMATH_CALUDE_fourth_power_difference_l336_33661

theorem fourth_power_difference (a b : ℝ) : 
  (a - b)^4 = a^4 - 4*a^3*b + 6*a^2*b^2 - 4*a*b^3 + b^4 :=
by
  sorry

-- Given condition
axiom fourth_power_sum (a b : ℝ) : 
  (a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4

end NUMINAMATH_CALUDE_fourth_power_difference_l336_33661


namespace NUMINAMATH_CALUDE_x_plus_z_equals_15_l336_33637

theorem x_plus_z_equals_15 (x y z : ℝ) 
  (h1 : |x| + x + z = 15) 
  (h2 : x + |y| - y = 8) : 
  x + z = 15 := by
sorry

end NUMINAMATH_CALUDE_x_plus_z_equals_15_l336_33637


namespace NUMINAMATH_CALUDE_tv_horizontal_length_l336_33604

/-- Represents a rectangular TV screen -/
structure TVScreen where
  horizontal : ℝ
  vertical : ℝ
  diagonal : ℝ

/-- The TV screen satisfies the 16:9 aspect ratio and has a 36-inch diagonal -/
def is_valid_tv_screen (tv : TVScreen) : Prop :=
  tv.horizontal / tv.vertical = 16 / 9 ∧ 
  tv.diagonal = 36 ∧
  tv.diagonal^2 = tv.horizontal^2 + tv.vertical^2

/-- The theorem stating the horizontal length of the TV screen -/
theorem tv_horizontal_length (tv : TVScreen) 
  (h : is_valid_tv_screen tv) : 
  tv.horizontal = (16 * 36) / Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_tv_horizontal_length_l336_33604


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_prism_l336_33645

theorem sphere_surface_area_rectangular_prism (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * π * radius^2 = 50 * π := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_prism_l336_33645


namespace NUMINAMATH_CALUDE_inequality_solution_set_l336_33631

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_inc : ∀ x y, x < y → x ∈ [-1, 1] → y ∈ [-1, 1] → f x < f y)
  (h_dom : ∀ x, f x ≠ 0 → x ∈ [-1, 1]) :
  {x : ℝ | f (x - 1/2) + f (1/4 - x) < 0} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l336_33631


namespace NUMINAMATH_CALUDE_inequality_solution_set_l336_33657

theorem inequality_solution_set :
  let S := {x : ℝ | (x + 1/2) * (3/2 - x) ≥ 0}
  S = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l336_33657


namespace NUMINAMATH_CALUDE_clothes_transport_equals_savings_l336_33656

/-- Represents Mr. Yadav's monthly financial breakdown --/
structure MonthlyFinances where
  salary : ℝ
  consumable_rate : ℝ
  clothes_transport_rate : ℝ
  savings_rate : ℝ

/-- Calculates the yearly savings based on monthly finances --/
def yearly_savings (m : MonthlyFinances) : ℝ :=
  12 * m.savings_rate * m.salary

/-- Theorem stating that the monthly amount spent on clothes and transport
    is equal to the monthly savings --/
theorem clothes_transport_equals_savings
  (m : MonthlyFinances)
  (h1 : m.consumable_rate = 0.6)
  (h2 : m.clothes_transport_rate = 0.5 * (1 - m.consumable_rate))
  (h3 : m.savings_rate = 1 - m.consumable_rate - m.clothes_transport_rate)
  (h4 : yearly_savings m = 48456) :
  m.clothes_transport_rate * m.salary = m.savings_rate * m.salary :=
by sorry

end NUMINAMATH_CALUDE_clothes_transport_equals_savings_l336_33656


namespace NUMINAMATH_CALUDE_solve_income_problem_l336_33696

def income_problem (day2 day3 day4 day5 average : ℚ) : Prop :=
  let known_days := [day2, day3, day4, day5]
  let total := 5 * average
  let sum_known := day2 + day3 + day4 + day5
  let day1 := total - sum_known
  (day2 = 150) ∧ (day3 = 750) ∧ (day4 = 400) ∧ (day5 = 500) ∧ (average = 420) →
  day1 = 300

theorem solve_income_problem :
  ∀ day2 day3 day4 day5 average,
  income_problem day2 day3 day4 day5 average :=
by
  sorry

end NUMINAMATH_CALUDE_solve_income_problem_l336_33696


namespace NUMINAMATH_CALUDE_slope_greater_than_one_line_passes_through_point_distance_not_sqrt_two_not_four_lines_l336_33670

-- Define the line equations
def line1 (x y : ℝ) : Prop := 5 * x - 4 * y + 1 = 0
def line2 (m x y : ℝ) : Prop := (2 + m) * x + 4 * y - 2 + m = 0
def line3 (x y : ℝ) : Prop := x + y - 1 = 0
def line4 (x y : ℝ) : Prop := 2 * x + 2 * y + 1 = 0

-- Define points
def point_A : ℝ × ℝ := (-1, 2)
def point_B : ℝ × ℝ := (3, -1)

-- Statement 1
theorem slope_greater_than_one : 
  ∃ m : ℝ, (∀ x y : ℝ, line1 x y → y = m * x + (1/4)) ∧ m > 1 := by sorry

-- Statement 2
theorem line_passes_through_point :
  ∀ m : ℝ, line2 m (-1) 1 := by sorry

-- Statement 3
theorem distance_not_sqrt_two :
  ∃ d : ℝ, (d = (|1 + 2|) / Real.sqrt (2^2 + 2^2)) ∧ d ≠ Real.sqrt 2 := by sorry

-- Statement 4
theorem not_four_lines :
  ¬(∃ (lines : Finset (ℝ → ℝ → Prop)), lines.card = 4 ∧
    (∀ l ∈ lines, ∃ d1 d2 : ℝ, d1 = 1 ∧ d2 = 4 ∧
      (∀ x y : ℝ, l x y → 
        (Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2) = d1 ∧
         Real.sqrt ((x - point_B.1)^2 + (y - point_B.2)^2) = d2)))) := by sorry

end NUMINAMATH_CALUDE_slope_greater_than_one_line_passes_through_point_distance_not_sqrt_two_not_four_lines_l336_33670


namespace NUMINAMATH_CALUDE_not_general_term_formula_l336_33679

def alternating_sequence (n : ℕ) : ℤ := (-1)^(n + 1)

theorem not_general_term_formula :
  ∃ n : ℕ, ((-1 : ℤ)^n ≠ alternating_sequence n) ∧
  ((-1 : ℤ)^(n + 1) = alternating_sequence n) ∧
  ((-1 : ℤ)^(n - 1) = alternating_sequence n) ∧
  (if n % 2 = 0 then -1 else 1 : ℤ) = alternating_sequence n :=
sorry

end NUMINAMATH_CALUDE_not_general_term_formula_l336_33679


namespace NUMINAMATH_CALUDE_max_sum_of_seventh_powers_l336_33687

theorem max_sum_of_seventh_powers (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) :
  ∃ (m : ℝ), m = 128 ∧ a^7 + b^7 + c^7 + d^7 ≤ m ∧ 
  ∃ (a' b' c' d' : ℝ), a'^6 + b'^6 + c'^6 + d'^6 = 64 ∧ a'^7 + b'^7 + c'^7 + d'^7 = m :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_seventh_powers_l336_33687


namespace NUMINAMATH_CALUDE_simplify_power_of_product_l336_33664

theorem simplify_power_of_product (y : ℝ) : (3 * y^4)^2 = 9 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_of_product_l336_33664


namespace NUMINAMATH_CALUDE_toaster_msrp_l336_33665

/-- The MSRP of a toaster given specific conditions -/
theorem toaster_msrp (x : ℝ) : 
  x + 0.2 * x + 0.5 * (x + 0.2 * x) = 54 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_toaster_msrp_l336_33665


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_five_l336_33646

theorem reciprocal_of_negative_five :
  ∃ x : ℝ, x * (-5) = 1 ∧ x = -(1/5) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_five_l336_33646


namespace NUMINAMATH_CALUDE_no_finite_planes_cover_all_cubes_l336_33652

/-- Represents a plane in 3D space -/
structure Plane where
  -- We don't need to define the specifics of a plane for this statement

/-- Represents a cube in the integer grid -/
structure GridCube where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Checks if a plane intersects a grid cube -/
def plane_intersects_cube (p : Plane) (c : GridCube) : Prop :=
  sorry -- Definition not needed for the statement

/-- The main theorem stating that it's impossible to have a finite number of planes
    intersecting all cubes in the integer grid -/
theorem no_finite_planes_cover_all_cubes :
  ∀ (planes : Finset Plane), ∃ (c : GridCube),
    ∀ (p : Plane), p ∈ planes → ¬(plane_intersects_cube p c) := by
  sorry


end NUMINAMATH_CALUDE_no_finite_planes_cover_all_cubes_l336_33652


namespace NUMINAMATH_CALUDE_homework_time_ratio_l336_33621

/-- Represents the time spent on each subject in minutes -/
structure HomeworkTime where
  biology : ℕ
  history : ℕ
  geography : ℕ

/-- Represents the ratio of time spent on two subjects -/
structure TimeRatio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of time spent on geography to history -/
def calculateRatio (time : HomeworkTime) : TimeRatio :=
  { numerator := time.geography, denominator := time.history }

theorem homework_time_ratio (time : HomeworkTime) :
  time.biology = 20 →
  time.history = 2 * time.biology →
  time.geography > time.history →
  time.geography > time.biology →
  time.biology + time.history + time.geography = 180 →
  calculateRatio time = { numerator := 3, denominator := 1 } := by
  sorry

#check homework_time_ratio

end NUMINAMATH_CALUDE_homework_time_ratio_l336_33621


namespace NUMINAMATH_CALUDE_richard_david_age_diff_l336_33642

-- Define the ages of the three sons
def david_age : ℕ := 14
def scott_age : ℕ := 6
def richard_age : ℕ := 20

-- Define the conditions
axiom david_scott_age_diff : david_age = scott_age + 8
axiom david_past_age : david_age = 11 + 3
axiom richard_future_age : richard_age + 8 = 2 * (scott_age + 8)

-- Define the theorem to prove
theorem richard_david_age_diff : richard_age = david_age + 6 := by
  sorry

end NUMINAMATH_CALUDE_richard_david_age_diff_l336_33642


namespace NUMINAMATH_CALUDE_even_increasing_neg_implies_l336_33626

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- A function f: ℝ → ℝ is increasing on (-∞, 0) if
    for all x, y ∈ (-∞, 0), x < y implies f(x) < f(y) -/
def IncreasingOnNegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → x < 0 → y ≤ 0 → f x < f y

theorem even_increasing_neg_implies (f : ℝ → ℝ)
    (h_even : IsEven f) (h_inc : IncreasingOnNegatives f) :
    f (-1) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_neg_implies_l336_33626


namespace NUMINAMATH_CALUDE_race_distance_proof_l336_33695

/-- The distance of the race where B beats C by 100 m, given the conditions from the problem. -/
def race_distance : ℝ := 700

theorem race_distance_proof (Va Vb Vc : ℝ) 
  (h1 : Va / Vb = 1000 / 900)
  (h2 : Va / Vc = 600 / 472.5)
  (h3 : Vb / Vc = (race_distance - 100) / race_distance) : 
  race_distance = 700 := by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l336_33695


namespace NUMINAMATH_CALUDE_x_minus_y_equals_eight_l336_33673

theorem x_minus_y_equals_eight (x y : ℝ) 
  (hx : x + (-3) = 0) 
  (hy : |y| = 5) 
  (hxy : x * y < 0) : 
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_eight_l336_33673


namespace NUMINAMATH_CALUDE_candidate_votes_l336_33619

theorem candidate_votes (total_votes : ℕ) (invalid_percentage : ℚ) (candidate_percentage : ℚ) : 
  total_votes = 560000 →
  invalid_percentage = 15 / 100 →
  candidate_percentage = 75 / 100 →
  ⌊(total_votes : ℚ) * (1 - invalid_percentage) * candidate_percentage⌋ = 357000 := by
sorry

end NUMINAMATH_CALUDE_candidate_votes_l336_33619


namespace NUMINAMATH_CALUDE_initial_sweets_count_prove_initial_sweets_count_l336_33624

theorem initial_sweets_count : ℕ → Prop :=
  fun S => 
    (S / 2 + 4 + 7 = S) → 
    (S = 22)

-- Proof
theorem prove_initial_sweets_count : initial_sweets_count 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_sweets_count_prove_initial_sweets_count_l336_33624


namespace NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l336_33654

-- Problem 1
def problem1 (a : ℚ) : ℚ := a * (a - 4) - (a + 6) * (a - 2)

theorem problem1_solution :
  problem1 (-1/2) = 16 := by sorry

-- Problem 2
def problem2 (x y : ℤ) : ℤ := (x + 2*y) * (x - 2*y) - (2*x - y) * (-2*x - y)

theorem problem2_solution :
  problem2 8 (-8) = 0 := by sorry

end NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l336_33654


namespace NUMINAMATH_CALUDE_solid_circles_count_l336_33636

def circleSequence (n : ℕ) : ℕ := n * (n + 3) / 2 + 1

theorem solid_circles_count (total : ℕ) (h : total = 2019) :
  ∃ n : ℕ, circleSequence n ≤ total ∧ circleSequence (n + 1) > total ∧ n = 62 :=
by sorry

end NUMINAMATH_CALUDE_solid_circles_count_l336_33636


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l336_33678

/-- Given a cube with face perimeter 20 cm, its volume is 125 cubic centimeters. -/
theorem cube_volume_from_face_perimeter :
  ∀ (cube : ℝ → ℝ), 
  (∃ (side : ℝ), side > 0 ∧ 4 * side = 20) →
  cube (20 / 4) = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l336_33678


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l336_33620

theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 3 * a →
  a + b + c = 180 →
  c = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l336_33620


namespace NUMINAMATH_CALUDE_christmas_tree_decoration_l336_33602

theorem christmas_tree_decoration (b t : ℕ) : 
  (t = b + 1) →  -- Chuck's condition
  (2 * b = t - 1) →  -- Huck's condition
  (b = 3 ∧ t = 4) :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_christmas_tree_decoration_l336_33602


namespace NUMINAMATH_CALUDE_seven_people_six_seats_l336_33686

/-- The number of ways to seat 6 people from a group of 7 at a circular table -/
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  n * Nat.factorial (k - 1)

/-- Theorem stating the number of seating arrangements for 7 people at a circular table with 6 seats -/
theorem seven_people_six_seats :
  seating_arrangements 7 6 = 840 := by
  sorry

end NUMINAMATH_CALUDE_seven_people_six_seats_l336_33686


namespace NUMINAMATH_CALUDE_virgo_boat_trip_duration_l336_33610

/-- Represents the duration of a trip to Virgo island -/
structure VirgoTrip where
  boat_time : ℝ
  plane_time : ℝ
  total_time : ℝ

/-- Conditions for a valid Virgo trip -/
def is_valid_virgo_trip (trip : VirgoTrip) : Prop :=
  trip.plane_time = 4 * trip.boat_time ∧
  trip.total_time = trip.boat_time + trip.plane_time ∧
  trip.total_time = 10

theorem virgo_boat_trip_duration :
  ∀ (trip : VirgoTrip), is_valid_virgo_trip trip → trip.boat_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_virgo_boat_trip_duration_l336_33610


namespace NUMINAMATH_CALUDE_nursery_school_fraction_l336_33634

/-- Given a nursery school with the following conditions:
  1. 20 students are under 3 years old
  2. 50 students are not between 3 and 4 years old
  3. There are 300 children in total
  Prove that the fraction of students who are 4 years old or older is 1/10 -/
theorem nursery_school_fraction (under_three : ℕ) (not_between_three_and_four : ℕ) (total : ℕ)
  (h1 : under_three = 20)
  (h2 : not_between_three_and_four = 50)
  (h3 : total = 300) :
  (not_between_three_and_four - under_three) / total = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_nursery_school_fraction_l336_33634


namespace NUMINAMATH_CALUDE_ancient_chinese_pi_l336_33616

/-- Proves that for a cylinder with given dimensions, the implied value of π is 3 -/
theorem ancient_chinese_pi (c h v : ℝ) (hc : c = 48) (hh : h = 11) (hv : v = 2112) :
  let r := c / (2 * π)
  v = π * r^2 * h → π = 3 := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_pi_l336_33616


namespace NUMINAMATH_CALUDE_inverse_scalar_multiple_l336_33663

/-- Given a 2x2 matrix B and a constant l, prove that B^(-1) = l * B implies e = -3 and l = 1/19 -/
theorem inverse_scalar_multiple (B : Matrix (Fin 2) (Fin 2) ℝ) (l : ℝ) :
  B 0 0 = 3 ∧ B 0 1 = 4 ∧ B 1 0 = 7 ∧ B 1 1 = B.det / (3 * B 1 1 - 28) →
  B⁻¹ = l • B →
  B 1 1 = -3 ∧ l = 1 / 19 := by
sorry

end NUMINAMATH_CALUDE_inverse_scalar_multiple_l336_33663


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l336_33600

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b in ℝ², if a is parallel to b and a = (m, 4) and b = (3, -2), then m = -6 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  parallel a b → m = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l336_33600


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l336_33655

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l336_33655


namespace NUMINAMATH_CALUDE_a_faster_than_b_l336_33613

/-- Represents a person sawing wood --/
structure Sawyer where
  name : String
  segments_per_piece : ℕ
  total_segments : ℕ

/-- Calculates the number of pieces sawed by a sawyer --/
def pieces_sawed (s : Sawyer) : ℕ := s.total_segments / s.segments_per_piece

/-- Calculates the number of cuts made by a sawyer --/
def cuts_made (s : Sawyer) : ℕ := (s.segments_per_piece - 1) * (pieces_sawed s)

/-- Theorem stating that A takes less time to saw one piece of wood --/
theorem a_faster_than_b (a b : Sawyer)
  (ha : a.name = "A" ∧ a.segments_per_piece = 3 ∧ a.total_segments = 24)
  (hb : b.name = "B" ∧ b.segments_per_piece = 2 ∧ b.total_segments = 28) :
  cuts_made a > cuts_made b := by sorry

end NUMINAMATH_CALUDE_a_faster_than_b_l336_33613


namespace NUMINAMATH_CALUDE_extra_flowers_l336_33698

theorem extra_flowers (tulips roses used : ℕ) : 
  tulips = 36 → roses = 37 → used = 70 → tulips + roses - used = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_flowers_l336_33698


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l336_33693

theorem sum_of_four_numbers : 4321 + 3214 + 2143 + 1432 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l336_33693


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_quadratic_equation_solutions_l336_33682

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x => 2 * x^2 + 4 * x - 6
  (f 1 = 0 ∧ f (-3) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = -3) :=
sorry

theorem linear_quadratic_equation_solutions :
  let g : ℝ → ℝ := λ x => 2 * (x - 3) - 3 * x * (x - 3)
  (g 3 = 0 ∧ g (2/3) = 0) ∧
  (∀ x : ℝ, g x = 0 → x = 3 ∨ x = 2/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_quadratic_equation_solutions_l336_33682


namespace NUMINAMATH_CALUDE_trees_per_square_meter_l336_33672

/-- Given a forest and a square-shaped street, calculate the number of trees per square meter in the forest. -/
theorem trees_per_square_meter
  (street_side : ℝ)
  (forest_area_multiplier : ℝ)
  (total_trees : ℕ)
  (h1 : street_side = 100)
  (h2 : forest_area_multiplier = 3)
  (h3 : total_trees = 120000) :
  (total_trees : ℝ) / (forest_area_multiplier * street_side^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_trees_per_square_meter_l336_33672


namespace NUMINAMATH_CALUDE_maggi_ate_five_cupcakes_l336_33674

/-- Calculates the number of cupcakes Maggi ate -/
def cupcakes_eaten (initial_packages : ℕ) (cupcakes_per_package : ℕ) (cupcakes_left : ℕ) : ℕ :=
  initial_packages * cupcakes_per_package - cupcakes_left

/-- Proves that Maggi ate 5 cupcakes -/
theorem maggi_ate_five_cupcakes :
  cupcakes_eaten 3 4 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_maggi_ate_five_cupcakes_l336_33674


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l336_33641

theorem imaginary_unit_sum (i : ℂ) (hi : i * i = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l336_33641


namespace NUMINAMATH_CALUDE_initial_cartons_processed_l336_33623

/-- Proves that the initial number of cartons processed is 400 --/
theorem initial_cartons_processed (num_customers : ℕ) (returned_cartons : ℕ) (total_accepted : ℕ) :
  num_customers = 4 →
  returned_cartons = 60 →
  total_accepted = 160 →
  (num_customers * (total_accepted / num_customers + returned_cartons)) = 400 := by
sorry

end NUMINAMATH_CALUDE_initial_cartons_processed_l336_33623


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l336_33606

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions,
    prove that the 5th term equals 10. -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 8)
  (h_fourth : a 4 = 7) :
  a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l336_33606


namespace NUMINAMATH_CALUDE_square_point_selection_probability_square_point_selection_probability_is_three_fifths_l336_33689

/-- The probability of selecting two points from the vertices and center of a square,
    such that their distance is not less than the side length of the square. -/
theorem square_point_selection_probability : ℚ :=
  let total_selections := (5 : ℕ).choose 2
  let favorable_selections := (4 : ℕ).choose 2
  (favorable_selections : ℚ) / total_selections

/-- The probability of selecting two points from the vertices and center of a square,
    such that their distance is not less than the side length of the square, is 3/5. -/
theorem square_point_selection_probability_is_three_fifths :
  square_point_selection_probability = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_point_selection_probability_square_point_selection_probability_is_three_fifths_l336_33689


namespace NUMINAMATH_CALUDE_remainder_divisibility_l336_33630

theorem remainder_divisibility (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l336_33630


namespace NUMINAMATH_CALUDE_toys_sold_l336_33611

/-- Given a selling price, cost price per toy, and a gain equal to the cost of 3 toys,
    prove that the number of toys sold is 18. -/
theorem toys_sold (selling_price : ℕ) (cost_per_toy : ℕ) (h1 : selling_price = 18900) 
    (h2 : cost_per_toy = 900) : 
  (selling_price - 3 * cost_per_toy) / cost_per_toy = 18 := by
  sorry

end NUMINAMATH_CALUDE_toys_sold_l336_33611


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l336_33658

-- Define the arithmetic sequence
def a (n : ℕ) : ℚ := n + 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 1) / a n + a n / a (n + 1) - 2

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_geometric_sequence :
  (a 4 - a 2 = 2) ∧ 
  (a 3 * a 3 = a 1 * a 7) →
  (∀ n : ℕ, a n = n + 1) ∧
  (∀ n : ℕ, S n = n / (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l336_33658


namespace NUMINAMATH_CALUDE_quadrilateral_angles_l336_33660

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral (q : Quadrilateral) : Prop := sorry

def angle_ABD (q : Quadrilateral) : ℝ := sorry
def angle_CBD (q : Quadrilateral) : ℝ := sorry
def angle_ADC (q : Quadrilateral) : ℝ := sorry

def AB_equals_BC (q : Quadrilateral) : Prop := sorry

def angle_A (q : Quadrilateral) : ℝ := sorry
def angle_C (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_angles 
  (q : Quadrilateral) 
  (h_convex : is_convex_quadrilateral q)
  (h_ABD : angle_ABD q = 65)
  (h_CBD : angle_CBD q = 35)
  (h_ADC : angle_ADC q = 130)
  (h_AB_BC : AB_equals_BC q) :
  angle_A q = 57.5 ∧ angle_C q = 72.5 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_angles_l336_33660


namespace NUMINAMATH_CALUDE_symmetric_difference_A_B_l336_33633

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}
def B : Set ℝ := {x : ℝ | x < 0}

-- Define set difference
def set_difference (M N : Set ℝ) : Set ℝ := {x : ℝ | x ∈ M ∧ x ∉ N}

-- Define symmetric difference
def symmetric_difference (M N : Set ℝ) : Set ℝ := 
  (set_difference M N) ∪ (set_difference N M)

-- State the theorem
theorem symmetric_difference_A_B : 
  symmetric_difference A B = {x : ℝ | x < -1 ∨ (0 ≤ x ∧ x < 1)} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_A_B_l336_33633


namespace NUMINAMATH_CALUDE_mathematicians_set_l336_33694

-- Define the set of famous people
inductive FamousPerson
| BillGates
| Gauss
| YuanLongping
| Nobel
| ChenJingrun
| HuaLuogeng
| Gorky
| Einstein

-- Define a function to determine if a person is a mathematician
def isMathematician : FamousPerson → Prop
| FamousPerson.Gauss => True
| FamousPerson.ChenJingrun => True
| FamousPerson.HuaLuogeng => True
| _ => False

-- Define the set of all famous people
def allFamousPeople : Set FamousPerson :=
  {FamousPerson.BillGates, FamousPerson.Gauss, FamousPerson.YuanLongping,
   FamousPerson.Nobel, FamousPerson.ChenJingrun, FamousPerson.HuaLuogeng,
   FamousPerson.Gorky, FamousPerson.Einstein}

-- Theorem: The set of mathematicians is equal to {Gauss, Chen Jingrun, Hua Luogeng}
theorem mathematicians_set :
  {p ∈ allFamousPeople | isMathematician p} =
  {FamousPerson.Gauss, FamousPerson.ChenJingrun, FamousPerson.HuaLuogeng} :=
by sorry

end NUMINAMATH_CALUDE_mathematicians_set_l336_33694


namespace NUMINAMATH_CALUDE_art_book_cost_l336_33614

theorem art_book_cost (total_cost : ℕ) (num_math num_art num_science : ℕ) (cost_math cost_science : ℕ) :
  total_cost = 30 ∧
  num_math = 2 ∧
  num_art = 3 ∧
  num_science = 6 ∧
  cost_math = 3 ∧
  cost_science = 3 →
  (total_cost - num_math * cost_math - num_science * cost_science) / num_art = 2 :=
by sorry

end NUMINAMATH_CALUDE_art_book_cost_l336_33614


namespace NUMINAMATH_CALUDE_board_cut_lengths_l336_33647

/-- Given a board of 180 cm cut into three pieces, prove the lengths of the pieces. -/
theorem board_cut_lengths :
  ∀ (L M S : ℝ),
  L + M + S = 180 ∧
  L = M + S + 30 ∧
  M = L / 2 - 10 →
  L = 105 ∧ M = 42.5 ∧ S = 32.5 :=
by
  sorry

end NUMINAMATH_CALUDE_board_cut_lengths_l336_33647


namespace NUMINAMATH_CALUDE_function_properties_l336_33691

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * a * x)
def g (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem function_properties (a k b : ℝ) (h_k : k ≠ 0) :
  -- Part 1
  (f a 1 = Real.exp 1 ∧ ∀ x, g k b x = -g k b (-x)) →
  a = 1/2 ∧ b = 0 ∧
  -- Part 2
  (∀ x > 0, f (1/2) x > g k 0 x) →
  k < Real.exp 1 ∧
  -- Part 3
  (∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ f (1/2) x₁ = g k 0 x₁ ∧ f (1/2) x₂ = g k 0 x₂) →
  x₁ * x₂ < 1 :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l336_33691


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l336_33615

theorem intersection_point_of_lines (x y : ℚ) : 
  (3 * y = -2 * x + 6 ∧ -2 * y = 7 * x + 4) ↔ (x = -24/17 ∧ y = 50/17) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l336_33615


namespace NUMINAMATH_CALUDE_crushing_load_square_pillars_l336_33607

theorem crushing_load_square_pillars (T H : ℝ) (hT : T = 5) (hH : H = 10) :
  (30 * T^5) / H^3 = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_crushing_load_square_pillars_l336_33607


namespace NUMINAMATH_CALUDE_train_crossing_time_l336_33667

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 280 →
  train_speed_kmh = 50.4 →
  crossing_time = train_length / (train_speed_kmh * (1000 / 3600)) →
  crossing_time = 20 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l336_33667


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l336_33635

theorem quadratic_solution_property (p q : ℝ) : 
  (3 * p^2 + 4 * p - 8 = 0) → 
  (3 * q^2 + 4 * q - 8 = 0) → 
  (p - 2) * (q - 2) = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l336_33635


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l336_33669

theorem largest_stamps_per_page (book1_stamps book2_stamps : ℕ) 
  (h1 : book1_stamps = 924) 
  (h2 : book2_stamps = 1386) : 
  Nat.gcd book1_stamps book2_stamps = 462 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l336_33669


namespace NUMINAMATH_CALUDE_tiles_needed_to_complete_pool_l336_33601

/-- Given a pool with blue and red tiles, calculate the number of additional tiles needed to complete it. -/
theorem tiles_needed_to_complete_pool 
  (blue_tiles : ℕ) 
  (red_tiles : ℕ) 
  (total_required : ℕ) 
  (h1 : blue_tiles = 48)
  (h2 : red_tiles = 32)
  (h3 : total_required = 100) :
  total_required - (blue_tiles + red_tiles) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_to_complete_pool_l336_33601


namespace NUMINAMATH_CALUDE_subtraction_result_l336_33638

theorem subtraction_result : 3.56 - 2.15 = 1.41 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l336_33638


namespace NUMINAMATH_CALUDE_no_nonzero_integer_solution_l336_33677

theorem no_nonzero_integer_solution (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_integer_solution_l336_33677


namespace NUMINAMATH_CALUDE_james_works_six_hours_l336_33618

/-- Calculates the time James spends working on chores given the following conditions:
  * There are 3 bedrooms, 1 living room, and 2 bathrooms to clean
  * Bedrooms each take 20 minutes to clean
  * Living room takes as long as the 3 bedrooms combined
  * Bathroom takes twice as long as the living room
  * Outside cleaning takes twice as long as cleaning the house
  * Chores are split with 2 siblings who are just as fast -/
def james_working_time : ℕ :=
  let num_bedrooms : ℕ := 3
  let num_livingrooms : ℕ := 1
  let num_bathrooms : ℕ := 2
  let bedroom_cleaning_time : ℕ := 20
  let livingroom_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time
  let bathroom_cleaning_time : ℕ := 2 * livingroom_cleaning_time
  let inside_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time +
                                  num_livingrooms * livingroom_cleaning_time +
                                  num_bathrooms * bathroom_cleaning_time
  let outside_cleaning_time : ℕ := 2 * inside_cleaning_time
  let total_cleaning_time : ℕ := inside_cleaning_time + outside_cleaning_time
  let num_siblings : ℕ := 2
  let james_time_minutes : ℕ := total_cleaning_time / (num_siblings + 1)
  james_time_minutes / 60

theorem james_works_six_hours : james_working_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_works_six_hours_l336_33618


namespace NUMINAMATH_CALUDE_chinese_heritage_tv_event_is_random_l336_33612

/-- Represents a TV event -/
structure TVEvent where
  program : String
  canOccur : Bool
  hasUncertainty : Bool

/-- Classifies an event as certain, impossible, or random -/
inductive EventClassification
  | Certain
  | Impossible
  | Random

/-- Determines if an event is random based on its properties -/
def isRandomEvent (e : TVEvent) : Bool :=
  e.canOccur ∧ e.hasUncertainty

/-- Classifies a TV event based on its properties -/
def classifyTVEvent (e : TVEvent) : EventClassification :=
  if isRandomEvent e then EventClassification.Random
  else if e.canOccur then EventClassification.Certain
  else EventClassification.Impossible

/-- The main theorem stating that turning on the TV and broadcasting
    "Chinese Intangible Cultural Heritage" is a random event -/
theorem chinese_heritage_tv_event_is_random :
  let e := TVEvent.mk "Chinese Intangible Cultural Heritage" true true
  classifyTVEvent e = EventClassification.Random := by
  sorry


end NUMINAMATH_CALUDE_chinese_heritage_tv_event_is_random_l336_33612


namespace NUMINAMATH_CALUDE_crude_oil_mixture_l336_33668

/-- Proves that 30 gallons from the second source is needed to obtain 50 gallons of 55% hydrocarbon crude oil -/
theorem crude_oil_mixture (x y : ℝ) : 
  x + y = 50 →                   -- Total amount is 50 gallons
  0.25 * x + 0.75 * y = 0.55 * 50 →  -- Hydrocarbon balance equation
  y = 30 := by sorry

end NUMINAMATH_CALUDE_crude_oil_mixture_l336_33668


namespace NUMINAMATH_CALUDE_minimum_framing_feet_l336_33627

def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

def enlarged_width : ℕ := original_width * enlargement_factor
def enlarged_height : ℕ := original_height * enlargement_factor

def final_width : ℕ := enlarged_width + 2 * border_width
def final_height : ℕ := enlarged_height + 2 * border_width

def perimeter_inches : ℕ := 2 * (final_width + final_height)

def inches_per_foot : ℕ := 12

theorem minimum_framing_feet :
  (perimeter_inches + inches_per_foot - 1) / inches_per_foot = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_framing_feet_l336_33627


namespace NUMINAMATH_CALUDE_only_three_divides_2002_power_l336_33608

theorem only_three_divides_2002_power : 
  ∀ p : ℕ, Prime p → p < 17 → (p ∣ 2002^2002 - 1) ↔ p = 3 := by
sorry

end NUMINAMATH_CALUDE_only_three_divides_2002_power_l336_33608


namespace NUMINAMATH_CALUDE_find_x_l336_33603

theorem find_x : ∃ x : ℕ, (2^x : ℝ) - (2^(x-2) : ℝ) = 3 * (2^12 : ℝ) ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l336_33603


namespace NUMINAMATH_CALUDE_increasing_interval_transformed_l336_33685

-- Define an even function f
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define an increasing function on an interval
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

-- Main theorem
theorem increasing_interval_transformed (f : ℝ → ℝ) :
  even_function f →
  increasing_on f 2 6 →
  increasing_on (fun x ↦ f (2 - x)) 4 8 :=
sorry

end NUMINAMATH_CALUDE_increasing_interval_transformed_l336_33685


namespace NUMINAMATH_CALUDE_range_of_z_plus_4_minus_3i_l336_33649

/-- The range of |z+4-3i| when |z| = 2 -/
theorem range_of_z_plus_4_minus_3i (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (w : ℂ), Complex.abs w = 2 ∧ Complex.abs (w + 4 - 3*Complex.I) = 3 ∧
  ∃ (v : ℂ), Complex.abs v = 2 ∧ Complex.abs (v + 4 - 3*Complex.I) = 7 ∧
  ∀ (u : ℂ), Complex.abs u = 2 → 3 ≤ Complex.abs (u + 4 - 3*Complex.I) ∧ Complex.abs (u + 4 - 3*Complex.I) ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_plus_4_minus_3i_l336_33649


namespace NUMINAMATH_CALUDE_camper_difference_l336_33625

/-- Represents the number of campers for each week -/
structure CamperCounts where
  threeWeeksAgo : ℕ
  twoWeeksAgo : ℕ
  lastWeek : ℕ

/-- The camping site scenario -/
def campingSite : CamperCounts → Prop
  | ⟨threeWeeksAgo, twoWeeksAgo, lastWeek⟩ =>
    threeWeeksAgo + twoWeeksAgo + lastWeek = 150 ∧
    twoWeeksAgo = 40 ∧
    lastWeek = 80 ∧
    threeWeeksAgo < twoWeeksAgo

theorem camper_difference (c : CamperCounts) (h : campingSite c) :
  c.twoWeeksAgo - c.threeWeeksAgo = 10 := by
  sorry

end NUMINAMATH_CALUDE_camper_difference_l336_33625


namespace NUMINAMATH_CALUDE_julia_spent_114_on_animal_food_l336_33666

/-- The total amount spent on animal food --/
def total_spent (weekly_total : ℕ) (rabbit_food_cost : ℕ) (rabbit_weeks : ℕ) (parrot_weeks : ℕ) : ℕ :=
  (weekly_total - rabbit_food_cost) * parrot_weeks + rabbit_food_cost * rabbit_weeks

/-- Proof that Julia spent $114 on animal food --/
theorem julia_spent_114_on_animal_food :
  total_spent 30 12 5 3 = 114 := by
  sorry

end NUMINAMATH_CALUDE_julia_spent_114_on_animal_food_l336_33666


namespace NUMINAMATH_CALUDE_cosh_leq_exp_squared_l336_33648

theorem cosh_leq_exp_squared (k : ℝ) :
  (∀ x : ℝ, Real.cosh x ≤ Real.exp (k * x^2)) ↔ k ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cosh_leq_exp_squared_l336_33648


namespace NUMINAMATH_CALUDE_trig_expression_equality_l336_33680

theorem trig_expression_equality : 
  (Real.sin (30 * π / 180) * Real.cos (24 * π / 180) + 
   Real.cos (150 * π / 180) * Real.cos (84 * π / 180)) / 
  (Real.sin (34 * π / 180) * Real.cos (16 * π / 180) + 
   Real.cos (146 * π / 180) * Real.cos (76 * π / 180)) = 
  Real.sin (51 * π / 180) / Real.sin (55 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l336_33680


namespace NUMINAMATH_CALUDE_expression_evaluation_l336_33675

theorem expression_evaluation (x y z : ℝ) : (x + (y + z)) - ((-x + y) + z) = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l336_33675


namespace NUMINAMATH_CALUDE_star_difference_l336_33688

def star (x y : ℝ) : ℝ := x * y - 3 * x + y

theorem star_difference : (star 6 5) - (star 5 6) = -4 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l336_33688


namespace NUMINAMATH_CALUDE_pear_juice_percentage_l336_33676

-- Define the juice production rates
def pear_juice_rate : ℚ := 10 / 4
def orange_juice_rate : ℚ := 12 / 3

-- Define the number of fruits used in the blend
def pears_in_blend : ℕ := 8
def oranges_in_blend : ℕ := 6

-- Define the total amount of juice in the blend
def total_juice : ℚ := pear_juice_rate * pears_in_blend + orange_juice_rate * oranges_in_blend

-- Define the amount of pear juice in the blend
def pear_juice_in_blend : ℚ := pear_juice_rate * pears_in_blend

-- Theorem statement
theorem pear_juice_percentage :
  (pear_juice_in_blend / total_juice) * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_pear_juice_percentage_l336_33676


namespace NUMINAMATH_CALUDE_max_product_sum_l336_33640

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∃ (q : ℕ), q = A * M * C + A * M + M * C + C * A ∧
   ∀ (q' : ℕ), q' = A * M * C + A * M + M * C + C * A → q' ≤ q) ∧
  (A * M * C + A * M + M * C + C * A ≤ 200) :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l336_33640


namespace NUMINAMATH_CALUDE_triangle_theorem_l336_33681

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h3 : A + B + C = π)

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) :
  2 * t.a^2 = t.b^2 + t.c^2 ∧ 
  (t.a = 5 ∧ Real.cos t.A = 25/31 → t.a + t.b + t.c = 14) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l336_33681


namespace NUMINAMATH_CALUDE_fraction_transformation_l336_33622

theorem fraction_transformation (a b : ℕ) (h : a < b) :
  (∃ x : ℕ, (a + x : ℚ) / (b + x) = 1 / 2) ∧
  (¬ ∃ y z : ℕ, ((a + y : ℚ) * z) / ((b + y) * z) = 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l336_33622


namespace NUMINAMATH_CALUDE_two_digit_number_reverse_sum_l336_33662

theorem two_digit_number_reverse_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 7 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_reverse_sum_l336_33662


namespace NUMINAMATH_CALUDE_largest_packet_size_l336_33692

theorem largest_packet_size (jonathan_sets elena_sets : ℕ) 
  (h1 : jonathan_sets = 36) 
  (h2 : elena_sets = 60) : 
  Nat.gcd jonathan_sets elena_sets = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_packet_size_l336_33692


namespace NUMINAMATH_CALUDE_scientific_notation_439000_l336_33644

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_439000 :
  toScientificNotation 439000 = ScientificNotation.mk 4.39 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_439000_l336_33644


namespace NUMINAMATH_CALUDE_initial_chicken_wings_l336_33629

theorem initial_chicken_wings 
  (num_friends : ℕ) 
  (additional_wings : ℕ) 
  (wings_per_friend : ℕ) 
  (h1 : num_friends = 4) 
  (h2 : additional_wings = 7) 
  (h3 : wings_per_friend = 4) : 
  num_friends * wings_per_friend - additional_wings = 9 := by
sorry

end NUMINAMATH_CALUDE_initial_chicken_wings_l336_33629


namespace NUMINAMATH_CALUDE_binomial_22_10_l336_33617

theorem binomial_22_10 (h1 : Nat.choose 20 8 = 125970)
                       (h2 : Nat.choose 20 9 = 167960)
                       (h3 : Nat.choose 20 10 = 184756) :
  Nat.choose 22 10 = 646646 := by
  sorry

end NUMINAMATH_CALUDE_binomial_22_10_l336_33617


namespace NUMINAMATH_CALUDE_sum_of_extrema_l336_33683

theorem sum_of_extrema (x y : ℝ) (h : 1 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 4) :
  ∃ (min max : ℝ),
    (∀ z w : ℝ, 1 ≤ z^2 + w^2 ∧ z^2 + w^2 ≤ 4 → min ≤ z^2 - z*w + w^2 ∧ z^2 - z*w + w^2 ≤ max) ∧
    min + max = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extrema_l336_33683


namespace NUMINAMATH_CALUDE_dean_5000th_number_l336_33684

/-- Represents the number of numbers spoken by a player in a given round -/
def numbers_spoken (player : Nat) (round : Nat) : Nat :=
  player + round - 1

/-- Calculates the sum of numbers spoken by all players up to a given round -/
def total_numbers_spoken (round : Nat) : Nat :=
  (1 + 2 + 3 + 4) * round + (0 + 1 + 2 + 3) * (round * (round - 1) / 2)

/-- Calculates the starting number for a player in a given round -/
def start_number (player : Nat) (round : Nat) : Nat :=
  total_numbers_spoken (round - 1) + 
  (if player > 1 then (numbers_spoken 1 round + numbers_spoken 2 round + numbers_spoken 3 round) else 0) + 1

/-- The main theorem to be proved -/
theorem dean_5000th_number : 
  ∃ (round : Nat), start_number 4 round ≤ 5000 ∧ 5000 ≤ start_number 4 round + numbers_spoken 4 round - 1 :=
by sorry

end NUMINAMATH_CALUDE_dean_5000th_number_l336_33684


namespace NUMINAMATH_CALUDE_complement_A_union_B_l336_33605

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x : ℝ | Real.log (x - 2) ≤ 0}

-- State the theorem
theorem complement_A_union_B : 
  (Set.compl A) ∪ B = Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l336_33605


namespace NUMINAMATH_CALUDE_zero_subset_A_l336_33632

def A : Set ℝ := {x | x > -3}

theorem zero_subset_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_zero_subset_A_l336_33632


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l336_33643

/-- Given two positive integers with LCM 36 and ratio 2:3, prove their sum is 30 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 36)
  (h_ratio : a * 3 = b * 2) : 
  a + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l336_33643


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l336_33609

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- Theorem: For a geometric sequence with common ratio q, if a_1 + a_3 = 10 and a_4 + a_6 = 5/4, then q = 1/2 -/
theorem geometric_sequence_ratio 
  (a q : ℝ) 
  (h1 : geometric_sequence a q 1 + geometric_sequence a q 3 = 10)
  (h2 : geometric_sequence a q 4 + geometric_sequence a q 6 = 5/4) :
  q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l336_33609


namespace NUMINAMATH_CALUDE_parabola_m_range_l336_33671

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadratic function of the form y = ax² + 4ax + c -/
structure QuadraticFunction where
  a : ℝ
  c : ℝ
  h : a ≠ 0

theorem parabola_m_range 
  (f : QuadraticFunction)
  (A B C : Point)
  (h1 : A.x = m)
  (h2 : B.x = m + 2)
  (h3 : C.x = -2)  -- vertex x-coordinate for y = ax² + 4ax + c is always -2
  (h4 : A.y = f.a * A.x^2 + 4 * f.a * A.x + f.c)
  (h5 : B.y = f.a * B.x^2 + 4 * f.a * B.x + f.c)
  (h6 : C.y = f.a * C.x^2 + 4 * f.a * C.x + f.c)
  (h7 : C.y ≥ B.y)
  (h8 : B.y > A.y)
  : m < -3 := by sorry

end NUMINAMATH_CALUDE_parabola_m_range_l336_33671


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l336_33650

theorem solution_satisfies_system :
  ∃ (x y z : ℝ), 
    (3 * x + 2 * y - z = 1) ∧
    (4 * x - 5 * y + 3 * z = 11) ∧
    (x = 1 ∧ y = 1 ∧ z = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l336_33650


namespace NUMINAMATH_CALUDE_abc_def_ratio_l336_33659

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6) :
  a * b * c / (d * e * f) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l336_33659


namespace NUMINAMATH_CALUDE_inequality_solution_set_l336_33690

theorem inequality_solution_set (x : ℝ) : 
  (1 / (x^2 + 1) > 4 / x + 25 / 10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l336_33690


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l336_33628

theorem hyperbola_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let c := Real.sqrt (a^2 + b^2)  -- focal distance
  (c / a = e) →
  (∀ a' b', a' > 0 → b' > 0 → c / a' = e → (b'^2 + 1) / (3 * a') ≥ (b^2 + 1) / (3 * a)) →
  (b^2 + 1) / (3 * a) = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l336_33628


namespace NUMINAMATH_CALUDE_fraction_multiplication_simplification_l336_33651

theorem fraction_multiplication_simplification :
  (4 * 7) / (14 * 10) * (5 * 10 * 14) / (4 * 5 * 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_simplification_l336_33651


namespace NUMINAMATH_CALUDE_three_families_ten_lines_form_150_triangles_l336_33697

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : ℕ)

/-- Calculates the maximum number of triangles formed by three families of parallel lines -/
def max_triangles (f1 f2 f3 : LineFamily) : ℕ :=
  sorry

/-- Theorem stating that three families of 10 parallel lines form 150 triangles -/
theorem three_families_ten_lines_form_150_triangles :
  ∀ (f1 f2 f3 : LineFamily),
    f1.count = 10 → f2.count = 10 → f3.count = 10 →
    max_triangles f1 f2 f3 = 150 :=
by sorry

end NUMINAMATH_CALUDE_three_families_ten_lines_form_150_triangles_l336_33697


namespace NUMINAMATH_CALUDE_quadratic_congruence_solutions_l336_33699

theorem quadratic_congruence_solutions (x : ℕ) : 
  (x^2 + x - 6) % 143 = 0 ↔ x ∈ ({2, 41, 101, 140} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_congruence_solutions_l336_33699


namespace NUMINAMATH_CALUDE_cow_count_is_fifteen_l336_33639

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (ac : AnimalCount) : ℕ :=
  2 * ac.ducks + 4 * ac.cows

/-- The total number of heads in the group -/
def totalHeads (ac : AnimalCount) : ℕ :=
  ac.ducks + ac.cows

/-- The main theorem stating that the number of cows is 15 -/
theorem cow_count_is_fifteen :
  ∃ (ac : AnimalCount), totalLegs ac = 2 * totalHeads ac + 30 ∧ ac.cows = 15 := by
  sorry

#check cow_count_is_fifteen

end NUMINAMATH_CALUDE_cow_count_is_fifteen_l336_33639


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l336_33653

theorem sum_of_a_and_b (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l336_33653
