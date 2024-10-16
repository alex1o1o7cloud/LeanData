import Mathlib

namespace NUMINAMATH_CALUDE_angle_sum_in_cyclic_quad_l886_88624

-- Define the quadrilateral ABCD and point E
variable (A B C D E : Point)

-- Define the cyclic property of quadrilateral ABCD
def is_cyclic_quad (A B C D : Point) : Prop := sorry

-- Define angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem angle_sum_in_cyclic_quad 
  (h_cyclic : is_cyclic_quad A B C D)
  (h_angle_A : angle_measure B A C = 40)
  (h_equal_angles : angle_measure C E D = angle_measure E C D) :
  angle_measure A B C + angle_measure A D C = 160 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_cyclic_quad_l886_88624


namespace NUMINAMATH_CALUDE_system_solution_l886_88675

theorem system_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^(Real.log y / Real.log 10) + y^(Real.log (x^(1/2)) / Real.log 10) = 110) 
  (h2 : x * y = 1000) : 
  (x = 10 ∧ y = 100) ∨ (x = 100 ∧ y = 10) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l886_88675


namespace NUMINAMATH_CALUDE_part_one_part_two_l886_88646

noncomputable section

-- Define the functions
def f (b : ℝ) (x : ℝ) : ℝ := (2*x + b) * Real.exp x
def F (b : ℝ) (x : ℝ) : ℝ := b*x - Real.log x
def g (b : ℝ) (x : ℝ) : ℝ := b*x^2 - 2*x - F b x

-- Part 1
theorem part_one (b : ℝ) :
  b < 0 ∧ 
  (∃ (M : Set ℝ), ∀ (x y : ℝ), x ∈ M → y ∈ M → x < y → 
    ((f b x < f b y ↔ F b x < F b y) ∨ (f b x > f b y ↔ F b x > F b y))) →
  b < -2 :=
sorry

-- Part 2
theorem part_two (b : ℝ) :
  b > 0 ∧ 
  (∀ x ∈ Set.Icc 1 (Real.exp 1), g b x ≥ -2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), g b x = -2) →
  b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l886_88646


namespace NUMINAMATH_CALUDE_complex_equation_solution_l886_88633

theorem complex_equation_solution (a : ℝ) : 
  (1 : ℂ) + a * I = I * (2 - I) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l886_88633


namespace NUMINAMATH_CALUDE_total_initial_money_l886_88672

/-- Represents the money redistribution game between three friends --/
structure MoneyRedistribution where
  amy_initial : ℝ
  jan_initial : ℝ
  toy_initial : ℝ
  amy_final : ℝ
  jan_final : ℝ
  toy_final : ℝ

/-- The theorem stating the total initial amount of money --/
theorem total_initial_money (game : MoneyRedistribution) 
  (h1 : game.amy_initial = 50)
  (h2 : game.toy_initial = 50)
  (h3 : game.amy_final = game.amy_initial)
  (h4 : game.toy_final = game.toy_initial)
  (h5 : game.amy_final = 2 * (2 * (game.amy_initial - (game.jan_initial + game.toy_initial))))
  (h6 : game.jan_final = 2 * (2 * game.jan_initial - (2 * game.toy_initial + (game.amy_initial - (game.jan_initial + game.toy_initial)))))
  (h7 : game.toy_final = 2 * game.toy_initial - (game.amy_final + game.jan_final - (2 * game.toy_initial + 2 * (game.amy_initial - (game.jan_initial + game.toy_initial)))))
  : game.amy_initial + game.jan_initial + game.toy_initial = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_total_initial_money_l886_88672


namespace NUMINAMATH_CALUDE_travel_time_calculation_l886_88602

/-- Given a speed of 65 km/hr and a distance of 195 km, the travel time is 3 hours -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (time : ℝ) :
  speed = 65 → distance = 195 → time = distance / speed → time = 3 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l886_88602


namespace NUMINAMATH_CALUDE_system_solution_l886_88662

theorem system_solution (a b c d x y z u : ℝ) : 
  (a^3 * x + a^2 * y + a * z + u = 0) →
  (b^3 * x + b^2 * y + b * z + u = 0) →
  (c^3 * x + c^2 * y + c * z + u = 0) →
  (d^3 * x + d^2 * y + d * z + u = 1) →
  (x = 1 / ((d-a)*(d-b)*(d-c))) →
  (y = -(a+b+c) / ((d-a)*(d-b)*(d-c))) →
  (z = (a*b + b*c + c*a) / ((d-a)*(d-b)*(d-c))) →
  (u = -(a*b*c) / ((d-a)*(d-b)*(d-c))) →
  (a ≠ d) → (b ≠ d) → (c ≠ d) →
  (a^3 * x + a^2 * y + a * z + u = 0) ∧
  (b^3 * x + b^2 * y + b * z + u = 0) ∧
  (c^3 * x + c^2 * y + c * z + u = 0) ∧
  (d^3 * x + d^2 * y + d * z + u = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l886_88662


namespace NUMINAMATH_CALUDE_line_parallel_to_y_axis_l886_88619

/-- A line passing through the point (-1, 3) and parallel to the y-axis has the equation x = -1 -/
theorem line_parallel_to_y_axis (line : Set (ℝ × ℝ)) : 
  ((-1, 3) ∈ line) → 
  (∀ (x y₁ y₂ : ℝ), ((x, y₁) ∈ line ∧ (x, y₂) ∈ line) → y₁ = y₂) →
  (line = {p : ℝ × ℝ | p.1 = -1}) :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_y_axis_l886_88619


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l886_88616

/-- Represents the number of households selected in a stratum -/
structure StratumSample where
  total : ℕ
  selected : ℕ

/-- Represents the stratified sample of households -/
structure StratifiedSample where
  high_income : StratumSample
  middle_income : StratumSample
  low_income : StratumSample

def total_households (s : StratifiedSample) : ℕ :=
  s.high_income.total + s.middle_income.total + s.low_income.total

def total_selected (s : StratifiedSample) : ℕ :=
  s.high_income.selected + s.middle_income.selected + s.low_income.selected

theorem stratified_sample_theorem (s : StratifiedSample) :
  s.high_income.total = 120 →
  s.middle_income.total = 200 →
  s.low_income.total = 160 →
  s.high_income.selected = 6 →
  total_households s = 480 →
  total_selected s = 24 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l886_88616


namespace NUMINAMATH_CALUDE_mike_earnings_l886_88637

/-- Mike's total earnings for the week -/
def total_earnings (first_job_wages : ℕ) (second_job_hours : ℕ) (second_job_rate : ℕ) : ℕ :=
  first_job_wages + second_job_hours * second_job_rate

/-- Theorem stating Mike's total earnings for the week -/
theorem mike_earnings : 
  total_earnings 52 12 9 = 160 := by
  sorry

end NUMINAMATH_CALUDE_mike_earnings_l886_88637


namespace NUMINAMATH_CALUDE_council_arrangements_l886_88623

/-- The number of distinct arrangements of chairs and stools around a round table -/
def distinct_arrangements (chairs : ℕ) (stools : ℕ) : ℕ :=
  Nat.choose (chairs + stools - 1) (stools - 1)

/-- Theorem: There are 55 distinct arrangements of 9 chairs and 3 stools around a round table -/
theorem council_arrangements :
  distinct_arrangements 9 3 = 55 := by
  sorry

end NUMINAMATH_CALUDE_council_arrangements_l886_88623


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l886_88638

theorem right_triangle_segment_ratio (x y z u v : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
    (h4 : x^2 + y^2 = z^2) (h5 : x * z = u * (u + v)) (h6 : y * z = v * (u + v)) (h7 : 3 * y = 4 * x) :
    9 * v = 16 * u := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l886_88638


namespace NUMINAMATH_CALUDE_planned_speed_calculation_l886_88659

theorem planned_speed_calculation (distance : ℝ) (speed_multiplier : ℝ) (time_saved : ℝ) 
  (h1 : distance = 180)
  (h2 : speed_multiplier = 1.2)
  (h3 : time_saved = 0.5)
  : ∃ v : ℝ, v > 0 ∧ distance / v = distance / (speed_multiplier * v) + time_saved ∧ v = 60 := by
  sorry

end NUMINAMATH_CALUDE_planned_speed_calculation_l886_88659


namespace NUMINAMATH_CALUDE_special_sequence_14th_term_l886_88647

/-- A sequence of positive real numbers satisfying certain conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (a 2 = 2) ∧ 
  (a 8 = 8) ∧ 
  (∀ n ≥ 2, Real.sqrt (a (n - 1)) * Real.sqrt (a (n + 1)) = a n)

/-- The 14th term of the special sequence is 32 -/
theorem special_sequence_14th_term (a : ℕ → ℝ) (h : SpecialSequence a) : a 14 = 32 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_14th_term_l886_88647


namespace NUMINAMATH_CALUDE_packages_per_box_l886_88696

/-- Given that Julie bought two boxes of standard paper, each package contains 250 sheets,
    25 sheets are used per newspaper, and 100 newspapers can be printed,
    prove that there are 5 packages in each box. -/
theorem packages_per_box (boxes : ℕ) (sheets_per_package : ℕ) (sheets_per_newspaper : ℕ) (total_newspapers : ℕ) :
  boxes = 2 →
  sheets_per_package = 250 →
  sheets_per_newspaper = 25 →
  total_newspapers = 100 →
  (boxes * sheets_per_package * (total_newspapers * sheets_per_newspaper / (boxes * sheets_per_package)) : ℚ) = total_newspapers * sheets_per_newspaper →
  total_newspapers * sheets_per_newspaper / (boxes * sheets_per_package) = 5 := by
  sorry

end NUMINAMATH_CALUDE_packages_per_box_l886_88696


namespace NUMINAMATH_CALUDE_highest_page_number_with_19_sevens_l886_88656

/-- Counts the number of occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ :=
  sorry

/-- Counts the total occurrences of a digit in a range of natural numbers -/
def countDigitInRange (start finish : ℕ) (d : ℕ) : ℕ :=
  sorry

/-- The highest page number that can be reached with a given number of sevens -/
def highestPageNumber (numSevens : ℕ) : ℕ :=
  sorry

theorem highest_page_number_with_19_sevens :
  highestPageNumber 19 = 99 :=
sorry

end NUMINAMATH_CALUDE_highest_page_number_with_19_sevens_l886_88656


namespace NUMINAMATH_CALUDE_rowing_speed_in_still_water_l886_88601

/-- Represents the rowing scenario with upstream and downstream times, current speed, and still water speed. -/
structure RowingScenario where
  upstream_time : ℝ
  downstream_time : ℝ
  current_speed : ℝ
  still_water_speed : ℝ

/-- Theorem stating that given the conditions, the man's rowing speed in still water is 3.6 km/hr. -/
theorem rowing_speed_in_still_water 
  (scenario : RowingScenario)
  (h1 : scenario.upstream_time = 2 * scenario.downstream_time)
  (h2 : scenario.current_speed = 1.2)
  : scenario.still_water_speed = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_rowing_speed_in_still_water_l886_88601


namespace NUMINAMATH_CALUDE_x0_value_l886_88617

open Real

noncomputable def f (x : ℝ) : ℝ := x * (2016 + log x)

theorem x0_value (x0 : ℝ) (h : deriv f x0 = 2017) : x0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l886_88617


namespace NUMINAMATH_CALUDE_bus_journey_time_l886_88611

/-- Represents the journey of Xiao Ming to school -/
structure Journey where
  subway_time : ℕ -- Time taken by subway in minutes
  bus_time : ℕ -- Time taken by bus in minutes
  transfer_time : ℕ -- Time taken for transfer in minutes
  total_time : ℕ -- Total time of the journey in minutes

/-- Theorem stating the correct time spent on the bus -/
theorem bus_journey_time (j : Journey) 
  (h1 : j.subway_time = 30)
  (h2 : j.bus_time = 50)
  (h3 : j.transfer_time = 6)
  (h4 : j.total_time = 40)
  (h5 : j.total_time = j.subway_time + j.bus_time + j.transfer_time) :
  ∃ (actual_bus_time : ℕ), actual_bus_time = 10 ∧ 
    j.total_time = (j.subway_time - (j.subway_time - actual_bus_time)) + actual_bus_time + j.transfer_time :=
by sorry

end NUMINAMATH_CALUDE_bus_journey_time_l886_88611


namespace NUMINAMATH_CALUDE_complex_equation_solution_l886_88609

-- Define the operation
def determinant (a b c d : ℂ) : ℂ := a * d - b * c

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), determinant 1 (-1) z (z * Complex.I) = 4 + 2 * Complex.I ∧ z = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l886_88609


namespace NUMINAMATH_CALUDE_michaels_fish_count_l886_88680

/-- Given Michael's initial fish count and the number of fish Ben gives him,
    prove that the total number of fish Michael has now is equal to the sum of these two quantities. -/
theorem michaels_fish_count (initial : Real) (given : Real) :
  initial + given = initial + given :=
by sorry

end NUMINAMATH_CALUDE_michaels_fish_count_l886_88680


namespace NUMINAMATH_CALUDE_mona_unique_players_l886_88685

/-- The number of unique players Mona grouped with in a video game --/
def unique_players (groups : ℕ) (players_per_group : ℕ) (groups_with_two_repeats : ℕ) (groups_with_one_repeat : ℕ) : ℕ :=
  groups * players_per_group - (2 * groups_with_two_repeats + groups_with_one_repeat)

/-- Theorem stating the number of unique players Mona grouped with --/
theorem mona_unique_players :
  unique_players 25 4 8 5 = 79 := by
  sorry

end NUMINAMATH_CALUDE_mona_unique_players_l886_88685


namespace NUMINAMATH_CALUDE_closest_fraction_l886_88663

def medals_won : ℚ := 23 / 120

def fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction :
  ∃ (x : ℚ), x ∈ fractions ∧
  ∀ (y : ℚ), y ∈ fractions → |medals_won - x| ≤ |medals_won - y| ∧
  x = 1/5 :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_l886_88663


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l886_88692

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + (m+1)*x + 16 = (x + a)^2) → m = 7 ∨ m = -9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l886_88692


namespace NUMINAMATH_CALUDE_contractor_absent_days_l886_88653

theorem contractor_absent_days 
  (total_days : ℕ) 
  (pay_per_day : ℚ) 
  (fine_per_day : ℚ) 
  (total_received : ℚ) 
  (h1 : total_days = 30)
  (h2 : pay_per_day = 25)
  (h3 : fine_per_day = 7.5)
  (h4 : total_received = 425) :
  ∃ (absent_days : ℕ), 
    absent_days = 10 ∧ 
    (total_days - absent_days) * pay_per_day - absent_days * fine_per_day = total_received :=
by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l886_88653


namespace NUMINAMATH_CALUDE_band_sections_fraction_l886_88628

theorem band_sections_fraction (trumpet_fraction trombone_fraction : ℝ) 
  (h1 : trumpet_fraction = 0.5)
  (h2 : trombone_fraction = 0.125) :
  trumpet_fraction + trombone_fraction = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_band_sections_fraction_l886_88628


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l886_88670

/-- The charge per color copy at print shop X -/
def charge_x : ℚ := 125/100

/-- The charge per color copy at print shop Y -/
def charge_y : ℚ := 275/100

/-- The number of color copies -/
def num_copies : ℕ := 40

/-- The difference in charges between print shop Y and X for num_copies color copies -/
def charge_difference : ℚ := num_copies * charge_y - num_copies * charge_x

theorem print_shop_charge_difference : charge_difference = 60 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charge_difference_l886_88670


namespace NUMINAMATH_CALUDE_students_walking_distance_l886_88661

/-- The problem of finding the distance students need to walk --/
theorem students_walking_distance 
  (teacher_speed : ℝ) 
  (teacher_initial_distance : ℝ)
  (student1_initial_distance : ℝ)
  (student2_initial_distance : ℝ)
  (student3_initial_distance : ℝ)
  (h1 : teacher_speed = 1.5)
  (h2 : teacher_initial_distance = 235)
  (h3 : student1_initial_distance = 87)
  (h4 : student2_initial_distance = 59)
  (h5 : student3_initial_distance = 26) :
  ∃ x : ℝ, 
    x = 42 ∧ 
    teacher_initial_distance - teacher_speed * x = 
      (student1_initial_distance - x) + 
      (student2_initial_distance - x) + 
      (student3_initial_distance - x) := by
  sorry

end NUMINAMATH_CALUDE_students_walking_distance_l886_88661


namespace NUMINAMATH_CALUDE_polar_equation_perpendicular_line_l886_88631

/-- The polar equation of a line passing through (2,0) and perpendicular to the polar axis -/
theorem polar_equation_perpendicular_line (ρ θ : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ y = 0 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (∀ (x y : ℝ), x = 2 → y = ρ * Real.sin θ) →
  ρ * Real.cos θ = 2 :=
by sorry

end NUMINAMATH_CALUDE_polar_equation_perpendicular_line_l886_88631


namespace NUMINAMATH_CALUDE_lumberjack_chopped_25_trees_l886_88665

/-- Represents the lumberjack's work -/
structure LumberjackWork where
  logs_per_tree : ℕ
  firewood_per_log : ℕ
  total_firewood : ℕ

/-- Calculates the number of trees chopped based on the lumberjack's work -/
def trees_chopped (work : LumberjackWork) : ℕ :=
  work.total_firewood / (work.logs_per_tree * work.firewood_per_log)

/-- Theorem stating that given the specific conditions, the lumberjack chopped 25 trees -/
theorem lumberjack_chopped_25_trees :
  let work := LumberjackWork.mk 4 5 500
  trees_chopped work = 25 := by
  sorry

#eval trees_chopped (LumberjackWork.mk 4 5 500)

end NUMINAMATH_CALUDE_lumberjack_chopped_25_trees_l886_88665


namespace NUMINAMATH_CALUDE_inequality_proof_l886_88681

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * ((x + y + z) / 3)) ^ (1/4) ≤ (((x + y) / 2) * ((y + z) / 2) * ((z + x) / 2)) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l886_88681


namespace NUMINAMATH_CALUDE_symmetric_point_in_third_quadrant_l886_88604

/-- A point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to x-axis -/
def symmetricXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Definition of third quadrant -/
def isThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The main theorem -/
theorem symmetric_point_in_third_quadrant :
  let P : Point := ⟨-2, 1⟩
  let P' := symmetricXAxis P
  isThirdQuadrant P' :=
by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_in_third_quadrant_l886_88604


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l886_88634

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) → 
  a ∈ Set.Ioo 1 5 ∪ Set.singleton 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l886_88634


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l886_88693

/-- Two congruent right circular cones with a sphere inside -/
structure ConeFigure where
  /-- Base radius of each cone -/
  base_radius : ℝ
  /-- Height of each cone -/
  cone_height : ℝ
  /-- Distance from base to intersection point of axes -/
  intersection_distance : ℝ
  /-- Radius of the sphere -/
  sphere_radius : ℝ
  /-- Condition: base radius is 5 -/
  base_radius_eq : base_radius = 5
  /-- Condition: cone height is 10 -/
  cone_height_eq : cone_height = 10
  /-- Condition: intersection distance is 5 -/
  intersection_eq : intersection_distance = 5
  /-- Condition: sphere lies within both cones -/
  sphere_within_cones : sphere_radius ≤ intersection_distance

/-- The maximum possible value of r^2 is 80 -/
theorem max_sphere_radius_squared (cf : ConeFigure) : 
  ∃ (max_r : ℝ), ∀ (r : ℝ), cf.sphere_radius = r → r^2 ≤ max_r^2 ∧ max_r^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_l886_88693


namespace NUMINAMATH_CALUDE_isabel_piggy_bank_l886_88694

theorem isabel_piggy_bank (initial_amount : ℝ) (toy_fraction : ℝ) (book_fraction : ℝ) : 
  initial_amount = 204 →
  toy_fraction = 1/2 →
  book_fraction = 1/2 →
  initial_amount * (1 - toy_fraction) * (1 - book_fraction) = 51 := by
sorry

end NUMINAMATH_CALUDE_isabel_piggy_bank_l886_88694


namespace NUMINAMATH_CALUDE_circles_intersect_l886_88654

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 8*p.2 - 8 = 0}

/-- Circle C₂ with equation x² + y² - 4x - 5 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 5 = 0}

/-- The center of circle C₁ -/
def center_C₁ : ℝ × ℝ := (-1, -4)

/-- The radius of circle C₁ -/
def radius_C₁ : ℝ := 5

/-- The center of circle C₂ -/
def center_C₂ : ℝ × ℝ := (2, 0)

/-- The radius of circle C₂ -/
def radius_C₂ : ℝ := 3

/-- Theorem stating that circles C₁ and C₂ are intersecting -/
theorem circles_intersect : ∃ p : ℝ × ℝ, p ∈ C₁ ∩ C₂ := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l886_88654


namespace NUMINAMATH_CALUDE_joshua_finish_time_difference_l886_88606

/-- Race parameters -/
def race_length : ℕ := 15
def uphill_length : ℕ := 5
def flat_length : ℕ := race_length - uphill_length

/-- Runner speeds (in minutes per mile) -/
def malcolm_flat_speed : ℕ := 4
def joshua_flat_speed : ℕ := 6
def malcolm_uphill_additional : ℕ := 2
def joshua_uphill_additional : ℕ := 3

/-- Calculate total race time for a runner -/
def total_race_time (flat_speed uphill_additional : ℕ) : ℕ :=
  flat_speed * flat_length + (flat_speed + uphill_additional) * uphill_length

/-- Theorem: Joshua finishes 35 minutes after Malcolm -/
theorem joshua_finish_time_difference :
  total_race_time joshua_flat_speed joshua_uphill_additional -
  total_race_time malcolm_flat_speed malcolm_uphill_additional = 35 := by
  sorry


end NUMINAMATH_CALUDE_joshua_finish_time_difference_l886_88606


namespace NUMINAMATH_CALUDE_solve_for_q_l886_88607

theorem solve_for_q (k l q : ℚ) : 
  (7 / 8 = k / 96) → 
  (7 / 8 = (k + l) / 112) → 
  (7 / 8 = (q - l) / 144) → 
  q = 140 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l886_88607


namespace NUMINAMATH_CALUDE_candy_box_problem_l886_88641

/-- Given the number of chocolate boxes, caramel boxes, and total pieces of candy,
    calculate the number of pieces in each box. -/
def pieces_per_box (chocolate_boxes caramel_boxes total_pieces : ℕ) : ℕ :=
  total_pieces / (chocolate_boxes + caramel_boxes)

/-- Theorem stating that given 7 boxes of chocolate candy, 3 boxes of caramel candy,
    and a total of 80 pieces, there are 8 pieces in each box. -/
theorem candy_box_problem :
  pieces_per_box 7 3 80 = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_problem_l886_88641


namespace NUMINAMATH_CALUDE_standing_students_count_l886_88649

/-- Given a school meeting with the following conditions:
  * total_attendees: The total number of attendees at the meeting
  * seated_students: The number of seated students
  * seated_teachers: The number of seated teachers

  This theorem proves that the number of standing students is equal to 25.
-/
theorem standing_students_count
  (total_attendees : Nat)
  (seated_students : Nat)
  (seated_teachers : Nat)
  (h1 : total_attendees = 355)
  (h2 : seated_students = 300)
  (h3 : seated_teachers = 30) :
  total_attendees - (seated_students + seated_teachers) = 25 := by
  sorry

#check standing_students_count

end NUMINAMATH_CALUDE_standing_students_count_l886_88649


namespace NUMINAMATH_CALUDE_cubic_root_sum_l886_88679

theorem cubic_root_sum (a b c d : ℝ) (h1 : a ≠ 0) 
  (h2 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
  (h3 : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -13 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l886_88679


namespace NUMINAMATH_CALUDE_T1_T2_T3_l886_88635

-- Define the types for pib and maa
variable (Pib Maa : Type)

-- Define the belongs_to relation
variable (belongs_to : Maa → Pib → Prop)

-- P1: Every pib is a collection of maas
axiom P1 : ∀ p : Pib, ∃ m : Maa, belongs_to m p

-- P2: Any two distinct pibs have one and only one maa in common
axiom P2 : ∀ p1 p2 : Pib, p1 ≠ p2 → ∃! m : Maa, belongs_to m p1 ∧ belongs_to m p2

-- P3: Every maa belongs to two and only two pibs
axiom P3 : ∀ m : Maa, ∃! p1 p2 : Pib, p1 ≠ p2 ∧ belongs_to m p1 ∧ belongs_to m p2

-- P4: There are exactly four pibs
axiom P4 : ∃! (a b c d : Pib), ∀ p : Pib, p = a ∨ p = b ∨ p = c ∨ p = d

-- T1: There are exactly six maas
theorem T1 : ∃! (a b c d e f : Maa), ∀ m : Maa, m = a ∨ m = b ∨ m = c ∨ m = d ∨ m = e ∨ m = f :=
sorry

-- T2: There are exactly three maas in each pib
theorem T2 : ∀ p : Pib, ∃! (a b c : Maa), (∀ m : Maa, belongs_to m p ↔ (m = a ∨ m = b ∨ m = c)) :=
sorry

-- T3: For each maa there is exactly one other maa not in the same pib with it
theorem T3 : ∀ m1 : Maa, ∃! m2 : Maa, m1 ≠ m2 ∧ ∀ p : Pib, ¬(belongs_to m1 p ∧ belongs_to m2 p) :=
sorry

end NUMINAMATH_CALUDE_T1_T2_T3_l886_88635


namespace NUMINAMATH_CALUDE_red_crayons_count_l886_88664

/-- Represents the number of crayons of each color in a crayon box. -/
structure CrayonBox where
  total : ℕ
  blue : ℕ
  green : ℕ
  pink : ℕ
  red : ℕ

/-- Calculates the number of red crayons in a crayon box. -/
def redCrayons (box : CrayonBox) : ℕ :=
  box.total - (box.blue + box.green + box.pink)

/-- Theorem stating the number of red crayons in the given crayon box. -/
theorem red_crayons_count (box : CrayonBox) 
  (h1 : box.total = 24)
  (h2 : box.blue = 6)
  (h3 : box.green = 2 * box.blue / 3)
  (h4 : box.pink = 6) :
  redCrayons box = 8 := by
  sorry

#eval redCrayons { total := 24, blue := 6, green := 4, pink := 6, red := 8 }

end NUMINAMATH_CALUDE_red_crayons_count_l886_88664


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l886_88615

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l886_88615


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l886_88690

/-- Given a line segment with one endpoint (5,4) and midpoint (3.5,10.5),
    the sum of the coordinates of the other endpoint is 19. -/
theorem endpoint_coordinate_sum : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun x₁ y₁ x_mid y_mid x₂ y₂ =>
    x₁ = 5 ∧ y₁ = 4 ∧ x_mid = 3.5 ∧ y_mid = 10.5 ∧
    x_mid = (x₁ + x₂) / 2 ∧ y_mid = (y₁ + y₂) / 2 →
    x₂ + y₂ = 19

/-- Proof of the theorem -/
theorem endpoint_coordinate_sum_proof : 
  ∃ x₂ y₂, endpoint_coordinate_sum 5 4 3.5 10.5 x₂ y₂ := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l886_88690


namespace NUMINAMATH_CALUDE_bankers_discount_calculation_l886_88655

/-- Banker's discount calculation -/
theorem bankers_discount_calculation 
  (present_worth : ℚ) 
  (true_discount : ℚ) 
  (h1 : present_worth = 400)
  (h2 : true_discount = 20) : 
  (true_discount * (present_worth + true_discount)) / present_worth = 21 :=
by
  sorry

#check bankers_discount_calculation

end NUMINAMATH_CALUDE_bankers_discount_calculation_l886_88655


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l886_88620

-- Statement 1
theorem inequality_one (a : ℝ) (h : a > 3) : a + 4 / (a - 3) ≥ 7 := by
  sorry

-- Statement 2
theorem inequality_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  4 / x + 9 / y ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l886_88620


namespace NUMINAMATH_CALUDE_james_birthday_stickers_l886_88639

/-- The number of stickers James got for his birthday -/
def birthday_stickers (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem james_birthday_stickers :
  birthday_stickers 39 61 = 22 := by
  sorry

end NUMINAMATH_CALUDE_james_birthday_stickers_l886_88639


namespace NUMINAMATH_CALUDE_full_seasons_count_l886_88643

/-- The number of days until the final season premiere -/
def days_until_premiere : ℕ := 10

/-- The number of episodes per season -/
def episodes_per_season : ℕ := 15

/-- The number of episodes Joe watches per day -/
def episodes_per_day : ℕ := 6

/-- The number of full seasons already aired -/
def full_seasons : ℕ := (days_until_premiere * episodes_per_day) / episodes_per_season

theorem full_seasons_count : full_seasons = 4 := by
  sorry

end NUMINAMATH_CALUDE_full_seasons_count_l886_88643


namespace NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l886_88650

-- Define the number of sides on each die
def dice_sides : ℕ := 8

-- Define the set of possible prime sums
def prime_sums : Set ℕ := {2, 3, 5, 7, 11, 13}

-- Define a function to count favorable outcomes
def count_favorable_outcomes : ℕ := 29

-- Define the total number of possible outcomes
def total_outcomes : ℕ := dice_sides * dice_sides

-- Theorem statement
theorem probability_prime_sum_two_dice :
  (count_favorable_outcomes : ℚ) / total_outcomes = 29 / 64 :=
sorry

end NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l886_88650


namespace NUMINAMATH_CALUDE_james_cheezits_consumption_l886_88627

/-- Represents the number of bags of Cheezits James ate -/
def bags_of_cheezits : ℕ := sorry

/-- Represents the weight of each bag of Cheezits in ounces -/
def bag_weight : ℕ := 2

/-- Represents the number of calories in an ounce of Cheezits -/
def calories_per_ounce : ℕ := 150

/-- Represents the duration of James' run in minutes -/
def run_duration : ℕ := 40

/-- Represents the number of calories burned per minute during the run -/
def calories_burned_per_minute : ℕ := 12

/-- Represents the excess calories James consumed -/
def excess_calories : ℕ := 420

theorem james_cheezits_consumption :
  bags_of_cheezits * (bag_weight * calories_per_ounce) - 
  (run_duration * calories_burned_per_minute) = excess_calories ∧
  bags_of_cheezits = 3 := by sorry

end NUMINAMATH_CALUDE_james_cheezits_consumption_l886_88627


namespace NUMINAMATH_CALUDE_line_perp_to_plane_perp_to_line_in_plane_l886_88677

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- State the theorem
theorem line_perp_to_plane_perp_to_line_in_plane
  (a b : Line) (α : Plane)
  (h1 : perp_line_plane a α)
  (h2 : subset_line_plane b α) :
  perp_line_line a b :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_perp_to_line_in_plane_l886_88677


namespace NUMINAMATH_CALUDE_triangle_max_area_triangle_area_eight_exists_l886_88686

/-- Triangle with sides a, b, c and area S -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  h1 : 4 * S = a^2 - (b - c)^2
  h2 : b + c = 8
  h3 : a > 0 ∧ b > 0 ∧ c > 0 -- Ensuring positive side lengths

/-- The maximum area of a triangle satisfying the given conditions is 8 -/
theorem triangle_max_area (t : Triangle) : t.S ≤ 8 := by
  sorry

/-- There exists a triangle satisfying the conditions with area equal to 8 -/
theorem triangle_area_eight_exists : ∃ t : Triangle, t.S = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_triangle_area_eight_exists_l886_88686


namespace NUMINAMATH_CALUDE_base_six_addition_l886_88651

/-- Given a base 6 addition problem 3XY_6 + 23_6 = 41X_6, prove that X + Y = 7 in base 10 -/
theorem base_six_addition (X Y : ℕ) : 
  (3 * 6^2 + X * 6 + Y) + (2 * 6 + 3) = 4 * 6^2 + X * 6 → X + Y = 7 :=
by sorry

end NUMINAMATH_CALUDE_base_six_addition_l886_88651


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_third_roll_l886_88657

-- Define a die roll as a number between 1 and 6
def DieRoll : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of two die rolls
def SumTwoDice (roll1 roll2 : DieRoll) : ℕ := roll1.val + roll2.val

-- Define the probability space
def TotalOutcomes : ℕ := 6 * 6 * 6

-- Define the favorable outcomes
def FavorableOutcomes : ℕ := 51

-- The main theorem
theorem probability_sum_greater_than_third_roll :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 17 / 72 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_third_roll_l886_88657


namespace NUMINAMATH_CALUDE_option_C_most_suitable_l886_88658

/-- Represents a survey option -/
inductive SurveyOption
  | A  -- Understanding the sleep time of middle school students nationwide
  | B  -- Understanding the water quality of a river
  | C  -- Surveying the vision of all classmates
  | D  -- Surveying the number of fish in a pond

/-- Defines what makes a survey comprehensive -/
def isComprehensive (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.C => true
  | _ => false

/-- Theorem stating that option C is the most suitable for a comprehensive survey -/
theorem option_C_most_suitable :
  ∀ s : SurveyOption, isComprehensive s → s = SurveyOption.C :=
sorry

end NUMINAMATH_CALUDE_option_C_most_suitable_l886_88658


namespace NUMINAMATH_CALUDE_lunch_spending_l886_88610

theorem lunch_spending (total : ℝ) (difference : ℝ) (friend_spent : ℝ) : 
  total = 72 → difference = 11 → friend_spent = total / 2 + difference / 2 → friend_spent = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_lunch_spending_l886_88610


namespace NUMINAMATH_CALUDE_different_meal_combinations_l886_88660

theorem different_meal_combinations (n : Nat) (h : n = 12) :
  (n * (n - 1) : Nat) = 132 := by
  sorry

end NUMINAMATH_CALUDE_different_meal_combinations_l886_88660


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l886_88698

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_roots : 3 * (a 3)^2 - 11 * (a 3) + 9 = 0 ∧ 3 * (a 9)^2 - 11 * (a 9) + 9 = 0) :
  (a 6)^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l886_88698


namespace NUMINAMATH_CALUDE_total_silk_dyed_l886_88669

theorem total_silk_dyed (green_silk : ℕ) (pink_silk : ℕ) 
  (h1 : green_silk = 61921) (h2 : pink_silk = 49500) : 
  green_silk + pink_silk = 111421 := by
  sorry

end NUMINAMATH_CALUDE_total_silk_dyed_l886_88669


namespace NUMINAMATH_CALUDE_chess_group_players_l886_88603

theorem chess_group_players (n : ℕ) : n > 0 →
  (n * (n - 1)) / 2 = 21 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_players_l886_88603


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l886_88689

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def quadratic_roots (a b c : ℤ) : Set ℤ :=
  {x : ℤ | a * x^2 + b * x + c = 0}

theorem unique_k_for_prime_roots : 
  ∃! k : ℤ, ∀ x ∈ quadratic_roots 1 (-76) k, is_prime (x.natAbs) :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l886_88689


namespace NUMINAMATH_CALUDE_candidate_a_vote_percentage_l886_88605

/-- Represents the percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.6

/-- Represents the percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 1 - democrat_percentage

/-- Represents the percentage of Republican voters expected to vote for candidate A -/
def republican_vote_percentage : ℝ := 0.2

/-- Represents the total percentage of registered voters expected to vote for candidate A -/
def total_vote_percentage : ℝ := 0.5

/-- Represents the percentage of Democratic voters expected to vote for candidate A -/
def democrat_vote_percentage : ℝ := 0.7

theorem candidate_a_vote_percentage :
  democrat_percentage * democrat_vote_percentage +
  republican_percentage * republican_vote_percentage =
  total_vote_percentage :=
sorry

end NUMINAMATH_CALUDE_candidate_a_vote_percentage_l886_88605


namespace NUMINAMATH_CALUDE_vector_parallel_k_l886_88695

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (2, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

theorem vector_parallel_k (k : ℝ) :
  parallel ((k * a.1 - b.1, k * a.2 - b.2) : ℝ × ℝ) (a.1 + 3 * b.1, a.2 + 3 * b.2) →
  k = -1/3 :=
sorry

end NUMINAMATH_CALUDE_vector_parallel_k_l886_88695


namespace NUMINAMATH_CALUDE_probability_graduate_degree_l886_88612

/-- Represents the number of college graduates with a graduate degree -/
def G : ℕ := 3

/-- Represents the number of college graduates without a graduate degree -/
def C : ℕ := 16

/-- Represents the number of non-college graduates -/
def N : ℕ := 24

/-- The ratio of college graduates with a graduate degree to non-college graduates is 1:8 -/
axiom ratio_G_N : G * 8 = N * 1

/-- The ratio of college graduates without a graduate degree to non-college graduates is 2:3 -/
axiom ratio_C_N : C * 3 = N * 2

/-- The probability that a randomly picked college graduate has a graduate degree -/
def prob_graduate_degree : ℚ := G / (G + C)

/-- Theorem: The probability that a randomly picked college graduate has a graduate degree is 3/19 -/
theorem probability_graduate_degree : prob_graduate_degree = 3 / 19 := by sorry

end NUMINAMATH_CALUDE_probability_graduate_degree_l886_88612


namespace NUMINAMATH_CALUDE_probability_non_defective_pencils_l886_88640

/-- The probability of selecting 3 non-defective pencils from a box of 11 pencils with 2 defective ones -/
theorem probability_non_defective_pencils (total_pencils : Nat) (defective_pencils : Nat) (selected_pencils : Nat) :
  total_pencils = 11 →
  defective_pencils = 2 →
  selected_pencils = 3 →
  (Nat.choose (total_pencils - defective_pencils) selected_pencils : ℚ) / 
  (Nat.choose total_pencils selected_pencils : ℚ) = 28 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_defective_pencils_l886_88640


namespace NUMINAMATH_CALUDE_hiking_equipment_cost_l886_88621

/-- Calculates the total cost of hiking equipment --/
theorem hiking_equipment_cost (hoodie_cost : ℚ) (boot_cost : ℚ) (flashlight_percentage : ℚ) (discount_percentage : ℚ) : 
  hoodie_cost = 80 →
  flashlight_percentage = 20 / 100 →
  boot_cost = 110 →
  discount_percentage = 10 / 100 →
  hoodie_cost + (flashlight_percentage * hoodie_cost) + (boot_cost - discount_percentage * boot_cost) = 195 := by
  sorry

end NUMINAMATH_CALUDE_hiking_equipment_cost_l886_88621


namespace NUMINAMATH_CALUDE_rational_cosine_terms_l886_88600

theorem rational_cosine_terms (x : ℝ) 
  (hS : ∃ q : ℚ, (Real.sin (64 * x) + Real.sin (65 * x)) = ↑q)
  (hC : ∃ q : ℚ, (Real.cos (64 * x) + Real.cos (65 * x)) = ↑q) :
  ∃ (q1 q2 : ℚ), Real.cos (64 * x) = ↑q1 ∧ Real.cos (65 * x) = ↑q2 :=
sorry

end NUMINAMATH_CALUDE_rational_cosine_terms_l886_88600


namespace NUMINAMATH_CALUDE_ralph_sock_purchase_l886_88666

/-- Represents the number of pairs of socks at each price point -/
structure SockPurchase where
  two_dollar : ℕ
  three_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the purchase satisfies the given conditions -/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.three_dollar + p.five_dollar = 15 ∧
  2 * p.two_dollar + 3 * p.three_dollar + 5 * p.five_dollar = 36 ∧
  p.two_dollar ≥ 1 ∧ p.three_dollar ≥ 1 ∧ p.five_dollar ≥ 1

/-- The theorem to be proved -/
theorem ralph_sock_purchase :
  ∃ (p : SockPurchase), is_valid_purchase p ∧ p.two_dollar = 11 :=
sorry

end NUMINAMATH_CALUDE_ralph_sock_purchase_l886_88666


namespace NUMINAMATH_CALUDE_triangle_theorem_l886_88626

-- Define a triangle with sides a, b, c opposite to angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given equation
def satisfies_equation (t : Triangle) : Prop :=
  (t.b + t.c) / (2 * t.a * t.b * t.c) + (Real.cos t.B + Real.cos t.C - 2) / (t.b^2 + t.c^2 - t.a^2) = 0

-- Define the arithmetic sequence property
def is_arithmetic_sequence (t : Triangle) : Prop :=
  t.b + t.c = 2 * t.a

-- Define the additional conditions
def has_specific_area_and_cosA (t : Triangle) : Prop :=
  t.a * t.b * Real.sin t.C / 2 = 15 * Real.sqrt 7 / 4 ∧ Real.cos t.A = 9/16

-- State the theorem
theorem triangle_theorem (t : Triangle) :
  satisfies_equation t →
  is_arithmetic_sequence t ∧
  (has_specific_area_and_cosA t → t.a = 5 * Real.sqrt 6 / 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l886_88626


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l886_88625

theorem matrix_inverse_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, 5; 3, 2]
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-2, 5; 3, -7]
  A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l886_88625


namespace NUMINAMATH_CALUDE_brians_trip_distance_l886_88684

/-- Calculates the distance traveled given car efficiency and gas used -/
def distance_traveled (efficiency : ℝ) (gas_used : ℝ) : ℝ :=
  efficiency * gas_used

/-- Proves that Brian's car travels 60 miles given the conditions -/
theorem brians_trip_distance :
  let efficiency : ℝ := 20
  let gas_used : ℝ := 3
  distance_traveled efficiency gas_used = 60 := by
  sorry

end NUMINAMATH_CALUDE_brians_trip_distance_l886_88684


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l886_88671

/-- The rectangular coordinate equation equivalent to the polar equation ρ = 4sin θ -/
theorem polar_to_rectangular (x y ρ θ : ℝ) 
  (h1 : ρ = 4 * Real.sin θ)
  (h2 : x = ρ * Real.cos θ)
  (h3 : y = ρ * Real.sin θ)
  (h4 : ρ^2 = x^2 + y^2) :
  x^2 + y^2 - 4*y = 0 := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l886_88671


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l886_88614

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ)
  (ha : arithmetic_sequence a)
  (hb : arithmetic_sequence b)
  (h : ∀ n : ℕ+, sum_of_arithmetic_sequence a n / sum_of_arithmetic_sequence b n = (n + 1) / (2 * n - 1)) :
  (a 3 + a 7) / (b 1 + b 9) = 10 / 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l886_88614


namespace NUMINAMATH_CALUDE_max_percent_x_correct_l886_88613

/-- The maximum percentage of liquid X in the resulting solution --/
def max_percent_x : ℝ := 1.71

/-- Percentage of liquid X in solution A --/
def percent_x_a : ℝ := 0.8

/-- Percentage of liquid X in solution B --/
def percent_x_b : ℝ := 1.8

/-- Percentage of liquid X in solution C --/
def percent_x_c : ℝ := 3

/-- Percentage of liquid Y in solution A --/
def percent_y_a : ℝ := 2

/-- Percentage of liquid Y in solution B --/
def percent_y_b : ℝ := 1

/-- Percentage of liquid Y in solution C --/
def percent_y_c : ℝ := 0.5

/-- Amount of solution A in grams --/
def amount_a : ℝ := 500

/-- Amount of solution B in grams --/
def amount_b : ℝ := 700

/-- Amount of solution C in grams --/
def amount_c : ℝ := 300

/-- Maximum combined percentage of liquids X and Y in the resulting solution --/
def max_combined_percent : ℝ := 2.5

/-- Theorem stating that the maximum percentage of liquid X in the resulting solution is correct --/
theorem max_percent_x_correct :
  let total_amount := amount_a + amount_b + amount_c
  let amount_x := percent_x_a / 100 * amount_a + percent_x_b / 100 * amount_b + percent_x_c / 100 * amount_c
  let amount_y := percent_y_a / 100 * amount_a + percent_y_b / 100 * amount_b + percent_y_c / 100 * amount_c
  (amount_x + amount_y) / total_amount * 100 ≤ max_combined_percent ∧
  amount_x / total_amount * 100 = max_percent_x :=
by sorry

end NUMINAMATH_CALUDE_max_percent_x_correct_l886_88613


namespace NUMINAMATH_CALUDE_ned_trips_theorem_l886_88673

/-- The number of trays Ned can carry in one trip -/
def trays_per_trip : ℕ := 8

/-- The number of trays on the first table -/
def trays_table1 : ℕ := 27

/-- The number of trays on the second table -/
def trays_table2 : ℕ := 5

/-- The total number of trays Ned needs to pick up -/
def total_trays : ℕ := trays_table1 + trays_table2

/-- The number of trips Ned will make -/
def num_trips : ℕ := (total_trays + trays_per_trip - 1) / trays_per_trip

theorem ned_trips_theorem : num_trips = 4 := by
  sorry

end NUMINAMATH_CALUDE_ned_trips_theorem_l886_88673


namespace NUMINAMATH_CALUDE_squirrels_and_nuts_l886_88622

theorem squirrels_and_nuts :
  let num_squirrels : ℕ := 4
  let num_nuts : ℕ := 2
  num_squirrels - num_nuts = 2 :=
by sorry

end NUMINAMATH_CALUDE_squirrels_and_nuts_l886_88622


namespace NUMINAMATH_CALUDE_hexagon_height_is_six_l886_88697

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a hexagon --/
structure Hexagon where
  height : ℝ

/-- Given a 9x16 rectangle that can be cut into two congruent hexagons
    and repositioned to form a different rectangle, 
    prove that the height of each hexagon is 6 --/
theorem hexagon_height_is_six 
  (original : Rectangle)
  (new : Rectangle)
  (hex1 hex2 : Hexagon)
  (h1 : original.width = 16 ∧ original.height = 9)
  (h2 : hex1 = hex2)
  (h3 : original.width * original.height = new.width * new.height)
  (h4 : new.width = new.height)
  (h5 : hex1.height + hex2.height = new.height)
  : hex1.height = 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_height_is_six_l886_88697


namespace NUMINAMATH_CALUDE_rectangular_enclosure_fence_posts_l886_88687

/-- Calculates the number of fence posts required for a rectangular enclosure --/
def fencePostsRequired (length width postSpacing : ℕ) : ℕ :=
  2 * (length / postSpacing + width / postSpacing) + 4

/-- Proves that the minimum number of fence posts for the given dimensions is 30 --/
theorem rectangular_enclosure_fence_posts :
  fencePostsRequired 72 48 8 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_enclosure_fence_posts_l886_88687


namespace NUMINAMATH_CALUDE_pet_store_birds_l886_88648

/-- The number of bird cages in the pet store -/
def num_cages : ℝ := 6.0

/-- The number of parrots in each cage -/
def parrots_per_cage : ℝ := 6.0

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℝ := 2.0

/-- The total number of birds in the pet store -/
def total_birds : ℝ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds : total_birds = 48.0 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l886_88648


namespace NUMINAMATH_CALUDE_cube_sum_prime_power_l886_88678

theorem cube_sum_prime_power (a b p n : ℕ) : 
  0 < a ∧ 0 < b ∧ 0 < p ∧ 0 < n ∧ Nat.Prime p ∧ a^3 + b^3 = p^n →
  (∃ k : ℕ, 0 < k ∧
    ((a = 2^(k-1) ∧ b = 2^(k-1) ∧ p = 2 ∧ n = 3*k - 2) ∨
     (a = 2 * 3^(k-1) ∧ b = 3^(k-1) ∧ p = 3 ∧ n = 3*k - 1) ∨
     (a = 3^(k-1) ∧ b = 2 * 3^(k-1) ∧ p = 3 ∧ n = 3*k - 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_sum_prime_power_l886_88678


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_150_75_l886_88652

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem largest_two_digit_prime_factor_of_binomial_150_75 :
  (∃ (p : ℕ), p.Prime ∧ p ∣ binomial 150 75 ∧ 10 ≤ p ∧ p < 100) ∧
  (∀ (q : ℕ), q.Prime → q ∣ binomial 150 75 → 10 ≤ q → q < 100 → q ≤ 73) :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_150_75_l886_88652


namespace NUMINAMATH_CALUDE_ballet_arrangement_l886_88629

/-- The number of boys participating in the ballet -/
def num_boys : ℕ := 5

/-- The distance between each girl and her two assigned boys (in meters) -/
def distance : ℕ := 5

/-- The maximum number of girls that can participate in the ballet -/
def max_girls : ℕ := 20

/-- Theorem stating the maximum number of girls that can participate in the ballet -/
theorem ballet_arrangement (n : ℕ) (d : ℕ) (m : ℕ) 
  (h1 : n = num_boys) 
  (h2 : d = distance) 
  (h3 : m = max_girls) :
  m = n * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ballet_arrangement_l886_88629


namespace NUMINAMATH_CALUDE_circle_radius_range_l886_88699

/-- Given two circles O₁ and O₂, with O₁ having radius 1 and O₂ having radius r,
    and the distance between their centers being 5,
    if there exists a point P on O₂ such that PO₁ = 2,
    then the radius r of O₂ is between 3 and 7 inclusive. -/
theorem circle_radius_range (r : ℝ) :
  let O₁ : ℝ × ℝ := (0, 0)  -- Assuming O₁ is at the origin for simplicity
  let O₂ : ℝ × ℝ := (5, 0)  -- Assuming O₂ is on the x-axis
  ∃ (P : ℝ × ℝ), 
    (P.1 - O₂.1)^2 + P.2^2 = r^2 ∧  -- P is on circle O₂
    (P.1 - O₁.1)^2 + P.2^2 = 4      -- PO₁ = 2
  → 3 ≤ r ∧ r ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_range_l886_88699


namespace NUMINAMATH_CALUDE_rafael_remaining_hours_l886_88682

def hours_worked : ℕ := 18
def hourly_rate : ℕ := 20
def total_earnings : ℕ := 760

theorem rafael_remaining_hours : 
  (total_earnings - hours_worked * hourly_rate) / hourly_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_rafael_remaining_hours_l886_88682


namespace NUMINAMATH_CALUDE_max_value_x_minus_y_l886_88630

theorem max_value_x_minus_y (θ : Real) (x y : Real)
  (h1 : x = Real.sin θ)
  (h2 : y = Real.cos θ)
  (h3 : 0 ≤ θ ∧ θ ≤ 2 * Real.pi)
  (h4 : (x^2 + y^2)^2 = x + y) :
  ∃ (θ_max : Real), 
    0 ≤ θ_max ∧ θ_max ≤ 2 * Real.pi ∧
    ∀ (θ' : Real), 0 ≤ θ' ∧ θ' ≤ 2 * Real.pi →
      Real.sin θ' - Real.cos θ' ≤ Real.sin θ_max - Real.cos θ_max ∧
      Real.sin θ_max - Real.cos θ_max = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_y_l886_88630


namespace NUMINAMATH_CALUDE_problem_solution_l886_88645

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := x^3 + (1-a)*x^2 - a*(a+2)*x + b

/-- The derivative of f(x) with respect to x -/
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 2*(1-a)*x - a*(a+2)

theorem problem_solution :
  (∀ a b : ℝ, f a b 0 = 0 ∧ f_derivative a 0 = -3 → (a = -3 ∨ a = 1) ∧ b = 0) ∧
  (∀ a b : ℝ, (∃ x y : ℝ, x ≠ y ∧ f_derivative a x = 0 ∧ f_derivative a y = 0) →
    a < -1/2 ∨ a > -1/2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l886_88645


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l886_88632

theorem geometric_series_first_term 
  (a r : ℝ) 
  (sum_condition : a / (1 - r) = 12)
  (sum_squares_condition : a^2 / (1 - r^2) = 54) :
  a = 72 / 11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l886_88632


namespace NUMINAMATH_CALUDE_total_soccer_games_l886_88674

theorem total_soccer_games (games_attended : ℕ) (games_missed : ℕ) 
  (h1 : games_attended = 32) 
  (h2 : games_missed = 32) : 
  games_attended + games_missed = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_soccer_games_l886_88674


namespace NUMINAMATH_CALUDE_base_seven_sum_l886_88688

/-- Given A, B, and C are non-zero distinct digits in base 7 satisfying the equation, prove B + C = 6 in base 7 -/
theorem base_seven_sum (A B C : ℕ) : 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A < 7 ∧ B < 7 ∧ C < 7 ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B) = 7^3 * A + 7^2 * A + 7 * A →
  B + C = 6 :=
by sorry

end NUMINAMATH_CALUDE_base_seven_sum_l886_88688


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l886_88644

theorem magnitude_of_complex_fraction : Complex.abs (1 / (1 - 2 * Complex.I)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l886_88644


namespace NUMINAMATH_CALUDE_morgan_change_calculation_l886_88683

/-- Calculates the change Morgan receives after buying lunch items and paying with a $50 bill. -/
theorem morgan_change_calculation (hamburger onion_rings smoothie side_salad chocolate_cake : ℚ)
  (h1 : hamburger = 5.75)
  (h2 : onion_rings = 2.50)
  (h3 : smoothie = 3.25)
  (h4 : side_salad = 3.75)
  (h5 : chocolate_cake = 4.20) :
  50 - (hamburger + onion_rings + smoothie + side_salad + chocolate_cake) = 30.55 := by
  sorry

#eval 50 - (5.75 + 2.50 + 3.25 + 3.75 + 4.20)

end NUMINAMATH_CALUDE_morgan_change_calculation_l886_88683


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l886_88667

/-- Represents a repeating decimal with a single repeating digit -/
def repeating_decimal_single (d : ℕ) : ℚ :=
  d / 9

/-- Represents a repeating decimal with two repeating digits -/
def repeating_decimal_double (d : ℕ) : ℚ :=
  d / 99

theorem sum_of_repeating_decimals : 
  repeating_decimal_single 2 + repeating_decimal_double 2 = 8 / 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l886_88667


namespace NUMINAMATH_CALUDE_xy_value_l886_88608

theorem xy_value (x y : ℝ) (h1 : |x| = 2) (h2 : y = 3) (h3 : x * y < 0) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l886_88608


namespace NUMINAMATH_CALUDE_A_D_relationship_l886_88618

-- Define propositions
variable (A B C D : Prop)

-- Define the relationships between propositions
variable (h1 : A → B)
variable (h2 : ¬(B → A))
variable (h3 : B ↔ C)
variable (h4 : C → D)
variable (h5 : ¬(D → C))

-- Theorem to prove
theorem A_D_relationship : (A → D) ∧ ¬(D → A) := by sorry

end NUMINAMATH_CALUDE_A_D_relationship_l886_88618


namespace NUMINAMATH_CALUDE_veggie_votes_l886_88676

theorem veggie_votes (total_votes meat_votes : ℕ) 
  (h1 : total_votes = 672)
  (h2 : meat_votes = 335) : 
  total_votes - meat_votes = 337 := by
sorry

end NUMINAMATH_CALUDE_veggie_votes_l886_88676


namespace NUMINAMATH_CALUDE_direct_proportion_n_value_l886_88668

/-- A direct proportion function passing through (n, -9) with decreasing y as x increases -/
def DirectProportionFunction (n : ℝ) : ℝ → ℝ := fun x ↦ -n * x

theorem direct_proportion_n_value (n : ℝ) :
  (DirectProportionFunction n n = -9) ∧  -- The graph passes through (n, -9)
  (∀ x₁ x₂, x₁ < x₂ → DirectProportionFunction n x₁ > DirectProportionFunction n x₂) →  -- y decreases as x increases
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_direct_proportion_n_value_l886_88668


namespace NUMINAMATH_CALUDE_angelinas_speed_l886_88691

/-- Proves that Angelina's speed from grocery to gym is 24 meters per second -/
theorem angelinas_speed (v : ℝ) : 
  v > 0 →  -- Assume positive speed
  720 / v - 480 / (2 * v) = 40 →  -- Time difference condition
  2 * v = 24 := by
sorry

end NUMINAMATH_CALUDE_angelinas_speed_l886_88691


namespace NUMINAMATH_CALUDE_extreme_point_property_and_max_value_l886_88642

open Real

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2)^3 - a * x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := |f a x|

-- Theorem statement
theorem extreme_point_property_and_max_value (a : ℝ) :
  a > 0 →
  ∃ x₀ x₁ : ℝ,
    x₀ ≠ x₁ ∧
    (∀ x : ℝ, f a x₀ ≥ f a x ∨ f a x₀ ≤ f a x) ∧
    f a x₀ = f a x₁ →
    x₁ + 2 * x₀ = 6 ∧
    (∀ x : ℝ, x ∈ Set.Icc 0 6 → g a x ≤ 40) ∧
    (∃ x : ℝ, x ∈ Set.Icc 0 6 ∧ g a x = 40) →
    a = 4 ∨ a = 12 :=
by sorry

end NUMINAMATH_CALUDE_extreme_point_property_and_max_value_l886_88642


namespace NUMINAMATH_CALUDE_positive_correlation_from_arrangement_l886_88636

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A scatter plot is a list of points -/
def ScatterPlot := List Point

/-- 
  A function that determines if a scatter plot has a general 
  bottom-left to top-right arrangement 
-/
def isBottomLeftToTopRight (plot : ScatterPlot) : Prop :=
  sorry

/-- 
  A function that calculates the correlation coefficient 
  between x and y coordinates in a scatter plot
-/
def correlationCoefficient (plot : ScatterPlot) : ℝ :=
  sorry

/-- 
  Theorem: If a scatter plot has a general bottom-left to top-right arrangement,
  then the correlation between x and y coordinates is positive
-/
theorem positive_correlation_from_arrangement (plot : ScatterPlot) :
  isBottomLeftToTopRight plot → correlationCoefficient plot > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_correlation_from_arrangement_l886_88636
