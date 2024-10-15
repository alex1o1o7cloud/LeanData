import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l1452_145231

theorem problem_solution (x y z : ℚ) 
  (eq1 : 102 * x - 5 * y = 25)
  (eq2 : 3 * y - x = 10)
  (eq3 : z^2 = y - x) : 
  x = 125 / 301 ∧ 10 - x = 2885 / 301 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1452_145231


namespace NUMINAMATH_CALUDE_james_candy_bar_sales_l1452_145210

/-- Proves that James sells 5 boxes of candy bars given the conditions of the fundraiser -/
theorem james_candy_bar_sales :
  let boxes_to_bars : ℕ → ℕ := λ x => 10 * x
  let selling_price : ℚ := 3/2
  let buying_price : ℚ := 1
  let profit_per_bar : ℚ := selling_price - buying_price
  let total_profit : ℚ := 25
  ∃ (num_boxes : ℕ), 
    (boxes_to_bars num_boxes : ℚ) * profit_per_bar = total_profit ∧ 
    num_boxes = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_james_candy_bar_sales_l1452_145210


namespace NUMINAMATH_CALUDE_digit_150_is_5_l1452_145204

/-- The decimal expansion of 7/29 -/
def decimal_expansion : List Nat := [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

/-- The length of the repeating block in the decimal expansion of 7/29 -/
def repeat_length : Nat := decimal_expansion.length

/-- The 150th digit after the decimal point in the decimal expansion of 7/29 -/
def digit_150 : Nat := decimal_expansion[(150 - 1) % repeat_length]

theorem digit_150_is_5 : digit_150 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_5_l1452_145204


namespace NUMINAMATH_CALUDE_sally_peaches_l1452_145205

/-- The number of peaches Sally picked from the orchard -/
def picked_peaches : ℕ := 42

/-- The total number of peaches at the stand after picking -/
def total_peaches : ℕ := 55

/-- The number of peaches Sally had before picking more -/
def initial_peaches : ℕ := total_peaches - picked_peaches

theorem sally_peaches : initial_peaches = 13 := by
  sorry

end NUMINAMATH_CALUDE_sally_peaches_l1452_145205


namespace NUMINAMATH_CALUDE_max_intersections_sine_line_l1452_145200

theorem max_intersections_sine_line (φ : ℝ) : 
  ∃ (n : ℕ), n ≤ 4 ∧ 
  (∀ (m : ℕ), (∃ (S : Finset ℝ), S.card = m ∧ 
    (∀ x ∈ S, x ∈ Set.Icc 0 Real.pi ∧ 3 * Real.sin (3 * x + φ) = 2)) → m ≤ n) :=
sorry

end NUMINAMATH_CALUDE_max_intersections_sine_line_l1452_145200


namespace NUMINAMATH_CALUDE_votes_for_candidate_D_l1452_145209

def total_votes : ℕ := 1000000
def invalid_percentage : ℚ := 25 / 100
def candidate_A_percentage : ℚ := 45 / 100
def candidate_B_percentage : ℚ := 30 / 100
def candidate_C_percentage : ℚ := 20 / 100
def candidate_D_percentage : ℚ := 5 / 100

theorem votes_for_candidate_D :
  (total_votes : ℚ) * (1 - invalid_percentage) * candidate_D_percentage = 37500 := by
  sorry

end NUMINAMATH_CALUDE_votes_for_candidate_D_l1452_145209


namespace NUMINAMATH_CALUDE_summer_locations_l1452_145217

/-- Represents a location with temperature data --/
structure Location where
  temperatures : Finset ℕ
  median : ℕ
  mean : ℕ
  mode : Option ℕ
  variance : Option ℚ

/-- Checks if a location meets the summer criterion --/
def meetsSummerCriterion (loc : Location) : Prop :=
  loc.temperatures.card = 5 ∧ ∀ t ∈ loc.temperatures.toSet, t ≥ 22

/-- Location A --/
def locationA : Location := {
  temperatures := {},  -- We don't know the exact temperatures
  median := 24,
  mean := 0,  -- Not given
  mode := some 22,
  variance := none
}

/-- Location B --/
def locationB : Location := {
  temperatures := {},  -- We don't know the exact temperatures
  median := 27,
  mean := 24,
  mode := none,
  variance := none
}

/-- Location C --/
def locationC : Location := {
  temperatures := {32},  -- We only know one temperature
  median := 0,  -- Not given
  mean := 26,
  mode := none,
  variance := some (108/10)
}

theorem summer_locations :
  meetsSummerCriterion locationA ∧
  meetsSummerCriterion locationC ∧
  ¬ (meetsSummerCriterion locationB) :=
sorry

end NUMINAMATH_CALUDE_summer_locations_l1452_145217


namespace NUMINAMATH_CALUDE_spider_leg_count_l1452_145229

/-- The number of legs a single spider has -/
def spider_legs : ℕ := 8

/-- The number of spiders in the group -/
def group_size : ℕ := spider_legs / 2 + 10

/-- The total number of spider legs in the group -/
def total_legs : ℕ := group_size * spider_legs

theorem spider_leg_count : total_legs = 112 := by
  sorry

end NUMINAMATH_CALUDE_spider_leg_count_l1452_145229


namespace NUMINAMATH_CALUDE_optimal_usage_time_l1452_145248

/-- Profit function for the yacht (in ten thousand yuan) -/
def profit (x : ℕ+) : ℚ := -x^2 + 22*x - 49

/-- Average annual profit function -/
def avgProfit (x : ℕ+) : ℚ := profit x / x

/-- Theorem stating that 7 years maximizes the average annual profit -/
theorem optimal_usage_time :
  ∀ x : ℕ+, avgProfit 7 ≥ avgProfit x :=
sorry

end NUMINAMATH_CALUDE_optimal_usage_time_l1452_145248


namespace NUMINAMATH_CALUDE_curve_to_line_equation_l1452_145275

/-- Proves that the curve parameterized by (x,y) = (3t + 6, 5t - 8) 
    can be expressed as the line equation y = (5/3)x - 18 -/
theorem curve_to_line_equation : 
  ∀ (t x y : ℝ), x = 3*t + 6 ∧ y = 5*t - 8 → y = (5/3)*x - 18 := by
  sorry

end NUMINAMATH_CALUDE_curve_to_line_equation_l1452_145275


namespace NUMINAMATH_CALUDE_aziz_parents_move_year_l1452_145270

/-- The year Aziz's parents moved to America -/
def year_parents_moved (current_year : ℕ) (aziz_age : ℕ) (years_before_birth : ℕ) : ℕ :=
  current_year - aziz_age - years_before_birth

/-- Proof that Aziz's parents moved to America in 1982 -/
theorem aziz_parents_move_year :
  year_parents_moved 2021 36 3 = 1982 := by
  sorry

end NUMINAMATH_CALUDE_aziz_parents_move_year_l1452_145270


namespace NUMINAMATH_CALUDE_line_symmetry_l1452_145276

/-- Given a point (x, y) on a line, returns the x-coordinate of its symmetric point with respect to x = 1 -/
def symmetric_x (x : ℝ) : ℝ := 2 - x

/-- The original line -/
def original_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to x = 1 -/
theorem line_symmetry :
  ∀ x y : ℝ, original_line x y → symmetric_line (symmetric_x x) y :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l1452_145276


namespace NUMINAMATH_CALUDE_remainder_9876543210_mod_101_l1452_145237

theorem remainder_9876543210_mod_101 : 9876543210 % 101 = 31 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9876543210_mod_101_l1452_145237


namespace NUMINAMATH_CALUDE_inequality_1_solution_inequality_2_solution_l1452_145207

-- Define the solution sets
def solution_set_1 : Set ℝ := {x | -2 < x ∧ x < 1}
def solution_set_2 : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem for the first inequality
theorem inequality_1_solution (x : ℝ) : 
  |2*x + 1| < 3 ↔ x ∈ solution_set_1 :=
sorry

-- Theorem for the second inequality
theorem inequality_2_solution (x : ℝ) :
  |x - 2| + |x - 3| > 3 ↔ x ∈ solution_set_2 :=
sorry

end NUMINAMATH_CALUDE_inequality_1_solution_inequality_2_solution_l1452_145207


namespace NUMINAMATH_CALUDE_equation_solution_l1452_145294

theorem equation_solution : ∃ x : ℕ, (81^20 + 81^20 + 81^20 + 81^20 + 81^20 + 81^20 = 3^x) ∧ x = 81 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1452_145294


namespace NUMINAMATH_CALUDE_unique_base_solution_l1452_145277

-- Define a function to convert a number from base h to decimal
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldl (fun acc d => acc * h + d) 0

-- Define the equation in base h
def equation_holds (h : Nat) : Prop :=
  to_decimal [7, 3, 6, 4] h + to_decimal [8, 4, 2, 1] h = to_decimal [1, 7, 2, 8, 5] h

-- Theorem statement
theorem unique_base_solution :
  ∃! h : Nat, h > 1 ∧ equation_holds h :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l1452_145277


namespace NUMINAMATH_CALUDE_teacher_distribution_l1452_145274

/-- The number of ways to distribute teachers to schools -/
def distribute_teachers (total_teachers : ℕ) (female_teachers : ℕ) (schools : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute teachers to schools with constraints -/
def distribute_teachers_constrained (total_teachers : ℕ) (female_teachers : ℕ) (schools : ℕ) : ℕ :=
  sorry

theorem teacher_distribution :
  distribute_teachers 4 2 3 = 36 ∧
  distribute_teachers_constrained 4 2 3 = 30 :=
sorry

end NUMINAMATH_CALUDE_teacher_distribution_l1452_145274


namespace NUMINAMATH_CALUDE_gcd_1549_1023_l1452_145298

theorem gcd_1549_1023 : Nat.gcd 1549 1023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1549_1023_l1452_145298


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l1452_145251

-- Define the types for line and plane
variable (m : Line) (α : Plane)

-- Define the property of having no common points
def noCommonPoints (l : Line) (p : Plane) : Prop := sorry

-- Define the property of being parallel
def isParallel (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem line_parallel_to_plane (h : noCommonPoints m α) : isParallel m α := by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l1452_145251


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1452_145255

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt (x^2 + (y + 2)^2) = 10) ↔
  (y^2 / 25 + x^2 / 21 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1452_145255


namespace NUMINAMATH_CALUDE_p_is_true_l1452_145243

theorem p_is_true (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p :=
by sorry

end NUMINAMATH_CALUDE_p_is_true_l1452_145243


namespace NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l1452_145219

theorem sum_reciprocals_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1) :
  4 ≤ (1/a + 1/b) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀^2 + b₀^2 = 1 ∧ 1/a₀ + 1/b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l1452_145219


namespace NUMINAMATH_CALUDE_bacterial_eradication_l1452_145283

/-- Represents the state of the bacterial culture at a given minute -/
structure BacterialState where
  minute : ℕ
  infected : ℕ
  nonInfected : ℕ

/-- Models the evolution of the bacterial culture over time -/
def evolve (n : ℕ) : ℕ → BacterialState
  | 0 => ⟨0, 1, n - 1⟩
  | t + 1 => 
    let prev := evolve n t
    ⟨t + 1, 2 * prev.infected, (prev.nonInfected * 2) - (2 * prev.infected)⟩

/-- Theorem stating that the bacterial culture will be eradicated in n minutes -/
theorem bacterial_eradication (n : ℕ) (h : n > 0) : 
  (evolve n (n - 1)).nonInfected = 0 ∧ (evolve n n).infected = 0 := by
  sorry


end NUMINAMATH_CALUDE_bacterial_eradication_l1452_145283


namespace NUMINAMATH_CALUDE_complex_equation_implies_fourth_quadrant_l1452_145244

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_equation_implies_fourth_quadrant (z : ℂ) :
  (z + 3 * Complex.I) * (3 + Complex.I) = 7 - Complex.I →
  is_in_fourth_quadrant z := by
sorry

end NUMINAMATH_CALUDE_complex_equation_implies_fourth_quadrant_l1452_145244


namespace NUMINAMATH_CALUDE_marsha_pay_per_mile_l1452_145208

/-- Calculates the pay per mile for a delivery driver given their daily pay and distances driven --/
def pay_per_mile (daily_pay : ℚ) (first_distance second_distance : ℚ) : ℚ :=
  let third_distance := second_distance / 2
  let total_distance := first_distance + second_distance + third_distance
  daily_pay / total_distance

/-- Proves that Marsha's pay per mile is $2 given the specified conditions --/
theorem marsha_pay_per_mile :
  pay_per_mile 104 10 28 = 2 := by
  sorry

end NUMINAMATH_CALUDE_marsha_pay_per_mile_l1452_145208


namespace NUMINAMATH_CALUDE_solve_linear_system_l1452_145296

theorem solve_linear_system (a b : ℝ) 
  (eq1 : 3 * a + 2 * b = 5) 
  (eq2 : 2 * a + 3 * b = 4) : 
  a - b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_system_l1452_145296


namespace NUMINAMATH_CALUDE_selected_students_in_range_l1452_145250

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  totalPopulation : ℕ
  sampleSize : ℕ
  initialSelection : ℕ
  rangeStart : ℕ
  rangeEnd : ℕ

/-- Calculates the number of selected items in a given range for a systematic sample -/
def selectedInRange (s : SystematicSample) : ℕ :=
  sorry

/-- Theorem stating the number of selected students in the given range -/
theorem selected_students_in_range :
  let s : SystematicSample := {
    totalPopulation := 100,
    sampleSize := 25,
    initialSelection := 4,
    rangeStart := 46,
    rangeEnd := 78
  }
  selectedInRange s = 8 := by sorry

end NUMINAMATH_CALUDE_selected_students_in_range_l1452_145250


namespace NUMINAMATH_CALUDE_stacy_heather_walk_l1452_145249

/-- The problem of determining the time difference between Stacy and Heather's start times -/
theorem stacy_heather_walk (total_distance : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) 
  (heather_distance : ℝ) :
  total_distance = 25 →
  heather_speed = 5 →
  stacy_speed = heather_speed + 1 →
  heather_distance = 10.272727272727273 →
  ∃ (time_diff : ℝ), 
    time_diff * 60 = 24 ∧ 
    heather_distance / heather_speed = 
      (total_distance - heather_distance) / stacy_speed - time_diff :=
by sorry


end NUMINAMATH_CALUDE_stacy_heather_walk_l1452_145249


namespace NUMINAMATH_CALUDE_james_works_six_hours_l1452_145264

-- Define the cleaning times and number of rooms
def num_bedrooms : ℕ := 3
def num_bathrooms : ℕ := 2
def bedroom_cleaning_time : ℕ := 20 -- in minutes

-- Define the relationships between cleaning times
def living_room_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time
def bathroom_cleaning_time : ℕ := 2 * living_room_cleaning_time
def house_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time + living_room_cleaning_time + num_bathrooms * bathroom_cleaning_time
def outside_cleaning_time : ℕ := 2 * house_cleaning_time
def total_cleaning_time : ℕ := house_cleaning_time + outside_cleaning_time
def num_siblings : ℕ := 3

-- Define James' working time
def james_working_time : ℚ := (total_cleaning_time / num_siblings) / 60

-- Theorem statement
theorem james_works_six_hours : james_working_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_works_six_hours_l1452_145264


namespace NUMINAMATH_CALUDE_charity_ticket_sales_l1452_145227

theorem charity_ticket_sales (total_tickets : ℕ) (total_revenue : ℕ) 
  (h : total_tickets = 180 ∧ total_revenue = 2800) :
  ∃ (full_price : ℕ) (half_price_count : ℕ),
    full_price > 0 ∧
    half_price_count + (total_tickets - half_price_count) = total_tickets ∧
    half_price_count * (full_price / 2) + (total_tickets - half_price_count) * full_price = total_revenue ∧
    half_price_count = 328 := by
  sorry

end NUMINAMATH_CALUDE_charity_ticket_sales_l1452_145227


namespace NUMINAMATH_CALUDE_vector_triangle_l1452_145218

/-- Given vectors a and b, if 4a, 3b - 2a, and c form a triangle, then c = (4, -6) -/
theorem vector_triangle (a b c : ℝ × ℝ) : 
  a = (1, -3) → 
  b = (-2, 4) → 
  4 • a + (3 • b - 2 • a) + c = (0, 0) → 
  c = (4, -6) := by
sorry

end NUMINAMATH_CALUDE_vector_triangle_l1452_145218


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l1452_145240

theorem equation_represents_two_lines :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    x^2 - 50*y^2 - 16*x + 64 = 0 ↔ (x = a*y + b ∨ x = -a*y + b) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l1452_145240


namespace NUMINAMATH_CALUDE_sum_not_prime_l1452_145242

theorem sum_not_prime (a b c d : ℕ+) (h : a * b = c * d) : 
  ¬ Nat.Prime (a.val + b.val + c.val + d.val) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_prime_l1452_145242


namespace NUMINAMATH_CALUDE_volume_difference_l1452_145259

/-- The volume of space inside a sphere and outside a combined cylinder and cone -/
theorem volume_difference (r_sphere : ℝ) (r_base : ℝ) (h_cylinder : ℝ) (h_cone : ℝ) 
  (hr_sphere : r_sphere = 6)
  (hr_base : r_base = 4)
  (hh_cylinder : h_cylinder = 10)
  (hh_cone : h_cone = 5) :
  (4 / 3 * π * r_sphere^3) - (π * r_base^2 * h_cylinder + 1 / 3 * π * r_base^2 * h_cone) = 304 / 3 * π :=
sorry

end NUMINAMATH_CALUDE_volume_difference_l1452_145259


namespace NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l1452_145284

theorem sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds : 
  7 < Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) ∧ 
  Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) < 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l1452_145284


namespace NUMINAMATH_CALUDE_cube_sum_problem_l1452_145224

theorem cube_sum_problem (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 + b^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l1452_145224


namespace NUMINAMATH_CALUDE_provisions_problem_l1452_145245

/-- The number of days the provisions last for the initial group -/
def initial_days : ℕ := 20

/-- The number of additional men that join the group -/
def additional_men : ℕ := 200

/-- The number of days the provisions last after the additional men join -/
def final_days : ℕ := 16

/-- The initial number of men in the group -/
def initial_men : ℕ := 800

theorem provisions_problem :
  initial_men * initial_days = (initial_men + additional_men) * final_days :=
by sorry

end NUMINAMATH_CALUDE_provisions_problem_l1452_145245


namespace NUMINAMATH_CALUDE_derivative_f_at_3_l1452_145256

def f (x : ℝ) := x^2

theorem derivative_f_at_3 : 
  deriv f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_3_l1452_145256


namespace NUMINAMATH_CALUDE_function_value_at_specific_point_l1452_145288

/-- Given a function f(x) = ax^3 + b*sin(x) + 4 where a and b are real numbers,
    and f(lg(log_2(10))) = 5, prove that f(lg(lg(2))) = 3 -/
theorem function_value_at_specific_point
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^3 + b * Real.sin x + 4)
  (h2 : f (Real.log 10 / Real.log 2) = 5) :
  f (Real.log (Real.log 2) / Real.log 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_specific_point_l1452_145288


namespace NUMINAMATH_CALUDE_barney_towel_problem_l1452_145293

/-- The number of days without clean towels for Barney -/
def days_without_clean_towels (total_towels : ℕ) (towels_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  let towels_used_in_missed_week := towels_per_day * days_in_week
  let remaining_towels := total_towels - towels_used_in_missed_week
  let days_with_clean_towels := remaining_towels / towels_per_day
  days_in_week - days_with_clean_towels

/-- Theorem stating that Barney will not have clean towels for 5 days in the following week -/
theorem barney_towel_problem :
  days_without_clean_towels 18 2 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_barney_towel_problem_l1452_145293


namespace NUMINAMATH_CALUDE_exists_nonparallel_quadrilateral_from_identical_triangles_l1452_145202

/-- A triangle in 2D space --/
structure Triangle :=
  (a b c : ℝ × ℝ)

/-- A quadrilateral in 2D space --/
structure Quadrilateral :=
  (a b c d : ℝ × ℝ)

/-- Check if two line segments are parallel --/
def are_parallel (p1 p2 q1 q2 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := q1
  let (x4, y4) := q2
  (x2 - x1) * (y4 - y3) = (y2 - y1) * (x4 - x3)

/-- Check if a quadrilateral has parallel sides --/
def has_parallel_sides (q : Quadrilateral) : Prop :=
  are_parallel q.a q.b q.c q.d ∨ are_parallel q.a q.d q.b q.c

/-- Check if a quadrilateral is convex --/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Function to construct a quadrilateral from four triangles --/
def construct_quadrilateral (t1 t2 t3 t4 : Triangle) : Quadrilateral := sorry

/-- Theorem: There exists a convex quadrilateral formed by four identical triangles that does not have parallel sides --/
theorem exists_nonparallel_quadrilateral_from_identical_triangles :
  ∃ (t : Triangle) (q : Quadrilateral),
    q = construct_quadrilateral t t t t ∧
    is_convex q ∧
    ¬has_parallel_sides q :=
sorry

end NUMINAMATH_CALUDE_exists_nonparallel_quadrilateral_from_identical_triangles_l1452_145202


namespace NUMINAMATH_CALUDE_michaels_brothers_ages_l1452_145241

theorem michaels_brothers_ages (michael_age : ℕ) (older_brother_age : ℕ) (younger_brother_age : ℕ) :
  older_brother_age = 2 * (michael_age - 1) + 1 →
  younger_brother_age = older_brother_age / 3 →
  michael_age + older_brother_age + younger_brother_age = 28 →
  younger_brother_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_michaels_brothers_ages_l1452_145241


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_eightyfive_l1452_145295

theorem largest_multiple_of_seven_less_than_negative_eightyfive :
  ∀ n : ℤ, n * 7 < -85 → n * 7 ≤ -91 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_eightyfive_l1452_145295


namespace NUMINAMATH_CALUDE_friendly_pair_solution_l1452_145278

/-- Definition of a friendly number pair -/
def is_friendly_pair (m n : ℚ) : Prop :=
  m / 2 + n / 4 = (m + n) / (2 + 4)

/-- Theorem: If (a, 3) is a friendly number pair, then a = -3/4 -/
theorem friendly_pair_solution (a : ℚ) :
  is_friendly_pair a 3 → a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_friendly_pair_solution_l1452_145278


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l1452_145271

/-- The area of the region between two concentric circles -/
theorem area_between_concentric_circles
  (r : ℝ) -- radius of the inner circle
  (h : r > 0) -- assumption that r is positive
  (width : ℝ) -- width of the region between circles
  (h_width : width = 3 * r - r) -- definition of width
  (h_width_value : width = 4) -- given width value
  : (π * (3 * r)^2 - π * r^2) = 8 * π * r^2 := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l1452_145271


namespace NUMINAMATH_CALUDE_quadratic_condition_for_x_greater_than_two_l1452_145230

theorem quadratic_condition_for_x_greater_than_two :
  (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_for_x_greater_than_two_l1452_145230


namespace NUMINAMATH_CALUDE_power_three_mod_eight_l1452_145225

theorem power_three_mod_eight : 3^20 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_eight_l1452_145225


namespace NUMINAMATH_CALUDE_volume_change_specific_l1452_145222

/-- Represents the change in volume of a rectangular parallelepiped -/
def volume_change (a b c : ℝ) (da db dc : ℝ) : ℝ :=
  b * c * da + a * c * db + a * b * dc

/-- Theorem stating the change in volume for specific dimensions and changes -/
theorem volume_change_specific :
  let a : ℝ := 8
  let b : ℝ := 6
  let c : ℝ := 3
  let da : ℝ := 0.1
  let db : ℝ := 0.05
  let dc : ℝ := -0.15
  volume_change a b c da db dc = -4.2 := by
  sorry

#eval volume_change 8 6 3 0.1 0.05 (-0.15)

end NUMINAMATH_CALUDE_volume_change_specific_l1452_145222


namespace NUMINAMATH_CALUDE_pattern_boundary_length_l1452_145238

theorem pattern_boundary_length (square_area : ℝ) (num_points : ℕ) : square_area = 144 ∧ num_points = 4 →
  ∃ (boundary_length : ℝ), boundary_length = 18 * Real.pi + 36 := by
  sorry

end NUMINAMATH_CALUDE_pattern_boundary_length_l1452_145238


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l1452_145223

theorem termite_ridden_not_collapsing 
  (total_homes : ℕ) 
  (termite_ridden : ℕ) 
  (collapsing : ℕ) 
  (h1 : termite_ridden = (5 * total_homes) / 8)
  (h2 : collapsing = (11 * termite_ridden) / 16) :
  (termite_ridden - collapsing) = (25 * total_homes) / 128 :=
by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l1452_145223


namespace NUMINAMATH_CALUDE_sleeping_bag_wholesale_cost_l1452_145287

theorem sleeping_bag_wholesale_cost :
  ∀ (wholesale_cost selling_price : ℝ),
    selling_price = wholesale_cost * 1.12 →
    selling_price = 28 →
    wholesale_cost = 25 := by sorry

end NUMINAMATH_CALUDE_sleeping_bag_wholesale_cost_l1452_145287


namespace NUMINAMATH_CALUDE_fraction_equality_l1452_145269

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (h : (a + 2*b) / a = 4) : 
  a / (b - a) = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1452_145269


namespace NUMINAMATH_CALUDE_function_property_l1452_145216

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b) :
  is_even_function f :=
sorry

end NUMINAMATH_CALUDE_function_property_l1452_145216


namespace NUMINAMATH_CALUDE_line_equation_general_form_l1452_145212

/-- A line passing through a point with a given direction vector -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The general form of a line equation: ax + by + c = 0 -/
structure GeneralForm where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line passing through (5,4) with direction vector (1,2),
    its general form equation is 2x - y - 6 = 0 -/
theorem line_equation_general_form (l : Line) 
    (h1 : l.point = (5, 4)) 
    (h2 : l.direction = (1, 2)) : 
    ∃ (gf : GeneralForm), gf.a = 2 ∧ gf.b = -1 ∧ gf.c = -6 :=
sorry

end NUMINAMATH_CALUDE_line_equation_general_form_l1452_145212


namespace NUMINAMATH_CALUDE_mom_tshirt_count_l1452_145257

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- The total number of t-shirts Mom will have -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_tshirt_count : total_shirts = 426 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_count_l1452_145257


namespace NUMINAMATH_CALUDE_polynomial_problems_l1452_145247

theorem polynomial_problems :
  (∀ x y, ∃ k, (2 - b) * x^2 + (a + 3) * x + (-6) * y + 7 = k) →
  (a - b)^2 = 25 ∧
  (∀ x y, ∃ k, (-1 - n) * x^2 + (-m + 6) * x + (-18) * y + 5 = k) →
  n = -1 ∧ m = 6 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_problems_l1452_145247


namespace NUMINAMATH_CALUDE_circle_area_increase_l1452_145253

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1452_145253


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1452_145267

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (x, 1)
  parallel a b → x = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1452_145267


namespace NUMINAMATH_CALUDE_f_properties_l1452_145261

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2))^2 + Real.sqrt 3 * Real.sin x

theorem f_properties :
  (∃ (M : ℝ), ∀ x, f x ≤ M ∧ (∃ x, f x = M) ∧ M = 3) ∧
  (∀ k : ℤ, f (Real.pi / 3 + 2 * k * Real.pi) = 3) ∧
  (∀ α : ℝ, Real.tan (α / 2) = 1 / 2 → f α = (8 + 4 * Real.sqrt 3) / 5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1452_145261


namespace NUMINAMATH_CALUDE_max_term_binomial_expansion_max_term_value_max_term_specific_case_l1452_145289

theorem max_term_binomial_expansion (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ k : ℕ, k ≤ n ∧
  ∀ j : ℕ, j ≤ n →
    (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
by sorry

theorem max_term_value (n : ℕ) (x : ℝ) (h : x > 0) :
  let k := ⌊n * x / (1 + x)⌋ + 1
  ∃ m : ℕ, m ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose m : ℝ) * x^m ≥ (n.choose j : ℝ) * x^j ∧
    (m = k ∨ m = k - 1) :=
by sorry

theorem max_term_specific_case :
  let n : ℕ := 210
  let x : ℝ := Real.sqrt 13
  let k : ℕ := 165
  ∃ m : ℕ, m ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose m : ℝ) * x^m ≥ (n.choose j : ℝ) * x^j ∧
    m = k :=
by sorry

end NUMINAMATH_CALUDE_max_term_binomial_expansion_max_term_value_max_term_specific_case_l1452_145289


namespace NUMINAMATH_CALUDE_inverse_proportion_l1452_145285

/-- Given that x is inversely proportional to y, prove that when x = 5 for y = -4, 
    then x = 2 for y = -10 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) 
    (h1 : x * y = k)  -- x is inversely proportional to y
    (h2 : 5 * (-4) = k)  -- x = 5 when y = -4
    : x = 2 ∧ y = -10 → x * y = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l1452_145285


namespace NUMINAMATH_CALUDE_salary_problem_l1452_145280

/-- Salary problem -/
theorem salary_problem 
  (jan feb mar apr may : ℕ)  -- Salaries for each month
  (h1 : (jan + feb + mar + apr) / 4 = 8000)  -- Average for Jan-Apr
  (h2 : (feb + mar + apr + may) / 4 = 8800)  -- Average for Feb-May
  (h3 : may = 6500)  -- May's salary
  : jan = 3300 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l1452_145280


namespace NUMINAMATH_CALUDE_investment_ratio_is_two_thirds_l1452_145220

/-- Represents the investments and profit shares of three partners A, B, and C. -/
structure Partnership where
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  b_profit_share : ℝ

/-- Theorem stating that under given conditions, the ratio of B's investment to C's investment is 2:3 -/
theorem investment_ratio_is_two_thirds (p : Partnership) 
  (h1 : p.b_investment > 0)
  (h2 : p.c_investment > 0)
  (h3 : p.total_profit = 4400)
  (h4 : p.b_profit_share = 800)
  (h5 : 3 * p.b_investment = p.b_investment + p.c_investment) :
  p.b_investment / p.c_investment = 2 / 3 := by
  sorry

#check investment_ratio_is_two_thirds

end NUMINAMATH_CALUDE_investment_ratio_is_two_thirds_l1452_145220


namespace NUMINAMATH_CALUDE_unique_base_representation_l1452_145268

/-- The fraction we're considering -/
def fraction : ℚ := 8 / 65

/-- The repeating digits in the base-k representation -/
def repeating_digits : List ℕ := [2, 4]

/-- 
Given a positive integer k, this function should return true if and only if
the base-k representation of the fraction is 0.24242424...
-/
def is_correct_representation (k : ℕ) : Prop :=
  k > 0 ∧ 
  fraction = (2 / k + 4 / k^2) / (1 - 1 / k^2)

/-- The theorem to be proved -/
theorem unique_base_representation : 
  ∃! k : ℕ, is_correct_representation k ∧ k = 18 :=
sorry

end NUMINAMATH_CALUDE_unique_base_representation_l1452_145268


namespace NUMINAMATH_CALUDE_trigonometric_simplification_max_value_cosine_function_l1452_145236

open Real

theorem trigonometric_simplification (α : ℝ) :
  (sin (2 * π - α) * tan (π - α) * cos (-π + α)) / (sin (5 * π + α) * sin (π / 2 + α)) = tan α :=
sorry

theorem max_value_cosine_function :
  let f : ℝ → ℝ := λ x ↦ 2 * cos x - cos (2 * x)
  ∃ (max_value : ℝ), max_value = 3 / 2 ∧
    ∀ x, f x ≤ max_value ∧
    ∀ k : ℤ, f (π / 3 + 2 * π * ↑k) = max_value ∧ f (-π / 3 + 2 * π * ↑k) = max_value :=
sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_max_value_cosine_function_l1452_145236


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l1452_145282

theorem largest_n_binomial_equality : ∃ (n : ℕ), (
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧
  (∀ m : ℕ, Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n)
) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l1452_145282


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_l1452_145214

theorem sin_50_plus_sqrt3_tan_10 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_l1452_145214


namespace NUMINAMATH_CALUDE_model1_best_fit_l1452_145258

-- Define the coefficient of determination for each model
def R2_model1 : ℝ := 0.98
def R2_model2 : ℝ := 0.80
def R2_model3 : ℝ := 0.50
def R2_model4 : ℝ := 0.25

-- Define a function to compare R² values
def better_fit (a b : ℝ) : Prop := a > b

-- Theorem stating that Model 1 has the best fitting effect
theorem model1_best_fit :
  better_fit R2_model1 R2_model2 ∧
  better_fit R2_model1 R2_model3 ∧
  better_fit R2_model1 R2_model4 :=
by sorry

end NUMINAMATH_CALUDE_model1_best_fit_l1452_145258


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1452_145234

/-- Given two vectors a and b in ℝ³, where a is parallel to b, prove that m + n = 4 -/
theorem parallel_vectors_sum (a b : ℝ × ℝ × ℝ) (m n : ℝ) : 
  a = (2, -1, 3) → b = (4, m, n) → (∃ (k : ℝ), a = k • b) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1452_145234


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l1452_145221

/-- Given that Marilyn starts with 51 bottle caps and shares 36 with Nancy, 
    prove that she ends up with 15 bottle caps. -/
theorem marilyn_bottle_caps 
  (start : ℕ) 
  (shared : ℕ) 
  (h1 : start = 51) 
  (h2 : shared = 36) : 
  start - shared = 15 := by
sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l1452_145221


namespace NUMINAMATH_CALUDE_cosine_vertical_shift_l1452_145262

theorem cosine_vertical_shift 
  (a b c d : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (hmax : d + a = 7) 
  (hmin : d - a = 1) : 
  d = 4 := by
sorry

end NUMINAMATH_CALUDE_cosine_vertical_shift_l1452_145262


namespace NUMINAMATH_CALUDE_work_completion_time_l1452_145273

theorem work_completion_time 
  (work_rate_b : ℝ) 
  (work_rate_combined : ℝ) 
  (days_b : ℝ) 
  (days_combined : ℝ) :
  work_rate_b = 1 / days_b →
  work_rate_combined = 1 / days_combined →
  days_b = 6 →
  days_combined = 3.75 →
  work_rate_combined = work_rate_b + 1 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1452_145273


namespace NUMINAMATH_CALUDE_tim_soda_cans_l1452_145233

/-- The number of soda cans Tim has at the end of the scenario -/
def final_soda_cans (initial : ℕ) (taken : ℕ) : ℕ :=
  let remaining := initial - taken
  remaining + remaining / 2

/-- Theorem stating that Tim ends up with 24 cans of soda -/
theorem tim_soda_cans : final_soda_cans 22 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_tim_soda_cans_l1452_145233


namespace NUMINAMATH_CALUDE_reverse_digit_numbers_base_9_11_l1452_145203

def is_three_digit_base (n : ℕ) (base : ℕ) : Prop :=
  base ^ 2 ≤ n ∧ n < base ^ 3

def digits_base (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

def reverse_digits (l : List ℕ) : List ℕ :=
  sorry

theorem reverse_digit_numbers_base_9_11 :
  ∃! (S : Finset ℕ),
    (∀ n ∈ S,
      is_three_digit_base n 9 ∧
      is_three_digit_base n 11 ∧
      digits_base n 9 = reverse_digits (digits_base n 11)) ∧
    S.card = 2 ∧
    245 ∈ S ∧
    490 ∈ S :=
  sorry

end NUMINAMATH_CALUDE_reverse_digit_numbers_base_9_11_l1452_145203


namespace NUMINAMATH_CALUDE_test_mean_score_l1452_145263

theorem test_mean_score (mean : ℝ) (std_dev : ℝ) (lowest_score : ℝ) : 
  std_dev = 10 →
  lowest_score = mean - 2 * std_dev →
  lowest_score = 20 →
  mean = 40 := by
sorry

end NUMINAMATH_CALUDE_test_mean_score_l1452_145263


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1452_145292

def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1452_145292


namespace NUMINAMATH_CALUDE_grandfathers_age_l1452_145226

/-- Given the conditions about a family's ages, prove the grandfather's age 5 years ago. -/
theorem grandfathers_age (father_age : ℕ) (h1 : father_age = 58) :
  ∃ (son_age grandfather_age : ℕ),
    father_age - son_age = son_age ∧ 
    (son_age - 5) * 2 = grandfather_age ∧
    grandfather_age = 48 :=
by sorry

end NUMINAMATH_CALUDE_grandfathers_age_l1452_145226


namespace NUMINAMATH_CALUDE_lucas_class_size_l1452_145254

theorem lucas_class_size (n : ℕ) (best_rank : ℕ) (worst_rank : ℕ)
  (h1 : best_rank = 30)
  (h2 : worst_rank = 45)
  (h3 : n = best_rank + worst_rank - 1) :
  n = 74 := by
sorry

end NUMINAMATH_CALUDE_lucas_class_size_l1452_145254


namespace NUMINAMATH_CALUDE_investment_result_approx_17607_l1452_145290

/-- Calculates the final amount of an investment after tax and compound interest --/
def investment_after_tax (initial_investment : ℝ) (interest_rate : ℝ) (tax_rate : ℝ) (years : ℕ) : ℝ :=
  let compound_factor := 1 + interest_rate * (1 - tax_rate)
  initial_investment * compound_factor ^ years

/-- Theorem stating that the investment result is approximately $17,607 --/
theorem investment_result_approx_17607 :
  ∃ ε > 0, |investment_after_tax 15000 0.05 0.10 4 - 17607| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_result_approx_17607_l1452_145290


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1452_145228

/-- The coefficient of x^2 in the expansion of (1-ax)(1+x)^5 -/
def coefficient_x_squared (a : ℝ) : ℝ := 10 - 5*a

theorem expansion_coefficient (a : ℝ) : coefficient_x_squared a = 5 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1452_145228


namespace NUMINAMATH_CALUDE_lewis_savings_l1452_145297

/-- Lewis's savings calculation -/
theorem lewis_savings (weekly_earnings weekly_rent harvest_weeks : ℕ) 
  (h1 : weekly_earnings = 491)
  (h2 : weekly_rent = 216)
  (h3 : harvest_weeks = 1181) : 
  (weekly_earnings - weekly_rent) * harvest_weeks = 324775 := by
  sorry

#eval (491 - 216) * 1181  -- To verify the result

end NUMINAMATH_CALUDE_lewis_savings_l1452_145297


namespace NUMINAMATH_CALUDE_xy_value_l1452_145235

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^(Real.sqrt y) = 27) (h2 : (Real.sqrt x)^y = 9) : 
  x * y = 16 * Real.rpow 3 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1452_145235


namespace NUMINAMATH_CALUDE_origin_constructible_l1452_145232

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the condition that A is above and to the left of B
def A_above_left_of_B : Prop :=
  A.1 < B.1 ∧ A.2 > B.2

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Theorem stating that the origin can be constructed
theorem origin_constructible (h : A_above_left_of_B) :
  ∃ (construction : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ)), construction A B = O :=
sorry

end NUMINAMATH_CALUDE_origin_constructible_l1452_145232


namespace NUMINAMATH_CALUDE_small_triangles_in_large_triangle_l1452_145239

theorem small_triangles_in_large_triangle :
  let large_side : ℝ := 15
  let small_side : ℝ := 3
  let area (side : ℝ) := (Real.sqrt 3 / 4) * side^2
  let num_small_triangles := (area large_side) / (area small_side)
  num_small_triangles = 25 := by sorry

end NUMINAMATH_CALUDE_small_triangles_in_large_triangle_l1452_145239


namespace NUMINAMATH_CALUDE_magnitude_of_complex_product_l1452_145279

theorem magnitude_of_complex_product : 
  Complex.abs ((7 - 4*I) * (3 + 10*I)) = Real.sqrt 7085 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_product_l1452_145279


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l1452_145265

/-- The selling price of two discounted items -/
def discounted_price (a : ℝ) : ℝ :=
  let original_price := a
  let markup_percentage := 0.5
  let discount_percentage := 0.2
  let marked_up_price := original_price * (1 + markup_percentage)
  let discounted_price := marked_up_price * (1 - discount_percentage)
  2 * discounted_price

/-- Theorem stating that the discounted price of two items is 2.4 times the original price -/
theorem discounted_price_theorem (a : ℝ) : discounted_price a = 2.4 * a := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_theorem_l1452_145265


namespace NUMINAMATH_CALUDE_probability_theorem_l1452_145201

def shirts : ℕ := 6
def pants : ℕ := 7
def socks : ℕ := 8
def total_articles : ℕ := shirts + pants + socks
def selected_articles : ℕ := 4

def probability_two_shirts_one_pant_one_sock : ℚ :=
  (Nat.choose shirts 2 * Nat.choose pants 1 * Nat.choose socks 1) /
  Nat.choose total_articles selected_articles

theorem probability_theorem :
  probability_two_shirts_one_pant_one_sock = 40 / 285 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1452_145201


namespace NUMINAMATH_CALUDE_impossible_triangle_angles_l1452_145281

-- Define a triangle
structure Triangle where
  -- We don't need to specify the actual properties of a triangle here

-- Define the sum of interior angles of a triangle
def sum_of_interior_angles (t : Triangle) : ℝ := 180

-- Theorem: It is impossible for the sum of the interior angles of a triangle to be 360°
theorem impossible_triangle_angles (t : Triangle) : sum_of_interior_angles t ≠ 360 := by
  sorry

end NUMINAMATH_CALUDE_impossible_triangle_angles_l1452_145281


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1452_145252

theorem fraction_to_decimal : (45 : ℚ) / (2^2 * 5^3) = (9 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1452_145252


namespace NUMINAMATH_CALUDE_circle_area_ratio_concentric_circles_area_ratio_l1452_145291

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > r₁ → 
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = (r₂^2 / r₁^2) - 1 :=
by sorry

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_concentric_circles_area_ratio_l1452_145291


namespace NUMINAMATH_CALUDE_inequality_proof_l1452_145215

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) :
  (a + b + c) / 4 ≥ (Real.sqrt (a * b - 1)) / (b + c) + 
                    (Real.sqrt (b * c - 1)) / (c + a) + 
                    (Real.sqrt (c * a - 1)) / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1452_145215


namespace NUMINAMATH_CALUDE_cost_equalization_l1452_145206

theorem cost_equalization (X Y Z : ℝ) (h : X < Y ∧ Y < Z) :
  let E := (X + Y + Z) / 3
  let payment_to_bernardo := E - X - (Z - Y) / 2
  let payment_to_carlos := (Z - Y) / 2
  (X + payment_to_bernardo + payment_to_carlos = E) ∧
  (Y - payment_to_bernardo = E) ∧
  (Z - payment_to_carlos = E) := by
sorry

end NUMINAMATH_CALUDE_cost_equalization_l1452_145206


namespace NUMINAMATH_CALUDE_remaining_note_denomination_l1452_145272

theorem remaining_note_denomination 
  (total_amount : ℕ) 
  (total_notes : ℕ) 
  (fifty_notes : ℕ) 
  (h1 : total_amount = 10350)
  (h2 : total_notes = 108)
  (h3 : fifty_notes = 97) :
  (total_amount - 50 * fifty_notes) / (total_notes - fifty_notes) = 500 := by
sorry


end NUMINAMATH_CALUDE_remaining_note_denomination_l1452_145272


namespace NUMINAMATH_CALUDE_choir_dance_team_equation_l1452_145299

theorem choir_dance_team_equation (x : ℤ) : 
  (46 + x = 3 * (30 - x)) ↔ 
  (∃ (initial_choir initial_dance final_choir final_dance : ℤ),
    initial_choir = 46 ∧ 
    initial_dance = 30 ∧ 
    final_choir = initial_choir + x ∧ 
    final_dance = initial_dance - x ∧ 
    final_choir = 3 * final_dance) :=
by sorry

end NUMINAMATH_CALUDE_choir_dance_team_equation_l1452_145299


namespace NUMINAMATH_CALUDE_smallest_inverse_domain_l1452_145286

-- Define the function g
def g (x : ℝ) : ℝ := (x + 1)^2 - 6

-- State the theorem
theorem smallest_inverse_domain (d : ℝ) :
  (∀ x y, x ∈ Set.Ici d → y ∈ Set.Ici d → g x = g y → x = y) ∧ 
  (∀ d' < d, ∃ x y, x ∈ Set.Ici d' → y ∈ Set.Ici d' → x ≠ y ∧ g x = g y) ↔ 
  d = -1 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_domain_l1452_145286


namespace NUMINAMATH_CALUDE_container_capacity_l1452_145246

theorem container_capacity : ∀ (C : ℝ),
  (C > 0) →  -- Ensure the capacity is positive
  (0.35 * C + 48 = 0.75 * C) →
  C = 120 := by
sorry

end NUMINAMATH_CALUDE_container_capacity_l1452_145246


namespace NUMINAMATH_CALUDE_sneeze_interval_l1452_145213

/-- Given a sneezing fit lasting 2 minutes with 40 sneezes in total, 
    prove that the time between each sneeze is 3 seconds. -/
theorem sneeze_interval (duration_minutes : ℕ) (total_sneezes : ℕ) 
  (h1 : duration_minutes = 2) 
  (h2 : total_sneezes = 40) : 
  (duration_minutes * 60) / total_sneezes = 3 := by
  sorry

end NUMINAMATH_CALUDE_sneeze_interval_l1452_145213


namespace NUMINAMATH_CALUDE_job_completion_time_l1452_145211

/-- The time it takes for Annie to complete the job alone -/
def annie_time : ℝ := 10

/-- The time the person works before Annie takes over -/
def person_work_time : ℝ := 3

/-- The time it takes Annie to complete the remaining work after the person stops -/
def annie_remaining_time : ℝ := 8

/-- The time it takes for the person to complete the job alone -/
def person_total_time : ℝ := 15

theorem job_completion_time :
  (person_work_time / person_total_time) + (annie_remaining_time / annie_time) = 1 :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l1452_145211


namespace NUMINAMATH_CALUDE_exists_valid_classification_l1452_145266

/-- Represents a team of students -/
structure Team :=
  (members : Finset Nat)
  (size_eq_six : members.card = 6)

/-- Classification of teams as GOOD or OK -/
def TeamClassification := Team → Bool

/-- Partition of students into teams -/
structure Partition :=
  (teams : Finset Team)
  (covers_all_students : (teams.biUnion Team.members).card = 24)
  (team_count_eq_four : teams.card = 4)

/-- Counts the number of GOOD teams in a partition -/
def countGoodTeams (c : TeamClassification) (p : Partition) : Nat :=
  (p.teams.filter (λ t => c t)).card

/-- Theorem stating the existence of a valid team classification -/
theorem exists_valid_classification : ∃ (c : TeamClassification),
  (∀ (p : Partition), countGoodTeams c p = 3 ∨ countGoodTeams c p = 1) ∧
  (∃ (p1 p2 : Partition), countGoodTeams c p1 = 3 ∧ countGoodTeams c p2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_classification_l1452_145266


namespace NUMINAMATH_CALUDE_tan_product_simplification_l1452_145260

theorem tan_product_simplification :
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_simplification_l1452_145260
