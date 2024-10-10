import Mathlib

namespace franks_reading_average_l2303_230370

/-- Calculates the average pages read per day given the pages and days for three books --/
def average_pages_per_day (pages1 pages2 pages3 : ℕ) (days1 days2 days3 : ℕ) : ℚ :=
  (pages1 + pages2 + pages3 : ℚ) / (days1 + days2 + days3)

/-- Theorem stating that the average pages per day for Frank's reading is as calculated --/
theorem franks_reading_average :
  average_pages_per_day 249 379 480 3 5 6 = 79.14 := by
  sorry

end franks_reading_average_l2303_230370


namespace travel_time_difference_proof_l2303_230387

/-- The length of Telegraph Road in kilometers -/
def telegraph_road_length : ℝ := 162

/-- The lengths of the four detours on Telegraph Road in kilometers -/
def telegraph_detours : List ℝ := [5.2, 2.7, 3.8, 4.4]

/-- The length of Pardee Road in meters -/
def pardee_road_length : ℝ := 12000

/-- The increase in length of Pardee Road due to road work in kilometers -/
def pardee_road_increase : ℝ := 2.5

/-- The constant speed of travel in kilometers per hour -/
def travel_speed : ℝ := 80

/-- The difference in travel time between Telegraph Road and Pardee Road in minutes -/
def travel_time_difference : ℝ := 122.7

theorem travel_time_difference_proof :
  let telegraph_total := telegraph_road_length + (telegraph_detours.sum)
  let pardee_total := (pardee_road_length / 1000) + pardee_road_increase
  let telegraph_time := (telegraph_total / travel_speed) * 60
  let pardee_time := (pardee_total / travel_speed) * 60
  telegraph_time - pardee_time = travel_time_difference := by
  sorry

end travel_time_difference_proof_l2303_230387


namespace geometric_sequence_general_term_l2303_230338

/-- Given a geometric sequence {a_n} where a_3 = 9 and a_6 = 243, 
    prove that the general term formula is a_n = 3^(n-1) -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a3 : a 3 = 9) 
  (h_a6 : a 6 = 243) : 
  ∀ n : ℕ, a n = 3^(n - 1) := by
  sorry

end geometric_sequence_general_term_l2303_230338


namespace determinant_minimization_l2303_230354

theorem determinant_minimization (a b : ℤ) : 
  let Δ := 36 * a - 81 * b
  ∃ (c : ℕ+), 
    (∀ k : ℕ+, Δ = k → c ≤ k) ∧ 
    Δ = c ∧
    c = 9 ∧
    (∀ a' b' : ℕ+, 36 * a' - 81 * b' = c → a + b ≤ a' + b') ∧
    a = 7 ∧ 
    b = 3 := by
  sorry

end determinant_minimization_l2303_230354


namespace parabola_comparison_l2303_230398

theorem parabola_comparison : ∀ x : ℝ, x^2 - x + 3 < x^2 - x + 5 := by
  sorry

end parabola_comparison_l2303_230398


namespace subtraction_in_base_8_l2303_230317

def base_8_to_decimal (n : ℕ) : ℕ := sorry

def decimal_to_base_8 (n : ℕ) : ℕ := sorry

theorem subtraction_in_base_8 :
  decimal_to_base_8 (base_8_to_decimal 2101 - base_8_to_decimal 1245) = 634 := by
  sorry

end subtraction_in_base_8_l2303_230317


namespace train_crossing_time_l2303_230314

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 133.33333333333334 →
  train_speed_kmh = 60 →
  crossing_time = 8 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end train_crossing_time_l2303_230314


namespace unique_number_l2303_230375

theorem unique_number : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end unique_number_l2303_230375


namespace checkerboard_coverable_iff_even_area_uncoverable_checkerboards_l2303_230341

/-- A checkerboard -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- The property of a checkerboard being completely coverable by dominoes -/
def is_coverable (board : Checkerboard) : Prop :=
  (board.rows * board.cols) % 2 = 0

/-- Theorem stating that a checkerboard is coverable iff its area is even -/
theorem checkerboard_coverable_iff_even_area (board : Checkerboard) :
  is_coverable board ↔ Even (board.rows * board.cols) :=
sorry

/-- Function to check if a checkerboard is coverable -/
def check_coverable (board : Checkerboard) : Bool :=
  (board.rows * board.cols) % 2 = 0

/-- Theorem stating which of the given checkerboards are not coverable -/
theorem uncoverable_checkerboards :
  let boards := [
    Checkerboard.mk 4 4,
    Checkerboard.mk 5 5,
    Checkerboard.mk 5 7,
    Checkerboard.mk 6 6,
    Checkerboard.mk 7 3
  ]
  (boards.filter (λ b => ¬check_coverable b)).map (λ b => (b.rows, b.cols)) =
    [(5, 5), (5, 7), (7, 3)] :=
sorry

end checkerboard_coverable_iff_even_area_uncoverable_checkerboards_l2303_230341


namespace time_spent_on_type_a_l2303_230316

/-- Represents the time allocation for an exam with three problem types. -/
structure ExamTime where
  totalTime : ℕ  -- Total exam time in minutes
  totalQuestions : ℕ  -- Total number of questions
  typeACount : ℕ  -- Number of Type A problems
  typeBCount : ℕ  -- Number of Type B problems
  typeCCount : ℕ  -- Number of Type C problems
  typeATime : ℚ  -- Time for one Type A problem
  typeBTime : ℚ  -- Time for one Type B problem
  typeCTime : ℚ  -- Time for one Type C problem

/-- Theorem stating the time spent on Type A problems in the given exam scenario. -/
theorem time_spent_on_type_a (exam : ExamTime) : 
  exam.totalTime = 240 ∧ 
  exam.totalQuestions = 300 ∧ 
  exam.typeACount = 25 ∧ 
  exam.typeBCount = 100 ∧ 
  exam.typeCCount = 175 ∧ 
  exam.typeATime = 4 * exam.typeBTime ∧ 
  exam.typeBTime = 2 * exam.typeCTime ∧ 
  exam.typeACount * exam.typeATime + exam.typeBCount * exam.typeBTime = exam.totalTime / 2 
  → exam.typeACount * exam.typeATime = 60 := by
  sorry

end time_spent_on_type_a_l2303_230316


namespace cos_alpha_value_l2303_230362

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3) 
  (h2 : Real.pi/2 ≤ α) 
  (h3 : α ≤ Real.pi) : 
  Real.cos α = -2 * Real.sqrt 2 / 3 := by
  sorry

end cos_alpha_value_l2303_230362


namespace max_area_rectangle_max_area_520_perimeter_l2303_230303

/-- The perimeter of the rectangle in meters -/
def perimeter : ℝ := 520

/-- Theorem: Maximum area of a rectangle with given perimeter -/
theorem max_area_rectangle (l w : ℝ) (h1 : l > 0) (h2 : w > 0) (h3 : 2 * l + 2 * w = perimeter) :
  l * w ≤ (perimeter / 4) ^ 2 :=
sorry

/-- Corollary: The maximum area of a rectangle with perimeter 520 meters is 16900 square meters -/
theorem max_area_520_perimeter :
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * l + 2 * w = perimeter ∧ l * w = 16900 :=
sorry

end max_area_rectangle_max_area_520_perimeter_l2303_230303


namespace x_negative_necessary_not_sufficient_l2303_230330

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- Theorem statement
theorem x_negative_necessary_not_sufficient :
  (∀ x : ℝ, ln (x + 1) < 0 → x < 0) ∧
  ¬(∀ x : ℝ, x < 0 → ln (x + 1) < 0) :=
by
  sorry


end x_negative_necessary_not_sufficient_l2303_230330


namespace complex_on_imaginary_axis_l2303_230352

theorem complex_on_imaginary_axis (a : ℝ) :
  let z : ℂ := (a^2 - 2*a) + (a^2 - a - 2)*I
  (z.re = 0) ↔ (a = 2 ∨ a = 0) := by sorry

end complex_on_imaginary_axis_l2303_230352


namespace box_surface_area_l2303_230363

/-- Calculates the surface area of the interior of a box formed by removing square corners from a rectangular sheet and folding up the remaining flaps. -/
def interior_surface_area (sheet_length sheet_width corner_size : ℕ) : ℕ :=
  let base_length := sheet_length - 2 * corner_size
  let base_width := sheet_width - 2 * corner_size
  let base_area := base_length * base_width
  let side_area1 := 2 * (base_length * corner_size)
  let side_area2 := 2 * (base_width * corner_size)
  base_area + side_area1 + side_area2

/-- The surface area of the interior of the box is 812 square units. -/
theorem box_surface_area :
  interior_surface_area 28 36 7 = 812 :=
by sorry

end box_surface_area_l2303_230363


namespace quadratic_expression_minimum_l2303_230374

theorem quadratic_expression_minimum :
  ∀ x y : ℝ, 2 * x^2 + 3 * y^2 - 12 * x + 6 * y + 25 ≥ 4 ∧
  ∃ x₀ y₀ : ℝ, 2 * x₀^2 + 3 * y₀^2 - 12 * x₀ + 6 * y₀ + 25 = 4 :=
by sorry

end quadratic_expression_minimum_l2303_230374


namespace q_factor_change_l2303_230342

theorem q_factor_change (w h z : ℝ) (q : ℝ → ℝ → ℝ → ℝ) 
  (hq : q w h z = 5 * w / (4 * h * z^2)) :
  q (4*w) (2*h) (3*z) = (2/9) * q w h z := by
sorry

end q_factor_change_l2303_230342


namespace min_cone_volume_with_sphere_l2303_230336

/-- The minimum volume of a cone containing a sphere of radius 1 that touches the base of the cone -/
theorem min_cone_volume_with_sphere (h r : ℝ) : 
  h > 0 → r > 0 → (1 : ℝ) ≤ h →
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x^2 + (y - 1)^2 = r^2 ∧ y = h - 1) →
  (1/3 * π * r^2 * h) ≥ 8*π/3 :=
by sorry

end min_cone_volume_with_sphere_l2303_230336


namespace dog_drying_time_l2303_230308

/-- Time to dry a short-haired dog in minutes -/
def short_hair_time : ℕ := 10

/-- Time to dry a full-haired dog in minutes -/
def full_hair_time : ℕ := 2 * short_hair_time

/-- Number of short-haired dogs -/
def num_short_hair : ℕ := 6

/-- Number of full-haired dogs -/
def num_full_hair : ℕ := 9

/-- Total time to dry all dogs in hours -/
def total_time_hours : ℚ := (num_short_hair * short_hair_time + num_full_hair * full_hair_time) / 60

theorem dog_drying_time : total_time_hours = 4 := by
  sorry

end dog_drying_time_l2303_230308


namespace inequality_system_solution_set_l2303_230345

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 5 ≥ 0 ∧ x < 7) ↔ (5 ≤ x ∧ x < 7) := by sorry

end inequality_system_solution_set_l2303_230345


namespace problem_statement_l2303_230328

theorem problem_statement (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 16 := by
sorry

end problem_statement_l2303_230328


namespace triangle_area_l2303_230324

/-- The area of a triangle with base 2t and height 3t + 2, where t = 6 -/
theorem triangle_area (t : ℝ) (h : t = 6) : (1/2 : ℝ) * (2*t) * (3*t + 2) = 120 := by
  sorry

end triangle_area_l2303_230324


namespace min_team_size_is_six_l2303_230334

/-- Represents the job parameters and conditions -/
structure JobParameters where
  totalDays : ℕ
  initialDays : ℕ
  initialWorkCompleted : ℚ
  initialTeamSize : ℕ
  rateIncreaseDay : ℕ
  rateIncreaseFactor : ℚ

/-- Calculates the minimum team size required from the rate increase day -/
def minTeamSizeAfterRateIncrease (params : JobParameters) : ℕ :=
  sorry

/-- Theorem stating that the minimum team size after rate increase is 6 -/
theorem min_team_size_is_six (params : JobParameters)
  (h1 : params.totalDays = 40)
  (h2 : params.initialDays = 10)
  (h3 : params.initialWorkCompleted = 1/4)
  (h4 : params.initialTeamSize = 12)
  (h5 : params.rateIncreaseDay = 20)
  (h6 : params.rateIncreaseFactor = 2) :
  minTeamSizeAfterRateIncrease params = 6 :=
sorry

end min_team_size_is_six_l2303_230334


namespace car_average_speed_l2303_230350

/-- Given a car traveling at different speeds for two hours, 
    calculate its average speed. -/
theorem car_average_speed 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : speed1 = 20) 
  (h2 : speed2 = 60) : 
  (speed1 + speed2) / 2 = 40 := by
  sorry

end car_average_speed_l2303_230350


namespace northward_distance_l2303_230368

/-- Calculates the northward distance given total driving time, speed, and westward distance -/
theorem northward_distance 
  (total_time : ℝ) 
  (speed : ℝ) 
  (westward_distance : ℝ) 
  (h1 : total_time = 6) 
  (h2 : speed = 25) 
  (h3 : westward_distance = 95) : 
  speed * total_time - westward_distance = 55 := by
sorry

end northward_distance_l2303_230368


namespace max_area_is_10000_l2303_230301

/-- Represents a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- The perimeter of the garden is 400 feet -/
def perimeterConstraint (g : Garden) : Prop :=
  2 * g.length + 2 * g.width = 400

/-- The length of the garden is at least 100 feet -/
def lengthConstraint (g : Garden) : Prop :=
  g.length ≥ 100

/-- The width of the garden is at least 50 feet -/
def widthConstraint (g : Garden) : Prop :=
  g.width ≥ 50

/-- The area of the garden -/
def area (g : Garden) : ℝ :=
  g.length * g.width

/-- The maximum area of the garden satisfying all constraints is 10000 square feet -/
theorem max_area_is_10000 :
  ∃ (g : Garden),
    perimeterConstraint g ∧
    lengthConstraint g ∧
    widthConstraint g ∧
    area g = 10000 ∧
    ∀ (g' : Garden),
      perimeterConstraint g' ∧
      lengthConstraint g' ∧
      widthConstraint g' →
      area g' ≤ 10000 :=
by sorry

end max_area_is_10000_l2303_230301


namespace movie_screening_guests_l2303_230360

theorem movie_screening_guests :
  ∀ G : ℕ,
  G / 2 + 15 + (G - (G / 2 + 15)) = G →  -- Total guests = women + men + children
  G - (15 / 5 + 4) = 43 →                -- Guests who stayed
  G = 50 :=
by
  sorry

end movie_screening_guests_l2303_230360


namespace total_interest_earned_l2303_230378

def initial_investment : ℝ := 2000
def interest_rate : ℝ := 0.12
def time_period : ℕ := 4

theorem total_interest_earned :
  let final_amount := initial_investment * (1 + interest_rate) ^ time_period
  final_amount - initial_investment = 1147.04 := by
  sorry

end total_interest_earned_l2303_230378


namespace equation_solution_l2303_230325

theorem equation_solution : ∃! x : ℝ, (x^2 + x)^2 + Real.sqrt (x^2 - 1) = 0 ∧ x = -1 := by sorry

end equation_solution_l2303_230325


namespace regular_polygon_perimeter_l2303_230356

/-- A regular polygon with side length 8 and exterior angle 90 degrees has a perimeter of 32 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 8 ∧ 
  exterior_angle = 90 ∧ 
  (n : ℝ) * exterior_angle = 360 →
  n * side_length = 32 :=
by sorry

end regular_polygon_perimeter_l2303_230356


namespace max_dot_product_ellipse_l2303_230307

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def center : ℝ × ℝ := (0, 0)

noncomputable def left_focus : ℝ × ℝ := (-1, 0)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem max_dot_product_ellipse :
  ∃ (max : ℝ), max = 6 ∧
  ∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
  dot_product (P.1 - center.1, P.2 - center.2) (P.1 - left_focus.1, P.2 - left_focus.2) ≤ max :=
sorry

end max_dot_product_ellipse_l2303_230307


namespace sufficient_condition_for_quadratic_inequality_l2303_230355

theorem sufficient_condition_for_quadratic_inequality :
  ∀ x : ℝ, x ≥ 3 → x^2 - 2*x - 3 ≥ 0 := by
  sorry

end sufficient_condition_for_quadratic_inequality_l2303_230355


namespace officer_arrival_time_l2303_230311

/-- The designated arrival time for an officer traveling from A to B -/
noncomputable def designated_arrival_time (s v : ℝ) : ℝ :=
  (v + Real.sqrt (9 * v^2 + 6 * v * s)) / v

theorem officer_arrival_time (s v : ℝ) (h_s : s > 0) (h_v : v > 0) :
  let t := designated_arrival_time s v
  let initial_speed := s / (t + 2)
  s / initial_speed = t + 2 ∧
  s / (2 * initial_speed) + 1 + s / (2 * (initial_speed + v)) = t :=
by sorry

end officer_arrival_time_l2303_230311


namespace extra_eyes_percentage_l2303_230397

def total_frogs : ℕ := 150
def extra_eyes : ℕ := 5

def percentage_with_extra_eyes : ℚ :=
  (extra_eyes : ℚ) / (total_frogs : ℚ) * 100

def rounded_percentage : ℕ := 
  (percentage_with_extra_eyes + 1/2).floor.toNat

theorem extra_eyes_percentage :
  rounded_percentage = 3 :=
sorry

end extra_eyes_percentage_l2303_230397


namespace parabola_line_intersection_l2303_230384

/-- Given a parabola and two points on it, prove the y-intercept of the line through these points -/
theorem parabola_line_intersection (A B : ℝ × ℝ) (a : ℝ) : 
  (A.1^2 = A.2) →  -- A is on the parabola
  (B.1^2 = B.2) →  -- B is on the parabola
  (A.1 < 0) →  -- A is on the left side of y-axis
  (B.1 > 0) →  -- B is on the right side of y-axis
  (∃ k : ℝ, A.2 = k * A.1 + a ∧ B.2 = k * B.1 + a) →  -- Line AB has equation y = kx + a
  (A.1 * B.1 + A.2 * B.2 > 0) →  -- ∠AOB is acute
  a > 1 := by
sorry


end parabola_line_intersection_l2303_230384


namespace lowest_class_size_l2303_230361

theorem lowest_class_size (n : ℕ) : n > 0 ∧ 12 ∣ n ∧ 24 ∣ n → n ≥ 24 :=
by sorry

end lowest_class_size_l2303_230361


namespace m_equals_one_sufficient_not_necessary_l2303_230365

theorem m_equals_one_sufficient_not_necessary (m : ℝ) :
  (m = 1 → |m| = 1) ∧ (∃ m : ℝ, |m| = 1 ∧ m ≠ 1) :=
by sorry

end m_equals_one_sufficient_not_necessary_l2303_230365


namespace magic_rectangle_unique_z_l2303_230383

/-- Represents a 3x3 magic rectangle with some fixed values --/
structure MagicRectangle where
  x : ℕ+
  y : ℕ+
  u : ℕ+
  z : ℕ+

/-- The sum of each row and column in the magic rectangle --/
def row_col_sum (m : MagicRectangle) : ℕ :=
  3 + m.x + 21

/-- The magic rectangle property: all rows and columns have the same sum --/
def is_magic_rectangle (m : MagicRectangle) : Prop :=
  (row_col_sum m = m.y + 25 + m.z) ∧
  (row_col_sum m = 15 + m.u + 4) ∧
  (row_col_sum m = 3 + m.y + 15) ∧
  (row_col_sum m = m.x + 25 + m.u) ∧
  (row_col_sum m = 21 + m.z + 4)

theorem magic_rectangle_unique_z :
  ∀ m : MagicRectangle, is_magic_rectangle m → m.z = 20 :=
by sorry

end magic_rectangle_unique_z_l2303_230383


namespace equation_solution_l2303_230359

theorem equation_solution (x y : ℝ) 
  (h : |x - Real.log y| + Real.sin (π * x) = x + Real.log y) : 
  x = 0 ∧ Real.exp (-1/2) ≤ y ∧ y ≤ Real.exp (1/2) := by
  sorry

end equation_solution_l2303_230359


namespace classroom_benches_l2303_230382

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the number of benches needed given the number of students and students per bench -/
def benchesNeeded (students : ℕ) (studentsPerBench : ℕ) : ℕ := sorry

theorem classroom_benches :
  let studentsBase5 : ℕ := 312
  let studentsPerBench : ℕ := 3
  let studentsBase10 : ℕ := base5ToBase10 studentsBase5
  benchesNeeded studentsBase10 studentsPerBench = 28 := by sorry

end classroom_benches_l2303_230382


namespace sphere_surface_area_of_rectangular_solid_l2303_230353

/-- The surface area of a sphere circumscribing a rectangular solid -/
theorem sphere_surface_area_of_rectangular_solid (l w h : ℝ) (S : ℝ) :
  l = 2 →
  w = 2 →
  h = 1 →
  S = 4 * Real.pi * ((l^2 + w^2 + h^2) / 4) →
  S = 9 * Real.pi :=
by sorry

end sphere_surface_area_of_rectangular_solid_l2303_230353


namespace solve_for_q_l2303_230351

theorem solve_for_q (p q : ℝ) (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p*q = 9) :
  q = (9 + 3*Real.sqrt 5) / 2 := by sorry

end solve_for_q_l2303_230351


namespace complement_M_intersect_N_l2303_230337

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + 2*x - 8 ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end complement_M_intersect_N_l2303_230337


namespace grapes_purchased_l2303_230344

/-- The problem of calculating the amount of grapes purchased -/
theorem grapes_purchased (grape_cost mango_cost total_paid : ℕ) (mango_amount : ℕ) : 
  grape_cost = 70 →
  mango_amount = 9 →
  mango_cost = 65 →
  total_paid = 1145 →
  ∃ (grape_amount : ℕ), grape_amount * grape_cost + mango_amount * mango_cost = total_paid ∧ grape_amount = 8 :=
by sorry

end grapes_purchased_l2303_230344


namespace rhombus_area_l2303_230340

/-- A rhombus with side length √113 and diagonals differing by 8 units has an area of 194 square units. -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 113 → 
  diag_diff = 8 → 
  area = 194 → 
  ∃ (d₁ d₂ : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ d₂ - d₁ = diag_diff ∧ d₁ * d₂ / 2 = area ∧ 
    d₁^2 / 4 + d₂^2 / 4 = side^2 :=
by sorry

end rhombus_area_l2303_230340


namespace inhabitable_earth_surface_fraction_l2303_230369

theorem inhabitable_earth_surface_fraction :
  let total_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (2 : ℚ) / 3
  land_fraction * inhabitable_land_fraction = (2 : ℚ) / 9 :=
by sorry

end inhabitable_earth_surface_fraction_l2303_230369


namespace interest_rate_calculation_l2303_230339

/-- Proves that the rate of interest is 7% given the problem conditions -/
theorem interest_rate_calculation (loan_amount interest_paid : ℚ) : 
  loan_amount = 1500 →
  interest_paid = 735 →
  ∃ (rate : ℚ), 
    (interest_paid = loan_amount * rate * rate / 100) ∧
    (rate = 7) := by
  sorry

end interest_rate_calculation_l2303_230339


namespace three_power_gt_cube_l2303_230367

theorem three_power_gt_cube (n : ℕ) (h : n ≠ 3) : 3^n > n^3 := by
  sorry

end three_power_gt_cube_l2303_230367


namespace ducks_in_other_flock_other_flock_size_l2303_230386

/-- Calculates the number of ducks in the other flock given the conditions of the problem -/
theorem ducks_in_other_flock (original_flock : ℕ) (net_increase_per_year : ℕ) (years : ℕ) (combined_flock : ℕ) : ℕ :=
  let final_original_flock := original_flock + net_increase_per_year * years
  combined_flock - final_original_flock

/-- Proves that the number of ducks in the other flock is 150 given the problem conditions -/
theorem other_flock_size :
  ducks_in_other_flock 100 10 5 300 = 150 := by
  sorry

end ducks_in_other_flock_other_flock_size_l2303_230386


namespace polynomial_evaluation_l2303_230357

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 4*x - 12 = 0) :
  x^3 - 4*x^2 - 12*x + 16 = 16 := by
  sorry

end polynomial_evaluation_l2303_230357


namespace point_c_not_in_region_point_a_in_region_point_b_in_region_point_d_in_region_main_result_l2303_230377

/-- Defines the plane region x + y - 1 ≤ 0 -/
def in_plane_region (x y : ℝ) : Prop := x + y - 1 ≤ 0

/-- The point (-1,3) is not in the plane region -/
theorem point_c_not_in_region : ¬ in_plane_region (-1) 3 := by sorry

/-- Point A (0,0) is in the plane region -/
theorem point_a_in_region : in_plane_region 0 0 := by sorry

/-- Point B (-1,1) is in the plane region -/
theorem point_b_in_region : in_plane_region (-1) 1 := by sorry

/-- Point D (2,-3) is in the plane region -/
theorem point_d_in_region : in_plane_region 2 (-3) := by sorry

/-- The main theorem combining all results -/
theorem main_result : 
  ¬ in_plane_region (-1) 3 ∧ 
  in_plane_region 0 0 ∧ 
  in_plane_region (-1) 1 ∧ 
  in_plane_region 2 (-3) := by sorry

end point_c_not_in_region_point_a_in_region_point_b_in_region_point_d_in_region_main_result_l2303_230377


namespace triangle_larger_segment_l2303_230326

theorem triangle_larger_segment (a b c h x : ℝ) : 
  a = 30 → b = 70 → c = 80 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 65 :=
by sorry

end triangle_larger_segment_l2303_230326


namespace min_boxes_fit_l2303_230399

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

/-- The large box dimensions -/
def largeBox : BoxDimensions := ⟨12, 14, 16⟩

/-- The approximate dimensions of the small irregular boxes -/
def smallBox : BoxDimensions := ⟨3, 7, 2⟩

/-- Theorem stating that at least 64 small boxes can fit into the large box -/
theorem min_boxes_fit (irreg_shape : Prop) : ∃ n : ℕ, n ≥ 64 ∧ n * boxVolume smallBox ≤ boxVolume largeBox := by
  sorry

end min_boxes_fit_l2303_230399


namespace cucumbers_for_twenty_apples_l2303_230366

/-- The number of cucumbers that can be bought for the price of 20 apples,
    given the cost equivalences between apples, bananas, and cucumbers. -/
theorem cucumbers_for_twenty_apples :
  -- Condition 1: Ten apples cost the same as five bananas
  ∀ (apple_cost banana_cost : ℝ),
  10 * apple_cost = 5 * banana_cost →
  -- Condition 2: Three bananas cost the same as four cucumbers
  ∀ (cucumber_cost : ℝ),
  3 * banana_cost = 4 * cucumber_cost →
  -- Conclusion: 20 apples are equivalent in cost to 13 cucumbers
  20 * apple_cost = 13 * cucumber_cost :=
by
  sorry

end cucumbers_for_twenty_apples_l2303_230366


namespace basketball_free_throws_count_l2303_230331

/-- Represents the scoring of a basketball team -/
structure BasketballScore where
  two_points : ℕ
  three_points : ℕ
  free_throws : ℕ

/-- Calculates the total score of a basketball team -/
def total_score (score : BasketballScore) : ℕ :=
  2 * score.two_points + 3 * score.three_points + score.free_throws

/-- Theorem: Given the conditions, the number of free throws is 12 -/
theorem basketball_free_throws_count 
  (score : BasketballScore) 
  (h1 : 3 * score.three_points = 2 * (2 * score.two_points))
  (h2 : score.free_throws = score.two_points + 1)
  (h3 : total_score score = 79) : 
  score.free_throws = 12 := by
  sorry

#check basketball_free_throws_count

end basketball_free_throws_count_l2303_230331


namespace inequality_proof_l2303_230315

variable (x y z : ℝ)

def condition (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z + x * y + y * z + z * x = x + y + z + 1

theorem inequality_proof (h : condition x y z) :
  (1 / 3) * (Real.sqrt ((1 + x^2) / (1 + x)) + 
             Real.sqrt ((1 + y^2) / (1 + y)) + 
             Real.sqrt ((1 + z^2) / (1 + z))) ≤ ((x + y + z) / 3) ^ (5/8) := by
  sorry

end inequality_proof_l2303_230315


namespace triangle_properties_l2303_230388

/-- Given a triangle ABC where sides a and b are roots of x^2 - 2√3x + 2 = 0,
    and cos(A + B) = 1/2, prove the following properties -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (∃ x y : ℝ, x^2 - 2 * Real.sqrt 3 * x + 2 = 0 ∧ y^2 - 2 * Real.sqrt 3 * y + 2 = 0 ∧ x = a ∧ y = b) →
  Real.cos (A + B) = 1/2 →
  C = Real.pi * 2/3 ∧
  (a^2 + b^2 - 2*a*b*Real.cos C) = 10 ∧
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 :=
by sorry

end triangle_properties_l2303_230388


namespace roses_kept_l2303_230372

/-- Given that Ian had 20 roses and gave away specific numbers to different people,
    prove that he kept exactly 1 rose. -/
theorem roses_kept (total : ℕ) (mother grandmother sister : ℕ)
    (h1 : total = 20)
    (h2 : mother = 6)
    (h3 : grandmother = 9)
    (h4 : sister = 4) :
    total - (mother + grandmother + sister) = 1 := by
  sorry

end roses_kept_l2303_230372


namespace double_base_exponent_l2303_230332

theorem double_base_exponent (a b x : ℝ) (hb : b ≠ 0) :
  (2 * a)^(2 * b) = a^b * x^b → x = 4 * a := by
  sorry

end double_base_exponent_l2303_230332


namespace subset_condition_l2303_230304

def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}

def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem subset_condition (m : ℝ) : S m ⊆ P → m ≤ 3 := by
  sorry

end subset_condition_l2303_230304


namespace binary_1100_is_12_l2303_230347

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1100_is_12 :
  binary_to_decimal [false, false, true, true] = 12 := by
  sorry

end binary_1100_is_12_l2303_230347


namespace impossible_digit_assignment_l2303_230306

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  vertices : ℕ
  sides_eq : sides = n
  vertices_eq : vertices = n

/-- Assignment of digits to vertices -/
def DigitAssignment (n : ℕ) := Fin n → Fin 10

/-- Predicate to check if an assignment satisfies the condition -/
def SatisfiesCondition (n : ℕ) (assignment : DigitAssignment n) : Prop :=
  ∀ (i j : Fin 10), i ≠ j → 
    ∃ (v w : Fin n), v.val + 1 = w.val ∨ (v.val = n - 1 ∧ w.val = 0) ∧ 
      assignment v = i ∧ assignment w = j

theorem impossible_digit_assignment :
  ¬ ∃ (assignment : DigitAssignment 45), SatisfiesCondition 45 assignment := by
  sorry

end impossible_digit_assignment_l2303_230306


namespace no_integer_solutions_l2303_230333

theorem no_integer_solutions : ¬∃ (x y : ℤ), (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end no_integer_solutions_l2303_230333


namespace max_x_squared_y_value_l2303_230327

theorem max_x_squared_y_value (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x^3 + y^3 + 3*x*y = 1) :
  ∃ (M : ℝ), M = 4/27 ∧ x^2 * y ≤ M :=
by sorry

end max_x_squared_y_value_l2303_230327


namespace intersection_of_A_and_B_l2303_230319

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Define set B (domain of log(4x - x^2))
def B : Set ℝ := {x | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 1 4 := by sorry

end intersection_of_A_and_B_l2303_230319


namespace polynomial_remainder_l2303_230391

theorem polynomial_remainder (x : ℝ) : (x^13 + 1) % (x - 1) = 2 := by
  sorry

end polynomial_remainder_l2303_230391


namespace g_2010_equals_342_l2303_230376

/-- The function g satisfies the given property for positive integers -/
def g_property (g : ℕ+ → ℕ) : Prop :=
  ∀ (x y m : ℕ+), x + y = 2^(m : ℕ) → g x + g y = 3 * m^2

/-- The main theorem stating that g(2010) = 342 -/
theorem g_2010_equals_342 (g : ℕ+ → ℕ) (h : g_property g) : g 2010 = 342 := by
  sorry

end g_2010_equals_342_l2303_230376


namespace arithmetic_sequence_properties_l2303_230395

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  S : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- Given conditions for the arithmetic sequence -/
def given_conditions (seq : ArithmeticSequence) : Prop :=
  seq.a 5 + seq.a 9 = -2 ∧ seq.S 3 = 57

/-- Theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : given_conditions seq) :
  (∀ n : ℕ, seq.a n = 27 - 4 * n) ∧
  (∃ m : ℕ, ∀ n : ℕ, seq.S n ≤ m ∧ seq.S n = m ↔ n = 6) ∧ 
  (∃ m : ℕ, m = 78 ∧ ∀ n : ℕ, seq.S n ≤ m) :=
by sorry

end arithmetic_sequence_properties_l2303_230395


namespace tangent_through_origin_l2303_230394

/-- Given a curve y = x^a + 1 where a is a real number,
    if the tangent line to this curve at the point (1, 2) passes through the origin,
    then a = 2. -/
theorem tangent_through_origin (a : ℝ) : 
  (∀ x y : ℝ, y = x^a + 1) →
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * (x - 1) + 2 ∧ y = m * x + b) →
  (0 = 0 * 0 + b) →
  a = 2 :=
sorry

end tangent_through_origin_l2303_230394


namespace boat_license_plates_l2303_230393

def letter_choices : ℕ := 3
def digit_choices : ℕ := 10
def num_digits : ℕ := 4

theorem boat_license_plates :
  letter_choices * digit_choices^num_digits = 30000 :=
sorry

end boat_license_plates_l2303_230393


namespace power_product_equals_sum_of_exponents_l2303_230348

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end power_product_equals_sum_of_exponents_l2303_230348


namespace special_integers_characterization_l2303_230371

/-- The set of integers that are divisible by all integers not exceeding their square root -/
def SpecialIntegers : Set ℕ :=
  {n : ℕ | ∀ m : ℕ, m ≤ Real.sqrt n → n % m = 0}

/-- Theorem stating that SpecialIntegers is equal to the set {2, 4, 6, 8, 12, 24} -/
theorem special_integers_characterization :
  SpecialIntegers = {2, 4, 6, 8, 12, 24} := by
  sorry


end special_integers_characterization_l2303_230371


namespace factorization_equality_l2303_230329

theorem factorization_equality (x : ℝ) : 6 * x^2 + 5 * x - 1 = (6 * x - 1) * (x + 1) := by
  sorry

end factorization_equality_l2303_230329


namespace third_dog_summer_avg_distance_proof_l2303_230335

/-- Represents the average daily distance walked by the third dog in summer -/
def third_dog_summer_avg_distance : ℝ := 2.2

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Represents the number of weekend days in a month -/
def weekend_days : ℕ := 8

/-- Represents the distance walked by the third dog on a summer weekday -/
def third_dog_summer_distance : ℝ := 3

theorem third_dog_summer_avg_distance_proof :
  third_dog_summer_avg_distance = 
    (third_dog_summer_distance * (days_in_month - weekend_days)) / days_in_month :=
by sorry

end third_dog_summer_avg_distance_proof_l2303_230335


namespace cubic_root_sum_l2303_230389

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 24*r^2 + 50*r - 24 = 0 →
  s^3 - 24*s^2 + 50*s - 24 = 0 →
  t^3 - 24*t^2 + 50*t - 24 = 0 →
  (r / (1/r + s*t)) + (s / (1/s + t*r)) + (t / (1/t + r*s)) = 19.04 := by
sorry

end cubic_root_sum_l2303_230389


namespace employees_without_increase_l2303_230313

theorem employees_without_increase (total : ℕ) (salary_percent : ℚ) (travel_percent : ℚ) (both_percent : ℚ) :
  total = 480 →
  salary_percent = 1/10 →
  travel_percent = 1/5 →
  both_percent = 1/20 →
  (total : ℚ) - (salary_percent + travel_percent - both_percent) * total = 360 := by
  sorry

end employees_without_increase_l2303_230313


namespace tetris_arrangement_exists_l2303_230323

/-- Represents a Tetris piece type -/
inductive TetrisPiece
  | O | I | T | S | Z | L | J

/-- Represents a position on the 6x6 grid -/
structure Position where
  x : Fin 6
  y : Fin 6

/-- Represents a placed Tetris piece on the grid -/
structure PlacedPiece where
  piece : TetrisPiece
  positions : List Position

/-- Checks if a list of placed pieces forms a valid arrangement -/
def isValidArrangement (pieces : List PlacedPiece) : Prop :=
  -- Each position on the 6x6 grid is covered exactly once
  ∀ (x y : Fin 6), ∃! p : PlacedPiece, p ∈ pieces ∧ Position.mk x y ∈ p.positions

/-- Checks if all piece types are used at least once -/
def allPiecesUsed (pieces : List PlacedPiece) : Prop :=
  ∀ t : TetrisPiece, ∃ p : PlacedPiece, p ∈ pieces ∧ p.piece = t

/-- Main theorem: There exists a valid arrangement of Tetris pieces -/
theorem tetris_arrangement_exists : 
  ∃ (pieces : List PlacedPiece), isValidArrangement pieces ∧ allPiecesUsed pieces :=
sorry

end tetris_arrangement_exists_l2303_230323


namespace min_value_of_max_expression_l2303_230305

theorem min_value_of_max_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let M := max (x * y + 2 / z) (max (z + 2 / y) (y + z + 1 / x))
  M ≥ 3 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    max (x' * y' + 2 / z') (max (z' + 2 / y') (y' + z' + 1 / x')) = 3 :=
by sorry

end min_value_of_max_expression_l2303_230305


namespace max_sum_cubes_l2303_230392

theorem max_sum_cubes (a b c d : ℝ) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 16)
  (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d) :
  a^3 + b^3 + c^3 + d^3 ≤ 64 ∧ 
  ∃ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 16 ∧ 
                   x ≠ y ∧ y ≠ z ∧ z ≠ w ∧
                   x^3 + y^3 + z^3 + w^3 = 64 :=
by sorry

end max_sum_cubes_l2303_230392


namespace ball_events_properties_l2303_230310

-- Define the sample space
def Ω : Type := Fin 8

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A, B, and C
def A : Set Ω := sorry
def B : Set Ω := sorry
def C : Set Ω := sorry

-- Theorem statement
theorem ball_events_properties :
  (P (A ∩ C) = 0) ∧
  (P (A ∩ B) = P A * P B) ∧
  (P (B ∩ C) = P B * P C) := by
  sorry

end ball_events_properties_l2303_230310


namespace equation_solution_l2303_230343

theorem equation_solution : 
  ∃! x : ℝ, x > 0 ∧ Real.sqrt (3 * x - 2) + 9 / Real.sqrt (3 * x - 2) = 6 := by
  sorry

end equation_solution_l2303_230343


namespace cross_area_l2303_230318

-- Define the grid size
def gridSize : Nat := 6

-- Define the center point of the cross
def centerPoint : (Nat × Nat) := (3, 3)

-- Define the arm length of the cross
def armLength : Nat := 1

-- Define the boundary points of the cross
def boundaryPoints : List (Nat × Nat) := [(3, 1), (1, 3), (3, 3), (3, 5), (5, 3)]

-- Define the interior points of the cross
def interiorPoints : List (Nat × Nat) := [(3, 2), (2, 3), (4, 3), (3, 4)]

-- Theorem: The area of the cross is 6 square units
theorem cross_area : Nat := by
  sorry

end cross_area_l2303_230318


namespace orange_trees_remaining_fruit_l2303_230349

theorem orange_trees_remaining_fruit (num_trees : ℕ) (fruits_per_tree : ℕ) (fraction_picked : ℚ) : 
  num_trees = 8 → 
  fruits_per_tree = 200 → 
  fraction_picked = 2/5 → 
  (num_trees * fruits_per_tree) - (num_trees * fruits_per_tree * fraction_picked) = 960 := by
sorry

end orange_trees_remaining_fruit_l2303_230349


namespace bridge_length_calculation_l2303_230309

/-- The length of the bridge in meters -/
def bridge_length : ℝ := 200

/-- The time it takes for the train to cross the bridge in seconds -/
def bridge_crossing_time : ℝ := 10

/-- The time it takes for the train to pass a lamp post on the bridge in seconds -/
def lamppost_passing_time : ℝ := 5

/-- The length of the train in meters -/
def train_length : ℝ := 200

theorem bridge_length_calculation :
  bridge_length = train_length := by sorry

end bridge_length_calculation_l2303_230309


namespace oranges_remaining_proof_l2303_230385

/-- The number of oranges Michaela needs to eat to get full -/
def michaela_oranges : ℕ := 20

/-- The number of oranges Cassandra needs to eat to get full -/
def cassandra_oranges : ℕ := 2 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 90

/-- The number of oranges remaining after both Michaela and Cassandra eat until they're full -/
def remaining_oranges : ℕ := total_oranges - (michaela_oranges + cassandra_oranges)

theorem oranges_remaining_proof : remaining_oranges = 30 := by
  sorry

end oranges_remaining_proof_l2303_230385


namespace vertex_in_second_quadrant_l2303_230390

/-- The quadratic function f(x) = -(x+1)^2 + 2 -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 2

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := -1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := f vertex_x

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The vertex of f(x) = -(x+1)^2 + 2 is in the second quadrant -/
theorem vertex_in_second_quadrant : is_in_second_quadrant vertex_x vertex_y := by
  sorry

end vertex_in_second_quadrant_l2303_230390


namespace square_perimeter_ratio_l2303_230320

theorem square_perimeter_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 / b^2 = 16 / 25) → ((4 * a) / (4 * b) = 4 / 5) :=
sorry

end square_perimeter_ratio_l2303_230320


namespace xiao_ying_pays_20_yuan_l2303_230312

/-- Represents the price of flowers in yuan -/
structure FlowerPrices where
  rose : ℚ
  carnation : ℚ
  lily : ℚ

/-- The conditions from Xiao Hong's and Xiao Li's purchases -/
def satisfies_conditions (p : FlowerPrices) : Prop :=
  3 * p.rose + 7 * p.carnation + p.lily = 14 ∧
  4 * p.rose + 10 * p.carnation + p.lily = 16

/-- Xiao Ying's purchase -/
def xiao_ying_purchase (p : FlowerPrices) : ℚ :=
  2 * (p.rose + p.carnation + p.lily)

/-- The main theorem to prove -/
theorem xiao_ying_pays_20_yuan (p : FlowerPrices) :
  satisfies_conditions p → xiao_ying_purchase p = 20 := by
  sorry


end xiao_ying_pays_20_yuan_l2303_230312


namespace one_approval_probability_l2303_230358

/-- The probability of a voter approving the council's measures -/
def approval_rate : ℝ := 0.6

/-- The number of voters polled -/
def num_polled : ℕ := 4

/-- The probability of exactly one voter approving out of the polled voters -/
def prob_one_approval : ℝ := 4 * (approval_rate * (1 - approval_rate)^3)

/-- Theorem stating that the probability of exactly one voter approving is 0.1536 -/
theorem one_approval_probability : prob_one_approval = 0.1536 := by
  sorry

end one_approval_probability_l2303_230358


namespace ethanol_in_fuel_tank_l2303_230379

/-- Proves that the total amount of ethanol in a fuel tank is 30 gallons given specific conditions -/
theorem ethanol_in_fuel_tank (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) 
  (h1 : tank_capacity = 218)
  (h2 : fuel_a_volume = 122)
  (h3 : fuel_a_ethanol_percent = 0.12)
  (h4 : fuel_b_ethanol_percent = 0.16) :
  fuel_a_volume * fuel_a_ethanol_percent + (tank_capacity - fuel_a_volume) * fuel_b_ethanol_percent = 30 := by
  sorry

end ethanol_in_fuel_tank_l2303_230379


namespace abs_z_squared_l2303_230381

-- Define a complex number z
variable (z : ℂ)

-- State the theorem
theorem abs_z_squared (h : z + Complex.abs z = 2 + 8*I) : Complex.abs z ^ 2 = 289 := by
  sorry

end abs_z_squared_l2303_230381


namespace two_red_or_blue_marbles_probability_l2303_230321

/-- The probability of drawing two marbles consecutively where both are either red or blue
    from a bag containing 5 red, 3 blue, and 7 yellow marbles, with replacement. -/
theorem two_red_or_blue_marbles_probability :
  let red_marbles : ℕ := 5
  let blue_marbles : ℕ := 3
  let yellow_marbles : ℕ := 7
  let total_marbles : ℕ := red_marbles + blue_marbles + yellow_marbles
  let prob_red_or_blue : ℚ := (red_marbles + blue_marbles : ℚ) / total_marbles
  (prob_red_or_blue * prob_red_or_blue) = 64 / 225 := by
  sorry

end two_red_or_blue_marbles_probability_l2303_230321


namespace rectangular_solid_surface_area_l2303_230300

theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a → Prime b → Prime c → 
  a * b * c = 399 → 
  2 * (a * b + b * c + c * a) = 422 := by
sorry

end rectangular_solid_surface_area_l2303_230300


namespace arithmetic_sequence_unique_formula_arithmetic_sequence_possible_formulas_l2303_230302

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_unique_formula 
  (a₁ d : ℤ) 
  (h1 : arithmetic_sequence a₁ d 11 = 0) 
  (h2 : sum_arithmetic_sequence a₁ d 14 = 98) :
  ∀ n : ℕ, arithmetic_sequence a₁ d n = 22 - 2 * n :=
sorry

theorem arithmetic_sequence_possible_formulas 
  (a₁ d : ℤ) 
  (h1 : a₁ ≥ 6) 
  (h2 : arithmetic_sequence a₁ d 11 > 0) 
  (h3 : sum_arithmetic_sequence a₁ d 14 ≤ 77) :
  (∀ n : ℕ, arithmetic_sequence a₁ d n = 12 - n) ∨ 
  (∀ n : ℕ, arithmetic_sequence a₁ d n = 13 - n) :=
sorry

end arithmetic_sequence_unique_formula_arithmetic_sequence_possible_formulas_l2303_230302


namespace solution_set_rational_inequality_l2303_230346

theorem solution_set_rational_inequality :
  ∀ x : ℝ, x ≠ 0 → ((x - 1) / x ≥ 2 ↔ -1 ≤ x ∧ x < 0) := by sorry

end solution_set_rational_inequality_l2303_230346


namespace clock_problem_l2303_230364

/-- Represents a time on a 12-hour digital clock -/
structure Time where
  hour : Nat
  minute : Nat
  second : Nat
  deriving Repr

/-- Adds a duration to a given time -/
def addDuration (t : Time) (hours minutes seconds : Nat) : Time :=
  sorry

/-- Calculates the sum of hour, minute, and second values of a time -/
def timeSum (t : Time) : Nat :=
  sorry

theorem clock_problem :
  let initial_time : Time := ⟨3, 0, 0⟩
  let final_time := addDuration initial_time 85 58 30
  final_time = ⟨4, 58, 30⟩ ∧ timeSum final_time = 92 := by sorry

end clock_problem_l2303_230364


namespace jogger_difference_l2303_230373

def jogger_problem (tyson martha alexander christopher natasha : ℕ) : Prop :=
  martha = max 0 (tyson - 15) ∧
  alexander = tyson + 22 ∧
  christopher = 20 * tyson ∧
  natasha = 2 * (martha + alexander) ∧
  christopher = 80

theorem jogger_difference (tyson martha alexander christopher natasha : ℕ) 
  (h : jogger_problem tyson martha alexander christopher natasha) : 
  christopher - natasha = 28 := by
sorry

end jogger_difference_l2303_230373


namespace f_has_unique_zero_l2303_230396

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x + (a / 2) * x^2

theorem f_has_unique_zero (a : ℝ) (h : a ∈ Set.Icc (-Real.exp 1) 0) :
  ∃! x, f a x = 0 := by
  sorry

end f_has_unique_zero_l2303_230396


namespace log_equation_solution_l2303_230380

theorem log_equation_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq2 : q ≠ 2) :
  Real.log p + Real.log q = Real.log (2 * (p + q)) → p = (2 * (q - 1)) / (q - 2) := by
  sorry

end log_equation_solution_l2303_230380


namespace doll_price_is_five_l2303_230322

/-- Represents the inventory and financial data of Stella's antique shop --/
structure AntiqueShop where
  num_dolls : ℕ
  num_clocks : ℕ
  num_glasses : ℕ
  clock_price : ℕ
  glass_price : ℕ
  total_cost : ℕ
  total_profit : ℕ

/-- Calculates the price of each doll given the shop's data --/
def calculate_doll_price (shop : AntiqueShop) : ℕ :=
  let total_revenue := shop.total_cost + shop.total_profit
  let clock_revenue := shop.num_clocks * shop.clock_price
  let glass_revenue := shop.num_glasses * shop.glass_price
  let doll_revenue := total_revenue - clock_revenue - glass_revenue
  doll_revenue / shop.num_dolls

/-- Theorem stating that the doll price is $5 given Stella's shop data --/
theorem doll_price_is_five (shop : AntiqueShop) 
  (h1 : shop.num_dolls = 3)
  (h2 : shop.num_clocks = 2)
  (h3 : shop.num_glasses = 5)
  (h4 : shop.clock_price = 15)
  (h5 : shop.glass_price = 4)
  (h6 : shop.total_cost = 40)
  (h7 : shop.total_profit = 25) :
  calculate_doll_price shop = 5 := by
  sorry

#eval calculate_doll_price {
  num_dolls := 3,
  num_clocks := 2,
  num_glasses := 5,
  clock_price := 15,
  glass_price := 4,
  total_cost := 40,
  total_profit := 25
}

end doll_price_is_five_l2303_230322
