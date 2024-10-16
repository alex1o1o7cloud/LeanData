import Mathlib

namespace NUMINAMATH_CALUDE_exists_F_for_P_l2601_260129

/-- A ternary polynomial with real coefficients -/
def TernaryPolynomial := ℝ → ℝ → ℝ → ℝ

/-- The conditions that P must satisfy -/
def SatisfiesConditions (P : TernaryPolynomial) : Prop :=
  ∀ x y z : ℝ, 
    P x y z = P x y (x*y - z) ∧
    P x y z = P x (z*x - y) z ∧
    P x y z = P (y*z - x) y z

/-- The theorem statement -/
theorem exists_F_for_P (P : TernaryPolynomial) (h : SatisfiesConditions P) :
  ∃ F : ℝ → ℝ, ∀ x y z : ℝ, P x y z = F (x^2 + y^2 + z^2 - x*y*z) :=
sorry

end NUMINAMATH_CALUDE_exists_F_for_P_l2601_260129


namespace NUMINAMATH_CALUDE_consecutive_cubes_l2601_260131

theorem consecutive_cubes (a b c d : ℤ) (y z w x v : ℤ) : 
  (d = c + 1 ∧ c = b + 1 ∧ b = a + 1) → 
  (v = x + 1 ∧ x = w + 1 ∧ w = z + 1 ∧ z = y + 1) →
  ((a^3 + b^3 + c^3 = d^3) ↔ (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6)) ∧
  (y^3 + z^3 + w^3 + x^3 ≠ v^3) := by
sorry

end NUMINAMATH_CALUDE_consecutive_cubes_l2601_260131


namespace NUMINAMATH_CALUDE_root_equation_value_l2601_260186

theorem root_equation_value (b c : ℝ) : 
  (2 : ℝ)^2 - b * 2 + c = 0 → 4 * b - 2 * c + 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l2601_260186


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_smallest_sum_is_achievable_l2601_260121

def S : Finset Int := {8, -7, 2, -4, 20}

theorem smallest_sum_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a + b + c ≥ -9 :=
by sorry

theorem smallest_sum_is_achievable :
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_smallest_sum_is_achievable_l2601_260121


namespace NUMINAMATH_CALUDE_initial_girls_count_l2601_260115

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 20) = b) →
  (7 * (b - 54) = g - 20) →
  g = 39 := by
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l2601_260115


namespace NUMINAMATH_CALUDE_september_electricity_usage_l2601_260126

theorem september_electricity_usage
  (october_usage : ℕ)
  (savings_percentage : ℚ)
  (h1 : october_usage = 1400)
  (h2 : savings_percentage = 30 / 100)
  (h3 : october_usage = (1 - savings_percentage) * september_usage) :
  september_usage = 2000 :=
sorry

end NUMINAMATH_CALUDE_september_electricity_usage_l2601_260126


namespace NUMINAMATH_CALUDE_largest_integer_problem_l2601_260154

theorem largest_integer_problem (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →  -- Four different integers
  (a + b + c + d) / 4 = 76 →  -- Average is 76
  a ≥ 37 →  -- Smallest integer is at least 37
  d ≤ 190 :=  -- Largest integer is at most 190
by sorry

end NUMINAMATH_CALUDE_largest_integer_problem_l2601_260154


namespace NUMINAMATH_CALUDE_wooden_strip_sawing_time_l2601_260141

theorem wooden_strip_sawing_time 
  (initial_length : ℝ) 
  (initial_sections : ℕ) 
  (initial_time : ℝ) 
  (final_sections : ℕ) 
  (h1 : initial_length = 12) 
  (h2 : initial_sections = 4) 
  (h3 : initial_time = 12) 
  (h4 : final_sections = 8) : 
  (initial_time / (initial_sections - 1)) * (final_sections - 1) = 28 := by
sorry

end NUMINAMATH_CALUDE_wooden_strip_sawing_time_l2601_260141


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_427_l2601_260160

theorem smallest_next_divisor_after_427 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : m % 2 = 0) (h3 : m % 427 = 0) :
  ∃ (d : ℕ), d > 427 ∧ m % d = 0 ∧ d = 434 ∧ 
  ∀ (x : ℕ), 427 < x ∧ x < 434 → m % x ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_427_l2601_260160


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l2601_260100

/-- The number of yellow highlighters in Kaya's teacher's desk -/
def yellow_highlighters : ℕ := 11 - (4 + 5)

/-- The total number of highlighters -/
def total_highlighters : ℕ := 11

/-- The number of pink highlighters -/
def pink_highlighters : ℕ := 4

/-- The number of blue highlighters -/
def blue_highlighters : ℕ := 5

theorem yellow_highlighters_count :
  yellow_highlighters = 2 :=
by sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l2601_260100


namespace NUMINAMATH_CALUDE_square_equals_1369_l2601_260125

theorem square_equals_1369 (x : ℤ) (h : x^2 = 1369) : (x + 1) * (x - 1) = 1368 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_1369_l2601_260125


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2601_260128

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 9 = 1 ∨ y^2 / 25 - x^2 / 9 = 1

theorem hyperbola_standard_equation
  (center_origin : ℝ × ℝ)
  (real_axis_length : ℝ)
  (imaginary_axis_length : ℝ)
  (h1 : center_origin = (0, 0))
  (h2 : real_axis_length = 10)
  (h3 : imaginary_axis_length = 6) :
  ∀ x y : ℝ, hyperbola_equation x y := by
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2601_260128


namespace NUMINAMATH_CALUDE_truncated_cube_properties_l2601_260192

/-- A space-filling cube arrangement --/
structure CubeArrangement where
  /-- The space is filled with equal cubes --/
  space_filled : Bool
  /-- Eight cubes converge at each vertex --/
  eight_cubes_at_vertex : Bool

/-- A truncated cube in the arrangement --/
structure TruncatedCube where
  /-- The number of faces after truncation --/
  num_faces : Nat
  /-- The number of octagonal faces --/
  num_octagonal_faces : Nat
  /-- The number of triangular faces --/
  num_triangular_faces : Nat

/-- The result of truncating and joining cubes in the arrangement --/
def truncate_and_join (arr : CubeArrangement) : TruncatedCube × Rational :=
  sorry

/-- Theorem stating the properties of truncated cubes and space occupation --/
theorem truncated_cube_properties (arr : CubeArrangement) :
  arr.space_filled ∧ arr.eight_cubes_at_vertex →
  let (truncated_cube, octahedra_space) := truncate_and_join arr
  truncated_cube.num_faces = 14 ∧
  truncated_cube.num_octagonal_faces = 6 ∧
  truncated_cube.num_triangular_faces = 8 ∧
  octahedra_space = 5/6 :=
  sorry

end NUMINAMATH_CALUDE_truncated_cube_properties_l2601_260192


namespace NUMINAMATH_CALUDE_m_3_sufficient_not_necessary_l2601_260135

def A (m : ℝ) : Set ℝ := {-1, m^2}
def B : Set ℝ := {2, 9}

theorem m_3_sufficient_not_necessary :
  (∀ m : ℝ, m = 3 → A m ∩ B = {9}) ∧
  ¬(∀ m : ℝ, A m ∩ B = {9} → m = 3) := by
  sorry

end NUMINAMATH_CALUDE_m_3_sufficient_not_necessary_l2601_260135


namespace NUMINAMATH_CALUDE_tv_screen_length_tv_screen_length_approx_l2601_260181

theorem tv_screen_length (diagonal : ℝ) (ratio_length_height : ℚ) : ℝ :=
  let length := Real.sqrt ((ratio_length_height ^ 2 * diagonal ^ 2) / (1 + ratio_length_height ^ 2))
  length

theorem tv_screen_length_approx :
  ∃ ε > 0, abs (tv_screen_length 27 (4/3) - 21.6) < ε :=
sorry

end NUMINAMATH_CALUDE_tv_screen_length_tv_screen_length_approx_l2601_260181


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_surface_area_l2601_260194

/-- Given an equilateral cone with volume V and a tetrahedron circumscribed around it 
    with an equilateral triangle base, this theorem proves the volume and surface area 
    of the tetrahedron. -/
theorem tetrahedron_volume_and_surface_area 
  (V : ℝ) -- Volume of the equilateral cone
  (h : V > 0) -- Assumption that volume is positive
  : 
  ∃ (K F : ℝ), 
    K = (3 * V * Real.sqrt 3) / Real.pi ∧ 
    F = 9 * Real.sqrt 3 * (((3 * V ^ 2) / Real.pi ^ 2) ^ (1/3 : ℝ)) ∧
    K > 0 ∧ 
    F > 0
  := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_surface_area_l2601_260194


namespace NUMINAMATH_CALUDE_quadratic_roots_coefficients_l2601_260133

theorem quadratic_roots_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 2) →
  b = -3 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_coefficients_l2601_260133


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2601_260177

theorem trigonometric_identities (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 11 / 14 ∧
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2601_260177


namespace NUMINAMATH_CALUDE_target_hit_probability_l2601_260162

theorem target_hit_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h_prob_A : prob_A = 0.8) 
  (h_prob_B : prob_B = 0.7) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2601_260162


namespace NUMINAMATH_CALUDE_min_production_avoids_loss_less_than_min_production_incurs_loss_l2601_260147

/-- The minimum production quantity to avoid a loss -/
def min_production : ℝ := 150

/-- The unit selling price in million yuan -/
def unit_price : ℝ := 0.25

/-- The total cost function in million yuan for x units -/
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The total revenue function in million yuan for x units -/
def total_revenue (x : ℝ) : ℝ := unit_price * x

/-- Theorem stating that the minimum production quantity to avoid a loss is 150 units -/
theorem min_production_avoids_loss :
  ∀ x : ℝ, x ≥ min_production → total_revenue x ≥ total_cost x :=
by
  sorry

/-- Theorem stating that any production quantity less than 150 units results in a loss -/
theorem less_than_min_production_incurs_loss :
  ∀ x : ℝ, 0 ≤ x ∧ x < min_production → total_revenue x < total_cost x :=
by
  sorry

end NUMINAMATH_CALUDE_min_production_avoids_loss_less_than_min_production_incurs_loss_l2601_260147


namespace NUMINAMATH_CALUDE_correct_sample_ids_l2601_260185

/-- A function that generates the sample IDs based on the given conditions -/
def generateSampleIDs (populationSize : Nat) (sampleSize : Nat) : List Nat :=
  (List.range sampleSize).map (fun i => 6 * i + 3)

/-- The theorem stating that the generated sample IDs match the expected result -/
theorem correct_sample_ids :
  generateSampleIDs 60 10 = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57] := by
  sorry

#eval generateSampleIDs 60 10

end NUMINAMATH_CALUDE_correct_sample_ids_l2601_260185


namespace NUMINAMATH_CALUDE_breakfast_cooking_time_l2601_260156

theorem breakfast_cooking_time (num_sausages num_eggs egg_time total_time : ℕ) 
  (h1 : num_sausages = 3)
  (h2 : num_eggs = 6)
  (h3 : egg_time = 4)
  (h4 : total_time = 39) :
  ∃ (sausage_time : ℕ), 
    sausage_time * num_sausages + egg_time * num_eggs = total_time ∧ 
    sausage_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cooking_time_l2601_260156


namespace NUMINAMATH_CALUDE_donation_analysis_l2601_260130

/-- Represents the donation amounts and their frequencies --/
def donation_data : List (ℕ × ℕ) := [(5, 1), (10, 5), (15, 3), (20, 1)]

/-- Total number of students in the sample --/
def sample_size : ℕ := 10

/-- Total number of students in the school --/
def school_size : ℕ := 2200

/-- Calculates the mode of the donation data --/
def mode (data : List (ℕ × ℕ)) : ℕ := sorry

/-- Calculates the median of the donation data --/
def median (data : List (ℕ × ℕ)) : ℕ := sorry

/-- Calculates the average donation amount --/
def average (data : List (ℕ × ℕ)) : ℚ := sorry

/-- Estimates the total donation for the school --/
def estimate_total (avg : ℚ) (school_size : ℕ) : ℕ := sorry

theorem donation_analysis :
  mode donation_data = 10 ∧
  median donation_data = 10 ∧
  average donation_data = 12 ∧
  estimate_total (average donation_data) school_size = 26400 := by sorry

end NUMINAMATH_CALUDE_donation_analysis_l2601_260130


namespace NUMINAMATH_CALUDE_sum_x_y_equals_twenty_l2601_260113

theorem sum_x_y_equals_twenty (x y : ℝ) 
  (h1 : |x| - x + y = 13) 
  (h2 : x - |y| + y = 7) : 
  x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_twenty_l2601_260113


namespace NUMINAMATH_CALUDE_road_repaving_l2601_260169

theorem road_repaving (total_repaved : ℕ) (previously_repaved : ℕ) :
  total_repaved = 4938 ∧ previously_repaved = 4133 →
  total_repaved - previously_repaved = 805 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l2601_260169


namespace NUMINAMATH_CALUDE_invalid_league_schedule_l2601_260161

/-- Represents a league schedule --/
structure LeagueSchedule where
  num_teams : Nat
  num_dates : Nat
  max_games_per_date : Nat

/-- Calculate the total number of games in a round-robin tournament --/
def total_games (schedule : LeagueSchedule) : Nat :=
  schedule.num_teams * (schedule.num_teams - 1) / 2

/-- Check if a schedule is valid --/
def is_valid_schedule (schedule : LeagueSchedule) : Prop :=
  total_games schedule ≤ schedule.num_dates * schedule.max_games_per_date

/-- Theorem stating that the given schedule is invalid --/
theorem invalid_league_schedule : 
  ¬ is_valid_schedule ⟨20, 5, 8⟩ := by
  sorry

#eval total_games ⟨20, 5, 8⟩

end NUMINAMATH_CALUDE_invalid_league_schedule_l2601_260161


namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l2601_260119

def east_wind_prob : ℚ := 9/30
def rain_prob : ℚ := 11/30
def both_prob : ℚ := 8/30

theorem conditional_probability_rain_given_east_wind :
  (both_prob / east_wind_prob : ℚ) = 8/9 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l2601_260119


namespace NUMINAMATH_CALUDE_triangle_side_length_l2601_260168

theorem triangle_side_length (a c b : ℝ) (B : ℝ) : 
  a = 3 * Real.sqrt 3 → 
  c = 2 → 
  B = 150 * π / 180 → 
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B → 
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2601_260168


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2601_260132

/-- 
Given a boat that covers the same distance downstream and upstream,
with known travel times and stream speed, this theorem proves
the speed of the boat in still water.
-/
theorem boat_speed_in_still_water 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (stream_speed : ℝ) 
  (h1 : downstream_time = 1)
  (h2 : upstream_time = 1.5)
  (h3 : stream_speed = 3) : 
  ∃ (boat_speed : ℝ), boat_speed = 15 ∧ 
    downstream_time * (boat_speed + stream_speed) = 
    upstream_time * (boat_speed - stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2601_260132


namespace NUMINAMATH_CALUDE_map_distance_calculation_l2601_260190

/-- Given a map scale and a measured distance on the map, calculate the actual distance in kilometers. -/
theorem map_distance_calculation (scale : ℚ) (map_distance : ℚ) (actual_distance : ℚ) :
  scale = 1 / 1000000 →
  map_distance = 12 →
  actual_distance = map_distance / scale / 100000 →
  actual_distance = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_calculation_l2601_260190


namespace NUMINAMATH_CALUDE_coinciding_vertices_theorem_l2601_260144

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of marked vertices in a polygon -/
def MarkedVertices (n : ℕ) := Finset (Fin n)

/-- The number of coinciding vertices when overlaying two polygons -/
def CoincidingVertices (n : ℕ) (p1 p2 : RegularPolygon n) (m1 m2 : MarkedVertices n) : ℕ := sorry

/-- Rotates a polygon by a given angle -/
def RotatePolygon (n : ℕ) (p : RegularPolygon n) (angle : ℝ) : RegularPolygon n := sorry

theorem coinciding_vertices_theorem :
  ∀ (p1 p2 : RegularPolygon 16) (m1 m2 : MarkedVertices 16),
  (m1.card = 7 ∧ m2.card = 7) →
  ∃ (angle : ℝ), CoincidingVertices 16 p1 (RotatePolygon 16 p2 angle) m1 m2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_vertices_theorem_l2601_260144


namespace NUMINAMATH_CALUDE_height_percentage_difference_l2601_260164

theorem height_percentage_difference (A B : ℝ) (h : B = A * (1 + 1/3)) :
  (B - A) / B = 1/4 := by sorry

end NUMINAMATH_CALUDE_height_percentage_difference_l2601_260164


namespace NUMINAMATH_CALUDE_specific_classroom_seats_l2601_260114

/-- Represents a tiered classroom with increasing seats per row -/
structure TieredClassroom where
  rows : ℕ
  firstRowSeats : ℕ
  seatIncrease : ℕ

/-- Calculates the number of seats in the nth row -/
def seatsInRow (c : TieredClassroom) (n : ℕ) : ℕ :=
  c.firstRowSeats + (n - 1) * c.seatIncrease

/-- Calculates the total number of seats in the classroom -/
def totalSeats (c : TieredClassroom) : ℕ :=
  (c.firstRowSeats + seatsInRow c c.rows) * c.rows / 2

/-- Theorem stating the total number of seats in the specific classroom configuration -/
theorem specific_classroom_seats :
  let c : TieredClassroom := { rows := 22, firstRowSeats := 22, seatIncrease := 2 }
  totalSeats c = 946 := by sorry

end NUMINAMATH_CALUDE_specific_classroom_seats_l2601_260114


namespace NUMINAMATH_CALUDE_amithab_january_expenditure_l2601_260175

def january_expenditure (avg_jan_jun avg_feb_jul july_expenditure : ℝ) : ℝ :=
  6 * avg_feb_jul - 6 * avg_jan_jun + july_expenditure

theorem amithab_january_expenditure :
  january_expenditure 4200 4250 1500 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_amithab_january_expenditure_l2601_260175


namespace NUMINAMATH_CALUDE_average_marks_proof_l2601_260104

-- Define the marks for each subject
def physics : ℝ := 125
def chemistry : ℝ := 15
def mathematics : ℝ := 55

-- Define the conditions
theorem average_marks_proof :
  -- Average of all three subjects is 65
  (physics + chemistry + mathematics) / 3 = 65 ∧
  -- Average of physics and mathematics is 90
  (physics + mathematics) / 2 = 90 ∧
  -- Average of physics and chemistry is 70
  (physics + chemistry) / 2 = 70 ∧
  -- Physics marks are 125
  physics = 125 →
  -- Prove that chemistry is the subject that averages 70 with physics
  (physics + chemistry) / 2 = 70 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_proof_l2601_260104


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2601_260179

theorem complex_fraction_sum (x y : ℝ) :
  (1 - Complex.I) / (2 + Complex.I) = Complex.mk x y →
  x + y = -2/5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2601_260179


namespace NUMINAMATH_CALUDE_symmetry_points_l2601_260188

/-- Given points M, N, P, and Q in a 2D plane, prove that Q has coordinates (b,a) -/
theorem symmetry_points (a b : ℝ) : 
  let M : ℝ × ℝ := (a, b)
  let N : ℝ × ℝ := (a, -b)  -- M symmetric to N w.r.t. x-axis
  let P : ℝ × ℝ := (-a, -b) -- P symmetric to N w.r.t. y-axis
  let Q : ℝ × ℝ := (b, a)   -- Q symmetric to P w.r.t. line x+y=0
  Q = (b, a) := by sorry

end NUMINAMATH_CALUDE_symmetry_points_l2601_260188


namespace NUMINAMATH_CALUDE_distance_from_blast_site_l2601_260180

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 330

/-- The time between the first blast and when the man heard the second blast, in seconds -/
def time_between_blasts : ℝ := 30 * 60 + 24

/-- The time between the first and second blasts, in seconds -/
def time_between_actual_blasts : ℝ := 30 * 60

/-- The distance the man traveled when he heard the second blast -/
def distance_traveled : ℝ := speed_of_sound * (time_between_blasts - time_between_actual_blasts)

theorem distance_from_blast_site :
  distance_traveled = 7920 := by sorry

end NUMINAMATH_CALUDE_distance_from_blast_site_l2601_260180


namespace NUMINAMATH_CALUDE_area_of_PQRS_l2601_260166

/-- A circle with an inscribed square ABCD and another inscribed square PQRS -/
structure InscribedSquares where
  /-- The radius of the circle -/
  r : ℝ
  /-- The side length of square ABCD -/
  s : ℝ
  /-- Half the side length of square PQRS -/
  t : ℝ
  /-- The area of square ABCD is 4 -/
  h_area : s^2 = 4
  /-- The radius of the circle is related to the side of ABCD -/
  h_radius : r^2 = 2 * s^2
  /-- Relationship between r, s, and t based on the Pythagorean theorem -/
  h_pythagorean : (s/2 + t)^2 + t^2 = r^2

/-- The area of square PQRS in the configuration of InscribedSquares -/
def areaOfPQRS (cfg : InscribedSquares) : ℝ := (2 * cfg.t)^2

/-- Theorem stating that the area of PQRS is 2 - √3 -/
theorem area_of_PQRS (cfg : InscribedSquares) : areaOfPQRS cfg = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_PQRS_l2601_260166


namespace NUMINAMATH_CALUDE_cosine_difference_l2601_260151

theorem cosine_difference (α β : Real) 
  (h1 : Real.sin α - Real.sin β = 1/2) 
  (h2 : Real.cos α - Real.cos β = 1/3) : 
  Real.cos (α - β) = 59/72 := by
  sorry

end NUMINAMATH_CALUDE_cosine_difference_l2601_260151


namespace NUMINAMATH_CALUDE_lorelai_ate_180_jellybeans_l2601_260198

-- Define the number of jellybeans each person has
def gigi_jellybeans : ℕ := 15
def rory_jellybeans : ℕ := gigi_jellybeans + 30

-- Define the total number of jellybeans both girls have
def total_girls_jellybeans : ℕ := gigi_jellybeans + rory_jellybeans

-- Define the number of jellybeans Lorelai has eaten
def lorelai_jellybeans : ℕ := 3 * total_girls_jellybeans

-- Theorem to prove
theorem lorelai_ate_180_jellybeans : lorelai_jellybeans = 180 := by
  sorry

end NUMINAMATH_CALUDE_lorelai_ate_180_jellybeans_l2601_260198


namespace NUMINAMATH_CALUDE_smallest_n_with_hcf_condition_l2601_260182

theorem smallest_n_with_hcf_condition : 
  ∃ (n : ℕ), n > 0 ∧ n ≠ 11 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n ∧ m ≠ 11 → Nat.gcd (m - 11) (3 * m + 20) = 1) ∧
  Nat.gcd (n - 11) (3 * n + 20) > 1 ∧
  n = 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_hcf_condition_l2601_260182


namespace NUMINAMATH_CALUDE_altitude_properties_l2601_260124

-- Define the triangle ABC
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (-3, -1)

-- Define vector BC
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define the altitude AD
def AD : ℝ × ℝ → Prop := λ D => 
  -- AD is perpendicular to BC
  (D.1 - A.1) * BC.1 + (D.2 - A.2) * BC.2 = 0 ∧
  -- D lies on line BC
  ∃ t : ℝ, D = (B.1 + t * BC.1, B.2 + t * BC.2)

-- Theorem statement
theorem altitude_properties : 
  ∃ D : ℝ × ℝ, AD D ∧ 
    ((D.1 - A.1)^2 + (D.2 - A.2)^2 = 5) ∧ 
    D = (1, 1) :=
sorry

end NUMINAMATH_CALUDE_altitude_properties_l2601_260124


namespace NUMINAMATH_CALUDE_triangle_lines_l2601_260174

-- Define the triangle ABC
def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (-5, 2)

-- Define the altitude BD
def altitude_BD (x y : ℝ) : Prop := 4 * x - 3 * y - 24 = 0

-- Define the median BE
def median_BE (x y : ℝ) : Prop := x - 7 * y - 6 = 0

-- Theorem statement
theorem triangle_lines :
  (∀ x y : ℝ, altitude_BD x y ↔ 
    (x - B.1) * (C.2 - A.2) = (y - B.2) * (C.1 - A.1)) ∧
  (∀ x y : ℝ, median_BE x y ↔ 
    2 * (x - B.1) = (A.1 + C.1) - 2 * B.1 ∧
    2 * (y - B.2) = (A.2 + C.2) - 2 * B.2) :=
sorry

end NUMINAMATH_CALUDE_triangle_lines_l2601_260174


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2601_260191

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2601_260191


namespace NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l2601_260197

theorem absolute_value_square_sum_zero (x y : ℝ) :
  |x + 2| + (y - 1)^2 = 0 → x = -2 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l2601_260197


namespace NUMINAMATH_CALUDE_tax_revenue_change_l2601_260152

theorem tax_revenue_change (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax := 0.86 * T
  let new_consumption := 1.15 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (new_revenue / original_revenue - 1) * 100 = -1.1 := by
sorry

end NUMINAMATH_CALUDE_tax_revenue_change_l2601_260152


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_C_condition_l2601_260107

-- Define the sets A, B, and C
def A : Set ℝ := {y | ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ y = 2^x}
def B : Set ℝ := {x | 0 < Real.log x ∧ Real.log x < 1}
def C (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2*t}

-- Theorem 1: Intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < Real.exp 1} := by sorry

-- Theorem 2: Condition for t when A ∩ C = C
theorem intersection_A_C_condition (t : ℝ) : A ∩ C t = C t → t ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_C_condition_l2601_260107


namespace NUMINAMATH_CALUDE_certain_number_problem_l2601_260105

theorem certain_number_problem (x : ℤ) (h : x + 14 = 56) : 3 * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2601_260105


namespace NUMINAMATH_CALUDE_max_sum_fraction_min_sum_fraction_l2601_260184

def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The maximum value of A/B + C/P given six different digits from Digits -/
theorem max_sum_fraction (A B C P Q R : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧
                B ≠ C ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧
                C ≠ P ∧ C ≠ Q ∧ C ≠ R ∧
                P ≠ Q ∧ P ≠ R ∧
                Q ≠ R)
  (h_in_digits : A ∈ Digits ∧ B ∈ Digits ∧ C ∈ Digits ∧ 
                 P ∈ Digits ∧ Q ∈ Digits ∧ R ∈ Digits) :
  (A : ℚ) / B + (C : ℚ) / P ≤ 13 :=
sorry

/-- The minimum value of Q/R + P/C using the remaining digits -/
theorem min_sum_fraction (A B C P Q R : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧
                B ≠ C ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧
                C ≠ P ∧ C ≠ Q ∧ C ≠ R ∧
                P ≠ Q ∧ P ≠ R ∧
                Q ≠ R)
  (h_in_digits : A ∈ Digits ∧ B ∈ Digits ∧ C ∈ Digits ∧ 
                 P ∈ Digits ∧ Q ∈ Digits ∧ R ∈ Digits) :
  (Q : ℚ) / R + (P : ℚ) / C ≥ 23 / 21 :=
sorry

end NUMINAMATH_CALUDE_max_sum_fraction_min_sum_fraction_l2601_260184


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2601_260158

/-- Isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  /-- Length of lateral sides AB and CD -/
  lateral_side : ℝ
  /-- Ratio of AH : AK : AC -/
  ratio_ah : ℝ
  ratio_ak : ℝ
  ratio_ac : ℝ
  /-- Conditions -/
  lateral_positive : lateral_side > 0
  ratio_positive : ratio_ah > 0 ∧ ratio_ak > 0 ∧ ratio_ac > 0
  ratio_order : ratio_ah < ratio_ak ∧ ratio_ak < ratio_ac

/-- The area of the isosceles trapezoid with given properties is 180 -/
theorem isosceles_trapezoid_area
  (t : IsoscelesTrapezoid)
  (h1 : t.lateral_side = 10)
  (h2 : t.ratio_ah = 5 ∧ t.ratio_ak = 14 ∧ t.ratio_ac = 15) :
  ∃ (area : ℝ), area = 180 :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2601_260158


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l2601_260163

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = !![3, 7; -2, -5]) : 
  (A^3)⁻¹ = !![13, -15; -14, -29] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l2601_260163


namespace NUMINAMATH_CALUDE_program_output_correct_l2601_260167

/-- The output function of Xiao Wang's program -/
def program_output (n : ℕ+) : ℚ :=
  n / (n^2 + 1)

/-- The theorem stating the correctness of the program output -/
theorem program_output_correct (n : ℕ+) :
  program_output n = n / (n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_program_output_correct_l2601_260167


namespace NUMINAMATH_CALUDE_initial_tax_rate_calculation_l2601_260138

/-- Proves that given an annual income of $36,000, if lowering the tax rate to 32% 
    results in a savings of $5,040, then the initial tax rate was 46%. -/
theorem initial_tax_rate_calculation 
  (annual_income : ℝ) 
  (new_tax_rate : ℝ) 
  (savings : ℝ) 
  (h1 : annual_income = 36000)
  (h2 : new_tax_rate = 32)
  (h3 : savings = 5040) :
  ∃ (initial_tax_rate : ℝ), 
    initial_tax_rate = 46 ∧ 
    (initial_tax_rate / 100 * annual_income) - (new_tax_rate / 100 * annual_income) = savings :=
by sorry

end NUMINAMATH_CALUDE_initial_tax_rate_calculation_l2601_260138


namespace NUMINAMATH_CALUDE_cards_per_page_l2601_260122

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) :
  (new_cards + old_cards) / pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l2601_260122


namespace NUMINAMATH_CALUDE_emily_necklaces_l2601_260170

theorem emily_necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) (h1 : total_beads = 16) (h2 : beads_per_necklace = 8) :
  total_beads / beads_per_necklace = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_necklaces_l2601_260170


namespace NUMINAMATH_CALUDE_zoe_earnings_per_candy_bar_l2601_260101

def trip_cost : ℚ := 485
def grandma_contribution : ℚ := 250
def candy_bars_to_sell : ℕ := 188

theorem zoe_earnings_per_candy_bar :
  (trip_cost - grandma_contribution) / candy_bars_to_sell = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_zoe_earnings_per_candy_bar_l2601_260101


namespace NUMINAMATH_CALUDE_sheelas_monthly_income_l2601_260123

/-- Given that Sheela's deposit is 20% of her monthly income, 
    prove that her monthly income is Rs. 25000 -/
theorem sheelas_monthly_income 
  (deposit : ℝ) 
  (deposit_percentage : ℝ) 
  (h1 : deposit = 5000)
  (h2 : deposit_percentage = 0.20)
  (h3 : deposit = deposit_percentage * sheelas_income) : 
  sheelas_income = 25000 :=
by
  sorry

#check sheelas_monthly_income

end NUMINAMATH_CALUDE_sheelas_monthly_income_l2601_260123


namespace NUMINAMATH_CALUDE_gold_bar_distribution_l2601_260137

theorem gold_bar_distribution (initial_bars : ℕ) (lost_bars : ℕ) (friends : ℕ) 
  (h1 : initial_bars = 100)
  (h2 : lost_bars = 20)
  (h3 : friends = 4)
  (h4 : friends > 0) :
  (initial_bars - lost_bars) / friends = 20 := by
  sorry

end NUMINAMATH_CALUDE_gold_bar_distribution_l2601_260137


namespace NUMINAMATH_CALUDE_shortest_distance_to_x_axis_l2601_260172

/-- Two points on a parabola -/
structure PointsOnParabola where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  on_parabola₁ : x₁^2 = 4*y₁
  on_parabola₂ : x₂^2 = 4*y₂
  distance : Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 6

/-- Theorem: The shortest distance from the midpoint of AB to the x-axis is 2 -/
theorem shortest_distance_to_x_axis (p : PointsOnParabola) :
  (p.y₁ + p.y₂) / 2 ≥ 2 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_x_axis_l2601_260172


namespace NUMINAMATH_CALUDE_square_root_extraction_l2601_260110

theorem square_root_extraction : 413^2 = 170569 ∧ 419^2 = 175561 := by
  sorry

end NUMINAMATH_CALUDE_square_root_extraction_l2601_260110


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2601_260120

def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 7 * x + 1
def parabola2 (x : ℝ) : ℝ := 8 * x^2 + 5 * x + 1

theorem parabolas_intersection :
  ∃! (s : Set (ℝ × ℝ)), s = {(-2, 23), (0, 1)} ∧
  (∀ (x y : ℝ), (x, y) ∈ s ↔ parabola1 x = y ∧ parabola2 x = y) :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2601_260120


namespace NUMINAMATH_CALUDE_gcf_98_140_245_l2601_260127

theorem gcf_98_140_245 : Nat.gcd 98 (Nat.gcd 140 245) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_98_140_245_l2601_260127


namespace NUMINAMATH_CALUDE_total_windows_needed_l2601_260108

/-- Proves that the total number of windows needed for a building is 9,
    given the specified conditions. -/
theorem total_windows_needed (installed : ℕ) (install_time : ℕ) (remaining_time : ℕ)
    (h1 : installed = 6)
    (h2 : install_time = 6)
    (h3 : remaining_time = 18) :
  installed + remaining_time / install_time = 9 := by
  sorry

#check total_windows_needed

end NUMINAMATH_CALUDE_total_windows_needed_l2601_260108


namespace NUMINAMATH_CALUDE_acute_angle_trig_inequality_l2601_260148

theorem acute_angle_trig_inequality (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  1/2 < Real.sqrt 3 / 2 * Real.sin α + 1/2 * Real.cos α ∧
  Real.sqrt 3 / 2 * Real.sin α + 1/2 * Real.cos α ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_trig_inequality_l2601_260148


namespace NUMINAMATH_CALUDE_polynomial_sum_l2601_260196

theorem polynomial_sum (d a b c e : ℤ) (h : d ≠ 0) :
  (10 * d + 15 + 12 * d^2 + 2 * d^3) + (4 * d - 3 + 2 * d^2) = a * d^3 + b * d^2 + c * d + e →
  a + b + c + e = 42 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2601_260196


namespace NUMINAMATH_CALUDE_complex_magnitude_l2601_260145

theorem complex_magnitude (z : ℂ) (h : z^4 = 80 - 96*I) : Complex.abs z = 5^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2601_260145


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l2601_260178

theorem consecutive_sum_product (a : ℤ) : (3*a + 3) * (3*a + 12) ≠ 111111111 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_product_l2601_260178


namespace NUMINAMATH_CALUDE_money_distribution_l2601_260111

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500)
  (h2 : A + C = 200)
  (h3 : B + C = 350) : 
  C = 50 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l2601_260111


namespace NUMINAMATH_CALUDE_g_sum_of_5_and_neg_5_l2601_260143

def g (x : ℝ) : ℝ := 2 * x^8 + 3 * x^6 - 4 * x^4 + 5

theorem g_sum_of_5_and_neg_5 (h : g 5 = 7) : g 5 + g (-5) = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_of_5_and_neg_5_l2601_260143


namespace NUMINAMATH_CALUDE_solve_equation_l2601_260165

theorem solve_equation (x : ℝ) : 3 + 2 * (x - 3) = 24.16 → x = 13.58 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2601_260165


namespace NUMINAMATH_CALUDE_expression_simplification_l2601_260139

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) :
  (2*a + 2) / a / (4 / a^2) - a / (a + 1) = (a^3 + 2*a^2 - a) / (2*a + 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2601_260139


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2601_260116

theorem diophantine_equation_solutions : 
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ S ↔ 
      (0 < x ∧ 0 < y ∧ x < y ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 2007)) ∧
    S.card = 7 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2601_260116


namespace NUMINAMATH_CALUDE_family_ages_solution_l2601_260117

/-- Represents the ages of the family members -/
structure FamilyAges where
  father : ℕ
  person : ℕ
  sister : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ∃ u : ℕ,
    ages.father + 6 = 3 * (ages.person - u) ∧
    ages.father = ages.person + ages.sister - u ∧
    ages.person = ages.father - u ∧
    ages.father + 19 = 2 * ages.sister

/-- The theorem to be proved -/
theorem family_ages_solution :
  ∃ ages : FamilyAges,
    satisfiesConditions ages ∧
    ages.father = 69 ∧
    ages.person = 47 ∧
    ages.sister = 44 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l2601_260117


namespace NUMINAMATH_CALUDE_intersection_determines_a_l2601_260102

theorem intersection_determines_a (a : ℝ) : 
  let A : Set ℝ := {1, 2}
  let B : Set ℝ := {a, a^2 + 3}
  A ∩ B = {1} → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_determines_a_l2601_260102


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2601_260150

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_s

/-- Proof that a train traveling at 60 km/hr and crossing a pole in 18 seconds has a length of approximately 300.06 meters -/
theorem train_length_proof (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |train_length 60 18 - 300.06| < δ :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2601_260150


namespace NUMINAMATH_CALUDE_exponential_graph_condition_l2601_260118

/-- A function f : ℝ → ℝ does not pass through the first quadrant if
    for all x > 0, f(x) ≤ 0 or for all x ≥ 0, f(x) < 0 -/
def not_pass_first_quadrant (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x ≤ 0) ∨ (∀ x ≥ 0, f x < 0)

theorem exponential_graph_condition
  (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1)
  (h : not_pass_first_quadrant (fun x ↦ a^x + b - 1)) :
  0 < a ∧ a < 1 ∧ b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_exponential_graph_condition_l2601_260118


namespace NUMINAMATH_CALUDE_cards_per_layer_in_house_of_cards_l2601_260189

/-- Proves that given 16 decks of 52 cards each, and a house of cards with 32 layers
    where each layer has the same number of cards, the number of cards per layer is 26. -/
theorem cards_per_layer_in_house_of_cards 
  (num_decks : ℕ) 
  (cards_per_deck : ℕ) 
  (num_layers : ℕ) 
  (h1 : num_decks = 16) 
  (h2 : cards_per_deck = 52) 
  (h3 : num_layers = 32) : 
  (num_decks * cards_per_deck) / num_layers = 26 := by
  sorry

#eval (16 * 52) / 32  -- Expected output: 26

end NUMINAMATH_CALUDE_cards_per_layer_in_house_of_cards_l2601_260189


namespace NUMINAMATH_CALUDE_not_adjacent_2010_2011_l2601_260109

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Ordering function for natural numbers based on digit sum and then the number itself -/
def digit_sum_order (a b : ℕ) : Prop :=
  (digit_sum a < digit_sum b) ∨ (digit_sum a = digit_sum b ∧ a ≤ b)

theorem not_adjacent_2010_2011 (start : ℕ) :
  ¬ ∃ (i : ℕ), i < 99 ∧
    (((start + i = 2010 ∧ start + (i + 1) = 2011) ∨
     (start + i = 2011 ∧ start + (i + 1) = 2010)) ∧
    (∀ j k : ℕ, j < k ∧ k < 100 → digit_sum_order (start + j) (start + k))) :=
sorry

end NUMINAMATH_CALUDE_not_adjacent_2010_2011_l2601_260109


namespace NUMINAMATH_CALUDE_inequality_proof_l2601_260173

theorem inequality_proof (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a * d = b * c) : 
  (a - d)^2 ≥ 4*d + 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2601_260173


namespace NUMINAMATH_CALUDE_area_outside_rectangle_in_square_l2601_260134

/-- The area of the region outside a rectangle contained within a square -/
theorem area_outside_rectangle_in_square (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 8 ∧ rect_length = 4 ∧ rect_width = 2 →
  square_side^2 - rect_length * rect_width = 56 := by
sorry


end NUMINAMATH_CALUDE_area_outside_rectangle_in_square_l2601_260134


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l2601_260183

theorem unique_quadratic_root (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l2601_260183


namespace NUMINAMATH_CALUDE_integral_roots_system_l2601_260171

theorem integral_roots_system : ∃! (x y z : ℤ),
  (z : ℝ) ^ (x : ℝ) = (y : ℝ) ^ (2 * x : ℝ) ∧
  (2 : ℝ) ^ (z : ℝ) = 2 * (4 : ℝ) ^ (x : ℝ) ∧
  x + y + z = 16 ∧
  x = 4 ∧ y = 3 ∧ z = 9 := by
sorry

end NUMINAMATH_CALUDE_integral_roots_system_l2601_260171


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2601_260142

def U : Finset Int := {-1, 0, 1, 2, 3}
def A : Finset Int := {0, 1, 2}
def B : Finset Int := {2, 3}

theorem intersection_with_complement :
  A ∩ (U \ B) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2601_260142


namespace NUMINAMATH_CALUDE_sum_product_bounds_l2601_260155

theorem sum_product_bounds (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  -(1/2) ≤ a*b + b*c + c*a ∧ a*b + b*c + c*a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l2601_260155


namespace NUMINAMATH_CALUDE_four_adjacent_squares_l2601_260199

/-- A square in a plane -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ
  is_square : IsSquare vertices

/-- Two vertices are adjacent if they are consecutive in the cyclic order of the square -/
def adjacent (s : Square) (i j : Fin 4) : Prop :=
  (j = i + 1) ∨ (i = 3 ∧ j = 0)

/-- A square shares two adjacent vertices with another square -/
def shares_adjacent_vertices (s1 s2 : Square) : Prop :=
  ∃ (i j : Fin 4), adjacent s1 i j ∧ s1.vertices i = s2.vertices 0 ∧ s1.vertices j = s2.vertices 1

/-- The main theorem: there are exactly 4 squares sharing adjacent vertices with a given square -/
theorem four_adjacent_squares (s : Square) :
  ∃! (squares : Finset Square), squares.card = 4 ∧
    ∀ s' ∈ squares, shares_adjacent_vertices s s' :=
  sorry

end NUMINAMATH_CALUDE_four_adjacent_squares_l2601_260199


namespace NUMINAMATH_CALUDE_max_pairs_proof_max_pairs_achievable_l2601_260193

/-- The maximum number of pairs that can be chosen from the set {1, 2, ..., 2017}
    such that a_i < b_i, no two pairs share a common element, and all sums a_i + b_i
    are distinct and less than or equal to 2017. -/
def max_pairs : ℕ := 806

theorem max_pairs_proof :
  ∀ (k : ℕ) (a b : Fin k → ℕ),
  (∀ i : Fin k, a i < b i) →
  (∀ i : Fin k, b i ≤ 2017) →
  (∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j) →
  (∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j) →
  (∀ i : Fin k, a i + b i ≤ 2017) →
  k ≤ max_pairs :=
by sorry

theorem max_pairs_achievable :
  ∃ (k : ℕ) (a b : Fin k → ℕ),
  k = max_pairs ∧
  (∀ i : Fin k, a i < b i) ∧
  (∀ i : Fin k, b i ≤ 2017) ∧
  (∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j) ∧
  (∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j) ∧
  (∀ i : Fin k, a i + b i ≤ 2017) :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_proof_max_pairs_achievable_l2601_260193


namespace NUMINAMATH_CALUDE_product_of_roots_l2601_260103

theorem product_of_roots (x : ℝ) : (x - 4) * (x + 5) = -24 → ∃ y : ℝ, (x * y = 4 ∧ (y - 4) * (y + 5) = -24) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2601_260103


namespace NUMINAMATH_CALUDE_janet_stickers_l2601_260112

theorem janet_stickers (x : ℕ) : 
  x + 53 = 56 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_janet_stickers_l2601_260112


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2601_260176

theorem linear_equation_solution (x y a : ℝ) : 
  x = 1 → y = 2 → x + a = 3 * y - 2 → a = 3 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2601_260176


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2601_260159

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a line with slope m -/
structure Line where
  m : ℝ

theorem parabola_line_intersection
  (para : Parabola)
  (line : Line)
  (A B : Point)
  (h_line_slope : line.m = 2 * Real.sqrt 2)
  (h_on_parabola_A : A.y ^ 2 = 2 * para.p * A.x)
  (h_on_parabola_B : B.y ^ 2 = 2 * para.p * B.x)
  (h_on_line_A : A.y = line.m * (A.x - para.p / 2))
  (h_on_line_B : B.y = line.m * (B.x - para.p / 2))
  (h_x_order : A.x < B.x)
  (h_distance : Real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2) = 9) :
  (∃ (C : Point),
    C.y ^ 2 = 8 * C.x ∧
    (C.x = A.x + 0 * (B.x - A.x) ∧ C.y = A.y + 0 * (B.y - A.y) ∨
     C.x = A.x + 2 * (B.x - A.x) ∧ C.y = A.y + 2 * (B.y - A.y))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2601_260159


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2601_260149

theorem rationalize_denominator :
  ∃ (A B C : ℤ), (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2601_260149


namespace NUMINAMATH_CALUDE_initial_average_problem_l2601_260106

theorem initial_average_problem (n : ℕ) (A : ℝ) (added_value : ℝ) (new_average : ℝ) 
  (h1 : n = 15)
  (h2 : added_value = 14)
  (h3 : new_average = 54)
  (h4 : (n : ℝ) * A + n * added_value = n * new_average) :
  A = 40 := by
sorry

end NUMINAMATH_CALUDE_initial_average_problem_l2601_260106


namespace NUMINAMATH_CALUDE_vector_perpendicular_to_sum_l2601_260187

/-- Given vectors a and b in ℝ², prove that a is perpendicular to (a + b) -/
theorem vector_perpendicular_to_sum (a b : ℝ × ℝ) (ha : a = (2, -1)) (hb : b = (1, 7)) :
  a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0 := by
  sorry

#check vector_perpendicular_to_sum

end NUMINAMATH_CALUDE_vector_perpendicular_to_sum_l2601_260187


namespace NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l2601_260146

/-- Represents the duration of a song in seconds -/
def SongDuration := ℕ

/-- Represents a playlist of songs -/
def Playlist := List SongDuration

/-- Creates a playlist with 12 songs, where each song is 20 seconds longer than the previous one -/
def createPlaylist : Playlist :=
  List.range 12 |>.map (fun i => 20 * (i + 1))

/-- The duration of the favorite song in seconds -/
def favoriteSongDuration : SongDuration := 4 * 60

/-- The total listening time in seconds -/
def totalListeningTime : SongDuration := 5 * 60

/-- Calculates the probability of not hearing the entire favorite song within the first 5 minutes -/
def probabilityNotHearingFavoriteSong (playlist : Playlist) : ℚ :=
  let totalArrangements := Nat.factorial playlist.length
  let favorableArrangements := 3 * Nat.factorial (playlist.length - 2)
  1 - (favorableArrangements : ℚ) / totalArrangements

theorem probability_not_hearing_favorite_song :
  probabilityNotHearingFavoriteSong createPlaylist = 43 / 44 := by
  sorry

#eval probabilityNotHearingFavoriteSong createPlaylist

end NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l2601_260146


namespace NUMINAMATH_CALUDE_investment_value_after_eight_years_l2601_260140

/-- Calculates the final value of an investment under simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that given the conditions, the investment value after 8 years is $660 -/
theorem investment_value_after_eight_years 
  (P : ℝ) -- Initial investment (principal)
  (h1 : simple_interest P 0.04 3 = 560) -- Value after 3 years
  : simple_interest P 0.04 8 = 660 := by
  sorry

#check investment_value_after_eight_years

end NUMINAMATH_CALUDE_investment_value_after_eight_years_l2601_260140


namespace NUMINAMATH_CALUDE_smallest_number_property_l2601_260195

/-- The smallest natural number divisible by 5 with a digit sum of 100 -/
def smallest_number : ℕ := 599999999995

/-- Function to calculate the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- Theorem stating that 599999999995 is the smallest natural number divisible by 5 with a digit sum of 100 -/
theorem smallest_number_property :
  (∀ m : ℕ, m < smallest_number → (m % 5 = 0 → digit_sum m ≠ 100)) ∧
  smallest_number % 5 = 0 ∧
  digit_sum smallest_number = 100 :=
by sorry

#eval smallest_number
#eval digit_sum smallest_number
#eval smallest_number % 5

end NUMINAMATH_CALUDE_smallest_number_property_l2601_260195


namespace NUMINAMATH_CALUDE_pond_water_volume_l2601_260136

/-- Calculates the water volume in a pond after a given number of days -/
def water_volume (initial_volume : ℕ) (evaporation_rate : ℕ) (water_added : ℕ) (add_interval : ℕ) (days : ℕ) : ℕ :=
  initial_volume - evaporation_rate * days + (days / add_interval) * water_added

theorem pond_water_volume :
  water_volume 500 1 10 7 35 = 515 := by
  sorry

end NUMINAMATH_CALUDE_pond_water_volume_l2601_260136


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l2601_260153

theorem product_of_sum_of_squares (a b c d : ℤ) :
  let m := a^2 + b^2
  let n := c^2 + d^2
  m * n = (a*c - b*d)^2 + (a*d + b*c)^2 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l2601_260153


namespace NUMINAMATH_CALUDE_grassy_area_length_l2601_260157

/-- The length of the grassy area in a rectangular plot with a gravel path -/
theorem grassy_area_length 
  (total_length : ℝ) 
  (path_width : ℝ) 
  (h1 : total_length = 110) 
  (h2 : path_width = 2.5) : 
  total_length - 2 * path_width = 105 := by
sorry

end NUMINAMATH_CALUDE_grassy_area_length_l2601_260157
