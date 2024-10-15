import Mathlib

namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l3156_315690

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d > 0) :
  let decimal := (n : ℚ) / d
  let whole_part := (decimal.floor : ℤ)
  let fractional_part := decimal - whole_part
  let expanded := fractional_part * (10 ^ 10 : ℚ)  -- Multiply by a large power of 10 to see digits
  ∃ k, 0 < k ∧ k ≤ 10 ∧ (expanded.floor : ℤ) % (10 ^ k) ≠ 0 ∧
      ∀ j, 0 < j ∧ j < k → (expanded.floor : ℤ) % (10 ^ j) = 0 →
  (n = 5 ∧ d = 3125) → k - 1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l3156_315690


namespace NUMINAMATH_CALUDE_cristina_croissants_l3156_315631

/-- The number of croissants Cristina baked -/
def total_croissants (num_guests : ℕ) (croissants_per_guest : ℕ) : ℕ :=
  num_guests * croissants_per_guest

/-- Proof that Cristina baked 14 croissants -/
theorem cristina_croissants :
  total_croissants 7 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_cristina_croissants_l3156_315631


namespace NUMINAMATH_CALUDE_boxes_theorem_l3156_315686

def boxes_problem (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ) : Prop :=
  let neither := total - (markers + erasers - both)
  neither = 3

theorem boxes_theorem : boxes_problem 12 7 5 3 := by
  sorry

end NUMINAMATH_CALUDE_boxes_theorem_l3156_315686


namespace NUMINAMATH_CALUDE_property_1_property_2_property_3_f_satisfies_all_properties_l3156_315657

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Property 1: f(xy) = f(x)f(y)
theorem property_1 : ∀ x y : ℝ, f (x * y) = f x * f y := by sorry

-- Property 2: f'(x) is an even function
theorem property_2 : ∀ x : ℝ, (deriv f) (-x) = (deriv f) x := by sorry

-- Property 3: f(x) is monotonically increasing on (0, +∞)
theorem property_3 : ∀ x y : ℝ, 0 < x → x < y → f x < f y := by sorry

-- Main theorem: f(x) = x^3 satisfies all three properties
theorem f_satisfies_all_properties :
  (∀ x y : ℝ, f (x * y) = f x * f y) ∧
  (∀ x : ℝ, (deriv f) (-x) = (deriv f) x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_property_1_property_2_property_3_f_satisfies_all_properties_l3156_315657


namespace NUMINAMATH_CALUDE_incenter_coords_l3156_315689

/-- Triangle ABC with incenter I -/
structure TriangleWithIncenter where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Length of side CA -/
  CA : ℝ
  /-- Incenter I of the triangle -/
  I : ℝ × ℝ
  /-- Coordinates of incenter I as (x, y, z) where x⃗A + y⃗B + z⃗C = ⃗I -/
  coords : ℝ × ℝ × ℝ

/-- The theorem stating that the coordinates of the incenter are (2/9, 1/3, 4/9) -/
theorem incenter_coords (t : TriangleWithIncenter) 
  (h1 : t.AB = 6)
  (h2 : t.BC = 8)
  (h3 : t.CA = 4)
  (h4 : t.coords.1 + t.coords.2.1 + t.coords.2.2 = 1) :
  t.coords = (2/9, 1/3, 4/9) := by
  sorry

end NUMINAMATH_CALUDE_incenter_coords_l3156_315689


namespace NUMINAMATH_CALUDE_room_tiling_l3156_315622

theorem room_tiling (room_length room_width : ℕ) 
  (border_tile_size inner_tile_size : ℕ) : 
  room_length = 16 → 
  room_width = 12 → 
  border_tile_size = 1 → 
  inner_tile_size = 2 → 
  (2 * (room_length - 2 + room_width - 2) + 4) + 
  ((room_length - 2) * (room_width - 2)) / (inner_tile_size ^ 2) = 87 :=
by
  sorry

end NUMINAMATH_CALUDE_room_tiling_l3156_315622


namespace NUMINAMATH_CALUDE_num_winning_scores_l3156_315670

/-- Represents a cross country meet with 3 teams of 4 runners each -/
structure CrossCountryMeet where
  numTeams : Nat
  runnersPerTeam : Nat
  totalRunners : Nat
  (team_count : numTeams = 3)
  (runner_count : runnersPerTeam = 4)
  (total_runners : totalRunners = numTeams * runnersPerTeam)

/-- Calculates the total score of all runners -/
def totalScore (meet : CrossCountryMeet) : Nat :=
  meet.totalRunners * (meet.totalRunners + 1) / 2

/-- Calculates the minimum possible winning score -/
def minWinningScore (meet : CrossCountryMeet) : Nat :=
  meet.runnersPerTeam * (meet.runnersPerTeam + 1) / 2

/-- Calculates the maximum possible winning score -/
def maxWinningScore (meet : CrossCountryMeet) : Nat :=
  totalScore meet / meet.numTeams

/-- Theorem stating the number of different winning scores possible -/
theorem num_winning_scores (meet : CrossCountryMeet) :
  (maxWinningScore meet - minWinningScore meet + 1) = 17 := by
  sorry


end NUMINAMATH_CALUDE_num_winning_scores_l3156_315670


namespace NUMINAMATH_CALUDE_stating_min_natives_correct_stating_min_natives_sufficient_l3156_315605

/-- Represents the minimum number of natives required for the joke-sharing problem. -/
def min_natives (k : ℕ) : ℕ := 2^k

/-- 
Theorem stating that min_natives(k) is the smallest number of natives needed
for each native to know at least k jokes (apart from their own) after crossing the river.
-/
theorem min_natives_correct (k : ℕ) :
  ∀ N : ℕ, (∀ native : Fin N, 
    (∃ known_jokes : Finset (Fin N), 
      known_jokes.card ≥ k ∧ 
      native ∉ known_jokes ∧
      (∀ joke ∈ known_jokes, joke ≠ native))) 
    → N ≥ min_natives k :=
by
  sorry

/-- 
Theorem stating that min_natives(k) is sufficient for each native to know
at least k jokes (apart from their own) after crossing the river.
-/
theorem min_natives_sufficient (k : ℕ) :
  ∃ crossing_strategy : Unit,
    ∀ native : Fin (min_natives k),
      ∃ known_jokes : Finset (Fin (min_natives k)),
        known_jokes.card ≥ k ∧
        native ∉ known_jokes ∧
        (∀ joke ∈ known_jokes, joke ≠ native) :=
by
  sorry

end NUMINAMATH_CALUDE_stating_min_natives_correct_stating_min_natives_sufficient_l3156_315605


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l3156_315659

theorem square_area_error_percentage (s : ℝ) (h : s > 0) :
  let measured_side := 1.06 * s
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let error_percentage := (area_error / actual_area) * 100
  error_percentage = 12.36 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l3156_315659


namespace NUMINAMATH_CALUDE_max_value_inequality_equality_case_l3156_315606

theorem max_value_inequality (a : ℝ) : (∀ x > 1, x + 1 / (x - 1) ≥ a) → a ≤ 3 := by sorry

theorem equality_case : ∃ x > 1, x + 1 / (x - 1) = 3 := by sorry

end NUMINAMATH_CALUDE_max_value_inequality_equality_case_l3156_315606


namespace NUMINAMATH_CALUDE_rational_equation_result_l3156_315679

theorem rational_equation_result (x y : ℚ) 
  (h : |2*x - 3*y + 1| + (x + 3*y + 5)^2 = 0) : 
  (-2*x*y)^2 * (-y^2) * 6*x*y^2 = 192 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_result_l3156_315679


namespace NUMINAMATH_CALUDE_regular_survey_rate_l3156_315621

/-- Proves that the regular rate for completing a survey is Rs. 30 given the specified conditions. -/
theorem regular_survey_rate
  (total_surveys : ℕ)
  (cellphone_rate_factor : ℚ)
  (cellphone_surveys : ℕ)
  (total_earnings : ℕ)
  (h1 : total_surveys = 100)
  (h2 : cellphone_rate_factor = 1.20)
  (h3 : cellphone_surveys = 50)
  (h4 : total_earnings = 3300) :
  ∃ (regular_rate : ℚ),
    regular_rate = 30 ∧
    regular_rate * (total_surveys - cellphone_surveys : ℚ) +
    (regular_rate * cellphone_rate_factor) * cellphone_surveys = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_regular_survey_rate_l3156_315621


namespace NUMINAMATH_CALUDE_complex_sum_equality_l3156_315669

/-- Given complex numbers B, Q, R, and T, prove that their sum is equal to 1 + 9i -/
theorem complex_sum_equality (B Q R T : ℂ) 
  (hB : B = 3 + 2*I)
  (hQ : Q = -5)
  (hR : R = 2*I)
  (hT : T = 3 + 5*I) :
  B - Q + R + T = 1 + 9*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l3156_315669


namespace NUMINAMATH_CALUDE_unique_modular_residue_l3156_315664

theorem unique_modular_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ -150 ≡ n [ZMOD 17] := by sorry

end NUMINAMATH_CALUDE_unique_modular_residue_l3156_315664


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3156_315692

theorem max_value_trig_expression (x : ℝ) : 11 - 8 * Real.cos x - 2 * (Real.sin x)^2 ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3156_315692


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l3156_315663

theorem cos_seven_pi_sixths : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l3156_315663


namespace NUMINAMATH_CALUDE_quadratic_root_implication_l3156_315633

theorem quadratic_root_implication (a b : ℝ) : 
  (∃ x : ℝ, a * x^2 + b * x + 6 = 0 ∧ x = -2) → 
  6 * a - 3 * b + 6 = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implication_l3156_315633


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3156_315626

/-- Given a geometric sequence with first term a₁ and common ratio q,
    if S₁, S₃, and 2a₃ form an arithmetic sequence, then q = -1/2 -/
theorem geometric_sequence_common_ratio
  (a₁ : ℝ) (q : ℝ) (S₁ S₃ : ℝ)
  (h₁ : S₁ = a₁)
  (h₂ : S₃ = a₁ + a₁ * q + a₁ * q^2)
  (h₃ : 2 * S₃ = S₁ + 2 * a₁ * q^2)
  (h₄ : a₁ ≠ 0) :
  q = -1/2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3156_315626


namespace NUMINAMATH_CALUDE_ab_is_zero_l3156_315628

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Given complex equation -/
def complex_equation (a b : ℝ) : Prop :=
  (1 + i) / (1 - i) = (a : ℂ) + b * i

/-- Theorem stating that if the complex equation holds, then ab = 0 -/
theorem ab_is_zero (a b : ℝ) (h : complex_equation a b) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_is_zero_l3156_315628


namespace NUMINAMATH_CALUDE_solution_exists_iff_a_in_interval_l3156_315636

/-- The system of equations has a solution within the specified square if and only if
    a is in the given interval for some integer k. -/
theorem solution_exists_iff_a_in_interval :
  ∀ (a : ℝ), ∃ (x y : ℝ),
    (x * Real.sin a - y * Real.cos a = 2 * Real.sin a - Real.cos a) ∧
    (x - 3 * y + 13 = 0) ∧
    (5 ≤ x ∧ x ≤ 9) ∧
    (3 ≤ y ∧ y ≤ 7)
  ↔
    ∃ (k : ℤ), π/4 + k * π ≤ a ∧ a ≤ Real.arctan (5/3) + k * π :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_iff_a_in_interval_l3156_315636


namespace NUMINAMATH_CALUDE_fraction_simplification_l3156_315662

theorem fraction_simplification (x : ℝ) : (3*x - 4)/4 + (5 - 2*x)/3 = (x + 8)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3156_315662


namespace NUMINAMATH_CALUDE_line_up_count_distribution_count_l3156_315696

/-- Represents a student --/
inductive Student : Type
| A
| B
| C
| D
| E

/-- Represents a line-up of students --/
def LineUp := List Student

/-- Represents a distribution of students into classes --/
def Distribution := List (List Student)

/-- Checks if two students are adjacent in a line-up --/
def areAdjacent (s1 s2 : Student) (lineup : LineUp) : Prop := sorry

/-- Checks if a distribution is valid (three non-empty classes) --/
def isValidDistribution (d : Distribution) : Prop := sorry

/-- Counts the number of valid line-ups --/
def countValidLineUps : Nat := sorry

/-- Counts the number of valid distributions --/
def countValidDistributions : Nat := sorry

theorem line_up_count :
  countValidLineUps = 12 := by sorry

theorem distribution_count :
  countValidDistributions = 150 := by sorry

end NUMINAMATH_CALUDE_line_up_count_distribution_count_l3156_315696


namespace NUMINAMATH_CALUDE_tan_to_sin_cos_ratio_l3156_315685

theorem tan_to_sin_cos_ratio (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_to_sin_cos_ratio_l3156_315685


namespace NUMINAMATH_CALUDE_min_value_of_f_l3156_315642

open Real

noncomputable def f (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / 
  (4 - x^2 - 10 * x * y - 25 * y^2)^(7/2)

theorem min_value_of_f :
  ∃ (min : ℝ), min = 5/32 ∧ ∀ (x y : ℝ), f x y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3156_315642


namespace NUMINAMATH_CALUDE_album_slots_equal_sum_of_photos_l3156_315658

/-- The number of photos brought by each person --/
def cristina_photos : ℕ := 7
def john_photos : ℕ := 10
def sarah_photos : ℕ := 9
def clarissa_photos : ℕ := 14

/-- The total number of slots in the photo album --/
def album_slots : ℕ := cristina_photos + john_photos + sarah_photos + clarissa_photos

/-- Theorem stating that the number of slots in the photo album
    is equal to the sum of photos brought by all four people --/
theorem album_slots_equal_sum_of_photos :
  album_slots = cristina_photos + john_photos + sarah_photos + clarissa_photos :=
by sorry

end NUMINAMATH_CALUDE_album_slots_equal_sum_of_photos_l3156_315658


namespace NUMINAMATH_CALUDE_competition_theorem_l3156_315674

/-- Represents a team in the competition -/
inductive Team
| A
| B
| E

/-- Represents an event in the competition -/
inductive Event
| Vaulting
| GrenadeThrowingv
| Other1
| Other2
| Other3

/-- Represents a place in an event -/
inductive Place
| First
| Second
| Third

/-- The scoring system for the competition -/
structure ScoringSystem where
  first : ℕ
  second : ℕ
  third : ℕ
  first_gt_second : first > second
  second_gt_third : second > third
  third_pos : third > 0

/-- The result of a single event -/
structure EventResult where
  event : Event
  first : Team
  second : Team
  third : Team

/-- The final scores of the teams -/
structure FinalScores where
  team_A : ℕ
  team_B : ℕ
  team_E : ℕ

/-- The competition results -/
structure CompetitionResults where
  scoring : ScoringSystem
  events : List EventResult
  scores : FinalScores

/-- The main theorem to prove -/
theorem competition_theorem (r : CompetitionResults) : 
  r.scores.team_A = 22 ∧ 
  r.scores.team_B = 9 ∧ 
  r.scores.team_E = 9 ∧
  (∃ e : EventResult, e ∈ r.events ∧ e.event = Event.Vaulting ∧ e.first = Team.B) →
  r.events.length = 5 ∧
  (∃ e : EventResult, e ∈ r.events ∧ e.event = Event.GrenadeThrowingv ∧ e.second = Team.B) :=
by sorry

end NUMINAMATH_CALUDE_competition_theorem_l3156_315674


namespace NUMINAMATH_CALUDE_range_of_f_l3156_315612

def f (x : ℤ) : ℤ := x^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3156_315612


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_hundred_is_one_fourth_more_than_fiftysix_l3156_315609

theorem thirty_percent_less_than_hundred_is_one_fourth_more_than_fiftysix : ∃ x : ℝ, 
  (100 - 0.3 * 100 = x + 0.25 * x) ∧ x = 56 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_hundred_is_one_fourth_more_than_fiftysix_l3156_315609


namespace NUMINAMATH_CALUDE_E_equals_three_iff_x_equals_y_infinite_solutions_exist_l3156_315624

def E (x y : ℕ) : ℚ :=
  x / y + (x + 1) / (y + 1) + (x + 2) / (y + 2)

theorem E_equals_three_iff_x_equals_y (x y : ℕ) :
  E x y = 3 ↔ x = y :=
sorry

theorem infinite_solutions_exist (k : ℕ) :
  ∃ x y : ℕ, E x y = 11 * k + 3 :=
sorry

end NUMINAMATH_CALUDE_E_equals_three_iff_x_equals_y_infinite_solutions_exist_l3156_315624


namespace NUMINAMATH_CALUDE_lines_equal_angles_with_plane_l3156_315640

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a line in 3D space
def Line : Type := ℝ → ℝ × ℝ × ℝ

-- Define the angle between a line and a plane
def angle_line_plane (l : Line) (p : Plane) : ℝ := sorry

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define intersecting lines
def intersecting (l1 l2 : Line) : Prop := sorry

-- Define skew lines
def skew (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem lines_equal_angles_with_plane (l1 l2 : Line) (p : Plane) 
  (h_distinct : l1 ≠ l2) 
  (h_equal_angles : angle_line_plane l1 p = angle_line_plane l2 p) :
  parallel l1 l2 ∨ intersecting l1 l2 ∨ skew l1 l2 := by sorry

end NUMINAMATH_CALUDE_lines_equal_angles_with_plane_l3156_315640


namespace NUMINAMATH_CALUDE_non_negative_integer_solutions_l3156_315647

def is_solution (x y : ℕ) : Prop := 2 * x + y = 5

theorem non_negative_integer_solutions :
  {p : ℕ × ℕ | is_solution p.1 p.2} = {(0, 5), (1, 3), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_non_negative_integer_solutions_l3156_315647


namespace NUMINAMATH_CALUDE_becky_necklaces_l3156_315688

theorem becky_necklaces (initial : ℕ) : 
  initial - 3 + 5 - 15 = 37 → initial = 50 := by
  sorry

#check becky_necklaces

end NUMINAMATH_CALUDE_becky_necklaces_l3156_315688


namespace NUMINAMATH_CALUDE_choir_arrangement_min_choir_members_l3156_315650

theorem choir_arrangement (n : ℕ) : 
  (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 990 :=
by sorry

theorem min_choir_members : 
  ∃ (n : ℕ), n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n = 990 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_min_choir_members_l3156_315650


namespace NUMINAMATH_CALUDE_modified_chessboard_no_tiling_l3156_315607

/-- Represents a chessboard cell --/
inductive Cell
| White
| Black

/-- Represents a 2x1 tile --/
structure Tile :=
  (first : Cell)
  (second : Cell)

/-- Represents the modified chessboard --/
def ModifiedChessboard : Type :=
  Fin 8 → Fin 8 → Option Cell

/-- A valid 2x1 tile covers one white and one black cell --/
def isValidTile (t : Tile) : Prop :=
  (t.first = Cell.White ∧ t.second = Cell.Black) ∨
  (t.first = Cell.Black ∧ t.second = Cell.White)

/-- A tiling of the modified chessboard --/
def Tiling : Type :=
  List Tile

/-- Checks if a tiling is valid for the modified chessboard --/
def isValidTiling (t : Tiling) (mb : ModifiedChessboard) : Prop :=
  sorry

theorem modified_chessboard_no_tiling :
  ∀ (mb : ModifiedChessboard),
    (mb 0 0 = none) →  -- Bottom-left square removed
    (mb 7 7 = none) →  -- Top-right square removed
    (∀ i j, i ≠ 0 ∨ j ≠ 0 → i ≠ 7 ∨ j ≠ 7 → mb i j ≠ none) →  -- All other squares present
    (∀ i j, (i + j) % 2 = 0 → mb i j = some Cell.White) →  -- White cells
    (∀ i j, (i + j) % 2 = 1 → mb i j = some Cell.Black) →  -- Black cells
    ¬∃ (t : Tiling), isValidTiling t mb :=
by
  sorry

end NUMINAMATH_CALUDE_modified_chessboard_no_tiling_l3156_315607


namespace NUMINAMATH_CALUDE_marys_double_counted_sheep_l3156_315675

/-- Given Mary's animal counting problem, prove that she double-counted 7 sheep. -/
theorem marys_double_counted_sheep :
  let marys_count : ℕ := 60
  let actual_animals : ℕ := 56
  let forgotten_pigs : ℕ := 3
  let double_counted_sheep : ℕ := marys_count - actual_animals + forgotten_pigs
  double_counted_sheep = 7 := by sorry

end NUMINAMATH_CALUDE_marys_double_counted_sheep_l3156_315675


namespace NUMINAMATH_CALUDE_car_fuel_consumption_l3156_315643

/-- Represents the distance a car can travel with a given amount of fuel -/
def distance_traveled (fuel_fraction : ℚ) (distance : ℚ) : ℚ := distance / fuel_fraction

/-- Represents the remaining distance a car can travel -/
def remaining_distance (total_distance : ℚ) (traveled_distance : ℚ) : ℚ :=
  total_distance - traveled_distance

theorem car_fuel_consumption 
  (initial_distance : ℚ) 
  (initial_fuel_fraction : ℚ) 
  (h1 : initial_distance = 165) 
  (h2 : initial_fuel_fraction = 3/8) : 
  remaining_distance (distance_traveled 1 initial_fuel_fraction) initial_distance = 275 := by
  sorry

#eval remaining_distance (distance_traveled 1 (3/8)) 165

end NUMINAMATH_CALUDE_car_fuel_consumption_l3156_315643


namespace NUMINAMATH_CALUDE_train_distance_problem_l3156_315661

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 16) (h2 : v2 = 21) (h3 : d = 60) :
  let t := d / (v2 - v1)
  let d1 := v1 * t
  let d2 := v2 * t
  d1 + d2 = 444 := by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l3156_315661


namespace NUMINAMATH_CALUDE_hyperbola_equation_correct_l3156_315623

/-- A hyperbola is defined by its equation, asymptotes, and a point it passes through. -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x²/a²) - (y²/b²) = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The hyperbola satisfies its equation at the given point -/
def satisfies_equation (h : Hyperbola) : Prop :=
  h.equation h.point.1 h.point.2

/-- The asymptotes of the hyperbola have the correct slope -/
def has_correct_asymptotes (h : Hyperbola) : Prop :=
  h.asymptote_slope = 1 / 2

/-- Theorem stating that the given hyperbola equation is correct -/
theorem hyperbola_equation_correct (h : Hyperbola)
  (heq : h.equation = fun x y => x^2 / 8 - y^2 / 2 = 1)
  (hpoint : h.point = (4, Real.sqrt 2))
  (hslope : h.asymptote_slope = 1 / 2)
  : satisfies_equation h ∧ has_correct_asymptotes h := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_equation_correct_l3156_315623


namespace NUMINAMATH_CALUDE_seven_digit_multiple_of_each_l3156_315656

/-- A function that returns the set of digits of a positive integer -/
def digits (n : ℕ+) : Finset ℕ :=
  sorry

/-- The theorem statement -/
theorem seven_digit_multiple_of_each : ∃ (n : ℕ+),
  (digits n).card = 7 ∧
  ∀ d ∈ digits n, d > 0 ∧ n % d = 0 →
  digits n = {1, 2, 3, 6, 7, 8, 9} :=
sorry

end NUMINAMATH_CALUDE_seven_digit_multiple_of_each_l3156_315656


namespace NUMINAMATH_CALUDE_slurpee_change_l3156_315604

/-- Calculates the change received when buying Slurpees -/
theorem slurpee_change (money_given : ℕ) (slurpee_cost : ℕ) (slurpees_bought : ℕ) : 
  money_given = 20 → slurpee_cost = 2 → slurpees_bought = 6 →
  money_given - (slurpee_cost * slurpees_bought) = 8 := by
  sorry

end NUMINAMATH_CALUDE_slurpee_change_l3156_315604


namespace NUMINAMATH_CALUDE_marble_probability_l3156_315648

/-- The probability of drawing either a green or purple marble from a bag -/
theorem marble_probability (green purple orange : ℕ) 
  (h_green : green = 5)
  (h_purple : purple = 4)
  (h_orange : orange = 6) :
  (green + purple : ℚ) / (green + purple + orange) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3156_315648


namespace NUMINAMATH_CALUDE_machine_worked_three_minutes_l3156_315695

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  shirts_per_minute : ℕ
  shirts_made_yesterday : ℕ

/-- The number of minutes the machine worked yesterday -/
def minutes_worked_yesterday (machine : ShirtMachine) : ℕ :=
  machine.shirts_made_yesterday / machine.shirts_per_minute

/-- Theorem stating that the machine worked for 3 minutes yesterday -/
theorem machine_worked_three_minutes (machine : ShirtMachine) 
    (h1 : machine.shirts_per_minute = 3)
    (h2 : machine.shirts_made_yesterday = 9) : 
  minutes_worked_yesterday machine = 3 := by
  sorry

end NUMINAMATH_CALUDE_machine_worked_three_minutes_l3156_315695


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3156_315632

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 1300000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.3
    exponent := 6
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3156_315632


namespace NUMINAMATH_CALUDE_reverse_increase_l3156_315639

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d3 * 100 + d2 * 10 + d1

theorem reverse_increase (n : ℕ) : 
  n = 253 → 
  (n / 100 + (n / 10) % 10 + n % 10 = 10) → 
  ((n / 10) % 10 = (n / 100 + n % 10)) → 
  reverse_number n - n = 99 :=
by sorry

end NUMINAMATH_CALUDE_reverse_increase_l3156_315639


namespace NUMINAMATH_CALUDE_smallest_longer_leg_length_l3156_315613

/-- Represents a 30-60-90 triangle --/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hypotenuse_eq : hypotenuse = 2 * shorterLeg
  longerLeg_eq : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of three connected 30-60-90 triangles --/
structure TriangleSequence where
  largest : Triangle30_60_90
  middle : Triangle30_60_90
  smallest : Triangle30_60_90
  connection1 : largest.longerLeg = middle.hypotenuse
  connection2 : middle.longerLeg = smallest.hypotenuse
  largest_hypotenuse : largest.hypotenuse = 12
  middle_special : middle.hypotenuse = middle.longerLeg

theorem smallest_longer_leg_length (seq : TriangleSequence) : 
  seq.smallest.longerLeg = 4.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_longer_leg_length_l3156_315613


namespace NUMINAMATH_CALUDE_abc_problem_l3156_315691

theorem abc_problem (a b c : ℝ) 
  (h1 : 2011 * (a + b + c) = 1) 
  (h2 : a * b + a * c + b * c = 2011 * a * b * c) : 
  a^2011 * b^2011 + c^2011 = (1 : ℝ) / 2011^2011 := by
  sorry

end NUMINAMATH_CALUDE_abc_problem_l3156_315691


namespace NUMINAMATH_CALUDE_point_coordinates_sum_of_coordinates_l3156_315651

/-- Given three points X, Y, and Z in the plane satisfying certain ratios,
    prove that X has specific coordinates. -/
theorem point_coordinates (X Y Z : ℝ × ℝ) : 
  Y = (2, 3) →
  Z = (5, 1) →
  (dist X Z) / (dist X Y) = 1/3 →
  (dist Z Y) / (dist X Y) = 2/3 →
  X = (6.5, 0) :=
by sorry

/-- The sum of coordinates of point X -/
def sum_coordinates (X : ℝ × ℝ) : ℝ :=
  X.1 + X.2

/-- Prove that the sum of coordinates of X is 6.5 -/
theorem sum_of_coordinates (X : ℝ × ℝ) :
  X = (6.5, 0) →
  sum_coordinates X = 6.5 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_of_coordinates_l3156_315651


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_one_unique_a_for_inequality_l3156_315678

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x

-- Part I
theorem tangent_line_at_negative_one (h : f 1 (-1) = 0) :
  ∃ (m : ℝ), ∀ (x : ℝ), f 1 x = m * (x + 1) + f 1 (-1) ∧ m = -2 :=
sorry

-- Part II
theorem unique_a_for_inequality (h : ∀ x, x ∈ Set.Icc 0 1 → (1/4 * x - 1/4 ≤ f 1 x ∧ f 1 x ≤ 1/4 * x + 1/4)) :
  ∀ a > 0, (∀ x, x ∈ Set.Icc 0 1 → (1/4 * x - 1/4 ≤ f a x ∧ f a x ≤ 1/4 * x + 1/4)) ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_negative_one_unique_a_for_inequality_l3156_315678


namespace NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l3156_315620

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem twentieth_term_of_specific_sequence :
  arithmetic_sequence 2 3 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l3156_315620


namespace NUMINAMATH_CALUDE_probability_no_shaded_square_correct_l3156_315687

/-- The probability of a randomly chosen rectangle not including a shaded square
    in a 2 by 2001 rectangle with the middle unit square of each row shaded. -/
def probability_no_shaded_square : ℚ :=
  1001 / 2001

/-- The number of columns in the rectangle. -/
def num_columns : ℕ := 2001

/-- The number of rows in the rectangle. -/
def num_rows : ℕ := 2

/-- The total number of rectangles that can be formed in a single row. -/
def rectangles_per_row : ℕ := (num_columns + 1).choose 2

/-- The number of rectangles in a single row that include the shaded square. -/
def shaded_rectangles_per_row : ℕ := (num_columns + 1) / 2 * (num_columns / 2)

theorem probability_no_shaded_square_correct :
  probability_no_shaded_square = 1 - (3 * shaded_rectangles_per_row) / (3 * rectangles_per_row) :=
sorry

end NUMINAMATH_CALUDE_probability_no_shaded_square_correct_l3156_315687


namespace NUMINAMATH_CALUDE_perfect_square_pairs_l3156_315603

theorem perfect_square_pairs (m n : ℤ) :
  (∃ a : ℤ, m^2 + n = a^2) ∧ (∃ b : ℤ, n^2 + m = b^2) →
  (m = 0 ∧ ∃ k : ℤ, n = k^2) ∨
  (n = 0 ∧ ∃ k : ℤ, m = k^2) ∨
  (m = 1 ∧ n = -1) ∨
  (m = -1 ∧ n = 1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_pairs_l3156_315603


namespace NUMINAMATH_CALUDE_johns_run_l3156_315667

/-- Theorem: John's total distance traveled is 5 miles -/
theorem johns_run (solo_speed : ℝ) (dog_speed : ℝ) (total_time : ℝ) (dog_time : ℝ) :
  solo_speed = 4 →
  dog_speed = 6 →
  total_time = 1 →
  dog_time = 0.5 →
  dog_speed * dog_time + solo_speed * (total_time - dog_time) = 5 := by
  sorry

#check johns_run

end NUMINAMATH_CALUDE_johns_run_l3156_315667


namespace NUMINAMATH_CALUDE_alice_prob_three_turns_correct_l3156_315615

/-- Represents the player who has the ball -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- The probability of keeping the ball for each player -/
def keep_prob (p : Player) : ℚ :=
  match p with
  | Player.Alice => 2/3
  | Player.Bob => 3/4

/-- The probability of tossing the ball for each player -/
def toss_prob (p : Player) : ℚ :=
  match p with
  | Player.Alice => 1/3
  | Player.Bob => 1/4

/-- The probability of Alice having the ball after three turns -/
def alice_prob_after_three_turns : ℚ := 203/432

theorem alice_prob_three_turns_correct :
  alice_prob_after_three_turns =
    keep_prob Player.Alice * keep_prob Player.Alice * keep_prob Player.Alice +
    toss_prob Player.Alice * toss_prob Player.Bob * keep_prob Player.Alice +
    keep_prob Player.Alice * toss_prob Player.Alice * toss_prob Player.Bob +
    toss_prob Player.Alice * keep_prob Player.Bob * toss_prob Player.Bob :=
by sorry

end NUMINAMATH_CALUDE_alice_prob_three_turns_correct_l3156_315615


namespace NUMINAMATH_CALUDE_local_max_implies_a_gt_half_l3156_315646

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2*a - 1) * x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2*a*x + 2*a

theorem local_max_implies_a_gt_half (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  a > 1/2 :=
sorry

end NUMINAMATH_CALUDE_local_max_implies_a_gt_half_l3156_315646


namespace NUMINAMATH_CALUDE_min_rain_day4_overflow_l3156_315665

/-- Represents the rainstorm scenario -/
structure RainstormScenario where
  capacity : ℝ  -- capacity in feet
  drain_rate : ℝ  -- drain rate in inches per day
  day1_rain : ℝ  -- rain on day 1 in inches
  days : ℕ  -- number of days
  overflow_day : ℕ  -- day when overflow occurs

/-- Calculates the minimum amount of rain on the last day to cause overflow -/
def min_rain_to_overflow (scenario : RainstormScenario) : ℝ :=
  sorry

/-- Theorem stating the minimum amount of rain on day 4 to cause overflow -/
theorem min_rain_day4_overflow (scenario : RainstormScenario) 
  (h1 : scenario.capacity = 6)
  (h2 : scenario.drain_rate = 3)
  (h3 : scenario.day1_rain = 10)
  (h4 : scenario.days = 4)
  (h5 : scenario.overflow_day = 4) :
  min_rain_to_overflow scenario = 4 :=
  sorry

end NUMINAMATH_CALUDE_min_rain_day4_overflow_l3156_315665


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l3156_315637

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_b_value :
  ∃ b : ℝ, collinear 4 (-6) (b + 3) 4 (3*b - 2) 3 ∧ b = 17/7 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l3156_315637


namespace NUMINAMATH_CALUDE_problem_solution_l3156_315653

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- Theorem to prove
theorem problem_solution :
  (p ∧ q) ∧
  ¬(p ∧ ¬q) ∧
  (¬p ∨ q) ∧
  ¬(¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3156_315653


namespace NUMINAMATH_CALUDE_range_and_minimum_value_l3156_315634

def f (a x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

theorem range_and_minimum_value (a : ℝ) :
  (a = 1 → Set.range (fun x => f 1 x) ∩ Set.Icc 0 2 = Set.Icc 0 9) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 2 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 2 → f a x ≤ f a y) →
  (∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ 3) →
  (a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_range_and_minimum_value_l3156_315634


namespace NUMINAMATH_CALUDE_interest_equivalence_l3156_315608

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The problem statement -/
theorem interest_equivalence (P : ℝ) : 
  simple_interest 100 0.05 8 = simple_interest P 0.10 2 → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_interest_equivalence_l3156_315608


namespace NUMINAMATH_CALUDE_ninth_ninety_ninth_digit_sum_l3156_315638

def decimal_expansion (n : ℕ) (d : ℕ) : ℚ := n / d

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem ninth_ninety_ninth_digit_sum (n : ℕ) : 
  nth_digit_after_decimal (decimal_expansion 2 9 + decimal_expansion 3 11 + decimal_expansion 5 13) 999 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ninth_ninety_ninth_digit_sum_l3156_315638


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3156_315600

theorem ratio_x_to_y (x y : ℝ) (h : y = x * (1 - 0.8333333333333334)) :
  x / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3156_315600


namespace NUMINAMATH_CALUDE_expression_evaluation_l3156_315684

theorem expression_evaluation :
  (2^2003 * 3^2002 * 5) / 6^2003 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3156_315684


namespace NUMINAMATH_CALUDE_community_center_chairs_l3156_315625

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : Nat) : Nat :=
  (n / 100) * 25 + ((n % 100) / 10) * 5 + (n % 10)

/-- Calculates the number of chairs needed given a capacity in base 5 and people per chair -/
def chairsNeeded (capacityBase5 : Nat) (peoplePerChair : Nat) : Nat :=
  (base5ToBase10 capacityBase5) / peoplePerChair

theorem community_center_chairs :
  chairsNeeded 310 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_community_center_chairs_l3156_315625


namespace NUMINAMATH_CALUDE_dagger_five_eighths_three_fourths_l3156_315611

-- Define the operation †
def dagger (m n p q : ℚ) : ℚ := 2 * m * p * (q / n)

-- Theorem statement
theorem dagger_five_eighths_three_fourths :
  dagger (5/8) (8/8) (3/4) (4/4) = 15 := by
  sorry

end NUMINAMATH_CALUDE_dagger_five_eighths_three_fourths_l3156_315611


namespace NUMINAMATH_CALUDE_enrollment_analysis_l3156_315629

def summit_ridge : ℕ := 1560
def pine_hills : ℕ := 1150
def oak_valley : ℕ := 1950
def maple_town : ℕ := 1840

def enrollments : List ℕ := [summit_ridge, pine_hills, oak_valley, maple_town]

theorem enrollment_analysis :
  (List.maximum enrollments).get! - (List.minimum enrollments).get! = 800 ∧
  (List.sum enrollments) / enrollments.length = 1625 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_analysis_l3156_315629


namespace NUMINAMATH_CALUDE_five_line_triangle_bounds_l3156_315602

/-- A line in a plane --/
structure Line where
  -- Add necessary fields here
  
/-- A region in a plane --/
structure Region where
  -- Add necessary fields here

/-- Represents a configuration of lines in a plane --/
structure PlaneConfiguration where
  lines : List Line
  regions : List Region

/-- Checks if lines are in general position --/
def is_general_position (config : PlaneConfiguration) : Prop :=
  sorry

/-- Counts the number of triangular regions --/
def count_triangles (config : PlaneConfiguration) : Nat :=
  sorry

/-- Main theorem about triangles in a plane divided by five lines --/
theorem five_line_triangle_bounds 
  (config : PlaneConfiguration) 
  (h1 : config.lines.length = 5)
  (h2 : config.regions.length = 16)
  (h3 : is_general_position config) :
  3 ≤ count_triangles config ∧ count_triangles config ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_five_line_triangle_bounds_l3156_315602


namespace NUMINAMATH_CALUDE_state_tax_deduction_l3156_315649

theorem state_tax_deduction (hourly_wage : ℝ) (tax_rate : ℝ) : 
  hourly_wage = 25 → tax_rate = 0.024 → hourly_wage * tax_rate * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_state_tax_deduction_l3156_315649


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3156_315672

theorem arithmetic_expression_equality : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 - (-9) = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3156_315672


namespace NUMINAMATH_CALUDE_proportion_solution_l3156_315666

theorem proportion_solution (x : ℝ) : 
  (1.25 / x = 15 / 26.5) → x = 33.125 / 15 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3156_315666


namespace NUMINAMATH_CALUDE_problem_solution_l3156_315699

noncomputable section

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem problem_solution :
  (∀ a : ℝ, A ∩ B a = A ∪ B a → a = 1) ∧
  (∀ a : ℝ, A ∩ B a = B a → a ≤ -1 ∨ a = 1) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l3156_315699


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_range_part2_l3156_315630

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 2 ≥ 3} = {x : ℝ | x ≤ -3/4 ∨ x ≥ 3/4} := by sorry

-- Part 2
theorem solution_range_part2 :
  ∀ a : ℝ, a > 0 → (∃ x : ℝ, f x a < a/2 + 1) ↔ a > 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_range_part2_l3156_315630


namespace NUMINAMATH_CALUDE_b_21_mod_12_l3156_315676

/-- Definition of b_n as the integer obtained by writing all integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that b_21 mod 12 = 9 -/
theorem b_21_mod_12 : b 21 % 12 = 9 := by sorry

end NUMINAMATH_CALUDE_b_21_mod_12_l3156_315676


namespace NUMINAMATH_CALUDE_principal_is_300_l3156_315619

/-- Given a principal amount P and an interest rate R, 
    if increasing the rate by 6% for 5 years results in 90 more interest,
    then P must be 300. -/
theorem principal_is_300 (P R : ℝ) : 
  (P * (R + 6) * 5) / 100 = (P * R * 5) / 100 + 90 → P = 300 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_300_l3156_315619


namespace NUMINAMATH_CALUDE_hexagon_area_division_l3156_315644

/-- A hexagon constructed from unit squares -/
structure Hexagon :=
  (area : ℝ)
  (line_PQ : ℝ → ℝ)
  (area_below : ℝ)
  (area_above : ℝ)
  (XQ : ℝ)
  (QY : ℝ)

/-- The theorem statement -/
theorem hexagon_area_division (h : Hexagon) :
  h.area = 8 ∧
  h.area_below = h.area_above ∧
  h.area_below = 1 + (1/2 * 4 * (3/2)) ∧
  h.XQ + h.QY = 4 →
  h.XQ / h.QY = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_area_division_l3156_315644


namespace NUMINAMATH_CALUDE_common_factor_proof_l3156_315680

theorem common_factor_proof (x y : ℝ) : ∃ (k : ℝ), 5*x^2 - 25*x^2*y = 5*x^2 * k :=
sorry

end NUMINAMATH_CALUDE_common_factor_proof_l3156_315680


namespace NUMINAMATH_CALUDE_legos_set_cost_l3156_315654

def total_earnings : ℕ := 45
def car_price : ℕ := 5
def num_cars : ℕ := 3

theorem legos_set_cost : total_earnings - (car_price * num_cars) = 30 := by
  sorry

end NUMINAMATH_CALUDE_legos_set_cost_l3156_315654


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3156_315683

/-- Represents the systematic sampling problem -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  sampling_interval : ℕ
  first_random_number : ℕ

/-- Calculates the number of selected students within a given range -/
def selected_students_in_range (s : SystematicSampling) (lower : ℕ) (upper : ℕ) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.total_students = 1000)
  (h2 : s.sample_size = 50)
  (h3 : s.sampling_interval = 20)
  (h4 : s.first_random_number = 15) :
  selected_students_in_range s 601 785 = 9 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3156_315683


namespace NUMINAMATH_CALUDE_bagel_savings_theorem_l3156_315601

/-- The cost of a single bagel in dollars -/
def single_bagel_cost : ℚ := 2.25

/-- The cost of a dozen bagels in dollars -/
def dozen_bagels_cost : ℚ := 24

/-- The number of bagels in a dozen -/
def dozen : ℕ := 12

/-- The savings per bagel in cents when buying a dozen -/
def savings_per_bagel : ℚ :=
  ((single_bagel_cost * dozen - dozen_bagels_cost) / dozen) * 100

theorem bagel_savings_theorem :
  savings_per_bagel = 25 := by sorry

end NUMINAMATH_CALUDE_bagel_savings_theorem_l3156_315601


namespace NUMINAMATH_CALUDE_log_inequality_solution_set_l3156_315668

-- Define the logarithm function with base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the solution set
def solution_set : Set ℝ := { x | x > -1 ∧ x ≤ 0 }

-- Theorem statement
theorem log_inequality_solution_set :
  { x : ℝ | x > -1 ∧ log10 (x + 1) ≤ 0 } = solution_set :=
sorry

end NUMINAMATH_CALUDE_log_inequality_solution_set_l3156_315668


namespace NUMINAMATH_CALUDE_other_workers_count_l3156_315682

def total_workers : ℕ := 5
def chosen_workers : ℕ := 2
def probability_jack_and_jill : ℚ := 1/10

theorem other_workers_count :
  let other_workers := total_workers - 2
  probability_jack_and_jill = 1 / (total_workers.choose chosen_workers) →
  other_workers = 3 := by
sorry

end NUMINAMATH_CALUDE_other_workers_count_l3156_315682


namespace NUMINAMATH_CALUDE_galaxy_distance_in_miles_l3156_315681

/-- The number of miles in one light-year -/
def miles_per_light_year : ℝ := 6 * 10^12

/-- The distance to the observed galaxy in thousand million light-years -/
def galaxy_distance_thousand_million_light_years : ℝ := 13.4

/-- Conversion factor from thousand million to billion -/
def thousand_million_to_billion : ℝ := 1

theorem galaxy_distance_in_miles :
  let distance_light_years := galaxy_distance_thousand_million_light_years * thousand_million_to_billion * 10^9
  let distance_miles := distance_light_years * miles_per_light_year
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |distance_miles - 8 * 10^22| < ε * (8 * 10^22) :=
sorry

end NUMINAMATH_CALUDE_galaxy_distance_in_miles_l3156_315681


namespace NUMINAMATH_CALUDE_subset_implies_m_range_l3156_315616

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_implies_m_range (m : ℝ) : B m ⊆ A → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_range_l3156_315616


namespace NUMINAMATH_CALUDE_intersection_complement_M_and_N_l3156_315671

def U : Set Int := {-2, -1, 0, 1, 2}

def M : Set Int := {y | ∃ x, y = 2^x}

def N : Set Int := {x | x^2 - x - 2 = 0}

theorem intersection_complement_M_and_N :
  (U \ M) ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_M_and_N_l3156_315671


namespace NUMINAMATH_CALUDE_sarah_and_matt_age_sum_l3156_315673

/-- Given the age relationship between Sarah and Matt, prove that the sum of their current ages is 41 years. -/
theorem sarah_and_matt_age_sum :
  ∀ (sarah_age matt_age : ℝ),
  sarah_age = matt_age + 8 →
  sarah_age + 10 = 3 * (matt_age - 5) →
  sarah_age + matt_age = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_sarah_and_matt_age_sum_l3156_315673


namespace NUMINAMATH_CALUDE_problem_statement_l3156_315693

def f (x : ℝ) := |2*x - 1|

theorem problem_statement (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : f a > f c) (h4 : f c > f b) : 
  2 - a < 2*c := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3156_315693


namespace NUMINAMATH_CALUDE_exists_determining_question_l3156_315655

-- Define the types of guests
inductive GuestType
| Human
| Vampire

-- Define the possible answers
inductive Answer
| Bal
| Da

-- Define a question as a function that takes a GuestType and returns an Answer
def Question := GuestType → Answer

-- Define a function to determine the guest type based on the answer
def determineGuestType (q : Question) (a : Answer) : GuestType := 
  match a with
  | Answer.Bal => GuestType.Human
  | Answer.Da => GuestType.Vampire

-- Theorem statement
theorem exists_determining_question : 
  ∃ (q : Question), 
    (∀ (g : GuestType), (determineGuestType q (q g)) = g) :=
sorry

end NUMINAMATH_CALUDE_exists_determining_question_l3156_315655


namespace NUMINAMATH_CALUDE_cost_of_pens_l3156_315618

theorem cost_of_pens (box_size : ℕ) (box_cost : ℚ) (total_pens : ℕ) : 
  box_size = 150 → box_cost = 45 → total_pens = 4500 → 
  (total_pens : ℚ) * (box_cost / box_size) = 1350 := by
sorry

end NUMINAMATH_CALUDE_cost_of_pens_l3156_315618


namespace NUMINAMATH_CALUDE_billy_experiment_result_l3156_315677

/-- Represents the mouse population dynamics in Billy's experiment --/
structure MousePopulation where
  initial_mice : ℕ
  pups_per_mouse : ℕ
  final_population : ℕ

/-- Calculates the number of pups eaten per adult mouse --/
def pups_eaten_per_adult (pop : MousePopulation) : ℕ :=
  let first_gen_total := pop.initial_mice + pop.initial_mice * pop.pups_per_mouse
  let second_gen_total := first_gen_total + first_gen_total * pop.pups_per_mouse
  let total_eaten := second_gen_total - pop.final_population
  total_eaten / first_gen_total

/-- Theorem stating that in Billy's experiment, each adult mouse ate 2 pups --/
theorem billy_experiment_result :
  let pop : MousePopulation := {
    initial_mice := 8,
    pups_per_mouse := 6,
    final_population := 280
  }
  pups_eaten_per_adult pop = 2 := by
  sorry


end NUMINAMATH_CALUDE_billy_experiment_result_l3156_315677


namespace NUMINAMATH_CALUDE_function_value_at_ln_half_l3156_315645

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (5^x) / (5^x + 1)

theorem function_value_at_ln_half (a : ℝ) :
  (f a (Real.log 2) = 4) → (f a (Real.log (1/2)) = -3) := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_ln_half_l3156_315645


namespace NUMINAMATH_CALUDE_alex_grocery_delivery_l3156_315610

/-- Alex's grocery delivery problem -/
theorem alex_grocery_delivery 
  (savings : ℝ) 
  (car_cost : ℝ) 
  (trip_charge : ℝ) 
  (grocery_percentage : ℝ) 
  (num_trips : ℕ) 
  (h1 : savings = 14500)
  (h2 : car_cost = 14600)
  (h3 : trip_charge = 1.5)
  (h4 : grocery_percentage = 0.05)
  (h5 : num_trips = 40)
  : ∃ (grocery_worth : ℝ), 
    trip_charge * num_trips + grocery_percentage * grocery_worth = car_cost - savings ∧ 
    grocery_worth = 800 := by
  sorry

end NUMINAMATH_CALUDE_alex_grocery_delivery_l3156_315610


namespace NUMINAMATH_CALUDE_wages_comparison_l3156_315698

theorem wages_comparison (E R C : ℝ) 
  (hC_E : C = E * 1.7)
  (hC_R : C = R * 1.3076923076923077) :
  R = E * 1.3 :=
by sorry

end NUMINAMATH_CALUDE_wages_comparison_l3156_315698


namespace NUMINAMATH_CALUDE_shadows_parallel_l3156_315697

-- Define a structure for a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a structure for a point on a projection plane
structure ProjectedPoint where
  x : ℝ
  y : ℝ

-- Define a structure for a light source (parallel lighting)
structure ParallelLight where
  direction : Point3D

-- Define a function to project a 3D point onto a plane
def project (p : Point3D) (plane : ℝ) (light : ParallelLight) : ProjectedPoint :=
  sorry

-- Define a function to check if two line segments are parallel
def areParallel (p1 p2 q1 q2 : ProjectedPoint) : Prop :=
  sorry

-- Theorem statement
theorem shadows_parallel 
  (A B C : Point3D) 
  (plane1 plane2 : ℝ) 
  (light : ParallelLight) :
  let A1 := project A plane1 light
  let A2 := project A plane2 light
  let B1 := project B plane1 light
  let B2 := project B plane2 light
  let C1 := project C plane1 light
  let C2 := project C plane2 light
  areParallel A1 A2 B1 B2 ∧ areParallel B1 B2 C1 C2 :=
sorry

end NUMINAMATH_CALUDE_shadows_parallel_l3156_315697


namespace NUMINAMATH_CALUDE_olivia_remaining_money_l3156_315627

/-- Given an initial amount of money and an amount spent, 
    calculate the remaining amount. -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that given an initial amount of 78 dollars 
    and a spent amount of 15 dollars, the remaining amount is 63 dollars. -/
theorem olivia_remaining_money :
  remaining_money 78 15 = 63 := by
  sorry

end NUMINAMATH_CALUDE_olivia_remaining_money_l3156_315627


namespace NUMINAMATH_CALUDE_aaron_gave_five_sweets_l3156_315694

/-- Represents the number of sweets given to a friend -/
def sweets_given_to_friend (initial_cherry initial_strawberry initial_pineapple : ℕ) 
  (remaining : ℕ) : ℕ :=
  initial_cherry / 2 + initial_strawberry / 2 + initial_pineapple / 2 - remaining

/-- Proves that Aaron gave 5 cherry sweets to his friend -/
theorem aaron_gave_five_sweets : 
  sweets_given_to_friend 30 40 50 55 = 5 := by
  sorry

#eval sweets_given_to_friend 30 40 50 55

end NUMINAMATH_CALUDE_aaron_gave_five_sweets_l3156_315694


namespace NUMINAMATH_CALUDE_probability_theorem_l3156_315652

-- Define the probabilities of events A1, A2, A3
def P_A1 : ℚ := 1/2
def P_A2 : ℚ := 1/5
def P_A3 : ℚ := 3/10

-- Define the conditional probabilities
def P_B_given_A1 : ℚ := 5/11
def P_B_given_A2 : ℚ := 4/11
def P_B_given_A3 : ℚ := 4/11

-- Define the probability of event B
def P_B : ℚ := 9/22

-- Define the theorem to be proved
theorem probability_theorem :
  (P_B_given_A1 = 5/11) ∧
  (P_B = 9/22) ∧
  (P_A1 + P_A2 + P_A3 = 1) := by
  sorry

#check probability_theorem

end NUMINAMATH_CALUDE_probability_theorem_l3156_315652


namespace NUMINAMATH_CALUDE_translation_result_l3156_315617

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translates a point vertically -/
def translateVertical (p : Point2D) (dy : ℝ) : Point2D :=
  { x := p.x, y := p.y + dy }

/-- Translates a point horizontally -/
def translateHorizontal (p : Point2D) (dx : ℝ) : Point2D :=
  { x := p.x - dx, y := p.y }

/-- The theorem to be proved -/
theorem translation_result :
  let initial_point : Point2D := { x := 3, y := -2 }
  let after_vertical := translateVertical initial_point 3
  let final_point := translateHorizontal after_vertical 2
  final_point = { x := 1, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l3156_315617


namespace NUMINAMATH_CALUDE_exercise_time_is_1910_l3156_315614

/-- The total exercise time for Javier, Sanda, Luis, and Nita -/
def total_exercise_time : ℕ :=
  let javier := 50 * 10
  let sanda := 90 * 3 + 75 * 2 + 45 * 4
  let luis := 60 * 5 + 30 * 3
  let nita := 100 * 2 + 55 * 4
  javier + sanda + luis + nita

/-- Theorem stating that the total exercise time is 1910 minutes -/
theorem exercise_time_is_1910 : total_exercise_time = 1910 := by
  sorry

end NUMINAMATH_CALUDE_exercise_time_is_1910_l3156_315614


namespace NUMINAMATH_CALUDE_equal_numbers_product_l3156_315635

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 17.6 →
  a = 15 →
  b = 20 →
  c = 22 →
  d = e →
  d * e = 240.25 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l3156_315635


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_100_l3156_315641

theorem closest_integer_to_cube_root_100 :
  ∃ n : ℤ, ∀ m : ℤ, |n ^ 3 - 100| ≤ |m ^ 3 - 100| ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_100_l3156_315641


namespace NUMINAMATH_CALUDE_total_vehicles_is_282_l3156_315660

/-- The number of vehicles Kendra saw during her road trip -/
def total_vehicles : ℕ :=
  let morning_minivans := 20
  let morning_sedans := 17
  let morning_suvs := 12
  let morning_trucks := 8
  let morning_motorcycles := 5

  let afternoon_minivans := 22
  let afternoon_sedans := 13
  let afternoon_suvs := 15
  let afternoon_trucks := 10
  let afternoon_motorcycles := 7

  let evening_minivans := 15
  let evening_sedans := 19
  let evening_suvs := 18
  let evening_trucks := 14
  let evening_motorcycles := 10

  let night_minivans := 10
  let night_sedans := 12
  let night_suvs := 20
  let night_trucks := 20
  let night_motorcycles := 15

  let total_minivans := morning_minivans + afternoon_minivans + evening_minivans + night_minivans
  let total_sedans := morning_sedans + afternoon_sedans + evening_sedans + night_sedans
  let total_suvs := morning_suvs + afternoon_suvs + evening_suvs + night_suvs
  let total_trucks := morning_trucks + afternoon_trucks + evening_trucks + night_trucks
  let total_motorcycles := morning_motorcycles + afternoon_motorcycles + evening_motorcycles + night_motorcycles

  total_minivans + total_sedans + total_suvs + total_trucks + total_motorcycles

theorem total_vehicles_is_282 : total_vehicles = 282 := by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_is_282_l3156_315660
