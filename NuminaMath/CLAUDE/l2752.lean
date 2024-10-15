import Mathlib

namespace NUMINAMATH_CALUDE_trajectory_of_P_l2752_275278

-- Define the line l
def line_l (θ : ℝ) (x y : ℝ) : Prop := x * Real.cos θ + y * Real.sin θ = 1

-- Define the perpendicularity condition
def perpendicular_to_l (x y : ℝ) : Prop := ∃ θ, line_l θ x y

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem trajectory_of_P : ∀ x y : ℝ, perpendicular_to_l x y → x^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l2752_275278


namespace NUMINAMATH_CALUDE_soccer_goal_ratio_l2752_275216

/-- Prove the ratio of goals scored by The Spiders to The Kickers in the first period -/
theorem soccer_goal_ratio :
  let kickers_first := 2
  let kickers_second := 2 * kickers_first
  let spiders_second := 2 * kickers_second
  let total_goals := 15
  let spiders_first := total_goals - (kickers_first + kickers_second + spiders_second)
  (spiders_first : ℚ) / kickers_first = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_soccer_goal_ratio_l2752_275216


namespace NUMINAMATH_CALUDE_valid_cube_assignment_exists_l2752_275248

/-- A cube is represented as a set of 8 vertices -/
def Cube := Fin 8

/-- An edge in the cube is a pair of vertices -/
def Edge := Prod Cube Cube

/-- The set of edges in a cube -/
def cubeEdges : Set Edge := sorry

/-- An assignment of natural numbers to the vertices of a cube -/
def Assignment := Cube → ℕ+

/-- Checks if one number divides another -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- The main theorem stating the existence of a valid assignment -/
theorem valid_cube_assignment_exists : ∃ (f : Assignment),
  (∀ (e : Edge), e ∈ cubeEdges →
    (divides (f e.1) (f e.2) ∨ divides (f e.2) (f e.1))) ∧
  (∀ (v w : Cube), (Prod.mk v w) ∉ cubeEdges →
    ¬(divides (f v) (f w)) ∧ ¬(divides (f w) (f v))) := by
  sorry

end NUMINAMATH_CALUDE_valid_cube_assignment_exists_l2752_275248


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l2752_275286

/-- Represents a 4x4x4 cube composed of unit cubes -/
structure Cube :=
  (size : Nat)
  (total_units : Nat)
  (painted_per_face : Nat)

/-- The number of unpainted unit cubes in the cube -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - (c.painted_per_face * 6)

/-- Theorem stating the number of unpainted cubes in the specific cube configuration -/
theorem unpainted_cubes_count (c : Cube) 
  (h1 : c.size = 4)
  (h2 : c.total_units = 64)
  (h3 : c.painted_per_face = 4) :
  unpainted_cubes c = 40 := by
  sorry


end NUMINAMATH_CALUDE_unpainted_cubes_count_l2752_275286


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l2752_275240

/-- Two points in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of parallel to x-axis for a line segment -/
def parallelToXAxis (p1 p2 : Point2D) : Prop :=
  p1.y = p2.y

theorem parallel_line_k_value (A B : Point2D) (k : ℝ) 
    (hA : A = ⟨2, 3⟩) 
    (hB : B = ⟨4, k⟩) 
    (hParallel : parallelToXAxis A B) : 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l2752_275240


namespace NUMINAMATH_CALUDE_second_day_average_speed_l2752_275296

/-- Represents the driving conditions and results over two days -/
structure DrivingData where
  total_distance : ℝ
  total_time : ℝ
  total_fuel : ℝ
  first_day_time_diff : ℝ
  first_day_speed_diff : ℝ
  first_day_efficiency : ℝ
  second_day_efficiency : ℝ

/-- Theorem stating that given the driving conditions, the average speed on the second day is 35 mph -/
theorem second_day_average_speed
  (data : DrivingData)
  (h1 : data.total_distance = 680)
  (h2 : data.total_time = 18)
  (h3 : data.total_fuel = 22.5)
  (h4 : data.first_day_time_diff = 2)
  (h5 : data.first_day_speed_diff = 5)
  (h6 : data.first_day_efficiency = 25)
  (h7 : data.second_day_efficiency = 30) :
  ∃ (second_day_speed : ℝ),
    second_day_speed = 35 ∧
    (second_day_speed + data.first_day_speed_diff) * (data.total_time / 2 + data.first_day_time_diff / 2) +
    second_day_speed * (data.total_time / 2 - data.first_day_time_diff / 2) = data.total_distance :=
by sorry

#check second_day_average_speed

end NUMINAMATH_CALUDE_second_day_average_speed_l2752_275296


namespace NUMINAMATH_CALUDE_inequality_proof_l2752_275230

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2752_275230


namespace NUMINAMATH_CALUDE_committee_problem_l2752_275205

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_problem :
  let total_students : ℕ := 10
  let committee_size : ℕ := 5
  let shared_members : ℕ := 3

  -- Number of different 5-student committees from 10 students
  (choose total_students committee_size = 252) ∧ 
  
  -- Number of ways to choose two 5-student committees with exactly 3 overlapping members
  ((choose total_students committee_size * 
    choose committee_size shared_members * 
    choose (total_students - committee_size) (committee_size - shared_members)) / 2 = 12600) :=
by sorry

end NUMINAMATH_CALUDE_committee_problem_l2752_275205


namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l2752_275288

theorem students_in_both_band_and_chorus 
  (total_students : ℕ) 
  (band_students : ℕ) 
  (chorus_students : ℕ) 
  (band_or_chorus_students : ℕ) : 
  total_students = 200 →
  band_students = 70 →
  chorus_students = 95 →
  band_or_chorus_students = 150 →
  band_students + chorus_students - band_or_chorus_students = 15 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l2752_275288


namespace NUMINAMATH_CALUDE_triangle_rotation_theorem_l2752_275239

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  O : Point
  P : Point
  Q : Point

/-- Rotates a point -90° around the origin -/
def rotate90Clockwise (p : Point) : Point :=
  { x := p.y, y := -p.x }

theorem triangle_rotation_theorem (t : Triangle) :
  t.O = { x := 0, y := 0 } →
  t.Q = { x := 3, y := 0 } →
  t.P.x > 0 →
  t.P.y > 0 →
  (t.P.y - t.O.y) / (t.P.x - t.O.x) = 1 →
  (t.Q.x - t.O.x) * (t.P.x - t.Q.x) + (t.Q.y - t.O.y) * (t.P.y - t.Q.y) = 0 →
  rotate90Clockwise t.P = { x := 3, y := -3 } := by
  sorry

#check triangle_rotation_theorem

end NUMINAMATH_CALUDE_triangle_rotation_theorem_l2752_275239


namespace NUMINAMATH_CALUDE_dedekind_cut_B_dedekind_cut_D_l2752_275289

-- Define a Dedekind cut
def DedekindCut (M N : Set ℚ) : Prop :=
  (M ∪ N = Set.univ) ∧ 
  (M ∩ N = ∅) ∧ 
  (∀ x ∈ M, ∀ y ∈ N, x < y) ∧
  M.Nonempty ∧ 
  N.Nonempty

-- Statement B
theorem dedekind_cut_B : 
  ∃ M N : Set ℚ, DedekindCut M N ∧ 
  (¬∃ x, x = Sup M) ∧ 
  (∃ y, y = Inf N) :=
sorry

-- Statement D
theorem dedekind_cut_D : 
  ∃ M N : Set ℚ, DedekindCut M N ∧ 
  (¬∃ x, x = Sup M) ∧ 
  (¬∃ y, y = Inf N) :=
sorry

end NUMINAMATH_CALUDE_dedekind_cut_B_dedekind_cut_D_l2752_275289


namespace NUMINAMATH_CALUDE_total_sheets_is_400_l2752_275281

/-- Calculates the total number of sheets of paper used for all students --/
def total_sheets (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) : ℕ :=
  num_classes * students_per_class * sheets_per_student

/-- Proves that the total number of sheets used is 400 --/
theorem total_sheets_is_400 :
  total_sheets 4 20 5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_is_400_l2752_275281


namespace NUMINAMATH_CALUDE_power_function_through_point_l2752_275232

/-- If f(x) = x^n is a power function and f(2) = √2, then f(4) = 2 -/
theorem power_function_through_point (n : ℝ) (f : ℝ → ℝ) : 
  (∀ x > 0, f x = x ^ n) →    -- f is a power function
  f 2 = Real.sqrt 2 →         -- f passes through (2, √2)
  f 4 = 2 := by               -- then f(4) = 2
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2752_275232


namespace NUMINAMATH_CALUDE_no_solutions_for_2500_l2752_275269

theorem no_solutions_for_2500 :
  ¬ ∃ (a₂ a₀ : ℤ), 2500 = a₂ * 10^4 + a₀ ∧ 0 ≤ a₂ ∧ a₂ ≤ 9 ∧ 0 ≤ a₀ ∧ a₀ ≤ 1000 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_2500_l2752_275269


namespace NUMINAMATH_CALUDE_purple_bows_count_l2752_275287

/-- Given a bag of bows with the following properties:
    - 1/4 of the bows are red
    - 1/3 of the bows are blue
    - 1/6 of the bows are purple
    - The remaining 60 bows are yellow
    This theorem proves that there are 40 purple bows. -/
theorem purple_bows_count (total : ℕ) (red blue purple yellow : ℕ) : 
  red + blue + purple + yellow = total →
  4 * red = total →
  3 * blue = total →
  6 * purple = total →
  yellow = 60 →
  purple = 40 := by
  sorry

#check purple_bows_count

end NUMINAMATH_CALUDE_purple_bows_count_l2752_275287


namespace NUMINAMATH_CALUDE_fair_haired_men_nonmanagerial_percentage_l2752_275263

/-- Represents the hair color distribution in the company -/
structure HairColorDistribution where
  fair : ℝ
  dark : ℝ
  red : ℝ
  ratio_fair_dark_red : fair / dark = 4 / 9 ∧ fair / red = 4 / 7

/-- Represents the gender distribution in the company -/
structure GenderDistribution where
  women : ℝ
  men : ℝ
  ratio_women_men : women / men = 3 / 5

/-- Represents the position distribution in the company -/
structure PositionDistribution where
  managerial : ℝ
  nonmanagerial : ℝ
  ratio_managerial_nonmanagerial : managerial / nonmanagerial = 1 / 4

/-- Represents the distribution of fair-haired employees -/
structure FairHairedDistribution where
  women_percentage : ℝ
  women_percentage_is_40 : women_percentage = 0.4
  women_managerial_percentage : ℝ
  women_managerial_percentage_is_60 : women_managerial_percentage = 0.6
  men_nonmanagerial_percentage : ℝ
  men_nonmanagerial_percentage_is_70 : men_nonmanagerial_percentage = 0.7

/-- Theorem: The percentage of fair-haired men in non-managerial positions is 42% -/
theorem fair_haired_men_nonmanagerial_percentage
  (hair : HairColorDistribution)
  (gender : GenderDistribution)
  (position : PositionDistribution)
  (fair_haired : FairHairedDistribution) :
  (1 - fair_haired.women_percentage) * fair_haired.men_nonmanagerial_percentage = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_fair_haired_men_nonmanagerial_percentage_l2752_275263


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_five_primes_l2752_275224

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem arithmetic_mean_reciprocals_first_five_primes :
  let reciprocals := first_five_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 2927 / 11550 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_five_primes_l2752_275224


namespace NUMINAMATH_CALUDE_chord_count_l2752_275221

/-- The number of points on the circumference of a circle -/
def n : ℕ := 7

/-- The number of points needed to form a chord -/
def r : ℕ := 2

/-- The number of different chords that can be drawn -/
def num_chords : ℕ := n.choose r

theorem chord_count : num_chords = 21 := by
  sorry

end NUMINAMATH_CALUDE_chord_count_l2752_275221


namespace NUMINAMATH_CALUDE_binomial_50_2_l2752_275267

theorem binomial_50_2 : Nat.choose 50 2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_binomial_50_2_l2752_275267


namespace NUMINAMATH_CALUDE_cube_opposite_face_l2752_275222

-- Define the faces of the cube
inductive Face : Type
| A | B | C | D | E | F

-- Define the adjacency relation between faces
def adjacent : Face → Face → Prop := sorry

-- Define the opposite relation between faces
def opposite : Face → Face → Prop := sorry

-- Define the shares_vertex relation between faces
def shares_vertex : Face → Face → Prop := sorry

-- Define the shares_edge relation between faces
def shares_edge : Face → Face → Prop := sorry

theorem cube_opposite_face :
  -- Condition 2: Face B is adjacent to Face A
  adjacent Face.B Face.A →
  -- Condition 3: Face C and Face D are adjacent to each other
  adjacent Face.C Face.D →
  -- Condition 3: Face C shares a vertex with Face A
  shares_vertex Face.C Face.A →
  -- Condition 3: Face D shares a vertex with Face A
  shares_vertex Face.D Face.A →
  -- Condition 4: Face E and Face F share an edge with each other
  shares_edge Face.E Face.F →
  -- Condition 4: Face E does not share an edge with Face A
  ¬ shares_edge Face.E Face.A →
  -- Condition 4: Face F does not share an edge with Face A
  ¬ shares_edge Face.F Face.A →
  -- Conclusion: Face F is opposite to Face A
  opposite Face.F Face.A := by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l2752_275222


namespace NUMINAMATH_CALUDE_benny_baseball_gear_spending_l2752_275277

/-- The amount Benny spent on baseball gear --/
def amount_spent (initial_amount left_over : ℕ) : ℕ :=
  initial_amount - left_over

/-- Theorem: Benny spent $47 on baseball gear --/
theorem benny_baseball_gear_spending :
  amount_spent 79 32 = 47 := by
  sorry

end NUMINAMATH_CALUDE_benny_baseball_gear_spending_l2752_275277


namespace NUMINAMATH_CALUDE_find_particular_number_l2752_275260

theorem find_particular_number (x : ℤ) : x - 29 + 64 = 76 → x = 41 := by
  sorry

end NUMINAMATH_CALUDE_find_particular_number_l2752_275260


namespace NUMINAMATH_CALUDE_uncorrelated_variables_l2752_275298

/-- Represents a variable in our correlation problem -/
structure Variable where
  name : String

/-- Represents a pair of variables -/
structure VariablePair where
  var1 : Variable
  var2 : Variable

/-- Defines what it means for two variables to be correlated -/
def are_correlated (pair : VariablePair) : Prop :=
  sorry  -- The actual definition would go here

/-- The list of variable pairs we're considering -/
def variable_pairs : List VariablePair :=
  [ { var1 := { name := "Grain yield" }, var2 := { name := "Amount of fertilizer used" } },
    { var1 := { name := "College entrance examination scores" }, var2 := { name := "Time spent on review" } },
    { var1 := { name := "Sales of goods" }, var2 := { name := "Advertising expenses" } },
    { var1 := { name := "Number of books sold at fixed price" }, var2 := { name := "Sales revenue" } } ]

/-- The theorem we want to prove -/
theorem uncorrelated_variables : 
  ∃ (pair : VariablePair), pair ∈ variable_pairs ∧ ¬(are_correlated pair) :=
sorry


end NUMINAMATH_CALUDE_uncorrelated_variables_l2752_275298


namespace NUMINAMATH_CALUDE_solve_for_a_l2752_275276

theorem solve_for_a (x : ℝ) (a : ℝ) (h1 : x = 0.3) 
  (h2 : (a * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3) : a = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2752_275276


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l2752_275204

theorem square_difference_of_integers (a b : ℕ) 
  (h1 : a + b = 60) 
  (h2 : a - b = 16) : 
  a^2 - b^2 = 960 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l2752_275204


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l2752_275294

/-- Represents a square tile pattern -/
structure TilePattern :=
  (side : ℕ)
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Extends a tile pattern by adding a border of white tiles -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 2,
    black_tiles := p.black_tiles,
    white_tiles := p.white_tiles + (p.side + 2)^2 - p.side^2 }

/-- The ratio of black tiles to white tiles in a pattern -/
def tile_ratio (p : TilePattern) : ℚ :=
  p.black_tiles / p.white_tiles

theorem extended_pattern_ratio :
  let original := TilePattern.mk 5 13 12
  let extended := extend_pattern original
  tile_ratio extended = 13 / 36 := by sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l2752_275294


namespace NUMINAMATH_CALUDE_fraction_equality_l2752_275272

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) : 
  (x + 4 * y) / (4 * x - y) = 10 / 57 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2752_275272


namespace NUMINAMATH_CALUDE_kiwifruit_problem_l2752_275254

-- Define the structure for weight difference and box count
structure WeightDifference :=
  (difference : ℝ)
  (count : ℕ)

-- Define the problem parameters
def standard_weight : ℝ := 25
def total_boxes : ℕ := 20
def selling_price_per_kg : ℝ := 10.6

-- Define the weight differences
def weight_differences : List WeightDifference := [
  ⟨-3, 1⟩, ⟨-2, 4⟩, ⟨-1.5, 2⟩, ⟨0, 3⟩, ⟨1, 2⟩, ⟨2.5, 8⟩
]

-- Calculate the total overweight
def total_overweight : ℝ :=
  weight_differences.foldr (λ wd acc => acc + wd.difference * wd.count) 0

-- Calculate the total weight
def total_weight : ℝ :=
  standard_weight * total_boxes + total_overweight

-- Calculate the total selling price
def total_selling_price : ℝ :=
  total_weight * selling_price_per_kg

-- Theorem to prove
theorem kiwifruit_problem :
  total_overweight = 8 ∧ total_selling_price = 5384.8 := by
  sorry


end NUMINAMATH_CALUDE_kiwifruit_problem_l2752_275254


namespace NUMINAMATH_CALUDE_solution_implies_range_l2752_275247

/-- The function f(x) = x^2 - 4x - 2 -/
def f (x : ℝ) := x^2 - 4*x - 2

/-- The theorem stating that if x^2 - 4x - 2 - a > 0 has solutions in (1,4), then a < -2 -/
theorem solution_implies_range (a : ℝ) : 
  (∃ x ∈ Set.Ioo 1 4, f x > a) → a < -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_range_l2752_275247


namespace NUMINAMATH_CALUDE_joy_meets_grandma_l2752_275217

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days until Joy sees her grandma -/
def days_until_meeting : ℕ := 2

/-- The time zone difference between Joy and her grandma in hours -/
def time_zone_difference : ℤ := 3

/-- The total number of hours until Joy sees her grandma -/
def total_hours : ℕ := hours_per_day * days_until_meeting

theorem joy_meets_grandma : total_hours = 48 := by sorry

end NUMINAMATH_CALUDE_joy_meets_grandma_l2752_275217


namespace NUMINAMATH_CALUDE_vacuum_time_per_room_l2752_275291

-- Define the total vacuuming time in hours
def total_time : ℝ := 2

-- Define the number of rooms
def num_rooms : ℕ := 6

-- Define the function to convert hours to minutes
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- Theorem statement
theorem vacuum_time_per_room : 
  (hours_to_minutes total_time) / num_rooms = 20 := by
  sorry

end NUMINAMATH_CALUDE_vacuum_time_per_room_l2752_275291


namespace NUMINAMATH_CALUDE_night_temperature_l2752_275226

/-- Given the temperature changes throughout a day, prove the night temperature. -/
theorem night_temperature (morning_temp : ℝ) (noon_rise : ℝ) (night_drop : ℝ) :
  morning_temp = 22 →
  noon_rise = 6 →
  night_drop = 10 →
  morning_temp + noon_rise - night_drop = 18 := by
  sorry

end NUMINAMATH_CALUDE_night_temperature_l2752_275226


namespace NUMINAMATH_CALUDE_purple_shells_count_l2752_275214

/-- Represents the number of shells of each color --/
structure ShellCounts where
  total : Nat
  pink : Nat
  yellow : Nat
  blue : Nat
  orange : Nat

/-- Theorem stating that the number of purple shells is 13 --/
theorem purple_shells_count (s : ShellCounts) 
  (h1 : s.total = 65)
  (h2 : s.pink = 8)
  (h3 : s.yellow = 18)
  (h4 : s.blue = 12)
  (h5 : s.orange = 14) :
  s.total - (s.pink + s.yellow + s.blue + s.orange) = 13 := by
  sorry

#check purple_shells_count

end NUMINAMATH_CALUDE_purple_shells_count_l2752_275214


namespace NUMINAMATH_CALUDE_test_questions_l2752_275270

theorem test_questions (total_points : ℕ) (five_point_questions : ℕ) (points_per_five_point : ℕ) (points_per_ten_point : ℕ) : 
  total_points = 200 →
  five_point_questions = 20 →
  points_per_five_point = 5 →
  points_per_ten_point = 10 →
  ∃ (ten_point_questions : ℕ),
    five_point_questions * points_per_five_point + ten_point_questions * points_per_ten_point = total_points ∧
    five_point_questions + ten_point_questions = 30 :=
by sorry

end NUMINAMATH_CALUDE_test_questions_l2752_275270


namespace NUMINAMATH_CALUDE_complex_multiplication_l2752_275274

theorem complex_multiplication (z : ℂ) (h : z = 1 + Complex.I) : (1 + z) * z = 1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2752_275274


namespace NUMINAMATH_CALUDE_min_value_product_l2752_275243

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z * (x + y + z) = 1) : 
  (x + y) * (y + z) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l2752_275243


namespace NUMINAMATH_CALUDE_homeless_families_donation_l2752_275284

theorem homeless_families_donation (total spent first_set second_set : ℝ) 
  (h1 : total = 900)
  (h2 : first_set = 325)
  (h3 : second_set = 260) :
  total - (first_set + second_set) = 315 := by
sorry

end NUMINAMATH_CALUDE_homeless_families_donation_l2752_275284


namespace NUMINAMATH_CALUDE_tank_dimension_proof_l2752_275246

/-- Proves that for a rectangular tank with given dimensions and insulation cost,
    the third dimension is 3 feet. -/
theorem tank_dimension_proof (x : ℝ) : 
  let length : ℝ := 4
  let width : ℝ := 5
  let cost_per_sqft : ℝ := 20
  let total_cost : ℝ := 1880
  let surface_area : ℝ := 2 * (length * width + length * x + width * x)
  surface_area * cost_per_sqft = total_cost → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_tank_dimension_proof_l2752_275246


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_three_l2752_275279

/-- The function f(x) defined as x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem stating that if f(x) is decreasing on (-∞, 4), then a ≤ -3 -/
theorem decreasing_f_implies_a_leq_neg_three (a : ℝ) :
  (∀ x y, x < y → y < 4 → f a x > f a y) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_three_l2752_275279


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2752_275220

def geometric_sequence (n : ℕ) : ℝ := (-3) ^ (n - 1)

theorem geometric_sequence_sum :
  let a := geometric_sequence
  (a 1) + |a 2| + (a 3) + |a 4| + (a 5) = 121 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2752_275220


namespace NUMINAMATH_CALUDE_inner_rectangle_area_l2752_275233

theorem inner_rectangle_area (a b : ℕ) : 
  a > 2 → 
  b > 2 → 
  (3 * a + 4) * (b + 3) = 65 → 
  a * b = 3 :=
by sorry

end NUMINAMATH_CALUDE_inner_rectangle_area_l2752_275233


namespace NUMINAMATH_CALUDE_symmetric_points_on_parabola_l2752_275266

/-- Two points on a parabola, symmetric with respect to a line -/
theorem symmetric_points_on_parabola (x₁ x₂ y₁ y₂ m : ℝ) :
  y₁ = 2 * x₁^2 →                          -- A is on the parabola
  y₂ = 2 * x₂^2 →                          -- B is on the parabola
  (y₂ - y₁) / (x₂ - x₁) = -1 →             -- A and B are symmetric (slope condition)
  (y₂ + y₁) / 2 = (x₂ + x₁) / 2 + m →      -- Midpoint of A and B lies on y = x + m
  x₁ * x₂ = -3/4 →                         -- Given condition
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_parabola_l2752_275266


namespace NUMINAMATH_CALUDE_parallelogram_angle_ratio_l2752_275265

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : Point)

-- Define the angles of the parallelogram
def angle (p : Parallelogram) (v : Fin 4) : ℝ :=
  sorry

-- State the theorem
theorem parallelogram_angle_ratio (p : Parallelogram) :
  ∃ (k : ℝ), k > 0 ∧
    angle p 0 = k ∧
    angle p 1 = 2 * k ∧
    angle p 2 = k ∧
    angle p 3 = 2 * k :=
  sorry

end NUMINAMATH_CALUDE_parallelogram_angle_ratio_l2752_275265


namespace NUMINAMATH_CALUDE_hoseok_number_l2752_275282

theorem hoseok_number : ∃ n : ℤ, n / 6 = 11 ∧ n = 66 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_number_l2752_275282


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2752_275209

/-- Given a geometric sequence with first term 1000 and sixth term 125, 
    the fourth term is 125. -/
theorem geometric_sequence_fourth_term :
  ∀ (a : ℝ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * (a 1)⁻¹ * a 0) →  -- Geometric sequence definition
  a 0 = 1000 →                                 -- First term is 1000
  a 5 = 125 →                                  -- Sixth term is 125
  a 3 = 125 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2752_275209


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2752_275261

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h : i * i = -1) :
  i / z = 1 + i → z = (1 + i) / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2752_275261


namespace NUMINAMATH_CALUDE_planning_committee_subcommittees_l2752_275228

theorem planning_committee_subcommittees (total_members : ℕ) (professor_count : ℕ) (subcommittee_size : ℕ) : 
  total_members = 12 →
  professor_count = 5 →
  subcommittee_size = 5 →
  (Nat.choose total_members subcommittee_size) - (Nat.choose (total_members - professor_count) subcommittee_size) = 771 :=
by sorry

end NUMINAMATH_CALUDE_planning_committee_subcommittees_l2752_275228


namespace NUMINAMATH_CALUDE_curve_and_point_properties_l2752_275251

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = (p.1 + 1)^2}

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the property of the curve
def curve_property (p : ℝ × ℝ) : Prop :=
  p ∈ C → (p.1 - 1)^2 + p.2^2 = (p.1 + 1)^2

-- Define the equation of the curve
def curve_equation (p : ℝ × ℝ) : Prop :=
  p ∈ C → p.2^2 = 4 * p.1

-- Define the properties for point M
def M_properties (m : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  m ∈ C ∧ a ∈ C ∧ b ∈ C ∧
  ∃ (k : ℝ), k ≠ 0 ∧
    (a.2 - m.2) / (a.1 - m.1) = k ∧
    (b.2 - m.2) / (b.1 - m.1) = -k

-- Define the properties for points D and E
def DE_properties (d e : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  d ∈ C ∧ e ∈ C ∧
  (d.2 - e.2) / (d.1 - e.1) = -(b.1 - a.1) / (b.2 - a.2) ∧
  (d.1 - F.1) * (b.1 - a.1) + (d.2 - F.2) * (b.2 - a.2) = 0 ∧
  (e.1 - d.1)^2 + (e.2 - d.2)^2 = 64

-- State the theorem
theorem curve_and_point_properties :
  (∀ p, curve_property p) →
  (∀ p, curve_equation p) →
  ∀ m a b d e,
    M_properties m a b →
    DE_properties d e a b →
    m = (1, 2) ∨ m = (1, -2) := by sorry

end NUMINAMATH_CALUDE_curve_and_point_properties_l2752_275251


namespace NUMINAMATH_CALUDE_jason_total_spent_l2752_275253

/-- The amount Jason spent on the flute -/
def flute_cost : ℚ := 142.46

/-- The amount Jason spent on the music tool -/
def music_tool_cost : ℚ := 8.89

/-- The amount Jason spent on the song book -/
def song_book_cost : ℚ := 7

/-- The total amount Jason spent at the music store -/
def total_spent : ℚ := flute_cost + music_tool_cost + song_book_cost

/-- Theorem stating that the total amount Jason spent is $158.35 -/
theorem jason_total_spent : total_spent = 158.35 := by sorry

end NUMINAMATH_CALUDE_jason_total_spent_l2752_275253


namespace NUMINAMATH_CALUDE_minimum_explorers_l2752_275255

theorem minimum_explorers (large_capacity small_capacity : ℕ) 
  (h1 : large_capacity = 24)
  (h2 : small_capacity = 9)
  (explorers : ℕ) :
  (∃ k : ℕ, explorers = k * large_capacity - 4) ∧
  (∃ m : ℕ, explorers = m * small_capacity - 4) →
  explorers ≥ 68 :=
by sorry

end NUMINAMATH_CALUDE_minimum_explorers_l2752_275255


namespace NUMINAMATH_CALUDE_gambler_outcome_l2752_275237

def gamble (initial_amount : ℚ) (bet_sequence : List Bool) : ℚ :=
  bet_sequence.foldl
    (fun amount win =>
      if win then amount + amount / 2
      else amount - amount / 2)
    initial_amount

theorem gambler_outcome :
  let initial_amount : ℚ := 100
  let bet_sequence : List Bool := [true, false, true, false]
  let final_amount := gamble initial_amount bet_sequence
  final_amount = 56.25 ∧ initial_amount - final_amount = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_gambler_outcome_l2752_275237


namespace NUMINAMATH_CALUDE_ice_cream_cost_l2752_275208

theorem ice_cream_cost (ice_cream_cartons yoghurt_cartons : ℕ) 
  (yoghurt_cost : ℚ) (cost_difference : ℚ) :
  ice_cream_cartons = 19 →
  yoghurt_cartons = 4 →
  yoghurt_cost = 1 →
  cost_difference = 129 →
  ∃ (ice_cream_cost : ℚ), 
    ice_cream_cost * ice_cream_cartons = yoghurt_cost * yoghurt_cartons + cost_difference ∧
    ice_cream_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l2752_275208


namespace NUMINAMATH_CALUDE_right_triangle_existence_and_uniqueness_l2752_275218

theorem right_triangle_existence_and_uniqueness 
  (c d : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c > d) :
  ∃! (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    a - b = d := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_existence_and_uniqueness_l2752_275218


namespace NUMINAMATH_CALUDE_graph_passes_through_second_and_fourth_quadrants_l2752_275275

-- Define the function
def f (x : ℝ) : ℝ := -3 * x

-- State the theorem
theorem graph_passes_through_second_and_fourth_quadrants :
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧ 
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_graph_passes_through_second_and_fourth_quadrants_l2752_275275


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_24_l2752_275290

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ k : ℕ, p^2 - 1 = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_24_l2752_275290


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_is_3_l2752_275283

def chocolate_bar_cost (total_cost : ℕ) (num_chocolate_bars : ℕ) (num_gummy_packs : ℕ) (num_chip_bags : ℕ) (gummy_cost : ℕ) (chip_cost : ℕ) : ℕ :=
  (total_cost - (num_gummy_packs * gummy_cost + num_chip_bags * chip_cost)) / num_chocolate_bars

theorem chocolate_bar_cost_is_3 :
  chocolate_bar_cost 150 10 10 20 2 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_is_3_l2752_275283


namespace NUMINAMATH_CALUDE_inequality_solution_l2752_275235

theorem inequality_solution (x : ℝ) : 
  (5 ≤ (x - 1) / (3 * x - 7) ∧ (x - 1) / (3 * x - 7) < 10) ↔ 
  (69 / 29 < x ∧ x ≤ 17 / 7) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2752_275235


namespace NUMINAMATH_CALUDE_prime_fraction_characterization_l2752_275212

theorem prime_fraction_characterization (k x y : ℕ+) :
  (∃ p : ℕ, Nat.Prime p ∧ (x : ℝ)^(k : ℕ) * y / ((x : ℝ)^2 + (y : ℝ)^2) = p) ↔ k = 2 ∨ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_fraction_characterization_l2752_275212


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l2752_275229

theorem sum_of_min_max_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ x', ∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l2752_275229


namespace NUMINAMATH_CALUDE_octagon_placement_l2752_275231

/-- A set of numbers from 1 to 12 -/
def CardSet : Set ℕ := {n | 1 ≤ n ∧ n ≤ 12}

/-- A function representing the placement of numbers on the octagon vertices -/
def Placement := Fin 8 → ℕ

/-- Predicate to check if a placement is valid according to the given conditions -/
def ValidPlacement (p : Placement) : Prop :=
  (∀ i, p i ∈ CardSet) ∧
  (∀ i, (p i + p ((i + 4) % 8)) % 3 = 0)

/-- The set of numbers not placed on the octagon -/
def NotPlaced (p : Placement) : Set ℕ := CardSet \ (Set.range p)

/-- Main theorem -/
theorem octagon_placement :
  ∀ p : Placement, ValidPlacement p → NotPlaced p = {3, 6, 9, 12} := by sorry

end NUMINAMATH_CALUDE_octagon_placement_l2752_275231


namespace NUMINAMATH_CALUDE_angle_of_inclination_range_l2752_275210

theorem angle_of_inclination_range (θ : ℝ) (α : ℝ) :
  (∃ x y : ℝ, Real.sqrt 3 * x + y * Real.cos θ - 1 = 0) →
  (α = Real.arctan (Real.sqrt 3 / (-Real.cos θ))) →
  π / 3 ≤ α ∧ α ≤ 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_range_l2752_275210


namespace NUMINAMATH_CALUDE_jackie_eligible_for_free_shipping_l2752_275223

def shampoo_price : ℝ := 12.50
def conditioner_price : ℝ := 15.00
def face_cream_price : ℝ := 20.00
def discount_rate : ℝ := 0.10
def free_shipping_threshold : ℝ := 75.00

def total_cost : ℝ := 2 * shampoo_price + 3 * conditioner_price + face_cream_price

def discounted_cost : ℝ := total_cost * (1 - discount_rate)

theorem jackie_eligible_for_free_shipping :
  discounted_cost ≥ free_shipping_threshold := by
  sorry

end NUMINAMATH_CALUDE_jackie_eligible_for_free_shipping_l2752_275223


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2752_275295

theorem isosceles_triangle_perimeter (m : ℝ) :
  (3 : ℝ) ^ 2 - (m + 1) * 3 + 2 * m = 0 →
  ∃ (a b : ℝ),
    a ^ 2 - (m + 1) * a + 2 * m = 0 ∧
    b ^ 2 - (m + 1) * b + 2 * m = 0 ∧
    ((a = b ∧ a + a + b = 10) ∨ (a ≠ b ∧ a + a + b = 11)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2752_275295


namespace NUMINAMATH_CALUDE_distance_between_points_on_lines_l2752_275203

/-- Given two lines and points A and B on these lines, prove that the distance between A and B is 5 -/
theorem distance_between_points_on_lines (a : ℝ) :
  let line1 := λ (x y : ℝ) => 3 * a * x - y - 2 = 0
  let line2 := λ (x y : ℝ) => (2 * a - 1) * x + 3 * a * y - 3 = 0
  let A : ℝ × ℝ := (0, -2)
  let B : ℝ × ℝ := (-3, 2)
  line1 A.1 A.2 ∧ line2 B.1 B.2 →
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 := by
sorry


end NUMINAMATH_CALUDE_distance_between_points_on_lines_l2752_275203


namespace NUMINAMATH_CALUDE_smallest_difference_l2752_275299

def digits : List Nat := [2, 4, 5, 6, 9]

def is_valid_arrangement (a b : Nat) : Prop :=
  ∃ (x y z u v : Nat),
    x ∈ digits ∧ y ∈ digits ∧ z ∈ digits ∧ u ∈ digits ∧ v ∈ digits ∧
    x ≠ y ∧ x ≠ z ∧ x ≠ u ∧ x ≠ v ∧
    y ≠ z ∧ y ≠ u ∧ y ≠ v ∧
    z ≠ u ∧ z ≠ v ∧
    u ≠ v ∧
    a = 100 * x + 10 * y + z ∧
    b = 10 * u + v

theorem smallest_difference :
  ∀ a b : Nat,
    is_valid_arrangement a b →
    a > b →
    a - b ≥ 149 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_l2752_275299


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_64_l2752_275259

theorem smallest_divisible_by_18_and_64 : ∀ n : ℕ, n > 0 → n % 18 = 0 → n % 64 = 0 → n ≥ 576 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_64_l2752_275259


namespace NUMINAMATH_CALUDE_triple_tilde_47_l2752_275236

-- Define the tilde operation
def tilde (N : ℝ) : ℝ := 0.4 * N + 2

-- State the theorem
theorem triple_tilde_47 : tilde (tilde (tilde 47)) = 6.128 := by sorry

end NUMINAMATH_CALUDE_triple_tilde_47_l2752_275236


namespace NUMINAMATH_CALUDE_power_equality_l2752_275262

theorem power_equality (m n : ℤ) (P Q : ℝ) (h1 : P = 2^m) (h2 : Q = 3^n) :
  P^(2*n) * Q^m = 12^(m*n) := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2752_275262


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2752_275268

/-- Represents a geometric sequence -/
def GeometricSequence (a r : ℝ) : ℕ → ℝ := fun n ↦ a * r ^ (n - 1)

/-- Theorem: In a geometric sequence where the first term is 3 and the fifth term is 243, the third term is 27 -/
theorem geometric_sequence_third_term
  (a r : ℝ)
  (h1 : GeometricSequence a r 1 = 3)
  (h5 : GeometricSequence a r 5 = 243) :
  GeometricSequence a r 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2752_275268


namespace NUMINAMATH_CALUDE_range_of_a_l2752_275244

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 1 > 0) ∧ 
  (∀ x : ℝ, a*x^2 + x - 1 ≤ 0) → 
  -2 < a ∧ a ≤ -1/4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2752_275244


namespace NUMINAMATH_CALUDE_village_assistant_selection_l2752_275227

theorem village_assistant_selection (n : ℕ) (k : ℕ) : 
  n = 10 → k = 3 → (Nat.choose 9 3) - (Nat.choose 7 3) = 49 := by
  sorry

end NUMINAMATH_CALUDE_village_assistant_selection_l2752_275227


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2752_275293

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2) :
  (x + 2) * (x - 2) + 3 * (1 - x) = 1 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2752_275293


namespace NUMINAMATH_CALUDE_pancake_fundraiser_l2752_275285

/-- The civic league's pancake breakfast fundraiser --/
theorem pancake_fundraiser 
  (pancake_price : ℝ) 
  (bacon_price : ℝ) 
  (pancake_stacks_sold : ℕ) 
  (bacon_slices_sold : ℕ) 
  (h1 : pancake_price = 4)
  (h2 : bacon_price = 2)
  (h3 : pancake_stacks_sold = 60)
  (h4 : bacon_slices_sold = 90) :
  pancake_price * pancake_stacks_sold + bacon_price * bacon_slices_sold = 420 :=
by sorry

end NUMINAMATH_CALUDE_pancake_fundraiser_l2752_275285


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2752_275213

/-- Given a quadratic function f(x) = x^2 + ax + b where a and b are distinct real numbers,
    if f(a) = f(b), then f(2) = 4 -/
theorem quadratic_function_property (a b : ℝ) (h_distinct : a ≠ b) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f a = f b) → f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2752_275213


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l2752_275249

/-- Proves that it takes 2 years for a man's age to be twice his son's age -/
theorem mans_age_twice_sons (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 33 →
  age_difference = 35 →
  ∃ (years : ℕ), years = 2 ∧ 
    (son_age + age_difference + years) = 2 * (son_age + years) := by
  sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l2752_275249


namespace NUMINAMATH_CALUDE_max_value_on_interval_l2752_275280

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 2*x^2 + 5

-- State the theorem
theorem max_value_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) (2 : ℝ) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) → f x ≤ f c ∧
  f c = 13 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l2752_275280


namespace NUMINAMATH_CALUDE_problem_1_l2752_275234

theorem problem_1 : 6 - (-12) / (-3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2752_275234


namespace NUMINAMATH_CALUDE_red_balls_per_box_l2752_275207

theorem red_balls_per_box (total_balls : ℕ) (num_boxes : ℕ) (balls_per_box : ℕ) :
  total_balls = 10 →
  num_boxes = 5 →
  total_balls = num_boxes * balls_per_box →
  balls_per_box = 2 := by
sorry

end NUMINAMATH_CALUDE_red_balls_per_box_l2752_275207


namespace NUMINAMATH_CALUDE_weekly_syrup_cost_l2752_275273

/-- Calculates the weekly cost of syrup for a convenience store selling soda -/
theorem weekly_syrup_cost
  (weekly_soda_sales : ℕ)
  (gallons_per_box : ℕ)
  (cost_per_box : ℕ)
  (h_weekly_soda_sales : weekly_soda_sales = 180)
  (h_gallons_per_box : gallons_per_box = 30)
  (h_cost_per_box : cost_per_box = 40) :
  (weekly_soda_sales / gallons_per_box) * cost_per_box = 240 :=
by sorry

end NUMINAMATH_CALUDE_weekly_syrup_cost_l2752_275273


namespace NUMINAMATH_CALUDE_piano_lesson_discount_percentage_l2752_275225

/-- Calculates the discount percentage on piano lessons given the piano cost, number of lessons,
    cost per lesson, and total cost after discount. -/
theorem piano_lesson_discount_percentage
  (piano_cost : ℝ)
  (num_lessons : ℕ)
  (cost_per_lesson : ℝ)
  (total_cost_after_discount : ℝ)
  (h1 : piano_cost = 500)
  (h2 : num_lessons = 20)
  (h3 : cost_per_lesson = 40)
  (h4 : total_cost_after_discount = 1100) :
  (1 - (total_cost_after_discount - piano_cost) / (num_lessons * cost_per_lesson)) * 100 = 25 :=
by sorry


end NUMINAMATH_CALUDE_piano_lesson_discount_percentage_l2752_275225


namespace NUMINAMATH_CALUDE_tricycle_wheels_l2752_275211

theorem tricycle_wheels (num_bicycles num_tricycles bicycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 24)
  (h2 : num_tricycles = 14)
  (h3 : bicycle_wheels = 2)
  (h4 : total_wheels = 90)
  : (total_wheels - num_bicycles * bicycle_wheels) / num_tricycles = 3 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_wheels_l2752_275211


namespace NUMINAMATH_CALUDE_regular_9gon_coloring_l2752_275292

-- Define a regular 9-gon
structure RegularNineGon where
  vertices : Fin 9 → Point

-- Define a coloring of the vertices
inductive Color
| Black
| White

def Coloring := Fin 9 → Color

-- Define adjacency in the 9-gon
def adjacent (i j : Fin 9) : Prop :=
  (i.val + 1) % 9 = j.val ∨ (j.val + 1) % 9 = i.val

-- Define an isosceles triangle in the 9-gon
def isoscelesTriangle (i j k : Fin 9) (polygon : RegularNineGon) : Prop :=
  let d := (i.val - j.val + 9) % 9
  (i.val - k.val + 9) % 9 = d ∨ (j.val - k.val + 9) % 9 = d

theorem regular_9gon_coloring 
  (polygon : RegularNineGon) 
  (coloring : Coloring) : 
  (∃ i j : Fin 9, adjacent i j ∧ coloring i = coloring j) ∧ 
  (∃ i j k : Fin 9, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    coloring i = coloring j ∧ coloring j = coloring k ∧ 
    isoscelesTriangle i j k polygon) :=
by sorry

end NUMINAMATH_CALUDE_regular_9gon_coloring_l2752_275292


namespace NUMINAMATH_CALUDE_brittany_age_theorem_l2752_275219

/-- Brittany's age when she returns from vacation -/
def brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) : ℕ :=
  rebecca_age + age_difference + vacation_duration

/-- Theorem stating Brittany's age after vacation -/
theorem brittany_age_theorem (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ)
  (h1 : rebecca_age = 25)
  (h2 : age_difference = 3)
  (h3 : vacation_duration = 4) :
  brittany_age_after_vacation rebecca_age age_difference vacation_duration = 32 := by
  sorry

#check brittany_age_theorem

end NUMINAMATH_CALUDE_brittany_age_theorem_l2752_275219


namespace NUMINAMATH_CALUDE_genuine_items_count_l2752_275215

theorem genuine_items_count (total_purses total_handbags : ℕ) 
  (h1 : total_purses = 26)
  (h2 : total_handbags = 24)
  (h3 : total_purses / 2 + total_handbags / 4 = (total_purses + total_handbags) - 31) :
  31 = total_purses + total_handbags - (total_purses / 2 + total_handbags / 4) :=
by sorry

end NUMINAMATH_CALUDE_genuine_items_count_l2752_275215


namespace NUMINAMATH_CALUDE_local_extremum_and_minimum_l2752_275242

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the derivative of f(x)
def f_prime (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_and_minimum (a b : ℝ) :
  (f a b 1 = 10) ∧ 
  (f_prime a b 1 = 0) →
  (a = 4 ∧ b = -11) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 3, f a b x ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_local_extremum_and_minimum_l2752_275242


namespace NUMINAMATH_CALUDE_water_flow_speed_l2752_275201

/-- The speed of the water flow given the ship's travel times and distances -/
theorem water_flow_speed (x y : ℝ) : 
  (135 / (x + y) + 70 / (x - y) = 12.5) →
  (75 / (x + y) + 110 / (x - y) = 12.5) →
  y = 3.2 := by
sorry

end NUMINAMATH_CALUDE_water_flow_speed_l2752_275201


namespace NUMINAMATH_CALUDE_special_function_is_identity_l2752_275297

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧ ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem special_function_is_identity (f : ℝ → ℝ) (h : special_function f) : 
  ∀ x : ℝ, f x = x := by sorry

end NUMINAMATH_CALUDE_special_function_is_identity_l2752_275297


namespace NUMINAMATH_CALUDE_coin_jar_problem_l2752_275238

theorem coin_jar_problem (y : ℕ) :
  (5 * y + 10 * y + 25 * y = 1440) → y = 36 :=
by sorry

end NUMINAMATH_CALUDE_coin_jar_problem_l2752_275238


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l2752_275206

-- Define the three given points
def p1 : ℝ × ℝ := (1, 5)
def p2 : ℝ × ℝ := (4, -3)
def p3 : ℝ × ℝ := (9, 5)

-- Define the ellipse type
structure Ellipse where
  endpoints : List (ℝ × ℝ)
  h_endpoints : endpoints.length = 3

-- Define the function to calculate the distance between foci
def focalDistance (e : Ellipse) : ℝ := sorry

-- Theorem statement
theorem ellipse_focal_distance :
  ∀ e : Ellipse, e.endpoints = [p1, p2, p3] → focalDistance e = 14 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l2752_275206


namespace NUMINAMATH_CALUDE_range_of_a_l2752_275250

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 = 0) ∧ 
  (∀ x : ℝ, x^2 - 2*x + a > 0) →
  a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2752_275250


namespace NUMINAMATH_CALUDE_jellybean_difference_l2752_275264

/-- The number of jellybeans each person has -/
structure JellybeanCount where
  tino : ℕ
  lee : ℕ
  arnold : ℕ

/-- The conditions of the jellybean problem -/
def jellybean_problem (j : JellybeanCount) : Prop :=
  j.tino > j.lee ∧
  j.arnold = j.lee / 2 ∧
  j.arnold = 5 ∧
  j.tino = 34

/-- The theorem stating the difference between Tino's and Lee's jellybean counts -/
theorem jellybean_difference (j : JellybeanCount) 
  (h : jellybean_problem j) : j.tino - j.lee = 24 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_difference_l2752_275264


namespace NUMINAMATH_CALUDE_handle_break_even_point_l2752_275257

/-- Represents the break-even point calculation for a company producing handles --/
theorem handle_break_even_point
  (fixed_cost : ℝ)
  (variable_cost : ℝ)
  (selling_price : ℝ)
  (break_even_quantity : ℝ)
  (h1 : fixed_cost = 7640)
  (h2 : variable_cost = 0.60)
  (h3 : selling_price = 4.60)
  (h4 : break_even_quantity = 1910) :
  fixed_cost + variable_cost * break_even_quantity = selling_price * break_even_quantity :=
by
  sorry

#check handle_break_even_point

end NUMINAMATH_CALUDE_handle_break_even_point_l2752_275257


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l2752_275245

theorem no_function_satisfies_condition :
  ∀ (f : ℝ → ℝ), ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ f (x + y^2) < f x + y :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l2752_275245


namespace NUMINAMATH_CALUDE_even_quadratic_implies_k_eq_one_l2752_275271

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The quadratic function f(x) = kx^2 + (k-1)x + 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

/-- If f(x) = kx^2 + (k-1)x + 2 is an even function, then k = 1 -/
theorem even_quadratic_implies_k_eq_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_k_eq_one_l2752_275271


namespace NUMINAMATH_CALUDE_tilted_cube_segment_length_l2752_275202

/-- Represents a tilted cube container with liquid -/
structure TiltedCube where
  edge_length : ℝ
  initial_fill_ratio : ℝ
  kb_length : ℝ
  lc_length : ℝ

/-- The length of segment LC in the tilted cube -/
def segment_lc_length (cube : TiltedCube) : ℝ := cube.lc_length

theorem tilted_cube_segment_length 
  (cube : TiltedCube)
  (h1 : cube.edge_length = 12)
  (h2 : cube.initial_fill_ratio = 5/8)
  (h3 : cube.lc_length = 2 * cube.kb_length)
  (h4 : cube.edge_length * (cube.initial_fill_ratio * cube.edge_length) = 
        (cube.lc_length + cube.kb_length) * cube.edge_length / 2) :
  segment_lc_length cube = 10 := by
sorry

end NUMINAMATH_CALUDE_tilted_cube_segment_length_l2752_275202


namespace NUMINAMATH_CALUDE_vector_problem_l2752_275256

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![4, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c (m : ℝ) : Fin 2 → ℝ := ![2, m]

/-- Dot product of two vectors in R² -/
def dot (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Parallel vectors in R² -/
def parallel (u v : Fin 2 → ℝ) : Prop := ∃ k : ℝ, u = fun i => k * (v i)

theorem vector_problem (m : ℝ) :
  (dot a (c m) < m^2 → (m > 4 ∨ m < -2)) ∧
  (parallel (fun i => a i + c m i) b → m = -14) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2752_275256


namespace NUMINAMATH_CALUDE_volume_from_vessel_b_l2752_275241

def vessel_a_concentration : ℝ := 0.45
def vessel_b_concentration : ℝ := 0.30
def vessel_c_concentration : ℝ := 0.10
def vessel_a_volume : ℝ := 4
def vessel_c_volume : ℝ := 6
def resultant_concentration : ℝ := 0.26

theorem volume_from_vessel_b (x : ℝ) : 
  vessel_a_concentration * vessel_a_volume + 
  vessel_b_concentration * x + 
  vessel_c_concentration * vessel_c_volume = 
  resultant_concentration * (vessel_a_volume + x + vessel_c_volume) → 
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_volume_from_vessel_b_l2752_275241


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l2752_275258

/-- The number of distinct permutations of the word "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l2752_275258


namespace NUMINAMATH_CALUDE_train_distance_45_minutes_l2752_275252

/-- Represents the distance traveled by a train in miles -/
def train_distance (time : ℕ) : ℕ :=
  (time / 2 : ℕ)

/-- Proves that a train traveling 1 mile every 2 minutes will cover 22 miles in 45 minutes -/
theorem train_distance_45_minutes : train_distance 45 = 22 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_45_minutes_l2752_275252


namespace NUMINAMATH_CALUDE_periodic_function_value_l2752_275200

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_value (f : ℝ → ℝ) (h1 : is_periodic f 1.5) (h2 : f 1 = 20) :
  f 13 = 20 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l2752_275200
