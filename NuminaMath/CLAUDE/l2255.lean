import Mathlib

namespace NUMINAMATH_CALUDE_tan_product_eighths_pi_l2255_225510

theorem tan_product_eighths_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_eighths_pi_l2255_225510


namespace NUMINAMATH_CALUDE_bat_pattern_area_l2255_225509

/-- A bat pattern is composed of squares and triangles -/
structure BatPattern where
  large_squares : Nat
  medium_squares : Nat
  triangles : Nat
  large_square_area : ℝ
  medium_square_area : ℝ
  triangle_area : ℝ

/-- The total area of a bat pattern -/
def total_area (b : BatPattern) : ℝ :=
  b.large_squares * b.large_square_area +
  b.medium_squares * b.medium_square_area +
  b.triangles * b.triangle_area

/-- Theorem: The area of the specific bat pattern is 27 -/
theorem bat_pattern_area :
  ∃ (b : BatPattern),
    b.large_squares = 2 ∧
    b.medium_squares = 2 ∧
    b.triangles = 3 ∧
    b.large_square_area = 8 ∧
    b.medium_square_area = 4 ∧
    b.triangle_area = 1 ∧
    total_area b = 27 := by
  sorry

end NUMINAMATH_CALUDE_bat_pattern_area_l2255_225509


namespace NUMINAMATH_CALUDE_johnny_fish_count_l2255_225521

theorem johnny_fish_count (total : ℕ) (sony_multiplier : ℕ) (johnny_count : ℕ) : 
  total = 120 →
  sony_multiplier = 7 →
  total = johnny_count + sony_multiplier * johnny_count →
  johnny_count = 15 := by
sorry

end NUMINAMATH_CALUDE_johnny_fish_count_l2255_225521


namespace NUMINAMATH_CALUDE_intersection_point_unique_l2255_225558

/-- Two lines in a 2D plane --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in a 2D plane --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def lies_on (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The intersection point of two lines --/
def intersection_point (l1 l2 : Line2D) : Point2D :=
  { x := 1, y := 0 }

theorem intersection_point_unique (l1 l2 : Line2D) :
  l1 = Line2D.mk 1 (-4) (-1) →
  l2 = Line2D.mk 2 1 (-2) →
  let p := intersection_point l1 l2
  lies_on p l1 ∧ lies_on p l2 ∧
  ∀ q : Point2D, lies_on q l1 → lies_on q l2 → q = p :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l2255_225558


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2255_225580

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → 1/a < 1/b) ∧
  (∃ a b : ℝ, 1/a < 1/b ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2255_225580


namespace NUMINAMATH_CALUDE_photos_per_album_l2255_225570

/-- Given 180 total photos divided equally among 9 albums, prove that each album contains 20 photos. -/
theorem photos_per_album (total_photos : ℕ) (num_albums : ℕ) (photos_per_album : ℕ) : 
  total_photos = 180 → num_albums = 9 → total_photos = num_albums * photos_per_album → photos_per_album = 20 := by
  sorry

end NUMINAMATH_CALUDE_photos_per_album_l2255_225570


namespace NUMINAMATH_CALUDE_min_value_expression_l2255_225520

theorem min_value_expression (x : ℝ) (hx : x > 0) : 
  (x + 5) / Real.sqrt (x + 1) ≥ 4 ∧ 
  ∃ y : ℝ, y > 0 ∧ (y + 5) / Real.sqrt (y + 1) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2255_225520


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2255_225596

theorem absolute_value_inequality (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ ((-9 ≤ x ∧ x ≤ -5) ∨ (1 ≤ x ∧ x ≤ 5)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2255_225596


namespace NUMINAMATH_CALUDE_range_of_a_valid_a_set_is_closed_l2255_225595

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the set of valid a values
def valid_a_set : Set ℝ := {a | a ≤ -2 ∨ a = 1}

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : a ≥ 0) (h2 : p a ∧ q a) : a ∈ valid_a_set := by
  sorry

-- Additional helper theorem to show the set is closed
theorem valid_a_set_is_closed : IsClosed valid_a_set := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_valid_a_set_is_closed_l2255_225595


namespace NUMINAMATH_CALUDE_center_after_transformations_l2255_225522

-- Define the initial center coordinates
def initial_center : ℝ × ℝ := (3, -4)

-- Define the reflection across x-axis function
def reflect_x (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Define the translation function
def translate_right (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 + units, point.2)

-- Theorem statement
theorem center_after_transformations :
  let reflected := reflect_x initial_center
  let final := translate_right reflected 5
  final = (8, 4) := by sorry

end NUMINAMATH_CALUDE_center_after_transformations_l2255_225522


namespace NUMINAMATH_CALUDE_candy_mix_cost_per_pound_l2255_225585

/-- Proves that the desired cost per pound of mixed candy is $2.00 given the specified conditions --/
theorem candy_mix_cost_per_pound
  (total_weight : ℝ)
  (cost_A : ℝ)
  (cost_B : ℝ)
  (weight_A : ℝ)
  (h_total_weight : total_weight = 5)
  (h_cost_A : cost_A = 3.2)
  (h_cost_B : cost_B = 1.7)
  (h_weight_A : weight_A = 1)
  : (weight_A * cost_A + (total_weight - weight_A) * cost_B) / total_weight = 2 := by
  sorry

#check candy_mix_cost_per_pound

end NUMINAMATH_CALUDE_candy_mix_cost_per_pound_l2255_225585


namespace NUMINAMATH_CALUDE_max_value_of_function_l2255_225523

theorem max_value_of_function (x : ℝ) (h : x^2 + x + 1 ≠ 0) :
  ∃ (M : ℝ), M = 13/3 ∧ ∀ (y : ℝ), (3*x^2 + 3*x + 4) / (x^2 + x + 1) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2255_225523


namespace NUMINAMATH_CALUDE_inscribed_triangle_regular_polygon_sides_l2255_225563

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle :=
  (A B C : ℝ × ℝ)  -- Vertices of the triangle
  (center : ℝ × ℝ)  -- Center of the circle
  (radius : ℝ)  -- Radius of the circle

/-- Calculates the angle at a vertex of a triangle -/
def angle (t : InscribedTriangle) (v : Fin 3) : ℝ :=
  sorry  -- Definition of angle calculation

/-- Represents a regular polygon inscribed in a circle -/
structure RegularPolygon :=
  (n : ℕ)  -- Number of sides
  (center : ℝ × ℝ)  -- Center of the circle
  (radius : ℝ)  -- Radius of the circle

/-- Checks if two points are adjacent vertices of a regular polygon -/
def areAdjacentVertices (p : RegularPolygon) (v1 v2 : ℝ × ℝ) : Prop :=
  sorry  -- Definition of adjacency check

theorem inscribed_triangle_regular_polygon_sides 
  (t : InscribedTriangle) 
  (p : RegularPolygon) 
  (h1 : angle t 1 = angle t 2)  -- ∠B = ∠C
  (h2 : angle t 1 = 3 * angle t 0)  -- ∠B = 3∠A
  (h3 : t.center = p.center ∧ t.radius = p.radius)  -- Same circle
  (h4 : areAdjacentVertices p t.B t.C)  -- B and C are adjacent vertices
  : p.n = 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_regular_polygon_sides_l2255_225563


namespace NUMINAMATH_CALUDE_julia_drove_214_miles_l2255_225525

/-- Calculates the number of miles driven given the total cost, daily rental rate, and per-mile rate -/
def miles_driven (total_cost daily_rate mile_rate : ℚ) : ℚ :=
  (total_cost - daily_rate) / mile_rate

/-- Proves that Julia drove 214 miles given the rental conditions -/
theorem julia_drove_214_miles :
  let total_cost : ℚ := 46.12
  let daily_rate : ℚ := 29
  let mile_rate : ℚ := 0.08
  miles_driven total_cost daily_rate mile_rate = 214 := by
    sorry

#eval miles_driven 46.12 29 0.08

end NUMINAMATH_CALUDE_julia_drove_214_miles_l2255_225525


namespace NUMINAMATH_CALUDE_modulus_of_z_l2255_225527

theorem modulus_of_z (z : ℂ) (r θ : ℝ) (h1 : z + 1/z = r) (h2 : r = 2 * Real.sin θ) (h3 : |r| < 3) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2255_225527


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l2255_225588

theorem part_to_whole_ratio (N P : ℚ) 
  (h1 : (1/4) * (1/3) * P = 25)
  (h2 : (2/5) * N = 300) : 
  P / N = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l2255_225588


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2255_225537

/-- An arithmetic sequence with sum of first n terms S_n = 2n^2 - 25n -/
def S (n : ℕ) : ℤ := 2 * n^2 - 25 * n

/-- The nth term of the arithmetic sequence -/
def a (n : ℕ) : ℤ := 4 * n - 27

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = 4 * n - 27) ∧
  (∀ n : ℕ, n ≠ 6 → S n > S 6) ∧
  S 6 = -78 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2255_225537


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2255_225576

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + 17 + (2 * x) + 15 + (2 * x + 6)) / 5 = 26 → x = 82 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2255_225576


namespace NUMINAMATH_CALUDE_sum_abc_l2255_225535

theorem sum_abc (a b c : ℚ) 
  (eq1 : 2 * a + 3 * b + c = 27) 
  (eq2 : 4 * a + 6 * b + 5 * c = 71) : 
  a + b + c = 115 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_l2255_225535


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2255_225543

theorem sine_cosine_inequality (a : ℝ) : 
  (∀ x : ℝ, Real.sin x ^ 6 + Real.cos x ^ 6 + 2 * a * Real.sin x * Real.cos x ≥ 0) ↔ 
  |a| ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2255_225543


namespace NUMINAMATH_CALUDE_lathe_problem_l2255_225571

/-- Represents the work efficiency of a lathe -/
structure Efficiency : Type :=
  (value : ℝ)

/-- Represents a lathe with its efficiency and start time -/
structure Lathe : Type :=
  (efficiency : Efficiency)
  (startTime : ℝ)

/-- The number of parts processed by a lathe after a given time -/
def partsProcessed (l : Lathe) (time : ℝ) : ℝ :=
  l.efficiency.value * (time - l.startTime)

theorem lathe_problem (a b c : Lathe) :
  a.startTime = c.startTime - 10 →
  c.startTime = b.startTime - 5 →
  partsProcessed b (b.startTime + 10) = partsProcessed c (b.startTime + 10) →
  partsProcessed a (c.startTime + 30) = partsProcessed c (c.startTime + 30) →
  ∃ t : ℝ, t = 15 ∧ partsProcessed a (b.startTime + t) = partsProcessed b (b.startTime + t) :=
by sorry

end NUMINAMATH_CALUDE_lathe_problem_l2255_225571


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l2255_225508

/-- Converts a binary number (represented as a list of 0s and 1s) to decimal -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

theorem binary_sum_theorem :
  let binary1 := [1, 0, 1, 1, 0, 0, 1]
  let binary2 := [0, 0, 0, 1, 1, 1]
  let binary3 := [0, 1, 0, 1]
  (binary_to_decimal binary1) + (binary_to_decimal binary2) + (binary_to_decimal binary3) = 143 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l2255_225508


namespace NUMINAMATH_CALUDE_square_sum_equals_six_l2255_225511

theorem square_sum_equals_six (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_six_l2255_225511


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l2255_225517

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem: The equation of the tangent line to y = x^3 - 3x^2 + 1 at (0, 1) is y = 1
theorem tangent_line_at_origin (x : ℝ) : 
  (f' 0) * x + f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l2255_225517


namespace NUMINAMATH_CALUDE_base4_addition_subtraction_l2255_225516

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (a b c : ℕ) : ℕ := a * 4^2 + b * 4^1 + c * 4^0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d := n / 64
  let r := n % 64
  let c := r / 16
  let r' := r % 16
  let b := r' / 4
  let a := r' % 4
  (d, c, b, a)

theorem base4_addition_subtraction :
  let x := base4ToBase10 2 0 3
  let y := base4ToBase10 3 2 1
  let z := base4ToBase10 1 1 2
  base10ToBase4 (x + y - z) = (1, 0, 1, 2) := by sorry

end NUMINAMATH_CALUDE_base4_addition_subtraction_l2255_225516


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l2255_225592

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

-- State the theorem
theorem subset_implies_m_values (m : ℝ) : B m ⊆ A → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l2255_225592


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2255_225579

def A : Set ℝ := {x | x^2 ≤ 1}
def B : Set ℝ := {x | x > 0}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2255_225579


namespace NUMINAMATH_CALUDE_year_2078_is_wu_xu_l2255_225528

/-- Represents the Heavenly Stems in the Chinese calendar system -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches in the Chinese calendar system -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Chinese calendar system -/
structure ChineseYear where
  stem : HeavenlyStem
  branch : EarthlyBranch

/-- The number of Heavenly Stems -/
def numHeavenlyStems : Nat := 10

/-- The number of Earthly Branches -/
def numEarthlyBranches : Nat := 12

/-- The starting year of the reform and opening up period -/
def reformStartYear : Nat := 1978

/-- Function to get the next Heavenly Stem in the cycle -/
def nextHeavenlyStem (s : HeavenlyStem) : HeavenlyStem := sorry

/-- Function to get the next Earthly Branch in the cycle -/
def nextEarthlyBranch (b : EarthlyBranch) : EarthlyBranch := sorry

/-- Function to get the Chinese Year representation for a given year -/
def getChineseYear (year : Nat) : ChineseYear := sorry

/-- Theorem stating that the year 2078 is represented as "Wu Xu" -/
theorem year_2078_is_wu_xu :
  let year2016 := ChineseYear.mk HeavenlyStem.Bing EarthlyBranch.Shen
  let year2078 := getChineseYear 2078
  year2078 = ChineseYear.mk HeavenlyStem.Wu EarthlyBranch.Xu := by
  sorry

end NUMINAMATH_CALUDE_year_2078_is_wu_xu_l2255_225528


namespace NUMINAMATH_CALUDE_panda_bamboo_transport_l2255_225538

/-- Represents the maximum number of bamboo sticks that can be transported -/
def max_bamboo_transported (initial_bamboo : ℕ) (capacity : ℕ) (consumption : ℕ) : ℕ :=
  initial_bamboo - consumption * (2 * (initial_bamboo / capacity) - 1)

/-- Theorem stating that the maximum number of bamboo sticks transported is 165 -/
theorem panda_bamboo_transport :
  max_bamboo_transported 200 50 5 = 165 := by
  sorry

end NUMINAMATH_CALUDE_panda_bamboo_transport_l2255_225538


namespace NUMINAMATH_CALUDE_derivative_at_one_l2255_225546

-- Define the function f(x) = (x-2)²
def f (x : ℝ) : ℝ := (x - 2)^2

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = -2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2255_225546


namespace NUMINAMATH_CALUDE_equation_solution_l2255_225564

theorem equation_solution : 
  ∃ y : ℝ, (1/8: ℝ)^(3*y+12) = (64 : ℝ)^(y+4) ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2255_225564


namespace NUMINAMATH_CALUDE_inequality_proof_l2255_225545

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) ≥ (a*b + b*c + c*a)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2255_225545


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l2255_225518

/-- Given two adjacent points of a square at (1,2) and (5,5), the area of the square is 25. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (5, 5)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := distance^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l2255_225518


namespace NUMINAMATH_CALUDE_total_pencils_l2255_225547

theorem total_pencils (drawer : ℕ) (desk : ℕ) (added : ℕ) 
  (h1 : drawer = 43)
  (h2 : desk = 19)
  (h3 : added = 16) :
  drawer + desk + added = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l2255_225547


namespace NUMINAMATH_CALUDE_field_trip_lunch_cost_l2255_225549

/-- Calculates the total cost of lunches for a field trip. -/
def total_lunch_cost (num_children : ℕ) (num_chaperones : ℕ) (num_teachers : ℕ) (num_extra : ℕ) (cost_per_lunch : ℕ) : ℕ :=
  (num_children + num_chaperones + num_teachers + num_extra) * cost_per_lunch

/-- Proves that the total cost of lunches for the given field trip is $308. -/
theorem field_trip_lunch_cost :
  total_lunch_cost 35 5 1 3 7 = 308 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_lunch_cost_l2255_225549


namespace NUMINAMATH_CALUDE_math_test_score_distribution_l2255_225514

theorem math_test_score_distribution (total_students : ℕ) (percentile_80_score : ℕ) :
  total_students = 1200 →
  percentile_80_score = 103 →
  (∃ (students_above_threshold : ℕ),
    students_above_threshold ≥ 240 ∧
    students_above_threshold = total_students - (total_students * 80 / 100)) := by
  sorry

end NUMINAMATH_CALUDE_math_test_score_distribution_l2255_225514


namespace NUMINAMATH_CALUDE_initial_books_correct_l2255_225536

/-- The number of books in the special collection at the beginning of the month. -/
def initial_books : ℕ := 75

/-- The number of books loaned out during the month. -/
def loaned_books : ℕ := 60

/-- The percentage of loaned books that are returned by the end of the month. -/
def return_rate : ℚ := 70 / 100

/-- The number of books in the special collection at the end of the month. -/
def final_books : ℕ := 57

/-- Theorem stating that the initial number of books is correct given the conditions. -/
theorem initial_books_correct : 
  initial_books = final_books + (loaned_books - (return_rate * loaned_books).floor) :=
sorry

end NUMINAMATH_CALUDE_initial_books_correct_l2255_225536


namespace NUMINAMATH_CALUDE_total_turtles_is_100_l2255_225561

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The difference between Marion's and Martha's turtles -/
def difference : ℕ := 20

/-- The number of turtles Marion received -/
def marion_turtles : ℕ := martha_turtles + difference

/-- The total number of turtles received by Martha and Marion -/
def total_turtles : ℕ := martha_turtles + marion_turtles

theorem total_turtles_is_100 : total_turtles = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_turtles_is_100_l2255_225561


namespace NUMINAMATH_CALUDE_complex_number_problem_l2255_225515

theorem complex_number_problem (z z₁ z₂ : ℂ) : 
  z₁ = 5 + 10 * Complex.I ∧ 
  z₂ = 3 - 4 * Complex.I ∧ 
  1 / z = 1 / z₁ + 1 / z₂ → 
  z = 5 - (5 / 2) * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2255_225515


namespace NUMINAMATH_CALUDE_simplify_expression_l2255_225551

theorem simplify_expression (x : ℝ) : (3*x + 25) + (150*x - 5) + x^2 = x^2 + 153*x + 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2255_225551


namespace NUMINAMATH_CALUDE_average_temperature_l2255_225572

def temperatures : List ℝ := [53, 59, 61, 55, 50]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 55.6 := by sorry

end NUMINAMATH_CALUDE_average_temperature_l2255_225572


namespace NUMINAMATH_CALUDE_solution_set_equality_l2255_225541

-- Define the solution set of |8x+9| < 7
def solution_set : Set ℝ := {x : ℝ | |8*x + 9| < 7}

-- Define the inequality ax^2 + bx > 2
def inequality (a b : ℝ) (x : ℝ) : Prop := a*x^2 + b*x > 2

-- State the theorem
theorem solution_set_equality (a b : ℝ) : 
  (∀ x : ℝ, x ∈ solution_set ↔ inequality a b x) → a + b = -13 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2255_225541


namespace NUMINAMATH_CALUDE_library_shelf_capacity_l2255_225534

/-- Given a library with a total number of books and shelves, 
    calculate the number of books per shelf. -/
def books_per_shelf (total_books : ℕ) (total_shelves : ℕ) : ℕ :=
  total_books / total_shelves

/-- Theorem stating that in a library with 14240 books and 1780 shelves,
    each shelf holds 8 books. -/
theorem library_shelf_capacity : books_per_shelf 14240 1780 = 8 := by
  sorry

end NUMINAMATH_CALUDE_library_shelf_capacity_l2255_225534


namespace NUMINAMATH_CALUDE_tommys_balloons_l2255_225581

theorem tommys_balloons (initial_balloons : ℕ) (final_balloons : ℕ) : 
  initial_balloons = 26 → final_balloons = 60 → final_balloons - initial_balloons = 34 := by
  sorry

end NUMINAMATH_CALUDE_tommys_balloons_l2255_225581


namespace NUMINAMATH_CALUDE_new_persons_joined_l2255_225507

/-- Proves that 20 new persons joined the group given the initial conditions and final average age -/
theorem new_persons_joined (initial_avg : ℝ) (new_avg : ℝ) (final_avg : ℝ) (initial_count : ℕ) : 
  initial_avg = 16 → new_avg = 15 → final_avg = 15.5 → initial_count = 20 → 
  ∃ (new_count : ℕ), 
    (initial_count * initial_avg + new_count * new_avg) / (initial_count + new_count) = final_avg ∧
    new_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_new_persons_joined_l2255_225507


namespace NUMINAMATH_CALUDE_roller_coaster_tickets_l2255_225567

/-- Calculates the total number of tickets needed for a group of friends riding roller coasters -/
theorem roller_coaster_tickets (
  first_coaster_cost : ℕ)
  (discount_rate : ℚ)
  (discount_threshold : ℕ)
  (new_coaster_cost : ℕ)
  (num_friends : ℕ)
  (first_coaster_rides : ℕ)
  (new_coaster_rides : ℕ)
  (h1 : first_coaster_cost = 6)
  (h2 : discount_rate = 15 / 100)
  (h3 : discount_threshold = 10)
  (h4 : new_coaster_cost = 8)
  (h5 : num_friends = 8)
  (h6 : first_coaster_rides = 2)
  (h7 : new_coaster_rides = 1)
  : ℕ :=
  160

#check roller_coaster_tickets

end NUMINAMATH_CALUDE_roller_coaster_tickets_l2255_225567


namespace NUMINAMATH_CALUDE_unique_point_equal_angles_l2255_225556

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus
def F : ℝ × ℝ := (1, 0)

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- Define a chord passing through F
def is_chord_through_F (A B : ℝ × ℝ) : Prop :=
  is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2 ∧ 
  ∃ t : ℝ, (1 - t) • A + t • B = F

-- Define equality of angles APF and BPF
def angles_equal (A B : ℝ × ℝ) : Prop :=
  (A.2 / (A.1 - P.1)) + (B.2 / (B.1 - P.1)) = 0

-- Theorem statement
theorem unique_point_equal_angles :
  ∀ A B : ℝ × ℝ, is_chord_through_F A B → angles_equal A B ∧
  ∀ p : ℝ, p > 0 ∧ p ≠ 2 → ∃ A' B' : ℝ × ℝ, is_chord_through_F A' B' ∧ ¬angles_equal A' B' :=
sorry

end NUMINAMATH_CALUDE_unique_point_equal_angles_l2255_225556


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2255_225566

theorem quadratic_roots_sum_of_squares (a b c : ℝ) : 
  (∀ x, x^2 - 7*x + c = 0 ↔ x = a ∨ x = b) →
  a^2 + b^2 = 17 →
  c = 16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2255_225566


namespace NUMINAMATH_CALUDE_equation_solvable_for_small_primes_l2255_225578

theorem equation_solvable_for_small_primes :
  ∀ p : ℕ, p ≤ 100 → Prime p → ∃ x y : ℕ, y^37 ≡ x^3 + 11 [ZMOD p] :=
by sorry

end NUMINAMATH_CALUDE_equation_solvable_for_small_primes_l2255_225578


namespace NUMINAMATH_CALUDE_number_equation_solution_l2255_225519

theorem number_equation_solution : ∃ x : ℝ, 
  x^(5/4) * 12^(1/4) * 60^(3/4) = 300 ∧ 
  ∀ ε > 0, |x - 6| < ε :=
by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2255_225519


namespace NUMINAMATH_CALUDE_expression_value_l2255_225569

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (sum_zero : x + y + z = 0) (sum_prod_nonzero : x*y + x*z + y*z ≠ 0) :
  (x^6 + y^6 + z^6) / (x*y*z * (x*y + x*z + y*z)) = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2255_225569


namespace NUMINAMATH_CALUDE_set_B_equality_l2255_225553

def A : Set ℤ := {-1, 0, 1, 2}

def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2*x}

theorem set_B_equality : B = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_set_B_equality_l2255_225553


namespace NUMINAMATH_CALUDE_larger_number_proof_larger_number_is_1891_l2255_225530

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def has_at_most_three_decimal_places (n : ℕ) : Prop := n < 1000

theorem larger_number_proof (small : ℕ) (large : ℕ) : Prop :=
  large - small = 1355 ∧
  large / small = 6 ∧
  large % small = 15 ∧
  is_prime (sum_of_digits large) ∧
  has_at_most_three_decimal_places small ∧
  has_at_most_three_decimal_places large ∧
  large = 1891

theorem larger_number_is_1891 : ∃ (small : ℕ) (large : ℕ), larger_number_proof small large := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_larger_number_is_1891_l2255_225530


namespace NUMINAMATH_CALUDE_sandro_children_l2255_225598

/-- Calculates the total number of children Sandro has -/
def total_children (sons : ℕ) (daughter_ratio : ℕ) : ℕ :=
  sons + sons * daughter_ratio

theorem sandro_children :
  let sons := 3
  let daughter_ratio := 6
  total_children sons daughter_ratio = 21 := by
  sorry

end NUMINAMATH_CALUDE_sandro_children_l2255_225598


namespace NUMINAMATH_CALUDE_determinant_equality_l2255_225577

theorem determinant_equality (p q r s : ℝ) : 
  (p * s - q * r = 7) → ((p + 2 * r) * s - (q + 2 * s) * r = 7) := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l2255_225577


namespace NUMINAMATH_CALUDE_circus_performance_legs_on_ground_l2255_225587

/-- Calculates the total number of legs/paws/hands on the ground in a circus performance --/
def circus_legs_on_ground (total_dogs : ℕ) (total_cats : ℕ) (total_horses : ℕ) (acrobats_one_hand : ℕ) (acrobats_two_hands : ℕ) : ℕ :=
  let dogs_on_back_legs := total_dogs / 2
  let dogs_on_all_fours := total_dogs - dogs_on_back_legs
  let cats_on_back_legs := total_cats / 3
  let cats_on_all_fours := total_cats - cats_on_back_legs
  let horses_on_hind_legs := 2
  let horses_on_all_fours := total_horses - horses_on_hind_legs
  
  let dog_paws := dogs_on_back_legs * 2 + dogs_on_all_fours * 4
  let cat_paws := cats_on_back_legs * 2 + cats_on_all_fours * 4
  let horse_hooves := horses_on_hind_legs * 2 + horses_on_all_fours * 4
  let acrobat_hands := acrobats_one_hand * 1 + acrobats_two_hands * 2
  
  dog_paws + cat_paws + horse_hooves + acrobat_hands

theorem circus_performance_legs_on_ground :
  circus_legs_on_ground 20 10 5 4 2 = 118 := by
  sorry

end NUMINAMATH_CALUDE_circus_performance_legs_on_ground_l2255_225587


namespace NUMINAMATH_CALUDE_total_items_is_110_l2255_225529

/-- The number of croissants each person eats per day -/
def croissants_per_person : ℕ := 7

/-- The number of cakes each person eats per day -/
def cakes_per_person : ℕ := 18

/-- The number of pizzas each person eats per day -/
def pizzas_per_person : ℕ := 30

/-- The number of people eating -/
def number_of_people : ℕ := 2

/-- The total number of items consumed by both people in a day -/
def total_items : ℕ := 
  (croissants_per_person + cakes_per_person + pizzas_per_person) * number_of_people

theorem total_items_is_110 : total_items = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_items_is_110_l2255_225529


namespace NUMINAMATH_CALUDE_total_rock_is_16_l2255_225548

/-- The amount of rock costing $30 per ton -/
def rock_30 : ℕ := 8

/-- The amount of rock costing $40 per ton -/
def rock_40 : ℕ := 8

/-- The total amount of rock needed -/
def total_rock : ℕ := rock_30 + rock_40

theorem total_rock_is_16 : total_rock = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_rock_is_16_l2255_225548


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2255_225582

theorem water_tank_capacity : ∀ (c : ℝ),
  (c / 3 : ℝ) / c = 1 / 3 →
  ((c / 3 + 7) : ℝ) / c = 2 / 5 →
  c = 105 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2255_225582


namespace NUMINAMATH_CALUDE_sawyer_octopus_count_l2255_225555

-- Define the number of legs Sawyer saw
def total_legs : ℕ := 40

-- Define the number of legs each octopus has
def legs_per_octopus : ℕ := 8

-- Theorem statement
theorem sawyer_octopus_count :
  total_legs / legs_per_octopus = 5 := by
  sorry

end NUMINAMATH_CALUDE_sawyer_octopus_count_l2255_225555


namespace NUMINAMATH_CALUDE_unattainable_y_value_l2255_225532

theorem unattainable_y_value (x y : ℝ) :
  x ≠ -5/4 →
  y = (2 - 3*x) / (4*x + 5) →
  y ≠ -3/4 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l2255_225532


namespace NUMINAMATH_CALUDE_polynomial_integer_roots_l2255_225542

theorem polynomial_integer_roots (p : ℤ → ℤ) 
  (h1 : ∃ a : ℤ, p a = 1) 
  (h3 : ∃ b : ℤ, p b = 3) : 
  ¬(∃ y1 y2 : ℤ, y1 ≠ y2 ∧ p y1 = 2 ∧ p y2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_integer_roots_l2255_225542


namespace NUMINAMATH_CALUDE_equation_system_solution_l2255_225552

theorem equation_system_solution :
  ∀ x y : ℝ,
  x * y * (x + y) = 30 ∧
  x^3 + y^3 = 35 →
  ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l2255_225552


namespace NUMINAMATH_CALUDE_jessie_muffin_division_l2255_225531

/-- The number of muffins each person receives when 35 muffins are divided equally among Jessie and her friends -/
def muffins_per_person (total_muffins : ℕ) (num_friends : ℕ) : ℕ :=
  total_muffins / (num_friends + 1)

/-- Theorem stating that when 35 muffins are divided equally among Jessie and her 6 friends, each person will receive 5 muffins -/
theorem jessie_muffin_division :
  muffins_per_person 35 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jessie_muffin_division_l2255_225531


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2255_225502

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, (Complex.mk 2 a) / (Complex.mk 2 (-1)) = Complex.I * b) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2255_225502


namespace NUMINAMATH_CALUDE_circles_intersect_l2255_225593

/-- Definition of circle C1 -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 3*y + 2 = 0

/-- Definition of circle C2 -/
def C2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 3*y + 1 = 0

/-- Theorem stating that C1 and C2 are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l2255_225593


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_lub_l2255_225573

/-- A trapezoid inscribed in a unit circle -/
structure InscribedTrapezoid where
  /-- The length of side AB -/
  s₁ : ℝ
  /-- The length of side CD -/
  s₂ : ℝ
  /-- The distance from the center to the intersection of diagonals -/
  d : ℝ
  /-- s₁ and s₂ are between 0 and 2 (diameter of unit circle) -/
  h_s₁_bounds : 0 < s₁ ∧ s₁ ≤ 2
  h_s₂_bounds : 0 < s₂ ∧ s₂ ≤ 2
  /-- d is positive (intersection is not at the center) -/
  h_d_pos : d > 0

/-- The least upper bound of (s₁ - s₂) / d for inscribed trapezoids is 2 -/
theorem inscribed_trapezoid_lub :
  ∀ T : InscribedTrapezoid, (T.s₁ - T.s₂) / T.d ≤ 2 ∧
  ∀ ε > 0, ∃ T : InscribedTrapezoid, (T.s₁ - T.s₂) / T.d > 2 - ε := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_lub_l2255_225573


namespace NUMINAMATH_CALUDE_chefs_and_waiters_arrangements_l2255_225594

/-- The number of ways to arrange chefs and waiters in a row --/
def arrangements (num_chefs num_waiters : ℕ) : ℕ :=
  if num_chefs + num_waiters ≠ 5 then 0
  else if num_chefs ≠ 2 then 0
  else if num_waiters ≠ 3 then 0
  else 36

/-- Theorem stating that the number of arrangements for 2 chefs and 3 waiters is 36 --/
theorem chefs_and_waiters_arrangements :
  arrangements 2 3 = 36 := by sorry

end NUMINAMATH_CALUDE_chefs_and_waiters_arrangements_l2255_225594


namespace NUMINAMATH_CALUDE_grandmas_will_l2255_225503

theorem grandmas_will (total : ℕ) (shelby_share : ℕ) (other_grandchildren : ℕ) (one_share : ℕ) :
  total = 124600 ∧
  shelby_share = total / 2 ∧
  other_grandchildren = 10 ∧
  one_share = 6230 ∧
  (total - shelby_share) / other_grandchildren = one_share →
  total = 124600 :=
by sorry

end NUMINAMATH_CALUDE_grandmas_will_l2255_225503


namespace NUMINAMATH_CALUDE_part_one_part_two_l2255_225550

-- Define the propositions r(x) and s(x)
def r (m : ℝ) (x : ℝ) : Prop := Real.sin x + Real.cos x > m
def s (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 > 0

-- Part 1
theorem part_one (m : ℝ) : 
  (∀ x ∈ Set.Ioo (1/2 : ℝ) 2, s m x) → m > -2 :=
sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x : ℝ, (r m x ∧ ¬s m x) ∨ (¬r m x ∧ s m x)) →
  m ∈ Set.Iic (-2) ∪ Set.Ioc (-Real.sqrt 2) 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2255_225550


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2255_225568

theorem f_derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 0 = -4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2255_225568


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_range_l2255_225540

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(1-a)*x + 2

-- State the theorem
theorem decreasing_function_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Iic 4, ∀ y ∈ Set.Iic 4, x < y → f a x > f a y) →
  a ∈ Set.Ici 5 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_range_l2255_225540


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_half_l2255_225505

/-- The equation has a unique solution if and only if a = 1/2 -/
theorem unique_solution_iff_a_eq_half (a : ℝ) (ha : a > 0) :
  (∃! x : ℝ, 2 * a * x = x^2 - 2 * a * Real.log x) ↔ a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_half_l2255_225505


namespace NUMINAMATH_CALUDE_sprint_team_total_miles_l2255_225584

theorem sprint_team_total_miles :
  let num_people : Float := 150.0
  let miles_per_person : Float := 5.0
  let total_miles := num_people * miles_per_person
  total_miles = 750.0 := by
sorry

end NUMINAMATH_CALUDE_sprint_team_total_miles_l2255_225584


namespace NUMINAMATH_CALUDE_janice_stair_climb_l2255_225544

/-- The number of times Janice goes up the stairs in a day. -/
def times_up : ℕ := 5

/-- The number of flights of stairs for each trip up. -/
def flights_per_trip : ℕ := 3

/-- The number of times Janice goes down the stairs in a day. -/
def times_down : ℕ := 3

/-- The total number of flights walked (up and down) in a day. -/
def total_flights : ℕ := 24

theorem janice_stair_climb :
  times_up * flights_per_trip + times_down * flights_per_trip = total_flights :=
by sorry

end NUMINAMATH_CALUDE_janice_stair_climb_l2255_225544


namespace NUMINAMATH_CALUDE_special_operation_l2255_225562

theorem special_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum : a + b = 12) (product : a * b = 35) : 
  (1 : ℚ) / a + (1 : ℚ) / b = 12 / 35 := by
  sorry

end NUMINAMATH_CALUDE_special_operation_l2255_225562


namespace NUMINAMATH_CALUDE_parabolas_intersection_circle_l2255_225512

/-- The parabolas y = (x - 2)^2 and x + 1 = (y + 2)^2 intersect at four points that lie on a circle with radius squared equal to 3/2 -/
theorem parabolas_intersection_circle (x y : ℝ) : 
  (y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) → 
  (x - 5/2)^2 + (y + 3/2)^2 = 3/2 := by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_circle_l2255_225512


namespace NUMINAMATH_CALUDE_library_reorganization_l2255_225506

theorem library_reorganization (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 2025 →
  books_per_initial_box = 25 →
  books_per_new_box = 28 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 21 := by
sorry

end NUMINAMATH_CALUDE_library_reorganization_l2255_225506


namespace NUMINAMATH_CALUDE_largest_domain_is_plus_minus_one_l2255_225501

def is_valid_domain (S : Set ℝ) : Prop :=
  (∀ x ∈ S, x ≠ 0) ∧ 
  (∀ x ∈ S, (1 / x) ∈ S) ∧
  (∃ g : ℝ → ℝ, ∀ x ∈ S, g x + g (1 / x) = 2 * x)

theorem largest_domain_is_plus_minus_one :
  ∀ S : Set ℝ, is_valid_domain S → S ⊆ {-1, 1} := by sorry

end NUMINAMATH_CALUDE_largest_domain_is_plus_minus_one_l2255_225501


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2255_225597

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ x, x^3 - 20*x^2 + 96*x - 91 = 0 ↔ (x = p ∨ x = q ∨ x = r)) →
  (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20*s^2 + 96*s - 91) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 225 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2255_225597


namespace NUMINAMATH_CALUDE_mod_equivalence_l2255_225504

theorem mod_equivalence (m : ℕ) : 
  198 * 935 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 30 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l2255_225504


namespace NUMINAMATH_CALUDE_school_supplies_cost_l2255_225526

/-- Calculate the total amount spent on school supplies --/
theorem school_supplies_cost 
  (original_backpack_price : ℕ) 
  (original_binder_price : ℕ) 
  (backpack_price_increase : ℕ) 
  (binder_price_decrease : ℕ) 
  (num_binders : ℕ) 
  (h1 : original_backpack_price = 50)
  (h2 : original_binder_price = 20)
  (h3 : backpack_price_increase = 5)
  (h4 : binder_price_decrease = 2)
  (h5 : num_binders = 3) :
  (original_backpack_price + backpack_price_increase) + 
  num_binders * (original_binder_price - binder_price_decrease) = 109 :=
by sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l2255_225526


namespace NUMINAMATH_CALUDE_soccer_season_games_l2255_225539

/-- Represents a soccer team's season performance -/
structure SoccerSeason where
  totalGames : ℕ
  firstGames : ℕ
  firstWins : ℕ
  remainingWins : ℕ

/-- Conditions for the soccer season -/
def validSeason (s : SoccerSeason) : Prop :=
  s.totalGames % 2 = 0 ∧ 
  s.firstGames = 36 ∧
  s.firstWins = 16 ∧
  s.remainingWins ≥ (s.totalGames - s.firstGames) * 3 / 4 ∧
  (s.firstWins + s.remainingWins) * 100 = s.totalGames * 62

theorem soccer_season_games (s : SoccerSeason) (h : validSeason s) : s.totalGames = 84 :=
sorry

end NUMINAMATH_CALUDE_soccer_season_games_l2255_225539


namespace NUMINAMATH_CALUDE_total_gold_stars_l2255_225591

def gold_stars_yesterday : ℕ := 4
def gold_stars_today : ℕ := 3

theorem total_gold_stars : gold_stars_yesterday + gold_stars_today = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_gold_stars_l2255_225591


namespace NUMINAMATH_CALUDE_a_initial_is_9000_l2255_225575

/-- Represents the initial investment and profit distribution scenario -/
structure BusinessScenario where
  a_initial : ℕ  -- A's initial investment
  b_investment : ℕ  -- B's investment
  b_join_time : ℕ  -- Time when B joined (in months)
  total_time : ℕ  -- Total time of the year (in months)
  profit_ratio : Rat  -- Profit ratio (A:B)

/-- Calculates the initial investment of A given the business scenario -/
def calculate_a_initial (scenario : BusinessScenario) : ℕ :=
  (scenario.b_investment * scenario.b_join_time * 2) / scenario.total_time

/-- Theorem stating that A's initial investment is 9000 given the specific conditions -/
theorem a_initial_is_9000 : 
  let scenario : BusinessScenario := {
    a_initial := 9000,
    b_investment := 27000,
    b_join_time := 2,
    total_time := 12,
    profit_ratio := 2/1
  }
  calculate_a_initial scenario = 9000 := by
  sorry

#eval calculate_a_initial {
  a_initial := 9000,
  b_investment := 27000,
  b_join_time := 2,
  total_time := 12,
  profit_ratio := 2/1
}

end NUMINAMATH_CALUDE_a_initial_is_9000_l2255_225575


namespace NUMINAMATH_CALUDE_solve_equation_l2255_225599

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 1

-- State the theorem
theorem solve_equation (a : ℝ) (h1 : 0 < a) (h2 : a < 3) (h3 : f a = 7) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2255_225599


namespace NUMINAMATH_CALUDE_strawberry_harvest_l2255_225500

/-- Calculates the total number of strawberries harvested from a square garden -/
theorem strawberry_harvest (garden_side : ℝ) (plants_per_sqft : ℝ) (strawberries_per_plant : ℝ) :
  garden_side = 10 →
  plants_per_sqft = 5 →
  strawberries_per_plant = 12 →
  garden_side * garden_side * plants_per_sqft * strawberries_per_plant = 6000 := by
  sorry

#check strawberry_harvest

end NUMINAMATH_CALUDE_strawberry_harvest_l2255_225500


namespace NUMINAMATH_CALUDE_equation_solutions_l2255_225533

theorem equation_solutions :
  (∃ x : ℝ, 6 * x - 7 = 4 * x - 5 ∧ x = 1) ∧
  (∃ x : ℝ, (1/2) * x - 6 = (3/4) * x ∧ x = -24) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2255_225533


namespace NUMINAMATH_CALUDE_extreme_value_and_min_max_l2255_225586

/-- Function f(x) = 2x³ + ax² + bx + 1 -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

/-- Derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem extreme_value_and_min_max (a b : ℝ) : 
  f a b 1 = -6 ∧ f' a b 1 = 0 →
  a = 3 ∧ b = -12 ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x ≤ 21) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x ≥ -6) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x = 21) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x = -6) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_min_max_l2255_225586


namespace NUMINAMATH_CALUDE_min_sum_squares_l2255_225557

/-- B-neighborhood of A is defined as the solution set of |x - A| < B where A ∈ ℝ and B > 0 -/
def neighborhood (A : ℝ) (B : ℝ) : Set ℝ :=
  {x : ℝ | |x - A| < B}

theorem min_sum_squares (a b : ℝ) :
  neighborhood (a + b - 3) (a + b) = Set.Ioo (-3 : ℝ) 3 →
  ∃ (m : ℝ), m = 9/2 ∧ ∀ x y : ℝ, x^2 + y^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2255_225557


namespace NUMINAMATH_CALUDE_soap_discount_theorem_l2255_225560

/-- The original price of a bar of soap in yuan -/
def original_price : ℝ := 2

/-- The discount rate for the first method (applied to all bars except the first) -/
def discount_rate1 : ℝ := 0.3

/-- The discount rate for the second method (applied to all bars) -/
def discount_rate2 : ℝ := 0.2

/-- The cost of n bars using the first discount method -/
def cost1 (n : ℕ) : ℝ := original_price + (n - 1) * original_price * (1 - discount_rate1)

/-- The cost of n bars using the second discount method -/
def cost2 (n : ℕ) : ℝ := n * original_price * (1 - discount_rate2)

/-- The minimum number of bars needed for the first method to provide more discount -/
def min_bars : ℕ := 4

theorem soap_discount_theorem :
  ∀ n : ℕ, n ≥ min_bars → cost1 n < cost2 n ∧
  ∀ m : ℕ, m < min_bars → cost1 m ≥ cost2 m :=
sorry

end NUMINAMATH_CALUDE_soap_discount_theorem_l2255_225560


namespace NUMINAMATH_CALUDE_inequality_proof_l2255_225574

theorem inequality_proof (a b : ℝ) : (a^2 - 1) * (b^2 - 1) ≤ 0 → a^2 + b^2 - 1 - a^2*b^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2255_225574


namespace NUMINAMATH_CALUDE_not_divides_power_plus_one_l2255_225590

theorem not_divides_power_plus_one (n k : ℕ) (h1 : n = 2^2007 * k + 1) (h2 : Odd k) :
  ¬(n ∣ 2^(n - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_plus_one_l2255_225590


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2255_225559

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 9*n + 20 ≤ 0 ∧ n = 5 ∧ ∀ (m : ℤ), m^2 - 9*m + 20 ≤ 0 → m ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2255_225559


namespace NUMINAMATH_CALUDE_donny_spending_friday_sunday_l2255_225589

def monday_savings : ℝ := 15

def savings_increase_rate : ℝ := 0.1

def friday_spending_rate : ℝ := 0.5

def saturday_savings_decrease : ℝ := 0.2

def sunday_spending_rate : ℝ := 0.4

def tuesday_savings (monday : ℝ) : ℝ := monday * (1 + savings_increase_rate)

def wednesday_savings (tuesday : ℝ) : ℝ := tuesday * (1 + savings_increase_rate)

def thursday_savings (wednesday : ℝ) : ℝ := wednesday * (1 + savings_increase_rate)

def total_savings_thursday (mon tue wed thu : ℝ) : ℝ := mon + tue + wed + thu

def friday_spending (total : ℝ) : ℝ := total * friday_spending_rate

def saturday_savings (thursday : ℝ) : ℝ := thursday * (1 - saturday_savings_decrease)

def total_savings_saturday (friday_remaining saturday : ℝ) : ℝ := friday_remaining + saturday

def sunday_spending (total : ℝ) : ℝ := total * sunday_spending_rate

theorem donny_spending_friday_sunday : 
  let tue := tuesday_savings monday_savings
  let wed := wednesday_savings tue
  let thu := thursday_savings wed
  let total_thu := total_savings_thursday monday_savings tue wed thu
  let fri_spend := friday_spending total_thu
  let fri_remaining := total_thu - fri_spend
  let sat := saturday_savings thu
  let total_sat := total_savings_saturday fri_remaining sat
  let sun_spend := sunday_spending total_sat
  fri_spend + sun_spend = 55.13 := by sorry

end NUMINAMATH_CALUDE_donny_spending_friday_sunday_l2255_225589


namespace NUMINAMATH_CALUDE_conor_carrot_count_l2255_225565

/-- Represents the number of vegetables Conor can chop in a day -/
structure DailyVegetables where
  eggplants : ℕ
  carrots : ℕ
  potatoes : ℕ

/-- Represents Conor's weekly vegetable chopping -/
def WeeklyVegetables (d : DailyVegetables) (workDays : ℕ) : ℕ :=
  workDays * (d.eggplants + d.carrots + d.potatoes)

/-- Theorem stating the number of carrots Conor can chop in a day -/
theorem conor_carrot_count :
  ∀ (d : DailyVegetables),
    d.eggplants = 12 →
    d.potatoes = 8 →
    WeeklyVegetables d 4 = 116 →
    d.carrots = 9 := by
  sorry


end NUMINAMATH_CALUDE_conor_carrot_count_l2255_225565


namespace NUMINAMATH_CALUDE_factor_expression_l2255_225524

theorem factor_expression (b : ℝ) : 63 * b^2 + 189 * b = 63 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2255_225524


namespace NUMINAMATH_CALUDE_min_max_f_on_I_l2255_225583

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^2 * (x - 2)

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem min_max_f_on_I :
  ∃ (min max : ℝ), min = -64 ∧ max = 0 ∧
  (∀ x ∈ I, f x ≥ min) ∧
  (∀ x ∈ I, f x ≤ max) ∧
  (∃ x₁ ∈ I, f x₁ = min) ∧
  (∃ x₂ ∈ I, f x₂ = max) :=
sorry

end NUMINAMATH_CALUDE_min_max_f_on_I_l2255_225583


namespace NUMINAMATH_CALUDE_triple_overlap_area_is_six_l2255_225554

/-- Represents a rectangular carpet with width and length -/
structure Carpet where
  width : ℝ
  length : ℝ

/-- Represents the auditorium floor -/
structure Auditorium where
  width : ℝ
  length : ℝ

/-- Calculates the area of triple overlap given three carpets and an auditorium -/
def tripleOverlapArea (c1 c2 c3 : Carpet) (a : Auditorium) : ℝ :=
  sorry

/-- Theorem stating that the area of triple overlap is 6 square meters -/
theorem triple_overlap_area_is_six 
  (c1 : Carpet) 
  (c2 : Carpet) 
  (c3 : Carpet) 
  (a : Auditorium) 
  (h1 : c1.width = 6 ∧ c1.length = 8)
  (h2 : c2.width = 6 ∧ c2.length = 6)
  (h3 : c3.width = 5 ∧ c3.length = 7)
  (h4 : a.width = 10 ∧ a.length = 10) :
  tripleOverlapArea c1 c2 c3 a = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_overlap_area_is_six_l2255_225554


namespace NUMINAMATH_CALUDE_student_bicycle_speed_l2255_225513

/-- Given two students A and B traveling 12 km, where A's speed is 1.2 times B's,
    and A arrives 1/6 hour earlier, B's speed is 12 km/h. -/
theorem student_bicycle_speed (distance : ℝ) (speed_ratio : ℝ) (time_diff : ℝ) :
  distance = 12 →
  speed_ratio = 1.2 →
  time_diff = 1/6 →
  ∃ (speed_B : ℝ),
    distance / speed_B - distance / (speed_ratio * speed_B) = time_diff ∧
    speed_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_bicycle_speed_l2255_225513
