import Mathlib

namespace chord_length_midway_l73_7364

theorem chord_length_midway (r : ℝ) (x y : ℝ) : 
  (24 : ℝ) ^ 2 / 4 + x^2 = r^2 →
  (32 : ℝ) ^ 2 / 4 + y^2 = r^2 →
  x + y = 14 →
  let d := (x - y) / 2
  2 * Real.sqrt (r^2 - d^2) = 2 * Real.sqrt 249 := by sorry

end chord_length_midway_l73_7364


namespace range_of_r_l73_7336

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^6 + 6*x^3 + 9

-- State the theorem
theorem range_of_r :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ r x = y) ↔ y ≥ 9 :=
by sorry

end range_of_r_l73_7336


namespace square_fence_perimeter_l73_7343

/-- The number of posts in the fence -/
def num_posts : ℕ := 36

/-- The width of each post in inches -/
def post_width_inches : ℕ := 6

/-- The space between adjacent posts in feet -/
def space_between_posts : ℕ := 4

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- Conversion factor from inches to feet -/
def inches_to_feet : ℚ := 1 / 12

theorem square_fence_perimeter :
  let posts_per_side : ℕ := num_posts / num_sides
  let post_width_feet : ℚ := post_width_inches * inches_to_feet
  let gaps_per_side : ℕ := posts_per_side - 1
  let side_length : ℚ := gaps_per_side * space_between_posts + posts_per_side * post_width_feet
  num_sides * side_length = 130 := by sorry

end square_fence_perimeter_l73_7343


namespace base_b_not_divisible_by_five_l73_7387

theorem base_b_not_divisible_by_five (b : ℤ) (h : b ∈ ({3, 5, 7, 10, 12} : Set ℤ)) : 
  ¬ (5 ∣ ((b - 1)^2)) := by
sorry

end base_b_not_divisible_by_five_l73_7387


namespace divisible_by_four_even_equivalence_l73_7366

theorem divisible_by_four_even_equivalence :
  (∀ n : ℤ, 4 ∣ n → Even n) ↔ (∀ n : ℤ, ¬Even n → ¬(4 ∣ n)) := by sorry

end divisible_by_four_even_equivalence_l73_7366


namespace triangle_properties_l73_7333

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.c * Real.sin t.C - t.a * Real.sin t.A = (Real.sqrt 3 * t.c - t.b) * Real.sin t.B) :
  -- Part 1: Angle A is 30 degrees (π/6 radians)
  t.A = π / 6 ∧
  -- Part 2: If a = 1, the maximum area is (2 + √3) / 4
  (t.a = 1 → 
    ∃ (S : ℝ), S = (2 + Real.sqrt 3) / 4 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * t.b * t.c * Real.sin t.A → S' ≤ S) :=
by sorry

end triangle_properties_l73_7333


namespace jony_start_block_l73_7386

/-- Represents Jony's walk along Sunrise Boulevard -/
structure JonyWalk where
  walkTime : ℕ            -- Walking time in minutes
  speed : ℕ               -- Speed in meters per minute
  blockLength : ℕ         -- Length of each block in meters
  turnAroundBlock : ℕ     -- Block number where Jony turns around
  stopBlock : ℕ           -- Block number where Jony stops

/-- Calculates the starting block number for Jony's walk -/
def calculateStartBlock (walk : JonyWalk) : ℕ :=
  sorry

/-- Theorem stating that given the conditions of Jony's walk, his starting block is 10 -/
theorem jony_start_block :
  let walk : JonyWalk := {
    walkTime := 40,
    speed := 100,
    blockLength := 40,
    turnAroundBlock := 90,
    stopBlock := 70
  }
  calculateStartBlock walk = 10 := by
  sorry

end jony_start_block_l73_7386


namespace triangle_area_bound_l73_7352

theorem triangle_area_bound (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) / 4 ≤ Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_bound_l73_7352


namespace trapezoid_perimeter_l73_7385

/-- Represents a trapezoid with sides AB, BC, CD, and DA -/
structure Trapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + t.BC + t.CD + t.DA

/-- Theorem: Perimeter of a specific trapezoid ABCD -/
theorem trapezoid_perimeter (x y : ℝ) (hx : x ≠ 0) :
  ∃ (ABCD : Trapezoid),
    ABCD.AB = 2 * x ∧
    ABCD.CD = 4 * x ∧
    ABCD.BC = y ∧
    ABCD.DA = 2 * y ∧
    perimeter ABCD = 6 * x + 3 * y := by
  sorry

#check trapezoid_perimeter

end trapezoid_perimeter_l73_7385


namespace binomial_variance_example_l73_7320

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a random variable -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Given a random variable X following the binomial distribution B(6, 1/3), its variance D(X) is 4/3 -/
theorem binomial_variance_example :
  let X : BinomialDistribution := ⟨6, 1/3, by norm_num⟩
  variance X = 4/3 := by sorry

end binomial_variance_example_l73_7320


namespace moral_education_story_time_l73_7304

/-- Proves that telling a 7-minute "Moral Education Story" every week for 20 weeks equals 2 hours and 20 minutes -/
theorem moral_education_story_time :
  let story_duration : ℕ := 7  -- Duration of one story in minutes
  let weeks : ℕ := 20  -- Number of weeks
  let total_minutes : ℕ := story_duration * weeks
  let hours : ℕ := total_minutes / 60
  let remaining_minutes : ℕ := total_minutes % 60
  (hours = 2 ∧ remaining_minutes = 20) := by
  sorry


end moral_education_story_time_l73_7304


namespace expression_value_l73_7319

theorem expression_value : 
  (49 * 91^3 + 338 * 343^2) / (66^3 - 176 * 121) / (39^3 * 7^5 / 1331000) = 5 / 13 := by
  sorry

end expression_value_l73_7319


namespace product_remainder_mod_five_l73_7313

theorem product_remainder_mod_five : ∃ k : ℕ, 2532 * 3646 * 2822 * 3716 * 101 = 5 * k + 4 := by
  sorry

end product_remainder_mod_five_l73_7313


namespace sqrt_identity_l73_7390

theorem sqrt_identity (t : ℝ) : 
  Real.sqrt (t^6 + t^4 + t^2) = |t| * Real.sqrt (t^4 + t^2 + 1) := by
  sorry

end sqrt_identity_l73_7390


namespace quadratic_roots_relation_l73_7325

theorem quadratic_roots_relation (q : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^2 - 5*x₁ + q = 0) ∧ 
    (x₂^2 - 5*x₂ + q = 0) ∧ 
    (x₃^2 - 7*x₃ + 2*q = 0) ∧ 
    (x₄^2 - 7*x₄ + 2*q = 0) ∧ 
    (x₃ = 2*x₁ ∨ x₃ = 2*x₂ ∨ x₄ = 2*x₁ ∨ x₄ = 2*x₂)) →
  q = 6 := by
sorry

end quadratic_roots_relation_l73_7325


namespace factorial_divisor_differences_l73_7373

def divisors (n : ℕ) : List ℕ := sorry

def consecutive_differences (l : List ℕ) : List ℕ := sorry

def is_non_decreasing (l : List ℕ) : Prop := sorry

theorem factorial_divisor_differences (n : ℕ) :
  n ≥ 3 ∧ is_non_decreasing (consecutive_differences (divisors (n.factorial))) ↔ n = 3 ∨ n = 4 := by
  sorry

end factorial_divisor_differences_l73_7373


namespace speech_competition_probability_l73_7335

theorem speech_competition_probability 
  (m n : ℕ) 
  (prob_at_least_one_female : ℝ) 
  (h1 : prob_at_least_one_female = 4/5) :
  1 - prob_at_least_one_female = 1/5 := by
  sorry

end speech_competition_probability_l73_7335


namespace divisibility_by_203_l73_7305

theorem divisibility_by_203 (n : ℕ+) : 
  (2013^n.val - 1803^n.val - 1781^n.val + 1774^n.val) % 203 = 0 := by
sorry

end divisibility_by_203_l73_7305


namespace train_length_approx_100_l73_7314

/-- Calculates the length of a train given its speed, the time it takes to cross a platform, and the length of the platform. -/
def train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : ℝ :=
  speed * time - platform_length

/-- Theorem stating that a train with given parameters has a length of approximately 100 meters. -/
theorem train_length_approx_100 (speed : ℝ) (time : ℝ) (platform_length : ℝ) 
  (h1 : speed = 60 * 1000 / 3600) -- 60 km/hr converted to m/s
  (h2 : time = 14.998800095992321)
  (h3 : platform_length = 150) :
  ∃ ε > 0, |train_length speed time platform_length - 100| < ε :=
sorry

end train_length_approx_100_l73_7314


namespace book_pages_count_l73_7322

/-- Count the occurrences of digit 1 in a number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Count the occurrences of digit 1 in page numbers from 1 to n -/
def countOnesInPages (n : ℕ) : ℕ := sorry

/-- The number of pages in the book -/
def numPages : ℕ := 318

/-- The total count of digit 1 in the book's page numbers -/
def totalOnes : ℕ := 171

theorem book_pages_count :
  (countOnesInPages numPages = totalOnes) ∧ 
  (∀ m : ℕ, m < numPages → countOnesInPages m < totalOnes) := by sorry

end book_pages_count_l73_7322


namespace geometry_propositions_l73_7353

-- Define the basic types
variable (α β : Plane) (l m : Line)

-- Define the relationships
def perpendicular_to_plane (line : Line) (plane : Plane) : Prop := sorry
def contained_in_plane (line : Line) (plane : Plane) : Prop := sorry
def parallel_planes (plane1 plane2 : Plane) : Prop := sorry
def perpendicular_planes (plane1 plane2 : Plane) : Prop := sorry
def perpendicular_lines (line1 line2 : Line) : Prop := sorry
def parallel_lines (line1 line2 : Line) : Prop := sorry

-- State the theorem
theorem geometry_propositions 
  (h1 : perpendicular_to_plane l α) 
  (h2 : contained_in_plane m β) :
  (parallel_planes α β → perpendicular_lines l m) ∧ 
  ¬(perpendicular_planes α β → parallel_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) := by sorry

end geometry_propositions_l73_7353


namespace range_of_a_for_subset_l73_7324

/-- The solution set of x^2 - ax - x < 0 -/
def M (a : ℝ) : Set ℝ :=
  {x | x^2 - a*x - x < 0}

/-- The solution set of x^2 - 2x - 3 ≤ 0 -/
def N : Set ℝ :=
  {x | x^2 - 2*x - 3 ≤ 0}

/-- The theorem stating the range of a for which M(a) ⊆ N -/
theorem range_of_a_for_subset : 
  {a : ℝ | M a ⊆ N} = {a : ℝ | -2 ≤ a ∧ a ≤ 2} := by
sorry

end range_of_a_for_subset_l73_7324


namespace right_triangle_max_sin_product_l73_7355

theorem right_triangle_max_sin_product (A B C : Real) : 
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ -- Angles are non-negative
  A + B + C = π ∧ -- Sum of angles in a triangle
  C = π / 2 → -- Right angle condition
  ∀ (x y : Real), 0 ≤ x ∧ 0 ≤ y ∧ x + y = π / 2 → 
    Real.sin x * Real.sin y ≤ 1 / 2 :=
by sorry

end right_triangle_max_sin_product_l73_7355


namespace product_remainder_by_10_l73_7358

theorem product_remainder_by_10 : (2456 * 7294 * 91803) % 10 = 2 := by
  sorry

end product_remainder_by_10_l73_7358


namespace circle_tangents_theorem_l73_7350

/-- Given two circles with radii x and y touching a circle with radius R,
    and the distance between points of contact a, this theorem proves
    the squared lengths of their common tangents. -/
theorem circle_tangents_theorem
  (R x y a : ℝ)
  (h_pos_R : R > 0)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_pos_a : a > 0) :
  (∃ (l_ext : ℝ), l_ext^2 = (a/R)^2 * (R+x)*(R+y) ∨ l_ext^2 = (a/R)^2 * (R-x)*(R-y)) ∧
  (∃ (l_int : ℝ), l_int^2 = (a/R)^2 * (R+y)*(R-x)) :=
by sorry

end circle_tangents_theorem_l73_7350


namespace age_difference_l73_7374

theorem age_difference (C D m : ℕ) : 
  C = D + m →                    -- Chris is m years older than Daniel
  C - 1 = 3 * (D - 1) →          -- Last year Chris was 3 times as old as Daniel
  C * D = 72 →                   -- This year, the product of their ages is 72
  m = 9 := by
  sorry

end age_difference_l73_7374


namespace expression_evaluation_l73_7382

theorem expression_evaluation (a b : ℚ) (ha : a = 3/4) (hb : b = 4/3) :
  let numerator := (a/b + b/a + 2) * ((a+b)/(2*a) - b/(a+b))
  let denominator := (a + 2*b + b^2/a) * (a/(a+b) + b/(a-b))
  numerator / denominator = -7/24 := by
sorry

end expression_evaluation_l73_7382


namespace percentage_difference_l73_7371

theorem percentage_difference (y e w z : ℝ) (P : ℝ) : 
  w = e * (1 - P / 100) →
  e = y * 0.6 →
  z = y * 0.54 →
  z = w * (1 + 0.5000000000000002) →
  P = 40 := by
sorry

end percentage_difference_l73_7371


namespace complex_arithmetic_calculation_l73_7326

theorem complex_arithmetic_calculation : 
  ((15 - 2) + (4 / (1 / 2)) - (6 * 8)) * (100 - 24) / 38 = -54 := by
sorry

end complex_arithmetic_calculation_l73_7326


namespace root_in_interval_l73_7384

noncomputable def f (x : ℝ) := Real.log x + x - 2

theorem root_in_interval : ∃ (k : ℤ), ∃ (x₀ : ℝ),
  x₀ > 0 ∧ 
  f x₀ = 0 ∧
  x₀ > k ∧ 
  x₀ < k + 1 ∧
  k = 1 := by
  sorry

end root_in_interval_l73_7384


namespace boat_speed_ratio_l73_7356

theorem boat_speed_ratio (v : ℝ) (c : ℝ) (d : ℝ) 
  (hv : v = 24) -- Boat speed in still water
  (hc : c = 6)  -- River current speed
  (hd : d = 3)  -- Distance traveled downstream and upstream
  : (2 * d) / ((d / (v + c)) + (d / (v - c))) / v = 15 / 16 := by
  sorry

end boat_speed_ratio_l73_7356


namespace levels_beaten_l73_7340

theorem levels_beaten (total_levels : ℕ) (ratio : ℚ) : total_levels = 32 ∧ ratio = 3 / 1 → 
  ∃ (beaten : ℕ), beaten = 24 ∧ beaten * (1 + 1 / ratio) = total_levels := by
sorry

end levels_beaten_l73_7340


namespace certain_number_exists_l73_7389

theorem certain_number_exists : ∃ N : ℝ, (5/6 : ℝ) * N = (5/16 : ℝ) * N + 150 := by
  sorry

end certain_number_exists_l73_7389


namespace intersecting_circles_sum_l73_7381

/-- Given two intersecting circles with centers on the line x - y + c = 0 and
    intersection points A(1, 3) and B(m, -1), prove that m + c = -1 -/
theorem intersecting_circles_sum (m c : ℝ) : 
  (∃ (circle1 circle2 : Set (ℝ × ℝ)),
    (∃ (center1 center2 : ℝ × ℝ),
      center1 ∈ circle1 ∧ center2 ∈ circle2 ∧
      center1.1 - center1.2 + c = 0 ∧ center2.1 - center2.2 + c = 0) ∧
    (1, 3) ∈ circle1 ∩ circle2 ∧ (m, -1) ∈ circle1 ∩ circle2) →
  m + c = -1 := by
sorry

end intersecting_circles_sum_l73_7381


namespace subtraction_multiplication_theorem_l73_7354

theorem subtraction_multiplication_theorem : ((3.65 - 1.27) * 2) = 4.76 := by
  sorry

end subtraction_multiplication_theorem_l73_7354


namespace square_diagonals_equal_l73_7397

/-- A structure representing a parallelogram -/
structure Parallelogram :=
  (diagonals_equal : Bool)

/-- A structure representing a square, which is a special case of a parallelogram -/
structure Square extends Parallelogram

/-- Theorem stating that the diagonals of a parallelogram are equal -/
axiom parallelogram_diagonals_equal :
  ∀ (p : Parallelogram), p.diagonals_equal = true

/-- Theorem stating that a square is a parallelogram -/
axiom square_is_parallelogram :
  ∀ (s : Square), ∃ (p : Parallelogram), s = ⟨p⟩

/-- Theorem to prove: The diagonals of a square are equal -/
theorem square_diagonals_equal (s : Square) :
  s.diagonals_equal = true := by sorry

end square_diagonals_equal_l73_7397


namespace chord_length_when_m_1_shortest_chord_line_equation_l73_7331

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 11 = 0

-- Define the line l
def line_l (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the chord length function
noncomputable def chord_length (m : ℝ) : ℝ := sorry

-- Define the shortest chord condition
def is_shortest_chord (m : ℝ) : Prop := 
  ∀ m', chord_length m ≤ chord_length m'

-- Theorem 1: Chord length when m = 1
theorem chord_length_when_m_1 : 
  chord_length 1 = 6 * Real.sqrt 13 / 13 := sorry

-- Theorem 2: Equation of line l for shortest chord
theorem shortest_chord_line_equation :
  ∃ m, is_shortest_chord m ∧ 
    ∀ x y, line_l m x y ↔ x - y - 2 = 0 := sorry

end chord_length_when_m_1_shortest_chord_line_equation_l73_7331


namespace tourists_scientific_correct_l73_7347

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- The number of tourists per year -/
def tourists_per_year : ℕ := 876000

/-- The scientific notation representation of the number of tourists -/
def tourists_scientific : ScientificNotation where
  coefficient := 8.76
  exponent := 5
  one_le_coeff_lt_ten := by sorry

/-- Theorem stating that the scientific notation representation is correct -/
theorem tourists_scientific_correct : 
  (tourists_scientific.coefficient * (10 : ℝ) ^ tourists_scientific.exponent) = tourists_per_year := by sorry

end tourists_scientific_correct_l73_7347


namespace total_scissors_is_86_l73_7365

/-- Calculates the total number of scissors after changes in two drawers -/
def totalScissorsAfterChanges (
  initialScissors1 : ℕ) (initialScissors2 : ℕ) 
  (addedScissors1 : ℕ) (addedScissors2 : ℕ) : ℕ :=
  (initialScissors1 + addedScissors1) + (initialScissors2 + addedScissors2)

/-- Proves that the total number of scissors after changes is 86 -/
theorem total_scissors_is_86 :
  totalScissorsAfterChanges 39 27 13 7 = 86 := by
  sorry

end total_scissors_is_86_l73_7365


namespace food_drive_problem_l73_7328

/-- Represents the food drive problem in Ms. Perez's class -/
theorem food_drive_problem (total_students : ℕ) (half_students_12_cans : ℕ) (students_4_cans : ℕ) (total_cans : ℕ) :
  total_students = 30 →
  half_students_12_cans = total_students / 2 →
  students_4_cans = 13 →
  total_cans = 232 →
  half_students_12_cans * 12 + students_4_cans * 4 = total_cans →
  total_students - (half_students_12_cans + students_4_cans) = 2 :=
by sorry

end food_drive_problem_l73_7328


namespace x_minus_y_equals_nine_l73_7318

theorem x_minus_y_equals_nine (x y : ℕ) (h1 : 3^x * 4^y = 19683) (h2 : x = 9) :
  x - y = 9 := by
  sorry

end x_minus_y_equals_nine_l73_7318


namespace min_value_range_l73_7312

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- State the theorem
theorem min_value_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 a, f x ≥ f a) → a ∈ Set.Ioo 1 3 := by sorry

end min_value_range_l73_7312


namespace trays_per_trip_is_eight_l73_7329

-- Define the problem parameters
def trays_table1 : ℕ := 27
def trays_table2 : ℕ := 5
def total_trips : ℕ := 4

-- Define the total number of trays
def total_trays : ℕ := trays_table1 + trays_table2

-- Theorem statement
theorem trays_per_trip_is_eight :
  total_trays / total_trips = 8 :=
by
  sorry

end trays_per_trip_is_eight_l73_7329


namespace inequality_system_solution_l73_7396

theorem inequality_system_solution (x : ℝ) : 
  (7 - 2*(x + 1) ≥ 1 - 6*x ∧ (1 + 2*x) / 3 > x - 1) ↔ -1 ≤ x ∧ x < 4 := by
  sorry

end inequality_system_solution_l73_7396


namespace imaginary_unit_power_l73_7349

theorem imaginary_unit_power (i : ℂ) : i * i = -1 → i^2015 = -i := by
  sorry

end imaginary_unit_power_l73_7349


namespace nine_oclock_right_angle_l73_7308

/-- The angle between clock hands at a given hour -/
def clock_angle (hour : ℕ) : ℝ :=
  sorry

/-- A right angle is 90 degrees -/
def is_right_angle (angle : ℝ) : Prop :=
  angle = 90

theorem nine_oclock_right_angle : is_right_angle (clock_angle 9) := by
  sorry

end nine_oclock_right_angle_l73_7308


namespace calculate_face_value_l73_7344

/-- The relationship between banker's discount, true discount, and face value -/
def bankers_discount_relation (bd td fv : ℚ) : Prop :=
  bd = td + td^2 / fv

/-- Given the banker's discount and true discount, calculate the face value -/
theorem calculate_face_value (bd td : ℚ) (h : bankers_discount_relation bd td 300) :
  bd = 72 ∧ td = 60 → 300 = 300 := by sorry

end calculate_face_value_l73_7344


namespace sin_equal_of_sum_pi_l73_7399

theorem sin_equal_of_sum_pi (α β : Real) (h : α + β = Real.pi) : Real.sin α = Real.sin β := by
  sorry

end sin_equal_of_sum_pi_l73_7399


namespace bowling_ball_weight_l73_7327

theorem bowling_ball_weight :
  ∀ (ball_weight canoe_weight : ℚ),
    9 * ball_weight = 4 * canoe_weight →
    3 * canoe_weight = 112 →
    ball_weight = 448 / 27 := by
  sorry

end bowling_ball_weight_l73_7327


namespace no_solution_exists_l73_7301

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_solution_exists : ∀ n : ℕ, n * sum_of_digits n ≠ 100200300 := by
  sorry

end no_solution_exists_l73_7301


namespace cricket_team_right_handed_players_l73_7341

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 67)
  (h2 : throwers = 37)
  (h3 : (total_players - throwers) % 3 = 0)
  (h4 : throwers ≤ total_players) :
  throwers + ((total_players - throwers) * 2 / 3) = 57 := by
sorry

end cricket_team_right_handed_players_l73_7341


namespace quadratic_always_positive_range_l73_7309

theorem quadratic_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end quadratic_always_positive_range_l73_7309


namespace shelly_friends_in_classes_l73_7383

/-- The number of friends Shelly made in classes -/
def friends_in_classes : ℕ := sorry

/-- The number of friends Shelly made in after-school clubs -/
def friends_in_clubs : ℕ := sorry

/-- The amount of thread needed for each keychain in inches -/
def thread_per_keychain : ℕ := 12

/-- The total amount of thread needed in inches -/
def total_thread : ℕ := 108

/-- Theorem stating that Shelly made 6 friends in classes -/
theorem shelly_friends_in_classes : 
  friends_in_classes = 6 ∧
  friends_in_clubs = friends_in_classes / 2 ∧
  friends_in_classes * thread_per_keychain + friends_in_clubs * thread_per_keychain = total_thread :=
sorry

end shelly_friends_in_classes_l73_7383


namespace triangle_area_l73_7321

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Vectors m and n
  let m := (Real.sin C, Real.sin B * Real.cos A)
  let n := (b, 2 * c)
  -- m · n = 0
  m.1 * n.1 + m.2 * n.2 = 0 →
  -- a = 2√3
  a = 2 * Real.sqrt 3 →
  -- sin B + sin C = 1
  Real.sin B + Real.sin C = 1 →
  -- Area of triangle ABC is √3
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
by sorry

end triangle_area_l73_7321


namespace autumn_pencils_l73_7339

def pencil_count (initial misplaced broken found bought : ℕ) : ℕ :=
  initial - misplaced - broken + found + bought

theorem autumn_pencils : pencil_count 20 7 3 4 2 = 16 := by
  sorry

end autumn_pencils_l73_7339


namespace jennifer_fruits_left_l73_7361

def fruits_left (pears oranges apples cherries grapes : ℕ) 
  (pears_given oranges_given apples_given cherries_given grapes_given : ℕ) : ℕ :=
  (pears - pears_given) + (oranges - oranges_given) + (apples - apples_given) + 
  (cherries - cherries_given) + (grapes - grapes_given)

theorem jennifer_fruits_left : 
  let pears : ℕ := 15
  let oranges : ℕ := 30
  let apples : ℕ := 2 * pears
  let cherries : ℕ := oranges / 2
  let grapes : ℕ := 3 * apples
  fruits_left pears oranges apples cherries grapes 3 5 5 7 3 = 157 := by
  sorry

end jennifer_fruits_left_l73_7361


namespace polynomial_remainder_theorem_l73_7379

theorem polynomial_remainder_theorem (a b : ℚ) : 
  let f : ℚ → ℚ := λ x ↦ a * x^3 - 6 * x^2 + b * x - 5
  (f 2 = 3 ∧ f (-1) = 7) → (a = -2/3 ∧ b = -52/3) := by
  sorry

end polynomial_remainder_theorem_l73_7379


namespace complex_arithmetic_expression_result_l73_7338

theorem complex_arithmetic_expression_result : 
  let expr := 3034 - ((1002 / 20.04) * (43.8 - 9.2^2) + Real.sqrt 144) / (3.58 * (76 - 8.23^3))
  ∃ ε > 0, abs (expr - 1.17857142857) < ε :=
by sorry

end complex_arithmetic_expression_result_l73_7338


namespace product_of_digits_not_divisible_by_5_l73_7388

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem product_of_digits_not_divisible_by_5 (numbers : List ℕ) :
  numbers = [3640, 3855, 3922, 4025, 4120] →
  (∃ n ∈ numbers, ¬ is_divisible_by_5 n) →
  (∃ n ∈ numbers, ¬ is_divisible_by_5 n ∧ hundreds_digit n * tens_digit n = 18) :=
by sorry

end product_of_digits_not_divisible_by_5_l73_7388


namespace derivative_f_at_zero_l73_7351

/-- The function f(x) = x(x-1)(x-2) -/
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2)

/-- The theorem stating that the derivative of f at x=0 is 2 -/
theorem derivative_f_at_zero : 
  deriv f 0 = 2 := by sorry

end derivative_f_at_zero_l73_7351


namespace chocolate_sales_l73_7302

theorem chocolate_sales (C S : ℝ) (n : ℕ) 
  (h1 : 81 * C = n * S)  -- Cost price of 81 chocolates equals selling price of n chocolates
  (h2 : S = 1.8 * C)     -- Selling price is 1.8 times the cost price (derived from 80% gain)
  : n = 45 := by
  sorry

end chocolate_sales_l73_7302


namespace triangle_side_length_l73_7306

/-- Given a triangle ABC where sin A, sin B, sin C form an arithmetic sequence,
    B = 30°, and the area is 3/2, prove that the length of side b is √3 + 1. -/
theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c →
  -- sin A, sin B, sin C form an arithmetic sequence
  2 * Real.sin B = Real.sin A + Real.sin C →
  -- B = 30°
  B = π / 6 →
  -- Area of triangle ABC is 3/2
  1/2 * a * c * Real.sin B = 3/2 →
  -- b is opposite to angle B
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  -- Conclusion: length of side b is √3 + 1
  b = Real.sqrt 3 + 1 := by
sorry

end triangle_side_length_l73_7306


namespace bill_left_with_411_l73_7367

/-- Calculates the amount of money Bill is left with after all transactions and expenses -/
def billsRemainingMoney : ℝ :=
  let merchantA_sale := 8 * 9
  let merchantB_sale := 15 * 11
  let sheriff_fine := 80
  let merchantC_sale := 25 * 8
  let protection_cost := 30
  let passerby_sale := 12 * 7
  
  let total_earnings := merchantA_sale + merchantB_sale + merchantC_sale + passerby_sale
  let total_expenses := sheriff_fine + protection_cost
  
  total_earnings - total_expenses

/-- Theorem stating that Bill is left with $411 after all transactions and expenses -/
theorem bill_left_with_411 : billsRemainingMoney = 411 := by
  sorry

end bill_left_with_411_l73_7367


namespace diophantine_equation_implication_l73_7363

-- Define the property of not being a perfect square
def NotPerfectSquare (n : ℤ) : Prop := ∀ m : ℤ, n ≠ m^2

-- Define a nontrivial integer solution
def HasNontrivialSolution (f : ℤ → ℤ → ℤ → ℤ) : Prop :=
  ∃ x y z : ℤ, f x y z = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)

-- Define a nontrivial integer solution for 4 variables
def HasNontrivialSolution4 (f : ℤ → ℤ → ℤ → ℤ → ℤ) : Prop :=
  ∃ x y z w : ℤ, f x y z w = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ w ≠ 0)

theorem diophantine_equation_implication (a b : ℤ) 
  (ha : NotPerfectSquare a) (hb : NotPerfectSquare b)
  (h : HasNontrivialSolution4 (fun x y z w => x^2 - a*y^2 - b*z^2 + a*b*w^2)) :
  HasNontrivialSolution (fun x y z => x^2 - a*y^2 - b*z^2) :=
by sorry

end diophantine_equation_implication_l73_7363


namespace flower_shop_ratio_l73_7357

/-- Flower shop problem -/
theorem flower_shop_ratio : 
  ∀ (roses lilacs gardenias : ℕ),
  roses = 3 * lilacs →
  lilacs = 10 →
  roses + lilacs + gardenias = 45 →
  gardenias / lilacs = 1 / 2 := by
  sorry

end flower_shop_ratio_l73_7357


namespace sum_product_identity_l73_7391

theorem sum_product_identity (a b : ℝ) (h : a + b = a * b) :
  (a^3 + b^3 - a^3 * b^3)^3 + 27 * a^6 * b^6 = 0 := by
  sorry

end sum_product_identity_l73_7391


namespace hall_reunion_attendance_l73_7334

theorem hall_reunion_attendance (total : ℕ) (oates : ℕ) (both : ℕ) (hall : ℕ) : 
  total = 150 → oates = 70 → both = 28 → total = oates + hall - both → hall = 108 := by
  sorry

end hall_reunion_attendance_l73_7334


namespace seashell_count_l73_7317

def initial_seashells (name : String) : ℕ :=
  match name with
  | "Henry" => 11
  | "John" => 24
  | "Adam" => 17
  | "Leo" => 83 - (11 + 24 + 17)
  | _ => 0

def final_seashells (name : String) : ℕ :=
  match name with
  | "Henry" => initial_seashells "Henry" + 3
  | "John" => initial_seashells "John" - 5
  | "Adam" => initial_seashells "Adam"
  | "Leo" => initial_seashells "Leo" - (initial_seashells "Leo" / 10 * 4) + 5
  | _ => 0

theorem seashell_count :
  final_seashells "Henry" + final_seashells "John" + 
  final_seashells "Adam" + final_seashells "Leo" = 74 :=
by sorry

end seashell_count_l73_7317


namespace cost_price_calculation_l73_7311

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 1110 ∧ profit_percentage = 20 → 
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 925 := by
sorry

end cost_price_calculation_l73_7311


namespace max_consecutive_positive_terms_l73_7303

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = a (n - 1) + a (n + 2)

/-- The property that a sequence has k consecutive positive terms starting from index n -/
def HasConsecutivePositiveTerms (a : ℕ → ℝ) (n k : ℕ) : Prop :=
  ∀ i : ℕ, i ∈ Finset.range k → a (n + i) > 0

/-- The main theorem stating that the maximum number of consecutive positive terms is 5 -/
theorem max_consecutive_positive_terms
  (a : ℕ → ℝ) (h : RecurrenceSequence a) :
  (∃ n k : ℕ, k > 5 ∧ HasConsecutivePositiveTerms a n k) → False :=
sorry

end max_consecutive_positive_terms_l73_7303


namespace min_white_surface_fraction_l73_7330

/-- Represents a cube with given edge length -/
structure Cube where
  edge_length : ℕ

/-- Represents the large cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cubes : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.edge_length^2

/-- Theorem: The minimum fraction of white surface area in the described cube configuration is 1/12 -/
theorem min_white_surface_fraction (lc : LargeCube) 
  (h1 : lc.edge_length = 4)
  (h2 : lc.small_cubes = 64)
  (h3 : lc.red_cubes = 48)
  (h4 : lc.white_cubes = 16) :
  ∃ (white_area : ℕ), 
    white_area ≤ lc.white_cubes ∧ 
    (white_area : ℚ) / (surface_area ⟨lc.edge_length⟩ : ℚ) = 1/12 ∧
    ∀ (other_white_area : ℕ), 
      other_white_area ≤ lc.white_cubes → 
      (other_white_area : ℚ) / (surface_area ⟨lc.edge_length⟩ : ℚ) ≥ 1/12 := by
  sorry

end min_white_surface_fraction_l73_7330


namespace sum_with_radical_conjugate_l73_7378

theorem sum_with_radical_conjugate : 
  let x : ℝ := 16 - Real.sqrt 2023
  let y : ℝ := 16 + Real.sqrt 2023
  x + y = 32 := by sorry

end sum_with_radical_conjugate_l73_7378


namespace smaller_cube_edge_length_l73_7392

theorem smaller_cube_edge_length :
  ∀ (s : ℝ),
  (8 : ℝ) * s^3 = 1000 →
  s = 5 :=
by sorry

end smaller_cube_edge_length_l73_7392


namespace smallest_valid_n_l73_7345

def is_valid (n : ℕ) : Prop :=
  ∀ m : ℕ+, ∃ S : Finset ℕ, S ⊆ Finset.range n ∧ 
    (S.prod id : ℕ) ≡ m [ZMOD 100]

theorem smallest_valid_n :
  is_valid 17 ∧ ∀ k < 17, ¬ is_valid k :=
sorry

end smallest_valid_n_l73_7345


namespace cylinder_max_lateral_area_l73_7315

theorem cylinder_max_lateral_area (sphere_area : ℝ) (h_sphere_area : sphere_area = 20 * Real.pi) :
  let R := (sphere_area / (4 * Real.pi)) ^ (1/2)
  ∃ (r l : ℝ), r > 0 ∧ l > 0 ∧ 
    r^2 + (l/2)^2 = R^2 ∧ 
    ∀ (r' l' : ℝ), r' > 0 → l' > 0 → r'^2 + (l'/2)^2 = R^2 → 
      2 * Real.pi * r * l ≤ 2 * Real.pi * r' * l' :=
by sorry

end cylinder_max_lateral_area_l73_7315


namespace calculate_expression_l73_7395

theorem calculate_expression : (-3)^0 + Real.sqrt 8 + (-3)^2 - 4 * (Real.sqrt 2 / 2) = 10 := by
  sorry

end calculate_expression_l73_7395


namespace jennys_score_is_14_total_questions_correct_l73_7398

/-- Represents a quiz with a specific scoring system -/
structure Quiz where
  totalQuestions : ℕ
  correctAnswers : ℕ
  incorrectAnswers : ℕ
  unansweredQuestions : ℕ
  correctPoints : ℚ
  incorrectPoints : ℚ

/-- Calculates the total score for a given quiz -/
def calculateScore (q : Quiz) : ℚ :=
  q.correctPoints * q.correctAnswers + q.incorrectPoints * q.incorrectAnswers

/-- Jenny's quiz results -/
def jennysQuiz : Quiz :=
  { totalQuestions := 25
    correctAnswers := 16
    incorrectAnswers := 4
    unansweredQuestions := 5
    correctPoints := 1
    incorrectPoints := -1/2 }

/-- Theorem stating that Jenny's quiz score is 14 -/
theorem jennys_score_is_14 : calculateScore jennysQuiz = 14 := by
  sorry

/-- Theorem verifying the total number of questions -/
theorem total_questions_correct :
  jennysQuiz.correctAnswers + jennysQuiz.incorrectAnswers + jennysQuiz.unansweredQuestions =
  jennysQuiz.totalQuestions := by
  sorry

end jennys_score_is_14_total_questions_correct_l73_7398


namespace circle_center_sum_l73_7393

/-- Given a circle with equation x^2 + y^2 = 4x - 12y - 8, 
    the sum of the coordinates of its center is -4. -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 12*y - 8) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 8) ∧ h + k = -4) :=
by sorry

end circle_center_sum_l73_7393


namespace parallelogram_area_equals_rectangle_area_l73_7310

/-- Represents a rectangle with a given base and area -/
structure Rectangle where
  base : ℝ
  area : ℝ

/-- Represents a parallelogram with a given base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Theorem: Given a rectangle with base 6 and area 24, and a parallelogram sharing the same base and height,
    the area of the parallelogram is 24 -/
theorem parallelogram_area_equals_rectangle_area 
  (rect : Rectangle) 
  (para : Parallelogram) 
  (h1 : rect.base = 6) 
  (h2 : rect.area = 24) 
  (h3 : para.base = rect.base) 
  (h4 : para.height = rect.area / rect.base) : 
  para.base * para.height = 24 := by
  sorry

#check parallelogram_area_equals_rectangle_area

end parallelogram_area_equals_rectangle_area_l73_7310


namespace inverse_g_at_neg43_l73_7368

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_g_at_neg43 :
  Function.invFun g (-43) = -2 :=
sorry

end inverse_g_at_neg43_l73_7368


namespace inequality_proof_l73_7307

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  1 / (1/a + 1/b) + 1 / (1/c + 1/d) ≤ 1 / (1/(a+c) + 1/(b+d)) := by
  sorry

end inequality_proof_l73_7307


namespace brick_height_proof_l73_7337

/-- Proves that the height of each brick is 6 cm given the wall and brick dimensions --/
theorem brick_height_proof (wall_length wall_width wall_height : ℝ)
                           (brick_length brick_width : ℝ)
                           (num_bricks : ℕ) :
  wall_length = 700 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 5600 →
  ∃ (h : ℝ), h = 6 ∧ 
    wall_length * wall_width * wall_height = 
    num_bricks * brick_length * brick_width * h :=
by
  sorry

end brick_height_proof_l73_7337


namespace new_average_price_six_toys_average_l73_7342

def average_price (n : ℕ) (total_cost : ℚ) : ℚ := total_cost / n

theorem new_average_price 
  (n : ℕ) 
  (old_avg : ℚ) 
  (additional_cost : ℚ) : 
  average_price (n + 1) (n * old_avg + additional_cost) = 
    (n * old_avg + additional_cost) / (n + 1) :=
by
  sorry

theorem six_toys_average 
  (dhoni_toys : ℕ) 
  (dhoni_avg : ℚ) 
  (david_toy_price : ℚ) 
  (h1 : dhoni_toys = 5) 
  (h2 : dhoni_avg = 10) 
  (h3 : david_toy_price = 16) :
  average_price (dhoni_toys + 1) (dhoni_toys * dhoni_avg + david_toy_price) = 11 :=
by
  sorry

end new_average_price_six_toys_average_l73_7342


namespace polynomial_expansion_properties_l73_7369

theorem polynomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) ∧
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) := by
  sorry

end polynomial_expansion_properties_l73_7369


namespace equation_solution_l73_7380

theorem equation_solution :
  ∃ x : ℚ, x - 2 ≠ 0 ∧ (2 / (x - 2) = (1 + x) / (x - 2) + 1) ∧ x = 3 / 2 := by
  sorry

end equation_solution_l73_7380


namespace quadratic_inequality_range_l73_7316

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end quadratic_inequality_range_l73_7316


namespace library_book_count_l73_7359

/-- The number of books in the library after a series of transactions --/
def books_in_library (initial : ℕ) (taken_tuesday : ℕ) (returned_wednesday : ℕ) (taken_thursday : ℕ) : ℕ :=
  initial - taken_tuesday + returned_wednesday - taken_thursday

/-- Theorem: The number of books in the library after the given transactions is 150 --/
theorem library_book_count : 
  books_in_library 250 120 35 15 = 150 := by
  sorry

#eval books_in_library 250 120 35 15

end library_book_count_l73_7359


namespace sufficient_not_necessary_l73_7360

theorem sufficient_not_necessary (a b : ℝ) :
  (0 < a ∧ a < b) → (1/4 : ℝ)^a > (1/4 : ℝ)^b ∧
  ∃ a' b' : ℝ, (1/4 : ℝ)^a' > (1/4 : ℝ)^b' ∧ ¬(0 < a' ∧ a' < b') :=
by sorry

end sufficient_not_necessary_l73_7360


namespace trip_price_calculation_egypt_trip_price_l73_7376

theorem trip_price_calculation (num_people : ℕ) (discount_per_person : ℕ) (total_cost_after_discount : ℕ) : ℕ :=
  let total_discount := num_people * discount_per_person
  let total_cost_before_discount := total_cost_after_discount + total_discount
  let original_price_per_person := total_cost_before_discount / num_people
  original_price_per_person

theorem egypt_trip_price : 
  trip_price_calculation 2 14 266 = 147 := by
  sorry

end trip_price_calculation_egypt_trip_price_l73_7376


namespace c2h5cl_formed_equals_c2h6_used_l73_7394

-- Define the chemical reaction
structure Reaction where
  c2h6 : ℝ
  cl2 : ℝ
  c2h5cl : ℝ
  hcl : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.c2h6 = r.cl2 ∧ r.c2h6 = r.c2h5cl ∧ r.c2h6 = r.hcl

-- Theorem statement
theorem c2h5cl_formed_equals_c2h6_used 
  (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : r.c2h6 = 3) 
  (h3 : r.c2h5cl = 3) : 
  r.c2h5cl = r.c2h6 := by
  sorry


end c2h5cl_formed_equals_c2h6_used_l73_7394


namespace carla_leaf_collection_l73_7372

/-- Represents the number of items Carla needs to collect each day -/
def daily_items : ℕ := 5

/-- Represents the number of days Carla has to collect items -/
def total_days : ℕ := 10

/-- Represents the number of bugs Carla needs to collect -/
def bugs_to_collect : ℕ := 20

/-- Calculates the total number of items Carla needs to collect -/
def total_items : ℕ := daily_items * total_days

/-- Calculates the number of leaves Carla needs to collect -/
def leaves_to_collect : ℕ := total_items - bugs_to_collect

theorem carla_leaf_collection :
  leaves_to_collect = 30 := by sorry

end carla_leaf_collection_l73_7372


namespace car_truck_distance_difference_l73_7375

theorem car_truck_distance_difference 
  (truck_distance : ℝ) 
  (truck_time : ℝ) 
  (car_time : ℝ) 
  (speed_difference : ℝ) 
  (h1 : truck_distance = 296)
  (h2 : truck_time = 8)
  (h3 : car_time = 5.5)
  (h4 : speed_difference = 18) : 
  let truck_speed := truck_distance / truck_time
  let car_speed := truck_speed + speed_difference
  let car_distance := car_speed * car_time
  car_distance - truck_distance = 6.5 := by
sorry

end car_truck_distance_difference_l73_7375


namespace min_value_of_z_l73_7362

variable (a b x : ℝ)
variable (h : a ≠ b)

def z (x : ℝ) : ℝ := (x - a)^3 + (x - b)^3

theorem min_value_of_z :
  ∃ (x : ℝ), ∀ (y : ℝ), z a b x ≤ z a b y ↔ x = (a + b) / 2 :=
sorry

end min_value_of_z_l73_7362


namespace min_value_fraction_sum_l73_7346

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (1/a + 2/b) ≥ 9 := by
  sorry

end min_value_fraction_sum_l73_7346


namespace fourth_sunday_january_l73_7370

-- Define the year N
def N : ℕ := sorry

-- Define the day of the week as an enumeration
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define a function to determine if a year is a leap year
def isLeapYear (year : ℕ) : Bool := sorry

-- Define a function to get the next day of the week
def nextDay (day : DayOfWeek) : DayOfWeek := sorry

-- Define a function to add days to a given day of the week
def addDays (start : DayOfWeek) (days : ℕ) : DayOfWeek := sorry

-- State the theorem
theorem fourth_sunday_january (h1 : 2000 < N ∧ N < 2100)
  (h2 : addDays DayOfWeek.Tuesday 364 = DayOfWeek.Tuesday)
  (h3 : addDays (nextDay (addDays DayOfWeek.Tuesday 364)) 730 = DayOfWeek.Friday)
  : addDays DayOfWeek.Saturday 22 = DayOfWeek.Sunday := by
  sorry

end fourth_sunday_january_l73_7370


namespace correct_total_amount_paid_l73_7300

/-- Calculates the total amount paid for fruits with discounts --/
def totalAmountPaid (
  peachPrice peachCount peachDiscountThreshold peachDiscount : ℚ)
  (applePrice appleCount appleDiscountThreshold appleDiscount : ℚ)
  (orangePrice orangeCount orangeDiscountThreshold orangeDiscount : ℚ)
  (grapefruitPrice grapefruitCount grapefruitDiscountThreshold grapefruitDiscount : ℚ)
  (bundleDiscountThreshold1 bundleDiscountThreshold2 bundleDiscountThreshold3 bundleDiscount : ℚ) : ℚ :=
  let peachTotal := peachPrice * peachCount
  let appleTotal := applePrice * appleCount
  let orangeTotal := orangePrice * orangeCount
  let grapefruitTotal := grapefruitPrice * grapefruitCount
  let peachDiscountTimes := (peachTotal / peachDiscountThreshold).floor
  let appleDiscountTimes := (appleTotal / appleDiscountThreshold).floor
  let orangeDiscountTimes := (orangeTotal / orangeDiscountThreshold).floor
  let grapefruitDiscountTimes := (grapefruitTotal / grapefruitDiscountThreshold).floor
  let totalBeforeDiscount := peachTotal + appleTotal + orangeTotal + grapefruitTotal
  let individualDiscounts := peachDiscountTimes * peachDiscount + 
                             appleDiscountTimes * appleDiscount + 
                             orangeDiscountTimes * orangeDiscount + 
                             grapefruitDiscountTimes * grapefruitDiscount
  let bundleDiscountApplied := if peachCount ≥ bundleDiscountThreshold1 ∧ 
                                  appleCount ≥ bundleDiscountThreshold2 ∧ 
                                  orangeCount ≥ bundleDiscountThreshold3 
                               then bundleDiscount else 0
  totalBeforeDiscount - individualDiscounts - bundleDiscountApplied

theorem correct_total_amount_paid : 
  totalAmountPaid 0.4 400 10 2 0.6 150 15 3 0.5 200 7 1.5 1 80 20 4 100 50 100 10 = 333 := by
  sorry


end correct_total_amount_paid_l73_7300


namespace linear_correlation_proof_l73_7348

/-- Determines if two variables are linearly correlated based on the correlation coefficient and critical value -/
def are_linearly_correlated (r : ℝ) (r_critical : ℝ) : Prop :=
  |r| > r_critical

/-- Theorem stating that given conditions lead to linear correlation -/
theorem linear_correlation_proof (r r_critical : ℝ) 
  (h1 : r = -0.9362)
  (h2 : r_critical = 0.8013) :
  are_linearly_correlated r r_critical :=
by
  sorry

#check linear_correlation_proof

end linear_correlation_proof_l73_7348


namespace amy_red_balloons_l73_7332

theorem amy_red_balloons (total green blue : ℕ) (h1 : total = 67) (h2 : green = 17) (h3 : blue = 21) : total - green - blue = 29 := by
  sorry

end amy_red_balloons_l73_7332


namespace sequence_properties_l73_7377

def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * a n + n

def b (n : ℕ) (a : ℕ → ℝ) : ℝ := n * (1 - a n)

def geometric_sequence (u : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, u (n + 1) = r * u n

def sum_of_sequence (u : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum u

theorem sequence_properties (a : ℕ → ℝ) :
  (∀ n : ℕ, S n a = 2 * a n + n) →
  (geometric_sequence (λ n => a n - 1)) ∧
  (∀ n : ℕ, sum_of_sequence (b · a) n = (n - 1) * 2^(n + 1) + 2) :=
by sorry

end sequence_properties_l73_7377


namespace tire_cost_l73_7323

theorem tire_cost (n : ℕ+) (total_cost battery_cost : ℚ) 
  (h1 : total_cost = 224)
  (h2 : battery_cost = 56) :
  (total_cost - battery_cost) / n = (224 - 56) / n :=
by sorry

end tire_cost_l73_7323
