import Mathlib

namespace modular_inverse_of_3_mod_199_l996_99672

theorem modular_inverse_of_3_mod_199 : ∃ x : ℕ, 0 < x ∧ x < 199 ∧ (3 * x) % 199 = 1 :=
by
  use 133
  sorry

end modular_inverse_of_3_mod_199_l996_99672


namespace independent_recruitment_probabilities_l996_99671

/-- Represents a student in the independent recruitment process -/
inductive Student
| A
| B
| C

/-- The probability of passing the review for each student -/
def review_prob (s : Student) : ℝ :=
  match s with
  | Student.A => 0.5
  | Student.B => 0.6
  | Student.C => 0.4

/-- The probability of passing the cultural test after passing the review for each student -/
def cultural_test_prob (s : Student) : ℝ :=
  match s with
  | Student.A => 0.6
  | Student.B => 0.5
  | Student.C => 0.75

/-- The probability of obtaining qualification for independent recruitment for each student -/
def qualification_prob (s : Student) : ℝ :=
  review_prob s * cultural_test_prob s

/-- The number of students who obtain qualification for independent recruitment -/
def num_qualified : Fin 4 → ℝ
| 0 => (1 - qualification_prob Student.A) * (1 - qualification_prob Student.B) * (1 - qualification_prob Student.C)
| 1 => 3 * qualification_prob Student.A * (1 - qualification_prob Student.B) * (1 - qualification_prob Student.C)
| 2 => 3 * qualification_prob Student.A * qualification_prob Student.B * (1 - qualification_prob Student.C)
| 3 => qualification_prob Student.A * qualification_prob Student.B * qualification_prob Student.C

/-- The expected value of the number of students who obtain qualification -/
def expected_num_qualified : ℝ :=
  1 * num_qualified 1 + 2 * num_qualified 2 + 3 * num_qualified 3

theorem independent_recruitment_probabilities :
  (∀ s : Student, qualification_prob s = 0.3) ∧ expected_num_qualified = 0.9 := by
  sorry

end independent_recruitment_probabilities_l996_99671


namespace bowling_ball_weight_l996_99619

theorem bowling_ball_weight (b c k : ℝ) 
  (h1 : 9 * b = 6 * c)
  (h2 : c + k = 42)
  (h3 : 3 * k = 2 * c) :
  b = 16.8 := by
  sorry

end bowling_ball_weight_l996_99619


namespace tom_seashells_l996_99632

theorem tom_seashells (initial_seashells : ℕ) (given_away : ℕ) :
  initial_seashells = 5 →
  given_away = 2 →
  initial_seashells - given_away = 3 := by
  sorry

end tom_seashells_l996_99632


namespace zero_product_probability_l996_99692

def S : Finset ℤ := {-3, -2, -1, 0, 0, 2, 4, 5}

def different_pairs (s : Finset ℤ) : Finset (ℤ × ℤ) :=
  (s.product s).filter (λ (a, b) => a ≠ b)

def zero_product_pairs (s : Finset ℤ) : Finset (ℤ × ℤ) :=
  (different_pairs s).filter (λ (a, b) => a * b = 0)

theorem zero_product_probability :
  (zero_product_pairs S).card / (different_pairs S).card = 3 / 14 := by
  sorry

end zero_product_probability_l996_99692


namespace jesse_pencils_l996_99686

/-- Given that Jesse starts with 78 pencils and gives away 44 pencils,
    prove that he ends up with 34 pencils. -/
theorem jesse_pencils :
  let initial_pencils : ℕ := 78
  let pencils_given_away : ℕ := 44
  initial_pencils - pencils_given_away = 34 :=
by sorry

end jesse_pencils_l996_99686


namespace quadratic_equation_solution_l996_99670

theorem quadratic_equation_solution (c : ℝ) : 
  ((-5 : ℝ)^2 + c * (-5) - 45 = 0) → c = -4 := by
  sorry

end quadratic_equation_solution_l996_99670


namespace compare_roots_l996_99685

theorem compare_roots : 3^(1/3) > 2^(1/2) ∧ 2^(1/2) > 8^(1/8) ∧ 8^(1/8) > 9^(1/9) := by
  sorry

end compare_roots_l996_99685


namespace largest_integer_in_interval_l996_99626

theorem largest_integer_in_interval : ∃ (x : ℤ), 
  (∀ (y : ℤ), (1/5 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/12 → y ≤ x) ∧
  (1/5 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 7/12 :=
by
  -- The proof goes here
  sorry

end largest_integer_in_interval_l996_99626


namespace mean_of_remaining_numbers_l996_99683

def numbers : List ℤ := [1871, 2011, 2059, 2084, 2113, 2167, 2198, 2210]

theorem mean_of_remaining_numbers :
  ∀ (subset : List ℤ),
    subset ⊆ numbers →
    subset.length = 6 →
    (subset.sum : ℚ) / 6 = 2100 →
    let remaining := numbers.filter (λ x => x ∉ subset)
    (remaining.sum : ℚ) / 2 = 2056.5 := by
  sorry

end mean_of_remaining_numbers_l996_99683


namespace don_remaining_rum_l996_99628

/-- The amount of rum Sally gave Don on his pancakes, in ounces. -/
def initial_rum : ℝ := 10

/-- The maximum factor by which Don can consume rum for a healthy diet. -/
def max_factor : ℝ := 3

/-- The amount of rum Don had earlier that day, in ounces. -/
def earlier_rum : ℝ := 12

/-- Calculates the maximum amount of rum Don can consume for a healthy diet. -/
def max_rum : ℝ := initial_rum * max_factor

/-- Calculates the total amount of rum Don has consumed so far. -/
def consumed_rum : ℝ := initial_rum + earlier_rum

/-- Theorem stating how much rum Don can have after eating all of the rum and pancakes. -/
theorem don_remaining_rum : max_rum - consumed_rum = 8 := by sorry

end don_remaining_rum_l996_99628


namespace number_reduced_by_six_times_l996_99697

/-- 
Given a natural number N that does not end in zero, and a digit a (1 ≤ a ≤ 9) in N,
if replacing a with 0 reduces N by 6 times, then N = 12a.
-/
theorem number_reduced_by_six_times (N : ℕ) (a : ℕ) : 
  (1 ≤ a ∧ a ≤ 9) →  -- a is a single digit
  (∃ k : ℕ, N = 12 * 10^k + 2 * a * 10^k) →  -- N has the form 12a in base 10
  (N % 10 ≠ 0) →  -- N does not end in zero
  (∃ N' : ℕ, N' = N / 10^k ∧ N = 6 * (N' - a * 10^k + 0)) →  -- replacing a with 0 reduces N by 6 times
  N = 12 * a := by
sorry

end number_reduced_by_six_times_l996_99697


namespace complementary_of_35_is_55_l996_99631

/-- The complementary angle of a given angle in degrees -/
def complementaryAngle (angle : ℝ) : ℝ := 90 - angle

/-- Theorem: The complementary angle of 35° is 55° -/
theorem complementary_of_35_is_55 :
  complementaryAngle 35 = 55 := by
  sorry

end complementary_of_35_is_55_l996_99631


namespace mean_value_theorem_for_f_l996_99650

-- Define the function f(x) = x² + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

theorem mean_value_theorem_for_f :
  ∃ c ∈ Set.Ioo (-1 : ℝ) 2,
    f 2 - f (-1) = f' c * (2 - (-1)) ∧
    c = 1 / 2 := by
  sorry

#check mean_value_theorem_for_f

end mean_value_theorem_for_f_l996_99650


namespace system_of_equations_solutions_l996_99687

theorem system_of_equations_solutions :
  -- First system of equations
  (∃ x y : ℝ, 2*x - y = 3 ∧ x + y = 3 ∧ x = 2 ∧ y = 1) ∧
  -- Second system of equations
  (∃ x y : ℝ, x/4 + y/3 = 3 ∧ 3*x - 2*(y-1) = 11 ∧ x = 6 ∧ y = 9/2) :=
by sorry

end system_of_equations_solutions_l996_99687


namespace unique_solution_is_seven_l996_99657

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem unique_solution_is_seven :
  ∃! n : ℕ, n > 0 ∧ n^2 * factorial n + factorial n = 5040 :=
by
  sorry

end unique_solution_is_seven_l996_99657


namespace sameGradePercentage_is_32_percent_l996_99645

/-- Represents the number of students who received the same grade on both tests for each grade. -/
structure SameGradeCount where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Calculates the percentage of students who received the same grade on both tests. -/
def sameGradePercentage (totalStudents : ℕ) (sameGrades : SameGradeCount) : ℚ :=
  let sameGradeTotal := sameGrades.a + sameGrades.b + sameGrades.c + sameGrades.d + sameGrades.e
  (sameGradeTotal : ℚ) / (totalStudents : ℚ) * 100

/-- Theorem stating that the percentage of students who received the same grade on both tests is 32%. -/
theorem sameGradePercentage_is_32_percent :
  let totalStudents := 50
  let sameGrades := SameGradeCount.mk 4 6 3 2 1
  sameGradePercentage totalStudents sameGrades = 32 := by
  sorry

end sameGradePercentage_is_32_percent_l996_99645


namespace train_arrival_theorem_l996_99693

/-- Represents a time with day, hour, and minute -/
structure Time where
  day : String
  hour : Nat
  minute : Nat

/-- Represents the journey of the train -/
structure TrainJourney where
  startTime : Time
  firstLegDuration : Nat
  secondLegDuration : Nat
  layoverDuration : Nat
  timeZonesCrossed : Nat
  timeZoneDifference : Nat

def calculateArrivalTime (journey : TrainJourney) : Time :=
  sorry

theorem train_arrival_theorem (journey : TrainJourney) 
  (h1 : journey.startTime = ⟨"Tuesday", 5, 0⟩)
  (h2 : journey.firstLegDuration = 12)
  (h3 : journey.secondLegDuration = 21)
  (h4 : journey.layoverDuration = 3)
  (h5 : journey.timeZonesCrossed = 2)
  (h6 : journey.timeZoneDifference = 1) :
  calculateArrivalTime journey = ⟨"Wednesday", 9, 0⟩ :=
by
  sorry

#check train_arrival_theorem

end train_arrival_theorem_l996_99693


namespace equation_solution_l996_99601

theorem equation_solution : ∃ m : ℤ, 3^4 - m = 4^3 + 2 ∧ m = 15 := by
  sorry

end equation_solution_l996_99601


namespace april_flower_sale_earnings_l996_99640

theorem april_flower_sale_earnings 
  (rose_price : ℕ)
  (initial_roses : ℕ)
  (remaining_roses : ℕ)
  (h1 : rose_price = 7)
  (h2 : initial_roses = 9)
  (h3 : remaining_roses = 4) :
  (initial_roses - remaining_roses) * rose_price = 35 :=
by sorry

end april_flower_sale_earnings_l996_99640


namespace extended_pattern_ratio_l996_99678

/-- Represents a rectangular floor pattern with black and white tiles -/
structure FloorPattern where
  width : ℕ
  height : ℕ
  blackTiles : ℕ
  whiteTiles : ℕ

/-- Adds a border of white tiles to a floor pattern -/
def addWhiteBorder (pattern : FloorPattern) : FloorPattern :=
  { width := pattern.width + 2
  , height := pattern.height + 2
  , blackTiles := pattern.blackTiles
  , whiteTiles := pattern.whiteTiles + (pattern.width + 2) * (pattern.height + 2) - (pattern.width * pattern.height)
  }

/-- Calculates the ratio of black tiles to white tiles -/
def tileRatio (pattern : FloorPattern) : ℚ :=
  pattern.blackTiles / pattern.whiteTiles

theorem extended_pattern_ratio :
  let initialPattern : FloorPattern :=
    { width := 5
    , height := 7
    , blackTiles := 14
    , whiteTiles := 21
    }
  let extendedPattern := addWhiteBorder initialPattern
  tileRatio extendedPattern = 2 / 7 := by
  sorry

end extended_pattern_ratio_l996_99678


namespace unique_positive_integer_l996_99696

theorem unique_positive_integer : ∃! (x : ℕ), x > 0 ∧ 15 * x = x^2 + 56 :=
by
  -- The proof goes here
  sorry

end unique_positive_integer_l996_99696


namespace smallest_fraction_greater_than_five_sixths_l996_99625

theorem smallest_fraction_greater_than_five_sixths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 5 / 6 →
    81 / 97 ≤ (a : ℚ) / b :=
by sorry

end smallest_fraction_greater_than_five_sixths_l996_99625


namespace divisibility_of_2_pow_55_plus_1_l996_99611

theorem divisibility_of_2_pow_55_plus_1 : 
  ∃ k : ℤ, 2^55 + 1 = 33 * k := by
  sorry

end divisibility_of_2_pow_55_plus_1_l996_99611


namespace nth_equation_l996_99676

theorem nth_equation (n : ℕ) :
  (n + 1 : ℚ) / ((n + 1)^2 - 1) - 1 / (n * (n + 1) * (n + 2)) = 1 / (n + 1) := by
  sorry

end nth_equation_l996_99676


namespace original_amount_proof_l996_99684

def transaction (x : ℚ) : ℚ :=
  ((2/3 * x + 10) * 2/3 + 20)

theorem original_amount_proof :
  ∃ (x : ℚ), x > 0 ∧ transaction x = x ∧ x = 48 := by
  sorry

end original_amount_proof_l996_99684


namespace brick_length_proof_l996_99682

/-- The length of a brick in centimeters. -/
def brick_length : ℝ := 25

/-- The width of a brick in centimeters. -/
def brick_width : ℝ := 11.25

/-- The height of a brick in centimeters. -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters. -/
def wall_length : ℝ := 800

/-- The width of the wall in centimeters. -/
def wall_width : ℝ := 600

/-- The height of the wall in centimeters. -/
def wall_height : ℝ := 22.5

/-- The number of bricks needed to build the wall. -/
def num_bricks : ℕ := 6400

/-- The volume of the wall in cubic centimeters. -/
def wall_volume : ℝ := wall_length * wall_width * wall_height

/-- The volume of a single brick in cubic centimeters. -/
def brick_volume : ℝ := brick_length * brick_width * brick_height

theorem brick_length_proof : 
  brick_length * brick_width * brick_height * num_bricks = wall_volume :=
by sorry

end brick_length_proof_l996_99682


namespace zach_monday_miles_l996_99655

/-- Calculates the number of miles driven on Monday given the rental conditions and total cost --/
def miles_driven_monday (flat_fee : ℚ) (cost_per_mile : ℚ) (thursday_miles : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - flat_fee - cost_per_mile * thursday_miles) / cost_per_mile

/-- Proves that Zach drove 620 miles on Monday given the rental conditions and total cost --/
theorem zach_monday_miles :
  let flat_fee : ℚ := 150
  let cost_per_mile : ℚ := 1/2
  let thursday_miles : ℚ := 744
  let total_cost : ℚ := 832
  miles_driven_monday flat_fee cost_per_mile thursday_miles total_cost = 620 := by
  sorry

end zach_monday_miles_l996_99655


namespace range_of_a_l996_99606

theorem range_of_a (a : ℝ) : 
  (∀ b : ℝ, ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ |x^2 + a*x + b| ≥ 1) → 
  a ≥ 1 ∨ a ≤ -3 := by
sorry

end range_of_a_l996_99606


namespace tangent_line_theorem_l996_99691

/-- The function f(x) -/
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a

/-- The function g(x) -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The derivative of f(x) -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 4*a*x + b

/-- The derivative of g(x) -/
def g_deriv (x : ℝ) : ℝ := 2*x - 3

theorem tangent_line_theorem (a b : ℝ) :
  f a b 2 = 0 ∧ g 2 = 0 ∧ f_deriv a b 2 = g_deriv 2 →
  a = -3 ∧ b = 1 ∧ ∀ x y, y = x - 2 ↔ f a b x = y ∧ g x = y :=
sorry

end tangent_line_theorem_l996_99691


namespace two_distinct_solutions_l996_99674

theorem two_distinct_solutions (a : ℝ) : 
  (16 * (a - 3) > 0) →
  (a > 0) →
  (a^2 - 16*a + 48 > 0) →
  (a ≠ 19) →
  (∃ (x₁ x₂ : ℝ), x₁ = a + 4 * Real.sqrt (a - 3) ∧ 
                   x₂ = a - 4 * Real.sqrt (a - 3) ∧ 
                   x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂) →
  (a > 3 ∧ a < 4) ∨ (a > 12 ∧ a < 19) ∨ (a > 19) :=
by sorry


end two_distinct_solutions_l996_99674


namespace enclosure_posts_count_l996_99607

/-- Calculates the number of posts needed for a rectangular enclosure with a stone wall --/
def calculate_posts (length width wall_length post_spacing : ℕ) : ℕ :=
  let long_side := max length width
  let short_side := min length width
  let long_side_posts := long_side / post_spacing + 1
  let short_side_posts := (short_side / post_spacing + 1) - 1
  long_side_posts + 2 * short_side_posts

/-- The number of posts required for the given enclosure is 19 --/
theorem enclosure_posts_count :
  calculate_posts 50 80 120 10 = 19 := by
  sorry

end enclosure_posts_count_l996_99607


namespace line_plane_perpendicular_l996_99639

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel m n → 
  subset n β → 
  plane_perpendicular α β :=
sorry

end line_plane_perpendicular_l996_99639


namespace sin_equality_proof_l996_99630

theorem sin_equality_proof (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) :
  Real.sin (n * π / 180) = Real.sin (721 * π / 180) → n = 1 := by
  sorry

end sin_equality_proof_l996_99630


namespace max_min_values_part1_unique_b_part2_l996_99615

noncomputable section

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - Real.log x

theorem max_min_values_part1 :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x ≤ max) ∧
    (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x = max) ∧
    (∀ x ∈ Set.Icc (1/2 : ℝ) 2, min ≤ f (-1) 3 x) ∧
    (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x = min) ∧
    max = 2 ∧
    min = Real.log 2 + 5/4 :=
sorry

theorem unique_b_part2 :
  ∃! b : ℝ,
    b > 0 ∧
    (∀ x ∈ Set.Ioo 0 (Real.exp 1),
      f 0 b x ≥ 3) ∧
    (∃ x ∈ Set.Ioo 0 (Real.exp 1),
      f 0 b x = 3) ∧
    b = Real.exp 2 :=
sorry

end max_min_values_part1_unique_b_part2_l996_99615


namespace validSquaresCount_l996_99664

/-- Represents a square on the checkerboard -/
structure Square :=
  (x : Nat) -- x-coordinate of the top-left corner
  (y : Nat) -- y-coordinate of the top-left corner
  (size : Nat) -- side length of the square

/-- Defines the 10x10 checkerboard -/
def checkerboard : Nat := 10

/-- Checks if a square contains at least 8 black squares -/
def hasAtLeast8BlackSquares (s : Square) : Bool :=
  -- Implementation details omitted
  sorry

/-- Counts the number of valid squares on the checkerboard -/
def countValidSquares : Nat :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that there are exactly 115 valid squares -/
theorem validSquaresCount : countValidSquares = 115 := by
  sorry

end validSquaresCount_l996_99664


namespace binomial_18_4_l996_99662

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end binomial_18_4_l996_99662


namespace min_value_expression_l996_99658

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b)^2 ≥ 7 ∧
  (2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b)^2 = 7 ↔ a = b ∧ b = c) :=
by sorry

end min_value_expression_l996_99658


namespace distance_from_origin_l996_99648

theorem distance_from_origin (x y : ℝ) (h1 : |y| = 15) (h2 : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) (h3 : x > 2) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (334 + 4 * Real.sqrt 105) := by sorry

end distance_from_origin_l996_99648


namespace kayak_production_sum_l996_99675

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem kayak_production_sum :
  let a := 6  -- First term (February production)
  let r := 3  -- Common ratio
  let n := 5  -- Number of months (February to June)
  geometric_sum a r n = 726 := by
sorry

end kayak_production_sum_l996_99675


namespace cutlery_theorem_l996_99649

def cutlery_count (initial_knives : ℕ) : ℕ :=
  let initial_teaspoons := 2 * initial_knives
  let additional_knives := initial_knives / 3
  let additional_teaspoons := (2 * initial_teaspoons) / 3
  let total_knives := initial_knives + additional_knives
  let total_teaspoons := initial_teaspoons + additional_teaspoons
  total_knives + total_teaspoons

theorem cutlery_theorem : cutlery_count 24 = 112 := by
  sorry

end cutlery_theorem_l996_99649


namespace fraction_C_is_simplest_l996_99609

-- Define the fractions
def fraction_A (m n : ℚ) : ℚ := 2 * m / (10 * m * n)
def fraction_B (m n : ℚ) : ℚ := (m^2 - n^2) / (m + n)
def fraction_C (m n : ℚ) : ℚ := (m^2 + n^2) / (m + n)
def fraction_D (a : ℚ) : ℚ := 2 * a / a^2

-- Define what it means for a fraction to be in simplest form
def is_simplest_form (f : ℚ) : Prop := 
  ∀ (g : ℚ), g ≠ 1 → g ≠ -1 → f ≠ g * (↑(f.num) / ↑(f.den))

-- Theorem statement
theorem fraction_C_is_simplest : 
  ∀ (m n : ℚ), m + n ≠ 0 → is_simplest_form (fraction_C m n) := by sorry

end fraction_C_is_simplest_l996_99609


namespace arc_length_45_degrees_l996_99604

/-- Given a circle with circumference 80 feet, proves that an arc corresponding
    to a central angle of 45° has a length of 10 feet. -/
theorem arc_length_45_degrees (circle : Real) (arc : Real) : 
  circle = 80 → -- The circumference of the circle is 80 feet
  arc = circle * (45 / 360) → -- The arc length is proportional to its central angle (45°)
  arc = 10 := by -- The arc length is 10 feet
sorry

end arc_length_45_degrees_l996_99604


namespace bridesmaid_dresses_completion_time_l996_99651

/-- Calculates the number of weeks needed to complete bridesmaid dresses -/
def weeks_to_complete_dresses (hours_per_dress : ℕ) (num_bridesmaids : ℕ) (hours_per_week : ℕ) : ℕ :=
  (hours_per_dress * num_bridesmaids) / hours_per_week

/-- Proves that it takes 15 weeks to complete the bridesmaid dresses under given conditions -/
theorem bridesmaid_dresses_completion_time :
  weeks_to_complete_dresses 12 5 4 = 15 := by
  sorry

#eval weeks_to_complete_dresses 12 5 4

end bridesmaid_dresses_completion_time_l996_99651


namespace angle_sum_around_point_l996_99600

theorem angle_sum_around_point (x : ℝ) : 
  (3 * x + 6 * x + x + 2 * x = 360) → x = 30 := by
  sorry

end angle_sum_around_point_l996_99600


namespace vector_at_zero_l996_99667

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  vector : ℝ → Fin 3 → ℝ

/-- The vector at a given parameter value -/
def vectorAt (line : ParameterizedLine) (t : ℝ) : Fin 3 → ℝ := line.vector t

theorem vector_at_zero (line : ParameterizedLine) 
  (h1 : vectorAt line 1 = ![2, 4, 9])
  (h2 : vectorAt line (-1) = ![-1, 1, 2]) :
  vectorAt line 0 = ![1/2, 5/2, 11/2] := by
  sorry

end vector_at_zero_l996_99667


namespace burattino_awake_journey_fraction_l996_99647

theorem burattino_awake_journey_fraction (x : ℝ) (h : x > 0) :
  let distance_before_sleep := x / 2
  let distance_slept := x / 3
  let distance_after_wake := x - (distance_before_sleep + distance_slept)
  distance_after_wake = distance_slept / 2 →
  (distance_before_sleep + distance_after_wake) / x = 2 / 3 := by
sorry

end burattino_awake_journey_fraction_l996_99647


namespace system_solution_proof_l996_99641

theorem system_solution_proof :
  let x₁ : ℚ := 3/2
  let x₂ : ℚ := 1/2
  (3 * x₁ - 5 * x₂ = 2) ∧ (2 * x₁ + 4 * x₂ = 5) := by
  sorry

end system_solution_proof_l996_99641


namespace rod_cutting_l996_99644

theorem rod_cutting (rod_length : ℝ) (piece_length : ℚ) : 
  rod_length = 58.75 →
  piece_length = 137 + 2/3 →
  ⌊(rod_length * 100) / (piece_length : ℝ)⌋ = 14 := by
  sorry

end rod_cutting_l996_99644


namespace tens_digit_of_sum_is_one_l996_99690

/-- 
Theorem: For any three-digit number where the hundreds digit is 3 more than the units digit,
the tens digit of the sum of this number and its reverse is always 1.
-/
theorem tens_digit_of_sum_is_one (c b : ℕ) (h1 : c < 10) (h2 : b < 10) : 
  (((202 * c + 20 * b + 303) / 10) % 10) = 1 := by
  sorry

end tens_digit_of_sum_is_one_l996_99690


namespace ellipse_properties_l996_99623

/-- An ellipse with specific properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  max_distance_to_foci : ℝ
  min_distance_to_foci : ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

/-- Theorem about a specific ellipse and its properties -/
theorem ellipse_properties (C : Ellipse) 
    (h1 : C.center = (0, 0))
    (h2 : C.foci_on_x_axis = true)
    (h3 : C.max_distance_to_foci = 3)
    (h4 : C.min_distance_to_foci = 1) :
  (∃ (x y : ℝ), standard_equation 4 3 x y) ∧ 
  (∃ (P F₁ F₂ : ℝ × ℝ), 
    (standard_equation 4 3 P.1 P.2) →
    (F₁ = (-1, 0) ∧ F₂ = (1, 0)) →
    (∀ (Q : ℝ × ℝ), standard_equation 4 3 Q.1 Q.2 → 
      dot_product (P.1 - F₁.1, P.2 - F₁.2) (P.1 - F₂.1, P.2 - F₂.2) ≤ 3 ∧
      dot_product (Q.1 - F₁.1, Q.2 - F₁.2) (Q.1 - F₂.1, Q.2 - F₂.2) ≥ 2)) :=
by
  sorry


end ellipse_properties_l996_99623


namespace length_PF1_is_seven_halves_l996_99652

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of the foci -/
def focus1 (c : ℝ) : ℝ × ℝ := (-c, 0)
def focus2 (c : ℝ) : ℝ × ℝ := (c, 0)

/-- Definition of point P -/
def point_P (c y : ℝ) : ℝ × ℝ := (c, y)

/-- The line through F₂ and P is perpendicular to x-axis -/
def line_perpendicular (c y : ℝ) : Prop := 
  point_P c y = (c, y)

/-- Theorem: Length of PF₁ is 7/2 -/
theorem length_PF1_is_seven_halves (c y : ℝ) : 
  is_on_ellipse c y → 
  line_perpendicular c y → 
  let p := point_P c y
  let f1 := focus1 c
  Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) = 7/2 := by sorry

end length_PF1_is_seven_halves_l996_99652


namespace cube_surface_area_increase_l996_99654

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_surface_area := 6 * L^2
  let new_edge_length := 1.2 * L
  let new_surface_area := 6 * new_edge_length^2
  let percentage_increase := (new_surface_area - original_surface_area) / original_surface_area * 100
  percentage_increase = 44 := by sorry

end cube_surface_area_increase_l996_99654


namespace friday_production_to_meet_target_l996_99680

/-- The number of toys that need to be produced on Friday to meet the weekly target -/
def friday_production (weekly_target : ℕ) (mon_to_wed_daily : ℕ) (thursday : ℕ) : ℕ :=
  weekly_target - (3 * mon_to_wed_daily + thursday)

/-- Theorem stating the required Friday production to meet the weekly target -/
theorem friday_production_to_meet_target :
  friday_production 6500 1200 800 = 2100 := by
  sorry

end friday_production_to_meet_target_l996_99680


namespace complex_cube_theorem_l996_99663

theorem complex_cube_theorem (z : ℂ) (h : z = 1 - I) :
  ((1 + I) / z) ^ 3 = -I := by sorry

end complex_cube_theorem_l996_99663


namespace range_of_x_l996_99621

-- Define the set A as [2, 5]
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 5 }

-- Define the set B as {x | x < 1 ∨ x > 4}
def B : Set ℝ := { x | x < 1 ∨ x > 4 }

-- Define the statement S
def S (x : ℝ) : Prop := x ∈ A ∨ x ∈ B

-- Define the range R as [1, 2)
def R : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem range_of_x (x : ℝ) : ¬(S x) → x ∈ R := by
  sorry

end range_of_x_l996_99621


namespace valid_rod_pairs_l996_99646

def is_valid_polygon (a b c d e : ℕ) : Prop :=
  a + b + c + d > e ∧ a + b + c + e > d ∧ a + b + d + e > c ∧ 
  a + c + d + e > b ∧ b + c + d + e > a

def count_valid_pairs : ℕ → ℕ → ℕ → ℕ := sorry

theorem valid_rod_pairs : 
  let rod_lengths : List ℕ := List.range 50
  let selected_rods : List ℕ := [8, 12, 20]
  let remaining_rods : List ℕ := rod_lengths.filter (λ x => x ∉ selected_rods)
  count_valid_pairs 8 12 20 = 135 := by sorry

end valid_rod_pairs_l996_99646


namespace always_less_than_log_sum_implies_less_than_one_l996_99608

theorem always_less_than_log_sum_implies_less_than_one (a : ℝ) : 
  (∀ x : ℝ, a < Real.log (|x - 3| + |x + 7|)) → a < 1 := by
  sorry

end always_less_than_log_sum_implies_less_than_one_l996_99608


namespace derivative_at_zero_l996_99605

/-- Given a function f where f(x) = x^2 + 2x * f'(1), prove that f'(0) = -4 -/
theorem derivative_at_zero (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 0 = -4 := by
  sorry

end derivative_at_zero_l996_99605


namespace negative_a_squared_times_a_fourth_l996_99616

theorem negative_a_squared_times_a_fourth (a : ℝ) : (-a)^2 * a^4 = a^6 := by
  sorry

end negative_a_squared_times_a_fourth_l996_99616


namespace labourer_monthly_income_l996_99698

/-- Represents the financial situation of a labourer over a 10-month period --/
structure LabourerFinances where
  monthlyIncome : ℝ
  firstSixMonthsExpenditure : ℝ
  nextFourMonthsExpenditure : ℝ
  savings : ℝ

/-- Theorem stating the labourer's monthly income given the problem conditions --/
theorem labourer_monthly_income 
  (finances : LabourerFinances)
  (h1 : finances.firstSixMonthsExpenditure = 90 * 6)
  (h2 : finances.monthlyIncome * 6 < finances.firstSixMonthsExpenditure)
  (h3 : finances.nextFourMonthsExpenditure = 60 * 4)
  (h4 : finances.monthlyIncome * 4 = finances.nextFourMonthsExpenditure + finances.savings)
  (h5 : finances.savings = 30) :
  finances.monthlyIncome = 81 := by
  sorry

end labourer_monthly_income_l996_99698


namespace triangle_largest_angle_and_type_l996_99603

-- Define the triangle with angle ratio 4:3:2
def triangle_angles (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 180 ∧
  4 * b = 3 * a ∧ 3 * c = 2 * b

-- Theorem statement
theorem triangle_largest_angle_and_type 
  (a b c : ℝ) (h : triangle_angles a b c) : 
  a = 80 ∧ a < 90 ∧ b < 90 ∧ c < 90 := by
  sorry


end triangle_largest_angle_and_type_l996_99603


namespace cookies_left_l996_99668

def dozen : ℕ := 12

theorem cookies_left (total : ℕ) (eaten_percent : ℚ) (h1 : total = 2 * dozen) (h2 : eaten_percent = 1/4) :
  total - (eaten_percent * total).floor = 18 := by
  sorry

end cookies_left_l996_99668


namespace prob_different_colors_example_l996_99679

/-- A box containing colored balls. -/
structure Box where
  white : ℕ
  black : ℕ

/-- The probability of drawing two balls of different colors with replacement. -/
def prob_different_colors (b : Box) : ℚ :=
  (b.white * b.black + b.black * b.white) / ((b.white + b.black) * (b.white + b.black))

/-- Theorem: The probability of drawing two balls of different colors from a box 
    containing 2 white balls and 3 black balls, with replacement, is 12/25. -/
theorem prob_different_colors_example : 
  prob_different_colors ⟨2, 3⟩ = 12 / 25 := by
  sorry

#eval prob_different_colors ⟨2, 3⟩

end prob_different_colors_example_l996_99679


namespace log_5_18_l996_99620

-- Define the given conditions
variable (a b : ℝ)
variable (h1 : Real.log 2 / Real.log 10 = a)
variable (h2 : Real.log 3 / Real.log 10 = b)

-- State the theorem to be proved
theorem log_5_18 : Real.log 18 / Real.log 5 = (a + 2*b) / (1 - a) := by
  sorry

end log_5_18_l996_99620


namespace finite_steps_33_disks_infinite_steps_32_disks_l996_99665

/-- Represents a board with disks -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (disks : Nat)

/-- Represents a move on the board -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Represents the state of the game after some number of steps -/
structure GameState :=
  (board : Board)
  (step : Nat)

/-- Predicate to check if a game state is valid -/
def isValid (state : GameState) : Prop :=
  state.board.disks ≤ state.board.rows * state.board.cols

/-- Predicate to check if a move is valid given the previous move -/
def isValidMove (prevMove : Option Move) (currMove : Move) : Prop :=
  match prevMove with
  | none => true
  | some Move.Up => currMove = Move.Left ∨ currMove = Move.Right
  | some Move.Down => currMove = Move.Left ∨ currMove = Move.Right
  | some Move.Left => currMove = Move.Up ∨ currMove = Move.Down
  | some Move.Right => currMove = Move.Up ∨ currMove = Move.Down

/-- Theorem: With 33 disks on a 5x9 board, only finitely many steps are possible -/
theorem finite_steps_33_disks (board : Board) (h : board.rows = 5 ∧ board.cols = 9 ∧ board.disks = 33) :
  ∃ n : Nat, ∀ state : GameState, state.board = board → state.step > n → ¬isValid state :=
sorry

/-- Theorem: With 32 disks on a 5x9 board, infinitely many steps are possible -/
theorem infinite_steps_32_disks (board : Board) (h : board.rows = 5 ∧ board.cols = 9 ∧ board.disks = 32) :
  ∀ n : Nat, ∃ state : GameState, state.board = board ∧ state.step = n ∧ isValid state :=
sorry

end finite_steps_33_disks_infinite_steps_32_disks_l996_99665


namespace final_lights_on_l996_99669

/-- The number of lights -/
def n : ℕ := 56

/-- Function to count lights turned on by pressing every k-th switch -/
def count_lights (k : ℕ) : ℕ :=
  n / k

/-- Function to count lights affected by both operations -/
def count_overlap : ℕ :=
  n / 15

/-- The final number of lights turned on -/
def lights_on : ℕ :=
  count_lights 3 + count_lights 5 - count_overlap

theorem final_lights_on :
  lights_on = 26 := by
  sorry

end final_lights_on_l996_99669


namespace simplify_cube_root_exponent_sum_l996_99636

theorem simplify_cube_root_exponent_sum (a b c : ℝ) : 
  ∃ (k : ℝ) (m n p : ℕ), 
    (54 * a^6 * b^8 * c^14)^(1/3) = k * a^m * b^n * c^p ∧ m + n + p = 8 :=
by sorry

end simplify_cube_root_exponent_sum_l996_99636


namespace tangent_product_equals_three_l996_99618

theorem tangent_product_equals_three :
  Real.tan (20 * π / 180) * Real.tan (40 * π / 180) * Real.tan (60 * π / 180) * Real.tan (80 * π / 180) = 3 := by
  sorry

end tangent_product_equals_three_l996_99618


namespace negation_of_conditional_l996_99659

theorem negation_of_conditional (a b : ℝ) :
  ¬(a > b → a - 1 > b - 1) ↔ (a ≤ b → a - 1 ≤ b - 1) := by
  sorry

end negation_of_conditional_l996_99659


namespace walter_hushpuppies_per_guest_l996_99610

/-- Calculates the number of hushpuppies per guest given the number of guests,
    cooking rate, and total cooking time. -/
def hushpuppies_per_guest (guests : ℕ) (hushpuppies_per_batch : ℕ) 
    (minutes_per_batch : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes / minutes_per_batch * hushpuppies_per_batch) / guests

/-- Proves that given the specified conditions, each guest will eat 5 hushpuppies. -/
theorem walter_hushpuppies_per_guest : 
  hushpuppies_per_guest 20 10 8 80 = 5 := by
  sorry

end walter_hushpuppies_per_guest_l996_99610


namespace custom_mult_solution_l996_99677

/-- Custom multiplication operation -/
def star_mult (a b : ℝ) : ℝ := a * b + a + b

/-- Theorem stating that if 3 * x = 27 under the custom multiplication, then x = 6 -/
theorem custom_mult_solution :
  (∀ a b : ℝ, star_mult a b = a * b + a + b) →
  star_mult 3 x = 27 →
  x = 6 := by sorry

end custom_mult_solution_l996_99677


namespace simplify_expression_l996_99653

theorem simplify_expression (x : ℝ) :
  2 * x * (4 * x^2 - 3) - 4 * (x^2 - 3 * x + 6) = 8 * x^3 - 4 * x^2 + 6 * x - 24 :=
by sorry

end simplify_expression_l996_99653


namespace gold_coin_distribution_l996_99673

theorem gold_coin_distribution (x y : ℕ) (h : x^2 - y^2 = 49*(x - y)) : x + y = 49 := by
  sorry

end gold_coin_distribution_l996_99673


namespace card_arrangement_count_l996_99643

/-- The number of ways to arrange 6 cards into 3 envelopes -/
def arrangement_count : ℕ := 18

/-- The number of envelopes -/
def num_envelopes : ℕ := 3

/-- The number of cards -/
def num_cards : ℕ := 6

/-- The number of cards per envelope -/
def cards_per_envelope : ℕ := 2

/-- Cards 1 and 2 are in the same envelope -/
def cards_1_2_together : Prop := True

theorem card_arrangement_count :
  arrangement_count = num_envelopes * (num_cards - cards_per_envelope).choose cards_per_envelope :=
sorry

end card_arrangement_count_l996_99643


namespace interest_rate_difference_l996_99633

theorem interest_rate_difference (principal : ℝ) (original_rate higher_rate : ℝ) 
  (h1 : principal = 500)
  (h2 : principal * higher_rate / 100 - principal * original_rate / 100 = 30) :
  higher_rate - original_rate = 6 := by
  sorry

end interest_rate_difference_l996_99633


namespace line_arrangements_with_restriction_l996_99635

def number_of_students : ℕ := 5

def number_of_restricted_students : ℕ := 2

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def arrangements_with_restricted_together (n : ℕ) (r : ℕ) : ℕ :=
  (Nat.factorial (n - r + 1)) * (Nat.factorial r)

theorem line_arrangements_with_restriction :
  total_arrangements number_of_students - 
  arrangements_with_restricted_together number_of_students number_of_restricted_students = 72 := by
  sorry

end line_arrangements_with_restriction_l996_99635


namespace max_sum_of_squares_l996_99689

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 12 →
  a * b + c + d = 54 →
  a * d + b * c = 105 →
  c * d = 50 →
  a^2 + b^2 + c^2 + d^2 ≤ 124 :=
by sorry

end max_sum_of_squares_l996_99689


namespace program_output_l996_99694

def program (initial_A initial_B : Int) : (Int × Int × Int) :=
  let A₁ := if initial_A < 0 then -initial_A else initial_A
  let B₁ := initial_B * initial_B
  let A₂ := A₁ + B₁
  let C := A₂ - 2 * B₁
  let A₃ := A₂ / C
  let B₂ := B₁ * C + 1
  (A₃, B₂, C)

theorem program_output : program (-6) 2 = (5, 9, 2) := by
  sorry

end program_output_l996_99694


namespace share_distribution_l996_99622

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 378 →
  total = a + b + c →
  12 * a = 8 * b →
  12 * a = 6 * c →
  a = 84 := by
sorry

end share_distribution_l996_99622


namespace poultry_pricing_l996_99661

theorem poultry_pricing :
  ∃ (c d g : ℕ+),
    3 * c + d = 2 * g ∧
    c + 2 * d + 3 * g = 25 ∧
    c = 2 ∧ d = 4 ∧ g = 5 := by
  sorry

end poultry_pricing_l996_99661


namespace solution_set_when_a_eq_one_range_of_a_given_f_geq_three_halves_l996_99642

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |a * x + 1| + |x - a|
def g (x : ℝ) : ℝ := x^2 + x

-- Theorem for part (1)
theorem solution_set_when_a_eq_one :
  ∀ x : ℝ, (g x ≥ f 1 x) ↔ (x ≥ 1 ∨ x ≤ -3) :=
sorry

-- Theorem for part (2)
theorem range_of_a_given_f_geq_three_halves :
  (∀ x : ℝ, ∃ a : ℝ, a > 0 ∧ f a x ≥ 3/2) →
  (∀ a : ℝ, a > 0 → f a x ≥ 3/2 → a ≥ Real.sqrt 2 / 2) :=
sorry

end solution_set_when_a_eq_one_range_of_a_given_f_geq_three_halves_l996_99642


namespace smallest_q_is_6_l996_99699

/-- Three consecutive terms of an arithmetic sequence -/
structure ArithmeticTriple where
  p : ℝ
  q : ℝ
  r : ℝ
  positive : 0 < p ∧ 0 < q ∧ 0 < r
  consecutive : ∃ d : ℝ, p + d = q ∧ q + d = r

/-- The product of the three terms equals 216 -/
def productIs216 (t : ArithmeticTriple) : Prop :=
  t.p * t.q * t.r = 216

theorem smallest_q_is_6 (t : ArithmeticTriple) (h : productIs216 t) :
    t.q ≥ 6 ∧ ∃ t' : ArithmeticTriple, productIs216 t' ∧ t'.q = 6 := by
  sorry

end smallest_q_is_6_l996_99699


namespace function_zeros_l996_99688

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def count_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

theorem function_zeros (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f (2 * Real.pi))
  (h_zero_3 : f 3 = 0)
  (h_zero_4 : f 4 = 0) :
  count_zeros f 0 10 ≥ 11 := by
  sorry

end function_zeros_l996_99688


namespace derivative_of_f_l996_99681

noncomputable def f (x : ℝ) : ℝ := x^3 / 3 + 1 / x

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : 
  deriv f x = x^2 - 1 / x^2 := by sorry

end derivative_of_f_l996_99681


namespace last_number_is_odd_l996_99613

/-- The operation of choosing two numbers and replacing them with their absolute difference -/
def boardOperation (numbers : List Int) : List Int :=
  sorry

/-- The process of repeatedly applying the operation until only one number remains -/
def boardProcess (initialNumbers : List Int) : Int :=
  sorry

/-- The list of integers from 1 to 2018 -/
def initialBoard : List Int :=
  List.range 2018

theorem last_number_is_odd :
  Odd (boardProcess initialBoard) :=
by sorry

end last_number_is_odd_l996_99613


namespace min_value_of_fraction_sum_l996_99656

theorem min_value_of_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 9/y ≥ 1/a + 9/b) ∧ 1/a + 9/b = 16 :=
sorry

end min_value_of_fraction_sum_l996_99656


namespace simplify_fraction_product_l996_99638

theorem simplify_fraction_product : 24 * (3 / 4) * (2 / 11) * (5 / 8) = 45 / 22 := by
  sorry

end simplify_fraction_product_l996_99638


namespace initial_peanuts_count_l996_99627

/-- 
Given a box of peanuts, prove that the initial number of peanuts was 10,
when 8 peanuts were added and the final count is 18.
-/
theorem initial_peanuts_count (initial final added : ℕ) 
  (h1 : added = 8)
  (h2 : final = 18)
  (h3 : final = initial + added) : 
  initial = 10 := by
  sorry

end initial_peanuts_count_l996_99627


namespace five_digit_reverse_multiplication_l996_99614

theorem five_digit_reverse_multiplication (a b c d e : Nat) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 →
  4 * (a * 10000 + b * 1000 + c * 100 + d * 10 + e) = e * 10000 + d * 1000 + c * 100 + b * 10 + a →
  a + b + c + d + e = 27 := by
sorry

end five_digit_reverse_multiplication_l996_99614


namespace yard_fencing_l996_99637

theorem yard_fencing (length width : ℝ) : 
  length > 0 → 
  width > 0 → 
  length * width = 320 → 
  2 * width + length = 56 → 
  length = 40 := by
sorry

end yard_fencing_l996_99637


namespace least_positive_integer_to_multiple_of_five_l996_99695

theorem least_positive_integer_to_multiple_of_five : ∃ (n : ℕ), n > 0 ∧ (567 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (567 + m) % 5 = 0 → n ≤ m :=
by sorry

end least_positive_integer_to_multiple_of_five_l996_99695


namespace smallest_excluded_number_l996_99612

theorem smallest_excluded_number : ∃ n : ℕ, 
  (∀ k ∈ Finset.range 200, k + 1 ≠ 128 ∧ k + 1 ≠ 129 → n % (k + 1) = 0) ∧
  (∀ m : ℕ, m < 128 → 
    ¬∃ n : ℕ, (∀ k ∈ Finset.range 200, k + 1 ≠ m ∧ k + 1 ≠ m + 1 → n % (k + 1) = 0)) :=
by sorry

end smallest_excluded_number_l996_99612


namespace a_range_l996_99666

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- State the theorem
theorem a_range (a : ℝ) :
  (f a 3 < f a 4) ∧
  (∀ n : ℕ, n ≥ 8 → f a n > f a (n + 1)) →
  -1/7 < a ∧ a < -1/17 :=
sorry

end a_range_l996_99666


namespace total_spend_l996_99629

-- Define the given conditions
def num_tshirts : ℕ := 3
def cost_per_tshirt : ℕ := 20
def cost_pants : ℕ := 50

-- State the theorem
theorem total_spend : 
  num_tshirts * cost_per_tshirt + cost_pants = 110 := by
  sorry

end total_spend_l996_99629


namespace increasing_function_properties_l996_99617

/-- A function f is increasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_properties
  (f : ℝ → ℝ) (hf : IncreasingFunction f) :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b)) ∧
  (∀ a b : ℝ, f a + f b ≥ f (-a) + f (-b) → a + b ≥ 0) :=
by sorry

end increasing_function_properties_l996_99617


namespace student_rank_from_right_l996_99602

theorem student_rank_from_right 
  (total_students : Nat) 
  (rank_from_left : Nat) 
  (h1 : total_students = 21)
  (h2 : rank_from_left = 6) :
  total_students - rank_from_left + 1 = 17 :=
by sorry

end student_rank_from_right_l996_99602


namespace max_ice_cream_servings_l996_99660

/-- Represents a day in February --/
structure FebruaryDay where
  dayOfMonth : Nat
  dayOfWeek : Nat
  h1 : dayOfMonth ≥ 1 ∧ dayOfMonth ≤ 28
  h2 : dayOfWeek ≥ 1 ∧ dayOfWeek ≤ 7

/-- Defines the ice cream eating rules --/
def iceCreamServings (day : FebruaryDay) : Nat :=
  if day.dayOfMonth % 2 = 0 ∧ (day.dayOfWeek = 3 ∨ day.dayOfWeek = 4) then 7
  else if (day.dayOfWeek = 1 ∨ day.dayOfWeek = 2) ∧ day.dayOfMonth % 2 = 1 then 3
  else if day.dayOfWeek = 5 then day.dayOfMonth
  else 0

/-- Theorem stating the maximum number of ice cream servings in February --/
theorem max_ice_cream_servings :
  (∃ (days : List FebruaryDay), days.length = 28 ∧
    (∀ d ∈ days, d.dayOfMonth ≥ 1 ∧ d.dayOfMonth ≤ 28 ∧ d.dayOfWeek ≥ 1 ∧ d.dayOfWeek ≤ 7) ∧
    (∀ i j, i ≠ j → (days.get i).dayOfMonth ≠ (days.get j).dayOfMonth) ∧
    (List.sum (days.map iceCreamServings) ≤ 110)) ∧
  (∃ (optimalDays : List FebruaryDay), optimalDays.length = 28 ∧
    (∀ d ∈ optimalDays, d.dayOfMonth ≥ 1 ∧ d.dayOfMonth ≤ 28 ∧ d.dayOfWeek ≥ 1 ∧ d.dayOfWeek ≤ 7) ∧
    (∀ i j, i ≠ j → (optimalDays.get i).dayOfMonth ≠ (optimalDays.get j).dayOfMonth) ∧
    (List.sum (optimalDays.map iceCreamServings) = 110)) := by
  sorry


end max_ice_cream_servings_l996_99660


namespace kitten_weight_l996_99634

theorem kitten_weight (kitten smaller_dog larger_dog : ℝ) 
  (total_weight : kitten + smaller_dog + larger_dog = 36)
  (larger_comparison : kitten + larger_dog = 2 * smaller_dog)
  (smaller_comparison : kitten + smaller_dog = larger_dog) :
  kitten = 9 := by
sorry

end kitten_weight_l996_99634


namespace rectangle_uniquely_symmetric_l996_99624

-- Define the properties
def axisymmetric (shape : Type) : Prop := sorry

def centrally_symmetric (shape : Type) : Prop := sorry

-- Define the shapes
def equilateral_triangle : Type := sorry
def rectangle : Type := sorry
def parallelogram : Type := sorry
def regular_pentagon : Type := sorry

-- Theorem statement
theorem rectangle_uniquely_symmetric :
  (axisymmetric equilateral_triangle ∧ centrally_symmetric equilateral_triangle) = False ∧
  (axisymmetric rectangle ∧ centrally_symmetric rectangle) = True ∧
  (axisymmetric parallelogram ∧ centrally_symmetric parallelogram) = False ∧
  (axisymmetric regular_pentagon ∧ centrally_symmetric regular_pentagon) = False :=
sorry

end rectangle_uniquely_symmetric_l996_99624
