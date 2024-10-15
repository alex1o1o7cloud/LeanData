import Mathlib

namespace NUMINAMATH_CALUDE_sophie_chocolates_l2769_276912

theorem sophie_chocolates :
  ∃ (x : ℕ), x ≥ 150 ∧ x % 15 = 7 ∧ ∀ (y : ℕ), y ≥ 150 ∧ y % 15 = 7 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_sophie_chocolates_l2769_276912


namespace NUMINAMATH_CALUDE_factors_lcm_gcd_of_24_60_180_l2769_276989

def numbers : List Nat := [24, 60, 180]

theorem factors_lcm_gcd_of_24_60_180 :
  (∃ (common_factors : List Nat), common_factors.length = 6 ∧ 
    ∀ n ∈ common_factors, ∀ m ∈ numbers, n ∣ m) ∧
  Nat.lcm 24 (Nat.lcm 60 180) = 180 ∧
  Nat.gcd 24 (Nat.gcd 60 180) = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_lcm_gcd_of_24_60_180_l2769_276989


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2769_276935

/-- 
Given an arithmetic sequence where:
- The first term is 2/3
- The second term is 1
- The third term is 4/3

Prove that the eighth term of this sequence is 3.
-/
theorem arithmetic_sequence_eighth_term : 
  ∀ (a : ℕ → ℚ), 
    (a 1 = 2/3) →
    (a 2 = 1) →
    (a 3 = 4/3) →
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
    a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2769_276935


namespace NUMINAMATH_CALUDE_cube_can_be_threaded_tetrahedron_can_be_threaded_l2769_276962

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a frame (cube or tetrahedron)
structure Frame where
  vertices : List Point3D
  edges : List (Point3D × Point3D)

-- Define a hole in the plane
structure Hole where
  boundary : Point2D → Bool

-- Function to check if a hole is valid (closed and non-self-intersecting)
def isValidHole (h : Hole) : Prop :=
  sorry

-- Function to check if a frame can be threaded through a hole
def canThreadThrough (f : Frame) (h : Hole) : Prop :=
  sorry

-- Theorem for cube
theorem cube_can_be_threaded :
  ∃ (cubef : Frame) (h : Hole), isValidHole h ∧ canThreadThrough cubef h :=
sorry

-- Theorem for tetrahedron
theorem tetrahedron_can_be_threaded :
  ∃ (tetf : Frame) (h : Hole), isValidHole h ∧ canThreadThrough tetf h :=
sorry

end NUMINAMATH_CALUDE_cube_can_be_threaded_tetrahedron_can_be_threaded_l2769_276962


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l2769_276906

/-- Represents the number of students in each category -/
structure StudentPopulation where
  total : ℕ
  junior : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Represents the sample size and the number of students to be drawn from each category -/
structure SampleSize where
  total : ℕ
  junior : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Calculates the correct sample size for stratified sampling -/
def calculateStratifiedSample (pop : StudentPopulation) (sampleTotal : ℕ) : SampleSize :=
  { total := sampleTotal,
    junior := (sampleTotal * pop.junior) / pop.total,
    undergraduate := (sampleTotal * pop.undergraduate) / pop.total,
    graduate := (sampleTotal * pop.graduate) / pop.total }

/-- Theorem: The calculated stratified sample is correct for the given population -/
theorem stratified_sample_correct (pop : StudentPopulation) (sample : SampleSize) :
  pop.total = 5400 ∧ 
  pop.junior = 1500 ∧ 
  pop.undergraduate = 3000 ∧ 
  pop.graduate = 900 ∧
  sample.total = 180 →
  calculateStratifiedSample pop sample.total = 
    { total := 180, junior := 50, undergraduate := 100, graduate := 30 } := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_correct_l2769_276906


namespace NUMINAMATH_CALUDE_paulson_spending_percentage_l2769_276975

theorem paulson_spending_percentage 
  (income_increase : Real) 
  (expenditure_increase : Real) 
  (savings_increase : Real) : 
  income_increase = 0.20 → 
  expenditure_increase = 0.10 → 
  savings_increase = 0.50 → 
  ∃ (original_income : Real) (spending_percentage : Real),
    spending_percentage = 0.75 ∧ 
    original_income > 0 ∧
    (1 + income_increase) * original_income - 
    (1 + expenditure_increase) * spending_percentage * original_income = 
    (1 + savings_increase) * (original_income - spending_percentage * original_income) :=
by sorry

end NUMINAMATH_CALUDE_paulson_spending_percentage_l2769_276975


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l2769_276916

def jeff_scores : List ℝ := [89, 92, 88, 95, 91]

theorem jeff_scores_mean : (jeff_scores.sum / jeff_scores.length) = 91 := by
  sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l2769_276916


namespace NUMINAMATH_CALUDE_cake_slices_problem_l2769_276923

theorem cake_slices_problem (num_cakes : ℕ) (price_per_slice donation1 donation2 total_raised : ℚ) :
  num_cakes = 10 →
  price_per_slice = 1 →
  donation1 = 1/2 →
  donation2 = 1/4 →
  total_raised = 140 →
  ∃ (slices_per_cake : ℕ), 
    slices_per_cake = 8 ∧
    (num_cakes * slices_per_cake : ℚ) * (price_per_slice + donation1 + donation2) = total_raised :=
by sorry

end NUMINAMATH_CALUDE_cake_slices_problem_l2769_276923


namespace NUMINAMATH_CALUDE_lines_parallel_iff_l2769_276974

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + 2 * a = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ (x y : ℝ), line1 a x y ↔ ∃ (k : ℝ), line2 a (x + k) (y + k)

-- Theorem statement
theorem lines_parallel_iff : ∀ (a : ℝ), parallel a ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_l2769_276974


namespace NUMINAMATH_CALUDE_circle_ratio_l2769_276948

theorem circle_ratio (r R a b : ℝ) (hr : r > 0) (hR : R > r) (ha : a > 0) (hb : b > 0) 
  (h : π * R^2 = (a / b) * (π * R^2 - π * r^2)) :
  R / r = Real.sqrt a / Real.sqrt (a - b) := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l2769_276948


namespace NUMINAMATH_CALUDE_tiffany_bags_total_l2769_276911

theorem tiffany_bags_total (monday_bags : ℕ) (next_day_bags : ℕ) 
  (h1 : monday_bags = 4) 
  (h2 : next_day_bags = 8) : 
  monday_bags + next_day_bags = 12 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_total_l2769_276911


namespace NUMINAMATH_CALUDE_salt_proportion_is_one_twenty_first_l2769_276904

/-- The proportion of salt in a saltwater solution -/
def salt_proportion (salt_mass : ℚ) (water_mass : ℚ) : ℚ :=
  salt_mass / (salt_mass + water_mass)

/-- Proof that the proportion of salt in the given saltwater solution is 1/21 -/
theorem salt_proportion_is_one_twenty_first :
  let salt_mass : ℚ := 50
  let water_mass : ℚ := 1000
  salt_proportion salt_mass water_mass = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_salt_proportion_is_one_twenty_first_l2769_276904


namespace NUMINAMATH_CALUDE_average_problem_l2769_276951

theorem average_problem (y : ℝ) : 
  (15 + 25 + 35 + y) / 4 = 30 → y = 45 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l2769_276951


namespace NUMINAMATH_CALUDE_average_roots_quadratic_l2769_276901

theorem average_roots_quadratic (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 4 * x₁ - 5 = 0) → 
  (3 * x₂^2 + 4 * x₂ - 5 = 0) → 
  x₁ ≠ x₂ → 
  (x₁ + x₂) / 2 = -2/3 := by
sorry

end NUMINAMATH_CALUDE_average_roots_quadratic_l2769_276901


namespace NUMINAMATH_CALUDE_distance_B_to_x_axis_l2769_276980

def point_B : ℝ × ℝ := (2, -3)

def distance_to_x_axis (p : ℝ × ℝ) : ℝ := |p.2|

theorem distance_B_to_x_axis :
  distance_to_x_axis point_B = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_B_to_x_axis_l2769_276980


namespace NUMINAMATH_CALUDE_problem_statements_l2769_276924

theorem problem_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (ab ≤ 1 → 1/a + 1/b ≥ 2) ∧
  (a + b = 4 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 4 ∧ 1/x + 9/y ≤ 1/a + 9/b ∧ 1/a + 9/b = 4) ∧
  (a^2 + b^2 = 4 → ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = 4 → x*y ≤ a*b ∧ a*b = 2) ∧
  ¬(2*a + b = 1 → ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2*x + y = 1 → x*y ≤ a*b ∧ a*b = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l2769_276924


namespace NUMINAMATH_CALUDE_selection_theorem_l2769_276992

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of girls -/
def num_girls : ℕ := 5

/-- The total number of boys -/
def num_boys : ℕ := 7

/-- The number of representatives to be selected -/
def num_representatives : ℕ := 5

theorem selection_theorem :
  /- At least one girl is selected -/
  (choose (num_girls + num_boys) num_representatives - choose num_boys num_representatives = 771) ∧
  /- Boy A and Girl B are selected -/
  (choose (num_girls + num_boys - 2) (num_representatives - 2) = 120) ∧
  /- At least one of Boy A or Girl B is selected -/
  (choose (num_girls + num_boys) num_representatives - choose (num_girls + num_boys - 2) num_representatives = 540) :=
by sorry

end NUMINAMATH_CALUDE_selection_theorem_l2769_276992


namespace NUMINAMATH_CALUDE_price_difference_year_l2769_276972

def price_P (n : ℕ) : ℚ := 420/100 + 40/100 * n
def price_Q (n : ℕ) : ℚ := 630/100 + 15/100 * n

theorem price_difference_year : 
  ∃ n : ℕ, price_P n = price_Q n + 40/100 ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_price_difference_year_l2769_276972


namespace NUMINAMATH_CALUDE_divisibility_of_7386038_l2769_276986

theorem divisibility_of_7386038 : ∃ (k : ℕ), 7386038 = 7 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_of_7386038_l2769_276986


namespace NUMINAMATH_CALUDE_sector_area_l2769_276952

/-- Given a circular sector with perimeter 8 cm and central angle 2 radians, its area is 4 cm² -/
theorem sector_area (r : ℝ) (l : ℝ) : 
  l + 2 * r = 8 →  -- Perimeter condition
  l = 2 * r →      -- Arc length condition (derived from central angle)
  (1 / 2) * 2 * r^2 = 4 := by  -- Area calculation
sorry

end NUMINAMATH_CALUDE_sector_area_l2769_276952


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_l2769_276918

theorem max_value_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x*y = 5) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2 * x + y ≤ z → z ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_l2769_276918


namespace NUMINAMATH_CALUDE_integer_fraction_theorem_l2769_276939

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 1 ∧ b ≥ 1 ∧
  (∃ (k₁ k₂ : ℤ), (a^2 + b : ℤ) = k₁ * (b^2 - a) ∧ (b^2 + a : ℤ) = k₂ * (a^2 - b))

def solution_set : Set (ℕ × ℕ) :=
  {(2,2), (3,3), (1,2), (2,1), (2,3), (3,2)}

theorem integer_fraction_theorem :
  ∀ (a b : ℕ), is_valid_pair a b ↔ (a, b) ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_integer_fraction_theorem_l2769_276939


namespace NUMINAMATH_CALUDE_garden_area_with_fountain_garden_area_calculation_l2769_276927

/-- Calculates the new available area for planting in a rectangular garden with a circular fountain -/
theorem garden_area_with_fountain (perimeter : ℝ) (side : ℝ) (fountain_radius : ℝ) : ℝ :=
  let length := (perimeter - 2 * side) / 2
  let garden_area := length * side
  let fountain_area := Real.pi * fountain_radius^2
  garden_area - fountain_area

/-- Proves that the new available area for planting is approximately 37185.84 square meters -/
theorem garden_area_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |garden_area_with_fountain 950 100 10 - 37185.84| < ε :=
sorry

end NUMINAMATH_CALUDE_garden_area_with_fountain_garden_area_calculation_l2769_276927


namespace NUMINAMATH_CALUDE_five_hour_study_score_l2769_276966

/-- Represents a student's test score based on study time -/
structure TestScore where
  studyTime : ℝ
  score : ℝ

/-- The maximum possible score on a test -/
def maxScore : ℝ := 100

/-- Calculates the potential score based on study time and effectiveness -/
def potentialScore (effectiveness : ℝ) (studyTime : ℝ) : ℝ :=
  effectiveness * studyTime

/-- Theorem: Given the conditions, the score for 5 hours of study is 100 -/
theorem five_hour_study_score :
  ∀ (effectiveness : ℝ),
  effectiveness > 0 →
  potentialScore effectiveness 2 = 80 →
  min (potentialScore effectiveness 5) maxScore = 100 := by
sorry

end NUMINAMATH_CALUDE_five_hour_study_score_l2769_276966


namespace NUMINAMATH_CALUDE_non_adjacent_book_arrangements_l2769_276944

/-- Represents the number of books of each subject -/
structure BookCounts where
  chinese : Nat
  math : Nat
  physics : Nat

/-- Calculates the total number of books -/
def totalBooks (counts : BookCounts) : Nat :=
  counts.chinese + counts.math + counts.physics

/-- Calculates the number of permutations of n items -/
def permutations (n : Nat) : Nat :=
  Nat.factorial n

/-- Calculates the number of arrangements where books of the same subject are not adjacent -/
def nonAdjacentArrangements (counts : BookCounts) : Nat :=
  let total := totalBooks counts
  let allArrangements := permutations total
  let chineseAdjacent := (permutations (total - counts.chinese + 1)) * (permutations counts.chinese)
  let mathAdjacent := (permutations (total - counts.math + 1)) * (permutations counts.math)
  let bothAdjacent := (permutations (total - counts.chinese - counts.math + 2)) * 
                      (permutations counts.chinese) * (permutations counts.math)
  allArrangements - chineseAdjacent - mathAdjacent + bothAdjacent

theorem non_adjacent_book_arrangements :
  let counts : BookCounts := { chinese := 2, math := 2, physics := 1 }
  nonAdjacentArrangements counts = 48 := by
  sorry

end NUMINAMATH_CALUDE_non_adjacent_book_arrangements_l2769_276944


namespace NUMINAMATH_CALUDE_probability_smaller_divides_larger_l2769_276941

def S : Finset ℕ := {1, 2, 3, 6, 9}

def divides_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  S.product S |>.filter (fun p => p.1 < p.2 ∧ p.2 % p.1 = 0)

theorem probability_smaller_divides_larger :
  (divides_pairs S).card / (S.product S |>.filter (fun p => p.1 ≠ p.2)).card = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_smaller_divides_larger_l2769_276941


namespace NUMINAMATH_CALUDE_eighteen_letter_arrangements_l2769_276902

theorem eighteen_letter_arrangements :
  let n : ℕ := 6
  let total_letters : ℕ := 3 * n
  let arrangement_count : ℕ := (Finset.range (n + 1)).sum (fun k => (Nat.choose n k)^3)
  ∀ (arrangements : Finset (Fin total_letters → Fin 3)),
    (∀ i : Fin total_letters, 
      (arrangements.card = arrangement_count) ∧
      (arrangements.card = (Finset.filter (fun arr => 
        (∀ j : Fin n, arr (j) ≠ 0) ∧
        (∀ j : Fin n, arr (j + n) ≠ 1) ∧
        (∀ j : Fin n, arr (j + 2*n) ≠ 2) ∧
        (arrangements.filter (fun arr => arr i = 0)).card = n ∧
        (arrangements.filter (fun arr => arr i = 1)).card = n ∧
        (arrangements.filter (fun arr => arr i = 2)).card = n
      ) arrangements).card)) := by
  sorry

#check eighteen_letter_arrangements

end NUMINAMATH_CALUDE_eighteen_letter_arrangements_l2769_276902


namespace NUMINAMATH_CALUDE_salary_D_value_l2769_276999

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_E : ℕ := 9000
def average_salary : ℕ := 8000
def num_people : ℕ := 5

theorem salary_D_value :
  ∃ (salary_D : ℕ),
    (salary_A + salary_B + salary_C + salary_D + salary_E) / num_people = average_salary ∧
    salary_D = 9000 := by
  sorry

end NUMINAMATH_CALUDE_salary_D_value_l2769_276999


namespace NUMINAMATH_CALUDE_angle_and_function_properties_l2769_276957

-- Define the angle equivalence relation
def angle_equiv (a b : ℝ) : Prop := ∃ k : ℤ, a = b + k * 360

-- Define evenness for functions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Theorem statement
theorem angle_and_function_properties :
  (angle_equiv (-497) 2023) ∧
  (is_even_function (λ x => Real.sin ((2/3)*x - 7*Real.pi/2))) :=
by sorry

end NUMINAMATH_CALUDE_angle_and_function_properties_l2769_276957


namespace NUMINAMATH_CALUDE_lizas_rent_calculation_l2769_276983

def initial_balance : ℚ := 800
def paycheck : ℚ := 1500
def electricity_bill : ℚ := 117
def internet_bill : ℚ := 100
def phone_bill : ℚ := 70
def final_balance : ℚ := 1563

theorem lizas_rent_calculation :
  ∃ (rent : ℚ), 
    initial_balance - rent + paycheck - electricity_bill - internet_bill - phone_bill = final_balance ∧
    rent = 450 :=
by sorry

end NUMINAMATH_CALUDE_lizas_rent_calculation_l2769_276983


namespace NUMINAMATH_CALUDE_consecutive_squares_divisible_by_five_l2769_276949

theorem consecutive_squares_divisible_by_five (n : ℤ) :
  ∃ k : ℤ, (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_divisible_by_five_l2769_276949


namespace NUMINAMATH_CALUDE_equation_solutions_l2769_276938

theorem equation_solutions : 
  ∃! (s : Set ℝ), (∀ x ∈ s, |x - 2| = |x - 5| + |x - 8|) ∧ s = {5, 11} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2769_276938


namespace NUMINAMATH_CALUDE_interest_earned_proof_l2769_276915

def initial_investment : ℝ := 1200
def annual_interest_rate : ℝ := 0.12
def compounding_periods : ℕ := 4

def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

theorem interest_earned_proof :
  let final_amount := compound_interest initial_investment annual_interest_rate compounding_periods
  let total_interest := final_amount - initial_investment
  ∃ ε > 0, |total_interest - 688.22| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_earned_proof_l2769_276915


namespace NUMINAMATH_CALUDE_sum_of_odd_function_at_specific_points_l2769_276900

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The sum of v(-3.14), v(-1.57), v(1.57), and v(3.14) is zero for any odd function v -/
theorem sum_of_odd_function_at_specific_points (v : ℝ → ℝ) (h : IsOdd v) :
  v (-3.14) + v (-1.57) + v 1.57 + v 3.14 = 0 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_odd_function_at_specific_points_l2769_276900


namespace NUMINAMATH_CALUDE_travel_time_difference_l2769_276934

/-- Represents the travel times for different modes of transportation --/
structure TravelTimes where
  drivingTimeMinutes : ℕ
  driveToAirportMinutes : ℕ
  waitToBoardMinutes : ℕ
  exitPlaneMinutes : ℕ

/-- Calculates the total airplane travel time --/
def airplaneTravelTime (t : TravelTimes) : ℕ :=
  t.driveToAirportMinutes + t.waitToBoardMinutes + (t.drivingTimeMinutes / 3) + t.exitPlaneMinutes

/-- Theorem stating the time difference between driving and flying --/
theorem travel_time_difference (t : TravelTimes) 
  (h1 : t.drivingTimeMinutes = 195)
  (h2 : t.driveToAirportMinutes = 10)
  (h3 : t.waitToBoardMinutes = 20)
  (h4 : t.exitPlaneMinutes = 10) :
  t.drivingTimeMinutes - airplaneTravelTime t = 90 := by
  sorry


end NUMINAMATH_CALUDE_travel_time_difference_l2769_276934


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_simplify_to_polynomial_l2769_276968

-- Problem 1
theorem simplify_and_evaluate (a : ℚ) : 
  a = -2 → (a^2 + a) / (a^2 - 3*a) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2/3 := by
  sorry

-- Problem 2
theorem simplify_to_polynomial (x : ℚ) : 
  (x^2 - 1) / (x - 4) / ((x + 1) / (4 - x)) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_simplify_to_polynomial_l2769_276968


namespace NUMINAMATH_CALUDE_find_set_A_l2769_276914

def U : Set ℕ := {1,2,3,4,5,6,7,8}

theorem find_set_A (A B : Set ℕ)
  (h1 : A ∩ (U \ B) = {1,8})
  (h2 : (U \ A) ∩ B = {2,6})
  (h3 : (U \ A) ∩ (U \ B) = {4,7}) :
  A = {1,3,5,8} := by
  sorry

end NUMINAMATH_CALUDE_find_set_A_l2769_276914


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2769_276973

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2769_276973


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2769_276991

/-- Triangle ABC with specific properties -/
structure TriangleABC where
  -- Sides opposite to angles A, B, C
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles A, B, C
  A : ℝ
  B : ℝ
  C : ℝ
  -- Given conditions
  side_angle_relation : (2 * a - b) * Real.cos C = c * Real.cos B
  c_value : c = 2
  area : (1/2) * a * b * Real.sin C = Real.sqrt 3
  -- Triangle properties
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- Main theorem about the properties of Triangle ABC -/
theorem triangle_abc_properties (t : TriangleABC) : 
  t.C = Real.pi / 3 ∧ t.a + t.b + t.c = 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2769_276991


namespace NUMINAMATH_CALUDE_initial_lot_cost_l2769_276959

/-- Represents the cost and composition of a lot of tickets -/
structure TicketLot where
  firstClass : ℕ
  secondClass : ℕ
  firstClassCost : ℕ
  secondClassCost : ℕ

/-- Calculates the total cost of a ticket lot -/
def totalCost (lot : TicketLot) : ℕ :=
  lot.firstClass * lot.firstClassCost + lot.secondClass * lot.secondClassCost

/-- Theorem: The cost of the initial lot of tickets is 110 Rs -/
theorem initial_lot_cost (initialLot interchangedLot : TicketLot) : 
  initialLot.firstClass + initialLot.secondClass = 18 →
  initialLot.firstClassCost = 10 →
  initialLot.secondClassCost = 3 →
  interchangedLot.firstClass = initialLot.secondClass →
  interchangedLot.secondClass = initialLot.firstClass →
  interchangedLot.firstClassCost = initialLot.firstClassCost →
  interchangedLot.secondClassCost = initialLot.secondClassCost →
  totalCost interchangedLot = 124 →
  totalCost initialLot = 110 := by
  sorry

end NUMINAMATH_CALUDE_initial_lot_cost_l2769_276959


namespace NUMINAMATH_CALUDE_flower_bunch_count_l2769_276910

theorem flower_bunch_count (total_flowers : ℕ) (flowers_per_bunch : ℕ) (bunches : ℕ) : 
  total_flowers = 12 * 6 →
  flowers_per_bunch = 9 →
  bunches = total_flowers / flowers_per_bunch →
  bunches = 8 := by
sorry

end NUMINAMATH_CALUDE_flower_bunch_count_l2769_276910


namespace NUMINAMATH_CALUDE_square_of_real_number_proposition_l2769_276913

theorem square_of_real_number_proposition :
  ∃ (p q : Prop), (∀ x : ℝ, x^2 > 0 ∨ x^2 = 0) ↔ (p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_square_of_real_number_proposition_l2769_276913


namespace NUMINAMATH_CALUDE_max_value_of_f_l2769_276976

/-- The quadratic function f(x) = -2x^2 + 16x - 14 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 16 * x - 14

/-- Theorem: The maximum value of f(x) = -2x^2 + 16x - 14 is -14 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = -14 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2769_276976


namespace NUMINAMATH_CALUDE_largest_c_for_no_integer_in_interval_l2769_276969

theorem largest_c_for_no_integer_in_interval :
  ∃ (c : ℝ), c = 6 - 4 * Real.sqrt 2 ∧
  (∀ (n : ℕ), ∀ (k : ℤ),
    (n : ℝ) * Real.sqrt 2 - c / (n : ℝ) < (k : ℝ) →
    (k : ℝ) < (n : ℝ) * Real.sqrt 2 + c / (n : ℝ)) ∧
  (∀ (c' : ℝ), c' > c →
    ∃ (n : ℕ), ∃ (k : ℤ),
      (n : ℝ) * Real.sqrt 2 - c' / (n : ℝ) ≤ (k : ℝ) ∧
      (k : ℝ) ≤ (n : ℝ) * Real.sqrt 2 + c' / (n : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_no_integer_in_interval_l2769_276969


namespace NUMINAMATH_CALUDE_square_sum_equals_product_l2769_276960

theorem square_sum_equals_product (x y z t : ℤ) :
  x^2 + y^2 + z^2 + t^2 = 2*x*y*z*t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_product_l2769_276960


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2769_276971

/-- 
Given a quadratic equation x^2 + bx + 4 = 0 with two equal real roots,
prove that b = 4 or b = -4.
-/
theorem quadratic_equal_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 + b*y + 4 = 0 → y = x) → 
  b = 4 ∨ b = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2769_276971


namespace NUMINAMATH_CALUDE_min_games_correct_l2769_276964

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  white_rook_students : ℕ
  black_elephant_students : ℕ
  games_per_white_student : ℕ
  total_games : ℕ

/-- The specific tournament described in the problem -/
def tournament : ChessTournament :=
  { white_rook_students := 15
  , black_elephant_students := 20
  , games_per_white_student := 20
  , total_games := 300 }

/-- The minimum number of games after which one can guarantee
    that at least one White Rook student has played all their games -/
def min_games_for_guarantee (t : ChessTournament) : ℕ :=
  (t.white_rook_students - 1) * t.games_per_white_student

theorem min_games_correct (t : ChessTournament) :
  min_games_for_guarantee t = (t.white_rook_students - 1) * t.games_per_white_student ∧
  min_games_for_guarantee t < t.total_games ∧
  ∀ n, n < min_games_for_guarantee t → 
    ∃ i j, i < t.white_rook_students ∧ j < t.games_per_white_student ∧
           n < i * t.games_per_white_student + j :=
by sorry

#eval min_games_for_guarantee tournament  -- Should output 280

end NUMINAMATH_CALUDE_min_games_correct_l2769_276964


namespace NUMINAMATH_CALUDE_B_subset_A_l2769_276961

def A : Set ℕ := {2, 0, 3}
def B : Set ℕ := {2, 3}

theorem B_subset_A : B ⊆ A := by sorry

end NUMINAMATH_CALUDE_B_subset_A_l2769_276961


namespace NUMINAMATH_CALUDE_agricultural_machinery_growth_rate_l2769_276933

/-- The average growth rate for May and June in an agricultural machinery factory --/
theorem agricultural_machinery_growth_rate :
  ∀ (april_production : ℕ) (total_production : ℕ) (growth_rate : ℝ),
  april_production = 500 →
  total_production = 1820 →
  april_production + 
    april_production * (1 + growth_rate) + 
    april_production * (1 + growth_rate)^2 = total_production →
  growth_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_agricultural_machinery_growth_rate_l2769_276933


namespace NUMINAMATH_CALUDE_gcd_2475_7350_l2769_276936

theorem gcd_2475_7350 : Nat.gcd 2475 7350 = 225 := by sorry

end NUMINAMATH_CALUDE_gcd_2475_7350_l2769_276936


namespace NUMINAMATH_CALUDE_problem_solution_l2769_276942

theorem problem_solution (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : 3 ≤ y ∧ y ≤ 5) :
  (4 ≤ x + y ∧ x + y ≤ 8) ∧
  (∀ a b, (1 ≤ a ∧ a ≤ 3 ∧ 3 ≤ b ∧ b ≤ 5) → x + y + 1/x + 16/y ≤ a + b + 1/a + 16/b) ∧
  (∃ a b, (1 ≤ a ∧ a ≤ 3 ∧ 3 ≤ b ∧ b ≤ 5) ∧ x + y + 1/x + 16/y = a + b + 1/a + 16/b ∧ a + b + 1/a + 16/b = 10) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l2769_276942


namespace NUMINAMATH_CALUDE_jenny_mike_earnings_l2769_276946

theorem jenny_mike_earnings (t : ℝ) : 
  (t + 3) * (4 * t - 6) = (4 * t - 7) * (t + 3) + 3 → t = 3 := by
  sorry

end NUMINAMATH_CALUDE_jenny_mike_earnings_l2769_276946


namespace NUMINAMATH_CALUDE_average_team_goals_l2769_276931

-- Define the average goals per game for each player
def carter_goals : ℚ := 4
def shelby_goals : ℚ := carter_goals / 2
def judah_goals : ℚ := 2 * shelby_goals - 3
def morgan_goals : ℚ := judah_goals + 1
def alex_goals : ℚ := carter_goals / 2 - 2
def taylor_goals : ℚ := 1 / 3

-- Define the total goals per game for the team
def team_goals : ℚ := carter_goals + shelby_goals + judah_goals + morgan_goals + alex_goals + taylor_goals

-- Theorem statement
theorem average_team_goals : team_goals = 28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_team_goals_l2769_276931


namespace NUMINAMATH_CALUDE_auction_price_problem_l2769_276925

theorem auction_price_problem (tv_initial_cost : ℝ) (tv_price_increase_ratio : ℝ) 
  (phone_price_increase_ratio : ℝ) (total_received : ℝ) :
  tv_initial_cost = 500 →
  tv_price_increase_ratio = 2 / 5 →
  phone_price_increase_ratio = 0.4 →
  total_received = 1260 →
  ∃ (phone_initial_price : ℝ),
    phone_initial_price = 400 ∧
    total_received = tv_initial_cost * (1 + tv_price_increase_ratio) + 
                     phone_initial_price * (1 + phone_price_increase_ratio) :=
by sorry

end NUMINAMATH_CALUDE_auction_price_problem_l2769_276925


namespace NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_294_l2769_276932

theorem sum_of_two_smallest_prime_factors_of_294 :
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p < q ∧
    p ∣ 294 ∧ 
    q ∣ 294 ∧
    (∀ (r : ℕ), Nat.Prime r → r ∣ 294 → r = p ∨ r ≥ q) ∧
    p + q = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_294_l2769_276932


namespace NUMINAMATH_CALUDE_line_contains_point_l2769_276945

/-- Given a line equation 2 - kx = -4y that contains the point (2, -1), prove that k = -1 -/
theorem line_contains_point (k : ℝ) : 
  (2 - k * 2 = -4 * (-1)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l2769_276945


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2769_276917

theorem sin_cos_identity : 
  (4 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180)) / Real.cos (20 * π / 180) - Real.tan (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2769_276917


namespace NUMINAMATH_CALUDE_tommys_estimate_l2769_276965

theorem tommys_estimate (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - ε) > x - y := by
  sorry

end NUMINAMATH_CALUDE_tommys_estimate_l2769_276965


namespace NUMINAMATH_CALUDE_speed_ratio_l2769_276954

-- Define the speeds of A and B
def speed_A : ℝ := sorry
def speed_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := 600

-- Define the time when they are first equidistant
def first_equidistant_time : ℝ := 3

-- Define the time when they are second equidistant
def second_equidistant_time : ℝ := 12

-- Define the condition for being equidistant at the first time
def first_equidistant_condition : Prop :=
  (first_equidistant_time * speed_A) = abs (-initial_B_position + first_equidistant_time * speed_B)

-- Define the condition for being equidistant at the second time
def second_equidistant_condition : Prop :=
  (second_equidistant_time * speed_A) = abs (-initial_B_position + second_equidistant_time * speed_B)

-- Theorem stating that the ratio of speeds is 1:5
theorem speed_ratio : 
  first_equidistant_condition → second_equidistant_condition → speed_A / speed_B = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_speed_ratio_l2769_276954


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2769_276953

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2769_276953


namespace NUMINAMATH_CALUDE_tailor_trim_problem_l2769_276907

/-- Given a square cloth with side length 18 feet, if 4 feet are trimmed from two opposite edges
    and x feet are trimmed from the other two edges, resulting in 120 square feet of remaining cloth,
    then x = 6. -/
theorem tailor_trim_problem (x : ℝ) : 
  (18 : ℝ) > 0 ∧ x > 0 ∧ (18 - 4 - 4 : ℝ) * (18 - x) = 120 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_tailor_trim_problem_l2769_276907


namespace NUMINAMATH_CALUDE_gcd_n4_plus_16_and_n_plus_3_l2769_276970

theorem gcd_n4_plus_16_and_n_plus_3 (n : ℕ) (h1 : n > 9) (h2 : n ≠ 94) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n4_plus_16_and_n_plus_3_l2769_276970


namespace NUMINAMATH_CALUDE_symmetric_sequence_second_term_l2769_276903

def is_symmetric (s : Fin 21 → ℕ) : Prop :=
  ∀ i : Fin 21, s i = s (20 - i)

def is_arithmetic_sequence (s : Fin 11 → ℕ) (a d : ℕ) : Prop :=
  ∀ i : Fin 11, s i = a + i * d

theorem symmetric_sequence_second_term 
  (c : Fin 21 → ℕ) 
  (h_sym : is_symmetric c) 
  (h_arith : is_arithmetic_sequence (fun i => c (i + 10)) 1 2) : 
  c 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sequence_second_term_l2769_276903


namespace NUMINAMATH_CALUDE_jelly_servings_count_jelly_servings_mixed_number_l2769_276937

-- Define the total amount of jelly in tablespoons
def total_jelly : ℚ := 113 / 3

-- Define the serving size in tablespoons
def serving_size : ℚ := 3 / 2

-- Define the number of servings
def num_servings : ℚ := total_jelly / serving_size

-- Theorem to prove
theorem jelly_servings_count :
  num_servings = 226 / 9 := by sorry

-- Proof that the result is equivalent to 25 1/9
theorem jelly_servings_mixed_number :
  ∃ (n : ℕ) (m : ℚ), n = 25 ∧ m = 1 / 9 ∧ num_servings = n + m := by sorry

end NUMINAMATH_CALUDE_jelly_servings_count_jelly_servings_mixed_number_l2769_276937


namespace NUMINAMATH_CALUDE_lotus_growth_model_l2769_276990

def y (x : ℕ) : ℚ := (32 / 3) * (3 / 2) ^ x

theorem lotus_growth_model :
  (y 2 = 24) ∧ 
  (y 3 = 36) ∧ 
  (∀ n : ℕ, y n ≤ 10 * y 0 → n ≤ 5) ∧
  (y 6 > 10 * y 0) := by
  sorry

end NUMINAMATH_CALUDE_lotus_growth_model_l2769_276990


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_parallel_lines_l2769_276978

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_parallel_lines
  (l m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : contained_in m β)
  (h3 : parallel_lines l m) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_parallel_lines_l2769_276978


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l2769_276997

theorem exponential_equation_solution :
  ∃ y : ℝ, (3 : ℝ) ^ (y - 4) = 9 ^ (y + 2) → y = -8 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l2769_276997


namespace NUMINAMATH_CALUDE_zeros_when_a_1_b_neg_2_range_of_a_for_two_distinct_zeros_l2769_276926

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

-- Theorem for part (1)
theorem zeros_when_a_1_b_neg_2 :
  let f := f 1 (-2)
  ∀ x, f x = 0 ↔ x = 3 ∨ x = -1 := by sorry

-- Theorem for part (2)
theorem range_of_a_for_two_distinct_zeros :
  ∀ a : ℝ, (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) ↔ 0 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_zeros_when_a_1_b_neg_2_range_of_a_for_two_distinct_zeros_l2769_276926


namespace NUMINAMATH_CALUDE_cemc_employee_change_l2769_276967

/-- The net change in employees for Canadian Excellent Mathematics Corporation in 2018 -/
theorem cemc_employee_change (t : ℕ) (h : t = 120) : 
  (((t : ℚ) * (1 + 0.25) + (40 : ℚ) * (1 - 0.35)) - (t + 40 : ℚ)).floor = 16 := by
  sorry

end NUMINAMATH_CALUDE_cemc_employee_change_l2769_276967


namespace NUMINAMATH_CALUDE_new_average_after_exclusion_l2769_276908

/-- Theorem: New average after excluding students with low marks -/
theorem new_average_after_exclusion
  (total_students : ℕ)
  (original_average : ℚ)
  (excluded_students : ℕ)
  (excluded_average : ℚ)
  (h1 : total_students = 33)
  (h2 : original_average = 90)
  (h3 : excluded_students = 3)
  (h4 : excluded_average = 40) :
  let remaining_students := total_students - excluded_students
  let total_marks := total_students * original_average
  let excluded_marks := excluded_students * excluded_average
  let remaining_marks := total_marks - excluded_marks
  (remaining_marks / remaining_students : ℚ) = 95 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_exclusion_l2769_276908


namespace NUMINAMATH_CALUDE_inequality_proof_l2769_276979

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 1 ≥ x*y + x + y := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2769_276979


namespace NUMINAMATH_CALUDE_intersection_M_N_l2769_276977

def M : Set ℝ := {x | (x - 3) / (x + 1) ≤ 0}
def N : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_M_N : M ∩ N = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2769_276977


namespace NUMINAMATH_CALUDE_sum_of_multiples_l2769_276993

theorem sum_of_multiples (a b : ℤ) (ha : 6 ∣ a) (hb : 9 ∣ b) : 3 ∣ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l2769_276993


namespace NUMINAMATH_CALUDE_speed_equivalence_l2769_276922

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def speed_mps : ℝ := 18.334799999999998

/-- The speed in kilometers per hour -/
def speed_kmph : ℝ := 66.00528

/-- Theorem stating that the given speed in km/h is equivalent to the speed in m/s -/
theorem speed_equivalence : speed_kmph = speed_mps * mps_to_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l2769_276922


namespace NUMINAMATH_CALUDE_largest_perimeter_l2769_276919

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Checks if the given lengths can form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem: The largest possible perimeter of a triangle with sides 7, 9, and an integer y is 31 --/
theorem largest_perimeter :
  ∀ y : ℕ,
    is_valid_triangle ⟨7, 9, y⟩ →
    perimeter ⟨7, 9, y⟩ ≤ 31 ∧
    ∃ (y' : ℕ), is_valid_triangle ⟨7, 9, y'⟩ ∧ perimeter ⟨7, 9, y'⟩ = 31 :=
by sorry


end NUMINAMATH_CALUDE_largest_perimeter_l2769_276919


namespace NUMINAMATH_CALUDE_total_stars_l2769_276928

theorem total_stars (num_students : ℕ) (stars_per_student : ℕ) 
  (h1 : num_students = 186) 
  (h2 : stars_per_student = 5) : 
  num_students * stars_per_student = 930 := by
  sorry

end NUMINAMATH_CALUDE_total_stars_l2769_276928


namespace NUMINAMATH_CALUDE_orange_count_l2769_276984

/-- Given a fruit farm that packs oranges, calculate the total number of oranges. -/
theorem orange_count (oranges_per_box : ℝ) (boxes_per_day : ℝ) 
  (h1 : oranges_per_box = 10.0) 
  (h2 : boxes_per_day = 2650.0) : 
  oranges_per_box * boxes_per_day = 26500.0 := by
  sorry

#check orange_count

end NUMINAMATH_CALUDE_orange_count_l2769_276984


namespace NUMINAMATH_CALUDE_ellipse_problem_l2769_276909

/-- The ellipse problem -/
theorem ellipse_problem (a b : ℝ) (P : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧ a > b →  -- a and b are positive real numbers with a > b
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) →  -- P is on the ellipse
  ((P.1 + 1)^2 + P.2^2)^(1/2) - ((P.1 - 1)^2 + P.2^2)^(1/2) = a / 2 →  -- |PF₁| - |PF₂| = a/2
  (P.1 - 1) * (P.1 + 1) + P.2^2 = 0 →  -- PF₂ is perpendicular to F₁F₂
  (∃ (m : ℝ), (1 + m * P.2)^2 / 4 + P.2^2 / 3 = 1 ∧  -- equation of ellipse G
              (∃ (M N : ℝ × ℝ), M ≠ N ∧  -- M and N are distinct points
                (M.1^2 / 4 + M.2^2 / 3 = 1) ∧ (N.1^2 / 4 + N.2^2 / 3 = 1) ∧  -- M and N are on the ellipse
                (M.1 - 1 = m * M.2) ∧ (N.1 - 1 = m * N.2) ∧  -- M and N are on line l passing through F₂
                ((0 - M.2) * (N.1 - 1)) / ((0 - N.2) * (M.1 - 1)) = 2))  -- ratio of areas of triangles BF₂M and BF₂N is 2
  := by sorry


end NUMINAMATH_CALUDE_ellipse_problem_l2769_276909


namespace NUMINAMATH_CALUDE_max_product_863_l2769_276940

/-- A type representing the digits we can use -/
inductive Digit
  | three
  | five
  | six
  | eight
  | nine

/-- A function to convert our Digit type to a natural number -/
def digit_to_nat (d : Digit) : Nat :=
  match d with
  | Digit.three => 3
  | Digit.five => 5
  | Digit.six => 6
  | Digit.eight => 8
  | Digit.nine => 9

/-- A type representing a valid combination of digits -/
structure DigitCombination where
  a : Digit
  b : Digit
  c : Digit
  d : Digit
  e : Digit
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- Calculate the product of the three-digit and two-digit numbers -/
def calculate_product (combo : DigitCombination) : Nat :=
  (100 * digit_to_nat combo.a + 10 * digit_to_nat combo.b + digit_to_nat combo.c) *
  (10 * digit_to_nat combo.d + digit_to_nat combo.e)

/-- The main theorem -/
theorem max_product_863 :
  ∀ combo : DigitCombination,
    calculate_product combo ≤ calculate_product
      { a := Digit.eight
      , b := Digit.six
      , c := Digit.three
      , d := Digit.nine
      , e := Digit.five
      , all_different := by simp } :=
by
  sorry


end NUMINAMATH_CALUDE_max_product_863_l2769_276940


namespace NUMINAMATH_CALUDE_vector_properties_l2769_276981

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, 1]
def c : Fin 2 → ℝ := ![3, -5]

theorem vector_properties :
  (∃ (k : ℝ), a + 2 • b = k • c) ∧
  ‖a + c‖ = 2 * ‖b‖ := by
sorry

end NUMINAMATH_CALUDE_vector_properties_l2769_276981


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequality_system_l2769_276988

theorem integer_solutions_of_inequality_system :
  {x : ℤ | x + 2 > 0 ∧ 2 * x - 1 ≤ 0} = {-1, 0} := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequality_system_l2769_276988


namespace NUMINAMATH_CALUDE_price_equation_system_l2769_276956

/-- Represents the price of a basketball in yuan -/
def basketball_price : ℝ := sorry

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := sorry

/-- The total cost of 3 basketballs and 4 soccer balls is 330 yuan -/
axiom total_cost : 3 * basketball_price + 4 * soccer_ball_price = 330

/-- The price of a basketball is 5 yuan less than the price of a soccer ball -/
axiom price_difference : basketball_price = soccer_ball_price - 5

/-- The system of equations accurately represents the given conditions -/
theorem price_equation_system : 
  (3 * basketball_price + 4 * soccer_ball_price = 330) ∧ 
  (basketball_price = soccer_ball_price - 5) :=
by sorry

end NUMINAMATH_CALUDE_price_equation_system_l2769_276956


namespace NUMINAMATH_CALUDE_ellipse_max_value_l2769_276929

/-- The maximum value of x + 2y for points on the ellipse x^2/16 + y^2/12 = 1 is 8 -/
theorem ellipse_max_value (x y : ℝ) : 
  x^2/16 + y^2/12 = 1 → x + 2*y ≤ 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l2769_276929


namespace NUMINAMATH_CALUDE_region_is_rectangle_l2769_276905

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The region defined by the given inequalities -/
def Region : Set Point2D :=
  {p : Point2D | -1 ≤ p.x ∧ p.x ≤ 1 ∧ 2 ≤ p.y ∧ p.y ≤ 4}

/-- Definition of a rectangle in 2D -/
def IsRectangle (S : Set Point2D) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), x1 < x2 ∧ y1 < y2 ∧
    S = {p : Point2D | x1 ≤ p.x ∧ p.x ≤ x2 ∧ y1 ≤ p.y ∧ p.y ≤ y2}

/-- Theorem: The defined region is a rectangle -/
theorem region_is_rectangle : IsRectangle Region := by
  sorry

end NUMINAMATH_CALUDE_region_is_rectangle_l2769_276905


namespace NUMINAMATH_CALUDE_bennetts_brothers_l2769_276955

theorem bennetts_brothers (aaron_brothers : ℕ) (bennett_brothers : ℕ) : 
  aaron_brothers = 4 → 
  bennett_brothers = 2 * aaron_brothers - 2 → 
  bennett_brothers = 6 := by
sorry

end NUMINAMATH_CALUDE_bennetts_brothers_l2769_276955


namespace NUMINAMATH_CALUDE_total_distance_four_runners_l2769_276930

/-- The total distance run by four runners, where one runner ran 51 miles
    and the other three ran the same distance of 48 miles each, is 195 miles. -/
theorem total_distance_four_runners :
  ∀ (katarina tomas tyler harriet : ℕ),
    katarina = 51 →
    tomas = 48 →
    tyler = 48 →
    harriet = 48 →
    katarina + tomas + tyler + harriet = 195 :=
by
  sorry

end NUMINAMATH_CALUDE_total_distance_four_runners_l2769_276930


namespace NUMINAMATH_CALUDE_barbier_theorem_for_delta_curves_l2769_276987

-- Define a Δ-curve
class DeltaCurve where
  height : ℝ
  is_convex : Bool
  can_rotate_in_triangle : Bool
  always_touches_sides : Bool

-- Define the length of a Δ-curve
def length_of_delta_curve (K : DeltaCurve) : ℝ := sorry

-- Define the approximation of a Δ-curve by circular arcs
def approximate_by_circular_arcs (K : DeltaCurve) (n : ℕ) : DeltaCurve := sorry

-- Theorem: The length of any Δ-curve with height h is 2πh/3
theorem barbier_theorem_for_delta_curves (K : DeltaCurve) :
  length_of_delta_curve K = 2 * Real.pi * K.height / 3 := by sorry

end NUMINAMATH_CALUDE_barbier_theorem_for_delta_curves_l2769_276987


namespace NUMINAMATH_CALUDE_tangent_slope_values_l2769_276958

-- Define the curve
def curve (x : ℝ) : ℝ := x^2

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 2 * x

-- Define the tangent line equation
def tangent_line (t : ℝ) (x : ℝ) : ℝ := t^2 + 2*t*(x - t)

-- Theorem statement
theorem tangent_slope_values :
  ∃ (t : ℝ), (tangent_line t 1 = 0) ∧ 
  ((curve_derivative t = 0) ∨ (curve_derivative t = 4)) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_values_l2769_276958


namespace NUMINAMATH_CALUDE_inverse_of_matrix_A_l2769_276995

theorem inverse_of_matrix_A :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; 1, 2]
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1, -2; -1/2, 3/2]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_matrix_A_l2769_276995


namespace NUMINAMATH_CALUDE_quadrilateral_interior_angles_mean_l2769_276996

/-- The mean value of the measures of the four interior angles of any quadrilateral is 90°. -/
theorem quadrilateral_interior_angles_mean :
  let sum_of_angles : ℝ := 360
  let number_of_angles : ℕ := 4
  (sum_of_angles / number_of_angles : ℝ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_interior_angles_mean_l2769_276996


namespace NUMINAMATH_CALUDE_correct_calculation_l2769_276982

theorem correct_calculation : 
  (Real.sqrt 27 / Real.sqrt 3 = 3) ∧ 
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) ∧ 
  (5 * Real.sqrt 2 - 4 * Real.sqrt 2 ≠ 1) ∧ 
  (2 * Real.sqrt 3 * 3 * Real.sqrt 3 ≠ 6 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_correct_calculation_l2769_276982


namespace NUMINAMATH_CALUDE_f_of_3_equals_0_l2769_276985

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2 - 2*(x - 1)

-- State the theorem
theorem f_of_3_equals_0 : f 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_0_l2769_276985


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2769_276947

/-- Given two planar vectors a and b satisfying certain conditions, 
    prove that the cosine of the angle between them is -√10/10 -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) : 
  (2 • a + b = (3, 3)) → 
  (a - 2 • b = (-1, 4)) → 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = -Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2769_276947


namespace NUMINAMATH_CALUDE_smallest_possible_d_l2769_276920

theorem smallest_possible_d (c d : ℝ) : 
  (1 < c) → 
  (c < d) → 
  (1 + c ≤ d) → 
  (1 / c + 1 / d ≤ 1) → 
  d ≥ (3 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l2769_276920


namespace NUMINAMATH_CALUDE_unique_prime_digit_l2769_276950

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The six-digit number as a function of B -/
def number (B : ℕ) : ℕ := 304200 + B

/-- Theorem stating that there is a unique B that makes the number prime, and it's 1 -/
theorem unique_prime_digit :
  ∃! B : ℕ, B < 10 ∧ isPrime (number B) ∧ B = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_digit_l2769_276950


namespace NUMINAMATH_CALUDE_trig_special_angles_sum_l2769_276998

theorem trig_special_angles_sum : 
  Real.sin (π / 2) + 2 * Real.cos 0 - 3 * Real.sin (3 * π / 2) + 10 * Real.cos π = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_special_angles_sum_l2769_276998


namespace NUMINAMATH_CALUDE_farmer_land_problem_l2769_276943

theorem farmer_land_problem (original_land : ℚ) : 
  (9 / 10 : ℚ) * original_land = 10 → original_land = 11 + 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_problem_l2769_276943


namespace NUMINAMATH_CALUDE_dividend_proof_l2769_276963

theorem dividend_proof (divisor quotient dividend : ℕ) : 
  divisor = 12 → quotient = 999809 → dividend = 11997708 → 
  dividend / divisor = quotient ∧ dividend % divisor = 0 := by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l2769_276963


namespace NUMINAMATH_CALUDE_square_2209_product_l2769_276921

theorem square_2209_product (x : ℤ) (h : x^2 = 2209) : (x + 2) * (x - 2) = 2205 := by
  sorry

end NUMINAMATH_CALUDE_square_2209_product_l2769_276921


namespace NUMINAMATH_CALUDE_wang_yue_more_stable_l2769_276994

def li_na_scores : List ℝ := [80, 70, 90, 70]
def wang_yue_scores (a : ℝ) : List ℝ := [80, a, 70, 90]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem wang_yue_more_stable (a : ℝ) :
  a ≥ 70 →
  median li_na_scores + 5 = median (wang_yue_scores a) →
  variance (wang_yue_scores a) < variance li_na_scores :=
sorry

end NUMINAMATH_CALUDE_wang_yue_more_stable_l2769_276994
