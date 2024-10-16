import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_decimal_expansion_unique_l3937_393767

theorem sqrt_decimal_expansion_unique 
  (p n : ℚ) 
  (hp : 0 < p) 
  (hn : 0 < n) 
  (hp_not_square : ¬ ∃ (m : ℚ), p = m ^ 2) 
  (hn_not_square : ¬ ∃ (m : ℚ), n = m ^ 2) : 
  ¬ ∃ (k : ℤ), Real.sqrt p - Real.sqrt n = k := by
  sorry

end NUMINAMATH_CALUDE_sqrt_decimal_expansion_unique_l3937_393767


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l3937_393761

theorem line_equation_through_point_with_slope (x y : ℝ) :
  let point : ℝ × ℝ := (-2, 1)
  let slope : ℝ := Real.tan (135 * π / 180)
  (x - point.1) * slope = y - point.2 →
  x + y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l3937_393761


namespace NUMINAMATH_CALUDE_d_equals_square_iff_l3937_393719

/-- Move the last digit of a number to the first position -/
def moveLastToFirst (a : ℕ) : ℕ :=
  sorry

/-- Square a number -/
def square (b : ℕ) : ℕ :=
  sorry

/-- Move the first digit of a number to the end -/
def moveFirstToLast (c : ℕ) : ℕ :=
  sorry

/-- The d(a) function as described in the problem -/
def d (a : ℕ) : ℕ :=
  moveFirstToLast (square (moveLastToFirst a))

/-- Check if a number is of the form 222...21 -/
def is222_21 (a : ℕ) : Prop :=
  sorry

/-- The main theorem -/
theorem d_equals_square_iff (a : ℕ) :
  d a = a^2 ↔ a = 1 ∨ a = 2 ∨ a = 3 ∨ is222_21 a :=
sorry

end NUMINAMATH_CALUDE_d_equals_square_iff_l3937_393719


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3937_393785

theorem geometric_series_ratio (r : ℝ) (h : r ≠ 1) :
  (∃ (a : ℝ), a / (1 - r) = 64 * (a * r^4) / (1 - r)) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3937_393785


namespace NUMINAMATH_CALUDE_carrot_count_l3937_393755

/-- The number of carrots initially on the scale -/
def initial_carrots : ℕ := 20

/-- The total weight of carrots in grams -/
def total_weight : ℕ := 3640

/-- The average weight of remaining carrots in grams -/
def avg_weight_remaining : ℕ := 180

/-- The average weight of removed carrots in grams -/
def avg_weight_removed : ℕ := 190

/-- The number of removed carrots -/
def removed_carrots : ℕ := 4

theorem carrot_count : 
  total_weight = (initial_carrots - removed_carrots) * avg_weight_remaining + 
                 removed_carrots * avg_weight_removed := by
  sorry

end NUMINAMATH_CALUDE_carrot_count_l3937_393755


namespace NUMINAMATH_CALUDE_sum_of_numbers_ge_04_l3937_393773

theorem sum_of_numbers_ge_04 : 
  let numbers := [0.8, 1/2, 0.3]
  let sum_ge_04 := (numbers.filter (λ x => x ≥ 0.4)).sum
  sum_ge_04 = 1.3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_ge_04_l3937_393773


namespace NUMINAMATH_CALUDE_loss_calculation_l3937_393757

/-- Calculates the loss for the investor with larger capital -/
def loss_larger_investor (total_loss : ℚ) : ℚ :=
  (9 / 10) * total_loss

theorem loss_calculation (total_loss : ℚ) (pyarelal_loss : ℚ) 
  (h1 : total_loss = 900) 
  (h2 : pyarelal_loss = loss_larger_investor total_loss) : 
  pyarelal_loss = 810 := by
  sorry

end NUMINAMATH_CALUDE_loss_calculation_l3937_393757


namespace NUMINAMATH_CALUDE_point_p_coordinates_l3937_393790

/-- A point P with coordinates (m+3, m-1) that lies on the y-axis -/
structure PointP where
  m : ℝ
  x : ℝ := m + 3
  y : ℝ := m - 1
  on_y_axis : x = 0

/-- Theorem: If a point P(m+3, m-1) lies on the y-axis, then its coordinates are (0, -4) -/
theorem point_p_coordinates (P : PointP) : (P.x = 0 ∧ P.y = -4) := by
  sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l3937_393790


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3937_393797

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := ∃ (y : ℝ), x = y^2

-- Define the concept of simplest quadratic radical
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ 
  ∀ y : ℝ, (QuadraticRadical y ∧ y ≠ x) → (∃ z : ℝ, z ≠ 1 ∧ y = z * x)

-- Theorem statement
theorem simplest_quadratic_radical :
  SimplestQuadraticRadical (Real.sqrt 6) ∧
  ¬SimplestQuadraticRadical (Real.sqrt 12) ∧
  ¬SimplestQuadraticRadical (Real.sqrt (1/3)) ∧
  ¬SimplestQuadraticRadical (Real.sqrt 0.3) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3937_393797


namespace NUMINAMATH_CALUDE_isabel_homework_completion_l3937_393778

/-- Given that Isabel had 72.0 homework problems in total, each problem has 5 sub tasks,
    and she has to solve 200 sub tasks, prove that she finished 40 homework problems. -/
theorem isabel_homework_completion (total : ℝ) (subtasks_per_problem : ℕ) (subtasks_solved : ℕ) 
    (h1 : total = 72.0)
    (h2 : subtasks_per_problem = 5)
    (h3 : subtasks_solved = 200) :
    (subtasks_solved : ℝ) / subtasks_per_problem = 40 := by
  sorry

#check isabel_homework_completion

end NUMINAMATH_CALUDE_isabel_homework_completion_l3937_393778


namespace NUMINAMATH_CALUDE_milk_exchange_theorem_l3937_393799

/-- Represents the number of liters of milk obtainable from a given number of empty bottles -/
def milk_obtained (empty_bottles : ℕ) : ℕ :=
  let full_bottles := empty_bottles / 4
  let remaining_empty := empty_bottles % 4
  if full_bottles = 0 then
    0
  else
    full_bottles + milk_obtained (full_bottles + remaining_empty)

/-- Theorem stating that 43 empty bottles can be exchanged for 14 liters of milk -/
theorem milk_exchange_theorem :
  milk_obtained 43 = 14 := by
  sorry

end NUMINAMATH_CALUDE_milk_exchange_theorem_l3937_393799


namespace NUMINAMATH_CALUDE_find_lesser_number_l3937_393730

theorem find_lesser_number (x y : ℝ) : 
  x + y = 60 → 
  x - y = 10 → 
  min x y = 25 := by
sorry

end NUMINAMATH_CALUDE_find_lesser_number_l3937_393730


namespace NUMINAMATH_CALUDE_exam_average_l3937_393759

theorem exam_average (total_candidates : ℕ) (first_ten_avg : ℚ) (last_eleven_avg : ℚ) (eleventh_candidate_score : ℕ) :
  total_candidates = 22 →
  first_ten_avg = 55 →
  last_eleven_avg = 40 →
  eleventh_candidate_score = 66 →
  (((first_ten_avg * 10) + eleventh_candidate_score + (last_eleven_avg * 11 - eleventh_candidate_score)) / total_candidates : ℚ) = 45 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l3937_393759


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3937_393788

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 2*b = 2) :
  (1/a + 1/b) ≥ 3/2 + Real.sqrt 2 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ 1/a₀ + 1/b₀ = 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3937_393788


namespace NUMINAMATH_CALUDE_bus_assignment_count_l3937_393725

def num_buses : ℕ := 6
def num_destinations : ℕ := 4
def num_restricted_buses : ℕ := 2

def choose (n k : ℕ) : ℕ := Nat.choose n k

def arrange (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem bus_assignment_count : 
  choose num_destinations 1 * arrange (num_buses - num_restricted_buses) (num_destinations - 1) = 240 := by
  sorry

end NUMINAMATH_CALUDE_bus_assignment_count_l3937_393725


namespace NUMINAMATH_CALUDE_marley_has_31_fruits_l3937_393717

/-- The number of fruits Marley has -/
def marley_total_fruits : ℕ :=
  let louis_oranges : ℕ := 5
  let louis_apples : ℕ := 3
  let samantha_oranges : ℕ := 8
  let samantha_apples : ℕ := 7
  let marley_oranges : ℕ := 2 * louis_oranges
  let marley_apples : ℕ := 3 * samantha_apples
  marley_oranges + marley_apples

/-- Theorem stating that Marley has 31 fruits in total -/
theorem marley_has_31_fruits : marley_total_fruits = 31 := by
  sorry

end NUMINAMATH_CALUDE_marley_has_31_fruits_l3937_393717


namespace NUMINAMATH_CALUDE_systematic_sampling_first_group_l3937_393743

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalStudents : Nat
  numGroups : Nat
  sampleSize : Nat
  groupSize : Nat
  sixteenthGroupDraw : Nat

/-- Theorem for systematic sampling -/
theorem systematic_sampling_first_group
  (setup : SystematicSampling)
  (h1 : setup.totalStudents = 160)
  (h2 : setup.numGroups = 20)
  (h3 : setup.sampleSize = 20)
  (h4 : setup.groupSize = setup.totalStudents / setup.numGroups)
  (h5 : setup.sixteenthGroupDraw = 126) :
  ∃ (firstGroupDraw : Nat), firstGroupDraw = 6 ∧
    setup.sixteenthGroupDraw = (16 - 1) * setup.groupSize + firstGroupDraw :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_group_l3937_393743


namespace NUMINAMATH_CALUDE_vector_relations_l3937_393712

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define parallelism
def parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

-- Define collinearity (same as parallelism for vectors)
def collinear (v w : V) : Prop := parallel v w

-- Theorem: Only "Equal vectors must be collinear" is true
theorem vector_relations :
  (∀ v w : V, parallel v w → v = w) = false ∧ 
  (∀ v w : V, v ≠ w → ¬(parallel v w)) = false ∧
  (∀ v w : V, collinear v w → v = w) = false ∧
  (∀ v w : V, v = w → collinear v w) = true :=
sorry

end NUMINAMATH_CALUDE_vector_relations_l3937_393712


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l3937_393715

/-- Simple interest calculation -/
theorem simple_interest_time_calculation
  (principal : ℝ)
  (simple_interest : ℝ)
  (rate : ℝ)
  (h1 : principal = 400)
  (h2 : simple_interest = 140)
  (h3 : rate = 17.5) :
  (simple_interest * 100) / (principal * rate) = 2 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l3937_393715


namespace NUMINAMATH_CALUDE_permutation_100_2_l3937_393728

/-- The number of permutations of n distinct objects taken k at a time -/
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

/-- The permutation A₁₀₀² equals 9900 -/
theorem permutation_100_2 : permutation 100 2 = 9900 := by sorry

end NUMINAMATH_CALUDE_permutation_100_2_l3937_393728


namespace NUMINAMATH_CALUDE_max_difference_theorem_l3937_393742

/-- The maximum difference between the sum of ball numbers for two people --/
def maxDifference : ℕ := 9644

/-- The total number of balls --/
def totalBalls : ℕ := 200

/-- The starting number of the balls --/
def startNumber : ℕ := 101

/-- The ending number of the balls --/
def endNumber : ℕ := 300

/-- The number of balls each person takes --/
def ballsPerPerson : ℕ := 100

/-- The ball number that person A takes --/
def ballA : ℕ := 102

/-- The ball number that person B takes --/
def ballB : ℕ := 280

theorem max_difference_theorem :
  ∀ (sumA sumB : ℕ),
  sumA ≤ (startNumber + endNumber) * ballsPerPerson / 2 - (ballB - ballA) →
  sumB ≥ (startNumber + endNumber - totalBalls + 1) * ballsPerPerson / 2 + (ballB - ballA) →
  sumA - sumB ≤ maxDifference :=
sorry

end NUMINAMATH_CALUDE_max_difference_theorem_l3937_393742


namespace NUMINAMATH_CALUDE_five_digit_reverse_multiplication_l3937_393765

theorem five_digit_reverse_multiplication (a b c d e : Nat) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 →
  4 * (a * 10000 + b * 1000 + c * 100 + d * 10 + e) = e * 10000 + d * 1000 + c * 100 + b * 10 + a →
  a + b + c + d + e = 27 := by
sorry

end NUMINAMATH_CALUDE_five_digit_reverse_multiplication_l3937_393765


namespace NUMINAMATH_CALUDE_temperatures_median_and_range_l3937_393747

def temperatures : List ℝ := [12, 9, 10, 6, 11, 12, 17]

def median (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

theorem temperatures_median_and_range :
  median temperatures = 11 ∧ range temperatures = 11 := by
  sorry

end NUMINAMATH_CALUDE_temperatures_median_and_range_l3937_393747


namespace NUMINAMATH_CALUDE_total_sales_proof_l3937_393786

def window_screen_sales (march_sales : ℕ) : ℕ :=
  let february_sales := march_sales / 4
  let january_sales := february_sales / 2
  january_sales + february_sales + march_sales

theorem total_sales_proof (march_sales : ℕ) (h : march_sales = 8800) :
  window_screen_sales march_sales = 12100 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_proof_l3937_393786


namespace NUMINAMATH_CALUDE_satellite_units_l3937_393736

/-- Represents a satellite with modular units and sensors. -/
structure Satellite where
  units : ℕ  -- Number of modular units
  non_upgraded_per_unit : ℕ  -- Number of non-upgraded sensors per unit
  total_upgraded : ℕ  -- Total number of upgraded sensors

/-- The conditions given in the problem. -/
def satellite_conditions (s : Satellite) : Prop :=
  -- Condition 2: Non-upgraded sensors per unit is 1/8 of total upgraded
  s.non_upgraded_per_unit = s.total_upgraded / 8 ∧
  -- Condition 3: 25% of all sensors are upgraded
  s.total_upgraded = (s.units * s.non_upgraded_per_unit + s.total_upgraded) / 4

/-- The theorem stating that a satellite satisfying the given conditions has 24 units. -/
theorem satellite_units (s : Satellite) (h : satellite_conditions s) : s.units = 24 := by
  sorry


end NUMINAMATH_CALUDE_satellite_units_l3937_393736


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l3937_393782

theorem rectangle_area_problem (total_area area1 area2 : ℝ) :
  total_area = 48 ∧ area1 = 24 ∧ area2 = 13 →
  total_area - (area1 + area2) = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l3937_393782


namespace NUMINAMATH_CALUDE_largest_number_l3937_393753

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_square_of_prime (n : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ n = p * p

theorem largest_number (P Q R S T : ℕ) : 
  (2 ≤ P ∧ P ≤ 19) →
  (2 ≤ Q ∧ Q ≤ 19) →
  (2 ≤ R ∧ R ≤ 19) →
  (2 ≤ S ∧ S ≤ 19) →
  (2 ≤ T ∧ T ≤ 19) →
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T →
  (P ≥ 10 ∧ P < 100 ∧ is_prime P ∧ is_prime (digit_sum P)) →
  (∃ k : ℕ, Q = 5 * k) →
  (R % 2 = 1 ∧ ¬is_prime R) →
  is_square_of_prime S →
  (is_prime T ∧ T = (P + Q) / 2) →
  Q ≥ P ∧ Q ≥ R ∧ Q ≥ S ∧ Q ≥ T :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l3937_393753


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l3937_393720

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - x - m = 0

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, quadratic_equation x m

-- Theorem statement
theorem quadratic_real_roots_range (m : ℝ) : has_real_roots m → m ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l3937_393720


namespace NUMINAMATH_CALUDE_absolute_value_equality_implies_inequality_l3937_393740

theorem absolute_value_equality_implies_inequality (m : ℝ) : 
  |m - 9| = 9 - m → m ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equality_implies_inequality_l3937_393740


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3937_393798

theorem reciprocal_problem (x : ℚ) : 7 * x = 3 → 70 * (1 / x) = 490 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3937_393798


namespace NUMINAMATH_CALUDE_weekly_production_total_l3937_393783

def john_rate : ℕ := 20
def jane_rate : ℕ := 15
def john_hours : List ℕ := [8, 6, 7, 5, 4]
def jane_hours : List ℕ := [7, 7, 6, 7, 8]

theorem weekly_production_total :
  (john_hours.map (· * john_rate)).sum + (jane_hours.map (· * jane_rate)).sum = 1125 := by
  sorry

end NUMINAMATH_CALUDE_weekly_production_total_l3937_393783


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3937_393763

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3937_393763


namespace NUMINAMATH_CALUDE_product_evaluation_l3937_393721

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3937_393721


namespace NUMINAMATH_CALUDE_smallest_valid_l3937_393758

/-- A positive integer n is valid if 2n is a perfect square and 3n is a perfect cube. -/
def is_valid (n : ℕ+) : Prop :=
  ∃ k m : ℕ+, 2 * n = k^2 ∧ 3 * n = m^3

/-- 72 is the smallest positive integer that is valid. -/
theorem smallest_valid : (∀ n : ℕ+, n < 72 → ¬ is_valid n) ∧ is_valid 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_l3937_393758


namespace NUMINAMATH_CALUDE_johns_age_l3937_393776

/-- John's age in years -/
def john_age : ℕ := sorry

/-- John's dad's age in years -/
def dad_age : ℕ := sorry

/-- John is 18 years younger than his dad -/
axiom age_difference : john_age = dad_age - 18

/-- The sum of John's and his dad's ages is 74 years -/
axiom age_sum : john_age + dad_age = 74

/-- Theorem: John's age is 28 years -/
theorem johns_age : john_age = 28 := by sorry

end NUMINAMATH_CALUDE_johns_age_l3937_393776


namespace NUMINAMATH_CALUDE_kittens_count_l3937_393760

-- Define the number of puppies
def num_puppies : ℕ := 32

-- Define the number of kittens in terms of puppies
def num_kittens : ℕ := 2 * num_puppies + 14

-- Theorem to prove
theorem kittens_count : num_kittens = 78 := by
  sorry

end NUMINAMATH_CALUDE_kittens_count_l3937_393760


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3937_393787

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 650 → boys = 272 → girls = total_students - boys → 
  girls - boys = 106 := by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3937_393787


namespace NUMINAMATH_CALUDE_sum_90_to_99_l3937_393789

/-- The sum of consecutive integers from 90 to 99 is equal to 945. -/
theorem sum_90_to_99 : (Finset.range 10).sum (fun i => i + 90) = 945 := by
  sorry

end NUMINAMATH_CALUDE_sum_90_to_99_l3937_393789


namespace NUMINAMATH_CALUDE_average_girls_per_grade_l3937_393709

/-- Represents a grade with its student composition -/
structure Grade where
  girls : ℕ
  boys : ℕ
  clubGirls : ℕ
  clubBoys : ℕ

/-- The total number of grades -/
def totalGrades : ℕ := 3

/-- List of grades with their student composition -/
def grades : List Grade := [
  { girls := 28, boys := 35, clubGirls := 6, clubBoys := 6 },
  { girls := 45, boys := 42, clubGirls := 7, clubBoys := 8 },
  { girls := 38, boys := 51, clubGirls := 3, clubBoys := 7 }
]

/-- Calculate the total number of girls across all grades -/
def totalGirls : ℕ := (grades.map (·.girls)).sum

/-- Theorem: The average number of girls per grade is 37 -/
theorem average_girls_per_grade :
  totalGirls / totalGrades = 37 := by sorry

end NUMINAMATH_CALUDE_average_girls_per_grade_l3937_393709


namespace NUMINAMATH_CALUDE_three_heads_in_ten_flips_l3937_393793

/-- The probability of flipping exactly k heads in n flips of an unfair coin -/
def unfair_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The main theorem: probability of 3 heads in 10 flips of a coin with 1/3 probability of heads -/
theorem three_heads_in_ten_flips :
  unfair_coin_probability 10 3 (1/3) = 15360 / 59049 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_ten_flips_l3937_393793


namespace NUMINAMATH_CALUDE_missy_patient_count_l3937_393704

/-- Represents the total number of patients Missy is attending to -/
def total_patients : ℕ := 12

/-- Represents the time (in minutes) it takes to serve all patients -/
def total_serving_time : ℕ := 64

/-- Represents the time (in minutes) to serve a standard care patient -/
def standard_serving_time : ℕ := 5

/-- Represents the fraction of patients with special dietary requirements -/
def special_diet_fraction : ℚ := 1 / 3

/-- Represents the increase in serving time for special dietary patients -/
def special_diet_time_increase : ℚ := 1 / 5

theorem missy_patient_count :
  total_patients = 12 ∧
  (special_diet_fraction * total_patients : ℚ) * 
    (standard_serving_time : ℚ) * (1 + special_diet_time_increase) +
  ((1 - special_diet_fraction) * total_patients : ℚ) * 
    (standard_serving_time : ℚ) = total_serving_time := by
  sorry

end NUMINAMATH_CALUDE_missy_patient_count_l3937_393704


namespace NUMINAMATH_CALUDE_ratio_problem_l3937_393751

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a / b = 3)
  (h2 : b / c = 2)
  (h3 : c / d = 5) :
  d / a = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3937_393751


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3937_393734

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 2*(m-3)*x + 16 = (a*x + b)^2) →
  (m = 7 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3937_393734


namespace NUMINAMATH_CALUDE_art_display_side_length_l3937_393706

/-- Represents the dimensions of a glass pane -/
structure GlassPane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the square art display -/
structure ArtDisplay where
  pane : GlassPane
  border_width : ℝ
  horizontal_panes : ℕ
  vertical_panes : ℕ
  is_square : horizontal_panes * pane.width + (horizontal_panes + 1) * border_width = 
              vertical_panes * pane.height + (vertical_panes + 1) * border_width

/-- The side length of the square display is 17.4 inches -/
theorem art_display_side_length (display : ArtDisplay) 
  (h1 : display.horizontal_panes = 4)
  (h2 : display.vertical_panes = 3)
  (h3 : display.border_width = 3) :
  display.horizontal_panes * display.pane.width + (display.horizontal_panes + 1) * display.border_width = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_art_display_side_length_l3937_393706


namespace NUMINAMATH_CALUDE_cart_max_speed_l3937_393731

/-- The maximum speed of a cart on a circular track -/
theorem cart_max_speed (R a : ℝ) (hR : R > 0) (ha : a > 0) :
  ∃ v_max : ℝ,
    v_max = (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2))^(1/4) ∧
    ∀ v : ℝ,
      v ≤ v_max →
      (v^2 / (4 * Real.pi * R))^2 + (v^2 / R)^2 ≤ a^2 :=
by sorry

end NUMINAMATH_CALUDE_cart_max_speed_l3937_393731


namespace NUMINAMATH_CALUDE_correct_prices_l3937_393705

/-- Represents the purchase and sale of golden passion fruit -/
structure GoldenPassionFruit where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  weight_ratio : ℝ
  price_difference : ℝ
  profit_margin : ℝ

/-- Calculates the unit prices and minimum selling price for golden passion fruit -/
def calculate_prices (gpf : GoldenPassionFruit) : 
  (ℝ × ℝ × ℝ) :=
  let first_batch_price := 20
  let second_batch_price := first_batch_price - gpf.price_difference
  let min_selling_price := 25
  (first_batch_price, second_batch_price, min_selling_price)

/-- Theorem stating the correctness of the calculated prices -/
theorem correct_prices (gpf : GoldenPassionFruit) 
  (h1 : gpf.first_batch_cost = 3600)
  (h2 : gpf.second_batch_cost = 5400)
  (h3 : gpf.weight_ratio = 2)
  (h4 : gpf.price_difference = 5)
  (h5 : gpf.profit_margin = 0.5) :
  let (first_price, second_price, min_price) := calculate_prices gpf
  first_price = 20 ∧ 
  second_price = 15 ∧ 
  min_price ≥ 25 ∧
  min_price * (gpf.first_batch_cost / first_price + gpf.second_batch_cost / second_price) ≥ 
    (gpf.first_batch_cost + gpf.second_batch_cost) * (1 + gpf.profit_margin) :=
by sorry


end NUMINAMATH_CALUDE_correct_prices_l3937_393705


namespace NUMINAMATH_CALUDE_triplet_sum_not_two_l3937_393701

theorem triplet_sum_not_two : ∃! (x y z : ℝ), 
  ((x = 2.2 ∧ y = -3.2 ∧ z = 2.0) ∨
   (x = 3/4 ∧ y = 1/2 ∧ z = 3/4) ∨
   (x = 4 ∧ y = -6 ∧ z = 4) ∨
   (x = 0.4 ∧ y = 0.5 ∧ z = 1.1) ∨
   (x = 2/3 ∧ y = 1/3 ∧ z = 1)) ∧
  x + y + z ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_triplet_sum_not_two_l3937_393701


namespace NUMINAMATH_CALUDE_job_completion_time_l3937_393727

theorem job_completion_time (efficiency_ratio : ℝ) (joint_completion_time : ℝ) :
  efficiency_ratio = (1 : ℝ) / 2 →
  joint_completion_time = 15 →
  ∃ (solo_completion_time : ℝ),
    solo_completion_time = (3 / 2) * joint_completion_time ∧
    solo_completion_time = 45 / 2 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l3937_393727


namespace NUMINAMATH_CALUDE_age_ratio_this_year_l3937_393713

def yoongi_age_last_year : ℕ := 6
def grandfather_age_last_year : ℕ := 62

def yoongi_age_this_year : ℕ := yoongi_age_last_year + 1
def grandfather_age_this_year : ℕ := grandfather_age_last_year + 1

theorem age_ratio_this_year :
  grandfather_age_this_year / yoongi_age_this_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_this_year_l3937_393713


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_ellipse_equation_l3937_393794

/-- Definition of the ellipse based on the sum of distances from two foci -/
def is_on_ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 0)^2 + (y + Real.sqrt 3)^2) +
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) = 4

/-- Definition of the line y = kx + √3 -/
def is_on_line (k x y : ℝ) : Prop := y = k * x + Real.sqrt 3

/-- Definition of a point being on the circle with diameter AB passing through origin -/
def is_on_circle (xA yA xB yB x y : ℝ) : Prop :=
  x * (xA + xB) + y * (yA + yB) = xA * xB + yA * yB

theorem ellipse_and_line_intersection :
  ∃ (k : ℝ),
    (∃ (xA yA xB yB : ℝ),
      is_on_ellipse xA yA ∧ is_on_ellipse xB yB ∧
      is_on_line k xA yA ∧ is_on_line k xB yB ∧
      is_on_circle xA yA xB yB 0 0) ∧
    k = Real.sqrt 11 / 2 ∨ k = -Real.sqrt 11 / 2 := by sorry

theorem ellipse_equation :
  ∀ (x y : ℝ), is_on_ellipse x y ↔ x^2 + y^2 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_ellipse_equation_l3937_393794


namespace NUMINAMATH_CALUDE_binary_multiplication_example_l3937_393784

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec toBinary (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
    toBinary n

theorem binary_multiplication_example :
  let a := [false, true, true, false, true, true]  -- 110110₂
  let b := [true, true, true]  -- 111₂
  let result := [false, true, false, false, true, false, false, true]  -- 10010010₂
  binaryToNat a * binaryToNat b = binaryToNat result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_example_l3937_393784


namespace NUMINAMATH_CALUDE_distance_between_trees_l3937_393722

/-- Given a yard of length 150 meters with 11 trees planted at equal distances,
    including one tree at each end, the distance between consecutive trees is 15 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 150 ∧ num_trees = 11 →
  (yard_length / (num_trees - 1 : ℝ)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3937_393722


namespace NUMINAMATH_CALUDE_magical_stack_with_151_fixed_l3937_393724

/-- A stack of cards is magical if at least one card from each pile retains its original position after restacking -/
def is_magical (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≤ n ∧ b > n ∧ b ≤ 2*n ∧
  (a = 2*a - 1 ∨ b = 2*(b - n))

theorem magical_stack_with_151_fixed (n : ℕ) 
  (h_magical : is_magical n) 
  (h_151_fixed : 151 ≤ n ∧ 151 = 2*151 - 1) : 
  n = 226 ∧ 2*n = 452 := by sorry

end NUMINAMATH_CALUDE_magical_stack_with_151_fixed_l3937_393724


namespace NUMINAMATH_CALUDE_batsman_second_set_matches_l3937_393772

/-- Given information about a batsman's performance, prove the number of matches in the second set -/
theorem batsman_second_set_matches 
  (first_set_matches : ℕ) 
  (total_matches : ℕ) 
  (first_set_average : ℝ) 
  (second_set_average : ℝ) 
  (total_average : ℝ) 
  (h1 : first_set_matches = 35)
  (h2 : total_matches = 49)
  (h3 : first_set_average = 36)
  (h4 : second_set_average = 15)
  (h5 : total_average = 30) :
  total_matches - first_set_matches = 14 := by
  sorry

#check batsman_second_set_matches

end NUMINAMATH_CALUDE_batsman_second_set_matches_l3937_393772


namespace NUMINAMATH_CALUDE_units_digit_of_seven_pow_five_cubed_l3937_393746

theorem units_digit_of_seven_pow_five_cubed : 7^(5^3) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_pow_five_cubed_l3937_393746


namespace NUMINAMATH_CALUDE_overall_profit_l3937_393750

def refrigerator_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def refrigerator_loss_percent : ℝ := 0.03
def mobile_profit_percent : ℝ := 0.10

def refrigerator_selling_price : ℝ := refrigerator_cost * (1 - refrigerator_loss_percent)
def mobile_selling_price : ℝ := mobile_cost * (1 + mobile_profit_percent)

def total_cost : ℝ := refrigerator_cost + mobile_cost
def total_selling_price : ℝ := refrigerator_selling_price + mobile_selling_price

theorem overall_profit : total_selling_price - total_cost = 350 := by
  sorry

end NUMINAMATH_CALUDE_overall_profit_l3937_393750


namespace NUMINAMATH_CALUDE_max_value_of_function_l3937_393749

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  x * (1 - 2*x) ≤ 1/8 ∧ ∃ x₀, 0 < x₀ ∧ x₀ < 1/2 ∧ x₀ * (1 - 2*x₀) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3937_393749


namespace NUMINAMATH_CALUDE_hundredth_count_is_twelve_l3937_393718

/-- Represents the circular arrangement of stones with the given counting pattern. -/
def StoneCircle :=
  { n : ℕ // n > 0 ∧ n ≤ 12 }

/-- The label assigned to a stone after a certain number of counts. -/
def label (count : ℕ) : StoneCircle → ℕ :=
  sorry

/-- The original stone number corresponding to a given label. -/
def originalStone (label : ℕ) : StoneCircle :=
  sorry

/-- Theorem stating that the 100th count corresponds to the original stone number 12. -/
theorem hundredth_count_is_twelve :
  originalStone 100 = ⟨12, by norm_num⟩ :=
sorry

end NUMINAMATH_CALUDE_hundredth_count_is_twelve_l3937_393718


namespace NUMINAMATH_CALUDE_vector_problem_l3937_393781

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

theorem vector_problem :
  (∃ k : ℝ, k * a.1 + 2 * b.1 = 14 * (k * a.2 + 2 * b.2) / (-4) ∧ k = -1) ∧
  (∃ c : ℝ × ℝ, (c.1^2 + c.2^2 = 1) ∧
    ((c.1 + 3)^2 + (c.2 - 2)^2 = 20) ∧
    ((c = (5/13, -12/13)) ∨ (c = (1, 0)))) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3937_393781


namespace NUMINAMATH_CALUDE_common_difference_from_sum_condition_l3937_393764

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_from_sum_condition (seq : ArithmeticSequence) 
    (h : seq.S 4 / 12 - seq.S 3 / 9 = 1) : 
    common_difference seq = 6 := by
  sorry


end NUMINAMATH_CALUDE_common_difference_from_sum_condition_l3937_393764


namespace NUMINAMATH_CALUDE_two_mixers_one_tv_cost_is_7000_l3937_393700

/-- The cost of a mixer in Rupees -/
def mixer_cost : ℕ := sorry

/-- The cost of a TV in Rupees -/
def tv_cost : ℕ := 4200

/-- The total cost of two mixers and one TV in Rupees -/
def two_mixers_one_tv_cost : ℕ := 2 * mixer_cost + tv_cost

theorem two_mixers_one_tv_cost_is_7000 :
  (two_mixers_one_tv_cost = 7000) ∧ (2 * tv_cost + mixer_cost = 9800) :=
sorry

end NUMINAMATH_CALUDE_two_mixers_one_tv_cost_is_7000_l3937_393700


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3937_393780

def C : Set Nat := {66, 68, 71, 73, 75}

theorem smallest_prime_factor_in_C : 
  ∃ (n : Nat), n ∈ C ∧ (∀ m ∈ C, ∀ p q : Nat, Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q) ∧ n = 66 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3937_393780


namespace NUMINAMATH_CALUDE_second_derivative_of_y_l3937_393771

noncomputable def y (x : ℝ) : ℝ := (1 - x^2) / Real.sin x

theorem second_derivative_of_y (x : ℝ) (h : Real.sin x ≠ 0) :
  (deriv^[2] y) x = (-2*x*Real.sin x - (1 - x^2)*Real.cos x) / (Real.sin x)^2 :=
sorry

end NUMINAMATH_CALUDE_second_derivative_of_y_l3937_393771


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3937_393754

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 3) → x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3937_393754


namespace NUMINAMATH_CALUDE_train_crossing_time_l3937_393739

/-- Given a train and two platforms, calculate the time to cross the first platform -/
theorem train_crossing_time (Lt Lp1 Lp2 Tp2 : ℝ) (h1 : Lt = 30)
    (h2 : Lp1 = 180) (h3 : Lp2 = 250) (h4 : Tp2 = 20) :
  (Lt + Lp1) / ((Lt + Lp2) / Tp2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3937_393739


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l3937_393735

theorem quadratic_root_proof : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ 
  (a * (2 - Real.sqrt 7)^2 + b * (2 - Real.sqrt 7) + c = 0) ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x^2 - 4*x - 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l3937_393735


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3937_393791

/-- An ellipse with parametric equations x = 3cos(φ) and y = 5sin(φ) -/
structure Ellipse where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ φ, x φ = 3 * Real.cos φ
  h_y : ∀ φ, y φ = 5 * Real.sin φ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating that the eccentricity of the given ellipse is 4/5 -/
theorem ellipse_eccentricity (e : Ellipse) : eccentricity e = 4/5 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3937_393791


namespace NUMINAMATH_CALUDE_value_of_A_l3937_393726

def round_down_tens (n : ℕ) : ℕ := n / 10 * 10

theorem value_of_A (A : ℕ) : 
  A < 10 → 
  round_down_tens (900 + 10 * A + 7) = 930 → 
  A = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_A_l3937_393726


namespace NUMINAMATH_CALUDE_inequality_proof_l3937_393738

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9)
  (h5 : a^2 + b^2 + c^2 + d^2 = 21) :
  a * b - c * d ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3937_393738


namespace NUMINAMATH_CALUDE_cone_no_rectangular_cross_section_cone_unique_no_rectangular_cross_section_l3937_393777

-- Define the types of geometric shapes we're considering
inductive GeometricShape
| Cone
| Cylinder
| TriangularPrism
| RectangularPrism

-- Define a function that determines if a shape can have a rectangular cross-section
def canHaveRectangularCrossSection (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Cone => False
  | _ => True

-- Theorem statement
theorem cone_no_rectangular_cross_section :
  ∀ (shape : GeometricShape),
    ¬(canHaveRectangularCrossSection shape) ↔ shape = GeometricShape.Cone :=
by
  sorry

-- Alternative formulation focusing on the unique property of the cone
theorem cone_unique_no_rectangular_cross_section :
  ∃! (shape : GeometricShape), ¬(canHaveRectangularCrossSection shape) :=
by
  sorry

end NUMINAMATH_CALUDE_cone_no_rectangular_cross_section_cone_unique_no_rectangular_cross_section_l3937_393777


namespace NUMINAMATH_CALUDE_paper_clips_in_2_cases_l3937_393748

/-- The number of paper clips in 2 cases -/
def paperClipsIn2Cases (c b : ℕ) : ℕ := 2 * c * b * 400

/-- Theorem: The number of paper clips in 2 cases is 2 * c * b * 400 -/
theorem paper_clips_in_2_cases (c b : ℕ) : paperClipsIn2Cases c b = 2 * c * b * 400 := by
  sorry

end NUMINAMATH_CALUDE_paper_clips_in_2_cases_l3937_393748


namespace NUMINAMATH_CALUDE_solution_value_l3937_393723

theorem solution_value (a : ℝ) : (1 + 1) * a = 2 * (2 * 1 - a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3937_393723


namespace NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l3937_393737

theorem product_of_numbers_with_sum_and_difference 
  (x y : ℝ) (sum_eq : x + y = 60) (diff_eq : x - y = 10) : x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l3937_393737


namespace NUMINAMATH_CALUDE_triangle_classification_l3937_393741

theorem triangle_classification (a b : ℝ) (A B : ℝ) (h_positive : 0 < A ∧ A < π) 
  (h_eq : a * Real.cos A = b * Real.cos B) :
  A = B ∨ A + B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_classification_l3937_393741


namespace NUMINAMATH_CALUDE_unique_B_for_divisible_by_7_l3937_393703

def is_divisible_by_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def four_digit_number (B : ℕ) : ℕ := 4000 + 100 * B + 10 * B + 3

theorem unique_B_for_divisible_by_7 :
  ∀ B : ℕ, B < 10 →
    is_divisible_by_7 (four_digit_number B) →
    B = 0 := by sorry

end NUMINAMATH_CALUDE_unique_B_for_divisible_by_7_l3937_393703


namespace NUMINAMATH_CALUDE_water_purifier_theorem_l3937_393779

/-- Represents a water purifier type -/
inductive PurifierType
| A
| B

/-- Represents the costs and prices of water purifiers -/
structure PurifierInfo where
  cost_A : ℝ
  cost_B : ℝ
  price_A : ℝ
  price_B : ℝ
  filter_cost_A : ℝ
  filter_cost_B : ℝ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  num_A : ℕ
  num_B : ℕ

/-- The main theorem about water purifier costs and purchasing plans -/
theorem water_purifier_theorem 
  (info : PurifierInfo)
  (h1 : info.cost_B = info.cost_A + 600)
  (h2 : 36000 / info.cost_A = 2 * (27000 / info.cost_B))
  (h3 : info.price_A = 1350)
  (h4 : info.price_B = 2100)
  (h5 : info.filter_cost_A = 400)
  (h6 : info.filter_cost_B = 500) :
  info.cost_A = 1200 ∧ 
  info.cost_B = 1800 ∧
  (∃ (plans : List PurchasePlan), 
    (∀ p ∈ plans, 
      p.num_A * info.cost_A + p.num_B * info.cost_B ≤ 60000 ∧ 
      p.num_B ≤ 8) ∧
    plans.length = 4) ∧
  (∃ (num_filters_A num_filters_B : ℕ),
    num_filters_A + num_filters_B = 6 ∧
    ∃ (p : PurchasePlan), 
      p.num_A * (info.price_A - info.cost_A) + 
      p.num_B * (info.price_B - info.cost_B) - 
      (num_filters_A * info.filter_cost_A + num_filters_B * info.filter_cost_B) = 5250) :=
by sorry

end NUMINAMATH_CALUDE_water_purifier_theorem_l3937_393779


namespace NUMINAMATH_CALUDE_smallest_addend_for_divisibility_l3937_393766

theorem smallest_addend_for_divisibility (a b : ℕ) (ha : a = 87908235) (hb : b = 12587) :
  let x := (b - (a % b)) % b
  (a + x) % b = 0 ∧ ∀ y : ℕ, y < x → (a + y) % b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_addend_for_divisibility_l3937_393766


namespace NUMINAMATH_CALUDE_correct_stratified_sample_teaching_l3937_393756

/-- Represents the composition of staff in a school -/
structure SchoolStaff where
  total : ℕ
  administrative : ℕ
  teaching : ℕ
  support : ℕ

/-- Calculates the number of teaching staff to be included in a stratified sample -/
def stratifiedSampleTeaching (staff : SchoolStaff) (sampleSize : ℕ) : ℕ :=
  (staff.teaching * sampleSize) / staff.total

/-- Theorem stating the correct number of teaching staff in the stratified sample -/
theorem correct_stratified_sample_teaching (staff : SchoolStaff) (sampleSize : ℕ) :
  staff.total = 200 ∧ 
  staff.administrative = 24 ∧ 
  staff.teaching = 10 * staff.support ∧
  staff.teaching + staff.support + staff.administrative = staff.total ∧
  sampleSize = 50 →
  stratifiedSampleTeaching staff sampleSize = 40 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_teaching_l3937_393756


namespace NUMINAMATH_CALUDE_quadratic_roots_angles_l3937_393733

theorem quadratic_roots_angles (Az m n φ ψ : ℝ) (hAz : Az ≠ 0) :
  (∀ x, Az * x^2 - m * x + n = 0 ↔ x = Real.tan φ ∨ x = Real.tan ψ) →
  Real.tan (φ + ψ) = m / (1 - n) ∧ Real.tan (φ - ψ) = Real.sqrt (m^2 - 4*n) / (1 + n) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_angles_l3937_393733


namespace NUMINAMATH_CALUDE_infinitely_many_planes_through_point_parallel_to_line_l3937_393792

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Predicate to check if a point is outside a line -/
def isOutside (P : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Predicate to check if a plane passes through a point -/
def passesThroughPoint (plane : Plane3D) (P : Point3D) : Prop :=
  sorry

/-- Predicate to check if a plane is parallel to a line -/
def isParallelToLine (plane : Plane3D) (l : Line3D) : Prop :=
  sorry

/-- The main theorem -/
theorem infinitely_many_planes_through_point_parallel_to_line
  (P : Point3D) (l : Line3D) (h : isOutside P l) :
  ∃ (f : ℕ → Plane3D), Function.Injective f ∧
    (∀ n : ℕ, passesThroughPoint (f n) P ∧ isParallelToLine (f n) l) :=
  sorry

end NUMINAMATH_CALUDE_infinitely_many_planes_through_point_parallel_to_line_l3937_393792


namespace NUMINAMATH_CALUDE_ages_sum_l3937_393775

theorem ages_sum (a b c : ℕ) : 
  a = b + c + 18 ∧ 
  a^2 = (b + c)^2 + 2016 → 
  a + b + c = 112 :=
by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l3937_393775


namespace NUMINAMATH_CALUDE_expression_evaluation_l3937_393769

theorem expression_evaluation (m n : ℤ) (hm : m = 1) (hn : n = -2) :
  ((3*m + n) * (m - n) - (2*m - n)^2 + (m - 2*n) * (m + 2*n)) / (n / 2) = 28 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3937_393769


namespace NUMINAMATH_CALUDE_solution_when_a_is_3_solution_when_a_is_neg_l3937_393752

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (1 - 2*a) * x - 2 < 0

-- Define the solution set for a = 3
def solution_set_a3 : Set ℝ :=
  {x | -1/3 < x ∧ x < 2}

-- Define the solution set for a < 0
def solution_set_a_neg (a : ℝ) : Set ℝ :=
  if -1/2 < a ∧ a < 0 then
    {x | x < 2 ∨ x > -1/a}
  else if a = -1/2 then
    {x | x ≠ 2}
  else
    {x | x > 2 ∨ x < -1/a}

-- Theorem for a = 3
theorem solution_when_a_is_3 :
  ∀ x, x ∈ solution_set_a3 ↔ quadratic_inequality 3 x :=
sorry

-- Theorem for a < 0
theorem solution_when_a_is_neg :
  ∀ a, a < 0 → ∀ x, x ∈ solution_set_a_neg a ↔ quadratic_inequality a x :=
sorry

end NUMINAMATH_CALUDE_solution_when_a_is_3_solution_when_a_is_neg_l3937_393752


namespace NUMINAMATH_CALUDE_Z_in_second_quadrant_l3937_393708

-- Define the complex number Z
def Z : ℂ := Complex.I * (1 + Complex.I)

-- Theorem statement
theorem Z_in_second_quadrant : 
  Real.sign (Z.re) = -1 ∧ Real.sign (Z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_Z_in_second_quadrant_l3937_393708


namespace NUMINAMATH_CALUDE_mikes_investment_interest_l3937_393768

/-- Calculates the total interest earned from a two-part investment --/
def total_interest (total_investment : ℚ) (amount_at_lower_rate : ℚ) (lower_rate : ℚ) (higher_rate : ℚ) : ℚ :=
  let amount_at_higher_rate := total_investment - amount_at_lower_rate
  let interest_lower := amount_at_lower_rate * lower_rate
  let interest_higher := amount_at_higher_rate * higher_rate
  interest_lower + interest_higher

/-- Theorem stating that Mike's investment yields $624 in interest --/
theorem mikes_investment_interest :
  total_interest 6000 1800 (9/100) (11/100) = 624 := by
  sorry

end NUMINAMATH_CALUDE_mikes_investment_interest_l3937_393768


namespace NUMINAMATH_CALUDE_quadratic_b_value_l3937_393716

/-- A quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x-coordinate on a quadratic function -/
def QuadraticFunction.y (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_b_value (f : QuadraticFunction) (y₁ y₂ : ℝ) 
  (h₁ : f.y 2 = y₁)
  (h₂ : f.y (-2) = y₂)
  (h₃ : y₁ - y₂ = -12) :
  f.b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_b_value_l3937_393716


namespace NUMINAMATH_CALUDE_anita_blueberry_cartons_l3937_393795

/-- Represents the number of cartons of berries in Anita's berry cobbler problem -/
structure BerryCobbler where
  total : ℕ
  strawberries : ℕ
  to_buy : ℕ

/-- Calculates the number of blueberry cartons Anita has -/
def blueberry_cartons (bc : BerryCobbler) : ℕ :=
  bc.total - bc.strawberries - bc.to_buy

/-- Theorem stating that Anita has 9 cartons of blueberries -/
theorem anita_blueberry_cartons :
  ∀ (bc : BerryCobbler),
    bc.total = 26 → bc.strawberries = 10 → bc.to_buy = 7 →
    blueberry_cartons bc = 9 := by
  sorry

end NUMINAMATH_CALUDE_anita_blueberry_cartons_l3937_393795


namespace NUMINAMATH_CALUDE_max_k_value_l3937_393729

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + 2 * k * (x / y + y / x)) :
  k ≤ Real.sqrt (5 / 6) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3937_393729


namespace NUMINAMATH_CALUDE_ark5_ensures_metabolic_energy_needs_l3937_393774

-- Define the Ark5 enzyme
def Ark5 : Type := Unit

-- Define cancer cells
def CancerCell : Type := Unit

-- Define the function that represents the ability to balance energy
def balanceEnergy (a : Ark5) (c : CancerCell) : Prop := sorry

-- Define the function that represents the ability to proliferate without limit
def proliferateWithoutLimit (c : CancerCell) : Prop := sorry

-- Define the function that represents the state of energy scarcity
def energyScarcity : Prop := sorry

-- Define the function that represents cell death due to lack of energy
def dieFromLackOfEnergy (c : CancerCell) : Prop := sorry

-- Define the function that represents ensuring metabolic energy needs
def ensureMetabolicEnergyNeeds (a : Ark5) (c : CancerCell) : Prop := sorry

-- Theorem statement
theorem ark5_ensures_metabolic_energy_needs :
  ∀ (a : Ark5) (c : CancerCell),
    (¬balanceEnergy a c → (energyScarcity → proliferateWithoutLimit c)) ∧
    (¬balanceEnergy a c → (energyScarcity → dieFromLackOfEnergy c)) →
    ensureMetabolicEnergyNeeds a c :=
by
  sorry

end NUMINAMATH_CALUDE_ark5_ensures_metabolic_energy_needs_l3937_393774


namespace NUMINAMATH_CALUDE_exchange_ways_eq_six_l3937_393796

/-- The number of ways to exchange 100 yuan into 20 yuan and 10 yuan bills -/
def exchange_ways : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 20 * p.1 + 10 * p.2 = 100) (Finset.product (Finset.range 6) (Finset.range 11))).card

/-- Theorem stating that there are exactly 6 ways to exchange 100 yuan into 20 yuan and 10 yuan bills -/
theorem exchange_ways_eq_six : exchange_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_exchange_ways_eq_six_l3937_393796


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_3a2_eq_b2_plus_1_l3937_393732

theorem no_integer_solutions_for_3a2_eq_b2_plus_1 :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_3a2_eq_b2_plus_1_l3937_393732


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3937_393702

theorem greatest_divisor_with_remainders : ∃ n : ℕ, 
  (∀ m : ℕ, (6215 % m = 23 ∧ 7373 % m = 29) → m ≤ n) ∧
  6215 % n = 23 ∧ 7373 % n = 29 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3937_393702


namespace NUMINAMATH_CALUDE_infinitely_many_divisors_l3937_393711

theorem infinitely_many_divisors (a : ℕ) :
  Set.Infinite {n : ℕ | n ∣ a^(n - a + 1) - 1} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisors_l3937_393711


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l3937_393714

/-- A quadratic function f(x) = -2x^2 - 8x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := -2 * x^2 - 8 * x + m

theorem quadratic_points_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f m (-1) = y₁)
  (h₂ : f m (-2) = y₂)
  (h₃ : f m (-4) = y₃) :
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l3937_393714


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3937_393744

theorem trigonometric_equality (a b c : ℝ) (α β : ℝ) 
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = 0)
  (h3 : ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  Real.sin (α - β) ^ 2 = c^2 / (a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3937_393744


namespace NUMINAMATH_CALUDE_chocolate_difference_l3937_393770

theorem chocolate_difference (robert nickel jessica : ℕ) 
  (h1 : robert = 23) 
  (h2 : nickel = 8) 
  (h3 : jessica = 15) : 
  (robert - nickel = 15) ∧ (jessica - nickel = 7) := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_l3937_393770


namespace NUMINAMATH_CALUDE_log_condition_equivalence_l3937_393762

theorem log_condition_equivalence (m n : ℝ) (hm : m > 0 ∧ m ≠ 1) (hn : n > 0) :
  Real.log n / Real.log m < 0 ↔ (m - 1) * (n - 1) < 0 := by
  sorry

end NUMINAMATH_CALUDE_log_condition_equivalence_l3937_393762


namespace NUMINAMATH_CALUDE_polygon_sides_l3937_393745

theorem polygon_sides (sum_interior_angles : ℕ) : sum_interior_angles = 1440 → ∃ n : ℕ, n = 10 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3937_393745


namespace NUMINAMATH_CALUDE_shuffleboard_games_total_l3937_393710

/-- Proves that the total number of games played is 32 given the conditions of the shuffleboard game. -/
theorem shuffleboard_games_total (jerry_wins dave_wins ken_wins : ℕ) 
  (h1 : ken_wins = dave_wins + 5)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : jerry_wins = 7) : 
  jerry_wins + dave_wins + ken_wins = 32 := by
  sorry

end NUMINAMATH_CALUDE_shuffleboard_games_total_l3937_393710


namespace NUMINAMATH_CALUDE_lottery_winner_prize_l3937_393707

def lottery_prize (num_tickets : ℕ) (first_ticket_price : ℕ) (price_increase : ℕ) (profit : ℕ) : ℕ :=
  let total_revenue := (num_tickets * (2 * first_ticket_price + (num_tickets - 1) * price_increase)) / 2
  total_revenue - profit

theorem lottery_winner_prize :
  lottery_prize 5 1 1 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_lottery_winner_prize_l3937_393707
