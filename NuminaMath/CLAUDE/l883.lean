import Mathlib

namespace NUMINAMATH_CALUDE_triangle_equals_four_l883_88340

/-- Given that △ is a digit and △7₁₂ = △3₁₃, prove that △ = 4 -/
theorem triangle_equals_four (triangle : ℕ) 
  (h1 : triangle < 10) 
  (h2 : triangle * 12 + 7 = triangle * 13 + 3) : 
  triangle = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_equals_four_l883_88340


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_24hour_l883_88322

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The largest possible sum of digits in a 24-hour format digital watch display is 24 -/
theorem largest_sum_of_digits_24hour : 
  ∀ t : Time24, sumOfDigitsTime24 t ≤ 24 ∧ 
  ∃ t' : Time24, sumOfDigitsTime24 t' = 24 := by
  sorry

#check largest_sum_of_digits_24hour

end NUMINAMATH_CALUDE_largest_sum_of_digits_24hour_l883_88322


namespace NUMINAMATH_CALUDE_smallest_winning_k_l883_88361

/-- Represents a square on the game board --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the game state --/
structure GameState where
  board : Square → Option Char
  mike_moves : Nat
  harry_moves : Nat

/-- Checks if a sequence forms a winning pattern --/
def is_winning_sequence (s : List Char) : Bool :=
  s = ['H', 'M', 'M'] || s = ['M', 'M', 'H']

/-- Checks if there's a winning sequence on the board --/
def has_winning_sequence (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for Mike --/
def MikeStrategy := Nat → List Square

/-- Represents a strategy for Harry --/
def HarryStrategy := GameState → List Square

/-- Simulates a game with given strategies --/
def play_game (k : Nat) (mike_strat : MikeStrategy) (harry_strat : HarryStrategy) : Bool :=
  sorry

/-- Defines what it means for Mike to have a winning strategy --/
def mike_has_winning_strategy (k : Nat) : Prop :=
  ∃ (mike_strat : MikeStrategy), ∀ (harry_strat : HarryStrategy), 
    play_game k mike_strat harry_strat = true

/-- The main theorem stating that 16 is the smallest k for which Mike has a winning strategy --/
theorem smallest_winning_k : 
  (mike_has_winning_strategy 16) ∧ 
  (∀ k < 16, ¬(mike_has_winning_strategy k)) :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_k_l883_88361


namespace NUMINAMATH_CALUDE_even_function_inequality_l883_88339

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonic on (-∞, 0] if it's either
    nondecreasing or nonincreasing on that interval -/
def IsMonotonicOnNegative (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y) ∨ (∀ x y, x ≤ y ∧ y ≤ 0 → f y ≤ f x)

theorem even_function_inequality (f : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_monotonic : IsMonotonicOnNegative f)
  (h_inequality : f (-2) < f 1) :
  f 5 < f (-3) ∧ f (-3) < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l883_88339


namespace NUMINAMATH_CALUDE_rationalized_denominator_product_l883_88351

theorem rationalized_denominator_product (A B C : ℤ) : 
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C → A * B * C = 180 := by
  sorry

end NUMINAMATH_CALUDE_rationalized_denominator_product_l883_88351


namespace NUMINAMATH_CALUDE_perfect_fruits_count_perfect_fruits_theorem_l883_88307

/-- Represents the fruit types in the batch -/
inductive FruitType
| Apple
| Orange
| Mango

/-- Represents the size of a fruit -/
inductive Size
| Small
| Medium
| Large

/-- Represents the ripeness stage of a fruit -/
inductive Ripeness
| Unripe
| PartlyRipe
| FullyRipe

/-- Defines the characteristics of the fruit batch -/
structure FruitBatch where
  totalFruits : ℕ
  apples : ℕ
  oranges : ℕ
  mangoes : ℕ
  appleSizeDistribution : Size → ℚ
  appleRipenessDistribution : Ripeness → ℚ
  orangeSizeDistribution : Size → ℚ
  orangeRipenessDistribution : Ripeness → ℚ
  mangoSizeDistribution : Size → ℚ
  mangoRipenessDistribution : Ripeness → ℚ

/-- Defines what makes a fruit perfect based on its type -/
def isPerfect (t : FruitType) (s : Size) (r : Ripeness) : Prop :=
  match t with
  | FruitType.Apple => (s = Size.Medium ∨ s = Size.Large) ∧ r = Ripeness.FullyRipe
  | FruitType.Orange => s = Size.Large ∧ r = Ripeness.FullyRipe
  | FruitType.Mango => (s = Size.Medium ∨ s = Size.Large) ∧ (r = Ripeness.PartlyRipe ∨ r = Ripeness.FullyRipe)

/-- The main theorem to prove -/
theorem perfect_fruits_count (batch : FruitBatch) : ℕ :=
  sorry

/-- The theorem statement -/
theorem perfect_fruits_theorem (batch : FruitBatch) :
  batch.totalFruits = 120 ∧
  batch.apples = 60 ∧
  batch.oranges = 40 ∧
  batch.mangoes = 20 ∧
  batch.appleSizeDistribution Size.Small = 1/4 ∧
  batch.appleSizeDistribution Size.Medium = 1/2 ∧
  batch.appleSizeDistribution Size.Large = 1/4 ∧
  batch.appleRipenessDistribution Ripeness.Unripe = 1/3 ∧
  batch.appleRipenessDistribution Ripeness.PartlyRipe = 1/6 ∧
  batch.appleRipenessDistribution Ripeness.FullyRipe = 1/2 ∧
  batch.orangeSizeDistribution Size.Small = 1/3 ∧
  batch.orangeSizeDistribution Size.Medium = 1/3 ∧
  batch.orangeSizeDistribution Size.Large = 1/3 ∧
  batch.orangeRipenessDistribution Ripeness.Unripe = 1/2 ∧
  batch.orangeRipenessDistribution Ripeness.PartlyRipe = 1/4 ∧
  batch.orangeRipenessDistribution Ripeness.FullyRipe = 1/4 ∧
  batch.mangoSizeDistribution Size.Small = 1/5 ∧
  batch.mangoSizeDistribution Size.Medium = 2/5 ∧
  batch.mangoSizeDistribution Size.Large = 2/5 ∧
  batch.mangoRipenessDistribution Ripeness.Unripe = 1/4 ∧
  batch.mangoRipenessDistribution Ripeness.PartlyRipe = 1/2 ∧
  batch.mangoRipenessDistribution Ripeness.FullyRipe = 1/4 →
  perfect_fruits_count batch = 55 := by
  sorry


end NUMINAMATH_CALUDE_perfect_fruits_count_perfect_fruits_theorem_l883_88307


namespace NUMINAMATH_CALUDE_tan_105_degrees_l883_88311

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l883_88311


namespace NUMINAMATH_CALUDE_percentage_problem_l883_88376

theorem percentage_problem (P : ℝ) : P = (354.2 * 6 * 100) / 1265 ↔ (P / 100) * 1265 / 6 = 354.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l883_88376


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l883_88333

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l883_88333


namespace NUMINAMATH_CALUDE_irrational_approximation_l883_88345

theorem irrational_approximation (x : ℝ) (h_pos : x > 0) (h_irr : Irrational x) :
  ∀ N : ℕ, ∃ p q : ℤ, q > N ∧ q > 0 ∧ |x - (p : ℝ) / q| < 1 / q^2 := by
  sorry

end NUMINAMATH_CALUDE_irrational_approximation_l883_88345


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l883_88308

theorem quadratic_roots_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ a * c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l883_88308


namespace NUMINAMATH_CALUDE_workshop_average_salary_l883_88332

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℚ)
  (non_technician_salary : ℚ)
  (h1 : total_workers = 22)
  (h2 : technicians = 7)
  (h3 : technician_salary = 1000)
  (h4 : non_technician_salary = 780) :
  let non_technicians := total_workers - technicians
  let total_salary := technicians * technician_salary + non_technicians * non_technician_salary
  total_salary / total_workers = 850 := by
sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l883_88332


namespace NUMINAMATH_CALUDE_trigonometric_identity_l883_88393

theorem trigonometric_identity : 
  Real.cos (12 * π / 180) * Real.sin (42 * π / 180) - 
  Real.sin (12 * π / 180) * Real.cos (42 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l883_88393


namespace NUMINAMATH_CALUDE_monday_grading_percentage_l883_88329

/-- The percentage of exams graded on Monday -/
def monday_percentage : ℝ := 40

/-- The total number of exams -/
def total_exams : ℕ := 120

/-- The percentage of remaining exams graded on Tuesday -/
def tuesday_percentage : ℝ := 75

/-- The number of exams left to grade after Tuesday -/
def exams_left : ℕ := 12

theorem monday_grading_percentage :
  monday_percentage = 40 ∧
  (total_exams : ℝ) - (monday_percentage / 100) * total_exams -
    (tuesday_percentage / 100) * ((100 - monday_percentage) / 100 * total_exams) = exams_left :=
by sorry

end NUMINAMATH_CALUDE_monday_grading_percentage_l883_88329


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l883_88323

theorem quadratic_inequality_minimum (a b c : ℝ) : 
  (∀ x, ax^2 + b*x + c < 0 ↔ -1 < x ∧ x < 3) →
  (∃ m, ∀ a b c, (∀ x, ax^2 + b*x + c < 0 ↔ -1 < x ∧ x < 3) → 
    b - 2*c + 1/a ≥ m ∧ 
    (∃ a₀ b₀ c₀, (∀ x, a₀*x^2 + b₀*x + c₀ < 0 ↔ -1 < x ∧ x < 3) ∧ 
      b₀ - 2*c₀ + 1/a₀ = m)) ∧
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l883_88323


namespace NUMINAMATH_CALUDE_gcd_12345_23456_34567_l883_88354

theorem gcd_12345_23456_34567 : Nat.gcd 12345 (Nat.gcd 23456 34567) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_23456_34567_l883_88354


namespace NUMINAMATH_CALUDE_hyperbola_relation_l883_88391

/-- Two hyperbolas M and N with the given properties -/
structure HyperbolaPair where
  /-- Eccentricity of hyperbola M -/
  e₁ : ℝ
  /-- Eccentricity of hyperbola N -/
  e₂ : ℝ
  /-- Half the length of the transverse axis of hyperbola N -/
  a : ℝ
  /-- Half the length of the conjugate axis of both hyperbolas -/
  b : ℝ
  /-- M and N are centered at the origin -/
  center_origin : True
  /-- Symmetric axes are coordinate axes -/
  symmetric_axes : True
  /-- Length of transverse axis of M is twice that of N -/
  transverse_axis_relation : True
  /-- Conjugate axes of M and N are equal -/
  conjugate_axis_equal : True
  /-- e₁ and e₂ are positive -/
  e₁_pos : e₁ > 0
  e₂_pos : e₂ > 0
  /-- a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- Definition of e₂ for hyperbola N -/
  e₂_def : e₂^2 = 1 + b^2 / a^2
  /-- Definition of e₁ for hyperbola M -/
  e₁_def : e₁^2 = 1 + b^2 / (4*a^2)

/-- The point (e₁, e₂) satisfies the equation of the hyperbola 4x²-y²=3 -/
theorem hyperbola_relation (h : HyperbolaPair) : 4 * h.e₁^2 - h.e₂^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_relation_l883_88391


namespace NUMINAMATH_CALUDE_circle_intersection_l883_88379

/-- The equation of the circle C -/
def C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

/-- The equation of the line l -/
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

/-- Theorem stating when C represents a circle and the value of m when C intersects l -/
theorem circle_intersection :
  (∃ (m : ℝ), ∀ (x y : ℝ), C x y m → m < 5) ∧
  (∃ (m : ℝ), ∀ (x y : ℝ), C x y m → l x y → 
    ∃ (M N : ℝ × ℝ), C M.1 M.2 m ∧ C N.1 N.2 m ∧ l M.1 M.2 ∧ l N.1 N.2 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = (4 / Real.sqrt 5)^2 → m = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_l883_88379


namespace NUMINAMATH_CALUDE_eric_pencils_l883_88375

theorem eric_pencils (containers : ℕ) (additional_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : containers = 5)
  (h2 : additional_pencils = 30)
  (h3 : total_pencils = 36)
  (h4 : total_pencils % containers = 0) :
  total_pencils - additional_pencils = 6 := by
  sorry

end NUMINAMATH_CALUDE_eric_pencils_l883_88375


namespace NUMINAMATH_CALUDE_existence_of_x0_iff_b_negative_l883_88348

open Real

theorem existence_of_x0_iff_b_negative (a b : ℝ) (ha : a > 0) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ log x₀ > a * sqrt x₀ + b / sqrt x₀) ↔ b < 0 :=
sorry

end NUMINAMATH_CALUDE_existence_of_x0_iff_b_negative_l883_88348


namespace NUMINAMATH_CALUDE_exam_questions_attempted_student_exam_result_l883_88353

theorem exam_questions_attempted (correct_score : ℕ) (wrong_penalty : ℕ) 
  (total_score : ℤ) (correct_answers : ℕ) : ℕ :=
  let wrong_answers := total_score - correct_score * correct_answers
  correct_answers + wrong_answers.toNat

-- Statement of the problem
theorem student_exam_result : 
  exam_questions_attempted 4 1 130 38 = 60 := by
  sorry

end NUMINAMATH_CALUDE_exam_questions_attempted_student_exam_result_l883_88353


namespace NUMINAMATH_CALUDE_quadratic_equation_necessary_not_sufficient_l883_88331

theorem quadratic_equation_necessary_not_sufficient :
  ∀ x : ℝ, 
    (x = 5 → x^2 - 4*x - 5 = 0) ∧ 
    ¬(x^2 - 4*x - 5 = 0 → x = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_necessary_not_sufficient_l883_88331


namespace NUMINAMATH_CALUDE_arithmetic_sequence_669th_term_l883_88321

/-- For an arithmetic sequence with first term 1 and common difference 3,
    the 669th term is 2005. -/
theorem arithmetic_sequence_669th_term : 
  ∀ (a : ℕ → ℤ), 
    (a 1 = 1) → 
    (∀ n : ℕ, a (n + 1) - a n = 3) → 
    (a 669 = 2005) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_669th_term_l883_88321


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l883_88318

theorem sum_of_a_and_b (a b : ℝ) (h1 : a + 3*b = 27) (h2 : 5*a + 4*b = 47) : 
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l883_88318


namespace NUMINAMATH_CALUDE_prime_dates_count_l883_88315

/-- A prime date is a date where both the month and day are prime numbers -/
def PrimeDate (month : ℕ) (day : ℕ) : Prop :=
  Nat.Prime month ∧ Nat.Prime day

/-- The list of prime months in our scenario -/
def PrimeMonths : List ℕ := [2, 3, 5, 7, 11, 13]

/-- The number of days in each prime month for a non-leap year -/
def DaysInPrimeMonth (month : ℕ) : ℕ :=
  if month = 2 then 28
  else if month = 11 then 30
  else 31

/-- The list of prime days -/
def PrimeDays : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

/-- The number of prime dates in a given month -/
def PrimeDatesInMonth (month : ℕ) : ℕ :=
  (PrimeDays.filter (· ≤ DaysInPrimeMonth month)).length

theorem prime_dates_count : 
  (PrimeMonths.map PrimeDatesInMonth).sum = 62 := by
  sorry

end NUMINAMATH_CALUDE_prime_dates_count_l883_88315


namespace NUMINAMATH_CALUDE_pie_shop_revenue_l883_88368

/-- The revenue calculation for a pie shop --/
theorem pie_shop_revenue : 
  (price_per_slice : ℕ) → 
  (slices_per_pie : ℕ) → 
  (number_of_pies : ℕ) → 
  price_per_slice = 5 →
  slices_per_pie = 4 →
  number_of_pies = 9 →
  price_per_slice * slices_per_pie * number_of_pies = 180 := by
  sorry

end NUMINAMATH_CALUDE_pie_shop_revenue_l883_88368


namespace NUMINAMATH_CALUDE_tiles_per_square_foot_l883_88383

def wall1_length : ℝ := 5
def wall1_width : ℝ := 8
def wall2_length : ℝ := 7
def wall2_width : ℝ := 8
def turquoise_cost : ℝ := 13
def purple_cost : ℝ := 11
def total_savings : ℝ := 768

theorem tiles_per_square_foot :
  let total_area := wall1_length * wall1_width + wall2_length * wall2_width
  let cost_difference := turquoise_cost - purple_cost
  let total_tiles := total_savings / cost_difference
  total_tiles / total_area = 4 := by sorry

end NUMINAMATH_CALUDE_tiles_per_square_foot_l883_88383


namespace NUMINAMATH_CALUDE_hamster_lifespan_difference_l883_88362

/-- Represents the lifespans of a hamster, bat, and frog. -/
structure AnimalLifespans where
  hamster : ℕ
  bat : ℕ
  frog : ℕ

/-- The conditions of the problem. -/
def problemConditions (a : AnimalLifespans) : Prop :=
  a.bat = 10 ∧
  a.frog = 4 * a.hamster ∧
  a.hamster + a.bat + a.frog = 30

/-- The theorem to be proved. -/
theorem hamster_lifespan_difference (a : AnimalLifespans) 
  (h : problemConditions a) : a.bat - a.hamster = 6 := by
  sorry

#check hamster_lifespan_difference

end NUMINAMATH_CALUDE_hamster_lifespan_difference_l883_88362


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l883_88350

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 - 9x + 4 -/
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 4

theorem quadratic_discriminant :
  discriminant a b c = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l883_88350


namespace NUMINAMATH_CALUDE_cosine_range_theorem_l883_88373

theorem cosine_range_theorem (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  x ∈ {x | Real.cos x ≤ 1/2} ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3) := by
sorry

end NUMINAMATH_CALUDE_cosine_range_theorem_l883_88373


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l883_88386

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 4 = 5 →
  a 8 = 6 →
  a 2 * a 10 = 30 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l883_88386


namespace NUMINAMATH_CALUDE_triangle_theorem_l883_88310

/-- Theorem about a triangle ABC with specific conditions -/
theorem triangle_theorem (a b c A B C : ℝ) : 
  -- Given conditions
  (2 * b * Real.cos C = 2 * a - c) →  -- Condition from the problem
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area condition
  (b = 2) →  -- Given value of b
  -- Conclusions to prove
  (B = Real.pi / 3) ∧  -- 60 degrees in radians
  (a = 2) ∧ 
  (c = 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l883_88310


namespace NUMINAMATH_CALUDE_women_in_third_group_l883_88356

/-- Represents the work rate of a single person -/
structure WorkRate where
  rate : ℝ
  positive : rate > 0

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work rate of a group -/
def totalWorkRate (m w : WorkRate) (group : WorkGroup) : ℝ :=
  group.men • m.rate + group.women • w.rate

theorem women_in_third_group 
  (m w : WorkRate)
  (group1 group2 group3 : WorkGroup) :
  totalWorkRate m w group1 = totalWorkRate m w group2 →
  group1.men = 3 →
  group1.women = 8 →
  group2.men = 6 →
  group2.women = 2 →
  group3.men = 4 →
  totalWorkRate m w group3 = 0.9285714285714286 * totalWorkRate m w group1 →
  group3.women = 5 := by
  sorry


end NUMINAMATH_CALUDE_women_in_third_group_l883_88356


namespace NUMINAMATH_CALUDE_at_least_one_is_one_l883_88371

theorem at_least_one_is_one (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 1/a + 1/b + 1/c) : 
  a = 1 ∨ b = 1 ∨ c = 1 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_is_one_l883_88371


namespace NUMINAMATH_CALUDE_paintings_distribution_l883_88328

theorem paintings_distribution (total_paintings : ℕ) (num_rooms : ℕ) (paintings_per_room : ℕ) :
  total_paintings = 32 →
  num_rooms = 4 →
  paintings_per_room = total_paintings / num_rooms →
  paintings_per_room = 8 := by
  sorry

end NUMINAMATH_CALUDE_paintings_distribution_l883_88328


namespace NUMINAMATH_CALUDE_compare_f_values_l883_88398

/-- Given 0 < a < 1, this function satisfies f(log_a x) = (a(x^2 - 1)) / (x(a^2 - 1)) for any x > 0 -/
noncomputable def f (a : ℝ) (t : ℝ) : ℝ := sorry

/-- Theorem: For 0 < a < 1, given function f and m > n > 0, we have f(1/n) > f(1/m) -/
theorem compare_f_values (a m n : ℝ) (ha : 0 < a) (ha' : a < 1) (hmn : m > n) (hn : n > 0) :
  f a (1/n) > f a (1/m) := by sorry

end NUMINAMATH_CALUDE_compare_f_values_l883_88398


namespace NUMINAMATH_CALUDE_bisection_method_max_experiments_l883_88349

theorem bisection_method_max_experiments (n : ℕ) (h : n = 33) :
  ∃ k : ℕ, k = 6 ∧ ∀ m : ℕ, 2^m < n → m < k :=
sorry

end NUMINAMATH_CALUDE_bisection_method_max_experiments_l883_88349


namespace NUMINAMATH_CALUDE_egg_sales_income_l883_88369

theorem egg_sales_income (num_hens : ℕ) (eggs_per_hen_per_week : ℕ) (price_per_dozen : ℕ) (num_weeks : ℕ) :
  num_hens = 10 →
  eggs_per_hen_per_week = 12 →
  price_per_dozen = 3 →
  num_weeks = 4 →
  (num_hens * eggs_per_hen_per_week * num_weeks / 12) * price_per_dozen = 120 := by
  sorry

#check egg_sales_income

end NUMINAMATH_CALUDE_egg_sales_income_l883_88369


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l883_88301

def M : Set ℝ := {x | x^2 ≤ 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l883_88301


namespace NUMINAMATH_CALUDE_floor_minus_y_eq_zero_l883_88303

theorem floor_minus_y_eq_zero (y : ℝ) (h : ⌊y⌋ + ⌈y⌉ = 2 * y) : ⌊y⌋ - y = 0 := by
  sorry

end NUMINAMATH_CALUDE_floor_minus_y_eq_zero_l883_88303


namespace NUMINAMATH_CALUDE_equation_solution_l883_88367

theorem equation_solution (x : ℝ) :
  (2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x)) ↔
  (∃ k : ℤ, x = (π / 16) * (4 * ↑k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l883_88367


namespace NUMINAMATH_CALUDE_cd_cost_with_tax_l883_88302

/-- The cost of a CD including sales tax -/
def total_cost (price : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate)

/-- Theorem stating that the total cost of a CD priced at $14.99 with 15% sales tax is $17.24 -/
theorem cd_cost_with_tax : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |total_cost 14.99 0.15 - 17.24| < ε :=
sorry

end NUMINAMATH_CALUDE_cd_cost_with_tax_l883_88302


namespace NUMINAMATH_CALUDE_final_parity_after_odd_changes_not_even_after_33_changes_l883_88377

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- Function to change the parity -/
def changeParity (p : Parity) : Parity :=
  match p with
  | Parity.Even => Parity.Odd
  | Parity.Odd => Parity.Even

/-- Function to apply n changes to initial parity -/
def applyNChanges (initial : Parity) (n : Nat) : Parity :=
  match n with
  | 0 => initial
  | k + 1 => changeParity (applyNChanges initial k)

theorem final_parity_after_odd_changes 
  (initial : Parity) (n : Nat) (h : Odd n) :
  applyNChanges initial n ≠ initial := by
  sorry

/-- Main theorem: After 33 changes, an initially even number cannot be even -/
theorem not_even_after_33_changes :
  applyNChanges Parity.Even 33 ≠ Parity.Even := by
  sorry

end NUMINAMATH_CALUDE_final_parity_after_odd_changes_not_even_after_33_changes_l883_88377


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achievable_l883_88319

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) 
  (pos₁ : 0 < y₁) (pos₂ : 0 < y₂) (pos₃ : 0 < y₃)
  (sum_constraint : y₁ + 3 * y₂ + 5 * y₃ = 120) :
  720 / 7 ≤ y₁^2 + y₂^2 + y₃^2 :=
by sorry

theorem min_sum_squares_achievable :
  ∃ (y₁ y₂ y₃ : ℝ), 0 < y₁ ∧ 0 < y₂ ∧ 0 < y₃ ∧ 
  y₁ + 3 * y₂ + 5 * y₃ = 120 ∧
  y₁^2 + y₂^2 + y₃^2 = 720 / 7 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achievable_l883_88319


namespace NUMINAMATH_CALUDE_initial_tagged_fish_l883_88326

/-- The number of fish initially caught and tagged -/
def T : ℕ := sorry

/-- The total number of fish in the pond -/
def N : ℕ := 800

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 40

/-- The number of tagged fish in the second catch -/
def tagged_in_second : ℕ := 2

theorem initial_tagged_fish :
  (T : ℚ) / N = tagged_in_second / second_catch ∧ T = 40 := by sorry

end NUMINAMATH_CALUDE_initial_tagged_fish_l883_88326


namespace NUMINAMATH_CALUDE_andrews_balloons_l883_88374

/-- Given a number of blue and purple balloons, calculates how many balloons are left after sharing half of the total. -/
def balloons_left (blue : ℕ) (purple : ℕ) : ℕ :=
  (blue + purple) / 2

/-- Theorem stating that given 303 blue balloons and 453 purple balloons, 
    the number of balloons left after sharing half is 378. -/
theorem andrews_balloons : balloons_left 303 453 = 378 := by
  sorry

end NUMINAMATH_CALUDE_andrews_balloons_l883_88374


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l883_88327

theorem min_sum_of_squares (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → a^2 + b^2 ≥ m) ∧ 
  (∃ (c d : ℝ), (c + 5)^2 + (d - 12)^2 = 14^2 ∧ c^2 + d^2 = m) ∧
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l883_88327


namespace NUMINAMATH_CALUDE_exist_consecutive_lucky_years_l883_88330

/-- Returns the first two digits of a four-digit number -/
def firstTwoDigits (n : ℕ) : ℕ := n / 100

/-- Returns the last two digits of a four-digit number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- Checks if a year is lucky -/
def isLuckyYear (year : ℕ) : Prop :=
  year % (firstTwoDigits year + lastTwoDigits year) = 0

/-- Theorem: There exist two consecutive lucky years -/
theorem exist_consecutive_lucky_years :
  ∃ (y : ℕ), 1000 ≤ y ∧ y < 9999 ∧ isLuckyYear y ∧ isLuckyYear (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_exist_consecutive_lucky_years_l883_88330


namespace NUMINAMATH_CALUDE_tom_gathering_plates_l883_88394

/-- The number of plates used during a multi-day stay with multiple meals per day -/
def plates_used (people : ℕ) (days : ℕ) (meals_per_day : ℕ) (courses_per_meal : ℕ) (plates_per_course : ℕ) : ℕ :=
  people * days * meals_per_day * courses_per_meal * plates_per_course

/-- Theorem: Given the conditions from Tom's gathering, the total number of plates used is 1728 -/
theorem tom_gathering_plates :
  plates_used 12 6 4 3 2 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_tom_gathering_plates_l883_88394


namespace NUMINAMATH_CALUDE_car_price_before_discount_l883_88397

theorem car_price_before_discount 
  (discount_percentage : ℝ) 
  (price_after_discount : ℝ) 
  (h1 : discount_percentage = 55) 
  (h2 : price_after_discount = 450000) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_percentage / 100) = price_after_discount ∧ 
    original_price = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_car_price_before_discount_l883_88397


namespace NUMINAMATH_CALUDE_rod_cutting_l883_88387

theorem rod_cutting (rod_length_m : ℕ) (piece_length_cm : ℕ) : 
  rod_length_m = 17 → piece_length_cm = 85 → (rod_length_m * 100) / piece_length_cm = 20 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l883_88387


namespace NUMINAMATH_CALUDE_least_possible_difference_l883_88396

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → Odd y → Odd z → 
  (∀ d : ℤ, z - x ≥ d → d ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l883_88396


namespace NUMINAMATH_CALUDE_composition_equals_26_l883_88335

-- Define the functions f and g
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := 2 * x

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

-- State the theorem
theorem composition_equals_26 : f (g_inv (f_inv (f_inv (g (f 23))))) = 26 := by sorry

end NUMINAMATH_CALUDE_composition_equals_26_l883_88335


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_optimization_l883_88370

/-- Given a positive real number S, this theorem states that for any rectangle with area S and perimeter p,
    the expression S / (2S + p + 2) is maximized when the rectangle is a square, 
    and the maximum value is S / (2(√S + 1)²). -/
theorem rectangle_area_perimeter_optimization (S : ℝ) (hS : S > 0) :
  ∀ (a b : ℝ), a > 0 → b > 0 → a * b = S →
    S / (2 * S + 2 * (a + b) + 2) ≤ S / (2 * (Real.sqrt S + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_optimization_l883_88370


namespace NUMINAMATH_CALUDE_fuel_consumption_result_l883_88314

/-- Represents the fuel consumption problem --/
structure FuelConsumption where
  initial_capacity : ℝ
  january_level : ℝ
  may_level : ℝ

/-- Calculates the total fuel consumption given the problem parameters --/
def total_consumption (fc : FuelConsumption) : ℝ :=
  (fc.initial_capacity - fc.january_level) + (fc.initial_capacity - fc.may_level)

/-- Theorem stating that the total fuel consumption is 4582 L --/
theorem fuel_consumption_result (fc : FuelConsumption) 
  (h1 : fc.initial_capacity = 3000)
  (h2 : fc.january_level = 180)
  (h3 : fc.may_level = 1238) :
  total_consumption fc = 4582 := by
  sorry

#eval total_consumption ⟨3000, 180, 1238⟩

end NUMINAMATH_CALUDE_fuel_consumption_result_l883_88314


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l883_88395

theorem arithmetic_sequence_ratio (x y d₁ d₂ : ℝ) (h₁ : d₁ ≠ 0) (h₂ : d₂ ≠ 0) : 
  (x + 4 * d₁ = y) → (x + 5 * d₂ = y) → d₁ / d₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l883_88395


namespace NUMINAMATH_CALUDE_whitney_cant_afford_l883_88346

def poster_price : ℚ := 7.5
def notebook_price : ℚ := 5.25
def bookmark_price : ℚ := 3.1
def pencil_price : ℚ := 1.15
def sales_tax_rate : ℚ := 0.08
def initial_money : ℚ := 40

def total_cost (poster_qty notebook_qty bookmark_qty pencil_qty : ℕ) : ℚ :=
  let subtotal := poster_price * poster_qty + notebook_price * notebook_qty + 
                  bookmark_price * bookmark_qty + pencil_price * pencil_qty
  subtotal * (1 + sales_tax_rate)

theorem whitney_cant_afford (poster_qty notebook_qty bookmark_qty pencil_qty : ℕ) 
  (h_poster : poster_qty = 3)
  (h_notebook : notebook_qty = 4)
  (h_bookmark : bookmark_qty = 5)
  (h_pencil : pencil_qty = 2) :
  total_cost poster_qty notebook_qty bookmark_qty pencil_qty > initial_money :=
by sorry

end NUMINAMATH_CALUDE_whitney_cant_afford_l883_88346


namespace NUMINAMATH_CALUDE_harmonic_numbers_theorem_l883_88372

/-- Definition of harmonic numbers -/
def are_harmonic (a b c : ℝ) : Prop :=
  1/b - 1/a = 1/c - 1/b

/-- Theorem: For harmonic numbers x, 5, 3 where x > 5, x = 15 -/
theorem harmonic_numbers_theorem (x : ℝ) 
  (h1 : are_harmonic x 5 3)
  (h2 : x > 5) : 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_numbers_theorem_l883_88372


namespace NUMINAMATH_CALUDE_problem_statement_l883_88352

theorem problem_statement (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4)
  (h2 : c ^ 3 = d ^ 2)
  (h3 : c - a = 19) :
  d - b = 757 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l883_88352


namespace NUMINAMATH_CALUDE_james_recovery_time_l883_88360

/-- Calculates the total number of days before James can resume heavy lifting after an injury -/
def time_to_resume_heavy_lifting (
  initial_pain_duration : ℕ
  ) (healing_time_multiplier : ℕ
  ) (additional_caution_period : ℕ
  ) (light_exercises_duration : ℕ
  ) (potential_complication_duration : ℕ
  ) (moderate_intensity_duration : ℕ
  ) (transition_to_heavy_lifting : ℕ
  ) : ℕ :=
  let initial_healing_time := initial_pain_duration * healing_time_multiplier
  let total_initial_recovery := initial_healing_time + additional_caution_period
  let light_exercises_with_complication := light_exercises_duration + potential_complication_duration
  let total_before_transition := total_initial_recovery + light_exercises_with_complication + moderate_intensity_duration
  total_before_transition + transition_to_heavy_lifting

/-- Theorem stating that given the specific conditions, James will take 67 days to resume heavy lifting -/
theorem james_recovery_time : 
  time_to_resume_heavy_lifting 3 5 3 14 7 7 21 = 67 := by
  sorry

end NUMINAMATH_CALUDE_james_recovery_time_l883_88360


namespace NUMINAMATH_CALUDE_intersection_of_four_convex_sets_l883_88392

-- Define a type for points in a plane
variable {Point : Type}

-- Define a type for convex sets in a plane
variable {ConvexSet : Type}

-- Define a function to check if a point is in a convex set
variable (in_set : Point → ConvexSet → Prop)

-- Define a function to check if a set is convex
variable (is_convex : ConvexSet → Prop)

-- Define a function to represent the intersection of sets
variable (intersection : List ConvexSet → Set Point)

-- Theorem statement
theorem intersection_of_four_convex_sets
  (C1 C2 C3 C4 : ConvexSet)
  (convex1 : is_convex C1)
  (convex2 : is_convex C2)
  (convex3 : is_convex C3)
  (convex4 : is_convex C4)
  (intersect_three1 : (intersection [C1, C2, C3]).Nonempty)
  (intersect_three2 : (intersection [C1, C2, C4]).Nonempty)
  (intersect_three3 : (intersection [C1, C3, C4]).Nonempty)
  (intersect_three4 : (intersection [C2, C3, C4]).Nonempty) :
  (intersection [C1, C2, C3, C4]).Nonempty :=
sorry

end NUMINAMATH_CALUDE_intersection_of_four_convex_sets_l883_88392


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l883_88338

-- Define the function g(x)
def g (x : ℝ) : ℝ := -3 * x^3 + 5 * x + 1

-- State the theorem
theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) := by
  sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l883_88338


namespace NUMINAMATH_CALUDE_boat_rental_cost_l883_88384

theorem boat_rental_cost (students : ℕ) (boat_capacity : ℕ) (rental_fee : ℕ) 
  (h1 : students = 42)
  (h2 : boat_capacity = 6)
  (h3 : rental_fee = 125) :
  (((students + boat_capacity - 1) / boat_capacity) * rental_fee) = 875 :=
by
  sorry

#check boat_rental_cost

end NUMINAMATH_CALUDE_boat_rental_cost_l883_88384


namespace NUMINAMATH_CALUDE_factorization_equality_l883_88347

theorem factorization_equality (x : ℝ) : x^2 - x - 6 = (x - 3) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l883_88347


namespace NUMINAMATH_CALUDE_janous_inequality_l883_88381

theorem janous_inequality (x y : ℝ) (hx : x > -1) (hy : y > -1) (hsum : x + y = 1) :
  x / (y + 1) + y / (x + 1) ≥ 2 / 3 ∧
  (x / (y + 1) + y / (x + 1) = 2 / 3 ↔ x = 1 / 2 ∧ y = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l883_88381


namespace NUMINAMATH_CALUDE_count_special_numbers_is_324_l883_88325

/-- The count of 5-digit numbers beginning with 2 that have exactly three identical digits (which are not 2) -/
def count_special_numbers : ℕ :=
  4 * 9 * 9

/-- The theorem stating that the count of special numbers is 324 -/
theorem count_special_numbers_is_324 : count_special_numbers = 324 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_is_324_l883_88325


namespace NUMINAMATH_CALUDE_water_added_to_alcohol_solution_l883_88359

/-- Proves that adding 5 liters of water to a 15-liter solution with 26% alcohol 
    results in a new solution with 19.5% alcohol -/
theorem water_added_to_alcohol_solution :
  let initial_volume : ℝ := 15
  let initial_alcohol_percentage : ℝ := 0.26
  let water_added : ℝ := 5
  let final_alcohol_percentage : ℝ := 0.195
  let initial_alcohol_volume := initial_volume * initial_alcohol_percentage
  let final_volume := initial_volume + water_added
  initial_alcohol_volume / final_volume = final_alcohol_percentage := by
  sorry


end NUMINAMATH_CALUDE_water_added_to_alcohol_solution_l883_88359


namespace NUMINAMATH_CALUDE_subtraction_and_division_l883_88306

theorem subtraction_and_division : ((-120) - (-60)) / (-30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_and_division_l883_88306


namespace NUMINAMATH_CALUDE_expand_polynomial_l883_88365

/-- Proves the expansion of (12x^2 + 5x - 3) * (3x^3 + 2) -/
theorem expand_polynomial (x : ℝ) :
  (12 * x^2 + 5 * x - 3) * (3 * x^3 + 2) =
  36 * x^5 + 15 * x^4 - 9 * x^3 + 24 * x^2 + 10 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l883_88365


namespace NUMINAMATH_CALUDE_triangle_inequality_l883_88343

/-- Given a triangle ABC with side lengths a, b, c, circumradius R, inradius r, and semiperimeter p,
    prove that (a / (p - a)) + (b / (p - b)) + (c / (p - c)) ≥ 3R / r,
    with equality if and only if the triangle is equilateral. -/
theorem triangle_inequality (a b c R r p : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hR : R > 0) (hr : r > 0) (hp : p > 0) (h_semi : p = (a + b + c) / 2)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
    (a / (p - a)) + (b / (p - b)) + (c / (p - c)) ≥ 3 * R / r ∧
    ((a / (p - a)) + (b / (p - b)) + (c / (p - c)) = 3 * R / r ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l883_88343


namespace NUMINAMATH_CALUDE_music_festival_group_formation_l883_88390

def total_friends : ℕ := 10
def musicians : ℕ := 4
def non_musicians : ℕ := 6
def group_size : ℕ := 4

theorem music_festival_group_formation :
  (Nat.choose total_friends group_size) - (Nat.choose non_musicians group_size) = 195 :=
sorry

end NUMINAMATH_CALUDE_music_festival_group_formation_l883_88390


namespace NUMINAMATH_CALUDE_two_person_subcommittees_of_six_l883_88366

/-- The number of two-person sub-committees from a six-person committee -/
def two_person_subcommittees (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: The number of two-person sub-committees from a six-person committee is 15 -/
theorem two_person_subcommittees_of_six :
  two_person_subcommittees 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_of_six_l883_88366


namespace NUMINAMATH_CALUDE_same_team_probability_l883_88317

/-- The probability of two students choosing the same team out of three teams -/
theorem same_team_probability (num_teams : ℕ) (h : num_teams = 3) :
  (num_teams : ℚ) / (num_teams^2 : ℚ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_same_team_probability_l883_88317


namespace NUMINAMATH_CALUDE_min_value_expression_l883_88358

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l883_88358


namespace NUMINAMATH_CALUDE_sin_70_degrees_l883_88313

theorem sin_70_degrees (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by sorry

end NUMINAMATH_CALUDE_sin_70_degrees_l883_88313


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_l883_88357

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the two fixed points
def point1 : ℝ × ℝ := (0, 2)
def point2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem equation_describes_ellipse :
  ∀ x y : ℝ, equation x y ↔ 
    (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧
      Real.sqrt ((p.1 - point1.1)^2 + (p.2 - point1.2)^2) +
      Real.sqrt ((p.1 - point2.1)^2 + (p.2 - point2.2)^2) = 12) :=
by sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_l883_88357


namespace NUMINAMATH_CALUDE_b_finish_days_l883_88300

/-- The number of days A needs to complete the entire work -/
def a_total_days : ℝ := 20

/-- The number of days B needs to complete the entire work -/
def b_total_days : ℝ := 30

/-- The number of days A worked before leaving -/
def a_worked_days : ℝ := 10

/-- Theorem: Given the conditions, B can finish the remaining work in 15 days -/
theorem b_finish_days : 
  ∃ (b_days : ℝ), 
    (1 / b_total_days) * b_days = 1 - (a_worked_days / a_total_days) ∧ 
    b_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_b_finish_days_l883_88300


namespace NUMINAMATH_CALUDE_intersection_and_min_area_l883_88355

noncomputable section

-- Define the hyperbola C₁
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (2 * a^2) = 1

-- Define the parabola C₂
def C₂ (a : ℝ) (x y : ℝ) : Prop := y^2 = -4 * Real.sqrt 3 * a * x

-- Define the focus F₁
def F₁ (a : ℝ) : ℝ × ℝ := (-Real.sqrt 3 * a, 0)

-- Define a chord through F₁
def chord_through_F₁ (a k : ℝ) (x y : ℝ) : Prop := y = k * (x + Real.sqrt 3 * a)

-- Define the area of triangle AOB
def area_AOB (a k : ℝ) : ℝ := 6 * a^2 * Real.sqrt (1 + 1/k^2)

theorem intersection_and_min_area (a : ℝ) (h : a > 0) :
  (∃! (p q : ℝ × ℝ), p ≠ q ∧ C₁ a p.1 p.2 ∧ C₂ a p.1 p.2 ∧ C₁ a q.1 q.2 ∧ C₂ a q.1 q.2) ∧
  (∀ k : ℝ, area_AOB a k ≥ 6 * a^2) ∧
  (∃ k : ℝ, chord_through_F₁ a k (-Real.sqrt 3 * a) 0 ∧ area_AOB a k = 6 * a^2) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_min_area_l883_88355


namespace NUMINAMATH_CALUDE_trig_identity_l883_88336

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l883_88336


namespace NUMINAMATH_CALUDE_smallest_divisor_sum_of_squares_l883_88344

theorem smallest_divisor_sum_of_squares (n : ℕ) : n ≥ 2 →
  (∃ a b : ℕ, 
    a > 1 ∧ 
    a ∣ n ∧ 
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    b ∣ n ∧
    n = a^2 + b^2) →
  n = 8 ∨ n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_sum_of_squares_l883_88344


namespace NUMINAMATH_CALUDE_board_numbers_product_l883_88378

theorem board_numbers_product (a b c d e : ℤ) : 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {6, 9, 10, 13, 13, 14, 17, 17, 20, 21} →
  a * b * c * d * e = 4320 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_product_l883_88378


namespace NUMINAMATH_CALUDE_longest_piece_length_l883_88305

/-- Given a rope of length 92.5 inches cut into three pieces in the ratio 3:5:8,
    the length of the longest piece is 46.25 inches. -/
theorem longest_piece_length (total_length : ℝ) (ratio_1 ratio_2 ratio_3 : ℕ) 
    (h1 : total_length = 92.5)
    (h2 : ratio_1 = 3)
    (h3 : ratio_2 = 5)
    (h4 : ratio_3 = 8) :
    (ratio_3 : ℝ) * total_length / ((ratio_1 : ℝ) + (ratio_2 : ℝ) + (ratio_3 : ℝ)) = 46.25 :=
by sorry

end NUMINAMATH_CALUDE_longest_piece_length_l883_88305


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l883_88363

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l883_88363


namespace NUMINAMATH_CALUDE_sixtieth_point_coordinates_l883_88364

/-- Represents a point with integer coordinates -/
structure Point where
  x : ℕ
  y : ℕ

/-- The sequence of points -/
def pointSequence : ℕ → Point := sorry

/-- The sum of x and y coordinates for the nth point -/
def coordinateSum (n : ℕ) : ℕ := (pointSequence n).x + (pointSequence n).y

/-- The row number for a given point in the sequence -/
def rowNumber (n : ℕ) : ℕ := sorry

/-- The property that the coordinate sum increases by 1 for every n points -/
axiom coordinate_sum_property (n : ℕ) :
  ∀ k, k > n → coordinateSum k = coordinateSum n + (rowNumber k - rowNumber n)

/-- The main theorem: The 60th point has coordinates (5,7) -/
theorem sixtieth_point_coordinates :
  pointSequence 60 = Point.mk 5 7 := by sorry

end NUMINAMATH_CALUDE_sixtieth_point_coordinates_l883_88364


namespace NUMINAMATH_CALUDE_raj_earnings_l883_88304

/-- Calculates the total earnings for Raj over two weeks given the hours worked and wage difference --/
def total_earnings (hours_week1 hours_week2 : ℕ) (wage_difference : ℚ) : ℚ :=
  let hourly_wage := wage_difference / (hours_week2 - hours_week1)
  (hours_week1 + hours_week2) * hourly_wage

/-- Proves that Raj's total earnings for the first two weeks of July is $198.00 --/
theorem raj_earnings :
  let hours_week1 : ℕ := 12
  let hours_week2 : ℕ := 18
  let wage_difference : ℚ := 39.6
  total_earnings hours_week1 hours_week2 wage_difference = 198 := by
  sorry

#eval total_earnings 12 18 (39.6 : ℚ)

end NUMINAMATH_CALUDE_raj_earnings_l883_88304


namespace NUMINAMATH_CALUDE_alok_payment_l883_88320

def chapati_quantity : ℕ := 16
def rice_quantity : ℕ := 5
def vegetable_quantity : ℕ := 7
def icecream_quantity : ℕ := 6

def chapati_price : ℕ := 6
def rice_price : ℕ := 45
def vegetable_price : ℕ := 70

def total_cost : ℕ := chapati_quantity * chapati_price + 
                       rice_quantity * rice_price + 
                       vegetable_quantity * vegetable_price

theorem alok_payment : total_cost = 811 := by
  sorry

end NUMINAMATH_CALUDE_alok_payment_l883_88320


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l883_88388

theorem difference_of_squares_factorization (x : ℝ) : 
  x^2 - 4 = (x + 2) * (x - 2) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l883_88388


namespace NUMINAMATH_CALUDE_hotel_discount_l883_88324

/-- Calculate the discount for a hotel stay given the number of nights, cost per night, and total amount paid. -/
theorem hotel_discount (nights : ℕ) (cost_per_night : ℕ) (total_paid : ℕ) : 
  nights = 3 → cost_per_night = 250 → total_paid = 650 → 
  nights * cost_per_night - total_paid = 100 := by
sorry

end NUMINAMATH_CALUDE_hotel_discount_l883_88324


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l883_88309

theorem sqrt_sum_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (Real.sqrt x + Real.sqrt y)^8 ≥ 64 * x * y * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l883_88309


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l883_88316

theorem complex_fraction_equality : 2 + (3 / (4 + (5/6))) = 76/29 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l883_88316


namespace NUMINAMATH_CALUDE_triple_lcm_equation_solution_l883_88382

theorem triple_lcm_equation_solution (a b c n : ℕ+) 
  (h1 : a^2 + b^2 = n * Nat.lcm a b + n^2)
  (h2 : b^2 + c^2 = n * Nat.lcm b c + n^2)
  (h3 : c^2 + a^2 = n * Nat.lcm c a + n^2) :
  a = n ∧ b = n ∧ c = n := by
sorry

end NUMINAMATH_CALUDE_triple_lcm_equation_solution_l883_88382


namespace NUMINAMATH_CALUDE_one_fourths_in_three_eighths_l883_88342

theorem one_fourths_in_three_eighths (x : ℚ) : x = 3/8 → (x / (1/4)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourths_in_three_eighths_l883_88342


namespace NUMINAMATH_CALUDE_geese_count_l883_88389

/-- The number of ducks in the marsh -/
def ducks : ℕ := 37

/-- The total number of birds in the marsh -/
def total_birds : ℕ := 95

/-- The number of geese in the marsh -/
def geese : ℕ := total_birds - ducks

theorem geese_count : geese = 58 := by sorry

end NUMINAMATH_CALUDE_geese_count_l883_88389


namespace NUMINAMATH_CALUDE_min_value_theorem_l883_88312

theorem min_value_theorem (a b : ℝ) (h : a * b > 0) :
  (4 * b / a + (a - 2 * b) / b) ≥ 2 ∧
  (4 * b / a + (a - 2 * b) / b = 2 ↔ a = 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l883_88312


namespace NUMINAMATH_CALUDE_fraction_decomposition_l883_88341

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 7) (h2 : x ≠ -6) :
  (3 * x + 5) / (x^2 - x - 42) = 2 / (x - 7) + 1 / (x + 6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l883_88341


namespace NUMINAMATH_CALUDE_exists_integer_sqrt_20n_is_integer_l883_88385

theorem exists_integer_sqrt_20n_is_integer : ∃ n : ℤ, ∃ m : ℤ, 20 * n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_sqrt_20n_is_integer_l883_88385


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l883_88380

/-- A linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point (x, y) lies on the regression line -/
def on_line (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : on_line l₁ s t)
  (h₂ : on_line l₂ s t) :
  ∃ (x y : ℝ), on_line l₁ x y ∧ on_line l₂ x y ∧ x = s ∧ y = t :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersection_l883_88380


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l883_88334

theorem det_of_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 4, 7]
  Matrix.det A = -1 := by
sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l883_88334


namespace NUMINAMATH_CALUDE_negation_of_p_l883_88337

-- Define the proposition p
def p : Prop := ∃ m : ℝ, m > 0 ∧ ∃ x : ℝ, m * x^2 + x - 2*m = 0

-- State the theorem
theorem negation_of_p : ¬p ↔ ∀ m : ℝ, m > 0 → ∀ x : ℝ, m * x^2 + x - 2*m ≠ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l883_88337


namespace NUMINAMATH_CALUDE_butterfly_black_dots_l883_88399

/-- The number of black dots per butterfly -/
def blackDotsPerButterfly (totalButterflies : ℕ) (totalBlackDots : ℕ) : ℕ :=
  totalBlackDots / totalButterflies

/-- Theorem stating that each butterfly has 12 black dots -/
theorem butterfly_black_dots :
  blackDotsPerButterfly 397 4764 = 12 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_black_dots_l883_88399
