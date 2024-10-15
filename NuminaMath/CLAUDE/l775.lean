import Mathlib

namespace NUMINAMATH_CALUDE_percentage_problem_l775_77596

/-- The problem statement --/
theorem percentage_problem (P : ℝ) : 
  (P / 100) * 200 = (60 / 100) * 50 + 30 → P = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l775_77596


namespace NUMINAMATH_CALUDE_expression_zero_at_two_l775_77522

theorem expression_zero_at_two (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  x = 2 → (1 / (x - 1) + 3 / (1 - x^2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_zero_at_two_l775_77522


namespace NUMINAMATH_CALUDE_min_bullseyes_theorem_l775_77589

/-- Represents the archery tournament scenario -/
structure ArcheryTournament where
  total_shots : Nat
  halfway_shots : Nat
  chelsea_lead : Nat
  chelsea_min_score : Nat
  opponent_min_score : Nat

/-- Calculates the minimum number of bullseyes needed for Chelsea to guarantee victory -/
def min_bullseyes_for_victory (tournament : ArcheryTournament) : Nat :=
  let remaining_shots := tournament.total_shots - tournament.halfway_shots
  let chelsea_max_score := remaining_shots * 10
  let opponent_max_score := remaining_shots * 10 - tournament.chelsea_lead
  let chelsea_guaranteed_points := remaining_shots * tournament.chelsea_min_score
  let n := (opponent_max_score - chelsea_guaranteed_points + 10 - 1) / (10 - tournament.chelsea_min_score)
  n + 1

/-- The theorem states that for the given tournament conditions, 
    the minimum number of bullseyes needed for Chelsea to guarantee victory is 87 -/
theorem min_bullseyes_theorem (tournament : ArcheryTournament) 
  (h1 : tournament.total_shots = 200)
  (h2 : tournament.halfway_shots = 100)
  (h3 : tournament.chelsea_lead = 70)
  (h4 : tournament.chelsea_min_score = 5)
  (h5 : tournament.opponent_min_score = 3) :
  min_bullseyes_for_victory tournament = 87 := by
  sorry

end NUMINAMATH_CALUDE_min_bullseyes_theorem_l775_77589


namespace NUMINAMATH_CALUDE_yogurt_combinations_l775_77551

theorem yogurt_combinations (flavors : ℕ) (toppings : ℕ) :
  flavors = 6 → toppings = 8 →
  flavors * (toppings.choose 3) = 336 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l775_77551


namespace NUMINAMATH_CALUDE_student_seat_occupancy_l775_77504

/-- Proves that the fraction of occupied student seats is 4/5 --/
theorem student_seat_occupancy
  (total_chairs : ℕ)
  (rows : ℕ)
  (chairs_per_row : ℕ)
  (awardee_rows : ℕ)
  (admin_teacher_rows : ℕ)
  (parent_rows : ℕ)
  (vacant_student_seats : ℕ)
  (h1 : total_chairs = rows * chairs_per_row)
  (h2 : rows = 10)
  (h3 : chairs_per_row = 15)
  (h4 : awardee_rows = 1)
  (h5 : admin_teacher_rows = 2)
  (h6 : parent_rows = 2)
  (h7 : vacant_student_seats = 15) :
  let student_rows := rows - (awardee_rows + admin_teacher_rows + parent_rows)
  let student_chairs := student_rows * chairs_per_row
  let occupied_student_chairs := student_chairs - vacant_student_seats
  (occupied_student_chairs : ℚ) / student_chairs = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_student_seat_occupancy_l775_77504


namespace NUMINAMATH_CALUDE_tangent_length_is_six_l775_77564

/-- Circle C with equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- Line l with equation x + my - 1 = 0 -/
def line_l (m x y : ℝ) : Prop :=
  x + m*y - 1 = 0

/-- Point A with coordinates (-4, m) -/
def point_A (m : ℝ) : ℝ × ℝ :=
  (-4, m)

/-- Theorem stating that the length of the tangent from A to C is 6 -/
theorem tangent_length_is_six (m : ℝ) : 
  line_l m (2 : ℝ) 1 →  -- line l passes through (2, 1)
  ∃ (B : ℝ × ℝ), 
    circle_C B.1 B.2 ∧  -- B is on circle C
    (∀ (x y : ℝ), circle_C x y → ((x - (-4))^2 + (y - m)^2 ≥ (B.1 - (-4))^2 + (B.2 - m)^2)) ∧  -- AB is tangent
    ((B.1 - (-4))^2 + (B.2 - m)^2 = 36) :=  -- |AB|^2 = 6^2
  sorry

end NUMINAMATH_CALUDE_tangent_length_is_six_l775_77564


namespace NUMINAMATH_CALUDE_black_lambs_count_l775_77590

/-- The total number of lambs -/
def total_lambs : ℕ := 6048

/-- The number of white lambs -/
def white_lambs : ℕ := 193

/-- Theorem: The number of black lambs is 5855 -/
theorem black_lambs_count : total_lambs - white_lambs = 5855 := by
  sorry

end NUMINAMATH_CALUDE_black_lambs_count_l775_77590


namespace NUMINAMATH_CALUDE_marked_price_calculation_l775_77510

theorem marked_price_calculation (total_price : ℝ) (discount_percentage : ℝ) : 
  total_price = 50 →
  discount_percentage = 60 →
  ∃ (marked_price : ℝ), 
    marked_price = 62.50 ∧ 
    2 * marked_price * (1 - discount_percentage / 100) = total_price :=
by sorry

end NUMINAMATH_CALUDE_marked_price_calculation_l775_77510


namespace NUMINAMATH_CALUDE_even_function_theorem_l775_77537

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The functional equation satisfied by f -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y - 2 * x * y - 1

theorem even_function_theorem (f : ℝ → ℝ) 
    (heven : EvenFunction f) 
    (heq : SatisfiesFunctionalEquation f) : 
    ∀ x, f x = -x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_theorem_l775_77537


namespace NUMINAMATH_CALUDE_firefighter_ratio_l775_77516

theorem firefighter_ratio (doug kai eli : ℕ) : 
  doug = 20 →
  eli = kai / 2 →
  doug + kai + eli = 110 →
  kai / doug = 3 := by
sorry

end NUMINAMATH_CALUDE_firefighter_ratio_l775_77516


namespace NUMINAMATH_CALUDE_max_dominoes_formula_l775_77539

/-- Represents a grid of size 2n × 2n -/
structure Grid (n : ℕ+) where
  size : ℕ := 2 * n

/-- Represents a domino placement on the grid -/
structure DominoPlacement (n : ℕ+) where
  grid : Grid n
  num_dominoes : ℕ
  valid : Prop  -- This represents the validity of the placement according to the rules

/-- The maximum number of dominoes that can be placed on a 2n × 2n grid -/
def max_dominoes (n : ℕ+) : ℕ := n * (n + 1) / 2

/-- Theorem stating that the maximum number of dominoes is n(n+1)/2 -/
theorem max_dominoes_formula (n : ℕ+) :
  ∀ (p : DominoPlacement n), p.valid → p.num_dominoes ≤ max_dominoes n :=
sorry

end NUMINAMATH_CALUDE_max_dominoes_formula_l775_77539


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l775_77597

theorem smallest_prime_dividing_sum : ∃ p : Nat, 
  Prime p ∧ 
  p ∣ (4^15 + 7^12) ∧ 
  ∀ q : Nat, Prime q → q ∣ (4^15 + 7^12) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l775_77597


namespace NUMINAMATH_CALUDE_imaginary_product_implies_zero_l775_77573

/-- If the product of (1-ai) and i is a pure imaginary number, then a = 0 -/
theorem imaginary_product_implies_zero (a : ℝ) : 
  (∃ b : ℝ, (1 - a * Complex.I) * Complex.I = b * Complex.I) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_product_implies_zero_l775_77573


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l775_77580

theorem cube_root_equation_solution :
  ∃! x : ℝ, (2 - x / 2) ^ (1/3 : ℝ) = -3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l775_77580


namespace NUMINAMATH_CALUDE_egg_cost_calculation_l775_77509

def dozen : ℕ := 12

theorem egg_cost_calculation (total_cost : ℚ) (num_dozens : ℕ) 
  (h1 : total_cost = 18) 
  (h2 : num_dozens = 3) : 
  total_cost / (num_dozens * dozen) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_egg_cost_calculation_l775_77509


namespace NUMINAMATH_CALUDE_at_op_difference_zero_l775_77558

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - (x + y)

-- State the theorem
theorem at_op_difference_zero : at_op 7 4 - at_op 4 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_zero_l775_77558


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l775_77548

/-- The number of students playing soccer in the fifth grade at Parkway Elementary School -/
def students_playing_soccer (total_students : ℕ) (boys : ℕ) (boys_percentage : ℚ) (girls_not_playing : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of students playing soccer -/
theorem parkway_elementary_soccer 
  (total_students : ℕ) 
  (boys : ℕ) 
  (boys_percentage : ℚ) 
  (girls_not_playing : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : boys_percentage = 86 / 100)
  (h4 : girls_not_playing = 89) :
  students_playing_soccer total_students boys boys_percentage girls_not_playing = 250 := by
    sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l775_77548


namespace NUMINAMATH_CALUDE_tangent_sum_and_double_sum_l775_77524

theorem tangent_sum_and_double_sum (α β : Real) 
  (h1 : Real.tan α = 1/7) (h2 : Real.tan β = 1/3) : 
  Real.tan (α + β) = 1/2 ∧ Real.tan (α + 2*β) = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_sum_and_double_sum_l775_77524


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l775_77527

theorem min_value_a_plus_b (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a * b^2 = 4) :
  a + b ≥ 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ ≠ 0 ∧ b₀ ≠ 0 ∧ a₀ * b₀^2 = 4 ∧ a₀ + b₀ = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l775_77527


namespace NUMINAMATH_CALUDE_population_decrease_rate_l775_77536

theorem population_decrease_rate (initial_population : ℝ) (final_population : ℝ) (years : ℕ) 
  (h1 : initial_population = 8000)
  (h2 : final_population = 3920)
  (h3 : years = 2) :
  ∃ (rate : ℝ), initial_population * (1 - rate)^years = final_population ∧ rate = 0.3 := by
sorry

end NUMINAMATH_CALUDE_population_decrease_rate_l775_77536


namespace NUMINAMATH_CALUDE_least_value_x_minus_y_plus_z_l775_77592

theorem least_value_x_minus_y_plus_z (x y z : ℕ+) 
  (h : (3 : ℕ) * x = (4 : ℕ) * y ∧ (4 : ℕ) * y = (7 : ℕ) * z) : 
  (∀ a b c : ℕ+, (3 : ℕ) * a = (4 : ℕ) * b ∧ (4 : ℕ) * b = (7 : ℕ) * c → 
    (x : ℤ) - (y : ℤ) + (z : ℤ) ≤ (a : ℤ) - (b : ℤ) + (c : ℤ)) ∧
  (x : ℤ) - (y : ℤ) + (z : ℤ) = 19 :=
sorry

end NUMINAMATH_CALUDE_least_value_x_minus_y_plus_z_l775_77592


namespace NUMINAMATH_CALUDE_martin_fruits_l775_77518

/-- Represents the number of fruits Martin initially had -/
def initial_fruits : ℕ := 150

/-- Represents the number of oranges Martin has after eating -/
def remaining_oranges : ℕ := 50

/-- Represents the fraction of fruits Martin ate -/
def eaten_fraction : ℚ := 1/2

theorem martin_fruits :
  (initial_fruits : ℚ) * (1 - eaten_fraction) = remaining_oranges * 3 :=
sorry

end NUMINAMATH_CALUDE_martin_fruits_l775_77518


namespace NUMINAMATH_CALUDE_tangent_circles_exist_l775_77556

-- Define the circle k
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a ray
def Ray (origin : ℝ × ℝ) (direction : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (origin.1 + t * direction.1, origin.2 + t * direction.2)}

-- Define tangency between a circle and a ray
def IsTangent (c : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ c ∧ p ∈ r ∧ ∀ q : ℝ × ℝ, q ∈ c ∩ r → q = p

-- Main theorem
theorem tangent_circles_exist
  (k : Set (ℝ × ℝ))
  (O : ℝ × ℝ)
  (r : ℝ)
  (A : ℝ × ℝ)
  (e f : Set (ℝ × ℝ))
  (hk : k = Circle O r)
  (hA : A ∈ k)
  (he : e = Ray A (1, 0))  -- Arbitrary direction for e
  (hf : f = Ray A (0, 1))  -- Arbitrary direction for f
  (hef : e ≠ f) :
  ∃ c : Set (ℝ × ℝ), ∃ center : ℝ × ℝ, ∃ radius : ℝ,
    c = Circle center radius ∧
    IsTangent c k ∧
    IsTangent c e ∧
    IsTangent c f :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_exist_l775_77556


namespace NUMINAMATH_CALUDE_total_exercise_hours_l775_77563

/-- Exercise duration in minutes for each person -/
def natasha_minutes : ℕ := 30 * 7
def esteban_minutes : ℕ := 10 * 9
def charlotte_minutes : ℕ := 20 + 45 + 70 + 100

/-- Total exercise duration in minutes -/
def total_minutes : ℕ := natasha_minutes + esteban_minutes + charlotte_minutes

/-- Conversion factor from minutes to hours -/
def minutes_per_hour : ℕ := 60

/-- Total exercise duration in hours -/
def total_hours : ℚ := total_minutes / minutes_per_hour

/-- Theorem: The total hours of exercise for all three individuals is 8.92 hours -/
theorem total_exercise_hours : total_hours = 892 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_exercise_hours_l775_77563


namespace NUMINAMATH_CALUDE_f_root_exists_l775_77570

noncomputable def f (x : ℝ) := Real.log x / Real.log 3 + x

theorem f_root_exists : ∃ x ∈ Set.Ioo 3 4, f x - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_root_exists_l775_77570


namespace NUMINAMATH_CALUDE_remainder_difference_l775_77553

theorem remainder_difference (d r : ℕ) : 
  d > 1 → 
  2023 % d = r → 
  2459 % d = r → 
  3571 % d = r → 
  d - r = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_difference_l775_77553


namespace NUMINAMATH_CALUDE_ball_probability_l775_77552

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 60)
  (h_white : white = 22)
  (h_green : green = 18)
  (h_yellow : yellow = 17)
  (h_red : red = 3)
  (h_purple : purple = 1)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 14 / 15 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l775_77552


namespace NUMINAMATH_CALUDE_supplemental_tanks_needed_l775_77567

def total_diving_time : ℕ := 8
def primary_tank_duration : ℕ := 2
def supplemental_tank_duration : ℕ := 1

theorem supplemental_tanks_needed :
  (total_diving_time - primary_tank_duration) / supplemental_tank_duration = 6 :=
by sorry

end NUMINAMATH_CALUDE_supplemental_tanks_needed_l775_77567


namespace NUMINAMATH_CALUDE_percentage_50_59_range_l775_77582

/-- Represents the frequency distribution of scores in Mrs. Lopez's geometry class -/
structure ScoreDistribution :=
  (score_90_100 : Nat)
  (score_80_89 : Nat)
  (score_70_79 : Nat)
  (score_60_69 : Nat)
  (score_50_59 : Nat)
  (score_below_50 : Nat)

/-- Calculates the total number of students -/
def totalStudents (dist : ScoreDistribution) : Nat :=
  dist.score_90_100 + dist.score_80_89 + dist.score_70_79 + 
  dist.score_60_69 + dist.score_50_59 + dist.score_below_50

/-- The actual score distribution in Mrs. Lopez's class -/
def lopezClassDist : ScoreDistribution :=
  { score_90_100 := 3
  , score_80_89 := 6
  , score_70_79 := 8
  , score_60_69 := 4
  , score_50_59 := 3
  , score_below_50 := 4
  }

/-- Theorem stating that the percentage of students who scored in the 50%-59% range is 3/28 * 100% -/
theorem percentage_50_59_range (dist : ScoreDistribution) :
  dist = lopezClassDist →
  (dist.score_50_59 : Rat) / (totalStudents dist : Rat) = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_50_59_range_l775_77582


namespace NUMINAMATH_CALUDE_log_function_through_point_l775_77540

/-- Given a logarithmic function that passes through the point (4, 2), prove that its base is 2 -/
theorem log_function_through_point (f : ℝ → ℝ) (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x > 0, f x = Real.log x / Real.log a) 
  (h4 : f 4 = 2) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_function_through_point_l775_77540


namespace NUMINAMATH_CALUDE_ed_remaining_money_l775_77508

-- Define the hotel rates
def night_rate : ℝ := 1.50
def morning_rate : ℝ := 2

-- Define Ed's initial money
def initial_money : ℝ := 80

-- Define the duration of stay
def night_hours : ℝ := 6
def morning_hours : ℝ := 4

-- Theorem to prove
theorem ed_remaining_money :
  let night_cost := night_rate * night_hours
  let morning_cost := morning_rate * morning_hours
  let total_cost := night_cost + morning_cost
  let remaining_money := initial_money - total_cost
  remaining_money = 63 := by sorry

end NUMINAMATH_CALUDE_ed_remaining_money_l775_77508


namespace NUMINAMATH_CALUDE_polygon_diagonals_l775_77598

theorem polygon_diagonals (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 150 → (n - 2) * 180 = n * interior_angle → n - 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l775_77598


namespace NUMINAMATH_CALUDE_quadratic_polynomial_unique_l775_77584

theorem quadratic_polynomial_unique (q : ℝ → ℝ) : 
  (∀ x, q x = 2 * x^2 - 6 * x - 36) →
  q (-3) = 0 ∧ q 6 = 0 ∧ q 2 = -40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_unique_l775_77584


namespace NUMINAMATH_CALUDE_students_not_in_biology_or_chemistry_l775_77586

theorem students_not_in_biology_or_chemistry
  (total : ℕ)
  (biology_percent : ℚ)
  (chemistry_percent : ℚ)
  (both_percent : ℚ)
  (h_total : total = 880)
  (h_biology : biology_percent = 40 / 100)
  (h_chemistry : chemistry_percent = 30 / 100)
  (h_both : both_percent = 10 / 100) :
  total - (total * biology_percent + total * chemistry_percent - total * both_percent).floor = 352 :=
by sorry

end NUMINAMATH_CALUDE_students_not_in_biology_or_chemistry_l775_77586


namespace NUMINAMATH_CALUDE_stratified_sampling_most_representative_l775_77501

-- Define a type for sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

-- Define a type for high school grades
inductive HighSchoolGrade
  | First
  | Second
  | Third

-- Define a population with subgroups
structure Population where
  subgroups : List HighSchoolGrade

-- Define a characteristic being studied
structure Characteristic where
  name : String
  hasSignificantDifferences : Bool

-- Define a function to determine the most representative sampling method
def mostRepresentativeSamplingMethod (pop : Population) (char : Characteristic) : SamplingMethod :=
  if char.hasSignificantDifferences then
    SamplingMethod.Stratified
  else
    SamplingMethod.SimpleRandom

-- Theorem statement
theorem stratified_sampling_most_representative 
  (pop : Population) 
  (char : Characteristic) 
  (h1 : pop.subgroups = [HighSchoolGrade.First, HighSchoolGrade.Second, HighSchoolGrade.Third]) 
  (h2 : char.name = "Understanding of Jingma") 
  (h3 : char.hasSignificantDifferences = true) :
  mostRepresentativeSamplingMethod pop char = SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_representative_l775_77501


namespace NUMINAMATH_CALUDE_decimal_sum_l775_77519

theorem decimal_sum : (0.08 : ℚ) + (0.003 : ℚ) + (0.0070 : ℚ) = (0.09 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l775_77519


namespace NUMINAMATH_CALUDE_problem_statement_l775_77572

theorem problem_statement (x : ℕ) (h : x = 3) : x + x * (Nat.factorial x)^x = 651 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l775_77572


namespace NUMINAMATH_CALUDE_apples_bought_proof_l775_77513

/-- The price of an orange in reals -/
def orange_price : ℝ := 2

/-- The price of an apple in reals -/
def apple_price : ℝ := 3

/-- An orange costs the same as half an apple plus half a real -/
axiom orange_price_relation : orange_price = apple_price / 2 + 1 / 2

/-- One-third of an apple costs the same as one-quarter of an orange plus half a real -/
axiom apple_price_relation : apple_price / 3 = orange_price / 4 + 1 / 2

/-- The number of apples that can be bought with the value of 5 oranges plus 5 reals -/
def apples_bought : ℕ := 5

theorem apples_bought_proof : 
  (5 * orange_price + 5) / apple_price = apples_bought := by sorry

end NUMINAMATH_CALUDE_apples_bought_proof_l775_77513


namespace NUMINAMATH_CALUDE_curves_intersection_l775_77554

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 2 * x^3 + x^2 - 5 * x + 2

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 4

/-- Intersection points of the two curves -/
def intersection_points : Set (ℝ × ℝ) := {(-1, -7), (3, 41)}

theorem curves_intersection :
  ∀ p : ℝ × ℝ, p ∈ intersection_points ↔ 
    (curve1 p.1 = curve2 p.1 ∧ p.2 = curve1 p.1) ∧
    ∀ x : ℝ, curve1 x = curve2 x → x = p.1 ∨ x = (if p.1 = -1 then 3 else -1) := by
  sorry

end NUMINAMATH_CALUDE_curves_intersection_l775_77554


namespace NUMINAMATH_CALUDE_line_intersection_problem_l775_77557

/-- The problem statement as a theorem -/
theorem line_intersection_problem :
  ∃ (m b : ℝ),
    b ≠ 0 ∧
    (∃! k, ∃ y₁ y₂, 
      y₁ = k^2 + 4*k + 4 ∧
      y₂ = m*k + b ∧
      |y₁ - y₂| = 6) ∧
    (8 = m*2 + b) ∧
    m = 2 * Real.sqrt 6 ∧
    b = 8 - 4 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_problem_l775_77557


namespace NUMINAMATH_CALUDE_largest_number_l775_77546

theorem largest_number (x y z : ℝ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y + z = 82) (h4 : z - y = 10) (h5 : y - x = 4) :
  z = 106 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l775_77546


namespace NUMINAMATH_CALUDE_weekly_income_proof_l775_77503

/-- Proves that a weekly income of $500 satisfies the given conditions -/
theorem weekly_income_proof (income : ℝ) : 
  income - 0.2 * income - 55 = 345 → income = 500 := by
  sorry

end NUMINAMATH_CALUDE_weekly_income_proof_l775_77503


namespace NUMINAMATH_CALUDE_inequality_solution_set_l775_77571

theorem inequality_solution_set (x : ℝ) : (2 * x - 1) / (3 * x + 1) > 1 ↔ -2 < x ∧ x < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l775_77571


namespace NUMINAMATH_CALUDE_indeterminate_relation_product_and_means_l775_77577

/-- Given two positive real numbers, their arithmetic mean, and their geometric mean,
    the relationship between the product of the numbers and the product of their means
    cannot be determined. -/
theorem indeterminate_relation_product_and_means (a b : ℝ) (A G : ℝ) 
    (ha : 0 < a) (hb : 0 < b)
    (hA : A = (a + b) / 2)
    (hG : G = Real.sqrt (a * b)) :
    ¬ ∀ (R : ℝ → ℝ → Prop), R (a * b) (A * G) ∨ R (A * G) (a * b) := by
  sorry

end NUMINAMATH_CALUDE_indeterminate_relation_product_and_means_l775_77577


namespace NUMINAMATH_CALUDE_shortest_chord_through_M_l775_77550

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 2*y + 10 = 0

-- Define point M
def point_M : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Theorem statement
theorem shortest_chord_through_M :
  ∀ (l : ℝ × ℝ → Prop),
    (∀ x y, l (x, y) ↔ line_equation x y) →
    (l point_M) →
    (∀ other_line : ℝ × ℝ → Prop,
      (other_line point_M) →
      (∃ p, circle_equation p.1 p.2 ∧ other_line p) →
      (∃ p q : ℝ × ℝ, 
        p ≠ q ∧ 
        circle_equation p.1 p.2 ∧ circle_equation q.1 q.2 ∧ 
        l p ∧ l q ∧
        other_line p ∧ other_line q →
        (p.1 - q.1)^2 + (p.2 - q.2)^2 ≤ 
        (p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
  sorry

end NUMINAMATH_CALUDE_shortest_chord_through_M_l775_77550


namespace NUMINAMATH_CALUDE_sin_arctan_equation_l775_77544

theorem sin_arctan_equation (y : ℝ) (hy : y > 0) 
  (h : Real.sin (Real.arctan y) = 1 / (2 * y)) : 
  y^2 = (1 + Real.sqrt 17) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_arctan_equation_l775_77544


namespace NUMINAMATH_CALUDE_terrell_hike_distance_l775_77578

/-- The total distance hiked by Terrell over two days -/
def total_distance (saturday_distance sunday_distance : ℝ) : ℝ :=
  saturday_distance + sunday_distance

/-- Theorem stating that Terrell's total hiking distance is 9.8 miles -/
theorem terrell_hike_distance :
  total_distance 8.2 1.6 = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_terrell_hike_distance_l775_77578


namespace NUMINAMATH_CALUDE_equal_area_line_equation_l775_77595

-- Define the circle arrangement
def circle_arrangement : List (ℝ × ℝ) :=
  [(1, 1), (3, 1), (5, 1), (7, 1), (1, 3), (3, 3), (5, 3), (1, 5), (3, 5), (5, 5)]

-- Define the line with slope 2
def line_slope : ℝ := 2

-- Define the function to check if a line divides the area equally
def divides_area_equally (a b c : ℤ) : Prop := sorry

-- Define the function to check if three integers are coprime
def are_coprime (a b c : ℤ) : Prop := sorry

-- Main theorem
theorem equal_area_line_equation :
  ∃ (a b c : ℤ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    are_coprime a b c ∧
    divides_area_equally a b c ∧
    a^2 + b^2 + c^2 = 86 :=
  sorry

end NUMINAMATH_CALUDE_equal_area_line_equation_l775_77595


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l775_77588

/-- Theorem: Given a cuboid with edges x cm, 5 cm, and 6 cm, and a volume of 180 cm³,
    the length of the first edge (x) is 6 cm. -/
theorem cuboid_edge_length (x : ℝ) : x * 5 * 6 = 180 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l775_77588


namespace NUMINAMATH_CALUDE_remainder_problem_l775_77594

theorem remainder_problem (n : ℤ) (h : n % 9 = 4) : (4 * n - 3) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l775_77594


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l775_77531

theorem probability_three_white_balls (total_balls : ℕ) (white_balls : ℕ) (drawn_balls : ℕ) :
  total_balls = 15 →
  white_balls = 7 →
  drawn_balls = 3 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l775_77531


namespace NUMINAMATH_CALUDE_sequence_property_l775_77528

theorem sequence_property (u : ℕ → ℤ) : 
  (∀ n m : ℕ, u (n * m) = u n + u m) → 
  (∀ n : ℕ, u n = 0) := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l775_77528


namespace NUMINAMATH_CALUDE_lance_workdays_per_week_l775_77506

/-- Given Lance's work schedule and earnings, prove the number of workdays per week -/
theorem lance_workdays_per_week 
  (total_weekly_hours : ℕ) 
  (hourly_wage : ℚ) 
  (daily_earnings : ℚ) 
  (h1 : total_weekly_hours = 35)
  (h2 : hourly_wage = 9)
  (h3 : daily_earnings = 63)
  (h4 : ∃ (daily_hours : ℚ), daily_hours * hourly_wage = daily_earnings ∧ 
        daily_hours * (total_weekly_hours / daily_hours) = total_weekly_hours) :
  total_weekly_hours / (daily_earnings / hourly_wage) = 5 := by
  sorry

end NUMINAMATH_CALUDE_lance_workdays_per_week_l775_77506


namespace NUMINAMATH_CALUDE_area_of_EFGH_l775_77523

/-- The length of the shorter side of each smaller rectangle -/
def short_side : ℝ := 3

/-- The number of smaller rectangles used to form EFGH -/
def num_rectangles : ℕ := 4

/-- The number of rectangles placed horizontally -/
def horizontal_rectangles : ℕ := 2

/-- The number of rectangles placed vertically -/
def vertical_rectangles : ℕ := 2

/-- The ratio of the longer side to the shorter side of each smaller rectangle -/
def side_ratio : ℝ := 2

theorem area_of_EFGH : 
  let longer_side := short_side * side_ratio
  let width := short_side * horizontal_rectangles
  let length := longer_side * vertical_rectangles
  width * length = 72 := by sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l775_77523


namespace NUMINAMATH_CALUDE_correlation_properties_l775_77599

/-- The linear correlation coefficient between two variables -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- The strength of linear correlation between two variables -/
def correlation_strength (r : ℝ) : ℝ := sorry

theorem correlation_properties (x y : ℝ → ℝ) (r : ℝ) 
  (h : r = correlation_coefficient x y) :
  (r > 0 → ∀ t₁ t₂, t₁ < t₂ → x t₁ < x t₂ → y t₁ < y t₂) ∧ 
  (∀ ε > 0, ∃ δ > 0, |r| > 1 - δ → correlation_strength r > 1 - ε) ∧
  (r = 1 ∨ r = -1 → ∃ a b : ℝ, ∀ t, y t = a * x t + b) :=
sorry

end NUMINAMATH_CALUDE_correlation_properties_l775_77599


namespace NUMINAMATH_CALUDE_lukes_coin_piles_l775_77529

theorem lukes_coin_piles (num_quarter_piles : ℕ) : 
  (∃ (num_dime_piles : ℕ), 
    num_quarter_piles = num_dime_piles ∧ 
    3 * num_quarter_piles + 3 * num_dime_piles = 30) → 
  num_quarter_piles = 5 := by
sorry

end NUMINAMATH_CALUDE_lukes_coin_piles_l775_77529


namespace NUMINAMATH_CALUDE_ellipse_y_axis_iff_m_greater_n_l775_77561

/-- The equation of an ellipse with foci on the y-axis -/
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  ∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

/-- The condition for m and n -/
def m_greater_n (m n : ℝ) : Prop :=
  m > n ∧ n > 0

theorem ellipse_y_axis_iff_m_greater_n (m n : ℝ) :
  is_ellipse_y_axis m n ↔ m_greater_n m n :=
sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_iff_m_greater_n_l775_77561


namespace NUMINAMATH_CALUDE_photo_arrangements_eq_24_l775_77512

/-- The number of different arrangements for a teacher and two boys and two girls standing in a row,
    with the requirement that the two girls must stand together and the teacher cannot stand at either end. -/
def photo_arrangements : ℕ :=
  let n_people : ℕ := 5
  let n_boys : ℕ := 2
  let n_girls : ℕ := 2
  let n_teacher : ℕ := 1
  let girls_together : ℕ := 1  -- Treat the two girls as one unit
  let teacher_positions : ℕ := n_people - 2  -- Teacher can't be at either end
  Nat.factorial n_people / (Nat.factorial n_boys * Nat.factorial girls_together * Nat.factorial n_teacher)
    * teacher_positions

theorem photo_arrangements_eq_24 : photo_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_eq_24_l775_77512


namespace NUMINAMATH_CALUDE_num_al_sandwiches_l775_77534

/-- Represents the number of different types of bread available at the deli. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available at the deli. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available at the deli. -/
def num_cheeses : ℕ := 5

/-- Represents whether ham is available at the deli. -/
def ham_available : Prop := True

/-- Represents whether turkey is available at the deli. -/
def turkey_available : Prop := True

/-- Represents whether cheddar cheese is available at the deli. -/
def cheddar_available : Prop := True

/-- Represents whether rye bread is available at the deli. -/
def rye_available : Prop := True

/-- Represents the number of sandwiches with ham and cheddar cheese combination. -/
def ham_cheddar_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and turkey combination. -/
def rye_turkey_combos : ℕ := num_cheeses

/-- Theorem stating the number of different sandwiches Al could order. -/
theorem num_al_sandwiches : 
  num_breads * num_meats * num_cheeses - ham_cheddar_combos - rye_turkey_combos = 165 := by
  sorry

end NUMINAMATH_CALUDE_num_al_sandwiches_l775_77534


namespace NUMINAMATH_CALUDE_equation_solution_l775_77560

theorem equation_solution : ∃ y : ℝ, (4 * y - 2) / (5 * y - 5) = 3 / 4 ∧ y = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l775_77560


namespace NUMINAMATH_CALUDE_farmer_milk_production_l775_77559

/-- Calculates the total milk production for a farmer in a week -/
def total_milk_production (num_cows : ℕ) (milk_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  num_cows * milk_per_day * days_in_week

/-- Proves that a farmer with 52 cows, each producing 5 liters of milk per day,
    will get 1820 liters of milk in a week (7 days) -/
theorem farmer_milk_production :
  total_milk_production 52 5 7 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_farmer_milk_production_l775_77559


namespace NUMINAMATH_CALUDE_expression_evaluation_l775_77511

theorem expression_evaluation (x y : ℚ) 
  (hx : x = -1) 
  (hy : y = -1/2) : 
  4*x*y + (2*x^2 + 5*x*y - y^2) - 2*(x^2 + 3*x*y) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l775_77511


namespace NUMINAMATH_CALUDE_polynomial_as_sum_of_squares_l775_77555

theorem polynomial_as_sum_of_squares (x : ℝ) :
  x^4 - 2*x^3 + 6*x^2 - 2*x + 1 = (x^2 - x)^2 + (x - 1)^2 + (2*x)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_as_sum_of_squares_l775_77555


namespace NUMINAMATH_CALUDE_f_properties_l775_77593

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * Real.pi - x) - Real.cos (Real.pi / 2 + x) + 1

theorem f_properties :
  (∀ x, -1 ≤ f x ∧ f x ≤ 3) ∧
  (∀ k : ℤ, ∀ x, -5 * Real.pi / 6 + 2 * k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + 2 * k * Real.pi → 
    ∀ y, x ≤ y → f x ≤ f y) ∧
  (∀ α, f α = 13 / 5 → Real.pi / 6 < α ∧ α < 2 * Real.pi / 3 → 
    Real.cos (2 * α) = (7 - 24 * Real.sqrt 3) / 50) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l775_77593


namespace NUMINAMATH_CALUDE_min_value_S_l775_77521

/-- The minimum value of (x-a)^2 + (ln x - a)^2 is 1/2, where x > 0 and a is real. -/
theorem min_value_S (x a : ℝ) (hx : x > 0) : 
  ∃ (min : ℝ), min = (1/2 : ℝ) ∧ ∀ y > 0, (y - a)^2 + (Real.log y - a)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_S_l775_77521


namespace NUMINAMATH_CALUDE_bottle_caps_given_l775_77530

theorem bottle_caps_given (initial_caps : Real) (remaining_caps : Real) 
  (h1 : initial_caps = 7.0)
  (h2 : remaining_caps = 5.0) :
  initial_caps - remaining_caps = 2.0 := by
sorry

end NUMINAMATH_CALUDE_bottle_caps_given_l775_77530


namespace NUMINAMATH_CALUDE_vector_rotation_angle_l775_77576

theorem vector_rotation_angle (p : ℂ) (α : ℝ) (h_p : p ≠ 0) :
  p + p * Complex.exp (2 * α * Complex.I) = p * Complex.exp (α * Complex.I) →
  α = π / 3 + 2 * π * ↑k ∨ α = -π / 3 + 2 * π * ↑n :=
by sorry

end NUMINAMATH_CALUDE_vector_rotation_angle_l775_77576


namespace NUMINAMATH_CALUDE_no_cyclic_knight_tour_5x5_l775_77515

/-- Represents a chessboard --/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a knight's move --/
inductive KnightMove
  | move : Nat → Nat → KnightMove

/-- Represents a tour on the chessboard --/
structure Tour :=
  (moves : List KnightMove)
  (cyclic : Bool)

/-- Defines a valid knight's move --/
def isValidKnightMove (m : KnightMove) : Prop :=
  match m with
  | KnightMove.move x y => (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2)

/-- Defines if a tour visits each square exactly once --/
def visitsEachSquareOnce (t : Tour) (b : Chessboard) : Prop :=
  t.moves.length = b.rows * b.cols

/-- Theorem: It's impossible for a knight to make a cyclic tour on a 5x5 chessboard
    visiting each square exactly once --/
theorem no_cyclic_knight_tour_5x5 :
  ∀ (t : Tour),
    t.cyclic →
    (∀ (m : KnightMove), m ∈ t.moves → isValidKnightMove m) →
    visitsEachSquareOnce t (Chessboard.mk 5 5) →
    False :=
sorry

end NUMINAMATH_CALUDE_no_cyclic_knight_tour_5x5_l775_77515


namespace NUMINAMATH_CALUDE_bus_trip_distance_l775_77502

/-- Given a bus trip with specific conditions, prove that the trip distance is 550 miles. -/
theorem bus_trip_distance (speed : ℝ) (distance : ℝ) : 
  speed = 50 →  -- The actual average speed is 50 mph
  distance / speed = distance / (speed + 5) + 1 →  -- The trip would take 1 hour less if speed increased by 5 mph
  distance = 550 := by
sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l775_77502


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l775_77565

/-- Given a quadratic inequality ax² + bx + c < 0 with solution set {x | x < -2 ∨ x > -1/2},
    prove that the solution set for ax² - bx + c > 0 is {x | 1/2 < x ∧ x < 2} -/
theorem quadratic_inequality_solution_set
  (a b c : ℝ)
  (h : ∀ x : ℝ, (a * x^2 + b * x + c < 0) ↔ (x < -2 ∨ x > -(1/2))) :
  ∀ x : ℝ, (a * x^2 - b * x + c > 0) ↔ (1/2 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l775_77565


namespace NUMINAMATH_CALUDE_x_values_l775_77541

theorem x_values (x : ℝ) : x ∈ ({1, 2, x^2} : Set ℝ) → x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l775_77541


namespace NUMINAMATH_CALUDE_cubic_identity_fraction_l775_77579

theorem cubic_identity_fraction (x y z : ℝ) :
  ((x - y)^3 + (y - z)^3 + (z - x)^3) / (15 * (x - y) * (y - z) * (z - x)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_fraction_l775_77579


namespace NUMINAMATH_CALUDE_markup_discount_profit_l775_77575

/-- Given a markup percentage and a discount percentage, calculate the profit percentage -/
def profit_percentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem stating that a 75% markup followed by a 30% discount results in a 22.5% profit -/
theorem markup_discount_profit : profit_percentage 0.75 0.3 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_markup_discount_profit_l775_77575


namespace NUMINAMATH_CALUDE_polynomial_factorization_l775_77583

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 12) * (x^2 + 6*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l775_77583


namespace NUMINAMATH_CALUDE_inequality_proof_l775_77505

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/2 + z/3 ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l775_77505


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l775_77532

def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {y | y^2 - 2*y - 3 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l775_77532


namespace NUMINAMATH_CALUDE_line_segment_length_l775_77591

structure Line where
  points : Fin 5 → ℝ
  consecutive : ∀ i : Fin 4, points i < points (Fin.succ i)

def Line.segment (l : Line) (i j : Fin 5) : ℝ :=
  |l.points j - l.points i|

theorem line_segment_length (l : Line) 
  (h1 : l.segment 1 2 = 3 * l.segment 2 3)
  (h2 : l.segment 3 4 = 7)
  (h3 : l.segment 0 1 = 5)
  (h4 : l.segment 0 2 = 11) :
  l.segment 0 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l775_77591


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l775_77538

/-- Given a quadratic function y = ax² + bx + c, if (2, y₁) and (-2, y₂) are points on this function
    and y₁ - y₂ = -16, then b = -4. -/
theorem quadratic_coefficient (a b c y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = -16 →
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l775_77538


namespace NUMINAMATH_CALUDE_leaves_per_sub_branch_l775_77514

/-- Given a farm with trees, branches, and sub-branches, calculate the number of leaves per sub-branch. -/
theorem leaves_per_sub_branch 
  (num_trees : ℕ) 
  (branches_per_tree : ℕ) 
  (sub_branches_per_branch : ℕ) 
  (total_leaves : ℕ) 
  (h1 : num_trees = 4)
  (h2 : branches_per_tree = 10)
  (h3 : sub_branches_per_branch = 40)
  (h4 : total_leaves = 96000) :
  total_leaves / (num_trees * branches_per_tree * sub_branches_per_branch) = 60 := by
  sorry

#check leaves_per_sub_branch

end NUMINAMATH_CALUDE_leaves_per_sub_branch_l775_77514


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l775_77562

theorem tax_free_items_cost 
  (total_paid : ℝ) 
  (sales_tax : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_paid = 30)
  (h2 : sales_tax = 1.28)
  (h3 : tax_rate = 0.08) : 
  total_paid - sales_tax / tax_rate = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l775_77562


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l775_77542

theorem isosceles_triangle_base_angle (base_angle : ℝ) (top_angle : ℝ) : 
  -- The triangle is isosceles
  -- The top angle is 20° more than twice the base angle
  top_angle = 2 * base_angle + 20 →
  -- The sum of angles in a triangle is 180°
  base_angle + base_angle + top_angle = 180 →
  -- The base angle is 40°
  base_angle = 40 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l775_77542


namespace NUMINAMATH_CALUDE_division_decomposition_l775_77574

theorem division_decomposition : (36 : ℕ) / 3 = (30 / 3) + (6 / 3) := by sorry

end NUMINAMATH_CALUDE_division_decomposition_l775_77574


namespace NUMINAMATH_CALUDE_expr1_simplification_expr1_evaluation_expr2_simplification_expr2_evaluation_l775_77500

-- Define variables
variable (x y : ℝ)

-- First expression
def expr1 (x y : ℝ) : ℝ := 3*x^2*y - (2*x*y^2 - 2*(x*y - 1.5*x^2*y) + x*y) + 3*x*y^2

-- Second expression
def expr2 (x y : ℝ) : ℝ := (2*x + 3*y) - 4*y - (3*x - 2*y)

-- Theorem for the first expression
theorem expr1_simplification : 
  expr1 x y = x*y^2 + x*y := by sorry

-- Theorem for the evaluation of the first expression
theorem expr1_evaluation : 
  expr1 (-3) (-2) = -6 := by sorry

-- Theorem for the second expression
theorem expr2_simplification :
  expr2 x y = -x + y := by sorry

-- Theorem for the evaluation of the second expression
theorem expr2_evaluation :
  expr2 (-3) 2 = 5 := by sorry

end NUMINAMATH_CALUDE_expr1_simplification_expr1_evaluation_expr2_simplification_expr2_evaluation_l775_77500


namespace NUMINAMATH_CALUDE_not_prime_5n_plus_3_l775_77566

theorem not_prime_5n_plus_3 (n : ℕ) (k m : ℤ) 
  (h1 : 2 * n + 1 = k^2) 
  (h2 : 3 * n + 1 = m^2) : 
  ¬ Nat.Prime (5 * n + 3) := by
sorry

end NUMINAMATH_CALUDE_not_prime_5n_plus_3_l775_77566


namespace NUMINAMATH_CALUDE_other_number_proof_l775_77549

theorem other_number_proof (a b : ℕ) (h1 : a + b = 62) (h2 : a = 27) : b = 35 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l775_77549


namespace NUMINAMATH_CALUDE_system_solution_l775_77581

theorem system_solution (x y z : ℝ) 
  (eq1 : x * y = 5 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 5 * y - 3 * z)
  (eq3 : x * z = 18 - 2 * x - 5 * z)
  (pos_x : x > 0) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l775_77581


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l775_77568

theorem sum_of_four_consecutive_integers (S : ℤ) :
  (∃ n : ℤ, S = n + (n + 1) + (n + 2) + (n + 3)) ↔ (S - 6) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l775_77568


namespace NUMINAMATH_CALUDE_clubsuit_calculation_l775_77525

-- Define the new operation
def clubsuit (x y : ℤ) : ℤ := x^2 - y^2

-- Theorem statement
theorem clubsuit_calculation : clubsuit 5 (clubsuit 6 7) = -144 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_calculation_l775_77525


namespace NUMINAMATH_CALUDE_sum_of_cubes_product_l775_77545

theorem sum_of_cubes_product : ∃ a b : ℤ, a^3 + b^3 = 35 ∧ a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_product_l775_77545


namespace NUMINAMATH_CALUDE_double_elim_advantage_l775_77587

-- Define the probability of team A winning against other teams
variable (p : ℝ)

-- Define the conditions
def knockout_prob := p^2
def double_elim_prob := p^3 * (3 - 2*p)

-- State the theorem
theorem double_elim_advantage (h1 : 1/2 < p) (h2 : p < 1) :
  knockout_prob p < double_elim_prob p :=
sorry

end NUMINAMATH_CALUDE_double_elim_advantage_l775_77587


namespace NUMINAMATH_CALUDE_intersection_line_canonical_equations_l775_77535

/-- Given two planes in 3D space, this theorem states that their line of intersection
    can be represented by specific canonical equations. -/
theorem intersection_line_canonical_equations
  (plane1 : x + y + z = 2)
  (plane2 : x - y - 2*z = -2)
  : ∃ (t : ℝ), x = -t ∧ y = 3*t + 2 ∧ z = -2*t :=
sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_equations_l775_77535


namespace NUMINAMATH_CALUDE_fraction_simplification_l775_77543

theorem fraction_simplification : 1 / (Real.sqrt 3 - 2) = -Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l775_77543


namespace NUMINAMATH_CALUDE_min_value_d_l775_77526

/-- Given positive integers a, b, c, and d where a < b < c < d, and a system of equations
    with exactly one solution, the minimum value of d is 602. -/
theorem min_value_d (a b c d : ℕ+) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : ∃! (x y : ℝ), 3 * x + y = 3004 ∧ y = |x - a| + |x - b| + |x - c| + |x - d|) :
  d = 602 := by
  sorry

end NUMINAMATH_CALUDE_min_value_d_l775_77526


namespace NUMINAMATH_CALUDE_dogs_in_park_l775_77533

/-- The number of dogs in the park -/
def D : ℕ := 88

/-- The number of dogs running -/
def running : ℕ := 12

/-- The number of dogs doing nothing -/
def doing_nothing : ℕ := 10

theorem dogs_in_park :
  D = running + D / 2 + D / 4 + doing_nothing :=
sorry


end NUMINAMATH_CALUDE_dogs_in_park_l775_77533


namespace NUMINAMATH_CALUDE_gcd_1443_999_l775_77520

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by sorry

end NUMINAMATH_CALUDE_gcd_1443_999_l775_77520


namespace NUMINAMATH_CALUDE_count_true_propositions_l775_77507

/-- The number of true propositions among the original, converse, inverse, and contrapositive
    of the statement "For real numbers a, b, c, and d, if a=b and c=d, then a+c=b+d" -/
def num_true_propositions : ℕ := 2

/-- The original proposition -/
def original_prop (a b c d : ℝ) : Prop :=
  (a = b ∧ c = d) → (a + c = b + d)

theorem count_true_propositions :
  (∀ a b c d : ℝ, original_prop a b c d) ∧
  (∃ a b c d : ℝ, ¬(a + c = b + d → a = b ∧ c = d)) ∧
  num_true_propositions = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_true_propositions_l775_77507


namespace NUMINAMATH_CALUDE_unique_remainder_sum_equal_l775_77547

/-- The sum of distinct remainders when dividing a natural number by all smaller positive natural numbers -/
def sumDistinctRemainders (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => n % (k + 1))

/-- Theorem stating that 3 is the only natural number equal to the sum of its distinct remainders -/
theorem unique_remainder_sum_equal : ∀ n : ℕ, n > 0 → (sumDistinctRemainders n = n ↔ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_sum_equal_l775_77547


namespace NUMINAMATH_CALUDE_probability_two_copresidents_selected_l775_77585

def choose (n k : ℕ) : ℕ := Nat.choose n k

def prob_copresident_selected (n : ℕ) : ℚ :=
  (choose (n - 2) 2 : ℚ) / (choose n 4 : ℚ)

def total_probability : ℚ :=
  (1 : ℚ) / 3 * (prob_copresident_selected 6 + prob_copresident_selected 8 + prob_copresident_selected 9)

theorem probability_two_copresidents_selected : total_probability = 82 / 315 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_copresidents_selected_l775_77585


namespace NUMINAMATH_CALUDE_missing_number_l775_77517

theorem missing_number (x : ℤ) : x + 64 = 16 → x = -48 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_l775_77517


namespace NUMINAMATH_CALUDE_julia_tag_total_l775_77569

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 16

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 14

/-- The total number of kids Julia played tag with over two days -/
def total_kids : ℕ := monday_kids + tuesday_kids

theorem julia_tag_total : total_kids = 30 := by sorry

end NUMINAMATH_CALUDE_julia_tag_total_l775_77569
