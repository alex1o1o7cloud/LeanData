import Mathlib

namespace NUMINAMATH_CALUDE_N_is_composite_l1030_103048

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Nat.Prime N := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l1030_103048


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_21_with_cube_root_between_9_and_9_1_l1030_103045

theorem unique_integer_divisible_by_21_with_cube_root_between_9_and_9_1 :
  ∃! n : ℕ+, (21 ∣ n) ∧ (9 < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 9.1) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_21_with_cube_root_between_9_and_9_1_l1030_103045


namespace NUMINAMATH_CALUDE_sum_of_bounds_l1030_103022

def U : Type := ℝ

def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

def complement_A : Set ℝ := {x | x > 4 ∨ x < 3}

theorem sum_of_bounds (a b : ℝ) :
  A a b = (Set.univ \ complement_A) → a + b = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_bounds_l1030_103022


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1030_103000

theorem arithmetic_sequence_problem :
  ∃ (a b c : ℝ), 
    (a > b ∧ b > c) ∧  -- Monotonically decreasing
    (b - a = c - b) ∧  -- Arithmetic sequence
    (a + b + c = 12) ∧ -- Sum is 12
    (a * b * c = 48) ∧ -- Product is 48
    (a = 6 ∧ b = 4 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1030_103000


namespace NUMINAMATH_CALUDE_blue_notebook_cost_l1030_103003

/-- Represents the cost of notebooks in dollars -/
def TotalCost : ℕ := 37

/-- Represents the total number of notebooks -/
def TotalNotebooks : ℕ := 12

/-- Represents the number of red notebooks -/
def RedNotebooks : ℕ := 3

/-- Represents the cost of each red notebook in dollars -/
def RedNotebookCost : ℕ := 4

/-- Represents the number of green notebooks -/
def GreenNotebooks : ℕ := 2

/-- Represents the cost of each green notebook in dollars -/
def GreenNotebookCost : ℕ := 2

/-- Calculates the number of blue notebooks -/
def BlueNotebooks : ℕ := TotalNotebooks - RedNotebooks - GreenNotebooks

/-- Theorem: The cost of each blue notebook is 3 dollars -/
theorem blue_notebook_cost : 
  (TotalCost - RedNotebooks * RedNotebookCost - GreenNotebooks * GreenNotebookCost) / BlueNotebooks = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_notebook_cost_l1030_103003


namespace NUMINAMATH_CALUDE_number_puzzle_l1030_103011

theorem number_puzzle : ∃ x : ℝ, (100 - x = x + 40) ∧ (x = 30) := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l1030_103011


namespace NUMINAMATH_CALUDE_batting_average_calculation_l1030_103013

/-- Calculates the batting average given the total innings, highest score, score difference, and average excluding extremes -/
def batting_average (total_innings : ℕ) (highest_score : ℕ) (score_difference : ℕ) (avg_excluding_extremes : ℚ) : ℚ :=
  let lowest_score := highest_score - score_difference
  let runs_excluding_extremes := avg_excluding_extremes * (total_innings - 2)
  let total_runs := runs_excluding_extremes + highest_score + lowest_score
  total_runs / total_innings

theorem batting_average_calculation :
  batting_average 46 179 150 58 = 60 := by
  sorry

end NUMINAMATH_CALUDE_batting_average_calculation_l1030_103013


namespace NUMINAMATH_CALUDE_tangent_line_property_l1030_103050

/-- Given a line tangent to ln x and e^x, prove that 1/x₁ - 2/(x₂-1) = 1 --/
theorem tangent_line_property (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) : 
  (∃ (m b : ℝ), 
    (∀ x, m * x + b = (1 / x₁) * x + Real.log x₁ - 1) ∧
    (∀ x, m * x + b = Real.exp x₂ * x - Real.exp x₂ * (x₂ - 1))) →
  1 / x₁ - 2 / (x₂ - 1) = 1 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_property_l1030_103050


namespace NUMINAMATH_CALUDE_jakes_snake_length_l1030_103036

theorem jakes_snake_length (j p : ℕ) : 
  j = p + 12 →  -- Jake's snake is 12 inches longer than Penny's snake
  j + p = 70 →  -- The combined length of the two snakes is 70 inches
  j = 41        -- Jake's snake is 41 inches long
:= by sorry

end NUMINAMATH_CALUDE_jakes_snake_length_l1030_103036


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1030_103024

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if two lines are parallel. -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Returns true if a point lies on a line. -/
def pointOnLine (l : Line) (x y : ℝ) : Prop := y = l.slope * x + l.yIntercept

/-- The given line y = -3x + 6 -/
def givenLine : Line := { slope := -3, yIntercept := 6 }

theorem y_intercept_of_parallel_line :
  ∀ b : Line,
    parallel b givenLine →
    pointOnLine b 3 (-2) →
    b.yIntercept = 7 :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1030_103024


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1030_103091

def f (x : ℝ) : ℝ := x^2

theorem intersection_x_coordinate 
  (A B C E : ℝ × ℝ) 
  (hA : A = (2, f 2)) 
  (hB : B = (8, f 8)) 
  (hC : C.1 = (A.1 + B.1) / 2 ∧ C.2 = (A.2 + B.2) / 2) 
  (hE : E.1^2 = E.2 ∧ E.2 = C.2) : 
  E.1 = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1030_103091


namespace NUMINAMATH_CALUDE_janes_current_age_l1030_103065

theorem janes_current_age :
  let min_age : ℕ := 25
  let years_until_dara_eligible : ℕ := 14
  let years_until_half_age : ℕ := 6
  let dara_current_age : ℕ := min_age - years_until_dara_eligible
  ∀ jane_age : ℕ,
    (dara_current_age + years_until_half_age = (jane_age + years_until_half_age) / 2) →
    jane_age = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_current_age_l1030_103065


namespace NUMINAMATH_CALUDE_sum_of_exponents_outside_radical_l1030_103085

-- Define the expression
def original_expression (a b c : ℝ) : ℝ := (24 * a^4 * b^6 * c^11) ^ (1/3)

-- Define the simplified expression
def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^3 * ((3 * a * c^2) ^ (1/3))

-- State the theorem
theorem sum_of_exponents_outside_radical :
  ∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
  original_expression a b c = simplified_expression a b c ∧
  (1 + 2 + 3 = 6) := by sorry

end NUMINAMATH_CALUDE_sum_of_exponents_outside_radical_l1030_103085


namespace NUMINAMATH_CALUDE_juniors_in_sports_l1030_103064

def total_students : ℕ := 500
def junior_percentage : ℚ := 40 / 100
def sports_percentage : ℚ := 70 / 100

theorem juniors_in_sports :
  (total_students : ℚ) * junior_percentage * sports_percentage = 140 := by
  sorry

end NUMINAMATH_CALUDE_juniors_in_sports_l1030_103064


namespace NUMINAMATH_CALUDE_cyclist_climbing_speed_l1030_103059

/-- Proves that the climbing speed is 20 m/min given the specified conditions -/
theorem cyclist_climbing_speed 
  (hill_length : ℝ) 
  (total_time : ℝ) 
  (climbing_speed : ℝ) :
  hill_length = 400 ∧ 
  total_time = 30 ∧ 
  (∃ t : ℝ, t > 0 ∧ t < 30 ∧ 
    hill_length = climbing_speed * t ∧ 
    hill_length = 2 * climbing_speed * (total_time - t)) →
  climbing_speed = 20 := by
  sorry

#check cyclist_climbing_speed

end NUMINAMATH_CALUDE_cyclist_climbing_speed_l1030_103059


namespace NUMINAMATH_CALUDE_substance_mass_proof_l1030_103079

/-- The volume of 1 gram of the substance in cubic centimeters -/
def volume_per_gram : ℝ := 1.3333333333333335

/-- The number of cubic centimeters in 1 cubic meter -/
def cm3_per_m3 : ℝ := 1000000

/-- The number of grams in 1 kilogram -/
def grams_per_kg : ℝ := 1000

/-- The mass of 1 cubic meter of the substance in kilograms -/
def mass_per_m3 : ℝ := 750

theorem substance_mass_proof :
  mass_per_m3 = cm3_per_m3 / (grams_per_kg * volume_per_gram) := by
  sorry

end NUMINAMATH_CALUDE_substance_mass_proof_l1030_103079


namespace NUMINAMATH_CALUDE_discount_percentage_is_ten_percent_l1030_103093

/-- Calculates the discount percentage on a retail price given the wholesale price, retail price, and profit percentage. -/
def discount_percentage (wholesale_price retail_price profit_percentage : ℚ) : ℚ :=
  let profit := wholesale_price * profit_percentage / 100
  let selling_price := wholesale_price + profit
  let discount_amount := retail_price - selling_price
  (discount_amount / retail_price) * 100

/-- Proves that the discount percentage is 10% given the problem conditions. -/
theorem discount_percentage_is_ten_percent :
  discount_percentage 90 120 20 = 10 := by
  sorry

#eval discount_percentage 90 120 20

end NUMINAMATH_CALUDE_discount_percentage_is_ten_percent_l1030_103093


namespace NUMINAMATH_CALUDE_determine_relationship_l1030_103008

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - x - 2 > 0}
def Q : Set ℝ := {x : ℝ | |x - 1| > 1}

-- Define the possible relationships
inductive Relationship
  | sufficient_not_necessary
  | necessary_not_sufficient
  | necessary_and_sufficient
  | neither_sufficient_nor_necessary

-- Theorem to prove
theorem determine_relationship : Relationship :=
  sorry

end NUMINAMATH_CALUDE_determine_relationship_l1030_103008


namespace NUMINAMATH_CALUDE_octagon_diagonal_intersections_l1030_103088

/-- The number of vertices in a regular octagon -/
def n : ℕ := 8

/-- The number of diagonals in a regular octagon -/
def num_diagonals : ℕ := n * (n - 3) / 2

/-- The number of distinct intersection points of diagonals in the interior of a regular octagon -/
def num_intersection_points : ℕ := Nat.choose n 4

theorem octagon_diagonal_intersections :
  num_intersection_points = 70 :=
sorry

end NUMINAMATH_CALUDE_octagon_diagonal_intersections_l1030_103088


namespace NUMINAMATH_CALUDE_trip_cost_is_127_l1030_103031

/-- Represents a car with its specifications and trip details -/
structure Car where
  efficiency : ℝ  -- miles per gallon
  tankCapacity : ℝ  -- gallons
  initialMileage : ℝ  -- miles
  firstFillUpPrice : ℝ  -- dollars per gallon
  secondFillUpPrice : ℝ  -- dollars per gallon

/-- Calculates the total cost of a road trip given a car's specifications -/
def totalTripCost (c : Car) : ℝ :=
  c.tankCapacity * (c.firstFillUpPrice + c.secondFillUpPrice)

/-- Theorem stating that the total cost of the trip is $127.00 -/
theorem trip_cost_is_127 (c : Car) 
    (h1 : c.efficiency = 30)
    (h2 : c.tankCapacity = 20)
    (h3 : c.initialMileage = 1728)
    (h4 : c.firstFillUpPrice = 3.1)
    (h5 : c.secondFillUpPrice = 3.25) :
  totalTripCost c = 127 := by
  sorry

#eval totalTripCost { efficiency := 30, tankCapacity := 20, initialMileage := 1728, firstFillUpPrice := 3.1, secondFillUpPrice := 3.25 }

end NUMINAMATH_CALUDE_trip_cost_is_127_l1030_103031


namespace NUMINAMATH_CALUDE_cupcake_distribution_l1030_103081

/-- Represents the number of cupcakes in a pack --/
inductive PackSize
  | five : PackSize
  | ten : PackSize
  | fifteen : PackSize
  | twenty : PackSize

/-- Returns the number of cupcakes in a pack --/
def packSizeToInt (p : PackSize) : Nat :=
  match p with
  | PackSize.five => 5
  | PackSize.ten => 10
  | PackSize.fifteen => 15
  | PackSize.twenty => 20

/-- Calculates the total number of cupcakes from a given number of packs --/
def totalCupcakes (packSize : PackSize) (numPacks : Nat) : Nat :=
  (packSizeToInt packSize) * numPacks

/-- Represents Jean's initial purchase --/
def initialPurchase : Nat :=
  totalCupcakes PackSize.fifteen 4 + totalCupcakes PackSize.twenty 2

/-- The number of children in the orphanage --/
def numChildren : Nat := 220

/-- The theorem to prove --/
theorem cupcake_distribution :
  totalCupcakes PackSize.ten 8 + totalCupcakes PackSize.five 8 + initialPurchase = numChildren := by
  sorry

end NUMINAMATH_CALUDE_cupcake_distribution_l1030_103081


namespace NUMINAMATH_CALUDE_systematic_sampling_20_4_l1030_103092

def is_systematic_sample (n : ℕ) (k : ℕ) (sample : List ℕ) : Prop :=
  sample.length = k ∧
  ∀ i, i ∈ sample → i ≤ n ∧
  ∀ i j, i < j → i ∈ sample → j ∈ sample → (j - i) = n / k

theorem systematic_sampling_20_4 :
  is_systematic_sample 20 4 [5, 10, 15, 20] := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_20_4_l1030_103092


namespace NUMINAMATH_CALUDE_binomial_square_condition_l1030_103069

/-- If 9x^2 - 18x + a is the square of a binomial, then a = 9 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l1030_103069


namespace NUMINAMATH_CALUDE_prob_not_blue_twelve_sided_die_l1030_103049

-- Define the die
structure Die :=
  (sides : ℕ)
  (red_faces : ℕ)
  (yellow_faces : ℕ)
  (green_faces : ℕ)
  (blue_faces : ℕ)

-- Define the specific die from the problem
def twelve_sided_die : Die :=
  { sides := 12
  , red_faces := 5
  , yellow_faces := 4
  , green_faces := 2
  , blue_faces := 1 }

-- Define the probability of not rolling a blue face
def prob_not_blue (d : Die) : ℚ :=
  (d.sides - d.blue_faces) / d.sides

-- Theorem statement
theorem prob_not_blue_twelve_sided_die :
  prob_not_blue twelve_sided_die = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_blue_twelve_sided_die_l1030_103049


namespace NUMINAMATH_CALUDE_abc_inequality_l1030_103053

theorem abc_inequality (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l1030_103053


namespace NUMINAMATH_CALUDE_prob_both_blue_is_one_third_l1030_103033

/-- Represents a jar containing buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The initial state of Jar C -/
def initial_jar_c : Jar := { red := 6, blue := 10 }

/-- The number of buttons removed from each color -/
def buttons_removed : ℕ := 2

/-- The final state of Jar C after removal -/
def final_jar_c : Jar := 
  { red := initial_jar_c.red - buttons_removed,
    blue := initial_jar_c.blue - buttons_removed }

/-- The state of Jar D after buttons are added -/
def jar_d : Jar := { red := buttons_removed, blue := buttons_removed }

/-- The probability of selecting a blue button from a jar -/
def prob_blue (jar : Jar) : ℚ :=
  jar.blue / (jar.red + jar.blue)

theorem prob_both_blue_is_one_third :
  prob_blue final_jar_c * prob_blue jar_d = 1/3 := by
  sorry

#eval prob_blue final_jar_c -- Expected: 2/3
#eval prob_blue jar_d -- Expected: 1/2
#eval prob_blue final_jar_c * prob_blue jar_d -- Expected: 1/3

end NUMINAMATH_CALUDE_prob_both_blue_is_one_third_l1030_103033


namespace NUMINAMATH_CALUDE_tiffany_homework_problems_l1030_103058

/-- The total number of problems Tiffany had to complete -/
def total_problems (math_pages reading_pages science_pages history_pages : ℕ)
                   (math_problems_per_page reading_problems_per_page science_problems_per_page history_problems_per_page : ℕ) : ℕ :=
  math_pages * math_problems_per_page +
  reading_pages * reading_problems_per_page +
  science_pages * science_problems_per_page +
  history_pages * history_problems_per_page

/-- Theorem stating that the total number of problems is 46 -/
theorem tiffany_homework_problems :
  total_problems 6 4 3 2 3 3 4 2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_homework_problems_l1030_103058


namespace NUMINAMATH_CALUDE_hearts_ratio_equals_half_l1030_103028

-- Define the ♥ operation
def hearts (n m : ℕ) : ℕ := n^4 * m^3

-- Theorem statement
theorem hearts_ratio_equals_half : 
  (hearts 2 4) / (hearts 4 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hearts_ratio_equals_half_l1030_103028


namespace NUMINAMATH_CALUDE_unripe_oranges_eaten_l1030_103016

theorem unripe_oranges_eaten (total : ℕ) (uneaten : ℕ) : 
  total = 96 →
  uneaten = 78 →
  (1 : ℚ) / 8 = (total / 2 - uneaten) / (total / 2) := by
  sorry

end NUMINAMATH_CALUDE_unripe_oranges_eaten_l1030_103016


namespace NUMINAMATH_CALUDE_inequality_solution_l1030_103055

theorem inequality_solution (m n : ℝ) 
  (h : ∀ x, mx + n > 0 ↔ x < (1/2)) : 
  ∀ x, n*x - m < 0 ↔ x < -2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1030_103055


namespace NUMINAMATH_CALUDE_unique_solution_gcd_system_l1030_103082

theorem unique_solution_gcd_system (a b c : ℕ+) :
  a + b = (Nat.gcd a b)^2 ∧
  b + c = (Nat.gcd b c)^2 ∧
  c + a = (Nat.gcd c a)^2 →
  a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_gcd_system_l1030_103082


namespace NUMINAMATH_CALUDE_sqrt_difference_comparison_l1030_103094

theorem sqrt_difference_comparison (m : ℝ) (h : m > 1) :
  Real.sqrt (m + 1) - Real.sqrt m < Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_comparison_l1030_103094


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1030_103021

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the theorem
theorem sin_2alpha_value (α : ℝ) :
  (Real.cos α * P.1 = Real.sin α * P.2) →  -- Terminal side passes through P
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1030_103021


namespace NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l1030_103083

/-- 
Given a geometric sequence with positive terms, where:
- a₁ is the first term
- q is the common ratio
- S is the sum of the first 4 terms
- P is the product of the first 4 terms
- M is the sum of the reciprocals of the first 4 terms

Prove that if S = 9 and P = 81/4, then M = 2
-/
theorem geometric_sequence_reciprocal_sum 
  (a₁ q : ℝ) 
  (h_positive : a₁ > 0 ∧ q > 0) 
  (h_sum : a₁ * (1 - q^4) / (1 - q) = 9) 
  (h_product : a₁^4 * q^6 = 81/4) : 
  (1/a₁) * (1 - (1/q)^4) / (1 - 1/q) = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l1030_103083


namespace NUMINAMATH_CALUDE_max_sum_rational_l1030_103001

theorem max_sum_rational (x y : ℚ) : 
  x > 0 ∧ y > 0 ∧ 
  (∃ a b c d : ℕ, x = a / c ∧ y = b / d ∧ 
    a + b = 9 ∧ c + d = 10 ∧
    ∀ m n : ℕ, m * c = n * a → m = c ∧ n = a ∧
    ∀ m n : ℕ, m * d = n * b → m = d ∧ n = b) →
  x + y ≤ 73 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_rational_l1030_103001


namespace NUMINAMATH_CALUDE_harold_remaining_amount_l1030_103071

def calculate_remaining_amount (primary_income : ℚ) (freelance_income : ℚ) 
  (rent : ℚ) (car_payment : ℚ) (car_insurance : ℚ) (internet : ℚ) 
  (groceries : ℚ) (miscellaneous : ℚ) : ℚ :=
  let total_income := primary_income + freelance_income
  let electricity := 0.25 * car_payment
  let water_sewage := 0.15 * rent
  let total_expenses := rent + car_payment + car_insurance + electricity + water_sewage + internet + groceries + miscellaneous
  let amount_before_savings := total_income - total_expenses
  let savings := (2 / 3) * amount_before_savings
  amount_before_savings - savings

theorem harold_remaining_amount :
  calculate_remaining_amount 2500 500 700 300 125 75 200 150 = 423.34 := by
  sorry

end NUMINAMATH_CALUDE_harold_remaining_amount_l1030_103071


namespace NUMINAMATH_CALUDE_happy_street_weekly_total_l1030_103014

/-- The number of cars traveling down Happy Street each day of the week -/
structure WeeklyTraffic where
  tuesday : ℕ
  monday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Conditions for the traffic on Happy Street -/
def happy_street_traffic : WeeklyTraffic where
  tuesday := 25
  monday := 25 - (25 * 20 / 100)
  wednesday := (25 - (25 * 20 / 100)) + 2
  thursday := 10
  friday := 10
  saturday := 5
  sunday := 5

/-- The total number of cars traveling down Happy Street in a week -/
def total_weekly_traffic (w : WeeklyTraffic) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday + w.sunday

/-- Theorem stating that the total number of cars traveling down Happy Street in a week is 97 -/
theorem happy_street_weekly_total :
  total_weekly_traffic happy_street_traffic = 97 := by
  sorry

end NUMINAMATH_CALUDE_happy_street_weekly_total_l1030_103014


namespace NUMINAMATH_CALUDE_particular_number_divisibility_l1030_103034

theorem particular_number_divisibility (n : ℕ) : 
  n % 5 = 0 ∧ n / 5 = (320 / 4) + 220 → n / 3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_divisibility_l1030_103034


namespace NUMINAMATH_CALUDE_four_position_assignments_l1030_103026

def number_of_assignments (n : ℕ) : ℕ := n.factorial

theorem four_position_assignments :
  number_of_assignments 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_position_assignments_l1030_103026


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l1030_103054

theorem floor_plus_self_eq_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l1030_103054


namespace NUMINAMATH_CALUDE_seed_germination_problem_l1030_103015

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  0.25 * x + 0.3 * 200 = 0.27 * (x + 200) → 
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l1030_103015


namespace NUMINAMATH_CALUDE_tunnel_digging_problem_l1030_103068

theorem tunnel_digging_problem (total_length : ℝ) (team_a_rate : ℝ) (team_b_rate : ℝ) (remaining_distance : ℝ) :
  total_length = 1200 ∧ 
  team_a_rate = 12 ∧ 
  team_b_rate = 8 ∧ 
  remaining_distance = 200 →
  (total_length - remaining_distance) / (team_a_rate + team_b_rate) = 50 := by
sorry

end NUMINAMATH_CALUDE_tunnel_digging_problem_l1030_103068


namespace NUMINAMATH_CALUDE_angie_salary_is_80_l1030_103099

/-- Represents Angie's monthly finances -/
structure MonthlyFinances where
  necessities : ℕ
  taxes : ℕ
  leftover : ℕ

/-- Calculates the monthly salary based on expenses and leftover amount -/
def calculate_salary (finances : MonthlyFinances) : ℕ :=
  finances.necessities + finances.taxes + finances.leftover

/-- Theorem stating that Angie's monthly salary is $80 -/
theorem angie_salary_is_80 (angie : MonthlyFinances) 
  (h1 : angie.necessities = 42)
  (h2 : angie.taxes = 20)
  (h3 : angie.leftover = 18) :
  calculate_salary angie = 80 := by
  sorry

#eval calculate_salary { necessities := 42, taxes := 20, leftover := 18 }

end NUMINAMATH_CALUDE_angie_salary_is_80_l1030_103099


namespace NUMINAMATH_CALUDE_sector_central_angle_l1030_103056

theorem sector_central_angle (area : ℝ) (radius : ℝ) (h1 : area = 3 * π / 8) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1030_103056


namespace NUMINAMATH_CALUDE_max_a_value_l1030_103035

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := True

-- Define the line equation
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

-- Define the condition for no lattice points
def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 200 → is_lattice_point x y →
    line_equation m x ≠ y

-- State the theorem
theorem max_a_value :
  ∀ a : ℚ, (∀ m : ℚ, 1/2 < m ∧ m < a → no_lattice_points m) →
    a ≤ 101/201 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1030_103035


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l1030_103032

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  midpoint_segment : ℝ
  shorter_base : ℝ
  h1 : longer_base = 115
  h2 : midpoint_segment = 5
  h3 : midpoint_segment = (longer_base - shorter_base) / 2

/-- Theorem: In a trapezoid with the given properties, the shorter base has length 105 -/
theorem trapezoid_shorter_base (T : Trapezoid) : T.shorter_base = 105 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l1030_103032


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l1030_103096

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧
  a * b * c * d * e = 15120 →
  e = 9 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l1030_103096


namespace NUMINAMATH_CALUDE_perfect_square_unique_l1030_103009

/-- Checks if a quadratic expression ax^2 + bx + c is a perfect square trinomial -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  b^2 = 4*a*c ∧ a > 0

theorem perfect_square_unique :
  ¬ is_perfect_square_trinomial 1 0 1 ∧     -- x^2 + 1
  ¬ is_perfect_square_trinomial 1 2 (-1) ∧  -- x^2 + 2x - 1
  ¬ is_perfect_square_trinomial 1 1 1 ∧     -- x^2 + x + 1
  is_perfect_square_trinomial 1 4 4         -- x^2 + 4x + 4
  :=
sorry

end NUMINAMATH_CALUDE_perfect_square_unique_l1030_103009


namespace NUMINAMATH_CALUDE_expression_value_l1030_103043

theorem expression_value (x y : ℝ) (h : x^2 - 4*x - 1 = 0) :
  (2*x - 3)^2 - (x + y)*(x - y) - y^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1030_103043


namespace NUMINAMATH_CALUDE_solution_to_system_l1030_103007

theorem solution_to_system (x y m : ℝ) 
  (eq1 : 4 * x + 2 * y = 3 * m)
  (eq2 : 3 * x + y = m + 2)
  (opposite : y = -x) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l1030_103007


namespace NUMINAMATH_CALUDE_base_b_problem_l1030_103078

theorem base_b_problem : ∃! (b : ℕ), b > 1 ∧ (2 * b + 9)^2 = 7 * b^2 + 3 * b + 4 := by
  sorry

end NUMINAMATH_CALUDE_base_b_problem_l1030_103078


namespace NUMINAMATH_CALUDE_exam_time_ratio_l1030_103038

theorem exam_time_ratio :
  let total_questions : ℕ := 200
  let type_a_questions : ℕ := 50
  let type_b_questions : ℕ := total_questions - type_a_questions
  let exam_duration_hours : ℕ := 3
  let exam_duration_minutes : ℕ := exam_duration_hours * 60
  let time_for_type_a : ℕ := 72
  let time_for_type_b : ℕ := exam_duration_minutes - time_for_type_a
  (time_for_type_a : ℚ) / time_for_type_b = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_ratio_l1030_103038


namespace NUMINAMATH_CALUDE_coefficient_x4_is_correct_l1030_103044

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ :=
  4 * (x^4 - 2*x^5) + 3 * (x^3 - 3*x^4 + 2*x^6) - (5*x^5 - 2*x^4)

/-- The coefficient of x^4 in the simplified expression -/
def coefficient_x4 : ℝ := -3

theorem coefficient_x4_is_correct :
  (deriv (deriv (deriv (deriv expression)))) 0 / 24 = coefficient_x4 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_correct_l1030_103044


namespace NUMINAMATH_CALUDE_inverse_function_properties_l1030_103029

/-- Given a function g(x) = (3x+2)/(2x-5), this theorem proves properties of its inverse function. -/
theorem inverse_function_properties (x : ℝ) :
  let g := λ x : ℝ => (3*x + 2) / (2*x - 5)
  let g_inv := λ x : ℝ => (-5*x + 2) / (-2*x + 3)
  (∀ x, x ≠ 5/2 → g (g_inv x) = x) ∧
  (∀ x, x ≠ 2/3 → g_inv (g x) = x) ∧
  ((-5) / (-2) = 2.5) := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_properties_l1030_103029


namespace NUMINAMATH_CALUDE_granola_net_profit_l1030_103090

/-- Calculates the net profit from selling granola bags --/
theorem granola_net_profit
  (cost_per_bag : ℝ)
  (total_bags : ℕ)
  (full_price : ℝ)
  (discounted_price : ℝ)
  (bags_sold_full : ℕ)
  (bags_sold_discounted : ℕ)
  (h1 : cost_per_bag = 3)
  (h2 : total_bags = 20)
  (h3 : full_price = 6)
  (h4 : discounted_price = 4)
  (h5 : bags_sold_full = 15)
  (h6 : bags_sold_discounted = 5)
  (h7 : bags_sold_full + bags_sold_discounted = total_bags) :
  (full_price * bags_sold_full + discounted_price * bags_sold_discounted) - (cost_per_bag * total_bags) = 50 := by
  sorry

#check granola_net_profit

end NUMINAMATH_CALUDE_granola_net_profit_l1030_103090


namespace NUMINAMATH_CALUDE_cricket_game_run_rate_l1030_103041

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let firstPartRuns := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.target - firstPartRuns
  remainingRuns / remainingOvers

/-- Theorem statement for the cricket game scenario -/
theorem cricket_game_run_rate 
  (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 3.2)
  (h4 : game.target = 282) :
  requiredRunRate game = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_cricket_game_run_rate_l1030_103041


namespace NUMINAMATH_CALUDE_even_integers_between_fractions_l1030_103005

theorem even_integers_between_fractions :
  let lower_bound : ℚ := 23/5
  let upper_bound : ℚ := 47/3
  (Finset.filter (fun n => n % 2 = 0) (Finset.Icc ⌈lower_bound⌉ ⌊upper_bound⌋)).card = 5 :=
by sorry

end NUMINAMATH_CALUDE_even_integers_between_fractions_l1030_103005


namespace NUMINAMATH_CALUDE_balloon_arrangements_count_l1030_103063

/-- The number of distinct arrangements of letters in a word with 7 letters,
    where two letters are each repeated twice. -/
def balloonArrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of distinct arrangements of letters
    in a word with the given conditions is 1260. -/
theorem balloon_arrangements_count :
  balloonArrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_count_l1030_103063


namespace NUMINAMATH_CALUDE_solution_set_is_two_lines_l1030_103070

/-- The solution set of the equation (2x - y)^2 = 4x^2 - y^2 -/
def SolutionSet : Set (ℝ × ℝ) :=
  {(x, y) | (2*x - y)^2 = 4*x^2 - y^2}

/-- The set consisting of two lines: y = 0 and y = 2x -/
def TwoLines : Set (ℝ × ℝ) :=
  {(x, y) | y = 0 ∨ y = 2*x}

/-- Theorem stating that the solution set of the equation is equivalent to two lines -/
theorem solution_set_is_two_lines : SolutionSet = TwoLines := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_two_lines_l1030_103070


namespace NUMINAMATH_CALUDE_orange_price_l1030_103080

theorem orange_price (apple_price : ℚ) (total_fruit : ℕ) (initial_avg : ℚ) 
  (oranges_removed : ℕ) (final_avg : ℚ) :
  apple_price = 40 / 100 →
  total_fruit = 10 →
  initial_avg = 54 / 100 →
  oranges_removed = 4 →
  final_avg = 50 / 100 →
  ∃ (orange_price : ℚ),
    orange_price = 60 / 100 ∧
    ∃ (apples oranges : ℕ),
      apples + oranges = total_fruit ∧
      (apple_price * apples + orange_price * oranges) / total_fruit = initial_avg ∧
      (apple_price * apples + orange_price * (oranges - oranges_removed)) / 
        (total_fruit - oranges_removed) = final_avg :=
by
  sorry

end NUMINAMATH_CALUDE_orange_price_l1030_103080


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l1030_103051

theorem min_perimeter_triangle (a b c : ℕ) (h_integer : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_cosA : Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) = 11/16)
  (h_cosB : Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 7/8)
  (h_cosC : Real.cos (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = -1/4)
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) : 
  a + b + c ≥ 9 := by
  sorry

#check min_perimeter_triangle

end NUMINAMATH_CALUDE_min_perimeter_triangle_l1030_103051


namespace NUMINAMATH_CALUDE_smallest_root_of_g_l1030_103042

def g (x : ℝ) : ℝ := 12 * x^4 - 8 * x^2 + 1

theorem smallest_root_of_g :
  let r := Real.sqrt (1/6)
  (g r = 0) ∧ (∀ x : ℝ, g x = 0 → x ≥ 0 → x ≥ r) :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_g_l1030_103042


namespace NUMINAMATH_CALUDE_min_xy_value_l1030_103076

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (x + 1) * (y + 1) = 2 * x + 2 * y + 4) : 
  x * y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x + 1) * (y + 1) = 2 * x + 2 * y + 4 ∧ x * y = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l1030_103076


namespace NUMINAMATH_CALUDE_contrapositive_relation_l1030_103057

theorem contrapositive_relation (p r s : Prop) :
  (¬p ↔ r) → (r → s) → (s ↔ (¬p → p)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_relation_l1030_103057


namespace NUMINAMATH_CALUDE_matt_completes_in_100_days_l1030_103087

/-- The rate at which Matt and Peter complete work together -/
def combined_rate : ℚ := 1 / 20

/-- The rate at which Peter completes work alone -/
def peter_rate : ℚ := 1 / 25

/-- The rate at which Matt completes work alone -/
def matt_rate : ℚ := combined_rate - peter_rate

/-- The number of days Matt takes to complete the work alone -/
def matt_days : ℚ := 1 / matt_rate

theorem matt_completes_in_100_days : matt_days = 100 := by
  sorry

end NUMINAMATH_CALUDE_matt_completes_in_100_days_l1030_103087


namespace NUMINAMATH_CALUDE_members_neither_subject_count_l1030_103046

/-- The number of club members taking neither computer science nor robotics -/
def membersNeitherSubject (totalMembers csMembers roboticsMembers bothSubjects : ℕ) : ℕ :=
  totalMembers - (csMembers + roboticsMembers - bothSubjects)

/-- Theorem stating the number of club members taking neither subject -/
theorem members_neither_subject_count :
  membersNeitherSubject 150 80 70 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_members_neither_subject_count_l1030_103046


namespace NUMINAMATH_CALUDE_marble_collection_weight_l1030_103095

/-- The weight of Courtney's marble collection -/
def total_weight (jar1_count : ℕ) (jar1_weight : ℚ) (jar2_weight : ℚ) (jar3_weight : ℚ) (jar4_weight : ℚ) : ℚ :=
  let jar2_count := 2 * jar1_count
  let jar3_count := (1 : ℚ) / 4 * jar1_count
  let jar4_count := (3 : ℚ) / 5 * jar2_count
  jar1_count * jar1_weight + jar2_count * jar2_weight + jar3_count * jar3_weight + jar4_count * jar4_weight

/-- Theorem stating the total weight of Courtney's marble collection -/
theorem marble_collection_weight :
  total_weight 80 (35 / 100) (45 / 100) (25 / 100) (55 / 100) = 1578 / 10 := by
  sorry

end NUMINAMATH_CALUDE_marble_collection_weight_l1030_103095


namespace NUMINAMATH_CALUDE_total_questions_formula_l1030_103074

/-- Represents the number of questions completed by three girls in 2 hours -/
def total_questions (fiona_questions : ℕ) (r : ℚ) : ℚ :=
  let shirley_questions := r * fiona_questions
  let kiana_questions := (fiona_questions + shirley_questions) / 2
  2 * (fiona_questions + shirley_questions + kiana_questions)

/-- Theorem stating the total number of questions completed by three girls in 2 hours -/
theorem total_questions_formula (r : ℚ) : 
  total_questions 36 r = 108 + 108 * r := by
  sorry

end NUMINAMATH_CALUDE_total_questions_formula_l1030_103074


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1030_103047

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1030_103047


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_l1030_103027

/-- A circle with diameter endpoints at (0,0) and (10, -6) -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 + 3)^2 = 34}

/-- The x-axis -/
def XAxis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

/-- The point (0,0) -/
def Origin : ℝ × ℝ := (0, 0)

theorem circle_x_axis_intersection :
  ∃ p : ℝ × ℝ, p ∈ Circle ∩ XAxis ∧ p ≠ Origin ∧ p.1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_x_axis_intersection_l1030_103027


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l1030_103017

def num_islands : ℕ := 8
def prob_treasure_no_traps : ℚ := 1/3
def prob_treasure_and_traps : ℚ := 1/6
def prob_traps_no_treasure : ℚ := 1/6
def prob_neither : ℚ := 1/3

def target_treasure_islands : ℕ := 4
def target_treasure_and_traps_islands : ℕ := 2

theorem pirate_treasure_probability :
  let prob_treasure := prob_treasure_no_traps + prob_treasure_and_traps
  let prob_non_treasure := prob_traps_no_treasure + prob_neither
  (Nat.choose num_islands target_treasure_islands) *
  (Nat.choose target_treasure_islands target_treasure_and_traps_islands) *
  (prob_treasure ^ target_treasure_islands) *
  (prob_treasure_and_traps ^ target_treasure_and_traps_islands) *
  (prob_treasure_no_traps ^ (target_treasure_islands - target_treasure_and_traps_islands)) *
  (prob_non_treasure ^ (num_islands - target_treasure_islands)) =
  105 / 104976 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l1030_103017


namespace NUMINAMATH_CALUDE_complex_number_theorem_l1030_103067

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_theorem (z : ℂ) 
  (h1 : is_purely_imaginary z) 
  (h2 : is_purely_imaginary ((z + 2)^2 + 5)) : 
  z = Complex.I * 3 ∨ z = Complex.I * (-3) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l1030_103067


namespace NUMINAMATH_CALUDE_octagon_game_areas_l1030_103072

/-- A regular octagon inscribed in a circle of radius 2 -/
structure RegularOctagon :=
  (radius : ℝ)
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : ∀ i : Fin 8, (vertices i).1^2 + (vertices i).2^2 = radius^2)

/-- The set of vertices selected by a player -/
def PlayerSelection := Finset (Fin 8)

/-- Predicate for optimal play -/
def OptimalPlay (octagon : RegularOctagon) (alice_selection : PlayerSelection) (bob_selection : PlayerSelection) : Prop :=
  sorry

/-- The area of the convex polygon formed by a player's selection -/
def PolygonArea (octagon : RegularOctagon) (selection : PlayerSelection) : ℝ :=
  sorry

/-- The main theorem -/
theorem octagon_game_areas (octagon : RegularOctagon) (alice_selection : PlayerSelection) (bob_selection : PlayerSelection) :
  octagon.radius = 2 →
  OptimalPlay octagon alice_selection bob_selection →
  alice_selection.card = 4 →
  bob_selection.card = 4 →
  (PolygonArea octagon alice_selection = 2 * Real.sqrt 2 ∨
   PolygonArea octagon alice_selection = 4 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_octagon_game_areas_l1030_103072


namespace NUMINAMATH_CALUDE_isi_club_member_count_l1030_103089

/-- Represents a club with committees and members -/
structure Club where
  committee_count : ℕ
  member_count : ℕ
  committees_per_member : ℕ
  common_members : ℕ

/-- The ISI club satisfies the given conditions -/
def isi_club : Club :=
  { committee_count := 5,
    member_count := 10,
    committees_per_member := 2,
    common_members := 1 }

/-- Theorem: The ISI club has 10 members -/
theorem isi_club_member_count :
  isi_club.member_count = (isi_club.committee_count.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_isi_club_member_count_l1030_103089


namespace NUMINAMATH_CALUDE_no_solution_system_l1030_103097

theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 12) := by
sorry

end NUMINAMATH_CALUDE_no_solution_system_l1030_103097


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1030_103086

/-- The surface area of a cylinder with height 4 and base circumference 2π is 10π. -/
theorem cylinder_surface_area :
  ∀ (h r : ℝ), h = 4 ∧ 2 * π * r = 2 * π →
  2 * π * r * h + 2 * π * r^2 = 10 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1030_103086


namespace NUMINAMATH_CALUDE_expression_evaluation_l1030_103073

theorem expression_evaluation : 
  (-2/3)^2023 * (3/2)^2022 = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1030_103073


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l1030_103061

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l1030_103061


namespace NUMINAMATH_CALUDE_max_value_problem_1_max_value_problem_2_min_value_problem_3_l1030_103039

-- Problem 1
theorem max_value_problem_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  (1/2) * x * (1 - 2*x) ≤ 1/16 :=
sorry

-- Problem 2
theorem max_value_problem_2 (x : ℝ) (h : x < 3) :
  4 / (x - 3) + x ≤ -1 :=
sorry

-- Problem 3
theorem min_value_problem_3 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 4) :
  1/x + 3/y ≥ 1 + Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_1_max_value_problem_2_min_value_problem_3_l1030_103039


namespace NUMINAMATH_CALUDE_simplify_expression_l1030_103060

theorem simplify_expression (a b : ℝ) : (2*a - b) - (2*a + b) = -2*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1030_103060


namespace NUMINAMATH_CALUDE_expectation_problem_l1030_103052

/-- Given E(X) + E(2X + 1) = 8, prove that E(X) = 7/3 -/
theorem expectation_problem (X : ℝ → ℝ) (E : (ℝ → ℝ) → ℝ) 
  (h : E X + E (λ x => 2 * X x + 1) = 8) :
  E X = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_expectation_problem_l1030_103052


namespace NUMINAMATH_CALUDE_certain_number_proof_l1030_103020

theorem certain_number_proof (h : 16 * 21.3 = 340.8) : 213 * 16 = 3408 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1030_103020


namespace NUMINAMATH_CALUDE_matthews_contribution_l1030_103030

/-- Given the cost of pastries in euros, the amount of yen Hiroko has, and the exchange rate,
    calculate Matthew's contribution in euros. -/
theorem matthews_contribution
  (pastry_cost : ℝ)
  (hiroko_yen : ℝ)
  (exchange_rate : ℝ)
  (h1 : pastry_cost = 18)
  (h2 : hiroko_yen = 2500)
  (h3 : exchange_rate = 140) :
  pastry_cost - (hiroko_yen / exchange_rate) = 0.143 := by
  sorry

end NUMINAMATH_CALUDE_matthews_contribution_l1030_103030


namespace NUMINAMATH_CALUDE_function_properties_l1030_103062

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

-- State the theorem
theorem function_properties (a b : ℝ) :
  (f a 1 > 0) →
  (∀ x, f a x > b ↔ -1 < x ∧ x < 3) →
  ((a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ b = -3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1030_103062


namespace NUMINAMATH_CALUDE_cubic_roots_product_l1030_103025

theorem cubic_roots_product (α₁ α₂ α₃ : ℂ) : 
  (5 * α₁^3 - 6 * α₁^2 + 7 * α₁ + 8 = 0) ∧ 
  (5 * α₂^3 - 6 * α₂^2 + 7 * α₂ + 8 = 0) ∧ 
  (5 * α₃^3 - 6 * α₃^2 + 7 * α₃ + 8 = 0) →
  (α₁^2 + α₁*α₂ + α₂^2) * (α₂^2 + α₂*α₃ + α₃^2) * (α₁^2 + α₁*α₃ + α₃^2) = 764/625 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_product_l1030_103025


namespace NUMINAMATH_CALUDE_min_sum_distances_l1030_103012

/-- The minimum sum of distances from a point on the unit circle to two specific lines -/
theorem min_sum_distances : 
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 = 1
  let d1 : ℝ × ℝ → ℝ := λ (x, y) ↦ |3*x - 4*y - 10| / 5
  let d2 : ℝ × ℝ → ℝ := λ (x, y) ↦ |x - 3|
  ∃ (x y : ℝ), P (x, y) ∧ 
    ∀ (a b : ℝ), P (a, b) → d1 (x, y) + d2 (x, y) ≤ d1 (a, b) + d2 (a, b) ∧
    d1 (x, y) + d2 (x, y) = 5 - 4 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_l1030_103012


namespace NUMINAMATH_CALUDE_f_properties_l1030_103075

/-- The quadratic function f(x) = x^2 + ax + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- Theorem stating the properties of the function f --/
theorem f_properties (a : ℝ) :
  (∃ (s : Set ℝ), ∀ x, f a x > 0 ↔ x ∈ s) ∧
  (∀ x > 0, f a x ≥ 0) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1030_103075


namespace NUMINAMATH_CALUDE_mascot_prices_and_reduction_l1030_103002

/-- The price of a small mascot in yuan -/
def small_price : ℝ := 80

/-- The price of a large mascot in yuan -/
def large_price : ℝ := 120

/-- The price reduction in yuan -/
def price_reduction : ℝ := 10

theorem mascot_prices_and_reduction :
  /- Price of large mascot is 1.5 times the price of small mascot -/
  (large_price = 1.5 * small_price) ∧
  /- Number of small mascots purchased with 1200 yuan is 5 more than large mascots -/
  ((1200 / small_price) - (1200 / large_price) = 5) ∧
  /- Total sales revenue in February equals 75000 yuan -/
  ((small_price - price_reduction) * (500 + 10 * price_reduction) +
   (large_price - price_reduction) * 300 = 75000) := by
  sorry

end NUMINAMATH_CALUDE_mascot_prices_and_reduction_l1030_103002


namespace NUMINAMATH_CALUDE_angle_equality_l1030_103018

theorem angle_equality (α β : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2)
  (h3 : 0 < β) (h4 : β < Real.pi / 2)
  (h5 : Real.cos α + Real.cos β - Real.cos (α + β) = 3/2) :
  α = Real.pi / 3 ∧ β = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_l1030_103018


namespace NUMINAMATH_CALUDE_initial_student_count_l1030_103019

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (dropped_score : ℝ) : 
  initial_avg = 60.5 → new_avg = 64.0 → dropped_score = 8 → 
  ∃ n : ℕ, n > 0 ∧ 
    initial_avg * n = new_avg * (n - 1) + dropped_score ∧
    n = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_student_count_l1030_103019


namespace NUMINAMATH_CALUDE_total_crayons_l1030_103066

def new_crayons : ℕ := 2
def used_crayons : ℕ := 4
def broken_crayons : ℕ := 8

theorem total_crayons : new_crayons + used_crayons + broken_crayons = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l1030_103066


namespace NUMINAMATH_CALUDE_fraction_simplification_l1030_103010

theorem fraction_simplification (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) :
  x / (x - y) - y / (x + y) = (x^2 + y^2) / (x^2 - y^2) := by
  sorry


end NUMINAMATH_CALUDE_fraction_simplification_l1030_103010


namespace NUMINAMATH_CALUDE_breakfast_cost_theorem_l1030_103077

/-- The cost of each meal in Herman's breakfast purchases -/
def meal_cost (people : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_spent : ℕ) : ℚ :=
  total_spent / (people * days_per_week * weeks)

/-- Theorem stating that the meal cost is $4 given the problem conditions -/
theorem breakfast_cost_theorem :
  meal_cost 4 5 16 1280 = 4 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_theorem_l1030_103077


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1030_103023

theorem election_votes_calculation 
  (winning_percentage : Real) 
  (majority : Nat) 
  (total_votes : Nat) : 
  winning_percentage = 0.6 → 
  majority = 1504 → 
  (winning_percentage - (1 - winning_percentage)) * total_votes = majority → 
  total_votes = 7520 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1030_103023


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_481_over_12_l1030_103037

theorem sqrt_fraction_sum_equals_sqrt_481_over_12 :
  Real.sqrt (9 / 16 + 25 / 9) = Real.sqrt 481 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_481_over_12_l1030_103037


namespace NUMINAMATH_CALUDE_sector_central_angle_l1030_103004

/-- Given a sector with perimeter 10 and area 4, prove that its central angle is 1/2 radian -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 10) (h2 : (1/2) * l * r = 4) :
  l / r = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1030_103004


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l1030_103098

theorem stratified_sampling_medium_supermarkets :
  let total_supermarkets : ℕ := 200 + 400 + 1400
  let medium_supermarkets : ℕ := 400
  let sample_size : ℕ := 100
  (medium_supermarkets * sample_size) / total_supermarkets = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l1030_103098


namespace NUMINAMATH_CALUDE_midpoint_theorem_ap_twice_pb_theorem_l1030_103006

-- Define the line and points
def Line := ℝ → ℝ → Prop
def Point := ℝ × ℝ

-- Define the given point P
def P : Point := (-3, 1)

-- Define the properties of points A and B
def on_x_axis (A : Point) : Prop := A.2 = 0
def on_y_axis (B : Point) : Prop := B.1 = 0

-- Define the property of P being the midpoint of AB
def is_midpoint (P A B : Point) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define the property of AP = 2PB
def ap_twice_pb (P A B : Point) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (2 * (P.1 - B.1), 2 * (P.2 - B.2))

-- Define the equations of the lines
def line_eq1 (x y : ℝ) : Prop := x - 3*y + 6 = 0
def line_eq2 (x y : ℝ) : Prop := x - 6*y + 9 = 0

-- Theorem 1
theorem midpoint_theorem (l : Line) (A B : Point) :
  on_x_axis A → on_y_axis B → is_midpoint P A B →
  (∀ x y, l x y ↔ line_eq1 x y) :=
sorry

-- Theorem 2
theorem ap_twice_pb_theorem (l : Line) (A B : Point) :
  on_x_axis A → on_y_axis B → ap_twice_pb P A B →
  (∀ x y, l x y ↔ line_eq2 x y) :=
sorry

end NUMINAMATH_CALUDE_midpoint_theorem_ap_twice_pb_theorem_l1030_103006


namespace NUMINAMATH_CALUDE_min_pages_per_day_l1030_103040

theorem min_pages_per_day (total_pages : ℕ) (days_in_week : ℕ) : 
  total_pages = 220 → days_in_week = 7 → 
  ∃ (min_pages : ℕ), 
    min_pages * days_in_week ≥ total_pages ∧ 
    ∀ (x : ℕ), x * days_in_week ≥ total_pages → x ≥ min_pages ∧
    min_pages = 32 := by
  sorry

end NUMINAMATH_CALUDE_min_pages_per_day_l1030_103040


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1030_103084

theorem floor_equation_solution (a : ℝ) : 
  (∀ n : ℕ, 4 * ⌊a * n⌋ = n + ⌊a * ⌊a * n⌋⌋) ↔ a = 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1030_103084
