import Mathlib

namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1501_150149

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l m n : Line) (α : Plane) :
  parallel l m → parallel m n → perpendicular l α → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1501_150149


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1501_150174

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 1 / a 0

/-- Sum of the first n terms of a geometric sequence -/
def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ := sorry

/-- The main theorem -/
theorem geometric_sequence_sum_ratio 
  (seq : GeometricSequence) 
  (h : seq.a 6 = 8 * seq.a 3) : 
  sum_n seq 6 / sum_n seq 3 = 9 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1501_150174


namespace NUMINAMATH_CALUDE_theater_ticket_price_l1501_150134

/-- Calculates the ticket price for a theater performance --/
theorem theater_ticket_price
  (capacity : ℕ)
  (fill_rate : ℚ)
  (num_performances : ℕ)
  (total_earnings : ℕ)
  (h1 : capacity = 400)
  (h2 : fill_rate = 4/5)
  (h3 : num_performances = 3)
  (h4 : total_earnings = 28800) :
  (total_earnings : ℚ) / ((capacity : ℚ) * fill_rate * num_performances) = 30 :=
by
  sorry

#check theater_ticket_price

end NUMINAMATH_CALUDE_theater_ticket_price_l1501_150134


namespace NUMINAMATH_CALUDE_red_ball_probability_l1501_150101

/-- The probability of drawing a red ball from a bag with 1 red ball and 4 white balls is 0.2 -/
theorem red_ball_probability (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 1 →
  white_balls = 4 →
  (red_balls : ℚ) / total_balls = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l1501_150101


namespace NUMINAMATH_CALUDE_least_common_period_l1501_150133

-- Define the property that f satisfies the given functional equation
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

-- Define the property of being periodic with period p
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_common_period :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
    (∃ p : ℝ, p > 0 ∧ IsPeriodic f p) →
    (∀ q : ℝ, q > 0 → IsPeriodic f q → q ≥ 36) :=
  sorry

end NUMINAMATH_CALUDE_least_common_period_l1501_150133


namespace NUMINAMATH_CALUDE_binomial_8_choose_3_l1501_150153

theorem binomial_8_choose_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_3_l1501_150153


namespace NUMINAMATH_CALUDE_approximateValuesOfSqrt3_cannot_form_set_l1501_150136

-- Define a type for the concept of "group of objects"
structure GroupOfObjects where
  elements : Set ℝ
  description : String

-- Define the properties required for a set
def hasDeterminacy (g : GroupOfObjects) : Prop :=
  ∀ x, x ∈ g.elements → (∃ y, y = x)

def hasDistinctness (g : GroupOfObjects) : Prop :=
  ∀ x y, x ∈ g.elements → y ∈ g.elements → x = y → x = y

def hasUnorderedness (g : GroupOfObjects) : Prop :=
  ∀ x y, x ∈ g.elements → y ∈ g.elements → x ≠ y → y ∈ g.elements

-- Define what it means for a group of objects to be able to form a set
def canFormSet (g : GroupOfObjects) : Prop :=
  hasDeterminacy g ∧ hasDistinctness g ∧ hasUnorderedness g

-- Define the group of all approximate values of √3
def approximateValuesOfSqrt3 : GroupOfObjects :=
  { elements := {x : ℝ | ∃ ε > 0, |x^2 - 3| < ε},
    description := "All approximate values of √3" }

-- The theorem to prove
theorem approximateValuesOfSqrt3_cannot_form_set :
  ¬(canFormSet approximateValuesOfSqrt3) :=
sorry

end NUMINAMATH_CALUDE_approximateValuesOfSqrt3_cannot_form_set_l1501_150136


namespace NUMINAMATH_CALUDE_variance_of_defective_parts_l1501_150199

def defective_parts : List ℕ := [3, 3, 0, 2, 3, 0, 3]

def mean (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

def variance (list : List ℕ) : ℚ :=
  let μ := mean list
  (list.map (fun x => ((x : ℚ) - μ) ^ 2)).sum / list.length

theorem variance_of_defective_parts :
  variance defective_parts = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_defective_parts_l1501_150199


namespace NUMINAMATH_CALUDE_no_five_digit_perfect_square_with_all_even_or_odd_digits_l1501_150193

theorem no_five_digit_perfect_square_with_all_even_or_odd_digits : 
  ¬ ∃ (n : ℕ), 
    (∃ (k : ℕ), n = k^2) ∧ 
    (10000 ≤ n ∧ n < 100000) ∧
    (∀ (d₁ d₂ : ℕ), d₁ < 5 → d₂ < 5 → d₁ ≠ d₂ → 
      (n / 10^d₁ % 10) ≠ (n / 10^d₂ % 10)) ∧
    ((∀ (d : ℕ), d < 5 → Even (n / 10^d % 10)) ∨ 
     (∀ (d : ℕ), d < 5 → Odd (n / 10^d % 10))) :=
by sorry

end NUMINAMATH_CALUDE_no_five_digit_perfect_square_with_all_even_or_odd_digits_l1501_150193


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1501_150182

/-- The equation represents a hyperbola with foci on the x-axis -/
def is_hyperbola_x_axis (k : ℝ) : Prop :=
  ∃ (x y : ℝ → ℝ), ∀ t : ℝ, (x t)^2 / (k + 3) + (y t)^2 / (k + 2) = 1

theorem hyperbola_condition (k : ℝ) :
  is_hyperbola_x_axis k ↔ -3 < k ∧ k < -2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1501_150182


namespace NUMINAMATH_CALUDE_logarithmic_function_value_l1501_150119

noncomputable def f (a : ℝ) (x : ℝ) := (a^2 + a - 5) * Real.log x / Real.log a

theorem logarithmic_function_value (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (a^2 + a - 5 = 1) →
  f a (1/8) = -3 :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_function_value_l1501_150119


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_l1501_150144

theorem corresponding_angles_equal (α β γ : ℝ) :
  α + β + γ = 180 ∧ (180 - α) + β + γ = 180 →
  α = 180 - α ∧ β = β ∧ γ = γ := by
  sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_l1501_150144


namespace NUMINAMATH_CALUDE_frequency_distribution_necessary_sufficient_l1501_150157

/-- Represents a student's test score -/
def TestScore := ℕ

/-- Represents a group of students who took the test -/
def StudentGroup := List TestScore

/-- Represents the different score ranges -/
inductive ScoreRange
  | AboveOrEqual120
  | Between90And120
  | Between75And90
  | Between60And75
  | Below60

/-- Function to calculate the proportion of students in each score range -/
def calculateProportions (students : StudentGroup) : ScoreRange → ℚ :=
  sorry

/-- Function to perform frequency distribution -/
def frequencyDistribution (students : StudentGroup) : ScoreRange → ℕ :=
  sorry

/-- Theorem stating that frequency distribution is necessary and sufficient
    to determine the proportions of students in different score ranges -/
theorem frequency_distribution_necessary_sufficient
  (students : StudentGroup)
  (h : students.length = 800) :
  (∀ range, calculateProportions students range =
    (frequencyDistribution students range : ℚ) / 800) :=
sorry

end NUMINAMATH_CALUDE_frequency_distribution_necessary_sufficient_l1501_150157


namespace NUMINAMATH_CALUDE_line_up_count_proof_l1501_150145

/-- The number of ways to arrange 5 people in a line with the youngest not first or last -/
def lineUpCount : ℕ := 72

/-- The number of possible positions for the youngest person -/
def youngestPositions : ℕ := 3

/-- The number of ways to arrange the other 4 people -/
def otherArrangements : ℕ := 24

theorem line_up_count_proof :
  lineUpCount = youngestPositions * otherArrangements :=
by sorry

end NUMINAMATH_CALUDE_line_up_count_proof_l1501_150145


namespace NUMINAMATH_CALUDE_project_distribution_count_l1501_150175

/-- The number of districts --/
def num_districts : ℕ := 4

/-- The number of projects to be sponsored --/
def num_projects : ℕ := 3

/-- The maximum number of projects allowed in a single district --/
def max_projects_per_district : ℕ := 2

/-- The total number of possible distributions of projects among districts --/
def total_distributions : ℕ := num_districts ^ num_projects

/-- The number of invalid distributions (more than 2 projects in a district) --/
def invalid_distributions : ℕ := num_districts

theorem project_distribution_count :
  (total_distributions - invalid_distributions) = 60 := by
  sorry

end NUMINAMATH_CALUDE_project_distribution_count_l1501_150175


namespace NUMINAMATH_CALUDE_decimal_difference_l1501_150141

def repeating_decimal : ℚ := 9 / 11
def terminating_decimal : ℚ := 81 / 100

theorem decimal_difference : repeating_decimal - terminating_decimal = 9 / 1100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l1501_150141


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1501_150125

/-- Represents a trapezoid PQRU -/
structure Trapezoid where
  PQ : ℝ
  RU : ℝ

/-- The theorem stating the length of PQ in the given trapezoid -/
theorem trapezoid_segment_length (PQRU : Trapezoid) 
  (h1 : PQRU.PQ / PQRU.RU = 5 / 2)
  (h2 : PQRU.PQ + PQRU.RU = 180) : 
  PQRU.PQ = 900 / 7 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l1501_150125


namespace NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l1501_150159

theorem smallest_fourth_lucky_number :
  let first_three : List Nat := [68, 24, 85]
  let sum_first_three := first_three.sum
  let sum_digits_first_three := (first_three.map (fun n => n / 10 + n % 10)).sum
  ∀ x : Nat,
    x ≥ 10 ∧ x < 100 →
    (sum_first_three + x) * 1/4 = sum_digits_first_three + x / 10 + x % 10 →
    x ≥ 93 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l1501_150159


namespace NUMINAMATH_CALUDE_ocean_area_scientific_notation_l1501_150143

/-- The total area of the global ocean in million square kilometers -/
def ocean_area : ℝ := 36200

/-- The conversion factor from million to scientific notation -/
def million_to_scientific : ℝ := 10^6

theorem ocean_area_scientific_notation :
  ocean_area * million_to_scientific = 3.62 * 10^8 := by
  sorry

end NUMINAMATH_CALUDE_ocean_area_scientific_notation_l1501_150143


namespace NUMINAMATH_CALUDE_gold_distribution_l1501_150186

/-- Given an arithmetic sequence with 10 terms, if the sum of the first 3 terms
    is 4 and the sum of the last 4 terms is 3, then the common difference
    of the sequence is 7/78. -/
theorem gold_distribution (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
  (∀ n, n ≥ 10 → a n = 0) →             -- 10 terms
  a 9 + a 8 + a 7 = 4 →                 -- sum of first 3 terms is 4
  a 0 + a 1 + a 2 + a 3 = 3 →           -- sum of last 4 terms is 3
  a 1 - a 0 = 7 / 78 :=                 -- common difference is 7/78
by sorry

end NUMINAMATH_CALUDE_gold_distribution_l1501_150186


namespace NUMINAMATH_CALUDE_triangle_area_l1501_150188

theorem triangle_area (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin_sum : Real.sin (A + B) = 3/5)
  (h_sin_diff : Real.sin (A - B) = 1/5)
  (h_AB : 3 = 3) :
  (1/2) * 3 * (2 * Real.sqrt 6 - 2) = (6 + 3 * Real.sqrt 6) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1501_150188


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1501_150160

/-- Given an arithmetic sequence with 3n terms, prove that t₁ = t₂ -/
theorem arithmetic_sequence_property (n : ℕ) (s₁ s₂ s₃ : ℝ) :
  let t₁ := s₂^2 - s₁*s₃
  let t₂ := ((s₁ - s₃)/2)^2
  (s₁ + s₃ = 2*s₂) → t₁ = t₂ := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1501_150160


namespace NUMINAMATH_CALUDE_final_racers_count_l1501_150124

/-- Calculates the number of racers remaining after each elimination round -/
def remaining_racers (initial : ℕ) (first_elim : ℕ) (second_elim_frac : ℚ) (third_elim_frac : ℚ) : ℕ :=
  let after_first := initial - first_elim
  let after_second := after_first - (after_first * second_elim_frac).floor
  (after_second - (after_second * third_elim_frac).floor).toNat

/-- Theorem stating that given the initial conditions, 30 racers remain for the final section -/
theorem final_racers_count :
  remaining_racers 100 10 (1/3) (1/2) = 30 := by
  sorry


end NUMINAMATH_CALUDE_final_racers_count_l1501_150124


namespace NUMINAMATH_CALUDE_oldest_child_age_oldest_child_age_proof_l1501_150184

/-- Proves that given four children with an average age of 8 years, 
    and three of them being 5, 7, and 10 years old, 
    the age of the fourth child is 10 years. -/
theorem oldest_child_age 
  (total_children : Nat)
  (average_age : ℚ)
  (younger_children_ages : List Nat)
  (h1 : total_children = 4)
  (h2 : average_age = 8)
  (h3 : younger_children_ages = [5, 7, 10])
  : Nat :=
10

theorem oldest_child_age_proof : oldest_child_age 4 8 [5, 7, 10] rfl rfl rfl = 10 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_oldest_child_age_proof_l1501_150184


namespace NUMINAMATH_CALUDE_inequality_proof_l1501_150187

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1501_150187


namespace NUMINAMATH_CALUDE_point_range_theorem_l1501_150140

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 3 * x - 2 * y + a = 0

-- Define the condition for two points being on the same side of the line
def same_side (x1 y1 x2 y2 a : ℝ) : Prop :=
  (3 * x1 - 2 * y1 + a) * (3 * x2 - 2 * y2 + a) > 0

-- Theorem statement
theorem point_range_theorem (a : ℝ) :
  line_equation 3 (-1) a ∧ line_equation (-4) (-3) a ∧ same_side 3 (-1) (-4) (-3) a
  ↔ a < -11 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_point_range_theorem_l1501_150140


namespace NUMINAMATH_CALUDE_anns_age_is_30_l1501_150177

/-- Represents the ages of Ann and Barbara at different points in time. -/
structure AgeRelation where
  a : ℕ  -- Ann's current age
  b : ℕ  -- Barbara's current age

/-- The condition that the sum of their present ages is 50 years. -/
def sum_of_ages (ages : AgeRelation) : Prop :=
  ages.a + ages.b = 50

/-- The complex age relation described in the problem. -/
def age_relation (ages : AgeRelation) : Prop :=
  ∃ (y : ℕ), 
    ages.b = ages.a / 2 + 2 * y ∧
    ages.a - ages.b = y

/-- The theorem stating that given the conditions, Ann's age is 30 years. -/
theorem anns_age_is_30 (ages : AgeRelation) 
  (h1 : sum_of_ages ages) 
  (h2 : age_relation ages) : 
  ages.a = 30 := by sorry

end NUMINAMATH_CALUDE_anns_age_is_30_l1501_150177


namespace NUMINAMATH_CALUDE_f_increasing_interval_f_not_increasing_below_one_l1501_150163

/-- The function f(x) = |x-1| + |x+1| -/
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

/-- The interval of increase for f(x) -/
def interval_of_increase : Set ℝ := { x | x ≥ 1 }

/-- Theorem stating that the interval of increase for f(x) is [1, +∞) -/
theorem f_increasing_interval :
  ∀ x y, x ∈ interval_of_increase → y ∈ interval_of_increase → x < y → f x < f y :=
by sorry

/-- Theorem stating that f(x) is not increasing for x < 1 -/
theorem f_not_increasing_below_one :
  ∃ x y, x < 1 ∧ y < 1 ∧ x < y ∧ f x ≥ f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_interval_f_not_increasing_below_one_l1501_150163


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l1501_150132

theorem min_perimeter_triangle (d e f : ℕ) : 
  d > 0 → e > 0 → f > 0 →
  (d^2 + e^2 - f^2 : ℚ) / (2 * d * e) = 3 / 5 →
  (d^2 + f^2 - e^2 : ℚ) / (2 * d * f) = 9 / 10 →
  (e^2 + f^2 - d^2 : ℚ) / (2 * e * f) = -1 / 3 →
  d + e + f ≥ 50 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l1501_150132


namespace NUMINAMATH_CALUDE_right_triangle_roots_l1501_150108

-- Define the equation
def equation (m x : ℝ) : Prop := x^2 - (2*m + 1)*x + m^2 + m = 0

-- Define the roots
def roots (m : ℝ) : Set ℝ := {x | equation m x}

theorem right_triangle_roots (m : ℝ) :
  let a := (2*m + 1 + 1) / 2
  let b := (2*m + 1 - 1) / 2
  (∀ x ∈ roots m, x = a ∨ x = b) →
  a^2 + b^2 = 5^2 →
  m = 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_roots_l1501_150108


namespace NUMINAMATH_CALUDE_alok_chapati_order_l1501_150120

-- Define the variables
def rice_plates : ℕ := 5
def vegetable_plates : ℕ := 7
def ice_cream_cups : ℕ := 6
def chapati_cost : ℕ := 6
def rice_cost : ℕ := 45
def vegetable_cost : ℕ := 70
def total_paid : ℕ := 1111

-- Define the theorem
theorem alok_chapati_order :
  ∃ (chapatis : ℕ), 
    chapatis * chapati_cost + 
    rice_plates * rice_cost + 
    vegetable_plates * vegetable_cost + 
    ice_cream_cups * (total_paid - (chapatis * chapati_cost + rice_plates * rice_cost + vegetable_plates * vegetable_cost)) / ice_cream_cups = 
    total_paid ∧ 
    chapatis = 66 := by
  sorry

end NUMINAMATH_CALUDE_alok_chapati_order_l1501_150120


namespace NUMINAMATH_CALUDE_problem_statement_l1501_150176

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + 2 * a + b = 16) :
  (∀ x y : ℝ, x > 0 → y > 0 → x * y + 2 * x + y = 16 → a * b ≥ x * y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x * y + 2 * x + y = 16 → 2 * a + b ≤ 2 * x + y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x * y + 2 * x + y = 16 → 1 / (a + 1) + 1 / (b + 2) ≤ 1 / (x + 1) + 1 / (y + 2)) ∧
  (a * b = 8 ∨ 2 * a + b = 8 ∨ 1 / (a + 1) + 1 / (b + 2) = Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1501_150176


namespace NUMINAMATH_CALUDE_beth_class_size_l1501_150129

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Calculates the final number of students in Beth's class after n years -/
def finalStudents (initialStudents : ℕ) (joiningStart : ℕ) (joiningDiff : ℕ) 
                  (leavingStart : ℕ) (leavingDiff : ℕ) (years : ℕ) : ℕ :=
  initialStudents + 
  (arithmeticSum joiningStart joiningDiff years) - 
  (arithmeticSum leavingStart leavingDiff years)

theorem beth_class_size :
  finalStudents 150 30 5 15 3 4 = 222 := by
  sorry

end NUMINAMATH_CALUDE_beth_class_size_l1501_150129


namespace NUMINAMATH_CALUDE_base6_addition_problem_l1501_150167

/-- Converts a base 6 number to its decimal representation -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Checks if the given digit satisfies the base 6 addition problem -/
def satisfiesAdditionProblem (digit : Nat) : Prop :=
  let num1 := base6ToDecimal [4, 3, 2, digit]
  let num2 := base6ToDecimal [digit, 5, 1]
  let num3 := base6ToDecimal [digit, 3]
  let sum := base6ToDecimal [5, 3, digit, 0]
  num1 + num2 + num3 = sum

theorem base6_addition_problem :
  ∃! (digit : Nat), digit < 6 ∧ satisfiesAdditionProblem digit :=
sorry

end NUMINAMATH_CALUDE_base6_addition_problem_l1501_150167


namespace NUMINAMATH_CALUDE_johns_age_l1501_150158

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

end NUMINAMATH_CALUDE_johns_age_l1501_150158


namespace NUMINAMATH_CALUDE_square_difference_306_294_l1501_150116

theorem square_difference_306_294 : 306^2 - 294^2 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_306_294_l1501_150116


namespace NUMINAMATH_CALUDE_aye_aye_friendship_l1501_150151

theorem aye_aye_friendship (n : ℕ) (h : n = 23) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧ 
  (∃ (f : Fin n → Finset (Fin n)), 
    (∀ x, x ∉ f x) ∧
    (∀ x y, y ∈ f x ↔ x ∈ f y) ∧
    (f i).card = k ∧ (f j).card = k) :=
by sorry


end NUMINAMATH_CALUDE_aye_aye_friendship_l1501_150151


namespace NUMINAMATH_CALUDE_line_bisecting_circle_min_value_l1501_150154

/-- Given a line that always bisects the circumference of a circle, 
    prove the minimum value of 1/a + 1/b -/
theorem line_bisecting_circle_min_value (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, 2*a*x - b*y + 2 = 0 → 
    x^2 + y^2 + 2*x - 4*y + 1 = 0 → 
    -- The line bisects the circle (implicit condition)
    True) → 
  (1/a + 1/b) ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_line_bisecting_circle_min_value_l1501_150154


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l1501_150197

theorem a_eq_one_sufficient_not_necessary :
  ∃ (a : ℝ), a ^ 2 = a ∧ a ≠ 1 ∧
  ∀ (b : ℝ), b = 1 → b ^ 2 = b :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l1501_150197


namespace NUMINAMATH_CALUDE_z_range_l1501_150127

theorem z_range (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x + y = x * y) (hxyz : x + y + z = x * y * z) :
  1 < z ∧ z ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_z_range_l1501_150127


namespace NUMINAMATH_CALUDE_A_and_C_work_time_l1501_150171

-- Define work rates
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12
def work_rate_BC : ℚ := 1 / 3

-- Define the theorem
theorem A_and_C_work_time :
  let work_rate_C : ℚ := work_rate_BC - work_rate_B
  let work_rate_AC : ℚ := work_rate_A + work_rate_C
  (1 : ℚ) / work_rate_AC = 2 := by sorry

end NUMINAMATH_CALUDE_A_and_C_work_time_l1501_150171


namespace NUMINAMATH_CALUDE_isosceles_triangle_most_stable_l1501_150131

-- Define the shapes
inductive Shape
  | RegularHexagon
  | Square
  | Pentagon
  | IsoscelesTriangle

-- Define a function to get the number of sides for each shape
def numSides (s : Shape) : Nat :=
  match s with
  | .RegularHexagon => 6
  | .Square => 4
  | .Pentagon => 5
  | .IsoscelesTriangle => 3

-- Define stability as inversely proportional to the number of sides
def stability (s : Shape) : Nat := 7 - numSides s

-- Theorem: Isosceles triangle is the most stable shape
theorem isosceles_triangle_most_stable :
  ∀ s : Shape, s ≠ Shape.IsoscelesTriangle → 
    stability Shape.IsoscelesTriangle > stability s :=
by sorry

#check isosceles_triangle_most_stable

end NUMINAMATH_CALUDE_isosceles_triangle_most_stable_l1501_150131


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1501_150165

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 4 ∧ (5026 - k) % 5 = 0 ∧ ∀ (m : ℕ), m < k → (5026 - m) % 5 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1501_150165


namespace NUMINAMATH_CALUDE_range_of_m_l1501_150130

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 4 * x + y - x * y = 0) (h2 : x * y ≥ m^2 - 6*m) : 
  -2 ≤ m ∧ m ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1501_150130


namespace NUMINAMATH_CALUDE_possible_m_values_l1501_150104

def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

theorem possible_m_values (m : ℝ) : B m ⊆ A m → m = 0 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_possible_m_values_l1501_150104


namespace NUMINAMATH_CALUDE_power_multiplication_l1501_150100

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1501_150100


namespace NUMINAMATH_CALUDE_max_m_value_l1501_150155

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_eq : 2/a + 1/b = 1/4) :
  (∀ m : ℝ, 2*a + b ≥ 9*m) → (∃ m_max : ℝ, m_max = 4 ∧ ∀ m : ℝ, (2*a + b ≥ 9*m → m ≤ m_max)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1501_150155


namespace NUMINAMATH_CALUDE_angle_extrema_l1501_150138

/-- The angle formed by the construction described in the problem -/
def constructionAngle (x : Fin n → ℝ) : ℝ :=
  sorry

/-- The theorem stating that the angle is minimal for descending sequences and maximal for ascending sequences -/
theorem angle_extrema (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, x i > 0) :
  (∀ i j, i < j → x i ≥ x j) →
  (∀ y : Fin n → ℝ, (∀ i, y i > 0) → constructionAngle x ≤ constructionAngle y) ∧
  (∀ i j, i < j → x i ≤ x j) →
  (∀ y : Fin n → ℝ, (∀ i, y i > 0) → constructionAngle x ≥ constructionAngle y) :=
sorry

end NUMINAMATH_CALUDE_angle_extrema_l1501_150138


namespace NUMINAMATH_CALUDE_product_of_roots_l1501_150190

theorem product_of_roots (x : ℝ) : 
  (25 * x^2 + 60 * x - 375 = 0) → 
  (∃ r₁ r₂ : ℝ, (25 * r₁^2 + 60 * r₁ - 375 = 0) ∧ 
                (25 * r₂^2 + 60 * r₂ - 375 = 0) ∧ 
                (r₁ * r₂ = -15)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1501_150190


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_l1501_150123

theorem tan_alpha_2_implies (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6/13 ∧
  3 * Real.sin α ^ 2 + 3 * Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_l1501_150123


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1501_150162

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*x)/(61*y)) :
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1501_150162


namespace NUMINAMATH_CALUDE_andy_remaining_demerits_l1501_150142

/-- The number of additional demerits Andy can get before being fired -/
def additional_demerits (max_demerits : ℕ) (lateness_instances : ℕ) (demerits_per_lateness : ℕ) (joke_demerits : ℕ) : ℕ :=
  max_demerits - (lateness_instances * demerits_per_lateness + joke_demerits)

/-- Theorem stating that Andy can get 23 more demerits before being fired -/
theorem andy_remaining_demerits :
  additional_demerits 50 6 2 15 = 23 := by
  sorry

end NUMINAMATH_CALUDE_andy_remaining_demerits_l1501_150142


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1501_150115

/-- The number of ordered pairs (m,n) of positive integers satisfying the given conditions -/
def solution_count : ℕ := 3

/-- Predicate defining the conditions for valid pairs -/
def is_valid_pair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 128

theorem count_valid_pairs :
  (∃! (s : Finset (ℕ × ℕ)), ∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_pair p.1 p.2) ∧
  (∃ (s : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_pair p.1 p.2) ∧ s.card = solution_count) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1501_150115


namespace NUMINAMATH_CALUDE_milk_replacement_amount_l1501_150126

/-- Represents the amount of milk removed and replaced with water in each operation -/
def x : ℝ := 9

/-- The capacity of the vessel in litres -/
def vessel_capacity : ℝ := 90

/-- The amount of pure milk remaining after the operations in litres -/
def final_pure_milk : ℝ := 72.9

/-- Theorem stating that the amount of milk removed and replaced with water in each operation is correct -/
theorem milk_replacement_amount : 
  vessel_capacity - x - (vessel_capacity - x) * x / vessel_capacity = final_pure_milk := by
  sorry

end NUMINAMATH_CALUDE_milk_replacement_amount_l1501_150126


namespace NUMINAMATH_CALUDE_dog_sled_race_l1501_150139

theorem dog_sled_race (total_sleds : ℕ) (pairs : ℕ) (triples : ℕ) 
  (h1 : total_sleds = 315)
  (h2 : pairs + triples = total_sleds)
  (h3 : (6 * pairs + 2 * triples) * 10 = (2 * pairs + 3 * triples) * 5) :
  pairs = 225 ∧ triples = 90 := by
  sorry

end NUMINAMATH_CALUDE_dog_sled_race_l1501_150139


namespace NUMINAMATH_CALUDE_lock_and_key_theorem_l1501_150189

/-- The number of scientists in the team -/
def n : ℕ := 7

/-- The minimum number of scientists required to open the door -/
def k : ℕ := 4

/-- The number of scientists that can be absent -/
def m : ℕ := n - k

/-- The number of unique locks required -/
def num_locks : ℕ := Nat.choose n m

/-- The number of keys each scientist must have -/
def num_keys : ℕ := Nat.choose (n - 1) m

theorem lock_and_key_theorem :
  (num_locks = 35) ∧ (num_keys = 20) :=
sorry

end NUMINAMATH_CALUDE_lock_and_key_theorem_l1501_150189


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l1501_150121

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 31)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0)
  : total_players - (total_players - throwers) / 3 = 57 := by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l1501_150121


namespace NUMINAMATH_CALUDE_horner_v3_equals_16_l1501_150110

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^6 - 5x^4 + 2x^3 - x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ :=
  3 * x^6 - 5 * x^4 + 2 * x^3 - x^2 + 2 * x + 1

/-- v3 in Horner's method for f(x) -/
def v3 (x : ℝ) : ℝ :=
  (((3 * x - 0) * x - 5) * x + 2)

theorem horner_v3_equals_16 :
  v3 2 = 16 := by sorry

end NUMINAMATH_CALUDE_horner_v3_equals_16_l1501_150110


namespace NUMINAMATH_CALUDE_males_band_not_orchestra_l1501_150178

/-- Represents the school band and orchestra at West Valley High -/
structure MusicGroups where
  band_females : ℕ
  band_males : ℕ
  orchestra_females : ℕ
  orchestra_males : ℕ
  both_females : ℕ
  total_students : ℕ

/-- The specific music groups at West Valley High -/
def westValleyHigh : MusicGroups :=
  { band_females := 120
  , band_males := 100
  , orchestra_females := 90
  , orchestra_males := 110
  , both_females := 70
  , total_students := 250 }

/-- The number of males in the band who are not in the orchestra is 0 -/
theorem males_band_not_orchestra (g : MusicGroups) (h : g = westValleyHigh) :
  g.band_males - (g.band_males + g.orchestra_males - (g.total_students - (g.band_females + g.orchestra_females - g.both_females))) = 0 := by
  sorry


end NUMINAMATH_CALUDE_males_band_not_orchestra_l1501_150178


namespace NUMINAMATH_CALUDE_inequality_proof_l1501_150102

theorem inequality_proof (x : ℝ) (h : x > 0) : x^8 - x^5 - 1/x + 1/x^4 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1501_150102


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l1501_150137

/-- A line passing through two given points intersects the y-axis at a specific point -/
theorem line_y_axis_intersection (x₁ y₁ x₂ y₂ : ℚ) 
  (h₁ : x₁ = 3) (h₂ : y₁ = 20) (h₃ : x₂ = -9) (h₄ : y₂ = -6) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0, b) = (0, 27/2) :=
by sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l1501_150137


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1501_150106

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- ABC is an acute-angled scalene triangle -/
def Triangle (A B C : Point) : Prop := sorry

/-- AH is an altitude of triangle ABC -/
def IsAltitude (H : Point) (A B C : Point) : Prop := sorry

/-- AM is a median of triangle ABC -/
def IsMedian (M : Point) (A B C : Point) : Prop := sorry

/-- O is the center of the circumscribed circle ω of triangle ABC -/
def IsCircumcenter (O : Point) (A B C : Point) (ω : Circle) : Prop := sorry

/-- Two lines intersect at a point -/
def Intersect (l1 l2 : Line) (P : Point) : Prop := sorry

/-- A line intersects a circle at a point -/
def IntersectCircle (l : Line) (c : Circle) (P : Point) : Prop := sorry

theorem intersection_of_lines 
  (A B C H M O D E F X Y : Point) 
  (ω : Circle) :
  Triangle A B C →
  IsAltitude H A B C →
  IsMedian M A B C →
  IsCircumcenter O A B C ω →
  Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) D →  -- OH and AM
  Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) E →  -- AB and CD
  Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) F →  -- BD and AC
  IntersectCircle (Line.mk 0 0 0) ω X →  -- EH and ω
  IntersectCircle (Line.mk 0 0 0) ω Y →  -- FH and ω
  ∃ P, Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) P ∧  -- BY and CX
      Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) P ∧  -- CX and AH
      Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) P    -- AH and BY
:= by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1501_150106


namespace NUMINAMATH_CALUDE_sqrt_6_simplest_l1501_150109

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y^2 = x → ¬∃ z : ℝ, z > 0 ∧ z < y ∧ z^2 = x

theorem sqrt_6_simplest :
  is_simplest_sqrt 6 ∧
  ¬is_simplest_sqrt (1/6) ∧
  ¬is_simplest_sqrt 0.6 ∧
  ¬is_simplest_sqrt 60 :=
sorry

end NUMINAMATH_CALUDE_sqrt_6_simplest_l1501_150109


namespace NUMINAMATH_CALUDE_class_size_is_50_l1501_150166

/-- The number of students in class 4(1) -/
def class_size : ℕ := 50

/-- The number of students in the basketball group -/
def basketball_group : ℕ := class_size / 2 + 1

/-- The number of students in the table tennis group -/
def table_tennis_group : ℕ := (class_size - basketball_group) / 2 + 2

/-- The number of students in the chess group -/
def chess_group : ℕ := (class_size - basketball_group - table_tennis_group) / 2 + 3

/-- The number of students in the broadcasting group -/
def broadcasting_group : ℕ := 2

theorem class_size_is_50 :
  class_size = 50 ∧
  basketball_group > class_size / 2 ∧
  table_tennis_group = (class_size - basketball_group) / 2 + 2 ∧
  chess_group = (class_size - basketball_group - table_tennis_group) / 2 + 3 ∧
  broadcasting_group = 2 ∧
  class_size = basketball_group + table_tennis_group + chess_group + broadcasting_group :=
by sorry

end NUMINAMATH_CALUDE_class_size_is_50_l1501_150166


namespace NUMINAMATH_CALUDE_distribute_balls_into_boxes_l1501_150122

theorem distribute_balls_into_boxes (n : ℕ) (k : ℕ) : 
  n = 5 → k = 4 → (Nat.choose (n + k - 1) (k - 1)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_balls_into_boxes_l1501_150122


namespace NUMINAMATH_CALUDE_c_profit_share_l1501_150196

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_profit_share (total_investment : ℕ) (partner_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (partner_investment * total_profit) / total_investment

theorem c_profit_share :
  let a_investment : ℕ := 5000
  let b_investment : ℕ := 8000
  let c_investment : ℕ := 9000
  let total_investment : ℕ := a_investment + b_investment + c_investment
  let total_profit : ℕ := 88000
  calculate_profit_share total_investment c_investment total_profit = 36000 := by
  sorry

end NUMINAMATH_CALUDE_c_profit_share_l1501_150196


namespace NUMINAMATH_CALUDE_simplify_expression_l1501_150112

theorem simplify_expression (a : ℝ) (h : a ≤ (1/2 : ℝ)) :
  Real.sqrt (1 - 4*a + 4*a^2) + |2*a - 1| = 2 - 4*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1501_150112


namespace NUMINAMATH_CALUDE_dealership_van_sales_l1501_150191

/-- Calculates the expected number of vans to be sold given the truck-to-van ratio and the number of trucks expected to be sold. -/
def expected_vans (truck_ratio : ℕ) (van_ratio : ℕ) (trucks_sold : ℕ) : ℕ :=
  (van_ratio * trucks_sold) / truck_ratio

/-- Theorem stating that given a 3:5 ratio of trucks to vans and an expected sale of 45 trucks, 
    the expected number of vans to be sold is 75. -/
theorem dealership_van_sales : expected_vans 3 5 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_dealership_van_sales_l1501_150191


namespace NUMINAMATH_CALUDE_encyclopedia_chapters_l1501_150181

theorem encyclopedia_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) (h1 : total_pages = 3962) (h2 : pages_per_chapter = 566) :
  total_pages / pages_per_chapter = 7 := by
sorry

end NUMINAMATH_CALUDE_encyclopedia_chapters_l1501_150181


namespace NUMINAMATH_CALUDE_max_value_sin_cos_function_l1501_150117

theorem max_value_sin_cos_function :
  let f : ℝ → ℝ := λ x => Real.sin (π / 2 + x) * Real.cos (π / 6 - x)
  ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = (2 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_function_l1501_150117


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1501_150113

/-- The eccentricity of a hyperbola with equation x^2 - y^2/4 = 1 is √5 -/
theorem hyperbola_eccentricity :
  let a : ℝ := 1  -- semi-major axis
  let b : ℝ := 2  -- semi-minor axis
  let c : ℝ := Real.sqrt (a^2 + b^2)  -- distance from center to focus
  let e : ℝ := c / a  -- eccentricity
  e = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1501_150113


namespace NUMINAMATH_CALUDE_kim_payment_share_l1501_150146

/-- Represents the time (in days) it takes a person to complete the work alone -/
structure WorkTime where
  days : ℚ
  days_positive : days > 0

/-- Calculates the work rate (portion of work done per day) given the work time -/
def work_rate (wt : WorkTime) : ℚ := 1 / wt.days

/-- Calculates the share of payment for a person given their work rate and the total work rate -/
def payment_share (individual_rate total_rate : ℚ) : ℚ := individual_rate / total_rate

theorem kim_payment_share 
  (kim : WorkTime)
  (david : WorkTime)
  (lisa : WorkTime)
  (h_kim : kim.days = 3)
  (h_david : david.days = 2)
  (h_lisa : lisa.days = 4)
  (total_payment : ℚ)
  (h_total_payment : total_payment = 200) :
  payment_share (work_rate kim) (work_rate kim + work_rate david + work_rate lisa) * total_payment = 800 / 13 :=
sorry

end NUMINAMATH_CALUDE_kim_payment_share_l1501_150146


namespace NUMINAMATH_CALUDE_conversation_on_thursday_l1501_150103

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the library schedule -/
def LibrarySchedule := DayOfWeek → Bool

/-- Represents a boy's visiting schedule -/
def VisitSchedule := ℕ → DayOfWeek

/-- The number of days between visits for each boy -/
def visitIntervals : List ℕ := [2, 3, 4]

/-- The library is closed on Wednesdays -/
def libraryClosedWednesday (schedule : LibrarySchedule) : Prop :=
  schedule DayOfWeek.Wednesday = false

/-- All boys met again on a Monday -/
def metAgainOnMonday (schedules : List VisitSchedule) : Prop :=
  ∀ s ∈ schedules, ∃ n : ℕ, s n = DayOfWeek.Monday

/-- The conversation day is the same for all boys -/
def conversationDay (day : DayOfWeek) (schedules : List VisitSchedule) : Prop :=
  ∀ s ∈ schedules, ∃ n : ℕ, s n = day

/-- Main theorem: The conversation occurred on a Thursday -/
theorem conversation_on_thursday
  (library_schedule : LibrarySchedule)
  (boy_schedules : List VisitSchedule)
  (h_closed : libraryClosedWednesday library_schedule)
  (h_intervals : boy_schedules.length = visitIntervals.length)
  (h_monday : metAgainOnMonday boy_schedules)
  (h_adjust : ∀ s ∈ boy_schedules, ∀ n : ℕ,
    library_schedule (s n) = false → s (n + 1) = DayOfWeek.Thursday) :
  conversationDay DayOfWeek.Thursday boy_schedules :=
sorry

end NUMINAMATH_CALUDE_conversation_on_thursday_l1501_150103


namespace NUMINAMATH_CALUDE_thirteen_sided_polygon_diagonals_l1501_150105

/-- The number of diagonals in a polygon with n sides. -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals connected to a single vertex in a polygon with n sides. -/
def diagonals_per_vertex (n : ℕ) : ℕ := n - 3

/-- The number of diagonals in a polygon with n sides where one vertex is not connected to any diagonal. -/
def diagonals_with_disconnected_vertex (n : ℕ) : ℕ :=
  diagonals n - diagonals_per_vertex n

theorem thirteen_sided_polygon_diagonals :
  diagonals_with_disconnected_vertex 13 = 55 := by
  sorry

#eval diagonals_with_disconnected_vertex 13

end NUMINAMATH_CALUDE_thirteen_sided_polygon_diagonals_l1501_150105


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l1501_150194

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l1501_150194


namespace NUMINAMATH_CALUDE_kitae_pencils_l1501_150107

def total_pens : ℕ := 12
def pencil_cost : ℕ := 1000
def pen_cost : ℕ := 1300
def total_spent : ℕ := 15000

theorem kitae_pencils (pencils : ℕ) (pens : ℕ) 
  (h1 : pencils + pens = total_pens)
  (h2 : pencil_cost * pencils + pen_cost * pens = total_spent) :
  pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_kitae_pencils_l1501_150107


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1501_150118

theorem polynomial_divisibility (n : ℕ) (hn : n > 0) :
  ∃ Q : Polynomial ℚ, (n^2 * X^(n+2) - (2*n^2 + 2*n - 1) * X^(n+1) + (n+1)^2 * X^n - X - 1) = (X - 1)^3 * Q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1501_150118


namespace NUMINAMATH_CALUDE_h_nonzero_l1501_150179

/-- A polynomial of degree 4 with four distinct roots, one of which is 0 -/
structure QuarticPolynomial where
  f : ℝ
  g : ℝ
  h : ℝ
  roots : Finset ℝ
  distinct_roots : roots.card = 4
  zero_root : (0 : ℝ) ∈ roots
  is_root (x : ℝ) : x ∈ roots → x^4 + f*x^3 + g*x^2 + h*x = 0

theorem h_nonzero (Q : QuarticPolynomial) : Q.h ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_h_nonzero_l1501_150179


namespace NUMINAMATH_CALUDE_scientific_notation_700_3_l1501_150164

/-- Definition of scientific notation -/
def is_scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

/-- Theorem: 700.3 in scientific notation -/
theorem scientific_notation_700_3 :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation 700.3 a n ∧ a = 7.003 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_700_3_l1501_150164


namespace NUMINAMATH_CALUDE_trailing_zeros_500_factorial_l1501_150156

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- Definition to count trailing zeros -/
def trailingZeros (n : ℕ) : ℕ :=
  Nat.log 10 (Nat.gcd n (10^(Nat.log 2 n + 1)))

/-- Theorem: The number of trailing zeros in 500! is 124 -/
theorem trailing_zeros_500_factorial :
  trailingZeros (factorial 500) = 124 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_500_factorial_l1501_150156


namespace NUMINAMATH_CALUDE_S_min_at_24_l1501_150111

/-- The sequence term a_n -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- The sum S_n of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n ^ 2 - 48 * n

/-- Theorem stating that S_n is minimized when n = 24 -/
theorem S_min_at_24 : ∀ n : ℕ, S 24 ≤ S n := by sorry

end NUMINAMATH_CALUDE_S_min_at_24_l1501_150111


namespace NUMINAMATH_CALUDE_black_cards_remaining_l1501_150161

/-- Represents a standard deck of playing cards -/
structure Deck where
  total_cards : Nat
  black_cards : Nat
  removed_cards : Nat

/-- Theorem: The number of black cards remaining in a standard deck after removing 4 black cards is 22 -/
theorem black_cards_remaining (d : Deck) 
  (h1 : d.total_cards = 52)
  (h2 : d.black_cards = 26)
  (h3 : d.removed_cards = 4) :
  d.black_cards - d.removed_cards = 22 := by
  sorry

end NUMINAMATH_CALUDE_black_cards_remaining_l1501_150161


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_pow6_l1501_150148

theorem nearest_integer_to_3_plus_sqrt2_pow6 :
  ∃ n : ℤ, n = 7414 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - (m : ℝ)| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_pow6_l1501_150148


namespace NUMINAMATH_CALUDE_total_tickets_is_91_l1501_150172

/-- The total number of tickets needed for Janet's family's amusement park visits -/
def total_tickets : ℕ :=
  let family_size : ℕ := 4
  let adults : ℕ := 2
  let children : ℕ := 2
  let roller_coaster_adult : ℕ := 7
  let roller_coaster_child : ℕ := 5
  let giant_slide_adult : ℕ := 4
  let giant_slide_child : ℕ := 3
  let adult_roller_coaster_rides : ℕ := 3
  let child_roller_coaster_rides : ℕ := 2
  let adult_giant_slide_rides : ℕ := 5
  let child_giant_slide_rides : ℕ := 3

  let roller_coaster_tickets := 
    adults * roller_coaster_adult * adult_roller_coaster_rides +
    children * roller_coaster_child * child_roller_coaster_rides
  
  let giant_slide_tickets :=
    1 * giant_slide_adult * adult_giant_slide_rides +
    1 * giant_slide_child * child_giant_slide_rides

  roller_coaster_tickets + giant_slide_tickets

theorem total_tickets_is_91 : total_tickets = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_91_l1501_150172


namespace NUMINAMATH_CALUDE_range_of_a_l1501_150195

/-- The function f(x) = x|x^2 - 12| -/
def f (x : ℝ) : ℝ := x * abs (x^2 - 12)

theorem range_of_a (m : ℝ) (h_m : m > 0) :
  (∃ (a : ℝ), ∀ (y : ℝ), y ∈ Set.range (fun x => f x) ↔ y ∈ Set.Icc 0 (a * m^2)) →
  (∃ (a : ℝ), a ≥ 1 ∧ ∀ (b : ℝ), b ≥ 1 → ∃ (m : ℝ), m > 0 ∧
    (∀ (y : ℝ), y ∈ Set.range (fun x => f x) ↔ y ∈ Set.Icc 0 (b * m^2))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1501_150195


namespace NUMINAMATH_CALUDE_train_length_train_length_approx_l1501_150170

/-- The length of a train given its speed and time to cross a post -/
theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  speed_m_s * time_seconds

/-- Theorem stating that a train with speed 40 km/hr crossing a post in 25.2 seconds has a length of approximately 280 meters -/
theorem train_length_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |train_length 40 25.2 - 280| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_approx_l1501_150170


namespace NUMINAMATH_CALUDE_john_widget_production_rate_l1501_150180

/-- Represents the number of widgets John can make in an hour -/
def widgets_per_hour : ℕ := 20

/-- Represents the number of hours John works per day -/
def hours_per_day : ℕ := 8

/-- Represents the number of days John works per week -/
def days_per_week : ℕ := 5

/-- Represents the total number of widgets John makes in a week -/
def widgets_per_week : ℕ := 800

/-- Proves that the number of widgets John can make in an hour is 20 -/
theorem john_widget_production_rate : 
  widgets_per_hour * (hours_per_day * days_per_week) = widgets_per_week :=
by sorry

end NUMINAMATH_CALUDE_john_widget_production_rate_l1501_150180


namespace NUMINAMATH_CALUDE_a_capital_is_15000_l1501_150183

/-- The amount of money partner a put into the business -/
def a_capital : ℝ := sorry

/-- The amount of money partner b put into the business -/
def b_capital : ℝ := 25000

/-- The total profit of the business -/
def total_profit : ℝ := 9600

/-- The percentage of profit a receives for managing the business -/
def management_fee_percentage : ℝ := 0.1

/-- The total amount a receives -/
def a_total_received : ℝ := 4200

theorem a_capital_is_15000 :
  a_capital = 15000 :=
by
  sorry

#check a_capital_is_15000

end NUMINAMATH_CALUDE_a_capital_is_15000_l1501_150183


namespace NUMINAMATH_CALUDE_english_chinese_difference_l1501_150168

def hours_english : ℕ := 6
def hours_chinese : ℕ := 3

theorem english_chinese_difference : hours_english - hours_chinese = 3 := by
  sorry

end NUMINAMATH_CALUDE_english_chinese_difference_l1501_150168


namespace NUMINAMATH_CALUDE_prism_with_18_edges_has_8_faces_l1501_150150

/-- The number of faces in a prism given the number of edges -/
def prism_faces (edges : ℕ) : ℕ :=
  (edges / 3) + 2

theorem prism_with_18_edges_has_8_faces :
  prism_faces 18 = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_18_edges_has_8_faces_l1501_150150


namespace NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l1501_150135

theorem product_of_numbers_with_sum_and_difference 
  (x y : ℝ) (sum_eq : x + y = 60) (diff_eq : x - y = 10) : x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l1501_150135


namespace NUMINAMATH_CALUDE_sum_of_integers_l1501_150198

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 7)
  (eq2 : q - r + s = 8)
  (eq3 : r - s + p = 4)
  (eq4 : s - p + q = 3) :
  p + q + r + s = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1501_150198


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1501_150152

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1501_150152


namespace NUMINAMATH_CALUDE_cubic_equation_transformation_l1501_150114

theorem cubic_equation_transformation (p q r : ℝ) : 
  (p^3 - 5*p^2 + 6*p - 7 = 0) → 
  (q^3 - 5*q^2 + 6*q - 7 = 0) → 
  (r^3 - 5*r^2 + 6*r - 7 = 0) → 
  (∀ x : ℝ, x^3 - 10*x^2 + 25*x + 105 = 0 ↔ 
    (x = (p + q + r)/(p - 1) ∨ x = (p + q + r)/(q - 1) ∨ x = (p + q + r)/(r - 1))) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_transformation_l1501_150114


namespace NUMINAMATH_CALUDE_min_sum_squares_l1501_150173

theorem min_sum_squares (x y z t : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : 0 ≤ t)
  (h5 : |x - y| + |y - z| + |z - t| + |t - x| = 4) :
  2 ≤ x^2 + y^2 + z^2 + t^2 ∧ ∃ (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    |a - b| + |b - c| + |c - d| + |d - a| = 4 ∧ a^2 + b^2 + c^2 + d^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1501_150173


namespace NUMINAMATH_CALUDE_function_inequality_l1501_150128

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f(-x₁) > f(-x₂) -/
theorem function_inequality (f : ℝ → ℝ) (x₁ x₂ : ℝ)
  (h1 : ∀ x, f (x + 1) = f (-x - 1))
  (h2 : ∀ x₁ x₂, x₁ ≥ 1 ∧ x₂ ≥ 1 ∧ x₁ < x₂ → f x₁ < f x₂)
  (h3 : x₁ < 0)
  (h4 : x₂ > 0)
  (h5 : x₁ + x₂ < -2) :
  f (-x₁) > f (-x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1501_150128


namespace NUMINAMATH_CALUDE_opposite_silver_is_orange_l1501_150192

-- Define the colors
inductive Color
| Orange | Blue | Yellow | Black | Silver | Pink

-- Define the cube faces
inductive Face
| Top | Bottom | Front | Back | Left | Right

-- Define the cube structure
structure Cube where
  color : Face → Color

-- Define the three views of the cube
def view1 (c : Cube) : Prop :=
  c.color Face.Top = Color.Black ∧
  c.color Face.Front = Color.Blue ∧
  c.color Face.Right = Color.Yellow

def view2 (c : Cube) : Prop :=
  c.color Face.Top = Color.Black ∧
  c.color Face.Front = Color.Pink ∧
  c.color Face.Right = Color.Yellow

def view3 (c : Cube) : Prop :=
  c.color Face.Top = Color.Black ∧
  c.color Face.Front = Color.Silver ∧
  c.color Face.Right = Color.Yellow

-- Define the theorem
theorem opposite_silver_is_orange (c : Cube) :
  view1 c → view2 c → view3 c →
  c.color Face.Back = Color.Orange :=
sorry

end NUMINAMATH_CALUDE_opposite_silver_is_orange_l1501_150192


namespace NUMINAMATH_CALUDE_triangle_angles_l1501_150185

theorem triangle_angles (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) * (a + b - c) = 3 * a * b →
  Real.sin A ^ 2 = Real.sin B ^ 2 + Real.sin C ^ 2 →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  A = π / 6 ∧ B = π / 3 ∧ C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_l1501_150185


namespace NUMINAMATH_CALUDE_inequality_proof_l1501_150147

theorem inequality_proof (x y z : ℝ) 
  (h1 : y > 2*z) 
  (h2 : 2*z > 4*x) 
  (h3 : 2*(x^3 + y^3 + z^3) + 15*(x*y^2 + y*z^2 + z*x^2) > 16*(x^2*y + y^2*z + z^2*x) + 2*x*y*z) : 
  4*x + y > 4*z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1501_150147


namespace NUMINAMATH_CALUDE_point_on_same_side_l1501_150169

def sameSideOfLine (p1 p2 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x1 + y1 - 1) * (x2 + y2 - 1) > 0

def referencePt : ℝ × ℝ := (1, 2)

theorem point_on_same_side : 
  sameSideOfLine (-1, 3) referencePt ∧ 
  ¬sameSideOfLine (0, 0) referencePt ∧ 
  ¬sameSideOfLine (-1, 1) referencePt ∧ 
  ¬sameSideOfLine (2, -3) referencePt :=
by sorry

end NUMINAMATH_CALUDE_point_on_same_side_l1501_150169
