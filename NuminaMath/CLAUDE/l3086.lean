import Mathlib

namespace max_value_quadratic_l3086_308607

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 4) :
  ∃ (y : ℝ), y = x * (8 - 2 * x) ∧ ∀ (z : ℝ), z = x * (8 - 2 * x) → z ≤ y ∧ y = 8 := by
  sorry

end max_value_quadratic_l3086_308607


namespace missing_digit_divisible_by_three_l3086_308603

theorem missing_digit_divisible_by_three (x : Nat) :
  x < 10 →
  (1357 * 10 + x) * 10 + 2 % 3 = 0 →
  x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
  sorry

end missing_digit_divisible_by_three_l3086_308603


namespace triangle_side_length_l3086_308650

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- sides

-- State the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (ha : t.a = Real.sqrt 5) 
  (hc : t.c = 2) 
  (hcosA : Real.cos t.A = 2/3) : 
  t.b = 3 := by
  sorry

end triangle_side_length_l3086_308650


namespace range_of_a_l3086_308643

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 :=
by sorry

end range_of_a_l3086_308643


namespace number_of_boys_l3086_308669

theorem number_of_boys (total_amount : ℕ) (total_children : ℕ) (boy_amount : ℕ) (girl_amount : ℕ) 
  (h1 : total_amount = 460)
  (h2 : total_children = 41)
  (h3 : boy_amount = 12)
  (h4 : girl_amount = 8) :
  ∃ (boys : ℕ), boys = 33 ∧ 
    boys * boy_amount + (total_children - boys) * girl_amount = total_amount :=
by sorry

end number_of_boys_l3086_308669


namespace product_of_two_fifteens_l3086_308601

theorem product_of_two_fifteens : ∀ (a b : ℕ), a = 15 → b = 15 → a * b = 225 := by
  sorry

end product_of_two_fifteens_l3086_308601


namespace sum_to_135_mod_7_l3086_308699

/-- The sum of integers from 1 to n -/
def sum_to (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that the sum of integers from 1 to 135, when divided by 7, has a remainder of 3 -/
theorem sum_to_135_mod_7 : sum_to 135 % 7 = 3 := by sorry

end sum_to_135_mod_7_l3086_308699


namespace ratio_problem_l3086_308651

theorem ratio_problem (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 := by
  sorry

end ratio_problem_l3086_308651


namespace bad_carrots_l3086_308655

/-- The number of bad carrots in Carol and her mother's carrot picking scenario -/
theorem bad_carrots (carol_carrots : ℕ) (mother_carrots : ℕ) (good_carrots : ℕ) : 
  carol_carrots = 29 → mother_carrots = 16 → good_carrots = 38 →
  carol_carrots + mother_carrots - good_carrots = 7 := by
  sorry

end bad_carrots_l3086_308655


namespace problem_solution_l3086_308673

theorem problem_solution : 
  (1 + 3/4 - 3/8 + 5/6) / (-1/24) = -53 ∧ 
  -2^2 + (-4) / 2 * (1/2) + |(-3)| = -2 := by
  sorry

end problem_solution_l3086_308673


namespace min_l_shapes_5x5_grid_l3086_308663

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- An L-shaped figure made of 3 cells --/
structure LShape where
  x : Fin 5
  y : Fin 5
  orientation : Fin 4

/-- Check if an L-shape is within the grid bounds --/
def LShape.isValid (l : LShape) : Bool :=
  match l.orientation with
  | 0 => l.x < 4 ∧ l.y < 4
  | 1 => l.x > 0 ∧ l.y < 4
  | 2 => l.x < 4 ∧ l.y > 0
  | 3 => l.x > 0 ∧ l.y > 0

/-- Check if two L-shapes overlap --/
def LShape.overlaps (l1 l2 : LShape) : Bool :=
  sorry

/-- Check if a set of L-shapes is valid (non-overlapping and within bounds) --/
def isValidPlacement (shapes : List LShape) : Bool :=
  sorry

/-- Check if no more L-shapes can be added to a given set of shapes --/
def isMaximalPlacement (shapes : List LShape) : Bool :=
  sorry

/-- The main theorem --/
theorem min_l_shapes_5x5_grid :
  ∃ (shapes : List LShape),
    shapes.length = 4 ∧
    isValidPlacement shapes ∧
    isMaximalPlacement shapes ∧
    ∀ (otherShapes : List LShape),
      isValidPlacement otherShapes ∧ isMaximalPlacement otherShapes →
      otherShapes.length ≥ 4 :=
  sorry

end min_l_shapes_5x5_grid_l3086_308663


namespace quadratic_always_positive_l3086_308667

-- Define the quadratic expression
def quadratic (k x : ℝ) : ℝ := x^2 - (k - 4)*x - k + 7

-- State the theorem
theorem quadratic_always_positive (k : ℝ) :
  (∀ x, quadratic k x > 0) ↔ k > -2 ∧ k < 6 := by
  sorry

end quadratic_always_positive_l3086_308667


namespace right_triangle_perimeter_l3086_308698

theorem right_triangle_perimeter : ∃ (a c : ℕ), 
  11^2 + a^2 = c^2 ∧ 11 + a + c = 132 := by
  sorry

end right_triangle_perimeter_l3086_308698


namespace smallest_cube_ending_528_l3086_308662

theorem smallest_cube_ending_528 :
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 528 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 528 → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_cube_ending_528_l3086_308662


namespace dolphin_training_hours_l3086_308638

/-- Calculates the number of hours each trainer spends training dolphins -/
def trainer_hours (num_dolphins : ℕ) (hours_per_dolphin : ℕ) (num_trainers : ℕ) : ℕ :=
  (num_dolphins * hours_per_dolphin) / num_trainers

theorem dolphin_training_hours :
  trainer_hours 4 3 2 = 6 := by
  sorry

end dolphin_training_hours_l3086_308638


namespace birds_on_fence_l3086_308636

theorem birds_on_fence (initial_birds additional_birds : ℕ) : 
  initial_birds = 2 → additional_birds = 4 → initial_birds + additional_birds = 6 := by
sorry

end birds_on_fence_l3086_308636


namespace square_sum_equality_l3086_308617

theorem square_sum_equality (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end square_sum_equality_l3086_308617


namespace real_part_of_complex_fraction_l3086_308606

theorem real_part_of_complex_fraction : 
  Complex.re (5 / (1 - Complex.I * 2)) = 1 := by sorry

end real_part_of_complex_fraction_l3086_308606


namespace leftover_seashells_proof_l3086_308680

/-- The number of leftover seashells after packaging -/
def leftover_seashells (derek_shells : ℕ) (emily_shells : ℕ) (fiona_shells : ℕ) (package_size : ℕ) : ℕ :=
  (derek_shells + emily_shells + fiona_shells) % package_size

theorem leftover_seashells_proof (derek_shells emily_shells fiona_shells package_size : ℕ) 
  (h_package_size : package_size > 0) :
  leftover_seashells derek_shells emily_shells fiona_shells package_size = 
  (derek_shells + emily_shells + fiona_shells) % package_size :=
by
  sorry

#eval leftover_seashells 58 73 31 10

end leftover_seashells_proof_l3086_308680


namespace factorial_difference_quotient_l3086_308605

theorem factorial_difference_quotient : (Nat.factorial 15 - Nat.factorial 14 - Nat.factorial 13) / Nat.factorial 11 = 30420 := by
  sorry

end factorial_difference_quotient_l3086_308605


namespace decimal_to_binary_34_l3086_308648

theorem decimal_to_binary_34 : 
  (34 : ℕ) = (1 * 2^5 + 0 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0 : ℕ) := by
  sorry

end decimal_to_binary_34_l3086_308648


namespace point_transformation_to_polar_coordinates_l3086_308641

theorem point_transformation_to_polar_coordinates :
  ∀ (x y : ℝ),
    2 * x = 6 ∧ Real.sqrt 3 * y = -3 →
    ∃ (ρ θ : ℝ),
      ρ = 2 * Real.sqrt 3 ∧
      θ = 11 * π / 6 ∧
      ρ > 0 ∧
      0 ≤ θ ∧ θ < 2 * π ∧
      x = ρ * Real.cos θ ∧
      y = ρ * Real.sin θ :=
by sorry

end point_transformation_to_polar_coordinates_l3086_308641


namespace student_selection_count_l3086_308624

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_selected : ℕ := 4

def select_students (b g s : ℕ) : ℕ :=
  (Nat.choose b 3 * Nat.choose g 1) +
  (Nat.choose b 2 * Nat.choose g 2) +
  (Nat.choose b 1 * Nat.choose g 3)

theorem student_selection_count :
  select_students num_boys num_girls total_selected = 34 := by
  sorry

end student_selection_count_l3086_308624


namespace power_of_two_multiplication_l3086_308677

theorem power_of_two_multiplication : 2^4 * 2^4 * 2^4 = 2^12 := by
  sorry

end power_of_two_multiplication_l3086_308677


namespace polynomial_properties_l3086_308674

variable (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)

-- Define the polynomial equality
def poly_eq (x : ℝ) : Prop :=
  (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5

-- Theorem statement
theorem polynomial_properties :
  (∀ x, poly_eq a₀ a₁ a₂ a₃ a₄ a₅ x) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) ∧
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) :=
by sorry

end polynomial_properties_l3086_308674


namespace divisors_of_2700_l3086_308693

def number_of_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_2700 : number_of_divisors 2700 = 36 := by sorry

end divisors_of_2700_l3086_308693


namespace cow_chicken_problem_l3086_308633

/-- Given a group of cows and chickens, if the total number of legs is 18 more than
    twice the total number of heads, then the number of cows is 9. -/
theorem cow_chicken_problem (cows chickens : ℕ) : 
  4 * cows + 2 * chickens = 2 * (cows + chickens) + 18 → cows = 9 := by
  sorry

end cow_chicken_problem_l3086_308633


namespace satellite_survey_is_census_l3086_308686

/-- Represents a survey type -/
inductive SurveyType
| Sample
| Census

/-- Represents a survey option -/
structure SurveyOption where
  description : String
  type : SurveyType

/-- Determines if a survey option is suitable for a census -/
def isSuitableForCensus (survey : SurveyOption) : Prop :=
  survey.type = SurveyType.Census

/-- The satellite component quality survey -/
def satelliteComponentSurvey : SurveyOption :=
  { description := "Investigating the quality of components of the satellite \"Zhangheng-1\""
    type := SurveyType.Census }

/-- Theorem stating that the satellite component survey is suitable for a census -/
theorem satellite_survey_is_census : 
  isSuitableForCensus satelliteComponentSurvey := by
  sorry


end satellite_survey_is_census_l3086_308686


namespace unique_solution_l3086_308653

theorem unique_solution (x y z : ℝ) 
  (hx : x > 2) (hy : y > 2) (hz : z > 2)
  (heq : ((x + 3)^2) / (y + z - 3) + ((y + 5)^2) / (z + x - 5) + ((z + 7)^2) / (x + y - 7) = 45) :
  x = 13 ∧ y = 11 ∧ z = 6 := by
sorry

end unique_solution_l3086_308653


namespace line_chart_appropriate_for_temperature_over_week_l3086_308635

-- Define the types of charts
inductive ChartType
| Bar
| Line
| Pie

-- Define the characteristics of the data
structure TemperatureData :=
  (measurements : List Float)
  (timePoints : List String)
  (duration : Nat)

-- Define the requirements for the chart
structure ChartRequirements :=
  (showQuantity : Bool)
  (showChangeOverTime : Bool)

-- Define a function to determine the appropriate chart type
def appropriateChartType (data : TemperatureData) (req : ChartRequirements) : ChartType :=
  sorry

-- Theorem statement
theorem line_chart_appropriate_for_temperature_over_week :
  ∀ (data : TemperatureData) (req : ChartRequirements),
    data.duration = 7 →
    req.showQuantity = true →
    req.showChangeOverTime = true →
    appropriateChartType data req = ChartType.Line :=
  sorry

end line_chart_appropriate_for_temperature_over_week_l3086_308635


namespace zero_power_is_zero_l3086_308695

theorem zero_power_is_zero (n : ℕ) : 0^n = 0 := by
  sorry

end zero_power_is_zero_l3086_308695


namespace chocolate_bars_in_large_box_l3086_308664

theorem chocolate_bars_in_large_box :
  let small_boxes : ℕ := 15
  let bars_per_small_box : ℕ := 25
  let total_bars : ℕ := small_boxes * bars_per_small_box
  total_bars = 375 := by
  sorry

end chocolate_bars_in_large_box_l3086_308664


namespace x_squared_geq_one_necessary_not_sufficient_for_x_geq_one_l3086_308600

theorem x_squared_geq_one_necessary_not_sufficient_for_x_geq_one :
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ x < 1) := by
sorry

end x_squared_geq_one_necessary_not_sufficient_for_x_geq_one_l3086_308600


namespace inequality_proof_l3086_308654

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1/2) :
  (1/(1-a)) + (1/(1-b)) ≥ 4 ∧ ((1/(1-a)) + (1/(1-b)) = 4 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end inequality_proof_l3086_308654


namespace arithmetic_computation_l3086_308616

theorem arithmetic_computation : -12 * 3 - (-4 * -5) + (-8 * -6) + 2 = -6 := by
  sorry

end arithmetic_computation_l3086_308616


namespace fibLastDigitsCyclic_l3086_308679

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Sequence of last digits of Fibonacci numbers -/
def fibLastDigits : ℕ → ℕ := λ n => lastDigit (fib n)

/-- Period of a sequence -/
def isPeriodic (f : ℕ → ℕ) (p : ℕ) : Prop :=
  ∀ n, f (n + p) = f n

/-- Theorem: The sequence of last digits of Fibonacci numbers is cyclic -/
theorem fibLastDigitsCyclic : ∃ p : ℕ, p > 0 ∧ isPeriodic fibLastDigits p :=
  sorry

end fibLastDigitsCyclic_l3086_308679


namespace odd_periodic_function_property_l3086_308682

open Real

-- Define the properties of the function f
def is_odd_and_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 3) = -f x)

-- State the theorem
theorem odd_periodic_function_property 
  (f : ℝ → ℝ) 
  (α : ℝ) 
  (h_f : is_odd_and_periodic f) 
  (h_α : tan α = 2) : 
  f (15 * sin α * cos α) = 0 := by
sorry

end odd_periodic_function_property_l3086_308682


namespace piecewise_function_sum_l3086_308622

theorem piecewise_function_sum (f : ℝ → ℝ) (a b c : ℤ) : 
  (∀ x > 0, f x = a * x + b) →
  (∀ x < 0, f x = b * x + c) →
  (f 0 = a * b) →
  (f 2 = 7) →
  (f 0 = 1) →
  (f (-2) = -8) →
  a + b + c = 8 := by
sorry

end piecewise_function_sum_l3086_308622


namespace distribute_six_balls_three_boxes_l3086_308690

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 5 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 5 := by
  sorry

end distribute_six_balls_three_boxes_l3086_308690


namespace parabola_intersection_circle_radius_squared_l3086_308685

theorem parabola_intersection_circle_radius_squared (x y : ℝ) : 
  y = (x - 2)^2 ∧ x + 6 = (y - 5)^2 → 
  (x - 5/2)^2 + (y - 9/2)^2 = 83/4 :=
by sorry

end parabola_intersection_circle_radius_squared_l3086_308685


namespace min_digits_to_remove_l3086_308645

def original_number : ℕ := 123454321

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

def sum_of_digits (n : ℕ) : ℕ :=
  (digits n).sum

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def remove_digits (n : ℕ) (indices : List ℕ) : ℕ :=
  let digits := digits n
  let new_digits := (List.enum digits).filter (λ (i, _) => ¬ indices.contains i)
  new_digits.foldl (λ acc (_, d) => acc * 10 + d) 0

theorem min_digits_to_remove :
  ∃ (indices : List ℕ),
    indices.length = 2 ∧
    is_divisible_by_9 (remove_digits original_number indices) ∧
    ∀ (other_indices : List ℕ),
      other_indices.length < 2 →
      ¬ is_divisible_by_9 (remove_digits original_number other_indices) :=
by sorry

end min_digits_to_remove_l3086_308645


namespace defective_shipped_percentage_l3086_308621

/-- Represents the percentage of defective units shipped from each stage -/
structure DefectiveShipped :=
  (stage1 : ℝ)
  (stage2 : ℝ)
  (stage3 : ℝ)

/-- Represents the percentage of defective units in each stage -/
structure DefectivePercentage :=
  (stage1 : ℝ)
  (stage2 : ℝ)
  (stage3 : ℝ)

/-- Represents the percentage of defective units shipped from each stage -/
structure ShippedPercentage :=
  (stage1 : ℝ)
  (stage2 : ℝ)
  (stage3 : ℝ)

/-- Calculates the percentage of total units that are defective and shipped -/
def calculate_defective_shipped (dp : DefectivePercentage) (sp : ShippedPercentage) : ℝ :=
  let ds : DefectiveShipped := {
    stage1 := dp.stage1 * sp.stage1,
    stage2 := (1 - dp.stage1) * dp.stage2 * sp.stage2,
    stage3 := (1 - dp.stage1) * (1 - dp.stage2) * dp.stage3 * sp.stage3
  }
  ds.stage1 + ds.stage2 + ds.stage3

/-- Theorem: Given the production process conditions, 2% of total units are defective and shipped -/
theorem defective_shipped_percentage :
  let dp : DefectivePercentage := { stage1 := 0.06, stage2 := 0.08, stage3 := 0.10 }
  let sp : ShippedPercentage := { stage1 := 0.05, stage2 := 0.07, stage3 := 0.10 }
  calculate_defective_shipped dp sp = 0.02 := by
  sorry


end defective_shipped_percentage_l3086_308621


namespace cotangent_sum_equality_l3086_308630

/-- Given a triangle ABC, A'B'C' is the triangle formed by its medians -/
def MedianTriangle (A B C : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The sum of cotangents of angles in a triangle -/
def SumOfCotangents (A B C : ℝ × ℝ) : ℝ := sorry

theorem cotangent_sum_equality (A B C : ℝ × ℝ) :
  let (A', B', C') := MedianTriangle A B C
  SumOfCotangents A B C = SumOfCotangents A' B' C' := by sorry

end cotangent_sum_equality_l3086_308630


namespace card_ratio_l3086_308661

/-- Prove that given the conditions in the problem, the ratio of football cards to hockey cards is 4:1 -/
theorem card_ratio (total_cards : ℕ) (hockey_cards : ℕ) (s : ℕ) :
  total_cards = 1750 →
  hockey_cards = 200 →
  total_cards = (s * hockey_cards - 50) + (s * hockey_cards) + hockey_cards →
  (s * hockey_cards) / hockey_cards = 4 :=
by sorry

end card_ratio_l3086_308661


namespace michael_born_in_1979_l3086_308675

/-- The year of the first AMC 8 -/
def first_amc8_year : ℕ := 1985

/-- The number of AMC 8 competitions Michael has taken -/
def michaels_amc8_number : ℕ := 10

/-- Michael's age when he took his AMC 8 -/
def michaels_age : ℕ := 15

/-- Function to calculate the year of a given AMC 8 competition -/
def amc8_year (n : ℕ) : ℕ := first_amc8_year + n - 1

/-- Michael's birth year -/
def michaels_birth_year : ℕ := amc8_year michaels_amc8_number - michaels_age

theorem michael_born_in_1979 : michaels_birth_year = 1979 := by
  sorry

end michael_born_in_1979_l3086_308675


namespace equation_solution_verify_solution_l3086_308613

/-- The solution to the equation √((3x-1)/(x+4)) + 3 - 4√((x+4)/(3x-1)) = 0 -/
theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3 * x - 1)) = 0) ∧
           x = 5 / 2 := by
  sorry

/-- Verification that 5/2 is indeed the solution -/
theorem verify_solution :
  let x : ℝ := 5 / 2
  Real.sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3 * x - 1)) = 0 := by
  sorry

end equation_solution_verify_solution_l3086_308613


namespace six_n_divisors_l3086_308610

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem six_n_divisors (n : ℕ) 
  (h1 : divisor_count n = 10)
  (h2 : divisor_count (2 * n) = 20)
  (h3 : divisor_count (3 * n) = 15) :
  divisor_count (6 * n) = 30 := by
  sorry

end six_n_divisors_l3086_308610


namespace remainder_of_3_600_mod_19_l3086_308649

theorem remainder_of_3_600_mod_19 : 3^600 % 19 = 11 := by
  sorry

end remainder_of_3_600_mod_19_l3086_308649


namespace area_product_eq_volume_squared_l3086_308646

/-- Represents a rectangular box with dimensions x, y, and z, and diagonal d. -/
structure RectBox where
  x : ℝ
  y : ℝ
  z : ℝ
  d : ℝ
  h_positive : x > 0 ∧ y > 0 ∧ z > 0
  h_diagonal : d^2 = x^2 + y^2 + z^2

/-- The volume of a rectangular box. -/
def volume (box : RectBox) : ℝ := box.x * box.y * box.z

/-- The product of the areas of the bottom, side, and front of a rectangular box. -/
def areaProduct (box : RectBox) : ℝ := (box.x * box.y) * (box.y * box.z) * (box.z * box.x)

/-- Theorem stating that the product of the areas is equal to the square of the volume. -/
theorem area_product_eq_volume_squared (box : RectBox) :
  areaProduct box = (volume box)^2 := by sorry

end area_product_eq_volume_squared_l3086_308646


namespace negative_five_minus_two_i_in_third_quadrant_l3086_308611

/-- A complex number z is in the third quadrant if its real part is negative and its imaginary part is negative. -/
def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- The complex number -5-2i is in the third quadrant. -/
theorem negative_five_minus_two_i_in_third_quadrant :
  is_in_third_quadrant (-5 - 2*I) := by
  sorry

end negative_five_minus_two_i_in_third_quadrant_l3086_308611


namespace saree_discount_problem_l3086_308625

/-- Proves that the first discount percentage is 20% given the conditions of the saree pricing problem --/
theorem saree_discount_problem (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 350 →
  final_price = 266 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 0.20 := by
  sorry

end saree_discount_problem_l3086_308625


namespace minimum_c_value_l3086_308642

-- Define the curve
def on_curve (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the inequality condition
def inequality_holds (c : ℝ) : Prop := ∀ x y : ℝ, on_curve x y → x + y + c ≥ 0

-- State the theorem
theorem minimum_c_value : 
  (∃ c_min : ℝ, (∀ c : ℝ, c ≥ c_min ↔ inequality_holds c) ∧ c_min = Real.sqrt 2 - 1) :=
sorry

end minimum_c_value_l3086_308642


namespace first_day_income_l3086_308618

/-- Given a sequence where each term is double the previous term,
    and the 10th term is 18, prove that the first term is 0.03515625 -/
theorem first_day_income (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 2 * a n) (h2 : a 10 = 18) :
  a 1 = 0.03515625 := by
  sorry

end first_day_income_l3086_308618


namespace domain_of_f_l3086_308672

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (2 * x + 1) / Real.log (1/2))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/2 < x ∧ x ≠ 0} :=
by sorry

end domain_of_f_l3086_308672


namespace spending_ratio_l3086_308683

def initial_amount : ℕ := 200
def spent_on_books : ℕ := 30
def spent_on_clothes : ℕ := 55
def spent_on_snacks : ℕ := 25
def spent_on_gift : ℕ := 20
def spent_on_electronics : ℕ := 40

def total_spent : ℕ := spent_on_books + spent_on_clothes + spent_on_snacks + spent_on_gift + spent_on_electronics
def unspent : ℕ := initial_amount - total_spent

theorem spending_ratio : 
  (total_spent : ℚ) / (unspent : ℚ) = 17 / 3 := by sorry

end spending_ratio_l3086_308683


namespace fifteenth_clap_theorem_l3086_308623

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  circumference : ℝ
  a_lap_time : ℝ
  b_lap_time : ℝ
  a_reverse_laps : ℕ

/-- Calculates the time and distance for A and B to clap hands 15 times -/
def clap_hands_15_times (track : CircularTrack) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct time and distance for the 15th clap -/
theorem fifteenth_clap_theorem (track : CircularTrack) 
  (h1 : track.circumference = 400)
  (h2 : track.a_lap_time = 4)
  (h3 : track.b_lap_time = 7)
  (h4 : track.a_reverse_laps = 10) :
  let (time, distance) := clap_hands_15_times track
  time = 66 + 2/11 ∧ distance = 3781 + 9/11 := by
  sorry

end fifteenth_clap_theorem_l3086_308623


namespace logarithm_power_sum_l3086_308668

theorem logarithm_power_sum (a b : ℝ) (ha : a = Real.log 25) (hb : b = Real.log 36) :
  (5 : ℝ) ^ (a / b) + (6 : ℝ) ^ (b / a) = 11 := by
  sorry

end logarithm_power_sum_l3086_308668


namespace hair_group_existence_l3086_308609

theorem hair_group_existence (population : ℕ) (max_hairs : ℕ) 
  (h1 : population ≥ 8000000) 
  (h2 : max_hairs = 400000) : 
  ∃ (hair_count : ℕ), hair_count ≤ max_hairs ∧ 
  (∃ (group : Finset (Fin population)), 
    group.card ≥ 20 ∧ 
    ∀ (person : Fin population), person ∈ group → 
      (∃ (f : Fin population → ℕ), f person = hair_count ∧ f person ≤ max_hairs)) :=
sorry

end hair_group_existence_l3086_308609


namespace right_triangle_area_l3086_308626

theorem right_triangle_area (a b : ℝ) (h1 : a = 24) (h2 : b = 30) : 
  (1/2 : ℝ) * a * b = 360 := by
  sorry

end right_triangle_area_l3086_308626


namespace kitten_growth_ratio_l3086_308619

/-- Given the initial length and final length of a kitten, and knowing that the final length is twice the intermediate length, prove that the ratio of intermediate length to initial length is 2. -/
theorem kitten_growth_ratio (L₀ L₂ L₄ : ℝ) (h₀ : L₀ = 4) (h₄ : L₄ = 16) (h_double : L₄ = 2 * L₂) : L₂ / L₀ = 2 := by
  sorry

end kitten_growth_ratio_l3086_308619


namespace base_prime_repr_360_l3086_308615

/-- Base prime representation of a natural number -/
def BasePrimeRepr : ℕ → List ℕ := sorry

/-- Check if a list represents a valid base prime representation -/
def IsValidBasePrimeRepr (l : List ℕ) : Prop := sorry

theorem base_prime_repr_360 :
  let repr := BasePrimeRepr 360
  IsValidBasePrimeRepr repr ∧ repr = [3, 2, 1] := by sorry

end base_prime_repr_360_l3086_308615


namespace floor_abs_negative_real_l3086_308665

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end floor_abs_negative_real_l3086_308665


namespace fence_perimeter_l3086_308659

/-- The number of posts in the fence -/
def total_posts : ℕ := 24

/-- The width of each post in inches -/
def post_width : ℚ := 5

/-- The space between adjacent posts in feet -/
def post_spacing : ℚ := 6

/-- The number of posts on each side of the square fence -/
def posts_per_side : ℕ := 7

/-- The length of one side of the square fence in feet -/
def side_length : ℚ := post_spacing * 6 + posts_per_side * (post_width / 12)

/-- The outer perimeter of the square fence in feet -/
def outer_perimeter : ℚ := 4 * side_length

theorem fence_perimeter : outer_perimeter = 156 := by sorry

end fence_perimeter_l3086_308659


namespace base_conversion_sum_l3086_308647

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [2, 5, 3]
def base1 : Nat := 8
def den1 : List Nat := [1, 3]
def base2 : Nat := 4
def num2 : List Nat := [1, 4, 4]
def base3 : Nat := 5
def den2 : List Nat := [2, 2]
def base4 : Nat := 3

-- State the theorem
theorem base_conversion_sum :
  (baseToDecimal num1 base1 : Rat) / (baseToDecimal den1 base2 : Rat) +
  (baseToDecimal num2 base3 : Rat) / (baseToDecimal den2 base4 : Rat) =
  30.125 := by sorry

end base_conversion_sum_l3086_308647


namespace units_digit_product_l3086_308632

theorem units_digit_product (n : ℕ) : (2^2023 * 5^2024 * 11^2025) % 10 = 0 := by
  sorry

end units_digit_product_l3086_308632


namespace trio_ball_theorem_l3086_308631

/-- The number of minutes each child plays in the trio-ball game -/
def trio_ball_play_time (total_time : ℕ) (num_children : ℕ) (players_per_game : ℕ) : ℕ :=
  (total_time * players_per_game) / num_children

/-- Theorem stating that each child plays for 60 minutes in the given scenario -/
theorem trio_ball_theorem :
  trio_ball_play_time 120 6 3 = 60 := by
  sorry

#eval trio_ball_play_time 120 6 3

end trio_ball_theorem_l3086_308631


namespace negation_of_square_nonnegative_l3086_308678

theorem negation_of_square_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end negation_of_square_nonnegative_l3086_308678


namespace lamp_post_ratio_l3086_308629

theorem lamp_post_ratio (k m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 9 * x = k ∧ 99 * x = m) → m / k = 11 := by
  sorry

end lamp_post_ratio_l3086_308629


namespace largest_prime_factor_l3086_308671

def numbers : List Nat := [55, 63, 85, 94, 133]

def has_largest_prime_factor (n : Nat) (lst : List Nat) : Prop :=
  ∀ m ∈ lst, ∃ p q : Nat, 
    Nat.Prime p ∧ 
    n = p * q ∧ 
    ∀ r s : Nat, (Nat.Prime r ∧ m = r * s) → r ≤ p

theorem largest_prime_factor : 
  has_largest_prime_factor 94 numbers := by
  sorry

end largest_prime_factor_l3086_308671


namespace triangle_polynomial_roots_l3086_308657

theorem triangle_polynomial_roots (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hac : a + c > b) (hbc : b + c > a) :
  ¬ (∃ x y : ℝ, x < 1/3 ∧ y < 1/3 ∧ a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) :=
by sorry

end triangle_polynomial_roots_l3086_308657


namespace simplify_trig_expression_l3086_308602

theorem simplify_trig_expression :
  2 * Real.sqrt (1 + Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4 - 4 * Real.cos 4 := by
  sorry

end simplify_trig_expression_l3086_308602


namespace evaluate_expression_l3086_308670

theorem evaluate_expression (x y : ℤ) (hx : x = 5) (hy : y = -3) :
  y * (y - 2 * x + 1) = 36 := by
  sorry

end evaluate_expression_l3086_308670


namespace lentil_dishes_count_l3086_308697

/-- Represents the menu of a vegan restaurant -/
structure VeganMenu :=
  (total_dishes : ℕ)
  (beans_lentils : ℕ)
  (beans_seitan : ℕ)
  (only_beans : ℕ)
  (only_seitan : ℕ)
  (only_lentils : ℕ)

/-- The conditions of the vegan restaurant menu problem -/
def menu_conditions (m : VeganMenu) : Prop :=
  m.total_dishes = 10 ∧
  m.beans_lentils = 2 ∧
  m.beans_seitan = 2 ∧
  m.only_beans = (m.total_dishes - m.beans_lentils - m.beans_seitan) / 2 ∧
  m.only_beans = 3 * m.only_seitan

/-- Theorem stating that the number of dishes including lentils is 2 -/
theorem lentil_dishes_count (m : VeganMenu) (h : menu_conditions m) : 
  m.beans_lentils + m.only_lentils = 2 := by
  sorry


end lentil_dishes_count_l3086_308697


namespace proposition_truths_l3086_308684

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := 
  (∃ (q : ℚ), a + b = q) → (∃ (r s : ℚ), a = r ∧ b = s)

def proposition2 (a b : ℝ) : Prop :=
  a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)

def proposition3 (a b : ℝ) : Prop :=
  ∀ x, a * x + b > 0 ↔ x > -b / a

def proposition4 (a b c : ℝ) : Prop :=
  (∃ x, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0

-- Theorem stating which propositions are true
theorem proposition_truths :
  (∃ a b : ℝ, ¬ proposition1 a b) ∧
  (∀ a b : ℝ, proposition2 a b) ∧
  (∃ a b : ℝ, ¬ proposition3 a b) ∧
  (∀ a b c : ℝ, proposition4 a b c) :=
sorry

end proposition_truths_l3086_308684


namespace abc_value_l3086_308628

theorem abc_value :
  let a := -(2017 * 2017 - 2017) / (2016 * 2016 + 2016)
  let b := -(2018 * 2018 - 2018) / (2017 * 2017 + 2017)
  let c := -(2019 * 2019 - 2019) / (2018 * 2018 + 2018)
  a * b * c = -1 := by
sorry

end abc_value_l3086_308628


namespace identity_value_l3086_308676

theorem identity_value (a b c : ℝ) (m n : ℤ) :
  (∀ x : ℝ, (x^n + c)^m = (a*x^m + 1)*(b*x^m + 1)) →
  |a + b + c| = 3 :=
by sorry

end identity_value_l3086_308676


namespace eight_cos_squared_25_minus_tan_40_minus_4_equals_sqrt_3_l3086_308681

theorem eight_cos_squared_25_minus_tan_40_minus_4_equals_sqrt_3 :
  8 * (Real.cos (25 * π / 180))^2 - Real.tan (40 * π / 180) - 4 = Real.sqrt 3 := by
  sorry

end eight_cos_squared_25_minus_tan_40_minus_4_equals_sqrt_3_l3086_308681


namespace two_row_arrangement_count_l3086_308612

/-- The number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.descFactorial n k

theorem two_row_arrangement_count
  (n k k₁ k₂ : ℕ)
  (h₁ : k₁ + k₂ = k)
  (h₂ : 1 ≤ k)
  (h₃ : k ≤ n) :
  (permutations n k₁) * (permutations (n - k₁) k₂) = permutations n k :=
sorry

end two_row_arrangement_count_l3086_308612


namespace fraction_evaluation_l3086_308620

theorem fraction_evaluation : 
  (1 / 5 - 1 / 7) / (3 / 8 + 2 / 9) = 144 / 1505 := by sorry

end fraction_evaluation_l3086_308620


namespace f_properties_l3086_308694

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) ^ (a * x^2 - 4*x + 3)

theorem f_properties :
  (∀ x > 2, ∀ y > x, f 1 y < f 1 x) ∧
  (∃ x, f 1 x = 2 → 1 = 1) ∧
  (∀ a, (∀ x < 2, ∀ y < x, f a y < f a x) → 0 ≤ a ∧ a ≤ 1) :=
sorry

end f_properties_l3086_308694


namespace max_value_theorem_l3086_308652

theorem max_value_theorem (x y z : ℝ) (h : 2*x + 3*y + 5*z = 29) :
  Real.sqrt (2*x + 1) + Real.sqrt (3*y + 4) + Real.sqrt (5*z + 6) ≤ 2 * Real.sqrt 30 := by
  sorry

end max_value_theorem_l3086_308652


namespace perpendicular_lines_k_value_l3086_308658

/-- A line in 2D space represented by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if two lines are perpendicular --/
def perpendicular (l1 l2 : ParametricLine) : Prop :=
  ∃ m1 m2 : ℝ, (∀ t : ℝ, l1.y t = m1 * l1.x t + (l1.y 0 - m1 * l1.x 0)) ∧
              (∀ s : ℝ, l2.y s = m2 * l2.x s + (l2.y 0 - m2 * l2.x 0)) ∧
              m1 * m2 = -1

theorem perpendicular_lines_k_value :
  ∀ k : ℝ,
  let l1 : ParametricLine := {
    x := λ t => 1 - 2*t,
    y := λ t => 2 + k*t
  }
  let l2 : ParametricLine := {
    x := λ s => s,
    y := λ s => 1 - 2*s
  }
  perpendicular l1 l2 → k = -1 := by
  sorry

#check perpendicular_lines_k_value

end perpendicular_lines_k_value_l3086_308658


namespace family_size_l3086_308660

def total_spent : ℕ := 119
def adult_ticket_price : ℕ := 21
def child_ticket_price : ℕ := 14
def adult_tickets_purchased : ℕ := 4

theorem family_size :
  ∃ (child_tickets : ℕ),
    adult_tickets_purchased * adult_ticket_price + child_tickets * child_ticket_price = total_spent ∧
    adult_tickets_purchased + child_tickets = 6 :=
by
  sorry

end family_size_l3086_308660


namespace mass_percentage_Ca_in_mixture_l3086_308627

/-- Molar mass of calcium in g/mol -/
def molar_mass_Ca : ℝ := 40.08

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- Molar mass of sulfur in g/mol -/
def molar_mass_S : ℝ := 32.07

/-- Molar mass of calcium oxide (CaO) in g/mol -/
def molar_mass_CaO : ℝ := molar_mass_Ca + molar_mass_O

/-- Molar mass of calcium carbonate (CaCO₃) in g/mol -/
def molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O

/-- Molar mass of calcium sulfate (CaSO₄) in g/mol -/
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

/-- Percentage of CaO in the mixed compound -/
def percent_CaO : ℝ := 40

/-- Percentage of CaCO₃ in the mixed compound -/
def percent_CaCO3 : ℝ := 30

/-- Percentage of CaSO₄ in the mixed compound -/
def percent_CaSO4 : ℝ := 30

/-- Theorem: The mass percentage of Ca in the mixed compound is approximately 49.432% -/
theorem mass_percentage_Ca_in_mixture : 
  ∃ (x : ℝ), abs (x - 49.432) < 0.001 ∧ 
  x = (percent_CaO / 100 * (molar_mass_Ca / molar_mass_CaO * 100)) +
      (percent_CaCO3 / 100 * (molar_mass_Ca / molar_mass_CaCO3 * 100)) +
      (percent_CaSO4 / 100 * (molar_mass_Ca / molar_mass_CaSO4 * 100)) :=
by sorry

end mass_percentage_Ca_in_mixture_l3086_308627


namespace optimal_selling_price_l3086_308634

/-- The optimal selling price problem -/
theorem optimal_selling_price (purchase_price : ℝ) (initial_price : ℝ) (initial_volume : ℝ) 
  (price_volume_relation : ℝ → ℝ) (profit_function : ℝ → ℝ) :
  purchase_price = 40 →
  initial_price = 50 →
  initial_volume = 50 →
  (∀ x, price_volume_relation x = initial_volume - x) →
  (∀ x, profit_function x = (initial_price + x) * (price_volume_relation x) - purchase_price * (price_volume_relation x)) →
  ∃ max_profit : ℝ, ∀ x, profit_function x ≤ max_profit ∧ profit_function 20 = max_profit →
  initial_price + 20 = 70 := by
sorry

end optimal_selling_price_l3086_308634


namespace sum_of_squares_quadratic_solution_l3086_308640

theorem sum_of_squares_quadratic_solution : 
  ∀ (s₁ s₂ : ℝ), s₁^2 - 10*s₁ + 7 = 0 → s₂^2 - 10*s₂ + 7 = 0 → s₁^2 + s₂^2 = 86 := by
  sorry

end sum_of_squares_quadratic_solution_l3086_308640


namespace function_characterization_l3086_308604

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, x * (f (x + 1) - f x) = f x)
  (h2 : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ c : ℝ, (|c| ≤ 1) ∧ (∀ x : ℝ, f x = c * x) :=
sorry

end function_characterization_l3086_308604


namespace problem_grid_square_count_l3086_308656

/-- Represents a square grid with some line segments removed -/
structure SquareGrid :=
  (size : Nat)
  (removed_segments : List (Nat × Nat × Nat × Nat))

/-- Counts the number of squares in a SquareGrid -/
def count_squares (grid : SquareGrid) : Nat :=
  sorry

/-- The specific 4x4 grid with two line segments removed as described in the problem -/
def problem_grid : SquareGrid :=
  { size := 4,
    removed_segments := [(1, 1, 1, 2), (2, 2, 3, 2)] }

/-- Theorem stating that the number of squares in the problem grid is 22 -/
theorem problem_grid_square_count :
  count_squares problem_grid = 22 := by sorry

end problem_grid_square_count_l3086_308656


namespace erica_pie_fraction_l3086_308692

theorem erica_pie_fraction (apple_fraction : ℚ) : 
  (apple_fraction + 3/4 = 95/100) → apple_fraction = 1/5 := by
sorry

end erica_pie_fraction_l3086_308692


namespace sphere_surface_area_l3086_308614

theorem sphere_surface_area (V : ℝ) (r : ℝ) (h : V = 72 * Real.pi) :
  4 * Real.pi * r^2 = 36 * Real.pi * (2^(1/3))^2 :=
by
  sorry

end sphere_surface_area_l3086_308614


namespace triangle_area_l3086_308608

theorem triangle_area (a b c : ℝ) (B : ℝ) : 
  B = 2 * Real.pi / 3 →
  b = Real.sqrt 13 →
  a + c = 4 →
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4 := by
sorry

end triangle_area_l3086_308608


namespace triangle_with_lattice_point_is_equilateral_l3086_308691

/-- A triangle in a plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- The perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ := sorry

/-- Whether a point is a lattice point --/
def is_lattice_point (p : ℝ × ℝ) : Prop := sorry

/-- Whether a point is on or inside a triangle --/
def point_in_triangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Whether two triangles are congruent --/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Whether a triangle is equilateral --/
def is_equilateral (t : Triangle) : Prop := sorry

theorem triangle_with_lattice_point_is_equilateral (t : Triangle) :
  perimeter t = 3 + 2 * Real.sqrt 3 →
  (∀ t' : Triangle, congruent t t' → ∃ p : ℝ × ℝ, is_lattice_point p ∧ point_in_triangle p t') →
  is_equilateral t :=
sorry

end triangle_with_lattice_point_is_equilateral_l3086_308691


namespace max_value_of_4x_plus_3y_l3086_308687

theorem max_value_of_4x_plus_3y (x y : ℝ) : 
  x^2 + y^2 = 16*x + 8*y + 10 → (4*x + 3*y ≤ 42) ∧ ∃ x y, x^2 + y^2 = 16*x + 8*y + 10 ∧ 4*x + 3*y = 42 := by
  sorry

end max_value_of_4x_plus_3y_l3086_308687


namespace spelling_bee_probability_l3086_308637

/-- The probability of selecting all girls in a spelling bee competition -/
theorem spelling_bee_probability (total : ℕ) (girls : ℕ) (selected : ℕ) 
  (h_total : total = 8) 
  (h_girls : girls = 5)
  (h_selected : selected = 3) :
  (Nat.choose girls selected : ℚ) / (Nat.choose total selected) = 5 / 28 := by
  sorry

end spelling_bee_probability_l3086_308637


namespace portrait_problem_l3086_308689

theorem portrait_problem (total_students : ℕ) (before_lunch : ℕ) (after_lunch : ℕ) 
  (h1 : total_students = 24)
  (h2 : before_lunch = total_students / 3)
  (h3 : after_lunch = 10) :
  total_students - (before_lunch + after_lunch) = 6 := by
  sorry

end portrait_problem_l3086_308689


namespace negation_of_odd_function_implication_l3086_308639

-- Define what it means for a function to be odd
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem negation_of_odd_function_implication :
  (¬ (IsOdd f → IsOdd (fun x ↦ f (-x)))) ↔ (¬ IsOdd f → ¬ IsOdd (fun x ↦ f (-x))) :=
by sorry

end negation_of_odd_function_implication_l3086_308639


namespace sandy_correct_sums_l3086_308688

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 65)
  (h3 : correct_marks = 3)
  (h4 : incorrect_marks = 2) :
  ∃ (correct_sums : ℕ) (incorrect_sums : ℕ),
    correct_sums + incorrect_sums = total_sums ∧
    (correct_sums : ℤ) * correct_marks - incorrect_sums * incorrect_marks = total_marks ∧
    correct_sums = 25 :=
by sorry

end sandy_correct_sums_l3086_308688


namespace ghee_mixture_proof_l3086_308666

/-- Proves that adding 20 kg of pure ghee to a 30 kg mixture of 50% pure ghee and 50% vanaspati
    results in a mixture where vanaspati constitutes 30% of the total. -/
theorem ghee_mixture_proof (original_quantity : ℝ) (pure_ghee_added : ℝ) 
  (h1 : original_quantity = 30)
  (h2 : pure_ghee_added = 20) : 
  let initial_vanaspati := 0.5 * original_quantity
  let total_after_addition := original_quantity + pure_ghee_added
  initial_vanaspati / total_after_addition = 0.3 := by
  sorry

#check ghee_mixture_proof

end ghee_mixture_proof_l3086_308666


namespace parallelogram_base_length_l3086_308696

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 32) 
  (h2 : height = 8) 
  (h3 : area = base * height) : 
  base = 4 := by
sorry

end parallelogram_base_length_l3086_308696


namespace sallys_number_l3086_308644

theorem sallys_number (n : ℕ) : 
  (1000 ≤ n ∧ n ≤ 9999) ∧ 
  (∀ d : ℕ, 2 ≤ d ∧ d ≤ 9 → n % d = 1) ↔ 
  (n = 2521 ∨ n = 5041 ∨ n = 7561) :=
sorry

end sallys_number_l3086_308644
