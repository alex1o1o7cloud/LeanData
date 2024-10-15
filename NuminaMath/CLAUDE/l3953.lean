import Mathlib

namespace NUMINAMATH_CALUDE_workshop_workers_count_l3953_395384

/-- Proves that the total number of workers in a workshop is 14, given specific salary conditions -/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  (W : ℚ) * 8000 = 70000 + (N : ℚ) * 6000 →
  W = 7 + N →
  W = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l3953_395384


namespace NUMINAMATH_CALUDE_julie_school_year_hours_l3953_395340

/-- Julie's summer work and earnings information -/
structure SummerWork where
  hoursPerWeek : ℕ
  weeks : ℕ
  earnings : ℕ

/-- Julie's school year work information -/
structure SchoolYearWork where
  weeks : ℕ
  targetEarnings : ℕ

/-- Calculate the required hours per week during school year -/
def calculateSchoolYearHours (summer : SummerWork) (schoolYear : SchoolYearWork) : ℕ :=
  let hourlyRate := summer.earnings / (summer.hoursPerWeek * summer.weeks)
  let weeklyEarningsNeeded := schoolYear.targetEarnings / schoolYear.weeks
  weeklyEarningsNeeded / hourlyRate

/-- Theorem stating that Julie needs to work 10 hours per week during the school year -/
theorem julie_school_year_hours 
    (summer : SummerWork) 
    (schoolYear : SchoolYearWork) 
    (h1 : summer.hoursPerWeek = 40)
    (h2 : summer.weeks = 10)
    (h3 : summer.earnings = 4000)
    (h4 : schoolYear.weeks = 40)
    (h5 : schoolYear.targetEarnings = 4000) :
  calculateSchoolYearHours summer schoolYear = 10 := by
  sorry

#eval calculateSchoolYearHours 
  { hoursPerWeek := 40, weeks := 10, earnings := 4000 } 
  { weeks := 40, targetEarnings := 4000 }

end NUMINAMATH_CALUDE_julie_school_year_hours_l3953_395340


namespace NUMINAMATH_CALUDE_odd_decreasing_function_theorem_l3953_395306

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is decreasing if f(x₁) > f(x₂) for all x₁ < x₂ in its domain -/
def IsDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b → f x₁ > f x₂

theorem odd_decreasing_function_theorem (f : ℝ → ℝ) (a : ℝ) 
    (h_odd : IsOdd f)
    (h_decreasing : IsDecreasing f (-1) 1)
    (h_condition : f (1 + a) + f (1 - a^2) < 0) :
    a ∈ Set.Ioo (-1) 0 := by
  sorry


end NUMINAMATH_CALUDE_odd_decreasing_function_theorem_l3953_395306


namespace NUMINAMATH_CALUDE_triangle_properties_l3953_395398

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- The main theorem about the specific acute triangle. -/
theorem triangle_properties (t : AcuteTriangle)
    (h1 : Real.sqrt 3 * t.a - 2 * t.b * Real.sin t.A = 0)
    (h2 : t.a + t.c = 5)
    (h3 : t.a > t.c)
    (h4 : t.b = Real.sqrt 7) :
    t.B = π/3 ∧ (1/2 * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3953_395398


namespace NUMINAMATH_CALUDE_solution_set_not_empty_or_specific_interval_l3953_395357

theorem solution_set_not_empty_or_specific_interval (a : ℝ) :
  ∃ x : ℝ, a * (x - a) * (a * x + a) ≥ 0 ∧
  ¬(∀ x : ℝ, a * (x - a) * (a * x + a) < 0) ∧
  ¬(∀ x : ℝ, (a * (x - a) * (a * x + a) ≥ 0) ↔ (a ≤ x ∧ x ≤ -1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_not_empty_or_specific_interval_l3953_395357


namespace NUMINAMATH_CALUDE_fraction_difference_to_fifth_power_l3953_395354

theorem fraction_difference_to_fifth_power :
  (3/4 - 1/8)^5 = 3125/32768 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_to_fifth_power_l3953_395354


namespace NUMINAMATH_CALUDE_expected_sufferers_l3953_395331

theorem expected_sufferers (sample_size : ℕ) (probability : ℚ) (h1 : sample_size = 400) (h2 : probability = 1/4) :
  ↑sample_size * probability = 100 := by
  sorry

end NUMINAMATH_CALUDE_expected_sufferers_l3953_395331


namespace NUMINAMATH_CALUDE_city_wall_length_l3953_395396

/-- Represents a city layout with 5 congruent squares in an isosceles cross shape -/
structure CityLayout where
  square_side : ℝ
  num_squares : Nat
  num_squares_eq : num_squares = 5

/-- Calculates the perimeter of the city layout -/
def perimeter (city : CityLayout) : ℝ :=
  12 * city.square_side

/-- Calculates the area of the city layout -/
def area (city : CityLayout) : ℝ :=
  city.num_squares * city.square_side^2

/-- Theorem stating that if the perimeter equals the area, then the perimeter is 28.8 km -/
theorem city_wall_length (city : CityLayout) :
  perimeter city = area city → perimeter city = 28.8 := by
  sorry


end NUMINAMATH_CALUDE_city_wall_length_l3953_395396


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3953_395339

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 1 ∧ y ≥ 2 → x + y ≥ 3) ∧
  ∃ x y, x + y ≥ 3 ∧ ¬(x ≥ 1 ∧ y ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3953_395339


namespace NUMINAMATH_CALUDE_triangle_existence_l3953_395320

theorem triangle_existence (q : ℝ) (α β γ : ℝ) 
  (h_positive : q > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum : α + β + γ = Real.pi) : 
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * b * Real.sin γ) / 2 = q^2 ∧
    Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) = α ∧
    Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) = β ∧
    Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) = γ :=
by sorry


end NUMINAMATH_CALUDE_triangle_existence_l3953_395320


namespace NUMINAMATH_CALUDE_original_number_is_six_l3953_395372

/-- Represents a person in the circle with their chosen number and announced average -/
structure Person where
  chosen : ℝ
  announced : ℝ

/-- The circle of 12 people -/
def Circle := Fin 12 → Person

theorem original_number_is_six
  (circle : Circle)
  (h_average : ∀ i : Fin 12, (circle i).announced = ((circle (i - 1)).chosen + (circle (i + 1)).chosen) / 2)
  (h_person : ∃ i : Fin 12, (circle i).announced = 8 ∧
    (circle (i - 1)).announced = 5 ∧ (circle (i + 1)).announced = 11) :
  ∃ i : Fin 12, (circle i).announced = 8 ∧ (circle i).chosen = 6 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_six_l3953_395372


namespace NUMINAMATH_CALUDE_shirt_cost_proof_l3953_395315

/-- The cost of the shirt Macey wants to buy -/
def shirt_cost : ℚ := 3

/-- The amount Macey has already saved -/
def saved_amount : ℚ := 3/2

/-- The number of weeks Macey needs to save -/
def weeks_to_save : ℕ := 3

/-- The amount Macey saves per week -/
def weekly_savings : ℚ := 1/2

theorem shirt_cost_proof : 
  shirt_cost = saved_amount + weeks_to_save * weekly_savings := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_proof_l3953_395315


namespace NUMINAMATH_CALUDE_prob_three_non_defective_pencils_l3953_395383

/-- The probability of selecting 3 non-defective pencils from a box of 7 pencils with 2 defective pencils -/
theorem prob_three_non_defective_pencils :
  let total_pencils : ℕ := 7
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let ways_to_select_all := Nat.choose total_pencils selected_pencils
  let ways_to_select_non_defective := Nat.choose non_defective_pencils selected_pencils
  (ways_to_select_non_defective : ℚ) / ways_to_select_all = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_prob_three_non_defective_pencils_l3953_395383


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_2006_l3953_395382

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2006 :
  units_digit (factorial_sum 2006) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_2006_l3953_395382


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3953_395344

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Theorem statement
theorem sixth_term_of_geometric_sequence :
  ∀ (r : ℝ),
  (geometric_sequence 16 r 8 = 11664) →
  (geometric_sequence 16 r 6 = 3888) :=
by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3953_395344


namespace NUMINAMATH_CALUDE_opposite_values_l3953_395368

theorem opposite_values (a b c m : ℚ) 
  (eq1 : a + 2*b + 3*c = m) 
  (eq2 : a + b + 2*c = m) : 
  b = -c := by
sorry

end NUMINAMATH_CALUDE_opposite_values_l3953_395368


namespace NUMINAMATH_CALUDE_min_distance_exp_to_line_l3953_395334

/-- The minimum distance from a point on y = e^x to y = x is √2/2 -/
theorem min_distance_exp_to_line :
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  let g : ℝ → ℝ := fun x ↦ x
  ∃ (x₀ : ℝ), ∀ (x : ℝ),
    Real.sqrt ((x - g x)^2 + (f x - g x)^2) ≥ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_exp_to_line_l3953_395334


namespace NUMINAMATH_CALUDE_escalator_problem_l3953_395323

/-- Represents the escalator system in the shopping mall -/
structure EscalatorSystem where
  boyStepRate : ℕ
  girlStepRate : ℕ
  boyStepsToTop : ℕ
  girlStepsToTop : ℕ
  escalatorSpeed : ℝ
  exposedSteps : ℕ

/-- The conditions of the problem -/
def problemConditions (sys : EscalatorSystem) : Prop :=
  sys.boyStepRate = 2 * sys.girlStepRate ∧
  sys.boyStepsToTop = 27 ∧
  sys.girlStepsToTop = 18 ∧
  sys.escalatorSpeed > 0

/-- The theorem to prove -/
theorem escalator_problem (sys : EscalatorSystem) 
  (h : problemConditions sys) : 
  sys.exposedSteps = 54 ∧ 
  ∃ (boySteps : ℕ), boySteps = 198 ∧ 
    (boySteps = 3 * sys.boyStepsToTop + 2 * sys.exposedSteps) :=
sorry

end NUMINAMATH_CALUDE_escalator_problem_l3953_395323


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3953_395328

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (x, 4)
  are_parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3953_395328


namespace NUMINAMATH_CALUDE_particular_number_exists_l3953_395376

theorem particular_number_exists : ∃ x : ℝ, 4 * 25 * x = 812 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_exists_l3953_395376


namespace NUMINAMATH_CALUDE_last_digits_divisible_by_three_l3953_395361

theorem last_digits_divisible_by_three :
  ∃ (S : Finset Nat), (∀ n ∈ S, n < 10) ∧ (Finset.card S = 10) ∧
  (∀ d ∈ S, ∃ (m : Nat), m % 3 = 0 ∧ m % 10 = d) :=
sorry

end NUMINAMATH_CALUDE_last_digits_divisible_by_three_l3953_395361


namespace NUMINAMATH_CALUDE_johns_apartment_paint_area_l3953_395367

/-- Represents the dimensions of a bedroom -/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a single bedroom -/
def area_to_paint (dim : BedroomDimensions) (unpainted_area : ℝ) : ℝ :=
  2 * (dim.length * dim.height + dim.width * dim.height) + 
  dim.length * dim.width - unpainted_area

/-- Theorem stating the total area to be painted in John's apartment -/
theorem johns_apartment_paint_area :
  let bedroom_dim : BedroomDimensions := ⟨15, 12, 10⟩
  let unpainted_area : ℝ := 70
  let num_bedrooms : ℕ := 2
  num_bedrooms * (area_to_paint bedroom_dim unpainted_area) = 1300 := by
  sorry


end NUMINAMATH_CALUDE_johns_apartment_paint_area_l3953_395367


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3953_395359

/-- Expresses the sum of repeating decimals 0.3̅, 0.07̅, and 0.008̅ as a common fraction -/
theorem repeating_decimal_sum : 
  (1 : ℚ) / 3 + 7 / 99 + 8 / 999 = 418 / 999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3953_395359


namespace NUMINAMATH_CALUDE_combined_female_average_score_l3953_395353

theorem combined_female_average_score 
  (a b c d : ℕ) 
  (adam_avg : (71 * a + 76 * b) / (a + b) = 74)
  (baker_avg : (81 * c + 90 * d) / (c + d) = 84)
  (male_avg : (71 * a + 81 * c) / (a + c) = 79) :
  (76 * b + 90 * d) / (b + d) = 84 :=
sorry

end NUMINAMATH_CALUDE_combined_female_average_score_l3953_395353


namespace NUMINAMATH_CALUDE_average_monthly_sales_l3953_395395

def may_sales : ℝ := 150
def june_sales : ℝ := 75
def july_sales : ℝ := 50
def august_sales : ℝ := 175

def total_months : ℕ := 4

def total_sales : ℝ := may_sales + june_sales + july_sales + august_sales

theorem average_monthly_sales : 
  total_sales / total_months = 112.5 := by sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l3953_395395


namespace NUMINAMATH_CALUDE_sin_equality_integer_solutions_l3953_395394

theorem sin_equality_integer_solutions (m : ℤ) :
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.sin (750 * π / 180) →
  m = 30 ∨ m = 150 := by
sorry

end NUMINAMATH_CALUDE_sin_equality_integer_solutions_l3953_395394


namespace NUMINAMATH_CALUDE_candidates_scientific_notation_l3953_395374

/-- The number of candidates for the high school entrance examination in Guangdong Province in 2023 -/
def candidates : ℝ := 1108200

/-- The scientific notation representation of the number of candidates -/
def scientific_notation : ℝ := 1.1082 * (10 ^ 6)

/-- Theorem stating that the number of candidates is equal to its scientific notation representation -/
theorem candidates_scientific_notation : candidates = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_candidates_scientific_notation_l3953_395374


namespace NUMINAMATH_CALUDE_floor_times_self_equals_54_l3953_395324

theorem floor_times_self_equals_54 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 54 ∧ x = 54 / 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_equals_54_l3953_395324


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3953_395338

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 →
  a 2 = 4 →
  (∀ k ≥ 2, 2 * a k = a (k - 1) + a (k + 1)) →
  a n = 301 →
  n = 101 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3953_395338


namespace NUMINAMATH_CALUDE_all_normal_all_false_l3953_395362

-- Define the possible types of people
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define the four people
structure Person :=
  (name : String)
  (type : PersonType)

-- Define the statements made
def statement1 (mr_b : Person) : Prop := mr_b.type = PersonType.Knight
def statement2 (mr_a : Person) (mr_b : Person) : Prop := 
  mr_a.type = PersonType.Knight ∧ mr_b.type = PersonType.Knight
def statement3 (mr_b : Person) : Prop := mr_b.type = PersonType.Knight

-- Define the problem setup
def problem_setup (mr_a mrs_a mr_b mrs_b : Person) : Prop :=
  mr_a.name = "Mr. A" ∧
  mrs_a.name = "Mrs. A" ∧
  mr_b.name = "Mr. B" ∧
  mrs_b.name = "Mrs. B"

-- Theorem statement
theorem all_normal_all_false 
  (mr_a mrs_a mr_b mrs_b : Person) 
  (h_setup : problem_setup mr_a mrs_a mr_b mrs_b) :
  (mr_a.type = PersonType.Normal ∧
   mrs_a.type = PersonType.Normal ∧
   mr_b.type = PersonType.Normal ∧
   mrs_b.type = PersonType.Normal) ∧
  (¬statement1 mr_b ∧
   ¬statement2 mr_a mr_b ∧
   ¬statement3 mr_b) :=
by sorry


end NUMINAMATH_CALUDE_all_normal_all_false_l3953_395362


namespace NUMINAMATH_CALUDE_flagpole_height_l3953_395369

/-- Given a lamppost height and shadow length, calculate the height of another object with a known shadow length -/
theorem flagpole_height
  (lamppost_height : ℝ) 
  (lamppost_shadow : ℝ) 
  (flagpole_shadow : ℝ) 
  (h1 : lamppost_height = 50)
  (h2 : lamppost_shadow = 12)
  (h3 : flagpole_shadow = 18 / 12)  -- Convert 18 inches to feet
  : ∃ (flagpole_height : ℝ), 
    flagpole_height * lamppost_shadow = lamppost_height * flagpole_shadow ∧ 
    flagpole_height * 12 = 75 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l3953_395369


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l3953_395321

theorem expansion_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℚ) : 
  (∀ x, (2*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  2^8 - 1 = 255 →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l3953_395321


namespace NUMINAMATH_CALUDE_happy_children_count_l3953_395312

theorem happy_children_count (total : ℕ) (sad : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) :
  total = 60 →
  sad = 10 →
  neither = 20 →
  boys = 17 →
  girls = 43 →
  happy_boys = 6 →
  sad_girls = 4 →
  neither_boys = 5 →
  ∃ (happy : ℕ), happy = 30 ∧ happy + sad + neither = total :=
by sorry

end NUMINAMATH_CALUDE_happy_children_count_l3953_395312


namespace NUMINAMATH_CALUDE_white_marbles_in_basket_c_l3953_395309

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : ℕ
  color2 : String
  count2 : ℕ

/-- The greatest difference between marble counts in any basket -/
def greatestDifference : ℕ := 6

/-- Basket A containing red and yellow marbles -/
def basketA : Basket := ⟨"red", 4, "yellow", 2⟩

/-- Basket B containing green and yellow marbles -/
def basketB : Basket := ⟨"green", 6, "yellow", 1⟩

/-- Basket C containing white and yellow marbles -/
def basketC : Basket := ⟨"white", 15, "yellow", 9⟩

/-- Theorem stating that the number of white marbles in Basket C is 15 -/
theorem white_marbles_in_basket_c :
  basketC.color1 = "white" ∧ basketC.count1 = 15 :=
by sorry

end NUMINAMATH_CALUDE_white_marbles_in_basket_c_l3953_395309


namespace NUMINAMATH_CALUDE_xyz_sum_l3953_395316

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 53)
  (h2 : y * z + x = 53)
  (h3 : z * x + y = 53) : 
  x + y + z = 54 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l3953_395316


namespace NUMINAMATH_CALUDE_vector_addition_l3953_395349

theorem vector_addition (a b : ℝ × ℝ) : 
  a = (-1, 6) → b = (3, -2) → a + b = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l3953_395349


namespace NUMINAMATH_CALUDE_M_greater_than_N_l3953_395373

theorem M_greater_than_N (a : ℝ) : 2 * a * (a - 2) > (a + 1) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l3953_395373


namespace NUMINAMATH_CALUDE_correct_date_l3953_395335

-- Define a type for days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a type for months
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

-- Define a structure for a date
structure Date where
  day : Nat
  month : Month
  dayOfWeek : DayOfWeek

def nextDay (d : Date) : Date := sorry
def addDays (d : Date) (n : Nat) : Date := sorry

-- The main theorem
theorem correct_date (d : Date) : 
  (nextDay d).month ≠ Month.September ∧ 
  (addDays d 7).month = Month.September ∧
  (addDays d 2).dayOfWeek ≠ DayOfWeek.Wednesday →
  d = Date.mk 25 Month.August DayOfWeek.Wednesday :=
by sorry

end NUMINAMATH_CALUDE_correct_date_l3953_395335


namespace NUMINAMATH_CALUDE_gold_copper_alloy_ratio_l3953_395352

theorem gold_copper_alloy_ratio 
  (G : ℝ) 
  (h_G : G > 9) : 
  let x := 9 / (G - 9)
  x * G + (1 - x) * 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gold_copper_alloy_ratio_l3953_395352


namespace NUMINAMATH_CALUDE_integral_abs_x_minus_one_l3953_395317

-- Define the function to be integrated
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem integral_abs_x_minus_one : ∫ x in (-2)..2, f x = 5 := by
  sorry

end NUMINAMATH_CALUDE_integral_abs_x_minus_one_l3953_395317


namespace NUMINAMATH_CALUDE_multiply_powers_same_base_l3953_395332

theorem multiply_powers_same_base (x : ℝ) : x * x^2 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_same_base_l3953_395332


namespace NUMINAMATH_CALUDE_fraction_equality_l3953_395351

theorem fraction_equality : 2 / 3 = (2 + 4) / (3 + 6) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3953_395351


namespace NUMINAMATH_CALUDE_pool_capacity_l3953_395356

theorem pool_capacity (C : ℝ) 
  (h1 : 0.45 * C + 300 = 0.75 * C) : C = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l3953_395356


namespace NUMINAMATH_CALUDE_card_collection_solution_l3953_395305

/-- Represents the card collection problem --/
structure CardCollection where
  total_cards : Nat
  damaged_cards : Nat
  full_box_capacity : Nat
  damaged_box_capacity : Nat

/-- Calculates the number of cards in the last partially filled box of undamaged cards --/
def last_box_count (cc : CardCollection) : Nat :=
  (cc.total_cards - cc.damaged_cards) % cc.full_box_capacity

/-- Theorem stating the solution to the card collection problem --/
theorem card_collection_solution (cc : CardCollection) 
  (h1 : cc.total_cards = 120)
  (h2 : cc.damaged_cards = 18)
  (h3 : cc.full_box_capacity = 10)
  (h4 : cc.damaged_box_capacity = 5) :
  last_box_count cc = 2 := by
  sorry

#eval last_box_count { total_cards := 120, damaged_cards := 18, full_box_capacity := 10, damaged_box_capacity := 5 }

end NUMINAMATH_CALUDE_card_collection_solution_l3953_395305


namespace NUMINAMATH_CALUDE_apples_per_pie_l3953_395325

theorem apples_per_pie (total_apples : ℕ) (handed_out : ℕ) (num_pies : ℕ) :
  total_apples = 51 →
  handed_out = 41 →
  num_pies = 2 →
  (total_apples - handed_out) / num_pies = 5 := by
sorry

end NUMINAMATH_CALUDE_apples_per_pie_l3953_395325


namespace NUMINAMATH_CALUDE_library_sunday_visitors_l3953_395389

/-- Calculates the average number of visitors on Sundays in a library -/
theorem library_sunday_visitors
  (total_days : Nat)
  (sunday_count : Nat)
  (non_sunday_visitors : Nat)
  (total_average : Nat)
  (h1 : total_days = 30)
  (h2 : sunday_count = 5)
  (h3 : non_sunday_visitors = 240)
  (h4 : total_average = 295) :
  (total_average * total_days - non_sunday_visitors * (total_days - sunday_count)) / sunday_count = 570 := by
  sorry

end NUMINAMATH_CALUDE_library_sunday_visitors_l3953_395389


namespace NUMINAMATH_CALUDE_grunters_win_probabilities_l3953_395329

/-- The number of games played -/
def num_games : ℕ := 6

/-- The probability of winning a single game -/
def win_prob : ℚ := 7/10

/-- The probability of winning all games -/
def prob_win_all : ℚ := 117649/1000000

/-- The probability of winning exactly 5 out of 6 games -/
def prob_win_five : ℚ := 302526/1000000

/-- Theorem stating the probabilities for winning all games and winning exactly 5 out of 6 games -/
theorem grunters_win_probabilities :
  (win_prob ^ num_games = prob_win_all) ∧
  (Nat.choose num_games 5 * win_prob ^ 5 * (1 - win_prob) ^ 1 = prob_win_five) := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probabilities_l3953_395329


namespace NUMINAMATH_CALUDE_min_value_of_trig_function_l3953_395380

open Real

theorem min_value_of_trig_function :
  ∃ (x : ℝ), ∀ (y : ℝ), 2 * sin (π / 3 - x) - cos (π / 6 + x) ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_trig_function_l3953_395380


namespace NUMINAMATH_CALUDE_symmetric_seven_zeros_sum_l3953_395350

/-- A function representing |(1-x^2)(x^2+ax+b)| - c -/
def f (a b c x : ℝ) : ℝ := |(1 - x^2) * (x^2 + a*x + b)| - c

/-- Symmetry condition: f is symmetric about x = -2 -/
def is_symmetric (a b c : ℝ) : Prop :=
  ∀ x, f a b c (x + 2) = f a b c (-x - 2)

/-- The function has exactly 7 zeros -/
def has_seven_zeros (a b c : ℝ) : Prop :=
  ∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, f a b c x = 0

theorem symmetric_seven_zeros_sum (a b c : ℝ) :
  is_symmetric a b c →
  has_seven_zeros a b c →
  c ≠ 0 →
  a + b + c = 32 := by sorry

end NUMINAMATH_CALUDE_symmetric_seven_zeros_sum_l3953_395350


namespace NUMINAMATH_CALUDE_cards_distribution_l3953_395319

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) 
  (h2 : num_people = 8) : 
  (num_people - (total_cards % num_people)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l3953_395319


namespace NUMINAMATH_CALUDE_mixed_rectangles_count_even_l3953_395341

/-- Represents a tiling of an m × n grid using 2×2 and 1×3 mosaics -/
def GridTiling (m n : ℕ) : Type := Unit

/-- Counts the number of 1×2 rectangles with one cell from a 2×2 mosaic and one from a 1×3 mosaic -/
def countMixedRectangles (tiling : GridTiling m n) : ℕ := sorry

/-- Theorem stating that the count of mixed rectangles is even -/
theorem mixed_rectangles_count_even (m n : ℕ) (tiling : GridTiling m n) :
  Even (countMixedRectangles tiling) := by sorry

end NUMINAMATH_CALUDE_mixed_rectangles_count_even_l3953_395341


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l3953_395313

theorem hamburgers_left_over (made served : ℕ) (h1 : made = 9) (h2 : served = 3) :
  made - served = 6 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_l3953_395313


namespace NUMINAMATH_CALUDE_correct_average_calculation_l3953_395399

theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 15 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n : ℚ) * initial_avg - wrong_num + correct_num = n * 16 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l3953_395399


namespace NUMINAMATH_CALUDE_two_and_three_digit_problem_l3953_395326

theorem two_and_three_digit_problem :
  ∃ (x y : ℕ), 
    10 ≤ x ∧ x < 100 ∧
    100 ≤ y ∧ y < 1000 ∧
    1000 * x + y = 7 * x * y ∧
    x + y = 1074 := by
  sorry

end NUMINAMATH_CALUDE_two_and_three_digit_problem_l3953_395326


namespace NUMINAMATH_CALUDE_greatest_x_given_lcm_l3953_395322

def is_lcm (a b c m : ℕ) : Prop := 
  (∀ n : ℕ, n % a = 0 ∧ n % b = 0 ∧ n % c = 0 → m ∣ n) ∧
  (m % a = 0 ∧ m % b = 0 ∧ m % c = 0)

theorem greatest_x_given_lcm : 
  ∀ x : ℕ, is_lcm x 15 21 105 → x ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_given_lcm_l3953_395322


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l3953_395378

theorem max_abs_z_on_circle (z : ℂ) (h : Complex.abs (z - Complex.I * 2) = 1) :
  ∃ (z_max : ℂ), Complex.abs z_max = 3 ∧ 
  ∀ (w : ℂ), Complex.abs (w - Complex.I * 2) = 1 → Complex.abs w ≤ Complex.abs z_max :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l3953_395378


namespace NUMINAMATH_CALUDE_gigi_initial_flour_l3953_395311

/-- The amount of flour required for one batch of cookies -/
def flour_per_batch : ℕ := 2

/-- The number of batches Gigi has already baked -/
def baked_batches : ℕ := 3

/-- The number of additional batches Gigi can make with the remaining flour -/
def future_batches : ℕ := 7

/-- The total amount of flour in Gigi's bag initially -/
def initial_flour : ℕ := flour_per_batch * (baked_batches + future_batches)

theorem gigi_initial_flour :
  initial_flour = 20 := by sorry

end NUMINAMATH_CALUDE_gigi_initial_flour_l3953_395311


namespace NUMINAMATH_CALUDE_expression_evaluation_l3953_395377

theorem expression_evaluation : (25 + 15)^2 - (25^2 + 15^2 + 150) = 600 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3953_395377


namespace NUMINAMATH_CALUDE_distance_from_p_to_ad_l3953_395318

/-- Square with side length 6 -/
structure Square :=
  (side : ℝ)
  (is_six : side = 6)

/-- Point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Circle in 2D space -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Given a square ABCD, find the distance from point P to side AD, where P is an intersection
    point of two circles: one centered at M (midpoint of CD) with radius 3, and another centered
    at A with radius 5. -/
def distance_to_side (s : Square) : ℝ :=
  let a := Point.mk 0 s.side
  let d := Point.mk 0 0
  let m := Point.mk (s.side / 2) 0
  let circle_m := Circle.mk m 3
  let circle_a := Circle.mk a 5
  -- The actual calculation of the distance would go here
  sorry

/-- The theorem stating that the distance from P to AD is equal to some specific value -/
theorem distance_from_p_to_ad (s : Square) : ∃ x : ℝ, distance_to_side s = x :=
  sorry

end NUMINAMATH_CALUDE_distance_from_p_to_ad_l3953_395318


namespace NUMINAMATH_CALUDE_max_distance_difference_l3953_395330

/-- The hyperbola E with equation x²/m - y²/3 = 1 where m > 0 -/
structure Hyperbola where
  m : ℝ
  h_m_pos : m > 0

/-- The eccentricity of the hyperbola -/
def eccentricity (E : Hyperbola) : ℝ := 2

/-- The right focus F of the hyperbola -/
def right_focus (E : Hyperbola) : ℝ × ℝ := sorry

/-- Point A -/
def point_A : ℝ × ℝ := (0, 1)

/-- A point P on the right branch of the hyperbola -/
def point_P (E : Hyperbola) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the maximum value of |PF| - |PA| -/
theorem max_distance_difference (E : Hyperbola) :
  ∃ (max : ℝ), ∀ (P : ℝ × ℝ), P = point_P E →
    distance P (right_focus E) - distance P point_A ≤ max ∧
    max = Real.sqrt 5 - 2 :=
sorry

end NUMINAMATH_CALUDE_max_distance_difference_l3953_395330


namespace NUMINAMATH_CALUDE_m_range_l3953_395303

-- Define the condition function
def condition (x : ℝ) (m : ℝ) : Prop := 0 ≤ x ∧ x ≤ m

-- Define the quadratic inequality
def quadratic_inequality (x : ℝ) : Prop := x^2 - 3*x + 2 ≤ 0

-- Define the necessary but not sufficient relationship
def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ x, quadratic_inequality x → condition x m) ∧
  (∃ x, condition x m ∧ ¬quadratic_inequality x)

-- Theorem statement
theorem m_range (m : ℝ) :
  necessary_not_sufficient m ↔ m ∈ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3953_395303


namespace NUMINAMATH_CALUDE_min_value_theorem_l3953_395381

theorem min_value_theorem (x : ℝ) (h : x > 9) :
  (x^2 + 81) / (x - 9) ≥ 27 ∧ ∃ y > 9, (y^2 + 81) / (y - 9) = 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3953_395381


namespace NUMINAMATH_CALUDE_painter_rooms_problem_l3953_395343

theorem painter_rooms_problem (hours_per_room : ℕ) (rooms_painted : ℕ) (remaining_hours : ℕ) :
  hours_per_room = 7 →
  rooms_painted = 5 →
  remaining_hours = 49 →
  rooms_painted + remaining_hours / hours_per_room = 12 :=
by sorry

end NUMINAMATH_CALUDE_painter_rooms_problem_l3953_395343


namespace NUMINAMATH_CALUDE_jack_initial_money_l3953_395366

def initial_bottles : ℕ := 4
def bottle_cost : ℚ := 2
def cheese_weight : ℚ := 1/2
def cheese_cost_per_pound : ℚ := 10
def remaining_money : ℚ := 71

theorem jack_initial_money :
  let total_bottles := initial_bottles + 2 * initial_bottles
  let water_cost := total_bottles * bottle_cost
  let cheese_cost := cheese_weight * cheese_cost_per_pound
  let total_spent := water_cost + cheese_cost
  total_spent + remaining_money = 100 := by sorry

end NUMINAMATH_CALUDE_jack_initial_money_l3953_395366


namespace NUMINAMATH_CALUDE_orange_count_proof_l3953_395302

/-- The number of apples in the basket -/
def num_apples : ℕ := 10

/-- The number of oranges added to the basket -/
def added_oranges : ℕ := 5

/-- The initial number of oranges in the basket -/
def initial_oranges : ℕ := 5

theorem orange_count_proof :
  (num_apples : ℚ) = (1 / 2 : ℚ) * ((num_apples : ℚ) + (initial_oranges : ℚ) + (added_oranges : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_orange_count_proof_l3953_395302


namespace NUMINAMATH_CALUDE_stamp_cost_theorem_l3953_395387

theorem stamp_cost_theorem (total_stamps : ℕ) (high_value_stamps : ℕ) (high_value : ℚ) (low_value : ℚ) :
  total_stamps = 20 →
  high_value_stamps = 18 →
  high_value = 37 / 100 →
  low_value = 20 / 100 →
  (high_value_stamps * high_value + (total_stamps - high_value_stamps) * low_value) = 706 / 100 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_theorem_l3953_395387


namespace NUMINAMATH_CALUDE_descent_route_length_l3953_395345

/- Define the hiking trip parameters -/
def forest_speed : ℝ := 8
def rocky_speed : ℝ := 5
def snowy_speed : ℝ := 3
def forest_time : ℝ := 1
def rocky_time : ℝ := 1
def snowy_time : ℝ := 0.5
def speed_multiplier : ℝ := 1.5
def total_days : ℝ := 2

/- Define the theorem -/
theorem descent_route_length :
  let grassland_speed := forest_speed * speed_multiplier
  let sandy_speed := rocky_speed * speed_multiplier
  let descent_distance := grassland_speed * forest_time + sandy_speed * rocky_time
  descent_distance = 19.5 := by sorry

end NUMINAMATH_CALUDE_descent_route_length_l3953_395345


namespace NUMINAMATH_CALUDE_no_integer_solution_for_dog_nails_l3953_395336

theorem no_integer_solution_for_dog_nails :
  ¬ ∃ (x : ℕ), 16 * x + 64 = 113 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_dog_nails_l3953_395336


namespace NUMINAMATH_CALUDE_proportion_fourth_number_l3953_395385

theorem proportion_fourth_number (x y : ℝ) : 
  (0.75 : ℝ) / x = 5 / y → x = 1.05 → y = 7 := by sorry

end NUMINAMATH_CALUDE_proportion_fourth_number_l3953_395385


namespace NUMINAMATH_CALUDE_equation_solution_l3953_395391

theorem equation_solution : ∃! x : ℝ, (x^2 - x - 2) / (x + 2) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3953_395391


namespace NUMINAMATH_CALUDE_marble_difference_is_seventeen_l3953_395360

/-- Calculates the difference in marbles between John and Ben after Ben gives half his marbles to John -/
def marbleDifference (benInitial : ℕ) (johnInitial : ℕ) : ℕ :=
  let benFinal := benInitial - benInitial / 2
  let johnFinal := johnInitial + benInitial / 2
  johnFinal - benFinal

/-- Proves that the difference in marbles between John and Ben is 17 after the transfer -/
theorem marble_difference_is_seventeen :
  marbleDifference 18 17 = 17 := by
  sorry

#eval marbleDifference 18 17

end NUMINAMATH_CALUDE_marble_difference_is_seventeen_l3953_395360


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3953_395386

theorem possible_values_of_a (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 3 = (x + b) * (x + c)) →
  (a = 1 ∨ a = 9) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3953_395386


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3953_395392

theorem yellow_balls_count (red blue yellow green : ℕ) : 
  red + blue + yellow + green = 531 →
  red + blue = yellow + green + 31 →
  yellow = green + 22 →
  yellow = 136 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3953_395392


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l3953_395363

/-- Calculates the maximum number of complete books that can be read given the reading speed, book length, and available time. -/
def max_complete_books_read (reading_speed : ℕ) (book_length : ℕ) (available_time : ℕ) : ℕ :=
  (available_time * reading_speed) / book_length

/-- Proves that Robert can read at most 2 complete 360-page books in 8 hours at a speed of 120 pages per hour. -/
theorem robert_reading_capacity :
  max_complete_books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l3953_395363


namespace NUMINAMATH_CALUDE_dog_distribution_theorem_l3953_395314

/-- The number of ways to distribute 12 dogs into three groups -/
def dog_distribution_ways : ℕ :=
  (Nat.choose 11 3) * (Nat.choose 7 4)

/-- Theorem stating the number of ways to distribute the dogs -/
theorem dog_distribution_theorem : dog_distribution_ways = 5775 := by
  sorry

end NUMINAMATH_CALUDE_dog_distribution_theorem_l3953_395314


namespace NUMINAMATH_CALUDE_goldfish_equality_exists_l3953_395388

theorem goldfish_equality_exists : ∃ n : ℕ+, 
  8 * (5 : ℝ)^n.val = 200 * (3 : ℝ)^n.val + 20 * ((3 : ℝ)^n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_exists_l3953_395388


namespace NUMINAMATH_CALUDE_meetings_percentage_is_24_l3953_395375

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 10 * 60

/-- Represents the duration of a break in minutes -/
def break_minutes : ℕ := 30

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 75

/-- Calculates the effective work minutes (excluding break) -/
def effective_work_minutes : ℕ := work_day_minutes - break_minutes

/-- Calculates the total meeting minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Theorem stating that the percentage of effective work day spent in meetings is 24% -/
theorem meetings_percentage_is_24 : 
  (total_meeting_minutes : ℚ) / (effective_work_minutes : ℚ) * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_meetings_percentage_is_24_l3953_395375


namespace NUMINAMATH_CALUDE_roots_sum_abs_l3953_395301

theorem roots_sum_abs (a b c m : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  abs a + abs b + abs c = 94 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_abs_l3953_395301


namespace NUMINAMATH_CALUDE_correct_calculation_l3953_395300

theorem correct_calculation (y : ℝ) : 3 * y^2 - 2 * y^2 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3953_395300


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l3953_395370

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l3953_395370


namespace NUMINAMATH_CALUDE_quadratic_sum_abc_l3953_395342

/-- Given a quadratic polynomial 12x^2 - 72x + 432, prove that when written in the form a(x+b)^2 + c, 
    the sum of a, b, and c is 333. -/
theorem quadratic_sum_abc (x : ℝ) : 
  ∃ (a b c : ℝ), (12 * x^2 - 72 * x + 432 = a * (x + b)^2 + c) ∧ (a + b + c = 333) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_abc_l3953_395342


namespace NUMINAMATH_CALUDE_equation_solutions_l3953_395365

theorem equation_solutions :
  (∀ x : ℝ, 9 * x^2 - 25 = 0 ↔ x = 5/3 ∨ x = -5/3) ∧
  (∀ x : ℝ, (x + 1)^3 - 27 = 0 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3953_395365


namespace NUMINAMATH_CALUDE_classroom_difference_l3953_395379

/-- Proves that the difference between the total number of students and books in 6 classrooms is 90 -/
theorem classroom_difference : 
  let students_per_classroom : ℕ := 20
  let books_per_classroom : ℕ := 5
  let num_classrooms : ℕ := 6
  let total_students : ℕ := students_per_classroom * num_classrooms
  let total_books : ℕ := books_per_classroom * num_classrooms
  total_students - total_books = 90 := by
  sorry


end NUMINAMATH_CALUDE_classroom_difference_l3953_395379


namespace NUMINAMATH_CALUDE_compare_powers_l3953_395355

theorem compare_powers : (4 ^ 12 : ℕ) < 9 ^ 8 ∧ 9 ^ 8 = 3 ^ 16 := by sorry

end NUMINAMATH_CALUDE_compare_powers_l3953_395355


namespace NUMINAMATH_CALUDE_jan_roses_cost_l3953_395364

theorem jan_roses_cost : 
  let dozen : ℕ := 12
  let roses_bought : ℕ := 5 * dozen
  let cost_per_rose : ℕ := 6
  let discount_rate : ℚ := 4/5
  (roses_bought * cost_per_rose : ℚ) * discount_rate = 288 := by
  sorry

end NUMINAMATH_CALUDE_jan_roses_cost_l3953_395364


namespace NUMINAMATH_CALUDE_icosahedron_painting_ways_l3953_395393

/-- Represents a regular icosahedron -/
structure Icosahedron where
  faces : Nat
  rotationalSymmetries : Nat

/-- Represents the number of ways to paint an icosahedron -/
def paintingWays (i : Icosahedron) (colors : Nat) : Nat :=
  Nat.factorial (colors - 1) / i.rotationalSymmetries

/-- Theorem stating the number of distinguishable ways to paint an icosahedron -/
theorem icosahedron_painting_ways (i : Icosahedron) (h1 : i.faces = 20) (h2 : i.rotationalSymmetries = 60) :
  paintingWays i 20 = Nat.factorial 19 / 60 := by
  sorry

#check icosahedron_painting_ways

end NUMINAMATH_CALUDE_icosahedron_painting_ways_l3953_395393


namespace NUMINAMATH_CALUDE_student_calculation_l3953_395327

theorem student_calculation (chosen_number : ℕ) (h : chosen_number = 121) : 
  2 * chosen_number - 138 = 104 := by
  sorry

#check student_calculation

end NUMINAMATH_CALUDE_student_calculation_l3953_395327


namespace NUMINAMATH_CALUDE_f_inequalities_l3953_395310

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

theorem f_inequalities :
  (∀ x, f 3 x < 0 ↔ -1 < x ∧ x < 3) ∧
  (∀ x, f (-1) x > 0 ↔ x ≠ -1) ∧
  (∀ a, a > -1 → ∀ x, f a x > 0 ↔ x < -1 ∨ x > a) ∧
  (∀ a, a < -1 → ∀ x, f a x > 0 ↔ x < a ∨ x > -1) :=
by sorry

end NUMINAMATH_CALUDE_f_inequalities_l3953_395310


namespace NUMINAMATH_CALUDE_equation_solution_l3953_395348

theorem equation_solution : 
  ∃! x : ℝ, 4 * (4 ^ x) + Real.sqrt (16 * (16 ^ x)) = 64 ∧ x = (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3953_395348


namespace NUMINAMATH_CALUDE_biased_coin_probability_l3953_395307

def coin_prob (n : Nat) : ℚ :=
  match n with
  | 1 => 3/4
  | 2 => 1/2
  | 3 => 1/4
  | 4 => 1/3
  | 5 => 2/3
  | 6 => 3/5
  | 7 => 4/7
  | _ => 0

theorem biased_coin_probability :
  (coin_prob 1 * coin_prob 2 * (1 - coin_prob 3) * (1 - coin_prob 4) *
   (1 - coin_prob 5) * (1 - coin_prob 6) * (1 - coin_prob 7)) = 3/560 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l3953_395307


namespace NUMINAMATH_CALUDE_trapezium_side_length_l3953_395337

theorem trapezium_side_length 
  (a b h area : ℝ) 
  (h1 : b = 28) 
  (h2 : h = 21) 
  (h3 : area = 504) 
  (h4 : area = (a + b) * h / 2) : 
  a = 20 := by sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l3953_395337


namespace NUMINAMATH_CALUDE_weight_of_barium_fluoride_l3953_395346

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of moles of Barium fluoride -/
def moles_BaF2 : ℝ := 3

/-- The molecular weight of Barium fluoride (BaF2) in g/mol -/
def molecular_weight_BaF2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_F

/-- The weight of Barium fluoride in grams -/
def weight_BaF2 : ℝ := moles_BaF2 * molecular_weight_BaF2

theorem weight_of_barium_fluoride : weight_BaF2 = 525.99 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_barium_fluoride_l3953_395346


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l3953_395390

/-- 
Given a quadratic equation (k+2)x^2 + 6x + k^2 + k - 2 = 0 where 0 is one of its roots,
prove that k = 1.
-/
theorem quadratic_root_zero (k : ℝ) : 
  (∀ x, (k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0 ↔ x = 0 ∨ x = -(6 / (k + 2))) →
  k + 2 ≠ 0 →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l3953_395390


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l3953_395358

def ring_arrangements (total_rings : ℕ) (rings_to_use : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings rings_to_use * 
  Nat.factorial rings_to_use * 
  Nat.choose (rings_to_use + fingers - 1) (fingers - 1)

theorem ring_arrangement_count :
  ring_arrangements 8 5 4 = 376320 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l3953_395358


namespace NUMINAMATH_CALUDE_rotate_minus_two_zero_l3953_395304

/-- Rotate a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotate_minus_two_zero :
  rotate90Clockwise (-2, 0) = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_minus_two_zero_l3953_395304


namespace NUMINAMATH_CALUDE_phils_remaining_pages_l3953_395371

/-- Given an initial number of books, pages per book, and books lost,
    calculate the total number of pages remaining. -/
def remaining_pages (initial_books : ℕ) (pages_per_book : ℕ) (books_lost : ℕ) : ℕ :=
  (initial_books - books_lost) * pages_per_book

/-- Theorem stating that with 10 initial books, 100 pages per book,
    and 2 books lost, the remaining pages total 800. -/
theorem phils_remaining_pages :
  remaining_pages 10 100 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_phils_remaining_pages_l3953_395371


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3953_395333

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3953_395333


namespace NUMINAMATH_CALUDE_zoe_calorie_intake_l3953_395308

-- Define the quantities
def strawberries : ℕ := 12
def yogurt_ounces : ℕ := 6
def calories_per_strawberry : ℕ := 4
def calories_per_yogurt_ounce : ℕ := 17

-- Define the total calories
def total_calories : ℕ := strawberries * calories_per_strawberry + yogurt_ounces * calories_per_yogurt_ounce

-- Theorem statement
theorem zoe_calorie_intake : total_calories = 150 := by
  sorry

end NUMINAMATH_CALUDE_zoe_calorie_intake_l3953_395308


namespace NUMINAMATH_CALUDE_cube_sum_power_of_two_l3953_395397

theorem cube_sum_power_of_two (x y : ℤ) :
  x^3 + y^3 = 2^30 ↔ (x = 0 ∧ y = 2^10) ∨ (x = 2^10 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_power_of_two_l3953_395397


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3953_395347

theorem trigonometric_equation_solution (α : Real) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3953_395347
