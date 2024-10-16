import Mathlib

namespace NUMINAMATH_CALUDE_division_of_fractions_l1601_160139

theorem division_of_fractions : (5 : ℚ) / 6 / ((2 : ℚ) / 3) = (5 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1601_160139


namespace NUMINAMATH_CALUDE_high_school_student_distribution_l1601_160131

theorem high_school_student_distribution :
  ∀ (total juniors sophomores freshmen seniors : ℕ),
    total = 800 →
    juniors = (27 * total) / 100 →
    sophomores = total - (75 * total) / 100 →
    seniors = 160 →
    freshmen = total - (juniors + sophomores + seniors) →
    freshmen - sophomores = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_high_school_student_distribution_l1601_160131


namespace NUMINAMATH_CALUDE_seven_digit_palindromes_count_l1601_160162

/-- A function that counts the number of seven-digit palindromes with leading digit 1 or 2 -/
def count_seven_digit_palindromes : ℕ :=
  let leading_digits := 2  -- Number of choices for the leading digit (1 or 2)
  let middle_digits := 10 * 10 * 10  -- Number of choices for the middle three digits
  leading_digits * middle_digits

/-- Theorem stating that the number of seven-digit palindromes with leading digit 1 or 2 is 2000 -/
theorem seven_digit_palindromes_count : count_seven_digit_palindromes = 2000 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_palindromes_count_l1601_160162


namespace NUMINAMATH_CALUDE_holly_chocolate_milk_container_size_l1601_160174

/-- Represents the amount of chocolate milk Holly has throughout the day -/
structure ChocolateMilk where
  initial : ℕ  -- Initial amount of chocolate milk
  breakfast : ℕ  -- Amount drunk at breakfast
  lunch : ℕ  -- Amount drunk at lunch
  dinner : ℕ  -- Amount drunk at dinner
  final : ℕ  -- Final amount of chocolate milk
  new_container : ℕ  -- Size of the new container bought at lunch

/-- Theorem stating the size of the new container Holly bought -/
theorem holly_chocolate_milk_container_size 
  (h : ChocolateMilk) 
  (h_initial : h.initial = 16)
  (h_breakfast : h.breakfast = 8)
  (h_lunch : h.lunch = 8)
  (h_dinner : h.dinner = 8)
  (h_final : h.final = 56)
  : h.new_container = 64 := by
  sorry

end NUMINAMATH_CALUDE_holly_chocolate_milk_container_size_l1601_160174


namespace NUMINAMATH_CALUDE_beth_crayons_l1601_160175

/-- The number of crayon packs Beth has -/
def num_packs : ℕ := 8

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 12

/-- The number of extra crayons Beth has -/
def extra_crayons : ℕ := 15

/-- The number of crayons Beth borrowed from her friend -/
def borrowed_crayons : ℕ := 7

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := num_packs * crayons_per_pack + extra_crayons + borrowed_crayons

theorem beth_crayons :
  total_crayons = 118 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l1601_160175


namespace NUMINAMATH_CALUDE_ratio_from_S1_ratio_from_S1_S2_ratio_from_S2_l1601_160147

/-- Represents a trapezoid with diagonals intersecting at a point -/
structure Trapezoid where
  S : ℝ  -- Area of the trapezoid
  S1 : ℝ  -- Area of triangle OBC
  S2 : ℝ  -- Area of triangle OCD
  S3 : ℝ  -- Area of triangle ODA
  S4 : ℝ  -- Area of triangle AOB
  AD : ℝ  -- Length of side AD
  BC : ℝ  -- Length of side BC

/-- There exists a function that determines AD/BC given S1/S -/
theorem ratio_from_S1 (t : Trapezoid) : 
  ∃ f : ℝ → ℝ, t.AD / t.BC = f (t.S1 / t.S) :=
sorry

/-- There exists a function that determines AD/BC given (S1+S2)/S -/
theorem ratio_from_S1_S2 (t : Trapezoid) : 
  ∃ f : ℝ → ℝ, t.AD / t.BC = f ((t.S1 + t.S2) / t.S) :=
sorry

/-- There exists a function that determines AD/BC given S2/S -/
theorem ratio_from_S2 (t : Trapezoid) : 
  ∃ f : ℝ → ℝ, t.AD / t.BC = f (t.S2 / t.S) :=
sorry

end NUMINAMATH_CALUDE_ratio_from_S1_ratio_from_S1_S2_ratio_from_S2_l1601_160147


namespace NUMINAMATH_CALUDE_total_nut_weight_l1601_160142

/-- Represents the weight of nuts in kilograms -/
structure NutWeight where
  almonds : Float
  pecans : Float
  walnuts : Float
  cashews : Float
  pistachios : Float
  brazilNuts : Float
  macadamiaNuts : Float
  hazelnuts : Float

/-- Conversion rate from ounces to kilograms -/
def ounceToKgRate : Float := 0.0283495

/-- Weights of nuts bought by the chef -/
def chefNuts : NutWeight where
  almonds := 0.14
  pecans := 0.38
  walnuts := 0.22
  cashews := 0.47
  pistachios := 0.29
  brazilNuts := 6 * ounceToKgRate
  macadamiaNuts := 4.5 * ounceToKgRate
  hazelnuts := 7.3 * ounceToKgRate

/-- Theorem stating the total weight of nuts bought by the chef -/
theorem total_nut_weight : 
  chefNuts.almonds + chefNuts.pecans + chefNuts.walnuts + chefNuts.cashews + 
  chefNuts.pistachios + chefNuts.brazilNuts + chefNuts.macadamiaNuts + 
  chefNuts.hazelnuts = 2.1128216 := by
  sorry

end NUMINAMATH_CALUDE_total_nut_weight_l1601_160142


namespace NUMINAMATH_CALUDE_mass_of_man_in_boat_l1601_160190

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth sink_depth : Real) : Real :=
  boat_length * boat_breadth * sink_depth * 1000

/-- Theorem stating the mass of the man in the given problem. -/
theorem mass_of_man_in_boat : mass_of_man 3 2 0.02 = 120 := by
  sorry

#eval mass_of_man 3 2 0.02

end NUMINAMATH_CALUDE_mass_of_man_in_boat_l1601_160190


namespace NUMINAMATH_CALUDE_power_addition_equality_l1601_160182

theorem power_addition_equality : 2^345 + 9^4 / 9^2 = 2^345 + 81 := by
  sorry

end NUMINAMATH_CALUDE_power_addition_equality_l1601_160182


namespace NUMINAMATH_CALUDE_delivery_problem_l1601_160146

theorem delivery_problem (total : ℕ) (cider : ℕ) (beer : ℕ) 
  (h_total : total = 180)
  (h_cider : cider = 40)
  (h_beer : beer = 80) :
  let mixture := total - (cider + beer)
  (cider / 2 + beer / 2 + mixture / 2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_delivery_problem_l1601_160146


namespace NUMINAMATH_CALUDE_inscribed_hemisphere_volume_l1601_160171

/-- Given a cone with height 4 cm and slant height 5 cm, the volume of an inscribed hemisphere
    whose base lies on the base of the cone is (1152/125)π cm³. -/
theorem inscribed_hemisphere_volume (h : ℝ) (l : ℝ) (r : ℝ) :
  h = 4 →
  l = 5 →
  l^2 = h^2 + r^2 →
  (∃ x, x > 0 ∧ x < h ∧ r^2 + (l - x)^2 = h^2 ∧ x^2 + r^2 = r^2) →
  (2/3) * π * ((12/5)^3) = (1152/125) * π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_hemisphere_volume_l1601_160171


namespace NUMINAMATH_CALUDE_all_positive_rationals_in_X_l1601_160165

theorem all_positive_rationals_in_X (X : Set ℚ) 
  (h1 : ∀ x : ℚ, 2021 ≤ x ∧ x ≤ 2022 → x ∈ X) 
  (h2 : ∀ x y : ℚ, x ∈ X → y ∈ X → (x / y) ∈ X) :
  ∀ q : ℚ, 0 < q → q ∈ X := by
  sorry

end NUMINAMATH_CALUDE_all_positive_rationals_in_X_l1601_160165


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l1601_160187

theorem integer_fraction_characterization (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℚ) / (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1) = k.val) ↔
  (∃ l : ℕ+, (a = 2 * l ∧ b = 1) ∨ 
             (a = l ∧ b = 2 * l) ∨ 
             (a = 8 * l.val ^ 4 - l ∧ b = 2 * l)) :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l1601_160187


namespace NUMINAMATH_CALUDE_hundred_days_from_friday_is_sunday_l1601_160130

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem hundred_days_from_friday_is_sunday :
  advanceDay DayOfWeek.Friday 100 = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_hundred_days_from_friday_is_sunday_l1601_160130


namespace NUMINAMATH_CALUDE_butterfly_development_time_l1601_160186

/-- The time (in days) a butterfly spends in a cocoon -/
def cocoon_time : ℕ := 30

/-- The time (in days) a butterfly spends as a larva -/
def larva_time : ℕ := 3 * cocoon_time

/-- The total time (in days) from butterfly egg to butterfly -/
def total_time : ℕ := larva_time + cocoon_time

theorem butterfly_development_time : total_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_development_time_l1601_160186


namespace NUMINAMATH_CALUDE_train_speed_problem_l1601_160197

/-- Proves that given a journey of 133.33 km, if a train arrives on time at 100 kmph
    and arrives 20 minutes late at speed v kmph, then v = 80 kmph. -/
theorem train_speed_problem (journey_length : ℝ) (normal_speed : ℝ) (late_time : ℝ) (v : ℝ) :
  journey_length = 133.33 →
  normal_speed = 100 →
  late_time = 1/3 →
  journey_length / normal_speed + late_time = journey_length / v →
  v = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1601_160197


namespace NUMINAMATH_CALUDE_last_digit_of_2008_power_last_digit_of_2008_to_2008_l1601_160107

theorem last_digit_of_2008_power (n : ℕ) : n > 0 → (2008^n) % 10 = (2008^(n % 4)) % 10 := by sorry

theorem last_digit_of_2008_to_2008 : (2008^2008) % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_2008_power_last_digit_of_2008_to_2008_l1601_160107


namespace NUMINAMATH_CALUDE_max_excellent_courses_l1601_160176

/-- A course video with two attributes: number of views and expert score -/
structure CourseVideo where
  views : ℕ
  expertScore : ℕ

/-- Defines when one course video is not inferior to another -/
def notInferior (a b : CourseVideo) : Prop :=
  a.views ≥ b.views ∨ a.expertScore ≥ b.expertScore

/-- Defines an excellent course video -/
def isExcellent (a : CourseVideo) (courses : Finset CourseVideo) : Prop :=
  ∀ b ∈ courses, b ≠ a → notInferior a b

/-- Theorem: It's possible to have 5 excellent course videos among 5 courses -/
theorem max_excellent_courses (courses : Finset CourseVideo) (h : courses.card = 5) :
  ∃ excellentCourses : Finset CourseVideo,
    excellentCourses ⊆ courses ∧
    excellentCourses.card = 5 ∧
    ∀ a ∈ excellentCourses, isExcellent a courses := by
  sorry

end NUMINAMATH_CALUDE_max_excellent_courses_l1601_160176


namespace NUMINAMATH_CALUDE_knights_seating_probability_correct_l1601_160129

/-- The probability of three knights being seated with empty chairs on either side
    when randomly placed around a circular table with n chairs. -/
def knights_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2))
  else
    0

/-- Theorem stating the probability of three knights being seated with empty chairs
    on either side when randomly placed around a circular table with n chairs. -/
theorem knights_seating_probability_correct (n : ℕ) (h : n ≥ 6) :
  knights_seating_probability n =
    (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_knights_seating_probability_correct_l1601_160129


namespace NUMINAMATH_CALUDE_train_crossing_time_l1601_160117

/-- Given a train crossing a platform, calculate the time it takes to cross a signal pole -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 333.33)
  (h3 : platform_crossing_time = 38)
  : ∃ (signal_pole_crossing_time : ℝ),
    signal_pole_crossing_time = train_length / ((train_length + platform_length) / platform_crossing_time) ∧
    (signal_pole_crossing_time ≥ 17.9 ∧ signal_pole_crossing_time ≤ 18.1) :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1601_160117


namespace NUMINAMATH_CALUDE_geese_survival_theorem_l1601_160178

/-- Represents the number of geese that survived the first year given the total number of eggs laid -/
def geese_survived_first_year (total_eggs : ℕ) : ℕ :=
  let hatched_eggs := (2 * total_eggs) / 3
  let survived_first_month := (3 * hatched_eggs) / 4
  let not_survived_first_year := (3 * survived_first_month) / 5
  survived_first_month - not_survived_first_year

/-- Theorem stating that the number of geese surviving the first year is 1/5 of the total eggs laid -/
theorem geese_survival_theorem (total_eggs : ℕ) :
  geese_survived_first_year total_eggs = total_eggs / 5 := by
  sorry

#eval geese_survived_first_year 60  -- Should output 12

end NUMINAMATH_CALUDE_geese_survival_theorem_l1601_160178


namespace NUMINAMATH_CALUDE_infinite_nested_sqrt_three_l1601_160163

theorem infinite_nested_sqrt_three : ∃ x > 0, x^2 = 3 + 2*x ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_infinite_nested_sqrt_three_l1601_160163


namespace NUMINAMATH_CALUDE_zibo_barbecue_analysis_l1601_160155

/-- Contingency table data --/
structure ContingencyData where
  male_very_like : ℕ
  male_average : ℕ
  female_very_like : ℕ
  female_average : ℕ

/-- Chi-square test result --/
inductive ChiSquareResult
  | Significant
  | NotSignificant

/-- Distribution of ξ --/
def DistributionXi := List (ℕ × ℚ)

/-- Theorem statement --/
theorem zibo_barbecue_analysis 
  (data : ContingencyData)
  (total_sample : ℕ)
  (chi_square_formula : ContingencyData → ℝ)
  (chi_square_critical : ℝ)
  (calculate_distribution : ContingencyData → DistributionXi)
  (calculate_expectation : DistributionXi → ℚ)
  (h_total : data.male_very_like + data.male_average + data.female_very_like + data.female_average = total_sample)
  (h_female_total : data.female_very_like + data.female_average = 100)
  (h_average_total : data.male_average + data.female_average = 70)
  (h_female_very_like : data.female_very_like = 2 * data.male_average)
  : 
  let chi_square_value := chi_square_formula data
  let result := if chi_square_value < chi_square_critical then ChiSquareResult.NotSignificant else ChiSquareResult.Significant
  let distribution := calculate_distribution data
  let expectation := calculate_expectation distribution
  result = ChiSquareResult.NotSignificant ∧ expectation = 17 / 6 := by
  sorry

end NUMINAMATH_CALUDE_zibo_barbecue_analysis_l1601_160155


namespace NUMINAMATH_CALUDE_variance_transformation_l1601_160122

/-- Given a sample of 10 data points, this function represents their variance. -/
def sample_variance (x : Fin 10 → ℝ) : ℝ := sorry

/-- Given a sample of 10 data points, this function represents the variance of the transformed data. -/
def transformed_variance (x : Fin 10 → ℝ) : ℝ := 
  sample_variance (fun i => 2 * x i - 1)

/-- Theorem stating the relationship between the original variance and the transformed variance. -/
theorem variance_transformation (x : Fin 10 → ℝ) 
  (h : sample_variance x = 8) : transformed_variance x = 32 := by
  sorry

end NUMINAMATH_CALUDE_variance_transformation_l1601_160122


namespace NUMINAMATH_CALUDE_max_product_with_geometric_mean_l1601_160138

theorem max_product_with_geometric_mean (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ)^((a + b) / 2) = Real.sqrt 3 → ab ≤ (1 / 4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_geometric_mean_l1601_160138


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l1601_160125

/-- An isosceles triangle with one angle of 40 degrees has two equal angles of 70 degrees each. -/
theorem isosceles_triangle_angles (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Two angles are equal (isosceles property)
  c = 40 →           -- The third angle is 40°
  a = 70 :=          -- Each of the two equal angles is 70°
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l1601_160125


namespace NUMINAMATH_CALUDE_min_transport_cost_l1601_160114

/-- Represents the transportation problem between two villages and two destinations -/
structure TransportProblem where
  villageA_supply : ℝ
  villageB_supply : ℝ
  destX_demand : ℝ
  destY_demand : ℝ
  costA_to_X : ℝ
  costA_to_Y : ℝ
  costB_to_X : ℝ
  costB_to_Y : ℝ

/-- Calculates the total transportation cost given the amount transported from A to X -/
def totalCost (p : TransportProblem) (x : ℝ) : ℝ :=
  p.costA_to_X * x + p.costA_to_Y * (p.villageA_supply - x) +
  p.costB_to_X * (p.destX_demand - x) + p.costB_to_Y * (x - (p.villageA_supply + p.villageB_supply - p.destX_demand - p.destY_demand))

/-- The specific problem instance -/
def vegetableProblem : TransportProblem :=
  { villageA_supply := 80
  , villageB_supply := 60
  , destX_demand := 65
  , destY_demand := 75
  , costA_to_X := 50
  , costA_to_Y := 30
  , costB_to_X := 60
  , costB_to_Y := 45 }

/-- Theorem stating that the minimum transportation cost for the vegetable problem is 6100 -/
theorem min_transport_cost :
  ∃ x, x ≥ 0 ∧ x ≤ vegetableProblem.villageA_supply ∧
       x ≤ vegetableProblem.destX_demand ∧
       x ≥ (vegetableProblem.villageA_supply + vegetableProblem.villageB_supply - vegetableProblem.destX_demand - vegetableProblem.destY_demand) ∧
       totalCost vegetableProblem x = 6100 ∧
       ∀ y, y ≥ 0 → y ≤ vegetableProblem.villageA_supply →
             y ≤ vegetableProblem.destX_demand →
             y ≥ (vegetableProblem.villageA_supply + vegetableProblem.villageB_supply - vegetableProblem.destX_demand - vegetableProblem.destY_demand) →
             totalCost vegetableProblem x ≤ totalCost vegetableProblem y :=
by sorry


end NUMINAMATH_CALUDE_min_transport_cost_l1601_160114


namespace NUMINAMATH_CALUDE_train_speed_problem_l1601_160188

/-- Proves that given the conditions of the train problem, the speed of Train A is 43 miles per hour. -/
theorem train_speed_problem (speed_B : ℝ) (headstart : ℝ) (overtake_distance : ℝ) 
  (h1 : speed_B = 45)
  (h2 : headstart = 2)
  (h3 : overtake_distance = 180) :
  ∃ (speed_A : ℝ) (overtake_time : ℝ), 
    speed_A = 43 ∧ 
    speed_A * (headstart + overtake_time) = overtake_distance ∧
    speed_B * overtake_time = overtake_distance :=
by
  sorry


end NUMINAMATH_CALUDE_train_speed_problem_l1601_160188


namespace NUMINAMATH_CALUDE_estimate_fish_population_l1601_160123

/-- Estimate the total number of fish in a pond using mark-recapture method -/
theorem estimate_fish_population (initially_caught marked_in_second_catch second_catch : ℕ) :
  initially_caught = 30 →
  marked_in_second_catch = 2 →
  second_catch = 50 →
  (initially_caught * second_catch) / marked_in_second_catch = 750 :=
by sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l1601_160123


namespace NUMINAMATH_CALUDE_function_floor_property_l1601_160168

theorem function_floor_property (f : ℝ → ℝ) :
  (∃ g : ℝ → ℝ, ∀ x y : ℝ, f x + f y = ⌊g (x + y)⌋) →
  ∃ n : ℤ, ∀ x : ℝ, f x = n / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_floor_property_l1601_160168


namespace NUMINAMATH_CALUDE_jane_calculation_l1601_160121

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - 2 * (y - 3 * z) = 25)
  (h2 : x - 2 * y - 3 * z = 7) :
  x - 2 * y = 13 := by
sorry

end NUMINAMATH_CALUDE_jane_calculation_l1601_160121


namespace NUMINAMATH_CALUDE_parabola_intersection_l1601_160137

-- Define the parabola
def parabola (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + c

theorem parabola_intersection (a c : ℝ) (h_a : a < 0) :
  parabola a c 0 = 9 →
  parabola a c 2 = 8.1 →
  -9/49 < a ∧ a < -1/4 ∧
  ∃ x, 6 < x ∧ x < 7 ∧ parabola a c x = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1601_160137


namespace NUMINAMATH_CALUDE_scientific_notation_of_400000_l1601_160144

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The number we want to represent in scientific notation -/
def number : ℝ := 400000

/-- The expected scientific notation representation -/
def expected : ScientificNotation :=
  { coefficient := 4
  , exponent := 5
  , is_valid := by sorry }

theorem scientific_notation_of_400000 :
  toScientificNotation number = expected := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_400000_l1601_160144


namespace NUMINAMATH_CALUDE_perimeter_inscribable_equivalence_l1601_160145

/-- Triangle represented by its side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a line segment intersecting two sides of a triangle -/
structure IntersectingLine (T : Triangle) where
  A' : ℝ  -- Distance from A to A' on AC
  B' : ℝ  -- Distance from B to B' on BC

/-- Condition for the perimeter of the inner triangle -/
def perimeterCondition (T : Triangle) (L : IntersectingLine T) : Prop :=
  L.A' + L.B' + (T.c - L.A' - L.B') = T.a + T.b - T.c

/-- Condition for the quadrilateral to be inscribable -/
def inscribableCondition (T : Triangle) (L : IntersectingLine T) : Prop :=
  T.c + (T.a + T.b - T.c - (L.A' + L.B')) = (T.a - L.A') + (T.b - L.B')

theorem perimeter_inscribable_equivalence (T : Triangle) (L : IntersectingLine T) :
  perimeterCondition T L ↔ inscribableCondition T L := by sorry

end NUMINAMATH_CALUDE_perimeter_inscribable_equivalence_l1601_160145


namespace NUMINAMATH_CALUDE_next_number_with_property_is_four_digit_number_l1601_160115

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that splits a four-digit number into its hundreds-tens and ones-tens parts -/
def splitNumber (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

/-- The property we're looking for in the number -/
def hasDesiredProperty (n : ℕ) : Prop :=
  let (a, b) := splitNumber n
  isPerfectSquare (a * b)

/-- Theorem stating that 1832 is the next number after 1818 with the desired property -/
theorem next_number_with_property :
  ∀ n : ℕ, 1818 < n ∧ n < 1832 → ¬(hasDesiredProperty n) ∧ hasDesiredProperty 1832 :=
by sorry

/-- Theorem stating that 1832 is indeed a four-digit number -/
theorem is_four_digit_number :
  1000 ≤ 1832 ∧ 1832 < 10000 :=
by sorry

end NUMINAMATH_CALUDE_next_number_with_property_is_four_digit_number_l1601_160115


namespace NUMINAMATH_CALUDE_min_sum_positive_reals_l1601_160109

theorem min_sum_positive_reals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2 * a + 8 * b - a * b = 0 → x + y ≤ a + b ∧ x + y = 6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_positive_reals_l1601_160109


namespace NUMINAMATH_CALUDE_quadratic_completion_l1601_160135

/-- The quadratic function we're working with -/
def f (x : ℝ) : ℝ := x^2 - 24*x + 50

/-- The completed square form of our quadratic -/
def g (x b c : ℝ) : ℝ := (x + b)^2 + c

/-- Theorem stating that f can be written in the form of g, and b + c = -106 -/
theorem quadratic_completion (b c : ℝ) : 
  (∀ x, f x = g x b c) → b + c = -106 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l1601_160135


namespace NUMINAMATH_CALUDE_vector_linear_combination_l1601_160151

/-- Given vectors a, b, and c in R^2, prove that if c = x*a + y*b,
    then x + y = 8/3 -/
theorem vector_linear_combination (a b c : ℝ × ℝ) (x y : ℝ) 
    (h1 : a = (2, 3))
    (h2 : b = (3, 3))
    (h3 : c = (7, 8))
    (h4 : c = x • a + y • b) :
  x + y = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l1601_160151


namespace NUMINAMATH_CALUDE_geometric_series_equality_l1601_160133

def C (n : ℕ) : ℚ := 2048 * (1 - (1 / 2^n))

def D (n : ℕ) : ℚ := (6144 / 3) * (1 - (1 / (-2)^n))

theorem geometric_series_equality (n : ℕ) (h : n ≥ 1) : C n = D n ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l1601_160133


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l1601_160194

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (l m : Line) (α : Plane) :
  parallel l m → perpendicular l α → perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l1601_160194


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1601_160157

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ (r₁ r₂ r₃ : ℕ+), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    (∀ (x : ℝ), x^3 - 6*x^2 + p*x - q = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃))) →
  p + q = 17 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1601_160157


namespace NUMINAMATH_CALUDE_evaluate_expression_l1601_160140

theorem evaluate_expression : 10010 - 12 * 3 * 2 = 9938 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1601_160140


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1601_160180

-- Define the function f(x) = -x^3 + 3x^2
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem tangent_line_equation (a : ℝ) :
  ∃ b : ℝ, f' a = 3 ∧ f a = 3*a + b :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1601_160180


namespace NUMINAMATH_CALUDE_point_b_coordinates_l1601_160141

/-- Given points A and C, and the condition that vector AB is -2 times vector BC,
    prove that the coordinates of point B are (-2, -1). -/
theorem point_b_coordinates (A B C : ℝ × ℝ) : 
  A = (2, 3) → 
  C = (0, 1) → 
  B - A = -2 * (C - B) →
  B = (-2, -1) := by
sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l1601_160141


namespace NUMINAMATH_CALUDE_puppy_weight_l1601_160102

theorem puppy_weight (puppy smaller_cat larger_cat bird : ℝ) 
  (total_weight : puppy + smaller_cat + larger_cat + bird = 34)
  (larger_cat_weight : puppy + larger_cat = 3 * bird)
  (smaller_cat_weight : puppy + smaller_cat = 2 * bird) :
  puppy = 17 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l1601_160102


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1601_160106

theorem smallest_constant_inequality (x y : ℝ) :
  ∃ (C : ℝ), C = 4/3 ∧ (∀ (x y : ℝ), 1 + (x + y)^2 ≤ C * (1 + x^2) * (1 + y^2)) ∧
  (∀ (D : ℝ), (∀ (x y : ℝ), 1 + (x + y)^2 ≤ D * (1 + x^2) * (1 + y^2)) → C ≤ D) :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1601_160106


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1601_160154

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ r₁ r₂ : ℝ, r₁ + r₂ = 4 ∧ r₁ * r₂ = 6 ∧ 
      3 * r₁^2 - p * r₁ + q = 0 ∧ 
      3 * r₂^2 - p * r₂ + q = 0)) → 
  p = 12 ∧ q = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1601_160154


namespace NUMINAMATH_CALUDE_laptop_price_calculation_l1601_160126

/-- Calculate the total selling price of a laptop given the original price, discount rate, coupon value, and tax rate -/
def totalSellingPrice (originalPrice discountRate couponValue taxRate : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  let priceAfterCoupon := discountedPrice - couponValue
  let finalPrice := priceAfterCoupon * (1 + taxRate)
  finalPrice

/-- Theorem stating that the total selling price of the laptop is 908.5 dollars -/
theorem laptop_price_calculation :
  totalSellingPrice 1200 0.30 50 0.15 = 908.5 := by
  sorry


end NUMINAMATH_CALUDE_laptop_price_calculation_l1601_160126


namespace NUMINAMATH_CALUDE_book_sale_price_l1601_160108

theorem book_sale_price (total_books : ℕ) (sold_books : ℕ) (unsold_books : ℕ) (total_amount : ℕ) : 
  sold_books = (2 : ℕ) * total_books / 3 →
  unsold_books = 36 →
  sold_books + unsold_books = total_books →
  total_amount = 288 →
  total_amount / sold_books = 4 := by
sorry

end NUMINAMATH_CALUDE_book_sale_price_l1601_160108


namespace NUMINAMATH_CALUDE_complement_of_A_l1601_160172

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}

theorem complement_of_A :
  (U \ A) = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1601_160172


namespace NUMINAMATH_CALUDE_pyramid_cube_tiling_exists_l1601_160159

/-- A shape constructed from a cube with a pyramid on one face -/
structure PyramidCube where
  -- The edge length of the base cube
  cube_edge : ℝ
  -- The height of the pyramid (assumed to be equal to cube_edge)
  pyramid_height : ℝ
  -- Assumption that the pyramid height equals the cube edge length
  height_eq_edge : pyramid_height = cube_edge

/-- A tiling of 3D space using congruent copies of a shape -/
structure Tiling (shape : PyramidCube) where
  -- The set of positions (as points in ℝ³) where shapes are placed
  positions : Set (Fin 3 → ℝ)
  -- Property ensuring the tiling is seamless (no gaps)
  seamless : sorry
  -- Property ensuring the tiling has no overlaps
  no_overlap : sorry

/-- Theorem stating that a space-filling tiling exists for the PyramidCube shape -/
theorem pyramid_cube_tiling_exists :
  ∃ (shape : PyramidCube) (tiling : Tiling shape), True :=
sorry

end NUMINAMATH_CALUDE_pyramid_cube_tiling_exists_l1601_160159


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1601_160128

def A : Set ℝ := {x | x^2 ≠ 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a (a : ℝ) (h : A ∪ B a = A) : a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1601_160128


namespace NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l1601_160110

theorem arcsin_arccos_equation_solution :
  ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧
  Real.arcsin x + Real.arcsin (2*x) = Real.arccos x + Real.arccos (2*x) ∧
  x = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l1601_160110


namespace NUMINAMATH_CALUDE_min_value_x_plus_sqrt_x2_y2_l1601_160113

theorem min_value_x_plus_sqrt_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  ∃ (min : ℝ), min = 8/5 ∧ ∀ (z : ℝ), z > 0 → 2 * z + (2 - 2 * z) = 2 →
    x + Real.sqrt (x^2 + y^2) ≥ min ∧ z + Real.sqrt (z^2 + (2 - 2 * z)^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_sqrt_x2_y2_l1601_160113


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1601_160111

-- Define the triangle type
structure Triangle where
  inradius : ℝ
  area : ℝ

-- Theorem statement
theorem triangle_perimeter (t : Triangle) (h1 : t.inradius = 2.5) (h2 : t.area = 75) :
  2 * t.area / t.inradius = 60 := by
  sorry

#check triangle_perimeter

end NUMINAMATH_CALUDE_triangle_perimeter_l1601_160111


namespace NUMINAMATH_CALUDE_quadratic_vertex_on_line_quadratic_intersects_line_l1601_160134

/-- The quadratic function parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 1

/-- The line y = x - 1 -/
def g (x : ℝ) : ℝ := x - 1

/-- The line y = x + b parameterized by b -/
def h (b : ℝ) (x : ℝ) : ℝ := x + b

/-- The vertex of a quadratic function ax^2 + bx + c is at (-b/(2a), f(-b/(2a))) -/
def vertex (m : ℝ) : ℝ × ℝ := (m, f m m)

theorem quadratic_vertex_on_line (m : ℝ) : 
  g (vertex m).1 = (vertex m).2 :=
sorry

theorem quadratic_intersects_line (m b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = h b x₁ ∧ f m x₂ = h b x₂) ↔ b > -5/4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_on_line_quadratic_intersects_line_l1601_160134


namespace NUMINAMATH_CALUDE_diamond_45_15_l1601_160104

/-- The diamond operation on positive real numbers -/
noncomputable def diamond (x y : ℝ) : ℝ :=
  x / y

/-- Axioms for the diamond operation -/
axiom diamond_positive (x y : ℝ) : 0 < x → 0 < y → 0 < diamond x y

axiom diamond_prop1 (x y : ℝ) : 0 < x → 0 < y → diamond (x * y) y = x * diamond y y

axiom diamond_prop2 (x : ℝ) : 0 < x → diamond (diamond x 1) x = diamond x 1

axiom diamond_def (x y : ℝ) : 0 < x → 0 < y → diamond x y = x / y

axiom diamond_one : diamond 1 1 = 1

/-- Theorem: 45 ◇ 15 = 3 -/
theorem diamond_45_15 : diamond 45 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_45_15_l1601_160104


namespace NUMINAMATH_CALUDE_certain_number_problem_l1601_160173

theorem certain_number_problem : ∃! x : ℕ+, 220030 = (x + 445) * (2 * (x - 445)) + 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1601_160173


namespace NUMINAMATH_CALUDE_first_half_speed_l1601_160166

/-- Proves that given a journey of 3600 miles completed in 30 hours, 
    where the second half is traveled at 180 mph, 
    the average speed for the first half of the journey is 90 mph. -/
theorem first_half_speed (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 3600 →
  total_time = 30 →
  second_half_speed = 180 →
  (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 90 :=
by sorry

end NUMINAMATH_CALUDE_first_half_speed_l1601_160166


namespace NUMINAMATH_CALUDE_total_topping_combinations_l1601_160181

/-- Represents the number of cheese options -/
def cheese_options : ℕ := 3

/-- Represents the number of meat options -/
def meat_options : ℕ := 4

/-- Represents the number of vegetable options -/
def vegetable_options : ℕ := 5

/-- Represents whether pepperoni is a meat option -/
def pepperoni_is_meat_option : Prop := True

/-- Represents whether peppers is a vegetable option -/
def peppers_is_vegetable_option : Prop := True

/-- Represents the restriction that pepperoni and peppers cannot be chosen together -/
def pepperoni_peppers_restriction : Prop := True

/-- Theorem stating the total number of pizza topping combinations -/
theorem total_topping_combinations : 
  cheese_options * meat_options * vegetable_options - 
  cheese_options * (meat_options - 1) = 57 := by
  sorry


end NUMINAMATH_CALUDE_total_topping_combinations_l1601_160181


namespace NUMINAMATH_CALUDE_max_value_theorem_l1601_160112

theorem max_value_theorem (x y : ℝ) (h : 2 * x^2 + x * y - y^2 = 1) :
  ∃ (M : ℝ), M = Real.sqrt 2 / 4 ∧ 
  ∀ (z : ℝ), z = (x - 2*y) / (5*x^2 - 2*x*y + 2*y^2) → z ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1601_160112


namespace NUMINAMATH_CALUDE_systematic_sampling_l1601_160160

theorem systematic_sampling (total : Nat) (sample_size : Nat) (drawn : Nat) : 
  total = 800 → 
  sample_size = 50 → 
  drawn = 7 → 
  ∃ (selected : Nat), 
    selected = drawn + 2 * (total / sample_size) ∧ 
    33 ≤ selected ∧ 
    selected ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1601_160160


namespace NUMINAMATH_CALUDE_flour_needed_l1601_160167

/-- The amount of flour Katie needs in pounds -/
def katie_flour : ℕ := 3

/-- The additional amount of flour Sheila needs compared to Katie in pounds -/
def sheila_extra : ℕ := 2

/-- The total amount of flour needed by Katie and Sheila -/
def total_flour : ℕ := katie_flour + (katie_flour + sheila_extra)

theorem flour_needed : total_flour = 8 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_l1601_160167


namespace NUMINAMATH_CALUDE_trains_meeting_time_l1601_160148

/-- The time taken for two trains to meet under specific conditions -/
theorem trains_meeting_time : 
  let train1_length : ℝ := 300
  let train1_crossing_time : ℝ := 20
  let train2_length : ℝ := 450
  let train2_speed_kmh : ℝ := 90
  let train1_speed : ℝ := train1_length / train1_crossing_time
  let train2_speed : ℝ := train2_speed_kmh * 1000 / 3600
  let relative_speed : ℝ := train1_speed + train2_speed
  let total_distance : ℝ := train1_length + train2_length
  let meeting_time : ℝ := total_distance / relative_speed
  meeting_time = 18.75 := by sorry

end NUMINAMATH_CALUDE_trains_meeting_time_l1601_160148


namespace NUMINAMATH_CALUDE_cubic_function_property_l1601_160127

/-- Given a cubic function f(x) = ax³ + bx - 4 where a and b are constants,
    if f(-2) = 2, then f(2) = -10 -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^3 + b * x - 4)
    (h2 : f (-2) = 2) : 
  f 2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1601_160127


namespace NUMINAMATH_CALUDE_sequence_general_term_l1601_160195

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = n^2 - 4n,
    prove that the general term a_n is equal to 2n - 5. -/
theorem sequence_general_term (a : ℕ → ℤ) (S : ℕ → ℤ)
    (h : ∀ n : ℕ, S n = n^2 - 4*n) :
  ∀ n : ℕ, a n = 2*n - 5 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1601_160195


namespace NUMINAMATH_CALUDE_symmetric_about_origin_l1601_160185

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function g: ℝ → ℝ is even if g(-x) = g(x) for all x ∈ ℝ -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

/-- A function v: ℝ → ℝ is symmetric about the origin if v(-x) = -v(x) for all x ∈ ℝ -/
def SymmetricAboutOrigin (v : ℝ → ℝ) : Prop := ∀ x, v (-x) = -v x

/-- Main theorem: If f is odd and g is even, then v(x) = f(x)|g(x)| is symmetric about the origin -/
theorem symmetric_about_origin (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  SymmetricAboutOrigin (fun x ↦ f x * |g x|) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_about_origin_l1601_160185


namespace NUMINAMATH_CALUDE_max_value_of_m_l1601_160193

theorem max_value_of_m :
  (∀ x : ℝ, x < m → x^2 - 2*x - 8 > 0) →
  (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m) →
  (∀ ε > 0, ∃ x : ℝ, x < -2 + ε ∧ x ≥ m) →
  m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_m_l1601_160193


namespace NUMINAMATH_CALUDE_point_on_graph_l1601_160116

theorem point_on_graph (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a * x - 1
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_graph_l1601_160116


namespace NUMINAMATH_CALUDE_towel_price_calculation_l1601_160169

theorem towel_price_calculation (price1 price2 avg_price : ℕ) 
  (h1 : price1 = 100)
  (h2 : price2 = 150)
  (h3 : avg_price = 145) : 
  ∃ (unknown_price : ℕ), 
    (3 * price1 + 5 * price2 + 2 * unknown_price) / 10 = avg_price ∧ 
    unknown_price = 200 := by
sorry

end NUMINAMATH_CALUDE_towel_price_calculation_l1601_160169


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l1601_160196

theorem coefficient_x4_in_expansion : 
  let expansion := (fun x => (1 - x)^5 * (2*x + 1))
  ∃ (a b c d e f : ℚ), 
    ∀ x, expansion x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f ∧ b = -15 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l1601_160196


namespace NUMINAMATH_CALUDE_required_speed_for_average_l1601_160152

/-- Proves the required speed for the last part of a journey to achieve a desired average speed --/
theorem required_speed_for_average 
  (total_time : ℝ) 
  (initial_time : ℝ) 
  (initial_speed : ℝ) 
  (desired_avg_speed : ℝ) 
  (h1 : total_time = 5) 
  (h2 : initial_time = 3) 
  (h3 : initial_speed = 60) 
  (h4 : desired_avg_speed = 70) : 
  (desired_avg_speed * total_time - initial_speed * initial_time) / (total_time - initial_time) = 85 := by
  sorry

#check required_speed_for_average

end NUMINAMATH_CALUDE_required_speed_for_average_l1601_160152


namespace NUMINAMATH_CALUDE_ellipse_to_hyperbola_l1601_160164

/-- Given an ellipse with equation x²/4 + y²/2 = 1, 
    prove that the equation of the hyperbola with its vertices at the foci of the ellipse 
    and its foci at the vertices of the ellipse is x²/2 - y²/2 = 1 -/
theorem ellipse_to_hyperbola (x y : ℝ) :
  (x^2 / 4 + y^2 / 2 = 1) →
  ∃ (a b : ℝ), (a^2 = 2 ∧ b^2 = 2) ∧
  (x^2 / a^2 - y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_to_hyperbola_l1601_160164


namespace NUMINAMATH_CALUDE_expression_equality_l1601_160161

theorem expression_equality : 
  (Real.log 5) ^ 0 + (9 / 4) ^ (1 / 2) + Real.sqrt ((1 - Real.sqrt 2) ^ 2) - 2 ^ (Real.log 2 / Real.log 4) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1601_160161


namespace NUMINAMATH_CALUDE_elizabeth_ate_four_bananas_l1601_160156

/-- The number of bananas Elizabeth ate -/
def bananas_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Elizabeth ate 4 bananas -/
theorem elizabeth_ate_four_bananas :
  let initial := 12
  let remaining := 8
  bananas_eaten initial remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_ate_four_bananas_l1601_160156


namespace NUMINAMATH_CALUDE_vanessa_score_l1601_160120

theorem vanessa_score (team_score : ℕ) (other_players : ℕ) (other_avg : ℚ) : 
  team_score = 48 → 
  other_players = 6 → 
  other_avg = 3.5 → 
  team_score - (other_players : ℚ) * other_avg = 27 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_score_l1601_160120


namespace NUMINAMATH_CALUDE_cube_intersection_probability_l1601_160103

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  vertices : Finset (Fin 8)
  faces : Finset (Finset (Fin 4))
  vertex_count : vertices.card = 8
  face_count : faces.card = 6

/-- A function that determines if three vertices form a plane intersecting the cube's interior -/
def plane_intersects_interior (c : Cube) (v1 v2 v3 : Fin 8) : Prop :=
  sorry

/-- The probability of three randomly chosen distinct vertices forming a plane
    that intersects the interior of the cube -/
def intersection_probability (c : Cube) : ℚ :=
  sorry

theorem cube_intersection_probability (c : Cube) :
  intersection_probability c = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_cube_intersection_probability_l1601_160103


namespace NUMINAMATH_CALUDE_quadratic_single_solution_l1601_160184

theorem quadratic_single_solution (p : ℝ) : 
  (∃! y : ℝ, 2 * y^2 - 8 * y = p) → p = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_single_solution_l1601_160184


namespace NUMINAMATH_CALUDE_carton_height_theorem_l1601_160119

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dim : BoxDimensions) : ℕ :=
  dim.length * dim.width * dim.height

/-- Calculates the area of a rectangle given its length and width -/
def rectangleArea (length width : ℕ) : ℕ :=
  length * width

/-- Calculates the number of smaller rectangles that can fit in a larger rectangle -/
def fitRectangles (largeLength largeWidth smallLength smallWidth : ℕ) : ℕ :=
  (largeLength / smallLength) * (largeWidth / smallWidth)

/-- The main theorem about the carton height -/
theorem carton_height_theorem 
  (cartonLength cartonWidth : ℕ)
  (soapBox : BoxDimensions)
  (maxSoapBoxes : ℕ) :
  cartonLength = 30 →
  cartonWidth = 42 →
  soapBox.length = 7 →
  soapBox.width = 6 →
  soapBox.height = 5 →
  maxSoapBoxes = 360 →
  ∃ (cartonHeight : ℕ), cartonHeight = 60 ∧
    cartonHeight * fitRectangles cartonLength cartonWidth soapBox.length soapBox.width = 
    maxSoapBoxes * soapBox.height :=
sorry

end NUMINAMATH_CALUDE_carton_height_theorem_l1601_160119


namespace NUMINAMATH_CALUDE_car_speed_ratio_l1601_160101

/-- Two cars traveling towards each other meet at a point. 
    v₁ is the speed of the first car (from A to B).
    v₂ is the speed of the second car (from B to A).
    D is the distance between A and B.
    t is the time from start until the cars meet. -/
theorem car_speed_ratio 
  (v₁ v₂ D t : ℝ) 
  (h₁ : v₁ > 0)
  (h₂ : v₂ > 0)
  (h₃ : D > 0)
  (h₄ : t > 0)
  (h₅ : v₁ * t + v₂ * t = D)  -- Total distance covered by both cars
  (h₆ : D - v₁ * t = v₁)      -- Remaining distance for first car
  (h₇ : D - v₂ * t = 4 * v₂)  -- Remaining distance for second car
  : v₁ / v₂ = 2 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_ratio_l1601_160101


namespace NUMINAMATH_CALUDE_geneticallyModifiedMicroorganismsAllocation_l1601_160105

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  homeElectronics : ℝ
  foodAdditives : ℝ
  industrialLubricants : ℝ
  basicAstrophysics : ℝ
  geneticallyModifiedMicroorganisms : ℝ

/-- The total budget percentage --/
def totalBudgetPercentage : ℝ := 100

/-- The total degrees in a circle --/
def totalDegrees : ℝ := 360

/-- Theorem stating the percentage allocated to genetically modified microorganisms --/
theorem geneticallyModifiedMicroorganismsAllocation (budget : BudgetAllocation) : 
  budget.microphotonics = 12 ∧ 
  budget.homeElectronics = 24 ∧ 
  budget.foodAdditives = 15 ∧ 
  budget.industrialLubricants = 8 ∧ 
  budget.basicAstrophysics * (totalBudgetPercentage / totalDegrees) = 12 ∧
  budget.microphotonics + budget.homeElectronics + budget.foodAdditives + 
    budget.industrialLubricants + budget.basicAstrophysics + 
    budget.geneticallyModifiedMicroorganisms = totalBudgetPercentage →
  budget.geneticallyModifiedMicroorganisms = 29 := by
  sorry


end NUMINAMATH_CALUDE_geneticallyModifiedMicroorganismsAllocation_l1601_160105


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1601_160199

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (a : Fin 2 → Real) (b : Fin 2 → Real)
  (ha : a = ![- 2, Real.cos α])
  (hb : b = ![- 1, Real.sin α])
  (parallel : ∃ (k : Real), a = k • b) :
  Real.tan (α + π / 4) = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1601_160199


namespace NUMINAMATH_CALUDE_alternative_increase_is_nineteen_cents_l1601_160153

/-- Represents the fine structure for overdue books in a library --/
structure OverdueFine where
  initial_fine : ℚ
  standard_increase : ℚ
  alternative_increase : ℚ
  fifth_day_fine : ℚ

/-- Calculates the fine for a given number of days overdue --/
def calculate_fine (f : OverdueFine) (days : ℕ) : ℚ :=
  f.initial_fine + (days - 1) * min f.standard_increase f.alternative_increase

/-- Theorem stating that the alternative increase is $0.19 --/
theorem alternative_increase_is_nineteen_cents 
  (f : OverdueFine) 
  (h1 : f.initial_fine = 7/100)
  (h2 : f.standard_increase = 30/100)
  (h3 : f.fifth_day_fine = 86/100) : 
  f.alternative_increase = 19/100 := by
  sorry

#eval let f : OverdueFine := {
  initial_fine := 7/100,
  standard_increase := 30/100,
  alternative_increase := 19/100,
  fifth_day_fine := 86/100
}
calculate_fine f 5

end NUMINAMATH_CALUDE_alternative_increase_is_nineteen_cents_l1601_160153


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1601_160150

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  Real.sqrt (a + 1) + Real.sqrt (b + 2) ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1601_160150


namespace NUMINAMATH_CALUDE_complex_power_2006_l1601_160170

def i : ℂ := Complex.I

theorem complex_power_2006 : ((1 + i) / (1 - i)) ^ 2006 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2006_l1601_160170


namespace NUMINAMATH_CALUDE_classroom_problem_l1601_160191

/-- Calculates the final number of children in a classroom after some changes -/
def final_children_count (initial_boys initial_girls boys_left girls_entered : ℕ) : ℕ :=
  (initial_boys - boys_left) + (initial_girls + girls_entered)

/-- Proves that the final number of children in the classroom is 8 -/
theorem classroom_problem :
  let initial_boys : ℕ := 5
  let initial_girls : ℕ := 4
  let boys_left : ℕ := 3
  let girls_entered : ℕ := 2
  final_children_count initial_boys initial_girls boys_left girls_entered = 8 := by
  sorry

#eval final_children_count 5 4 3 2

end NUMINAMATH_CALUDE_classroom_problem_l1601_160191


namespace NUMINAMATH_CALUDE_projection_region_area_l1601_160183

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- The region inside the trapezoid with the given projection property -/
def ProjectionRegion (t : IsoscelesTrapezoid) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem -/
theorem projection_region_area (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 1) (h2 : t.base2 = 2) (h3 : t.height = 1) : 
  area (ProjectionRegion t) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_projection_region_area_l1601_160183


namespace NUMINAMATH_CALUDE_dollar_cube_difference_l1601_160192

-- Define the $ operation for real numbers
def dollar (a b : ℝ) : ℝ := (a - b)^3

-- Theorem statement
theorem dollar_cube_difference (x y : ℝ) :
  dollar ((x - y)^3) ((y - x)^3) = -8 * (y - x)^9 := by
  sorry

end NUMINAMATH_CALUDE_dollar_cube_difference_l1601_160192


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1601_160118

theorem partial_fraction_decomposition_product : 
  ∀ (A B C : ℚ),
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 →
    (x^2 - 23) / (x^3 - 3*x^2 - 4*x + 12) = 
    A / (x - 1) + B / (x + 3) + C / (x - 4)) →
  A * B * C = 11/36 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1601_160118


namespace NUMINAMATH_CALUDE_min_lateral_surface_area_cone_l1601_160143

/-- Given a cone with volume 4π/3, its minimum lateral surface area is 2√3π. -/
theorem min_lateral_surface_area_cone (r h : ℝ) (h_volume : (1/3) * π * r^2 * h = (4/3) * π) :
  ∃ (S_min : ℝ), S_min = 2 * Real.sqrt 3 * π ∧ 
  ∀ (S : ℝ), S = π * r * Real.sqrt (r^2 + h^2) → S ≥ S_min :=
sorry

end NUMINAMATH_CALUDE_min_lateral_surface_area_cone_l1601_160143


namespace NUMINAMATH_CALUDE_production_rate_equation_l1601_160189

theorem production_rate_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_diff : x = y + 4) :
  (100 / x = 80 / y) ↔ 
  (∃ (rate_A rate_B : ℝ), 
    rate_A = x ∧ 
    rate_B = y ∧ 
    rate_A > rate_B ∧ 
    rate_A - rate_B = 4 ∧
    (100 / rate_A) = (80 / rate_B)) :=
by sorry

end NUMINAMATH_CALUDE_production_rate_equation_l1601_160189


namespace NUMINAMATH_CALUDE_sharon_drive_distance_l1601_160158

theorem sharon_drive_distance :
  let usual_time : ℝ := 180
  let snowstorm_time : ℝ := 300
  let speed_decrease : ℝ := 30
  let distance : ℝ := 157.5
  let usual_speed : ℝ := distance / usual_time
  let snowstorm_speed : ℝ := usual_speed - speed_decrease / 60
  (distance / 2) / usual_speed + (distance / 2) / snowstorm_speed = snowstorm_time :=
by sorry

end NUMINAMATH_CALUDE_sharon_drive_distance_l1601_160158


namespace NUMINAMATH_CALUDE_scalene_triangle_unique_x_l1601_160149

/-- Represents a scalene triangle with specific properties -/
structure ScaleneTriangle where
  -- One angle is 45 degrees
  angle1 : ℝ
  angle1_eq : angle1 = 45
  -- Another angle is x degrees
  angle2 : ℝ
  -- The third angle
  angle3 : ℝ
  -- The sum of all angles is 180 degrees
  angle_sum : angle1 + angle2 + angle3 = 180
  -- The sides opposite angle1 and angle2 are equal
  equal_sides : True
  -- The triangle is scalene (all sides are different)
  is_scalene : True

/-- 
Theorem: In a scalene triangle with one angle of 45° and another angle of x°, 
where the side lengths opposite these two angles are equal, 
the only possible value for x is 45°.
-/
theorem scalene_triangle_unique_x (t : ScaleneTriangle) : t.angle2 = 45 := by
  sorry

#check scalene_triangle_unique_x

end NUMINAMATH_CALUDE_scalene_triangle_unique_x_l1601_160149


namespace NUMINAMATH_CALUDE_trevor_eggs_left_l1601_160124

/-- Given the number of eggs laid by each chicken and the number of eggs dropped,
    prove that the number of eggs Trevor has left is equal to the total number
    of eggs collected minus the number of eggs dropped. -/
theorem trevor_eggs_left (gertrude blanche nancy martha dropped : ℕ) :
  gertrude + blanche + nancy + martha - dropped =
  (gertrude + blanche + nancy + martha) - dropped :=
by sorry

end NUMINAMATH_CALUDE_trevor_eggs_left_l1601_160124


namespace NUMINAMATH_CALUDE_product_of_largest_primes_l1601_160177

def largest_one_digit_primes : Finset Nat := {5, 7}
def largest_two_digit_prime : Nat := 97

theorem product_of_largest_primes : 
  (Finset.prod largest_one_digit_primes id) * largest_two_digit_prime = 3395 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_l1601_160177


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l1601_160136

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ 
  p ∣ (Nat.factorial 15 + Nat.factorial 17) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (Nat.factorial 15 + Nat.factorial 17) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l1601_160136


namespace NUMINAMATH_CALUDE_classroom_students_classroom_students_proof_l1601_160132

theorem classroom_students : ℕ → Prop :=
  fun S : ℕ =>
    let boys := S / 3
    let girls := S - boys
    let girls_with_dogs := (40 * girls) / 100
    let girls_with_cats := (20 * girls) / 100
    let girls_without_pets := girls - girls_with_dogs - girls_with_cats
    girls_without_pets = 8 → S = 30

-- The proof goes here
theorem classroom_students_proof : classroom_students 30 := by
  sorry

end NUMINAMATH_CALUDE_classroom_students_classroom_students_proof_l1601_160132


namespace NUMINAMATH_CALUDE_ellipse_properties_l1601_160198

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt 3 / 2
  let d := 2 * Real.sqrt 5 / 5
  let c := e * a
  (c^2 = a^2 - b^2) →
  (a * b / Real.sqrt (a^2 + b^2) = d) →
  (∃ (k : ℝ), 
    (a = 2 ∧ b = 1) ∧
    (∀ (x y : ℝ), x^2/4 + y^2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    (k = 3 * Real.sqrt 14 / 14 ∨ k = -3 * Real.sqrt 14 / 14) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      x1^2/4 + y1^2 = 1 ∧
      x2^2/4 + y2^2 = 1 ∧
      y1 = k * x1 + 5/3 ∧
      y2 = k * x2 + 5/3 ∧
      x1 = 2 * x2 ∧
      y1 - 5/3 = 2 * (y2 - 5/3))) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1601_160198


namespace NUMINAMATH_CALUDE_haley_origami_papers_l1601_160100

/-- The number of origami papers Haley has to give away -/
def total_papers : ℕ := 48

/-- The number of Haley's cousins -/
def num_cousins : ℕ := 6

/-- The number of papers each cousin would receive if Haley distributes all her papers equally -/
def papers_per_cousin : ℕ := 8

/-- Theorem stating that the total number of origami papers Haley has to give away is 48 -/
theorem haley_origami_papers :
  total_papers = num_cousins * papers_per_cousin :=
by sorry

end NUMINAMATH_CALUDE_haley_origami_papers_l1601_160100


namespace NUMINAMATH_CALUDE_sue_fill_time_l1601_160179

def jim_rate : ℚ := 1 / 30
def tony_rate : ℚ := 1 / 90
def combined_rate : ℚ := 1 / 15

def sue_time : ℚ := 45

theorem sue_fill_time (sue_rate : ℚ) 
  (h1 : sue_rate = 1 / sue_time)
  (h2 : jim_rate + sue_rate + tony_rate = combined_rate) : 
  sue_time = 45 := by sorry

end NUMINAMATH_CALUDE_sue_fill_time_l1601_160179
