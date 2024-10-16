import Mathlib

namespace NUMINAMATH_CALUDE_find_value_of_A_l1782_178220

theorem find_value_of_A : ∃ A : ℚ, 
  (∃ B : ℚ, B - A = 0.99 ∧ B = 10 * A) → A = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_find_value_of_A_l1782_178220


namespace NUMINAMATH_CALUDE_dans_age_proof_l1782_178274

/-- Dan's present age -/
def dans_age : ℕ := 8

/-- Theorem stating that Dan's age after 20 years will be 7 times his age 4 years ago -/
theorem dans_age_proof : dans_age + 20 = 7 * (dans_age - 4) := by
  sorry

end NUMINAMATH_CALUDE_dans_age_proof_l1782_178274


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1782_178273

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

theorem complement_of_M_in_U : Set.compl M = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1782_178273


namespace NUMINAMATH_CALUDE_entire_line_purple_exactly_integers_purple_not_exactly_rationals_purple_l1782_178283

-- Define the coloring function
def Coloring := ℝ → Bool

-- Define the property of being purple
def isPurple (c : Coloring) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ y z : ℝ, |x - y| < ε ∧ |x - z| < ε ∧ c y ≠ c z

-- Theorem for part a
theorem entire_line_purple :
  ∃ c : Coloring, ∀ x : ℝ, isPurple c x :=
sorry

-- Theorem for part b
theorem exactly_integers_purple :
  ∃ c : Coloring, ∀ x : ℝ, isPurple c x ↔ ∃ n : ℤ, x = n :=
sorry

-- Theorem for part c
theorem not_exactly_rationals_purple :
  ¬ ∃ c : Coloring, ∀ x : ℝ, isPurple c x ↔ ∃ q : ℚ, x = q :=
sorry

end NUMINAMATH_CALUDE_entire_line_purple_exactly_integers_purple_not_exactly_rationals_purple_l1782_178283


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1782_178248

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 16) :
  a 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1782_178248


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_prism_l1782_178243

/-- The surface area of a sphere circumscribing a right square prism -/
theorem sphere_surface_area_of_prism (base_edge : ℝ) (height : ℝ) 
  (h_base : base_edge = 2) (h_height : height = 3) :
  4 * π * ((base_edge^2 + base_edge^2 + height^2).sqrt / 2)^2 = 17 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_prism_l1782_178243


namespace NUMINAMATH_CALUDE_tables_in_hall_l1782_178263

theorem tables_in_hall : ℕ :=
  let total_legs : ℕ := 724
  let stools_per_table : ℕ := 8
  let stool_legs : ℕ := 4
  let table_legs : ℕ := 5

  have h : ∃ (t : ℕ), t * (stools_per_table * stool_legs + table_legs) = total_legs :=
    sorry

  have unique : ∀ (t : ℕ), t * (stools_per_table * stool_legs + table_legs) = total_legs → t = 19 :=
    sorry

  19

/- Proof omitted -/

end NUMINAMATH_CALUDE_tables_in_hall_l1782_178263


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1782_178252

theorem log_sum_equals_two (a b : ℝ) (h1 : 2^a = Real.sqrt 10) (h2 : 5^b = Real.sqrt 10) :
  1/a + 1/b = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1782_178252


namespace NUMINAMATH_CALUDE_largest_possible_median_l1782_178207

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  s.card = 5 ∧ (s.filter (λ x => x ≤ m)).card ≥ 3 ∧ (s.filter (λ x => x ≥ m)).card ≥ 3

theorem largest_possible_median :
  ∀ x y : ℤ, y = 2 * x →
  ∃ m : ℤ, is_median m {x, y, 3, 7, 9} ∧
    ∀ m' : ℤ, is_median m' {x, y, 3, 7, 9} → m' ≤ m ∧ m = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_median_l1782_178207


namespace NUMINAMATH_CALUDE_average_production_before_today_l1782_178298

theorem average_production_before_today 
  (n : ℕ) 
  (today_production : ℕ) 
  (new_average : ℕ) 
  (h1 : n = 9)
  (h2 : today_production = 90)
  (h3 : new_average = 45) :
  (n * (n + 1) * new_average - (n + 1) * today_production) / n = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_production_before_today_l1782_178298


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1782_178211

theorem quadratic_form_equivalence (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 44 = (x + m)^2 + 8) → 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1782_178211


namespace NUMINAMATH_CALUDE_apple_street_length_in_km_l1782_178266

/-- The length of Apple Street in meters -/
def apple_street_length : ℝ := 3200

/-- The distance between intersections in meters -/
def intersection_distance : ℝ := 200

/-- The number of numbered intersections -/
def numbered_intersections : ℕ := 15

/-- The total number of intersections -/
def total_intersections : ℕ := numbered_intersections + 1

theorem apple_street_length_in_km :
  apple_street_length / 1000 = 3.2 := by sorry

end NUMINAMATH_CALUDE_apple_street_length_in_km_l1782_178266


namespace NUMINAMATH_CALUDE_students_per_table_is_three_l1782_178261

/-- The number of students sitting at each table in Miss Smith's English class --/
def students_per_table : ℕ :=
  let total_students : ℕ := 47
  let num_tables : ℕ := 6
  let students_in_bathroom : ℕ := 3
  let students_in_canteen : ℕ := 3 * students_in_bathroom
  let new_students : ℕ := 2 * 4
  let foreign_exchange_students : ℕ := 3 * 3
  let absent_students : ℕ := students_in_bathroom + students_in_canteen + new_students + foreign_exchange_students
  let present_students : ℕ := total_students - absent_students
  present_students / num_tables

theorem students_per_table_is_three : students_per_table = 3 := by
  sorry

end NUMINAMATH_CALUDE_students_per_table_is_three_l1782_178261


namespace NUMINAMATH_CALUDE_unique_solution_l1782_178246

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x ^ 3 + f y ^ 3 + 3 * x * y) - 3 * x^2 * y^2 * f x

/-- The theorem stating that there is a unique function satisfying the equation -/
theorem unique_solution :
  ∃! f : ℝ → ℝ, SatisfiesEquation f ∧ ∀ x : ℝ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1782_178246


namespace NUMINAMATH_CALUDE_son_age_problem_l1782_178271

theorem son_age_problem (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_son_age_problem_l1782_178271


namespace NUMINAMATH_CALUDE_number_divided_by_three_equals_number_minus_five_l1782_178224

theorem number_divided_by_three_equals_number_minus_five : 
  ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_number_divided_by_three_equals_number_minus_five_l1782_178224


namespace NUMINAMATH_CALUDE_scientific_notation_10870_l1782_178256

theorem scientific_notation_10870 :
  10870 = 1.087 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_10870_l1782_178256


namespace NUMINAMATH_CALUDE_exactly_two_correct_propositions_l1782_178291

-- Define the concept of related curves
def related_curves (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (l : ℝ → ℝ → Prop), ∃ (x1 y1 x2 y2 : ℝ),
    C1 x1 y1 ∧ C2 x2 y2 ∧
    (∀ x y, l x y ↔ (y - y1) = (x - x1) * ((y2 - y1) / (x2 - x1))) ∧
    (∀ x y, l x y → (C1 x y ∨ C2 x y))

-- Define the curves
def C1_1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2_1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0
def C1_2 (x y : ℝ) : Prop := 4*y^2 - x^2 = 1
def C2_2 (x y : ℝ) : Prop := x^2 - 4*y^2 = 1
def C1_3 (x y : ℝ) : Prop := y = Real.log x
def C2_3 (x y : ℝ) : Prop := y = x^2 - x

-- Define the propositions
def prop1 : Prop := ∃! (l1 l2 : ℝ → ℝ → Prop), 
  related_curves C1_1 C2_1 ∧ (∀ x y, l1 x y → (C1_1 x y ∨ C2_1 x y)) ∧
  (∀ x y, l2 x y → (C1_1 x y ∨ C2_1 x y)) ∧ l1 ≠ l2

def prop2 : Prop := related_curves C1_2 C2_2

def prop3 : Prop := related_curves C1_3 C2_3

-- The theorem to prove
theorem exactly_two_correct_propositions : 
  (prop1 ∧ ¬prop2 ∧ prop3) ∨ (prop1 ∧ prop2 ∧ ¬prop3) ∨ (¬prop1 ∧ prop2 ∧ prop3) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_correct_propositions_l1782_178291


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l1782_178206

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number -/
def base7Number : List Nat := [6, 3, 4, 5, 2]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 6740 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l1782_178206


namespace NUMINAMATH_CALUDE_arts_group_size_proof_l1782_178268

/-- The number of days it takes one student to complete the project -/
def one_student_days : ℕ := 60

/-- The number of additional students who join after the first day -/
def additional_students : ℕ := 15

/-- The number of days the additional students work -/
def additional_days : ℕ := 2

/-- The total amount of work to be done, normalized to 1 -/
def total_work : ℚ := 1

/-- The number of students in the original arts group -/
def arts_group_size : ℕ := 10

theorem arts_group_size_proof :
  (arts_group_size : ℚ) / one_student_days +
  (arts_group_size + additional_students : ℚ) * additional_days / one_student_days = total_work :=
by sorry

end NUMINAMATH_CALUDE_arts_group_size_proof_l1782_178268


namespace NUMINAMATH_CALUDE_grid_property_l1782_178292

-- Define a 4x4 grid of rational numbers
def Grid := Matrix (Fin 4) (Fin 4) ℚ

-- Define what it means for a row to be an arithmetic sequence
def is_arithmetic_row (g : Grid) (i : Fin 4) : Prop :=
  ∃ a d : ℚ, ∀ j : Fin 4, g i j = a + d * j

-- Define what it means for a column to be an arithmetic sequence
def is_arithmetic_col (g : Grid) (j : Fin 4) : Prop :=
  ∃ a d : ℚ, ∀ i : Fin 4, g i j = a + d * i

-- Main theorem
theorem grid_property (g : Grid) : 
  (∀ i : Fin 4, is_arithmetic_row g i) →
  (∀ j : Fin 4, is_arithmetic_col g j) →
  g 0 0 = 3 →
  g 0 3 = 18 →
  g 3 0 = 11 →
  g 3 3 = 50 →
  g 1 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_grid_property_l1782_178292


namespace NUMINAMATH_CALUDE_multiply_squared_terms_l1782_178226

theorem multiply_squared_terms (a : ℝ) : 3 * a^2 * (2 * a^2) = 6 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_squared_terms_l1782_178226


namespace NUMINAMATH_CALUDE_cos_is_periodic_l1782_178201

-- Define the concept of a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- Define the concept of a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, f = fun x ↦ a * Real.cos (b * x) + c * Real.sin (b * x)

-- State the theorem
theorem cos_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric (fun x ↦ Real.cos x) →
  IsPeriodic (fun x ↦ Real.cos x) :=
by
  sorry

end NUMINAMATH_CALUDE_cos_is_periodic_l1782_178201


namespace NUMINAMATH_CALUDE_exists_product_sum_20000_l1782_178267

theorem exists_product_sum_20000 : 
  ∃ k m : ℕ, 1 ≤ k ∧ k < m ∧ m ≤ 999 ∧ k * (k + 1) + m * (m + 1) = 20000 := by
  sorry

end NUMINAMATH_CALUDE_exists_product_sum_20000_l1782_178267


namespace NUMINAMATH_CALUDE_second_to_first_ratio_l1782_178262

/-- Represents the amount of food eaten by each guinea pig -/
structure GuineaPigFood where
  first : ℚ
  second : ℚ
  third : ℚ

/-- Calculates the total food eaten by all guinea pigs -/
def totalFood (gpf : GuineaPigFood) : ℚ :=
  gpf.first + gpf.second + gpf.third

/-- Theorem: The ratio of food eaten by the second guinea pig to the first guinea pig is 2:1 -/
theorem second_to_first_ratio (gpf : GuineaPigFood) : 
  gpf.first = 2 → 
  gpf.third = gpf.second + 3 → 
  totalFood gpf = 13 → 
  gpf.second / gpf.first = 2 := by
sorry

end NUMINAMATH_CALUDE_second_to_first_ratio_l1782_178262


namespace NUMINAMATH_CALUDE_systems_equivalence_l1782_178259

-- Define the systems of equations
def system1 (x y a b : ℝ) : Prop :=
  2 * (x + 1) - y = 7 ∧ x + b * y = a

def system2 (x y a b : ℝ) : Prop :=
  a * x + y = b ∧ 3 * x + 2 * (y - 1) = 9

-- Theorem statement
theorem systems_equivalence :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), system1 x y a b ∧ system2 x y a b) →
  (∃! (x y : ℝ), x = 3 ∧ y = 1 ∧ system1 x y a b ∧ system2 x y a b) ∧
  (3 * a - b)^2023 = -1 :=
sorry

end NUMINAMATH_CALUDE_systems_equivalence_l1782_178259


namespace NUMINAMATH_CALUDE_correct_sunset_time_l1782_178275

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the sunset time given sunrise time and daylight duration -/
def calculateSunset (sunrise : Time) (daylight : Duration) : Time :=
  { hours := (sunrise.hours + daylight.hours + (sunrise.minutes + daylight.minutes) / 60) % 24,
    minutes := (sunrise.minutes + daylight.minutes) % 60 }

theorem correct_sunset_time :
  let sunrise : Time := { hours := 7, minutes := 12 }
  let daylight : Duration := { hours := 9, minutes := 45 }
  let calculated_sunset : Time := calculateSunset sunrise daylight
  calculated_sunset = { hours := 16, minutes := 57 } :=
by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l1782_178275


namespace NUMINAMATH_CALUDE_final_debt_calculation_l1782_178286

def calculate_debt (initial_loan : ℝ) (repayment1_percent : ℝ) (loan2 : ℝ) 
                   (repayment2_percent : ℝ) (loan3 : ℝ) (repayment3_percent : ℝ) : ℝ :=
  let debt1 := initial_loan * (1 - repayment1_percent)
  let debt2 := debt1 + loan2
  let debt3 := debt2 * (1 - repayment2_percent)
  let debt4 := debt3 + loan3
  debt4 * (1 - repayment3_percent)

theorem final_debt_calculation :
  calculate_debt 40 0.25 25 0.5 30 0.1 = 51.75 := by
  sorry

end NUMINAMATH_CALUDE_final_debt_calculation_l1782_178286


namespace NUMINAMATH_CALUDE_minimum_point_implies_b_greater_than_one_l1782_178295

theorem minimum_point_implies_b_greater_than_one (a b : ℝ) (hb : b ≠ 0) :
  let f := fun x : ℝ ↦ (x - b) * (x^2 + a*x + b)
  (∀ x, f b ≤ f x) →
  b > 1 := by
sorry

end NUMINAMATH_CALUDE_minimum_point_implies_b_greater_than_one_l1782_178295


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l1782_178289

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / y ∧ y = (3 + Real.sqrt 29) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l1782_178289


namespace NUMINAMATH_CALUDE_sufficient_condition_for_line_parallel_plane_not_necessary_condition_for_line_parallel_plane_l1782_178205

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (planeParallel : Plane → Plane → Prop)

-- Define the parallel relation for a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the subset relation for a line and a plane
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem sufficient_condition_for_line_parallel_plane 
  (α β : Plane) (m : Line) :
  (planeParallel α β ∧ lineInPlane m β) → lineParallelPlane m α :=
sorry

-- State that the condition is not necessary
theorem not_necessary_condition_for_line_parallel_plane 
  (α β : Plane) (m : Line) :
  ¬(lineParallelPlane m α → (planeParallel α β ∧ lineInPlane m β)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_line_parallel_plane_not_necessary_condition_for_line_parallel_plane_l1782_178205


namespace NUMINAMATH_CALUDE_poetry_class_attendance_l1782_178202

/-- The number of people who initially attended the poetry class. -/
def initial_attendees : ℕ := 45

/-- The number of people who arrived late to the class. -/
def late_arrivals : ℕ := 15

/-- The number of lollipops given away by the teacher. -/
def lollipops_given : ℕ := 12

/-- The ratio of attendees to lollipops. -/
def attendee_lollipop_ratio : ℕ := 5

theorem poetry_class_attendance :
  (initial_attendees + late_arrivals) / attendee_lollipop_ratio = lollipops_given :=
by sorry

end NUMINAMATH_CALUDE_poetry_class_attendance_l1782_178202


namespace NUMINAMATH_CALUDE_income_for_given_tax_l1782_178225

/-- Proves that given the tax conditions, an income of $56,000 results in a total tax of $8,000 --/
theorem income_for_given_tax : ∀ (I : ℝ),
  (min I 40000 * 0.12 + max (I - 40000) 0 * 0.20 = 8000) → I = 56000 := by
  sorry

end NUMINAMATH_CALUDE_income_for_given_tax_l1782_178225


namespace NUMINAMATH_CALUDE_fraction_comparison_l1782_178278

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  b / (a - c) < a / (b - d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1782_178278


namespace NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l1782_178282

/-- The maximum number of parts into which three planes can divide space -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts into which three planes can divide space is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l1782_178282


namespace NUMINAMATH_CALUDE_sequence_sum_l1782_178287

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = n² + 1, prove a₁ + a₉ = 19 -/
theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2 + 1) : 
    a 1 + a 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1782_178287


namespace NUMINAMATH_CALUDE_saras_savings_jar_l1782_178253

theorem saras_savings_jar (total_amount : ℕ) (total_bills : ℕ) 
  (h1 : total_amount = 84)
  (h2 : total_bills = 58) : 
  ∃ (ones twos : ℕ), 
    ones + twos = total_bills ∧ 
    ones + 2 * twos = total_amount ∧
    ones = 32 := by
  sorry

end NUMINAMATH_CALUDE_saras_savings_jar_l1782_178253


namespace NUMINAMATH_CALUDE_day_crew_load_fraction_l1782_178221

/-- 
Proves that the day crew loads 8/11 of all boxes given the conditions about night and day crews.
-/
theorem day_crew_load_fraction 
  (D : ℝ) -- Number of boxes loaded by each day crew worker
  (W : ℝ) -- Number of workers in the day crew
  (h1 : D > 0) -- Assumption that D is positive
  (h2 : W > 0) -- Assumption that W is positive
  : (D * W) / ((D * W) + ((3/4 * D) * (1/2 * W))) = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_day_crew_load_fraction_l1782_178221


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1782_178254

-- Problem 1
theorem problem_1 : -17 - (-6) + 8 - 2 = -5 := by sorry

-- Problem 2
theorem problem_2 : -1^2024 + 16 / (-2)^3 * |(-3) - 1| = -9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1782_178254


namespace NUMINAMATH_CALUDE_total_adoption_cost_l1782_178296

def cat_cost : ℕ := 50
def adult_dog_cost : ℕ := 100
def puppy_cost : ℕ := 150
def num_cats : ℕ := 2
def num_adult_dogs : ℕ := 3
def num_puppies : ℕ := 2

theorem total_adoption_cost :
  cat_cost * num_cats + adult_dog_cost * num_adult_dogs + puppy_cost * num_puppies = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_adoption_cost_l1782_178296


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l1782_178241

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  947 = 23 * q + r ∧ 
  q > 0 ∧ 
  r > 0 ∧ 
  ∀ (q' r' : ℕ), (947 = 23 * q' + r' ∧ q' > 0 ∧ r' > 0) → q' - r' ≤ q - r ∧
  q - r = 37 := by
sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l1782_178241


namespace NUMINAMATH_CALUDE_unique_point_not_on_parabola_l1782_178281

/-- A parabola passing through points (-1,0) and (2,0) -/
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the parabola passes through (-1,0) and (2,0) -/
def parabola_condition (a b c : ℝ) : Prop :=
  parabola a b c (-1) = 0 ∧ parabola a b c 2 = 0

/-- The point P(x_0 + 1, 2x_0^2 - 2) -/
def point_P (x_0 : ℝ) : ℝ × ℝ := (x_0 + 1, 2 * x_0^2 - 2)

/-- The theorem stating that P(-1, 6) is the only point satisfying the conditions -/
theorem unique_point_not_on_parabola :
  ∀ (a b c : ℝ), a ≠ 0 → parabola_condition a b c →
  (∀ (x_0 : ℝ), point_P x_0 ≠ (x_0 + 1, parabola a b c (x_0 + 1))) →
  point_P (-3) = (-1, 6) := by sorry

end NUMINAMATH_CALUDE_unique_point_not_on_parabola_l1782_178281


namespace NUMINAMATH_CALUDE_polynomial_divisibility_implies_r_values_l1782_178215

theorem polynomial_divisibility_implies_r_values : 
  ∀ (r : ℝ), (∃ (p : ℝ → ℝ), (∀ x, 10 * x^3 - 10 * x^2 - 52 * x + 60 = (x - r)^2 * p x)) → 
  (r = (2 + Real.sqrt 30) / 5 ∨ r = (2 - Real.sqrt 30) / 5) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_implies_r_values_l1782_178215


namespace NUMINAMATH_CALUDE_distance_to_line_segment_equidistant_points_vertical_line_equidistant_points_diagonal_l1782_178285

-- Define the distance function from a point to a line segment
def distance_point_to_segment (P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the line segment l: x-y-3=0 (3 ≤ x ≤ 5)
def line_segment_l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - 3 = 0 ∧ 3 ≤ p.1 ∧ p.1 ≤ 5}

-- Theorem 1
theorem distance_to_line_segment :
  distance_point_to_segment (1, 1) line_segment_l = Real.sqrt 5 := by sorry

-- Define the set of points equidistant from two line segments
def equidistant_points (l₁ l₂ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | distance_point_to_segment p l₁ = distance_point_to_segment p l₂}

-- Define line segments AB and CD for Theorem 2
def line_segment_AB : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
def line_segment_CD : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

-- Theorem 2
theorem equidistant_points_vertical_line :
  equidistant_points line_segment_AB line_segment_CD = {p : ℝ × ℝ | p.1 = 0} := by sorry

-- Define line segments AB and CD for Theorem 3
def line_segment_AB' : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0 ∧ -1 ≤ p.1 ∧ p.1 ≤ 1}
def line_segment_CD' : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0 ∧ -1 ≤ p.2 ∧ p.2 ≤ 1}

-- Theorem 3
theorem equidistant_points_diagonal :
  equidistant_points line_segment_AB' line_segment_CD' = {p : ℝ × ℝ | p.1^2 - p.2^2 = 0} := by sorry

end NUMINAMATH_CALUDE_distance_to_line_segment_equidistant_points_vertical_line_equidistant_points_diagonal_l1782_178285


namespace NUMINAMATH_CALUDE_complex_power_equality_smallest_power_is_minimal_l1782_178251

/-- The smallest positive integer n for which (a+bi)^(n+1) = (a-bi)^(n+1) holds for some positive real a and b -/
def smallest_power : ℕ := 3

theorem complex_power_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.mk a b)^4 = (Complex.mk a (-b))^4 → b / a = 1 :=
by sorry

theorem smallest_power_is_minimal (n : ℕ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  n < smallest_power →
  (Complex.mk a b)^(n + 1) ≠ (Complex.mk a (-b))^(n + 1) :=
by sorry

#check smallest_power
#check complex_power_equality
#check smallest_power_is_minimal

end NUMINAMATH_CALUDE_complex_power_equality_smallest_power_is_minimal_l1782_178251


namespace NUMINAMATH_CALUDE_quarter_power_equality_l1782_178242

theorem quarter_power_equality (x : ℝ) : (1 / 4 : ℝ) ^ x = 0.25 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quarter_power_equality_l1782_178242


namespace NUMINAMATH_CALUDE_problem_solution_l1782_178239

theorem problem_solution (x y z : ℝ) :
  (1.5 * x = 0.3 * y) →
  (x = 20) →
  (0.6 * y = z) →
  z = 60 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1782_178239


namespace NUMINAMATH_CALUDE_perfect_square_identification_l1782_178257

theorem perfect_square_identification :
  ¬ ∃ (x : ℕ), 7^2051 = x^2 ∧
  ∃ (a b c d : ℕ), 6^2048 = a^2 ∧ 8^2050 = b^2 ∧ 9^2052 = c^2 ∧ 10^2040 = d^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_identification_l1782_178257


namespace NUMINAMATH_CALUDE_land_development_profit_l1782_178236

theorem land_development_profit (cost_per_acre : ℝ) (sale_price_per_acre : ℝ) (profit : ℝ) (acres : ℝ) : 
  cost_per_acre = 70 →
  sale_price_per_acre = 200 →
  profit = 6000 →
  sale_price_per_acre * (acres / 2) - cost_per_acre * acres = profit →
  acres = 200 := by
sorry

end NUMINAMATH_CALUDE_land_development_profit_l1782_178236


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l1782_178284

theorem third_root_of_cubic (c d : ℚ) :
  (∀ x : ℚ, c * x^3 + (c + 3*d) * x^2 + (2*d - 4*c) * x + (10 - c) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 76/11) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l1782_178284


namespace NUMINAMATH_CALUDE_ram_work_time_l1782_178279

/-- Ram's efficiency compared to Krish's -/
def ram_efficiency : ℚ := 1/2

/-- Time taken by Ram and Krish working together (in days) -/
def combined_time : ℕ := 7

/-- Time taken by Ram working alone (in days) -/
def ram_alone_time : ℕ := 21

theorem ram_work_time :
  ram_efficiency * combined_time * 2 = ram_alone_time := by
  sorry

end NUMINAMATH_CALUDE_ram_work_time_l1782_178279


namespace NUMINAMATH_CALUDE_first_quadrant_half_angle_l1782_178280

theorem first_quadrant_half_angle (α : Real) : 0 < α ∧ α < π / 2 → 0 < α / 2 ∧ α / 2 < π / 4 := by
  sorry

end NUMINAMATH_CALUDE_first_quadrant_half_angle_l1782_178280


namespace NUMINAMATH_CALUDE_average_weight_increase_l1782_178222

/-- Proves that replacing a person weighing 76 kg with a person weighing 119.4 kg
    in a group of 7 people increases the average weight by 6.2 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 7 * initial_average
  let new_total := initial_total - 76 + 119.4
  let new_average := new_total / 7
  new_average - initial_average = 6.2 := by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l1782_178222


namespace NUMINAMATH_CALUDE_max_curved_sides_l1782_178227

/-- A figure formed by the intersection of circles -/
structure IntersectionFigure where
  n : ℕ
  n_ge_two : n ≥ 2

/-- The number of curved sides in an intersection figure -/
def curved_sides (F : IntersectionFigure) : ℕ := 2 * F.n - 2

/-- The theorem stating the maximum number of curved sides -/
theorem max_curved_sides (F : IntersectionFigure) :
  curved_sides F ≤ 2 * F.n - 2 :=
sorry

end NUMINAMATH_CALUDE_max_curved_sides_l1782_178227


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1782_178219

theorem sin_cos_identity : 
  (4 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180)) / Real.cos (20 * π / 180) - Real.tan (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1782_178219


namespace NUMINAMATH_CALUDE_range_of_x_l1782_178238

theorem range_of_x (a b c x : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) 
  (h2 : a + 2*b + 3*c > |x + 1|) : -7 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1782_178238


namespace NUMINAMATH_CALUDE_prob_two_even_in_six_dice_l1782_178260

/-- A fair 10-sided die with faces numbered from 1 to 10 -/
def TenSidedDie : Type := Fin 10

/-- The probability of rolling an even number on a 10-sided die -/
def probEven : ℚ := 1/2

/-- The probability of rolling an odd number on a 10-sided die -/
def probOdd : ℚ := 1/2

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The number of dice that should show an even number -/
def numEven : ℕ := 2

/-- The probability of rolling exactly two even numbers when rolling six fair 10-sided dice -/
theorem prob_two_even_in_six_dice : 
  (numDice.choose numEven : ℚ) * probEven ^ numEven * probOdd ^ (numDice - numEven) = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_even_in_six_dice_l1782_178260


namespace NUMINAMATH_CALUDE_equal_balls_probability_l1782_178276

/-- Represents the urn state -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a single draw operation -/
inductive Draw
  | Red
  | Blue

/-- Performs a single draw operation on the urn state -/
def drawOperation (state : UrnState) (draw : Draw) : UrnState :=
  match draw with
  | Draw.Red => UrnState.mk (state.red + 1) state.blue
  | Draw.Blue => UrnState.mk state.red (state.blue + 1)

/-- Performs a sequence of draw operations on the urn state -/
def performOperations (initial : UrnState) (draws : List Draw) : UrnState :=
  draws.foldl drawOperation initial

/-- Calculates the probability of a specific sequence of draws -/
def sequenceProbability (draws : List Draw) : ℚ :=
  sorry

/-- Calculates the number of valid sequences that result in 4 red and 4 blue balls -/
def validSequencesCount : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem equal_balls_probability :
  let initialState := UrnState.mk 2 1
  let finalState := UrnState.mk 4 4
  (validSequencesCount * sequenceProbability (List.replicate 5 Draw.Red)) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_equal_balls_probability_l1782_178276


namespace NUMINAMATH_CALUDE_sum_of_squares_divisibility_l1782_178290

theorem sum_of_squares_divisibility (a b c : ℤ) :
  9 ∣ (a^2 + b^2 + c^2) → 
  (9 ∣ (a^2 - b^2)) ∨ (9 ∣ (a^2 - c^2)) ∨ (9 ∣ (b^2 - c^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisibility_l1782_178290


namespace NUMINAMATH_CALUDE_angle_positions_l1782_178228

-- Define the quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define the position of an angle
inductive AnglePosition
  | InQuadrant (q : Quadrant)
  | OnPositiveYAxis

-- Function to determine the position of 2θ
def doubleThetaPosition (θ : Real) : AnglePosition := sorry

-- Function to determine the position of θ/2
def halfThetaPosition (θ : Real) : Quadrant := sorry

-- Theorem statement
theorem angle_positions (θ : Real) 
  (h : ∃ (k : ℤ), 180 + k * 360 < θ ∧ θ < 270 + k * 360) : 
  (doubleThetaPosition θ = AnglePosition.InQuadrant Quadrant.First ∨
   doubleThetaPosition θ = AnglePosition.InQuadrant Quadrant.Second ∨
   doubleThetaPosition θ = AnglePosition.OnPositiveYAxis) ∧
  (halfThetaPosition θ = Quadrant.Second ∨
   halfThetaPosition θ = Quadrant.Fourth) := by
  sorry

end NUMINAMATH_CALUDE_angle_positions_l1782_178228


namespace NUMINAMATH_CALUDE_mean_temperature_l1782_178294

def temperatures : List ℝ := [75, 78, 80, 76, 77]

theorem mean_temperature : (temperatures.sum / temperatures.length : ℝ) = 77.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1782_178294


namespace NUMINAMATH_CALUDE_binomial_prob_properties_l1782_178209

/-- A binomial distribution with parameters n and p -/
structure BinomialDist where
  n : ℕ+
  p : ℝ
  h_p_pos : 0 < p
  h_p_lt_one : p < 1

/-- The probability that X is odd in a binomial distribution -/
noncomputable def prob_odd (b : BinomialDist) : ℝ :=
  (1 - (1 - 2*b.p)^b.n.val) / 2

/-- The probability that X is even in a binomial distribution -/
noncomputable def prob_even (b : BinomialDist) : ℝ :=
  1 - prob_odd b

theorem binomial_prob_properties (b : BinomialDist) :
  (prob_odd b + prob_even b = 1) ∧
  (b.p = 1/2 → prob_odd b = prob_even b) ∧
  (0 < b.p ∧ b.p < 1/2 → ∀ m : ℕ+, m < b.n → prob_odd ⟨m, b.p, b.h_p_pos, b.h_p_lt_one⟩ < prob_odd b) :=
by sorry

end NUMINAMATH_CALUDE_binomial_prob_properties_l1782_178209


namespace NUMINAMATH_CALUDE_choose_four_diff_suits_standard_deck_l1782_178247

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h_total := rfl }

/-- The number of ways to choose 4 cards of different suits from a standard deck -/
def choose_four_diff_suits (d : Deck) : Nat :=
  d.cards_per_suit ^ d.num_suits

/-- Theorem stating the number of ways to choose 4 cards of different suits from a standard deck -/
theorem choose_four_diff_suits_standard_deck :
  choose_four_diff_suits standard_deck = 28561 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_diff_suits_standard_deck_l1782_178247


namespace NUMINAMATH_CALUDE_circle_equation_l1782_178217

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def satisfiesConditions (c : Circle) : Prop :=
  let (a, b) := c.center
  -- Condition 1: chord on y-axis has length 2
  c.radius^2 = a^2 + 1 ∧
  -- Condition 2: ratio of arc lengths divided by x-axis is 3:1
  (c.radius^2 = 2 * b^2) ∧
  -- Condition 3: distance from center to line x - 2y = 0 is √5/5
  |a - 2*b| / Real.sqrt 5 = Real.sqrt 5 / 5

-- Theorem statement
theorem circle_equation (c : Circle) :
  satisfiesConditions c →
  ((∃ x y, (x + 1)^2 + (y + 1)^2 = 2) ∨ (∃ x y, (x - 1)^2 + (y - 1)^2 = 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1782_178217


namespace NUMINAMATH_CALUDE_cube_root_of_64_equals_2_to_m_l1782_178269

theorem cube_root_of_64_equals_2_to_m (m : ℝ) : (64 : ℝ)^(1/3) = 2^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_64_equals_2_to_m_l1782_178269


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1782_178245

-- Define the hyperbola
def Hyperbola (center focus vertex : ℝ × ℝ) : Prop :=
  let (h, k) := center
  let (_, f_y) := focus
  let (_, v_y) := vertex
  let a : ℝ := |k - v_y|
  let c : ℝ := |f_y - k|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  ∀ x y : ℝ, ((y - k)^2 / a^2) - ((x - h)^2 / b^2) = 1

-- State the theorem
theorem hyperbola_sum (center focus vertex : ℝ × ℝ) 
  (h : Hyperbola center focus vertex) 
  (hc : center = (3, 1)) 
  (hf : focus = (3, 9)) 
  (hv : vertex = (3, -2)) : 
  let (h, k) := center
  let (_, f_y) := focus
  let (_, v_y) := vertex
  let a : ℝ := |k - v_y|
  let c : ℝ := |f_y - k|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  h + k + a + b = 7 + Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1782_178245


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1782_178234

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expr := 2*(x^2 - 2*x^3 + x) + 4*(x + 3*x^3 - 2*x^2 + 2*x^5 + x^3) - 6*(2 + x - 5*x^3 - x^2)
  ∃ (a b c d e : ℝ), expr = a*x^5 + b*x^4 + 42*x^3 + c*x^2 + d*x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1782_178234


namespace NUMINAMATH_CALUDE_sequence_property_main_theorem_l1782_178299

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then 1 else (1 / 3) * (4 / 3) ^ (n - 2)

def sequence_S (n : ℕ) : ℚ := (4 / 3) ^ (n - 1)

theorem sequence_property : ∀ n : ℕ, n ≥ 1 → 3 * sequence_a (n + 1) = sequence_S n :=
  sorry

theorem main_theorem : ∀ n : ℕ, n ≥ 1 → 
  sequence_a n = if n = 1 then 1 else (1 / 3) * (4 / 3) ^ (n - 2) :=
  sorry

end NUMINAMATH_CALUDE_sequence_property_main_theorem_l1782_178299


namespace NUMINAMATH_CALUDE_distribute_three_books_twelve_students_l1782_178232

/-- The number of ways to distribute n identical objects among k people,
    where no person can receive more than one object. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose k n

theorem distribute_three_books_twelve_students :
  distribute 3 12 = 220 := by
  sorry

end NUMINAMATH_CALUDE_distribute_three_books_twelve_students_l1782_178232


namespace NUMINAMATH_CALUDE_f_monotonic_decreasing_on_interval_l1782_178264

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_monotonic_decreasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_decreasing_on_interval_l1782_178264


namespace NUMINAMATH_CALUDE_decimal_difference_value_l1782_178250

/-- The value of the repeating decimal 0.727272... -/
def repeating_decimal : ℚ := 8 / 11

/-- The value of the terminating decimal 0.72 -/
def terminating_decimal : ℚ := 72 / 100

/-- The difference between the repeating decimal 0.727272... and the terminating decimal 0.72 -/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_value : decimal_difference = 8 / 1100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_value_l1782_178250


namespace NUMINAMATH_CALUDE_state_fraction_l1782_178249

theorem state_fraction (total_states : ℕ) (period_states : ℕ) 
  (h1 : total_states = 22) (h2 : period_states = 12) : 
  (period_states : ℚ) / total_states = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_state_fraction_l1782_178249


namespace NUMINAMATH_CALUDE_ivans_chess_claim_impossible_l1782_178214

theorem ivans_chess_claim_impossible : ¬ ∃ (n : ℕ), n > 0 ∧ n + 3*n + 6*n = 64 := by
  sorry

end NUMINAMATH_CALUDE_ivans_chess_claim_impossible_l1782_178214


namespace NUMINAMATH_CALUDE_new_average_after_doubling_l1782_178203

/-- Theorem: New average after doubling marks -/
theorem new_average_after_doubling (n : ℕ) (original_average : ℝ) :
  n > 0 →
  let total_marks := n * original_average
  let doubled_marks := 2 * total_marks
  let new_average := doubled_marks / n
  new_average = 2 * original_average := by
  sorry

/-- Given problem as an example -/
example : 
  let n : ℕ := 25
  let original_average : ℝ := 70
  let total_marks := n * original_average
  let doubled_marks := 2 * total_marks
  let new_average := doubled_marks / n
  new_average = 140 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_doubling_l1782_178203


namespace NUMINAMATH_CALUDE_initial_roses_count_l1782_178231

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 3

/-- The number of roses after adding more flowers -/
def final_roses : ℕ := 12

/-- The number of orchids after adding more flowers -/
def final_orchids : ℕ := 2

/-- The difference between the number of roses and orchids after adding flowers -/
def rose_orchid_difference : ℕ := 10

theorem initial_roses_count : initial_roses = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_count_l1782_178231


namespace NUMINAMATH_CALUDE_ratio_to_percentage_difference_l1782_178208

theorem ratio_to_percentage_difference (A B : ℝ) (h : A / B = 3 / 4) :
  (B - A) / B = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percentage_difference_l1782_178208


namespace NUMINAMATH_CALUDE_decision_block_two_exits_other_blocks_not_two_exits_l1782_178240

/-- Enumeration of program block types -/
inductive ProgramBlock
  | Output
  | Processing
  | Decision
  | StartEnd

/-- Function to determine the number of exits for each program block -/
def num_exits (block : ProgramBlock) : Nat :=
  match block with
  | ProgramBlock.Output => 1
  | ProgramBlock.Processing => 1
  | ProgramBlock.Decision => 2
  | ProgramBlock.StartEnd => 0

/-- Theorem stating that only the Decision block has two exits -/
theorem decision_block_two_exits :
  ∀ (block : ProgramBlock), num_exits block = 2 ↔ block = ProgramBlock.Decision :=
by sorry

/-- Corollary: No other block type has two exits -/
theorem other_blocks_not_two_exits :
  ∀ (block : ProgramBlock), block ≠ ProgramBlock.Decision → num_exits block ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_decision_block_two_exits_other_blocks_not_two_exits_l1782_178240


namespace NUMINAMATH_CALUDE_two_digit_multiples_of_6_and_9_l1782_178237

theorem two_digit_multiples_of_6_and_9 : 
  (Finset.filter (fun n => n % 6 = 0 ∧ n % 9 = 0) (Finset.range 90 \ Finset.range 10)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_multiples_of_6_and_9_l1782_178237


namespace NUMINAMATH_CALUDE_residue_of_negative_1237_mod_29_l1782_178213

theorem residue_of_negative_1237_mod_29 : Int.mod (-1237) 29 = 10 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_1237_mod_29_l1782_178213


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l1782_178229

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation for a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "subset of" relation for a line and a plane
variable (subset_of : Line → Plane → Prop)

variable (α β : Plane)
variable (m : Line)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary
  (h1 : α ≠ β)
  (h2 : subset_of m α) :
  (parallel_planes α β → parallel_line_plane m β) ∧
  ¬(parallel_line_plane m β → parallel_planes α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l1782_178229


namespace NUMINAMATH_CALUDE_car_owners_without_motorcycle_l1782_178258

theorem car_owners_without_motorcycle (total : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ)
  (h1 : total = 351)
  (h2 : car_owners = 331)
  (h3 : motorcycle_owners = 45)
  (h4 : car_owners + motorcycle_owners - total ≥ 0) :
  car_owners - (car_owners + motorcycle_owners - total) = 306 := by
  sorry

end NUMINAMATH_CALUDE_car_owners_without_motorcycle_l1782_178258


namespace NUMINAMATH_CALUDE_find_m_l1782_178288

theorem find_m : ∃ m : ℝ, ∀ x y : ℝ, (2*x + y)*(x - 2*y) = 2*x^2 - m*x*y - 2*y^2 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1782_178288


namespace NUMINAMATH_CALUDE_mashed_potatoes_bacon_difference_l1782_178200

/-- The number of students who suggested adding bacon -/
def bacon_students : ℕ := 269

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes_students : ℕ := 330

/-- The number of students who suggested adding tomatoes -/
def tomatoes_students : ℕ := 76

/-- The theorem stating the difference between the number of students who suggested
    mashed potatoes and those who suggested bacon -/
theorem mashed_potatoes_bacon_difference :
  mashed_potatoes_students - bacon_students = 61 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_bacon_difference_l1782_178200


namespace NUMINAMATH_CALUDE_same_color_probability_l1782_178244

def totalBalls : ℕ := 20
def greenBalls : ℕ := 8
def redBalls : ℕ := 5
def blueBalls : ℕ := 7

theorem same_color_probability : 
  (greenBalls : ℚ) ^ 2 / totalBalls ^ 2 + 
  (redBalls : ℚ) ^ 2 / totalBalls ^ 2 + 
  (blueBalls : ℚ) ^ 2 / totalBalls ^ 2 = 345 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1782_178244


namespace NUMINAMATH_CALUDE_k_range_l1782_178270

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := 3 / (x + 1) < 1

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (k : ℝ) : Prop :=
  (∀ x, p x k → q x) ∧ ¬(∀ x, q x → p x k)

-- Theorem statement
theorem k_range :
  ∀ k : ℝ, sufficient_not_necessary k ↔ k > 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_l1782_178270


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l1782_178218

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 
  log_base 3 (x - 1) + log_base (Real.sqrt 3) (x^2 - 1) + log_base (1/3) (x - 1) = 3 ∧
  x = Real.sqrt (1 + 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l1782_178218


namespace NUMINAMATH_CALUDE_cindys_calculation_l1782_178272

theorem cindys_calculation (x : ℝ) (h : (x - 5) / 7 = 15) : (x - 7) / 5 = 20.6 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l1782_178272


namespace NUMINAMATH_CALUDE_john_memory_card_cost_l1782_178265

/-- Calculates the amount spent on memory cards given the following conditions:
  * Pictures taken per day
  * Number of years
  * Images per memory card
  * Cost per memory card
-/
def memory_card_cost (pictures_per_day : ℕ) (years : ℕ) (images_per_card : ℕ) (cost_per_card : ℕ) : ℕ :=
  let total_pictures := pictures_per_day * years * 365
  let cards_needed := (total_pictures + images_per_card - 1) / images_per_card
  cards_needed * cost_per_card

/-- Theorem stating that under the given conditions, John spends $13140 on memory cards -/
theorem john_memory_card_cost :
  memory_card_cost 10 3 50 60 = 13140 := by
  sorry


end NUMINAMATH_CALUDE_john_memory_card_cost_l1782_178265


namespace NUMINAMATH_CALUDE_truck_loading_time_l1782_178255

theorem truck_loading_time (worker1_time worker2_time combined_time : ℝ) 
  (h1 : worker1_time = 6)
  (h2 : combined_time = 2.4)
  (h3 : 1 / worker1_time + 1 / worker2_time = 1 / combined_time) :
  worker2_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_truck_loading_time_l1782_178255


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l1782_178216

-- Define the value of a trillion
def trillion : ℝ := 10^12

-- Define the GDP value in trillions
def gdp_trillions : ℝ := 121

-- Theorem statement
theorem gdp_scientific_notation :
  gdp_trillions * trillion = 1.21 * 10^14 := by
  sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l1782_178216


namespace NUMINAMATH_CALUDE_sum_of_S_and_T_is_five_l1782_178293

theorem sum_of_S_and_T_is_five : 
  ∀ (S T : ℝ),
  let line_length : ℝ := 5
  let num_parts : ℕ := 20
  let part_length : ℝ := line_length / num_parts
  S = 5 * part_length →
  T = line_length - 5 * part_length →
  S + T = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_S_and_T_is_five_l1782_178293


namespace NUMINAMATH_CALUDE_apples_in_baskets_l1782_178223

theorem apples_in_baskets (num_baskets : ℕ) (total_apples : ℕ) (apples_per_basket : ℕ) :
  num_baskets = 37 →
  total_apples = 629 →
  num_baskets * apples_per_basket = total_apples →
  apples_per_basket = 17 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_baskets_l1782_178223


namespace NUMINAMATH_CALUDE_count_valid_triples_l1782_178210

def validTriple (x y z : ℕ+) : Prop :=
  Nat.lcm x.val y.val = 120 ∧ 
  Nat.lcm x.val z.val = 450 ∧ 
  Nat.lcm y.val z.val = 180

theorem count_valid_triples : 
  ∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    (∀ t ∈ s, validTriple t.1 t.2.1 t.2.2) ∧ 
    s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_count_valid_triples_l1782_178210


namespace NUMINAMATH_CALUDE_viju_aju_age_ratio_l1782_178235

/-- Given that Viju's age 5 years ago was 16 and that four years from now, 
    the ratio of ages of Viju to Aju will be 5:2, 
    prove that the present age ratio of Viju to Aju is 7:2. -/
theorem viju_aju_age_ratio :
  ∀ (viju_age aju_age : ℕ),
    viju_age - 5 = 16 →
    (viju_age + 4) * 2 = (aju_age + 4) * 5 →
    ∃ (k : ℕ), k > 0 ∧ viju_age = 7 * k ∧ aju_age = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_viju_aju_age_ratio_l1782_178235


namespace NUMINAMATH_CALUDE_rainbow_preschool_students_l1782_178230

theorem rainbow_preschool_students (half_day_percent : ℝ) (full_day_count : ℕ) : 
  half_day_percent = 0.25 →
  full_day_count = 60 →
  ∃ total_students : ℕ, 
    (1 - half_day_percent) * (total_students : ℝ) = full_day_count ∧
    total_students = 80 :=
by sorry

end NUMINAMATH_CALUDE_rainbow_preschool_students_l1782_178230


namespace NUMINAMATH_CALUDE_problem_solution_l1782_178204

theorem problem_solution (a b : ℤ) (ha : a = 4) (hb : b = -1) : 
  -a^2 - b^2 + a*b = -21 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1782_178204


namespace NUMINAMATH_CALUDE_point_B_coordinates_l1782_178212

def point_A : ℝ × ℝ := (-1, 5)
def vector_a : ℝ × ℝ := (2, 3)

theorem point_B_coordinates :
  ∀ (B : ℝ × ℝ),
  (B.1 - point_A.1, B.2 - point_A.2) = (3 * vector_a.1, 3 * vector_a.2) →
  B = (5, 14) := by
sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l1782_178212


namespace NUMINAMATH_CALUDE_max_notebooks_inequality_l1782_178277

/-- Represents the budget in dollars -/
def budget : ℝ := 500

/-- Represents the regular price per notebook in dollars -/
def regularPrice : ℝ := 10

/-- Represents the discount rate as a decimal -/
def discountRate : ℝ := 0.2

/-- Represents the threshold number of notebooks for the discount to apply -/
def discountThreshold : ℕ := 15

/-- Theorem stating that the maximum number of notebooks that can be purchased
    is represented by the inequality 10 × 0.8x ≤ 500 -/
theorem max_notebooks_inequality :
  ∀ x : ℝ, x > discountThreshold →
    (x = budget / (regularPrice * (1 - discountRate))) ↔ 
    (regularPrice * (1 - discountRate) * x ≤ budget) :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_inequality_l1782_178277


namespace NUMINAMATH_CALUDE_age_sum_proof_l1782_178297

/-- Given that Ashley's age is 8 and the ratio of Ashley's age to Mary's age is 4:7,
    prove that the sum of their ages is 22. -/
theorem age_sum_proof (ashley_age mary_age : ℕ) : 
  ashley_age = 8 → 
  ashley_age * 7 = mary_age * 4 → 
  ashley_age + mary_age = 22 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l1782_178297


namespace NUMINAMATH_CALUDE_abs_equation_solution_l1782_178233

theorem abs_equation_solution : 
  {x : ℝ | |2005 * x - 2005| = 2005} = {0, 2} := by
sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l1782_178233
