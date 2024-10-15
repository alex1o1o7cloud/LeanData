import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_ratio_l3553_355357

theorem sum_of_squares_ratio (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 3) 
  (h2 : a/x + b/y + c/z = -3) : 
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_ratio_l3553_355357


namespace NUMINAMATH_CALUDE_problem_solution_l3553_355343

theorem problem_solution : (2023^2 - 2023 - 1) / 2023 = 2022 - 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3553_355343


namespace NUMINAMATH_CALUDE_smoking_health_correlation_l3553_355320

-- Define smoking and health as variables
variable (smoking health : ℝ)

-- Define the concept of "harmful to health"
def is_harmful_to_health (x y : ℝ) : Prop := 
  ∀ δ > 0, ∃ ε > 0, ∀ x' y', |x' - x| < ε → |y' - y| < δ → y' < y

-- Define negative correlation
def negative_correlation (x y : ℝ) : Prop :=
  ∀ δ > 0, ∃ ε > 0, ∀ x₁ x₂ y₁ y₂, 
    |x₁ - x| < ε → |x₂ - x| < ε → |y₁ - y| < δ → |y₂ - y| < δ →
    (x₁ < x₂ → y₁ > y₂) ∧ (x₁ > x₂ → y₁ < y₂)

-- Theorem statement
theorem smoking_health_correlation 
  (h : is_harmful_to_health smoking health) : 
  negative_correlation smoking health :=
sorry

end NUMINAMATH_CALUDE_smoking_health_correlation_l3553_355320


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3553_355302

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 2| + |x - 2| ≤ 4} = Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3553_355302


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3553_355392

theorem trigonometric_identity (α : ℝ) :
  Real.cos (3 / 2 * Real.pi + 4 * α) + Real.sin (3 * Real.pi - 8 * α) - Real.sin (4 * Real.pi - 12 * α) =
  4 * Real.cos (2 * α) * Real.cos (4 * α) * Real.sin (6 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3553_355392


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3553_355362

/-- A quadrilateral with vertices at (3,-1), (-1,4), (2,3), and (9,9) -/
def Quadrilateral : List (ℝ × ℝ) := [(3, -1), (-1, 4), (2, 3), (9, 9)]

/-- One side of the quadrilateral is horizontal -/
axiom horizontal_side : ∃ (a b : ℝ) (y : ℝ), ((a, y) ∈ Quadrilateral ∧ (b, y) ∈ Quadrilateral) ∧ a ≠ b

/-- The area of the quadrilateral -/
def area : ℝ := 22.5

/-- Theorem: The area of the quadrilateral is 22.5 -/
theorem quadrilateral_area : 
  let vertices := Quadrilateral
  area = (1/2) * abs (
    (vertices[0].1 * vertices[1].2 + vertices[1].1 * vertices[2].2 + 
     vertices[2].1 * vertices[3].2 + vertices[3].1 * vertices[0].2) - 
    (vertices[1].1 * vertices[0].2 + vertices[2].1 * vertices[1].2 + 
     vertices[3].1 * vertices[2].2 + vertices[0].1 * vertices[3].2)
  ) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3553_355362


namespace NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_l3553_355373

theorem angle_with_complement_40percent_of_supplement (x : ℝ) : 
  (90 - x = (2/5) * (180 - x)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_l3553_355373


namespace NUMINAMATH_CALUDE_polynomial_negative_values_l3553_355309

theorem polynomial_negative_values (a x : ℝ) (h : 0 < x ∧ x < a) : 
  (a - x)^6 - 3*a*(a - x)^5 + 5/2*a^2*(a - x)^4 - 1/2*a^4*(a - x)^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_negative_values_l3553_355309


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l3553_355336

theorem smallest_sum_proof : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/6, 1/3 + 1/9]
  (∀ x ∈ sums, 1/3 + 1/9 ≤ x) ∧ (1/3 + 1/9 = 4/9) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l3553_355336


namespace NUMINAMATH_CALUDE_candy_count_difference_l3553_355326

/-- The number of candies Bryan has -/
def bryan_candies : ℕ := 50

/-- The number of candies Ben has -/
def ben_candies : ℕ := 20

/-- The difference in candy count between Bryan and Ben -/
def candy_difference : ℕ := bryan_candies - ben_candies

theorem candy_count_difference :
  candy_difference = 30 :=
by sorry

end NUMINAMATH_CALUDE_candy_count_difference_l3553_355326


namespace NUMINAMATH_CALUDE_cups_filled_l3553_355371

-- Define the volume of water in milliliters
def water_volume : ℕ := 1000

-- Define the cup size in milliliters
def cup_size : ℕ := 200

-- Theorem to prove
theorem cups_filled (water_volume : ℕ) (cup_size : ℕ) :
  water_volume = 1000 → cup_size = 200 → water_volume / cup_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_cups_filled_l3553_355371


namespace NUMINAMATH_CALUDE_solution_set_implies_k_value_l3553_355352

theorem solution_set_implies_k_value (k : ℝ) : 
  (∀ x : ℝ, |k * x - 4| ≤ 2 ↔ 1 ≤ x ∧ x ≤ 3) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_k_value_l3553_355352


namespace NUMINAMATH_CALUDE_equation_solution_l3553_355341

theorem equation_solution : ∃ x : ℝ, 13 + Real.sqrt (x + 5 * 3 - 3 * 3) = 14 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3553_355341


namespace NUMINAMATH_CALUDE_relationship_l3553_355375

-- Define the real numbers a, b, and c
variable (a b c : ℝ)

-- Define the conditions
axiom eq_a : 2 * a^3 + a = 2
axiom eq_b : b * Real.log b / Real.log 2 = 1
axiom eq_c : c * Real.log c / Real.log 5 = 1

-- State the theorem to be proved
theorem relationship : c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_l3553_355375


namespace NUMINAMATH_CALUDE_complex_number_properties_l3553_355305

theorem complex_number_properties : ∃ (z : ℂ), 
  z = 2 / (Complex.I - 1) ∧ 
  z^2 = 2 * Complex.I ∧ 
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3553_355305


namespace NUMINAMATH_CALUDE_field_trip_girls_l3553_355306

theorem field_trip_girls (num_vans : ℕ) (students_per_van : ℕ) (num_boys : ℕ) : 
  num_vans = 5 → 
  students_per_van = 28 → 
  num_boys = 60 → 
  num_vans * students_per_van - num_boys = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_field_trip_girls_l3553_355306


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3553_355329

def total_balls : ℕ := 6 + 5 + 2

def red_balls : ℕ := 6

theorem probability_two_red_balls :
  let prob_first_red : ℚ := red_balls / total_balls
  let prob_second_red : ℚ := (red_balls - 1) / (total_balls - 1)
  prob_first_red * prob_second_red = 5 / 26 := by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3553_355329


namespace NUMINAMATH_CALUDE_square_sum_equation_l3553_355330

theorem square_sum_equation (x y : ℝ) : 
  (x^2 + y^2)^2 = x^2 + y^2 + 12 → x^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equation_l3553_355330


namespace NUMINAMATH_CALUDE_unique_number_with_property_l3553_355377

/-- A four-digit natural number -/
def FourDigitNumber (x y z w : ℕ) : ℕ := 1000 * x + 100 * y + 10 * z + w

/-- The property that the sum of the number and its digits equals 2003 -/
def HasProperty (x y z w : ℕ) : Prop :=
  FourDigitNumber x y z w + x + y + z + w = 2003

/-- The theorem stating that 1978 is the only four-digit number satisfying the property -/
theorem unique_number_with_property :
  (∃! n : ℕ, ∃ x y z w : ℕ, 
    x ≠ 0 ∧ 
    n = FourDigitNumber x y z w ∧ 
    HasProperty x y z w) ∧
  (∃ x y z w : ℕ, 
    x ≠ 0 ∧ 
    1978 = FourDigitNumber x y z w ∧ 
    HasProperty x y z w) :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_property_l3553_355377


namespace NUMINAMATH_CALUDE_symmetry_axes_count_other_rotation_axes_count_l3553_355349

/-- Enumeration of regular polyhedra -/
inductive RegularPolyhedron
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

/-- Function to calculate the number of symmetry axes for a regular polyhedron -/
def symmetryAxes (p : RegularPolyhedron) : Nat :=
  match p with
  | RegularPolyhedron.Tetrahedron => 3
  | RegularPolyhedron.Cube => 9
  | RegularPolyhedron.Octahedron => 9
  | RegularPolyhedron.Dodecahedron => 16
  | RegularPolyhedron.Icosahedron => 16

/-- Function to calculate the number of other rotation axes for a regular polyhedron -/
def otherRotationAxes (p : RegularPolyhedron) : Nat :=
  match p with
  | RegularPolyhedron.Tetrahedron => 4
  | RegularPolyhedron.Cube => 10
  | RegularPolyhedron.Octahedron => 10
  | RegularPolyhedron.Dodecahedron => 16
  | RegularPolyhedron.Icosahedron => 16

/-- Theorem stating the number of symmetry axes for each regular polyhedron -/
theorem symmetry_axes_count :
  (∀ p : RegularPolyhedron, symmetryAxes p = 
    match p with
    | RegularPolyhedron.Tetrahedron => 3
    | RegularPolyhedron.Cube => 9
    | RegularPolyhedron.Octahedron => 9
    | RegularPolyhedron.Dodecahedron => 16
    | RegularPolyhedron.Icosahedron => 16) :=
by sorry

/-- Theorem stating the number of other rotation axes for each regular polyhedron -/
theorem other_rotation_axes_count :
  (∀ p : RegularPolyhedron, otherRotationAxes p = 
    match p with
    | RegularPolyhedron.Tetrahedron => 4
    | RegularPolyhedron.Cube => 10
    | RegularPolyhedron.Octahedron => 10
    | RegularPolyhedron.Dodecahedron => 16
    | RegularPolyhedron.Icosahedron => 16) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axes_count_other_rotation_axes_count_l3553_355349


namespace NUMINAMATH_CALUDE_johnson_prescription_l3553_355301

/-- Represents a prescription with a fixed daily dose -/
structure Prescription where
  totalDays : ℕ
  remainingPills : ℕ
  daysElapsed : ℕ
  dailyDose : ℕ

/-- Calculates the daily dose given a prescription -/
def calculateDailyDose (p : Prescription) : ℕ :=
  (p.totalDays * p.dailyDose - p.remainingPills) / p.daysElapsed

/-- Theorem stating that for the given prescription, the daily dose is 2 pills -/
theorem johnson_prescription :
  ∃ (p : Prescription),
    p.totalDays = 30 ∧
    p.remainingPills = 12 ∧
    p.daysElapsed = 24 ∧
    calculateDailyDose p = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_johnson_prescription_l3553_355301


namespace NUMINAMATH_CALUDE_gre_exam_month_l3553_355322

-- Define the months as an enumeration
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

def next_month : Month → Month
| Month.January => Month.February
| Month.February => Month.March
| Month.March => Month.April
| Month.April => Month.May
| Month.May => Month.June
| Month.June => Month.July
| Month.July => Month.August
| Month.August => Month.September
| Month.September => Month.October
| Month.October => Month.November
| Month.November => Month.December
| Month.December => Month.January

def months_later (start : Month) (n : Nat) : Month :=
  match n with
  | 0 => start
  | n + 1 => next_month (months_later start n)

theorem gre_exam_month (start_month : Month) (preparation_months : Nat) :
  start_month = Month.June ∧ preparation_months = 5 →
  months_later start_month preparation_months = Month.November :=
by sorry

end NUMINAMATH_CALUDE_gre_exam_month_l3553_355322


namespace NUMINAMATH_CALUDE_x_value_when_y_is_3_l3553_355333

/-- The inverse square relationship between x and y -/
def inverse_square_relation (x y : ℝ) (k : ℝ) : Prop :=
  x = k / (y ^ 2)

/-- Theorem: Given the inverse square relationship between x and y,
    and the condition that x ≈ 0.1111111111111111 when y = 9,
    prove that x = 1 when y = 3 -/
theorem x_value_when_y_is_3
  (h1 : ∃ k, ∀ x y, inverse_square_relation x y k)
  (h2 : ∃ x, inverse_square_relation x 9 (9 * 0.1111111111111111)) :
  ∃ x, inverse_square_relation x 3 1 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_3_l3553_355333


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3553_355387

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 + 12 * x
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3553_355387


namespace NUMINAMATH_CALUDE_max_volume_side_length_l3553_355315

def sheet_length : ℝ := 90
def sheet_width : ℝ := 48

def container_volume (x : ℝ) : ℝ :=
  (sheet_length - 2 * x) * (sheet_width - 2 * x) * x

theorem max_volume_side_length :
  ∃ (x : ℝ), x > 0 ∧ x < sheet_width / 2 ∧ x < sheet_length / 2 ∧
  ∀ (y : ℝ), y > 0 → y < sheet_width / 2 → y < sheet_length / 2 →
  container_volume y ≤ container_volume x ∧
  x = 10 :=
sorry

end NUMINAMATH_CALUDE_max_volume_side_length_l3553_355315


namespace NUMINAMATH_CALUDE_evaluate_expression_l3553_355390

theorem evaluate_expression : 8^8 * 27^8 * 8^27 * 27^27 = 216^35 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3553_355390


namespace NUMINAMATH_CALUDE_puppy_group_arrangements_eq_2520_l3553_355389

/-- The number of ways to divide 12 puppies into groups of 4, 6, and 2,
    with Coco in the 4-puppy group and Rocky in the 6-puppy group. -/
def puppy_group_arrangements : ℕ :=
  Nat.choose 10 3 * Nat.choose 7 5

/-- Theorem stating that the number of puppy group arrangements is 2520. -/
theorem puppy_group_arrangements_eq_2520 :
  puppy_group_arrangements = 2520 := by
  sorry

#eval puppy_group_arrangements

end NUMINAMATH_CALUDE_puppy_group_arrangements_eq_2520_l3553_355389


namespace NUMINAMATH_CALUDE_age_difference_l3553_355358

/-- Proves that the difference between Rahul's and Sachin's ages is 9 years -/
theorem age_difference (sachin_age rahul_age : ℝ) : 
  sachin_age = 31.5 → 
  sachin_age / rahul_age = 7 / 9 → 
  rahul_age - sachin_age = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_age_difference_l3553_355358


namespace NUMINAMATH_CALUDE_bird_families_count_l3553_355311

/-- The number of bird families that flew away for winter -/
def flew_away : ℕ := 32

/-- The number of bird families that stayed near the mountain -/
def stayed : ℕ := 35

/-- The initial number of bird families living near the mountain -/
def initial_families : ℕ := flew_away + stayed

theorem bird_families_count : initial_families = 67 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_count_l3553_355311


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3553_355314

theorem polynomial_division_theorem (x : ℝ) :
  8 * x^3 - 2 * x^2 + 4 * x - 7 = (x - 1) * (8 * x^2 + 6 * x + 10) + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3553_355314


namespace NUMINAMATH_CALUDE_ax5_plus_by5_l3553_355319

theorem ax5_plus_by5 (a b x y : ℝ) 
  (eq1 : a*x + b*y = 1)
  (eq2 : a*x^2 + b*y^2 = 2)
  (eq3 : a*x^3 + b*y^3 = 5)
  (eq4 : a*x^4 + b*y^4 = 15) :
  a*x^5 + b*y^5 = -40 := by
  sorry

end NUMINAMATH_CALUDE_ax5_plus_by5_l3553_355319


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l3553_355348

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 12 * x^2 - 58 * x + 70
  ∃ x : ℝ, f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l3553_355348


namespace NUMINAMATH_CALUDE_min_cost_for_range_l3553_355351

/-- The cost of a "yes" answer in rubles -/
def yes_cost : ℕ := 2

/-- The cost of a "no" answer in rubles -/
def no_cost : ℕ := 1

/-- The range of possible hidden numbers -/
def number_range : Set ℕ := Finset.range 144

/-- The minimum cost function for guessing a number in a given set -/
noncomputable def min_cost (S : Set ℕ) : ℕ :=
  sorry

/-- The theorem stating that the minimum cost to guess any number in [1, 144] is 11 rubles -/
theorem min_cost_for_range : min_cost number_range = 11 :=
  sorry

end NUMINAMATH_CALUDE_min_cost_for_range_l3553_355351


namespace NUMINAMATH_CALUDE_smallest_top_number_l3553_355372

/-- Represents the pyramid structure -/
structure Pyramid :=
  (layer1 : Fin 15 → ℕ)
  (layer2 : Fin 10 → ℕ)
  (layer3 : Fin 6 → ℕ)
  (layer4 : Fin 3 → ℕ)
  (layer5 : ℕ)

/-- The numbering rule for layers 2-5 -/
def validNumbering (p : Pyramid) : Prop :=
  (∀ i : Fin 10, p.layer2 i = p.layer1 (3*i) + p.layer1 (3*i+1) + p.layer1 (3*i+2)) ∧
  (∀ i : Fin 6, p.layer3 i = p.layer2 (3*i) + p.layer2 (3*i+1) + p.layer2 (3*i+2)) ∧
  (∀ i : Fin 3, p.layer4 i = p.layer3 (2*i) + p.layer3 (2*i+1) + p.layer3 (2*i+2)) ∧
  (p.layer5 = p.layer4 0 + p.layer4 1 + p.layer4 2)

/-- The bottom layer contains numbers 1 to 15 -/
def validBottomLayer (p : Pyramid) : Prop :=
  (∀ i : Fin 15, p.layer1 i ∈ Finset.range 16 \ {0}) ∧
  (∀ i j : Fin 15, i ≠ j → p.layer1 i ≠ p.layer1 j)

/-- The theorem stating the smallest possible number for the top block -/
theorem smallest_top_number (p : Pyramid) 
  (h1 : validNumbering p) (h2 : validBottomLayer p) : 
  p.layer5 ≥ 155 :=
sorry

end NUMINAMATH_CALUDE_smallest_top_number_l3553_355372


namespace NUMINAMATH_CALUDE_incorrect_deduction_l3553_355324

/-- Definition of an exponential function -/
def IsExponentialFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 1 ∧ ∀ x, f x = a^x

/-- Definition of a power function -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, α > 1 ∧ ∀ x, f x = x^α

/-- Definition of an increasing function -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem -/
theorem incorrect_deduction :
  (∀ f : ℝ → ℝ, IsExponentialFunction f → IsIncreasing f) →
  ¬(∀ f : ℝ → ℝ, IsPowerFunction f → IsIncreasing f) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_deduction_l3553_355324


namespace NUMINAMATH_CALUDE_coloring_books_removed_l3553_355385

theorem coloring_books_removed (initial_stock : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 86 →
  shelves = 7 →
  books_per_shelf = 7 →
  initial_stock - (shelves * books_per_shelf) = 37 := by
sorry

end NUMINAMATH_CALUDE_coloring_books_removed_l3553_355385


namespace NUMINAMATH_CALUDE_equation_satisfied_l3553_355335

theorem equation_satisfied (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l3553_355335


namespace NUMINAMATH_CALUDE_wetland_area_scientific_notation_l3553_355342

/-- Represents the scientific notation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a number to its scientific notation representation -/
def toScientificNotation (n : ℝ) : ScientificNotation :=
  sorry

theorem wetland_area_scientific_notation :
  toScientificNotation (29.47 * 1000) = ScientificNotation.mk 2.947 4 :=
sorry

end NUMINAMATH_CALUDE_wetland_area_scientific_notation_l3553_355342


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3553_355321

theorem water_tank_capacity (tank_capacity : ℝ) : 
  (0.6 * tank_capacity - (0.7 * tank_capacity) = 45) → 
  tank_capacity = 450 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3553_355321


namespace NUMINAMATH_CALUDE_license_plate_increase_l3553_355367

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^3
  let new_plates := 26^3 * 10^3 * 5
  new_plates / old_plates = 130 := by sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3553_355367


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3553_355303

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 84 →
  E = 4 * F + 18 →
  D + E + F = 180 →
  F = 15.6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3553_355303


namespace NUMINAMATH_CALUDE_chessboard_division_exists_l3553_355386

-- Define a chessboard piece
structure ChessboardPiece where
  total_squares : ℕ
  black_squares : ℕ

-- Define a chessboard division
structure ChessboardDivision where
  piece1 : ChessboardPiece
  piece2 : ChessboardPiece

-- Define the property of being a valid chessboard division
def is_valid_division (d : ChessboardDivision) : Prop :=
  d.piece1.total_squares + d.piece2.total_squares = 64 ∧
  d.piece1.total_squares = d.piece2.total_squares + 4 ∧
  d.piece2.black_squares = d.piece1.black_squares + 4 ∧
  d.piece1.black_squares + d.piece2.black_squares = 32

-- Theorem statement
theorem chessboard_division_exists : ∃ d : ChessboardDivision, is_valid_division d :=
sorry

end NUMINAMATH_CALUDE_chessboard_division_exists_l3553_355386


namespace NUMINAMATH_CALUDE_question_probabilities_l3553_355364

def total_questions : ℕ := 5
def algebra_questions : ℕ := 2
def geometry_questions : ℕ := 3

theorem question_probabilities :
  let prob_algebra_then_geometry := (algebra_questions : ℚ) / total_questions * 
                                    (geometry_questions : ℚ) / (total_questions - 1)
  let prob_geometry_given_algebra := (geometry_questions : ℚ) / (total_questions - 1)
  prob_algebra_then_geometry = 3 / 10 ∧ prob_geometry_given_algebra = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_question_probabilities_l3553_355364


namespace NUMINAMATH_CALUDE_exists_integer_root_polynomial_l3553_355339

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Function to evaluate a quadratic polynomial at a given x -/
def evaluate (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- Predicate to check if a quadratic polynomial has integer roots -/
def has_integer_roots (p : QuadraticPolynomial) : Prop :=
  ∃ (r₁ r₂ : ℤ), p.a * r₁^2 + p.b * r₁ + p.c = 0 ∧ p.a * r₂^2 + p.b * r₂ + p.c = 0

/-- The main theorem -/
theorem exists_integer_root_polynomial :
  ∃ (p : QuadraticPolynomial),
    p.a = 1 ∧
    (evaluate p (-1) ≤ evaluate ⟨1, 10, 20⟩ (-1) ∧ evaluate p (-1) ≥ evaluate ⟨1, 20, 10⟩ (-1)) ∧
    has_integer_roots p :=
by
  sorry

end NUMINAMATH_CALUDE_exists_integer_root_polynomial_l3553_355339


namespace NUMINAMATH_CALUDE_total_flooring_cost_l3553_355378

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a room given its dimensions -/
def roomArea (d : RoomDimensions) : ℝ := d.length * d.width

/-- Calculates the cost of flooring for a room given its area and slab rate -/
def roomCost (area : ℝ) (slabRate : ℝ) : ℝ := area * slabRate

/-- Theorem: The total cost of flooring for the house is Rs. 81,390 -/
theorem total_flooring_cost : 
  let room1 : RoomDimensions := ⟨5.5, 3.75⟩
  let room2 : RoomDimensions := ⟨6, 4.2⟩
  let room3 : RoomDimensions := ⟨4.8, 3.25⟩
  let slabRate1 : ℝ := 1200
  let slabRate2 : ℝ := 1350
  let slabRate3 : ℝ := 1450
  let totalCost : ℝ := 
    roomCost (roomArea room1) slabRate1 + 
    roomCost (roomArea room2) slabRate2 + 
    roomCost (roomArea room3) slabRate3
  totalCost = 81390 := by
  sorry

end NUMINAMATH_CALUDE_total_flooring_cost_l3553_355378


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l3553_355353

noncomputable def nested_sqrt_sequence : ℕ → ℝ
  | 0 => Real.sqrt 86
  | n + 1 => Real.sqrt (86 + 41 * nested_sqrt_sequence n)

theorem nested_sqrt_value :
  ∃ (limit : ℝ), limit = Real.sqrt (86 + 41 * limit) ∧ limit = 43 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l3553_355353


namespace NUMINAMATH_CALUDE_condition_p_neither_sufficient_nor_necessary_for_q_l3553_355360

theorem condition_p_neither_sufficient_nor_necessary_for_q :
  ¬(∀ x : ℝ, (1 / x ≤ 1) → (x^2 - 2*x ≥ 0)) ∧
  ¬(∀ x : ℝ, (x^2 - 2*x ≥ 0) → (1 / x ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_p_neither_sufficient_nor_necessary_for_q_l3553_355360


namespace NUMINAMATH_CALUDE_tom_allowance_l3553_355317

theorem tom_allowance (initial_allowance : ℝ) 
  (first_week_fraction : ℝ) (second_week_fraction : ℝ) : 
  initial_allowance = 12 →
  first_week_fraction = 1/3 →
  second_week_fraction = 1/4 →
  let remaining_after_first_week := initial_allowance - (initial_allowance * first_week_fraction)
  let final_remaining := remaining_after_first_week - (remaining_after_first_week * second_week_fraction)
  final_remaining = 6 := by
sorry

end NUMINAMATH_CALUDE_tom_allowance_l3553_355317


namespace NUMINAMATH_CALUDE_b_25_mod_35_l3553_355354

/-- b_n is the integer obtained by writing all integers from 1 to n from left to right, each repeated twice -/
def b (n : ℕ) : ℕ :=
  -- Definition of b_n goes here
  sorry

/-- The remainder when b_25 is divided by 35 is 6 -/
theorem b_25_mod_35 : b 25 % 35 = 6 := by
  sorry

end NUMINAMATH_CALUDE_b_25_mod_35_l3553_355354


namespace NUMINAMATH_CALUDE_alien_tree_age_l3553_355395

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The age of the tree in the alien's base-8 system --/
def alienAge : Nat := base8ToBase10 3 6 7

theorem alien_tree_age : alienAge = 247 := by
  sorry

end NUMINAMATH_CALUDE_alien_tree_age_l3553_355395


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_to_2023_l3553_355300

theorem opposite_of_negative_one_to_2023 :
  ∀ n : ℕ, n = 2023 → Odd n → (-((-1)^n)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_to_2023_l3553_355300


namespace NUMINAMATH_CALUDE_birds_in_marsh_l3553_355346

theorem birds_in_marsh (geese ducks : ℕ) (h1 : geese = 58) (h2 : ducks = 37) :
  geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_marsh_l3553_355346


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_property_l3553_355365

/-- Given a geometric sequence {aₙ} satisfying certain conditions, 
    prove that tan(a₁a₁₃) = √3 -/
theorem geometric_sequence_tan_property 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_condition : a 3 * a 11 + 2 * (a 7)^2 = 4 * Real.pi) : 
  Real.tan (a 1 * a 13) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_property_l3553_355365


namespace NUMINAMATH_CALUDE_elderly_in_sample_is_18_l3553_355399

/-- Represents the distribution of employees in a company and their sampling --/
structure EmployeeSampling where
  total : ℕ
  young : ℕ
  elderly : ℕ
  sampledYoung : ℕ
  middleAged : ℕ := 2 * elderly
  youngRatio : ℚ := young / total
  elderlyRatio : ℚ := elderly / total

/-- The number of elderly employees in the sample given the conditions --/
def elderlyInSample (e : EmployeeSampling) : ℚ :=
  e.elderlyRatio * (e.sampledYoung / e.youngRatio)

/-- Theorem stating the number of elderly employees in the sample --/
theorem elderly_in_sample_is_18 (e : EmployeeSampling) 
    (h1 : e.total = 430)
    (h2 : e.young = 160)
    (h3 : e.sampledYoung = 32)
    (h4 : e.total = e.young + e.middleAged + e.elderly) :
  elderlyInSample e = 18 := by
  sorry

#eval elderlyInSample { total := 430, young := 160, elderly := 90, sampledYoung := 32 }

end NUMINAMATH_CALUDE_elderly_in_sample_is_18_l3553_355399


namespace NUMINAMATH_CALUDE_average_weight_of_a_and_b_l3553_355376

theorem average_weight_of_a_and_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 47 →
  b = 39 →
  (a + b) / 2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_a_and_b_l3553_355376


namespace NUMINAMATH_CALUDE_problem_solution_l3553_355384

theorem problem_solution (a b : ℝ) (h1 : |a| = 5) (h2 : b = -2) (h3 : a * b > 0) :
  a + b = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3553_355384


namespace NUMINAMATH_CALUDE_ellipse_equation_l3553_355370

noncomputable section

-- Define the ellipse C
def C (x y : ℝ) (a b : ℝ) : Prop := y^2 / a^2 + x^2 / b^2 = 1

-- Define the foci
def F₁ (c : ℝ) : ℝ × ℝ := (-c, 0)
def F₂ (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define point A
def A : ℝ × ℝ := (2, 0)

-- Define the slope product condition
def slope_product (c : ℝ) : Prop :=
  let k_AF₁ := (0 - (-c)) / (2 - 0)
  let k_AF₂ := (0 - c) / (2 - 0)
  k_AF₁ * k_AF₂ = -1/4

-- Define the distance sum condition for point B
def distance_sum (a : ℝ) : Prop := 2*a = 2*Real.sqrt 2

-- Main theorem
theorem ellipse_equation (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (∃ c : ℝ, slope_product c ∧ distance_sum a) →
  (∀ x y : ℝ, C x y a b ↔ y^2/2 + x^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3553_355370


namespace NUMINAMATH_CALUDE_apple_weight_is_quarter_pound_l3553_355361

/-- The weight of a small apple in pounds -/
def apple_weight : ℝ := 0.25

/-- The cost of apples per pound in dollars -/
def cost_per_pound : ℝ := 2

/-- The total amount spent on apples in dollars -/
def total_spent : ℝ := 7

/-- The number of days the apples should last -/
def days : ℕ := 14

/-- Theorem stating that the weight of a small apple is 0.25 pounds -/
theorem apple_weight_is_quarter_pound :
  apple_weight = total_spent / (cost_per_pound * days) := by sorry

end NUMINAMATH_CALUDE_apple_weight_is_quarter_pound_l3553_355361


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3553_355347

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, n.val^2 + 5*n.val + 13 = m^2) → n.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3553_355347


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l3553_355310

theorem fraction_sum_zero (a b c : ℝ) 
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l3553_355310


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3553_355359

theorem arithmetic_calculation : 12 - (-18) + (-7) = 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3553_355359


namespace NUMINAMATH_CALUDE_winter_olympics_volunteer_allocation_l3553_355337

theorem winter_olympics_volunteer_allocation :
  let n_volunteers : ℕ := 5
  let n_events : ℕ := 4
  let allocation_schemes : ℕ := (n_volunteers.choose 2) * n_events.factorial
  allocation_schemes = 240 :=
by sorry

end NUMINAMATH_CALUDE_winter_olympics_volunteer_allocation_l3553_355337


namespace NUMINAMATH_CALUDE_exam_average_is_36_l3553_355304

/-- The overall average of marks obtained by all boys in an examination. -/
def overall_average (total_boys : ℕ) (passed_boys : ℕ) (avg_passed : ℕ) (avg_failed : ℕ) : ℚ :=
  let failed_boys := total_boys - passed_boys
  ((passed_boys * avg_passed + failed_boys * avg_failed) : ℚ) / total_boys

/-- Theorem stating that the overall average of marks is 36 given the conditions. -/
theorem exam_average_is_36 :
  overall_average 120 105 39 15 = 36 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_is_36_l3553_355304


namespace NUMINAMATH_CALUDE_p_is_cubic_l3553_355363

/-- The polynomial under consideration -/
def p (x : ℝ) : ℝ := 2^3 + 2^2*x - 2*x^2 - x^3

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

theorem p_is_cubic : degree p = 3 := by sorry

end NUMINAMATH_CALUDE_p_is_cubic_l3553_355363


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l3553_355334

/-- Given a line y = -3x + 5 and two points (-6, y₁) and (3, y₂) on this line,
    prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = -3 * (-6) + 5) →  -- Point (-6, y₁) lies on the line
  (y₂ = -3 * 3 + 5) →     -- Point (3, y₂) lies on the line
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l3553_355334


namespace NUMINAMATH_CALUDE_probability_of_blank_in_specific_lottery_l3553_355344

/-- The probability of getting a blank in a lottery with prizes and blanks. -/
def probability_of_blank (prizes : ℕ) (blanks : ℕ) : ℚ :=
  blanks / (prizes + blanks)

/-- Theorem stating that the probability of getting a blank in a lottery 
    with 10 prizes and 25 blanks is 5/7. -/
theorem probability_of_blank_in_specific_lottery : 
  probability_of_blank 10 25 = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_blank_in_specific_lottery_l3553_355344


namespace NUMINAMATH_CALUDE_minimize_F_l3553_355382

/-- The optimization problem -/
def OptimizationProblem (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  x₁ ≥ 0 ∧ x₂ ≥ 0 ∧
  -2 * x₁ + x₂ + x₃ = 2 ∧
  x₁ - 2 * x₂ + x₄ = 2 ∧
  x₁ + x₂ + x₅ = 5

/-- The objective function -/
def F (x₁ x₂ : ℝ) : ℝ := x₂ - x₁

/-- The theorem stating the minimum value of F and the point where it's achieved -/
theorem minimize_F :
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ),
    OptimizationProblem x₁ x₂ x₃ x₄ x₅ ∧
    F x₁ x₂ = -3 ∧
    x₁ = 4 ∧ x₂ = 1 ∧ x₃ = 9 ∧ x₄ = 0 ∧ x₅ = 0 ∧
    ∀ (y₁ y₂ y₃ y₄ y₅ : ℝ), OptimizationProblem y₁ y₂ y₃ y₄ y₅ → F y₁ y₂ ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_minimize_F_l3553_355382


namespace NUMINAMATH_CALUDE_fraction_product_equivalence_l3553_355379

theorem fraction_product_equivalence (f g : ℝ → ℝ) :
  ∀ x : ℝ, g x ≠ 0 → (f x / g x > 0 ↔ f x * g x > 0) := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equivalence_l3553_355379


namespace NUMINAMATH_CALUDE_some_seniors_not_club_members_l3553_355308

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Senior : U → Prop)
variable (Punctual : U → Prop)
variable (ClubMember : U → Prop)

-- State the theorem
theorem some_seniors_not_club_members
  (h1 : ∃ x, Senior x ∧ ¬Punctual x)
  (h2 : ∀ x, ClubMember x → Punctual x) :
  ∃ x, Senior x ∧ ¬ClubMember x :=
by
  sorry


end NUMINAMATH_CALUDE_some_seniors_not_club_members_l3553_355308


namespace NUMINAMATH_CALUDE_state_fair_earnings_l3553_355325

theorem state_fair_earnings :
  let ticket_price : ℚ := 5
  let food_price : ℚ := 8
  let ride_price : ℚ := 4
  let souvenir_price : ℚ := 15
  let total_ticket_sales : ℚ := 2520
  let num_attendees : ℚ := total_ticket_sales / ticket_price
  let food_buyers_ratio : ℚ := 2/3
  let ride_goers_ratio : ℚ := 1/4
  let souvenir_buyers_ratio : ℚ := 1/8
  let food_earnings : ℚ := num_attendees * food_buyers_ratio * food_price
  let ride_earnings : ℚ := num_attendees * ride_goers_ratio * ride_price
  let souvenir_earnings : ℚ := num_attendees * souvenir_buyers_ratio * souvenir_price
  let total_earnings : ℚ := total_ticket_sales + food_earnings + ride_earnings + souvenir_earnings
  total_earnings = 6657 := by sorry

end NUMINAMATH_CALUDE_state_fair_earnings_l3553_355325


namespace NUMINAMATH_CALUDE_sugar_weighing_l3553_355368

theorem sugar_weighing (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p ≠ q) :
  p / q + q / p > 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_weighing_l3553_355368


namespace NUMINAMATH_CALUDE_equilateral_triangles_count_l3553_355327

/-- Counts the number of equilateral triangles in an equilateral triangular grid -/
def count_equilateral_triangles (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) * (n + 3) / 24

/-- The side length of the equilateral triangular grid -/
def grid_side_length : ℕ := 4

/-- Theorem: The number of equilateral triangles in a grid of side length 4 is 35 -/
theorem equilateral_triangles_count :
  count_equilateral_triangles grid_side_length = 35 := by
  sorry

#eval count_equilateral_triangles grid_side_length

end NUMINAMATH_CALUDE_equilateral_triangles_count_l3553_355327


namespace NUMINAMATH_CALUDE_largest_digit_change_l3553_355356

def original_sum : ℕ := 2570
def correct_sum : ℕ := 2580
def num1 : ℕ := 725
def num2 : ℕ := 864
def num3 : ℕ := 991

theorem largest_digit_change :
  ∃ (d : ℕ), d ≤ 9 ∧ 
  (num1 + num2 + (num3 - 10) = correct_sum) ∧
  (∀ (d' : ℕ), d' > d → 
    (num1 + num2 + num3 - d' * 10 ≠ correct_sum ∧ 
     num1 + (num2 - d' * 10) + num3 ≠ correct_sum ∧
     (num1 - d' * 10) + num2 + num3 ≠ correct_sum)) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_change_l3553_355356


namespace NUMINAMATH_CALUDE_max_value_expression_l3553_355312

theorem max_value_expression (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (|7*a + 8*b - a*b| + |2*a + 8*b - 6*a*b|) / (a * Real.sqrt (1 + b^2)) ≤ 9 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3553_355312


namespace NUMINAMATH_CALUDE_statement_C_is_incorrect_l3553_355396

theorem statement_C_is_incorrect : ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_statement_C_is_incorrect_l3553_355396


namespace NUMINAMATH_CALUDE_parabola_transformation_transformation_is_right_shift_2_l3553_355388

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ := (x + 5) * (x - 3)

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ := (x + 3) * (x - 5)

-- Define the transformation
def transformation (x : ℝ) : ℝ := x + 2

-- Theorem stating the transformation between the two parabolas
theorem parabola_transformation :
  ∀ x : ℝ, parabola1 x = parabola2 (transformation x) :=
by
  sorry

-- Theorem stating that the transformation is a shift of 2 units to the right
theorem transformation_is_right_shift_2 :
  ∀ x : ℝ, transformation x = x + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_transformation_is_right_shift_2_l3553_355388


namespace NUMINAMATH_CALUDE_equal_water_levels_l3553_355393

/-- Represents a pool with initial height and drain time -/
structure Pool where
  initial_height : ℝ
  drain_time : ℝ

/-- The time when water levels in two pools become equal -/
def equal_level_time (pool_a pool_b : Pool) : ℝ :=
  1 -- The actual value we want to prove

theorem equal_water_levels (pool_a pool_b : Pool) :
  pool_b.initial_height = 1.5 * pool_a.initial_height →
  pool_a.drain_time = 2 →
  pool_b.drain_time = 1.5 →
  equal_level_time pool_a pool_b = 1 := by
  sorry

#check equal_water_levels

end NUMINAMATH_CALUDE_equal_water_levels_l3553_355393


namespace NUMINAMATH_CALUDE_m_range_l3553_355366

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 + m * x + 1 > 0

def q (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ∀ x y : ℝ, x^2 / (m - 1) + y^2 / (m - 2) = 1 ↔ 
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the set of m values
def M : Set ℝ := {m : ℝ | (0 ≤ m ∧ m ≤ 1) ∨ (2 ≤ m ∧ m < 4)}

-- State the theorem
theorem m_range : 
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ ∀ m : ℝ, m ∈ M :=
sorry

end NUMINAMATH_CALUDE_m_range_l3553_355366


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l3553_355394

theorem pure_imaginary_solutions (x : ℂ) :
  (x^4 - 5*x^3 + 10*x^2 - 50*x - 75 = 0) ∧ (∃ k : ℝ, x = k * I) ↔
  (x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l3553_355394


namespace NUMINAMATH_CALUDE_qualified_products_l3553_355398

theorem qualified_products (defect_rate : ℝ) (total_items : ℕ) : 
  defect_rate = 0.005 →
  total_items = 18000 →
  ⌊(1 - defect_rate) * total_items⌋ = 17910 := by
sorry

end NUMINAMATH_CALUDE_qualified_products_l3553_355398


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3553_355374

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let A : ℝ × ℝ := (x, 6)
  let B : ℝ × ℝ := reflect_over_y_axis A
  A.1 + A.2 + B.1 + B.2 = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3553_355374


namespace NUMINAMATH_CALUDE_triangle_sides_from_heights_and_median_l3553_355328

/-- Given a triangle with heights m₁ and m₂ corresponding to sides a and b respectively,
    and median k₃ corresponding to side c, prove that the sides a and b can be expressed as:
    a = m₂ / sin(γ) and b = m₁ / sin(γ), where γ is the angle opposite to side c. -/
theorem triangle_sides_from_heights_and_median 
  (m₁ m₂ k₃ : ℝ) (γ : ℝ) (hm₁ : m₁ > 0) (hm₂ : m₂ > 0) (hk₃ : k₃ > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ (a b : ℝ), a = m₂ / Real.sin γ ∧ b = m₁ / Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_from_heights_and_median_l3553_355328


namespace NUMINAMATH_CALUDE_estimate_larger_than_original_l3553_355332

theorem estimate_larger_than_original 
  (x y ε δ : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : x > y) 
  (hε : ε > 0) 
  (hδ : δ > 0) 
  (hεδ : ε ≠ δ) : 
  (x + ε) - (y - δ) > x - y := by
sorry

end NUMINAMATH_CALUDE_estimate_larger_than_original_l3553_355332


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3553_355323

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (3 * a + 9 * b) % 63 = 45 ∧ (7 * a) % 63 = 1 ∧ (13 * b) % 63 = 1 :=
by sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3553_355323


namespace NUMINAMATH_CALUDE_unique_solution_xy_l3553_355318

/-- The unique solution to the system of equations x^y + 3 = y^x and 2x^y = y^x + 11 -/
theorem unique_solution_xy : ∃! (x y : ℕ+), 
  (x : ℝ) ^ (y : ℝ) + 3 = (y : ℝ) ^ (x : ℝ) ∧ 
  2 * (x : ℝ) ^ (y : ℝ) = (y : ℝ) ^ (x : ℝ) + 11 ∧
  x = 14 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_xy_l3553_355318


namespace NUMINAMATH_CALUDE_garden_multiplier_l3553_355380

theorem garden_multiplier (width length perimeter : ℝ) 
  (h1 : perimeter = 2 * length + 2 * width)
  (h2 : perimeter = 100)
  (h3 : length = 38)
  (h4 : ∃ m : ℝ, length = m * width + 2) :
  ∃ m : ℝ, length = m * width + 2 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_garden_multiplier_l3553_355380


namespace NUMINAMATH_CALUDE_derivative_y_l3553_355338

noncomputable def y (x : ℝ) : ℝ := Real.cos x / x

theorem derivative_y (x : ℝ) (hx : x ≠ 0) :
  deriv y x = -((x * Real.sin x + Real.cos x) / x^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_y_l3553_355338


namespace NUMINAMATH_CALUDE_system_of_equations_proof_l3553_355397

theorem system_of_equations_proof (a b c d : ℂ) 
  (eq1 : a - b - c + d = 12)
  (eq2 : a + b - c - d = 6)
  (eq3 : 2*a + c - d = 15) :
  (b - d)^2 = 9 := by sorry

end NUMINAMATH_CALUDE_system_of_equations_proof_l3553_355397


namespace NUMINAMATH_CALUDE_sock_pairs_count_l3553_355369

def total_socks : ℕ := 12
def white_socks : ℕ := 5
def brown_socks : ℕ := 5
def blue_socks : ℕ := 2

def same_color_pairs : ℕ := Nat.choose white_socks 2 + Nat.choose brown_socks 2 + Nat.choose blue_socks 2

theorem sock_pairs_count : same_color_pairs = 21 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l3553_355369


namespace NUMINAMATH_CALUDE_gcd_nine_factorial_seven_factorial_squared_l3553_355316

theorem gcd_nine_factorial_seven_factorial_squared :
  Nat.gcd (Nat.factorial 9) ((Nat.factorial 7)^2) = 362880 := by
  sorry

end NUMINAMATH_CALUDE_gcd_nine_factorial_seven_factorial_squared_l3553_355316


namespace NUMINAMATH_CALUDE_cubic_polynomial_determinant_l3553_355383

/-- Given a cubic polynomial x^3 + sx^2 + px + q with roots a, b, and c,
    the determinant of the matrix [[s + a, 1, 1], [1, s + b, 1], [1, 1, s + c]]
    is equal to s^3 + sp - q - 2s - 2(p - s) -/
theorem cubic_polynomial_determinant (s p q a b c : ℝ) : 
  a^3 + s*a^2 + p*a + q = 0 →
  b^3 + s*b^2 + p*b + q = 0 →
  c^3 + s*c^2 + p*c + q = 0 →
  Matrix.det ![![s + a, 1, 1], ![1, s + b, 1], ![1, 1, s + c]] = s^3 + s*p - q - 2*s - 2*(p - s) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_determinant_l3553_355383


namespace NUMINAMATH_CALUDE_perpendicular_vectors_y_value_l3553_355345

theorem perpendicular_vectors_y_value :
  let a : Fin 3 → ℝ := ![1, 2, 6]
  let b : Fin 3 → ℝ := ![2, y, -1]
  (∀ i : Fin 3, (a • b) = 0) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_y_value_l3553_355345


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3553_355313

-- Define the set A
def A : Set ℝ := {x | x^2 - x ≤ 0}

-- Define the function f
def f (x : ℝ) : ℝ := 2 - x

-- Define the set B as the range of f on A
def B : Set ℝ := f '' A

-- State the theorem
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3553_355313


namespace NUMINAMATH_CALUDE_first_four_terms_l3553_355391

def a (n : ℕ) : ℚ := (1 + (-1)^(n+1)) / 2

theorem first_four_terms :
  (a 1 = 1) ∧ (a 2 = 0) ∧ (a 3 = 1) ∧ (a 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_first_four_terms_l3553_355391


namespace NUMINAMATH_CALUDE_machine_work_time_l3553_355355

theorem machine_work_time (x : ℝ) (h1 : x > 0) 
  (h2 : 1/x + 1/2 + 1/6 = 11/12) : x = 4 := by
  sorry

#check machine_work_time

end NUMINAMATH_CALUDE_machine_work_time_l3553_355355


namespace NUMINAMATH_CALUDE_vince_earnings_per_head_l3553_355381

/-- Represents Vince's hair salon business model -/
structure HairSalon where
  earningsPerHead : ℝ
  customersPerMonth : ℕ
  monthlyRentAndElectricity : ℝ
  recreationPercentage : ℝ
  monthlySavings : ℝ

/-- Theorem stating that Vince's earnings per head is $72 -/
theorem vince_earnings_per_head (salon : HairSalon)
    (h1 : salon.customersPerMonth = 80)
    (h2 : salon.monthlyRentAndElectricity = 280)
    (h3 : salon.recreationPercentage = 0.2)
    (h4 : salon.monthlySavings = 872)
    (h5 : salon.earningsPerHead * ↑salon.customersPerMonth * (1 - salon.recreationPercentage) =
          salon.earningsPerHead * ↑salon.customersPerMonth - salon.monthlyRentAndElectricity - salon.monthlySavings) :
    salon.earningsPerHead = 72 := by
  sorry

#check vince_earnings_per_head

end NUMINAMATH_CALUDE_vince_earnings_per_head_l3553_355381


namespace NUMINAMATH_CALUDE_carol_wins_probability_l3553_355340

/-- Represents the probability of tossing a six -/
def prob_six : ℚ := 1 / 6

/-- Represents the probability of not tossing a six -/
def prob_not_six : ℚ := 1 - prob_six

/-- Represents the number of players -/
def num_players : ℕ := 4

/-- The probability of Carol winning in one cycle -/
def prob_carol_win_cycle : ℚ := prob_not_six^2 * prob_six * prob_not_six

/-- The probability of no one winning in one cycle -/
def prob_no_win_cycle : ℚ := prob_not_six^num_players

/-- Theorem: The probability of Carol being the first to toss a six 
    in a repeated die-tossing game with four players is 125/671 -/
theorem carol_wins_probability : 
  prob_carol_win_cycle / (1 - prob_no_win_cycle) = 125 / 671 := by
  sorry

end NUMINAMATH_CALUDE_carol_wins_probability_l3553_355340


namespace NUMINAMATH_CALUDE_line_inclination_sine_l3553_355331

/-- Given a straight line 3x - 4y + 5 = 0 with angle of inclination α, prove that sin(α) = 3/5 -/
theorem line_inclination_sine (x y : ℝ) (α : ℝ) 
  (h : 3 * x - 4 * y + 5 = 0) 
  (h_incl : α = Real.arctan (3 / 4)) : 
  Real.sin α = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_sine_l3553_355331


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt3_over_2_l3553_355350

theorem cos_sin_sum_equals_sqrt3_over_2 :
  Real.cos (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (43 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt3_over_2_l3553_355350


namespace NUMINAMATH_CALUDE_height_to_radius_ratio_l3553_355307

/-- A regular triangular prism -/
structure RegularTriangularPrism where
  /-- The cosine of the dihedral angle between a face and the base -/
  cos_dihedral_angle : ℝ
  /-- The height of the prism -/
  height : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ

/-- Theorem: For a regular triangular prism where the cosine of the dihedral angle 
    between a face and the base is 1/6, the ratio of the height to the radius 
    of the inscribed sphere is 7 -/
theorem height_to_radius_ratio (prism : RegularTriangularPrism) 
    (h : prism.cos_dihedral_angle = 1/6) : 
    prism.height / prism.inscribed_radius = 7 := by
  sorry

end NUMINAMATH_CALUDE_height_to_radius_ratio_l3553_355307
