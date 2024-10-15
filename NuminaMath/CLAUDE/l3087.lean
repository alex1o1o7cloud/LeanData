import Mathlib

namespace NUMINAMATH_CALUDE_school_population_after_new_students_l3087_308793

theorem school_population_after_new_students (initial_avg_age initial_num_students new_students new_avg_age avg_decrease : ℝ) :
  initial_avg_age = 48 →
  new_students = 120 →
  new_avg_age = 32 →
  avg_decrease = 4 →
  (initial_avg_age * initial_num_students + new_avg_age * new_students) / (initial_num_students + new_students) = initial_avg_age - avg_decrease →
  initial_num_students + new_students = 480 := by
sorry

end NUMINAMATH_CALUDE_school_population_after_new_students_l3087_308793


namespace NUMINAMATH_CALUDE_wall_width_calculation_l3087_308745

theorem wall_width_calculation (mirror_side : ℝ) (wall_length : ℝ) :
  mirror_side = 18 →
  wall_length = 20.25 →
  (mirror_side * mirror_side) * 2 = wall_length * (648 / wall_length) :=
by
  sorry

#check wall_width_calculation

end NUMINAMATH_CALUDE_wall_width_calculation_l3087_308745


namespace NUMINAMATH_CALUDE_integer_division_problem_l3087_308786

theorem integer_division_problem (D d q r : ℤ) 
  (h1 : D = q * d + r) 
  (h2 : D + 65 = q * (d + 5) + r) : q = 13 := by
  sorry

end NUMINAMATH_CALUDE_integer_division_problem_l3087_308786


namespace NUMINAMATH_CALUDE_log_expression_equality_l3087_308790

theorem log_expression_equality : 
  (Real.log 3 / Real.log 2 + Real.log 3 / Real.log 8) / (Real.log 9 / Real.log 4) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l3087_308790


namespace NUMINAMATH_CALUDE_binary_digit_difference_l3087_308726

theorem binary_digit_difference : ∃ (n m : ℕ), n = 400 ∧ m = 1600 ∧ 
  (Nat.log 2 m + 1) - (Nat.log 2 n + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l3087_308726


namespace NUMINAMATH_CALUDE_octopus_leg_configuration_l3087_308719

-- Define the possible number of legs for an octopus
inductive LegCount : Type where
  | six : LegCount
  | seven : LegCount
  | eight : LegCount

-- Define the colors of the octopuses
inductive Color : Type where
  | blue : Color
  | green : Color
  | yellow : Color
  | red : Color

-- Define a function to determine if an octopus is telling the truth
def isTruthful (legs : LegCount) : Prop :=
  match legs with
  | LegCount.six => True
  | LegCount.seven => False
  | LegCount.eight => True

-- Define a function to convert LegCount to a natural number
def legCountToNat (legs : LegCount) : Nat :=
  match legs with
  | LegCount.six => 6
  | LegCount.seven => 7
  | LegCount.eight => 8

-- Define the claims made by each octopus
def claim (color : Color) : Nat :=
  match color with
  | Color.blue => 28
  | Color.green => 27
  | Color.yellow => 26
  | Color.red => 25

-- Define the theorem
theorem octopus_leg_configuration :
  ∃ (legs : Color → LegCount),
    (legs Color.green = LegCount.six) ∧
    (legs Color.blue = LegCount.seven) ∧
    (legs Color.yellow = LegCount.seven) ∧
    (legs Color.red = LegCount.seven) ∧
    (∀ c, isTruthful (legs c) ↔ (legCountToNat (legs Color.blue) + legCountToNat (legs Color.green) + legCountToNat (legs Color.yellow) + legCountToNat (legs Color.red) = claim c)) :=
sorry

end NUMINAMATH_CALUDE_octopus_leg_configuration_l3087_308719


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l3087_308740

theorem two_numbers_with_given_means (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (Real.sqrt (a * b) = Real.sqrt 5) → 
  (2 / (1/a + 1/b) = 5/3) → 
  ((a = 5 ∧ b = 1) ∨ (a = 1 ∧ b = 5)) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l3087_308740


namespace NUMINAMATH_CALUDE_largest_area_cross_section_passes_through_center_exists_larger_radius_non_center_cross_section_l3087_308795

-- Define a convex centrally symmetric polyhedron
structure ConvexCentrallySymmetricPolyhedron where
  -- Add necessary fields and properties
  is_convex : Bool
  is_centrally_symmetric : Bool

-- Define a cross-section of the polyhedron
structure CrossSection where
  polyhedron : ConvexCentrallySymmetricPolyhedron
  plane : Plane
  passes_through_center : Bool

-- Define the area of a cross-section
def area (cs : CrossSection) : ℝ := sorry

-- Define the radius of the smallest enclosing circle of a cross-section
def smallest_enclosing_circle_radius (cs : CrossSection) : ℝ := sorry

-- Theorem 1: The cross-section with the largest area passes through the center
theorem largest_area_cross_section_passes_through_center 
  (p : ConvexCentrallySymmetricPolyhedron) :
  ∀ (cs : CrossSection), cs.polyhedron = p → 
    ∃ (center_cs : CrossSection), 
      center_cs.polyhedron = p ∧ 
      center_cs.passes_through_center = true ∧
      area center_cs ≥ area cs :=
sorry

-- Theorem 2: There exists a cross-section not passing through the center with a larger 
-- radius of the smallest enclosing circle than the cross-section passing through the center
theorem exists_larger_radius_non_center_cross_section 
  (p : ConvexCentrallySymmetricPolyhedron) :
  ∃ (cs_non_center cs_center : CrossSection), 
    cs_non_center.polyhedron = p ∧ 
    cs_center.polyhedron = p ∧
    cs_non_center.passes_through_center = false ∧
    cs_center.passes_through_center = true ∧
    smallest_enclosing_circle_radius cs_non_center > smallest_enclosing_circle_radius cs_center :=
sorry

end NUMINAMATH_CALUDE_largest_area_cross_section_passes_through_center_exists_larger_radius_non_center_cross_section_l3087_308795


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3087_308709

/-- An isosceles triangle with altitude 10 to its base and perimeter 40 has area 75 -/
theorem isosceles_triangle_area (b s : ℝ) : 
  b > 0 → s > 0 →  -- positive base and side lengths
  2 * s + 2 * b = 40 →  -- perimeter condition
  s^2 = b^2 + 100 →  -- Pythagorean theorem with altitude 10
  b * 10 = 75 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3087_308709


namespace NUMINAMATH_CALUDE_billie_baking_days_l3087_308789

/-- The number of days Billie bakes pumpkin pies -/
def days_baking : ℕ := 11

/-- The number of pies Billie bakes per day -/
def pies_per_day : ℕ := 3

/-- The number of cans of whipped cream needed to cover one pie -/
def cans_per_pie : ℕ := 2

/-- The number of pies eaten -/
def pies_eaten : ℕ := 4

/-- The number of cans of whipped cream needed for the remaining pies -/
def cans_needed : ℕ := 58

theorem billie_baking_days :
  days_baking * pies_per_day - pies_eaten = cans_needed / cans_per_pie := by sorry

end NUMINAMATH_CALUDE_billie_baking_days_l3087_308789


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3087_308775

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a > 1 → a^2 > 1) ∧ 
  (∃ a, a^2 > 1 ∧ ¬(a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3087_308775


namespace NUMINAMATH_CALUDE_linear_function_relationship_l3087_308753

/-- A linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

theorem linear_function_relationship (y₁ y₂ : ℝ) 
  (h1 : f (-3) = y₁) 
  (h2 : f 4 = y₂) : 
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_relationship_l3087_308753


namespace NUMINAMATH_CALUDE_larger_number_problem_l3087_308702

theorem larger_number_problem (x y : ℚ) : 
  (5 * y = 6 * x) → 
  (x + y = 42) → 
  (y > x) →
  y = 252 / 11 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3087_308702


namespace NUMINAMATH_CALUDE_ryan_marbles_count_l3087_308701

/-- The number of friends Ryan shares his marbles with -/
def num_friends : ℕ := 9

/-- The number of marbles each friend receives -/
def marbles_per_friend : ℕ := 8

/-- Ryan's total number of marbles -/
def total_marbles : ℕ := num_friends * marbles_per_friend

theorem ryan_marbles_count : total_marbles = 72 := by
  sorry

end NUMINAMATH_CALUDE_ryan_marbles_count_l3087_308701


namespace NUMINAMATH_CALUDE_ab_value_l3087_308785

theorem ab_value (a b : ℝ) (h : |a + 3| + (b - 2)^2 = 0) : a^b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3087_308785


namespace NUMINAMATH_CALUDE_total_door_replacement_cost_l3087_308720

/-- The total cost of replacing doors for John -/
theorem total_door_replacement_cost :
  let num_bedroom_doors : ℕ := 3
  let num_outside_doors : ℕ := 2
  let outside_door_cost : ℕ := 20
  let bedroom_door_cost : ℕ := outside_door_cost / 2
  let total_cost : ℕ := num_bedroom_doors * bedroom_door_cost + num_outside_doors * outside_door_cost
  total_cost = 70 := by sorry

end NUMINAMATH_CALUDE_total_door_replacement_cost_l3087_308720


namespace NUMINAMATH_CALUDE_sampling_is_systematic_l3087_308700

/-- Represents a student ID number -/
structure StudentID where
  lastThreeDigits : Nat
  inv_range : 1 ≤ lastThreeDigits ∧ lastThreeDigits ≤ 818

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | Stratified
  | Systematic
  | RandomNumberTable

/-- Represents the selection criteria for inspection -/
def isSelected (id : StudentID) : Bool :=
  id.lastThreeDigits % 100 = 16

/-- Theorem stating that the sampling method is systematic -/
theorem sampling_is_systematic (ids : List StudentID) 
  (h1 : ∀ id ∈ ids, 1 ≤ id.lastThreeDigits ∧ id.lastThreeDigits ≤ 818) 
  (h2 : ∀ id ∈ ids, isSelected id ↔ id.lastThreeDigits % 100 = 16) : 
  SamplingMethod.Systematic = SamplingMethod.Systematic := by
  sorry

end NUMINAMATH_CALUDE_sampling_is_systematic_l3087_308700


namespace NUMINAMATH_CALUDE_train_length_calculation_l3087_308703

-- Define the given parameters
def train_speed : ℝ := 60  -- km/h
def man_speed : ℝ := 6     -- km/h
def passing_time : ℝ := 12 -- seconds

-- Define the theorem
theorem train_length_calculation :
  let relative_speed : ℝ := train_speed + man_speed
  let relative_speed_mps : ℝ := relative_speed * (5 / 18)
  let train_length : ℝ := relative_speed_mps * passing_time
  train_length = 220 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3087_308703


namespace NUMINAMATH_CALUDE_age_inconsistency_l3087_308784

theorem age_inconsistency (a b c d : ℝ) : 
  (a + c + d) / 3 = 30 →
  (a + c) / 2 = 32 →
  (b + d) / 2 = 34 →
  ¬(0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :=
by
  sorry

#check age_inconsistency

end NUMINAMATH_CALUDE_age_inconsistency_l3087_308784


namespace NUMINAMATH_CALUDE_only_solution_is_two_l3087_308768

theorem only_solution_is_two : 
  ∀ n : ℕ, n > 0 → ((n + 1) ∣ (2 * n^2 + 5 * n)) ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_only_solution_is_two_l3087_308768


namespace NUMINAMATH_CALUDE_markus_to_son_age_ratio_l3087_308747

/-- Represents the ages of Markus, his son, and his grandson. -/
structure FamilyAges where
  markus : ℕ
  son : ℕ
  grandson : ℕ

/-- The conditions given in the problem. -/
def problemConditions (ages : FamilyAges) : Prop :=
  ages.grandson = 20 ∧
  ages.son = 2 * ages.grandson ∧
  ages.markus + ages.son + ages.grandson = 140

/-- The theorem stating that under the given conditions, 
    the ratio of Markus's age to his son's age is 2:1. -/
theorem markus_to_son_age_ratio 
  (ages : FamilyAges) 
  (h : problemConditions ages) : 
  ages.markus * 1 = ages.son * 2 := by
  sorry

#check markus_to_son_age_ratio

end NUMINAMATH_CALUDE_markus_to_son_age_ratio_l3087_308747


namespace NUMINAMATH_CALUDE_apex_angle_of_identical_cones_l3087_308752

/-- The apex angle of a cone is the angle between its generatrices in the axial section. -/
def apex_angle (cone : Type) : ℝ := sorry

/-- A cone with apex at point A -/
structure Cone (A : Type) where
  apex : A
  angle : ℝ

/-- Three cones touch each other externally -/
def touch_externally (c1 c2 c3 : Cone A) : Prop := sorry

/-- A cone touches another cone internally -/
def touch_internally (c1 c2 : Cone A) : Prop := sorry

theorem apex_angle_of_identical_cones 
  (A : Type) 
  (c1 c2 c3 c4 : Cone A) 
  (h1 : touch_externally c1 c2 c3)
  (h2 : c1.angle = c2.angle)
  (h3 : c3.angle = π / 3)
  (h4 : touch_internally c1 c4)
  (h5 : touch_internally c2 c4)
  (h6 : touch_internally c3 c4)
  (h7 : c4.angle = 5 * π / 6) :
  c1.angle = 2 * Real.arctan (Real.sqrt 3 - 1) := by sorry

end NUMINAMATH_CALUDE_apex_angle_of_identical_cones_l3087_308752


namespace NUMINAMATH_CALUDE_advanced_math_group_arrangements_l3087_308762

/-- The number of students in the advanced mathematics study group -/
def total_students : ℕ := 5

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- Student A -/
def student_A : ℕ := 1

/-- Student B -/
def student_B : ℕ := 2

/-- The number of arrangements where A and B must stand next to each other -/
def arrangements_adjacent : ℕ := 48

/-- The number of arrangements where A and B must not stand next to each other -/
def arrangements_not_adjacent : ℕ := 72

/-- The number of arrangements where A cannot stand at the far left and B cannot stand at the far right -/
def arrangements_restricted : ℕ := 78

theorem advanced_math_group_arrangements :
  (total_students = num_boys + num_girls) ∧
  (arrangements_adjacent = 48) ∧
  (arrangements_not_adjacent = 72) ∧
  (arrangements_restricted = 78) := by
  sorry

end NUMINAMATH_CALUDE_advanced_math_group_arrangements_l3087_308762


namespace NUMINAMATH_CALUDE_passengers_from_other_continents_l3087_308766

theorem passengers_from_other_continents : 
  ∀ (total : ℕ) (north_america europe africa asia other : ℚ),
    total = 96 →
    north_america = 1/4 →
    europe = 1/8 →
    africa = 1/12 →
    asia = 1/6 →
    other = 1 - (north_america + europe + africa + asia) →
    (other * total : ℚ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_passengers_from_other_continents_l3087_308766


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3087_308759

theorem intersection_of_lines :
  let x : ℚ := 77 / 32
  let y : ℚ := 57 / 20
  (8 * x - 5 * y = 10) ∧ (9 * x + y^2 = 25) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3087_308759


namespace NUMINAMATH_CALUDE_max_value_complex_fraction_l3087_308742

theorem max_value_complex_fraction (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs ((Complex.I * Real.sqrt 3 - z) / (Real.sqrt 2 - z)) ≤ Real.sqrt 7 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_fraction_l3087_308742


namespace NUMINAMATH_CALUDE_indeterminate_equation_solutions_l3087_308767

theorem indeterminate_equation_solutions :
  ∀ x y : ℤ, 2 * (x + y) = x * y + 7 ↔ 
    (x = 3 ∧ y = -1) ∨ (x = 5 ∧ y = 1) ∨ (x = 1 ∧ y = 5) ∨ (x = -1 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_indeterminate_equation_solutions_l3087_308767


namespace NUMINAMATH_CALUDE_school_capacity_l3087_308797

theorem school_capacity (total_capacity : ℕ) (known_school_capacity : ℕ) (num_schools : ℕ) (num_known_schools : ℕ) :
  total_capacity = 1480 →
  known_school_capacity = 400 →
  num_schools = 4 →
  num_known_schools = 2 →
  (total_capacity - num_known_schools * known_school_capacity) / (num_schools - num_known_schools) = 340 := by
  sorry

end NUMINAMATH_CALUDE_school_capacity_l3087_308797


namespace NUMINAMATH_CALUDE_lemonade_amount_l3087_308771

/-- Represents the components of lemonade -/
structure LemonadeComponents where
  water : ℝ
  syrup : ℝ
  lemon_juice : ℝ

/-- Calculates the total amount of lemonade -/
def total_lemonade (c : LemonadeComponents) : ℝ :=
  c.water + c.syrup + c.lemon_juice

/-- Theorem stating the amount of lemonade made given the conditions -/
theorem lemonade_amount (c : LemonadeComponents) 
  (h1 : c.water = 4 * c.syrup) 
  (h2 : c.syrup = 2 * c.lemon_juice)
  (h3 : c.lemon_juice = 3) : 
  total_lemonade c = 24 := by
  sorry

#check lemonade_amount

end NUMINAMATH_CALUDE_lemonade_amount_l3087_308771


namespace NUMINAMATH_CALUDE_gcd_sequence_a_odd_l3087_308765

def sequence_a (a₁ : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => (sequence_a a₁ n)^2 - (sequence_a a₁ n) - 1

theorem gcd_sequence_a_odd (a₁ : ℤ) (n : ℕ) :
  Nat.gcd (Int.natAbs (sequence_a a₁ (n + 1))) (2 * (n + 1) + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sequence_a_odd_l3087_308765


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3087_308730

theorem quadratic_inequality (z : ℝ) : z^2 - 40*z + 360 ≤ 16*z ↔ 8 ≤ z ∧ z ≤ 45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3087_308730


namespace NUMINAMATH_CALUDE_unique_base_for_special_palindrome_l3087_308761

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digits : List ℕ), digits.length > 0 ∧ 
    (digits.reverse = digits) ∧ 
    (n = digits.foldl (λ acc d => acc * base + d) 0)

theorem unique_base_for_special_palindrome : 
  ∃! (r : ℕ), 
    r % 2 = 0 ∧ 
    r ≥ 18 ∧ 
    (∃ (x : ℕ), 
      x = 5 * r^3 + 5 * r^2 + 5 * r + 5 ∧
      is_palindrome (x^2) r ∧
      (∃ (a b c d : ℕ), 
        x^2 = a * r^7 + b * r^6 + c * r^5 + d * r^4 + 
              d * r^3 + c * r^2 + b * r + a ∧
        d - c = 2)) ∧
    r = 24 :=
sorry

end NUMINAMATH_CALUDE_unique_base_for_special_palindrome_l3087_308761


namespace NUMINAMATH_CALUDE_value_of_a_l3087_308787

theorem value_of_a (a b : ℝ) (h1 : |a| = 5) (h2 : b = 4) (h3 : a < b) : a = -5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3087_308787


namespace NUMINAMATH_CALUDE_program_count_l3087_308729

/-- The total number of courses available --/
def total_courses : ℕ := 7

/-- The number of courses in a program --/
def program_size : ℕ := 5

/-- The number of math courses available --/
def math_courses : ℕ := 2

/-- The number of non-math courses available (excluding English) --/
def non_math_courses : ℕ := total_courses - math_courses - 1

/-- The minimum number of math courses required in a program --/
def min_math_courses : ℕ := 2

/-- Calculates the number of ways to choose a program --/
def calculate_programs : ℕ :=
  Nat.choose non_math_courses (program_size - min_math_courses - 1) +
  Nat.choose non_math_courses (program_size - math_courses - 1)

theorem program_count : calculate_programs = 6 := by sorry

end NUMINAMATH_CALUDE_program_count_l3087_308729


namespace NUMINAMATH_CALUDE_height_ratio_of_isosceles_triangles_l3087_308774

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base_angle : ℝ
  side_length : ℝ
  base_length : ℝ
  height : ℝ

/-- The problem statement -/
theorem height_ratio_of_isosceles_triangles
  (triangle_A triangle_B : IsoscelesTriangle)
  (h_vertical_angle : 180 - 2 * triangle_A.base_angle = 180 - 2 * triangle_B.base_angle)
  (h_base_angle_A : triangle_A.base_angle = 40)
  (h_base_angle_B : triangle_B.base_angle = 50)
  (h_side_ratio : triangle_B.side_length / triangle_A.side_length = 5 / 3)
  (h_area_ratio : (triangle_B.base_length * triangle_B.height) / (triangle_A.base_length * triangle_A.height) = 25 / 9) :
  triangle_B.height / triangle_A.height = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_height_ratio_of_isosceles_triangles_l3087_308774


namespace NUMINAMATH_CALUDE_max_value_implies_m_l3087_308744

open Real

theorem max_value_implies_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = sin (x + π/2) + cos (x - π/2) + m) →
  (∃ x₀, ∀ x, f x ≤ f x₀) →
  (∃ x₁, f x₁ = 2 * sqrt 2) →
  m = sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l3087_308744


namespace NUMINAMATH_CALUDE_magical_green_knights_fraction_l3087_308760

theorem magical_green_knights_fraction (total : ℕ) (total_pos : 0 < total) :
  let green := total / 3
  let yellow := total - green
  let magical := total / 5
  let green_magical_fraction := magical_green / green
  let yellow_magical_fraction := magical_yellow / yellow
  green_magical_fraction = 3 * yellow_magical_fraction →
  magical_green + magical_yellow = magical →
  green_magical_fraction = 9 / 25 :=
by sorry

end NUMINAMATH_CALUDE_magical_green_knights_fraction_l3087_308760


namespace NUMINAMATH_CALUDE_beta_value_l3087_308773

open Real

def operation (a b c d : ℝ) : ℝ := a * d - b * c

theorem beta_value (α β : ℝ) : 
  cos α = 1/7 →
  operation (sin α) (sin β) (cos α) (cos β) = 3 * Real.sqrt 3 / 14 →
  0 < β →
  β < α →
  α < π/2 →
  β = π/3 := by sorry

end NUMINAMATH_CALUDE_beta_value_l3087_308773


namespace NUMINAMATH_CALUDE_ice_cream_sales_for_video_games_l3087_308776

theorem ice_cream_sales_for_video_games :
  let game_cost : ℕ := 60
  let ice_cream_price : ℕ := 5
  let num_games : ℕ := 2
  let total_cost : ℕ := game_cost * num_games
  let ice_creams_needed : ℕ := total_cost / ice_cream_price
  ice_creams_needed = 24 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_sales_for_video_games_l3087_308776


namespace NUMINAMATH_CALUDE_expression_evaluation_l3087_308708

theorem expression_evaluation : (-1)^2 + (1/2 - 7/12 + 5/6) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3087_308708


namespace NUMINAMATH_CALUDE_geometry_relations_l3087_308735

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (a b : Line) (α β : Plane) 
  (h_different_lines : a ≠ b) 
  (h_different_planes : α ≠ β) :
  (parallel_lines a b ∧ parallel_line_plane a α → parallel_line_plane b α) ∧
  (perpendicular_planes α β ∧ parallel_line_plane a α → perpendicular_line_plane a β) ∧
  (perpendicular_planes α β ∧ perpendicular_line_plane a β → parallel_line_plane a α) ∧
  (perpendicular_lines a b ∧ perpendicular_line_plane a α ∧ perpendicular_line_plane b β → perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_relations_l3087_308735


namespace NUMINAMATH_CALUDE_A_intersection_B_equals_A_l3087_308764

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define set A
def A : Set ℝ := {x | f x = x}

-- Define set B
def B : Set ℝ := {x | f (f x) = x}

-- Theorem statement
theorem A_intersection_B_equals_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_A_intersection_B_equals_A_l3087_308764


namespace NUMINAMATH_CALUDE_power_minus_self_even_l3087_308749

theorem power_minus_self_even (a n : ℕ+) : 
  ∃ k : ℤ, (a^n.val - a : ℤ) = 2 * k := by sorry

end NUMINAMATH_CALUDE_power_minus_self_even_l3087_308749


namespace NUMINAMATH_CALUDE_no_solution_implies_a_range_l3087_308707

theorem no_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ∈ Set.Iic 8 :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_range_l3087_308707


namespace NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l3087_308717

theorem no_bounded_function_satisfying_inequality :
  ¬∃ (f : ℝ → ℝ), 
    (∃ (M : ℝ), ∀ x, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y, (f (x + y))^2 ≥ (f x)^2 + 2*(f (x*y)) + (f y)^2) := by
  sorry

end NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l3087_308717


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l3087_308743

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l3087_308743


namespace NUMINAMATH_CALUDE_house_cleaning_time_l3087_308779

theorem house_cleaning_time (sawyer_time nick_time joint_time : ℝ) 
  (h1 : sawyer_time / 2 = nick_time / 3)
  (h2 : joint_time = 3.6)
  (h3 : 1 / sawyer_time + 1 / nick_time = 1 / joint_time) :
  sawyer_time = 6 := by
sorry

end NUMINAMATH_CALUDE_house_cleaning_time_l3087_308779


namespace NUMINAMATH_CALUDE_function_expression_l3087_308778

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_expression 
  (ω : ℝ) 
  (φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 < φ ∧ φ < π/2) 
  (h_symmetry : f (1/3) ω φ = 0) 
  (h_amplitude : ∃ (x y : ℝ), f x ω φ - f y ω φ = 4) :
  ∀ x, f x ω φ = Real.sqrt 3 * Real.sin (π/2 * x - π/6) :=
sorry

end NUMINAMATH_CALUDE_function_expression_l3087_308778


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3087_308741

/-- Given two vectors a and b in ℝ², prove that |a - b| = 5 -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  a = (-2, 1) →
  a + b = (-1, -2) →
  ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3087_308741


namespace NUMINAMATH_CALUDE_approximation_of_2026_l3087_308748

def approximate_to_hundredth (x : ℚ) : ℚ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem approximation_of_2026 :
  approximate_to_hundredth 2.026 = 2.03 := by
  sorry

end NUMINAMATH_CALUDE_approximation_of_2026_l3087_308748


namespace NUMINAMATH_CALUDE_geometric_sequence_unique_solution_l3087_308727

/-- A geometric sequence is defined by its first term and common ratio. -/
structure GeometricSequence where
  first_term : ℚ
  common_ratio : ℚ

/-- Get the nth term of a geometric sequence. -/
def GeometricSequence.nth_term (seq : GeometricSequence) (n : ℕ) : ℚ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem geometric_sequence_unique_solution :
  ∃! (seq : GeometricSequence),
    seq.nth_term 2 = 37 + 1/3 ∧
    seq.nth_term 6 = 2 + 1/3 ∧
    seq.first_term = 74 + 2/3 ∧
    seq.common_ratio = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_unique_solution_l3087_308727


namespace NUMINAMATH_CALUDE_sine_cosine_sum_equals_one_l3087_308796

theorem sine_cosine_sum_equals_one :
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_equals_one_l3087_308796


namespace NUMINAMATH_CALUDE_stating_isosceles_triangles_properties_l3087_308736

/-- 
Represents the number of isosceles triangles with vertices of the same color 
in a regular (6n+1)-gon with k red vertices and the rest blue.
-/
def P (n : ℕ) (k : ℕ) : ℕ := sorry

/-- 
Theorem stating the properties of P for a regular (6n+1)-gon 
with k red vertices and the rest blue.
-/
theorem isosceles_triangles_properties (n : ℕ) (k : ℕ) : 
  (P n (k + 1) - P n k = 3 * k - 9 * n) ∧ 
  (P n k = 3 * n * (6 * n + 1) - 9 * k * n + (3 * k * (k - 1)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_stating_isosceles_triangles_properties_l3087_308736


namespace NUMINAMATH_CALUDE_percentage_problem_l3087_308783

theorem percentage_problem (P : ℝ) : 
  (0.15 * (P / 100) * 0.5 * 5200 = 117) → P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3087_308783


namespace NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l3087_308770

theorem deal_or_no_deal_elimination (total_boxes : ℕ) (high_value_boxes : ℕ) 
  (elimination_target : ℚ) :
  total_boxes = 30 →
  high_value_boxes = 9 →
  elimination_target = 1/3 →
  ∃ (boxes_to_eliminate : ℕ),
    boxes_to_eliminate = 3 ∧
    (total_boxes - boxes_to_eliminate : ℚ) * elimination_target ≤ high_value_boxes ∧
    ∀ (n : ℕ), n < boxes_to_eliminate →
      (total_boxes - n : ℚ) * elimination_target > high_value_boxes :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l3087_308770


namespace NUMINAMATH_CALUDE_mango_rate_proof_l3087_308754

def grape_quantity : ℕ := 10
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def total_paid : ℕ := 1195

theorem mango_rate_proof :
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_proof_l3087_308754


namespace NUMINAMATH_CALUDE_set_M_equals_three_two_four_three_one_l3087_308732

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | (m - 1)*x - 1 = 0}

-- Define the set M
def M : Set ℝ := {m : ℝ | A ∩ B m = B m}

-- Theorem statement
theorem set_M_equals_three_two_four_three_one : M = {3/2, 4/3, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_M_equals_three_two_four_three_one_l3087_308732


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3087_308716

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ∃ (x y : ℝ),
  x ≥ 0 ∧ 
  y = Real.sqrt x ∧ 
  x^2 / a^2 - y^2 / b^2 = 1 ∧
  (∃ (m : ℝ), m * (x + 1) = y ∧ m = 1 / (2 * Real.sqrt x)) →
  (Real.sqrt (a^2 + b^2)) / a = (Real.sqrt 5 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3087_308716


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l3087_308792

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (x + 3) * (x - 1) = 2 * x - 4 ↔ x^2 + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l3087_308792


namespace NUMINAMATH_CALUDE_correlation_relationships_l3087_308722

-- Define the type for relationships
inductive Relationship
  | AppleProductionClimate
  | StudentID
  | TreeDiameterHeight
  | PointCoordinates

-- Define a predicate for correlation relationships
def IsCorrelation (r : Relationship) : Prop :=
  match r with
  | Relationship.AppleProductionClimate => true
  | Relationship.TreeDiameterHeight => true
  | _ => false

-- Theorem statement
theorem correlation_relationships :
  (∀ r : Relationship, IsCorrelation r ↔ 
    (r = Relationship.AppleProductionClimate ∨ 
     r = Relationship.TreeDiameterHeight)) := by
  sorry

end NUMINAMATH_CALUDE_correlation_relationships_l3087_308722


namespace NUMINAMATH_CALUDE_median_and_altitude_lengths_l3087_308728

/-- Right triangle DEF with given side lengths and midpoint N -/
structure RightTriangleDEF where
  DE : ℝ
  DF : ℝ
  N : ℝ × ℝ
  is_right_angle : DE^2 + DF^2 = (DE + DF)^2 / 2
  side_lengths : DE = 6 ∧ DF = 8
  N_is_midpoint : N = ((DE + DF) / 2, DF / 2)

/-- Theorem about median and altitude lengths in the right triangle -/
theorem median_and_altitude_lengths (t : RightTriangleDEF) :
  let DN := Real.sqrt ((t.DE + t.N.1)^2 + t.N.2^2)
  let altitude := 2 * (t.DE * t.DF) / (t.DE + t.DF)
  DN = 5 ∧ altitude = 4.8 := by
  sorry


end NUMINAMATH_CALUDE_median_and_altitude_lengths_l3087_308728


namespace NUMINAMATH_CALUDE_solution_exists_for_a_in_range_l3087_308777

/-- The system of equations has a solution for a given 'a' -/
def has_solution (a : ℝ) : Prop :=
  ∃ b x y : ℝ, x^2 + y^2 + 2*a*(a + y - x) = 49 ∧ y = 8 / ((x - b)^2 + 1)

/-- The theorem stating the range of 'a' for which the system has a solution -/
theorem solution_exists_for_a_in_range :
  ∀ a : ℝ, -15 ≤ a ∧ a < 7 → has_solution a :=
sorry

end NUMINAMATH_CALUDE_solution_exists_for_a_in_range_l3087_308777


namespace NUMINAMATH_CALUDE_davids_math_marks_l3087_308721

def english_marks : ℝ := 74
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 67
def biology_marks : ℝ := 90
def average_marks : ℝ := 75.6
def num_subjects : ℕ := 5

theorem davids_math_marks :
  let total_marks := average_marks * num_subjects
  let known_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  let math_marks := total_marks - known_marks
  math_marks = 65 := by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l3087_308721


namespace NUMINAMATH_CALUDE_incorrect_statement_l3087_308788

theorem incorrect_statement : ¬ (∀ x : ℝ, |x| = x ↔ x = 0 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l3087_308788


namespace NUMINAMATH_CALUDE_least_sum_m_n_l3087_308725

theorem least_sum_m_n (m n : ℕ+) (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^(m : ℕ) = k * n^(n : ℕ)) (h3 : ¬ ∃ k : ℕ, m = k * n) :
  ∀ p q : ℕ+, 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ k : ℕ, p^(p : ℕ) = k * q^(q : ℕ)) → 
    (¬ ∃ k : ℕ, p = k * q) → 
    (m + n ≤ p + q) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l3087_308725


namespace NUMINAMATH_CALUDE_f_properties_l3087_308763

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

theorem f_properties (a : ℝ) :
  (∀ x, a ≤ 0 → (deriv (f a)) x < 0) ∧
  (a > 0 → ∀ x, x < -Real.log a → (deriv (f a)) x < 0) ∧
  (a > 0 → ∀ x, x > -Real.log a → (deriv (f a)) x > 0) ∧
  (a ≥ 1 → ∀ x, f a x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3087_308763


namespace NUMINAMATH_CALUDE_urn_problem_l3087_308710

theorem urn_problem (M : ℕ) : 
  (5 / 12 : ℚ) * (10 / (10 + M)) + (7 / 12 : ℚ) * (M / (10 + M)) = 62 / 100 → M = 7 :=
by sorry

end NUMINAMATH_CALUDE_urn_problem_l3087_308710


namespace NUMINAMATH_CALUDE_class_sports_census_suitable_l3087_308750

/-- Represents a survey --/
inductive Survey
  | LightBulbLifespan
  | ClassSportsActivity
  | YangtzeRiverFish
  | PlasticBagDisposal

/-- Represents the characteristics of a survey --/
structure SurveyCharacteristics where
  population_size : ℕ
  data_collection_time : ℕ
  resource_intensity : ℕ

/-- Determines if a survey is feasible and practical for a census --/
def is_census_suitable (s : Survey) (c : SurveyCharacteristics) : Prop :=
  c.population_size ≤ 1000 ∧ c.data_collection_time ≤ 7 ∧ c.resource_intensity ≤ 5

/-- The characteristics of the class sports activity survey --/
def class_sports_characteristics : SurveyCharacteristics :=
  { population_size := 30
  , data_collection_time := 1
  , resource_intensity := 2 }

/-- Theorem stating that the class sports activity survey is suitable for a census --/
theorem class_sports_census_suitable :
  is_census_suitable Survey.ClassSportsActivity class_sports_characteristics :=
sorry

end NUMINAMATH_CALUDE_class_sports_census_suitable_l3087_308750


namespace NUMINAMATH_CALUDE_die_roll_probability_l3087_308739

def roll_outcome := Fin 6

def is_valid_outcome (m n : roll_outcome) : Prop :=
  m.val + 1 = 2 * (n.val + 1)

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 3

theorem die_roll_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_die_roll_probability_l3087_308739


namespace NUMINAMATH_CALUDE_sparrow_swallow_system_l3087_308738

/-- Represents the weight of a sparrow in taels -/
def sparrow_weight : ℝ := sorry

/-- Represents the weight of a swallow in taels -/
def swallow_weight : ℝ := sorry

/-- The total weight of five sparrows and six swallows is 16 taels -/
axiom total_weight : 5 * sparrow_weight + 6 * swallow_weight = 16

/-- Exchanging one sparrow with one swallow results in equal weights for both groups -/
axiom exchange_equal : 4 * sparrow_weight + swallow_weight = 5 * swallow_weight + sparrow_weight

/-- The system of equations representing the sparrow and swallow weight problem -/
theorem sparrow_swallow_system :
  (5 * sparrow_weight + 6 * swallow_weight = 16) ∧
  (4 * sparrow_weight + swallow_weight = 5 * swallow_weight + sparrow_weight) :=
sorry

end NUMINAMATH_CALUDE_sparrow_swallow_system_l3087_308738


namespace NUMINAMATH_CALUDE_rectangle_square_division_l3087_308756

theorem rectangle_square_division (n : ℕ) : 
  (∃ (a b : ℚ), a > 0 ∧ b > 0 ∧ 
    (∃ (p q : ℕ), p > q ∧ 
      (a * b / n).sqrt / (a * b / (n + 76)).sqrt = p / q)) → 
  n = 324 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_division_l3087_308756


namespace NUMINAMATH_CALUDE_pizza_combinations_l3087_308706

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  Nat.choose n 4 + Nat.choose n 3 = 126 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3087_308706


namespace NUMINAMATH_CALUDE_zoe_pool_cleaning_earnings_l3087_308714

/-- Represents Zoe's earnings and babysitting information --/
structure ZoeEarnings where
  total : ℕ
  zacharyRate : ℕ
  julieRate : ℕ
  chloeRate : ℕ
  zacharyEarnings : ℕ

/-- Calculates Zoe's earnings from pool cleaning --/
def poolCleaningEarnings (z : ZoeEarnings) : ℕ :=
  let zacharyHours := z.zacharyEarnings / z.zacharyRate
  let chloeHours := zacharyHours * 5
  let julieHours := zacharyHours * 3
  let babysittingEarnings := 
    zacharyHours * z.zacharyRate + 
    chloeHours * z.chloeRate + 
    julieHours * z.julieRate
  z.total - babysittingEarnings

/-- Theorem stating that Zoe's pool cleaning earnings are $5,200 --/
theorem zoe_pool_cleaning_earnings :
  poolCleaningEarnings {
    total := 8000,
    zacharyRate := 15,
    julieRate := 10,
    chloeRate := 5,
    zacharyEarnings := 600
  } = 5200 := by
  sorry

end NUMINAMATH_CALUDE_zoe_pool_cleaning_earnings_l3087_308714


namespace NUMINAMATH_CALUDE_geometric_ratio_is_four_l3087_308794

/-- An arithmetic sequence with a_1 = 2 and non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d

/-- Three terms of an arithmetic sequence form a geometric sequence -/
def forms_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ a 3 = a 1 * q ∧ a 11 = a 1 * q^2

/-- The common ratio of the geometric sequence formed by a_1, a_3, and a_11 is 4 -/
theorem geometric_ratio_is_four
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : forms_geometric_sequence a) :
  ∃ q : ℝ, q = 4 ∧ a 3 = a 1 * q ∧ a 11 = a 1 * q^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_ratio_is_four_l3087_308794


namespace NUMINAMATH_CALUDE_chocolate_cost_450_l3087_308772

/-- The cost of buying a specific number of chocolate candies, given the cost and quantity per box. -/
def chocolate_cost (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem: The cost of 450 chocolate candies is $120, given that a box of 30 candies costs $8. -/
theorem chocolate_cost_450 : chocolate_cost 30 8 450 = 120 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_450_l3087_308772


namespace NUMINAMATH_CALUDE_product_xyz_l3087_308782

theorem product_xyz (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) : 
  x * y * z = -1 := by sorry

end NUMINAMATH_CALUDE_product_xyz_l3087_308782


namespace NUMINAMATH_CALUDE_solution_set_m_zero_solution_set_all_reals_l3087_308798

-- Define the inequality function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 1) * x + 2

-- Part 1: Solution set when m = 0
theorem solution_set_m_zero :
  {x : ℝ | f 0 x > 0} = {x : ℝ | -2 < x ∧ x < 1} :=
sorry

-- Part 2: Range of m when solution set is ℝ
theorem solution_set_all_reals (m : ℝ) :
  ({x : ℝ | f m x > 0} = Set.univ) ↔ (1 < m ∧ m < 9) :=
sorry

end NUMINAMATH_CALUDE_solution_set_m_zero_solution_set_all_reals_l3087_308798


namespace NUMINAMATH_CALUDE_boxes_per_hand_for_seven_people_l3087_308705

/-- Given a group of people and the total number of boxes they can hold, 
    calculate the number of boxes one person can hold in each hand. -/
def boxes_per_hand (num_people : ℕ) (total_boxes : ℕ) : ℕ :=
  (total_boxes / num_people) / 2

/-- Theorem stating that given 7 people holding 14 boxes in total, 
    each person can hold 1 box in each hand. -/
theorem boxes_per_hand_for_seven_people : 
  boxes_per_hand 7 14 = 1 := by sorry

end NUMINAMATH_CALUDE_boxes_per_hand_for_seven_people_l3087_308705


namespace NUMINAMATH_CALUDE_emily_oranges_l3087_308734

theorem emily_oranges (betty_oranges sandra_oranges emily_oranges : ℕ) : 
  betty_oranges = 12 →
  sandra_oranges = 3 * betty_oranges →
  emily_oranges = 7 * sandra_oranges →
  emily_oranges = 252 := by
sorry

end NUMINAMATH_CALUDE_emily_oranges_l3087_308734


namespace NUMINAMATH_CALUDE_sector_area_for_unit_radian_l3087_308746

theorem sector_area_for_unit_radian (arc_length : Real) (h : arc_length = 6) :
  let radius := arc_length  -- From definition of radian: 1 = arc_length / radius
  let sector_area := (1 / 2) * radius * arc_length
  sector_area = 18 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_for_unit_radian_l3087_308746


namespace NUMINAMATH_CALUDE_binomial_unique_parameters_l3087_308755

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_unique_parameters :
  ∀ ξ : BinomialRV, expectation ξ = 12 → variance ξ = 2.4 → ξ.n = 15 ∧ ξ.p = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_unique_parameters_l3087_308755


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3087_308769

theorem complex_equation_solution (z : ℂ) : z * (2 + I) = 1 + 3 * I → z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3087_308769


namespace NUMINAMATH_CALUDE_pen_and_notebook_cost_l3087_308715

theorem pen_and_notebook_cost (pen_cost : ℝ) (price_difference : ℝ) : 
  pen_cost = 4.5 → price_difference = 1.8 → pen_cost + (pen_cost - price_difference) = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_pen_and_notebook_cost_l3087_308715


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l3087_308737

/-- The number of ropes after n cuts -/
def num_ropes (n : ℕ) : ℕ := 1 + 4 * n

/-- The problem statement -/
theorem rope_cutting_problem :
  ∃ n : ℕ, num_ropes n = 2021 ∧ n = 505 := by sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l3087_308737


namespace NUMINAMATH_CALUDE_equation_solution_l3087_308713

def f (x : ℝ) : ℝ := 2 * x - 3

theorem equation_solution :
  let d : ℝ := 4
  ∃ x : ℝ, 2 * (f x) - 21 = f (x - d) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3087_308713


namespace NUMINAMATH_CALUDE_cube_root_of_negative_64_l3087_308733

theorem cube_root_of_negative_64 : ∃ b : ℝ, b^3 = -64 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_64_l3087_308733


namespace NUMINAMATH_CALUDE_sum_of_y_coefficients_l3087_308712

-- Define the polynomials
def p (x y : ℝ) := 5*x + 3*y - 2
def q (x y : ℝ) := 2*x + 5*y + 7

-- Define the expanded product
def expanded_product (x y : ℝ) := p x y * q x y

-- Define a function to extract coefficients of terms with y
def y_coefficients (x y : ℝ) : List ℝ := 
  [31, 15, 11]  -- Coefficients of xy, y², and y respectively

-- Theorem statement
theorem sum_of_y_coefficients :
  (y_coefficients 0 0).sum = 57 :=
sorry

end NUMINAMATH_CALUDE_sum_of_y_coefficients_l3087_308712


namespace NUMINAMATH_CALUDE_mary_shirts_fraction_l3087_308718

theorem mary_shirts_fraction (blue_initial : ℕ) (brown_initial : ℕ) (total_left : ℕ) :
  blue_initial = 26 →
  brown_initial = 36 →
  total_left = 37 →
  ∃ (f : ℚ), 
    (blue_initial / 2 + brown_initial * (1 - f) = total_left) ∧
    (f = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_mary_shirts_fraction_l3087_308718


namespace NUMINAMATH_CALUDE_unique_A_for_club_equation_l3087_308791

-- Define the ♣ operation
def club (A B : ℝ) : ℝ := 4 * A + 2 * B + 6

-- Theorem statement
theorem unique_A_for_club_equation : ∃! A : ℝ, club A 6 = 70 ∧ A = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_A_for_club_equation_l3087_308791


namespace NUMINAMATH_CALUDE_article_cost_proof_l3087_308780

theorem article_cost_proof (sp1 sp2 : ℝ) (gain_percentage : ℝ) :
  sp1 = 348 ∧ sp2 = 350 ∧ gain_percentage = 0.05 →
  ∃ (cost gain : ℝ),
    sp1 = cost + gain ∧
    sp2 = cost + gain + gain_percentage * gain ∧
    cost = 308 :=
by sorry

end NUMINAMATH_CALUDE_article_cost_proof_l3087_308780


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3087_308724

/-- Given the equation 3x - y = 9, prove that y can be expressed as 3x - 9 -/
theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x - y = 9) : y = 3 * x - 9 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3087_308724


namespace NUMINAMATH_CALUDE_no_isosceles_triangle_36_degree_l3087_308757

theorem no_isosceles_triangle_36_degree (a b : ℕ+) : ¬ ∃ θ : ℝ,
  θ = 36 * π / 180 ∧
  (a : ℝ) * ((5 : ℝ).sqrt - 1) / 2 = b :=
sorry

end NUMINAMATH_CALUDE_no_isosceles_triangle_36_degree_l3087_308757


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a1_l3087_308758

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) ^ 2 = a n * a (n + 2)

theorem arithmetic_geometric_sequence_a1 (a : ℕ → ℚ) 
  (h_seq : ArithmeticGeometricSequence a) 
  (h_sum : a 1 + a 6 = 11) 
  (h_prod : a 3 * a 4 = 32 / 9) : 
  a 1 = 32 / 3 ∨ a 1 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a1_l3087_308758


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l3087_308781

theorem linear_function_decreasing (k : ℝ) :
  (∀ x y : ℝ, x < y → ((k + 2) * x + 1) > ((k + 2) * y + 1)) ↔ k < -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l3087_308781


namespace NUMINAMATH_CALUDE_square_side_length_from_hexagons_l3087_308723

/-- The side length of a square formed by repositioning two congruent hexagons cut from a rectangle -/
def square_side_length (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  sorry

/-- The height of each hexagon cut from the rectangle -/
def hexagon_height (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  sorry

theorem square_side_length_from_hexagons
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (h1 : rectangle_width = 9)
  (h2 : rectangle_height = 27)
  (h3 : square_side_length rectangle_width rectangle_height =
        2 * hexagon_height rectangle_width rectangle_height) :
  square_side_length rectangle_width rectangle_height = 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_from_hexagons_l3087_308723


namespace NUMINAMATH_CALUDE_number_problem_l3087_308799

theorem number_problem (x y : ℝ) : 
  (x^2)/2 + 5*y = 15 ∧ x + y = 10 → x = 5 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3087_308799


namespace NUMINAMATH_CALUDE_find_a_l3087_308704

def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + a) ^ 2

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, (deriv (f a)) x = 4 * (2 * x + a)) ∧ (deriv (f a)) 2 = 20 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3087_308704


namespace NUMINAMATH_CALUDE_parabola_inscribed_triangle_l3087_308731

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
def Parabola (p : ℝ) :=
  {point : Point | point.y^2 = 2 * p * point.x}

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Calculates the area of a quadrilateral -/
def quadrilateralArea (q : Quadrilateral) : ℝ :=
  sorry

/-- Main theorem -/
theorem parabola_inscribed_triangle 
  (p : ℝ) 
  (parabola : Parabola p)
  (ABC : Triangle)
  (AFBC : Quadrilateral)
  (h1 : ABC.B.y = 0) -- B is on x-axis
  (h2 : ABC.C.y = 0) -- C is on x-axis
  (h3 : ABC.A.y^2 = 2 * p * ABC.A.x) -- A is on parabola
  (h4 : (ABC.B.x - ABC.A.x) * (ABC.C.x - ABC.A.x) + (ABC.B.y - ABC.A.y) * (ABC.C.y - ABC.A.y) = 0) -- ABC is right-angled
  (h5 : quadrilateralArea AFBC = 8 * p^2) -- Area of AFBC is 8p^2
  : ∃ (D : Point), triangleArea ⟨ABC.A, ABC.C, D⟩ = 15/2 * p^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_inscribed_triangle_l3087_308731


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3087_308751

theorem coin_flip_probability (n : ℕ) : n = 6 →
  (1 + n + n * (n - 1) / 2 : ℚ) / 2^n = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3087_308751


namespace NUMINAMATH_CALUDE_race_distance_difference_l3087_308711

/-- In a race scenario where:
  * The race distance is 240 meters
  * Runner A finishes in 23 seconds
  * Runner A beats runner B by 7 seconds
This theorem proves that A beats B by 56 meters -/
theorem race_distance_difference (race_distance : ℝ) (a_time : ℝ) (time_difference : ℝ) :
  race_distance = 240 ∧ 
  a_time = 23 ∧ 
  time_difference = 7 →
  (race_distance - (race_distance / (a_time + time_difference)) * a_time) = 56 :=
by sorry

end NUMINAMATH_CALUDE_race_distance_difference_l3087_308711
