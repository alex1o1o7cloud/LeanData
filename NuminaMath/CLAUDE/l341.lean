import Mathlib

namespace NUMINAMATH_CALUDE_final_ratio_is_11_to_14_l341_34174

/-- Represents the number of students in a school --/
structure School where
  boys : ℕ
  girls : ℕ

def initial_school : School :=
  { boys := 120,
    girls := 160 }

def students_left : School :=
  { boys := 10,
    girls := 20 }

def final_school : School :=
  { boys := initial_school.boys - students_left.boys,
    girls := initial_school.girls - students_left.girls }

theorem final_ratio_is_11_to_14 :
  ∃ (k : ℕ), k > 0 ∧ final_school.boys = 11 * k ∧ final_school.girls = 14 * k :=
sorry

end NUMINAMATH_CALUDE_final_ratio_is_11_to_14_l341_34174


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l341_34100

-- Define the original number
def original_number : ℝ := 1300000

-- Define the scientific notation components
def coefficient : ℝ := 1.3
def exponent : ℕ := 6

-- Theorem statement
theorem scientific_notation_equivalence :
  original_number = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l341_34100


namespace NUMINAMATH_CALUDE_debate_team_girls_l341_34105

/-- The number of boys on the debate team -/
def num_boys : ℕ := 11

/-- The number of groups the team can be split into -/
def num_groups : ℕ := 8

/-- The number of students in each group -/
def students_per_group : ℕ := 7

/-- The total number of students on the debate team -/
def total_students : ℕ := num_groups * students_per_group

/-- The number of girls on the debate team -/
def num_girls : ℕ := total_students - num_boys

theorem debate_team_girls : num_girls = 45 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_girls_l341_34105


namespace NUMINAMATH_CALUDE_white_balls_count_l341_34167

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  green = 18 ∧
  yellow = 17 ∧
  red = 3 ∧
  purple = 1 ∧
  prob = 95 / 100 ∧
  prob = (total - (red + purple)) / total →
  total - (green + yellow + red + purple) = 21 :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l341_34167


namespace NUMINAMATH_CALUDE_sixth_term_of_sequence_l341_34177

theorem sixth_term_of_sequence (a : ℕ → ℕ) (h : ∀ n, a n = 2 * n + 1) : a 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_sequence_l341_34177


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l341_34176

/-- Represents a participant's scores in a two-day math competition -/
structure Participant where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ

/-- The maximum possible two-day success ratio for Delta given the competition conditions -/
theorem delta_max_success_ratio 
  (gamma : Participant)
  (total_points : ℕ)
  (h_total : gamma.day1_attempted + gamma.day2_attempted = total_points)
  (h_gamma_day1 : gamma.day1_score = 210 ∧ gamma.day1_attempted = 360)
  (h_gamma_day2 : gamma.day2_score = 150 ∧ gamma.day2_attempted = 240)
  (h_gamma_ratio : (gamma.day1_score + gamma.day2_score : ℚ) / total_points = 3/5) :
  ∃ (delta : Participant),
    (delta.day1_attempted + delta.day2_attempted = total_points) ∧
    (delta.day1_attempted ≠ gamma.day1_attempted) ∧
    (delta.day1_score > 0 ∧ delta.day2_score > 0) ∧
    ((delta.day1_score : ℚ) / delta.day1_attempted < (gamma.day1_score : ℚ) / gamma.day1_attempted) ∧
    ((delta.day2_score : ℚ) / delta.day2_attempted < (gamma.day2_score : ℚ) / gamma.day2_attempted) ∧
    ((delta.day1_score + delta.day2_score : ℚ) / total_points ≤ 1/4) ∧
    ∀ (delta' : Participant),
      (delta'.day1_attempted + delta'.day2_attempted = total_points) →
      (delta'.day1_attempted ≠ gamma.day1_attempted) →
      (delta'.day1_score > 0 ∧ delta'.day2_score > 0) →
      ((delta'.day1_score : ℚ) / delta'.day1_attempted < (gamma.day1_score : ℚ) / gamma.day1_attempted) →
      ((delta'.day2_score : ℚ) / delta'.day2_attempted < (gamma.day2_score : ℚ) / gamma.day2_attempted) →
      ((delta'.day1_score + delta'.day2_score : ℚ) / total_points ≤ (delta.day1_score + delta.day2_score : ℚ) / total_points) := by
  sorry


end NUMINAMATH_CALUDE_delta_max_success_ratio_l341_34176


namespace NUMINAMATH_CALUDE_second_year_probability_l341_34178

/-- Represents the academic year of a student -/
inductive AcademicYear
| FirstYear
| SecondYear
| ThirdYear
| Postgraduate

/-- Represents the department of a student -/
inductive Department
| Science
| Arts
| Engineering

/-- Represents the number of students in each academic year and department -/
def studentCount : AcademicYear → Department → ℕ
| AcademicYear.FirstYear, Department.Science => 300
| AcademicYear.FirstYear, Department.Arts => 200
| AcademicYear.FirstYear, Department.Engineering => 100
| AcademicYear.SecondYear, Department.Science => 250
| AcademicYear.SecondYear, Department.Arts => 150
| AcademicYear.SecondYear, Department.Engineering => 50
| AcademicYear.ThirdYear, Department.Science => 300
| AcademicYear.ThirdYear, Department.Arts => 200
| AcademicYear.ThirdYear, Department.Engineering => 50
| AcademicYear.Postgraduate, Department.Science => 200
| AcademicYear.Postgraduate, Department.Arts => 100
| AcademicYear.Postgraduate, Department.Engineering => 100

/-- The total number of students in the sample -/
def totalStudents : ℕ := 2000

/-- Theorem: The probability of selecting a second-year student from the group of students
    who are not third-year and not in the Science department is 2/7 -/
theorem second_year_probability :
  let nonThirdYearNonScience := (studentCount AcademicYear.FirstYear Department.Arts
                                + studentCount AcademicYear.FirstYear Department.Engineering
                                + studentCount AcademicYear.SecondYear Department.Arts
                                + studentCount AcademicYear.SecondYear Department.Engineering
                                + studentCount AcademicYear.Postgraduate Department.Arts
                                + studentCount AcademicYear.Postgraduate Department.Engineering)
  let secondYearNonScience := (studentCount AcademicYear.SecondYear Department.Arts
                              + studentCount AcademicYear.SecondYear Department.Engineering)
  (secondYearNonScience : ℚ) / nonThirdYearNonScience = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_second_year_probability_l341_34178


namespace NUMINAMATH_CALUDE_polynomial_identity_l341_34144

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l341_34144


namespace NUMINAMATH_CALUDE_not_perfect_square_l341_34108

theorem not_perfect_square (m : ℕ) : ¬ ∃ (n : ℕ), ((4 * 10^(2*m+1) + 5) / 9 : ℚ) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l341_34108


namespace NUMINAMATH_CALUDE_compound_interest_principal_l341_34130

/-- Given a sum of 5292 after 2 years with an interest rate of 5% per annum compounded yearly, 
    prove that the principal amount is 4800. -/
theorem compound_interest_principal (sum : ℝ) (years : ℕ) (rate : ℝ) (principal : ℝ) : 
  sum = 5292 →
  years = 2 →
  rate = 0.05 →
  sum = principal * (1 + rate) ^ years →
  principal = 4800 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l341_34130


namespace NUMINAMATH_CALUDE_people_per_car_l341_34152

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 3) :
  total_people / num_cars = 21 :=
by sorry

end NUMINAMATH_CALUDE_people_per_car_l341_34152


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l341_34182

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x - 1| - a

-- Theorem 1
theorem range_of_a (a : ℝ) :
  (∃ x, f a x - 2 * |x - 7| ≤ 0) → a ≥ -12 := by
  sorry

-- Theorem 2
theorem range_of_m (m : ℝ) :
  (∀ x, f 1 x + |x + 7| ≥ m) → m ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l341_34182


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l341_34122

theorem marble_fraction_after_tripling (total : ℝ) (h : total > 0) :
  let initial_green := (3/4 : ℝ) * total
  let initial_yellow := (1/4 : ℝ) * total
  let new_yellow := 3 * initial_yellow
  let new_total := initial_green + new_yellow
  new_yellow / new_total = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l341_34122


namespace NUMINAMATH_CALUDE_factorization_equality_l341_34191

theorem factorization_equality (x : ℝ) :
  (x^2 + 3*x - 3) * (x^2 + 3*x + 1) - 5 = (x + 1) * (x + 2) * (x + 4) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l341_34191


namespace NUMINAMATH_CALUDE_vector_at_t_zero_l341_34163

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point : ℝ → ℝ × ℝ × ℝ

/-- Given conditions for the parameterized line -/
def line_conditions (L : ParameterizedLine) : Prop :=
  L.point 1 = (2, 5, 7) ∧ L.point 4 = (8, -7, 1)

theorem vector_at_t_zero 
  (L : ParameterizedLine) 
  (h : line_conditions L) : 
  L.point 0 = (0, 9, 9) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_t_zero_l341_34163


namespace NUMINAMATH_CALUDE_girls_multiple_of_five_l341_34187

/-- Represents the number of students in a group -/
structure GroupComposition :=
  (boys : ℕ)
  (girls : ℕ)

/-- Checks if a given number of boys and girls can be divided into the specified number of groups -/
def canDivideIntoGroups (totalBoys totalGirls groups : ℕ) : Prop :=
  ∃ (composition : GroupComposition),
    composition.boys * groups = totalBoys ∧
    composition.girls * groups = totalGirls

theorem girls_multiple_of_five (totalBoys totalGirls : ℕ) :
  totalBoys = 10 →
  canDivideIntoGroups totalBoys totalGirls 5 →
  ∃ (k : ℕ), totalGirls = 5 * k ∧ k ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_girls_multiple_of_five_l341_34187


namespace NUMINAMATH_CALUDE_imaginary_unit_fraction_l341_34110

theorem imaginary_unit_fraction : 
  ∃ (i : ℂ), i * i = -1 ∧ (i^2019) / (1 + i) = -1/2 - 1/2 * i :=
by sorry

end NUMINAMATH_CALUDE_imaginary_unit_fraction_l341_34110


namespace NUMINAMATH_CALUDE_inequality_implication_l341_34166

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l341_34166


namespace NUMINAMATH_CALUDE_divisor_sum_implies_exponent_sum_l341_34150

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ :=
  (a * (r^(n+1) - 1)) / (r - 1)

def sum_of_divisors (i j : ℕ) : ℕ :=
  (sum_of_geometric_series 1 2 i) * (sum_of_geometric_series 1 5 j)

theorem divisor_sum_implies_exponent_sum (i j : ℕ) :
  sum_of_divisors i j = 930 → i + j = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_implies_exponent_sum_l341_34150


namespace NUMINAMATH_CALUDE_walk_distance_l341_34173

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the final position after walking in four segments -/
def finalPosition (d : ℝ) : Point :=
  { x := d + d,  -- East distance: second segment + fourth segment
    y := -d + d + d }  -- South, then North, then North again

/-- Theorem stating that if the final position is 40 meters north of the start,
    then the distance walked in each segment must be 40 meters -/
theorem walk_distance (d : ℝ) :
  (finalPosition d).y = 40 → d = 40 := by sorry

end NUMINAMATH_CALUDE_walk_distance_l341_34173


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l341_34151

/-- Given a quadratic function f(x) = ax^2 + bx + c with vertex (5, -3) and
    one x-intercept at (1, 0), the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept_of_quadratic 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 5 = -3 ∧ ∀ x, f x ≤ f 5) -- Vertex condition
  (h3 : f 1 = 0) -- Given x-intercept
  : ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 9 :=
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l341_34151


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l341_34153

theorem arctan_equation_solution (x : ℝ) : 
  x = Real.sqrt ((1 - 3 * Real.sqrt 3) / 13) ∨ 
  x = -Real.sqrt ((1 - 3 * Real.sqrt 3) / 13) → 
  Real.arctan (2 / x) + Real.arctan (1 / (2 * x^2)) = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l341_34153


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l341_34124

-- Define the sample space
def sampleSpace : ℕ := 10 -- (5 choose 2)

-- Define the events
def exactlyOneMale (outcome : ℕ) : Prop := sorry
def exactlyTwoFemales (outcome : ℕ) : Prop := sorry

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : ℕ → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

-- Define complementary events
def complementary (e1 e2 : ℕ → Prop) : Prop :=
  ∀ outcome, e1 outcome ↔ ¬(e2 outcome)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutuallyExclusive exactlyOneMale exactlyTwoFemales ∧
  ¬(complementary exactlyOneMale exactlyTwoFemales) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l341_34124


namespace NUMINAMATH_CALUDE_fraction_equality_l341_34116

theorem fraction_equality : (3023 - 2990)^2 / 121 = 9 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l341_34116


namespace NUMINAMATH_CALUDE_add_negative_numbers_l341_34120

theorem add_negative_numbers : -10 + (-12) = -22 := by
  sorry

end NUMINAMATH_CALUDE_add_negative_numbers_l341_34120


namespace NUMINAMATH_CALUDE_gcd_consecutive_pairs_l341_34125

theorem gcd_consecutive_pairs (m n : ℕ) (h : m > n) :
  (∀ k : ℕ, k ∈ Finset.range (m - n) → Nat.gcd (n + k + 1) (m + k + 1) = 1) ↔ m = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_pairs_l341_34125


namespace NUMINAMATH_CALUDE_john_current_age_l341_34162

/-- John's current age -/
def john_age : ℕ := sorry

/-- John's sister's current age -/
def sister_age : ℕ := sorry

/-- John's sister is twice his age -/
axiom sister_twice_age : sister_age = 2 * john_age

/-- When John is 50, his sister will be 60 -/
axiom future_ages : sister_age + (50 - john_age) = 60

theorem john_current_age : john_age = 10 := by sorry

end NUMINAMATH_CALUDE_john_current_age_l341_34162


namespace NUMINAMATH_CALUDE_problem_solution_l341_34146

theorem problem_solution :
  let M : ℕ := 3009 / 3
  let N : ℕ := (2 * M) / 3
  let X : ℤ := M - N
  X = 335 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l341_34146


namespace NUMINAMATH_CALUDE_inequality_equivalence_l341_34107

theorem inequality_equivalence (x y : ℝ) : y - x < Real.sqrt (x^2) ↔ y < 0 ∨ y < 2*x := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l341_34107


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l341_34111

/-- The number of rows in the triangular array -/
def N : ℕ := 77

/-- The total number of coins in the triangular array -/
def total_coins : ℕ := 3003

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem stating the properties of the triangular array and the sum of digits of N -/
theorem triangular_array_coin_sum :
  (N * (N + 1)) / 2 = total_coins ∧ sum_of_digits N = 14 := by
  sorry

#eval sum_of_digits N

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l341_34111


namespace NUMINAMATH_CALUDE_pencils_remaining_l341_34189

/-- Given a box of pencils with an initial count and a number of pencils taken,
    prove that the remaining number of pencils is the difference between the initial count and the number taken. -/
theorem pencils_remaining (initial_count taken : ℕ) : 
  initial_count = 79 → taken = 4 → initial_count - taken = 75 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remaining_l341_34189


namespace NUMINAMATH_CALUDE_least_positive_difference_l341_34181

def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => r * geometric_sequence a₁ r n

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

def sequence_A : ℕ → ℝ := geometric_sequence 3 2

def sequence_B : ℕ → ℝ := arithmetic_sequence 15 30

def valid_term_A (n : ℕ) : Prop := sequence_A n ≤ 300

def valid_term_B (n : ℕ) : Prop := sequence_B n ≤ 300

theorem least_positive_difference :
  ∃ (m n : ℕ), valid_term_A m ∧ valid_term_B n ∧
    ∀ (i j : ℕ), valid_term_A i → valid_term_B j →
      |sequence_A m - sequence_B n| ≤ |sequence_A i - sequence_B j| ∧
      |sequence_A m - sequence_B n| = 3 :=
sorry

end NUMINAMATH_CALUDE_least_positive_difference_l341_34181


namespace NUMINAMATH_CALUDE_coprime_sequence_solution_l341_34193

/-- Represents the sequence of ones and twos constructed from multiples of a, b, and c -/
def constructSequence (a b c : ℕ) : List ℕ := sorry

/-- Checks if two natural numbers are coprime -/
def areCoprime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem coprime_sequence_solution :
  ∀ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 →
    areCoprime a b ∧ areCoprime b c ∧ areCoprime a c →
    (let seq := constructSequence a b c
     seq.count 1 = 356 ∧ 
     seq.count 2 = 36 ∧
     seq.take 16 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]) →
    a = 7 ∧ b = 9 ∧ c = 23 := by
  sorry

end NUMINAMATH_CALUDE_coprime_sequence_solution_l341_34193


namespace NUMINAMATH_CALUDE_card_width_is_15_l341_34183

/-- A rectangular card with a given perimeter and width-length relationship -/
structure RectangularCard where
  length : ℝ
  width : ℝ
  perimeter_eq : length * 2 + width * 2 = 46
  width_length_rel : width = length + 7

/-- The width of the rectangular card is 15 cm -/
theorem card_width_is_15 (card : RectangularCard) : card.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_card_width_is_15_l341_34183


namespace NUMINAMATH_CALUDE_smallest_integer_solution_eight_satisfies_smallest_integer_is_eight_l341_34180

theorem smallest_integer_solution : ∀ x : ℤ, x < 3 * x - 15 → x ≥ 8 :=
by
  sorry

theorem eight_satisfies : (8 : ℤ) < 3 * 8 - 15 :=
by
  sorry

theorem smallest_integer_is_eight : 
  (∀ x : ℤ, x < 3 * x - 15 → x ≥ 8) ∧ (8 < 3 * 8 - 15) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_eight_satisfies_smallest_integer_is_eight_l341_34180


namespace NUMINAMATH_CALUDE_money_distribution_l341_34148

theorem money_distribution (total : ℕ) (vasim_share : ℕ) : 
  vasim_share = 1500 →
  ∃ (faruk_share ranjith_share : ℕ),
    faruk_share + vasim_share + ranjith_share = total ∧
    5 * faruk_share = 3 * vasim_share ∧
    6 * faruk_share = 3 * ranjith_share ∧
    ranjith_share - faruk_share = 900 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l341_34148


namespace NUMINAMATH_CALUDE_bus_fraction_proof_l341_34139

def total_distance : ℝ := 30.000000000000007

theorem bus_fraction_proof :
  let distance_by_foot : ℝ := (1/3) * total_distance
  let distance_by_car : ℝ := 2
  let distance_by_bus : ℝ := total_distance - distance_by_foot - distance_by_car
  distance_by_bus / total_distance = 3/5 := by sorry

end NUMINAMATH_CALUDE_bus_fraction_proof_l341_34139


namespace NUMINAMATH_CALUDE_cylinder_height_from_cube_water_l341_34192

/-- The height of a cylinder filled with water from a cube -/
theorem cylinder_height_from_cube_water (cube_edge : ℝ) (cylinder_base_area : ℝ) 
  (h_cube_edge : cube_edge = 6)
  (h_cylinder_base : cylinder_base_area = 18)
  (h_water_conserved : cube_edge ^ 3 = cylinder_base_area * cylinder_height) :
  cylinder_height = 12 := by
  sorry

#check cylinder_height_from_cube_water

end NUMINAMATH_CALUDE_cylinder_height_from_cube_water_l341_34192


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l341_34123

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 ∧ a + b = 49 ∧ ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → c + d ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l341_34123


namespace NUMINAMATH_CALUDE_chessboard_coloring_l341_34113

/-- A color used to paint the chessboard squares -/
inductive Color
| Red
| Green
| Blue

/-- A chessboard configuration is a function from (row, column) to Color -/
def ChessboardConfig := Fin 4 → Fin 19 → Color

/-- The theorem statement -/
theorem chessboard_coloring (config : ChessboardConfig) :
  ∃ (r₁ r₂ : Fin 4) (c₁ c₂ : Fin 19),
    r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧
    config r₁ c₁ = config r₁ c₂ ∧
    config r₁ c₁ = config r₂ c₁ ∧
    config r₁ c₁ = config r₂ c₂ :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_l341_34113


namespace NUMINAMATH_CALUDE_cistern_solution_l341_34106

/-- Represents the time (in hours) it takes to fill or empty the cistern -/
structure CisternTime where
  fill : ℝ
  empty : ℝ
  both : ℝ

/-- The cistern filling problem -/
def cistern_problem (t : CisternTime) : Prop :=
  t.fill = 10 ∧ 
  t.empty = 12 ∧ 
  t.both = 60 ∧
  t.both = (t.fill * t.empty) / (t.empty - t.fill)

theorem cistern_solution :
  ∃ t : CisternTime, cistern_problem t :=
sorry

end NUMINAMATH_CALUDE_cistern_solution_l341_34106


namespace NUMINAMATH_CALUDE_pencil_sale_problem_l341_34185

theorem pencil_sale_problem (total_students : Nat) (first_group : Nat) (second_group : Nat) (third_group : Nat)
  (first_group_pencils : Nat) (third_group_pencils : Nat) (total_pencils : Nat) :
  total_students = first_group + second_group + third_group →
  first_group = 2 →
  third_group = 2 →
  first_group_pencils = 2 →
  third_group_pencils = 1 →
  total_pencils = 24 →
  ∃ (second_group_pencils : Nat),
    second_group_pencils * second_group + first_group_pencils * first_group + third_group_pencils * third_group = total_pencils ∧
    second_group_pencils = 3 :=
by sorry


end NUMINAMATH_CALUDE_pencil_sale_problem_l341_34185


namespace NUMINAMATH_CALUDE_min_games_for_90_percent_win_l341_34112

theorem min_games_for_90_percent_win (N : ℕ) : 
  (∀ k : ℕ, k < N → (2 + k : ℚ) / (5 + k) ≤ 9/10) ∧
  (2 + N : ℚ) / (5 + N) > 9/10 →
  N = 26 :=
sorry

end NUMINAMATH_CALUDE_min_games_for_90_percent_win_l341_34112


namespace NUMINAMATH_CALUDE_combined_prism_volume_l341_34171

/-- The volume of a structure consisting of a triangular prism on top of a rectangular prism -/
theorem combined_prism_volume (rect_length rect_width rect_height tri_base tri_height tri_length : ℝ) :
  rect_length = 6 →
  rect_width = 4 →
  rect_height = 2 →
  tri_base = 3 →
  tri_height = 3 →
  tri_length = 4 →
  (rect_length * rect_width * rect_height) + (1/2 * tri_base * tri_height * tri_length) = 66 := by
  sorry

end NUMINAMATH_CALUDE_combined_prism_volume_l341_34171


namespace NUMINAMATH_CALUDE_complement_of_A_l341_34164

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x < 0}

-- State the theorem
theorem complement_of_A (x : ℝ) : x ∈ (Set.compl A) ↔ 0 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l341_34164


namespace NUMINAMATH_CALUDE_expression_evaluation_l341_34117

theorem expression_evaluation :
  let a : ℤ := -1
  let b : ℤ := 2
  3 * (a^2 * b + a * b^2) - 2 * (a^2 * b - 1) - 2 * a * b^2 - 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l341_34117


namespace NUMINAMATH_CALUDE_complex_expression_magnitude_l341_34140

theorem complex_expression_magnitude : 
  Complex.abs ((18 - 5 * Complex.I) * (14 + 6 * Complex.I) - (3 - 12 * Complex.I) * (4 + 9 * Complex.I)) = Real.sqrt 146365 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_magnitude_l341_34140


namespace NUMINAMATH_CALUDE_additive_function_characterization_l341_34134

def is_additive (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

theorem additive_function_characterization (f : ℚ → ℚ) (h : is_additive f) :
  ∃ k : ℚ, ∀ x : ℚ, f x = k * x := by
  sorry

end NUMINAMATH_CALUDE_additive_function_characterization_l341_34134


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l341_34137

theorem arithmetic_calculation : 5 * 7 + 10 * 4 - 35 / 5 + 18 / 3 = 74 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l341_34137


namespace NUMINAMATH_CALUDE_direct_variation_problem_l341_34194

theorem direct_variation_problem (k : ℝ) :
  (∀ x y : ℝ, 5 * y = k * x^2) →
  (5 * 8 = k * 2^2) →
  (5 * 32 = k * 4^2) :=
by
  sorry

end NUMINAMATH_CALUDE_direct_variation_problem_l341_34194


namespace NUMINAMATH_CALUDE_tangent_circle_center_slope_l341_34101

-- Define the circles u₁ and u₂
def u₁ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 32 = 0
def u₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 20*y + 128 = 0

-- Define the condition for a point (x, y) to be on the line y = bx
def on_line (x y b : ℝ) : Prop := y = b * x

-- Define the condition for a circle to be externally tangent to u₁
def externally_tangent_u₁ (x y r : ℝ) : Prop :=
  r + 12 = Real.sqrt ((x + 4)^2 + (y - 10)^2)

-- Define the condition for a circle to be internally tangent to u₂
def internally_tangent_u₂ (x y r : ℝ) : Prop :=
  8 - r = Real.sqrt ((x - 4)^2 + (y - 10)^2)

-- State the theorem
theorem tangent_circle_center_slope :
  ∃ n : ℝ, 
    (∀ b : ℝ, b > 0 → 
      (∃ x y r : ℝ, 
        on_line x y b ∧ 
        externally_tangent_u₁ x y r ∧ 
        internally_tangent_u₂ x y r) → 
      n ≤ b) ∧
    n^2 = 69/25 := by sorry

end NUMINAMATH_CALUDE_tangent_circle_center_slope_l341_34101


namespace NUMINAMATH_CALUDE_naoh_equals_nano3_l341_34104

/-- Represents the number of moles of a chemical substance -/
def Moles : Type := ℝ

/-- Represents the chemical reaction between NH4NO3 and NaOH to produce NaNO3 -/
structure Reaction where
  nh4no3_initial : Moles
  naoh_combined : Moles
  nano3_formed : Moles

/-- The reaction has a 1:1 molar ratio between NH4NO3 and NaOH to produce NaNO3 -/
axiom molar_ratio (r : Reaction) : r.nh4no3_initial = r.naoh_combined

/-- The number of moles of NH4NO3 initially present equals the number of moles of NaNO3 formed -/
axiom conservation (r : Reaction) : r.nh4no3_initial = r.nano3_formed

/-- The number of moles of NaOH combined equals the number of moles of NaNO3 formed -/
theorem naoh_equals_nano3 (r : Reaction) : r.naoh_combined = r.nano3_formed := by
  sorry

end NUMINAMATH_CALUDE_naoh_equals_nano3_l341_34104


namespace NUMINAMATH_CALUDE_m_range_l341_34128

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the condition that m is positive
def m_positive (m : ℝ) : Prop := m > 0

-- Define the condition about the relationship between p and q
def condition (m : ℝ) : Prop := 
  ∀ x, (¬(p x) → ¬(q x m)) ∧ ∃ x, (¬(p x) ∧ q x m)

-- State the theorem
theorem m_range (m : ℝ) : 
  m_positive m → condition m → m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_m_range_l341_34128


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l341_34172

/-- The volume of a pizza slice -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1/2 →
  diameter = 18 →
  num_pieces = 16 →
  (π * (diameter/2)^2 * thickness) / num_pieces = 2.53125 * π := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l341_34172


namespace NUMINAMATH_CALUDE_triangle_inequality_l341_34109

theorem triangle_inequality (a b c Δ : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : Δ > 0) 
  (h_heron : Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) 
  (h_semiperimeter : s = (a + b + c) / 2) : 
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 := by
sorry


end NUMINAMATH_CALUDE_triangle_inequality_l341_34109


namespace NUMINAMATH_CALUDE_sector_central_angle_l341_34188

/-- Given a sector with arc length 4 cm and area 4 cm², prove that its central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) : 
  r * θ = 4 → (1/2) * r^2 * θ = 4 → θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l341_34188


namespace NUMINAMATH_CALUDE_initial_average_weight_l341_34133

theorem initial_average_weight 
  (initial_count : ℕ) 
  (new_student_weight : ℝ) 
  (new_average : ℝ) : 
  initial_count = 29 →
  new_student_weight = 10 →
  new_average = 27.4 →
  ∃ (initial_average : ℝ),
    initial_average * initial_count + new_student_weight = 
    new_average * (initial_count + 1) ∧
    initial_average = 28 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_weight_l341_34133


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l341_34126

def bracket (x y z : ℚ) : ℚ := (x + y) / z

theorem nested_bracket_equals_two :
  bracket (bracket 45 15 60) (bracket 3 3 6) (bracket 24 6 30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_equals_two_l341_34126


namespace NUMINAMATH_CALUDE_decimal_difference_l341_34158

/- Define the repeating decimal 0.̅72 -/
def repeating_decimal : ℚ := 72 / 99

/- Define the terminating decimal 0.72 -/
def terminating_decimal : ℚ := 72 / 100

/- Theorem statement -/
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 2 / 275 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l341_34158


namespace NUMINAMATH_CALUDE_vanessa_phone_pictures_l341_34165

theorem vanessa_phone_pictures :
  ∀ (phone_pics camera_pics num_albums pics_per_album : ℕ),
    camera_pics = 7 →
    num_albums = 5 →
    pics_per_album = 6 →
    phone_pics + camera_pics = num_albums * pics_per_album →
    phone_pics = 23 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_phone_pictures_l341_34165


namespace NUMINAMATH_CALUDE_ball_drawing_problem_l341_34136

theorem ball_drawing_problem (n : ℕ+) : 
  (3 * n) / ((n + 3) * (n + 2) : ℝ) = 7 / 30 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_problem_l341_34136


namespace NUMINAMATH_CALUDE_sin_90_degrees_l341_34145

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l341_34145


namespace NUMINAMATH_CALUDE_sum_m_n_equals_negative_two_l341_34157

/-- A polynomial in x and y -/
def polynomial (m n : ℝ) (x y : ℝ) : ℝ := m * x^2 - n * x * y - 2 * x * y + y - 3

/-- The condition that the polynomial has no quadratic terms when simplified -/
def no_quadratic_terms (m n : ℝ) : Prop :=
  ∀ x y, polynomial m n x y = (-n - 2) * x * y + y - 3

theorem sum_m_n_equals_negative_two (m n : ℝ) (h : no_quadratic_terms m n) : m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_negative_two_l341_34157


namespace NUMINAMATH_CALUDE_min_sphere_surface_area_l341_34198

/-- Given a rectangular parallelepiped with volume 12, height 4, and all vertices on the surface of a sphere,
    prove that the minimum surface area of the sphere is 22π. -/
theorem min_sphere_surface_area (a b c : ℝ) (h_volume : a * b * c = 12) (h_height : c = 4)
  (h_on_sphere : ∃ (r : ℝ), a^2 + b^2 + c^2 = 4 * r^2) :
  ∃ (S : ℝ), S = 22 * Real.pi ∧ ∀ (r : ℝ), (a^2 + b^2 + c^2 = 4 * r^2) → 4 * Real.pi * r^2 ≥ S := by
  sorry

end NUMINAMATH_CALUDE_min_sphere_surface_area_l341_34198


namespace NUMINAMATH_CALUDE_product_of_integers_l341_34138

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 22)
  (diff_squares_eq : x^2 - y^2 = 44) :
  x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l341_34138


namespace NUMINAMATH_CALUDE_inverse_of_inverse_fourteen_l341_34168

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x - 4

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 4) / 3

-- Theorem statement
theorem inverse_of_inverse_fourteen (h : ∀ x, g (g_inv x) = x) :
  g_inv (g_inv 14) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_inverse_fourteen_l341_34168


namespace NUMINAMATH_CALUDE_lisa_packing_peanuts_l341_34141

/-- The amount of packing peanuts needed for a large order in grams -/
def large_order_peanuts : ℕ := 200

/-- The amount of packing peanuts needed for a small order in grams -/
def small_order_peanuts : ℕ := 50

/-- The number of large orders Lisa has sent -/
def large_orders : ℕ := 3

/-- The number of small orders Lisa has sent -/
def small_orders : ℕ := 4

/-- The total amount of packing peanuts used by Lisa -/
def total_peanuts : ℕ := large_order_peanuts * large_orders + small_order_peanuts * small_orders

theorem lisa_packing_peanuts : total_peanuts = 800 := by
  sorry

end NUMINAMATH_CALUDE_lisa_packing_peanuts_l341_34141


namespace NUMINAMATH_CALUDE_smallest_multiple_of_3_4_5_l341_34199

theorem smallest_multiple_of_3_4_5 : ∃ n : ℕ+, (∀ m : ℕ+, 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m → n ≤ m) ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_3_4_5_l341_34199


namespace NUMINAMATH_CALUDE_farmer_apples_l341_34154

/-- The number of apples given to the neighbor -/
def apples_given (initial current : ℕ) : ℕ := initial - current

/-- Theorem: The number of apples given to the neighbor is the difference between
    the initial number of apples and the current number of apples -/
theorem farmer_apples (initial current : ℕ) (h : initial ≥ current) :
  apples_given initial current = initial - current :=
by sorry

end NUMINAMATH_CALUDE_farmer_apples_l341_34154


namespace NUMINAMATH_CALUDE_expression_evaluation_l341_34175

theorem expression_evaluation :
  let x : ℚ := 7/6
  (x - 1) / x / (x - (2*x - 1) / x) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l341_34175


namespace NUMINAMATH_CALUDE_solution_of_equation_l341_34196

theorem solution_of_equation : ∃ x : ℝ, (3 / (x - 2) = 1) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l341_34196


namespace NUMINAMATH_CALUDE_fiona_final_piles_count_l341_34135

/-- Represents the number of distinct final pile configurations in Fiona's card arranging process. -/
def fiona_final_piles (n : ℕ) : ℕ :=
  if n ≥ 2 then 2^(n-2) else 1

/-- The theorem stating the number of distinct final pile configurations in Fiona's card arranging process. -/
theorem fiona_final_piles_count (n : ℕ) :
  (∀ k : ℕ, k < n → ∃ (m : ℕ), m ≤ n ∧ fiona_final_piles k = fiona_final_piles m) →
  fiona_final_piles n = if n ≥ 2 then 2^(n-2) else 1 :=
by sorry

end NUMINAMATH_CALUDE_fiona_final_piles_count_l341_34135


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l341_34160

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The line y = x - 1 -/
def line_equation (p : Point) : Prop :=
  p.y = p.x - 1

/-- Theorem: Given the conditions, the hyperbola has the equation x²/2 - y²/5 = 1 -/
theorem hyperbola_theorem (h : Hyperbola) (f m n : Point) :
  -- Center at origin
  hyperbola_equation h ⟨0, 0⟩ →
  -- Focus at (√7, 0)
  f = ⟨Real.sqrt 7, 0⟩ →
  -- M and N are on the hyperbola and the line
  hyperbola_equation h m ∧ line_equation m →
  hyperbola_equation h n ∧ line_equation n →
  -- Midpoint x-coordinate is -2/3
  (m.x + n.x) / 2 = -2/3 →
  -- The hyperbola equation is x²/2 - y²/5 = 1
  h.a^2 = 2 ∧ h.b^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l341_34160


namespace NUMINAMATH_CALUDE_lunch_combinations_eq_27_l341_34147

/-- Represents a category of food items in the cafeteria -/
structure FoodCategory where
  options : Finset String
  size_eq_three : options.card = 3

/-- Represents the cafeteria menu -/
structure CafeteriaMenu where
  main_dishes : FoodCategory
  beverages : FoodCategory
  snacks : FoodCategory

/-- A function to calculate the number of distinct lunch combinations -/
def count_lunch_combinations (menu : CafeteriaMenu) : ℕ :=
  menu.main_dishes.options.card * menu.beverages.options.card * menu.snacks.options.card

/-- Theorem stating that the number of distinct lunch combinations is 27 -/
theorem lunch_combinations_eq_27 (menu : CafeteriaMenu) :
  count_lunch_combinations menu = 27 := by
  sorry

#check lunch_combinations_eq_27

end NUMINAMATH_CALUDE_lunch_combinations_eq_27_l341_34147


namespace NUMINAMATH_CALUDE_octopus_puzzle_l341_34119

structure Octopus where
  color : String
  legs : Nat
  statement : Bool

def isLying (o : Octopus) : Bool :=
  (o.legs = 7 ∧ ¬o.statement) ∨ (o.legs = 8 ∧ o.statement)

def totalLegs (os : List Octopus) : Nat :=
  os.foldl (fun acc o => acc + o.legs) 0

theorem octopus_puzzle :
  ∃ (green blue red : Octopus),
    [green, blue, red].all (fun o => o.legs = 7 ∨ o.legs = 8) ∧
    isLying green ∧
    ¬isLying blue ∧
    isLying red ∧
    green.statement = (totalLegs [green, blue, red] = 21) ∧
    blue.statement = ¬green.statement ∧
    red.statement = (¬green.statement ∧ ¬blue.statement) ∧
    green.legs = 7 ∧
    blue.legs = 8 ∧
    red.legs = 7 :=
  sorry

#check octopus_puzzle

end NUMINAMATH_CALUDE_octopus_puzzle_l341_34119


namespace NUMINAMATH_CALUDE_distance_between_AB_l341_34195

/-- The distance between two points A and B, where two motorcyclists meet twice -/
def distance_AB : ℝ := 125

/-- The distance of the first meeting point from B -/
def distance_first_meeting : ℝ := 50

/-- The distance of the second meeting point from A -/
def distance_second_meeting : ℝ := 25

/-- Theorem stating that the distance between A and B is 125 km -/
theorem distance_between_AB : 
  distance_AB = distance_first_meeting + distance_second_meeting :=
by sorry

end NUMINAMATH_CALUDE_distance_between_AB_l341_34195


namespace NUMINAMATH_CALUDE_unique_prime_pair_l341_34149

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Nat.Prime (p + q) ∧ 
  Nat.Prime (p^2 + q^2 - q) ∧ 
  p = 3 ∧ 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l341_34149


namespace NUMINAMATH_CALUDE_radio_price_rank_l341_34159

theorem radio_price_rank (n : ℕ) (prices : Finset ℕ) (radio_price : ℕ) :
  n = 58 →
  prices.card = n + 1 →
  radio_price ∈ prices →
  (∀ p ∈ prices, p ≤ radio_price) →
  (prices.filter (λ p => p < radio_price)).card = 41 →
  (prices.filter (λ p => p ≤ radio_price)).card = n + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_radio_price_rank_l341_34159


namespace NUMINAMATH_CALUDE_remainder_theorem_remainder_problem_l341_34197

/-- A polynomial of the form Dx^4 + Ex^2 + Fx - 2 -/
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x - 2

/-- The remainder theorem -/
theorem remainder_theorem {p : ℝ → ℝ} {a r : ℝ} :
  (∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + r) ↔ p a = r :=
sorry

theorem remainder_problem (D E F : ℝ) :
  (∃ r : ℝ, ∀ x, q D E F x = (x - 2) * r + 14) →
  (∃ s : ℝ, ∀ x, q D E F x = (x + 2) * s - 18) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_remainder_problem_l341_34197


namespace NUMINAMATH_CALUDE_delta_phi_equation_l341_34156

def δ (x : ℝ) : ℝ := 3 * x + 8

def φ (x : ℝ) : ℝ := 8 * x + 7

theorem delta_phi_equation (x : ℝ) : δ (φ x) = 7 ↔ x = -11/12 := by sorry

end NUMINAMATH_CALUDE_delta_phi_equation_l341_34156


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l341_34142

theorem absolute_value_inequality (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x + 1| < 2) ↔ (-3 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l341_34142


namespace NUMINAMATH_CALUDE_parabola_coefficient_l341_34179

/-- Theorem: For a parabola y = ax² + bx + c passing through points (4, 0), (t/3, 0), and (0, 60), the value of a is 45/t. -/
theorem parabola_coefficient (a b c t : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ (x = 4 ∨ x = t/3)) → 
  a * 0^2 + b * 0 + c = 60 →
  a = 45 / t :=
by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l341_34179


namespace NUMINAMATH_CALUDE_final_running_distance_l341_34118

/-- Calculates the final daily running distance after a 5-week program -/
theorem final_running_distance
  (initial_distance : ℕ)  -- Initial daily running distance in miles
  (increase_rate : ℕ)     -- Weekly increase in miles
  (increase_weeks : ℕ)    -- Number of weeks with distance increase
  (h1 : initial_distance = 3)
  (h2 : increase_rate = 1)
  (h3 : increase_weeks = 4)
  : initial_distance + increase_rate * increase_weeks = 7 :=
by sorry

end NUMINAMATH_CALUDE_final_running_distance_l341_34118


namespace NUMINAMATH_CALUDE_system_solution_l341_34143

theorem system_solution : ∃ (x y : ℝ), x + y = 8 ∧ x - 3*y = 4 ∧ x = 7 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l341_34143


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l341_34131

-- Define a geometric sequence of three terms
def is_geometric_sequence (a b c : ℝ) : Prop := b * b = a * c

-- Theorem statement
theorem geometric_sequence_middle_term :
  ∀ m : ℝ, is_geometric_sequence 1 m 4 → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l341_34131


namespace NUMINAMATH_CALUDE_trapezium_area_and_shorter_side_l341_34132

theorem trapezium_area_and_shorter_side (a b h : ℝ) :
  a = 24 ∧ b = 18 ∧ h = 15 →
  (1/2 : ℝ) * (a + b) * h = 315 ∧ min a b = 18 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_and_shorter_side_l341_34132


namespace NUMINAMATH_CALUDE_ten_thousand_one_divides_eight_digit_repeated_l341_34170

/-- Represents an 8-digit positive integer with repeated digits -/
def EightDigitRepeated : Type := 
  {n : ℕ // 10000000 ≤ n ∧ n < 100000000 ∧ ∃ a b c d : ℕ, n = a * 10000000 + b * 1000000 + c * 100000 + d * 10000 + a * 1000 + b * 100 + c * 10 + d}

/-- Theorem stating that 10001 is a factor of any EightDigitRepeated number -/
theorem ten_thousand_one_divides_eight_digit_repeated (z : EightDigitRepeated) : 
  10001 ∣ z.val := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_one_divides_eight_digit_repeated_l341_34170


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l341_34161

def second_order_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ → ℕ, (∀ n, d (n + 1) = d n + 2) ∧
               (∀ n, a (n + 1) = a n + d n)

theorem fifth_term_of_sequence
  (a : ℕ → ℕ)
  (h : second_order_arithmetic_sequence a)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 7)
  (h4 : a 4 = 13) :
  a 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l341_34161


namespace NUMINAMATH_CALUDE_book_page_ratio_l341_34115

/-- Given a set of books with specific page counts, prove the ratio of pages between middle and shortest books --/
theorem book_page_ratio (longest middle shortest : ℕ) : 
  longest = 396 → 
  shortest = longest / 4 → 
  middle = 297 → 
  middle / shortest = 3 := by
sorry

end NUMINAMATH_CALUDE_book_page_ratio_l341_34115


namespace NUMINAMATH_CALUDE_denis_neighbors_l341_34129

-- Define the students
inductive Student : Type
| Anya : Student
| Borya : Student
| Vera : Student
| Gena : Student
| Denis : Student

-- Define the line as a list of students
def Line := List Student

-- Define a function to check if two students are next to each other in the line
def next_to (s1 s2 : Student) (line : Line) : Prop :=
  ∃ i, (line.get? i = some s1 ∧ line.get? (i+1) = some s2) ∨
       (line.get? i = some s2 ∧ line.get? (i+1) = some s1)

-- Define the conditions
def valid_line (line : Line) : Prop :=
  (line.length = 5) ∧
  (line.head? = some Student.Borya) ∧
  (next_to Student.Vera Student.Anya line) ∧
  (¬ next_to Student.Vera Student.Gena line) ∧
  (¬ next_to Student.Anya Student.Borya line) ∧
  (¬ next_to Student.Anya Student.Gena line) ∧
  (¬ next_to Student.Borya Student.Gena line)

-- Theorem to prove
theorem denis_neighbors (line : Line) (h : valid_line line) :
  next_to Student.Denis Student.Anya line ∧ next_to Student.Denis Student.Gena line :=
sorry

end NUMINAMATH_CALUDE_denis_neighbors_l341_34129


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l341_34103

theorem arithmetic_expression_equality :
  4^2 * 10 + 5 * 12 + 12 * 4 + 24 / 3 * 9 = 340 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l341_34103


namespace NUMINAMATH_CALUDE_hypergeom_expected_and_variance_l341_34114

/-- Hypergeometric distribution parameters -/
structure HyperGeomParams where
  N : ℕ  -- Population size
  K : ℕ  -- Number of success states in the population
  n : ℕ  -- Number of draws
  h1 : K ≤ N
  h2 : n ≤ N

/-- Expected value of a hypergeometric distribution -/
def expected_value (p : HyperGeomParams) : ℚ :=
  (p.n * p.K : ℚ) / p.N

/-- Variance of a hypergeometric distribution -/
def variance (p : HyperGeomParams) : ℚ :=
  (p.n * p.K * (p.N - p.K) * (p.N - p.n) : ℚ) / (p.N^2 * (p.N - 1))

/-- Theorem: Expected value and variance for the given problem -/
theorem hypergeom_expected_and_variance :
  ∃ (p : HyperGeomParams),
    p.N = 100 ∧ p.K = 10 ∧ p.n = 3 ∧
    expected_value p = 3/10 ∧
    variance p = 51/200 := by
  sorry

end NUMINAMATH_CALUDE_hypergeom_expected_and_variance_l341_34114


namespace NUMINAMATH_CALUDE_function_equivalence_l341_34169

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 + 2*x

-- State the theorem
theorem function_equivalence : ∀ x : ℝ, f (x - 1) = x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l341_34169


namespace NUMINAMATH_CALUDE_rectangle_perimeter_reduction_l341_34127

theorem rectangle_perimeter_reduction (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  2 * (0.9 * a + 0.8 * b) = 0.88 * 2 * (a + b) → 
  2 * (0.8 * a + 0.9 * b) = 0.82 * 2 * (a + b) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_reduction_l341_34127


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l341_34186

theorem rope_cutting_problem (a b c : ℕ) 
  (ha : a = 45) (hb : b = 60) (hc : c = 75) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l341_34186


namespace NUMINAMATH_CALUDE_binomial_square_constant_l341_34184

theorem binomial_square_constant (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 9 * x^2 + 27 * x + a = (3 * x + b)^2) → a = 81 / 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l341_34184


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l341_34121

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -1/2 + (5/4) * Complex.I →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -1/2 - (5/4) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l341_34121


namespace NUMINAMATH_CALUDE_unique_prime_between_squares_l341_34190

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 9 ∧ 
  ∃ m : ℕ, p + 2 = m^2 ∧
  m = n + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_squares_l341_34190


namespace NUMINAMATH_CALUDE_problem_solution_l341_34155

theorem problem_solution : 
  let x := ((12 ^ 5) * (6 ^ 4)) / ((3 ^ 2) * (36 ^ 2)) + (Real.sqrt 9 * Real.log 27)
  ∃ ε > 0, |x - 27657.887510597983| < ε := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l341_34155


namespace NUMINAMATH_CALUDE_equality_preserved_under_addition_l341_34102

theorem equality_preserved_under_addition (a b : ℝ) : a = b → a + 3 = 3 + b := by
  sorry

end NUMINAMATH_CALUDE_equality_preserved_under_addition_l341_34102
