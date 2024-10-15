import Mathlib

namespace NUMINAMATH_CALUDE_union_and_intersection_when_a_is_two_range_of_a_for_necessary_but_not_sufficient_l464_46453

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 2 + a}

-- Theorem for part (1)
theorem union_and_intersection_when_a_is_two :
  (A ∪ B 2 = {x | -1 < x ∧ x < 4}) ∧ (A ∩ B 2 = {x | 0 < x ∧ x < 3}) := by sorry

-- Theorem for part (2)
theorem range_of_a_for_necessary_but_not_sufficient :
  {a : ℝ | ∀ x, x ∈ B a → x ∈ A} ∩ {a : ℝ | ∃ x, x ∈ B a ∧ x ∉ A} = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_when_a_is_two_range_of_a_for_necessary_but_not_sufficient_l464_46453


namespace NUMINAMATH_CALUDE_problem_statement_l464_46475

theorem problem_statement (x y : ℝ) 
  (h1 : (4 : ℝ)^y = 1 / (8 * (Real.sqrt 2)^(x + 2)))
  (h2 : (9 : ℝ)^x * (3 : ℝ)^y = 3 * Real.sqrt 3) :
  (5 : ℝ)^(x + y) = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l464_46475


namespace NUMINAMATH_CALUDE_gwen_candy_weight_l464_46443

/-- The amount of candy Gwen received given the total amount and Frank's amount -/
def gwens_candy (total frank : ℕ) : ℕ := total - frank

/-- Theorem stating that Gwen received 7 pounds of candy -/
theorem gwen_candy_weight :
  let total := 17
  let frank := 10
  gwens_candy total frank = 7 := by sorry

end NUMINAMATH_CALUDE_gwen_candy_weight_l464_46443


namespace NUMINAMATH_CALUDE_not_reach_54_after_60_ops_l464_46440

/-- Represents the possible operations on the board number -/
inductive Operation
| MultTwo
| DivTwo
| MultThree
| DivThree

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.MultTwo => n * 2
  | Operation.DivTwo => n / 2
  | Operation.MultThree => n * 3
  | Operation.DivThree => n / 3

/-- Applies a sequence of operations to a number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- Theorem: After 60 operations starting from 12, it's impossible to reach 54 -/
theorem not_reach_54_after_60_ops (ops : List Operation) :
  ops.length = 60 → applyOperations 12 ops ≠ 54 := by
  sorry

end NUMINAMATH_CALUDE_not_reach_54_after_60_ops_l464_46440


namespace NUMINAMATH_CALUDE_different_tens_digit_probability_l464_46465

def lower_bound : ℕ := 30
def upper_bound : ℕ := 89
def num_integers : ℕ := 7

def favorable_outcomes : ℕ := 27000000
def total_outcomes : ℕ := Nat.choose (upper_bound - lower_bound + 1) num_integers

theorem different_tens_digit_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 6750 / 9655173 := by
  sorry

end NUMINAMATH_CALUDE_different_tens_digit_probability_l464_46465


namespace NUMINAMATH_CALUDE_student_desk_arrangement_impossibility_l464_46460

theorem student_desk_arrangement_impossibility :
  ∀ (total_students total_desks : ℕ) 
    (girls boys : ℕ) 
    (girls_with_boys boys_with_girls : ℕ),
  total_students = 450 →
  total_desks = 225 →
  girls + boys = total_students →
  2 * total_desks = total_students →
  2 * girls_with_boys = girls →
  2 * boys_with_girls = boys →
  False :=
by sorry

end NUMINAMATH_CALUDE_student_desk_arrangement_impossibility_l464_46460


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l464_46423

/-- Calculate the total amount owed after one year with simple interest -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) :
  principal = 35 →
  rate = 0.05 →
  time = 1 →
  principal + principal * rate * time = 36.75 := by
  sorry


end NUMINAMATH_CALUDE_simple_interest_calculation_l464_46423


namespace NUMINAMATH_CALUDE_irregular_hexagon_perimeter_l464_46429

/-- An irregular hexagon with specific angle measurements and equal side lengths -/
structure IrregularHexagon where
  -- Side length of the hexagon
  side_length : ℝ
  -- Assumption that all sides are equal
  all_sides_equal : True
  -- Three nonadjacent angles measure 120°
  three_angles_120 : True
  -- The other three angles measure 60°
  three_angles_60 : True
  -- The enclosed area of the hexagon
  area : ℝ
  -- The area is 24
  area_is_24 : area = 24

/-- The perimeter of an irregular hexagon with the given conditions -/
def perimeter (h : IrregularHexagon) : ℝ := 6 * h.side_length

/-- Theorem stating that the perimeter of the irregular hexagon is 24 / (3^(1/4)) -/
theorem irregular_hexagon_perimeter (h : IrregularHexagon) : 
  perimeter h = 24 / Real.rpow 3 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_irregular_hexagon_perimeter_l464_46429


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l464_46427

theorem wrong_mark_calculation (correct_mark : ℕ) (num_pupils : ℕ) :
  correct_mark = 45 →
  num_pupils = 44 →
  ∃ (wrong_mark : ℕ),
    (wrong_mark - correct_mark : ℚ) / num_pupils = 1/2 ∧
    wrong_mark = 67 :=
by sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l464_46427


namespace NUMINAMATH_CALUDE_jeff_bought_two_stars_l464_46405

/-- The number of ninja throwing stars Jeff bought from Chad -/
def stars_bought_by_jeff (eric_stars chad_stars jeff_stars total_stars : ℕ) : ℕ :=
  chad_stars - (total_stars - eric_stars - jeff_stars)

theorem jeff_bought_two_stars :
  let eric_stars : ℕ := 4
  let chad_stars : ℕ := 2 * eric_stars
  let jeff_stars : ℕ := 6
  let total_stars : ℕ := 16
  stars_bought_by_jeff eric_stars chad_stars jeff_stars total_stars = 2 := by
sorry

end NUMINAMATH_CALUDE_jeff_bought_two_stars_l464_46405


namespace NUMINAMATH_CALUDE_fifth_row_sum_in_spiral_grid_l464_46456

/-- Represents a spiral arrangement of numbers in a square grid -/
def SpiralGrid (n : ℕ) := Matrix (Fin n) (Fin n) ℕ

/-- Creates a spiral grid of size n × n with numbers from 1 to n^2 -/
def createSpiralGrid (n : ℕ) : SpiralGrid n :=
  sorry

/-- Returns the numbers in a specific row of the spiral grid -/
def getRowNumbers (grid : SpiralGrid 20) (row : Fin 20) : List ℕ :=
  sorry

/-- Theorem: In a 20x20 spiral grid, the sum of the greatest and least numbers 
    in the fifth row is 565 -/
theorem fifth_row_sum_in_spiral_grid :
  let grid := createSpiralGrid 20
  let fifthRowNumbers := getRowNumbers grid 4
  (List.maximum fifthRowNumbers).getD 0 + (List.minimum fifthRowNumbers).getD 0 = 565 := by
  sorry

end NUMINAMATH_CALUDE_fifth_row_sum_in_spiral_grid_l464_46456


namespace NUMINAMATH_CALUDE_correct_num_dogs_l464_46468

/-- Represents the number of dogs Carly worked on --/
def num_dogs : ℕ := 11

/-- Represents the total number of nails trimmed --/
def total_nails : ℕ := 164

/-- Represents the number of dogs with three legs --/
def three_legged_dogs : ℕ := 3

/-- Represents the number of dogs with three nails on one paw --/
def three_nailed_dogs : ℕ := 2

/-- Represents the number of dogs with an extra nail on one paw --/
def extra_nailed_dogs : ℕ := 1

/-- Represents the number of nails on a regular dog --/
def nails_per_regular_dog : ℕ := 4 * 4

/-- Theorem stating that the number of dogs is correct given the conditions --/
theorem correct_num_dogs :
  num_dogs * nails_per_regular_dog
  - three_legged_dogs * 4
  - three_nailed_dogs
  + extra_nailed_dogs
  = total_nails :=
by sorry

end NUMINAMATH_CALUDE_correct_num_dogs_l464_46468


namespace NUMINAMATH_CALUDE_m_range_proof_l464_46471

theorem m_range_proof (m : ℝ) (h_pos : m > 0) : 
  (∀ x : ℝ, x ≤ -1 → (3 * m - 1) * 2^x < 1) → 
  m < 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_proof_l464_46471


namespace NUMINAMATH_CALUDE_bonus_calculation_l464_46425

/-- Represents the wages of a worker for three months -/
structure Wages where
  october : ℝ
  november : ℝ
  december : ℝ

/-- Calculates the bonus based on the given wages -/
def calculate_bonus (w : Wages) : ℝ :=
  0.2 * (w.october + w.november + w.december)

theorem bonus_calculation (w : Wages) 
  (h1 : w.october / w.november = 3/2 / (4/3))
  (h2 : w.november / w.december = 2 / (8/3))
  (h3 : w.december = w.october + 450) :
  calculate_bonus w = 1494 := by
  sorry

#eval calculate_bonus { october := 2430, november := 2160, december := 2880 }

end NUMINAMATH_CALUDE_bonus_calculation_l464_46425


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l464_46461

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses the digits of a ThreeDigitNumber -/
def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.units
  tens := n.tens
  units := n.hundreds
  is_valid := by sorry

theorem unique_number_satisfying_conditions :
  ∃! (n : ThreeDigitNumber),
    n.hundreds + n.tens + n.units = 20 ∧
    ∃ (m : ThreeDigitNumber),
      n.toNat - 16 = m.toNat ∧
      m = n.reverse ∧
    n.hundreds = 9 ∧ n.tens = 7 ∧ n.units = 4 := by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l464_46461


namespace NUMINAMATH_CALUDE_household_electricity_most_suitable_l464_46422

/-- Represents an investigation option --/
inductive InvestigationOption
  | ProductPopularity
  | TVViewershipRatings
  | AmmunitionExplosivePower
  | HouseholdElectricityConsumption

/-- Defines what makes an investigation suitable for a census method --/
def suitableForCensus (option : InvestigationOption) : Prop :=
  match option with
  | InvestigationOption.HouseholdElectricityConsumption => True
  | _ => False

/-- Theorem stating that investigating household electricity consumption is most suitable for census --/
theorem household_electricity_most_suitable :
    ∀ option : InvestigationOption,
      suitableForCensus option →
      option = InvestigationOption.HouseholdElectricityConsumption :=
by
  sorry

/-- Definition of a census method --/
def censusMethod (population : Type) (examine : population → Prop) : Prop :=
  ∀ subject : population, examine subject

#check household_electricity_most_suitable

end NUMINAMATH_CALUDE_household_electricity_most_suitable_l464_46422


namespace NUMINAMATH_CALUDE_difference_of_percentages_l464_46469

theorem difference_of_percentages (x y : ℝ) : 
  0.60 * (50 + x) - 0.45 * (30 + y) = 16.5 + 0.60 * x - 0.45 * y :=
by sorry

end NUMINAMATH_CALUDE_difference_of_percentages_l464_46469


namespace NUMINAMATH_CALUDE_technicians_count_l464_46403

/-- Proves the number of technicians in a workshop with given salary conditions -/
theorem technicians_count (total_workers : ℕ) (avg_salary : ℕ) (tech_salary : ℕ) (rest_salary : ℕ) :
  total_workers = 14 ∧ 
  avg_salary = 8000 ∧ 
  tech_salary = 10000 ∧ 
  rest_salary = 6000 → 
  ∃ (tech_count : ℕ),
    tech_count = 7 ∧ 
    tech_count ≤ total_workers ∧
    tech_count * tech_salary + (total_workers - tech_count) * rest_salary = total_workers * avg_salary :=
by sorry

end NUMINAMATH_CALUDE_technicians_count_l464_46403


namespace NUMINAMATH_CALUDE_plot_length_l464_46414

/-- The length of a rectangular plot given specific conditions -/
theorem plot_length (breadth : ℝ) (length : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 32 →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  2 * (length + breadth) * cost_per_meter = total_cost →
  length = 66 := by
sorry

end NUMINAMATH_CALUDE_plot_length_l464_46414


namespace NUMINAMATH_CALUDE_infinite_primes_with_solutions_l464_46432

theorem infinite_primes_with_solutions (S : Finset Nat) (h : ∀ p ∈ S, Nat.Prime p) :
  ∃ p : Nat, p ∉ S ∧ Nat.Prime p ∧ ∃ x : ℤ, x^2 + x + 1 = p := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_with_solutions_l464_46432


namespace NUMINAMATH_CALUDE_polynomial_value_l464_46415

theorem polynomial_value (x : ℝ) (h : x^2 + 2*x - 2 = 0) : 4 - 2*x - x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l464_46415


namespace NUMINAMATH_CALUDE_gcd_cube_plus_five_cube_l464_46464

theorem gcd_cube_plus_five_cube (n : ℕ) (h : n > 2^5) : Nat.gcd (n^3 + 5^3) (n + 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_five_cube_l464_46464


namespace NUMINAMATH_CALUDE_cube_sum_equals_110_l464_46401

theorem cube_sum_equals_110 (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_110_l464_46401


namespace NUMINAMATH_CALUDE_wire_service_reporters_l464_46486

theorem wire_service_reporters (total : ℝ) (x y both other_politics : ℝ) :
  x = 0.3 * total →
  y = 0.1 * total →
  both = 0.1 * total →
  other_politics = 0.25 * (x + y - both + other_politics) →
  total - (x + y - both + other_politics) = 0.45 * total :=
by sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l464_46486


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l464_46450

theorem stratified_sampling_problem (total_population : ℕ) (first_stratum : ℕ) (sample_first_stratum : ℕ) (total_sample : ℕ) :
  total_population = 1500 →
  first_stratum = 700 →
  sample_first_stratum = 14 →
  (sample_first_stratum : ℚ) / total_sample = (first_stratum : ℚ) / total_population →
  total_sample = 30 :=
by
  sorry

#check stratified_sampling_problem

end NUMINAMATH_CALUDE_stratified_sampling_problem_l464_46450


namespace NUMINAMATH_CALUDE_integral_reciprocal_sqrt_one_minus_x_squared_l464_46470

open Real MeasureTheory

theorem integral_reciprocal_sqrt_one_minus_x_squared : 
  ∫ x in (Set.Icc 0 (1 / Real.sqrt 2)), 1 / ((1 - x^2) * Real.sqrt (1 - x^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_sqrt_one_minus_x_squared_l464_46470


namespace NUMINAMATH_CALUDE_sum_equals_twelve_l464_46494

theorem sum_equals_twelve (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_twelve_l464_46494


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l464_46493

/-- Represents a cube with integers on its faces -/
structure Cube where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- The sum of vertex products for a cube -/
def vertexSum (cube : Cube) : ℕ :=
  2 * (cube.a * cube.b * cube.c +
       cube.a * cube.b * cube.f +
       cube.d * cube.b * cube.c +
       cube.d * cube.b * cube.f)

/-- The sum of face numbers for a cube -/
def faceSum (cube : Cube) : ℕ :=
  cube.a + cube.b + cube.c + cube.d + cube.e + cube.f

/-- Theorem stating the relationship between vertex sum and face sum -/
theorem cube_sum_theorem (cube : Cube) 
  (h1 : vertexSum cube = 1332)
  (h2 : cube.b = cube.e) : 
  faceSum cube = 47 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l464_46493


namespace NUMINAMATH_CALUDE_complement_M_N_l464_46411

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}

theorem complement_M_N : M \ N = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_complement_M_N_l464_46411


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l464_46447

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) :
  area = 144 →
  side * side = area →
  perimeter = 4 * side →
  perimeter = 48 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l464_46447


namespace NUMINAMATH_CALUDE_marks_leftover_amount_marks_leftover_is_980_l464_46413

/-- Calculates the amount Mark has leftover each week after his raise and new expenses -/
theorem marks_leftover_amount (old_wage : ℝ) (raise_percentage : ℝ) 
  (hours_per_day : ℝ) (days_per_week : ℝ) (old_bills : ℝ) (trainer_cost : ℝ) : ℝ :=
  let new_wage := old_wage * (1 + raise_percentage / 100)
  let weekly_hours := hours_per_day * days_per_week
  let weekly_earnings := new_wage * weekly_hours
  let weekly_expenses := old_bills + trainer_cost
  weekly_earnings - weekly_expenses

/-- Proves that Mark has $980 leftover each week after his raise and new expenses -/
theorem marks_leftover_is_980 : 
  marks_leftover_amount 40 5 8 5 600 100 = 980 := by
  sorry

end NUMINAMATH_CALUDE_marks_leftover_amount_marks_leftover_is_980_l464_46413


namespace NUMINAMATH_CALUDE_distance_B_to_center_l464_46462

/-- A circle with radius √52 and points A, B, C satisfying given conditions -/
structure NotchedCircle where
  -- Define the circle
  center : ℝ × ℝ
  radius : ℝ
  radius_eq : radius = Real.sqrt 52

  -- Define points A, B, C
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

  -- Conditions
  on_circle_A : (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2
  on_circle_B : (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2
  on_circle_C : (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2

  AB_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64  -- 8^2 = 64
  BC_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 16  -- 4^2 = 16

  right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

/-- The square of the distance from point B to the center of the circle is 20 -/
theorem distance_B_to_center (nc : NotchedCircle) :
  (nc.B.1 - nc.center.1)^2 + (nc.B.2 - nc.center.2)^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_distance_B_to_center_l464_46462


namespace NUMINAMATH_CALUDE_no_numbers_equal_seven_times_digit_sum_l464_46435

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem no_numbers_equal_seven_times_digit_sum :
  ∀ n : ℕ, n > 0 ∧ n < 2000 → n ≠ 7 * (sum_of_digits n) :=
by
  sorry

end NUMINAMATH_CALUDE_no_numbers_equal_seven_times_digit_sum_l464_46435


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l464_46406

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.exp (x * Real.sin (5 * x)) - 1 else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_derivative_f_at_zero_l464_46406


namespace NUMINAMATH_CALUDE_unique_absolute_value_complex_root_l464_46431

theorem unique_absolute_value_complex_root : ∃! r : ℝ, 
  (∃ z : ℂ, z^2 - 6*z + 20 = 0 ∧ Complex.abs z = r) ∧ r ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_absolute_value_complex_root_l464_46431


namespace NUMINAMATH_CALUDE_valid_allocations_count_l464_46495

/-- The number of male volunteers -/
def num_males : ℕ := 4

/-- The number of female volunteers -/
def num_females : ℕ := 3

/-- The total number of volunteers -/
def total_volunteers : ℕ := num_males + num_females

/-- The maximum number of people allowed in a group -/
def max_group_size : ℕ := 5

/-- A function to calculate the number of valid allocation plans -/
def count_valid_allocations : ℕ :=
  let three_four_split := (Nat.choose total_volunteers 3 - 1) * 2
  let two_five_split := (Nat.choose total_volunteers 2 - Nat.choose num_females 2) * 2
  three_four_split + two_five_split

/-- Theorem stating that the number of valid allocation plans is 104 -/
theorem valid_allocations_count : count_valid_allocations = 104 := by
  sorry


end NUMINAMATH_CALUDE_valid_allocations_count_l464_46495


namespace NUMINAMATH_CALUDE_rectangle_length_l464_46434

theorem rectangle_length (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  (2*l + 2*w) / w = 5 → l * w = 150 → l = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l464_46434


namespace NUMINAMATH_CALUDE_function_symmetric_about_two_lines_is_periodic_l464_46489

/-- Given a function f: ℝ → ℝ that is symmetric about x = a and x = b (where a ≠ b),
    prove that f is periodic with period 2b - 2a. -/
theorem function_symmetric_about_two_lines_is_periodic
  (f : ℝ → ℝ) (a b : ℝ) (h_neq : a ≠ b)
  (h_sym_a : ∀ x, f (a - x) = f (a + x))
  (h_sym_b : ∀ x, f (b - x) = f (b + x)) :
  ∀ x, f x = f (x + 2*b - 2*a) :=
sorry

end NUMINAMATH_CALUDE_function_symmetric_about_two_lines_is_periodic_l464_46489


namespace NUMINAMATH_CALUDE_sets_problem_l464_46478

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B : Set ℝ := {x | 0 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ 2*m}

-- Theorem statement
theorem sets_problem :
  (∀ x : ℝ, x ∈ A ∩ B ↔ 4 ≤ x ∧ x < 5) ∧
  (∀ x : ℝ, x ∈ (Set.univ \ A) ∪ B ↔ -1 < x ∧ x < 5) ∧
  (∀ m : ℝ, B ∩ C m = C m ↔ m < -2 ∨ (2 < m ∧ m < 5/2)) :=
sorry

end NUMINAMATH_CALUDE_sets_problem_l464_46478


namespace NUMINAMATH_CALUDE_light_travel_distance_l464_46476

/-- The distance light travels in one year (in miles) -/
def light_year : ℕ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℕ := 500

/-- The expected distance light travels in 500 years (in miles) -/
def expected_distance : ℕ := 2935 * (10^12)

theorem light_travel_distance :
  (light_year * years : ℕ) = expected_distance := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l464_46476


namespace NUMINAMATH_CALUDE_distance_after_walk_l464_46408

/-- The distance from the starting point after walking 5 miles east, 
    turning 45 degrees north, and walking 7 miles. -/
theorem distance_after_walk (east_distance : ℝ) (angle : ℝ) (final_distance : ℝ) 
  (h1 : east_distance = 5)
  (h2 : angle = 45)
  (h3 : final_distance = 7) : 
  Real.sqrt (74 + 35 * Real.sqrt 2) = 
    Real.sqrt ((east_distance + final_distance * Real.sqrt 2 / 2) ^ 2 + 
               (final_distance * Real.sqrt 2 / 2) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_distance_after_walk_l464_46408


namespace NUMINAMATH_CALUDE_ellipse_and_chord_properties_l464_46428

-- Define the ellipse C₁
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the line l₂
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 2)

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ ∧ line x₂ y₂

theorem ellipse_and_chord_properties :
  -- The ellipse equation is correct
  (∀ x y, ellipse x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  -- The chord length is correct
  (∀ x₁ y₁ x₂ y₂, intersection_points x₁ y₁ x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 6 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_chord_properties_l464_46428


namespace NUMINAMATH_CALUDE_exponent_equality_l464_46444

theorem exponent_equality (a b c d : ℝ) (x y z q : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b) 
  (h3 : c^(4*y) = a^(5*z)) 
  (h4 : c^(4*y) = d) : 
  2*x * 5*z = 3*q * 4*y := by
sorry

end NUMINAMATH_CALUDE_exponent_equality_l464_46444


namespace NUMINAMATH_CALUDE_prob_all_fives_four_dice_l464_46426

-- Define a standard six-sided die
def standard_die := Finset.range 6

-- Define the probability of getting a specific number on a standard die
def prob_specific_number (die : Finset Nat) : ℚ :=
  1 / die.card

-- Define the number of dice
def num_dice : Nat := 4

-- Define the desired outcome (all fives)
def all_fives (n : Nat) : Bool := n = 5

-- Theorem: The probability of getting all fives on four standard six-sided dice is 1/1296
theorem prob_all_fives_four_dice : 
  (prob_specific_number standard_die) ^ num_dice = 1 / 1296 :=
sorry

end NUMINAMATH_CALUDE_prob_all_fives_four_dice_l464_46426


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l464_46441

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌈(-34 : ℝ) / 4⌉⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l464_46441


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l464_46491

/-- Given a circle with equation x^2 + y^2 - 6x + 14y = -28, 
    the sum of the x-coordinate and y-coordinate of its center is -4 -/
theorem circle_center_coordinate_sum : 
  ∃ (h k : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 6*x + 14*y = -28 ↔ (x - h)^2 + (y - k)^2 = 30) ∧ h + k = -4 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l464_46491


namespace NUMINAMATH_CALUDE_functional_equation_problem_l464_46459

theorem functional_equation_problem (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + g y) = -x + y + 1) :
  ∀ x y : ℝ, g (x + f y) = -x + y - 1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l464_46459


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l464_46438

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 →
  n ∈ Finset.range 1982 →
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l464_46438


namespace NUMINAMATH_CALUDE_max_sphere_radius_in_glass_l464_46466

theorem max_sphere_radius_in_glass (x : ℝ) :
  let r := (3 * 2^(1/3)) / 4
  let glass_curve := fun x => x^4
  let sphere_equation := fun (x y : ℝ) => x^2 + (y - r)^2 = r^2
  (∃ y, y = glass_curve x ∧ sphere_equation x y) ∧
  (∀ r' > r, ∃ x y, y < glass_curve x ∧ x^2 + (y - r')^2 = r'^2) ∧
  sphere_equation 0 0 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_radius_in_glass_l464_46466


namespace NUMINAMATH_CALUDE_fifth_term_is_89_l464_46477

def sequence_rule (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n + 1) = (seq n + seq (n + 2)) / 3

theorem fifth_term_is_89 (seq : ℕ → ℕ) (h_rule : sequence_rule seq) 
  (h_first : seq 1 = 2) (h_fourth : seq 4 = 34) : seq 5 = 89 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_89_l464_46477


namespace NUMINAMATH_CALUDE_book_profit_rate_l464_46480

/-- Calculate the rate of profit given cost price and selling price -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at Rs 50 and sold at Rs 90 is 80% -/
theorem book_profit_rate : rate_of_profit 50 90 = 80 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_rate_l464_46480


namespace NUMINAMATH_CALUDE_hermione_badges_l464_46402

theorem hermione_badges (total luna celestia : ℕ) (h1 : total = 83) (h2 : luna = 17) (h3 : celestia = 52) :
  total - luna - celestia = 14 := by
  sorry

end NUMINAMATH_CALUDE_hermione_badges_l464_46402


namespace NUMINAMATH_CALUDE_root_of_equation_l464_46481

theorem root_of_equation (x : ℝ) : 
  (2 * x^3 - 3 * x^2 - 13 * x + 10) * (x - 1) = 0 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_of_equation_l464_46481


namespace NUMINAMATH_CALUDE_people_who_left_gym_l464_46407

theorem people_who_left_gym (initial_people : ℕ) (people_came_in : ℕ) (current_people : ℕ)
  (h1 : initial_people = 16)
  (h2 : people_came_in = 5)
  (h3 : current_people = 19) :
  initial_people + people_came_in - current_people = 2 := by
sorry

end NUMINAMATH_CALUDE_people_who_left_gym_l464_46407


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_div_i_l464_46467

-- Define the complex number z
def z : ℂ := Complex.mk 1 (-3)

-- Theorem statement
theorem imaginary_part_of_z_div_i : (z / Complex.I).im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_div_i_l464_46467


namespace NUMINAMATH_CALUDE_cross_section_properties_l464_46474

/-- Regular triangular prism with given dimensions -/
structure RegularTriangularPrism where
  base_side_length : ℝ
  height : ℝ

/-- Cross-section of the prism -/
structure CrossSection where
  area : ℝ
  angle_with_base : ℝ

/-- Theorem about the cross-section of a specific regular triangular prism -/
theorem cross_section_properties (prism : RegularTriangularPrism) 
  (h1 : prism.base_side_length = 6)
  (h2 : prism.height = (1/3) * Real.sqrt 7) :
  ∃ (cs : CrossSection), cs.area = 39/4 ∧ cs.angle_with_base = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_properties_l464_46474


namespace NUMINAMATH_CALUDE_fraction_ordering_l464_46409

theorem fraction_ordering : (8 : ℚ) / 24 < 6 / 17 ∧ 6 / 17 < 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l464_46409


namespace NUMINAMATH_CALUDE_smallest_odd_polygon_is_seven_l464_46436

/-- A polygon with an odd number of sides that can be divided into parallelograms -/
structure OddPolygon where
  sides : ℕ
  is_odd : Odd sides
  divisible_into_parallelograms : Bool

/-- The smallest number of sides for an OddPolygon -/
def smallest_odd_polygon_sides : ℕ := 7

/-- Theorem stating that the smallest number of sides for an OddPolygon is 7 -/
theorem smallest_odd_polygon_is_seven :
  ∀ (p : OddPolygon), p.divisible_into_parallelograms → p.sides ≥ smallest_odd_polygon_sides :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_polygon_is_seven_l464_46436


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l464_46404

/-- Given a hyperbola C with equation (x^2 / a^2) - (y^2 / b^2) = 1, where a > 0, b > 0, 
    and eccentricity √10, its asymptotes are y = ±3x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := Real.sqrt 10  -- eccentricity
  (∀ p ∈ C, (p.1^2 / a^2 - p.2^2 / b^2 = 1)) →
  (e^2 = (a^2 + b^2) / a^2) →
  (∃ k : ℝ, k = b / a ∧ 
    (∀ (x y : ℝ), (x, y) ∈ C → (y = k * x ∨ y = -k * x)) ∧
    k = 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l464_46404


namespace NUMINAMATH_CALUDE_modular_arithmetic_expression_l464_46483

theorem modular_arithmetic_expression : (240 * 15 - 33 * 8 + 6) % 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_expression_l464_46483


namespace NUMINAMATH_CALUDE_triangle_properties_l464_46457

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  t.c = Real.sqrt 7 ∧
  4 * (Real.sin ((t.A + t.B) / 2))^2 - Real.cos (2 * t.C) = 7/2 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ (1/2 * t.a * t.b * Real.sin t.C) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l464_46457


namespace NUMINAMATH_CALUDE_boat_license_count_l464_46430

/-- The number of possible letters for a boat license -/
def num_letters : Nat := 3

/-- The number of possible digits for each position in a boat license -/
def num_digits : Nat := 10

/-- The number of digit positions in a boat license -/
def num_positions : Nat := 5

/-- The total number of possible boat licenses -/
def total_licenses : Nat := num_letters * (num_digits ^ num_positions)

theorem boat_license_count : total_licenses = 300000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l464_46430


namespace NUMINAMATH_CALUDE_sum_base6_numbers_l464_46442

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The sum of the given base 6 numbers equals 3153₆ --/
theorem sum_base6_numbers :
  let n1 := [4, 3, 2, 1]  -- 1234₆
  let n2 := [4, 5, 6]     -- 654₆
  let n3 := [1, 2, 3]     -- 321₆
  let n4 := [6, 5]        -- 56₆
  base10ToBase6 (base6ToBase10 n1 + base6ToBase10 n2 + base6ToBase10 n3 + base6ToBase10 n4) = [3, 1, 5, 3] :=
by sorry

end NUMINAMATH_CALUDE_sum_base6_numbers_l464_46442


namespace NUMINAMATH_CALUDE_alex_bill_correct_l464_46482

/-- Calculates the cell phone bill based on the given parameters. -/
def calculate_bill (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
                   (discount : ℚ) (texts_sent : ℕ) (hours_talked : ℕ) : ℚ :=
  let text_charge := text_cost * texts_sent
  let extra_minutes := max (hours_talked * 60 - 25 * 60) 0
  let extra_minute_charge := extra_minute_cost * extra_minutes
  let subtotal := base_cost + text_charge + extra_minute_charge
  let final_bill := if hours_talked > 35 then subtotal - discount else subtotal
  final_bill

theorem alex_bill_correct :
  calculate_bill 30 0.1 0.12 5 150 36 = 119.2 := by
  sorry

end NUMINAMATH_CALUDE_alex_bill_correct_l464_46482


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_squares_l464_46499

theorem geometric_arithmetic_sequence_sum_squares (x y z : ℝ) 
  (h1 : (4*y)^2 = (3*x)*(5*z))  -- Geometric sequence condition
  (h2 : y^2 = (x^2 + z^2)/2)    -- Arithmetic sequence condition
  : x^2 + z^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_squares_l464_46499


namespace NUMINAMATH_CALUDE_max_value_3x_4y_l464_46446

theorem max_value_3x_4y (x y : ℝ) : 
  y^2 = (1 - x) * (1 + x) → 
  ∃ (M : ℝ), M = 5 ∧ ∀ (x' y' : ℝ), y'^2 = (1 - x') * (1 + x') → 3*x' + 4*y' ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_3x_4y_l464_46446


namespace NUMINAMATH_CALUDE_simplify_expression_l464_46449

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l464_46449


namespace NUMINAMATH_CALUDE_excellent_chinese_or_math_excellent_all_subjects_l464_46412

def excellent_chinese : Finset ℕ := sorry
def excellent_math : Finset ℕ := sorry
def excellent_english : Finset ℕ := sorry

axiom total_excellent : (excellent_chinese ∪ excellent_math ∪ excellent_english).card = 18
axiom chinese_count : excellent_chinese.card = 9
axiom math_count : excellent_math.card = 11
axiom english_count : excellent_english.card = 8
axiom chinese_math_count : (excellent_chinese ∩ excellent_math).card = 5
axiom math_english_count : (excellent_math ∩ excellent_english).card = 3
axiom chinese_english_count : (excellent_chinese ∩ excellent_english).card = 4

theorem excellent_chinese_or_math : 
  (excellent_chinese ∪ excellent_math).card = 15 := by sorry

theorem excellent_all_subjects : 
  (excellent_chinese ∩ excellent_math ∩ excellent_english).card = 2 := by sorry

end NUMINAMATH_CALUDE_excellent_chinese_or_math_excellent_all_subjects_l464_46412


namespace NUMINAMATH_CALUDE_min_draws_for_eighteen_l464_46488

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  purple : Nat

/-- The minimum number of balls needed to guarantee at least n of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in our problem -/
def problemCounts : BallCounts :=
  { red := 34, green := 25, yellow := 18, blue := 21, purple := 13 }

/-- The theorem stating the minimum number of draws needed -/
theorem min_draws_for_eighteen (counts : BallCounts) :
  counts = problemCounts → minDraws counts 18 = 82 :=
  sorry

end NUMINAMATH_CALUDE_min_draws_for_eighteen_l464_46488


namespace NUMINAMATH_CALUDE_larger_number_ratio_l464_46416

theorem larger_number_ratio (m v : ℝ) (h1 : m < v) (h2 : v - m/4 = 5*(3*m/4)) : v = 4*m := by
  sorry

end NUMINAMATH_CALUDE_larger_number_ratio_l464_46416


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_twentyseven_equals_negative_sqrt_three_l464_46433

theorem sqrt_twelve_minus_sqrt_twentyseven_equals_negative_sqrt_three :
  Real.sqrt 12 - Real.sqrt 27 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_twentyseven_equals_negative_sqrt_three_l464_46433


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l464_46496

theorem arithmetic_series_sum (A : ℕ) : A = 380 := by
  sorry

#check arithmetic_series_sum

end NUMINAMATH_CALUDE_arithmetic_series_sum_l464_46496


namespace NUMINAMATH_CALUDE_complex_magnitude_l464_46463

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 15)
  (h2 : Complex.abs (2 * z + 3 * w) = 10)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = 4.5 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l464_46463


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l464_46487

/-- Given a rectangle with sides a and b (in decimeters), prove that its perimeter is 20 decimeters
    if the sum of two sides is 10 and the sum of three sides is 14. -/
theorem rectangle_perimeter (a b : ℝ) : 
  a + b = 10 → a + a + b = 14 → 2 * (a + b) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l464_46487


namespace NUMINAMATH_CALUDE_problem_solution_l464_46458

theorem problem_solution (A B : Set ℝ) (a b : ℝ) : 
  A = {2, 3} →
  B = {x : ℝ | x^2 + a*x + b = 0} →
  A ∩ B = {2} →
  A ∪ B = A →
  (a + b = 0 ∨ a + b = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l464_46458


namespace NUMINAMATH_CALUDE_sum_between_15_and_16_l464_46448

theorem sum_between_15_and_16 : 
  let a : ℚ := 10/3
  let b : ℚ := 19/4
  let c : ℚ := 123/20
  15 < a + b + c ∧ a + b + c < 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_between_15_and_16_l464_46448


namespace NUMINAMATH_CALUDE_polynomial_simplification_l464_46418

theorem polynomial_simplification (x : ℝ) : 
  (3*x^2 + 4*x + 6)*(x-2) - (x-2)*(x^2 + 5*x - 72) + (2*x - 7)*(x-2)*(x+4) = 
  4*x^3 - 8*x^2 + 50*x - 100 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l464_46418


namespace NUMINAMATH_CALUDE_twenty_four_is_forty_eight_percent_of_fifty_l464_46490

theorem twenty_four_is_forty_eight_percent_of_fifty :
  ∃ x : ℝ, (24 : ℝ) / x = 48 / 100 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_is_forty_eight_percent_of_fifty_l464_46490


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l464_46472

/-- The compound interest formula: A = P * (1 + r)^n -/
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem interest_rate_calculation
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Rate of interest (as a decimal)
  (h1 : compound_interest P r 3 = 800)
  (h2 : compound_interest P r 4 = 820) :
  r = 0.025 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l464_46472


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l464_46437

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 where a > 0,
    if one of its asymptote equations is y = -2x, then a = 1 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2/a^2 - y^2/4 = 1 ∧ y = -2*x) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l464_46437


namespace NUMINAMATH_CALUDE_inequality_condition_l464_46492

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∀ x y, x < y → x ≤ 2 → y ≤ 2 → f x < f y)

/-- The main theorem -/
theorem inequality_condition (f : ℝ → ℝ) (a : ℝ) 
  (h : special_function f) : 
  f (a^2 + 3*a + 2) < f (a^2 - a + 2) ↔ a > -1 ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l464_46492


namespace NUMINAMATH_CALUDE_vertex_y_is_negative_three_l464_46479

/-- Quadratic function f(x) = 2x^2 - 4x - 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 1

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

theorem vertex_y_is_negative_three :
  f vertex_x = -3 := by
  sorry

end NUMINAMATH_CALUDE_vertex_y_is_negative_three_l464_46479


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l464_46420

/-- A regular octagon with a square constructed outward on one side -/
structure RegularOctagonWithSquare where
  /-- The vertices of the octagon -/
  vertices : Fin 8 → ℝ × ℝ
  /-- The square constructed outward on one side -/
  square : Fin 4 → ℝ × ℝ
  /-- The octagon is regular -/
  regular : ∀ i j : Fin 8, dist (vertices i) (vertices ((i + 1) % 8)) = dist (vertices j) (vertices ((j + 1) % 8))
  /-- The square is connected to the octagon -/
  square_connected : ∃ i : Fin 8, square 0 = vertices i ∧ square 1 = vertices ((i + 1) % 8)

/-- Point B where two diagonals intersect inside the octagon -/
def intersection_point (o : RegularOctagonWithSquare) : ℝ × ℝ := sorry

/-- Angle ABC in the octagon -/
def angle_ABC (o : RegularOctagonWithSquare) : ℝ := sorry

/-- Theorem: The measure of angle ABC is 22.5° -/
theorem angle_ABC_measure (o : RegularOctagonWithSquare) : angle_ABC o = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l464_46420


namespace NUMINAMATH_CALUDE_gcf_of_45_135_60_l464_46451

theorem gcf_of_45_135_60 : Nat.gcd 45 (Nat.gcd 135 60) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_45_135_60_l464_46451


namespace NUMINAMATH_CALUDE_expression_evaluation_l464_46497

theorem expression_evaluation : 
  let a := (1/4 + 1/12 - 7/18 - 1/36)
  let b := 1/36
  (b / a + a / b) = -10/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l464_46497


namespace NUMINAMATH_CALUDE_negation_of_existence_l464_46473

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l464_46473


namespace NUMINAMATH_CALUDE_unique_pythagorean_triple_l464_46424

/-- A function to check if a triple of natural numbers is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The theorem stating that (5, 12, 13) is the only Pythagorean triple among the given options -/
theorem unique_pythagorean_triple :
  ¬ isPythagoreanTriple 3 4 5 ∧
  ¬ isPythagoreanTriple 1 1 2 ∧
  isPythagoreanTriple 5 12 13 ∧
  ¬ isPythagoreanTriple 1 3 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_pythagorean_triple_l464_46424


namespace NUMINAMATH_CALUDE_fifth_color_count_l464_46421

/-- Represents the number of marbles of each color in a box -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  fifth : ℕ

/-- Defines the properties of the marble counts as given in the problem -/
def valid_marble_count (m : MarbleCount) : Prop :=
  m.red = 25 ∧
  m.green = 3 * m.red ∧
  m.yellow = m.green / 5 ∧
  m.blue = 2 * m.yellow ∧
  m.fifth = (m.red + m.blue) + (m.red + m.blue) / 2 ∧
  m.red + m.green + m.yellow + m.blue + m.fifth = 4 * m.green

theorem fifth_color_count (m : MarbleCount) (h : valid_marble_count m) : m.fifth = 155 := by
  sorry

end NUMINAMATH_CALUDE_fifth_color_count_l464_46421


namespace NUMINAMATH_CALUDE_integer_triples_theorem_l464_46454

def satisfies_conditions (a b c : ℤ) : Prop :=
  a + b + c = 24 ∧ a^2 + b^2 + c^2 = 210 ∧ a * b * c = 440

def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(11, 8, 5), (8, 11, 5), (8, 5, 11), (5, 8, 11), (11, 5, 8), (5, 11, 8)}

theorem integer_triples_theorem :
  ∀ (a b c : ℤ), satisfies_conditions a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_integer_triples_theorem_l464_46454


namespace NUMINAMATH_CALUDE_range_of_f_l464_46498

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 1)

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, x^2 ≠ 1 ∧ f x = y} = {y : ℝ | y < 0 ∨ y > 0} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l464_46498


namespace NUMINAMATH_CALUDE_robin_gum_packages_l464_46484

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := 15

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := 135

/-- The number of packages Robin has -/
def num_packages : ℕ := total_pieces / pieces_per_package

theorem robin_gum_packages : num_packages = 9 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l464_46484


namespace NUMINAMATH_CALUDE_locus_properties_l464_46455

/-- The locus of point R in a specific geometric configuration -/
def locus_equation (a b c x y : ℝ) : Prop :=
  b^2 * x^2 - 2*a*b*x*y + a*(a - c)*y^2 - b^2*c*x + 2*a*b*c*y = 0

/-- The type of curve represented by the locus equation -/
inductive CurveType
  | Ellipse
  | Hyperbola

/-- Theorem stating the properties of the locus and its curve type -/
theorem locus_properties (a b c : ℝ) (h1 : b > 0) (h2 : c > 0) (h3 : a ≠ c) :
  ∃ (curve_type : CurveType),
    (∀ x y : ℝ, locus_equation a b c x y) ∧
    ((a < 0 → curve_type = CurveType.Ellipse) ∧
     (a > 0 → curve_type = CurveType.Hyperbola)) :=
by sorry

end NUMINAMATH_CALUDE_locus_properties_l464_46455


namespace NUMINAMATH_CALUDE_combined_molecular_weight_l464_46417

/-- Atomic weight of Carbon in g/mol -/
def carbon_weight : Float := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : Float := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def oxygen_weight : Float := 16.00

/-- Molecular weight of Butanoic acid (C4H8O2) in g/mol -/
def butanoic_weight : Float :=
  4 * carbon_weight + 8 * hydrogen_weight + 2 * oxygen_weight

/-- Molecular weight of Propanoic acid (C3H6O2) in g/mol -/
def propanoic_weight : Float :=
  3 * carbon_weight + 6 * hydrogen_weight + 2 * oxygen_weight

/-- Number of moles of Butanoic acid in the mixture -/
def butanoic_moles : Float := 9

/-- Number of moles of Propanoic acid in the mixture -/
def propanoic_moles : Float := 5

/-- Theorem: The combined molecular weight of the mixture is 1163.326 grams -/
theorem combined_molecular_weight :
  butanoic_moles * butanoic_weight + propanoic_moles * propanoic_weight = 1163.326 := by
  sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_l464_46417


namespace NUMINAMATH_CALUDE_industrial_park_investment_l464_46485

theorem industrial_park_investment
  (total_investment : ℝ)
  (return_rate_A : ℝ)
  (return_rate_B : ℝ)
  (total_return : ℝ)
  (h1 : total_investment = 2000)
  (h2 : return_rate_A = 0.054)
  (h3 : return_rate_B = 0.0828)
  (h4 : total_return = 122.4)
  : ∃ (investment_A investment_B : ℝ),
    investment_A + investment_B = total_investment ∧
    investment_A * return_rate_A + investment_B * return_rate_B = total_return ∧
    investment_A = 1500 ∧
    investment_B = 500 := by
  sorry

end NUMINAMATH_CALUDE_industrial_park_investment_l464_46485


namespace NUMINAMATH_CALUDE_point_on_x_axis_with_distance_3_from_y_axis_l464_46439

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance function from a point to the y-axis
def distanceToYAxis (p : Point2D) : ℝ := |p.x|

-- State the theorem
theorem point_on_x_axis_with_distance_3_from_y_axis 
  (P : Point2D) 
  (h1 : P.y = 0)  -- P is on the x-axis
  (h2 : distanceToYAxis P = 3) : 
  (P.x = 3 ∧ P.y = 0) ∨ (P.x = -3 ∧ P.y = 0) := by
  sorry


end NUMINAMATH_CALUDE_point_on_x_axis_with_distance_3_from_y_axis_l464_46439


namespace NUMINAMATH_CALUDE_geometric_series_sum_l464_46419

theorem geometric_series_sum : ∀ (a r : ℚ), 
  a = 1 → r = 1/3 → abs r < 1 → 
  (∑' n, a * r^n) = 3/2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l464_46419


namespace NUMINAMATH_CALUDE_thirteen_divides_six_digit_reverse_perm_l464_46452

/-- A 6-digit positive integer whose first three digits are a permutation of its last three digits taken in reverse order. -/
def SixDigitReversePerm : Type :=
  {n : ℕ // 100000 ≤ n ∧ n < 1000000 ∧ ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ a ≠ 0 ∧
    (n = 100000*a + 10000*b + 1000*c + 100*c + 10*b + a ∨
     n = 100000*a + 10000*c + 1000*b + 100*b + 10*c + a ∨
     n = 100000*b + 10000*a + 1000*c + 100*c + 10*a + b ∨
     n = 100000*b + 10000*c + 1000*a + 100*a + 10*c + b ∨
     n = 100000*c + 10000*a + 1000*b + 100*b + 10*a + c ∨
     n = 100000*c + 10000*b + 1000*a + 100*a + 10*b + c)}

theorem thirteen_divides_six_digit_reverse_perm (x : SixDigitReversePerm) :
  13 ∣ x.val :=
sorry

end NUMINAMATH_CALUDE_thirteen_divides_six_digit_reverse_perm_l464_46452


namespace NUMINAMATH_CALUDE_range_of_m_l464_46400

-- Define the conditions
def p (x : ℝ) : Prop := (x - 2) * (x - 6) ≤ 32
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  (0 < m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l464_46400


namespace NUMINAMATH_CALUDE_inequality_implication_l464_46410

theorem inequality_implication (a b : ℝ) (h : a > b) : b - a < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l464_46410


namespace NUMINAMATH_CALUDE_tan_585_degrees_l464_46445

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_585_degrees_l464_46445
