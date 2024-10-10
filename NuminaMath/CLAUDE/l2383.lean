import Mathlib

namespace function_change_proof_l2383_238318

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The initial x value -/
def x_initial : ℝ := 2

/-- The final x value -/
def x_final : ℝ := 2.5

/-- The change in x -/
def delta_x : ℝ := x_final - x_initial

theorem function_change_proof :
  f x_final - f x_initial = 2.25 := by
  sorry

end function_change_proof_l2383_238318


namespace abs_sum_lt_sum_abs_iff_product_neg_l2383_238354

theorem abs_sum_lt_sum_abs_iff_product_neg (a b : ℝ) :
  |a + b| < |a| + |b| ↔ a * b < 0 :=
sorry

end abs_sum_lt_sum_abs_iff_product_neg_l2383_238354


namespace cube_volume_surface_area_l2383_238374

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s^3 = 8*x^2 ∧ 6*s^2 = 4*x) → x = 1/216 := by
  sorry

end cube_volume_surface_area_l2383_238374


namespace two_thousand_eighth_number_without_two_l2383_238315

/-- A function that checks if a positive integer contains the digit 2 -/
def containsTwo (n : ℕ) : Bool :=
  String.contains (toString n) '2'

/-- A function that generates the sequence of numbers without 2 -/
def sequenceWithoutTwo : ℕ → ℕ
  | 0 => 0
  | n + 1 => if containsTwo (sequenceWithoutTwo n + 1)
              then sequenceWithoutTwo n + 2
              else sequenceWithoutTwo n + 1

theorem two_thousand_eighth_number_without_two :
  sequenceWithoutTwo 2008 = 3781 := by
  sorry

end two_thousand_eighth_number_without_two_l2383_238315


namespace cube_of_product_equality_l2383_238377

theorem cube_of_product_equality (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by
  sorry

end cube_of_product_equality_l2383_238377


namespace centimeters_per_kilometer_l2383_238331

-- Define the conversion factors
def meters_per_kilometer : ℝ := 1000
def centimeters_per_meter : ℝ := 100

-- Theorem statement
theorem centimeters_per_kilometer : 
  meters_per_kilometer * centimeters_per_meter = 100000 := by
  sorry

end centimeters_per_kilometer_l2383_238331


namespace prob_at_least_one_value_l2383_238329

/-- The probability of picking a road from A to B that is at least 5 miles long -/
def prob_AB : ℚ := 2/3

/-- The probability of picking a road from B to C that is at least 5 miles long -/
def prob_BC : ℚ := 3/4

/-- The probability that at least one of the randomly picked roads (one from A to B, one from B to C) is at least 5 miles long -/
def prob_at_least_one : ℚ := 1 - (1 - prob_AB) * (1 - prob_BC)

theorem prob_at_least_one_value : prob_at_least_one = 11/12 := by
  sorry

end prob_at_least_one_value_l2383_238329


namespace chess_board_numbering_specific_cell_number_l2383_238309

/-- Represents the numbering system of an infinite chessboard where each cell
    is assigned the smallest possible number not yet used for numbering any
    preceding cells in the same row or column. -/
noncomputable def chessBoardNumber (row : Nat) (col : Nat) : Nat :=
  sorry

/-- The number assigned to a cell is equal to the XOR of (row - 1) and (col - 1) -/
theorem chess_board_numbering (row col : Nat) :
  chessBoardNumber row col = Nat.xor (row - 1) (col - 1) :=
sorry

/-- The cell at the intersection of the 100th row and the 1000th column
    receives the number 921 -/
theorem specific_cell_number :
  chessBoardNumber 100 1000 = 921 :=
sorry

end chess_board_numbering_specific_cell_number_l2383_238309


namespace completing_square_form_l2383_238370

theorem completing_square_form (x : ℝ) :
  x^2 - 2*x - 1 = 0 ↔ (x - 1)^2 = 2 :=
by sorry

end completing_square_form_l2383_238370


namespace geometric_sequence_increasing_condition_l2383_238365

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

/-- The condition "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  (GeometricSequence a q) →
  (¬(q > 1 → IncreasingSequence a) ∧ ¬(IncreasingSequence a → q > 1)) :=
sorry

end geometric_sequence_increasing_condition_l2383_238365


namespace diamond_operation_result_l2383_238304

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the diamond operation
def diamond : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.two
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.one
  | Element.four, Element.one => Element.two
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.three

theorem diamond_operation_result :
  diamond (diamond Element.three Element.one) (diamond Element.four Element.two) = Element.one := by
  sorry

end diamond_operation_result_l2383_238304


namespace sixth_power_sum_l2383_238356

theorem sixth_power_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 := by
  sorry

end sixth_power_sum_l2383_238356


namespace bowtie_equation_solution_l2383_238300

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 5 y = 15 ∧ y = 90 := by sorry

end bowtie_equation_solution_l2383_238300


namespace quartic_factorization_and_solutions_l2383_238338

theorem quartic_factorization_and_solutions :
  ∃ (x₁ x₂ x₃ x₄ : ℂ),
    (∀ x : ℂ, x^4 + 1 = (x^2 + Real.sqrt 2 * x + 1) * (x^2 - Real.sqrt 2 * x + 1)) ∧
    x₁ = -Real.sqrt 2 / 2 + Complex.I * Real.sqrt 2 / 2 ∧
    x₂ = -Real.sqrt 2 / 2 - Complex.I * Real.sqrt 2 / 2 ∧
    x₃ = Real.sqrt 2 / 2 + Complex.I * Real.sqrt 2 / 2 ∧
    x₄ = Real.sqrt 2 / 2 - Complex.I * Real.sqrt 2 / 2 ∧
    {x | x^4 + 1 = 0} = {x₁, x₂, x₃, x₄} := by
  sorry

end quartic_factorization_and_solutions_l2383_238338


namespace soda_cost_calculation_l2383_238393

/-- The cost of a single soda, given the total cost of sandwiches and sodas, and the cost of a single sandwich. -/
def soda_cost (total_cost sandwich_cost : ℚ) : ℚ :=
  (total_cost - 2 * sandwich_cost) / 4

theorem soda_cost_calculation (total_cost sandwich_cost : ℚ) 
  (h1 : total_cost = (8.36 : ℚ))
  (h2 : sandwich_cost = (2.44 : ℚ)) :
  soda_cost total_cost sandwich_cost = (0.87 : ℚ) := by
  sorry

end soda_cost_calculation_l2383_238393


namespace D_most_stable_l2383_238376

-- Define the variances for each person
def variance_A : ℝ := 0.56
def variance_B : ℝ := 0.60
def variance_C : ℝ := 0.50
def variance_D : ℝ := 0.45

-- Define a function to determine if one variance is more stable than another
def more_stable (v1 v2 : ℝ) : Prop := v1 < v2

-- Theorem stating that D has the most stable performance
theorem D_most_stable :
  more_stable variance_D variance_C ∧
  more_stable variance_D variance_A ∧
  more_stable variance_D variance_B :=
by sorry

end D_most_stable_l2383_238376


namespace line_intersects_circle_l2383_238319

/-- The line ax-y+2a=0 (a∈R) intersects the circle x^2+y^2=5 -/
theorem line_intersects_circle (a : ℝ) : 
  ∃ (x y : ℝ), (a * x - y + 2 * a = 0) ∧ (x^2 + y^2 = 5) := by
  sorry

end line_intersects_circle_l2383_238319


namespace greatest_rope_piece_length_l2383_238386

theorem greatest_rope_piece_length : Nat.gcd 48 (Nat.gcd 60 72) = 12 := by
  sorry

end greatest_rope_piece_length_l2383_238386


namespace exists_polygon_with_1980_degrees_l2383_238330

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- 1980 degrees is a valid sum of interior angles for some polygon -/
theorem exists_polygon_with_1980_degrees :
  ∃ (n : ℕ), sum_interior_angles n = 1980 :=
sorry

end exists_polygon_with_1980_degrees_l2383_238330


namespace count_arrangements_eq_21_l2383_238301

/-- A function that counts the number of valid arrangements of digits 1, 1, 2, 5, 0 -/
def countArrangements : ℕ :=
  let digits : List ℕ := [1, 1, 2, 5, 0]
  let isValidArrangement (arr : List ℕ) : Bool :=
    arr.length = 5 ∧ 
    arr.head? ≠ some 0 ∧ 
    (arr.getLast? = some 0 ∨ arr.getLast? = some 5)

  -- Count valid arrangements
  sorry

/-- The theorem stating that the number of valid arrangements is 21 -/
theorem count_arrangements_eq_21 : countArrangements = 21 := by
  sorry

end count_arrangements_eq_21_l2383_238301


namespace semicircle_radius_l2383_238317

/-- The radius of a semi-circle given its perimeter -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 113) : 
  ∃ (radius : ℝ), radius = perimeter / (Real.pi + 2) := by
  sorry

end semicircle_radius_l2383_238317


namespace largest_divisor_of_difference_of_squares_l2383_238339

theorem largest_divisor_of_difference_of_squares (m n : ℤ) :
  (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) →  -- m and n are even
  n < m →                              -- n is less than m
  (∀ x : ℤ, x ∣ (m^2 - n^2) → x ≤ 8) ∧ -- 8 is an upper bound for divisors
  8 ∣ (m^2 - n^2)                      -- 8 divides m^2 - n^2
  := by sorry

end largest_divisor_of_difference_of_squares_l2383_238339


namespace sampling_methods_appropriateness_l2383_238357

/-- Represents a sampling scenario with a population size and sample size -/
structure SamplingScenario where
  populationSize : Nat
  sampleSize : Nat

/-- Determines if simple random sampling is appropriate for a given scenario -/
def isSimpleRandomSamplingAppropriate (scenario : SamplingScenario) : Prop :=
  scenario.sampleSize ≤ scenario.populationSize ∧ scenario.sampleSize ≤ 10

/-- Determines if systematic sampling is appropriate for a given scenario -/
def isSystematicSamplingAppropriate (scenario : SamplingScenario) : Prop :=
  scenario.sampleSize > 10 ∧ scenario.populationSize ≥ 100

theorem sampling_methods_appropriateness :
  let scenario1 : SamplingScenario := ⟨10, 2⟩
  let scenario2 : SamplingScenario := ⟨1000, 50⟩
  isSimpleRandomSamplingAppropriate scenario1 ∧
  isSystematicSamplingAppropriate scenario2 :=
by sorry

end sampling_methods_appropriateness_l2383_238357


namespace edmund_earns_64_dollars_l2383_238303

/-- Calculates the amount Edmund earns for extra chores over two weeks -/
def edmunds_earnings (normal_chores_per_week : ℕ) (chores_per_day : ℕ) 
  (days : ℕ) (payment_per_extra_chore : ℕ) : ℕ :=
  let total_chores := chores_per_day * days
  let normal_total_chores := normal_chores_per_week * (days / 7)
  let extra_chores := total_chores - normal_total_chores
  extra_chores * payment_per_extra_chore

/-- Theorem stating that Edmund earns $64 for extra chores over two weeks -/
theorem edmund_earns_64_dollars :
  edmunds_earnings 12 4 14 2 = 64 := by
  sorry


end edmund_earns_64_dollars_l2383_238303


namespace test_score_problem_l2383_238340

theorem test_score_problem (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (total_score : ℤ) :
  total_questions = 30 →
  correct_points = 3 →
  incorrect_points = 1 →
  total_score = 78 →
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 27 :=
by sorry

end test_score_problem_l2383_238340


namespace cube_volume_from_side_area_l2383_238326

theorem cube_volume_from_side_area :
  ∀ (side_area : ℝ) (volume : ℝ),
    side_area = 64 →
    volume = (side_area.sqrt) ^ 3 →
    volume = 512 := by
  sorry

end cube_volume_from_side_area_l2383_238326


namespace inequality_proof_l2383_238307

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 / (2 * x)) + (1 / (2 * y)) + (1 / (2 * z)) > (1 / (y + z)) + (1 / (z + x)) + (1 / (x + y)) := by
  sorry

end inequality_proof_l2383_238307


namespace runner_stops_in_quarter_A_l2383_238368

def track_circumference : ℝ := 80
def distance_run : ℝ := 2000
def num_quarters : ℕ := 4

theorem runner_stops_in_quarter_A :
  ∀ (start_point : ℝ) (quarters : Fin num_quarters),
  start_point ∈ Set.Icc 0 track_circumference →
  ∃ (n : ℕ), distance_run = n * track_circumference + start_point :=
by sorry

end runner_stops_in_quarter_A_l2383_238368


namespace max_ac_value_l2383_238389

theorem max_ac_value (a b c d : ℤ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : c > d) 
  (h4 : d ≥ -2021) 
  (h5 : (a + b) * (d + a) = (b + c) * (c + d)) 
  (h6 : b + c ≠ 0) 
  (h7 : d + a ≠ 0) : 
  a * c ≤ 510050 := by
  sorry

end max_ac_value_l2383_238389


namespace base8_subtraction_l2383_238355

-- Define a function to convert base 8 to decimal
def base8ToDecimal (n : ℕ) : ℕ := sorry

-- Define a function to convert decimal to base 8
def decimalToBase8 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_subtraction : base8ToDecimal 52 - base8ToDecimal 27 = base8ToDecimal 23 := by
  sorry

end base8_subtraction_l2383_238355


namespace complex_square_roots_l2383_238358

theorem complex_square_roots (z : ℂ) : 
  z ^ 2 = -115 + 66 * I ↔ z = 3 + 11 * I ∨ z = -3 - 11 * I := by
  sorry

end complex_square_roots_l2383_238358


namespace one_third_of_390_l2383_238372

theorem one_third_of_390 : (1 / 3 : ℚ) * 390 = 130 := by sorry

end one_third_of_390_l2383_238372


namespace division_problem_l2383_238349

theorem division_problem : (160 : ℝ) / (10 + 11 * 2) = 5 := by
  sorry

end division_problem_l2383_238349


namespace clay_target_permutations_l2383_238333

theorem clay_target_permutations : 
  (Nat.factorial 9) / ((Nat.factorial 3) * (Nat.factorial 3) * (Nat.factorial 3)) = 1680 := by
  sorry

end clay_target_permutations_l2383_238333


namespace min_value_of_expression_l2383_238350

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a ≥ 6 ∧
  ((a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a = 6 ↔ a = b ∧ b = c) :=
by sorry

end min_value_of_expression_l2383_238350


namespace completing_square_result_l2383_238347

theorem completing_square_result (x : ℝ) :
  x^2 - 4*x + 2 = 0 → (x - 2)^2 = 2 := by
sorry

end completing_square_result_l2383_238347


namespace arithmetic_square_root_of_negative_five_squared_l2383_238324

theorem arithmetic_square_root_of_negative_five_squared (x : ℝ) : 
  x = 5 ∧ x * x = (-5)^2 ∧ x ≥ 0 :=
sorry

end arithmetic_square_root_of_negative_five_squared_l2383_238324


namespace sum_interior_angles_pentagon_l2383_238328

-- Define a pentagon as a polygon with 5 sides
def Pentagon : Nat := 5

-- Theorem stating that the sum of interior angles of a pentagon is 540 degrees
theorem sum_interior_angles_pentagon :
  (Pentagon - 2) * 180 = 540 := by
  sorry

end sum_interior_angles_pentagon_l2383_238328


namespace three_distinct_triangles60_l2383_238312

/-- A triangle with integer side lengths and one 60° angle -/
structure Triangle60 where
  a : ℕ
  b : ℕ
  c : ℕ
  coprime : Nat.gcd a (Nat.gcd b c) = 1
  angle60 : a^2 + b^2 + c^2 = 2 * max a (max b c)^2

/-- The existence of at least three distinct triangles with integer side lengths and one 60° angle -/
theorem three_distinct_triangles60 : ∃ (t1 t2 t3 : Triangle60),
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
  (t1.a, t1.b, t1.c) ≠ (5, 7, 8) ∧
  (t2.a, t2.b, t2.c) ≠ (5, 7, 8) ∧
  (t3.a, t3.b, t3.c) ≠ (5, 7, 8) :=
sorry

end three_distinct_triangles60_l2383_238312


namespace absolute_value_equation_implies_zero_product_l2383_238306

theorem absolute_value_equation_implies_zero_product (x y : ℝ) (hy : y > 0) :
  |x - Real.log (y^2)| = x + Real.log (y^2) → x * (y - 1)^2 = 0 := by
  sorry

end absolute_value_equation_implies_zero_product_l2383_238306


namespace andy_math_problem_l2383_238332

theorem andy_math_problem (start_num end_num count : ℕ) : 
  end_num = 125 → count = 46 → end_num - start_num + 1 = count → start_num = 80 := by
  sorry

end andy_math_problem_l2383_238332


namespace special_number_value_l2383_238367

/-- A number with specified digits in certain decimal places -/
def SpecialNumber : ℝ :=
  60 + 0.06

/-- Proof that the SpecialNumber is equal to 60.06 -/
theorem special_number_value : SpecialNumber = 60.06 := by
  sorry

#check special_number_value

end special_number_value_l2383_238367


namespace rhombus_perimeter_l2383_238375

/-- A rhombus with given diagonal lengths has the specified perimeter. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side_length := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side_length = 52 := by sorry

end rhombus_perimeter_l2383_238375


namespace binomial_coefficient_equality_l2383_238387

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 8 x = Nat.choose 8 (2*x - 1)) → (x = 1 ∨ x = 3) :=
by sorry

end binomial_coefficient_equality_l2383_238387


namespace even_iff_divisible_by_72_l2383_238308

theorem even_iff_divisible_by_72 (n : ℕ) : 
  Even n ↔ 72 ∣ (3^n + 63) := by sorry

end even_iff_divisible_by_72_l2383_238308


namespace min_cases_for_shirley_order_l2383_238346

/-- Represents the number of boxes of each cookie type sold -/
structure CookiesSold where
  trefoils : Nat
  samoas : Nat
  thinMints : Nat

/-- Represents the composition of each case -/
structure CaseComposition where
  trefoils : Nat
  samoas : Nat
  thinMints : Nat

/-- Calculates the minimum number of cases needed to fulfill the orders -/
def minCasesNeeded (sold : CookiesSold) (composition : CaseComposition) : Nat :=
  max
    (((sold.trefoils + composition.trefoils - 1) / composition.trefoils) : Nat)
    (max
      (((sold.samoas + composition.samoas - 1) / composition.samoas) : Nat)
      (((sold.thinMints + composition.thinMints - 1) / composition.thinMints) : Nat))

theorem min_cases_for_shirley_order :
  let sold : CookiesSold := { trefoils := 54, samoas := 36, thinMints := 48 }
  let composition : CaseComposition := { trefoils := 4, samoas := 3, thinMints := 5 }
  minCasesNeeded sold composition = 14 := by
  sorry

end min_cases_for_shirley_order_l2383_238346


namespace fraction_sum_equality_l2383_238323

theorem fraction_sum_equality : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end fraction_sum_equality_l2383_238323


namespace complex_fraction_real_condition_l2383_238385

theorem complex_fraction_real_condition (a : ℝ) : 
  (((1 : ℂ) + 2 * I) / (a + I)).im = 0 ↔ a = (1/2 : ℝ) :=
sorry

end complex_fraction_real_condition_l2383_238385


namespace min_value_reciprocal_sum_l2383_238359

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
  sorry

end min_value_reciprocal_sum_l2383_238359


namespace possible_m_values_l2383_238396

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | x * m + 1 = 0}

theorem possible_m_values : 
  {m : ℝ | B m ⊆ A} = {-1/2, 0, 1/3} := by sorry

end possible_m_values_l2383_238396


namespace complex_subtraction_l2383_238394

theorem complex_subtraction : (7 - 3*Complex.I) - (2 + 5*Complex.I) = 5 - 8*Complex.I := by
  sorry

end complex_subtraction_l2383_238394


namespace loads_to_wash_l2383_238321

theorem loads_to_wash (total : ℕ) (washed : ℕ) (h1 : total = 14) (h2 : washed = 8) :
  total - washed = 6 := by
  sorry

end loads_to_wash_l2383_238321


namespace function_value_at_half_l2383_238314

def real_function_property (f : ℝ → ℝ) : Prop :=
  f 1 = -1 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

theorem function_value_at_half (f : ℝ → ℝ) (h : real_function_property f) : f (1/2) = -2 := by
  sorry

end function_value_at_half_l2383_238314


namespace probability_of_drawing_red_ball_l2383_238390

theorem probability_of_drawing_red_ball (white_balls red_balls : ℕ) 
  (h1 : white_balls = 5) (h2 : red_balls = 2) : 
  (red_balls : ℚ) / (white_balls + red_balls) = 2 / 7 :=
by sorry

end probability_of_drawing_red_ball_l2383_238390


namespace problem_statement_l2383_238391

theorem problem_statement (p q : Prop) 
  (h1 : p ∨ q) 
  (h2 : ¬(p ∧ q)) 
  (h3 : ¬p) : 
  ¬p ∧ q := by
  sorry

end problem_statement_l2383_238391


namespace dress_price_difference_l2383_238398

/-- Proves that the final price of a dress is $2.3531875 more than the original price
    given specific discounts, increases, and taxes. -/
theorem dress_price_difference (original_price : ℝ) : 
  original_price * 0.85 = 78.2 →
  let price_after_sale := 78.2
  let price_after_increase := price_after_sale * 1.25
  let price_after_coupon := price_after_increase * 0.9
  let final_price := price_after_coupon * 1.0725
  final_price - original_price = 2.3531875 := by
  sorry

end dress_price_difference_l2383_238398


namespace finite_zero_additions_l2383_238336

/-- Represents the state of the board at any given time -/
def BoardState := List ℕ

/-- The process of updating the board -/
def update_board (a b : ℕ) (state : BoardState) : BoardState :=
  sorry

/-- Predicate to check if we need to add two zeros -/
def need_zeros (state : BoardState) : Prop :=
  sorry

/-- The main theorem statement -/
theorem finite_zero_additions (a b : ℕ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    let state := (update_board a b)^[n] []
    ¬(need_zeros state) :=
  sorry

end finite_zero_additions_l2383_238336


namespace total_pictures_taken_l2383_238361

def pictures_already_taken : ℕ := 28
def pictures_at_dolphin_show : ℕ := 16

theorem total_pictures_taken : 
  pictures_already_taken + pictures_at_dolphin_show = 44 := by
  sorry

end total_pictures_taken_l2383_238361


namespace triangle_angle_measure_l2383_238313

theorem triangle_angle_measure (A B C : ℝ) (exterior_angle : ℝ) :
  -- An exterior angle of triangle ABC is 110°
  exterior_angle = 110 →
  -- ∠A = ∠B
  A = B →
  -- Triangle inequality (to ensure it's a valid triangle)
  A + B + C = 180 →
  -- Prove that ∠A is either 70° or 55°
  A = 70 ∨ A = 55 := by
  sorry

end triangle_angle_measure_l2383_238313


namespace honey_lasts_16_days_l2383_238371

/-- Represents the number of days Tabitha can enjoy honey in her tea -/
def honey_days : ℕ :=
  let evening_servings : ℕ := 2 * 2  -- 2 cups with 2 servings each
  let morning_servings : ℕ := 1 * 1  -- 1 cup with 1 serving
  let afternoon_servings : ℕ := 1 * 1  -- 1 cup with 1 serving
  let daily_servings : ℕ := evening_servings + morning_servings + afternoon_servings
  let container_ounces : ℕ := 16
  let servings_per_ounce : ℕ := 6
  let total_servings : ℕ := container_ounces * servings_per_ounce
  total_servings / daily_servings

theorem honey_lasts_16_days : honey_days = 16 := by
  sorry

end honey_lasts_16_days_l2383_238371


namespace triangle_preserves_triangle_parallelogram_preserves_parallelogram_square_not_always_square_rhombus_not_always_rhombus_l2383_238399

-- Define the oblique projection method
structure ObliqueProjection where
  -- Add necessary fields for oblique projection

-- Define geometric shapes
structure Triangle where
  -- Add necessary fields for triangle

structure Parallelogram where
  -- Add necessary fields for parallelogram

structure Square where
  -- Add necessary fields for square

structure Rhombus where
  -- Add necessary fields for rhombus

-- Define the intuitive diagram function
def intuitiveDiagram (op : ObliqueProjection) (shape : Type) : Type :=
  sorry

-- Theorem statements
theorem triangle_preserves_triangle (op : ObliqueProjection) (t : Triangle) :
  intuitiveDiagram op Triangle = Triangle :=
sorry

theorem parallelogram_preserves_parallelogram (op : ObliqueProjection) (p : Parallelogram) :
  intuitiveDiagram op Parallelogram = Parallelogram :=
sorry

theorem square_not_always_square (op : ObliqueProjection) :
  ¬(∀ (s : Square), intuitiveDiagram op Square = Square) :=
sorry

theorem rhombus_not_always_rhombus (op : ObliqueProjection) :
  ¬(∀ (r : Rhombus), intuitiveDiagram op Rhombus = Rhombus) :=
sorry

end triangle_preserves_triangle_parallelogram_preserves_parallelogram_square_not_always_square_rhombus_not_always_rhombus_l2383_238399


namespace cycle_price_calculation_l2383_238316

/-- Calculates the total amount a buyer pays for a cycle, given the initial cost,
    loss percentage, and sales tax percentage. -/
def totalCyclePrice (initialCost : ℚ) (lossPercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let sellingPrice := initialCost * (1 - lossPercentage / 100)
  let salesTax := sellingPrice * (salesTaxPercentage / 100)
  sellingPrice + salesTax

/-- Theorem stating that for a cycle bought at Rs. 1400, sold at 20% loss,
    with 5% sales tax, the total price is Rs. 1176. -/
theorem cycle_price_calculation :
  totalCyclePrice 1400 20 5 = 1176 := by
  sorry

end cycle_price_calculation_l2383_238316


namespace angle_C_measure_l2383_238335

structure Quadrilateral where
  A : Real
  B : Real
  C : Real
  D : Real
  sum_angles : A + B + C + D = 360

def adjacent_angle_ratio (q : Quadrilateral) : Prop :=
  q.A / q.B = 2 / 7

theorem angle_C_measure (q : Quadrilateral) (h : adjacent_angle_ratio q) : q.C = 40 := by
  sorry

end angle_C_measure_l2383_238335


namespace letters_with_both_count_l2383_238352

/-- Represents the number of letters in the alphabet. -/
def total_letters : ℕ := 40

/-- Represents the number of letters with a straight line but no dot. -/
def line_only : ℕ := 24

/-- Represents the number of letters with a dot but no straight line. -/
def dot_only : ℕ := 6

/-- Represents the number of letters with both a dot and a straight line. -/
def both : ℕ := total_letters - line_only - dot_only

theorem letters_with_both_count :
  both = 10 :=
sorry

end letters_with_both_count_l2383_238352


namespace cyclist_distance_difference_l2383_238384

/-- The difference in miles traveled between two cyclists after 3 hours -/
theorem cyclist_distance_difference (carlos_start : ℝ) (carlos_total : ℝ) (dana_total : ℝ) : 
  carlos_start = 5 → 
  carlos_total = 50 → 
  dana_total = 40 → 
  carlos_total - dana_total = 10 := by
  sorry

#check cyclist_distance_difference

end cyclist_distance_difference_l2383_238384


namespace max_net_revenue_l2383_238380

/-- Represents the net revenue function for a movie theater --/
def net_revenue (x : ℕ) : ℤ :=
  if x ≤ 10 then 1000 * x - 5750
  else -30 * x * x + 1300 * x - 5750

/-- Theorem stating the maximum net revenue and optimal ticket price --/
theorem max_net_revenue :
  ∃ (max_revenue : ℕ) (optimal_price : ℕ),
    max_revenue = 8830 ∧
    optimal_price = 22 ∧
    (∀ (x : ℕ), x ≥ 6 → x ≤ 38 → net_revenue x ≤ net_revenue optimal_price) :=
by sorry

end max_net_revenue_l2383_238380


namespace polynomial_coefficient_sum_l2383_238341

theorem polynomial_coefficient_sum (d : ℝ) (h : d ≠ 0) :
  let p := (10 * d + 17 + 12 * d^2) + (6 * d + 3)
  ∃ a b c : ℤ, p = a * d + b + c * d^2 ∧ a + b + c = 48 := by
  sorry

end polynomial_coefficient_sum_l2383_238341


namespace geometric_sequence_sum_l2383_238351

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The main theorem: For a geometric sequence satisfying given conditions, a₂ + a₆ = 34 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsGeometricSequence a →
  a 3 + a 5 = 20 →
  a 4 = 8 →
  a 2 + a 6 = 34 := by
  sorry


end geometric_sequence_sum_l2383_238351


namespace license_plate_count_l2383_238382

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def num_plate_digits : ℕ := 5

/-- The number of letters in a license plate -/
def num_plate_letters : ℕ := 2

/-- The number of possible positions for the letter block -/
def num_letter_block_positions : ℕ := num_plate_digits + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := num_digits ^ num_plate_digits * (num_letters ^ num_plate_letters) * num_letter_block_positions

theorem license_plate_count :
  total_license_plates = 40560000 := by
  sorry

end license_plate_count_l2383_238382


namespace cookie_orange_cost_ratio_l2383_238311

/-- Cost of items in Susie and Calvin's purchases -/
structure ItemCosts where
  orange : ℚ
  muffin : ℚ
  cookie : ℚ

/-- Susie's purchase -/
def susie_purchase (costs : ItemCosts) : ℚ :=
  3 * costs.muffin + 5 * costs.orange

/-- Calvin's purchase -/
def calvin_purchase (costs : ItemCosts) : ℚ :=
  5 * costs.muffin + 10 * costs.orange + 4 * costs.cookie

theorem cookie_orange_cost_ratio :
  ∀ (costs : ItemCosts),
  costs.muffin = 2 * costs.orange →
  calvin_purchase costs = 3 * susie_purchase costs →
  costs.cookie = (13/4) * costs.orange :=
by sorry

end cookie_orange_cost_ratio_l2383_238311


namespace cube_minimizes_edge_sum_squares_l2383_238378

/-- A parallelepiped with edges a, b, c and volume V -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  V : ℝ
  volume_eq : a * b * c = V
  positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- The sum of squares of edges meeting at one vertex -/
def edge_sum_squares (p : Parallelepiped) : ℝ := p.a^2 + p.b^2 + p.c^2

/-- Theorem: The cube minimizes the sum of squares of edges among parallelepipeds of equal volume -/
theorem cube_minimizes_edge_sum_squares (V : ℝ) (hV : 0 < V) :
  ∀ p : Parallelepiped, p.V = V →
  edge_sum_squares p ≥ 3 * V^(2/3) ∧
  (edge_sum_squares p = 3 * V^(2/3) ↔ p.a = p.b ∧ p.b = p.c) :=
sorry


end cube_minimizes_edge_sum_squares_l2383_238378


namespace mall_product_properties_l2383_238395

/-- Represents the shopping mall's product pricing and sales model -/
structure ProductModel where
  purchase_price : ℝ
  min_selling_price : ℝ
  max_selling_price : ℝ
  sales_volume : ℝ → ℝ
  profit : ℝ → ℝ

/-- The specific product model for the shopping mall -/
def mall_product : ProductModel :=
  { purchase_price := 30
    min_selling_price := 30
    max_selling_price := 55
    sales_volume := λ x => -2 * x + 140
    profit := λ x => (x - 30) * (-2 * x + 140) }

theorem mall_product_properties (x : ℝ) :
  let m := mall_product
  (x ≥ m.min_selling_price ∧ x ≤ m.max_selling_price) →
  (m.profit 35 = 350 ∧
   m.profit 40 = 600 ∧
   ∀ y, m.min_selling_price ≤ y ∧ y ≤ m.max_selling_price → m.profit y ≠ 900) :=
by sorry


end mall_product_properties_l2383_238395


namespace quadratic_factorization_l2383_238397

theorem quadratic_factorization (x : ℝ) : 
  (x^2 - 4*x + 3 = (x-1)*(x-3)) ∧ 
  (4*x^2 + 12*x - 7 = (2*x+7)*(2*x-1)) := by
  sorry

#check quadratic_factorization

end quadratic_factorization_l2383_238397


namespace quadratic_complete_square_l2383_238334

theorem quadratic_complete_square :
  ∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by sorry

end quadratic_complete_square_l2383_238334


namespace mobile_phone_cost_l2383_238373

def refrigerator_cost : ℝ := 15000
def refrigerator_loss_percent : ℝ := 0.04
def mobile_profit_percent : ℝ := 0.10
def overall_profit : ℝ := 200

theorem mobile_phone_cost (mobile_cost : ℝ) : 
  (refrigerator_cost * (1 - refrigerator_loss_percent) + mobile_cost * (1 + mobile_profit_percent)) - 
  (refrigerator_cost + mobile_cost) = overall_profit →
  mobile_cost = 6000 := by
sorry

end mobile_phone_cost_l2383_238373


namespace trig_inequality_l2383_238392

theorem trig_inequality (a b c x : ℝ) :
  -(Real.sin (π/4 - (b-c)/2))^2 ≤ Real.sin (a*x + b) * Real.cos (a*x + c) ∧
  Real.sin (a*x + b) * Real.cos (a*x + c) ≤ (Real.cos (π/4 - (b-c)/2))^2 := by
  sorry

end trig_inequality_l2383_238392


namespace det_matrix_2x2_l2383_238305

theorem det_matrix_2x2 (A : Matrix (Fin 2) (Fin 2) ℝ) : 
  A = ![![6, -2], ![-5, 3]] → Matrix.det A = 8 := by
  sorry

end det_matrix_2x2_l2383_238305


namespace candy_count_l2383_238322

/-- The number of candy pieces caught by Tabitha and her friends -/
def total_candy (tabitha stan julie carlos veronica benjamin kelly : ℕ) : ℕ :=
  tabitha + stan + julie + carlos + veronica + benjamin + kelly

/-- Theorem stating the total number of candy pieces caught by the friends -/
theorem candy_count : ∃ (tabitha stan julie carlos veronica benjamin kelly : ℕ),
  tabitha = 22 ∧
  stan = tabitha / 3 + 4 ∧
  julie = tabitha / 2 ∧
  carlos = 2 * stan ∧
  veronica = julie + stan - 5 ∧
  benjamin = (tabitha + carlos) / 2 + 9 ∧
  kelly = stan * julie / tabitha ∧
  total_candy tabitha stan julie carlos veronica benjamin kelly = 119 := by
  sorry

#check candy_count

end candy_count_l2383_238322


namespace pave_hall_l2383_238381

/-- Calculates the number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  let hall_area := hall_length * hall_width * 100
  let stone_area := stone_length * stone_width
  (hall_area / stone_area).ceil.toNat

/-- Theorem stating that 5400 stones are required to pave the given hall -/
theorem pave_hall : stones_required 36 15 2 5 = 5400 := by
  sorry

end pave_hall_l2383_238381


namespace ellipse_string_length_l2383_238342

/-- Represents an ellipse with semi-major axis 'a' and semi-minor axis 'b' --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_a_ge_b : a ≥ b

/-- The length of the string used in the pin-and-string method for drawing an ellipse --/
def string_length (e : Ellipse) : ℝ := 2 * e.a

/-- Theorem stating that for an ellipse with semi-major axis 6 cm and semi-minor axis 4 cm,
    the length of the string used in the pin-and-string method is 12 cm --/
theorem ellipse_string_length :
  let e : Ellipse := ⟨6, 4, by norm_num, by norm_num, by norm_num⟩
  string_length e = 12 := by sorry

end ellipse_string_length_l2383_238342


namespace ellipse_focus_coordinates_l2383_238337

/-- Given an ellipse with specified major and minor axis endpoints, 
    prove that the focus with the smaller y-coordinate has coordinates (5 - √5, 2) -/
theorem ellipse_focus_coordinates 
  (major_endpoint1 : ℝ × ℝ)
  (major_endpoint2 : ℝ × ℝ)
  (minor_endpoint1 : ℝ × ℝ)
  (minor_endpoint2 : ℝ × ℝ)
  (h1 : major_endpoint1 = (2, 2))
  (h2 : major_endpoint2 = (8, 2))
  (h3 : minor_endpoint1 = (5, 4))
  (h4 : minor_endpoint2 = (5, 0)) :
  ∃ (focus : ℝ × ℝ), focus = (5 - Real.sqrt 5, 2) ∧ 
  (∀ (other_focus : ℝ × ℝ), other_focus.2 ≤ focus.2) :=
by sorry

end ellipse_focus_coordinates_l2383_238337


namespace incorrect_statement_l2383_238310

theorem incorrect_statement : ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end incorrect_statement_l2383_238310


namespace moe_eating_time_l2383_238327

/-- The time taken for Moe to eat a certain number of cuttlebone pieces -/
theorem moe_eating_time (X : ℝ) : 
  (200 : ℝ) / 800 * X = X / 4 := by sorry

end moe_eating_time_l2383_238327


namespace log_equation_holds_l2383_238364

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 := by
  sorry

end log_equation_holds_l2383_238364


namespace correct_calculation_l2383_238383

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end correct_calculation_l2383_238383


namespace intersecting_squares_area_difference_l2383_238302

theorem intersecting_squares_area_difference :
  let A : ℝ := 12^2
  let B : ℝ := 9^2
  let C : ℝ := 7^2
  let D : ℝ := 3^2
  ∀ (E F G : ℝ),
  (A + E - (B + F)) - (C + G - (B + D + F)) = 103 :=
by sorry

end intersecting_squares_area_difference_l2383_238302


namespace max_distance_between_spheres_max_distance_achieved_l2383_238325

def sphere1_center : ℝ × ℝ × ℝ := (-4, -10, 5)
def sphere1_radius : ℝ := 20

def sphere2_center : ℝ × ℝ × ℝ := (10, 7, -16)
def sphere2_radius : ℝ := 90

theorem max_distance_between_spheres :
  ∀ (p1 p2 : ℝ × ℝ × ℝ),
  ‖p1 - sphere1_center‖ = sphere1_radius →
  ‖p2 - sphere2_center‖ = sphere2_radius →
  ‖p1 - p2‖ ≤ 140.433 :=
by sorry

theorem max_distance_achieved :
  ∃ (p1 p2 : ℝ × ℝ × ℝ),
  ‖p1 - sphere1_center‖ = sphere1_radius ∧
  ‖p2 - sphere2_center‖ = sphere2_radius ∧
  ‖p1 - p2‖ = 140.433 :=
by sorry

end max_distance_between_spheres_max_distance_achieved_l2383_238325


namespace train_speed_l2383_238363

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length crossing_time : Real) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 235.03)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45.0036 := by
  sorry

#eval (140 + 235.03) / 30 * 3.6

end train_speed_l2383_238363


namespace pythagorean_triple_6_8_10_l2383_238369

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_6_8_10 :
  is_pythagorean_triple 6 8 10 ∧
  ¬ is_pythagorean_triple 1 1 2 ∧
  ¬ is_pythagorean_triple 1 2 2 ∧
  ¬ is_pythagorean_triple 5 12 15 :=
sorry

end pythagorean_triple_6_8_10_l2383_238369


namespace unit_circle_sector_angle_l2383_238344

/-- The radian measure of a central angle in a unit circle, given the area of the sector -/
def central_angle (area : ℝ) : ℝ := 2 * area

theorem unit_circle_sector_angle (area : ℝ) (h : area = 1) : 
  central_angle area = 2 := by
  sorry

end unit_circle_sector_angle_l2383_238344


namespace lemonade_solution_water_amount_l2383_238348

/-- The amount of lemonade syrup in the solution -/
def lemonade_syrup : ℝ := 7

/-- The amount of solution removed and replaced with water -/
def removed_amount : ℝ := 2.1428571428571423

/-- The desired concentration of lemonade syrup after adjustment -/
def desired_concentration : ℝ := 0.20

/-- The amount of water in the original solution -/
def water_amount : ℝ := 25.857142857142854

theorem lemonade_solution_water_amount :
  (lemonade_syrup / (lemonade_syrup + water_amount + removed_amount) = desired_concentration) :=
by sorry

end lemonade_solution_water_amount_l2383_238348


namespace sum_ages_after_ten_years_l2383_238320

/-- Given Ann's age and Tom's age relative to Ann's, calculate the sum of their ages after a certain number of years. -/
def sum_ages_after_years (ann_age : ℕ) (tom_age_multiplier : ℕ) (years_later : ℕ) : ℕ :=
  (ann_age + years_later) + (ann_age * tom_age_multiplier + years_later)

/-- Theorem stating that given Ann's age is 6 and Tom's age is twice Ann's, the sum of their ages 10 years later is 38. -/
theorem sum_ages_after_ten_years :
  sum_ages_after_years 6 2 10 = 38 := by
  sorry

end sum_ages_after_ten_years_l2383_238320


namespace lineman_drinks_eight_ounces_l2383_238379

/-- Represents the water consumption scenario of a football team -/
structure WaterConsumption where
  linemen_count : ℕ
  skill_players_count : ℕ
  cooler_capacity : ℕ
  skill_player_consumption : ℕ
  waiting_skill_players : ℕ

/-- Calculates the amount of water each lineman drinks -/
def lineman_consumption (wc : WaterConsumption) : ℚ :=
  let skill_players_drinking := wc.skill_players_count - wc.waiting_skill_players
  let total_skill_consumption := skill_players_drinking * wc.skill_player_consumption
  (wc.cooler_capacity - total_skill_consumption) / wc.linemen_count

/-- Theorem stating that each lineman drinks 8 ounces of water -/
theorem lineman_drinks_eight_ounces (wc : WaterConsumption) 
  (h1 : wc.linemen_count = 12)
  (h2 : wc.skill_players_count = 10)
  (h3 : wc.cooler_capacity = 126)
  (h4 : wc.skill_player_consumption = 6)
  (h5 : wc.waiting_skill_players = 5) :
  lineman_consumption wc = 8 := by
  sorry

#eval lineman_consumption {
  linemen_count := 12,
  skill_players_count := 10,
  cooler_capacity := 126,
  skill_player_consumption := 6,
  waiting_skill_players := 5
}

end lineman_drinks_eight_ounces_l2383_238379


namespace bryans_annual_travel_time_l2383_238345

/-- Represents the time in minutes for each leg of Bryan's journey --/
structure JourneyTime where
  walkToBus : ℕ
  busRide : ℕ
  walkToJob : ℕ

/-- Calculates the total annual travel time in hours --/
def annualTravelTime (j : JourneyTime) (daysWorked : ℕ) : ℕ :=
  let totalDailyMinutes := 2 * (j.walkToBus + j.busRide + j.walkToJob)
  (totalDailyMinutes * daysWorked) / 60

/-- Theorem stating that Bryan spends 365 hours per year traveling to and from work --/
theorem bryans_annual_travel_time :
  let j : JourneyTime := { walkToBus := 5, busRide := 20, walkToJob := 5 }
  annualTravelTime j 365 = 365 := by
  sorry

end bryans_annual_travel_time_l2383_238345


namespace new_device_improvement_l2383_238388

/-- Represents the sample mean and variance of a device's measurements -/
structure DeviceStats where
  mean : ℝ
  variance : ℝ

/-- Determines if there's significant improvement based on the given criterion -/
def significantImprovement (old new : DeviceStats) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

/-- The old device's statistics -/
def oldDevice : DeviceStats :=
  { mean := 10.3, variance := 0.04 }

/-- The new device's statistics -/
def newDevice : DeviceStats :=
  { mean := 10, variance := 0.036 }

/-- Theorem stating that the new device shows significant improvement -/
theorem new_device_improvement :
  significantImprovement oldDevice newDevice := by
  sorry

#check new_device_improvement

end new_device_improvement_l2383_238388


namespace elena_pen_purchase_l2383_238366

/-- The number of brand X pens Elena purchased -/
def brand_x_pens : ℕ := 9

/-- The number of brand Y pens Elena purchased -/
def brand_y_pens : ℕ := 12 - brand_x_pens

/-- The cost of a single brand X pen -/
def cost_x : ℚ := 4

/-- The cost of a single brand Y pen -/
def cost_y : ℚ := 2.2

/-- The total cost of all pens -/
def total_cost : ℚ := 42

theorem elena_pen_purchase :
  (brand_x_pens : ℚ) * cost_x + (brand_y_pens : ℚ) * cost_y = total_cost ∧
  brand_x_pens + brand_y_pens = 12 :=
by sorry

end elena_pen_purchase_l2383_238366


namespace min_value_theorem_l2383_238353

theorem min_value_theorem (x y : ℝ) :
  (y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 ≥ 1/6 ∧
  ((y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 = 1/6 ↔ x = 5/2 ∧ y = 5/6) :=
by sorry

end min_value_theorem_l2383_238353


namespace basketball_league_games_l2383_238343

/-- The number of games played in a basketball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264. -/
theorem basketball_league_games :
  total_games 12 4 = 264 := by
  sorry

end basketball_league_games_l2383_238343


namespace middle_circle_radius_l2383_238362

/-- Given three circles in a geometric sequence with radii r₁, r₂, and r₃,
    where r₁ = 5 cm and r₃ = 20 cm, prove that r₂ = 10 cm. -/
theorem middle_circle_radius (r₁ r₂ r₃ : ℝ) 
    (h_geom_seq : r₂^2 = r₁ * r₃)
    (h_r₁ : r₁ = 5)
    (h_r₃ : r₃ = 20) : 
  r₂ = 10 := by
  sorry

end middle_circle_radius_l2383_238362


namespace binomial_expansion_example_l2383_238360

theorem binomial_expansion_example : 104^3 + 3*(104^2)*2 + 3*104*(2^2) + 2^3 = 106^3 := by
  sorry

end binomial_expansion_example_l2383_238360
