import Mathlib

namespace NUMINAMATH_CALUDE_product_no_linear_quadratic_terms_l2365_236566

theorem product_no_linear_quadratic_terms 
  (p q : ℚ) 
  (h : ∀ x : ℚ, (x + 3*p) * (x^2 - x + 1/3*q) = x^3 + p*q) : 
  p = 1/3 ∧ q = 3 ∧ p^2020 * q^2021 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_no_linear_quadratic_terms_l2365_236566


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_one_l2365_236526

def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem intersection_nonempty_implies_a_greater_than_one (a : ℝ) :
  (A ∩ C a).Nonempty → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_one_l2365_236526


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l2365_236515

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem max_sum_arithmetic_sequence
  (a₁ : ℚ)
  (h1 : a₁ = 13)
  (h2 : sum_arithmetic_sequence a₁ d 3 = sum_arithmetic_sequence a₁ d 11) :
  ∃ (n : ℕ), ∀ (m : ℕ), sum_arithmetic_sequence a₁ d n ≥ sum_arithmetic_sequence a₁ d m ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l2365_236515


namespace NUMINAMATH_CALUDE_cube_angle_sum_prove_cube_angle_sum_l2365_236586

/-- The sum of three right angles and one angle formed by a face diagonal in a cube is 330 degrees. -/
theorem cube_angle_sum : ℝ → Prop :=
  fun (cube_angle_sum : ℝ) =>
    let right_angle : ℝ := 90
    let face_diagonal_angle : ℝ := 60
    cube_angle_sum = 3 * right_angle + face_diagonal_angle ∧ cube_angle_sum = 330

/-- Proof of the theorem -/
theorem prove_cube_angle_sum : ∃ (x : ℝ), cube_angle_sum x :=
  sorry

end NUMINAMATH_CALUDE_cube_angle_sum_prove_cube_angle_sum_l2365_236586


namespace NUMINAMATH_CALUDE_f_inequality_l2365_236573

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, f (x + 1) = f (-(x + 1)))
variable (h2 : ∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y)
variable (x₁ x₂ : ℝ)
variable (h3 : x₁ < 0)
variable (h4 : x₂ > 0)
variable (h5 : x₁ + x₂ < -2)

-- State the theorem
theorem f_inequality : f (-x₁) > f (-x₂) := by sorry

end NUMINAMATH_CALUDE_f_inequality_l2365_236573


namespace NUMINAMATH_CALUDE_not_or_necessary_not_sufficient_for_not_and_l2365_236570

theorem not_or_necessary_not_sufficient_for_not_and (p q : Prop) :
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬(p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_not_or_necessary_not_sufficient_for_not_and_l2365_236570


namespace NUMINAMATH_CALUDE_pole_distance_difference_l2365_236548

theorem pole_distance_difference (h₁ h₂ d : ℝ) 
  (h_h₁ : h₁ = 6)
  (h_h₂ : h₂ = 11)
  (h_d : d = 12) : 
  Real.sqrt ((h₂ - h₁)^2 + d^2) - d = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_pole_distance_difference_l2365_236548


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l2365_236519

theorem fibonacci_like_sequence
  (a : ℕ → ℕ)  -- Sequence of natural numbers
  (seq_length : ℕ)  -- Length of the sequence
  (h_length : seq_length = 10)  -- The sequence has 10 numbers
  (h_rec : ∀ n, 3 ≤ n → n ≤ seq_length → a n = a (n-1) + a (n-2))  -- Recurrence relation
  (h_seventh : a 7 = 42)  -- The seventh number is 42
  (h_ninth : a 9 = 110)  -- The ninth number is 110
  : a 4 = 10 :=  -- The fourth number is 10
by sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l2365_236519


namespace NUMINAMATH_CALUDE_car_costs_theorem_l2365_236557

def cost_of_old_car : ℝ := 1800
def cost_of_second_oldest_car : ℝ := 900
def cost_of_new_car : ℝ := 2 * cost_of_old_car
def sale_price_old_car : ℝ := 1800
def sale_price_second_oldest_car : ℝ := 900
def loan_amount : ℝ := cost_of_new_car - (sale_price_old_car + sale_price_second_oldest_car)
def annual_interest_rate : ℝ := 0.05
def years_passed : ℝ := 2
def remaining_debt : ℝ := 2000

theorem car_costs_theorem :
  cost_of_old_car = 1800 ∧
  cost_of_second_oldest_car = 900 ∧
  cost_of_new_car = 2 * cost_of_old_car ∧
  cost_of_new_car = 4 * cost_of_second_oldest_car ∧
  sale_price_old_car = 1800 ∧
  sale_price_second_oldest_car = 900 ∧
  loan_amount = cost_of_new_car - (sale_price_old_car + sale_price_second_oldest_car) ∧
  remaining_debt = 2000 :=
by sorry

end NUMINAMATH_CALUDE_car_costs_theorem_l2365_236557


namespace NUMINAMATH_CALUDE_hilt_snow_amount_l2365_236559

/-- The amount of snow at Brecknock Elementary School in inches -/
def school_snow : ℕ := 17

/-- The additional amount of snow at Mrs. Hilt's house compared to the school in inches -/
def additional_snow : ℕ := 12

/-- The total amount of snow at Mrs. Hilt's house in inches -/
def hilt_snow : ℕ := school_snow + additional_snow

/-- Theorem stating that the amount of snow at Mrs. Hilt's house is 29 inches -/
theorem hilt_snow_amount : hilt_snow = 29 := by sorry

end NUMINAMATH_CALUDE_hilt_snow_amount_l2365_236559


namespace NUMINAMATH_CALUDE_divisibility_condition_l2365_236571

/-- A function that checks if a five-digit number in the form 4AB2B is divisible by 11 -/
def isDivisibleBy11 (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ (4 + a * 1000 + b * 100 + 2 * 10 + b) % 11 = 0

/-- Theorem stating the conditions for a number in the form 4AB2B to be divisible by 11 -/
theorem divisibility_condition :
  ∀ b : ℕ, b < 10 → isDivisibleBy11 6 b ∧ (∀ a : ℕ, a < 10 ∧ a ≠ 6 → ¬isDivisibleBy11 a b) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2365_236571


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2365_236524

theorem sum_of_squares_of_roots : ∃ (a b c d : ℝ),
  (∀ x : ℝ, x^4 - 15*x^2 + 56 = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) →
  a^2 + b^2 + c^2 + d^2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2365_236524


namespace NUMINAMATH_CALUDE_family_size_family_size_proof_l2365_236584

theorem family_size : ℕ → Prop :=
  fun n =>
    ∀ (b : ℕ),
      -- Peter has b brothers and 3b sisters
      (3 * b = n - b - 1) →
      -- Louise has b + 1 brothers and 3b - 1 sisters
      (3 * b - 1 = 2 * (b + 1)) →
      n = 13

-- The proof is omitted
theorem family_size_proof : family_size 13 := by sorry

end NUMINAMATH_CALUDE_family_size_family_size_proof_l2365_236584


namespace NUMINAMATH_CALUDE_spinner_probability_l2365_236532

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/2 →
  p_B = 1/8 →
  p_C = p_D →
  p_A + p_B + p_C + p_D = 1 →
  p_C = 3/16 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l2365_236532


namespace NUMINAMATH_CALUDE_range_of_a_l2365_236561

def p (a : ℝ) : Prop := ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.sqrt (a*x^2 - x + a)

theorem range_of_a :
  (∃ a, p a ∨ q a) ∧ (∀ a, ¬(p a ∧ q a)) →
  ∃ S : Set ℝ, S = {a | a ∈ (Set.Ioo 0 (1/2)) ∪ (Set.Ici 1)} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2365_236561


namespace NUMINAMATH_CALUDE_julia_normal_mile_time_l2365_236512

/-- Represents Julia's running times -/
structure JuliaRunningTimes where
  normalMileTime : ℝ
  newShoesMileTime : ℝ

/-- The conditions of the problem -/
def problemConditions (j : JuliaRunningTimes) : Prop :=
  j.newShoesMileTime = 13 ∧
  5 * j.newShoesMileTime = 5 * j.normalMileTime + 15

/-- The theorem stating Julia's normal mile time -/
theorem julia_normal_mile_time (j : JuliaRunningTimes) 
  (h : problemConditions j) : j.normalMileTime = 10 := by
  sorry


end NUMINAMATH_CALUDE_julia_normal_mile_time_l2365_236512


namespace NUMINAMATH_CALUDE_selection_with_condition_l2365_236503

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 10

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of students excluding the two specific students -/
def remaining_students : ℕ := total_students - 2

theorem selection_with_condition :
  (choose total_students selected_students) - (choose remaining_students selected_students) = 140 := by
  sorry

end NUMINAMATH_CALUDE_selection_with_condition_l2365_236503


namespace NUMINAMATH_CALUDE_midpoint_triangle_is_equilateral_l2365_236518

-- Define the points in the plane
variable (A B C D E F G M N P : ℝ × ℝ)

-- Define the conditions
def is_midpoint (M A B : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def is_equilateral_triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 ∧
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = (Z.1 - X.1)^2 + (Z.2 - X.2)^2

-- State the theorem
theorem midpoint_triangle_is_equilateral
  (h1 : is_midpoint M A B)
  (h2 : is_midpoint P G F)
  (h3 : is_midpoint N E F)
  (h4 : is_equilateral_triangle B C E)
  (h5 : is_equilateral_triangle C D F)
  (h6 : is_equilateral_triangle D A G) :
  is_equilateral_triangle M N P :=
sorry

end NUMINAMATH_CALUDE_midpoint_triangle_is_equilateral_l2365_236518


namespace NUMINAMATH_CALUDE_sum_18_29_base4_l2365_236565

/-- Converts a number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_18_29_base4 :
  toBase4 (18 + 29) = [2, 3, 3] :=
sorry

end NUMINAMATH_CALUDE_sum_18_29_base4_l2365_236565


namespace NUMINAMATH_CALUDE_large_loans_required_l2365_236522

/-- Represents the number of loans of each type required to buy an apartment -/
structure LoanCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Two equivalent ways to buy the apartment -/
def way1 : LoanCombination := { small := 9, medium := 6, large := 1 }
def way2 : LoanCombination := { small := 3, medium := 2, large := 3 }

/-- The theorem states that 4 large loans are required to buy the apartment -/
theorem large_loans_required : ∃ (n : ℕ), n = 4 ∧ 
  way1.small * n + way1.medium * n + way1.large * n = 
  way2.small * n + way2.medium * n + way2.large * n :=
sorry

end NUMINAMATH_CALUDE_large_loans_required_l2365_236522


namespace NUMINAMATH_CALUDE_girls_who_bought_balloons_l2365_236568

def initial_balloons : ℕ := 3 * 12
def boys_bought : ℕ := 3
def remaining_balloons : ℕ := 21

theorem girls_who_bought_balloons :
  initial_balloons - remaining_balloons - boys_bought = 12 :=
by sorry

end NUMINAMATH_CALUDE_girls_who_bought_balloons_l2365_236568


namespace NUMINAMATH_CALUDE_equation_proof_l2365_236544

theorem equation_proof : 42 / (7 - 4/3) = 126/17 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2365_236544


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_equals_one_l2365_236528

theorem fraction_to_zero_power_equals_one :
  (7342983001 / -195843720384 : ℚ) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_equals_one_l2365_236528


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l2365_236588

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 2x + 1 is 28 -/
theorem square_area_on_parabola : 
  ∃ (x₁ x₂ : ℝ),
    (x₁^2 + 2*x₁ + 1 = 7) ∧ 
    (x₂^2 + 2*x₂ + 1 = 7) ∧ 
    ((x₂ - x₁)^2 = 28) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l2365_236588


namespace NUMINAMATH_CALUDE_f_monotone_increasing_a_value_for_odd_function_l2365_236552

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_monotone_increasing (a : ℝ) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
sorry

theorem a_value_for_odd_function :
  (∀ x : ℝ, f a x = -f a (-x)) → a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_a_value_for_odd_function_l2365_236552


namespace NUMINAMATH_CALUDE_girls_total_distance_l2365_236564

/-- The number of laps run by the boys -/
def boys_laps : ℕ := 27

/-- The number of additional laps run by the girls compared to the boys -/
def extra_girls_laps : ℕ := 9

/-- The length of each boy's lap in miles -/
def boys_lap_length : ℚ := 3/4

/-- The length of the first type of girl's lap in miles -/
def girls_lap_length1 : ℚ := 3/4

/-- The length of the second type of girl's lap in miles -/
def girls_lap_length2 : ℚ := 7/8

/-- The total number of laps run by the girls -/
def girls_laps : ℕ := boys_laps + extra_girls_laps

/-- The number of laps of each type run by the girls -/
def girls_laps_each_type : ℕ := girls_laps / 2

theorem girls_total_distance :
  girls_laps_each_type * girls_lap_length1 + girls_laps_each_type * girls_lap_length2 = 29.25 := by
  sorry

end NUMINAMATH_CALUDE_girls_total_distance_l2365_236564


namespace NUMINAMATH_CALUDE_cannot_determine_unique_order_l2365_236505

/-- Represents a query about the relative ordering of 3 weights -/
structure Query where
  a : Fin 5
  b : Fin 5
  c : Fin 5
  h₁ : a ≠ b
  h₂ : b ≠ c
  h₃ : a ≠ c

/-- Represents a permutation of 5 weights -/
def Permutation := Fin 5 → Fin 5

/-- Checks if a permutation is consistent with a query -/
def consistentWithQuery (p : Permutation) (q : Query) : Prop :=
  p q.a < p q.b ∧ p q.b < p q.c

/-- Checks if a permutation is consistent with all queries in a list -/
def consistentWithAllQueries (p : Permutation) (qs : List Query) : Prop :=
  ∀ q ∈ qs, consistentWithQuery p q

theorem cannot_determine_unique_order :
  ∀ (qs : List Query),
    qs.length = 9 →
    ∃ (p₁ p₂ : Permutation),
      p₁ ≠ p₂ ∧
      consistentWithAllQueries p₁ qs ∧
      consistentWithAllQueries p₂ qs :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_unique_order_l2365_236505


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2365_236513

/-- The line equation passing through a fixed point for all values of k -/
def line_equation (k x y : ℝ) : Prop :=
  (2*k - 1) * x - (k - 2) * y - (k + 4) = 0

/-- The theorem stating that the line passes through (2, 3) for all k -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k 2 3 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2365_236513


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_is_ten_l2365_236543

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat

/-- The minimum number of gumballs needed to guarantee 4 of the same color -/
def minGumballsForFour (machine : GumballMachine) : Nat :=
  10

/-- Theorem stating that for a machine with 9 red, 7 white, and 8 blue gumballs,
    the minimum number of gumballs needed to guarantee 4 of the same color is 10 -/
theorem min_gumballs_for_four_is_ten (machine : GumballMachine)
    (h_red : machine.red = 9)
    (h_white : machine.white = 7)
    (h_blue : machine.blue = 8) :
    minGumballsForFour machine = 10 := by
  sorry


end NUMINAMATH_CALUDE_min_gumballs_for_four_is_ten_l2365_236543


namespace NUMINAMATH_CALUDE_f_min_value_l2365_236580

def f (x : ℝ) : ℝ := |x - 1| + |x + 4| - 5

theorem f_min_value :
  ∀ x : ℝ, f x ≥ 0 ∧ ∃ y : ℝ, f y = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_min_value_l2365_236580


namespace NUMINAMATH_CALUDE_smaller_solution_cube_root_equation_l2365_236521

theorem smaller_solution_cube_root_equation (x : ℝ) :
  (Real.rpow x (1/3 : ℝ) + Real.rpow (16 - x) (1/3 : ℝ) = 2) →
  (x = (1 - Real.sqrt 21 / 3)^3 ∨ x = (1 + Real.sqrt 21 / 3)^3) ∧
  x ≤ (1 + Real.sqrt 21 / 3)^3 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_cube_root_equation_l2365_236521


namespace NUMINAMATH_CALUDE_rainville_total_rainfall_2007_l2365_236562

/-- Calculates the total rainfall for a year given the average monthly rainfall -/
def total_rainfall (average_monthly_rainfall : ℝ) : ℝ :=
  average_monthly_rainfall * 12

/-- Represents the rainfall data for Rainville from 2005 to 2007 -/
structure RainvilleRainfall where
  rainfall_2005 : ℝ
  rainfall_increase_2006 : ℝ
  rainfall_increase_2007 : ℝ

/-- Theorem stating the total rainfall in Rainville for 2007 -/
theorem rainville_total_rainfall_2007 (data : RainvilleRainfall) 
  (h1 : data.rainfall_2005 = 50)
  (h2 : data.rainfall_increase_2006 = 3)
  (h3 : data.rainfall_increase_2007 = 5) :
  total_rainfall (data.rainfall_2005 + data.rainfall_increase_2006 + data.rainfall_increase_2007) = 696 :=
by sorry

end NUMINAMATH_CALUDE_rainville_total_rainfall_2007_l2365_236562


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2365_236596

def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmeticSequence a →
  (a 1 + a 2 = 3) →
  (a 3 + a 4 = 5) →
  (a 7 + a 8 = 9) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2365_236596


namespace NUMINAMATH_CALUDE_tower_height_l2365_236517

/-- The height of a tower given specific angle measurements -/
theorem tower_height (angle1 angle2 : Real) (distance : Real) (height : Real) : 
  angle1 = Real.pi / 6 →  -- 30 degrees in radians
  angle2 = Real.pi / 4 →  -- 45 degrees in radians
  distance = 20 → 
  Real.tan angle1 = height / (height + distance) →
  height = 10 * (Real.sqrt 3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_tower_height_l2365_236517


namespace NUMINAMATH_CALUDE_debate_team_girls_l2365_236589

theorem debate_team_girls (boys : ℕ) (groups : ℕ) (group_size : ℕ) (girls : ℕ) : 
  boys = 26 → 
  groups = 8 → 
  group_size = 9 → 
  groups * group_size = boys + girls → 
  girls = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_debate_team_girls_l2365_236589


namespace NUMINAMATH_CALUDE_min_value_fraction_l2365_236527

theorem min_value_fraction (a b : ℝ) (h1 : a > 2*b) (h2 : b > 0) :
  (a^4 + 1) / (b * (a - 2*b)) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2365_236527


namespace NUMINAMATH_CALUDE_paradise_park_capacity_l2365_236549

/-- A Ferris wheel in paradise park -/
structure FerrisWheel where
  total_seats : ℕ
  broken_seats : ℕ
  people_per_seat : ℕ

/-- The capacity of a Ferris wheel is the number of people it can hold on functioning seats -/
def FerrisWheel.capacity (fw : FerrisWheel) : ℕ :=
  (fw.total_seats - fw.broken_seats) * fw.people_per_seat

/-- The paradise park with its three Ferris wheels -/
def paradise_park : List FerrisWheel :=
  [{ total_seats := 18, broken_seats := 10, people_per_seat := 15 },
   { total_seats := 25, broken_seats := 7,  people_per_seat := 15 },
   { total_seats := 30, broken_seats := 12, people_per_seat := 15 }]

/-- The total capacity of all Ferris wheels in paradise park -/
def total_park_capacity : ℕ :=
  (paradise_park.map FerrisWheel.capacity).sum

theorem paradise_park_capacity :
  total_park_capacity = 660 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_capacity_l2365_236549


namespace NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l2365_236597

theorem polynomial_coefficient_theorem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, x^6 = a₀ + a₁*(2+x) + a₂*(2+x)^2 + a₃*(2+x)^3 + a₄*(2+x)^4 + a₅*(2+x)^5 + a₆*(2+x)^6) →
  (a₃ = -160 ∧ a₁ + a₃ + a₅ = -364) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l2365_236597


namespace NUMINAMATH_CALUDE_grace_earnings_l2365_236554

/-- Calculates the number of weeks needed to earn a target amount given a weekly rate and biweekly payment schedule. -/
def weeksToEarn (weeklyRate : ℕ) (targetAmount : ℕ) : ℕ :=
  let biweeklyEarnings := weeklyRate * 2
  let numPayments := targetAmount / biweeklyEarnings
  numPayments * 2

/-- Proves that it takes 6 weeks to earn 1800 dollars with a weekly rate of 300 dollars and biweekly payments. -/
theorem grace_earnings : weeksToEarn 300 1800 = 6 := by
  sorry

end NUMINAMATH_CALUDE_grace_earnings_l2365_236554


namespace NUMINAMATH_CALUDE_fort_blocks_count_l2365_236555

/-- Represents the dimensions of a rectangular fort --/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for a fort with given dimensions and specifications --/
def calculateFortBlocks (d : FortDimensions) : ℕ :=
  let totalVolume := d.length * d.width * d.height
  let internalLength := d.length - 2
  let internalWidth := d.width - 2
  let internalHeight := d.height - 1
  let internalVolume := internalLength * internalWidth * internalHeight
  let partitionVolume := 1 * internalWidth * internalHeight
  totalVolume - internalVolume + partitionVolume

/-- Theorem stating that a fort with the given dimensions requires 458 blocks --/
theorem fort_blocks_count :
  let fortDims : FortDimensions := ⟨14, 12, 6⟩
  calculateFortBlocks fortDims = 458 := by
  sorry

#eval calculateFortBlocks ⟨14, 12, 6⟩

end NUMINAMATH_CALUDE_fort_blocks_count_l2365_236555


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2365_236506

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℝ := 0.056

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 5.6
    exponent := -2
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  original_number = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2365_236506


namespace NUMINAMATH_CALUDE_equation_is_false_l2365_236576

theorem equation_is_false : 4.58 - (0.45 + 2.58) ≠ 4.58 - 2.58 + 0.45 ∨ 4.58 - (0.45 + 2.58) ≠ 2.45 := by
  sorry

end NUMINAMATH_CALUDE_equation_is_false_l2365_236576


namespace NUMINAMATH_CALUDE_arc_square_region_area_coefficients_sum_l2365_236581

/-- Represents a circular arc --/
structure CircularArc where
  radius : ℝ
  centralAngle : ℝ

/-- Represents the region formed by three circular arcs and a square --/
structure ArcSquareRegion where
  arcs : Fin 3 → CircularArc
  squareSideLength : ℝ

/-- The area of the region inside the arcs but outside the square --/
noncomputable def regionArea (r : ArcSquareRegion) : ℝ :=
  sorry

/-- Coefficients of the area expression a√b + cπ - d --/
structure AreaCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem arc_square_region_area_coefficients_sum :
  ∀ r : ArcSquareRegion,
  (∀ i : Fin 3, (r.arcs i).radius = 6 ∧ (r.arcs i).centralAngle = 45 * π / 180) →
  r.squareSideLength = 12 →
  ∃ coeff : AreaCoefficients,
    regionArea r = coeff.c * π - coeff.d ∧
    coeff.a + coeff.b + coeff.c + coeff.d = 174 :=
sorry

end NUMINAMATH_CALUDE_arc_square_region_area_coefficients_sum_l2365_236581


namespace NUMINAMATH_CALUDE_unique_positive_number_l2365_236535

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x - 4 = 21 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l2365_236535


namespace NUMINAMATH_CALUDE_second_number_is_sixty_l2365_236575

theorem second_number_is_sixty :
  ∀ (a b : ℝ),
  (a + b + 20 + 60) / 4 = (10 + 70 + 28) / 3 + 4 →
  (a = 60 ∨ b = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_second_number_is_sixty_l2365_236575


namespace NUMINAMATH_CALUDE_sorting_abc_l2365_236569

theorem sorting_abc (a b c : Real)
  (ha : 0 ≤ a ∧ a ≤ Real.pi / 2)
  (hb : 0 ≤ b ∧ b ≤ Real.pi / 2)
  (hc : 0 ≤ c ∧ c ≤ Real.pi / 2)
  (ca : Real.cos a = a)
  (sb : Real.sin (Real.cos b) = b)
  (cs : Real.cos (Real.sin c) = c) :
  b < a ∧ a < c := by
sorry

end NUMINAMATH_CALUDE_sorting_abc_l2365_236569


namespace NUMINAMATH_CALUDE_orange_ratio_problem_l2365_236546

theorem orange_ratio_problem (michaela_oranges : ℕ) (total_oranges : ℕ) (remaining_oranges : ℕ) :
  michaela_oranges = 20 →
  total_oranges = 90 →
  remaining_oranges = 30 →
  (total_oranges - remaining_oranges - michaela_oranges) / michaela_oranges = 2 :=
by sorry

end NUMINAMATH_CALUDE_orange_ratio_problem_l2365_236546


namespace NUMINAMATH_CALUDE_square_difference_630_570_l2365_236595

theorem square_difference_630_570 : 630^2 - 570^2 = 72000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_630_570_l2365_236595


namespace NUMINAMATH_CALUDE_tangent_and_intersection_l2365_236511

-- Define the curve C
def C (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := -12 * x + 8

-- Theorem statement
theorem tangent_and_intersection :
  -- The tangent line at x = 1 has the equation y = -12x + 8
  (∀ x, tangent_line x = -12 * x + 8) ∧
  -- The tangent line touches the curve at x = 1
  (C 1 = tangent_line 1) ∧
  -- The tangent line is indeed tangent to the curve at x = 1
  (deriv C 1 = -12) ∧
  -- The tangent line intersects the curve at two additional points
  (C (-2) = tangent_line (-2)) ∧
  (C (2/3) = tangent_line (2/3)) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_intersection_l2365_236511


namespace NUMINAMATH_CALUDE_water_in_bucket_l2365_236577

theorem water_in_bucket (initial_amount poured_out : ℚ) 
  (h1 : initial_amount = 15/8)
  (h2 : poured_out = 9/8) : 
  initial_amount - poured_out = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_water_in_bucket_l2365_236577


namespace NUMINAMATH_CALUDE_joe_fish_compared_to_sam_l2365_236508

theorem joe_fish_compared_to_sam (harry_fish joe_fish sam_fish : ℕ) 
  (harry_joe_ratio : harry_fish = 4 * joe_fish)
  (joe_sam_ratio : ∃ x : ℕ, joe_fish = x * sam_fish)
  (sam_fish_count : sam_fish = 7)
  (harry_fish_count : harry_fish = 224) :
  ∃ x : ℕ, joe_fish = 8 * sam_fish := by
  sorry

end NUMINAMATH_CALUDE_joe_fish_compared_to_sam_l2365_236508


namespace NUMINAMATH_CALUDE_same_row_both_shows_l2365_236599

/-- Represents a seating arrangement for a show -/
def SeatingArrangement := Fin 50 → Fin 7

/-- The number of rows in the cinema -/
def num_rows : Nat := 7

/-- The number of children attending the shows -/
def num_children : Nat := 50

/-- Theorem: There exist at least two children who sat in the same row during both shows -/
theorem same_row_both_shows (morning_seating evening_seating : SeatingArrangement) :
  ∃ (i j : Fin 50), i ≠ j ∧
    morning_seating i = morning_seating j ∧
    evening_seating i = evening_seating j :=
sorry

end NUMINAMATH_CALUDE_same_row_both_shows_l2365_236599


namespace NUMINAMATH_CALUDE_heavy_operator_daily_rate_l2365_236583

theorem heavy_operator_daily_rate
  (total_workers : ℕ)
  (num_laborers : ℕ)
  (laborer_rate : ℕ)
  (total_payroll : ℕ)
  (h1 : total_workers = 31)
  (h2 : num_laborers = 1)
  (h3 : laborer_rate = 82)
  (h4 : total_payroll = 3952) :
  (total_payroll - num_laborers * laborer_rate) / (total_workers - num_laborers) = 129 := by
sorry

end NUMINAMATH_CALUDE_heavy_operator_daily_rate_l2365_236583


namespace NUMINAMATH_CALUDE_cube_cutting_l2365_236578

theorem cube_cutting (n s : ℕ) : 
  n > s → 
  n^3 - s^3 = 152 → 
  n = 6 ∧ s = 4 := by sorry

end NUMINAMATH_CALUDE_cube_cutting_l2365_236578


namespace NUMINAMATH_CALUDE_orange_bin_count_l2365_236542

theorem orange_bin_count (initial : ℕ) (removed : ℕ) (added : ℕ) : 
  initial = 40 → removed = 37 → added = 7 → initial - removed + added = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_count_l2365_236542


namespace NUMINAMATH_CALUDE_expression_simplification_l2365_236590

theorem expression_simplification (x : ℝ) :
  4 * x^3 + 5 * x + 9 - (3 * x^3 - 2 * x + 1) + 2 * x^2 - (x^2 - 4 * x - 6) =
  x^3 + x^2 + 11 * x + 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2365_236590


namespace NUMINAMATH_CALUDE_bennys_kids_l2365_236567

/-- Prove that Benny has 18 kids given the conditions of the problem -/
theorem bennys_kids : ℕ :=
  let total_money : ℕ := 360
  let apple_cost : ℕ := 4
  let apples_per_kid : ℕ := 5
  let num_kids : ℕ := 18
  have h1 : total_money ≥ apple_cost * apples_per_kid * num_kids := by sorry
  have h2 : apples_per_kid > 0 := by sorry
  num_kids

/- Proof omitted -/

end NUMINAMATH_CALUDE_bennys_kids_l2365_236567


namespace NUMINAMATH_CALUDE_roots_form_triangle_l2365_236533

/-- The roots of the equation (x-1)(x^2-2x+m) = 0 can form a triangle if and only if 3/4 < m ≤ 1 -/
theorem roots_form_triangle (m : ℝ) : 
  (∃ a b c : ℝ, 
    (a - 1) * (a^2 - 2*a + m) = 0 ∧
    (b - 1) * (b^2 - 2*b + m) = 0 ∧
    (c - 1) * (c^2 - 2*c + m) = 0 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b) ↔
  (3/4 < m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_form_triangle_l2365_236533


namespace NUMINAMATH_CALUDE_arrangement_count_l2365_236591

def number_of_arrangements (n_male n_female : ℕ) : ℕ :=
  sorry

theorem arrangement_count :
  let n_male := 2
  let n_female := 3
  number_of_arrangements n_male n_female = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l2365_236591


namespace NUMINAMATH_CALUDE_solution_set_a_eq_1_range_of_a_l2365_236530

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1| + |a * x - 3 * a|

-- Part 1: Solution set when a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≥ 9/2 ∨ x ≤ -1/2} :=
sorry

-- Part 2: Range of a when solution set is ℝ
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f a x ≥ 5) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_1_range_of_a_l2365_236530


namespace NUMINAMATH_CALUDE_fourth_power_sum_of_cubic_roots_l2365_236520

theorem fourth_power_sum_of_cubic_roots (a b c : ℝ) : 
  (a^3 - 3*a + 1 = 0) → 
  (b^3 - 3*b + 1 = 0) → 
  (c^3 - 3*c + 1 = 0) → 
  a^4 + b^4 + c^4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_of_cubic_roots_l2365_236520


namespace NUMINAMATH_CALUDE_susan_spending_l2365_236547

def carnival_spending (initial_amount food_cost : ℝ) : ℝ :=
  let ride_cost := 2 * food_cost
  let game_cost := 0.5 * food_cost
  let total_spent := food_cost + ride_cost + game_cost
  initial_amount - total_spent

theorem susan_spending :
  carnival_spending 80 15 = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_susan_spending_l2365_236547


namespace NUMINAMATH_CALUDE_number_operations_equivalence_l2365_236560

theorem number_operations_equivalence (x : ℝ) : ((x * (5/6)) / (2/3)) - 2 = (x * (5/4)) - 2 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_equivalence_l2365_236560


namespace NUMINAMATH_CALUDE_point_transformation_l2365_236514

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def transform_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_y_180 |> reflect_yz |> reflect_xz |> rotate_y_180 |> reflect_xz

theorem point_transformation :
  transform_point (2, 3, 4) = (-2, 3, 4) := by
  sorry


end NUMINAMATH_CALUDE_point_transformation_l2365_236514


namespace NUMINAMATH_CALUDE_not_integer_fraction_l2365_236545

theorem not_integer_fraction (a b : ℤ) : ¬ (∃ (k : ℤ), (a^2 + b^2) = k * (a^2 - b^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_not_integer_fraction_l2365_236545


namespace NUMINAMATH_CALUDE_decompose_6058_l2365_236592

theorem decompose_6058 : 6058 = 6 * 1000 + 5 * 10 + 8 * 1 := by
  sorry

end NUMINAMATH_CALUDE_decompose_6058_l2365_236592


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2365_236574

theorem simple_interest_rate_calculation (P : ℝ) (R : ℝ) : 
  P * (1 + 7 * R / 100) = 7 / 6 * P → R = 100 / 49 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2365_236574


namespace NUMINAMATH_CALUDE_number_puzzle_l2365_236525

theorem number_puzzle (x : ℝ) : (x - 26) / 2 = 37 → 48 - x / 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2365_236525


namespace NUMINAMATH_CALUDE_expression_equals_one_l2365_236504

theorem expression_equals_one :
  (121^2 - 11^2) / (91^2 - 13^2) * ((91-13)*(91+13)) / ((121-11)*(121+11)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2365_236504


namespace NUMINAMATH_CALUDE_harmonic_sum_increase_l2365_236541

theorem harmonic_sum_increase (k : ℕ) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_increase_l2365_236541


namespace NUMINAMATH_CALUDE_ac_value_l2365_236539

def letter_value (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

def word_value (w : String) : ℕ :=
  (w.toList.map letter_value).sum * w.length

theorem ac_value : word_value "ac" = 8 := by
  sorry

end NUMINAMATH_CALUDE_ac_value_l2365_236539


namespace NUMINAMATH_CALUDE_distance_ratio_theorem_l2365_236585

theorem distance_ratio_theorem (x : ℝ) (h1 : x^2 + (-9)^2 = 18^2) :
  |(-9)| / 18 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_distance_ratio_theorem_l2365_236585


namespace NUMINAMATH_CALUDE_max_triangles_theorem_max_squares_theorem_l2365_236553

/-- The maximum number of identical, non-overlapping, and parallel triangles that can be fitted into a triangle -/
def max_triangles_in_triangle : ℕ := 6

/-- The maximum number of identical, non-overlapping, and parallel squares that can be fitted into a square -/
def max_squares_in_square : ℕ := 8

/-- Theorem stating the maximum number of identical, non-overlapping, and parallel triangles that can be fitted into a triangle -/
theorem max_triangles_theorem : max_triangles_in_triangle = 6 := by sorry

/-- Theorem stating the maximum number of identical, non-overlapping, and parallel squares that can be fitted into a square -/
theorem max_squares_theorem : max_squares_in_square = 8 := by sorry

end NUMINAMATH_CALUDE_max_triangles_theorem_max_squares_theorem_l2365_236553


namespace NUMINAMATH_CALUDE_find_number_l2365_236550

theorem find_number : ∃ x : ℝ, 0.5 * x = 0.2 * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2365_236550


namespace NUMINAMATH_CALUDE_table_length_is_77_l2365_236507

/-- Represents the dimensions of a rectangular table. -/
structure TableDimensions where
  length : ℕ
  width : ℕ

/-- Represents the dimensions of a paper sheet. -/
structure SheetDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the length of a table given its width and the dimensions of the paper sheets used to cover it. -/
def calculateTableLength (tableWidth : ℕ) (sheet : SheetDimensions) : ℕ :=
  sheet.length + (tableWidth - sheet.width)

/-- Theorem stating that for a table of width 80 cm covered with 5x8 cm sheets,
    where each sheet is placed 1 cm higher and 1 cm to the right of the previous one,
    the length of the table is 77 cm. -/
theorem table_length_is_77 :
  let tableWidth : ℕ := 80
  let sheet : SheetDimensions := ⟨5, 8⟩
  calculateTableLength tableWidth sheet = 77 := by
  sorry

#check table_length_is_77

end NUMINAMATH_CALUDE_table_length_is_77_l2365_236507


namespace NUMINAMATH_CALUDE_time_to_restaurant_is_10_minutes_l2365_236587

/-- Time in minutes to walk from Park Office to Hidden Lake -/
def time_to_hidden_lake : ℕ := 15

/-- Time in minutes to walk from Hidden Lake to Park Office -/
def time_from_hidden_lake : ℕ := 7

/-- Total time in minutes for the entire journey (including Lake Park restaurant) -/
def total_time : ℕ := 32

/-- Time in minutes to walk from Park Office to Lake Park restaurant -/
def time_to_restaurant : ℕ := total_time - (time_to_hidden_lake + time_from_hidden_lake)

theorem time_to_restaurant_is_10_minutes : time_to_restaurant = 10 := by
  sorry

end NUMINAMATH_CALUDE_time_to_restaurant_is_10_minutes_l2365_236587


namespace NUMINAMATH_CALUDE_mean_of_five_integers_l2365_236594

theorem mean_of_five_integers (p q r s t : ℤ) 
  (h1 : (p + q + r) / 3 = 9)
  (h2 : (s + t) / 2 = 14) :
  (p + q + r + s + t) / 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_five_integers_l2365_236594


namespace NUMINAMATH_CALUDE_max_edges_triangle_free_30_max_edges_k4_free_30_l2365_236536

/-- The maximum number of edges in a triangle-free graph with 30 vertices -/
def max_edges_triangle_free (n : ℕ) : ℕ :=
  if n = 30 then 225 else 0

/-- The maximum number of edges in a K₄-free graph with 30 vertices -/
def max_edges_k4_free (n : ℕ) : ℕ :=
  if n = 30 then 300 else 0

/-- Theorem stating the maximum number of edges in a triangle-free graph with 30 vertices -/
theorem max_edges_triangle_free_30 :
  max_edges_triangle_free 30 = 225 := by sorry

/-- Theorem stating the maximum number of edges in a K₄-free graph with 30 vertices -/
theorem max_edges_k4_free_30 :
  max_edges_k4_free 30 = 300 := by sorry

end NUMINAMATH_CALUDE_max_edges_triangle_free_30_max_edges_k4_free_30_l2365_236536


namespace NUMINAMATH_CALUDE_simplify_sqrt_450_l2365_236582

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_450_l2365_236582


namespace NUMINAMATH_CALUDE_tower_blocks_sum_l2365_236556

/-- The total number of blocks in a tower after adding more blocks -/
def total_blocks (initial : Real) (added : Real) : Real :=
  initial + added

/-- Theorem: The total number of blocks is the sum of initial and added blocks -/
theorem tower_blocks_sum (initial : Real) (added : Real) :
  total_blocks initial added = initial + added := by
  sorry

end NUMINAMATH_CALUDE_tower_blocks_sum_l2365_236556


namespace NUMINAMATH_CALUDE_surface_area_of_circumscribed_sphere_l2365_236598

/-- A regular tetrahedron with edge length √2 -/
structure RegularTetrahedron where
  edgeLength : ℝ
  isRegular : edgeLength = Real.sqrt 2

/-- A sphere circumscribing a regular tetrahedron -/
structure CircumscribedSphere (t : RegularTetrahedron) where
  radius : ℝ
  containsVertices : True  -- This is a placeholder for the condition that all vertices are on the sphere

/-- The surface area of a sphere circumscribing a regular tetrahedron with edge length √2 is 3π -/
theorem surface_area_of_circumscribed_sphere (t : RegularTetrahedron) (s : CircumscribedSphere t) :
  4 * Real.pi * s.radius ^ 2 = 3 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_circumscribed_sphere_l2365_236598


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l2365_236551

theorem floor_ceiling_sum : ⌊(-3.75 : ℝ)⌋ + ⌈(34.25 : ℝ)⌉ = 31 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l2365_236551


namespace NUMINAMATH_CALUDE_line_intersects_parabola_once_l2365_236579

/-- The line x = k intersects the parabola x = -2y^2 - 3y + 5 at exactly one point if and only if k = 49/8 -/
theorem line_intersects_parabola_once (k : ℝ) : 
  (∃! y : ℝ, k = -2 * y^2 - 3 * y + 5) ↔ k = 49/8 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_once_l2365_236579


namespace NUMINAMATH_CALUDE_sixteen_integer_lengths_l2365_236510

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths possible for line segments
    drawn from a vertex to points on the hypotenuse -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem stating that for a right triangle with legs 24 and 25,
    there are exactly 16 distinct integer lengths possible -/
theorem sixteen_integer_lengths :
  ∃ (t : RightTriangle), t.de = 24 ∧ t.ef = 25 ∧ countIntegerLengths t = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_integer_lengths_l2365_236510


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l2365_236531

theorem cube_surface_area_increase :
  ∀ (s : ℝ), s > 0 →
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.3 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.69 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l2365_236531


namespace NUMINAMATH_CALUDE_trailing_zeros_15_factorial_base_15_l2365_236593

/-- The number of trailing zeros in n! in base b --/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The factorial of a natural number --/
def factorial (n : ℕ) : ℕ := sorry

/-- 15 factorial --/
def factorial15 : ℕ := factorial 15

theorem trailing_zeros_15_factorial_base_15 :
  trailingZeros factorial15 15 = 3 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_15_factorial_base_15_l2365_236593


namespace NUMINAMATH_CALUDE_sally_pens_taken_home_l2365_236509

def total_pens : ℕ := 5230
def num_students : ℕ := 89
def pens_per_student : ℕ := 58

def pens_distributed : ℕ := num_students * pens_per_student
def pens_remaining : ℕ := total_pens - pens_distributed
def pens_in_locker : ℕ := pens_remaining / 2
def pens_taken_home : ℕ := pens_remaining - pens_in_locker

theorem sally_pens_taken_home : pens_taken_home = 34 := by
  sorry

end NUMINAMATH_CALUDE_sally_pens_taken_home_l2365_236509


namespace NUMINAMATH_CALUDE_max_value_of_f_l2365_236501

-- Define the function f(x) = √(x-3) + √(6-x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 3) + Real.sqrt (6 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 6 ∧
  f x = Real.sqrt 6 ∧
  ∀ (y : ℝ), 3 ≤ y ∧ y ≤ 6 → f y ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2365_236501


namespace NUMINAMATH_CALUDE_garage_sale_items_count_l2365_236563

theorem garage_sale_items_count (prices : Finset ℕ) (radio_price : ℕ) : 
  radio_price ∈ prices →
  (prices.filter (λ x => x > radio_price)).card = 15 →
  (prices.filter (λ x => x < radio_price)).card = 22 →
  prices.card = 38 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_items_count_l2365_236563


namespace NUMINAMATH_CALUDE_problem_statement_l2365_236558

theorem problem_statement (a b c : ℕ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a ≥ b ∧ b ≥ c) (h3 : Nat.Prime ((a - c) / 2))
  (h4 : a^2 + b^2 + c^2 - 2*(a*b + b*c + c*a) = b) :
  Nat.Prime b ∨ ∃ k : ℕ, b = k^2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2365_236558


namespace NUMINAMATH_CALUDE_remainder_777_444_mod_13_l2365_236540

theorem remainder_777_444_mod_13 : 777^444 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_777_444_mod_13_l2365_236540


namespace NUMINAMATH_CALUDE_divide_multiply_add_subtract_l2365_236537

theorem divide_multiply_add_subtract (x n : ℝ) : x = 40 → ((x / n) * 5 + 10 - 12 = 48 ↔ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_divide_multiply_add_subtract_l2365_236537


namespace NUMINAMATH_CALUDE_robin_gum_count_l2365_236572

/-- The number of packages of gum Robin has -/
def num_packages : ℕ := 25

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := 42

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 1050 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l2365_236572


namespace NUMINAMATH_CALUDE_output_for_15_l2365_236523

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 25 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 38 := by sorry

end NUMINAMATH_CALUDE_output_for_15_l2365_236523


namespace NUMINAMATH_CALUDE_gear_diameter_relation_l2365_236534

/-- Represents a circular gear with a diameter and revolutions per minute. -/
structure Gear where
  diameter : ℝ
  rpm : ℝ

/-- Represents a system of two interconnected gears. -/
structure GearSystem where
  gearA : Gear
  gearB : Gear
  /-- The gears travel at the same circumferential rate -/
  same_rate : gearA.diameter * gearA.rpm = gearB.diameter * gearB.rpm

/-- Theorem stating the relationship between gear diameters given their rpm ratio -/
theorem gear_diameter_relation (sys : GearSystem) 
  (h1 : sys.gearB.diameter = 50)
  (h2 : sys.gearA.rpm = 5 * sys.gearB.rpm) :
  sys.gearA.diameter = 10 := by
  sorry

end NUMINAMATH_CALUDE_gear_diameter_relation_l2365_236534


namespace NUMINAMATH_CALUDE_parallel_planes_from_skew_lines_l2365_236502

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_from_skew_lines 
  (α β : Plane) (l m : Line) : 
  α ≠ β →
  skew l m →
  parallel_line_plane l α →
  parallel_line_plane m α →
  parallel_line_plane l β →
  parallel_line_plane m β →
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_skew_lines_l2365_236502


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2365_236538

/-- Given a quadratic function f(x) = x^2 - x + a, if f(-t) < 0, then f(t+1) < 0 -/
theorem quadratic_function_property (a t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - x + a
  f (-t) < 0 → f (t + 1) < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2365_236538


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2365_236500

theorem square_sum_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a - 1/b - 1/(a+b) = 0) : (b/a + a/b)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2365_236500


namespace NUMINAMATH_CALUDE_final_state_only_beads_l2365_236529

/-- Represents the types of items in the exchange system -/
inductive Item
  | Gold
  | Pearl
  | Bead

/-- Represents the state of items in the exchange system -/
structure ItemState :=
  (gold : ℕ)
  (pearl : ℕ)
  (bead : ℕ)

/-- Represents an exchange rule -/
structure ExchangeRule :=
  (input1 : Item)
  (input2 : Item)
  (output : Item)

/-- Applies an exchange rule to the current state -/
def applyExchange (state : ItemState) (rule : ExchangeRule) : ItemState :=
  sorry

/-- Checks if an exchange rule can be applied to the current state -/
def canApplyExchange (state : ItemState) (rule : ExchangeRule) : Prop :=
  sorry

/-- Represents the exchange system with initial state and rules -/
structure ExchangeSystem :=
  (initialState : ItemState)
  (rules : List ExchangeRule)

/-- Defines the final state after all possible exchanges -/
def finalState (system : ExchangeSystem) : ItemState :=
  sorry

/-- Theorem: The final state after all exchanges will only have beads -/
theorem final_state_only_beads (system : ExchangeSystem) :
  system.initialState = ItemState.mk 24 26 25 →
  system.rules = [
    ExchangeRule.mk Item.Gold Item.Pearl Item.Bead,
    ExchangeRule.mk Item.Gold Item.Bead Item.Pearl,
    ExchangeRule.mk Item.Pearl Item.Bead Item.Gold
  ] →
  ∃ n : ℕ, finalState system = ItemState.mk 0 0 n :=
sorry

end NUMINAMATH_CALUDE_final_state_only_beads_l2365_236529


namespace NUMINAMATH_CALUDE_max_value_product_l2365_236516

theorem max_value_product (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a^2 - b*c) * (b^2 - c*a) * (c^2 - a*b) ≤ 1/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l2365_236516
