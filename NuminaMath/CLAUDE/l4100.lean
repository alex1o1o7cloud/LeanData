import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_box_volume_l4100_410084

theorem rectangular_box_volume (x : ℕ) (h : x > 0) :
  let volume := 10 * x^3
  (volume = 60 ∨ volume = 80 ∨ volume = 100 ∨ volume = 120 ∨ volume = 200) →
  volume = 80 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l4100_410084


namespace NUMINAMATH_CALUDE_election_ratio_l4100_410033

theorem election_ratio (X Y : ℝ) 
  (h1 : 0.74 * X + 0.5000000000000002 * Y = 0.66 * (X + Y)) 
  (h2 : X > 0) 
  (h3 : Y > 0) : 
  X / Y = 2 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l4100_410033


namespace NUMINAMATH_CALUDE_james_jello_cost_l4100_410062

/-- The cost to fill a bathtub with jello --/
def jello_cost (bathtub_capacity : ℝ) (cubic_foot_to_gallon : ℝ) (gallon_weight : ℝ) 
                (jello_mix_ratio : ℝ) (jello_mix_cost : ℝ) : ℝ :=
  bathtub_capacity * cubic_foot_to_gallon * gallon_weight * jello_mix_ratio * jello_mix_cost

/-- Theorem: The cost to fill James' bathtub with jello is $270 --/
theorem james_jello_cost : 
  jello_cost 6 7.5 8 1.5 0.5 = 270 := by
  sorry

#eval jello_cost 6 7.5 8 1.5 0.5

end NUMINAMATH_CALUDE_james_jello_cost_l4100_410062


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_y_l4100_410015

theorem max_value_of_x_plus_y : 
  ∃ (M : ℝ), M = 4 ∧ 
  ∀ (x y : ℝ), x^2 + y + 3*x - 3 = 0 → x + y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_y_l4100_410015


namespace NUMINAMATH_CALUDE_dormitory_arrangement_l4100_410030

/-- Given:
  - If each dormitory houses 4 students, there would be 20 students without accommodation.
  - If each dormitory houses 8 students, one dormitory would be neither full nor empty,
    with the rest being completely full.
  Prove that there are 44 new students needing accommodation and 6 dormitories provided. -/
theorem dormitory_arrangement (num_dorms : ℕ) (num_students : ℕ) : 
  (4 * num_dorms + 20 = num_students) →
  (∃ k : ℕ, 0 < k ∧ k < 8 ∧ 8 * (num_dorms - 1) + k = num_students) →
  num_students = 44 ∧ num_dorms = 6 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_arrangement_l4100_410030


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_leq_neg_one_l4100_410069

def M (m : ℝ) : Set ℝ := {x | x - m < 0}

def N : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x}

theorem intersection_empty_implies_m_leq_neg_one (m : ℝ) :
  M m ∩ N = ∅ → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_m_leq_neg_one_l4100_410069


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_l4100_410083

def is_valid_abc (a b c : ℕ) : Prop :=
  a > 3 ∧ b > 3 ∧ c > 3 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

def form_number (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 321

theorem smallest_multiple_of_seven :
  ∀ a b c : ℕ,
    is_valid_abc a b c →
    form_number a b c ≥ 468321 ∨ ¬(form_number a b c % 7 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_l4100_410083


namespace NUMINAMATH_CALUDE_reseating_women_l4100_410024

/-- The number of ways to reseat n women in a line, where each woman can sit in her original seat or within two positions on either side. -/
def T : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 4
  | 3 => 7
  | (n + 4) => T (n + 3) + T (n + 2) + T (n + 1)

/-- Theorem stating that the number of ways to reseat 10 women under the given conditions is 480. -/
theorem reseating_women : T 10 = 480 := by
  sorry

end NUMINAMATH_CALUDE_reseating_women_l4100_410024


namespace NUMINAMATH_CALUDE_equation_solution_l4100_410099

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4100_410099


namespace NUMINAMATH_CALUDE_correct_calculation_l4100_410000

theorem correct_calculation (x : ℝ) (h : 21 * x = 63) : x + 40 = 43 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l4100_410000


namespace NUMINAMATH_CALUDE_vinegar_left_is_60_l4100_410043

/-- Represents the pickle-making scenario with given supplies and constraints. -/
structure PickleSupplies where
  jars : ℕ
  cucumbers : ℕ
  vinegar : ℕ
  pickles_per_cucumber : ℕ
  pickles_per_jar : ℕ
  vinegar_per_jar : ℕ

/-- Calculates the amount of vinegar left after making pickles. -/
def vinegar_left (supplies : PickleSupplies) : ℕ :=
  let jar_capacity := supplies.jars * supplies.pickles_per_jar
  let cucumber_capacity := supplies.cucumbers * supplies.pickles_per_cucumber
  let vinegar_capacity := supplies.vinegar / supplies.vinegar_per_jar * supplies.pickles_per_jar
  let pickles_made := min jar_capacity (min cucumber_capacity vinegar_capacity)
  let jars_used := (pickles_made + supplies.pickles_per_jar - 1) / supplies.pickles_per_jar
  supplies.vinegar - jars_used * supplies.vinegar_per_jar

/-- Theorem stating that given the specific supplies and constraints, 60 ounces of vinegar are left. -/
theorem vinegar_left_is_60 :
  vinegar_left {
    jars := 4,
    cucumbers := 10,
    vinegar := 100,
    pickles_per_cucumber := 6,
    pickles_per_jar := 12,
    vinegar_per_jar := 10
  } = 60 := by
  sorry

end NUMINAMATH_CALUDE_vinegar_left_is_60_l4100_410043


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l4100_410051

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-4, y)
  collinear a b → y = -6 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l4100_410051


namespace NUMINAMATH_CALUDE_square_field_area_l4100_410010

/-- The area of a square field given the time and speed of a horse running around it -/
theorem square_field_area (time : ℝ) (speed : ℝ) : 
  time = 10 → speed = 12 → (time * speed / 4) ^ 2 = 900 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l4100_410010


namespace NUMINAMATH_CALUDE_absolute_value_positive_l4100_410001

theorem absolute_value_positive (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_positive_l4100_410001


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l4100_410077

theorem systematic_sampling_probability 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (h1 : total_students = 52) 
  (h2 : selected_students = 10) :
  (1 - 2 / total_students) * (selected_students / (total_students - 2)) = 5 / 26 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l4100_410077


namespace NUMINAMATH_CALUDE_min_value_parabola_l4100_410068

theorem min_value_parabola (x y : ℝ) (h : x^2 = 4*y) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x' y' : ℝ), x'^2 = 4*y' →
    Real.sqrt ((x' - 3)^2 + (y' - 1)^2) + y' ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_parabola_l4100_410068


namespace NUMINAMATH_CALUDE_find_numbers_l4100_410041

theorem find_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 12) : x = 26 ∧ y = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_numbers_l4100_410041


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l4100_410049

/-- Represents a repeating decimal with a single repeating digit -/
def RepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / 999

theorem repeating_decimal_division (h : RepeatingDecimal 0 4 / RepeatingDecimal 1 6 = 4 / 15) : 
  RepeatingDecimal 0 4 / RepeatingDecimal 1 6 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l4100_410049


namespace NUMINAMATH_CALUDE_product_digit_sum_l4100_410056

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def product : ℕ := number1 * number2

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem product_digit_sum :
  thousands_digit product + units_digit product = 3 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l4100_410056


namespace NUMINAMATH_CALUDE_unique_number_five_times_less_than_digit_sum_l4100_410003

def sum_of_digits (x : ℝ) : ℕ :=
  sorry

theorem unique_number_five_times_less_than_digit_sum :
  ∃! x : ℝ, x ≠ 0 ∧ x = (sum_of_digits x : ℝ) / 5 ∧ x = 1.8 :=
sorry

end NUMINAMATH_CALUDE_unique_number_five_times_less_than_digit_sum_l4100_410003


namespace NUMINAMATH_CALUDE_round_trip_time_l4100_410059

/-- The total time for a round trip between two points, given the distance and speeds in each direction -/
theorem round_trip_time (distance : ℝ) (speed_to : ℝ) (speed_from : ℝ) :
  distance = 19.999999999999996 →
  speed_to = 25 →
  speed_from = 4 →
  (distance / speed_to) + (distance / speed_from) = 5.8 := by
  sorry

#check round_trip_time

end NUMINAMATH_CALUDE_round_trip_time_l4100_410059


namespace NUMINAMATH_CALUDE_playground_girls_count_l4100_410074

theorem playground_girls_count (total_children boys : ℕ) 
  (h1 : total_children = 117) 
  (h2 : boys = 40) : 
  total_children - boys = 77 := by
sorry

end NUMINAMATH_CALUDE_playground_girls_count_l4100_410074


namespace NUMINAMATH_CALUDE_triangle_inequality_proof_l4100_410054

theorem triangle_inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > 
  a^3 + b^3 + c^3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_proof_l4100_410054


namespace NUMINAMATH_CALUDE_hyperbola_properties_l4100_410008

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the asymptotic lines
def asymptotic_lines (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Theorem statement
theorem hyperbola_properties
  (a b : ℝ)
  (h_positive : a > 0 ∧ b > 0)
  (h_point : hyperbola a b (-2) (Real.sqrt 6))
  (h_asymptotic : ∀ x y, hyperbola a b x y → asymptotic_lines x y) :
  -- 1) The equation of C is x^2 - y^2/2 = 1
  (∀ x y, hyperbola a b x y ↔ x^2 - y^2/2 = 1) ∧
  -- 2) P cannot be the midpoint of any chord AB of C
  (∀ A B : ℝ × ℝ,
    (hyperbola a b A.1 A.2 ∧ hyperbola a b B.1 B.2) →
    (∃ k : ℝ, A.2 - point_P.2 = k * (A.1 - point_P.1) ∧
              B.2 - point_P.2 = k * (B.1 - point_P.1)) →
    point_P ≠ ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l4100_410008


namespace NUMINAMATH_CALUDE_least_number_divisible_l4100_410097

theorem least_number_divisible (n : ℕ) : n = 861 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 24 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 32 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 54 * k)) ∧
  (∃ k1 k2 k3 k4 : ℕ, (n + 3) = 24 * k1 ∧ (n + 3) = 32 * k2 ∧ (n + 3) = 36 * k3 ∧ (n + 3) = 54 * k4) :=
by sorry

#check least_number_divisible

end NUMINAMATH_CALUDE_least_number_divisible_l4100_410097


namespace NUMINAMATH_CALUDE_class_overlap_difference_l4100_410088

theorem class_overlap_difference (total students_geometry students_biology : ℕ) 
  (h1 : total = 232)
  (h2 : students_geometry = 144)
  (h3 : students_biology = 119) :
  (min students_geometry students_biology) - 
  (students_geometry + students_biology - total) = 88 :=
by sorry

end NUMINAMATH_CALUDE_class_overlap_difference_l4100_410088


namespace NUMINAMATH_CALUDE_x_is_integer_l4100_410098

theorem x_is_integer (x : ℝ) (h1 : ∃ n : ℤ, x^3 - x = n) (h2 : ∃ m : ℤ, x^4 - x = m) : ∃ k : ℤ, x = k := by
  sorry

end NUMINAMATH_CALUDE_x_is_integer_l4100_410098


namespace NUMINAMATH_CALUDE_problem_1_l4100_410004

theorem problem_1 : Real.sqrt 8 / Real.sqrt 2 + (Real.sqrt 5 + 3) * (Real.sqrt 5 - 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l4100_410004


namespace NUMINAMATH_CALUDE_third_arrangement_is_goat_monkey_donkey_l4100_410023

-- Define the animals
inductive Animal : Type
| Monkey : Animal
| Donkey : Animal
| Goat : Animal

-- Define a seating arrangement as a triple of animals
def Arrangement := (Animal × Animal × Animal)

-- Define the property of an animal being in a specific position
def isInPosition (a : Animal) (pos : Nat) (arr : Arrangement) : Prop :=
  match pos, arr with
  | 0, (x, _, _) => x = a
  | 1, (_, x, _) => x = a
  | 2, (_, _, x) => x = a
  | _, _ => False

-- Define the property that each animal has been in each position
def eachAnimalInEachPosition (arr1 arr2 arr3 : Arrangement) : Prop :=
  ∀ (a : Animal) (p : Nat), p < 3 → 
    isInPosition a p arr1 ∨ isInPosition a p arr2 ∨ isInPosition a p arr3

-- Main theorem
theorem third_arrangement_is_goat_monkey_donkey 
  (arr1 arr2 arr3 : Arrangement)
  (h1 : isInPosition Animal.Monkey 2 arr1)
  (h2 : isInPosition Animal.Donkey 1 arr2)
  (h3 : eachAnimalInEachPosition arr1 arr2 arr3) :
  arr3 = (Animal.Goat, Animal.Monkey, Animal.Donkey) :=
sorry

end NUMINAMATH_CALUDE_third_arrangement_is_goat_monkey_donkey_l4100_410023


namespace NUMINAMATH_CALUDE_supplies_to_budget_ratio_l4100_410075

def total_budget : ℚ := 3000
def food_fraction : ℚ := 1/3
def wages : ℚ := 1250

def supplies : ℚ := total_budget - (food_fraction * total_budget) - wages

theorem supplies_to_budget_ratio : 
  supplies / total_budget = 1/4 := by sorry

end NUMINAMATH_CALUDE_supplies_to_budget_ratio_l4100_410075


namespace NUMINAMATH_CALUDE_last_two_digits_28_l4100_410050

theorem last_two_digits_28 (n : ℕ) (h : Odd n) (h_pos : 0 < n) :
  2^(2*n) * (2^(2*n + 1) - 1) ≡ 28 [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_28_l4100_410050


namespace NUMINAMATH_CALUDE_inequality_proof_l4100_410090

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4100_410090


namespace NUMINAMATH_CALUDE_percentage_calculation_l4100_410047

theorem percentage_calculation (total : ℝ) (result : ℝ) (percentage : ℝ) :
  total = 50 →
  result = 2.125 →
  percentage = 4.25 →
  (percentage / 100) * total = result :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l4100_410047


namespace NUMINAMATH_CALUDE_factorial_different_remainders_l4100_410011

theorem factorial_different_remainders (n : ℕ) : n ≥ 2 →
  (∀ i j, 1 ≤ i ∧ i < j ∧ j < n → Nat.factorial i % n ≠ Nat.factorial j % n) ↔ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_different_remainders_l4100_410011


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l4100_410094

theorem simplify_complex_fraction (x : ℝ) 
  (h1 : x ≠ 4) 
  (h2 : x ≠ 2) 
  (h3 : x ≠ 5) 
  (h4 : x ≠ 3) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 
  (x - 3)^2 / ((x - 4) * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l4100_410094


namespace NUMINAMATH_CALUDE_power_of_three_l4100_410040

theorem power_of_three (a b : ℕ+) (h : 3^(a : ℕ) * 3^(b : ℕ) = 81) :
  (3^(a : ℕ))^(b : ℕ) = 3^4 := by
sorry

end NUMINAMATH_CALUDE_power_of_three_l4100_410040


namespace NUMINAMATH_CALUDE_circle_arrangement_impossibility_l4100_410082

theorem circle_arrangement_impossibility :
  ¬ ∃ (a : Fin 7 → ℕ),
    (∀ i : Fin 7, ∃ j : Fin 7, (a j + a ((j + 1) % 7) + a ((j + 2) % 7) = i + 1)) ∧
    (∀ i j : Fin 7, i ≠ j → 
      (a i + a ((i + 1) % 7) + a ((i + 2) % 7)) ≠ (a j + a ((j + 1) % 7) + a ((j + 2) % 7))) :=
by sorry

end NUMINAMATH_CALUDE_circle_arrangement_impossibility_l4100_410082


namespace NUMINAMATH_CALUDE_clock_strike_problem_l4100_410017

/-- Represents a clock that strikes at regular intervals. -/
structure Clock where
  interval : ℕ

/-- Calculates the time of the last strike given two clocks and total strikes. -/
def lastStrikeTime (clock1 clock2 : Clock) (totalStrikes : ℕ) : ℕ :=
  sorry

/-- Calculates the time between first and last strikes. -/
def timeBetweenStrikes (clock1 clock2 : Clock) (totalStrikes : ℕ) : ℕ :=
  sorry

theorem clock_strike_problem :
  let clock1 : Clock := { interval := 2 }
  let clock2 : Clock := { interval := 3 }
  let totalStrikes : ℕ := 13
  timeBetweenStrikes clock1 clock2 totalStrikes = 18 :=
by sorry

end NUMINAMATH_CALUDE_clock_strike_problem_l4100_410017


namespace NUMINAMATH_CALUDE_greatest_number_under_150_with_odd_factors_l4100_410042

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_under_150_with_odd_factors : 
  (∀ n : ℕ, n < 150 ∧ has_odd_number_of_factors n → n ≤ 144) ∧ 
  144 < 150 ∧ 
  has_odd_number_of_factors 144 :=
sorry

end NUMINAMATH_CALUDE_greatest_number_under_150_with_odd_factors_l4100_410042


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l4100_410019

theorem smallest_angle_in_triangle (d e f : ℝ) (F : ℝ) (h1 : d = 2) (h2 : e = 2) (h3 : f > 4 * Real.sqrt 2) :
  let y := Real.pi
  F ≥ y ∧ ∀ z, z < y → F > z :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l4100_410019


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4100_410093

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a > 1 → a^2 > 1) ∧ 
  (∃ a, a^2 > 1 ∧ ¬(a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4100_410093


namespace NUMINAMATH_CALUDE_division_problem_l4100_410092

theorem division_problem (dividend quotient divisor remainder : ℕ) 
  (h1 : remainder = 8)
  (h2 : divisor = 3 * remainder + 3)
  (h3 : dividend = 251)
  (h4 : dividend = divisor * quotient + remainder) :
  ∃ (m : ℕ), divisor = m * quotient ∧ m = 3 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l4100_410092


namespace NUMINAMATH_CALUDE_problem_statement_l4100_410020

theorem problem_statement (x : ℝ) (h : x^5 + x^4 + x = -1) :
  x^1997 + x^1998 + x^1999 + x^2000 + x^2001 + x^2002 + x^2003 + x^2004 + x^2005 + x^2006 + x^2007 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4100_410020


namespace NUMINAMATH_CALUDE_thirty_percent_of_hundred_l4100_410064

theorem thirty_percent_of_hundred : (30 : ℝ) = (30 / 100) * 100 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_hundred_l4100_410064


namespace NUMINAMATH_CALUDE_parallel_planes_k_value_l4100_410048

/-- Given two planes α and β with normal vectors n₁ and n₂ respectively,
    prove that if the planes are parallel, then k = 6. -/
theorem parallel_planes_k_value (n₁ n₂ : ℝ × ℝ × ℝ) (k : ℝ) :
  n₁ = (1, 2, -3) →
  n₂ = (-2, -4, k) →
  (∃ (c : ℝ), c ≠ 0 ∧ n₁ = c • n₂) →
  k = 6 := by sorry

end NUMINAMATH_CALUDE_parallel_planes_k_value_l4100_410048


namespace NUMINAMATH_CALUDE_statement_D_not_always_true_l4100_410052

-- Define the space
variable (Space : Type)

-- Define lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the specific lines and planes
variable (a b c : Line)
variable (α β : Plane)

-- State the theorem
theorem statement_D_not_always_true :
  ¬(∀ (b c : Line) (α : Plane),
    (subset b α ∧ ¬subset c α ∧ parallel_line_plane c α) → parallel b c) :=
by sorry

end NUMINAMATH_CALUDE_statement_D_not_always_true_l4100_410052


namespace NUMINAMATH_CALUDE_olyas_numbers_proof_l4100_410065

def first_number : ℕ := 929
def second_number : ℕ := 20
def third_number : ℕ := 2

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem olyas_numbers_proof :
  (100 ≤ first_number ∧ first_number < 1000) ∧
  (second_number = sum_of_digits first_number) ∧
  (third_number = sum_of_digits second_number) ∧
  (∃ (a b : ℕ), first_number = 100 * a + 10 * b + a ∧
                second_number = 10 * b + 0 ∧
                third_number = b) :=
by sorry

end NUMINAMATH_CALUDE_olyas_numbers_proof_l4100_410065


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l4100_410061

/-- Proves that a cement mixture with the given composition weighs 40 pounds -/
theorem cement_mixture_weight :
  ∀ (W : ℝ),
  (1/4 : ℝ) * W + (2/5 : ℝ) * W + 14 = W →
  W = 40 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l4100_410061


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l4100_410096

theorem sum_of_cubes_equation (x y : ℝ) :
  x^3 + 21*x*y + y^3 = 343 → x + y = 7 ∨ x + y = -14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l4100_410096


namespace NUMINAMATH_CALUDE_lance_licks_l4100_410012

/-- The number of licks it takes Dan to get to the center of a lollipop -/
def dan_licks : ℕ := 58

/-- The number of licks it takes Michael to get to the center of a lollipop -/
def michael_licks : ℕ := 63

/-- The number of licks it takes Sam to get to the center of a lollipop -/
def sam_licks : ℕ := 70

/-- The number of licks it takes David to get to the center of a lollipop -/
def david_licks : ℕ := 70

/-- The average number of licks it takes for all 5 people to get to the center of a lollipop -/
def average_licks : ℕ := 60

/-- The number of people in the group -/
def num_people : ℕ := 5

/-- The theorem stating how many licks it takes Lance to get to the center of a lollipop -/
theorem lance_licks : 
  (num_people * average_licks) - (dan_licks + michael_licks + sam_licks + david_licks) = 39 := by
  sorry

end NUMINAMATH_CALUDE_lance_licks_l4100_410012


namespace NUMINAMATH_CALUDE_slope_y_intercept_ratio_l4100_410014

/-- A line in the coordinate plane with slope m, y-intercept b, and x-intercept 2 -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  x_intercept_eq_two : m * 2 + b = 0  -- condition for x-intercept = 2

/-- The slope is some fraction of the y-intercept -/
def slope_fraction (k : Line) (c : ℝ) : Prop :=
  k.m = c * k.b

theorem slope_y_intercept_ratio (k : Line) :
  ∃ c : ℝ, slope_fraction k c ∧ c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_slope_y_intercept_ratio_l4100_410014


namespace NUMINAMATH_CALUDE_cindys_age_l4100_410071

/-- Given the ages of siblings, prove Cindy's age -/
theorem cindys_age (cindy jan marcia greg : ℕ) 
  (h1 : jan = cindy + 2)
  (h2 : marcia = 2 * jan)
  (h3 : greg = marcia + 2)
  (h4 : greg = 16) :
  cindy = 5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_age_l4100_410071


namespace NUMINAMATH_CALUDE_sphere_area_equals_volume_l4100_410067

theorem sphere_area_equals_volume (r : ℝ) (h : r > 0) :
  4 * Real.pi * r^2 = (4/3) * Real.pi * r^3 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_area_equals_volume_l4100_410067


namespace NUMINAMATH_CALUDE_third_number_proof_l4100_410081

theorem third_number_proof (a b c : ℝ) : 
  a = 6 → b = 16 → (a + b + c) / 3 = 13 → c = 17 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l4100_410081


namespace NUMINAMATH_CALUDE_angle_measure_proof_l4100_410039

theorem angle_measure_proof (A B : ℝ) : 
  (A = B ∨ A + B = 180) →  -- Parallel sides condition
  A = 3 * B - 20 →         -- Relationship between A and B
  A = 10 ∨ A = 130 :=      -- Conclusion
by sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l4100_410039


namespace NUMINAMATH_CALUDE_tomato_pick_ratio_l4100_410085

/-- Represents the number of tomatoes picked in each week and the remaining tomatoes -/
structure TomatoPicks where
  initial : ℕ
  first_week : ℕ
  second_week : ℕ
  third_week : ℕ
  remaining : ℕ

/-- Calculates the ratio of tomatoes picked in the third week to the second week -/
def pick_ratio (picks : TomatoPicks) : ℚ :=
  picks.third_week / picks.second_week

/-- Theorem stating the ratio of tomatoes picked in the third week to the second week -/
theorem tomato_pick_ratio : 
  ∀ (picks : TomatoPicks), 
  picks.initial = 100 ∧ 
  picks.first_week = picks.initial / 4 ∧
  picks.second_week = 20 ∧
  picks.remaining = 15 ∧
  picks.initial = picks.first_week + picks.second_week + picks.third_week + picks.remaining
  → pick_ratio picks = 2 := by
  sorry

#check tomato_pick_ratio

end NUMINAMATH_CALUDE_tomato_pick_ratio_l4100_410085


namespace NUMINAMATH_CALUDE_overlap_area_of_intersecting_rectangles_l4100_410060

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of overlap between two rectangles intersecting at 45 degrees -/
noncomputable def overlapArea (r1 r2 : Rectangle) : ℝ :=
  min r1.width r2.width * min r1.height r2.height

/-- The theorem stating the area of the overlapping region -/
theorem overlap_area_of_intersecting_rectangles :
  let r1 : Rectangle := { width := 3, height := 12 }
  let r2 : Rectangle := { width := 4, height := 10 }
  rectangleArea r1 + rectangleArea r2 - overlapArea r1 r2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_of_intersecting_rectangles_l4100_410060


namespace NUMINAMATH_CALUDE_unique_operation_equals_one_l4100_410027

theorem unique_operation_equals_one : 
  (-3 + (-3) ≠ 1) ∧ 
  (-3 - (-3) ≠ 1) ∧ 
  (-3 / (-3) = 1) ∧ 
  (-3 * (-3) ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_operation_equals_one_l4100_410027


namespace NUMINAMATH_CALUDE_area_inequalities_l4100_410034

/-- An acute-angled triangle -/
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

/-- Area of the orthic triangle -/
def orthic_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the tangential triangle -/
def tangential_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the contact triangle -/
def contact_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the excentral triangle -/
def excentral_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the medial triangle -/
def medial_area (t : AcuteTriangle) : ℝ := sorry

/-- A triangle is equilateral if all its angles are equal -/
def is_equilateral (t : AcuteTriangle) : Prop :=
  t.A = t.B ∧ t.B = t.C

/-- The main theorem -/
theorem area_inequalities (t : AcuteTriangle) :
  orthic_area t ≤ tangential_area t ∧
  tangential_area t = contact_area t ∧
  contact_area t ≤ excentral_area t ∧
  excentral_area t ≤ medial_area t ∧
  (orthic_area t = medial_area t ↔ is_equilateral t) :=
sorry

end NUMINAMATH_CALUDE_area_inequalities_l4100_410034


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l4100_410073

theorem triangle_square_side_ratio (perimeter : ℝ) (triangle_side square_side : ℝ) : 
  perimeter > 0 → 
  triangle_side * 3 = perimeter → 
  square_side * 4 = perimeter → 
  triangle_side / square_side = 4 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l4100_410073


namespace NUMINAMATH_CALUDE_coordinates_of_N_l4100_410089

-- Define the point M
def M : ℝ × ℝ := (-1, 3)

-- Define the length of MN
def MN_length : ℝ := 4

-- Define the property that MN is parallel to y-axis
def parallel_to_y_axis (N : ℝ × ℝ) : Prop :=
  N.1 = M.1

-- Define the distance between M and N
def distance (N : ℝ × ℝ) : ℝ :=
  |N.2 - M.2|

-- Theorem statement
theorem coordinates_of_N :
  ∃ N : ℝ × ℝ, parallel_to_y_axis N ∧ distance N = MN_length ∧ (N = (-1, -1) ∨ N = (-1, 7)) :=
sorry

end NUMINAMATH_CALUDE_coordinates_of_N_l4100_410089


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_S_l4100_410037

/-- The product of all non-zero digits of a positive integer -/
def p (n : ℕ+) : ℕ :=
  sorry

/-- The sum of p(n) for n from 1 to 999 -/
def S : ℕ :=
  (Finset.range 999).sum (fun i => p ⟨i + 1, Nat.succ_pos i⟩)

/-- 103 is the largest prime factor of S -/
theorem largest_prime_factor_of_S :
  ∃ (m : ℕ), S = 103 * m ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ S → q ≤ 103 :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_S_l4100_410037


namespace NUMINAMATH_CALUDE_cylinder_prism_pyramid_elements_l4100_410079

/-- Represents a cylinder unwrapped into a prism with a pyramid attached -/
structure CylinderPrismPyramid where
  /-- Number of faces in the original prism -/
  prism_faces : ℕ
  /-- Number of edges in the original prism -/
  prism_edges : ℕ
  /-- Number of vertices in the original prism -/
  prism_vertices : ℕ
  /-- Number of faces added by the pyramid -/
  pyramid_faces : ℕ
  /-- Number of edges added by the pyramid -/
  pyramid_edges : ℕ
  /-- Number of vertices added by the pyramid -/
  pyramid_vertices : ℕ

/-- The total number of geometric elements in the CylinderPrismPyramid -/
def total_elements (cpp : CylinderPrismPyramid) : ℕ :=
  cpp.prism_faces + cpp.prism_edges + cpp.prism_vertices +
  cpp.pyramid_faces + cpp.pyramid_edges + cpp.pyramid_vertices

/-- Theorem stating that the total number of elements is 31 -/
theorem cylinder_prism_pyramid_elements :
  ∀ cpp : CylinderPrismPyramid,
  cpp.prism_faces = 5 ∧ 
  cpp.prism_edges = 10 ∧ 
  cpp.prism_vertices = 8 ∧
  cpp.pyramid_faces = 3 ∧
  cpp.pyramid_edges = 4 ∧
  cpp.pyramid_vertices = 1 →
  total_elements cpp = 31 := by
  sorry

#check cylinder_prism_pyramid_elements

end NUMINAMATH_CALUDE_cylinder_prism_pyramid_elements_l4100_410079


namespace NUMINAMATH_CALUDE_investment_sum_l4100_410032

theorem investment_sum (raghu_investment : ℝ) : raghu_investment = 2400 →
  let trishul_investment := raghu_investment * 0.9
  let vishal_investment := trishul_investment * 1.1
  raghu_investment + trishul_investment + vishal_investment = 6936 := by
sorry

end NUMINAMATH_CALUDE_investment_sum_l4100_410032


namespace NUMINAMATH_CALUDE_people_on_boats_l4100_410078

theorem people_on_boats (total_boats : Nat) (boats_with_four : Nat) (boats_with_five : Nat)
  (h1 : total_boats = 7)
  (h2 : boats_with_four = 4)
  (h3 : boats_with_five = 3)
  (h4 : total_boats = boats_with_four + boats_with_five) :
  boats_with_four * 4 + boats_with_five * 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_people_on_boats_l4100_410078


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4100_410031

/-- Given a geometric sequence where the first term is x, the second term is 3x + 3, 
    and the third term is 5x + 5, the fourth term of this sequence is -5/4. -/
theorem geometric_sequence_fourth_term (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (3*x + 3) = x * r ∧ (5*x + 5) = (3*x + 3) * r) → 
  ∃ t : ℝ, t = -5/4 ∧ t = (5*x + 5) * (3*x + 3) / x := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4100_410031


namespace NUMINAMATH_CALUDE_problem_statement_l4100_410002

theorem problem_statement (p q : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
   {a : ℝ | (a + p) * (a + q) * (a + 20) / (a + 4)^2 = 0} = {x, y, z}) ∧
  (∃! x : ℝ, (x + 3*p) * (x + 4) * (x + 10) / ((x + q) * (x + 20)) = 0) →
  100 * p + q = 430 / 3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4100_410002


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l4100_410044

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 1

-- Theorem statement
theorem max_value_of_f_in_interval :
  ∃ (m : ℝ), m = 2 ∧ ∀ x ∈ interval, f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l4100_410044


namespace NUMINAMATH_CALUDE_always_greater_than_m_l4100_410070

theorem always_greater_than_m (m : ℚ) : m + 2 > m := by
  sorry

end NUMINAMATH_CALUDE_always_greater_than_m_l4100_410070


namespace NUMINAMATH_CALUDE_goods_train_speed_l4100_410007

/-- Calculates the speed of a goods train given the conditions described in the problem -/
theorem goods_train_speed
  (passenger_train_speed : ℝ)
  (goods_train_length : ℝ)
  (passing_time : ℝ)
  (h_passenger_speed : passenger_train_speed = 100)
  (h_goods_length : goods_train_length = 400)
  (h_passing_time : passing_time = 12) :
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 20 ∧
    (goods_train_speed + passenger_train_speed) * passing_time / 3.6 = goods_train_length :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l4100_410007


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_a_greater_than_one_l4100_410087

/-- Represents a quadratic function of the form f(x) = 2ax^2 - x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

/-- Condition for exactly one solution in (0, 1) -/
def has_exactly_one_solution_in_interval (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo 0 1 ∧ f a x = 0

theorem quadratic_solution_implies_a_greater_than_one :
  ∀ a : ℝ, has_exactly_one_solution_in_interval a → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_a_greater_than_one_l4100_410087


namespace NUMINAMATH_CALUDE_quadratic_comparison_l4100_410058

/-- Proves that for a quadratic function y = a(x-1)^2 + 3 where a < 0,
    if (-1, y₁) and (2, y₂) are points on the graph, then y₁ < y₂ -/
theorem quadratic_comparison (a : ℝ) (y₁ y₂ : ℝ)
    (h₁ : a < 0)
    (h₂ : y₁ = a * (-1 - 1)^2 + 3)
    (h₃ : y₂ = a * (2 - 1)^2 + 3) :
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_comparison_l4100_410058


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l4100_410009

theorem quadratic_root_proof (x : ℝ) : 
  x = (-31 - Real.sqrt 481) / 12 → 6 * x^2 + 31 * x + 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l4100_410009


namespace NUMINAMATH_CALUDE_circle_polar_to_cartesian_l4100_410080

/-- Given a circle with polar equation ρ = 2cos θ, its Cartesian equation is (x-1)^2 + y^2 = 1 -/
theorem circle_polar_to_cartesian :
  ∀ (x y ρ θ : ℝ),
  (ρ = 2 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  ((x - 1)^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_polar_to_cartesian_l4100_410080


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l4100_410057

theorem inscribed_hexagon_area (circle_area : ℝ) (h : circle_area = 576 * Real.pi) :
  let r := Real.sqrt (circle_area / Real.pi)
  let hexagon_area := 6 * ((r^2 * Real.sqrt 3) / 4)
  hexagon_area = 864 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l4100_410057


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l4100_410029

/-- The lateral surface area of a cylinder with base radius 1 and slant height 2 is 4π. -/
theorem cylinder_lateral_surface_area : 
  ∀ (r h : ℝ), r = 1 → h = 2 → 2 * π * r * h = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l4100_410029


namespace NUMINAMATH_CALUDE_ritas_swimming_hours_l4100_410028

/-- Calculates the total swimming hours for Rita based on given conditions -/
def total_swimming_hours (backstroke_hours breaststroke_hours butterfly_hours monthly_freestyle_sidestroke_hours months : ℕ) : ℕ :=
  backstroke_hours + breaststroke_hours + butterfly_hours + (monthly_freestyle_sidestroke_hours * months)

/-- Proves that Rita's total swimming hours equal 1500 -/
theorem ritas_swimming_hours :
  total_swimming_hours 50 9 121 220 6 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ritas_swimming_hours_l4100_410028


namespace NUMINAMATH_CALUDE_room_population_problem_l4100_410066

theorem room_population_problem (initial_men initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  (2 * (initial_women - 3)) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_room_population_problem_l4100_410066


namespace NUMINAMATH_CALUDE_distribution_count_correct_l4100_410005

/-- The number of ways to distribute 4 distinct objects into 4 distinct containers 
    such that exactly one container contains 2 objects and the others contain 1 object each -/
def distributionCount : ℕ := 144

/-- The number of universities -/
def numUniversities : ℕ := 4

/-- The number of students -/
def numStudents : ℕ := 4

theorem distribution_count_correct :
  distributionCount = 
    (numStudents.choose 2) * (numUniversities * (numUniversities - 1) * (numUniversities - 2)) :=
by sorry

end NUMINAMATH_CALUDE_distribution_count_correct_l4100_410005


namespace NUMINAMATH_CALUDE_last_remaining_200_l4100_410095

/-- The last remaining number after the marking process -/
def lastRemainingNumber (n : ℕ) : ℕ :=
  if n ≤ 1 then n else 2 * lastRemainingNumber ((n + 1) / 2)

/-- The theorem stating that for 200 numbers, the last remaining is 128 -/
theorem last_remaining_200 :
  lastRemainingNumber 200 = 128 := by
  sorry

end NUMINAMATH_CALUDE_last_remaining_200_l4100_410095


namespace NUMINAMATH_CALUDE_rectangle_perimeter_bound_l4100_410026

theorem rectangle_perimeter_bound (a b : ℝ) (h : a > 0 ∧ b > 0) (area_gt_perimeter : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_bound_l4100_410026


namespace NUMINAMATH_CALUDE_kids_went_home_l4100_410016

theorem kids_went_home (initial_kids : ℝ) (remaining_kids : ℕ) : 
  initial_kids = 22.0 → remaining_kids = 8 → initial_kids - remaining_kids = 14 := by
  sorry

end NUMINAMATH_CALUDE_kids_went_home_l4100_410016


namespace NUMINAMATH_CALUDE_tetrahedra_triangle_inequality_l4100_410063

/-- A finite graph -/
structure FiniteGraph where
  -- We don't need to specify the internal structure of the graph
  -- as it's not directly used in the theorem statement

/-- The number of triangles in a finite graph -/
def numTriangles (G : FiniteGraph) : ℕ := sorry

/-- The number of tetrahedra in a finite graph -/
def numTetrahedra (G : FiniteGraph) : ℕ := sorry

/-- The main theorem stating the inequality between tetrahedra and triangles -/
theorem tetrahedra_triangle_inequality (G : FiniteGraph) :
  (numTetrahedra G)^3 ≤ (3/32) * (numTriangles G)^4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedra_triangle_inequality_l4100_410063


namespace NUMINAMATH_CALUDE_area_between_circles_l4100_410046

-- Define the radius of the inner circle
def inner_radius : ℝ := 2

-- Define the radius of the outer circle
def outer_radius : ℝ := 2 * inner_radius

-- Define the width of the gray region
def width : ℝ := outer_radius - inner_radius

-- Theorem statement
theorem area_between_circles (h : width = 2) : 
  π * outer_radius^2 - π * inner_radius^2 = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l4100_410046


namespace NUMINAMATH_CALUDE_circle_tangent_theorem_l4100_410091

-- Define the circle
def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*a*x + 2*y - 1 = 0}

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (-5, a)

-- Define the condition for the tangent lines
def TangentCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) + (x₁ + x₂ - 2) / (y₁ + y₂) = 0

theorem circle_tangent_theorem :
  ∀ a : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ,
    P a ∈ Circle a →
    (x₁, y₁) ∈ Circle a →
    (x₂, y₂) ∈ Circle a →
    TangentCondition x₁ y₁ x₂ y₂ →
    a = 3 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_theorem_l4100_410091


namespace NUMINAMATH_CALUDE_vacation_emails_l4100_410025

theorem vacation_emails (a₁ : ℝ) (r : ℝ) (n : ℕ) (h1 : a₁ = 16) (h2 : r = 1/2) (h3 : n = 4) :
  a₁ * (1 - r^n) / (1 - r) = 30 := by
  sorry

end NUMINAMATH_CALUDE_vacation_emails_l4100_410025


namespace NUMINAMATH_CALUDE_complement_of_union_l4100_410021

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {4, 5, 6}

theorem complement_of_union : 
  U \ (A ∪ B) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l4100_410021


namespace NUMINAMATH_CALUDE_last_installment_theorem_l4100_410038

/-- Represents the installment payment plan for a TV set. -/
structure TVInstallmentPlan where
  total_price : ℕ
  num_installments : ℕ
  installment_amount : ℕ
  interest_rate : ℚ
  first_installment_at_purchase : Bool

/-- Calculates the value of the last installment in a TV installment plan. -/
def last_installment_value (plan : TVInstallmentPlan) : ℕ :=
  plan.installment_amount

/-- Theorem stating that the last installment value is equal to the regular installment amount. -/
theorem last_installment_theorem (plan : TVInstallmentPlan)
  (h1 : plan.total_price = 10000)
  (h2 : plan.num_installments = 20)
  (h3 : plan.installment_amount = 1000)
  (h4 : plan.interest_rate = 6 / 100)
  (h5 : plan.first_installment_at_purchase = true) :
  last_installment_value plan = 1000 := by
  sorry

#eval last_installment_value {
  total_price := 10000,
  num_installments := 20,
  installment_amount := 1000,
  interest_rate := 6 / 100,
  first_installment_at_purchase := true
}

end NUMINAMATH_CALUDE_last_installment_theorem_l4100_410038


namespace NUMINAMATH_CALUDE_recruit_line_unique_solution_l4100_410086

/-- Represents the position of a person in the line of recruits -/
structure Position :=
  (front : ℕ)  -- number of people in front
  (behind : ℕ) -- number of people behind

/-- The line of recruits -/
structure RecruitLine :=
  (total : ℕ)
  (peter : Position)
  (nikolai : Position)
  (denis : Position)

/-- Conditions of the problem -/
def problem_conditions (line : RecruitLine) : Prop :=
  line.peter.front = 50 ∧
  line.nikolai.front = 100 ∧
  line.denis.front = 170 ∧
  (line.peter.behind = 4 * line.denis.behind ∨
   line.nikolai.behind = 4 * line.denis.behind ∨
   line.peter.behind = 4 * line.nikolai.behind) ∧
  line.total = line.denis.front + 1 + line.denis.behind

/-- The theorem to be proved -/
theorem recruit_line_unique_solution :
  ∃! line : RecruitLine, problem_conditions line ∧ line.total = 301 :=
sorry

end NUMINAMATH_CALUDE_recruit_line_unique_solution_l4100_410086


namespace NUMINAMATH_CALUDE_equation_of_l_equation_of_l_l4100_410053

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3*x + 4*y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2*x - 3*y + 8 = 0

-- Define the perpendicular line to l
def perp_line (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define the point of symmetry
def sym_point : ℝ × ℝ := (1, -1)

-- Theorem for the equation of line l
theorem equation_of_l :
  ∃ (M : ℝ × ℝ), l₁ M.1 M.2 ∧ l₂ M.1 M.2 →
  ∀ (x y : ℝ), (x - 2*y + 5 = 0) ↔ (∃ (k : ℝ), x - M.1 = k * (-2) ∧ y - M.2 = k * 1) :=
sorry

-- Theorem for the equation of line l'
theorem equation_of_l' :
  ∀ (x y : ℝ), (3*x + 4*y + 7 = 0) ↔
  (∃ (x' y' : ℝ), l₁ x' y' ∧ x = 2 * sym_point.1 - x' ∧ y = 2 * sym_point.2 - y') :=
sorry

end NUMINAMATH_CALUDE_equation_of_l_equation_of_l_l4100_410053


namespace NUMINAMATH_CALUDE_triangle_inequality_l4100_410006

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4100_410006


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l4100_410018

def matrix_element (i j : Nat) : Int :=
  if j % i = 0 then 1 else -1

def sum_3j : Int :=
  (matrix_element 3 2) + (matrix_element 3 3) + (matrix_element 3 4) + (matrix_element 3 5)

def sum_i4 : Int :=
  (matrix_element 2 4) + (matrix_element 3 4) + (matrix_element 4 4)

theorem matrix_sum_theorem : sum_3j + sum_i4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l4100_410018


namespace NUMINAMATH_CALUDE_andrew_age_proof_l4100_410035

/-- Andrew's age in years -/
def andrew_age : ℚ := 30 / 7

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℚ := 15 * andrew_age

theorem andrew_age_proof :
  andrew_age = 30 / 7 ∧
  grandfather_age = 15 * andrew_age ∧
  grandfather_age - andrew_age = 60 :=
sorry

end NUMINAMATH_CALUDE_andrew_age_proof_l4100_410035


namespace NUMINAMATH_CALUDE_systematic_sampling_l4100_410045

/-- Systematic sampling problem -/
theorem systematic_sampling
  (total_students : ℕ)
  (num_groups : ℕ)
  (group_size : ℕ)
  (group_16_number : ℕ)
  (h1 : total_students = 160)
  (h2 : num_groups = 20)
  (h3 : group_size = total_students / num_groups)
  (h4 : group_16_number = 126) :
  ∃ (first_group_number : ℕ),
    first_group_number ∈ Finset.range group_size ∧
    first_group_number + (16 - 1) * group_size = group_16_number :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l4100_410045


namespace NUMINAMATH_CALUDE_eggs_for_cake_l4100_410013

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of eggs Megan bought -/
def bought : ℕ := dozen

/-- The number of eggs Megan's neighbor gave her -/
def given : ℕ := dozen

/-- The number of eggs Megan used for an omelet -/
def omelet : ℕ := 2

/-- The number of eggs Megan plans to use for her next meals -/
def meal_plan : ℕ := 3 * 3

theorem eggs_for_cake :
  ∃ (cake : ℕ),
    bought + given - omelet - (bought + given - omelet) / 2 - meal_plan = cake ∧
    cake = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_for_cake_l4100_410013


namespace NUMINAMATH_CALUDE_triangle_inequality_inner_point_l4100_410055

/-- Given a triangle ABC and a point P on side AB, prove that PC · AB < PA · BC + PB · AC -/
theorem triangle_inequality_inner_point (A B C P : ℝ × ℝ) : 
  (P.1 > A.1 ∧ P.1 < B.1) → -- P is an inner point of AB
  (dist P C * dist A B < dist P A * dist B C + dist P B * dist A C) := by
  sorry

#check triangle_inequality_inner_point

end NUMINAMATH_CALUDE_triangle_inequality_inner_point_l4100_410055


namespace NUMINAMATH_CALUDE_continuity_at_one_l4100_410036

def f (x : ℝ) := -3 * x^2 - 6

theorem continuity_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_one_l4100_410036


namespace NUMINAMATH_CALUDE_gcd_of_135_and_81_l4100_410076

theorem gcd_of_135_and_81 : Nat.gcd 135 81 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_135_and_81_l4100_410076


namespace NUMINAMATH_CALUDE_seminar_scheduling_l4100_410072

theorem seminar_scheduling (n : ℕ) (h : n = 5) : 
  (n! / 2 : ℕ) = 60 :=
sorry

end NUMINAMATH_CALUDE_seminar_scheduling_l4100_410072


namespace NUMINAMATH_CALUDE_square_a_minus_2b_l4100_410022

theorem square_a_minus_2b (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a - 2*b)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_a_minus_2b_l4100_410022
