import Mathlib

namespace NUMINAMATH_CALUDE_polygon_sides_l2535_253512

theorem polygon_sides (n : ℕ) : n > 2 → (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2535_253512


namespace NUMINAMATH_CALUDE_profit_percentage_is_18_percent_l2535_253513

def cost_price : ℝ := 460
def selling_price : ℝ := 542.8

theorem profit_percentage_is_18_percent :
  (selling_price - cost_price) / cost_price * 100 = 18 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_is_18_percent_l2535_253513


namespace NUMINAMATH_CALUDE_power_function_through_point_l2535_253506

/-- A power function that passes through the point (9, 3) -/
def f (x : ℝ) : ℝ := x^(1/2)

/-- Theorem stating that f(9) = 3 -/
theorem power_function_through_point : f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2535_253506


namespace NUMINAMATH_CALUDE_exists_x0_implies_a_value_l2535_253540

noncomputable section

open Real

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x + exp (x - a)
def g (a x : ℝ) : ℝ := log (x + 2) - 4 * exp (a - x)

-- State the theorem
theorem exists_x0_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) →
  a = -log 2 - 1 := by
sorry

end

end NUMINAMATH_CALUDE_exists_x0_implies_a_value_l2535_253540


namespace NUMINAMATH_CALUDE_income_comparison_l2535_253598

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mary = tim * (1 + 0.7)) :
  mary = juan * 1.02 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l2535_253598


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2535_253599

def U : Finset Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Finset Nat := {3, 4, 5}
def B : Finset Nat := {1, 3, 6}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2535_253599


namespace NUMINAMATH_CALUDE_substitution_remainder_l2535_253580

/-- Calculates the number of substitution combinations for a given number of substitutions -/
def substitutionCombinations (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 11 * (23 - k) * substitutionCombinations k

/-- The total number of substitution combinations for up to 5 substitutions -/
def totalCombinations : ℕ :=
  (List.range 6).map substitutionCombinations |>.sum

/-- Theorem stating the remainder when dividing the total number of substitution combinations by 1000 -/
theorem substitution_remainder :
  totalCombinations % 1000 = 586 := by
  sorry

end NUMINAMATH_CALUDE_substitution_remainder_l2535_253580


namespace NUMINAMATH_CALUDE_infinitely_many_special_numbers_l2535_253585

/-- A natural number n such that n^2 + 1 has no divisors of the form k^2 + 1 except 1 and itself. -/
def SpecialNumber (n : ℕ) : Prop :=
  ∀ k : ℕ, k^2 + 1 ∣ n^2 + 1 → k^2 + 1 = 1 ∨ k^2 + 1 = n^2 + 1

/-- The set of SpecialNumbers is infinite. -/
theorem infinitely_many_special_numbers : Set.Infinite {n : ℕ | SpecialNumber n} := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_special_numbers_l2535_253585


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2535_253588

theorem max_value_of_expression (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_one : a + b + c + d = 1) :
  (∀ x y z w : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → w ≥ 0 → x + y + z + w = 1 →
    (x*y)/(x+y) + (x*z)/(x+z) + (x*w)/(x+w) + 
    (y*z)/(y+z) + (y*w)/(y+w) + (z*w)/(z+w) ≤ 1/2) ∧
  ((a*b)/(a+b) + (a*c)/(a+c) + (a*d)/(a+d) + 
   (b*c)/(b+c) + (b*d)/(b+d) + (c*d)/(c+d) = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2535_253588


namespace NUMINAMATH_CALUDE_peach_difference_l2535_253502

theorem peach_difference (audrey_peaches paul_peaches : ℕ) 
  (h1 : audrey_peaches = 26) 
  (h2 : paul_peaches = 48) : 
  paul_peaches - audrey_peaches = 22 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l2535_253502


namespace NUMINAMATH_CALUDE_car_speed_problem_l2535_253560

/-- Proves that given a car traveling for two hours with a speed of 90 km/h in the first hour
    and an average speed of 72.5 km/h over the two hours, the speed in the second hour must be 55 km/h. -/
theorem car_speed_problem (speed_first_hour : ℝ) (average_speed : ℝ) (speed_second_hour : ℝ) :
  speed_first_hour = 90 →
  average_speed = 72.5 →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_second_hour = 55 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2535_253560


namespace NUMINAMATH_CALUDE_square_triangle_ratio_l2535_253575

theorem square_triangle_ratio (a : ℝ) (h : a > 0) :
  let square_side := a
  let triangle_leg := a * Real.sqrt 2
  let triangle_hypotenuse := triangle_leg * Real.sqrt 2
  triangle_hypotenuse / square_side = 2 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_ratio_l2535_253575


namespace NUMINAMATH_CALUDE_box_dimensions_l2535_253592

theorem box_dimensions (a b c : ℝ) 
  (h1 : a + c = 17)
  (h2 : a + b = 13)
  (h3 : b + c = 20)
  (h4 : a < b)
  (h5 : b < c) :
  a = 5 ∧ b = 8 ∧ c = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_l2535_253592


namespace NUMINAMATH_CALUDE_first_prize_tickets_characterization_l2535_253597

def is_valid_ticket (n : Nat) : Prop := n ≥ 0 ∧ n ≤ 9999

def is_first_prize (n : Nat) : Prop := n % 1000 = 418

def first_prize_tickets : Set Nat :=
  {n : Nat | is_valid_ticket n ∧ is_first_prize n}

theorem first_prize_tickets_characterization :
  first_prize_tickets = {0418, 1418, 2418, 3418, 4418, 5418, 6418, 7418, 8418, 9418} :=
by sorry

end NUMINAMATH_CALUDE_first_prize_tickets_characterization_l2535_253597


namespace NUMINAMATH_CALUDE_first_part_distance_is_18_l2535_253546

/-- Represents a cyclist's trip with given parameters -/
structure CyclistTrip where
  totalTime : ℝ
  speed1 : ℝ
  speed2 : ℝ
  distance2 : ℝ
  returnSpeed : ℝ

/-- Calculates the distance of the first part of the trip -/
def firstPartDistance (trip : CyclistTrip) : ℝ :=
  sorry

/-- Theorem stating that the first part of the trip is 18 miles long -/
theorem first_part_distance_is_18 (trip : CyclistTrip) 
  (h1 : trip.totalTime = 7.2)
  (h2 : trip.speed1 = 9)
  (h3 : trip.speed2 = 10)
  (h4 : trip.distance2 = 12)
  (h5 : trip.returnSpeed = 7.5) :
  firstPartDistance trip = 18 :=
sorry

end NUMINAMATH_CALUDE_first_part_distance_is_18_l2535_253546


namespace NUMINAMATH_CALUDE_fraction_invariance_l2535_253591

theorem fraction_invariance (a b m n : ℚ) (h : b ≠ 0) (h' : b + n ≠ 0) :
  a / b = m / n → (a + m) / (b + n) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_invariance_l2535_253591


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2535_253522

/-- The sum of an arithmetic sequence with first term a, last term l, and common difference d -/
def arithmetic_sum (a l d : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

/-- The theorem stating that the remainder of the sum of the given arithmetic sequence when divided by 8 is 2 -/
theorem arithmetic_sequence_sum_remainder :
  (arithmetic_sum 3 299 8) % 8 = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2535_253522


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l2535_253543

theorem prime_square_mod_180 (p : Nat) (h_prime : Prime p) (h_gt_5 : p > 5) :
  p ^ 2 % 180 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l2535_253543


namespace NUMINAMATH_CALUDE_share_ratio_l2535_253550

/-- Proves that given a total amount of 527 and three shares A = 372, B = 93, and C = 62, 
    the ratio of A's share to B's share is 4:1. -/
theorem share_ratio (total : ℕ) (A B C : ℕ) 
  (h_total : total = 527)
  (h_A : A = 372)
  (h_B : B = 93)
  (h_C : C = 62)
  (h_sum : A + B + C = total) : 
  A / B = 4 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l2535_253550


namespace NUMINAMATH_CALUDE_even_number_less_than_square_l2535_253561

theorem even_number_less_than_square (m : ℕ) (h1 : m > 1) (h2 : Even m) : m < m^2 := by
  sorry

end NUMINAMATH_CALUDE_even_number_less_than_square_l2535_253561


namespace NUMINAMATH_CALUDE_sales_increase_percentage_l2535_253589

theorem sales_increase_percentage (price_reduction : ℝ) (receipts_increase : ℝ) : 
  price_reduction = 30 → receipts_increase = 5 → 
  ∃ (sales_increase : ℝ), sales_increase = 50 ∧ 
  (100 - price_reduction) / 100 * (1 + sales_increase / 100) = 1 + receipts_increase / 100 :=
by sorry

end NUMINAMATH_CALUDE_sales_increase_percentage_l2535_253589


namespace NUMINAMATH_CALUDE_digit_2023_of_7_18_l2535_253573

/-- The 2023rd digit past the decimal point in the decimal expansion of 7/18 is 3 -/
theorem digit_2023_of_7_18 : ∃ (d : ℕ), d = 3 ∧ 
  (∃ (a b : ℕ+) (s : Finset ℕ), 
    (7 : ℚ) / 18 = (a : ℚ) / b ∧ 
    s.card = 2023 ∧ 
    (∀ n ∈ s, (10 ^ n * ((7 : ℚ) / 18) % 1).floor % 10 = d) ∧
    (∀ m < 2023, m ∉ s)) :=
by sorry

end NUMINAMATH_CALUDE_digit_2023_of_7_18_l2535_253573


namespace NUMINAMATH_CALUDE_gcd_of_4557_1953_5115_l2535_253554

theorem gcd_of_4557_1953_5115 : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_4557_1953_5115_l2535_253554


namespace NUMINAMATH_CALUDE_percentage_problem_l2535_253539

theorem percentage_problem : ∃ x : ℝ, 
  (x / 100) * 150 - (20 / 100) * 250 = 43 ∧ 
  x = 62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2535_253539


namespace NUMINAMATH_CALUDE_x_over_y_is_negative_one_l2535_253555

theorem x_over_y_is_negative_one (x y : ℝ) 
  (h1 : 3 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 8) 
  (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -1 := by sorry

end NUMINAMATH_CALUDE_x_over_y_is_negative_one_l2535_253555


namespace NUMINAMATH_CALUDE_longest_altitudes_sum_is_14_l2535_253574

/-- A triangle with sides 6, 8, and 10 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = 6)
  (hb : b = 8)
  (hc : c = 10)

/-- The sum of the lengths of the two longest altitudes in the triangle -/
def longest_altitudes_sum (t : Triangle) : ℝ := sorry

/-- Theorem stating that the sum of the lengths of the two longest altitudes is 14 -/
theorem longest_altitudes_sum_is_14 (t : Triangle) : longest_altitudes_sum t = 14 := by
  sorry

end NUMINAMATH_CALUDE_longest_altitudes_sum_is_14_l2535_253574


namespace NUMINAMATH_CALUDE_expression_simplification_l2535_253596

theorem expression_simplification (x y z : ℝ) (h : y ≠ 0) :
  (6 * x^3 * y^4 * z - 4 * x^2 * y^3 * z + 2 * x * y^3) / (2 * x * y^3) = 3 * x^2 * y * z - 2 * x * z + 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2535_253596


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l2535_253509

theorem quadratic_inequality_always_nonnegative :
  ∀ x : ℝ, 4 * x^2 - 4 * x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l2535_253509


namespace NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l2535_253572

/-- Definition of the complex number z in terms of m -/
def z (m : ℝ) : ℂ := Complex.mk (3*m - 2) (m - 1)

/-- z is purely imaginary if and only if m = 2/3 -/
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * Complex.im (z m) ↔ m = 2/3 := by
  sorry

/-- z lies in the fourth quadrant if and only if 2/3 < m < 1 -/
theorem z_in_fourth_quadrant (m : ℝ) : 
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0) ↔ (2/3 < m ∧ m < 1) := by
  sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l2535_253572


namespace NUMINAMATH_CALUDE_choir_size_proof_l2535_253526

theorem choir_size_proof : Nat.lcm (Nat.lcm 9 10) 11 = 990 := by
  sorry

end NUMINAMATH_CALUDE_choir_size_proof_l2535_253526


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2535_253511

/-- The set of digits to choose from -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- Predicate to check if a number is odd -/
def is_odd (n : Nat) : Bool := n % 2 = 1

/-- Predicate to check if a number is even -/
def is_even (n : Nat) : Bool := n % 2 = 0

/-- The set of four-digit numbers formed from the given digits -/
def valid_numbers : Finset (Fin 10000) :=
  sorry

/-- Theorem stating the number of valid four-digit numbers -/
theorem count_valid_numbers : Finset.card valid_numbers = 180 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2535_253511


namespace NUMINAMATH_CALUDE_inequality_range_l2535_253520

theorem inequality_range (x y k : ℝ) : 
  x > 0 → y > 0 → x + y = k → 
  (∀ x y, x > 0 → y > 0 → x + y = k → (x + 1/x) * (y + 1/y) ≥ (k/2 + 2/k)^2) ↔ 
  (k > 0 ∧ k ≤ 2 * Real.sqrt (2 + Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2535_253520


namespace NUMINAMATH_CALUDE_min_students_both_l2535_253510

-- Define the classroom
structure Classroom where
  total : ℕ
  glasses : ℕ
  blue_shirts : ℕ
  both : ℕ

-- Define the conditions
def valid_classroom (c : Classroom) : Prop :=
  c.glasses = (3 * c.total) / 7 ∧
  c.blue_shirts = (4 * c.total) / 9 ∧
  c.both ≤ min c.glasses c.blue_shirts ∧
  c.total ≥ c.glasses + c.blue_shirts - c.both

-- Theorem statement
theorem min_students_both (c : Classroom) (h : valid_classroom c) :
  ∃ (c_min : Classroom), valid_classroom c_min ∧ c_min.both = 8 ∧
  ∀ (c' : Classroom), valid_classroom c' → c'.both ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_students_both_l2535_253510


namespace NUMINAMATH_CALUDE_quadratic_discriminant_with_specific_roots_l2535_253537

/-- The discriminant of a quadratic polynomial with specific root conditions -/
theorem quadratic_discriminant_with_specific_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = x - 2) ∧
  (∃! x, a * x^2 + b * x + c = 1 - x / 2) →
  b^2 - 4 * a * c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_with_specific_roots_l2535_253537


namespace NUMINAMATH_CALUDE_pythagorean_triple_l2535_253534

theorem pythagorean_triple (n : ℕ) (h1 : n ≥ 3) (h2 : Odd n) :
  n^2 + ((n^2 - 1) / 2)^2 = ((n^2 + 1) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_l2535_253534


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_digits_l2535_253593

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 3 ∨ d = 6

def contains_both (n : ℕ) : Prop :=
  3 ∈ n.digits 10 ∧ 6 ∈ n.digits 10

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem smallest_valid_number_last_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 3 = 0 ∧
    m % 6 = 0 ∧
    is_valid_number m ∧
    contains_both m ∧
    (∀ k : ℕ, k > 0 ∧ k % 3 = 0 ∧ k % 6 = 0 ∧ is_valid_number k ∧ contains_both k → m ≤ k) ∧
    last_four_digits m = 3630 :=
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_digits_l2535_253593


namespace NUMINAMATH_CALUDE_optimal_rectangular_enclosure_area_l2535_253518

theorem optimal_rectangular_enclosure_area
  (perimeter : ℝ)
  (min_length : ℝ)
  (min_width : ℝ)
  (h_perimeter : perimeter = 400)
  (h_min_length : min_length = 100)
  (h_min_width : min_width = 50) :
  ∃ (length width : ℝ),
    length ≥ min_length ∧
    width ≥ min_width ∧
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℝ),
      l ≥ min_length →
      w ≥ min_width →
      2 * (l + w) = perimeter →
      l * w ≤ length * width ∧
      length * width = 10000 :=
by sorry

end NUMINAMATH_CALUDE_optimal_rectangular_enclosure_area_l2535_253518


namespace NUMINAMATH_CALUDE_inequality_proof_l2535_253519

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  1/(1-a) + 1/(1-b) + 1/(1-c) ≥ 2/(1+a) + 2/(1+b) + 2/(1+c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2535_253519


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2535_253501

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => geometric_sequence a q n * q

theorem geometric_sequence_sum (a q : ℝ) (h1 : geometric_sequence a q 0 + geometric_sequence a q 1 = 20) 
  (h2 : geometric_sequence a q 2 + geometric_sequence a q 3 = 60) : 
  geometric_sequence a q 4 + geometric_sequence a q 5 = 180 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2535_253501


namespace NUMINAMATH_CALUDE_final_pens_count_l2535_253521

def calculate_final_pens (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (alex_takes_percent : ℕ) (sharon_gets : ℕ) : ℕ :=
  let after_mike := initial + mike_gives
  let after_cindy := after_mike * cindy_multiplier
  let alex_takes := (after_cindy * alex_takes_percent + 99) / 100  -- Rounding up
  let after_alex := after_cindy - alex_takes
  after_alex - sharon_gets

theorem final_pens_count :
  calculate_final_pens 20 22 2 15 19 = 52 := by
  sorry

end NUMINAMATH_CALUDE_final_pens_count_l2535_253521


namespace NUMINAMATH_CALUDE_football_players_count_l2535_253536

/-- Calculates the number of students playing football given the total number of students,
    the number of students playing cricket, the number of students playing neither sport,
    and the number of students playing both sports. -/
def students_playing_football (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ) : ℕ :=
  total - neither - cricket + both

/-- Theorem stating that the number of students playing football is 325 -/
theorem football_players_count :
  students_playing_football 450 175 50 100 = 325 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l2535_253536


namespace NUMINAMATH_CALUDE_angle_sum_is_420_l2535_253569

/-- A geometric configuration with six angles A, B, C, D, E, and F -/
structure GeometricConfiguration where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

/-- The theorem stating that if E = 30°, then the sum of all angles is 420° -/
theorem angle_sum_is_420 (config : GeometricConfiguration) 
  (h_E : config.E = 30) : 
  config.A + config.B + config.C + config.D + config.E + config.F = 420 := by
  sorry

#check angle_sum_is_420

end NUMINAMATH_CALUDE_angle_sum_is_420_l2535_253569


namespace NUMINAMATH_CALUDE_expression_simplification_l2535_253571

theorem expression_simplification (x : ℝ) : (x + 1)^2 + 2*(x + 1)*(5 - x) + (5 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2535_253571


namespace NUMINAMATH_CALUDE_f_of_2_eq_0_l2535_253553

def f (x : ℝ) : ℝ := (x - 1)^2 - (x - 1)

theorem f_of_2_eq_0 : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_0_l2535_253553


namespace NUMINAMATH_CALUDE_quadratic_sum_l2535_253595

/-- A quadratic function passing through (1,0) and (-5,0) with minimum value 25 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≥ 25) ∧
  QuadraticFunction a b c 1 = 0 ∧
  QuadraticFunction a b c (-5) = 0 →
  a + b + c = 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2535_253595


namespace NUMINAMATH_CALUDE_vector_equation_l2535_253559

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (A B C D : V)

theorem vector_equation : A - D + D - C - (A - B) = B - C := by sorry

end NUMINAMATH_CALUDE_vector_equation_l2535_253559


namespace NUMINAMATH_CALUDE_floor_with_57_diagonal_tiles_has_841_total_tiles_l2535_253590

/-- Represents a rectangular floor covered with square tiles -/
structure TiledFloor where
  length : ℕ
  width : ℕ
  diagonal_tiles : ℕ

/-- The total number of tiles on a rectangular floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.length * floor.width

/-- Theorem stating that a rectangular floor with 57 tiles on its diagonals has 841 tiles in total -/
theorem floor_with_57_diagonal_tiles_has_841_total_tiles :
  ∃ (floor : TiledFloor), floor.diagonal_tiles = 57 ∧ total_tiles floor = 841 := by
  sorry


end NUMINAMATH_CALUDE_floor_with_57_diagonal_tiles_has_841_total_tiles_l2535_253590


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2535_253535

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → a^2 > 2*a) ∧
  (∃ a, a^2 > 2*a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2535_253535


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l2535_253542

theorem gcd_lcm_sum : Nat.gcd 45 125 + Nat.lcm 50 15 = 155 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l2535_253542


namespace NUMINAMATH_CALUDE_stating_public_foundation_share_l2535_253527

/-- Represents the charity donation problem -/
structure CharityDonation where
  X : ℝ  -- Total amount raised in dollars
  Y : ℝ  -- Percentage donated to public foundation
  Z : ℕ+  -- Number of organizations in public foundation
  W : ℕ+  -- Number of local non-profit groups
  A : ℝ  -- Amount received by each local non-profit group in dollars
  h1 : X > 0  -- Total amount raised is positive
  h2 : 0 < Y ∧ Y < 100  -- Percentage is between 0 and 100
  h3 : W * A = X * (100 - Y) / 100  -- Equation for local non-profit groups

/-- 
Theorem stating that each organization in the public foundation 
receives YX / (100Z) dollars
-/
theorem public_foundation_share (c : CharityDonation) :
  (c.Y * c.X) / (100 * c.Z) = 
  (c.X * c.Y / 100) / c.Z :=
sorry

end NUMINAMATH_CALUDE_stating_public_foundation_share_l2535_253527


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2535_253570

theorem intersection_of_three_lines (k : ℚ) : 
  (∃ (x y : ℚ), y = 6*x + 5 ∧ y = -3*x - 30 ∧ y = 4*x + k) → 
  k = -25/9 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2535_253570


namespace NUMINAMATH_CALUDE_parabola_coefficients_l2535_253514

/-- A parabola with vertex (4, -1), vertical axis of symmetry, and passing through (0, -5) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a ≠ 0 → -b / (2 * a) = 4
  vertex_y : a * 4^2 + b * 4 + c = -1
  symmetry : a ≠ 0 → -b / (2 * a) = 4
  point : a * 0^2 + b * 0 + c = -5

theorem parabola_coefficients (p : Parabola) : p.a = -1/4 ∧ p.b = 2 ∧ p.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l2535_253514


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2535_253565

theorem complex_equation_solution :
  let z : ℂ := (1 - I)^2 + 1 + 3*I
  ∀ a b : ℝ, z^2 + a*z + b = 1 - I → a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2535_253565


namespace NUMINAMATH_CALUDE_point_on_parabola_l2535_253529

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2 - 3 * x + 1

/-- Theorem: The point (1/2, 0) lies on the parabola y = 2x^2 - 3x + 1 -/
theorem point_on_parabola : parabola (1/2) 0 := by sorry

end NUMINAMATH_CALUDE_point_on_parabola_l2535_253529


namespace NUMINAMATH_CALUDE_weight_of_three_liters_l2535_253581

/-- Given that 1 liter weighs 2.2 pounds, prove that 3 liters weigh 6.6 pounds. -/
theorem weight_of_three_liters (weight_per_liter : ℝ) (h : weight_per_liter = 2.2) :
  3 * weight_per_liter = 6.6 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_three_liters_l2535_253581


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2535_253552

theorem jelly_bean_probability (p_red p_orange p_green : ℝ) 
  (h_red : p_red = 0.1)
  (h_orange : p_orange = 0.4)
  (h_green : p_green = 0.2)
  (h_sum : p_red + p_orange + p_green + p_yellow = 1)
  (h_nonneg : p_yellow ≥ 0) :
  p_yellow = 0.3 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2535_253552


namespace NUMINAMATH_CALUDE_exam_score_problem_l2535_253548

theorem exam_score_problem (total_questions : ℕ) 
  (correct_score wrong_score total_score : ℤ) : 
  total_questions = 80 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 130 →
  ∃ (correct_answers wrong_answers : ℕ),
    correct_answers + wrong_answers = total_questions ∧
    correct_score * correct_answers + wrong_score * wrong_answers = total_score ∧
    correct_answers = 42 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2535_253548


namespace NUMINAMATH_CALUDE_irreducible_fractions_count_l2535_253566

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem irreducible_fractions_count : 
  (∃! (count : ℕ), ∃ (S : Finset ℕ), 
    S.card = count ∧ 
    (∀ n ∈ S, (1 : ℚ) / 16 < (n : ℚ) / 15 ∧ (n : ℚ) / 15 < (1 : ℚ) / 15) ∧
    (∀ n ∈ S, is_coprime n 15) ∧
    (∀ n : ℕ, (1 : ℚ) / 16 < (n : ℚ) / 15 ∧ (n : ℚ) / 15 < (1 : ℚ) / 15 → 
      is_coprime n 15 → n ∈ S)) ∧
  count = 8 :=
sorry

end NUMINAMATH_CALUDE_irreducible_fractions_count_l2535_253566


namespace NUMINAMATH_CALUDE_remaining_speed_calculation_l2535_253500

/-- Given a trip with the following characteristics:
  * Total distance of 80 miles
  * First 30 miles traveled at 30 mph
  * Average speed for the entire trip is 40 mph
  Prove that the speed for the remaining part of the trip is 50 mph -/
theorem remaining_speed_calculation (total_distance : ℝ) (first_part_distance : ℝ) 
  (first_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 80 ∧ 
  first_part_distance = 30 ∧ 
  first_part_speed = 30 ∧ 
  average_speed = 40 →
  (total_distance - first_part_distance) / 
    (total_distance / average_speed - first_part_distance / first_part_speed) = 50 :=
by sorry

end NUMINAMATH_CALUDE_remaining_speed_calculation_l2535_253500


namespace NUMINAMATH_CALUDE_value_of_y_l2535_253523

theorem value_of_y (y : ℝ) (h : 2/3 - 1/4 = 4/y) : y = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2535_253523


namespace NUMINAMATH_CALUDE_point_on_axis_l2535_253556

-- Define a point P in 2D space
def P (m : ℝ) : ℝ × ℝ := (m, 2 - m)

-- Define what it means for a point to lie on the coordinate axis
def lies_on_coordinate_axis (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∨ p.2 = 0

-- Theorem statement
theorem point_on_axis (m : ℝ) : 
  lies_on_coordinate_axis (P m) → m = 0 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_axis_l2535_253556


namespace NUMINAMATH_CALUDE_no_hexagon_with_special_point_l2535_253507

-- Define a hexagon as a set of 6 points in 2D space
def Hexagon := Fin 6 → ℝ × ℝ

-- Define convexity for a hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define a point being inside a hexagon
def is_inside (p : ℝ × ℝ) (h : Hexagon) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem no_hexagon_with_special_point :
  ¬ ∃ (h : Hexagon) (m : ℝ × ℝ),
    is_convex h ∧
    is_inside m h ∧
    (∀ i j : Fin 6, i ≠ j → distance (h i) (h j) > 1) ∧
    (∀ i : Fin 6, distance m (h i) < 1) :=
by sorry

end NUMINAMATH_CALUDE_no_hexagon_with_special_point_l2535_253507


namespace NUMINAMATH_CALUDE_invitations_per_pack_is_four_l2535_253557

/-- A structure representing the invitation problem --/
structure InvitationProblem where
  total_invitations : ℕ
  num_packs : ℕ
  invitations_per_pack : ℕ
  h1 : total_invitations = num_packs * invitations_per_pack

/-- Theorem stating that given the conditions of the problem, the number of invitations per pack is 4 --/
theorem invitations_per_pack_is_four (problem : InvitationProblem)
  (h2 : problem.total_invitations = 12)
  (h3 : problem.num_packs = 3) :
  problem.invitations_per_pack = 4 := by
  sorry

end NUMINAMATH_CALUDE_invitations_per_pack_is_four_l2535_253557


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2535_253558

theorem tagged_fish_in_second_catch 
  (initial_tagged : ℕ) 
  (second_catch : ℕ) 
  (total_fish : ℕ) 
  (h1 : initial_tagged = 40)
  (h2 : second_catch = 40)
  (h3 : total_fish = 800) :
  ∃ (tagged_in_second : ℕ), 
    tagged_in_second = 2 ∧ 
    (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish :=
by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2535_253558


namespace NUMINAMATH_CALUDE_store_pricing_l2535_253576

/-- Represents the price structure of a store selling chairs, tables, and shelves. -/
structure StorePrice where
  chair : ℝ
  table : ℝ
  shelf : ℝ

/-- Defines the properties of the store's pricing and discount policy. -/
def ValidStorePrice (p : StorePrice) : Prop :=
  p.chair + p.table = 72 ∧
  2 * p.chair + p.table = 0.6 * (p.chair + 2 * p.table) ∧
  p.chair + p.table + p.shelf = 95

/-- Calculates the discounted price for a combination of items. -/
def DiscountedPrice (p : StorePrice) : ℝ :=
  0.9 * (p.chair + 2 * p.table + p.shelf)

/-- Theorem stating the correct prices for the store items and the discounted combination. -/
theorem store_pricing (p : StorePrice) (h : ValidStorePrice p) :
  p.table = 63 ∧ DiscountedPrice p = 142.2 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_l2535_253576


namespace NUMINAMATH_CALUDE_common_roots_product_l2535_253532

theorem common_roots_product (C D E : ℝ) : 
  ∃ (u v w t : ℂ), 
    (u^3 + C*u^2 + D*u + 20 = 0) ∧ 
    (v^3 + C*v^2 + D*v + 20 = 0) ∧ 
    (w^3 + C*w^2 + D*w + 20 = 0) ∧
    (u^3 + E*u^2 + 70 = 0) ∧ 
    (v^3 + E*v^2 + 70 = 0) ∧ 
    (t^3 + E*t^2 + 70 = 0) ∧
    (u ≠ v) ∧ (u ≠ w) ∧ (v ≠ w) ∧
    (u ≠ t) ∧ (v ≠ t) →
    u * v = 2 * Real.rpow 175 (1/3) :=
sorry

end NUMINAMATH_CALUDE_common_roots_product_l2535_253532


namespace NUMINAMATH_CALUDE_james_writing_pages_l2535_253579

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  hours_per_week : ℕ
  people_per_day : ℕ

/-- Calculate pages written daily per person -/
def pages_per_person_daily (scenario : WritingScenario) : ℚ :=
  (scenario.pages_per_hour * scenario.hours_per_week : ℚ) / (7 * scenario.people_per_day)

/-- Theorem: James writes 5 pages daily to each person -/
theorem james_writing_pages (james : WritingScenario) 
  (h1 : james.pages_per_hour = 10)
  (h2 : james.hours_per_week = 7)
  (h3 : james.people_per_day = 2) :
  pages_per_person_daily james = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_pages_l2535_253579


namespace NUMINAMATH_CALUDE_power_relationship_l2535_253584

theorem power_relationship (a b c : ℝ) 
  (ha : a = Real.rpow 0.8 5.2)
  (hb : b = Real.rpow 0.8 5.5)
  (hc : c = Real.rpow 5.2 0.1) : 
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_power_relationship_l2535_253584


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l2535_253551

theorem sum_of_specific_numbers : 12534 + 25341 + 53412 + 34125 = 125412 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l2535_253551


namespace NUMINAMATH_CALUDE_elections_with_past_officers_count_l2535_253545

def total_candidates : ℕ := 16
def past_officers : ℕ := 7
def positions : ℕ := 5

def elections_with_past_officers : ℕ := Nat.choose total_candidates positions - Nat.choose (total_candidates - past_officers) positions

theorem elections_with_past_officers_count : elections_with_past_officers = 4242 := by
  sorry

end NUMINAMATH_CALUDE_elections_with_past_officers_count_l2535_253545


namespace NUMINAMATH_CALUDE_ammonia_formation_l2535_253567

-- Define the chemical reaction
structure Reaction where
  koh : ℝ  -- moles of Potassium hydroxide
  nh4i : ℝ  -- moles of Ammonium iodide
  nh3 : ℝ  -- moles of Ammonia formed

-- Define the balanced equation
axiom balanced_equation (r : Reaction) : r.koh = r.nh4i ∧ r.koh = r.nh3

-- Theorem: Given 3 moles of KOH, the number of moles of NH3 formed is 3
theorem ammonia_formation (r : Reaction) (h : r.koh = 3) : r.nh3 = 3 := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ammonia_formation_l2535_253567


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_distinct_l2535_253538

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 20) ∧
  (n / 1000 ≠ n / 100 % 10) ∧ (n / 1000 ≠ n / 10 % 10) ∧ (n / 1000 ≠ n % 10) ∧
  (n / 100 % 10 ≠ n / 10 % 10) ∧ (n / 100 % 10 ≠ n % 10) ∧
  (n / 10 % 10 ≠ n % 10)

theorem largest_four_digit_sum_20_distinct : 
  ∀ n : ℕ, is_valid_number n → n ≤ 9821 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_distinct_l2535_253538


namespace NUMINAMATH_CALUDE_valid_pairs_l2535_253533

def is_valid_pair (m n : ℕ+) : Prop :=
  (m^2 - n) ∣ (m + n^2) ∧ (n^2 - m) ∣ (n + m^2)

theorem valid_pairs :
  ∀ m n : ℕ+, is_valid_pair m n ↔ 
    ((m = 2 ∧ n = 2) ∨ 
     (m = 3 ∧ n = 3) ∨ 
     (m = 1 ∧ n = 2) ∨ 
     (m = 2 ∧ n = 1) ∨ 
     (m = 3 ∧ n = 2) ∨ 
     (m = 2 ∧ n = 3)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2535_253533


namespace NUMINAMATH_CALUDE_symmetry_sum_l2535_253504

/-- Two points are symmetric with respect to the origin if the sum of their coordinates is (0, 0) -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

theorem symmetry_sum (a b : ℝ) :
  symmetric_wrt_origin (-2022, -1) (a, b) → a + b = 2023 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l2535_253504


namespace NUMINAMATH_CALUDE_dark_tiles_fraction_is_three_fourths_l2535_253586

/-- Represents a square tiling pattern -/
structure TilingPattern where
  size : ℕ
  dark_tiles_per_corner : ℕ
  is_symmetrical : Bool

/-- Calculates the fraction of dark tiles in a tiling pattern -/
def fraction_of_dark_tiles (pattern : TilingPattern) : ℚ :=
  if pattern.is_symmetrical
  then (4 * pattern.dark_tiles_per_corner : ℚ) / (pattern.size * pattern.size : ℚ)
  else 0

/-- Theorem stating that a 4x4 symmetrical pattern with 3 dark tiles per corner 
    has 3/4 of its tiles dark -/
theorem dark_tiles_fraction_is_three_fourths 
  (pattern : TilingPattern) 
  (h1 : pattern.size = 4) 
  (h2 : pattern.dark_tiles_per_corner = 3) 
  (h3 : pattern.is_symmetrical = true) : 
  fraction_of_dark_tiles pattern = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_dark_tiles_fraction_is_three_fourths_l2535_253586


namespace NUMINAMATH_CALUDE_rod_solution_l2535_253587

/-- Represents a rod divided into parts -/
structure Rod where
  m : ℕ  -- number of red parts
  n : ℕ  -- number of black parts
  x : ℕ  -- number of coinciding lines
  total_segments : ℕ  -- total number of segments after cutting
  longest_segments : ℕ  -- number of longest segments

/-- Conditions for the rod problem -/
def rod_conditions (r : Rod) : Prop :=
  r.m > r.n ∧
  r.total_segments = 170 ∧
  r.longest_segments = 100

/-- Theorem stating the solution to the rod problem -/
theorem rod_solution (r : Rod) (h : rod_conditions r) : r.m = 13 ∧ r.n = 156 :=
sorry

/-- Lemma stating that x + 1 is a common divisor of m and n -/
lemma common_divisor (r : Rod) : (r.x + 1) ∣ r.m ∧ (r.x + 1) ∣ r.n :=
sorry

end NUMINAMATH_CALUDE_rod_solution_l2535_253587


namespace NUMINAMATH_CALUDE_man_birth_year_l2535_253544

theorem man_birth_year (x : ℕ) (h1 : x > 0) (h2 : x^2 - 10 - x > 1850) 
  (h3 : x^2 - 10 - x < 1900) : x^2 - 10 - x = 1882 := by
  sorry

end NUMINAMATH_CALUDE_man_birth_year_l2535_253544


namespace NUMINAMATH_CALUDE_new_average_calculation_l2535_253541

theorem new_average_calculation (num_students : ℕ) (original_avg : ℝ) 
  (increase_percent : ℝ) (bonus : ℝ) (new_avg : ℝ) : 
  num_students = 37 → 
  original_avg = 73 → 
  increase_percent = 65 → 
  bonus = 15 → 
  new_avg = original_avg * (1 + increase_percent / 100) + bonus →
  new_avg = 135.45 := by
sorry

end NUMINAMATH_CALUDE_new_average_calculation_l2535_253541


namespace NUMINAMATH_CALUDE_polygon_30_sides_diagonals_l2535_253517

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_polygon_30_sides_diagonals_l2535_253517


namespace NUMINAMATH_CALUDE_intersection_probability_l2535_253594

def U : Finset Nat := {1, 2, 3, 4, 5}
def I : Finset (Finset Nat) := Finset.powerset U

def favorable_pairs : Nat :=
  (Finset.powerset U).card * (Finset.filter (fun s => s.card = 3) (Finset.powerset U)).card

def total_pairs : Nat := (I.card.choose 2)

theorem intersection_probability :
  (favorable_pairs : ℚ) / total_pairs = 5 / 62 := by sorry

end NUMINAMATH_CALUDE_intersection_probability_l2535_253594


namespace NUMINAMATH_CALUDE_cubic_equation_solvable_l2535_253583

theorem cubic_equation_solvable (a b c d : ℝ) (ha : a ≠ 0) :
  ∃ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ∧
  (∃ (e f g h i j k l m n : ℝ),
    x = (e * (f^(1/3)) + g * (h^(1/3)) + i * (j^(1/3)) + k) /
        (l * (m^(1/2)) + n)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solvable_l2535_253583


namespace NUMINAMATH_CALUDE_both_readers_count_l2535_253549

def total_workers : ℕ := 72

def saramago_readers : ℕ := total_workers / 4
def kureishi_readers : ℕ := total_workers * 5 / 8

def both_readers : ℕ := 8

theorem both_readers_count :
  saramago_readers + kureishi_readers - both_readers + 
  (saramago_readers - both_readers - 1) = total_workers :=
by sorry

end NUMINAMATH_CALUDE_both_readers_count_l2535_253549


namespace NUMINAMATH_CALUDE_students_in_school_l2535_253578

theorem students_in_school (total_students : ℕ) (trip_fraction : ℚ) (home_fraction : ℚ) : 
  total_students = 1000 →
  trip_fraction = 1/2 →
  home_fraction = 1/2 →
  (total_students - (trip_fraction * total_students).floor - 
   (home_fraction * (total_students - (trip_fraction * total_students).floor)).floor) = 250 := by
  sorry

#check students_in_school

end NUMINAMATH_CALUDE_students_in_school_l2535_253578


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l2535_253508

/-- Given a hyperbola defined by the equation x²/16 - y²/25 = 1, 
    the slopes of its asymptotes are ±5/4. -/
theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), x^2/16 - y^2/25 = 1 →
  ∃ (m : ℝ), m = 5/4 ∧ (y = m*x ∨ y = -m*x) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l2535_253508


namespace NUMINAMATH_CALUDE_ellipse_locus_l2535_253516

/-- Given two fixed points F₁ and F₂ on the x-axis, and a point M such that
    the sum of its distances to F₁ and F₂ is constant, prove that the locus of M
    is an ellipse with F₁ and F₂ as foci. -/
theorem ellipse_locus (F₁ F₂ M : ℝ × ℝ) (d : ℝ) :
  F₁ = (-4, 0) →
  F₂ = (4, 0) →
  d = 10 →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) +
    Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = d →
  M.1^2 / 25 + M.2^2 / 9 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_locus_l2535_253516


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_210_l2535_253515

theorem greatest_prime_factor_of_210 : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ 210 ∧ ∀ (q : ℕ), q.Prime → q ∣ 210 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_210_l2535_253515


namespace NUMINAMATH_CALUDE_freshman_class_size_l2535_253531

theorem freshman_class_size :
  ∃! n : ℕ,
    n < 600 ∧
    n % 17 = 16 ∧
    n % 19 = 18 ∧
    n = 322 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l2535_253531


namespace NUMINAMATH_CALUDE_x_over_z_equals_five_l2535_253503

theorem x_over_z_equals_five (x y z : ℚ) 
  (eq1 : x + y = 2 * x + z)
  (eq2 : x - 2 * y = 4 * z)
  (eq3 : x + y + z = 21)
  (eq4 : y / z = 6) :
  x / z = 5 := by sorry

end NUMINAMATH_CALUDE_x_over_z_equals_five_l2535_253503


namespace NUMINAMATH_CALUDE_track_circumference_is_720_l2535_253530

/-- Represents the circumference of a circular track given specific meeting conditions of two travelers -/
def track_circumference (first_meeting_distance : ℝ) (second_meeting_remaining : ℝ) : ℝ :=
  let half_circumference := 360
  2 * half_circumference

/-- Theorem stating that under the given conditions, the track circumference is 720 yards -/
theorem track_circumference_is_720 :
  track_circumference 150 90 = 720 :=
by
  -- The proof would go here
  sorry

#eval track_circumference 150 90

end NUMINAMATH_CALUDE_track_circumference_is_720_l2535_253530


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l2535_253577

theorem fraction_equality_solution (x : ℚ) :
  (x + 11) / (x - 4) = (x - 1) / (x + 6) → x = -31/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l2535_253577


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2535_253524

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ), 
  (15 * x^2 + 90 * x + 405 = a * (x + b)^2 + c) ∧ (a + b + c = 288) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2535_253524


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l2535_253564

-- Define the days of the week
inductive Day : Type
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day
  | Sunday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Define a function to add days
def addDays (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (addDays d m)

-- Theorem statement
theorem tomorrow_is_saturday (dayBeforeYesterday : Day) :
  addDays dayBeforeYesterday 5 = Day.Monday →
  addDays dayBeforeYesterday 7 = Day.Saturday :=
by
  sorry


end NUMINAMATH_CALUDE_tomorrow_is_saturday_l2535_253564


namespace NUMINAMATH_CALUDE_complex_number_subtraction_l2535_253582

theorem complex_number_subtraction (z : ℂ) (a b : ℝ) : 
  z = 2 + I → Complex.re z = a → Complex.im z = b → a - b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_subtraction_l2535_253582


namespace NUMINAMATH_CALUDE_sugar_calculation_l2535_253528

/-- Given a recipe with a sugar to flour ratio and an amount of flour,
    calculate the amount of sugar needed. -/
def sugar_amount (sugar_flour_ratio : ℚ) (flour_amount : ℚ) : ℚ :=
  sugar_flour_ratio * flour_amount

theorem sugar_calculation (sugar_flour_ratio flour_amount : ℚ) :
  sugar_flour_ratio = 10 / 1 →
  flour_amount = 5 →
  sugar_amount sugar_flour_ratio flour_amount = 50 := by
sorry

end NUMINAMATH_CALUDE_sugar_calculation_l2535_253528


namespace NUMINAMATH_CALUDE_xy_equality_l2535_253525

theorem xy_equality (x y : ℝ) : 4 * x * y - 3 * x * y = x * y := by
  sorry

end NUMINAMATH_CALUDE_xy_equality_l2535_253525


namespace NUMINAMATH_CALUDE_cubic_root_transformation_l2535_253562

theorem cubic_root_transformation (p : ℝ) : 
  p^3 + p - 3 = 0 → (p^2)^3 + 2*(p^2)^2 + p^2 - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_transformation_l2535_253562


namespace NUMINAMATH_CALUDE_equality_from_fraction_l2535_253568

theorem equality_from_fraction (a b : ℝ) (n : ℕ+) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_eq : ((a + b)^(n : ℕ) - (a - b)^(n : ℕ)) / ((a + b)^(n : ℕ) + (a - b)^(n : ℕ)) = a / b) :
  a = b := by sorry

end NUMINAMATH_CALUDE_equality_from_fraction_l2535_253568


namespace NUMINAMATH_CALUDE_find_x_l2535_253547

-- Define the variables
variable (a b x : ℝ)
variable (r : ℝ)

-- State the theorem
theorem find_x (h1 : b ≠ 0) (h2 : r = (3 * a) ^ (2 * b)) (h3 : r = a ^ b * x ^ (2 * b)) : x = 3 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2535_253547


namespace NUMINAMATH_CALUDE_brothers_selection_probability_l2535_253505

theorem brothers_selection_probability (p_x p_y p_both : ℚ) :
  p_x = 1/3 → p_y = 2/5 → p_both = p_x * p_y → p_both = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_brothers_selection_probability_l2535_253505


namespace NUMINAMATH_CALUDE_triangle_side_length_l2535_253563

/-- In a triangle ABC, given angle A, angle B, and side AC, prove the length of AB --/
theorem triangle_side_length (A B C : Real) (angleA angleB : Real) (sideAC : Real) :
  -- Conditions
  angleA = 105 * Real.pi / 180 →
  angleB = 45 * Real.pi / 180 →
  sideAC = 2 →
  -- Triangle angle sum property
  angleA + angleB + C = Real.pi →
  -- Sine rule
  sideAC / Real.sin angleB = A / Real.sin C →
  -- Conclusion
  A = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2535_253563
