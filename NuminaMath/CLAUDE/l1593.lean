import Mathlib

namespace NUMINAMATH_CALUDE_ozone_effect_significant_l1593_159393

/-- Represents the data from the experiment -/
structure ExperimentData where
  control_group : List Float
  experimental_group : List Float

/-- Calculates the median of a sorted list -/
def median (sorted_list : List Float) : Float :=
  sorry

/-- Counts the number of elements less than a given value -/
def count_less_than (list : List Float) (value : Float) : Nat :=
  sorry

/-- Calculates K² statistic -/
def calculate_k_squared (a b c d : Nat) : Float :=
  sorry

/-- Main theorem: K² value is greater than the critical value for 95% confidence -/
theorem ozone_effect_significant (data : ExperimentData) :
  let all_data := data.control_group ++ data.experimental_group
  let m := median all_data
  let a := count_less_than data.control_group m
  let b := data.control_group.length - a
  let c := count_less_than data.experimental_group m
  let d := data.experimental_group.length - c
  let k_squared := calculate_k_squared a b c d
  k_squared > 3.841 := by
  sorry

#check ozone_effect_significant

end NUMINAMATH_CALUDE_ozone_effect_significant_l1593_159393


namespace NUMINAMATH_CALUDE_other_diagonal_length_l1593_159366

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with one diagonal of 17 cm and area of 170 cm², 
    the other diagonal is 20 cm -/
theorem other_diagonal_length :
  ∀ (r : Rhombus), r.d1 = 17 ∧ r.area = 170 → r.d2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l1593_159366


namespace NUMINAMATH_CALUDE_product_def_l1593_159309

theorem product_def (a b c d e f : ℝ) : 
  a * b * c = 130 →
  b * c * d = 65 →
  c * d * e = 500 →
  (a * f) / (c * d) = 1 →
  d * e * f = 250 := by
sorry

end NUMINAMATH_CALUDE_product_def_l1593_159309


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_6370_l1593_159367

theorem largest_prime_factor_of_6370 : (Nat.factors 6370).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_6370_l1593_159367


namespace NUMINAMATH_CALUDE_sin_arccos_three_fifths_l1593_159381

theorem sin_arccos_three_fifths : Real.sin (Real.arccos (3/5)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_three_fifths_l1593_159381


namespace NUMINAMATH_CALUDE_license_plate_count_l1593_159380

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 5

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 2

/-- The number of odd single-digit numbers -/
def odd_digits : ℕ := 5

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of license plate combinations -/
def license_plate_combinations : ℕ :=
  (choose alphabet_size 2) * (choose letter_positions 2) * (choose (letter_positions - 2) 2) * 
  (alphabet_size - 2) * odd_digits * (odd_digits - 1)

theorem license_plate_count : license_plate_combinations = 936000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1593_159380


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1593_159317

theorem trigonometric_identity : 
  3.4173 * Real.sin (2 * Real.pi / 17) + Real.sin (4 * Real.pi / 17) - 
  Real.sin (6 * Real.pi / 17) - 0.5 * Real.sin (8 * Real.pi / 17) = 
  8 * Real.sin (2 * Real.pi / 17) ^ 3 * Real.cos (Real.pi / 17) ^ 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1593_159317


namespace NUMINAMATH_CALUDE_gcd_102_238_l1593_159385

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l1593_159385


namespace NUMINAMATH_CALUDE_percentage_relation_l1593_159386

theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.05 * C) 
  (h2 : A = 0.20 * B) : 
  B = 0.25 * C := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l1593_159386


namespace NUMINAMATH_CALUDE_income_comparison_l1593_159331

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.6 * juan) 
  (h2 : mary = 0.84 * juan) : 
  (mary - tim) / tim * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l1593_159331


namespace NUMINAMATH_CALUDE_total_candies_l1593_159371

theorem total_candies (chocolate_boxes caramel_boxes pieces_per_box : ℕ) :
  chocolate_boxes = 6 →
  caramel_boxes = 4 →
  pieces_per_box = 9 →
  chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = 90 :=
by
  sorry

#check total_candies

end NUMINAMATH_CALUDE_total_candies_l1593_159371


namespace NUMINAMATH_CALUDE_xy_values_l1593_159340

theorem xy_values (x y : ℝ) 
  (eq1 : x / (x^2 * y^2 - 1) - 1 / x = 4)
  (eq2 : (x^2 * y) / (x^2 * y^2 - 1) + y = 2) :
  x * y = 1 / Real.sqrt 2 ∨ x * y = -(1 / Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_xy_values_l1593_159340


namespace NUMINAMATH_CALUDE_cosine_from_tangent_third_quadrant_l1593_159373

theorem cosine_from_tangent_third_quadrant (α : Real) :
  α ∈ Set.Ioo (π) (3*π/2) →  -- α is in the third quadrant
  Real.tan α = 1/2 →         -- tan(α) = 1/2
  Real.cos α = -2*Real.sqrt 5/5 := by
sorry

end NUMINAMATH_CALUDE_cosine_from_tangent_third_quadrant_l1593_159373


namespace NUMINAMATH_CALUDE_adams_forgotten_lawns_l1593_159388

/-- Adam's lawn mowing problem -/
theorem adams_forgotten_lawns (dollars_per_lawn : ℕ) (total_lawns : ℕ) (actual_earnings : ℕ) :
  dollars_per_lawn = 9 →
  total_lawns = 12 →
  actual_earnings = 36 →
  total_lawns - (actual_earnings / dollars_per_lawn) = 8 := by
  sorry

end NUMINAMATH_CALUDE_adams_forgotten_lawns_l1593_159388


namespace NUMINAMATH_CALUDE_least_difference_l1593_159348

theorem least_difference (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  y - x > 3 ∧
  Even x ∧ Odd y ∧ Odd z →
  ∀ w, w = z - x → w ≥ 7 := by
sorry

end NUMINAMATH_CALUDE_least_difference_l1593_159348


namespace NUMINAMATH_CALUDE_peanuts_added_l1593_159306

theorem peanuts_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 4 → final = 6 → final = initial + added → added = 2 := by
sorry

end NUMINAMATH_CALUDE_peanuts_added_l1593_159306


namespace NUMINAMATH_CALUDE_x_cubed_term_is_seventh_l1593_159304

/-- The exponent of the binomial expansion -/
def n : ℕ := 16

/-- The general term of the expansion -/
def T (r : ℕ) : ℚ → ℚ := λ x => 2^r * Nat.choose n r * x^(8 - 5/6 * r)

/-- The index of the term containing x^3 -/
def r : ℕ := 6

theorem x_cubed_term_is_seventh :
  T r = T 6 ∧ 8 - 5/6 * r = 3 ∧ r + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_term_is_seventh_l1593_159304


namespace NUMINAMATH_CALUDE_puppy_sleep_ratio_l1593_159375

theorem puppy_sleep_ratio (connor_sleep : ℕ) (luke_extra_sleep : ℕ) (puppy_sleep : ℕ) : 
  connor_sleep = 6 →
  luke_extra_sleep = 2 →
  puppy_sleep = 16 →
  (puppy_sleep : ℚ) / (connor_sleep + luke_extra_sleep) = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppy_sleep_ratio_l1593_159375


namespace NUMINAMATH_CALUDE_prime_power_gcd_condition_l1593_159312

theorem prime_power_gcd_condition (n : ℕ) (h : n > 1) :
  (∀ m : ℕ, 1 ≤ m ∧ m < n →
    Nat.gcd n ((n - m) / Nat.gcd n m) = 1) ↔
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ k > 0 ∧ n = p^k :=
by sorry

end NUMINAMATH_CALUDE_prime_power_gcd_condition_l1593_159312


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1593_159345

/-- The polynomial function f(x) = x^8 + 5x^7 + 10x^6 + 1728x^5 - 1380x^4 -/
def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 1728*x^5 - 1380*x^4

/-- Theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1593_159345


namespace NUMINAMATH_CALUDE_cube_of_product_l1593_159308

theorem cube_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l1593_159308


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1593_159346

theorem fraction_subtraction_simplification :
  (9 : ℚ) / 19 - 3 / 57 - 1 / 3 = 5 / 57 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1593_159346


namespace NUMINAMATH_CALUDE_sin_decreasing_interval_l1593_159315

/-- The monotonic decreasing interval of sin(π/3 - 2x) -/
theorem sin_decreasing_interval (k : ℤ) :
  let f : ℝ → ℝ := λ x => Real.sin (π/3 - 2*x)
  ∀ x ∈ Set.Icc (k*π - π/12) (k*π + 5*π/12),
    ∀ y ∈ Set.Icc (k*π - π/12) (k*π + 5*π/12),
      x ≤ y → f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_sin_decreasing_interval_l1593_159315


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1593_159338

/-- Given that y varies inversely as x, prove that if y = 6 when x = 3, then y = 3/2 when x = 12 -/
theorem inverse_variation_problem (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → y x = k / x) →  -- y varies inversely as x
  y 3 = 6 →                     -- y = 6 when x = 3
  y 12 = 3 / 2 :=               -- y = 3/2 when x = 12
by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l1593_159338


namespace NUMINAMATH_CALUDE_derivative_at_one_is_one_fourth_l1593_159356

open Real

-- Define the function f
noncomputable def f (f'1 : ℝ) : ℝ → ℝ := λ x => log x - 3 * f'1 * x

-- State the theorem
theorem derivative_at_one_is_one_fourth (f'1 : ℝ) :
  (∀ x > 0, f f'1 x = log x - 3 * f'1 * x) →
  deriv (f f'1) 1 = 1/4 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_is_one_fourth_l1593_159356


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1593_159344

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 175/11) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1593_159344


namespace NUMINAMATH_CALUDE_prob_heart_or_king_is_31_52_l1593_159382

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 52)
  (hearts : ℕ := 13)
  (kings : ℕ := 4)
  (king_of_hearts : ℕ := 1)

/-- The probability of drawing at least one heart or king when drawing two cards without replacement -/
def prob_heart_or_king (d : Deck) : ℚ :=
  1 - (d.total_cards - (d.hearts + d.kings - d.king_of_hearts)) * (d.total_cards - 1 - (d.hearts + d.kings - d.king_of_hearts)) /
      (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability of drawing at least one heart or king is 31/52 -/
theorem prob_heart_or_king_is_31_52 (d : Deck) :
  prob_heart_or_king d = 31 / 52 := by
  sorry

end NUMINAMATH_CALUDE_prob_heart_or_king_is_31_52_l1593_159382


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1593_159311

/-- Given that point A(2, -5) is symmetric with respect to the x-axis to point (m, n), prove that m + n = 7. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (2 = m ∧ -5 = -n) → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1593_159311


namespace NUMINAMATH_CALUDE_prime_squared_plus_two_l1593_159364

theorem prime_squared_plus_two (p : ℕ) : 
  Nat.Prime p → (Nat.Prime (p^2 + 2) ↔ p = 3) := by sorry

end NUMINAMATH_CALUDE_prime_squared_plus_two_l1593_159364


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l1593_159301

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem largest_three_digit_product :
  ∀ n x y : ℕ,
    n ≥ 100 ∧ n < 1000 →
    is_prime x ∧ is_prime y ∧ is_prime (10 * y + x) →
    x < 10 ∧ y < 10 →
    x ≠ y ∧ x ≠ (10 * y + x) ∧ y ≠ (10 * y + x) →
    n = x * y * (10 * y + x) →
    n ≤ 777 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l1593_159301


namespace NUMINAMATH_CALUDE_choose_14_3_l1593_159321

theorem choose_14_3 : Nat.choose 14 3 = 364 := by
  sorry

end NUMINAMATH_CALUDE_choose_14_3_l1593_159321


namespace NUMINAMATH_CALUDE_inequality_proof_l1593_159334

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1593_159334


namespace NUMINAMATH_CALUDE_luke_trips_l1593_159398

/-- The number of trays Luke can carry in one trip -/
def trays_per_trip : ℕ := 4

/-- The number of trays on the first table -/
def trays_table1 : ℕ := 20

/-- The number of trays on the second table -/
def trays_table2 : ℕ := 16

/-- The total number of trips Luke will make -/
def total_trips : ℕ := (trays_table1 / trays_per_trip) + (trays_table2 / trays_per_trip)

theorem luke_trips : total_trips = 9 := by
  sorry

end NUMINAMATH_CALUDE_luke_trips_l1593_159398


namespace NUMINAMATH_CALUDE_discount_doubles_with_time_ratio_two_l1593_159352

/-- Represents the true discount calculation for a bill -/
structure BillDiscount where
  face_value : ℝ
  initial_discount : ℝ
  time_ratio : ℝ

/-- Calculates the discount for a different time period based on the time ratio -/
def discount_for_different_time (bill : BillDiscount) : ℝ :=
  bill.initial_discount * bill.time_ratio

/-- Theorem stating that for a bill of 110 with initial discount of 10 and time ratio of 2,
    the discount for the different time period is 20 -/
theorem discount_doubles_with_time_ratio_two :
  let bill := BillDiscount.mk 110 10 2
  discount_for_different_time bill = 20 := by
  sorry

#eval discount_for_different_time (BillDiscount.mk 110 10 2)

end NUMINAMATH_CALUDE_discount_doubles_with_time_ratio_two_l1593_159352


namespace NUMINAMATH_CALUDE_david_trip_expenses_l1593_159310

theorem david_trip_expenses (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 1800)
  (h2 : remaining_amount = 500)
  (h3 : initial_amount > remaining_amount) :
  initial_amount - remaining_amount - remaining_amount = 800 := by
  sorry

end NUMINAMATH_CALUDE_david_trip_expenses_l1593_159310


namespace NUMINAMATH_CALUDE_martha_blue_butterflies_l1593_159395

/-- Proves that Martha has 4 blue butterflies given the conditions of the problem -/
theorem martha_blue_butterflies :
  ∀ (total blue yellow black : ℕ),
    total = 11 →
    black = 5 →
    blue = 2 * yellow →
    total = blue + yellow + black →
    blue = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_blue_butterflies_l1593_159395


namespace NUMINAMATH_CALUDE_GH_distance_is_40_l1593_159333

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of a diagonal -/
  diagonal_length : ℝ
  /-- The distance from point G to vertex A -/
  GA_distance : ℝ
  /-- The distance from point G to vertex D -/
  GD_distance : ℝ
  /-- The base angle at the longer base (AD) -/
  base_angle : ℝ
  /-- Assumption that the diagonal length is 20√5 -/
  diagonal_length_eq : diagonal_length = 20 * Real.sqrt 5
  /-- Assumption that GA distance is 20 -/
  GA_distance_eq : GA_distance = 20
  /-- Assumption that GD distance is 40 -/
  GD_distance_eq : GD_distance = 40
  /-- Assumption that the base angle is π/4 -/
  base_angle_eq : base_angle = Real.pi / 4

/-- The main theorem stating that GH distance is 40 -/
theorem GH_distance_is_40 (t : IsoscelesTrapezoid) : ℝ := by
  sorry

#check GH_distance_is_40

end NUMINAMATH_CALUDE_GH_distance_is_40_l1593_159333


namespace NUMINAMATH_CALUDE_B_interval_when_m_less_than_half_A_union_B_equals_A_iff_l1593_159377

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m+1)*x + 2*m < 0}

-- Statement 1: When m < 1/2, B = (2m, 1)
theorem B_interval_when_m_less_than_half (m : ℝ) (h : m < 1/2) :
  B m = Set.Ioo (2*m) 1 :=
sorry

-- Statement 2: A ∪ B = A if and only if -1/2 ≤ m ≤ 1
theorem A_union_B_equals_A_iff (m : ℝ) :
  A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_B_interval_when_m_less_than_half_A_union_B_equals_A_iff_l1593_159377


namespace NUMINAMATH_CALUDE_sequence_formula_l1593_159319

theorem sequence_formula (a b : ℕ → ℝ) : 
  (∀ n, a n > 0 ∧ b n > 0) →  -- Each term is positive
  (∀ n, 2 * b n = a n + a (n + 1)) →  -- Arithmetic sequence condition
  (∀ n, (a (n + 1))^2 = b n * b (n + 1)) →  -- Geometric sequence condition
  a 1 = 1 →  -- Initial condition
  a 2 = 3 →  -- Initial condition
  (∀ n, a n = (n^2 + n) / 2) :=  -- General term formula
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l1593_159319


namespace NUMINAMATH_CALUDE_triangle_angle_measurement_l1593_159396

theorem triangle_angle_measurement (D E F : ℝ) : 
  D = 85 →                  -- Measure of ∠D is 85 degrees
  E = 4 * F + 15 →          -- Measure of ∠E is 15 degrees more than four times the measure of ∠F
  D + E + F = 180 →         -- Sum of angles in a triangle is 180 degrees
  F = 16                    -- Measure of ∠F is 16 degrees
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_measurement_l1593_159396


namespace NUMINAMATH_CALUDE_girls_average_age_l1593_159376

/-- Proves that the average age of girls is 11 years given the school's statistics -/
theorem girls_average_age (total_students : ℕ) (boys_avg_age : ℚ) (school_avg_age : ℚ) (num_girls : ℕ) :
  total_students = 604 →
  boys_avg_age = 12 →
  school_avg_age = 47/4 →
  num_girls = 151 →
  (total_students * school_avg_age - (total_students - num_girls) * boys_avg_age) / num_girls = 11 := by
  sorry


end NUMINAMATH_CALUDE_girls_average_age_l1593_159376


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_l1593_159336

theorem consecutive_negative_integers_product (n : ℤ) :
  n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 2240 →
  |n - (n + 1)| = 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_l1593_159336


namespace NUMINAMATH_CALUDE_tooth_fairy_payment_l1593_159322

theorem tooth_fairy_payment (total_money : ℕ) (teeth_count : ℕ) (h1 : total_money = 54) (h2 : teeth_count = 18) :
  total_money / teeth_count = 3 := by
sorry

end NUMINAMATH_CALUDE_tooth_fairy_payment_l1593_159322


namespace NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l1593_159324

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 210) = 216 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l1593_159324


namespace NUMINAMATH_CALUDE_solve_for_y_l1593_159329

theorem solve_for_y (x y : ℚ) (h1 : x - y = 20) (h2 : 3 * (x + y) = 15) : y = -15/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1593_159329


namespace NUMINAMATH_CALUDE_curve_is_line_segment_l1593_159302

-- Define the parametric equations
def x (t : ℝ) : ℝ := 3 * t^2 + 2
def y (t : ℝ) : ℝ := t^2 - 1

-- Define the parameter range
def t_range : Set ℝ := {t | 0 ≤ t ∧ t ≤ 5}

-- Theorem statement
theorem curve_is_line_segment :
  ∃ (a b c : ℝ), ∀ t ∈ t_range, a * x t + b * y t + c = 0 ∧
  ∃ (x_min x_max : ℝ), (∀ t ∈ t_range, x_min ≤ x t ∧ x t ≤ x_max) ∧
  x_min < x_max :=
sorry

end NUMINAMATH_CALUDE_curve_is_line_segment_l1593_159302


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1593_159360

/-- The area of a rectangle inscribed in an ellipse -/
theorem inscribed_rectangle_area :
  ∀ (a b : ℝ),
  (a^2 / 4 + b^2 / 8 = 1) →  -- Rectangle vertices satisfy ellipse equation
  (2 * a = b) →             -- Length along x-axis is twice the length along y-axis
  (4 * a * b = 16 / 3) :=   -- Area of the rectangle is 16/3
by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1593_159360


namespace NUMINAMATH_CALUDE_calculation_proof_l1593_159326

theorem calculation_proof : 2325 + 300 / 75 - 425 * 2 = 1479 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1593_159326


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_decomposition_l1593_159318

/-- A cyclic quadrilateral is a quadrilateral that can be circumscribed about a circle. -/
def CyclicQuadrilateral : Type := sorry

/-- A decomposition of a quadrilateral into n smaller quadrilaterals. -/
def Decomposition (Q : CyclicQuadrilateral) (n : ℕ) : Type := sorry

/-- Predicate to check if all quadrilaterals in a decomposition are cyclic. -/
def AllCyclic (d : Decomposition Q n) : Prop := sorry

theorem cyclic_quadrilateral_decomposition (n : ℕ) (Q : CyclicQuadrilateral) 
  (h : n ≥ 4) : 
  ∃ (d : Decomposition Q n), AllCyclic d := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_decomposition_l1593_159318


namespace NUMINAMATH_CALUDE_cube_sum_simplification_l1593_159320

theorem cube_sum_simplification (a b c : ℝ) :
  (a^3 + b^3) / (c^3 + b^3) = (a + b) / (c + b) ↔ a + c = b :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_simplification_l1593_159320


namespace NUMINAMATH_CALUDE_counterexample_exists_l1593_159316

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a⁻¹ ≥ b⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1593_159316


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1593_159307

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x - 1) * (x + a) > 0}
  if a < -1 then
    S = {x : ℝ | x > -a ∨ x < 1}
  else if a = -1 then
    S = {x : ℝ | x ≠ 1}
  else
    S = {x : ℝ | x < -a ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1593_159307


namespace NUMINAMATH_CALUDE_negation_of_existence_logarithm_equality_negation_l1593_159342

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ P x) ↔ (∀ x : ℝ, x > 0 → ¬ P x) :=
by sorry

theorem logarithm_equality_negation :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_logarithm_equality_negation_l1593_159342


namespace NUMINAMATH_CALUDE_prism_volume_l1593_159341

/-- 
Given a right rectangular prism with face areas 10, 15, and 6 square inches,
prove that its volume is 30 cubic inches.
-/
theorem prism_volume (l w h : ℝ) 
  (area1 : l * w = 10)
  (area2 : w * h = 15)
  (area3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1593_159341


namespace NUMINAMATH_CALUDE_wire_cutting_l1593_159343

theorem wire_cutting (total_length : ℝ) (shorter_length : ℝ) : 
  total_length = 50 →
  shorter_length + (5/2 * shorter_length) = total_length →
  shorter_length = 100/7 :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_l1593_159343


namespace NUMINAMATH_CALUDE_lucy_current_fish_l1593_159337

/-- The number of fish Lucy wants to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy would have after buying -/
def total_fish_after : ℕ := 280

/-- The current number of fish in Lucy's aquarium -/
def current_fish : ℕ := total_fish_after - fish_to_buy

theorem lucy_current_fish : current_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_current_fish_l1593_159337


namespace NUMINAMATH_CALUDE_maria_josh_age_sum_l1593_159362

/-- Proves that given the conditions about Maria and Josh's ages, the sum of their current ages is 31 years -/
theorem maria_josh_age_sum : 
  ∀ (maria josh : ℝ), 
  (maria = josh + 8) → 
  (maria + 6 = 3 * (josh - 3)) → 
  (maria + josh = 31) := by
sorry

end NUMINAMATH_CALUDE_maria_josh_age_sum_l1593_159362


namespace NUMINAMATH_CALUDE_odd_digits_157_base5_l1593_159392

/-- Represents a number in base 5 as a list of digits (least significant digit first) -/
def Base5Rep := List Nat

/-- Converts a natural number to its base 5 representation -/
def toBase5 (n : Nat) : Base5Rep :=
  sorry

/-- Counts the number of odd digits in a base 5 representation -/
def countOddDigits (rep : Base5Rep) : Nat :=
  sorry

/-- The number of odd digits in the base-5 representation of 157₁₀ is 3 -/
theorem odd_digits_157_base5 : countOddDigits (toBase5 157) = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_157_base5_l1593_159392


namespace NUMINAMATH_CALUDE_triangle_base_length_l1593_159313

/-- Proves that a triangle with area 54 square meters and height 6 meters has a base of 18 meters -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 54 →
  height = 6 →
  area = (base * height) / 2 →
  base = 18 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l1593_159313


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1593_159335

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 3rd to 7th terms equals 450. -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → SumCondition a → a 2 + a 8 = 180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1593_159335


namespace NUMINAMATH_CALUDE_initial_salary_correct_l1593_159323

/-- Kirt's initial monthly salary -/
def initial_salary : ℝ := 6000

/-- Kirt's salary increase rate after one year -/
def salary_increase_rate : ℝ := 0.30

/-- Kirt's total earnings after 3 years -/
def total_earnings : ℝ := 259200

/-- Theorem stating that the initial salary satisfies the given conditions -/
theorem initial_salary_correct : 
  12 * initial_salary + 24 * (initial_salary * (1 + salary_increase_rate)) = total_earnings := by
  sorry

#eval initial_salary

end NUMINAMATH_CALUDE_initial_salary_correct_l1593_159323


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1593_159391

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (r₁ + r₂ = 10 ∧ |r₁ - r₂| = 12) → (a = 1 ∧ b = -10 ∧ c = -11) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1593_159391


namespace NUMINAMATH_CALUDE_unique_line_exists_l1593_159368

def intersectionPoints (k m n c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((k, k^2 + 8*k + c), (k, m*k + n))

def verticalDistance (p q : ℝ × ℝ) : ℝ :=
  |p.2 - q.2|

theorem unique_line_exists (c : ℝ) : 
  ∃! m n : ℝ, n ≠ 0 ∧ 
  (∃ k : ℝ, verticalDistance (intersectionPoints k m n c).1 (intersectionPoints k m n c).2 = 4) ∧
  (m * 2 + n = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_line_exists_l1593_159368


namespace NUMINAMATH_CALUDE_sanya_towels_count_l1593_159349

/-- The number of bath towels Sanya can wash in one wash -/
def towels_per_wash : ℕ := 7

/-- The number of hours Sanya has per day for washing -/
def hours_per_day : ℕ := 2

/-- The number of days it takes to wash all towels -/
def days_to_wash_all : ℕ := 7

/-- The total number of bath towels Sanya has -/
def total_towels : ℕ := towels_per_wash * hours_per_day * days_to_wash_all

theorem sanya_towels_count : total_towels = 98 := by
  sorry

end NUMINAMATH_CALUDE_sanya_towels_count_l1593_159349


namespace NUMINAMATH_CALUDE_find_number_l1593_159328

theorem find_number : ∃ x : ℝ, (x / 18) - 29 = 6 ∧ x = 630 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1593_159328


namespace NUMINAMATH_CALUDE_solve_for_y_l1593_159339

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1593_159339


namespace NUMINAMATH_CALUDE_jogging_time_calculation_l1593_159394

theorem jogging_time_calculation (total_distance : ℝ) (total_time : ℝ) (initial_speed : ℝ) (later_speed : ℝ)
  (h1 : total_distance = 160)
  (h2 : total_time = 8)
  (h3 : initial_speed = 15)
  (h4 : later_speed = 10) :
  ∃ (initial_time : ℝ),
    initial_time * initial_speed + (total_time - initial_time) * later_speed = total_distance ∧
    initial_time = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_jogging_time_calculation_l1593_159394


namespace NUMINAMATH_CALUDE_construction_material_order_l1593_159361

theorem construction_material_order (concrete bricks stone total : ℝ) : 
  concrete = 0.17 →
  bricks = 0.17 →
  stone = 0.5 →
  total = concrete + bricks + stone →
  total = 0.84 := by
sorry

end NUMINAMATH_CALUDE_construction_material_order_l1593_159361


namespace NUMINAMATH_CALUDE_topsoil_cost_calculation_l1593_159399

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yard_to_cubic_foot : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def topsoil_volume_cubic_yards : ℝ := 8

/-- Calculate the cost of topsoil given its volume in cubic yards -/
def topsoil_cost (volume_cubic_yards : ℝ) : ℝ :=
  volume_cubic_yards * cubic_yard_to_cubic_foot * topsoil_cost_per_cubic_foot

theorem topsoil_cost_calculation :
  topsoil_cost topsoil_volume_cubic_yards = 1728 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_calculation_l1593_159399


namespace NUMINAMATH_CALUDE_milk_bag_probability_l1593_159378

theorem milk_bag_probability (total_bags : ℕ) (expired_bags : ℕ) (selected_bags : ℕ) : 
  total_bags = 5 → 
  expired_bags = 2 → 
  selected_bags = 2 → 
  (Nat.choose (total_bags - expired_bags) selected_bags : ℚ) / (Nat.choose total_bags selected_bags) = 3/10 := by
sorry

end NUMINAMATH_CALUDE_milk_bag_probability_l1593_159378


namespace NUMINAMATH_CALUDE_fraction_domain_l1593_159358

theorem fraction_domain (x : ℝ) : 
  (∃ y : ℝ, y = 5 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_domain_l1593_159358


namespace NUMINAMATH_CALUDE_smallest_n_for_good_sequence_2014_l1593_159365

/-- A sequence of real numbers is good if it satisfies certain conditions. -/
def IsGoodSequence (a : ℕ → ℝ) : Prop :=
  (∃ k : ℕ+, a k = 2014) ∧
  (∃ n : ℕ+, a 0 = n) ∧
  ∀ i : ℕ, a (i + 1) = 2 * a i + 1 ∨ a (i + 1) = a i / (a i + 2)

/-- The smallest positive integer n such that there exists a good sequence with aₙ = 2014 is 60. -/
theorem smallest_n_for_good_sequence_2014 :
  ∃ (a : ℕ → ℝ), IsGoodSequence a ∧ a 60 = 2014 ∧
  ∀ (b : ℕ → ℝ) (m : ℕ), m < 60 → IsGoodSequence b → b m ≠ 2014 := by
  sorry

#check smallest_n_for_good_sequence_2014

end NUMINAMATH_CALUDE_smallest_n_for_good_sequence_2014_l1593_159365


namespace NUMINAMATH_CALUDE_potato_peeling_time_l1593_159397

theorem potato_peeling_time (julie_rate ted_rate initial_time : ℝ) 
  (h1 : julie_rate = 1 / 10)  -- Julie's peeling rate per hour
  (h2 : ted_rate = 1 / 8)     -- Ted's peeling rate per hour
  (h3 : initial_time = 4)     -- Time they work together
  : (1 - (julie_rate + ted_rate) * initial_time) / julie_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_potato_peeling_time_l1593_159397


namespace NUMINAMATH_CALUDE_walking_distance_l1593_159370

/-- If a person walks 1.5 miles in 45 minutes, they will travel 3 miles in 90 minutes at the same rate. -/
theorem walking_distance (distance : ℝ) (time : ℝ) (new_time : ℝ) 
  (h1 : distance = 1.5)
  (h2 : time = 45)
  (h3 : new_time = 90) :
  (distance / time) * new_time = 3 := by
  sorry

#check walking_distance

end NUMINAMATH_CALUDE_walking_distance_l1593_159370


namespace NUMINAMATH_CALUDE_x_value_proof_l1593_159314

theorem x_value_proof (a b c x : ℝ) 
  (eq1 : a - b + c = 5)
  (eq2 : a^2 + b^2 + c^2 = 29)
  (eq3 : a*b + b*c + a*c = x^2) :
  x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1593_159314


namespace NUMINAMATH_CALUDE_divisor_sum_five_l1593_159327

/-- d(m) is the number of positive divisors of m -/
def d (m : ℕ+) : ℕ := sorry

/-- Theorem: For a positive integer n, d(n) + d(n+1) = 5 if and only if n = 3 or n = 4 -/
theorem divisor_sum_five (n : ℕ+) : d n + d (n + 1) = 5 ↔ n = 3 ∨ n = 4 := by sorry

end NUMINAMATH_CALUDE_divisor_sum_five_l1593_159327


namespace NUMINAMATH_CALUDE_mrs_hilt_candy_distribution_l1593_159332

/-- Mrs. Hilt's candy distribution problem -/
theorem mrs_hilt_candy_distribution 
  (chocolate_per_student : ℕ) 
  (chocolate_students : ℕ) 
  (hard_candy_per_student : ℕ) 
  (hard_candy_students : ℕ) 
  (gummy_per_student : ℕ) 
  (gummy_students : ℕ) 
  (h1 : chocolate_per_student = 2) 
  (h2 : chocolate_students = 3) 
  (h3 : hard_candy_per_student = 4) 
  (h4 : hard_candy_students = 2) 
  (h5 : gummy_per_student = 6) 
  (h6 : gummy_students = 4) : 
  chocolate_per_student * chocolate_students + 
  hard_candy_per_student * hard_candy_students + 
  gummy_per_student * gummy_students = 38 := by
sorry

end NUMINAMATH_CALUDE_mrs_hilt_candy_distribution_l1593_159332


namespace NUMINAMATH_CALUDE_sqrt_3600_equals_60_l1593_159384

theorem sqrt_3600_equals_60 : Real.sqrt 3600 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3600_equals_60_l1593_159384


namespace NUMINAMATH_CALUDE_dryer_weight_l1593_159330

def bridge_weight_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryer_count : ℕ := 3
def loaded_truck_weight : ℕ := 24000

theorem dryer_weight (h1 : bridge_weight_limit = 20000)
                     (h2 : empty_truck_weight = 12000)
                     (h3 : soda_crates = 20)
                     (h4 : soda_crate_weight = 50)
                     (h5 : dryer_count = 3)
                     (h6 : loaded_truck_weight = 24000) :
  let soda_weight := soda_crates * soda_crate_weight
  let produce_weight := 2 * soda_weight
  let truck_soda_produce_weight := empty_truck_weight + soda_weight + produce_weight
  let total_dryer_weight := loaded_truck_weight - truck_soda_produce_weight
  total_dryer_weight / dryer_count = 3000 := by
sorry

end NUMINAMATH_CALUDE_dryer_weight_l1593_159330


namespace NUMINAMATH_CALUDE_custom_operation_equality_l1593_159303

/-- Custom operation $ for real numbers -/
def dollar (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y : ℝ) :
  dollar ((x + y)^2) ((x - y)^2) = 4 * (x^2 + y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l1593_159303


namespace NUMINAMATH_CALUDE_problem_solution_l1593_159372

-- Define the function f
def f (x : ℝ) : ℝ := |x| - |x - 1|

-- Theorem statement
theorem problem_solution :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ |m - 1| → m ≤ 2) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a^2 + b^2 = 2 → a + b ≥ 2*a*b) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1593_159372


namespace NUMINAMATH_CALUDE_slope_product_is_negative_one_l1593_159359

/-- Parabola C: y^2 = 2px (p > 0) passing through (2, 2) -/
def parabola_C (p : ℝ) : Set (ℝ × ℝ) :=
  {point | point.2^2 = 2 * p * point.1 ∧ p > 0}

/-- Point Q on the parabola -/
def point_Q : ℝ × ℝ := (2, 2)

/-- Point M through which the intersecting line passes -/
def point_M : ℝ × ℝ := (2, 0)

/-- Origin O -/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem: The product of slopes of OA and OB is -1 -/
theorem slope_product_is_negative_one
  (p : ℝ)
  (h_p : p > 0)
  (h_Q : point_Q ∈ parabola_C p)
  (A B : ℝ × ℝ)
  (h_A : A ∈ parabola_C p)
  (h_B : B ∈ parabola_C p)
  (h_line : ∃ (m : ℝ), A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2)
  (k1 : ℝ) (h_k1 : k1 = (A.2 - origin.2) / (A.1 - origin.1))
  (k2 : ℝ) (h_k2 : k2 = (B.2 - origin.2) / (B.1 - origin.1)) :
  k1 * k2 = -1 := by sorry

end NUMINAMATH_CALUDE_slope_product_is_negative_one_l1593_159359


namespace NUMINAMATH_CALUDE_pizza_order_l1593_159387

theorem pizza_order (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 2) (h2 : total_slices = 28) :
  total_slices / slices_per_pizza = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l1593_159387


namespace NUMINAMATH_CALUDE_probability_three_in_same_group_l1593_159347

/-- The number of people to be partitioned -/
def total_people : ℕ := 15

/-- The number of groups -/
def num_groups : ℕ := 6

/-- The sizes of the groups -/
def group_sizes : List ℕ := [3, 3, 3, 2, 2, 2]

/-- The number of people we're interested in (Petruk, Gareng, and Bagong) -/
def num_interested : ℕ := 3

/-- The probability that Petruk, Gareng, and Bagong are in the same group -/
def probability_same_group : ℚ := 3 / 455

theorem probability_three_in_same_group :
  let total_ways := (total_people.factorial) / (group_sizes.map Nat.factorial).prod
  let favorable_ways := 3 * ((total_people - num_interested).factorial) / 
    ((group_sizes.tail.map Nat.factorial).prod)
  (favorable_ways : ℚ) / total_ways = probability_same_group := by
  sorry

end NUMINAMATH_CALUDE_probability_three_in_same_group_l1593_159347


namespace NUMINAMATH_CALUDE_speaker_must_be_trulalya_l1593_159351

/-- Represents the two brothers --/
inductive Brother
| T1 -- Tralyalya
| T2 -- Trulalya

/-- Represents the two possible card suits --/
inductive Suit
| Orange
| Purple

/-- Represents the statement made by a brother --/
structure Statement where
  speaker : Brother
  claimed_suit : Suit

/-- Represents the actual state of the cards --/
structure CardState where
  T1_card : Suit
  T2_card : Suit

/-- Determines if a statement is truthful given the actual card state --/
def is_truthful (s : Statement) (cs : CardState) : Prop :=
  match s.speaker with
  | Brother.T1 => s.claimed_suit = cs.T1_card
  | Brother.T2 => s.claimed_suit = cs.T2_card

/-- The main theorem: Given the conditions, the speaker must be Trulalya (T2) --/
theorem speaker_must_be_trulalya :
  ∀ (s : Statement) (cs : CardState),
    s.claimed_suit = Suit.Purple →
    is_truthful s cs →
    s.speaker = Brother.T2 :=
by sorry

end NUMINAMATH_CALUDE_speaker_must_be_trulalya_l1593_159351


namespace NUMINAMATH_CALUDE_problem_solution_l1593_159325

noncomputable def problem (e₁ e₂ OA OB : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  let A := OA
  let B := OB
  -- e₁ and e₂ are unit vectors in the direction of x-axis and y-axis
  e₁ = (1, 0) ∧ e₂ = (0, 1) ∧
  -- OA = e₁ + e₂
  OA = (e₁.1 + e₂.1, e₁.2 + e₂.2) ∧
  -- OB = 5e₁ + 3e₂
  OB = (5 * e₁.1 + 3 * e₂.1, 5 * e₁.2 + 3 * e₂.2) ∧
  -- AB ⟂ AC
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- |AB| = |AC|
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →
  OB.1 * C.1 + OB.2 * C.2 = 6 ∨ OB.1 * C.1 + OB.2 * C.2 = 10

theorem problem_solution :
  ∀ (e₁ e₂ OA OB : ℝ × ℝ) (C : ℝ × ℝ), problem e₁ e₂ OA OB C :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1593_159325


namespace NUMINAMATH_CALUDE_projection_problem_l1593_159379

/-- Given that the projection of (2, -3) onto some vector results in (1, -3/2),
    prove that the projection of (-3, 2) onto the same vector is (-24/13, 36/13) -/
theorem projection_problem (v : ℝ × ℝ) :
  let u₁ : ℝ × ℝ := (2, -3)
  let u₂ : ℝ × ℝ := (-3, 2)
  let proj₁ : ℝ × ℝ := (1, -3/2)
  (∃ (k : ℝ), v = k • proj₁) →
  (u₁ • v / (v • v)) • v = proj₁ →
  (u₂ • v / (v • v)) • v = (-24/13, 36/13) := by
  sorry


end NUMINAMATH_CALUDE_projection_problem_l1593_159379


namespace NUMINAMATH_CALUDE_sqrt_four_squared_l1593_159355

theorem sqrt_four_squared : (Real.sqrt 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_l1593_159355


namespace NUMINAMATH_CALUDE_line_equation_proof_l1593_159390

/-- Given a line defined by (-3, 4) · ((x, y) - (2, -6)) = 0, 
    prove that its slope-intercept form is y = (3/4)x - 7.5 
    and consequently, m = 3/4 and b = -7.5 -/
theorem line_equation_proof (x y : ℝ) : 
  (-3 : ℝ) * (x - 2) + 4 * (y + 6) = 0 → 
  y = (3/4 : ℝ) * x - (15/2 : ℝ) ∧ 
  (3/4 : ℝ) = (3/4 : ℝ) ∧ 
  -(15/2 : ℝ) = -(15/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1593_159390


namespace NUMINAMATH_CALUDE_eldorado_license_plates_l1593_159357

/-- The number of letters in the alphabet used for license plates -/
def alphabet_size : ℕ := 26

/-- The number of digits used for license plates -/
def digit_size : ℕ := 10

/-- The number of letter positions in a license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate -/
def digit_positions : ℕ := 4

/-- The total number of possible valid license plates in Eldorado -/
def total_license_plates : ℕ := alphabet_size ^ letter_positions * digit_size ^ digit_positions

theorem eldorado_license_plates :
  total_license_plates = 175760000 :=
by sorry

end NUMINAMATH_CALUDE_eldorado_license_plates_l1593_159357


namespace NUMINAMATH_CALUDE_trailing_zeros_of_999999999996_squared_l1593_159383

/-- The number of trailing zeros in 999,999,999,996^2 is 11 -/
theorem trailing_zeros_of_999999999996_squared : 
  (999999999996 : ℕ)^2 % 10^12 = 16 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_999999999996_squared_l1593_159383


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1593_159369

/-- A quadratic function with real coefficients -/
def f (a b x : ℝ) := x^2 + a*x + b

/-- The theorem statement -/
theorem quadratic_function_properties
  (a b : ℝ)
  (h_range : Set.range (f a b) = Set.Ici 0)
  (c m : ℝ)
  (h_solution_set : { x | f a b x < m } = Set.Ioo c (c + 2*Real.sqrt 2)) :
  m = 2 ∧
  ∃ (min_value : ℝ),
    min_value = 3 + 2*Real.sqrt 2 ∧
    ∀ (x y : ℝ), x > 1 → y > 0 → x + y = m →
      1 / (x - 1) + 2 / y ≥ min_value :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1593_159369


namespace NUMINAMATH_CALUDE_final_value_is_correct_l1593_159350

/-- Calculates the final value of sold games in USD given initial conditions and exchange rates -/
def final_value_usd (initial_value : ℝ) (usd_to_eur : ℝ) (eur_to_jpy : ℝ) (jpy_to_usd : ℝ) : ℝ :=
  let tripled_value := initial_value * 3
  let eur_value := tripled_value * usd_to_eur
  let jpy_value := eur_value * eur_to_jpy
  let sold_portion := 0.4
  let sold_value_jpy := jpy_value * sold_portion
  sold_value_jpy * jpy_to_usd

/-- Theorem stating that the final value of sold games is $225.42 given the initial conditions -/
theorem final_value_is_correct :
  final_value_usd 200 0.85 130 0.0085 = 225.42 := by
  sorry

#eval final_value_usd 200 0.85 130 0.0085

end NUMINAMATH_CALUDE_final_value_is_correct_l1593_159350


namespace NUMINAMATH_CALUDE_propositions_P_and_Q_l1593_159300

theorem propositions_P_and_Q : 
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 1/a + 1/b > 3) ∧
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_propositions_P_and_Q_l1593_159300


namespace NUMINAMATH_CALUDE_min_additions_for_54_l1593_159374

/-- A type representing a way to split the number 123456789 into parts -/
def Splitting := List Nat

/-- The original number we're working with -/
def originalNumber : Nat := 123456789

/-- Function to check if a splitting is valid (uses all digits in order) -/
def isValidSplitting (s : Splitting) : Prop :=
  s.foldl (· * 10 + ·) 0 = originalNumber

/-- Function to calculate the sum of a splitting -/
def sumOfSplitting (s : Splitting) : Nat :=
  s.sum

/-- The target sum we want to achieve -/
def targetSum : Nat := 54

/-- Theorem stating that the minimum number of addition signs needed is 7 -/
theorem min_additions_for_54 :
  (∃ (s : Splitting), isValidSplitting s ∧ sumOfSplitting s = targetSum) ∧
  (∀ (s : Splitting), isValidSplitting s ∧ sumOfSplitting s = targetSum → s.length ≥ 8) ∧
  (∃ (s : Splitting), isValidSplitting s ∧ sumOfSplitting s = targetSum ∧ s.length = 8) :=
sorry

end NUMINAMATH_CALUDE_min_additions_for_54_l1593_159374


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l1593_159363

theorem polynomial_division_degree (f q d r : Polynomial ℝ) :
  Polynomial.degree f = 17 →
  Polynomial.degree q = 10 →
  Polynomial.degree r = 5 →
  f = d * q + r →
  Polynomial.degree d = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l1593_159363


namespace NUMINAMATH_CALUDE_greatest_integer_with_nonpositive_product_l1593_159354

theorem greatest_integer_with_nonpositive_product (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_coprime : Nat.Coprime a b) :
  ∀ n : ℕ, n > a * b →
    ∃ x y : ℤ, (n : ℤ) = a * x + b * y ∧ x * y > 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_nonpositive_product_l1593_159354


namespace NUMINAMATH_CALUDE_problem_statement_l1593_159389

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (1/4) * x + (3/(4*x)) - 1

def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 4

theorem problem_statement (b : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g b x₂) ↔ b ≥ 17/8 :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1593_159389


namespace NUMINAMATH_CALUDE_meaningful_condition_l1593_159305

def is_meaningful (x : ℝ) : Prop :=
  x > -1 ∧ x ≠ 1

theorem meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x + 1) / (x - 1)) ↔ is_meaningful x :=
sorry

end NUMINAMATH_CALUDE_meaningful_condition_l1593_159305


namespace NUMINAMATH_CALUDE_money_distribution_l1593_159353

theorem money_distribution (total : ℕ) (a b c d : ℕ) : 
  a + b + c + d = total →
  5 * b = 2 * a →
  4 * b = 2 * c →
  3 * b = 2 * d →
  c = d + 500 →
  d = 1500 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l1593_159353
