import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3466_346621

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  P : ℝ × ℝ
  h_on_ellipse : (P.1^2 / E.a^2) + (P.2^2 / E.b^2) = 1

/-- The foci of the ellipse -/
def foci (E : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The eccentricity of an ellipse -/
def eccentricity (E : Ellipse) : ℝ := sorry

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem ellipse_eccentricity (E : Ellipse) (P : PointOnEllipse E) :
  let (F₁, F₂) := foci E
  dot_product (P.P.1 - F₁.1, P.P.2 - F₁.2) (P.P.1 - F₂.1, P.P.2 - F₂.2) = 0 →
  Real.tan (angle (P.P.1 - F₁.1, P.P.2 - F₁.2) (F₂.1 - F₁.1, F₂.2 - F₁.2)) = 1/2 →
  eccentricity E = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3466_346621


namespace NUMINAMATH_CALUDE_laptop_discount_theorem_l3466_346658

theorem laptop_discount_theorem (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  first_discount = 0.3 →
  second_discount = 0.5 →
  (1 - (1 - first_discount) * (1 - second_discount)) = 0.65 := by
sorry

end NUMINAMATH_CALUDE_laptop_discount_theorem_l3466_346658


namespace NUMINAMATH_CALUDE_max_time_digit_sum_l3466_346620

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The theorem stating the maximum sum of digits in a 24-hour time display -/
theorem max_time_digit_sum :
  (∃ (t : Time24), ∀ (t' : Time24), timeDigitSum t' ≤ timeDigitSum t) ∧
  (∀ (t : Time24), timeDigitSum t ≤ 24) :=
sorry

end NUMINAMATH_CALUDE_max_time_digit_sum_l3466_346620


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l3466_346657

theorem smallest_k_for_inequality : ∃ k : ℕ+, (k = 55 ∧ ∀ m : ℕ+, (m : ℝ)^5026 ≥ 2013^2013 → m ≥ k) := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l3466_346657


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l3466_346639

theorem ninth_term_of_sequence (n : ℕ) : 
  let a : ℕ → ℝ := fun n => Real.sqrt (3 * (2 * n - 1))
  a 14 = 9 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l3466_346639


namespace NUMINAMATH_CALUDE_range_of_m_l3466_346616

theorem range_of_m (x m : ℝ) : 
  (2 * x - m ≤ 3 ∧ -5 < x ∧ x < 4) ↔ m ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3466_346616


namespace NUMINAMATH_CALUDE_max_table_coverage_max_table_side_optimal_l3466_346601

/-- The side length of each square tablecloth in centimeters -/
def tablecloth_side : ℝ := 144

/-- The number of tablecloths available -/
def num_tablecloths : ℕ := 3

/-- The maximum side length of the square table that can be covered -/
def max_table_side : ℝ := 183

/-- Theorem stating that the maximum side length of a square table that can be completely
    covered by three square tablecloths, each with a side length of 144 cm, is 183 cm -/
theorem max_table_coverage :
  ∀ (table_side : ℝ),
  table_side ≤ max_table_side →
  (table_side ^ 2 : ℝ) ≤ num_tablecloths * tablecloth_side ^ 2 :=
by sorry

/-- Theorem stating that 183 cm is the largest possible side length for the table -/
theorem max_table_side_optimal :
  ∀ (larger_side : ℝ),
  larger_side > max_table_side →
  (larger_side ^ 2 : ℝ) > num_tablecloths * tablecloth_side ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_table_coverage_max_table_side_optimal_l3466_346601


namespace NUMINAMATH_CALUDE_shifted_linear_to_proportional_l3466_346655

/-- A linear function y = ax + b -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- Shift a linear function to the left by h units -/
def shift_left (f : LinearFunction) (h : ℝ) : LinearFunction :=
  { a := f.a, b := f.a * h + f.b }

/-- A function is directly proportional if it passes through the origin -/
def is_directly_proportional (f : LinearFunction) : Prop :=
  f.b = 0

/-- The main theorem -/
theorem shifted_linear_to_proportional (m : ℝ) : 
  let f : LinearFunction := { a := 2, b := m - 1 }
  let shifted_f := shift_left f 3
  is_directly_proportional shifted_f → m = -5 := by
sorry

end NUMINAMATH_CALUDE_shifted_linear_to_proportional_l3466_346655


namespace NUMINAMATH_CALUDE_investment_problem_l3466_346696

/-- Investment problem -/
theorem investment_problem 
  (x_investment : ℕ) 
  (y_investment : ℕ) 
  (z_join_time : ℕ) 
  (total_profit : ℕ) 
  (z_profit_share : ℕ) 
  (h1 : x_investment = 36000)
  (h2 : y_investment = 42000)
  (h3 : z_join_time = 4)
  (h4 : total_profit = 13860)
  (h5 : z_profit_share = 4032) :
  ∃ z_investment : ℕ, z_investment = 52000 ∧ 
    (x_investment * 12 + y_investment * 12) * z_profit_share = 
    z_investment * (12 - z_join_time) * (total_profit - z_profit_share) :=
sorry

end NUMINAMATH_CALUDE_investment_problem_l3466_346696


namespace NUMINAMATH_CALUDE_james_total_distance_l3466_346649

/-- Calculates the total distance driven given a series of driving segments -/
def total_distance (speeds : List ℝ) (times : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) speeds times)

/-- The total distance James drove under the given conditions -/
theorem james_total_distance :
  let speeds : List ℝ := [30, 60, 75, 60, 70]
  let times : List ℝ := [0.5, 0.75, 1.5, 2, 4]
  total_distance speeds times = 572.5 := by
  sorry

#check james_total_distance

end NUMINAMATH_CALUDE_james_total_distance_l3466_346649


namespace NUMINAMATH_CALUDE_no_valid_triangle_difference_l3466_346684

theorem no_valid_triangle_difference (n : ℕ) : 
  ((n + 3) * (n + 4)) / 2 - (n * (n + 1)) / 2 ≠ 111 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_triangle_difference_l3466_346684


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l3466_346697

/-- Calculates the total number of heartbeats during a race --/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (running_pace : ℕ) (resting_time : ℕ) : ℕ :=
  let running_time := race_distance * running_pace
  let total_time := running_time + resting_time
  total_time * heart_rate

/-- Theorem: The athlete's heart beats 29250 times during the race --/
theorem athlete_heartbeats : 
  total_heartbeats 150 30 6 15 = 29250 := by
  sorry

end NUMINAMATH_CALUDE_athlete_heartbeats_l3466_346697


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3466_346617

theorem shaded_area_calculation (π : Real) :
  let semicircle_area := π * 2^2 / 2
  let quarter_circle_area := π * 1^2 / 4
  semicircle_area - 2 * quarter_circle_area = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3466_346617


namespace NUMINAMATH_CALUDE_standard_flowchart_property_l3466_346642

/-- Represents a flowchart --/
structure Flowchart where
  start_points : Nat
  end_points : Nat

/-- A flowchart is standard if it has exactly one start point and at least one end point --/
def is_standard (f : Flowchart) : Prop :=
  f.start_points = 1 ∧ f.end_points ≥ 1

/-- Theorem stating that a standard flowchart has exactly one start point and at least one end point --/
theorem standard_flowchart_property (f : Flowchart) (h : is_standard f) :
  f.start_points = 1 ∧ f.end_points ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_standard_flowchart_property_l3466_346642


namespace NUMINAMATH_CALUDE_num_beakers_is_three_l3466_346635

def volume_per_tube : ℚ := 7
def num_tubes : ℕ := 6
def volume_per_beaker : ℚ := 14

theorem num_beakers_is_three : 
  (volume_per_tube * num_tubes) / volume_per_beaker = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_beakers_is_three_l3466_346635


namespace NUMINAMATH_CALUDE_inequality_proof_l3466_346628

theorem inequality_proof (x y : ℝ) (h : x > y) : -2 * x < -2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3466_346628


namespace NUMINAMATH_CALUDE_x_power_minus_reciprocal_l3466_346648

theorem x_power_minus_reciprocal (x : ℂ) : 
  x - (1 / x) = Complex.I * Real.sqrt 3 → x^2188 - (1 / x^2188) = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_minus_reciprocal_l3466_346648


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3466_346600

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3466_346600


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_squares_possible_l3466_346665

/-- Given three real numbers that form a geometric sequence and are not all equal,
    it's possible for their squares to form an arithmetic sequence. -/
theorem geometric_to_arithmetic_squares_possible (a b c : ℝ) :
  (∃ r : ℝ, r ≠ 1 ∧ b = a * r ∧ c = b * r) →  -- Geometric sequence condition
  (a ≠ b ∨ b ≠ c) →                          -- Not all equal condition
  ∃ x y z : ℝ, x = a^2 ∧ y = b^2 ∧ z = c^2 ∧  -- Squares of a, b, c
            y - x = z - y                    -- Arithmetic sequence condition
    := by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_squares_possible_l3466_346665


namespace NUMINAMATH_CALUDE_spending_problem_solution_l3466_346677

def spending_problem (initial_money : ℝ) : Prop :=
  let remaining_after_first := initial_money - (initial_money / 2 + 200)
  let spent_at_second := remaining_after_first / 2 + 300
  initial_money - (initial_money / 2 + 200) - spent_at_second = 350

theorem spending_problem_solution :
  spending_problem 3000 := by sorry

end NUMINAMATH_CALUDE_spending_problem_solution_l3466_346677


namespace NUMINAMATH_CALUDE_sine_special_angle_l3466_346692

theorem sine_special_angle (α : Real) 
  (h1 : π / 2 < α ∧ α < π) 
  (h2 : Real.sin (-π - α) = Real.sqrt 5 / 5) : 
  Real.sin (α - 3 * π / 2) = 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sine_special_angle_l3466_346692


namespace NUMINAMATH_CALUDE_not_equivalent_expression_l3466_346659

theorem not_equivalent_expression (x : ℝ) : 
  (3 * (x + 2) = 3 * x + 6) ∧
  ((-9 * x - 18) / (-3) = 3 * x + 6) ∧
  ((1/3) * (9 * x + 18) = 3 * x + 6) ∧
  ((1/3) * (3 * x) + (2/3) * 9 ≠ 3 * x + 6) :=
by sorry

end NUMINAMATH_CALUDE_not_equivalent_expression_l3466_346659


namespace NUMINAMATH_CALUDE_larger_divided_by_smaller_l3466_346676

theorem larger_divided_by_smaller (L S : ℕ) (h1 : L - S = 2395) (h2 : S = 476) (h3 : L % S = 15) :
  L / S = 6 := by
  sorry

end NUMINAMATH_CALUDE_larger_divided_by_smaller_l3466_346676


namespace NUMINAMATH_CALUDE_heine_valentine_treats_l3466_346640

/-- Given a total number of biscuits and dogs, calculate the number of biscuits per dog -/
def biscuits_per_dog (total_biscuits : ℕ) (num_dogs : ℕ) : ℕ :=
  total_biscuits / num_dogs

/-- Theorem: Mrs. Heine's Valentine's Day treats distribution -/
theorem heine_valentine_treats :
  let total_biscuits := 6
  let num_dogs := 2
  biscuits_per_dog total_biscuits num_dogs = 3 := by
  sorry

end NUMINAMATH_CALUDE_heine_valentine_treats_l3466_346640


namespace NUMINAMATH_CALUDE_adam_red_balls_l3466_346674

/-- The number of red balls in Adam's collection --/
def red_balls (total blue pink orange : ℕ) : ℕ :=
  total - (blue + pink + orange)

/-- Theorem stating the number of red balls in Adam's collection --/
theorem adam_red_balls :
  ∀ (total blue pink orange : ℕ),
    total = 50 →
    blue = 10 →
    orange = 5 →
    pink = 3 * orange →
    red_balls total blue pink orange = 20 := by
  sorry

end NUMINAMATH_CALUDE_adam_red_balls_l3466_346674


namespace NUMINAMATH_CALUDE_reading_time_calculation_l3466_346689

def total_homework_time : ℕ := 60
def math_time : ℕ := 15
def spelling_time : ℕ := 18

theorem reading_time_calculation :
  total_homework_time - (math_time + spelling_time) = 27 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l3466_346689


namespace NUMINAMATH_CALUDE_position_of_2000_l3466_346624

/-- Represents the column number (1 to 5) in the table -/
inductive Column
| one
| two
| three
| four
| five

/-- Represents a position in the table -/
structure Position where
  row : Nat
  column : Column

/-- Function to determine the position of a given even number in the table -/
def positionOfEvenNumber (n : Nat) : Position :=
  sorry

/-- The arrangement of positive even numbers follows the pattern described in the problem -/
axiom arrangement_pattern : ∀ n : Nat, n % 2 = 0 → n > 0 → 
  (positionOfEvenNumber n).column = Column.one ↔ n % 8 = 0

/-- Theorem stating that 2000 is in Row 250, Column 1 -/
theorem position_of_2000 : positionOfEvenNumber 2000 = { row := 250, column := Column.one } :=
  sorry

end NUMINAMATH_CALUDE_position_of_2000_l3466_346624


namespace NUMINAMATH_CALUDE_negative_division_equals_positive_division_of_negative_integers_l3466_346609

theorem negative_division_equals_positive (a b : Int) (h : b ≠ 0) :
  (-a) / (-b) = a / b :=
sorry

theorem division_of_negative_integers :
  (-81) / (-9) = 9 :=
sorry

end NUMINAMATH_CALUDE_negative_division_equals_positive_division_of_negative_integers_l3466_346609


namespace NUMINAMATH_CALUDE_round_984530_to_nearest_ten_thousand_l3466_346627

-- Define a function to round to the nearest ten thousand
def roundToNearestTenThousand (n : ℤ) : ℤ :=
  (n + 5000) / 10000 * 10000

-- State the theorem
theorem round_984530_to_nearest_ten_thousand :
  roundToNearestTenThousand 984530 = 980000 := by
  sorry

end NUMINAMATH_CALUDE_round_984530_to_nearest_ten_thousand_l3466_346627


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3466_346685

theorem triangle_angle_proof (a b c : ℝ) (A : ℝ) (S : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  S = (1/2) * b * c * Real.sin A →
  b^2 + c^2 = (1/3) * a^2 + (4 * Real.sqrt 3 / 3) * S →
  A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3466_346685


namespace NUMINAMATH_CALUDE_complex_calculations_l3466_346682

theorem complex_calculations :
  (∃ (i : ℂ), i * i = -1) →
  (∃ (z₁ z₂ : ℂ),
    (1 - i) * (-1/2 + (Real.sqrt 3)/2 * i) * (1 + i) = z₁ ∧
    z₁ = -1 + Real.sqrt 3 * i ∧
    (2 + 2*i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i))^2010 = z₂ ∧
    z₂ = -1 - 2*i) :=
by sorry

end NUMINAMATH_CALUDE_complex_calculations_l3466_346682


namespace NUMINAMATH_CALUDE_remainder_sum_l3466_346606

theorem remainder_sum (a b : ℤ) 
  (ha : a % 80 = 75) 
  (hb : b % 90 = 85) : 
  (a + b) % 40 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l3466_346606


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_22_over_3_l3466_346602

theorem greatest_integer_less_than_negative_22_over_3 :
  ⌊-22 / 3⌋ = -8 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_22_over_3_l3466_346602


namespace NUMINAMATH_CALUDE_correct_result_l3466_346662

variables {a b c : ℤ}

theorem correct_result (A : ℤ) (h : A + 2 * (a * b + 2 * b * c - 4 * a * c) = 3 * a * b - 2 * a * c + 5 * b * c) :
  A - 2 * (a * b + 2 * b * c - 4 * a * c) = -a * b + 14 * a * c - 3 * b * c := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l3466_346662


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3466_346607

theorem sufficient_not_necessary :
  (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) ∧
  (∀ x : ℝ, x > 2 → x^2 > 4) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3466_346607


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l3466_346686

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l3466_346686


namespace NUMINAMATH_CALUDE_bus_passengers_l3466_346679

theorem bus_passengers (initial : ℕ) (got_on : ℕ) (got_off : ℕ) (final : ℕ) : 
  got_on = 7 → got_off = 9 → final = 26 → initial + got_on - got_off = final → initial = 28 := by
sorry

end NUMINAMATH_CALUDE_bus_passengers_l3466_346679


namespace NUMINAMATH_CALUDE_garden_breadth_l3466_346603

/-- 
Given a rectangular garden with perimeter 800 meters and length 300 meters,
prove that its breadth is 100 meters.
-/
theorem garden_breadth (perimeter length breadth : ℝ) 
  (h1 : perimeter = 800)
  (h2 : length = 300)
  (h3 : perimeter = 2 * (length + breadth)) : 
  breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l3466_346603


namespace NUMINAMATH_CALUDE_cycle_price_proof_l3466_346633

/-- Proves that given a cycle sold at a 10% loss for Rs. 1620, its original price was Rs. 1800. -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1620)
  (h2 : loss_percentage = 10) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l3466_346633


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3466_346615

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : 2 * Real.cos (2*α) = Real.cos (α - π/4)) : 
  Real.sin (2*α) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3466_346615


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3466_346636

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3466_346636


namespace NUMINAMATH_CALUDE_unique_k_with_prime_roots_l3466_346608

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The quadratic equation x^2 - 75x + k = 0 has two prime roots -/
def hasPrimeRoots (k : ℤ) : Prop :=
  ∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 75 ∧ p * q = k

/-- There is exactly one integer k such that x^2 - 75x + k = 0 has two prime roots -/
theorem unique_k_with_prime_roots :
  ∃! (k : ℤ), hasPrimeRoots k :=
sorry

end NUMINAMATH_CALUDE_unique_k_with_prime_roots_l3466_346608


namespace NUMINAMATH_CALUDE_cube_sum_rational_l3466_346664

theorem cube_sum_rational (a b c : ℚ) 
  (h1 : a - b + c = 3) 
  (h2 : a^2 + b^2 + c^2 = 3) : 
  a^3 + b^3 + c^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_rational_l3466_346664


namespace NUMINAMATH_CALUDE_exponential_regression_model_l3466_346681

/-- Given a model y = ce^(kx) and its transformed linear equation z = 0.3x + 4 where z = ln y,
    prove that c = e^4 and k = 0.3 -/
theorem exponential_regression_model (c k : ℝ) : 
  (∀ x y : ℝ, y = c * Real.exp (k * x)) →
  (∀ x z : ℝ, z = 0.3 * x + 4) →
  (∀ y : ℝ, z = Real.log y) →
  c = Real.exp 4 ∧ k = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_regression_model_l3466_346681


namespace NUMINAMATH_CALUDE_frequency_distribution_necessary_sufficient_l3466_346650

/-- Represents a sample of data -/
structure Sample (α : Type) where
  data : List α

/-- Represents a frequency distribution of a sample -/
structure FrequencyDistribution (α : Type) where
  ranges : List (α × α)
  counts : List Nat

/-- Represents the proportion of data points falling within a range -/
def proportion {α : Type} (s : Sample α) (range : α × α) : ℝ := sorry

/-- Main theorem: The frequency distribution is necessary and sufficient to determine
    the proportion of data points falling within any range in a sample -/
theorem frequency_distribution_necessary_sufficient
  {α : Type} [LinearOrder α] (s : Sample α) :
  ∃ (fd : FrequencyDistribution α),
    (∀ (range : α × α), ∃ (p : ℝ), proportion s range = p) ↔
    (∀ (range : α × α), ∃ (count : Nat), count ∈ fd.counts) :=
  sorry

end NUMINAMATH_CALUDE_frequency_distribution_necessary_sufficient_l3466_346650


namespace NUMINAMATH_CALUDE_pattern_equation_l3466_346643

theorem pattern_equation (n : ℕ+) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equation_l3466_346643


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3466_346619

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x > 3 ∨ x < -1}

-- Define set B
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3466_346619


namespace NUMINAMATH_CALUDE_cake_recipe_proof_l3466_346694

/-- Represents the amounts of ingredients in cups -/
structure Recipe :=
  (flour : ℚ)
  (sugar : ℚ)
  (cocoa : ℚ)
  (milk : ℚ)

def original_recipe : Recipe :=
  { flour := 3/4
  , sugar := 2/3
  , cocoa := 1/3
  , milk := 1/2 }

def doubled_recipe : Recipe :=
  { flour := 2 * original_recipe.flour
  , sugar := 2 * original_recipe.sugar
  , cocoa := 2 * original_recipe.cocoa
  , milk := 2 * original_recipe.milk }

def already_added : Recipe :=
  { flour := 1/2
  , sugar := 1/4
  , cocoa := 0
  , milk := 0 }

def additional_needed : Recipe :=
  { flour := doubled_recipe.flour - already_added.flour
  , sugar := doubled_recipe.sugar - already_added.sugar
  , cocoa := doubled_recipe.cocoa - already_added.cocoa
  , milk := doubled_recipe.milk - already_added.milk }

theorem cake_recipe_proof :
  additional_needed.flour = 1 ∧
  additional_needed.sugar = 13/12 ∧
  additional_needed.cocoa = 2/3 ∧
  additional_needed.milk = 1 :=
sorry

end NUMINAMATH_CALUDE_cake_recipe_proof_l3466_346694


namespace NUMINAMATH_CALUDE_dave_won_ten_tickets_l3466_346654

/-- Calculates the number of tickets Dave won later at the arcade --/
def tickets_won_later (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Proves that Dave won 10 tickets later at the arcade --/
theorem dave_won_ten_tickets :
  tickets_won_later 11 5 16 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dave_won_ten_tickets_l3466_346654


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l3466_346666

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = 1 + (2/Real.sqrt 5)*t ∧ y = 1 + (1/Real.sqrt 5)*t

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    line_l t₁ A.1 A.2 ∧ curve_C A.1 A.2 ∧
    line_l t₂ B.1 B.2 ∧ curve_C B.1 B.2 ∧
    t₁ ≠ t₂

-- State the theorem
theorem intersection_distance_sum :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
  Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
  4 * Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l3466_346666


namespace NUMINAMATH_CALUDE_prob_same_heads_sum_l3466_346646

/-- Represents a coin with a given probability of landing heads -/
structure Coin where
  prob_heads : ℚ
  prob_heads_nonneg : 0 ≤ prob_heads
  prob_heads_le_one : prob_heads ≤ 1

/-- The set of three coins: two fair and one biased -/
def coin_set : Finset Coin := sorry

/-- The probability of getting the same number of heads when flipping the coin set twice -/
noncomputable def prob_same_heads (coins : Finset Coin) : ℚ := sorry

/-- The sum of numerator and denominator of the reduced fraction of prob_same_heads -/
noncomputable def sum_num_denom (coins : Finset Coin) : ℕ := sorry

theorem prob_same_heads_sum (h1 : coin_set.card = 3)
  (h2 : ∃ (c : Coin), c ∈ coin_set ∧ c.prob_heads = 1/2)
  (h3 : ∃ (c : Coin), c ∈ coin_set ∧ c.prob_heads = 3/5)
  (h4 : (coin_set.filter (fun c => c.prob_heads = 1/2)).card = 2) :
  sum_num_denom coin_set = 263 := by sorry

end NUMINAMATH_CALUDE_prob_same_heads_sum_l3466_346646


namespace NUMINAMATH_CALUDE_pedestrians_meeting_l3466_346698

/-- The problem of two pedestrians meeting --/
theorem pedestrians_meeting 
  (distance : ℝ) 
  (initial_meeting_time : ℝ) 
  (adjusted_meeting_time : ℝ) 
  (speed_multiplier_1 : ℝ) 
  (speed_multiplier_2 : ℝ) 
  (h1 : distance = 105) 
  (h2 : initial_meeting_time = 7.5) 
  (h3 : adjusted_meeting_time = 8 + 1/13) 
  (h4 : speed_multiplier_1 = 1.5) 
  (h5 : speed_multiplier_2 = 0.5) :
  ∃ (speed1 speed2 : ℝ), 
    speed1 = 8 ∧ 
    speed2 = 6 ∧ 
    initial_meeting_time * (speed1 + speed2) = distance ∧ 
    adjusted_meeting_time * (speed_multiplier_1 * speed1 + speed_multiplier_2 * speed2) = distance :=
by sorry


end NUMINAMATH_CALUDE_pedestrians_meeting_l3466_346698


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3466_346680

/-- Represents a number in a given base -/
def representIn (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Converts a representation in a given base to a natural number -/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ), a > 2 ∧ b > 2 ∧
  representIn 8 a = [1, 1] ∧
  representIn 8 b = [2, 2] ∧
  (∀ (n : ℕ) (a' b' : ℕ), a' > 2 → b' > 2 →
    representIn n a' = [1, 1] →
    representIn n b' = [2, 2] →
    n ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3466_346680


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3466_346691

theorem quadratic_equation_solution :
  ∃ (x1 x2 : ℝ), 
    x1 > 0 ∧ x2 > 0 ∧
    (1/2 * (4 * x1^2 - 1) = (x1^2 - 75*x1 - 15) * (x1^2 + 50*x1 + 10)) ∧
    (1/2 * (4 * x2^2 - 1) = (x2^2 - 75*x2 - 15) * (x2^2 + 50*x2 + 10)) ∧
    x1 = (75 + Real.sqrt 5773) / 2 ∧
    x2 = (-50 + Real.sqrt 2356) / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3466_346691


namespace NUMINAMATH_CALUDE_tourists_scientific_notation_l3466_346699

-- Define the number of tourists
def tourists : ℝ := 4.55e9

-- Theorem statement
theorem tourists_scientific_notation :
  tourists = 4.55 * (10 : ℝ) ^ 9 :=
by sorry

end NUMINAMATH_CALUDE_tourists_scientific_notation_l3466_346699


namespace NUMINAMATH_CALUDE_point_coordinates_in_second_quadrant_l3466_346695

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the second quadrant
def second_quadrant (p : Point) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Define distance to x-axis
def distance_to_x_axis (p : Point) : ℝ :=
  |p.2|

-- Define distance to y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  |p.1|

-- Theorem statement
theorem point_coordinates_in_second_quadrant (M : Point) 
  (h1 : second_quadrant M)
  (h2 : distance_to_x_axis M = 1)
  (h3 : distance_to_y_axis M = 2) :
  M = (-2, 1) :=
sorry

end NUMINAMATH_CALUDE_point_coordinates_in_second_quadrant_l3466_346695


namespace NUMINAMATH_CALUDE_shuttle_speed_kph_l3466_346630

/-- Conversion factor from seconds to hours -/
def seconds_per_hour : ℕ := 3600

/-- Speed of the space shuttle in kilometers per second -/
def shuttle_speed_kps : ℝ := 6

/-- Theorem stating that the space shuttle's speed in kilometers per hour
    is equal to 21600 -/
theorem shuttle_speed_kph :
  shuttle_speed_kps * seconds_per_hour = 21600 := by
  sorry

end NUMINAMATH_CALUDE_shuttle_speed_kph_l3466_346630


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3466_346675

/-- Represents a unit cube -/
structure UnitCube where
  volume : ℝ := 1
  surfaceArea : ℝ := 6

/-- Represents the custom shape described in the problem -/
structure CustomShape where
  baseCubes : Fin 5 → UnitCube
  topCube : UnitCube
  bottomCube : UnitCube

/-- Calculates the total volume of the CustomShape -/
def totalVolume (shape : CustomShape) : ℝ :=
  7  -- 5 base cubes + 1 top cube + 1 bottom cube

/-- Calculates the total surface area of the CustomShape -/
def totalSurfaceArea (shape : CustomShape) : ℝ :=
  28  -- As calculated in the problem

/-- The main theorem to be proved -/
theorem volume_to_surface_area_ratio (shape : CustomShape) :
  totalVolume shape / totalSurfaceArea shape = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3466_346675


namespace NUMINAMATH_CALUDE_dave_first_six_l3466_346637

/-- The probability of tossing a six on a single throw -/
def prob_six : ℚ := 1 / 6

/-- The probability of not tossing a six on a single throw -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of players before Dave in each round -/
def players_before_dave : ℕ := 3

/-- The total number of players -/
def total_players : ℕ := 4

/-- The probability that Dave is the first to toss a six -/
theorem dave_first_six : 
  (prob_six * prob_not_six ^ players_before_dave) / 
  (1 - prob_not_six ^ total_players) = 125 / 671 := by
  sorry

end NUMINAMATH_CALUDE_dave_first_six_l3466_346637


namespace NUMINAMATH_CALUDE_coin_equation_solution_l3466_346668

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of quarters on the left side of the equation -/
def left_quarters : ℕ := 30

/-- The number of dimes on the left side of the equation -/
def left_dimes : ℕ := 20

/-- The number of quarters on the right side of the equation -/
def right_quarters : ℕ := 5

theorem coin_equation_solution :
  ∃ n : ℕ, 
    left_quarters * quarter_value + left_dimes * dime_value = 
    right_quarters * quarter_value + n * dime_value ∧
    n = 83 := by
  sorry

end NUMINAMATH_CALUDE_coin_equation_solution_l3466_346668


namespace NUMINAMATH_CALUDE_company_production_l3466_346638

/-- Represents the daily water bottle production of a company -/
def DailyProduction : Type :=
  { bottles : ℕ // bottles > 0 }

/-- Represents the capacity of a case in bottles -/
def CaseCapacity : Type :=
  { capacity : ℕ // capacity > 0 }

/-- Represents the number of cases required for daily production -/
def RequiredCases : Type :=
  { cases : ℕ // cases > 0 }

/-- Calculates the total number of bottles produced daily -/
def calculateDailyProduction (capacity : CaseCapacity) (required : RequiredCases) : DailyProduction :=
  ⟨capacity.val * required.val, sorry⟩

theorem company_production
  (case_capacity : CaseCapacity)
  (required_cases : RequiredCases)
  (h1 : case_capacity.val = 9)
  (h2 : required_cases.val = 8000) :
  (calculateDailyProduction case_capacity required_cases).val = 72000 :=
sorry

end NUMINAMATH_CALUDE_company_production_l3466_346638


namespace NUMINAMATH_CALUDE_blake_poured_out_02_gallons_l3466_346687

/-- The amount of water Blake poured out, given initial and remaining amounts -/
def water_poured_out (initial : Real) (remaining : Real) : Real :=
  initial - remaining

/-- Theorem: Blake poured out 0.2 gallons of water -/
theorem blake_poured_out_02_gallons :
  let initial := 0.8
  let remaining := 0.6
  water_poured_out initial remaining = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_blake_poured_out_02_gallons_l3466_346687


namespace NUMINAMATH_CALUDE_set_A_is_correct_l3466_346626

-- Define the universe set U
def U : Set ℝ := {x | x > 0}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := {x | 0 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | x ≥ 3}

-- Theorem statement
theorem set_A_is_correct : A = U \ complement_A_in_U := by sorry

end NUMINAMATH_CALUDE_set_A_is_correct_l3466_346626


namespace NUMINAMATH_CALUDE_sequence_general_term_l3466_346661

def sequence_property (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  ∀ n : ℕ, n > 0 → (n + 1 : ℝ) * a (n + 1) - n * (a n)^2 + (n + 1 : ℝ) * a n * a (n + 1) - n * a n = 0

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_property a) :
  ∀ n : ℕ, n > 0 → a n = 1 / n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3466_346661


namespace NUMINAMATH_CALUDE_expression_evaluation_l3466_346605

theorem expression_evaluation : -2^3 / (-2) + (-2)^2 * (-5) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3466_346605


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l3466_346660

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - abs x - 6

-- Theorem stating that f has exactly two zeros
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l3466_346660


namespace NUMINAMATH_CALUDE_intersection_distance_l3466_346610

-- Define the lines and point A
def l₁ (t : ℝ) : ℝ × ℝ := (1 + 3*t, 2 - 4*t)
def l₂ (x y : ℝ) : Prop := 2*x - 4*y = 5
def A : ℝ × ℝ := (1, 2)

-- State the theorem
theorem intersection_distance :
  ∃ (t : ℝ), 
    let B := l₁ t
    l₂ B.1 B.2 ∧ 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l3466_346610


namespace NUMINAMATH_CALUDE_not_in_second_quadrant_l3466_346618

/-- A linear function f(x) = x - 1 -/
def f (x : ℝ) : ℝ := x - 1

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The linear function f(x) = x - 1 does not pass through the second quadrant -/
theorem not_in_second_quadrant :
  ∀ x : ℝ, ¬(second_quadrant x (f x)) :=
by sorry

end NUMINAMATH_CALUDE_not_in_second_quadrant_l3466_346618


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3466_346634

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔
  (a < 2 ∧ a ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3466_346634


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l3466_346611

-- Define the circles C₁ and C₂
def C₁ (x y a : ℝ) : Prop := x^2 + y^2 - a = 0
def C₂ (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*a*y + 3 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - 4*y + 5 = 0

-- Define symmetry between circles with respect to a line
def symmetric_circles (C₁ C₂ : (ℝ → ℝ → ℝ → Prop)) (l : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a : ℝ, ∀ x y : ℝ, C₁ x y a ∧ C₂ x y a → l x y

-- Theorem statement
theorem circle_symmetry_line :
  symmetric_circles C₁ C₂ line_l :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l3466_346611


namespace NUMINAMATH_CALUDE_number_ratio_problem_l3466_346641

theorem number_ratio_problem (N : ℝ) : 
  (1/3 : ℝ) * (2/5 : ℝ) * N = 15 ∧ (40/100 : ℝ) * N = 180 → 
  15 / N = 1 / 7.5 :=
by sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l3466_346641


namespace NUMINAMATH_CALUDE_evaluate_polynomial_l3466_346651

theorem evaluate_polynomial (y : ℝ) (h : y = 2) : y^4 + y^3 + y^2 + y + 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_l3466_346651


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3466_346673

/-- Given a geometric sequence with first term a and common ratio r,
    prove that if the first three terms are of the form a, 3a+3, 6a+6,
    then the fourth term is -24. -/
theorem geometric_sequence_fourth_term
  (a r : ℝ) -- a is the first term, r is the common ratio
  (h1 : (3*a + 3) = a * r) -- second term = first term * r
  (h2 : (6*a + 6) = (3*a + 3) * r) -- third term = second term * r
  : a * r^3 = -24 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3466_346673


namespace NUMINAMATH_CALUDE_largest_triangle_area_l3466_346647

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The area of triangle ABC given p -/
def triangle_area (p : ℝ) : ℝ := 2 * |((p - 1) * (p - 3))|

theorem largest_triangle_area :
  ∃ (max_area : ℝ),
    (∀ p : ℝ, 0 ≤ p → p ≤ 4 → triangle_area p ≤ max_area) ∧
    (∃ p : ℝ, 0 ≤ p ∧ p ≤ 4 ∧ triangle_area p = max_area) ∧
    max_area = 2 :=
sorry

end NUMINAMATH_CALUDE_largest_triangle_area_l3466_346647


namespace NUMINAMATH_CALUDE_thief_speed_l3466_346629

/-- The speed of a thief given chase conditions -/
theorem thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ) : 
  initial_distance = 0.2 →
  policeman_speed = 10 →
  thief_distance = 0.8 →
  ∃ (thief_speed : ℝ), 
    thief_speed = 8 ∧ 
    (initial_distance + thief_distance) / policeman_speed = thief_distance / thief_speed :=
by sorry

end NUMINAMATH_CALUDE_thief_speed_l3466_346629


namespace NUMINAMATH_CALUDE_taxi_ride_cost_l3466_346663

/-- Calculates the total cost of a taxi ride -/
def taxi_cost (base_fare : ℝ) (per_mile_rate : ℝ) (tax_rate : ℝ) (distance : ℝ) : ℝ :=
  let fare_without_tax := base_fare + per_mile_rate * distance
  let tax := tax_rate * fare_without_tax
  fare_without_tax + tax

/-- Theorem: The total cost of an 8-mile taxi ride is $4.84 -/
theorem taxi_ride_cost :
  taxi_cost 2.00 0.30 0.10 8 = 4.84 := by
  sorry

end NUMINAMATH_CALUDE_taxi_ride_cost_l3466_346663


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l3466_346667

theorem correct_quotient_proof (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 63) : D / 21 = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l3466_346667


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3466_346631

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 1 ∧ 
  n % 5 = 3 ∧ 
  n % 8 = 4 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 3 = 1 ∧ m % 5 = 3 ∧ m % 8 = 4 → m ≥ n) ∧
  n = 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3466_346631


namespace NUMINAMATH_CALUDE_square_difference_l3466_346612

theorem square_difference (x : ℤ) (h : x^2 = 1764) : (x + 2) * (x - 2) = 1760 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3466_346612


namespace NUMINAMATH_CALUDE_correct_apple_count_l3466_346690

/-- Represents the types of apples Aria needs to buy. -/
structure AppleCount where
  red : ℕ
  granny : ℕ
  golden : ℕ

/-- Calculates the total number of apples Aria needs to buy for two weeks. -/
def totalApplesForTwoWeeks (normalDays weekDays : ℕ) (normalMix specialMix : AppleCount) : AppleCount :=
  { red := normalDays * normalMix.red + weekDays * specialMix.red,
    granny := normalDays * normalMix.granny + weekDays * specialMix.granny,
    golden := (normalDays + weekDays) * normalMix.golden }

/-- Theorem stating the correct number of apples Aria needs to buy for two weeks. -/
theorem correct_apple_count :
  let normalDays : ℕ := 10
  let weekDays : ℕ := 4
  let normalMix : AppleCount := { red := 1, granny := 2, golden := 1 }
  let specialMix : AppleCount := { red := 2, granny := 1, golden := 1 }
  let result := totalApplesForTwoWeeks normalDays weekDays normalMix specialMix
  result.red = 18 ∧ result.granny = 24 ∧ result.golden = 14 := by sorry

end NUMINAMATH_CALUDE_correct_apple_count_l3466_346690


namespace NUMINAMATH_CALUDE_leila_marathon_distance_l3466_346645

/-- Represents the total distance covered in marathons -/
structure MarathonDistance where
  miles : ℕ
  yards : ℕ

/-- Calculates the total distance covered in multiple marathons -/
def totalDistance (numMarathons : ℕ) (marathonMiles : ℕ) (marathonYards : ℕ) (yardsPerMile : ℕ) : MarathonDistance :=
  sorry

/-- Theorem stating the total distance covered by Leila in her marathons -/
theorem leila_marathon_distance :
  let numMarathons : ℕ := 15
  let marathonMiles : ℕ := 26
  let marathonYards : ℕ := 385
  let yardsPerMile : ℕ := 1760
  let result := totalDistance numMarathons marathonMiles marathonYards yardsPerMile
  result.miles = 393 ∧ result.yards = 495 ∧ result.yards < yardsPerMile :=
by sorry

end NUMINAMATH_CALUDE_leila_marathon_distance_l3466_346645


namespace NUMINAMATH_CALUDE_average_with_added_number_l3466_346644

theorem average_with_added_number (x : ℝ) : 
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 := by sorry

end NUMINAMATH_CALUDE_average_with_added_number_l3466_346644


namespace NUMINAMATH_CALUDE_function_properties_and_triangle_perimeter_l3466_346688

noncomputable def f (m : ℝ) (θ : ℝ) (x : ℝ) : ℝ := (m + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

theorem function_properties_and_triangle_perimeter
  (m : ℝ)
  (θ : ℝ)
  (h1 : ∀ x, f m θ x = -f m θ (-x))  -- f is an odd function
  (h2 : f m θ (π/4) = 0)
  (h3 : 0 < θ)
  (h4 : θ < π) :
  (∀ x, f m θ x = -1/2 * Real.sin (4*x)) ∧
  (∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    f m θ (Real.arccos (1/2) / 2 + π/24) = -1/2 ∧
    1 = 1 ∧
    a * b = 2 * Real.sqrt 3 ∧
    a + b + 1 = 3 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_and_triangle_perimeter_l3466_346688


namespace NUMINAMATH_CALUDE_optimal_solution_satisfies_criteria_l3466_346653

/-- Represents the optimal solution for the medicine problem -/
def optimal_solution : ℕ × ℕ := (6, 3)

/-- Vitamin contents of the first medicine -/
def medicine1_contents : Fin 4 → ℕ
| 0 => 3  -- Vitamin A
| 1 => 1  -- Vitamin B
| 2 => 1  -- Vitamin C
| 3 => 0  -- Vitamin D

/-- Vitamin contents of the second medicine -/
def medicine2_contents : Fin 4 → ℕ
| 0 => 0  -- Vitamin A
| 1 => 1  -- Vitamin B
| 2 => 3  -- Vitamin C
| 3 => 2  -- Vitamin D

/-- Daily vitamin requirements -/
def daily_requirements : Fin 4 → ℕ
| 0 => 3  -- Vitamin A
| 1 => 9  -- Vitamin B
| 2 => 15 -- Vitamin C
| 3 => 2  -- Vitamin D

/-- Cost of medicines in fillér -/
def medicine_costs : Fin 2 → ℕ
| 0 => 20  -- Cost of medicine 1
| 1 => 60  -- Cost of medicine 2

/-- Theorem stating that the optimal solution satisfies all criteria -/
theorem optimal_solution_satisfies_criteria :
  let (x, y) := optimal_solution
  (x + y = 9) ∧ 
  (medicine_costs 0 * x + medicine_costs 1 * y = 300) ∧
  (x + 2 * y = 12) ∧
  (∀ i : Fin 4, medicine1_contents i * x + medicine2_contents i * y ≥ daily_requirements i) :=
by sorry

end NUMINAMATH_CALUDE_optimal_solution_satisfies_criteria_l3466_346653


namespace NUMINAMATH_CALUDE_absent_students_percentage_l3466_346622

theorem absent_students_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (boys_absent_fraction : ℚ) (girls_absent_fraction : ℚ) :
  total_students = 120 →
  boys = 70 →
  girls = 50 →
  boys_absent_fraction = 1 / 7 →
  girls_absent_fraction = 1 / 5 →
  (↑(boys_absent_fraction * boys + girls_absent_fraction * girls) : ℚ) / total_students = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_absent_students_percentage_l3466_346622


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l3466_346683

theorem equation_has_real_roots (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l3466_346683


namespace NUMINAMATH_CALUDE_andrena_debelyn_difference_l3466_346632

/-- The number of dolls each person has after the gift exchange --/
structure DollCounts where
  debelyn : ℕ
  christel : ℕ
  andrena : ℕ

/-- Calculate the final doll counts based on initial counts and gifts --/
def finalCounts (debelynInitial christelInitial : ℕ) : DollCounts :=
  let debelynFinal := debelynInitial - 2
  let christelFinal := christelInitial - 5
  let andrenaFinal := christelFinal + 2
  { debelyn := debelynFinal
  , christel := christelFinal
  , andrena := andrenaFinal }

/-- Theorem stating the difference in doll counts between Andrena and Debelyn --/
theorem andrena_debelyn_difference : 
  let counts := finalCounts 20 24
  counts.andrena - counts.debelyn = 3 := by sorry

end NUMINAMATH_CALUDE_andrena_debelyn_difference_l3466_346632


namespace NUMINAMATH_CALUDE_max_quarters_exact_solution_l3466_346693

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- Tony's total money in dollars -/
def total_money : ℚ := 490 / 100

/-- 
  Given that Tony has the same number of quarters and dimes, and his total money is $4.90,
  prove that the maximum number of quarters he can have is 14.
-/
theorem max_quarters : 
  ∀ q : ℕ, 
  (q : ℚ) * (quarter_value + dime_value) ≤ total_money → 
  q ≤ 14 :=
by sorry

/-- Prove that 14 quarters and 14 dimes exactly equal $4.90 -/
theorem exact_solution : 
  (14 : ℚ) * (quarter_value + dime_value) = total_money :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_exact_solution_l3466_346693


namespace NUMINAMATH_CALUDE_gretchen_earnings_l3466_346671

/-- Calculates the total earnings for Gretchen's caricature drawings over a weekend -/
def weekend_earnings (price_per_drawing : ℕ) (saturday_sales : ℕ) (sunday_sales : ℕ) : ℕ :=
  price_per_drawing * (saturday_sales + sunday_sales)

/-- Proves that Gretchen's earnings for the weekend are $800 -/
theorem gretchen_earnings :
  weekend_earnings 20 24 16 = 800 := by
sorry

end NUMINAMATH_CALUDE_gretchen_earnings_l3466_346671


namespace NUMINAMATH_CALUDE_cone_surface_area_ratio_l3466_346656

/-- For a cone whose lateral surface unfolds into a sector with a central angle of 120° and radius 1,
    the ratio of its surface area to its lateral surface area is 4:3 -/
theorem cone_surface_area_ratio :
  let sector_angle : Real := 120 * π / 180
  let sector_radius : Real := 1
  let lateral_surface_area : Real := π * sector_radius^2 * (sector_angle / (2 * π))
  let base_radius : Real := sector_radius * sector_angle / (2 * π)
  let base_area : Real := π * base_radius^2
  let surface_area : Real := lateral_surface_area + base_area
  (surface_area / lateral_surface_area) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_ratio_l3466_346656


namespace NUMINAMATH_CALUDE_min_value_theorem_l3466_346652

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  ∃ (min_val : ℝ), min_val = 8 + 4 * Real.sqrt 3 ∧
  ∀ z, z = (x + 1) * (y + 1) / (x * y) → z ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3466_346652


namespace NUMINAMATH_CALUDE_sams_book_count_l3466_346623

/-- The number of books Sam bought at the school's book fair -/
def total_books (adventure_books mystery_books crime_books : ℝ) : ℝ :=
  adventure_books + mystery_books + crime_books

/-- Theorem stating the total number of books Sam bought -/
theorem sams_book_count :
  total_books 13 17 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sams_book_count_l3466_346623


namespace NUMINAMATH_CALUDE_total_lives_after_joining_l3466_346613

theorem total_lives_after_joining (initial_players : Nat) (joined_players : Nat) (lives_per_player : Nat) : 
  initial_players = 8 → joined_players = 2 → lives_per_player = 6 → 
  (initial_players + joined_players) * lives_per_player = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_after_joining_l3466_346613


namespace NUMINAMATH_CALUDE_gcd_bound_for_special_fraction_l3466_346672

theorem gcd_bound_for_special_fraction (a b : ℕ+) 
  (h : ∃ (k : ℤ), (a.1 + 1 : ℚ) / b.1 + (b.1 + 1 : ℚ) / a.1 = k) : 
  Nat.gcd a.1 b.1 ≤ Real.sqrt (a.1 + b.1) := by
  sorry

end NUMINAMATH_CALUDE_gcd_bound_for_special_fraction_l3466_346672


namespace NUMINAMATH_CALUDE_pen_purchase_ratio_l3466_346678

/-- The ratio of fountain pens to ballpoint pens in a purchase scenario --/
theorem pen_purchase_ratio (x y : ℕ) (h1 : (2 * x + y) * 3 = 3 * (2 * y + x)) :
  y = 4 * x := by
  sorry

#check pen_purchase_ratio

end NUMINAMATH_CALUDE_pen_purchase_ratio_l3466_346678


namespace NUMINAMATH_CALUDE_smallest_winning_number_l3466_346614

theorem smallest_winning_number : ∃ N : ℕ, 
  (N = 46) ∧ 
  (0 ≤ N) ∧ 
  (N ≤ 999) ∧ 
  (9 * N - 80 < 1000) ∧ 
  (27 * N - 240 ≥ 1000) ∧ 
  (∀ k : ℕ, k < N → (9 * k - 80 ≥ 1000 ∨ 27 * k - 240 < 1000)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l3466_346614


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3466_346625

-- Define the fraction 7/12
def fraction : ℚ := 7 / 12

-- Define the decimal approximation
def decimal_approx : ℝ := 0.5833

-- Define the maximum allowed error due to rounding
def max_error : ℝ := 0.00005

-- Theorem statement
theorem fraction_to_decimal :
  |((fraction : ℝ) - decimal_approx)| < max_error := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3466_346625


namespace NUMINAMATH_CALUDE_f_equality_l3466_346669

noncomputable def f (x : ℝ) : ℝ := Real.arctan ((2 * x) / (1 - x^2))

theorem f_equality (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : 3 - 4 * x^2 ≠ 0) :
  f ((x - 4 * x^3) / (3 - 4 * x^2)) = f x :=
by sorry

end NUMINAMATH_CALUDE_f_equality_l3466_346669


namespace NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_2_mod_25_l3466_346604

theorem largest_four_digit_negative_congruent_to_2_mod_25 :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n % 25 = 2 → n ≤ -1023 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_2_mod_25_l3466_346604


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3466_346670

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3466_346670
