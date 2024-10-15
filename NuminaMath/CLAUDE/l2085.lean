import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l2085_208536

open Real

theorem range_of_a (x₁ x₂ a : ℝ) (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_distinct : x₁ ≠ x₂)
  (h_equation : x₁ + a * (x₂ - 2 * ℯ * x₁) * (log x₂ - log x₁) = 0) :
  a < 0 ∨ a ≥ 1 / ℯ := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2085_208536


namespace NUMINAMATH_CALUDE_infinitely_many_superabundant_numbers_l2085_208516

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Define superabundant numbers
def is_superabundant (m : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k < m → (sigma m : ℚ) / m > (sigma k : ℚ) / k

-- Define the set of superabundant numbers
def superabundant_set : Set ℕ :=
  {m : ℕ | is_superabundant m}

-- Theorem statement
theorem infinitely_many_superabundant_numbers :
  Set.Infinite superabundant_set := by sorry

end NUMINAMATH_CALUDE_infinitely_many_superabundant_numbers_l2085_208516


namespace NUMINAMATH_CALUDE_exists_real_less_than_negative_one_l2085_208598

theorem exists_real_less_than_negative_one : ∃ x : ℝ, x < -1 := by
  sorry

end NUMINAMATH_CALUDE_exists_real_less_than_negative_one_l2085_208598


namespace NUMINAMATH_CALUDE_inequality_proof_l2085_208519

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (a + c + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2085_208519


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2085_208548

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + r^2 + 5 * r - 3) - (r^3 + 3 * r^2 + 9 * r - 2) = r^3 - 2 * r^2 - 4 * r - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2085_208548


namespace NUMINAMATH_CALUDE_cab_ride_cost_per_mile_l2085_208572

/-- Calculates the cost per mile for Briar's cab rides -/
theorem cab_ride_cost_per_mile
  (days : ℕ)
  (distance_to_event : ℝ)
  (total_cost : ℝ)
  (h1 : days = 7)
  (h2 : distance_to_event = 200)
  (h3 : total_cost = 7000) :
  total_cost / (2 * days * distance_to_event) = 2.5 := by
  sorry

#check cab_ride_cost_per_mile

end NUMINAMATH_CALUDE_cab_ride_cost_per_mile_l2085_208572


namespace NUMINAMATH_CALUDE_matrix_power_sum_l2085_208502

/-- Given a 3x3 matrix C and a natural number m, 
    if C^m equals a specific matrix and C has a specific form,
    then b + m = 310 where b is an element of C. -/
theorem matrix_power_sum (b m : ℕ) (C : Matrix (Fin 3) (Fin 3) ℕ) : 
  C^m = !![1, 33, 3080; 1, 1, 65; 1, 0, 1] ∧ 
  C = !![1, 3, b; 0, 1, 5; 1, 0, 1] → 
  b + m = 310 := by sorry

end NUMINAMATH_CALUDE_matrix_power_sum_l2085_208502


namespace NUMINAMATH_CALUDE_second_replaced_man_age_is_23_l2085_208561

/-- The age of the second replaced man in a group where:
  * There are 8 men initially
  * Two men are replaced
  * The average age increases by 2 years after replacement
  * One of the replaced men is 21 years old
  * The average age of the two new men is 30 years
-/
def second_replaced_man_age : ℕ := by
  -- Define the initial number of men
  let initial_count : ℕ := 8
  -- Define the age increase after replacement
  let age_increase : ℕ := 2
  -- Define the age of the first replaced man
  let first_replaced_age : ℕ := 21
  -- Define the average age of the new men
  let new_men_avg_age : ℕ := 30

  -- The actual proof would go here
  sorry

theorem second_replaced_man_age_is_23 : second_replaced_man_age = 23 := by
  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_second_replaced_man_age_is_23_l2085_208561


namespace NUMINAMATH_CALUDE_derivative_at_one_l2085_208567

-- Define the function
def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2085_208567


namespace NUMINAMATH_CALUDE_remaining_sausage_meat_l2085_208571

/-- Calculates the remaining sausage meat in ounces after some links are eaten -/
theorem remaining_sausage_meat 
  (total_pounds : ℕ) 
  (total_links : ℕ) 
  (eaten_links : ℕ) 
  (h1 : total_pounds = 10) 
  (h2 : total_links = 40) 
  (h3 : eaten_links = 12) : 
  (total_pounds * 16 - (total_pounds * 16 / total_links) * eaten_links : ℕ) = 112 := by
  sorry

#check remaining_sausage_meat

end NUMINAMATH_CALUDE_remaining_sausage_meat_l2085_208571


namespace NUMINAMATH_CALUDE_archer_arrow_cost_l2085_208564

/-- Represents the archer's arrow usage and costs -/
structure ArcherData where
  shots_per_week : ℕ
  recovery_rate : ℚ
  personal_expense_rate : ℚ
  personal_expense : ℚ

/-- Calculates the cost per arrow given the archer's data -/
def cost_per_arrow (data : ArcherData) : ℚ :=
  let total_cost := data.personal_expense / data.personal_expense_rate
  let arrows_lost := data.shots_per_week * (1 - data.recovery_rate)
  total_cost / arrows_lost

/-- Theorem stating that the cost per arrow is $5.50 given the specific conditions -/
theorem archer_arrow_cost :
  let data : ArcherData := {
    shots_per_week := 800,
    recovery_rate := 1/5,
    personal_expense_rate := 3/10,
    personal_expense := 1056
  }
  cost_per_arrow data = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_archer_arrow_cost_l2085_208564


namespace NUMINAMATH_CALUDE_walkers_on_same_side_l2085_208552

/-- Represents a person walking around a regular pentagon -/
structure Walker where
  speed : ℝ
  startPosition : ℕ

/-- The time when two walkers start walking on the same side of a regular pentagon -/
def timeOnSameSide (perimeterLength : ℝ) (walker1 walker2 : Walker) : ℝ :=
  sorry

/-- Theorem stating the time when two specific walkers start on the same side of a regular pentagon -/
theorem walkers_on_same_side :
  let perimeterLength : ℝ := 2000
  let walker1 : Walker := { speed := 50, startPosition := 0 }
  let walker2 : Walker := { speed := 46, startPosition := 2 }
  timeOnSameSide perimeterLength walker1 walker2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_walkers_on_same_side_l2085_208552


namespace NUMINAMATH_CALUDE_peters_contribution_l2085_208526

/-- Given four friends pooling money for a purchase, prove Peter's contribution --/
theorem peters_contribution (john quincy andrew peter : ℝ) : 
  john > 0 ∧ 
  peter = 2 * john ∧ 
  quincy = peter + 20 ∧ 
  andrew = 1.15 * quincy ∧ 
  john + peter + quincy + andrew = 1211 →
  peter = 370.80 := by
  sorry

end NUMINAMATH_CALUDE_peters_contribution_l2085_208526


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2085_208593

theorem sqrt_sum_fractions : 
  Real.sqrt (4/25 + 9/49) = Real.sqrt 421 / 35 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2085_208593


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l2085_208566

theorem fibonacci_like_sequence
  (a : ℕ → ℕ)  -- Sequence of natural numbers
  (seq_length : ℕ)  -- Length of the sequence
  (h_length : seq_length = 10)  -- The sequence has 10 numbers
  (h_rec : ∀ n, 3 ≤ n → n ≤ seq_length → a n = a (n-1) + a (n-2))  -- Recurrence relation
  (h_seventh : a 7 = 42)  -- The seventh number is 42
  (h_ninth : a 9 = 110)  -- The ninth number is 110
  : a 4 = 10 :=  -- The fourth number is 10
by sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l2085_208566


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2085_208557

theorem chocolate_distribution (num_boxes : ℕ) (total_pieces : ℕ) (h1 : num_boxes = 6) (h2 : total_pieces = 3000) :
  total_pieces / num_boxes = 500 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2085_208557


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2085_208514

/-- The equation of the line passing through the intersection points of two circles -/
theorem intersection_line_equation (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) :
  c1_center = (-8, -6) →
  c2_center = (4, 5) →
  r1 = 10 →
  r2 = Real.sqrt 41 →
  ∃ (x y : ℝ), ((x - c1_center.1)^2 + (y - c1_center.2)^2 = r1^2) ∧
                ((x - c2_center.1)^2 + (y - c2_center.2)^2 = r2^2) ∧
                (x + y = -59/11) :=
by sorry


end NUMINAMATH_CALUDE_intersection_line_equation_l2085_208514


namespace NUMINAMATH_CALUDE_number_of_gharials_l2085_208546

/-- Represents the number of flies eaten per day by one frog -/
def flies_per_frog : ℕ := 30

/-- Represents the number of frogs eaten per day by one fish -/
def frogs_per_fish : ℕ := 8

/-- Represents the number of fish eaten per day by one gharial -/
def fish_per_gharial : ℕ := 15

/-- Represents the total number of flies eaten per day in the swamp -/
def total_flies_eaten : ℕ := 32400

/-- Proves that the number of gharials in the swamp is 9 -/
theorem number_of_gharials : ℕ := by
  sorry

end NUMINAMATH_CALUDE_number_of_gharials_l2085_208546


namespace NUMINAMATH_CALUDE_inverse_f_486_l2085_208562

def f (x : ℝ) : ℝ := sorry

theorem inverse_f_486 (h1 : f 5 = 2) (h2 : ∀ x, f (3 * x) = 3 * f x) :
  f 1215 = 486 := by sorry

end NUMINAMATH_CALUDE_inverse_f_486_l2085_208562


namespace NUMINAMATH_CALUDE_trip_cost_proof_l2085_208559

/-- Calculates the total cost of a trip for two people with a discount -/
def total_cost (original_price discount : ℕ) : ℕ :=
  2 * (original_price - discount)

/-- Proves that the total cost of the trip for two people is $266 -/
theorem trip_cost_proof (original_price discount : ℕ) 
  (h1 : original_price = 147) 
  (h2 : discount = 14) : 
  total_cost original_price discount = 266 := by
  sorry

#eval total_cost 147 14

end NUMINAMATH_CALUDE_trip_cost_proof_l2085_208559


namespace NUMINAMATH_CALUDE_sum_digits_M_times_2013_l2085_208517

/-- A number composed of n consecutive ones -/
def consecutive_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

/-- Theorem: The sum of digits of M × 2013 is 1200, where M is composed of 200 consecutive ones -/
theorem sum_digits_M_times_2013 :
  sum_of_digits (consecutive_ones 200 * 2013) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_M_times_2013_l2085_208517


namespace NUMINAMATH_CALUDE_largest_divisor_for_multiples_of_three_l2085_208578

def f (n : ℕ) : ℕ := n * (n + 2) * (n + 4) * (n + 6) * (n + 8)

theorem largest_divisor_for_multiples_of_three :
  ∃ (d : ℕ), d = 288 ∧
  (∀ (n : ℕ), 3 ∣ n → d ∣ f n) ∧
  (∀ (m : ℕ), m > d → ∃ (n : ℕ), 3 ∣ n ∧ ¬(m ∣ f n)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_for_multiples_of_three_l2085_208578


namespace NUMINAMATH_CALUDE_sin_squared_minus_two_sin_range_l2085_208586

theorem sin_squared_minus_two_sin_range :
  ∀ x : ℝ, -1 ≤ Real.sin x ^ 2 - 2 * Real.sin x ∧ Real.sin x ^ 2 - 2 * Real.sin x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_two_sin_range_l2085_208586


namespace NUMINAMATH_CALUDE_camera_and_lens_cost_l2085_208540

theorem camera_and_lens_cost
  (old_camera_cost : ℝ)
  (new_camera_percentage : ℝ)
  (lens_original_price : ℝ)
  (lens_discount : ℝ)
  (h1 : old_camera_cost = 4000)
  (h2 : new_camera_percentage = 1.3)
  (h3 : lens_original_price = 400)
  (h4 : lens_discount = 200) :
  old_camera_cost * new_camera_percentage + (lens_original_price - lens_discount) = 5400 :=
by sorry

end NUMINAMATH_CALUDE_camera_and_lens_cost_l2085_208540


namespace NUMINAMATH_CALUDE_factor_of_x4_plus_8_l2085_208594

theorem factor_of_x4_plus_8 (x : ℝ) : 
  (x^2 - 2*x + 4) * (x^2 + 2*x + 4) = x^4 + 8 := by
  sorry

end NUMINAMATH_CALUDE_factor_of_x4_plus_8_l2085_208594


namespace NUMINAMATH_CALUDE_new_cost_relation_l2085_208575

/-- Represents the manufacturing cost function -/
def cost (k t b : ℝ) : ℝ := k * (t * b) ^ 4

/-- Theorem: New cost after doubling batches and reducing time by 25% -/
theorem new_cost_relation (k t b : ℝ) (h_pos : t > 0 ∧ b > 0) :
  cost k (0.75 * t) (2 * b) = 25.62890625 * cost k t b := by
  sorry

#check new_cost_relation

end NUMINAMATH_CALUDE_new_cost_relation_l2085_208575


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2085_208580

-- Define the prices and quantities
def pasta_price : ℝ := 1.70
def pasta_quantity : ℝ := 3
def beef_price : ℝ := 8.20
def beef_quantity : ℝ := 0.5
def sauce_price : ℝ := 2.30
def sauce_quantity : ℝ := 3
def quesadillas_price : ℝ := 11.50
def discount_rate : ℝ := 0.10
def vat_rate : ℝ := 0.05

-- Define the total cost function
def total_cost : ℝ :=
  let pasta_cost := pasta_price * pasta_quantity
  let beef_cost := beef_price * beef_quantity
  let sauce_cost := sauce_price * sauce_quantity
  let discounted_sauce_cost := sauce_cost * (1 - discount_rate)
  let subtotal := pasta_cost + beef_cost + discounted_sauce_cost + quesadillas_price
  let vat := subtotal * vat_rate
  subtotal + vat

-- Theorem statement
theorem total_cost_is_correct : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ abs (total_cost - 28.26) < ε :=
sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2085_208580


namespace NUMINAMATH_CALUDE_greatest_circle_center_distance_l2085_208565

theorem greatest_circle_center_distance
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 18)
  (h_height : rectangle_height = 20)
  (h_diameter : circle_diameter = 8)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧
    ∀ (d' : ℝ), d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        (x₁ - circle_diameter / 2 ≥ 0) ∧
        (y₁ - circle_diameter / 2 ≥ 0) ∧
        (x₁ + circle_diameter / 2 ≤ rectangle_width) ∧
        (y₁ + circle_diameter / 2 ≤ rectangle_height) ∧
        (x₂ - circle_diameter / 2 ≥ 0) ∧
        (y₂ - circle_diameter / 2 ≥ 0) ∧
        (x₂ + circle_diameter / 2 ≤ rectangle_width) ∧
        (y₂ + circle_diameter / 2 ≤ rectangle_height) ∧
        d' = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_circle_center_distance_l2085_208565


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_m_geq_6_l2085_208545

-- Define the conditions p and q as functions
def p (x : ℝ) : Prop := (x - 1) / x ≤ 0
def q (x m : ℝ) : Prop := 4^x + 2^x - m ≤ 0

-- State the theorem
theorem sufficient_condition_implies_m_geq_6 :
  (∀ x m : ℝ, p x → q x m) → ∀ m : ℝ, m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_m_geq_6_l2085_208545


namespace NUMINAMATH_CALUDE_prime_sequence_ones_digit_l2085_208541

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) :
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > 5 →
  q = p + 8 →
  r = q + 8 →
  s = r + 8 →
  ones_digit p = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_sequence_ones_digit_l2085_208541


namespace NUMINAMATH_CALUDE_probability_seven_chairs_probability_n_chairs_l2085_208504

/-- The probability of three knights being seated at a round table with empty chairs on either side of each knight. -/
def knight_seating_probability (n : ℕ) : ℚ :=
  if n = 7 then 1 / 35
  else if n ≥ 6 then (n - 4) * (n - 5) / ((n - 1) * (n - 2))
  else 0

/-- Theorem stating the probability for 7 chairs -/
theorem probability_seven_chairs :
  knight_seating_probability 7 = 1 / 35 := by sorry

/-- Theorem stating the probability for n chairs (n ≥ 6) -/
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) :
  knight_seating_probability n = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := by sorry

end NUMINAMATH_CALUDE_probability_seven_chairs_probability_n_chairs_l2085_208504


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2085_208533

/-- Arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ S 4 = 20 ∧
  ∀ n : ℕ, S n = n * (a 1) + (n * (n - 1)) / 2 * (a 2 - a 1)

/-- Theorem stating the common difference and S_6 for the given arithmetic sequence -/
theorem arithmetic_sequence_properties (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h : arithmetic_sequence a S) : (a 2 - a 1 = 3) ∧ (S 6 = 48) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2085_208533


namespace NUMINAMATH_CALUDE_tower_height_calculation_l2085_208581

-- Define the tower and measurement points
structure Tower :=
  (height : ℝ)

structure MeasurementPoints :=
  (distanceAD : ℝ)
  (angleA : ℝ)
  (angleD : ℝ)

-- Define the theorem
theorem tower_height_calculation (t : Tower) (m : MeasurementPoints) 
  (h_distanceAD : m.distanceAD = 129)
  (h_angleA : m.angleA = 45)
  (h_angleD : m.angleD = 60) :
  t.height = 305 := by
  sorry


end NUMINAMATH_CALUDE_tower_height_calculation_l2085_208581


namespace NUMINAMATH_CALUDE_wrapping_paper_area_formula_l2085_208573

/-- The area of wrapping paper required for a box -/
def wrapping_paper_area (w : ℝ) (h : ℝ) : ℝ :=
  (4 * w + h) * (2 * w + h)

/-- Theorem: The area of the wrapping paper for a box with width w, length 2w, and height h -/
theorem wrapping_paper_area_formula (w : ℝ) (h : ℝ) :
  wrapping_paper_area w h = 8 * w^2 + 6 * w * h + h^2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_formula_l2085_208573


namespace NUMINAMATH_CALUDE_unique_integers_square_sum_l2085_208585

theorem unique_integers_square_sum : ∃! (A B : ℕ), 
  A ≤ 9 ∧ B ≤ 9 ∧ (1001 * A + 110 * B)^2 = 57108249 ∧ 10 * A + B = 75 := by
  sorry

end NUMINAMATH_CALUDE_unique_integers_square_sum_l2085_208585


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2085_208532

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m + 1, -3)
  let b : ℝ × ℝ := (2, 3)
  parallel a b → m = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2085_208532


namespace NUMINAMATH_CALUDE_tree_watering_boys_l2085_208520

theorem tree_watering_boys (total_trees : ℕ) (trees_per_boy : ℕ) (h1 : total_trees = 29) (h2 : trees_per_boy = 3) :
  ∃ (num_boys : ℕ), num_boys * trees_per_boy ≥ total_trees ∧ (num_boys - 1) * trees_per_boy < total_trees ∧ num_boys = 10 :=
sorry

end NUMINAMATH_CALUDE_tree_watering_boys_l2085_208520


namespace NUMINAMATH_CALUDE_three_distinct_zeros_l2085_208512

-- Define the piecewise function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp (-x) - 1/2
  else x^3 - 3*m*x - 2

-- Theorem statement
theorem three_distinct_zeros (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f m x = 0 ∧ f m y = 0 ∧ f m z = 0) ↔ m > 1 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_zeros_l2085_208512


namespace NUMINAMATH_CALUDE_paving_cost_l2085_208535

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 1000) :
  length * width * rate = 20625 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2085_208535


namespace NUMINAMATH_CALUDE_not_parallel_if_intersect_and_contain_perpendicular_if_parallel_and_perpendicular_l2085_208522

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the basic relations
variable (intersects : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Theorem 1
theorem not_parallel_if_intersect_and_contain 
  (a b : Line) (α : Plane) (P : Point) :
  intersects a α ∧ contains α b → ¬ parallel a b := by sorry

-- Theorem 2
theorem perpendicular_if_parallel_and_perpendicular 
  (a b : Line) (α : Plane) :
  parallel a b ∧ perpendicular b α → perpendicular a α := by sorry

end NUMINAMATH_CALUDE_not_parallel_if_intersect_and_contain_perpendicular_if_parallel_and_perpendicular_l2085_208522


namespace NUMINAMATH_CALUDE_sum_is_square_l2085_208544

theorem sum_is_square (x y z : ℕ+) 
  (h1 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / z)
  (h2 : Nat.gcd (Nat.gcd x.val y.val) z.val = 1) :
  ∃ n : ℕ, x.val + y.val = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_square_l2085_208544


namespace NUMINAMATH_CALUDE_pauls_erasers_l2085_208515

/-- Represents the number of crayons and erasers Paul has --/
structure PaulsSupplies where
  initialCrayons : ℕ
  finalCrayons : ℕ
  erasers : ℕ

/-- Defines the conditions of Paul's supplies --/
def validSupplies (s : PaulsSupplies) : Prop :=
  s.initialCrayons = 601 ∧
  s.finalCrayons = 336 ∧
  s.erasers = s.finalCrayons + 70

/-- Theorem stating the number of erasers Paul got for his birthday --/
theorem pauls_erasers (s : PaulsSupplies) (h : validSupplies s) : s.erasers = 406 := by
  sorry

end NUMINAMATH_CALUDE_pauls_erasers_l2085_208515


namespace NUMINAMATH_CALUDE_sequence_equality_l2085_208599

-- Define the sequence a_n
def a (n : ℕ) (x : ℝ) : ℝ := 1 + x^(n+1) + x^(n+2)

-- State the theorem
theorem sequence_equality (x : ℝ) (h : (a 2 x)^2 = (a 1 x) * (a 3 x)) :
  ∀ n ≥ 3, (a n x)^2 = (a (n-1) x) * (a (n+1) x) :=
by sorry

end NUMINAMATH_CALUDE_sequence_equality_l2085_208599


namespace NUMINAMATH_CALUDE_unique_integer_sqrt_l2085_208574

theorem unique_integer_sqrt (x y : ℕ) : x = 25530 ∧ y = 29464 ↔ 
  ∃ (z : ℕ), z > 0 ∧ z * z = x * x + y * y ∧
  ∀ (a b : ℕ), (a = 37615 ∧ b = 26855) ∨ 
               (a = 15123 ∧ b = 32477) ∨ 
               (a = 28326 ∧ b = 28614) ∨ 
               (a = 22536 ∧ b = 27462) →
               ¬∃ (w : ℕ), w > 0 ∧ w * w = a * a + b * b :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_sqrt_l2085_208574


namespace NUMINAMATH_CALUDE_percentage_students_taking_music_l2085_208595

/-- The percentage of students taking music in a school with various electives -/
theorem percentage_students_taking_music
  (total_students : ℕ)
  (dance_percent : ℝ)
  (art_percent : ℝ)
  (drama_percent : ℝ)
  (sports_percent : ℝ)
  (photography_percent : ℝ)
  (h_total : total_students = 3000)
  (h_dance : dance_percent = 12.5)
  (h_art : art_percent = 22)
  (h_drama : drama_percent = 13.5)
  (h_sports : sports_percent = 15)
  (h_photo : photography_percent = 8) :
  100 - (dance_percent + art_percent + drama_percent + sports_percent + photography_percent) = 29 := by
  sorry

end NUMINAMATH_CALUDE_percentage_students_taking_music_l2085_208595


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_equals_8_l2085_208551

theorem mean_equality_implies_x_equals_8 :
  let mean1 := (8 + 10 + 24) / 3
  let mean2 := (16 + x + 18) / 3
  mean1 = mean2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_equals_8_l2085_208551


namespace NUMINAMATH_CALUDE_straw_length_theorem_l2085_208556

/-- The total length of overlapping straws -/
def total_length (straw_length : ℕ) (overlap : ℕ) (num_straws : ℕ) : ℕ :=
  straw_length + (straw_length - overlap) * (num_straws - 1)

/-- Theorem: The total length of 30 straws is 576 cm -/
theorem straw_length_theorem :
  total_length 25 6 30 = 576 := by
  sorry

end NUMINAMATH_CALUDE_straw_length_theorem_l2085_208556


namespace NUMINAMATH_CALUDE_remainder_problem_l2085_208596

theorem remainder_problem (x : ℤ) : x % 82 = 5 → (x + 17) % 41 = 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2085_208596


namespace NUMINAMATH_CALUDE_sequence_increasing_and_divergent_l2085_208538

open Real MeasureTheory Interval Set

noncomputable section

variables (a b : ℝ) (f g : ℝ → ℝ)

def I (n : ℕ) := ∫ x in a..b, (f x)^(n+1) / (g x)^n

theorem sequence_increasing_and_divergent
  (hab : a < b)
  (hf : ContinuousOn f (Icc a b))
  (hg : ContinuousOn g (Icc a b))
  (hfg_pos : ∀ x ∈ Icc a b, 0 < f x ∧ 0 < g x)
  (hfg_int : ∫ x in a..b, f x = ∫ x in a..b, g x)
  (hfg_neq : f ≠ g) :
  (∀ n : ℕ, I a b f g n < I a b f g (n + 1)) ∧
  (∀ M : ℝ, ∃ N : ℕ, ∀ n ≥ N, M < I a b f g n) :=
sorry

end NUMINAMATH_CALUDE_sequence_increasing_and_divergent_l2085_208538


namespace NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_nine_min_value_achieved_l2085_208576

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → 4*a + b ≤ 4*x + y :=
by sorry

theorem min_value_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  4*a + b ≥ 9 :=
by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 ∧ 4*x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_nine_min_value_achieved_l2085_208576


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2085_208537

theorem saree_price_calculation (P : ℝ) : 
  (P * (1 - 0.20) * (1 - 0.15) = 272) → P = 400 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2085_208537


namespace NUMINAMATH_CALUDE_expression_equality_l2085_208583

theorem expression_equality : (2^5 * 9^2) / (8^2 * 3^5) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2085_208583


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l2085_208530

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem: The wet surface area of a cistern with given dimensions is 134 square meters -/
theorem cistern_wet_surface_area :
  wetSurfaceArea 10 8 1.5 = 134 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l2085_208530


namespace NUMINAMATH_CALUDE_fifth_group_number_l2085_208539

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_elements : ℕ
  sample_size : ℕ
  first_drawn : ℕ
  h_positive : 0 < total_elements
  h_sample_size : 0 < sample_size
  h_first_drawn : first_drawn ≤ total_elements
  h_divisible : total_elements % sample_size = 0

/-- The number drawn in a specific group -/
def number_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_drawn + (group - 1) * (s.total_elements / s.sample_size)

/-- Theorem stating the number drawn in the fifth group -/
theorem fifth_group_number (s : SystematicSampling)
  (h1 : s.total_elements = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.first_drawn = 3) :
  number_in_group s 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fifth_group_number_l2085_208539


namespace NUMINAMATH_CALUDE_bryans_mineral_samples_per_shelf_l2085_208577

/-- Given Bryan's mineral collection setup, prove the number of samples per shelf. -/
theorem bryans_mineral_samples_per_shelf :
  let total_samples : ℕ := 455
  let total_shelves : ℕ := 7
  let samples_per_shelf : ℕ := total_samples / total_shelves
  samples_per_shelf = 65 := by
  sorry

end NUMINAMATH_CALUDE_bryans_mineral_samples_per_shelf_l2085_208577


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2085_208569

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence)
  (h1 : seq.a 1 + seq.a 3 = 8)
  (h2 : seq.a 4 ^ 2 = seq.a 2 * seq.a 9) :
  seq.a 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2085_208569


namespace NUMINAMATH_CALUDE_even_function_implies_a_value_l2085_208508

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (2 * x + 3 * a)

-- State the theorem
theorem even_function_implies_a_value :
  (∀ x : ℝ, f a x = f a (-x)) → a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_value_l2085_208508


namespace NUMINAMATH_CALUDE_ice_cream_earnings_theorem_l2085_208506

def ice_cream_earnings (daily_increase : ℕ) : List ℕ :=
  [10, 10 + daily_increase, 10 + 2 * daily_increase, 10 + 3 * daily_increase, 10 + 4 * daily_increase]

theorem ice_cream_earnings_theorem (daily_increase : ℕ) :
  (List.sum (ice_cream_earnings daily_increase) = 90) → daily_increase = 4 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_earnings_theorem_l2085_208506


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_primes_l2085_208568

def first_17_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

def is_divisible_by_all_except_two_consecutive (n : Nat) (primes : List Nat) (i : Nat) : Prop :=
  ∀ (p : Nat), p ∈ primes → (p ≠ primes[i]! ∧ p ≠ primes[i+1]!) → n % p = 0

theorem smallest_number_divisible_by_primes : ∃ (n : Nat),
  is_divisible_by_all_except_two_consecutive n first_17_primes 15 ∧
  ∀ (m : Nat), m < n → ¬is_divisible_by_all_except_two_consecutive m first_17_primes 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_primes_l2085_208568


namespace NUMINAMATH_CALUDE_shoe_difference_l2085_208534

def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

theorem shoe_difference : anthony_shoes - jim_shoes = 2 := by
  sorry

end NUMINAMATH_CALUDE_shoe_difference_l2085_208534


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2085_208553

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / (perimeter ^ 2) = Real.sqrt 3 / 36 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2085_208553


namespace NUMINAMATH_CALUDE_missing_digit_is_4_l2085_208524

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem missing_digit_is_4 (n : ℕ) (h1 : n ≥ 35204 ∧ n < 35304) 
  (h2 : is_divisible_by_9 n) : 
  ∃ (d : ℕ), d < 10 ∧ n = 35204 + d * 10 ∧ d = 4 := by
  sorry

#check missing_digit_is_4

end NUMINAMATH_CALUDE_missing_digit_is_4_l2085_208524


namespace NUMINAMATH_CALUDE_point_transformation_l2085_208555

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


end NUMINAMATH_CALUDE_point_transformation_l2085_208555


namespace NUMINAMATH_CALUDE_max_value_problem_l2085_208550

theorem max_value_problem (m n k : ℕ) (a b c : ℕ → ℕ) :
  (∀ i ∈ Finset.range m, a i % 3 = 1) →
  (∀ i ∈ Finset.range n, b i % 3 = 2) →
  (∀ i ∈ Finset.range k, c i % 3 = 0) →
  (∀ i j, i ≠ j → (a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j ∧ 
                   a i ≠ b j ∧ a i ≠ c j ∧ b i ≠ c j)) →
  (Finset.sum (Finset.range m) a + Finset.sum (Finset.range n) b + 
   Finset.sum (Finset.range k) c = 2007) →
  4 * m + 3 * n + 5 * k ≤ 256 := by
sorry

end NUMINAMATH_CALUDE_max_value_problem_l2085_208550


namespace NUMINAMATH_CALUDE_jesse_carpet_amount_l2085_208513

/-- The amount of carpet Jesse already has -/
def carpet_already_has (room_length room_width additional_carpet_needed : ℝ) : ℝ :=
  room_length * room_width - additional_carpet_needed

/-- Theorem: Jesse already has 16 square feet of carpet -/
theorem jesse_carpet_amount :
  carpet_already_has 11 15 149 = 16 := by
  sorry

end NUMINAMATH_CALUDE_jesse_carpet_amount_l2085_208513


namespace NUMINAMATH_CALUDE_flu_outbreak_theorem_l2085_208500

/-- Represents the state of a dwarf --/
inductive DwarfState
| Sick
| Healthy
| Immune

/-- Represents the population of dwarves --/
structure DwarfPopulation where
  sick : Set Nat
  healthy : Set Nat
  immune : Set Nat

/-- Represents the flu outbreak --/
structure FluOutbreak where
  initialVaccinated : Bool
  population : Nat → DwarfPopulation

/-- The flu lasts indefinitely if some dwarves are initially vaccinated --/
def fluLastsIndefinitely (outbreak : FluOutbreak) : Prop :=
  outbreak.initialVaccinated ∧
  ∀ n : Nat, ∃ i : Nat, i ∈ (outbreak.population n).sick

/-- The flu eventually ends if no dwarves are initially immune --/
def fluEventuallyEnds (outbreak : FluOutbreak) : Prop :=
  ¬outbreak.initialVaccinated ∧
  ∃ n : Nat, ∀ i : Nat, i ∉ (outbreak.population n).sick

theorem flu_outbreak_theorem (outbreak : FluOutbreak) :
  (outbreak.initialVaccinated → fluLastsIndefinitely outbreak) ∧
  (¬outbreak.initialVaccinated → fluEventuallyEnds outbreak) := by
  sorry


end NUMINAMATH_CALUDE_flu_outbreak_theorem_l2085_208500


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l2085_208531

theorem greatest_three_digit_divisible_by_3_6_5 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 ∣ n ∧ 6 ∣ n ∧ 5 ∣ n → n ≤ 990 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l2085_208531


namespace NUMINAMATH_CALUDE_bens_initial_money_l2085_208542

theorem bens_initial_money (initial_amount : ℕ) : 
  (((initial_amount - 600) + 800) - 1200 = 1000) → 
  initial_amount = 2000 := by
  sorry

end NUMINAMATH_CALUDE_bens_initial_money_l2085_208542


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l2085_208591

def biased_coin_prob : ℚ := 3/4
def die_sides : ℕ := 6

theorem coin_and_die_probability :
  let heads_prob := biased_coin_prob
  let three_prob := 1 / die_sides
  heads_prob * three_prob = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l2085_208591


namespace NUMINAMATH_CALUDE_not_proportional_l2085_208527

/-- A function f is directly proportional to x if there exists a constant k such that f x = k * x for all x -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- A function f is inversely proportional to x if there exists a constant k such that f x = k / x for all non-zero x -/
def InverselyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function defined by the equation x^2 + y = 1 -/
def f (x : ℝ) : ℝ := 1 - x^2

theorem not_proportional : ¬(DirectlyProportional f) ∧ ¬(InverselyProportional f) := by
  sorry

end NUMINAMATH_CALUDE_not_proportional_l2085_208527


namespace NUMINAMATH_CALUDE_partnership_capital_fraction_l2085_208543

theorem partnership_capital_fraction :
  ∀ (T : ℚ) (x : ℚ),
    x > 0 →
    T > 0 →
    x * T + (1/4) * T + (1/5) * T + ((11/20 - x) * T) = T →
    805 / 2415 = x →
    x = 161 / 483 := by
  sorry

end NUMINAMATH_CALUDE_partnership_capital_fraction_l2085_208543


namespace NUMINAMATH_CALUDE_f_min_max_l2085_208590

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the domain
def domain : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem f_min_max :
  (∃ x ∈ domain, f x = 0) ∧
  (∀ x ∈ domain, f x ≥ 0) ∧
  (∃ x ∈ domain, f x = 9) ∧
  (∀ x ∈ domain, f x ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_f_min_max_l2085_208590


namespace NUMINAMATH_CALUDE_prism_volume_l2085_208511

/-- The volume of a right rectangular prism with face areas 18, 12, and 8 square inches -/
theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 18) 
  (h2 : y * z = 12) 
  (h3 : x * z = 8) : 
  x * y * z = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2085_208511


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2085_208563

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 500) 
  (h2 : bridge_length = 350) 
  (h3 : crossing_time = 60) : 
  ∃ (speed : ℝ), abs (speed - 14.1667) < 0.0001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2085_208563


namespace NUMINAMATH_CALUDE_girls_from_clay_l2085_208582

/-- Represents a school in the science camp --/
inductive School
| Jonas
| Clay
| Maple

/-- Represents the gender of a student --/
inductive Gender
| Boy
| Girl

/-- Represents the distribution of students in the science camp --/
structure CampDistribution where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  jonas_students : ℕ
  clay_students : ℕ
  maple_students : ℕ
  jonas_boys : ℕ

/-- The actual distribution of students in the science camp --/
def camp_distribution : CampDistribution :=
  { total_students := 120
  , total_boys := 70
  , total_girls := 50
  , jonas_students := 50
  , clay_students := 40
  , maple_students := 30
  , jonas_boys := 30
  }

/-- Theorem stating that the number of girls from Clay Middle School is 10 --/
theorem girls_from_clay (d : CampDistribution) (h : d = camp_distribution) :
  ∃ (clay_girls : ℕ), clay_girls = 10 ∧
  clay_girls = d.clay_students - (d.total_boys - d.jonas_boys) :=
by sorry

end NUMINAMATH_CALUDE_girls_from_clay_l2085_208582


namespace NUMINAMATH_CALUDE_total_turtles_count_l2085_208560

/-- Represents the total number of turtles in the lake -/
def total_turtles : ℕ := sorry

/-- Represents the number of striped male adult common turtles -/
def striped_male_adult_common : ℕ := 70

/-- Percentage of common turtles in the lake -/
def common_percentage : ℚ := 1/2

/-- Percentage of female common turtles -/
def common_female_percentage : ℚ := 3/5

/-- Percentage of striped male common turtles among male common turtles -/
def striped_male_common_percentage : ℚ := 1/4

/-- Percentage of adult striped male common turtles among striped male common turtles -/
def adult_striped_male_common_percentage : ℚ := 4/5

theorem total_turtles_count : total_turtles = 1760 := by sorry

end NUMINAMATH_CALUDE_total_turtles_count_l2085_208560


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l2085_208521

/-- Represents scientific notation as a pair of a coefficient and an exponent -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  let billion : ℝ := 1000000000
  let amount : ℝ := 10.58 * billion
  toScientificNotation amount = ScientificNotation.mk 1.058 10 (by norm_num) :=
by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l2085_208521


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2085_208509

/-- Given a rectangle with width 10 meters and area 150 square meters,
    if its length is increased such that the new area is 4/3 times the original area,
    then the new perimeter of the rectangle is 60 meters. -/
theorem rectangle_perimeter (width : ℝ) (original_area : ℝ) (new_area : ℝ) :
  width = 10 →
  original_area = 150 →
  new_area = (4/3) * original_area →
  2 * (new_area / width + width) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2085_208509


namespace NUMINAMATH_CALUDE_concentric_circle_through_point_l2085_208503

/-- Given a circle with equation x^2 + y^2 - 4x + 6y + 3 = 0,
    prove that (x - 2)^2 + (y + 3)^2 = 25 represents a circle
    that is concentric with the given circle and passes through (-1, 1) -/
theorem concentric_circle_through_point
  (h : ∀ x y : ℝ, x^2 + y^2 - 4*x + 6*y + 3 = 0 → (x - 2)^2 + (y + 3)^2 = 10) :
  (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 25 →
    ∃ k : ℝ, k > 0 ∧ (x - 2)^2 + (y + 3)^2 = k * ((x - 2)^2 + (y + 3)^2 - 10)) ∧
  ((-1 - 2)^2 + (1 + 3)^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_concentric_circle_through_point_l2085_208503


namespace NUMINAMATH_CALUDE_min_value_x_l2085_208523

theorem min_value_x (x : ℝ) (h1 : x > 0) (h2 : 2 * Real.log x ≥ Real.log 8 + Real.log x) (h3 : x ≤ 32) :
  x ≥ 8 ∧ ∀ y : ℝ, y > 0 → 2 * Real.log y ≥ Real.log 8 + Real.log y → y ≤ 32 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_l2085_208523


namespace NUMINAMATH_CALUDE_expression_simplification_l2085_208597

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 2) :
  (1 + 1 / (m - 2)) / ((m^2 - m) / (m - 2)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2085_208597


namespace NUMINAMATH_CALUDE_simplify_expression_l2085_208588

theorem simplify_expression (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 2*b) (h3 : a ≠ b) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2085_208588


namespace NUMINAMATH_CALUDE_expression_value_when_b_is_3_l2085_208589

theorem expression_value_when_b_is_3 :
  let b : ℝ := 3
  let expr := (3 * b⁻¹ + b⁻¹ / 3) / b^2
  expr = 10 / 81 := by sorry

end NUMINAMATH_CALUDE_expression_value_when_b_is_3_l2085_208589


namespace NUMINAMATH_CALUDE_summer_work_hours_adjustment_l2085_208505

theorem summer_work_hours_adjustment (
  original_hours_per_week : ℝ)
  (original_weeks : ℕ)
  (total_earnings : ℝ)
  (lost_weeks : ℕ)
  (h1 : original_hours_per_week = 20)
  (h2 : original_weeks = 12)
  (h3 : total_earnings = 3000)
  (h4 : lost_weeks = 2)
  (h5 : total_earnings = original_hours_per_week * original_weeks * (total_earnings / (original_hours_per_week * original_weeks)))
  : ∃ new_hours_per_week : ℝ,
    new_hours_per_week * (original_weeks - lost_weeks) * (total_earnings / (original_hours_per_week * original_weeks)) = total_earnings ∧
    new_hours_per_week = 24 :=
by sorry

end NUMINAMATH_CALUDE_summer_work_hours_adjustment_l2085_208505


namespace NUMINAMATH_CALUDE_system_solution_l2085_208525

theorem system_solution (x y : ℝ) : 
  (x + 2*y = 6 ∧ 5*x - 4*y = 2) ↔ (x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2085_208525


namespace NUMINAMATH_CALUDE_original_number_proof_l2085_208529

theorem original_number_proof (x : ℝ) : x * 1.2 = 288 → x = 240 := by sorry

end NUMINAMATH_CALUDE_original_number_proof_l2085_208529


namespace NUMINAMATH_CALUDE_jimin_calculation_l2085_208518

theorem jimin_calculation (x : ℤ) : x + 20 = 60 → 34 - x = -6 := by
  sorry

end NUMINAMATH_CALUDE_jimin_calculation_l2085_208518


namespace NUMINAMATH_CALUDE_axis_of_symmetry_shifted_sine_l2085_208510

theorem axis_of_symmetry_shifted_sine (k : ℤ) :
  let f : ℝ → ℝ := λ x => 2 * Real.sin (2 * (x + π / 12))
  let axis : ℝ := k * π / 2 + π / 6
  ∀ x : ℝ, f (axis - x) = f (axis + x) :=
by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_shifted_sine_l2085_208510


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2085_208501

theorem sqrt_sum_fractions : 
  Real.sqrt ((1 : ℝ) / 25 + (1 : ℝ) / 36) = Real.sqrt 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2085_208501


namespace NUMINAMATH_CALUDE_bullet_problem_l2085_208528

theorem bullet_problem (n : ℕ) (h1 : n > 4) :
  (5 * (n - 4) = n) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_bullet_problem_l2085_208528


namespace NUMINAMATH_CALUDE_solve_equation_l2085_208549

theorem solve_equation : ∃ x : ℝ, 25 - 5 = 3 + x - 4 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2085_208549


namespace NUMINAMATH_CALUDE_certain_number_equation_l2085_208587

theorem certain_number_equation : ∃ x : ℚ, (40 * x + (12 + 8) * 3 / 5 = 1212) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2085_208587


namespace NUMINAMATH_CALUDE_gcd_150_450_l2085_208570

theorem gcd_150_450 : Nat.gcd 150 450 = 150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_150_450_l2085_208570


namespace NUMINAMATH_CALUDE_hot_chocolate_servings_l2085_208554

/-- Represents the recipe requirements for 6 servings --/
structure Recipe :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)
  (vanilla : ℚ)

/-- Represents the available ingredients --/
structure Available :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)
  (vanilla : ℚ)

/-- Calculates the number of servings possible for a given ingredient --/
def servings_for_ingredient (required : ℚ) (available : ℚ) : ℚ :=
  (available / required) * 6

/-- Finds the minimum number of servings possible across all ingredients --/
def max_servings (recipe : Recipe) (available : Available) : ℚ :=
  min
    (servings_for_ingredient recipe.chocolate available.chocolate)
    (min
      (servings_for_ingredient recipe.sugar available.sugar)
      (min
        (servings_for_ingredient recipe.milk available.milk)
        (servings_for_ingredient recipe.vanilla available.vanilla)))

theorem hot_chocolate_servings
  (recipe : Recipe)
  (available : Available)
  (h_recipe : recipe = { chocolate := 3, sugar := 1/2, milk := 6, vanilla := 3/2 })
  (h_available : available = { chocolate := 8, sugar := 3, milk := 15, vanilla := 5 }) :
  max_servings recipe available = 15 := by
  sorry

end NUMINAMATH_CALUDE_hot_chocolate_servings_l2085_208554


namespace NUMINAMATH_CALUDE_heart_then_club_probability_l2085_208592

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def numHearts : ℕ := 13

/-- Number of clubs in a standard deck -/
def numClubs : ℕ := 13

/-- Probability of drawing a heart followed by a club from a standard deck -/
def probHeartThenClub : ℚ := numHearts / standardDeck * numClubs / (standardDeck - 1)

theorem heart_then_club_probability :
  probHeartThenClub = 13 / 204 := by sorry

end NUMINAMATH_CALUDE_heart_then_club_probability_l2085_208592


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l2085_208584

theorem quadratic_form_minimum (x y : ℝ) :
  3 * x^2 + 4 * x * y + y^2 - 6 * x + 2 * y + 9 ≥ 9/4 ∧
  (3 * x^2 + 4 * x * y + y^2 - 6 * x + 2 * y + 9 = 9/4 ↔ x = 3/2 ∧ y = -3/4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l2085_208584


namespace NUMINAMATH_CALUDE_polynomial_sum_l2085_208507

/-- Given polynomial f -/
def f (a b c : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Given polynomial g -/
def g (a b c : ℤ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + a

/-- The main theorem -/
theorem polynomial_sum (a b c : ℤ) : c ≠ 0 →
  f a b c 1 = 0 →
  (∀ x : ℝ, g a b c x = 0 ↔ ∃ y : ℝ, f a b c y = 0 ∧ x = y^2) →
  a^2013 + b^2013 + c^2013 = -1 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_sum_l2085_208507


namespace NUMINAMATH_CALUDE_third_defendant_guilty_l2085_208558

-- Define the set of defendants
inductive Defendant : Type
  | A
  | B
  | C

-- Define the accusation function
def accuses : Defendant → Defendant → Prop := sorry

-- Define the truth-telling property
def tells_truth (d : Defendant) : Prop := sorry

-- Define the guilt property
def is_guilty (d : Defendant) : Prop := sorry

-- Define the condition that each defendant accuses one of the other two
axiom each_accuses_one : ∀ d₁ d₂ d₃ : Defendant, d₁ ≠ d₂ → d₁ ≠ d₃ → d₂ ≠ d₃ → 
  (accuses d₁ d₂ ∨ accuses d₁ d₃) ∧ (accuses d₂ d₁ ∨ accuses d₂ d₃) ∧ (accuses d₃ d₁ ∨ accuses d₃ d₂)

-- Define the condition that the first defendant (A) is the only one telling the truth
axiom A_tells_truth : tells_truth Defendant.A ∧ ¬tells_truth Defendant.B ∧ ¬tells_truth Defendant.C

-- Define the condition that if accusations were changed, B would be the only one telling the truth
axiom if_changed_B_tells_truth : 
  ∀ d₁ d₂ d₃ : Defendant, d₁ ≠ d₂ → d₁ ≠ d₃ → d₂ ≠ d₃ → 
  (accuses d₁ d₂ → accuses d₁ d₃) → (accuses d₂ d₁ → accuses d₂ d₃) → (accuses d₃ d₁ → accuses d₃ d₂) →
  tells_truth Defendant.B ∧ ¬tells_truth Defendant.A ∧ ¬tells_truth Defendant.C

-- Theorem: Given the conditions, the third defendant (C) is guilty
theorem third_defendant_guilty : is_guilty Defendant.C := by
  sorry

end NUMINAMATH_CALUDE_third_defendant_guilty_l2085_208558


namespace NUMINAMATH_CALUDE_max_candies_eaten_is_27_l2085_208547

/-- Represents a box of candies with a label -/
structure CandyBox where
  label : Nat
  candies : Nat

/-- Represents the state of all candy boxes -/
def GameState := List CandyBox

/-- Initializes the game state with three boxes -/
def initialState : GameState :=
  [{ label := 4, candies := 10 }, { label := 7, candies := 10 }, { label := 10, candies := 10 }]

/-- Performs one operation on the game state -/
def performOperation (state : GameState) (boxIndex : Nat) : Option GameState :=
  sorry

/-- Calculates the total number of candies eaten after a sequence of operations -/
def candiesEaten (operations : List Nat) : Nat :=
  sorry

/-- The maximum number of candies that can be eaten -/
def maxCandiesEaten : Nat :=
  sorry

/-- Theorem stating the maximum number of candies that can be eaten is 27 -/
theorem max_candies_eaten_is_27 : maxCandiesEaten = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_is_27_l2085_208547


namespace NUMINAMATH_CALUDE_negative_three_point_fourteen_greater_than_negative_pi_l2085_208579

theorem negative_three_point_fourteen_greater_than_negative_pi : -3.14 > -π := by
  sorry

end NUMINAMATH_CALUDE_negative_three_point_fourteen_greater_than_negative_pi_l2085_208579
