import Mathlib

namespace NUMINAMATH_CALUDE_nails_in_toolshed_l3615_361507

theorem nails_in_toolshed (initial_nails : ℕ) (nails_to_buy : ℕ) (total_nails : ℕ) :
  initial_nails = 247 →
  nails_to_buy = 109 →
  total_nails = 500 →
  total_nails = initial_nails + nails_to_buy + (total_nails - initial_nails - nails_to_buy) →
  total_nails - initial_nails - nails_to_buy = 144 :=
by sorry

end NUMINAMATH_CALUDE_nails_in_toolshed_l3615_361507


namespace NUMINAMATH_CALUDE_log_inequality_l3615_361583

theorem log_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  Real.log (Real.sqrt (x₁ * x₂)) = (Real.log x₁ + Real.log x₂) / 2 ∧
  Real.log (Real.sqrt (x₁ * x₂)) < Real.log ((x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3615_361583


namespace NUMINAMATH_CALUDE_bill_calculation_l3615_361599

theorem bill_calculation (a b c : ℝ) 
  (h1 : a - (b - c) = 11) 
  (h2 : a - b - c = 3) : 
  a - b = 7 := by
sorry

end NUMINAMATH_CALUDE_bill_calculation_l3615_361599


namespace NUMINAMATH_CALUDE_bank_account_final_balance_l3615_361551

-- Define the initial balance and transactions
def initial_balance : ℚ := 500
def first_withdrawal : ℚ := 200
def second_withdrawal_ratio : ℚ := 1/3
def first_deposit_ratio : ℚ := 1/5
def second_deposit_ratio : ℚ := 3/7

-- Define the theorem
theorem bank_account_final_balance :
  let balance_after_first_withdrawal := initial_balance - first_withdrawal
  let balance_after_second_withdrawal := balance_after_first_withdrawal * (1 - second_withdrawal_ratio)
  let balance_after_first_deposit := balance_after_second_withdrawal * (1 + first_deposit_ratio)
  let final_balance := balance_after_first_deposit / (1 - second_deposit_ratio)
  final_balance = 420 := by
  sorry

end NUMINAMATH_CALUDE_bank_account_final_balance_l3615_361551


namespace NUMINAMATH_CALUDE_ferry_distance_ratio_l3615_361535

/-- Represents a ferry with speed and travel time -/
structure Ferry where
  speed : ℝ
  time : ℝ

/-- The problem setup -/
def ferryProblem : Prop :=
  ∃ (P Q : Ferry),
    P.speed = 8 ∧
    P.time = 2 ∧
    Q.speed = P.speed + 4 ∧
    Q.time = P.time + 2 ∧
    Q.speed * Q.time / (P.speed * P.time) = 3

/-- The theorem to prove -/
theorem ferry_distance_ratio :
  ferryProblem := by sorry

end NUMINAMATH_CALUDE_ferry_distance_ratio_l3615_361535


namespace NUMINAMATH_CALUDE_arrangement_count_l3615_361586

def valid_arrangements (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => (n.choose k) ^ 3)

theorem arrangement_count :
  valid_arrangements 4 =
    (Finset.sum (Finset.range 5) (fun k =>
      (Nat.choose 4 k) ^ 3)) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3615_361586


namespace NUMINAMATH_CALUDE_irrational_equality_l3615_361539

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem irrational_equality (α β : ℝ) (h_irrational_α : Irrational α) (h_irrational_β : Irrational β) 
  (h_equality : ∀ x : ℝ, x > 0 → floor (α * floor (β * x)) = floor (β * floor (α * x))) :
  α = β :=
sorry

end NUMINAMATH_CALUDE_irrational_equality_l3615_361539


namespace NUMINAMATH_CALUDE_integer_roots_quadratic_l3615_361545

theorem integer_roots_quadratic (n : ℤ) : 
  (∃ x y : ℤ, x^2 + (n+1)*x + 2*n - 1 = 0 ∧ y^2 + (n+1)*y + 2*n - 1 = 0) → 
  (n = 1 ∨ n = 5) := by
sorry

end NUMINAMATH_CALUDE_integer_roots_quadratic_l3615_361545


namespace NUMINAMATH_CALUDE_relay_race_total_time_l3615_361544

/-- The total time for a relay race with four athletes -/
def relay_race_time (athlete1_time athlete2_extra athlete3_less athlete4_less : ℕ) : ℕ :=
  let athlete2_time := athlete1_time + athlete2_extra
  let athlete3_time := athlete2_time - athlete3_less
  let athlete4_time := athlete1_time - athlete4_less
  athlete1_time + athlete2_time + athlete3_time + athlete4_time

/-- Theorem stating that the total time for the given relay race is 200 seconds -/
theorem relay_race_total_time :
  relay_race_time 55 10 15 25 = 200 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_total_time_l3615_361544


namespace NUMINAMATH_CALUDE_community_center_tables_l3615_361502

/-- The number of chairs per table -/
def chairs_per_table : ℕ := 8

/-- The number of legs per chair -/
def legs_per_chair : ℕ := 4

/-- The number of legs per table -/
def legs_per_table : ℕ := 3

/-- The total number of legs from all chairs and tables -/
def total_legs : ℕ := 759

/-- The number of tables in the community center -/
def num_tables : ℕ := 22

theorem community_center_tables :
  chairs_per_table * num_tables * legs_per_chair + num_tables * legs_per_table = total_legs :=
sorry

end NUMINAMATH_CALUDE_community_center_tables_l3615_361502


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3615_361516

theorem arithmetic_sequence_sum (x y z : ℤ) : 
  (x + y + z = 72) →
  (∃ (x y z : ℤ), x + y + z = 72 ∧ y - x = 1) ∧
  (∃ (x y z : ℤ), x + y + z = 72 ∧ y - x = 2) ∧
  (¬ ∃ (x y z : ℤ), x + y + z = 72 ∧ y - x = 2 ∧ Odd x) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3615_361516


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3615_361532

theorem rationalize_denominator :
  let x : ℝ := Real.rpow 3 (1/3)
  (1 / (x + Real.rpow 27 (1/3) - Real.rpow 9 (1/3))) = (x^2 + 3*x + 3) / (3 * 21) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3615_361532


namespace NUMINAMATH_CALUDE_perpendicular_iff_x_eq_neg_one_third_l3615_361584

/-- Two vectors in R² -/
def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b : Fin 2 → ℝ := ![3, 1]

/-- Dot product of two vectors in R² -/
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Theorem: Vectors a and b are perpendicular if and only if x = -1/3 -/
theorem perpendicular_iff_x_eq_neg_one_third (x : ℝ) : 
  dot_product (a x) b = 0 ↔ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_iff_x_eq_neg_one_third_l3615_361584


namespace NUMINAMATH_CALUDE_class_mood_distribution_l3615_361550

theorem class_mood_distribution (total_children : Nat) (happy_children : Nat) (sad_children : Nat) (anxious_children : Nat)
  (total_boys : Nat) (total_girls : Nat) (happy_boys : Nat) (sad_girls : Nat) (anxious_girls : Nat)
  (h1 : total_children = 80)
  (h2 : happy_children = 25)
  (h3 : sad_children = 15)
  (h4 : anxious_children = 20)
  (h5 : total_boys = 35)
  (h6 : total_girls = 45)
  (h7 : happy_boys = 10)
  (h8 : sad_girls = 6)
  (h9 : anxious_girls = 12)
  (h10 : total_children = total_boys + total_girls) :
  (total_boys - (happy_boys + (sad_children - sad_girls) + (anxious_children - anxious_girls)) = 8) ∧
  (happy_children - happy_boys = 15) :=
by sorry

end NUMINAMATH_CALUDE_class_mood_distribution_l3615_361550


namespace NUMINAMATH_CALUDE_book_page_numbering_l3615_361582

/-- The total number of digits used to number pages in a book -/
def total_digits (n : ℕ) : ℕ :=
  let single_digit := min n 9
  let double_digit := max 0 (min n 99 - 9)
  let triple_digit := max 0 (n - 99)
  single_digit + 2 * double_digit + 3 * triple_digit

/-- Theorem stating that a book with 266 pages uses 690 digits for page numbering -/
theorem book_page_numbering :
  total_digits 266 = 690 := by
  sorry

end NUMINAMATH_CALUDE_book_page_numbering_l3615_361582


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_m_value_l3615_361547

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h_pos : a > 0

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y = p.a * x^2

/-- Theorem: Value of m for symmetric points on parabola -/
theorem parabola_symmetric_points_m_value
  (p : Parabola)
  (h_focus_directrix : (1 : ℝ) / (4 * p.a) = 1/4)
  (A B : ParabolaPoint p)
  (h_symmetric : ∃ m : ℝ, (A.y + B.y) / 2 = ((A.x + B.x) / 2) + m ∧
                           (B.y - A.y) = (B.x - A.x))
  (h_product : A.x * B.x = -1/2) :
  ∃ m : ℝ, (A.y + B.y) / 2 = ((A.x + B.x) / 2) + m ∧ m = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_symmetric_points_m_value_l3615_361547


namespace NUMINAMATH_CALUDE_triangle_area_combinations_l3615_361503

theorem triangle_area_combinations (a b c : ℝ) (A B C : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (A > 0 ∧ B > 0 ∧ C > 0) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a = Real.sqrt 3 ∧ b = 2 ∧ (Real.sin B + Real.sin C) / Real.sin A = (a + c) / (b - c) →
    1/2 * a * c * Real.sin B = 3 * (Real.sqrt 7 - Real.sqrt 3) / 8) ∧
  (a = Real.sqrt 3 ∧ b = 2 ∧ Real.cos ((B - C) / 2)^2 - Real.sin B * Real.sin C = 1/4 →
    1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_combinations_l3615_361503


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l3615_361593

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - |x + a|

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l3615_361593


namespace NUMINAMATH_CALUDE_x_equals_y_plus_m_percent_l3615_361530

-- Define the relationship between x, y, and m
def is_m_percent_more (x y m : ℝ) : Prop :=
  x = y + (m / 100) * y

-- Theorem statement
theorem x_equals_y_plus_m_percent (x y m : ℝ) :
  is_m_percent_more x y m → x = (100 + m) / 100 * y := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_plus_m_percent_l3615_361530


namespace NUMINAMATH_CALUDE_average_of_first_12_even_numbers_l3615_361506

def first_12_even_numbers : List ℤ :=
  [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]

theorem average_of_first_12_even_numbers :
  (List.sum first_12_even_numbers) / (List.length first_12_even_numbers) = -1 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_12_even_numbers_l3615_361506


namespace NUMINAMATH_CALUDE_expression_evaluation_l3615_361580

theorem expression_evaluation : 
  (3 - 4 * (3 - 5)⁻¹)⁻¹ = (1 : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3615_361580


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3615_361508

/-- Properties of a quadratic equation -/
theorem quadratic_equation_properties
  (a b c : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  -- Statement 1
  (∀ x, f x = 0 ↔ x = 1 ∨ x = 2) → 2 * a - c = 0 ∧
  -- Statement 2
  (b = 2 * a + c → b^2 - 4 * a * c > 0) ∧
  -- Statement 3
  (∀ m, f m = 0 → b^2 - 4 * a * c = (2 * a * m + b)^2) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_properties_l3615_361508


namespace NUMINAMATH_CALUDE_reciprocal_determinant_solution_ratios_l3615_361504

/-- Given a 2x2 matrix with determinant D ≠ 0, prove that the determinant of its adjugate divided by D is equal to 1/D -/
theorem reciprocal_determinant (a b c d : ℝ) (h : a * d - b * c ≠ 0) :
  let D := a * d - b * c
  (d / D) * (a / D) - (-c / D) * (-b / D) = 1 / D := by sorry

/-- For a system of two linear equations in three variables,
    prove that the ratios of the solutions are given by specific 2x2 determinants -/
theorem solution_ratios (a b c d e f : ℝ) 
  (h1 : ∀ x y z : ℝ, a * x + b * y + c * z = 0 → d * x + e * y + f * z = 0) :
  ∃ (k : ℝ), k ≠ 0 ∧
    (b * f - c * e) * k = (c * d - a * f) * k ∧
    (c * d - a * f) * k = (a * e - b * d) * k := by sorry

end NUMINAMATH_CALUDE_reciprocal_determinant_solution_ratios_l3615_361504


namespace NUMINAMATH_CALUDE_triangle_identity_l3615_361572

/-- The triangle operation on pairs of real numbers -/
def triangle (a b c d : ℝ) : ℝ × ℝ := (a*c + b*d, a*d + b*c)

/-- Theorem: If (u,v) △ (x,y) = (u,v) for all real u and v, then (x,y) = (1,0) -/
theorem triangle_identity (x y : ℝ) : 
  (∀ u v : ℝ, triangle u v x y = (u, v)) → (x, y) = (1, 0) := by sorry

end NUMINAMATH_CALUDE_triangle_identity_l3615_361572


namespace NUMINAMATH_CALUDE_three_digit_prime_with_special_property_l3615_361552

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_different (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

def last_digit_is_sum_of_first_two (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones = hundreds + tens

theorem three_digit_prime_with_special_property (p : ℕ) 
  (h_prime : Nat.Prime p)
  (h_three_digit : is_three_digit p)
  (h_different : all_digits_different p)
  (h_sum : last_digit_is_sum_of_first_two p) :
  p % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_prime_with_special_property_l3615_361552


namespace NUMINAMATH_CALUDE_democrat_count_l3615_361536

theorem democrat_count (total : ℕ) (difference : ℕ) (h1 : total = 434) (h2 : difference = 30) :
  let democrats := (total - difference) / 2
  democrats = 202 := by
sorry

end NUMINAMATH_CALUDE_democrat_count_l3615_361536


namespace NUMINAMATH_CALUDE_same_solution_value_l3615_361523

theorem same_solution_value (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 5 = 1 ∧ c * x + 15 = -5) ↔ c = 15 := by
sorry

end NUMINAMATH_CALUDE_same_solution_value_l3615_361523


namespace NUMINAMATH_CALUDE_quadratic_linear_relationship_l3615_361529

/-- Given a quadratic function y₁ and a linear function y₂, prove the relationship between b and c -/
theorem quadratic_linear_relationship (a b c : ℝ) : 
  let y₁ := fun x => (x + 2*a) * (x - 2*b)
  let y₂ := fun x => -x + 2*b
  let y := fun x => y₁ x + y₂ x
  a + 2 = b → 
  y c = 0 → 
  (c = 5 - 2*b ∨ c = 2*b) := by sorry

end NUMINAMATH_CALUDE_quadratic_linear_relationship_l3615_361529


namespace NUMINAMATH_CALUDE_line_slope_135_degrees_l3615_361596

/-- The slope of a line in degrees -/
def Slope : Type := ℝ

/-- The equation of a line in the form mx + y + c = 0 -/
structure Line where
  m : ℝ
  c : ℝ

/-- The tangent of an angle in degrees -/
noncomputable def tan_degrees (θ : ℝ) : ℝ := sorry

theorem line_slope_135_degrees (l : Line) (h : l.c = 2) : 
  (tan_degrees 135 = -l.m) → l.m = 1 := by sorry

end NUMINAMATH_CALUDE_line_slope_135_degrees_l3615_361596


namespace NUMINAMATH_CALUDE_max_distance_complex_numbers_l3615_361543

theorem max_distance_complex_numbers :
  ∃ (M : ℝ), M = 81 + 9 * Real.sqrt 5 ∧
  ∀ (z : ℂ), Complex.abs z = 3 →
  Complex.abs ((1 + 2*Complex.I) * z^2 - z^4) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_numbers_l3615_361543


namespace NUMINAMATH_CALUDE_set_relationship_l3615_361556

-- Define the sets
def set1 : Set ℝ := {x | (1 : ℝ) / x ≤ 1}
def set2 : Set ℝ := {x | Real.log x ≥ 0}

-- Theorem statement
theorem set_relationship : Set.Subset set2 set1 ∧ ¬(set1 = set2) := by
  sorry

end NUMINAMATH_CALUDE_set_relationship_l3615_361556


namespace NUMINAMATH_CALUDE_simplify_expression_l3615_361509

theorem simplify_expression (y : ℝ) : (3/2 - 5*y) - (5/2 + 7*y) = -1 - 12*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3615_361509


namespace NUMINAMATH_CALUDE_intersection_condition_l3615_361559

theorem intersection_condition (a : ℝ) : 
  let A : Set ℝ := {0, 1}
  let B : Set ℝ := {x | x > a}
  (∃! x, x ∈ A ∩ B) → 0 ≤ a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3615_361559


namespace NUMINAMATH_CALUDE_train_journey_duration_l3615_361500

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference in minutes between two times -/
def timeDifferenceInMinutes (t1 t2 : Time) : Nat :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Represents the state of clock hands -/
inductive ClockHandState
  | Symmetrical
  | NotSymmetrical

theorem train_journey_duration (stationArrival : Time)
                               (trainDeparture : Time)
                               (destinationArrival : Time)
                               (stationDeparture : Time)
                               (boardingState : ClockHandState)
                               (alightingState : ClockHandState) :
  stationArrival = ⟨8, 0⟩ →
  trainDeparture = ⟨8, 35⟩ →
  destinationArrival = ⟨14, 15⟩ →
  stationDeparture = ⟨15, 0⟩ →
  boardingState = ClockHandState.Symmetrical →
  alightingState = ClockHandState.Symmetrical →
  timeDifferenceInMinutes trainDeparture stationDeparture = 385 :=
by sorry

end NUMINAMATH_CALUDE_train_journey_duration_l3615_361500


namespace NUMINAMATH_CALUDE_social_media_weekly_time_l3615_361564

/-- Calculates the weekly time spent on social media given daily phone usage and social media ratio -/
def weekly_social_media_time (daily_phone_time : ℝ) (social_media_ratio : ℝ) : ℝ :=
  daily_phone_time * social_media_ratio * 7

/-- Theorem: Given 8 hours daily phone usage with half on social media, weekly social media time is 28 hours -/
theorem social_media_weekly_time : 
  weekly_social_media_time 8 0.5 = 28 := by
  sorry


end NUMINAMATH_CALUDE_social_media_weekly_time_l3615_361564


namespace NUMINAMATH_CALUDE_smallest_transactions_to_exceed_fee_l3615_361569

/-- Represents the types of transactions --/
inductive TransactionType
| Autodebit
| Cheque
| CashWithdrawal

/-- Represents the cost of each transaction type --/
def transactionCost : TransactionType → ℚ
| TransactionType.Autodebit => 0.60
| TransactionType.Cheque => 0.50
| TransactionType.CashWithdrawal => 0.45

/-- Calculates the total cost for the first 25 transactions --/
def firstTwentyFiveCost : ℚ := 15 * transactionCost TransactionType.Autodebit +
                                5 * transactionCost TransactionType.Cheque +
                                5 * transactionCost TransactionType.CashWithdrawal

/-- Theorem stating that 29 is the smallest number of transactions to exceed $15.95 --/
theorem smallest_transactions_to_exceed_fee :
  ∀ n : ℕ, n ≥ 29 ↔ 
    firstTwentyFiveCost + (n - 25 : ℕ) * transactionCost TransactionType.Autodebit > 15.95 :=
by sorry

end NUMINAMATH_CALUDE_smallest_transactions_to_exceed_fee_l3615_361569


namespace NUMINAMATH_CALUDE_tank_capacity_l3615_361538

theorem tank_capacity (x : ℚ) 
  (h1 : 2/3 * x - 15 = 1/3 * x) : x = 45 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l3615_361538


namespace NUMINAMATH_CALUDE_complex_modulus_l3615_361554

theorem complex_modulus (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3615_361554


namespace NUMINAMATH_CALUDE_arrival_time_difference_l3615_361546

-- Define the distance to the park
def distance_to_park : ℝ := 3

-- Define Jack's speed
def jack_speed : ℝ := 3

-- Define Jill's speed
def jill_speed : ℝ := 12

-- Define the conversion factor from hours to minutes
def hours_to_minutes : ℝ := 60

-- Theorem statement
theorem arrival_time_difference : 
  (distance_to_park / jack_speed - distance_to_park / jill_speed) * hours_to_minutes = 45 := by
  sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l3615_361546


namespace NUMINAMATH_CALUDE_additional_grazing_area_l3615_361579

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 23^2 - π * 12^2 = 385 * π := by
  sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l3615_361579


namespace NUMINAMATH_CALUDE_even_function_order_l3615_361527

def f (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem even_function_order (b c : ℝ) 
  (h : ∀ x, f x b c = f (-x) b c) : 
  f 1 b c < f (-2) b c ∧ f (-2) b c < f 3 b c := by
  sorry

end NUMINAMATH_CALUDE_even_function_order_l3615_361527


namespace NUMINAMATH_CALUDE_min_even_integers_l3615_361534

theorem min_even_integers (a b c d e f g h : ℤ) : 
  a + b + c = 30 → 
  a + b + c + d + e = 49 → 
  a + b + c + d + e + f + g + h = 78 → 
  ∃ (evens : Finset ℤ), evens ⊆ {a, b, c, d, e, f, g, h} ∧ 
                         evens.card = 2 ∧
                         (∀ x ∈ evens, Even x) ∧
                         (∀ (other_evens : Finset ℤ), 
                           other_evens ⊆ {a, b, c, d, e, f, g, h} → 
                           (∀ x ∈ other_evens, Even x) → 
                           other_evens.card ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l3615_361534


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3615_361592

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3615_361592


namespace NUMINAMATH_CALUDE_sum_of_angles_equals_360_l3615_361562

-- Define the angles as real numbers
variable (A B C D F G : ℝ)

-- Define the property of being a quadrilateral
def is_quadrilateral (A B C D : ℝ) : Prop :=
  A + B + C + D = 360

-- State the theorem
theorem sum_of_angles_equals_360 
  (h : is_quadrilateral A B C D) : A + B + C + D + F + G = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_equals_360_l3615_361562


namespace NUMINAMATH_CALUDE_complex_equation_solution_product_l3615_361521

theorem complex_equation_solution_product (x : ℂ) :
  x^3 + x^2 + 3*x = 2 + 2*Complex.I →
  ∃ (x₁ x₂ : ℂ), x₁ ≠ x₂ ∧ 
    x₁^3 + x₁^2 + 3*x₁ = 2 + 2*Complex.I ∧
    x₂^3 + x₂^2 + 3*x₂ = 2 + 2*Complex.I ∧
    (x₁.re * x₂.re = 1 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_product_l3615_361521


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l3615_361571

theorem geometric_mean_of_4_and_9 :
  ∃ x : ℝ, x^2 = 4 * 9 ∧ (x = 6 ∨ x = -6) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l3615_361571


namespace NUMINAMATH_CALUDE_apple_selling_price_l3615_361566

-- Define the cost price
def cost_price : ℚ := 17

-- Define the selling price as a function of the cost price
def selling_price (cp : ℚ) : ℚ := (5 / 6) * cp

-- Theorem stating that the selling price is 5/6 of the cost price
theorem apple_selling_price :
  selling_price cost_price = (5 / 6) * cost_price :=
by sorry

end NUMINAMATH_CALUDE_apple_selling_price_l3615_361566


namespace NUMINAMATH_CALUDE_reciprocal_of_proper_fraction_greater_than_one_l3615_361561

-- Define a proper fraction
def ProperFraction (n d : ℕ) : Prop := 0 < n ∧ n < d

-- Theorem statement
theorem reciprocal_of_proper_fraction_greater_than_one {n d : ℕ} (h : ProperFraction n d) :
  (d : ℝ) / (n : ℝ) > 1 := by
  sorry


end NUMINAMATH_CALUDE_reciprocal_of_proper_fraction_greater_than_one_l3615_361561


namespace NUMINAMATH_CALUDE_equation_solution_l3615_361573

theorem equation_solution : ∃! x : ℚ, (2 * x) / (x + 3) + 1 = 7 / (2 * x + 6) :=
  by
    use 1/6
    sorry

end NUMINAMATH_CALUDE_equation_solution_l3615_361573


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3615_361581

theorem sum_of_coefficients : 
  let p (x : ℝ) := -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)
  p 1 = 45 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3615_361581


namespace NUMINAMATH_CALUDE_prob_diamond_or_club_half_l3615_361577

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (cards_per_suit : ℕ)
  (diamond_club_cards : ℕ)

/-- Probability of drawing a diamond or club from the top of a shuffled deck -/
def prob_diamond_or_club (d : Deck) : ℚ :=
  d.diamond_club_cards / d.total_cards

/-- Theorem stating the probability of drawing a diamond or club is 1/2 -/
theorem prob_diamond_or_club_half (d : Deck) 
  (h1 : d.total_cards = 52) 
  (h2 : d.cards_per_suit = 13) 
  (h3 : d.diamond_club_cards = 2 * d.cards_per_suit) : 
  prob_diamond_or_club d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_diamond_or_club_half_l3615_361577


namespace NUMINAMATH_CALUDE_dave_ticket_problem_l3615_361512

theorem dave_ticket_problem (total_used : ℕ) (difference : ℕ) 
  (h1 : total_used = 12) (h2 : difference = 5) : 
  ∃ (clothes_tickets : ℕ), 
    clothes_tickets + (clothes_tickets + difference) = total_used ∧ 
    clothes_tickets = 7 := by
  sorry

end NUMINAMATH_CALUDE_dave_ticket_problem_l3615_361512


namespace NUMINAMATH_CALUDE_cube_sum_equals_negative_27_l3615_361522

theorem cube_sum_equals_negative_27 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) : 
  a^3 + b^3 + c^3 = -27 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_negative_27_l3615_361522


namespace NUMINAMATH_CALUDE_slate_rock_probability_l3615_361525

/-- The probability of choosing two slate rocks without replacement from a field of rocks. -/
theorem slate_rock_probability (slate_rocks pumice_rocks granite_rocks : ℕ) 
  (h_slate : slate_rocks = 10)
  (h_pumice : pumice_rocks = 11)
  (h_granite : granite_rocks = 4) :
  let total_rocks := slate_rocks + pumice_rocks + granite_rocks
  (slate_rocks : ℚ) / total_rocks * ((slate_rocks - 1) : ℚ) / (total_rocks - 1) = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_slate_rock_probability_l3615_361525


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3615_361548

theorem square_root_of_nine (x : ℝ) : x ^ 2 = 9 ↔ x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3615_361548


namespace NUMINAMATH_CALUDE_ticket_revenue_calculation_l3615_361555

/-- Calculates the total revenue from ticket sales given the specified conditions -/
theorem ticket_revenue_calculation (total_tickets : ℕ) (student_price nonstudent_price : ℚ)
  (student_tickets : ℕ) (h1 : total_tickets = 821) (h2 : student_price = 2)
  (h3 : nonstudent_price = 3) (h4 : student_tickets = 530) :
  (student_tickets : ℚ) * student_price +
  ((total_tickets - student_tickets) : ℚ) * nonstudent_price = 1933 := by
  sorry

end NUMINAMATH_CALUDE_ticket_revenue_calculation_l3615_361555


namespace NUMINAMATH_CALUDE_grade_10_sample_size_l3615_361568

/-- Represents the number of students to be sampled from a grade in a stratified sampling scenario -/
def stratified_sample (total_sample : ℕ) (grade_ratio : ℕ) (total_ratio : ℕ) : ℕ :=
  (grade_ratio * total_sample) / total_ratio

/-- Theorem stating that in a stratified sampling of 65 students from three grades with a ratio of 4:4:5, 
    the number of students to be sampled from the first grade is 20 -/
theorem grade_10_sample_size :
  stratified_sample 65 4 (4 + 4 + 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_grade_10_sample_size_l3615_361568


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3615_361594

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : 2 * x₁^2 - 3 * x₁ - 1 = 0 ∧ 2 * x₂^2 - 3 * x₂ - 1 = 0) : 
  (1 / x₁ + 1 / x₂ = -3) ∧ 
  ((x₁^2 - x₂^2)^2 = 153 / 16) ∧ 
  (2 * x₁^2 + 3 * x₂ = 11 / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3615_361594


namespace NUMINAMATH_CALUDE_danny_found_caps_l3615_361598

/-- Represents the number of bottle caps Danny had initially -/
def initial_caps : ℕ := 6

/-- Represents the total number of bottle caps Danny has now -/
def total_caps : ℕ := 28

/-- Represents the number of bottle caps Danny found at the park -/
def caps_found : ℕ := total_caps - initial_caps

theorem danny_found_caps : caps_found = 22 := by
  sorry

end NUMINAMATH_CALUDE_danny_found_caps_l3615_361598


namespace NUMINAMATH_CALUDE_ratio_is_two_l3615_361575

/-- An isosceles right triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- Length of OP -/
  a : ℝ
  /-- Length of OQ -/
  b : ℝ
  /-- Assumption that a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- The area of the square PQRS is 2/5 of the area of triangle AOB -/
  area_ratio : (a^2 + b^2) / ((2*a + b)^2 / 2) = 2/5

/-- The main theorem -/
theorem ratio_is_two (t : IsoscelesRightTriangleWithSquare) : t.a / t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_two_l3615_361575


namespace NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l3615_361588

theorem percentage_of_boys_studying_science 
  (total_boys : ℕ) 
  (boys_from_school_A : ℕ) 
  (boys_not_studying_science : ℕ) 
  (h1 : total_boys = 550)
  (h2 : boys_from_school_A = (20 : ℕ) * total_boys / 100)
  (h3 : boys_not_studying_science = 77) :
  (boys_from_school_A - boys_not_studying_science) * 100 / boys_from_school_A = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l3615_361588


namespace NUMINAMATH_CALUDE_pascals_remaining_distance_l3615_361519

/-- Proves that Pascal's remaining cycling distance is 256 miles -/
theorem pascals_remaining_distance (current_speed : ℝ) (reduced_speed : ℝ) (increased_speed : ℝ)
  (h1 : current_speed = 8)
  (h2 : reduced_speed = current_speed - 4)
  (h3 : increased_speed = current_speed * 1.5)
  (h4 : ∃ (t : ℝ), current_speed * t = reduced_speed * (t + 16))
  (h5 : ∃ (t : ℝ), increased_speed * t = reduced_speed * (t + 16)) :
  ∃ (distance : ℝ), distance = 256 ∧ 
    (∃ (t : ℝ), distance = current_speed * t ∧
                distance = reduced_speed * (t + 16) ∧
                distance = increased_speed * (t - 16)) :=
sorry

end NUMINAMATH_CALUDE_pascals_remaining_distance_l3615_361519


namespace NUMINAMATH_CALUDE_remainder_x_50_divided_by_x_plus_1_cubed_l3615_361533

theorem remainder_x_50_divided_by_x_plus_1_cubed (x : ℚ) :
  (x^50) % (x + 1)^3 = 1225*x^2 + 2450*x + 1176 := by
  sorry

end NUMINAMATH_CALUDE_remainder_x_50_divided_by_x_plus_1_cubed_l3615_361533


namespace NUMINAMATH_CALUDE_cookie_cost_l3615_361526

/-- The cost of cookies is equal to the sum of money Diane has and the additional money she needs. -/
theorem cookie_cost (money_has : ℕ) (money_needs : ℕ) (cost : ℕ) : 
  money_has = 27 → money_needs = 38 → cost = money_has + money_needs := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_l3615_361526


namespace NUMINAMATH_CALUDE_triangle_count_is_nine_l3615_361531

/-- Represents the triangular grid structure described in the problem -/
structure TriangularGrid :=
  (top_row : Nat)
  (middle_row : Nat)
  (bottom_row : Nat)
  (has_inverted_triangle : Bool)

/-- Calculates the total number of triangles in the given grid -/
def count_triangles (grid : TriangularGrid) : Nat :=
  let small_triangles := grid.top_row + grid.middle_row + grid.bottom_row
  let medium_triangles := if grid.top_row ≥ 3 then 1 else 0 +
                          if grid.middle_row + grid.bottom_row ≥ 3 then 1 else 0
  let large_triangle := if grid.has_inverted_triangle then 1 else 0
  small_triangles + medium_triangles + large_triangle

/-- The specific grid described in the problem -/
def problem_grid : TriangularGrid :=
  { top_row := 3,
    middle_row := 2,
    bottom_row := 1,
    has_inverted_triangle := true }

theorem triangle_count_is_nine :
  count_triangles problem_grid = 9 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_nine_l3615_361531


namespace NUMINAMATH_CALUDE_max_product_theorem_l3615_361520

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 10000 ∧ a < 100000 ∧ b ≥ 10000 ∧ b < 100000 ∧
  (∀ d : ℕ, d < 10 → (d.digits 10).count d + (a.digits 10).count d + (b.digits 10).count d = 1)

def max_product : ℕ := 96420 * 87531

theorem max_product_theorem :
  ∀ a b : ℕ, is_valid_pair a b → a * b ≤ max_product :=
by sorry

end NUMINAMATH_CALUDE_max_product_theorem_l3615_361520


namespace NUMINAMATH_CALUDE_two_digit_integer_property_l3615_361541

theorem two_digit_integer_property (a b k : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0) : 
  let n := 10 * a + b
  let m := 10 * b + a
  n = k * (a - b) → m = (k - 9) * (a - b) := by
sorry

end NUMINAMATH_CALUDE_two_digit_integer_property_l3615_361541


namespace NUMINAMATH_CALUDE_practice_time_for_second_recital_l3615_361597

/-- Represents the relationship between practice time and mistakes for a recital -/
structure Recital where
  practice_time : ℝ
  mistakes : ℝ

/-- The constant product of practice time and mistakes -/
def inverse_relation_constant (r : Recital) : ℝ :=
  r.practice_time * r.mistakes

theorem practice_time_for_second_recital
  (first_recital : Recital)
  (h1 : first_recital.practice_time = 5)
  (h2 : first_recital.mistakes = 12)
  (h3 : ∀ r : Recital, inverse_relation_constant r = inverse_relation_constant first_recital)
  (h4 : ∃ second_recital : Recital,
    (first_recital.mistakes + second_recital.mistakes) / 2 = 8) :
  ∃ second_recital : Recital, second_recital.practice_time = 15 := by
sorry

end NUMINAMATH_CALUDE_practice_time_for_second_recital_l3615_361597


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3615_361537

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) → 
  a + h + k = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3615_361537


namespace NUMINAMATH_CALUDE_graph_shift_l3615_361565

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the transformation
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x => g x - 3

-- Theorem statement
theorem graph_shift (x y : ℝ) : 
  y = transform g x ↔ y + 3 = g x := by sorry

end NUMINAMATH_CALUDE_graph_shift_l3615_361565


namespace NUMINAMATH_CALUDE_integral_equality_l3615_361595

open Real MeasureTheory

theorem integral_equality : ∫ (x : ℝ) in (0)..(1), 
  Real.exp (Real.sqrt ((1 - x) / (1 + x))) / ((1 + x) * Real.sqrt (1 - x^2)) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equality_l3615_361595


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3615_361511

theorem smallest_integer_with_remainders (n : ℕ) : 
  (n > 1) → 
  (∀ m : ℕ, m > 1 ∧ m < n → ¬(m % 6 = 1 ∧ m % 7 = 1 ∧ m % 9 = 1)) → 
  (n % 6 = 1 ∧ n % 7 = 1 ∧ n % 9 = 1) → 
  (n = 127 ∧ 120 < n ∧ n < 199) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3615_361511


namespace NUMINAMATH_CALUDE_barycentric_coords_proportional_to_areas_l3615_361510

-- Define a triangle ABC
variable (A B C : ℝ × ℝ)

-- Define a point P inside the triangle
variable (P : ℝ × ℝ)

-- Define the area function
noncomputable def area (X Y Z : ℝ × ℝ) : ℝ := sorry

-- Define the barycentric coordinates
def barycentric_coords (P A B C : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

-- State the theorem
theorem barycentric_coords_proportional_to_areas :
  ∃ (k : ℝ), k ≠ 0 ∧ 
    barycentric_coords P A B C = 
      (k * area P B C, k * area P C A, k * area P A B) := by sorry

end NUMINAMATH_CALUDE_barycentric_coords_proportional_to_areas_l3615_361510


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_sum_l3615_361542

theorem imaginary_part_of_complex_sum (i : ℂ) (h : i * i = -1) :
  Complex.im ((1 / (i - 2)) + (2 / (1 - 2*i))) = 3/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_sum_l3615_361542


namespace NUMINAMATH_CALUDE_marys_nickels_l3615_361587

/-- Given that Mary initially had 7 nickels and now has 12 nickels,
    prove that Mary's dad gave her 5 nickels. -/
theorem marys_nickels (initial : ℕ) (final : ℕ) (given : ℕ) :
  initial = 7 → final = 12 → given = final - initial → given = 5 := by
  sorry

end NUMINAMATH_CALUDE_marys_nickels_l3615_361587


namespace NUMINAMATH_CALUDE_race_speeds_l3615_361570

theorem race_speeds (x : ℝ) (h : x > 0) : 
  ∃ (a b : ℝ),
    (1000 = a * x) ∧ 
    (1000 - 167 = b * x) ∧
    (a = 1000 / x) ∧ 
    (b = 833 / x) := by
  sorry

end NUMINAMATH_CALUDE_race_speeds_l3615_361570


namespace NUMINAMATH_CALUDE_preimage_of_one_two_l3615_361557

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, 2 * p.1 - 3 * p.2)

theorem preimage_of_one_two :
  f (1, 0) = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_one_two_l3615_361557


namespace NUMINAMATH_CALUDE_inverse_direct_proportionality_l3615_361553

/-- Given two real numbers are inversely proportional -/
def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

/-- Given two real numbers are directly proportional -/
def directly_proportional (z y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ z = k * y

/-- Main theorem -/
theorem inverse_direct_proportionality
  (x y z : ℝ → ℝ)
  (h_inv : ∀ t, inversely_proportional (x t) (y t))
  (h_dir : ∀ t, directly_proportional (z t) (y t))
  (h_x : x 9 = 40)
  (h_z : z 10 = 45) :
  x 20 = 18 ∧ z 20 = 90 := by
  sorry


end NUMINAMATH_CALUDE_inverse_direct_proportionality_l3615_361553


namespace NUMINAMATH_CALUDE_difference_a_minus_c_l3615_361576

/-- Given that the average of a, b, and d is 110, and the average of b, c, and d is 150, prove that a - c = -120 -/
theorem difference_a_minus_c (a b c d : ℝ) 
  (h1 : (a + b + d) / 3 = 110) 
  (h2 : (b + c + d) / 3 = 150) : 
  a - c = -120 := by
  sorry

end NUMINAMATH_CALUDE_difference_a_minus_c_l3615_361576


namespace NUMINAMATH_CALUDE_range_of_m_max_min_distance_exists_line_l_l3615_361589

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - m = 0

-- Define point A
def point_A (m : ℝ) : ℝ × ℝ := (m, -2)

-- Define the condition for a point to be inside the circle
def inside_circle (x y m : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 < 5 + m

-- Theorem 1
theorem range_of_m (m : ℝ) : 
  (∃ x y, circle_C x y m ∧ inside_circle m (-2) m) → -1 < m ∧ m < 4 :=
sorry

-- Theorem 2
theorem max_min_distance (x y : ℝ) :
  circle_C x y 4 → 4 ≤ (x - 4)^2 + (y - 2)^2 ∧ (x - 4)^2 + (y - 2)^2 ≤ 64 :=
sorry

-- Define the line l
def line_l (k b : ℝ) (x y : ℝ) : Prop := y = x + b

-- Theorem 3
theorem exists_line_l :
  ∃ b, (b = -4 ∨ b = 1) ∧
    ∃ x₁ y₁ x₂ y₂, 
      circle_C x₁ y₁ 4 ∧ circle_C x₂ y₂ 4 ∧
      line_l 1 b x₁ y₁ ∧ line_l 1 b x₂ y₂ ∧
      (x₁ + x₂ = 0) ∧ (y₁ + y₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_max_min_distance_exists_line_l_l3615_361589


namespace NUMINAMATH_CALUDE_product_xyz_equals_one_l3615_361585

theorem product_xyz_equals_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) 
  (h3 : z + 1/x = 2) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_one_l3615_361585


namespace NUMINAMATH_CALUDE_percent_of_percent_l3615_361574

theorem percent_of_percent (x : ℝ) : (30 / 100) * (70 / 100) * x = (21 / 100) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l3615_361574


namespace NUMINAMATH_CALUDE_minimum_tip_percentage_l3615_361501

theorem minimum_tip_percentage
  (meal_cost : ℝ)
  (total_paid : ℝ)
  (h_meal_cost : meal_cost = 35.50)
  (h_total_paid : total_paid = 37.275)
  (h_tip_less_than_8 : (total_paid - meal_cost) / meal_cost < 0.08) :
  (total_paid - meal_cost) / meal_cost = 0.05 :=
by sorry

end NUMINAMATH_CALUDE_minimum_tip_percentage_l3615_361501


namespace NUMINAMATH_CALUDE_math_statements_l3615_361505

theorem math_statements :
  (∃ x : ℚ, x < -1 ∧ x > 1/x) ∧
  (∃ y : ℝ, y ≥ 0 ∧ -y ≥ y) ∧
  (∀ z : ℚ, z < 0 → z^2 > z) :=
by sorry

end NUMINAMATH_CALUDE_math_statements_l3615_361505


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3615_361558

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧
  ¬(∀ x : ℝ, x < 2 → x^2 - 3*x + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3615_361558


namespace NUMINAMATH_CALUDE_circle_coverage_fraction_l3615_361515

/-- The fraction of a smaller circle's area not covered by a larger circle when placed inside it -/
theorem circle_coverage_fraction (dX dY : ℝ) (h_dX : dX = 16) (h_dY : dY = 18) (h_inside : dX < dY) :
  (π * (dY / 2)^2 - π * (dX / 2)^2) / (π * (dX / 2)^2) = 17 / 64 := by
  sorry

end NUMINAMATH_CALUDE_circle_coverage_fraction_l3615_361515


namespace NUMINAMATH_CALUDE_midpoint_set_properties_l3615_361563

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  perimeter : ℝ

/-- The set of midpoints of segments with one end in F and the other in G -/
def midpoint_set (F G : ConvexPolygon) : Set (ℝ × ℝ) := sorry

theorem midpoint_set_properties (F G : ConvexPolygon) :
  let H := midpoint_set F G
  ∃ (sides_H : ℕ) (perimeter_H : ℝ),
    (ConvexPolygon.sides F).max (ConvexPolygon.sides G) ≤ sides_H ∧
    sides_H ≤ (ConvexPolygon.sides F) + (ConvexPolygon.sides G) ∧
    perimeter_H = (ConvexPolygon.perimeter F + ConvexPolygon.perimeter G) / 2 ∧
    (∀ (x y : ℝ × ℝ), x ∈ H → y ∈ H → (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • x + t • y ∈ H)) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_set_properties_l3615_361563


namespace NUMINAMATH_CALUDE_rancher_problem_solution_l3615_361518

/-- Represents the rancher's cattle problem -/
structure CattleProblem where
  initial_cattle : ℕ
  dead_cattle : ℕ
  price_reduction : ℚ
  loss_amount : ℚ

/-- Calculates the original price per head of cattle -/
def original_price (p : CattleProblem) : ℚ :=
  p.loss_amount / (p.initial_cattle - p.dead_cattle : ℚ)

/-- Calculates the total amount the rancher would have made -/
def total_amount (p : CattleProblem) : ℚ :=
  (p.initial_cattle : ℚ) * original_price p

/-- Theorem stating the solution to the rancher's problem -/
theorem rancher_problem_solution (p : CattleProblem) 
  (h1 : p.initial_cattle = 340)
  (h2 : p.dead_cattle = 172)
  (h3 : p.price_reduction = 150)
  (h4 : p.loss_amount = 25200) :
  total_amount p = 49813.40 := by
  sorry

end NUMINAMATH_CALUDE_rancher_problem_solution_l3615_361518


namespace NUMINAMATH_CALUDE_expand_polynomial_l3615_361514

theorem expand_polynomial (x : ℝ) : 
  (13 * x^2 + 5 * x + 3) * (3 * x^3) = 39 * x^5 + 15 * x^4 + 9 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3615_361514


namespace NUMINAMATH_CALUDE_portfolio_annual_yield_is_correct_l3615_361591

structure Security where
  quantity : ℕ
  initialPrice : ℝ
  priceAfter180Days : ℝ

def Portfolio : List Security := [
  ⟨1000, 95.3, 98.6⟩,
  ⟨1000, 89.5, 93.4⟩,
  ⟨1000, 92.1, 96.2⟩,
  ⟨1, 100000, 104300⟩,
  ⟨1, 200000, 209420⟩,
  ⟨40, 3700, 3900⟩,
  ⟨500, 137, 142⟩
]

def calculateAnnualYield (portfolio : List Security) : ℝ :=
  sorry

theorem portfolio_annual_yield_is_correct :
  abs (calculateAnnualYield Portfolio - 9.21) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_portfolio_annual_yield_is_correct_l3615_361591


namespace NUMINAMATH_CALUDE_smallest_cookie_boxes_l3615_361524

theorem smallest_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(∃ (k : ℕ), 15 * m - 2 = 11 * k)) ∧ 
  (∃ (k : ℕ), 15 * n - 2 = 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_boxes_l3615_361524


namespace NUMINAMATH_CALUDE_mothers_age_l3615_361517

theorem mothers_age (person_age mother_age : ℕ) : 
  person_age = (2 * mother_age) / 5 →
  person_age + 10 = (mother_age + 10) / 2 →
  mother_age = 50 := by
sorry

end NUMINAMATH_CALUDE_mothers_age_l3615_361517


namespace NUMINAMATH_CALUDE_parallelogram_division_slope_l3615_361549

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ := (10, 30)
  v2 : ℝ × ℝ := (10, 80)
  v3 : ℝ × ℝ := (25, 125)
  v4 : ℝ × ℝ := (25, 75)

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- Predicate to check if a line divides a parallelogram into two congruent polygons -/
def divides_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- Theorem stating the slope of the line that divides the parallelogram -/
theorem parallelogram_division_slope (p : Parallelogram) (l : Line) :
  divides_into_congruent_polygons p l → l.slope = 24 / 7 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_division_slope_l3615_361549


namespace NUMINAMATH_CALUDE_sotka_not_divisible_by_nine_l3615_361513

/-- Represents a mapping of letters to digits -/
def LetterMapping := Char → Nat

/-- Checks if a number represented by a string is divisible by a given number -/
def isDivisible (s : String) (n : Nat) (mapping : LetterMapping) : Prop :=
  (s.toList.map mapping).sum % n = 0

/-- Ensures that each letter maps to a unique digit between 0 and 9 -/
def isValidMapping (mapping : LetterMapping) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → mapping c₁ ≠ mapping c₂ ∧ mapping c₁ < 10 ∧ mapping c₂ < 10

theorem sotka_not_divisible_by_nine :
  ∀ mapping : LetterMapping,
    isValidMapping mapping →
    isDivisible "ДЕВЯНОСТО" 90 mapping →
    isDivisible "ДЕВЯТКА" 9 mapping →
    mapping 'О' = 0 →
    ¬ isDivisible "СОТКА" 9 mapping :=
by
  sorry

end NUMINAMATH_CALUDE_sotka_not_divisible_by_nine_l3615_361513


namespace NUMINAMATH_CALUDE_consecutive_sum_39_l3615_361528

theorem consecutive_sum_39 (n : ℕ) : 
  n + (n + 1) = 39 → n = 19 := by
sorry

end NUMINAMATH_CALUDE_consecutive_sum_39_l3615_361528


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3615_361567

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 380) : 
  x + (x + 1) = 39 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3615_361567


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l3615_361560

/-- The area of wrapping paper required for a rectangular box --/
theorem wrapping_paper_area
  (w v h : ℝ)
  (h_pos : 0 < h)
  (w_pos : 0 < w)
  (v_pos : 0 < v)
  (v_lt_w : v < w) :
  let paper_width := 3 * v
  let paper_length := w
  paper_width * paper_length = 3 * w * v :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l3615_361560


namespace NUMINAMATH_CALUDE_correct_propositions_are_123_l3615_361590

-- Define the type for propositions
inductive GeometricProposition
  | frustum_def
  | frustum_edges
  | cone_def
  | hemisphere_rotation

-- Define a function to check if a proposition is correct
def is_correct_proposition (p : GeometricProposition) : Prop :=
  match p with
  | GeometricProposition.frustum_def => True
  | GeometricProposition.frustum_edges => True
  | GeometricProposition.cone_def => True
  | GeometricProposition.hemisphere_rotation => False

-- Define the set of all propositions
def all_propositions : Set GeometricProposition :=
  {GeometricProposition.frustum_def, GeometricProposition.frustum_edges, 
   GeometricProposition.cone_def, GeometricProposition.hemisphere_rotation}

-- Define the set of correct propositions
def correct_propositions : Set GeometricProposition :=
  {p ∈ all_propositions | is_correct_proposition p}

-- Theorem to prove
theorem correct_propositions_are_123 :
  correct_propositions = {GeometricProposition.frustum_def, 
                          GeometricProposition.frustum_edges, 
                          GeometricProposition.cone_def} := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_are_123_l3615_361590


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l3615_361578

theorem complex_on_real_axis (a : ℝ) : 
  let z : ℂ := (a - Complex.I) * (1 + Complex.I)
  (z.im = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l3615_361578


namespace NUMINAMATH_CALUDE_min_value_expression_l3615_361540

theorem min_value_expression (x : ℝ) : 
  (∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) - 200 ≥ (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200) → 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200 = -6680.25 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3615_361540
