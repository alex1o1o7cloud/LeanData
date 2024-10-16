import Mathlib

namespace NUMINAMATH_CALUDE_collective_purchase_equation_l2107_210787

theorem collective_purchase_equation (x y : ℤ) : 
  (8 * x - 3 = y) → (7 * x + 4 = y) := by
  sorry

end NUMINAMATH_CALUDE_collective_purchase_equation_l2107_210787


namespace NUMINAMATH_CALUDE_garbage_collection_average_l2107_210725

theorem garbage_collection_average (total_garbage : ℝ) 
  (h1 : total_garbage = 900) 
  (h2 : ∃ x : ℝ, total_garbage = x + x / 2) : 
  ∃ x : ℝ, x + x / 2 = total_garbage ∧ x = 600 :=
by sorry

end NUMINAMATH_CALUDE_garbage_collection_average_l2107_210725


namespace NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_error_percentage_division_vs_multiplication_proof_l2107_210724

theorem error_percentage_division_vs_multiplication : ℝ → Prop :=
  fun x =>
    let correct_result := 2 * x
    let incorrect_result := x / 10
    let error := correct_result - incorrect_result
    let percentage_error := (error / correct_result) * 100
    percentage_error = 95

-- The proof is omitted
theorem error_percentage_division_vs_multiplication_proof :
  ∀ x : ℝ, x ≠ 0 → error_percentage_division_vs_multiplication x :=
sorry

end NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_error_percentage_division_vs_multiplication_proof_l2107_210724


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l2107_210744

-- Define the set of real numbers less than 2
def lessThanTwo : Set ℝ := {x | x < 2}

-- State the theorem
theorem solution_set_absolute_value_inequality :
  {x : ℝ | |x - 2| > x - 2} = lessThanTwo := by
  sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l2107_210744


namespace NUMINAMATH_CALUDE_square_root_equals_arithmetic_square_root_l2107_210745

theorem square_root_equals_arithmetic_square_root (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x ∧ y = Real.sqrt x) ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_square_root_equals_arithmetic_square_root_l2107_210745


namespace NUMINAMATH_CALUDE_three_students_same_group_probability_l2107_210708

theorem three_students_same_group_probability 
  (total_groups : ℕ) 
  (student_count : ℕ) 
  (h1 : total_groups = 4) 
  (h2 : student_count ≥ 3) 
  (h3 : student_count % total_groups = 0) :
  (1 : ℚ) / (total_groups ^ 2) = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_three_students_same_group_probability_l2107_210708


namespace NUMINAMATH_CALUDE_ratio_problem_l2107_210731

theorem ratio_problem (a b : ℝ) (h1 : a / b = 5) (h2 : a = 65) : b = 13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2107_210731


namespace NUMINAMATH_CALUDE_sum_of_odd_periodic_function_l2107_210765

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x

theorem sum_of_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_periodic : has_period_4 f) 
  (h_f1 : f 1 = -1) : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_odd_periodic_function_l2107_210765


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l2107_210729

theorem reciprocal_sum_of_quadratic_roots : 
  ∀ x₁ x₂ : ℝ, 
  (x₁^2 + x₁ = 5*x₁ + 6) → 
  (x₂^2 + x₂ = 5*x₂ + 6) → 
  x₁ ≠ x₂ →
  (1/x₁ + 1/x₂ = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l2107_210729


namespace NUMINAMATH_CALUDE_shiela_paintings_distribution_l2107_210709

/-- Given Shiela has 18 paintings and each relative gets 9 paintings, 
    prove that she is giving paintings to 2 relatives. -/
theorem shiela_paintings_distribution (total_paintings : ℕ) (paintings_per_relative : ℕ) 
  (h1 : total_paintings = 18) 
  (h2 : paintings_per_relative = 9) : 
  total_paintings / paintings_per_relative = 2 := by
  sorry

end NUMINAMATH_CALUDE_shiela_paintings_distribution_l2107_210709


namespace NUMINAMATH_CALUDE_people_off_first_stop_l2107_210727

/-- Represents the number of people who got off at the first stop -/
def first_stop_off : ℕ := sorry

/-- The initial number of people on the bus -/
def initial_people : ℕ := 50

/-- The number of people who got off at the second stop -/
def second_stop_off : ℕ := 8

/-- The number of people who got on at the second stop -/
def second_stop_on : ℕ := 2

/-- The number of people who got off at the third stop -/
def third_stop_off : ℕ := 4

/-- The number of people who got on at the third stop -/
def third_stop_on : ℕ := 3

/-- The final number of people on the bus after the third stop -/
def final_people : ℕ := 28

theorem people_off_first_stop :
  initial_people - first_stop_off - (second_stop_off - second_stop_on) - (third_stop_off - third_stop_on) = final_people ∧
  first_stop_off = 15 := by sorry

end NUMINAMATH_CALUDE_people_off_first_stop_l2107_210727


namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l2107_210713

def f (x : ℝ) : ℝ := |x + 1|

theorem inequality_solution_and_function_property :
  (∀ x : ℝ, f (x + 8) ≥ 10 - f x ↔ x ≤ -10 ∨ x ≥ 0) ∧
  (∀ x y : ℝ, |x| > 1 → |y| < 1 → f y < |x| * f (y / x^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l2107_210713


namespace NUMINAMATH_CALUDE_two_folds_verify_square_l2107_210719

/-- A quadrilateral on a transparent sheet. -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A fold on the transparent sheet. -/
inductive Fold
  | PerpendicularBisector : Fold
  | Diagonal : Fold

/-- Function to check if a quadrilateral is a square after applying folds. -/
def isSquareAfterFolds (q : Quadrilateral) (folds : List Fold) : Prop :=
  sorry

/-- Theorem stating that two folds are necessary and sufficient to verify a square. -/
theorem two_folds_verify_square (q : Quadrilateral) :
  (∃ (folds : List Fold), folds.length = 2 ∧ isSquareAfterFolds q folds) ∧
  (∀ (folds : List Fold), folds.length < 2 → ¬isSquareAfterFolds q folds) :=
sorry

end NUMINAMATH_CALUDE_two_folds_verify_square_l2107_210719


namespace NUMINAMATH_CALUDE_product_statistics_l2107_210788

def product_ratings : List ℝ := [9.6, 10.1, 9.7, 9.8, 10.0, 9.7, 10.0, 9.8, 10.1, 10.2]

def sum_of_squares : ℝ := 98.048

def improvement : ℝ := 0.2

def is_first_class (rating : ℝ) : Prop := rating ≥ 10

theorem product_statistics :
  let n : ℕ := product_ratings.length
  let mean : ℝ := (product_ratings.sum) / n
  let variance : ℝ := sum_of_squares / n - mean ^ 2
  let new_mean : ℝ := mean + improvement
  let new_variance : ℝ := variance
  (mean = 9.9) ∧
  (variance = 0.038) ∧
  (new_mean = 10.1) ∧
  (new_variance = 0.038) :=
sorry

end NUMINAMATH_CALUDE_product_statistics_l2107_210788


namespace NUMINAMATH_CALUDE_pens_sold_l2107_210792

/-- Given Paul's initial and remaining number of pens, prove the number of pens sold -/
theorem pens_sold (initial_pens remaining_pens : ℕ) 
  (h1 : initial_pens = 42)
  (h2 : remaining_pens = 19) :
  initial_pens - remaining_pens = 23 := by
  sorry

end NUMINAMATH_CALUDE_pens_sold_l2107_210792


namespace NUMINAMATH_CALUDE_sin_alpha_terminal_side_l2107_210717

/-- Given a point P on the terminal side of angle α with coordinates (3a, 4a) where a < 0, prove that sin α = -4/5 -/
theorem sin_alpha_terminal_side (a : ℝ) (α : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (3 * a, 4 * a)
  (P.1 = 3 * a ∧ P.2 = 4 * a) → Real.sin α = -4/5 :=
by sorry

end NUMINAMATH_CALUDE_sin_alpha_terminal_side_l2107_210717


namespace NUMINAMATH_CALUDE_middle_number_is_four_or_five_l2107_210715

/-- Represents a triple of positive integers -/
structure IntTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Checks if the triple satisfies all given conditions -/
def satisfiesConditions (t : IntTriple) : Prop :=
  t.a < t.b ∧ t.b < t.c ∧ t.a + t.b + t.c = 15

/-- Represents the set of all possible triples satisfying the conditions -/
def possibleTriples : Set IntTriple :=
  {t : IntTriple | satisfiesConditions t}

/-- Casey cannot determine the other two numbers -/
def caseyUncertain (t : IntTriple) : Prop :=
  ∃ t' ∈ possibleTriples, t'.a = t.a ∧ t' ≠ t

/-- Tracy cannot determine the other two numbers -/
def tracyUncertain (t : IntTriple) : Prop :=
  ∃ t' ∈ possibleTriples, t'.c = t.c ∧ t' ≠ t

/-- Stacy cannot determine the other two numbers -/
def stacyUncertain (t : IntTriple) : Prop :=
  ∃ t' ∈ possibleTriples, t'.b = t.b ∧ t' ≠ t

/-- The main theorem stating that the middle number must be 4 or 5 -/
theorem middle_number_is_four_or_five :
  ∀ t ∈ possibleTriples,
    caseyUncertain t → tracyUncertain t → stacyUncertain t →
    t.b = 4 ∨ t.b = 5 :=
sorry

end NUMINAMATH_CALUDE_middle_number_is_four_or_five_l2107_210715


namespace NUMINAMATH_CALUDE_extra_bananas_l2107_210738

theorem extra_bananas (total_children absent_children original_bananas : ℕ) 
  (h1 : total_children = 640)
  (h2 : absent_children = 320)
  (h3 : original_bananas = 2) : 
  (total_children * original_bananas) / (total_children - absent_children) - original_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_extra_bananas_l2107_210738


namespace NUMINAMATH_CALUDE_work_completion_time_l2107_210723

/-- Given two workers A and B who complete a work together in a certain number of days,
    this function calculates the time it takes for them to complete the work together. -/
def time_to_complete (time_A : ℝ) (time_together : ℝ) : ℝ :=
  time_together

/-- Theorem stating that if A and B complete the work in 9 days together,
    and A alone can do the work in 18 days, then A and B together can complete
    the work in 9 days. -/
theorem work_completion_time (time_A : ℝ) (time_together : ℝ)
    (h1 : time_A = 18)
    (h2 : time_together = 9) :
    time_to_complete time_A time_together = 9 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l2107_210723


namespace NUMINAMATH_CALUDE_new_premium_calculation_l2107_210789

def calculate_new_premium (initial_premium : ℝ) (accident_increase_percent : ℝ) 
  (ticket_increase : ℝ) (late_payment_increase : ℝ) (num_accidents : ℕ) 
  (num_tickets : ℕ) (num_late_payments : ℕ) : ℝ :=
  initial_premium + 
  (initial_premium * accident_increase_percent * num_accidents : ℝ) +
  (ticket_increase * num_tickets) +
  (late_payment_increase * num_late_payments)

theorem new_premium_calculation :
  calculate_new_premium 125 0.12 7 15 2 4 3 = 228 := by
  sorry

end NUMINAMATH_CALUDE_new_premium_calculation_l2107_210789


namespace NUMINAMATH_CALUDE_add_1850_minutes_to_3_15pm_l2107_210766

/-- Represents a time of day in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Adds minutes to a given time and wraps around to the next day if necessary -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

/-- Converts a number of minutes to days, hours, and minutes -/
def minutesToDHM (m : Nat) : (Nat × Nat × Nat) :=
  sorry

theorem add_1850_minutes_to_3_15pm (start : Time) (h : start.hours = 15 ∧ start.minutes = 15) :
  let end_time := addMinutes start 1850
  end_time.hours = 22 ∧ end_time.minutes = 5 ∧ (minutesToDHM 1850).1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_add_1850_minutes_to_3_15pm_l2107_210766


namespace NUMINAMATH_CALUDE_amit_work_days_l2107_210733

theorem amit_work_days (ananthu_days : ℕ) (amit_worked : ℕ) (total_days : ℕ) :
  ananthu_days = 30 →
  amit_worked = 3 →
  total_days = 27 →
  ∃ (amit_days : ℕ),
    amit_days = 15 ∧
    (3 : ℝ) / amit_days + (total_days - amit_worked : ℝ) / ananthu_days = 1 :=
by sorry

end NUMINAMATH_CALUDE_amit_work_days_l2107_210733


namespace NUMINAMATH_CALUDE_senior_junior_ratio_l2107_210714

theorem senior_junior_ratio (j s : ℕ) (hj : j > 0) (hs : s > 0) : 
  (3 * j : ℚ) / 4 = (1 * s : ℚ) / 2 → (s : ℚ) / j = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_senior_junior_ratio_l2107_210714


namespace NUMINAMATH_CALUDE_Z_in_first_quadrant_l2107_210761

/-- A complex number is in the first quadrant if its real and imaginary parts are both positive. -/
def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

/-- The sum of two complex numbers (5+4i) and (-1+2i) -/
def Z : ℂ := Complex.mk 5 4 + Complex.mk (-1) 2

/-- Theorem: Z is located in the first quadrant of the complex plane -/
theorem Z_in_first_quadrant : is_in_first_quadrant Z := by
  sorry

end NUMINAMATH_CALUDE_Z_in_first_quadrant_l2107_210761


namespace NUMINAMATH_CALUDE_divisor_problem_l2107_210707

theorem divisor_problem (x : ℕ) : 
  (95 / x = 6 ∧ 95 % x = 5) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2107_210707


namespace NUMINAMATH_CALUDE_log_equation_holds_l2107_210750

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l2107_210750


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_vertices_l2107_210732

/-- Given two points (1,6) and (5,2) as adjacent vertices of a square, prove that the area of the square is 32. -/
theorem square_area_from_adjacent_vertices : 
  let p1 : ℝ × ℝ := (1, 6)
  let p2 : ℝ × ℝ := (5, 2)
  32 = (((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) : ℝ) := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_vertices_l2107_210732


namespace NUMINAMATH_CALUDE_between_negative_two_and_zero_l2107_210704

def numbers : Set ℝ := {3, 1, -3, -1}

theorem between_negative_two_and_zero :
  ∃ x ∈ numbers, -2 < x ∧ x < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_between_negative_two_and_zero_l2107_210704


namespace NUMINAMATH_CALUDE_multiply_by_point_nine_l2107_210735

theorem multiply_by_point_nine (x : ℝ) : 0.9 * x = 0.0063 → x = 0.007 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_point_nine_l2107_210735


namespace NUMINAMATH_CALUDE_mothers_daughters_ages_l2107_210740

theorem mothers_daughters_ages (mother_age daughter_age : ℕ) : 
  mother_age = 40 →
  daughter_age + 2 * mother_age = 95 →
  mother_age + 2 * daughter_age = 70 := by
  sorry

end NUMINAMATH_CALUDE_mothers_daughters_ages_l2107_210740


namespace NUMINAMATH_CALUDE_phone_charges_count_l2107_210769

def daily_mileages : List Nat := [135, 259, 159, 189]
def charge_interval : Nat := 106

theorem phone_charges_count : 
  (daily_mileages.sum / charge_interval : Nat) = 7 := by
  sorry

end NUMINAMATH_CALUDE_phone_charges_count_l2107_210769


namespace NUMINAMATH_CALUDE_third_term_base_l2107_210759

theorem third_term_base (x : ℝ) (some_number : ℝ) 
  (h1 : 625^(-x) + 25^(-2*x) + some_number^(-4*x) = 14)
  (h2 : x = 0.25) : 
  some_number = 125/1744 := by
sorry

end NUMINAMATH_CALUDE_third_term_base_l2107_210759


namespace NUMINAMATH_CALUDE_solution_set_of_f_greater_than_one_l2107_210795

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

-- State the theorem
theorem solution_set_of_f_greater_than_one :
  {x : ℝ | f x > 1} = Set.Ioo (2/3) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_greater_than_one_l2107_210795


namespace NUMINAMATH_CALUDE_area_ratio_of_triangles_l2107_210773

theorem area_ratio_of_triangles : 
  let mnp_sides : Fin 3 → ℝ := ![7, 24, 25]
  let qrs_sides : Fin 3 → ℝ := ![9, 12, 15]
  let mnp_area := (mnp_sides 0 * mnp_sides 1) / 2
  let qrs_area := (qrs_sides 0 * qrs_sides 1) / 2
  mnp_area / qrs_area = 14 / 9 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_of_triangles_l2107_210773


namespace NUMINAMATH_CALUDE_fraction_value_l2107_210793

theorem fraction_value (x y : ℝ) (h : (1 / x) + (1 / y) = 2) :
  (2 * x + 5 * x * y + 2 * y) / (x - 3 * x * y + y) = -9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2107_210793


namespace NUMINAMATH_CALUDE_number_of_pots_l2107_210758

/-- Given a collection of pots where each pot contains 71 flowers,
    and there are 10011 flowers in total, prove that there are 141 pots. -/
theorem number_of_pots (flowers_per_pot : ℕ) (total_flowers : ℕ) (h1 : flowers_per_pot = 71) (h2 : total_flowers = 10011) :
  total_flowers / flowers_per_pot = 141 := by
  sorry


end NUMINAMATH_CALUDE_number_of_pots_l2107_210758


namespace NUMINAMATH_CALUDE_alvez_family_children_count_l2107_210780

/-- Represents the Alvez family structure -/
structure AlvezFamily where
  num_children : ℕ
  mother_age : ℕ
  children_ages : Fin num_children → ℕ

/-- Given conditions of the Alvez family -/
def alvez_family_conditions (family : AlvezFamily) : Prop :=
  let total_members := family.num_children + 2
  let father_age := 50
  let total_age := family.mother_age + father_age + (Finset.sum (Finset.univ : Finset (Fin family.num_children)) family.children_ages)
  let mother_children_age := family.mother_age + (Finset.sum (Finset.univ : Finset (Fin family.num_children)) family.children_ages)
  (total_age / total_members = 22) ∧
  (mother_children_age / (family.num_children + 1) = 15)

/-- The theorem to be proved -/
theorem alvez_family_children_count :
  ∃ (family : AlvezFamily), alvez_family_conditions family ∧ family.num_children = 3 :=
sorry

end NUMINAMATH_CALUDE_alvez_family_children_count_l2107_210780


namespace NUMINAMATH_CALUDE_function_definition_l2107_210753

-- Define the function property
def is_function {A B : Type} (f : A → B) : Prop :=
  ∀ x : A, ∃! y : B, f x = y

-- State the theorem
theorem function_definition {A B : Type} (f : A → B) :
  is_function f ↔ ∀ x : A, ∃! y : B, y = f x :=
by sorry

end NUMINAMATH_CALUDE_function_definition_l2107_210753


namespace NUMINAMATH_CALUDE_group_bill_proof_l2107_210778

def restaurant_bill (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

theorem group_bill_proof (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ)
  (h1 : total_people = 13)
  (h2 : num_kids = 9)
  (h3 : adult_meal_cost = 7) :
  restaurant_bill total_people num_kids adult_meal_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_group_bill_proof_l2107_210778


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2107_210752

-- Problem 1
theorem problem_1 : 18 + (-12) + (-18) = -12 := by sorry

-- Problem 2
theorem problem_2 : (1 + 3 / 7) + (-(2 + 1 / 3)) + (2 + 4 / 7) + (-(1 + 2 / 3)) = 0 := by sorry

-- Problem 3
theorem problem_3 : (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2 := by sorry

-- Problem 4
theorem problem_4 : -(1 ^ 2023) - ((-2) ^ 3) - ((-2) * (-3)) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2107_210752


namespace NUMINAMATH_CALUDE_expression_simplification_l2107_210770

theorem expression_simplification (x y : ℝ) (h : y ≠ 0) :
  ((x + 3*y)^2 - (x + y)*(x - y)) / (2*y) = 3*x + 5*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2107_210770


namespace NUMINAMATH_CALUDE_pump_water_in_half_hour_l2107_210755

/-- Given a pump that moves 560 gallons of water per hour, 
    prove that it will move 280 gallons in 30 minutes. -/
theorem pump_water_in_half_hour (pump_rate : ℝ) (time : ℝ) : 
  pump_rate = 560 → time = 0.5 → pump_rate * time = 280 := by
  sorry

end NUMINAMATH_CALUDE_pump_water_in_half_hour_l2107_210755


namespace NUMINAMATH_CALUDE_soccer_substitution_remainder_l2107_210743

/-- Represents the number of ways to make substitutions in a soccer game -/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitute_players := total_players - starting_players
  let rec ways_for_n (n : ℕ) : ℕ :=
    if n = 0 then 1
    else starting_players * (substitute_players - n + 1) * ways_for_n (n - 1)
  (List.range (max_substitutions + 1)).map ways_for_n |> List.sum

/-- The main theorem stating the remainder of substitution ways when divided by 1000 -/
theorem soccer_substitution_remainder :
  substitution_ways 22 11 4 % 1000 = 25 := by
  sorry


end NUMINAMATH_CALUDE_soccer_substitution_remainder_l2107_210743


namespace NUMINAMATH_CALUDE_red_cube_possible_l2107_210720

/-- Represents a small cube with colored faces -/
structure SmallCube where
  blue_faces : Nat
  red_faces : Nat

/-- Represents the larger cube assembled from small cubes -/
structure LargeCube where
  small_cubes : List SmallCube
  visible_red_faces : Nat

/-- The theorem to be proved -/
theorem red_cube_possible 
  (cubes : List SmallCube) 
  (h1 : cubes.length = 8)
  (h2 : ∀ c ∈ cubes, c.blue_faces + c.red_faces = 6)
  (h3 : (cubes.map SmallCube.blue_faces).sum = 16)
  (h4 : ∃ lc : LargeCube, lc.small_cubes = cubes ∧ lc.visible_red_faces = 8) :
  ∃ lc : LargeCube, lc.small_cubes = cubes ∧ lc.visible_red_faces = 24 := by
  sorry

end NUMINAMATH_CALUDE_red_cube_possible_l2107_210720


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l2107_210747

/-- A function with specific symmetry and derivative properties -/
class SpecialFunction (f : ℝ → ℝ) where
  symmetric : ∀ x, f x = f (-2 - x)
  derivative_property : ∀ x, x < -1 → (x + 1) * (f x + (x + 1) * (deriv f) x) < 0

/-- The solution set of the inequality xf(x-1) > f(0) -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f (x - 1) > f 0}

/-- The theorem stating the solution set of the inequality -/
theorem solution_set_is_open_interval
  (f : ℝ → ℝ) [SpecialFunction f] :
  SolutionSet f = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l2107_210747


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2107_210757

theorem quadratic_roots_relation (b c : ℚ) : 
  (∃ r₁ r₂ : ℚ, r₁ ≠ r₂ ∧ 
    (∀ x : ℚ, x^2 + b*x + c = 0 ↔ x = r₁ ∨ x = r₂) ∧
    (∃ s₁ s₂ : ℚ, s₁ ≠ s₂ ∧ 
      (∀ x : ℚ, 3*x^2 - 5*x - 7 = 0 ↔ x = s₁ ∨ x = s₂) ∧
      r₁ = s₁ + 3 ∧ r₂ = s₂ + 3)) →
  c = 35/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2107_210757


namespace NUMINAMATH_CALUDE_smallest_term_of_sequence_l2107_210734

def a (n : ℕ+) : ℤ := n^2 - 9*n - 100

theorem smallest_term_of_sequence (n : ℕ+) :
  ∃ m : ℕ+, (m = 4 ∨ m = 5) ∧ ∀ k : ℕ+, a m ≤ a k :=
sorry

end NUMINAMATH_CALUDE_smallest_term_of_sequence_l2107_210734


namespace NUMINAMATH_CALUDE_route_time_difference_l2107_210763

/-- Represents the time added by a red light -/
def red_light_time : ℕ := 3

/-- Represents the number of stoplights on the first route -/
def num_stoplights : ℕ := 3

/-- Represents the time for the first route if all lights are green -/
def first_route_green_time : ℕ := 10

/-- Represents the time for the second route -/
def second_route_time : ℕ := 14

/-- Calculates the time for the first route when all lights are red -/
def first_route_red_time : ℕ := first_route_green_time + num_stoplights * red_light_time

/-- Proves that the difference between the first route with all red lights and the second route is 5 minutes -/
theorem route_time_difference : first_route_red_time - second_route_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_route_time_difference_l2107_210763


namespace NUMINAMATH_CALUDE_other_focus_coordinates_l2107_210711

/-- A hyperbola with given axes of symmetry and one focus on the y-axis -/
structure Hyperbola where
  x_axis : ℝ
  y_axis : ℝ
  focus_on_y_axis : ℝ × ℝ

/-- The other focus of the hyperbola -/
def other_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Theorem stating that the other focus has coordinates (-2, 2) -/
theorem other_focus_coordinates (h : Hyperbola) 
  (hx : h.x_axis = -1)
  (hy : h.y_axis = 2)
  (hf : h.focus_on_y_axis.1 = 0 ∧ h.focus_on_y_axis.2 = 2) :
  other_focus h = (-2, 2) := by sorry

end NUMINAMATH_CALUDE_other_focus_coordinates_l2107_210711


namespace NUMINAMATH_CALUDE_fraction_calculation_l2107_210777

theorem fraction_calculation : (2 / 8 : ℚ) + (4 / 16 : ℚ) * (3 / 9 : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2107_210777


namespace NUMINAMATH_CALUDE_stadium_seats_problem_l2107_210749

/-- Represents the number of seats in the n-th row -/
def a (n : ℕ) : ℕ := n + 1

/-- The total number of seats in the first n rows -/
def total_seats (n : ℕ) : ℕ := n * (n + 3) / 2

/-- The sum of the first n terms of the sequence a_n / (n(n+1)^2) -/
def S (n : ℕ) : ℚ := n / (n + 1)

theorem stadium_seats_problem :
  (total_seats 20 = 230) ∧ (S 20 = 20 / 21) := by
  sorry

end NUMINAMATH_CALUDE_stadium_seats_problem_l2107_210749


namespace NUMINAMATH_CALUDE_no_formula_matches_l2107_210706

def x : List ℕ := [1, 2, 3, 4, 5]
def y : List ℕ := [5, 15, 33, 61, 101]

def formula_a (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 - x + 1
def formula_b (x : ℕ) : ℕ := 3 * x^3 + x^2 + x + 1
def formula_c (x : ℕ) : ℕ := 2 * x^3 + x^2 + x + 1
def formula_d (x : ℕ) : ℕ := 2 * x^3 + x^2 + x - 1

theorem no_formula_matches : 
  (∃ i, List.get! x i ≠ 0 ∧ formula_a (List.get! x i) ≠ List.get! y i) ∧
  (∃ i, List.get! x i ≠ 0 ∧ formula_b (List.get! x i) ≠ List.get! y i) ∧
  (∃ i, List.get! x i ≠ 0 ∧ formula_c (List.get! x i) ≠ List.get! y i) ∧
  (∃ i, List.get! x i ≠ 0 ∧ formula_d (List.get! x i) ≠ List.get! y i) :=
by sorry

end NUMINAMATH_CALUDE_no_formula_matches_l2107_210706


namespace NUMINAMATH_CALUDE_sally_and_fred_onions_l2107_210775

/-- The number of onions Sally and Fred have after giving some to Sara -/
def remaining_onions (sally_onions fred_onions given_onions : ℕ) : ℕ :=
  sally_onions + fred_onions - given_onions

/-- Theorem stating that Sally and Fred have 10 onions after giving some to Sara -/
theorem sally_and_fred_onions :
  remaining_onions 5 9 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sally_and_fred_onions_l2107_210775


namespace NUMINAMATH_CALUDE_binary_1101_is_13_l2107_210798

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0 b

theorem binary_1101_is_13 :
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_is_13_l2107_210798


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l2107_210754

-- Define the parabola function
def f (x : ℝ) : ℝ := -(x + 1)^2 + 3

-- Define the theorem
theorem y1_greater_than_y2 :
  ∀ y₁ y₂ : ℝ, f 1 = y₁ → f 2 = y₂ → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l2107_210754


namespace NUMINAMATH_CALUDE_article_cost_l2107_210797

theorem article_cost (selling_price_high selling_price_low : ℝ) 
  (h1 : selling_price_high = 360)
  (h2 : selling_price_low = 340)
  (h3 : selling_price_high - selling_price_low = 20)
  (h4 : ∀ cost, 
    (selling_price_high - cost) = (selling_price_low - cost) * 1.05) :
  ∃ cost : ℝ, cost = 60 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l2107_210797


namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l2107_210785

/-- The units digit of a number in base 8 -/
def units_digit_base8 (n : ℕ) : ℕ := n % 8

/-- Addition in base 8 -/
def add_base8 (a b : ℕ) : ℕ := (a + b) % 8

theorem units_digit_sum_base8 :
  units_digit_base8 (add_base8 67 54) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l2107_210785


namespace NUMINAMATH_CALUDE_common_tangent_sum_l2107_210784

theorem common_tangent_sum (a b c : ℕ+) : 
  (∃ (x y : ℚ), y = x^2 + 12/5 ∧ (a : ℚ) * x + (b : ℚ) * y = c) ∧ 
  (∃ (x y : ℚ), x = y^2 + 99/10 ∧ (a : ℚ) * x + (b : ℚ) * y = c) ∧ 
  (∃ (m : ℚ), (a : ℚ) * x + (b : ℚ) * y = c ↔ y = m * x + (c / b : ℚ)) ∧
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 →
  a.val + b.val + c.val = 11 := by
sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l2107_210784


namespace NUMINAMATH_CALUDE_min_sum_squared_l2107_210768

theorem min_sum_squared (x₁ x₂ : ℝ) (h : x₁ * x₂ = 2013) : 
  (x₁ + x₂)^2 ≥ 8052 ∧ ∃ y₁ y₂ : ℝ, y₁ * y₂ = 2013 ∧ (y₁ + y₂)^2 = 8052 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squared_l2107_210768


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l2107_210772

theorem fraction_sum_inequality (α β a b : ℝ) (hα : α > 0) (hβ : β > 0)
  (ha : α ≤ a ∧ a ≤ β) (hb : α ≤ b ∧ b ≤ β) :
  b / a + a / b ≤ β / α + α / β ∧
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β) ∨ (a = β ∧ b = α)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l2107_210772


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l2107_210736

def vector_a : ℝ × ℝ := (3, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (-12, x - 4)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_imply_x_value :
  ∀ x : ℝ, parallel vector_a (vector_b x) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l2107_210736


namespace NUMINAMATH_CALUDE_russian_in_top_three_l2107_210751

structure ChessTournament where
  total_players : Nat
  russian_players : Nat
  foreign_players : Nat
  games_per_pair : Nat
  total_points : Nat
  russian_points : Nat
  foreign_points : Nat

def valid_tournament (t : ChessTournament) : Prop :=
  t.total_players = 11 ∧
  t.russian_players = 4 ∧
  t.foreign_players = 7 ∧
  t.games_per_pair = 2 ∧
  t.total_points = t.total_players * (t.total_players - 1) ∧
  t.russian_points = t.foreign_points ∧
  t.russian_points + t.foreign_points = t.total_points

theorem russian_in_top_three (t : ChessTournament) (h : valid_tournament t) :
  ∃ (top_three : Finset Nat) (russian : Nat),
    top_three.card = 3 ∧
    russian ∈ top_three ∧
    russian ≤ t.russian_players :=
  sorry

end NUMINAMATH_CALUDE_russian_in_top_three_l2107_210751


namespace NUMINAMATH_CALUDE_fish_to_rice_value_l2107_210760

/-- Represents the exchange rate between fish and bread -/
def fish_to_bread : ℚ := 3 / 5

/-- Represents the exchange rate between bread and rice -/
def bread_to_rice : ℚ := 5 / 2

/-- Theorem stating that one fish is worth 3/2 bags of rice -/
theorem fish_to_rice_value : 
  (fish_to_bread * bread_to_rice)⁻¹ = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_fish_to_rice_value_l2107_210760


namespace NUMINAMATH_CALUDE_M_value_l2107_210742

def M : ℕ → ℕ 
  | 0 => 0
  | n + 1 => (2*n + 2)^2 + (2*n)^2 - (2*n - 2)^2 + M n

theorem M_value : M 25 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_M_value_l2107_210742


namespace NUMINAMATH_CALUDE_fruit_problem_max_a_l2107_210728

/-- Represents the fruit purchase and sale problem -/
def FruitProblem (totalCost totalWeight cherryPrice cantaloupePrice : ℝ)
  (secondTotalWeight secondMaxCost minProfit : ℝ)
  (cherrySellingPrice cantaloupeSellingPrice : ℝ) :=
  ∀ (a : ℕ),
    let n := (secondMaxCost - 6 * secondTotalWeight) / 29
    (35 * n + 6 * (secondTotalWeight - n) ≤ secondMaxCost) ∧
    (20 * (n - a) + 4 * (secondTotalWeight - n - 2 * a) ≥ minProfit) →
    a ≤ 35

/-- The maximum value of a in the fruit problem is 35 -/
theorem fruit_problem_max_a :
  FruitProblem 9160 560 35 6 300 5280 2120 55 10 :=
sorry

end NUMINAMATH_CALUDE_fruit_problem_max_a_l2107_210728


namespace NUMINAMATH_CALUDE_smallest_base_for_inequality_l2107_210721

theorem smallest_base_for_inequality (k : ℕ) (h : k = 6) : 
  ∀ b : ℕ, b > 0 → b ≤ 4 ↔ b^16 ≤ 64^k :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_inequality_l2107_210721


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l2107_210701

theorem alcohol_solution_proof (initial_volume : ℝ) (initial_percentage : ℝ) 
  (target_percentage : ℝ) (added_alcohol : ℝ) : 
  initial_volume = 100 ∧ 
  initial_percentage = 0.20 ∧ 
  target_percentage = 0.30 ∧
  added_alcohol = 14.2857 →
  (initial_volume * initial_percentage + added_alcohol) / (initial_volume + added_alcohol) = target_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l2107_210701


namespace NUMINAMATH_CALUDE_common_external_tangent_intercept_l2107_210783

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The common external tangent problem --/
theorem common_external_tangent_intercept 
  (c1 : Circle) 
  (c2 : Circle) 
  (h1 : c1.center = (3, 2)) 
  (h2 : c1.radius = 5) 
  (h3 : c2.center = (12, 10)) 
  (h4 : c2.radius = 7) :
  ∃ (m b : ℝ), m > 0 ∧ 
    (∀ (x y : ℝ), y = m * x + b → 
      ((x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∨
       (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2)) ∧
    b = -313/17 := by
  sorry

end NUMINAMATH_CALUDE_common_external_tangent_intercept_l2107_210783


namespace NUMINAMATH_CALUDE_volleyball_starters_count_l2107_210756

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The number of ways to choose 6 starters from 15 players, with at least one of 3 specific players -/
def volleyball_starters : ℕ :=
  binomial 15 6 - binomial 12 6

theorem volleyball_starters_count : volleyball_starters = 4081 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_starters_count_l2107_210756


namespace NUMINAMATH_CALUDE_parts_per_day_calculation_l2107_210705

/-- The number of parts initially planned per day -/
def initial_parts_per_day : ℕ := 142

/-- The number of days with initial production rate -/
def initial_days : ℕ := 3

/-- The increase in parts per day after the initial days -/
def increase_in_parts : ℕ := 5

/-- The total number of parts produced -/
def total_parts : ℕ := 675

/-- The number of extra parts produced compared to the plan -/
def extra_parts : ℕ := 100

/-- The number of days after the initial period -/
def additional_days : ℕ := 1

theorem parts_per_day_calculation :
  initial_parts_per_day * initial_days + 
  (initial_parts_per_day + increase_in_parts) * additional_days = 
  total_parts - extra_parts :=
by sorry

#check parts_per_day_calculation

end NUMINAMATH_CALUDE_parts_per_day_calculation_l2107_210705


namespace NUMINAMATH_CALUDE_base8_246_to_base10_l2107_210796

/-- Converts a base 8 number to base 10 -/
def base8_to_base10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

/-- The base 10 representation of 246₈ is 166 -/
theorem base8_246_to_base10 : base8_to_base10 2 4 6 = 166 := by
  sorry

end NUMINAMATH_CALUDE_base8_246_to_base10_l2107_210796


namespace NUMINAMATH_CALUDE_unique_solution_when_k_zero_no_unique_solution_when_k_nonzero_k_zero_only_unique_solution_l2107_210781

/-- The equation has exactly one solution when k = 0 -/
theorem unique_solution_when_k_zero : ∃! x : ℝ, (x + 3) / (0 * x - 2) = x :=
sorry

/-- For any k ≠ 0, the equation has either no solution or more than one solution -/
theorem no_unique_solution_when_k_nonzero (k : ℝ) (hk : k ≠ 0) :
  ¬(∃! x : ℝ, (x + 3) / (k * x - 2) = x) :=
sorry

/-- k = 0 is the only value for which the equation has exactly one solution -/
theorem k_zero_only_unique_solution :
  ∀ k : ℝ, (∃! x : ℝ, (x + 3) / (k * x - 2) = x) ↔ k = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_when_k_zero_no_unique_solution_when_k_nonzero_k_zero_only_unique_solution_l2107_210781


namespace NUMINAMATH_CALUDE_fixed_salary_is_1000_l2107_210722

/-- Calculates the commission for the old scheme -/
def old_commission (sales : ℝ) : ℝ := 0.05 * sales

/-- Calculates the commission for the new scheme -/
def new_commission (sales : ℝ) : ℝ := 0.025 * (sales - 4000)

/-- Theorem: The fixed salary in the new scheme is 1000 -/
theorem fixed_salary_is_1000 (total_sales : ℝ) (fixed_salary : ℝ) :
  total_sales = 12000 →
  fixed_salary + new_commission total_sales = old_commission total_sales + 600 →
  fixed_salary = 1000 := by
  sorry

#check fixed_salary_is_1000

end NUMINAMATH_CALUDE_fixed_salary_is_1000_l2107_210722


namespace NUMINAMATH_CALUDE_g_of_50_l2107_210762

/-- A function satisfying the given property for all positive real numbers -/
def SatisfyingFunction (g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → x * g y - y * g x = g (x / y) + x - y

/-- The theorem stating that any function satisfying the property has g(50) = -24.5 -/
theorem g_of_50 (g : ℝ → ℝ) (h : SatisfyingFunction g) : g 50 = -24.5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_50_l2107_210762


namespace NUMINAMATH_CALUDE_fiftieth_term_is_199_l2107_210718

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
theorem fiftieth_term_is_199 :
  arithmetic_sequence 3 4 50 = 199 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_199_l2107_210718


namespace NUMINAMATH_CALUDE_seongjun_has_500_ttakji_l2107_210771

/-- The number of ttakji Seongjun has -/
def seongjun_ttakji : ℕ := sorry

/-- The number of ttakji Seunga has -/
def seunga_ttakji : ℕ := 100

/-- The relationship between Seongjun's and Seunga's ttakji -/
axiom ttakji_relationship : (3 / 4 : ℚ) * seongjun_ttakji - 25 = 7 * (seunga_ttakji - 50)

theorem seongjun_has_500_ttakji : seongjun_ttakji = 500 := by sorry

end NUMINAMATH_CALUDE_seongjun_has_500_ttakji_l2107_210771


namespace NUMINAMATH_CALUDE_largest_factorial_with_100_zeros_l2107_210703

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- The largest positive integer n such that n! ends with exactly 100 zeros -/
theorem largest_factorial_with_100_zeros : 
  (∀ m : ℕ, m > 409 → trailingZeros m > 100) ∧ 
  trailingZeros 409 = 100 :=
sorry

end NUMINAMATH_CALUDE_largest_factorial_with_100_zeros_l2107_210703


namespace NUMINAMATH_CALUDE_four_students_three_events_outcomes_l2107_210782

/-- The number of possible outcomes for champions in a competition --/
def championOutcomes (students : ℕ) (events : ℕ) : ℕ :=
  students ^ events

/-- Theorem: Given 4 students and 3 events, the number of possible outcomes for champions is 64 --/
theorem four_students_three_events_outcomes :
  championOutcomes 4 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_events_outcomes_l2107_210782


namespace NUMINAMATH_CALUDE_min_sphere_surface_area_l2107_210700

theorem min_sphere_surface_area (a b c : ℝ) (h1 : a * b * c = 4) (h2 : a * b = 1) :
  let r := (3 * Real.sqrt 2) / 2
  4 * Real.pi * r^2 = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_min_sphere_surface_area_l2107_210700


namespace NUMINAMATH_CALUDE_well_digging_rate_l2107_210799

/-- Calculates the rate per cubic meter for digging a cylindrical well -/
theorem well_digging_rate (depth : ℝ) (diameter : ℝ) (total_cost : ℝ) : 
  depth = 14 →
  diameter = 3 →
  total_cost = 1583.3626974092558 →
  ∃ (rate : ℝ), abs (rate - 15.993) < 0.001 ∧ 
    rate = total_cost / (Real.pi * (diameter / 2)^2 * depth) := by
  sorry

end NUMINAMATH_CALUDE_well_digging_rate_l2107_210799


namespace NUMINAMATH_CALUDE_parabola_translation_l2107_210726

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = x^2 --/
def original_parabola : Parabola := { a := 1, b := 0, c := 0 }

/-- Translates a parabola vertically --/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + d }

/-- Translates a parabola horizontally --/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * d + p.b, c := p.a * d^2 - p.b * d + p.c }

/-- The resulting parabola after translations --/
def result_parabola : Parabola :=
  translate_horizontal (translate_vertical original_parabola 3) 5

theorem parabola_translation :
  result_parabola.a = 1 ∧
  result_parabola.b = -10 ∧
  result_parabola.c = 28 := by
  sorry

#check parabola_translation

end NUMINAMATH_CALUDE_parabola_translation_l2107_210726


namespace NUMINAMATH_CALUDE_object_properties_l2107_210737

-- Define the possible colors
inductive Color
| Red
| Blue
| Green

-- Define the shape property
structure Object where
  color : Color
  isRound : Bool

-- Define the conditions
axiom condition1 (obj : Object) : obj.isRound → (obj.color = Color.Red ∨ obj.color = Color.Blue)
axiom condition2 (obj : Object) : ¬obj.isRound → (obj.color ≠ Color.Red ∧ obj.color ≠ Color.Green)
axiom condition3 (obj : Object) : (obj.color = Color.Blue ∨ obj.color = Color.Green) → obj.isRound

-- Theorem to prove
theorem object_properties (obj : Object) : 
  obj.isRound ∧ (obj.color = Color.Red ∨ obj.color = Color.Blue) :=
by sorry

end NUMINAMATH_CALUDE_object_properties_l2107_210737


namespace NUMINAMATH_CALUDE_square_overlap_ratio_l2107_210791

theorem square_overlap_ratio (a b : ℝ) 
  (h1 : 0.52 * a^2 = a^2 - (a^2 - 0.73 * b^2)) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  a / b = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_square_overlap_ratio_l2107_210791


namespace NUMINAMATH_CALUDE_locus_P_is_correct_l2107_210739

/-- The locus of points P that are the second intersection of line OM and circle OAN,
    where O is the center of a circle with radius r, A(c, 0) is a point on its diameter,
    and M and N are symmetrical points on the circle with respect to OA. -/
def locus_P (r c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; (x^2 + y^2 - 2*c*x)^2 - r^2*(x^2 + y^2) = 0}

/-- Theorem stating that the locus_P is the correct description of the geometric locus. -/
theorem locus_P_is_correct (r c : ℝ) (hr : r > 0) (hc : c ≠ 0) :
  ∀ p : ℝ × ℝ, p ∈ locus_P r c ↔ 
    ∃ (m n : ℝ × ℝ),
      (∀ x y, (x, y) = m → x^2 + y^2 = r^2) ∧
      (∀ x y, (x, y) = n → x^2 + y^2 = r^2) ∧
      (∃ t, m.1 = t * n.1 ∧ m.2 = -t * n.2) ∧
      (∃ s, p = (s * m.1, s * m.2)) ∧
      (∃ u v, p.1^2 + p.2^2 + 2*u*p.1 + 2*v*p.2 = 0 ∧
              c^2 + 2*u*c = 0 ∧
              0^2 + 0^2 + 2*u*0 + 2*v*0 = 0) :=
by sorry

end NUMINAMATH_CALUDE_locus_P_is_correct_l2107_210739


namespace NUMINAMATH_CALUDE_remainder_of_sum_is_zero_l2107_210790

-- Define the arithmetic sequence
def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the arithmetic sequence
def sumArithmeticSequence (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

-- Theorem statement
theorem remainder_of_sum_is_zero :
  let a₁ := 3
  let aₙ := 309
  let d := 6
  let n := (aₙ - a₁) / d + 1
  (sumArithmeticSequence a₁ aₙ n) % 6 = 0 := by
    sorry

end NUMINAMATH_CALUDE_remainder_of_sum_is_zero_l2107_210790


namespace NUMINAMATH_CALUDE_sin_cos_sum_special_angles_l2107_210716

theorem sin_cos_sum_special_angles :
  Real.sin (36 * π / 180) * Real.cos (24 * π / 180) + 
  Real.cos (36 * π / 180) * Real.sin (156 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_special_angles_l2107_210716


namespace NUMINAMATH_CALUDE_additional_distance_for_average_speed_l2107_210741

def initial_distance : ℝ := 20
def initial_speed : ℝ := 25
def second_speed : ℝ := 40
def desired_average_speed : ℝ := 35

theorem additional_distance_for_average_speed :
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = desired_average_speed ∧
    additional_distance = 64 := by
  sorry

end NUMINAMATH_CALUDE_additional_distance_for_average_speed_l2107_210741


namespace NUMINAMATH_CALUDE_open_box_volume_l2107_210794

theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_square_side : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 38)
  (h3 : cut_square_side = 8) :
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 5632 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l2107_210794


namespace NUMINAMATH_CALUDE_initial_number_proof_l2107_210748

theorem initial_number_proof : 
  ∃ x : ℝ, (3 * (2 * x + 9) = 81) ∧ (x = 9) := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l2107_210748


namespace NUMINAMATH_CALUDE_gcd_228_1995_l2107_210710

theorem gcd_228_1995 : Nat.gcd 228 1995 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l2107_210710


namespace NUMINAMATH_CALUDE_bank_balance_after_two_years_l2107_210779

/-- The amount of money in a bank account after a given number of years,
    given an initial amount and an annual interest rate. -/
def bank_balance (initial_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + interest_rate) ^ years

/-- Theorem stating that $100 invested for 2 years at 10% annual interest results in $121 -/
theorem bank_balance_after_two_years :
  bank_balance 100 0.1 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_bank_balance_after_two_years_l2107_210779


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2107_210746

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2107_210746


namespace NUMINAMATH_CALUDE_crease_length_in_folded_rectangle_l2107_210767

/-- Represents a folded rectangle with given dimensions and fold properties -/
structure FoldedRectangle where
  width : ℝ
  fold_distance : ℝ
  crease_length : ℝ
  fold_angle : ℝ

/-- Theorem stating the crease length in a specific folded rectangle configuration -/
theorem crease_length_in_folded_rectangle (r : FoldedRectangle) 
  (h1 : r.width = 8)
  (h2 : r.fold_distance = 2)
  (h3 : Real.tan r.fold_angle = 3) : 
  r.crease_length = 2/3 := by
  sorry

#check crease_length_in_folded_rectangle

end NUMINAMATH_CALUDE_crease_length_in_folded_rectangle_l2107_210767


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l2107_210764

theorem sqrt_D_irrational (a : ℤ) : 
  let b : ℤ := a + 2
  let c : ℤ := a^2 + b
  let D : ℤ := a^2 + b^2 + c^2
  Irrational (Real.sqrt D) := by
sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l2107_210764


namespace NUMINAMATH_CALUDE_pizza_slices_left_l2107_210730

theorem pizza_slices_left (total_slices : ℕ) (people : ℕ) (slices_per_person : ℕ) :
  total_slices = 16 →
  people = 6 →
  slices_per_person = 2 →
  total_slices - (people * slices_per_person) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l2107_210730


namespace NUMINAMATH_CALUDE_total_vegetables_bought_l2107_210786

/-- The number of vegetables bought by Marcel and Dale -/
def total_vegetables (marcel_corn : ℕ) (dale_corn : ℕ) (marcel_potatoes : ℕ) (dale_potatoes : ℕ) : ℕ :=
  marcel_corn + dale_corn + marcel_potatoes + dale_potatoes

/-- Theorem stating the total number of vegetables bought by Marcel and Dale -/
theorem total_vegetables_bought :
  ∃ (marcel_corn marcel_potatoes dale_potatoes : ℕ),
    marcel_corn = 10 ∧
    marcel_potatoes = 4 ∧
    dale_potatoes = 8 ∧
    total_vegetables marcel_corn (marcel_corn / 2) marcel_potatoes dale_potatoes = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetables_bought_l2107_210786


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l2107_210774

theorem triangle_area_inequality (a b c α β γ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α = 2 * Real.sqrt (b * c))
  (hβ : β = 2 * Real.sqrt (a * c))
  (hγ : γ = 2 * Real.sqrt (a * b)) :
  a / α + b / β + c / γ ≥ 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l2107_210774


namespace NUMINAMATH_CALUDE_triangle_area_approx_l2107_210702

/-- The area of a triangle with sides 35 cm, 23 cm, and 41 cm is approximately 402.65 cm² --/
theorem triangle_area_approx (a b c : ℝ) (ha : a = 35) (hb : b = 23) (hc : c = 41) :
  ∃ (area : ℝ), abs (area - ((a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) / 2) - 402.65) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_approx_l2107_210702


namespace NUMINAMATH_CALUDE_trivia_team_groups_l2107_210712

theorem trivia_team_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) 
  (h1 : total_students = 58)
  (h2 : not_picked = 10)
  (h3 : students_per_group = 6) :
  (total_students - not_picked) / students_per_group = 8 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l2107_210712


namespace NUMINAMATH_CALUDE_zongzi_pricing_and_max_purchase_l2107_210776

/-- Represents the zongzi types -/
inductive ZongziType
| A
| B

/-- Represents the price and quantity information for zongzi -/
structure ZongziInfo where
  type : ZongziType
  amount_spent : ℝ
  quantity : ℕ

/-- Theorem for zongzi pricing and maximum purchase -/
theorem zongzi_pricing_and_max_purchase 
  (info_A : ZongziInfo) 
  (info_B : ZongziInfo) 
  (total_zongzi : ℕ) 
  (max_total_amount : ℝ) :
  info_A.type = ZongziType.A →
  info_B.type = ZongziType.B →
  info_A.amount_spent = 1200 →
  info_B.amount_spent = 800 →
  info_B.quantity = info_A.quantity + 50 →
  info_A.amount_spent / info_A.quantity = 2 * (info_B.amount_spent / info_B.quantity) →
  total_zongzi = 200 →
  max_total_amount = 1150 →
  ∃ (unit_price_A unit_price_B : ℝ) (max_quantity_A : ℕ),
    unit_price_A = 8 ∧
    unit_price_B = 4 ∧
    max_quantity_A = 87 ∧
    unit_price_A * max_quantity_A + unit_price_B * (total_zongzi - max_quantity_A) ≤ max_total_amount ∧
    ∀ (quantity_A : ℕ), 
      quantity_A > max_quantity_A →
      unit_price_A * quantity_A + unit_price_B * (total_zongzi - quantity_A) > max_total_amount :=
by sorry

end NUMINAMATH_CALUDE_zongzi_pricing_and_max_purchase_l2107_210776
