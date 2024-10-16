import Mathlib

namespace NUMINAMATH_CALUDE_square_root_divided_by_18_l942_94225

theorem square_root_divided_by_18 : Real.sqrt 5184 / 18 = 4 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_18_l942_94225


namespace NUMINAMATH_CALUDE_clock_right_angles_five_days_l942_94278

/-- Represents the number of times clock hands are at right angles in a given number of days -/
def clockRightAngles (days : ℕ) : ℕ :=
  let intervalsPerClockFace : ℕ := 12
  let degreesPerInterval : ℕ := 360 / intervalsPerClockFace
  let minutesForRightAngle : ℕ := 15
  let rightAnglesPerTwelveHours : ℕ := 22
  let hoursPerDay : ℕ := 24
  let rightAnglesPerDay : ℕ := rightAnglesPerTwelveHours * 2
  rightAnglesPerDay * days

/-- Theorem stating that the hands of a clock are at right angles 220 times in 5 days -/
theorem clock_right_angles_five_days :
  clockRightAngles 5 = 220 := by
  sorry

end NUMINAMATH_CALUDE_clock_right_angles_five_days_l942_94278


namespace NUMINAMATH_CALUDE_major_premise_is_false_l942_94257

theorem major_premise_is_false : ¬ ∀ (a : ℝ) (n : ℕ), (a^(1/n : ℝ))^n = a := by sorry

end NUMINAMATH_CALUDE_major_premise_is_false_l942_94257


namespace NUMINAMATH_CALUDE_carson_carpool_expense_l942_94210

/-- Represents the carpool scenario with given parameters --/
structure CarpoolScenario where
  num_friends : Nat
  one_way_miles : Nat
  gas_price : Rat
  miles_per_gallon : Nat
  days_per_week : Nat
  weeks_per_month : Nat

/-- Calculates the monthly gas expense per person for a given carpool scenario --/
def monthly_gas_expense_per_person (scenario : CarpoolScenario) : Rat :=
  let total_miles := 2 * scenario.one_way_miles * scenario.days_per_week * scenario.weeks_per_month
  let total_gallons := total_miles / scenario.miles_per_gallon
  let total_cost := total_gallons * scenario.gas_price
  total_cost / scenario.num_friends

/-- The given carpool scenario --/
def carson_carpool : CarpoolScenario :=
  { num_friends := 5
  , one_way_miles := 21
  , gas_price := 5/2
  , miles_per_gallon := 30
  , days_per_week := 5
  , weeks_per_month := 4
  }

/-- Theorem stating that the monthly gas expense per person for Carson's carpool is $14 --/
theorem carson_carpool_expense :
  monthly_gas_expense_per_person carson_carpool = 14 := by
  sorry


end NUMINAMATH_CALUDE_carson_carpool_expense_l942_94210


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l942_94231

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + 2 * a 6 + a 10 = 120) →
  (a 3 + a 9 = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l942_94231


namespace NUMINAMATH_CALUDE_basketball_weight_proof_l942_94202

/-- The weight of one basketball in pounds -/
def basketball_weight : ℚ := 125 / 9

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℚ := 25

theorem basketball_weight_proof :
  (9 : ℚ) * basketball_weight = (5 : ℚ) * bicycle_weight ∧
  (3 : ℚ) * bicycle_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_basketball_weight_proof_l942_94202


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l942_94271

/-- Calculates the number of sampled students within a given interval using systematic sampling -/
def sampledStudentsInInterval (totalStudents : ℕ) (sampleSize : ℕ) (intervalStart : ℕ) (intervalEnd : ℕ) : ℕ :=
  let intervalSize := intervalEnd - intervalStart + 1
  let samplingInterval := totalStudents / sampleSize
  intervalSize / samplingInterval

theorem systematic_sampling_theorem :
  sampledStudentsInInterval 1221 37 496 825 = 10 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l942_94271


namespace NUMINAMATH_CALUDE_inequality_proof_l942_94296

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l942_94296


namespace NUMINAMATH_CALUDE_reciprocal_of_two_l942_94235

theorem reciprocal_of_two (m : ℚ) : m - 3 = 1 / 2 → m = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_l942_94235


namespace NUMINAMATH_CALUDE_chalk_per_box_l942_94245

def total_chalk : ℕ := 3484
def full_boxes : ℕ := 194

theorem chalk_per_box : total_chalk / full_boxes = 18 := by
  sorry

end NUMINAMATH_CALUDE_chalk_per_box_l942_94245


namespace NUMINAMATH_CALUDE_equation_holds_for_all_y_l942_94284

theorem equation_holds_for_all_y (x : ℚ) : 
  (∀ y : ℚ, 8 * x * y - 12 * y + 2 * x - 3 = 0) ↔ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_y_l942_94284


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l942_94265

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 12 ∧ b = 16 ∧ c^2 = a^2 + b^2 → c = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l942_94265


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l942_94224

theorem smallest_sum_of_a_and_b (a b : ℝ) : 
  a > 0 → b > 0 → 
  ((3 * a)^2 ≥ 16 * b) → 
  ((4 * b)^2 ≥ 12 * a) → 
  a + b ≥ 70/3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l942_94224


namespace NUMINAMATH_CALUDE_inequality_solution_set_l942_94232

theorem inequality_solution_set (x : ℝ) :
  |x + 3| - |2*x - 1| < x/2 + 1 ↔ x < -2/5 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l942_94232


namespace NUMINAMATH_CALUDE_sets_equality_implies_coefficients_l942_94244

def A : Set ℝ := {-1, 3}

def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem sets_equality_implies_coefficients (a b : ℝ) : 
  A = B a b → a = -2 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_implies_coefficients_l942_94244


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_ferris_wheel_capacity_proof_l942_94277

theorem ferris_wheel_capacity (small_seats large_seats : ℕ) 
  (large_seat_capacity : ℕ) (total_large_capacity : ℕ) : Prop :=
  small_seats = 3 ∧ 
  large_seats = 7 ∧ 
  large_seat_capacity = 12 ∧
  total_large_capacity = 84 →
  ¬∃ (small_seat_capacity : ℕ), 
    ∀ (total_capacity : ℕ), 
      total_capacity = small_seats * small_seat_capacity + total_large_capacity

theorem ferris_wheel_capacity_proof : 
  ferris_wheel_capacity 3 7 12 84 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_ferris_wheel_capacity_proof_l942_94277


namespace NUMINAMATH_CALUDE_product_sequence_l942_94233

theorem product_sequence (seq : List ℕ) : 
  (∀ i, i + 3 < seq.length → seq[i]! * seq[i+1]! * seq[i+2]! * seq[i+3]! = 120) →
  (∃ i j k, i < j ∧ j < k ∧ k < seq.length ∧ seq[i]! = 2 ∧ seq[j]! = 4 ∧ seq[k]! = 3) →
  (∃ x, x ∈ seq ∧ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_product_sequence_l942_94233


namespace NUMINAMATH_CALUDE_reciprocal_not_always_plus_minus_one_l942_94206

theorem reciprocal_not_always_plus_minus_one : 
  ¬ (∀ x : ℝ, x ≠ 0 → (1 / x = 1 ∨ 1 / x = -1)) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_not_always_plus_minus_one_l942_94206


namespace NUMINAMATH_CALUDE_common_root_exists_polynomial_common_root_l942_94242

def coefficients : Finset Int := {-7, 4, -3, 6}

theorem common_root_exists (a b c d : Int) 
  (h : {a, b, c, d} = coefficients) : 
  (a : ℝ) + b + c + d = 0 := by sorry

theorem polynomial_common_root (a b c d : Int) 
  (h : {a, b, c, d} = coefficients) :
  ∃ (x : ℝ), a * x^3 + b * x^2 + c * x + d = 0 := by sorry

end NUMINAMATH_CALUDE_common_root_exists_polynomial_common_root_l942_94242


namespace NUMINAMATH_CALUDE_conditional_probability_suitable_joint_structure_l942_94205

/-- The probability of a child having a suitable joint structure given that they have a suitable physique -/
theorem conditional_probability_suitable_joint_structure 
  (total : ℕ) 
  (physique : ℕ) 
  (joint : ℕ) 
  (both : ℕ) 
  (h_total : total = 20)
  (h_physique : physique = 4)
  (h_joint : joint = 5)
  (h_both : both = 2) :
  (both : ℚ) / physique = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_conditional_probability_suitable_joint_structure_l942_94205


namespace NUMINAMATH_CALUDE_asymptotic_necessary_not_sufficient_l942_94221

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the asymptotic line equation
def asymptotic_line (a b x y : ℝ) : Prop := y = (b/a) * x ∨ y = -(b/a) * x

-- Theorem stating that the asymptotic line is a necessary but not sufficient condition for the hyperbola
theorem asymptotic_necessary_not_sufficient (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) :
  (∀ x y, hyperbola a b x y → asymptotic_line a b x y) ∧
  (∃ x y, asymptotic_line a b x y ∧ ¬hyperbola a b x y) :=
sorry

end NUMINAMATH_CALUDE_asymptotic_necessary_not_sufficient_l942_94221


namespace NUMINAMATH_CALUDE_horner_v1_value_l942_94269

/-- Horner's method for polynomial evaluation -/
def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = 0.5x^5 + 4x^4 - 3x^2 + x - 1 -/
def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

theorem horner_v1_value :
  let x : ℝ := 3
  let v0 : ℝ := 0.5
  let v1 : ℝ := horner_step v0 x 4
  v1 = 5.5 := by sorry

end NUMINAMATH_CALUDE_horner_v1_value_l942_94269


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l942_94268

/-- The greatest exponent x such that 3^x divides 22! is 9 -/
theorem greatest_power_of_three_in_factorial : 
  (∃ x : ℕ, x = 9 ∧ 
    (∀ y : ℕ, 3^y ∣ Nat.factorial 22 → y ≤ x) ∧
    (3^x ∣ Nat.factorial 22)) := by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l942_94268


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l942_94263

def polynomial (p q : ℚ) (x : ℚ) : ℚ :=
  x^6 - x^5 + x^4 - p*x^3 + q*x^2 + 6*x - 8

theorem polynomial_divisibility (p q : ℚ) :
  (∀ x, (x + 2 = 0 ∨ x - 1 = 0 ∨ x - 3 = 0) → polynomial p q x = 0) ↔
  (p = -26/3 ∧ q = -26/3) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l942_94263


namespace NUMINAMATH_CALUDE_undergrads_playing_sports_l942_94291

theorem undergrads_playing_sports (total_students : ℕ) 
  (grad_percent : ℚ) (grad_not_playing : ℚ) (undergrad_not_playing : ℚ) 
  (total_not_playing : ℚ) :
  total_students = 800 →
  grad_percent = 1/4 →
  grad_not_playing = 1/2 →
  undergrad_not_playing = 1/5 →
  total_not_playing = 3/10 →
  (total_students : ℚ) * (1 - grad_percent) * (1 - undergrad_not_playing) = 480 :=
by sorry

end NUMINAMATH_CALUDE_undergrads_playing_sports_l942_94291


namespace NUMINAMATH_CALUDE_fruit_salad_problem_l942_94220

/-- Fruit salad problem -/
theorem fruit_salad_problem (green_grapes red_grapes raspberries blueberries pineapple : ℕ) :
  red_grapes = 3 * green_grapes + 7 →
  raspberries = green_grapes - 5 →
  blueberries = 4 * raspberries →
  pineapple = blueberries / 2 + 5 →
  green_grapes + red_grapes + raspberries + blueberries + pineapple = 350 →
  red_grapes = 100 := by
  sorry

#check fruit_salad_problem

end NUMINAMATH_CALUDE_fruit_salad_problem_l942_94220


namespace NUMINAMATH_CALUDE_election_winner_votes_l942_94299

theorem election_winner_votes (total_votes : ℝ) (winner_votes : ℝ) : 
  (winner_votes = 0.62 * total_votes) →
  (winner_votes - (total_votes - winner_votes) = 384) →
  (winner_votes = 992) :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l942_94299


namespace NUMINAMATH_CALUDE_solve_chalk_problem_l942_94279

def chalk_problem (siblings friends chalk_per_person lost_chalk : ℕ) : Prop :=
  let total_people : ℕ := siblings + friends
  let total_chalk_needed : ℕ := total_people * chalk_per_person
  let available_chalk : ℕ := total_chalk_needed - lost_chalk
  let mom_brought : ℕ := total_chalk_needed - available_chalk
  mom_brought = 2

theorem solve_chalk_problem :
  chalk_problem 4 3 3 2 := by sorry

end NUMINAMATH_CALUDE_solve_chalk_problem_l942_94279


namespace NUMINAMATH_CALUDE_inequality_proof_l942_94283

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) 
  (hb : 0 < b ∧ b ≤ 1) 
  (hc : 0 < c ∧ c ≤ 1) 
  (h_sum : a^2 + b^2 + c^2 = 2) : 
  (1 - b^2) / a + (1 - c^2) / b + (1 - a^2) / c ≤ 5/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l942_94283


namespace NUMINAMATH_CALUDE_total_birds_l942_94274

def geese : ℕ := 58
def ducks : ℕ := 37

theorem total_birds : geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_l942_94274


namespace NUMINAMATH_CALUDE_solve_equation_l942_94295

theorem solve_equation (x : ℝ) : 3 * x + 12 = (1/3) * (6 * x + 36) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l942_94295


namespace NUMINAMATH_CALUDE_three_zeros_a_range_l942_94255

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1/4

noncomputable def g (x : ℝ) : ℝ := -Real.log x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := min (f a x) (g x)

theorem three_zeros_a_range (a : ℝ) :
  (∃ x y z : ℝ, x < y ∧ y < z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    h a x = 0 ∧ h a y = 0 ∧ h a z = 0 ∧
    (∀ w : ℝ, w > 0 → h a w = 0 → w = x ∨ w = y ∨ w = z)) →
  -5/4 < a ∧ a < -3/4 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_a_range_l942_94255


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l942_94251

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l942_94251


namespace NUMINAMATH_CALUDE_odometer_sum_of_squares_l942_94219

/-- Represents a car's odometer reading as a 3-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Converts an OdometerReading to a natural number -/
def OdometerReading.toNat (r : OdometerReading) : Nat :=
  100 * r.hundreds + 10 * r.tens + r.ones

/-- Reverses the digits of an OdometerReading -/
def OdometerReading.reverse (r : OdometerReading) : OdometerReading where
  hundreds := r.ones
  tens := r.tens
  ones := r.hundreds
  valid := by sorry

theorem odometer_sum_of_squares (initial : OdometerReading) 
  (h1 : initial.hundreds + initial.tens + initial.ones ≤ 9)
  (h2 : ∃ (hours : Nat), 
    (OdometerReading.toNat (OdometerReading.reverse initial) - OdometerReading.toNat initial) = 75 * hours) :
  initial.hundreds^2 + initial.tens^2 + initial.ones^2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_of_squares_l942_94219


namespace NUMINAMATH_CALUDE_factors_imply_unique_h_k_l942_94250

-- Define the polynomial
def P (h k : ℝ) (x : ℝ) : ℝ := 3 * x^4 - h * x^2 + k * x - 12

-- State the theorem
theorem factors_imply_unique_h_k :
  ∀ h k : ℝ,
  (∀ x : ℝ, P h k x = 0 ↔ x = 3 ∨ x = -4) →
  ∃! (h' k' : ℝ), P h' k' = P h k :=
by sorry

end NUMINAMATH_CALUDE_factors_imply_unique_h_k_l942_94250


namespace NUMINAMATH_CALUDE_point_distance_inequality_l942_94289

/-- Given points A(0,2), B(0,1), and D(t,0) with t > 0, and M(x,y) on line segment AD,
    if |AM| ≤ 2|BM| always holds, then t ≥ 2√3/3. -/
theorem point_distance_inequality (t : ℝ) (h_t : t > 0) :
  (∀ x y : ℝ, y = (2*t - 2*x)/t →
    x^2 + (y - 2)^2 ≤ 4 * (x^2 + (y - 1)^2)) →
  t ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_point_distance_inequality_l942_94289


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l942_94209

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 7

theorem monotonic_decreasing_interval_of_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≥ f y} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l942_94209


namespace NUMINAMATH_CALUDE_average_popped_percentage_is_82_l942_94261

/-- Represents a bag of popcorn kernels -/
structure PopcornBag where
  popped : ℕ
  total : ℕ

/-- Calculates the percentage of popped kernels in a bag -/
def percentPopped (bag : PopcornBag) : ℚ :=
  (bag.popped : ℚ) / (bag.total : ℚ) * 100

theorem average_popped_percentage_is_82 (bag1 bag2 bag3 : PopcornBag)
    (h1 : bag1 = ⟨60, 75⟩)
    (h2 : bag2 = ⟨42, 50⟩)
    (h3 : bag3 = ⟨82, 100⟩) :
    (percentPopped bag1 + percentPopped bag2 + percentPopped bag3) / 3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_average_popped_percentage_is_82_l942_94261


namespace NUMINAMATH_CALUDE_one_totally_damaged_carton_l942_94270

/-- Represents the milk delivery problem --/
structure MilkDelivery where
  normal_cartons : ℕ
  jars_per_carton : ℕ
  cartons_shortage : ℕ
  damaged_cartons : ℕ
  damaged_jars_per_carton : ℕ
  good_jars : ℕ

/-- Calculates the number of totally damaged cartons --/
def totally_damaged_cartons (md : MilkDelivery) : ℕ :=
  let total_cartons := md.normal_cartons - md.cartons_shortage
  let total_jars := total_cartons * md.jars_per_carton
  let partially_damaged_jars := md.damaged_cartons * md.damaged_jars_per_carton
  let undamaged_jars := total_jars - partially_damaged_jars
  let additional_damaged_jars := undamaged_jars - md.good_jars
  additional_damaged_jars / md.jars_per_carton

/-- Theorem stating that the number of totally damaged cartons is 1 --/
theorem one_totally_damaged_carton (md : MilkDelivery) 
    (h1 : md.normal_cartons = 50)
    (h2 : md.jars_per_carton = 20)
    (h3 : md.cartons_shortage = 20)
    (h4 : md.damaged_cartons = 5)
    (h5 : md.damaged_jars_per_carton = 3)
    (h6 : md.good_jars = 565) :
    totally_damaged_cartons md = 1 := by
  sorry

#eval totally_damaged_cartons ⟨50, 20, 20, 5, 3, 565⟩

end NUMINAMATH_CALUDE_one_totally_damaged_carton_l942_94270


namespace NUMINAMATH_CALUDE_dice_roll_probability_l942_94249

/-- The probability of rolling a number less than four on a six-sided die -/
def prob_first_die : ℚ := 1 / 2

/-- The probability of rolling a number greater than five on an eight-sided die -/
def prob_second_die : ℚ := 3 / 8

/-- The probability of both events occurring -/
def prob_both : ℚ := prob_first_die * prob_second_die

theorem dice_roll_probability : prob_both = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l942_94249


namespace NUMINAMATH_CALUDE_science_marks_calculation_l942_94292

def average_marks : ℝ := 75
def num_subjects : ℕ := 5
def math_marks : ℝ := 76
def social_marks : ℝ := 82
def english_marks : ℝ := 67
def biology_marks : ℝ := 85

theorem science_marks_calculation :
  ∃ (science_marks : ℝ),
    (math_marks + social_marks + english_marks + biology_marks + science_marks) / num_subjects = average_marks ∧
    science_marks = 65 := by
  sorry

end NUMINAMATH_CALUDE_science_marks_calculation_l942_94292


namespace NUMINAMATH_CALUDE_income_increase_theorem_l942_94275

/-- Represents the student's income sources and total income -/
structure StudentIncome where
  scholarship : ℝ
  partTimeJob : ℝ
  parentalSupport : ℝ
  totalIncome : ℝ

/-- Theorem stating the relationship between income sources and total income increase -/
theorem income_increase_theorem (income : StudentIncome) 
  (h1 : income.scholarship + income.partTimeJob + income.parentalSupport = income.totalIncome)
  (h2 : 2 * income.scholarship + income.partTimeJob + income.parentalSupport = 1.05 * income.totalIncome)
  (h3 : income.scholarship + 2 * income.partTimeJob + income.parentalSupport = 1.15 * income.totalIncome) :
  income.scholarship + income.partTimeJob + 2 * income.parentalSupport = 1.8 * income.totalIncome := by
  sorry


end NUMINAMATH_CALUDE_income_increase_theorem_l942_94275


namespace NUMINAMATH_CALUDE_total_dolls_l942_94200

/-- Given that Hannah has 5 times as many dolls as her sister, and her sister has 8 dolls,
    prove that the total number of dolls they have together is 48. -/
theorem total_dolls (hannah_multiplier sister_dolls : ℕ) 
  (h1 : hannah_multiplier = 5)
  (h2 : sister_dolls = 8) :
  hannah_multiplier * sister_dolls + sister_dolls = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l942_94200


namespace NUMINAMATH_CALUDE_good_number_characterization_l942_94222

def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k.val + 1 + (a k).val + 1 : ℕ) = m ^ 2

def notGoodNumbers : Set ℕ := {1, 2, 4, 6, 7, 9, 11}

theorem good_number_characterization (n : ℕ) :
  n ≠ 0 → (isGoodNumber n ↔ n ∉ notGoodNumbers) :=
by sorry

end NUMINAMATH_CALUDE_good_number_characterization_l942_94222


namespace NUMINAMATH_CALUDE_product_of_digits_3545_l942_94203

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def units_digit (n : ℕ) : ℕ :=
  n % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem product_of_digits_3545 :
  let n := 3545
  ¬ is_divisible_by_3 n ∧ units_digit n * tens_digit n = 20 :=
by sorry

end NUMINAMATH_CALUDE_product_of_digits_3545_l942_94203


namespace NUMINAMATH_CALUDE_probability_one_person_two_days_l942_94238

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of days
def num_days : ℕ := 2

-- Define the number of students required each day
def students_per_day : ℕ := 2

-- Define the function to calculate combinations
def C (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the total number of ways to select students for two days
def total_ways : ℕ := (C num_students students_per_day) * (C num_students students_per_day)

-- Define the number of ways exactly 1 person participates for two consecutive days
def favorable_ways : ℕ := (C num_students 1) * (C (num_students - 1) 1) * (C (num_students - 2) 1)

-- State the theorem
theorem probability_one_person_two_days :
  (favorable_ways : ℚ) / total_ways = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_one_person_two_days_l942_94238


namespace NUMINAMATH_CALUDE_reflect_x_three_two_l942_94253

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system. -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The coordinates of (3,2) with respect to the x-axis are (3,-2). -/
theorem reflect_x_three_two :
  reflect_x (3, 2) = (3, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_three_two_l942_94253


namespace NUMINAMATH_CALUDE_mara_marbles_l942_94294

theorem mara_marbles (mara_bags : ℕ) (markus_bags : ℕ) (markus_marbles_per_bag : ℕ) :
  mara_bags = 12 →
  markus_bags = 2 →
  markus_marbles_per_bag = 13 →
  ∃ (mara_marbles_per_bag : ℕ),
    mara_bags * mara_marbles_per_bag + 2 = markus_bags * markus_marbles_per_bag ∧
    mara_marbles_per_bag = 2 :=
by sorry

end NUMINAMATH_CALUDE_mara_marbles_l942_94294


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l942_94273

theorem missing_fraction_sum (x : ℚ) : 
  (1/3 : ℚ) + (1/2 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-2/15 : ℚ) + (-17/30 : ℚ) = 
  (13333333333333333 : ℚ) / 100000000000000000 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l942_94273


namespace NUMINAMATH_CALUDE_hash_four_one_l942_94234

-- Define the # operation
def hash (a b : ℤ) : ℤ := (a + b + 2) * (a - b - 2)

-- Theorem statement
theorem hash_four_one : hash 4 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_one_l942_94234


namespace NUMINAMATH_CALUDE_A_D_mutually_exclusive_not_complementary_l942_94215

-- Define the sample space for a die throw
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the events
def A : Set Nat := {n ∈ Ω | n % 2 = 1}
def B : Set Nat := {n ∈ Ω | n % 2 = 0}
def C : Set Nat := {n ∈ Ω | n % 2 = 0}
def D : Set Nat := {2, 4}

-- Define mutually exclusive
def mutually_exclusive (X Y : Set Nat) : Prop := X ∩ Y = ∅

-- Define complementary
def complementary (X Y : Set Nat) : Prop := X ∪ Y = Ω

-- Theorem to prove
theorem A_D_mutually_exclusive_not_complementary :
  mutually_exclusive A D ∧ ¬complementary A D :=
by sorry

end NUMINAMATH_CALUDE_A_D_mutually_exclusive_not_complementary_l942_94215


namespace NUMINAMATH_CALUDE_largest_beta_exponent_l942_94282

open Real

/-- Given a sequence of points in a plane with specific distance properties, 
    this theorem proves the largest possible exponent β for which r_n ≥ Cn^β holds. -/
theorem largest_beta_exponent 
  (O : ℝ × ℝ) 
  (P : ℕ → ℝ × ℝ) 
  (r : ℕ → ℝ) 
  (α : ℝ) 
  (h_alpha : 0 < α ∧ α < 1)
  (h_distance : ∀ n m : ℕ, n ≠ m → dist (P n) (P m) ≥ (r n) ^ α)
  (h_r_increasing : ∀ n : ℕ, r n ≤ r (n + 1))
  (h_r_def : ∀ n : ℕ, dist O (P n) = r n) :
  ∃ (C : ℝ) (h_C : C > 0), ∀ n : ℕ, r n ≥ C * n ^ (1 / (2 * (1 - α))) ∧ 
  ∀ β : ℝ, (∃ (D : ℝ) (h_D : D > 0), ∀ n : ℕ, r n ≥ D * n ^ β) → β ≤ 1 / (2 * (1 - α)) :=
sorry

end NUMINAMATH_CALUDE_largest_beta_exponent_l942_94282


namespace NUMINAMATH_CALUDE_distance_AB_equals_5_l942_94246

-- Define the line l₁
def line_l₁ (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the curve C
def curve_C (x y φ : ℝ) : Prop :=
  x = 1 + Real.sqrt 3 * Real.cos φ ∧
  y = Real.sqrt 3 * Real.sin φ ∧
  0 ≤ φ ∧ φ ≤ Real.pi

-- Define the line l₂ in polar coordinates
def line_l₂ (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi / 3) + 3 * Real.sqrt 3 = 0

-- Define the intersection point A of l₁ and C
def point_A : ℝ × ℝ := sorry

-- Define the intersection point B of l₁ and l₂
def point_B : ℝ × ℝ := sorry

-- Theorem statement
theorem distance_AB_equals_5 :
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_AB_equals_5_l942_94246


namespace NUMINAMATH_CALUDE_product_and_multiple_l942_94243

theorem product_and_multiple : ∃ (ε : ℝ) (x : ℝ), 
  (ε > 0 ∧ ε < 1) ∧ 
  (abs (198 * 2 - 400) < ε) ∧ 
  (2 * x = 56) ∧ 
  (9 * x = 252) := by
  sorry

end NUMINAMATH_CALUDE_product_and_multiple_l942_94243


namespace NUMINAMATH_CALUDE_luke_game_points_l942_94297

theorem luke_game_points (points_per_round : ℕ) (num_rounds : ℕ) (total_points : ℕ) : 
  points_per_round = 327 → num_rounds = 193 → total_points = points_per_round * num_rounds → total_points = 63111 := by
  sorry

end NUMINAMATH_CALUDE_luke_game_points_l942_94297


namespace NUMINAMATH_CALUDE_at_least_one_negative_l942_94262

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 + 1/b = b^2 + 1/a) :
  a < 0 ∨ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l942_94262


namespace NUMINAMATH_CALUDE_equation_has_solution_l942_94252

/-- The equation has at least one solution for all real values of the parameter 'a' -/
theorem equation_has_solution (a : ℝ) : ∃ x : ℝ, 
  (2 - 2 * a * (x + 1)) / (|x| - x) = Real.sqrt (1 - a - a * x) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_solution_l942_94252


namespace NUMINAMATH_CALUDE_basketball_court_perimeter_l942_94212

/-- The perimeter of a rectangular basketball court is 96 meters -/
theorem basketball_court_perimeter :
  ∀ (length width : ℝ),
  length = width + 14 →
  (length = 31 ∧ width = 17) →
  2 * (length + width) = 96 := by
  sorry

end NUMINAMATH_CALUDE_basketball_court_perimeter_l942_94212


namespace NUMINAMATH_CALUDE_eighteen_to_binary_l942_94228

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

def binary_to_decimal (l : List ℕ) : ℕ :=
  l.foldl (fun acc d => 2 * acc + d) 0

theorem eighteen_to_binary :
  decimal_to_binary 18 = [1, 0, 0, 1, 0] ∧
  binary_to_decimal [1, 0, 0, 1, 0] = 18 :=
sorry

end NUMINAMATH_CALUDE_eighteen_to_binary_l942_94228


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l942_94230

theorem fraction_to_zero_power :
  let f : ℚ := -574839201 / 1357924680
  f ≠ 0 →
  f^0 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l942_94230


namespace NUMINAMATH_CALUDE_horse_feeding_amount_l942_94288

/-- Calculates the amount of food each horse receives at each feeding --/
def food_per_horse_per_feeding (num_horses : ℕ) (feedings_per_day : ℕ) (days : ℕ) (bags_bought : ℕ) (pounds_per_bag : ℕ) : ℕ :=
  (bags_bought * pounds_per_bag) / (num_horses * feedings_per_day * days)

/-- Theorem stating the amount of food each horse receives at each feeding --/
theorem horse_feeding_amount :
  food_per_horse_per_feeding 25 2 60 60 1000 = 20 := by
  sorry

#eval food_per_horse_per_feeding 25 2 60 60 1000

end NUMINAMATH_CALUDE_horse_feeding_amount_l942_94288


namespace NUMINAMATH_CALUDE_tv_sales_effect_l942_94240

theorem tv_sales_effect (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 0.82 * P
  let new_quantity := 1.72 * Q
  let original_value := P * Q
  let new_value := new_price * new_quantity
  (new_value / original_value - 1) * 100 = 41.04 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_effect_l942_94240


namespace NUMINAMATH_CALUDE_clock_angle_at_8_clock_angle_at_8_is_120_l942_94281

/-- The angle between clock hands at 8:00 -/
theorem clock_angle_at_8 : ℝ :=
  let total_degrees : ℝ := 360
  let hours_on_clock : ℕ := 12
  let current_hour : ℕ := 8
  let degrees_per_hour : ℝ := total_degrees / hours_on_clock
  let hour_hand_angle : ℝ := degrees_per_hour * current_hour
  let minute_hand_angle : ℝ := 0
  let angle_diff : ℝ := |hour_hand_angle - minute_hand_angle|
  min angle_diff (total_degrees - angle_diff)

/-- Theorem: The smaller angle between the hour-hand and minute-hand of a clock at 8:00 is 120° -/
theorem clock_angle_at_8_is_120 : clock_angle_at_8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_8_clock_angle_at_8_is_120_l942_94281


namespace NUMINAMATH_CALUDE_abc_sum_l942_94264

theorem abc_sum (A B C : ℕ+) (h1 : Nat.gcd A.val (Nat.gcd B.val C.val) = 1)
  (h2 : (A : ℝ) * Real.log 5 / Real.log 500 + (B : ℝ) * Real.log 2 / Real.log 500 = C) :
  A + B + C = 6 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l942_94264


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l942_94258

/-- A chess tournament with the given conditions -/
structure ChessTournament where
  num_players : ℕ
  games_per_player : ℕ
  losses_per_player : ℕ
  no_ties : Bool

/-- Calculates the total number of games in the tournament -/
def total_games (t : ChessTournament) : ℕ :=
  t.num_players * (t.num_players - 1) / 2

/-- Calculates the number of wins for each player -/
def wins_per_player (t : ChessTournament) : ℕ :=
  t.games_per_player - t.losses_per_player

/-- The theorem to be proved -/
theorem chess_tournament_theorem (t : ChessTournament) 
  (h1 : t.num_players = 200)
  (h2 : t.games_per_player = 199)
  (h3 : t.losses_per_player = 30)
  (h4 : t.no_ties = true) :
  total_games t = 19900 ∧ wins_per_player t = 169 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_theorem_l942_94258


namespace NUMINAMATH_CALUDE_train_length_l942_94216

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 12 → speed * time * (1000 / 3600) = 240 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l942_94216


namespace NUMINAMATH_CALUDE_pi_half_irrational_l942_94241

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l942_94241


namespace NUMINAMATH_CALUDE_symmetric_point_complex_l942_94286

def symmetric_about_imaginary_axis (z : ℂ) : ℂ := -Complex.re z + Complex.im z * Complex.I

theorem symmetric_point_complex : 
  let A : ℂ := 2 + Complex.I
  let B : ℂ := symmetric_about_imaginary_axis A
  B = -2 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_complex_l942_94286


namespace NUMINAMATH_CALUDE_reflection_of_point_l942_94211

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis reflection of a point -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The x-axis reflection of point (-2, 3) is (-2, -3) -/
theorem reflection_of_point :
  let P : Point := { x := -2, y := 3 }
  reflect_x P = { x := -2, y := -3 } := by
sorry

end NUMINAMATH_CALUDE_reflection_of_point_l942_94211


namespace NUMINAMATH_CALUDE_round_trip_time_l942_94214

/-- Proves that given a round trip with specified conditions, the outbound journey takes 180 minutes -/
theorem round_trip_time (speed_out speed_return : ℝ) (total_time : ℝ) : 
  speed_out = 100 →
  speed_return = 150 →
  total_time = 5 →
  (total_time * speed_out * speed_return) / (speed_out + speed_return) / speed_out * 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_time_l942_94214


namespace NUMINAMATH_CALUDE_brand_w_households_l942_94287

theorem brand_w_households (total : ℕ) (neither : ℕ) (both : ℕ) : 
  total = 200 →
  neither = 80 →
  both = 40 →
  ∃ (w b : ℕ), w + b + both + neither = total ∧ b = 3 * both ∧ w = 40 :=
by sorry

end NUMINAMATH_CALUDE_brand_w_households_l942_94287


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l942_94290

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 2 + a 9 = 11) →
  (a 4 + a 10 = 14) →
  (a 6 + a 11 = 17) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l942_94290


namespace NUMINAMATH_CALUDE_peaches_per_box_is_15_l942_94260

/-- Given the initial number of peaches per basket, the number of baskets,
    the number of peaches eaten by farmers, and the number of smaller boxes,
    calculate the number of peaches in each smaller box. -/
def peaches_per_box (initial_peaches_per_basket : ℕ) (num_baskets : ℕ) 
                    (peaches_eaten : ℕ) (num_smaller_boxes : ℕ) : ℕ :=
  ((initial_peaches_per_basket * num_baskets) - peaches_eaten) / num_smaller_boxes

/-- Theorem stating that given the specific conditions in the problem,
    the number of peaches in each smaller box is 15. -/
theorem peaches_per_box_is_15 :
  peaches_per_box 25 5 5 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_peaches_per_box_is_15_l942_94260


namespace NUMINAMATH_CALUDE_evelyn_winning_strategy_l942_94276

/-- Represents a player in the game -/
inductive Player
| Odin
| Evelyn

/-- Represents the state of a box in the game -/
structure Box where
  value : ℕ
  isEmpty : Bool

/-- Represents the game state -/
structure GameState where
  boxes : List Box
  currentPlayer : Player

/-- Defines a valid move in the game -/
def isValidMove (player : Player) (oldValue newValue : ℕ) : Prop :=
  match player with
  | Player.Odin => newValue < oldValue ∧ Odd newValue
  | Player.Evelyn => newValue < oldValue ∧ Even newValue

/-- Defines the winning condition for Evelyn -/
def isEvelynWin (state : GameState) : Prop :=
  let k := state.boxes.length / 3
  (state.boxes.filter (fun b => b.value = 0)).length = k ∧
  (state.boxes.filter (fun b => b.value ≠ 0)).all (fun b => b.value = 1)

/-- Defines the winning condition for Odin -/
def isOdinWin (state : GameState) : Prop :=
  let k := state.boxes.length / 3
  (state.boxes.filter (fun b => b.value = 0)).length = k ∧
  ¬(state.boxes.filter (fun b => b.value ≠ 0)).all (fun b => b.value = 1)

/-- Theorem stating that Evelyn has a winning strategy for all k -/
theorem evelyn_winning_strategy (k : ℕ) (h : k > 0) :
  ∃ (strategy : GameState → ℕ → ℕ),
    ∀ (initialState : GameState),
      initialState.boxes.length = 3 * k →
      initialState.currentPlayer = Player.Odin →
      (initialState.boxes.all (fun b => b.isEmpty)) →
      (∃ (finalState : GameState),
        (finalState.boxes.all (fun b => ¬b.isEmpty)) ∧
        (isEvelynWin finalState ∨
         (¬∃ (move : ℕ → ℕ), isValidMove Player.Odin (move 0) (move 1)))) :=
sorry

end NUMINAMATH_CALUDE_evelyn_winning_strategy_l942_94276


namespace NUMINAMATH_CALUDE_lcm_of_48_and_64_l942_94254

theorem lcm_of_48_and_64 :
  let a := 48
  let b := 64
  let hcf := 16
  lcm a b = 192 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_48_and_64_l942_94254


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l942_94237

theorem five_digit_multiple_of_nine :
  ∃ (n : ℕ), n = 56781 ∧ n % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l942_94237


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l942_94201

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l942_94201


namespace NUMINAMATH_CALUDE_problem_proof_l942_94298

theorem problem_proof : (5 * 12) / (180 / 3) + 61 = 62 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l942_94298


namespace NUMINAMATH_CALUDE_r_profit_share_l942_94280

/-- Represents the profit share of a partner in a business partnership --/
def ProfitShare (initial_ratio : ℚ) (months_full : ℕ) (months_reduced : ℕ) (reduction_factor : ℚ) (total_profit : ℚ) : ℚ :=
  let total_investment := 12 * (4 + 6 + 10)
  let partner_investment := initial_ratio * months_full + initial_ratio * reduction_factor * months_reduced
  (partner_investment / total_investment) * total_profit

theorem r_profit_share :
  let p_ratio : ℚ := 4
  let q_ratio : ℚ := 6
  let r_ratio : ℚ := 10
  let total_profit : ℚ := 4650
  ProfitShare r_ratio 12 0 1 total_profit = 2325 := by
  sorry

end NUMINAMATH_CALUDE_r_profit_share_l942_94280


namespace NUMINAMATH_CALUDE_system_solution_l942_94248

/-- The function φ(t) = 2t^3 + t - 2 -/
def φ (t : ℝ) : ℝ := 2 * t^3 + t - 2

/-- The system of equations -/
def satisfies_system (x y z : ℝ) : Prop :=
  x^5 = φ y ∧ y^5 = φ z ∧ z^5 = φ x

theorem system_solution (x y z : ℝ) (h : satisfies_system x y z) :
  x = y ∧ y = z ∧ φ x = x^5 := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l942_94248


namespace NUMINAMATH_CALUDE_intersection_empty_iff_m_nonnegative_l942_94293

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*x + m = 0}
def B : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem intersection_empty_iff_m_nonnegative (m : ℝ) :
  A m ∩ B = ∅ ↔ m ∈ Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_m_nonnegative_l942_94293


namespace NUMINAMATH_CALUDE_triangle_rotation_l942_94239

theorem triangle_rotation (α β γ : ℝ) (k m : ℤ) (h1 : 15 * α = 360 * k)
    (h2 : 6 * β = 360 * m) (h3 : α + β + γ = 180) :
  ∃ (n : ℕ+), n * γ = 360 * (n / 5 : ℤ) ∧ ∀ (n' : ℕ+), n' < n → ¬(∃ (l : ℤ), n' * γ = 360 * l) := by
  sorry

end NUMINAMATH_CALUDE_triangle_rotation_l942_94239


namespace NUMINAMATH_CALUDE_jeans_sale_savings_l942_94259

/-- Calculates the total savings when purchasing jeans with given prices and discounts -/
def total_savings (fox_price pony_price : ℚ) (fox_discount pony_discount : ℚ) 
  (fox_quantity pony_quantity : ℕ) : ℚ :=
  let regular_total := fox_price * fox_quantity + pony_price * pony_quantity
  let discounted_total := (fox_price * (1 - fox_discount)) * fox_quantity + 
                          (pony_price * (1 - pony_discount)) * pony_quantity
  regular_total - discounted_total

/-- Theorem stating that the total savings is $18 under the given conditions -/
theorem jeans_sale_savings :
  let fox_price : ℚ := 15
  let pony_price : ℚ := 18
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let pony_discount : ℚ := 1/2
  let fox_discount : ℚ := 1/2 - pony_discount
  total_savings fox_price pony_price fox_discount pony_discount fox_quantity pony_quantity = 18 :=
by sorry

end NUMINAMATH_CALUDE_jeans_sale_savings_l942_94259


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l942_94229

theorem greatest_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * |y| + 6 < 24) → y ≤ 5 ∧ ∃ (z : ℤ), z > 5 ∧ ¬(3 * |z| + 6 < 24) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l942_94229


namespace NUMINAMATH_CALUDE_actual_average_height_l942_94204

/-- The number of boys in the class -/
def num_boys : ℕ := 50

/-- The initially calculated average height in cm -/
def initial_avg : ℝ := 183

/-- The incorrectly recorded height of the first boy in cm -/
def incorrect_height1 : ℝ := 166

/-- The actual height of the first boy in cm -/
def actual_height1 : ℝ := 106

/-- The incorrectly recorded height of the second boy in cm -/
def incorrect_height2 : ℝ := 175

/-- The actual height of the second boy in cm -/
def actual_height2 : ℝ := 190

/-- Conversion factor from cm to feet -/
def cm_to_feet : ℝ := 30.48

/-- Theorem stating that the actual average height of the boys is approximately 5.98 feet -/
theorem actual_average_height :
  let total_height := num_boys * initial_avg
  let corrected_total := total_height - (incorrect_height1 - actual_height1) + (actual_height2 - incorrect_height2)
  let actual_avg_cm := corrected_total / num_boys
  let actual_avg_feet := actual_avg_cm / cm_to_feet
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |actual_avg_feet - 5.98| < ε :=
by sorry

end NUMINAMATH_CALUDE_actual_average_height_l942_94204


namespace NUMINAMATH_CALUDE_cookies_taken_theorem_l942_94213

/-- Calculates the number of cookies taken out in 6 days given the initial count,
    remaining count after 10 days, and assuming equal daily consumption. -/
def cookies_taken_in_six_days (initial_count : ℕ) (remaining_count : ℕ) : ℕ :=
  let total_taken := initial_count - remaining_count
  let daily_taken := total_taken / 10
  6 * daily_taken

/-- Theorem stating that given 150 initial cookies and 45 remaining after 10 days,
    the number of cookies taken in 6 days is 63. -/
theorem cookies_taken_theorem :
  cookies_taken_in_six_days 150 45 = 63 := by
  sorry

#eval cookies_taken_in_six_days 150 45

end NUMINAMATH_CALUDE_cookies_taken_theorem_l942_94213


namespace NUMINAMATH_CALUDE_trip_distance_l942_94218

theorem trip_distance (total_time hiking_speed canoe_speed hiking_distance : ℝ) 
  (h1 : total_time = 5.5)
  (h2 : hiking_speed = 5)
  (h3 : canoe_speed = 12)
  (h4 : hiking_distance = 27) :
  hiking_distance + (total_time - hiking_distance / hiking_speed) * canoe_speed = 28.2 := by
  sorry

#check trip_distance

end NUMINAMATH_CALUDE_trip_distance_l942_94218


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l942_94207

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(503 * m.val ≡ 1019 * m.val [ZMOD 48])) ∧
  (503 * n.val ≡ 1019 * n.val [ZMOD 48]) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l942_94207


namespace NUMINAMATH_CALUDE_max_min_sum_zero_l942_94267

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_min_sum_zero :
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∀ x, f x ≥ n) ∧ (m + n = 0) := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_zero_l942_94267


namespace NUMINAMATH_CALUDE_solve_triangle_equation_l942_94217

-- Define the ∆ operation
def triangle (A B : ℕ) : ℕ := 2 * A + B

-- Theorem statement
theorem solve_triangle_equation : 
  ∃ x : ℕ, triangle (triangle 3 2) x = 20 ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_triangle_equation_l942_94217


namespace NUMINAMATH_CALUDE_gmat_test_results_l942_94247

theorem gmat_test_results (first_correct : ℝ) (second_correct : ℝ) (neither_correct : ℝ)
  (h1 : first_correct = 85)
  (h2 : second_correct = 80)
  (h3 : neither_correct = 5)
  : first_correct + second_correct - (100 - neither_correct) = 70 := by
  sorry

end NUMINAMATH_CALUDE_gmat_test_results_l942_94247


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_5_l942_94256

-- Define a function to calculate the sum of digits in a number
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property of being the first year after 2020 with digit sum 5
def isFirstYearAfter2020WithDigitSum5 (year : ℕ) : Prop :=
  year > 2020 ∧
  sumOfDigits year = 5 ∧
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 5

-- Theorem statement
theorem first_year_after_2020_with_digit_sum_5 :
  sumOfDigits 2020 = 4 →
  isFirstYearAfter2020WithDigitSum5 2021 :=
by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_5_l942_94256


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l942_94227

/-- The sum of roots of a quadratic equation x^2 + (m-1)x + (m+n) = 0 is 1 - m -/
theorem sum_of_roots_quadratic (m n : ℝ) (hm : m ≠ 1) (hn : n ≠ -m) :
  let f : ℝ → ℝ := λ x => x^2 + (m-1)*x + (m+n)
  (∃ r s : ℝ, f r = 0 ∧ f s = 0) → r + s = 1 - m :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l942_94227


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l942_94236

theorem square_minus_product_plus_square : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l942_94236


namespace NUMINAMATH_CALUDE_problem_statements_l942_94208

theorem problem_statements :
  (∀ x : ℝ, (Real.sqrt (x + 1) * (2 * x - 1) ≥ 0) ↔ (x ≥ 1/2)) ∧
  (∀ x y : ℝ, (x > 1 ∧ y > 2) → (x + y > 3)) ∧
  (∃ x y : ℝ, (x + y > 3) ∧ ¬(x > 1 ∧ y > 2)) ∧
  (∀ x : ℝ, Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) > 2) ∧
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l942_94208


namespace NUMINAMATH_CALUDE_prime_divides_n6_minus_1_implies_n_greater_than_sqrt_p_minus_1_l942_94223

theorem prime_divides_n6_minus_1_implies_n_greater_than_sqrt_p_minus_1 
  (p : ℕ) (n : ℕ) (h_prime : Nat.Prime p) (h_n_ge_2 : n ≥ 2) 
  (h_div : p ∣ (n^6 - 1)) : n > Real.sqrt p - 1 :=
sorry

end NUMINAMATH_CALUDE_prime_divides_n6_minus_1_implies_n_greater_than_sqrt_p_minus_1_l942_94223


namespace NUMINAMATH_CALUDE_fraction_inequality_l942_94226

theorem fraction_inequality (a b : ℝ) (h : a > b) : a / 4 > b / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l942_94226


namespace NUMINAMATH_CALUDE_first_term_value_l942_94272

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_value 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a5 : a 5 = 9) 
  (h_a3_a2 : 2 * a 3 = a 2 + 6) : 
  a 1 = -3 := by
sorry

end NUMINAMATH_CALUDE_first_term_value_l942_94272


namespace NUMINAMATH_CALUDE_strawberry_picking_l942_94266

/-- The number of baskets Lilibeth fills -/
def baskets : ℕ := 6

/-- The number of strawberries each basket holds -/
def strawberries_per_basket : ℕ := 50

/-- The number of Lilibeth's friends who pick the same amount as her -/
def friends : ℕ := 3

/-- The total number of strawberries picked by Lilibeth and her friends -/
def total_strawberries : ℕ := (friends + 1) * (baskets * strawberries_per_basket)

theorem strawberry_picking :
  total_strawberries = 1200 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picking_l942_94266


namespace NUMINAMATH_CALUDE_simplify_fraction_l942_94285

theorem simplify_fraction : (180 : ℚ) / 1260 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l942_94285
