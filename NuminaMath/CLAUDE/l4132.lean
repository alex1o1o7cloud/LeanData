import Mathlib

namespace m_range_l4132_413218

-- Define propositions p and q
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- State the theorem
theorem m_range (m : ℝ) :
  m > 0 ∧
  necessary_not_sufficient (p) (q m) →
  0 < m ∧ m ≤ 3 :=
by sorry

end m_range_l4132_413218


namespace f_derivative_at_one_l4132_413238

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_derivative_at_one : 
  deriv f 1 = 1 := by sorry

end f_derivative_at_one_l4132_413238


namespace intersection_of_A_and_B_l4132_413279

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | ∃ y, y = Real.log x}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l4132_413279


namespace wrong_observation_value_l4132_413224

theorem wrong_observation_value (n : ℕ) (initial_mean correct_value new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : correct_value = 60)
  (h4 : new_mean = 36.5) :
  ∃ wrong_value : ℝ,
    n * initial_mean - wrong_value + correct_value = n * new_mean ∧
    wrong_value = 35 := by
  sorry

end wrong_observation_value_l4132_413224


namespace max_integers_greater_than_26_l4132_413286

theorem max_integers_greater_than_26 (a b c d e : ℤ) :
  a + b + c + d + e = 3 →
  ∃ (count : ℕ), count ≤ 4 ∧
    count = (if a > 26 then 1 else 0) +
            (if b > 26 then 1 else 0) +
            (if c > 26 then 1 else 0) +
            (if d > 26 then 1 else 0) +
            (if e > 26 then 1 else 0) ∧
    ∀ (other_count : ℕ),
      other_count > count →
      ¬(∃ (a' b' c' d' e' : ℤ),
        a' + b' + c' + d' + e' = 3 ∧
        other_count = (if a' > 26 then 1 else 0) +
                      (if b' > 26 then 1 else 0) +
                      (if c' > 26 then 1 else 0) +
                      (if d' > 26 then 1 else 0) +
                      (if e' > 26 then 1 else 0)) :=
by sorry

end max_integers_greater_than_26_l4132_413286


namespace insufficient_info_to_determine_sum_l4132_413287

/-- Represents a class with boys, girls, and a teacher -/
structure Classroom where
  numBoys : ℕ
  numGirls : ℕ
  avgAgeBoys : ℝ
  avgAgeGirls : ℝ
  avgAgeAll : ℝ
  teacherAge : ℕ

/-- The conditions given in the problem -/
def classroomConditions (c : Classroom) : Prop :=
  c.numBoys > 0 ∧
  c.numGirls > 0 ∧
  c.avgAgeBoys = c.avgAgeGirls ∧
  c.avgAgeGirls = c.avgAgeBoys ∧
  c.avgAgeAll = c.avgAgeBoys + c.avgAgeGirls ∧
  c.teacherAge = 42

/-- Theorem stating that the given conditions are insufficient to determine b + g -/
theorem insufficient_info_to_determine_sum (c : Classroom) 
  (h : classroomConditions c) : 
  ∃ (c1 c2 : Classroom), classroomConditions c1 ∧ classroomConditions c2 ∧ 
  c1.avgAgeBoys + c1.avgAgeGirls ≠ c2.avgAgeBoys + c2.avgAgeGirls :=
sorry

end insufficient_info_to_determine_sum_l4132_413287


namespace smallest_multiple_l4132_413243

theorem smallest_multiple (n : ℕ) : n = 714 ↔ 
  n > 0 ∧ 
  n % 17 = 0 ∧ 
  (n - 7) % 101 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 17 = 0 → (m - 7) % 101 = 0 → m ≥ n :=
by sorry

end smallest_multiple_l4132_413243


namespace exponential_function_fixed_point_l4132_413213

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 3) - 4
  f (-3) = -3 := by sorry

end exponential_function_fixed_point_l4132_413213


namespace cubic_derivative_problem_l4132_413258

/-- Given a cubic function f(x) = ax³ + bx² + 3 where b is a constant,
    if f'(1) = -5, then f'(2) = -4 -/
theorem cubic_derivative_problem (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + 3
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 2 * b * x
  f' 1 = -5 → f' 2 = -4 := by
  sorry

end cubic_derivative_problem_l4132_413258


namespace platform_length_l4132_413214

/-- Calculates the length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_tree : ℝ)
  (time_platform : ℝ)
  (h1 : train_length = 600)
  (h2 : time_tree = 60)
  (h3 : time_platform = 105) :
  let train_speed := train_length / time_tree
  let platform_length := train_speed * time_platform - train_length
  platform_length = 450 :=
by
  sorry

end platform_length_l4132_413214


namespace book_arrangement_proof_l4132_413293

/-- The number of distinct arrangements of books on a shelf. -/
def distinct_arrangements (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem: The number of distinct arrangements of 7 books with 3 identical copies is 840. -/
theorem book_arrangement_proof :
  distinct_arrangements 7 3 = 840 := by
  sorry

end book_arrangement_proof_l4132_413293


namespace greatest_k_for_inequality_l4132_413250

theorem greatest_k_for_inequality : ∃! k : ℕ, 
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 → a * b * c = 1 → 
    (1 / a + 1 / b + 1 / c + k / (a + b + c + 1) ≥ 3 + k / 4)) ∧
  (∀ k' : ℕ, k' > k → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧
      1 / a + 1 / b + 1 / c + k' / (a + b + c + 1) < 3 + k' / 4) ∧
  k = 13 :=
by sorry

end greatest_k_for_inequality_l4132_413250


namespace ellipse_hyperbola_min_eccentricity_l4132_413251

/-- Given an ellipse and a hyperbola with the same foci, prove the minimum value of 3e₁² + e₂² -/
theorem ellipse_hyperbola_min_eccentricity (c : ℝ) (e₁ e₂ : ℝ) : 
  c > 0 → -- Foci are distinct points
  e₁ > 0 → -- Eccentricity of ellipse is positive
  e₂ > 0 → -- Eccentricity of hyperbola is positive
  e₁ * e₂ = 1 → -- Relationship between eccentricities due to shared foci and asymptote condition
  3 * e₁^2 + e₂^2 ≥ 2 * Real.sqrt 3 :=
by sorry

end ellipse_hyperbola_min_eccentricity_l4132_413251


namespace power_of_two_representation_l4132_413226

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), 2^n = 7*x^2 + y^2 ∧ Odd x ∧ Odd y :=
by sorry

end power_of_two_representation_l4132_413226


namespace dots_on_abc_l4132_413248

/-- Represents a die face with a number of dots -/
structure DieFace :=
  (dots : Nat)
  (h : dots ≥ 1 ∧ dots ≤ 6)

/-- Represents a die with six faces -/
structure Die :=
  (faces : Fin 6 → DieFace)
  (opposite_sum : ∀ i : Fin 3, (faces i).dots + (faces (i + 3)).dots = 7)
  (all_different : ∀ i j : Fin 6, i ≠ j → (faces i).dots ≠ (faces j).dots)

/-- Represents the configuration of four glued dice -/
structure GluedDice :=
  (dice : Fin 4 → Die)
  (glued_faces_same : ∀ i j : Fin 4, i ≠ j → ∃ fi fj : Fin 6, 
    (dice i).faces fi = (dice j).faces fj)

/-- The main theorem stating the number of dots on faces A, B, and C -/
theorem dots_on_abc (gd : GluedDice) : 
  ∃ (a b c : DieFace), 
    a.dots = 2 ∧ b.dots = 2 ∧ c.dots = 6 ∧
    (∃ (i j k : Fin 4) (fi fj fk : Fin 6), 
      i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
      a = (gd.dice i).faces fi ∧
      b = (gd.dice j).faces fj ∧
      c = (gd.dice k).faces fk) :=
sorry

end dots_on_abc_l4132_413248


namespace five_students_left_l4132_413247

/-- Calculates the number of students who left during the year. -/
def students_who_left (initial_students new_students final_students : ℕ) : ℕ :=
  initial_students + new_students - final_students

/-- Proves that 5 students left during the year given the problem conditions. -/
theorem five_students_left : students_who_left 31 11 37 = 5 := by
  sorry

end five_students_left_l4132_413247


namespace impossible_coin_probabilities_l4132_413269

theorem impossible_coin_probabilities :
  ¬∃ (p₁ p₂ : ℝ), 0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧
    (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
    p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end impossible_coin_probabilities_l4132_413269


namespace tangent_line_equation_l4132_413282

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 3*x + 1

-- Define the point on the tangent line
def point : ℝ × ℝ := (2, 5)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 7*x - y - 9 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (k : ℝ), 
    (∀ x, (deriv f) x = 2*x + 3) ∧ 
    (deriv f) point.1 = k ∧
    ∀ x y, tangent_line x y ↔ y - point.2 = k * (x - point.1) :=
sorry

end tangent_line_equation_l4132_413282


namespace hand_74_falls_off_after_20_minutes_l4132_413215

/-- Represents a clock hand with its rotation speed and fall-off time. -/
structure ClockHand where
  speed : ℕ
  fallOffTime : ℚ

/-- Represents a clock with multiple hands. -/
def Clock := List ClockHand

/-- Creates a clock with the specified number of hands. -/
def createClock (n : ℕ) : Clock :=
  List.range n |>.map (fun i => { speed := i + 1, fallOffTime := 0 })

/-- Calculates the fall-off time for a specific hand in the clock. -/
def calculateFallOffTime (clock : Clock) (handSpeed : ℕ) : ℚ :=
  sorry

/-- Theorem: The 74th hand in a 150-hand clock falls off after 20 minutes. -/
theorem hand_74_falls_off_after_20_minutes :
  let clock := createClock 150
  calculateFallOffTime clock 74 = 1/3 := by
  sorry

end hand_74_falls_off_after_20_minutes_l4132_413215


namespace dhoni_leftover_earnings_l4132_413260

theorem dhoni_leftover_earnings (rent dishwasher bills car groceries leftover : ℚ) : 
  rent = 20/100 →
  dishwasher = 15/100 →
  bills = 10/100 →
  car = 8/100 →
  groceries = 12/100 →
  leftover = 1 - (rent + dishwasher + bills + car + groceries) →
  leftover = 35/100 := by
sorry

end dhoni_leftover_earnings_l4132_413260


namespace runner_speed_impossibility_l4132_413242

/-- Proves that a runner cannot achieve an average speed of 12 mph over 24 miles
    when two-thirds of the distance has been run at 8 mph -/
theorem runner_speed_impossibility (total_distance : ℝ) (initial_speed : ℝ) (target_speed : ℝ) :
  total_distance = 24 →
  initial_speed = 8 →
  target_speed = 12 →
  (2 / 3 : ℝ) * total_distance / initial_speed = total_distance / target_speed :=
by sorry

end runner_speed_impossibility_l4132_413242


namespace ellipse_equation_specific_l4132_413270

/-- Represents an ellipse in the Cartesian coordinate plane -/
structure Ellipse where
  center : ℝ × ℝ
  foci_axis : ℝ × ℝ
  minor_axis_length : ℝ
  eccentricity : ℝ

/-- The equation of an ellipse given its parameters -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (x - e.center.1)^2 / a^2 + (y - e.center.2)^2 / b^2 = 1 ∧
    e.minor_axis_length = 2 * b ∧
    e.eccentricity = Real.sqrt (1 - b^2 / a^2)

theorem ellipse_equation_specific (e : Ellipse) :
  e.center = (0, 0) →
  e.foci_axis = (1, 0) →
  e.minor_axis_length = 2 →
  e.eccentricity = Real.sqrt 2 / 2 →
  ∀ (x y : ℝ), ellipse_equation e x y ↔ x^2 / 2 + y^2 = 1 := by
  sorry

end ellipse_equation_specific_l4132_413270


namespace seats_between_17_and_39_l4132_413277

/-- The number of seats in the row -/
def total_seats : ℕ := 50

/-- The seat number of the first person -/
def seat1 : ℕ := 17

/-- The seat number of the second person -/
def seat2 : ℕ := 39

/-- The number of seats between two given seat numbers (exclusive) -/
def seats_between (a b : ℕ) : ℕ := 
  if a < b then b - a - 1 else a - b - 1

theorem seats_between_17_and_39 : 
  seats_between seat1 seat2 = 21 := by sorry

end seats_between_17_and_39_l4132_413277


namespace altitude_sum_diff_values_l4132_413244

/-- A right triangle with sides 7, 24, and 25 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 7
  side_b : b = 24
  side_c : c = 25

/-- The two longest altitudes in the right triangle -/
def longest_altitudes (t : RightTriangle) : ℝ × ℝ := (t.a, t.b)

/-- The sum and difference of the two longest altitudes -/
def altitude_sum_diff (t : RightTriangle) : ℝ × ℝ :=
  let (alt1, alt2) := longest_altitudes t
  (alt1 + alt2, |alt1 - alt2|)

theorem altitude_sum_diff_values (t : RightTriangle) :
  altitude_sum_diff t = (31, 17) := by sorry

end altitude_sum_diff_values_l4132_413244


namespace alfred_storage_period_l4132_413292

/-- Calculates the number of years Alfred stores maize -/
def years_storing_maize (
  monthly_storage : ℕ             -- tonnes stored per month
  ) (stolen : ℕ)                  -- tonnes stolen
  (donated : ℕ)                   -- tonnes donated
  (final_amount : ℕ)              -- final amount of maize in tonnes
  : ℕ :=
  (final_amount + stolen - donated) / (monthly_storage * 12)

/-- Theorem stating that Alfred stores maize for 2 years -/
theorem alfred_storage_period :
  years_storing_maize 1 5 8 27 = 2 := by
  sorry

end alfred_storage_period_l4132_413292


namespace product_of_integers_with_given_lcm_and_gcd_l4132_413232

theorem product_of_integers_with_given_lcm_and_gcd :
  ∀ x y : ℕ+, 
  x.val > 0 ∧ y.val > 0 →
  Nat.lcm x.val y.val = 60 →
  Nat.gcd x.val y.val = 5 →
  x.val * y.val = 300 := by
sorry

end product_of_integers_with_given_lcm_and_gcd_l4132_413232


namespace fair_attendance_l4132_413204

/-- Proves the number of adults attending a fair given admission fees, total attendance, and total amount collected. -/
theorem fair_attendance 
  (child_fee : ℚ) 
  (adult_fee : ℚ) 
  (total_people : ℕ) 
  (total_amount : ℚ) 
  (h1 : child_fee = 3/2) 
  (h2 : adult_fee = 4) 
  (h3 : total_people = 2200) 
  (h4 : total_amount = 5050) : 
  ∃ (adults : ℕ), adults = 700 ∧ 
    ∃ (children : ℕ), 
      children + adults = total_people ∧ 
      child_fee * children + adult_fee * adults = total_amount := by
  sorry

end fair_attendance_l4132_413204


namespace tank_emptying_rate_l4132_413255

/-- Proves that given a tank of 30 cubic feet, with an inlet pipe rate of 5 cubic inches/min,
    one outlet pipe rate of 9 cubic inches/min, and a total emptying time of 4320 minutes
    when all pipes are open, the rate of the second outlet pipe is 8 cubic inches/min. -/
theorem tank_emptying_rate (tank_volume : ℝ) (inlet_rate : ℝ) (outlet_rate1 : ℝ)
    (emptying_time : ℝ) (inches_per_foot : ℝ) :
  tank_volume = 30 →
  inlet_rate = 5 →
  outlet_rate1 = 9 →
  emptying_time = 4320 →
  inches_per_foot = 12 →
  ∃ (outlet_rate2 : ℝ),
    outlet_rate2 = 8 ∧
    tank_volume * inches_per_foot^3 = (outlet_rate1 + outlet_rate2 - inlet_rate) * emptying_time :=
by sorry

end tank_emptying_rate_l4132_413255


namespace correct_age_difference_l4132_413276

/-- The difference between Priya's father's age and Priya's age -/
def ageDifference (priyaAge fatherAge : ℕ) : ℕ :=
  fatherAge - priyaAge

theorem correct_age_difference :
  let priyaAge : ℕ := 11
  let fatherAge : ℕ := 42
  let futureSum : ℕ := 69
  let yearsLater : ℕ := 8
  (priyaAge + yearsLater) + (fatherAge + yearsLater) = futureSum →
  ageDifference priyaAge fatherAge = 31 := by
  sorry

end correct_age_difference_l4132_413276


namespace geometric_sequence_property_l4132_413241

/-- Given a geometric sequence {a_n} where a_4 + a_8 = π, 
    prove that a_6(a_2 + 2a_6 + a_10) = π² -/
theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 4 + a 8 = Real.pi →            -- Given condition
  a 6 * (a 2 + 2 * a 6 + a 10) = Real.pi ^ 2 := by
sorry

end geometric_sequence_property_l4132_413241


namespace sum_of_first_100_terms_l4132_413210

-- Define the function f
def f (n : ℕ) : ℤ :=
  if n % 2 = 1 then n^2 else -(n^2)

-- Define the sequence a_n
def a (n : ℕ) : ℤ := f n + f (n + 1)

-- State the theorem
theorem sum_of_first_100_terms :
  (Finset.range 100).sum (λ i => a (i + 1)) = 100 := by
  sorry

end sum_of_first_100_terms_l4132_413210


namespace quadratic_sum_equals_113_l4132_413211

theorem quadratic_sum_equals_113 (x y : ℝ) 
  (eq1 : 3*x + 2*y = 7) 
  (eq2 : 2*x + 3*y = 8) : 
  13*x^2 + 22*x*y + 13*y^2 = 113 := by
  sorry

end quadratic_sum_equals_113_l4132_413211


namespace calculation_result_l4132_413230

theorem calculation_result : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := by
  sorry

end calculation_result_l4132_413230


namespace anne_heavier_than_douglas_l4132_413233

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference in weight between Anne and Douglas -/
def weight_difference : ℕ := anne_weight - douglas_weight

theorem anne_heavier_than_douglas : weight_difference = 15 := by
  sorry

end anne_heavier_than_douglas_l4132_413233


namespace original_price_calculation_l4132_413257

theorem original_price_calculation (total_sale : ℝ) (profit_rate : ℝ) (loss_rate : ℝ)
  (h1 : total_sale = 660)
  (h2 : profit_rate = 0.1)
  (h3 : loss_rate = 0.1) :
  ∃ (original_price : ℝ),
    original_price = total_sale / (1 + profit_rate) + total_sale / (1 - loss_rate) :=
by
  sorry

end original_price_calculation_l4132_413257


namespace hyperbola_focus_to_asymptote_distance_l4132_413291

/-- Given a hyperbola with equation y²/9 - x²/b² = 1 and eccentricity 2,
    the distance from its focus to its asymptote is 3√3 -/
theorem hyperbola_focus_to_asymptote_distance
  (b : ℝ) -- Parameter b of the hyperbola
  (h1 : ∀ x y, y^2/9 - x^2/b^2 = 1) -- Equation of the hyperbola
  (h2 : 2 = (Real.sqrt (9 + b^2)) / 3) -- Eccentricity is 2
  : ∃ (focus : ℝ × ℝ) (asymptote : ℝ → ℝ),
    (∀ x, asymptote x = (Real.sqrt 3 / 3) * x ∨ asymptote x = -(Real.sqrt 3 / 3) * x) ∧
    Real.sqrt ((asymptote (focus.1) - focus.2)^2 / (1 + (Real.sqrt 3 / 3)^2)) = 3 * Real.sqrt 3 :=
sorry

end hyperbola_focus_to_asymptote_distance_l4132_413291


namespace infinitely_many_solutions_l4132_413219

theorem infinitely_many_solutions :
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ,
    let (a, b) := f n
    (a > 0 ∧ b > 0) ∧ a^2 - b^2 = a * b - 1 :=
by sorry

end infinitely_many_solutions_l4132_413219


namespace three_integers_with_difference_and_quotient_l4132_413278

theorem three_integers_with_difference_and_quotient :
  ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a = b - c ∧ b = c / a := by
  sorry

end three_integers_with_difference_and_quotient_l4132_413278


namespace pizza_production_l4132_413201

theorem pizza_production (craig_day1 : ℕ) (craig_increase : ℕ) (heather_decrease : ℕ)
  (h1 : craig_day1 = 40)
  (h2 : craig_increase = 60)
  (h3 : heather_decrease = 20) :
  let heather_day1 := 4 * craig_day1
  let craig_day2 := craig_day1 + craig_increase
  let heather_day2 := craig_day2 - heather_decrease
  heather_day1 + craig_day1 + heather_day2 + craig_day2 = 380 := by
  sorry


end pizza_production_l4132_413201


namespace modular_inverse_of_3_mod_185_l4132_413220

theorem modular_inverse_of_3_mod_185 :
  ∃ x : ℕ, x < 185 ∧ (3 * x) % 185 = 1 :=
by
  use 62
  sorry

end modular_inverse_of_3_mod_185_l4132_413220


namespace triangle_angle_C_l4132_413206

/-- Given a triangle ABC with side lengths a and c, and angle A, prove that C is either 60° or 120°. -/
theorem triangle_angle_C (a c : ℝ) (A : Real) (h1 : a = 2) (h2 : c = Real.sqrt 6) (h3 : A = π / 4) :
  let C := Real.arcsin ((c * Real.sin A) / a)
  C = π / 3 ∨ C = 2 * π / 3 := by
  sorry

end triangle_angle_C_l4132_413206


namespace house_price_calculation_house_price_proof_l4132_413299

theorem house_price_calculation (selling_price : ℝ) 
  (profit_rate : ℝ) (commission_rate : ℝ) : ℝ :=
  let original_price := selling_price / (1 + profit_rate - commission_rate)
  original_price

theorem house_price_proof :
  house_price_calculation 100000 0.2 0.05 = 100000 / 1.15 := by
  sorry

end house_price_calculation_house_price_proof_l4132_413299


namespace compare_exponentials_l4132_413207

theorem compare_exponentials (h1 : 0 < 0.7) (h2 : 0.7 < 0.8) (h3 : 0.8 < 1) :
  0.8^0.7 > 0.7^0.7 ∧ 0.7^0.7 > 0.7^0.8 := by
  sorry

end compare_exponentials_l4132_413207


namespace monotonic_f_implies_a_range_l4132_413252

/-- A function f is monotonic on ℝ if it is either non-decreasing or non-increasing on ℝ. -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∨ (∀ x y : ℝ, x ≤ y → f y ≤ f x)

/-- The function f(x) = x^3 + ax^2 + (a+6)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

theorem monotonic_f_implies_a_range (a : ℝ) :
  Monotonic (f a) → -3 < a ∧ a < 6 := by
  sorry

#check monotonic_f_implies_a_range

end monotonic_f_implies_a_range_l4132_413252


namespace extremum_at_three_l4132_413263

noncomputable def f (x : ℝ) : ℝ := (x - 2) / Real.exp x

theorem extremum_at_three :
  ∀ x₀ : ℝ, (∀ x : ℝ, f x ≤ f x₀) ∨ (∀ x : ℝ, f x ≥ f x₀) → x₀ = 3 :=
by sorry

end extremum_at_three_l4132_413263


namespace last_digit_of_large_exponentiation_l4132_413281

/-- The last digit of a number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Exponentiation modulo 10 -/
def powMod10 (base exponent : ℕ) : ℕ :=
  (base ^ (exponent % 4)) % 10

theorem last_digit_of_large_exponentiation :
  lastDigit (powMod10 954950230952380948328708 470128749397540235934750230) = 4 := by
  sorry

end last_digit_of_large_exponentiation_l4132_413281


namespace nine_bulb_configurations_l4132_413231

def f : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | n + 4 => f (n + 3) + f (n + 2) + f (n + 1) + f n

def circularConfigurations (n : ℕ) : ℕ :=
  f n - 3 * f 3 - 2 * f 2 - f 1

theorem nine_bulb_configurations :
  circularConfigurations 9 = 367 := by sorry

end nine_bulb_configurations_l4132_413231


namespace cab_driver_average_income_l4132_413225

def daily_incomes : List ℝ := [600, 250, 450, 400, 800]

theorem cab_driver_average_income :
  (daily_incomes.sum / daily_incomes.length : ℝ) = 500 := by
  sorry

end cab_driver_average_income_l4132_413225


namespace committee_selection_count_l4132_413254

/-- The number of committee members -/
def total_members : ℕ := 5

/-- The number of roles to be filled -/
def roles_to_fill : ℕ := 3

/-- The number of members ineligible for the entertainment officer role -/
def ineligible_members : ℕ := 2

/-- The number of ways to select members for the given roles under the specified conditions -/
def selection_count : ℕ := 36

theorem committee_selection_count : 
  (total_members - ineligible_members) * 
  (total_members - 1) * 
  (total_members - 2) = selection_count :=
by sorry

end committee_selection_count_l4132_413254


namespace simplify_square_roots_l4132_413200

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 400) + (Real.sqrt 98 / Real.sqrt 56) = (3 + 2 * Real.sqrt 7) / 4 := by
  sorry

end simplify_square_roots_l4132_413200


namespace stratified_sampling_proportion_l4132_413205

theorem stratified_sampling_proportion (total : ℕ) (first_year : ℕ) (second_year : ℕ) 
  (sample_first : ℕ) (sample_second : ℕ) :
  total = first_year + second_year →
  first_year * sample_second = second_year * sample_first →
  sample_first = 6 →
  first_year = 30 →
  second_year = 40 →
  sample_second = 8 :=
by sorry

end stratified_sampling_proportion_l4132_413205


namespace remaining_oranges_l4132_413259

def initial_oranges : ℕ := 60
def oranges_taken : ℕ := 35

theorem remaining_oranges :
  initial_oranges - oranges_taken = 25 := by
  sorry

end remaining_oranges_l4132_413259


namespace ratio_expression_l4132_413235

theorem ratio_expression (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end ratio_expression_l4132_413235


namespace heart_then_ten_probability_l4132_413239

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of hearts in a deck
def num_hearts : ℕ := 13

-- Define the number of 10s in a deck
def num_tens : ℕ := 4

-- Define the probability of the event
def prob_heart_then_ten : ℚ := 1 / total_cards

-- State the theorem
theorem heart_then_ten_probability :
  prob_heart_then_ten = (num_hearts * num_tens) / (total_cards * (total_cards - 1)) :=
sorry

end heart_then_ten_probability_l4132_413239


namespace dresser_contents_l4132_413221

/-- Given a dresser with pants, shorts, and shirts in the ratio 7 : 7 : 10,
    prove that if there are 14 pants, there are 20 shirts. -/
theorem dresser_contents (pants shorts shirts : ℕ) : 
  pants = 14 →
  pants * 10 = shirts * 7 →
  shirts = 20 := by
  sorry

end dresser_contents_l4132_413221


namespace geometric_sequence_property_geometric_sequence_sum_l4132_413268

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m

/-- The property that if m + n = p + q, then a_m * a_n = a_p * a_q for a geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q :=
sorry

theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) 
  (h_sum : a 4 + a 8 = -3) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 :=
sorry

end geometric_sequence_property_geometric_sequence_sum_l4132_413268


namespace range_of_average_l4132_413285

theorem range_of_average (α β : ℝ) (h1 : -π/2 < α) (h2 : α < β) (h3 : β < π/2) :
  -π/2 < (α + β) / 2 ∧ (α + β) / 2 < 0 := by
  sorry

end range_of_average_l4132_413285


namespace greatest_possible_median_l4132_413266

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 18 →
  k < m → m < r → r < s → s < t →
  t = 40 →
  r ≤ 23 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 40) / 5 = 18 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 40 ∧
    r' = 23 := by
  sorry

end greatest_possible_median_l4132_413266


namespace handshake_theorem_l4132_413203

theorem handshake_theorem (n : ℕ) (h : n = 30) :
  let total_handshakes := n * 3 / 2
  total_handshakes = 45 := by
  sorry

end handshake_theorem_l4132_413203


namespace equal_positive_integers_l4132_413280

theorem equal_positive_integers (a b c n : ℕ+) 
  (eq1 : a^2 + b^2 = n * Nat.lcm a b + n^2)
  (eq2 : b^2 + c^2 = n * Nat.lcm b c + n^2)
  (eq3 : c^2 + a^2 = n * Nat.lcm c a + n^2) :
  a = b ∧ b = c := by
  sorry

end equal_positive_integers_l4132_413280


namespace function_property_l4132_413275

def is_periodic (f : ℕ → ℕ) (period : ℕ) : Prop :=
  ∀ n, f (n + period) = f n

theorem function_property (f : ℕ → ℕ) 
  (h1 : ∀ n, f n ≠ 1)
  (h2 : ∀ n, f (n + 1) + f (n + 3) = f (n + 5) * f (n + 7) - 1375) :
  (is_periodic f 4) ∧ 
  (∀ n k, (f (n + 4 * k + 1) - 1) * (f (n + 4 * k + 3) - 1) = 1376) :=
sorry

end function_property_l4132_413275


namespace annika_age_l4132_413256

theorem annika_age (hans_age : ℕ) (annika_age : ℕ) : 
  hans_age = 8 →
  annika_age + 4 = 3 * (hans_age + 4) →
  annika_age = 32 := by
  sorry

end annika_age_l4132_413256


namespace ingrids_tax_rate_l4132_413234

/-- Calculates the tax rate of the second person given the tax rate of the first person,
    both incomes, and their combined tax rate. -/
def calculate_second_tax_rate (first_tax_rate first_income second_income combined_tax_rate : ℚ) : ℚ :=
  let combined_income := first_income + second_income
  let total_tax := combined_tax_rate * combined_income
  let first_tax := first_tax_rate * first_income
  let second_tax := total_tax - first_tax
  second_tax / second_income

/-- Proves that given the specified conditions, Ingrid's tax rate is 40.00% -/
theorem ingrids_tax_rate :
  let john_tax_rate : ℚ := 30 / 100
  let john_income : ℚ := 56000
  let ingrid_income : ℚ := 74000
  let combined_tax_rate : ℚ := 3569 / 10000
  calculate_second_tax_rate john_tax_rate john_income ingrid_income combined_tax_rate = 40 / 100 :=
by sorry

end ingrids_tax_rate_l4132_413234


namespace ones_digit_of_large_power_l4132_413229

theorem ones_digit_of_large_power : ∃ n : ℕ, n < 10 ∧ 34^(34*(17^17)) ≡ n [ZMOD 10] :=
by
  -- The proof goes here
  sorry

end ones_digit_of_large_power_l4132_413229


namespace completing_square_sum_l4132_413246

theorem completing_square_sum (x : ℝ) : 
  (∃ m n : ℝ, (x^2 - 6*x = 1) ↔ ((x - m)^2 = n)) → 
  (∃ m n : ℝ, (x^2 - 6*x = 1) ↔ ((x - m)^2 = n) ∧ m + n = 13) :=
by sorry

end completing_square_sum_l4132_413246


namespace committee_formation_count_l4132_413297

/-- The number of ways to form a committee of size k from n eligible members. -/
def committee_count (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of members in the club. -/
def total_members : ℕ := 12

/-- The size of the committee to be formed. -/
def committee_size : ℕ := 5

/-- The number of ineligible members (Casey). -/
def ineligible_members : ℕ := 1

/-- The number of eligible members for the committee. -/
def eligible_members : ℕ := total_members - ineligible_members

theorem committee_formation_count :
  committee_count eligible_members committee_size = 462 := by
  sorry

end committee_formation_count_l4132_413297


namespace system_solution_l4132_413288

theorem system_solution :
  ∃! (x y : ℚ), 4 * x - 3 * y = 2 ∧ 6 * x + 5 * y = 1 ∧ x = 13/38 ∧ y = -4/19 := by
  sorry

end system_solution_l4132_413288


namespace negation_of_existence_is_universal_not_negation_of_even_prime_l4132_413217

theorem negation_of_existence_is_universal_not (P : ℕ → Prop) :
  (¬ ∃ n, P n) ↔ (∀ n, ¬ P n) := by sorry

theorem negation_of_even_prime :
  (¬ ∃ n : ℕ, Even n ∧ Prime n) ↔ (∀ n : ℕ, Even n → ¬ Prime n) := by sorry

end negation_of_existence_is_universal_not_negation_of_even_prime_l4132_413217


namespace rope_cutting_game_winner_l4132_413295

/-- Represents a player in the rope-cutting game -/
inductive Player : Type
| A : Player
| B : Player

/-- Determines if a number is a power of 3 -/
def isPowerOfThree (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3^k

/-- Represents the rope-cutting game -/
def RopeCuttingGame (a b : ℕ) : Prop :=
  a > 1 ∧ b > 1

/-- Determines if a player has a winning strategy -/
def hasWinningStrategy (p : Player) (a b : ℕ) : Prop :=
  RopeCuttingGame a b →
    (p = Player.B ↔ (a = 2 ∧ b = 3) ∨ isPowerOfThree a)

/-- Main theorem: Player B has a winning strategy iff a = 2 and b = 3, or a is a power of 3 -/
theorem rope_cutting_game_winner (a b : ℕ) :
  RopeCuttingGame a b →
    hasWinningStrategy Player.B a b := by
  sorry

end rope_cutting_game_winner_l4132_413295


namespace square_side_length_average_l4132_413272

theorem square_side_length_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 16) (h₂ : a₂ = 49) (h₃ : a₃ = 169) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 8 := by
  sorry

end square_side_length_average_l4132_413272


namespace time_after_2011_minutes_l4132_413209

/-- Represents a time with day, hour, and minute components -/
structure DateTime where
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Converts total minutes to a DateTime structure -/
def minutesToDateTime (totalMinutes : ℕ) : DateTime :=
  let totalHours := totalMinutes / 60
  let days := totalHours / 24
  let hours := totalHours % 24
  let minutes := totalMinutes % 60
  { day := days + 1, hour := hours, minute := minutes }

/-- The starting date and time -/
def startDateTime : DateTime := { day := 1, hour := 0, minute := 0 }

/-- The number of minutes elapsed -/
def elapsedMinutes : ℕ := 2011

theorem time_after_2011_minutes :
  minutesToDateTime elapsedMinutes = { day := 2, hour := 9, minute := 31 } := by
  sorry

end time_after_2011_minutes_l4132_413209


namespace divisors_of_1800_power_l4132_413265

theorem divisors_of_1800_power (n : Nat) : 
  (∃ (a b c : Nat), (a + 1) * (b + 1) * (c + 1) = 180 ∧
   n = 2^a * 3^b * 5^c ∧ n ∣ 1800^1800) ↔ n ∈ Finset.range 109 :=
by sorry

#check divisors_of_1800_power

end divisors_of_1800_power_l4132_413265


namespace square_cutting_l4132_413290

theorem square_cutting (a b : ℕ+) : 
  4 * a ^ 2 + 3 * b ^ 2 + 10 * a * b = 144 ↔ a = 2 ∧ b = 4 :=
by sorry

end square_cutting_l4132_413290


namespace greatest_of_three_consecutive_integers_l4132_413273

theorem greatest_of_three_consecutive_integers (x y z : ℤ) : 
  (y = x + 1) → (z = y + 1) → (x + y + z = 39) → 
  (max x (max y z) = 14) :=
by sorry

end greatest_of_three_consecutive_integers_l4132_413273


namespace gcd_108_45_l4132_413228

theorem gcd_108_45 : Nat.gcd 108 45 = 9 := by
  sorry

end gcd_108_45_l4132_413228


namespace number_equality_l4132_413227

theorem number_equality : ∃ x : ℝ, (30 / 100) * x = (15 / 100) * 40 ∧ x = 20 := by
  sorry

end number_equality_l4132_413227


namespace binary_1010101_is_85_l4132_413262

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1010101_is_85 :
  binary_to_decimal [true, false, true, false, true, false, true] = 85 := by
  sorry

end binary_1010101_is_85_l4132_413262


namespace blue_faces_cube_l4132_413261

theorem blue_faces_cube (n : ℕ) : n > 0 →
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 := by sorry

end blue_faces_cube_l4132_413261


namespace polynomial_coefficient_sum_l4132_413271

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x + 3) * (3 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 32 := by
sorry

end polynomial_coefficient_sum_l4132_413271


namespace problem_solution_l4132_413216

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1) 2, x^2 - 2*x - m ≤ 0}

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - 2*a) * (x - (a + 1)) ≤ 0}

-- State the theorem
theorem problem_solution :
  (B = Set.Ici 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ B → x ∈ A a) ∧ (∃ x : ℝ, x ∈ A a ∧ x ∉ B) → a ≥ 2) :=
sorry

end problem_solution_l4132_413216


namespace total_money_l4132_413245

/-- The total amount of money A, B, and C have between them is 700, given:
  * A and C together have 300
  * B and C together have 600
  * C has 200 -/
theorem total_money (A B C : ℕ) : 
  A + C = 300 → B + C = 600 → C = 200 → A + B + C = 700 := by
  sorry

end total_money_l4132_413245


namespace indefinite_integral_equality_l4132_413298

/-- The derivative of -(8/9) · √((1 + ∜(x³)) / ∜(x³))³ with respect to x
    is equal to (√(1 + ∜(x³))) / (x² · ⁸√x) for x > 0 -/
theorem indefinite_integral_equality (x : ℝ) (h : x > 0) :
  deriv (fun x => -(8/9) * Real.sqrt ((1 + x^(1/4)) / x^(1/4))^3) x =
  (Real.sqrt (1 + x^(3/4))) / (x^2 * x^(1/8)) :=
sorry

end indefinite_integral_equality_l4132_413298


namespace supermarket_profit_analysis_l4132_413249

/-- Represents a supermarket area with its operating income and net profit percentages -/
structure Area where
  name : String
  operatingIncomePercentage : Float
  netProfitPercentage : Float

/-- Calculates the operating profit rate for an area given the total operating profit rate -/
def calculateOperatingProfitRate (area : Area) (totalOperatingProfitRate : Float) : Float :=
  (area.netProfitPercentage / area.operatingIncomePercentage) * totalOperatingProfitRate

theorem supermarket_profit_analysis 
  (freshArea dailyNecessitiesArea deliArea dairyArea otherArea : Area)
  (totalOperatingProfitRate : Float) :
  freshArea.name = "Fresh Area" →
  freshArea.operatingIncomePercentage = 48.6 →
  freshArea.netProfitPercentage = 65.8 →
  dailyNecessitiesArea.name = "Daily Necessities Area" →
  dailyNecessitiesArea.operatingIncomePercentage = 10.8 →
  dailyNecessitiesArea.netProfitPercentage = 20.2 →
  deliArea.name = "Deli Area" →
  deliArea.operatingIncomePercentage = 15.8 →
  deliArea.netProfitPercentage = -4.3 →
  dairyArea.name = "Dairy Area" →
  dairyArea.operatingIncomePercentage = 20.1 →
  dairyArea.netProfitPercentage = 16.5 →
  otherArea.name = "Other Area" →
  otherArea.operatingIncomePercentage = 4.7 →
  otherArea.netProfitPercentage = 1.8 →
  totalOperatingProfitRate = 32.5 →
  (freshArea.netProfitPercentage > 50) ∧ 
  (calculateOperatingProfitRate dailyNecessitiesArea totalOperatingProfitRate > 
   max (calculateOperatingProfitRate freshArea totalOperatingProfitRate)
       (max (calculateOperatingProfitRate deliArea totalOperatingProfitRate)
            (max (calculateOperatingProfitRate dairyArea totalOperatingProfitRate)
                 (calculateOperatingProfitRate otherArea totalOperatingProfitRate)))) ∧
  (calculateOperatingProfitRate freshArea totalOperatingProfitRate > 40) := by
  sorry

end supermarket_profit_analysis_l4132_413249


namespace problem_statement_l4132_413222

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^x < a^y

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem problem_statement (a : ℝ) (h1 : a > 0) (h2 : (p a ∨ q a) ∧ ¬(p a ∧ q a)) :
  a ≥ 4 ∨ (0 < a ∧ a ≤ 1) :=
sorry

end problem_statement_l4132_413222


namespace arctan_sum_equals_pi_third_l4132_413236

theorem arctan_sum_equals_pi_third (n : ℕ+) : 
  Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/n) = π/3 → n = 84 := by
  sorry

end arctan_sum_equals_pi_third_l4132_413236


namespace stating_reduce_to_zero_iff_even_odds_l4132_413284

/-- 
Given a natural number n, this function returns true if it's possible to reduce
all numbers in the sequence 1 to n to zero by repeatedly replacing any two numbers
with their difference, and false otherwise.
-/
def canReduceToZero (n : ℕ) : Prop :=
  Even ((n + 1) / 2)

/-- 
Theorem stating that for a sequence of integers from 1 to n, it's possible to reduce
all numbers to zero using the given operation if and only if the number of odd integers
in the sequence is even.
-/
theorem reduce_to_zero_iff_even_odds (n : ℕ) :
  canReduceToZero n ↔ Even ((n + 1) / 2) := by sorry

end stating_reduce_to_zero_iff_even_odds_l4132_413284


namespace expression_equals_one_l4132_413237

theorem expression_equals_one : 
  (144^2 - 12^2) / (120^2 - 18^2) * ((120-18)*(120+18)) / ((144-12)*(144+12)) = 1 := by
  sorry

end expression_equals_one_l4132_413237


namespace susan_initial_money_l4132_413264

theorem susan_initial_money (S : ℝ) : 
  S - (S / 5 + S / 4 + 120) = 1200 → S = 2400 := by
  sorry

end susan_initial_money_l4132_413264


namespace hcl_effects_l4132_413274

-- Define the initial state of distilled water
structure DistilledWater :=
  (temp : ℝ)
  (pH : ℝ)
  (c_H : ℝ)
  (c_OH : ℝ)
  (Kw : ℝ)

-- Define the state after adding HCl
structure WaterWithHCl :=
  (temp : ℝ)
  (pH : ℝ)
  (c_H : ℝ)
  (c_OH : ℝ)
  (Kw : ℝ)
  (c_HCl : ℝ)

-- Define the theorem
theorem hcl_effects 
  (initial : DistilledWater) 
  (final : WaterWithHCl) 
  (h_temp : final.temp = initial.temp) 
  (h_HCl : final.c_HCl > 0) :
  (final.Kw = initial.Kw) ∧ 
  (final.pH < initial.pH) ∧ 
  (final.c_OH < initial.c_OH) ∧ 
  (final.c_H - final.c_HCl < initial.c_H) :=
sorry

end hcl_effects_l4132_413274


namespace product_four_consecutive_odd_integers_is_nine_l4132_413212

theorem product_four_consecutive_odd_integers_is_nine :
  ∃ n : ℤ, (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) = 9 :=
by sorry

end product_four_consecutive_odd_integers_is_nine_l4132_413212


namespace no_four_identical_digits_in_1990_denominator_l4132_413202

theorem no_four_identical_digits_in_1990_denominator :
  ¬ ∃ (A : ℕ) (d : ℕ), 
    A > 0 ∧ A < 1990 ∧ d < 10 ∧
    ∃ (k : ℕ), (A * 10^k) % 1990 = d * 1111 :=
by sorry

end no_four_identical_digits_in_1990_denominator_l4132_413202


namespace whitewashing_cost_is_16820_l4132_413283

/-- Calculates the cost of whitewashing a room with given dimensions and openings. -/
def whitewashing_cost (room_length room_width room_height : ℝ)
                      (door1_length door1_width : ℝ)
                      (door2_length door2_width : ℝ)
                      (window1_length window1_width : ℝ)
                      (window2_length window2_width : ℝ)
                      (window3_length window3_width : ℝ)
                      (window4_length window4_width : ℝ)
                      (window5_length window5_width : ℝ)
                      (cost_per_sqft : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let openings_area := door1_length * door1_width + door2_length * door2_width +
                       window1_length * window1_width + window2_length * window2_width +
                       window3_length * window3_width + window4_length * window4_width +
                       window5_length * window5_width
  let whitewash_area := wall_area - openings_area
  whitewash_area * cost_per_sqft

/-- The cost of whitewashing the room with given dimensions and openings is Rs. 16820. -/
theorem whitewashing_cost_is_16820 :
  whitewashing_cost 40 20 15 7 4 5 3 5 4 4 3 3 3 4 2.5 6 4 10 = 16820 := by
  sorry


end whitewashing_cost_is_16820_l4132_413283


namespace order_of_abc_l4132_413208

noncomputable def a : ℝ := (4 - Real.log 4) / Real.exp 2
noncomputable def b : ℝ := Real.log 2 / 2
noncomputable def c : ℝ := 1 / Real.exp 1

theorem order_of_abc : b < a ∧ a < c := by sorry

end order_of_abc_l4132_413208


namespace function_composition_l4132_413289

theorem function_composition (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end function_composition_l4132_413289


namespace quadratic_form_ratio_l4132_413223

theorem quadratic_form_ratio (x : ℝ) :
  let f := x^2 + 2600*x + 2600
  ∃ d e : ℝ, (∀ x, f = (x + d)^2 + e) ∧ e / d = -1298 := by
sorry

end quadratic_form_ratio_l4132_413223


namespace largest_C_inequality_l4132_413253

theorem largest_C_inequality : 
  ∃ (C : ℝ), C = 17/4 ∧ 
  (∀ (x y : ℝ), y ≥ 4*x ∧ x > 0 → x^2 + y^2 ≥ C*x*y) ∧
  (∀ (C' : ℝ), C' > C → 
    ∃ (x y : ℝ), y ≥ 4*x ∧ x > 0 ∧ x^2 + y^2 < C'*x*y) :=
sorry

end largest_C_inequality_l4132_413253


namespace no_common_points_l4132_413267

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem stating that there are no common points
theorem no_common_points : ¬ ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end no_common_points_l4132_413267


namespace only_statement5_true_l4132_413294

-- Define the statements as functions
def statement1 (a b : ℝ) : Prop := b * (a + b) = b * a + b * b
def statement2 (x y : ℝ) : Prop := Real.log (x + y) = Real.log x + Real.log y
def statement3 (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2
def statement4 (a b : ℝ) : Prop := b^(a + b) = b^a + b^b
def statement5 (x y : ℝ) : Prop := x^2 / y^2 = (x / y)^2

-- Theorem stating that only statement5 is true for all real numbers
theorem only_statement5_true :
  (∀ x y : ℝ, statement5 x y) ∧
  (∃ a b : ℝ, ¬statement1 a b) ∧
  (∃ x y : ℝ, ¬statement2 x y) ∧
  (∃ x y : ℝ, ¬statement3 x y) ∧
  (∃ a b : ℝ, ¬statement4 a b) :=
sorry

end only_statement5_true_l4132_413294


namespace min_value_of_expression_l4132_413296

theorem min_value_of_expression (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0) 
  (h_inequality : x₁^2 - 4*a*x₁ + 3*a^2 < 0 ∧ x₂^2 - 4*a*x₂ + 3*a^2 < 0) :
  ∃ (m : ℝ), m = (4 * Real.sqrt 3) / 3 ∧ 
  ∀ (y₁ y₂ : ℝ), (y₁^2 - 4*a*y₁ + 3*a^2 < 0 ∧ y₂^2 - 4*a*y₂ + 3*a^2 < 0) → 
  (y₁ + y₂ + a / (y₁ * y₂)) ≥ m :=
sorry

end min_value_of_expression_l4132_413296


namespace expand_and_simplify_l4132_413240

theorem expand_and_simplify (y : ℝ) : 5 * (6 * y^2 - 3 * y + 2 - 4 * y^3) = -20 * y^3 + 30 * y^2 - 15 * y + 10 := by
  sorry

end expand_and_simplify_l4132_413240
