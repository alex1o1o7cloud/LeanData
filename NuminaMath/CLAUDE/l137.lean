import Mathlib

namespace NUMINAMATH_CALUDE_rotation_result_l137_13746

-- Define the shapes
inductive Shape
| Triangle
| Circle
| Square

-- Define the position of shapes in the figure
structure Figure :=
(pos1 : Shape)
(pos2 : Shape)
(pos3 : Shape)

-- Define the rotation operation
def rotate120 (f : Figure) : Figure :=
{ pos1 := f.pos3,
  pos2 := f.pos1,
  pos3 := f.pos2 }

-- Theorem statement
theorem rotation_result (f : Figure) 
  (h1 : f.pos1 ≠ f.pos2) 
  (h2 : f.pos2 ≠ f.pos3) 
  (h3 : f.pos3 ≠ f.pos1) : 
  rotate120 f = 
  { pos1 := f.pos3,
    pos2 := f.pos1,
    pos3 := f.pos2 } := by
  sorry

#check rotation_result

end NUMINAMATH_CALUDE_rotation_result_l137_13746


namespace NUMINAMATH_CALUDE_total_balloons_l137_13777

/-- Given an initial number of balloons and an additional number of balloons,
    the total number of balloons is equal to their sum. -/
theorem total_balloons (initial additional : ℕ) :
  initial + additional = (initial + additional) := by sorry

end NUMINAMATH_CALUDE_total_balloons_l137_13777


namespace NUMINAMATH_CALUDE_tangent_perpendicular_condition_l137_13771

/-- The function f(x) = x³ - x² + ax + b -/
def f (a b x : ℝ) : ℝ := x^3 - x^2 + a*x + b

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_perpendicular_condition (a b : ℝ) : 
  (f_derivative a 1) * 2 = -1 ↔ a = -3/2 := by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_condition_l137_13771


namespace NUMINAMATH_CALUDE_no_base_ends_with_one_l137_13705

theorem no_base_ends_with_one : 
  ∀ b : ℕ, 3 ≤ b ∧ b ≤ 10 → ¬(842 % b = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_base_ends_with_one_l137_13705


namespace NUMINAMATH_CALUDE_replaced_man_age_l137_13760

theorem replaced_man_age
  (total_men : Nat)
  (age_increase : Nat)
  (known_man_age : Nat)
  (women_avg_age : Nat)
  (h1 : total_men = 7)
  (h2 : age_increase = 4)
  (h3 : known_man_age = 26)
  (h4 : women_avg_age = 42) :
  ∃ (replaced_man_age : Nat),
    replaced_man_age = 30 ∧
    (∃ (initial_avg : ℚ),
      (total_men : ℚ) * initial_avg =
        (total_men - 2 : ℚ) * (initial_avg + age_increase) +
        2 * women_avg_age -
        (known_man_age + replaced_man_age : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_replaced_man_age_l137_13760


namespace NUMINAMATH_CALUDE_min_b_over_a_l137_13706

open Real

/-- The minimum value of b/a given the conditions on f(x) -/
theorem min_b_over_a (f : ℝ → ℝ) (a b : ℝ) (h : ∀ x > 0, f x ≤ 0) :
  (∀ x > 0, f x = log x + (2 * exp 2 - a) * x - b / 2) →
  ∃ m : ℝ, m = -2 / exp 2 ∧ ∀ k : ℝ, (∃ a' b' : ℝ, (∀ x > 0, f x = log x + (2 * exp 2 - a') * x - b' / 2) ∧ b' / a' = k) → k ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_b_over_a_l137_13706


namespace NUMINAMATH_CALUDE_equation_solution_l137_13707

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ (x^2 - x + 2) / (x - 1) = x + 3 ∧ x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l137_13707


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l137_13719

theorem min_value_trig_expression (α β : ℝ) :
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 ≥ 100 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ α β : ℝ, (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l137_13719


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l137_13749

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 4) (h2 : l^2 + w^2 = d^2) :
  l * w = (20 / 41) * d^2 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l137_13749


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l137_13703

theorem simplify_and_evaluate (a b : ℤ) (ha : a = 2) (hb : b = -1) :
  (a + 3*b)^2 + (a + 3*b)*(a - 3*b) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l137_13703


namespace NUMINAMATH_CALUDE_supplements_calculation_l137_13775

/-- The number of boxes of supplements delivered by Mr. Anderson -/
def boxes_of_supplements : ℕ := 760 - 472

/-- The total number of boxes of medicine delivered -/
def total_boxes : ℕ := 760

/-- The number of boxes of vitamins delivered -/
def vitamin_boxes : ℕ := 472

theorem supplements_calculation :
  boxes_of_supplements = 288 :=
by sorry

end NUMINAMATH_CALUDE_supplements_calculation_l137_13775


namespace NUMINAMATH_CALUDE_second_number_is_six_l137_13733

theorem second_number_is_six (x y : ℝ) (h : 3 * y - x = 2 * y + 6) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_six_l137_13733


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l137_13762

theorem quadratic_equations_solutions :
  (∀ x, 4 * x^2 = 12 * x ↔ x = 0 ∨ x = 3) ∧
  (∀ x, 3/4 * x^2 - 2*x - 1/2 = 0 ↔ x = (4 + Real.sqrt 22) / 3 ∨ x = (4 - Real.sqrt 22) / 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l137_13762


namespace NUMINAMATH_CALUDE_shooting_probabilities_l137_13735

/-- Probability of hitting a specific ring in one shot -/
def ring_probability : Fin 3 → ℝ
| 0 => 0.13  -- 10-ring
| 1 => 0.28  -- 9-ring
| 2 => 0.31  -- 8-ring

/-- The sum of probabilities for 10-ring and 9-ring -/
def prob_10_or_9 : ℝ := ring_probability 0 + ring_probability 1

/-- The probability of hitting less than 9 rings -/
def prob_less_than_9 : ℝ := 1 - prob_10_or_9

theorem shooting_probabilities :
  prob_10_or_9 = 0.41 ∧ prob_less_than_9 = 0.59 := by sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l137_13735


namespace NUMINAMATH_CALUDE_candle_height_ratio_time_l137_13738

/-- Represents a candle with its initial height and burning time. -/
structure Candle where
  initial_height : ℝ
  burning_time : ℝ

/-- The problem setup -/
def candle_problem : Prop :=
  let candle_a : Candle := { initial_height := 12, burning_time := 6 }
  let candle_b : Candle := { initial_height := 15, burning_time := 5 }
  let burn_rate (c : Candle) : ℝ := c.initial_height / c.burning_time
  let height_at_time (c : Candle) (t : ℝ) : ℝ := c.initial_height - (burn_rate c) * t
  ∃ t : ℝ, t > 0 ∧ height_at_time candle_a t = (1/3) * height_at_time candle_b t ∧ t = 7

/-- The theorem to be proved -/
theorem candle_height_ratio_time : candle_problem := by
  sorry

end NUMINAMATH_CALUDE_candle_height_ratio_time_l137_13738


namespace NUMINAMATH_CALUDE_amy_flash_drive_files_l137_13734

theorem amy_flash_drive_files (initial_music : ℕ) (initial_video : ℕ) (deleted : ℕ) (downloaded : ℕ)
  (h1 : initial_music = 26)
  (h2 : initial_video = 36)
  (h3 : deleted = 48)
  (h4 : downloaded = 15) :
  initial_music + initial_video - deleted + downloaded = 29 := by
  sorry

end NUMINAMATH_CALUDE_amy_flash_drive_files_l137_13734


namespace NUMINAMATH_CALUDE_calculation_proof_l137_13741

theorem calculation_proof : 2⁻¹ + Real.sin (30 * π / 180) - (π - 3.14)^0 + abs (-3) - Real.sqrt 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l137_13741


namespace NUMINAMATH_CALUDE_square_divisibility_l137_13791

theorem square_divisibility (n : ℤ) : ∃ k : ℤ, n^2 = 4*k ∨ n^2 = 4*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l137_13791


namespace NUMINAMATH_CALUDE_conference_attendees_payment_registration_l137_13780

theorem conference_attendees_payment_registration (
  early_registration : Real) 
  (mid_registration : Real)
  (late_registration : Real)
  (credit_card_percent : Real)
  (debit_card_percent : Real)
  (other_payment_percent : Real) :
  early_registration = 80 →
  mid_registration = 12 →
  late_registration = 100 - early_registration - mid_registration →
  credit_card_percent + debit_card_percent + other_payment_percent = 100 →
  credit_card_percent = 20 →
  debit_card_percent = 60 →
  other_payment_percent = 20 →
  early_registration + mid_registration = 
    (credit_card_percent + debit_card_percent + other_payment_percent) * 
    (early_registration + mid_registration) / 100 :=
by sorry

end NUMINAMATH_CALUDE_conference_attendees_payment_registration_l137_13780


namespace NUMINAMATH_CALUDE_shirts_cost_after_discount_l137_13712

def first_shirt_cost : ℝ := 15
def price_difference : ℝ := 6
def discount_rate : ℝ := 0.1

def second_shirt_cost : ℝ := first_shirt_cost - price_difference
def total_cost : ℝ := first_shirt_cost + second_shirt_cost
def discounted_cost : ℝ := total_cost * (1 - discount_rate)

theorem shirts_cost_after_discount :
  discounted_cost = 21.60 := by sorry

end NUMINAMATH_CALUDE_shirts_cost_after_discount_l137_13712


namespace NUMINAMATH_CALUDE_kenneth_remaining_money_l137_13742

def remaining_money (initial_amount baguette_cost water_cost baguette_count water_count : ℕ) : ℕ :=
  initial_amount - (baguette_cost * baguette_count + water_cost * water_count)

theorem kenneth_remaining_money :
  remaining_money 50 2 1 2 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_remaining_money_l137_13742


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l137_13770

/-- The eccentricity of an ellipse with given properties is 1/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (d₁ d₂ : ℝ → ℝ → ℝ) : 
  a > b ∧ b > 0 →
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → d₁ x y + d₂ x y = 2*a) →
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → d₁ x y + d₂ x y = 4*c) →
  c/a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l137_13770


namespace NUMINAMATH_CALUDE_kendra_minivans_count_l137_13797

/-- The number of minivans Kendra saw in the afternoon -/
def afternoon_minivans : ℕ := 4

/-- The number of minivans Kendra saw in the evening -/
def evening_minivans : ℕ := 1

/-- The total number of minivans Kendra saw during her trip -/
def total_minivans : ℕ := afternoon_minivans + evening_minivans

theorem kendra_minivans_count : total_minivans = 5 := by
  sorry

end NUMINAMATH_CALUDE_kendra_minivans_count_l137_13797


namespace NUMINAMATH_CALUDE_joey_exam_in_six_weeks_l137_13785

/-- Joey's SAT exam preparation schedule --/
structure SATPrep where
  weekday_hours : ℕ  -- Hours studied per weekday night
  weekday_nights : ℕ  -- Number of weekday nights studied per week
  weekend_hours : ℕ  -- Hours studied per weekend day
  total_hours : ℕ  -- Total hours to be studied

/-- Calculate the number of weeks until Joey's SAT exam --/
def weeks_until_exam (prep : SATPrep) : ℚ :=
  prep.total_hours / (prep.weekday_hours * prep.weekday_nights + prep.weekend_hours * 2)

/-- Theorem: Joey's SAT exam is 6 weeks away --/
theorem joey_exam_in_six_weeks (prep : SATPrep) 
  (h1 : prep.weekday_hours = 2)
  (h2 : prep.weekday_nights = 5)
  (h3 : prep.weekend_hours = 3)
  (h4 : prep.total_hours = 96) : 
  weeks_until_exam prep = 6 := by
  sorry

end NUMINAMATH_CALUDE_joey_exam_in_six_weeks_l137_13785


namespace NUMINAMATH_CALUDE_bicycle_rental_theorem_l137_13727

/-- Represents the rental time for a bicycle. -/
inductive RentalTime
  | LessThanTwo
  | TwoToThree
  | ThreeToFour

/-- Calculates the rental fee based on the rental time. -/
def rentalFee (time : RentalTime) : ℕ :=
  match time with
  | RentalTime.LessThanTwo => 0
  | RentalTime.TwoToThree => 2
  | RentalTime.ThreeToFour => 4

/-- Represents the probabilities for each rental time for a person. -/
structure RentalProbabilities where
  lessThanTwo : ℚ
  twoToThree : ℚ
  threeToFour : ℚ

/-- The rental probabilities for person A. -/
def probA : RentalProbabilities :=
  { lessThanTwo := 1/4, twoToThree := 1/2, threeToFour := 1/4 }

/-- The rental probabilities for person B. -/
def probB : RentalProbabilities :=
  { lessThanTwo := 1/2, twoToThree := 1/4, threeToFour := 1/4 }

/-- Calculates the probability that two people pay the same fee. -/
def probSameFee (pA pB : RentalProbabilities) : ℚ :=
  pA.lessThanTwo * pB.lessThanTwo +
  pA.twoToThree * pB.twoToThree +
  pA.threeToFour * pB.threeToFour

/-- Calculates the expected value of the sum of fees for two people. -/
def expectedSumFees (pA pB : RentalProbabilities) : ℚ :=
  0 * (pA.lessThanTwo * pB.lessThanTwo) +
  2 * (pA.lessThanTwo * pB.twoToThree + pA.twoToThree * pB.lessThanTwo) +
  4 * (pA.lessThanTwo * pB.threeToFour + pA.twoToThree * pB.twoToThree + pA.threeToFour * pB.lessThanTwo) +
  6 * (pA.twoToThree * pB.threeToFour + pA.threeToFour * pB.twoToThree) +
  8 * (pA.threeToFour * pB.threeToFour)

theorem bicycle_rental_theorem :
  probSameFee probA probB = 5/16 ∧
  expectedSumFees probA probB = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_rental_theorem_l137_13727


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_expansion_l137_13740

/-- The repeating decimal 0.73246̅ expressed as a fraction with denominator 999900 -/
def repeating_decimal : ℚ :=
  731514 / 999900

/-- The repeating decimal 0.73246̅ as a real number -/
noncomputable def decimal_expansion : ℝ :=
  0.73 + (246 : ℝ) / 1000 * (1 / (1 - 1/1000))

theorem repeating_decimal_equals_expansion :
  (repeating_decimal : ℝ) = decimal_expansion :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_expansion_l137_13740


namespace NUMINAMATH_CALUDE_room_length_calculation_l137_13759

/-- Given a rectangular room with width 12 m, surrounded by a 2 m wide veranda on all sides,
    and the area of the veranda being 148 m², the length of the room is 21 m. -/
theorem room_length_calculation (room_width : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_width = 12 →
  veranda_width = 2 →
  veranda_area = 148 →
  ∃ (room_length : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
    room_length * room_width = veranda_area ∧
    room_length = 21 :=
by sorry

end NUMINAMATH_CALUDE_room_length_calculation_l137_13759


namespace NUMINAMATH_CALUDE_boys_in_jakes_class_l137_13763

/-- Calculates the number of boys in a class given the ratio of girls to boys and the total number of students -/
def number_of_boys (girls_ratio : ℕ) (boys_ratio : ℕ) (total_students : ℕ) : ℕ :=
  (boys_ratio * total_students) / (girls_ratio + boys_ratio)

/-- Proves that in a class with a 3:4 ratio of girls to boys and 35 total students, there are 20 boys -/
theorem boys_in_jakes_class :
  number_of_boys 3 4 35 = 20 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_jakes_class_l137_13763


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l137_13729

/-- Given a rectangle with perimeter 60 feet and area 221 square feet, 
    the length of the longer side is 17 feet. -/
theorem rectangle_longer_side (x y : ℝ) 
  (h_perimeter : 2 * x + 2 * y = 60) 
  (h_area : x * y = 221) 
  (h_longer : x ≥ y) : x = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l137_13729


namespace NUMINAMATH_CALUDE_hyperbola_parabola_focus_coincidence_l137_13728

/-- The value of p for which the right focus of the hyperbola x^2 - y^2/3 = 1 
    coincides with the focus of the parabola y^2 = 2px -/
theorem hyperbola_parabola_focus_coincidence (p : ℝ) : 
  (∃ (x y : ℝ), x^2 - y^2/3 = 1 ∧ y^2 = 2*p*x ∧ 
   (x, y) = (2, 0) ∧ (x, y) = (p/2, 0)) → 
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_focus_coincidence_l137_13728


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l137_13739

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}

theorem complement_of_N_in_M :
  (M \ N) = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l137_13739


namespace NUMINAMATH_CALUDE_triangle_inequality_l137_13786

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l137_13786


namespace NUMINAMATH_CALUDE_no_solution_exists_l137_13717

theorem no_solution_exists : ¬∃ x : ℝ, (16 : ℝ)^(3*x - 1) = (64 : ℝ)^(2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l137_13717


namespace NUMINAMATH_CALUDE_function_properties_l137_13756

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x ∈ Set.Ioo (-3 : ℝ) 2, f a b x > 0) ∧
    (∀ x ∈ Set.Iic (-3 : ℝ) ∪ Set.Ici 2, f a b x < 0) ∧
    (f a b (-3) = 0 ∧ f a b 2 = 0) →
    (∀ x, f a b x = -3 * x^2 - 3 * x + 18) ∧
    (∀ c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c ≤ 0) ↔ c ≤ -25/12) ∧
    (∃ y_max : ℝ, y_max = -3 ∧
      ∀ x > -1, (f a b x - 21) / (x + 1) ≤ y_max) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l137_13756


namespace NUMINAMATH_CALUDE_range_of_m_solution_when_m_minimum_l137_13781

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |5 - 2*x| - |2*x - 1|

-- Theorem for the range of m
theorem range_of_m :
  (∃ x, f m x = 0) → m ∈ Set.Ici 4 :=
sorry

-- Theorem for the solution of the inequality when m is minimum
theorem solution_when_m_minimum :
  let m : ℝ := 4
  ∀ x, |x - 3| + |x + m| ≤ 2*m ↔ x ∈ Set.Icc (-9/2) (7/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_solution_when_m_minimum_l137_13781


namespace NUMINAMATH_CALUDE_next_integer_divisibility_l137_13711

theorem next_integer_divisibility (n : ℕ) :
  ∃ k : ℤ, (k : ℝ) = ⌊(Real.sqrt 3 + 1)^(2*n)⌋ + 1 ∧ (2^(n+1) : ℤ) ∣ k :=
sorry

end NUMINAMATH_CALUDE_next_integer_divisibility_l137_13711


namespace NUMINAMATH_CALUDE_linear_equation_solution_l137_13751

theorem linear_equation_solution (m : ℝ) : (2 * m + 2 = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l137_13751


namespace NUMINAMATH_CALUDE_semicircle_radius_l137_13790

theorem semicircle_radius (x y z : ℝ) (h_right_angle : x^2 + y^2 = z^2)
  (h_xy_area : π * x^2 / 2 = 12 * π) (h_xz_arc : π * y = 10 * π) :
  z / 2 = 2 * Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l137_13790


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l137_13788

theorem quadratic_through_origin (a : ℝ) : 
  (∀ x y : ℝ, y = (a - 1) * x^2 - x + a^2 - 1 → (x = 0 → y = 0)) → 
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l137_13788


namespace NUMINAMATH_CALUDE_correct_division_result_l137_13702

theorem correct_division_result (incorrect_divisor incorrect_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 48)
  (h2 : incorrect_quotient = 24)
  (h3 : correct_divisor = 36) :
  (incorrect_divisor * incorrect_quotient) / correct_divisor = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_division_result_l137_13702


namespace NUMINAMATH_CALUDE_distance_between_foci_l137_13773

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) + Real.sqrt ((x + 6)^2 + (y - 7)^2) = 24

/-- The first focus of the ellipse -/
def F₁ : ℝ × ℝ := (2, -3)

/-- The second focus of the ellipse -/
def F₂ : ℝ × ℝ := (-6, 7)

/-- The theorem stating the distance between the foci -/
theorem distance_between_foci :
  Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l137_13773


namespace NUMINAMATH_CALUDE_smallest_divisor_divisor_is_four_l137_13743

theorem smallest_divisor (d : ℕ) : d > 0 ∧ d > 3 ∧
  (∃ n : ℤ, n % d = 1 ∧ (3 * n) % d = 3) →
  d ≥ 4 :=
sorry

theorem divisor_is_four : ∃ d : ℕ, d > 0 ∧ d > 3 ∧
  (∃ n : ℤ, n % d = 1 ∧ (3 * n) % d = 3) ∧
  ∀ k : ℕ, k > 0 ∧ k > 3 ∧ (∃ m : ℤ, m % k = 1 ∧ (3 * m) % k = 3) →
  k ≥ d :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_divisor_is_four_l137_13743


namespace NUMINAMATH_CALUDE_xy_value_l137_13722

theorem xy_value (x y : ℝ) (h : |x^3 - 1/8| + Real.sqrt (y - 4) = 0) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l137_13722


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l137_13726

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := sorry

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 23

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 21

/-- The number of extra apples -/
def extra_apples : ℕ := 35

/-- Theorem stating that the number of red apples ordered is 33 -/
theorem cafeteria_red_apples : 
  red_apples = 33 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_red_apples_l137_13726


namespace NUMINAMATH_CALUDE_multiples_of_nine_count_l137_13745

theorem multiples_of_nine_count (N : ℕ) : 
  (∃ (count : ℕ), count = (Nat.div N 9 - Nat.div 10 9 + 1) ∧ count = 1110) → N = 9989 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_nine_count_l137_13745


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l137_13708

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l137_13708


namespace NUMINAMATH_CALUDE_regular_pentagon_ratio_sum_l137_13730

/-- For a regular pentagon with side length a and diagonal length b, (a/b + b/a) = √5 -/
theorem regular_pentagon_ratio_sum (a b : ℝ) (h : a / b = (Real.sqrt 5 - 1) / 2) :
  a / b + b / a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_ratio_sum_l137_13730


namespace NUMINAMATH_CALUDE_subtract_like_terms_l137_13799

theorem subtract_like_terms (a b : ℝ) : 5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l137_13799


namespace NUMINAMATH_CALUDE_road_trip_ratio_l137_13783

/-- Road trip distance calculation -/
theorem road_trip_ratio : 
  ∀ (D R : ℝ),
  D > 0 →
  R > 0 →
  D / 2 = 40 →
  2 * (D + R * D + 40) = 560 - (D + R * D + 40) →
  R = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_road_trip_ratio_l137_13783


namespace NUMINAMATH_CALUDE_axis_of_symmetry_translated_trig_l137_13752

/-- The axis of symmetry of a translated trigonometric function -/
theorem axis_of_symmetry_translated_trig (k : ℤ) :
  let f (x : ℝ) := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)
  let g (x : ℝ) := f (x + π / 6)
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), 
    g x = A * Real.sin (B * x + C) ∧
    (x = k * π / 2 - π / 12) → (B * x + C = n * π + π / 2) :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_translated_trig_l137_13752


namespace NUMINAMATH_CALUDE_correct_average_weight_l137_13710

/-- Given a class of boys with an incorrect average weight due to a misread measurement,
    calculate the correct average weight. -/
theorem correct_average_weight
  (n : ℕ) -- number of boys
  (initial_avg : ℝ) -- initial (incorrect) average weight
  (misread_weight : ℝ) -- weight that was misread
  (correct_weight : ℝ) -- correct weight for the misread value
  (h1 : n = 20) -- there are 20 boys
  (h2 : initial_avg = 58.4) -- initial average was 58.4 kg
  (h3 : misread_weight = 56) -- misread weight was 56 kg
  (h4 : correct_weight = 60) -- correct weight is 60 kg
  : (n * initial_avg + correct_weight - misread_weight) / n = 58.6 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l137_13710


namespace NUMINAMATH_CALUDE_julia_tag_game_l137_13724

/-- 
Given that Julia played tag with a total of 18 kids over two days,
and she played with 14 kids on Tuesday, prove that she played with 4 kids on Monday.
-/
theorem julia_tag_game (total : ℕ) (tuesday : ℕ) (monday : ℕ) 
    (h1 : total = 18) 
    (h2 : tuesday = 14) 
    (h3 : total = monday + tuesday) : 
  monday = 4 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l137_13724


namespace NUMINAMATH_CALUDE_dani_pants_after_five_years_l137_13784

/-- Calculates the total number of pants after a given number of years -/
def totalPantsAfterYears (initialPants : ℕ) (pairsPerYear : ℕ) (pantsPerPair : ℕ) (years : ℕ) : ℕ :=
  initialPants + years * pairsPerYear * pantsPerPair

/-- Theorem: Given the initial conditions, Dani will have 90 pants after 5 years -/
theorem dani_pants_after_five_years :
  totalPantsAfterYears 50 4 2 5 = 90 := by
  sorry

#eval totalPantsAfterYears 50 4 2 5

end NUMINAMATH_CALUDE_dani_pants_after_five_years_l137_13784


namespace NUMINAMATH_CALUDE_bus_students_problem_l137_13709

theorem bus_students_problem (initial_students final_students : ℕ) 
  (h1 : initial_students = 28)
  (h2 : final_students = 58) :
  (0.4 : ℝ) * (final_students - initial_students) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_problem_l137_13709


namespace NUMINAMATH_CALUDE_waitress_hourly_wage_l137_13755

/-- Calculates the hourly wage of a waitress given her work hours, tips, and total earnings -/
theorem waitress_hourly_wage 
  (monday_hours tuesday_hours wednesday_hours : ℕ)
  (monday_tips tuesday_tips wednesday_tips : ℚ)
  (total_earnings : ℚ)
  (h1 : monday_hours = 7)
  (h2 : tuesday_hours = 5)
  (h3 : wednesday_hours = 7)
  (h4 : monday_tips = 18)
  (h5 : tuesday_tips = 12)
  (h6 : wednesday_tips = 20)
  (h7 : total_earnings = 240) :
  let total_hours := monday_hours + tuesday_hours + wednesday_hours
  let total_tips := monday_tips + tuesday_tips + wednesday_tips
  let hourly_wage := (total_earnings - total_tips) / total_hours
  hourly_wage = 10 := by
sorry

end NUMINAMATH_CALUDE_waitress_hourly_wage_l137_13755


namespace NUMINAMATH_CALUDE_matrix_A_nonsingular_l137_13794

/-- Prove that the matrix A defined by the given conditions is nonsingular -/
theorem matrix_A_nonsingular 
  (k : ℕ) 
  (i j : Fin k → ℕ)
  (h_i : ∀ m n, m < n → i m < i n)
  (h_j : ∀ m n, m < n → j m < j n)
  (A : Matrix (Fin k) (Fin k) ℚ)
  (h_A : ∀ r s, A r s = (Nat.choose (i r + j s) (i r) : ℚ)) :
  Matrix.det A ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_A_nonsingular_l137_13794


namespace NUMINAMATH_CALUDE_only_23_is_prime_l137_13725

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 0 → d ∣ n → d = 1 ∨ d = n

theorem only_23_is_prime :
  isPrime 23 ∧
  ¬isPrime 20 ∧
  ¬isPrime 21 ∧
  ¬isPrime 25 ∧
  ¬isPrime 27 :=
by
  sorry

end NUMINAMATH_CALUDE_only_23_is_prime_l137_13725


namespace NUMINAMATH_CALUDE_sum_squares_regression_example_l137_13766

/-- Given a total sum of squared deviations and a correlation coefficient,
    calculate the sum of squares due to regression -/
def sum_squares_regression (total_sum_squared_dev : ℝ) (correlation_coeff : ℝ) : ℝ :=
  total_sum_squared_dev * correlation_coeff^2

/-- Theorem stating that given the specified conditions, 
    the sum of squares due to regression is 72 -/
theorem sum_squares_regression_example :
  sum_squares_regression 120 0.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_regression_example_l137_13766


namespace NUMINAMATH_CALUDE_right_triangular_prism_relation_l137_13715

/-- 
Given a right triangular prism with mutually perpendicular lateral edges of lengths a, b, and c,
and base height h, prove that 1/h^2 = 1/a^2 + 1/b^2 + 1/c^2.
-/
theorem right_triangular_prism_relation (a b c h : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hh : h > 0) :
  1 / h^2 = 1 / a^2 + 1 / b^2 + 1 / c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_prism_relation_l137_13715


namespace NUMINAMATH_CALUDE_cleanup_solution_l137_13789

/-- The time spent cleaning up eggs and toilet paper -/
def cleanup_problem (time_per_roll : ℕ) (total_time : ℕ) (num_eggs : ℕ) (num_rolls : ℕ) : Prop :=
  ∃ (time_per_egg : ℕ),
    time_per_egg * num_eggs + time_per_roll * num_rolls * 60 = total_time * 60 ∧
    time_per_egg = 15

/-- Theorem stating the solution to the cleanup problem -/
theorem cleanup_solution :
  cleanup_problem 30 225 60 7 := by
  sorry

end NUMINAMATH_CALUDE_cleanup_solution_l137_13789


namespace NUMINAMATH_CALUDE_debt_payment_average_l137_13750

theorem debt_payment_average (total_payments : ℕ) (first_payment_amount : ℕ) 
  (first_payment_count : ℕ) (payment_increase : ℕ) :
  total_payments = 52 →
  first_payment_count = 12 →
  first_payment_amount = 410 →
  payment_increase = 65 →
  (first_payment_count * first_payment_amount + 
   (total_payments - first_payment_count) * (first_payment_amount + payment_increase)) / 
   total_payments = 460 :=
by sorry

end NUMINAMATH_CALUDE_debt_payment_average_l137_13750


namespace NUMINAMATH_CALUDE_correct_evaluation_l137_13747

-- Define the expression
def expression : ℤ → ℤ → ℤ → ℤ := λ a b c => a - b * c

-- Define the order of operations
def evaluate_expression (a b c : ℤ) : ℤ :=
  a - (b * c)

-- Theorem statement
theorem correct_evaluation :
  evaluate_expression 65 13 2 = 39 :=
by
  sorry

#eval evaluate_expression 65 13 2

end NUMINAMATH_CALUDE_correct_evaluation_l137_13747


namespace NUMINAMATH_CALUDE_number_puzzle_l137_13736

theorem number_puzzle (x : ℝ) : (72 / 6 + x = 17) ↔ (x = 5) := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l137_13736


namespace NUMINAMATH_CALUDE_box_volume_theorem_l137_13720

theorem box_volume_theorem : ∃ (x y z : ℕ+), 
  (x : ℚ) / 2 = (y : ℚ) / 5 ∧ (y : ℚ) / 5 = (z : ℚ) / 7 ∧ 
  (x : ℕ) * y * z = 70 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_theorem_l137_13720


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l137_13737

/-- Calculates the time period for a simple interest problem -/
theorem simple_interest_time_period 
  (P : ℝ) (R : ℝ) (A : ℝ) 
  (h_P : P = 1300)
  (h_R : R = 5)
  (h_A : A = 1456) :
  ∃ T : ℝ, T = 2.4 ∧ A = P + (P * R * T / 100) := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l137_13737


namespace NUMINAMATH_CALUDE_shortest_distance_l137_13776

theorem shortest_distance (a b : ℝ) (ha : a = 8) (hb : b = 6) :
  Real.sqrt (a ^ 2 + b ^ 2) = 10 := by
sorry

end NUMINAMATH_CALUDE_shortest_distance_l137_13776


namespace NUMINAMATH_CALUDE_square_overlap_percentage_l137_13792

/-- The percentage of overlap between two squares forming a rectangle -/
theorem square_overlap_percentage (s1 s2 l w : ℝ) (h1 : s1 = 10) (h2 : s2 = 15) 
  (h3 : l = 25) (h4 : w = 20) : 
  (min s1 s2)^2 / (l * w) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_square_overlap_percentage_l137_13792


namespace NUMINAMATH_CALUDE_base10_to_base7_5423_l137_13774

/-- Converts a base 10 number to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to a natural number --/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 7 * acc) 0

theorem base10_to_base7_5423 :
  toBase7 5423 = [5, 4, 5, 1, 2] ∧ fromBase7 [5, 4, 5, 1, 2] = 5423 := by sorry

end NUMINAMATH_CALUDE_base10_to_base7_5423_l137_13774


namespace NUMINAMATH_CALUDE_inequality_system_solution_l137_13723

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x > a ∧ x ≥ 3) ↔ x ≥ 3) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l137_13723


namespace NUMINAMATH_CALUDE_sum_of_f_evaluations_l137_13754

/-- The operation f defined for rational numbers -/
def f (a b c : ℚ) : ℚ := a^2 + 2*b*c

/-- Theorem stating the sum of specific f evaluations -/
theorem sum_of_f_evaluations : 
  f 1 23 76 + f 23 76 1 + f 76 1 23 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_evaluations_l137_13754


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_l137_13744

/-- Profit function given price x -/
def profit (x : ℝ) : ℝ :=
  let P := -750 * x + 15000
  x * P - 4 * P - 7000

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 12

/-- The maximum profit achieved at the optimal price -/
def max_profit : ℝ := 41000

/-- Theorem stating that the optimal price maximizes profit -/
theorem optimal_price_maximizes_profit :
  profit optimal_price = max_profit ∧
  ∀ x : ℝ, profit x ≤ max_profit :=
sorry

/-- Theorem stating that the maximum profit is achieved at the optimal price -/
theorem max_profit_at_optimal_price :
  ∀ x : ℝ, x ≠ optimal_price → profit x < max_profit :=
sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_l137_13744


namespace NUMINAMATH_CALUDE_recipe_total_cups_l137_13700

/-- Given a recipe with a butter:flour:sugar ratio of 1:6:4, prove that when 8 cups of sugar are used, the total cups of ingredients is 22. -/
theorem recipe_total_cups (butter flour sugar total : ℚ) : 
  butter / sugar = 1 / 4 →
  flour / sugar = 6 / 4 →
  sugar = 8 →
  total = butter + flour + sugar →
  total = 22 := by
sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l137_13700


namespace NUMINAMATH_CALUDE_triangle_angle_and_max_area_l137_13795

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m and n, prove the measure of angle C and the maximum area. -/
theorem triangle_angle_and_max_area 
  (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle angle sum
  m = (Real.sin A, Real.sin B) →           -- Definition of m
  n = (Real.cos B, Real.cos A) →           -- Definition of n
  m.1 * n.1 + m.2 * n.2 = -Real.sin (2 * C) →  -- Dot product condition
  c = 2 * Real.sqrt 3 →                    -- Given value of c
  C = 2 * π / 3 ∧                          -- Angle C
  (∃ (S : ℝ), S ≤ Real.sqrt 3 ∧            -- Maximum area
    ∀ (S' : ℝ), S' = 1/2 * a * b * Real.sin C → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_and_max_area_l137_13795


namespace NUMINAMATH_CALUDE_chess_matches_l137_13779

theorem chess_matches (n : ℕ) (m : ℕ) (h1 : n = 5) (h2 : m = 3) :
  (n * (n - 1) * m) / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_chess_matches_l137_13779


namespace NUMINAMATH_CALUDE_water_added_proof_l137_13767

/-- The amount of water added to a pool given initial and final amounts -/
def water_added (initial : Real) (final : Real) : Real :=
  final - initial

/-- Theorem: Given an initial amount of 1 bucket and a final amount of 9.8 buckets,
    the amount of water added later is 8.8 buckets -/
theorem water_added_proof :
  water_added 1 9.8 = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_water_added_proof_l137_13767


namespace NUMINAMATH_CALUDE_max_d_value_l137_13778

def a (n : ℕ) : ℕ := 150 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (k : ℕ), d k = 601 ∧ ∀ (n : ℕ), d n ≤ 601 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l137_13778


namespace NUMINAMATH_CALUDE_part_one_part_two_l137_13772

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| - |x + m|

-- Part 1
theorem part_one :
  ∀ x : ℝ, (f x 2 + 2 < 0) ↔ (x > 1/2) :=
sorry

-- Part 2
theorem part_two :
  (∀ x ∈ Set.Icc 0 2, f x m + |x - 4| > 0) ↔ m ∈ Set.Ioo (-4) 1 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l137_13772


namespace NUMINAMATH_CALUDE_natural_number_pairs_satisfying_equation_and_condition_l137_13769

theorem natural_number_pairs_satisfying_equation_and_condition :
  ∀ x y : ℕ,
    (2^(10*x + 24*y - 493) + 1 = 9 * 2^(5*x + 12*y - 248) ∧ x + y > 40) ↔
    ((x = 4 ∧ y = 36) ∨ (x = 49 ∧ y = 0) ∨ (x = 37 ∧ y = 7)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_satisfying_equation_and_condition_l137_13769


namespace NUMINAMATH_CALUDE_brian_running_time_l137_13768

theorem brian_running_time (todd_time brian_time : ℕ) : 
  todd_time = 88 → 
  brian_time = todd_time + 8 → 
  brian_time = 96 := by
sorry

end NUMINAMATH_CALUDE_brian_running_time_l137_13768


namespace NUMINAMATH_CALUDE_javier_exercise_time_is_350_l137_13798

/-- Calculates Javier's total exercise time given the conditions of the problem. -/
def javierExerciseTime (javier_daily_time : ℕ) (sanda_daily_time : ℕ) (sanda_days : ℕ) (total_time : ℕ) : ℕ :=
  let javier_days := (total_time - sanda_daily_time * sanda_days) / javier_daily_time
  javier_days * javier_daily_time

/-- Proves that Javier's total exercise time is 350 minutes given the problem conditions. -/
theorem javier_exercise_time_is_350 :
  javierExerciseTime 50 90 3 620 = 350 := by
  sorry

end NUMINAMATH_CALUDE_javier_exercise_time_is_350_l137_13798


namespace NUMINAMATH_CALUDE_one_root_condition_l137_13787

theorem one_root_condition (k : ℝ) : 
  (∃! x : ℝ, Real.log (k * x) = 2 * Real.log (x + 1)) → (k < 0 ∨ k = 4) :=
sorry

end NUMINAMATH_CALUDE_one_root_condition_l137_13787


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l137_13761

theorem factorization_of_quadratic (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l137_13761


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l137_13757

def infinitely_many_n (k : ℤ) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ ¬((n : ℤ) + k ∣ Nat.choose (2*n) n)

theorem binomial_coefficient_divisibility :
  infinitely_many_n (-1) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l137_13757


namespace NUMINAMATH_CALUDE_subway_ways_l137_13716

theorem subway_ways (total : ℕ) (bus : ℕ) (subway : ℕ) : 
  total = 7 → bus = 4 → total = bus + subway → subway = 3 := by
  sorry

end NUMINAMATH_CALUDE_subway_ways_l137_13716


namespace NUMINAMATH_CALUDE_equation_solution_l137_13793

theorem equation_solution : ∃ x : ℝ, (3*x + 4*x = 600 - (2*x + 6*x + x)) ∧ x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l137_13793


namespace NUMINAMATH_CALUDE_proposition_false_implies_a_equals_one_l137_13718

theorem proposition_false_implies_a_equals_one (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (1 - a) * x < 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_proposition_false_implies_a_equals_one_l137_13718


namespace NUMINAMATH_CALUDE_green_height_l137_13732

/-- The heights of the dwarves -/
structure DwarfHeights where
  blue : ℝ
  black : ℝ
  yellow : ℝ
  red : ℝ
  green : ℝ

/-- The conditions of the problem -/
def dwarfProblem (h : DwarfHeights) : Prop :=
  h.blue = 88 ∧
  h.black = 84 ∧
  h.yellow = 76 ∧
  (h.blue + h.black + h.yellow + h.red + h.green) / 5 = 81.6 ∧
  ((h.blue + h.black + h.yellow + h.green) / 4) = ((h.blue + h.black + h.yellow + h.red) / 4 - 6)

theorem green_height (h : DwarfHeights) (hc : dwarfProblem h) : h.green = 68 := by
  sorry

end NUMINAMATH_CALUDE_green_height_l137_13732


namespace NUMINAMATH_CALUDE_calculate_expression_l137_13753

theorem calculate_expression : (3 - Real.pi) ^ 0 + (1/2) ^ (-1 : ℤ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l137_13753


namespace NUMINAMATH_CALUDE_tons_to_pounds_l137_13765

-- Define the basic units
def ounces_per_pound : ℕ := 16

-- Define the packet weight in ounces
def packet_weight_ounces : ℕ := 16 * ounces_per_pound + 4

-- Define the number of packets
def num_packets : ℕ := 1840

-- Define the capacity of the gunny bag in tons
def bag_capacity_tons : ℕ := 13

-- Define the weight of all packets in ounces
def total_weight_ounces : ℕ := num_packets * packet_weight_ounces

-- Define the relation between tons and pounds
def pounds_per_ton : ℕ := 2000

-- Theorem statement
theorem tons_to_pounds : 
  total_weight_ounces = bag_capacity_tons * pounds_per_ton * ounces_per_pound :=
sorry

end NUMINAMATH_CALUDE_tons_to_pounds_l137_13765


namespace NUMINAMATH_CALUDE_prob_four_ones_l137_13796

/-- The number of sides on a standard die -/
def die_sides : ℕ := 6

/-- The probability of rolling a specific number on a standard die -/
def prob_single_roll : ℚ := 1 / die_sides

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- Theorem: The probability of rolling four 1s on four standard dice is 1/1296 -/
theorem prob_four_ones (die_sides : ℕ) (prob_single_roll : ℚ) (num_dice : ℕ) :
  die_sides = 6 →
  prob_single_roll = 1 / die_sides →
  num_dice = 4 →
  prob_single_roll ^ num_dice = 1 / 1296 := by
  sorry

#check prob_four_ones

end NUMINAMATH_CALUDE_prob_four_ones_l137_13796


namespace NUMINAMATH_CALUDE_total_people_count_l137_13714

theorem total_people_count (num_students : ℕ) (ratio : ℕ) : 
  num_students = 37500 →
  ratio = 15 →
  num_students + (num_students / ratio) = 40000 := by
sorry

end NUMINAMATH_CALUDE_total_people_count_l137_13714


namespace NUMINAMATH_CALUDE_last_locker_theorem_l137_13704

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction the student is walking -/
inductive Direction
| Forward
| Backward

/-- 
Simulates the student's locker-opening process and returns the number of the last locker opened.
n: The total number of lockers
-/
def lastLockerOpened (n : Nat) : Nat :=
  sorry

/-- The main theorem stating that for 1024 lockers, the last one opened is number 854 -/
theorem last_locker_theorem : lastLockerOpened 1024 = 854 := by
  sorry

end NUMINAMATH_CALUDE_last_locker_theorem_l137_13704


namespace NUMINAMATH_CALUDE_emily_number_is_3000_l137_13758

def is_valid_number (n : ℕ) : Prop :=
  n % 250 = 0 ∧ n % 60 = 0 ∧ 1000 < n ∧ n < 4000

theorem emily_number_is_3000 : ∃! n : ℕ, is_valid_number n :=
  sorry

end NUMINAMATH_CALUDE_emily_number_is_3000_l137_13758


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l137_13748

theorem sum_of_quadratic_roots (x : ℝ) : 
  x^2 - 17*x + 54 = 0 → ∃ r s : ℝ, r + s = 17 ∧ r * s = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l137_13748


namespace NUMINAMATH_CALUDE_cosine_value_in_triangle_l137_13731

theorem cosine_value_in_triangle (a b c : ℝ) (h : 3 * a^2 + 3 * b^2 - 3 * c^2 = 2 * a * b) :
  let cosC := (a^2 + b^2 - c^2) / (2 * a * b)
  cosC = 1/3 := by sorry

end NUMINAMATH_CALUDE_cosine_value_in_triangle_l137_13731


namespace NUMINAMATH_CALUDE_inequality_theorem_l137_13713

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l137_13713


namespace NUMINAMATH_CALUDE_plane_equation_proof_l137_13764

/-- A plane in 3D space represented by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if two planes are parallel -/
def Plane.isParallelTo (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

theorem plane_equation_proof (given_plane : Plane) (point : Point3D) :
  given_plane.A = -2 ∧ given_plane.B = 1 ∧ given_plane.C = -3 ∧ given_plane.D = 7 →
  point.x = 1 ∧ point.y = 4 ∧ point.z = -2 →
  ∃ (result_plane : Plane),
    result_plane.A = 2 ∧ 
    result_plane.B = -1 ∧ 
    result_plane.C = 3 ∧ 
    result_plane.D = 8 ∧
    point.liesOn result_plane ∧
    result_plane.isParallelTo given_plane :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l137_13764


namespace NUMINAMATH_CALUDE_largest_invertible_interval_for_f_l137_13701

/-- The quadratic function f(x) = 3x^2 - 6x - 9 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 9

/-- The theorem stating that [1, ∞) is the largest interval containing x=2 where f is invertible -/
theorem largest_invertible_interval_for_f :
  ∃ (a : ℝ), a = 1 ∧ 
  (∀ x ∈ Set.Ici a, Function.Injective (f ∘ (λ t => t + a))) ∧
  (∀ b < a, ¬ Function.Injective (f ∘ (λ t => t + b))) ∧
  (2 ∈ Set.Ici a) :=
sorry

end NUMINAMATH_CALUDE_largest_invertible_interval_for_f_l137_13701


namespace NUMINAMATH_CALUDE_smallest_number_l137_13721

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def is_smallest (n : Nat) (lst : List Nat) : Prop :=
  ∀ m ∈ lst, n ≤ m

theorem smallest_number :
  let n1 := base_to_decimal [8, 5] 9
  let n2 := base_to_decimal [2, 1, 0] 6
  let n3 := base_to_decimal [1, 0, 0, 0] 4
  let n4 := base_to_decimal [1, 1, 1, 1, 1, 1] 2
  is_smallest n4 [n1, n2, n3, n4] := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l137_13721


namespace NUMINAMATH_CALUDE_third_to_first_l137_13782

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Definition of a point being in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Theorem: If P is in the third quadrant, then Q(-a, -b) is in the first quadrant -/
theorem third_to_first (P : Point) (hP : isInThirdQuadrant P) :
  let Q : Point := ⟨-P.x, -P.y⟩
  isInFirstQuadrant Q := by
  sorry

end NUMINAMATH_CALUDE_third_to_first_l137_13782
