import Mathlib

namespace inverse_statement_is_false_l1981_198193

-- Define the set S of all non-zero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 3 * a * b

-- Define what it means for an element to be an inverse under *
def is_inverse (a b : ℝ) : Prop := star a b = 1/3 ∧ star b a = 1/3

-- The theorem to be proved
theorem inverse_statement_is_false :
  ∀ a ∈ S, ¬(is_inverse a (1/(3*a))) := by
  sorry

end inverse_statement_is_false_l1981_198193


namespace tan_600_degrees_equals_sqrt_3_l1981_198135

theorem tan_600_degrees_equals_sqrt_3 : Real.tan (600 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_600_degrees_equals_sqrt_3_l1981_198135


namespace expression_simplification_l1981_198144

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x / (x - 1) - 1) / ((x^2 + 2*x + 1) / (x^2 - 1)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l1981_198144


namespace tangent_line_equation_l1981_198180

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : (x^2 / E.a^2) + (y^2 / E.b^2) = 1

/-- The equation of a line -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The equation of the tangent line to an ellipse at a point on the ellipse -/
theorem tangent_line_equation (E : Ellipse) (P : PointOnEllipse E) :
  ∃ (L : Line), L.a = P.x / E.a^2 ∧ L.b = P.y / E.b^2 ∧ L.c = -1 ∧
  (∀ (x y : ℝ), (x^2 / E.a^2) + (y^2 / E.b^2) ≤ 1 → L.a * x + L.b * y + L.c ≤ 0) :=
sorry

end tangent_line_equation_l1981_198180


namespace circle_area_equals_circumference_l1981_198194

theorem circle_area_equals_circumference (r : ℝ) (h : r > 0) :
  π * r^2 = 2 * π * r → r = 2 := by
  sorry

end circle_area_equals_circumference_l1981_198194


namespace asha_remaining_money_l1981_198127

/-- Calculates the remaining money for Asha after spending 3/4 of her total money --/
def remaining_money (brother_loan sister_loan father_loan mother_loan granny_gift savings : ℚ) : ℚ :=
  let total := brother_loan + sister_loan + father_loan + mother_loan + granny_gift + savings
  total - (3/4 * total)

/-- Theorem stating that Asha remains with $65 after spending --/
theorem asha_remaining_money :
  remaining_money 20 0 40 30 70 100 = 65 := by
  sorry

end asha_remaining_money_l1981_198127


namespace system_solution_l1981_198153

/-- Given a system of equations and the condition that a ≠ bc, 
    prove that x = 1, y = 0, and z = 0 are the solutions. -/
theorem system_solution (a b c : ℝ) (h : a ≠ b * c) :
  ∃! (x y z : ℝ), 
    a = (a * x + c * y) / (b * z + 1) ∧
    b = (b * x + y) / (b * z + 1) ∧
    c = (a * z + c) / (b * z + 1) ∧
    x = 1 ∧ y = 0 ∧ z = 0 := by
  sorry

end system_solution_l1981_198153


namespace petrol_expense_percentage_l1981_198169

/-- Represents the problem of calculating the percentage of income spent on petrol --/
theorem petrol_expense_percentage
  (total_income : ℝ)
  (petrol_expense : ℝ)
  (rent_expense : ℝ)
  (rent_percentage : ℝ)
  (h1 : petrol_expense = 300)
  (h2 : rent_expense = 210)
  (h3 : rent_percentage = 30)
  (h4 : rent_expense = (rent_percentage / 100) * (total_income - petrol_expense)) :
  (petrol_expense / total_income) * 100 = 30 := by
  sorry

end petrol_expense_percentage_l1981_198169


namespace cos_negative_ninety_degrees_l1981_198126

theorem cos_negative_ninety_degrees : Real.cos (-(π / 2)) = 0 := by sorry

end cos_negative_ninety_degrees_l1981_198126


namespace second_outlet_rate_calculation_l1981_198122

/-- Represents the rate of the second outlet pipe in cubic inches per minute -/
def second_outlet_rate : ℝ := 9

/-- Tank volume in cubic feet -/
def tank_volume : ℝ := 30

/-- Inlet pipe rate in cubic inches per minute -/
def inlet_rate : ℝ := 3

/-- First outlet pipe rate in cubic inches per minute -/
def first_outlet_rate : ℝ := 6

/-- Time to empty the tank when all pipes are open, in minutes -/
def emptying_time : ℝ := 4320

/-- Conversion factor from cubic feet to cubic inches -/
def cubic_feet_to_inches : ℝ := 12 ^ 3

theorem second_outlet_rate_calculation :
  second_outlet_rate = 
    (tank_volume * cubic_feet_to_inches - emptying_time * (inlet_rate - first_outlet_rate)) / 
    emptying_time := by
  sorry

end second_outlet_rate_calculation_l1981_198122


namespace magician_hourly_rate_l1981_198152

/-- Proves that the hourly rate for a magician who works 3 hours per day for 2 weeks
    and receives a total payment of $2520 is $60 per hour. -/
theorem magician_hourly_rate :
  let hours_per_day : ℕ := 3
  let days : ℕ := 14
  let total_payment : ℕ := 2520
  let total_hours : ℕ := hours_per_day * days
  let hourly_rate : ℚ := total_payment / total_hours
  hourly_rate = 60 := by
  sorry

#check magician_hourly_rate

end magician_hourly_rate_l1981_198152


namespace ellipse_equation_and_max_area_l1981_198155

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of an ellipse -/
structure Foci where
  left : Point
  right : Point

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

theorem ellipse_equation_and_max_area 
  (C : Ellipse) 
  (P : Point)
  (F : Foci)
  (h_P_on_C : P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1)
  (h_P_coords : P.x = 1 ∧ P.y = Real.sqrt 2 / 2)
  (h_PF_sum : distance P F.left + distance P F.right = 2 * Real.sqrt 2) :
  (∃ (a b : ℝ), C.a = a ∧ C.b = b ∧ a^2 = 2 ∧ b^2 = 1) ∧
  (∃ (max_area : ℝ), 
    (∀ (Q : Point) (h_Q_on_C : Q.x^2 / C.a^2 + Q.y^2 / C.b^2 = 1),
      abs (P.x * Q.y - P.y * Q.x) / 2 ≤ max_area) ∧
    max_area = Real.sqrt 2 / 2) :=
by sorry

end ellipse_equation_and_max_area_l1981_198155


namespace first_term_exceeding_thousand_l1981_198185

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Predicate to check if a term exceeds 1000 -/
def exceedsThousand (x : ℝ) : Prop :=
  x > 1000

theorem first_term_exceeding_thousand :
  let a₁ := 2
  let d := 3
  (∀ n < 334, ¬(exceedsThousand (arithmeticSequenceTerm a₁ d n))) ∧
  exceedsThousand (arithmeticSequenceTerm a₁ d 334) :=
by sorry

end first_term_exceeding_thousand_l1981_198185


namespace no_valid_base_l1981_198196

theorem no_valid_base : ¬ ∃ (b : ℕ), 0 < b ∧ b^6 ≤ 196 ∧ 196 < b^7 := by
  sorry

end no_valid_base_l1981_198196


namespace mri_to_xray_ratio_l1981_198130

/-- The cost of an x-ray in dollars -/
def x_ray_cost : ℝ := 250

/-- The cost of an MRI as a multiple of the x-ray cost -/
def mri_cost (k : ℝ) : ℝ := k * x_ray_cost

/-- The insurance coverage percentage -/
def insurance_coverage : ℝ := 0.8

/-- The amount Mike paid in dollars -/
def mike_payment : ℝ := 200

/-- The theorem stating the ratio of MRI cost to x-ray cost -/
theorem mri_to_xray_ratio :
  ∃ k : ℝ,
    (1 - insurance_coverage) * (x_ray_cost + mri_cost k) = mike_payment ∧
    k = 3 :=
sorry

end mri_to_xray_ratio_l1981_198130


namespace faster_pipe_rate_l1981_198167

/-- Given two pipes with different filling rates, prove that the faster pipe is 4 times faster than the slower pipe. -/
theorem faster_pipe_rate (slow_rate fast_rate : ℝ) : 
  slow_rate > 0 →
  fast_rate > slow_rate →
  (1 : ℝ) / slow_rate = 180 →
  1 / (slow_rate + fast_rate) = 36 →
  fast_rate = 4 * slow_rate :=
by sorry

end faster_pipe_rate_l1981_198167


namespace line_parameterization_l1981_198123

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 30

/-- The parameterization of the line -/
def parameterization (g : ℝ → ℝ) (t : ℝ) : ℝ × ℝ := (g t, 18 * t - 10)

/-- The theorem stating that g(t) = 9t + 10 satisfies the line equation and parameterization -/
theorem line_parameterization (g : ℝ → ℝ) :
  (∀ t, line_equation (g t) (18 * t - 10)) ↔ (∀ t, g t = 9 * t + 10) :=
sorry

end line_parameterization_l1981_198123


namespace factorization_problem1_factorization_problem2_l1981_198147

-- Problem 1
theorem factorization_problem1 (m : ℝ) : 
  m * (m - 5) - 2 * (5 - m)^2 = -(m - 5) * (m - 10) := by sorry

-- Problem 2
theorem factorization_problem2 (x : ℝ) : 
  -4 * x^3 + 8 * x^2 - 4 * x = -4 * x * (x - 1)^2 := by sorry

end factorization_problem1_factorization_problem2_l1981_198147


namespace laptop_cost_proof_l1981_198132

theorem laptop_cost_proof (x y : ℝ) (h1 : y = 3 * x) (h2 : x + y = 2000) : x = 500 := by
  sorry

end laptop_cost_proof_l1981_198132


namespace max_discriminant_quadratic_l1981_198191

theorem max_discriminant_quadratic (a b c u v w : ℤ) :
  u ≠ v ∧ u ≠ w ∧ v ≠ w →
  a * u^2 + b * u + c = 0 →
  a * v^2 + b * v + c = 0 →
  a * w^2 + b * w + c = 2 →
  ∃ (max : ℤ), max = 16 ∧ b^2 - 4*a*c ≤ max :=
by sorry

end max_discriminant_quadratic_l1981_198191


namespace arithmetic_sequence_common_difference_l1981_198160

theorem arithmetic_sequence_common_difference (d : ℕ+) : 
  (∃ n : ℕ, 1 + (n - 1) * d.val = 81) → d ≠ 3 := by sorry

end arithmetic_sequence_common_difference_l1981_198160


namespace race_start_distance_l1981_198166

theorem race_start_distance (speed_a speed_b : ℝ) (total_distance : ℝ) (start_distance : ℝ) : 
  speed_a = (5 / 3) * speed_b →
  total_distance = 200 →
  total_distance / speed_a = (total_distance - start_distance) / speed_b →
  start_distance = 80 := by
sorry

end race_start_distance_l1981_198166


namespace trapezoid_median_equals_12_l1981_198172

/-- Given a triangle and a trapezoid with equal areas and altitudes, where the triangle's base is 24 inches and one base of the trapezoid is twice the other, prove the trapezoid's median is 12 inches. -/
theorem trapezoid_median_equals_12 (h : ℝ) (x : ℝ) : 
  h > 0 →  -- Altitude is positive
  (1/2) * 24 * h = ((x + 2*x) / 2) * h →  -- Equal areas
  (x + 2*x) / 2 = 12 :=
by sorry

end trapezoid_median_equals_12_l1981_198172


namespace power_congruence_l1981_198140

theorem power_congruence (h : 5^500 ≡ 1 [ZMOD 2000]) : 5^15000 ≡ 1 [ZMOD 2000] := by
  sorry

end power_congruence_l1981_198140


namespace smallest_sector_angle_l1981_198189

/-- Represents the properties of the circle division problem -/
structure CircleDivision where
  n : ℕ  -- number of sectors
  a₁ : ℕ  -- first term of the arithmetic sequence
  d : ℕ   -- common difference of the arithmetic sequence
  sum : ℕ -- sum of all angles

/-- The circle division satisfies the problem conditions -/
def validCircleDivision (cd : CircleDivision) : Prop :=
  cd.n = 15 ∧
  cd.sum = 360 ∧
  ∀ i : ℕ, i > 0 ∧ i ≤ cd.n → (cd.a₁ + (i - 1) * cd.d) > 0

/-- The theorem stating the smallest possible sector angle -/
theorem smallest_sector_angle (cd : CircleDivision) :
  validCircleDivision cd →
  (∃ cd' : CircleDivision, validCircleDivision cd' ∧ cd'.a₁ < cd.a₁) ∨ cd.a₁ = 10 :=
sorry

end smallest_sector_angle_l1981_198189


namespace magic_square_difference_l1981_198117

/-- Represents a 3x3 magic square with some given values -/
structure MagicSquare where
  x : ℝ
  y : ℝ
  isValid : x - 2 = 2*y + y ∧ x - 2 = -2 + y + 6

/-- Proves that in the given magic square, y - x = -6 -/
theorem magic_square_difference (ms : MagicSquare) : ms.y - ms.x = -6 := by
  sorry

end magic_square_difference_l1981_198117


namespace triangle_angle_measure_l1981_198108

theorem triangle_angle_measure
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle opposite to A, B, C respectively
  (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = (1/2) * b)
  (h2 : a > b)
  (h3 : 0 < A ∧ A < π) -- Ensuring A is a valid angle measure
  (h4 : 0 < B ∧ B < π) -- Ensuring B is a valid angle measure
  (h5 : 0 < C ∧ C < π) -- Ensuring C is a valid angle measure
  (h6 : A + B + C = π) -- Sum of angles in a triangle
  : B = π/6 := by
  sorry

#check triangle_angle_measure

end triangle_angle_measure_l1981_198108


namespace inequality_holds_for_p_greater_than_two_l1981_198199

theorem inequality_holds_for_p_greater_than_two (p q : ℝ) 
  (hp : p > 2) (hq : q > 0) : 
  4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 4 * p * q) / (p + q) > 3 * p^3 * q := by
  sorry

end inequality_holds_for_p_greater_than_two_l1981_198199


namespace kim_water_consumption_l1981_198139

/-- Proves that the total amount of water Kim drinks is 60 ounces -/
theorem kim_water_consumption (quart_to_ounce : ℚ) (bottle_volume : ℚ) (can_volume : ℚ) :
  quart_to_ounce = 32 →
  bottle_volume = 3/2 →
  can_volume = 12 →
  bottle_volume * quart_to_ounce + can_volume = 60 := by
  sorry

end kim_water_consumption_l1981_198139


namespace factorial_division_l1981_198102

theorem factorial_division : Nat.factorial 6 / Nat.factorial (6 - 3) = 120 := by
  sorry

end factorial_division_l1981_198102


namespace system_solutions_l1981_198137

-- Define the system of equations
def system (t x y z : ℝ) : Prop :=
  t * (x + y + z) = 0 ∧ t * (x + y) + z = 1 ∧ t * x + y + z = 2

-- State the theorem
theorem system_solutions :
  ∀ t x y z : ℝ,
    (t = 0 → system t x y z ↔ y = 1 ∧ z = 1) ∧
    (t ≠ 0 ∧ t ≠ 1 → system t x y z ↔ x = 2 / (t - 1) ∧ y = -1 / (t - 1) ∧ z = -1 / (t - 1)) ∧
    (t = 1 → ¬∃ x y z, system t x y z) :=
by sorry

end system_solutions_l1981_198137


namespace quadrilateral_area_l1981_198104

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def inscribed_in_parabola (q : Quadrilateral) : Prop :=
  parabola q.A.1 = q.A.2 ∧ parabola q.B.1 = q.B.2 ∧
  parabola q.C.1 = q.C.2 ∧ parabola q.D.1 = q.D.2

def angle_BAD_is_right (q : Quadrilateral) : Prop :=
  (q.B.1 - q.A.1) * (q.D.1 - q.A.1) + (q.B.2 - q.A.2) * (q.D.2 - q.A.2) = 0

def AC_parallel_to_x_axis (q : Quadrilateral) : Prop :=
  q.A.2 = q.C.2

def AC_bisects_BAD (q : Quadrilateral) : Prop :=
  (q.C.1 - q.A.1)^2 + (q.C.2 - q.A.2)^2 =
  (q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2

def diagonal_BD_length (q : Quadrilateral) (p : ℝ) : Prop :=
  (q.B.1 - q.D.1)^2 + (q.B.2 - q.D.2)^2 = p^2

-- The theorem
theorem quadrilateral_area (q : Quadrilateral) (p : ℝ) :
  inscribed_in_parabola q →
  angle_BAD_is_right q →
  AC_parallel_to_x_axis q →
  AC_bisects_BAD q →
  diagonal_BD_length q p →
  (q.A.1 - q.C.1) * (q.B.2 - q.D.2) / 2 = (p^2 - 4) / 4 := by
  sorry

end quadrilateral_area_l1981_198104


namespace stating_downstream_speed_l1981_198103

/-- Represents the rowing speeds of a man in different conditions. -/
structure RowingSpeeds where
  upstream : ℝ
  still_water : ℝ
  downstream : ℝ

/-- 
Theorem stating that given a man's upstream rowing speed and still water speed,
we can determine his downstream speed.
-/
theorem downstream_speed (speeds : RowingSpeeds) 
  (h_upstream : speeds.upstream = 7)
  (h_still_water : speeds.still_water = 20)
  (h_average : speeds.still_water = (speeds.upstream + speeds.downstream) / 2) :
  speeds.downstream = 33 := by
  sorry

#check downstream_speed

end stating_downstream_speed_l1981_198103


namespace trigonometric_equation_solution_l1981_198162

open Real

theorem trigonometric_equation_solution (z : ℝ) :
  cos z ≠ 0 →
  sin z ≠ 0 →
  (5.38 * (1 / (cos z)^4) = 160/9 - (2 * ((cos (2*z) / sin (2*z)) * (cos z / sin z) + 1)) / (sin z)^2) →
  ∃ k : ℤ, z = (π/6) * (3 * k + 1) ∨ z = (π/6) * (3 * k - 1) :=
sorry

end trigonometric_equation_solution_l1981_198162


namespace class_average_problem_l1981_198151

theorem class_average_problem (N : ℝ) (h : N > 0) :
  let total_average : ℝ := 80
  let three_fourths_average : ℝ := 76
  let one_fourth_average : ℝ := (4 * total_average * N - 3 * three_fourths_average * N) / N
  one_fourth_average = 92 := by sorry

end class_average_problem_l1981_198151


namespace painters_work_days_l1981_198113

/-- Given that 5 painters take 1.8 work-days to finish a job, prove that 4 painters
    working at the same rate will take 2.25 work-days to finish the same job. -/
theorem painters_work_days (initial_painters : ℕ) (initial_days : ℝ) 
  (new_painters : ℕ) (new_days : ℝ) :
  initial_painters = 5 →
  initial_days = 1.8 →
  new_painters = 4 →
  (initial_painters : ℝ) * initial_days = (new_painters : ℝ) * new_days →
  new_days = 2.25 := by
sorry

end painters_work_days_l1981_198113


namespace even_sine_function_l1981_198157

theorem even_sine_function (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  (∀ x, f (-x) = f x) →
  φ = π / 2 := by
  sorry

end even_sine_function_l1981_198157


namespace tan_eleven_pi_sixths_l1981_198116

theorem tan_eleven_pi_sixths : Real.tan (11 * Real.pi / 6) = -1 / Real.sqrt 3 := by
  sorry

end tan_eleven_pi_sixths_l1981_198116


namespace william_car_wash_time_l1981_198118

/-- Represents the time in minutes for each car washing task -/
structure CarWashTime where
  windows : ℕ
  body : ℕ
  tires : ℕ
  waxing : ℕ

/-- Calculates the total time for washing a normal car -/
def normalCarTime (t : CarWashTime) : ℕ :=
  t.windows + t.body + t.tires + t.waxing

/-- Theorem: William's total car washing time is 96 minutes -/
theorem william_car_wash_time :
  ∀ (t : CarWashTime),
  t.windows = 4 →
  t.body = 7 →
  t.tires = 4 →
  t.waxing = 9 →
  2 * normalCarTime t + 2 * normalCarTime t = 96 := by
  sorry

#check william_car_wash_time

end william_car_wash_time_l1981_198118


namespace M_subset_N_l1981_198149

def M : Set ℚ := {x | ∃ k : ℤ, x = k / 3 + 1 / 6}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 6 + 1 / 3}

theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l1981_198149


namespace peanut_butter_cans_l1981_198128

theorem peanut_butter_cans (n : ℕ) (initial_avg_price remaining_avg_price returned_avg_price : ℚ)
  (h1 : initial_avg_price = 365/10)
  (h2 : remaining_avg_price = 30/1)
  (h3 : returned_avg_price = 495/10)
  (h4 : n * initial_avg_price = (n - 2) * remaining_avg_price + 2 * returned_avg_price) :
  n = 6 := by
  sorry

end peanut_butter_cans_l1981_198128


namespace percent_relation_l1981_198173

theorem percent_relation (a b : ℝ) (h : a = 1.8 * b) : 
  4 * b / a = 20 / 9 := by sorry

end percent_relation_l1981_198173


namespace special_linear_function_unique_l1981_198115

/-- A linear function f such that f(f(x)) = x + 2 -/
def special_linear_function (f : ℝ → ℝ) : Prop :=
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧ 
  (∀ x, f (f x) = x + 2)

/-- The unique linear function satisfying f(f(x)) = x + 2 is f(x) = x + 1 -/
theorem special_linear_function_unique (f : ℝ → ℝ) :
  special_linear_function f → (∀ x, f x = x + 1) :=
by sorry

end special_linear_function_unique_l1981_198115


namespace diophantine_equation_solution_l1981_198119

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(23,24,30), (12,30,31), (9,30,32), (4,30,33), (15,22,36), (9,18,40), (4,15,42)}

theorem diophantine_equation_solution :
  {t : ℕ × ℕ × ℕ | is_valid_triple t.1 t.2.1 t.2.2} = solution_set := by
  sorry

end diophantine_equation_solution_l1981_198119


namespace wax_needed_l1981_198145

theorem wax_needed (total_wax : ℕ) (available_wax : ℕ) (h1 : total_wax = 288) (h2 : available_wax = 28) :
  total_wax - available_wax = 260 := by
  sorry

end wax_needed_l1981_198145


namespace max_min_difference_is_five_l1981_198195

/-- Given non-zero real numbers a and b satisfying a² + b² = 25,
    prove that the difference between the maximum and minimum values
    of the function y = (ax + b) / (x² + 1) is 5. -/
theorem max_min_difference_is_five (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : a^2 + b^2 = 25) :
  let f : ℝ → ℝ := λ x => (a * x + b) / (x^2 + 1)
  ∃ y₁ y₂ : ℝ, (∀ x, f x ≤ y₁) ∧ (∀ x, f x ≥ y₂) ∧ y₁ - y₂ = 5 :=
by sorry

end max_min_difference_is_five_l1981_198195


namespace average_marks_combined_classes_l1981_198187

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 30) 
  (h2 : n2 = 50) 
  (h3 : avg1 = 30) 
  (h4 : avg2 = 60) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 48.75 := by
  sorry

end average_marks_combined_classes_l1981_198187


namespace line_relationship_indeterminate_l1981_198186

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- Perpendicularity relation between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- States that two lines have an indeterminate relationship -/
def indeterminate_relationship (l1 l2 : Line3D) : Prop := sorry

theorem line_relationship_indeterminate 
  (l1 l2 l3 l4 : Line3D) 
  (h1 : perpendicular l1 l2)
  (h2 : parallel l2 l3)
  (h3 : perpendicular l3 l4)
  (h4 : l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l2 ≠ l3 ∧ l2 ≠ l4 ∧ l3 ≠ l4) :
  indeterminate_relationship l1 l4 := by
  sorry

end line_relationship_indeterminate_l1981_198186


namespace variations_formula_l1981_198159

/-- The number of r-class variations from n elements where the first s elements occur -/
def variations (n r s : ℕ) : ℕ :=
  (Nat.factorial (n - s) * Nat.factorial r) / (Nat.factorial (r - s) * Nat.factorial (n - r))

/-- Theorem stating the number of r-class variations from n elements where the first s elements occur -/
theorem variations_formula (n r s : ℕ) (h1 : s < r) (h2 : r ≤ n) :
  variations n r s = (Nat.factorial (n - s) * Nat.factorial r) / (Nat.factorial (r - s) * Nat.factorial (n - r)) :=
by sorry

end variations_formula_l1981_198159


namespace square_of_nine_l1981_198179

theorem square_of_nine (x : ℝ) : x^2 = 9 → x = 3 ∨ x = -3 := by
  sorry

end square_of_nine_l1981_198179


namespace fraction_reciprocal_sum_ge_two_l1981_198161

theorem fraction_reciprocal_sum_ge_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / b + b / a ≥ 2 := by
  sorry

end fraction_reciprocal_sum_ge_two_l1981_198161


namespace second_train_speed_l1981_198133

/-- The speed of the first train in km/h -/
def speed_first : ℝ := 40

/-- The time difference between the departure of the two trains in hours -/
def time_difference : ℝ := 1

/-- The distance at which the two trains meet, in km -/
def meeting_distance : ℝ := 200

/-- The speed of the second train in km/h -/
def speed_second : ℝ := 50

theorem second_train_speed :
  speed_second = meeting_distance / (meeting_distance / speed_first - time_difference) :=
by sorry

end second_train_speed_l1981_198133


namespace negation_existential_quadratic_l1981_198158

theorem negation_existential_quadratic (x : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by sorry

end negation_existential_quadratic_l1981_198158


namespace average_age_of_population_l1981_198136

/-- The average age of a population given the ratio of men to women and their respective average ages -/
theorem average_age_of_population 
  (ratio_men_to_women : ℚ) 
  (avg_age_men : ℝ) 
  (avg_age_women : ℝ) :
  ratio_men_to_women = 2/3 →
  avg_age_men = 37 →
  avg_age_women = 42 →
  let total_population := ratio_men_to_women + 1
  let weighted_age_men := ratio_men_to_women * avg_age_men
  let weighted_age_women := 1 * avg_age_women
  (weighted_age_men + weighted_age_women) / total_population = 40 :=
by sorry


end average_age_of_population_l1981_198136


namespace x_bound_y_bound_l1981_198141

/-- Represents the position of a particle -/
structure Position where
  x : ℕ
  y : ℕ

/-- Calculates the position of a particle after n minutes -/
def particlePosition (n : ℕ) : Position :=
  sorry

/-- The initial rightward movement is 2 units -/
axiom initial_rightward : (particlePosition 1).x = 2

/-- The y-coordinate doesn't change in the first minute -/
axiom initial_upward : (particlePosition 1).y = 0

/-- The x-coordinate never decreases -/
axiom x_nondecreasing (n : ℕ) : (particlePosition n).x ≤ (particlePosition (n + 1)).x

/-- The y-coordinate never decreases -/
axiom y_nondecreasing (n : ℕ) : (particlePosition n).y ≤ (particlePosition (n + 1)).y

/-- The x-coordinate is bounded by the initial movement plus subsequent rightward movements -/
theorem x_bound (n : ℕ) : 
  (particlePosition n).x ≤ 2 + 2 * (n / 4) * ((n / 4) + 1) :=
  sorry

/-- The y-coordinate is bounded by the sum of upward movements -/
theorem y_bound (n : ℕ) : 
  (particlePosition n).y ≤ (n - 1) * (n / 4) :=
  sorry

end x_bound_y_bound_l1981_198141


namespace bob_remaining_corn_l1981_198146

/-- Represents the amount of corn, either in bushels or individual ears. -/
inductive CornAmount
| bushels (n : ℕ)
| ears (n : ℕ)

/-- Converts CornAmount to total number of ears. -/
def to_ears (amount : CornAmount) (ears_per_bushel : ℕ) : ℕ :=
  match amount with
  | CornAmount.bushels n => n * ears_per_bushel
  | CornAmount.ears n => n

/-- Calculates the remaining ears of corn after giving some away. -/
def remaining_corn (initial : CornAmount) (given_away : List CornAmount) (ears_per_bushel : ℕ) : ℕ :=
  to_ears initial ears_per_bushel - (given_away.map (λ a => to_ears a ears_per_bushel)).sum

theorem bob_remaining_corn :
  let initial := CornAmount.bushels 120
  let given_away := [
    CornAmount.bushels 15,  -- Terry
    CornAmount.bushels 8,   -- Jerry
    CornAmount.bushels 25,  -- Linda
    CornAmount.ears 42,     -- Stacy
    CornAmount.bushels 9,   -- Susan
    CornAmount.bushels 4,   -- Tim (bushels)
    CornAmount.ears 18      -- Tim (ears)
  ]
  let ears_per_bushel := 15
  remaining_corn initial given_away ears_per_bushel = 825 := by
  sorry

#eval remaining_corn (CornAmount.bushels 120) [
  CornAmount.bushels 15,
  CornAmount.bushels 8,
  CornAmount.bushels 25,
  CornAmount.ears 42,
  CornAmount.bushels 9,
  CornAmount.bushels 4,
  CornAmount.ears 18
] 15

end bob_remaining_corn_l1981_198146


namespace symmetric_difference_equality_l1981_198105

open Set

theorem symmetric_difference_equality (A B K : Set α) : 
  symmDiff A K = symmDiff B K → A = B :=
by sorry

end symmetric_difference_equality_l1981_198105


namespace number_of_balls_correct_l1981_198154

/-- The number of balls in a box, which is as much greater than 40 as it is less than 60. -/
def number_of_balls : ℕ := 50

/-- The condition that the number of balls is as much greater than 40 as it is less than 60. -/
def ball_condition (x : ℕ) : Prop := x - 40 = 60 - x

theorem number_of_balls_correct : ball_condition number_of_balls := by
  sorry

end number_of_balls_correct_l1981_198154


namespace boat_current_speed_ratio_l1981_198134

/-- Proves that the ratio of boat speed to current speed is 4:1 given upstream and downstream travel times -/
theorem boat_current_speed_ratio 
  (distance : ℝ) 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (h_upstream : upstream_time = 6) 
  (h_downstream : downstream_time = 10) 
  (h_positive_distance : distance > 0) :
  ∃ (boat_speed current_speed : ℝ),
    boat_speed > 0 ∧ 
    current_speed > 0 ∧
    distance = upstream_time * (boat_speed - current_speed) ∧
    distance = downstream_time * (boat_speed + current_speed) ∧
    boat_speed = 4 * current_speed :=
sorry

end boat_current_speed_ratio_l1981_198134


namespace linear_equation_implies_mn_one_l1981_198111

/-- If (m+2)x^(|m|-1) + y^(2n) = 5 is a linear equation in x and y, where m and n are real numbers, then mn = 1 -/
theorem linear_equation_implies_mn_one (m n : ℝ) : 
  (∃ a b c : ℝ, ∀ x y : ℝ, (m + 2) * x^(|m| - 1) + y^(2*n) = 5 ↔ a*x + b*y = c) → 
  m * n = 1 := by
sorry

end linear_equation_implies_mn_one_l1981_198111


namespace total_money_proof_l1981_198183

def sam_money : ℕ := 75

def billy_money (sam : ℕ) : ℕ := 2 * sam - 25

def total_money (sam : ℕ) : ℕ := sam + billy_money sam

theorem total_money_proof : total_money sam_money = 200 := by
  sorry

end total_money_proof_l1981_198183


namespace unique_fraction_representation_l1981_198131

theorem unique_fraction_representation (p : ℕ) (hp : p > 2) (hprime : Nat.Prime p) :
  ∃! (x y : ℕ), x ≠ y ∧ (2 : ℚ) / p = 1 / x + 1 / y :=
by sorry

end unique_fraction_representation_l1981_198131


namespace civil_service_exam_probability_l1981_198101

theorem civil_service_exam_probability 
  (pass_rate_written : ℝ) 
  (pass_rate_overall : ℝ) 
  (h1 : pass_rate_written = 0.2) 
  (h2 : pass_rate_overall = 0.04) :
  pass_rate_overall / pass_rate_written = 0.2 :=
sorry

end civil_service_exam_probability_l1981_198101


namespace inequality_solution_range_l1981_198142

theorem inequality_solution_range :
  ∀ (a : ℝ), (∃ x ∈ Set.Icc 0 3, x^2 - a*x - a + 1 ≥ 0) ↔ a ≤ 5/2 :=
by sorry

end inequality_solution_range_l1981_198142


namespace executive_committee_selection_l1981_198163

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem executive_committee_selection (total_members senior_members committee_size : ℕ) 
  (h1 : total_members = 30)
  (h2 : senior_members = 10)
  (h3 : committee_size = 5) :
  (choose senior_members 2 * choose (total_members - senior_members) 3 +
   choose senior_members 3 * choose (total_members - senior_members) 2 +
   choose senior_members 4 * choose (total_members - senior_members) 1 +
   choose senior_members 5) = 78552 := by
  sorry

end executive_committee_selection_l1981_198163


namespace division_simplification_l1981_198107

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
by sorry

end division_simplification_l1981_198107


namespace base_8_units_digit_l1981_198164

theorem base_8_units_digit : ∃ n : ℕ, (356 * 78 + 49) % 8 = 1 ∧ (356 * 78 + 49) = 8 * n + 1 := by
  sorry

end base_8_units_digit_l1981_198164


namespace range_of_3a_minus_b_l1981_198190

theorem range_of_3a_minus_b (a b : ℝ) (ha : -5 < a ∧ a < 2) (hb : 1 < b ∧ b < 4) :
  -19 < 3 * a - b ∧ 3 * a - b < 5 := by
  sorry

end range_of_3a_minus_b_l1981_198190


namespace quadratic_with_odd_coeff_no_rational_roots_l1981_198198

theorem quadratic_with_odd_coeff_no_rational_roots (a b c : ℤ) :
  Odd a → Odd b → Odd c → ¬ IsSquare (b^2 - 4*a*c) := by
  sorry

end quadratic_with_odd_coeff_no_rational_roots_l1981_198198


namespace prob_sum_seven_or_eleven_l1981_198121

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of ways to roll a sum of 7 -/
def waysToRollSeven : ℕ := 6

/-- The number of ways to roll a sum of 11 -/
def waysToRollEleven : ℕ := 2

/-- The total number of favorable outcomes (sum of 7 or 11) -/
def favorableOutcomes : ℕ := waysToRollSeven + waysToRollEleven

/-- The probability of rolling a sum of 7 or 11 with two fair six-sided dice -/
theorem prob_sum_seven_or_eleven : 
  (favorableOutcomes : ℚ) / totalOutcomes = 2 / 9 := by
  sorry

end prob_sum_seven_or_eleven_l1981_198121


namespace cylinder_volume_increase_l1981_198170

/-- The increase in radius and height of a cylinder that results in quadrupling its volume -/
theorem cylinder_volume_increase (x : ℝ) : x > 0 →
  π * (10 + x)^2 * (5 + x) = 4 * (π * 10^2 * 5) →
  x = 10 := by
sorry

end cylinder_volume_increase_l1981_198170


namespace rectangular_to_cylindrical_l1981_198156

theorem rectangular_to_cylindrical :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 2
  let r : ℝ := 6
  let θ : ℝ := 5 * Real.pi / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ ∧
  z = z :=
by
  sorry

#check rectangular_to_cylindrical

end rectangular_to_cylindrical_l1981_198156


namespace distance_for_boy_problem_l1981_198197

/-- Calculates the distance traveled given time in minutes and speed in meters per second -/
def distance_traveled (time_minutes : ℕ) (speed_meters_per_second : ℕ) : ℕ :=
  time_minutes * 60 * speed_meters_per_second

/-- Theorem: Given 36 minutes and a speed of 4 meters per second, the distance traveled is 8640 meters -/
theorem distance_for_boy_problem : distance_traveled 36 4 = 8640 := by
  sorry

end distance_for_boy_problem_l1981_198197


namespace triangles_in_hexagon_with_center_l1981_198129

/-- The number of triangles formed by 7 points of a regular hexagon (including center) --/
def num_triangles_hexagon : ℕ :=
  Nat.choose 7 3 - 3

theorem triangles_in_hexagon_with_center :
  num_triangles_hexagon = 32 := by
  sorry

end triangles_in_hexagon_with_center_l1981_198129


namespace intersection_when_a_is_four_subset_iff_a_geq_four_l1981_198150

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem 1: When a = 4, A ∩ B = A
theorem intersection_when_a_is_four :
  A ∩ B 4 = A := by sorry

-- Theorem 2: A ⊆ B if and only if a ≥ 4
theorem subset_iff_a_geq_four (a : ℝ) :
  A ⊆ B a ↔ a ≥ 4 := by sorry

end intersection_when_a_is_four_subset_iff_a_geq_four_l1981_198150


namespace infinite_geometric_series_sum_l1981_198106

theorem infinite_geometric_series_sum : 
  let a : ℝ := 1/4  -- first term
  let r : ℝ := 1/2  -- common ratio
  let S : ℝ := ∑' n, a * r^n  -- infinite sum
  S = 1/2 := by
sorry

end infinite_geometric_series_sum_l1981_198106


namespace table_wobbles_l1981_198124

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a table with four legs -/
structure Table where
  leg1 : Point3D
  leg2 : Point3D
  leg3 : Point3D
  leg4 : Point3D

/-- Checks if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧ 
    a * p1.x + b * p1.y + c * p1.z + d = 0 ∧
    a * p2.x + b * p2.y + c * p2.z + d = 0 ∧
    a * p3.x + b * p3.y + c * p3.z + d = 0 ∧
    a * p4.x + b * p4.y + c * p4.z + d = 0

/-- Defines a square table with given leg lengths -/
def squareTable : Table :=
  { leg1 := ⟨0, 0, 70⟩
  , leg2 := ⟨1, 0, 71⟩
  , leg3 := ⟨1, 1, 72.5⟩
  , leg4 := ⟨0, 1, 72⟩ }

/-- Theorem: The square table with given leg lengths wobbles -/
theorem table_wobbles : ¬areCoplanar squareTable.leg1 squareTable.leg2 squareTable.leg3 squareTable.leg4 := by
  sorry

end table_wobbles_l1981_198124


namespace perp_to_countless_lines_necessary_not_sufficient_l1981_198114

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Perpendicularity between a line and a plane -/
def perp_line_plane (l : Line3D) (α : Plane3D) : Prop := sorry

/-- A line is perpendicular to countless lines within a plane -/
def perp_to_countless_lines (l : Line3D) (α : Plane3D) : Prop := sorry

/-- Main theorem: The statement "Line l is perpendicular to countless lines within plane α" 
    is a necessary but not sufficient condition for "l ⊥ α" -/
theorem perp_to_countless_lines_necessary_not_sufficient (l : Line3D) (α : Plane3D) :
  (perp_line_plane l α → perp_to_countless_lines l α) ∧
  ∃ l' α', perp_to_countless_lines l' α' ∧ ¬perp_line_plane l' α' := by
  sorry

end perp_to_countless_lines_necessary_not_sufficient_l1981_198114


namespace peggy_doll_ratio_l1981_198125

/-- Represents the number of dolls in various situations --/
structure DollCount where
  initial : Nat
  fromGrandmother : Nat
  final : Nat

/-- Calculates the ratio of birthday/Christmas dolls to grandmother's dolls --/
def dollRatio (d : DollCount) : Rat :=
  let birthdayChristmas := d.final - d.initial - d.fromGrandmother
  birthdayChristmas / d.fromGrandmother

/-- Theorem stating the ratio of dolls Peggy received --/
theorem peggy_doll_ratio (d : DollCount) 
  (h1 : d.initial = 6)
  (h2 : d.fromGrandmother = 30)
  (h3 : d.final = 51) :
  dollRatio d = 1/2 := by
  sorry

#eval dollRatio ⟨6, 30, 51⟩

end peggy_doll_ratio_l1981_198125


namespace intersection_points_count_l1981_198184

/-- A triangle with sides divided into p equal segments, where p is an odd prime -/
structure DividedTriangle where
  p : ℕ
  is_odd_prime : Nat.Prime p ∧ p % 2 = 1

/-- The number of intersection points in a divided triangle -/
def intersection_points (t : DividedTriangle) : ℕ := 3 * (t.p - 1)^2

/-- Theorem: The number of intersection points in a divided triangle is 3(p-1)^2 -/
theorem intersection_points_count (t : DividedTriangle) : 
  intersection_points t = 3 * (t.p - 1)^2 := by
  sorry


end intersection_points_count_l1981_198184


namespace simple_interest_problem_l1981_198120

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem simple_interest_problem :
  let principal : ℝ := 10000
  let rate : ℝ := 0.08
  let time : ℝ := 1
  simple_interest principal rate time = 800 := by sorry

end simple_interest_problem_l1981_198120


namespace second_part_speed_l1981_198109

/-- Represents a bicycle trip with three parts -/
structure BicycleTrip where
  total_distance : ℝ
  time_per_part : ℝ
  speed_first_part : ℝ
  speed_last_part : ℝ

/-- Theorem stating the speed of the second part of the trip -/
theorem second_part_speed (trip : BicycleTrip)
  (h_distance : trip.total_distance = 12)
  (h_time : trip.time_per_part = 0.25)
  (h_speed1 : trip.speed_first_part = 16)
  (h_speed3 : trip.speed_last_part = 20) :
  let distance1 := trip.speed_first_part * trip.time_per_part
  let distance3 := trip.speed_last_part * trip.time_per_part
  let distance2 := trip.total_distance - (distance1 + distance3)
  distance2 / trip.time_per_part = 12 := by
  sorry

#check second_part_speed

end second_part_speed_l1981_198109


namespace log_inequality_l1981_198192

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + Real.sqrt x) < Real.sqrt x := by
  sorry

end log_inequality_l1981_198192


namespace percent_relation_l1981_198182

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10 / 3) * a := by sorry

end percent_relation_l1981_198182


namespace ellie_wide_reflections_count_l1981_198110

/-- The number of times Sarah sees her reflection in tall mirror rooms -/
def sarah_tall_reflections : ℕ := 10

/-- The number of times Sarah sees her reflection in wide mirror rooms -/
def sarah_wide_reflections : ℕ := 5

/-- The number of times Ellie sees her reflection in tall mirror rooms -/
def ellie_tall_reflections : ℕ := 6

/-- The number of times both Sarah and Ellie passed through tall mirror rooms -/
def tall_room_visits : ℕ := 3

/-- The number of times both Sarah and Ellie passed through wide mirror rooms -/
def wide_room_visits : ℕ := 5

/-- The total number of reflections for both Sarah and Ellie -/
def total_reflections : ℕ := 88

/-- The number of times Ellie sees her reflection in wide mirror rooms -/
def ellie_wide_reflections : ℕ := 3

theorem ellie_wide_reflections_count :
  sarah_tall_reflections * tall_room_visits +
  sarah_wide_reflections * wide_room_visits +
  ellie_tall_reflections * tall_room_visits +
  ellie_wide_reflections * wide_room_visits = total_reflections :=
by sorry

end ellie_wide_reflections_count_l1981_198110


namespace sufficient_not_necessary_l1981_198176

theorem sufficient_not_necessary (a b : ℝ) :
  (b > a ∧ a > 0 → (a + 2) / (b + 2) > a / b) ∧
  ∃ a b : ℝ, (a + 2) / (b + 2) > a / b ∧ ¬(b > a ∧ a > 0) :=
sorry

end sufficient_not_necessary_l1981_198176


namespace roots_sum_sixth_power_l1981_198165

theorem roots_sum_sixth_power (u v : ℝ) : 
  u^2 - 3 * u * Real.sqrt 3 + 3 = 0 →
  v^2 - 3 * v * Real.sqrt 3 + 3 = 0 →
  u^6 + v^6 = 178767 := by
sorry

end roots_sum_sixth_power_l1981_198165


namespace prime_sum_squares_l1981_198112

theorem prime_sum_squares (p q : ℕ) : 
  Prime p ∧ Prime q ∧ Prime (2^2 + p^2 + q^2) ↔ (p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3) :=
sorry

end prime_sum_squares_l1981_198112


namespace max_k_value_l1981_198171

theorem max_k_value (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a + b + c = a * b + b * c + c * a) :
  ∃ k : ℝ, k = 1 ∧ 
  ∀ k' : ℝ, 
    ((a + b + c) * ((1 / (a + b)) + (1 / (c + b)) + (1 / (a + c)) - k') ≥ k') → 
    k' ≤ k :=
by sorry

end max_k_value_l1981_198171


namespace sum_of_fractions_simplification_l1981_198177

theorem sum_of_fractions_simplification 
  (p q r : ℝ) 
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) 
  (h_sum : p + q + r = 1) :
  1 / (q^2 + r^2 - p^2) + 1 / (p^2 + r^2 - q^2) + 1 / (p^2 + q^2 - r^2) = 3 / (1 - 2*q*r) :=
by sorry

end sum_of_fractions_simplification_l1981_198177


namespace complex_sum_of_parts_l1981_198175

theorem complex_sum_of_parts (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  (z * Complex.mk 1 2 = 5) → x + y = -1 := by
  sorry

end complex_sum_of_parts_l1981_198175


namespace expression_evaluation_l1981_198138

theorem expression_evaluation :
  (2 ^ 2010 * 3 ^ 2012 * 5 ^ 2) / 6 ^ 2011 = 37.5 := by
  sorry

end expression_evaluation_l1981_198138


namespace expression_value_l1981_198148

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 4)  -- absolute value of m is 4
  : a + b - (c * d) ^ 2021 - 3 * m = -13 ∨ a + b - (c * d) ^ 2021 - 3 * m = 11 :=
by sorry

end expression_value_l1981_198148


namespace negation_of_universal_proposition_l1981_198100

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end negation_of_universal_proposition_l1981_198100


namespace forum_member_count_l1981_198143

/-- The number of members in an online forum. -/
def forum_members : ℕ := 200

/-- The average number of questions posted per hour by each member. -/
def questions_per_hour : ℕ := 3

/-- The ratio of answers to questions posted by each member. -/
def answer_to_question_ratio : ℕ := 3

/-- The total number of posts (questions and answers) in a day. -/
def total_daily_posts : ℕ := 57600

/-- The number of hours in a day. -/
def hours_per_day : ℕ := 24

theorem forum_member_count :
  forum_members * (questions_per_hour * hours_per_day * (1 + answer_to_question_ratio)) = total_daily_posts :=
by sorry

end forum_member_count_l1981_198143


namespace star_A_B_equals_result_l1981_198178

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {y | y ≥ 1}

-- Define the operation *
def star (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∪ Y ∧ x ∉ X ∩ Y}

-- State the theorem
theorem star_A_B_equals_result : star A B = {x | (0 ≤ x ∧ x < 1) ∨ x > 3} := by sorry

end star_A_B_equals_result_l1981_198178


namespace election_margin_theorem_l1981_198168

/-- Represents an election with two candidates -/
structure Election where
  total_votes : ℕ
  winner_votes : ℕ
  winner_percentage : ℚ

/-- Calculates the margin of victory in an election -/
def margin_of_victory (e : Election) : ℕ :=
  e.winner_votes - (e.total_votes - e.winner_votes)

/-- Theorem stating the margin of victory for the given election scenario -/
theorem election_margin_theorem (e : Election) 
  (h1 : e.winner_percentage = 65 / 100)
  (h2 : e.winner_votes = 650) :
  margin_of_victory e = 300 := by
sorry

#eval margin_of_victory { total_votes := 1000, winner_votes := 650, winner_percentage := 65 / 100 }

end election_margin_theorem_l1981_198168


namespace right_triangle_sin_z_l1981_198181

theorem right_triangle_sin_z (X Y Z : ℝ) : 
  -- XYZ is a right triangle
  0 ≤ X ∧ X < π/2 ∧ 0 ≤ Y ∧ Y < π/2 ∧ 0 ≤ Z ∧ Z < π/2 ∧ X + Y + Z = π/2 →
  -- sin X = 3/5
  Real.sin X = 3/5 →
  -- cos Y = 0
  Real.cos Y = 0 →
  -- Then sin Z = 3/5
  Real.sin Z = 3/5 := by sorry

end right_triangle_sin_z_l1981_198181


namespace area_of_wxuv_l1981_198174

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the division of a rectangle into four smaller rectangles -/
structure RectangleDivision where
  pqxw : Rectangle
  qrsx : Rectangle
  xstu : Rectangle
  wxuv : Rectangle

theorem area_of_wxuv (div : RectangleDivision)
  (h1 : div.pqxw.area = 9)
  (h2 : div.qrsx.area = 10)
  (h3 : div.xstu.area = 15) :
  div.wxuv.area = 27 / 2 := by
  sorry

end area_of_wxuv_l1981_198174


namespace simplify_expression_1_simplify_expression_2_simplify_and_evaluate_expression_l1981_198188

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  5 * a * b^2 - 3 * a * b^2 + (1/3) * a * b^2 = (7/3) * a * b^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℝ) :
  (7 * m^2 * n - 5 * m) - (4 * m^2 * n - 5 * m) = 3 * m^2 * n := by sorry

-- Problem 3
theorem simplify_and_evaluate_expression (x y : ℝ) 
  (hx : x = -1/4) (hy : y = 2) :
  2 * x^2 * y - 2 * (x * y^2 + 2 * x^2 * y) + 2 * (x^2 * y - 3 * x * y^2) = 8 := by sorry

end simplify_expression_1_simplify_expression_2_simplify_and_evaluate_expression_l1981_198188
