import Mathlib

namespace smallest_AAB_existence_AAB_l1412_141228

-- Define the structure for our number
structure SpecialNumber where
  A : Nat
  B : Nat
  AB : Nat
  AAB : Nat

-- Define the conditions
def validNumber (n : SpecialNumber) : Prop :=
  1 ≤ n.A ∧ n.A ≤ 9 ∧
  1 ≤ n.B ∧ n.B ≤ 9 ∧
  n.AB = 10 * n.A + n.B ∧
  n.AAB = 110 * n.A + n.B ∧
  n.AB = n.AAB / 8

-- Define the theorem
theorem smallest_AAB :
  ∀ n : SpecialNumber, validNumber n → n.AAB ≥ 221 := by
  sorry

-- Define the existence of such a number
theorem existence_AAB :
  ∃ n : SpecialNumber, validNumber n ∧ n.AAB = 221 := by
  sorry

end smallest_AAB_existence_AAB_l1412_141228


namespace dice_events_properties_l1412_141276

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A, B, C, and D
def A : Set Ω := {ω | ω.1 = 2}
def B : Set Ω := {ω | ω.2 < 5}
def C : Set Ω := {ω | (ω.1.val + ω.2.val) % 2 = 1}
def D : Set Ω := {ω | ω.1.val + ω.2.val = 9}

-- State the theorem
theorem dice_events_properties :
  (¬(A ∩ B = ∅) ∧ P (A ∩ B) = P A * P B) ∧
  (A ∩ D = ∅ ∧ P (A ∩ D) ≠ P A * P D) ∧
  (¬(A ∩ C = ∅) ∧ P (A ∩ C) = P A * P C) :=
sorry

end dice_events_properties_l1412_141276


namespace circle_tangent_and_center_range_l1412_141232

-- Define the given points and lines
def A : ℝ × ℝ := (0, 3)
def l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C
def C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 1}

-- Define the condition for the center of C
def center_condition (center : ℝ × ℝ) : Prop :=
  center.2 = l center.1 ∧ center.2 = center.1 - 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  3 * x + 4 * y - 12 = 0

-- Define the condition for point M
def M_condition (center : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  M ∈ C center ∧ (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4 * ((M.1 - center.1)^2 + (M.2 - center.2)^2)

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 12/5

-- State the theorem
theorem circle_tangent_and_center_range :
  ∃ (center : ℝ × ℝ),
    center_condition center ∧
    (∀ (x y : ℝ), (x, y) ∈ C center → tangent_line x y) ∧
    (∃ (M : ℝ × ℝ), M_condition center M) ∧
    a_range center.1 := by sorry

end circle_tangent_and_center_range_l1412_141232


namespace donny_gas_change_l1412_141258

/-- Calculates the change Donny will receive after filling up his gas tank. -/
theorem donny_gas_change (tank_capacity : ℝ) (current_fuel : ℝ) (cost_per_liter : ℝ) (amount_paid : ℝ)
  (h1 : tank_capacity = 150)
  (h2 : current_fuel = 38)
  (h3 : cost_per_liter = 3)
  (h4 : amount_paid = 350) :
  amount_paid - (tank_capacity - current_fuel) * cost_per_liter = 14 := by
  sorry

end donny_gas_change_l1412_141258


namespace absolute_difference_of_points_on_curve_l1412_141227

theorem absolute_difference_of_points_on_curve (e p q : ℝ) : 
  (p^2 + e^4 = 4 * e^2 * p + 6) →
  (q^2 + e^4 = 4 * e^2 * q + 6) →
  |p - q| = 2 * Real.sqrt (3 * e^4 + 6) := by
  sorry

end absolute_difference_of_points_on_curve_l1412_141227


namespace problem_statement_l1412_141202

theorem problem_statement : (Real.sqrt 5 + 2)^2 + (-1/2)⁻¹ - Real.sqrt 49 = 4 * Real.sqrt 5 := by
  sorry

end problem_statement_l1412_141202


namespace PA_square_property_l1412_141259

theorem PA_square_property : 
  {PA : ℕ | 
    10 ≤ PA ∧ PA < 100 ∧ 
    1000 ≤ PA^2 ∧ PA^2 < 10000 ∧
    (PA^2 / 1000 = PA / 10) ∧ 
    (PA^2 % 10 = PA % 10)} = {95, 96} := by
  sorry

end PA_square_property_l1412_141259


namespace hall_breadth_is_15_meters_l1412_141244

-- Define the hall length in meters
def hall_length : ℝ := 36

-- Define stone dimensions in decimeters
def stone_length : ℝ := 6
def stone_width : ℝ := 5

-- Define the number of stones
def num_stones : ℕ := 1800

-- Define the conversion factor from square decimeters to square meters
def dm2_to_m2 : ℝ := 0.01

-- Statement to prove
theorem hall_breadth_is_15_meters :
  let stone_area_m2 := stone_length * stone_width * dm2_to_m2
  let total_area_m2 := stone_area_m2 * num_stones
  let hall_breadth := total_area_m2 / hall_length
  hall_breadth = 15 := by sorry

end hall_breadth_is_15_meters_l1412_141244


namespace jose_chickens_l1412_141205

/-- Given that Jose has 46 fowls in total and 18 ducks, prove that he has 28 chickens. -/
theorem jose_chickens : 
  ∀ (total_fowls ducks chickens : ℕ), 
    total_fowls = 46 → 
    ducks = 18 → 
    total_fowls = ducks + chickens → 
    chickens = 28 := by
  sorry

end jose_chickens_l1412_141205


namespace vertical_line_slope_undefined_l1412_141298

/-- The slope of a line passing through two distinct points with the same x-coordinate does not exist -/
theorem vertical_line_slope_undefined (y : ℝ) (h : y ≠ -3) :
  ¬∃ m : ℝ, ∀ x, x = 5 → (y - (-3)) = m * (x - 5) :=
sorry

end vertical_line_slope_undefined_l1412_141298


namespace three_speakers_from_different_companies_l1412_141206

/-- The number of companies in the meeting -/
def total_companies : ℕ := 5

/-- The number of representatives from Company A -/
def company_a_reps : ℕ := 2

/-- The number of representatives from each of the other companies -/
def other_company_reps : ℕ := 1

/-- The number of speakers at the meeting -/
def num_speakers : ℕ := 3

/-- The number of scenarios where 3 speakers come from 3 different companies -/
def num_scenarios : ℕ := 16

theorem three_speakers_from_different_companies :
  (Nat.choose company_a_reps 1 * Nat.choose (total_companies - 1) 2) +
  (Nat.choose (total_companies - 1) 3) = num_scenarios :=
sorry

end three_speakers_from_different_companies_l1412_141206


namespace smallest_number_l1412_141240

theorem smallest_number (a b c d : ℝ) 
  (ha : a = -Real.sqrt 2) 
  (hb : b = 0) 
  (hc : c = 3.14) 
  (hd : d = 2021) : 
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end smallest_number_l1412_141240


namespace circumscribed_sphere_surface_area_l1412_141233

/-- The surface area of a sphere circumscribing a cube with side length 4 is 48π. -/
theorem circumscribed_sphere_surface_area (cube_side : ℝ) (h : cube_side = 4) :
  let sphere_radius := cube_side * Real.sqrt 3 / 2
  4 * Real.pi * sphere_radius^2 = 48 * Real.pi :=
by sorry

end circumscribed_sphere_surface_area_l1412_141233


namespace f_zero_at_three_l1412_141295

/-- The function f(x) = 3x^3 + 2x^2 - 5x + s -/
def f (s : ℝ) (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - 5 * x + s

/-- Theorem: f(3) = 0 if and only if s = -84 -/
theorem f_zero_at_three (s : ℝ) : f s 3 = 0 ↔ s = -84 := by
  sorry

end f_zero_at_three_l1412_141295


namespace rational_equation_solution_l1412_141287

theorem rational_equation_solution : 
  ∃! y : ℚ, (y^2 - 12*y + 32) / (y - 2) + (3*y^2 + 11*y - 14) / (3*y - 1) = -5 ∧ y = -17/6 := by
  sorry

end rational_equation_solution_l1412_141287


namespace age_ratio_proof_l1412_141274

theorem age_ratio_proof (my_current_age brother_current_age : ℕ) 
  (h1 : my_current_age = 20)
  (h2 : my_current_age + 10 + (brother_current_age + 10) = 45) :
  (my_current_age + 10) / (brother_current_age + 10) = 2 := by
  sorry

end age_ratio_proof_l1412_141274


namespace circle_area_below_line_l1412_141252

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 - 14*y + 33 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop :=
  y = 7

-- Theorem statement
theorem circle_area_below_line :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_y = 7 ∧
    radius > 0 ∧
    (π * radius^2 / 2 : ℝ) = 25 * π / 2 :=
sorry

end circle_area_below_line_l1412_141252


namespace twenty_first_term_equals_203_l1412_141290

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * seq.common_difference

theorem twenty_first_term_equals_203 :
  ∃ (seq : ArithmeticSequence),
    seq.first_term = 3 ∧
    nth_term seq 2 = 13 ∧
    nth_term seq 3 = 23 ∧
    nth_term seq 21 = 203 := by
  sorry

end twenty_first_term_equals_203_l1412_141290


namespace sandy_age_l1412_141243

theorem sandy_age :
  ∀ (S M J : ℕ),
    S = M - 14 →
    J = S + 6 →
    9 * S = 7 * M →
    6 * S = 5 * J →
    S = 49 :=
by
  sorry

end sandy_age_l1412_141243


namespace binomial_seven_four_l1412_141210

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end binomial_seven_four_l1412_141210


namespace distance_walked_proof_l1412_141245

/-- Calculates the distance walked by a person given their step length, steps per minute, and duration of walk. -/
def distanceWalked (stepLength : ℝ) (stepsPerMinute : ℝ) (durationMinutes : ℝ) : ℝ :=
  stepLength * stepsPerMinute * durationMinutes

/-- Proves that a person walking 0.75 meters per step at 70 steps per minute for 13 minutes covers 682.5 meters. -/
theorem distance_walked_proof :
  distanceWalked 0.75 70 13 = 682.5 := by
  sorry

#eval distanceWalked 0.75 70 13

end distance_walked_proof_l1412_141245


namespace inequality_proof_l1412_141203

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) :=
by sorry

end inequality_proof_l1412_141203


namespace ellipse_equation_from_properties_l1412_141237

/-- An ellipse with specified properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  min_distance_to_focus : ℝ
  eccentricity : ℝ

/-- The equation of an ellipse given its properties -/
def ellipse_equation (E : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 8 + y^2 / 4 = 1

/-- Theorem stating that an ellipse with the given properties has the specified equation -/
theorem ellipse_equation_from_properties (E : Ellipse) 
  (h1 : E.center = (0, 0))
  (h2 : E.foci_on_x_axis = true)
  (h3 : E.min_distance_to_focus = 2 * Real.sqrt 2 - 2)
  (h4 : E.eccentricity = Real.sqrt 2 / 2) :
  ellipse_equation E = fun x y => x^2 / 8 + y^2 / 4 = 1 :=
sorry

end ellipse_equation_from_properties_l1412_141237


namespace green_sequin_rows_jane_green_sequin_rows_l1412_141234

/-- Calculates the number of rows of green sequins in Jane's costume. -/
theorem green_sequin_rows (blue_rows : Nat) (blue_per_row : Nat) 
  (purple_rows : Nat) (purple_per_row : Nat) (green_per_row : Nat) 
  (total_sequins : Nat) : Nat :=
  let blue_sequins := blue_rows * blue_per_row
  let purple_sequins := purple_rows * purple_per_row
  let blue_and_purple := blue_sequins + purple_sequins
  let green_sequins := total_sequins - blue_and_purple
  let green_rows := green_sequins / green_per_row
  green_rows

/-- Proves that Jane sews 9 rows of green sequins. -/
theorem jane_green_sequin_rows : 
  green_sequin_rows 6 8 5 12 6 162 = 9 := by
  sorry

end green_sequin_rows_jane_green_sequin_rows_l1412_141234


namespace parabola_vertex_y_coordinate_l1412_141216

/-- Given a quadratic function f(x) = 5x^2 + 20x + 45, 
    prove that the y-coordinate of its vertex is 25. -/
theorem parabola_vertex_y_coordinate :
  let f : ℝ → ℝ := λ x ↦ 5 * x^2 + 20 * x + 45
  ∃ h k : ℝ, (∀ x, f x = 5 * (x - h)^2 + k) ∧ k = 25 :=
by sorry

end parabola_vertex_y_coordinate_l1412_141216


namespace coefficient_x_cubed_is_zero_l1412_141255

/-- Given q(x) = x⁴ - 4x + 5, the coefficient of x³ in (q(x))² is 0 -/
theorem coefficient_x_cubed_is_zero (x : ℝ) : 
  let q := fun (x : ℝ) => x^4 - 4*x + 5
  (q x)^2 = x^8 - 8*x^5 + 10*x^4 + 16*x^2 - 40*x + 25 :=
by
  sorry

end coefficient_x_cubed_is_zero_l1412_141255


namespace fuel_mixture_cost_l1412_141246

/-- Represents the cost of the other liquid per gallon -/
def other_liquid_cost : ℝ := 3

/-- The total volume of the mixture in gallons -/
def total_volume : ℝ := 12

/-- The cost of the final fuel mixture per gallon -/
def final_fuel_cost : ℝ := 8

/-- The cost of oil per gallon -/
def oil_cost : ℝ := 15

/-- The volume of one of the liquids used in the mixture -/
def one_liquid_volume : ℝ := 7

theorem fuel_mixture_cost : 
  one_liquid_volume * other_liquid_cost + (total_volume - one_liquid_volume) * oil_cost = 
  total_volume * final_fuel_cost :=
sorry

end fuel_mixture_cost_l1412_141246


namespace solve_system_l1412_141292

theorem solve_system (s t : ℚ) 
  (eq1 : 7 * s + 8 * t = 150)
  (eq2 : s = 2 * t + 3) : 
  s = 162 / 11 := by
  sorry

end solve_system_l1412_141292


namespace similar_triangles_perimeter_area_l1412_141256

/-- Given two similar triangles with corresponding median lengths and sum of perimeters,
    prove their individual perimeters and area ratio -/
theorem similar_triangles_perimeter_area (median1 median2 perimeter_sum : ℝ) :
  median1 = 10 →
  median2 = 4 →
  perimeter_sum = 140 →
  ∃ (perimeter1 perimeter2 area1 area2 : ℝ),
    perimeter1 = 100 ∧
    perimeter2 = 40 ∧
    perimeter1 + perimeter2 = perimeter_sum ∧
    (area1 / area2 = 25 / 4) ∧
    (median1 / median2)^2 = area1 / area2 ∧
    median1 / median2 = perimeter1 / perimeter2 :=
by sorry

end similar_triangles_perimeter_area_l1412_141256


namespace sam_speed_l1412_141280

/-- Represents a point on the route --/
structure Point where
  position : ℝ

/-- Represents a person traveling on the route --/
structure Traveler where
  start : Point
  speed : ℝ

/-- The scenario of Sam and Nik's travel --/
structure TravelScenario where
  sam : Traveler
  nik : Traveler
  meetingPoint : Point
  totalTime : ℝ

/-- The given travel scenario --/
def givenScenario : TravelScenario where
  sam := { start := { position := 0 }, speed := 0 }  -- speed will be calculated
  nik := { start := { position := 1000 }, speed := 0 }  -- speed is not needed for the problem
  meetingPoint := { position := 600 }
  totalTime := 20

theorem sam_speed (scenario : TravelScenario) :
  scenario.sam.start.position = 0 ∧
  scenario.nik.start.position = 1000 ∧
  scenario.meetingPoint.position = 600 ∧
  scenario.totalTime = 20 →
  scenario.sam.speed = 50 := by
  sorry

end sam_speed_l1412_141280


namespace distance_to_intersecting_line_l1412_141291

/-- Ellipse G with equation x^2/8 + y^2/4 = 1 -/
def ellipse_G (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

/-- Left focus F1(-2,0) -/
def F1 : ℝ × ℝ := (-2, 0)

/-- Right focus F2(2,0) -/
def F2 : ℝ × ℝ := (2, 0)

/-- Line l intersects ellipse G at points A and B -/
def intersects_ellipse (l : Set (ℝ × ℝ)) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ l ∧ B ∈ l ∧ ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2

/-- OA is perpendicular to OB -/
def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: For ellipse G, if line l intersects G at A and B with OA ⊥ OB,
    then the distance from O to l is 2√6/3 -/
theorem distance_to_intersecting_line :
  ∀ l : Set (ℝ × ℝ),
  intersects_ellipse l →
  (∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ l ∧ B ∈ l ∧ ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2 ∧ perpendicular A B) →
  distance_point_to_line (0, 0) l = 2 * Real.sqrt 6 / 3 :=
sorry

end distance_to_intersecting_line_l1412_141291


namespace even_function_shift_l1412_141264

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- Theorem statement
theorem even_function_shift (a : ℝ) :
  is_even (fun x ↦ f (x + a)) → a = 2 := by
  sorry

end even_function_shift_l1412_141264


namespace investment_sum_proof_l1412_141222

/-- Proves that a sum invested at 15% p.a. simple interest for two years yields
    Rs. 420 more interest than if invested at 12% p.a. for the same period,
    then the sum is Rs. 7000. -/
theorem investment_sum_proof (P : ℚ) 
  (h1 : P * (15 / 100) * 2 - P * (12 / 100) * 2 = 420) : P = 7000 := by
  sorry

end investment_sum_proof_l1412_141222


namespace greatest_integer_less_than_negative_seventeen_thirds_l1412_141226

theorem greatest_integer_less_than_negative_seventeen_thirds :
  Int.floor (-17 / 3 : ℚ) = -6 := by sorry

end greatest_integer_less_than_negative_seventeen_thirds_l1412_141226


namespace max_abs_diff_f_g_l1412_141215

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the absolute difference function
def absDiff (x : ℝ) : ℝ := |f x - g x|

-- State the theorem
theorem max_abs_diff_f_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ 
  (∀ x, x ∈ Set.Icc 0 1 → absDiff x ≤ absDiff c) ∧
  absDiff c = 4/27 :=
sorry

end max_abs_diff_f_g_l1412_141215


namespace odd_2n_plus_1_l1412_141284

theorem odd_2n_plus_1 (n : ℤ) : ¬ (∃ k : ℤ, 2 * n + 1 = 2 * k) := by
  sorry

end odd_2n_plus_1_l1412_141284


namespace fran_required_speed_l1412_141221

/-- Given Joann's bike ride parameters and Fran's ride time, calculate Fran's required speed -/
theorem fran_required_speed 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4) 
  (h3 : fran_time = 2.5) : 
  (joann_speed * joann_time) / fran_time = 24 := by
sorry

end fran_required_speed_l1412_141221


namespace cow_inheritance_problem_l1412_141201

theorem cow_inheritance_problem (x y : ℕ) (z : ℝ) 
  (h1 : x^2 = 10*y + z)
  (h2 : z < 10)
  (h3 : Odd y)
  (h4 : x^2 % 10 = 6) :
  (10 - z) / 2 = 2 := by
  sorry

end cow_inheritance_problem_l1412_141201


namespace reinforcement_arrival_time_l1412_141250

/-- Calculates the number of days passed before reinforcement arrived -/
def days_before_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) 
  (reinforcement : ℕ) (remaining_provisions : ℕ) : ℕ :=
  (initial_garrison * initial_provisions - (initial_garrison + reinforcement) * remaining_provisions) / initial_garrison

/-- Theorem stating that 15 days passed before reinforcement arrived -/
theorem reinforcement_arrival_time :
  days_before_reinforcement 2000 65 3000 20 = 15 := by sorry

end reinforcement_arrival_time_l1412_141250


namespace auto_credit_percentage_l1412_141213

def auto_finance_credit : ℝ := 40
def total_consumer_credit : ℝ := 342.857

theorem auto_credit_percentage :
  let total_auto_credit := 3 * auto_finance_credit
  let percentage := (total_auto_credit / total_consumer_credit) * 100
  ∃ ε > 0, |percentage - 35| < ε :=
sorry

end auto_credit_percentage_l1412_141213


namespace range_of_a_l1412_141231

-- Define the function f
def f (x : ℝ) : ℝ := 3*x + 2*x^3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 2, f x = 3*x + 2*x^3) →
  (f (a - 1) + f (1 - 2*a) < 0) →
  a ∈ Set.Ioo 0 (3/2) :=
by sorry

end range_of_a_l1412_141231


namespace digit_150_of_5_over_13_l1412_141217

def decimal_representation (n d : ℕ) : ℚ := n / d

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem digit_150_of_5_over_13 : 
  nth_digit_after_decimal (decimal_representation 5 13) 150 = 5 := by sorry

end digit_150_of_5_over_13_l1412_141217


namespace color_change_probability_l1412_141214

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  green_duration : ℕ
  yellow_duration : ℕ
  red_duration : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycle_duration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green_duration + cycle.yellow_duration + cycle.red_duration

/-- Calculates the number of seconds where a color change can be observed in a 4-second interval -/
def change_observation_duration (cycle : TrafficLightCycle) : ℕ := 12

/-- Theorem: The probability of observing a color change during a random 4-second interval
    in the given traffic light cycle is 0.12 -/
theorem color_change_probability (cycle : TrafficLightCycle) 
  (h1 : cycle.green_duration = 45)
  (h2 : cycle.yellow_duration = 5)
  (h3 : cycle.red_duration = 50)
  (h4 : change_observation_duration cycle = 12) :
  (change_observation_duration cycle : ℚ) / (cycle_duration cycle) = 12 / 100 := by
  sorry

end color_change_probability_l1412_141214


namespace peanut_butter_duration_l1412_141225

/-- The number of days that peanut butter lasts for Phoebe and her dog -/
def peanut_butter_days (servings_per_jar : ℕ) (num_jars : ℕ) (daily_consumption : ℕ) : ℕ :=
  (servings_per_jar * num_jars) / daily_consumption

/-- Theorem stating how long 4 jars of peanut butter will last for Phoebe and her dog -/
theorem peanut_butter_duration :
  let servings_per_jar : ℕ := 15
  let num_jars : ℕ := 4
  let phoebe_consumption : ℕ := 1
  let dog_consumption : ℕ := 1
  let daily_consumption : ℕ := phoebe_consumption + dog_consumption
  peanut_butter_days servings_per_jar num_jars daily_consumption = 30 := by
  sorry

end peanut_butter_duration_l1412_141225


namespace pet_snake_cost_l1412_141247

def initial_amount : ℕ := 73
def amount_left : ℕ := 18

theorem pet_snake_cost : initial_amount - amount_left = 55 := by sorry

end pet_snake_cost_l1412_141247


namespace third_place_winnings_l1412_141281

theorem third_place_winnings (num_people : ℕ) (contribution : ℝ) (first_place_percentage : ℝ) :
  num_people = 8 →
  contribution = 5 →
  first_place_percentage = 0.8 →
  let total_pot := num_people * contribution
  let first_place_amount := first_place_percentage * total_pot
  let remaining_amount := total_pot - first_place_amount
  remaining_amount / 2 = 4 := by sorry

end third_place_winnings_l1412_141281


namespace intersection_sum_l1412_141229

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 2
def g (x y : ℝ) : Prop := x + 2*y = 2

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
  f x₁ = y₁ ∧ g x₁ y₁ ∧
  f x₂ = y₂ ∧ g x₂ y₂ ∧
  f x₃ = y₃ ∧ g x₃ y₃

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    f x₁ = y₁ ∧ g x₁ y₁ ∧
    f x₂ = y₂ ∧ g x₂ y₂ ∧
    f x₃ = y₃ ∧ g x₃ y₃ ∧
    x₁ + x₂ + x₃ = 4 ∧
    y₁ + y₂ + y₃ = 1 :=
by
  sorry

end intersection_sum_l1412_141229


namespace polynomial_sum_of_squares_l1412_141268

/-- A polynomial with real coefficients that is non-negative for all real inputs
    can be expressed as the sum of squares of two polynomials. -/
theorem polynomial_sum_of_squares
  (P : Polynomial ℝ)
  (h : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 :=
sorry

end polynomial_sum_of_squares_l1412_141268


namespace complement_union_A_B_l1412_141211

open Set

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end complement_union_A_B_l1412_141211


namespace managers_salary_l1412_141200

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 1000 →
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + salary_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary = 22500 := by
  sorry

end managers_salary_l1412_141200


namespace curve_is_hyperbola_l1412_141294

/-- The equation of the curve in polar form -/
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.cos θ - Real.sin θ)

/-- The equation of the curve in Cartesian form -/
def cartesian_equation (x y : ℝ) : Prop :=
  x * y + x + y + (1/2) = 0

/-- Theorem stating that the curve is a hyperbola -/
theorem curve_is_hyperbola :
  ∃ (x y : ℝ), cartesian_equation x y ∧
  ∃ (r θ : ℝ), polar_equation r θ ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
sorry

end curve_is_hyperbola_l1412_141294


namespace first_child_2019th_number_l1412_141272

/-- Represents the counting game with three children -/
def CountingGame :=
  { n : ℕ | n > 0 ∧ n ≤ 10000 }

/-- The sequence of numbers said by the first child -/
def first_child_sequence (n : ℕ) : ℕ :=
  3 * n * n - 2 * n + 1

/-- The number of complete cycles before the 2019th number -/
def complete_cycles : ℕ := 36

/-- The position of the 2019th number within its cycle -/
def position_in_cycle : ℕ := 93

/-- The 2019th number said by the first child -/
theorem first_child_2019th_number :
  ∃ (game : CountingGame),
    first_child_sequence complete_cycles +
    position_in_cycle = 5979 :=
sorry

end first_child_2019th_number_l1412_141272


namespace garage_sale_games_l1412_141251

/-- The number of games Luke bought from a friend -/
def games_from_friend : ℕ := 2

/-- The number of games that didn't work -/
def broken_games : ℕ := 2

/-- The number of good games Luke ended up with -/
def good_games : ℕ := 2

/-- The number of games Luke bought at the garage sale -/
def games_from_garage_sale : ℕ := 2

theorem garage_sale_games :
  games_from_friend + games_from_garage_sale - broken_games = good_games :=
by sorry

end garage_sale_games_l1412_141251


namespace smallest_divisor_after_437_l1412_141224

theorem smallest_divisor_after_437 (m : ℕ) (h1 : 10000 ≤ m ∧ m ≤ 99999) 
  (h2 : Odd m) (h3 : m % 437 = 0) : 
  (∃ (d : ℕ), d > 437 ∧ d ∣ m ∧ ∀ (x : ℕ), 437 < x ∧ x < d → ¬(x ∣ m)) → 
  (Nat.minFac (m / 437) = 19 ∨ Nat.minFac (m / 437) = 23) :=
sorry

end smallest_divisor_after_437_l1412_141224


namespace min_tablets_extracted_l1412_141286

/-- The least number of tablets to extract to ensure at least two of each kind -/
def leastTablets (tabletA : ℕ) (tabletB : ℕ) : ℕ :=
  max (tabletB + 2) (tabletA + 2)

theorem min_tablets_extracted (tabletA tabletB : ℕ) 
  (hA : tabletA = 10) (hB : tabletB = 16) :
  leastTablets tabletA tabletB = 18 := by
  sorry

end min_tablets_extracted_l1412_141286


namespace sum_of_squares_of_roots_l1412_141271

theorem sum_of_squares_of_roots (a b : ℝ) : 
  a^2 - 15*a + 6 = 0 → 
  b^2 - 15*b + 6 = 0 → 
  a^2 + b^2 = 213 := by
sorry

end sum_of_squares_of_roots_l1412_141271


namespace mia_sock_purchase_l1412_141249

/-- Represents the number of pairs of socks at each price point --/
structure SockPurchase where
  twoDoller : ℕ
  threeDoller : ℕ
  fiveDoller : ℕ

/-- Checks if the given SockPurchase satisfies the problem conditions --/
def isValidPurchase (p : SockPurchase) : Prop :=
  p.twoDoller + p.threeDoller + p.fiveDoller = 15 ∧
  2 * p.twoDoller + 3 * p.threeDoller + 5 * p.fiveDoller = 35 ∧
  p.twoDoller ≥ 1 ∧ p.threeDoller ≥ 1 ∧ p.fiveDoller ≥ 1

theorem mia_sock_purchase :
  ∃ (p : SockPurchase), isValidPurchase p ∧ p.twoDoller = 12 := by
  sorry

end mia_sock_purchase_l1412_141249


namespace valid_rectangle_exists_l1412_141273

/-- Represents a point in a triangular grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a rectangle in the triangular grid -/
structure Rectangle where
  bottomLeft : GridPoint
  width : ℕ
  height : ℕ

/-- Counts the number of grid points on the boundary of a rectangle -/
def boundaryPoints (rect : Rectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- Counts the number of grid points in the interior of a rectangle -/
def interiorPoints (rect : Rectangle) : ℕ :=
  rect.width * rect.height + (rect.width - 1) * (rect.height - 1)

/-- Checks if a rectangle satisfies the required conditions -/
def isValidRectangle (rect : Rectangle) : Prop :=
  boundaryPoints rect = interiorPoints rect

/-- Main theorem: There exists a valid rectangle in the triangular grid -/
theorem valid_rectangle_exists : ∃ (rect : Rectangle), isValidRectangle rect :=
  sorry

end valid_rectangle_exists_l1412_141273


namespace divisibility_by_nine_l1412_141208

theorem divisibility_by_nine : ∃ d : ℕ, d < 10 ∧ (2345 * 10 + d) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end divisibility_by_nine_l1412_141208


namespace error_percentage_l1412_141219

theorem error_percentage (x : ℝ) (h : x > 0) : 
  (|4*x - x/4|) / (4*x) = 15/16 := by
sorry

end error_percentage_l1412_141219


namespace units_digit_27_times_64_l1412_141296

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The property that the units digit of a product depends only on the units digits of its factors -/
axiom units_digit_product (a b : ℕ) : 
  units_digit (a * b) = units_digit (units_digit a * units_digit b)

/-- The theorem stating that the units digit of 27 · 64 is 8 -/
theorem units_digit_27_times_64 : units_digit (27 * 64) = 8 := by
  sorry

end units_digit_27_times_64_l1412_141296


namespace david_rosy_age_difference_l1412_141299

theorem david_rosy_age_difference :
  ∀ (david_age rosy_age : ℕ),
    david_age > rosy_age →
    rosy_age = 8 →
    david_age + 4 = 2 * (rosy_age + 4) →
    david_age - rosy_age = 12 := by
  sorry

end david_rosy_age_difference_l1412_141299


namespace inequality_solution_set_l1412_141241

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x - 3) > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end inequality_solution_set_l1412_141241


namespace tangent_line_at_one_l1412_141288

/-- The function f(x) = -x³ + 4x -/
def f (x : ℝ) : ℝ := -x^3 + 4*x

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := -3*x^2 + 4

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = x + 2 :=
by sorry

end tangent_line_at_one_l1412_141288


namespace solution_set_f_derivative_positive_l1412_141238

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem solution_set_f_derivative_positive :
  {x : ℝ | (deriv f) x > 0} = {x : ℝ | x < -2 ∨ x > 0} := by sorry

end solution_set_f_derivative_positive_l1412_141238


namespace probability_both_selected_l1412_141262

theorem probability_both_selected (prob_X prob_Y : ℚ) 
  (h1 : prob_X = 1/3) (h2 : prob_Y = 2/5) : 
  prob_X * prob_Y = 2/15 := by
  sorry

end probability_both_selected_l1412_141262


namespace lawn_width_proof_l1412_141257

/-- The width of a rectangular lawn with specific conditions -/
def lawn_width : ℝ := 50

theorem lawn_width_proof (length : ℝ) (road_width : ℝ) (total_road_area : ℝ) :
  length = 80 →
  road_width = 10 →
  total_road_area = 1200 →
  lawn_width = (total_road_area - (length * road_width) + (road_width * road_width)) / road_width :=
by
  sorry

#check lawn_width_proof

end lawn_width_proof_l1412_141257


namespace min_value_of_expression_l1412_141236

theorem min_value_of_expression (a b : ℝ) (ha : a > 1) (hab : a * b = 2 * a + b) :
  (∀ x y : ℝ, x > 1 ∧ x * y = 2 * x + y → (a + 1) * (b + 2) ≤ (x + 1) * (y + 2)) ∧
  (a + 1) * (b + 2) = 18 := by
sorry

end min_value_of_expression_l1412_141236


namespace inequalities_always_satisfied_l1412_141223

theorem inequalities_always_satisfied (a b c x y z : ℝ) 
  (hx : x < a) (hy : y < b) (hz : z < c) : 
  (x * y * c < a * b * z) ∧ 
  (x^2 + c < a^2 + z) ∧ 
  (x^2 * y^2 * z^2 < a^2 * b^2 * c^2) := by
  sorry

end inequalities_always_satisfied_l1412_141223


namespace jennifer_remaining_money_l1412_141263

def total_money : ℚ := 180

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_money : ℚ := total_money - (sandwich_fraction * total_money + museum_fraction * total_money + book_fraction * total_money)

theorem jennifer_remaining_money :
  remaining_money = 24 := by sorry

end jennifer_remaining_money_l1412_141263


namespace solution_set_inequality_l1412_141275

theorem solution_set_inequality (x : ℝ) :
  (x + 5) * (1 - x) ≥ 8 ↔ -3 ≤ x ∧ x ≤ -1 := by
  sorry

end solution_set_inequality_l1412_141275


namespace number_of_sets_l1412_141242

/-- The number of flowers in each set -/
def flowers_per_set : ℕ := 90

/-- The total number of flowers bought -/
def total_flowers : ℕ := 270

/-- Theorem: The number of sets of flowers bought is 3 -/
theorem number_of_sets : total_flowers / flowers_per_set = 3 := by
  sorry

end number_of_sets_l1412_141242


namespace line_through_point_not_perpendicular_l1412_141270

/-- A line in the form y = k(x-2) passes through (2,0) and is not perpendicular to the x-axis -/
theorem line_through_point_not_perpendicular (k : ℝ) : 
  ∃ (x y : ℝ), y = k * (x - 2) → 
  (x = 2 ∧ y = 0) ∧ 
  (k ≠ 0 → ∃ (m : ℝ), m ≠ 0 ∧ k = 1 / m) :=
sorry

end line_through_point_not_perpendicular_l1412_141270


namespace election_combinations_theorem_l1412_141278

/-- Represents a club with members of different genders and ages -/
structure Club where
  total_members : Nat
  girls : Nat
  boys : Nat
  girls_age_order : Fin girls → Nat
  boys_age_order : Fin boys → Nat

/-- Represents the election rules for the club -/
structure ElectionRules where
  president_must_be_girl : Bool
  vp_must_be_boy : Bool
  vp_younger_than_president : Bool

/-- Calculates the number of ways to elect a president and vice-president -/
def election_combinations (club : Club) (rules : ElectionRules) : Nat :=
  sorry

/-- Theorem stating the number of election combinations for the given club and rules -/
theorem election_combinations_theorem (club : Club) (rules : ElectionRules) :
  club.total_members = 25 ∧
  club.girls = 13 ∧
  club.boys = 12 ∧
  rules.president_must_be_girl = true ∧
  rules.vp_must_be_boy = true ∧
  rules.vp_younger_than_president = true →
  election_combinations club rules = 78 :=
by
  sorry

end election_combinations_theorem_l1412_141278


namespace four_students_three_lectures_l1412_141293

/-- The number of ways students can choose lectures -/
def lecture_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 4 students choosing from 3 lectures results in 81 different selections -/
theorem four_students_three_lectures : 
  lecture_choices 4 3 = 81 := by
  sorry

end four_students_three_lectures_l1412_141293


namespace madhav_rank_from_last_l1412_141267

theorem madhav_rank_from_last (total_students : ℕ) (madhav_rank_start : ℕ) 
  (h1 : total_students = 31) (h2 : madhav_rank_start = 17) : 
  total_students - madhav_rank_start + 1 = 15 := by
  sorry

end madhav_rank_from_last_l1412_141267


namespace imaginary_part_of_z_l1412_141261

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2) : 
  z.im = -1 := by sorry

end imaginary_part_of_z_l1412_141261


namespace unique_positive_number_l1412_141248

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 / x := by
  sorry

end unique_positive_number_l1412_141248


namespace sum_integers_from_neg50_to_75_l1412_141282

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_from_neg50_to_75 :
  sum_integers (-50) 75 = 1575 := by
  sorry

end sum_integers_from_neg50_to_75_l1412_141282


namespace platform_length_l1412_141218

/-- Given a train of length 300 meters that crosses a platform in 33 seconds
    and a signal pole in 18 seconds, the length of the platform is 250 meters. -/
theorem platform_length
  (train_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 33)
  (h3 : pole_crossing_time = 18) :
  (train_length + platform_crossing_time * (train_length / pole_crossing_time) - train_length) = 250 :=
by sorry

end platform_length_l1412_141218


namespace child_ticket_cost_l1412_141235

/-- Proves that the cost of a child ticket is 1 dollar given the conditions of the problem -/
theorem child_ticket_cost
  (adult_ticket_cost : ℕ)
  (total_attendees : ℕ)
  (total_revenue : ℕ)
  (child_attendees : ℕ)
  (h1 : adult_ticket_cost = 8)
  (h2 : total_attendees = 22)
  (h3 : total_revenue = 50)
  (h4 : child_attendees = 18) :
  let adult_attendees : ℕ := total_attendees - child_attendees
  let child_ticket_cost : ℚ := (total_revenue - adult_ticket_cost * adult_attendees) / child_attendees
  child_ticket_cost = 1 := by sorry

end child_ticket_cost_l1412_141235


namespace two_by_six_grid_triangles_l1412_141297

/-- Represents a rectangular grid with diagonal lines --/
structure DiagonalGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (has_center_diagonals : Bool)

/-- Counts the number of triangles in a diagonal grid --/
def count_triangles (grid : DiagonalGrid) : ℕ :=
  sorry

/-- Theorem stating that a 2x6 grid with center diagonals has at least 88 triangles --/
theorem two_by_six_grid_triangles :
  ∀ (grid : DiagonalGrid),
    grid.rows = 2 ∧ 
    grid.cols = 6 ∧ 
    grid.has_center_diagonals = true →
    count_triangles grid ≥ 88 :=
by sorry

end two_by_six_grid_triangles_l1412_141297


namespace janet_earnings_l1412_141209

/-- Calculates the hourly earnings for checking social media posts. -/
def hourly_earnings (pay_per_post : ℚ) (seconds_per_post : ℕ) : ℚ :=
  let posts_per_hour : ℕ := 3600 / seconds_per_post
  pay_per_post * posts_per_hour

/-- Proves that given a pay rate of 25 cents per post and a checking time of 10 seconds per post, the hourly earnings are $90. -/
theorem janet_earnings : hourly_earnings (25 / 100) 10 = 90 := by
  sorry

end janet_earnings_l1412_141209


namespace stamps_problem_l1412_141253

theorem stamps_problem (cj kj aj : ℕ) : 
  cj = 2 * kj + 5 →  -- CJ has 5 more than twice the number of stamps that KJ has
  kj * 2 = aj →      -- KJ has half as many stamps as AJ
  cj + kj + aj = 930 →  -- The three boys have 930 stamps in total
  aj = 370 :=         -- Prove that AJ has 370 stamps
by
  sorry


end stamps_problem_l1412_141253


namespace line_segment_lattice_points_l1412_141269

/-- The number of lattice points on a line segment with given endpoints -/
def latticePointCount (x1 y1 x2 y2 : Int) : Nat :=
  sorry

theorem line_segment_lattice_points :
  latticePointCount 5 10 68 178 = 22 := by sorry

end line_segment_lattice_points_l1412_141269


namespace parabola_unique_intersection_l1412_141265

/-- A parabola defined by y = x^2 - 6x + m -/
def parabola (x m : ℝ) : ℝ := x^2 - 6*x + m

/-- Condition for the parabola to intersect the x-axis -/
def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x, parabola x m = 0

/-- Condition for the parabola to have exactly one intersection with the x-axis -/
def unique_intersection (m : ℝ) : Prop :=
  ∃! x, parabola x m = 0

theorem parabola_unique_intersection :
  ∃! m, unique_intersection m ∧ m = 9 :=
sorry

end parabola_unique_intersection_l1412_141265


namespace correct_quotient_proof_l1412_141254

theorem correct_quotient_proof (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 49) : D / 21 = 28 := by
  sorry

end correct_quotient_proof_l1412_141254


namespace units_digit_of_sum_l1412_141212

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def power_of_two (n : ℕ) : ℕ := 2^n

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def sum_powers_of_two (n : ℕ) : ℕ := (List.range n).map power_of_two |>.sum

theorem units_digit_of_sum (n : ℕ) : 
  (sum_factorials n + sum_powers_of_two n) % 10 = 9 :=
by sorry

end units_digit_of_sum_l1412_141212


namespace toys_ratio_saturday_to_wednesday_l1412_141277

/-- The number of rabbits Junior has -/
def num_rabbits : ℕ := 16

/-- The number of toys bought on Monday -/
def toys_monday : ℕ := 6

/-- The number of toys bought on Wednesday -/
def toys_wednesday : ℕ := 2 * toys_monday

/-- The number of toys bought on Friday -/
def toys_friday : ℕ := 4 * toys_monday

/-- The number of toys each rabbit has when split evenly -/
def toys_per_rabbit : ℕ := 3

/-- The total number of toys -/
def total_toys : ℕ := num_rabbits * toys_per_rabbit

/-- The number of toys bought on Saturday -/
def toys_saturday : ℕ := total_toys - (toys_monday + toys_wednesday + toys_friday)

theorem toys_ratio_saturday_to_wednesday :
  toys_saturday * 2 = toys_wednesday := by sorry

end toys_ratio_saturday_to_wednesday_l1412_141277


namespace smallest_value_of_sum_of_cubes_l1412_141289

theorem smallest_value_of_sum_of_cubes (a b : ℂ) 
  (h1 : Complex.abs (a + b) = 2)
  (h2 : Complex.abs (a^2 + b^2) = 8) :
  20 ≤ Complex.abs (a^3 + b^3) :=
by sorry

end smallest_value_of_sum_of_cubes_l1412_141289


namespace workout_days_l1412_141285

/-- Represents the number of squats performed on a given day -/
def squats_on_day (initial_squats : ℕ) (daily_increase : ℕ) (day : ℕ) : ℕ :=
  initial_squats + (day - 1) * daily_increase

/-- Represents the problem of determining the number of consecutive workout days -/
theorem workout_days (initial_squats : ℕ) (daily_increase : ℕ) (target_squats : ℕ) : 
  initial_squats = 30 → 
  daily_increase = 5 → 
  target_squats = 45 → 
  ∃ (n : ℕ), n = 4 ∧ squats_on_day initial_squats daily_increase n = target_squats :=
by
  sorry


end workout_days_l1412_141285


namespace farmers_wheat_cleaning_l1412_141239

theorem farmers_wheat_cleaning (original_rate : ℕ) (new_rate : ℕ) (last_day_acres : ℕ) :
  original_rate = 80 →
  new_rate = original_rate + 10 →
  last_day_acres = 30 →
  ∃ (total_acres : ℕ) (planned_days : ℕ),
    total_acres = 480 ∧
    planned_days * original_rate = total_acres ∧
    (planned_days - 1) * new_rate + last_day_acres = total_acres :=
by
  sorry

end farmers_wheat_cleaning_l1412_141239


namespace initial_men_count_is_eight_l1412_141204

/-- The number of men in the initial group -/
def initial_men_count : ℕ := sorry

/-- The increase in average age when two men are replaced -/
def average_age_increase : ℕ := 2

/-- The age of the first man being replaced -/
def first_replaced_man_age : ℕ := 21

/-- The age of the second man being replaced -/
def second_replaced_man_age : ℕ := 23

/-- The average age of the two new men -/
def new_men_average_age : ℕ := 30

theorem initial_men_count_is_eight :
  initial_men_count = 8 := by sorry

end initial_men_count_is_eight_l1412_141204


namespace expected_ones_three_dice_l1412_141230

-- Define a standard die
def standardDie : Finset Nat := Finset.range 6

-- Define the probability of rolling a 1 on a single die
def probOne : ℚ := 1 / 6

-- Define the probability of not rolling a 1 on a single die
def probNotOne : ℚ := 1 - probOne

-- Define the number of dice
def numDice : Nat := 3

-- Define the expected value function for discrete random variables
def expectedValue (outcomes : Finset Nat) (prob : Nat → ℚ) : ℚ :=
  Finset.sum outcomes (λ x => x * prob x)

-- Statement of the theorem
theorem expected_ones_three_dice :
  expectedValue (Finset.range (numDice + 1)) (λ k =>
    (numDice.choose k : ℚ) * probOne ^ k * probNotOne ^ (numDice - k)) = 1 / 2 := by
  sorry


end expected_ones_three_dice_l1412_141230


namespace intersection_points_count_l1412_141260

/-- Represents a line with a specific number of points -/
structure Line where
  numPoints : ℕ

/-- Represents a configuration of two parallel lines -/
structure ParallelLines where
  line1 : Line
  line2 : Line

/-- Calculates the number of intersection points for a given configuration of parallel lines -/
def intersectionPoints (pl : ParallelLines) : ℕ :=
  (pl.line1.numPoints.choose 2) * (pl.line2.numPoints.choose 2)

/-- The specific configuration of parallel lines in our problem -/
def problemConfig : ParallelLines :=
  { line1 := { numPoints := 10 }
    line2 := { numPoints := 11 } }

theorem intersection_points_count :
  intersectionPoints problemConfig = 2475 := by
  sorry

end intersection_points_count_l1412_141260


namespace apple_to_cucumber_ratio_l1412_141279

/-- Given the cost ratios of fruits, calculate the equivalent number of cucumbers for 20 apples -/
theorem apple_to_cucumber_ratio 
  (apple_banana_ratio : ℚ) 
  (banana_cucumber_ratio : ℚ) 
  (h1 : apple_banana_ratio = 10 / 5)  -- 10 apples = 5 bananas
  (h2 : banana_cucumber_ratio = 3 / 4)  -- 3 bananas = 4 cucumbers
  : (20 : ℚ) / apple_banana_ratio * banana_cucumber_ratio⁻¹ = 40 / 3 :=
by sorry

end apple_to_cucumber_ratio_l1412_141279


namespace intersection_implies_sum_l1412_141266

-- Define the functions
def f (x p q : ℝ) : ℝ := -|x - p| + q
def g (x r s : ℝ) : ℝ := |x - r| + s

-- State the theorem
theorem intersection_implies_sum (p q r s : ℝ) :
  (f 3 p q = g 3 r s) ∧ 
  (f 5 p q = g 5 r s) ∧ 
  (f 3 p q = 6) ∧ 
  (f 5 p q = 2) ∧ 
  (g 3 r s = 6) ∧ 
  (g 5 r s = 2) →
  p + r = 8 := by sorry

end intersection_implies_sum_l1412_141266


namespace sum_infinite_geometric_series_l1412_141207

def geometric_series (a : ℝ) (r : ℝ) := 
  fun n : ℕ => a * r ^ n

theorem sum_infinite_geometric_series :
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let series := geometric_series a r
  (∑' n, series n) = 3 / 2 := by sorry

end sum_infinite_geometric_series_l1412_141207


namespace geometric_sequence_property_l1412_141283

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_prod : a 3 * a 11 = 16) :
  a 7 = 4 := by
sorry

end geometric_sequence_property_l1412_141283


namespace product_of_areas_is_perfect_square_l1412_141220

/-- A convex quadrilateral divided by its diagonals -/
structure ConvexQuadrilateral where
  /-- The areas of the four triangles formed by the diagonals -/
  area₁ : ℤ
  area₂ : ℤ
  area₃ : ℤ
  area₄ : ℤ

/-- The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals is a perfect square -/
theorem product_of_areas_is_perfect_square (q : ConvexQuadrilateral) :
  ∃ (n : ℤ), q.area₁ * q.area₂ * q.area₃ * q.area₄ = n ^ 2 := by
  sorry


end product_of_areas_is_perfect_square_l1412_141220
