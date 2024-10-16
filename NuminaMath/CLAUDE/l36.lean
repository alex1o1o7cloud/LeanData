import Mathlib

namespace NUMINAMATH_CALUDE_henry_age_is_20_l36_3661

/-- Henry's present age -/
def henry_age : ℕ := sorry

/-- Jill's present age -/
def jill_age : ℕ := sorry

/-- The sum of Henry and Jill's present ages is 33 -/
axiom sum_of_ages : henry_age + jill_age = 33

/-- Six years ago, Henry was twice the age of Jill -/
axiom ages_relation : henry_age - 6 = 2 * (jill_age - 6)

/-- Henry's present age is 20 years -/
theorem henry_age_is_20 : henry_age = 20 := by sorry

end NUMINAMATH_CALUDE_henry_age_is_20_l36_3661


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l36_3633

/-- f(n) is the exponent of 2 in the prime factorization of n! -/
def f (n : ℕ+) : ℕ :=
  sorry

/-- For any positive integer a, there exist infinitely many positive integers n
    such that n - f(n) = a -/
theorem infinitely_many_solutions (a : ℕ+) :
  ∃ (S : Set ℕ+), Infinite S ∧ ∀ n ∈ S, n.val - f n = a.val := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l36_3633


namespace NUMINAMATH_CALUDE_power_sixteen_divided_by_eight_l36_3637

theorem power_sixteen_divided_by_eight (m : ℕ) : 
  m = 16^1000 → m / 8 = 2^3997 := by
  sorry

end NUMINAMATH_CALUDE_power_sixteen_divided_by_eight_l36_3637


namespace NUMINAMATH_CALUDE_complex_real_condition_l36_3607

theorem complex_real_condition (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  ((m^2 + Complex.I) * (1 - m * Complex.I)).im = 0 →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l36_3607


namespace NUMINAMATH_CALUDE_tanya_score_percentage_l36_3643

/-- Tanya's score on the math quiz -/
def score : ℚ := 20 / 25

/-- The percentage equivalent of Tanya's score -/
def percentage : ℚ := 80 / 100

theorem tanya_score_percentage : score = percentage := by sorry

end NUMINAMATH_CALUDE_tanya_score_percentage_l36_3643


namespace NUMINAMATH_CALUDE_three_numbers_sum_l36_3680

theorem three_numbers_sum (s : ℕ) :
  let A := Finset.range (4 * s) 
  ∀ (S : Finset ℕ), S ⊆ A → S.card = 2 * s + 2 →
    ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x + y = 2 * z :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l36_3680


namespace NUMINAMATH_CALUDE_profit_difference_is_640_l36_3691

/-- Calculates the difference between profit shares of two partners given their investments and the profit share of a third partner. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℕ) : ℕ :=
  let total_parts := invest_a + invest_b + invest_c
  let part_value := b_profit * total_parts / invest_b
  let a_profit := part_value * invest_a / total_parts
  let c_profit := part_value * invest_c / total_parts
  c_profit - a_profit

/-- Theorem stating that given the investments and b's profit share, the difference between a's and c's profit shares is 640. -/
theorem profit_difference_is_640 :
  profit_share_difference 8000 10000 12000 1600 = 640 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_is_640_l36_3691


namespace NUMINAMATH_CALUDE_log_sqrt8_512sqrt8_l36_3654

theorem log_sqrt8_512sqrt8 : Real.log (512 * Real.sqrt 8) / Real.log (Real.sqrt 8) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt8_512sqrt8_l36_3654


namespace NUMINAMATH_CALUDE_songs_ratio_l36_3673

def initial_songs : ℕ := 54
def deleted_songs : ℕ := 9

theorem songs_ratio :
  let kept_songs := initial_songs - deleted_songs
  let ratio := kept_songs / deleted_songs
  ratio = 5 := by sorry

end NUMINAMATH_CALUDE_songs_ratio_l36_3673


namespace NUMINAMATH_CALUDE_quadratic_solution_l36_3660

theorem quadratic_solution (a : ℝ) : (1 : ℝ)^2 + 1 + 2*a = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l36_3660


namespace NUMINAMATH_CALUDE_sum_angles_S_and_R_l36_3688

-- Define the circle and points
variable (circle : Type) (E F R G H : circle)

-- Define the measure of an arc
variable (arc_measure : circle → circle → ℝ)

-- Define the measure of an angle
variable (angle_measure : circle → ℝ)

-- State the theorem
theorem sum_angles_S_and_R (h1 : arc_measure F R = 60)
                           (h2 : arc_measure R G = 48) :
  angle_measure S + angle_measure R = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_angles_S_and_R_l36_3688


namespace NUMINAMATH_CALUDE_dodecagon_diagonal_intersection_probability_l36_3697

/-- A regular dodecagon -/
structure RegularDodecagon where
  -- Add any necessary properties here

/-- Represents a diagonal in the dodecagon -/
structure Diagonal where
  -- Add any necessary properties here

/-- The probability that two randomly chosen diagonals intersect inside a regular dodecagon -/
def intersection_probability (d : RegularDodecagon) : ℚ :=
  495 / 1431

/-- Theorem stating that the probability of two randomly chosen diagonals 
    intersecting inside a regular dodecagon is 495/1431 -/
theorem dodecagon_diagonal_intersection_probability (d : RegularDodecagon) :
  intersection_probability d = 495 / 1431 := by
  sorry


end NUMINAMATH_CALUDE_dodecagon_diagonal_intersection_probability_l36_3697


namespace NUMINAMATH_CALUDE_percent_of_x_is_z_l36_3652

theorem percent_of_x_is_z (x y z : ℝ) 
  (h1 : 0.45 * z = 0.96 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.6 * x := by
sorry

end NUMINAMATH_CALUDE_percent_of_x_is_z_l36_3652


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l36_3649

theorem sqrt_equation_solution : 
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l36_3649


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l36_3613

theorem geometric_arithmetic_sequence_sum : 
  ∃ (x y : ℝ), 3 < x ∧ x < y ∧ y < 9 ∧ 
  (x^2 = 3*y) ∧ (2*y = x + 9) ∧ 
  (x + y = 11.25) := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l36_3613


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_to_fourth_power_l36_3664

theorem sum_of_squared_differences_to_fourth_power :
  (6^2 - 3^2)^4 + (7^2 - 2^2)^4 = 4632066 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_differences_to_fourth_power_l36_3664


namespace NUMINAMATH_CALUDE_power_of_two_representation_l36_3619

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), Odd x ∧ Odd y ∧ 2^n = 7*x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_representation_l36_3619


namespace NUMINAMATH_CALUDE_canoe_water_removal_rate_l36_3685

theorem canoe_water_removal_rate 
  (distance : ℝ) 
  (paddling_speed : ℝ) 
  (water_intake_rate : ℝ) 
  (sinking_threshold : ℝ) 
  (h1 : distance = 2) 
  (h2 : paddling_speed = 3) 
  (h3 : water_intake_rate = 8) 
  (h4 : sinking_threshold = 40) : 
  ∃ (min_removal_rate : ℝ), 
    min_removal_rate = 7 ∧ 
    ∀ (removal_rate : ℝ), 
      removal_rate ≥ min_removal_rate → 
      (water_intake_rate - removal_rate) * (distance / paddling_speed * 60) ≤ sinking_threshold :=
by sorry

end NUMINAMATH_CALUDE_canoe_water_removal_rate_l36_3685


namespace NUMINAMATH_CALUDE_mallory_journey_expenses_l36_3678

theorem mallory_journey_expenses :
  let initial_fuel_cost : ℚ := 45
  let miles_per_tank : ℚ := 500
  let total_miles : ℚ := 2000
  let food_cost_ratio : ℚ := 3/5
  let hotel_nights : ℕ := 3
  let hotel_cost_per_night : ℚ := 80
  let fuel_cost_increase : ℚ := 5

  let num_refills : ℕ := (total_miles / miles_per_tank).ceil.toNat
  let fuel_costs : List ℚ := List.range num_refills |>.map (λ i => initial_fuel_cost + i * fuel_cost_increase)
  let total_fuel_cost : ℚ := fuel_costs.sum
  let food_cost : ℚ := food_cost_ratio * total_fuel_cost
  let hotel_cost : ℚ := hotel_nights * hotel_cost_per_night
  let total_expenses : ℚ := total_fuel_cost + food_cost + hotel_cost

  total_expenses = 576
  := by sorry

end NUMINAMATH_CALUDE_mallory_journey_expenses_l36_3678


namespace NUMINAMATH_CALUDE_rectangle_area_l36_3632

/-- The area of a rectangle with diagonal x and length three times its width -/
theorem rectangle_area (x : ℝ) : 
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w ^ 2 + l ^ 2 = x ^ 2 → w * l = (3 / 10) * x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l36_3632


namespace NUMINAMATH_CALUDE_complex_numbers_count_is_25_l36_3663

def S : Finset ℕ := {0, 1, 2, 3, 4, 5}

def complex_numbers_count : ℕ :=
  (S.filter (λ b => b ≠ 0)).card * (S.card - 1)

theorem complex_numbers_count_is_25 : complex_numbers_count = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_numbers_count_is_25_l36_3663


namespace NUMINAMATH_CALUDE_negative_two_times_inequality_l36_3681

theorem negative_two_times_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_times_inequality_l36_3681


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l36_3671

theorem largest_angle_in_pentagon (P Q R S T : ℝ) : 
  P = 70 → 
  Q = 100 → 
  R = S → 
  T = 3 * R - 25 → 
  P + Q + R + S + T = 540 → 
  max P (max Q (max R (max S T))) = 212 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l36_3671


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_count_six_digit_numbers_with_zero_l36_3656

theorem six_digit_numbers_with_zero (total : ℕ) (no_zero : ℕ) : ℕ :=
  total - no_zero

theorem count_six_digit_numbers_with_zero :
  six_digit_numbers_with_zero 900000 531441 = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_count_six_digit_numbers_with_zero_l36_3656


namespace NUMINAMATH_CALUDE_truncated_cone_angle_l36_3675

theorem truncated_cone_angle (R : ℝ) (h : ℝ) (r : ℝ) : 
  h = R → 
  (12 * r) / Real.sqrt 3 = 3 * R * Real.sqrt 3 → 
  Real.arctan (h / (R - r)) = Real.arctan 4 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cone_angle_l36_3675


namespace NUMINAMATH_CALUDE_sqrt_6_irrational_l36_3672

theorem sqrt_6_irrational : Irrational (Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_irrational_l36_3672


namespace NUMINAMATH_CALUDE_thermostat_problem_l36_3639

theorem thermostat_problem (initial_temp : ℝ) (final_temp : ℝ) (x : ℝ) 
  (h1 : initial_temp = 40)
  (h2 : final_temp = 59) : 
  (((initial_temp * 2 - 30) * 0.7) + x = final_temp) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_thermostat_problem_l36_3639


namespace NUMINAMATH_CALUDE_orvin_max_balloons_l36_3653

/-- Represents the price of a balloon in cents -/
def regularPrice : ℕ := 200

/-- Represents the number of balloons Orvin can afford at regular price -/
def regularAffordable : ℕ := 40

/-- Represents the maximum number of discounted balloons -/
def maxDiscounted : ℕ := 10

/-- Calculates the total money Orvin has in cents -/
def totalMoney : ℕ := regularPrice * regularAffordable

/-- Calculates the price of a discounted balloon in cents -/
def discountedPrice : ℕ := regularPrice / 2

/-- Calculates the cost of buying a regular and a discounted balloon in cents -/
def pairCost : ℕ := regularPrice + discountedPrice

/-- Represents the maximum number of balloons Orvin can buy -/
def maxBalloons : ℕ := 42

theorem orvin_max_balloons :
  regularPrice > 0 →
  (totalMoney - (maxDiscounted / 2 * pairCost)) / regularPrice + maxDiscounted = maxBalloons :=
by sorry

end NUMINAMATH_CALUDE_orvin_max_balloons_l36_3653


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l36_3657

/-- Given a rectangle with perimeter 100 meters and length-to-width ratio 5:2,
    prove that its diagonal length is (5 * sqrt 290) / 7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 100) →  -- Perimeter condition
  (length / width = 5 / 2) →      -- Ratio condition
  Real.sqrt (length^2 + width^2) = (5 * Real.sqrt 290) / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l36_3657


namespace NUMINAMATH_CALUDE_cos_2017pi_over_6_l36_3630

theorem cos_2017pi_over_6 : Real.cos (2017 * Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2017pi_over_6_l36_3630


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l36_3651

/-- Proves that given a journey of 1.5 km, if a person arrives 7 minutes late when traveling
    at speed v km/hr, and arrives 8 minutes early when traveling at 6 km/hr, then v = 10 km/hr. -/
theorem journey_speed_calculation (v : ℝ) : 
  (∃ t : ℝ, 
    1.5 = v * (t - 7/60) ∧ 
    1.5 = 6 * (t - 8/60)) → 
  v = 10 := by
sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l36_3651


namespace NUMINAMATH_CALUDE_construction_delay_construction_delay_days_l36_3624

/-- Represents the total work in man-days -/
def total_work (men : ℕ) (days : ℕ) : ℕ := men * days

/-- Represents the construction scenario -/
structure Construction :=
  (initial_men : ℕ)
  (additional_men : ℕ)
  (initial_days : ℕ)
  (total_days : ℕ)

/-- Theorem stating the relationship between different work scenarios -/
theorem construction_delay (c : Construction) 
  (h1 : c.initial_men = 100)
  (h2 : c.additional_men = 100)
  (h3 : c.initial_days = 50)
  (h4 : c.total_days = 100) :
  total_work c.initial_men c.total_days = 
  total_work (c.initial_men + c.additional_men) (c.total_days - c.initial_days) + 
  total_work c.initial_men c.initial_days :=
sorry

/-- Theorem proving the delay in construction -/
theorem construction_delay_days (c : Construction) 
  (h1 : c.initial_men = 100)
  (h2 : c.additional_men = 100)
  (h3 : c.initial_days = 50)
  (h4 : c.total_days = 100) :
  (total_work c.initial_men c.total_days) / c.initial_men = 150 :=
sorry

end NUMINAMATH_CALUDE_construction_delay_construction_delay_days_l36_3624


namespace NUMINAMATH_CALUDE_inequality_solution_set_l36_3682

theorem inequality_solution_set :
  {x : ℝ | x * (x - 1) ≥ x} = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l36_3682


namespace NUMINAMATH_CALUDE_divisor_property_l36_3683

theorem divisor_property (k : ℕ) (h : 5^k - k^5 = 1) : 15^k = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_l36_3683


namespace NUMINAMATH_CALUDE_trapezoid_properties_l36_3600

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  side1 : ℝ
  side2 : ℝ
  diagonal_is_bisector : Bool

/-- Properties of the specific trapezoid in the problem -/
def problem_trapezoid : IsoscelesTrapezoid :=
  { side1 := 6
  , side2 := 6.25
  , diagonal_is_bisector := true }

/-- The length of the diagonal from the acute angle vertex -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ := sorry

/-- The area of the trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating the properties of the specific trapezoid -/
theorem trapezoid_properties :
  let t := problem_trapezoid
  abs (diagonal_length t - 10.423) < 0.001 ∧
  abs (trapezoid_area t - 32) < 0.001 := by sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l36_3600


namespace NUMINAMATH_CALUDE_tangent_perpendicular_point_l36_3694

theorem tangent_perpendicular_point (x y : ℝ) : 
  y = 1 / x →  -- P is on the curve y = 1/x
  ((-1 / x^2) * (1 / 4) = -1) →  -- Tangent line is perpendicular to x - 4y - 8 = 0
  ((x = -1/2 ∧ y = -2) ∨ (x = 1/2 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_point_l36_3694


namespace NUMINAMATH_CALUDE_binomial_10_3_l36_3612

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l36_3612


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l36_3626

theorem sum_of_squares_and_products (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + c^2 = 58) (h5 : a*b + b*c + c*a = 32) :
  a + b + c = Real.sqrt 122 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l36_3626


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l36_3603

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 3 → (42 * x - 37) / (x^2 - 4*x + 3) = N₁ / (x - 1) + N₂ / (x - 3)) →
  N₁ * N₂ = -445/4 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l36_3603


namespace NUMINAMATH_CALUDE_negation_false_implies_proposition_true_l36_3621

theorem negation_false_implies_proposition_true (P : Prop) : 
  ¬(¬P) → P :=
sorry

end NUMINAMATH_CALUDE_negation_false_implies_proposition_true_l36_3621


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_positive_m_for_unique_solution_l36_3658

theorem unique_quadratic_solution (m : ℝ) :
  (∃! x : ℝ, 16 * x^2 + m * x + 4 = 0) ↔ m = 16 ∨ m = -16 :=
by sorry

theorem positive_m_for_unique_solution :
  ∃ m : ℝ, m > 0 ∧ (∃! x : ℝ, 16 * x^2 + m * x + 4 = 0) ∧ m = 16 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_positive_m_for_unique_solution_l36_3658


namespace NUMINAMATH_CALUDE_cone_surface_area_special_case_l36_3611

/-- Represents a cone with given slant height and lateral surface property -/
structure Cone where
  slant_height : ℝ
  lateral_surface_semicircle : Prop

/-- Calculates the surface area of a cone -/
def surface_area (c : Cone) : ℝ :=
  sorry -- Definition to be implemented

/-- Theorem: The surface area of a cone with slant height 2 and lateral surface
    that unfolds into a semicircle is 3π -/
theorem cone_surface_area_special_case :
  ∀ (c : Cone), c.slant_height = 2 → c.lateral_surface_semicircle →
  surface_area c = 3 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_cone_surface_area_special_case_l36_3611


namespace NUMINAMATH_CALUDE_cos_neg_sixty_degrees_l36_3648

theorem cos_neg_sixty_degrees : Real.cos (-(60 * π / 180)) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_neg_sixty_degrees_l36_3648


namespace NUMINAMATH_CALUDE_quadratic_statements_l36_3623

variable (a b c : ℝ)
variable (x₀ : ℝ)

def quadratic_equation (x : ℝ) := a * x^2 + b * x + c

theorem quadratic_statements (h : a ≠ 0) :
  (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
  ((∃ x y, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) →
    ∃ u v, u ≠ v ∧ quadratic_equation u = 0 ∧ quadratic_equation v = 0) ∧
  (quadratic_equation x₀ = 0 → b^2 - 4*a*c = (2*a*x₀ + b)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_statements_l36_3623


namespace NUMINAMATH_CALUDE_two_digit_number_relationship_l36_3696

theorem two_digit_number_relationship :
  ∀ (tens units : ℕ),
    tens * 10 + units = 16 →
    tens + units = 7 →
    ∃ (k : ℕ), units = k * tens →
    units = 6 * tens :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_relationship_l36_3696


namespace NUMINAMATH_CALUDE_running_is_experimental_l36_3640

/-- Represents an investigation method -/
inductive InvestigationMethod
  | Experimental
  | NonExperimental

/-- Represents the characteristics of an investigation -/
structure Investigation where
  description : String
  quantitative : Bool
  directlyMeasurable : Bool
  controlledSetting : Bool

/-- Determines if an investigation is suitable for the experimental method -/
def isSuitableForExperiment (i : Investigation) : InvestigationMethod :=
  if i.quantitative && i.directlyMeasurable && i.controlledSetting then
    InvestigationMethod.Experimental
  else
    InvestigationMethod.NonExperimental

/-- The investigation of running distance in 10 seconds -/
def runningInvestigation : Investigation where
  description := "How many meters you can run in 10 seconds"
  quantitative := true
  directlyMeasurable := true
  controlledSetting := true

/-- Theorem stating that the running investigation is suitable for the experimental method -/
theorem running_is_experimental :
  isSuitableForExperiment runningInvestigation = InvestigationMethod.Experimental := by
  sorry


end NUMINAMATH_CALUDE_running_is_experimental_l36_3640


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_five_l36_3605

/-- Given two vectors a and b in ℝ², prove that if a is perpendicular to b, then m = 5 -/
theorem perpendicular_vectors_m_equals_five :
  let a : ℝ × ℝ := (1, -1)
  let b : ℝ × ℝ := (m + 1, 2 * m - 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_five_l36_3605


namespace NUMINAMATH_CALUDE_article_sale_price_l36_3608

/-- Given an article with unknown cost price, prove that the selling price
    incurring a loss equal to the profit at $852 is $448, given that the
    selling price for a 50% profit is $975. -/
theorem article_sale_price (cost : ℝ) (loss_price : ℝ) : 
  (852 - cost = cost - loss_price) →  -- Profit at $852 equals loss at loss_price
  (cost + 0.5 * cost = 975) →         -- 50% profit price is $975
  loss_price = 448 := by
  sorry

end NUMINAMATH_CALUDE_article_sale_price_l36_3608


namespace NUMINAMATH_CALUDE_division_problem_l36_3655

theorem division_problem (divisor : ℕ) : 
  (15 / divisor = 4) ∧ (15 % divisor = 3) → divisor = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l36_3655


namespace NUMINAMATH_CALUDE_no_four_integers_product_square_l36_3628

theorem no_four_integers_product_square : ¬∃ (a b c d : ℕ+), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (∃ (m : ℕ), (a * b + 2006 : ℕ) = m^2) ∧
  (∃ (n : ℕ), (a * c + 2006 : ℕ) = n^2) ∧
  (∃ (p : ℕ), (a * d + 2006 : ℕ) = p^2) ∧
  (∃ (q : ℕ), (b * c + 2006 : ℕ) = q^2) ∧
  (∃ (r : ℕ), (b * d + 2006 : ℕ) = r^2) ∧
  (∃ (s : ℕ), (c * d + 2006 : ℕ) = s^2) :=
by
  sorry

end NUMINAMATH_CALUDE_no_four_integers_product_square_l36_3628


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l36_3690

theorem sum_of_reciprocals_of_roots (p q : ℝ) : 
  p^2 - 16*p + 9 = 0 → 
  q^2 - 16*q + 9 = 0 → 
  p ≠ q → 
  1/p + 1/q = 16/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l36_3690


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l36_3609

/-- An arithmetic sequence with a_5 = 5a_3 has S_9/S_5 = 9 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- sum formula
  a 5 = 5 * a 3 →  -- given condition
  S 9 / S 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l36_3609


namespace NUMINAMATH_CALUDE_least_sum_of_four_primes_l36_3646

-- Define a function that checks if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function that represents the sum of 4 different primes greater than 10
def sumOfFourPrimes (a b c d : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧
  a > 10 ∧ b > 10 ∧ c > 10 ∧ d > 10 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Theorem statement
theorem least_sum_of_four_primes :
  ∀ n : ℕ, (∃ a b c d : ℕ, sumOfFourPrimes a b c d ∧ a + b + c + d = n) →
  n ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_four_primes_l36_3646


namespace NUMINAMATH_CALUDE_tshirt_price_correct_l36_3689

/-- The regular price of a T-shirt -/
def regular_price : ℝ := 14.5

/-- The total number of T-shirts purchased -/
def total_shirts : ℕ := 12

/-- The total cost of the purchase -/
def total_cost : ℝ := 120

/-- The cost of a group of three T-shirts (two at regular price, one at $1) -/
def group_cost (price : ℝ) : ℝ := 2 * price + 1

/-- The number of groups of three T-shirts -/
def num_groups : ℕ := total_shirts / 3

theorem tshirt_price_correct :
  group_cost regular_price * num_groups = total_cost ∧
  regular_price > 0 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_price_correct_l36_3689


namespace NUMINAMATH_CALUDE_journal_involvement_l36_3644

theorem journal_involvement (total_students : ℕ) 
  (total_percentage : ℚ) (boys_percentage : ℚ) (girls_percentage : ℚ)
  (h1 : total_students = 75000)
  (h2 : total_percentage = 5 / 300)  -- 1 2/3% as a fraction
  (h3 : boys_percentage = 7 / 300)   -- 2 1/3% as a fraction
  (h4 : girls_percentage = 2 / 300)  -- 2/3% as a fraction
  : ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    ↑boys * boys_percentage + ↑girls * girls_percentage = ↑total_students * total_percentage ∧
    boys * boys_percentage = 700 ∧
    girls * girls_percentage = 300 :=
sorry

end NUMINAMATH_CALUDE_journal_involvement_l36_3644


namespace NUMINAMATH_CALUDE_bus_system_stops_l36_3659

/-- Represents a bus system in a city -/
structure BusSystem where
  num_routes : ℕ
  stops_per_route : ℕ
  travel_without_transfer : Prop
  unique_intersection : Prop
  min_stops : Prop

/-- Theorem: In a bus system with 57 routes, where you can travel between any two stops without transferring,
    each pair of routes intersects at exactly one stop, and each route has at least three stops,
    the number of stops on each route is 40. -/
theorem bus_system_stops (bs : BusSystem) 
  (h1 : bs.num_routes = 57)
  (h2 : bs.travel_without_transfer)
  (h3 : bs.unique_intersection)
  (h4 : bs.min_stops)
  : bs.stops_per_route = 40 := by
  sorry

end NUMINAMATH_CALUDE_bus_system_stops_l36_3659


namespace NUMINAMATH_CALUDE_sphere_plane_distance_l36_3635

/-- Given a sphere and a plane cutting it, this theorem relates the radius of the sphere,
    the radius of the circular section, and the distance from the sphere's center to the plane. -/
theorem sphere_plane_distance (R r d : ℝ) : R = 2 * Real.sqrt 3 → r = 2 → d ^ 2 + r ^ 2 = R ^ 2 → d = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_plane_distance_l36_3635


namespace NUMINAMATH_CALUDE_hyperbola_properties_l36_3634

/-- The original hyperbola equation -/
def original_hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

/-- The new hyperbola equation -/
def new_hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 12 = 1

/-- Definition of asymptotes for a hyperbola -/
def has_same_asymptotes (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    ∀ (x y : ℝ), (f x y ↔ g (k * x) (k * y))

/-- The main theorem to prove -/
theorem hyperbola_properties :
  has_same_asymptotes original_hyperbola new_hyperbola ∧
  new_hyperbola 2 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l36_3634


namespace NUMINAMATH_CALUDE_rhind_papyrus_fraction_decomposition_l36_3620

theorem rhind_papyrus_fraction_decomposition : 
  2 / 73 = 1 / 60 + 1 / 219 + 1 / 292 + 1 / 365 := by
  sorry

end NUMINAMATH_CALUDE_rhind_papyrus_fraction_decomposition_l36_3620


namespace NUMINAMATH_CALUDE_min_value_quadratic_l36_3692

theorem min_value_quadratic (x y : ℝ) :
  y = x^2 + 12*x + 5 →
  ∀ z : ℝ, y ≥ -31 ∧ (∃ w : ℝ, w^2 + 12*w + 5 = -31) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l36_3692


namespace NUMINAMATH_CALUDE_garden_ant_count_l36_3693

/-- Represents the dimensions and ant density of a rectangular garden --/
structure Garden where
  width : ℝ  -- width in feet
  length : ℝ  -- length in feet
  antDensity : ℝ  -- ants per square inch

/-- Conversion factor from feet to inches --/
def feetToInches : ℝ := 12

/-- Calculates the approximate number of ants in the garden --/
def approximateAntCount (g : Garden) : ℝ :=
  g.width * feetToInches * g.length * feetToInches * g.antDensity

/-- Theorem stating that the number of ants in the given garden is approximately 30 million --/
theorem garden_ant_count :
  let g : Garden := { width := 350, length := 300, antDensity := 2 }
  ∃ ε > 0, |approximateAntCount g - 30000000| < ε := by
  sorry

end NUMINAMATH_CALUDE_garden_ant_count_l36_3693


namespace NUMINAMATH_CALUDE_fixed_point_on_line_fixed_point_coordinates_l36_3674

/-- The fixed point through which the line ax + y + 1 = 0 always passes -/
def fixed_point : ℝ × ℝ := sorry

/-- The line equation ax + y + 1 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y + 1 = 0

/-- The theorem stating that the fixed point satisfies the line equation for all values of a -/
theorem fixed_point_on_line : ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) := sorry

/-- The theorem proving that the fixed point is (0, -1) -/
theorem fixed_point_coordinates : fixed_point = (0, -1) := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_fixed_point_coordinates_l36_3674


namespace NUMINAMATH_CALUDE_angle_terminal_side_trig_sum_l36_3616

theorem angle_terminal_side_trig_sum (α : Real) :
  (∃ (P : Real × Real), P = (-4/5, 3/5) ∧ P.1 = -4/5 ∧ P.2 = 3/5 ∧ 
   P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  2 * Real.sin α + Real.cos α = 2/5 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_trig_sum_l36_3616


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l36_3606

/-- The probability of selecting either a blue or purple jelly bean from a bag -/
theorem jelly_bean_probability :
  let red : ℕ := 8
  let green : ℕ := 9
  let yellow : ℕ := 10
  let blue : ℕ := 12
  let purple : ℕ := 5
  let total : ℕ := red + green + yellow + blue + purple
  let blue_or_purple : ℕ := blue + purple
  (blue_or_purple : ℚ) / total = 17 / 44 :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l36_3606


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l36_3602

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (1 + 4 * i) = -6 / 17 - (10 / 17) * i :=
by
  sorry


end NUMINAMATH_CALUDE_complex_fraction_simplification_l36_3602


namespace NUMINAMATH_CALUDE_problem_solution_l36_3601

theorem problem_solution (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 2) 
  (h3 : a^2 + b^2 + c^2 = 3) : 
  1 / (a * b + c - 1) + 1 / (b * c + a - 1) + 1 / (c * a + b - 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l36_3601


namespace NUMINAMATH_CALUDE_g_2022_l36_3627

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    g(x - y) = 2022 * (g x + g y) - 2021 * x * y for all real x and y,
    prove that g(2022) = 2043231 -/
theorem g_2022 (g : ℝ → ℝ) 
    (h : ∀ x y : ℝ, g (x - y) = 2022 * (g x + g y) - 2021 * x * y) : 
  g 2022 = 2043231 := by
  sorry

end NUMINAMATH_CALUDE_g_2022_l36_3627


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l36_3665

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l36_3665


namespace NUMINAMATH_CALUDE_increasing_interval_of_f_l36_3618

-- Define the function
def f (x : ℝ) : ℝ := 3*x - x^3

-- State the theorem
theorem increasing_interval_of_f :
  ∀ x : ℝ, (∀ y : ℝ, x < y → f x < f y) ↔ x ∈ Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_increasing_interval_of_f_l36_3618


namespace NUMINAMATH_CALUDE_triangle_side_length_l36_3642

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : B = 45 * π / 180)
  (h2 : C = 60 * π / 180)
  (h3 : c = 1)
  (h4 : a + b + c = A + B + C)
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h6 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h7 : A + B + C = π) : 
  b = Real.sqrt 6 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l36_3642


namespace NUMINAMATH_CALUDE_sin_double_angle_tangent_two_l36_3650

theorem sin_double_angle_tangent_two (α : Real) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_tangent_two_l36_3650


namespace NUMINAMATH_CALUDE_marnie_bracelets_l36_3667

/-- The number of bags of 50 beads Marnie bought -/
def bags_50 : ℕ := 5

/-- The number of bags of 100 beads Marnie bought -/
def bags_100 : ℕ := 2

/-- The number of beads in each bag of 50 -/
def beads_per_bag_50 : ℕ := 50

/-- The number of beads in each bag of 100 -/
def beads_per_bag_100 : ℕ := 100

/-- The number of beads needed to make one bracelet -/
def beads_per_bracelet : ℕ := 50

/-- The total number of beads Marnie bought -/
def total_beads : ℕ := bags_50 * beads_per_bag_50 + bags_100 * beads_per_bag_100

/-- The number of bracelets Marnie can make -/
def bracelets : ℕ := total_beads / beads_per_bracelet

theorem marnie_bracelets : bracelets = 9 := by sorry

end NUMINAMATH_CALUDE_marnie_bracelets_l36_3667


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l36_3686

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 9 = 10 →
  a 5 = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l36_3686


namespace NUMINAMATH_CALUDE_inequality_solution_range_l36_3677

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x - 4

-- State the theorem
theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ f x > a) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l36_3677


namespace NUMINAMATH_CALUDE_cassie_parrot_count_l36_3668

/-- Represents the number of nails Cassie needs to cut for her pets -/
def total_nails : ℕ := 113

/-- Represents the number of dogs Cassie has -/
def num_dogs : ℕ := 4

/-- Represents the number of nails each dog has -/
def nails_per_dog : ℕ := 16

/-- Represents the number of claws each regular parrot has -/
def claws_per_parrot : ℕ := 6

/-- Represents the number of claws the special parrot with an extra toe has -/
def claws_special_parrot : ℕ := 7

/-- Theorem stating that the number of parrots Cassie has is 8 -/
theorem cassie_parrot_count : 
  ∃ (num_parrots : ℕ), 
    num_parrots * claws_per_parrot + 
    (claws_special_parrot - claws_per_parrot) + 
    (num_dogs * nails_per_dog) = total_nails ∧ 
    num_parrots = 8 := by
  sorry

end NUMINAMATH_CALUDE_cassie_parrot_count_l36_3668


namespace NUMINAMATH_CALUDE_inequality_solution_set_l36_3666

theorem inequality_solution_set :
  {x : ℝ | (1/2: ℝ)^x ≤ (1/2 : ℝ)^(x+1) + 1} = {x : ℝ | x ≥ -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l36_3666


namespace NUMINAMATH_CALUDE_correct_rainwater_collection_l36_3662

/-- Represents the water collection problem --/
structure WaterCollection where
  tankCapacity : ℕ        -- Tank capacity in liters
  riverWater : ℕ          -- Water collected from river daily in milliliters
  daysToFill : ℕ          -- Number of days to fill the tank
  rainWater : ℕ           -- Water collected from rain daily in milliliters

/-- Theorem stating the correct amount of rainwater collected daily --/
theorem correct_rainwater_collection (w : WaterCollection) 
  (h1 : w.tankCapacity = 50)
  (h2 : w.riverWater = 1700)
  (h3 : w.daysToFill = 20)
  : w.rainWater = 800 := by
  sorry

#check correct_rainwater_collection

end NUMINAMATH_CALUDE_correct_rainwater_collection_l36_3662


namespace NUMINAMATH_CALUDE_triangle_point_C_l36_3629

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (M : Point2D) (A : Point2D) (B : Point2D) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

/-- Check if a point is on the angle bisector of a triangle -/
def isOnAngleBisector (L : Point2D) (T : Triangle) : Prop :=
  -- This is a simplified condition for being on the angle bisector
  -- In reality, this would involve more complex geometric calculations
  true

theorem triangle_point_C (A M L : Point2D) (h1 : A.x = 2 ∧ A.y = 8)
    (h2 : M.x = 4 ∧ M.y = 11) (h3 : L.x = 6 ∧ L.y = 6) :
    ∃ (T : Triangle), T.A = A ∧ isMidpoint M A T.B ∧ isOnAngleBisector L T ∧ T.C.x = 14 ∧ T.C.y = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_point_C_l36_3629


namespace NUMINAMATH_CALUDE_village_households_l36_3645

/-- The number of households in a village given water consumption data. -/
theorem village_households (water_per_household : ℕ) (total_water : ℕ) 
  (h1 : water_per_household = 200)
  (h2 : total_water = 2000)
  (h3 : total_water = water_per_household * (total_water / water_per_household)) :
  total_water / water_per_household = 10 := by
  sorry

#check village_households

end NUMINAMATH_CALUDE_village_households_l36_3645


namespace NUMINAMATH_CALUDE_melanie_picked_zero_pears_l36_3676

/-- The number of pears Melanie picked -/
def melanie_pears : ℕ := 0

/-- The number of plums Alyssa picked -/
def alyssa_plums : ℕ := 17

/-- The number of plums Jason picked -/
def jason_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := 27

theorem melanie_picked_zero_pears :
  alyssa_plums + jason_plums = total_plums → melanie_pears = 0 := by
  sorry

end NUMINAMATH_CALUDE_melanie_picked_zero_pears_l36_3676


namespace NUMINAMATH_CALUDE_rational_equation_sum_l36_3638

theorem rational_equation_sum (A B : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 5 →
    (B * x - 11) / (x^2 - 7*x + 10) = A / (x - 2) + 3 / (x - 5)) →
  A + B = 5 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_sum_l36_3638


namespace NUMINAMATH_CALUDE_coin_toss_and_match_probability_l36_3614

/-- Represents the outcome of a coin toss -/
inductive CoinToss
| Head
| Tail

/-- Represents the weather condition during a match -/
inductive Weather
| Rainy
| NotRainy

/-- Represents the result of a match -/
inductive MatchResult
| Draw
| NotDraw

/-- Represents a football match with its associated coin toss, weather, and result -/
structure Match where
  toss : CoinToss
  weather : Weather
  result : MatchResult

def coin_tosses : ℕ := 25
def heads_count : ℕ := 11
def draw_on_heads : ℕ := 7
def rainy_on_tails : ℕ := 4

/-- The main theorem to prove -/
theorem coin_toss_and_match_probability :
  (coin_tosses - heads_count = 14) ∧
  (∀ m : Match, m.toss = CoinToss.Head → m.result = MatchResult.Draw → 
               m.toss = CoinToss.Tail → m.weather = Weather.Rainy → False) :=
by sorry

end NUMINAMATH_CALUDE_coin_toss_and_match_probability_l36_3614


namespace NUMINAMATH_CALUDE_alicia_remaining_masks_l36_3604

/-- The number of mask sets remaining in Alicia's collection after donation -/
def remaining_masks (initial : ℕ) (donated : ℕ) : ℕ :=
  initial - donated

/-- Theorem stating that Alicia has 39 mask sets left after donating to the museum -/
theorem alicia_remaining_masks :
  remaining_masks 90 51 = 39 := by
  sorry

end NUMINAMATH_CALUDE_alicia_remaining_masks_l36_3604


namespace NUMINAMATH_CALUDE_no_prime_divisor_8t_plus_5_l36_3617

theorem no_prime_divisor_8t_plus_5 (x : ℕ+) :
  ∀ p : ℕ, Prime p → p % 8 = 5 →
    ¬(p ∣ (8 * x^4 - 2)) ∧
    ¬(p ∣ (8 * x^4 - 1)) ∧
    ¬(p ∣ (8 * x^4)) ∧
    ¬(p ∣ (8 * x^4 + 1)) :=
sorry

end NUMINAMATH_CALUDE_no_prime_divisor_8t_plus_5_l36_3617


namespace NUMINAMATH_CALUDE_pizza_toppings_l36_3699

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (h_total : total_slices = 24)
  (h_pep : pepperoni_slices = 15)
  (h_mush : mushroom_slices = 16)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l36_3699


namespace NUMINAMATH_CALUDE_train_speed_problem_l36_3631

theorem train_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_increase : ℝ) 
  (h1 : distance = 600)
  (h2 : time_diff = 4)
  (h3 : speed_increase = 12) :
  ∃ (normal_speed : ℝ),
    normal_speed > 0 ∧
    (distance / normal_speed) - (distance / (normal_speed + speed_increase)) = time_diff ∧
    normal_speed = 37 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l36_3631


namespace NUMINAMATH_CALUDE_square_diff_product_l36_3687

theorem square_diff_product (x y : ℝ) (hx : x = Real.sqrt 3 + 1) (hy : y = Real.sqrt 3 - 1) :
  x^2 * y - x * y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_product_l36_3687


namespace NUMINAMATH_CALUDE_max_value_ab_l36_3684

theorem max_value_ab (a b : ℕ) : 
  a > 1 → b > 1 → a^b * b^a + a^b + b^a = 5329 → a^b ≤ 64 := by
sorry

end NUMINAMATH_CALUDE_max_value_ab_l36_3684


namespace NUMINAMATH_CALUDE_ant_path_problem_l36_3669

/-- Represents the ant's path in the rectangle -/
structure AntPath where
  rectangle_width : ℝ
  rectangle_height : ℝ
  start_point : ℝ
  path_angle : ℝ

/-- The problem statement -/
theorem ant_path_problem (path : AntPath) :
  path.rectangle_width = 150 ∧
  path.rectangle_height = 18 ∧
  path.path_angle = π / 4 ∧
  path.start_point ≥ 0 ∧
  path.start_point ≤ path.rectangle_height ∧
  (∃ (n : ℕ), 
    path.start_point + n * path.rectangle_height - 2 * n * path.start_point = path.rectangle_width / 2) →
  min path.start_point (path.rectangle_height - path.start_point) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ant_path_problem_l36_3669


namespace NUMINAMATH_CALUDE_pages_per_day_to_finish_on_time_l36_3622

/-- Given a 66-page paper due in 6 days, prove that 11 pages per day are required to finish on time. -/
theorem pages_per_day_to_finish_on_time :
  let total_pages : ℕ := 66
  let days_until_due : ℕ := 6
  let pages_per_day : ℕ := total_pages / days_until_due
  pages_per_day = 11 := by sorry

end NUMINAMATH_CALUDE_pages_per_day_to_finish_on_time_l36_3622


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l36_3698

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = (x+1)(x-a) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x + 1) * (x - a)

/-- If f(x) = (x+1)(x-a) is an even function, then a = 1 -/
theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l36_3698


namespace NUMINAMATH_CALUDE_container_volume_maximized_l36_3641

/-- The total length of the steel bar used to make the container frame -/
def total_length : ℝ := 14.8

/-- The function representing the volume of the container -/
def volume (width : ℝ) : ℝ :=
  width * (width + 0.5) * (3.2 - 2 * width)

/-- The width that maximizes the container's volume -/
def optimal_width : ℝ := 1

theorem container_volume_maximized :
  ∀ w : ℝ, 0 < w → w < 1.6 → volume w ≤ volume optimal_width :=
sorry

end NUMINAMATH_CALUDE_container_volume_maximized_l36_3641


namespace NUMINAMATH_CALUDE_sales_solution_l36_3695

def sales_problem (s1 s2 s3 s4 s6 : ℕ) (average : ℕ) : Prop :=
  let total := average * 6
  let known_sum := s1 + s2 + s3 + s4 + s6
  let s5 := total - known_sum
  s5 = 6562

theorem sales_solution (s1 s2 s3 s4 s6 average : ℕ) 
  (h1 : s1 = 6435) (h2 : s2 = 6927) (h3 : s3 = 6855) (h4 : s4 = 7230) 
  (h5 : s6 = 6791) (h6 : average = 6800) :
  sales_problem s1 s2 s3 s4 s6 average :=
by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l36_3695


namespace NUMINAMATH_CALUDE_complex_equation_solution_l36_3647

theorem complex_equation_solution (z : ℂ) : (z - 2*Complex.I) * (2 - Complex.I) = 5 → z = 2 + 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l36_3647


namespace NUMINAMATH_CALUDE_fraction_product_equality_l36_3679

theorem fraction_product_equality : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 = 24 / 77 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l36_3679


namespace NUMINAMATH_CALUDE_tires_in_parking_lot_parking_lot_tire_count_l36_3615

/-- The number of tires in a parking lot with four-wheel drive cars and spare tires -/
theorem tires_in_parking_lot (num_cars : ℕ) (wheels_per_car : ℕ) (has_spare : Bool) : ℕ :=
  let regular_tires := num_cars * wheels_per_car
  let spare_tires := if has_spare then num_cars else 0
  regular_tires + spare_tires

/-- Proof that there are 150 tires in the parking lot with 30 four-wheel drive cars and spare tires -/
theorem parking_lot_tire_count :
  tires_in_parking_lot 30 4 true = 150 := by
  sorry

end NUMINAMATH_CALUDE_tires_in_parking_lot_parking_lot_tire_count_l36_3615


namespace NUMINAMATH_CALUDE_count_valid_numbers_l36_3625

/-- The set of digits that can be used to form the numbers. -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A function that checks if a number is even. -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A function that checks if a three-digit number is less than 600. -/
def lessThan600 (n : Nat) : Bool := n < 600 ∧ n ≥ 100

/-- The set of valid hundreds digits (1 to 5). -/
def validHundreds : Finset Nat := {1, 2, 3, 4, 5}

/-- The set of valid units digits (0, 2, 4). -/
def validUnits : Finset Nat := {0, 2, 4}

/-- The main theorem stating the number of valid three-digit numbers. -/
theorem count_valid_numbers : 
  (validHundreds.card * digits.card * validUnits.card : Nat) = 90 := by sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l36_3625


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l36_3636

theorem system_of_equations_solution (a b x y : ℝ) : 
  (x - y * Real.sqrt (x^2 - y^2)) / Real.sqrt (1 - x^2 + y^2) = a ∧
  (y - x * Real.sqrt (x^2 - y^2)) / Real.sqrt (1 - x^2 + y^2) = b →
  x = (a + b * Real.sqrt (a^2 - b^2)) / Real.sqrt (1 - a^2 + b^2) ∧
  y = (b + a * Real.sqrt (a^2 - b^2)) / Real.sqrt (1 - a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l36_3636


namespace NUMINAMATH_CALUDE_least_positive_angle_solution_l36_3610

-- Define the problem
theorem least_positive_angle_solution (θ : Real) : 
  (θ > 0 ∧ 
   Real.cos (10 * π / 180) = Real.sin (35 * π / 180) + Real.sin θ ∧ 
   ∀ φ, φ > 0 ∧ Real.cos (10 * π / 180) = Real.sin (35 * π / 180) + Real.sin φ → θ ≤ φ) → 
  θ = 32.5 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_solution_l36_3610


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l36_3670

theorem power_mod_thirteen : 7^1234 ≡ 4 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l36_3670
