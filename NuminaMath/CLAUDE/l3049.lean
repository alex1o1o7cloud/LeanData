import Mathlib

namespace NUMINAMATH_CALUDE_production_days_calculation_l3049_304991

/-- Proves that the number of days is 4, given the conditions from the problem -/
theorem production_days_calculation (n : ℕ) : 
  (∀ (average_past : ℝ) (production_today : ℝ) (new_average : ℝ),
    average_past = 50 ∧
    production_today = 90 ∧
    new_average = 58 ∧
    (n : ℝ) * average_past + production_today = (n + 1 : ℝ) * new_average) →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3049_304991


namespace NUMINAMATH_CALUDE_sqrt_neg_a_rational_implies_a_opposite_perfect_square_l3049_304910

theorem sqrt_neg_a_rational_implies_a_opposite_perfect_square (a : ℝ) :
  (∃ q : ℚ, q^2 = -a) → ∃ n : ℕ, a = -(n^2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_a_rational_implies_a_opposite_perfect_square_l3049_304910


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l3049_304979

/-- The number of ways to distribute students among communities -/
def distribute_students (n_students : ℕ) (n_communities : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the correct number of arrangements -/
theorem student_distribution_theorem :
  distribute_students 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l3049_304979


namespace NUMINAMATH_CALUDE_camping_payment_difference_l3049_304988

/-- Represents the camping trip expenses and calculations --/
structure CampingExpenses where
  alan_paid : ℝ
  beth_paid : ℝ
  chris_paid : ℝ
  picnic_cost : ℝ
  total_cost : ℝ
  alan_share : ℝ
  beth_share : ℝ
  chris_share : ℝ

/-- Calculates the difference between what Alan and Beth need to pay Chris --/
def payment_difference (expenses : CampingExpenses) : ℝ :=
  (expenses.alan_share - expenses.alan_paid) - (expenses.beth_share - expenses.beth_paid)

/-- Theorem stating that the payment difference is 30 --/
theorem camping_payment_difference :
  ∃ (expenses : CampingExpenses),
    expenses.alan_paid = 110 ∧
    expenses.beth_paid = 140 ∧
    expenses.chris_paid = 190 ∧
    expenses.picnic_cost = 60 ∧
    expenses.total_cost = expenses.alan_paid + expenses.beth_paid + expenses.chris_paid + expenses.picnic_cost ∧
    expenses.alan_share = expenses.total_cost / 3 ∧
    expenses.beth_share = expenses.total_cost / 3 ∧
    expenses.chris_share = expenses.total_cost / 3 ∧
    payment_difference expenses = 30 := by
  sorry

end NUMINAMATH_CALUDE_camping_payment_difference_l3049_304988


namespace NUMINAMATH_CALUDE_digit_sum_property_l3049_304935

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem digit_sum_property (n : ℕ) 
  (h1 : sum_of_digits n = 100)
  (h2 : sum_of_digits (44 * n) = 800) : 
  sum_of_digits (3 * n) = 300 := by sorry

end NUMINAMATH_CALUDE_digit_sum_property_l3049_304935


namespace NUMINAMATH_CALUDE_complex_power_simplification_l3049_304919

theorem complex_power_simplification :
  (Complex.exp (Complex.I * (123 * π / 180)))^25 = 
  -Complex.cos (15 * π / 180) - Complex.I * Complex.sin (15 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l3049_304919


namespace NUMINAMATH_CALUDE_count_divisible_numbers_count_divisible_numbers_proof_l3049_304903

theorem count_divisible_numbers : ℕ → Prop :=
  fun n => 
    (∃ (S : Finset ℕ), 
      (∀ x ∈ S, 1000 ≤ x ∧ x ≤ 3000 ∧ 12 ∣ x ∧ 18 ∣ x ∧ 24 ∣ x) ∧
      (∀ x, 1000 ≤ x ∧ x ≤ 3000 ∧ 12 ∣ x ∧ 18 ∣ x ∧ 24 ∣ x → x ∈ S) ∧
      S.card = n) →
    n = 28

theorem count_divisible_numbers_proof : count_divisible_numbers 28 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_count_divisible_numbers_proof_l3049_304903


namespace NUMINAMATH_CALUDE_parabola_coefficients_l3049_304936

/-- A parabola with vertex (h, k), vertical axis of symmetry, passing through point (x₀, y₀) -/
structure Parabola where
  h : ℝ
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- The quadratic function representing the parabola -/
def quadratic_function (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

theorem parabola_coefficients (p : Parabola) 
  (h_vertex : p.h = 2 ∧ p.k = -3)
  (h_point : p.x₀ = 0 ∧ p.y₀ = 1)
  (h_passes : quadratic_function p p.x₀ = p.y₀) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -4 ∧ c = 1 ∧
  ∀ x, quadratic_function p x = a * x^2 + b * x + c :=
sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l3049_304936


namespace NUMINAMATH_CALUDE_two_sided_iced_cubes_count_l3049_304980

/-- Represents a cube cake with icing -/
structure IcedCake where
  size : Nat
  hasTopIcing : Bool
  hasSideIcing : Bool
  hasVerticalStrip : Bool

/-- Counts the number of 1x1x1 cubes with exactly two iced sides -/
def countTwoSidedIcedCubes (cake : IcedCake) : Nat :=
  sorry

/-- Theorem stating that a 5x5x5 cake with specified icing has 27 two-sided iced cubes -/
theorem two_sided_iced_cubes_count (cake : IcedCake) :
  cake.size = 5 ∧ cake.hasTopIcing ∧ cake.hasSideIcing ∧ cake.hasVerticalStrip →
  countTwoSidedIcedCubes cake = 27 := by
  sorry

end NUMINAMATH_CALUDE_two_sided_iced_cubes_count_l3049_304980


namespace NUMINAMATH_CALUDE_brets_nap_time_l3049_304939

/-- Represents the duration of Bret's train journey and activities --/
structure TrainJourney where
  totalTime : ℝ
  readingTime : ℝ
  eatingTime : ℝ
  movieTime : ℝ
  chattingTime : ℝ
  browsingTime : ℝ
  waitingTime : ℝ
  workingTime : ℝ

/-- Calculates the remaining time for napping given a TrainJourney --/
def remainingTimeForNap (journey : TrainJourney) : ℝ :=
  journey.totalTime - (journey.readingTime + journey.eatingTime + journey.movieTime + 
    journey.chattingTime + journey.browsingTime + journey.waitingTime + journey.workingTime)

/-- Theorem stating that for Bret's specific journey, the remaining time for napping is 4.75 hours --/
theorem brets_nap_time (journey : TrainJourney) 
  (h1 : journey.totalTime = 15)
  (h2 : journey.readingTime = 2)
  (h3 : journey.eatingTime = 1)
  (h4 : journey.movieTime = 3)
  (h5 : journey.chattingTime = 1)
  (h6 : journey.browsingTime = 0.75)
  (h7 : journey.waitingTime = 0.5)
  (h8 : journey.workingTime = 2) :
  remainingTimeForNap journey = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_brets_nap_time_l3049_304939


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l3049_304964

theorem sum_of_x_solutions (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 289) : 
  ∃ x1 x2 : ℝ, x1^2 + y^2 = 289 ∧ x2^2 + y^2 = 289 ∧ x1 + x2 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l3049_304964


namespace NUMINAMATH_CALUDE_same_terminal_side_l3049_304984

theorem same_terminal_side (k : ℤ) : 
  -330 = k * 360 + 30 :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_l3049_304984


namespace NUMINAMATH_CALUDE_inverse_variation_problems_l3049_304902

/-- Two real numbers vary inversely if their product is constant -/
def VaryInversely (r s : ℝ) : Prop :=
  ∃ k : ℝ, ∀ r' s', r' * s' = k

theorem inverse_variation_problems
  (h : VaryInversely r s)
  (h1 : r = 1500 ↔ s = 0.25) :
  (r = 3000 → s = 0.125) ∧ (s = 0.15 → r = 2500) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problems_l3049_304902


namespace NUMINAMATH_CALUDE_airplane_travel_time_l3049_304952

/-- Proves that an airplane traveling 3600 km against the wind in 5 hours,
    with a still air speed of 810 km/h, takes 4 hours to travel the same distance with the wind. -/
theorem airplane_travel_time
  (distance : ℝ)
  (time_against : ℝ)
  (speed_still : ℝ)
  (h_distance : distance = 3600)
  (h_time_against : time_against = 5)
  (h_speed_still : speed_still = 810)
  : ∃ (wind_speed : ℝ),
    (distance / (speed_still - wind_speed) = time_against) ∧
    (distance / (speed_still + wind_speed) = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_airplane_travel_time_l3049_304952


namespace NUMINAMATH_CALUDE_polynomial_d_value_l3049_304924

/-- Represents a polynomial of degree 4 -/
structure Polynomial4 (α : Type) [Field α] where
  a : α
  b : α
  c : α
  d : α

/-- Calculates the sum of coefficients for a polynomial of degree 4 -/
def sumCoefficients {α : Type} [Field α] (p : Polynomial4 α) : α :=
  1 + p.a + p.b + p.c + p.d

/-- Calculates the mean of zeros for a polynomial of degree 4 -/
def meanZeros {α : Type} [Field α] (p : Polynomial4 α) : α :=
  -p.a / 4

/-- The main theorem -/
theorem polynomial_d_value
  {α : Type} [Field α]
  (p : Polynomial4 α)
  (h1 : meanZeros p = p.d)
  (h2 : p.d = sumCoefficients p)
  (h3 : p.d = 3) :
  p.d = 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_d_value_l3049_304924


namespace NUMINAMATH_CALUDE_quadratic_function_properties_g_zero_for_negative_m_g_max_abs_value_case1_g_max_abs_value_case2_l3049_304929

def f (x : ℝ) := (x + 1)^2 - 4

def g (m : ℝ) (x : ℝ) := m * f x + 1

theorem quadratic_function_properties :
  (∀ x, f x ≥ -4) ∧ f (-2) = -3 ∧ f 0 = -3 := by sorry

theorem g_zero_for_negative_m (m : ℝ) (hm : m < 0) :
  ∃! x, x ≤ 1 ∧ g m x = 0 := by sorry

theorem g_max_abs_value_case1 (m : ℝ) (hm : 0 < m ∧ m ≤ 8/7) :
  ∀ x ∈ [-3, 3/2], |g m x| ≤ 9/4 * m + 1 := by sorry

theorem g_max_abs_value_case2 (m : ℝ) (hm : m > 8/7) :
  ∀ x ∈ [-3, 3/2], |g m x| ≤ 4 * m - 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_g_zero_for_negative_m_g_max_abs_value_case1_g_max_abs_value_case2_l3049_304929


namespace NUMINAMATH_CALUDE_store_distance_ratio_l3049_304976

/-- Represents the distances between locations in Jason's commute --/
structure CommuteDistances where
  house_to_first : ℝ
  first_to_second : ℝ
  second_to_third : ℝ
  third_to_work : ℝ

/-- Theorem stating the ratio of distances between stores --/
theorem store_distance_ratio (d : CommuteDistances) :
  d.house_to_first = 4 ∧
  d.first_to_second = 6 ∧
  d.third_to_work = 4 ∧
  d.second_to_third > d.first_to_second ∧
  d.house_to_first + d.first_to_second + d.second_to_third + d.third_to_work = 24 →
  d.second_to_third / d.first_to_second = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_store_distance_ratio_l3049_304976


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3049_304994

theorem negation_of_universal_proposition (a : ℝ) :
  (¬ ∀ x > 0, Real.log x = a) ↔ (∃ x > 0, Real.log x ≠ a) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3049_304994


namespace NUMINAMATH_CALUDE_beaver_count_l3049_304983

theorem beaver_count (initial_beavers additional_beaver : ℝ) 
  (h1 : initial_beavers = 2.0) 
  (h2 : additional_beaver = 1.0) : 
  initial_beavers + additional_beaver = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_beaver_count_l3049_304983


namespace NUMINAMATH_CALUDE_valve2_opened_implies_3_and_4_opened_l3049_304990

-- Define the valve states
inductive ValveState
| Opened
| Closed

-- Define the valve system
structure ValveSystem :=
(valve1 : ValveState)
(valve2 : ValveState)
(valve3 : ValveState)
(valve4 : ValveState)
(valve5 : ValveState)

-- Define the rules
def rule1 (vs : ValveSystem) : Prop :=
vs.valve2 = ValveState.Opened → vs.valve3 = ValveState.Opened ∧ vs.valve1 = ValveState.Closed

def rule2 (vs : ValveSystem) : Prop :=
(vs.valve1 = ValveState.Opened ∨ vs.valve3 = ValveState.Opened) → vs.valve5 = ValveState.Closed

def rule3 (vs : ValveSystem) : Prop :=
¬(vs.valve4 = ValveState.Closed ∧ vs.valve5 = ValveState.Closed)

-- Define a valid valve system
def validValveSystem (vs : ValveSystem) : Prop :=
rule1 vs ∧ rule2 vs ∧ rule3 vs

-- Theorem to prove
theorem valve2_opened_implies_3_and_4_opened (vs : ValveSystem) 
  (h : validValveSystem vs) (h2 : vs.valve2 = ValveState.Opened) : 
  vs.valve3 = ValveState.Opened ∧ vs.valve4 = ValveState.Opened :=
sorry

end NUMINAMATH_CALUDE_valve2_opened_implies_3_and_4_opened_l3049_304990


namespace NUMINAMATH_CALUDE_distance_to_school_l3049_304982

/-- The distance from the neighborhood to the school in meters. -/
def school_distance : ℝ := 960

/-- The initial speed of student A in meters per minute. -/
def speed_A_initial : ℝ := 40

/-- The initial speed of student B in meters per minute. -/
def speed_B_initial : ℝ := 60

/-- The speed of student A after increasing in meters per minute. -/
def speed_A_increased : ℝ := 60

/-- The speed of student B after decreasing in meters per minute. -/
def speed_B_decreased : ℝ := 40

/-- The time difference in minutes between A and B's arrival at school. -/
def time_difference : ℝ := 2

/-- Theorem stating that given the conditions, the distance to school is 960 meters. -/
theorem distance_to_school :
  ∀ (distance : ℝ),
  (∃ (time_A time_B : ℝ),
    distance / 2 = speed_A_initial * time_A
    ∧ distance / 2 = speed_A_increased * (time_B - time_A)
    ∧ distance = speed_B_initial * time_A + speed_B_decreased * (time_B - time_A)
    ∧ time_B + time_difference = time_A)
  → distance = school_distance :=
by sorry

end NUMINAMATH_CALUDE_distance_to_school_l3049_304982


namespace NUMINAMATH_CALUDE_new_girl_weight_l3049_304937

theorem new_girl_weight (W : ℝ) (new_weight : ℝ) :
  (W - 40 + new_weight) / 20 = W / 20 + 2 →
  new_weight = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_new_girl_weight_l3049_304937


namespace NUMINAMATH_CALUDE_triangle_side_length_l3049_304932

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 4 → b = 6 → C = 2 * π / 3 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 2 * Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3049_304932


namespace NUMINAMATH_CALUDE_joe_initial_cars_l3049_304961

/-- The number of cars Joe will have after getting more cars -/
def total_cars : ℕ := 62

/-- The number of additional cars Joe will get -/
def additional_cars : ℕ := 12

/-- Theorem: Joe's initial number of cars is 50 -/
theorem joe_initial_cars : 
  total_cars - additional_cars = 50 := by
  sorry

end NUMINAMATH_CALUDE_joe_initial_cars_l3049_304961


namespace NUMINAMATH_CALUDE_square_function_not_property_P_l3049_304973

/-- Property P for a function f --/
def has_property_P (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f ((x₁ + x₂) / 2) = (f x₁ + f x₂) / 2

/-- The square function --/
def square_function (x : ℝ) : ℝ := x^2

/-- Theorem: The square function does not have property P --/
theorem square_function_not_property_P : ¬(has_property_P square_function) := by
  sorry

end NUMINAMATH_CALUDE_square_function_not_property_P_l3049_304973


namespace NUMINAMATH_CALUDE_emily_phone_bill_l3049_304928

/-- Calculates the total cost of a cell phone plan based on usage --/
def calculate_phone_bill (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
  (extra_data_cost : ℚ) (texts_sent : ℕ) (hours_talked : ℕ) (data_used : ℕ) : ℚ :=
  let text_charge := text_cost * texts_sent
  let extra_minutes := max (hours_talked - 25) 0 * 60
  let minute_charge := extra_minute_cost * extra_minutes
  let extra_data := max (data_used - 15) 0
  let data_charge := extra_data_cost * extra_data
  base_cost + text_charge + minute_charge + data_charge

/-- Theorem stating that Emily's phone bill is $59.00 --/
theorem emily_phone_bill : 
  calculate_phone_bill 30 0.1 0.15 5 150 26 16 = 59 := by
  sorry

end NUMINAMATH_CALUDE_emily_phone_bill_l3049_304928


namespace NUMINAMATH_CALUDE_tissue_magnification_l3049_304921

/-- Given a circular piece of tissue magnified by an electron microscope, 
    this theorem proves the relationship between the magnified image diameter 
    and the actual tissue diameter. -/
theorem tissue_magnification (magnification : ℝ) (magnified_diameter : ℝ) 
  (h1 : magnification = 1000) 
  (h2 : magnified_diameter = 2) :
  magnified_diameter / magnification = 0.002 := by
  sorry

end NUMINAMATH_CALUDE_tissue_magnification_l3049_304921


namespace NUMINAMATH_CALUDE_divisor_sum_and_totient_inequality_divisor_sum_and_totient_equality_l3049_304960

def σ (n : ℕ) : ℕ := sorry

def φ (n : ℕ) : ℕ := sorry

theorem divisor_sum_and_totient_inequality (n : ℕ) :
  n ≠ 0 → (1 : ℝ) / σ n + (1 : ℝ) / φ n ≥ 2 / n :=
sorry

theorem divisor_sum_and_totient_equality (n : ℕ) :
  n ≠ 0 → ((1 : ℝ) / σ n + (1 : ℝ) / φ n = 2 / n ↔ n = 1) :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_and_totient_inequality_divisor_sum_and_totient_equality_l3049_304960


namespace NUMINAMATH_CALUDE_pentagon_coverage_theorem_l3049_304941

/-- Represents the tiling of a plane with squares and pentagons -/
structure PlaneTiling where
  /-- The number of smaller squares in each large square -/
  total_squares : ℕ
  /-- The number of smaller squares used to form pentagons in each large square -/
  pentagon_squares : ℕ

/-- Calculates the percentage of the plane covered by pentagons -/
def pentagon_coverage_percentage (tiling : PlaneTiling) : ℚ :=
  (tiling.pentagon_squares : ℚ) / (tiling.total_squares : ℚ) * 100

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest_integer (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- Theorem stating that the percentage of the plane covered by pentagons
    in the given tiling is 56% when rounded to the nearest integer -/
theorem pentagon_coverage_theorem (tiling : PlaneTiling) 
  (h1 : tiling.total_squares = 9)
  (h2 : tiling.pentagon_squares = 5) : 
  round_to_nearest_integer (pentagon_coverage_percentage tiling) = 56 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_coverage_theorem_l3049_304941


namespace NUMINAMATH_CALUDE_hyperbola_C_properties_l3049_304966

/-- Hyperbola C with distance √2 from focus to asymptote -/
structure HyperbolaC where
  b : ℝ
  b_pos : b > 0
  focus_to_asymptote : ∃ (c : ℝ), b * c / Real.sqrt (b^2 + 2) = Real.sqrt 2

/-- Intersection points of line l with hyperbola C -/
structure IntersectionPoints (h : HyperbolaC) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  l_passes_through_2_0 : ∃ (m : ℝ), A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2
  A_on_C : A.1^2 - A.2^2 = 2
  B_on_C : B.1^2 - B.2^2 = 2
  A_B_right_branch : A.1 > 0 ∧ B.1 > 0

/-- Main theorem -/
theorem hyperbola_C_properties (h : HyperbolaC) :
  (∀ (x y : ℝ), x^2/2 - y^2/h.b^2 = 1 ↔ x^2 - y^2 = 2) ∧
  (∀ (i : IntersectionPoints h),
    ∃ (N : ℝ × ℝ), N = (1, 0) ∧
      (i.A.1 - N.1) * (i.B.1 - N.1) + (i.A.2 - N.2) * (i.B.2 - N.2) = -1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_C_properties_l3049_304966


namespace NUMINAMATH_CALUDE_student_weight_l3049_304938

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight + sister_weight = 116)
  (h2 : student_weight - 5 = 2 * sister_weight) : 
  student_weight = 79 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l3049_304938


namespace NUMINAMATH_CALUDE_drive_time_calculation_l3049_304922

/-- Given a person drives 120 miles in 3 hours, prove that driving 200 miles
    at the same speed will take 5 hours. -/
theorem drive_time_calculation (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ)
    (h1 : distance1 = 120)
    (h2 : time1 = 3)
    (h3 : distance2 = 200) :
  let speed := distance1 / time1
  distance2 / speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_drive_time_calculation_l3049_304922


namespace NUMINAMATH_CALUDE_gemstones_for_four_sets_l3049_304908

/-- Calculates the number of gemstones needed for earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let earrings_per_set := 2
  let magnets_per_earring := 2
  let buttons_per_earring := magnets_per_earring / 2
  let gemstones_per_earring := buttons_per_earring * 3
  num_sets * earrings_per_set * gemstones_per_earring

/-- Proves that 4 sets of earrings require 24 gemstones -/
theorem gemstones_for_four_sets : gemstones_needed 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gemstones_for_four_sets_l3049_304908


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3049_304997

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 38) (h2 : zero_books = 2) (h3 : one_book = 12) 
  (h4 : two_books = 10) (h5 : avg_books = 2) : ∃ (max_books : ℕ), max_books = 5 ∧ 
  (∀ (student_books : ℕ), student_books ≤ max_books) := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l3049_304997


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l3049_304999

theorem arithmetic_sequence_count :
  ∀ (a l d : ℝ) (n : ℕ),
    a = 2.5 →
    l = 62.5 →
    d = 5 →
    l = a + (n - 1) * d →
    n = 13 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l3049_304999


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3049_304942

theorem x_squared_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^4 + 1/x^4 = 47 → x^2 + 1/x^2 = 7 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3049_304942


namespace NUMINAMATH_CALUDE_two_year_increase_l3049_304909

/-- Given an initial amount that increases by 1/8th of itself each year,
    calculate the amount after a given number of years. -/
def amount_after_years (initial_amount : ℚ) (years : ℕ) : ℚ :=
  initial_amount * (1 + 1/8) ^ years

/-- Theorem: If an initial amount of 1600 increases by 1/8th of itself each year for two years,
    the final amount will be 2025. -/
theorem two_year_increase : amount_after_years 1600 2 = 2025 := by
  sorry

#eval amount_after_years 1600 2

end NUMINAMATH_CALUDE_two_year_increase_l3049_304909


namespace NUMINAMATH_CALUDE_log_27_3_l3049_304927

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_27_3 : log 27 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l3049_304927


namespace NUMINAMATH_CALUDE_total_cost_approx_l3049_304981

/-- Calculate the final price of an item after discounts and tax -/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (taxRate : ℝ) : ℝ :=
  let priceAfterDiscount1 := originalPrice * (1 - discount1)
  let priceAfterDiscount2 := priceAfterDiscount1 * (1 - discount2)
  let taxAmount := priceAfterDiscount2 * taxRate
  priceAfterDiscount2 + taxAmount

/-- Calculate the total cost of all items -/
def totalCost (item1Price : ℝ) (item2Price : ℝ) (item3Price : ℝ) : ℝ :=
  let item1 := finalPrice item1Price 0.25 0.15 0.07
  let item2 := finalPrice item2Price 0.30 0 0.10
  let item3 := finalPrice item3Price 0.20 0 0.05
  item1 + item2 + item3

/-- Theorem: The total cost for all three items is approximately $335.93 -/
theorem total_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ abs (totalCost 200 150 100 - 335.93) < ε :=
sorry

end NUMINAMATH_CALUDE_total_cost_approx_l3049_304981


namespace NUMINAMATH_CALUDE_andrew_remaining_vacation_days_l3049_304943

def vacation_days_earned (days_worked : ℕ) : ℕ :=
  days_worked / 10

def days_count_for_vacation (total_days_worked public_holidays sick_leave : ℕ) : ℕ :=
  total_days_worked - public_holidays - sick_leave

theorem andrew_remaining_vacation_days 
  (total_days_worked : ℕ) 
  (public_holidays : ℕ) 
  (sick_leave : ℕ) 
  (march_vacation : ℕ) 
  (h1 : total_days_worked = 290)
  (h2 : public_holidays = 10)
  (h3 : sick_leave = 5)
  (h4 : march_vacation = 5) :
  vacation_days_earned (days_count_for_vacation total_days_worked public_holidays sick_leave) - 
  (march_vacation + 2 * march_vacation) = 12 :=
by
  sorry

#eval vacation_days_earned (days_count_for_vacation 290 10 5) - (5 + 2 * 5)

end NUMINAMATH_CALUDE_andrew_remaining_vacation_days_l3049_304943


namespace NUMINAMATH_CALUDE_triangle_area_arithmetic_angles_l3049_304945

/-- Given a triangle ABC where angles A, B, and C form an arithmetic sequence,
    and sides a = 1 and b = √3, the area of the triangle is √3/2. -/
theorem triangle_area_arithmetic_angles (A B C : ℝ) (a b c : ℝ) : 
  A + C = 2 * B → -- angles form arithmetic sequence
  A + B + C = π → -- sum of angles in a triangle
  a = 1 → -- given side length
  b = Real.sqrt 3 → -- given side length
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_arithmetic_angles_l3049_304945


namespace NUMINAMATH_CALUDE_intersection_M_N_l3049_304963

def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_M_N : M ∩ N = {x | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3049_304963


namespace NUMINAMATH_CALUDE_rectangle_half_size_existence_l3049_304989

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Checks if rectangle B has half the perimeter and area of rectangle A -/
def isHalfSized (A B : Rectangle) : Prop :=
  perimeter B = (perimeter A) / 2 ∧ area B = (area A) / 2

theorem rectangle_half_size_existence :
  -- Case 1: Rectangle A with sides 6 and 1
  (∃ B : Rectangle, isHalfSized ⟨6, 1⟩ B) ∧
  -- Case 2: Rectangle A with sides 2 and 1
  (¬ ∃ B : Rectangle, isHalfSized ⟨2, 1⟩ B) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_half_size_existence_l3049_304989


namespace NUMINAMATH_CALUDE_odd_function_theorem_l3049_304948

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Definition of function g in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

/-- Theorem: If f is odd, g(x) = f(x) + 2, and g(1) = 1, then g(-1) = 3 -/
theorem odd_function_theorem (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : g f 1 = 1) : g f (-1) = 3 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_theorem_l3049_304948


namespace NUMINAMATH_CALUDE_circle_diameter_relation_l3049_304916

theorem circle_diameter_relation (R S : Real) (h : R > 0 ∧ S > 0) :
  (R * R) / (S * S) = 0.16 → R / S = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_relation_l3049_304916


namespace NUMINAMATH_CALUDE_max_value_bound_max_value_achievable_l3049_304987

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def max_value (a b c : V) : ℝ :=
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2

theorem max_value_bound (a b c : V) (ha : ‖a‖ = 3) (hb : ‖b‖ = 4) (hc : ‖c‖ = 2) :
  max_value a b c ≤ 253 :=
sorry

theorem max_value_achievable :
  ∃ (a b c : V), ‖a‖ = 3 ∧ ‖b‖ = 4 ∧ ‖c‖ = 2 ∧ max_value a b c = 253 :=
sorry

end NUMINAMATH_CALUDE_max_value_bound_max_value_achievable_l3049_304987


namespace NUMINAMATH_CALUDE_room_length_proof_l3049_304907

/-- Given the width, total cost, and rate of paving a room's floor, 
    prove that the length of the room is 5.5 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) 
    (h1 : width = 3.75)
    (h2 : total_cost = 20625)
    (h3 : paving_rate = 1000) : 
  (total_cost / paving_rate) / width = 5.5 := by
  sorry

#check room_length_proof

end NUMINAMATH_CALUDE_room_length_proof_l3049_304907


namespace NUMINAMATH_CALUDE_sonnys_cookies_l3049_304900

/-- Given an initial number of cookie boxes and the number of boxes given to brother, sister, and cousin,
    calculate the number of boxes left for Sonny. -/
def cookies_left (initial : ℕ) (to_brother : ℕ) (to_sister : ℕ) (to_cousin : ℕ) : ℕ :=
  initial - (to_brother + to_sister + to_cousin)

/-- Theorem stating that given 45 initial boxes of cookies, after giving away 12 to brother,
    9 to sister, and 7 to cousin, the number of boxes left for Sonny is 17. -/
theorem sonnys_cookies : cookies_left 45 12 9 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sonnys_cookies_l3049_304900


namespace NUMINAMATH_CALUDE_kennel_total_is_45_l3049_304930

/-- Represents the number of dogs in a kennel with specific characteristics. -/
structure KennelDogs where
  long_fur : ℕ
  brown : ℕ
  neither : ℕ
  long_fur_and_brown : ℕ

/-- Calculates the total number of dogs in the kennel. -/
def total_dogs (k : KennelDogs) : ℕ :=
  k.long_fur + k.brown - k.long_fur_and_brown + k.neither

/-- Theorem stating the total number of dogs in the kennel is 45. -/
theorem kennel_total_is_45 (k : KennelDogs) 
    (h1 : k.long_fur = 29)
    (h2 : k.brown = 17)
    (h3 : k.neither = 8)
    (h4 : k.long_fur_and_brown = 9) :
  total_dogs k = 45 := by
  sorry

end NUMINAMATH_CALUDE_kennel_total_is_45_l3049_304930


namespace NUMINAMATH_CALUDE_expected_draws_for_specific_box_l3049_304986

/-- A box containing red and white balls -/
structure Box where
  red : ℕ
  white : ℕ

/-- The expected number of draws needed to pick a white ball -/
def expectedDraws (b : Box) : ℚ :=
  -- Definition to be proved
  11/9

/-- Theorem stating the expected number of draws for a specific box configuration -/
theorem expected_draws_for_specific_box :
  let b : Box := ⟨2, 8⟩
  expectedDraws b = 11/9 := by
  sorry


end NUMINAMATH_CALUDE_expected_draws_for_specific_box_l3049_304986


namespace NUMINAMATH_CALUDE_new_person_weight_l3049_304968

/-- Given a group of 8 people where one person weighing 55 kg is replaced by a new person,
    and the average weight of the group increases by 2.5 kg, prove that the weight of the new person is 75 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 55 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 75 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3049_304968


namespace NUMINAMATH_CALUDE_jan_skips_after_training_l3049_304923

/-- The number of skips Jan does in 5 minutes after doubling her initial speed -/
def total_skips (initial_speed : ℕ) (time : ℕ) : ℕ :=
  2 * initial_speed * time

/-- Theorem stating that Jan does 700 skips in 5 minutes after doubling her initial speed of 70 skips per minute -/
theorem jan_skips_after_training :
  total_skips 70 5 = 700 := by
  sorry

end NUMINAMATH_CALUDE_jan_skips_after_training_l3049_304923


namespace NUMINAMATH_CALUDE_ratio_c_to_d_l3049_304913

theorem ratio_c_to_d (a b c d : ℝ) 
  (hab : a / b = 3 / 4)
  (hbc : b / c = 7 / 9)
  (had : a / d = 0.4166666666666667) :
  c / d = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_c_to_d_l3049_304913


namespace NUMINAMATH_CALUDE_ivan_share_increase_l3049_304992

theorem ivan_share_increase (p v s i : ℝ) 
  (h1 : p + v + s + i > 0)
  (h2 : 2*p + v + s + i = 1.3*(p + v + s + i))
  (h3 : p + 2*v + s + i = 1.25*(p + v + s + i))
  (h4 : p + v + 3*s + i = 1.5*(p + v + s + i)) :
  ∃ k : ℝ, k > 6 ∧ k*i > 0.6*(p + v + s + k*i) := by
  sorry

end NUMINAMATH_CALUDE_ivan_share_increase_l3049_304992


namespace NUMINAMATH_CALUDE_percentage_calculation_l3049_304959

theorem percentage_calculation (P : ℝ) : 
  (P / 100) * (30 / 100) * (50 / 100) * 5200 = 117 → P = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3049_304959


namespace NUMINAMATH_CALUDE_min_side_length_two_triangles_l3049_304967

/-- Given two triangles ABC and DBC sharing side BC, with known side lengths,
    prove that the minimum integral length of BC is 16 cm. -/
theorem min_side_length_two_triangles 
  (AB AC DC BD : ℝ) 
  (h_AB : AB = 7)
  (h_AC : AC = 18)
  (h_DC : DC = 10)
  (h_BD : BD = 25) :
  (∃ (BC : ℕ), BC ≥ 16 ∧ ∀ (n : ℕ), n < 16 → 
    (n : ℝ) ≤ AC - AB ∨ (n : ℝ) ≤ BD - DC) :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_two_triangles_l3049_304967


namespace NUMINAMATH_CALUDE_man_sold_portion_l3049_304993

theorem man_sold_portion (lot_value : ℝ) (sold_amount : ℝ) : 
  lot_value = 9200 → 
  sold_amount = 460 → 
  sold_amount / (lot_value / 2) = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_man_sold_portion_l3049_304993


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l3049_304958

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on a parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : para.eq x y

/-- Circle passing through two points -/
def circle_eq (P₁ P₂ : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - P₁.1) * (x - P₂.1) + (y - P₁.2) * (y - P₂.2) = 0

/-- Main theorem -/
theorem parabola_circle_tangency (para : Parabola) (P₁ P₂ : ParabolaPoint para)
    (h : |P₁.y - P₂.y| = 4 * para.p) :
    ∃! (P : ℝ × ℝ), P ≠ (P₁.x, P₁.y) ∧ P ≠ (P₂.x, P₂.y) ∧
      para.eq P.1 P.2 ∧ circle_eq (P₁.x, P₁.y) (P₂.x, P₂.y) P.1 P.2 :=
  sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l3049_304958


namespace NUMINAMATH_CALUDE_fraction_equality_l3049_304995

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 12)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 8) :
  m / q = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3049_304995


namespace NUMINAMATH_CALUDE_rent_increase_theorem_l3049_304971

theorem rent_increase_theorem (num_friends : ℕ) (initial_avg_rent : ℚ) 
  (increased_rent : ℚ) (increase_percentage : ℚ) :
  num_friends = 4 →
  initial_avg_rent = 800 →
  increased_rent = 1400 →
  increase_percentage = 20 / 100 →
  let total_rent : ℚ := initial_avg_rent * num_friends
  let new_increased_rent : ℚ := increased_rent * (1 + increase_percentage)
  let new_total_rent : ℚ := total_rent - increased_rent + new_increased_rent
  let new_avg_rent : ℚ := new_total_rent / num_friends
  new_avg_rent = 870 := by sorry

end NUMINAMATH_CALUDE_rent_increase_theorem_l3049_304971


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l3049_304915

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  beads_percent : ℝ
  papers_percent : ℝ
  silver_coins_percent : ℝ
  gold_coins_percent : ℝ

/-- Theorem stating the percentage of gold coins in the urn -/
theorem gold_coins_percentage (u : UrnComposition) 
  (h1 : u.beads_percent = 15)
  (h2 : u.papers_percent = 10)
  (h3 : u.silver_coins_percent + u.gold_coins_percent = 75)
  (h4 : u.silver_coins_percent = 0.3 * 75) :
  u.gold_coins_percent = 52.5 := by
  sorry

#check gold_coins_percentage

end NUMINAMATH_CALUDE_gold_coins_percentage_l3049_304915


namespace NUMINAMATH_CALUDE_james_tylenol_dosage_l3049_304955

/-- Represents the dosage schedule and total daily intake of Tylenol tablets -/
structure TylenolDosage where
  tablets_per_dose : ℕ
  hours_between_doses : ℕ
  total_daily_mg : ℕ

/-- Calculates the mg per tablet given a TylenolDosage -/
def mg_per_tablet (dosage : TylenolDosage) : ℕ :=
  let doses_per_day := 24 / dosage.hours_between_doses
  let tablets_per_day := doses_per_day * dosage.tablets_per_dose
  dosage.total_daily_mg / tablets_per_day

/-- Theorem: Given James' Tylenol dosage schedule, each tablet contains 375 mg -/
theorem james_tylenol_dosage :
  let james_dosage : TylenolDosage := {
    tablets_per_dose := 2,
    hours_between_doses := 6,
    total_daily_mg := 3000
  }
  mg_per_tablet james_dosage = 375 := by
  sorry

end NUMINAMATH_CALUDE_james_tylenol_dosage_l3049_304955


namespace NUMINAMATH_CALUDE_apple_juice_production_l3049_304965

/-- Calculates the amount of apples used for apple juice production in million tons -/
def applesForJuice (totalApples : ℝ) (ciderPercent : ℝ) (freshPercent : ℝ) (juicePercent : ℝ) : ℝ :=
  let ciderApples := ciderPercent * totalApples
  let remainingApples := totalApples - ciderApples
  let freshApples := freshPercent * remainingApples
  let exportedApples := remainingApples - freshApples
  juicePercent * exportedApples

theorem apple_juice_production :
  applesForJuice 6 0.3 0.4 0.6 = 1.512 := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_production_l3049_304965


namespace NUMINAMATH_CALUDE_f_composition_result_l3049_304912

-- Define the function f for complex numbers
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 - 1

-- State the theorem
theorem f_composition_result : f (f (f (f (2 + I)))) = 3589 - 1984 * I := by
  sorry

end NUMINAMATH_CALUDE_f_composition_result_l3049_304912


namespace NUMINAMATH_CALUDE_copy_machines_output_l3049_304904

/-- The rate of the first copy machine in copies per minute -/
def rate1 : ℕ := 30

/-- The rate of the second copy machine in copies per minute -/
def rate2 : ℕ := 55

/-- The time period in minutes -/
def time : ℕ := 30

/-- The total number of copies made by both machines in the given time period -/
def total_copies : ℕ := rate1 * time + rate2 * time

theorem copy_machines_output : total_copies = 2550 := by
  sorry

end NUMINAMATH_CALUDE_copy_machines_output_l3049_304904


namespace NUMINAMATH_CALUDE_lcm_problem_l3049_304978

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 18 m = 54) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l3049_304978


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3049_304918

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : 
  train_length = 360 ∧ 
  train_speed_kmh = 90 ∧ 
  time_to_pass = 20 → 
  (train_speed_kmh * 1000 / 3600) * time_to_pass - train_length = 140 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3049_304918


namespace NUMINAMATH_CALUDE_age_puzzle_l3049_304956

theorem age_puzzle (A N : ℕ) (h1 : A = 30) (h2 : (A + 5) * N - (A - 5) * N = A) : N = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l3049_304956


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3049_304917

theorem fractional_equation_solution : 
  ∃ x : ℝ, (1 / (x - 5) = 1) ∧ (x = 6) := by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3049_304917


namespace NUMINAMATH_CALUDE_square_area_increase_l3049_304926

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.5 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l3049_304926


namespace NUMINAMATH_CALUDE_bridge_length_l3049_304906

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 250 →
  crossing_time = 32 →
  train_speed_kmh = 45 →
  ∃ (bridge_length : ℝ), bridge_length = 150 := by
  sorry


end NUMINAMATH_CALUDE_bridge_length_l3049_304906


namespace NUMINAMATH_CALUDE_last_twelve_average_l3049_304920

theorem last_twelve_average (total_count : Nat) (total_average : ℚ) (first_twelve_average : ℚ) (thirteenth_result : ℚ) :
  total_count = 25 →
  total_average = 24 →
  first_twelve_average = 14 →
  thirteenth_result = 228 →
  (total_count * total_average = 12 * first_twelve_average + thirteenth_result + 12 * ((total_count * total_average - 12 * first_twelve_average - thirteenth_result) / 12)) ∧
  ((total_count * total_average - 12 * first_twelve_average - thirteenth_result) / 12 = 17) := by
sorry

end NUMINAMATH_CALUDE_last_twelve_average_l3049_304920


namespace NUMINAMATH_CALUDE_chocolate_boxes_price_l3049_304977

/-- The price of the small box of chocolates -/
def small_box_price : ℝ := 6

/-- The price of the large box of chocolates -/
def large_box_price : ℝ := small_box_price + 3

/-- The total cost of both boxes -/
def total_cost : ℝ := 15

theorem chocolate_boxes_price :
  small_box_price + large_box_price = total_cost ∧
  large_box_price = small_box_price + 3 ∧
  small_box_price = 6 := by
sorry

end NUMINAMATH_CALUDE_chocolate_boxes_price_l3049_304977


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l3049_304974

/-- A quadratic function g(x) with integer coefficients -/
def g (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- The theorem stating that under given conditions, m = 5 -/
theorem quadratic_function_m_value (a b c m : ℤ) :
  g a b c 2 = 0 →
  70 < g a b c 6 →
  g a b c 6 < 80 →
  110 < g a b c 7 →
  g a b c 7 < 120 →
  2000 * m < g a b c 50 →
  g a b c 50 < 2000 * (m + 1) →
  m = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l3049_304974


namespace NUMINAMATH_CALUDE_cone_base_circumference_l3049_304970

/-- Given a circular piece of paper with radius 5 inches, when a 300° sector is removed
    and the remaining sector is used to form a right circular cone,
    the circumference of the base of the cone is 25π/3 inches. -/
theorem cone_base_circumference :
  let original_radius : ℝ := 5
  let removed_angle : ℝ := 300
  let full_circle_angle : ℝ := 360
  let remaining_fraction : ℝ := (full_circle_angle - removed_angle) / full_circle_angle
  let cone_base_circumference : ℝ := 2 * π * original_radius * remaining_fraction
  cone_base_circumference = 25 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l3049_304970


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l3049_304934

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x ≤ 3}

-- Theorem for A ∪ B = B
theorem union_condition (a : ℝ) : A ∪ B a = B a ↔ a < 2 := by sorry

-- Theorem for A ∩ B = B
theorem intersection_condition (a : ℝ) : A ∩ B a = B a ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l3049_304934


namespace NUMINAMATH_CALUDE_f_composition_nonnegative_iff_a_geq_three_l3049_304933

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 1

theorem f_composition_nonnegative_iff_a_geq_three (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_nonnegative_iff_a_geq_three_l3049_304933


namespace NUMINAMATH_CALUDE_rabbit_clearing_theorem_l3049_304946

/-- Represents the area one rabbit can clear in a day given the land dimensions, number of rabbits, and days to clear -/
def rabbit_clearing_rate (length width : ℕ) (num_rabbits days_to_clear : ℕ) : ℚ :=
  (length * width : ℚ) / 9 / (num_rabbits * days_to_clear)

/-- Theorem stating that given the specific conditions, one rabbit clears 10 square yards per day -/
theorem rabbit_clearing_theorem :
  rabbit_clearing_rate 200 900 100 20 = 10 := by
  sorry

#eval rabbit_clearing_rate 200 900 100 20

end NUMINAMATH_CALUDE_rabbit_clearing_theorem_l3049_304946


namespace NUMINAMATH_CALUDE_fraction_equality_l3049_304954

theorem fraction_equality (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 2) 
  (h2 : c / d = 1 / 2) 
  (h3 : e / f = 1 / 2) 
  (h4 : 3 * b - 2 * d + f ≠ 0) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3049_304954


namespace NUMINAMATH_CALUDE_binary_101110_equals_46_l3049_304925

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_101110_equals_46 :
  binary_to_decimal [false, true, true, true, true, false, true] = 46 := by
  sorry

end NUMINAMATH_CALUDE_binary_101110_equals_46_l3049_304925


namespace NUMINAMATH_CALUDE_part1_part2_l3049_304911

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 ≤ 0}

def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

theorem part1 (m : ℝ) (h : ∀ x, q m x → p x) (h' : ∃ x, p x ∧ ¬q m x) :
  0 ≤ m ∧ m ≤ 1 := by sorry

theorem part2 (m : ℝ) (h : ∀ x ∈ A, x^2 + m ≥ 4 + 3*x) :
  m ≥ 25/4 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3049_304911


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3049_304901

open Real

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Condition 1: sin C + 2sin C cos B = sin A
  sin C + 2 * sin C * cos B = sin A →
  -- Condition 2: C ∈ (0, π/2)
  0 < C ∧ C < π / 2 →
  -- Condition 3: a = √6
  a = Real.sqrt 6 →
  -- Condition 4: cos B = 1/3
  cos B = 1 / 3 →
  -- Conclusion: b = 12/5
  b = 12 / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3049_304901


namespace NUMINAMATH_CALUDE_red_then_black_combinations_l3049_304975

def standard_deck : ℕ := 52
def red_cards : ℕ := 26
def black_cards : ℕ := 26

theorem red_then_black_combinations : 
  standard_deck = red_cards + black_cards →
  red_cards * black_cards = 676 := by
  sorry

end NUMINAMATH_CALUDE_red_then_black_combinations_l3049_304975


namespace NUMINAMATH_CALUDE_rons_height_l3049_304972

/-- Proves that Ron's height is 13 feet given the water depth and its relation to Ron's height -/
theorem rons_height (water_depth : ℝ) (h1 : water_depth = 208) 
  (h2 : ∃ (rons_height : ℝ), water_depth = 16 * rons_height) : 
  ∃ (rons_height : ℝ), rons_height = 13 := by
  sorry

end NUMINAMATH_CALUDE_rons_height_l3049_304972


namespace NUMINAMATH_CALUDE_original_number_exists_l3049_304985

theorem original_number_exists : ∃ x : ℝ, 3 * (2 * x + 5) = 117 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_l3049_304985


namespace NUMINAMATH_CALUDE_sum_remainder_l3049_304947

theorem sum_remainder (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 7 = 2 →
  (3 * c) % 7 = 1 →
  (4 * b) % 7 = (2 + b) % 7 →
  (a + b + c) % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_l3049_304947


namespace NUMINAMATH_CALUDE_center_radius_sum_l3049_304998

/-- Definition of the circle D -/
def D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - 14*p.1 + p.2^2 + 10*p.2 = -34}

/-- Center of the circle D -/
def center : ℝ × ℝ := sorry

/-- Radius of the circle D -/
def radius : ℝ := sorry

/-- Theorem stating the sum of center coordinates and radius -/
theorem center_radius_sum :
  center.1 + center.2 + radius = 2 + 2 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_center_radius_sum_l3049_304998


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l3049_304996

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(Nat.gcd (m - 17) (6 * m + 7) > 1)) ∧
  Nat.gcd (n - 17) (6 * n + 7) > 1 ∧
  n = 126 := by
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l3049_304996


namespace NUMINAMATH_CALUDE_no_primes_in_range_l3049_304940

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ k ∈ Set.Ioo (n! + 2) (n! + n + 1), ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l3049_304940


namespace NUMINAMATH_CALUDE_infinitely_many_m_for_composite_sum_l3049_304905

theorem infinitely_many_m_for_composite_sum : 
  ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    ∀ (m : ℕ+), m ∈ S → 
      ∀ (n : ℕ+), ∃ (a b : ℕ+), a * b = n^4 + m ∧ a ≠ 1 ∧ b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_m_for_composite_sum_l3049_304905


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l3049_304914

theorem perfect_square_trinomial_m_value (m : ℝ) :
  (∃ (a b : ℝ), ∀ y : ℝ, y^2 - m*y + 9 = (a*y + b)^2) →
  m = 6 ∨ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l3049_304914


namespace NUMINAMATH_CALUDE_liquid_distribution_l3049_304953

theorem liquid_distribution (n : ℕ) (a : ℝ) (h : n ≥ 2) :
  ∃ (x : ℕ → ℝ),
    (∀ k, 1 ≤ k ∧ k ≤ n → x k > 0) ∧
    (∀ k, 2 ≤ k ∧ k ≤ n → (1 - 1/n) * x k + (1/n) * x (k-1) = a) ∧
    ((1 - 1/n) * x 1 + (1/n) * x n = a) ∧
    (x 1 = a * n * (n-2) / (n-1)^2) ∧
    (x 2 = a * (n^2 - 2*n + 2) / (n-1)^2) ∧
    (∀ k, 3 ≤ k ∧ k ≤ n → x k = a) :=
by
  sorry

#check liquid_distribution

end NUMINAMATH_CALUDE_liquid_distribution_l3049_304953


namespace NUMINAMATH_CALUDE_midpoint_chain_l3049_304949

/-- Given a line segment AB with several midpoints, prove that AB = 96 -/
theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 3) →        -- AG = 3
  (B - A = 96) :=      -- AB = 96
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l3049_304949


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3049_304944

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals :
  num_diagonals decagon_sides = 35 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3049_304944


namespace NUMINAMATH_CALUDE_five_point_thirty_five_million_equals_scientific_notation_l3049_304931

-- Define 5.35 million
def five_point_thirty_five_million : ℝ := 5.35 * 1000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 5.35 * (10 ^ 6)

-- Theorem to prove equality
theorem five_point_thirty_five_million_equals_scientific_notation : 
  five_point_thirty_five_million = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_five_point_thirty_five_million_equals_scientific_notation_l3049_304931


namespace NUMINAMATH_CALUDE_test_score_calculation_l3049_304951

theorem test_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (incorrect_penalty : ℕ) (total_score : ℕ) :
  total_questions = 30 →
  correct_answers = 19 →
  incorrect_penalty = 5 →
  total_score = 325 →
  ∃ (points_per_correct : ℕ),
    points_per_correct * correct_answers - incorrect_penalty * (total_questions - correct_answers) = total_score ∧
    points_per_correct = 20 :=
by sorry

end NUMINAMATH_CALUDE_test_score_calculation_l3049_304951


namespace NUMINAMATH_CALUDE_not_twenty_percent_less_l3049_304957

theorem not_twenty_percent_less (a b : ℝ) (h : a = b * 1.2) : 
  ¬(b = a * 0.8) := by
  sorry

end NUMINAMATH_CALUDE_not_twenty_percent_less_l3049_304957


namespace NUMINAMATH_CALUDE_polynomial_square_decomposition_l3049_304962

theorem polynomial_square_decomposition (P : Polynomial ℝ) 
  (R : Polynomial ℝ) (h : P^2 = R.comp (Polynomial.X^2)) :
  ∃ Q : Polynomial ℝ, P = Q.comp (Polynomial.X^2) ∨ 
    P = Polynomial.X * Q.comp (Polynomial.X^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_square_decomposition_l3049_304962


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3049_304969

theorem system_solution_ratio (k x y z : ℝ) : 
  x + k*y + 2*z = 0 →
  2*x + k*y + 3*z = 0 →
  3*x + 5*y + 4*z = 0 →
  x ≠ 0 →
  y ≠ 0 →
  z ≠ 0 →
  x*z / (y^2) = -25 := by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3049_304969


namespace NUMINAMATH_CALUDE_more_stable_scores_lower_variance_problem_solution_l3049_304950

/-- Represents an athlete with their test score variance -/
structure Athlete where
  name : String
  variance : ℝ

/-- Determines if an athlete has more stable test scores than another -/
def has_more_stable_scores (a b : Athlete) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two athletes with equal average scores, 
    the athlete with lower variance has more stable test scores -/
theorem more_stable_scores_lower_variance 
  (a b : Athlete) 
  (h_avg : ℝ) -- average score of both athletes
  (h_equal_avg : True) -- assumption that both athletes have equal average scores
  : has_more_stable_scores a b ↔ a.variance < b.variance :=
by sorry

/-- Application to the specific problem -/
def athlete_A : Athlete := ⟨"A", 0.024⟩
def athlete_B : Athlete := ⟨"B", 0.008⟩

theorem problem_solution : has_more_stable_scores athlete_B athlete_A :=
by sorry

end NUMINAMATH_CALUDE_more_stable_scores_lower_variance_problem_solution_l3049_304950
