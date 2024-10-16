import Mathlib

namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l237_23750

theorem circle_diameter_from_area :
  ∀ (A r d : ℝ),
  A = 225 * Real.pi →
  A = Real.pi * r^2 →
  d = 2 * r →
  d = 30 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l237_23750


namespace NUMINAMATH_CALUDE_box_volume_problem_l237_23716

theorem box_volume_problem :
  ∃! (x : ℕ+), (2 * x.val - 5 > 0) ∧
  ((x.val^2 + 5) * (2 * x.val - 5) * (x.val + 25) < 1200) := by
  sorry

end NUMINAMATH_CALUDE_box_volume_problem_l237_23716


namespace NUMINAMATH_CALUDE_parabola_triangle_problem_l237_23797

/-- Given three distinct points A, B, C on the parabola y = x^2, where AB is parallel to the x-axis
    and ABC forms a right triangle with area 2016, prove that the y-coordinate of C is 4064255 -/
theorem parabola_triangle_problem (A B C : ℝ × ℝ) : 
  (∃ m n : ℝ, A = (m, m^2) ∧ B = (n, n^2) ∧ C = ((m+n)/2, ((m+n)/2)^2)) →  -- Points on y = x^2
  (A.2 = B.2) →  -- AB parallel to x-axis
  (C.1 = (A.1 + B.1) / 2) →  -- C is above midpoint of AB (right angle)
  (abs (B.1 - A.1) * abs (C.2 - A.2) / 2 = 2016) →  -- Area of triangle ABC
  C.2 = 4064255 := by
  sorry

end NUMINAMATH_CALUDE_parabola_triangle_problem_l237_23797


namespace NUMINAMATH_CALUDE_probability_less_than_one_third_l237_23764

/-- The probability of selecting a number less than 1/3 from the interval (0, 1/2) is 2/3 -/
theorem probability_less_than_one_third : 
  let total_interval : ℝ := 1/2 - 0
  let desired_interval : ℝ := 1/3 - 0
  desired_interval / total_interval = 2/3 := by
sorry

end NUMINAMATH_CALUDE_probability_less_than_one_third_l237_23764


namespace NUMINAMATH_CALUDE_total_money_proof_l237_23755

def sally_money : ℕ := 100
def jolly_money : ℕ := 50

theorem total_money_proof :
  (sally_money - 20 = 80) ∧ (jolly_money + 20 = 70) →
  sally_money + jolly_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_money_proof_l237_23755


namespace NUMINAMATH_CALUDE_second_planner_cheaper_at_31_l237_23704

/-- Represents the cost function for an event planner -/
structure PlannerCost where
  initial_fee : ℕ
  per_guest : ℕ

/-- Calculates the total cost for a given number of guests -/
def total_cost (p : PlannerCost) (guests : ℕ) : ℕ :=
  p.initial_fee + p.per_guest * guests

/-- First planner's pricing structure -/
def planner1 : PlannerCost := ⟨150, 20⟩

/-- Second planner's pricing structure -/
def planner2 : PlannerCost := ⟨300, 15⟩

/-- Theorem stating that 31 is the minimum number of guests for which the second planner is cheaper -/
theorem second_planner_cheaper_at_31 :
  (∀ g : ℕ, g < 31 → total_cost planner1 g ≤ total_cost planner2 g) ∧
  (∀ g : ℕ, g ≥ 31 → total_cost planner2 g < total_cost planner1 g) :=
sorry

end NUMINAMATH_CALUDE_second_planner_cheaper_at_31_l237_23704


namespace NUMINAMATH_CALUDE_min_value_a1_plus_a7_l237_23745

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a n > 0 ∧ a (n + 1) = r * a n

/-- The theorem stating the minimum value of a₁ + a₇ in a positive geometric sequence where a₃ * a₅ = 64 -/
theorem min_value_a1_plus_a7 (a : ℕ → ℝ) 
    (h_geom : is_positive_geometric_sequence a) 
    (h_prod : a 3 * a 5 = 64) : 
  (∀ b : ℕ → ℝ, is_positive_geometric_sequence b → b 3 * b 5 = 64 → a 1 + a 7 ≤ b 1 + b 7) → 
  a 1 + a 7 = 16 := by
sorry

end NUMINAMATH_CALUDE_min_value_a1_plus_a7_l237_23745


namespace NUMINAMATH_CALUDE_orange_profit_calculation_l237_23715

/-- Calculates the profit from an orange selling operation -/
def orange_profit (buy_quantity : ℕ) (buy_price : ℚ) (sell_quantity : ℕ) (sell_price : ℚ) 
                  (transport_cost : ℚ) (storage_fee : ℚ) : ℚ :=
  let total_cost := buy_price + 2 * transport_cost + storage_fee
  let revenue := sell_price
  revenue - total_cost

/-- The profit from the orange selling operation is -4r -/
theorem orange_profit_calculation : 
  orange_profit 11 10 10 11 2 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_orange_profit_calculation_l237_23715


namespace NUMINAMATH_CALUDE_expression_range_l237_23794

theorem expression_range (a b c : ℝ) 
  (h1 : a - b + c = 0)
  (h2 : c > 0)
  (h3 : 3 * a - 2 * b + c > 0) :
  4/3 < (a + 3*b + 7*c) / (2*a + b) ∧ (a + 3*b + 7*c) / (2*a + b) < 7/2 :=
sorry

end NUMINAMATH_CALUDE_expression_range_l237_23794


namespace NUMINAMATH_CALUDE_ordering_proof_l237_23727

theorem ordering_proof (a b c : ℝ) : 
  a = (1/2)^(1/3) → b = (1/3)^(1/2) → c = Real.log (3/Real.pi) → c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ordering_proof_l237_23727


namespace NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l237_23725

theorem min_sum_given_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  x + y ≥ 6 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1 / (a + 1) + 1 / (b + 1) = 1 / 2 ∧ a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l237_23725


namespace NUMINAMATH_CALUDE_stratified_sample_male_count_l237_23742

theorem stratified_sample_male_count :
  let total_male : ℕ := 560
  let total_female : ℕ := 420
  let sample_size : ℕ := 280
  let total_students : ℕ := total_male + total_female
  let male_ratio : ℚ := total_male / total_students
  male_ratio * sample_size = 160 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_male_count_l237_23742


namespace NUMINAMATH_CALUDE_problem_sample_is_valid_problem_sample_sequence_correct_l237_23708

/-- Represents a systematic sample -/
structure SystematicSample where
  first : ℕ
  interval : ℕ
  size : ℕ
  population : ℕ

/-- Checks if a systematic sample is valid -/
def isValidSystematicSample (s : SystematicSample) : Prop :=
  s.first > 0 ∧
  s.first ≤ s.population ∧
  s.interval > 0 ∧
  s.size > 0 ∧
  s.population ≥ s.size ∧
  ∀ i : ℕ, i < s.size → s.first + i * s.interval ≤ s.population

/-- The specific systematic sample from the problem -/
def problemSample : SystematicSample :=
  { first := 3
    interval := 10
    size := 6
    population := 60 }

/-- Theorem stating that the problem's sample is valid -/
theorem problem_sample_is_valid : isValidSystematicSample problemSample := by
  sorry

/-- The sequence of numbers in the systematic sample -/
def sampleSequence (s : SystematicSample) : List ℕ :=
  List.range s.size |>.map (λ i => s.first + i * s.interval)

/-- Theorem stating that the sample sequence matches the given answer -/
theorem problem_sample_sequence_correct :
  sampleSequence problemSample = [3, 13, 23, 33, 43, 53] := by
  sorry

end NUMINAMATH_CALUDE_problem_sample_is_valid_problem_sample_sequence_correct_l237_23708


namespace NUMINAMATH_CALUDE_traffic_light_color_change_probability_l237_23714

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the total time when color changes occur -/
def colorChangeDuration (cycle : TrafficLightCycle) : ℕ :=
  3 * 5  -- 5 seconds for each color change

/-- Theorem: The probability of observing a color change is 3/20 -/
theorem traffic_light_color_change_probability
  (cycle : TrafficLightCycle)
  (h1 : cycle.green = 45)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 50)
  (h4 : colorChangeDuration cycle = 15) :
  (colorChangeDuration cycle : ℚ) / (cycleDuration cycle : ℚ) = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_color_change_probability_l237_23714


namespace NUMINAMATH_CALUDE_johnnys_age_reference_l237_23762

/-- Proves that Johnny was referring to 3 years ago -/
theorem johnnys_age_reference : 
  ∀ (current_age : ℕ) (years_ago : ℕ),
  current_age = 8 →
  current_age + 2 = 2 * (current_age - years_ago) →
  years_ago = 3 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_age_reference_l237_23762


namespace NUMINAMATH_CALUDE_ivan_journey_time_l237_23787

/-- Represents the journey details of Ivan and Peter --/
structure Journey where
  distance : ℝ
  ivan_speed : ℝ
  peter_speed : ℝ
  peter_wait_time : ℝ
  cafe_time : ℝ

/-- The theorem stating Ivan's total journey time --/
theorem ivan_journey_time (j : Journey) 
  (h1 : j.distance > 0)
  (h2 : j.ivan_speed > 0)
  (h3 : j.peter_speed > 0)
  (h4 : j.peter_wait_time = 10)
  (h5 : j.cafe_time = 30)
  (h6 : j.distance / (3 * j.ivan_speed) = j.distance / j.peter_speed + j.peter_wait_time)
  (h7 : j.distance / j.ivan_speed = 2 * (j.distance / j.peter_speed + j.peter_wait_time + j.cafe_time))
  : j.distance / j.ivan_speed = 75 := by
  sorry


end NUMINAMATH_CALUDE_ivan_journey_time_l237_23787


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l237_23724

theorem power_zero_eq_one (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l237_23724


namespace NUMINAMATH_CALUDE_alice_profit_l237_23700

def total_bracelets : ℕ := 52
def design_a_bracelets : ℕ := 30
def design_b_bracelets : ℕ := 22
def cost_a : ℚ := 2
def cost_b : ℚ := 4.5
def given_away_a : ℕ := 5
def given_away_b : ℕ := 3
def sell_price_a : ℚ := 0.25
def sell_price_b : ℚ := 0.5

def total_cost : ℚ := design_a_bracelets * cost_a + design_b_bracelets * cost_b
def remaining_a : ℕ := design_a_bracelets - given_away_a
def remaining_b : ℕ := design_b_bracelets - given_away_b
def total_revenue : ℚ := remaining_a * sell_price_a + remaining_b * sell_price_b
def profit : ℚ := total_revenue - total_cost

theorem alice_profit :
  profit = -143.25 :=
sorry

end NUMINAMATH_CALUDE_alice_profit_l237_23700


namespace NUMINAMATH_CALUDE_equation_solution_l237_23710

theorem equation_solution : 
  ∃! x : ℝ, 45 - (28 - (37 - (15 - x))) = 58 ∧ x = 19 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l237_23710


namespace NUMINAMATH_CALUDE_tobias_played_one_week_l237_23759

/-- Calculates the number of weeks Tobias played given the conditions of the problem -/
def tobias_weeks (nathan_hours_per_day : ℕ) (nathan_weeks : ℕ) (tobias_hours_per_day : ℕ) (total_hours : ℕ) : ℕ :=
  let nathan_total_hours := nathan_hours_per_day * 7 * nathan_weeks
  let tobias_total_hours := total_hours - nathan_total_hours
  tobias_total_hours / (tobias_hours_per_day * 7)

/-- Theorem stating that Tobias played for 1 week given the problem conditions -/
theorem tobias_played_one_week :
  tobias_weeks 3 2 5 77 = 1 := by
  sorry

#eval tobias_weeks 3 2 5 77

end NUMINAMATH_CALUDE_tobias_played_one_week_l237_23759


namespace NUMINAMATH_CALUDE_cone_base_radius_l237_23738

/-- Given a cone with slant height 5 cm and lateral surface area 15π cm², 
    prove that the radius of its base is 3 cm. -/
theorem cone_base_radius (l : ℝ) (L : ℝ) (r : ℝ) : 
  l = 5 → L = 15 * Real.pi → L = Real.pi * r * l → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l237_23738


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l237_23758

/-- A complex number is pure imaginary if its real part is zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The theorem states that if (a - i)² * i³ is a pure imaginary number,
    then the real number a must be equal to 0 -/
theorem pure_imaginary_condition (a : ℝ) :
  IsPureImaginary ((a - Complex.I)^2 * Complex.I^3) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l237_23758


namespace NUMINAMATH_CALUDE_correct_num_students_l237_23792

/-- The number of students in the class -/
def num_students : ℕ := 20

/-- The cost of one pack of instant noodles in yuan -/
def noodle_cost : ℚ := 3.5

/-- The cost of one sausage in yuan -/
def sausage_cost : ℚ := 7.5

/-- The total amount spent in yuan -/
def total_spent : ℚ := 290

/-- Theorem stating that the number of students is correct given the problem conditions -/
theorem correct_num_students :
  (num_students : ℚ) * (2 * noodle_cost + sausage_cost) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_correct_num_students_l237_23792


namespace NUMINAMATH_CALUDE_journey_distance_is_correct_l237_23784

/-- Represents the cab fare structure and journey details -/
structure CabJourney where
  baseFare : ℝ
  peakRateFirst2Miles : ℝ
  peakRateAfter2Miles : ℝ
  toll1 : ℝ
  toll2 : ℝ
  tipPercentage : ℝ
  totalPaid : ℝ

/-- Calculates the distance of the journey based on the given fare structure and total paid -/
def calculateDistance (journey : CabJourney) : ℝ :=
  sorry

/-- Theorem stating that the calculated distance for the given journey is 6.58 miles -/
theorem journey_distance_is_correct (journey : CabJourney) 
  (h1 : journey.baseFare = 3)
  (h2 : journey.peakRateFirst2Miles = 5)
  (h3 : journey.peakRateAfter2Miles = 4)
  (h4 : journey.toll1 = 1.5)
  (h5 : journey.toll2 = 2.5)
  (h6 : journey.tipPercentage = 0.15)
  (h7 : journey.totalPaid = 39.57) :
  calculateDistance journey = 6.58 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_is_correct_l237_23784


namespace NUMINAMATH_CALUDE_sculpture_height_l237_23718

/-- Converts feet to inches -/
def feet_to_inches (feet : ℝ) : ℝ := feet * 12

theorem sculpture_height :
  let base_height : ℝ := 2
  let total_height_feet : ℝ := 3
  let total_height_inches : ℝ := feet_to_inches total_height_feet
  let sculpture_height : ℝ := total_height_inches - base_height
  sculpture_height = 34 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_height_l237_23718


namespace NUMINAMATH_CALUDE_remaining_problems_to_grade_l237_23770

theorem remaining_problems_to_grade 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (h1 : total_worksheets = 25) 
  (h2 : graded_worksheets = 12) 
  (h3 : problems_per_worksheet = 15) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 195 := by
sorry

end NUMINAMATH_CALUDE_remaining_problems_to_grade_l237_23770


namespace NUMINAMATH_CALUDE_inequality_proof_l237_23785

theorem inequality_proof (a b c A α : Real) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hA : a + b + c = A) (hA1 : A ≤ 1) (hα : α > 0) : 
  (1/a - a)^α + (1/b - b)^α + (1/c - c)^α ≥ 3*(3/A - A/3)^α := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l237_23785


namespace NUMINAMATH_CALUDE_factor_expression_l237_23740

theorem factor_expression (x : ℝ) : 3*x*(x-5) + 4*(x-5) + 6*x = (3*x + 4)*(x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l237_23740


namespace NUMINAMATH_CALUDE_symmetric_function_value_l237_23752

/-- A function is symmetric to 2^(x-a) about y=-x -/
def SymmetricAboutNegativeX (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, f x = y ↔ 2^(-y - a) = -x

theorem symmetric_function_value (f : ℝ → ℝ) (a : ℝ) 
  (h_sym : SymmetricAboutNegativeX f a) 
  (h_sum : f (-2) + f (-4) = 1) : 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_function_value_l237_23752


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l237_23776

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l237_23776


namespace NUMINAMATH_CALUDE_meals_left_to_distribute_l237_23726

theorem meals_left_to_distribute (initial_meals additional_meals distributed_meals : ℕ) :
  initial_meals + additional_meals - distributed_meals =
  (initial_meals + additional_meals) - distributed_meals :=
by sorry

end NUMINAMATH_CALUDE_meals_left_to_distribute_l237_23726


namespace NUMINAMATH_CALUDE_extremum_value_l237_23798

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the derivative of f(x)
def f_prime (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem extremum_value (a b : ℝ) : 
  f a b 1 = 10 ∧ f_prime a b 1 = 0 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_extremum_value_l237_23798


namespace NUMINAMATH_CALUDE_initial_distance_between_trains_l237_23790

/-- Proves that the initial distance between two trains is 200 meters. -/
theorem initial_distance_between_trains (length1 length2 : ℝ) (speed1 speed2 : ℝ) (time : ℝ) :
  length1 = 90 →
  length2 = 100 →
  speed1 = 71 * 1000 / 3600 →
  speed2 = 89 * 1000 / 3600 →
  time = 4.499640028797696 →
  speed1 * time + speed2 * time = 200 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_between_trains_l237_23790


namespace NUMINAMATH_CALUDE_base_n_representation_of_b_l237_23767

theorem base_n_representation_of_b (n : ℕ) (a b : ℤ) (x y : ℚ) : 
  n > 9 →
  x^2 - a*x + b = 0 →
  y^2 - a*y + b = 0 →
  (x = n ∨ y = n) →
  2*x - y = 6 →
  a = 2*n + 7 →
  b = 14 :=
by sorry

end NUMINAMATH_CALUDE_base_n_representation_of_b_l237_23767


namespace NUMINAMATH_CALUDE_fraction_simplification_l237_23701

theorem fraction_simplification (b y θ : ℝ) (h : b^2 + y^2 ≠ 0) :
  (Real.sqrt (b^2 + y^2) + (y^2 - b^2) / Real.sqrt (b^2 + y^2) * Real.cos θ) / (b^2 + y^2) =
  (b^2 * (Real.sqrt (b^2 + y^2) - Real.cos θ) + y^2 * (Real.sqrt (b^2 + y^2) + Real.cos θ)) /
  (b^2 + y^2)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l237_23701


namespace NUMINAMATH_CALUDE_sum_of_two_arithmetic_sequences_l237_23717

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem sum_of_two_arithmetic_sequences :
  let seq1 := arithmetic_sequence 1 10 5
  let seq2 := arithmetic_sequence 9 10 5
  List.sum seq1 + List.sum seq2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_arithmetic_sequences_l237_23717


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l237_23711

theorem largest_n_for_equation : 
  (∃ (x y z : ℕ+), 6^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 9) ∧ 
  (∀ (n : ℕ+), n > 6 → ¬∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 9) :=
by sorry

#check largest_n_for_equation

end NUMINAMATH_CALUDE_largest_n_for_equation_l237_23711


namespace NUMINAMATH_CALUDE_cubic_polynomial_w_value_l237_23780

theorem cubic_polynomial_w_value (p q r : ℂ) (u v w : ℂ) : 
  p^3 + 5*p^2 + 7*p - 18 = 0 →
  q^3 + 5*q^2 + 7*q - 18 = 0 →
  r^3 + 5*r^2 + 7*r - 18 = 0 →
  (p+q)^3 + u*(p+q)^2 + v*(p+q) + w = 0 →
  (q+r)^3 + u*(q+r)^2 + v*(q+r) + w = 0 →
  (r+p)^3 + u*(r+p)^2 + v*(r+p) + w = 0 →
  w = 179 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_w_value_l237_23780


namespace NUMINAMATH_CALUDE_club_equation_solution_l237_23753

-- Define the operation ♣
def club (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 5

-- Theorem statement
theorem club_equation_solution :
  ∃ B : ℝ, club 4 B = 101 ∧ B = 24 := by
  sorry

end NUMINAMATH_CALUDE_club_equation_solution_l237_23753


namespace NUMINAMATH_CALUDE_simplify_fraction_l237_23713

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x - 2) / (x^2 - 2*x + 1)) = -x^2 - x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l237_23713


namespace NUMINAMATH_CALUDE_mean_of_cubic_solutions_l237_23741

theorem mean_of_cubic_solutions (x : ℝ) :
  x^3 + 2*x^2 - 13*x - 10 = 0 →
  ∃ (s : Finset ℝ), s.card = 3 ∧ (∀ y ∈ s, y^3 + 2*y^2 - 13*y - 10 = 0) ∧
  (s.sum id) / s.card = -1 :=
sorry

end NUMINAMATH_CALUDE_mean_of_cubic_solutions_l237_23741


namespace NUMINAMATH_CALUDE_fixed_point_on_linear_function_l237_23723

/-- Given a linear function y = kx + b where 3k - b = 2, 
    prove that the point (-3, -2) lies on the graph of the function. -/
theorem fixed_point_on_linear_function (k b : ℝ) 
  (h : 3 * k - b = 2) : 
  k * (-3) + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_linear_function_l237_23723


namespace NUMINAMATH_CALUDE_twentieth_fisherman_catch_l237_23768

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) 
  (fish_per_nineteen : ℕ) (h1 : total_fishermen = 20) 
  (h2 : total_fish = 10000) (h3 : fish_per_nineteen = 400) : 
  total_fish - (total_fishermen - 1) * fish_per_nineteen = 2400 :=
by
  sorry

#check twentieth_fisherman_catch

end NUMINAMATH_CALUDE_twentieth_fisherman_catch_l237_23768


namespace NUMINAMATH_CALUDE_total_jellybeans_l237_23778

def dozen : ℕ := 12

def caleb_jellybeans : ℕ := 3 * dozen

def sophie_jellybeans : ℕ := caleb_jellybeans / 2

theorem total_jellybeans : caleb_jellybeans + sophie_jellybeans = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_l237_23778


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l237_23751

theorem simultaneous_equations_solution (m : ℝ) : 
  (m ≠ 1) ↔ (∃ x y : ℝ, y = m * x + 3 ∧ y = (2 * m - 1) * x + 4) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l237_23751


namespace NUMINAMATH_CALUDE_path_of_vertex_A_l237_23748

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  ab : ℝ
  bc : ℝ

/-- Calculates the path traveled by vertex A of a rectangle when rotated 90° around D and translated -/
def pathTraveledByA (rect : Rectangle) (rotationAngle : ℝ) (translation : ℝ) : ℝ :=
  sorry

/-- Theorem stating the path traveled by vertex A of the specific rectangle -/
theorem path_of_vertex_A :
  let rect : Rectangle := { ab := 3, bc := 5 }
  let rotationAngle : ℝ := π / 2  -- 90° in radians
  let translation : ℝ := 3
  pathTraveledByA rect rotationAngle translation = 2.5 * π + 3 := by
  sorry

end NUMINAMATH_CALUDE_path_of_vertex_A_l237_23748


namespace NUMINAMATH_CALUDE_sixteen_fifth_equals_four_tenth_l237_23703

theorem sixteen_fifth_equals_four_tenth : 16^5 = 4^10 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_fifth_equals_four_tenth_l237_23703


namespace NUMINAMATH_CALUDE_function_properties_l237_23706

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 + b * x

-- Define the function g
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x + (3 * a * x^2 + 2 * x + b)

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = -g a b (-x)) →  -- g is an odd function
  (∃ C, ∀ x, f a b x = -1/3 * x^3 + x^2 + C) ∧ 
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≤ 4 * Real.sqrt 2 / 3) ∧
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≥ 4 / 3) ∧
  (g (-1/3) 0 (Real.sqrt 2) = 4 * Real.sqrt 2 / 3) ∧
  (g (-1/3) 0 2 = 4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l237_23706


namespace NUMINAMATH_CALUDE_red_boxcars_count_l237_23732

/-- The number of blue boxcars -/
def num_blue_boxcars : ℕ := 4

/-- The number of black boxcars -/
def num_black_boxcars : ℕ := 7

/-- The capacity of a black boxcar in pounds -/
def black_boxcar_capacity : ℕ := 4000

/-- The capacity of a blue boxcar in pounds -/
def blue_boxcar_capacity : ℕ := 2 * black_boxcar_capacity

/-- The capacity of a red boxcar in pounds -/
def red_boxcar_capacity : ℕ := 3 * blue_boxcar_capacity

/-- The total capacity of all boxcars in pounds -/
def total_capacity : ℕ := 132000

/-- The number of red boxcars -/
def num_red_boxcars : ℕ := 
  (total_capacity - num_black_boxcars * black_boxcar_capacity - num_blue_boxcars * blue_boxcar_capacity) / red_boxcar_capacity

theorem red_boxcars_count : num_red_boxcars = 3 := by
  sorry

end NUMINAMATH_CALUDE_red_boxcars_count_l237_23732


namespace NUMINAMATH_CALUDE_sqrt_of_sum_of_squares_plus_seven_l237_23791

theorem sqrt_of_sum_of_squares_plus_seven (a b : ℝ) : 
  a = Real.sqrt 5 + 2 → b = Real.sqrt 5 - 2 → Real.sqrt (a^2 + b^2 + 7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sum_of_squares_plus_seven_l237_23791


namespace NUMINAMATH_CALUDE_sin_increasing_omega_range_l237_23736

theorem sin_increasing_omega_range (ω : ℝ) (f : ℝ → ℝ) :
  ω > 0 →
  (∀ x ∈ Set.Icc 0 (π / 3), f x = Real.sin (ω * x)) →
  StrictMonoOn f (Set.Icc 0 (π / 3)) →
  ω ∈ Set.Ioo 0 (3 / 2) :=
sorry

end NUMINAMATH_CALUDE_sin_increasing_omega_range_l237_23736


namespace NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l237_23799

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l237_23799


namespace NUMINAMATH_CALUDE_smallest_root_property_l237_23731

theorem smallest_root_property : ∃ a : ℝ, 
  (∀ x : ℝ, x^2 - 9*x - 10 = 0 → a ≤ x) ∧ 
  (a^2 - 9*a - 10 = 0) ∧
  (a^4 - 909*a = 910) := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_property_l237_23731


namespace NUMINAMATH_CALUDE_inequality_implication_l237_23788

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l237_23788


namespace NUMINAMATH_CALUDE_inequality_solution_l237_23783

theorem inequality_solution (x : ℝ) : 
  x ≠ 0 → (x > (9 : ℝ) / x ↔ (x > -3 ∧ x < 0) ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l237_23783


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l237_23786

/-- A color representation --/
inductive Color
| Black
| White

/-- A grid of colors --/
def Grid := Fin 3 → Fin 7 → Color

/-- A rectangle in the grid --/
structure Rectangle where
  x1 : Fin 7
  x2 : Fin 7
  y1 : Fin 3
  y2 : Fin 3
  h_distinct : x1 ≠ x2 ∧ y1 ≠ y2

/-- Check if a rectangle has vertices of the same color --/
def sameColorVertices (g : Grid) (r : Rectangle) : Prop :=
  g r.y1 r.x1 = g r.y1 r.x2 ∧
  g r.y1 r.x1 = g r.y2 r.x1 ∧
  g r.y1 r.x1 = g r.y2 r.x2

/-- Theorem: In any 3x7 grid coloring, there exists a rectangle with vertices of the same color --/
theorem exists_same_color_rectangle (g : Grid) : ∃ r : Rectangle, sameColorVertices g r := by
  sorry

end NUMINAMATH_CALUDE_exists_same_color_rectangle_l237_23786


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l237_23795

theorem area_between_concentric_circles : 
  let r₁ : ℝ := 12
  let r₂ : ℝ := 7
  (π * r₁^2 - π * r₂^2) = 95 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l237_23795


namespace NUMINAMATH_CALUDE_f_odd_and_periodic_l237_23761

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def condition1 (f : ℝ → ℝ) : Prop := ∀ x, f (10 + x) = f (10 - x)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x, f (20 + x) = -f (20 - x)

-- State the theorem
theorem f_odd_and_periodic (h1 : condition1 f) (h2 : condition2 f) :
  (∀ x, f (x + 40) = f x) ∧ (∀ x, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_periodic_l237_23761


namespace NUMINAMATH_CALUDE_sunflower_height_feet_l237_23777

def feet_to_inches (feet : ℕ) : ℕ := feet * 12

def sister_height_inches : ℕ := feet_to_inches 4 + 3

def sunflower_height_inches : ℕ := sister_height_inches + 21

def inches_to_feet (inches : ℕ) : ℕ := inches / 12

theorem sunflower_height_feet :
  inches_to_feet sunflower_height_inches = 6 :=
sorry

end NUMINAMATH_CALUDE_sunflower_height_feet_l237_23777


namespace NUMINAMATH_CALUDE_B_power_101_l237_23707

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 0]]

theorem B_power_101 : B ^ 101 = B := by sorry

end NUMINAMATH_CALUDE_B_power_101_l237_23707


namespace NUMINAMATH_CALUDE_planes_perpendicular_l237_23763

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersects : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular
  (α β : Plane) (a b c : Line)
  (h1 : contains α a)
  (h2 : contains α b)
  (h3 : intersects a b)
  (h4 : perpendicular c a)
  (h5 : perpendicular c b)
  (h6 : parallel c β) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l237_23763


namespace NUMINAMATH_CALUDE_combined_average_score_l237_23796

/-- Combined average score of two groups given their individual averages and size ratio -/
theorem combined_average_score 
  (morning_avg : ℝ) 
  (evening_avg : ℝ) 
  (morning_students : ℝ) 
  (evening_students : ℝ) 
  (h1 : morning_avg = 82)
  (h2 : evening_avg = 68)
  (h3 : morning_students / evening_students = 5 / 7) :
  (morning_avg * morning_students + evening_avg * evening_students) / (morning_students + evening_students) = 72 :=
by sorry

end NUMINAMATH_CALUDE_combined_average_score_l237_23796


namespace NUMINAMATH_CALUDE_pages_multiple_l237_23743

theorem pages_multiple (beatrix_pages cristobal_extra_pages : ℕ) 
  (h1 : beatrix_pages = 704)
  (h2 : cristobal_extra_pages = 1423)
  (h3 : ∃ x : ℕ, x * beatrix_pages + 15 = cristobal_extra_pages) :
  ∃ x : ℕ, x * beatrix_pages + 15 = cristobal_extra_pages ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_pages_multiple_l237_23743


namespace NUMINAMATH_CALUDE_hua_method_is_golden_ratio_l237_23769

/-- Represents the possible methods used in optimal selection -/
inductive OptimalSelectionMethod
  | GoldenRatio
  | Mean
  | Mode
  | Median

/-- The optimal selection method popularized by Hua Luogeng -/
def huaMethod : OptimalSelectionMethod := OptimalSelectionMethod.GoldenRatio

/-- Theorem stating that Hua Luogeng's optimal selection method uses the Golden ratio -/
theorem hua_method_is_golden_ratio :
  huaMethod = OptimalSelectionMethod.GoldenRatio :=
by sorry

end NUMINAMATH_CALUDE_hua_method_is_golden_ratio_l237_23769


namespace NUMINAMATH_CALUDE_cubic_sum_value_l237_23722

theorem cubic_sum_value (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 10) :
  Complex.abs (w^3 + z^3) = 26 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_value_l237_23722


namespace NUMINAMATH_CALUDE_triangle_prime_sides_area_not_integer_l237_23737

theorem triangle_prime_sides_area_not_integer 
  (a b c : ℕ) 
  (ha : Prime a) 
  (hb : Prime b) 
  (hc : Prime c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ¬ (∃ S : ℕ, S^2 * 16 = (a + b + c) * ((a + b + c) - 2*a) * ((a + b + c) - 2*b) * ((a + b + c) - 2*c)) :=
sorry

end NUMINAMATH_CALUDE_triangle_prime_sides_area_not_integer_l237_23737


namespace NUMINAMATH_CALUDE_equation_proof_l237_23781

theorem equation_proof : 169 + 2 * 13 * 7 + 49 = 400 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l237_23781


namespace NUMINAMATH_CALUDE_product_of_roots_l237_23747

theorem product_of_roots (x : ℝ) : 
  (2 * x^3 - 24 * x^2 + 96 * x + 56 = 0) → 
  (∃ r₁ r₂ r₃ : ℝ, (x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ * r₂ * r₃ = -28) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l237_23747


namespace NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l237_23728

/-- A symmetric scanning code is a 5x5 grid that remains unchanged when rotated by multiples of 90° or reflected across diagonal or midpoint lines. -/
def SymmetricScanningCode : Type := Unit

/-- The number of distinct symmetry groups in a 5x5 symmetric scanning code -/
def numSymmetryGroups : ℕ := 5

/-- The number of color choices for each symmetry group -/
def numColorChoices : ℕ := 2

/-- The total number of color combinations for all symmetry groups -/
def totalColorCombinations : ℕ := numColorChoices ^ numSymmetryGroups

/-- The number of invalid color combinations (all white or all black) -/
def invalidColorCombinations : ℕ := 2

theorem count_symmetric_scanning_codes :
  (totalColorCombinations - invalidColorCombinations : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l237_23728


namespace NUMINAMATH_CALUDE_initial_workers_count_l237_23754

/-- Represents the productivity of workers in digging holes -/
structure DiggingProductivity where
  initialWorkers : ℕ
  initialDepth : ℝ
  initialTime : ℝ
  newDepth : ℝ
  newTime : ℝ
  extraWorkers : ℕ

/-- Proves that the initial number of workers is 45 given the conditions -/
theorem initial_workers_count (p : DiggingProductivity) 
  (h1 : p.initialDepth = 30)
  (h2 : p.initialTime = 8)
  (h3 : p.newDepth = 45)
  (h4 : p.newTime = 6)
  (h5 : p.extraWorkers = 45)
  (h6 : p.initialWorkers > 0)
  (h7 : p.initialDepth > 0)
  (h8 : p.initialTime > 0)
  (h9 : p.newDepth > 0)
  (h10 : p.newTime > 0) :
  p.initialWorkers = 45 := by
  sorry


end NUMINAMATH_CALUDE_initial_workers_count_l237_23754


namespace NUMINAMATH_CALUDE_max_elevation_l237_23721

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 200 * t - 20 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (t' : ℝ), s t' ≤ s t ∧ s t = 500 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_l237_23721


namespace NUMINAMATH_CALUDE_line_parallel_plane_l237_23782

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)
variable (parallelPlane : Plane → Plane → Prop)

-- Define the intersection of planes
variable (intersect : Plane → Plane → Line)

-- Define the "not subset of" relation for a line and a plane
variable (notSubset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane 
  (l m : Line) (α β : Plane) 
  (h1 : intersect α β = m) 
  (h2 : notSubset l α) 
  (h3 : parallelLine l m) : 
  parallelLinePlane l α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_l237_23782


namespace NUMINAMATH_CALUDE_parking_rate_proof_l237_23760

/-- Proves that the monthly parking rate is $35 given the conditions --/
theorem parking_rate_proof (weekly_rate : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) 
  (yearly_savings : ℕ) (monthly_rate : ℕ) : 
  weekly_rate = 10 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  yearly_savings = 100 →
  (weeks_per_year * weekly_rate) - (months_per_year * monthly_rate) = yearly_savings →
  monthly_rate = 35 := by
sorry

end NUMINAMATH_CALUDE_parking_rate_proof_l237_23760


namespace NUMINAMATH_CALUDE_households_with_only_bike_l237_23793

/-- Proves the number of households with only a bike given the provided information -/
theorem households_with_only_bike
  (total_households : ℕ)
  (households_without_car_or_bike : ℕ)
  (households_with_both_car_and_bike : ℕ)
  (households_with_car : ℕ)
  (h1 : total_households = 90)
  (h2 : households_without_car_or_bike = 11)
  (h3 : households_with_both_car_and_bike = 22)
  (h4 : households_with_car = 44) :
  total_households - households_without_car_or_bike - households_with_car - households_with_both_car_and_bike = 35 :=
by sorry

end NUMINAMATH_CALUDE_households_with_only_bike_l237_23793


namespace NUMINAMATH_CALUDE_stamps_problem_l237_23719

theorem stamps_problem (A B C D : ℕ) : 
  A + B + C + D = 251 →
  A = 2 * B + 2 →
  A = 3 * C + 6 →
  A = 4 * D - 16 →
  D = 32 := by
sorry

end NUMINAMATH_CALUDE_stamps_problem_l237_23719


namespace NUMINAMATH_CALUDE_arrangements_eq_36_l237_23705

/-- The number of students in the row -/
def n : ℕ := 5

/-- A function that calculates the number of arrangements given the conditions -/
def arrangements (n : ℕ) : ℕ :=
  let positions := n - 1  -- Possible positions for A (excluding ends)
  let pairs := 2  -- A and B can be arranged in 2 ways next to each other
  let others := n - 2  -- Remaining students to arrange
  positions * pairs * (others.factorial)

/-- The theorem stating that the number of arrangements is 36 -/
theorem arrangements_eq_36 : arrangements n = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_eq_36_l237_23705


namespace NUMINAMATH_CALUDE_sum_of_squares_l237_23749

theorem sum_of_squares (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 1) (h5 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5 + 1) :
  a^2 + b^2 + c^2 = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l237_23749


namespace NUMINAMATH_CALUDE_marble_distribution_l237_23766

theorem marble_distribution (n : ℕ) (hn : n = 450) :
  (Finset.filter (fun m : ℕ => m > 1 ∧ n / m > 1) (Finset.range (n + 1))).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l237_23766


namespace NUMINAMATH_CALUDE_fewest_students_twenty_two_satisfies_fewest_students_is_22_l237_23702

theorem fewest_students (n : ℕ) : (n % 5 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6) → n ≥ 22 :=
by sorry

theorem twenty_two_satisfies : 22 % 5 = 2 ∧ 22 % 6 = 4 ∧ 22 % 8 = 6 :=
by sorry

theorem fewest_students_is_22 : 
  ∃ n : ℕ, n % 5 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ ∀ m : ℕ, (m % 5 = 2 ∧ m % 6 = 4 ∧ m % 8 = 6) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_fewest_students_twenty_two_satisfies_fewest_students_is_22_l237_23702


namespace NUMINAMATH_CALUDE_resulting_polygon_has_24_sides_l237_23720

/-- Calculates the number of sides in the resulting polygon formed by sequentially 
    attaching regular polygons from triangle to octagon. -/
def resulting_polygon_sides : ℕ :=
  let initial_triangle := 3
  let square_addition := 4 - 2
  let pentagon_addition := 5 - 2
  let hexagon_addition := 6 - 2
  let heptagon_addition := 7 - 2
  let octagon_addition := 8 - 1
  initial_triangle + square_addition + pentagon_addition + 
  hexagon_addition + heptagon_addition + octagon_addition

/-- The resulting polygon has 24 sides. -/
theorem resulting_polygon_has_24_sides : resulting_polygon_sides = 24 := by
  sorry

end NUMINAMATH_CALUDE_resulting_polygon_has_24_sides_l237_23720


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l237_23730

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x = k / (y ^ 2)) 
  (h2 : 1 = k / (2 ^ 2)) (h3 : 0.1111111111111111 = k / (y ^ 2)) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l237_23730


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l237_23734

theorem parabola_vertex_y_coordinate (x y : ℝ) :
  y = -6 * x^2 + 24 * x - 7 →
  ∃ h k : ℝ, y = -6 * (x - h)^2 + k ∧ k = 17 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l237_23734


namespace NUMINAMATH_CALUDE_candy_distribution_l237_23789

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) (h1 : total_candy = 30) (h2 : num_friends = 4) :
  total_candy - (total_candy / num_friends) * num_friends = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l237_23789


namespace NUMINAMATH_CALUDE_cos_A_eq_11_15_l237_23779

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_A_eq_C (q : Quadrilateral) : Prop :=
  sorry

def side_AB_eq_150 (q : Quadrilateral) : Prop :=
  sorry

def side_CD_eq_150 (q : Quadrilateral) : Prop :=
  sorry

def side_AD_ne_BC (q : Quadrilateral) : Prop :=
  sorry

def perimeter_eq_520 (q : Quadrilateral) : Prop :=
  sorry

-- Define cos A
def cos_A (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem cos_A_eq_11_15 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_angle : angle_A_eq_C q)
  (h_AB : side_AB_eq_150 q)
  (h_CD : side_CD_eq_150 q)
  (h_AD_ne_BC : side_AD_ne_BC q)
  (h_perimeter : perimeter_eq_520 q) :
  cos_A q = 11/15 := by sorry

end NUMINAMATH_CALUDE_cos_A_eq_11_15_l237_23779


namespace NUMINAMATH_CALUDE_range_of_a_l237_23744

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a + 2 = 0}

-- State the theorem
theorem range_of_a (a : ℝ) : B a ⊆ A → a ∈ Set.Ioo (-1 : ℝ) (18/7 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l237_23744


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l237_23712

/-- The eccentricity of a hyperbola given its equation and asymptote angle -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1 →
  (angle_between_asymptotes : ℝ) → angle_between_asymptotes = π / 3 →
  ∃ (e : ℝ), (e = 2*Real.sqrt 3/3 ∨ e = 2) ∧ 
  e^2 * a^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l237_23712


namespace NUMINAMATH_CALUDE_hypotenuse_length_l237_23733

-- Define the points A and B
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (-2, 4)
def O : ℝ × ℝ := (0, 0)

-- Define the properties
theorem hypotenuse_length :
  -- A and B are on the graph of y = x^2
  (A.2 = A.1^2) →
  (B.2 = B.1^2) →
  -- Triangle ABO forms a right triangle at O
  ((A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0) →
  -- A and B are symmetric about the y-axis
  (A.1 = -B.1) →
  (A.2 = B.2) →
  -- The x-coordinate of A is 2
  (A.1 = 2) →
  -- The length of hypotenuse AB is 4
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l237_23733


namespace NUMINAMATH_CALUDE_power_mod_eleven_l237_23771

theorem power_mod_eleven : 3^251 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l237_23771


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l237_23746

/-- The line ρ cos θ + 2ρ sin θ = 1 in polar coordinates -/
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ + 2 * ρ * Real.sin θ = 1

/-- The third quadrant in Cartesian coordinates -/
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem: The line ρ cos θ + 2ρ sin θ = 1 does not pass through the third quadrant -/
theorem line_not_in_third_quadrant :
  ¬∃ (x y : ℝ), (∃ (ρ θ : ℝ), polar_line ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ third_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l237_23746


namespace NUMINAMATH_CALUDE_simplify_exponential_fraction_l237_23729

theorem simplify_exponential_fraction (n : ℕ) :
  (3^(n+5) - 3 * 3^n) / (3 * 3^(n+4)) = 240 / 81 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponential_fraction_l237_23729


namespace NUMINAMATH_CALUDE_number_problem_l237_23709

theorem number_problem (x : ℝ) : 35 + 3 * x = 56 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l237_23709


namespace NUMINAMATH_CALUDE_range_of_a_l237_23775

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (4 * x - 3) ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the condition that ¬p is a necessary but not sufficient condition for ¬q
def condition (a : ℝ) : Prop :=
  (∀ x : ℝ, q x a → p x) ∧ (∃ x : ℝ, p x ∧ ¬(q x a))

-- State the theorem
theorem range_of_a (a : ℝ) :
  condition a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l237_23775


namespace NUMINAMATH_CALUDE_gcd_72_120_168_l237_23735

theorem gcd_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by sorry

end NUMINAMATH_CALUDE_gcd_72_120_168_l237_23735


namespace NUMINAMATH_CALUDE_min_length_AB_l237_23765

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Define the chord MN
def chord_MN (M N : ℝ × ℝ) : Prop := 
  circle_C M.1 M.2 ∧ circle_C N.1 N.2

-- Define the perpendicularity condition
def perpendicular_CM_CN (C M N : ℝ × ℝ) : Prop := 
  (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2) = 0

-- Define the midpoint condition
def midpoint_P (P M N : ℝ × ℝ) : Prop := 
  P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- Define the angle condition
def angle_APB_geq_pi_div_2 (A P B : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) ≤ 0

-- Main theorem
theorem min_length_AB : 
  ∀ (M N P A B : ℝ × ℝ),
    chord_MN M N →
    perpendicular_CM_CN (2, 4) M N →
    midpoint_P P M N →
    line_l A.1 A.2 →
    line_l B.1 B.2 →
    angle_APB_geq_pi_div_2 A P B →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ ((6 * Real.sqrt 5) / 5 + 2)^2 :=
sorry

end NUMINAMATH_CALUDE_min_length_AB_l237_23765


namespace NUMINAMATH_CALUDE_marble_problem_solution_l237_23772

/-- Represents the number of marbles of each color in a box -/
structure MarbleBox where
  red : ℕ
  green : ℕ
  yellow : ℕ
  other : ℕ

/-- The conditions of the marble problem -/
def marble_problem (box : MarbleBox) : Prop :=
  box.red = 20 ∧
  box.green = 3 * box.red ∧
  box.yellow = box.green / 5 ∧
  box.red + box.green + box.yellow + box.other = 4 * box.green

/-- The theorem stating the solution to the marble problem -/
theorem marble_problem_solution (box : MarbleBox) :
  marble_problem box → box.other = 148 := by
  sorry


end NUMINAMATH_CALUDE_marble_problem_solution_l237_23772


namespace NUMINAMATH_CALUDE_inequality_of_powers_l237_23773

theorem inequality_of_powers (a b c d x : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d ≥ 0) 
  (h5 : a + d = b + c) (h6 : x > 0) : 
  x^a + x^d ≥ x^b + x^c := by
sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l237_23773


namespace NUMINAMATH_CALUDE_arithmetic_perfect_power_sequence_exists_l237_23756

/-- An arithmetic sequence of perfect powers -/
def ArithmeticPerfectPowerSequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ (d : ℕ), ∀ i j : ℕ, i < n ∧ j < n →
    (∃ (base exponent : ℕ), exponent > 1 ∧ a i = base ^ exponent) ∧
    (a j - a i = d * (j - i))

theorem arithmetic_perfect_power_sequence_exists :
  ∃ (a : ℕ → ℕ), ArithmeticPerfectPowerSequence a 2003 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_perfect_power_sequence_exists_l237_23756


namespace NUMINAMATH_CALUDE_product_defect_rate_l237_23774

theorem product_defect_rate (stage1_defect_rate stage2_defect_rate : ℝ) 
  (h1 : stage1_defect_rate = 0.1)
  (h2 : stage2_defect_rate = 0.03) :
  1 - (1 - stage1_defect_rate) * (1 - stage2_defect_rate) = 0.127 := by
  sorry

end NUMINAMATH_CALUDE_product_defect_rate_l237_23774


namespace NUMINAMATH_CALUDE_equation_solution_l237_23757

theorem equation_solution : ∃ y : ℚ, y + 5/8 = 2/9 + 1/2 ∧ y = 7/72 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l237_23757


namespace NUMINAMATH_CALUDE_max_value_of_objective_function_l237_23739

theorem max_value_of_objective_function (x y : ℤ) : 
  x + 2*y - 5 ≤ 0 → 
  x - y - 2 ≤ 0 → 
  x ≥ 0 → 
  2*x + 3*y + 1 ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_objective_function_l237_23739
