import Mathlib

namespace intersection_union_when_a_2_complement_intersection_condition_l3380_338067

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}

def B (a : ℝ) : Set ℝ := {x | |x| < a}

theorem intersection_union_when_a_2 :
  A ∩ B 2 = {x | 1/2 ≤ x ∧ x < 2} ∧
  A ∪ B 2 = {x | -2 < x ∧ x ≤ 3} := by sorry

theorem complement_intersection_condition (a : ℝ) :
  (Aᶜ ∩ B a = B a) ↔ a ≤ 1/2 := by sorry

end intersection_union_when_a_2_complement_intersection_condition_l3380_338067


namespace characterization_of_n_l3380_338045

/-- A positive integer is square-free if it is not divisible by any perfect square greater than 1 -/
def IsSquareFree (n : ℕ+) : Prop :=
  ∀ (d : ℕ+), d * d ∣ n → d = 1

/-- The condition that for all positive integers x and y, if n divides x^n - y^n, then n^2 divides x^n - y^n -/
def Condition (n : ℕ+) : Prop :=
  ∀ (x y : ℕ+), n ∣ (x ^ n.val - y ^ n.val) → n.val * n.val ∣ (x ^ n.val - y ^ n.val)

/-- The main theorem stating the characterization of n satisfying the condition -/
theorem characterization_of_n (n : ℕ+) :
  Condition n ↔ (∃ (m : ℕ+), IsSquareFree m ∧ (n = m ∨ n = 2 * m)) :=
sorry

end characterization_of_n_l3380_338045


namespace g_composition_3_l3380_338039

def g : ℕ → ℕ
| x => if x % 2 = 0 then x / 2
       else if x < 10 then 3 * x + 2
       else x - 1

theorem g_composition_3 : g (g (g (g (g 3)))) = 16 := by sorry

end g_composition_3_l3380_338039


namespace series_sum_equals_one_l3380_338040

/-- The sum of the series ∑(n=1 to ∞) (4n-3)/3^n is equal to 1 -/
theorem series_sum_equals_one :
  (∑' n : ℕ, (4 * n - 3 : ℝ) / 3^n) = 1 := by sorry

end series_sum_equals_one_l3380_338040


namespace no_positive_integer_solutions_for_modified_quadratic_l3380_338048

theorem no_positive_integer_solutions_for_modified_quadratic :
  ∀ A : ℕ, 1 ≤ A → A ≤ 9 →
    ¬∃ x : ℕ, x > 0 ∧ x^2 - (10 * A + 1) * x + (10 * A + A) = 0 :=
by sorry

end no_positive_integer_solutions_for_modified_quadratic_l3380_338048


namespace min_value_expression_l3380_338028

theorem min_value_expression (c : ℝ) (a b : ℝ) (hc : c > 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h_eq : 4 * a^2 - 2 * a * b + b^2 - c = 0)
  (h_max : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → 4 * x^2 - 2 * x * y + y^2 - c = 0 → |2 * x + y| ≤ |2 * a + b|) :
  ∃ (k : ℝ), k = 1/a + 2/b + 4/c ∧ k ≥ -1 ∧ (∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z > 0 →
    4 * x^2 - 2 * x * y + y^2 - z = 0 → 1/x + 2/y + 4/z ≥ k) := by
  sorry

end min_value_expression_l3380_338028


namespace sector_area_l3380_338090

theorem sector_area (r a b : ℝ) : 
  r = 1 →  -- radius is 1 cm
  a = 1 →  -- arc length is 1 cm
  b = (1/2) * r * a →  -- area formula for a sector
  b = 1/2  -- the area of the sector is 1/2 cm²
:= by sorry

end sector_area_l3380_338090


namespace apples_sold_l3380_338085

/-- The amount of apples sold in a store --/
theorem apples_sold (kidney : ℕ) (golden : ℕ) (canada : ℕ) (left : ℕ) : 
  kidney + golden + canada - left = (kidney + golden + canada) - left :=
by sorry

end apples_sold_l3380_338085


namespace percentage_not_sold_l3380_338062

def initial_stock : ℕ := 1200
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
def books_not_sold : ℕ := initial_stock - total_sales

theorem percentage_not_sold : 
  (books_not_sold : ℚ) / initial_stock * 100 = 66.5 := by sorry

end percentage_not_sold_l3380_338062


namespace geometric_sequence_product_l3380_338058

/-- Given a positive geometric sequence {a_n}, prove that a_8 * a_12 = 16,
    where a_1 and a_19 are the roots of x^2 - 10x + 16 = 0 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = a n * r) →
  (a 1)^2 - 10 * (a 1) + 16 = 0 →
  (a 19)^2 - 10 * (a 19) + 16 = 0 →
  a 8 * a 12 = 16 := by
sorry

end geometric_sequence_product_l3380_338058


namespace sqrt_of_neg_two_squared_l3380_338002

theorem sqrt_of_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end sqrt_of_neg_two_squared_l3380_338002


namespace kens_height_l3380_338086

/-- Given the heights of Ivan and Jackie, and the relationship between the averages,
    prove Ken's height. -/
theorem kens_height (h_ivan : ℝ) (h_jackie : ℝ) (h_ken : ℝ) :
  h_ivan = 175 →
  h_jackie = 175 →
  (h_ivan + h_jackie + h_ken) / 3 = 1.04 * (h_ivan + h_jackie) / 2 →
  h_ken = 196 := by
  sorry

end kens_height_l3380_338086


namespace traffic_class_drunk_drivers_l3380_338054

theorem traffic_class_drunk_drivers :
  ∀ (drunk_drivers speeders : ℕ),
    drunk_drivers + speeders = 45 →
    speeders = 7 * drunk_drivers - 3 →
    drunk_drivers = 6 := by
sorry

end traffic_class_drunk_drivers_l3380_338054


namespace quadratic_and_system_solution_l3380_338018

theorem quadratic_and_system_solution :
  (∃ x₁ x₂ : ℚ, (4 * (x₁ - 1)^2 - 25 = 0 ∧ x₁ = 7/2) ∧
                (4 * (x₂ - 1)^2 - 25 = 0 ∧ x₂ = -3/2)) ∧
  (∃ x y : ℚ, (2*x - y = 4 ∧ 3*x + 2*y = 1) ∧ x = 9/7 ∧ y = -10/7) := by
  sorry

end quadratic_and_system_solution_l3380_338018


namespace nested_sqrt_calculation_l3380_338023

theorem nested_sqrt_calculation : Real.sqrt (32 * Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4))) = 16 := by
  sorry

end nested_sqrt_calculation_l3380_338023


namespace sum_of_coefficients_l3380_338001

theorem sum_of_coefficients (n : ℕ) : 
  (∀ x : ℝ, x ≠ 0 → (3 * x^2 + 1/x)^n = 256 → n = 4) := by
  sorry

end sum_of_coefficients_l3380_338001


namespace mixed_fraction_calculation_l3380_338070

theorem mixed_fraction_calculation : 
  (((5:ℚ)/2 - 10/3)^2) / ((17:ℚ)/4 + 7/6) = 5/39 := by sorry

end mixed_fraction_calculation_l3380_338070


namespace symmetric_axis_of_sine_function_l3380_338096

/-- Given a function y = 2sin(2x + φ) where |φ| < π/2, and the graph passes through (0, √3),
    prove that one symmetric axis of the graph is x = π/12 -/
theorem symmetric_axis_of_sine_function (φ : ℝ) (h1 : |φ| < π/2) 
    (h2 : 2 * Real.sin φ = Real.sqrt 3) : 
  ∃ (k : ℤ), π/12 = k * π/2 + π/4 - φ/2 := by
sorry

end symmetric_axis_of_sine_function_l3380_338096


namespace curve_is_part_of_ellipse_l3380_338016

-- Define the curve
def curve (x y : ℝ) : Prop := x = Real.sqrt (1 - 4 * y^2)

-- Define an ellipse
def is_ellipse (x y : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem curve_is_part_of_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), curve x y → x ≥ 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end curve_is_part_of_ellipse_l3380_338016


namespace fifth_term_is_seven_l3380_338072

/-- An arithmetic sequence with first term a, common difference d, and n-th term given by a + (n-1)d -/
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

/-- The second term of the sequence -/
def x : ℝ := 1

/-- Given an arithmetic sequence where the first three terms are -1, x, and 3, 
    the fifth term of this sequence is 7 -/
theorem fifth_term_is_seven :
  let a := -1
  let d := x - a
  arithmetic_sequence a d 5 = 7 := by sorry

end fifth_term_is_seven_l3380_338072


namespace triangle_property_l3380_338088

theorem triangle_property (A B C : Real) (a b c : Real) :
  2 * Real.sin (2 * A) * Real.cos A - Real.sin (3 * A) + Real.sqrt 3 * Real.cos A = Real.sqrt 3 →
  a = 1 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sin (2 * C) →
  A = π / 3 ∧ (1 / 2) * a * b * Real.sin C = Real.sqrt 3 / 6 :=
by sorry


end triangle_property_l3380_338088


namespace counterexample_exists_l3380_338093

theorem counterexample_exists : ∃ (a b c : ℝ), a > b ∧ ¬(a * c > b * c) := by
  sorry

end counterexample_exists_l3380_338093


namespace x_power_x_power_x_at_3_l3380_338037

theorem x_power_x_power_x_at_3 :
  let x : ℝ := 3
  (x^x)^(x^x) = 27^27 := by
  sorry

end x_power_x_power_x_at_3_l3380_338037


namespace stating_crabapple_sequence_count_l3380_338069

/-- Represents the number of students in the class -/
def num_students : ℕ := 11

/-- Represents the number of days the class meets -/
def num_days : ℕ := 3

/-- 
  Calculates the number of possible sequences for selecting students to receive a crabapple,
  given that no student can be selected on consecutive days.
-/
def crabapple_sequences (n : ℕ) (d : ℕ) : ℕ :=
  if d = 1 then n
  else if d = 2 then n * (n - 1)
  else n * (n - 1) * (n - 1)

/-- 
  Theorem stating that the number of possible sequences for selecting students
  to receive a crabapple over three days in a class of 11 students,
  where no student can be selected on consecutive days, is 1100.
-/
theorem crabapple_sequence_count :
  crabapple_sequences num_students num_days = 1100 := by
  sorry

end stating_crabapple_sequence_count_l3380_338069


namespace gumball_probability_l3380_338066

theorem gumball_probability (blue_twice_prob : ℚ) 
  (h1 : blue_twice_prob = 16/49) : 
  let blue_prob : ℚ := (blue_twice_prob.sqrt)
  let pink_prob : ℚ := 1 - blue_prob
  pink_prob = 3/7 := by
  sorry

end gumball_probability_l3380_338066


namespace abs_x_less_than_2_sufficient_not_necessary_l3380_338087

theorem abs_x_less_than_2_sufficient_not_necessary :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  ¬(∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) := by
sorry

end abs_x_less_than_2_sufficient_not_necessary_l3380_338087


namespace least_possible_QGK_l3380_338059

theorem least_possible_QGK : ∃ (G Q K : ℕ),
  (G ≥ 1 ∧ G ≤ 9) ∧
  (Q ≥ 0 ∧ Q ≤ 9) ∧
  (K ≥ 0 ∧ K ≤ 9) ∧
  (G ≠ K) ∧
  (10 * G + G) * G = 100 * Q + 10 * G + K ∧
  ∀ (G' Q' K' : ℕ),
    (G' ≥ 1 ∧ G' ≤ 9) →
    (Q' ≥ 0 ∧ Q' ≤ 9) →
    (K' ≥ 0 ∧ K' ≤ 9) →
    (G' ≠ K') →
    (10 * G' + G') * G' = 100 * Q' + 10 * G' + K' →
    100 * Q + 10 * G + K ≤ 100 * Q' + 10 * G' + K' ∧
  100 * Q + 10 * G + K = 044 :=
by sorry

end least_possible_QGK_l3380_338059


namespace find_fifth_month_sale_l3380_338078

def sales_problem (m1 m2 m3 m4 m6 avg : ℕ) : Prop :=
  ∃ m5 : ℕ, 
    (m1 + m2 + m3 + m4 + m5 + m6) / 6 = avg ∧
    m5 = 7560

theorem find_fifth_month_sale (m1 m2 m3 m4 m6 avg : ℕ) 
  (h1 : m1 = 7435)
  (h2 : m2 = 7920)
  (h3 : m3 = 7855)
  (h4 : m4 = 8230)
  (h5 : m6 = 6000)
  (h6 : avg = 7500) :
  sales_problem m1 m2 m3 m4 m6 avg :=
by
  sorry

end find_fifth_month_sale_l3380_338078


namespace sqrt_3_times_sqrt_6_equals_3_sqrt_2_l3380_338009

theorem sqrt_3_times_sqrt_6_equals_3_sqrt_2 : Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end sqrt_3_times_sqrt_6_equals_3_sqrt_2_l3380_338009


namespace bus_driver_compensation_l3380_338076

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_hours : ℝ)
  (overtime_rate_increase : ℝ) :
  regular_rate = 16 →
  regular_hours = 40 →
  overtime_hours = 8 →
  overtime_rate_increase = 0.75 →
  regular_rate * regular_hours +
  (regular_rate * (1 + overtime_rate_increase)) * overtime_hours = 864 :=
by sorry

end bus_driver_compensation_l3380_338076


namespace gym_class_distance_l3380_338000

/-- The total distance students have to run in gym class -/
def total_distance (track_length : ℕ) (completed_laps remaining_laps : ℕ) : ℕ :=
  track_length * (completed_laps + remaining_laps)

/-- Proof that the total distance to run is 1500 meters -/
theorem gym_class_distance : total_distance 150 6 4 = 1500 := by
  sorry

end gym_class_distance_l3380_338000


namespace pizza_feeding_capacity_l3380_338094

theorem pizza_feeding_capacity 
  (total_people : ℕ) 
  (pizza_cost : ℕ) 
  (earnings_per_night : ℕ) 
  (babysitting_nights : ℕ) : 
  total_people / (babysitting_nights * earnings_per_night / pizza_cost) = 3 :=
by
  -- Assuming:
  -- total_people = 15
  -- pizza_cost = 12
  -- earnings_per_night = 4
  -- babysitting_nights = 15
  sorry

#check pizza_feeding_capacity

end pizza_feeding_capacity_l3380_338094


namespace inequality_properties_l3380_338021

theorem inequality_properties (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a * b > b ^ 2) ∧ (1 / a < 1 / b) ∧ (a ^ 2 > a * b) := by
  sorry

end inequality_properties_l3380_338021


namespace simple_interest_time_period_l3380_338042

theorem simple_interest_time_period 
  (principal : ℝ)
  (amount1 : ℝ)
  (amount2 : ℝ)
  (rate_increase : ℝ)
  (h1 : principal = 825)
  (h2 : amount1 = 956)
  (h3 : amount2 = 1055)
  (h4 : rate_increase = 4) :
  ∃ (rate : ℝ) (time : ℝ),
    amount1 = principal + (principal * rate * time) / 100 ∧
    amount2 = principal + (principal * (rate + rate_increase) * time) / 100 ∧
    time = 3 := by
  sorry

end simple_interest_time_period_l3380_338042


namespace nomogram_relations_l3380_338043

-- Define the nomogram scales as real numbers
variables (x y z t r w v q s : ℝ)

-- Define y₁ as a function of y
def y₁ (y : ℝ) : ℝ := y

-- Theorem statement
theorem nomogram_relations :
  z = (x + 2 * y₁ y) / 3 ∧
  w = 2 * z ∧
  r = x - 2 ∧
  y + q = 6 ∧
  2 * s + t = 8 ∧
  3 * z - x - 2 * t + 6 = 0 ∧
  8 * z - 4 * t - v + 12 = 0 := by
sorry


end nomogram_relations_l3380_338043


namespace arithmetic_sequence_inequality_l3380_338089

/-- An arithmetic sequence with a non-zero common difference and positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a n > 0
  h3 : ∀ n, a (n + 1) = a n + d

/-- For an arithmetic sequence with non-zero common difference and positive terms, a₁ · a₈ < a₄ · a₅ -/
theorem arithmetic_sequence_inequality (seq : ArithmeticSequence) : seq.a 1 * seq.a 8 < seq.a 4 * seq.a 5 := by
  sorry

end arithmetic_sequence_inequality_l3380_338089


namespace equation_solution_l3380_338011

theorem equation_solution : ∃ x : ℝ, (10 - 2 * x = 16) ∧ (x = -3) := by
  sorry

end equation_solution_l3380_338011


namespace meeting_probability_l3380_338030

-- Define the time range in minutes
def timeRange : ℝ := 60

-- Define the waiting time in minutes
def waitTime : ℝ := 10

-- Define the probability of meeting function
def probabilityOfMeeting (arrivalRange1 : ℝ) (arrivalRange2 : ℝ) : ℚ :=
  sorry

theorem meeting_probability :
  (probabilityOfMeeting timeRange timeRange = 11/36) ∧
  (probabilityOfMeeting (timeRange/2) timeRange = 11/36) ∧
  (probabilityOfMeeting (5*timeRange/6) timeRange = 19/60) :=
sorry

end meeting_probability_l3380_338030


namespace vector_parallelism_transitivity_l3380_338006

/-- Given three non-zero vectors, if the first is parallel to the second and the second is parallel to the third, then the first is parallel to the third. -/
theorem vector_parallelism_transitivity 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : ∃ (k : ℝ), a = k • b) (hbc : ∃ (m : ℝ), b = m • c) : 
  ∃ (n : ℝ), a = n • c :=
sorry

end vector_parallelism_transitivity_l3380_338006


namespace smallest_valid_arrangement_l3380_338077

/-- Represents a circular table with chairs -/
structure CircularTable :=
  (num_chairs : ℕ)

/-- Checks if a seating arrangement is valid -/
def is_valid_arrangement (table : CircularTable) (seated : ℕ) : Prop :=
  seated > 0 ∧ seated ≤ table.num_chairs ∧ 
  ∀ (new_seat : ℕ), new_seat ≤ table.num_chairs → 
    ∃ (occupied_seat : ℕ), occupied_seat ≤ table.num_chairs ∧ 
      (new_seat = occupied_seat + 1 ∨ new_seat = occupied_seat - 1 ∨ 
       (occupied_seat = table.num_chairs ∧ new_seat = 1) ∨ 
       (occupied_seat = 1 ∧ new_seat = table.num_chairs))

/-- The theorem to be proved -/
theorem smallest_valid_arrangement (table : CircularTable) 
  (h : table.num_chairs = 100) : 
  (∃ (n : ℕ), is_valid_arrangement table n ∧ 
    ∀ (m : ℕ), m < n → ¬is_valid_arrangement table m) ∧
  (∃ (n : ℕ), is_valid_arrangement table n ∧ n = 20) :=
sorry

end smallest_valid_arrangement_l3380_338077


namespace no_solution_exists_l3380_338035

/-- Set A defined as {(x, y) | x = n, y = na + b, n ∈ ℤ} -/
def A (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ n : ℤ, p.1 = n ∧ p.2 = n * a + b}

/-- Set B defined as {(x, y) | x = m, y = 3m^2 + 15, m ∈ ℤ} -/
def B : Set (ℝ × ℝ) :=
  {p | ∃ m : ℤ, p.1 = m ∧ p.2 = 3 * m^2 + 15}

/-- Set C defined as {(x, y) | x^2 + y^2 ≤ 144, x, y ∈ ℝ} -/
def C : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 ≤ 144}

/-- Theorem stating that there do not exist real numbers a and b satisfying both conditions -/
theorem no_solution_exists : ¬∃ a b : ℝ, (A a b ∩ B).Nonempty ∧ (a, b) ∈ C := by
  sorry

end no_solution_exists_l3380_338035


namespace units_digit_problem_l3380_338098

theorem units_digit_problem : ∃ n : ℕ, (8 * 18 * 1978 - 8^3) % 10 = 0 := by
  sorry

end units_digit_problem_l3380_338098


namespace operation_sequence_l3380_338027

theorem operation_sequence (x : ℝ) : ((x / 4 + 3) * 7 - 5 = 30) ↔ (x = 8) := by
  sorry

end operation_sequence_l3380_338027


namespace correct_donations_l3380_338014

/-- Represents the donation amounts to each charity -/
structure CharityDonations where
  homeless : ℝ
  foodBank : ℝ
  parkRestoration : ℝ
  animalRescue : ℝ

/-- Calculates the total donations to charities given the bake sale earnings and conditions -/
def calculateDonations (totalEarnings personalDonation costOfIngredients : ℝ) : CharityDonations :=
  let remainingForCharity := totalEarnings - costOfIngredients
  let homelessShare := 0.30 * remainingForCharity + personalDonation
  let foodBankShare := 0.25 * remainingForCharity + personalDonation
  let parkRestorationShare := 0.20 * remainingForCharity + personalDonation
  let animalRescueShare := 0.25 * remainingForCharity + personalDonation
  { homeless := homelessShare
  , foodBank := foodBankShare
  , parkRestoration := parkRestorationShare
  , animalRescue := animalRescueShare }

theorem correct_donations :
  let donations := calculateDonations 500 15 110
  donations.homeless = 132 ∧
  donations.foodBank = 112.5 ∧
  donations.parkRestoration = 93 ∧
  donations.animalRescue = 112.5 := by
  sorry

end correct_donations_l3380_338014


namespace functional_equation_implies_even_l3380_338020

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + k * f b

theorem functional_equation_implies_even (f : ℝ → ℝ) (k : ℝ) 
    (h_eq : FunctionalEquation f k) (h_nonzero : ∃ x, f x ≠ 0) : 
    ∀ x : ℝ, f (-x) = f x := by
  sorry

end functional_equation_implies_even_l3380_338020


namespace length_comparison_l3380_338051

theorem length_comparison : 
  900/1000 < (2 : ℝ) ∧ (2 : ℝ) < 300/100 ∧ 300/100 < 80/10 ∧ 80/10 < 1000 := by
  sorry

end length_comparison_l3380_338051


namespace refrigerator_payment_proof_l3380_338008

def refrigerator_problem (first_payment second_payment third_payment : ℝ)
  (first_percent second_percent third_percent sales_tax_rate : ℝ)
  (delivery_fee : ℝ) : Prop :=
  let total_cost := first_payment / first_percent
  let sales_tax := sales_tax_rate * total_cost
  let total_with_tax_and_fee := total_cost + sales_tax + delivery_fee
  let total_payments := first_payment + second_payment + third_payment
  let remaining_payment := total_with_tax_and_fee - total_payments
  remaining_payment = 1137.50

theorem refrigerator_payment_proof :
  refrigerator_problem 875 650 1200 0.25 0.15 0.35 0.075 100 := by
  sorry

end refrigerator_payment_proof_l3380_338008


namespace descending_order_abcd_l3380_338068

theorem descending_order_abcd (a b c d : ℚ) 
  (h1 : 2006 = 9 * a) 
  (h2 : 2006 = 15 * b) 
  (h3 : 2006 = 32 * c) 
  (h4 : 2006 = 68 * d) : 
  a > b ∧ b > c ∧ c > d := by
  sorry

end descending_order_abcd_l3380_338068


namespace square_field_area_l3380_338080

/-- The area of a square field with a diagonal of 26 meters is 338 square meters. -/
theorem square_field_area (diagonal : ℝ) (h : diagonal = 26) :
  (diagonal ^ 2) / 2 = 338 := by
  sorry

end square_field_area_l3380_338080


namespace hash_prehash_eighteen_l3380_338060

-- Define the # operator
def hash (x : ℝ) : ℝ := x + 5

-- Define the # prefix operator
def prehash (x : ℝ) : ℝ := x - 5

-- Theorem statement
theorem hash_prehash_eighteen : prehash (hash 18) = 18 := by
  sorry

end hash_prehash_eighteen_l3380_338060


namespace intersection_point_of_function_and_inverse_l3380_338050

-- Define the function g
def g (c : ℤ) : ℝ → ℝ := λ x => 4 * x + c

-- State the theorem
theorem intersection_point_of_function_and_inverse (c : ℤ) :
  ∃ (d : ℤ), (g c (-4) = d ∧ g c d = -4) → d = -4 := by
  sorry

end intersection_point_of_function_and_inverse_l3380_338050


namespace problem_solution_l3380_338075

def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

theorem problem_solution :
  (∀ x : ℝ, f 2 x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 2) ∧
  (∀ k : ℝ, (∃ x : ℝ, (f 2 x + f 2 (-x)) / 3 < |k|) ↔ k < -2/3 ∨ k > 2/3) := by
  sorry

end problem_solution_l3380_338075


namespace fixed_point_on_line_l3380_338025

theorem fixed_point_on_line (k : ℝ) : 
  (k + 1) * 4 + (-6) + 2 - 4 * k = 0 := by sorry

end fixed_point_on_line_l3380_338025


namespace min_time_proof_l3380_338024

-- Define the quantities of honey and milk
def honey_pots : ℕ := 10
def milk_cans : ℕ := 22

-- Define the time taken by Pooh and Piglet for honey and milk
def pooh_honey_time : ℕ := 2
def pooh_milk_time : ℕ := 1
def piglet_honey_time : ℕ := 5
def piglet_milk_time : ℕ := 3

-- Define the function to calculate the minimum time
def min_consumption_time : ℕ :=
  -- The actual calculation is not implemented here
  30

-- State the theorem
theorem min_time_proof :
  min_consumption_time = 30 :=
sorry

end min_time_proof_l3380_338024


namespace rectangle_ratio_in_square_arrangement_l3380_338074

/-- Represents the arrangement of rectangles around a square -/
structure SquareArrangement where
  s : ℝ  -- side length of the inner square
  x : ℝ  -- longer side of each rectangle
  y : ℝ  -- shorter side of each rectangle

/-- The theorem stating the ratio of rectangle sides -/
theorem rectangle_ratio_in_square_arrangement
  (arr : SquareArrangement)
  (h1 : arr.s > 0)  -- inner square side length is positive
  (h2 : arr.s + 2 * arr.y = 3 * arr.s)  -- outer square side length relation
  (h3 : arr.x + arr.s = 3 * arr.s)  -- outer square side length relation in perpendicular direction
  : arr.x / arr.y = 2 := by
  sorry

#check rectangle_ratio_in_square_arrangement

end rectangle_ratio_in_square_arrangement_l3380_338074


namespace firstYearStudents2012_is_set_l3380_338092

/-- A type representing a student -/
structure Student :=
  (name : String)
  (year : Nat)
  (school : String)
  (enrollmentYear : Nat)

/-- Definition of a well-defined criterion for set membership -/
def hasWellDefinedCriterion (s : Set Student) : Prop :=
  ∀ x : Student, (x ∈ s) ∨ (x ∉ s)

/-- The set of all first-year high school students at a certain school in 2012 -/
def firstYearStudents2012 (school : String) : Set Student :=
  {s : Student | s.year = 1 ∧ s.school = school ∧ s.enrollmentYear = 2012}

/-- Theorem stating that the collection of first-year students in 2012 forms a set -/
theorem firstYearStudents2012_is_set (school : String) :
  hasWellDefinedCriterion (firstYearStudents2012 school) :=
sorry

end firstYearStudents2012_is_set_l3380_338092


namespace simplify_expression_l3380_338061

theorem simplify_expression (y : ℝ) : 3*y + 9*y^2 + 10 - (5 - 3*y - 9*y^2) = 18*y^2 + 6*y + 5 := by
  sorry

end simplify_expression_l3380_338061


namespace protective_clothing_production_l3380_338012

/-- Represents the situation of a factory producing protective clothing --/
theorem protective_clothing_production 
  (total_production : ℕ) 
  (overtime_increase : ℚ) 
  (days_ahead : ℕ) 
  (x : ℚ) 
  (h1 : total_production = 1000) 
  (h2 : overtime_increase = 1/5) 
  (h3 : days_ahead = 2) 
  (h4 : x > 0) :
  (total_production / x) - (total_production / ((1 + overtime_increase) * x)) = days_ahead :=
sorry

end protective_clothing_production_l3380_338012


namespace binary_multiplication_subtraction_l3380_338084

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

def a : List Bool := [true, false, true, true, false, true, true]
def b : List Bool := [true, false, true, true, true]
def c : List Bool := [false, true, false, true, false, true]
def result : List Bool := [true, false, false, false, false, true, false, false, false, false, true]

theorem binary_multiplication_subtraction :
  nat_to_binary (binary_to_nat a * binary_to_nat b - binary_to_nat c) = result :=
sorry

end binary_multiplication_subtraction_l3380_338084


namespace store_b_cheaper_for_40_l3380_338019

-- Define the rental fee functions
def y₁ (x : ℕ) : ℝ := 96 * x

def y₂ (x : ℕ) : ℝ :=
  if x ≤ 6 then 160 * x else 80 * x + 480

-- Theorem statement
theorem store_b_cheaper_for_40 :
  y₂ 40 < y₁ 40 := by
  sorry

end store_b_cheaper_for_40_l3380_338019


namespace arithmetic_sequence_specific_terms_l3380_338004

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + seq.diff * (n - 1)

/-- Theorem: In an arithmetic sequence where the 4th term is 23 and the 6th term is 47, the 8th term is 71 -/
theorem arithmetic_sequence_specific_terms
  (seq : ArithmeticSequence)
  (h4 : seq.nthTerm 4 = 23)
  (h6 : seq.nthTerm 6 = 47) :
  seq.nthTerm 8 = 71 := by
  sorry

end arithmetic_sequence_specific_terms_l3380_338004


namespace prob_one_male_one_female_proof_l3380_338071

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of students to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting one male and one female student -/
def prob_one_male_one_female : ℚ := 3 / 5

theorem prob_one_male_one_female_proof :
  (num_male.choose 1 * num_female.choose 1) / total_students.choose num_selected = prob_one_male_one_female :=
sorry

end prob_one_male_one_female_proof_l3380_338071


namespace regular_polygon_sides_l3380_338041

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 140 → n = 9 := by
  sorry

end regular_polygon_sides_l3380_338041


namespace set_B_when_a_is_2_A_equals_B_when_a_is_negative_one_l3380_338099

-- Define set A
def setA (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 3*a - 1) < 0}

-- Define set B (domain of log(x))
def setB : Set ℝ := {x | x > 0}

-- Theorem 1: When a=2, B = {x | 2 < x < 7}
theorem set_B_when_a_is_2 :
  setB = {x : ℝ | 2 < x ∧ x < 7} ∧ ∀ x ∈ setB, (x - 2) * (x - 7) < 0 :=
sorry

-- Theorem 2: A = B only when a = -1
theorem A_equals_B_when_a_is_negative_one :
  ∃! a : ℝ, setA a = setB ∧ a = -1 :=
sorry

end set_B_when_a_is_2_A_equals_B_when_a_is_negative_one_l3380_338099


namespace inequality_proof_l3380_338081

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a) ∧
  ((a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d) :=
by sorry


end inequality_proof_l3380_338081


namespace empty_set_proof_l3380_338032

theorem empty_set_proof : {x : ℝ | x^2 + 1 = 0} = ∅ := by
  sorry

end empty_set_proof_l3380_338032


namespace gcd_9013_4357_l3380_338029

theorem gcd_9013_4357 : Nat.gcd 9013 4357 = 1 := by
  sorry

end gcd_9013_4357_l3380_338029


namespace x_value_when_z_is_64_l3380_338013

/-- Given that x is directly proportional to y², y is inversely proportional to √z,
    and x = 4 when z = 16, prove that x = 1 when z = 64. -/
theorem x_value_when_z_is_64 
  (k : ℝ) (n : ℝ) -- Constants of proportionality
  (h1 : ∀ (y z : ℝ), x = k * y^2) -- x is directly proportional to y²
  (h2 : ∀ (y z : ℝ), y = n / Real.sqrt z) -- y is inversely proportional to √z
  (h3 : k * (n / Real.sqrt 16)^2 = 4) -- x = 4 when z = 16
  : k * (n / Real.sqrt 64)^2 = 1 := by
  sorry


end x_value_when_z_is_64_l3380_338013


namespace impossibleTransformation_l3380_338095

-- Define the colors
inductive Color
| Green
| Blue
| Red

-- Define the circle as a list of colors
def Circle := List Color

-- Define the initial and target states
def initialState : Circle := [Color.Green, Color.Blue, Color.Red]
def targetState : Circle := [Color.Blue, Color.Green, Color.Red]

-- Define the operations
def addBetweenDifferent (c : Circle) (i : Nat) (newColor : Color) : Circle := sorry
def addBetweenSame (c : Circle) (i : Nat) (newColor : Color) : Circle := sorry
def deleteMiddle (c : Circle) (i : Nat) : Circle := sorry

-- Define a single step transformation
def step (c : Circle) : Circle := sorry

-- Define the transformation process
def transform (c : Circle) (n : Nat) : Circle :=
  match n with
  | 0 => c
  | n + 1 => step (transform c n)

-- Theorem statement
theorem impossibleTransformation : 
  ∀ n : Nat, transform initialState n ≠ targetState := sorry

end impossibleTransformation_l3380_338095


namespace hyperbola_line_intersection_l3380_338083

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the line
structure Line where
  m : ℝ
  c : ℝ

-- Define the problem
theorem hyperbola_line_intersection
  (h : Hyperbola)
  (l : Line)
  (P Q R : Point)
  (h_eccentricity : h.e = Real.sqrt 3)
  (l_slope : l.m = 1)
  (intersect_y_axis : R.x = 0)
  (dot_product : P.x * Q.x + P.y * Q.y = -3)
  (segment_ratio : P.x - R.x = 3 * (R.x - Q.x))
  (on_line_P : P.y = l.m * P.x + l.c)
  (on_line_Q : Q.y = l.m * Q.x + l.c)
  (on_line_R : R.y = l.m * R.x + l.c)
  (on_hyperbola_P : 2 * P.x^2 - P.y^2 = 2 * h.a^2)
  (on_hyperbola_Q : 2 * Q.x^2 - Q.y^2 = 2 * h.a^2) :
  (l.c = 1 ∨ l.c = -1) ∧ h.a^2 = 1 ∧ h.b^2 = 2 :=
sorry

end hyperbola_line_intersection_l3380_338083


namespace unique_two_digit_multiple_l3380_338053

theorem unique_two_digit_multiple : ∃! t : ℕ, 10 ≤ t ∧ t < 100 ∧ (13 * t) % 100 = 26 := by
  sorry

end unique_two_digit_multiple_l3380_338053


namespace functional_equation_solution_l3380_338003

theorem functional_equation_solution (f : ℚ → ℝ) 
  (h : ∀ x y : ℚ, f (x + y) = f x + f y + 2 * x * y) : 
  ∃ k : ℝ, ∀ x : ℚ, f x = x^2 + k * x := by
sorry

end functional_equation_solution_l3380_338003


namespace board_cut_theorem_l3380_338097

/-- Given a board of length 120 cm cut into two pieces, where the longer piece is 15 cm longer
    than twice the length of the shorter piece, prove that the shorter piece is 35 cm long. -/
theorem board_cut_theorem (shorter_piece longer_piece : ℝ) : 
  shorter_piece + longer_piece = 120 →
  longer_piece = 2 * shorter_piece + 15 →
  shorter_piece = 35 := by
  sorry

end board_cut_theorem_l3380_338097


namespace total_diagonals_50_75_l3380_338007

def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem total_diagonals_50_75 : diagonals 50 + diagonals 75 = 3875 := by
  sorry

end total_diagonals_50_75_l3380_338007


namespace problem_solution_l3380_338034

theorem problem_solution (n x y : ℝ) 
  (h1 : x = 4 * n)
  (h2 : y = x / 2)
  (h3 : 2 * n + 3 = 0.20 * 25)
  (h4 : y^3 - 4 = (1/3) * x) :
  y = (16/3)^(1/3) := by
  sorry

end problem_solution_l3380_338034


namespace arithmetic_sequence_sum_l3380_338015

/-- An arithmetic sequence with a positive common difference -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  (a 1 + a 2 + a 3 = 15) →
  (a 1 * a 2 * a 3 = 80) →
  (a 11 + a 12 + a 13 = 105) :=
by sorry

end arithmetic_sequence_sum_l3380_338015


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3380_338047

/-- An isosceles triangle with side lengths 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Given side lengths
    (b = c) →                  -- Isosceles condition
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (a + b + c = 22)           -- Perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : 
  ∃ (a b c : ℝ), isosceles_triangle_perimeter a b c :=
sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3380_338047


namespace cable_section_length_l3380_338082

theorem cable_section_length :
  ∀ (total_length : ℝ) (sections : ℝ) (kept_sections : ℝ),
    total_length = 1000 →
    sections > 0 →
    kept_sections = 15 →
    kept_sections = (1/2) * (3/4) * (total_length / sections) →
    sections = total_length / 25 :=
by
  sorry

end cable_section_length_l3380_338082


namespace emily_trivia_score_l3380_338065

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round second_round last_round : ℤ) 
  (h1 : first_round = 16)
  (h2 : second_round = 33)
  (h3 : last_round = -48) :
  first_round + second_round + last_round = 1 := by
  sorry


end emily_trivia_score_l3380_338065


namespace smallest_n_for_P_less_than_1000th_l3380_338044

def P (n : ℕ) : ℚ :=
  (2^(n-1) * Nat.factorial (n-1)) / (Nat.factorial (2*n-1) * (2*n+1))

theorem smallest_n_for_P_less_than_1000th (n : ℕ) : n = 18 ↔ 
  (n > 0 ∧ P n < 1/1000 ∧ ∀ m : ℕ, m > 0 ∧ m < n → P m ≥ 1/1000) := by
  sorry

end smallest_n_for_P_less_than_1000th_l3380_338044


namespace remainder_seven_eight_mod_hundred_l3380_338055

theorem remainder_seven_eight_mod_hundred : 7^8 % 100 = 1 := by
  sorry

end remainder_seven_eight_mod_hundred_l3380_338055


namespace golden_ratio_logarithm_l3380_338063

theorem golden_ratio_logarithm (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 15 ∧ 
       Real.log p / Real.log 8 = Real.log (p + q) / Real.log 18) : 
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end golden_ratio_logarithm_l3380_338063


namespace student_meeting_probability_l3380_338036

def library_open_time : ℝ := 120

theorem student_meeting_probability (n : ℝ) : 
  (0 < n) → 
  (n < library_open_time) → 
  ((library_open_time - n)^2 / library_open_time^2 = 1/2) → 
  (n = 120 - 60 * Real.sqrt 2) :=
sorry

end student_meeting_probability_l3380_338036


namespace work_completion_l3380_338052

/-- The number of days B worked before leaving the job --/
def days_B_worked (a_rate b_rate : ℚ) (a_remaining_days : ℚ) : ℚ :=
  15 * (1 - 4 * a_rate)

theorem work_completion 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (a_remaining_days : ℚ) 
  (h1 : a_rate = 1 / 12)
  (h2 : b_rate = 1 / 15)
  (h3 : a_remaining_days = 4) :
  days_B_worked a_rate b_rate a_remaining_days = 10 := by
  sorry

#eval days_B_worked (1/12) (1/15) 4

end work_completion_l3380_338052


namespace frog_arrangements_eq_25200_l3380_338046

/-- The number of ways to arrange 8 frogs (3 green, 4 red, 1 blue) in a row,
    where green frogs cannot sit next to the blue frog. -/
def frog_arrangements : ℕ :=
  let total_frogs : ℕ := 8
  let green_frogs : ℕ := 3
  let red_frogs : ℕ := 4
  let blue_frogs : ℕ := 1
  let red_arrangements : ℕ := Nat.factorial red_frogs
  let blue_positions : ℕ := red_frogs + 1
  let green_positions : ℕ := total_frogs - 1
  let green_arrangements : ℕ := Nat.choose green_positions green_frogs * Nat.factorial green_frogs
  red_arrangements * blue_positions * green_arrangements

theorem frog_arrangements_eq_25200 : frog_arrangements = 25200 := by
  sorry

end frog_arrangements_eq_25200_l3380_338046


namespace unique_single_digit_polynomial_exists_l3380_338005

/-- A polynomial with single-digit coefficients -/
def SingleDigitPolynomial (p : Polynomial ℤ) : Prop :=
  ∀ i, (p.coeff i) ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℤ)

/-- The theorem statement -/
theorem unique_single_digit_polynomial_exists (n : ℤ) :
  ∃! p : Polynomial ℤ, SingleDigitPolynomial p ∧ p.eval (-2) = n ∧ p.eval (-5) = n := by
  sorry

end unique_single_digit_polynomial_exists_l3380_338005


namespace positive_even_product_sum_zero_l3380_338010

theorem positive_even_product_sum_zero (n : ℕ) (h_pos : n > 0) (h_even : Even n) :
  ∃ (a b : ℤ), (n : ℤ) = a * b ∧ a + b = 0 := by
  sorry

end positive_even_product_sum_zero_l3380_338010


namespace football_purchase_theorem_l3380_338049

/-- Represents the cost and quantity of footballs purchased by a school --/
structure FootballPurchase where
  type_a_cost : ℕ
  type_b_cost : ℕ
  type_a_quantity : ℕ
  type_b_quantity : ℕ
  total_cost : ℕ
  cost_difference : ℕ

/-- Represents the second purchase with budget constraints --/
structure SecondPurchase where
  budget : ℕ
  total_quantity : ℕ

/-- Theorem stating the costs of footballs and minimum quantity of type A footballs in second purchase --/
theorem football_purchase_theorem (fp : FootballPurchase) (sp : SecondPurchase) :
  fp.type_a_quantity = 50 ∧ 
  fp.type_b_quantity = 25 ∧ 
  fp.total_cost = 7500 ∧ 
  fp.cost_difference = 30 ∧ 
  fp.type_b_cost = fp.type_a_cost + fp.cost_difference ∧
  sp.budget = 4800 ∧
  sp.total_quantity = 50 →
  fp.type_a_cost = 90 ∧ 
  fp.type_b_cost = 120 ∧ 
  (∃ m : ℕ, m ≥ 40 ∧ m * fp.type_a_cost + (sp.total_quantity - m) * fp.type_b_cost ≤ sp.budget) :=
by sorry

end football_purchase_theorem_l3380_338049


namespace quadratic_a_value_l3380_338079

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem stating the value of 'a' for a quadratic function with given properties -/
theorem quadratic_a_value (f : QuadraticFunction) 
  (vertex : f.f (-2) = 3)
  (point : f.f 1 = 6) :
  f.a = 1/3 :=
sorry

end quadratic_a_value_l3380_338079


namespace three_fifths_square_specific_number_l3380_338022

theorem three_fifths_square_specific_number : 
  (3 / 5 : ℝ) * (14.500000000000002 ^ 2) = 126.15000000000002 := by
  sorry

end three_fifths_square_specific_number_l3380_338022


namespace complex_fraction_value_l3380_338056

theorem complex_fraction_value (a : ℝ) (z : ℂ) : 
  z = (a^2 - 1 : ℂ) + (a + 1 : ℂ) * Complex.I ∧ z.re = 0 → 
  (a + Complex.I^2016) / (1 + Complex.I) = 1 - Complex.I :=
by sorry

end complex_fraction_value_l3380_338056


namespace remainder_problem_l3380_338033

theorem remainder_problem (x : ℤ) : x % 84 = 25 → x % 14 = 11 := by
  sorry

end remainder_problem_l3380_338033


namespace min_distance_sum_five_digit_numbers_l3380_338064

theorem min_distance_sum_five_digit_numbers (x₁ x₂ x₃ x₄ x₅ : ℕ) :
  -- Define the constraints
  x₅ ≥ 9 →
  x₄ + x₅ ≥ 99 →
  x₃ + x₄ + x₅ ≥ 999 →
  x₂ + x₃ + x₄ + x₅ ≥ 9999 →
  x₁ + x₂ + x₃ + x₄ + x₅ = 99999 →
  -- The theorem to prove
  x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ ≥ 101105 :=
by sorry

#check min_distance_sum_five_digit_numbers

end min_distance_sum_five_digit_numbers_l3380_338064


namespace english_book_pages_l3380_338057

theorem english_book_pages :
  ∀ (english_pages chinese_pages : ℕ),
  english_pages = chinese_pages + 12 →
  3 * english_pages + 4 * chinese_pages = 1275 →
  english_pages = 189 :=
by
  sorry

end english_book_pages_l3380_338057


namespace line_equation_l3380_338026

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l₂ (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Define the midpoint P
def P : ℝ × ℝ := (0, 0)

-- Define the line l (to be proven)
def l (x y : ℝ) : Prop := y = -1/6 * x

-- Theorem statement
theorem line_equation (A B : ℝ × ℝ) :
  l₁ A.1 A.2 →
  l₂ B.1 B.2 →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∀ x y, l x y ↔ ∃ t, x = t * A.1 ∧ y = t * A.2 :=
sorry

end line_equation_l3380_338026


namespace base6_154_to_decimal_l3380_338073

/-- Converts a list of digits in base 6 to its decimal (base 10) representation -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

theorem base6_154_to_decimal :
  base6ToDecimal [4, 5, 1] = 70 := by
  sorry

#eval base6ToDecimal [4, 5, 1]

end base6_154_to_decimal_l3380_338073


namespace shadow_length_of_shorter_cycle_l3380_338091

/-- Given two similar right-angled triangles formed by cycles and their shadows,
    this theorem proves the length of the shadow for the shorter cycle. -/
theorem shadow_length_of_shorter_cycle
  (H1 : ℝ) (S1 : ℝ) (H2 : ℝ)
  (height1 : H1 = 2.5)
  (shadow1 : S1 = 5)
  (height2 : H2 = 2)
  (similar_triangles : H1 / S1 = H2 / S2)
  : S2 = 4 :=
by sorry

end shadow_length_of_shorter_cycle_l3380_338091


namespace simple_interest_problem_l3380_338017

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that the given conditions result in the correct principal -/
theorem simple_interest_problem :
  let interest : ℚ := 4016.25
  let rate : ℚ := 3
  let time : ℕ := 5
  calculate_principal interest rate time = 26775 := by
  sorry

end simple_interest_problem_l3380_338017


namespace jelly_bean_problem_l3380_338031

/-- The number of jelly beans initially in the barrel -/
def initial_jelly_beans : ℕ := 8000

/-- The number of people who took jelly beans -/
def total_people : ℕ := 10

/-- The number of people who took twice as many jelly beans -/
def first_group : ℕ := 6

/-- The number of people who took fewer jelly beans -/
def second_group : ℕ := 4

/-- The number of jelly beans taken by each person in the second group -/
def jelly_beans_per_second_group : ℕ := 400

/-- The number of jelly beans remaining in the barrel after everyone took their share -/
def remaining_jelly_beans : ℕ := 1600

theorem jelly_bean_problem :
  initial_jelly_beans = 
    (first_group * 2 * jelly_beans_per_second_group) + 
    (second_group * jelly_beans_per_second_group) + 
    remaining_jelly_beans :=
by
  sorry

#check jelly_bean_problem

end jelly_bean_problem_l3380_338031


namespace mary_nickels_l3380_338038

def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 5
def mom_multiplier : ℕ := 2

def final_nickels : ℕ := initial_nickels + dad_nickels + mom_multiplier * (initial_nickels + dad_nickels)

theorem mary_nickels : final_nickels = 36 := by
  sorry

end mary_nickels_l3380_338038
