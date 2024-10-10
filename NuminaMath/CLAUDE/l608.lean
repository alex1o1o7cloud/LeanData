import Mathlib

namespace multiply_powers_of_same_base_l608_60894

theorem multiply_powers_of_same_base (a b : ℝ) : 2 * a * b * b^2 = 2 * a * b^3 := by
  sorry

end multiply_powers_of_same_base_l608_60894


namespace handshakes_count_l608_60889

/-- The number of people in the gathering -/
def total_people : ℕ := 30

/-- The number of people who know each other (Group A) -/
def group_a : ℕ := 20

/-- The number of people who know no one (Group B) -/
def group_b : ℕ := 10

/-- The number of handshakes between Group A and Group B -/
def handshakes_between : ℕ := group_a * group_b

/-- The number of handshakes within Group B -/
def handshakes_within : ℕ := group_b * (group_b - 1) / 2

/-- The total number of handshakes -/
def total_handshakes : ℕ := handshakes_between + handshakes_within

theorem handshakes_count : total_handshakes = 245 := by
  sorry

end handshakes_count_l608_60889


namespace concentric_polygons_inequality_l608_60887

theorem concentric_polygons_inequality (n : ℕ) (R r : ℝ) (h : Fin n → ℝ) :
  n ≥ 3 →
  R > 0 →
  r > 0 →
  r < R →
  (∀ i, h i > 0) →
  (∀ i, h i ≤ R) →
  R * Real.cos (π / n) ≥ (Finset.sum Finset.univ h) / n ∧ (Finset.sum Finset.univ h) / n ≥ r :=
by sorry

end concentric_polygons_inequality_l608_60887


namespace cookie_calories_is_250_l608_60814

/-- The number of calories in a cookie, given the total lunch calories,
    burger calories, carrot stick calories, and number of carrot sticks. -/
def cookie_calories (total_lunch_calories burger_calories carrot_stick_calories num_carrot_sticks : ℕ) : ℕ :=
  total_lunch_calories - (burger_calories + carrot_stick_calories * num_carrot_sticks)

/-- Theorem stating that each cookie has 250 calories under the given conditions. -/
theorem cookie_calories_is_250 :
  cookie_calories 750 400 20 5 = 250 := by
  sorry

end cookie_calories_is_250_l608_60814


namespace man_walking_distance_l608_60863

theorem man_walking_distance (x t d : ℝ) : 
  (d = x * t) →                           -- distance = rate * time
  (d = (x + 1) * (3/4 * t)) →             -- faster speed condition
  (d = (x - 1) * (t + 3)) →               -- slower speed condition
  (d = 18) :=                             -- distance is 18 miles
by sorry

end man_walking_distance_l608_60863


namespace class_average_proof_l608_60872

theorem class_average_proof (group1_percent : Real) (group1_avg : Real)
                            (group2_percent : Real) (group2_avg : Real)
                            (group3_percent : Real) (group3_avg : Real)
                            (group4_percent : Real) (group4_avg : Real)
                            (group5_percent : Real) (group5_avg : Real)
                            (h1 : group1_percent = 0.25)
                            (h2 : group1_avg = 80)
                            (h3 : group2_percent = 0.35)
                            (h4 : group2_avg = 65)
                            (h5 : group3_percent = 0.20)
                            (h6 : group3_avg = 90)
                            (h7 : group4_percent = 0.10)
                            (h8 : group4_avg = 75)
                            (h9 : group5_percent = 0.10)
                            (h10 : group5_avg = 85)
                            (h11 : group1_percent + group2_percent + group3_percent + group4_percent + group5_percent = 1) :
  group1_percent * group1_avg + group2_percent * group2_avg + group3_percent * group3_avg +
  group4_percent * group4_avg + group5_percent * group5_avg = 76.75 := by
  sorry

#check class_average_proof

end class_average_proof_l608_60872


namespace complement_union_theorem_l608_60862

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define set A
def A : Set Nat := {0, 1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end complement_union_theorem_l608_60862


namespace red_bus_to_orange_car_ratio_l608_60865

/-- The lengths of buses and a car, measured in feet. -/
structure VehicleLengths where
  red_bus : ℝ
  orange_car : ℝ
  yellow_bus : ℝ

/-- The conditions of the problem. -/
def problem_conditions (v : VehicleLengths) : Prop :=
  ∃ (x : ℝ),
    v.red_bus = x * v.orange_car ∧
    v.yellow_bus = 3.5 * v.orange_car ∧
    v.yellow_bus = v.red_bus - 6 ∧
    v.red_bus = 48

/-- The theorem statement. -/
theorem red_bus_to_orange_car_ratio 
  (v : VehicleLengths) (h : problem_conditions v) :
  v.red_bus / v.orange_car = 4 := by
  sorry


end red_bus_to_orange_car_ratio_l608_60865


namespace employee_salary_problem_l608_60866

/-- Given two employees M and N with a total weekly salary of $605,
    where M's salary is 120% of N's, prove that N's salary is $275 per week. -/
theorem employee_salary_problem (total_salary m_salary n_salary : ℝ) :
  total_salary = 605 →
  m_salary = 1.2 * n_salary →
  total_salary = m_salary + n_salary →
  n_salary = 275 := by
sorry

end employee_salary_problem_l608_60866


namespace cheryl_material_usage_l608_60885

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 3 / 8)
  (h2 : material2 = 1 / 3)
  (h3 : leftover = 15 / 40) :
  material1 + material2 - leftover = 1 / 3 := by
sorry

end cheryl_material_usage_l608_60885


namespace apartment_tax_calculation_l608_60854

/-- Calculates the tax amount for an apartment --/
def calculate_tax (cadastral_value : ℝ) (tax_rate : ℝ) : ℝ :=
  cadastral_value * tax_rate

/-- Theorem: The tax amount for an apartment with a cadastral value of 3 million rubles
    and a tax rate of 0.1% is equal to 3000 rubles --/
theorem apartment_tax_calculation :
  let cadastral_value : ℝ := 3000000
  let tax_rate : ℝ := 0.001
  calculate_tax cadastral_value tax_rate = 3000 := by
sorry

/-- Additional information about the apartment (not used in the main calculation) --/
def apartment_area : ℝ := 70
def is_only_property : Prop := true

end apartment_tax_calculation_l608_60854


namespace correct_elderly_sample_l608_60808

/-- Represents the composition of employees in a company and its sample --/
structure EmployeeSample where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ
  sampledYoung : ℕ
  sampledElderly : ℕ

/-- Checks if the employee sample is valid according to the given conditions --/
def isValidSample (s : EmployeeSample) : Prop :=
  s.total = 430 ∧
  s.young = 160 ∧
  s.middleAged = 2 * s.elderly ∧
  s.total = s.young + s.middleAged + s.elderly ∧
  s.sampledYoung = 32

/-- Theorem stating that for a valid sample, the number of sampled elderly should be 18 --/
theorem correct_elderly_sample (s : EmployeeSample) 
  (h : isValidSample s) : s.sampledElderly = 18 := by
  sorry


end correct_elderly_sample_l608_60808


namespace cubic_inequality_l608_60802

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 > -36*x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end cubic_inequality_l608_60802


namespace arithmetic_sequence_1001st_term_l608_60837

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  p : ℚ
  q : ℚ
  first_term : ℚ
  second_term : ℚ
  third_term : ℚ
  fourth_term : ℚ
  is_arithmetic : ∃ (d : ℚ), second_term = first_term + d ∧
                              third_term = second_term + d ∧
                              fourth_term = third_term + d
  first_is_p : first_term = p
  second_is_12 : second_term = 12
  third_is_3p_minus_q : third_term = 3 * p - q
  fourth_is_3p_plus_2q : fourth_term = 3 * p + 2 * q

/-- The 1001st term of the sequence is 5545 -/
theorem arithmetic_sequence_1001st_term (seq : ArithmeticSequence) : 
  seq.first_term + 1000 * (seq.second_term - seq.first_term) = 5545 := by
  sorry


end arithmetic_sequence_1001st_term_l608_60837


namespace percentage_selected_state_A_l608_60849

theorem percentage_selected_state_A (
  total_A : ℕ) (total_B : ℕ) (percent_B : ℚ) (diff : ℕ) :
  total_A = 8000 →
  total_B = 8000 →
  percent_B = 7 / 100 →
  (total_B : ℚ) * percent_B = (total_A : ℚ) * (7 / 100 : ℚ) + diff →
  diff = 80 →
  (7 / 100 : ℚ) * total_A = (total_A : ℚ) * (7 / 100 : ℚ) :=
by sorry

end percentage_selected_state_A_l608_60849


namespace prob_same_color_is_17_25_l608_60869

def num_green_balls : ℕ := 8
def num_red_balls : ℕ := 2
def total_balls : ℕ := num_green_balls + num_red_balls

def prob_same_color : ℚ := (num_green_balls / total_balls)^2 + (num_red_balls / total_balls)^2

theorem prob_same_color_is_17_25 : prob_same_color = 17 / 25 := by
  sorry

end prob_same_color_is_17_25_l608_60869


namespace inverse_variation_problem_l608_60825

/-- Given that x varies inversely with y³ and x = 8 when y = 1, prove that x = 1 when y = 2 -/
theorem inverse_variation_problem (x y : ℝ) (h : ∀ y : ℝ, y ≠ 0 → ∃ k : ℝ, x * y^3 = k) :
  (∃ k : ℝ, 8 * 1^3 = k) → (∃ x : ℝ, x * 2^3 = 8) → (∃ x : ℝ, x = 1) :=
by sorry

end inverse_variation_problem_l608_60825


namespace z_in_third_quadrant_l608_60883

open Complex

theorem z_in_third_quadrant : 
  let z : ℂ := (2 + I) / (I^5 - 1)
  (z.re < 0 ∧ z.im < 0) := by sorry

end z_in_third_quadrant_l608_60883


namespace fraction_simplification_l608_60815

theorem fraction_simplification (a : ℚ) (h : a ≠ 2) :
  (a^2 / (a - 2)) - (4 / (a - 2)) = a + 2 := by
  sorry

end fraction_simplification_l608_60815


namespace bedroom_paint_area_l608_60877

/-- Calculates the total paintable area in multiple identical bedrooms -/
def total_paintable_area (
  num_bedrooms : ℕ
  ) (length width height : ℝ
  ) (unpaintable_area : ℝ
  ) : ℝ :=
  num_bedrooms * (2 * (length * height + width * height) - unpaintable_area)

/-- Proves that the total paintable area in the given conditions is 1288 square feet -/
theorem bedroom_paint_area :
  total_paintable_area 4 10 12 9 74 = 1288 := by
  sorry

end bedroom_paint_area_l608_60877


namespace custom_cartesian_product_of_A_and_B_l608_60861

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2*x - x^2)}
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define the custom cartesian product
def customCartesianProduct (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- Theorem statement
theorem custom_cartesian_product_of_A_and_B :
  customCartesianProduct A B = Set.Icc 0 1 ∪ Set.Ioi 2 := by
  sorry

end custom_cartesian_product_of_A_and_B_l608_60861


namespace terminal_side_quadrant_l608_60897

/-- Given an angle α that satisfies α = 45° + k · 180° where k is an integer,
    the terminal side of α falls in either the first or third quadrant. -/
theorem terminal_side_quadrant (k : ℤ) (α : Real) 
  (h : α = 45 + k * 180) : 
  (0 < α % 360 ∧ α % 360 < 90) ∨ (180 < α % 360 ∧ α % 360 < 270) :=
sorry

end terminal_side_quadrant_l608_60897


namespace equal_angles_on_curve_l608_60824

/-- Curve C defined by y² = 4x -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point on the x-axis -/
def xAxisPoint (x : ℝ) : ℝ × ℝ := (x, 0)

/-- Line passing through two points -/
def line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

/-- Angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem equal_angles_on_curve (m n : ℝ) (hm : m > 0) (hmn : m + n = 0)
    (A B : ℝ × ℝ) (hA : A ∈ C) (hB : B ∈ C)
    (hline : A ∈ line (xAxisPoint m) B ∧ B ∈ line (xAxisPoint m) A) :
  angle (A - xAxisPoint n) (xAxisPoint m - xAxisPoint n) =
  angle (B - xAxisPoint n) (xAxisPoint m - xAxisPoint n) := by
  sorry

end equal_angles_on_curve_l608_60824


namespace sqrt_meaningful_range_l608_60891

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end sqrt_meaningful_range_l608_60891


namespace no_integer_solutions_cube_equation_l608_60817

theorem no_integer_solutions_cube_equation :
  ¬ ∃ (x y z : ℤ), x^3 + y^3 = 9*z + 5 := by
  sorry

end no_integer_solutions_cube_equation_l608_60817


namespace hyperbola_s_squared_l608_60893

/-- Represents a hyperbola with the equation (y^2 / a^2) - (x^2 / b^2) = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) lies on the hyperbola --/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  (y^2 / h.a^2) - (x^2 / h.b^2) = 1

theorem hyperbola_s_squared (h : Hyperbola) :
  h.a = 3 →
  h.contains 0 (-3) →
  h.contains 4 (-2) →
  ∃ s, h.contains 2 s ∧ s^2 = 441/36 := by
  sorry

end hyperbola_s_squared_l608_60893


namespace final_attendance_is_1166_l608_60881

/-- Calculates the final number of spectators after a series of changes in attendance at a football game. -/
def final_attendance (initial_total initial_boys initial_girls : ℕ) : ℕ :=
  let initial_adults := initial_total - (initial_boys + initial_girls)
  
  -- After first quarter
  let boys_after_q1 := initial_boys - (initial_boys / 4)
  let girls_after_q1 := initial_girls - (initial_girls / 8)
  let adults_after_q1 := initial_adults - (initial_adults / 5)
  
  -- After halftime
  let boys_after_half := boys_after_q1 + (boys_after_q1 * 5 / 100)
  let girls_after_half := girls_after_q1 + (girls_after_q1 * 7 / 100)
  let adults_after_half := adults_after_q1 + 50
  
  -- After third quarter
  let boys_after_q3 := boys_after_half - (boys_after_half * 3 / 100)
  let girls_after_q3 := girls_after_half - (girls_after_half * 4 / 100)
  let adults_after_q3 := adults_after_half + (adults_after_half * 2 / 100)
  
  -- Final numbers
  let final_boys := boys_after_q3 + 15
  let final_girls := girls_after_q3 + 25
  let final_adults := adults_after_q3 - (adults_after_q3 / 100)
  
  final_boys + final_girls + final_adults

/-- Theorem stating that given the initial conditions, the final attendance is 1166. -/
theorem final_attendance_is_1166 : final_attendance 1300 350 450 = 1166 := by
  sorry

end final_attendance_is_1166_l608_60881


namespace term2017_is_one_sixty_fifth_l608_60899

/-- A proper fraction is a pair of natural numbers (n, d) where 0 < n < d -/
def ProperFraction := { p : ℕ × ℕ // 0 < p.1 ∧ p.1 < p.2 }

/-- The sequence of proper fractions arranged by increasing denominators and numerators -/
def properFractionSequence : ℕ → ProperFraction :=
  sorry

/-- The 2017th term of the proper fraction sequence -/
def term2017 : ProperFraction :=
  properFractionSequence 2017

theorem term2017_is_one_sixty_fifth :
  term2017 = ⟨(1, 65), sorry⟩ := by sorry

end term2017_is_one_sixty_fifth_l608_60899


namespace tank_capacity_proof_l608_60816

/-- The capacity of a gasoline tank in gallons -/
def tank_capacity : ℚ := 100 / 3

/-- The amount of gasoline added to the tank in gallons -/
def added_gasoline : ℚ := 5

/-- The initial fill level of the tank as a fraction of its capacity -/
def initial_fill : ℚ := 3 / 4

/-- The final fill level of the tank as a fraction of its capacity -/
def final_fill : ℚ := 9 / 10

theorem tank_capacity_proof :
  (final_fill * tank_capacity - initial_fill * tank_capacity = added_gasoline) ∧
  (tank_capacity > 0) := by
  sorry

end tank_capacity_proof_l608_60816


namespace kolya_purchase_options_l608_60830

-- Define the store's pricing rule
def item_price (rubles : ℕ) : ℕ := 100 * rubles + 99

-- Define Kolya's total purchase amount in kopecks
def total_purchase : ℕ := 20083

-- Define the possible number of items
def possible_items : Set ℕ := {17, 117}

-- Theorem statement
theorem kolya_purchase_options :
  ∀ n : ℕ, (∃ r : ℕ, n * item_price r = total_purchase) ↔ n ∈ possible_items :=
by sorry

end kolya_purchase_options_l608_60830


namespace min_value_fraction_min_value_achievable_l608_60858

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (b^2 + a + 1) / (a * b) ≥ 2 * Real.sqrt 10 + 6 :=
by sorry

theorem min_value_achievable (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 1 ∧
    (b₀^2 + a₀ + 1) / (a₀ * b₀) = 2 * Real.sqrt 10 + 6 :=
by sorry

end min_value_fraction_min_value_achievable_l608_60858


namespace two_tailed_coin_probability_l608_60888

/-- The probability of drawing the 2-tailed coin given that the flip resulted in tails -/
def prob_two_tailed_given_tails (total_coins : ℕ) (fair_coins : ℕ) (p_tails_fair : ℚ) : ℚ :=
  let two_tailed_coins := total_coins - fair_coins
  let p_two_tailed := two_tailed_coins / total_coins
  let p_tails_two_tailed := 1
  let p_tails := p_two_tailed * p_tails_two_tailed + (fair_coins / total_coins) * p_tails_fair
  (p_tails_two_tailed * p_two_tailed) / p_tails

theorem two_tailed_coin_probability :
  prob_two_tailed_given_tails 10 9 (1/2) = 2/11 := by
  sorry

end two_tailed_coin_probability_l608_60888


namespace sqrt_x_minus_2_real_l608_60844

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_x_minus_2_real_l608_60844


namespace six_thirty_six_am_metric_l608_60807

/-- Represents a time in the metric system -/
structure MetricTime where
  hours : Nat
  minutes : Nat

/-- Converts normal time (in minutes since midnight) to metric time -/
def normalToMetric (normalMinutes : Nat) : MetricTime :=
  let totalMetricMinutes := normalMinutes * 25 / 36
  { hours := totalMetricMinutes / 100
  , minutes := totalMetricMinutes % 100 }

theorem six_thirty_six_am_metric :
  normalToMetric (6 * 60 + 36) = { hours := 2, minutes := 75 } := by
  sorry

#eval 100 * (normalToMetric (6 * 60 + 36)).hours +
      10 * ((normalToMetric (6 * 60 + 36)).minutes / 10) +
      (normalToMetric (6 * 60 + 36)).minutes % 10

end six_thirty_six_am_metric_l608_60807


namespace greatest_multiple_of_9_l608_60859

def digits : List Nat := [3, 6, 7, 8, 9]

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

def list_to_number (l : List Nat) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

def is_permutation (l1 l2 : List Nat) : Prop :=
  l1.length = l2.length ∧ l1.toFinset = l2.toFinset

theorem greatest_multiple_of_9 :
  (∀ l : List Nat, l.length = 5 → is_permutation l digits →
    is_multiple_of_9 (list_to_number l) →
    list_to_number l ≤ 98763) ∧
  (list_to_number [9, 8, 7, 6, 3] = 98763) ∧
  (is_multiple_of_9 98763) :=
sorry

end greatest_multiple_of_9_l608_60859


namespace f_properties_l608_60811

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) :=
by sorry

end f_properties_l608_60811


namespace min_value_of_f_l608_60898

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x - 2| + 3 * |x - 3| + 4 * |x - 4|

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 8) := by
  sorry

end min_value_of_f_l608_60898


namespace sophies_doughnuts_price_l608_60832

theorem sophies_doughnuts_price (cupcake_price cupcake_count doughnut_count
  pie_price pie_count cookie_price cookie_count total_spent : ℚ) :
  cupcake_price = 2 →
  cupcake_count = 5 →
  doughnut_count = 6 →
  pie_price = 2 →
  pie_count = 4 →
  cookie_price = 0.60 →
  cookie_count = 15 →
  total_spent = 33 →
  cupcake_price * cupcake_count + doughnut_count * 1 + pie_price * pie_count + cookie_price * cookie_count = total_spent :=
by
  sorry

#check sophies_doughnuts_price

end sophies_doughnuts_price_l608_60832


namespace two_digit_addition_l608_60864

theorem two_digit_addition (A : ℕ) : A < 10 → (10 * A + 7 + 30 = 77) ↔ A = 4 := by sorry

end two_digit_addition_l608_60864


namespace red_balls_count_l608_60848

theorem red_balls_count (total : ℕ) (p : ℚ) (h_total : total = 12) (h_p : p = 1 / 22) :
  ∃ (r : ℕ), r ≤ total ∧ 
    (r : ℚ) / total * ((r - 1) : ℚ) / (total - 1) = p ∧
    r = 3 :=
by sorry

end red_balls_count_l608_60848


namespace volume_ratio_l608_60804

/-- Represents a square with side length 2 -/
structure Square :=
  (side : ℝ)
  (is_two : side = 2)

/-- Represents a pyramid formed by folding a square along its diagonal -/
structure Pyramid :=
  (base : Square)

/-- Represents a sphere circumscribing a pyramid -/
structure CircumscribedSphere :=
  (pyramid : Pyramid)

/-- The volume of a pyramid -/
def pyramid_volume (p : Pyramid) : ℝ := sorry

/-- The volume of a circumscribed sphere -/
def sphere_volume (s : CircumscribedSphere) : ℝ := sorry

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio (s : CircumscribedSphere) :
  sphere_volume s / pyramid_volume s.pyramid = 4 * Real.pi := by sorry

end volume_ratio_l608_60804


namespace initial_mean_calculation_l608_60842

theorem initial_mean_calculation (n : ℕ) (correct_value wrong_value : ℝ) (correct_mean : ℝ) :
  n = 20 ∧ 
  correct_value = 160 ∧ 
  wrong_value = 135 ∧ 
  correct_mean = 151.25 →
  (n * correct_mean - correct_value + wrong_value) / n = 152.5 := by
sorry

end initial_mean_calculation_l608_60842


namespace drummer_drum_sticks_l608_60860

/-- Calculates the total number of drum stick sets used by a drummer over multiple performances. -/
def total_drum_sticks (sticks_per_show : ℕ) (tossed_after_show : ℕ) (num_nights : ℕ) : ℕ :=
  (sticks_per_show + tossed_after_show) * num_nights

/-- Theorem stating that a drummer using 5 sets per show, tossing 6 sets after each show, 
    for 30 nights, uses 330 sets of drum sticks in total. -/
theorem drummer_drum_sticks : total_drum_sticks 5 6 30 = 330 := by
  sorry

end drummer_drum_sticks_l608_60860


namespace absolute_value_equation_solution_l608_60836

theorem absolute_value_equation_solution (x z : ℝ) :
  |5 * x - Real.log z| = 5 * x + 3 * Real.log z →
  x = 0 ∧ z = 1 := by
  sorry

end absolute_value_equation_solution_l608_60836


namespace shortest_distance_parabola_to_line_l608_60851

/-- The shortest distance from a point on the parabola y = x^2 to the line 2x - y = 4 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  let line := {p : ℝ × ℝ | 2 * p.1 - p.2 = 4}
  let distance (p : ℝ × ℝ) := |2 * p.1 - p.2 - 4| / Real.sqrt 5
  ∃ (p : ℝ × ℝ), p ∈ parabola ∧
    (∀ (q : ℝ × ℝ), q ∈ parabola → distance p ≤ distance q) ∧
    distance p = 3 * Real.sqrt 5 / 5 :=
sorry

end shortest_distance_parabola_to_line_l608_60851


namespace f_properties_l608_60867

def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3)

theorem f_properties :
  (∀ x : ℝ, f x ≥ -1) ∧
  (∃ x : ℝ, f x = -1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → f x ≤ -9/16) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-2) (-1) ∧ f x = -9/16) := by
  sorry

end f_properties_l608_60867


namespace product_of_proper_fractions_sum_of_proper_and_improper_l608_60810

-- Define a fraction as a pair of integers where the denominator is non-zero
def Fraction := { p : ℚ // p > 0 }

-- Define a proper fraction
def isProper (f : Fraction) : Prop := f.val < 1

-- Define an improper fraction
def isImproper (f : Fraction) : Prop := f.val ≥ 1

-- Statement 2
theorem product_of_proper_fractions (f g : Fraction) 
  (hf : isProper f) (hg : isProper g) : 
  isProper ⟨f.val * g.val, by sorry⟩ := by sorry

-- Statement 3
theorem sum_of_proper_and_improper (f g : Fraction) 
  (hf : isProper f) (hg : isImproper g) : 
  isImproper ⟨f.val + g.val, by sorry⟩ := by sorry

end product_of_proper_fractions_sum_of_proper_and_improper_l608_60810


namespace collinear_points_x_value_l608_60819

/-- Three points in a 2D plane are collinear if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem collinear_points_x_value :
  let p : ℝ × ℝ := (1, 1)
  let a : ℝ × ℝ := (2, -4)
  let b : ℝ × ℝ := (x, 9)
  collinear p a b → x = 3 := by
sorry


end collinear_points_x_value_l608_60819


namespace probability_five_successes_in_seven_trials_l608_60841

/-- The probability of getting exactly 5 successes in 7 trials with a success probability of 3/4 -/
theorem probability_five_successes_in_seven_trials :
  let n : ℕ := 7  -- number of trials
  let k : ℕ := 5  -- number of successes
  let p : ℚ := 3/4  -- probability of success on each trial
  Nat.choose n k * p^k * (1 - p)^(n - k) = 5103/16384 := by
  sorry

end probability_five_successes_in_seven_trials_l608_60841


namespace a_minus_b_values_l608_60847

theorem a_minus_b_values (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 13) (h3 : a * b > 0) :
  a - b = 10 ∨ a - b = -10 := by
  sorry

end a_minus_b_values_l608_60847


namespace max_profit_at_84_l608_60845

/-- The defect rate as a function of daily output --/
def defect_rate (x : ℕ) : ℚ :=
  if x ≤ 94 then 1 / (96 - x) else 2/3

/-- The daily profit as a function of daily output and profit per qualified instrument --/
def daily_profit (x : ℕ) (A : ℚ) : ℚ :=
  if x ≤ 94 
  then (x - 3*x / (2*(96 - x))) * A
  else 0

/-- Theorem: The daily profit is maximized when the daily output is 84 --/
theorem max_profit_at_84 (A : ℚ) (h : A > 0) :
  ∀ x : ℕ, x ≥ 1 → daily_profit 84 A ≥ daily_profit x A :=
by sorry

end max_profit_at_84_l608_60845


namespace number_exceeding_45_percent_l608_60874

theorem number_exceeding_45_percent : ∃ x : ℝ, x = 0.45 * x + 1000 ∧ x = 1000 / 0.55 := by
  sorry

end number_exceeding_45_percent_l608_60874


namespace y_derivative_l608_60828

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x + Real.sqrt x + 2

theorem y_derivative (x : ℝ) (h : x ≠ 0) : 
  deriv y x = (x * Real.cos x - Real.sin x) / x^2 + 1 / (2 * Real.sqrt x) :=
by sorry

end y_derivative_l608_60828


namespace range_of_m_l608_60846

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 2 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*m*x + 1 > 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (¬(p m ∨ q m)) → (m ≥ 1) :=
sorry

end range_of_m_l608_60846


namespace roots_quadratic_equation_bounds_l608_60871

theorem roots_quadratic_equation_bounds (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - a*x₁ + a^2 - a = 0 ∧ x₂^2 - a*x₂ + a^2 - a = 0) →
  (0 ≤ a ∧ a ≤ 4/3) :=
by sorry

end roots_quadratic_equation_bounds_l608_60871


namespace pie_crust_flour_usage_l608_60855

/-- Given that 40 pie crusts each use 1/8 cup of flour, 
    prove that 25 larger pie crusts using the same total amount of flour 
    will each use 1/5 cup of flour. -/
theorem pie_crust_flour_usage 
  (initial_crusts : ℕ) 
  (initial_flour_per_crust : ℚ)
  (new_crusts : ℕ) 
  (total_flour : ℚ) :
  initial_crusts = 40 →
  initial_flour_per_crust = 1/8 →
  new_crusts = 25 →
  total_flour = initial_crusts * initial_flour_per_crust →
  total_flour = new_crusts * (1/5 : ℚ) := by
sorry

end pie_crust_flour_usage_l608_60855


namespace remainder_3012_div_96_l608_60827

theorem remainder_3012_div_96 : 3012 % 96 = 36 := by
  sorry

end remainder_3012_div_96_l608_60827


namespace largest_square_tile_l608_60823

theorem largest_square_tile (board_width board_length tile_size : ℕ) : 
  board_width = 19 → 
  board_length = 29 → 
  tile_size > 0 →
  (∀ n : ℕ, n > 1 → (board_width % n = 0 ∧ board_length % n = 0) → False) →
  tile_size = 1 :=
by sorry

end largest_square_tile_l608_60823


namespace max_regions_quadratic_trinomials_l608_60818

/-- The maximum number of regions into which the coordinate plane can be divided by n quadratic trinomials -/
def max_regions (n : ℕ) : ℕ := n^2 + 1

/-- Theorem stating that the maximum number of regions created by n quadratic trinomials is n^2 + 1 -/
theorem max_regions_quadratic_trinomials (n : ℕ) :
  max_regions n = n^2 + 1 := by sorry

end max_regions_quadratic_trinomials_l608_60818


namespace new_quadratic_equation_l608_60834

theorem new_quadratic_equation (α β : ℝ) : 
  (3 * α^2 + 7 * α + 4 = 0) → 
  (3 * β^2 + 7 * β + 4 = 0) → 
  (21 * (α / (β - 1))^2 - 23 * (α / (β - 1)) + 6 = 0) ∧
  (21 * (β / (α - 1))^2 - 23 * (β / (α - 1)) + 6 = 0) := by
sorry

end new_quadratic_equation_l608_60834


namespace ceiling_floor_square_zero_l608_60843

theorem ceiling_floor_square_zero : 
  (Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ))^2 = 0 := by
  sorry

end ceiling_floor_square_zero_l608_60843


namespace staircase_cube_construction_l608_60829

/-- A staircase-brick with 3 steps of width 2, made of 12 unit cubes -/
structure StaircaseBrick where
  steps : Nat
  width : Nat
  volume : Nat
  steps_eq : steps = 3
  width_eq : width = 2
  volume_eq : volume = 12

/-- Predicate to check if a cube of side n can be built using staircase-bricks -/
def canBuildCube (n : Nat) : Prop :=
  ∃ (k : Nat), n^3 = k * 12

/-- Theorem stating that a cube of side n can be built using staircase-bricks
    if and only if n is a multiple of 12 -/
theorem staircase_cube_construction (n : Nat) :
  canBuildCube n ↔ ∃ (m : Nat), n = 12 * m :=
by sorry

end staircase_cube_construction_l608_60829


namespace infinite_series_sum_l608_60850

/-- The sum of the infinite series ∑(k=1 to ∞) k^2 / 3^k is equal to 4.5 -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k : ℝ)^2 / 3^k) = (9/2 : ℝ) := by
  sorry

end infinite_series_sum_l608_60850


namespace cosine_sum_identity_l608_60840

theorem cosine_sum_identity : 
  Real.cos (80 * π / 180) * Real.cos (20 * π / 180) + 
  Real.sin (80 * π / 180) * Real.sin (20 * π / 180) = 1/2 := by
  sorry

end cosine_sum_identity_l608_60840


namespace two_solutions_l608_60800

/-- Sum of digits function -/
def T (n : ℕ) : ℕ := sorry

/-- The number of solutions to the equation -/
def num_solutions : ℕ := 2

/-- Theorem stating that there are exactly 2 solutions -/
theorem two_solutions :
  (∃ (S : Finset ℕ), S.card = num_solutions ∧
    (∀ n, n ∈ S ↔ (n : ℕ) + T n + T (T n) = 2187) ∧
    (∀ n ∈ S, n > 0)) :=
sorry

end two_solutions_l608_60800


namespace exam_score_standard_deviation_l608_60831

/-- Given an exam with mean score 74 and standard deviation σ,
    prove that if 98 is 3σ above the mean and 58 is k⋅σ below the mean,
    then k = 2 -/
theorem exam_score_standard_deviation (σ : ℝ) (k : ℝ) 
    (h1 : 98 = 74 + 3 * σ)
    (h2 : 58 = 74 - k * σ) : 
    k = 2 := by sorry

end exam_score_standard_deviation_l608_60831


namespace dog_shampoo_time_l608_60882

theorem dog_shampoo_time (total_time hosing_time : ℕ) (shampoo_count : ℕ) : 
  total_time = 55 → 
  hosing_time = 10 → 
  shampoo_count = 3 → 
  (total_time - hosing_time) / shampoo_count = 15 := by
  sorry

end dog_shampoo_time_l608_60882


namespace equal_intercept_line_equation_l608_60820

/-- A straight line passing through (-3, -2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (-3, -2) -/
  passes_through_point : slope * (-3) + y_intercept = -2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : ∃ (a : ℝ), a ≠ 0 ∧ slope * a + y_intercept = 0 ∧ a + y_intercept = 0

/-- The equation of the line is either 2x - 3y = 0 or x + y + 5 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 2/3 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = -5) := by
  sorry

end equal_intercept_line_equation_l608_60820


namespace second_divisor_l608_60809

theorem second_divisor (n : ℕ) : 
  (n ≠ 12 ∧ n ≠ 18 ∧ n ≠ 21 ∧ n ≠ 28) →
  (1008 % n = 0) →
  (∀ m : ℕ, m < n → m ≠ 12 → m ≠ 18 → m ≠ 21 → m ≠ 28 → 1008 % m ≠ 0) →
  n = 14 :=
by sorry

end second_divisor_l608_60809


namespace problem_solution_l608_60857

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 2 = y^2) 
  (h3 : x / 5 = 3*y) : 
  x = 112.5 := by
sorry

end problem_solution_l608_60857


namespace equation_solution_l608_60890

theorem equation_solution :
  ∃! x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 4) :=
by
  use -1
  sorry

end equation_solution_l608_60890


namespace intersection_line_slope_l608_60868

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 8*y + 24 = 0

-- Define the slope of the line passing through the intersection points
def slope_of_intersection_line (circle1 circle2 : ℝ → ℝ → Prop) : ℝ := 1

-- Theorem statement
theorem intersection_line_slope :
  slope_of_intersection_line circle1 circle2 = 1 := by sorry

end intersection_line_slope_l608_60868


namespace area_enclosed_by_parabola_and_line_l608_60870

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2

-- Define the line function
def line (_ : ℝ) : ℝ := 1

-- Theorem statement
theorem area_enclosed_by_parabola_and_line :
  ∫ x in (-1)..1, (line x - parabola x) = 4/3 := by sorry

end area_enclosed_by_parabola_and_line_l608_60870


namespace prime_divisibility_problem_l608_60876

theorem prime_divisibility_problem (p : ℕ) (x : ℕ) (hp : Prime p) :
  (1 ≤ x ∧ x ≤ 2 * p) →
  (x^(p - 1) ∣ (p - 1)^x + 1) →
  ((p = 2 ∧ (x = 1 ∨ x = 2)) ∨ (p = 3 ∧ (x = 1 ∨ x = 3)) ∨ x = 1) := by
  sorry

end prime_divisibility_problem_l608_60876


namespace amit_work_days_l608_60801

/-- Proves that Amit worked for 3 days before leaving the work -/
theorem amit_work_days (total_days : ℕ) (amit_rate : ℚ) (ananthu_rate : ℚ) :
  total_days = 75 ∧ amit_rate = 1 / 15 ∧ ananthu_rate = 1 / 90 →
  ∃ x : ℚ, x = 3 ∧ x * amit_rate + (total_days - x) * ananthu_rate = 1 :=
by sorry

end amit_work_days_l608_60801


namespace construction_company_gravel_purchase_l608_60852

theorem construction_company_gravel_purchase
  (total_material : ℝ)
  (sand : ℝ)
  (gravel : ℝ)
  (h1 : total_material = 14.02)
  (h2 : sand = 8.11)
  (h3 : total_material = sand + gravel) :
  gravel = 5.91 :=
by
  sorry

end construction_company_gravel_purchase_l608_60852


namespace quadrilateral_theorem_l608_60835

structure Quadrilateral :=
  (C D X W P : ℝ × ℝ)
  (CD_parallel_WX : (D.1 - C.1) * (X.2 - W.2) = (D.2 - C.2) * (X.1 - W.1))
  (P_on_CW : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (C.1 + t * (W.1 - C.1), C.2 + t * (W.2 - C.2)))
  (CW_length : Real.sqrt ((W.1 - C.1)^2 + (W.2 - C.2)^2) = 56)
  (DP_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 16)
  (PX_length : Real.sqrt ((X.1 - P.1)^2 + (X.2 - P.2)^2) = 32)

theorem quadrilateral_theorem (q : Quadrilateral) :
  Real.sqrt ((q.W.1 - q.P.1)^2 + (q.W.2 - q.P.2)^2) = 112/3 := by
  sorry

end quadrilateral_theorem_l608_60835


namespace license_plate_count_l608_60879

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 26

/-- The number of odd digits --/
def num_odd_digits : ℕ := 5

/-- The number of even digits --/
def num_even_digits : ℕ := 5

/-- The total number of possible license plates --/
def total_plates : ℕ := num_letters ^ 3 * num_odd_digits * num_even_digits

theorem license_plate_count :
  total_plates = 439400 := by
  sorry

end license_plate_count_l608_60879


namespace solve_digit_equation_l608_60873

theorem solve_digit_equation (a b d v t r : ℕ) : 
  a + b = v →
  v + d = t →
  t + a = r →
  b + d + r = 18 →
  1 ≤ a ∧ a ≤ 9 →
  1 ≤ b ∧ b ≤ 9 →
  1 ≤ d ∧ d ≤ 9 →
  1 ≤ v ∧ v ≤ 9 →
  1 ≤ t ∧ t ≤ 9 →
  1 ≤ r ∧ r ≤ 9 →
  t = 9 := by
sorry

end solve_digit_equation_l608_60873


namespace tangent_points_collinearity_l608_60895

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a point
structure Point where
  coords : ℝ × ℝ

-- Define the property of three circles being pairwise non-intersecting
def pairwise_non_intersecting (c1 c2 c3 : Circle) : Prop :=
  sorry

-- Define the property of a point being on the internal tangent of two circles
def on_internal_tangent (p : Point) (c1 c2 : Circle) : Prop :=
  sorry

-- Define the property of a point being on the external tangent of two circles
def on_external_tangent (p : Point) (c1 c2 : Circle) : Prop :=
  sorry

-- Define the property of three points being collinear
def collinear (p1 p2 p3 : Point) : Prop :=
  sorry

-- Main theorem
theorem tangent_points_collinearity 
  (c1 c2 c3 : Circle)
  (A1 A2 A3 B1 B2 B3 : Point)
  (h_non_intersecting : pairwise_non_intersecting c1 c2 c3)
  (h_A1 : on_internal_tangent A1 c2 c3)
  (h_A2 : on_internal_tangent A2 c1 c3)
  (h_A3 : on_internal_tangent A3 c1 c2)
  (h_B1 : on_external_tangent B1 c2 c3)
  (h_B2 : on_external_tangent B2 c1 c3)
  (h_B3 : on_external_tangent B3 c1 c2) :
  (collinear A1 A2 B3) ∧ 
  (collinear A1 B2 A3) ∧ 
  (collinear B1 A2 A3) ∧ 
  (collinear B1 B2 B3) :=
sorry

end tangent_points_collinearity_l608_60895


namespace nell_initial_cards_l608_60878

/-- The number of cards Nell gave to John -/
def cards_to_john : ℕ := 195

/-- The number of cards Nell gave to Jeff -/
def cards_to_jeff : ℕ := 168

/-- The number of cards Nell has left -/
def cards_left : ℕ := 210

/-- The initial number of cards Nell had -/
def initial_cards : ℕ := cards_to_john + cards_to_jeff + cards_left

theorem nell_initial_cards : initial_cards = 573 := by
  sorry

end nell_initial_cards_l608_60878


namespace arithmetic_sequence_sum_l608_60896

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 1 + a 4 + a 7 = 39)
  (h_sum2 : a 2 + a 5 + a 8 = 33) :
  a 3 + a 6 + a 9 = 27 := by
  sorry

end arithmetic_sequence_sum_l608_60896


namespace multiple_with_binary_digits_l608_60813

theorem multiple_with_binary_digits (n : ℕ+) : 
  ∃ m : ℕ, 
    (n : ℕ) ∣ m ∧ 
    (Nat.digits 2 m).length ≤ n ∧ 
    ∀ d ∈ Nat.digits 2 m, d = 0 ∨ d = 1 := by
  sorry

end multiple_with_binary_digits_l608_60813


namespace point_relationships_l608_60822

def A : Set (ℝ × ℝ) := {(x, y) | x + 2*y - 1 ≥ 0 ∧ y ≤ x + 2 ∧ 2*x + y - 5 ≤ 0}

theorem point_relationships :
  (¬ ((0 : ℝ), (0 : ℝ)) ∈ A) ∧ ((1 : ℝ), (1 : ℝ)) ∈ A := by sorry

end point_relationships_l608_60822


namespace least_product_of_primes_above_30_l608_60821

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ, 
    is_prime p ∧ 
    is_prime q ∧ 
    p > 30 ∧ 
    q > 30 ∧ 
    p ≠ q ∧ 
    p * q = 1147 ∧ 
    (∀ p' q' : ℕ, is_prime p' → is_prime q' → p' > 30 → q' > 30 → p' ≠ q' → p' * q' ≥ 1147) :=
by sorry

end least_product_of_primes_above_30_l608_60821


namespace total_distance_is_thirteen_l608_60884

/-- Represents the time in minutes to travel one mile on a given day -/
def time_per_mile (day : Nat) : Nat :=
  12 + 6 * day

/-- Represents the distance traveled in miles on a given day -/
def distance (day : Nat) : Nat :=
  60 / (time_per_mile day)

/-- The total distance traveled over five days -/
def total_distance : Nat :=
  (List.range 5).map distance |>.sum

theorem total_distance_is_thirteen :
  total_distance = 13 := by sorry

end total_distance_is_thirteen_l608_60884


namespace yellow_marbles_count_l608_60856

theorem yellow_marbles_count (blue : ℕ) (red : ℕ) (yellow : ℕ) :
  blue = 7 →
  red = 11 →
  (yellow : ℚ) / (blue + red + yellow : ℚ) = 1/4 →
  yellow = 6 :=
by sorry

end yellow_marbles_count_l608_60856


namespace power_of_two_equality_l608_60838

theorem power_of_two_equality (x : ℕ) : (1 / 4 : ℝ) * (2 ^ 30) = 2 ^ x → x = 28 := by
  sorry

end power_of_two_equality_l608_60838


namespace square_sum_value_l608_60892

theorem square_sum_value (x y : ℝ) 
  (h1 : y + 6 = (x - 3)^2)
  (h2 : x + 6 = (y - 3)^2)
  (h3 : x ≠ y) : 
  x^2 + y^2 = 43 := by
sorry

end square_sum_value_l608_60892


namespace perpendicular_vector_l608_60833

theorem perpendicular_vector (a b : ℝ × ℝ) : 
  a = (Real.sqrt 3, Real.sqrt 5) →
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (b.1^2 + b.2^2 = 4) →
  (b = (-Real.sqrt (10) / 2, Real.sqrt 6 / 2) ∨ 
   b = (Real.sqrt (10) / 2, -Real.sqrt 6 / 2)) :=
by sorry

end perpendicular_vector_l608_60833


namespace inverse_function_sum_l608_60826

/-- Given two real numbers a and b, define f and its inverse --/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b

def f_inv (a b : ℝ) : ℝ → ℝ := λ x ↦ b * x^2 + a

/-- Theorem stating that if f and f_inv are inverse functions, then a + b = 1 --/
theorem inverse_function_sum (a b : ℝ) : 
  (∀ x, f a b (f_inv a b x) = x) → a + b = 1 := by
  sorry

end inverse_function_sum_l608_60826


namespace max_side_length_of_triangle_l608_60839

/-- A triangle with integer side lengths and perimeter 24 has a maximum side length of 11 -/
theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧ 
  a + b + c = 24 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  c ≤ 11 := by
  sorry

end max_side_length_of_triangle_l608_60839


namespace pell_solution_valid_pell_recurrence_relation_l608_60812

/-- Pell's equation solution type -/
structure PellSolution (D : ℕ) where
  x : ℤ
  y : ℤ
  eq : x^2 - D * y^2 = 1

/-- Generate the kth Pell solution -/
def genPellSolution (D : ℕ) (x₀ y₀ : ℤ) (k : ℕ) : PellSolution D :=
  sorry

theorem pell_solution_valid (D : ℕ) (x₀ y₀ : ℤ) (h : ¬ ∃ n : ℕ, n^2 = D) 
    (h₀ : x₀^2 - D * y₀^2 = 1) (k : ℕ) :
  let sol := genPellSolution D x₀ y₀ k
  sol.x^2 - D * sol.y^2 = 1 :=
sorry

theorem pell_recurrence_relation (D : ℕ) (x₀ y₀ : ℤ) (h : ¬ ∃ n : ℕ, n^2 = D) 
    (h₀ : x₀^2 - D * y₀^2 = 1) (k : ℕ) :
  let x₁ := (genPellSolution D x₀ y₀ (k+1)).x
  let x₂ := (genPellSolution D x₀ y₀ (k+2)).x
  let x := (genPellSolution D x₀ y₀ k).x
  x₂ = 2 * x₀ * x₁ - x :=
sorry

end pell_solution_valid_pell_recurrence_relation_l608_60812


namespace mike_five_dollar_bills_l608_60805

theorem mike_five_dollar_bills (total_amount : ℕ) (bill_denomination : ℕ) (h1 : total_amount = 45) (h2 : bill_denomination = 5) :
  total_amount / bill_denomination = 9 := by
  sorry

end mike_five_dollar_bills_l608_60805


namespace tangent_parallel_points_l608_60880

def f (x : ℝ) := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y →
    (3 * x^2 + 1 = 4) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end tangent_parallel_points_l608_60880


namespace imaginary_part_of_z_l608_60803

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = -(1/2 : ℂ) * (1 + Complex.I)) :
  Complex.im z = 1/2 := by
  sorry

end imaginary_part_of_z_l608_60803


namespace scientific_notation_320000_l608_60886

theorem scientific_notation_320000 : 
  320000 = 3.2 * (10 : ℝ) ^ 5 := by sorry

end scientific_notation_320000_l608_60886


namespace divisor_sum_l608_60806

theorem divisor_sum (k m : ℕ) 
  (h1 : 30^k ∣ 929260) 
  (h2 : 20^m ∣ 929260) : 
  (3^k - k^3) + (2^m - m^3) = 2 := by
sorry

end divisor_sum_l608_60806


namespace unity_community_club_ratio_l608_60853

theorem unity_community_club_ratio :
  ∀ (f m c : ℕ),
  f > 0 → m > 0 → c > 0 →
  (35 * f + 30 * m + 10 * c) / (f + m + c) = 25 →
  ∃ (k : ℕ), k > 0 ∧ f = k ∧ m = k ∧ c = k :=
by sorry

end unity_community_club_ratio_l608_60853


namespace cricketer_average_score_l608_60875

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_set_matches : ℕ) 
  (second_set_matches : ℕ) 
  (first_set_average : ℝ) 
  (second_set_average : ℝ) 
  (h1 : total_matches = first_set_matches + second_set_matches)
  (h2 : total_matches = 5)
  (h3 : first_set_matches = 2)
  (h4 : second_set_matches = 3)
  (h5 : first_set_average = 60)
  (h6 : second_set_average = 50) :
  (first_set_matches * first_set_average + second_set_matches * second_set_average) / total_matches = 54 := by
  sorry

end cricketer_average_score_l608_60875
