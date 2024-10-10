import Mathlib

namespace solution_set_properties_l1779_177914

-- Define the set M
def M : Set ℝ := {x | 3 - 2*x < 0}

-- Theorem statement
theorem solution_set_properties :
  0 ∉ M ∧ 2 ∈ M := by
  sorry

end solution_set_properties_l1779_177914


namespace taxi_charge_proof_l1779_177955

/-- The charge for each additional 1/5 mile in a taxi ride -/
def additional_fifth_mile_charge : ℚ := 0.40

/-- The initial charge for the first 1/5 mile -/
def initial_charge : ℚ := 3.00

/-- The total charge for an 8-mile ride -/
def total_charge_8_miles : ℚ := 18.60

/-- The length of the ride in miles -/
def ride_length : ℚ := 8

theorem taxi_charge_proof :
  initial_charge + (ride_length * 5 - 1) * additional_fifth_mile_charge = total_charge_8_miles :=
by sorry

end taxi_charge_proof_l1779_177955


namespace inscribed_cylinder_height_l1779_177928

theorem inscribed_cylinder_height (r_hemisphere r_cylinder : ℝ) (h_hemisphere : r_hemisphere = 7) (h_cylinder : r_cylinder = 3) :
  let h_cylinder := Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  h_cylinder = Real.sqrt 40 := by
  sorry

end inscribed_cylinder_height_l1779_177928


namespace remainder_8673_mod_7_l1779_177974

theorem remainder_8673_mod_7 : 8673 % 7 = 3 := by sorry

end remainder_8673_mod_7_l1779_177974


namespace distance_between_points_l1779_177900

/-- The distance between points (0,12) and (9,0) is 15 -/
theorem distance_between_points : Real.sqrt ((9 - 0)^2 + (0 - 12)^2) = 15 := by
  sorry

end distance_between_points_l1779_177900


namespace triangle_existence_l1779_177969

theorem triangle_existence (x : ℝ) (h : x > 1) :
  let a := x^4 + x^3 + 2*x^2 + x + 1
  let b := 2*x^3 + x^2 + 2*x + 1
  let c := x^4 - 1
  (a > c) ∧ (a > b) ∧ (a < b + c) := by sorry

end triangle_existence_l1779_177969


namespace particle_speed_l1779_177906

/-- A particle moves so that its position at time t is (3t + 5, 6t - 11).
    This function represents the particle's position vector at time t. -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 5, 6 * t - 11)

/-- The speed of the particle is the magnitude of the change in position vector
    per unit time interval. -/
theorem particle_speed : 
  let v := particle_position 1 - particle_position 0
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2) = 3 * Real.sqrt 5 := by
  sorry

end particle_speed_l1779_177906


namespace arithmetic_sequence_product_l1779_177986

/-- An increasing arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b → b 4 * b 5 = 21 → b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := by
  sorry


end arithmetic_sequence_product_l1779_177986


namespace nineteen_customers_without_fish_l1779_177993

/-- Represents the fish market scenario --/
structure FishMarket where
  total_customers : ℕ
  tuna_count : ℕ
  tuna_weight : ℕ
  regular_customer_request : ℕ
  special_customer_30lb : ℕ
  special_customer_20lb : ℕ
  max_cuts_per_tuna : ℕ

/-- Calculates the number of customers who will go home without fish --/
def customers_without_fish (market : FishMarket) : ℕ :=
  let total_weight := market.tuna_count * market.tuna_weight
  let weight_for_30lb := market.special_customer_30lb * 30
  let weight_for_20lb := market.special_customer_20lb * 20
  let remaining_weight := total_weight - weight_for_30lb - weight_for_20lb
  let remaining_customers := remaining_weight / market.regular_customer_request
  let total_served := market.special_customer_30lb + market.special_customer_20lb + remaining_customers
  market.total_customers - total_served

/-- Theorem stating that 19 customers will go home without fish --/
theorem nineteen_customers_without_fish (market : FishMarket) 
  (h1 : market.total_customers = 100)
  (h2 : market.tuna_count = 10)
  (h3 : market.tuna_weight = 200)
  (h4 : market.regular_customer_request = 25)
  (h5 : market.special_customer_30lb = 10)
  (h6 : market.special_customer_20lb = 15)
  (h7 : market.max_cuts_per_tuna = 8) :
  customers_without_fish market = 19 := by
  sorry

end nineteen_customers_without_fish_l1779_177993


namespace evaluate_expression_l1779_177903

theorem evaluate_expression : 8^6 * 27^6 * 8^15 * 27^15 = 216^21 := by
  sorry

end evaluate_expression_l1779_177903


namespace function_equality_implies_n_value_l1779_177915

/-- The function f(x) = 2x^2 - 3x + n -/
def f (n : ℚ) (x : ℚ) : ℚ := 2 * x^2 - 3 * x + n

/-- The function g(x) = 2x^2 - 3x + 5n -/
def g (n : ℚ) (x : ℚ) : ℚ := 2 * x^2 - 3 * x + 5 * n

/-- Theorem stating that if 3f(3) = 2g(3), then n = 9/7 -/
theorem function_equality_implies_n_value :
  ∀ n : ℚ, 3 * (f n 3) = 2 * (g n 3) → n = 9/7 := by
  sorry

end function_equality_implies_n_value_l1779_177915


namespace flour_cost_l1779_177940

/-- Represents the cost of ingredients and cake slices --/
structure CakeCost where
  flour : ℝ
  sugar : ℝ
  butter : ℝ
  eggs : ℝ
  total : ℝ
  sliceCount : ℕ
  sliceCost : ℝ
  dogAteCost : ℝ

/-- Theorem stating that given the total cost of ingredients and the cost of what the dog ate, 
    the cost of flour is $4 --/
theorem flour_cost (c : CakeCost) 
  (h1 : c.sugar = 2)
  (h2 : c.butter = 2.5)
  (h3 : c.eggs = 0.5)
  (h4 : c.total = c.flour + c.sugar + c.butter + c.eggs)
  (h5 : c.sliceCount = 6)
  (h6 : c.sliceCost = c.total / c.sliceCount)
  (h7 : c.dogAteCost = 6)
  (h8 : c.dogAteCost = 4 * c.sliceCost) :
  c.flour = 4 := by
  sorry

end flour_cost_l1779_177940


namespace arithmetic_mean_problem_l1779_177909

theorem arithmetic_mean_problem (x : ℚ) : 
  (x + 10 + 20 + 3*x + 18 + (3*x + 6)) / 5 = 30 → x = 96/7 := by
sorry

end arithmetic_mean_problem_l1779_177909


namespace sales_price_ratio_l1779_177978

/-- Proves the ratio of percent increase in units sold to combined percent decrease in price -/
theorem sales_price_ratio (P : ℝ) (U : ℝ) (h_positive : P > 0 ∧ U > 0) :
  let price_decrease := 0.20
  let additional_discount := 0.10
  let new_price := P * (1 - price_decrease)
  let new_units := U / (1 - price_decrease)
  let final_price := new_price * (1 - additional_discount)
  let percent_increase_units := (new_units - U) / U
  let percent_decrease_price := (P - final_price) / P
  (percent_increase_units / percent_decrease_price) = 1 / 1.12 :=
by
  sorry

end sales_price_ratio_l1779_177978


namespace arithmetic_geometric_sequence_product_l1779_177916

/-- An arithmetic sequence where each term is not 0 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≠ 0 ∧ ∃ d, ∀ k, a (k + 1) = a k + d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r ≠ 0, ∀ n, b (n + 1) = r * b n

theorem arithmetic_geometric_sequence_product (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  a 3 - (a 7)^2 / 2 + a 11 = 0 →
  b 7 = a 7 →
  b 1 * b 13 = 16 := by
  sorry

end arithmetic_geometric_sequence_product_l1779_177916


namespace weekly_fat_intake_l1779_177952

def morning_rice : ℕ := 3
def afternoon_rice : ℕ := 2
def evening_rice : ℕ := 5
def fat_per_cup : ℕ := 10
def days_in_week : ℕ := 7

theorem weekly_fat_intake : 
  (morning_rice + afternoon_rice + evening_rice) * fat_per_cup * days_in_week = 700 := by
  sorry

end weekly_fat_intake_l1779_177952


namespace imaginary_part_of_z_l1779_177907

theorem imaginary_part_of_z (z : ℂ) (h : z + 3 - 4*I = 1) : z.im = 4 := by
  sorry

end imaginary_part_of_z_l1779_177907


namespace M_intersect_N_eq_M_l1779_177983

def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | |x| < 2}

theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end M_intersect_N_eq_M_l1779_177983


namespace isosceles_triangle_perimeter_l1779_177939

theorem isosceles_triangle_perimeter : ∀ x : ℝ,
  x^2 - 8*x + 15 = 0 →
  x > 0 →
  x < 4 →
  2 + 2 + x = 7 :=
by
  sorry

end isosceles_triangle_perimeter_l1779_177939


namespace distance_after_movements_l1779_177926

/-- The distance between two points given a path with specific movements -/
theorem distance_after_movements (south west north east : ℝ) :
  south = 50 ∧ west = 80 ∧ north = 30 ∧ east = 10 →
  Real.sqrt ((south - north)^2 + (west - east)^2) = 50 * Real.sqrt 106 :=
by
  sorry

end distance_after_movements_l1779_177926


namespace set_membership_properties_l1779_177950

def A : Set Int := {x | ∃ k, x = 3 * k - 1}
def B : Set Int := {x | ∃ k, x = 3 * k + 1}
def C : Set Int := {x | ∃ k, x = 3 * k}

theorem set_membership_properties (a b c : Int) (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) :
  (2 * a ∈ B) ∧ (2 * b ∈ A) ∧ (a + b ∈ C) := by
  sorry

end set_membership_properties_l1779_177950


namespace marble_ratio_l1779_177971

theorem marble_ratio (total : ℕ) (white : ℕ) (removed : ℕ) (remaining : ℕ)
  (h1 : total = 50)
  (h2 : white = 20)
  (h3 : removed = 2 * (white - (total - white - (total - removed - white))))
  (h4 : remaining = 40)
  (h5 : total = remaining + removed) :
  (total - removed - white) = (total - white - (total - removed - white)) :=
by sorry

end marble_ratio_l1779_177971


namespace lice_check_time_l1779_177998

/-- The total number of hours required for lice checks -/
def total_hours (kindergarteners first_graders second_graders third_graders : ℕ) 
  (minutes_per_check : ℕ) : ℚ :=
  (kindergarteners + first_graders + second_graders + third_graders) * minutes_per_check / 60

/-- Theorem stating that the total time for lice checks is 3 hours -/
theorem lice_check_time : 
  total_hours 26 19 20 25 2 = 3 := by sorry

end lice_check_time_l1779_177998


namespace f_max_value_l1779_177943

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n : ℚ) / ((n + 32 : ℚ) * (S (n + 1) : ℚ))

theorem f_max_value :
  (∀ n : ℕ, f n ≤ 1/50) ∧ (∃ n : ℕ, f n = 1/50) := by sorry

end f_max_value_l1779_177943


namespace cycle_price_proof_l1779_177905

/-- Proves that a cycle sold at a 5% loss for 1330 had an original price of 1400 -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1330)
  (h2 : loss_percentage = 5) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

end cycle_price_proof_l1779_177905


namespace max_value_g_on_interval_l1779_177987

def g (x : ℝ) : ℝ := x * (x^2 - 1)

theorem max_value_g_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 1 → g y ≤ g x ∧
  g x = 0 :=
sorry

end max_value_g_on_interval_l1779_177987


namespace total_working_days_l1779_177996

/-- Represents the commute options for a worker over a period of working days. -/
structure CommuteData where
  /-- Number of days the worker drove to work in the morning -/
  morning_drives : ℕ
  /-- Number of days the worker took the subway home in the afternoon -/
  afternoon_subways : ℕ
  /-- Total number of subway commutes (morning or afternoon) -/
  total_subway_commutes : ℕ

/-- Theorem stating that given the specific commute data, the total number of working days is 15 -/
theorem total_working_days (data : CommuteData) 
  (h1 : data.morning_drives = 12)
  (h2 : data.afternoon_subways = 20)
  (h3 : data.total_subway_commutes = 15) :
  data.morning_drives + (data.total_subway_commutes - data.morning_drives) = 15 := by
  sorry

#check total_working_days

end total_working_days_l1779_177996


namespace armans_sister_age_l1779_177936

theorem armans_sister_age (arman_age sister_age : ℚ) : 
  arman_age = 6 * sister_age →
  arman_age + 4 = 40 →
  sister_age - 4 = 16 / 3 :=
by
  sorry

end armans_sister_age_l1779_177936


namespace breadth_is_five_l1779_177949

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  area_eq : area = 15 * breadth
  length_diff : length = breadth + 10

/-- The breadth of a rectangular plot with given properties is 5 meters -/
theorem breadth_is_five (plot : RectangularPlot) : plot.breadth = 5 := by
  sorry

end breadth_is_five_l1779_177949


namespace initial_number_proof_l1779_177941

theorem initial_number_proof : ∃ (n : ℕ), n = 427398 ∧ 
  (∃ (k : ℕ), n - 6 = 14 * k) ∧ 
  (∀ (m : ℕ), m < 6 → ¬∃ (j : ℕ), n - m = 14 * j) :=
by sorry

end initial_number_proof_l1779_177941


namespace reservoir_D_largest_l1779_177942

-- Define the initial amount of water (same for all reservoirs)
variable (a : ℝ)

-- Define the final amounts of water in each reservoir
def final_amount_A : ℝ := a * (1 + 0.10) * (1 - 0.05)
def final_amount_B : ℝ := a * (1 + 0.09) * (1 - 0.04)
def final_amount_C : ℝ := a * (1 + 0.08) * (1 - 0.03)
def final_amount_D : ℝ := a * (1 + 0.07) * (1 - 0.02)

-- Theorem stating that Reservoir D has the largest amount of water
theorem reservoir_D_largest (a : ℝ) (h : a > 0) : 
  final_amount_D a > final_amount_A a ∧ 
  final_amount_D a > final_amount_B a ∧ 
  final_amount_D a > final_amount_C a :=
by sorry

end reservoir_D_largest_l1779_177942


namespace fraction_subtraction_l1779_177911

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 11 = 12 / 77 := by sorry

end fraction_subtraction_l1779_177911


namespace find_x_l1779_177917

theorem find_x : ∃ x : ℤ, (9873 + x = 13800) ∧ (x = 3927) := by
  sorry

end find_x_l1779_177917


namespace range_of_a_l1779_177975

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → a < -x^2 + 2*x) → a < 0 := by
  sorry

end range_of_a_l1779_177975


namespace students_playing_both_sports_l1779_177921

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 38 →
  football = 26 →
  tennis = 20 →
  neither = 9 →
  football + tennis - (total - neither) = 17 := by
sorry

end students_playing_both_sports_l1779_177921


namespace min_quadratic_expression_l1779_177922

theorem min_quadratic_expression :
  ∃ (x : ℝ), ∀ (y : ℝ), 3 * x^2 - 18 * x + 7 ≤ 3 * y^2 - 18 * y + 7 :=
by
  -- The proof would go here
  sorry

end min_quadratic_expression_l1779_177922


namespace equivalent_division_l1779_177918

theorem equivalent_division (x : ℝ) :
  x / (4^3 / 8) * Real.sqrt (7 / 5) = x / ((8 * Real.sqrt 35) / 5) := by sorry

end equivalent_division_l1779_177918


namespace price_adjustment_theorem_l1779_177912

theorem price_adjustment_theorem (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let first_increase := 1.20
  let second_increase := 1.10
  let third_increase := 1.15
  let discount := 0.95
  let tax := 1.07
  let final_price := original_price * first_increase * second_increase * third_increase * discount * tax
  let required_decrease := 0.351852
  final_price * (1 - required_decrease) = original_price := by
sorry

end price_adjustment_theorem_l1779_177912


namespace max_sum_of_roots_l1779_177963

theorem max_sum_of_roots (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 8) :
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 8 ∧
  ∀ (a' b' c' : ℝ), a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → a' + b' + c' = 8 →
  Real.sqrt (3 * a' + 2) + Real.sqrt (3 * b' + 2) + Real.sqrt (3 * c' + 2) ≤
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ∧
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) = 3 * Real.sqrt 10 :=
sorry

end max_sum_of_roots_l1779_177963


namespace downstream_distance_proof_l1779_177973

/-- Calculates the distance traveled downstream given boat speed, stream speed, and time -/
def distance_downstream (boat_speed stream_speed time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat traveling downstream for 7 hours, with a speed of 24 km/hr in still water
    and a stream speed of 4 km/hr, travels 196 km -/
theorem downstream_distance_proof :
  distance_downstream 24 4 7 = 196 := by
  sorry

end downstream_distance_proof_l1779_177973


namespace complex_magnitude_and_argument_l1779_177990

theorem complex_magnitude_and_argument :
  ∃ (t : ℝ), t > 0 ∧ 
  (Complex.abs (9 + t * Complex.I) = 13 ↔ t = Real.sqrt 88) ∧
  Complex.arg (9 + t * Complex.I) ≠ π / 4 := by
  sorry

end complex_magnitude_and_argument_l1779_177990


namespace candy_mixture_cost_l1779_177925

theorem candy_mixture_cost (candy1_weight : ℝ) (candy1_cost : ℝ) (total_weight : ℝ) (mixture_cost : ℝ) :
  candy1_weight = 30 →
  candy1_cost = 8 →
  total_weight = 90 →
  mixture_cost = 6 →
  ∃ candy2_cost : ℝ,
    candy2_cost = 5 ∧
    candy1_weight * candy1_cost + (total_weight - candy1_weight) * candy2_cost = total_weight * mixture_cost :=
by sorry

end candy_mixture_cost_l1779_177925


namespace polynomial_square_b_value_l1779_177904

theorem polynomial_square_b_value (a b : ℚ) :
  (∃ p q : ℚ, ∀ x : ℚ, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + p*x + q)^2) →
  b = 25/64 := by
  sorry

end polynomial_square_b_value_l1779_177904


namespace inequality_conditions_l1779_177981

theorem inequality_conditions (x y z : ℝ) 
  (h1 : y - x < 1.5 * Real.sqrt (x^2))
  (h2 : z = 2 * (y + x)) :
  (x ≥ 0 → z < 7 * x) ∧ (x < 0 → z < 0) := by
  sorry

end inequality_conditions_l1779_177981


namespace symmetric_function_a_value_inequality_condition_a_range_l1779_177932

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + 2*a

-- Theorem 1
theorem symmetric_function_a_value (a : ℝ) :
  (∀ x : ℝ, f a x = f a (3 - x)) → a = -3 := by sorry

-- Theorem 2
theorem inequality_condition_a_range (a : ℝ) :
  (∃ x : ℝ, f a x ≤ -|2*x - 1| + a) → a ≤ -1/2 := by sorry

end symmetric_function_a_value_inequality_condition_a_range_l1779_177932


namespace angle_equality_l1779_177988

theorem angle_equality (angle1 angle2 angle3 : ℝ) : 
  (angle1 + angle2 = 90) →  -- angle1 and angle2 are complementary
  (angle2 + angle3 = 90) →  -- angle2 and angle3 are complementary
  (angle1 = 40) →           -- angle1 is 40 degrees
  (angle3 = 40) :=          -- conclusion: angle3 is 40 degrees
by
  sorry

#check angle_equality

end angle_equality_l1779_177988


namespace shift_f_equals_g_l1779_177947

def f (x : ℝ) : ℝ := -x^2

def g (x : ℝ) : ℝ := -x^2 + 2

def vertical_shift (h : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => h x + k

theorem shift_f_equals_g : vertical_shift f 2 = g := by sorry

end shift_f_equals_g_l1779_177947


namespace trapezoid_area_between_triangles_l1779_177902

/-- Given two concentric equilateral triangles with areas 25 and 4 square units respectively,
    prove that the area of one of the four congruent trapezoids formed between them is 5.25 square units. -/
theorem trapezoid_area_between_triangles
  (outer_area : ℝ) (inner_area : ℝ) (num_trapezoids : ℕ)
  (h_outer : outer_area = 25)
  (h_inner : inner_area = 4)
  (h_num : num_trapezoids = 4) :
  (outer_area - inner_area) / num_trapezoids = 5.25 := by
  sorry

end trapezoid_area_between_triangles_l1779_177902


namespace locus_of_point_c_l1779_177948

/-- Given a right triangle ABC with ∠C = 90°, where A is on the positive x-axis and B is on the positive y-axis,
    prove that the locus of point C is described by the equation y = (b/a)x, where ab/c ≤ x ≤ a. -/
theorem locus_of_point_c (a b c : ℝ) (A B C : ℝ × ℝ) :
  a > 0 → b > 0 →
  c^2 = a^2 + b^2 →
  A.1 > 0 → A.2 = 0 →
  B.1 = 0 → B.2 > 0 →
  C.1^2 + C.2^2 = a^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = b^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = a^2 →
  ∃ (x : ℝ), a*b/c ≤ x ∧ x ≤ a ∧ C = (x, b/a * x) :=
sorry


end locus_of_point_c_l1779_177948


namespace find_certain_number_l1779_177956

theorem find_certain_number (G N : ℕ) (h1 : G = 88) (h2 : N % G = 31) (h3 : 4521 % G = 33) : N = 4519 := by
  sorry

end find_certain_number_l1779_177956


namespace arithmetic_geometric_harmonic_means_l1779_177961

theorem arithmetic_geometric_harmonic_means (p q r : ℝ) : 
  ((p + q) / 2 = 10) →
  (Real.sqrt (p * q) = 12) →
  ((q + r) / 2 = 26) →
  (2 / (1 / p + 1 / r) = 8) →
  (r - p = 32) :=
by sorry

end arithmetic_geometric_harmonic_means_l1779_177961


namespace sector_central_angle_l1779_177910

theorem sector_central_angle (area : ℝ) (radius : ℝ) (h1 : area = 3 * π / 8) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 * π / 4 := by
sorry

end sector_central_angle_l1779_177910


namespace distance_between_cities_l1779_177966

/-- The distance between city A and city B in miles -/
def distance : ℝ := sorry

/-- The time taken for the trip from A to B in hours -/
def time_AB : ℝ := 3

/-- The time taken for the trip from B to A in hours -/
def time_BA : ℝ := 2.5

/-- The time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- The speed for the round trip if time was saved, in miles per hour -/
def speed_with_savings : ℝ := 80

theorem distance_between_cities :
  distance = 180 :=
by
  sorry

end distance_between_cities_l1779_177966


namespace selling_price_calculation_l1779_177954

def calculate_selling_price (purchase_price repair_cost transport_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_charges
  let profit := (total_cost * profit_percentage) / 100
  total_cost + profit

theorem selling_price_calculation :
  calculate_selling_price 13000 5000 1000 50 = 28500 := by
  sorry

end selling_price_calculation_l1779_177954


namespace angle_side_ratio_angle_sine_relation_two_solutions_l1779_177945

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem 1
theorem angle_side_ratio (t : Triangle) :
  t.A / t.B = 1 / 2 ∧ t.B / t.C = 2 / 3 →
  t.a / t.b = 1 / Real.sqrt 3 ∧ t.b / t.c = Real.sqrt 3 / 2 := by sorry

-- Theorem 2
theorem angle_sine_relation (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B := by sorry

-- Theorem 3
theorem two_solutions (t : Triangle) :
  t.A = π / 6 ∧ t.a = 3 ∧ t.b = 4 →
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧
    t1.A = t.A ∧ t1.a = t.a ∧ t1.b = t.b ∧
    t2.A = t.A ∧ t2.a = t.a ∧ t2.b = t.b := by sorry

end angle_side_ratio_angle_sine_relation_two_solutions_l1779_177945


namespace rhombus_diagonal_length_l1779_177965

/-- A rhombus with given area and diagonal ratio has a specific longer diagonal length -/
theorem rhombus_diagonal_length 
  (area : ℝ) 
  (diagonal_ratio : ℚ) 
  (h_area : area = 135) 
  (h_ratio : diagonal_ratio = 5 / 3) : 
  ∃ (d1 d2 : ℝ), d1 > d2 ∧ d1 / d2 = diagonal_ratio ∧ d1 * d2 / 2 = area ∧ d1 = 15 * Real.sqrt 2 :=
sorry

end rhombus_diagonal_length_l1779_177965


namespace k_gonal_number_formula_l1779_177913

/-- The n-th k-gonal number -/
def N (n k : ℕ) : ℚ :=
  ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n

/-- Theorem: The formula for the n-th k-gonal number -/
theorem k_gonal_number_formula (n k : ℕ) (h1 : n ≥ 1) (h2 : k ≥ 3) :
  N n k = ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n :=
by sorry

end k_gonal_number_formula_l1779_177913


namespace arrange_five_from_ten_eq_30240_l1779_177980

/-- The number of ways to arrange 5 distinct numbers from a set of 10 numbers -/
def arrange_five_from_ten : ℕ := 10 * 9 * 8 * 7 * 6

/-- Theorem stating that arranging 5 distinct numbers from a set of 10 numbers results in 30240 possibilities -/
theorem arrange_five_from_ten_eq_30240 : arrange_five_from_ten = 30240 := by
  sorry

end arrange_five_from_ten_eq_30240_l1779_177980


namespace longest_side_of_triangle_l1779_177946

/-- Given a triangle with side lengths 8, 2x+5, and 3x+2, and a perimeter of 40,
    the longest side of the triangle is 17. -/
theorem longest_side_of_triangle (x : ℝ) : 
  8 + (2*x + 5) + (3*x + 2) = 40 → 
  max 8 (max (2*x + 5) (3*x + 2)) = 17 := by
sorry

end longest_side_of_triangle_l1779_177946


namespace complex_equation_solution_l1779_177962

theorem complex_equation_solution (z : ℂ) :
  z * Complex.I = Complex.abs (1/2 - Complex.I * (Real.sqrt 3 / 2)) →
  z = -Complex.I :=
by sorry

end complex_equation_solution_l1779_177962


namespace inverse_direct_variation_l1779_177992

theorem inverse_direct_variation (k c : ℝ) (x y z : ℝ) : 
  (5 * y = k / (x ^ 2)) →
  (3 * z = c * x) →
  (5 * 25 = k / (2 ^ 2)) →
  (x = 4) →
  (z = 6) →
  (y = 6.25) := by
  sorry

end inverse_direct_variation_l1779_177992


namespace symmetric_points_sum_l1779_177953

/-- Given two points P and Q symmetric about the x-axis, prove that a + b = -1 -/
theorem symmetric_points_sum (a b : ℝ) : 
  (∃ P Q : ℝ × ℝ, 
    P = (a - 1, 5) ∧ 
    Q = (2, b - 1) ∧ 
    P.1 = Q.1 ∧ 
    P.2 = -Q.2) →
  a + b = -1 := by
sorry

end symmetric_points_sum_l1779_177953


namespace same_name_existence_l1779_177984

/-- Represents a child in the class -/
structure Child where
  forename : Nat
  surname : Nat

/-- The problem statement -/
theorem same_name_existence 
  (children : Finset Child) 
  (h_count : children.card = 33) 
  (h_range : ∀ c ∈ children, c.forename ≤ 10 ∧ c.surname ≤ 10) 
  (h_appear : ∀ n : Nat, n ≤ 10 → 
    (∃ c ∈ children, c.forename = n) ∧ 
    (∃ c ∈ children, c.surname = n)) :
  ∃ c1 c2 : Child, c1 ∈ children ∧ c2 ∈ children ∧ c1 ≠ c2 ∧ 
    c1.forename = c2.forename ∧ c1.surname = c2.surname :=
sorry

end same_name_existence_l1779_177984


namespace complement_A_eq_l1779_177929

/-- The universal set U -/
def U : Set Int := {-2, -1, 1, 3, 5}

/-- The set A -/
def A : Set Int := {-1, 3}

/-- The complement of A with respect to U -/
def complement_A : Set Int := {x | x ∈ U ∧ x ∉ A}

theorem complement_A_eq : complement_A = {-2, 1, 5} := by sorry

end complement_A_eq_l1779_177929


namespace ribbon_parts_l1779_177934

theorem ribbon_parts (total_length : ℝ) (used_parts : ℕ) (unused_length : ℝ) :
  total_length = 30 ∧ used_parts = 4 ∧ unused_length = 10 →
  ∃ (n : ℕ), n > 0 ∧ n * (total_length - unused_length) / used_parts = total_length / n :=
by sorry

end ribbon_parts_l1779_177934


namespace binomial_8_5_l1779_177901

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_8_5_l1779_177901


namespace log_expression_equals_four_l1779_177930

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_four :
  4 * log10 2 + 3 * log10 5 - log10 (1/5) = 4 := by sorry

end log_expression_equals_four_l1779_177930


namespace hemisphere_with_disk_surface_area_l1779_177982

/-- Given a hemisphere with base area 144π and an attached circular disk of radius 5,
    the total exposed surface area is 313π. -/
theorem hemisphere_with_disk_surface_area :
  ∀ (r : ℝ) (disk_radius : ℝ),
    r > 0 →
    disk_radius > 0 →
    π * r^2 = 144 * π →
    disk_radius = 5 →
    2 * π * r^2 + π * disk_radius^2 = 313 * π :=
by sorry

end hemisphere_with_disk_surface_area_l1779_177982


namespace inequality_proof_l1779_177995

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / (b + c) = b / (c + a) - c / (a + b)) :
  b / (c + a) ≥ (Real.sqrt 17 - 1) / 4 := by
  sorry

end inequality_proof_l1779_177995


namespace bumper_car_line_problem_l1779_177999

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 10 + 5 = 25) → initial_people = 30 := by
  sorry

end bumper_car_line_problem_l1779_177999


namespace seventy_five_days_after_wednesday_is_monday_l1779_177960

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def days_after (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | m + 1 => next_day (days_after start m)

theorem seventy_five_days_after_wednesday_is_monday :
  days_after DayOfWeek.Wednesday 75 = DayOfWeek.Monday := by
  sorry


end seventy_five_days_after_wednesday_is_monday_l1779_177960


namespace soccer_goals_product_l1779_177991

def first_ten_games : List Nat := [2, 5, 3, 6, 2, 4, 2, 5, 1, 3]

def total_first_ten : Nat := first_ten_games.sum

theorem soccer_goals_product (g11 g12 : Nat) : 
  g11 < 8 → 
  g12 < 8 → 
  (total_first_ten + g11) % 11 = 0 → 
  (total_first_ten + g11 + g12) % 12 = 0 → 
  g11 * g12 = 49 := by
  sorry

end soccer_goals_product_l1779_177991


namespace quadratic_inequality_l1779_177937

theorem quadratic_inequality (a x : ℝ) : 
  a * x^2 - (a + 1) * x + 1 < 0 ↔ 
    (a = 0 ∧ x > 1) ∨
    (a < 0 ∧ (x < 1/a ∨ x > 1)) ∨
    (0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨
    (a > 1 ∧ 1/a < x ∧ x < 1) ∨
    (a ≠ 1) := by sorry

end quadratic_inequality_l1779_177937


namespace factor_expression_l1779_177979

theorem factor_expression (b : ℝ) : 56 * b^2 + 168 * b = 56 * b * (b + 3) := by
  sorry

end factor_expression_l1779_177979


namespace inequality_proof_l1779_177908

theorem inequality_proof (a b c d e : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) :
  (a/b)^4 + (b/c)^4 + (c/d)^4 + (d/e)^4 + (e/a)^4 ≥ a/b + b/c + c/d + d/e + e/a :=
by sorry

end inequality_proof_l1779_177908


namespace container_weight_container_weight_proof_l1779_177968

/-- Given a container with weights p and q when three-quarters and one-third full respectively,
    the total weight when completely full is (8p - 3q) / 5 -/
theorem container_weight (p q : ℝ) : ℝ :=
  let three_quarters_weight := p
  let one_third_weight := q
  let full_weight := (8 * p - 3 * q) / 5
  full_weight

/-- Proof of the container weight theorem -/
theorem container_weight_proof (p q : ℝ) :
  container_weight p q = (8 * p - 3 * q) / 5 := by
  sorry

end container_weight_container_weight_proof_l1779_177968


namespace line_parameterization_l1779_177989

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f t, 20t - 10),
    prove that f t = 10t + 10 for all t. -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t, 2 * (f t) - 30 = 20 * t - 10) → 
  (∀ t, f t = 10 * t + 10) := by
sorry

end line_parameterization_l1779_177989


namespace train_speed_l1779_177958

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 240)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 20) :
  (train_length + bridge_length) / crossing_time = 19.5 := by
  sorry

#check train_speed

end train_speed_l1779_177958


namespace cricket_team_size_l1779_177985

theorem cricket_team_size :
  ∀ (n : ℕ) (captain_age wicket_keeper_age team_avg_age remaining_avg_age : ℝ),
    n > 0 →
    captain_age = 26 →
    wicket_keeper_age = captain_age + 3 →
    team_avg_age = 23 →
    remaining_avg_age = team_avg_age - 1 →
    team_avg_age * n = remaining_avg_age * (n - 2) + captain_age + wicket_keeper_age →
    n = 11 := by
  sorry

end cricket_team_size_l1779_177985


namespace henry_pill_cost_l1779_177920

/-- Calculates the total cost of pills for Henry over 21 days -/
def totalPillCost (daysTotal : ℕ) (pillsPerDay : ℕ) (pillType1Count : ℕ) (pillType2Count : ℕ)
  (pillType1Cost : ℚ) (pillType2Cost : ℚ) (pillType3ExtraCost : ℚ) 
  (discountRate : ℚ) (priceIncrease : ℚ) : ℚ :=
  let pillType3Count := pillsPerDay - (pillType1Count + pillType2Count)
  let pillType3Cost := pillType2Cost + pillType3ExtraCost
  let regularDayCost := pillType1Count * pillType1Cost + pillType2Count * pillType2Cost + 
                        pillType3Count * pillType3Cost
  let discountDays := daysTotal / 3
  let regularDays := daysTotal - discountDays
  let discountDayCost := (1 - discountRate) * (pillType1Count * pillType1Cost + pillType2Count * pillType2Cost) +
                         pillType3Count * (pillType3Cost + priceIncrease)
  regularDays * regularDayCost + discountDays * discountDayCost

/-- The total cost of Henry's pills over 21 days is $1485.10 -/
theorem henry_pill_cost : 
  totalPillCost 21 12 4 5 (3/2) 7 3 (1/5) (5/2) = 1485.1 := by
  sorry

end henry_pill_cost_l1779_177920


namespace polynomial_value_at_three_l1779_177976

theorem polynomial_value_at_three : 
  let x : ℝ := 3
  (x^5 : ℝ) - 7 * (x^3 : ℝ) = 54 := by
  sorry

end polynomial_value_at_three_l1779_177976


namespace expression_evaluation_l1779_177927

theorem expression_evaluation (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) :
  ((2*a + b)^2 - (2*a + b)*(2*a - b)) / (-1/2 * b) = 0 := by sorry

end expression_evaluation_l1779_177927


namespace sqrt_450_equals_15_sqrt_2_l1779_177997

theorem sqrt_450_equals_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_equals_15_sqrt_2_l1779_177997


namespace not_neighboring_root_eq1_neighboring_root_eq2_neighboring_root_eq3_l1779_177959

/-- Definition of a neighboring root equation -/
def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ (x - y = 1 ∨ y - x = 1)

/-- Theorem for the first equation -/
theorem not_neighboring_root_eq1 : ¬ is_neighboring_root_equation 1 (-1) (-6) :=
sorry

/-- Theorem for the second equation -/
theorem neighboring_root_eq2 : is_neighboring_root_equation 2 (-2 * Real.sqrt 3) 1 :=
sorry

/-- Theorem for the third equation -/
theorem neighboring_root_eq3 (m : ℝ) : 
  is_neighboring_root_equation 1 (-(m-2)) (-2*m) ↔ m = -1 ∨ m = -3 :=
sorry

end not_neighboring_root_eq1_neighboring_root_eq2_neighboring_root_eq3_l1779_177959


namespace octal_to_binary_l1779_177931

-- Define the octal number
def octal_177 : ℕ := 177

-- Define the binary number
def binary_1111111 : ℕ := 127

-- Theorem statement
theorem octal_to_binary :
  (octal_177 : ℕ) = binary_1111111 := by sorry

end octal_to_binary_l1779_177931


namespace clock_setback_radians_l1779_177919

theorem clock_setback_radians (minutes_per_revolution : ℝ) (radians_per_revolution : ℝ) 
  (setback_minutes : ℝ) : 
  minutes_per_revolution = 60 → 
  radians_per_revolution = 2 * Real.pi → 
  setback_minutes = 10 → 
  (setback_minutes / minutes_per_revolution) * radians_per_revolution = Real.pi / 3 := by
  sorry

end clock_setback_radians_l1779_177919


namespace basketball_contest_l1779_177935

/-- Calculates the total points scored in a basketball contest --/
def total_points (layups dunks free_throws three_pointers alley_oops half_court consecutive : ℕ) : ℕ :=
  layups + dunks + 2 * free_throws + 3 * three_pointers + 4 * alley_oops + 5 * half_court + consecutive

/-- Represents the basketball contest between Reggie and his brother --/
theorem basketball_contest :
  let reggie_points := total_points 4 2 3 2 1 1 2
  let brother_points := total_points 3 1 2 5 2 4 3
  brother_points - reggie_points = 25 := by
  sorry

end basketball_contest_l1779_177935


namespace max_page_number_l1779_177972

/-- The number of '2's available -/
def available_twos : ℕ := 34

/-- The number of '2's used in numbers from 1 to 99 -/
def twos_in_1_to_99 : ℕ := 19

/-- The number of '2's used in numbers from 100 to 199 -/
def twos_in_100_to_199 : ℕ := 10

/-- The highest page number that can be reached with the available '2's -/
def highest_page_number : ℕ := 199

theorem max_page_number :
  available_twos = twos_in_1_to_99 + twos_in_100_to_199 + 5 ∧
  highest_page_number = 199 :=
sorry

end max_page_number_l1779_177972


namespace isosceles_triangle_vertex_angle_l1779_177957

theorem isosceles_triangle_vertex_angle (a b h : ℝ) : 
  a > 0 → b > 0 → h > 0 →
  a^2 = 3 * b * h →
  b = 2 * a * Real.cos (π / 4) →
  h = a * Real.sin (π / 4) →
  let vertex_angle := π - 2 * (π / 4)
  vertex_angle = π / 2 :=
by sorry

end isosceles_triangle_vertex_angle_l1779_177957


namespace correct_ages_unique_solution_l1779_177977

/-- Represents the ages of family members -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  brother : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  (ages.father + 15 = 3 * (ages.father - 25)) ∧
  (ages.father + 15 = 2 * (ages.son + 15)) ∧
  (ages.brother = (ages.father + 15) / 2 + 7)

/-- Theorem stating that the ages 45, 15, and 37 satisfy the problem conditions -/
theorem correct_ages : satisfiesConditions { father := 45, son := 15, brother := 37 } := by
  sorry

/-- Theorem stating the uniqueness of the solution -/
theorem unique_solution (ages : FamilyAges) :
  satisfiesConditions ages → ages = { father := 45, son := 15, brother := 37 } := by
  sorry

end correct_ages_unique_solution_l1779_177977


namespace crackers_sales_total_l1779_177923

theorem crackers_sales_total (friday_sales : ℕ) 
  (h1 : friday_sales = 30) 
  (h2 : ∃ saturday_sales : ℕ, saturday_sales = 2 * friday_sales) 
  (h3 : ∃ sunday_sales : ℕ, sunday_sales = saturday_sales - 15) : 
  friday_sales + 2 * friday_sales + (2 * friday_sales - 15) = 135 := by
  sorry

end crackers_sales_total_l1779_177923


namespace weeks_to_save_is_36_l1779_177951

/-- The number of weeks Nina needs to save to buy all items -/
def weeks_to_save : ℕ :=
let video_game_cost : ℚ := 50
let headset_cost : ℚ := 70
let gift_cost : ℚ := 30
let sales_tax_rate : ℚ := 12 / 100
let weekly_allowance : ℚ := 10
let initial_savings_rate : ℚ := 33 / 100
let later_savings_rate : ℚ := 50 / 100
let initial_savings_weeks : ℕ := 6

let total_cost_before_tax : ℚ := video_game_cost + headset_cost + gift_cost
let total_cost_with_tax : ℚ := total_cost_before_tax * (1 + sales_tax_rate)
let gift_cost_with_tax : ℚ := gift_cost * (1 + sales_tax_rate)

let initial_savings : ℚ := weekly_allowance * initial_savings_rate * initial_savings_weeks
let remaining_gift_cost : ℚ := gift_cost_with_tax - initial_savings
let weeks_for_gift : ℕ := (remaining_gift_cost / (weekly_allowance * later_savings_rate)).ceil.toNat

let remaining_cost : ℚ := total_cost_with_tax - gift_cost_with_tax
let weeks_for_remaining : ℕ := (remaining_cost / (weekly_allowance * later_savings_rate)).ceil.toNat

initial_savings_weeks + weeks_for_gift + weeks_for_remaining

theorem weeks_to_save_is_36 : weeks_to_save = 36 := by sorry

end weeks_to_save_is_36_l1779_177951


namespace rectangle_square_overlap_ratio_l1779_177933

/-- Given a rectangle ABCD and a square EFGH, if the rectangle shares 60% of its area with the square,
    and the square shares 30% of its area with the rectangle, then the ratio of the rectangle's length
    to its width is 8. -/
theorem rectangle_square_overlap_ratio :
  ∀ (rect_area square_area overlap_area : ℝ) (rect_length rect_width : ℝ),
    rect_area > 0 →
    square_area > 0 →
    overlap_area > 0 →
    rect_length > 0 →
    rect_width > 0 →
    rect_area = rect_length * rect_width →
    overlap_area = 0.6 * rect_area →
    overlap_area = 0.3 * square_area →
    rect_length / rect_width = 8 := by
  sorry

end rectangle_square_overlap_ratio_l1779_177933


namespace battery_usage_difference_l1779_177967

theorem battery_usage_difference (flashlights remote_controllers wall_clock wireless_mouse toys : ℝ) 
  (h1 : flashlights = 3.5)
  (h2 : remote_controllers = 7.25)
  (h3 : wall_clock = 4.8)
  (h4 : wireless_mouse = 3.4)
  (h5 : toys = 15.75) :
  toys - (flashlights + remote_controllers + wall_clock + wireless_mouse) = -3.2 := by
  sorry

end battery_usage_difference_l1779_177967


namespace completing_square_equivalence_l1779_177970

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_square_equivalence_l1779_177970


namespace rabbit_run_time_l1779_177944

/-- The time taken for a rabbit to run from the end to the front of a moving line and back -/
theorem rabbit_run_time (line_length : ℝ) (line_speed : ℝ) (rabbit_speed : ℝ) : 
  line_length = 40 →
  line_speed = 3 →
  rabbit_speed = 5 →
  (line_length / (rabbit_speed - line_speed)) + (line_length / (rabbit_speed + line_speed)) = 25 :=
by sorry

end rabbit_run_time_l1779_177944


namespace wood_square_weight_relation_l1779_177938

/-- Represents the properties of a square piece of wood -/
structure WoodSquare where
  side_length : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two square pieces of wood with uniform density and thickness -/
theorem wood_square_weight_relation 
  (w1 w2 : WoodSquare)
  (uniform_density : True)  -- Represents the assumption of uniform density and thickness
  (h1 : w1.side_length = 4)
  (h2 : w1.weight = 16)
  (h3 : w2.side_length = 6) :
  w2.weight = 36 := by
  sorry

#check wood_square_weight_relation

end wood_square_weight_relation_l1779_177938


namespace triangle_side_length_l1779_177924

theorem triangle_side_length (a b : ℝ) (C : ℝ) (S : ℝ) : 
  a = 3 * Real.sqrt 2 →
  Real.cos C = 1 / 3 →
  S = 4 * Real.sqrt 3 →
  S = 1 / 2 * a * b * Real.sin C →
  b = 2 * Real.sqrt 3 := by
sorry

end triangle_side_length_l1779_177924


namespace system_solution_l1779_177994

theorem system_solution (a₁ a₂ c₁ c₂ : ℝ) :
  (∃ (x y : ℝ), a₁ * x + y = c₁ ∧ a₂ * x + y = c₂ ∧ x = 5 ∧ y = 10) →
  (∃ (x y : ℝ), a₁ * x + 2 * y = a₁ - c₁ ∧ a₂ * x + 2 * y = a₂ - c₂ ∧ x = -4 ∧ y = -5) :=
by sorry

end system_solution_l1779_177994


namespace arun_age_is_sixty_l1779_177964

/-- Given the ages of Arun, Gokul, and Madan, prove that Arun's age is 60 years. -/
theorem arun_age_is_sixty (arun_age gokul_age madan_age : ℕ) : 
  ((arun_age - 6) / 18 = gokul_age) →
  (gokul_age = madan_age - 2) →
  (madan_age = 5) →
  arun_age = 60 := by
  sorry

end arun_age_is_sixty_l1779_177964
