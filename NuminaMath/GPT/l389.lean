import Mathlib

namespace graph_of_f_does_not_pass_through_second_quadrant_l389_38915

def f (x : ℝ) : ℝ := x - 2

theorem graph_of_f_does_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = f x ∧ x < 0 ∧ y > 0 :=
sorry

end graph_of_f_does_not_pass_through_second_quadrant_l389_38915


namespace smallest_sum_of_integers_on_square_vertices_l389_38964

theorem smallest_sum_of_integers_on_square_vertices :
  ∃ (a b c d : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  (a % b = 0 ∨ b % a = 0) ∧ (c % a = 0 ∨ a % c = 0) ∧ 
  (d % b = 0 ∨ b % d = 0) ∧ (d % c = 0 ∨ c % d = 0) ∧ 
  a % c ≠ 0 ∧ a % d ≠ 0 ∧ b % c ≠ 0 ∧ b % d ≠ 0 ∧ 
  (a + b + c + d = 35) := sorry

end smallest_sum_of_integers_on_square_vertices_l389_38964


namespace proof_problem_l389_38991

-- definitions of the given conditions
variable (a b c : ℝ)
variables (h₁ : 6 < a) (h₂ : a < 10) 
variable (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) 
variable (h₄ : c = a + b)

-- statement to be proved
theorem proof_problem (h₁ : 6 < a) (h₂ : a < 10) (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) (h₄ : c = a + b) : 9 < c ∧ c < 30 := 
sorry

end proof_problem_l389_38991


namespace total_plant_count_l389_38969

-- Definitions for conditions.
def total_rows : ℕ := 96
def columns_per_row : ℕ := 24
def divided_rows : ℕ := total_rows / 3
def undivided_rows : ℕ := total_rows - divided_rows
def beans_in_undivided_row : ℕ := columns_per_row
def corn_in_divided_row : ℕ := columns_per_row / 2
def tomatoes_in_divided_row : ℕ := columns_per_row / 2

-- Total number of plants calculation.
def total_bean_plants : ℕ := undivided_rows * beans_in_undivided_row
def total_corn_plants : ℕ := divided_rows * corn_in_divided_row
def total_tomato_plants : ℕ := divided_rows * tomatoes_in_divided_row

def total_plants : ℕ := total_bean_plants + total_corn_plants + total_tomato_plants

-- Proof statement.
theorem total_plant_count : total_plants = 2304 :=
by
  sorry

end total_plant_count_l389_38969


namespace lost_marble_count_l389_38928

def initial_marble_count : ℕ := 16
def remaining_marble_count : ℕ := 9

theorem lost_marble_count : initial_marble_count - remaining_marble_count = 7 := by
  -- Proof goes here
  sorry

end lost_marble_count_l389_38928


namespace find_z_plus_inverse_y_l389_38976

theorem find_z_plus_inverse_y
  (x y z : ℝ)
  (h1 : x * y * z = 1)
  (h2 : x + 1/z = 10)
  (h3 : y + 1/x = 5) :
  z + 1/y = 17 / 49 :=
by
  sorry

end find_z_plus_inverse_y_l389_38976


namespace tires_usage_l389_38922

theorem tires_usage :
  let total_miles := 50000
  let first_part_miles := 40000
  let second_part_miles := 10000
  let num_tires_first_part := 5
  let num_tires_total := 7
  let total_tire_miles_first := first_part_miles * num_tires_first_part
  let total_tire_miles_second := second_part_miles * num_tires_total
  let combined_tire_miles := total_tire_miles_first + total_tire_miles_second
  let miles_per_tire := combined_tire_miles / num_tires_total
  miles_per_tire = 38571 := 
by
  sorry

end tires_usage_l389_38922


namespace students_not_take_test_l389_38986

theorem students_not_take_test
  (total_students : ℕ)
  (q1_correct : ℕ)
  (q2_correct : ℕ)
  (both_correct : ℕ)
  (h_total : total_students = 29)
  (h_q1 : q1_correct = 19)
  (h_q2 : q2_correct = 24)
  (h_both : both_correct = 19)
  : (total_students - (q1_correct + q2_correct - both_correct) = 5) :=
by
  sorry

end students_not_take_test_l389_38986


namespace ginger_total_water_l389_38975

def hours_worked : Nat := 8
def cups_per_bottle : Nat := 2
def bottles_drank_per_hour : Nat := 1
def bottles_for_plants : Nat := 5

theorem ginger_total_water : 
  (hours_worked * cups_per_bottle * bottles_drank_per_hour) + (bottles_for_plants * cups_per_bottle) = 26 :=
by
  sorry

end ginger_total_water_l389_38975


namespace smallest_possible_value_l389_38955

theorem smallest_possible_value 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ 9 / 2 :=
sorry

end smallest_possible_value_l389_38955


namespace max_value_x_plus_2y_l389_38994

theorem max_value_x_plus_2y (x y : ℝ) (h : |x| + |y| ≤ 1) : x + 2 * y ≤ 2 :=
sorry

end max_value_x_plus_2y_l389_38994


namespace cubes_divisible_by_9_l389_38902

theorem cubes_divisible_by_9 (n: ℕ) (h: n > 0) : 9 ∣ n^3 + (n + 1)^3 + (n + 2)^3 :=
by 
  sorry

end cubes_divisible_by_9_l389_38902


namespace number_of_pairs_of_shoes_l389_38926

/-- A box contains some pairs of shoes with a total of 10 shoes.
    If two shoes are selected at random, the probability that they are matching shoes is 1/9.
    Prove that the number of pairs of shoes in the box is 5. -/
theorem number_of_pairs_of_shoes (n : ℕ) (h1 : 2 * n = 10) 
  (h2 : ((n * (n - 1)) / (10 * (10 - 1))) = 1 / 9) : n = 5 := 
sorry

end number_of_pairs_of_shoes_l389_38926


namespace sum_of_numbers_less_than_2_l389_38930

theorem sum_of_numbers_less_than_2:
  ∀ (a b c : ℝ), a = 0.8 → b = 1/2 → c = 0.9 → a < 2 ∧ b < 2 ∧ c < 2 → a + b + c = 2.2 :=
by
  -- We are stating that if a = 0.8, b = 1/2, and c = 0.9, and all are less than 2, then their sum is 2.2
  sorry

end sum_of_numbers_less_than_2_l389_38930


namespace polygon_sides_l389_38987

theorem polygon_sides (x : ℕ) (h1 : 180 * (x - 2) = 3 * 360) : x = 8 := by
  sorry

end polygon_sides_l389_38987


namespace marissa_sunflower_height_l389_38970

def height_sister_in_inches : ℚ := 4 * 12 + 3
def height_difference_in_inches : ℚ := 21
def inches_to_cm (inches : ℚ) : ℚ := inches * 2.54
def cm_to_m (cm : ℚ) : ℚ := cm / 100

theorem marissa_sunflower_height :
  cm_to_m (inches_to_cm (height_sister_in_inches + height_difference_in_inches)) = 1.8288 :=
by sorry

end marissa_sunflower_height_l389_38970


namespace quadratic_distinct_real_roots_l389_38985

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 9 ∧ a * c * 4 < b^2) ↔ (m < -6 ∨ m > 6) :=
by
  sorry

end quadratic_distinct_real_roots_l389_38985


namespace monotonic_range_of_a_l389_38972

theorem monotonic_range_of_a (a : ℝ) :
  (a ≥ 9 ∨ a ≤ 3) → 
  ∀ x y : ℝ, (1 ≤ x ∧ x ≤ 4) → (1 ≤ y ∧ y ≤ 4) → x ≤ y → 
  (x^2 + (1-a)*x + 3) ≤ (y^2 + (1-a)*y + 3) :=
by
  intro ha x y hx hy hxy
  sorry

end monotonic_range_of_a_l389_38972


namespace simplify_and_evaluate_expression_l389_38993

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end simplify_and_evaluate_expression_l389_38993


namespace measure_angle_BCQ_l389_38968

/-- Given:
  - Segment AB has a length of 12 units.
  - Segment AC is 9 units long.
  - Segment AC : CB = 3 : 1.
  - A semi-circle is constructed with diameter AB.
  - Another smaller semi-circle is constructed with diameter CB.
  - A line segment CQ divides the combined area of the two semi-circles into two equal areas.

  Prove: The degree measure of angle BCQ is 11.25°.
-/ 
theorem measure_angle_BCQ (AB AC CB : ℝ) (hAB : AB = 12) (hAC : AC = 9) (hRatio : AC / CB = 3) :
  ∃ θ : ℝ, θ = 11.25 :=
by
  sorry

end measure_angle_BCQ_l389_38968


namespace shaded_triangle_area_l389_38981

/--
The large equilateral triangle shown consists of 36 smaller equilateral triangles.
Each of the smaller equilateral triangles has an area of 10 cm². 
The area of the shaded triangle is K cm².
Prove that K = 110 cm².
-/
theorem shaded_triangle_area 
  (n : ℕ) (area_small : ℕ) (area_total : ℕ) (K : ℕ)
  (H1 : n = 36)
  (H2 : area_small = 10)
  (H3 : area_total = n * area_small)
  (H4 : K = 110)
: K = 110 :=
by
  -- Adding 'sorry' indicating missing proof steps.
  sorry

end shaded_triangle_area_l389_38981


namespace tim_grew_cantaloupes_l389_38990

theorem tim_grew_cantaloupes (fred_cantaloupes : ℕ) (total_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : total_cantaloupes = 82) :
  ∃ tim_cantaloupes : ℕ, tim_cantaloupes = total_cantaloupes - fred_cantaloupes ∧ tim_cantaloupes = 44 :=
by
  sorry

end tim_grew_cantaloupes_l389_38990


namespace geometric_sequence_cannot_determine_a3_l389_38988

/--
Suppose we have a geometric sequence {a_n} such that 
the product of the first five terms a_1 * a_2 * a_3 * a_4 * a_5 = 32.
We aim to show that the value of a_3 cannot be determined with the given information.
-/
theorem geometric_sequence_cannot_determine_a3 (a : ℕ → ℝ) (r : ℝ) (h : a 0 * a 1 * a 2 * a 3 * a 4 = 32) : 
  ¬ ∃ x : ℝ, a 2 = x :=
sorry

end geometric_sequence_cannot_determine_a3_l389_38988


namespace principal_amount_l389_38918

theorem principal_amount (SI P R T : ℝ) 
  (h1 : R = 12) (h2 : T = 3) (h3 : SI = 3600) : 
  SI = P * R * T / 100 → P = 10000 :=
by
  intros h
  sorry

end principal_amount_l389_38918


namespace remainder_form_l389_38908

open Polynomial Int

-- Define the conditions
variable (f : Polynomial ℤ)
variable (h1 : ∀ n : ℤ, 3 ∣ eval n f)

-- Define the proof problem statement
theorem remainder_form (h1 : ∀ n : ℤ, 3 ∣ eval n f) :
  ∃ (M r : Polynomial ℤ), f = (X^3 - X) * M + C 3 * r :=
sorry

end remainder_form_l389_38908


namespace f_plus_2012_odd_l389_38932

def f : ℝ → ℝ → ℝ := sorry

lemma f_property (α β : ℝ) : f α β = 2012 := sorry

theorem f_plus_2012_odd : ∀ x : ℝ, f (-x) + 2012 = -(f x + 2012) :=
by
  sorry

end f_plus_2012_odd_l389_38932


namespace sufficient_but_not_necessary_not_necessary_l389_38901

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : (|a| > 0) := by
  sorry

theorem not_necessary (a : ℝ) : |a| > 0 → ¬(a = 0) ∧ (a ≠ 0 → |a| > 0 ∧ (¬(a > 0) → (|a| > 0))) := by
  sorry

end sufficient_but_not_necessary_not_necessary_l389_38901


namespace average_difference_l389_38904

def daily_differences : List ℤ := [2, -1, 3, 1, -2, 2, 1]

theorem average_difference :
  (daily_differences.sum : ℚ) / daily_differences.length = 0.857 :=
by
  sorry

end average_difference_l389_38904


namespace stuffed_animal_ratio_l389_38998

theorem stuffed_animal_ratio
  (K : ℕ)
  (h1 : 34 + K + (K + 5) = 175) :
  K / 34 = 2 :=
by sorry

end stuffed_animal_ratio_l389_38998


namespace complement_of_A_l389_38923

def U : Set ℤ := {-1, 2, 4}
def A : Set ℤ := {-1, 4}

theorem complement_of_A : U \ A = {2} := by
  sorry

end complement_of_A_l389_38923


namespace sum_A_J_l389_38939

variable (A B C D E F G H I J : ℕ)

-- Conditions
axiom h1 : C = 7
axiom h2 : A + B + C = 40
axiom h3 : B + C + D = 40
axiom h4 : C + D + E = 40
axiom h5 : D + E + F = 40
axiom h6 : E + F + G = 40
axiom h7 : F + G + H = 40
axiom h8 : G + H + I = 40
axiom h9 : H + I + J = 40

-- Proof statement
theorem sum_A_J : A + J = 33 :=
by
  sorry

end sum_A_J_l389_38939


namespace total_trip_length_is570_l389_38954

theorem total_trip_length_is570 (v D : ℝ) (h1 : (2:ℝ) + (2/3) + (6 * (D - 2 * v) / (5 * v)) = 2.75)
(h2 : (2:ℝ) + (50 / v) + (2/3) + (6 * (D - 2 * v - 50) / (5 * v)) = 2.33) :
D = 570 :=
sorry

end total_trip_length_is570_l389_38954


namespace part_a_part_b_l389_38945

variable (a b : ℝ)

-- Given conditions
variable (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4)

-- Requirement (a): Prove that a > b
theorem part_a : a > b := by 
  sorry

-- Requirement (b): Prove that a^2 + b^2 ≥ 2
theorem part_b : a^2 + b^2 ≥ 2 := by 
  sorry

end part_a_part_b_l389_38945


namespace equal_distribution_l389_38935

namespace MoneyDistribution

def Ann_initial := 777
def Bill_initial := 1111
def Charlie_initial := 1555
def target_amount := 1148
def Bill_to_Ann := 371
def Charlie_to_Bill := 408

theorem equal_distribution :
  (Bill_initial - Bill_to_Ann + Charlie_to_Bill = target_amount) ∧
  (Ann_initial + Bill_to_Ann = target_amount) ∧
  (Charlie_initial - Charlie_to_Bill = target_amount) :=
by
  sorry

end MoneyDistribution

end equal_distribution_l389_38935


namespace find_m_interval_l389_38947

def seq (x : ℕ → ℚ) : Prop :=
  (x 0 = 7) ∧ (∀ n : ℕ, x (n + 1) = (x n ^ 2 + 8 * x n + 9) / (x n + 7))

def m_spec (x : ℕ → ℚ) (m : ℕ) : Prop :=
  (x m ≤ 5 + 1 / 2^15)

theorem find_m_interval :
  ∃ (x : ℕ → ℚ) (m : ℕ), seq x ∧ m_spec x m ∧ 81 ≤ m ∧ m ≤ 242 :=
sorry

end find_m_interval_l389_38947


namespace quadratic_inequality_solution_l389_38905

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : (∀ x, ax^2 + bx + c = 0 ↔ x = 1 ∨ x = 3)) : 
  ∀ x, cx^2 + bx + a > 0 ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l389_38905


namespace inequality_proof_l389_38916

theorem inequality_proof (n : ℕ) (hn : n > 0) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end inequality_proof_l389_38916


namespace total_people_on_hike_l389_38992

def cars : Nat := 3
def people_per_car : Nat := 4
def taxis : Nat := 6
def people_per_taxi : Nat := 6
def vans : Nat := 2
def people_per_van : Nat := 5

theorem total_people_on_hike :
  cars * people_per_car + taxis * people_per_taxi + vans * people_per_van = 58 := by
  sorry

end total_people_on_hike_l389_38992


namespace find_explicit_formula_l389_38959

variable (f : ℝ → ℝ)

theorem find_explicit_formula 
  (h : ∀ x : ℝ, f (x - 1) = 2 * x^2 - 8 * x + 11) :
  ∀ x : ℝ, f x = 2 * x^2 - 4 * x + 5 :=
by
  sorry

end find_explicit_formula_l389_38959


namespace range_of_a_l389_38962

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3 - a) → (∀ x : ℝ, f 0 ≤ 0) → (0 ≤ a) :=
by
  intro h1 h2
  suffices h : -a ≤ 0 by
    simpa using h
  have : f 0 = -a
  simp [h1]
  sorry -- Proof steps are omitted

end range_of_a_l389_38962


namespace largest_rectangle_area_l389_38965

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end largest_rectangle_area_l389_38965


namespace water_left_after_operations_l389_38950

theorem water_left_after_operations :
  let initial_water := (3 : ℚ)
  let water_used := (4 / 3 : ℚ)
  let extra_water := (1 / 2 : ℚ)
  initial_water - water_used + extra_water = (13 / 6 : ℚ) := 
by
  -- Skips the proof, as the focus is on the problem statement
  sorry

end water_left_after_operations_l389_38950


namespace solve_x_l389_38920

theorem solve_x (x : ℝ) :
  (5 + 2 * x) / (7 + 3 * x) = (4 + 3 * x) / (9 + 4 * x) ↔
  x = (-5 + Real.sqrt 93) / 2 ∨ x = (-5 - Real.sqrt 93) / 2 :=
by
  sorry

end solve_x_l389_38920


namespace stratified_sample_sum_l389_38952

theorem stratified_sample_sum :
  let grains := 40
  let veg_oils := 10
  let animal_foods := 30
  let fruits_veggies := 20
  let total_varieties := grains + veg_oils + animal_foods + fruits_veggies
  let sample_size := 20
  let veg_oils_proportion := (veg_oils:ℚ) / total_varieties
  let fruits_veggies_proportion := (fruits_veggies:ℚ) / total_varieties
  let veg_oils_sample := sample_size * veg_oils_proportion
  let fruits_veggies_sample := sample_size * fruits_veggies_proportion
  veg_oils_sample + fruits_veggies_sample = 6 := sorry

end stratified_sample_sum_l389_38952


namespace ratio_of_population_l389_38980

theorem ratio_of_population (Z : ℕ) :
  let Y := 2 * Z
  let X := 3 * Y
  let W := X + Y
  X / (Z + W) = 2 / 3 :=
by
  sorry

end ratio_of_population_l389_38980


namespace sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l389_38949

theorem sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions :
  ∃ (a b : ℕ), (4 < a ∧ a < b ∧ b < 16) ∧
  (∃ r : ℚ, a = 4 * r ∧ b = 4 * r * r) ∧
  (a + b = 2 * b - a + 16) ∧
  a + b = 24 :=
by
  sorry

end sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l389_38949


namespace Lance_must_read_today_l389_38903

def total_pages : ℕ := 100
def pages_read_yesterday : ℕ := 35
def pages_read_tomorrow : ℕ := 27

noncomputable def pages_read_today : ℕ :=
  pages_read_yesterday - 5

noncomputable def pages_left_today : ℕ :=
  total_pages - (pages_read_yesterday + pages_read_today + pages_read_tomorrow)

theorem Lance_must_read_today :
  pages_read_today + pages_left_today = 38 :=
by 
  rw [pages_read_today, pages_left_today, pages_read_yesterday, pages_read_tomorrow, total_pages]
  simp
  sorry

end Lance_must_read_today_l389_38903


namespace flight_up_speed_l389_38921

variable (v : ℝ) -- speed on the flight up
variable (d : ℝ) -- distance to mother's place

/--
Given:
1. The speed on the way home was 72 mph.
2. The average speed for the trip was 91 mph.

Prove:
The speed on the flight up was 123.62 mph.
-/
theorem flight_up_speed
  (h1 : 72 > 0)
  (h2 : 91 > 0)
  (avg_speed_def : 91 = (2 * d) / ((d / v) + (d / 72))) :
  v = 123.62 :=
by
  sorry

end flight_up_speed_l389_38921


namespace find_difference_of_squares_l389_38977

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l389_38977


namespace quadratic_root_m_value_l389_38906

theorem quadratic_root_m_value (m : ℝ) (x : ℝ) (h : x = 1) (hx : x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end quadratic_root_m_value_l389_38906


namespace solve_quadratic_equation_l389_38960

theorem solve_quadratic_equation (x : ℝ) : 4 * (2 * x + 1) ^ 2 = 9 * (x - 3) ^ 2 ↔ x = -11 ∨ x = 1 := 
by sorry

end solve_quadratic_equation_l389_38960


namespace total_outfits_l389_38909

-- Define the quantities of each item.
def red_shirts : ℕ := 7
def green_shirts : ℕ := 8
def pants : ℕ := 10
def blue_hats : ℕ := 10
def red_hats : ℕ := 10
def scarves : ℕ := 5

-- The total number of outfits without having the same color of shirts and hats.
theorem total_outfits : 
  (red_shirts * pants * blue_hats * scarves) + (green_shirts * pants * red_hats * scarves) = 7500 := 
by sorry

end total_outfits_l389_38909


namespace angle_2016_in_third_quadrant_l389_38997

def quadrant (θ : ℤ) : ℤ :=
  let angle := θ % 360
  if 0 ≤ angle ∧ angle < 90 then 1
  else if 90 ≤ angle ∧ angle < 180 then 2
  else if 180 ≤ angle ∧ angle < 270 then 3
  else 4

theorem angle_2016_in_third_quadrant : 
  quadrant 2016 = 3 := 
by
  sorry

end angle_2016_in_third_quadrant_l389_38997


namespace total_leftover_tarts_l389_38929

def cherry_tarts := 0.08
def blueberry_tarts := 0.75
def peach_tarts := 0.08

theorem total_leftover_tarts : cherry_tarts + blueberry_tarts + peach_tarts = 0.91 := by
  sorry

end total_leftover_tarts_l389_38929


namespace no_adjacent_performers_probability_l389_38967

-- A definition to model the probability of non-adjacent performers in a circle of 6 people.
def probability_no_adjacent_performers : ℚ :=
  -- Given conditions: fair coin tosses by six people, modeling permutations
  -- and specific valid configurations derived from the problem.
  9 / 32

-- Proving the final probability calculation is correct
theorem no_adjacent_performers_probability :
  probability_no_adjacent_performers = 9 / 32 :=
by
  -- Using sorry to indicate the proof needs to be filled in, acknowledging the correct answer.
  sorry

end no_adjacent_performers_probability_l389_38967


namespace proposition_not_true_at_3_l389_38934

variable (P : ℕ → Prop)

theorem proposition_not_true_at_3
  (h1 : ∀ k : ℕ, P k → P (k + 1))
  (h2 : ¬ P 4) :
  ¬ P 3 :=
sorry

end proposition_not_true_at_3_l389_38934


namespace solution_l389_38974

-- Define the linear equations and their solutions
def system_of_equations (x y : ℕ) :=
  3 * x + y = 500 ∧ x + 2 * y = 250

-- Define the budget constraint
def budget_constraint (m : ℕ) :=
  150 * m + 50 * (25 - m) ≤ 2700

-- Define the purchasing plans and costs
def purchasing_plans (m n : ℕ) :=
  (m = 12 ∧ n = 13 ∧ 150 * m + 50 * n = 2450) ∨ 
  (m = 13 ∧ n = 12 ∧ 150 * m + 50 * n = 2550) ∨ 
  (m = 14 ∧ n = 11 ∧ 150 * m + 50 * n = 2650)

-- Define the Lean statement
theorem solution :
  (∃ x y, system_of_equations x y ∧ x = 150 ∧ y = 50) ∧
  (∃ m, budget_constraint m ∧ m ≤ 14) ∧
  (∃ m n, 12 ≤ m ∧ m ≤ 14 ∧ m + n = 25 ∧ purchasing_plans m n ∧ 150 * m + 50 * n = 2450) :=
sorry

end solution_l389_38974


namespace find_pencils_l389_38983

theorem find_pencils :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (6 ∣ n) ∧ (9 ∣ n) ∧ n % 7 = 1 ∧ n = 36 :=
by
  sorry

end find_pencils_l389_38983


namespace matrix_B_pow_66_l389_38927

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0], 
    ![-1, 0, 0], 
    ![0, 0, 1]]

theorem matrix_B_pow_66 : B^66 = ![![-1, 0, 0], ![0, -1, 0], ![0, 0, 1]] := by
  sorry

end matrix_B_pow_66_l389_38927


namespace find_lightest_bead_l389_38919

theorem find_lightest_bead (n : ℕ) (h : 0 < n) (H : ∀ b1 b2 b3 : ℕ, b1 + b2 + b3 = n → b1 > 0 ∧ b2 > 0 ∧ b3 > 0 → b1 ≤ 3 ∧ b2 ≤ 9 ∧ b3 ≤ 27) : n = 27 :=
sorry

end find_lightest_bead_l389_38919


namespace b_gives_c_start_l389_38999

variable (Va Vb Vc : ℝ)

-- Conditions given in the problem
def condition1 : Prop := Va / Vb = 1000 / 930
def condition2 : Prop := Va / Vc = 1000 / 800
def race_distance : ℝ := 1000

-- Proposition to prove
theorem b_gives_c_start (h1 : condition1 Va Vb) (h2 : condition2 Va Vc) :
  ∃ x : ℝ, (1000 - x) / 1000 = (930 / 800) :=
sorry

end b_gives_c_start_l389_38999


namespace polynomial_problem_l389_38984

noncomputable def F (x : ℝ) : ℝ := sorry

theorem polynomial_problem
  (F : ℝ → ℝ)
  (h1 : F 4 = 22)
  (h2 : ∀ x : ℝ, (F (2 * x) / F (x + 2) = 4 - (16 * x + 8) / (x^2 + x + 1))) :
  F 8 = 1078 / 9 := sorry

end polynomial_problem_l389_38984


namespace value_of_expression_l389_38914

theorem value_of_expression : 50^4 + 4 * 50^3 + 6 * 50^2 + 4 * 50 + 1 = 6765201 :=
by
  sorry

end value_of_expression_l389_38914


namespace mary_final_books_l389_38979

def mary_initial_books := 5
def mary_first_return := 3
def mary_first_checkout := 5
def mary_second_return := 2
def mary_second_checkout := 7

theorem mary_final_books :
  (mary_initial_books - mary_first_return + mary_first_checkout - mary_second_return + mary_second_checkout) = 12 := 
by 
  sorry

end mary_final_books_l389_38979


namespace work_problem_l389_38911

theorem work_problem (hA : ∀ n : ℝ, n = 15)
  (h_work_together : ∀ n : ℝ, 3 * (1/15 + 1/n) = 0.35) :  
  1/20 = 1/20 :=
by
  sorry

end work_problem_l389_38911


namespace sin_alpha_third_quadrant_l389_38951

theorem sin_alpha_third_quadrant 
  (α : ℝ) 
  (hcos : Real.cos α = -3 / 5) 
  (hquad : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.sin α = -4 / 5 := 
sorry

end sin_alpha_third_quadrant_l389_38951


namespace Bryce_grapes_l389_38971

theorem Bryce_grapes : 
  ∃ x : ℝ, (∀ y : ℝ, y = (1/3) * x → y = x - 7) → x = 21 / 2 :=
by
  sorry

end Bryce_grapes_l389_38971


namespace sufficient_but_not_necessary_condition_l389_38942

noncomputable def f (a x : ℝ) := x^2 + 2 * a * x - 2

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, x ≤ -2 → deriv (f a) x ≤ 0) ↔ a = 2 :=
sorry

end sufficient_but_not_necessary_condition_l389_38942


namespace fraction_power_l389_38936

variables (a b c : ℝ)

theorem fraction_power :
  ( ( -2 * a^2 * b ) / (3 * c) )^2 = ( 4 * a^4 * b^2 ) / ( 9 * c^2 ) := 
by sorry

end fraction_power_l389_38936


namespace difference_area_octagon_shaded_l389_38925

-- Definitions based on the given conditions
def radius : ℝ := 10
def pi_value : ℝ := 3.14

-- Lean statement for the given proof problem
theorem difference_area_octagon_shaded :
  ∃ S_octagon S_shaded, 
    10^2 * pi_value = 314 ∧
    (20 / 2^0.5)^2 = 200 ∧
    S_octagon = 200 - 114 ∧ -- transposed to reverse engineering step
    S_shaded = 28 ∧ -- needs refinement
    S_octagon - S_shaded = 86 :=
sorry

end difference_area_octagon_shaded_l389_38925


namespace arithmetic_sequence_value_l389_38963

theorem arithmetic_sequence_value (a : ℕ) (h : 2 * a = 12) : a = 6 :=
by
  sorry

end arithmetic_sequence_value_l389_38963


namespace father_present_age_l389_38956

theorem father_present_age (S F : ℕ) 
  (h1 : F = 3 * S + 3) 
  (h2 : F + 3 = 2 * (S + 3) + 10) : 
  F = 33 :=
by
  sorry

end father_present_age_l389_38956


namespace total_number_of_animals_l389_38961

-- Definitions for the number of each type of animal
def cats : ℕ := 645
def dogs : ℕ := 567
def rabbits : ℕ := 316
def reptiles : ℕ := 120

-- The statement to prove
theorem total_number_of_animals :
  cats + dogs + rabbits + reptiles = 1648 := by
  sorry

end total_number_of_animals_l389_38961


namespace unique_real_root_eq_l389_38938

theorem unique_real_root_eq (x : ℝ) : (∃! x, x = Real.sin x + 1993) :=
sorry

end unique_real_root_eq_l389_38938


namespace arithmetic_mean_of_fractions_l389_38996

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l389_38996


namespace remainder_div_modulo_l389_38933

theorem remainder_div_modulo (N : ℕ) (h1 : N % 19 = 7) : N % 20 = 6 :=
by
  sorry

end remainder_div_modulo_l389_38933


namespace initial_mat_weavers_eq_4_l389_38944

theorem initial_mat_weavers_eq_4 :
  ∃ x : ℕ, (x * 4 = 4) ∧ (14 * 14 = 49) ∧ (x = 4) :=
sorry

end initial_mat_weavers_eq_4_l389_38944


namespace evaluate_f_at_t_plus_one_l389_38982

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 1

-- Define the proposition to be proved
theorem evaluate_f_at_t_plus_one (t : ℝ) : f (t + 1) = 3 * t + 2 := by
  sorry

end evaluate_f_at_t_plus_one_l389_38982


namespace first_character_more_lines_than_second_l389_38973

theorem first_character_more_lines_than_second :
  let x := 2
  let second_character_lines := 3 * x + 6
  20 - second_character_lines = 8 := by
  sorry

end first_character_more_lines_than_second_l389_38973


namespace johnny_selection_process_l389_38907

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem johnny_selection_process : 
  binomial_coefficient 10 4 * binomial_coefficient 4 2 = 1260 :=
by
  sorry

end johnny_selection_process_l389_38907


namespace length_of_second_platform_l389_38957

-- Definitions
def length_train : ℝ := 230
def time_first_platform : ℝ := 15
def length_first_platform : ℝ := 130
def total_distance_first_platform : ℝ := length_train + length_first_platform
def time_second_platform : ℝ := 20

-- Statement to prove
theorem length_of_second_platform : 
  ∃ L : ℝ, (total_distance_first_platform / time_first_platform) = ((length_train + L) / time_second_platform) ∧ L = 250 :=
by
  sorry

end length_of_second_platform_l389_38957


namespace find_radius_k_l389_38995

/-- Mathematical conditions for the given geometry problem -/
structure problem_conditions where
  radius_F : ℝ := 15
  radius_G : ℝ := 4
  radius_H : ℝ := 3
  radius_I : ℝ := 3
  radius_J : ℝ := 1

/-- Proof problem statement defining the required theorem -/
theorem find_radius_k (conditions : problem_conditions) :
  let r := (137:ℝ) / 8
  20 * r = (342.5 : ℝ) :=
by
  sorry

end find_radius_k_l389_38995


namespace track_circumference_l389_38948

theorem track_circumference (x : ℕ) 
  (A_B_uniform_speeds_opposite : True) 
  (diametrically_opposite_start : True) 
  (same_start_time : True) 
  (first_meeting_B_150_yards : True) 
  (second_meeting_A_90_yards_before_complete_lap : True) : 
  2 * x = 720 :=
by
  sorry

end track_circumference_l389_38948


namespace painting_time_eq_l389_38912

theorem painting_time_eq (t : ℚ) : 
  (1/6 + 1/8 + 1/10) * (t - 2) = 1 := 
sorry

end painting_time_eq_l389_38912


namespace power_function_expression_l389_38989

theorem power_function_expression (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 2 = 4) :
  α = 2 ∧ (∀ x, f x = x ^ 2) :=
by
  sorry

end power_function_expression_l389_38989


namespace Aaron_initial_erasers_l389_38943

/-- 
  Given:
  - Aaron gives 34 erasers to Doris.
  - Aaron ends with 47 erasers.
  Prove:
  - Aaron started with 81 erasers.
-/ 
theorem Aaron_initial_erasers (gives : ℕ) (ends : ℕ) (start : ℕ) :
  gives = 34 → ends = 47 → start = ends + gives → start = 81 :=
by
  intros h_gives h_ends h_start
  sorry

end Aaron_initial_erasers_l389_38943


namespace parabola_one_intersection_l389_38966

theorem parabola_one_intersection (k : ℝ) :
  (∀ x : ℝ, x^2 - x + k = 0 → x = 0) → k = 1 / 4 :=
sorry

end parabola_one_intersection_l389_38966


namespace min_value_of_quadratic_expression_l389_38953

variable (x y z : ℝ)

theorem min_value_of_quadratic_expression 
  (h1 : 2 * x + 2 * y + z + 8 = 0) : 
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 = 9 :=
sorry

end min_value_of_quadratic_expression_l389_38953


namespace convert_base_10_to_base_5_l389_38958

theorem convert_base_10_to_base_5 :
  (256 : ℕ) = 2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0 :=
by
  sorry

end convert_base_10_to_base_5_l389_38958


namespace magnitude_of_vec_sum_l389_38946

noncomputable def vec_a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def vec_b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))
noncomputable def vec_sum : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_vec_sum : magnitude vec_sum = Real.sqrt 7 := 
by 
  sorry

end magnitude_of_vec_sum_l389_38946


namespace john_money_l389_38941

theorem john_money (cost_given : ℝ) : cost_given = 14 :=
by
  have gift_cost := 28
  have half_cost := gift_cost / 2
  exact sorry

end john_money_l389_38941


namespace rectangle_area_perimeter_max_l389_38910

-- Define the problem conditions
variables {A P : ℝ}

-- Main statement: prove that the maximum value of A / P^2 for a rectangle results in m+n = 17
theorem rectangle_area_perimeter_max (h1 : A = l * w) (h2 : P = 2 * (l + w)) :
  let m := 1
  let n := 16
  m + n = 17 :=
sorry

end rectangle_area_perimeter_max_l389_38910


namespace cookie_radius_l389_38900

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 13 := 
sorry

end cookie_radius_l389_38900


namespace isosceles_triangle_base_angle_l389_38924

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle : a + b + c = 180)
  (h_iso : a = b ∨ b = c ∨ a = c) (h_interior : a = 50 ∨ b = 50 ∨ c = 50) :
  c = 50 ∨ c = 65 :=
by sorry

end isosceles_triangle_base_angle_l389_38924


namespace arithmetic_seq_first_term_l389_38940

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (a : ℚ) (n : ℕ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
  (h2 : ∀ n, S (4 * n) / S n = 16) : a = 5 / 2 := 
sorry

end arithmetic_seq_first_term_l389_38940


namespace smallest_value_l389_38978

theorem smallest_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
    (h1 : a = 2 * b) (h2 : b = 2 * c) (h3 : 4 * c = a) :
    (Int.floor ((a + b : ℚ) / c) + Int.floor ((b + c : ℚ) / a) + Int.floor ((c + a : ℚ) / b)) = 8 := 
sorry

end smallest_value_l389_38978


namespace min_value_l389_38931

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1 / (a + 3) + 1 / (b + 3) + 1 / (c + 3) = 1 / 4) : 
  22.75 ≤ a + 3 * b + 2 * c := 
sorry

end min_value_l389_38931


namespace regular_polygon_sides_l389_38937

theorem regular_polygon_sides (n : ℕ) (h : 108 = 180 * (n - 2) / n) : n = 5 := 
sorry

end regular_polygon_sides_l389_38937


namespace min_value_f_when_a_eq_one_range_of_a_for_inequality_l389_38913

noncomputable def f (x a : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Question 1: When a = 1, find the minimum value of the function f(x)
theorem min_value_f_when_a_eq_one : ∃ x : ℝ, ∀ y : ℝ, f y 1 ≥ f x 1 ∧ f x 1 = 4 :=
by
  sorry

-- Question 2: For which values of a does f(x) ≥ 4/a + 1 hold for all real numbers x
theorem range_of_a_for_inequality : (∀ x : ℝ, f x a ≥ 4 / a + 1) ↔ (a < 0 ∨ a = 2) :=
by
  sorry

end min_value_f_when_a_eq_one_range_of_a_for_inequality_l389_38913


namespace sum_of_xy_l389_38917

theorem sum_of_xy (x y : ℝ) (h1 : x^3 - 6*x^2 + 12*x = 13) (h2 : y^3 + 3*y - 3*y^2 = -4) : x + y = 3 :=
by sorry

end sum_of_xy_l389_38917
