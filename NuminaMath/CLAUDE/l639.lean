import Mathlib

namespace NUMINAMATH_CALUDE_range_of_g_l639_63931

theorem range_of_g (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  -Real.pi^2 / 2 ≤ (Real.arccos x)^2 - (Real.arcsin x)^2 ∧ 
  (Real.arccos x)^2 - (Real.arcsin x)^2 ≤ Real.pi^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l639_63931


namespace NUMINAMATH_CALUDE_joan_bought_six_dozens_l639_63943

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Joan bought -/
def total_eggs : ℕ := 72

/-- The number of dozens of eggs Joan bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem joan_bought_six_dozens : dozens_bought = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_bought_six_dozens_l639_63943


namespace NUMINAMATH_CALUDE_photo_arrangement_l639_63975

theorem photo_arrangement (n_male : ℕ) (n_female : ℕ) : 
  n_male = 4 → n_female = 2 → (
    (3 : ℕ) *           -- ways to place "甲" in middle positions
    (4 : ℕ).factorial * -- ways to arrange remaining units
    (2 : ℕ).factorial   -- ways to arrange female students within their unit
  ) = 144 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_l639_63975


namespace NUMINAMATH_CALUDE_sum_minimized_at_24_l639_63990

/-- The sum of the first n terms of an arithmetic sequence with general term a_n = 2n - 49 -/
def S (n : ℕ) : ℝ := n^2 - 48*n

/-- The value of n that minimizes S_n -/
def n_min : ℕ := 24

theorem sum_minimized_at_24 :
  ∀ n : ℕ, n ≠ 0 → S n ≥ S n_min := by sorry

end NUMINAMATH_CALUDE_sum_minimized_at_24_l639_63990


namespace NUMINAMATH_CALUDE_identity_proof_l639_63997

theorem identity_proof (a b c d x y z u : ℝ) :
  (a*x + b*y + c*z + d*u)^2 + (b*x + c*y + d*z + a*u)^2 + (c*x + d*y + a*z + b*u)^2 + (d*x + a*y + b*z + c*u)^2
  = (d*x + c*y + b*z + a*u)^2 + (c*x + b*y + a*z + d*u)^2 + (b*x + a*y + d*z + c*u)^2 + (a*x + d*y + c*z + b*u)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l639_63997


namespace NUMINAMATH_CALUDE_value_of_x_l639_63911

theorem value_of_x : ∃ x : ℝ, (0.25 * x = 0.15 * 1500 - 20) ∧ x = 820 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l639_63911


namespace NUMINAMATH_CALUDE_problem_statement_l639_63929

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (9 / a + 1 / b ≥ 4) ∧ ((a + 3 / b) * (b + 3 / a) ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l639_63929


namespace NUMINAMATH_CALUDE_calculation_proof_l639_63922

theorem calculation_proof :
  (Real.sqrt 48 * Real.sqrt (1/2) + Real.sqrt 12 + Real.sqrt 24 = 4 * Real.sqrt 6 + 2 * Real.sqrt 3) ∧
  ((Real.sqrt 5 + 1)^2 + (Real.sqrt 13 + 3) * (Real.sqrt 13 - 3) = 10 + 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l639_63922


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l639_63950

theorem compare_negative_fractions : -10/11 > -11/12 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l639_63950


namespace NUMINAMATH_CALUDE_least_months_to_triple_l639_63901

theorem least_months_to_triple (rate : ℝ) (triple : ℝ) : ∃ (n : ℕ), n > 0 ∧ (1 + rate)^n > triple ∧ ∀ (m : ℕ), m > 0 → m < n → (1 + rate)^m ≤ triple :=
  by
  -- Let rate be 0.06 (6%) and triple be 3
  have h1 : rate = 0.06 := by sorry
  have h2 : triple = 3 := by sorry
  
  -- The answer is 19
  use 19
  
  sorry -- Skip the proof

end NUMINAMATH_CALUDE_least_months_to_triple_l639_63901


namespace NUMINAMATH_CALUDE_box_volume_in_cubic_yards_l639_63962

-- Define the conversion factor from feet to yards
def feet_to_yards : ℝ := 3

-- Define the volume of the box in cubic feet
def box_volume_cubic_feet : ℝ := 216

-- Theorem to prove
theorem box_volume_in_cubic_yards :
  box_volume_cubic_feet / (feet_to_yards ^ 3) = 8 := by
  sorry


end NUMINAMATH_CALUDE_box_volume_in_cubic_yards_l639_63962


namespace NUMINAMATH_CALUDE_f_at_2_l639_63964

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem f_at_2 : f 2 = 123 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l639_63964


namespace NUMINAMATH_CALUDE_abs_diff_inequality_l639_63956

theorem abs_diff_inequality (x : ℝ) : |x| - |x - 3| < 2 ↔ x < (5/2) := by sorry

end NUMINAMATH_CALUDE_abs_diff_inequality_l639_63956


namespace NUMINAMATH_CALUDE_greatest_value_l639_63906

theorem greatest_value (p : ℝ) (a b c d : ℝ) 
  (h1 : a + 1 = p) 
  (h2 : b - 2 = p) 
  (h3 : c + 3 = p) 
  (h4 : d - 4 = p) : 
  d > a ∧ d > b ∧ d > c :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_l639_63906


namespace NUMINAMATH_CALUDE_smallest_norm_w_l639_63979

variable (w : ℝ × ℝ)

def v : ℝ × ℝ := (4, 2)

theorem smallest_norm_w (h : ‖w + v‖ = 10) :
  ∃ (w_min : ℝ × ℝ), ‖w_min‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ w', ‖w' + v‖ = 10 → ‖w'‖ ≥ ‖w_min‖ :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_w_l639_63979


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l639_63914

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (9 * a^3 - 27 * a + 54 = 0) →
  (9 * b^3 - 27 * b + 54 = 0) →
  (9 * c^3 - 27 * c + 54 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l639_63914


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complements_l639_63976

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x ≤ 3} := by sorry

-- Theorem for (C_U A) ∪ (C_U B)
theorem union_complements : (Set.univ \ A) ∪ (Set.univ \ B) = {x | x ≤ 1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complements_l639_63976


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l639_63968

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x ≥ a^2 - a} = {a : ℝ | -1 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l639_63968


namespace NUMINAMATH_CALUDE_box_edge_length_and_capacity_l639_63919

/-- Given a cubical box that can contain 999.9999999999998 cubes of 10 cm edge length,
    prove that its edge length is 1 meter and it can contain 1000 cubes. -/
theorem box_edge_length_and_capacity (box_capacity : ℝ) 
  (h1 : box_capacity = 999.9999999999998) : ∃ (edge_length : ℝ),
  edge_length = 1 ∧ 
  (edge_length * 100 / 10)^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_box_edge_length_and_capacity_l639_63919


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l639_63965

/-- A square inscribed in a right triangle with one vertex at the right angle -/
def square_in_triangle_vertex (a b c : ℝ) (x : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (b - x) / x = a / b

/-- A square inscribed in a right triangle with one side on the hypotenuse -/
def square_in_triangle_hypotenuse (a b c : ℝ) (y : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ y / c = (b - y) / a

theorem inscribed_squares_ratio :
  ∀ (x y : ℝ),
    square_in_triangle_vertex 5 12 13 x →
    square_in_triangle_hypotenuse 6 8 10 y →
    x / y = 216 / 85 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l639_63965


namespace NUMINAMATH_CALUDE_cyclists_distance_l639_63918

/-- Calculates the distance between two cyclists traveling in opposite directions -/
def distance_between_cyclists (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem stating the distance between two cyclists after 2 hours -/
theorem cyclists_distance :
  let speed1 : ℝ := 10  -- Speed of first cyclist in km/h
  let speed2 : ℝ := 15  -- Speed of second cyclist in km/h
  let time : ℝ := 2     -- Time in hours
  distance_between_cyclists speed1 speed2 time = 50 := by
  sorry

#check cyclists_distance

end NUMINAMATH_CALUDE_cyclists_distance_l639_63918


namespace NUMINAMATH_CALUDE_brownies_per_person_l639_63926

theorem brownies_per_person (columns rows people : ℕ) 
  (h1 : columns = 6)
  (h2 : rows = 3)
  (h3 : people = 6) :
  (columns * rows) / people = 3 := by
sorry

end NUMINAMATH_CALUDE_brownies_per_person_l639_63926


namespace NUMINAMATH_CALUDE_multiply_37_23_l639_63969

theorem multiply_37_23 : 37 * 23 = 851 := by
  sorry

end NUMINAMATH_CALUDE_multiply_37_23_l639_63969


namespace NUMINAMATH_CALUDE_total_food_count_l639_63915

/-- The total number of hotdogs and hamburgers brought by neighbors -/
theorem total_food_count : ℕ := by
  -- Define the number of hotdogs brought by each neighbor
  let first_neighbor_hotdogs : ℕ := 75
  let second_neighbor_hotdogs : ℕ := first_neighbor_hotdogs - 25
  let third_neighbor_hotdogs : ℕ := 35
  let fourth_neighbor_hotdogs : ℕ := 2 * third_neighbor_hotdogs

  -- Define the number of hamburgers brought
  let one_neighbor_hamburgers : ℕ := 60
  let another_neighbor_hamburgers : ℕ := 3 * one_neighbor_hamburgers

  -- Calculate total hotdogs and hamburgers
  let total_hotdogs : ℕ := first_neighbor_hotdogs + second_neighbor_hotdogs + 
                           third_neighbor_hotdogs + fourth_neighbor_hotdogs
  let total_hamburgers : ℕ := one_neighbor_hamburgers + another_neighbor_hamburgers
  let total_food : ℕ := total_hotdogs + total_hamburgers

  -- Prove that the total is 470
  have : total_food = 470 := by sorry

  exact 470

end NUMINAMATH_CALUDE_total_food_count_l639_63915


namespace NUMINAMATH_CALUDE_train_length_l639_63986

/-- The length of a train given specific crossing times -/
theorem train_length (tree_crossing_time platform_crossing_time platform_length : ℝ) 
  (h1 : tree_crossing_time = 120)
  (h2 : platform_crossing_time = 240)
  (h3 : platform_length = 1200) : 
  ∃ (train_length : ℝ), train_length = 1200 ∧ 
    (train_length / tree_crossing_time) * platform_crossing_time = train_length + platform_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l639_63986


namespace NUMINAMATH_CALUDE_six_by_six_grid_squares_l639_63998

/-- The number of squares of a given size in a 6x6 grid -/
def squares_of_size (n : Nat) : Nat :=
  (7 - n) * (7 - n)

/-- The total number of squares in a 6x6 grid -/
def total_squares : Nat :=
  (squares_of_size 1) + (squares_of_size 2) + (squares_of_size 3) + 
  (squares_of_size 4) + (squares_of_size 5)

/-- Theorem: The total number of squares in a 6x6 grid is 55 -/
theorem six_by_six_grid_squares : total_squares = 55 := by
  sorry

end NUMINAMATH_CALUDE_six_by_six_grid_squares_l639_63998


namespace NUMINAMATH_CALUDE_jake_second_test_difference_l639_63996

def jake_test_scores (test1 test2 test3 test4 : ℕ) : Prop :=
  test1 = 80 ∧ 
  test3 = 65 ∧ 
  test3 = test4 ∧ 
  (test1 + test2 + test3 + test4) / 4 = 75

theorem jake_second_test_difference :
  ∀ test1 test2 test3 test4 : ℕ,
    jake_test_scores test1 test2 test3 test4 →
    test2 - test1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jake_second_test_difference_l639_63996


namespace NUMINAMATH_CALUDE_chandler_savings_weeks_l639_63993

def bike_cost : ℕ := 650
def birthday_money : ℕ := 50 + 35 + 15 + 20
def weekly_earnings : ℕ := 18

def weeks_to_save (cost birthday_money weekly_earnings : ℕ) : ℕ :=
  ((cost - birthday_money + weekly_earnings - 1) / weekly_earnings)

theorem chandler_savings_weeks :
  weeks_to_save bike_cost birthday_money weekly_earnings = 30 := by
  sorry

end NUMINAMATH_CALUDE_chandler_savings_weeks_l639_63993


namespace NUMINAMATH_CALUDE_remainder_equivalence_l639_63966

theorem remainder_equivalence (x : ℤ) : x % 5 = 4 → x % 61 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equivalence_l639_63966


namespace NUMINAMATH_CALUDE_min_value_expression_l639_63942

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b * c = 1) (h2 : a / b = 2) :
  a^2 + 4*a*b + 9*b^2 + 8*b*c + 3*c^2 ≥ 3 * (63 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l639_63942


namespace NUMINAMATH_CALUDE_f_difference_l639_63960

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as sigma(n) / n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(540) - f(180) = 7/90 -/
theorem f_difference : f 540 - f 180 = 7 / 90 := by sorry

end NUMINAMATH_CALUDE_f_difference_l639_63960


namespace NUMINAMATH_CALUDE_favorite_fruit_oranges_l639_63944

theorem favorite_fruit_oranges (total students_pears students_apples students_strawberries : ℕ) 
  (h_total : total = 450)
  (h_pears : students_pears = 120)
  (h_apples : students_apples = 147)
  (h_strawberries : students_strawberries = 113) :
  total - (students_pears + students_apples + students_strawberries) = 70 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_oranges_l639_63944


namespace NUMINAMATH_CALUDE_tim_manicure_payment_l639_63936

/-- The total amount paid for a manicure with tip, given the base cost and tip percentage. -/
def total_paid (base_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  base_cost * (1 + tip_percentage)

/-- Theorem stating that the total amount Tim paid for the manicure is $39. -/
theorem tim_manicure_payment : total_paid 30 0.3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tim_manicure_payment_l639_63936


namespace NUMINAMATH_CALUDE_equation_solution_l639_63961

theorem equation_solution : ∃ y : ℚ, (40 / 60 = Real.sqrt (y / 60)) ∧ y = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l639_63961


namespace NUMINAMATH_CALUDE_james_total_spending_l639_63954

def club_entry_fee : ℕ := 20
def friends_count : ℕ := 5
def rounds_for_friends : ℕ := 2
def james_drinks : ℕ := 6
def drink_cost : ℕ := 6
def food_cost : ℕ := 14
def tip_percentage : ℚ := 30 / 100

def total_drinks : ℕ := friends_count * rounds_for_friends + james_drinks

def order_cost : ℕ := total_drinks * drink_cost + food_cost

def tip_amount : ℚ := (order_cost : ℚ) * tip_percentage

def total_spending : ℚ := (club_entry_fee : ℚ) + (order_cost : ℚ) + tip_amount

theorem james_total_spending :
  total_spending = 163 := by sorry

end NUMINAMATH_CALUDE_james_total_spending_l639_63954


namespace NUMINAMATH_CALUDE_log_equation_solution_l639_63928

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → log 4 (x^3) + log (1/4) x = 12 → x = 4096 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l639_63928


namespace NUMINAMATH_CALUDE_square_area_percentage_l639_63916

/-- Given a rectangle enclosing a square, this theorem proves the percentage
    of the rectangle's area occupied by the square. -/
theorem square_area_percentage (s : ℝ) (h1 : s > 0) : 
  let w := 3 * s  -- width of rectangle
  let l := 3 * w / 2  -- length of rectangle
  let square_area := s^2
  let rectangle_area := l * w
  (square_area / rectangle_area) * 100 = 200 / 27 := by sorry

end NUMINAMATH_CALUDE_square_area_percentage_l639_63916


namespace NUMINAMATH_CALUDE_total_food_consumption_l639_63927

/-- Calculates the total daily food consumption for two armies with different rations -/
theorem total_food_consumption
  (food_per_soldier_side1 : ℕ)
  (food_difference : ℕ)
  (soldiers_side1 : ℕ)
  (soldier_difference : ℕ)
  (h1 : food_per_soldier_side1 = 10)
  (h2 : food_difference = 2)
  (h3 : soldiers_side1 = 4000)
  (h4 : soldier_difference = 500) :
  let food_per_soldier_side2 := food_per_soldier_side1 - food_difference
  let soldiers_side2 := soldiers_side1 - soldier_difference
  soldiers_side1 * food_per_soldier_side1 + soldiers_side2 * food_per_soldier_side2 = 68000 := by
sorry


end NUMINAMATH_CALUDE_total_food_consumption_l639_63927


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l639_63985

/-- Represents a geometric sequence -/
structure GeometricSequence where
  -- The sequence function
  a : ℕ → ℝ
  -- The common ratio
  r : ℝ
  -- The geometric sequence property
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence where the 5th term is -2 and the 8th term is -54, the 11th term is -1458 -/
theorem geometric_sequence_11th_term
  (seq : GeometricSequence)
  (h5 : seq.a 5 = -2)
  (h8 : seq.a 8 = -54) :
  seq.a 11 = -1458 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l639_63985


namespace NUMINAMATH_CALUDE_call_charge_relationship_l639_63948

/-- Represents the charge per call when exceeding 200 calls in a month -/
structure CallCharge where
  february : ℝ
  march : ℝ

/-- Represents the monthly bill for a customer -/
structure MonthlyBill where
  fixedRental : ℝ
  freeCallLimit : ℕ
  callsMade : ℕ
  chargePerExcessCall : ℝ

theorem call_charge_relationship (c : CallCharge) (febBill marchBill : MonthlyBill) :
  febBill.fixedRental = 350 ∧
  febBill.freeCallLimit = 200 ∧
  febBill.callsMade = 150 ∧
  marchBill.fixedRental = 350 ∧
  marchBill.freeCallLimit = 200 ∧
  marchBill.callsMade = 250 ∧
  c.february = febBill.chargePerExcessCall ∧
  c.march = 0.72 * c.february →
  c.march = marchBill.chargePerExcessCall :=
by
  sorry

end NUMINAMATH_CALUDE_call_charge_relationship_l639_63948


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_l639_63952

/-- The distance between Maxwell's and Brad's homes in kilometers -/
def total_distance : ℝ := 36

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- The distance traveled by Maxwell when they meet -/
def maxwell_distance : ℝ := 12

theorem maxwell_brad_meeting :
  maxwell_distance * brad_speed = (total_distance - maxwell_distance) * maxwell_speed :=
by sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_l639_63952


namespace NUMINAMATH_CALUDE_inequality_proof_l639_63957

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  (a^2 + b^2 + c^2 + d^2 ≥ 1/4) ∧ 
  (a^2/b + b^2/c + c^2/d + d^2/a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l639_63957


namespace NUMINAMATH_CALUDE_andrew_work_days_l639_63905

/-- Given that Andrew worked 2.5 hours each day and a total of 7.5 hours,
    prove that he spent 3 days working on the report. -/
theorem andrew_work_days (hours_per_day : ℝ) (total_hours : ℝ) 
    (h1 : hours_per_day = 2.5)
    (h2 : total_hours = 7.5) :
    total_hours / hours_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_andrew_work_days_l639_63905


namespace NUMINAMATH_CALUDE_jakes_weight_l639_63949

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 8 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 278) :
  jake_weight = 188 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l639_63949


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_21_over_5_l639_63946

theorem greatest_integer_less_than_negative_21_over_5 :
  Int.floor (-21 / 5 : ℚ) = -5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_21_over_5_l639_63946


namespace NUMINAMATH_CALUDE_negate_negate_eq_self_l639_63973

theorem negate_negate_eq_self (n : ℤ) : -(-n) = n := by sorry

end NUMINAMATH_CALUDE_negate_negate_eq_self_l639_63973


namespace NUMINAMATH_CALUDE_james_sticker_cost_l639_63945

/-- Calculates James's share of the cost for stickers --/
theorem james_sticker_cost (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) : 
  packs = 4 → 
  stickers_per_pack = 30 → 
  cost_per_sticker = 1/10 →
  (packs * stickers_per_pack * cost_per_sticker) / 2 = 6 := by
  sorry

#check james_sticker_cost

end NUMINAMATH_CALUDE_james_sticker_cost_l639_63945


namespace NUMINAMATH_CALUDE_exist_six_games_twelve_players_l639_63932

structure Tournament where
  players : Finset ℕ
  games : Finset (ℕ × ℕ)
  player_in_game : ∀ p ∈ players, ∃ g ∈ games, p ∈ g.1 :: g.2 :: []

theorem exist_six_games_twelve_players (t : Tournament) 
  (h1 : t.players.card = 20)
  (h2 : t.games.card = 14) :
  ∃ (subset_games : Finset (ℕ × ℕ)) (subset_players : Finset ℕ),
    subset_games ⊆ t.games ∧
    subset_games.card = 6 ∧
    subset_players ⊆ t.players ∧
    subset_players.card = 12 ∧
    ∀ g ∈ subset_games, g.1 ∈ subset_players ∧ g.2 ∈ subset_players :=
sorry

end NUMINAMATH_CALUDE_exist_six_games_twelve_players_l639_63932


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l639_63923

/-- For a parallelogram with area 162 sq m and base 9 m, the ratio of altitude to base is 2/1 -/
theorem parallelogram_altitude_base_ratio :
  ∀ (area base altitude : ℝ),
    area = 162 →
    base = 9 →
    area = base * altitude →
    altitude / base = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l639_63923


namespace NUMINAMATH_CALUDE_sqrt_of_four_l639_63984

theorem sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l639_63984


namespace NUMINAMATH_CALUDE_magical_stack_size_l639_63940

/-- A stack of cards is magical if it satisfies certain conditions -/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (card_197_position : ℕ)
  (card_197_retains_position : card_197_position = 197)
  (is_magical : ∃ (a b : ℕ), a ≤ n ∧ b > n ∧ b ≤ total_cards)

/-- The number of cards in a magical stack where card 197 retains its position is 590 -/
theorem magical_stack_size (stack : MagicalStack) : stack.total_cards = 590 :=
by sorry

end NUMINAMATH_CALUDE_magical_stack_size_l639_63940


namespace NUMINAMATH_CALUDE_binomial_equation_unique_solution_l639_63924

theorem binomial_equation_unique_solution :
  ∃! n : ℕ, (Nat.choose 15 n + Nat.choose 15 7 = Nat.choose 16 8) ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_equation_unique_solution_l639_63924


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l639_63958

theorem min_sum_with_reciprocal_constraint (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : 
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 9/w = 1 → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 9/b = 1 ∧ a + b = 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l639_63958


namespace NUMINAMATH_CALUDE_lending_duration_l639_63989

/-- Proves that the number of years the first part is lent is 8, given the problem conditions -/
theorem lending_duration (total_sum : ℚ) (second_part : ℚ) 
  (first_rate : ℚ) (second_rate : ℚ) (second_duration : ℚ) :
  total_sum = 2678 →
  second_part = 1648 →
  first_rate = 3/100 →
  second_rate = 5/100 →
  second_duration = 3 →
  ∃ (first_duration : ℚ),
    (total_sum - second_part) * first_rate * first_duration = 
    second_part * second_rate * second_duration ∧
    first_duration = 8 := by
  sorry

end NUMINAMATH_CALUDE_lending_duration_l639_63989


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l639_63995

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digits_sum (n : ℕ) : ℕ :=
  let d₁ := n / 10
  let d₂ := n % 10
  d₁ + d₂

def strictly_increasing_digits (n : ℕ) : Prop :=
  let d₁ := n / 10
  let d₂ := n % 10
  d₁ < d₂

theorem smallest_fourth_number :
  ∃ (n : ℕ),
    is_two_digit n ∧
    strictly_increasing_digits n ∧
    (∀ m, is_two_digit m → strictly_increasing_digits m →
      digits_sum 34 + digits_sum 18 + digits_sum 73 + digits_sum n +
      digits_sum m = (34 + 18 + 73 + n + m) / 6 →
      n ≤ m) ∧
    digits_sum 34 + digits_sum 18 + digits_sum 73 + digits_sum n =
      (34 + 18 + 73 + n) / 6 ∧
    n = 29 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l639_63995


namespace NUMINAMATH_CALUDE_tom_batteries_total_l639_63904

/-- The total number of batteries Tom used is 19, given the number of batteries used for each category. -/
theorem tom_batteries_total (flashlight_batteries : ℕ) (toy_batteries : ℕ) (controller_batteries : ℕ)
  (h1 : flashlight_batteries = 2)
  (h2 : toy_batteries = 15)
  (h3 : controller_batteries = 2) :
  flashlight_batteries + toy_batteries + controller_batteries = 19 := by
  sorry

end NUMINAMATH_CALUDE_tom_batteries_total_l639_63904


namespace NUMINAMATH_CALUDE_triangle_area_circumradius_angles_l639_63977

theorem triangle_area_circumradius_angles 
  (α β γ : Real) (R : Real) (S_Δ : Real) :
  (α + β + γ = π) →
  (R > 0) →
  (S_Δ > 0) →
  (S_Δ = 2 * R^2 * Real.sin α * Real.sin β * Real.sin γ) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_circumradius_angles_l639_63977


namespace NUMINAMATH_CALUDE_c_minus_three_equals_negative_two_l639_63978

/-- An invertible function g : ℝ → ℝ -/
def g : ℝ → ℝ :=
  sorry

/-- c is a real number such that g(c) = 3 and g(3) = c -/
def c : ℝ :=
  sorry

theorem c_minus_three_equals_negative_two (h1 : Function.Injective g) (h2 : g c = 3) (h3 : g 3 = c) :
  c - 3 = -2 :=
sorry

end NUMINAMATH_CALUDE_c_minus_three_equals_negative_two_l639_63978


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l639_63947

theorem complex_number_in_fourth_quadrant (m : ℝ) (h : 1 < m ∧ m < 2) :
  let z : ℂ := Complex.mk (m - 1) (m - 2)
  0 < z.re ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l639_63947


namespace NUMINAMATH_CALUDE_postcard_collection_average_l639_63910

/-- 
Given an arithmetic sequence with:
- First term: 10
- Common difference: 12
- Number of terms: 7
Prove that the average of all terms is 46.
-/
theorem postcard_collection_average : 
  let first_term := 10
  let common_diff := 12
  let num_days := 7
  let last_term := first_term + (num_days - 1) * common_diff
  (first_term + last_term) / 2 = 46 := by
sorry

end NUMINAMATH_CALUDE_postcard_collection_average_l639_63910


namespace NUMINAMATH_CALUDE_men_entered_room_l639_63907

/-- Proves that 2 men entered the room given the initial and final conditions --/
theorem men_entered_room : 
  ∀ (initial_men initial_women : ℕ),
  initial_men / initial_women = 4 / 5 →
  ∃ (men_entered : ℕ),
  2 * (initial_women - 3) = 24 ∧
  initial_men + men_entered = 14 →
  men_entered = 2 := by
sorry

end NUMINAMATH_CALUDE_men_entered_room_l639_63907


namespace NUMINAMATH_CALUDE_rectangle_midpoint_distances_l639_63900

theorem rectangle_midpoint_distances (a b : ℝ) (ha : a = 3) (hb : b = 5) :
  let vertex := (0 : ℝ × ℝ)
  let midpoints := [
    (a / 2, 0),
    (a, b / 2),
    (a / 2, b),
    (0, b / 2)
  ]
  (midpoints.map (λ m => Real.sqrt ((m.1 - vertex.1)^2 + (m.2 - vertex.2)^2))).sum = 13.1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_midpoint_distances_l639_63900


namespace NUMINAMATH_CALUDE_betty_height_in_feet_betty_is_three_feet_tall_l639_63937

/-- Given a dog's height, Carter's height relative to the dog, and Betty's height relative to Carter,
    calculate Betty's height in feet. -/
theorem betty_height_in_feet (dog_height : ℕ) (carter_ratio : ℕ) (betty_diff : ℕ) : ℕ :=
  let carter_height := dog_height * carter_ratio
  let betty_height_inches := carter_height - betty_diff
  betty_height_inches / 12

/-- Prove that Betty is 3 feet tall given the specific conditions. -/
theorem betty_is_three_feet_tall :
  betty_height_in_feet 24 2 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_betty_height_in_feet_betty_is_three_feet_tall_l639_63937


namespace NUMINAMATH_CALUDE_shaded_area_proof_l639_63909

theorem shaded_area_proof (square_side : ℝ) (triangle_side : ℝ) : 
  square_side = 40 →
  triangle_side = 25 →
  square_side^2 - 2 * (1/2 * triangle_side^2) = 975 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l639_63909


namespace NUMINAMATH_CALUDE_fred_has_four_dimes_l639_63938

/-- The number of dimes Fred has after his sister borrowed some -/
def fred_remaining_dimes (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Fred has 4 dimes after his sister borrowed 3 from his initial 7 -/
theorem fred_has_four_dimes :
  fred_remaining_dimes 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_four_dimes_l639_63938


namespace NUMINAMATH_CALUDE_field_ratio_proof_l639_63981

/-- Proves that for a rectangular field with length 24 meters and width 13.5 meters,
    the ratio of twice the width to the length is 9:8. -/
theorem field_ratio_proof (length width : ℝ) : 
  length = 24 → width = 13.5 → (2 * width) / length = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_proof_l639_63981


namespace NUMINAMATH_CALUDE_sum_of_symmetric_roots_l639_63933

/-- A function f: ℝ → ℝ that satisfies f(1-x) = f(1+x) for all real x -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - x) = f (1 + x)

/-- The theorem stating that if f is symmetric about 1 and has exactly 2009 real roots,
    then the sum of these roots is 2009 -/
theorem sum_of_symmetric_roots
  (f : ℝ → ℝ)
  (h_sym : SymmetricAboutOne f)
  (h_roots : ∃! (s : Finset ℝ), s.card = 2009 ∧ ∀ x ∈ s, f x = 0) :
  ∃ (s : Finset ℝ), s.card = 2009 ∧ (∀ x ∈ s, f x = 0) ∧ (s.sum id = 2009) :=
sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_roots_l639_63933


namespace NUMINAMATH_CALUDE_g_of_two_eq_zero_l639_63939

/-- The function g(x) = x^2 - 4x + 4 -/
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Theorem: g(2) = 0 -/
theorem g_of_two_eq_zero : g 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_eq_zero_l639_63939


namespace NUMINAMATH_CALUDE_tin_addition_theorem_l639_63983

/-- Proves that adding 1.5 kg of pure tin to a 12 kg alloy containing 45% copper 
    will result in a new alloy containing 40% copper. -/
theorem tin_addition_theorem (initial_mass : ℝ) (initial_copper_percentage : ℝ) 
    (final_copper_percentage : ℝ) (tin_added : ℝ) : 
    initial_mass = 12 →
    initial_copper_percentage = 0.45 →
    final_copper_percentage = 0.4 →
    tin_added = 1.5 →
    initial_mass * initial_copper_percentage = 
    final_copper_percentage * (initial_mass + tin_added) := by
  sorry

#check tin_addition_theorem

end NUMINAMATH_CALUDE_tin_addition_theorem_l639_63983


namespace NUMINAMATH_CALUDE_sum_of_c_values_l639_63959

theorem sum_of_c_values : ∃ (S : Finset ℤ),
  (∀ c ∈ S, c ≤ 30 ∧ 
    ∃ x y : ℚ, y = x^2 - 8*x - c ∧ 
    ∃ k : ℤ, (64 + 4*c = k^2)) ∧
  (∀ c : ℤ, c ≤ 30 → 
    (∃ x y : ℚ, y = x^2 - 8*x - c ∧ 
    ∃ k : ℤ, (64 + 4*c = k^2)) → 
    c ∈ S) ∧
  S.sum id = -11 :=
sorry

end NUMINAMATH_CALUDE_sum_of_c_values_l639_63959


namespace NUMINAMATH_CALUDE_disjunction_not_implies_both_true_l639_63982

theorem disjunction_not_implies_both_true :
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by sorry

end NUMINAMATH_CALUDE_disjunction_not_implies_both_true_l639_63982


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l639_63912

def total_marbles : ℕ := 240

theorem yellow_marbles_count (y b : ℕ) 
  (h1 : y + b = total_marbles) 
  (h2 : b = y - 2) : 
  y = 121 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l639_63912


namespace NUMINAMATH_CALUDE_savings_exceed_500_on_sunday_l639_63970

/-- The day of the week, starting from Sunday as 0 -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculate the total savings after n days -/
def totalSavings (n : ℕ) : ℚ :=
  (3^n - 1) / 2

/-- Convert number of days to day of the week -/
def toDayOfWeek (n : ℕ) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem savings_exceed_500_on_sunday :
  ∃ n : ℕ, totalSavings n > 500 ∧
    ∀ m : ℕ, m < n → totalSavings m ≤ 500 ∧
    toDayOfWeek n = DayOfWeek.Sunday :=
by sorry

end NUMINAMATH_CALUDE_savings_exceed_500_on_sunday_l639_63970


namespace NUMINAMATH_CALUDE_exponent_problem_l639_63935

theorem exponent_problem (x y : ℝ) (m n : ℕ) (h : x ≠ 0) (h' : y ≠ 0) :
  x^m * y^n / ((1/4) * x^3 * y) = 4 * x^2 → m = 5 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l639_63935


namespace NUMINAMATH_CALUDE_fraction_value_l639_63974

theorem fraction_value (a b c d : ℚ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l639_63974


namespace NUMINAMATH_CALUDE_scale_length_l639_63988

/-- The total length of a scale divided into equal parts -/
def total_length (num_parts : ℕ) (part_length : ℝ) : ℝ :=
  num_parts * part_length

/-- Theorem: The total length of a scale is 80 inches -/
theorem scale_length : total_length 4 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_l639_63988


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l639_63963

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ a m * a n = 16 * (a 1)^2) :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ a m * a n = 16 * (a 1)^2 ∧
    ∀ k l : ℕ, k > 0 → l > 0 → a k * a l = 16 * (a 1)^2 →
      1 / m + 4 / n ≤ 1 / k + 4 / l :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l639_63963


namespace NUMINAMATH_CALUDE_brownies_per_person_l639_63925

/-- Given a pan of brownies cut into columns and rows, calculate how many brownies each person can eat. -/
theorem brownies_per_person 
  (columns : ℕ) 
  (rows : ℕ) 
  (people : ℕ) 
  (h1 : columns = 6) 
  (h2 : rows = 3) 
  (h3 : people = 6) 
  : (columns * rows) / people = 3 := by
  sorry

end NUMINAMATH_CALUDE_brownies_per_person_l639_63925


namespace NUMINAMATH_CALUDE_function_property_implies_odd_l639_63992

theorem function_property_implies_odd (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y = f (x + y)) : 
  ∀ x : ℝ, f (-x) = -f x := by
sorry

end NUMINAMATH_CALUDE_function_property_implies_odd_l639_63992


namespace NUMINAMATH_CALUDE_sharons_salary_increase_l639_63941

theorem sharons_salary_increase (S : ℝ) (x : ℝ) : 
  S * 1.08 = 324 → S * (1 + x / 100) = 330 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sharons_salary_increase_l639_63941


namespace NUMINAMATH_CALUDE_task_completion_ways_l639_63934

theorem task_completion_ways (m₁ m₂ : ℕ) : ∃ N : ℕ, N = m₁ + m₂ := by
  sorry

end NUMINAMATH_CALUDE_task_completion_ways_l639_63934


namespace NUMINAMATH_CALUDE_min_value_expression_l639_63971

/-- Given two positive real numbers m and n, and two vectors a and b that are perpendicular,
    prove that the minimum value of 1/m + 2/n is 3 + 2√2 -/
theorem min_value_expression (m n : ℝ) (a b : ℝ × ℝ) 
  (hm : m > 0) (hn : n > 0)
  (ha : a = (m, 1)) (hb : b = (1, n - 1))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  (∀ x y, x > 0 → y > 0 → 1/x + 2/y ≥ 1/m + 2/n) → 1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l639_63971


namespace NUMINAMATH_CALUDE_divisibility_criterion_l639_63980

theorem divisibility_criterion (A m k : ℕ) (h_pos : A > 0) (h_m_pos : m > 0) (h_k_pos : k > 0) :
  let g := k * m + 1
  let remainders : List ℕ := sorry
  let sum_remainders := remainders.sum
  (A % m = 0) ↔ (sum_remainders % m = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l639_63980


namespace NUMINAMATH_CALUDE_minimum_fifth_quarter_score_l639_63987

def required_average : ℚ := 85
def num_quarters : ℕ := 5
def first_four_scores : List ℚ := [84, 80, 78, 82]

theorem minimum_fifth_quarter_score :
  let total_required := required_average * num_quarters
  let sum_first_four := first_four_scores.sum
  let min_fifth_score := total_required - sum_first_four
  min_fifth_score = 101 := by sorry

end NUMINAMATH_CALUDE_minimum_fifth_quarter_score_l639_63987


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l639_63951

theorem ceiling_floor_product_range (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 210 → -15 < y ∧ y < -14 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l639_63951


namespace NUMINAMATH_CALUDE_equation_solutions_and_first_m_first_m_above_1959_l639_63917

theorem equation_solutions_and_first_m (m n : ℕ+) :
  (8 * m - 7 = n^2) ↔ 
  (∃ s : ℕ, m = 1 + s * (s + 1) / 2 ∧ n = 2 * s + 1) :=
sorry

theorem first_m_above_1959 :
  (∃ m₀ : ℕ+, m₀ > 1959 ∧ 
   (∀ m : ℕ+, m > 1959 ∧ (∃ n : ℕ+, 8 * m - 7 = n^2) → m ≥ m₀) ∧
   m₀ = 2017) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_and_first_m_first_m_above_1959_l639_63917


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l639_63921

theorem inequality_implies_lower_bound (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, 4^x - 2^(x+1) - a ≤ 0) → a ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l639_63921


namespace NUMINAMATH_CALUDE_rabbit_walk_prob_l639_63920

/-- A random walk on a rectangular grid. -/
structure RandomWalk where
  width : ℕ
  height : ℕ
  start_x : ℕ
  start_y : ℕ

/-- The probability of ending on the top or bottom edge for a given random walk. -/
noncomputable def prob_top_bottom (walk : RandomWalk) : ℚ :=
  sorry

/-- The specific random walk described in the problem. -/
def rabbit_walk : RandomWalk :=
  { width := 6
    height := 5
    start_x := 2
    start_y := 3 }

/-- The main theorem stating the probability for the specific random walk. -/
theorem rabbit_walk_prob : prob_top_bottom rabbit_walk = 17 / 24 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_walk_prob_l639_63920


namespace NUMINAMATH_CALUDE_water_leak_proof_l639_63955

/-- A linear function representing the total water amount over time -/
def water_function (k b : ℝ) (t : ℝ) : ℝ := k * t + b

theorem water_leak_proof (k b : ℝ) :
  water_function k b 1 = 7 →
  water_function k b 2 = 12 →
  (k = 5 ∧ b = 2) ∧
  water_function k b 20 = 102 ∧
  ((water_function k b 1440 * 30) / 1500 : ℝ) = 144 :=
by sorry


end NUMINAMATH_CALUDE_water_leak_proof_l639_63955


namespace NUMINAMATH_CALUDE_samuel_coaching_fee_l639_63903

/-- Calculates the number of days in a month, assuming a non-leap year -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 1 | 3 | 5 | 7 | 8 | 10 | 12 => 31
  | 4 | 6 | 9 | 11 => 30
  | 2 => 28
  | _ => 0

/-- Calculates the total number of days from January 1 to a given date -/
def daysFromNewYear (month : Nat) (day : Nat) : Nat :=
  (List.range (month - 1)).foldl (fun acc m => acc + daysInMonth (m + 1)) day

/-- Represents the coaching period and daily fee -/
structure CoachingData where
  startMonth : Nat
  startDay : Nat
  endMonth : Nat
  endDay : Nat
  dailyFee : Nat

/-- Calculates the total coaching fee -/
def totalCoachingFee (data : CoachingData) : Nat :=
  let totalDays := daysFromNewYear data.endMonth data.endDay - daysFromNewYear data.startMonth data.startDay + 1
  totalDays * data.dailyFee

/-- Theorem: The total coaching fee for Samuel is 7084 dollars -/
theorem samuel_coaching_fee :
  let data : CoachingData := {
    startMonth := 1,
    startDay := 1,
    endMonth := 11,
    endDay := 4,
    dailyFee := 23
  }
  totalCoachingFee data = 7084 := by
  sorry


end NUMINAMATH_CALUDE_samuel_coaching_fee_l639_63903


namespace NUMINAMATH_CALUDE_negation_equivalence_l639_63953

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 4*x₀ + 1 < 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 4*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l639_63953


namespace NUMINAMATH_CALUDE_log_equation_solution_l639_63991

theorem log_equation_solution (b x : ℝ) 
  (h1 : b > 0) 
  (h2 : b ≠ 1) 
  (h3 : x ≠ 1) 
  (h4 : Real.log x / Real.log (b^3) + Real.log b / Real.log (x^3) = 1) : 
  x = b^((3 + Real.sqrt 5) / 2) ∨ x = b^((3 - Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l639_63991


namespace NUMINAMATH_CALUDE_number_fraction_problem_l639_63908

theorem number_fraction_problem (n : ℝ) : 
  (1/3) * (1/4) * (1/5) * n = 15 → (3/10) * n = 270 := by
sorry

end NUMINAMATH_CALUDE_number_fraction_problem_l639_63908


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_l639_63994

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x : ℝ | -a < x ∧ x ≤ a + 3}

-- Theorem statement
theorem set_operations_and_intersection (a : ℝ) : 
  (A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 8}) ∧ 
  ((Aᶜ : Set ℝ) ∩ B = {x : ℝ | 5 ≤ x ∧ x < 8}) ∧ 
  (C a ∩ A = C a ↔ a ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_l639_63994


namespace NUMINAMATH_CALUDE_complex_number_equality_l639_63913

theorem complex_number_equality (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / Complex.I
  (z.re = z.im) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l639_63913


namespace NUMINAMATH_CALUDE_mean_proportional_segment_l639_63999

theorem mean_proportional_segment (a b c : ℝ) : 
  a = 1 → b = 2 → c^2 = a * b → c > 0 → c = Real.sqrt 2 := by
  sorry

#check mean_proportional_segment

end NUMINAMATH_CALUDE_mean_proportional_segment_l639_63999


namespace NUMINAMATH_CALUDE_quadratic_linear_intersection_l639_63902

/-- Quadratic function -/
def y1 (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Linear function -/
def y2 (a b x : ℝ) : ℝ := a * x + b

/-- Theorem stating the main results -/
theorem quadratic_linear_intersection 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : y1 a b c 1 = 0) 
  (t : ℤ) 
  (h4 : t % 2 = 1) 
  (h5 : y1 a b c (t : ℝ) = 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ y1 a b c x1 = y2 a b x1 ∧ y1 a b c x2 = y2 a b x2) ∧ 
  (t = 1 ∨ t = -1) ∧
  (∀ A1 B1 : ℝ, y1 a b c A1 = y2 a b A1 → y1 a b c B1 = y2 a b B1 → 
    3/2 < |A1 - B1| ∧ |A1 - B1| < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_linear_intersection_l639_63902


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l639_63930

/-- Represents the number of employees in each title category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Represents the sample size for each title category -/
structure SampleSize where
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Calculates the stratified sample size for a given category -/
def stratifiedSampleSize (totalEmployees : ℕ) (categoryCount : ℕ) (sampleSize : ℕ) : ℕ :=
  (sampleSize * categoryCount) / totalEmployees

/-- Theorem: The stratified sampling results in the correct sample sizes -/
theorem stratified_sampling_correct 
  (employees : EmployeeCount) 
  (sample : SampleSize) : 
  employees.total = 150 ∧ 
  employees.senior = 15 ∧ 
  employees.intermediate = 45 ∧ 
  employees.junior = 90 ∧
  sample.senior = stratifiedSampleSize employees.total employees.senior 30 ∧
  sample.intermediate = stratifiedSampleSize employees.total employees.intermediate 30 ∧
  sample.junior = stratifiedSampleSize employees.total employees.junior 30 →
  sample.senior = 3 ∧ sample.intermediate = 9 ∧ sample.junior = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l639_63930


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l639_63967

theorem simplify_complex_fraction :
  1 / ((1 / (Real.sqrt 3 + 1)) + (2 / (Real.sqrt 5 - 1))) = 
  (Real.sqrt 3 + 2 * Real.sqrt 5 - 1) / (2 + 4 * Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l639_63967


namespace NUMINAMATH_CALUDE_triangle_theorem_l639_63972

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.c / (Real.sqrt 3 * Real.cos t.C) ∧
  t.a + t.b = 6 ∧
  t.a * t.b * Real.cos t.C = 4

-- State the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = π / 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l639_63972
