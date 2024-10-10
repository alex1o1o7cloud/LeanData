import Mathlib

namespace gumble_words_count_l1709_170925

def alphabet_size : ℕ := 25
def max_word_length : ℕ := 5

def words_with_b (n : ℕ) : ℕ :=
  alphabet_size^n - (alphabet_size - 1)^n

def total_words : ℕ :=
  words_with_b 1 + words_with_b 2 + words_with_b 3 + words_with_b 4 + words_with_b 5

theorem gumble_words_count :
  total_words = 1863701 :=
by sorry

end gumble_words_count_l1709_170925


namespace building_floor_ratio_l1709_170988

/-- Given three buildings A, B, and C, where:
  * Building A has 4 floors
  * Building B has 9 more floors than Building A
  * Building C has 59 floors
Prove that the ratio of floors in Building C to Building B is 59/13 -/
theorem building_floor_ratio : 
  (floors_A : ℕ) → 
  (floors_B : ℕ) → 
  (floors_C : ℕ) → 
  floors_A = 4 →
  floors_B = floors_A + 9 →
  floors_C = 59 →
  (floors_C : ℚ) / floors_B = 59 / 13 := by
sorry

end building_floor_ratio_l1709_170988


namespace monkeys_count_l1709_170923

theorem monkeys_count (termites : ℕ) (total_workers : ℕ) (h1 : termites = 622) (h2 : total_workers = 861) :
  total_workers - termites = 239 := by
  sorry

end monkeys_count_l1709_170923


namespace units_digit_of_n_squared_plus_two_to_n_l1709_170900

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) : 
  n = 1234^2 + 2^1234 → (n^2 + 2^n) % 10 = 1 := by
  sorry

end units_digit_of_n_squared_plus_two_to_n_l1709_170900


namespace two_digit_subtraction_pattern_l1709_170909

theorem two_digit_subtraction_pattern (a b : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) :
  (10 * a + b) - (10 * b + a) = 9 * (a - b) :=
by sorry

end two_digit_subtraction_pattern_l1709_170909


namespace lottery_expected_profit_l1709_170903

/-- The expected profit for buying one lottery ticket -/
theorem lottery_expected_profit :
  let ticket_cost : ℝ := 10
  let win_probability : ℝ := 0.02
  let prize : ℝ := 300
  let expected_profit := (prize - ticket_cost) * win_probability + (-ticket_cost) * (1 - win_probability)
  expected_profit = -4 := by sorry

end lottery_expected_profit_l1709_170903


namespace duke_record_breaking_l1709_170948

/-- Duke's basketball record breaking proof --/
theorem duke_record_breaking (points_to_tie : ℕ) (old_record : ℕ) 
  (free_throws : ℕ) (regular_baskets : ℕ) (normal_three_pointers : ℕ) :
  points_to_tie = 17 →
  old_record = 257 →
  free_throws = 5 →
  regular_baskets = 4 →
  normal_three_pointers = 2 →
  (free_throws * 1 + regular_baskets * 2 + (normal_three_pointers + 1) * 3) - points_to_tie = 5 := by
  sorry

#check duke_record_breaking

end duke_record_breaking_l1709_170948


namespace greatest_multiple_under_1000_l1709_170939

theorem greatest_multiple_under_1000 : ∃ (n : ℕ), n = 945 ∧ 
  n < 1000 ∧ 
  3 ∣ n ∧ 
  5 ∣ n ∧ 
  7 ∣ n ∧ 
  ∀ m : ℕ, m < 1000 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → m ≤ n :=
by sorry

end greatest_multiple_under_1000_l1709_170939


namespace max_ratio_three_digit_number_l1709_170951

theorem max_ratio_three_digit_number :
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    0 ≤ c ∧ c ≤ 9 →
    let N := 100 * a + 10 * b + c
    let S := a + b + c
    (N : ℚ) / S ≤ 100 ∧ 
    (∃ a' b' c', 
      1 ≤ a' ∧ a' ≤ 9 ∧ 
      0 ≤ b' ∧ b' ≤ 9 ∧ 
      0 ≤ c' ∧ c' ≤ 9 ∧ 
      let N' := 100 * a' + 10 * b' + c'
      let S' := a' + b' + c'
      (N' : ℚ) / S' = 100) :=
by sorry

end max_ratio_three_digit_number_l1709_170951


namespace bar_chart_best_for_rainfall_l1709_170915

-- Define the characteristics of the data
structure RainfallData where
  area : String
  seasons : Fin 4 → Float
  isRainfall : Bool

-- Define the types of charts
inductive ChartType
  | Bar
  | Line
  | Pie

-- Define a function to determine the best chart type
def bestChartType (data : RainfallData) : ChartType :=
  ChartType.Bar

-- Theorem stating that bar chart is the best choice for rainfall data
theorem bar_chart_best_for_rainfall (data : RainfallData) :
  data.isRainfall = true → bestChartType data = ChartType.Bar :=
by
  sorry

#check bar_chart_best_for_rainfall

end bar_chart_best_for_rainfall_l1709_170915


namespace right_side_difference_l1709_170919

/-- A triangle with specific side lengths -/
structure Triangle where
  left : ℝ
  right : ℝ
  base : ℝ

/-- The properties of our specific triangle -/
def special_triangle (t : Triangle) : Prop :=
  t.left = 12 ∧ 
  t.base = 24 ∧ 
  t.left + t.right + t.base = 50 ∧
  t.right > t.left

theorem right_side_difference (t : Triangle) (h : special_triangle t) : 
  t.right - t.left = 2 := by
  sorry

end right_side_difference_l1709_170919


namespace max_n_with_special_divisors_l1709_170969

theorem max_n_with_special_divisors (N : ℕ) : 
  (∃ (d : ℕ), d ∣ N ∧ d ≠ 1 ∧ d ≠ N ∧
   (∃ (a b : ℕ), a ∣ N ∧ b ∣ N ∧ a < b ∧
    (∀ (x : ℕ), x ∣ N → x < a ∨ x > b) ∧
    b = 21 * d)) →
  N ≤ 441 :=
sorry

end max_n_with_special_divisors_l1709_170969


namespace division_problem_l1709_170987

theorem division_problem (x y z : ℝ) (h1 : x / y = 3) (h2 : y / z = 5/2) : 
  z / x = 2/15 := by sorry

end division_problem_l1709_170987


namespace complex_modulus_problem_l1709_170981

theorem complex_modulus_problem (m : ℝ) :
  (Complex.I : ℂ) * Complex.I = -1 →
  (↑1 + m * Complex.I) * (↑3 + Complex.I) = Complex.I * (Complex.im ((↑1 + m * Complex.I) * (↑3 + Complex.I))) →
  Complex.abs ((↑m + ↑3 * Complex.I) / (↑1 - Complex.I)) = 3 := by
sorry

end complex_modulus_problem_l1709_170981


namespace fuel_fraction_proof_l1709_170949

def road_trip_fuel_calculation (total_fuel : ℝ) (first_third : ℝ) (second_third_fraction : ℝ) : Prop :=
  let second_third := total_fuel * second_third_fraction
  let final_third := total_fuel - first_third - second_third
  final_third / second_third = 1 / 2

theorem fuel_fraction_proof :
  road_trip_fuel_calculation 60 30 (1/3) :=
by
  sorry

end fuel_fraction_proof_l1709_170949


namespace jake_weight_loss_l1709_170932

def jake_weight : ℝ := 93
def total_weight : ℝ := 132

theorem jake_weight_loss : ∃ (x : ℝ), 
  x ≥ 0 ∧ 
  jake_weight - x = 2 * (total_weight - jake_weight) ∧ 
  x = 15 := by
sorry

end jake_weight_loss_l1709_170932


namespace arithmetic_pattern_l1709_170947

theorem arithmetic_pattern (n : ℕ) : 
  (10^n - 1) * 9 + (n + 1) = 10^(n+1) - 1 :=
sorry

end arithmetic_pattern_l1709_170947


namespace max_value_polynomial_l1709_170965

theorem max_value_polynomial (a b : ℝ) (h : a + b = 4) :
  (∃ x y : ℝ, x + y = 4 ∧ 
    a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 ≤ 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4) ∧
  (∀ x y : ℝ, x + y = 4 → 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 7225/56) ∧
  (∃ x y : ℝ, x + y = 4 ∧ 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 = 7225/56) :=
by sorry

end max_value_polynomial_l1709_170965


namespace scientific_notation_16907_l1709_170914

theorem scientific_notation_16907 :
  16907 = 1.6907 * (10 : ℝ)^4 := by
  sorry

end scientific_notation_16907_l1709_170914


namespace xy_addition_identity_l1709_170990

theorem xy_addition_identity (x y : ℝ) : -x*y - x*y = -2*(x*y) := by
  sorry

end xy_addition_identity_l1709_170990


namespace tims_age_l1709_170908

/-- Given that Tom's age is 6 years more than 200% of Tim's age, 
    and Tom is 22 years old, Tim's age is 8 years. -/
theorem tims_age (tom_age tim_age : ℕ) 
  (h1 : tom_age = 2 * tim_age + 6)  -- Tom's age relation to Tim's
  (h2 : tom_age = 22)               -- Tom's actual age
  : tim_age = 8 := by
  sorry

#check tims_age

end tims_age_l1709_170908


namespace f_min_at_neg_seven_l1709_170930

/-- The quadratic function f(x) = x^2 + 14x + 24 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 24

/-- Theorem: The function f(x) = x^2 + 14x + 24 attains its minimum value when x = -7 -/
theorem f_min_at_neg_seven :
  ∀ x : ℝ, f x ≥ f (-7) :=
by sorry

end f_min_at_neg_seven_l1709_170930


namespace empty_solution_set_implies_a_geq_3_l1709_170978

theorem empty_solution_set_implies_a_geq_3 (a : ℝ) : 
  (∀ x : ℝ, ¬((x - 2) / 5 + 2 > x - 4 / 5 ∧ x > a)) → a ≥ 3 := by
  sorry

end empty_solution_set_implies_a_geq_3_l1709_170978


namespace current_task2_hours_proof_l1709_170957

/-- Calculates the current hours spent on task 2 per day given work conditions -/
def current_task2_hours (total_weekly_hours : ℕ) (work_days : ℕ) (task1_daily_hours : ℕ) (task1_reduction : ℕ) : ℕ :=
  let task1_weekly_hours := task1_daily_hours * work_days
  let new_task1_weekly_hours := task1_weekly_hours - task1_reduction
  let task2_weekly_hours := total_weekly_hours - new_task1_weekly_hours
  task2_weekly_hours / work_days

theorem current_task2_hours_proof :
  current_task2_hours 40 5 5 5 = 4 := by
  sorry

end current_task2_hours_proof_l1709_170957


namespace intersection_of_3n_and_2m_plus_1_l1709_170916

theorem intersection_of_3n_and_2m_plus_1 :
  {x : ℤ | ∃ n : ℤ, x = 3 * n} ∩ {x : ℤ | ∃ m : ℤ, x = 2 * m + 1} =
  {x : ℤ | ∃ k : ℤ, x = 12 * k + 1 ∨ x = 12 * k + 5} :=
by sorry

end intersection_of_3n_and_2m_plus_1_l1709_170916


namespace money_left_after_purchase_l1709_170920

def birthday_money (grandmother aunt uncle cousin brother : ℕ) : ℕ :=
  grandmother + aunt + uncle + cousin + brother

def total_in_wallet : ℕ := 185

def game_costs (game1 game2 game3 game4 game5 : ℕ) : ℕ :=
  game1 + game1 + game2 + game3 + game4 + game5

theorem money_left_after_purchase 
  (grandmother aunt uncle cousin brother : ℕ)
  (game1 game2 game3 game4 game5 : ℕ)
  (h1 : grandmother = 30)
  (h2 : aunt = 35)
  (h3 : uncle = 40)
  (h4 : cousin = 25)
  (h5 : brother = 20)
  (h6 : game1 = 30)
  (h7 : game2 = 40)
  (h8 : game3 = 35)
  (h9 : game4 = 25)
  (h10 : game5 = 0)  -- We use 0 for the fifth game as it's already counted in game1
  : total_in_wallet - (birthday_money grandmother aunt uncle cousin brother + game_costs game1 game2 game3 game4 game5) = 25 := by
  sorry

end money_left_after_purchase_l1709_170920


namespace f_properties_l1709_170934

noncomputable section

def f (x : ℝ) : ℝ := x^2 * Real.log x - x + 1

theorem f_properties :
  (∀ x > 0, f x = x^2 * Real.log x - x + 1) →
  f (Real.exp 1) = Real.exp 2 - Real.exp 1 + 1 ∧
  (deriv f) 1 = 0 ∧
  (∀ x ≥ 1, f x ≥ (x - 1)^2) ∧
  (∀ m > 3/2, ∃ x ≥ 1, f x < m * (x - 1)^2) ∧
  (∀ m ≤ 3/2, ∀ x ≥ 1, f x ≥ m * (x - 1)^2) :=
by sorry

end

end f_properties_l1709_170934


namespace min_value_sum_reciprocals_min_value_achievable_l1709_170967

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_one : x + y + z = 1) : 
  1/x + 4/y + 9/z ≥ 36 :=
by
  sorry

theorem min_value_achievable : 
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 1 ∧ 1/x + 4/y + 9/z = 36 :=
by
  sorry

end min_value_sum_reciprocals_min_value_achievable_l1709_170967


namespace inequality_solution_l1709_170929

theorem inequality_solution (x : ℝ) : 
  (x + 1 ≠ 0) → ((2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2) :=
sorry

end inequality_solution_l1709_170929


namespace point_C_coordinates_l1709_170907

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)

-- Define vector AB
def vecAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector AC in terms of AB
def vecAC : ℝ × ℝ := (2 * vecAB.1, 2 * vecAB.2)

-- Define point C
def C : ℝ × ℝ := (A.1 + vecAC.1, A.2 + vecAC.2)

-- Theorem to prove
theorem point_C_coordinates : C = (-3, 9) := by
  sorry

end point_C_coordinates_l1709_170907


namespace modulus_of_complex_fraction_l1709_170953

theorem modulus_of_complex_fraction : 
  Complex.abs ((3 - 4 * Complex.I) / Complex.I) = 5 := by
  sorry

end modulus_of_complex_fraction_l1709_170953


namespace circle_through_point_touching_lines_l1709_170962

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Function to check if a circle touches a line
def touchesLine (c : Circle) (l : Line) : Prop := sorry

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Function to check if two lines are parallel
def areParallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem circle_through_point_touching_lines 
  (p : Point) (l1 l2 : Line) : 
  ∃ (c1 c2 : Circle), 
    (touchesLine c1 l1 ∧ touchesLine c1 l2 ∧ pointOnCircle p c1) ∧
    (touchesLine c2 l1 ∧ touchesLine c2 l2 ∧ pointOnCircle p c2) :=
by sorry

end circle_through_point_touching_lines_l1709_170962


namespace min_distance_ABCD_l1709_170912

/-- Given four points A, B, C, and D on a line, with AB = 12, BC = 6, and CD = 5,
    the minimum possible distance between A and D is 1. -/
theorem min_distance_ABCD (A B C D : ℝ) : 
  abs (B - A) = 12 →
  abs (C - B) = 6 →
  abs (D - C) = 5 →
  ∃ (A' B' C' D' : ℝ), 
    abs (B' - A') = 12 ∧
    abs (C' - B') = 6 ∧
    abs (D' - C') = 5 ∧
    abs (D' - A') = 1 ∧
    ∀ (A'' B'' C'' D'' : ℝ),
      abs (B'' - A'') = 12 →
      abs (C'' - B'') = 6 →
      abs (D'' - C'') = 5 →
      abs (D'' - A'') ≥ 1 :=
sorry

end min_distance_ABCD_l1709_170912


namespace original_workers_count_l1709_170991

/-- Given a work that can be completed by an unknown number of workers in 45 days,
    and that adding 10 workers allows the work to be completed in 35 days,
    prove that the original number of workers is 35. -/
theorem original_workers_count (work : ℝ) (h1 : work > 0) : ∃ (workers : ℕ),
  (workers : ℝ) * 45 = work ∧
  (workers + 10 : ℝ) * 35 = work ∧
  workers = 35 := by
sorry

end original_workers_count_l1709_170991


namespace not_square_among_powers_l1709_170942

theorem not_square_among_powers : 
  (∃ n : ℕ, 1^6 = n^2) ∧
  (∃ n : ℕ, 3^4 = n^2) ∧
  (∃ n : ℕ, 4^3 = n^2) ∧
  (∃ n : ℕ, 5^2 = n^2) ∧
  (¬ ∃ n : ℕ, 2^5 = n^2) := by
  sorry

end not_square_among_powers_l1709_170942


namespace ratio_of_linear_system_l1709_170938

theorem ratio_of_linear_system (x y a b : ℝ) (h1 : 4 * x - 2 * y = a) 
  (h2 : 5 * y - 10 * x = b) (h3 : b ≠ 0) : a / b = -1 / 5 := by
  sorry

end ratio_of_linear_system_l1709_170938


namespace class_size_proof_l1709_170960

theorem class_size_proof (boys_avg : ℝ) (girls_avg : ℝ) (class_avg : ℝ) (boys_girls_diff : ℕ) :
  boys_avg = 73 →
  girls_avg = 77 →
  class_avg = 74 →
  boys_girls_diff = 22 →
  ∃ (total_students : ℕ), total_students = 44 := by
  sorry

end class_size_proof_l1709_170960


namespace range_of_m_l1709_170904

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0

def q (m : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m + 2)*x₁ - 1 < (m + 2)*x₂ - 1

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m ≤ -2 ∨ m ≥ 1) :=
sorry

end range_of_m_l1709_170904


namespace show_receipts_l1709_170933

/-- Calculates the total receipts for a show given ticket prices and attendance. -/
def totalReceipts (adultPrice childPrice : ℚ) (numAdults : ℕ) : ℚ :=
  let numChildren := numAdults / 2
  adultPrice * numAdults + childPrice * numChildren

/-- Theorem stating that the total receipts for the show are 1026 dollars. -/
theorem show_receipts :
  totalReceipts (5.5) (2.5) 152 = 1026 := by
  sorry

#eval totalReceipts (5.5) (2.5) 152

end show_receipts_l1709_170933


namespace quadratic_inequality_range_l1709_170977

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x + 100 > 0) ↔ (0 ≤ m ∧ m < 400) :=
sorry

end quadratic_inequality_range_l1709_170977


namespace calculation_proof_l1709_170986

theorem calculation_proof : 
  (5^(2/3) - 5^(3/2)) / 5^(1/2) = 60 := by
  sorry

end calculation_proof_l1709_170986


namespace central_cell_value_l1709_170989

theorem central_cell_value (a b c d e f g h i : ℝ) 
  (row_prod : a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10)
  (col_prod : a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10)
  (square_prod : a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3) :
  e = 0.00081 := by
sorry

end central_cell_value_l1709_170989


namespace block_weight_difference_l1709_170993

/-- Given two blocks with different weights, prove the difference between their weights. -/
theorem block_weight_difference (yellow_weight green_weight : ℝ)
  (h1 : yellow_weight = 0.6)
  (h2 : green_weight = 0.4) :
  yellow_weight - green_weight = 0.2 := by
  sorry

end block_weight_difference_l1709_170993


namespace base7_to_base10_conversion_l1709_170998

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [6, 5, 4, 3, 2]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 6068 := by
  sorry

end base7_to_base10_conversion_l1709_170998


namespace calculate_Y_l1709_170917

theorem calculate_Y : 
  let P : ℚ := 208 / 4
  let Q : ℚ := P / 2
  let Y : ℚ := P - Q * (10 / 100)
  Y = 49.4 := by sorry

end calculate_Y_l1709_170917


namespace officer_assignment_count_l1709_170918

-- Define the set of people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carol : Person
| Dave : Person

-- Define the set of officer positions
inductive Position : Type
| President : Position
| Secretary : Position
| Treasurer : Position

-- Define a function to check if a person is qualified for a position
def isQualified (p : Person) (pos : Position) : Prop :=
  match pos with
  | Position.President => p = Person.Dave
  | _ => True

-- Define an assignment of officers
def OfficerAssignment := Position → Person

-- Define a valid assignment
def validAssignment (assignment : OfficerAssignment) : Prop :=
  (∀ pos, isQualified (assignment pos) pos) ∧
  (∀ pos1 pos2, pos1 ≠ pos2 → assignment pos1 ≠ assignment pos2)

-- State the theorem
theorem officer_assignment_count :
  ∃ (assignments : Finset OfficerAssignment),
    (∀ a ∈ assignments, validAssignment a) ∧
    assignments.card = 6 :=
sorry

end officer_assignment_count_l1709_170918


namespace pokemon_card_ratio_l1709_170999

theorem pokemon_card_ratio : 
  ∀ (jenny orlando richard : ℕ),
    jenny = 6 →
    orlando = jenny + 2 →
    ∃ k : ℕ, richard = k * orlando →
    jenny + orlando + richard = 38 →
    richard / orlando = 3 := by
  sorry

end pokemon_card_ratio_l1709_170999


namespace x_squared_gt_y_squared_necessary_not_sufficient_l1709_170945

theorem x_squared_gt_y_squared_necessary_not_sufficient (x y : ℝ) :
  (∀ x y, x < y ∧ y < 0 → x^2 > y^2) ∧
  (∃ x y, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) := by
  sorry

end x_squared_gt_y_squared_necessary_not_sufficient_l1709_170945


namespace parallel_to_y_axis_coordinates_l1709_170980

/-- Given two points M(a-3, a+4) and N(√5, 9) in a Cartesian coordinate system,
    if the line MN is parallel to the y-axis, then M has coordinates (√5, 7 + √5) -/
theorem parallel_to_y_axis_coordinates (a : ℝ) :
  let M : ℝ × ℝ := (a - 3, a + 4)
  let N : ℝ × ℝ := (Real.sqrt 5, 9)
  (M.1 = N.1) →  -- MN is parallel to y-axis iff x-coordinates are equal
  M = (Real.sqrt 5, 7 + Real.sqrt 5) := by
sorry

end parallel_to_y_axis_coordinates_l1709_170980


namespace dog_burrs_problem_l1709_170950

theorem dog_burrs_problem (burrs ticks : ℕ) : 
  ticks = 6 * burrs → 
  burrs + ticks = 84 → 
  burrs = 12 := by sorry

end dog_burrs_problem_l1709_170950


namespace gcd_lcm_product_l1709_170979

theorem gcd_lcm_product (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

end gcd_lcm_product_l1709_170979


namespace distorted_polygon_sides_l1709_170996

/-- A regular polygon with a distorted exterior angle -/
structure DistortedPolygon where
  -- The apparent exterior angle in degrees
  apparent_angle : ℝ
  -- The distortion factor
  distortion_factor : ℝ
  -- The number of sides
  sides : ℕ

/-- The theorem stating the number of sides for the given conditions -/
theorem distorted_polygon_sides (p : DistortedPolygon) 
  (h1 : p.apparent_angle = 18)
  (h2 : p.distortion_factor = 1.5)
  (h3 : p.apparent_angle * p.sides = 360 * p.distortion_factor) : 
  p.sides = 30 := by
  sorry

end distorted_polygon_sides_l1709_170996


namespace percentage_problem_l1709_170940

theorem percentage_problem (P : ℝ) : P = 50 → 30 = (P / 100) * 40 + 10 := by
  sorry

end percentage_problem_l1709_170940


namespace cube_sum_eq_prime_product_solution_l1709_170944

theorem cube_sum_eq_prime_product_solution :
  ∀ (x y p : ℕ+), 
    x^3 + y^3 = p * (x * y + p) ∧ Nat.Prime p.val →
    ((x = 8 ∧ y = 1 ∧ p = 19) ∨
     (x = 1 ∧ y = 8 ∧ p = 19) ∨
     (x = 7 ∧ y = 2 ∧ p = 13) ∨
     (x = 2 ∧ y = 7 ∧ p = 13) ∨
     (x = 5 ∧ y = 4 ∧ p = 7) ∨
     (x = 4 ∧ y = 5 ∧ p = 7)) :=
by sorry

end cube_sum_eq_prime_product_solution_l1709_170944


namespace opposite_expressions_l1709_170952

theorem opposite_expressions (x : ℝ) : (4 * x - 8 = -(3 * x - 6)) ↔ x = 2 := by
  sorry

end opposite_expressions_l1709_170952


namespace smallest_a_for_equation_l1709_170974

theorem smallest_a_for_equation : 
  (∀ a : ℝ, a < -8 → ¬∃ b : ℝ, a^4 + 2*a^2*b + 2*a*b + b^2 = 960) ∧ 
  (∃ b : ℝ, (-8)^4 + 2*(-8)^2*b + 2*(-8)*b + b^2 = 960) := by
  sorry

end smallest_a_for_equation_l1709_170974


namespace john_total_needed_l1709_170971

/-- The amount of money John has, in dollars. -/
def john_has : ℚ := 0.75

/-- The additional amount John needs, in dollars. -/
def john_needs_more : ℚ := 1.75

/-- The total amount John needs is the sum of what he has and what he needs more. -/
theorem john_total_needed : john_has + john_needs_more = 2.50 := by
  sorry

end john_total_needed_l1709_170971


namespace stamp_collection_value_l1709_170968

theorem stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℚ) 
  (h1 : total_stamps = 18)
  (h2 : sample_stamps = 6)
  (h3 : sample_value = 15) : 
  (total_stamps : ℚ) * (sample_value / sample_stamps) = 45 := by
  sorry

end stamp_collection_value_l1709_170968


namespace sallys_peaches_l1709_170922

/-- Given that Sally had 13 peaches initially and ended up with 55 peaches,
    prove that she picked 42 peaches. -/
theorem sallys_peaches (initial : ℕ) (final : ℕ) (h1 : initial = 13) (h2 : final = 55) :
  final - initial = 42 := by sorry

end sallys_peaches_l1709_170922


namespace function_ordering_l1709_170924

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically decreasing on an interval
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- State the theorem
theorem function_ordering (h1 : is_even f) (h2 : is_monotone_decreasing_on (fun x ↦ f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 :=
sorry

end function_ordering_l1709_170924


namespace expression_simplification_l1709_170972

theorem expression_simplification (x : ℝ) : 
  2 * x - 3 * (2 - x) + 4 * (2 + x) - 5 * (1 - 3 * x) = 24 * x - 3 := by
  sorry

end expression_simplification_l1709_170972


namespace cone_base_circumference_l1709_170975

theorem cone_base_circumference (r : ℝ) (sector_angle : ℝ) : 
  r = 6 → sector_angle = 300 → 
  2 * π * r * (360 - sector_angle) / 360 = 2 * π := by
  sorry

end cone_base_circumference_l1709_170975


namespace line_separation_parameter_range_l1709_170936

/-- Given a line 2x - y + a = 0 where the origin (0, 0) and the point (1, 1) 
    are on opposite sides of this line, prove that -1 < a < 0 -/
theorem line_separation_parameter_range :
  ∀ a : ℝ, 
  (∀ x y : ℝ, 2*x - y + a = 0 → 
    ((0 : ℝ) < 2*0 - 0 + a) ≠ ((0 : ℝ) < 2*1 - 1 + a)) →
  -1 < a ∧ a < 0 :=
by sorry

end line_separation_parameter_range_l1709_170936


namespace triangle_properties_l1709_170970

open Real

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle conditions
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / sin A = b / sin B ∧
  b / sin B = c / sin C →
  -- Part 1
  (b = a * cos C + (1/2) * c → A = π/3) ∧
  -- Part 2
  (b * cos C + c * cos B = Real.sqrt 7 ∧ b = 2 → c = 3) :=
by sorry

end triangle_properties_l1709_170970


namespace quadratic_minimum_value_l1709_170976

/-- The minimum value of a quadratic function -/
theorem quadratic_minimum_value (a k c : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + (a + k) * x + c
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (m = (-a^2 - 2*a*k - k^2 + 4*a*c) / (4*a)) :=
sorry

end quadratic_minimum_value_l1709_170976


namespace inequality_proof_l1709_170973

theorem inequality_proof (a b : ℝ) (h : a < b) : 1 - a > 1 - b := by
  sorry

end inequality_proof_l1709_170973


namespace buffet_price_theorem_l1709_170964

/-- Represents the price of an adult buffet ticket -/
def adult_price : ℝ := 30

/-- Represents the price of a child buffet ticket -/
def child_price : ℝ := 15

/-- Represents the discount rate for senior citizens -/
def senior_discount : ℝ := 0.1

/-- Calculates the total cost for the family's buffet -/
def total_cost (adult_price : ℝ) : ℝ :=
  2 * adult_price +  -- Cost for 2 adults
  2 * (1 - senior_discount) * adult_price +  -- Cost for 2 senior citizens
  3 * child_price  -- Cost for 3 children

theorem buffet_price_theorem :
  total_cost adult_price = 159 :=
by sorry

end buffet_price_theorem_l1709_170964


namespace class_representatives_count_l1709_170913

/-- Represents the number of boys in the class -/
def num_boys : ℕ := 5

/-- Represents the number of girls in the class -/
def num_girls : ℕ := 3

/-- Represents the number of subjects needing representatives -/
def num_subjects : ℕ := 5

/-- Calculates the number of ways to select representatives with fewer girls than boys -/
def count_fewer_girls : ℕ := sorry

/-- Calculates the number of ways to select representatives with Boy A as a representative but not for mathematics -/
def count_boy_a_not_math : ℕ := sorry

/-- Calculates the number of ways to select representatives with Girl B for Chinese and Boy A as a representative but not for mathematics -/
def count_girl_b_chinese_boy_a_not_math : ℕ := sorry

/-- Theorem stating the correct number of ways for each condition -/
theorem class_representatives_count :
  count_fewer_girls = 5520 ∧
  count_boy_a_not_math = 3360 ∧
  count_girl_b_chinese_boy_a_not_math = 360 := by sorry

end class_representatives_count_l1709_170913


namespace people_left_at_table_l1709_170928

theorem people_left_at_table (initial_people : ℕ) (people_who_left : ℕ) : 
  initial_people = 11 → people_who_left = 6 → initial_people - people_who_left = 5 :=
by sorry

end people_left_at_table_l1709_170928


namespace regression_analysis_conclusions_l1709_170946

-- Define the regression model
structure RegressionModel where
  R_squared : ℝ
  sum_of_squares_residuals : ℝ
  residual_plot : Set (ℝ × ℝ)

-- Define the concept of model fit
def better_fit (model1 model2 : RegressionModel) : Prop := sorry

-- Define the concept of evenly scattered residuals
def evenly_scattered_residuals (plot : Set (ℝ × ℝ)) : Prop := sorry

-- Define the concept of horizontal band
def horizontal_band (plot : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem stating the correct conclusions
theorem regression_analysis_conclusions 
  (model1 model2 : RegressionModel) (ε : ℝ) (hε : ε > 0) :
  -- Higher R² indicates better fit
  (model1.R_squared > model2.R_squared + ε → better_fit model1 model2) ∧ 
  -- Smaller sum of squares of residuals indicates better fit
  (model1.sum_of_squares_residuals < model2.sum_of_squares_residuals - ε → 
    better_fit model1 model2) ∧
  -- Evenly scattered residuals around a horizontal band indicate appropriate model
  (evenly_scattered_residuals model1.residual_plot ∧ 
   horizontal_band model1.residual_plot → 
   better_fit model1 model2) := by sorry


end regression_analysis_conclusions_l1709_170946


namespace correct_electric_bicycle_volumes_l1709_170983

/-- Represents the parking data for a day --/
structure ParkingData where
  totalVolume : ℕ
  regularFeeBefore : ℚ
  electricFeeBefore : ℚ
  regularFeeAfter : ℚ
  electricFeeAfter : ℚ
  regularVolumeBefore : ℕ
  regularVolumeAfter : ℕ
  incomeFactor : ℚ

/-- Theorem stating the correct parking volumes for electric bicycles --/
theorem correct_electric_bicycle_volumes (data : ParkingData)
  (h1 : data.totalVolume = 6882)
  (h2 : data.regularFeeBefore = 1/5)
  (h3 : data.electricFeeBefore = 1/2)
  (h4 : data.regularFeeAfter = 2/5)
  (h5 : data.electricFeeAfter = 1)
  (h6 : data.regularVolumeBefore = 5180)
  (h7 : data.regularVolumeAfter = 335)
  (h8 : data.incomeFactor = 3/2) :
  ∃ (x y : ℕ),
    x + y = data.totalVolume - data.regularVolumeBefore - data.regularVolumeAfter ∧
    data.regularFeeBefore * data.regularVolumeBefore +
    data.regularFeeAfter * data.regularVolumeAfter +
    data.electricFeeBefore * x + data.electricFeeAfter * y =
    data.incomeFactor * (data.electricFeeBefore * x + data.electricFeeAfter * y) ∧
    x = 1174 ∧ y = 193 := by
  sorry


end correct_electric_bicycle_volumes_l1709_170983


namespace pascal_triangle_p_row_zeros_l1709_170954

theorem pascal_triangle_p_row_zeros (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) : 
  Nat.choose p k ≡ 0 [MOD p] := by
  sorry

end pascal_triangle_p_row_zeros_l1709_170954


namespace product_equals_square_l1709_170963

theorem product_equals_square : 
  200 * 39.96 * 3.996 * 500 = (3996 : ℝ)^2 := by sorry

end product_equals_square_l1709_170963


namespace stella_restocks_six_bathrooms_l1709_170921

/-- The number of bathrooms Stella restocks -/
def num_bathrooms : ℕ :=
  let rolls_per_day : ℕ := 1
  let days_per_week : ℕ := 7
  let num_weeks : ℕ := 4
  let rolls_per_pack : ℕ := 12
  let packs_bought : ℕ := 14
  let rolls_per_bathroom : ℕ := rolls_per_day * days_per_week * num_weeks
  let total_rolls_bought : ℕ := packs_bought * rolls_per_pack
  total_rolls_bought / rolls_per_bathroom

theorem stella_restocks_six_bathrooms : num_bathrooms = 6 := by
  sorry

end stella_restocks_six_bathrooms_l1709_170921


namespace largest_circle_radius_l1709_170931

/-- Represents a standard chessboard --/
structure Chessboard :=
  (size : ℕ)
  (is_standard : size = 8)

/-- Represents a circle on the chessboard --/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Checks if a circle intersects any white square on the chessboard --/
def intersects_white_square (c : Circle) (b : Chessboard) : Prop :=
  sorry

/-- The largest circle that doesn't intersect any white square --/
def largest_circle (b : Chessboard) : Circle :=
  sorry

theorem largest_circle_radius (b : Chessboard) :
  (largest_circle b).radius = (Real.sqrt 10) / 2 :=
sorry

end largest_circle_radius_l1709_170931


namespace poorly_chosen_character_Lobster_poorly_chosen_l1709_170961

/-- Represents a character in "Alice's Adventures in Wonderland" --/
structure Character where
  name : String
  is_active : Bool
  appears_in_poem : Bool

/-- Defines what it means for a character to be poorly chosen --/
def is_poorly_chosen (c : Character) : Prop :=
  c.appears_in_poem ∧ ¬c.is_active

/-- Theorem stating that a character is poorly chosen if it only appears in a poem and is not active --/
theorem poorly_chosen_character (c : Character) :
  c.appears_in_poem ∧ ¬c.is_active → is_poorly_chosen c := by
  sorry

/-- The Lobster character --/
def Lobster : Character :=
  { name := "Lobster",
    is_active := false,
    appears_in_poem := true }

/-- Theorem specifically about the Lobster being poorly chosen --/
theorem Lobster_poorly_chosen : is_poorly_chosen Lobster := by
  sorry

end poorly_chosen_character_Lobster_poorly_chosen_l1709_170961


namespace system_solution_range_l1709_170941

theorem system_solution_range (x y a : ℝ) : 
  x + 3*y = 2 + a → 
  3*x + y = -4*a → 
  x + y > 2 → 
  a < -2 := by
sorry

end system_solution_range_l1709_170941


namespace arctan_sum_equation_l1709_170943

theorem arctan_sum_equation (n : ℕ+) : 
  (Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/(n : ℝ)) = π/4) ↔ n = 57 := by
  sorry

end arctan_sum_equation_l1709_170943


namespace abs_inequality_necessary_not_sufficient_l1709_170992

theorem abs_inequality_necessary_not_sufficient (x : ℝ) :
  (x * (x - 2) < 0 → abs (x - 1) < 2) ∧
  ¬(abs (x - 1) < 2 → x * (x - 2) < 0) := by
  sorry

end abs_inequality_necessary_not_sufficient_l1709_170992


namespace tan_expression_equals_neg_sqrt_three_l1709_170966

/-- A sequence is a geometric progression -/
def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is an arithmetic progression -/
def is_arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Main theorem -/
theorem tan_expression_equals_neg_sqrt_three
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_geom : is_geometric_progression a)
  (h_arith : is_arithmetic_progression b)
  (h_prod : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
  (h_sum : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 :=
sorry

end tan_expression_equals_neg_sqrt_three_l1709_170966


namespace iesha_book_count_l1709_170995

/-- Represents the number of books Iesha has -/
structure IeshasBooks where
  school : ℕ
  sports : ℕ

/-- The total number of books Iesha has -/
def total_books (b : IeshasBooks) : ℕ := b.school + b.sports

theorem iesha_book_count : 
  ∀ (b : IeshasBooks), b.school = 19 → b.sports = 39 → total_books b = 58 := by
  sorry

end iesha_book_count_l1709_170995


namespace gcd_1037_425_l1709_170956

theorem gcd_1037_425 : Nat.gcd 1037 425 = 17 := by
  sorry

end gcd_1037_425_l1709_170956


namespace product_inspection_probability_l1709_170935

theorem product_inspection_probability : 
  let p_good_as_defective : ℝ := 0.02
  let p_defective_as_good : ℝ := 0.01
  let num_good : ℕ := 3
  let num_defective : ℕ := 1
  let p_correct_good : ℝ := 1 - p_good_as_defective
  let p_correct_defective : ℝ := 1 - p_defective_as_good
  (p_correct_good ^ num_good) * (p_correct_defective ^ num_defective) = 0.932 :=
by sorry

end product_inspection_probability_l1709_170935


namespace original_deck_size_l1709_170937

/-- Represents a deck of cards with red and black cards -/
structure Deck where
  red : ℕ
  black : ℕ

/-- The probability of selecting a red card from the deck -/
def redProbability (d : Deck) : ℚ :=
  d.red / (d.red + d.black)

theorem original_deck_size :
  ∃ d : Deck,
    redProbability d = 2/5 ∧
    redProbability {red := d.red + 3, black := d.black} = 1/2 ∧
    d.red + d.black = 15 := by
  sorry

end original_deck_size_l1709_170937


namespace triangle_properties_l1709_170982

/-- Given a triangle ABC with angle A = π/3 and perimeter 6, 
    prove the relation between sides and find the maximum area -/
theorem triangle_properties (b c : ℝ) (h_perimeter : b + c ≤ 6) : 
  b * c + 12 = 4 * (b + c) ∧ 
  (∀ (b' c' : ℝ), b' + c' ≤ 6 → 
    (1/2 : ℝ) * b' * c' * Real.sqrt 3 ≤ Real.sqrt 3) := by
  sorry

#check triangle_properties

end triangle_properties_l1709_170982


namespace sesame_mass_scientific_notation_l1709_170985

theorem sesame_mass_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.00000201 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.01 ∧ n = -6 :=
sorry

end sesame_mass_scientific_notation_l1709_170985


namespace probability_failed_chinese_given_failed_math_l1709_170955

theorem probability_failed_chinese_given_failed_math 
  (total_students : ℕ) 
  (failed_math : ℕ) 
  (failed_chinese : ℕ) 
  (failed_both : ℕ) 
  (h1 : failed_math = (25 : ℕ) * total_students / 100)
  (h2 : failed_chinese = (10 : ℕ) * total_students / 100)
  (h3 : failed_both = (5 : ℕ) * total_students / 100)
  (h4 : total_students > 0) :
  (failed_both : ℚ) / failed_math = 1 / 5 := by
  sorry

end probability_failed_chinese_given_failed_math_l1709_170955


namespace vector_subtraction_scalar_multiplication_l1709_170958

theorem vector_subtraction_scalar_multiplication :
  (3 : ℝ) • (((⟨-3, 2, -5⟩ : ℝ × ℝ × ℝ) - ⟨1, 6, 2⟩) : ℝ × ℝ × ℝ) = ⟨-12, -12, -21⟩ := by
  sorry

end vector_subtraction_scalar_multiplication_l1709_170958


namespace distance_at_time_l1709_170994

/-- Represents a right-angled triangle with given hypotenuse and leg lengths -/
structure RightTriangle where
  hypotenuse : ℝ
  leg : ℝ

/-- Represents a moving point with a given speed -/
structure MovingPoint where
  speed : ℝ

theorem distance_at_time (triangle : RightTriangle) (point1 point2 : MovingPoint) :
  triangle.hypotenuse = 85 →
  triangle.leg = 75 →
  point1.speed = 8.5 →
  point2.speed = 5 →
  ∃ t : ℝ, t = 4 ∧ 
    let d1 := triangle.hypotenuse - point1.speed * t
    let d2 := triangle.leg - point2.speed * t
    d1 * d1 + d2 * d2 = 26 * 26 :=
by sorry

end distance_at_time_l1709_170994


namespace even_function_property_l1709_170911

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_prop : ∀ x, f (2 + x) = -f (2 - x)) :
  f 2010 = 0 := by sorry

end even_function_property_l1709_170911


namespace tank_full_time_l1709_170902

/-- Represents the time it takes to fill a tank with given parameters -/
def fill_time (tank_capacity : ℕ) (pipe_a_rate : ℕ) (pipe_b_rate : ℕ) (pipe_c_rate : ℕ) : ℕ :=
  let cycle_net_fill := pipe_a_rate + pipe_b_rate - pipe_c_rate
  let cycles := tank_capacity / cycle_net_fill
  let total_minutes := cycles * 3
  total_minutes - 1

/-- Theorem stating that the tank will be full after 50 minutes -/
theorem tank_full_time :
  fill_time 850 40 30 20 = 50 := by
  sorry

end tank_full_time_l1709_170902


namespace factorial_square_root_product_l1709_170905

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_square_root_product : (Real.sqrt (factorial 5 * factorial 4))^2 = 2880 := by
  sorry

end factorial_square_root_product_l1709_170905


namespace difference_of_squares_l1709_170901

theorem difference_of_squares (a b : ℝ) : (a - b) * (-a - b) = b^2 - a^2 := by
  sorry

end difference_of_squares_l1709_170901


namespace expression_evaluation_l1709_170910

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by sorry

end expression_evaluation_l1709_170910


namespace triangle_inequality_with_constant_l1709_170959

theorem triangle_inequality_with_constant (k : ℕ) : 
  (k > 0) → 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  k = 6 :=
sorry

end triangle_inequality_with_constant_l1709_170959


namespace smallest_integer_in_special_set_l1709_170906

theorem smallest_integer_in_special_set : ∀ n : ℤ,
  (n + 6 > 2 * ((7 * n + 21) / 7)) →
  (∀ m : ℤ, m < n → m + 6 ≤ 2 * ((7 * m + 21) / 7)) →
  n = -1 :=
by sorry

end smallest_integer_in_special_set_l1709_170906


namespace real_axis_length_l1709_170927

/-- A hyperbola with equation x²/a² - y²/b² = 1/4 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The standard hyperbola with equation x²/9 - y²/16 = 1 -/
def standard_hyperbola : Hyperbola where
  a := 3
  b := 4
  h_positive := by norm_num

theorem real_axis_length
  (C : Hyperbola)
  (h_asymptotes : C.a / C.b = standard_hyperbola.a / standard_hyperbola.b)
  (h_point : C.a^2 * 9 - C.b^2 * 12 = C.a^2 * C.b^2) :
  2 * C.a = 3 := by
  sorry

#check real_axis_length

end real_axis_length_l1709_170927


namespace solution_set_equality_max_value_g_l1709_170984

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := f x ≥ 1

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 1}

-- Define the function g
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem 1: The solution set of f(x) ≥ 1 is {x | x ≥ 1}
theorem solution_set_equality : 
  {x : ℝ | inequality_condition x} = solution_set := by sorry

-- Theorem 2: The maximum value of g(x) is 5/4
theorem max_value_g : 
  ∃ (x : ℝ), g x = 5/4 ∧ ∀ (y : ℝ), g y ≤ 5/4 := by sorry

end solution_set_equality_max_value_g_l1709_170984


namespace fraction_equality_l1709_170926

theorem fraction_equality (a b : ℝ) (h : (1/a + 1/b)/(1/a - 1/b) = 2023) : (a + b)/(a - b) = 2023 := by
  sorry

end fraction_equality_l1709_170926


namespace susan_chairs_l1709_170997

def chairs_problem (red_chairs : ℕ) (yellow_multiplier : ℕ) (blue_difference : ℕ) : Prop :=
  let yellow_chairs := red_chairs * yellow_multiplier
  let blue_chairs := yellow_chairs - blue_difference
  red_chairs + yellow_chairs + blue_chairs = 43

theorem susan_chairs : chairs_problem 5 4 2 := by
  sorry

end susan_chairs_l1709_170997
