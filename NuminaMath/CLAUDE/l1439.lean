import Mathlib

namespace oil_bill_ratio_l1439_143923

/-- The oil bill problem -/
theorem oil_bill_ratio (january_bill : ℝ) (february_bill : ℝ) : 
  january_bill = 119.99999999999994 →
  february_bill / january_bill = 3 / 2 →
  (february_bill + 20) / january_bill = 5 / 3 := by
sorry

end oil_bill_ratio_l1439_143923


namespace pauls_books_l1439_143984

theorem pauls_books (sold : ℕ) (left : ℕ) (h1 : sold = 137) (h2 : left = 105) :
  sold + left = 242 := by
  sorry

end pauls_books_l1439_143984


namespace chocolate_bars_bought_l1439_143972

theorem chocolate_bars_bought (bar_cost : ℝ) (paid : ℝ) (max_change : ℝ) :
  bar_cost = 1.35 →
  paid = 10 →
  max_change = 1 →
  ∃ n : ℕ, n * bar_cost ≤ paid ∧
           paid - n * bar_cost < max_change ∧
           ∀ m : ℕ, m > n → m * bar_cost > paid :=
by
  sorry

#check chocolate_bars_bought

end chocolate_bars_bought_l1439_143972


namespace sum_of_roots_cubic_equation_l1439_143903

theorem sum_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 4 * x^3 + 5 * x^2 - 8 * x
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ (r₁ + r₂ + r₃ = -1.25) :=
by sorry

end sum_of_roots_cubic_equation_l1439_143903


namespace triangle_properties_l1439_143902

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  (a * Real.sin B = b * Real.sin A) →
  (b * Real.sin C = c * Real.sin B) →
  (c * Real.sin A = a * Real.sin C) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  (Real.cos A = Real.sqrt 3 / 2) ∧
  (b = 1 → (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2) :=
by sorry

end triangle_properties_l1439_143902


namespace popcorn_selling_price_l1439_143971

/-- Calculate the selling price per bag of popcorn -/
theorem popcorn_selling_price 
  (cost_price : ℝ) 
  (num_bags : ℕ) 
  (total_profit : ℝ) 
  (h1 : cost_price = 4)
  (h2 : num_bags = 30)
  (h3 : total_profit = 120) : 
  (cost_price * num_bags + total_profit) / num_bags = 8 := by
  sorry

end popcorn_selling_price_l1439_143971


namespace transformation_result_l1439_143948

def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def rotate_x_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, -z)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_y_180
    |> reflect_yz
    |> reflect_xz
    |> rotate_x_180
    |> reflect_yz

theorem transformation_result :
  transform initial_point = (-2, 2, 2) := by
  sorry

end transformation_result_l1439_143948


namespace license_plate_increase_l1439_143904

theorem license_plate_increase : 
  let old_plates := 26 * 10^4
  let new_plates := 26^3 * 10^3
  new_plates / old_plates = 26^2 / 10 := by
  sorry

end license_plate_increase_l1439_143904


namespace smallest_x_value_l1439_143985

theorem smallest_x_value : ∃ x : ℝ, 
  (∀ y : ℝ, 3 * y^2 + 36 * y - 135 = 2 * y * (y + 16) → x ≤ y) ∧
  (3 * x^2 + 36 * x - 135 = 2 * x * (x + 16)) ∧
  x = -15 := by
  sorry

end smallest_x_value_l1439_143985


namespace unique_n_satisfying_conditions_l1439_143982

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there exists exactly one positive integer n > 1 
    satisfying the given conditions -/
theorem unique_n_satisfying_conditions : 
  ∃! n : ℕ, n > 1 ∧ 
    greatest_prime_factor n = Real.sqrt n ∧
    greatest_prime_factor (n + 72) = Real.sqrt (n + 72) :=
  sorry

end unique_n_satisfying_conditions_l1439_143982


namespace max_profit_is_12250_l1439_143943

/-- Represents the profit function for selling humidifiers -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 300 * x + 10000

/-- Represents the selling price of a humidifier -/
def selling_price (x : ℝ) : ℝ := 100 + x

/-- Represents the daily sales volume -/
def daily_sales (x : ℝ) : ℝ := 500 - 10 * x

/-- Theorem stating that the maximum profit is 12250 yuan -/
theorem max_profit_is_12250 :
  ∃ x : ℝ, 
    (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ 
    profit_function x = 12250 ∧
    selling_price x = 115 :=
sorry

end max_profit_is_12250_l1439_143943


namespace three_numbers_sum_l1439_143918

theorem three_numbers_sum (s : ℕ) :
  let A := Finset.range (4 * s) 
  ∀ (S : Finset ℕ), S ⊆ A → S.card = 2 * s + 2 →
    ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x + y = 2 * z :=
by sorry

end three_numbers_sum_l1439_143918


namespace min_f_correct_a_range_condition_l1439_143914

noncomputable section

def f (a x : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

def g (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

def min_f (a : ℝ) : ℝ :=
  if a ≤ 1 then 1 - a
  else if a < Real.exp 1 then a - (a + 1) * Real.log a - 1
  else Real.exp 1 - (a + 1) - a / Real.exp 1

theorem min_f_correct (a : ℝ) :
  ∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ min_f a := by sorry

theorem a_range_condition (a : ℝ) :
  a < 1 →
  (∃ x₁ ∈ Set.Icc (Real.exp 1) (Real.exp 2),
    ∀ x₂ ∈ Set.Icc (-2) 0, f a x₁ < g x₂) →
  (Real.exp 2 - 2 * Real.exp 1) / (Real.exp 1 + 1) < a := by sorry

end

end min_f_correct_a_range_condition_l1439_143914


namespace gold_bars_worth_l1439_143952

/-- Calculate the total worth of gold bars in a safe -/
theorem gold_bars_worth (rows : ℕ) (bars_per_row : ℕ) (worth_per_bar : ℕ) :
  rows = 4 →
  bars_per_row = 20 →
  worth_per_bar = 20000 →
  rows * bars_per_row * worth_per_bar = 1600000 := by
  sorry

#check gold_bars_worth

end gold_bars_worth_l1439_143952


namespace problem_statement_l1439_143905

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49/(x - 3)^2 = 23 := by
sorry

end problem_statement_l1439_143905


namespace franks_age_to_tys_age_ratio_l1439_143993

/-- Proves that the ratio of Frank's age in 5 years to Ty's current age is 3:1 -/
theorem franks_age_to_tys_age_ratio : 
  let karen_age : ℕ := 2
  let carla_age : ℕ := karen_age + 2
  let ty_age : ℕ := 2 * carla_age + 4
  let frank_future_age : ℕ := 36
  (frank_future_age : ℚ) / ty_age = 3 / 1 := by sorry

end franks_age_to_tys_age_ratio_l1439_143993


namespace city_population_ratio_l1439_143969

theorem city_population_ratio (pop_x pop_y pop_z : ℝ) 
  (h1 : pop_x = 5 * pop_y) 
  (h2 : pop_x / pop_z = 10) : 
  pop_y / pop_z = 2 := by
sorry

end city_population_ratio_l1439_143969


namespace triangle_side_ratio_l1439_143991

theorem triangle_side_ratio (A B C a b c : ℝ) : 
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are side lengths opposite to A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Sine rule
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  Real.sin A + Real.cos A - 2 / (Real.sin B + Real.cos B) = 0 →
  -- Conclusion
  (a + b) / c = Real.sqrt 2 := by
sorry

end triangle_side_ratio_l1439_143991


namespace vegetables_minus_fruits_l1439_143968

def cucumbers : ℕ := 6
def tomatoes : ℕ := 8
def apples : ℕ := 2
def bananas : ℕ := 4

def vegetables : ℕ := cucumbers + tomatoes
def fruits : ℕ := apples + bananas

theorem vegetables_minus_fruits : vegetables - fruits = 8 := by
  sorry

end vegetables_minus_fruits_l1439_143968


namespace complex_equation_solution_l1439_143999

theorem complex_equation_solution (z : ℂ) :
  Complex.I * z = 4 + 3 * Complex.I → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l1439_143999


namespace secretary_work_ratio_l1439_143987

/-- Represents the work hours of three secretaries on a project. -/
structure SecretaryWork where
  total : ℝ
  longest : ℝ
  second : ℝ
  third : ℝ

/-- Theorem stating the ratio of work hours for three secretaries. -/
theorem secretary_work_ratio (work : SecretaryWork) 
  (h_total : work.total = 120)
  (h_longest : work.longest = 75)
  (h_sum : work.second + work.third = work.total - work.longest) :
  ∃ (b c : ℝ), work.second = b ∧ work.third = c ∧ b + c = 45 := by
  sorry

#check secretary_work_ratio

end secretary_work_ratio_l1439_143987


namespace acute_triangle_properties_l1439_143901

theorem acute_triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (∀ x : ℝ, x^2 - 2 * Real.sqrt 3 * x + 2 = 0 → (x = a ∨ x = b)) →
  2 * Real.sin (A + B) - Real.sqrt 3 = 0 →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  c = Real.sqrt 6 ∧
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 :=
by sorry

end acute_triangle_properties_l1439_143901


namespace max_distance_sum_l1439_143964

/-- Given m ∈ ℝ, and lines l₁ and l₂ passing through points A and B respectively,
    and intersecting at point P ≠ A, B, the maximum value of |PA| + |PB| is 2√5. -/
theorem max_distance_sum (m : ℝ) : 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (2, 3)
  let l₁ := {(x, y) : ℝ × ℝ | x + m * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | m * x - y - 2 * m + 3 = 0}
  ∀ P : ℝ × ℝ, P ∈ l₁ ∩ l₂ → P ≠ A → P ≠ B →
    ‖P - A‖ + ‖P - B‖ ≤ 2 * Real.sqrt 5 :=
by sorry


end max_distance_sum_l1439_143964


namespace disjunction_false_implies_both_false_l1439_143990

theorem disjunction_false_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → ¬p ∧ ¬q := by
  sorry

end disjunction_false_implies_both_false_l1439_143990


namespace journal_involvement_l1439_143931

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

end journal_involvement_l1439_143931


namespace roommate_payment_is_757_l1439_143956

/-- Calculates the total payment for one roommate given the costs for rent, utilities, and groceries -/
def roommateTotalPayment (rent utilities groceries : ℕ) : ℚ :=
  (rent + utilities + groceries : ℚ) / 2

/-- Proves that one roommate's total payment is $757 given the specified costs -/
theorem roommate_payment_is_757 :
  roommateTotalPayment 1100 114 300 = 757 := by
  sorry

end roommate_payment_is_757_l1439_143956


namespace smallest_largest_a_sum_l1439_143966

theorem smallest_largest_a_sum (a b c : ℝ) (sum_eq : a + b + c = 5) (sum_sq_eq : a^2 + b^2 + c^2 = 8) :
  (∃ (a_min a_max : ℝ), 
    (∀ x : ℝ, (∃ y z : ℝ, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8) → a_min ≤ x ∧ x ≤ a_max) ∧
    a_min = 1 ∧ 
    a_max = 3 ∧ 
    a_min + a_max = 4) :=
by sorry

end smallest_largest_a_sum_l1439_143966


namespace salary_change_percentage_l1439_143947

theorem salary_change_percentage (S : ℝ) (x : ℝ) (h : S > 0) :
  S * (1 - x / 100) * (1 + x / 100) = 0.75 * S → x = 50 := by
  sorry

end salary_change_percentage_l1439_143947


namespace base4_calculation_l1439_143930

/-- Represents a number in base 4 --/
def Base4 : Type := ℕ

/-- Multiplication operation for base 4 numbers --/
def mul_base4 : Base4 → Base4 → Base4 := sorry

/-- Division operation for base 4 numbers --/
def div_base4 : Base4 → Base4 → Base4 := sorry

/-- Conversion from decimal to base 4 --/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Conversion from base 4 to decimal --/
def from_base4 (n : Base4) : ℕ := sorry

theorem base4_calculation :
  let a := to_base4 203
  let b := to_base4 21
  let c := to_base4 3
  let result := to_base4 110320
  mul_base4 (div_base4 a c) b = result := by sorry

end base4_calculation_l1439_143930


namespace greatest_integer_less_than_negative_seventeen_fourths_l1439_143924

theorem greatest_integer_less_than_negative_seventeen_fourths :
  ⌊-17/4⌋ = -5 := by
  sorry

end greatest_integer_less_than_negative_seventeen_fourths_l1439_143924


namespace final_distance_is_35_l1439_143958

/-- Represents the movements of the first car -/
structure FirstCarMovement where
  initial_run : ℝ
  right_turn : ℝ
  left_turn : ℝ

/-- Represents the movement of the second car -/
def SecondCarMovement : ℝ := 35

/-- Calculates the final distance between two cars given their movements -/
def finalDistance (initial_distance : ℝ) (first_car : FirstCarMovement) (second_car : ℝ) : ℝ :=
  initial_distance - (first_car.initial_run + 2 * first_car.right_turn + first_car.left_turn) - second_car

/-- Theorem stating that the final distance between the cars is 35 km -/
theorem final_distance_is_35 :
  let first_car : FirstCarMovement := ⟨25, 15, 25⟩
  let second_car : ℝ := SecondCarMovement
  finalDistance 150 first_car second_car = 35 := by
  sorry


end final_distance_is_35_l1439_143958


namespace village_households_l1439_143932

/-- The number of households in a village given water consumption data. -/
theorem village_households (water_per_household : ℕ) (total_water : ℕ) 
  (h1 : water_per_household = 200)
  (h2 : total_water = 2000)
  (h3 : total_water = water_per_household * (total_water / water_per_household)) :
  total_water / water_per_household = 10 := by
  sorry

#check village_households

end village_households_l1439_143932


namespace sqrt_three_plus_sqrt_seven_less_than_two_sqrt_five_l1439_143911

theorem sqrt_three_plus_sqrt_seven_less_than_two_sqrt_five : 
  Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end sqrt_three_plus_sqrt_seven_less_than_two_sqrt_five_l1439_143911


namespace triangle_inequality_triangle_equality_l1439_143920

/-- The area of a triangle with sides a, b, c -/
noncomputable def A (a b c : ℝ) : ℝ := sorry

/-- Function f as defined in the problem -/
noncomputable def f (a b c : ℝ) : ℝ := Real.sqrt (A a b c)

/-- The main theorem -/
theorem triangle_inequality (a b c a' b' c' : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (ha' : 0 < a') (hb' : 0 < b') (hc' : 0 < c') :
    f a b c + f a' b' c' ≤ f (a + a') (b + b') (c + c') :=
  sorry

/-- Condition for equality -/
theorem triangle_equality (a b c a' b' c' : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (ha' : 0 < a') (hb' : 0 < b') (hc' : 0 < c') :
    f a b c + f a' b' c' = f (a + a') (b + b') (c + c') ↔ a / a' = b / b' ∧ b / b' = c / c' :=
  sorry

end triangle_inequality_triangle_equality_l1439_143920


namespace x_intercept_ratio_l1439_143996

/-- Two lines with the same non-zero y-intercept -/
structure TwoLines where
  y_intercept : ℝ
  slope1 : ℝ
  slope2 : ℝ
  x_intercept1 : ℝ
  x_intercept2 : ℝ
  y_intercept_nonzero : y_intercept ≠ 0
  slope1_is_8 : slope1 = 8
  slope2_is_4 : slope2 = 4

/-- The ratio of x-intercepts is 1/2 -/
theorem x_intercept_ratio (l : TwoLines) : l.x_intercept1 / l.x_intercept2 = 1 / 2 := by
  sorry

end x_intercept_ratio_l1439_143996


namespace fraction_product_equality_l1439_143917

theorem fraction_product_equality : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 = 24 / 77 := by
  sorry

end fraction_product_equality_l1439_143917


namespace equation_solution_l1439_143998

theorem equation_solution : 
  let x : ℝ := 14.8 / 0.13
  0.05 * x + 0.04 * (30 + 2 * x) = 16 := by
sorry

end equation_solution_l1439_143998


namespace square_diff_product_l1439_143938

theorem square_diff_product (x y : ℝ) (hx : x = Real.sqrt 3 + 1) (hy : y = Real.sqrt 3 - 1) :
  x^2 * y - x * y^2 = 4 := by
  sorry

end square_diff_product_l1439_143938


namespace least_n_with_gcd_conditions_l1439_143978

theorem least_n_with_gcd_conditions : ∃ (n : ℕ), 
  (n > 500) ∧ 
  (Nat.gcd 70 (n + 150) = 35) ∧ 
  (Nat.gcd (n + 70) 150 = 50) ∧ 
  (∀ m : ℕ, m > 500 → Nat.gcd 70 (m + 150) = 35 → Nat.gcd (m + 70) 150 = 50 → m ≥ n) ∧
  n = 1015 := by
sorry

end least_n_with_gcd_conditions_l1439_143978


namespace sphere_surface_area_of_prism_l1439_143928

/-- The surface area of a sphere circumscribing a right square prism -/
theorem sphere_surface_area_of_prism (base_edge : ℝ) (height : ℝ) 
  (h_base : base_edge = 2) (h_height : height = 3) :
  4 * π * ((base_edge^2 + base_edge^2 + height^2).sqrt / 2)^2 = 17 * π :=
by sorry

end sphere_surface_area_of_prism_l1439_143928


namespace quadratic_expression_value_l1439_143953

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 2 * x + y = 11) 
  (h2 : x + 2 * y = 13) : 
  10 * x^2 - 6 * x * y + y^2 = 530 := by
sorry

end quadratic_expression_value_l1439_143953


namespace shared_angle_measure_l1439_143974

/-- A configuration of a regular pentagon sharing a side with an equilateral triangle -/
structure PentagonTriangleConfig where
  /-- The measure of an interior angle of the regular pentagon in degrees -/
  pentagon_angle : ℝ
  /-- The measure of an interior angle of the equilateral triangle in degrees -/
  triangle_angle : ℝ
  /-- The condition that the pentagon is regular -/
  pentagon_regular : pentagon_angle = 108
  /-- The condition that the triangle is equilateral -/
  triangle_equilateral : triangle_angle = 60

/-- The theorem stating that the angle formed by the shared side and the adjacent sides is 6 degrees -/
theorem shared_angle_measure (config : PentagonTriangleConfig) :
  let total_angle := config.pentagon_angle + config.triangle_angle
  let shared_angle := (180 - total_angle) / 2
  shared_angle = 6 := by sorry

end shared_angle_measure_l1439_143974


namespace smallest_fourth_power_b_l1439_143933

theorem smallest_fourth_power_b : ∃ (n : ℕ), 
  (7 + 7 * 18 + 7 * 18^2 = n^4) ∧ 
  (∀ (b : ℕ), b > 0 → b < 18 → ¬∃ (m : ℕ), 7 + 7 * b + 7 * b^2 = m^4) := by
  sorry

end smallest_fourth_power_b_l1439_143933


namespace signboard_white_area_l1439_143912

/-- Represents the dimensions of a letter stroke -/
structure StrokeDimensions where
  width : ℝ
  height : ℝ

/-- Represents a letter on the signboard -/
inductive Letter
| L
| A
| S
| T

/-- Calculates the area of a letter based on its strokes -/
def letterArea (letter : Letter) : ℝ :=
  match letter with
  | Letter.L => 9
  | Letter.A => 7.5
  | Letter.S => 13
  | Letter.T => 9

/-- Represents the signboard -/
structure Signboard where
  width : ℝ
  height : ℝ
  word : List Letter
  strokeWidth : ℝ

def signboard : Signboard :=
  { width := 6
  , height := 18
  , word := [Letter.L, Letter.A, Letter.S, Letter.T]
  , strokeWidth := 1 }

/-- Calculates the total area of the signboard -/
def totalArea (s : Signboard) : ℝ :=
  s.width * s.height

/-- Calculates the area covered by the letters -/
def coveredArea (s : Signboard) : ℝ :=
  s.word.map letterArea |> List.sum

/-- Calculates the white area remaining on the signboard -/
def whiteArea (s : Signboard) : ℝ :=
  totalArea s - coveredArea s

/-- Theorem stating that the white area of the given signboard is 69.5 square units -/
theorem signboard_white_area :
  whiteArea signboard = 69.5 := by
  sorry

end signboard_white_area_l1439_143912


namespace cuboid_length_problem_l1439_143935

/-- The surface area of a cuboid given its length, width, and height -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The length of a cuboid with surface area 700 m², breadth 14 m, and height 7 m is 12 m -/
theorem cuboid_length_problem :
  ∃ (l : ℝ), cuboidSurfaceArea l 14 7 = 700 ∧ l = 12 := by
  sorry

end cuboid_length_problem_l1439_143935


namespace max_c_value_max_c_attainable_l1439_143995

theorem max_c_value (c a b : ℕ+) (h1 : c ≤ 2017) 
  (h2 : 2^(a:ℕ) * 5^(b:ℕ) = (a^3 + a^2 + a + 1) * c) : c ≤ 1000 := by
  sorry

theorem max_c_attainable : ∃ (c a b : ℕ+), c = 1000 ∧ c ≤ 2017 ∧ 
  2^(a:ℕ) * 5^(b:ℕ) = (a^3 + a^2 + a + 1) * c := by
  sorry

end max_c_value_max_c_attainable_l1439_143995


namespace loan_time_period_l1439_143997

/-- Calculates the time period of a loan using simple interest -/
theorem loan_time_period (principal : ℝ) (interest : ℝ) (rate : ℝ) : 
  principal = 900 → 
  interest = 729 → 
  rate = 9 → 
  (principal * rate * 9) / 100 = interest :=
by
  sorry

end loan_time_period_l1439_143997


namespace letter_count_theorem_l1439_143970

structure LetterCounts where
  china : ℕ
  italy : ℕ
  india : ℕ

def january : LetterCounts := { china := 6, italy := 8, india := 4 }
def february : LetterCounts := { china := 9, italy := 5, india := 7 }

def percentageChange (old new : ℕ) : ℚ :=
  (new - old : ℚ) / old * 100

def tripleCount (count : LetterCounts) : LetterCounts :=
  { china := 3 * count.china,
    italy := 3 * count.italy,
    india := 3 * count.india }

def totalLetters (a b c : LetterCounts) : ℕ :=
  a.china + a.italy + a.india +
  b.china + b.italy + b.india +
  c.china + c.italy + c.india

theorem letter_count_theorem :
  percentageChange january.china february.china = 50 ∧
  percentageChange january.italy february.italy = -37.5 ∧
  percentageChange january.india february.india = 75 ∧
  totalLetters january february (tripleCount january) = 93 := by
  sorry

end letter_count_theorem_l1439_143970


namespace second_number_value_l1439_143937

theorem second_number_value (A B C : ℝ) 
  (sum_eq : A + B + C = 98)
  (ratio_AB : A / B = 2 / 3)
  (ratio_BC : B / C = 5 / 8)
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_C : C > 0) : 
  B = 30 := by
sorry

end second_number_value_l1439_143937


namespace sugar_price_increase_vs_inflation_sugar_price_increase_specific_l1439_143908

/-- The percentage by which the rate of increase of sugar price exceeds inflation --/
theorem sugar_price_increase_vs_inflation (initial_price final_price : ℝ) 
  (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  let total_sugar_increase := (final_price - initial_price) / initial_price * 100
  let total_inflation := ((1 + inflation_rate / 100) ^ years - 1) * 100
  total_sugar_increase - total_inflation

/-- Given specific values, prove that the difference is approximately 6.81% --/
theorem sugar_price_increase_specific :
  let initial_price : ℝ := 25
  let final_price : ℝ := 33.0625
  let inflation_rate : ℝ := 12
  let years : ℕ := 2
  abs (sugar_price_increase_vs_inflation initial_price final_price inflation_rate years - 6.81) < 0.01 :=
by
  sorry


end sugar_price_increase_vs_inflation_sugar_price_increase_specific_l1439_143908


namespace no_nonzero_real_solutions_l1439_143975

theorem no_nonzero_real_solutions :
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 2 / (a + b)) := by
  sorry

end no_nonzero_real_solutions_l1439_143975


namespace roller_plate_acceleration_l1439_143945

noncomputable def g : ℝ := 10
noncomputable def R : ℝ := 1
noncomputable def r : ℝ := 0.4
noncomputable def m : ℝ := 150
noncomputable def α : ℝ := Real.arccos 0.68

theorem roller_plate_acceleration 
  (h_no_slip : True) -- Assumption of no slipping
  (h_weightless : True) -- Assumption of weightless rollers
  : ∃ (plate_acc_mag plate_acc_dir roller_acc : ℝ),
    plate_acc_mag = 4 ∧ 
    plate_acc_dir = Real.arcsin 0.4 ∧
    roller_acc = 4 := by
  sorry

end roller_plate_acceleration_l1439_143945


namespace lcm_of_5_6_10_15_l1439_143994

theorem lcm_of_5_6_10_15 : 
  Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 15)) = 30 := by sorry

end lcm_of_5_6_10_15_l1439_143994


namespace existence_of_n_consecutive_with_one_prime_l1439_143951

theorem existence_of_n_consecutive_with_one_prime (n : ℕ) : 
  ∃ k : ℕ, ∃! i : Fin n, Nat.Prime ((k : ℕ) + i) := by
  sorry

end existence_of_n_consecutive_with_one_prime_l1439_143951


namespace coefficient_of_x_is_50_l1439_143979

def expression (x : ℝ) : ℝ := 5 * (x - 6) + 6 * (8 - 3 * x^2 + 3 * x) - 9 * (3 * x - 2)

theorem coefficient_of_x_is_50 : 
  ∃ (a b c : ℝ), ∀ x, expression x = a * x^2 + 50 * x + c :=
by sorry

end coefficient_of_x_is_50_l1439_143979


namespace union_A_B_range_of_a_l1439_143926

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | 1 < x ∧ x ≤ 8} := by sorry

-- Theorem for the range of a when A ∩ C is nonempty
theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by sorry

end union_A_B_range_of_a_l1439_143926


namespace tree_distance_l1439_143967

/-- Given a yard of length 250 meters with 51 trees planted at equal distances,
    including one at each end, the distance between consecutive trees is 5 meters. -/
theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 250 →
  num_trees = 51 →
  yard_length / (num_trees - 1) = 5 :=
by sorry

end tree_distance_l1439_143967


namespace power_sixteen_divided_by_eight_l1439_143942

theorem power_sixteen_divided_by_eight (m : ℕ) : 
  m = 16^1000 → m / 8 = 2^3997 := by
  sorry

end power_sixteen_divided_by_eight_l1439_143942


namespace quarter_power_equality_l1439_143927

theorem quarter_power_equality (x : ℝ) : (1 / 4 : ℝ) ^ x = 0.25 ↔ x = 1 := by
  sorry

end quarter_power_equality_l1439_143927


namespace f_composition_equals_one_over_e_l1439_143934

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

theorem f_composition_equals_one_over_e :
  f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by sorry

end f_composition_equals_one_over_e_l1439_143934


namespace rectangle_width_l1439_143910

theorem rectangle_width (perimeter length width : ℝ) : 
  perimeter = 16 ∧ 
  width = length + 2 ∧ 
  perimeter = 2 * (length + width) →
  width = 5 := by
sorry

end rectangle_width_l1439_143910


namespace unique_solution_inequality_l1439_143906

theorem unique_solution_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
by sorry

end unique_solution_inequality_l1439_143906


namespace smallest_angle_in_4_5_7_ratio_triangle_l1439_143955

theorem smallest_angle_in_4_5_7_ratio_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  b = (5/4) * a →
  c = (7/4) * a →
  a = 45 := by
sorry

end smallest_angle_in_4_5_7_ratio_triangle_l1439_143955


namespace project_selection_count_l1439_143977

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 3 additional projects --/
def selectProjects : ℕ :=
  choose 4 1 * choose 6 1 * choose 4 1 +
  choose 6 2 * choose 4 1 +
  choose 6 1 * choose 4 2

theorem project_selection_count :
  selectProjects = 192 := by sorry

end project_selection_count_l1439_143977


namespace balloons_left_after_sharing_l1439_143983

def blue_balloons : ℕ := 303
def purple_balloons : ℕ := 453

theorem balloons_left_after_sharing :
  (blue_balloons + purple_balloons) / 2 = 378 := by
  sorry

end balloons_left_after_sharing_l1439_143983


namespace quadratic_equation_rational_solutions_l1439_143988

theorem quadratic_equation_rational_solutions : 
  ∃ (c₁ c₂ : ℕ+), 
    (c₁ ≠ c₂) ∧
    (∀ c : ℕ+, (∃ x : ℚ, 3 * x^2 + 7 * x + c.val = 0) ↔ (c = c₁ ∨ c = c₂)) ∧
    (c₁.val * c₂.val = 8) :=
by sorry

end quadratic_equation_rational_solutions_l1439_143988


namespace isosceles_triangle_perimeter_l1439_143965

/-- An isosceles triangle with side lengths 6 and 9 -/
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (is_isosceles : side1 = side2 ∨ side1 = 9 ∨ side2 = 9)
  (has_length_6 : side1 = 6 ∨ side2 = 6)
  (has_length_9 : side1 = 9 ∨ side2 = 9)

/-- The perimeter of an isosceles triangle with side lengths 6 and 9 is either 21 or 24 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.side1 + t.side2 + (if t.side1 = t.side2 then t.side1 else 
    (if t.side1 = 9 ∨ t.side2 = 9 then 9 else 6)) = 21 ∨ 
  t.side1 + t.side2 + (if t.side1 = t.side2 then t.side1 else 
    (if t.side1 = 9 ∨ t.side2 = 9 then 9 else 6)) = 24 :=
by sorry

end isosceles_triangle_perimeter_l1439_143965


namespace case1_exists_case2_not_exists_l1439_143929

-- Define a tetrahedron as a collection of 6 edge lengths
def Tetrahedron := Fin 6 → ℝ

-- Define the property of a valid tetrahedron
def is_valid_tetrahedron (t : Tetrahedron) : Prop := sorry

-- Define the conditions for case 1
def satisfies_case1 (t : Tetrahedron) : Prop :=
  (∃ i j, i ≠ j ∧ t i < 0.01 ∧ t j < 0.01) ∧
  (∃ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    t a > 1000 ∧ t b > 1000 ∧ t c > 1000 ∧ t d > 1000)

-- Define the conditions for case 2
def satisfies_case2 (t : Tetrahedron) : Prop :=
  (∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    t i < 0.01 ∧ t j < 0.01 ∧ t k < 0.01 ∧ t l < 0.01) ∧
  (∃ a b, a ≠ b ∧ t a > 1000 ∧ t b > 1000)

-- Theorem for case 1
theorem case1_exists :
  ∃ t : Tetrahedron, is_valid_tetrahedron t ∧ satisfies_case1 t := by sorry

-- Theorem for case 2
theorem case2_not_exists :
  ¬ ∃ t : Tetrahedron, is_valid_tetrahedron t ∧ satisfies_case2 t := by sorry

end case1_exists_case2_not_exists_l1439_143929


namespace product_of_five_primes_with_491_l1439_143960

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_abc_abc (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 1000 * (100 * a + 10 * b + c) + (100 * a + 10 * b + c)

theorem product_of_five_primes_with_491 :
  ∃ p₁ p₂ p₃ p₄ : ℕ,
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧ is_prime 491 ∧
    is_abc_abc (p₁ * p₂ * p₃ * p₄ * 491) ∧
    p₁ * p₂ * p₃ * p₄ * 491 = 982982 :=
  sorry

end product_of_five_primes_with_491_l1439_143960


namespace cos_54_degrees_l1439_143922

theorem cos_54_degrees : Real.cos (54 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end cos_54_degrees_l1439_143922


namespace system_of_equations_solution_l1439_143976

theorem system_of_equations_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 75)
  (eq2 : y^2 + y*z + z^2 = 49)
  (eq3 : z^2 + x*z + x^2 = 124) :
  x*y + y*z + x*z = 70 := by
  sorry

end system_of_equations_solution_l1439_143976


namespace customer_count_is_twenty_l1439_143900

/-- The number of customers who bought marbles from Mr Julien's store -/
def number_of_customers (initial_marbles final_marbles marbles_per_customer : ℕ) : ℕ :=
  (initial_marbles - final_marbles) / marbles_per_customer

/-- Theorem stating that the number of customers who bought marbles is 20 -/
theorem customer_count_is_twenty :
  number_of_customers 400 100 15 = 20 := by
  sorry

end customer_count_is_twenty_l1439_143900


namespace quadratic_inequality_range_l1439_143919

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Icc (-1) 3 :=
by sorry

end quadratic_inequality_range_l1439_143919


namespace correct_land_equation_l1439_143961

/-- Represents the relationship between arable land and forest land areas -/
def land_relationship (x y : ℝ) : Prop :=
  x + y = 2000 ∧ y = x * (30 / 100)

/-- The correct system of equations for the land areas -/
theorem correct_land_equation :
  ∀ x y : ℝ,
  (x + y = 2000 ∧ y = x * (30 / 100)) ↔ land_relationship x y :=
by sorry

end correct_land_equation_l1439_143961


namespace system_of_equations_l1439_143973

/-- Given a system of equations with parameters n and m, prove specific values of m for different conditions. -/
theorem system_of_equations (n m x y : ℤ) : 
  (n * x + (n + 1) * y = n + 2) → 
  (x - 2 * y + m * x = -5) →
  (
    (n = 1 ∧ x + 2 * y = 3 ∧ x + y = 2 → m = -4) ∧
    (n = 3 ∧ ∃ (x y : ℤ), n * x + (n + 1) * y = n + 2 ∧ x - 2 * y + m * x = -5 → m = -2 ∨ m = 0)
  ) := by sorry

end system_of_equations_l1439_143973


namespace insurance_covers_80_percent_l1439_143949

/-- Represents the medication and insurance scenario for Tom --/
structure MedicationScenario where
  pills_per_day : ℕ
  doctor_visits_per_year : ℕ
  cost_per_visit : ℕ
  cost_per_pill : ℕ
  total_annual_payment : ℕ

/-- Calculates the percentage of medication cost covered by insurance --/
def insurance_coverage_percentage (scenario : MedicationScenario) : ℚ :=
  let total_pills := scenario.pills_per_day * 365
  let medication_cost := total_pills * scenario.cost_per_pill
  let doctor_cost := scenario.doctor_visits_per_year * scenario.cost_per_visit
  let total_cost := medication_cost + doctor_cost
  let insurance_coverage := total_cost - scenario.total_annual_payment
  (insurance_coverage : ℚ) / (medication_cost : ℚ) * 100

/-- Tom's specific medication scenario --/
def tom_scenario : MedicationScenario :=
  { pills_per_day := 2
  , doctor_visits_per_year := 2
  , cost_per_visit := 400
  , cost_per_pill := 5
  , total_annual_payment := 1530 }

/-- Theorem stating that the insurance covers 80% of Tom's medication cost --/
theorem insurance_covers_80_percent :
  insurance_coverage_percentage tom_scenario = 80 := by
  sorry

end insurance_covers_80_percent_l1439_143949


namespace ernesto_extra_distance_l1439_143989

/-- Given that Renaldo drove 15 kilometers, Ernesto drove some kilometers more than one-third of Renaldo's distance, and the total distance driven by both men is 27 kilometers, prove that Ernesto drove 7 kilometers more than one-third of Renaldo's distance. -/
theorem ernesto_extra_distance (renaldo_distance : ℝ) (ernesto_distance : ℝ) (total_distance : ℝ)
  (h1 : renaldo_distance = 15)
  (h2 : ernesto_distance > (1/3) * renaldo_distance)
  (h3 : total_distance = renaldo_distance + ernesto_distance)
  (h4 : total_distance = 27) :
  ernesto_distance - (1/3) * renaldo_distance = 7 := by
  sorry

end ernesto_extra_distance_l1439_143989


namespace polynomial_expansion_l1439_143992

theorem polynomial_expansion :
  (fun z : ℝ => 3 * z^3 + 4 * z^2 - 8 * z - 5) *
  (fun z : ℝ => 2 * z^4 - 3 * z^2 + 1) =
  (fun z : ℝ => 6 * z^7 + 12 * z^6 - 25 * z^5 - 20 * z^4 + 34 * z^2 - 8 * z - 5) :=
by sorry

end polynomial_expansion_l1439_143992


namespace inequality_solution_l1439_143962

theorem inequality_solution (c : ℝ) : 
  (4 * c / 3 ≤ 8 + 4 * c ∧ 8 + 4 * c < -3 * (1 + c)) ↔ 
  (c ≥ -3 ∧ c < -11/7) :=
by sorry

end inequality_solution_l1439_143962


namespace free_all_friends_time_l1439_143950

/-- Time to pick a cheap handcuff lock in minutes -/
def cheap_time : ℕ := 10

/-- Time to pick an expensive handcuff lock in minutes -/
def expensive_time : ℕ := 15

/-- Represents the handcuffs on a person -/
structure Handcuffs :=
  (left_hand right_hand left_ankle right_ankle : Bool)

/-- True if the handcuff is expensive, False if it's cheap -/
def friend1 : Handcuffs := ⟨true, true, false, false⟩
def friend2 : Handcuffs := ⟨false, false, true, true⟩
def friend3 : Handcuffs := ⟨true, false, false, false⟩
def friend4 : Handcuffs := ⟨true, true, true, true⟩
def friend5 : Handcuffs := ⟨false, true, true, true⟩
def friend6 : Handcuffs := ⟨false, false, false, false⟩

/-- Calculate the time needed to free a friend -/
def free_time (h : Handcuffs) : ℕ :=
  (if h.left_hand then expensive_time else cheap_time) +
  (if h.right_hand then expensive_time else cheap_time) +
  (if h.left_ankle then expensive_time else cheap_time) +
  (if h.right_ankle then expensive_time else cheap_time)

/-- The total time to free all friends -/
def total_time : ℕ :=
  free_time friend1 + free_time friend2 + free_time friend3 +
  free_time friend4 + free_time friend5 + free_time friend6

theorem free_all_friends_time :
  total_time = 300 := by sorry

end free_all_friends_time_l1439_143950


namespace kg_conversion_hour_conversion_l1439_143954

-- Define conversion factors
def grams_per_kg : ℝ := 1000
def minutes_per_hour : ℝ := 60

-- Theorem 1: Convert 70 kg 50 g to kg
theorem kg_conversion (mass_kg : ℝ) (mass_g : ℝ) :
  mass_kg + mass_g / grams_per_kg = 70.05 :=
by sorry

-- Theorem 2: Convert 3.7 hours to hours and minutes
theorem hour_conversion (hours : ℝ) :
  ∃ (whole_hours : ℕ) (minutes : ℕ),
    hours = whole_hours + minutes / minutes_per_hour ∧
    whole_hours = 3 ∧
    minutes = 42 :=
by sorry

end kg_conversion_hour_conversion_l1439_143954


namespace no_four_integers_product_square_l1439_143916

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

end no_four_integers_product_square_l1439_143916


namespace max_value_constraint_l1439_143939

theorem max_value_constraint (x y : ℝ) 
  (h1 : |x - y| ≤ 2) 
  (h2 : |3*x + y| ≤ 6) : 
  x^2 + y^2 ≤ 10 :=
by sorry

end max_value_constraint_l1439_143939


namespace sphere_plane_distance_l1439_143940

/-- Given a sphere and a plane cutting it, this theorem relates the radius of the sphere,
    the radius of the circular section, and the distance from the sphere's center to the plane. -/
theorem sphere_plane_distance (R r d : ℝ) : R = 2 * Real.sqrt 3 → r = 2 → d ^ 2 + r ^ 2 = R ^ 2 → d = 2 * Real.sqrt 2 := by
  sorry

end sphere_plane_distance_l1439_143940


namespace casas_alvero_prime_l1439_143981

/-- A polynomial with rational coefficients -/
def RationalPolynomial : Type := ℚ → ℚ

/-- The degree of a polynomial -/
def degree (p : RationalPolynomial) : ℕ := sorry

/-- The kth derivative of a polynomial -/
def derivative (p : RationalPolynomial) (k : ℕ) : RationalPolynomial := sorry

/-- Checks if a rational number is a root of a polynomial -/
def is_root (p : RationalPolynomial) (r : ℚ) : Prop := p r = 0

theorem casas_alvero_prime (p : RationalPolynomial) (d : ℕ) :
  degree p = d →
  Nat.Prime d →
  (∀ k : ℕ, 1 ≤ k → k ≤ d - 1 →
    ∃ r : ℚ, is_root p r ∧ is_root (derivative p k) r) →
  ∃ a b c : ℚ, ∀ x : ℚ, p x = c * (a * x + b) ^ d :=
sorry

end casas_alvero_prime_l1439_143981


namespace triangle_properties_l1439_143944

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The main theorem about properties of triangle ABC -/
theorem triangle_properties (t : Triangle) :
  (t.A - t.B = t.C → ¬(t.A = Real.pi/2 ∧ t.c > t.a ∧ t.c > t.b)) ∧
  (t.a^2 = t.b^2 - t.c^2 → t.B = Real.pi/2) ∧
  (t.A / (t.A + t.B + t.C) = 1/6 ∧ t.B / (t.A + t.B + t.C) = 1/3 ∧ t.C / (t.A + t.B + t.C) = 1/2 → t.C = Real.pi/2) ∧
  (t.a^2 / (t.a^2 + t.b^2 + t.c^2) = 9/50 ∧ t.b^2 / (t.a^2 + t.b^2 + t.c^2) = 16/50 ∧ t.c^2 / (t.a^2 + t.b^2 + t.c^2) = 25/50 → t.a^2 + t.b^2 = t.c^2) :=
by
  sorry


end triangle_properties_l1439_143944


namespace probability_of_white_ball_l1439_143936

/-- Given a bag with white and red balls, prove that the probability of drawing a white ball equals 2/6 -/
theorem probability_of_white_ball (b : ℕ) : 
  let white_balls := b - 4
  let red_balls := b + 46
  let total_balls := white_balls + red_balls
  let prob_white := white_balls / total_balls
  prob_white = 2 / 6 := by
sorry

end probability_of_white_ball_l1439_143936


namespace square_root_81_l1439_143925

theorem square_root_81 : ∀ (x : ℝ), x^2 = 81 ↔ x = 9 ∨ x = -9 := by sorry

end square_root_81_l1439_143925


namespace batsman_average_l1439_143986

/-- Calculates the average runs for a batsman given two sets of matches --/
def average_runs (matches1 : ℕ) (average1 : ℕ) (matches2 : ℕ) (average2 : ℕ) : ℚ :=
  let total_runs := matches1 * average1 + matches2 * average2
  let total_matches := matches1 + matches2
  (total_runs : ℚ) / total_matches

/-- Proves that the average runs for 45 matches is 42 given the specified conditions --/
theorem batsman_average :
  average_runs 30 50 15 26 = 42 := by
  sorry

end batsman_average_l1439_143986


namespace total_wheels_at_station_l1439_143963

/-- Calculates the total number of wheels at a train station -/
theorem total_wheels_at_station
  (num_trains : ℕ)
  (carriages_per_train : ℕ)
  (wheel_rows_per_carriage : ℕ)
  (wheels_per_row : ℕ)
  (h1 : num_trains = 4)
  (h2 : carriages_per_train = 4)
  (h3 : wheel_rows_per_carriage = 3)
  (h4 : wheels_per_row = 5) :
  num_trains * carriages_per_train * wheel_rows_per_carriage * wheels_per_row = 240 :=
by sorry

end total_wheels_at_station_l1439_143963


namespace system_of_equations_solution_l1439_143941

theorem system_of_equations_solution (a b x y : ℝ) : 
  (x - y * Real.sqrt (x^2 - y^2)) / Real.sqrt (1 - x^2 + y^2) = a ∧
  (y - x * Real.sqrt (x^2 - y^2)) / Real.sqrt (1 - x^2 + y^2) = b →
  x = (a + b * Real.sqrt (a^2 - b^2)) / Real.sqrt (1 - a^2 + b^2) ∧
  y = (b + a * Real.sqrt (a^2 - b^2)) / Real.sqrt (1 - a^2 + b^2) := by
sorry

end system_of_equations_solution_l1439_143941


namespace isosceles_when_root_is_one_right_angled_when_equal_roots_l1439_143959

/-- Triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Quadratic equation associated with the triangle -/
def quadratic_equation (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 - 2 * t.b * x - t.a + t.c

theorem isosceles_when_root_is_one (t : Triangle) :
  quadratic_equation t 1 = 0 → t.b = t.c :=
sorry

theorem right_angled_when_equal_roots (t : Triangle) :
  (∃ x : ℝ, ∀ y : ℝ, quadratic_equation t y = 0 ↔ y = x) →
  t.a^2 + t.b^2 = t.c^2 :=
sorry

end isosceles_when_root_is_one_right_angled_when_equal_roots_l1439_143959


namespace min_value_on_circle_l1439_143946

theorem min_value_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 2)^2 = 1) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a - 2)^2 + (b - 2)^2 = 1 → a^2 + b^2 ≥ m) ∧
  (m = 9 - 4 * Real.sqrt 2) :=
sorry

end min_value_on_circle_l1439_143946


namespace decreased_amount_l1439_143980

theorem decreased_amount (N : ℝ) (A : ℝ) (h1 : N = 50) (h2 : 0.20 * N - A = 6) : A = 4 := by
  sorry

end decreased_amount_l1439_143980


namespace sum_of_real_roots_of_quartic_l1439_143915

theorem sum_of_real_roots_of_quartic (x : ℝ) :
  let f : ℝ → ℝ := λ x => x^4 - 4*x - 1
  ∃ (r₁ r₂ : ℝ), (f r₁ = 0 ∧ f r₂ = 0) ∧ (∀ r : ℝ, f r = 0 → r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = Real.sqrt 2 :=
by sorry

end sum_of_real_roots_of_quartic_l1439_143915


namespace greatest_gcd_pentagonal_l1439_143909

def P (n : ℕ+) : ℕ := (n : ℕ).succ * n

theorem greatest_gcd_pentagonal (n : ℕ+) : 
  (Nat.gcd (6 * P n) (n.val - 2) : ℕ) ≤ 24 ∧ 
  ∃ m : ℕ+, (Nat.gcd (6 * P m) (m.val - 2) : ℕ) = 24 :=
sorry

end greatest_gcd_pentagonal_l1439_143909


namespace smallest_n_for_sqrt_difference_l1439_143957

theorem smallest_n_for_sqrt_difference (n : ℕ) : n ≥ 2501 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.01 :=
sorry

end smallest_n_for_sqrt_difference_l1439_143957


namespace sequence_sum_zero_l1439_143921

-- Define the sequence type
def Sequence := Fin 12 → ℤ

-- Define the property of sum of three consecutive terms being 40
def ConsecutiveSum (seq : Sequence) : Prop :=
  ∀ i : Fin 10, seq i + seq (i + 1) + seq (i + 2) = 40

-- Define the theorem
theorem sequence_sum_zero (seq : Sequence) 
  (h1 : ConsecutiveSum seq) 
  (h2 : seq 2 = 9) : 
  seq 0 + seq 11 = 0 :=
sorry

end sequence_sum_zero_l1439_143921


namespace dietitian_excess_calories_l1439_143907

/-- Calculates the excess calories consumed given the total lunch calories and the fraction eaten -/
def excess_calories (total_calories : ℕ) (fraction_eaten : ℚ) (recommended_calories : ℕ) : ℤ :=
  ⌊(fraction_eaten * total_calories : ℚ)⌋ - recommended_calories

/-- Proves that eating 3/4 of a 40-calorie lunch exceeds the recommended 25 calories by 5 -/
theorem dietitian_excess_calories :
  excess_calories 40 (3/4 : ℚ) 25 = 5 := by
  sorry

end dietitian_excess_calories_l1439_143907


namespace parallelogram_side_sum_l1439_143913

/-- A parallelogram with consecutive side lengths 12, 5y-3, 3x+2, and 9 has x+y equal to 86/15 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (3*x + 2 = 12) → (5*y - 3 = 9) → x + y = 86/15 := by
  sorry

end parallelogram_side_sum_l1439_143913
