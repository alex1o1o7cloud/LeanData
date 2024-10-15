import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_t_l1769_176963

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def t (m : ℝ) : ℝ := (2 * m + log m / m - m * log m) / 2

theorem max_value_of_t :
  ∃ (m : ℝ), m > 1 ∧ ∀ (x : ℝ), x > 1 → t x ≤ t m ∧ t m = (exp 2 + 1) / (2 * exp 1) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_t_l1769_176963


namespace NUMINAMATH_CALUDE_five_digit_division_sum_l1769_176934

theorem five_digit_division_sum (ABCDE : ℕ) : 
  ABCDE ≥ 10000 ∧ ABCDE < 100000 ∧ ABCDE % 6 = 0 ∧ ABCDE / 6 = 13579 →
  (ABCDE / 100) + (ABCDE % 100) = 888 := by
sorry

end NUMINAMATH_CALUDE_five_digit_division_sum_l1769_176934


namespace NUMINAMATH_CALUDE_university_box_cost_l1769_176904

theorem university_box_cost (box_length box_width box_height : ℝ)
  (box_cost : ℝ) (total_volume : ℝ) :
  box_length = 20 ∧ box_width = 20 ∧ box_height = 12 ∧
  box_cost = 0.5 ∧ total_volume = 2160000 →
  (⌈total_volume / (box_length * box_width * box_height)⌉ : ℝ) * box_cost = 225 := by
  sorry

end NUMINAMATH_CALUDE_university_box_cost_l1769_176904


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1769_176922

theorem gcd_lcm_sum : Nat.gcd 40 72 + Nat.lcm 48 18 = 152 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1769_176922


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_2007_odd_integers_l1769_176960

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square (n : ℕ) : ℕ := n * n

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_squares (list : List ℕ) : ℕ :=
  list.map square |> List.sum

theorem units_digit_of_sum_of_squares_2007_odd_integers :
  units_digit (sum_of_squares (first_n_odd_integers 2007)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_2007_odd_integers_l1769_176960


namespace NUMINAMATH_CALUDE_min_value_abs_sum_min_value_achievable_l1769_176961

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 4| ≥ 3 := by sorry

theorem min_value_achievable : ∃ x : ℝ, |x - 1| + |x - 4| = 3 := by sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_min_value_achievable_l1769_176961


namespace NUMINAMATH_CALUDE_number_of_men_l1769_176936

theorem number_of_men (max_handshakes : ℕ) : max_handshakes = 153 → ∃ n : ℕ, n = 18 ∧ max_handshakes = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_l1769_176936


namespace NUMINAMATH_CALUDE_max_payment_is_31_l1769_176943

def is_valid_number (n : ℕ) : Prop :=
  2000 ≤ n ∧ n ≤ 2099

def divisibility_payment (n : ℕ) : ℕ :=
  (if n % 1 = 0 then 1 else 0) +
  (if n % 3 = 0 then 3 else 0) +
  (if n % 5 = 0 then 5 else 0) +
  (if n % 7 = 0 then 7 else 0) +
  (if n % 9 = 0 then 9 else 0) +
  (if n % 11 = 0 then 11 else 0)

theorem max_payment_is_31 :
  ∃ n : ℕ, is_valid_number n ∧
    divisibility_payment n = 31 ∧
    ∀ m : ℕ, is_valid_number m → divisibility_payment m ≤ 31 :=
by sorry

end NUMINAMATH_CALUDE_max_payment_is_31_l1769_176943


namespace NUMINAMATH_CALUDE_cube_sum_solutions_l1769_176994

def is_cube_sum (a b c : ℕ+) : Prop :=
  ∃ n : ℕ+, 2^(Nat.factorial a.val) + 2^(Nat.factorial b.val) + 2^(Nat.factorial c.val) = n^3

theorem cube_sum_solutions :
  ∀ a b c : ℕ+, is_cube_sum a b c ↔ 
    ((a, b, c) = (1, 1, 2) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (2, 1, 1)) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_solutions_l1769_176994


namespace NUMINAMATH_CALUDE_area_ABC_is_72_l1769_176900

def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_ABC_is_72 :
  let area_XYZ := area_triangle X Y Z
  let area_ABC := area_XYZ / 0.1111111111111111
  area_ABC = 72 := by sorry

end NUMINAMATH_CALUDE_area_ABC_is_72_l1769_176900


namespace NUMINAMATH_CALUDE_bruce_savings_l1769_176915

-- Define the given amounts and rates
def aunt_money : ℝ := 87.32
def grandfather_money : ℝ := 152.68
def savings_rate : ℝ := 0.35
def interest_rate : ℝ := 0.025

-- Define the function to calculate the amount after one year
def amount_after_one_year (aunt_money grandfather_money savings_rate interest_rate : ℝ) : ℝ :=
  let total_money := aunt_money + grandfather_money
  let saved_amount := total_money * savings_rate
  let interest := saved_amount * interest_rate
  saved_amount + interest

-- Theorem statement
theorem bruce_savings : 
  amount_after_one_year aunt_money grandfather_money savings_rate interest_rate = 86.10 := by
  sorry

end NUMINAMATH_CALUDE_bruce_savings_l1769_176915


namespace NUMINAMATH_CALUDE_sqrt_difference_comparison_l1769_176973

theorem sqrt_difference_comparison (m : ℝ) (h : m > 1) :
  Real.sqrt (m + 1) - Real.sqrt m < Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_comparison_l1769_176973


namespace NUMINAMATH_CALUDE_laura_charge_account_balance_l1769_176959

/-- Calculates the total amount owed after applying simple interest --/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating that the total amount owed is $37.10 given the problem conditions --/
theorem laura_charge_account_balance : 
  total_amount_owed 35 0.06 1 = 37.10 := by
  sorry

end NUMINAMATH_CALUDE_laura_charge_account_balance_l1769_176959


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l1769_176951

theorem max_value_sum_of_roots (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  (Real.sqrt (49 + x) + Real.sqrt (49 - x) = 14 ↔ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l1769_176951


namespace NUMINAMATH_CALUDE_intersection_condition_l1769_176985

def A (a : ℝ) : Set ℝ := {-1, 0, a}

def B : Set ℝ := {x : ℝ | 1/3 < x ∧ x < 1}

theorem intersection_condition (a : ℝ) :
  (A a ∩ B).Nonempty → 1/3 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l1769_176985


namespace NUMINAMATH_CALUDE_factor_expression_l1769_176930

theorem factor_expression (y : ℝ) : 16 * y^3 + 8 * y^2 = 8 * y^2 * (2 * y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1769_176930


namespace NUMINAMATH_CALUDE_sin_4theta_l1769_176964

theorem sin_4theta (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (4 + Complex.I * Real.sqrt 3) / 5) : 
  Real.sin (4 * θ) = 208 * Real.sqrt 3 / 625 := by
  sorry

end NUMINAMATH_CALUDE_sin_4theta_l1769_176964


namespace NUMINAMATH_CALUDE_line_properties_l1769_176948

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y + 3 = 0

-- Define the point (0, -3)
def point : ℝ × ℝ := (0, -3)

-- Define the other line
def other_line (x y : ℝ) : Prop := x + (Real.sqrt 3 / 3) * y + Real.sqrt 3 = 0

theorem line_properties :
  (∀ x y, line_l x y ↔ other_line x y) ∧
  line_l point.1 point.2 ∧
  (∀ x y, line_l x y → y / x ≠ Real.tan (60 * π / 180)) ∧
  (∃ x, line_l x 0 ∧ x = -Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1769_176948


namespace NUMINAMATH_CALUDE_opposite_of_two_l1769_176913

-- Define the concept of opposite
def opposite (x : ℝ) : ℝ := -x

-- Theorem statement
theorem opposite_of_two : opposite 2 = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_l1769_176913


namespace NUMINAMATH_CALUDE_total_pamphlets_is_10700_l1769_176974

-- Define the printing rates and durations
def mike_initial_rate : ℕ := 600
def mike_initial_duration : ℕ := 9
def mike_final_duration : ℕ := 2

def leo_initial_rate : ℕ := 2 * mike_initial_rate
def leo_initial_duration : ℕ := mike_initial_duration / 3

def sally_initial_rate : ℕ := 3 * mike_initial_rate
def sally_initial_duration : ℕ := leo_initial_duration / 2
def sally_final_duration : ℕ := 1

-- Define the function to calculate total pamphlets
def calculate_total_pamphlets : ℕ :=
  -- Mike's pamphlets
  let mike_pamphlets := mike_initial_rate * mike_initial_duration + 
                        (mike_initial_rate / 3) * mike_final_duration

  -- Leo's pamphlets
  let leo_pamphlets := leo_initial_rate * 1 + 
                       (leo_initial_rate / 2) * 1 + 
                       (leo_initial_rate / 4) * 1

  -- Sally's pamphlets
  let sally_pamphlets := sally_initial_rate * sally_initial_duration + 
                         (leo_initial_rate / 2) * sally_final_duration

  mike_pamphlets + leo_pamphlets + sally_pamphlets

-- Theorem statement
theorem total_pamphlets_is_10700 :
  calculate_total_pamphlets = 10700 := by
  sorry

end NUMINAMATH_CALUDE_total_pamphlets_is_10700_l1769_176974


namespace NUMINAMATH_CALUDE_method1_saves_more_l1769_176947

/-- Represents the price of a badminton racket in yuan -/
def racket_price : ℕ := 20

/-- Represents the price of a shuttlecock in yuan -/
def shuttlecock_price : ℕ := 5

/-- Represents the number of rackets to be purchased -/
def num_rackets : ℕ := 4

/-- Represents the number of shuttlecocks to be purchased -/
def num_shuttlecocks : ℕ := 30

/-- Calculates the cost using discount method ① -/
def cost_method1 : ℕ := racket_price * num_rackets + shuttlecock_price * (num_shuttlecocks - num_rackets)

/-- Calculates the cost using discount method ② -/
def cost_method2 : ℚ := (racket_price * num_rackets + shuttlecock_price * num_shuttlecocks) * 92 / 100

/-- Theorem stating that discount method ① saves more money than method ② -/
theorem method1_saves_more : cost_method1 < cost_method2 := by
  sorry


end NUMINAMATH_CALUDE_method1_saves_more_l1769_176947


namespace NUMINAMATH_CALUDE_square_increasing_on_positive_reals_l1769_176907

theorem square_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → x₁^2 < x₂^2 := by
  sorry

end NUMINAMATH_CALUDE_square_increasing_on_positive_reals_l1769_176907


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1769_176978

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x

theorem tangent_line_equation (a : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → 
    |((f a x - f a 1) / (x - 1)) - 3| < ε) →
  ∃ b c : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 ∧ y = f a 1) → 
    3 * x - y - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1769_176978


namespace NUMINAMATH_CALUDE_expensive_candy_price_l1769_176905

/-- Proves that the price of the more expensive candy is $3 per pound -/
theorem expensive_candy_price
  (total_mixture : ℝ)
  (selling_price : ℝ)
  (expensive_amount : ℝ)
  (cheap_price : ℝ)
  (h1 : total_mixture = 80)
  (h2 : selling_price = 2.20)
  (h3 : expensive_amount = 16)
  (h4 : cheap_price = 2)
  : ∃ (expensive_price : ℝ),
    expensive_price * expensive_amount + cheap_price * (total_mixture - expensive_amount) =
    selling_price * total_mixture ∧ expensive_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_expensive_candy_price_l1769_176905


namespace NUMINAMATH_CALUDE_current_speed_l1769_176911

/-- Given a boat's downstream and upstream speeds, calculate the speed of the current. -/
theorem current_speed (downstream_time upstream_time : ℚ) : 
  downstream_time = 6 / 60 → 
  upstream_time = 10 / 60 → 
  (1 / downstream_time - 1 / upstream_time) / 2 = 2 := by
  sorry

#check current_speed

end NUMINAMATH_CALUDE_current_speed_l1769_176911


namespace NUMINAMATH_CALUDE_square_side_length_l1769_176997

theorem square_side_length 
  (x y : ℕ+) 
  (h1 : Nat.gcd x.val y.val = 5)
  (h2 : ∃ (s : ℝ), s > 0 ∧ x.val^2 + y.val^2 = 2 * s^2)
  (h3 : (169 : ℝ) / 6 * Nat.lcm x.val y.val = 2 * s^2) :
  s = 65 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_square_side_length_l1769_176997


namespace NUMINAMATH_CALUDE_dans_remaining_potatoes_l1769_176991

/-- Given an initial number of potatoes and a number of eaten potatoes,
    calculate the remaining number of potatoes. -/
def remaining_potatoes (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem stating that Dan's remaining potatoes is 3 given the initial conditions. -/
theorem dans_remaining_potatoes :
  remaining_potatoes 7 4 = 3 := by sorry

end NUMINAMATH_CALUDE_dans_remaining_potatoes_l1769_176991


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_five_l1769_176919

theorem smallest_five_digit_mod_five : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  n % 5 = 4 ∧ 
  ∀ m : ℕ, (m ≥ 10000 ∧ m < 100000 ∧ m % 5 = 4) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_five_l1769_176919


namespace NUMINAMATH_CALUDE_square_difference_forty_thirtynine_l1769_176944

theorem square_difference_forty_thirtynine : (40 : ℕ)^2 - (39 : ℕ)^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_forty_thirtynine_l1769_176944


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1769_176957

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1769_176957


namespace NUMINAMATH_CALUDE_pipe_fill_time_l1769_176979

/-- Given three pipes P, Q, and R that can fill a tank, this theorem proves
    that if P fills the tank in 2 hours, Q in 4 hours, and all pipes together
    in 1.2 hours, then R fills the tank in 12 hours. -/
theorem pipe_fill_time (fill_rate_P fill_rate_Q fill_rate_R : ℝ) : 
  fill_rate_P = 1 / 2 →
  fill_rate_Q = 1 / 4 →
  fill_rate_P + fill_rate_Q + fill_rate_R = 1 / 1.2 →
  fill_rate_R = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l1769_176979


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l1769_176968

/-- Represents the number of ways to distribute balls into boxes -/
def distribute_balls (total_balls : ℕ) (num_boxes : ℕ) (min_balls : ℕ → ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 15 ways to distribute 10 balls into 3 boxes -/
theorem ball_distribution_theorem :
  distribute_balls 10 3 (fun i => i) = 15 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l1769_176968


namespace NUMINAMATH_CALUDE_not_product_of_two_primes_l1769_176995

theorem not_product_of_two_primes (n : ℕ) (h : n ≥ 2) :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧
  (a * b * c ∣ 2^(4*n + 2) + 1) :=
sorry

end NUMINAMATH_CALUDE_not_product_of_two_primes_l1769_176995


namespace NUMINAMATH_CALUDE_jessicas_class_farm_trip_cost_l1769_176909

/-- Calculate the total cost for a field trip to a farm -/
def farm_trip_cost (num_students : ℕ) (num_adults : ℕ) (student_fee : ℕ) (adult_fee : ℕ) : ℕ :=
  num_students * student_fee + num_adults * adult_fee

/-- Theorem: The total cost for Jessica's class field trip to the farm is $199 -/
theorem jessicas_class_farm_trip_cost : farm_trip_cost 35 4 5 6 = 199 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_class_farm_trip_cost_l1769_176909


namespace NUMINAMATH_CALUDE_complex_product_real_l1769_176980

theorem complex_product_real (z₁ z₂ : ℂ) :
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  (z₁ * z₂).im = 0 ↔ z₂ = 4 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_real_l1769_176980


namespace NUMINAMATH_CALUDE_fraction_inequality_l1769_176939

theorem fraction_inequality (a : ℝ) : a > 1 → (2*a + 1)/(a - 1) > 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1769_176939


namespace NUMINAMATH_CALUDE_power_of_power_three_cubed_squared_l1769_176914

theorem power_of_power_three_cubed_squared : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_cubed_squared_l1769_176914


namespace NUMINAMATH_CALUDE_solution_set_l1769_176990

def system_solution (x : ℝ) : Prop :=
  x / 3 ≥ -1 ∧ 3 * x + 4 < 1

theorem solution_set : ∀ x : ℝ, system_solution x ↔ -3 ≤ x ∧ x < -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_l1769_176990


namespace NUMINAMATH_CALUDE_seating_arrangements_l1769_176916

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def totalArrangements : ℕ := factorial 10

def restrictedArrangements : ℕ := factorial 7 * factorial 4

theorem seating_arrangements :
  totalArrangements - restrictedArrangements = 3507840 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1769_176916


namespace NUMINAMATH_CALUDE_calculation_proof_l1769_176982

theorem calculation_proof : 3 * 16 + 3 * 17 + 3 * 20 + 11 = 170 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1769_176982


namespace NUMINAMATH_CALUDE_income_mean_difference_l1769_176912

def num_families : ℕ := 1500

def correct_largest_income_1 : ℝ := 150000
def correct_largest_income_2 : ℝ := 148000
def incorrect_largest_income : ℝ := 1500000

def sum_other_incomes : ℝ := sorry  -- This represents the sum S in the solution

theorem income_mean_difference :
  let actual_mean := (sum_other_incomes + correct_largest_income_1 + correct_largest_income_2) / num_families
  let incorrect_mean := (sum_other_incomes + 2 * incorrect_largest_income) / num_families
  incorrect_mean - actual_mean = 1801.33 := by sorry

end NUMINAMATH_CALUDE_income_mean_difference_l1769_176912


namespace NUMINAMATH_CALUDE_final_output_is_25_l1769_176935

def algorithm_output : ℕ → ℕ
| 0 => 25
| (n+1) => if 2*n + 1 < 10 then algorithm_output n else 2*(2*n + 1) + 3

theorem final_output_is_25 : algorithm_output 0 = 25 := by
  sorry

end NUMINAMATH_CALUDE_final_output_is_25_l1769_176935


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1769_176901

theorem expansion_coefficient (n : ℕ) : 
  ((-2)^n : ℤ) + ((-2)^(n-1) : ℤ) * n = -128 ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1769_176901


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1769_176970

theorem inequality_equivalence (x : ℝ) : 3 - 1 / (3 * x + 2) < 5 ↔ x < -7/6 ∨ x > -2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1769_176970


namespace NUMINAMATH_CALUDE_boat_distance_main_theorem_l1769_176933

/-- The distance between two boats given specific angles and fort height -/
theorem boat_distance (fort_height : ℝ) (angle1 angle2 base_angle : ℝ) : ℝ :=
  let boat_distance := 30
  by
    -- Assuming fort_height = 30, angle1 = 45°, angle2 = 30°, base_angle = 30°
    sorry

/-- Main theorem stating the distance between the boats is 30 meters -/
theorem main_theorem : boat_distance 30 (45 * π / 180) (30 * π / 180) (30 * π / 180) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_distance_main_theorem_l1769_176933


namespace NUMINAMATH_CALUDE_area_ratio_square_to_rectangle_l1769_176967

/-- The ratio of the area of a square with side length 48 cm to the area of a rectangle with dimensions 56 cm by 63 cm is 2/3. -/
theorem area_ratio_square_to_rectangle : 
  let square_side : ℝ := 48
  let rect_width : ℝ := 56
  let rect_height : ℝ := 63
  let square_area := square_side ^ 2
  let rect_area := rect_width * rect_height
  square_area / rect_area = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_square_to_rectangle_l1769_176967


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1769_176908

theorem x_minus_y_value (x y : ℝ) (h : |x + 2| + (y - 3)^2 = 0) : x - y = -5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1769_176908


namespace NUMINAMATH_CALUDE_incorrect_statement_about_parallelogram_l1769_176977

-- Define a parallelogram
structure Parallelogram :=
  (diagonals_bisect : Bool)
  (diagonals_perpendicular : Bool)

-- Define the properties of a parallelogram
def parallelogram_properties : Parallelogram :=
  { diagonals_bisect := true,
    diagonals_perpendicular := false }

-- Theorem to prove
theorem incorrect_statement_about_parallelogram :
  ¬(parallelogram_properties.diagonals_bisect ∧ parallelogram_properties.diagonals_perpendicular) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_about_parallelogram_l1769_176977


namespace NUMINAMATH_CALUDE_canned_food_bins_l1769_176955

theorem canned_food_bins (soup vegetables pasta : Real) 
  (h1 : soup = 0.125)
  (h2 : vegetables = 0.125)
  (h3 : pasta = 0.5) :
  soup + vegetables + pasta = 0.75 := by
sorry

end NUMINAMATH_CALUDE_canned_food_bins_l1769_176955


namespace NUMINAMATH_CALUDE_product_of_ab_l1769_176988

theorem product_of_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 31) : a * b = -11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_ab_l1769_176988


namespace NUMINAMATH_CALUDE_gcd_442872_312750_l1769_176940

theorem gcd_442872_312750 : Nat.gcd 442872 312750 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_442872_312750_l1769_176940


namespace NUMINAMATH_CALUDE_sin_pi_fourth_plus_alpha_l1769_176941

theorem sin_pi_fourth_plus_alpha (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.tan (α - π/4) = 1/3) : Real.sin (π/4 + α) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_fourth_plus_alpha_l1769_176941


namespace NUMINAMATH_CALUDE_smallest_prime_is_prime_q_value_l1769_176921

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def smallest_prime : ℕ := 2

theorem smallest_prime_is_prime : is_prime smallest_prime := by sorry

theorem q_value (p q : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h_relation : q = 13 * p + 1) 
  (h_smallest : p = smallest_prime) : 
  q = 29 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_is_prime_q_value_l1769_176921


namespace NUMINAMATH_CALUDE_five_sided_polygon_angle_sum_l1769_176966

theorem five_sided_polygon_angle_sum 
  (A B C x y : ℝ) 
  (h1 : A = 28)
  (h2 : B = 74)
  (h3 : C = 26)
  (h4 : A + B + (360 - x) + 90 + (116 - y) = 540) :
  x + y = 128 := by
  sorry

end NUMINAMATH_CALUDE_five_sided_polygon_angle_sum_l1769_176966


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l1769_176975

theorem square_perimeter_relation (perimeter_A : ℝ) (area_ratio : ℝ) : 
  perimeter_A = 36 →
  area_ratio = 1/3 →
  let side_A := perimeter_A / 4
  let area_A := side_A ^ 2
  let area_B := area_ratio * area_A
  let side_B := Real.sqrt area_B
  let perimeter_B := 4 * side_B
  perimeter_B = 12 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l1769_176975


namespace NUMINAMATH_CALUDE_solve_for_m_l1769_176998

-- Define the functions f and g
def f (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + m
def g (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + 5*m

-- State the theorem
theorem solve_for_m : 
  ∀ m : ℚ, 3 * (f m 5) = 2 * (g m 5) → m = 10/7 := by sorry

end NUMINAMATH_CALUDE_solve_for_m_l1769_176998


namespace NUMINAMATH_CALUDE_smallest_k_with_given_remainders_l1769_176999

theorem smallest_k_with_given_remainders : ∃! k : ℕ,
  k > 1 ∧
  k % 13 = 1 ∧
  k % 8 = 1 ∧
  k % 4 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 8 = 1 ∧ m % 4 = 1 → k ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_given_remainders_l1769_176999


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1769_176992

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1769_176992


namespace NUMINAMATH_CALUDE_store_a_cheaper_for_300_l1769_176976

def cost_store_a (x : ℕ) : ℝ :=
  if x ≤ 100 then 5 * x else 4 * x + 100

def cost_store_b (x : ℕ) : ℝ :=
  4.5 * x

theorem store_a_cheaper_for_300 :
  cost_store_a 300 < cost_store_b 300 :=
sorry

end NUMINAMATH_CALUDE_store_a_cheaper_for_300_l1769_176976


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1769_176938

-- Define the functions
def f (a b x : ℝ) : ℝ := -2 * abs (x - a) + b
def g (c d x : ℝ) : ℝ := 2 * abs (x - c) + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) : 
  (f a b 1 = 4 ∧ f a b 7 = 0 ∧ g c d 1 = 4 ∧ g c d 7 = 0) → a + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l1769_176938


namespace NUMINAMATH_CALUDE_blithe_toy_count_l1769_176971

/-- The number of toys Blithe has after losing some and finding some -/
def finalToyCount (initial lost found : ℕ) : ℕ :=
  initial - lost + found

/-- Theorem: Given Blithe's initial toy count, the number of toys lost, and the number of toys found,
    the final toy count is equal to the initial count minus the lost toys plus the found toys -/
theorem blithe_toy_count : finalToyCount 40 6 9 = 43 := by
  sorry

end NUMINAMATH_CALUDE_blithe_toy_count_l1769_176971


namespace NUMINAMATH_CALUDE_no_valid_list_exists_l1769_176906

theorem no_valid_list_exists : ¬ ∃ (list : List ℤ), 
  (list.length = 10) ∧ 
  (∀ i j k, i + 1 = j ∧ j + 1 = k → i < list.length ∧ k < list.length → 
    (list.get ⟨i, sorry⟩ * list.get ⟨j, sorry⟩ * list.get ⟨k, sorry⟩) % 6 = 0) ∧
  (∀ i j, i + 1 = j → j < list.length → 
    (list.get ⟨i, sorry⟩ * list.get ⟨j, sorry⟩) % 6 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_list_exists_l1769_176906


namespace NUMINAMATH_CALUDE_fraction_sum_is_one_equation_no_solution_l1769_176958

-- Problem 1
theorem fraction_sum_is_one (a b : ℝ) (h : a ≠ b) :
  a / (a - b) + b / (b - a) = 1 := by sorry

-- Problem 2
theorem equation_no_solution :
  ¬∃ x : ℝ, (1 / (x - 2) = (1 - x) / (2 - x) - 3) := by sorry

end NUMINAMATH_CALUDE_fraction_sum_is_one_equation_no_solution_l1769_176958


namespace NUMINAMATH_CALUDE_student_score_problem_l1769_176945

theorem student_score_problem (total_questions : ℕ) (student_score : ℤ) 
  (h1 : total_questions = 100)
  (h2 : student_score = 61) : 
  ∃ (correct_answers : ℕ), 
    correct_answers = 87 ∧ 
    student_score = correct_answers - 2 * (total_questions - correct_answers) :=
by
  sorry

end NUMINAMATH_CALUDE_student_score_problem_l1769_176945


namespace NUMINAMATH_CALUDE_initial_stock_proof_l1769_176950

/-- The initial number of books in John's bookshop -/
def initial_books : ℕ := 1400

/-- The number of books sold over 5 days -/
def books_sold : ℕ := 402

/-- The percentage of books sold, expressed as a real number between 0 and 1 -/
def percentage_sold : ℝ := 0.2871428571428571

theorem initial_stock_proof :
  (books_sold : ℝ) / initial_books = percentage_sold :=
by sorry

end NUMINAMATH_CALUDE_initial_stock_proof_l1769_176950


namespace NUMINAMATH_CALUDE_area_ratio_inscribed_squares_l1769_176989

/-- A square inscribed in a circle -/
structure InscribedSquare :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (side : ℝ)

/-- A square with two vertices on a side of another square and two vertices on a circle -/
structure PartiallyInscribedSquare :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (side : ℝ)

/-- The theorem stating the relationship between the areas of the two squares -/
theorem area_ratio_inscribed_squares 
  (ABCD : InscribedSquare) 
  (EFGH : PartiallyInscribedSquare) 
  (h1 : ABCD.center = EFGH.center) 
  (h2 : ABCD.radius = EFGH.radius) 
  (h3 : ABCD.side ^ 2 = 1) : 
  EFGH.side ^ 2 = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_inscribed_squares_l1769_176989


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l1769_176954

/-- An isosceles triangle with given properties has an area of 54 square centimeters -/
theorem isosceles_triangle_area (a b : ℝ) (h_isosceles : a = b) (h_perimeter : 2 * a + b = 36)
  (h_base_angles : 2 * Real.arccos ((a^2 - b^2/4) / a^2) = 130 * π / 180)
  (h_inradius : (a * b) / (a + b + (a^2 - b^2/4).sqrt) = 3) : 
  a * b * Real.sin (Real.arccos ((a^2 - b^2/4) / a^2)) / 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l1769_176954


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1769_176965

/-- The line ax + y + a + 1 = 0 always passes through the point (-1, -1) for all values of a. -/
theorem line_passes_through_fixed_point (a : ℝ) : a * (-1) + (-1) + a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1769_176965


namespace NUMINAMATH_CALUDE_expression_simplification_l1769_176926

theorem expression_simplification (a : ℝ) (h1 : a ≠ 3) (h2 : a ≠ -3) :
  (1 / (a + 3) + 1 / (a^2 - 9)) / ((a - 2) / (2 * a + 6)) = 2 / (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1769_176926


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_of_3_pow_6_minus_1_l1769_176903

theorem sum_of_prime_factors_of_3_pow_6_minus_1 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range ((3^6 - 1) + 1))) id) = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_of_3_pow_6_minus_1_l1769_176903


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1769_176993

def A : Set ℝ := {x | |x| < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1769_176993


namespace NUMINAMATH_CALUDE_chocolate_gum_pricing_l1769_176931

theorem chocolate_gum_pricing (c g : ℝ) 
  (h : (2 * c > 5 * g ∧ 3 * c ≤ 8 * g) ∨ (2 * c ≤ 5 * g ∧ 3 * c > 8 * g)) :
  7 * c < 19 * g := by
  sorry

end NUMINAMATH_CALUDE_chocolate_gum_pricing_l1769_176931


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1769_176932

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2 ∨ a = 5) (h2 : b = 2 ∨ b = 5) (h3 : a ≠ b) :
  ∃ (c : ℝ), c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1769_176932


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1769_176949

theorem sum_of_reciprocals (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = 2) :
  x + y = 4/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1769_176949


namespace NUMINAMATH_CALUDE_fraction_equality_l1769_176928

theorem fraction_equality (x : ℚ) : (4 + x) / (6 + x) = (2 + x) / (3 + x) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1769_176928


namespace NUMINAMATH_CALUDE_implicit_derivative_l1769_176902

noncomputable section

open Real

-- Define the implicit function
def F (x y : ℝ) : ℝ := log (sqrt (x^2 + y^2)) - arctan (y / x)

-- State the theorem
theorem implicit_derivative (x y : ℝ) (h1 : x ≠ 0) (h2 : x ≠ y) :
  let y' := (x + y) / (x - y)
  (∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → 
    |F (x + h) (y + y' * h) - F x y| ≤ ε * |h|) :=
sorry

end

end NUMINAMATH_CALUDE_implicit_derivative_l1769_176902


namespace NUMINAMATH_CALUDE_symmetric_point_l1769_176923

/-- The line of symmetry -/
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

/-- Check if two points are symmetric with respect to a line -/
def is_symmetric (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- The line connecting the points is perpendicular to the line of symmetry
  (y₂ - y₁) / (x₂ - x₁) = -1 ∧
  -- The midpoint of the two points lies on the line of symmetry
  line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

/-- Theorem: The point (5, -4) is symmetric to (-3, 4) with respect to the line x-y-1=0 -/
theorem symmetric_point : is_symmetric (-3) 4 5 (-4) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_l1769_176923


namespace NUMINAMATH_CALUDE_janet_total_earnings_l1769_176987

/-- Calculates Janet's total earnings from exterminator work and sculpture sales -/
def janet_earnings (exterminator_rate : ℕ) (sculpture_rate : ℕ) (hours_worked : ℕ) (sculpture1_weight : ℕ) (sculpture2_weight : ℕ) : ℕ :=
  exterminator_rate * hours_worked + sculpture_rate * (sculpture1_weight + sculpture2_weight)

theorem janet_total_earnings :
  janet_earnings 70 20 20 5 7 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_earnings_l1769_176987


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1769_176952

/-- The surface area of a cylinder with height 4 and base circumference 2π is 10π. -/
theorem cylinder_surface_area :
  ∀ (h r : ℝ), h = 4 ∧ 2 * π * r = 2 * π →
  2 * π * r * h + 2 * π * r^2 = 10 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1769_176952


namespace NUMINAMATH_CALUDE_supermarket_theorem_l1769_176983

/-- Represents the supermarket's agricultural product distribution problem -/
structure SupermarketProblem where
  total_boxes : ℕ
  brand_a_cost : ℝ
  brand_a_price : ℝ
  brand_b_cost : ℝ
  brand_b_price : ℝ
  total_expenditure : ℝ
  min_total_profit : ℝ

/-- Theorem for the supermarket problem -/
theorem supermarket_theorem (p : SupermarketProblem)
  (h_total : p.total_boxes = 100)
  (h_a_cost : p.brand_a_cost = 80)
  (h_a_price : p.brand_a_price = 120)
  (h_b_cost : p.brand_b_cost = 130)
  (h_b_price : p.brand_b_price = 200)
  (h_expenditure : p.total_expenditure = 10000)
  (h_min_profit : p.min_total_profit = 5600) :
  (∃ (x y : ℕ), x + y = p.total_boxes ∧ 
    p.brand_a_cost * x + p.brand_b_cost * y = p.total_expenditure ∧
    x = 60 ∧ y = 40) ∧
  (∃ (z : ℕ), z ≥ 54 ∧
    (p.brand_a_price - p.brand_a_cost) * (p.total_boxes - z) +
    (p.brand_b_price - p.brand_b_cost) * z ≥ p.min_total_profit) :=
by sorry


end NUMINAMATH_CALUDE_supermarket_theorem_l1769_176983


namespace NUMINAMATH_CALUDE_goldfish_equal_at_11_months_l1769_176956

/-- The number of months it takes for Brent and Gretel to have the same number of goldfish -/
def months_until_equal : ℕ := 11

/-- Brent's initial number of goldfish -/
def brent_initial : ℕ := 6

/-- Gretel's initial number of goldfish -/
def gretel_initial : ℕ := 150

/-- Brent's goldfish growth rate per month -/
def brent_growth_rate : ℝ := 2

/-- Gretel's goldfish growth rate per month -/
def gretel_growth_rate : ℝ := 1.5

/-- Brent's number of goldfish after n months -/
def brent_goldfish (n : ℕ) : ℝ := brent_initial * brent_growth_rate ^ n

/-- Gretel's number of goldfish after n months -/
def gretel_goldfish (n : ℕ) : ℝ := gretel_initial * gretel_growth_rate ^ n

theorem goldfish_equal_at_11_months :
  brent_goldfish months_until_equal = gretel_goldfish months_until_equal :=
sorry

end NUMINAMATH_CALUDE_goldfish_equal_at_11_months_l1769_176956


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1769_176962

-- System 1
theorem system_one_solution (x : ℝ) : 
  (2 * x > 1 - x ∧ x + 2 < 4 * x - 1) ↔ x > 1 :=
sorry

-- System 2
theorem system_two_solution (x : ℝ) : 
  ((2 / 3) * x + 5 > 1 - x ∧ x - 1 ≤ (3 / 4) * x - (1 / 8)) ↔ 
  (-12 / 5 < x ∧ x ≤ 7 / 2) :=
sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1769_176962


namespace NUMINAMATH_CALUDE_deposit_calculation_l1769_176924

/-- Calculates the deposit amount given an initial amount -/
def calculateDeposit (initialAmount : ℚ) : ℚ :=
  initialAmount * (30 / 100) * (25 / 100) * (20 / 100)

/-- Proves that the deposit calculation for Rs. 50,000 results in Rs. 750 -/
theorem deposit_calculation :
  calculateDeposit 50000 = 750 := by
  sorry

end NUMINAMATH_CALUDE_deposit_calculation_l1769_176924


namespace NUMINAMATH_CALUDE_roses_equation_initial_roses_count_l1769_176937

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses Jessica added to the vase -/
def added_roses : ℕ := 16

/-- The final number of roses in the vase -/
def final_roses : ℕ := 23

/-- Theorem stating that the initial number of roses plus the added roses equals the final number of roses -/
theorem roses_equation : initial_roses + added_roses = final_roses := by sorry

/-- Theorem proving that the initial number of roses is 7 -/
theorem initial_roses_count : initial_roses = 7 := by sorry

end NUMINAMATH_CALUDE_roses_equation_initial_roses_count_l1769_176937


namespace NUMINAMATH_CALUDE_transform_sine_to_cosine_l1769_176981

/-- Given a function f(x) = √3 * sin(2x), prove that translating it right by π/4 
    and then compressing its x-coordinates by half results in g(x) = -√3 * cos(4x) -/
theorem transform_sine_to_cosine (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sqrt 3 * Real.sin (2 * x)
  let g : ℝ → ℝ := λ x => -Real.sqrt 3 * Real.cos (4 * x)
  let h : ℝ → ℝ := λ x => f (x / 2 + π / 4)
  h x = g x := by
  sorry

end NUMINAMATH_CALUDE_transform_sine_to_cosine_l1769_176981


namespace NUMINAMATH_CALUDE_sector_central_angle_l1769_176918

theorem sector_central_angle (arc_length : Real) (radius : Real) (central_angle : Real) :
  arc_length = 4 * Real.pi ∧ radius = 8 →
  arc_length = (central_angle * Real.pi * radius) / 180 →
  central_angle = 90 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1769_176918


namespace NUMINAMATH_CALUDE_keaton_yearly_earnings_l1769_176969

/-- Represents Keaton's farm earnings -/
def farm_earnings (orange_harvest_interval : ℕ) (orange_price : ℕ) (apple_harvest_interval : ℕ) (apple_price : ℕ) : ℕ :=
  let months_in_year := 12
  let orange_harvests := months_in_year / orange_harvest_interval
  let apple_harvests := months_in_year / apple_harvest_interval
  orange_harvests * orange_price + apple_harvests * apple_price

/-- Keaton's yearly earnings from his farm of oranges and apples -/
theorem keaton_yearly_earnings :
  farm_earnings 2 50 3 30 = 420 :=
by sorry

end NUMINAMATH_CALUDE_keaton_yearly_earnings_l1769_176969


namespace NUMINAMATH_CALUDE_gcd_factorial_8_and_factorial_11_times_9_squared_l1769_176925

theorem gcd_factorial_8_and_factorial_11_times_9_squared :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 11 * 9^2) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_and_factorial_11_times_9_squared_l1769_176925


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1769_176910

def A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1769_176910


namespace NUMINAMATH_CALUDE_initial_machines_count_l1769_176953

/-- The number of machines in the initial group -/
def initial_machines : ℕ := 15

/-- The number of bags produced per minute by the initial group -/
def initial_production_rate : ℕ := 45

/-- The number of machines in the larger group -/
def larger_group_machines : ℕ := 150

/-- The number of bags produced by the larger group -/
def larger_group_production : ℕ := 3600

/-- The time taken by the larger group to produce the bags (in minutes) -/
def production_time : ℕ := 8

theorem initial_machines_count :
  initial_machines = 15 ∧
  initial_production_rate = 45 ∧
  larger_group_machines = 150 ∧
  larger_group_production = 3600 ∧
  production_time = 8 →
  initial_machines * larger_group_production = initial_production_rate * larger_group_machines * production_time :=
by sorry

end NUMINAMATH_CALUDE_initial_machines_count_l1769_176953


namespace NUMINAMATH_CALUDE_angie_salary_is_80_l1769_176996

/-- Represents Angie's monthly finances -/
structure MonthlyFinances where
  necessities : ℕ
  taxes : ℕ
  leftover : ℕ

/-- Calculates the monthly salary based on expenses and leftover amount -/
def calculate_salary (finances : MonthlyFinances) : ℕ :=
  finances.necessities + finances.taxes + finances.leftover

/-- Theorem stating that Angie's monthly salary is $80 -/
theorem angie_salary_is_80 (angie : MonthlyFinances) 
  (h1 : angie.necessities = 42)
  (h2 : angie.taxes = 20)
  (h3 : angie.leftover = 18) :
  calculate_salary angie = 80 := by
  sorry

#eval calculate_salary { necessities := 42, taxes := 20, leftover := 18 }

end NUMINAMATH_CALUDE_angie_salary_is_80_l1769_176996


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l1769_176946

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  (b - 2*a) * Real.cos C + c * Real.cos B = 0 →
  c = 2 →
  S = Real.sqrt 3 →
  S = 1/2 * a * b * Real.sin C →
  a^2 + b^2 - c^2 = 2*a*b * Real.cos C →
  C = π/3 ∧ a = 2 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l1769_176946


namespace NUMINAMATH_CALUDE_secondary_spermatocyte_may_have_two_y_l1769_176927

-- Define the different stages of cell division
inductive CellDivisionStage
  | PrimarySpermatocyte
  | SecondarySpermatocyte
  | SpermatogoniumMitosis
  | SpermatogoniumMeiosis

-- Define the possible Y chromosome counts
inductive YChromosomeCount
  | Zero
  | One
  | Two

-- Define a function that returns the possible Y chromosome counts for each stage
def possibleYChromosomeCounts (stage : CellDivisionStage) : Set YChromosomeCount :=
  match stage with
  | CellDivisionStage.PrimarySpermatocyte => {YChromosomeCount.One}
  | CellDivisionStage.SecondarySpermatocyte => {YChromosomeCount.Zero, YChromosomeCount.One, YChromosomeCount.Two}
  | CellDivisionStage.SpermatogoniumMitosis => {YChromosomeCount.One}
  | CellDivisionStage.SpermatogoniumMeiosis => {YChromosomeCount.One}

-- Theorem stating that secondary spermatocytes may contain two Y chromosomes
theorem secondary_spermatocyte_may_have_two_y :
  YChromosomeCount.Two ∈ possibleYChromosomeCounts CellDivisionStage.SecondarySpermatocyte :=
by sorry

end NUMINAMATH_CALUDE_secondary_spermatocyte_may_have_two_y_l1769_176927


namespace NUMINAMATH_CALUDE_nearly_regular_polyhedra_theorem_l1769_176942

/-- A structure representing a polyhedron -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Definition of a nearly regular polyhedron -/
def NearlyRegularPolyhedron (p : Polyhedron) : Prop := sorry

/-- Intersection of two polyhedra -/
def intersect (p1 p2 : Polyhedron) : Polyhedron := sorry

/-- Tetrahedron -/
def Tetrahedron : Polyhedron := ⟨4, 6, 4⟩

/-- Octahedron -/
def Octahedron : Polyhedron := ⟨8, 12, 6⟩

/-- Cube -/
def Cube : Polyhedron := ⟨6, 12, 8⟩

/-- Dodecahedron -/
def Dodecahedron : Polyhedron := ⟨12, 30, 20⟩

/-- Icosahedron -/
def Icosahedron : Polyhedron := ⟨20, 30, 12⟩

/-- The set of nearly regular polyhedra -/
def NearlyRegularPolyhedra : Set Polyhedron := sorry

theorem nearly_regular_polyhedra_theorem :
  ∃ (p1 p2 p3 p4 p5 : Polyhedron),
    p1 ∈ NearlyRegularPolyhedra ∧
    p2 ∈ NearlyRegularPolyhedra ∧
    p3 ∈ NearlyRegularPolyhedra ∧
    p4 ∈ NearlyRegularPolyhedra ∧
    p5 ∈ NearlyRegularPolyhedra ∧
    p1 = intersect Tetrahedron Octahedron ∧
    p2 = intersect Cube Octahedron ∧
    p3 = intersect Dodecahedron Icosahedron ∧
    NearlyRegularPolyhedron p4 ∧
    NearlyRegularPolyhedron p5 :=
  sorry

end NUMINAMATH_CALUDE_nearly_regular_polyhedra_theorem_l1769_176942


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l1769_176972

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 35 years older than his son and the son's present age is 33. -/
theorem man_son_age_ratio :
  let son_age : ℕ := 33
  let man_age : ℕ := son_age + 35
  let son_age_in_two_years : ℕ := son_age + 2
  let man_age_in_two_years : ℕ := man_age + 2
  man_age_in_two_years = 2 * son_age_in_two_years := by
  sorry

#check man_son_age_ratio

end NUMINAMATH_CALUDE_man_son_age_ratio_l1769_176972


namespace NUMINAMATH_CALUDE_five_people_arrangement_with_restriction_l1769_176917

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where one specific person cannot be first or last -/
def arrangementsWithRestriction (n : ℕ) : ℕ :=
  (n - 2) * Nat.factorial (n - 1)

/-- Theorem: There are 72 ways to arrange 5 people in a line where one specific person cannot be first or last -/
theorem five_people_arrangement_with_restriction :
  arrangementsWithRestriction 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_people_arrangement_with_restriction_l1769_176917


namespace NUMINAMATH_CALUDE_min_jam_prob_route_l1769_176929

structure Route where
  segments : List (Char × Char)

def no_jam_prob (r : Route) (probs : List ℚ) : ℚ :=
  probs.prod

def jam_prob (r : Route) (probs : List ℚ) : ℚ :=
  1 - no_jam_prob r probs

theorem min_jam_prob_route (route1 route2 route3 : Route)
  (probs1 probs2 probs3 : List ℚ) :
  route1.segments = [('A', 'C'), ('C', 'D'), ('D', 'B')] →
  route2.segments = [('A', 'C'), ('C', 'F'), ('F', 'B')] →
  route3.segments = [('A', 'E'), ('E', 'F'), ('F', 'B')] →
  probs1 = [9/10, 14/15, 5/6] →
  probs2 = [9/10, 9/10, 15/16] →
  probs3 = [9/10, 9/10, 19/20] →
  jam_prob route1 probs1 < jam_prob route2 probs2 ∧
  jam_prob route1 probs1 < jam_prob route3 probs3 :=
by sorry

end NUMINAMATH_CALUDE_min_jam_prob_route_l1769_176929


namespace NUMINAMATH_CALUDE_outfit_count_l1769_176984

/-- The number of different outfits with different colored shirt and hat -/
def number_of_outfits (blue_shirts green_shirts pants blue_hats green_hats : ℕ) : ℕ :=
  (blue_shirts * green_hats * pants) + (green_shirts * blue_hats * pants)

/-- Theorem stating the number of outfits given the specific quantities -/
theorem outfit_count :
  number_of_outfits 7 6 7 10 9 = 861 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l1769_176984


namespace NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l1769_176920

-- Define a triangle
structure Triangle where
  a : ℝ  -- angle a
  b : ℝ  -- angle b
  c : ℝ  -- angle c
  sum_180 : a + b + c = 180  -- sum of angles in a triangle is 180 degrees

-- Define what it means for two angles to be complementary
def complementary (x y : ℝ) : Prop := x + y = 90

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.a = 90 ∨ t.b = 90 ∨ t.c = 90

-- Theorem statement
theorem complementary_angles_imply_right_triangle (t : Triangle) :
  (complementary t.a t.b ∨ complementary t.b t.c ∨ complementary t.a t.c) →
  is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l1769_176920


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1769_176986

theorem decimal_to_fraction :
  (0.36 : ℚ) = 9 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1769_176986
