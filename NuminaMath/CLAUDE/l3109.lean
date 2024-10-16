import Mathlib

namespace NUMINAMATH_CALUDE_carson_gardening_time_l3109_310961

/-- Represents the gardening tasks Carson needs to complete -/
structure GardeningTasks where
  mow_lines : ℕ
  mow_time_per_line : ℕ
  flower_rows : ℕ
  flowers_per_row : ℕ
  planting_time_per_flower : ℚ
  garden_sections : ℕ
  watering_time_per_section : ℕ
  hedges : ℕ
  trimming_time_per_hedge : ℕ

/-- Calculates the total gardening time in minutes -/
def total_gardening_time (tasks : GardeningTasks) : ℚ :=
  tasks.mow_lines * tasks.mow_time_per_line +
  tasks.flower_rows * tasks.flowers_per_row * tasks.planting_time_per_flower +
  tasks.garden_sections * tasks.watering_time_per_section +
  tasks.hedges * tasks.trimming_time_per_hedge

/-- Theorem stating that Carson's total gardening time is 162 minutes -/
theorem carson_gardening_time :
  let tasks : GardeningTasks := {
    mow_lines := 40,
    mow_time_per_line := 2,
    flower_rows := 10,
    flowers_per_row := 8,
    planting_time_per_flower := 1/2,
    garden_sections := 4,
    watering_time_per_section := 3,
    hedges := 5,
    trimming_time_per_hedge := 6
  }
  total_gardening_time tasks = 162 := by
  sorry


end NUMINAMATH_CALUDE_carson_gardening_time_l3109_310961


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_16_l3109_310997

theorem arithmetic_sqrt_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_16_l3109_310997


namespace NUMINAMATH_CALUDE_path_area_and_cost_l3109_310928

/-- Calculates the area of a rectangular path surrounding a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem path_area_and_cost (field_length field_width path_width cost_per_unit : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 40)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 600 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1200 := by
  sorry

#eval path_area 75 40 2.5
#eval construction_cost (path_area 75 40 2.5) 2

end NUMINAMATH_CALUDE_path_area_and_cost_l3109_310928


namespace NUMINAMATH_CALUDE_mixed_decimal_to_vulgar_fraction_l3109_310935

theorem mixed_decimal_to_vulgar_fraction :
  (4 + 13 / 50 : ℚ) = 4.26 ∧
  (1 + 3 / 20 : ℚ) = 1.15 ∧
  (3 + 2 / 25 : ℚ) = 3.08 ∧
  (2 + 37 / 100 : ℚ) = 2.37 := by
  sorry

end NUMINAMATH_CALUDE_mixed_decimal_to_vulgar_fraction_l3109_310935


namespace NUMINAMATH_CALUDE_thirty_percent_of_hundred_l3109_310955

theorem thirty_percent_of_hundred : (30 : ℝ) = 100 * (30 / 100) := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_hundred_l3109_310955


namespace NUMINAMATH_CALUDE_min_value_of_max_expression_l3109_310971

theorem min_value_of_max_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let M := max (x * y + 2 / z) (max (z + 2 / y) (y + z + 1 / x))
  M ≥ 3 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    max (x' * y' + 2 / z') (max (z' + 2 / y') (y' + z' + 1 / x')) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_max_expression_l3109_310971


namespace NUMINAMATH_CALUDE_tunnel_length_tunnel_length_proof_l3109_310930

/-- Calculates the length of a tunnel given train and time information -/
theorem tunnel_length (train_length : ℝ) (train_speed : ℝ) (exit_time : ℝ) : ℝ :=
  let tunnel_length := train_speed * exit_time / 60 - train_length
  2

theorem tunnel_length_proof :
  tunnel_length 1 60 3 = 2 := by sorry

end NUMINAMATH_CALUDE_tunnel_length_tunnel_length_proof_l3109_310930


namespace NUMINAMATH_CALUDE_square_area_3_square_area_3_proof_l3109_310996

/-- The area of a square with side length 3 is 9 -/
theorem square_area_3 : Real → Prop :=
  fun area =>
    let side_length : Real := 3
    area = side_length ^ 2

#check square_area_3 9

/-- Proof of the theorem -/
theorem square_area_3_proof : square_area_3 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_3_square_area_3_proof_l3109_310996


namespace NUMINAMATH_CALUDE_hexagon_minus_triangle_area_l3109_310949

/-- The area of a hexagon with side length 2 and height 4, minus the area of an inscribed equilateral triangle with side length 4 -/
theorem hexagon_minus_triangle_area : 
  let hexagon_side : ℝ := 2
  let hexagon_height : ℝ := 4
  let triangle_side : ℝ := 4
  let hexagon_area : ℝ := 6 * (1/2 * hexagon_side * hexagon_height)
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  hexagon_area - triangle_area = 24 - 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_minus_triangle_area_l3109_310949


namespace NUMINAMATH_CALUDE_dust_retention_proof_l3109_310910

/-- The average dust retention of a ginkgo leaf in milligrams per year. -/
def ginkgo_retention : ℝ := 40

/-- The average dust retention of a locust leaf in milligrams per year. -/
def locust_retention : ℝ := 22

/-- The number of ginkgo leaves. -/
def num_ginkgo_leaves : ℕ := 50000

theorem dust_retention_proof :
  -- Condition 1: Ginkgo retention is 4mg less than twice locust retention
  ginkgo_retention = 2 * locust_retention - 4 ∧
  -- Condition 2: Total retention of ginkgo and locust is 62mg
  ginkgo_retention + locust_retention = 62 ∧
  -- Result 1: Ginkgo retention is 40mg
  ginkgo_retention = 40 ∧
  -- Result 2: Locust retention is 22mg
  locust_retention = 22 ∧
  -- Result 3: Total retention of 50,000 ginkgo leaves is 2kg
  (ginkgo_retention * num_ginkgo_leaves) / 1000000 = 2 :=
by sorry

end NUMINAMATH_CALUDE_dust_retention_proof_l3109_310910


namespace NUMINAMATH_CALUDE_position_of_2018_l3109_310902

def digit_sum (n : ℕ) : ℕ := sorry

def ascending_seq_digit_sum_11 : List ℕ := sorry

theorem position_of_2018 : 
  ascending_seq_digit_sum_11.indexOf 2018 = 133 := by sorry

end NUMINAMATH_CALUDE_position_of_2018_l3109_310902


namespace NUMINAMATH_CALUDE_shares_multiple_l3109_310937

/-- Represents the shares of money for three children -/
structure Shares where
  anusha : ℕ
  babu : ℕ
  esha : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem shares_multiple (s : Shares) 
  (h1 : 12 * s.anusha = 8 * s.babu)
  (h2 : ∃ k : ℕ, 8 * s.babu = k * s.esha)
  (h3 : s.anusha + s.babu + s.esha = 378)
  (h4 : s.anusha = 84) :
  ∃ k : ℕ, 8 * s.babu = 6 * s.esha := by
  sorry


end NUMINAMATH_CALUDE_shares_multiple_l3109_310937


namespace NUMINAMATH_CALUDE_glen_village_impossibility_l3109_310929

theorem glen_village_impossibility : ¬ ∃ (h c : ℕ), 21 * h + 6 * c = 96 := by
  sorry

#check glen_village_impossibility

end NUMINAMATH_CALUDE_glen_village_impossibility_l3109_310929


namespace NUMINAMATH_CALUDE_shiela_painting_distribution_l3109_310957

/-- Given Shiela has 18 paintings and 2 grandmothers, prove that each grandmother
    receives 9 paintings when the paintings are distributed equally. -/
theorem shiela_painting_distribution
  (total_paintings : ℕ)
  (num_grandmothers : ℕ)
  (h1 : total_paintings = 18)
  (h2 : num_grandmothers = 2)
  : total_paintings / num_grandmothers = 9 := by
  sorry

end NUMINAMATH_CALUDE_shiela_painting_distribution_l3109_310957


namespace NUMINAMATH_CALUDE_total_toy_cost_l3109_310903

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

theorem total_toy_cost : football_cost + marbles_cost = 12.30 := by
  sorry

end NUMINAMATH_CALUDE_total_toy_cost_l3109_310903


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3109_310924

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 5 → x > 3) ∧ 
  (∃ x : ℝ, x > 3 ∧ ¬(x > 5)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3109_310924


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l3109_310917

theorem loan_principal_calculation (principal : ℝ) : 
  (principal * 0.08 * 10 = principal - 1540) → principal = 7700 := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l3109_310917


namespace NUMINAMATH_CALUDE_abs_difference_over_sum_equals_sqrt_three_sevenths_l3109_310905

theorem abs_difference_over_sum_equals_sqrt_three_sevenths
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 5*a*b) :
  |((a - b) / (a + b))| = Real.sqrt (3/7) :=
by sorry

end NUMINAMATH_CALUDE_abs_difference_over_sum_equals_sqrt_three_sevenths_l3109_310905


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3109_310953

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) : 
  let sphere_area := 4 * π * r^2
  let base_area := π * r^2
  let excluded_base_area := (1/4) * base_area
  let hemisphere_curved_area := (1/2) * sphere_area
  hemisphere_curved_area + base_area - excluded_base_area = 275 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3109_310953


namespace NUMINAMATH_CALUDE_pool_filling_time_l3109_310946

/-- Proves the time required to fill a pool given its volume and water delivery rates -/
theorem pool_filling_time 
  (pool_volume : ℝ) 
  (hose1_rate : ℝ) 
  (hose2_rate : ℝ) 
  (hose1_count : ℕ) 
  (hose2_count : ℕ) 
  (h1 : pool_volume = 15000) 
  (h2 : hose1_rate = 2) 
  (h3 : hose2_rate = 3) 
  (h4 : hose1_count = 2) 
  (h5 : hose2_count = 2) : 
  (pool_volume / (hose1_count * hose1_rate + hose2_count * hose2_rate)) / 60 = 25 := by
  sorry

#check pool_filling_time

end NUMINAMATH_CALUDE_pool_filling_time_l3109_310946


namespace NUMINAMATH_CALUDE_minimum_economic_loss_l3109_310918

def repair_times : List ℕ := [12, 17, 8, 18, 23, 30, 14]
def num_workers : ℕ := 3
def loss_per_minute : ℕ := 2

def distribute_work (times : List ℕ) (workers : ℕ) : List ℕ :=
  sorry

def calculate_waiting_time (distribution : List ℕ) : ℕ :=
  sorry

def economic_loss (waiting_time : ℕ) (loss_per_minute : ℕ) : ℕ :=
  sorry

theorem minimum_economic_loss :
  economic_loss (calculate_waiting_time (distribute_work repair_times num_workers)) loss_per_minute = 364 := by
  sorry

end NUMINAMATH_CALUDE_minimum_economic_loss_l3109_310918


namespace NUMINAMATH_CALUDE_cos_105_degrees_l3109_310945

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l3109_310945


namespace NUMINAMATH_CALUDE_power_of_three_equality_l3109_310981

theorem power_of_three_equality (n : ℕ) : 3^n = 27 * 9^2 * (81^3) / 3^4 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l3109_310981


namespace NUMINAMATH_CALUDE_possible_m_values_l3109_310919

theorem possible_m_values (m : ℝ) : 
  (2 ∈ ({m - 1, 2 * m, m^2 - 1} : Set ℝ)) → 
  (m ∈ ({3, Real.sqrt 3, -Real.sqrt 3} : Set ℝ)) := by
sorry

end NUMINAMATH_CALUDE_possible_m_values_l3109_310919


namespace NUMINAMATH_CALUDE_find_a_l3109_310932

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}

-- Define set P
def P (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

-- Define the complement of P with respect to U
def complement_P (a : ℝ) : Set ℝ := {-1}

-- Theorem statement
theorem find_a : ∃ a : ℝ, (U a = P a ∪ complement_P a) ∧ (a = 2) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3109_310932


namespace NUMINAMATH_CALUDE_gcd_5_factorial_7_factorial_l3109_310976

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_5_factorial_7_factorial : 
  Nat.gcd (factorial 5) (factorial 7) = factorial 5 := by sorry

end NUMINAMATH_CALUDE_gcd_5_factorial_7_factorial_l3109_310976


namespace NUMINAMATH_CALUDE_cubic_equation_coefficient_sum_of_squares_l3109_310993

theorem cubic_equation_coefficient_sum_of_squares :
  ∀ (p q r s t u : ℤ),
  (∀ x, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_coefficient_sum_of_squares_l3109_310993


namespace NUMINAMATH_CALUDE_quadratic_equation_real_solutions_l3109_310970

theorem quadratic_equation_real_solutions (x y z : ℝ) :
  (∃ z, 16 * z^2 + 4 * x * y * z + (y^2 - 3) = 0) ↔ x ≤ -2 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_solutions_l3109_310970


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3109_310941

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3109_310941


namespace NUMINAMATH_CALUDE_linear_equation_exponent_values_l3109_310922

theorem linear_equation_exponent_values (m n : ℤ) : 
  (∀ x y : ℝ, ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ a * x + b * y + c = 5 * x^(3*m-2*n) - 2 * y^(n-m) + 11) →
  m = 0 ∧ n = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_values_l3109_310922


namespace NUMINAMATH_CALUDE_remainder_after_adding_150_l3109_310900

theorem remainder_after_adding_150 (n : ℤ) :
  n % 6 = 1 → (n + 150) % 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_after_adding_150_l3109_310900


namespace NUMINAMATH_CALUDE_pears_for_strawberries_l3109_310950

-- Define variables for each type of fruit
variable (s r c b p : ℚ)

-- Define exchange rates
def exchange1 : Prop := 11 * s = 14 * r
def exchange2 : Prop := 22 * c = 21 * r
def exchange3 : Prop := 10 * c = 3 * b
def exchange4 : Prop := 5 * p = 2 * b

-- Theorem to prove
theorem pears_for_strawberries 
  (h1 : exchange1 s r)
  (h2 : exchange2 c r)
  (h3 : exchange3 c b)
  (h4 : exchange4 p b) :
  7 * s = 7 * p :=
sorry

end NUMINAMATH_CALUDE_pears_for_strawberries_l3109_310950


namespace NUMINAMATH_CALUDE_final_number_is_two_thirds_l3109_310925

def board_numbers : List ℚ := List.map (λ k => k / 2016) (List.range 2016)

def transform (a b : ℚ) : ℚ := 3 * a * b - 2 * a - 2 * b + 2

theorem final_number_is_two_thirds :
  ∃ (moves : List (ℚ × ℚ)),
    moves.length = 2015 ∧
    (moves.foldl
      (λ board (a, b) => (transform a b) :: (board.filter (λ x => x ≠ a ∧ x ≠ b)))
      board_numbers).head? = some (2/3) :=
sorry

end NUMINAMATH_CALUDE_final_number_is_two_thirds_l3109_310925


namespace NUMINAMATH_CALUDE_bulls_win_in_seven_games_l3109_310938

def probability_heat_wins : ℚ := 3/4

def games_to_win : ℕ := 4

def total_games : ℕ := 7

theorem bulls_win_in_seven_games :
  let probability_bulls_wins := 1 - probability_heat_wins
  let combinations := Nat.choose (total_games - 1) (games_to_win - 1)
  let probability_tied_after_six := combinations * (probability_bulls_wins ^ (games_to_win - 1)) * (probability_heat_wins ^ (games_to_win - 1))
  let probability_bulls_win_last := probability_bulls_wins
  probability_tied_after_six * probability_bulls_win_last = 540/16384 := by
  sorry

end NUMINAMATH_CALUDE_bulls_win_in_seven_games_l3109_310938


namespace NUMINAMATH_CALUDE_exists_number_with_properties_l3109_310931

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_number_with_properties : ∃ n : ℕ, 
  2019 ∣ n ∧ 2019 ∣ sum_of_digits n := by sorry

end NUMINAMATH_CALUDE_exists_number_with_properties_l3109_310931


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l3109_310990

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (height : ℝ) 
  (stripe_width : ℝ) 
  (h_diameter : diameter = 30) 
  (h_height : height = 80) 
  (h_stripe_width : stripe_width = 3) :
  stripe_width * height = 240 := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l3109_310990


namespace NUMINAMATH_CALUDE_first_candidate_percentage_is_70_percent_l3109_310909

/-- The percentage of votes the first candidate received in an election with two candidates -/
def first_candidate_percentage (total_votes : ℕ) (second_candidate_votes : ℕ) : ℚ :=
  (total_votes - second_candidate_votes : ℚ) / total_votes * 100

/-- Theorem stating that the first candidate received 70% of the votes -/
theorem first_candidate_percentage_is_70_percent :
  first_candidate_percentage 800 240 = 70 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_percentage_is_70_percent_l3109_310909


namespace NUMINAMATH_CALUDE_alcohol_fraction_in_mixture_l3109_310927

theorem alcohol_fraction_in_mixture (alcohol_water_ratio : ℚ) :
  alcohol_water_ratio = 2 / 3 →
  (alcohol_water_ratio / (1 + alcohol_water_ratio)) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_fraction_in_mixture_l3109_310927


namespace NUMINAMATH_CALUDE_B_when_a_is_3_range_of_a_when_A_equals_B_l3109_310992

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | (a - 2) * x^2 + 2 * (a - 2) * x - 3 < 0}

-- Theorem 1: When a = 3, B = (-3, 1)
theorem B_when_a_is_3 : B 3 = Set.Ioo (-3) 1 := by sorry

-- Theorem 2: When A = B = ℝ, a ∈ (-1, 2]
theorem range_of_a_when_A_equals_B :
  (∀ x, x ∈ B a) ↔ a ∈ Set.Ioc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_B_when_a_is_3_range_of_a_when_A_equals_B_l3109_310992


namespace NUMINAMATH_CALUDE_peanut_butter_cost_l3109_310921

/-- The cost of the jar of peanut butter given the cost of bread, initial money, and money left over -/
theorem peanut_butter_cost
  (bread_cost : ℝ)
  (bread_quantity : ℕ)
  (initial_money : ℝ)
  (money_left : ℝ)
  (h1 : bread_cost = 2.25)
  (h2 : bread_quantity = 3)
  (h3 : initial_money = 14)
  (h4 : money_left = 5.25) :
  initial_money - money_left - bread_cost * bread_quantity = 2 :=
by sorry

end NUMINAMATH_CALUDE_peanut_butter_cost_l3109_310921


namespace NUMINAMATH_CALUDE_fibonacci_pythagorean_hypotenuse_l3109_310934

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_pythagorean_hypotenuse (k : ℕ) (h : k ≥ 2) :
  fibonacci (2 * k + 1) = fibonacci k ^ 2 + fibonacci (k + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_pythagorean_hypotenuse_l3109_310934


namespace NUMINAMATH_CALUDE_product_remainder_l3109_310978

theorem product_remainder (a b c d e : ℕ) (h1 : a = 12457) (h2 : b = 12463) (h3 : c = 12469) (h4 : d = 12473) (h5 : e = 12479) :
  (a * b * c * d * e) % 18 = 3 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_l3109_310978


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3109_310926

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

def complement_N (S : Set ℕ) : Set ℕ := {n : ℕ | n ∉ S}

theorem intersection_complement_theorem :
  A ∩ (complement_N B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3109_310926


namespace NUMINAMATH_CALUDE_circle_c_equation_l3109_310956

/-- A circle C with center on the line 2x-y-7=0 and intersecting the y-axis at points (0, -4) and (0, -2) -/
structure CircleC where
  /-- The center of the circle lies on the line 2x-y-7=0 -/
  center_on_line : ∀ (x y : ℝ), y = 2*x - 7 → (∃ (t : ℝ), x = t ∧ y = 2*t - 7)
  /-- The circle intersects the y-axis at points (0, -4) and (0, -2) -/
  intersects_y_axis : ∃ (r : ℝ), r > 0 ∧ 
    (∃ (cx cy : ℝ), cx^2 + (cy + 4)^2 = r^2 ∧ cx^2 + (cy + 2)^2 = r^2)

/-- The equation of circle C is (x-2)^2+(y+3)^2=5 -/
theorem circle_c_equation (c : CircleC) : 
  ∃ (cx cy : ℝ), (∀ (x y : ℝ), (x - cx)^2 + (y - cy)^2 = 5 ↔ 
    (x - 2)^2 + (y + 3)^2 = 5) :=
sorry

end NUMINAMATH_CALUDE_circle_c_equation_l3109_310956


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3109_310998

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3109_310998


namespace NUMINAMATH_CALUDE_base6_arithmetic_sum_l3109_310975

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ := sorry

/-- Calculates the number of terms in the arithmetic sequence --/
def numTerms (first last step : ℕ) : ℕ := (last - first) / step + 1

/-- Calculates the sum of an arithmetic sequence --/
def arithmeticSum (n first last : ℕ) : ℕ := n * (first + last) / 2

theorem base6_arithmetic_sum :
  let first := base6ToBase10 2
  let last := base6ToBase10 50
  let step := base6ToBase10 2
  let n := numTerms first last step
  let sum := arithmeticSum n first last
  base10ToBase6 sum = 1040 := by sorry

end NUMINAMATH_CALUDE_base6_arithmetic_sum_l3109_310975


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3109_310944

/-- A regular polygon with side length 10 and perimeter 60 has 6 sides -/
theorem regular_polygon_sides (s : ℕ) (side_length perimeter : ℝ) : 
  side_length = 10 → perimeter = 60 → s * side_length = perimeter → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3109_310944


namespace NUMINAMATH_CALUDE_triangle_properties_l3109_310965

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  (t.b - t.c)^2 = t.a^2 - t.b * t.c ∧
  t.a = 2 ∧
  Real.sin t.C = 2 * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π/3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A : ℝ) = (2 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3109_310965


namespace NUMINAMATH_CALUDE_random_sampling_correct_l3109_310987

/-- Represents a random number table row -/
def RandomTableRow := List Nat

/-- Checks if a number is a valid bag number (000-799) -/
def isValidBagNumber (n : Nat) : Bool :=
  n >= 0 && n <= 799

/-- Extracts valid bag numbers from a list of numbers -/
def extractValidBagNumbers (numbers : List Nat) : List Nat :=
  numbers.filter isValidBagNumber

/-- Represents the given random number table row -/
def givenRow : RandomTableRow :=
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79]

/-- The expected result -/
def expectedResult : List Nat := [785, 567, 199, 507, 175]

theorem random_sampling_correct :
  let startIndex := 6  -- 7th column (0-based index)
  let relevantNumbers := givenRow.drop startIndex
  let validBagNumbers := extractValidBagNumbers relevantNumbers
  validBagNumbers.take 5 = expectedResult := by sorry

end NUMINAMATH_CALUDE_random_sampling_correct_l3109_310987


namespace NUMINAMATH_CALUDE_correct_calculation_l3109_310959

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * x^2 * y = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3109_310959


namespace NUMINAMATH_CALUDE_candle_ratio_problem_l3109_310942

/-- Given the ratio of red candles to blue candles and the number of red candles,
    calculate the number of blue candles. -/
theorem candle_ratio_problem (red_candles : ℕ) (red_ratio blue_ratio : ℕ) 
    (h_red : red_candles = 45)
    (h_ratio : red_ratio = 5 ∧ blue_ratio = 3) :
    red_candles * blue_ratio = red_ratio * 27 :=
by sorry

end NUMINAMATH_CALUDE_candle_ratio_problem_l3109_310942


namespace NUMINAMATH_CALUDE_right_triangle_area_l3109_310958

theorem right_triangle_area (p q r : ℝ) : 
  p > 0 → q > 0 → r > 0 →
  p + q + r = 16 →
  p^2 + q^2 + r^2 = 98 →
  p^2 + q^2 = r^2 →
  (1/2) * p * q = 8 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3109_310958


namespace NUMINAMATH_CALUDE_last_part_distance_calculation_l3109_310920

/-- Calculates the distance of the last part of a trip given the total distance,
    first part distance, speeds for different parts, and average speed. -/
def last_part_distance (total_distance first_part_distance first_part_speed
                        average_speed last_part_speed : ℝ) : ℝ :=
  total_distance - first_part_distance

theorem last_part_distance_calculation (total_distance : ℝ) (first_part_distance : ℝ)
    (first_part_speed : ℝ) (average_speed : ℝ) (last_part_speed : ℝ)
    (h1 : total_distance = 100)
    (h2 : first_part_distance = 30)
    (h3 : first_part_speed = 60)
    (h4 : average_speed = 40)
    (h5 : last_part_speed = 35) :
  last_part_distance total_distance first_part_distance first_part_speed average_speed last_part_speed = 70 := by
sorry

end NUMINAMATH_CALUDE_last_part_distance_calculation_l3109_310920


namespace NUMINAMATH_CALUDE_ski_track_problem_l3109_310964

/-- Ski track problem -/
theorem ski_track_problem (a b : ℝ) (h : b / 3 < a ∧ a < 3 * b) :
  let whole_track_speed := 3 * a * b / (2 * (a + b))
  let first_segment_speed := 2 * a * b / (3 * b - a)
  let second_segment_speed := 2 * a * b / (a + b)
  let third_segment_speed := 2 * a * b / (3 * a - b)
  (∃ t₁ t₂ t₃ : ℝ,
    t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0 ∧
    2 / (t₁ + t₂) = a ∧
    2 / (t₂ + t₃) = b ∧
    1 / t₂ = 2 / (t₁ + t₃) ∧
    3 / (t₁ + t₂ + t₃) = whole_track_speed ∧
    1 / t₁ = first_segment_speed ∧
    1 / t₂ = second_segment_speed ∧
    1 / t₃ = third_segment_speed) :=
by
  sorry


end NUMINAMATH_CALUDE_ski_track_problem_l3109_310964


namespace NUMINAMATH_CALUDE_particle_acceleration_l3109_310994

-- Define the displacement function
def s (t : ℝ) : ℝ := t^2 - t + 6

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 2 * t - 1

-- Define the acceleration function as the derivative of velocity
def a (t : ℝ) : ℝ := 2

-- Theorem statement
theorem particle_acceleration (t : ℝ) (h : t ∈ Set.Icc 1 4) :
  a t = 2 := by
  sorry

end NUMINAMATH_CALUDE_particle_acceleration_l3109_310994


namespace NUMINAMATH_CALUDE_no_intersection_l3109_310989

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem stating that there are no intersection points
theorem no_intersection :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ circle_eq x y :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l3109_310989


namespace NUMINAMATH_CALUDE_odd_sum_selections_count_l3109_310947

/-- The number of ways to select k elements from n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The set of numbers from 1 to 11 -/
def ballNumbers : Finset ℕ := sorry

/-- The number of odd numbers in ballNumbers -/
def oddCount : ℕ := sorry

/-- The number of even numbers in ballNumbers -/
def evenCount : ℕ := sorry

/-- The number of ways to select 5 balls with an odd sum -/
def oddSumSelections : ℕ := sorry

theorem odd_sum_selections_count :
  oddSumSelections = 236 := by sorry

end NUMINAMATH_CALUDE_odd_sum_selections_count_l3109_310947


namespace NUMINAMATH_CALUDE_radical_simplification_l3109_310933

theorem radical_simplification (x : ℝ) (h : 4 < x ∧ x < 7) :
  (((x - 4) ^ 4) ^ (1/4 : ℝ)) + (((x - 7) ^ 4) ^ (1/4 : ℝ)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3109_310933


namespace NUMINAMATH_CALUDE_soda_difference_l3109_310954

theorem soda_difference (diet_soda : ℕ) (regular_soda : ℕ) 
  (h1 : diet_soda = 19) (h2 : regular_soda = 60) : 
  regular_soda - diet_soda = 41 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l3109_310954


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3109_310916

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 2 * x + 15 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y - 2 * y + 15 = 0 → y = x) ↔ 
  (m = -2 + 6 * Real.sqrt 5 ∨ m = -2 - 6 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3109_310916


namespace NUMINAMATH_CALUDE_melissa_oranges_l3109_310969

theorem melissa_oranges (initial_oranges : ℕ) (taken_oranges : ℕ) 
  (h1 : initial_oranges = 70) (h2 : taken_oranges = 19) : 
  initial_oranges - taken_oranges = 51 := by
  sorry

end NUMINAMATH_CALUDE_melissa_oranges_l3109_310969


namespace NUMINAMATH_CALUDE_tank_fill_time_main_theorem_l3109_310967

-- Define the fill rates of the pipes
def fill_rate_1 : ℚ := 1 / 18
def fill_rate_2 : ℚ := 1 / 60
def empty_rate : ℚ := 1 / 45

-- Define the combined rate
def combined_rate : ℚ := fill_rate_1 + fill_rate_2 - empty_rate

-- Theorem statement
theorem tank_fill_time :
  combined_rate = 1 / 20 := by sorry

-- Time to fill the tank
def fill_time : ℚ := 1 / combined_rate

-- Main theorem
theorem main_theorem :
  fill_time = 20 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_main_theorem_l3109_310967


namespace NUMINAMATH_CALUDE_father_age_is_33_l3109_310968

/-- The present age of the son -/
def son_age : ℕ := sorry

/-- The present age of the father -/
def father_age : ℕ := sorry

/-- The father's present age is 3 years more than 3 times the son's present age -/
axiom condition1 : father_age = 3 * son_age + 3

/-- In 3 years, the father's age will be 10 years more than twice the son's age -/
axiom condition2 : father_age + 3 = 2 * (son_age + 3) + 10

/-- The present age of the father is 33 years -/
theorem father_age_is_33 : father_age = 33 := by sorry

end NUMINAMATH_CALUDE_father_age_is_33_l3109_310968


namespace NUMINAMATH_CALUDE_sandys_puppies_l3109_310911

/-- Given that Sandy initially had 8 puppies and gave away 4,
    prove that she now has 4 puppies. -/
theorem sandys_puppies :
  let initial_puppies : ℕ := 8
  let puppies_given_away : ℕ := 4
  let remaining_puppies := initial_puppies - puppies_given_away
  remaining_puppies = 4 :=
by sorry

end NUMINAMATH_CALUDE_sandys_puppies_l3109_310911


namespace NUMINAMATH_CALUDE_median_mean_difference_l3109_310915

/-- The distribution of scores on an algebra quiz -/
structure ScoreDistribution where
  score_70 : ℝ
  score_80 : ℝ
  score_90 : ℝ
  score_100 : ℝ

/-- The properties of the score distribution -/
def valid_distribution (d : ScoreDistribution) : Prop :=
  d.score_70 = 0.1 ∧
  d.score_80 = 0.35 ∧
  d.score_90 = 0.3 ∧
  d.score_100 = 0.25 ∧
  d.score_70 + d.score_80 + d.score_90 + d.score_100 = 1

/-- Calculate the mean score -/
def mean_score (d : ScoreDistribution) : ℝ :=
  70 * d.score_70 + 80 * d.score_80 + 90 * d.score_90 + 100 * d.score_100

/-- The median score -/
def median_score : ℝ := 90

/-- The main theorem: the difference between median and mean is 3 -/
theorem median_mean_difference (d : ScoreDistribution) 
  (h : valid_distribution d) : median_score - mean_score d = 3 := by
  sorry

end NUMINAMATH_CALUDE_median_mean_difference_l3109_310915


namespace NUMINAMATH_CALUDE_perpendicular_bisector_theorem_l3109_310995

/-- A structure representing a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A function to construct the perpendicular bisector points A', B', and C' -/
def constructPerpendicularBisectorPoints (t : Triangle) : Triangle :=
  sorry

/-- A predicate to check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- A predicate to check if a triangle has angles 30°, 30°, and 120° -/
def has30_30_120Angles (t : Triangle) : Prop :=
  sorry

/-- The main theorem -/
theorem perpendicular_bisector_theorem (t : Triangle) :
  let t' := constructPerpendicularBisectorPoints t
  isEquilateral t' ↔ (isEquilateral t ∨ has30_30_120Angles t) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_theorem_l3109_310995


namespace NUMINAMATH_CALUDE_star_sqrt_eleven_l3109_310991

-- Define the ¤ operation
def star (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- Theorem statement
theorem star_sqrt_eleven : star (Real.sqrt 11) (Real.sqrt 11) = 44 := by
  sorry

end NUMINAMATH_CALUDE_star_sqrt_eleven_l3109_310991


namespace NUMINAMATH_CALUDE_stock_market_value_l3109_310939

/-- Calculates the market value of a stock given its dividend rate, yield, and face value. -/
def market_value (dividend_rate : ℚ) (yield : ℚ) (face_value : ℚ) : ℚ :=
  (dividend_rate * face_value / yield) * 100

/-- Theorem stating that a 13% stock yielding 8% with a face value of $100 has a market value of $162.50 -/
theorem stock_market_value :
  let dividend_rate : ℚ := 13 / 100
  let yield : ℚ := 8 / 100
  let face_value : ℚ := 100
  market_value dividend_rate yield face_value = 162.5 := by
  sorry

#eval market_value (13/100) (8/100) 100

end NUMINAMATH_CALUDE_stock_market_value_l3109_310939


namespace NUMINAMATH_CALUDE_system_solution_l3109_310999

theorem system_solution :
  ∃! (x y : ℚ), (7 * x = -10 - 3 * y) ∧ (4 * x = 5 * y - 32) ∧
  (x = -219 / 88) ∧ (y = 97 / 22) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3109_310999


namespace NUMINAMATH_CALUDE_binomial_12_11_l3109_310952

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by sorry

end NUMINAMATH_CALUDE_binomial_12_11_l3109_310952


namespace NUMINAMATH_CALUDE_parabola_equation_l3109_310980

/-- A parabola with vertex (h, k) and vertical axis of symmetry has the form y = a(x-h)^2 + k -/
def is_vertical_parabola (a h k : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = a * (x - h)^2 + k

theorem parabola_equation (f : ℝ → ℝ) :
  (∀ x, f x = -3 * x^2 + 18 * x - 22) →
  is_vertical_parabola (-3) 3 5 f ∧
  f 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3109_310980


namespace NUMINAMATH_CALUDE_apartment_length_l3109_310948

/-- Proves that the length of an apartment with given specifications is 16 feet -/
theorem apartment_length : 
  ∀ (width : ℝ) (total_rooms : ℕ) (living_room_size : ℝ),
    width = 10 →
    total_rooms = 6 →
    living_room_size = 60 →
    ∃ (room_size : ℝ),
      room_size = living_room_size / 3 ∧
      width * 16 = living_room_size + (total_rooms - 1) * room_size :=
by sorry

end NUMINAMATH_CALUDE_apartment_length_l3109_310948


namespace NUMINAMATH_CALUDE_unique_pizza_combinations_l3109_310982

/-- The number of available toppings -/
def n : ℕ := 8

/-- The number of toppings on each pizza -/
def k : ℕ := 5

/-- Binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of unique five-topping pizzas with 8 available toppings is 56 -/
theorem unique_pizza_combinations : binomial n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_pizza_combinations_l3109_310982


namespace NUMINAMATH_CALUDE_linear_function_passes_through_points_l3109_310963

/-- A linear function y = kx - k passing through (-1, 4) also passes through (1, 0) -/
theorem linear_function_passes_through_points :
  ∃ k : ℝ, (k * (-1) - k = 4) ∧ (k * 1 - k = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_points_l3109_310963


namespace NUMINAMATH_CALUDE_estimate_pi_l3109_310940

theorem estimate_pi (n : ℕ) (m : ℕ) (h1 : n = 120) (h2 : m = 34) :
  let π_estimate : ℚ := 4 * (m : ℚ) / (n : ℚ) + 2
  π_estimate = 47 / 15 := by
  sorry

end NUMINAMATH_CALUDE_estimate_pi_l3109_310940


namespace NUMINAMATH_CALUDE_square_difference_from_sum_and_product_l3109_310960

theorem square_difference_from_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_from_sum_and_product_l3109_310960


namespace NUMINAMATH_CALUDE_min_a_for_inequality_solution_set_inequality_l3109_310977

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2) * abs (x - 2)

-- Theorem for part (1)
theorem min_a_for_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 1, f x ≤ a) ↔ a ≥ 4 := by sorry

-- Theorem for part (2)
theorem solution_set_inequality :
  {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4} ∪ {x : ℝ | -4 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_solution_set_inequality_l3109_310977


namespace NUMINAMATH_CALUDE_cookies_difference_l3109_310985

def initial_cookies : ℕ := 41
def cookies_given : ℕ := 9
def cookies_eaten : ℕ := 18

theorem cookies_difference : cookies_eaten - cookies_given = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookies_difference_l3109_310985


namespace NUMINAMATH_CALUDE_field_width_calculation_l3109_310907

/-- Proves that the width of each field is 250 meters -/
theorem field_width_calculation (num_fields : ℕ) (field_length : ℝ) (total_area_km2 : ℝ) :
  num_fields = 8 →
  field_length = 300 →
  total_area_km2 = 0.6 →
  ∃ (width : ℝ), width = 250 ∧ 
    (num_fields * field_length * width = total_area_km2 * 1000000) :=
by sorry

end NUMINAMATH_CALUDE_field_width_calculation_l3109_310907


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3109_310908

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3109_310908


namespace NUMINAMATH_CALUDE_jake_weight_loss_l3109_310988

theorem jake_weight_loss (jake_current sister_current total_current : ℕ) 
  (h1 : jake_current + sister_current = total_current)
  (h2 : jake_current = 156)
  (h3 : total_current = 224) :
  ∃ (weight_loss : ℕ), jake_current - weight_loss = 2 * (sister_current - weight_loss) ∧ weight_loss = 20 := by
sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l3109_310988


namespace NUMINAMATH_CALUDE_prime_solution_existence_l3109_310983

theorem prime_solution_existence (p : ℕ) : 
  Prime p ↔ (p = 2 ∨ p = 3 ∨ p = 7) ∧ 
  (∃ x y : ℕ+, x * (y^2 - p) + y * (x^2 - p) = 5 * p) :=
sorry

end NUMINAMATH_CALUDE_prime_solution_existence_l3109_310983


namespace NUMINAMATH_CALUDE_divisibility_properties_l3109_310914

theorem divisibility_properties (n : ℤ) :
  -- Part (a)
  (n = 3 → ∃ m₁ m₂ : ℤ, m₁ = -5 ∧ m₂ = 9 ∧
    ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 ↔ m = m₁ ∨ m = m₂) ∧
  -- Part (b)
  (∃ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0) ∧
  -- Part (c)
  (∃ k : ℕ, ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l3109_310914


namespace NUMINAMATH_CALUDE_exists_four_digit_sum_21_div_14_l3109_310936

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem exists_four_digit_sum_21_div_14 : 
  ∃ (n : ℕ), is_four_digit n ∧ digit_sum n = 21 ∧ n % 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_four_digit_sum_21_div_14_l3109_310936


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3109_310984

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) : 
  (4 / 3 * Real.pi * p^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p)^3) = 1 / 13.5 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3109_310984


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3109_310904

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {y | y^2 + y = 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3109_310904


namespace NUMINAMATH_CALUDE_dog_fruits_total_l3109_310923

/-- Represents the number of fruits eaten by each dog -/
structure DogFruits where
  apples : ℕ
  blueberries : ℕ
  bonnies : ℕ
  cherries : ℕ

/-- The conditions of the problem and the theorem to prove -/
theorem dog_fruits_total (df : DogFruits) : 
  df.apples = 3 * df.blueberries →
  df.blueberries = (3 * df.bonnies) / 4 →
  df.cherries = 5 * df.apples →
  df.bonnies = 60 →
  df.apples + df.blueberries + df.bonnies + df.cherries = 915 := by
  sorry

#check dog_fruits_total

end NUMINAMATH_CALUDE_dog_fruits_total_l3109_310923


namespace NUMINAMATH_CALUDE_next_in_series_is_2425_l3109_310973

def series : List Nat := [2, 5, 31, 241]

def next_multiplier (n : Nat) : Nat :=
  match n with
  | 0 => 2
  | 1 => 5
  | _ => 7 + 3 * (n - 2)

def next_addend (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | 1 => 6
  | _ => 10 + 5 * (n - 2)

def next_number (series : List Nat) : Nat :=
  match series.reverse with
  | [] => 0
  | x :: _ => x * next_multiplier (series.length - 1) + next_addend (series.length - 1)

theorem next_in_series_is_2425 : next_number series = 2425 := by
  sorry

end NUMINAMATH_CALUDE_next_in_series_is_2425_l3109_310973


namespace NUMINAMATH_CALUDE_geometric_sequence_306th_term_l3109_310951

/-- Given a geometric sequence with first term 9 and second term -18,
    the 306th term is -9 * 2^305 -/
theorem geometric_sequence_306th_term :
  ∀ (a : ℕ → ℤ), a 1 = 9 ∧ a 2 = -18 →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * (-2)) →
  a 306 = -9 * 2^305 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_306th_term_l3109_310951


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l3109_310979

/-- 
Given a line mx - y - 3 - m = 0, if its intercepts on the x-axis and y-axis are equal, 
then m = -3 or m = -1.
-/
theorem line_equal_intercepts (m : ℝ) : 
  (∃ (a : ℝ), a ≠ 0 ∧ m * a - 3 - m = 0 ∧ -3 - m = a) → 
  (m = -3 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l3109_310979


namespace NUMINAMATH_CALUDE_four_squares_sum_l3109_310986

theorem four_squares_sum (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a^2 + b^2 + c^2 + d^2 = 90 →
  a + b + c + d = 16 →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 := by
sorry

end NUMINAMATH_CALUDE_four_squares_sum_l3109_310986


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3109_310913

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x ≤ 0} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3109_310913


namespace NUMINAMATH_CALUDE_root_in_interval_l3109_310912

noncomputable section

variables (a b : ℝ) (h : b ≥ 2*a) (h' : a > 0)

def f (x : ℝ) := 2*(a^x) - b^x

theorem root_in_interval :
  ∃ x, x ∈ Set.Ioo 0 1 ∧ f a b x = 0 :=
sorry

end

end NUMINAMATH_CALUDE_root_in_interval_l3109_310912


namespace NUMINAMATH_CALUDE_february_first_is_sunday_l3109_310974

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day n days after a given day -/
def nDaysAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (nDaysAfter d m)

theorem february_first_is_sunday (h : nDaysAfter DayOfWeek.Sunday 13 = DayOfWeek.Saturday) :
  DayOfWeek.Sunday = nDaysAfter DayOfWeek.Sunday 0 :=
by sorry

end NUMINAMATH_CALUDE_february_first_is_sunday_l3109_310974


namespace NUMINAMATH_CALUDE_steamer_problem_l3109_310972

theorem steamer_problem :
  ∃ (a b c n k p x : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    1 ≤ n ∧ n ≤ 31 ∧
    1 ≤ k ∧ k ≤ 12 ∧
    p ≥ 0 ∧
    a * b * c * n * k * p + x^3 = 4752862 := by
  sorry

end NUMINAMATH_CALUDE_steamer_problem_l3109_310972


namespace NUMINAMATH_CALUDE_stratified_sample_medium_supermarkets_l3109_310906

/-- Given a population of supermarkets with the following properties:
  * total_supermarkets: The total number of supermarkets
  * medium_supermarkets: The number of medium-sized supermarkets
  * sample_size: The size of the stratified sample to be taken
  
  This theorem proves that the number of medium-sized supermarkets
  to be selected in the sample is equal to the expected value. -/
theorem stratified_sample_medium_supermarkets
  (total_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (sample_size : ℕ)
  (h1 : total_supermarkets = 2000)
  (h2 : medium_supermarkets = 400)
  (h3 : sample_size = 100)
  : (medium_supermarkets * sample_size) / total_supermarkets = 20 := by
  sorry

#check stratified_sample_medium_supermarkets

end NUMINAMATH_CALUDE_stratified_sample_medium_supermarkets_l3109_310906


namespace NUMINAMATH_CALUDE_m_range_l3109_310943

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 2) →
  m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3109_310943


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l3109_310966

theorem repeating_decimal_subtraction : 
  ∃ (a b c : ℚ), 
    (∀ n : ℕ, a = (234 / 10^3 + 234 / (10^3 * (1000^n - 1)))) ∧
    (∀ n : ℕ, b = (567 / 10^3 + 567 / (10^3 * (1000^n - 1)))) ∧
    (∀ n : ℕ, c = (891 / 10^3 + 891 / (10^3 * (1000^n - 1)))) ∧
    a - b - c = -408 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l3109_310966


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3109_310962

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0 → x + y + z = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3109_310962


namespace NUMINAMATH_CALUDE_salt_price_reduction_l3109_310901

/-- Given a 20% reduction in the price of salt allows 10 kgs more to be purchased for Rs. 400,
    prove that the original price per kg of salt was Rs. 10. -/
theorem salt_price_reduction (P : ℝ) 
  (h1 : P > 0) -- The price is positive
  (h2 : ∃ (X : ℝ), 400 / P = X ∧ 400 / (0.8 * P) = X + 10) -- Condition from the problem
  : P = 10 := by
  sorry

end NUMINAMATH_CALUDE_salt_price_reduction_l3109_310901
