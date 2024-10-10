import Mathlib

namespace landscape_ratio_is_8_to_1_l2439_243985

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playgroundArea : ℝ

/-- The ratio of breadth to length as a pair of integers -/
def BreadthLengthRatio := ℕ × ℕ

/-- Calculates the ratio of breadth to length -/
def calculateRatio (l : Landscape) : BreadthLengthRatio :=
  sorry

theorem landscape_ratio_is_8_to_1 (l : Landscape) 
  (h1 : ∃ n : ℝ, l.breadth = n * l.length)
  (h2 : l.playgroundArea = 3200)
  (h3 : l.length * l.breadth = 9 * l.playgroundArea)
  (h4 : l.breadth = 480) : 
  calculateRatio l = (8, 1) :=
sorry

end landscape_ratio_is_8_to_1_l2439_243985


namespace revenue_change_l2439_243994

theorem revenue_change 
  (T : ℝ) -- Original tax
  (C : ℝ) -- Original consumption
  (tax_reduction : ℝ) -- Tax reduction percentage
  (consumption_increase : ℝ) -- Consumption increase percentage
  (h1 : tax_reduction = 0.20) -- 20% tax reduction
  (h2 : consumption_increase = 0.15) -- 15% consumption increase
  : 
  (1 - tax_reduction) * (1 + consumption_increase) * T * C - T * C = -0.08 * T * C :=
by sorry

end revenue_change_l2439_243994


namespace james_total_earnings_l2439_243912

def january_earnings : ℕ := 4000

def february_earnings : ℕ := 2 * january_earnings

def march_earnings : ℕ := february_earnings - 2000

def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings : total_earnings = 18000 := by
  sorry

end james_total_earnings_l2439_243912


namespace incenter_vector_sum_implies_right_angle_l2439_243981

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a vector from a point to another point
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

-- Define vector addition
def add_vectors (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define scalar multiplication of a vector
def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define the angle at a vertex of a triangle
def angle_at_vertex (t : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem incenter_vector_sum_implies_right_angle (t : Triangle) :
  let I := incenter t
  let IA := vector I t.A
  let IB := vector I t.B
  let IC := vector I t.C
  add_vectors (scalar_mult 3 IA) (add_vectors (scalar_mult 4 IB) (scalar_mult 5 IC)) = (0, 0) →
  angle_at_vertex t t.C = 90 :=
sorry

end incenter_vector_sum_implies_right_angle_l2439_243981


namespace red_stamp_price_l2439_243959

theorem red_stamp_price (simon_stamps : ℕ) (peter_stamps : ℕ) (white_stamp_price : ℚ) (price_difference : ℚ) :
  simon_stamps = 30 →
  peter_stamps = 80 →
  white_stamp_price = 0.20 →
  price_difference = 1 →
  ∃ (red_stamp_price : ℚ), 
    red_stamp_price * simon_stamps - white_stamp_price * peter_stamps = price_difference ∧
    red_stamp_price = 17 / 30 := by
  sorry

end red_stamp_price_l2439_243959


namespace two_digit_integers_theorem_l2439_243911

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def share_digit (a b : ℕ) : Prop :=
  (a / 10 = b / 10) ∨ (a / 10 = b % 10) ∨ (a % 10 = b / 10) ∨ (a % 10 = b % 10)

theorem two_digit_integers_theorem (a b : ℕ) :
  is_two_digit a ∧ is_two_digit b ∧
  ((a = b + 12) ∨ (b = a + 12)) ∧
  share_digit a b ∧
  ((digit_sum a = digit_sum b + 3) ∨ (digit_sum b = digit_sum a + 3)) →
  (∃ t : ℕ, 2 ≤ t ∧ t ≤ 8 ∧ a = 11 * t + 10 ∧ b = 11 * t - 2) ∨
  (∃ s : ℕ, 1 ≤ s ∧ s ≤ 6 ∧ a = 11 * s + 1 ∧ b = 11 * s + 13) :=
by sorry

end two_digit_integers_theorem_l2439_243911


namespace chessTeamArrangements_eq_12_l2439_243904

/-- The number of ways to arrange a chess team with 3 boys and 2 girls in a specific order -/
def chessTeamArrangements : ℕ :=
  let numBoys : ℕ := 3
  let numGirls : ℕ := 2
  let girlArrangements : ℕ := Nat.factorial numGirls
  let boyArrangements : ℕ := Nat.factorial numBoys
  girlArrangements * boyArrangements

/-- Theorem stating that the number of arrangements is 12 -/
theorem chessTeamArrangements_eq_12 : chessTeamArrangements = 12 := by
  sorry

end chessTeamArrangements_eq_12_l2439_243904


namespace product_of_roots_l2439_243946

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 50 = 0) → 
  ∃ p q r : ℝ, (x - p)*(x - q)*(x - r) = x^3 - 15*x^2 + 75*x - 50 ∧ p*q*r = 50 := by
  sorry

end product_of_roots_l2439_243946


namespace truck_sales_l2439_243930

theorem truck_sales (total_vehicles : ℕ) (car_truck_difference : ℕ) : 
  total_vehicles = 69 → car_truck_difference = 27 → 
  ∃ trucks : ℕ, trucks = 21 ∧ trucks + (trucks + car_truck_difference) = total_vehicles :=
by
  sorry

end truck_sales_l2439_243930


namespace average_difference_l2439_243934

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (x + 50 + 45) / 3 + 5 → x = 10 := by
  sorry

end average_difference_l2439_243934


namespace prop_p_or_q_false_iff_a_in_range_l2439_243919

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, a^2 * x^2 + a * x - 2 = 0 ∧ -1 ≤ x ∧ x ≤ 1

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- Theorem statement
theorem prop_p_or_q_false_iff_a_in_range (a : ℝ) :
  (¬(p a ∨ q a)) ↔ ((-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)) :=
sorry

end prop_p_or_q_false_iff_a_in_range_l2439_243919


namespace arithmetic_progression_x_value_l2439_243971

theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ := 2*x - 2
  let a₂ := 2*x + 2
  let a₃ := 4*x + 4
  (a₂ - a₁ = a₃ - a₂) → x = 1 := by
sorry

end arithmetic_progression_x_value_l2439_243971


namespace similar_triangle_perimeter_l2439_243999

theorem similar_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ∃ (k : ℝ), k > 0 ∧ 
  (k * a = 18 ∨ k * b = 18) →
  k * (a + b + c) = 72 :=
by
  sorry

end similar_triangle_perimeter_l2439_243999


namespace polynomial_is_square_of_binomial_l2439_243933

/-- The polynomial 4x^2 + 16x + 16 is the square of a binomial. -/
theorem polynomial_is_square_of_binomial :
  ∃ (r s : ℝ), ∀ x, 4 * x^2 + 16 * x + 16 = (r * x + s)^2 :=
sorry

end polynomial_is_square_of_binomial_l2439_243933


namespace hyperbola_construction_uniqueness_l2439_243972

/-- A tangent line to a hyperbola at its vertex -/
structure Tangent where
  line : Line

/-- An asymptote of a hyperbola -/
structure Asymptote where
  line : Line

/-- Linear eccentricity of a hyperbola -/
def LinearEccentricity : Type := ℝ

/-- A hyperbola -/
structure Hyperbola where
  -- Define necessary components of a hyperbola

/-- Two hyperbolas are congruent if they have the same shape and size -/
def congruent (h1 h2 : Hyperbola) : Prop := sorry

/-- Two hyperbolas are parallel translations if one can be obtained from the other by a translation -/
def parallel_translation (h1 h2 : Hyperbola) (dir : Vec) : Prop := sorry

/-- Main theorem: Given a tangent, an asymptote, and linear eccentricity, 
    there exist exactly two congruent hyperbolas satisfying these conditions -/
theorem hyperbola_construction_uniqueness 
  (t : Tangent) (a₁ : Asymptote) (c : LinearEccentricity) :
  ∃! (h1 h2 : Hyperbola), 
    (∃ (dir : Vec), parallel_translation h1 h2 dir) ∧ 
    congruent h1 h2 ∧
    -- Additional conditions to ensure h1 and h2 satisfy t, a₁, and c
    sorry := by
  sorry

end hyperbola_construction_uniqueness_l2439_243972


namespace five_integers_with_remainder_one_l2439_243945

theorem five_integers_with_remainder_one : 
  ∃! (S : Finset ℕ), 
    S.card = 5 ∧ 
    (∀ n ∈ S, n ≤ 50) ∧ 
    (∀ n ∈ S, n % 11 = 1) :=
by sorry

end five_integers_with_remainder_one_l2439_243945


namespace power_of_special_sum_l2439_243921

theorem power_of_special_sum (a b : ℝ) 
  (h : b = Real.sqrt (1 - 2*a) + Real.sqrt (2*a - 1) + 3) : 
  a ^ b = (1/8 : ℝ) := by
  sorry

end power_of_special_sum_l2439_243921


namespace simplify_sqrt_sum_l2439_243939

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l2439_243939


namespace max_value_on_circle_l2439_243991

theorem max_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) :
  ∃ (max : ℝ), (∀ (a b : ℝ), a^2 + b^2 = 2 → 3*a + 4*b ≤ max) ∧ max = 5 * Real.sqrt 2 := by
  sorry

end max_value_on_circle_l2439_243991


namespace g_definition_l2439_243908

-- Define the function g
def g : ℝ → ℝ := fun x ↦ 2*x - 11

-- Theorem statement
theorem g_definition : ∀ x : ℝ, g (x + 2) = 2*x - 3 := by
  sorry

end g_definition_l2439_243908


namespace circle_constraint_extrema_sum_l2439_243980

theorem circle_constraint_extrema_sum (x y : ℝ) :
  x^2 + y^2 = 1 →
  ∃ (min max : ℝ),
    (∀ x' y' : ℝ, x'^2 + y'^2 = 1 → 
      min ≤ (x'-3)^2 + (y'+4)^2 ∧ (x'-3)^2 + (y'+4)^2 ≤ max) ∧
    min + max = 52 := by
  sorry

end circle_constraint_extrema_sum_l2439_243980


namespace max_baseball_hits_percentage_l2439_243970

theorem max_baseball_hits_percentage (total_hits : ℕ) (home_runs triples doubles : ℕ) 
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 4)
  (h4 : doubles = 10) :
  (total_hits - (home_runs + triples + doubles)) / total_hits * 100 = 68 := by
  sorry

end max_baseball_hits_percentage_l2439_243970


namespace car_wheels_count_l2439_243973

theorem car_wheels_count (cars : ℕ) (motorcycles : ℕ) (total_wheels : ℕ) 
  (h1 : cars = 19)
  (h2 : motorcycles = 11)
  (h3 : total_wheels = 117)
  (h4 : ∀ m : ℕ, m ≤ motorcycles → 2 * m ≤ total_wheels) :
  ∃ (wheels_per_car : ℕ), wheels_per_car * cars + 2 * motorcycles = total_wheels ∧ wheels_per_car = 5 := by
  sorry

end car_wheels_count_l2439_243973


namespace same_solution_implies_a_equals_one_l2439_243975

theorem same_solution_implies_a_equals_one :
  (∃ x : ℝ, 2 - a - x = 0 ∧ 2*x + 1 = 3) → a = 1 := by
  sorry

end same_solution_implies_a_equals_one_l2439_243975


namespace marching_band_weight_is_245_l2439_243925

/-- Represents the total weight carried by the Oprah Winfrey High School marching band. -/
def marching_band_weight : ℕ :=
  let trumpet_clarinet_weight := 5
  let trombone_weight := 10
  let tuba_weight := 20
  let drum_weight := 15
  let trumpet_count := 6
  let clarinet_count := 9
  let trombone_count := 8
  let tuba_count := 3
  let drum_count := 2
  (trumpet_clarinet_weight * (trumpet_count + clarinet_count)) +
  (trombone_weight * trombone_count) +
  (tuba_weight * tuba_count) +
  (drum_weight * drum_count)

/-- Theorem stating that the total weight carried by the marching band is 245 pounds. -/
theorem marching_band_weight_is_245 : marching_band_weight = 245 := by
  sorry

end marching_band_weight_is_245_l2439_243925


namespace sum_of_squares_l2439_243910

theorem sum_of_squares (x y : ℝ) : 
  x * y = 10 → x^2 * y + x * y^2 + x + y = 80 → x^2 + y^2 = 3980 / 121 := by
  sorry

end sum_of_squares_l2439_243910


namespace max_rented_trucks_is_twenty_l2439_243900

/-- Represents the truck rental scenario for a week -/
structure TruckRental where
  total : ℕ
  returned_percent : ℚ
  saturday_minimum : ℕ

/-- Calculates the maximum number of trucks that could have been rented out -/
def max_rented_trucks (rental : TruckRental) : ℕ :=
  min rental.total (2 * rental.saturday_minimum)

/-- Theorem stating the maximum number of trucks that could have been rented out -/
theorem max_rented_trucks_is_twenty (rental : TruckRental) :
    rental.total = 20 ∧ 
    rental.returned_percent = 1/2 ∧ 
    rental.saturday_minimum = 10 →
    max_rented_trucks rental = 20 := by
  sorry

end max_rented_trucks_is_twenty_l2439_243900


namespace frog_arrangement_count_l2439_243920

/-- Represents the number of ways to arrange frogs with given constraints -/
def frog_arrangements (total : ℕ) (green : ℕ) (red : ℕ) (blue : ℕ) : ℕ :=
  2 * (Nat.factorial blue) * (Nat.factorial red) * (Nat.factorial green)

/-- Theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 8 2 3 3 = 144 :=
by
  sorry

end frog_arrangement_count_l2439_243920


namespace supermarket_max_profit_l2439_243978

/-- Represents the daily profit function for a supermarket selling daily necessities -/
def daily_profit (x : ℝ) : ℝ :=
  (200 - 10 * (x - 50)) * (x - 40)

/-- The maximum daily profit achievable by the supermarket -/
def max_daily_profit : ℝ := 2250

theorem supermarket_max_profit :
  ∃ (x : ℝ), daily_profit x = max_daily_profit ∧
  ∀ (y : ℝ), daily_profit y ≤ max_daily_profit :=
by sorry

end supermarket_max_profit_l2439_243978


namespace x_squared_plus_y_squared_l2439_243979

theorem x_squared_plus_y_squared (x y : ℚ) 
  (h : 2002 * (x - 1)^2 + |x - 12*y + 1| = 0) : 
  x^2 + y^2 = 37 / 36 := by
sorry

end x_squared_plus_y_squared_l2439_243979


namespace least_number_divisibility_l2439_243968

theorem least_number_divisibility (x : ℕ) : 
  (x > 0) →
  (x / 5 = (x % 34) + 8) →
  (∀ y : ℕ, y > 0 → y / 5 = (y % 34) + 8 → y ≥ x) →
  x = 160 := by
sorry

end least_number_divisibility_l2439_243968


namespace train_speeds_l2439_243962

theorem train_speeds (distance : ℝ) (time : ℝ) (speed_difference : ℝ) 
  (h1 : distance = 450)
  (h2 : time = 5)
  (h3 : speed_difference = 6) :
  ∃ (speed1 speed2 : ℝ),
    speed2 = speed1 + speed_difference ∧
    distance = (speed1 + speed2) * time ∧
    speed1 = 42 ∧
    speed2 = 48 := by
  sorry

end train_speeds_l2439_243962


namespace smallest_sum_of_three_l2439_243905

def S : Finset Int := {-5, 30, -2, 15, -4}

theorem smallest_sum_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x + y + z = -11 ∧
  ∀ (d e f : Int), d ∈ S → e ∈ S → f ∈ S → 
  d ≠ e ∧ e ≠ f ∧ d ≠ f → 
  d + e + f ≥ -11 :=
by sorry

end smallest_sum_of_three_l2439_243905


namespace bhanu_petrol_expense_l2439_243943

def bhanu_expenditure (total_income : ℝ) : Prop :=
  let petrol_percent : ℝ := 0.30
  let rent_percent : ℝ := 0.30
  let petrol_expense : ℝ := petrol_percent * total_income
  let remaining_after_petrol : ℝ := total_income - petrol_expense
  let rent_expense : ℝ := rent_percent * remaining_after_petrol
  rent_expense = 210 ∧ petrol_expense = 300

theorem bhanu_petrol_expense : 
  ∃ (total_income : ℝ), bhanu_expenditure total_income :=
sorry

end bhanu_petrol_expense_l2439_243943


namespace intersection_M_N_l2439_243964

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 2*x + 8)}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = abs x + 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | -2 ≤ x ∧ x ≤ 4} := by sorry

end intersection_M_N_l2439_243964


namespace largest_valid_number_l2439_243916

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- Four-digit number
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧  -- All digits are different
  (∀ i j, i < j → (n / 10^i) % 10 ≤ (n / 10^j) % 10)  -- No two digits can be swapped to form a smaller number

theorem largest_valid_number : 
  is_valid_number 7089 ∧ ∀ m, is_valid_number m → m ≤ 7089 :=
sorry

end largest_valid_number_l2439_243916


namespace corrected_mean_l2439_243917

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * (36.5 : ℚ) :=
by sorry

end corrected_mean_l2439_243917


namespace certain_number_exists_l2439_243955

theorem certain_number_exists : ∃ x : ℝ, 220050 = (555 + x) * (2 * (x - 555)) + 50 := by
  sorry

end certain_number_exists_l2439_243955


namespace circle_fraction_range_l2439_243982

theorem circle_fraction_range (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  -(Real.sqrt 3 / 3) ≤ y / (x + 2) ∧ y / (x + 2) ≤ Real.sqrt 3 / 3 :=
sorry

end circle_fraction_range_l2439_243982


namespace no_integer_solutions_l2439_243944

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^3 + 21*y^2 + 5 = 0 := by
  sorry

end no_integer_solutions_l2439_243944


namespace range_of_a_l2439_243947

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x - 2 < 0) →
  a ∈ Set.Ioc (-8) 0 :=
by sorry

end range_of_a_l2439_243947


namespace min_value_of_quadratic_function_l2439_243906

theorem min_value_of_quadratic_function :
  ∃ (min : ℝ), min = -1 ∧ ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 - 12 * x + 8 * y + 15 ≥ min :=
by sorry

end min_value_of_quadratic_function_l2439_243906


namespace direct_proportion_through_points_l2439_243922

/-- A direct proportion function passing through (-1, 2) also passes through (1, -2) -/
theorem direct_proportion_through_points :
  ∀ (f : ℝ → ℝ) (k : ℝ),
    (∀ x, f x = k * x) →  -- f is a direct proportion function
    f (-1) = 2 →          -- f passes through (-1, 2)
    f 1 = -2 :=           -- f passes through (1, -2)
by
  sorry

end direct_proportion_through_points_l2439_243922


namespace rectangle_area_rectangle_area_is_220_l2439_243967

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_220 :
  rectangle_area 3025 10 = 220 := by
  sorry

end rectangle_area_rectangle_area_is_220_l2439_243967


namespace sprint_stats_change_l2439_243915

theorem sprint_stats_change (n : Nat) (avg_10 : ℝ) (var_10 : ℝ) (time_11 : ℝ) :
  n = 10 →
  avg_10 = 8.2 →
  var_10 = 2.2 →
  time_11 = 8.2 →
  let avg_11 := (n * avg_10 + time_11) / (n + 1)
  let var_11 := (n * var_10 + (time_11 - avg_10)^2) / (n + 1)
  avg_11 = avg_10 ∧ var_11 < var_10 := by
  sorry

#check sprint_stats_change

end sprint_stats_change_l2439_243915


namespace min_value_xy_over_x2_plus_y2_l2439_243928

theorem min_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/4 ≤ x ∧ x ≤ 3/5) (hy : 1/5 ≤ y ∧ y ≤ 2/3) :
  x * y / (x^2 + y^2) ≥ 24/73 := by
  sorry

end min_value_xy_over_x2_plus_y2_l2439_243928


namespace cos_180_degrees_l2439_243957

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end cos_180_degrees_l2439_243957


namespace floor_sqrt_80_l2439_243929

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l2439_243929


namespace double_acute_angle_range_l2439_243936

-- Define an acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem statement
theorem double_acute_angle_range (θ : ℝ) (h : is_acute_angle θ) :
  0 < 2 * θ ∧ 2 * θ < Real.pi :=
by sorry

end double_acute_angle_range_l2439_243936


namespace infinite_zeros_or_nines_in_difference_l2439_243969

/-- Represents an infinite decimal fraction -/
def InfiniteDecimalFraction := ℕ → Fin 10

/-- Given a set of 11 infinite decimal fractions, there exist two fractions
    whose difference has either infinite zeros or infinite nines -/
theorem infinite_zeros_or_nines_in_difference 
  (fractions : Fin 11 → InfiniteDecimalFraction) :
  ∃ i j : Fin 11, i ≠ j ∧ 
    (∀ k : ℕ, (fractions i k - fractions j k) % 10 = 0 ∨
              (fractions i k - fractions j k) % 10 = 9) :=
sorry

end infinite_zeros_or_nines_in_difference_l2439_243969


namespace odd_decreasing_properties_l2439_243995

-- Define an odd, decreasing function on ℝ
def odd_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x > f y)

-- Theorem statement
theorem odd_decreasing_properties {f : ℝ → ℝ} {a b : ℝ} 
  (h_f : odd_decreasing_function f) (h_sum : a + b ≤ 0) : 
  (f a * f (-a) ≤ 0) ∧ (f a + f b ≥ f (-a) + f (-b)) :=
by sorry

end odd_decreasing_properties_l2439_243995


namespace solution_set_l2439_243924

theorem solution_set (x : ℝ) :
  (x - 2) / (x - 4) ≥ 3 ∧ x ≠ 2 → 4 < x ∧ x ≤ 5 := by
  sorry

end solution_set_l2439_243924


namespace watermelon_seeds_l2439_243997

/-- Represents a watermelon slice with black and white seeds -/
structure WatermelonSlice where
  blackSeeds : ℕ
  whiteSeeds : ℕ

/-- Calculates the total number of seeds in a watermelon -/
def totalSeeds (slices : ℕ) (slice : WatermelonSlice) : ℕ :=
  slices * (slice.blackSeeds + slice.whiteSeeds)

/-- Theorem: The total number of seeds in the watermelon is 1600 -/
theorem watermelon_seeds :
  ∀ (slice : WatermelonSlice),
    slice.blackSeeds = 20 →
    slice.whiteSeeds = 20 →
    totalSeeds 40 slice = 1600 :=
by
  sorry

end watermelon_seeds_l2439_243997


namespace solve_equation_l2439_243909

theorem solve_equation (x : ℚ) (h : 5 * x - 8 = 15 * x + 18) : 3 * (x + 9) = 96 / 5 := by
  sorry

end solve_equation_l2439_243909


namespace intersection_of_M_and_N_l2439_243932

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end intersection_of_M_and_N_l2439_243932


namespace complex_product_QED_l2439_243974

theorem complex_product_QED (Q E D : ℂ) : 
  Q = 4 + 3*I ∧ E = 2*I ∧ D = 4 - 3*I → Q * E * D = 50*I :=
by
  sorry

end complex_product_QED_l2439_243974


namespace new_encoding_of_original_message_l2439_243996

/-- Represents the encoding of a character in the old system -/
def OldEncoding : Char → String
| 'A' => "011"
| 'B' => "011"
| 'C' => "0"
| _ => ""

/-- Represents the encoding of a character in the new system -/
def NewEncoding : Char → String
| 'A' => "21"
| 'B' => "122"
| 'C' => "1"
| _ => ""

/-- Decodes a string from the old encoding system -/
def decodeOld (s : String) : String := sorry

/-- Encodes a string using the new encoding system -/
def encodeNew (s : String) : String := sorry

/-- The original message in the old encoding -/
def originalMessage : String := "011011010011"

/-- Theorem stating that the new encoding of the original message is "211221121" -/
theorem new_encoding_of_original_message :
  encodeNew (decodeOld originalMessage) = "211221121" := by sorry

end new_encoding_of_original_message_l2439_243996


namespace friend_apple_rotations_l2439_243923

/-- Given the conditions of a juggling contest, prove the number of rotations made by each of Toby's friend's apples -/
theorem friend_apple_rotations 
  (toby_baseballs : ℕ)
  (toby_rotations_per_baseball : ℕ)
  (friend_apples : ℕ)
  (winner_total_rotations : ℕ)
  (h1 : toby_baseballs = 5)
  (h2 : toby_rotations_per_baseball = 80)
  (h3 : friend_apples = 4)
  (h4 : winner_total_rotations = 404)
  : (winner_total_rotations - toby_baseballs * toby_rotations_per_baseball) / friend_apples + toby_rotations_per_baseball = 81 := by
  sorry

end friend_apple_rotations_l2439_243923


namespace furniture_reimbursement_l2439_243976

/-- Calculates the reimbursement amount for an overcharged furniture purchase -/
theorem furniture_reimbursement
  (num_pieces : ℕ)
  (amount_paid : ℚ)
  (cost_per_piece : ℚ)
  (h1 : num_pieces = 150)
  (h2 : amount_paid = 20700)
  (h3 : cost_per_piece = 134) :
  amount_paid - (num_pieces : ℚ) * cost_per_piece = 600 := by
  sorry

end furniture_reimbursement_l2439_243976


namespace tan_sum_from_sin_cos_sum_l2439_243983

theorem tan_sum_from_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 116 / 85) 
  (h2 : Real.cos x + Real.cos y = 42 / 85) : 
  Real.tan x + Real.tan y = -232992832 / 5705296111 := by
  sorry

end tan_sum_from_sin_cos_sum_l2439_243983


namespace ratio_comparison_l2439_243927

theorem ratio_comparison (y : ℕ) (h : y > 4) : (3 : ℚ) / 4 < 4 / y := by
  sorry

end ratio_comparison_l2439_243927


namespace log_xy_value_l2439_243977

theorem log_xy_value (x y : ℝ) 
  (h1 : Real.log (x^2 * y^2) = 1) 
  (h2 : Real.log (x^3 * y) = 2) : 
  Real.log (x * y) = 1/2 := by
sorry

end log_xy_value_l2439_243977


namespace largest_power_dividing_factorial_l2439_243937

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_dividing_factorial :
  ∃ k : ℕ, k = 3 ∧
  (∀ m : ℕ, 2012^m ∣ factorial 2012 → m ≤ k) ∧
  2012^k ∣ factorial 2012 :=
by sorry

end largest_power_dividing_factorial_l2439_243937


namespace original_dish_price_l2439_243987

theorem original_dish_price (price : ℝ) : 
  (price * 0.9 + price * 0.15 = price * 0.9 + price * 0.9 * 0.15 + 1.26) → 
  price = 84 :=
by sorry

end original_dish_price_l2439_243987


namespace simplify_expressions_l2439_243984

theorem simplify_expressions :
  let exp1 := ((0.064 ^ (1/5)) ^ (-2.5)) ^ (2/3) - (3 * (3/8)) ^ (1/3) - π ^ 0
  let exp2 := (2 * Real.log 2 + Real.log 3) / (1 + (1/2) * Real.log 0.36 + (1/4) * Real.log 16)
  (exp1 = 0) ∧ (exp2 = (2 * Real.log 2 + Real.log 3) / Real.log 24) := by
  sorry

end simplify_expressions_l2439_243984


namespace farmer_apples_l2439_243907

/-- The number of apples the farmer gave away -/
def apples_given_away : ℕ := 88

/-- The number of apples the farmer has left -/
def apples_left : ℕ := 39

/-- The initial number of apples the farmer had -/
def initial_apples : ℕ := apples_given_away + apples_left

theorem farmer_apples : initial_apples = 127 := by
  sorry

end farmer_apples_l2439_243907


namespace quadratic_and_line_properties_l2439_243938

/-- Given a quadratic equation with two equal real roots, prove the value of m and the quadrants through which the corresponding line passes -/
theorem quadratic_and_line_properties :
  ∀ m : ℝ,
  (∀ x : ℝ, x^2 + (2*m + 1)*x + m^2 + 2 = 0 → (∃! r : ℝ, x = r)) →
  (m = 7/4 ∧ 
   ∀ x y : ℝ, y = (2*m - 3)*x - 4*m + 6 →
   (∃ x₁ y₁ : ℝ, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = (2*m - 3)*x₁ - 4*m + 6) ∧
   (∃ x₂ y₂ : ℝ, x₂ < 0 ∧ y₂ < 0 ∧ y₂ = (2*m - 3)*x₂ - 4*m + 6) ∧
   (∃ x₃ y₃ : ℝ, x₃ > 0 ∧ y₃ < 0 ∧ y₃ = (2*m - 3)*x₃ - 4*m + 6)) :=
by
  sorry


end quadratic_and_line_properties_l2439_243938


namespace intersection_S_T_l2439_243958

def S : Set ℝ := {x | x ≥ 1}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} := by sorry

end intersection_S_T_l2439_243958


namespace center_cell_is_seven_l2439_243960

/-- Represents a 3x3 grid with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two cells are adjacent in the grid -/
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Checks if two numbers are consecutive -/
def consecutive (a b : Fin 9) : Prop :=
  a.val + 1 = b.val ∨ b.val + 1 = a.val

/-- The main theorem -/
theorem center_cell_is_seven (g : Grid)
  (all_nums : ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n)
  (consec_adjacent : ∀ i₁ j₁ i₂ j₂ : Fin 3,
    consecutive (g i₁ j₁) (g i₂ j₂) → adjacent (i₁, j₁) (i₂, j₂))
  (corner_sum : g 0 0 + g 0 2 + g 2 0 + g 2 2 = 18) :
  g 1 1 = 7 := by
  sorry

end center_cell_is_seven_l2439_243960


namespace c_investment_value_l2439_243954

/-- Represents the investment and profit distribution in a partnership business -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Theorem stating that under the given conditions, C's investment is 9600 -/
theorem c_investment_value (p : Partnership)
  (h1 : p.a_investment = 2400)
  (h2 : p.b_investment = 7200)
  (h3 : p.total_profit = 9000)
  (h4 : p.a_profit_share = 1125)
  (h5 : p.a_profit_share * (p.a_investment + p.b_investment + p.c_investment) = p.a_investment * p.total_profit) :
  p.c_investment = 9600 := by
  sorry

#check c_investment_value

end c_investment_value_l2439_243954


namespace fixed_point_coordinates_l2439_243935

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation (3+k)x-2y+1-k=0 passes through the point A for any real k -/
def passes_through (A : Point) : Prop :=
  ∀ k : ℝ, (3 + k) * A.x - 2 * A.y + 1 - k = 0

/-- The fixed point A that the line passes through for all k has coordinates (1, 2) -/
theorem fixed_point_coordinates : 
  ∃ A : Point, passes_through A ∧ A.x = 1 ∧ A.y = 2 := by
  sorry

end fixed_point_coordinates_l2439_243935


namespace average_children_in_families_with_children_l2439_243951

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families * total_average) / (total_families - childless_families) = 3.75 := by
  sorry

end average_children_in_families_with_children_l2439_243951


namespace first_year_interest_rate_l2439_243902

/-- Proves that the first-year interest rate is 4% given the problem conditions --/
theorem first_year_interest_rate (initial_amount : ℝ) (final_amount : ℝ) (second_year_rate : ℝ) :
  initial_amount = 5000 →
  final_amount = 5460 →
  second_year_rate = 0.05 →
  ∃ (first_year_rate : ℝ),
    final_amount = initial_amount * (1 + first_year_rate) * (1 + second_year_rate) ∧
    first_year_rate = 0.04 :=
by sorry

end first_year_interest_rate_l2439_243902


namespace z_remainder_when_z_plus_3_div_9_is_integer_l2439_243966

theorem z_remainder_when_z_plus_3_div_9_is_integer (z : ℤ) :
  (∃ k : ℤ, (z + 3) / 9 = k) → z ≡ 6 [ZMOD 9] := by
  sorry

end z_remainder_when_z_plus_3_div_9_is_integer_l2439_243966


namespace other_endpoint_of_line_segment_l2439_243950

/-- Given a line segment with midpoint (2, 3) and one endpoint (5, -1), prove that the other endpoint is (-1, 7) -/
theorem other_endpoint_of_line_segment (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (2, 3) → endpoint1 = (5, -1) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-1, 7) := by
sorry

end other_endpoint_of_line_segment_l2439_243950


namespace board_numbers_product_l2439_243965

def pairwise_sums (a b c d e : ℤ) : Finset ℤ :=
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e}

theorem board_numbers_product (a b c d e : ℤ) :
  pairwise_sums a b c d e = {-1, 4, 6, 9, 10, 11, 15, 16, 20, 22} →
  a * b * c * d * e = -4914 := by
  sorry

end board_numbers_product_l2439_243965


namespace function_property_l2439_243990

-- Define the function f and its property
def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y)) = f (x + y) + f x * f y - x * y

-- Define α
def α (f : ℝ → ℝ) : ℝ := f 0

-- Theorem statement
theorem function_property (f : ℝ → ℝ) (h : f_property f) :
  (f (α f) * f (-(α f)) = 0) ∧
  (α f = 0) ∧
  (∀ x : ℝ, f x = x) :=
by sorry

end function_property_l2439_243990


namespace base_6_sum_theorem_l2439_243986

/-- Represents a base-6 number with three digits --/
def Base6Number (a b c : Nat) : Nat :=
  a * 36 + b * 6 + c

/-- Checks if a number is a valid base-6 digit (1-5) --/
def IsValidBase6Digit (n : Nat) : Prop :=
  0 < n ∧ n < 6

/-- The main theorem --/
theorem base_6_sum_theorem (A B C : Nat) 
  (h1 : IsValidBase6Digit A)
  (h2 : IsValidBase6Digit B)
  (h3 : IsValidBase6Digit C)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : Base6Number A B C + Base6Number B C 0 = Base6Number A C A) :
  A + B + C = Base6Number 1 5 0 := by
  sorry

#check base_6_sum_theorem

end base_6_sum_theorem_l2439_243986


namespace quadratic_inequality_l2439_243942

theorem quadratic_inequality (a b c A B C : ℝ) (ha : a ≠ 0) (hA : A ≠ 0)
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4*a*c| ≤ |B^2 - 4*A*C| := by
  sorry

end quadratic_inequality_l2439_243942


namespace factorization_equality_l2439_243940

theorem factorization_equality (m a : ℝ) : m * a^2 - m = m * (a + 1) * (a - 1) := by
  sorry

end factorization_equality_l2439_243940


namespace line_parametric_equation_l2439_243953

/-- The standard parametric equation of a line passing through a point with a given slope angle. -/
theorem line_parametric_equation (P : ℝ × ℝ) (θ : ℝ) :
  P = (1, -1) → θ = π / 3 →
  ∃ f g : ℝ → ℝ, 
    (∀ t, f t = 1 + (1/2) * t) ∧ 
    (∀ t, g t = -1 + (Real.sqrt 3 / 2) * t) ∧
    (∀ t, (f t, g t) ∈ {(x, y) | y - P.2 = Real.tan θ * (x - P.1)}) :=
sorry

end line_parametric_equation_l2439_243953


namespace f_max_on_interval_f_min_on_interval_f_max_attained_f_min_attained_l2439_243903

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f satisfies f(x+y) = f(x) + f(y) for all x, y -/
axiom f_additive : ∀ x y, f (x + y) = f x + f y

/-- f(x) < 0 when x > 0 -/
axiom f_neg_when_pos : ∀ x, x > 0 → f x < 0

/-- f(1) = -2 -/
axiom f_one : f 1 = -2

/-- The maximum value of f on [-3, 3] is 6 -/
theorem f_max_on_interval : 
  ∀ x, x ∈ Set.Icc (-3) 3 → f x ≤ 6 :=
sorry

/-- The minimum value of f on [-3, 3] is -6 -/
theorem f_min_on_interval : 
  ∀ x, x ∈ Set.Icc (-3) 3 → f x ≥ -6 :=
sorry

/-- The maximum value 6 is attained at -3 -/
theorem f_max_attained : f (-3) = 6 :=
sorry

/-- The minimum value -6 is attained at 3 -/
theorem f_min_attained : f 3 = -6 :=
sorry

end f_max_on_interval_f_min_on_interval_f_max_attained_f_min_attained_l2439_243903


namespace tons_to_kilograms_l2439_243988

-- Define the mass units
def ton : ℝ := 1000
def kilogram : ℝ := 1

-- State the theorem
theorem tons_to_kilograms : 24 * ton = 24000 * kilogram := by sorry

end tons_to_kilograms_l2439_243988


namespace blue_then_green_probability_l2439_243993

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  total_eq : sides = red + blue + yellow + green

/-- The probability of an event occurring -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

/-- The probability of two independent events occurring in sequence -/
def sequential_probability (p1 : ℚ) (p2 : ℚ) : ℚ :=
  p1 * p2

theorem blue_then_green_probability (d : ColoredDie) 
  (h : d = ⟨12, 5, 4, 2, 1, rfl⟩) : 
  sequential_probability (probability d.blue d.sides) (probability d.green d.sides) = 1/36 := by
  sorry

end blue_then_green_probability_l2439_243993


namespace add_fractions_three_ninths_seven_twelfths_l2439_243926

theorem add_fractions_three_ninths_seven_twelfths :
  3 / 9 + 7 / 12 = 11 / 12 := by sorry

end add_fractions_three_ninths_seven_twelfths_l2439_243926


namespace quadratic_rewrite_l2439_243989

theorem quadratic_rewrite (x : ℝ) :
  ∃ m : ℝ, 4 * x^2 - 16 * x - 448 = (x + m)^2 - 116 := by
  sorry

end quadratic_rewrite_l2439_243989


namespace proportion_and_equation_imply_c_value_l2439_243998

theorem proportion_and_equation_imply_c_value 
  (a b c : ℝ) 
  (h1 : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 7*k) 
  (h2 : a - b + 3 = c - 2*b) : 
  c = 21/2 := by
  sorry

end proportion_and_equation_imply_c_value_l2439_243998


namespace inequality_proof_l2439_243948

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / (a * b) + (b^2 + c^2) / (b * c) + (c^2 + a^2) / (c * a) ≥ 6 ∧
  (a + b) / 2 * (b + c) / 2 * (c + a) / 2 ≥ a * b * c := by
  sorry

end inequality_proof_l2439_243948


namespace system_solution_l2439_243963

theorem system_solution (x y : ℝ) (dot star : ℝ) : 
  (2 * x + y = dot ∧ 2 * x - y = 12 ∧ x = 5 ∧ y = star) → 
  (dot = 8 ∧ star = -2) := by
  sorry

end system_solution_l2439_243963


namespace ellipse_hyperbola_eccentricity_l2439_243949

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    prove that the eccentricity of the corresponding hyperbola is sqrt(5)/2 -/
theorem ellipse_hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 3/4) : 
  let c := Real.sqrt (a^2 + b^2)
  (c / a) = Real.sqrt 5 / 2 := by sorry

end ellipse_hyperbola_eccentricity_l2439_243949


namespace geometric_sequence_ratio_l2439_243992

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with ratio q
  a 1 = 2 →                     -- a_1 = 2
  (a 1 + a 2 + a 3 = 6) →       -- S_3 = 6
  (q = 1 ∨ q = -2) :=           -- q = 1 or q = -2
by sorry

end geometric_sequence_ratio_l2439_243992


namespace largest_square_with_four_interior_lattice_points_l2439_243901

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square in the plane -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Counts the number of lattice points strictly inside a square -/
def count_interior_lattice_points (s : Square) : ℕ :=
  sorry

/-- The theorem stating the area of the largest square with exactly 4 interior lattice points -/
theorem largest_square_with_four_interior_lattice_points :
  ∃ (s : Square),
    (count_interior_lattice_points s = 4) ∧
    (∀ (t : Square), count_interior_lattice_points t = 4 → t.side_length ≤ s.side_length) ∧
    (9 < s.side_length ^ 2) ∧ (s.side_length ^ 2 < 10) :=
  sorry

end largest_square_with_four_interior_lattice_points_l2439_243901


namespace solve_for_n_l2439_243918

theorem solve_for_n (P s k m n : ℝ) (h : P = s / (1 + k + m) ^ n) :
  n = Real.log (s / P) / Real.log (1 + k + m) :=
by sorry

end solve_for_n_l2439_243918


namespace double_markup_percentage_l2439_243914

theorem double_markup_percentage (initial_price : ℝ) (markup_percentage : ℝ) : 
  markup_percentage = 40 →
  let first_markup := initial_price * (1 + markup_percentage / 100)
  let second_markup := first_markup * (1 + markup_percentage / 100)
  (second_markup - initial_price) / initial_price * 100 = 96 := by
  sorry

end double_markup_percentage_l2439_243914


namespace cos_two_alpha_value_l2439_243931

theorem cos_two_alpha_value (α : Real) (h1 : π/8 < α) (h2 : α < 3*π/8) : 
  let f := fun x => Real.cos x * (Real.sin x + Real.cos x) - 1/2
  f α = Real.sqrt 2 / 6 → Real.cos (2 * α) = (Real.sqrt 2 - 4) / 6 := by
  sorry

end cos_two_alpha_value_l2439_243931


namespace solution_set_implies_a_equals_one_l2439_243961

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + a

-- Define the theorem
theorem solution_set_implies_a_equals_one :
  (∃ (a : ℝ), ∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) →
  (∃ (a : ℝ), a = 1 ∧ ∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) :=
by sorry


end solution_set_implies_a_equals_one_l2439_243961


namespace range_of_g_l2439_243941

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.cos x ^ 4 + Real.sin x ^ 2 ∧ Real.cos x ^ 4 + Real.sin x ^ 2 ≤ 1 := by
  sorry

end range_of_g_l2439_243941


namespace find_b_value_l2439_243956

theorem find_b_value (a b : ℝ) (eq1 : 3 * a + 2 = 2) (eq2 : b - 2 * a = 2) : b = 2 := by
  sorry

end find_b_value_l2439_243956


namespace modified_fibonacci_sum_l2439_243952

-- Define the modified Fibonacci sequence
def F : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => F (n + 1) + F n

-- Define the sum of the series
noncomputable def S : ℝ := ∑' n, (F n : ℝ) / 5^n

-- Theorem statement
theorem modified_fibonacci_sum : S = 10 / 19 := by sorry

end modified_fibonacci_sum_l2439_243952


namespace fraction_leading_zeros_l2439_243913

/-- The number of leading zeros in the decimal representation of a rational number -/
def leadingZeros (q : ℚ) : ℕ := sorry

/-- The fraction we're analyzing -/
def fraction : ℚ := 1 / (2^7 * 5^9)

theorem fraction_leading_zeros : leadingZeros fraction = 8 := by sorry

end fraction_leading_zeros_l2439_243913
