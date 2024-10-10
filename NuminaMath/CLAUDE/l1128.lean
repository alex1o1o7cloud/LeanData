import Mathlib

namespace edwards_toy_purchase_l1128_112825

/-- Proves that given an initial amount of $17.80, after purchasing 4 items at $0.95 each
    and one item at $6.00, the remaining amount is $8.00. -/
theorem edwards_toy_purchase (initial_amount : ℚ) (toy_car_price : ℚ) (race_track_price : ℚ)
    (num_toy_cars : ℕ) (h1 : initial_amount = 17.8)
    (h2 : toy_car_price = 0.95) (h3 : race_track_price = 6)
    (h4 : num_toy_cars = 4) : 
    initial_amount - (toy_car_price * num_toy_cars + race_track_price) = 8 := by
  sorry

end edwards_toy_purchase_l1128_112825


namespace circle_center_sum_l1128_112813

/-- The sum of the x and y coordinates of the center of a circle
    described by the equation x^2 + y^2 = 6x + 6y - 30 is equal to 6 -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 6*x + 6*y - 30) → ∃ h k : ℝ, (h + k = 6 ∧ (x - h)^2 + (y - k)^2 = (x^2 + y^2 - 6*x - 6*y + 30)) :=
by sorry

end circle_center_sum_l1128_112813


namespace committee_probability_l1128_112826

/-- The probability of selecting a committee with at least one boy and one girl -/
theorem committee_probability (total : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) : 
  total = 30 →
  boys = 12 →
  girls = 18 →
  committee_size = 6 →
  (1 : ℚ) - (Nat.choose boys committee_size + Nat.choose girls committee_size : ℚ) / 
    (Nat.choose total committee_size : ℚ) = 574287 / 593775 := by
  sorry

#check committee_probability

end committee_probability_l1128_112826


namespace gravelling_cost_theorem_l1128_112815

/-- The cost of gravelling a path around a rectangular plot -/
theorem gravelling_cost_theorem 
  (plot_length : ℝ) 
  (plot_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm_paise : ℝ) 
  (h1 : plot_length = 110) 
  (h2 : plot_width = 65) 
  (h3 : path_width = 2.5) 
  (h4 : cost_per_sqm_paise = 60) : 
  (((plot_length * plot_width) - ((plot_length - 2 * path_width) * (plot_width - 2 * path_width))) * (cost_per_sqm_paise / 100)) = 510 := by
  sorry

end gravelling_cost_theorem_l1128_112815


namespace largest_x_quadratic_inequality_l1128_112886

theorem largest_x_quadratic_inequality :
  ∀ x : ℝ, x^2 - 10*x + 24 ≤ 0 → x ≤ 6 :=
by sorry

end largest_x_quadratic_inequality_l1128_112886


namespace smallest_base_for_mnmn_cube_mnmn_cube_in_base_seven_smallest_base_is_seven_l1128_112804

def is_mnmn (n : ℕ) (b : ℕ) : Prop :=
  ∃ m n : ℕ, m < b ∧ n < b ∧ n = m * (b^3 + b) + n * (b^2 + 1)

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

theorem smallest_base_for_mnmn_cube :
  ∀ b : ℕ, b > 1 →
    (∃ n : ℕ, is_mnmn n b ∧ is_cube n) →
    b ≥ 7 :=
by sorry

theorem mnmn_cube_in_base_seven :
  ∃ n : ℕ, is_mnmn n 7 ∧ is_cube n :=
by sorry

theorem smallest_base_is_seven :
  (∀ b : ℕ, b > 1 → b < 7 → ¬∃ n : ℕ, is_mnmn n b ∧ is_cube n) ∧
  (∃ n : ℕ, is_mnmn n 7 ∧ is_cube n) :=
by sorry

end smallest_base_for_mnmn_cube_mnmn_cube_in_base_seven_smallest_base_is_seven_l1128_112804


namespace batsman_average_after_15th_inning_l1128_112820

def batsman_average (total_innings : ℕ) (last_inning_score : ℕ) (average_increase : ℕ) : ℚ :=
  let previous_average := (total_innings - 1 : ℚ) * (average_increase : ℚ) + (last_inning_score : ℚ) / (total_innings : ℚ)
  previous_average + average_increase

theorem batsman_average_after_15th_inning :
  batsman_average 15 75 3 = 33 := by
  sorry

end batsman_average_after_15th_inning_l1128_112820


namespace angle_at_point_l1128_112869

theorem angle_at_point (y : ℝ) : 
  y > 0 ∧ y + y + 140 = 360 → y = 110 := by
  sorry

end angle_at_point_l1128_112869


namespace new_cube_volume_l1128_112880

theorem new_cube_volume (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 64 →
  scale_factor = 2 →
  (scale_factor ^ 3) * original_volume = 512 :=
by sorry

end new_cube_volume_l1128_112880


namespace chips_division_l1128_112874

theorem chips_division (total_chips : ℕ) (ratio_small : ℕ) (ratio_large : ℕ) :
  total_chips = 100 →
  ratio_small = 4 →
  ratio_large = 6 →
  (ratio_large : ℚ) / (ratio_small + ratio_large : ℚ) * 100 = 60 := by
  sorry

end chips_division_l1128_112874


namespace complex_fraction_equality_l1128_112862

theorem complex_fraction_equality : (2 - I) / (1 - I) = 3/2 + I/2 := by
  sorry

end complex_fraction_equality_l1128_112862


namespace initial_salmon_count_l1128_112814

theorem initial_salmon_count (current_count : ℕ) (increase_factor : ℕ) (initial_count : ℕ) : 
  current_count = 5500 →
  increase_factor = 10 →
  current_count = (increase_factor + 1) * initial_count →
  initial_count = 550 := by
  sorry

end initial_salmon_count_l1128_112814


namespace correct_stratified_sample_l1128_112891

/-- Represents the number of water heaters in a sample from a specific factory -/
structure FactorySample where
  total : ℕ
  factory_a : ℕ
  factory_b : ℕ
  sample_size : ℕ

/-- Calculates the stratified sample size for a factory -/
def stratified_sample_size (total : ℕ) (factory : ℕ) (sample_size : ℕ) : ℕ :=
  (factory * sample_size) / total

/-- Theorem stating the correct stratified sample sizes for factories A and B -/
theorem correct_stratified_sample (fs : FactorySample) 
  (h1 : fs.total = 98)
  (h2 : fs.factory_a = 56)
  (h3 : fs.factory_b = 42)
  (h4 : fs.sample_size = 14) :
  stratified_sample_size fs.total fs.factory_a fs.sample_size = 8 ∧
  stratified_sample_size fs.total fs.factory_b fs.sample_size = 6 := by
  sorry

end correct_stratified_sample_l1128_112891


namespace line_segment_length_l1128_112855

/-- The length of a line segment with endpoints (1, 2) and (8, 6) is √65 -/
theorem line_segment_length : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (8, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 65 := by
  sorry

end line_segment_length_l1128_112855


namespace max_chickens_and_chicks_max_chicks_no_chickens_l1128_112881

/-- Represents the chicken coop problem -/
structure ChickenCoop where
  area : ℝ
  chicken_space : ℝ
  chick_space : ℝ
  chicken_feed : ℝ
  chick_feed : ℝ
  max_feed : ℝ

/-- Defines the specific chicken coop instance -/
def our_coop : ChickenCoop :=
  { area := 240
  , chicken_space := 4
  , chick_space := 2
  , chicken_feed := 160
  , chick_feed := 40
  , max_feed := 8000 }

/-- Theorem stating the maximum number of chickens and chicks -/
theorem max_chickens_and_chicks (coop : ChickenCoop) :
  ∃ (x y : ℕ), 
    x * coop.chicken_space + y * coop.chick_space = coop.area ∧
    x * coop.chicken_feed + y * coop.chick_feed ≤ coop.max_feed ∧
    x = 40 ∧ y = 40 ∧
    ∀ (a b : ℕ), 
      a * coop.chicken_space + b * coop.chick_space = coop.area →
      a * coop.chicken_feed + b * coop.chick_feed ≤ coop.max_feed →
      a ≤ x :=
by sorry

/-- Theorem stating the maximum number of chicks with no chickens -/
theorem max_chicks_no_chickens (coop : ChickenCoop) :
  ∃ (y : ℕ),
    y * coop.chick_space = coop.area ∧
    y * coop.chick_feed ≤ coop.max_feed ∧
    y = 120 ∧
    ∀ (b : ℕ),
      b * coop.chick_space = coop.area →
      b * coop.chick_feed ≤ coop.max_feed →
      b ≤ y :=
by sorry

end max_chickens_and_chicks_max_chicks_no_chickens_l1128_112881


namespace holds_age_ratio_l1128_112837

/-- Proves that the ratio of Hold's age to her son's age today is 3:1 -/
theorem holds_age_ratio : 
  ∀ (hold_age_today hold_age_8_years_ago son_age_today son_age_8_years_ago : ℕ),
  hold_age_today = 36 →
  hold_age_8_years_ago = hold_age_today - 8 →
  son_age_8_years_ago = son_age_today - 8 →
  hold_age_8_years_ago = 7 * son_age_8_years_ago →
  (hold_age_today : ℚ) / son_age_today = 3 := by
sorry

end holds_age_ratio_l1128_112837


namespace sphere_radius_when_area_equals_volume_l1128_112848

theorem sphere_radius_when_area_equals_volume (r : ℝ) (h : r > 0) :
  (4 * Real.pi * r^2) = (4/3 * Real.pi * r^3) → r = 3 := by
  sorry

end sphere_radius_when_area_equals_volume_l1128_112848


namespace at_least_one_not_in_area_l1128_112849

theorem at_least_one_not_in_area (p q : Prop) : 
  (¬p ∨ ¬q) ↔ (∃ trainee, trainee = "A" ∧ ¬p ∨ trainee = "B" ∧ ¬q) :=
sorry

end at_least_one_not_in_area_l1128_112849


namespace fortieth_term_is_81_l1128_112836

/-- An arithmetic sequence starting from 3 with common difference 2 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- The 40th term of the arithmetic sequence is 81 -/
theorem fortieth_term_is_81 : arithmeticSequence 40 = 81 := by
  sorry

end fortieth_term_is_81_l1128_112836


namespace bottle_cap_distribution_l1128_112866

theorem bottle_cap_distribution (initial : ℕ) (rebecca : ℕ) (siblings : ℕ) : 
  initial = 150 →
  rebecca = 42 →
  siblings = 5 →
  (initial + rebecca + 2 * rebecca) / (siblings + 1) = 46 :=
by sorry

end bottle_cap_distribution_l1128_112866


namespace exponential_function_through_point_l1128_112808

theorem exponential_function_through_point (f : ℝ → ℝ) :
  (∀ x : ℝ, ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ f x = a^x) →
  f 1 = 2 →
  f 2 = 4 := by
sorry

end exponential_function_through_point_l1128_112808


namespace complex_equation_solution_l1128_112817

theorem complex_equation_solution (a : ℝ) : 
  Complex.abs (a - 2 + (4 + 3 * Complex.I) / (1 + 2 * Complex.I)) = Real.sqrt 3 * a → 
  a = Real.sqrt 2 / 2 := by
sorry

end complex_equation_solution_l1128_112817


namespace gcd_factorial_seven_eight_l1128_112857

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcd_factorial_seven_eight_l1128_112857


namespace potatoes_per_bag_l1128_112830

/-- Proves that the number of pounds of potatoes in one bag is 20 -/
theorem potatoes_per_bag (potatoes_per_person : ℝ) (num_people : ℕ) (cost_per_bag : ℝ) (total_cost : ℝ) :
  potatoes_per_person = 1.5 →
  num_people = 40 →
  cost_per_bag = 5 →
  total_cost = 15 →
  (num_people * potatoes_per_person) / (total_cost / cost_per_bag) = 20 := by
  sorry

end potatoes_per_bag_l1128_112830


namespace A_prime_div_B_prime_l1128_112806

/-- The series A' as defined in the problem -/
noncomputable def A' : ℝ := ∑' n, if n % 5 ≠ 0 ∧ n % 2 ≠ 0 then ((-1) ^ ((n - 1) / 2 : ℕ)) / n^2 else 0

/-- The series B' as defined in the problem -/
noncomputable def B' : ℝ := ∑' n, if n % 5 = 0 ∧ n % 2 ≠ 0 then ((-1) ^ ((n / 5 - 1) / 2 : ℕ)) / n^2 else 0

/-- The main theorem stating that A' / B' = 26 -/
theorem A_prime_div_B_prime : A' / B' = 26 := by
  sorry

end A_prime_div_B_prime_l1128_112806


namespace unknown_number_value_l1128_112860

theorem unknown_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 45 * 25) : x = 49 := by
  sorry

end unknown_number_value_l1128_112860


namespace jewelry_sales_fraction_l1128_112824

theorem jewelry_sales_fraction (total_sales : ℕ) (stationery_sales : ℕ) :
  total_sales = 36 →
  stationery_sales = 15 →
  (total_sales : ℚ) / 3 + stationery_sales + (total_sales : ℚ) / 4 = total_sales :=
by
  sorry

end jewelry_sales_fraction_l1128_112824


namespace constant_term_of_expansion_l1128_112818

theorem constant_term_of_expansion (x : ℝ) (x_pos : x > 0) : 
  ∃ (c : ℝ), c = 240 ∧ 
  ∀ (k : ℕ), k ≤ 6 → 
    (Nat.choose 6 k * (2^k) * x^(6 - 3/2 * k : ℝ) = c ↔ k = 4) :=
sorry

end constant_term_of_expansion_l1128_112818


namespace part1_part2_l1128_112839

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + 2*b*x + c

-- Part 1
theorem part1 (b c : ℝ) : 
  (∀ x, f b c x = 0 ↔ x = -1 ∨ x = 1) → b = 0 ∧ c = -1 := by sorry

-- Part 2
theorem part2 (b : ℝ) :
  (∃ x₁ x₂, f b (b^2 + 2*b + 3) x₁ = 0 ∧ f b (b^2 + 2*b + 3) x₂ = 0 ∧ (x₁ + 1)*(x₂ + 1) = 8) →
  b = -2 := by sorry

end part1_part2_l1128_112839


namespace oranges_packed_in_week_l1128_112838

/-- The number of oranges packed in a full week given the daily packing rate and box capacity -/
theorem oranges_packed_in_week
  (oranges_per_box : ℕ)
  (boxes_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : oranges_per_box = 15)
  (h2 : boxes_per_day = 2150)
  (h3 : days_in_week = 7) :
  oranges_per_box * boxes_per_day * days_in_week = 225750 := by
  sorry

end oranges_packed_in_week_l1128_112838


namespace investment_total_l1128_112835

theorem investment_total (rate1 rate2 amount1 amount2 total_income : ℚ)
  (h1 : rate1 = 85 / 1000)
  (h2 : rate2 = 64 / 1000)
  (h3 : amount1 = 3000)
  (h4 : amount2 = 5000)
  (h5 : rate1 * amount1 + rate2 * amount2 = 575) :
  amount1 + amount2 = 8000 :=
by sorry

end investment_total_l1128_112835


namespace select_and_order_two_from_five_eq_twenty_l1128_112842

/-- The number of ways to select and order 2 items from a set of 5 distinct items -/
def select_and_order_two_from_five : ℕ :=
  5 * 4

/-- Theorem: The number of ways to select and order 2 items from a set of 5 distinct items is 20 -/
theorem select_and_order_two_from_five_eq_twenty :
  select_and_order_two_from_five = 20 := by
  sorry

end select_and_order_two_from_five_eq_twenty_l1128_112842


namespace division_calculation_l1128_112873

theorem division_calculation : (6 : ℚ) / (-1/2 + 1/3) = -36 := by sorry

end division_calculation_l1128_112873


namespace arithmetic_geometric_sequence_ratio_l1128_112822

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℚ) (d : ℚ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)  -- arithmetic sequence definition
  (h3 : (a 3 - a 1) * (a 9 - a 3) = (a 3 - a 1)^2) :  -- geometric sequence condition
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13/16 := by
sorry

end arithmetic_geometric_sequence_ratio_l1128_112822


namespace vector_problem_l1128_112831

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v = t • w

theorem vector_problem :
  (∃ k : ℝ, perpendicular (k • a + b) (a - 3 • b) ∧ k = 19) ∧
  (∃ k : ℝ, parallel (k • a + b) (a - 3 • b) ∧ k = -1/3) :=
sorry

end vector_problem_l1128_112831


namespace impossibleToTileModifiedChessboard_l1128_112883

/-- Represents a square on the chessboard -/
inductive Square
| Black
| White

/-- Represents the chessboard -/
def Chessboard := Array (Array Square)

/-- Creates a standard 8x8 chessboard -/
def createStandardChessboard : Chessboard :=
  sorry

/-- Removes the top-left and bottom-right squares from the chessboard -/
def removeCornerSquares (board : Chessboard) : Chessboard :=
  sorry

/-- Counts the number of black and white squares on the chessboard -/
def countSquares (board : Chessboard) : (Nat × Nat) :=
  sorry

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement where
  position1 : Nat × Nat
  position2 : Nat × Nat

/-- Checks if a domino placement is valid on the given chessboard -/
def isValidPlacement (board : Chessboard) (placement : DominoPlacement) : Bool :=
  sorry

/-- Main theorem: It's impossible to tile the modified chessboard with dominos -/
theorem impossibleToTileModifiedChessboard :
  ∀ (placements : List DominoPlacement),
    let board := removeCornerSquares createStandardChessboard
    let (blackCount, whiteCount) := countSquares board
    (blackCount ≠ whiteCount) ∧
    (∀ p ∈ placements, isValidPlacement board p) →
    placements.length < 31 :=
  sorry

end impossibleToTileModifiedChessboard_l1128_112883


namespace three_isosceles_right_triangles_l1128_112859

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 2 * x^2 + 4 * x - y^2 = 0

-- Define an isosceles right triangle with O as the right angle
def isosceles_right_triangle (A B : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  xA * xB + yA * yB = 0 ∧ xA^2 + yA^2 = xB^2 + yB^2

-- Main theorem
theorem three_isosceles_right_triangles :
  ∃ (S : Finset (ℝ × ℝ)),
    Finset.card S = 3 ∧
    (∀ A ∈ S, hyperbola A.1 A.2) ∧
    (∀ A B, A ∈ S → B ∈ S → A ≠ B → isosceles_right_triangle A B) ∧
    (∀ A B, hyperbola A.1 A.2 → hyperbola B.1 B.2 → 
      isosceles_right_triangle A B → (A ∈ S ∧ B ∈ S)) :=
sorry

end three_isosceles_right_triangles_l1128_112859


namespace johns_age_l1128_112854

theorem johns_age (age : ℕ) : 
  (age + 9 = 3 * (age - 11)) → age = 21 := by
  sorry

end johns_age_l1128_112854


namespace boat_current_speed_l1128_112884

/-- Given a boat traveling downstream at 15 km/h, and the distance traveled downstream
    in 4 hours equals the distance traveled upstream in 5 hours, prove that the speed
    of the water current is 1.5 km/h. -/
theorem boat_current_speed (v_d : ℝ) (t_d t_u : ℝ) (h1 : v_d = 15)
    (h2 : t_d = 4) (h3 : t_u = 5) (h4 : v_d * t_d = (2 * v_d - 15) * t_u / 2) :
    ∃ v_c : ℝ, v_c = 1.5 ∧ v_d = v_c + (2 * v_d - 15) / 2 := by
  sorry

end boat_current_speed_l1128_112884


namespace system_two_solutions_l1128_112805

theorem system_two_solutions (a : ℝ) :
  (∃! x y, a^2 - 2*a*x - 6*y + x^2 + y^2 = 0 ∧ (|x| - 4)^2 + (|y| - 3)^2 = 25) ↔
  a ∈ Set.Ioo (-12) (-6) ∪ {0} ∪ Set.Ioo 6 12 :=
by sorry

end system_two_solutions_l1128_112805


namespace quadratic_inequality_implies_a_geq_one_l1128_112872

theorem quadratic_inequality_implies_a_geq_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) → a ≥ 1 := by
  sorry

end quadratic_inequality_implies_a_geq_one_l1128_112872


namespace symmetry_axis_l1128_112899

-- Define a function f with the given property
def f (x : ℝ) : ℝ := sorry

-- State the condition that f(x) = f(3 - x) for all x
axiom f_symmetry (x : ℝ) : f x = f (3 - x)

-- Define the concept of an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem stating that x = 1.5 is an axis of symmetry for f
theorem symmetry_axis : is_axis_of_symmetry 1.5 f := by sorry

end symmetry_axis_l1128_112899


namespace f_of_one_plus_g_of_two_l1128_112876

def f (x : ℝ) : ℝ := 2 * x - 3

def g (x : ℝ) : ℝ := x + 1

theorem f_of_one_plus_g_of_two : f (1 + g 2) = 5 := by sorry

end f_of_one_plus_g_of_two_l1128_112876


namespace panda_babies_count_l1128_112843

def total_couples : ℕ := 100
def young_percentage : ℚ := 1/5
def adult_percentage : ℚ := 3/5
def old_percentage : ℚ := 1/5

def young_pregnancy_chance : ℚ := 2/5
def adult_pregnancy_chance : ℚ := 1/4
def old_pregnancy_chance : ℚ := 1/10

def average_babies_per_pregnancy : ℚ := 3/2

def young_babies : ℕ := 12
def adult_babies : ℕ := 22
def old_babies : ℕ := 3

theorem panda_babies_count :
  young_babies + adult_babies + old_babies = 37 :=
by sorry

end panda_babies_count_l1128_112843


namespace area_Ω_bound_l1128_112858

/-- Parabola C: y = (1/2)x^2 -/
def parabola_C (x y : ℝ) : Prop := y = (1/2) * x^2

/-- Circle D: x^2 + (y - 1/2)^2 = r^2, where r > 0 -/
def circle_D (x y r : ℝ) : Prop := x^2 + (y - 1/2)^2 = r^2 ∧ r > 0

/-- C and D have no common points -/
def no_intersection (r : ℝ) : Prop := ∀ x y : ℝ, parabola_C x y → ¬(circle_D x y r)

/-- Point A is on parabola C -/
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.1 A.2

/-- Region Ω formed by tangents from A to D -/
def region_Ω (r : ℝ) (A : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Area of region Ω -/
noncomputable def area_Ω (r : ℝ) (A : ℝ × ℝ) : ℝ := sorry

theorem area_Ω_bound (r : ℝ) (h : no_intersection r) :
  ∀ A : ℝ × ℝ, point_on_parabola A →
    0 < area_Ω r A ∧ area_Ω r A < π/16 := by sorry

end area_Ω_bound_l1128_112858


namespace max_abs_z_purely_imaginary_l1128_112890

theorem max_abs_z_purely_imaginary (z : ℂ) :
  (∃ (t : ℝ), (z - Complex.I) / (z - 1) = Complex.I * t) → Complex.abs z ≤ Real.sqrt 2 := by
  sorry

end max_abs_z_purely_imaginary_l1128_112890


namespace product_1011_2_112_3_l1128_112840

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.reverse.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a ternary number represented as a list of digits to its decimal equivalent -/
def ternary_to_decimal (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The main theorem stating that the product of 1011₂ and 112₃ in base 10 is 154 -/
theorem product_1011_2_112_3 : 
  (binary_to_decimal [true, true, false, true]) * 
  (ternary_to_decimal [2, 1, 1]) = 154 := by
  sorry

#eval binary_to_decimal [true, true, false, true]  -- Should output 11
#eval ternary_to_decimal [2, 1, 1]  -- Should output 14

end product_1011_2_112_3_l1128_112840


namespace isosceles_triangle_base_length_l1128_112865

/-- An isosceles triangle with perimeter 16 and one side of length 6 has a base of either 6 or 4. -/
theorem isosceles_triangle_base_length (a b : ℝ) : 
  a > 0 → b > 0 → 
  a + b + b = 16 → 
  (a = 6 ∨ b = 6) → 
  (a = 6 ∧ b = 5) ∨ (a = 4 ∧ b = 6) :=
by sorry

end isosceles_triangle_base_length_l1128_112865


namespace chocolate_bar_calculation_l1128_112803

/-- The number of chocolate bars in each box -/
def bars_per_box : ℕ := 5

/-- The number of boxes Tom needs to sell -/
def boxes_to_sell : ℕ := 170

/-- The total number of chocolate bars Tom needs to sell -/
def total_bars : ℕ := bars_per_box * boxes_to_sell

theorem chocolate_bar_calculation :
  total_bars = 850 := by sorry

end chocolate_bar_calculation_l1128_112803


namespace diagonal_game_winner_l1128_112833

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the outcome of the game -/
inductive Outcome
| FirstPlayerWins
| SecondPlayerWins

/-- The number of diagonals in a polygon with s sides -/
def num_diagonals (s : ℕ) : ℕ := s * (s - 3) / 2

/-- The winner of the diagonal drawing game in a (2n+1)-gon -/
def winner (n : ℕ) : Outcome :=
  if n % 2 = 0 then Outcome.FirstPlayerWins else Outcome.SecondPlayerWins

/-- The main theorem about the winner of the diagonal drawing game -/
theorem diagonal_game_winner (n : ℕ) (h : n > 1) :
  winner n = (if n % 2 = 0 then Outcome.FirstPlayerWins else Outcome.SecondPlayerWins) :=
sorry

end diagonal_game_winner_l1128_112833


namespace magnitude_of_z_l1128_112809

theorem magnitude_of_z (z : ℂ) (h : z * (1 - 2*Complex.I) = 4 + 2*Complex.I) : 
  Complex.abs z = 2 := by
  sorry

end magnitude_of_z_l1128_112809


namespace conference_attendees_l1128_112800

theorem conference_attendees (total : ℕ) (first_known : ℕ) : 
  total = 47 → first_known = 16 → 
  ∃ (women men : ℕ), 
    women + men = total ∧ 
    men = first_known + (women - 1) ∧
    women = 16 ∧ 
    men = 31 := by
  sorry

end conference_attendees_l1128_112800


namespace trivia_game_points_per_question_l1128_112896

theorem trivia_game_points_per_question 
  (first_half_correct : ℕ) 
  (second_half_correct : ℕ) 
  (final_score : ℕ) 
  (h1 : first_half_correct = 6)
  (h2 : second_half_correct = 4)
  (h3 : final_score = 30) :
  final_score / (first_half_correct + second_half_correct) = 3 := by
  sorry

end trivia_game_points_per_question_l1128_112896


namespace sin_squared_3x_maximum_l1128_112863

open Real

theorem sin_squared_3x_maximum (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (3 * x) ^ 2) :
  ∃ x, x ∈ Set.Ioo 0 0.6 ∧ f x = 1 ∧ ∀ y ∈ Set.Ioo 0 0.6, f y ≤ f x :=
by sorry

end sin_squared_3x_maximum_l1128_112863


namespace simplify_square_roots_l1128_112819

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 49) = 5 / 2 := by
  sorry

end simplify_square_roots_l1128_112819


namespace cube_root_27_div_fourth_root_16_l1128_112877

theorem cube_root_27_div_fourth_root_16 : (27 ^ (1/3)) / (16 ^ (1/4)) = 3/2 := by
  sorry

end cube_root_27_div_fourth_root_16_l1128_112877


namespace flower_arrangement_count_l1128_112841

def num_flowers : ℕ := 5
def num_vases : ℕ := 3

theorem flower_arrangement_count :
  (num_flowers * (num_flowers - 1) * (num_flowers - 2)) = 60 := by
  sorry

end flower_arrangement_count_l1128_112841


namespace no_consecutive_squares_arithmetic_sequence_l1128_112810

theorem no_consecutive_squares_arithmetic_sequence :
  ∀ (x y z w : ℕ+), ¬∃ (d : ℝ),
    (y : ℝ)^2 = (x : ℝ)^2 + d ∧
    (z : ℝ)^2 = (y : ℝ)^2 + d ∧
    (w : ℝ)^2 = (z : ℝ)^2 + d :=
by sorry

end no_consecutive_squares_arithmetic_sequence_l1128_112810


namespace extreme_values_and_monotonicity_l1128_112887

-- Define the function f
def f (x m n : ℝ) : ℝ := 2 * x^3 + 3 * m * x^2 + 3 * n * x - 6

-- Define the derivative of f
def f' (x m n : ℝ) : ℝ := 6 * x^2 + 6 * m * x + 3 * n

theorem extreme_values_and_monotonicity :
  ∃ (m n : ℝ),
    (f' 1 m n = 0 ∧ f' 2 m n = 0) ∧
    (m = -3 ∧ n = 4) ∧
    (∀ x, x < 1 → (f' x m n > 0)) ∧
    (∀ x, 1 < x ∧ x < 2 → (f' x m n < 0)) ∧
    (∀ x, x > 2 → (f' x m n > 0)) :=
by sorry

end extreme_values_and_monotonicity_l1128_112887


namespace purchase_percentage_l1128_112878

/-- Given a 25% price increase and a net difference in expenditure of 20,
    prove that the percentage of the required amount purchased is 16%. -/
theorem purchase_percentage (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.25 * P
  let R := (500 : ℝ) / 31.25
  let new_expenditure := new_price * (R / 100) * Q
  P * Q - new_expenditure = 20 → R = 16 := by sorry

end purchase_percentage_l1128_112878


namespace cyclic_ratio_inequality_l1128_112812

theorem cyclic_ratio_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b^2 / a + c^2 / b + a^2 / c ≥ Real.sqrt (3 * (a^2 + b^2 + c^2)) := by
  sorry

end cyclic_ratio_inequality_l1128_112812


namespace arithmetic_geometric_sequence_l1128_112811

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_2, a_5 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  (a 2) ^ 2 = a 1 * a 5

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) (h_geom : geometric_subseq a) : 
  a 2 = 3 := by
sorry

end arithmetic_geometric_sequence_l1128_112811


namespace no_real_solutions_l1128_112882

theorem no_real_solutions : 
  ¬∃ (x : ℝ), (x ≠ 2) ∧ ((x^3 - 8) / (x - 2) = 3*x) := by sorry

end no_real_solutions_l1128_112882


namespace volume_of_second_cylinder_l1128_112834

/-- Given two cylinders with the same height and radii in the ratio 1:3, 
    if the volume of the first cylinder is 40 cc, 
    then the volume of the second cylinder is 360 cc. -/
theorem volume_of_second_cylinder 
  (h : ℝ) -- height of both cylinders
  (r₁ : ℝ) -- radius of the first cylinder
  (r₂ : ℝ) -- radius of the second cylinder
  (h_positive : h > 0)
  (r₁_positive : r₁ > 0)
  (ratio : r₂ = 3 * r₁) -- radii ratio condition
  (volume₁ : ℝ) -- volume of the first cylinder
  (h_volume₁ : volume₁ = Real.pi * r₁^2 * h) -- volume formula for the first cylinder
  (volume₁_value : volume₁ = 40) -- given volume of the first cylinder
  : Real.pi * r₂^2 * h = 360 := by
  sorry

end volume_of_second_cylinder_l1128_112834


namespace alcohol_water_ratio_three_fifths_l1128_112892

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol_fraction : ℚ
  water_fraction : ℚ
  sum_is_one : alcohol_fraction + water_fraction = 1

/-- The ratio of alcohol to water in a mixture -/
def alcohol_to_water_ratio (m : Mixture) : ℚ := m.alcohol_fraction / m.water_fraction

/-- Theorem stating that for a mixture with 3/5 alcohol and 2/5 water, 
    the ratio of alcohol to water is 3:2 -/
theorem alcohol_water_ratio_three_fifths 
  (m : Mixture) 
  (h1 : m.alcohol_fraction = 3/5) 
  (h2 : m.water_fraction = 2/5) : 
  alcohol_to_water_ratio m = 3/2 := by
  sorry


end alcohol_water_ratio_three_fifths_l1128_112892


namespace completing_square_result_l1128_112867

theorem completing_square_result (x : ℝ) :
  (x^2 - 6*x + 5 = 0) ↔ ((x - 3)^2 = 4) :=
by sorry

end completing_square_result_l1128_112867


namespace yankees_to_mets_ratio_l1128_112821

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The given conditions of the problem -/
def baseball_town_conditions (fans : FanCounts) : Prop :=
  fans.yankees + fans.mets + fans.red_sox = 360 ∧
  fans.mets = 96 ∧
  5 * fans.mets = 4 * fans.red_sox

/-- The theorem to be proved -/
theorem yankees_to_mets_ratio (fans : FanCounts) 
  (h : baseball_town_conditions fans) : 
  3 * fans.mets = 2 * fans.yankees :=
sorry

end yankees_to_mets_ratio_l1128_112821


namespace quadratic_function_inequality_max_l1128_112856

theorem quadratic_function_inequality_max (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≤ 2 * a * x + b) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧
    (∀ a b c : ℝ, b^2 / (a^2 + c^2) ≤ M) ∧
    (∃ a b c : ℝ, b^2 / (a^2 + c^2) = M)) :=
by sorry

end quadratic_function_inequality_max_l1128_112856


namespace triangle_side_length_l1128_112844

/-- A triangle with circumradius 1 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (circumradius : ℝ)
  (h_circumradius : circumradius = 1)

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The circle passing through two points and the orthocenter -/
def circle_through_points_and_orthocenter (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The center of a circle -/
def center (c : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem triangle_side_length (t : Triangle) :
  center (circle_through_points_and_orthocenter t) ∈ circumcircle t →
  distance t.A t.C = Real.sqrt 3 := by
  sorry

end triangle_side_length_l1128_112844


namespace max_perimeter_special_triangle_max_perimeter_achievable_l1128_112897

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The maximum perimeter of a triangle ABC where a^2 = b^2 + c^2 - bc and a = 2 is 6 -/
theorem max_perimeter_special_triangle :
  ∀ t : Triangle,
    t.a^2 = t.b^2 + t.c^2 - t.b * t.c →
    t.a = 2 →
    perimeter t ≤ 6 :=
by
  sorry

/-- Corollary: There exists a triangle satisfying the conditions with perimeter equal to 6 -/
theorem max_perimeter_achievable :
  ∃ t : Triangle,
    t.a^2 = t.b^2 + t.c^2 - t.b * t.c ∧
    t.a = 2 ∧
    perimeter t = 6 :=
by
  sorry

end max_perimeter_special_triangle_max_perimeter_achievable_l1128_112897


namespace jelly_bean_probability_l1128_112871

theorem jelly_bean_probability 
  (red_prob : ℝ) 
  (orange_prob : ℝ) 
  (green_prob : ℝ) 
  (h1 : red_prob = 0.15)
  (h2 : orange_prob = 0.4)
  (h3 : green_prob = 0.1)
  (h4 : ∃ yellow_prob : ℝ, red_prob + orange_prob + yellow_prob + green_prob = 1) :
  ∃ yellow_prob : ℝ, yellow_prob = 0.35 ∧ red_prob + orange_prob + yellow_prob + green_prob = 1 := by
sorry

end jelly_bean_probability_l1128_112871


namespace quadratic_equation_root_l1128_112807

theorem quadratic_equation_root : ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 9 = 0) ∧
  ((-3 : ℝ)^2 - 9 = 0) := by
  sorry

end quadratic_equation_root_l1128_112807


namespace circle_symmetry_l1128_112895

/-- The equation of the original circle -/
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y + 1 = 0

/-- The equation of the line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  x - y + 3 = 0

/-- The equation of the symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 2)^2 = 1

/-- Theorem stating that the symmetric_circle is indeed symmetric to the original_circle
    with respect to the symmetry_line -/
theorem circle_symmetry :
  ∀ (x y : ℝ), original_circle x y ↔ 
  ∃ (x' y' : ℝ), symmetric_circle x' y' ∧ 
  ((x + x')/2 - (y + y')/2 + 3 = 0) ∧
  ((y' - y)/(x' - x) = -1) :=
sorry

end circle_symmetry_l1128_112895


namespace rectangular_to_polar_y_equals_x_l1128_112827

theorem rectangular_to_polar_y_equals_x :
  ∀ (x y ρ : ℝ) (θ : ℝ),
  (y = x) ↔ (θ = π / 4 ∧ ρ > 0) :=
by sorry

end rectangular_to_polar_y_equals_x_l1128_112827


namespace function_composition_nonnegative_implies_a_lower_bound_l1128_112894

theorem function_composition_nonnegative_implies_a_lower_bound 
  (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + 2 * x + 1) 
  (h2 : ∀ x, f (f x) ≥ 0) : 
  a ≥ (Real.sqrt 5 - 1) / 2 := by
sorry

end function_composition_nonnegative_implies_a_lower_bound_l1128_112894


namespace find_divisor_l1128_112888

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) (divisor : Nat) :
  dividend = quotient * divisor + remainder →
  dividend = 109 →
  quotient = 9 →
  remainder = 1 →
  divisor = 12 := by
sorry

end find_divisor_l1128_112888


namespace arithmetic_sequence_ninth_term_l1128_112868

/-- Given an arithmetic sequence where the 3rd term is 23 and the 5th term is 43,
    the 9th term is 83. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 3 = 23)  -- The 3rd term is 23
  (h2 : a 5 = 43)  -- The 5th term is 43
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- The sequence is arithmetic
  : a 9 = 83 :=
by
  sorry

end arithmetic_sequence_ninth_term_l1128_112868


namespace rectangular_field_area_l1128_112852

theorem rectangular_field_area : 
  let length : ℝ := 5.9
  let width : ℝ := 3
  length * width = 17.7 := by sorry

end rectangular_field_area_l1128_112852


namespace clinic_patient_count_l1128_112879

theorem clinic_patient_count (original_count current_count diagnosed_count : ℕ) : 
  current_count = 2 * original_count →
  diagnosed_count = 13 →
  (4 : ℕ) * diagnosed_count = current_count →
  original_count = 26 := by
  sorry

end clinic_patient_count_l1128_112879


namespace power_mod_seventeen_l1128_112870

theorem power_mod_seventeen : 7^2023 % 17 = 15 := by sorry

end power_mod_seventeen_l1128_112870


namespace fixed_distance_from_linear_combination_l1128_112802

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given vectors a and b, and a vector p satisfying the condition,
    prove that t = 9/8 and u = -1/8 make ‖p - (t*a + u*b)‖ constant. -/
theorem fixed_distance_from_linear_combination
  (a b p : E) (h : ‖p - b‖ = 3 * ‖p - a‖) :
  ∃ (c : ℝ), ∀ (q : E), ‖q - b‖ = 3 * ‖q - a‖ →
    ‖q - ((9/8 : ℝ) • a + (-1/8 : ℝ) • b)‖ = c :=
sorry

end fixed_distance_from_linear_combination_l1128_112802


namespace dividend_calculation_l1128_112885

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h1 : remainder = 1)
  (h2 : quotient = 54)
  (h3 : divisor = 4) :
  divisor * quotient + remainder = 217 := by
  sorry

end dividend_calculation_l1128_112885


namespace equation_solution_l1128_112816

theorem equation_solution : ∃ x : ℝ, 1 - 1 / ((1 - x)^3) = 1 / (1 - x) ∧ x = 1 := by
  sorry

end equation_solution_l1128_112816


namespace angle_B_is_pi_third_b_range_l1128_112847

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem --/
def condition (t : Triangle) : Prop :=
  cos t.C + (cos t.A - Real.sqrt 3 * sin t.A) * cos t.B = 0

/-- Theorem 1: Given the condition, angle B is π/3 --/
theorem angle_B_is_pi_third (t : Triangle) (h : condition t) : t.B = π / 3 := by
  sorry

/-- Additional condition for part 2 --/
def sum_sides_is_one (t : Triangle) : Prop :=
  t.a + t.c = 1

/-- Theorem 2: Given sum_sides_is_one and B = π/3, b is in [1/2, 1) --/
theorem b_range (t : Triangle) (h1 : sum_sides_is_one t) (h2 : t.B = π / 3) :
  1 / 2 ≤ t.b ∧ t.b < 1 := by
  sorry

end angle_B_is_pi_third_b_range_l1128_112847


namespace total_count_theorem_l1128_112846

/-- The total number of oysters and crabs counted over two days -/
def total_count (initial_oysters initial_crabs : ℕ) : ℕ :=
  let day1_total := initial_oysters + initial_crabs
  let day2_oysters := initial_oysters / 2
  let day2_crabs := initial_crabs * 2 / 3
  let day2_total := day2_oysters + day2_crabs
  day1_total + day2_total

/-- Theorem stating the total count of oysters and crabs over two days -/
theorem total_count_theorem (initial_oysters initial_crabs : ℕ) 
  (h1 : initial_oysters = 50) 
  (h2 : initial_crabs = 72) : 
  total_count initial_oysters initial_crabs = 195 := by
  sorry

end total_count_theorem_l1128_112846


namespace stem_and_leaf_preserves_info_l1128_112889

/-- Represents different types of statistical charts -/
inductive StatChart
  | BarChart
  | PieChart
  | LineChart
  | StemAndLeafPlot

/-- Predicate to determine if a chart loses information -/
def loses_information (chart : StatChart) : Prop :=
  match chart with
  | StatChart.BarChart => True
  | StatChart.PieChart => True
  | StatChart.LineChart => True
  | StatChart.StemAndLeafPlot => False

/-- Theorem stating that only the stem-and-leaf plot does not lose information -/
theorem stem_and_leaf_preserves_info :
  ∀ (chart : StatChart), ¬(loses_information chart) ↔ chart = StatChart.StemAndLeafPlot :=
by sorry


end stem_and_leaf_preserves_info_l1128_112889


namespace select_duty_officers_l1128_112823

def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem select_duty_officers : choose 20 3 = 1140 := by
  sorry

end select_duty_officers_l1128_112823


namespace heating_plant_consumption_l1128_112875

/-- Represents the fuel consumption of a heating plant -/
structure HeatingPlant where
  consumption_rate : ℝ  -- Liters per hour

/-- Given a heating plant that consumes 7 liters of fuel in 21 hours,
    prove that it will consume 30 liters of fuel in 90 hours -/
theorem heating_plant_consumption 
  (plant : HeatingPlant) 
  (h1 : plant.consumption_rate * 21 = 7) :
  plant.consumption_rate * 90 = 30 := by
  sorry

end heating_plant_consumption_l1128_112875


namespace transfer_equation_l1128_112853

def location_A : ℕ := 232
def location_B : ℕ := 146

theorem transfer_equation (x : ℤ) : 
  (location_A : ℤ) + x = 3 * ((location_B : ℤ) - x) ↔ 
  (location_A : ℤ) + x = 3 * ((location_B : ℤ) - x) :=
by sorry

end transfer_equation_l1128_112853


namespace sphere_expansion_l1128_112898

theorem sphere_expansion (r₁ r₂ : ℝ) (h : r₁ > 0) (h' : r₂ > 0) :
  (4 * π * r₂^2) = 4 * (4 * π * r₁^2) →
  ((4 / 3) * π * r₂^3) = 8 * ((4 / 3) * π * r₁^3) := by
  sorry

end sphere_expansion_l1128_112898


namespace mary_age_proof_l1128_112801

/-- Mary's age today -/
def mary_age : ℕ := 12

/-- Mary's father's age today -/
def father_age : ℕ := 4 * mary_age

theorem mary_age_proof :
  (father_age = 4 * mary_age) ∧
  (father_age - 3 = 5 * (mary_age - 3)) →
  mary_age = 12 :=
by sorry

end mary_age_proof_l1128_112801


namespace largest_integer_solution_l1128_112893

theorem largest_integer_solution : 
  ∀ x : ℤ, (3 * x - 4 : ℚ) / 2 < x - 1 → x ≤ 1 :=
by
  sorry

end largest_integer_solution_l1128_112893


namespace pizza_not_crust_percentage_l1128_112832

def pizza_weight : ℝ := 800
def crust_weight : ℝ := 200

theorem pizza_not_crust_percentage :
  (pizza_weight - crust_weight) / pizza_weight * 100 = 75 := by
  sorry

end pizza_not_crust_percentage_l1128_112832


namespace quadratic_maximum_value_l1128_112864

/-- A quadratic function f(x) = ax² + bx + c -/
def QuadraticFunction (a b c : ℝ) := fun (x : ℝ) ↦ a * x^2 + b * x + c

/-- The derivative of a quadratic function -/
def QuadraticDerivative (a b : ℝ) := fun (x : ℝ) ↦ 2 * a * x + b

theorem quadratic_maximum_value (a b c : ℝ) :
  (∀ x : ℝ, QuadraticFunction a b c x ≥ QuadraticDerivative a b x) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧
    (∀ k : ℝ, k ≤ M ↔ ∃ a' b' c' : ℝ, 
      (∀ x : ℝ, QuadraticFunction a' b' c' x ≥ QuadraticDerivative a' b' x) ∧
      k = b'^2 / (a'^2 + c'^2))) :=
by
  sorry

end quadratic_maximum_value_l1128_112864


namespace average_speed_two_part_journey_l1128_112845

theorem average_speed_two_part_journey 
  (total_distance : ℝ) 
  (first_part_ratio : ℝ) 
  (first_part_speed : ℝ) 
  (second_part_speed : ℝ) 
  (h1 : first_part_ratio = 0.35) 
  (h2 : first_part_speed = 35) 
  (h3 : second_part_speed = 65) 
  (h4 : first_part_ratio > 0 ∧ first_part_ratio < 1) :
  let second_part_ratio := 1 - first_part_ratio
  let first_part_time := (first_part_ratio * total_distance) / first_part_speed
  let second_part_time := (second_part_ratio * total_distance) / second_part_speed
  let total_time := first_part_time + second_part_time
  let average_speed := total_distance / total_time
  average_speed = 50 := by sorry

end average_speed_two_part_journey_l1128_112845


namespace three_digit_number_relation_l1128_112828

theorem three_digit_number_relation :
  ∀ a b c : ℕ,
  (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) →  -- three-digit number condition
  (100 * a + 10 * b + c = 56 * c) →                               -- 56 times last digit condition
  (100 * a + 10 * b + c = 112 * a) :=                             -- 112 times first digit (to prove)
by
  sorry

end three_digit_number_relation_l1128_112828


namespace company_picnic_volleyball_teams_l1128_112829

theorem company_picnic_volleyball_teams 
  (managers : ℕ) 
  (employees : ℕ) 
  (teams : ℕ) 
  (h1 : managers = 23) 
  (h2 : employees = 7) 
  (h3 : teams = 6) : 
  (managers + employees) / teams = 5 := by
  sorry

end company_picnic_volleyball_teams_l1128_112829


namespace counterexample_exists_l1128_112861

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end counterexample_exists_l1128_112861


namespace base10_to_base12_144_l1128_112850

/-- Converts a digit to its base 12 representation -/
def toBase12Digit (n : ℕ) : String :=
  if n < 10 then toString n
  else if n = 10 then "A"
  else if n = 11 then "B"
  else ""

/-- Converts a number from base 10 to base 12 -/
def toBase12 (n : ℕ) : String :=
  let d1 := n / 12
  let d0 := n % 12
  toBase12Digit d1 ++ toBase12Digit d0

theorem base10_to_base12_144 :
  toBase12 144 = "B10" := by sorry

end base10_to_base12_144_l1128_112850


namespace max_largest_integer_l1128_112851

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 45 →
  (max a (max b (max c (max d e)))) - (min a (min b (min c (min d e)))) = 10 →
  (max a (max b (max c (max d e)))) ≤ 215 :=
by sorry

end max_largest_integer_l1128_112851
