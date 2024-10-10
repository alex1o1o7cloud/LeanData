import Mathlib

namespace y_coordinate_difference_l1435_143509

/-- Given two points on a line, prove that the difference between their y-coordinates is 9 -/
theorem y_coordinate_difference (m n : ℝ) : 
  (m = (n / 3) - (2 / 5)) → 
  (m + 3 = ((n + 9) / 3) - (2 / 5)) → 
  ((n + 9) - n = 9) := by
  sorry

end y_coordinate_difference_l1435_143509


namespace system_of_equations_solution_l1435_143595

theorem system_of_equations_solution (x y z : ℝ) : 
  x + 3*y = 4*y^3 ∧ 
  y + 3*z = 4*z^3 ∧ 
  z + 3*x = 4*x^3 →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1) ∨
  (x = Real.cos (π/14) ∧ y = -Real.cos (5*π/14) ∧ z = Real.cos (3*π/14)) ∨
  (x = -Real.cos (π/14) ∧ y = Real.cos (5*π/14) ∧ z = -Real.cos (3*π/14)) ∨
  (x = Real.cos (π/7) ∧ y = -Real.cos (2*π/7) ∧ z = Real.cos (3*π/7)) ∨
  (x = -Real.cos (π/7) ∧ y = Real.cos (2*π/7) ∧ z = -Real.cos (3*π/7)) ∨
  (x = Real.cos (π/13) ∧ y = -Real.cos (π/13) ∧ z = Real.cos (3*π/13)) ∨
  (x = -Real.cos (π/13) ∧ y = Real.cos (π/13) ∧ z = -Real.cos (3*π/13)) :=
by sorry


end system_of_equations_solution_l1435_143595


namespace combustible_ice_reserves_scientific_notation_l1435_143507

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem combustible_ice_reserves_scientific_notation :
  scientific_notation 150000000000 = (1.5, 11) :=
sorry

end combustible_ice_reserves_scientific_notation_l1435_143507


namespace triple_sum_diverges_l1435_143594

open Real BigOperators

theorem triple_sum_diverges :
  let f (m n k : ℕ) := (1 : ℝ) / (m * (m + n + k) * (n + 1))
  ∃ (S : ℝ), ∀ (M N K : ℕ), (∑ m in Finset.range M, ∑ n in Finset.range N, ∑ k in Finset.range K, f m n k) ≤ S
  → false :=
sorry

end triple_sum_diverges_l1435_143594


namespace geometric_sequence_tenth_term_l1435_143516

theorem geometric_sequence_tenth_term
  (a₁ : ℚ)
  (a₂ : ℚ)
  (h₁ : a₁ = 4)
  (h₂ : a₂ = -2) :
  let r := a₂ / a₁
  let a_k (k : ℕ) := a₁ * r^(k - 1)
  a_k 10 = -1/128 := by sorry

end geometric_sequence_tenth_term_l1435_143516


namespace first_nonzero_digit_not_eventually_periodic_l1435_143572

/-- The first non-zero digit from the unit's place in the decimal representation of n! -/
def first_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The sequence of first non-zero digits is eventually periodic if there exists an N such that
    the sequence {a_n}_{n>N} is periodic -/
def eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ N T : ℕ, T > 0 ∧ ∀ n > N, a (n + T) = a n

theorem first_nonzero_digit_not_eventually_periodic :
  ¬ eventually_periodic first_nonzero_digit :=
sorry

end first_nonzero_digit_not_eventually_periodic_l1435_143572


namespace tan_sqrt3_sin_equality_l1435_143575

theorem tan_sqrt3_sin_equality : (Real.tan (10 * π / 180) - Real.sqrt 3) * Real.sin (40 * π / 180) = -1 := by
  sorry

end tan_sqrt3_sin_equality_l1435_143575


namespace price_increase_achieves_target_profit_l1435_143570

/-- Represents the price increase in yuan -/
def price_increase : ℝ := 5

/-- Initial profit per kilogram in yuan -/
def initial_profit_per_kg : ℝ := 10

/-- Initial daily sales volume in kilograms -/
def initial_sales_volume : ℝ := 500

/-- Decrease in sales volume per yuan of price increase -/
def sales_volume_decrease_rate : ℝ := 20

/-- Target daily profit in yuan -/
def target_daily_profit : ℝ := 6000

/-- Theorem stating that the given price increase achieves the target daily profit -/
theorem price_increase_achieves_target_profit :
  (initial_sales_volume - sales_volume_decrease_rate * price_increase) *
  (initial_profit_per_kg + price_increase) = target_daily_profit :=
by sorry

end price_increase_achieves_target_profit_l1435_143570


namespace estimate_pi_l1435_143567

theorem estimate_pi (total_beans : ℕ) (beans_in_circle : ℕ) 
  (h1 : total_beans = 80) (h2 : beans_in_circle = 64) : 
  (4 * beans_in_circle : ℝ) / total_beans = 3.2 :=
sorry

end estimate_pi_l1435_143567


namespace solve_equation_l1435_143582

theorem solve_equation (a : ℚ) (h : 3 * a + 6 * a / 4 = 6) : a = 4 / 3 := by
  sorry

end solve_equation_l1435_143582


namespace g_at_negative_three_l1435_143550

def g (x : ℝ) : ℝ := 3 * x^5 - 5 * x^4 + 7 * x^3 - 10 * x^2 - 12 * x + 36

theorem g_at_negative_three : g (-3) = -1341 := by sorry

end g_at_negative_three_l1435_143550


namespace remainder_difference_l1435_143517

theorem remainder_difference (d r : ℤ) : d > 1 →
  1134 % d = r →
  1583 % d = r →
  2660 % d = r →
  d - r = 213 := by sorry

end remainder_difference_l1435_143517


namespace monotonic_decreasing_interval_of_f_l1435_143515

/-- The function f(x) = 2 - x^2 -/
def f (x : ℝ) : ℝ := 2 - x^2

/-- The monotonic decreasing interval of f(x) = 2 - x^2 is (0, +∞) -/
theorem monotonic_decreasing_interval_of_f :
  ∀ x y, 0 < x → x < y → f y < f x :=
sorry

end monotonic_decreasing_interval_of_f_l1435_143515


namespace binomial_coefficient_problem_l1435_143569

theorem binomial_coefficient_problem (n : ℕ) (a b : ℝ) :
  (2 * n.choose 1 = 8) →
  n.choose 2 = 6 :=
by sorry

end binomial_coefficient_problem_l1435_143569


namespace collinear_vectors_x_value_l1435_143560

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = 3 :=
by
  sorry

end collinear_vectors_x_value_l1435_143560


namespace prime_divisor_equality_l1435_143534

theorem prime_divisor_equality (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hdiv : p ∣ q) : p = q := by
  sorry

end prime_divisor_equality_l1435_143534


namespace substitution_result_l1435_143524

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem substitution_result (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1 ∧ 1 + 3 * x^2 ≠ 0) :
  F ((3 * x - x^3) / (1 + 3 * x^2)) = 3 * F x :=
by sorry

end substitution_result_l1435_143524


namespace complex_absolute_value_l1435_143538

theorem complex_absolute_value (z : ℂ) (h : (3 - I) / (z - 3*I) = 1 + I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_absolute_value_l1435_143538


namespace propositions_truth_l1435_143581

theorem propositions_truth : 
  (∀ x : ℝ, x < 0 → abs x > x) ∧ 
  (∀ a b : ℝ, a * b < 0 ↔ a / b < 0) :=
by sorry

end propositions_truth_l1435_143581


namespace geometric_series_sum_l1435_143589

theorem geometric_series_sum : 
  let a : ℝ := 2/3
  let r : ℝ := 2/3
  let series_sum : ℝ := ∑' i, a * r^(i - 1)
  series_sum = 2 := by
sorry

end geometric_series_sum_l1435_143589


namespace z_value_l1435_143544

theorem z_value (x y z : ℝ) (h : (x + 1)⁻¹ + (y + 1)⁻¹ = z⁻¹) : 
  z = ((x + 1) * (y + 1)) / (x + y + 2) := by
  sorry

end z_value_l1435_143544


namespace pizza_order_count_l1435_143552

theorem pizza_order_count (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 168) :
  total_slices / slices_per_pizza = 21 := by
  sorry

end pizza_order_count_l1435_143552


namespace exterior_angle_of_regular_polygon_l1435_143510

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : (n - 2) * 180 = 720) :
  360 / n = 60 := by
  sorry

end exterior_angle_of_regular_polygon_l1435_143510


namespace gcd_612_840_468_l1435_143504

theorem gcd_612_840_468 : Nat.gcd 612 (Nat.gcd 840 468) = 12 := by
  sorry

end gcd_612_840_468_l1435_143504


namespace units_digit_sum_base_8_l1435_143583

-- Define a function to get the units digit in base 8
def units_digit_base_8 (n : ℕ) : ℕ := n % 8

-- Define the numbers in base 8
def num1 : ℕ := 64
def num2 : ℕ := 34

-- Theorem statement
theorem units_digit_sum_base_8 :
  units_digit_base_8 (num1 + num2) = 0 :=
by sorry

end units_digit_sum_base_8_l1435_143583


namespace train_trip_probability_l1435_143571

theorem train_trip_probability : ∀ (p₁ p₂ p₃ p₄ : ℝ),
  p₁ = 0.3 →
  p₂ = 0.1 →
  p₃ = 0.4 →
  p₁ + p₂ + p₃ + p₄ = 1 →
  p₄ = 0.2 := by
sorry

end train_trip_probability_l1435_143571


namespace arithmetic_sequence_2014th_term_l1435_143563

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_1_eq_1 : a 1 = 1
  d : ℝ
  d_ne_0 : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  is_geometric : (a 2) ^ 2 = a 1 * a 5

/-- The 2014th term of the arithmetic sequence is 4027 -/
theorem arithmetic_sequence_2014th_term (seq : ArithmeticSequence) : seq.a 2014 = 4027 := by
  sorry

end arithmetic_sequence_2014th_term_l1435_143563


namespace point_above_line_l1435_143557

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The line y = x -/
def line_y_eq_x : Set Point2D := {p : Point2D | p.y = p.x}

/-- The region above the line y = x -/
def region_above_line : Set Point2D := {p : Point2D | p.y > p.x}

/-- Theorem: Any point M(x, y) where y > x is located in the region above the line y = x -/
theorem point_above_line (M : Point2D) (h : M.y > M.x) : M ∈ region_above_line := by
  sorry

end point_above_line_l1435_143557


namespace arithmetic_mean_difference_l1435_143561

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 27) : 
  r - p = 34 := by
sorry

end arithmetic_mean_difference_l1435_143561


namespace tent_setup_plans_l1435_143555

theorem tent_setup_plans : 
  let total_students : ℕ := 50
  let valid_setup (x y : ℕ) := 3 * x + 2 * y = total_students
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => valid_setup p.1 p.2) (Finset.product (Finset.range (total_students + 1)) (Finset.range (total_students + 1)))).card ∧ n = 8
  := by sorry

end tent_setup_plans_l1435_143555


namespace houses_with_dogs_l1435_143588

/-- Given a group of houses, prove the number of houses with dogs -/
theorem houses_with_dogs 
  (total_houses : ℕ) 
  (houses_with_cats : ℕ) 
  (houses_with_both : ℕ) 
  (h1 : total_houses = 60) 
  (h2 : houses_with_cats = 30) 
  (h3 : houses_with_both = 10) : 
  ∃ (houses_with_dogs : ℕ), houses_with_dogs = 40 ∧ 
    houses_with_dogs + houses_with_cats - houses_with_both ≤ total_houses :=
by sorry

end houses_with_dogs_l1435_143588


namespace intersection_area_l1435_143584

/-- A regular cube with side length 2 units -/
structure Cube where
  side_length : ℝ
  is_regular : side_length = 2

/-- A plane that cuts the cube -/
structure IntersectingPlane where
  parallel_to_face : Bool
  at_middle : Bool

/-- The polygon formed by the intersection of the plane and the cube -/
def intersection_polygon (c : Cube) (p : IntersectingPlane) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem intersection_area (c : Cube) (p : IntersectingPlane) :
  p.parallel_to_face ∧ p.at_middle →
  area (intersection_polygon c p) = 4 :=
sorry

end intersection_area_l1435_143584


namespace friend_walking_rates_l1435_143541

theorem friend_walking_rates (trail_length : ℝ) (p_distance : ℝ) 
  (hp : trail_length = 33)
  (hpd : p_distance = 18) :
  let q_distance := trail_length - p_distance
  let rate_ratio := p_distance / q_distance
  (rate_ratio - 1) * 100 = 20 := by
  sorry

end friend_walking_rates_l1435_143541


namespace average_daily_low_temperature_l1435_143530

def daily_low_temperatures : List ℝ := [40, 47, 45, 41, 39, 43]

theorem average_daily_low_temperature :
  (daily_low_temperatures.sum / daily_low_temperatures.length : ℝ) = 42.5 := by
  sorry

end average_daily_low_temperature_l1435_143530


namespace fraction_sum_equality_l1435_143536

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (45 - b) + c / (54 - c) = 8) : 
  4 / (36 - a) + 5 / (45 - b) + 6 / (54 - c) = 11 / 9 := by
  sorry

end fraction_sum_equality_l1435_143536


namespace sqrt_25_equals_5_l1435_143551

theorem sqrt_25_equals_5 : Real.sqrt 25 = 5 := by
  sorry

end sqrt_25_equals_5_l1435_143551


namespace min_clicks_to_one_color_l1435_143554

/-- Represents a chessboard -/
def Chessboard := Fin 98 → Fin 98 → Bool

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  top_left : Fin 98 × Fin 98
  bottom_right : Fin 98 × Fin 98

/-- Applies a click to a rectangle on the chessboard -/
def applyClick (board : Chessboard) (rect : Rectangle) : Chessboard :=
  sorry

/-- Checks if the entire board is one color -/
def isOneColor (board : Chessboard) : Bool :=
  sorry

/-- Initial chessboard with alternating colors -/
def initialBoard : Chessboard :=
  sorry

/-- Theorem: The minimum number of clicks to make the chessboard one color is 98 -/
theorem min_clicks_to_one_color :
  ∀ (clicks : List Rectangle),
    isOneColor (clicks.foldl applyClick initialBoard) →
    clicks.length ≥ 98 :=
  sorry

end min_clicks_to_one_color_l1435_143554


namespace motorcycle_journey_time_ratio_l1435_143546

/-- Proves that the time taken to travel from A to B is 2 times the time taken to travel from B to C -/
theorem motorcycle_journey_time_ratio :
  ∀ (total_distance AB_distance BC_distance average_speed : ℝ),
  total_distance = 180 →
  AB_distance = 120 →
  BC_distance = 60 →
  average_speed = 20 →
  AB_distance = 2 * BC_distance →
  ∃ (AB_time BC_time : ℝ),
    AB_time > 0 ∧ BC_time > 0 ∧
    AB_time + BC_time = total_distance / average_speed ∧
    AB_time = AB_distance / average_speed ∧
    BC_time = BC_distance / average_speed ∧
    AB_time = 2 * BC_time :=
by sorry

end motorcycle_journey_time_ratio_l1435_143546


namespace cos_360_degrees_l1435_143568

theorem cos_360_degrees : Real.cos (2 * Real.pi) = 1 := by
  sorry

end cos_360_degrees_l1435_143568


namespace set_operations_and_range_l1435_143528

def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_range :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | (8 ≤ x ∧ x < 10) ∨ (2 < x ∧ x < 4)}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty → 4 ≤ a) :=
by sorry

end set_operations_and_range_l1435_143528


namespace egg_yolk_count_l1435_143543

theorem egg_yolk_count (total_eggs : ℕ) (double_yolk_eggs : ℕ) : 
  total_eggs = 12 → double_yolk_eggs = 5 → 
  (total_eggs - double_yolk_eggs) + 2 * double_yolk_eggs = 17 := by
  sorry

#check egg_yolk_count

end egg_yolk_count_l1435_143543


namespace medal_award_count_l1435_143514

/-- The number of sprinters in the event -/
def total_sprinters : ℕ := 12

/-- The number of American sprinters -/
def american_sprinters : ℕ := 5

/-- The number of medals to be awarded -/
def medals : ℕ := 3

/-- The maximum number of Americans that can receive medals -/
def max_american_medalists : ℕ := 2

/-- The function that calculates the number of ways to award medals -/
def award_medals : ℕ := sorry

theorem medal_award_count : award_medals = 1260 := by sorry

end medal_award_count_l1435_143514


namespace quadratic_monotone_iff_a_geq_one_l1435_143533

/-- A quadratic function f(x) = x^2 + 2ax + b is monotonically increasing
    on [-1, +∞) if and only if a ≥ 1 -/
theorem quadratic_monotone_iff_a_geq_one (a b : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x ≤ y → x^2 + 2*a*x + b ≤ y^2 + 2*a*y + b) ↔ a ≥ 1 :=
sorry

end quadratic_monotone_iff_a_geq_one_l1435_143533


namespace dealer_profit_is_25_percent_l1435_143520

/-- Represents a dishonest dealer's selling strategy -/
structure DishonestDealer where
  weight_reduction : ℝ  -- Percentage reduction in weight
  impurity_addition : ℝ  -- Percentage of impurities added
  
/-- Calculates the net profit percentage for a dishonest dealer -/
def net_profit_percentage (dealer : DishonestDealer) : ℝ :=
  sorry

/-- Theorem stating that the net profit percentage is 25% for the given conditions -/
theorem dealer_profit_is_25_percent :
  let dealer : DishonestDealer := { weight_reduction := 0.20, impurity_addition := 0.25 }
  net_profit_percentage dealer = 0.25 := by sorry

end dealer_profit_is_25_percent_l1435_143520


namespace no_solution_cube_equation_mod_9_l1435_143559

theorem no_solution_cube_equation_mod_9 :
  ∀ (x y z : ℤ), (x^3 + y^3) % 9 ≠ (z^3 + 4) % 9 :=
by sorry

end no_solution_cube_equation_mod_9_l1435_143559


namespace cistern_solution_l1435_143502

/-- Represents the time (in hours) it takes to fill or empty the cistern -/
structure CisternTime where
  fill : ℝ
  empty : ℝ
  both : ℝ

/-- The cistern filling problem -/
def cistern_problem (t : CisternTime) : Prop :=
  t.fill = 10 ∧ 
  t.empty = 12 ∧ 
  t.both = 60 ∧
  t.both = (t.fill * t.empty) / (t.empty - t.fill)

theorem cistern_solution :
  ∃ t : CisternTime, cistern_problem t :=
sorry

end cistern_solution_l1435_143502


namespace production_increase_l1435_143591

def planned_daily_production : ℕ := 500

def daily_changes : List ℤ := [40, -30, 90, -50, -20, -10, 20]

def actual_daily_production : List ℕ := 
  List.scanl (λ acc change => (acc : ℤ) + change |>.toNat) planned_daily_production daily_changes

def total_actual_production : ℕ := actual_daily_production.sum

def total_planned_production : ℕ := planned_daily_production * 7

theorem production_increase :
  total_actual_production = 3790 ∧ total_actual_production > total_planned_production :=
by sorry

end production_increase_l1435_143591


namespace mod_equivalence_l1435_143586

theorem mod_equivalence (m : ℕ) : 
  152 * 936 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 22 := by
  sorry

end mod_equivalence_l1435_143586


namespace geometric_series_sum_l1435_143574

theorem geometric_series_sum : 
  let a : ℚ := 1/5
  let r : ℚ := -1/3
  let n : ℕ := 7
  let series_sum := a * (1 - r^n) / (1 - r)
  series_sum = 1641/10935 := by sorry

end geometric_series_sum_l1435_143574


namespace max_min_difference_c_l1435_143500

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 18) : 
  ∃ (c_max c_min : ℝ), 
    (∀ c', a + b + c' = 3 ∧ a^2 + b^2 + c'^2 = 18 → c' ≤ c_max ∧ c' ≥ c_min) ∧
    c_max - c_min = 6 := by
  sorry

end max_min_difference_c_l1435_143500


namespace cheese_slices_total_l1435_143505

/-- The number of cheese slices used for ham sandwiches -/
def ham_cheese_slices (num_ham_sandwiches : ℕ) (cheese_per_ham : ℕ) : ℕ :=
  num_ham_sandwiches * cheese_per_ham

/-- The number of cheese slices used for grilled cheese sandwiches -/
def grilled_cheese_slices (num_grilled_cheese : ℕ) (cheese_per_grilled : ℕ) : ℕ :=
  num_grilled_cheese * cheese_per_grilled

/-- The total number of cheese slices used for both types of sandwiches -/
def total_cheese_slices (ham_slices : ℕ) (grilled_slices : ℕ) : ℕ :=
  ham_slices + grilled_slices

/-- Theorem: The total number of cheese slices used for 10 ham sandwiches
    (each requiring 2 slices) and 10 grilled cheese sandwiches
    (each requiring 3 slices) is equal to 50. -/
theorem cheese_slices_total :
  total_cheese_slices
    (ham_cheese_slices 10 2)
    (grilled_cheese_slices 10 3) = 50 := by
  sorry

end cheese_slices_total_l1435_143505


namespace square_difference_l1435_143548

theorem square_difference (x y z w : ℝ) 
  (sum_xy : x + y = 10)
  (diff_xy : x - y = 8)
  (sum_yz : y + z = 15)
  (sum_zw : z + w = 20) :
  x^2 - w^2 = 45 := by sorry

end square_difference_l1435_143548


namespace expansion_coefficient_sum_l1435_143526

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of x^k in the expansion of (1-2x)^n -/
def coeff (n k : ℕ) : ℤ :=
  (-2)^k * binomial n k

theorem expansion_coefficient_sum (n : ℕ) 
  (h : coeff n 1 + coeff n 4 = 70) : 
  coeff n 5 = -32 := by sorry

end expansion_coefficient_sum_l1435_143526


namespace distance_difference_l1435_143562

/-- The width of the streets in Longtown -/
def street_width : ℝ := 30

/-- The length of the longer side of the block -/
def block_length : ℝ := 500

/-- The length of the shorter side of the block -/
def block_width : ℝ := 300

/-- The distance Jenny runs around the block -/
def jenny_distance : ℝ := 2 * (block_length + block_width)

/-- The distance Jeremy runs around the block -/
def jeremy_distance : ℝ := 2 * ((block_length + 2 * street_width) + (block_width + 2 * street_width))

/-- Theorem stating the difference in distance run by Jeremy and Jenny -/
theorem distance_difference : jeremy_distance - jenny_distance = 240 := by
  sorry

end distance_difference_l1435_143562


namespace quadratic_equation_equivalence_l1435_143573

/-- Given two quadratic equations and a relationship between their roots, 
    prove the condition for the equations to be identical -/
theorem quadratic_equation_equivalence 
  (p q r s : ℝ) 
  (x₁ x₂ y₁ y₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) →
  (x₂^2 + p*x₂ + q = 0) →
  (y₁^2 + r*y₁ + s = 0) →
  (y₂^2 + r*y₂ + s = 0) →
  (y₁ = x₁/(x₁-1)) →
  (y₂ = x₂/(x₂-1)) →
  (x₁ ≠ 1) →
  (x₂ ≠ 1) →
  (p = -r ∧ q = s) →
  (p + q = 0) :=
by sorry

end quadratic_equation_equivalence_l1435_143573


namespace soup_donation_per_person_l1435_143508

theorem soup_donation_per_person
  (num_shelters : ℕ)
  (people_per_shelter : ℕ)
  (total_cans : ℕ)
  (h1 : num_shelters = 6)
  (h2 : people_per_shelter = 30)
  (h3 : total_cans = 1800) :
  total_cans / (num_shelters * people_per_shelter) = 10 := by
sorry

end soup_donation_per_person_l1435_143508


namespace midpoint_on_number_line_l1435_143590

theorem midpoint_on_number_line (A B C : ℝ) : 
  A = -7 → 
  |B - A| = 5 → 
  C = (A + B) / 2 → 
  C = -9/2 ∨ C = -19/2 :=
by sorry

end midpoint_on_number_line_l1435_143590


namespace prop_evaluation_l1435_143519

-- Define the propositions p and q
def p (x y : ℝ) : Prop := (x > y) → (-x < -y)
def q (x y : ℝ) : Prop := (x < y) → (x^2 < y^2)

-- State the theorem
theorem prop_evaluation : ∃ (x y : ℝ), (p x y ∨ q x y) ∧ (p x y ∧ ¬(q x y)) := by
  sorry

end prop_evaluation_l1435_143519


namespace simplify_sqrt_expression_l1435_143597

theorem simplify_sqrt_expression : 
  Real.sqrt 5 - Real.sqrt 20 + Real.sqrt 45 = 2 * Real.sqrt 5 := by
sorry

end simplify_sqrt_expression_l1435_143597


namespace george_christopher_age_difference_l1435_143564

theorem george_christopher_age_difference :
  ∀ (G C F : ℕ),
    C = 18 →
    F = C - 2 →
    G + C + F = 60 →
    G > C →
    G - C = 8 :=
by
  sorry

end george_christopher_age_difference_l1435_143564


namespace alices_number_l1435_143565

theorem alices_number (y : ℝ) : 3 * (3 * y + 15) = 135 → y = 10 := by
  sorry

end alices_number_l1435_143565


namespace circle_intersection_range_l1435_143593

theorem circle_intersection_range (a : ℝ) : 
  (∃! (p q : ℝ × ℝ), p ≠ q ∧ 
    ((p.1 - a)^2 + (p.2 - a)^2 = 4) ∧
    ((q.1 - a)^2 + (q.2 - a)^2 = 4) ∧
    (p.1^2 + p.2^2 = 4) ∧
    (q.1^2 + q.2^2 = 4)) →
  (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 ∧ a ≠ 0) :=
by sorry

end circle_intersection_range_l1435_143593


namespace f_plus_g_equals_one_l1435_143545

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_g_equals_one
  (h1 : is_even f)
  (h2 : is_odd g)
  (h3 : ∀ x, f x - g x = x^3 + x^2 + 1) :
  f 1 + g 1 = 1 := by
  sorry

end f_plus_g_equals_one_l1435_143545


namespace magazines_per_bookshelf_l1435_143521

theorem magazines_per_bookshelf
  (num_books : ℕ)
  (num_bookshelves : ℕ)
  (total_items : ℕ)
  (h1 : num_books = 23)
  (h2 : num_bookshelves = 29)
  (h3 : total_items = 2436) :
  (total_items - num_books) / num_bookshelves = 83 := by
sorry

end magazines_per_bookshelf_l1435_143521


namespace cristinas_croissants_l1435_143522

theorem cristinas_croissants (total_croissants : ℕ) (num_guests : ℕ) 
  (h1 : total_croissants = 17) 
  (h2 : num_guests = 7) : 
  total_croissants % num_guests = 3 := by
  sorry

end cristinas_croissants_l1435_143522


namespace second_number_value_l1435_143598

theorem second_number_value (x y z : ℝ) 
  (sum_eq : x + y + z = 120) 
  (ratio_xy : x / y = 3 / 4) 
  (ratio_yz : y / z = 4 / 7) : 
  y = 34 := by sorry

end second_number_value_l1435_143598


namespace inverse_proportion_l1435_143566

/-- Given that α is inversely proportional to β, prove that when α = 5 for β = 10, 
    then α = 25/2 for β = 4 -/
theorem inverse_proportion (α β : ℝ) (k : ℝ) (h1 : α * β = k) 
    (h2 : 5 * 10 = k) : 
  4 * (25/2 : ℝ) = k := by
  sorry

end inverse_proportion_l1435_143566


namespace sqrt_sum_inequality_l1435_143558

theorem sqrt_sum_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end sqrt_sum_inequality_l1435_143558


namespace gasoline_expense_gasoline_expense_proof_l1435_143579

/-- Calculates the amount spent on gasoline given the initial amount, known expenses, money received, and the amount left for the return trip. -/
theorem gasoline_expense (initial_amount : ℝ) (lunch_expense : ℝ) (gift_expense : ℝ) 
  (money_from_grandma : ℝ) (return_trip_money : ℝ) : ℝ :=
  let total_amount := initial_amount + money_from_grandma
  let known_expenses := lunch_expense + gift_expense
  let remaining_after_known_expenses := total_amount - known_expenses
  remaining_after_known_expenses - return_trip_money

/-- Proves that the amount spent on gasoline is $8 given the specific values from the problem. -/
theorem gasoline_expense_proof :
  gasoline_expense 50 15.65 10 20 36.35 = 8 := by
  sorry

end gasoline_expense_gasoline_expense_proof_l1435_143579


namespace probability_one_or_two_in_pascal_l1435_143542

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def ones_in_pascal (n : ℕ) : ℕ := if n = 0 then 0 else 2 * n - 1

/-- The number of 2's in the first n rows of Pascal's Triangle -/
def twos_in_pascal (n : ℕ) : ℕ := if n ≤ 2 then 0 else 2 * (n - 2)

/-- The probability of selecting 1 or 2 from the first 20 rows of Pascal's Triangle -/
theorem probability_one_or_two_in_pascal : 
  (ones_in_pascal 20 + twos_in_pascal 20 : ℚ) / pascal_triangle_elements 20 = 5 / 14 := by
  sorry

end probability_one_or_two_in_pascal_l1435_143542


namespace arithmetic_sequence_fourth_term_l1435_143527

/-- Given an arithmetic sequence {a_n} where a₃ + a₅ = 10, prove that a₄ = 5 -/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : a 3 + a 5 = 10) : 
  a 4 = 5 := by
sorry

end arithmetic_sequence_fourth_term_l1435_143527


namespace passes_count_is_32_l1435_143537

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the pool and swimming scenario --/
structure SwimmingScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def countPasses (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- The specific swimming scenario from the problem --/
def problemScenario : SwimmingScenario :=
  { poolLength := 100
    swimmer1 := { speed := 4, startPosition := 0 }
    swimmer2 := { speed := 3, startPosition := 100 }
    totalTime := 720 }

/-- Theorem stating that the number of passes in the given scenario is 32 --/
theorem passes_count_is_32 : countPasses problemScenario = 32 :=
  sorry

end passes_count_is_32_l1435_143537


namespace alyssa_cans_collected_l1435_143518

theorem alyssa_cans_collected (total_cans : ℕ) (abigail_cans : ℕ) (cans_needed : ℕ) 
  (h1 : total_cans = 100)
  (h2 : abigail_cans = 43)
  (h3 : cans_needed = 27) :
  total_cans - (abigail_cans + cans_needed) = 30 := by
  sorry

end alyssa_cans_collected_l1435_143518


namespace binomial_not_perfect_power_l1435_143596

theorem binomial_not_perfect_power (n k l m : ℕ) : 
  l ≥ 2 → 4 ≤ k → k ≤ n - 4 → (n.choose k) ≠ m^l := by
  sorry

end binomial_not_perfect_power_l1435_143596


namespace perpendicular_lines_l1435_143592

def line1 (x y : ℝ) : Prop := 3 * x - y = 6

def line2 (x y : ℝ) : Prop := y = -1/3 * x + 7/3

def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ m1 m2 : ℝ, (∀ x y, f x y ↔ y = m1 * x + 0) ∧ 
              (∀ x y, g x y ↔ y = m2 * x + 0) ∧
              m1 * m2 = -1

theorem perpendicular_lines :
  perpendicular line1 line2 ∧ line2 (-2) 3 := by sorry

end perpendicular_lines_l1435_143592


namespace eric_pencils_l1435_143577

/-- The number of boxes of pencils Eric has -/
def num_boxes : ℕ := 12

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 17

/-- The total number of pencils Eric has -/
def total_pencils : ℕ := num_boxes * pencils_per_box

theorem eric_pencils : total_pencils = 204 := by
  sorry

end eric_pencils_l1435_143577


namespace chocolate_bar_breaks_l1435_143547

/-- Represents a rectangular chocolate bar -/
structure ChocolateBar where
  rows : ℕ
  cols : ℕ

/-- Calculates the minimum number of breaks required to separate a chocolate bar into individual pieces -/
def min_breaks (bar : ChocolateBar) : ℕ :=
  (bar.rows - 1) * bar.cols + (bar.cols - 1)

theorem chocolate_bar_breaks (bar : ChocolateBar) (h1 : bar.rows = 5) (h2 : bar.cols = 8) :
  min_breaks bar = 39 := by
  sorry

#eval min_breaks ⟨5, 8⟩

end chocolate_bar_breaks_l1435_143547


namespace ray_walks_to_highschool_l1435_143578

/-- Represents the number of blocks Ray walks to the park -/
def blocks_to_park : ℕ := 4

/-- Represents the number of blocks Ray walks from the high school to home -/
def blocks_from_highschool_to_home : ℕ := 11

/-- Represents the number of times Ray walks his dog each day -/
def walks_per_day : ℕ := 3

/-- Represents the total number of blocks Ray's dog walks each day -/
def total_blocks_per_day : ℕ := 66

/-- Represents the number of blocks Ray walks to the high school -/
def blocks_to_highschool : ℕ := 7

theorem ray_walks_to_highschool :
  blocks_to_highschool = 7 ∧
  walks_per_day * (blocks_to_park + blocks_to_highschool + blocks_from_highschool_to_home) = total_blocks_per_day :=
by sorry

end ray_walks_to_highschool_l1435_143578


namespace y_derivative_l1435_143535

noncomputable def y (x : ℝ) : ℝ :=
  (3^x * (Real.log 3 * Real.sin (2*x) - 2 * Real.cos (2*x))) / ((Real.log 3)^2 + 4)

theorem y_derivative (x : ℝ) :
  deriv y x = 3^x * Real.sin (2*x) :=
by sorry

end y_derivative_l1435_143535


namespace students_answering_one_question_l1435_143529

/-- Represents the number of questions answered by students in each grade -/
structure GradeAnswers :=
  (g1 g2 g3 g4 g5 : Nat)

/-- The problem setup -/
structure ProblemSetup :=
  (total_students : Nat)
  (total_grades : Nat)
  (total_questions : Nat)
  (grade_answers : GradeAnswers)

/-- The conditions of the problem -/
def satisfies_conditions (setup : ProblemSetup) : Prop :=
  setup.total_students = 30 ∧
  setup.total_grades = 5 ∧
  setup.total_questions = 40 ∧
  setup.grade_answers.g1 < setup.grade_answers.g2 ∧
  setup.grade_answers.g2 < setup.grade_answers.g3 ∧
  setup.grade_answers.g3 < setup.grade_answers.g4 ∧
  setup.grade_answers.g4 < setup.grade_answers.g5 ∧
  setup.grade_answers.g1 ≥ 1 ∧
  setup.grade_answers.g2 ≥ 1 ∧
  setup.grade_answers.g3 ≥ 1 ∧
  setup.grade_answers.g4 ≥ 1 ∧
  setup.grade_answers.g5 ≥ 1

/-- The theorem to be proved -/
theorem students_answering_one_question (setup : ProblemSetup) 
  (h : satisfies_conditions setup) : 
  setup.total_students - (setup.total_questions - (setup.grade_answers.g1 + 
  setup.grade_answers.g2 + setup.grade_answers.g3 + setup.grade_answers.g4 + 
  setup.grade_answers.g5)) = 26 :=
by sorry

end students_answering_one_question_l1435_143529


namespace perimeter_is_200_l1435_143503

/-- A rectangle with an inscribed rhombus -/
structure RectangleWithRhombus where
  -- Length of half of side AB
  wa : ℝ
  -- Length of half of side BC
  xb : ℝ
  -- Length of diagonal WY of the rhombus
  wy : ℝ

/-- The perimeter of the rectangle -/
def perimeter (r : RectangleWithRhombus) : ℝ :=
  2 * (2 * r.wa + 2 * r.xb)

/-- Theorem: The perimeter of the rectangle is 200 -/
theorem perimeter_is_200 (r : RectangleWithRhombus)
    (h1 : r.wa = 20)
    (h2 : r.xb = 30)
    (h3 : r.wy = 50) :
    perimeter r = 200 := by
  sorry

end perimeter_is_200_l1435_143503


namespace shoes_price_calculation_l1435_143576

/-- The price of shoes after a markup followed by a discount -/
def monday_price (thursday_price : ℝ) (friday_markup : ℝ) (monday_discount : ℝ) : ℝ :=
  thursday_price * (1 + friday_markup) * (1 - monday_discount)

/-- Theorem stating that the Monday price is $50.60 given the specified conditions -/
theorem shoes_price_calculation :
  monday_price 50 0.15 0.12 = 50.60 := by
  sorry

#eval monday_price 50 0.15 0.12

end shoes_price_calculation_l1435_143576


namespace right_triangle_hypotenuse_l1435_143511

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 80 ∧ b = 150 ∧ c^2 = a^2 + b^2 → c = 170 := by
  sorry

end right_triangle_hypotenuse_l1435_143511


namespace cos_2alpha_value_l1435_143523

theorem cos_2alpha_value (α : ℝ) (h : Real.sin (α - 3 * Real.pi / 2) = 3 / 5) : 
  Real.cos (2 * α) = -7 / 25 := by
  sorry

end cos_2alpha_value_l1435_143523


namespace debate_team_girls_l1435_143501

/-- The number of boys on the debate team -/
def num_boys : ℕ := 11

/-- The number of groups the team can be split into -/
def num_groups : ℕ := 8

/-- The number of students in each group -/
def students_per_group : ℕ := 7

/-- The total number of students on the debate team -/
def total_students : ℕ := num_groups * students_per_group

/-- The number of girls on the debate team -/
def num_girls : ℕ := total_students - num_boys

theorem debate_team_girls : num_girls = 45 := by
  sorry

end debate_team_girls_l1435_143501


namespace triangle_on_parabola_ef_length_l1435_143580

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 0)

/-- Triangle DEF -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The theorem to be proved -/
theorem triangle_on_parabola_ef_length (t : Triangle) :
  t.D = vertex ∧
  (∀ x, (x, parabola x) = t.D ∨ (x, parabola x) = t.E ∨ (x, parabola x) = t.F) ∧
  t.E.2 = t.F.2 ∧
  (1/2 * (t.F.1 - t.E.1) * (t.E.2 - t.D.2) = 32) →
  t.F.1 - t.E.1 = 8 := by
  sorry

end triangle_on_parabola_ef_length_l1435_143580


namespace bike_cost_calculation_l1435_143506

/-- The cost of Trey's new bike -/
def bike_cost : ℕ := 112

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of bracelets Trey needs to sell each day -/
def bracelets_per_day : ℕ := 8

/-- The price of each bracelet in dollars -/
def price_per_bracelet : ℕ := 1

/-- Theorem stating that the bike cost is equal to the product of days, bracelets per day, and price per bracelet -/
theorem bike_cost_calculation : 
  bike_cost = days_in_two_weeks * bracelets_per_day * price_per_bracelet := by
  sorry

end bike_cost_calculation_l1435_143506


namespace candy_bar_distribution_l1435_143585

theorem candy_bar_distribution (total_bars : ℕ) (spare_bars : ℕ) (num_friends : ℕ) 
  (h1 : total_bars = 24)
  (h2 : spare_bars = 10)
  (h3 : num_friends = 7)
  : (total_bars - spare_bars) / num_friends = 2 := by
  sorry

end candy_bar_distribution_l1435_143585


namespace probability_qualified_product_l1435_143587

/-- The proportion of the first batch in the total mix -/
def batch1_proportion : ℝ := 0.30

/-- The proportion of the second batch in the total mix -/
def batch2_proportion : ℝ := 0.70

/-- The defect rate of the first batch -/
def batch1_defect_rate : ℝ := 0.05

/-- The defect rate of the second batch -/
def batch2_defect_rate : ℝ := 0.04

/-- The probability of selecting a qualified product from the mixed batches -/
theorem probability_qualified_product : 
  batch1_proportion * (1 - batch1_defect_rate) + batch2_proportion * (1 - batch2_defect_rate) = 0.957 := by
  sorry

end probability_qualified_product_l1435_143587


namespace sum_of_extremes_3point5_l1435_143513

/-- A number that rounds to 3.5 when rounded to one decimal place -/
def RoundsTo3Point5 (x : ℝ) : Prop :=
  (x ≥ 3.45) ∧ (x < 3.55)

/-- The theorem stating the sum of the largest and smallest 3-digit decimals
    that round to 3.5 is 6.99 -/
theorem sum_of_extremes_3point5 :
  ∃ (min max : ℝ),
    (∀ x, RoundsTo3Point5 x → x ≥ min) ∧
    (∀ x, RoundsTo3Point5 x → x ≤ max) ∧
    (RoundsTo3Point5 min) ∧
    (RoundsTo3Point5 max) ∧
    (min + max = 6.99) :=
by sorry

end sum_of_extremes_3point5_l1435_143513


namespace root_equation_sum_l1435_143531

theorem root_equation_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x - 1 = 0 → x^4 + a*x^2 + b*x + c = 0) →
  a + b + 4*c = -7 := by
sorry

end root_equation_sum_l1435_143531


namespace roots_arithmetic_progression_implies_sum_zero_l1435_143512

theorem roots_arithmetic_progression_implies_sum_zero 
  (a b c : ℝ) 
  (p₁ p₂ q₁ q₂ : ℝ) 
  (h₁ : a * p₁^2 + b * p₁ + c = 0)
  (h₂ : a * p₂^2 + b * p₂ + c = 0)
  (h₃ : c * q₁^2 + b * q₁ + a = 0)
  (h₄ : c * q₂^2 + b * q₂ + a = 0)
  (h₅ : ∃ (d : ℝ), d ≠ 0 ∧ q₁ - p₁ = d ∧ p₂ - q₁ = d ∧ q₂ - p₂ = d)
  (h₆ : p₁ ≠ q₁ ∧ q₁ ≠ p₂ ∧ p₂ ≠ q₂) :
  a + c = 0 := by
sorry

end roots_arithmetic_progression_implies_sum_zero_l1435_143512


namespace machine_value_after_two_years_l1435_143539

def initial_value : ℝ := 8000
def depreciation_rate : ℝ := 0.15

def market_value_after_two_years (initial : ℝ) (rate : ℝ) : ℝ :=
  initial * (1 - rate) * (1 - rate)

theorem machine_value_after_two_years :
  market_value_after_two_years initial_value depreciation_rate = 5780 := by
  sorry

end machine_value_after_two_years_l1435_143539


namespace repeating_decimal_equals_fraction_l1435_143556

/-- The repeating decimal 0.565656... expressed as a rational number -/
def repeating_decimal : ℚ := 0.56565656

/-- The fraction 56/99 -/
def fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to the fraction 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l1435_143556


namespace parabola_equation_l1435_143549

/-- Theorem: For a parabola y² = 2px where p > 0, if a line passing through its focus
    intersects the parabola at two points P(x₁, y₁) and Q(x₂, y₂) such that x₁ + x₂ = 2
    and |PQ| = 4, then the equation of the parabola is y² = 4x. -/
theorem parabola_equation (p : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  p > 0 →
  y₁^2 = 2*p*x₁ →
  y₂^2 = 2*p*x₂ →
  x₁ + x₂ = 2 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16 →
  ∀ x y, y^2 = 2*p*x → y^2 = 4*x :=
by sorry

end parabola_equation_l1435_143549


namespace chessboard_coloring_l1435_143525

/-- A color used to paint the chessboard squares -/
inductive Color
| Red
| Green
| Blue

/-- A chessboard configuration is a function from (row, column) to Color -/
def ChessboardConfig := Fin 4 → Fin 19 → Color

/-- The theorem statement -/
theorem chessboard_coloring (config : ChessboardConfig) :
  ∃ (r₁ r₂ : Fin 4) (c₁ c₂ : Fin 19),
    r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧
    config r₁ c₁ = config r₁ c₂ ∧
    config r₁ c₁ = config r₂ c₁ ∧
    config r₁ c₁ = config r₂ c₂ :=
sorry

end chessboard_coloring_l1435_143525


namespace jenny_rommel_age_difference_l1435_143532

/-- Given the ages and relationships of Tim, Rommel, and Jenny, prove that Jenny is 2 years older than Rommel -/
theorem jenny_rommel_age_difference :
  ∀ (tim_age rommel_age jenny_age : ℕ),
  tim_age = 5 →
  rommel_age = 3 * tim_age →
  jenny_age = tim_age + 12 →
  jenny_age - rommel_age = 2 :=
by
  sorry

end jenny_rommel_age_difference_l1435_143532


namespace intersection_of_A_and_B_l1435_143553

-- Define the sets A and B
def A : Set ℝ := {x | 3*x^2 - 14*x + 16 ≤ 0}
def B : Set ℝ := {x | (3*x - 7) / x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 7/3 < x ∧ x ≤ 8/3} := by sorry

end intersection_of_A_and_B_l1435_143553


namespace max_c_magnitude_l1435_143599

theorem max_c_magnitude (a b c : ℝ × ℝ) : 
  (‖a‖ = 1) → 
  (‖b‖ = 1) → 
  (a • b = 1/2) → 
  (‖a - b + c‖ ≤ 1) →
  (∃ (c : ℝ × ℝ), ‖c‖ = 2) ∧ 
  (∀ (c : ℝ × ℝ), ‖a - b + c‖ ≤ 1 → ‖c‖ ≤ 2) :=
by sorry

end max_c_magnitude_l1435_143599


namespace papaya_problem_l1435_143540

def remaining_green_papayas (initial : Nat) (friday_yellow : Nat) : Nat :=
  initial - friday_yellow - (2 * friday_yellow)

theorem papaya_problem :
  remaining_green_papayas 14 2 = 8 := by
  sorry

end papaya_problem_l1435_143540
