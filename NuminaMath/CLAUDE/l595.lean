import Mathlib

namespace fraction_equation_l595_59566

theorem fraction_equation : ∃ (A B C : ℤ), 
  (A : ℚ) / 999 + (B : ℚ) / 1000 + (C : ℚ) / 1001 = 1 / (999 * 1000 * 1001) :=
by
  -- We claim that A = 500, B = -1, C = -500 satisfy the equation
  use 500, -1, -500
  -- The proof goes here
  sorry

end fraction_equation_l595_59566


namespace geometric_sequence_property_l595_59573

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 2 = 1 →
  a 5 * a 6 = 4 →
  a 3 * a 4 = 2 :=
by
  sorry

end geometric_sequence_property_l595_59573


namespace sine_function_midline_l595_59530

theorem sine_function_midline (A B C D : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h4 : D > 0) :
  (∀ x, 1 ≤ A * Real.sin (B * x + C) + D ∧ A * Real.sin (B * x + C) + D ≤ 5) → D = 3 := by
sorry

end sine_function_midline_l595_59530


namespace charitable_woman_purse_l595_59597

/-- The charitable woman's purse problem -/
theorem charitable_woman_purse (P : ℚ) : 
  (P > 0) →
  (P - ((1/2) * P + 1) - ((1/2) * ((1/2) * P - 1) + 2) - ((1/2) * ((1/4) * P - 2.5) + 3) = 1) →
  P = 42 := by
  sorry

end charitable_woman_purse_l595_59597


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l595_59505

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_equation :
  let r₁ := (6 + Real.sqrt (36 - 32)) / 2
  let r₂ := (6 - Real.sqrt (36 - 32)) / 2
  r₁ + r₂ = 6 := by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l595_59505


namespace power_function_through_point_l595_59518

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_through_point :
  ∀ f : ℝ → ℝ, isPowerFunction f → f 2 = Real.sqrt 2 →
  ∀ x : ℝ, x ≥ 0 → f x = Real.sqrt x := by
  sorry

end power_function_through_point_l595_59518


namespace horse_saddle_ratio_l595_59527

theorem horse_saddle_ratio : ∀ (total_cost saddle_cost horse_cost : ℕ),
  total_cost = 5000 →
  saddle_cost = 1000 →
  horse_cost = total_cost - saddle_cost →
  (horse_cost : ℚ) / (saddle_cost : ℚ) = 4 := by
  sorry

end horse_saddle_ratio_l595_59527


namespace min_unique_points_is_eight_l595_59506

/-- A square with points marked on its sides -/
structure MarkedSquare where
  /-- The number of points marked on each side of the square -/
  pointsPerSide : ℕ
  /-- Condition: Each side has exactly 3 points -/
  threePointsPerSide : pointsPerSide = 3

/-- The minimum number of unique points marked on the square -/
def minUniquePoints (s : MarkedSquare) : ℕ :=
  s.pointsPerSide * 4 - 4

/-- Theorem: The minimum number of unique points marked on the square is 8 -/
theorem min_unique_points_is_eight (s : MarkedSquare) : 
  minUniquePoints s = 8 := by
  sorry

end min_unique_points_is_eight_l595_59506


namespace average_playing_time_l595_59548

/-- Calculates the average playing time given the hours played on three days,
    where the third day is 3 hours more than each of the first two days. -/
theorem average_playing_time (hours_day1 hours_day2 : ℕ) 
    (h1 : hours_day1 = hours_day2)
    (h2 : hours_day1 > 0) : 
  (hours_day1 + hours_day2 + (hours_day1 + 3)) / 3 = hours_day1 + 1 :=
by sorry

#check average_playing_time

end average_playing_time_l595_59548


namespace probability_ratio_equals_ways_ratio_l595_59575

/-- The number of balls --/
def n : ℕ := 25

/-- The number of bins --/
def m : ℕ := 6

/-- The number of ways to distribute n balls into m bins --/
def total_distributions : ℕ := Nat.choose (n + m - 1) n

/-- The number of ways to distribute balls according to the 5-5-3-3-2-2 pattern --/
def ways_p : ℕ := Nat.choose n 5 * Nat.choose 20 5 * Nat.choose 15 3 * Nat.choose 12 3 * Nat.choose 9 2 * Nat.choose 7 2

/-- The number of ways to distribute balls equally (4-4-4-4-4-5 pattern) --/
def ways_q : ℕ := Nat.choose n 4 * Nat.choose 21 4 * Nat.choose 17 4 * Nat.choose 13 4 * Nat.choose 9 4 * Nat.choose 5 5

/-- The probability of the 5-5-3-3-2-2 distribution --/
def p : ℚ := ways_p / total_distributions

/-- The probability of the equal distribution --/
def q : ℚ := ways_q / total_distributions

theorem probability_ratio_equals_ways_ratio : p / q = ways_p / ways_q := by
  sorry

end probability_ratio_equals_ways_ratio_l595_59575


namespace computer_operations_per_hour_l595_59508

/-- Represents the number of operations a computer can perform per second -/
structure ComputerSpeed :=
  (multiplications_per_second : ℕ)
  (additions_per_second : ℕ)

/-- Calculates the total number of operations per hour -/
def operations_per_hour (speed : ComputerSpeed) : ℕ :=
  (speed.multiplications_per_second + speed.additions_per_second) * 3600

/-- Theorem: A computer with the given speed performs 72 million operations per hour -/
theorem computer_operations_per_hour :
  let speed := ComputerSpeed.mk 15000 5000
  operations_per_hour speed = 72000000 := by
  sorry

end computer_operations_per_hour_l595_59508


namespace triangle_dimensions_l595_59531

theorem triangle_dimensions (a m : ℝ) (h1 : a = m + 4) (h2 : (a + 12) * (m + 12) = 5 * a * m) : 
  a = 12 ∧ m = 8 := by
  sorry

end triangle_dimensions_l595_59531


namespace trig_simplification_trig_value_given_tan_l595_59539

/-- Proves that the given trigonometric expression simplifies to -1 --/
theorem trig_simplification : 
  (Real.sqrt (1 - 2 * Real.sin (10 * π / 180) * Real.cos (10 * π / 180))) / 
  (Real.sin (170 * π / 180) - Real.sqrt (1 - Real.sin (170 * π / 180) ^ 2)) = -1 := by
  sorry

/-- Proves that given tan θ = 2, the expression 2 + sin θ * cos θ - cos² θ equals 11/5 --/
theorem trig_value_given_tan (θ : Real) (h : Real.tan θ = 2) : 
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 11/5 := by
  sorry

end trig_simplification_trig_value_given_tan_l595_59539


namespace circle_equation_k_value_l595_59553

/-- 
A circle in the xy-plane can be represented by an equation of the form
(x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius.
This theorem proves that for the equation x^2 + 14x + y^2 + 8y - k = 0 to
represent a circle of radius 8, k must equal 1.
-/
theorem circle_equation_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x + 7)^2 + (y + 4)^2 = 64) →
  k = 1 := by
sorry

end circle_equation_k_value_l595_59553


namespace juan_and_maria_distance_l595_59519

/-- The combined distance covered by two runners given their speeds and times -/
def combined_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem stating that the combined distance of Juan and Maria is 658.5 miles -/
theorem juan_and_maria_distance :
  combined_distance 9.5 30 8.3 45 = 658.5 := by
  sorry

end juan_and_maria_distance_l595_59519


namespace middle_angle_range_l595_59592

theorem middle_angle_range (β : Real) (h1 : 0 < β) (h2 : β < 90) : 
  ∃ (α γ : Real), 
    0 < α ∧ 0 < γ ∧ 
    α + β + γ = 180 ∧ 
    α ≤ β ∧ β ≤ γ :=
by sorry

end middle_angle_range_l595_59592


namespace waiting_room_ratio_l595_59565

theorem waiting_room_ratio : 
  ∀ (initial_waiting : ℕ) (arrivals : ℕ) (interview : ℕ),
    initial_waiting = 22 →
    arrivals = 3 →
    (initial_waiting + arrivals) % interview = 0 →
    interview ≠ 1 →
    interview < initial_waiting + arrivals →
    (initial_waiting + arrivals) / interview = 5 :=
by sorry

end waiting_room_ratio_l595_59565


namespace max_x_minus_y_l595_59529

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^2 + y^2) = x + y) : x - y ≤ 1/2 := by
  sorry

end max_x_minus_y_l595_59529


namespace toms_age_ratio_l595_59542

theorem toms_age_ratio (T N : ℚ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (number of years in the past)
  (T - N > 0) →  -- Tom's age N years ago was positive
  (T - 4*N > 0) →  -- Sum of children's ages N years ago was positive
  (T - N = 3 * (T - 4*N)) →  -- Condition about Tom's age N years ago
  T / N = 11 / 2 := by
  sorry

end toms_age_ratio_l595_59542


namespace exists_m_divisible_by_1988_l595_59568

/-- Given f(x) = 3x + 2, prove that there exists a positive integer m 
    such that f^(100)(m) is divisible by 1988 -/
theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ+, (3^100 : ℤ) * m.val + (3^100 - 1) ∣ 1988 := by
  sorry

end exists_m_divisible_by_1988_l595_59568


namespace last_three_digits_of_5_to_9000_l595_59589

theorem last_three_digits_of_5_to_9000 (h : 5^300 ≡ 1 [ZMOD 1250]) :
  5^9000 ≡ 1 [ZMOD 1000] := by
  sorry

end last_three_digits_of_5_to_9000_l595_59589


namespace elect_representatives_l595_59576

theorem elect_representatives (total_students : ℕ) (girls : ℕ) (representatives : ℕ) 
  (h1 : total_students = 10) 
  (h2 : girls = 3) 
  (h3 : representatives = 2) : 
  (Nat.choose total_students representatives - Nat.choose (total_students - girls) representatives) = 48 :=
sorry

end elect_representatives_l595_59576


namespace inequality1_solution_inequality2_solution_inequality3_solution_l595_59588

-- Define the inequalities
def inequality1 (x : ℝ) := 2 * x^2 - 3 * x + 1 ≥ 0
def inequality2 (x : ℝ) := x^2 - 2 * x - 3 < 0
def inequality3 (x : ℝ) := -3 * x^2 + 5 * x - 2 > 0

-- Define the solution sets
def solution1 : Set ℝ := {x | x ≤ 1/2 ∨ x ≥ 1}
def solution2 : Set ℝ := {x | -1 < x ∧ x < 3}
def solution3 : Set ℝ := {x | 2/3 < x ∧ x < 1}

-- Theorem statements
theorem inequality1_solution : ∀ x : ℝ, x ∈ solution1 ↔ inequality1 x := by sorry

theorem inequality2_solution : ∀ x : ℝ, x ∈ solution2 ↔ inequality2 x := by sorry

theorem inequality3_solution : ∀ x : ℝ, x ∈ solution3 ↔ inequality3 x := by sorry

end inequality1_solution_inequality2_solution_inequality3_solution_l595_59588


namespace power_function_properties_l595_59586

noncomputable def f (x : ℝ) : ℝ := x^(2/3)

theorem power_function_properties :
  (∃ a : ℝ, ∀ x : ℝ, f x = x^a) ∧ 
  f 8 = 4 ∧
  f 0 = 0 ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y < 0 → f y < f x) := by
  sorry

end power_function_properties_l595_59586


namespace tan_sum_pi_twelfths_l595_59538

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end tan_sum_pi_twelfths_l595_59538


namespace faster_train_speed_l595_59574

/-- The speed of the faster train given the conditions of the problem -/
def speed_of_faster_train (speed_difference : ℝ) (crossing_time : ℝ) (train_length : ℝ) : ℝ :=
  speed_difference * 2

/-- Theorem stating that the speed of the faster train is 72 kmph -/
theorem faster_train_speed :
  speed_of_faster_train 36 15 150 = 72 := by
  sorry

end faster_train_speed_l595_59574


namespace exterior_angle_theorem_l595_59593

-- Define the triangle RWU
structure Triangle (R W U : Type) where
  angle_SWR : ℝ  -- Exterior angle
  angle_WRU : ℝ  -- Interior angle
  angle_WUR : ℝ  -- Interior angle (to be proved)
  straight_line : Prop  -- RTQU forms a straight line

-- State the theorem
theorem exterior_angle_theorem 
  (t : Triangle R W U) 
  (h1 : t.angle_SWR = 50)
  (h2 : t.angle_WRU = 30)
  (h3 : t.straight_line) : 
  t.angle_WUR = 20 := by
sorry

end exterior_angle_theorem_l595_59593


namespace ceiling_minus_x_l595_59580

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 2) : ⌈x⌉ - x = 1/2 := by
  sorry

end ceiling_minus_x_l595_59580


namespace complex_magnitude_l595_59502

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I)^2 = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end complex_magnitude_l595_59502


namespace markers_leftover_l595_59545

theorem markers_leftover (total_markers : ℕ) (num_packages : ℕ) (h1 : total_markers = 154) (h2 : num_packages = 13) :
  total_markers % num_packages = 11 := by
sorry

end markers_leftover_l595_59545


namespace product_from_hcf_lcm_l595_59500

theorem product_from_hcf_lcm (a b : ℕ+) : 
  Nat.gcd a b = 22 → Nat.lcm a b = 2058 → a * b = 45276 := by
  sorry

end product_from_hcf_lcm_l595_59500


namespace saras_house_is_1000_l595_59533

def nadas_house_size : ℕ := 450

def saras_house_size (nadas_size : ℕ) : ℕ :=
  2 * nadas_size + 100

theorem saras_house_is_1000 : saras_house_size nadas_house_size = 1000 := by
  sorry

end saras_house_is_1000_l595_59533


namespace arithmetic_geometric_sequence_common_difference_l595_59510

/-- An arithmetic sequence with the given properties has a common difference of 2 -/
theorem arithmetic_geometric_sequence_common_difference :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence property
  d ≠ 0 →
  a 1 = 18 →
  (a 1) * (a 8) = (a 4)^2 →  -- geometric sequence property
  d = 2 := by
sorry

end arithmetic_geometric_sequence_common_difference_l595_59510


namespace six_term_sequence_count_l595_59517

def sequence_count (n : ℕ) (a b c : ℕ) : ℕ :=
  n.choose c * (n - c).choose b

theorem six_term_sequence_count : sequence_count 6 3 2 1 = 60 := by
  sorry

end six_term_sequence_count_l595_59517


namespace sum_of_marked_angles_l595_59525

/-- Represents the configuration of two overlapping triangles -/
structure OverlappingTriangles where
  -- The four marked angles
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  -- The marked angles are exterior to the quadrilateral formed by the overlap
  exterior_sum : p + q + r + s = 360
  -- Each angle is vertically opposite to another
  vertically_opposite : True

/-- The sum of marked angles in the overlapping triangles configuration is 720° -/
theorem sum_of_marked_angles (ot : OverlappingTriangles) : 
  ot.p + ot.q + ot.r + ot.s + ot.p + ot.q + ot.r + ot.s = 720 :=
by sorry

end sum_of_marked_angles_l595_59525


namespace ice_cream_sundaes_l595_59560

theorem ice_cream_sundaes (total_flavors : ℕ) (h_total : total_flavors = 8) :
  let required_flavor := 1
  let sundae_size := 2
  let max_sundaes := total_flavors - required_flavor
  max_sundaes = 7 :=
by
  sorry

end ice_cream_sundaes_l595_59560


namespace factorial_equation_unique_solution_l595_59561

theorem factorial_equation_unique_solution :
  ∃! m : ℕ+, (Nat.factorial 6) * (Nat.factorial 11) = 18 * (Nat.factorial m.val) * 2 :=
by
  sorry

end factorial_equation_unique_solution_l595_59561


namespace quadratic_inequality_range_l595_59559

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → a ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end quadratic_inequality_range_l595_59559


namespace winnie_min_checks_l595_59543

/-- Represents the arrangement of jars in Winnie the Pooh's closet -/
structure JarArrangement where
  total : Nat
  jam : Nat
  honey : Nat
  honey_consecutive : Bool

/-- Defines the minimum number of jars to check to find honey -/
def min_checks (arrangement : JarArrangement) : Nat :=
  1

/-- Theorem stating that for Winnie's specific arrangement, the minimum number of checks is 1 -/
theorem winnie_min_checks :
  ∀ (arrangement : JarArrangement),
    arrangement.total = 11 →
    arrangement.jam = 7 →
    arrangement.honey = 4 →
    arrangement.honey_consecutive = true →
    min_checks arrangement = 1 := by
  sorry

end winnie_min_checks_l595_59543


namespace proportion_problem_l595_59541

theorem proportion_problem :
  ∃ (a b c d : ℝ),
    (a / b = c / d) ∧
    (a + d = 14) ∧
    (b + c = 11) ∧
    (a^2 + b^2 + c^2 + d^2 = 221) ∧
    (a = 12 ∧ b = 8 ∧ c = 3 ∧ d = 2) := by
  sorry

end proportion_problem_l595_59541


namespace michaels_currency_problem_l595_59569

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Represents the problem of Michael's currency exchange and spending -/
theorem michaels_currency_problem :
  ∃ (d : ℕ),
    (5 / 8 : ℚ) * d - 75 = d ∧
    d = 200 ∧
    sum_of_digits d = 2 := by sorry

end michaels_currency_problem_l595_59569


namespace lexie_family_age_difference_l595_59524

/-- Given the ages and relationships in Lexie's family, prove the age difference between her brother and cousin. -/
theorem lexie_family_age_difference (lexie_age brother_age sister_age uncle_age cousin_age grandma_age : ℕ) : 
  lexie_age = 8 →
  brother_age = lexie_age - 6 →
  sister_age = 2 * lexie_age →
  grandma_age = 68 →
  uncle_age = grandma_age - 12 →
  uncle_age = 3 * sister_age →
  cousin_age = brother_age + 5 →
  cousin_age = uncle_age - 2 →
  cousin_age - brother_age = 5 := by
sorry

end lexie_family_age_difference_l595_59524


namespace customer_outreach_time_calculation_l595_59522

/-- Represents the daily work schedule of a social media account manager --/
structure WorkSchedule where
  total_time : ℝ
  marketing_time : ℝ
  customer_outreach_time : ℝ
  advertisement_time : ℝ

/-- Theorem stating the correct time spent on customer outreach posts --/
theorem customer_outreach_time_calculation (schedule : WorkSchedule) 
  (h1 : schedule.total_time = 8)
  (h2 : schedule.marketing_time = 2)
  (h3 : schedule.advertisement_time = schedule.customer_outreach_time / 2)
  (h4 : schedule.total_time = schedule.marketing_time + schedule.customer_outreach_time + schedule.advertisement_time) :
  schedule.customer_outreach_time = 4 := by
  sorry

end customer_outreach_time_calculation_l595_59522


namespace expression_simplification_l595_59598

theorem expression_simplification (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by sorry

end expression_simplification_l595_59598


namespace expansion_property_l595_59520

/-- Given that for some natural number n, in the expansion of (x^4 + 1/x)^n,
    the binomial coefficient of the third term is 35 more than that of the second term,
    prove that n = 10 and the constant term in the expansion is 45. -/
theorem expansion_property (n : ℕ) 
  (h : Nat.choose n 2 - Nat.choose n 1 = 35) : 
  n = 10 ∧ Nat.choose 10 8 = 45 := by
  sorry

end expansion_property_l595_59520


namespace framing_for_enlarged_picture_l595_59532

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  ((perimeter_inches + 11) / 12 : ℕ)

/-- Theorem stating the minimum number of linear feet of framing needed for the given picture specifications. -/
theorem framing_for_enlarged_picture :
  min_framing_feet 4 6 4 3 = 9 :=
by sorry

end framing_for_enlarged_picture_l595_59532


namespace largest_square_tile_size_l595_59555

theorem largest_square_tile_size (wall_width wall_length : ℕ) 
  (hw : wall_width = 24) (hl : wall_length = 18) :
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧
    wall_width % tile_size = 0 ∧ 
    wall_length % tile_size = 0 ∧
    ∀ (other_size : ℕ), 
      (wall_width % other_size = 0 ∧ wall_length % other_size = 0) → 
      other_size ≤ tile_size :=
by
  -- The proof would go here
  sorry

#check largest_square_tile_size

end largest_square_tile_size_l595_59555


namespace fish_price_proof_l595_59581

theorem fish_price_proof (discount_rate : ℝ) (discounted_price : ℝ) (package_weight : ℝ) :
  discount_rate = 0.4 →
  discounted_price = 2 →
  package_weight = 1/4 →
  (1 - discount_rate) * (1 / package_weight) * discounted_price = 40/3 := by
  sorry

end fish_price_proof_l595_59581


namespace total_raised_is_100_l595_59564

/-- The amount of money raised by a local business for charity -/
def total_raised (num_tickets : ℕ) (ticket_price : ℚ) (donation_15 : ℕ) (donation_20 : ℕ) : ℚ :=
  num_tickets * ticket_price + donation_15 * 15 + donation_20 * 20

/-- Proof that the total amount raised is $100.00 -/
theorem total_raised_is_100 : total_raised 25 2 2 1 = 100 := by
  sorry

end total_raised_is_100_l595_59564


namespace jean_spots_ratio_l595_59577

/-- Jean the jaguar's spot distribution --/
def jean_spots (total_spots upper_torso_spots side_spots : ℕ) : Prop :=
  total_spots = upper_torso_spots + side_spots ∧
  upper_torso_spots = 30 ∧
  side_spots = 10 ∧
  2 * upper_torso_spots = total_spots

theorem jean_spots_ratio :
  ∀ total_spots upper_torso_spots side_spots,
  jean_spots total_spots upper_torso_spots side_spots →
  (total_spots / 2 : ℚ) / total_spots = 1 / 2 := by
  sorry

end jean_spots_ratio_l595_59577


namespace square_diff_equals_32_l595_59503

theorem square_diff_equals_32 (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) :
  a^2 - b^2 = 32 := by
sorry

end square_diff_equals_32_l595_59503


namespace roots_expression_l595_59523

theorem roots_expression (r s : ℝ) (u v s t : ℂ) : 
  (u^2 + r*u + 1 = 0) → 
  (v^2 + r*v + 1 = 0) → 
  (s^2 + s*s + 1 = 0) → 
  (t^2 + s*t + 1 = 0) → 
  (u - s)*(v - s)*(u + t)*(v + t) = s^2 - r^2 := by
  sorry

end roots_expression_l595_59523


namespace era_burger_slices_l595_59551

/-- Given the distribution of burger slices among Era and her friends, prove that Era is left with 1 slice. -/
theorem era_burger_slices (total_burgers : ℕ) (friends : ℕ) (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) (friend4_slices : ℕ) : 
  total_burgers = 5 →
  friends = 4 →
  friend1_slices = 1 →
  friend2_slices = 2 →
  friend3_slices = 3 →
  friend4_slices = 3 →
  total_burgers * 2 - (friend1_slices + friend2_slices + friend3_slices + friend4_slices) = 1 := by
  sorry

end era_burger_slices_l595_59551


namespace shark_teeth_problem_l595_59590

theorem shark_teeth_problem (S : ℚ) 
  (hammerhead : ℚ → ℚ)
  (great_white : ℚ → ℚ)
  (h1 : hammerhead S = S / 6)
  (h2 : great_white S = 2 * (S + hammerhead S))
  (h3 : great_white S = 420) : 
  S = 180 := by sorry

end shark_teeth_problem_l595_59590


namespace complex_number_value_l595_59528

theorem complex_number_value (z : ℂ) (h : (z - 1) * Complex.I = Complex.abs (Complex.I + 1)) :
  z = 1 - Complex.I * Real.sqrt 2 := by
  sorry

end complex_number_value_l595_59528


namespace chord_equation_l595_59547

/-- Given a circle and a chord, prove the equation of the chord. -/
theorem chord_equation (x y : ℝ) :
  (x^2 + y^2 - 4*x - 5 = 0) →  -- Circle equation
  (∃ (a b : ℝ), (a + 3) / 2 = 3 ∧ (b + 1) / 2 = 1) →  -- Midpoint condition
  (x + y - 4 = 0) :=  -- Equation of line AB
by sorry

end chord_equation_l595_59547


namespace subset_of_complement_iff_a_in_range_l595_59570

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M (a : ℝ) : Set ℝ := {x | 3 * a - 1 < x ∧ x < 2 * a}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem subset_of_complement_iff_a_in_range (a : ℝ) :
  N ⊆ (U \ M a) ↔ a ≤ -1/2 ∨ a ≥ 1 := by sorry

end subset_of_complement_iff_a_in_range_l595_59570


namespace line_equation_l595_59554

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passesThrough (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- The main theorem
theorem line_equation (l : Line2D) :
  passesThrough l ⟨1, 4⟩ ∧ hasEqualIntercepts l →
  (l.a = 4 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -5) :=
sorry

end line_equation_l595_59554


namespace f_monotonicity_l595_59585

noncomputable def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 7

theorem f_monotonicity :
  (∀ x y, x < y ∧ y < 0 → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y) ∧
  (∀ x y, 2 < x ∧ x < y → f x < f y) :=
sorry

end f_monotonicity_l595_59585


namespace rock_collection_total_l595_59596

theorem rock_collection_total (igneous sedimentary : ℕ) : 
  igneous = sedimentary / 2 →
  (2 : ℕ) * igneous / 3 = 40 →
  igneous + sedimentary = 180 :=
by
  sorry

#check rock_collection_total

end rock_collection_total_l595_59596


namespace candy_mixture_cost_l595_59572

/-- Given a mixture of two types of candy, prove the cost of the second type. -/
theorem candy_mixture_cost
  (total_mixture : ℝ)
  (first_candy_weight : ℝ)
  (first_candy_cost : ℝ)
  (second_candy_weight : ℝ)
  (mixture_cost : ℝ)
  (h1 : total_mixture = first_candy_weight + second_candy_weight)
  (h2 : total_mixture = 45)
  (h3 : first_candy_weight = 15)
  (h4 : first_candy_cost = 8)
  (h5 : second_candy_weight = 30)
  (h6 : mixture_cost = 6) :
  ∃ (second_candy_cost : ℝ),
    second_candy_cost = 5 ∧
    total_mixture * mixture_cost =
      first_candy_weight * first_candy_cost +
      second_candy_weight * second_candy_cost :=
by sorry

end candy_mixture_cost_l595_59572


namespace train_speed_l595_59558

/-- Given a bridge and a train, calculate the train's speed in km/h -/
theorem train_speed (bridge_length train_length : ℝ) (crossing_time : ℝ) : 
  bridge_length = 200 →
  train_length = 100 →
  crossing_time = 60 →
  (bridge_length + train_length) / crossing_time * 3.6 = 18 := by
  sorry

#check train_speed

end train_speed_l595_59558


namespace fraction_sequence_2012th_term_l595_59562

/-- Represents the sequence of fractions as described in the problem -/
def fraction_sequence : ℕ → ℚ :=
  sorry

/-- The sum of the first n positive integers -/
def triangle_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem fraction_sequence_2012th_term :
  ∃ (n : ℕ), 
    triangle_number n ≤ 2012 ∧ 
    triangle_number (n + 1) > 2012 ∧
    63 * 64 / 2 = 2016 ∧
    fraction_sequence 2012 = 5 / 59 :=
by
  sorry

end fraction_sequence_2012th_term_l595_59562


namespace pet_store_cats_l595_59584

theorem pet_store_cats (initial_siamese : ℕ) (sold : ℕ) (remaining : ℕ) (initial_house : ℕ) : 
  initial_siamese = 13 → 
  sold = 10 → 
  remaining = 8 → 
  initial_siamese + initial_house - sold = remaining → 
  initial_house = 5 := by
sorry

end pet_store_cats_l595_59584


namespace trajectory_equation_l595_59556

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the property of being tangent internally to C₁ and tangent to C₂
def is_tangent_to_circles (x y : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    (∀ p q : ℝ, C₁ p q → (x - p)^2 + (y - q)^2 = (r - 1)^2) ∧
    (∀ p q : ℝ, C₂ p q → (x - p)^2 + (y - q)^2 = (r + 9)^2)

-- Theorem statement
theorem trajectory_equation :
  ∀ x y : ℝ, is_tangent_to_circles x y ↔ x^2/16 + y^2/7 = 1 :=
sorry

end trajectory_equation_l595_59556


namespace total_chocolate_pieces_l595_59507

theorem total_chocolate_pieces (num_boxes : ℕ) (pieces_per_box : ℕ) 
  (h1 : num_boxes = 6) 
  (h2 : pieces_per_box = 500) : 
  num_boxes * pieces_per_box = 3000 := by
  sorry

end total_chocolate_pieces_l595_59507


namespace circle_area_ratio_l595_59544

theorem circle_area_ratio (d : ℝ) (h : d > 0) : 
  (π * ((7 * d) / 2)^2) / (π * (d / 2)^2) = 49 := by
  sorry

end circle_area_ratio_l595_59544


namespace parabola_translation_l595_59512

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c + v }

theorem parabola_translation (x : ℝ) :
  let original := Parabola.mk 3 0 0
  let translated := translate original 1 2
  translated.a * x^2 + translated.b * x + translated.c = 3 * (x + 1)^2 + 2 := by
  sorry

end parabola_translation_l595_59512


namespace intersection_of_sets_l595_59511

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | 2 < x ∧ x < 4}
  let B : Set ℝ := {x | x > 5/3}
  A ∩ B = {x | 2 < x ∧ x < 3} := by
sorry

end intersection_of_sets_l595_59511


namespace solution_is_correct_l595_59563

/-- The equation to be solved -/
def equation (y : ℝ) : Prop :=
  1/6 + 6/y = 14/y - 1/14

/-- The theorem stating that y = 168/5 is the solution to the equation -/
theorem solution_is_correct : equation (168/5) := by
  sorry

end solution_is_correct_l595_59563


namespace solve_for_y_l595_59579

theorem solve_for_y (x y : ℝ) : 4 * x + y = 9 → y = 9 - 4 * x := by
  sorry

end solve_for_y_l595_59579


namespace radius_of_C₁_is_8_l595_59509

-- Define the points and circles
variable (O X Y Z : ℝ × ℝ)
variable (C₁ C₂ : Set (ℝ × ℝ))

-- Define the conditions
variable (h₁ : O ∈ C₂)
variable (h₂ : X ∈ C₁ ∩ C₂)
variable (h₃ : Y ∈ C₁ ∩ C₂)
variable (h₄ : Z ∈ C₂)
variable (h₅ : Z ∉ C₁)
variable (h₆ : ‖X - Z‖ = 15)
variable (h₇ : ‖O - Z‖ = 17)
variable (h₈ : ‖Y - Z‖ = 8)
variable (h₉ : (X - O) • (Z - O) = 0)  -- Right angle at X

-- Define the radius of C₁
def radius_C₁ (O X : ℝ × ℝ) : ℝ := ‖X - O‖

-- Theorem statement
theorem radius_of_C₁_is_8 :
  radius_C₁ O X = 8 :=
sorry

end radius_of_C₁_is_8_l595_59509


namespace number_of_intersection_points_l595_59514

-- Define the line equation
def line (x : ℝ) : ℝ := x + 3

-- Define the curve equation
def curve (x y : ℝ) : Prop := y^2 / 9 - (x * abs x) / 4 = 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  y = line x ∧ curve x y

-- Theorem statement
theorem number_of_intersection_points :
  ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    is_intersection_point p₁.1 p₁.2 ∧
    is_intersection_point p₂.1 p₂.2 ∧
    is_intersection_point p₃.1 p₃.2 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    ∀ (q : ℝ × ℝ), is_intersection_point q.1 q.2 → q = p₁ ∨ q = p₂ ∨ q = p₃ :=
by sorry

end number_of_intersection_points_l595_59514


namespace deceased_member_income_l595_59550

/-- Calculates the income of a deceased family member given the initial and final family situations. -/
theorem deceased_member_income
  (initial_members : ℕ)
  (initial_average_income : ℚ)
  (final_members : ℕ)
  (final_average_income : ℚ)
  (h1 : initial_members = 4)
  (h2 : final_members = 3)
  (h3 : initial_average_income = 735)
  (h4 : final_average_income = 650) :
  (initial_members : ℚ) * initial_average_income - (final_members : ℚ) * final_average_income = 990 :=
by sorry

end deceased_member_income_l595_59550


namespace quadratic_equation_integer_solutions_l595_59513

theorem quadratic_equation_integer_solutions :
  ∀ x y : ℤ, 
    x^2 + 2*x*y + 3*y^2 - 2*x + y + 1 = 0 ↔ 
    ((x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -1) ∨ (x = 3 ∧ y = -1)) :=
by sorry

end quadratic_equation_integer_solutions_l595_59513


namespace gasoline_needed_for_distance_l595_59582

/-- Given a car with fuel efficiency and a known fuel consumption for a specific distance,
    calculate the amount of gasoline needed for any distance. -/
theorem gasoline_needed_for_distance (fuel_efficiency : ℝ) (known_distance : ℝ) (known_gasoline : ℝ) (distance : ℝ) :
  fuel_efficiency = 20 →
  known_distance = 130 →
  known_gasoline = 6.5 →
  known_distance / known_gasoline = fuel_efficiency →
  distance / fuel_efficiency = distance / 20 := by
  sorry

end gasoline_needed_for_distance_l595_59582


namespace bag_selling_price_l595_59504

/-- The selling price of a discounted item -/
def selling_price (marked_price : ℝ) (discount_rate : ℝ) : ℝ :=
  marked_price * (1 - discount_rate)

/-- Theorem: The selling price of a bag marked at $80 with a 15% discount is $68 -/
theorem bag_selling_price :
  selling_price 80 0.15 = 68 := by
  sorry

end bag_selling_price_l595_59504


namespace solution_set_of_f_minimum_value_of_sum_l595_59591

-- Part I
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem solution_set_of_f (x : ℝ) : f (x + 3/2) ≥ 0 ↔ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) := by sorry

-- Part II
theorem minimum_value_of_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 ∧ ∃ (p' q' r' : ℝ), p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ 
    1/(3*p') + 1/(2*q') + 1/r' = 4 ∧ 3*p' + 2*q' + r' = 9/4 := by sorry

end solution_set_of_f_minimum_value_of_sum_l595_59591


namespace amp_composition_l595_59567

-- Define the & operations
def postfix_amp (x : ℤ) : ℤ := 8 - x
def prefix_amp (x : ℤ) : ℤ := x - 8

-- The theorem to prove
theorem amp_composition : prefix_amp (postfix_amp 15) = -15 := by
  sorry

end amp_composition_l595_59567


namespace shekar_average_marks_l595_59578

def shekar_scores : List ℕ := [76, 65, 82, 67, 55]

theorem shekar_average_marks :
  (shekar_scores.sum / shekar_scores.length : ℚ) = 69 := by
  sorry

end shekar_average_marks_l595_59578


namespace union_definition_l595_59552

theorem union_definition (A B : Set α) : 
  A ∪ B = {x | x ∈ A ∨ x ∈ B} := by
sorry

end union_definition_l595_59552


namespace max_n_for_positive_an_l595_59583

theorem max_n_for_positive_an (n : ℕ) : 
  (∀ k : ℕ, k > n → (19 : ℤ) - 2 * k ≤ 0) ∧ 
  ((19 : ℤ) - 2 * n > 0) → 
  n = 9 := by
sorry

end max_n_for_positive_an_l595_59583


namespace sum_of_angles_l595_59599

-- Define a rectangle
structure Rectangle where
  angles : ℕ
  is_rectangle : angles = 4

-- Define a square
structure Square where
  angles : ℕ
  is_square : angles = 4

-- Theorem statement
theorem sum_of_angles (rect : Rectangle) (sq : Square) : rect.angles + sq.angles = 8 := by
  sorry

end sum_of_angles_l595_59599


namespace emma_chocolates_l595_59521

theorem emma_chocolates (emma liam : ℕ) : 
  emma = liam + 10 →
  liam = emma / 3 →
  emma = 15 := by
sorry

end emma_chocolates_l595_59521


namespace total_eyes_in_pond_l595_59515

/-- The number of snakes in the pond -/
def num_snakes : ℕ := 18

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

/-- The number of spiders in the pond -/
def num_spiders : ℕ := 5

/-- The number of snails in the pond -/
def num_snails : ℕ := 15

/-- The number of eyes a snake has -/
def snake_eyes : ℕ := 2

/-- The number of eyes an alligator has -/
def alligator_eyes : ℕ := 2

/-- The number of eyes a spider has -/
def spider_eyes : ℕ := 8

/-- The number of eyes a snail has -/
def snail_eyes : ℕ := 2

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := num_snakes * snake_eyes + num_alligators * alligator_eyes + 
                      num_spiders * spider_eyes + num_snails * snail_eyes

theorem total_eyes_in_pond : total_eyes = 126 := by sorry

end total_eyes_in_pond_l595_59515


namespace cubic_equation_solutions_l595_59535

def is_solution (x y z : ℤ) : Prop :=
  x^3 + y^3 + z^3 - 3*x*y*z = 2003

theorem cubic_equation_solutions :
  ∀ x y z : ℤ, is_solution x y z ↔ 
    ((x = 668 ∧ y = 668 ∧ z = 667) ∨
     (x = 668 ∧ y = 667 ∧ z = 668) ∨
     (x = 667 ∧ y = 668 ∧ z = 668)) :=
by sorry

end cubic_equation_solutions_l595_59535


namespace linear_function_translation_l595_59516

/-- A linear function passing through a specific point -/
def passes_through (b : ℝ) : Prop :=
  3 = 2 + b

/-- The correct value of b for the translated line -/
def correct_b : ℝ := 1

/-- Theorem stating that the linear function passes through (2, 3) iff b = 1 -/
theorem linear_function_translation :
  passes_through correct_b ∧ 
  (∀ b : ℝ, passes_through b → b = correct_b) :=
sorry

end linear_function_translation_l595_59516


namespace negative_quarter_power_times_four_power_l595_59587

theorem negative_quarter_power_times_four_power (n : ℕ) :
  ((-0.25 : ℝ) ^ n) * (4 ^ (n + 1)) = 4 := by
  sorry

end negative_quarter_power_times_four_power_l595_59587


namespace projection_sphere_existence_l595_59546

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a type for a set of lines
def LineSet := List Line3D

-- Function to check if lines are pairwise non-parallel
def pairwiseNonParallel (lines : LineSet) : Prop := sorry

-- Function to perform orthogonal projection
def orthogonalProject (p : Point3D) (l : Line3D) : Point3D := sorry

-- Function to generate all points from repeated projections
def allProjectionPoints (o : Point3D) (lines : LineSet) : Set Point3D := sorry

-- Theorem statement
theorem projection_sphere_existence 
  (o : Point3D) 
  (lines : LineSet) 
  (h : pairwiseNonParallel lines) :
  ∃ (r : ℝ), ∀ p ∈ allProjectionPoints o lines, 
    (p.x - o.x)^2 + (p.y - o.y)^2 + (p.z - o.z)^2 ≤ r^2 := by
  sorry

end projection_sphere_existence_l595_59546


namespace jane_age_ratio_l595_59595

/-- Represents the ages of Jane and her children at two different times -/
structure FamilyAges where
  J : ℝ  -- Jane's current age
  M : ℝ  -- Years ago
  younger_sum : ℝ  -- Sum of ages of two younger children
  oldest : ℝ  -- Age of oldest child

/-- The conditions given in the problem -/
def satisfies_conditions (ages : FamilyAges) : Prop :=
  ages.J > 0 ∧ 
  ages.M > 0 ∧
  ages.J = 2 * ages.younger_sum ∧
  ages.J = ages.oldest / 2 ∧
  ages.J - ages.M = 3 * (ages.younger_sum - 2 * ages.M) ∧
  ages.J - ages.M = ages.oldest - ages.M

theorem jane_age_ratio (ages : FamilyAges) 
  (h : satisfies_conditions ages) : ages.J / ages.M = 10 := by
  sorry

end jane_age_ratio_l595_59595


namespace goats_sold_proof_l595_59557

/-- Represents the number of animals sold -/
def total_animals : ℕ := 80

/-- Represents the total reduction in legs -/
def total_leg_reduction : ℕ := 200

/-- Represents the number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- Represents the number of legs a goat has -/
def goat_legs : ℕ := 4

/-- Represents the number of goats sold -/
def goats_sold : ℕ := 20

/-- Represents the number of chickens sold -/
def chickens_sold : ℕ := total_animals - goats_sold

theorem goats_sold_proof :
  goats_sold * goat_legs + chickens_sold * chicken_legs = total_leg_reduction ∧
  goats_sold + chickens_sold = total_animals :=
by sorry

end goats_sold_proof_l595_59557


namespace expected_same_color_edges_l595_59536

/-- Represents a 3x3 board -/
def Board := Fin 3 → Fin 3 → Bool

/-- The number of squares in the board -/
def boardSize : Nat := 9

/-- The number of squares to be blackened -/
def blackSquares : Nat := 5

/-- The total number of pairs of adjacent squares -/
def totalAdjacentPairs : Nat := 12

/-- Calculates the probability that two adjacent squares have the same color -/
noncomputable def probSameColor : ℚ := 4 / 9

/-- Theorem: The expected number of edges between two squares of the same color 
    in a 3x3 board with 5 randomly blackened squares is 16/3 -/
theorem expected_same_color_edges :
  (totalAdjacentPairs : ℚ) * probSameColor = 16 / 3 := by sorry

end expected_same_color_edges_l595_59536


namespace inequality_proof_l595_59594

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_condition : a * b * c * (a + b + c) = a * b + b * c + c * a) :
  5 * (a + b + c) ≥ 7 + 8 * a * b * c := by
  sorry

end inequality_proof_l595_59594


namespace chameleon_color_change_l595_59549

theorem chameleon_color_change (total : ℕ) (blue_factor red_factor : ℕ) : 
  total = 140 → blue_factor = 5 → red_factor = 3 →
  ∃ (initial_blue : ℕ),
    initial_blue > 0 ∧
    initial_blue * blue_factor ≤ total ∧
    (total - initial_blue * blue_factor) * red_factor + initial_blue = total →
    initial_blue * blue_factor - initial_blue = 80 := by
  sorry

#check chameleon_color_change

end chameleon_color_change_l595_59549


namespace defective_probability_l595_59501

/-- The probability that both selected products are defective given that one is defective -/
theorem defective_probability (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ) 
  (h1 : total = genuine + defective)
  (h2 : total = 10)
  (h3 : genuine = 6)
  (h4 : defective = 4)
  (h5 : selected = 2) :
  (defective.choose 2 : ℚ) / (total.choose 2) / 
  ((defective.choose 1 * genuine.choose 1 + defective.choose 2 : ℚ) / total.choose 2) = 1 / 5 := by
sorry

end defective_probability_l595_59501


namespace chipped_marbles_possibilities_l595_59571

def marble_counts : List Nat := [15, 18, 20, 22, 24, 27, 30, 32, 35, 37]

def total_marbles : Nat := marble_counts.sum

theorem chipped_marbles_possibilities :
  ∀ n : Nat, n ∈ marble_counts →
  (total_marbles - n) % 5 = 0 →
  n % 5 = 0 →
  n ∈ [15, 20, 30, 35] :=
by sorry

end chipped_marbles_possibilities_l595_59571


namespace twenty_four_game_solution_l595_59534

theorem twenty_four_game_solution : 
  let a : ℝ := 5
  let b : ℝ := 5
  let c : ℝ := 5
  let d : ℝ := 1
  (a - d / b) * c = 24 := by
  sorry

end twenty_four_game_solution_l595_59534


namespace dining_sales_tax_percentage_l595_59526

/-- Proves that the sales tax percentage is 10% given the conditions of the dining problem -/
theorem dining_sales_tax_percentage : 
  ∀ (total_spent food_price tip_percentage sales_tax_percentage : ℝ),
  total_spent = 132 →
  food_price = 100 →
  tip_percentage = 20 →
  total_spent = food_price * (1 + sales_tax_percentage / 100) * (1 + tip_percentage / 100) →
  sales_tax_percentage = 10 := by
sorry


end dining_sales_tax_percentage_l595_59526


namespace magnitude_z_l595_59537

theorem magnitude_z (w z : ℂ) (h1 : w * z = 15 - 20 * I) (h2 : Complex.abs w = Real.sqrt 34) :
  Complex.abs z = (25 * Real.sqrt 34) / 34 := by
  sorry

end magnitude_z_l595_59537


namespace product_of_even_or_odd_is_even_l595_59540

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the product of two functions
def FunctionProduct (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x * g x

-- State the theorem
theorem product_of_even_or_odd_is_even 
  (f φ : ℝ → ℝ) 
  (h : (IsEven f ∧ IsEven φ) ∨ (IsOdd f ∧ IsOdd φ)) : 
  IsEven (FunctionProduct f φ) := by
  sorry

end product_of_even_or_odd_is_even_l595_59540
