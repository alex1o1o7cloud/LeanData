import Mathlib

namespace speed_of_sound_l396_39688

/-- The speed of sound given specific conditions --/
theorem speed_of_sound (travel_time : Real) (blast_interval : Real) (distance : Real) :
  travel_time = 30.0 + 25.0 / 60 → -- 30 minutes and 25 seconds in hours
  blast_interval = 0.5 → -- 30 minutes in hours
  distance = 8250 → -- distance in meters
  (distance / (travel_time - blast_interval)) * (1 / 3600) = 330 := by
  sorry

end speed_of_sound_l396_39688


namespace older_brother_running_distance_l396_39677

/-- The running speed of the older brother in meters per minute -/
def older_brother_speed : ℝ := 110

/-- The running speed of the younger brother in meters per minute -/
def younger_brother_speed : ℝ := 80

/-- The additional time the younger brother runs in minutes -/
def additional_time : ℝ := 30

/-- The additional distance the younger brother runs in meters -/
def additional_distance : ℝ := 900

/-- The distance run by the older brother in meters -/
def older_brother_distance : ℝ := 5500

theorem older_brother_running_distance :
  ∃ (t : ℝ), 
    t > 0 ∧
    (t + additional_time) * younger_brother_speed = t * older_brother_speed + additional_distance ∧
    t * older_brother_speed = older_brother_distance :=
by sorry

end older_brother_running_distance_l396_39677


namespace minimize_expression_l396_39642

theorem minimize_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧
  ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
sorry

end minimize_expression_l396_39642


namespace hexagram_arrangement_count_l396_39683

/-- A hexagram is a regular six-pointed star with 12 points of intersection -/
structure Hexagram :=
  (points : Fin 12 → α)

/-- The number of symmetries of a hexagram (rotations and reflections) -/
def hexagram_symmetries : ℕ := 12

/-- The number of distinct arrangements of 12 unique objects on a hexagram,
    considering rotations and reflections as equivalent -/
def distinct_hexagram_arrangements : ℕ := Nat.factorial 12 / hexagram_symmetries

theorem hexagram_arrangement_count :
  distinct_hexagram_arrangements = 39916800 := by sorry

end hexagram_arrangement_count_l396_39683


namespace sheep_to_cow_ratio_is_ten_to_one_l396_39660

/-- Represents the farm owned by Mr. Reyansh -/
structure Farm where
  num_cows : ℕ
  cow_water_daily : ℕ
  sheep_water_ratio : ℚ
  total_water_weekly : ℕ

/-- Calculates the ratio of sheep to cows on the farm -/
def sheep_to_cow_ratio (f : Farm) : ℚ :=
  let cow_water_weekly := f.num_cows * f.cow_water_daily * 7
  let sheep_water_weekly := f.total_water_weekly - cow_water_weekly
  let sheep_water_daily := sheep_water_weekly / 7
  let num_sheep := sheep_water_daily / (f.cow_water_daily * f.sheep_water_ratio)
  num_sheep / f.num_cows

/-- Theorem stating that the ratio of sheep to cows is 10:1 -/
theorem sheep_to_cow_ratio_is_ten_to_one (f : Farm) 
    (h1 : f.num_cows = 40)
    (h2 : f.cow_water_daily = 80)
    (h3 : f.sheep_water_ratio = 1/4)
    (h4 : f.total_water_weekly = 78400) :
  sheep_to_cow_ratio f = 10 := by
  sorry

#eval sheep_to_cow_ratio { num_cows := 40, cow_water_daily := 80, sheep_water_ratio := 1/4, total_water_weekly := 78400 }

end sheep_to_cow_ratio_is_ten_to_one_l396_39660


namespace largest_band_size_l396_39699

theorem largest_band_size : ∃ (m r x : ℕ),
  m < 150 ∧
  r * x + 3 = m ∧
  (r - 3) * (x + 2) = m ∧
  ∀ (m' r' x' : ℕ),
    m' < 150 →
    r' * x' + 3 = m' →
    (r' - 3) * (x' + 2) = m' →
    m' ≤ m ∧
  m = 107 := by
sorry

end largest_band_size_l396_39699


namespace line_passes_through_intercepts_l396_39629

/-- A line that intersects the x-axis at (3, 0) and the y-axis at (0, -5) -/
def line_equation (x y : ℝ) : Prop :=
  x / 3 - y / 5 = 1

/-- The x-intercept of the line -/
def x_intercept : ℝ := 3

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

theorem line_passes_through_intercepts :
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept := by
  sorry

end line_passes_through_intercepts_l396_39629


namespace b_minus_c_equals_one_l396_39682

theorem b_minus_c_equals_one (A B C : ℤ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h2 : A = 9 - 4)
  (h3 : B = A + 5)
  (h4 : C - 8 = 1) : 
  B - C = 1 := by
  sorry

end b_minus_c_equals_one_l396_39682


namespace polynomial_factorization_l396_39693

theorem polynomial_factorization (x : ℤ) : 
  x^5 + x^4 + 1 = (x^3 - x + 1) * (x^2 + x + 1) := by
  sorry

end polynomial_factorization_l396_39693


namespace smallest_N_l396_39671

theorem smallest_N (k : ℕ) (hk : k ≥ 1) :
  let N := 2 * k^3 + 3 * k^2 + k
  ∀ (a : Fin (2 * k + 1) → ℕ),
    (∀ i, a i ≥ 1) →
    (∀ i j, i ≠ j → a i ≠ a j) →
    (Finset.sum Finset.univ a ≥ N) →
    (∀ s : Finset (Fin (2 * k + 1)), s.card = k → Finset.sum s a ≤ N / 2) →
    ∀ M : ℕ, M < N →
      ¬∃ (b : Fin (2 * k + 1) → ℕ),
        (∀ i, b i ≥ 1) ∧
        (∀ i j, i ≠ j → b i ≠ b j) ∧
        (Finset.sum Finset.univ b ≥ M) ∧
        (∀ s : Finset (Fin (2 * k + 1)), s.card = k → Finset.sum s b ≤ M / 2) :=
by sorry

end smallest_N_l396_39671


namespace arithmetic_sequence_common_difference_l396_39675

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that the common difference d equals 2 when (S_3 / 3) - (S_2 / 2) = 1 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) 
  (h_condition : S 3 / 3 - S 2 / 2 = 1) :
  a 2 - a 1 = 2 :=
sorry

end arithmetic_sequence_common_difference_l396_39675


namespace average_age_of_first_seven_students_l396_39606

theorem average_age_of_first_seven_students 
  (total_students : Nat) 
  (average_age_all : ℚ) 
  (second_group_size : Nat) 
  (average_age_second_group : ℚ) 
  (age_last_student : ℚ) 
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : second_group_size = 7)
  (h4 : average_age_second_group = 16)
  (h5 : age_last_student = 15) :
  let first_group_size := total_students - second_group_size - 1
  let total_age := average_age_all * total_students
  let second_group_total_age := average_age_second_group * second_group_size
  let first_group_total_age := total_age - second_group_total_age - age_last_student
  first_group_total_age / first_group_size = 14 := by
  sorry

end average_age_of_first_seven_students_l396_39606


namespace bus_interval_theorem_l396_39648

/-- The interval between buses on a circular route -/
def interval (num_buses : ℕ) (total_time : ℕ) : ℕ :=
  total_time / num_buses

/-- The theorem stating the relationship between intervals for 2 and 3 buses -/
theorem bus_interval_theorem (initial_interval : ℕ) :
  initial_interval = 21 →
  interval 2 (2 * initial_interval) = 21 →
  interval 3 (2 * initial_interval) = 14 :=
by
  sorry

#check bus_interval_theorem

end bus_interval_theorem_l396_39648


namespace age_problem_l396_39669

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = c / 2 →
  a + b + c + d = 39 →
  b = 14 := by
sorry

end age_problem_l396_39669


namespace area_ratio_of_concentric_circles_l396_39687

/-- Two concentric circles with center Q -/
structure ConcentricCircles where
  center : Point
  smallerRadius : ℝ
  largerRadius : ℝ
  smallerRadius_pos : 0 < smallerRadius
  largerRadius_pos : 0 < largerRadius
  smallerRadius_lt_largerRadius : smallerRadius < largerRadius

/-- The arc length of a circle given its radius and central angle (in radians) -/
def arcLength (radius : ℝ) (angle : ℝ) : ℝ := radius * angle

theorem area_ratio_of_concentric_circles 
  (circles : ConcentricCircles) 
  (h : arcLength circles.smallerRadius (π/3) = arcLength circles.largerRadius (π/6)) : 
  (circles.smallerRadius^2) / (circles.largerRadius^2) = 1/4 := by
sorry

end area_ratio_of_concentric_circles_l396_39687


namespace range_of_a_l396_39620

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1 - 2*a

def g (a : ℝ) (x : ℝ) : ℝ := |x - a| - a*x

def has_two_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f a x₁ = 0 ∧ f a x₂ = 0

def has_minimum_value (a : ℝ) : Prop :=
  ∃ x₀, ∀ x, g a x₀ ≤ g a x

theorem range_of_a (a : ℝ) :
  a > 0 ∧ ¬(has_two_distinct_intersections a) ∧ has_minimum_value a →
  a ∈ Set.Ioo 0 (Real.sqrt 2 - 1) ∪ Set.Ioo (1/2) 1 :=
sorry

end range_of_a_l396_39620


namespace three_divided_by_p_l396_39618

theorem three_divided_by_p (p q : ℝ) 
  (h1 : 3 / q = 18) 
  (h2 : p - q = 0.33333333333333337) : 
  3 / p = 6 := by
  sorry

end three_divided_by_p_l396_39618


namespace diophantine_equation_solutions_l396_39643

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, m^2 - 2*m*n - 3*n^2 = 5 ↔ 
    ((m = 4 ∧ n = 1) ∨ (m = 2 ∧ n = -1) ∨ (m = -4 ∧ n = -1) ∨ (m = -2 ∧ n = 1)) :=
by sorry

end diophantine_equation_solutions_l396_39643


namespace cannot_reach_goal_l396_39657

/-- Represents the types of donuts --/
inductive DonutType
  | Plain
  | Glazed
  | Chocolate

/-- Represents the cost and price information for donuts --/
structure DonutInfo where
  costPerDozen : ℝ
  sellingPrice : ℝ

/-- The goal amount to be raised --/
def goalAmount : ℝ := 96

/-- The maximum number of dozens that can be bought --/
def maxDozens : ℕ := 6

/-- The number of donut types --/
def numTypes : ℕ := 3

/-- The donut information for each type --/
def donutData : DonutType → DonutInfo
  | DonutType.Plain => { costPerDozen := 2.4, sellingPrice := 1 }
  | DonutType.Glazed => { costPerDozen := 3.6, sellingPrice := 1.5 }
  | DonutType.Chocolate => { costPerDozen := 4.8, sellingPrice := 2 }

/-- Calculate the profit for a given number of dozens of a specific donut type --/
def profitForType (t : DonutType) (dozens : ℝ) : ℝ :=
  let info := donutData t
  dozens * (12 * info.sellingPrice - info.costPerDozen)

/-- The main theorem stating that the goal cannot be reached --/
theorem cannot_reach_goal :
  ∀ x : ℝ, x > 0 → x ≤ (maxDozens / numTypes : ℝ) →
  (profitForType DonutType.Plain x +
   profitForType DonutType.Glazed x +
   profitForType DonutType.Chocolate x) < goalAmount :=
sorry

end cannot_reach_goal_l396_39657


namespace stock_annual_return_l396_39673

/-- Calculates the annual return percentage given initial price and price increase -/
def annual_return_percentage (initial_price price_increase : ℚ) : ℚ :=
  (price_increase / initial_price) * 100

/-- Theorem: The annual return percentage for a stock with initial price 8000 and price increase 400 is 5% -/
theorem stock_annual_return :
  let initial_price : ℚ := 8000
  let price_increase : ℚ := 400
  annual_return_percentage initial_price price_increase = 5 := by
  sorry

#eval annual_return_percentage 8000 400

end stock_annual_return_l396_39673


namespace triangle_isosceles_if_2cosB_sinA_eq_sinC_l396_39638

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- State the theorem
theorem triangle_isosceles_if_2cosB_sinA_eq_sinC (t : Triangle) :
  2 * Real.cos t.B * Real.sin t.A = Real.sin t.C → isIsosceles t :=
by
  sorry


end triangle_isosceles_if_2cosB_sinA_eq_sinC_l396_39638


namespace initial_men_correct_l396_39601

/-- Represents the initial number of men working on the project -/
def initial_men : ℕ := 27

/-- Represents the number of days to complete the project with the initial group -/
def initial_days : ℕ := 40

/-- Represents the number of days worked before some men leave -/
def days_before_leaving : ℕ := 18

/-- Represents the number of men who leave the project -/
def men_leaving : ℕ := 12

/-- Represents the number of days to complete the project after some men leave -/
def remaining_days : ℕ := 40

/-- Theorem stating that the initial number of men is correct given the conditions -/
theorem initial_men_correct :
  (initial_men : ℚ) * (days_before_leaving : ℚ) / initial_days +
  (initial_men - men_leaving : ℚ) * remaining_days / initial_days = 1 :=
sorry

end initial_men_correct_l396_39601


namespace sequence_determination_l396_39645

theorem sequence_determination (p : ℕ) (hp : p.Prime ∧ p > 5) :
  ∀ (a : Fin ((p - 1) / 2) → ℕ),
  (∀ i, a i ∈ Finset.range ((p - 1) / 2 + 1) \ {0}) →
  (∀ i j, i ≠ j → ∃ r, (a i * a j) % p = r) →
  Function.Injective a :=
sorry

end sequence_determination_l396_39645


namespace negate_positive_negate_negative_positive_negative_positive_positive_l396_39610

-- Define the operations
def negate (x : ℝ) : ℝ := -x
def positive (x : ℝ) : ℝ := x

-- Theorem statements
theorem negate_positive (x : ℝ) : negate (positive x) = -x := by sorry

theorem negate_negative (x : ℝ) : negate (negate x) = x := by sorry

theorem positive_negative (x : ℝ) : positive (negate x) = -x := by sorry

theorem positive_positive (x : ℝ) : positive (positive x) = x := by sorry

end negate_positive_negate_negative_positive_negative_positive_positive_l396_39610


namespace smaller_angle_at_8_is_120_l396_39628

/-- The number of hour marks on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The hour being considered (8 o'clock) -/
def current_hour : ℕ := 8

/-- The angle between adjacent hour marks on a clock face -/
def angle_between_hours : ℚ := full_circle_degrees / clock_hours

/-- The position of the hour hand at the current hour -/
def hour_hand_position : ℚ := current_hour * angle_between_hours

/-- The smaller angle between clock hands at the given hour -/
def smaller_angle_at_hour (h : ℕ) : ℚ :=
  min (h * angle_between_hours) (full_circle_degrees - h * angle_between_hours)

theorem smaller_angle_at_8_is_120 :
  smaller_angle_at_hour current_hour = 120 :=
sorry

end smaller_angle_at_8_is_120_l396_39628


namespace total_liquid_consumed_l396_39639

/-- Proves that the total amount of liquid consumed by Yurim and Ji-in is 6300 milliliters -/
theorem total_liquid_consumed (yurim_liters : ℕ) (yurim_ml : ℕ) (jiin_ml : ℕ) :
  yurim_liters = 2 →
  yurim_ml = 600 →
  jiin_ml = 3700 →
  yurim_liters * 1000 + yurim_ml + jiin_ml = 6300 :=
by
  sorry

end total_liquid_consumed_l396_39639


namespace john_daily_gallons_l396_39696

-- Define the conversion rate from quarts to gallons
def quarts_per_gallon : ℚ := 4

-- Define the number of days in a week
def days_per_week : ℚ := 7

-- Define John's weekly water consumption in quarts
def john_weekly_quarts : ℚ := 42

-- Theorem to prove
theorem john_daily_gallons : 
  john_weekly_quarts / quarts_per_gallon / days_per_week = 1.5 := by
  sorry

end john_daily_gallons_l396_39696


namespace parallel_line_m_value_l396_39625

/-- Given a line passing through points A(-2, m) and B(m, 4) that is parallel to the line 2x + y - 1 = 0, prove that m = -8 -/
theorem parallel_line_m_value :
  ∀ m : ℝ,
  let A : ℝ × ℝ := (-2, m)
  let B : ℝ × ℝ := (m, 4)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_given := -2  -- Slope of 2x + y - 1 = 0
  slope_AB = slope_given → m = -8 :=
by
  sorry

end parallel_line_m_value_l396_39625


namespace greatest_common_multiple_9_15_under_110_l396_39611

theorem greatest_common_multiple_9_15_under_110 : ∃ n : ℕ, 
  (∀ m : ℕ, m < 110 ∧ 9 ∣ m ∧ 15 ∣ m → m ≤ n) ∧
  n < 110 ∧ 9 ∣ n ∧ 15 ∣ n ∧
  n = 90 := by
  sorry

end greatest_common_multiple_9_15_under_110_l396_39611


namespace third_term_value_l396_39651

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the specific sequence with given conditions
def specific_sequence (a : ℕ → ℝ) : Prop :=
  geometric_sequence a ∧ a 1 = -2 ∧ a 5 = -8

-- Theorem statement
theorem third_term_value (a : ℕ → ℝ) (h : specific_sequence a) : a 3 = -4 := by
  sorry

end third_term_value_l396_39651


namespace equation_solutions_l396_39658

theorem equation_solutions : 
  let f (x : ℝ) := 1 / (x^2 + 10*x - 8) + 1 / (x^2 + 3*x - 8) + 1 / (x^2 - 12*x - 8)
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -19 ∨ x = (5 + Real.sqrt 57) / 2 ∨ x = (5 - Real.sqrt 57) / 2 :=
by sorry

end equation_solutions_l396_39658


namespace price_reduction_achieves_profit_l396_39607

/-- Represents the daily sales and profit scenario of a store --/
structure StoreSales where
  initial_sales : ℕ := 20
  initial_profit_per_item : ℝ := 40
  sales_increase_rate : ℝ := 2
  min_profit_per_item : ℝ := 25

/-- Calculates the daily sales after a price reduction --/
def daily_sales (s : StoreSales) (price_reduction : ℝ) : ℝ :=
  s.initial_sales + s.sales_increase_rate * price_reduction

/-- Calculates the profit per item after a price reduction --/
def profit_per_item (s : StoreSales) (price_reduction : ℝ) : ℝ :=
  s.initial_profit_per_item - price_reduction

/-- Calculates the total daily profit after a price reduction --/
def daily_profit (s : StoreSales) (price_reduction : ℝ) : ℝ :=
  (daily_sales s price_reduction) * (profit_per_item s price_reduction)

/-- Theorem stating that a price reduction of 10 achieves the desired profit --/
theorem price_reduction_achieves_profit (s : StoreSales) :
  daily_profit s 10 = 1200 ∧ profit_per_item s 10 ≥ s.min_profit_per_item := by
  sorry


end price_reduction_achieves_profit_l396_39607


namespace floor_negative_seven_fourths_l396_39624

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_negative_seven_fourths_l396_39624


namespace complex_equation_solution_l396_39637

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l396_39637


namespace paco_salty_cookies_left_l396_39640

/-- The number of salty cookies Paco has left after sharing with friends -/
def salty_cookies_left (initial_salty : ℕ) (shared_ana : ℕ) (shared_juan : ℕ) : ℕ :=
  initial_salty - (shared_ana + shared_juan)

/-- Theorem stating that Paco has 12 salty cookies left -/
theorem paco_salty_cookies_left :
  salty_cookies_left 26 11 3 = 12 := by sorry

end paco_salty_cookies_left_l396_39640


namespace probability_prime_and_power_of_2_l396_39609

/-- The set of prime numbers between 1 and 8 (inclusive) -/
def primes_1_to_8 : Finset Nat := {2, 3, 5, 7}

/-- The set of powers of 2 between 1 and 8 (inclusive) -/
def powers_of_2_1_to_8 : Finset Nat := {1, 2, 4, 8}

/-- The number of sides on each die -/
def die_sides : Nat := 8

theorem probability_prime_and_power_of_2 :
  (Finset.card primes_1_to_8 * Finset.card powers_of_2_1_to_8) / (die_sides * die_sides) = 1 / 4 := by
  sorry

end probability_prime_and_power_of_2_l396_39609


namespace shaded_area_of_circles_l396_39614

theorem shaded_area_of_circles (r : ℝ) (h1 : r > 0) (h2 : π * r^2 = 81 * π) : 
  (π * r^2) / 2 + (π * (r/2)^2) / 2 = 50.625 * π := by sorry

end shaded_area_of_circles_l396_39614


namespace karen_cake_days_l396_39653

/-- The number of school days in a week -/
def school_days : ℕ := 5

/-- The number of days Karen packs ham sandwiches -/
def ham_days : ℕ := 3

/-- The probability of packing a ham sandwich and cake on the same day -/
def ham_cake_prob : ℚ := 12 / 100

/-- The number of days Karen packs a piece of cake -/
def cake_days : ℕ := sorry

theorem karen_cake_days :
  (ham_days : ℚ) / school_days * cake_days / school_days = ham_cake_prob →
  cake_days = 1 := by sorry

end karen_cake_days_l396_39653


namespace gcd_divides_n_plus_two_l396_39679

theorem gcd_divides_n_plus_two (a b n : ℤ) 
  (h_coprime : Nat.Coprime a.natAbs b.natAbs) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  ∃ k : ℤ, k * Int.gcd (a^2 + b^2 - n*a*b) (a + b) = n + 2 := by
  sorry

end gcd_divides_n_plus_two_l396_39679


namespace terrell_lifting_equivalence_l396_39698

/-- The number of times Terrell lifts the 40-pound weight -/
def original_lifts : ℕ := 12

/-- The weight of the original weight in pounds -/
def original_weight : ℕ := 40

/-- The weight of the new weight in pounds -/
def new_weight : ℕ := 30

/-- The total weight lifted with the original weight -/
def total_weight : ℕ := original_weight * original_lifts

/-- The number of times Terrell must lift the new weight to achieve the same total weight -/
def new_lifts : ℕ := total_weight / new_weight

theorem terrell_lifting_equivalence :
  new_lifts = 16 :=
sorry

end terrell_lifting_equivalence_l396_39698


namespace tan_plus_3sin_30_deg_l396_39662

theorem tan_plus_3sin_30_deg :
  let sin_30 : ℝ := 1/2
  let cos_30 : ℝ := Real.sqrt 3 / 2
  (sin_30 / cos_30) + 3 * sin_30 = 2 + 3 * Real.sqrt 3 := by sorry

end tan_plus_3sin_30_deg_l396_39662


namespace subset_implies_membership_condition_l396_39603

theorem subset_implies_membership_condition (A B : Set α) (h : A ⊆ B) :
  ∀ x, x ∈ A → x ∈ B := by
  sorry

end subset_implies_membership_condition_l396_39603


namespace garage_sale_pricing_l396_39602

theorem garage_sale_pricing (total_items : ℕ) (n : ℕ) 
  (h1 : total_items = 42)
  (h2 : n < total_items)
  (h3 : n = total_items - 24) : n = 19 := by
  sorry

end garage_sale_pricing_l396_39602


namespace average_difference_l396_39630

def average (a b c : ℕ) : ℚ :=
  (a + b + c : ℚ) / 3

theorem average_difference : average 20 40 60 - average 10 70 28 = 4 := by
  sorry

end average_difference_l396_39630


namespace inequality_implies_bound_l396_39654

theorem inequality_implies_bound (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, Real.exp x - x > a * x) → a < Real.exp 1 - 1 := by
  sorry

end inequality_implies_bound_l396_39654


namespace line_and_circle_equations_l396_39686

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4*x - 3*y - 5 = 0
def line3 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the intersection point of line1 and line2
def intersection_point : ℝ × ℝ := (2, 1)

-- Define line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Main theorem
theorem line_and_circle_equations :
  ∃ (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop),
    -- l passes through the intersection of line1 and line2
    (l (intersection_point.1) (intersection_point.2)) ∧
    -- l is perpendicular to line3
    (∀ x y, l x y → line3 x y → (x + 1 = y)) ∧
    -- C passes through (1,0)
    (C 1 0) ∧
    -- Center of C is on positive x-axis
    (∃ a > 0, ∀ x y, C x y ↔ (x - a)^2 + y^2 = a^2) ∧
    -- Chord intercepted by l on C has length 2√2
    (∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) ∧
    -- l is the line y = x - 1
    (∀ x y, l x y ↔ line_l x y) ∧
    -- C is the circle (x-3)^2 + y^2 = 4
    (∀ x y, C x y ↔ circle_C x y) :=
by sorry

end line_and_circle_equations_l396_39686


namespace characterization_of_n_l396_39627

/-- A bijection from {1, ..., n} to itself -/
def Bijection (n : ℕ) := { f : Fin n → Fin n // Function.Bijective f }

/-- The main theorem -/
theorem characterization_of_n (m : ℕ) (h_m : Even m) (h_m_pos : 0 < m) :
  ∀ n : ℕ, (∃ f : Bijection n,
    ∀ x y : Fin n, (m * x.val - y.val) % n = 0 →
      (n + 1) ∣ (f.val x).val^m - (f.val y).val) ↔
  Nat.Prime (n + 1) :=
sorry

end characterization_of_n_l396_39627


namespace integer_sum_l396_39670

theorem integer_sum (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 288) : 
  x.val + y.val = 35 := by
sorry

end integer_sum_l396_39670


namespace team_selection_count_l396_39663

def total_athletes : ℕ := 10
def veteran_players : ℕ := 2
def new_players : ℕ := 8
def team_size : ℕ := 3
def excluded_new_player : ℕ := 1

theorem team_selection_count :
  (Nat.choose veteran_players 1 * Nat.choose (new_players - excluded_new_player) 2) +
  (Nat.choose (new_players - excluded_new_player) team_size) = 77 := by
  sorry

end team_selection_count_l396_39663


namespace triangle_theorem_triangle_max_sum_l396_39666

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

theorem triangle_theorem (t : Triangle) (h : 2 * t.c - t.a = 2 * t.b * Real.cos t.A) :
  t.B = π / 3 := by sorry

theorem triangle_max_sum (t : Triangle) 
  (h1 : 2 * t.c - t.a = 2 * t.b * Real.cos t.A) 
  (h2 : t.b = 2 * Real.sqrt 3) :
  (∀ (s : Triangle), s.a + s.c ≤ 4 * Real.sqrt 3) ∧ 
  (∃ (s : Triangle), s.a + s.c = 4 * Real.sqrt 3) := by sorry

end triangle_theorem_triangle_max_sum_l396_39666


namespace inscribed_circle_area_ratio_l396_39692

theorem inscribed_circle_area_ratio (α : Real) (h : 0 < α ∧ α < π / 2) :
  let rhombus_area (a : Real) := a^2 * Real.sin α
  let circle_area (r : Real) := π * r^2
  let inscribed_circle_radius (a : Real) := (a * Real.sin α) / 2
  ∀ a > 0, circle_area (inscribed_circle_radius a) / rhombus_area a = (π / 4) * Real.sin α :=
sorry

end inscribed_circle_area_ratio_l396_39692


namespace parallelogram_area_l396_39604

/-- The area of a parallelogram with longer diagonal 5 and heights 2 and 3 -/
theorem parallelogram_area (d : ℝ) (h₁ h₂ : ℝ) (hd : d = 5) (hh₁ : h₁ = 2) (hh₂ : h₂ = 3) :
  (h₁ * h₂) / (((3 * Real.sqrt 21 + 8) / 25) : ℝ) = 150 / (3 * Real.sqrt 21 + 8) := by
sorry

end parallelogram_area_l396_39604


namespace equation_solution_l396_39615

theorem equation_solution :
  ∀ y : ℝ, (((36 * y + (36 * y + 55) ^ (1/3)) ^ (1/4)) = 11) → y = 7315/18 :=
by sorry

end equation_solution_l396_39615


namespace max_regions_50_lines_20_parallel_l396_39665

/-- The maximum number of regions created by n lines in a plane -/
def max_regions (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- The number of additional regions created by m parallel lines intersecting n non-parallel lines -/
def parallel_regions (m n : ℕ) : ℕ :=
  m * (n + 1)

/-- The maximum number of regions created by n lines in a plane, where p of them are parallel -/
def max_regions_with_parallel (n p : ℕ) : ℕ :=
  max_regions (n - p) + parallel_regions p (n - p)

theorem max_regions_50_lines_20_parallel :
  max_regions_with_parallel 50 20 = 1086 := by
  sorry

end max_regions_50_lines_20_parallel_l396_39665


namespace exists_prime_triplet_l396_39646

/-- A structure representing a prime triplet (a, b, c) -/
structure PrimeTriplet where
  a : Nat
  b : Nat
  c : Nat
  h_prime_a : Nat.Prime a
  h_prime_b : Nat.Prime b
  h_prime_c : Nat.Prime c
  h_order : a < b ∧ b < c ∧ c < 100
  h_geometric : (b + 1)^2 = (a + 1) * (c + 1)

/-- Theorem stating the existence of prime triplets satisfying the given conditions -/
theorem exists_prime_triplet : ∃ t : PrimeTriplet, True := by
  sorry

end exists_prime_triplet_l396_39646


namespace arithmetic_expression_equality_l396_39685

theorem arithmetic_expression_equality : (4 * 12) - (4 + 12) = 32 := by
  sorry

end arithmetic_expression_equality_l396_39685


namespace two_numbers_difference_l396_39691

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) : 
  |x - y| = 2 := by
sorry

end two_numbers_difference_l396_39691


namespace cubic_root_sum_cubes_l396_39689

theorem cubic_root_sum_cubes (a b c r s t : ℝ) : 
  r^3 - a*r^2 + b*r - c = 0 →
  s^3 - a*s^2 + b*s - c = 0 →
  t^3 - a*t^2 + b*t - c = 0 →
  r^3 + s^3 + t^3 = a^3 - 3*a*b + 3*c :=
by
  sorry

end cubic_root_sum_cubes_l396_39689


namespace alice_ball_drawing_l396_39612

/-- The number of balls in the bin -/
def n : ℕ := 20

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k balls from n balls with replacement -/
def num_possible_lists (n k : ℕ) : ℕ := n ^ k

theorem alice_ball_drawing :
  num_possible_lists n k = 160000 := by
  sorry

end alice_ball_drawing_l396_39612


namespace soap_cost_theorem_l396_39631

-- Define the given conditions
def months_per_bar : ℕ := 2
def cost_per_bar : ℚ := 8
def discount_rate : ℚ := 0.1
def discount_threshold : ℕ := 6
def months_in_year : ℕ := 12

-- Define the function to calculate the cost of soap for a year
def soap_cost_for_year : ℚ :=
  let bars_needed := months_in_year / months_per_bar
  let total_cost := bars_needed * cost_per_bar
  let discount := if bars_needed ≥ discount_threshold then discount_rate * total_cost else 0
  total_cost - discount

-- Theorem statement
theorem soap_cost_theorem : soap_cost_for_year = 43.2 := by
  sorry


end soap_cost_theorem_l396_39631


namespace circle_equation_l396_39636

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line --/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Checks if a circle is tangent to a line at a given point --/
def Circle.tangentTo (c : Circle) (l : Line) (p : ℝ × ℝ) : Prop :=
  l.contains p ∧
  (c.center.1 - p.1) ^ 2 + (c.center.2 - p.2) ^ 2 = c.radius ^ 2 ∧
  (c.center.1 - p.1) * l.a + (c.center.2 - p.2) * l.b = 0

/-- The main theorem --/
theorem circle_equation (c : Circle) :
  (c.center.2 = -4 * c.center.1) →  -- Center lies on y = -4x
  (c.tangentTo (Line.mk 1 1 (-1)) (3, -2)) →  -- Tangent to x + y - 1 = 0 at (3, -2)
  (∀ x y : ℝ, (x - 1)^2 + (y + 4)^2 = 8 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end circle_equation_l396_39636


namespace cos_pi_4_plus_alpha_l396_39622

theorem cos_pi_4_plus_alpha (α : ℝ) (h : Real.sin (π / 4 - α) = -2 / 5) :
  Real.cos (π / 4 + α) = -2 / 5 := by
  sorry

end cos_pi_4_plus_alpha_l396_39622


namespace not_necessarily_right_triangle_l396_39649

theorem not_necessarily_right_triangle (A B C : ℝ) : 
  A + B + C = 180 → A = B → A = 2 * C → ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
by sorry

end not_necessarily_right_triangle_l396_39649


namespace at_least_one_women_pair_probability_l396_39641

/-- The number of young men in the group -/
def num_men : ℕ := 5

/-- The number of young women in the group -/
def num_women : ℕ := 5

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to form pairs -/
def total_pairings : ℕ := (total_people.factorial) / ((2^num_pairs) * num_pairs.factorial)

/-- The number of ways to form pairs with at least one pair of two women -/
def favorable_pairings : ℕ := total_pairings - num_pairs.factorial

/-- The probability of at least one pair consisting of two young women -/
def probability : ℚ := favorable_pairings / total_pairings

theorem at_least_one_women_pair_probability :
  probability = 825 / 945 :=
sorry

end at_least_one_women_pair_probability_l396_39641


namespace renatas_transactions_l396_39600

/-- Represents Renata's financial transactions and final balance --/
theorem renatas_transactions (initial_amount casino_and_water_cost lottery_win final_balance : ℚ) :
  initial_amount = 10 →
  lottery_win = 65 →
  final_balance = 94 →
  casino_and_water_cost = 67 →
  initial_amount - 4 + 90 - casino_and_water_cost + lottery_win = final_balance :=
by sorry

end renatas_transactions_l396_39600


namespace acid_dilution_l396_39661

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution 
    results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.4 →
  added_water = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry

#check acid_dilution

end acid_dilution_l396_39661


namespace problem_statement_l396_39697

theorem problem_statement (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc1 : 0 ≤ c) (hc2 : c < -b) :
  (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end problem_statement_l396_39697


namespace dinitrogen_monoxide_molecular_weight_l396_39616

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound consisting of two nitrogen atoms and one oxygen atom -/
def dinitrogen_monoxide_weight : ℝ := 2 * nitrogen_weight + oxygen_weight

/-- Theorem stating that the molecular weight of Dinitrogen monoxide is 44.02 amu -/
theorem dinitrogen_monoxide_molecular_weight :
  dinitrogen_monoxide_weight = 44.02 := by sorry

end dinitrogen_monoxide_molecular_weight_l396_39616


namespace papaya_tree_first_year_growth_l396_39650

/-- The growth pattern of a papaya tree over 5 years -/
def PapayaTreeGrowth (first_year_growth : ℝ) : ℝ :=
  let second_year := 1.5 * first_year_growth
  let third_year := 1.5 * second_year
  let fourth_year := 2 * third_year
  let fifth_year := 0.5 * fourth_year
  first_year_growth + second_year + third_year + fourth_year + fifth_year

/-- Theorem stating that if a papaya tree grows to 23 feet in 5 years following the given pattern, 
    it must have grown 2 feet in the first year -/
theorem papaya_tree_first_year_growth :
  ∃ (x : ℝ), PapayaTreeGrowth x = 23 → x = 2 :=
sorry

end papaya_tree_first_year_growth_l396_39650


namespace inequality_solution_l396_39655

theorem inequality_solution (x : ℝ) : 
  (1 / 6 : ℝ) + |x - 1 / 3| < 1 / 2 ↔ 0 < x ∧ x < 2 / 3 :=
by sorry

end inequality_solution_l396_39655


namespace seashells_given_proof_l396_39632

def seashells_given_to_jessica (initial_seashells remaining_seashells : ℝ) : ℝ :=
  initial_seashells - remaining_seashells

theorem seashells_given_proof (initial_seashells remaining_seashells : ℝ) 
  (h1 : initial_seashells = 62.5) 
  (h2 : remaining_seashells = 30.75) : 
  seashells_given_to_jessica initial_seashells remaining_seashells = 31.75 := by
  sorry

end seashells_given_proof_l396_39632


namespace odd_function_negative_domain_l396_39608

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_pos : ∀ x > 0, f x = 2 * x - 3) : 
  ∀ x < 0, f x = 2 * x + 3 := by
sorry

end odd_function_negative_domain_l396_39608


namespace toad_ratio_proof_l396_39605

/-- Proves that the ratio of Sarah's toads to Jim's toads is 2 --/
theorem toad_ratio_proof (tim_toads jim_toads sarah_toads : ℕ) : 
  tim_toads = 30 →
  jim_toads = tim_toads + 20 →
  sarah_toads = 100 →
  sarah_toads / jim_toads = 2 := by
sorry

end toad_ratio_proof_l396_39605


namespace find_k_l396_39694

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 3 else n / 2

theorem find_k (k : ℤ) : 
  k % 2 = 1 → f (f (f k)) = 27 → k = 105 := by
  sorry

end find_k_l396_39694


namespace intersection_distance_l396_39633

/-- The circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

/-- The line l passing through (-4, 0) with slope angle π/4 -/
def l (x y : ℝ) : Prop := y = x + 4

/-- The intersection points of l and C₁ -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | C₁ p.1 p.2 ∧ l p.1 p.2}

/-- The theorem stating that the distance between the intersection points is √2 -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2 := by
  sorry

end intersection_distance_l396_39633


namespace unique_three_digit_number_l396_39684

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 35 = 0 ∧          -- multiple of 35
  (n / 100 + (n / 10) % 10 + n % 10 = 15) ∧  -- sum of digits is 15
  n = 735 := by
sorry

end unique_three_digit_number_l396_39684


namespace circle_center_l396_39681

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The original equation of the circle -/
def OriginalEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 4*y = 4

theorem circle_center :
  ∃ (r : ℝ), ∀ (x y : ℝ), OriginalEquation x y ↔ CircleEquation (-4) 2 r x y :=
by sorry

end circle_center_l396_39681


namespace fish_pond_flowers_l396_39644

/-- Calculates the number of flowers planted around a circular pond -/
def flowers_around_pond (perimeter : ℕ) (tree_spacing : ℕ) (flowers_between : ℕ) : ℕ :=
  (perimeter / tree_spacing) * flowers_between

/-- Theorem: The number of flowers planted around the fish pond is 39 -/
theorem fish_pond_flowers :
  flowers_around_pond 52 4 3 = 39 := by
  sorry

end fish_pond_flowers_l396_39644


namespace equal_even_odd_probability_l396_39623

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The probability of a single die showing an even number -/
def prob_even : ℚ := 1/2

/-- The probability of a single die showing an odd number -/
def prob_odd : ℚ := 1/2

/-- The number of dice that need to show even (and odd) for the event to occur -/
def target_even : ℕ := num_dice / 2

-- The theorem statement
theorem equal_even_odd_probability : 
  (Nat.choose num_dice target_even : ℚ) * prob_even ^ num_dice = 35/128 := by
  sorry

end equal_even_odd_probability_l396_39623


namespace abc_inequality_l396_39635

theorem abc_inequality (a b c t : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^t + b^t + c^t) ≥ 
  a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ∧
  (a * b * c * (a^t + b^t + c^t) = 
   a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ↔ 
   a = b ∧ b = c) :=
by sorry

end abc_inequality_l396_39635


namespace equation_solutions_l396_39621

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 6*x = -1 ↔ x = 3 - 2*Real.sqrt 2 ∨ x = 3 + 2*Real.sqrt 2) ∧
  (∀ x : ℝ, x*(2*x - 1) = 2*(2*x - 1) ↔ x = 1/2 ∨ x = 2) := by
  sorry

end equation_solutions_l396_39621


namespace prob_at_least_one_multiple_of_three_l396_39678

/-- The number of integers from 1 to 50 inclusive -/
def total_numbers : ℕ := 50

/-- The number of multiples of 3 from 1 to 50 inclusive -/
def multiples_of_three : ℕ := 16

/-- The probability of choosing a number that is not a multiple of 3 -/
def prob_not_multiple : ℚ := (total_numbers - multiples_of_three) / total_numbers

/-- The probability of choosing at least one multiple of 3 in two selections -/
def prob_at_least_one_multiple : ℚ := 1 - prob_not_multiple ^ 2

theorem prob_at_least_one_multiple_of_three :
  prob_at_least_one_multiple = 336 / 625 := by sorry

end prob_at_least_one_multiple_of_three_l396_39678


namespace arithmetic_geometric_ratio_l396_39674

/-- Given an arithmetic sequence with non-zero common difference,
    if a_5, a_9, and a_15 form a geometric sequence,
    then a_15 / a_9 = 3/2 -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_d_nonzero : d ≠ 0)
  (h_geom : (a 9) ^ 2 = (a 5) * (a 15)) :
  a 15 / a 9 = 3 / 2 := by
  sorry

end arithmetic_geometric_ratio_l396_39674


namespace exactly_one_false_proposition_l396_39664

theorem exactly_one_false_proposition :
  let prop1 := (∀ x : ℝ, (x ^ 2 - 3 * x + 2 ≠ 0) → (x ≠ 1)) ↔ (∀ x : ℝ, (x ≠ 1) → (x ^ 2 - 3 * x + 2 ≠ 0))
  let prop2 := (∀ x : ℝ, x > 2 → x ^ 2 - 3 * x + 2 > 0) ∧ (∃ x : ℝ, x ≤ 2 ∧ x ^ 2 - 3 * x + 2 > 0)
  let prop3 := ∀ p q : Prop, (p ∧ q → False) → (p → False) ∧ (q → False)
  let prop4 := (∃ x : ℝ, x ^ 2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x ^ 2 + x + 1 ≥ 0)
  ∃! i : Fin 4, ¬(match i with
    | 0 => prop1
    | 1 => prop2
    | 2 => prop3
    | 3 => prop4) :=
by
  sorry

end exactly_one_false_proposition_l396_39664


namespace library_book_count_l396_39652

/-- Represents the library with its bookshelves and books. -/
structure Library where
  num_bookshelves : Nat
  floors_per_bookshelf : Nat
  left_position : Nat
  right_position : Nat

/-- Calculates the total number of books in the library. -/
def total_books (lib : Library) : Nat :=
  let books_per_floor := lib.left_position + lib.right_position - 1
  let books_per_bookshelf := books_per_floor * lib.floors_per_bookshelf
  books_per_bookshelf * lib.num_bookshelves

/-- Theorem stating the total number of books in the library. -/
theorem library_book_count :
  ∀ (lib : Library),
    lib.num_bookshelves = 28 →
    lib.floors_per_bookshelf = 6 →
    lib.left_position = 9 →
    lib.right_position = 11 →
    total_books lib = 3192 := by
  sorry

#eval total_books ⟨28, 6, 9, 11⟩

end library_book_count_l396_39652


namespace max_l_pieces_theorem_max_l_pieces_5x10_max_l_pieces_5x9_l396_39672

/-- Represents an L-shaped piece consisting of 3 cells -/
structure LPiece where
  size : Nat
  size_eq : size = 3

/-- Represents a rectangular grid -/
structure Grid where
  rows : Nat
  cols : Nat

/-- Calculates the maximum number of L-shaped pieces that can be cut from a grid -/
def maxLPieces (g : Grid) (l : LPiece) : Nat :=
  (g.rows * g.cols) / l.size

/-- Theorem: The maximum number of L-shaped pieces in a grid is the floor of total cells divided by piece size -/
theorem max_l_pieces_theorem (g : Grid) (l : LPiece) :
  maxLPieces g l = ⌊(g.rows * g.cols : ℚ) / l.size⌋ :=
sorry

/-- Corollary: For a 5x10 grid, the maximum number of L-shaped pieces is 16 -/
theorem max_l_pieces_5x10 :
  maxLPieces { rows := 5, cols := 10 } { size := 3, size_eq := rfl } = 16 :=
sorry

/-- Corollary: For a 5x9 grid, the maximum number of L-shaped pieces is 15 -/
theorem max_l_pieces_5x9 :
  maxLPieces { rows := 5, cols := 9 } { size := 3, size_eq := rfl } = 15 :=
sorry

end max_l_pieces_theorem_max_l_pieces_5x10_max_l_pieces_5x9_l396_39672


namespace mantou_distribution_theorem_l396_39613

/-- Represents the distribution of mantou among monks -/
structure MantouDistribution where
  bigMonks : ℕ
  smallMonks : ℕ
  totalMonks : ℕ
  totalMantou : ℕ

/-- The mantou distribution satisfies the problem conditions -/
def isValidDistribution (d : MantouDistribution) : Prop :=
  d.bigMonks + d.smallMonks = d.totalMonks ∧
  d.totalMonks = 100 ∧
  d.totalMantou = 100 ∧
  3 * d.bigMonks + (1/3) * d.smallMonks = d.totalMantou

/-- The system of equations correctly represents the mantou distribution -/
theorem mantou_distribution_theorem (d : MantouDistribution) :
  isValidDistribution d ↔
  d.bigMonks + d.smallMonks = 100 ∧
  3 * d.bigMonks + (1/3) * d.smallMonks = 100 :=
sorry

end mantou_distribution_theorem_l396_39613


namespace stamp_reorganization_l396_39647

/-- Represents the stamp reorganization problem --/
theorem stamp_reorganization (
  initial_books : Nat)
  (pages_per_book : Nat)
  (initial_stamps_per_page : Nat)
  (new_stamps_per_page : Nat)
  (filled_books : Nat)
  (filled_pages_in_last_book : Nat)
  (h1 : initial_books = 10)
  (h2 : pages_per_book = 36)
  (h3 : initial_stamps_per_page = 5)
  (h4 : new_stamps_per_page = 8)
  (h5 : filled_books = 7)
  (h6 : filled_pages_in_last_book = 28) :
  (initial_books * pages_per_book * initial_stamps_per_page) -
  (filled_books * pages_per_book + filled_pages_in_last_book) * new_stamps_per_page = 8 := by
  sorry

#check stamp_reorganization

end stamp_reorganization_l396_39647


namespace eating_contest_l396_39626

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight pizza_weight sandwich_weight : ℕ)
  (jacob_pies noah_burgers jacob_pizzas jacob_sandwiches mason_hotdogs mason_sandwiches : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  pizza_weight = 15 →
  sandwich_weight = 3 →
  jacob_pies = noah_burgers - 3 →
  jacob_pizzas = jacob_sandwiches / 2 →
  mason_hotdogs = 3 * jacob_pies →
  mason_hotdogs = (3 : ℚ) / 2 * mason_sandwiches →
  noah_burgers = 8 →
  mason_hotdogs * hot_dog_weight = 30 := by sorry

end eating_contest_l396_39626


namespace other_divisor_is_57_l396_39656

theorem other_divisor_is_57 : 
  ∃ (x : ℕ), x ≠ 38 ∧ 
  114 % x = 0 ∧ 
  115 % x = 1 ∧
  115 % 38 = 1 ∧
  (∀ y : ℕ, y > x → 114 % y = 0 → y = 38 ∨ y = 114) :=
by sorry

end other_divisor_is_57_l396_39656


namespace two_children_gender_combinations_l396_39617

-- Define the possible genders
inductive Gender
| Male
| Female

-- Define a type for a pair of children's genders
def ChildrenGenders := (Gender × Gender)

-- Define the set of all possible gender combinations
def allGenderCombinations : Set ChildrenGenders :=
  {(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)}

-- Theorem statement
theorem two_children_gender_combinations :
  ∀ (family : ChildrenGenders), family ∈ allGenderCombinations :=
by sorry

end two_children_gender_combinations_l396_39617


namespace F_is_even_l396_39680

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property that f(-x) + f(x) = 0 for all x
axiom f_property : ∀ x, f (-x) + f x = 0

-- Define F(x) = |f(x)|
def F (x : ℝ) : ℝ := |f x|

-- Theorem statement
theorem F_is_even : ∀ x, F x = F (-x) := by sorry

end F_is_even_l396_39680


namespace sum_of_solutions_is_zero_l396_39619

theorem sum_of_solutions_is_zero (y : ℝ) (x₁ x₂ : ℝ) : 
  y = 10 → 
  x₁^2 + y^2 = 200 → 
  x₂^2 + y^2 = 200 → 
  x₁ + x₂ = 0 := by
sorry

end sum_of_solutions_is_zero_l396_39619


namespace max_sin_theta_is_one_l396_39668

theorem max_sin_theta_is_one (a b : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) :
  (∃ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ 0 ∧ a * Real.cos θ - b * Real.sin θ ≥ 0) →
  (∃ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ 0 ∧ a * Real.cos θ - b * Real.sin θ ≥ 0 ∧
    ∀ φ : ℝ, (a * Real.sin φ + b * Real.cos φ ≥ 0 ∧ a * Real.cos φ - b * Real.sin φ ≥ 0) →
      Real.sin θ ≥ Real.sin φ) →
  (∃ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ 0 ∧ a * Real.cos θ - b * Real.sin θ ≥ 0 ∧ Real.sin θ = 1) :=
sorry

end max_sin_theta_is_one_l396_39668


namespace inverse_proportion_constant_difference_l396_39695

/-- Given two inverse proportion functions and points satisfying certain conditions, 
    prove that the difference of their constants is 4. -/
theorem inverse_proportion_constant_difference 
  (k₁ k₂ : ℝ) 
  (f₁ : ℝ → ℝ) 
  (f₂ : ℝ → ℝ) 
  (a b : ℝ) 
  (h₁ : ∀ x, f₁ x = k₁ / x) 
  (h₂ : ∀ x, f₂ x = k₂ / x) 
  (h₃ : |f₁ a - f₂ a| = 2) 
  (h₄ : |f₂ b - f₁ b| = 3) 
  (h₅ : |b - a| = 10/3) : 
  k₂ - k₁ = 4 := by
sorry

end inverse_proportion_constant_difference_l396_39695


namespace bud_is_eight_years_old_l396_39634

def buds_age (uncle_age : ℕ) : ℕ :=
  uncle_age / 3

theorem bud_is_eight_years_old (uncle_age : ℕ) (h : uncle_age = 24) :
  buds_age uncle_age = 8 := by
  sorry

end bud_is_eight_years_old_l396_39634


namespace six_lines_intersections_l396_39676

/-- The maximum number of intersection points between n straight lines -/
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The maximum number of intersection points between 6 straight lines is 15 -/
theorem six_lines_intersections :
  max_intersections 6 = 15 := by
  sorry

end six_lines_intersections_l396_39676


namespace work_completion_fraction_l396_39667

theorem work_completion_fraction (x_days y_days z_days total_days : ℕ) 
  (hx : x_days = 14) 
  (hy : y_days = 20) 
  (hz : z_days = 25) 
  (ht : total_days = 5) : 
  (total_days : ℚ) * ((1 : ℚ) / x_days + (1 : ℚ) / y_days + (1 : ℚ) / z_days) = 113 / 140 := by
  sorry

end work_completion_fraction_l396_39667


namespace inequality_solution_l396_39690

theorem inequality_solution (x : ℝ) : 
  (6*x^2 + 18*x - 64) / ((3*x - 2)*(x + 5)) < 2 ↔ -5 < x ∧ x < 2/3 := by sorry

end inequality_solution_l396_39690


namespace complementary_angles_ratio_l396_39659

/-- Two complementary angles in a ratio of 5:4 have the larger angle measuring 50 degrees -/
theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of angles is 5:4
  max a b = 50 :=  -- larger angle measures 50 degrees
by sorry

end complementary_angles_ratio_l396_39659
