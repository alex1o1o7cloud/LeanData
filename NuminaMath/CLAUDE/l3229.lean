import Mathlib

namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l3229_322996

/-- The function f(x) = ax - 3 + 3 always passes through the point (3, 4) for any real number a. -/
theorem fixed_point_of_exponential_translation (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x - 3 + 3
  f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l3229_322996


namespace NUMINAMATH_CALUDE_hugo_climb_count_l3229_322912

def hugo_mountain_elevation : ℕ := 10000
def boris_mountain_elevation : ℕ := hugo_mountain_elevation - 2500
def boris_climb_count : ℕ := 4

theorem hugo_climb_count : 
  ∃ (x : ℕ), x * hugo_mountain_elevation = boris_climb_count * boris_mountain_elevation ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_hugo_climb_count_l3229_322912


namespace NUMINAMATH_CALUDE_truck_travel_distance_l3229_322984

/-- Represents the distance a truck can travel -/
def distance_traveled (diesel_amount : ℚ) : ℚ :=
  150 * diesel_amount / 5

/-- The theorem states that the truck travels 210 miles on 7 gallons of diesel -/
theorem truck_travel_distance : distance_traveled 7 = 210 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l3229_322984


namespace NUMINAMATH_CALUDE_volumes_equal_l3229_322938

/-- The volume of a solid of revolution obtained by rotating a region about the y-axis -/
noncomputable def VolumeOfRevolution (region : Set (ℝ × ℝ)) : ℝ := sorry

/-- The region bounded by x² = 4y, x² = -4y, x = 4, and x = -4 -/
def Region1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4*p.2 ∨ p.1^2 = -4*p.2 ∨ p.1 = 4 ∨ p.1 = -4}

/-- The region consisting of points (x, y) that satisfy x²y ≤ 16, x² + (y - 2)² ≥ 4, and x² + (y + 2)² ≥ 4 -/
def Region2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 * p.2 ≤ 16 ∧ p.1^2 + (p.2 - 2)^2 ≥ 4 ∧ p.1^2 + (p.2 + 2)^2 ≥ 4}

/-- The theorem stating that the volumes of revolution of the two regions are equal -/
theorem volumes_equal : VolumeOfRevolution Region1 = VolumeOfRevolution Region2 := by
  sorry

end NUMINAMATH_CALUDE_volumes_equal_l3229_322938


namespace NUMINAMATH_CALUDE_barn_hoot_difference_l3229_322947

/-- The number of hoots one barnyard owl makes per minute -/
def hoots_per_owl : ℕ := 5

/-- The number of hoots heard per minute from the barn -/
def hoots_heard : ℕ := 20

/-- The number of owls we're comparing to -/
def num_owls : ℕ := 3

/-- The difference between the hoots heard and the hoots from a specific number of owls -/
def hoot_difference (heard : ℕ) (owls : ℕ) : ℤ :=
  heard - (owls * hoots_per_owl)

theorem barn_hoot_difference :
  hoot_difference hoots_heard num_owls = 5 := by
  sorry

end NUMINAMATH_CALUDE_barn_hoot_difference_l3229_322947


namespace NUMINAMATH_CALUDE_polly_tweets_l3229_322929

/-- Represents the tweet rate per minute for different states of Polly --/
structure TweetRates where
  happy : Nat
  hungry : Nat
  mirror : Nat

/-- Represents the duration in minutes for different activities --/
structure ActivityDurations where
  happy : Nat
  hungry : Nat
  mirror : Nat

/-- Calculates the total number of tweets based on rates and durations --/
def totalTweets (rates : TweetRates) (durations : ActivityDurations) : Nat :=
  rates.happy * durations.happy +
  rates.hungry * durations.hungry +
  rates.mirror * durations.mirror

/-- Theorem stating that Polly's total tweets equal 1340 --/
theorem polly_tweets (rates : TweetRates) (durations : ActivityDurations)
    (h1 : rates.happy = 18)
    (h2 : rates.hungry = 4)
    (h3 : rates.mirror = 45)
    (h4 : durations.happy = 20)
    (h5 : durations.hungry = 20)
    (h6 : durations.mirror = 20) :
    totalTweets rates durations = 1340 := by
  sorry


end NUMINAMATH_CALUDE_polly_tweets_l3229_322929


namespace NUMINAMATH_CALUDE_louisa_travel_time_l3229_322970

theorem louisa_travel_time 
  (distance_day1 : ℝ) 
  (distance_day2 : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance_day1 = 200)
  (h2 : distance_day2 = 360)
  (h3 : time_difference = 4)
  (h4 : ∃ v : ℝ, v > 0 ∧ distance_day1 / v + time_difference = distance_day2 / v) :
  ∃ total_time : ℝ, total_time = 14 ∧ total_time = distance_day1 / (distance_day2 - distance_day1) * time_difference + distance_day2 / (distance_day2 - distance_day1) * time_difference :=
by sorry

end NUMINAMATH_CALUDE_louisa_travel_time_l3229_322970


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l3229_322906

/-- The number of games required in a single-elimination tournament -/
def games_required (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with 21 teams, 20 games are required to declare a winner -/
theorem single_elimination_tournament_games :
  games_required 21 = 20 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l3229_322906


namespace NUMINAMATH_CALUDE_inverse_36_mod_101_l3229_322987

theorem inverse_36_mod_101 : ∃ x : ℤ, 36 * x ≡ 1 [ZMOD 101] :=
by
  use 87
  sorry

end NUMINAMATH_CALUDE_inverse_36_mod_101_l3229_322987


namespace NUMINAMATH_CALUDE_triangle_longest_side_l3229_322953

theorem triangle_longest_side (y : ℝ) : 
  10 + (y + 6) + (3 * y + 5) = 49 →
  max 10 (max (y + 6) (3 * y + 5)) = 26 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l3229_322953


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3229_322982

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 15*m + 56 ≤ 0 → n ≤ m) ∧ (n^2 - 15*n + 56 ≤ 0) ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3229_322982


namespace NUMINAMATH_CALUDE_equation_transformation_l3229_322967

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 - x^3 - 2*x^2 - x + 1 = 0 ↔ x^2 * (y^2 - y - 4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l3229_322967


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3229_322909

/-- Given that a, 6, and b form an arithmetic sequence in that order, prove that a + b = 12 -/
theorem arithmetic_sequence_sum (a b : ℝ) 
  (h : ∃ d : ℝ, a + d = 6 ∧ b = a + 2*d) : 
  a + b = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3229_322909


namespace NUMINAMATH_CALUDE_a_value_is_negative_one_l3229_322940

/-- The coefficient of x^2 in the expansion of (1+ax)(1+x)^5 -/
def coefficient_x_squared (a : ℝ) : ℝ :=
  (Nat.choose 5 2 : ℝ) + a * (Nat.choose 5 1 : ℝ)

/-- The theorem stating that a = -1 given the coefficient of x^2 is 5 -/
theorem a_value_is_negative_one :
  ∃ a : ℝ, coefficient_x_squared a = 5 ∧ a = -1 :=
sorry

end NUMINAMATH_CALUDE_a_value_is_negative_one_l3229_322940


namespace NUMINAMATH_CALUDE_prob_success_constant_l3229_322942

/-- Represents the probability of finding the correct key on the kth attempt
    given n total keys. -/
def prob_success (n : ℕ) (k : ℕ) : ℚ :=
  if 1 ≤ k ∧ k ≤ n then 1 / n else 0

/-- Theorem stating that the probability of success on any attempt
    is 1/n for any valid k. -/
theorem prob_success_constant (n : ℕ) (k : ℕ) (h1 : n > 0) (h2 : 1 ≤ k) (h3 : k ≤ n) :
  prob_success n k = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_prob_success_constant_l3229_322942


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l3229_322926

theorem simultaneous_equations_solution (m : ℝ) : 
  (m ≠ 1) ↔ (∃ x y : ℝ, y = m * x + 3 ∧ y = (2 * m - 1) * x + 4) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l3229_322926


namespace NUMINAMATH_CALUDE_average_problem_l3229_322914

theorem average_problem (x : ℝ) (h : (47 + x) / 2 = 53) : 
  x = 59 ∧ |x - 47| = 12 ∧ x + 47 = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l3229_322914


namespace NUMINAMATH_CALUDE_sum_in_base5_l3229_322920

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Represents a number in base 5 -/
structure Base5 where
  value : ℕ

theorem sum_in_base5 : 
  let a := Base5.mk 132
  let b := Base5.mk 214
  let c := Base5.mk 341
  let sum := base10ToBase5 (base5ToBase10 a.value + base5ToBase10 b.value + base5ToBase10 c.value)
  sum = 1242 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base5_l3229_322920


namespace NUMINAMATH_CALUDE_housing_boom_result_l3229_322997

/-- Represents the housing development in Lawrence County -/
structure LawrenceCountyHousing where
  initial_houses : ℕ
  developer_a_rate : ℕ
  developer_a_months : ℕ
  developer_b_rate : ℕ
  developer_b_months : ℕ
  developer_c_rate : ℕ
  developer_c_months : ℕ
  final_houses : ℕ

/-- Calculates the total number of houses built by developers -/
def total_houses_built (h : LawrenceCountyHousing) : ℕ :=
  h.developer_a_rate * h.developer_a_months +
  h.developer_b_rate * h.developer_b_months +
  h.developer_c_rate * h.developer_c_months

/-- Theorem stating that the total houses built by developers is 405 -/
theorem housing_boom_result (h : LawrenceCountyHousing)
  (h_initial : h.initial_houses = 1426)
  (h_dev_a : h.developer_a_rate = 25 ∧ h.developer_a_months = 6)
  (h_dev_b : h.developer_b_rate = 15 ∧ h.developer_b_months = 9)
  (h_dev_c : h.developer_c_rate = 30 ∧ h.developer_c_months = 4)
  (h_final : h.final_houses = 2000) :
  total_houses_built h = 405 := by
  sorry


end NUMINAMATH_CALUDE_housing_boom_result_l3229_322997


namespace NUMINAMATH_CALUDE_f_3_equals_neg_26_l3229_322931

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_3_equals_neg_26 (a b : ℝ) :
  f a b (-3) = 10 → f a b 3 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_neg_26_l3229_322931


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l3229_322928

theorem sqrt_x_plus_inverse_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l3229_322928


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3229_322974

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x > -a ∧ x > -b) ↔ x > -b) → a ≥ b := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3229_322974


namespace NUMINAMATH_CALUDE_softball_team_savings_l3229_322972

/-- Calculates the savings for a softball team buying uniforms with a group discount. -/
theorem softball_team_savings 
  (team_size : ℕ) 
  (regular_shirt_price regular_pants_price regular_socks_price : ℚ)
  (discount_shirt_price discount_pants_price discount_socks_price : ℚ)
  (h1 : team_size = 12)
  (h2 : regular_shirt_price = 7.5)
  (h3 : regular_pants_price = 15)
  (h4 : regular_socks_price = 4.5)
  (h5 : discount_shirt_price = 6.75)
  (h6 : discount_pants_price = 13.5)
  (h7 : discount_socks_price = 3.75) :
  (team_size : ℚ) * ((regular_shirt_price + regular_pants_price + regular_socks_price) - 
  (discount_shirt_price + discount_pants_price + discount_socks_price)) = 36 :=
by sorry

end NUMINAMATH_CALUDE_softball_team_savings_l3229_322972


namespace NUMINAMATH_CALUDE_caroline_lassis_l3229_322998

/-- Given that Caroline can make 15 lassis from 3 mangoes, prove that she can make 90 lassis from 18 mangoes. -/
theorem caroline_lassis (mangoes_small : ℕ) (lassis_small : ℕ) (mangoes_large : ℕ) :
  mangoes_small = 3 →
  lassis_small = 15 →
  mangoes_large = 18 →
  (lassis_small * mangoes_large) / mangoes_small = 90 :=
by sorry

end NUMINAMATH_CALUDE_caroline_lassis_l3229_322998


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3229_322958

/-- Parabola intersecting with a line -/
theorem parabola_line_intersection 
  (a : ℝ) 
  (h_a : a ≠ 0) 
  (b : ℝ) 
  (h_intersection : b = 2 * 1 - 3 ∧ b = a * 1^2) :
  (a = -1 ∧ b = -1) ∧ 
  ∃ x y : ℝ, x = -3 ∧ y = -9 ∧ y = a * x^2 ∧ y = 2 * x - 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3229_322958


namespace NUMINAMATH_CALUDE_range_of_a_l3229_322955

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (4 * x - 3) ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the condition that ¬p is a necessary but not sufficient condition for ¬q
def condition (a : ℝ) : Prop :=
  (∀ x : ℝ, q x a → p x) ∧ (∃ x : ℝ, p x ∧ ¬(q x a))

-- State the theorem
theorem range_of_a (a : ℝ) :
  condition a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3229_322955


namespace NUMINAMATH_CALUDE_max_profit_is_120_l3229_322922

def profit_A (x : ℕ) : ℚ := -x^2 + 21*x
def profit_B (x : ℕ) : ℚ := 2*x
def total_profit (x : ℕ) : ℚ := profit_A x + profit_B (15 - x)

theorem max_profit_is_120 :
  ∃ x : ℕ, x > 0 ∧ x ≤ 15 ∧
  total_profit x = 120 ∧
  ∀ y : ℕ, y > 0 → y ≤ 15 → total_profit y ≤ total_profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_120_l3229_322922


namespace NUMINAMATH_CALUDE_number_of_daughters_l3229_322911

theorem number_of_daughters (a : ℕ) : 
  (a.Prime) → 
  (64 + a^2 = 16*a + 1) → 
  a = 7 := by sorry

end NUMINAMATH_CALUDE_number_of_daughters_l3229_322911


namespace NUMINAMATH_CALUDE_vector_sum_zero_l3229_322915

variable {E : Type*} [NormedAddCommGroup E]

/-- Given vectors CE, AC, DE, and AD in a normed additive commutative group E,
    prove that CE + AC - DE - AD = 0 -/
theorem vector_sum_zero (CE AC DE AD : E) :
  CE + AC - DE - AD = (0 : E) := by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l3229_322915


namespace NUMINAMATH_CALUDE_arnold_protein_consumption_l3229_322901

-- Define the protein content of each food item
def collagen_protein_per_2_scoops : ℕ := 18
def protein_powder_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

-- Define Arnold's consumption
def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steak_portions : ℕ := 1

-- Theorem to prove
theorem arnold_protein_consumption :
  (collagen_scoops * collagen_protein_per_2_scoops / 2) +
  (protein_powder_scoops * protein_powder_per_scoop) +
  (steak_portions * steak_protein) = 86 := by
  sorry

end NUMINAMATH_CALUDE_arnold_protein_consumption_l3229_322901


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3229_322939

theorem min_value_quadratic :
  ∃ (z_min : ℝ), z_min = -44 ∧ ∀ (x : ℝ), x^2 + 16*x + 20 ≥ z_min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3229_322939


namespace NUMINAMATH_CALUDE_hiking_team_gloves_l3229_322966

/-- The minimum number of gloves required for a hiking team -/
def min_gloves (participants : ℕ) : ℕ := 2 * participants

/-- Theorem: For 82 participants, the minimum number of gloves required is 164 -/
theorem hiking_team_gloves : min_gloves 82 = 164 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_gloves_l3229_322966


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3229_322923

theorem simplify_and_evaluate : 
  let x : ℚ := 3/2
  (3 + x)^2 - (x + 5) * (x - 1) = 17 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3229_322923


namespace NUMINAMATH_CALUDE_sufficient_condition_for_perpendicular_l3229_322921

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the perpendicular relation between a line and a plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the perpendicular relation between two planes
def perp_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry

-- Theorem statement
theorem sufficient_condition_for_perpendicular 
  (α β : Plane) (m n : Line) :
  perp_line_plane n α → 
  perp_line_plane n β → 
  perp_line_plane m α → 
  perp_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_perpendicular_l3229_322921


namespace NUMINAMATH_CALUDE_business_loss_l3229_322936

/-- Proves that the total loss in a business partnership is 1600 given the specified conditions -/
theorem business_loss (ashok_capital pyarelal_capital pyarelal_loss : ℚ) : 
  ashok_capital = (1 : ℚ) / 9 * pyarelal_capital →
  pyarelal_loss = 1440 →
  ashok_capital / pyarelal_capital * pyarelal_loss + pyarelal_loss = 1600 :=
by sorry

end NUMINAMATH_CALUDE_business_loss_l3229_322936


namespace NUMINAMATH_CALUDE_guesthouse_fixed_rate_l3229_322919

/-- A guesthouse charging system with a fixed rate for the first night and an additional fee for subsequent nights. -/
structure Guesthouse where
  first_night : ℕ  -- Fixed rate for the first night
  subsequent : ℕ  -- Fee for each subsequent night

/-- The total cost for a stay at the guesthouse. -/
def total_cost (g : Guesthouse) (nights : ℕ) : ℕ :=
  g.first_night + g.subsequent * (nights - 1)

theorem guesthouse_fixed_rate :
  ∃ (g : Guesthouse),
    total_cost g 5 = 220 ∧
    total_cost g 8 = 370 ∧
    g.first_night = 20 := by
  sorry

end NUMINAMATH_CALUDE_guesthouse_fixed_rate_l3229_322919


namespace NUMINAMATH_CALUDE_sequence_properties_l3229_322930

def sequence_a (n : ℕ) : ℚ := 1/10 * (3/2)^(n-1) - 2/5 * (-1)^n

def partial_sum (n : ℕ) : ℚ := 3 * sequence_a n + (-1)^n

theorem sequence_properties :
  (sequence_a 1 = 1/2) ∧
  (sequence_a 2 = -1/4) ∧
  (sequence_a 3 = 5/8) ∧
  (∀ n : ℕ, n ≥ 2 → sequence_a n + 2/5 * (-1)^n = 3/2 * (sequence_a (n-1) + 2/5 * (-1)^(n-1))) ∧
  (∀ n : ℕ, partial_sum n = 3 * sequence_a n + (-1)^n) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l3229_322930


namespace NUMINAMATH_CALUDE_ninth_day_skating_time_l3229_322918

def minutes_per_hour : ℕ := 60

def skating_time_first_5_days : ℕ := 75
def skating_time_next_3_days : ℕ := 90
def total_days : ℕ := 9
def target_average : ℕ := 85

def total_skating_time : ℕ := 
  (skating_time_first_5_days * 5) + (skating_time_next_3_days * 3)

theorem ninth_day_skating_time :
  (total_skating_time + 120) / total_days = target_average :=
sorry

end NUMINAMATH_CALUDE_ninth_day_skating_time_l3229_322918


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3229_322978

theorem triangle_angle_calculation (A B C : ℝ) : 
  A = 88 → B - C = 20 → A + B + C = 180 → C = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3229_322978


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3229_322951

/-- Represents a three-digit number in base 10 -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  tens_range : tens ≥ 0 ∧ tens ≤ 9
  ones_range : ones ≥ 0 ∧ ones ≤ 9
  hundreds_odd : Odd hundreds

/-- Converts a ThreeDigitNumber to its decimal representation -/
def toDecimal (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Sums all permutations of a ThreeDigitNumber -/
def sumPermutations (n : ThreeDigitNumber) : Nat :=
  toDecimal n +
  (100 * n.hundreds + n.tens + 10 * n.ones) +
  (100 * n.tens + 10 * n.ones + n.hundreds) +
  (100 * n.tens + n.hundreds + 10 * n.ones) +
  (100 * n.ones + 10 * n.hundreds + n.tens) +
  (100 * n.ones + 10 * n.tens + n.hundreds)

theorem unique_three_digit_number :
  ∀ n : ThreeDigitNumber, sumPermutations n = 3300 → toDecimal n = 192 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3229_322951


namespace NUMINAMATH_CALUDE_sin_cos_equality_relation_l3229_322980

open Real

theorem sin_cos_equality_relation :
  (∃ (α β : ℝ), (sin α = sin β ∧ cos α = cos β) ∧ α ≠ β) ∧
  (∀ (α β : ℝ), α = β → (sin α = sin β ∧ cos α = cos β)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equality_relation_l3229_322980


namespace NUMINAMATH_CALUDE_f_properties_l3229_322925

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-1, 3]
def interval : Set ℝ := Set.Icc (-1) 3

-- Theorem for monotonicity and extreme values
theorem f_properties :
  (∀ x y, x < y ∧ x < -1 → f x < f y) ∧  -- Increasing on (-∞, -1)
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧  -- Decreasing on (-1, 1)
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧  -- Increasing on (1, +∞)
  (∀ x ∈ interval, f x ≤ 18) ∧  -- Maximum value
  (∀ x ∈ interval, f x ≥ -2) ∧  -- Minimum value
  (∃ x ∈ interval, f x = 18) ∧  -- Maximum is attained
  (∃ x ∈ interval, f x = -2) :=  -- Minimum is attained
by sorry

end NUMINAMATH_CALUDE_f_properties_l3229_322925


namespace NUMINAMATH_CALUDE_lev_number_pairs_l3229_322981

theorem lev_number_pairs : 
  ∀ a b : ℕ, a + b + a * b = 1000 → 
  ((a = 6 ∧ b = 142) ∨ (a = 142 ∧ b = 6) ∨
   (a = 10 ∧ b = 90) ∨ (a = 90 ∧ b = 10) ∨
   (a = 12 ∧ b = 76) ∨ (a = 76 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_lev_number_pairs_l3229_322981


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l3229_322975

def bag_weights : List Nat := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : Nat) : Prop :=
  t ∈ bag_weights ∧
  ∃ (o c : Nat),
    o + c = (bag_weights.sum - t) ∧
    c = 2 * o

theorem turnip_bag_weights :
  ∀ t, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 :=
by sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l3229_322975


namespace NUMINAMATH_CALUDE_function_difference_constant_l3229_322985

open Function Real

theorem function_difference_constant 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (h_second_deriv : ∀ x, deriv (deriv f) x = deriv (deriv g) x) :
  ∃ C, ∀ x, f x - g x = C :=
sorry

end NUMINAMATH_CALUDE_function_difference_constant_l3229_322985


namespace NUMINAMATH_CALUDE_sheetrock_width_l3229_322990

/-- Given a rectangular piece of sheetrock with length 6 feet and area 30 square feet, its width is 5 feet. -/
theorem sheetrock_width (length : ℝ) (area : ℝ) (width : ℝ) : 
  length = 6 → area = 30 → area = length * width → width = 5 := by
  sorry

end NUMINAMATH_CALUDE_sheetrock_width_l3229_322990


namespace NUMINAMATH_CALUDE_chicken_pasta_pieces_is_two_l3229_322956

/-- Represents the number of chicken pieces in different orders and the total needed -/
structure ChickenOrders where
  barbecue_pieces : ℕ
  fried_dinner_pieces : ℕ
  fried_dinner_orders : ℕ
  chicken_pasta_orders : ℕ
  barbecue_orders : ℕ
  total_pieces : ℕ

/-- Calculates the number of chicken pieces in a Chicken Pasta order -/
def chicken_pasta_pieces (orders : ChickenOrders) : ℕ :=
  (orders.total_pieces -
   (orders.fried_dinner_pieces * orders.fried_dinner_orders +
    orders.barbecue_pieces * orders.barbecue_orders)) /
  orders.chicken_pasta_orders

/-- Theorem stating that the number of chicken pieces in a Chicken Pasta order is 2 -/
theorem chicken_pasta_pieces_is_two (orders : ChickenOrders)
  (h1 : orders.barbecue_pieces = 3)
  (h2 : orders.fried_dinner_pieces = 8)
  (h3 : orders.fried_dinner_orders = 2)
  (h4 : orders.chicken_pasta_orders = 6)
  (h5 : orders.barbecue_orders = 3)
  (h6 : orders.total_pieces = 37) :
  chicken_pasta_pieces orders = 2 := by
  sorry

end NUMINAMATH_CALUDE_chicken_pasta_pieces_is_two_l3229_322956


namespace NUMINAMATH_CALUDE_cherry_tomatoes_weight_l3229_322949

/-- Calculates the total weight of cherry tomatoes in grams -/
def total_weight_grams (initial_kg : ℝ) (additional_g : ℝ) : ℝ :=
  initial_kg * 1000 + additional_g

/-- Theorem: The total weight of cherry tomatoes is 2560 grams -/
theorem cherry_tomatoes_weight :
  total_weight_grams 2 560 = 2560 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomatoes_weight_l3229_322949


namespace NUMINAMATH_CALUDE_simplify_expression_l3229_322904

theorem simplify_expression (x : ℝ) : 3 * (4 * x^2)^4 = 768 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3229_322904


namespace NUMINAMATH_CALUDE_rival_to_jessie_award_ratio_l3229_322957

/-- Given that Scott won 4 awards, Jessie won 3 times as many awards as Scott,
    and the rival won 24 awards, prove that the ratio of awards won by the rival
    to Jessie is 2:1. -/
theorem rival_to_jessie_award_ratio :
  let scott_awards : ℕ := 4
  let jessie_awards : ℕ := 3 * scott_awards
  let rival_awards : ℕ := 24
  (rival_awards : ℚ) / jessie_awards = 2 := by sorry

end NUMINAMATH_CALUDE_rival_to_jessie_award_ratio_l3229_322957


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3229_322932

theorem inequality_and_equality_condition (a b c d : ℝ) :
  (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ≥ Real.sqrt ((a + c)^2 + (b + d)^2)) ∧
  (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) = Real.sqrt ((a + c)^2 + (b + d)^2) ↔ a * d = b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3229_322932


namespace NUMINAMATH_CALUDE_trig_identity_l3229_322933

theorem trig_identity : 
  (Real.cos (20 * π / 180)) / (Real.cos (35 * π / 180) * Real.sqrt (1 - Real.sin (20 * π / 180))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3229_322933


namespace NUMINAMATH_CALUDE_mechanic_job_duration_l3229_322983

/-- Proves that given a mechanic's hourly rate, parts cost, and total bill, the job duration can be calculated. -/
theorem mechanic_job_duration 
  (hourly_rate : ℝ) 
  (parts_cost : ℝ) 
  (total_bill : ℝ) 
  (h : hourly_rate = 45) 
  (p : parts_cost = 225) 
  (t : total_bill = 450) : 
  (total_bill - parts_cost) / hourly_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_job_duration_l3229_322983


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3229_322946

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

theorem quadratic_root_problem (a b c : ℤ) (m n : ℕ) :
  (∀ x : ℤ, a * x^2 + b * x + c = 0 ↔ x = m ∨ x = n) →
  m ≠ n →
  m > 0 →
  n > 0 →
  is_prime (a + b + c) →
  (∃ x : ℤ, a * x^2 + b * x + c = -55) →
  m = 2 →
  n = 17 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3229_322946


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3229_322950

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 1) % 823 = 0 ∧
  (n + 1) % 618 = 0 ∧
  (n + 1) % 3648 = 0 ∧
  (n + 1) % 60 = 0 ∧
  (n + 1) % 3917 = 0 ∧
  (n + 1) % 4203 = 0

theorem smallest_number_divisible_by_all :
  ∃ n : ℕ, is_divisible_by_all n ∧ ∀ m : ℕ, m < n → ¬is_divisible_by_all m :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3229_322950


namespace NUMINAMATH_CALUDE_not_nth_power_of_sum_of_powers_l3229_322988

theorem not_nth_power_of_sum_of_powers (p n : ℕ) (hp : Nat.Prime p) (hn : n > 1) :
  ¬ ∃ m : ℕ, (2^p : ℕ) + (3^p : ℕ) = m^n :=
sorry

end NUMINAMATH_CALUDE_not_nth_power_of_sum_of_powers_l3229_322988


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l3229_322903

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l3229_322903


namespace NUMINAMATH_CALUDE_perfect_squares_digit_parity_l3229_322924

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def units_digit (n : ℕ) : ℕ := n % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem perfect_squares_digit_parity (a b x y : ℕ) :
  is_perfect_square a →
  is_perfect_square b →
  units_digit a = 1 →
  tens_digit a = x →
  units_digit b = 6 →
  tens_digit b = y →
  Even x ∧ Odd y :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_digit_parity_l3229_322924


namespace NUMINAMATH_CALUDE_sqrt_5_minus_2_squared_l3229_322993

theorem sqrt_5_minus_2_squared : (Real.sqrt 5 - 2)^2 = 9 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_minus_2_squared_l3229_322993


namespace NUMINAMATH_CALUDE_type_b_soda_cans_l3229_322943

/-- The number of cans of type B soda that can be purchased for a given amount of money -/
theorem type_b_soda_cans 
  (T : ℕ) -- number of type A cans
  (P : ℕ) -- price in quarters for T cans of type A
  (R : ℚ) -- amount of dollars available
  (h1 : P > 0) -- ensure division by P is valid
  (h2 : T > 0) -- ensure division by T is valid
  : (2 * R * T.cast) / P.cast = (4 * R * T.cast) / (2 * P.cast) := by
  sorry

end NUMINAMATH_CALUDE_type_b_soda_cans_l3229_322943


namespace NUMINAMATH_CALUDE_range_of_y_l3229_322991

theorem range_of_y (y : ℝ) (h1 : 1 / y < 3) (h2 : 1 / y > -4) : y > 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_y_l3229_322991


namespace NUMINAMATH_CALUDE_least_positive_difference_l3229_322999

def geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

def arithmetic_sequence (b₁ d : ℝ) (n : ℕ) : ℝ := b₁ + (n - 1) * d

def sequence_A (n : ℕ) : ℝ := geometric_sequence 3 2 n

def sequence_B (n : ℕ) : ℝ := arithmetic_sequence 15 15 n

def valid_term_A (n : ℕ) : Prop := sequence_A n ≤ 300

def valid_term_B (n : ℕ) : Prop := sequence_B n ≤ 300

theorem least_positive_difference :
  ∃ (m n : ℕ), 
    valid_term_A m ∧ 
    valid_term_B n ∧ 
    (∀ (i j : ℕ), valid_term_A i → valid_term_B j → 
      |sequence_A i - sequence_B j| ≥ |sequence_A m - sequence_B n|) ∧
    |sequence_A m - sequence_B n| = 3 :=
sorry

end NUMINAMATH_CALUDE_least_positive_difference_l3229_322999


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3229_322961

theorem inequality_equivalence (a : ℝ) : (a + 1 < 0) ↔ (a < -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3229_322961


namespace NUMINAMATH_CALUDE_orthocenters_collinear_l3229_322952

-- Define the basic geometric objects
variable (A B C D O : Point)

-- Define the quadrilateral ABCD
def quadrilateral (A B C D : Point) : Prop := sorry

-- Define the inscribed circle O
def inscribedCircle (O : Point) (A B C D : Point) : Prop := sorry

-- Define the orthocenter of a triangle
def orthocenter (P Q R : Point) : Point := sorry

-- Define collinearity of points
def collinear (P Q R : Point) : Prop := sorry

-- State the theorem
theorem orthocenters_collinear 
  (h1 : quadrilateral A B C D) 
  (h2 : inscribedCircle O A B C D) : 
  collinear 
    (orthocenter O A B) 
    (orthocenter O B C) 
    (orthocenter O C D) ∧ 
  collinear 
    (orthocenter O C D) 
    (orthocenter O D A) 
    (orthocenter O A B) :=
sorry

end NUMINAMATH_CALUDE_orthocenters_collinear_l3229_322952


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l3229_322900

/-- Given a square with side length 4 cm, and an infinite series of squares where each subsequent
    square is formed by joining the midpoints of the sides of the previous square,
    the sum of the areas of all squares is 32 cm². -/
theorem sum_of_square_areas (first_square_side : ℝ) (h : first_square_side = 4) :
  let area_sequence : ℕ → ℝ := λ n => first_square_side^2 / 2^n
  ∑' n, area_sequence n = 32 :=
sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l3229_322900


namespace NUMINAMATH_CALUDE_consecutive_product_divisibility_l3229_322969

theorem consecutive_product_divisibility (k : ℤ) : 
  let n := k * (k + 1) * (k + 2) * (k + 3)
  (∃ m : ℤ, n = 11 * m) → 
  (∃ m : ℤ, n = 44 * m) ∧ 
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) * (k + 3) = 66 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_divisibility_l3229_322969


namespace NUMINAMATH_CALUDE_losing_candidate_vote_percentage_l3229_322907

/-- Given the total number of votes and the margin of loss, 
    calculate the percentage of votes received by the losing candidate. -/
theorem losing_candidate_vote_percentage
  (total_votes : ℕ)
  (loss_margin : ℕ)
  (h1 : total_votes = 7800)
  (h2 : loss_margin = 2340) :
  (total_votes - loss_margin) * 100 / total_votes = 70 := by
  sorry

end NUMINAMATH_CALUDE_losing_candidate_vote_percentage_l3229_322907


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3229_322908

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 12) % 8 = 0 ∧ (n - 12) % 12 = 0 ∧ (n - 12) % 22 = 0 ∧ (n - 12) % 24 = 0

theorem smallest_number_divisible_by_all : 
  (is_divisible_by_all 252) ∧ 
  (∀ m : ℕ, m < 252 → ¬(is_divisible_by_all m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3229_322908


namespace NUMINAMATH_CALUDE_smallest_positive_value_cubic_expression_l3229_322965

theorem smallest_positive_value_cubic_expression (a b c : ℕ+) :
  a^3 + b^3 + c^3 - 3*a*b*c ≥ 4 ∧ ∃ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_value_cubic_expression_l3229_322965


namespace NUMINAMATH_CALUDE_P_subset_M_l3229_322959

def M : Set ℕ := {0, 2}

def P : Set ℕ := {x | x ∈ M}

theorem P_subset_M : P ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_P_subset_M_l3229_322959


namespace NUMINAMATH_CALUDE_remaining_problems_to_grade_l3229_322976

theorem remaining_problems_to_grade 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (h1 : total_worksheets = 25) 
  (h2 : graded_worksheets = 12) 
  (h3 : problems_per_worksheet = 15) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 195 := by
sorry

end NUMINAMATH_CALUDE_remaining_problems_to_grade_l3229_322976


namespace NUMINAMATH_CALUDE_total_cost_is_thirteen_l3229_322902

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := 2

/-- The additional cost of a pen compared to a pencil in dollars -/
def pen_additional_cost : ℝ := 9

/-- The cost of a pen in dollars -/
def pen_cost : ℝ := pencil_cost + pen_additional_cost

/-- The total cost of both items in dollars -/
def total_cost : ℝ := pen_cost + pencil_cost

theorem total_cost_is_thirteen : total_cost = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_thirteen_l3229_322902


namespace NUMINAMATH_CALUDE_visitors_yesterday_l3229_322986

def total_visitors : ℕ := 829
def visitors_today : ℕ := 784

theorem visitors_yesterday (total : ℕ) (today : ℕ) (h1 : total = total_visitors) (h2 : today = visitors_today) :
  total - today = 45 := by
  sorry

end NUMINAMATH_CALUDE_visitors_yesterday_l3229_322986


namespace NUMINAMATH_CALUDE_root_value_theorem_l3229_322960

theorem root_value_theorem (m : ℝ) : 2 * m^2 - 3 * m - 1 = 0 → 4 * m^2 - 6 * m + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l3229_322960


namespace NUMINAMATH_CALUDE_tommy_initial_balloons_l3229_322944

/-- The number of balloons Tommy had initially -/
def initial_balloons : ℕ := 26

/-- The number of balloons Tommy's mom gave him -/
def mom_balloons : ℕ := 34

/-- The total number of balloons Tommy had after receiving more from his mom -/
def total_balloons : ℕ := 60

/-- Theorem: Tommy had 26 balloons to start with -/
theorem tommy_initial_balloons : 
  initial_balloons + mom_balloons = total_balloons :=
by sorry

end NUMINAMATH_CALUDE_tommy_initial_balloons_l3229_322944


namespace NUMINAMATH_CALUDE_equilateral_triangle_not_unique_from_angles_l3229_322916

/-- An equilateral triangle -/
structure EquilateralTriangle where
  /-- The side length of the equilateral triangle -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- The theorem stating that two angles do not uniquely determine an equilateral triangle -/
theorem equilateral_triangle_not_unique_from_angles :
  ∃ (t1 t2 : EquilateralTriangle), t1 ≠ t2 ∧ 
  (∀ (θ : ℝ), 0 < θ ∧ θ < π → 
    (θ = π/3 ↔ (∃ (i : Fin 3), θ = π/3))) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_not_unique_from_angles_l3229_322916


namespace NUMINAMATH_CALUDE_benny_initial_books_l3229_322937

/-- The number of books Benny had initially -/
def benny_initial : ℕ := sorry

/-- The number of books Tim has -/
def tim_books : ℕ := 33

/-- The number of books Sandy received from Benny -/
def sandy_received : ℕ := 10

/-- The total number of books they have together now -/
def total_books : ℕ := 47

theorem benny_initial_books : 
  benny_initial = 24 := by sorry

end NUMINAMATH_CALUDE_benny_initial_books_l3229_322937


namespace NUMINAMATH_CALUDE_elena_allowance_spending_l3229_322964

theorem elena_allowance_spending (A : ℝ) : ∃ (m s : ℝ),
  m = (1/4) * (A - s) ∧
  s = (1/10) * (A - m) ∧
  m + s = (4/13) * A :=
by sorry

end NUMINAMATH_CALUDE_elena_allowance_spending_l3229_322964


namespace NUMINAMATH_CALUDE_pizza_toppings_l3229_322989

theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (mushroom_slices : ℕ) :
  total_slices = 10 →
  cheese_slices = 5 →
  mushroom_slices = 7 →
  cheese_slices + mushroom_slices - total_slices = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3229_322989


namespace NUMINAMATH_CALUDE_complex_inequality_l3229_322994

theorem complex_inequality (a : ℝ) : 
  (1 - Complex.I) + (1 + Complex.I) * a ≠ 0 → a ≠ -1 ∧ a ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l3229_322994


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l3229_322954

/-- A color representation --/
inductive Color
| Black
| White

/-- A grid of colors --/
def Grid := Fin 3 → Fin 7 → Color

/-- A rectangle in the grid --/
structure Rectangle where
  x1 : Fin 7
  x2 : Fin 7
  y1 : Fin 3
  y2 : Fin 3
  h_distinct : x1 ≠ x2 ∧ y1 ≠ y2

/-- Check if a rectangle has vertices of the same color --/
def sameColorVertices (g : Grid) (r : Rectangle) : Prop :=
  g r.y1 r.x1 = g r.y1 r.x2 ∧
  g r.y1 r.x1 = g r.y2 r.x1 ∧
  g r.y1 r.x1 = g r.y2 r.x2

/-- Theorem: In any 3x7 grid coloring, there exists a rectangle with vertices of the same color --/
theorem exists_same_color_rectangle (g : Grid) : ∃ r : Rectangle, sameColorVertices g r := by
  sorry

end NUMINAMATH_CALUDE_exists_same_color_rectangle_l3229_322954


namespace NUMINAMATH_CALUDE_autumn_outing_problem_l3229_322979

/-- Autumn Outing Problem -/
theorem autumn_outing_problem 
  (bus_seats : ℕ) 
  (public_bus_seats : ℕ) 
  (bus_count : ℕ) 
  (teachers_per_bus : ℕ) 
  (extra_seats_buses : ℕ) 
  (extra_teachers_public : ℕ) 
  (h1 : bus_seats = 39)
  (h2 : public_bus_seats = 27)
  (h3 : bus_count + 2 = public_bus_count)
  (h4 : teachers_per_bus = 2)
  (h5 : extra_seats_buses = 3)
  (h6 : extra_teachers_public = 3)
  (h7 : bus_seats * bus_count = teachers_per_bus * bus_count + students + extra_seats_buses)
  (h8 : public_bus_seats * public_bus_count = teachers + students)
  (h9 : teachers = public_bus_count + extra_teachers_public) :
  teachers = 18 ∧ students = 330 := by
  sorry


end NUMINAMATH_CALUDE_autumn_outing_problem_l3229_322979


namespace NUMINAMATH_CALUDE_min_value_theorem_l3229_322995

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hbc : b + c ≥ a) :
  b / c + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3229_322995


namespace NUMINAMATH_CALUDE_exchange_equality_l3229_322968

theorem exchange_equality (a₁ b₁ a₂ b₂ : ℝ) 
  (h1 : a₁^2 + b₁^2 = 1)
  (h2 : a₂^2 + b₂^2 = 1)
  (h3 : a₁*a₂ + b₁*b₂ = 0) :
  (a₁^2 + a₂^2 = 1) ∧ (b₁^2 + b₂^2 = 1) ∧ (a₁*b₁ + a₂*b₂ = 0) := by
sorry

end NUMINAMATH_CALUDE_exchange_equality_l3229_322968


namespace NUMINAMATH_CALUDE_remaining_students_count_l3229_322917

/-- The number of groups with 15 students -/
def groups_15 : ℕ := 4

/-- The number of groups with 18 students -/
def groups_18 : ℕ := 2

/-- The number of students in each of the first 4 groups -/
def students_per_group_15 : ℕ := 15

/-- The number of students in each of the last 2 groups -/
def students_per_group_18 : ℕ := 18

/-- The number of students who left early from the first 4 groups -/
def left_early_15 : ℕ := 8

/-- The number of students who left early from the last 2 groups -/
def left_early_18 : ℕ := 5

/-- The total number of remaining students -/
def remaining_students : ℕ := 
  (groups_15 * students_per_group_15 - left_early_15) + 
  (groups_18 * students_per_group_18 - left_early_18)

theorem remaining_students_count : remaining_students = 83 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_count_l3229_322917


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3229_322927

theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3 / 4 : ℚ) * 16 * banana_value = 6 * orange_value →
  (1 / 3 : ℚ) * 9 * banana_value = (3 / 2 : ℚ) * orange_value :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3229_322927


namespace NUMINAMATH_CALUDE_dog_park_ratio_l3229_322941

theorem dog_park_ratio (total : ℕ) (running : ℕ) (doing_nothing : ℕ) 
  (h1 : total = 88)
  (h2 : running = 12)
  (h3 : doing_nothing = 10)
  (h4 : total / 4 = total / 4) : -- This represents that 1/4 of dogs are barking
  (total - running - (total / 4) - doing_nothing) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_park_ratio_l3229_322941


namespace NUMINAMATH_CALUDE_infiniteSum_eq_one_l3229_322963

/-- Sequence F defined recursively -/
def F : ℕ → ℚ
  | 0 => 0
  | 1 => 3/2
  | (n+2) => 5/2 * F (n+1) - F n

/-- The sum of 1/F(2^n) from n=0 to infinity -/
noncomputable def infiniteSum : ℚ := ∑' n, 1 / F (2^n)

/-- Theorem stating that the infinite sum is equal to 1 -/
theorem infiniteSum_eq_one : infiniteSum = 1 := by sorry

end NUMINAMATH_CALUDE_infiniteSum_eq_one_l3229_322963


namespace NUMINAMATH_CALUDE_num_distinct_configurations_l3229_322934

/-- The group of cube rotations -/
def CubeRotations : Type := Unit

/-- The number of elements in the group of cube rotations -/
def numRotations : ℕ := 4

/-- The number of configurations fixed by the identity rotation -/
def fixedByIdentity : ℕ := 56

/-- The number of configurations fixed by each 180-degree rotation -/
def fixedBy180Rotation : ℕ := 6

/-- The number of 180-degree rotations -/
def num180Rotations : ℕ := 3

/-- The total number of fixed points across all rotations -/
def totalFixedPoints : ℕ := fixedByIdentity + num180Rotations * fixedBy180Rotation

/-- The theorem stating the number of distinct configurations -/
theorem num_distinct_configurations : 
  (totalFixedPoints : ℚ) / numRotations = 19 / 2 := by sorry

end NUMINAMATH_CALUDE_num_distinct_configurations_l3229_322934


namespace NUMINAMATH_CALUDE_power_mod_eleven_l3229_322977

theorem power_mod_eleven : 3^251 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l3229_322977


namespace NUMINAMATH_CALUDE_equation_solution_l3229_322948

theorem equation_solution : 
  ∀ x y : ℕ, x^2 + x*y = y + 92 ↔ (x = 2 ∧ y = 88) ∨ (x = 8 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3229_322948


namespace NUMINAMATH_CALUDE_sqrt_three_diamond_sqrt_three_l3229_322905

-- Define the operation ¤
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt_three_diamond_sqrt_three : diamond (Real.sqrt 3) (Real.sqrt 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_diamond_sqrt_three_l3229_322905


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3229_322935

/-- Given the cost price and selling price of an article, calculate the profit percentage. -/
theorem profit_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 500 → selling_price = 675 → 
  (selling_price - cost_price) / cost_price * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3229_322935


namespace NUMINAMATH_CALUDE_victors_stickers_l3229_322945

theorem victors_stickers (flower_stickers : ℕ) (animal_stickers : ℕ) : 
  flower_stickers = 8 →
  animal_stickers = flower_stickers - 2 →
  flower_stickers + animal_stickers = 14 :=
by sorry

end NUMINAMATH_CALUDE_victors_stickers_l3229_322945


namespace NUMINAMATH_CALUDE_problem_statement_l3229_322962

-- Definition of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Definition of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

-- Definition of a periodic function
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_statement :
  (∀ (f : ℝ → ℝ), IsEven (fun x ↦ f x + f (-x))) ∧
  (∀ (f : ℝ → ℝ), IsOdd f → IsOdd (fun x ↦ f (x + 2)) → IsPeriodic f 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3229_322962


namespace NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l3229_322971

theorem largest_angle_convex_pentagon (x : ℝ) : 
  (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) = 540 →
  max (x + 2) (max (2 * x + 3) (max (3 * x + 4) (max (4 * x + 5) (5 * x + 6)))) = 538 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l3229_322971


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3229_322913

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3229_322913


namespace NUMINAMATH_CALUDE_tan_sum_specific_angles_l3229_322973

theorem tan_sum_specific_angles (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_specific_angles_l3229_322973


namespace NUMINAMATH_CALUDE_number_of_arrangements_l3229_322910

/-- The number of people in the line -/
def total_people : ℕ := 6

/-- The number of people in the adjacent group (Xiao Kai and 2 elderly) -/
def adjacent_group : ℕ := 3

/-- The number of volunteers -/
def volunteers : ℕ := 3

/-- The number of ways to arrange the adjacent group internally -/
def adjacent_group_arrangements : ℕ := 2

/-- The number of possible positions for the adjacent group in the line -/
def adjacent_group_positions : ℕ := total_people - adjacent_group - 1

/-- The number of arrangements for the volunteers -/
def volunteer_arrangements : ℕ := 6

theorem number_of_arrangements :
  adjacent_group_arrangements * adjacent_group_positions * volunteer_arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l3229_322910


namespace NUMINAMATH_CALUDE_employment_after_growth_and_new_category_l3229_322992

/-- Represents the employment data for town X -/
structure TownEmployment where
  initial_rate : ℝ
  annual_growth : ℝ
  years : ℕ
  male_percentage : ℝ
  tourism_percentage : ℝ
  female_edu_percentage : ℝ

/-- Theorem about employment percentages after growth and new category introduction -/
theorem employment_after_growth_and_new_category 
  (town : TownEmployment)
  (h_initial : town.initial_rate = 0.64)
  (h_growth : town.annual_growth = 0.02)
  (h_years : town.years = 5)
  (h_male : town.male_percentage = 0.55)
  (h_tourism : town.tourism_percentage = 0.1)
  (h_female_edu : town.female_edu_percentage = 0.6) :
  let final_rate := town.initial_rate + town.annual_growth * town.years
  let female_percentage := 1 - town.male_percentage
  (female_percentage = 0.45) ∧ 
  (town.female_edu_percentage > 0.5) := by
  sorry

#check employment_after_growth_and_new_category

end NUMINAMATH_CALUDE_employment_after_growth_and_new_category_l3229_322992
