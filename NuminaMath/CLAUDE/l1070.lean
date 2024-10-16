import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1070_107040

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The main theorem: For a geometric sequence satisfying given conditions, a₂ + a₆ = 34 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsGeometricSequence a →
  a 3 + a 5 = 20 →
  a 4 = 8 →
  a 2 + a 6 = 34 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l1070_107040


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l1070_107099

theorem roots_of_quadratic (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l1070_107099


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l1070_107002

/-- Given a rhombus with area 64/5 square centimeters and one diagonal 64/9 centimeters,
    prove that the other diagonal is 18/5 centimeters. -/
theorem rhombus_diagonal (area : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) : 
  area = 64/5 → 
  diagonal1 = 64/9 → 
  area = (diagonal1 * diagonal2) / 2 → 
  diagonal2 = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l1070_107002


namespace NUMINAMATH_CALUDE_ellipse_equation_l1070_107027

/-- An ellipse with the given properties has the equation x²/2 + 3y²/2 = 1 or 3x²/2 + y²/2 = 1 -/
theorem ellipse_equation (E : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ E ↔ ∃ (m n : ℝ), m * x^2 + n * y^2 = 1) →  -- E is an ellipse
  (0, 0) ∈ E →  -- center at origin
  (∃ (a : ℝ), (a, 0) ∈ E ∧ (-a, 0) ∈ E) ∨ (∃ (b : ℝ), (0, b) ∈ E ∧ (0, -b) ∈ E) →  -- foci on coordinate axis
  (∃ (x : ℝ), P = (x, x + 1) ∧ Q = (x, x + 1) ∧ P ∈ E ∧ Q ∈ E) →  -- P and Q on y = x + 1 and on E
  P.1 * Q.1 + P.2 * Q.2 = 0 →  -- OP · OQ = 0
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 5/2 →  -- |PQ|² = (√10/2)² = 5/2
  (∀ (x y : ℝ), (x, y) ∈ E ↔ x^2/2 + 3*y^2/2 = 1) ∨
  (∀ (x y : ℝ), (x, y) ∈ E ↔ 3*x^2/2 + y^2/2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1070_107027


namespace NUMINAMATH_CALUDE_comic_book_stacking_arrangements_l1070_107041

theorem comic_book_stacking_arrangements :
  let hulk_comics : ℕ := 8
  let ironman_comics : ℕ := 7
  let wolverine_comics : ℕ := 6
  let total_comics : ℕ := hulk_comics + ironman_comics + wolverine_comics
  let arrange_hulk : ℕ := Nat.factorial hulk_comics
  let arrange_ironman : ℕ := Nat.factorial ironman_comics
  let arrange_wolverine : ℕ := Nat.factorial wolverine_comics
  let arrange_within_groups : ℕ := arrange_hulk * arrange_ironman * arrange_wolverine
  let arrange_groups : ℕ := Nat.factorial 3
  arrange_within_groups * arrange_groups = 69657088000 :=
by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacking_arrangements_l1070_107041


namespace NUMINAMATH_CALUDE_special_bet_cost_l1070_107055

def lottery_numbers : ℕ := 36
def numbers_per_bet : ℕ := 7
def cost_per_bet : ℕ := 2

def consecutive_numbers_01_to_10 : ℕ := 3
def consecutive_numbers_11_to_20 : ℕ := 2
def single_number_21_to_30 : ℕ := 1
def single_number_31_to_36 : ℕ := 1

def ways_01_to_10 : ℕ := 10 - consecutive_numbers_01_to_10 + 1
def ways_11_to_20 : ℕ := 10 - consecutive_numbers_11_to_20 + 1
def ways_21_to_30 : ℕ := 10
def ways_31_to_36 : ℕ := 6

theorem special_bet_cost (total_combinations : ℕ) (total_cost : ℕ) :
  total_combinations = ways_01_to_10 * ways_11_to_20 * ways_21_to_30 * ways_31_to_36 ∧
  total_cost = total_combinations * cost_per_bet ∧
  total_cost = 8640 := by
  sorry

end NUMINAMATH_CALUDE_special_bet_cost_l1070_107055


namespace NUMINAMATH_CALUDE_johns_original_earnings_l1070_107065

/-- Given that John's weekly earnings increased by 20% to $72, prove that his original weekly earnings were $60. -/
theorem johns_original_earnings (current_earnings : ℝ) (increase_rate : ℝ) : 
  current_earnings = 72 ∧ increase_rate = 0.20 → 
  current_earnings / (1 + increase_rate) = 60 := by
  sorry

end NUMINAMATH_CALUDE_johns_original_earnings_l1070_107065


namespace NUMINAMATH_CALUDE_perpendicular_vectors_y_value_l1070_107014

theorem perpendicular_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) : 
  a = (2, 4) → 
  b = (-4, y) → 
  a.1 * b.1 + a.2 * b.2 = 0 → 
  y = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_y_value_l1070_107014


namespace NUMINAMATH_CALUDE_incorrect_equation_is_false_l1070_107021

/-- Represents the number of 1-yuan stamps purchased -/
def x : ℕ := sorry

/-- The total number of stamps purchased -/
def total_stamps : ℕ := 12

/-- The total amount spent in yuan -/
def total_spent : ℕ := 20

/-- The equation representing the correct relationship between x, total stamps, and total spent -/
def correct_equation : Prop := x + 2 * (total_stamps - x) = total_spent

/-- The incorrect equation to be proven false -/
def incorrect_equation : Prop := 2 * (total_stamps - x) - total_spent = x

theorem incorrect_equation_is_false :
  correct_equation → ¬incorrect_equation := by sorry

end NUMINAMATH_CALUDE_incorrect_equation_is_false_l1070_107021


namespace NUMINAMATH_CALUDE_lcm_1404_972_l1070_107039

theorem lcm_1404_972 : Nat.lcm 1404 972 = 88452 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1404_972_l1070_107039


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1070_107059

theorem absolute_value_inequality (x : ℝ) : 
  |x - 2| + |x + 1| < 4 ↔ x ∈ Set.Ioo (-7/2) (-1) ∪ Set.Ico (-1) (5/2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1070_107059


namespace NUMINAMATH_CALUDE_students_wanting_fruit_l1070_107007

theorem students_wanting_fruit (red_apples green_apples extra_fruit : ℕ) 
  (h1 : red_apples = 42)
  (h2 : green_apples = 7)
  (h3 : extra_fruit = 40) :
  red_apples + green_apples - extra_fruit = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_wanting_fruit_l1070_107007


namespace NUMINAMATH_CALUDE_greatest_k_value_l1070_107075

theorem greatest_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 8 = 0 ∧ 
    x₂^2 + k*x₂ + 8 = 0 ∧ 
    |x₁ - x₂| = Real.sqrt 89) →
  k ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_value_l1070_107075


namespace NUMINAMATH_CALUDE_blocks_remaining_l1070_107025

theorem blocks_remaining (initial_blocks : ℕ) (first_tower : ℕ) (second_tower : ℕ)
  (h1 : initial_blocks = 78)
  (h2 : first_tower = 19)
  (h3 : second_tower = 25) :
  initial_blocks - first_tower - second_tower = 34 := by
  sorry

end NUMINAMATH_CALUDE_blocks_remaining_l1070_107025


namespace NUMINAMATH_CALUDE_clock_hands_minimum_time_l1070_107036

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Converts minutes to Time structure -/
def minutesToTime (m : ℕ) : Time :=
  { hours := m / 60
    minutes := m % 60
    valid := by sorry }

theorem clock_hands_minimum_time :
  let t1 : Time := { hours := 0, minutes := 45, valid := by sorry }
  let t2 : Time := { hours := 3, minutes := 30, valid := by sorry }
  let diff := timeDifferenceInMinutes t1 t2
  let result := minutesToTime diff
  result.hours = 2 ∧ result.minutes = 45 := by sorry

end NUMINAMATH_CALUDE_clock_hands_minimum_time_l1070_107036


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1070_107074

theorem simplify_sqrt_expression : 
  Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1070_107074


namespace NUMINAMATH_CALUDE_solve_equation_l1070_107043

theorem solve_equation (x : ℝ) : 3 * x + 15 = (1 / 3) * (6 * x + 45) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1070_107043


namespace NUMINAMATH_CALUDE_units_digit_power_four_l1070_107051

theorem units_digit_power_four (a : ℤ) (n : ℕ) : 
  10 ∣ (a^(n+4) - a^n) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_power_four_l1070_107051


namespace NUMINAMATH_CALUDE_max_sum_of_product_l1070_107079

theorem max_sum_of_product (a b : ℤ) : 
  a ≠ b → a * b = -132 → a ≤ b → (∀ x y : ℤ, x ≠ y → x * y = -132 → x ≤ y → a + b ≥ x + y) → a + b = -1 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_product_l1070_107079


namespace NUMINAMATH_CALUDE_sams_coins_value_l1070_107061

/-- Represents the value of Sam's coins in dollars -/
def total_value : ℚ :=
  let total_coins : ℕ := 30
  let nickels : ℕ := 12
  let dimes : ℕ := total_coins - nickels
  let nickel_value : ℚ := 5 / 100
  let dime_value : ℚ := 10 / 100
  (nickels : ℚ) * nickel_value + (dimes : ℚ) * dime_value

theorem sams_coins_value : total_value = 2.40 := by
  sorry

end NUMINAMATH_CALUDE_sams_coins_value_l1070_107061


namespace NUMINAMATH_CALUDE_triangle_reconstruction_from_nagel_point_vertex_and_altitude_foot_l1070_107056

/- Define the necessary types and structures -/
structure Triangle where
  A : Point
  B : Point
  C : Point

structure Point where
  x : ℝ
  y : ℝ

/- Define the given information -/
def nagel_point (t : Triangle) : Point := sorry
def altitude_foot (t : Triangle) (v : Point) : Point := sorry

/- State the theorem -/
theorem triangle_reconstruction_from_nagel_point_vertex_and_altitude_foot 
  (N : Point) (B : Point) (E : Point) :
  ∃! (t : Triangle), 
    B = t.B ∧ 
    N = nagel_point t ∧ 
    E = altitude_foot t B := by
  sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_from_nagel_point_vertex_and_altitude_foot_l1070_107056


namespace NUMINAMATH_CALUDE_inequality_proof_l1070_107087

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1070_107087


namespace NUMINAMATH_CALUDE_min_odd_in_A_P_l1070_107023

/-- The set A_P for a polynomial P -/
def A_P (P : ℝ → ℝ) : Set ℝ := {x : ℝ | ∃ c : ℝ, P x = c}

/-- A polynomial is of degree 8 -/
def is_degree_8 (P : ℝ → ℝ) : Prop :=
  ∃ a₈ a₇ a₆ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ, a₈ ≠ 0 ∧
    ∀ x, P x = a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + 
           a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

theorem min_odd_in_A_P (P : ℝ → ℝ) (h : is_degree_8 P) (h8 : 8 ∈ A_P P) :
  ∃ x ∈ A_P P, Odd x :=
sorry

end NUMINAMATH_CALUDE_min_odd_in_A_P_l1070_107023


namespace NUMINAMATH_CALUDE_size_ratio_proof_l1070_107047

def anna_size : ℕ := 2

def becky_size (anna_size : ℕ) : ℕ := 3 * anna_size

def ginger_size : ℕ := 8

theorem size_ratio_proof (anna_size : ℕ) (becky_size : ℕ → ℕ) (ginger_size : ℕ)
  (h1 : anna_size = 2)
  (h2 : becky_size anna_size = 3 * anna_size)
  (h3 : ginger_size = 8)
  (h4 : ∃ k : ℕ, ginger_size = k * (becky_size anna_size - 4)) :
  ginger_size / (becky_size anna_size) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_size_ratio_proof_l1070_107047


namespace NUMINAMATH_CALUDE_ricardo_coin_difference_l1070_107019

/-- The total number of coins Ricardo has -/
def total_coins : ℕ := 2020

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of pennies Ricardo has -/
def num_pennies : ℕ → ℕ := λ p => p

/-- The number of nickels Ricardo has -/
def num_nickels : ℕ → ℕ := λ p => total_coins - p

/-- The total value of Ricardo's coins in cents -/
def total_value : ℕ → ℕ := λ p => 
  penny_value * num_pennies p + nickel_value * num_nickels p

/-- The constraint that Ricardo has at least one penny and one nickel -/
def valid_distribution : ℕ → Prop := λ p => 
  1 ≤ num_pennies p ∧ 1 ≤ num_nickels p

theorem ricardo_coin_difference : 
  ∃ (max_p min_p : ℕ), 
    valid_distribution max_p ∧ 
    valid_distribution min_p ∧ 
    (∀ p, valid_distribution p → total_value p ≤ total_value max_p) ∧
    (∀ p, valid_distribution p → total_value min_p ≤ total_value p) ∧
    total_value max_p - total_value min_p = 8072 := by
  sorry

end NUMINAMATH_CALUDE_ricardo_coin_difference_l1070_107019


namespace NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l1070_107035

def num_students : ℕ := 3
def num_teachers : ℕ := 2

def arrangement_count : ℕ := 72

theorem teachers_not_adjacent_arrangements :
  (Nat.factorial num_students) * (num_students + 1) * num_teachers = arrangement_count :=
by sorry

end NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l1070_107035


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1070_107032

theorem quadratic_inequality_solution_range (d : ℝ) :
  d > 0 →
  (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ d < 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1070_107032


namespace NUMINAMATH_CALUDE_cubic_root_power_sum_l1070_107085

theorem cubic_root_power_sum (p q n : ℝ) (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + p*x₁^2 + q*x₁ + n = 0 →
  x₂^3 + p*x₂^2 + q*x₂ + n = 0 →
  x₃^3 + p*x₃^2 + q*x₃ + n = 0 →
  q^2 = 2*n*p →
  x₁^4 + x₂^4 + x₃^4 = (x₁^2 + x₂^2 + x₃^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_power_sum_l1070_107085


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1070_107070

-- Define the speed of the stream
def stream_speed : ℝ := 5

-- Define the distance traveled downstream
def downstream_distance : ℝ := 81

-- Define the time taken to travel downstream
def downstream_time : ℝ := 3

-- Define the speed of the boat in still water
def boat_speed : ℝ := 22

-- Theorem statement
theorem boat_speed_in_still_water :
  boat_speed = downstream_distance / downstream_time - stream_speed :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1070_107070


namespace NUMINAMATH_CALUDE_share_calculation_l1070_107084

theorem share_calculation (total A B C : ℝ) 
  (h1 : total = 1800)
  (h2 : A = (2/5) * (B + C))
  (h3 : B = (1/5) * (A + C))
  (h4 : A + B + C = total) :
  A = 3600/7 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l1070_107084


namespace NUMINAMATH_CALUDE_initial_fish_count_l1070_107095

/-- The number of fish moved to a different tank -/
def fish_moved : ℕ := 68

/-- The number of fish remaining in the first tank -/
def fish_remaining : ℕ := 144

/-- The initial number of fish in the first tank -/
def initial_fish : ℕ := fish_moved + fish_remaining

theorem initial_fish_count : initial_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_initial_fish_count_l1070_107095


namespace NUMINAMATH_CALUDE_equation_solutions_l1070_107080

theorem equation_solutions (x : ℝ) : 
  (8 / (Real.sqrt (x - 9) - 10) + 2 / (Real.sqrt (x - 9) - 5) + 
   9 / (Real.sqrt (x - 9) + 5) + 15 / (Real.sqrt (x - 9) + 10) = 0) ↔ 
  (x = (70/23)^2 + 9 ∨ x = (25/11)^2 + 9 ∨ x = 575/34 + 9) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1070_107080


namespace NUMINAMATH_CALUDE_number_of_pickers_l1070_107030

/-- Given information about grape harvesting, calculate the number of pickers --/
theorem number_of_pickers (drums_per_day : ℕ) (total_drums : ℕ) (total_days : ℕ) 
  (h1 : drums_per_day = 108)
  (h2 : total_drums = 6264)
  (h3 : total_days = 58)
  (h4 : total_drums = drums_per_day * total_days) :
  total_drums / drums_per_day = 58 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pickers_l1070_107030


namespace NUMINAMATH_CALUDE_family_ages_l1070_107096

/-- Represents the ages of a family with a father, mother, and three daughters. -/
structure FamilyAges where
  father : ℕ
  mother : ℕ
  eldest : ℕ
  middle : ℕ
  youngest : ℕ

/-- The family ages satisfy the given conditions. -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  -- Total age is 90
  ages.father + ages.mother + ages.eldest + ages.middle + ages.youngest = 90 ∧
  -- Age difference between daughters is 2 years
  ages.eldest = ages.middle + 2 ∧
  ages.middle = ages.youngest + 2 ∧
  -- Mother's age is 10 years more than sum of daughters' ages
  ages.mother = ages.eldest + ages.middle + ages.youngest + 10 ∧
  -- Age difference between father and mother equals middle daughter's age
  ages.father - ages.mother = ages.middle

/-- The theorem stating the ages of the family members. -/
theorem family_ages : ∃ (ages : FamilyAges), satisfiesConditions ages ∧ 
  ages.father = 38 ∧ ages.mother = 31 ∧ ages.eldest = 9 ∧ ages.middle = 7 ∧ ages.youngest = 5 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_l1070_107096


namespace NUMINAMATH_CALUDE_angle_equality_l1070_107050

-- Define the problem statement
theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.sqrt 2 * Real.sin (π/6) = Real.cos θ - Real.sin θ) : 
  θ = π/12 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l1070_107050


namespace NUMINAMATH_CALUDE_inequality_implication_l1070_107072

theorem inequality_implication (m a b : ℝ) : a * m^2 > b * m^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1070_107072


namespace NUMINAMATH_CALUDE_min_n_is_correct_l1070_107098

/-- The minimum positive integer n for which the expansion of (x^2 + 1/(3x^3))^n contains a constant term -/
def min_n : ℕ := 5

/-- Predicate to check if the expansion contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (r : ℕ), 2 * n = 5 * r

theorem min_n_is_correct :
  (∀ m : ℕ, m > 0 ∧ m < min_n → ¬(has_constant_term m)) ∧
  has_constant_term min_n :=
sorry

end NUMINAMATH_CALUDE_min_n_is_correct_l1070_107098


namespace NUMINAMATH_CALUDE_third_jumper_height_l1070_107034

/-- The height of Ravi's jump in inches -/
def ravi_jump : ℝ := 39

/-- The height of the first next highest jumper in inches -/
def jumper1 : ℝ := 23

/-- The height of the second next highest jumper in inches -/
def jumper2 : ℝ := 27

/-- The factor by which Ravi can jump higher than the average of the next three highest jumpers -/
def ravi_factor : ℝ := 1.5

/-- The height of the third next highest jumper in inches -/
def jumper3 : ℝ := 28

theorem third_jumper_height :
  ravi_jump = ravi_factor * ((jumper1 + jumper2 + jumper3) / 3) :=
by sorry

end NUMINAMATH_CALUDE_third_jumper_height_l1070_107034


namespace NUMINAMATH_CALUDE_binomial_expectation_six_half_l1070_107077

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Theorem: The expected value of X ~ B(6, 1/2) is 3 -/
theorem binomial_expectation_six_half :
  let X : BinomialDistribution := ⟨6, 1/2, by norm_num⟩
  expectation X = 3 := by sorry

end NUMINAMATH_CALUDE_binomial_expectation_six_half_l1070_107077


namespace NUMINAMATH_CALUDE_hexagon_area_right_triangle_l1070_107010

/-- Given a right-angled triangle with hypotenuse c and sum of legs d,
    the area of the hexagon formed by the outer vertices of squares
    drawn on the sides of the triangle is c^2 + d^2. -/
theorem hexagon_area_right_triangle (c d : ℝ) (h : c > 0) (h' : d > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = d ∧ a^2 + b^2 = c^2 ∧
  (c^2 + a^2 + b^2 + 2*a*b : ℝ) = c^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_right_triangle_l1070_107010


namespace NUMINAMATH_CALUDE_probability_chords_intersect_2000_probability_chords_intersect_general_l1070_107091

/-- Given a circle with evenly spaced points, this function calculates the probability
    that chord AB intersects chord CD when five distinct points are randomly selected. -/
def probability_chords_intersect (n : ℕ) : ℚ :=
  if n < 5 then 0
  else 1 / 15

/-- Theorem stating that the probability of chord AB intersecting chord CD
    when five distinct points are randomly selected from 2000 evenly spaced
    points on a circle is 1/15. -/
theorem probability_chords_intersect_2000 :
  probability_chords_intersect 2000 = 1 / 15 := by
  sorry

/-- Theorem stating that the probability of chord AB intersecting chord CD
    is 1/15 for any number of evenly spaced points on a circle, as long as
    there are at least 5 points. -/
theorem probability_chords_intersect_general (n : ℕ) (h : n ≥ 5) :
  probability_chords_intersect n = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_chords_intersect_2000_probability_chords_intersect_general_l1070_107091


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l1070_107046

/-- Given vectors in ℝ², prove that if a + 3b is parallel to c, then m = -6 -/
theorem vector_parallel_condition (a b c : ℝ × ℝ) (m : ℝ) 
  (ha : a = (-2, 3))
  (hb : b = (3, 1))
  (hc : c = (-7, m))
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 3 • b = k • c) :
  m = -6 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l1070_107046


namespace NUMINAMATH_CALUDE_newsletter_cost_l1070_107005

def newsletter_cost_exists : Prop :=
  ∃ x : ℝ, 
    (14 * x < 16) ∧ 
    (19 * x > 21) ∧ 
    (∀ y : ℝ, (14 * y < 16) ∧ (19 * y > 21) → |x - 1.11| ≤ |y - 1.11|)

theorem newsletter_cost : newsletter_cost_exists := by sorry

end NUMINAMATH_CALUDE_newsletter_cost_l1070_107005


namespace NUMINAMATH_CALUDE_quadratic_intersection_properties_l1070_107022

/-- A quadratic function f(x) = x^2 + 2x + b intersecting both coordinate axes at three points -/
structure QuadraticIntersection (b : ℝ) :=
  (intersects_axes : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + b = 0 ∧ x₂^2 + 2*x₂ + b = 0)
  (y_intercept : b ≠ 0)

/-- The circle passing through the three intersection points -/
def intersection_circle (b : ℝ) (h : QuadraticIntersection b) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - (b + 1)*p.2 + b = 0}

/-- Main theorem: properties of the quadratic function and its intersection circle -/
theorem quadratic_intersection_properties (b : ℝ) (h : QuadraticIntersection b) :
  b < 1 ∧ 
  ∀ (p : ℝ × ℝ), p ∈ intersection_circle b h ↔ p.1^2 + p.2^2 + 2*p.1 - (b + 1)*p.2 + b = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_properties_l1070_107022


namespace NUMINAMATH_CALUDE_parabola_focal_chord_inclination_l1070_107063

theorem parabola_focal_chord_inclination (x y : ℝ) (α : ℝ) : 
  y^2 = 6*x →  -- parabola equation
  12 = 6 / (Real.sin α)^2 →  -- focal chord length condition
  α = π/4 ∨ α = 3*π/4 :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_parabola_focal_chord_inclination_l1070_107063


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l1070_107071

theorem modular_inverse_of_5_mod_23 :
  ∃ x : ℕ, x ≤ 22 ∧ (5 * x) % 23 = 1 :=
by
  use 14
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l1070_107071


namespace NUMINAMATH_CALUDE_hyperbola_intersection_slopes_product_l1070_107069

/-- Hyperbola C with asymptotic line equation y = ±√3x and point P(2,3) on it -/
structure Hyperbola :=
  (asymptote : ℝ → ℝ)
  (point : ℝ × ℝ)
  (h_asymptote : ∀ x, asymptote x = Real.sqrt 3 * x ∨ asymptote x = -Real.sqrt 3 * x)
  (h_point : point = (2, 3))

/-- Line l: y = kx + m -/
structure Line :=
  (k m : ℝ)

/-- Intersection points A and B of line l with hyperbola C -/
structure Intersection :=
  (A B : ℝ × ℝ)
  (k₁ k₂ : ℝ)

/-- The theorem to be proved -/
theorem hyperbola_intersection_slopes_product
  (C : Hyperbola) (l : Line) (I : Intersection) :
  ∃ (k m : ℝ), l.k = -3/2 ∧ I.k₁ * I.k₂ = -3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_slopes_product_l1070_107069


namespace NUMINAMATH_CALUDE_ratio_equality_l1070_107018

theorem ratio_equality (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1070_107018


namespace NUMINAMATH_CALUDE_set_intersection_equality_l1070_107060

-- Define set A
def A : Set ℝ := {y | ∃ x > 1, y = Real.log x}

-- Define set B
def B : Set ℝ := {-2, -1, 1, 2}

-- Theorem statement
theorem set_intersection_equality : (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l1070_107060


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1070_107090

theorem trigonometric_identity : 
  Real.sin (315 * π / 180) - Real.cos (135 * π / 180) + 2 * Real.sin (570 * π / 180) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1070_107090


namespace NUMINAMATH_CALUDE_pages_per_donut_l1070_107053

/-- Given Jean's writing and eating habits, calculate the number of pages she writes per donut. -/
theorem pages_per_donut (pages_written : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ)
  (h1 : pages_written = 12)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories = 900) :
  pages_written / (total_calories / calories_per_donut) = 2 :=
by sorry

end NUMINAMATH_CALUDE_pages_per_donut_l1070_107053


namespace NUMINAMATH_CALUDE_book_cost_price_l1070_107068

theorem book_cost_price (cost_price : ℝ) : cost_price = 2200 :=
  let selling_price_10_percent := 1.10 * cost_price
  let selling_price_15_percent := 1.15 * cost_price
  have h1 : selling_price_15_percent - selling_price_10_percent = 110 := by sorry
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l1070_107068


namespace NUMINAMATH_CALUDE_magazine_cover_theorem_l1070_107054

theorem magazine_cover_theorem (n : ℕ) (S : ℝ) (h1 : n = 15) (h2 : S > 0) :
  ∃ (remaining_area : ℝ), remaining_area ≥ (8 / 15) * S ∧
  ∃ (remaining_magazines : ℕ), remaining_magazines = n - 7 :=
by
  sorry

end NUMINAMATH_CALUDE_magazine_cover_theorem_l1070_107054


namespace NUMINAMATH_CALUDE_average_age_union_l1070_107026

-- Define the student groups and their properties
def StudentGroup := Type
variables (A B C : StudentGroup)

-- Define the number of students in each group
variables (a b c : ℕ)

-- Define the sum of ages in each group
variables (sumA sumB sumC : ℕ)

-- Define the average age function
def avgAge (sum : ℕ) (count : ℕ) : ℚ := sum / count

-- State the theorem
theorem average_age_union (h_disjoint : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_avgA : avgAge sumA a = 34)
  (h_avgB : avgAge sumB b = 25)
  (h_avgC : avgAge sumC c = 45)
  (h_avgAB : avgAge (sumA + sumB) (a + b) = 30)
  (h_avgAC : avgAge (sumA + sumC) (a + c) = 42)
  (h_avgBC : avgAge (sumB + sumC) (b + c) = 36) :
  avgAge (sumA + sumB + sumC) (a + b + c) = 33 := by
  sorry


end NUMINAMATH_CALUDE_average_age_union_l1070_107026


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1070_107028

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1070_107028


namespace NUMINAMATH_CALUDE_workshop_workers_count_l1070_107013

/-- Proves that the total number of workers in a workshop is 30, given specific salary conditions. -/
theorem workshop_workers_count :
  let avg_salary : ℕ := 8000
  let technician_count : ℕ := 10
  let technician_avg_salary : ℕ := 12000
  let non_technician_avg_salary : ℕ := 6000
  ∃ (total_workers : ℕ),
    (total_workers * avg_salary = 
      technician_count * technician_avg_salary + 
      (total_workers - technician_count) * non_technician_avg_salary) ∧
    total_workers = 30 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l1070_107013


namespace NUMINAMATH_CALUDE_line_intersects_plane_implies_skew_line_exists_l1070_107015

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate to check if a line intersects a plane -/
def intersects (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Predicate to check if a line is within a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Predicate to check if two lines are skew -/
def skew (l1 l2 : Line3D) : Prop := sorry

/-- Main theorem -/
theorem line_intersects_plane_implies_skew_line_exists (l : Line3D) (α : Plane3D) :
  intersects l α → ∃ m : Line3D, line_in_plane m α ∧ skew l m := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_plane_implies_skew_line_exists_l1070_107015


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1070_107052

theorem complex_expression_equality : 
  (Real.pi - 3.14) ^ 0 + |-Real.sqrt 3| - (1/2)⁻¹ - Real.sin (π/3) = -1 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1070_107052


namespace NUMINAMATH_CALUDE_negative_root_iff_negative_a_l1070_107093

theorem negative_root_iff_negative_a (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 1 = 0) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_root_iff_negative_a_l1070_107093


namespace NUMINAMATH_CALUDE_min_both_beethoven_chopin_l1070_107033

theorem min_both_beethoven_chopin 
  (total : ℕ) 
  (beethoven_fans : ℕ) 
  (chopin_fans : ℕ) 
  (h1 : total = 150) 
  (h2 : beethoven_fans = 120) 
  (h3 : chopin_fans = 95) :
  (beethoven_fans + chopin_fans - total : ℤ).natAbs ≥ 65 :=
by sorry

end NUMINAMATH_CALUDE_min_both_beethoven_chopin_l1070_107033


namespace NUMINAMATH_CALUDE_lcm_factor_is_one_l1070_107083

/-- Given two positive integers with specific properties, prove that a certain factor of their LCM is 1. -/
theorem lcm_factor_is_one (A B : ℕ+) (X : ℕ) 
  (hcf : Nat.gcd A B = 10)
  (a_val : A = 150)
  (lcm_fact : Nat.lcm A B = 10 * X * 15) : X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_is_one_l1070_107083


namespace NUMINAMATH_CALUDE_no_two_right_angles_l1070_107064

-- Define a triangle as a structure with three angles
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_is_180 : A + B + C = 180

-- Theorem: A triangle cannot have two right angles
theorem no_two_right_angles (t : Triangle) : ¬(t.A = 90 ∧ t.B = 90 ∨ t.A = 90 ∧ t.C = 90 ∨ t.B = 90 ∧ t.C = 90) := by
  sorry


end NUMINAMATH_CALUDE_no_two_right_angles_l1070_107064


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l1070_107017

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 36 * Real.pi) :
  A = Real.pi * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l1070_107017


namespace NUMINAMATH_CALUDE_quadratic_abs_inequality_l1070_107016

theorem quadratic_abs_inequality (x : ℝ) : 
  x^2 + 4*x - 96 > |x| ↔ x < -12 ∨ x > 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_abs_inequality_l1070_107016


namespace NUMINAMATH_CALUDE_g_composition_equals_71_l1070_107001

def g (n : ℤ) : ℤ :=
  if n < 5 then n^2 + 2*n - 1 else 2*n + 5

theorem g_composition_equals_71 : g (g (g 3)) = 71 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_71_l1070_107001


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1070_107049

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of an arithmetic sequence with first term a₁ and common difference d. -/
def arithmetic_sequence_term (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem arithmetic_sequence_formula (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a 1 = 1 → a 3 = -3 →
  ∀ n : ℕ, a n = -2 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1070_107049


namespace NUMINAMATH_CALUDE_stamp_cost_l1070_107076

/-- The cost of stamps problem -/
theorem stamp_cost (cost_per_stamp : ℕ) (num_stamps : ℕ) : 
  cost_per_stamp = 34 → num_stamps = 4 → cost_per_stamp * num_stamps = 136 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_l1070_107076


namespace NUMINAMATH_CALUDE_domain_of_f_sqrt_x_minus_2_l1070_107009

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-1) 0

-- State the theorem
theorem domain_of_f_sqrt_x_minus_2 :
  (∀ x ∈ domain_f_x_plus_1, f (x + 1) ∈ Set.Icc 0 1) →
  {x : ℝ | f (Real.sqrt x - 2) ∈ Set.Icc 0 1} = Set.Icc 4 9 :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_sqrt_x_minus_2_l1070_107009


namespace NUMINAMATH_CALUDE_min_p_plus_q_l1070_107086

theorem min_p_plus_q (p q : ℕ) (hp : p > 1) (hq : q > 1) 
  (h_eq : 15 * (p + 1) = 29 * (q + 1)) : 
  ∃ (p' q' : ℕ), p' > 1 ∧ q' > 1 ∧ 15 * (p' + 1) = 29 * (q' + 1) ∧ 
    p' + q' = 45 ∧ ∀ (p'' q'' : ℕ), p'' > 1 → q'' > 1 → 
      15 * (p'' + 1) = 29 * (q'' + 1) → p'' + q'' ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l1070_107086


namespace NUMINAMATH_CALUDE_product_equals_one_l1070_107003

theorem product_equals_one (a b c : ℝ) 
  (h1 : a^2 + 2 = b^4) 
  (h2 : b^2 + 2 = c^4) 
  (h3 : c^2 + 2 = a^4) : 
  (a^2 - 1) * (b^2 - 1) * (c^2 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l1070_107003


namespace NUMINAMATH_CALUDE_max_a_value_l1070_107088

/-- A function f defined on the positive reals satisfying certain properties -/
def f : ℝ → ℝ :=
  sorry

/-- The conditions on f -/
axiom f_add (x y : ℝ) : x > 0 → y > 0 → f x + f y = f (x * y)

axiom f_neg (x : ℝ) : x > 1 → f x < 0

axiom f_ineq (x y a : ℝ) : x > 0 → y > 0 → a > 0 → 
  f (Real.sqrt (x^2 + y^2)) ≤ f (a * Real.sqrt (x * y))

/-- The theorem stating the maximum value of a -/
theorem max_a_value : 
  (∀ x y : ℝ, x > 0 → y > 0 → f (Real.sqrt (x^2 + y^2)) ≤ f (a * Real.sqrt (x * y))) →
  a ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1070_107088


namespace NUMINAMATH_CALUDE_share_distribution_l1070_107044

/-- Proves that given a total amount of 945 to be distributed among A, B, and C in the ratio 2 : 3 : 4, the share of C is 420. -/
theorem share_distribution (total : ℕ) (a b c : ℕ) : 
  total = 945 →
  a + b + c = total →
  2 * b = 3 * a →
  4 * a = 3 * c →
  c = 420 := by
  sorry

end NUMINAMATH_CALUDE_share_distribution_l1070_107044


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1070_107037

theorem arithmetic_mean_problem (original_list : List ℝ) (x y z : ℝ) :
  (original_list.length = 12) →
  (original_list.sum / original_list.length = 40) →
  ((original_list.sum + x + y + z) / (original_list.length + 3) = 50) →
  (x + y = 100) →
  z = 170 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1070_107037


namespace NUMINAMATH_CALUDE_inequality_properties_l1070_107006

theorem inequality_properties (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) :
  (a * c < b * c) ∧ (a + b < b + c) ∧ (c / a > c / b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l1070_107006


namespace NUMINAMATH_CALUDE_min_elements_in_set_l1070_107066

theorem min_elements_in_set (S : Type) [Fintype S] 
  (X : Fin 100 → Set S)
  (h_nonempty : ∀ i, Set.Nonempty (X i))
  (h_distinct : ∀ i j, i ≠ j → X i ≠ X j)
  (h_disjoint : ∀ i : Fin 99, Disjoint (X i) (X (Fin.succ i)))
  (h_not_union : ∀ i : Fin 99, (X i) ∪ (X (Fin.succ i)) ≠ Set.univ) :
  Fintype.card S ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_elements_in_set_l1070_107066


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1070_107094

/-- Given that the solution set of ax² + 5x - 2 > 0 is {x | 1/2 < x < 2},
    prove that a = -2 and the solution set of ax² + 5x + a² - 1 > 0 is {x | -1/2 < x < 3} -/
theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → 
  (a = -2 ∧ ∀ x : ℝ, a*x^2 + 5*x + a^2 - 1 > 0 ↔ -1/2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1070_107094


namespace NUMINAMATH_CALUDE_problem_solution_l1070_107029

theorem problem_solution (m n : ℝ) 
  (h1 : (m * Real.exp m) / (4 * n^2) = (Real.log n + Real.log 2) / Real.exp m)
  (h2 : Real.exp (2 * m) = 1 / m) :
  (n = Real.exp m / 2) ∧ 
  (m + n < 7/5) ∧ 
  (1 < 2*n - m^2 ∧ 2*n - m^2 < 3/2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1070_107029


namespace NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l1070_107045

/-- The expected number of boy-girl adjacencies in a circular arrangement -/
theorem expected_boy_girl_adjacencies (n_boys n_girls : ℕ) (h : n_boys = 10 ∧ n_girls = 8) :
  let total := n_boys + n_girls
  let prob_boy_girl := (n_boys : ℚ) * n_girls / (total * (total - 1))
  total * (2 * prob_boy_girl) = 480 / 51 := by
  sorry

#check expected_boy_girl_adjacencies

end NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l1070_107045


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l1070_107012

/-- The function f(x) defined as ax - 1 + 3 where a > 0 and a ≠ 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 + 3

/-- Theorem stating that (0, 2) is a fixed point of f for all valid a -/
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 0 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l1070_107012


namespace NUMINAMATH_CALUDE_original_serving_size_l1070_107089

/-- Proves that the original serving size was 8 ounces -/
theorem original_serving_size (total_water : ℝ) (current_serving : ℝ) (serving_difference : ℕ) : 
  total_water = 64 →
  current_serving = 16 →
  (total_water / current_serving : ℝ) + serving_difference = total_water / 8 →
  8 = total_water / ((total_water / current_serving : ℝ) + serving_difference : ℝ) := by
sorry

end NUMINAMATH_CALUDE_original_serving_size_l1070_107089


namespace NUMINAMATH_CALUDE_car_speed_comparison_l1070_107067

theorem car_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  3 / (1/u + 1/v + 1/w) ≤ (u + v) / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l1070_107067


namespace NUMINAMATH_CALUDE_sin_translation_l1070_107000

/-- Given a function f(x) = 3sin(2x), translating its graph π/6 units to the left
    results in the function g(x) = 3sin(2x + π/3) -/
theorem sin_translation (x : ℝ) :
  (fun x => 3 * Real.sin (2 * x + π / 3)) x =
  (fun x => 3 * Real.sin (2 * (x + π / 6))) x := by
sorry

end NUMINAMATH_CALUDE_sin_translation_l1070_107000


namespace NUMINAMATH_CALUDE_evaluate_expression_l1070_107097

theorem evaluate_expression : (3^2)^2 - (2^3)^3 = -431 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1070_107097


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l1070_107004

theorem circle_radius_from_area (A : ℝ) (h : A = 64 * Real.pi) :
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l1070_107004


namespace NUMINAMATH_CALUDE_zero_point_implies_a_range_l1070_107020

/-- The function f(x) = x^2 + x - 2a has a zero point in the interval (-1, 1) if and only if a ∈ [-1/8, 1) -/
theorem zero_point_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 + x - 2*a = 0) ↔ -1/8 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_implies_a_range_l1070_107020


namespace NUMINAMATH_CALUDE_min_diagonal_rectangle_l1070_107024

/-- The minimum diagonal of a rectangle with perimeter 24 -/
theorem min_diagonal_rectangle (l w : ℝ) (h_perimeter : l + w = 12) :
  ∃ (d : ℝ), d = Real.sqrt (l^2 + w^2) ∧ 
  (∀ (l' w' : ℝ), l' + w' = 12 → Real.sqrt (l'^2 + w'^2) ≥ d) ∧
  d = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_diagonal_rectangle_l1070_107024


namespace NUMINAMATH_CALUDE_people_got_on_second_stop_is_two_l1070_107008

/-- The number of people who got on at the second stop of a bus journey -/
def people_got_on_second_stop : ℕ :=
  let initial_people : ℕ := 50
  let first_stop_off : ℕ := 15
  let second_stop_off : ℕ := 8
  let third_stop_off : ℕ := 4
  let third_stop_on : ℕ := 3
  let final_people : ℕ := 28
  initial_people - first_stop_off - second_stop_off + 
    (final_people - (initial_people - first_stop_off - second_stop_off - third_stop_off + third_stop_on))

theorem people_got_on_second_stop_is_two : 
  people_got_on_second_stop = 2 := by sorry

end NUMINAMATH_CALUDE_people_got_on_second_stop_is_two_l1070_107008


namespace NUMINAMATH_CALUDE_average_rainfall_leap_year_february_l1070_107082

/-- Calculates the average rainfall per hour in February of a leap year -/
theorem average_rainfall_leap_year_february (total_rainfall : ℝ) :
  total_rainfall = 420 →
  (35 : ℝ) / 58 = total_rainfall / (29 * 24) := by
  sorry

end NUMINAMATH_CALUDE_average_rainfall_leap_year_february_l1070_107082


namespace NUMINAMATH_CALUDE_farmer_bean_seedlings_l1070_107078

/-- Represents the farmer's planting scenario -/
structure FarmPlanting where
  bean_seedlings_per_row : ℕ
  pumpkin_seeds : ℕ
  pumpkin_seeds_per_row : ℕ
  radishes : ℕ
  radishes_per_row : ℕ
  rows_per_bed : ℕ
  total_beds : ℕ

/-- Calculates the total number of bean seedlings -/
def total_bean_seedlings (f : FarmPlanting) : ℕ :=
  let total_rows := f.total_beds * f.rows_per_bed
  let pumpkin_rows := f.pumpkin_seeds / f.pumpkin_seeds_per_row
  let radish_rows := f.radishes / f.radishes_per_row
  let bean_rows := total_rows - pumpkin_rows - radish_rows
  bean_rows * f.bean_seedlings_per_row

/-- Theorem stating that the farmer has 64 bean seedlings -/
theorem farmer_bean_seedlings :
  ∀ (f : FarmPlanting),
  f.bean_seedlings_per_row = 8 →
  f.pumpkin_seeds = 84 →
  f.pumpkin_seeds_per_row = 7 →
  f.radishes = 48 →
  f.radishes_per_row = 6 →
  f.rows_per_bed = 2 →
  f.total_beds = 14 →
  total_bean_seedlings f = 64 := by
  sorry

end NUMINAMATH_CALUDE_farmer_bean_seedlings_l1070_107078


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l1070_107081

/-- Tim's daily task count -/
def daily_tasks : ℕ := 100

/-- Tim's working days per week -/
def working_days : ℕ := 6

/-- Number of tasks paying $1.2 each -/
def tasks_1_2 : ℕ := 40

/-- Number of tasks paying $1.5 each -/
def tasks_1_5 : ℕ := 30

/-- Number of tasks paying $2 each -/
def tasks_2 : ℕ := 30

/-- Payment rate for the first group of tasks -/
def rate_1_2 : ℚ := 1.2

/-- Payment rate for the second group of tasks -/
def rate_1_5 : ℚ := 1.5

/-- Payment rate for the third group of tasks -/
def rate_2 : ℚ := 2

/-- Tim's weekly earnings -/
def weekly_earnings : ℚ := 918

theorem tim_weekly_earnings :
  daily_tasks = tasks_1_2 + tasks_1_5 + tasks_2 →
  working_days * (tasks_1_2 * rate_1_2 + tasks_1_5 * rate_1_5 + tasks_2 * rate_2) = weekly_earnings :=
by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l1070_107081


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l1070_107031

theorem inverse_proportion_k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x, x ≠ 0 → (k / x) = -1 ↔ x = 2) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l1070_107031


namespace NUMINAMATH_CALUDE_incircle_radius_not_less_than_one_l1070_107062

/-- Triangle ABC with sides a, b, c and incircle radius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ

/-- The theorem stating that the incircle radius of triangle ABC with BC = 3 and AC = 4 is not less than 1 -/
theorem incircle_radius_not_less_than_one (t : Triangle) (h1 : t.b = 3) (h2 : t.c = 4) : 
  t.r ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_incircle_radius_not_less_than_one_l1070_107062


namespace NUMINAMATH_CALUDE_cos_240_degrees_l1070_107092

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1070_107092


namespace NUMINAMATH_CALUDE_sequence_divisibility_l1070_107073

def sequence_a : ℕ → ℤ
  | 0 => 1
  | n + 1 => (sequence_a n)^2 + sequence_a n + 1

theorem sequence_divisibility (n : ℕ) : 
  (n ≥ 1) → (sequence_a n)^2 + 1 ∣ (sequence_a (n + 1))^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l1070_107073


namespace NUMINAMATH_CALUDE_mangoes_kelly_can_buy_l1070_107042

def mangoes_cost_per_half_pound : ℝ := 0.60
def kelly_budget : ℝ := 12

theorem mangoes_kelly_can_buy :
  let cost_per_pound : ℝ := 2 * mangoes_cost_per_half_pound
  let pounds_kelly_can_buy : ℝ := kelly_budget / cost_per_pound
  pounds_kelly_can_buy = 10 := by sorry

end NUMINAMATH_CALUDE_mangoes_kelly_can_buy_l1070_107042


namespace NUMINAMATH_CALUDE_cake_eaters_l1070_107038

theorem cake_eaters (n : ℕ) (h1 : n > 0) : 
  (∃ (portions : Fin n → ℚ), 
    (∀ i, portions i > 0) ∧ 
    (∃ i, portions i = 1/11) ∧ 
    (∃ i, portions i = 1/14) ∧ 
    (∀ i, portions i ≤ 1/11) ∧ 
    (∀ i, portions i ≥ 1/14) ∧ 
    (Finset.sum Finset.univ portions = 1)) ↔ 
  (n = 12 ∨ n = 13) :=
sorry

end NUMINAMATH_CALUDE_cake_eaters_l1070_107038


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l1070_107011

theorem binomial_coefficient_equation_solution (x : ℕ) : 
  (Nat.choose 28 (3*x) = Nat.choose 28 (x + 8)) ↔ (x = 4 ∨ x = 5) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l1070_107011


namespace NUMINAMATH_CALUDE_sandra_betty_orange_ratio_l1070_107048

theorem sandra_betty_orange_ratio :
  ∀ (emily sandra betty : ℕ),
    emily = 7 * sandra →
    betty = 12 →
    emily = 252 →
    sandra / betty = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sandra_betty_orange_ratio_l1070_107048


namespace NUMINAMATH_CALUDE_pencil_price_theorem_l1070_107058

/-- The price of a pencil in won -/
def pencil_price : ℚ := 5000 + 20

/-- The conversion factor from won to 10,000 won units -/
def conversion_factor : ℚ := 10000

/-- The price of the pencil in units of 10,000 won -/
def pencil_price_in_units : ℚ := pencil_price / conversion_factor

theorem pencil_price_theorem : pencil_price_in_units = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_theorem_l1070_107058


namespace NUMINAMATH_CALUDE_no_rearranged_power_of_two_l1070_107057

/-- Checks if all digits of a natural number are non-zero -/
def allDigitsNonZero (n : ℕ) : Prop := sorry

/-- Checks if two natural numbers have the same digits (possibly in different order) -/
def sameDigits (m n : ℕ) : Prop := sorry

/-- There do not exist two distinct powers of 2 with all non-zero digits that are rearrangements of each other -/
theorem no_rearranged_power_of_two : ¬∃ (a b : ℕ), a ≠ b ∧ 
  allDigitsNonZero (2^a) ∧ 
  allDigitsNonZero (2^b) ∧ 
  sameDigits (2^a) (2^b) := by
  sorry

end NUMINAMATH_CALUDE_no_rearranged_power_of_two_l1070_107057
