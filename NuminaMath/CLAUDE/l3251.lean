import Mathlib

namespace NUMINAMATH_CALUDE_chimney_bricks_proof_l3251_325176

/-- The number of hours it takes Brenda to build the chimney alone -/
def brenda_time : ℝ := 8

/-- The number of hours it takes Brandon to build the chimney alone -/
def brandon_time : ℝ := 12

/-- The decrease in combined output when working together (in bricks per hour) -/
def output_decrease : ℝ := 15

/-- The number of hours it takes Brenda and Brandon to build the chimney together -/
def combined_time : ℝ := 6

/-- The number of bricks in the chimney -/
def chimney_bricks : ℝ := 360

theorem chimney_bricks_proof : 
  combined_time * ((chimney_bricks / brenda_time + chimney_bricks / brandon_time) - output_decrease) = chimney_bricks := by
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_proof_l3251_325176


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_two_variables_l3251_325135

theorem arithmetic_geometric_mean_two_variables (a b : ℝ) : (a^2 + b^2) / 2 ≥ a * b := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_two_variables_l3251_325135


namespace NUMINAMATH_CALUDE_packaging_cost_per_cake_l3251_325196

/-- Proves that the cost of packaging per cake is $1 -/
theorem packaging_cost_per_cake
  (ingredient_cost_two_cakes : ℝ)
  (selling_price_per_cake : ℝ)
  (profit_per_cake : ℝ)
  (h1 : ingredient_cost_two_cakes = 12)
  (h2 : selling_price_per_cake = 15)
  (h3 : profit_per_cake = 8) :
  selling_price_per_cake - profit_per_cake - (ingredient_cost_two_cakes / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_packaging_cost_per_cake_l3251_325196


namespace NUMINAMATH_CALUDE_alice_bob_meet_l3251_325111

/-- The number of points on the circular path -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 11

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 9

/-- Function to calculate the position after a certain number of moves -/
def position_after_moves (start : ℕ) (move : ℕ) (turns : ℕ) : ℕ :=
  (start + move * turns - 1) % n + 1

theorem alice_bob_meet :
  position_after_moves n alice_move meeting_turns =
  position_after_moves n (n - bob_move) meeting_turns :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l3251_325111


namespace NUMINAMATH_CALUDE_minimum_candies_in_can_l3251_325144

theorem minimum_candies_in_can (red green : ℕ) : 
  (red > 0) →
  (green > 0) →
  ((3 * red) / 5 : ℚ) = (3 / 8) * ((3 * red) / 5 + (2 * green) / 5) →
  (∀ r g : ℕ, r > 0 ∧ g > 0 ∧ ((3 * r) / 5 : ℚ) = (3 / 8) * ((3 * r) / 5 + (2 * g) / 5) → r + g ≥ red + green) →
  red + green = 35 :=
by sorry

end NUMINAMATH_CALUDE_minimum_candies_in_can_l3251_325144


namespace NUMINAMATH_CALUDE_percentage_passed_both_l3251_325125

theorem percentage_passed_both (failed_hindi failed_english failed_both : ℚ) 
  (h1 : failed_hindi = 35 / 100)
  (h2 : failed_english = 45 / 100)
  (h3 : failed_both = 20 / 100) :
  1 - (failed_hindi + failed_english - failed_both) = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_both_l3251_325125


namespace NUMINAMATH_CALUDE_fourth_power_of_square_of_fourth_smallest_prime_l3251_325166

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem fourth_power_of_square_of_fourth_smallest_prime :
  (nthSmallestPrime 4)^2^4 = 5764801 := by sorry

end NUMINAMATH_CALUDE_fourth_power_of_square_of_fourth_smallest_prime_l3251_325166


namespace NUMINAMATH_CALUDE_special_square_numbers_l3251_325130

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A function that returns the first two digits of a six-digit number -/
def first_two_digits (n : ℕ) : ℕ :=
  n / 10000

/-- A function that returns the middle two digits of a six-digit number -/
def middle_two_digits (n : ℕ) : ℕ :=
  (n / 100) % 100

/-- A function that returns the last two digits of a six-digit number -/
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

/-- A function that checks if all digits of a six-digit number are non-zero -/
def all_digits_nonzero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [n / 100000, (n / 10000) % 10, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10] → d ≠ 0

/-- The main theorem stating that there are exactly 2 special square numbers -/
theorem special_square_numbers :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 100000 ≤ n ∧ n < 1000000 ∧
              all_digits_nonzero n ∧
              is_perfect_square n ∧
              is_perfect_square (first_two_digits n) ∧
              is_perfect_square (middle_two_digits n) ∧
              is_perfect_square (last_two_digits n)) ∧
    s.card = 2 := by
  sorry


end NUMINAMATH_CALUDE_special_square_numbers_l3251_325130


namespace NUMINAMATH_CALUDE_max_fly_path_2x1x1_box_l3251_325146

/-- Represents a rectangular box with dimensions a, b, and c -/
structure Box where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the maximum path length for a fly in the given box -/
def maxFlyPathLength (box : Box) : ℝ :=
  sorry

/-- Theorem stating the maximum fly path length for a 2x1x1 box -/
theorem max_fly_path_2x1x1_box :
  let box : Box := { a := 2, b := 1, c := 1 }
  maxFlyPathLength box = 4 + 4 * Real.sqrt 5 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_fly_path_2x1x1_box_l3251_325146


namespace NUMINAMATH_CALUDE_some_T_divisible_by_3_l3251_325106

def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

theorem some_T_divisible_by_3 : ∃ x ∈ T, 3 ∣ x := by
  sorry

end NUMINAMATH_CALUDE_some_T_divisible_by_3_l3251_325106


namespace NUMINAMATH_CALUDE_max_value_of_2sinx_l3251_325171

theorem max_value_of_2sinx :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), 2 * Real.sin x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2sinx_l3251_325171


namespace NUMINAMATH_CALUDE_lowest_cost_scheme_l3251_325133

-- Define excavator types
inductive ExcavatorType
| A
| B

-- Define the excavation capacity for each type
def excavation_capacity (t : ExcavatorType) : ℝ :=
  match t with
  | ExcavatorType.A => 30
  | ExcavatorType.B => 15

-- Define the hourly cost for each type
def hourly_cost (t : ExcavatorType) : ℝ :=
  match t with
  | ExcavatorType.A => 300
  | ExcavatorType.B => 180

-- Define the total excavation function
def total_excavation (a b : ℕ) : ℝ :=
  4 * (a * excavation_capacity ExcavatorType.A + b * excavation_capacity ExcavatorType.B)

-- Define the total cost function
def total_cost (a b : ℕ) : ℝ :=
  4 * (a * hourly_cost ExcavatorType.A + b * hourly_cost ExcavatorType.B)

-- Theorem statement
theorem lowest_cost_scheme :
  ∀ a b : ℕ,
    a + b = 12 →
    total_excavation a b ≥ 1080 →
    total_cost a b ≤ 12960 →
    total_cost 7 5 ≤ total_cost a b ∧
    total_cost 7 5 = 12000 :=
sorry

end NUMINAMATH_CALUDE_lowest_cost_scheme_l3251_325133


namespace NUMINAMATH_CALUDE_valid_drawings_for_ten_balls_l3251_325172

/-- The number of ways to draw balls from a box -/
def validDrawings (n k : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose n (k + 1)

/-- Theorem stating the number of valid ways to draw balls -/
theorem valid_drawings_for_ten_balls :
  validDrawings 10 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_valid_drawings_for_ten_balls_l3251_325172


namespace NUMINAMATH_CALUDE_M_geq_N_l3251_325177

theorem M_geq_N (a : ℝ) : 2 * a * (a - 2) + 3 ≥ (a - 1) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_M_geq_N_l3251_325177


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l3251_325199

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
def CyclicQuadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

/-- The distance between two points in a 2D plane. -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The area of a quadrilateral given its four vertices. -/
def quadrilateralArea (A B C D : ℝ × ℝ) : ℝ := sorry

theorem cyclic_quadrilateral_area 
  (A B C D : ℝ × ℝ) 
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_AB : distance A B = 2)
  (h_BC : distance B C = 6)
  (h_CD : distance C D = 4)
  (h_DA : distance D A = 4) :
  quadrilateralArea A B C D = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l3251_325199


namespace NUMINAMATH_CALUDE_power_product_simplification_l3251_325198

theorem power_product_simplification (a : ℝ) : (36 * a^9)^4 * (63 * a^9)^4 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l3251_325198


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3251_325131

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ, 
  (x₁^2 + 2019*x₁ + 1 = 0) → 
  (x₂^2 + 2019*x₂ + 1 = 0) → 
  (x₁ ≠ x₂) →
  (x₁*x₂ - x₁ - x₂ = 2020) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3251_325131


namespace NUMINAMATH_CALUDE_no_m_exists_for_equality_m_range_for_subset_l3251_325137

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem 1: No m exists such that P = S(m)
theorem no_m_exists_for_equality : ¬ ∃ m : ℝ, P = S m := by sorry

-- Theorem 2: The set of m such that P ⊆ S(m) is {m | m ≤ 3}
theorem m_range_for_subset : {m : ℝ | P ⊆ S m} = {m : ℝ | m ≤ 3} := by sorry

end NUMINAMATH_CALUDE_no_m_exists_for_equality_m_range_for_subset_l3251_325137


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l3251_325164

theorem unique_number_with_properties : ∃! n : ℕ, 
  50 < n ∧ n < 70 ∧ 
  ∃ k : ℤ, n - 3 = 5 * k ∧
  ∃ l : ℤ, n - 2 = 7 * l :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l3251_325164


namespace NUMINAMATH_CALUDE_horse_race_theorem_l3251_325119

def horse_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_valid_subset (s : List Nat) : Prop :=
  s.length = 5 ∧ s.toFinset ⊆ horse_primes.toFinset

def least_common_time (s : List Nat) : Nat :=
  s.foldl Nat.lcm 1

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem horse_race_theorem :
  ∃ (s : List Nat), is_valid_subset s ∧
    (∀ (t : List Nat), is_valid_subset t → least_common_time s ≤ least_common_time t) ∧
    least_common_time s = 2310 ∧
    sum_of_digits (least_common_time s) = 6 :=
  sorry

end NUMINAMATH_CALUDE_horse_race_theorem_l3251_325119


namespace NUMINAMATH_CALUDE_original_ribbon_length_is_correct_l3251_325128

/-- The length of ribbon tape used for one gift in meters -/
def ribbon_per_gift : ℝ := 0.84

/-- The number of gifts prepared -/
def num_gifts : ℕ := 10

/-- The length of leftover ribbon tape in meters -/
def leftover_ribbon : ℝ := 0.5

/-- The total length of the original ribbon tape in meters -/
def original_ribbon_length : ℝ := ribbon_per_gift * num_gifts + leftover_ribbon

theorem original_ribbon_length_is_correct :
  original_ribbon_length = 8.9 := by sorry

end NUMINAMATH_CALUDE_original_ribbon_length_is_correct_l3251_325128


namespace NUMINAMATH_CALUDE_second_employee_hourly_rate_l3251_325117

/-- Proves that the hourly rate of the second employee before subsidy is $22 -/
theorem second_employee_hourly_rate 
  (first_employee_rate : ℝ)
  (subsidy : ℝ)
  (weekly_savings : ℝ)
  (hours_per_week : ℝ)
  (h1 : first_employee_rate = 20)
  (h2 : subsidy = 6)
  (h3 : weekly_savings = 160)
  (h4 : hours_per_week = 40)
  : ∃ (second_employee_rate : ℝ), 
    hours_per_week * first_employee_rate - hours_per_week * (second_employee_rate - subsidy) = weekly_savings ∧ 
    second_employee_rate = 22 :=
by sorry

end NUMINAMATH_CALUDE_second_employee_hourly_rate_l3251_325117


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_negative_one_l3251_325179

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)

-- State the theorem
theorem even_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_negative_one_l3251_325179


namespace NUMINAMATH_CALUDE_nines_in_hundred_l3251_325103

def count_nines (n : ℕ) : ℕ :=
  (n / 10) + (n / 10)

theorem nines_in_hundred : count_nines 100 = 20 := by sorry

end NUMINAMATH_CALUDE_nines_in_hundred_l3251_325103


namespace NUMINAMATH_CALUDE_divisible_by_three_l3251_325141

theorem divisible_by_three (a b : ℕ) (h : 3 ∣ (a * b)) : 3 ∣ a ∨ 3 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l3251_325141


namespace NUMINAMATH_CALUDE_range_of_m_l3251_325115

-- Define the sets P and M
def P : Set ℝ := {x | x^2 ≤ 4}
def M (m : ℝ) : Set ℝ := {m}

-- State the theorem
theorem range_of_m (m : ℝ) : (P ∩ M m = M m) → m ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3251_325115


namespace NUMINAMATH_CALUDE_paved_road_time_l3251_325183

/-- Calculates the time spent on a paved road given total trip distance,
    dirt road travel time and speed, and speed difference between paved and dirt roads. -/
theorem paved_road_time (total_distance : ℝ) (dirt_time : ℝ) (dirt_speed : ℝ) (speed_diff : ℝ) :
  total_distance = 200 →
  dirt_time = 3 →
  dirt_speed = 32 →
  speed_diff = 20 →
  (total_distance - dirt_time * dirt_speed) / (dirt_speed + speed_diff) = 2 := by
  sorry

#check paved_road_time

end NUMINAMATH_CALUDE_paved_road_time_l3251_325183


namespace NUMINAMATH_CALUDE_prince_cd_spend_l3251_325116

/-- Calculates the amount spent on CDs given the total number of CDs,
    percentage of expensive CDs, prices, and buying pattern. -/
def calculate_cd_spend (total_cds : ℕ) (expensive_percentage : ℚ) 
                       (expensive_price cheap_price : ℚ) 
                       (expensive_bought_ratio : ℚ) : ℚ :=
  let expensive_cds : ℚ := expensive_percentage * total_cds
  let cheap_cds : ℚ := (1 - expensive_percentage) * total_cds
  let expensive_bought : ℚ := expensive_bought_ratio * expensive_cds
  expensive_bought * expensive_price + cheap_cds * cheap_price

/-- Proves that Prince spent $1000 on CDs given the problem conditions. -/
theorem prince_cd_spend : 
  calculate_cd_spend 200 (40/100) 10 5 (1/2) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_prince_cd_spend_l3251_325116


namespace NUMINAMATH_CALUDE_applicant_overall_score_l3251_325191

/-- Calculates the overall score given written test and interview scores and their weights -/
def overall_score (written_score interview_score : ℝ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

/-- Theorem stating that the overall score is 72 points given the specific scores and weights -/
theorem applicant_overall_score :
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  overall_score written_score interview_score written_weight interview_weight = 72 := by
  sorry

#eval overall_score 80 60 0.6 0.4

end NUMINAMATH_CALUDE_applicant_overall_score_l3251_325191


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l3251_325170

theorem set_membership_implies_value (a : ℝ) : 
  3 ∈ ({1, a, a - 2} : Set ℝ) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l3251_325170


namespace NUMINAMATH_CALUDE_complex_value_theorem_l3251_325110

theorem complex_value_theorem (z : ℂ) (h : (1 - z) / (1 + z) = I) : 
  Complex.abs (z + 1) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_value_theorem_l3251_325110


namespace NUMINAMATH_CALUDE_final_pen_count_l3251_325163

def pen_collection (initial_pens : ℕ) (mike_gives : ℕ) (sharon_takes : ℕ) : ℕ :=
  let after_mike := initial_pens + mike_gives
  let after_cindy := 2 * after_mike
  after_cindy - sharon_takes

theorem final_pen_count :
  pen_collection 7 22 19 = 39 := by
  sorry

end NUMINAMATH_CALUDE_final_pen_count_l3251_325163


namespace NUMINAMATH_CALUDE_A_profit_share_l3251_325153

-- Define the capital shares of partners A, B, C, and D
def share_A : ℚ := 1/3
def share_B : ℚ := 1/4
def share_C : ℚ := 1/5
def share_D : ℚ := 1 - (share_A + share_B + share_C)

-- Define the total profit
def total_profit : ℕ := 2445

-- Theorem statement
theorem A_profit_share :
  (share_A * total_profit : ℚ) = 815 := by
  sorry

end NUMINAMATH_CALUDE_A_profit_share_l3251_325153


namespace NUMINAMATH_CALUDE_choose_3_from_10_l3251_325192

theorem choose_3_from_10 : (Nat.choose 10 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_3_from_10_l3251_325192


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3251_325139

/-- The trajectory of the midpoint of chords passing through the origin of a circle --/
theorem midpoint_trajectory (x y : ℝ) :
  (0 < x) → (x ≤ 1) →
  (∃ (a b : ℝ), (a - 1)^2 + b^2 = 1 ∧ (x = a/2) ∧ (y = b/2)) →
  (x - 1/2)^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3251_325139


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_problem_l3251_325148

/-- The 'Crazy Silly School' series problem -/
theorem crazy_silly_school_series_problem 
  (total_books : ℕ) 
  (total_movies : ℕ) 
  (books_read : ℕ) 
  (movies_watched : ℕ) 
  (h1 : total_books = 25) 
  (h2 : total_movies = 35) 
  (h3 : books_read = 15) 
  (h4 : movies_watched = 29) :
  movies_watched - books_read = 14 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_problem_l3251_325148


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_l3251_325143

theorem quadratic_inequality_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - x + m > 0) → m > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_l3251_325143


namespace NUMINAMATH_CALUDE_line_transformation_l3251_325122

/-- Given a line ax + y - 7 = 0 transformed by matrix A to 9x + y - 91 = 0, prove a = 2 and b = 13 -/
theorem line_transformation (a b : ℝ) :
  (∀ x y : ℝ, a * x + y - 7 = 0 →
    ∃ x' y' : ℝ, x' = 3 * x ∧ y' = -x + b * y ∧ 9 * x' + y' - 91 = 0) →
  a = 2 ∧ b = 13 := by
  sorry


end NUMINAMATH_CALUDE_line_transformation_l3251_325122


namespace NUMINAMATH_CALUDE_cos_sum_thirteenth_l3251_325102

theorem cos_sum_thirteenth : 
  Real.cos (3 * Real.pi / 13) + Real.cos (5 * Real.pi / 13) + Real.cos (7 * Real.pi / 13) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_thirteenth_l3251_325102


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3251_325101

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, |x + a| < b ↔ 2 < x ∧ x < 4) → a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3251_325101


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l3251_325194

/-- The number of ways to place n different balls into m different boxes --/
def placeWays (n m : ℕ) : ℕ := sorry

/-- The number of ways to place n different balls into m different boxes, leaving k boxes empty --/
def placeWaysWithEmpty (n m k : ℕ) : ℕ := sorry

theorem ball_placement_theorem :
  (placeWaysWithEmpty 4 4 1 = 144) ∧ (placeWaysWithEmpty 4 4 2 = 84) := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l3251_325194


namespace NUMINAMATH_CALUDE_factor_expression_l3251_325173

theorem factor_expression (x : ℝ) : 75*x + 45 = 15*(5*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3251_325173


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l3251_325147

theorem binomial_expansion_properties :
  let n : ℕ := 15
  let last_three_sum := (n.choose (n-2)) + (n.choose (n-1)) + (n.choose n)
  let term (r : ℕ) := (n.choose r) * (3^r)
  ∃ (r₁ r₂ : ℕ),
    (last_three_sum = 121) ∧
    (∀ k, 0 ≤ k ∧ k ≤ n → (n.choose k) ≤ (n.choose r₁) ∧ (n.choose k) ≤ (n.choose r₂)) ∧
    (∀ k, 0 ≤ k ∧ k ≤ n → term k ≤ term r₁ ∧ term k ≤ term r₂) ∧
    r₁ = 11 ∧ r₂ = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l3251_325147


namespace NUMINAMATH_CALUDE_orange_distribution_l3251_325100

theorem orange_distribution (total_oranges : ℕ) (bad_oranges : ℕ) (num_students : ℕ) 
    (h1 : total_oranges = 108)
    (h2 : bad_oranges = 36)
    (h3 : num_students = 12)
    (h4 : bad_oranges < total_oranges) :
  (total_oranges / num_students) - ((total_oranges - bad_oranges) / num_students) = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l3251_325100


namespace NUMINAMATH_CALUDE_eggs_scrambled_l3251_325152

-- Define the parameters
def total_time : ℕ := 39
def time_per_sausage : ℕ := 5
def num_sausages : ℕ := 3
def time_per_egg : ℕ := 4

-- Define the theorem
theorem eggs_scrambled :
  ∃ (num_eggs : ℕ),
    num_eggs * time_per_egg = total_time - (num_sausages * time_per_sausage) ∧
    num_eggs = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_scrambled_l3251_325152


namespace NUMINAMATH_CALUDE_michael_fish_count_l3251_325123

theorem michael_fish_count (initial_fish : Float) (given_fish : Float) : 
  initial_fish = 49.0 → given_fish = 18.0 → initial_fish + given_fish = 67.0 := by
  sorry

end NUMINAMATH_CALUDE_michael_fish_count_l3251_325123


namespace NUMINAMATH_CALUDE_sum_of_squares_l3251_325129

theorem sum_of_squares (x y z : ℝ) :
  (x + y + z) / 3 = 10 →
  (x * y * z) ^ (1/3 : ℝ) = 7 →
  3 / (1/x + 1/y + 1/z) = 4 →
  x^2 + y^2 + z^2 = 385.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3251_325129


namespace NUMINAMATH_CALUDE_total_wheat_weight_l3251_325114

def wheat_weights : List ℝ := [91, 91, 91.5, 89, 91.2, 91.3, 88.7, 88.8, 91.8, 91.1]
def standard_weight : ℝ := 90

theorem total_wheat_weight :
  (wheat_weights.sum) = 905.4 := by
  sorry

end NUMINAMATH_CALUDE_total_wheat_weight_l3251_325114


namespace NUMINAMATH_CALUDE_perpendicular_line_unique_l3251_325150

-- Define a line by its coefficients (a, b, c) in the equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Line.throughPoint (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_unique :
  ∃! l : Line, l.throughPoint (3, 0) ∧
                l.perpendicular { a := 2, b := 1, c := -5 } ∧
                l = { a := 1, b := -2, c := -3 } := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_unique_l3251_325150


namespace NUMINAMATH_CALUDE_simplified_expression_constant_expression_l3251_325132

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 + 4 * x * y - 2 * x - 3
def B (x y : ℝ) : ℝ := -x^2 + x * y + 2

-- Theorem 1: Prove the simplified expression for 3A - 2(A + 2B)
theorem simplified_expression (x y : ℝ) :
  3 * A x y - 2 * (A x y + 2 * B x y) = 6 * x^2 - 2 * x - 11 := by sorry

-- Theorem 2: Prove the value of y when B + (1/2)A is constant for any x
theorem constant_expression (y : ℝ) :
  (∀ x : ℝ, ∃ c : ℝ, B x y + (1/2) * A x y = c) ↔ y = 1/3 := by sorry

end NUMINAMATH_CALUDE_simplified_expression_constant_expression_l3251_325132


namespace NUMINAMATH_CALUDE_range_of_a_l3251_325184

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 * Real.exp (y/x) - a * y^3 = 0) : a ≥ Real.exp 3 / 27 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3251_325184


namespace NUMINAMATH_CALUDE_f_composition_at_one_over_e_l3251_325142

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

-- State the theorem
theorem f_composition_at_one_over_e :
  f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_one_over_e_l3251_325142


namespace NUMINAMATH_CALUDE_trig_identity_l3251_325195

theorem trig_identity (x z : ℝ) : 
  (Real.sin x)^2 + (Real.sin (x + z))^2 - 2 * (Real.sin x) * (Real.sin z) * (Real.sin (x + z)) = (Real.sin z)^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3251_325195


namespace NUMINAMATH_CALUDE_total_oil_leaked_equals_11687_l3251_325120

/-- The amount of oil leaked before repairs, in liters -/
def oil_leaked_before : ℕ := 6522

/-- The amount of oil leaked during repairs, in liters -/
def oil_leaked_during : ℕ := 5165

/-- The total amount of oil leaked, in liters -/
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem total_oil_leaked_equals_11687 : total_oil_leaked = 11687 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_leaked_equals_11687_l3251_325120


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l3251_325104

theorem mod_congruence_unique_solution : ∃! n : ℕ, n ≤ 19 ∧ n ≡ -5678 [ZMOD 20] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l3251_325104


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3251_325174

/-- For a geometric sequence with positive terms and common ratio q where q^2 = 4,
    the ratio (a_3 + a_4) / (a_4 + a_5) equals 1/2. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- common ratio is q
  q^2 = 4 →
  (a 3 + a 4) / (a 4 + a 5) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3251_325174


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3251_325185

theorem sin_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3251_325185


namespace NUMINAMATH_CALUDE_point_coordinates_in_fourth_quadrant_l3251_325109

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the fourth quadrant
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

-- Define distance to x-axis
def distance_to_x_axis (p : Point2D) : ℝ :=
  |p.y|

-- Define distance to y-axis
def distance_to_y_axis (p : Point2D) : ℝ :=
  |p.x|

-- Theorem statement
theorem point_coordinates_in_fourth_quadrant (p : Point2D) 
  (h1 : in_fourth_quadrant p)
  (h2 : distance_to_x_axis p = 3)
  (h3 : distance_to_y_axis p = 8) :
  p = Point2D.mk 8 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_in_fourth_quadrant_l3251_325109


namespace NUMINAMATH_CALUDE_alice_paid_percentage_l3251_325157

def suggested_retail_price : ℝ → ℝ := id

def marked_price (P : ℝ) : ℝ := 0.6 * P

def alice_paid (P : ℝ) : ℝ := 0.4 * marked_price P

theorem alice_paid_percentage (P : ℝ) (h : P > 0) :
  alice_paid P / suggested_retail_price P = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_alice_paid_percentage_l3251_325157


namespace NUMINAMATH_CALUDE_parabola_translation_l3251_325197

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ := (x - 1)^2 - 4

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x - 4)^2 - 2

-- Define the translation
def translation_right : ℝ := 3
def translation_up : ℝ := 2

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = initial_parabola (x - translation_right) + translation_up :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3251_325197


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l3251_325175

theorem scientific_notation_proof : 
  (55000000 : ℝ) = 5.5 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l3251_325175


namespace NUMINAMATH_CALUDE_theta_value_l3251_325149

theorem theta_value (θ : Real)
  (h1 : 3 * Real.pi ≤ θ ∧ θ ≤ 4 * Real.pi)
  (h2 : Real.sqrt ((1 + Real.cos θ) / 2) + Real.sqrt ((1 - Real.cos θ) / 2) = Real.sqrt 6 / 2) :
  θ = 19 * Real.pi / 6 ∨ θ = 23 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_theta_value_l3251_325149


namespace NUMINAMATH_CALUDE_regression_line_properties_l3251_325107

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  b : ℝ  -- slope
  a : ℝ  -- intercept

/-- Represents a dataset -/
structure Dataset where
  points : List Point
  centroid : Point

/-- Checks if a point lies on the regression line -/
def pointOnLine (model : LinearRegression) (p : Point) : Prop :=
  p.y = model.b * p.x + model.a

/-- The main theorem stating that the regression line passes through the centroid
    but not necessarily through all data points -/
theorem regression_line_properties (data : Dataset) (model : LinearRegression) :
  (pointOnLine model data.centroid) ∧
  (∃ p ∈ data.points, ¬pointOnLine model p) := by
  sorry


end NUMINAMATH_CALUDE_regression_line_properties_l3251_325107


namespace NUMINAMATH_CALUDE_angle_bisector_length_squared_l3251_325126

/-- Given a triangle with sides a, b, and c, fa is the length of the angle bisector of angle α,
    and u and v are the lengths of the segments into which fa divides side a. -/
theorem angle_bisector_length_squared (a b c fa u v : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ fa > 0 ∧ u > 0 ∧ v > 0)
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b)
  (h_segments : u + v = a)
  (h_ratio : u / v = c / b) :
  fa^2 = b * c - u * v := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_squared_l3251_325126


namespace NUMINAMATH_CALUDE_alpha_gt_beta_not_sufficient_nor_necessary_for_sin_alpha_gt_sin_beta_l3251_325187

theorem alpha_gt_beta_not_sufficient_nor_necessary_for_sin_alpha_gt_sin_beta :
  ∃ (α β : Real),
    0 < α ∧ α < π/2 ∧
    0 < β ∧ β < π/2 ∧
    (
      (α > β ∧ ¬(Real.sin α > Real.sin β)) ∧
      (Real.sin α > Real.sin β ∧ ¬(α > β))
    ) :=
by sorry

end NUMINAMATH_CALUDE_alpha_gt_beta_not_sufficient_nor_necessary_for_sin_alpha_gt_sin_beta_l3251_325187


namespace NUMINAMATH_CALUDE_no_prime_intercept_lines_through_point_l3251_325112

-- Define a prime number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define a line with intercepts
def Line (a b : ℕ) := {(x, y) : ℝ × ℝ | x / a + y / b = 1}

-- Theorem statement
theorem no_prime_intercept_lines_through_point :
  ¬∃ (a b : ℕ), isPrime a ∧ isPrime b ∧ (6, 5) ∈ Line a b := by
  sorry

end NUMINAMATH_CALUDE_no_prime_intercept_lines_through_point_l3251_325112


namespace NUMINAMATH_CALUDE_sqrt_625_equals_5_to_m_l3251_325158

theorem sqrt_625_equals_5_to_m (m : ℝ) : (625 : ℝ)^(1/2) = 5^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_625_equals_5_to_m_l3251_325158


namespace NUMINAMATH_CALUDE_f_has_real_roots_a_range_l3251_325138

-- Define the quadratic function f
def f (a x : ℝ) : ℝ := x^2 + (2*a - 1)*x + 1 - 2*a

-- Theorem 1: For all a ∈ ℝ, f(x) = 1 has real roots
theorem f_has_real_roots (a : ℝ) : ∃ x : ℝ, f a x = 1 := by sorry

-- Theorem 2: If f has zero points in (-1,0) and (0,1/2), then 1/2 < a < 3/4
theorem a_range (a : ℝ) (h1 : f a (-1) > 0) (h2 : f a 0 < 0) (h3 : f a (1/2) > 0) :
  1/2 < a ∧ a < 3/4 := by sorry

end NUMINAMATH_CALUDE_f_has_real_roots_a_range_l3251_325138


namespace NUMINAMATH_CALUDE_function_composition_ratio_l3251_325189

/-- Given two functions f and g, prove that f(g(f(3))) / g(f(g(3))) = 59/19 -/
theorem function_composition_ratio
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = 3 * x + 2)
  (hg : ∀ x, g x = 2 * x - 3) :
  f (g (f 3)) / g (f (g 3)) = 59 / 19 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l3251_325189


namespace NUMINAMATH_CALUDE_C_power_50_l3251_325151

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 4; -8, -10]

theorem C_power_50 : C^50 = !![201, 200; -400, -449] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l3251_325151


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l3251_325156

/-- Given two people moving in opposite directions, this theorem proves
    the speed of the second person given the conditions of the problem. -/
theorem opposite_direction_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 4)
  (h2 : distance = 28)
  (h3 : speed1 = 4)
  (h4 : distance = time * (speed1 + speed2)) :
  speed2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_opposite_direction_speed_l3251_325156


namespace NUMINAMATH_CALUDE_warehouse_weight_limit_l3251_325167

theorem warehouse_weight_limit (P : ℕ) (certain_weight : ℝ) : 
  (P : ℝ) * 0.3 < 75 ∧ 
  (P : ℝ) * 0.2 = 48 ∧ 
  (P : ℝ) * 0.8 ≥ certain_weight ∧ 
  24 ≥ certain_weight ∧ 24 < 75 →
  certain_weight = 75 := by
sorry

end NUMINAMATH_CALUDE_warehouse_weight_limit_l3251_325167


namespace NUMINAMATH_CALUDE_expression_value_l3251_325159

theorem expression_value (a b : ℝ) (h1 : a = Real.sqrt 3 - Real.sqrt 2) (h2 : b = Real.sqrt 3 + Real.sqrt 2) :
  a^2 + 3*a*b + b^2 - a + b = 13 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3251_325159


namespace NUMINAMATH_CALUDE_vector_at_t_3_l3251_325186

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  -- The vector on the line at any given t
  vector : ℝ → ℝ × ℝ × ℝ

/-- Theorem: Given a parameterized line with specific points, 
    we can determine the vector at t = 3 -/
theorem vector_at_t_3 
  (line : ParametricLine)
  (h1 : line.vector (-1) = (1, 3, 8))
  (h2 : line.vector 2 = (0, -2, -4)) :
  line.vector 3 = (-1/3, -11/3, -8) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_t_3_l3251_325186


namespace NUMINAMATH_CALUDE_real_part_of_z_l3251_325127

def i : ℂ := Complex.I

theorem real_part_of_z (z : ℂ) (h : (1 + 2*i)*z = 3 + 4*i) : 
  Complex.re z = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3251_325127


namespace NUMINAMATH_CALUDE_polynomial_product_l3251_325134

-- Define the polynomials f, g, and h
def f (x : ℝ) : ℝ := x^4 - x^3 - 1
def g (x : ℝ) : ℝ := x^8 - x^6 - 2*x^4 + 1
def h (x : ℝ) : ℝ := x^4 + x^3 - 1

-- State the theorem
theorem polynomial_product :
  (∀ x, g x = f x * h x) → (∀ x, h x = x^4 + x^3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_l3251_325134


namespace NUMINAMATH_CALUDE_factor_expression_l3251_325155

theorem factor_expression (x : ℝ) : 4 * x * (x - 2) + 6 * (x - 2) = 2 * (x - 2) * (2 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3251_325155


namespace NUMINAMATH_CALUDE_floor_equality_l3251_325181

theorem floor_equality (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b > 1) :
  ⌊((a - b)^2 - 1 : ℚ) / (a * b)⌋ = ⌊((a - b)^2 - 1 : ℚ) / (a * b - 1)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_l3251_325181


namespace NUMINAMATH_CALUDE_modulus_of_z_l3251_325136

theorem modulus_of_z (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3251_325136


namespace NUMINAMATH_CALUDE_sqrt_16_divided_by_2_l3251_325180

theorem sqrt_16_divided_by_2 : Real.sqrt 16 / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_divided_by_2_l3251_325180


namespace NUMINAMATH_CALUDE_key_lime_juice_yield_l3251_325169

def recipe_amount : ℚ := 1/4
def tablespoons_per_cup : ℕ := 16
def key_limes_needed : ℕ := 8

theorem key_lime_juice_yield : 
  let doubled_amount : ℚ := 2 * recipe_amount
  let total_tablespoons : ℚ := doubled_amount * tablespoons_per_cup
  let juice_per_lime : ℚ := total_tablespoons / key_limes_needed
  juice_per_lime = 1 := by sorry

end NUMINAMATH_CALUDE_key_lime_juice_yield_l3251_325169


namespace NUMINAMATH_CALUDE_peters_mothers_age_l3251_325105

/-- Proves that Peter's mother's age is 60 given the problem conditions -/
theorem peters_mothers_age :
  ∀ (harriet_age peter_age mother_age : ℕ),
    harriet_age = 13 →
    peter_age + 4 = 2 * (harriet_age + 4) →
    peter_age = mother_age / 2 →
    mother_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_peters_mothers_age_l3251_325105


namespace NUMINAMATH_CALUDE_sequence_problem_l3251_325178

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem sequence_problem (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_eq : 2 * a 3 - (a 8)^2 + 2 * a 13 = 0)
    (h_b8 : b 8 = a 8) :
  b 4 * b 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3251_325178


namespace NUMINAMATH_CALUDE_fraction_comparison_l3251_325160

theorem fraction_comparison (a b c d : ℚ) 
  (h1 : a / b < c / d) 
  (h2 : b > d) 
  (h3 : d > 0) : 
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3251_325160


namespace NUMINAMATH_CALUDE_line_slope_l3251_325162

theorem line_slope (x y : ℝ) : x + 2 * y - 6 = 0 → (y - 3) = (-1/2) * x := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3251_325162


namespace NUMINAMATH_CALUDE_range_of_a_l3251_325118

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| < 1 ↔ (1/2 : ℝ) < x ∧ x < (3/2 : ℝ)) ↔ 
  ((1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3251_325118


namespace NUMINAMATH_CALUDE_product_sum_of_digits_l3251_325121

def repeat_digit (d : Nat) (n : Nat) : Nat :=
  d * ((10^n - 1) / 9)

def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem product_sum_of_digits :
  sum_of_digits (repeat_digit 4 2012 * repeat_digit 9 2012) = 18108 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_of_digits_l3251_325121


namespace NUMINAMATH_CALUDE_distance_between_points_l3251_325124

/-- The distance between points (3, 5) and (-4, 1) is √65 -/
theorem distance_between_points : Real.sqrt 65 = Real.sqrt ((3 - (-4))^2 + (5 - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3251_325124


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3251_325145

theorem fraction_multiplication (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (b * c / a^2) * (a / b^2) = c / (a * b) := by
sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3251_325145


namespace NUMINAMATH_CALUDE_eric_erasers_l3251_325168

/-- Given that Eric shares his erasers among 99 friends and each friend gets 94 erasers,
    prove that Eric has 9306 erasers in total. -/
theorem eric_erasers (num_friends : ℕ) (erasers_per_friend : ℕ) 
    (h1 : num_friends = 99) (h2 : erasers_per_friend = 94) : 
    num_friends * erasers_per_friend = 9306 := by
  sorry

end NUMINAMATH_CALUDE_eric_erasers_l3251_325168


namespace NUMINAMATH_CALUDE_value_of_y_l3251_325108

theorem value_of_y (x y : ℝ) (h1 : x^(2*y) = 16) (h2 : x = 16) : y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3251_325108


namespace NUMINAMATH_CALUDE_jeff_travel_distance_l3251_325165

def speed1 : ℝ := 80
def time1 : ℝ := 6
def speed2 : ℝ := 60
def time2 : ℝ := 4
def speed3 : ℝ := 40
def time3 : ℝ := 2

def total_distance : ℝ := speed1 * time1 + speed2 * time2 + speed3 * time3

theorem jeff_travel_distance : total_distance = 800 := by
  sorry

end NUMINAMATH_CALUDE_jeff_travel_distance_l3251_325165


namespace NUMINAMATH_CALUDE_weight_of_new_person_l3251_325188

theorem weight_of_new_person (initial_count : ℕ) (weight_increase : ℝ) (leaving_weight : ℝ) :
  initial_count = 12 →
  weight_increase = 4 →
  leaving_weight = 58 →
  (initial_count : ℝ) * weight_increase + leaving_weight = 106 :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l3251_325188


namespace NUMINAMATH_CALUDE_donated_books_count_l3251_325193

/-- Represents the number of books in the library over time --/
structure LibraryBooks where
  initial_old : ℕ
  bought_two_years_ago : ℕ
  bought_last_year : ℕ
  current_total : ℕ

/-- Calculates the number of old books donated --/
def books_donated (lib : LibraryBooks) : ℕ :=
  lib.initial_old + lib.bought_two_years_ago + lib.bought_last_year - lib.current_total

/-- Theorem stating the number of old books donated --/
theorem donated_books_count (lib : LibraryBooks) 
  (h1 : lib.initial_old = 500)
  (h2 : lib.bought_two_years_ago = 300)
  (h3 : lib.bought_last_year = lib.bought_two_years_ago + 100)
  (h4 : lib.current_total = 1000) :
  books_donated lib = 200 := by
  sorry

#eval books_donated ⟨500, 300, 400, 1000⟩

end NUMINAMATH_CALUDE_donated_books_count_l3251_325193


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3251_325113

/-- Given vectors a and b, if 3a - 2b + c = 0, then c = (-23, -12) -/
theorem vector_equation_solution (a b c : ℝ × ℝ) :
  a = (5, 2) →
  b = (-4, -3) →
  3 • a - 2 • b + c = (0, 0) →
  c = (-23, -12) := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3251_325113


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3251_325154

theorem sum_of_fifth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (sum_1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (sum_2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (sum_3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3251_325154


namespace NUMINAMATH_CALUDE_solution_of_fraction_equation_l3251_325161

theorem solution_of_fraction_equation :
  ∃ x : ℝ, (3 - x) / (4 + 2*x) = 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_fraction_equation_l3251_325161


namespace NUMINAMATH_CALUDE_sequence_square_l3251_325190

theorem sequence_square (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n - 1) →
  ∀ n : ℕ, n ≥ 1 → a n = n^2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_square_l3251_325190


namespace NUMINAMATH_CALUDE_emily_spent_28_dollars_l3251_325140

/-- Calculates the total cost of Emily's flower purchase --/
def emily_flower_cost (rose_price daisy_price tulip_price lily_price : ℕ)
  (rose_qty daisy_qty tulip_qty lily_qty : ℕ) : ℕ :=
  rose_price * rose_qty + daisy_price * daisy_qty + tulip_price * tulip_qty + lily_price * lily_qty

/-- Proves that Emily spent 28 dollars on flowers --/
theorem emily_spent_28_dollars :
  emily_flower_cost 4 3 5 6 2 3 1 1 = 28 := by
  sorry

end NUMINAMATH_CALUDE_emily_spent_28_dollars_l3251_325140


namespace NUMINAMATH_CALUDE_two_special_birth_years_l3251_325182

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem two_special_birth_years :
  ∃ (y1 y2 : ℕ),
    y1 ≠ y2 ∧
    y1 ≥ 1900 ∧ y1 ≤ 2021 ∧
    y2 ≥ 1900 ∧ y2 ≤ 2021 ∧
    2021 - y1 = sum_of_digits y1 ∧
    2021 - y2 = sum_of_digits y2 ∧
    2022 - y1 = 8 ∧
    2022 - y2 = 26 :=
by sorry

end NUMINAMATH_CALUDE_two_special_birth_years_l3251_325182
