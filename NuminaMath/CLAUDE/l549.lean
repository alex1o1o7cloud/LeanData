import Mathlib

namespace NUMINAMATH_CALUDE_total_volume_is_85_l549_54945

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The total volume of n cubes, each with side length s -/
def total_volume (n : ℕ) (s : ℝ) : ℝ := n * (cube_volume s)

/-- Carl's cubes -/
def carl_cubes : ℕ := 3
def carl_side_length : ℝ := 3

/-- Kate's cubes -/
def kate_cubes : ℕ := 4
def kate_side_length : ℝ := 1

/-- The theorem stating that the total volume of Carl's and Kate's cubes is 85 -/
theorem total_volume_is_85 : 
  total_volume carl_cubes carl_side_length + total_volume kate_cubes kate_side_length = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_is_85_l549_54945


namespace NUMINAMATH_CALUDE_wall_length_proof_l549_54926

/-- Proves that the length of a wall is 260 meters given specific brick and wall dimensions --/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_height : ℝ) (wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_height = 2 →
  wall_width = 0.75 →
  num_bricks = 26000 →
  (brick_length * brick_width * brick_height * num_bricks) / (wall_height * wall_width) = 260 := by
  sorry

#check wall_length_proof

end NUMINAMATH_CALUDE_wall_length_proof_l549_54926


namespace NUMINAMATH_CALUDE_liquid_level_rate_of_change_l549_54901

/-- The rate of change of liquid level height in a cylindrical container -/
theorem liquid_level_rate_of_change 
  (d : ℝ) -- diameter of the base
  (drain_rate : ℝ) -- rate at which liquid is drained
  (h : ℝ → ℝ) -- height of liquid as a function of time
  (t : ℝ) -- time variable
  (hd : d = 2) -- given diameter
  (hdrain : drain_rate = 0.01) -- given drain rate
  : deriv h t = -drain_rate / (π * (d/2)^2) := by
  sorry

#check liquid_level_rate_of_change

end NUMINAMATH_CALUDE_liquid_level_rate_of_change_l549_54901


namespace NUMINAMATH_CALUDE_caging_theorem_l549_54933

/-- The number of ways to cage 6 animals in 6 cages, where 4 cages are too small for 6 animals -/
def caging_arrangements : ℕ := 24

/-- The total number of animals -/
def total_animals : ℕ := 6

/-- The total number of cages -/
def total_cages : ℕ := 6

/-- The number of cages that are too small for most animals -/
def small_cages : ℕ := 4

/-- The number of animals that can't fit in the small cages -/
def large_animals : ℕ := 6

theorem caging_theorem : 
  caging_arrangements = 24 ∧ 
  total_animals = 6 ∧ 
  total_cages = 6 ∧ 
  small_cages = 4 ∧ 
  large_animals = 6 :=
sorry

end NUMINAMATH_CALUDE_caging_theorem_l549_54933


namespace NUMINAMATH_CALUDE_age_problem_contradiction_l549_54976

/-- Demonstrates the contradiction in the given age problem -/
theorem age_problem_contradiction (A B C D : ℕ) : 
  (A + B = B + C + 11) →  -- Condition 1
  (A + B + D = B + C + D + 8) →  -- Condition 2
  (A + C = 2 * D) →  -- Condition 3
  False := by sorry


end NUMINAMATH_CALUDE_age_problem_contradiction_l549_54976


namespace NUMINAMATH_CALUDE_four_isosceles_triangles_l549_54912

-- Define a Point type for 2D coordinates
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Triangle type
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

-- Function to calculate the squared distance between two points
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := distanceSquared t.a t.b
  let d2 := distanceSquared t.b t.c
  let d3 := distanceSquared t.c t.a
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the five triangles
def triangleA : Triangle := ⟨⟨1, 5⟩, ⟨3, 5⟩, ⟨2, 3⟩⟩
def triangleB : Triangle := ⟨⟨4, 3⟩, ⟨4, 5⟩, ⟨6, 3⟩⟩
def triangleC : Triangle := ⟨⟨1, 2⟩, ⟨3, 1⟩, ⟨5, 2⟩⟩
def triangleD : Triangle := ⟨⟨7, 3⟩, ⟨6, 5⟩, ⟨9, 3⟩⟩
def triangleE : Triangle := ⟨⟨8, 2⟩, ⟨9, 4⟩, ⟨10, 1⟩⟩

-- Theorem stating that exactly 4 out of 5 triangles are isosceles
theorem four_isosceles_triangles :
  (isIsosceles triangleA ∧ 
   isIsosceles triangleB ∧ 
   isIsosceles triangleC ∧ 
   ¬isIsosceles triangleD ∧ 
   isIsosceles triangleE) :=
by sorry

end NUMINAMATH_CALUDE_four_isosceles_triangles_l549_54912


namespace NUMINAMATH_CALUDE_sin_a_less_cos_b_in_obtuse_triangle_l549_54906

/-- In a triangle ABC where angle C is obtuse, sin A < cos B -/
theorem sin_a_less_cos_b_in_obtuse_triangle (A B C : ℝ) (h_triangle : A + B + C = π) (h_obtuse : C > π/2) : 
  Real.sin A < Real.cos B := by
sorry

end NUMINAMATH_CALUDE_sin_a_less_cos_b_in_obtuse_triangle_l549_54906


namespace NUMINAMATH_CALUDE_base_5_representation_of_425_l549_54946

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

theorem base_5_representation_of_425 :
  toBase5 425 = [3, 2, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_base_5_representation_of_425_l549_54946


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l549_54949

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [0, 1, 2, 3, 4]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 10738 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l549_54949


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l549_54963

/-- The length of the path traveled by the center of a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / Real.pi) :
  let path_length := 3 * (π * r / 4)
  path_length = 4.5 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l549_54963


namespace NUMINAMATH_CALUDE_southAmericanStampsCost_l549_54905

/-- Represents the number of stamps for a country in a specific decade -/
structure StampCount :=
  (fifties sixties seventies eighties : ℕ)

/-- Represents a country's stamp collection -/
structure Country :=
  (name : String)
  (price : ℚ)
  (counts : StampCount)

def colombia : Country :=
  { name := "Colombia"
  , price := 3 / 100
  , counts := { fifties := 7, sixties := 6, seventies := 12, eighties := 15 } }

def argentina : Country :=
  { name := "Argentina"
  , price := 6 / 100
  , counts := { fifties := 4, sixties := 8, seventies := 10, eighties := 9 } }

def southAmericanCountries : List Country := [colombia, argentina]

def stampsBefore1980s (c : Country) : ℕ :=
  c.counts.fifties + c.counts.sixties + c.counts.seventies

def totalCost (countries : List Country) : ℚ :=
  countries.map (fun c => (stampsBefore1980s c : ℚ) * c.price) |>.sum

theorem southAmericanStampsCost :
  totalCost southAmericanCountries = 207 / 100 := by sorry

end NUMINAMATH_CALUDE_southAmericanStampsCost_l549_54905


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_1976_l549_54932

theorem no_two_digit_factors_of_1976 : 
  ¬∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 1976 := by
sorry

end NUMINAMATH_CALUDE_no_two_digit_factors_of_1976_l549_54932


namespace NUMINAMATH_CALUDE_square_plus_one_divides_l549_54914

theorem square_plus_one_divides (n : ℕ) : (n^2 + 1) ∣ n ↔ n = 0 := by sorry

end NUMINAMATH_CALUDE_square_plus_one_divides_l549_54914


namespace NUMINAMATH_CALUDE_goods_train_speed_l549_54936

theorem goods_train_speed 
  (speed_A : ℝ) 
  (length_B : ℝ) 
  (passing_time : ℝ) 
  (h1 : speed_A = 70) 
  (h2 : length_B = 0.45) 
  (h3 : passing_time = 15 / 3600) : 
  ∃ (speed_B : ℝ), speed_B = 38 := by
sorry

end NUMINAMATH_CALUDE_goods_train_speed_l549_54936


namespace NUMINAMATH_CALUDE_area_of_right_triangle_with_inscribed_circle_l549_54960

-- Define the right triangle with inscribed circle
def RightTriangleWithInscribedCircle (a b c r : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧
  a^2 + b^2 = c^2 ∧
  c = 13 ∧
  (r + 6) = a ∧
  (r + 7) = b

-- Theorem statement
theorem area_of_right_triangle_with_inscribed_circle 
  (a b c r : ℝ) 
  (h : RightTriangleWithInscribedCircle a b c r) :
  (1/2 : ℝ) * a * b = 42 :=
by
  sorry

#check area_of_right_triangle_with_inscribed_circle

end NUMINAMATH_CALUDE_area_of_right_triangle_with_inscribed_circle_l549_54960


namespace NUMINAMATH_CALUDE_cultural_group_members_l549_54947

theorem cultural_group_members :
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 200 ∧ n % 7 = 4 ∧ n % 11 = 6 ∧
  (n = 116 ∨ n = 193) :=
by sorry

end NUMINAMATH_CALUDE_cultural_group_members_l549_54947


namespace NUMINAMATH_CALUDE_faster_walking_speed_l549_54929

/-- Proves that given a person walks 50 km at 10 km/hr, if they walked at a faster speed
    for the same time and covered 70 km, the faster speed is 14 km/hr -/
theorem faster_walking_speed (actual_distance : ℝ) (original_speed : ℝ) (extra_distance : ℝ)
    (h1 : actual_distance = 50)
    (h2 : original_speed = 10)
    (h3 : extra_distance = 20) :
    let time := actual_distance / original_speed
    let total_distance := actual_distance + extra_distance
    let faster_speed := total_distance / time
    faster_speed = 14 := by sorry

end NUMINAMATH_CALUDE_faster_walking_speed_l549_54929


namespace NUMINAMATH_CALUDE_man_work_time_l549_54961

/-- Represents the time taken to complete a piece of work -/
structure WorkTime where
  days : ℝ
  days_pos : days > 0

/-- Represents the rate at which work is completed -/
def WorkRate := ℝ

theorem man_work_time (total_work : ℝ) 
  (h_total_work_pos : total_work > 0)
  (combined_time : WorkTime) 
  (son_time : WorkTime) 
  (h_combined : combined_time.days = 3)
  (h_son : son_time.days = 7.5) :
  ∃ (man_time : WorkTime), man_time.days = 5 :=
sorry

end NUMINAMATH_CALUDE_man_work_time_l549_54961


namespace NUMINAMATH_CALUDE_dividend_calculation_l549_54962

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 12)
  (h3 : remainder = 8) :
  divisor * quotient + remainder = 176 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l549_54962


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l549_54994

theorem largest_number_in_ratio (a b c d : ℚ) : 
  a / b = -3/2 →
  b / c = 4/5 →
  c / d = -2/3 →
  a + b + c + d = 1344 →
  max a (max b (max c d)) = 40320 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l549_54994


namespace NUMINAMATH_CALUDE_hyperbolic_to_linear_transformation_l549_54927

theorem hyperbolic_to_linear_transformation (x y a b : ℝ) (h : 1 / y = a + b / x) :
  1 / y = a + b * (1 / x) := by sorry

end NUMINAMATH_CALUDE_hyperbolic_to_linear_transformation_l549_54927


namespace NUMINAMATH_CALUDE_daffodil_stamps_count_l549_54907

theorem daffodil_stamps_count 
  (rooster_stamps : ℕ) 
  (daffodil_stamps : ℕ) 
  (h1 : rooster_stamps = 2) 
  (h2 : rooster_stamps - daffodil_stamps = 0) : 
  daffodil_stamps = 2 :=
by sorry

end NUMINAMATH_CALUDE_daffodil_stamps_count_l549_54907


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l549_54988

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 150)
  (h2 : throwers = 90)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 5 = 0)
  : total_players - (total_players - throwers) / 5 = 138 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l549_54988


namespace NUMINAMATH_CALUDE_magazines_sold_l549_54937

theorem magazines_sold (total : ℝ) (newspapers : ℝ) (h1 : total = 425.0) (h2 : newspapers = 275.0) :
  total - newspapers = 150.0 := by
  sorry

end NUMINAMATH_CALUDE_magazines_sold_l549_54937


namespace NUMINAMATH_CALUDE_matrix_addition_result_l549_54997

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -2; -3, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 0; 7, -8]

theorem matrix_addition_result : A + B = !![-2, -2; 4, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_addition_result_l549_54997


namespace NUMINAMATH_CALUDE_unique_k_satisfying_conditions_l549_54998

/-- A sequence of binomial coefficients forms an arithmetic progression -/
def is_arithmetic_progression (n : ℕ) (j : ℕ) (k : ℕ) : Prop :=
  ∃ d : ℤ, ∀ i : ℕ, i < k → (n.choose (j + i + 1) : ℤ) - (n.choose (j + i) : ℤ) = d

/-- Condition for part a) -/
def condition_a (k : ℕ) : Prop :=
  ∀ n : ℕ, ¬∃ j : ℕ, j ≤ n - k + 1 ∧ is_arithmetic_progression n j k

/-- Condition for part b) -/
def condition_b (k : ℕ) : Prop :=
  ∃ n : ℕ, ∃ j : ℕ, j ≤ n - k + 2 ∧ is_arithmetic_progression n j (k - 1)

/-- The main theorem -/
theorem unique_k_satisfying_conditions :
  ∃! k : ℕ, k > 0 ∧ condition_a k ∧ condition_b k :=
sorry

end NUMINAMATH_CALUDE_unique_k_satisfying_conditions_l549_54998


namespace NUMINAMATH_CALUDE_gcd_of_840_and_1764_l549_54983

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_840_and_1764_l549_54983


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l549_54904

theorem problem_1 : |-5| + (3 - Real.sqrt 2) ^ 0 - 2 * Real.tan (π / 4) = 4 := by sorry

theorem problem_2 (a : ℝ) (h1 : a ≠ 3) (h2 : a ≠ -3) : 
  (a / (a^2 - 9)) / (1 + 3 / (a - 3)) = 1 / (a + 3) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l549_54904


namespace NUMINAMATH_CALUDE_fraction_decomposition_l549_54971

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 10) (h2 : x ≠ -2) :
  (7 * x + 3) / (x^2 - 8*x - 20) = (73/12) / (x - 10) + (11/12) / (x + 2) := by
  sorry

#check fraction_decomposition

end NUMINAMATH_CALUDE_fraction_decomposition_l549_54971


namespace NUMINAMATH_CALUDE_cos_2x_value_l549_54943

theorem cos_2x_value (x : ℝ) (h : Real.sin (π / 4 + x / 2) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_value_l549_54943


namespace NUMINAMATH_CALUDE_train_length_and_speed_l549_54900

/-- A train passes by an observer in t₁ seconds and through a bridge of length a meters in t₂ seconds at a constant speed. This theorem proves the formulas for the train's length and speed. -/
theorem train_length_and_speed (t₁ t₂ a : ℝ) (h₁ : t₁ > 0) (h₂ : t₂ > t₁) (h₃ : a > 0) :
  ∃ (L V : ℝ),
    L = (a * t₁) / (t₂ - t₁) ∧
    V = a / (t₂ - t₁) ∧
    L / t₁ = V ∧
    (L + a) / t₂ = V :=
by sorry

end NUMINAMATH_CALUDE_train_length_and_speed_l549_54900


namespace NUMINAMATH_CALUDE_expression_simplification_l549_54923

theorem expression_simplification (a b : ℝ) : 
  32 * a^2 * b^2 * (a^2 + b^2)^2 + (a^2 - b^2)^4 + 
  8 * a * b * (a^2 + b^2) * Real.sqrt (16 * a^2 * b^2 * (a^2 + b^2)^2 + (a^2 - b^2)^4) = 
  (a + b)^8 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l549_54923


namespace NUMINAMATH_CALUDE_sixth_grade_forgot_homework_percentage_l549_54913

/-- Represents the percentage of students who forgot their homework in a group -/
def forgot_homework_percentage (total : ℕ) (forgot : ℕ) : ℚ :=
  (forgot : ℚ) / (total : ℚ) * 100

/-- Calculates the total number of students who forgot their homework -/
def total_forgot (group_a_total : ℕ) (group_b_total : ℕ) 
  (group_a_forgot_percent : ℚ) (group_b_forgot_percent : ℚ) : ℕ :=
  (group_a_total * group_a_forgot_percent.num / group_a_forgot_percent.den).toNat +
  (group_b_total * group_b_forgot_percent.num / group_b_forgot_percent.den).toNat

theorem sixth_grade_forgot_homework_percentage :
  let group_a_total : ℕ := 20
  let group_b_total : ℕ := 80
  let group_a_forgot_percent : ℚ := 20 / 100
  let group_b_forgot_percent : ℚ := 15 / 100
  let total_students : ℕ := group_a_total + group_b_total
  let total_forgot : ℕ := total_forgot group_a_total group_b_total group_a_forgot_percent group_b_forgot_percent
  forgot_homework_percentage total_students total_forgot = 16 := by
sorry

end NUMINAMATH_CALUDE_sixth_grade_forgot_homework_percentage_l549_54913


namespace NUMINAMATH_CALUDE_cube_root_of_point_on_line_l549_54935

/-- For any point (a, b) on the graph of y = x - 1, the cube root of b - a is -1 -/
theorem cube_root_of_point_on_line (a b : ℝ) (h : b = a - 1) : 
  (b - a : ℝ) ^ (1/3 : ℝ) = -1 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_point_on_line_l549_54935


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l549_54910

def P : Set ℝ := {1, 2, 3}
def Q : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l549_54910


namespace NUMINAMATH_CALUDE_equal_winning_chance_l549_54985

/-- Represents a lottery ticket -/
structure LotteryTicket where
  id : ℕ

/-- Represents a lottery -/
structure Lottery where
  winningProbability : ℝ
  totalTickets : ℕ

/-- The probability of a ticket winning is equal to the lottery's winning probability -/
def ticketWinningProbability (lottery : Lottery) (ticket : LotteryTicket) : ℝ :=
  lottery.winningProbability

theorem equal_winning_chance (lottery : Lottery) 
    (h1 : lottery.winningProbability = 0.002)
    (h2 : lottery.totalTickets = 1000) :
    ∀ (t1 t2 : LotteryTicket), ticketWinningProbability lottery t1 = ticketWinningProbability lottery t2 :=
  sorry


end NUMINAMATH_CALUDE_equal_winning_chance_l549_54985


namespace NUMINAMATH_CALUDE_four_digit_number_counts_l549_54918

def digits : Finset ℕ := {1, 2, 3, 4, 5}

def four_digit_numbers_no_repetition : ℕ := sorry

def four_digit_numbers_with_repetition : ℕ := sorry

def odd_four_digit_numbers_no_repetition : ℕ := sorry

theorem four_digit_number_counts :
  four_digit_numbers_no_repetition = 120 ∧
  four_digit_numbers_with_repetition = 625 ∧
  odd_four_digit_numbers_no_repetition = 72 := by sorry

end NUMINAMATH_CALUDE_four_digit_number_counts_l549_54918


namespace NUMINAMATH_CALUDE_A_intersect_nat_l549_54941

def A : Set ℝ := {x | x - 2 < 0}

theorem A_intersect_nat : A ∩ Set.range (fun n : ℕ => (n : ℝ)) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_nat_l549_54941


namespace NUMINAMATH_CALUDE_greatest_integer_no_real_roots_l549_54953

theorem greatest_integer_no_real_roots (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 15 ≠ 0) ↔ c ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_no_real_roots_l549_54953


namespace NUMINAMATH_CALUDE_fraction_power_multiply_l549_54964

theorem fraction_power_multiply (a b c : ℚ) : 
  (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_multiply_l549_54964


namespace NUMINAMATH_CALUDE_orchestra_members_count_l549_54954

theorem orchestra_members_count : ∃! n : ℕ, 
  130 < n ∧ n < 260 ∧ 
  n % 6 = 1 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 3 ∧
  n = 241 := by
sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l549_54954


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l549_54966

/-- Represents the rates of biking, jogging, and swimming -/
structure Rates where
  biking : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (r : Rates) : Prop :=
  3 * r.biking + 2 * r.jogging + 4 * r.swimming = 84 ∧
  4 * r.biking + 3 * r.jogging + 2 * r.swimming = 106

/-- The theorem to be proved -/
theorem rates_sum_of_squares (r : Rates) : 
  satisfies_conditions r → r.biking^2 + r.jogging^2 + r.swimming^2 = 1125 := by
  sorry


end NUMINAMATH_CALUDE_rates_sum_of_squares_l549_54966


namespace NUMINAMATH_CALUDE_hyperbola_properties_l549_54978

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ((x - 1)^2 / a^2) - ((y - 1)^2 / b^2) = 1

-- Define the conditions
theorem hyperbola_properties :
  ∃ (t : ℝ),
    -- Center at (1, 1) is implicit in the hyperbola definition
    hyperbola 4 2 ∧  -- Passes through (4, 2)
    hyperbola 3 1 ∧  -- Vertex at (3, 1)
    hyperbola t 4 ∧  -- Passes through (t, 4)
    (t^2 = 64 ∨ t^2 = 36) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_properties_l549_54978


namespace NUMINAMATH_CALUDE_cosine_equality_l549_54902

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (980 * π / 180) → n = 100 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l549_54902


namespace NUMINAMATH_CALUDE_boys_in_second_class_l549_54940

theorem boys_in_second_class (boys_first_class : ℕ) (boys_second_class : ℕ) : 
  boys_first_class = 28 →
  boys_first_class = (7 * boys_second_class) / 8 →
  boys_second_class = 32 := by
sorry

end NUMINAMATH_CALUDE_boys_in_second_class_l549_54940


namespace NUMINAMATH_CALUDE_number_pattern_l549_54989

/-- Represents a number as a string of consecutive '1' digits -/
def ones (n : ℕ) : ℕ :=
  (10 ^ n - 1) / 9

/-- The main theorem to be proved -/
theorem number_pattern (n : ℕ) (h : n ≤ 123456) :
  n * 9 + (n + 1) = ones (n + 1) :=
sorry

end NUMINAMATH_CALUDE_number_pattern_l549_54989


namespace NUMINAMATH_CALUDE_chef_butter_remaining_l549_54952

/-- Represents the recipe and chef's actions for making brownies. -/
structure BrownieRecipe where
  /-- The amount of butter (in ounces) required per cup of baking mix. -/
  butter_per_cup : ℝ
  /-- The amount of baking mix (in cups) the chef planned to use. -/
  planned_baking_mix : ℝ
  /-- The amount of coconut oil (in ounces) the chef used. -/
  coconut_oil_used : ℝ

/-- Calculates the amount of butter remaining after substituting with coconut oil. -/
def butter_remaining (recipe : BrownieRecipe) : ℝ :=
  recipe.butter_per_cup * recipe.planned_baking_mix - recipe.coconut_oil_used

/-- Theorem stating that the chef had 4 ounces of butter remaining. -/
theorem chef_butter_remaining (recipe : BrownieRecipe)
    (h1 : recipe.butter_per_cup = 2)
    (h2 : recipe.planned_baking_mix = 6)
    (h3 : recipe.coconut_oil_used = 8) :
    butter_remaining recipe = 4 := by
  sorry

#eval butter_remaining { butter_per_cup := 2, planned_baking_mix := 6, coconut_oil_used := 8 }

end NUMINAMATH_CALUDE_chef_butter_remaining_l549_54952


namespace NUMINAMATH_CALUDE_pyramid_volume_is_2_root2_div_3_l549_54984

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with vertex P and base ABC -/
structure Pyramid where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Check if a triangle is equilateral -/
def isEquilateral (A B C : Point3D) : Prop := 
  distance A B = distance B C ∧ distance B C = distance C A

/-- Calculate the angle between three points -/
def angle (A P C : Point3D) : ℝ := sorry

/-- Calculate the volume of a pyramid -/
def pyramidVolume (p : Pyramid) : ℝ := sorry

theorem pyramid_volume_is_2_root2_div_3 (p : Pyramid) : 
  isEquilateral p.A p.B p.C →
  distance p.P p.A = distance p.P p.B ∧ 
  distance p.P p.B = distance p.P p.C →
  distance p.A p.B = 2 →
  angle p.A p.P p.C = Real.pi / 2 →
  pyramidVolume p = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_is_2_root2_div_3_l549_54984


namespace NUMINAMATH_CALUDE_stream_speed_l549_54928

/-- Proves that the speed of a stream is 4 km/hr given the conditions of the boat problem -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : boat_speed = 24)
  (h2 : distance = 168) (h3 : time = 6)
  (h4 : distance = (boat_speed + (distance / time - boat_speed)) * time) : 
  distance / time - boat_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l549_54928


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l549_54969

theorem units_digit_of_sum_of_cubes : 
  (42^3 + 24^3) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l549_54969


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l549_54957

/-- Proves that a train of length 240 m takes 24 seconds to pass a pole, given that it takes 89 seconds to pass a 650 m platform -/
theorem train_passing_pole_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_pass_platform : ℝ) 
  (h1 : train_length = 240)
  (h2 : platform_length = 650)
  (h3 : time_to_pass_platform = 89) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l549_54957


namespace NUMINAMATH_CALUDE_cakes_sold_daily_l549_54959

def cash_register_cost : ℕ := 1040
def bread_price : ℕ := 2
def bread_quantity : ℕ := 40
def cake_price : ℕ := 12
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2
def days_to_pay : ℕ := 8

def daily_bread_income : ℕ := bread_price * bread_quantity
def daily_expenses : ℕ := daily_rent + daily_electricity
def daily_profit_from_bread : ℕ := daily_bread_income - daily_expenses

theorem cakes_sold_daily (cakes_sold : ℕ) : 
  cakes_sold = 6 ↔ 
  days_to_pay * (daily_profit_from_bread + cake_price * cakes_sold) = cash_register_cost :=
by sorry

end NUMINAMATH_CALUDE_cakes_sold_daily_l549_54959


namespace NUMINAMATH_CALUDE_division_problem_l549_54942

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1200)
  (h2 : a = 1495)
  (h3 : a = b * q + 4) :
  q = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l549_54942


namespace NUMINAMATH_CALUDE_angie_coffee_amount_l549_54993

/-- Represents the number of cups of coffee brewed per pound of coffee. -/
def cupsPerPound : ℕ := 40

/-- Represents the number of cups of coffee Angie drinks per day. -/
def cupsPerDay : ℕ := 3

/-- Represents the number of days the coffee lasts. -/
def daysLasting : ℕ := 40

/-- Calculates the number of pounds of coffee Angie bought. -/
def coffeeAmount : ℕ := (cupsPerDay * daysLasting) / cupsPerPound

theorem angie_coffee_amount : coffeeAmount = 3 := by
  sorry

end NUMINAMATH_CALUDE_angie_coffee_amount_l549_54993


namespace NUMINAMATH_CALUDE_point_trajectory_l549_54990

/-- The trajectory of a point satisfying a specific equation -/
theorem point_trajectory (x y : ℝ) :
  (Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) →
  ((x^2 / 16 - y^2 / 9 = 1) ∧ (x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_trajectory_l549_54990


namespace NUMINAMATH_CALUDE_simplify_expression_l549_54955

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a^2 + b^2) :
  a^2 / b + b^2 / a - 1 / (a^2 * b^2) = (a^4 + 2*a*b + b^4 - 1) / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l549_54955


namespace NUMINAMATH_CALUDE_exists_acute_triangle_configuration_l549_54950

/-- A configuration of n points on a plane. -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ

/-- A triangle formed by three points. -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Predicate to check if a triangle is acute. -/
def is_acute (t : Triangle) : Prop :=
  sorry  -- Definition of acute triangle

/-- Function to get the i-th triangle from a point configuration. -/
def get_triangle (config : PointConfiguration n) (i : Fin n) : Triangle :=
  sorry  -- Definition to extract triangle from configuration

/-- Theorem stating the existence of a configuration with all acute triangles. -/
theorem exists_acute_triangle_configuration (n : ℕ) (h_odd : Odd n) (h_gt_3 : n > 3) :
  ∃ (config : PointConfiguration n), ∀ (i : Fin n), is_acute (get_triangle config i) :=
sorry

end NUMINAMATH_CALUDE_exists_acute_triangle_configuration_l549_54950


namespace NUMINAMATH_CALUDE_red_peaches_count_l549_54948

theorem red_peaches_count (yellow_peaches : ℕ) (red_yellow_difference : ℕ) 
  (h1 : yellow_peaches = 11)
  (h2 : red_yellow_difference = 8) :
  yellow_peaches + red_yellow_difference = 19 :=
by sorry

end NUMINAMATH_CALUDE_red_peaches_count_l549_54948


namespace NUMINAMATH_CALUDE_peaches_eaten_l549_54903

/-- Represents the state of peaches in a bowl --/
structure PeachBowl where
  total : ℕ
  ripe : ℕ
  unripe : ℕ

/-- Calculates the state of peaches after a given number of days --/
def ripenPeaches (initial : PeachBowl) (days : ℕ) (ripeningRate : ℕ) : PeachBowl :=
  { total := initial.total,
    ripe := min initial.total (initial.ripe + days * ripeningRate),
    unripe := max 0 (initial.total - (initial.ripe + days * ripeningRate)) }

/-- Theorem stating the number of peaches eaten --/
theorem peaches_eaten 
  (initial : PeachBowl)
  (ripeningRate : ℕ)
  (days : ℕ)
  (finalState : PeachBowl)
  (h1 : initial.total = 18)
  (h2 : initial.ripe = 4)
  (h3 : ripeningRate = 2)
  (h4 : days = 5)
  (h5 : finalState.ripe = finalState.unripe + 7)
  (h6 : finalState.total + 3 = (ripenPeaches initial days ripeningRate).total) :
  3 = initial.total - finalState.total :=
by
  sorry


end NUMINAMATH_CALUDE_peaches_eaten_l549_54903


namespace NUMINAMATH_CALUDE_kelly_initial_apples_l549_54995

theorem kelly_initial_apples (initial : ℕ) (to_pick : ℕ) (total : ℕ) 
  (h1 : to_pick = 49)
  (h2 : total = 105)
  (h3 : initial + to_pick = total) : 
  initial = 56 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_apples_l549_54995


namespace NUMINAMATH_CALUDE_boxes_with_neither_l549_54909

theorem boxes_with_neither (total : ℕ) (pencils : ℕ) (pens : ℕ) (both : ℕ) 
  (h1 : total = 12)
  (h2 : pencils = 8)
  (h3 : pens = 5)
  (h4 : both = 3) :
  total - (pencils + pens - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l549_54909


namespace NUMINAMATH_CALUDE_cubic_polynomials_constant_term_l549_54996

/-- Given two cubic polynomials p(x) and q(x) with specific root relationships,
    prove that there are only two possible values for the constant term d of p(x). -/
theorem cubic_polynomials_constant_term (c d : ℝ) : 
  (∃ (r s : ℝ), (r^3 + c*r + d = 0 ∧ s^3 + c*s + d = 0) ∧
   ((r+5)^3 + c*(r+5) + (d+210) = 0 ∧ (s-4)^3 + c*(s-4) + (d+210) = 0)) →
  (d = 240 ∨ d = 420) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_constant_term_l549_54996


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l549_54920

/-- Given a geometric sequence {aₙ} where a₁a₃ = a₄ = 4, prove that a₆ = 8 -/
theorem geometric_sequence_sixth_term (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- a is a geometric sequence
  a 1 * a 3 = 4 →  -- given condition
  a 4 = 4 →        -- given condition
  a 6 = 8 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l549_54920


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l549_54921

/-- Given two right triangles with sides 3, 4, and 5, where one triangle has a square
    inscribed with a vertex at the right angle (side length x) and the other has a square
    inscribed with a side on the hypotenuse (side length y), prove that x/y = 37/35 -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  (∃ (a b c d : ℝ), 
    a^2 + b^2 = c^2 ∧ a = 3 ∧ b = 4 ∧ c = 5 ∧
    x^2 = a * b - (a - x) * (b - x) ∧
    y * (a + b) = c * y) →
  x / y = 37 / 35 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l549_54921


namespace NUMINAMATH_CALUDE_second_strongest_in_final_probability_l549_54919

/-- Represents a player in the tournament -/
structure Player where
  strength : ℕ

/-- Represents a tournament with 8 players -/
structure Tournament where
  players : Fin 8 → Player
  strength_ordered : ∀ i j, i < j → (players i).strength > (players j).strength

/-- The probability that the second strongest player reaches the final -/
def probability_second_strongest_in_final (t : Tournament) : ℚ :=
  4 / 7

/-- Theorem stating that the probability of the second strongest player
    reaching the final is 4/7 -/
theorem second_strongest_in_final_probability (t : Tournament) :
  probability_second_strongest_in_final t = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_second_strongest_in_final_probability_l549_54919


namespace NUMINAMATH_CALUDE_pen_color_theorem_l549_54916

-- Define the universe of pens
variable (Pen : Type)

-- Define the property of being in the box
variable (inBox : Pen → Prop)

-- Define the property of being blue
variable (isBlue : Pen → Prop)

-- Theorem statement
theorem pen_color_theorem :
  (¬ ∀ p : Pen, inBox p → isBlue p) →
  ((∃ p : Pen, inBox p ∧ ¬ isBlue p) ∧
   (¬ ∀ p : Pen, inBox p → isBlue p)) :=
by sorry

end NUMINAMATH_CALUDE_pen_color_theorem_l549_54916


namespace NUMINAMATH_CALUDE_race_start_relation_l549_54991

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race conditions -/
structure RaceCondition where
  a : Runner
  b : Runner
  c : Runner
  race_length : ℝ
  a_c_start : ℝ
  b_c_start : ℝ

/-- Theorem stating the relation between the starts given by runners -/
theorem race_start_relation (cond : RaceCondition) 
  (h1 : cond.race_length = 1000)
  (h2 : cond.a_c_start = 600)
  (h3 : cond.b_c_start = 428.57) :
  ∃ (a_b_start : ℝ), a_b_start = 750 ∧ 
    (cond.race_length - a_b_start) / cond.race_length = 
    (cond.race_length - cond.b_c_start) / (cond.race_length - cond.a_c_start) :=
by sorry

end NUMINAMATH_CALUDE_race_start_relation_l549_54991


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l549_54958

theorem arithmetic_geometric_mean_ratio 
  (x y : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_eq : ((x + y) / 2) + Real.sqrt (x * y) = y - x) : 
  x / y = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l549_54958


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l549_54992

/-- Given an arithmetic sequence with first term 3 and second term 7,
    prove that the 5th term is 19. -/
theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                            -- first term is 3
    a 1 = 7 →                            -- second term is 7
    a 4 = 19 :=                          -- 5th term (index 4) is 19
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l549_54992


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l549_54938

theorem polar_to_cartesian (θ : Real) (x y : Real) :
  x = (2 * Real.sin θ + 4 * Real.cos θ) * Real.cos θ ∧
  y = (2 * Real.sin θ + 4 * Real.cos θ) * Real.sin θ →
  (x - 2)^2 + (y - 1)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l549_54938


namespace NUMINAMATH_CALUDE_judy_pencil_cost_l549_54965

-- Define the given conditions
def pencils_per_week : ℕ := 10
def days_per_week : ℕ := 5
def pencils_per_pack : ℕ := 30
def cost_per_pack : ℕ := 4
def total_days : ℕ := 45

-- Define the theorem
theorem judy_pencil_cost :
  let pencils_per_day : ℚ := pencils_per_week / days_per_week
  let total_pencils : ℚ := pencils_per_day * total_days
  let packs_needed : ℚ := total_pencils / pencils_per_pack
  let total_cost : ℚ := packs_needed * cost_per_pack
  total_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_judy_pencil_cost_l549_54965


namespace NUMINAMATH_CALUDE_five_classrooms_formed_l549_54925

/-- Represents the problem of forming classrooms with equal numbers of boys and girls -/
def ClassroomFormation (total_boys total_girls students_per_class : ℕ) : Prop :=
  ∃ (num_classrooms : ℕ),
    -- Each classroom has an equal number of boys and girls
    ∃ (boys_per_class : ℕ),
      2 * boys_per_class = students_per_class ∧
      -- The total number of students in all classrooms doesn't exceed the available students
      num_classrooms * students_per_class ≤ total_boys + total_girls ∧
      -- All boys and girls are assigned to classrooms
      num_classrooms * boys_per_class ≤ total_boys ∧
      num_classrooms * boys_per_class ≤ total_girls ∧
      -- This is the maximum number of classrooms possible
      ∀ (larger_num_classrooms : ℕ), larger_num_classrooms > num_classrooms →
        larger_num_classrooms * boys_per_class > total_boys ∨
        larger_num_classrooms * boys_per_class > total_girls

/-- The main theorem stating that 5 classrooms can be formed under the given conditions -/
theorem five_classrooms_formed :
  ClassroomFormation 56 44 25 ∧
  ∀ (n : ℕ), ClassroomFormation 56 44 25 → n ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_five_classrooms_formed_l549_54925


namespace NUMINAMATH_CALUDE_only_one_is_ultra_prime_l549_54981

-- Define f(n) as the sum of all divisors of n
def f (n : ℕ) : ℕ := sorry

-- Define g(n) = n + f(n)
def g (n : ℕ) : ℕ := n + f n

-- Define ultra-prime
def is_ultra_prime (n : ℕ) : Prop := f (g n) = 2 * n + 3

-- Theorem statement
theorem only_one_is_ultra_prime :
  ∃! (n : ℕ), n < 100 ∧ is_ultra_prime n :=
sorry

end NUMINAMATH_CALUDE_only_one_is_ultra_prime_l549_54981


namespace NUMINAMATH_CALUDE_cylinder_volume_l549_54968

/-- The volume of a cylinder with base radius 2 cm and height h cm is 4πh cm³ -/
theorem cylinder_volume (h : ℝ) : 
  let r : ℝ := 2
  let V : ℝ := π * r^2 * h
  V = 4 * π * h := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l549_54968


namespace NUMINAMATH_CALUDE_no_rational_solutions_for_equation_l549_54917

theorem no_rational_solutions_for_equation :
  ¬ ∃ (x y z : ℚ), 11 = x^5 + 2*y^5 + 5*z^5 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_for_equation_l549_54917


namespace NUMINAMATH_CALUDE_train_crossing_time_l549_54987

/-- Proves that a train with given length and speed takes a specific time to cross a pole --/
theorem train_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (h1 : train_length = 125) 
  (h2 : train_speed_kmh = 90) : 
  train_length / (train_speed_kmh * 1000 / 3600) = 5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l549_54987


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l549_54908

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 5 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l549_54908


namespace NUMINAMATH_CALUDE_worker_hours_per_day_l549_54934

/-- Represents a factory worker's productivity and work schedule -/
structure Worker where
  widgets_per_hour : ℕ
  days_per_week : ℕ
  widgets_per_week : ℕ

/-- Calculates the number of hours a worker works per day -/
def hours_per_day (w : Worker) : ℚ :=
  (w.widgets_per_week : ℚ) / (w.widgets_per_hour : ℚ) / (w.days_per_week : ℚ)

/-- Theorem stating that a worker with given productivity and output works 8 hours per day -/
theorem worker_hours_per_day (w : Worker)
    (h1 : w.widgets_per_hour = 20)
    (h2 : w.days_per_week = 5)
    (h3 : w.widgets_per_week = 800) :
    hours_per_day w = 8 := by
  sorry

end NUMINAMATH_CALUDE_worker_hours_per_day_l549_54934


namespace NUMINAMATH_CALUDE_regular_polygon_945_diagonals_has_45_sides_l549_54944

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 945 diagonals has 45 sides -/
theorem regular_polygon_945_diagonals_has_45_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 945 → n = 45 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_945_diagonals_has_45_sides_l549_54944


namespace NUMINAMATH_CALUDE_negation_equivalence_l549_54931

theorem negation_equivalence :
  (¬ ∃ x : ℤ, x^2 + 2*x + 1 ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l549_54931


namespace NUMINAMATH_CALUDE_money_distribution_l549_54939

theorem money_distribution (p q r : ℚ) : 
  p + q + r = 5000 →
  r = (2/3) * (p + q) →
  r = 2000 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l549_54939


namespace NUMINAMATH_CALUDE_percent_only_cat_owners_l549_54930

/-- Given a school with the following student statistics:
  * total_students: The total number of students
  * cat_owners: The number of students who own cats
  * dog_owners: The number of students who own dogs
  * both_owners: The number of students who own both cats and dogs

  This theorem proves that the percentage of students who own only cats is 8%.
-/
theorem percent_only_cat_owners
  (total_students : ℕ)
  (cat_owners : ℕ)
  (dog_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 80)
  (h3 : dog_owners = 150)
  (h4 : both_owners = 40) :
  (((cat_owners - both_owners : ℚ) / total_students) * 100 = 8) := by
  sorry

end NUMINAMATH_CALUDE_percent_only_cat_owners_l549_54930


namespace NUMINAMATH_CALUDE_jimmy_notebooks_l549_54922

/-- The number of notebooks Jimmy bought -/
def num_notebooks : ℕ := sorry

/-- The cost of one pen -/
def pen_cost : ℕ := 1

/-- The cost of one notebook -/
def notebook_cost : ℕ := 3

/-- The cost of one folder -/
def folder_cost : ℕ := 5

/-- The number of pens Jimmy bought -/
def num_pens : ℕ := 3

/-- The number of folders Jimmy bought -/
def num_folders : ℕ := 2

/-- The amount Jimmy paid with -/
def paid_amount : ℕ := 50

/-- The amount Jimmy received as change -/
def change_amount : ℕ := 25

theorem jimmy_notebooks :
  num_notebooks = 4 :=
sorry

end NUMINAMATH_CALUDE_jimmy_notebooks_l549_54922


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l549_54911

/-- The line mx-y+2m+1=0 passes through the point (-2, 1) for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), m * (-2) - 1 + 2 * m + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l549_54911


namespace NUMINAMATH_CALUDE_wall_clock_ring_interval_l549_54974

/-- Given a wall clock that rings 6 times a day at equal intervals, 
    prove that the time between two consecutive rings is 288 minutes. -/
theorem wall_clock_ring_interval (rings_per_day : ℕ) (minutes_per_day : ℕ) : 
  rings_per_day = 6 → 
  minutes_per_day = 1440 → 
  (minutes_per_day / (rings_per_day - 1) : ℚ) = 288 := by
  sorry

#check wall_clock_ring_interval

end NUMINAMATH_CALUDE_wall_clock_ring_interval_l549_54974


namespace NUMINAMATH_CALUDE_intersection_A_not_B_range_of_a_l549_54999

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x ≥ 2}

-- Define the complement of B
def not_B : Set ℝ := {x | x < 2}

-- Define the set C
def C (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Theorem 1: A ∩ (¬ᵣB) = {x | 1 < x < 2}
theorem intersection_A_not_B : A ∩ not_B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 2: The range of a is [1, +∞) when A ∩ C = C
theorem range_of_a (a : ℝ) : A ∩ C a = C a ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_not_B_range_of_a_l549_54999


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l549_54956

theorem mod_congruence_unique_solution :
  ∃! n : ℕ, n ≤ 6 ∧ n ≡ -7845 [ZMOD 7] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l549_54956


namespace NUMINAMATH_CALUDE_solve_equation_l549_54973

theorem solve_equation (x : ℝ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l549_54973


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_difference_of_roots_x2_minus_7x_plus_9_l549_54967

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 →
  |x₁ - x₂| = Real.sqrt ((b^2 - 4*a*c) / a^2) := by
sorry

theorem difference_of_roots_x2_minus_7x_plus_9 :
  let x₁ := (7 + Real.sqrt 13) / 2
  let x₂ := (7 - Real.sqrt 13) / 2
  x₁^2 - 7*x₁ + 9 = 0 ∧ x₂^2 - 7*x₂ + 9 = 0 →
  |x₁ - x₂| = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_difference_of_roots_x2_minus_7x_plus_9_l549_54967


namespace NUMINAMATH_CALUDE_exists_right_triangle_with_different_colors_l549_54972

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define a point in the coordinate plane
structure Point where
  x : ℤ
  y : ℤ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the condition that all three colors are used
axiom all_colors_used : 
  ∃ p1 p2 p3 : Point, coloring p1 ≠ coloring p2 ∧ coloring p2 ≠ coloring p3 ∧ coloring p3 ≠ coloring p1

-- Define a right triangle
def is_right_triangle (p1 p2 p3 : Point) : Prop := sorry

-- Theorem statement
theorem exists_right_triangle_with_different_colors :
  ∃ p1 p2 p3 : Point, 
    is_right_triangle p1 p2 p3 ∧ 
    coloring p1 ≠ coloring p2 ∧ 
    coloring p2 ≠ coloring p3 ∧ 
    coloring p3 ≠ coloring p1 := by sorry

end NUMINAMATH_CALUDE_exists_right_triangle_with_different_colors_l549_54972


namespace NUMINAMATH_CALUDE_tg_plus_ctg_l549_54915

theorem tg_plus_ctg (x : ℝ) (h : (1 / Real.cos x) - (1 / Real.sin x) = Real.sqrt 35) :
  (Real.tan x + (1 / Real.tan x) = 7) ∨ (Real.tan x + (1 / Real.tan x) = -5) := by
sorry

end NUMINAMATH_CALUDE_tg_plus_ctg_l549_54915


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l549_54980

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (1, -2)
  parallel a b → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l549_54980


namespace NUMINAMATH_CALUDE_election_results_l549_54970

/-- Election results theorem -/
theorem election_results 
  (vote_percentage_A : ℝ) 
  (vote_percentage_B : ℝ) 
  (vote_percentage_C : ℝ) 
  (vote_percentage_D : ℝ) 
  (majority_difference : ℕ) 
  (h1 : vote_percentage_A = 0.45) 
  (h2 : vote_percentage_B = 0.30) 
  (h3 : vote_percentage_C = 0.20) 
  (h4 : vote_percentage_D = 0.05) 
  (h5 : vote_percentage_A + vote_percentage_B + vote_percentage_C + vote_percentage_D = 1) 
  (h6 : majority_difference = 1620) : 
  ∃ (total_votes : ℕ), 
    total_votes = 10800 ∧ 
    (vote_percentage_A * total_votes : ℝ) = 4860 ∧ 
    (vote_percentage_B * total_votes : ℝ) = 3240 ∧ 
    (vote_percentage_C * total_votes : ℝ) = 2160 ∧ 
    (vote_percentage_D * total_votes : ℝ) = 540 ∧ 
    (vote_percentage_A * total_votes - vote_percentage_B * total_votes : ℝ) = majority_difference :=
by sorry


end NUMINAMATH_CALUDE_election_results_l549_54970


namespace NUMINAMATH_CALUDE_largest_three_digit_number_with_gcd_condition_l549_54924

theorem largest_three_digit_number_with_gcd_condition :
  ∃ (x : ℕ), 
    x ≤ 990 ∧ 
    100 ≤ x ∧ 
    x % 3 = 0 ∧
    Nat.gcd 15 (Nat.gcd x 20) = 5 ∧
    ∀ (y : ℕ), 
      100 ≤ y ∧ 
      y ≤ 999 ∧ 
      y % 3 = 0 ∧ 
      Nat.gcd 15 (Nat.gcd y 20) = 5 → 
      y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_with_gcd_condition_l549_54924


namespace NUMINAMATH_CALUDE_calc_1_calc_2_calc_3_calc_4_l549_54977

-- (1) 327 + 46 - 135 = 238
theorem calc_1 : 327 + 46 - 135 = 238 := by sorry

-- (2) 1000 - 582 - 128 = 290
theorem calc_2 : 1000 - 582 - 128 = 290 := by sorry

-- (3) (124 - 62) × 6 = 372
theorem calc_3 : (124 - 62) * 6 = 372 := by sorry

-- (4) 500 - 400 ÷ 5 = 420
theorem calc_4 : 500 - 400 / 5 = 420 := by sorry

end NUMINAMATH_CALUDE_calc_1_calc_2_calc_3_calc_4_l549_54977


namespace NUMINAMATH_CALUDE_novel_pages_l549_54986

theorem novel_pages (planned_days : ℕ) (actual_days : ℕ) (extra_pages_per_day : ℕ) 
  (h1 : planned_days = 20)
  (h2 : actual_days = 15)
  (h3 : extra_pages_per_day = 20) : 
  (planned_days * ((actual_days * extra_pages_per_day) / (planned_days - actual_days))) = 1200 :=
by sorry

end NUMINAMATH_CALUDE_novel_pages_l549_54986


namespace NUMINAMATH_CALUDE_test_score_calculation_l549_54951

/-- Calculates the total score for a test given the total number of problems,
    points for correct answers, points deducted for wrong answers,
    and the number of wrong answers. -/
def calculateScore (totalProblems : ℕ) (pointsPerCorrect : ℕ) (pointsPerWrong : ℕ) (wrongAnswers : ℕ) : ℤ :=
  (totalProblems - wrongAnswers : ℤ) * pointsPerCorrect - wrongAnswers * pointsPerWrong

/-- Theorem stating that for a test with 25 problems, 4 points for each correct answer,
    1 point deducted for each wrong answer, and 3 wrong answers, the total score is 85. -/
theorem test_score_calculation :
  calculateScore 25 4 1 3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l549_54951


namespace NUMINAMATH_CALUDE_sqrt_8_same_type_as_sqrt_2_l549_54982

-- Define what it means for a real number to be of the same type as √2
def same_type_as_sqrt_2 (x : ℝ) : Prop :=
  ∃ (q : ℚ), x = q * Real.sqrt 2

-- State the theorem
theorem sqrt_8_same_type_as_sqrt_2 :
  same_type_as_sqrt_2 (Real.sqrt 8) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_same_type_as_sqrt_2_l549_54982


namespace NUMINAMATH_CALUDE_cubic_equation_product_l549_54979

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016)
  (h₄ : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃))
  (h₅ : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -671/336 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_product_l549_54979


namespace NUMINAMATH_CALUDE_wage_decrease_hours_increase_l549_54975

theorem wage_decrease_hours_increase (W H : ℝ) (W_new H_new : ℝ) :
  W > 0 → H > 0 →
  W_new = 0.8 * W →
  W * H = W_new * H_new →
  (H_new - H) / H = 0.25 :=
sorry

end NUMINAMATH_CALUDE_wage_decrease_hours_increase_l549_54975
