import Mathlib

namespace NUMINAMATH_CALUDE_factor_expression_l2059_205986

theorem factor_expression (y : ℝ) : 3 * y^3 - 75 * y = 3 * y * (y + 5) * (y - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2059_205986


namespace NUMINAMATH_CALUDE_sum_of_digits_of_special_number_l2059_205926

/-- The least 6-digit number -/
def least_six_digit : ℕ := 100000

/-- Function to check if a number is 6-digit -/
def is_six_digit (n : ℕ) : Prop := n ≥ least_six_digit ∧ n < 1000000

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem sum_of_digits_of_special_number :
  ∃ n : ℕ,
    is_six_digit n ∧
    n % 4 = 2 ∧
    n % 610 = 2 ∧
    n % 15 = 2 ∧
    (∀ m : ℕ, m < n → ¬(is_six_digit m ∧ m % 4 = 2 ∧ m % 610 = 2 ∧ m % 15 = 2)) ∧
    sum_of_digits n = 17 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_special_number_l2059_205926


namespace NUMINAMATH_CALUDE_candy_distribution_l2059_205994

theorem candy_distribution (total : ℕ) (portions : ℕ) (increment : ℕ) (smallest : ℕ) : 
  total = 40 →
  portions = 4 →
  increment = 2 →
  (smallest + (smallest + increment) + (smallest + 2 * increment) + (smallest + 3 * increment) = total) →
  smallest = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2059_205994


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2059_205977

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2059_205977


namespace NUMINAMATH_CALUDE_max_value_of_z_l2059_205951

-- Define the system of inequalities
def system (x y : ℝ) : Prop :=
  x + y ≤ 4 ∧ y - 2*x + 2 ≤ 0 ∧ y ≥ 0

-- Define z as a function of x and y
def z (x y : ℝ) : ℝ := x + 2*y

-- Theorem statement
theorem max_value_of_z :
  ∃ (x y : ℝ), system x y ∧ z x y = 6 ∧
  ∀ (x' y' : ℝ), system x' y' → z x' y' ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2059_205951


namespace NUMINAMATH_CALUDE_video_rental_percentage_l2059_205993

theorem video_rental_percentage (a : ℕ) : 
  let action := a
  let drama := 5 * a
  let comedy := 10 * a
  let total := action + drama + comedy
  (comedy : ℚ) / total * 100 = 62.5 := by
sorry

end NUMINAMATH_CALUDE_video_rental_percentage_l2059_205993


namespace NUMINAMATH_CALUDE_vanessa_score_is_40_5_l2059_205939

/-- Calculates Vanessa's score in a basketball game. -/
def vanessaScore (totalTeamScore : ℝ) (numPlayers : ℕ) (otherPlayersAverage : ℝ) : ℝ :=
  totalTeamScore - (otherPlayersAverage * (numPlayers - 1 : ℝ))

/-- Proves that Vanessa's score is 40.5 points given the conditions of the game. -/
theorem vanessa_score_is_40_5 :
  vanessaScore 72 8 4.5 = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_score_is_40_5_l2059_205939


namespace NUMINAMATH_CALUDE_candy_redistribution_l2059_205997

/-- Represents the distribution of candies in boxes -/
def CandyDistribution := List Nat

/-- An operation on the candy distribution -/
def redistribute (dist : CandyDistribution) (i j : Nat) : CandyDistribution :=
  sorry

/-- Checks if a distribution is valid (total candies = n^2) -/
def isValidDistribution (n : Nat) (dist : CandyDistribution) : Prop :=
  sorry

/-- Checks if a distribution is the goal distribution (n candies in each box) -/
def isGoalDistribution (n : Nat) (dist : CandyDistribution) : Prop :=
  sorry

/-- Checks if a number is a power of 2 -/
def isPowerOfTwo (n : Nat) : Prop :=
  sorry

theorem candy_redistribution (n : Nat) :
  (n > 2) →
  (∀ (init : CandyDistribution), isValidDistribution n init →
    ∃ (final : CandyDistribution), isGoalDistribution n final ∧
      ∃ (ops : List (Nat × Nat)), final = ops.foldl (fun d (i, j) => redistribute d i j) init) ↔
  isPowerOfTwo n :=
sorry

end NUMINAMATH_CALUDE_candy_redistribution_l2059_205997


namespace NUMINAMATH_CALUDE_tangent_circles_t_value_l2059_205920

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y t : ℝ) : Prop := (x - t)^2 + y^2 = 1

-- Define the condition of external tangency
def externally_tangent (t : ℝ) : Prop := ∃ x y : ℝ, circle1 x y ∧ circle2 x y t

-- Theorem statement
theorem tangent_circles_t_value :
  ∀ t : ℝ, externally_tangent t → (t = 3 ∨ t = -3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_t_value_l2059_205920


namespace NUMINAMATH_CALUDE_orange_apple_difference_l2059_205932

/-- The number of apples Leif has -/
def num_apples : ℕ := 14

/-- The number of dozens of oranges Leif has -/
def dozens_oranges : ℕ := 2

/-- The number of oranges in a dozen -/
def oranges_per_dozen : ℕ := 12

/-- The total number of oranges Leif has -/
def num_oranges : ℕ := dozens_oranges * oranges_per_dozen

theorem orange_apple_difference :
  num_oranges - num_apples = 10 := by sorry

end NUMINAMATH_CALUDE_orange_apple_difference_l2059_205932


namespace NUMINAMATH_CALUDE_sin_sum_specific_angles_l2059_205943

theorem sin_sum_specific_angles (α β : Real) : 
  0 < α ∧ α < Real.pi → 
  0 < β ∧ β < Real.pi → 
  Real.cos α = -1/2 → 
  Real.sin β = Real.sqrt 3 / 2 → 
  Real.sin (α + β) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_specific_angles_l2059_205943


namespace NUMINAMATH_CALUDE_smoothie_size_l2059_205990

-- Define the constants from the problem
def packet_size : ℝ := 3
def water_per_packet : ℝ := 15
def total_smoothies : ℝ := 150
def total_packets : ℝ := 180

-- Define the theorem
theorem smoothie_size :
  let packets_per_smoothie := total_packets / total_smoothies
  let mix_per_smoothie := packets_per_smoothie * packet_size
  let water_per_smoothie := packets_per_smoothie * water_per_packet
  mix_per_smoothie + water_per_smoothie = 21.6 := by
sorry

end NUMINAMATH_CALUDE_smoothie_size_l2059_205990


namespace NUMINAMATH_CALUDE_shared_savings_theorem_l2059_205975

/-- Calculates the monthly savings per person for a shared down payment -/
def monthly_savings_per_person (down_payment : ℕ) (years : ℕ) : ℕ :=
  down_payment / (years * 12) / 2

/-- Theorem: Two people saving equally for a $108,000 down payment over 3 years each save $1,500 per month -/
theorem shared_savings_theorem :
  monthly_savings_per_person 108000 3 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_shared_savings_theorem_l2059_205975


namespace NUMINAMATH_CALUDE_mary_gave_one_blue_crayon_l2059_205938

/-- Given that Mary initially has 5 green crayons and 8 blue crayons,
    gives 3 green crayons to Becky, and has 9 crayons left afterwards,
    prove that Mary gave 1 blue crayon to Becky. -/
theorem mary_gave_one_blue_crayon 
  (initial_green : Nat) 
  (initial_blue : Nat)
  (green_given : Nat)
  (total_left : Nat)
  (h1 : initial_green = 5)
  (h2 : initial_blue = 8)
  (h3 : green_given = 3)
  (h4 : total_left = 9)
  (h5 : total_left = initial_green + initial_blue - green_given - blue_given)
  : blue_given = 1 := by
  sorry

#check mary_gave_one_blue_crayon

end NUMINAMATH_CALUDE_mary_gave_one_blue_crayon_l2059_205938


namespace NUMINAMATH_CALUDE_comics_reassembly_l2059_205970

theorem comics_reassembly (pages_per_comic : ℕ) (torn_pages : ℕ) (untorn_comics : ℕ) : 
  pages_per_comic = 25 →
  torn_pages = 150 →
  untorn_comics = 5 →
  (torn_pages / pages_per_comic + untorn_comics : ℕ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_comics_reassembly_l2059_205970


namespace NUMINAMATH_CALUDE_sector_perimeter_l2059_205914

/-- Given a sector with area 2 and central angle 4 radians, its perimeter is 6. -/
theorem sector_perimeter (A : ℝ) (θ : ℝ) (r : ℝ) (P : ℝ) : 
  A = 2 → θ = 4 → A = (1/2) * r^2 * θ → P = r * θ + 2 * r → P = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l2059_205914


namespace NUMINAMATH_CALUDE_bathroom_tiles_count_l2059_205963

-- Define the bathroom dimensions in feet
def bathroom_length : ℝ := 10
def bathroom_width : ℝ := 6

-- Define the tile side length in inches
def tile_side : ℝ := 6

-- Define the conversion factor from feet to inches
def inches_per_foot : ℝ := 12

theorem bathroom_tiles_count :
  (bathroom_length * inches_per_foot) * (bathroom_width * inches_per_foot) / (tile_side * tile_side) = 240 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_tiles_count_l2059_205963


namespace NUMINAMATH_CALUDE_football_game_attendance_l2059_205904

theorem football_game_attendance (S : ℕ) 
  (hMonday : ℕ → ℕ := λ x => x - 20)
  (hWednesday : ℕ → ℕ := λ x => x + 50)
  (hFriday : ℕ → ℕ := λ x => x * 2 - 20)
  (hExpected : ℕ := 350)
  (hActual : ℕ := hExpected + 40)
  (hTotal : ℕ → ℕ := λ x => x + hMonday x + hWednesday (hMonday x) + hFriday x) :
  hTotal S = hActual → S = 80 := by
sorry

end NUMINAMATH_CALUDE_football_game_attendance_l2059_205904


namespace NUMINAMATH_CALUDE_fraction_multiplication_equality_l2059_205903

theorem fraction_multiplication_equality : 
  (11/12 - 7/6 + 3/4 - 13/24) * (-48) = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equality_l2059_205903


namespace NUMINAMATH_CALUDE_network_coloring_l2059_205901

/-- A node in the network --/
structure Node where
  lines : Finset (Fin 10)

/-- A network of lines on a plane --/
structure Network where
  nodes : Finset Node
  adjacent : Node → Node → Prop

/-- A coloring of the network --/
def Coloring (n : Network) := Node → Fin 15

/-- A valid coloring of the network --/
def ValidColoring (n : Network) (c : Coloring n) : Prop :=
  ∀ (node1 node2 : Node), n.adjacent node1 node2 → c node1 ≠ c node2

/-- The main theorem: any network can be colored with at most 15 colors --/
theorem network_coloring (n : Network) : ∃ (c : Coloring n), ValidColoring n c := by
  sorry

end NUMINAMATH_CALUDE_network_coloring_l2059_205901


namespace NUMINAMATH_CALUDE_one_correct_proposition_l2059_205933

theorem one_correct_proposition : 
  (∃! n : Nat, n = 1 ∧ 
    ((∀ a b : ℝ, a > abs b → a^2 > b^2) ∧ 
     ¬(∀ a b c d : ℝ, a > b ∧ c > d → a - c > b - d) ∧
     ¬(∀ a b c d : ℝ, a > b ∧ c > d → a * c > b * d) ∧
     ¬(∀ a b c : ℝ, a > b ∧ b > 0 → c / a > c / b))) :=
by sorry

end NUMINAMATH_CALUDE_one_correct_proposition_l2059_205933


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2059_205929

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2059_205929


namespace NUMINAMATH_CALUDE_jon_website_earnings_l2059_205978

/-- Calculates Jon's earnings from his website in a 30-day month -/
theorem jon_website_earnings : 
  let pay_per_visit : ℚ := 0.1
  let visits_per_hour : ℕ := 50
  let hours_per_day : ℕ := 24
  let days_in_month : ℕ := 30
  (pay_per_visit * visits_per_hour * hours_per_day * days_in_month : ℚ) = 3600 := by
  sorry

end NUMINAMATH_CALUDE_jon_website_earnings_l2059_205978


namespace NUMINAMATH_CALUDE_max_altitude_triangle_ABC_l2059_205966

/-- Given a triangle ABC with the specified conditions, the maximum altitude on side BC is √3 + 1 -/
theorem max_altitude_triangle_ABC (A B C : Real) (h1 : 3 * (Real.sin B ^ 2 + Real.sin C ^ 2 - Real.sin A ^ 2) = 2 * Real.sqrt 3 * Real.sin B * Real.sin C) 
  (h2 : (1 / 2) * Real.sin A * (Real.sin B / Real.sin A) * (Real.sin C / Real.sin A) = Real.sqrt 6 + Real.sqrt 2) :
  ∃ (h : Real), h ≤ Real.sqrt 3 + 1 ∧ 
    ∀ (h' : Real), (∃ (a b c : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      3 * (Real.sin (b / a) ^ 2 + Real.sin (c / a) ^ 2 - Real.sin 1 ^ 2) = 2 * Real.sqrt 3 * Real.sin (b / a) * Real.sin (c / a) ∧
      (1 / 2) * a * b * Real.sin (c / a) = Real.sqrt 6 + Real.sqrt 2 ∧
      h' = (2 * (Real.sqrt 6 + Real.sqrt 2)) / c) → 
    h' ≤ h :=
by sorry

end NUMINAMATH_CALUDE_max_altitude_triangle_ABC_l2059_205966


namespace NUMINAMATH_CALUDE_vector_expression_l2059_205983

/-- Given vectors in ℝ², prove that c = 3a + 2b -/
theorem vector_expression (a b c : ℝ × ℝ) : 
  a = (1, -1) → b = (-1, 2) → c = (1, 1) → c = 3 • a + 2 • b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l2059_205983


namespace NUMINAMATH_CALUDE_prove_A_equals_five_l2059_205956

/-- Given that 14A and B73 are three-digit numbers, 14A + B73 = 418, and A and B are single digits, prove that A = 5 -/
theorem prove_A_equals_five (A B : ℕ) : 
  (100 ≤ 14 * A) ∧ (14 * A < 1000) ∧  -- 14A is a three-digit number
  (100 ≤ B * 100 + 73) ∧ (B * 100 + 73 < 1000) ∧  -- B73 is a three-digit number
  (14 * A + B * 100 + 73 = 418) ∧  -- 14A + B73 = 418
  (A < 10) ∧ (B < 10) →  -- A and B are single digits
  A = 5 := by sorry

end NUMINAMATH_CALUDE_prove_A_equals_five_l2059_205956


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2059_205900

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2059_205900


namespace NUMINAMATH_CALUDE_angle_value_l2059_205902

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem angle_value (a : ℕ → ℝ) (α : ℝ) :
  is_geometric_sequence a →
  (∀ x : ℝ, x^2 - 2*x*Real.sin α - Real.sqrt 3*Real.sin α = 0 ↔ (x = a 1 ∨ x = a 8)) →
  (a 1 + a 8)^2 = 2*a 3*a 6 + 6 →
  0 < α ∧ α < Real.pi/2 →
  α = Real.pi/3 := by sorry

end NUMINAMATH_CALUDE_angle_value_l2059_205902


namespace NUMINAMATH_CALUDE_base_16_digits_for_5_digit_base_4_l2059_205996

theorem base_16_digits_for_5_digit_base_4 (n : ℕ) (h : 256 ≤ n ∧ n ≤ 1023) :
  (Nat.log 16 n).succ = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_16_digits_for_5_digit_base_4_l2059_205996


namespace NUMINAMATH_CALUDE_function_positivity_implies_m_range_l2059_205972

/-- Given two functions f and g defined on real numbers, 
    prove that if at least one of f(x) or g(x) is positive for all real x,
    then the parameter m is in the open interval (0, 8) -/
theorem function_positivity_implies_m_range 
  (f g : ℝ → ℝ) 
  (m : ℝ) 
  (hf : f = fun x ↦ 2 * m * x^2 - 2 * (4 - m) * x + 1) 
  (hg : g = fun x ↦ m * x) 
  (h : ∀ x : ℝ, 0 < f x ∨ 0 < g x) : 
  0 < m ∧ m < 8 := by
  sorry

end NUMINAMATH_CALUDE_function_positivity_implies_m_range_l2059_205972


namespace NUMINAMATH_CALUDE_linear_mapping_midpoint_distance_l2059_205906

/-- Linear mapping from a segment of length 10 to a segment of length 5 -/
def LinearMapping (x y : ℝ) : Prop :=
  x / 10 = y / 5

/-- Theorem: In the given linear mapping, when x = 3, x + y = 4.5 -/
theorem linear_mapping_midpoint_distance (x y : ℝ) :
  LinearMapping x y → x = 3 → x + y = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_linear_mapping_midpoint_distance_l2059_205906


namespace NUMINAMATH_CALUDE_lake_distance_difference_l2059_205928

/-- The difference between the circumference and diameter of a circular lake -/
theorem lake_distance_difference (diameter : ℝ) (pi : ℝ) 
  (h1 : diameter = 2)
  (h2 : pi = 3.14) : 
  2 * pi * (diameter / 2) - diameter = 4.28 := by
  sorry

end NUMINAMATH_CALUDE_lake_distance_difference_l2059_205928


namespace NUMINAMATH_CALUDE_share_distribution_l2059_205962

theorem share_distribution (a b c d : ℝ) : 
  a + b + c + d = 1200 →
  a = (3/5) * (b + c + d) →
  b = (2/3) * (a + c + d) →
  c = (4/7) * (a + b + d) →
  a = 247.5 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l2059_205962


namespace NUMINAMATH_CALUDE_principal_calculation_l2059_205958

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  simple_interest / (rate * time)

/-- Theorem: Given the specified conditions, the principal is 44625 -/
theorem principal_calculation :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 1 / 100  -- 1% converted to decimal
  let time : ℕ := 9
  calculate_principal simple_interest rate time = 44625 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2059_205958


namespace NUMINAMATH_CALUDE_shorter_worm_length_l2059_205968

theorem shorter_worm_length (worm1_length worm2_length : Real) :
  worm1_length = 0.8 →
  worm2_length = worm1_length + 0.7 →
  min worm1_length worm2_length = 0.8 := by
sorry

end NUMINAMATH_CALUDE_shorter_worm_length_l2059_205968


namespace NUMINAMATH_CALUDE_variance_sum_random_nonrandom_l2059_205971

/-- A random function -/
def RandomFunction (α : Type*) := α → ℝ

/-- A non-random function -/
def NonRandomFunction (α : Type*) := α → ℝ

/-- Variance of a random function -/
noncomputable def variance (X : RandomFunction ℝ) (t : ℝ) : ℝ := sorry

/-- The sum of a random function and a non-random function -/
def sumFunction (X : RandomFunction ℝ) (φ : NonRandomFunction ℝ) : RandomFunction ℝ :=
  fun t => X t + φ t

/-- Theorem: The variance of the sum of a random function and a non-random function
    is equal to the variance of the random function -/
theorem variance_sum_random_nonrandom
  (X : RandomFunction ℝ) (φ : NonRandomFunction ℝ) (t : ℝ) :
  variance (sumFunction X φ) t = variance X t := by sorry

end NUMINAMATH_CALUDE_variance_sum_random_nonrandom_l2059_205971


namespace NUMINAMATH_CALUDE_exponent_addition_l2059_205905

theorem exponent_addition (a : ℝ) : a^3 + a^3 = 2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l2059_205905


namespace NUMINAMATH_CALUDE_inequality_proof_l2059_205918

theorem inequality_proof (a b c : ℝ) (ha : a = 31/32) (hb : b = Real.cos (1/4)) (hc : c = 4 * Real.sin (1/4)) : c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2059_205918


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2059_205991

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 5 = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 + m*y - 5 = 0 ∧ y = 5) :=
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2059_205991


namespace NUMINAMATH_CALUDE_tips_fraction_of_income_l2059_205967

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the total income of a waitress -/
def totalIncome (w : WaitressIncome) : ℚ :=
  w.salary + w.tips

/-- Theorem stating that if a waitress's tips are 3/4 of her salary, 
    then 3/7 of her total income comes from tips -/
theorem tips_fraction_of_income 
  (w : WaitressIncome) 
  (h : w.tips = 3/4 * w.salary) : 
  w.tips / totalIncome w = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_tips_fraction_of_income_l2059_205967


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l2059_205987

-- Define the polynomial Q
def Q (a b c d : ℝ) (x : ℝ) : ℝ := a + b * x + c * x^2 + d * x^3

-- State the theorem
theorem polynomial_uniqueness (a b c d : ℝ) :
  Q a b c d (-1) = 2 →
  Q a b c d = (fun x => x^3 - x^2 + x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l2059_205987


namespace NUMINAMATH_CALUDE_odd_multiple_of_nine_is_multiple_of_three_l2059_205942

theorem odd_multiple_of_nine_is_multiple_of_three :
  (∀ n : ℕ, 9 ∣ n → 3 ∣ n) →
  ∀ k : ℕ, Odd k → 9 ∣ k → 3 ∣ k :=
by
  sorry

end NUMINAMATH_CALUDE_odd_multiple_of_nine_is_multiple_of_three_l2059_205942


namespace NUMINAMATH_CALUDE_three_valid_rental_plans_l2059_205974

/-- Represents a rental plan for vehicles --/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental plan is valid for the given number of people --/
def isValidPlan (plan : RentalPlan) (totalPeople : ℕ) : Prop :=
  plan.typeA * 6 + plan.typeB * 4 = totalPeople

/-- Theorem stating that there are at least three different valid rental plans --/
theorem three_valid_rental_plans :
  ∃ (plan1 plan2 plan3 : RentalPlan),
    isValidPlan plan1 38 ∧
    isValidPlan plan2 38 ∧
    isValidPlan plan3 38 ∧
    plan1 ≠ plan2 ∧
    plan1 ≠ plan3 ∧
    plan2 ≠ plan3 := by
  sorry

end NUMINAMATH_CALUDE_three_valid_rental_plans_l2059_205974


namespace NUMINAMATH_CALUDE_system_equations_proof_l2059_205931

theorem system_equations_proof (x y a : ℝ) : 
  (3 * x + y = 2 + 3 * a) →
  (x + 3 * y = 2 + a) →
  (x + y < 0) →
  (a < -1) ∧ (|1 - a| + |a + 1/2| = 1/2 - 2 * a) := by
sorry

end NUMINAMATH_CALUDE_system_equations_proof_l2059_205931


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2059_205981

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) :
  (f' a b (-2/3) = 0 ∧ f' a b 1 = 0) →
  (a = -1/2 ∧ b = -2) ∧
  (∀ x : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f (-1/2) (-2) c x₁ = 0 ∧ f (-1/2) (-2) c x₂ = 0 ∧ f (-1/2) (-2) c x₃ = 0) →
    -22/27 < c ∧ c < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2059_205981


namespace NUMINAMATH_CALUDE_blue_spotted_fish_ratio_l2059_205988

theorem blue_spotted_fish_ratio (total_fish : ℕ) (blue_spotted_fish : ℕ) 
  (h1 : total_fish = 60) 
  (h2 : blue_spotted_fish = 10) : 
  (blue_spotted_fish : ℚ) / ((1 / 3 : ℚ) * total_fish) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_blue_spotted_fish_ratio_l2059_205988


namespace NUMINAMATH_CALUDE_money_duration_l2059_205976

def mowing_earnings : ℕ := 5
def weed_eating_earnings : ℕ := 58
def weekly_spending : ℕ := 7

theorem money_duration : 
  (mowing_earnings + weed_eating_earnings) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_duration_l2059_205976


namespace NUMINAMATH_CALUDE_n2o_molecular_weight_l2059_205940

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O -/
def n_nitrogen : ℕ := 2

/-- The number of oxygen atoms in N2O -/
def n_oxygen : ℕ := 1

/-- The number of moles of N2O -/
def n_moles : ℝ := 8

theorem n2o_molecular_weight :
  n_moles * (n_nitrogen * nitrogen_weight + n_oxygen * oxygen_weight) = 352.16 := by
  sorry

end NUMINAMATH_CALUDE_n2o_molecular_weight_l2059_205940


namespace NUMINAMATH_CALUDE_mary_pokemon_cards_l2059_205913

theorem mary_pokemon_cards (x : ℕ) : 
  x + 23 - 6 = 56 → x = 39 := by
sorry

end NUMINAMATH_CALUDE_mary_pokemon_cards_l2059_205913


namespace NUMINAMATH_CALUDE_log_not_always_decreasing_l2059_205941

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_not_always_decreasing :
  ¬ (∀ (a : ℝ), a > 0 → a ≠ 1 → 
    (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ < x₂ → log a x₁ > log a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_log_not_always_decreasing_l2059_205941


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_abs_l2059_205982

theorem quadratic_roots_sum_abs (p : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x - 6 = 0 ∧ y^2 + p*y - 6 = 0 ∧ |x| + |y| = 5) → 
  (p = 1 ∨ p = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_abs_l2059_205982


namespace NUMINAMATH_CALUDE_square_sum_equals_thirty_l2059_205916

theorem square_sum_equals_thirty (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a * b = 7) : 
  a^2 + b^2 = 30 := by sorry

end NUMINAMATH_CALUDE_square_sum_equals_thirty_l2059_205916


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2059_205998

-- Define the edge lengths
def edge_length_cube1 : ℚ := 4
def edge_length_cube2 : ℚ := 24  -- 2 feet = 24 inches

-- Define the volume ratio
def volume_ratio : ℚ := (edge_length_cube1 / edge_length_cube2) ^ 3

-- Theorem statement
theorem cube_volume_ratio :
  volume_ratio = 1 / 216 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2059_205998


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_3_equals_1_l2059_205950

theorem sqrt_2x_minus_3_equals_1 (x : ℝ) (h : x = 2) : Real.sqrt (2 * x - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_3_equals_1_l2059_205950


namespace NUMINAMATH_CALUDE_sale_price_is_twenty_l2059_205912

/-- The sale price of one bottle of detergent, given the number of loads per bottle and the cost per load when buying two bottles. -/
def sale_price (loads_per_bottle : ℕ) (cost_per_load : ℚ) : ℚ :=
  loads_per_bottle * cost_per_load

/-- Theorem stating that the sale price of one bottle of detergent is $20.00 -/
theorem sale_price_is_twenty :
  sale_price 80 (25 / 100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_is_twenty_l2059_205912


namespace NUMINAMATH_CALUDE_circle_equation_l2059_205989

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure CircleOnXAxis where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt2 : radius = Real.sqrt 2
  point_on_circle : (center.1 + 2)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle is either (x+1)^2 + y^2 = 2 or (x+3)^2 + y^2 = 2 -/
theorem circle_equation (c : CircleOnXAxis) :
  (∀ x y : ℝ, (x + 1)^2 + y^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∨
  (∀ x y : ℝ, (x + 3)^2 + y^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2059_205989


namespace NUMINAMATH_CALUDE_angle_sum_pi_half_l2059_205908

theorem angle_sum_pi_half (θ₁ θ₂ : Real) (h_acute₁ : 0 < θ₁ ∧ θ₁ < π/2) (h_acute₂ : 0 < θ₂ ∧ θ₂ < π/2)
  (h_eq : (Real.sin θ₁)^2020 / (Real.cos θ₂)^2018 + (Real.cos θ₁)^2020 / (Real.sin θ₂)^2018 = 1) :
  θ₁ + θ₂ = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_pi_half_l2059_205908


namespace NUMINAMATH_CALUDE_budget_this_year_l2059_205915

def cost_supply1 : ℕ := 13
def cost_supply2 : ℕ := 24
def remaining_last_year : ℕ := 6
def remaining_after_purchase : ℕ := 19

theorem budget_this_year :
  (cost_supply1 + cost_supply2 + remaining_after_purchase) - remaining_last_year = 50 := by
  sorry

end NUMINAMATH_CALUDE_budget_this_year_l2059_205915


namespace NUMINAMATH_CALUDE_fixed_points_of_quadratic_l2059_205957

/-- A quadratic function of the form f(x) = mx^2 - 2mx + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 3

/-- Theorem stating that (0, 3) and (2, 3) are fixed points of f for all non-zero m -/
theorem fixed_points_of_quadratic (m : ℝ) (h : m ≠ 0) :
  (f m 0 = 3) ∧ (f m 2 = 3) := by
  sorry

#check fixed_points_of_quadratic

end NUMINAMATH_CALUDE_fixed_points_of_quadratic_l2059_205957


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2059_205945

/-- Proves that the cost price of an article is 540 given the specified conditions -/
theorem cost_price_calculation (marked_up_price : ℝ → ℝ) (discounted_price : ℝ → ℝ) :
  (∀ x, marked_up_price x = x * 1.15) →
  (∀ x, discounted_price x = x * (1 - 0.2608695652173913)) →
  discounted_price (marked_up_price 540) = 459 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2059_205945


namespace NUMINAMATH_CALUDE_vessel_weight_percentage_l2059_205922

theorem vessel_weight_percentage (E P : ℝ) 
  (h1 : (1/2) * (E + P) = E + 0.42857142857142855 * P) : 
  (E / (E + P)) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_vessel_weight_percentage_l2059_205922


namespace NUMINAMATH_CALUDE_exists_x_sqrt_x_squared_neq_x_l2059_205955

theorem exists_x_sqrt_x_squared_neq_x : ∃ x : ℝ, Real.sqrt (x^2) ≠ x := by
  sorry

end NUMINAMATH_CALUDE_exists_x_sqrt_x_squared_neq_x_l2059_205955


namespace NUMINAMATH_CALUDE_solutions_rearrangements_l2059_205953

def word := "SOLUTIONS"

def vowels := ['O', 'I', 'U', 'O']
def consonants := ['S', 'L', 'T', 'N', 'S', 'S']

def vowel_arrangements := Nat.factorial 4 / Nat.factorial 2
def consonant_arrangements := Nat.factorial 6 / Nat.factorial 3

theorem solutions_rearrangements : 
  vowel_arrangements * consonant_arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_solutions_rearrangements_l2059_205953


namespace NUMINAMATH_CALUDE_pyramid_volume_l2059_205923

theorem pyramid_volume (total_surface_area : ℝ) (triangular_face_ratio : ℝ) :
  total_surface_area = 600 →
  triangular_face_ratio = 2 →
  ∃ (volume : ℝ),
    volume = (1/3) * (total_surface_area / (4 * triangular_face_ratio + 1)) * 
             (Real.sqrt ((4 * triangular_face_ratio + 1) * 
             (4 * triangular_face_ratio - 1) / (triangular_face_ratio^2))) *
             Real.sqrt (total_surface_area / (4 * triangular_face_ratio + 1)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2059_205923


namespace NUMINAMATH_CALUDE_root_minus_one_implies_k_equals_minus_two_l2059_205936

theorem root_minus_one_implies_k_equals_minus_two (k : ℝ) :
  ((-1 : ℝ)^2 - k*(-1) + 1 = 0) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_minus_one_implies_k_equals_minus_two_l2059_205936


namespace NUMINAMATH_CALUDE_odd_function_property_l2059_205995

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2059_205995


namespace NUMINAMATH_CALUDE_marks_garden_l2059_205925

theorem marks_garden (yellow purple green : ℕ) : 
  purple = yellow + (yellow * 4 / 5) →
  green = (yellow + purple) / 4 →
  yellow + purple + green = 35 →
  yellow = 10 := by
sorry

end NUMINAMATH_CALUDE_marks_garden_l2059_205925


namespace NUMINAMATH_CALUDE_max_distance_and_total_travel_l2059_205917

/-- Represents a car in the problem -/
structure Car where
  fuelCapacity : ℕ
  fuelEfficiency : ℕ

/-- Represents the problem setup -/
structure ProblemSetup where
  car : Car
  numCars : ℕ

/-- Defines the problem parameters -/
def problem : ProblemSetup :=
  { car := { fuelCapacity := 24, fuelEfficiency := 60 },
    numCars := 2 }

/-- Theorem stating the maximum distance and total distance traveled -/
theorem max_distance_and_total_travel (p : ProblemSetup)
  (h1 : p.numCars = 2)
  (h2 : p.car.fuelCapacity = 24)
  (h3 : p.car.fuelEfficiency = 60) :
  ∃ (maxDistance totalDistance : ℕ),
    maxDistance = 360 ∧
    totalDistance = 2160 ∧
    maxDistance ≤ (p.car.fuelCapacity * p.car.fuelEfficiency) / 2 ∧
    totalDistance = maxDistance * 2 * 3 := by
  sorry

#check max_distance_and_total_travel

end NUMINAMATH_CALUDE_max_distance_and_total_travel_l2059_205917


namespace NUMINAMATH_CALUDE_number_of_clerks_l2059_205911

/-- Proves that the number of clerks is 170 given the salary information -/
theorem number_of_clerks (total_avg : ℚ) (officer_avg : ℚ) (clerk_avg : ℚ) (num_officers : ℕ) :
  total_avg = 90 →
  officer_avg = 600 →
  clerk_avg = 84 →
  num_officers = 2 →
  ∃ (num_clerks : ℕ), 
    (num_officers * officer_avg + num_clerks * clerk_avg) / (num_officers + num_clerks) = total_avg ∧
    num_clerks = 170 := by
  sorry


end NUMINAMATH_CALUDE_number_of_clerks_l2059_205911


namespace NUMINAMATH_CALUDE_system_solution_l2059_205924

theorem system_solution : 
  ∃! (s : Set (ℝ × ℝ)), s = {(2, 4), (4, 2)} ∧
  ∀ (x y : ℝ), (x, y) ∈ s ↔ 
    ((x / y + y / x) * (x + y) = 15 ∧
     (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2059_205924


namespace NUMINAMATH_CALUDE_smallest_N_bound_l2059_205973

theorem smallest_N_bound (x : ℝ) (h : |x - 2| < 0.01) : 
  |x^2 - 4| < 0.0401 ∧ 
  ∀ ε > 0, ∃ y : ℝ, |y - 2| < 0.01 ∧ |y^2 - 4| ≥ 0.0401 - ε :=
sorry

end NUMINAMATH_CALUDE_smallest_N_bound_l2059_205973


namespace NUMINAMATH_CALUDE_ladder_distance_l2059_205937

theorem ladder_distance (ladder_length : ℝ) (elevation_angle : ℝ) (distance_to_wall : ℝ) :
  ladder_length = 9.2 →
  elevation_angle = 60 * π / 180 →
  distance_to_wall = ladder_length * Real.cos elevation_angle →
  distance_to_wall = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l2059_205937


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2059_205946

/-- A line passing through a point and parallel to another line -/
theorem line_through_point_parallel_to_line :
  let point : ℝ × ℝ := (2, 1)
  let parallel_line (x y : ℝ) := 2 * x - 3 * y + 1 = 0
  let target_line (x y : ℝ) := 2 * x - 3 * y - 1 = 0
  (∀ x y : ℝ, parallel_line x y ↔ y = 2/3 * x + 1/3) →
  (target_line point.1 point.2) ∧
  (∀ x y : ℝ, target_line x y ↔ y = 2/3 * x + 1/3) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2059_205946


namespace NUMINAMATH_CALUDE_reflection_of_point_A_l2059_205930

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflect a point over the origin -/
def reflectOverOrigin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

theorem reflection_of_point_A :
  let A : Point3D := { x := 2, y := 3, z := 4 }
  reflectOverOrigin A = { x := -2, y := -3, z := -4 } := by
  sorry

#check reflection_of_point_A

end NUMINAMATH_CALUDE_reflection_of_point_A_l2059_205930


namespace NUMINAMATH_CALUDE_complex_equation_roots_l2059_205969

theorem complex_equation_roots : 
  let z₁ : ℂ := 4 - 0.5 * I
  let z₂ : ℂ := -2 + 0.5 * I
  (z₁^2 - 2*z₁ = 7 - 3*I) ∧ (z₂^2 - 2*z₂ = 7 - 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l2059_205969


namespace NUMINAMATH_CALUDE_trivia_team_size_l2059_205960

theorem trivia_team_size (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  absent_members = 7 →
  points_per_member = 5 →
  total_points = 35 →
  absent_members + (total_points / points_per_member) = 14 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_size_l2059_205960


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2059_205984

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : 
  a 7 - a 8 = -8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2059_205984


namespace NUMINAMATH_CALUDE_min_value_and_t_value_l2059_205965

theorem min_value_and_t_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 2/b = 2) :
  (∃ (min : ℝ), min = 4 ∧ ∀ x y, x > 0 → y > 0 → 1/x + 2/y = 2 → 2*x + y ≥ min) ∧
  (∃ (t : ℝ), t = 6 ∧ 4^a = t ∧ 3^b = t) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_t_value_l2059_205965


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l2059_205907

theorem largest_divisor_of_n_squared_div_72 (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  12 = Nat.gcd n 72 ∧ ∀ m : ℕ, m ∣ n → m ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l2059_205907


namespace NUMINAMATH_CALUDE_divides_power_plus_one_l2059_205954

theorem divides_power_plus_one (n : ℕ) : (3 ^ (n + 1)) ∣ (2 ^ (3 ^ n) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divides_power_plus_one_l2059_205954


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2059_205935

/-- The common ratio of an infinite geometric series with given first term and sum -/
theorem geometric_series_common_ratio 
  (a : ℝ) 
  (S : ℝ) 
  (h1 : a = 400) 
  (h2 : S = 2500) 
  (h3 : a > 0) 
  (h4 : S > a) : 
  ∃ (r : ℝ), r = 21 / 25 ∧ S = a / (1 - r) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2059_205935


namespace NUMINAMATH_CALUDE_sum_of_products_nonzero_l2059_205985

/-- A 25x25 matrix with entries either 1 or -1 -/
def SignMatrix := Matrix (Fin 25) (Fin 25) Int

/-- Predicate to check if a matrix is a valid SignMatrix -/
def isValidSignMatrix (M : SignMatrix) : Prop :=
  ∀ i j, M i j = 1 ∨ M i j = -1

/-- Product of elements in a row -/
def rowProduct (M : SignMatrix) (i : Fin 25) : Int :=
  (List.range 25).foldl (fun acc j => acc * M i j) 1

/-- Product of elements in a column -/
def colProduct (M : SignMatrix) (j : Fin 25) : Int :=
  (List.range 25).foldl (fun acc i => acc * M i j) 1

/-- Sum of all row and column products -/
def sumOfProducts (M : SignMatrix) : Int :=
  (List.range 25).foldl (fun acc i => acc + rowProduct M i) 0 +
  (List.range 25).foldl (fun acc j => acc + colProduct M j) 0

theorem sum_of_products_nonzero (M : SignMatrix) (h : isValidSignMatrix M) :
  sumOfProducts M ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_products_nonzero_l2059_205985


namespace NUMINAMATH_CALUDE_product_evaluation_l2059_205944

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2059_205944


namespace NUMINAMATH_CALUDE_swim_time_ratio_l2059_205961

/-- Proves that the ratio of time taken to swim upstream to time taken to swim downstream is 2:1 -/
theorem swim_time_ratio (swim_speed : ℝ) (stream_speed : ℝ) 
  (h1 : swim_speed = 1.5) (h2 : stream_speed = 0.5) : 
  (swim_speed - stream_speed) / (swim_speed + stream_speed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_swim_time_ratio_l2059_205961


namespace NUMINAMATH_CALUDE_ones_digit_product_seven_consecutive_l2059_205947

theorem ones_digit_product_seven_consecutive (k : ℕ) (h1 : k > 0) (h2 : k % 5 = 1) : 
  (((k * (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6)) % 10) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_product_seven_consecutive_l2059_205947


namespace NUMINAMATH_CALUDE_solution_difference_l2059_205964

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

theorem solution_difference (x y : ℝ) :
  (floor x : ℝ) + frac y = 3.7 →
  frac x + (floor y : ℝ) = 8.2 →
  |x - y| = 5.5 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2059_205964


namespace NUMINAMATH_CALUDE_units_digit_product_l2059_205934

theorem units_digit_product (n : ℕ) : n = 3^401 * 7^402 * 23^403 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l2059_205934


namespace NUMINAMATH_CALUDE_bernoulli_inequalities_l2059_205919

theorem bernoulli_inequalities (α : ℝ) (n : ℕ) :
  (α > 0 ∧ n > 1 → (1 + α)^n > 1 + n * α) ∧
  (0 < α ∧ α ≤ 1 / n → (1 + α)^n < 1 + n * α + n^2 * α^2) := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequalities_l2059_205919


namespace NUMINAMATH_CALUDE_square_area_13m_l2059_205959

/-- The area of a square with side length 13 meters is 169 square meters. -/
theorem square_area_13m (side_length : ℝ) (h : side_length = 13) :
  side_length * side_length = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_13m_l2059_205959


namespace NUMINAMATH_CALUDE_cricket_average_score_l2059_205999

theorem cricket_average_score (total_matches : ℕ) (matches1 matches2 : ℕ) 
  (avg1 avg2 : ℚ) (h1 : total_matches = matches1 + matches2) 
  (h2 : matches1 = 2) (h3 : matches2 = 3) (h4 : avg1 = 30) (h5 : avg2 = 40) : 
  (matches1 * avg1 + matches2 * avg2) / total_matches = 36 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_score_l2059_205999


namespace NUMINAMATH_CALUDE_initial_number_problem_l2059_205921

theorem initial_number_problem : 
  let x : ℚ := 10
  ((x + 14) * 14 - 24) / 24 = 13 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_problem_l2059_205921


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2059_205979

theorem complex_modulus_problem (z : ℂ) (h : Complex.I * z = (1 - 2 * Complex.I)^2) : 
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2059_205979


namespace NUMINAMATH_CALUDE_total_dreams_calculation_l2059_205952

/-- The number of days in a year -/
def daysInYear : ℕ := 365

/-- The number of dreams per day in the current year -/
def dreamsPerDay : ℕ := 4

/-- The number of dreams in the current year -/
def dreamsThisYear : ℕ := dreamsPerDay * daysInYear

/-- The number of dreams in the previous year -/
def dreamsLastYear : ℕ := 2 * dreamsThisYear

/-- The total number of dreams over two years -/
def totalDreams : ℕ := dreamsLastYear + dreamsThisYear

theorem total_dreams_calculation :
  totalDreams = 4380 := by sorry

end NUMINAMATH_CALUDE_total_dreams_calculation_l2059_205952


namespace NUMINAMATH_CALUDE_negation_of_product_nonzero_implies_factors_nonzero_l2059_205909

theorem negation_of_product_nonzero_implies_factors_nonzero :
  (¬(∀ a b : ℝ, a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0)) ↔
  (∀ a b : ℝ, a * b = 0 → a = 0 ∨ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_product_nonzero_implies_factors_nonzero_l2059_205909


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2059_205948

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2059_205948


namespace NUMINAMATH_CALUDE_third_month_sales_l2059_205949

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_4 : ℕ := 7230
def sales_5 : ℕ := 6562
def sales_6 : ℕ := 6191
def average_sale : ℕ := 6700
def num_months : ℕ := 6

theorem third_month_sales :
  ∃ (sales_3 : ℕ),
    sales_3 = average_sale * num_months - (sales_1 + sales_2 + sales_4 + sales_5 + sales_6) ∧
    sales_3 = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sales_l2059_205949


namespace NUMINAMATH_CALUDE_balloons_distribution_l2059_205992

theorem balloons_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 236) (h2 : num_friends = 10) :
  total_balloons % num_friends = 6 := by
  sorry

end NUMINAMATH_CALUDE_balloons_distribution_l2059_205992


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l2059_205927

/-- The time at which a ball hits the ground when thrown upward -/
theorem ball_hit_ground_time : ∃ t : ℚ, t = 10/7 ∧ -4.9 * t^2 + 3.5 * t + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l2059_205927


namespace NUMINAMATH_CALUDE_pilot_course_cost_difference_pilot_course_cost_difference_holds_l2059_205980

/-- The cost difference between flight and ground school portions of a private pilot course -/
theorem pilot_course_cost_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_cost flight_cost ground_cost difference =>
    total_cost = 1275 ∧
    flight_cost = 950 ∧
    ground_cost = 325 ∧
    total_cost = flight_cost + ground_cost ∧
    flight_cost > ground_cost ∧
    difference = flight_cost - ground_cost ∧
    difference = 625

/-- The theorem holds for the given costs -/
theorem pilot_course_cost_difference_holds :
  ∃ (total_cost flight_cost ground_cost difference : ℕ),
    pilot_course_cost_difference total_cost flight_cost ground_cost difference :=
by
  sorry

end NUMINAMATH_CALUDE_pilot_course_cost_difference_pilot_course_cost_difference_holds_l2059_205980


namespace NUMINAMATH_CALUDE_average_milk_per_container_l2059_205910

-- Define the number of containers and their respective capacities
def containers_1_5 : ℕ := 6
def containers_0_67 : ℕ := 4
def containers_0_875 : ℕ := 5
def containers_2_33 : ℕ := 3
def containers_1_25 : ℕ := 2

def capacity_1_5 : ℚ := 3/2
def capacity_0_67 : ℚ := 67/100
def capacity_0_875 : ℚ := 875/1000
def capacity_2_33 : ℚ := 233/100
def capacity_1_25 : ℚ := 5/4

-- Define the total number of containers
def total_containers : ℕ := containers_1_5 + containers_0_67 + containers_0_875 + containers_2_33 + containers_1_25

-- Define the total amount of milk sold
def total_milk : ℚ := containers_1_5 * capacity_1_5 + containers_0_67 * capacity_0_67 + 
                      containers_0_875 * capacity_0_875 + containers_2_33 * capacity_2_33 + 
                      containers_1_25 * capacity_1_25

-- Theorem: The average amount of milk sold per container is 1.27725 liters
theorem average_milk_per_container : total_milk / total_containers = 127725 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_average_milk_per_container_l2059_205910
